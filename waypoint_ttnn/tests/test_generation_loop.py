# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Full Stage 6 generation loop correctness: seed a session from the SAME real image the
reference was seeded with, generate 2 real frames using the SAME noise draws the
reference used (see generation_loop.py's `noise_override` -- injecting the exact
reference noise isolates "is the loop's math right" from "did we happen to sample
different noise"), and compare both the intermediate latents and the final decoded RGB
pixels against capture_generation_loop.py's real two-pass-per-frame reference run.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "generation loop PCC check"
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from waypoint_ttnn.tt.full_model import WaypointWorldModel
from waypoint_ttnn.tt.vae_decoder import WaypointVAEDecoder
from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder
from waypoint_ttnn.tt.generation_loop import WaypointGenerator


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def postprocess_frame(frame_nhwc_torch: torch.Tensor, patch_size: int) -> torch.Tensor:
    x = frame_nhwc_torch.permute(0, 3, 1, 2)
    x = F.pixel_shuffle(x, patch_size)
    x = x.clamp(0, 1)
    x = (x * 255).round().to(torch.uint8)
    return x[0].permute(1, 2, 0)


def main():
    import ttnn

    ref = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/generation_loop.pt", weights_only=False)
    cfg_dict = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/transformer_config.pt", weights_only=False)
    hf_config = _Cfg(cfg_dict)

    tf_weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    vae_weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/vae/diffusion_pytorch_model.safetensors"
    )[0]
    tf_state_dict = load_file(tf_weights_path)
    vae_state_dict = load_file(vae_weights_path)

    seed_image = ref["seed_image"]  # [pixel_H, pixel_W, 3] uint8
    ref_latents = ref["frame_latents"].float()  # [3, 1, 1, C, H, W] (seed, frame1, frame2)
    ref_noise = ref["noise_draws"].float()  # [2, 1, 1, C, H, W]
    ref_decoded = ref["decoded_frames"]  # list of 3 uint8 tensors

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        world_model = WaypointWorldModel.from_state_dict(tf_state_dict, hf_config=hf_config, mesh_device=device)
        vae_encoder = WaypointVAEEncoder.from_state_dict(vae_state_dict, mesh_device=device)
        vae_decoder = WaypointVAEDecoder.from_state_dict(vae_state_dict, mesh_device=device)
        gen = WaypointGenerator(world_model, vae_encoder, vae_decoder, hf_config, device)

        mouse = torch.zeros(1, 1, 2)
        button = torch.zeros(1, 1, 256)
        scroll = torch.zeros(1, 1, 1)
        patch_size = vae_decoder.patch_size

        # --- Seed ---
        t_down = vae_encoder.t_downscale
        seed_rgb = seed_image.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255)  # [T,H,W,3]
        seed_rgb = seed_rgb.permute(0, 3, 1, 2)  # [T,3,H,W]
        seed_patchified = F.pixel_unshuffle(seed_rgb, patch_size)  # [T,12,h,w]
        seed_frames_nhwc = [
            ttnn.from_torch(
                seed_patchified[t : t + 1].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
                device=device, layout=ttnn.TILE_LAYOUT,
            )
            for t in range(t_down)
        ]
        seed_latent = gen.seed(seed_frames_nhwc, mouse, button, scroll)
        diff = (seed_latent - ref_latents[0]).abs()
        print(f"[test] seed latent vs reference: max {diff.max().item():.4f} mean {diff.mean().item():.5f} "
              f"corr {pearson_corr(seed_latent, ref_latents[0]):.6f}")

        computed_latents = [seed_latent]
        for i in range(2):
            latent = gen.step(mouse, button, scroll, noise_override=ref_noise[i])
            computed_latents.append(latent)
            diff = (latent - ref_latents[i + 1]).abs()
            print(f"[test] frame {i + 1} latent vs reference: max {diff.max().item():.4f} "
                  f"mean {diff.mean().item():.5f} corr {pearson_corr(latent, ref_latents[i + 1]):.6f}")

        # --- Decode each generated latent and compare pixels ---
        for i, latent in enumerate(computed_latents):
            out_frames = gen.decode_frame(latent)
            ref_rgb_frames = ref_decoded[i].float()  # [T_up, H, W, 3]
            assert len(out_frames) == ref_rgb_frames.shape[0]
            for f_idx, tt_frame in enumerate(out_frames):
                torch_frame = ttnn.to_torch(tt_frame).float()
                rgb = postprocess_frame(torch_frame, patch_size).float()
                ref_rgb = ref_rgb_frames[f_idx]
                d = (rgb - ref_rgb).abs()
                print(f"[test] decoded frame {i}/{f_idx}: max {d.max().item():.2f} "
                      f"mean {d.mean().item():.4f} corr {pearson_corr(rgb, ref_rgb):.6f}")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
