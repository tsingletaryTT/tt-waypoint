# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Visual (not just numeric) sanity check for the generation loop, using a REAL,
in-distribution seed image (`ref_activations/seed_frame_ref.png`, a real photo saved
during the Stage 0 reference run) instead of the synthetic random-static images every
other capture/test in this repo uses for cheap reproducibility.

This exists because `test_generation_loop.py`'s pixel-correlation-vs-reference metric
looked catastrophic (0.05-0.24) under a random-static seed, and that metric is genuinely
the wrong bar for this loop (see PORT_PLAN.md's Stage 6 section: an iterative,
self-referential process's own trajectory necessarily diverges from any other specific
trajectory once even one step differs, however slightly). The real question -- does this
produce something reasonable in actual use -- can only be answered by looking at it.
Saves PNGs to /tmp for inspection; does not compare against a captured reference at all.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "real-image visual check"
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from PIL import Image

from waypoint_ttnn.tt.full_model import WaypointWorldModel
from waypoint_ttnn.tt.vae_decoder import WaypointVAEDecoder
from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder
from waypoint_ttnn.tt.generation_loop import WaypointGenerator


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def main():
    import ttnn

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

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        world_model = WaypointWorldModel.from_state_dict(tf_state_dict, hf_config=hf_config, mesh_device=device)
        vae_encoder = WaypointVAEEncoder.from_state_dict(vae_state_dict, mesh_device=device)
        vae_decoder = WaypointVAEDecoder.from_state_dict(vae_state_dict, mesh_device=device)
        gen = WaypointGenerator(world_model, vae_encoder, vae_decoder, hf_config, device)

        mouse = torch.zeros(1, 1, 2)
        button = torch.zeros(1, 1, 256)
        scroll = torch.zeros(1, 1, 1)
        patch_size = 2

        vae_scale_factor = 16
        pixel_h = gen.latent_h * vae_scale_factor
        pixel_w = gen.latent_w * vae_scale_factor
        print(f"[real] target pixel size {pixel_h}x{pixel_w}")

        img = Image.open("/home/ttuser/code/tt-waypoint/ref_activations/seed_frame_ref.png").convert("RGB")
        img = img.resize((pixel_w, pixel_h))
        img_t = torch.from_numpy(np.array(img))  # [H,W,3] uint8
        img.save("/tmp/real_seed_resized.png")

        t_down = vae_encoder.t_downscale
        rgb = img_t.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255).permute(0, 3, 1, 2)
        patchified = F.pixel_unshuffle(rgb, patch_size)
        seed_frames_nhwc = [
            ttnn.from_torch(
                patchified[t : t + 1].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
                device=device, layout=ttnn.TILE_LAYOUT,
            )
            for t in range(t_down)
        ]

        torch.manual_seed(42)
        seed_latent = gen.seed(seed_frames_nhwc, mouse, button, scroll)
        print("[real] seeded. frame_timestamp now", gen.frame_timestamp)

        def save_latent(latent, name):
            frames = gen.decode_frame(latent)
            torch_frame = ttnn.to_torch(frames[0]).float().permute(0, 3, 1, 2)
            out = F.pixel_shuffle(torch_frame, patch_size).clamp(0, 1)
            out_uint8 = (out[0] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(out_uint8).save(f"/tmp/{name}.png")
            print(f"[real] saved {name}.png")

        save_latent(seed_latent, "real_seed_decoded")

        for i in range(2):
            latent = gen.step(mouse, button, scroll)
            save_latent(latent, f"real_gen_frame_{i+1}")
            print(f"[real] frame {i+1}: latent std/mean {latent.std().item():.4f}/{latent.mean().item():.4f}")
    finally:
        ttnn.close_mesh_device(device)
    print("[real] DONE")


if __name__ == "__main__":
    main()
