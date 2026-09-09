# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Re-checks the frozen-step bias finding using the REAL, PROPERLY-EVOLVED x from an
actual denoising trajectory (capture_generation_loop.py's step_trace) instead of
test_sigma_sweep.py's fixed-x-varying-sigma diagnostic. That earlier test fed the SAME
random noise x at every sigma value, including sigma=0.3 -- but in a real trajectory, x
at sigma=0.3 is a PARTIALLY DENOISED signal from 2 prior steps, not raw noise. Feeding
noise labeled "sigma=0.3" is out-of-distribution for the model and may have caused the
observed activation explosion in the REFERENCE ITSELF (not a bug in our port at all).
This test isolates that confound by using each step's real x_in exactly as the reference
model actually saw it.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "step trace verification"
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
from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder
from waypoint_ttnn.tt.rope import compute_rope_angles


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


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

    seed_image = ref["seed_image"]
    ts_mult = ref["ts_mult"]
    step_trace = ref["step_trace"]  # list of dicts: frame, step, sigma, dsigma, x_in, v_out

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        world_model = WaypointWorldModel.from_state_dict(tf_state_dict, hf_config=hf_config, mesh_device=device)
        vae_encoder = WaypointVAEEncoder.from_state_dict(vae_state_dict, mesh_device=device)

        mouse = torch.zeros(1, 1, 2)
        button = torch.zeros(1, 1, 256)
        scroll = torch.zeros(1, 1, 1)
        patch_size = 2

        t_down = vae_encoder.t_downscale
        seed_rgb = seed_image.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255)
        seed_rgb = seed_rgb.permute(0, 3, 1, 2)
        seed_patchified = F.pixel_unshuffle(seed_rgb, patch_size)
        seed_frames_nhwc = [
            ttnn.from_torch(
                seed_patchified[t : t + 1].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
                device=device, layout=ttnn.TILE_LAYOUT,
            )
            for t in range(t_down)
        ]
        enc_h = hf_config.height * hf_config.patch[0] * vae_encoder.spatial_downscale
        enc_w = hf_config.width * hf_config.patch[1] * vae_encoder.spatial_downscale
        seed_latent_tt = vae_encoder.encode(seed_frames_nhwc, enc_h, enc_w)
        seed_latent = ttnn.to_torch(seed_latent_tt).float().permute(0, 3, 1, 2).unsqueeze(1)

        rope0 = compute_rope_angles(hf_config, 0, ts_mult)
        zero_sigma = torch.zeros(1, 1)
        world_model.forward(seed_latent, zero_sigma, rope0, mouse, button, scroll, frame_idx=0, is_frozen=False)

        # Only frame 0 (the seed) is committed above; step_trace's "frame" field is 0 and
        # 1 for the TWO generated frames (frame_idx 1 and 2 in the real cache timeline).
        # frame_latents[i] is the REAL final committed latent for cache frame_idx i --
        # use it (not our own possibly-drifted output) to commit between frames, so
        # cache state stays aligned with the reference at every step.
        frame_latents_ref = ref["frame_latents"].float()
        prev_frame = -1
        for entry in step_trace:
            frame_idx = entry["frame"] + 1  # step_trace frame 0/1 -> cache frame_idx 1/2
            if entry["frame"] != prev_frame:
                if prev_frame >= 0:
                    commit_rope = compute_rope_angles(hf_config, prev_frame + 1, ts_mult)
                    world_model.forward(frame_latents_ref[prev_frame + 1], zero_sigma, commit_rope,
                                        mouse, button, scroll, frame_idx=prev_frame + 1, is_frozen=False)
                prev_frame = entry["frame"]

            step_i = entry["step"]
            sigma_val = entry["sigma"]
            x_in = entry["x_in"]
            v_ref = entry["v_out"]

            rope = compute_rope_angles(hf_config, frame_idx, ts_mult)
            sigma_t = torch.full((1, 1), sigma_val)
            v_computed = world_model.forward(x_in, sigma_t, rope, mouse, button, scroll,
                                              frame_idx=frame_idx, is_frozen=True)
            diff = (v_computed - v_ref).abs()
            print(f"[test] frame{entry['frame']} step{step_i} sigma={sigma_val:.4f}: "
                  f"max {diff.max().item():.4f} mean {diff.mean().item():.5f} "
                  f"corr {pearson_corr(v_computed, v_ref):.6f}  "
                  f"(computed std/mean {v_computed.std().item():.4f}/{v_computed.mean().item():.4f}, "
                  f"ref std/mean {v_ref.std().item():.4f}/{v_ref.mean().item():.4f})")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
