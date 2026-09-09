# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Isolates the ONE new code path test_generation_loop.py's multi-step denoising loop
exercises that no prior test covers: a single is_frozen=True forward call at frame_idx=1
(a frozen "mid-denoise" step), immediately after a real seed commit. Every prior
multi-frame test (test_two_frame_decode.py) only ever used is_frozen=False. This isolates
"is a single frozen call correct" from "does 4-step rectified-flow accumulation amplify
already-expected per-call noise" -- test_generation_loop.py showed a much bigger
degradation for generated frames than for the seeded frame, and this test exists to find
out which of those two explanations is right before accepting either.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "frozen-step isolation check"
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

    ref = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/frozen_step.pt", weights_only=False)
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
    x_input = ref["x_input"].float()
    sigma_value = ref["sigma_value"].item()
    v_ref = ref["v_output"].float()
    ts_mult = ref["ts_mult"]

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        world_model = WaypointWorldModel.from_state_dict(tf_state_dict, hf_config=hf_config, mesh_device=device)
        vae_encoder = WaypointVAEEncoder.from_state_dict(vae_state_dict, mesh_device=device)

        mouse = torch.zeros(1, 1, 2)
        button = torch.zeros(1, 1, 256)
        scroll = torch.zeros(1, 1, 1)
        patch_size = 2

        # --- Seed commit at frame_idx=0, matching capture_frozen_step.py exactly ---
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

        # --- The isolated frozen call ---
        rope1 = compute_rope_angles(hf_config, 1, ts_mult)
        sigma_t = torch.full((1, 1), sigma_value)
        v_computed = world_model.forward(x_input, sigma_t, rope1, mouse, button, scroll, frame_idx=1, is_frozen=True)

        diff = (v_computed - v_ref).abs()
        print(f"[test] frozen step v vs reference: max {diff.max().item():.4f} "
              f"mean {diff.mean().item():.5f} corr {pearson_corr(v_computed, v_ref):.6f}")
        print(f"[test] computed std/mean {v_computed.std().item():.4f}/{v_computed.mean().item():.4f}  "
              f"ref std/mean {v_ref.std().item():.4f}/{v_ref.mean().item():.4f}")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
