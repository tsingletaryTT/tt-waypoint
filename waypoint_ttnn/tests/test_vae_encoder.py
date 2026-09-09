# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""VAE encoder correctness: encode the same 4-frame RGB chunk the reference's own
capture_vae.py fed to vae.encode(), and compare the resulting latent.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "VAE encoder PCC check"
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    import ttnn

    ref = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/vae_capture.pt", weights_only=False)
    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/vae/diffusion_pytorch_model.safetensors"
    )[0]
    state_dict = load_file(weights_path)

    frames_uint8 = ref["frames_uint8_in"]  # [T, H, W, 3] uint8
    latent_ref = ref["latent0"].float()  # [1, 32, 16, 16]
    patch_size = 2

    rgb = frames_uint8.unsqueeze(0).permute(0, 1, 4, 2, 3).contiguous().float().div(255)  # [1, T, 3, H, W]
    patchified = F.pixel_unshuffle(rgb, patch_size)  # [1, T, 12, H/2, W/2]
    T = patchified.shape[1]
    base_h, base_w = patchified.shape[-2], patchified.shape[-1]

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        model = WaypointVAEEncoder.from_state_dict(state_dict, mesh_device=device)
        assert model.t_downscale == T, f"t_downscale {model.t_downscale} != captured frame count {T}"

        frames_nhwc = [
            ttnn.from_torch(
                patchified[:, t].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
                device=device, layout=ttnn.TILE_LAYOUT,
            )
            for t in range(T)
        ]
        latent_tt = model.encode(frames_nhwc, base_h, base_w)
        latent_computed = ttnn.to_torch(latent_tt).float().permute(0, 3, 1, 2)  # NHWC -> NCHW

        diff = (latent_computed - latent_ref).abs()
        corr = pearson_corr(latent_computed, latent_ref)
        print(f"[test] encoder latent vs reference: shape {latent_computed.shape} "
              f"max abs diff {diff.max().item():.4f} mean abs diff {diff.mean().item():.5f} corr {corr:.6f}")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
