# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""VAE decoder correctness: feed the SAME 3 consecutive latents through our streaming
TTNN decoder (real ttnn.conv2d/ttnn.upsample compute) that the reference's own streaming
decode() was run on (same session, no reset in between -- see capture_vae.py), and check
each call's output frames against the reference's.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "VAE decoder PCC check"
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from waypoint_ttnn.tt.vae_decoder import WaypointVAEDecoder


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def postprocess(frame_nhwc_ttnn_as_torch: torch.Tensor, patch_size: int) -> torch.Tensor:
    """frame: [1, H, W, 12] NHWC float -> [H*2, W*2, 3] uint8 RGB, matching
    ae_model.py's `_postprocess_output_frames` + the final uint8 cast in `decode()`."""
    x = frame_nhwc_ttnn_as_torch.permute(0, 3, 1, 2)  # NHWC -> NCHW
    x = F.pixel_shuffle(x, patch_size)
    x = x.clamp(0, 1)
    x = (x * 255).round().to(torch.uint8)
    return x[0].permute(1, 2, 0)  # CHW -> HWC


def main():
    import ttnn

    ref = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/vae_capture.pt", weights_only=False)
    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/vae/diffusion_pytorch_model.safetensors"
    )[0]
    state_dict = load_file(weights_path)

    latents = ref["latents"].float()  # [3, 1, 32, 16, 16]
    ref_decoded = ref["decoded_frames"]  # list of 3 tensors, each [4, 256, 256, 3] uint8
    patch_size = 2
    base_h, base_w = latents.shape[-2], latents.shape[-1]

    # conv2d's halo/untilize-with-halo ops need L1_SMALL scratch space -- ttnn.open_mesh_device
    # defaults l1_small_size to 0, which conv2d cannot work with at all (matches the
    # SD VAE decoder precedent: models/demos/vision/generative/stable_diffusion/wormhole/common.py).
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        model = WaypointVAEDecoder.from_state_dict(state_dict, mesh_device=device)

        for step_idx in range(latents.shape[0]):
            latent = latents[step_idx]  # [1, 32, 16, 16]
            latent_nhwc = ttnn.from_torch(
                latent.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
                device=device, layout=ttnn.TILE_LAYOUT,
            )
            out_frames = model.decode(latent_nhwc, base_h, base_w)
            ref_frames = ref_decoded[step_idx].float()  # [4, 256, 256, 3]

            print(f"[test] step {step_idx}: got {len(out_frames)} frames, ref has {ref_frames.shape[0]}")
            assert len(out_frames) == ref_frames.shape[0], "frame count mismatch vs reference"

            for f_idx, tt_frame in enumerate(out_frames):
                torch_frame = ttnn.to_torch(tt_frame).float()
                rgb = postprocess(torch_frame, patch_size).float()
                ref_rgb = ref_frames[f_idx]
                diff = (rgb - ref_rgb).abs()
                corr = pearson_corr(rgb, ref_rgb)
                print(f"[test]   frame {f_idx}: max abs diff {diff.max().item():.2f} "
                      f"mean abs diff {diff.mean().item():.4f} corr {corr:.6f}")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
