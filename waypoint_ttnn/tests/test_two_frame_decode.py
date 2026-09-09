# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Multi-frame KV cache correctness: frame 0 (prefill_forward, empty cache) then frame 1
(decode_forward, cache now holds frame 0's history) through the SAME per-layer cache
objects, checked against the reference's own two-consecutive-frame run. This is Stage 4's
previously-unverified item per PORT_PLAN.md: only the frame-0 empty-cache case had been
proven; this is the first real test of the ring-buffer write/read logic once it actually
holds history.

Ground truth: ~/code/tt-waypoint/waypoint_ttnn/capture_two_frames.py (needs a venv with
diffusers>=0.38 for AutoModel -- the shared .tenstorrent-venv doesn't have it; the
tt-skyreels venv does).

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "two-frame decode PCC check"
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import torch
from safetensors.torch import load_file

from waypoint_ttnn.tt.full_model import WaypointWorldModel


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    import ttnn

    ref = torch.load(
        "/home/ttuser/code/tt-waypoint/ref_activations/two_frames.pt", weights_only=False
    )
    cfg_dict = torch.load(
        "/home/ttuser/code/tt-waypoint/ref_activations/transformer_config.pt", weights_only=False
    )
    hf_config = _Cfg(cfg_dict)

    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    state_dict = load_file(weights_path)

    x0 = ref["x0"].float()
    x1 = ref["x1"].float()
    sigma = torch.tensor([[0.5]], dtype=torch.float32)
    mouse = torch.zeros(1, 1, 2)
    button = torch.zeros(1, 1, 256)
    scroll = torch.zeros(1, 1, 1)
    rope0 = tuple(t.float() for t in ref["rope_angles_frame0"])
    rope1 = tuple(t.float() for t in ref["rope_angles_frame1"])

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        model = WaypointWorldModel.from_state_dict(state_dict, hf_config=hf_config, mesh_device=device)

        y0 = model.forward(x0, sigma, rope0, mouse, button, scroll, frame_idx=0, is_frozen=False)
        y_ref0 = ref["wm_output_frame0"].float()
        corr0 = pearson_corr(y0, y_ref0)
        diff0 = (y0 - y_ref0).abs()
        print(f"[test] frame0 full model vs reference: max {diff0.max().item():.4f} "
              f"mean {diff0.mean().item():.5f} corr {corr0:.6f}")

        y1 = model.forward(x1, sigma, rope1, mouse, button, scroll, frame_idx=1, is_frozen=False)
        y_ref1 = ref["wm_output_frame1"].float()
        corr1 = pearson_corr(y1, y_ref1)
        diff1 = (y1 - y_ref1).abs()
        print(f"[test] frame1 full model vs reference (cache now holds frame0 history): "
              f"max {diff1.max().item():.4f} mean {diff1.mean().item():.5f} corr {corr1:.6f}")

        print(f"[test] frame1 vs frame0 corr delta: {corr1 - corr0:.6f} "
              "(a large negative delta would flag a real multi-frame cache bug, distinct "
              "from the already-documented bf16 accumulation baseline)")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
