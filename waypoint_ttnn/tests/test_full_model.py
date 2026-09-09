# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Full 24-layer WaypointWorldModel vs. the real HF reference's whole-model output, for
the same synthetic frame-0 input used in the layer-0 check. Ground truth captured via
~/code/tt-waypoint/waypoint_ttnn/capture_synthetic_frame0.py (the `wm_output` key).

Run under a gozer lease (needs the full stack; still 1x1 mesh for this correctness pass):
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "full_model PCC check"
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
        "/home/ttuser/code/tt-waypoint/ref_activations/synthetic_frame0.pt", weights_only=False
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

    y_ref = ref["wm_output"].float()  # [1, 1, 32, 32, 64]
    rope_angles = ref["attn__inputs"][2]
    rope_angles = (rope_angles[0].float(), rope_angles[1].float())

    # Use the actually-captured x_input rather than reconstructing via manual_seed(0)+randn:
    # verified those do NOT match bit-for-bit (max diff 3.27) because the capture ran in a
    # different venv/torch build than this one -- bf16 RNG draws for a given seed are not
    # portable across torch versions/backends. The capture script already saves x_input
    # precisely so downstream tests don't need to reproduce the draw.
    B, C, H, W = 1, 32, 32, 64
    x = ref["x_input"].float()
    sigma = torch.tensor([[0.5]], dtype=torch.float32)
    mouse = torch.zeros(1, 1, 2)
    button = torch.zeros(1, 1, 256)
    scroll = torch.zeros(1, 1, 1)

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        model = WaypointWorldModel.from_state_dict(state_dict, hf_config=hf_config, mesh_device=device)
        y_computed = model.forward(
            x, sigma, rope_angles, mouse, button, scroll, frame_idx=0, is_frozen=False
        )

        diff = (y_computed - y_ref).abs()
        corr = pearson_corr(y_computed, y_ref)
        print(f"[test] full model (24 layers) vs reference: max abs diff {diff.max().item():.6f}, "
              f"mean abs diff {diff.mean().item():.6f}, correlation {corr:.6f}")
        print("[test] y_ref std/mean:", y_ref.std().item(), y_ref.mean().item())
        print("[test] y_computed std/mean:", y_computed.std().item(), y_computed.mean().item())
        assert corr > 0.99, f"correlation too low: {corr}"
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
