# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Verifies compute_rope_angles bit-for-bit against real captured reference rope
tensors (frame 0 and frame 1) before trusting it for any other frame index. Pure
PyTorch -- no hardware needed."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from waypoint_ttnn.tt.rope import compute_rope_angles, compute_ts_mult


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def main():
    ref = torch.load(
        "/home/ttuser/code/tt-waypoint/ref_activations/two_frames.pt", weights_only=False
    )
    cfg_dict = torch.load(
        "/home/ttuser/code/tt-waypoint/ref_activations/transformer_config.pt", weights_only=False
    )
    hf_config = _Cfg(cfg_dict)

    ts_mult = compute_ts_mult(hf_config)
    print(f"[test] ts_mult = {ts_mult}")

    for frame_idx, key in ((0, "rope_angles_frame0"), (1, "rope_angles_frame1")):
        ref_cos, ref_sin = (t.float() for t in ref[key])
        cos, sin = compute_rope_angles(hf_config, frame_idx, ts_mult=ts_mult)
        dcos = (cos - ref_cos).abs()
        dsin = (sin - ref_sin).abs()
        print(f"[test] frame {frame_idx}: cos max diff {dcos.max().item():.8f}, "
              f"sin max diff {dsin.max().item():.8f}, shapes {cos.shape} vs {ref_cos.shape}")
        assert dcos.max().item() < 1e-5, f"cos mismatch at frame {frame_idx}"
        assert dsin.max().item() < 1e-5, f"sin mismatch at frame {frame_idx}"
    print("[test] DONE -- compute_rope_angles matches the reference bit-for-bit (up to fp32 rounding)")


if __name__ == "__main__":
    main()
