# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Computes OrthoRoPE angles for an arbitrary frame, matching model.py's
`OrthoRoPEAngles.forward` + `WorldModel.forward`'s pos_ids construction exactly.

Needed for Stage 6: every prior test reused a SINGLE captured `rope_angles` tensor
(frame 0 or frame 1, from a real reference run) because that was sufficient to verify
`FunctionalDecoder`/`WaypointWorldModel` in isolation. A real multi-frame interactive
loop needs rope angles for arbitrary frame indices, which can't come from a fixed
capture -- so this ports the (pure, weight-free, deterministic) computation itself.
Verified bit-for-bit against the captured frame-0 and frame-1 reference tensors in
test_rope.py before being trusted for any frame index beyond those two.
"""

from __future__ import annotations

import torch


def compute_ts_mult(hf_config) -> int:
    """`base_fps // latent_fps`, matching WorldEngineSetTimestepsStep's computation."""
    base_fps = getattr(hf_config, "base_fps", 60)
    inference_fps = getattr(hf_config, "inference_fps", base_fps)
    latent_fps = inference_fps / hf_config.temporal_compression
    return int(base_fps) // int(latent_fps)


def compute_rope_angles(hf_config, frame_idx: int, ts_mult: int | None = None):
    """Returns (cos, sin), each [1, 1, tokens_per_frame, d_head // 2] -- matches the
    shape/broadcast convention every prior test already consumed via captured
    `rope_angles` tuples. `frame_idx` is the RAW (unscaled) frame counter; the RoPE
    temporal position uses `frame_idx * ts_mult` (WorldModel.forward passes
    `frame_timestamp * ts_mult` as the rope input and the raw `frame_idx` separately
    for cache indexing -- these are deliberately different values, not a typo)."""
    if ts_mult is None:
        ts_mult = compute_ts_mult(hf_config)

    height, width = hf_config.height, hf_config.width
    T = height * width
    d_head = hf_config.d_model // hf_config.n_heads
    d_xy, d_t = d_head // 8, d_head // 4

    nyq = float(getattr(hf_config, "rope_nyquist_frac", 0.8))
    max_freq = min(height, width) * nyq
    n = (d_xy + 1) // 2
    xy = (torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32) * torch.pi).repeat_interleave(2)[:d_xy]

    theta = float(getattr(hf_config, "rope_theta", 10000.0))
    inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32) / d_t))
    inv_t = inv_t.repeat_interleave(2)

    idx = torch.arange(T)
    y_pos = idx.div(width, rounding_mode="floor")
    x_pos = idx.remainder(width)
    t_scalar = float(frame_idx * ts_mult)

    x = (2.0 * x_pos.float() + 1.0) / width - 1.0
    y = (2.0 * y_pos.float() + 1.0) / height - 1.0
    t = torch.full((T,), t_scalar, dtype=torch.float32)

    freqs = torch.cat((x.unsqueeze(-1) * xy, y.unsqueeze(-1) * xy, t.unsqueeze(-1) * inv_t), dim=-1)
    cos = freqs.cos()[None, None]  # [1, 1, T, d_head/2]
    sin = freqs.sin()[None, None]
    return cos, sin
