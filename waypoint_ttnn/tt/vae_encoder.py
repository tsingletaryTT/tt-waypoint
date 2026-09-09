# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""TTNN port of Overworld/Waypoint-1.5-1B's VAE encoder (`ChunkedStreamingTAEHV.encoder`)
-- turns a `t_downscale`-frame (4, for this checkpoint) chunk of patchified RGB into one
latent. Needed once per session (to seed generation from a real starting image), unlike
the decoder which runs once per generated frame -- see PORT_PLAN.md Stage 5.

ARCHITECTURE (indices match the real checkpoint's `encoder.<i>.*` keys exactly):

  0  Conv3x3(12->64, bias)
  1  ReLU
  2  TPool(64, stride=2)        1x1 conv 128->64, fuses 2 accumulated frames into 1
  3  Conv3x3(64->64, stride=2, no bias)
  4-6  MemBlock(64,64) x3
  7  TPool(64, stride=2)
  8  Conv3x3(64->64, stride=2, no bias)
  9-11  MemBlock(64,64) x3
  12  TPool(64, stride=1)       identity fuse (encoder_time_downscale[2]=False)
  13  Conv3x3(64->64, stride=2, no bias)
  14-16  MemBlock(64,64) x3
  17  Conv3x3(64->32, bias)     -> latent_channels

Reuses `TTConv2d`/`MemBlockTT`/`OtherTT`/`TWorkItem` from vae_decoder.py (already
hardware-verified there) rather than duplicating them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from models.common.lightweightmodule import LightweightModule

from waypoint_ttnn.tt.vae_decoder import MemBlockTT, OtherTT, TTConv2d, TWorkItem


class TPoolTT:
    """Accumulates `stride` consecutive frames (channel-concat), then fuses them into one
    via a 1x1 conv -- ports ae_model.py's TPool exactly."""

    kind = "tpool"

    def __init__(self, w: dict, prefix: str, device, channels: int, stride: int):
        self.stride = stride
        self.channels = channels
        self.conv = TTConv2d(w[f"{prefix}.conv.weight"], None, device, channels * stride, channels,
                              kernel_size=1, padding=0)

    def __call__(self, frames: list, height: int, width: int):
        import ttnn

        x = frames[0] if self.stride == 1 else ttnn.concat(frames, dim=-1)
        return self.conv(x, height, width)


@dataclass
class _EncoderState:
    memory: list = field(default_factory=list)
    work_queue: list = field(default_factory=list)


class WaypointVAEEncoder(LightweightModule):
    def __init__(self, layers: list, mesh_device, patch_size: int, t_downscale: int):
        self.layers = layers
        self.mesh_device = mesh_device
        self.patch_size = patch_size
        self.t_downscale = t_downscale
        self.state = _EncoderState(memory=[None] * len(layers))

    @classmethod
    def from_state_dict(cls, state_dict, *, mesh_device, patch_size=2, image_channels=3,
                         latent_channels=32, encoder_time_downscale=(True, True, False)):
        import ttnn

        w = {k[len("encoder."):]: v.float() for k, v in state_dict.items() if k.startswith("encoder.")}
        in_ch = image_channels * patch_size ** 2

        def conv(prefix, in_c, out_c, has_bias=True, stride=1):
            bias = w.get(f"{prefix}.bias") if has_bias else None
            c = TTConv2d(w[f"{prefix}.weight"], bias, mesh_device, in_c, out_c, stride=stride)
            return OtherTT(lambda x, h, wd, _c=c: _c(x, h, wd))

        def relu():
            return OtherTT(lambda x, h, wd: ttnn.relu(x))

        strides = [2 if t else 1 for t in encoder_time_downscale]
        f = 64
        layers = [
            conv("0", in_ch, f),
            relu(),
            TPoolTT(w, "2", mesh_device, f, strides[0]),
            conv("3", f, f, has_bias=False, stride=2),
            MemBlockTT(w, "4", mesh_device, f),
            MemBlockTT(w, "5", mesh_device, f),
            MemBlockTT(w, "6", mesh_device, f),
            TPoolTT(w, "7", mesh_device, f, strides[1]),
            conv("8", f, f, has_bias=False, stride=2),
            MemBlockTT(w, "9", mesh_device, f),
            MemBlockTT(w, "10", mesh_device, f),
            MemBlockTT(w, "11", mesh_device, f),
            TPoolTT(w, "12", mesh_device, f, strides[2]),
            conv("13", f, f, has_bias=False, stride=2),
            MemBlockTT(w, "14", mesh_device, f),
            MemBlockTT(w, "15", mesh_device, f),
            MemBlockTT(w, "16", mesh_device, f),
            conv("17", f, latent_channels),
        ]
        t_downscale = 1
        for s in strides:
            t_downscale *= s
        return cls(layers, mesh_device, patch_size, t_downscale)

    def reset(self):
        self.state = _EncoderState(memory=[None] * len(self.layers))

    def _hw_at(self, layer_idx: int, base_h: int, base_w: int):
        """Spatial size FEEDING layer `layer_idx` (i.e. its input, before it runs),
        given the encoder's fixed 3x stride-2 3x3 convs at indices 3, 8, 13. Strict `>`
        -- a halving conv's own INPUT is still the pre-halved size; only layers AFTER it
        see the smaller size (matches vae_decoder.py's analogous `_hw_at`, which uses the
        same strict `>` for its doublings)."""
        halvings = sum(1 for s_idx in (3, 8, 13) if layer_idx > s_idx)
        return base_h // (2 ** halvings), base_w // (2 ** halvings)

    def _step(self, base_height: int, base_width: int):
        """Ports ae_model.py's `_sequential_single_step`, generalized with a `tpool`
        branch (the decoder doesn't use TPool, so vae_decoder.py's version omits it)."""
        q = self.state.work_queue
        mem = self.state.memory
        while q:
            xt, i = q.pop(0)
            if i == len(self.layers):
                return xt
            cur_h, cur_w = self._hw_at(i, base_height, base_width)
            layer = self.layers[i]
            if layer.kind == "memblock":
                import ttnn

                past = mem[i]
                if past is None:
                    past = ttnn.multiply(xt, 0.0)
                xt_new = layer(xt, past, cur_h, cur_w)
                mem[i] = xt
                q.insert(0, TWorkItem(xt_new, i + 1))
            elif layer.kind == "tpool":
                if mem[i] is None:
                    mem[i] = []
                mem[i].append(xt)
                if len(mem[i]) == layer.stride:
                    xt_new = layer(mem[i], cur_h, cur_w)
                    mem[i] = []
                    q.insert(0, TWorkItem(xt_new, i + 1))
            else:
                xt_new = layer(xt, cur_h, cur_w)
                q.insert(0, TWorkItem(xt_new, i + 1))
        return None

    def encode(self, frames_nhwc: list, height: int, width: int):
        """frames_nhwc: list of `t_downscale` ttnn tensors, each [1, height, width,
        in_ch] (already pixel-unshuffled -- patchify happens on the host side, see
        test_vae_encoder.py, matching `_preprocess_input_frames`). Returns one ttnn
        latent tensor [1, height, width, latent_channels]. Matches `encode()`'s own
        `if latent is None: raise RuntimeError` contract -- exactly `t_downscale`
        frames must be pushed for one call to yield exactly one latent."""
        assert len(frames_nhwc) == self.t_downscale
        for f in frames_nhwc:
            self.state.work_queue.append(TWorkItem(f, 0))
        latent = self._step(height, width)
        if latent is None:
            raise RuntimeError("expected a latent after a full chunk")
        return latent
