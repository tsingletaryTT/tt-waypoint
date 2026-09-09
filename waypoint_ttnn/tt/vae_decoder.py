# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""TTNN port of Overworld/Waypoint-1.5-1B's VAE decoder (`ChunkedStreamingTAEHV`, a
TAEHV-family tiny autoencoder -- see ae_model.py in the HF repo's trust_remote_code
module). This is the path that actually turns generated latents into the RGB pixels a
viewer sees, so unlike the KV-cache host bookkeeping in functional_decoder.py, the conv
layers here run as real `ttnn.conv2d`/`ttnn.upsample` ops on device (Taylor's explicit
call -- see PORT_PLAN.md Stage 5).

ARCHITECTURE (reference `self.decoder`, an `nn.Sequential` -- indices below match the
real checkpoint's `decoder.<i>.*` state-dict keys exactly, confirmed against the actual
downloaded safetensors, not assumed from reading the class body alone):

  0  Clamp                      tanh(x/3)*3, applied to the raw latent
  1  Conv3x3(32->256, bias)
  2  ReLU
  3-5  MemBlock(256,256) x3
  6  Upsample(x2, nearest)
  7  TGrow(256, stride=1)       1x1 conv 256->256, no time-expansion (identity split)
  8  Conv3x3(256->128, no bias)
  9-11  MemBlock(128,128) x3
  12  Upsample(x2, nearest)
  13  TGrow(128, stride=2)      1x1 conv 128->256, split into 2 frames of 128ch
  14  Conv3x3(128->64, no bias)
  15-17  MemBlock(64,64) x3
  18  Upsample(x2, nearest)
  19  TGrow(64, stride=2)       1x1 conv 64->128, split into 2 frames of 64ch
  20  Conv3x3(64->64, no bias)
  21  ReLU
  22  Conv3x3(64->12, bias)     12 = image_channels(3) * patch_size(2)**2

Followed by `_postprocess_output_frames`: pixel_shuffle(patch_size=2) -> clamp(0,1).

STREAMING STATE MACHINE: the reference processes ONE frame at a time through a work
QUEUE (`_sequential_single_step` in ae_model.py), not a simple linear pass -- a `MemBlock`
needs the PREVIOUS frame's value at that exact layer (zeros on the first call, matching
`memory[i] is None -> xt*0`), and a `TGrow` layer turns one input frame into `stride`
output frames, each pushed back onto the queue for the remaining layers. This is ported
verbatim (same queue/memory bookkeeping, `WaypointVAEDecoder` in place of `nn.Sequential`
child indexing) because the actual per-call behavior (how many output frames a given
`decode()` call produces, and the `frames_to_trim` priming behavior on the very first
call) depends on it -- a naive "run the whole stack once" port would not match the real
streaming contract the pipeline actually depends on.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

import torch

from models.common.lightweightmodule import LightweightModule

TWorkItem = namedtuple("TWorkItem", ("tensor", "layer_index"))


def _conv_config():
    import ttnn

    return ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16)


def _compute_config(device):
    import ttnn

    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
    )


class TTConv2d:
    """3x3 (or 1x1) conv wrapper: caches ttnn-prepared weights/bias across calls, exactly
    the caching idiom used by tt-metal's own SDXL VAE decoder
    (models/demos/vision/generative/stable_diffusion/wormhole/tt/vae/ttnn_conv_block.py)."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], device, in_channels, out_channels,
                 kernel_size=3, padding=1):
        import ttnn

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv_config = _conv_config()
        self.compute_config = _compute_config(device)
        self.weight = ttnn.from_torch(weight.float())
        self.bias = ttnn.from_torch(bias.float().view(1, 1, 1, -1)) if bias is not None else None

    def __call__(self, x, height: int, width: int):
        import ttnn

        kwargs = dict(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=1,
            input_height=height,
            input_width=width,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(1, 1),
            padding=(self.padding, self.padding),
            dilation=(1, 1),
            groups=1,
            device=self.device,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            dtype=ttnn.bfloat16,
            return_weights_and_bias=True,
        )
        if self.bias is not None:
            out, [self.weight, self.bias] = ttnn.conv2d(
                input_tensor=x, weight_tensor=self.weight, bias_tensor=self.bias, **kwargs
            )
        else:
            out, [self.weight, _] = ttnn.conv2d(input_tensor=x, weight_tensor=self.weight, **kwargs)
        # conv2d returns a flattened [1, 1, N*H*W, C] tensor -- reshape back to proper
        # NHWC so every downstream op (concat, upsample, add, channel-slicing) sees
        # correct shape/stride info, matching the ttnn_vae_resnet.py/ttnn_vae_upsample.py
        # precedent's own explicit reshapes after each conv/before each non-conv op.
        return ttnn.reshape(out, [1, height, width, self.out_channels])


class MemBlockTT:
    kind = "memblock"

    def __init__(self, w: dict, prefix: str, device, channels: int):
        self.channels = channels
        self.conv0 = TTConv2d(w[f"{prefix}.conv.0.weight"], w[f"{prefix}.conv.0.bias"], device, channels * 2, channels)
        self.conv1 = TTConv2d(w[f"{prefix}.conv.2.weight"], w[f"{prefix}.conv.2.bias"], device, channels, channels)
        self.conv2 = TTConv2d(w[f"{prefix}.conv.4.weight"], w[f"{prefix}.conv.4.bias"], device, channels, channels)

    def __call__(self, x, past, height: int, width: int):
        import ttnn

        h = ttnn.concat([x, past], dim=-1)
        h = ttnn.relu(self.conv0(h, height, width))
        h = ttnn.relu(self.conv1(h, height, width))
        h = self.conv2(h, height, width)
        # in_channels == out_channels for every MemBlock in this checkpoint -> skip is
        # nn.Identity() in the reference (confirmed: no `decoder.<i>.skip.*` key exists
        # for any MemBlock index in the real state dict).
        return ttnn.relu(ttnn.add(h, x))


class TGrowTT:
    kind = "tgrow"

    def __init__(self, w: dict, prefix: str, device, channels: int, stride: int):
        self.stride = stride
        self.channels = channels
        self.conv = TTConv2d(w[f"{prefix}.conv.weight"], None, device, channels, channels * stride,
                              kernel_size=1, padding=0)

    def __call__(self, x, height: int, width: int):
        """Returns a list of `stride` single-frame tensors, earliest first (matches the
        reference's channel-major split + `reversed(...)` push order in
        `_sequential_single_step`)."""
        import ttnn

        y = self.conv(x, height, width)
        if self.stride == 1:
            return [y]
        # y: [1, H, W, channels*stride] (NHWC) -> split along the LAST (channel) axis
        # into `stride` chunks of `channels` each; chunk 0 is the earlier frame.
        chunks = []
        for i in range(self.stride):
            chunks.append(y[:, :, :, i * self.channels : (i + 1) * self.channels])
        return chunks


class OtherTT:
    """Wraps a plain op (conv, ReLU, Upsample, Clamp) that the queue machinery treats
    generically -- matches `_sequential_single_step`'s `else: xt = b(xt)` branch."""

    kind = "other"

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, height: int, width: int):
        return self.fn(x, height, width)


@dataclass
class _DecoderState:
    memory: list = field(default_factory=list)
    work_queue: list = field(default_factory=list)
    n_frames_decoded: int = 0


class WaypointVAEDecoder(LightweightModule):
    """Streaming TAEHV decoder: latents -> RGB frames, real ttnn.conv2d/ttnn.upsample
    compute on device. See module docstring for the exact layer layout and the
    queue-based streaming contract this replicates."""

    def __init__(self, layers: list, mesh_device, patch_size: int, frames_to_trim: int):
        self.layers = layers
        self.mesh_device = mesh_device
        self.patch_size = patch_size
        self.frames_to_trim = frames_to_trim
        self.state = _DecoderState(memory=[None] * len(layers))

    @classmethod
    def from_state_dict(cls, state_dict, *, mesh_device, patch_size=2, image_channels=3,
                         latent_channels=32, n_f=(256, 128, 64, 64),
                         decoder_time_upscale=(False, True, True)):
        import ttnn

        w = {k[len("decoder."):]: v.float() for k, v in state_dict.items() if k.startswith("decoder.")}

        def conv(prefix, in_ch, out_ch, has_bias=True, k=3, p=1):
            bias = w.get(f"{prefix}.bias") if has_bias else None
            return OtherTT(lambda x, h, wd, _c=TTConv2d(w[f"{prefix}.weight"], bias, mesh_device, in_ch, out_ch, k, p):
                            _c(x, h, wd))

        def upsample(scale=2):
            # ttnn.upsample requires an un-padded (ROW_MAJOR) input -- a TILE tensor whose
            # H*W isn't tile-aligned carries internal tile padding that trips
            # "Tiled input must be tile-aligned". Convert down, upsample, convert back up
            # (matches ttnn_vae_upsample.py's own post-upsample ttnn.to_layout(TILE)).
            def _up(x, h, wd):
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = ttnn.upsample(x, scale)
                return ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            return OtherTT(_up)

        def relu():
            return OtherTT(lambda x, h, wd: ttnn.relu(x))

        def clamp3():
            return OtherTT(lambda x, h, wd: ttnn.multiply(ttnn.tanh(ttnn.multiply(x, 1.0 / 3.0)), 3.0))

        strides = [2 if t else 1 for t in decoder_time_upscale]
        layers = [
            clamp3(),
            conv("1", latent_channels, n_f[0]),
            relu(),
            MemBlockTT(w, "3", mesh_device, n_f[0]),
            MemBlockTT(w, "4", mesh_device, n_f[0]),
            MemBlockTT(w, "5", mesh_device, n_f[0]),
            upsample(),
            TGrowTT(w, "7", mesh_device, n_f[0], strides[0]),
            conv("8", n_f[0], n_f[1], has_bias=False),
            MemBlockTT(w, "9", mesh_device, n_f[1]),
            MemBlockTT(w, "10", mesh_device, n_f[1]),
            MemBlockTT(w, "11", mesh_device, n_f[1]),
            upsample(),
            TGrowTT(w, "13", mesh_device, n_f[1], strides[1]),
            conv("14", n_f[1], n_f[2], has_bias=False),
            MemBlockTT(w, "15", mesh_device, n_f[2]),
            MemBlockTT(w, "16", mesh_device, n_f[2]),
            MemBlockTT(w, "17", mesh_device, n_f[2]),
            upsample(),
            TGrowTT(w, "19", mesh_device, n_f[2], strides[2]),
            conv("20", n_f[2], n_f[3], has_bias=False),
            relu(),
            conv("22", n_f[3], image_channels * patch_size ** 2),
        ]
        t_upscale = 1
        for s in strides:
            t_upscale *= s
        return cls(layers, mesh_device, patch_size, frames_to_trim=t_upscale - 1)

    # ---- streaming queue machinery (ports ae_model.py's `_sequential_single_step`) ----

    def _step(self, height: int, width: int):
        """Ports ae_model.py's `_sequential_single_step` exactly: drains the work queue
        until ONE fully-processed frame emerges, or returns None if the queue runs dry
        first. Does NOT touch the trim counter -- that belongs one layer up, in
        `_streaming_decode_step`, matching the reference's own layering. `height`/`width`
        are the base (latent) spatial size; the size feeding each layer is derived from it
        since our layers are plain callables, not shape-introspectable nn.Modules."""
        import ttnn

        q = self.state.work_queue
        mem = self.state.memory
        while q:
            xt, i = q.pop(0)
            cur_h, cur_w = self._hw_at(i, height, width)
            if i == len(self.layers):
                return xt
            layer = self.layers[i]
            if layer.kind == "memblock":
                past = mem[i]
                if past is None:
                    past = ttnn.multiply(xt, 0.0)
                xt_new = layer(xt, past, cur_h, cur_w)
                mem[i] = xt
                q.insert(0, TWorkItem(xt_new, i + 1))
            elif layer.kind == "tgrow":
                frames = layer(xt, cur_h, cur_w)
                for f in reversed(frames):
                    q.insert(0, TWorkItem(f, i + 1))
            else:
                xt_new = layer(xt, cur_h, cur_w)
                q.insert(0, TWorkItem(xt_new, i + 1))
        return None

    def _hw_at(self, layer_idx: int, base_h: int, base_w: int):
        """Spatial size feeding layer `layer_idx`, given the decoder's fixed 3x
        Upsample(x2) at indices 6, 12, 18."""
        doublings = sum(1 for up_idx in (6, 12, 18) if layer_idx > up_idx)
        return base_h * (2 ** doublings), base_w * (2 ** doublings)

    def reset(self):
        self.state = _DecoderState(memory=[None] * len(self.layers))

    def _streaming_decode_step(self, latent_nhwc, base_height: int, base_width: int):
        """Ports ae_model.py's `_streaming_decode_step` exactly: optionally pushes a new
        latent onto the queue, then loops the single-step machinery, silently discarding
        frames while the SESSION-WIDE (not per-call) counter is still <= frames_to_trim --
        this is what actually implements the priming behavior, not the outer loop in
        `decode` by itself. `latent_nhwc=None` just tries to drain an already-queued frame
        (used by `_flush_decoder`)."""
        if latent_nhwc is not None:
            self.state.work_queue.append(TWorkItem(latent_nhwc, 0))
        while True:
            xt = self._step(base_height, base_width)
            if xt is None:
                return None
            self.state.n_frames_decoded += 1
            if self.state.n_frames_decoded <= self.frames_to_trim:
                continue
            return xt

    def _flush_decoder(self, base_height: int, base_width: int):
        frames = []
        while (f := self._streaming_decode_step(None, base_height, base_width)) is not None:
            frames.append(f)
        return frames

    def decode(self, latent_nhwc, base_height: int, base_width: int):
        """latent_nhwc: ttnn tensor [1, base_height, base_width, latent_channels] (NHWC).
        Matches ae_model.py's `decode()` exactly: primes the pipeline with
        `frames_to_trim` extra internal (push, drain) rounds using this SAME first latent
        on the very first call in a session, so every call -- including the first --
        returns a real, full chunk of frames (usually `t_upscale`, 4 for this checkpoint).
        Returns a list of ttnn tensors, each [1, H, W, 12] (NHWC, still patchified)."""
        if self.state.n_frames_decoded == 0:
            for _ in range(self.frames_to_trim):
                self._streaming_decode_step(latent_nhwc, base_height, base_width)
                self._flush_decoder(base_height, base_width)
        first = self._streaming_decode_step(latent_nhwc, base_height, base_width)
        assert first is not None, "expected a decoded frame after a real (non-priming) push"
        return [first, *self._flush_decoder(base_height, base_width)]
