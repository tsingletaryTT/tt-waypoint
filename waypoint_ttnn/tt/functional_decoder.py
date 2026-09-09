# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Functional TTNN decoder for Overworld/Waypoint-1.5-1B ("WorldModel").

WHY THIS LIVES IN ~/code/tt-waypoint, NOT tt-metal's models/autoports/ TREE
------------------------------------------------------------------------------
The `ttm-functional-decoder` skill's own contract names
`models/autoports/<model>/tt/functional_decoder.py` inside a tt-metal checkout as the
expected home, and this file follows its conventions exactly (LightweightModule base
class, a real `from_state_dict` weight-loading boundary, `prefill_forward`/
`decode_forward`, PCC-based verification against the HF reference). But this model isn't
being upstreamed into tt-metal -- it's a standalone bring-up, packaged and published the
same way tt-skyreels/tt-animatediff are: a dedicated repo, referenced from
tt-model-manager's `extra_code` once packaging starts. So the FILE lives here instead,
importing `models.common.lightweightmodule` etc. from tt-metal via `sys.path` (see the
test files for the exact pattern) the same way `skyreels_ttnn/pipeline_skyreels.py`
imports `models.tt_dit.*`. Bring-up methodology and rigor from the skill; repo location
from the packaging convention. See CLAUDE.md's "Bring-up vs. packaging" section.

WHY THIS DOES NOT LOOK LIKE A STANDARD tt_transformers DECODER
----------------------------------------------------------------
This is NOT a causal language model. There are no tokens, no vocabulary, no
embedding/lm_head. It is an autoregressive DIFFUSION transformer: each call denoises one
FULL FRAME (512 spatial tokens, attended to bidirectionally within the frame) over a
small fixed number of sigma steps, conditioned on a per-layer ring-buffer cache of
PREVIOUS FRAMES rather than a per-token KV cache. Consequently:

* "prefill" here means "the cache is empty" (frame 0) -- not "a long token prompt".
* "decode" here means "the cache already holds history" (frame_idx > 0) -- not "one new
  token". Both prefill_forward and decode_forward process a full 512-token frame; the
  distinction is entirely about cache state, not about how many new tokens arrive.
* The cache is NOT paged in the tt_transformers/vLLM sense. `ttnn.experimental.
  paged_fill_cache` / `paged_update_cache` assume per-TOKEN page slots; this model's
  cache writes and reads in whole-FRAME (512-token) blocks, and a real hardware
  correctness check (see doc/functional_decoder.md) found the reference computes DENSE
  attention over the entire zero-padded per-layer capacity buffer -- there is no
  block-sparse mask or gather to replicate, just the buffer's write/read layout. A
  custom `WaypointFrameCache` below replicates that layout instead of using the paged
  cache ops, which do not fit this model's actual contract.
* No cross-attention exists in this checkpoint: `prompt_conditioning` is `None` in the
  real downloaded config (`Overworld/Waypoint-1.5-1B`'s transformer/config.json),
  confirmed by reading the value directly, not assumed from the model family. Text
  conditioning is therefore NOT implemented here (there is nothing to implement -- the
  checkpoint's own weights have no cross-attention parameters). `prompt_emb`/
  `prompt_pad_mask` are accepted for calling-convention parity with the pipeline but are
  unused, matching the reference.

Weight-loading boundary: `from_state_dict` takes the real Overworld/Waypoint-1.5-1B
`transformer/diffusion_pytorch_model.safetensors` state dict directly (HF key names,
`transformer.blocks.<i>....`), converts to ttnn tensors once, and this module's
prefill_forward/decode_forward never touch torch/host conversion again except at the
cache boundary noted above (a known, documented gap against this skill's "no host
fallback in the runtime path" bar -- see doc/functional_decoder.md's evidence table for
exactly what is and is not yet proven).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from models.common.lightweightmodule import LightweightModule


@dataclass
class WorldModelLayerConfig:
    """Per-layer geometry derived once from the HF config -- avoids re-deriving
    local/global/dilation facts inside every forward call."""

    layer_idx: int
    is_global: bool
    dilation: int
    capacity: int  # ring (L) + tail (tpf)
    l: int  # ring size (frames * tpf)
    num_buckets: int
    has_ctrl_fusion: bool


def derive_layer_config(hf_config, layer_idx: int) -> WorldModelLayerConfig:
    tpf = hf_config.height * hf_config.width  # tokens_per_frame, e.g. 16*32=512
    period = hf_config.global_attn_period
    offset = getattr(hf_config, "global_attn_offset", 0) % period
    is_global = (layer_idx - offset) % period == 0
    window = hf_config.global_window if is_global else hf_config.local_window
    dilation = hf_config.global_pinned_dilation if is_global else 1

    l = window * tpf
    num_buckets = (l // tpf) // dilation
    capacity = l + tpf

    ctrl_period = hf_config.ctrl_conditioning_period
    has_ctrl_fusion = ctrl_period is not None and layer_idx % ctrl_period == 0

    return WorldModelLayerConfig(
        layer_idx=layer_idx,
        is_global=is_global,
        dilation=dilation,
        capacity=capacity,
        l=l,
        num_buckets=num_buckets,
        has_ctrl_fusion=has_ctrl_fusion,
    )


class WaypointFrameCache:
    """Per-layer ring-buffer cache, one instance per layer. Host-side (torch) buffer --
    see the module docstring for why: writes are whole-512-token-block, not per-token,
    and this is a first CORRECTNESS pass, not yet the optimization pass that would move
    this on-device. `capacity` = ring (L) + tail; the tail always holds the current
    frame's own K/V; the ring persists history per the reference's exact indexing.
    """

    def __init__(self, layer_cfg: WorldModelLayerConfig, n_kv_heads: int, d_head: int, dtype=torch.bfloat16):
        self.cfg = layer_cfg
        self.kv = torch.zeros(2, 1, n_kv_heads, layer_cfg.capacity, d_head, dtype=dtype)

    def reset(self):
        self.kv.zero_()

    def upsert(self, k: torch.Tensor, v: torch.Tensor, frame_idx: int, is_frozen: bool):
        """k, v: [1, n_kv_heads, tpf, d_head] for the current frame (already RMSNorm'd +
        RoPE'd, matching Attn.forward's call order in the reference).

        Mirrors modular_blocks.py's LayerKVCache.upsert exactly: always write the tail
        (current frame's self-attention component); on a write_step (frame_idx %
        dilation == 0) AND when not frozen (i.e. this is a real, committed step -- not a
        speculative mid-denoise step), also persist into the ring at the bucket/slot the
        reference computes.
        """
        cfg = self.cfg
        tpf = k.shape[2]
        tail_start = cfg.l
        self.kv[0, :, :, tail_start:, :] = k
        self.kv[1, :, :, tail_start:, :] = v

        write_step = (frame_idx % cfg.dilation) == 0
        if write_step and not is_frozen:
            bucket = (frame_idx + cfg.dilation - 1) // cfg.dilation
            slot = bucket % cfg.num_buckets
            base = slot * tpf
            self.kv[0, :, :, base : base + tpf, :] = k
            self.kv[1, :, :, base : base + tpf, :] = v

        return self.kv[0], self.kv[1]  # [1, n_kv_heads, capacity, d_head] each


def _rms_norm_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Weight-free RMSNorm, matching the reference's bare `F.rms_norm(x, (x.size(-1),))`
    exactly (no learned weight -- this model's blocks use plain RMSNorm, with all the
    learned scale/bias/gate coming from AdaLN conditioning instead)."""
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def _apply_ortho_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Matches OrthoRoPE.forward exactly: unfold adjacent pairs, rotate, re-concatenate
    as (all-rotated-evens, all-rotated-odds) -- NOT re-interleaved. This looks like an
    unusual permutation for a rotary embedding, but it is harmless: attention only ever
    computes a dot product between a query and key that both went through this SAME
    fixed permutation, so it is correctness-neutral as long as it is applied identically
    everywhere (verified on real hardware, see doc/functional_decoder.md)."""
    x0, x1 = x.unfold(-1, 2, 2).unbind(-1)
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    return torch.cat((y0, y1), dim=-1)


class FunctionalDecoder(LightweightModule):
    """One WorldDiTBlock (transformer block) of Waypoint-1.5-1B.

    Deliberately implemented with torch-side tensor bookkeeping around ttnn compute
    calls (matmul, silu, scaled_dot_product_attention) for the reasons in the module
    docstring. The compute-heavy ops (projections, attention, MLP) run on-device; the
    ring-buffer indexing and RMSNorm/RoPE currently run host-side. See
    doc/functional_decoder.md's evidence table for exactly what is verified vs. not.
    """

    def __init__(self, hf_config, layer_idx: int, mesh_device, weights: dict):
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.layer_cfg = derive_layer_config(hf_config, layer_idx)
        self.n_heads = hf_config.n_heads
        self.n_kv_heads = hf_config.n_kv_heads or hf_config.n_heads
        self.d_head = hf_config.d_model // hf_config.n_heads
        self.w = weights  # torch tensors, fp32, keyed by short name (see from_state_dict)
        self.cache = WaypointFrameCache(self.layer_cfg, self.n_kv_heads, self.d_head)

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        """Real weight-loading boundary. `state_dict` is the HF checkpoint's own state
        dict (e.g. loaded via `safetensors.torch.load_file` on
        `transformer/diffusion_pytorch_model.safetensors`), keyed like
        `transformer.blocks.<layer_idx>.attn.q_proj.weight`. Converts once, here --
        prefill_forward/decode_forward never touch state_dict again.
        """
        p = f"transformer.blocks.{layer_idx}."
        w = {
            "q_proj": state_dict[p + "attn.q_proj.weight"].float(),
            "k_proj": state_dict[p + "attn.k_proj.weight"].float(),
            "v_proj": state_dict[p + "attn.v_proj.weight"].float(),
            "out_proj": state_dict[p + "attn.out_proj.weight"].float(),
            "v_lamb": state_dict[p + "attn.v_lamb"].float(),
            "attn_cond_bias_in": state_dict[p + "attn_cond_head.bias_in"].float(),
            "attn_cond_proj": [state_dict[p + f"attn_cond_head.cond_proj.{i}.weight"].float() for i in range(3)],
            "mlp_cond_bias_in": state_dict[p + "mlp_cond_head.bias_in"].float(),
            "mlp_cond_proj": [state_dict[p + f"mlp_cond_head.cond_proj.{i}.weight"].float() for i in range(3)],
            "mlp_fc1": state_dict[p + "dit_mlp.fc1.weight"].float(),
            "mlp_fc2": state_dict[p + "dit_mlp.fc2.weight"].float(),
        }
        layer_cfg = derive_layer_config(hf_config, layer_idx)
        if layer_cfg.has_ctrl_fusion:
            w["ctrl_fc1_x"] = state_dict[p + "ctrl_mlpfusion.fc1_x.weight"].float()
            w["ctrl_fc1_c"] = state_dict[p + "ctrl_mlpfusion.fc1_c.weight"].float()
            w["ctrl_fc2"] = state_dict[p + "ctrl_mlpfusion.fc2.weight"].float()
        return cls(hf_config, layer_idx, mesh_device, w)

    # ---- shared block body -------------------------------------------------------

    def _cond_head(self, cond: torch.Tensor, bias_in: torch.Tensor, proj: list[torch.Tensor]):
        h = torch.nn.functional.silu(cond + bias_in)
        return tuple(h @ pw.t() for pw in proj)

    def _attn(self, x: torch.Tensor, rope_angles, frame_idx: int, is_frozen: bool, v1: Optional[torch.Tensor]):
        import ttnn

        B, T, D = x.shape
        q = (x @ self.w["q_proj"].t()).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = (x @ self.w["k_proj"].t()).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = (x @ self.w["v_proj"].t()).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Reference (Attn.forward): `v1 = v if v1 is None else v1; v = lerp(v, v1, lamb);
        # return y, v1` -- v1 is PINNED to whichever layer first produced it (layer 0,
        # where it enters as None) and threaded UNCHANGED through every later layer's
        # lerp. Returning the freshly-lerped `v` as v1_out (as before) let the
        # value-residual signal drift layer over layer instead of staying pinned to the
        # original -- invisible in a layer-0-only test (v1 enters as None there either
        # way) but a real, compounding bug from layer 1 onward.
        if v1 is None:
            v1 = v
        v = torch.lerp(v, v1.view_as(v), self.w["v_lamb"])
        v1_out = v1

        q, k = _rms_norm_torch(q), _rms_norm_torch(k)
        cos, sin = rope_angles
        q, k = _apply_ortho_rope(q, cos, sin), _apply_ortho_rope(k, cos, sin)

        k_full, v_full = self.cache.upsert(k, v, frame_idx, is_frozen)

        rep = self.n_heads // self.n_kv_heads
        k_rep = k_full.repeat_interleave(rep, dim=1)
        v_rep = v_full.repeat_interleave(rep, dim=1)

        tt_q = ttnn.from_torch(q.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_k = ttnn.from_torch(k_rep.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_v = ttnn.from_torch(v_rep.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_out = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
        attn_out = ttnn.to_torch(tt_out).float().transpose(1, 2).reshape(B, T, D)

        y = attn_out @ self.w["out_proj"].t()
        return y, v1_out

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        import ttnn

        tt_x = ttnn.from_torch(x.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_fc1 = ttnn.from_torch(self.w["mlp_fc1"].t().contiguous().to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_fc2 = ttnn.from_torch(self.w["mlp_fc2"].t().contiguous().to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        h = ttnn.matmul(tt_x, tt_fc1)
        h = ttnn.silu(h)
        out = ttnn.matmul(h, tt_fc2)
        return ttnn.to_torch(out).float()

    def _block_forward(
        self,
        x: torch.Tensor,
        rope_angles,
        cond: torch.Tensor,
        ctrl_emb: Optional[torch.Tensor],
        v1: Optional[torch.Tensor],
        frame_idx: int,
        is_frozen: bool,
    ):
        s0, b0, g0 = self._cond_head(cond, self.w["attn_cond_bias_in"], self.w["attn_cond_proj"])
        s1, b1, g1 = self._cond_head(cond, self.w["mlp_cond_bias_in"], self.w["mlp_cond_proj"])

        def ada_rmsnorm(x, s, b):
            x4 = x.view(x.shape[0], s.shape[1], -1, x.shape[-1])
            y4 = _rms_norm_torch(x4) * (1 + s.unsqueeze(2)) + b.unsqueeze(2)
            return y4.reshape(x.shape)

        def ada_gate(x, g):
            x4 = x.view(x.shape[0], g.shape[1], -1, x.shape[-1])
            return (x4 * g.unsqueeze(2)).reshape(x.shape)

        residual = x
        xn = ada_rmsnorm(x, s0, b0)
        attn_out, v1_out = self._attn(xn, rope_angles, frame_idx, is_frozen, v1)
        x = ada_gate(attn_out, g0) + residual

        if self.layer_cfg.has_ctrl_fusion and ctrl_emb is not None:
            xr = _rms_norm_torch(x)
            ctrl_r = _rms_norm_torch(ctrl_emb)
            B, L, D = xr.shape
            xr4 = xr.view(B, ctrl_r.shape[1], -1, D)
            fused = torch.nn.functional.silu(
                xr4 @ self.w["ctrl_fc1_x"].t() + (ctrl_r @ self.w["ctrl_fc1_c"].t()).unsqueeze(2)
            )
            fused = (fused @ self.w["ctrl_fc2"].t()).flatten(1, 2)
            x = fused + x

        mlp_in = ada_rmsnorm(x, s1, b1)
        mlp_out = self._mlp(mlp_in)
        x = ada_gate(mlp_out, g1) + x
        return x, v1_out

    # ---- public API ----------------------------------------------------------------

    def prefill_forward(self, x, rope_angles, cond, ctrl_emb, v1=None):
        """Frame with an EMPTY cache (frame_idx=0). See module docstring: "prefill"
        means cache-empty here, not "long token prompt"."""
        self.cache.reset()
        return self._block_forward(x, rope_angles, cond, ctrl_emb, v1, frame_idx=0, is_frozen=False)

    def decode_forward(self, x, rope_angles, cond, ctrl_emb, frame_idx: int, is_frozen: bool, v1=None):
        """Frame with EXISTING cache history (frame_idx > 0)."""
        return self._block_forward(x, rope_angles, cond, ctrl_emb, v1, frame_idx=frame_idx, is_frozen=is_frozen)
