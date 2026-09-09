# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Full WorldModel assembly: patchify -> 24 FunctionalDecoder blocks (value-residual
threaded between them) -> AdaLN out_norm -> unpatchify, plus the noise/controller
conditioning embeddings. Mirrors ttm-full-model's mission (assemble embeddings, block
stack, final norm, and generation into a full model) adapted to this model's shape --
see functional_decoder.py's module docstring for why "prefill"/"decode" mean something
different here than in a causal LM.

Weight loading: same real-checkpoint boundary as FunctionalDecoder -- `from_state_dict`
takes the actual Overworld/Waypoint-1.5-1B transformer state dict directly.
"""
from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule

from waypoint_ttnn.tt.functional_decoder import FunctionalDecoder, _rms_norm_torch


class WaypointWorldModel(LightweightModule):
    def __init__(self, hf_config, mesh_device, weights: dict, layers: list[FunctionalDecoder]):
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.w = weights
        self.layers = layers
        self.patch = tuple(hf_config.patch)

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, mesh_device):
        w = {
            "patchify": state_dict["patchify.weight"].float(),
            "unpatchify_w": state_dict["unpatchify.weight"].float(),
            "unpatchify_b": state_dict["unpatchify.bias"].float(),
            "out_norm_fc": state_dict["out_norm.fc.weight"].float(),
            "denoise_fc1": state_dict["denoise_step_emb.mlp.fc1.weight"].float(),
            "denoise_fc2": state_dict["denoise_step_emb.mlp.fc2.weight"].float(),
            "ctrl_fc1": state_dict["ctrl_emb.mlp.fc1.weight"].float(),
            "ctrl_fc2": state_dict["ctrl_emb.mlp.fc2.weight"].float(),
        }
        layers = [
            FunctionalDecoder.from_state_dict(state_dict, hf_config=hf_config, layer_idx=i, mesh_device=mesh_device)
            for i in range(hf_config.n_layers)
        ]
        return cls(hf_config, mesh_device, w, layers)

    def _noise_conditioner(self, sigma: torch.Tensor) -> torch.Tensor:
        """Reference's NoiseConditioner.forward runs under `torch.autocast("cuda",
        enabled=False)` -- Fourier features AND the MLP itself are computed in float32,
        only the final embedding is cast back to the working dtype. Doing the MLP in
        bf16 (as every other block does) is a systematic bias, not harmless rounding
        noise: it's applied identically at every layer via AdaLN, so it compounds across
        the stack -- this is what took the 24-layer full-model correlation from 0.999
        (layer-0 alone) down to 0.94. Keep this one MLP in float32 to match."""
        import ttnn

        fourier_dim = 512
        half = fourier_dim // 2
        freq = torch.logspace(0, -1, steps=half, base=10_000.0, dtype=torch.float32)
        s = sigma.reshape(-1).float() * 1000
        phase = s[:, None] * freq[None, :]
        emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1) * (2 ** 0.5)

        tt_emb = ttnn.from_torch(emb.float(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        tt_fc1 = ttnn.from_torch(self.w["denoise_fc1"].t().contiguous().float(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        tt_fc2 = ttnn.from_torch(self.w["denoise_fc2"].t().contiguous().float(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        h = ttnn.silu(ttnn.matmul(tt_emb, tt_fc1))
        out = ttnn.matmul(h, tt_fc2)
        return ttnn.to_torch(out).float().view(sigma.shape[0], sigma.shape[1], -1)

    def _ctrl_embedding(self, mouse, button, scroll) -> torch.Tensor:
        import ttnn

        x = torch.cat((mouse.float(), button.float(), scroll.float()), dim=-1)
        tt_x = ttnn.from_torch(x.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_fc1 = ttnn.from_torch(self.w["ctrl_fc1"].t().contiguous().to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_fc2 = ttnn.from_torch(self.w["ctrl_fc2"].t().contiguous().to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        h = ttnn.silu(ttnn.matmul(tt_x, tt_fc1))
        out = ttnn.matmul(h, tt_fc2)
        return ttnn.to_torch(out).float()

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        import ttnn

        B, C, H, W = x.shape
        ph, pw = self.patch
        Hp, Wp = H // ph, W // pw
        patches = x.view(B, C, Hp, ph, Wp, pw).permute(0, 2, 4, 1, 3, 5).reshape(B * Hp * Wp, C * ph * pw)
        weight = self.w["patchify"].reshape(-1, C * ph * pw)  # [D, C*ph*pw]

        tt_patches = ttnn.from_torch(patches.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight.t().contiguous().to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_y = ttnn.matmul(tt_patches, tt_weight)
        y_flat = ttnn.to_torch(tt_y).float()
        D = weight.shape[0]
        return y_flat.view(B, Hp * Wp, D), (Hp, Wp)

    def _unpatchify(self, x: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        """x: [B, Hp*Wp, D]. Inverse of patchify: ConvTranspose2d(D, C, kernel=stride=patch,
        bias=True) == matmul + fold. Weight is [D, C, ph, pw] (ConvTranspose2d layout)."""
        import ttnn

        B, T, D = x.shape
        ph, pw = self.patch
        w = self.w["unpatchify_w"]  # [D, C, ph, pw]
        C = w.shape[1]
        w_flat = w.reshape(D, C * ph * pw)

        # ConvTranspose2d's bias is per OUTPUT CHANNEL ([C], broadcast over the ph x pw
        # sub-block), not per flattened (c, i, j) triplet -- expand it to match the
        # weight's own C-major flatten order (repeat_interleave, not tile/expand).
        bias_expanded = self.w["unpatchify_b"].repeat_interleave(ph * pw)  # [C*ph*pw]

        tt_x = ttnn.from_torch(x.reshape(B * T, D).to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_w = ttnn.from_torch(w_flat.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_y = ttnn.matmul(tt_x, tt_w)
        y = ttnn.to_torch(tt_y).float() + bias_expanded.view(1, -1)  # [B*T, C*ph*pw]

        y = y.view(B, hp, wp, C, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(B, C, hp * ph, wp * pw)
        return y

    def _out_norm(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        import ttnn

        fc_w = self.w["out_norm_fc"]
        cond_act = torch.nn.functional.silu(cond)
        ab = cond_act @ fc_w.t()
        a, b_ = ab.chunk(2, dim=-1)

        tt_x = ttnn.from_torch(x.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_a = ttnn.from_torch(a.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        tt_b = ttnn.from_torch(b_.to(torch.bfloat16), device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        sq_mean = ttnn.mean(ttnn.multiply(tt_x, tt_x), dim=-1, keepdim=True)
        rstd = ttnn.rsqrt(ttnn.add(sq_mean, 1e-6))
        normed = ttnn.multiply(tt_x, rstd)
        scaled = ttnn.multiply(normed, ttnn.add(tt_a, 1.0))
        out = ttnn.add(scaled, tt_b)
        return ttnn.to_torch(out).float()

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        rope_angles,
        mouse: torch.Tensor,
        button: torch.Tensor,
        scroll: torch.Tensor,
        frame_idx: int,
        is_frozen: bool,
    ) -> torch.Tensor:
        """x: [B, N=1, C, H, W]. Returns the same shape (denoised/updated latent)."""
        B, N, C, H, W = x.shape
        cond = self._noise_conditioner(sigma)
        ctrl_emb = self._ctrl_embedding(mouse, button, scroll)

        tokens, (hp, wp) = self._patchify(x.view(B * N, C, H, W))

        v1 = None
        h = tokens
        for i, layer in enumerate(self.layers):
            if frame_idx == 0:
                h, v1 = layer.prefill_forward(h, rope_angles, cond, ctrl_emb, v1=v1, is_frozen=is_frozen)
            else:
                h, v1 = layer.decode_forward(h, rope_angles, cond, ctrl_emb, frame_idx=frame_idx, is_frozen=is_frozen, v1=v1)

        h = self._out_norm(h, cond)
        h = torch.nn.functional.silu(h)
        out = self._unpatchify(h, hp, wp)
        return out.view(B, N, C, H, W)
