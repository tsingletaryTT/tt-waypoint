# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Stage 6: the full interactive generation loop, wiring together the already-verified
WaypointWorldModel (transformer) and WaypointVAEEncoder/WaypointVAEDecoder.

Ports the real pipeline's exact per-session/per-frame protocol (modular_blocks.py's
WorldEnginePrepareLatentsStep + WorldEngineDenoiseLoop, replicated directly rather than
going through diffusers' ModularPipeline machinery -- same rationale as every capture
script in this repo: full control, full visibility, no ~10min-per-call CPU reference
cost):

- **Seeding a session** (`seed()`, called ONCE): VAE-encode a real starting image into a
  latent, then commit it directly into the transformer's KV cache via a SINGLE unfrozen
  forward call at frame_idx=0, sigma=0. There is NO denoising loop for the seed frame --
  the real image's own encoded latent IS frame 0's history.
- **Generating a new frame** (`step()`, called once per frame thereafter): draw fresh
  noise, run K FROZEN forward calls (one per adjacent pair in `scheduler_sigmas`,
  rectified-flow integration `x = x + dsigma * v` where `v` is the transformer's
  velocity-field output -- NOT a denoised x directly), then ONE final UNFROZEN forward
  call at sigma=0 to commit the clean result into the cache's permanent ring for future
  frames to attend to.

`rope_angles` can't be reused from a fixed capture here (every prior test did exactly
that, since they only ever exercised frame 0/1) -- see rope.py, verified bit-for-bit
against real captured reference rope tensors before being trusted for arbitrary frames.
"""

from __future__ import annotations

from typing import Optional

import torch

from waypoint_ttnn.tt.rope import compute_rope_angles, compute_ts_mult


class WaypointGenerator:
    def __init__(self, world_model, vae_encoder, vae_decoder, hf_config, mesh_device):
        self.world_model = world_model
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.ts_mult = compute_ts_mult(hf_config)
        self.sigmas = torch.tensor(hf_config.scheduler_sigmas, dtype=torch.float32)

        self.channels = hf_config.channels
        ph, pw = hf_config.patch
        # The transformer's raw (pre-patchify) latent grid IS the VAE's own latent
        # output/input grid -- one shared representation, confirmed against the real
        # config (vae_scale_factor=16, pixel_H/W = latent_H/W * 16) and the VAE's own
        # verified test (base_h/w there was just an arbitrary smaller stand-in shape).
        self.latent_h = hf_config.height * ph
        self.latent_w = hf_config.width * pw

        self.frame_timestamp = 0

    # ---- latent <-> tensor boundary (transformer is plain torch, VAE is ttnn) -------

    def _latent_to_torch(self, latent_nhwc) -> torch.Tensor:
        import ttnn

        return ttnn.to_torch(latent_nhwc).float().permute(0, 3, 1, 2).unsqueeze(1)  # [1,1,C,H,W]

    def _latent_to_ttnn(self, latent_nchw: torch.Tensor):
        import ttnn

        x = latent_nchw.squeeze(1).permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)  # NHWC
        return ttnn.from_torch(x, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)

    # ---- public API ------------------------------------------------------------------

    def seed(self, seed_frames_nhwc: list, mouse: torch.Tensor, button: torch.Tensor,
              scroll: torch.Tensor) -> torch.Tensor:
        """`seed_frames_nhwc`: `vae_encoder.t_downscale` ttnn NHWC tensors, already
        pixel-unshuffled RGB (matching WaypointVAEEncoder.encode's own contract exactly
        -- see test_vae_encoder.py). Commits the encoded latent as frame 0's history.
        Returns the seed latent (torch [1,1,C,H,W]) so the caller can also decode+display
        it as the session's first visible frame if desired."""
        assert self.frame_timestamp == 0, "seed() must be called once, before any step()"
        # encode()'s height/width is the PATCHIFIED-RGB size feeding its 3 stride-2
        # convs, spatial_downscale times LARGER than the output latent -- not the
        # latent's own size (that's what decode_frame() uses instead, since the
        # decoder's input IS the raw latent).
        enc_h = self.latent_h * self.vae_encoder.spatial_downscale
        enc_w = self.latent_w * self.vae_encoder.spatial_downscale
        latent_tt = self.vae_encoder.encode(seed_frames_nhwc, enc_h, enc_w)
        latent = self._latent_to_torch(latent_tt)

        rope = compute_rope_angles(self.hf_config, self.frame_timestamp, self.ts_mult)
        zero_sigma = torch.zeros(1, 1)
        self.world_model.forward(latent, zero_sigma, rope, mouse, button, scroll,
                                  frame_idx=self.frame_timestamp, is_frozen=False)
        self.frame_timestamp += 1
        return latent

    def step(self, mouse: torch.Tensor, button: torch.Tensor, scroll: torch.Tensor,
              noise_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generates ONE new frame conditioned on the cached history + these controls.
        `noise_override` exists only for exact-reproducibility testing against a
        specific captured reference run (see test_generation_loop.py) -- production
        callers should never pass it, since a real session always wants fresh noise.
        Returns the clean latent (torch [1,1,C,H,W]); call `decode_frame` for pixels."""
        rope = compute_rope_angles(self.hf_config, self.frame_timestamp, self.ts_mult)
        if noise_override is not None:
            x = noise_override.float()
        else:
            x = torch.randn(1, 1, self.channels, self.latent_h, self.latent_w, dtype=torch.float32)

        for sigma, dsigma in zip(self.sigmas[:-1], self.sigmas.diff()):
            sigma_t = torch.full((1, 1), float(sigma))
            v = self.world_model.forward(x, sigma_t, rope, mouse, button, scroll,
                                          frame_idx=self.frame_timestamp, is_frozen=True)
            x = x + dsigma.item() * v

        zero_sigma = torch.zeros(1, 1)
        self.world_model.forward(x, zero_sigma, rope, mouse, button, scroll,
                                  frame_idx=self.frame_timestamp, is_frozen=False)
        self.frame_timestamp += 1
        return x

    def decode_frame(self, latent_nchw: torch.Tensor) -> list:
        """latent_nchw: torch [1,1,C,H,W] (from seed()/step()). Returns a list of ttnn
        NHWC [1,H,W,12] (still patchified) frames -- caller pixel-shuffles + casts to
        uint8, matching WaypointVAEDecoder's own contract (see test_vae_decoder.py's
        `postprocess` helper)."""
        latent_tt = self._latent_to_ttnn(latent_nchw)
        return self.vae_decoder.decode(latent_tt, self.latent_h, self.latent_w)
