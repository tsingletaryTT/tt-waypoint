# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Executes PORT_PLAN.md's benchmarking plan: warm vs cold, a transformer-vs-VAE-decode
latency breakdown, and a session-length scaling check (does per-frame latency stay flat
as the KV cache accumulates more history, matching Stage 4's "dense attention over the
full zero-padded capacity buffer regardless of how much of it is real history" finding).

Host-side wall-clock timing (`time.perf_counter` around each phase), not a
Tracy/tt-perf-report device-side profile -- informative for "is this in the right
ballpark, does latency grow with session length", not a claim of kernel-level
precision. A real perf-tuning pass would want the latter; not attempted here.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "Stage 6/7 benchmark"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image


#: How many generated frames to benchmark. Kept modest (not 50+) so this stays
#: tractable in one run -- the checkpoints below are still spread widely enough to see
#: a scaling trend if there is one.
N_STEPS = 16
#: Frame indices to call out explicitly in the summary (frame 1 is deliberately
#: excluded from any "warm" average -- it pays first-call JIT compile cost).
CHECKPOINTS = (1, 5, 10, 15)


def main():
    import ttnn

    from waypoint_ttnn import session as session_module
    from waypoint_ttnn.tt.generation_loop import WaypointGenerator

    t0 = time.perf_counter()
    device, world_model, vae_encoder, vae_decoder, hf_config = session_module.ensure_waypoint_models()
    t_load = time.perf_counter() - t0
    print(f"[bench] model load (weights + device open): {t_load:.1f}s")

    gen = WaypointGenerator(world_model, vae_encoder, vae_decoder, hf_config, device)

    mouse = torch.zeros(1, 1, 2)
    button = torch.zeros(1, 1, 256)
    scroll = torch.zeros(1, 1, 1)
    patch_size = vae_decoder.patch_size

    vae_scale_factor = 16
    pixel_h = gen.latent_h * vae_scale_factor
    pixel_w = gen.latent_w * vae_scale_factor

    img = Image.open(
        "/home/ttuser/code/tt-waypoint/ref_activations/seed_frame_ref.png"
    ).convert("RGB").resize((pixel_w, pixel_h))
    import numpy as np
    img_t = torch.from_numpy(np.array(img))

    t_down = vae_encoder.t_downscale
    rgb = img_t.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255).permute(0, 3, 1, 2)
    patchified = F.pixel_unshuffle(rgb, patch_size)
    seed_frames = [
        ttnn.from_torch(
            patchified[t : t + 1].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
            device=device, layout=ttnn.TILE_LAYOUT,
        )
        for t in range(t_down)
    ]

    # --- Seed: a once-per-session cost, reported separately from step()'s repeated cost ---
    t0 = time.perf_counter()
    seed_latent = gen.seed(seed_frames, mouse, button, scroll)
    t_seed = time.perf_counter() - t0

    t0 = time.perf_counter()
    gen.decode_frame(seed_latent)
    t_seed_decode = time.perf_counter() - t0
    print(f"[bench] seed(): encode+commit {t_seed:.2f}s, decode {t_seed_decode:.2f}s "
          f"(cold -- first-ever ttnn call, pays JIT compile)")

    # --- Per-frame breakdown: transformer (denoise+commit) vs VAE decode, separately ---
    rows = []
    for i in range(1, N_STEPS + 1):
        t0 = time.perf_counter()
        latent = gen.step(mouse, button, scroll)
        t_step = time.perf_counter() - t0

        t0 = time.perf_counter()
        gen.decode_frame(latent)
        t_decode = time.perf_counter() - t0

        rows.append((i, t_step, t_decode))
        marker = " <-- checkpoint" if i in CHECKPOINTS else ""
        print(f"[bench] frame {i:2d}: transformer(denoise+commit) {t_step:.2f}s, "
              f"vae_decode {t_decode:.2f}s, total {t_step + t_decode:.2f}s{marker}")

    # --- Summary: warm average (excludes frame 1, which pays first-call JIT cost) ---
    warm = rows[1:]
    if warm:
        avg_step = sum(r[1] for r in warm) / len(warm)
        avg_decode = sum(r[2] for r in warm) / len(warm)
        print(f"[bench] warm average (frames 2-{N_STEPS}): "
              f"transformer {avg_step:.2f}s, vae_decode {avg_decode:.2f}s, "
              f"total {avg_step + avg_decode:.2f}s ({1.0/(avg_step+avg_decode):.3f} fps effective)")

    print("[bench] checkpoint latencies (session-length scaling check -- Stage 4 "
          "predicts this should stay roughly FLAT, not grow, since attention always "
          "runs dense over the full fixed-size capacity buffer regardless of how much "
          "of it is real history):")
    for i, t_step, t_decode in rows:
        if i in CHECKPOINTS:
            print(f"[bench]   frame {i:2d}: transformer {t_step:.2f}s, decode {t_decode:.2f}s")

    ttnn.close_mesh_device(device)
    print("[bench] DONE")


if __name__ == "__main__":
    main()
