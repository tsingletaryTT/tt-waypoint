# Waypoint-1.5-1B TTNN bring-up — time/token/hardware log

Tracking wall-clock time, approximate token usage, and hardware configuration for the
Claude-driven bring-up of [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
on Tenstorrent Blackhole, per Taylor's explicit request to measure "how long it takes,
how many tokens it burns, and which hardware configuration."

## Hardware

**P300×2** (2× P300c boards, 4 chips total). This is the ONLY configuration available on
this box — confirmed via `tt-smi -ls`: no P150 boards are physically present, so
"P150x4" was never actually an open choice here. Can be opened as a (1,4) line mesh or a
(2,2) "QB2" mesh depending on what the model needs (see tt-skyreels' bring-up for why
that distinction matters on this specific box).

## Token accounting method (caveated)

This session shows a `total_tokens` remaining counter in the low-level system context.
I'm using it as `tokens_burned ≈ baseline_remaining − current_remaining` at each
checkpoint. This is an approximation, not a precise ledger: it can be affected by context
compaction/caching behavior outside my control, and it covers only tokens spent *in this
conversation* — not any separate tool-internal cost. Treated as directionally accurate,
not exact.

## Scoping phase (before this log existed — reconstructed from the conversation)

Not counted in the totals below (those start at the bring-up commit point), but for
context: architecture research (reading `transformer/model.py`, `vae/ae_model.py`,
`modular_blocks.py`, loading the reference pipeline components on CPU) took roughly one
conversation segment before Taylor said "I want to try."

## Bring-up phase

| Checkpoint | Wall clock (UTC) | `total_tokens` remaining | Notes |
|---|---|---|---|
| **Start** | 2026-09-09T19:58:39Z | 14,995,739 | Repo created, hardware confirmed (P300×2 only), this log started. |
| Reference model runs | 2026-09-09T20:11:44Z | 14,976,426 | **Milestone**: got the actual upstream reference PyTorch pipeline (CPU, no TT hardware) running end to end — seeded a world from an image, then generated 2 more controller-conditioned frames, no crashes. Hit and fixed one real bug of my own along the way (forced `dtype=torch.float32`, which the pipeline's internal VAE-decode step doesn't support — matched the README's `bfloat16` instead and it worked). Confirmed exact tensor contracts: `prompt_embeds` [1,512,2048], `button_tensor` [1,1,256], `mouse_tensor` [1,1,2], `scroll_tensor` [1,1,1], `scheduler_sigmas` [5], `frame_timestamp` [1,1], plus the `StaticKVCache` object shape. Elapsed: 13m5s; ≈19,313 tokens by the approximation above. Each single-frame CPU generation call itself takes ~8-10 minutes wall clock (unaccelerated `flex_attention` eager fallback, per its own warning) — a real, separate cost from my own working time, and a preview of how expensive *correctness-checking against this reference* will be at every TTNN porting step. |

*(Rows appended as work progresses. Elapsed time and token burn are computed against the
Start row.)*
