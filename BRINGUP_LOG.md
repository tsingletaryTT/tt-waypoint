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

| Sample frame captured, plan written | 2026-09-09T20:16:32Z | 14,959,743 | Saved and sent an actual generated frame (`seed_frame.png`) as tangible proof the reference pipeline works. Wrote `PORT_PLAN.md` (6 staged phases) and confirmed by reading real code that Stage 1 (UMT5-XL text encoder) is checkpoint-agnostic reuse from `models.tt_dit.pipelines.wan.text_encoder.TextEncoder` — no architecture guessing, verified against the actual source. Elapsed since start: 17m53s. |

| **Stage 1 hardware-verified** | 2026-09-09T20:20:50Z | 14,942,273 | **Milestone**: UMT5-XL text encoder running on REAL Blackhole hardware (P300×2 board, 1×1 mesh, one chip leased via gozer). Hit one real bug (`ftfy` missing from the shared `.tenstorrent-venv` — same package tt-skyreels' bring-up hit independently, now confirmed to also be a gap in tt-metal's own WAN text encoder module, not just SkyReels' port). Fixed, then: constructed `models.tt_dit.encoders.umt5.UMT5Encoder` directly (bypassing tt_dit's WAN-specific `TextEncoder` wrapper, which assumes a subfolder layout `google/umt5-xl` doesn't have), loaded `UMT5EncoderModel.from_pretrained("google/umt5-xl").state_dict()` into it — **zero missing, zero unexpected keys** — and ran a real forward pass on the chip. Output shape matched the torch reference exactly ([1,512,2048]); mean abs diff 0.0045 (expected bf16 noise, not a bug). Elapsed since start: 22m11s. |

| **Stage 2 hardware-verified** | 2026-09-09T20:49:28Z | 14,956,882 | **Milestone**: patchify (Conv2d-as-unfold-matmul, [1,32,32,64]→[1,2048,16,32]) and AdaLN/out_norm (weight-free RMSNorm + SiLU-conditioned scale/bias) both verified on real Blackhole hardware against captured reference activations (`ref_activations/block_activations.pt` — captured once via PyTorch forward hooks, so later stages don't need to re-run the ~10min CPU reference each time). Confirmed the exact latent geometry: VAE outputs a 32×64×32ch latent per frame, patchify(2,2) → 16×32=512 tokens, matching `tokens_per_frame`. Real bug-fixing along the way: the shared `.tenstorrent-venv` lacked `tensordict`/`einops` (needed `trust_remote_code`, now installed) and its `diffusers` predates `AutoModel`/`modular_pipelines` entirely — worked around by loading the transformer's weights directly via `safetensors.torch.load_file` rather than touching diffusers at all for hardware-side checks. Both diffs: mean abs ≈0.0023 (expected bf16 noise). Elapsed since start: 51m. |

| **Stage 3 hardware-verified** | 2026-09-09T20:51:32Z | 14,950,850 | **Milestone**: `NoiseConditioner` (Fourier features → MLP; mean abs diff 0.0093, bf16 noise) and `ControllerInputEmbedding` (plain MLP; **exact match, 0.0 diff**) both verified on real hardware. Stages 2-3 (patchify, unpatchify's inverse still pending explicit check but same op family as patchify, AdaLN, both conditioning embeddings) are now done. Remaining before a full frame: `CondHead`/`CrossAttention`/`MLPFusion` (straightforward, same linear-algebra pattern as everything so far) and then **Stage 4: the real risk** — local/global block-sparse causal attention with the per-layer dilated ring-buffer KV cache. Elapsed since start: 53m. |

| **Stage 4 first pass: real finding, hardware-verified** | 2026-09-09T21:09:51Z | 14,886,832 | **Milestone, and a genuine correction to my own hypothesis.** Set out to test "attention reduces to plain self-attention over the current frame for an empty cache" — this was WRONG, caught by verification, not assumed. The actual bug hunt (documented in full below) revealed the real behavior: the reference runs **dense (unmasked) attention over the ENTIRE per-layer capacity buffer** (8704 positions for a local layer: 16-frame ring + 1-frame tail), where never-written slots are zero and contribute their (small, predictable) softmax mass rather than being excluded. My first attempt — masking to only the "written" tail — was mathematically reasonable from reading the code but empirically wrong (mean abs diff 0.093, correlation 0.94 — a real bug, not bf16 noise). Verified the correct behavior first in pure PyTorch (mean abs diff 0.00046 using the real captured q/k/v) then on real Blackhole hardware (mean abs diff 0.018 — larger than shorter-sequence stages, plausibly genuine bf16 accumulation error over an 8704-long softmax vs. 512; flagged for revisit, not dismissed as noise). **Practical implication: Stage 4 does not need any block-sparse masking or gather logic at all for a first correctness pass** — just replicate the reference's exact zero-initialized ring-buffer layout and run plain dense attention over the whole buffer every time. This meaningfully de-risks the stage I scoped as "the hard part." Elapsed since start: 1h11m. |

### How the Stage 4 bug was actually found (for anyone continuing this)

1. First attempt used a captured activation that turned out to be from **frame index 2** (the pipeline pre-seeds multiple frames from the starting image before real generation begins), not frame 0 — invalidating the "empty cache" assumption before I'd even gotten to the real bug. Caught by checking `pos_ids["f_pos"].unique()` directly rather than trusting which call a "fire-once" hook happened to catch.
2. Fixed by calling `WorldModel.forward` **directly** with a synthetic frame 0 and a freshly-constructed `StaticKVCache` — full control, full visibility, and much faster than running the real pipeline (seconds vs. ~10 minutes).
3. Still 0.093 mean diff. Rather than keep guessing at the mask semantics, **monkeypatched `flex_attention` itself** to capture its exact `(q, k, v, block_mask)` arguments — ground truth, zero reimplementation risk.
4. Tested the real captured `q, k, v` two ways: sliced to just the "written" tail (0.094 mean — still wrong) vs. the full untouched capacity buffer (0.00046 mean — correct). That comparison is what actually located the bug.

*(Rows appended as work progresses. Elapsed time and token burn are computed against the
Start row.)*
