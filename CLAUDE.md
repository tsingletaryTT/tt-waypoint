# tt-waypoint — project log

From-scratch TTNN bring-up of [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
on Tenstorrent Blackhole (P300×2) — a custom autoregressive causal diffusion transformer
("world model": real-time interactive video generation conditioned on keyboard/mouse
input), not a packaging job like tt-skyreels. See [BRINGUP_LOG.md](BRINGUP_LOG.md) for
the full, timestamped history (wall-clock time, approximate token usage, every bug found
and how) and [PORT_PLAN.md](PORT_PLAN.md) for the staged plan.

## Bring-up vs. packaging: two different conventions, on purpose

This repo follows two different sets of conventions for two different phases of work,
and it's worth being explicit about which applies where:

- **Bring-up** (getting a TTNN implementation functionally correct against the HF
  reference) uses the `ttm-*` skill family's methodology and file-internal conventions:
  `LightweightModule` base class, a real `from_state_dict` weight-loading boundary,
  `prefill_forward`/`decode_forward`, PCC-based verification. Those skills (e.g.
  `ttm-functional-decoder`) name `models/autoports/<model>/tt/...` inside a tt-metal
  checkout as the file's home — but that convention exists for models being brought up
  FOR upstream tt-metal inclusion. This model isn't going upstream, so the files live
  here instead (`waypoint_ttnn/tt/functional_decoder.py`), importing
  `models.common`/`models.tt_dit`/etc. from tt-metal via `sys.path`, the same way
  `skyreels_ttnn/pipeline_skyreels.py` imports `models.tt_dit.*`. See
  `waypoint_ttnn/tt/functional_decoder.py`'s own docstring and
  [[agentic-bringup-plus-decoupled-repo]] (memory) for the full reasoning — this is a
  deliberate, repeatable pattern, not a one-off deviation.
- **Packaging** (once something here is functionally correct enough to serve) will
  follow tt-model-manager's own convention exactly, the same way tt-skyreels did: a
  `tt_model_package.yaml`, pushed to GitHub + HF as its own package. Not started yet —
  see PORT_PLAN.md's Stage 6.

## Status

See BRINGUP_LOG.md for the live, timestamped record. Stages 1-5 are hardware-verified:
text encoder reuse, patchify/AdaLN/conditioning embeddings, attention + multi-frame KV
cache (24-layer full-model correlation 0.948 -- the residual gap below the 0.99 bar is
diagnosed as ordinary bf16 hardware compounding over a deep stack, not a logic bug, see
BRINGUP_LOG.md), and the VAE (encoder + decoder both verified, real `ttnn.conv2d`/
`ttnn.upsample` compute). Not yet done: the full interactive generation loop and
packaging (Stage 6).

## Repo hosting

Public at [github.com/tsingletaryTT/tt-waypoint](https://github.com/tsingletaryTT/tt-waypoint),
committed to `main` regularly as work lands. Once the model reaches HF-ready status
(Stage 6 complete, packaged), it also gets published to Hugging Face under the `episod`
account (also public) — this repo's own HF namespace confusion during tt-skyreels'
bring-up (`tsingletary` vs. the actual authenticated `episod` identity) is exactly why
this is spelled out explicitly here rather than assumed.

## Reference activations aren't committed

`ref_activations/*.pt` (except the tiny `transformer_config.pt`/`vae_config.json`) are
gitignored — they're large captured tensors (one hit 97MB, close to GitHub's 100MB hard
block) that are fully reproducible by re-running the `capture_*.py` scripts (which ARE
tracked) against the real downloaded HF checkpoint. Regenerate them locally rather than
expecting them to be present after a fresh clone.
