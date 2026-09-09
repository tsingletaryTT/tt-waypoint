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

See BRINGUP_LOG.md for the live, timestamped record. Current state (updated as of the
`functional_decoder.py` layer-0 milestone): Stages 1-3 (text encoder reuse, patchify/
AdaLN, conditioning embeddings) hardware-verified. Stage 4 (attention + KV cache) — a
real wrong-hypothesis-caught-and-fixed finding (dense attention over the full zero-padded
capacity buffer, not a sparse block-mask) — verified for a single layer's frame-0 case
with PCC 0.999 against the HF reference (exceeds the `ttm-functional-decoder` skill's own
0.995 bar). Not yet done: multi-frame cache correctness (only frame 0's empty-cache case
is proven), all 24 layers assembled together, the VAE, a full interactive generation
loop, and packaging.
