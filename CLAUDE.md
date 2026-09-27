# tt-waypoint — project log

From-scratch TTNN bring-up of [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
on Tenstorrent Blackhole (requirement: one chip, serve profile `p150`; verified on one
chip of a P300c, since this box has no P150) — a custom autoregressive causal diffusion transformer
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
- **Packaging** follows tt-model-manager's own convention exactly. It was first a v5.1
  container (`tt_model_package.yaml`, 2026-09-09), then repackaged on 2026-09-15 as a v6
  thin bundle (`waypoint_ttnn` wheel + `tt-waypoint-models-closure` wheel, `ttnn` from
  PyPI) published as `episod/tt-waypoint`; the container manifest was deleted then.

## Status

See BRINGUP_LOG.md for the live, timestamped record. Stages 1-5 are hardware-verified:
text encoder reuse, patchify/AdaLN/conditioning embeddings, attention + multi-frame KV
cache (24-layer full-model correlation 0.948 -- the residual gap below the 0.99 bar is
diagnosed as ordinary bf16 hardware compounding over a deep stack, not a logic bug, see
BRINGUP_LOG.md), and the VAE (encoder + decoder both verified, real `ttnn.conv2d`/
`ttnn.upsample` compute). Stage 6 (interactive seed/step loop) and packaging are done
too: served and published as a v6 thin bundle (see above and BRINGUP_LOG.md).

**Hardware wording, to keep consistent everywhere:** the requirement is one Blackhole
chip, i.e. a 1x1 mesh (`SUPPORTED_MESH_SHAPES = {(1, 1)}`). The serve profile is named
`p150` because tt-model-manager needs a board label whose chip count equals the mesh's,
and P150 is the single-chip Blackhole board. Every verification ran on **one chip of a
P300c**; it has never run on a physical P150, and there is no multi-chip profile.

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

## Log

### 2026-09-27 — pin the weights, fetch only what's read (waypoint_ttnn 0.1.1)

Prompt (from a packaging-hygiene pass across three model repos, no hardware allowed):
pin `Overworld/Waypoint-1.5-1B` in `session.py`'s `snapshot_download`, restrict
`allow_patterns` to the files the server actually reads (verified by reading the load
code, not guessed), clarify the p150-vs-P300c hardware wording, bump 0.1.0 -> 0.1.1.

- **Pin**: `PINNED_WEIGHTS_REVISION = 391f92827075edcf4a8b3c8a2ddae010698f8636` (HF API
  sha on 2026-09-27; upstream lastModified 2026-07-16, before bring-up, so it is the
  verified revision). `$TT_MODEL_WEIGHTS_REVISION` (exported by tt-model-manager's v6
  run.sh) overrides it; an empty value falls back to the pin.
- **Filter**: the served closure (session.py, server/app.py, tt/*.py) opens exactly
  three snapshot files: `transformer/config.json`,
  `transformer/diffusion_pytorch_model.safetensors`,
  `vae/diffusion_pytorch_model.safetensors`. Nothing under tt/ or server/ touches the
  hub or the snapshot. That skips the root `model.safetensors` (3.72 GB), `assets/`
  (~250 MB) and the upstream .py sources. `vae/config.json` is not read (VAE shape is
  hardcoded), so it is not fetched.
- **Guard**: `waypoint_ttnn/tests/test_weights_pin.py` (pure Python, no ttnn) asserts
  the pin + filter reach the actual `snapshot_download` call, and parses session.py's
  own `os.path.join(snapshot_dir, ...)` reads to check they equal the filter. Seen to
  fail both ways: dropping `revision=` turns 2 tests red; adding a read of
  `vae/config.json` turns the coverage test red.
- **Not changed**: `HF_MODEL` (exported by run.sh) is still ignored in favour of the
  hardcoded repo id; the bundle's run.sh still sets `HF_HOME=$HERE/.hf`, so a
  `--with-weights` pull is re-downloaded at first serve (a tt-model-manager template
  issue, see the bench notes). A repackage must pass the same pin and the same
  allow-patterns to `package-thin` so the manifest matches the code.
