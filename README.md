# tt-waypoint

From-scratch TTNN bring-up of [Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
on Tenstorrent Blackhole (P300×2) — a custom autoregressive causal diffusion transformer
("world model": interactive video generation conditioned on mouse/button/scroll input),
brought up from scratch rather than reused from an existing TTNN model like
[tt-skyreels](https://github.com/tsingletaryTT/tt-skyreels). **Bring-up in progress, not
yet packaged or served** — see [Status](#status) below and
[BRINGUP_LOG.md](BRINGUP_LOG.md) for the full, timestamped history (wall-clock time,
approximate token usage, every bug found and how).

## What this is

Waypoint-1.5-1B denoises one full 512-token frame per step (bidirectional attention
within the frame), conditioned on a per-layer ring-buffer cache of previous frames rather
than a per-token KV cache, plus a small CNN VAE (`ChunkedStreamingTAEHV`) that turns
generated latents into RGB pixels. Nothing in tt-metal's `models.tt_dit` already covers
this shape, so every piece here — the attention/cache scheme, the conditioning heads, the
VAE's conv stack — was ported and hardware-verified from scratch against the real HF
reference, not reused wholesale the way tt-skyreels reuses `WanTransformer3DModel`. See
[PORT_PLAN.md](PORT_PLAN.md) for the staged plan and [CLAUDE.md](CLAUDE.md) for why the
bring-up methodology (the `ttm-*` skill family's conventions: `LightweightModule`,
`from_state_dict`, PCC-based verification) and the repo location (here, not tt-metal's
`models/autoports/` tree) follow two different, deliberate conventions.

- **Model**: Waypoint-1.5-1B, autoregressive world model, 24 transformer layers + CNN VAE
- **Hardware**: Tenstorrent Blackhole, P300×2 board (this box has no P150 — confirmed via
  `tt-smi -ls`, not assumed)
- **Weights**: [`Overworld/Waypoint-1.5-1B`](https://huggingface.co/Overworld/Waypoint-1.5-1B)
  (a pointer — never embedded here; downloaded to your own HF cache)

## Repo layout

| Path | What |
| --- | --- |
| `waypoint_ttnn/tt/functional_decoder.py` | One transformer block (`FunctionalDecoder`) — attention, per-layer KV cache, conditioning |
| `waypoint_ttnn/tt/full_model.py` | Full 24-layer assembly: patchify → blocks → out_norm → unpatchify |
| `waypoint_ttnn/tt/vae_decoder.py` | VAE decoder: latents → RGB, real `ttnn.conv2d`/`ttnn.upsample` |
| `waypoint_ttnn/tt/vae_encoder.py` | VAE encoder: RGB → latent (session seeding) |
| `waypoint_ttnn/tt/rope.py` | `compute_rope_angles` — OrthoRoPE for an arbitrary frame index |
| `waypoint_ttnn/tt/generation_loop.py` | `WaypointGenerator` — the full interactive seed/step loop |
| `waypoint_ttnn/session.py` | Mesh-device + model singleton, shared by the Gradio app |
| `app.py` | Local Gradio UI — seed a session from an image, then step it (port 7862) |
| `.disco/app.yaml` | [tt-discolike](https://github.com/tsingletaryTT/tt-discolike) catalog manifest |
| `waypoint_ttnn/tests/` | Hardware correctness tests, each checked against a real captured HF reference |
| `waypoint_ttnn/capture_*.py` | Scripts that capture reference activations from the real HF model (see [Reference activations](CLAUDE.md#reference-activations-arent-committed)) |
| `PORT_PLAN.md` | The staged bring-up plan, including the benchmarking plan |
| `BRINGUP_LOG.md` | Timestamped log: every stage, every bug, every hardware-verified number |

## Running the Gradio UI

```bash
pip install gradio
python app.py    # http://localhost:7862
```

Upload a starting image, click **Start session** (opens the device, loads weights —
slow on the first call), then pick a direction and click **Step** to generate the next
frame. Each step is a real forward pass on hardware (denoise + VAE decode), not a
simulation — see `app.py`'s own docstring for the button-mapping caveat (the real
model's 256-wide button vector has no published semantics, so this UI only drives
mouse/scroll).

## Running the tests

Every test needs the real `Overworld/Waypoint-1.5-1B` weights (downloaded automatically
on first use) and a captured reference activation file — regenerate those first (see
[CLAUDE.md](CLAUDE.md#reference-activations-arent-committed)), then run under a
[gozer](https://github.com/tsingletaryTT) chip lease (or adapt to your own device-locking
scheme):

```bash
gozer run --chips 1 --who "you:tt-waypoint" --reason "bring-up test" -- \
  python3 waypoint_ttnn/tests/test_full_model.py
```

## Status

Stages 1-5 hardware-verified (text encoder reuse, patchify/AdaLN, conditioning
embeddings, attention + multi-frame KV cache, VAE encode+decode). Stage 6 (the full
interactive loop, `generation_loop.py`) is wired and the SEED path is verified (latent
corr 0.9996, decoded-pixel corr ~0.96) — but **generated frames currently look wrong**:
decoded-pixel correlation against the real reference collapses to 0.05-0.24 for frames
produced by the multi-step denoising loop, root-caused to per-call bf16 noise that the
rectified-flow update sums explicitly (rather than diluting through a residual stream),
amplified further by the VAE decoder's saturating nonlinearity. See PORT_PLAN.md's Stage
6 section for the full isolation trail — this is a real, currently-open quality
limitation, not a discovered-and-fixed bug, so packaging and the HF push are on hold
until it's resolved or more firmly characterized as an accepted limitation. See
BRINGUP_LOG.md for exact numbers — the 24-layer full-model correlation (0.948) sitting
below the 0.99 target is a related, separately-documented finding (ordinary bf16
hardware compounding over a deep stack).

Once Stage 6's frame quality is resolved, this will be packaged and pushed to Hugging
Face under the `episod` account, public, the same way
[episod/tt-skyreels](https://huggingface.co/episod/tt-skyreels) was.

## License

Apache 2.0 (matching the upstream `Overworld/Waypoint-1.5-1B` weights' license terms —
see the weights repo for details).
