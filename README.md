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

Stages 1-6 hardware-verified end to end (text encoder reuse, patchify/AdaLN, conditioning
embeddings, attention + multi-frame KV cache, VAE encode+decode, full interactive
seed/step loop). Getting Stage 6 to this point took a real investigation — pixel
correlation against one specific reference run is the WRONG bar for an iterative,
self-referential denoising loop (any per-step difference compounds trajectories apart,
the same way two chaotic systems diverge from slightly different initial conditions), so
"generated frames don't match the reference bit-for-bit" initially looked alarming but
wasn't the right question. The right checks — every individual denoising step matches
the established correlation baseline (further improved via a real fp32-accumulation fix,
`compute_kernel_config` with `fp32_dest_acc_en`+`HiFi4`, exactly upstream's own
recommended lever for narrowing the bf16↔fp32 gap since true fp32 SDPA doesn't exist
anywhere in tt-metal — confirmed against branch/tag history and an open upstream issue),
and a real, in-distribution seed image produces visually coherent, plausible generated
frames — both pass. See PORT_PLAN.md's Stage 6 section and BRINGUP_LOG.md for the full
investigation trail, including why an earlier synthetic random-static seed test looked
catastrophic (even the reference's own output was unstructured noise under that
out-of-distribution input) and how that was resolved.

Packaged with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager)
(`tt_model_package.yaml`) and hardware-verified end to end: built, served, and exercised
through the real HTTP API (`POST /v1/sessions` to seed from an image, `POST
/v1/sessions/{id}/step` to advance a frame) against the actual running container — see
PORT_PLAN.md's Stage 7 section for the two real bugs found and fixed along the way.
Published, public, on Hugging Face:
[episod/tt-waypoint](https://huggingface.co/episod/tt-waypoint).

```bash
tt-model pull episod/tt-waypoint --with-weights
tt-model serve episod/tt-waypoint
```

## Benchmarks

First-correctness-pass numbers on a single chip (`waypoint_ttnn/benchmark.py`, real
hardware, untraced/un-batched/bf16-everywhere -- no perf-tuning pass attempted yet):
warm per-frame transformer latency (4 denoise steps + 1 commit) averages 27.4s; VAE
decode is essentially free by comparison (0.01s); effective steady-state is ~0.036 fps,
far from the config's own 60fps target. Session-length scaling stays flat (27-28s at
frame 1, 5, 10, and 15 alike) — a real, measured confirmation of Stage 4's dense-
attention-over-a-fixed-buffer finding, not just an assumption.

The upstream model card publishes real GPU numbers to compare against: **56 FPS**,
4-step unquantized, on a recommended RTX 5090 (72 FPS with w8a8 quantization) — about
**1,556x** faster than this bring-up's 0.036 FPS. Expected, not alarming: zero
performance work has been done here yet (no tracing, no kernel fusion, no batching
across sigma steps, no quantization) against a GPU vendor's own tuned reference stack.
See PORT_PLAN.md's benchmarking section for the full breakdown and caveats (including a
resolution/aspect-ratio difference between the two setups).

## License

Apache 2.0 (matching the upstream `Overworld/Waypoint-1.5-1B` weights' license terms —
see the weights repo for details).
