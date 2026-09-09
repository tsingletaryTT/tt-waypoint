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
| `waypoint_ttnn/tests/` | Hardware correctness tests, each checked against a real captured HF reference |
| `waypoint_ttnn/capture_*.py` | Scripts that capture reference activations from the real HF model (see [Reference activations](CLAUDE.md#reference-activations-arent-committed)) |
| `PORT_PLAN.md` | The staged bring-up plan |
| `BRINGUP_LOG.md` | Timestamped log: every stage, every bug, every hardware-verified number |

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
embeddings, attention + multi-frame KV cache, VAE encode+decode). Not yet done: the full
interactive generation loop and packaging with
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager) (Stage 6). See
BRINGUP_LOG.md for exact numbers and caveats — in particular, the 24-layer full-model
correlation (0.948) sits below the 0.99 target, diagnosed via a per-layer trace as
ordinary bf16 hardware compounding over a deep stack rather than a remaining logic bug.

Once Stage 6 lands, this will be packaged and pushed to Hugging Face under the `episod`
account, public, the same way [episod/tt-skyreels](https://huggingface.co/episod/tt-skyreels)
was.

## License

Apache 2.0 (matching the upstream `Overworld/Waypoint-1.5-1B` weights' license terms —
see the weights repo for details).
