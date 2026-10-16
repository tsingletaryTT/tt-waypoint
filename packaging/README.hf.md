---
tags:
- blackhole
- p150
- thin
- tt-dit-server
- tt-model-cache
- world-model
- interactive-video
license: apache-2.0
pipeline_tag: image-to-video
base_model:
- Overworld/Waypoint-1.5-1B
---

# tt-waypoint

[Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B) is Overworld's interactive "world model". It is an autoregressive causal diffusion transformer that generates video one frame at a time, steered by mouse and scroll input.

This package runs it on Tenstorrent Blackhole via TTNN:

- **What it is:** a from-scratch bring-up. Nothing in tt-metal's `models.tt_dit` covers this architecture.
- **How you use it:** a stateful, session-based HTTP API served from a small ASGI server.
- **What it needs:** **one Blackhole chip** (profile `P150`, a 1x1 mesh). All hardware verification so far ran on one chip of a P300c board.
- **Maturity:** experimental. Frames are coherent, but it runs roughly 1,400x below real time, and no performance work has been done yet.

Packaged with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) as a v6 thin bundle (manifest schema 6). It is a pip/venv install, not a container image.

## At a glance

| | |
| --- | --- |
| Architecture | 24-layer autoregressive causal diffusion transformer (per-layer ring-buffer frame cache, 512 tokens per frame), plus a small CNN VAE (`ChunkedStreamingTAEHV`) |
| Hardware | 1 Blackhole chip (profile `p150`, mesh `P150`, 1x1). Verified on one chip of a P300c; never run on a physical P150 |
| Input / output | Seed image: one, resized to 512x1024. Output: one generated 512x1024 frame per step, 4 denoising steps per frame. No text prompt |
| License | Apache-2.0 (weights and port code) |
| Status | Experimental: correct end to end per step and per component, not performance-tuned |
| Model CI v0 | not yet run |

## Intended use

**Direct use:** interactive demos and research into world-model inference on Tenstorrent hardware. Seed a session with an image, then step it forward one frame at a time with directional (mouse) and zoom (scroll) controls.

**Out-of-scope use:**
- **Real-time or interactive-rate play.** It runs at about 0.04 FPS against the upstream 60 FPS target.
- **Multiple concurrent users.** There is one active session per server process.
- **Keyboard or button control.** The 256-wide button vector has no published semantics, so it is always sent as zeros.
- **Text-prompted generation.** This checkpoint has `prompt_conditioning: null`.
- **Multi-chip meshes.**

## Quickstart

```bash
uv tool install tenstorrent   # once: the Tenstorrent CLI, `tt`
tt model pull episod/tt-waypoint
tt serve episod/tt-waypoint
```

`tt model pull` downloads two things:

- **This bundle:** a small venv-install recipe. It is `install.sh`/`run.sh`, the model's own source under `extra_code/`, and one empty dependency wheel; `ttnn`, `torch` and the HTTP stack come from an index at install time.
- **The [`Overworld/Waypoint-1.5-1B`](https://huggingface.co/Overworld/Waypoint-1.5-1B) weights:** revision `391f928`, into the bundle's own HF cache. Weights come down by default, and `tt` has no `--with-weights` flag. The manifest lists the three files the server reads, so the pull fetches only those (about 7.5 GB) and skips the rest of the 11.4 GB upstream repo.

`serve` starts the model's own HTTP server on port 20000, or the next free port if 20000 is busy. It is ready when it logs `Application startup complete`.

Measured on a fresh install of this bundle (P300c QuietBox, 2026-09-27):

| step | time |
| --- | --- |
| `pull --with-weights` (venv plus weights) | 4 min 40 s |
| Launch to ready | 7 s |
| First `POST /v1/sessions` | about 39 s, which includes first-time kernel compile |
| Each later step | about 25 s |

Without tt-cli, tt-model alone does the whole job:

```bash
tt-model pull  episod/tt-waypoint --with-weights
tt-model serve episod/tt-waypoint
```

## Serve profiles

| profile | hardware | mesh | sessions per process |
| --- | --- | --- | --- |
| `p150` | 1 Blackhole chip. Verified on one chip of a P300c; a physical P150 has not been tested | `P150` (1x1) | 1 |

No multi-chip profile exists. `SUPPORTED_MESH_SHAPES` is `{(1, 1)}`, and the server refuses any other `WAYPOINT_MESH_SHAPE`.

## Using it

This is **not an OpenAI-compatible API**. `/v1/models` exists only as a stub for tooling. The real interface is a stateful session: seed once from an image, then step.

```bash
# Seed a session from a real starting image (base64-encoded PNG/JPEG; resized to 512x1024):
curl -s localhost:20000/v1/sessions -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$(base64 -w0 my_photo.png)\"}"
# -> {"session_id": "...", "frame_index": 1, "frame_b64": "<base64 PNG>"}

# Step the session forward one frame, steered by direction/zoom:
curl -s localhost:20000/v1/sessions/<session_id>/step \
  -H 'Content-Type: application/json' -d '{"direction": "forward", "zoom": 0.0}'
# -> {"session_id": "...", "frame_index": 2, "frame_b64": "<base64 PNG>"}

# End the session:
curl -s -X DELETE localhost:20000/v1/sessions/<session_id>
```

| endpoint | purpose |
| --- | --- |
| `POST /v1/sessions` | Takes `{"image_b64": str}`. Seeds a new session and returns the decoded seed frame. **Discards any session already in progress** |
| `POST /v1/sessions/{id}/step` | Takes `{"direction": str = "forward", "zoom": float = 0.0}`. Generates one frame |
| `DELETE /v1/sessions/{id}` | Ends the session |
| `GET /health`, `GET /tt-liveness` | Readiness and liveness. Both return `503` until the model is loaded |
| `GET /v1/models` | Stub listing, for tooling |

- **`direction`:** one of `forward`, `back`, `left`, `right`, `forward_left`, `forward_right`, `back_left`, `back_right`. These map to the model's `[dx, dy]` mouse input.
- **`zoom`:** the scroll scalar. Only its sign has a documented meaning.
- **Seeding:** the API has no RNG seed parameter. Per-step noise is drawn inside the server.
- **Errors:**
  - `503`: the model is still starting.
  - `404`: the session ID is unknown, or a newer `POST /v1/sessions` superseded it.
  - `409`: you stepped with no active session.
  - `400`: the request was bad.

The [GitHub repo](https://github.com/tsingletaryTT/tt-waypoint) also has a local Gradio UI (`app.py`), which is a pure HTTP client for this same server.

## Expected performance

| metric | value | source |
| --- | --- | --- |
| Per-step denoiser correctness vs HF reference (8 real denoising steps, correlation) | about 0.959 on average | `BRINGUP_LOG.md` (fp32-accumulation row; step-trace row). The range before the fp32-accumulation fix was 0.897-0.971 |
| 24-layer transformer, single forward pass (correlation vs reference) | 0.963 | `BRINGUP_LOG.md` (fp32-accumulation row) |
| Single decoder layer PCC (layer 0, frame-0 prefill) | 0.999 | `BRINGUP_LOG.md` (ttm-* skills row) |
| VAE encoder / decoder (correlation vs reference) | 0.9996 / > 0.99 on all 12 frames | `BRINGUP_LOG.md` (Stage 5 rows) |
| Generated frames from a real seed image | Visually coherent and plausible. Inspected by eye; no numeric perceptual metric | bring-up log; re-checked on both bundles below |
| **Warm latency per generated frame, installed bundle** (4 denoise steps + 1 cache commit) | **median 24.95 s** (min 24.64, max 25.35), N=15 | published bundle `1dd34b8`, 2026-09-27 |
| Same, re-verified on this revision of the bundle | median 25.72 s (min 25.70, max 25.91), N=3 | this bundle, 2026-09-27. +3% vs the row above |
| Effective throughput | 0.040 FPS | 1 / median |
| Latency vs session length (steps 1 / 5 / 10 / 15 / 16) | 25.44 / 25.05 / 24.79 / 24.95 / 24.64 s | flat, no drift over 16 frames |
| First step after a cold start | 25.44 s. The first `POST /v1/sessions` on a cold process takes 38-39 s | includes kernel compile |
| Dev build, for comparison (tt-metal v0.78.0 from source, in-process `benchmark.py`) | 27.41 s per frame | `PORT_PLAN.md` |
| Upstream GPU reference, for scale | 56 FPS on an RTX 5090, 4-step unquantized | upstream model card |

**Methodology:** on one Blackhole chip of a P300c (1x1 mesh, `ttnn==0.78.0` from PyPI, eager execution, batch 1, one session), a session seeded from `ref_activations/seed_frame_ref.png` was stepped with the default controls (`forward`, zoom 0), timing host HTTP wall-clock per `POST .../step` after one untimed warmup step: N=15 on the published bundle `1dd34b8` and N=3 on this bundle revision (weights `391f928`), both on 2026-09-27.

**How to read the accuracy numbers:**

- **Why a trajectory match is the wrong test.** This model is iterative and self-referential: each frame's denoising runs on this port's own previous output. Any per-step difference, including ordinary bf16 rounding, therefore makes the generated trajectory diverge from any single reference run, the way two chaotic systems do. Comparing generated-frame pixels against one reference trajectory gives correlations as low as 0.05-0.24. The bring-up investigation found that this is the wrong bar, not a defect.
- **What the claim rests on:** correctness of each individual step, correctness of each component, and visual inspection of real-image sessions.
- **What it does not claim:** a bit-exact trajectory match.

The full investigation is in [BRINGUP_LOG.md](https://github.com/tsingletaryTT/tt-waypoint/blob/main/BRINGUP_LOG.md).

## Limitations

- **Far from real time.** It takes about 25 s per frame (0.04 FPS) against the model's 60 FPS target. No tracing, kernel fusion, batching across sigma steps or quantization has been attempted.
- **One session per process.** The KV-cache state lives on the same objects as the loaded weights. Starting a new session silently ends the previous one, and a stale ID returns `404`.
- **Single chip only.** There is no multi-chip or tensor-parallel path. Every hardware run used one chip of a P300c board. The `p150` profile states the requirement (one chip); it has not been tested on a physical P150 board.
- **Mouse and scroll control only.** The 256-wide button input is always zero, because no control-scheme mapping has been published. `zoom` has sign-only semantics.
- **The 24-layer correlation (0.963) is below the 0.99 bar.** The gap was diagnosed as bf16 compounding over a deep residual stack, with no localized logic bug. The in-repo `test_full_model.py` assert is left failing on purpose rather than weakened.
- **Resolution is fixed at 512x1024.** Upstream's GPU numbers are at 720p, so the performance comparison above is not a controlled one.
- **The weights land twice.** `run.sh` sets `HF_HOME` to `<bundle>/.hf`, so a `pull --with-weights` into a different cache is fetched again at first serve. The manifest's file filter keeps each fetch to the three files the server reads.

## Risks and safety considerations

- **Drift over long sessions.** Per-step bf16 noise accumulates. How many frames stay usable has not been measured. In a 16-frame `forward` session from the reference seed:
  - The view settled into a close-up grass texture by frame 4.
  - It lost contrast by frame 16: pixel standard deviation 42-50 on frames 1-8, 14 on frame 16.
- **Out-of-distribution seeds give noise.** A synthetic or random-static seed image produces unstructured output. The upstream reference behaves the same way under that input. Seed with a real, natural image.
- **Numerical deviation from the reference.** ttnn SDPA has no fp32 mode. This port uses bf16 with fp32 accumulation (`fp32_dest_acc_en`, HiFi4). Outputs are plausible but will not match GPU output pixel for pixel.
- **Generated video is synthetic.** It is not a faithful simulation of any real place.

## Licensing

- **Weights:** [`Overworld/Waypoint-1.5-1B`](https://huggingface.co/Overworld/Waypoint-1.5-1B), Apache-2.0.
- **Port and serving code:** the `waypoint_ttnn` source under `extra_code/`, Apache-2.0 ([repo LICENSE](https://github.com/tsingletaryTT/tt-waypoint/blob/main/LICENSE)).
- **Vendored tt-metal code:** `waypoint_ttnn/tt/lightweightmodule.py` is `models/common/lightweightmodule.py` from tt-metal v0.78.0, copied unchanged. Apache-2.0.

## Changelog

| date | change |
| --- | --- |
| 2026-10-16 (staged, not yet published) | Repackaged in the canonical v6 thin shape. The model's code ships as source under `extra_code/` instead of a `waypoint_ttnn` wheel, and the `tt_waypoint_models_closure` wheel is gone (its one class is now part of the model's code). `waypoint-ttnn` 0.1.2. `constraints.txt` pins every package, and the manifest limits the weights pull to the three files the server reads. `ttnn==0.78.0` and the weights revision are unchanged. The performance numbers above were measured on the 0.1.1 wheel-based bundle; this shape has not been served on hardware yet. |
| 2026-09-27 | Repackaged with `waypoint-ttnn` 0.1.1. The server now loads weights at the pinned revision `391f928`, overridable with `TT_MODEL_WEIGHTS_REVISION`, and fetches only the three files it reads. The manifest records that revision and `tt_metal_version` 0.78.0. `run.sh` exports the revision and picks chips from the bundle's device count. `ttnn==0.78.0` is unchanged. Re-verified on hardware from a fresh install with an empty HF cache: 25.72 s per frame, coherent frames. |
| 2026-09-15 | Repackaged as a v6 thin bundle (pip/venv), replacing the v5.1 container image. The served code is unchanged. `ttnn==0.78.0` comes from PyPI, and `torch==2.14.0+cpu` is pinned. Re-verified on hardware: mesh open, fresh weight fetch, seed and step, coherent frames. |
| 2026-09-10 | Benchmarked: 27.41 s per frame, flat across session length. Re-verified by a fresh `pull` and `serve` of the published package across several directions. |
| 2026-09-09 | First publish (v5.1 container). The card's port was corrected to 20000. `session.py` now resolves weights via `snapshot_download` instead of a host-only path. |

## Related packages

None. This is the only Tenstorrent port of Waypoint-1.5-1B.

## Feedback

- **Questions or problems with this package:** open a discussion at `https://huggingface.co/episod/tt-waypoint/discussions`. That is the one channel that reaches the bundle's author.
- **Problems with the `tt` tooling itself:** run `tt report issue`. It collects your environment and opens a prefilled issue against tenstorrent/tt-cli; it does not reach this package's author.
- **Product feedback:** <support@tenstorrent.com>.

## Provenance

The bundle is staged by [`packaging/package-thin.sh`](https://github.com/tsingletaryTT/tt-waypoint/blob/main/packaging/package-thin.sh) in the GitHub repo. `PROVENANCE.json` in this bundle records the source commit, the SHA-256 of every shipped file, and the SHA-256 of `requirements.txt` and `constraints.txt`.

| part of the bundle | what it is |
| --- | --- |
| `extra_code/waypoint_ttnn/` | The served-path closure of [tsingletaryTT/tt-waypoint](https://github.com/tsingletaryTT/tt-waypoint): `session.py`, the ASGI app, and the `tt/` model classes. The list of files is computed from the app's imports. Bring-up scripts, tests and docs are not included |
| `wheels/tt_waypoint_serve_deps-*.whl` | An empty wheel (metadata only). It makes `install.sh` install `huggingface_hub` and `safetensors` in a later step, so they resolve from PyPI and not from the PyTorch CPU index |
| `requirements.txt`, `constraints.txt` | `ttnn==0.78.0` and CPU `torch` with the HTTP stack, and a pin for every package they pull in |

| component | built from |
| --- | --- |
| tt-metal / ttnn | `ttnn==0.78.0` (PyPI). The manifest `tt_metal_version` is `0.78.0` |
| model code | [tsingletaryTT/tt-waypoint](https://github.com/tsingletaryTT/tt-waypoint), commit recorded in `PROVENANCE.json` |
| weights | `Overworld/Waypoint-1.5-1B` @ `391f92827075edcf4a8b3c8a2ddae010698f8636`, pinned in the manifest and in `session.py` |
| Model CI v0 | not yet run |
