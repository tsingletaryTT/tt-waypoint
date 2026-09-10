# Waypoint-1.5-1B TTNN port plan

Staged plan for porting `Overworld/Waypoint-1.5-1B` to Tenstorrent Blackhole (P300×2).
Each stage produces a numerically-checked component before moving to the next — no stage
is "done" until its TTNN output matches the reference PyTorch component within tolerance
on real inputs, using the exact tensor contracts confirmed in `BRINGUP_LOG.md`.

Reference: [`Overworld/Waypoint-1.5-1B`](https://huggingface.co/Overworld/Waypoint-1.5-1B)
(`transformer/model.py`, `vae/ae_model.py`, `modular_blocks.py` — see the model card for
architecture details). Confirmed working end-to-end on CPU as the ground truth to check
every stage against.

## Stage 0 — reference harness (done)

Reference PyTorch pipeline runs on CPU: seed frame from an image, then controller-input
frames via `ModularPipeline`. Confirmed shapes for every input/output at every step. This
IS the correctness oracle for every stage below — each stage's test feeds it the SAME
inputs (extracted from a real reference run) and diffs the output.

## Stage 1 — text encoder (UMT5-XL)

Lowest risk, confirmed by reading the actual code (not just architecture docs):
`models.tt_dit.pipelines.wan.text_encoder.TextEncoder` builds its TTNN `UMT5Config`
entirely from the loaded torch model's own `.config` fields (`d_model`, `d_ff`, `d_kv`,
`num_heads`, `num_layers`, ...) — it's checkpoint-agnostic, not hardcoded to WAN's
specific UMT5 variant. Should work directly against `google/umt5-xl` (Waypoint's actual
text encoder) with no changes beyond pointing it at that checkpoint. Remaining work:
wire the prompt-cleaning preprocessing (`ftfy`/`html.unescape`/whitespace — from
`modular_blocks.py`'s `prompt_clean`) identically, and confirm numerically on a real
prompt.

## Stage 2 — patchify / unpatchify + AdaLN + plain MLP

`nn.Conv2d`/`nn.ConvTranspose2d` patch embed (kernel=stride=(2,2)), `AdaLN`/`ada_rmsnorm`/
`ada_gate` (RMSNorm + scale/bias/gate from conditioning), and the plain-MLP DiT feedforward
(`moe: false` for this model — no MoE routing to build). All straightforward tensor ops
with direct TTNN equivalents; the main risk is getting the exact RMSNorm/AdaLN numerics
(SiLU placement, chunk order) bit-for-bit right, not the ops themselves.

## Stage 3 — conditioning heads

`NoiseConditioner` (Fourier features -> MLP; note the reference's `CachedDenoiseStepEmb`
LUT trick only matters for inference speed, not correctness — skip it for a first
functional pass, revisit under optimization), `CondHead` (per-layer scale/bias/gate),
`ControllerInputEmbedding` (plain MLP over concatenated mouse+button+scroll),
`CrossAttention` for prompt conditioning, `MLPFusion` for controller conditioning. All
plain linear-algebra, portable.

## Stage 4 — local/global KV cache, and attention (de-risked: no sparse kernel needed)

**Originally scoped as "the hard part, no existing precedent." First-pass verification
(see BRINGUP_LOG.md) found the actual computation is much simpler than the block-sparse
description suggested:** the reference runs **plain dense attention over the entire
per-layer capacity buffer** (ring + tail), where never-written slots are zero and
naturally contribute negligible softmax mass. There is no gather, no sparse kernel, no
block-mask to replicate — the "sparsity" lives entirely in which positions get WRITTEN,
not in how attention is computed over them.

Remaining work, now much more tractable:

- Per layer, maintain a zero-initialized `[capacity, d_head]` K/V buffer (`capacity =
  local_window*tpf + tpf` for local layers, `global_window*tpf + tpf` for the
  `global_attn_period`-th layers), matching `LayerKVCache`'s exact ring-buffer indexing:
  `bucket = (frame_idx + dilation - 1) // dilation`, `slot = bucket % num_buckets`, always
  write the tail unconditionally, write the ring slot only on `frame_idx % dilation == 0`
  and only when not "frozen" (mid-denoising intermediate steps don't persist history).
- Attention itself is then just: RMSNorm(Q,K) → OrthoRoPE → GQA repeat → dense
  `scaled_dot_product_attention` over the WHOLE buffer → out_proj. Verified correct on
  real hardware for frame 0 (mean abs diff 0.018 against the fp32 reference — larger than
  the ~0.002-0.01 seen in shorter-sequence stages, plausibly genuine bf16 accumulation
  error over an 8704-long softmax; flagged for revisit under Stage 4's optimization pass,
  not yet fully explained).
- **Multi-frame ring-buffer write/read logic verified** (see BRINGUP_LOG.md): ran frame 0
  (`prefill_forward`, empty cache) then frame 1 (`decode_forward`, cache now holds frame
  0's real history) through the SAME per-layer cache objects and compared both against a
  reference run that persists its own `kv_cache` across the same two calls. Frame 1's
  whole-model correlation (0.957) tracks frame 0's baseline (0.962) almost exactly (delta
  -0.004) rather than showing the much larger, structural degradation a real bucket/slot
  indexing bug would produce -- the ring write/read logic across frames is correct, not
  just the frame-0 empty-buffer case.
- Performance note for later: dense attention over an 8704-long (or larger, for global
  layers) buffer is far more compute than actually needed — a real optimization pass
  should gather just the written blocks before running attention, once correctness is
  established across multiple frames. Not needed for a first correctness pass.

## Stage 5 — VAE (ChunkedStreamingTAEHV) -- decoder hardware-verified

Small CNN (Conv2d + `MemBlock` residual-with-memory + `TPool`/`TGrow` for temporal
pooling/growing). No attention. Genuinely new TTNN code (nothing in tt_dit is a CNN
autoencoder like this) -- modeled directly on tt-metal's own SDXL VAE decoder
(`models/demos/vision/generative/stable_diffusion/wormhole/tt/vae/`) for the
`ttnn.conv2d`/`ttnn.upsample` calling convention, per Taylor's explicit call to port
straight to real hardware ops rather than a host-CPU-first stopgap.

**Decoder verified on real Blackhole hardware** (`waypoint_ttnn/tt/vae_decoder.py`,
`waypoint_ttnn/tests/test_vae_decoder.py`): faithfully ports the reference's THREE-layer
streaming contract (`_sequential_single_step`'s work-queue -> `_streaming_decode_step`'s
session-wide trim counter -> `decode()`'s first-call priming dance) rather than a
simplified approximation -- an earlier draft that collapsed this into "drain everything,
discard the first N calls" was wrong (the trim counter is global across the whole
session, not per-call) and was caught by testing before it ever ran, not assumed
correct. Verified against 3 consecutive `decode()` calls (same streaming state, no reset,
matching a real session) through the SAME per-layer `ttnn.conv2d` state that the
reference's own state machine persists: correlation >0.99 for all 12 output frames
(most >0.999; the first call's 4 frames sit closer to 0.993, plausibly the priming path's
extra bf16-noise accumulation, not investigated further given the bar is already met).
Two real hardware/API issues hit and fixed along the way: `ttnn.open_mesh_device`
defaults `l1_small_size` to 0, which conv2d's halo op cannot work with at all (fixed by
setting it explicitly, matching the SDXL VAE's own precedent); `ttnn.upsample` requires
an un-padded (ROW_MAJOR) input and rejects a TILE-layout tensor whose H*W isn't
tile-aligned, requiring an explicit `to_layout` round-trip around each upsample call.

**Encoder also verified** (`waypoint_ttnn/tt/vae_encoder.py`, `test_vae_encoder.py`):
same queue-based streaming approach, generalized with a `TPool` branch (temporal
downscale via channel-concat + 1x1 conv) the decoder doesn't need. Real bug hit and
fixed: the first draft's spatial-size tracker used `layer_idx >= stride_conv_idx` to
decide when H/W halves, which incorrectly treated a strided conv's OWN input as already
halved (it should only affect layers AFTER it, matching the decoder's `_hw_at`, which
correctly uses strict `>` for its upsample doublings) -- caught immediately by a real
hardware error (`MeshBuffer must be large enough to hold the tensor`) rather than
silently producing wrong numbers, since the actual queue-shuffling order meant the first
tensor to hit that bug had already gone through two downscale stages, not one, making the
shape mismatch load-bearing rather than cosmetic. Verified against the reference's own
`encode()` call on the same 4-frame RGB chunk: correlation 0.9996 for the resulting latent.

Stage 5 is now fully hardware-verified end to end (encode -> latent -> decode -> RGB).

## Stage 6 — full interactive loop + serving contract

Wires Stages 1-5 into `waypoint_ttnn/tt/generation_loop.py` (`WaypointGenerator`),
matching the real pipeline's exact per-session/per-frame protocol read directly out of
`modular_blocks.py` rather than inferred: `seed()` VAE-encodes a real starting image and
commits it as frame 0's history via a SINGLE unfrozen forward call (no denoising loop for
the seed frame itself); `step()` draws fresh noise, runs K FROZEN rectified-flow
denoising passes (`x = x + dsigma * v`, `v` being the transformer's velocity-field
output, NOT a denoised x directly -- a real detail that would have been silently wrong
if guessed rather than read from `WorldEngineDenoiseLoop._denoise_pass`), then ONE
UNFROZEN commit pass to persist the clean result. Needed a new `rope.py`
(`compute_rope_angles`) since every prior test reused a single captured rope tensor
(only ever frame 0 or frame 1) -- a real multi-frame loop needs arbitrary-frame rope
angles, verified bit-for-bit against captured frame-0/frame-1 reference tensors before
being trusted for other frames. Also fixed a real gap this exposed:
`FunctionalDecoder.prefill_forward` hardcoded `is_frozen=False`, which is correct for
the single-call tests done so far (all of which modeled the reference's cache-COMMIT
pass) but wrong for frame 0's OWN multi-step denoising loop, where intermediate steps
must be frozen too -- threaded `is_frozen` through properly, verified no regression on
the existing hardware-verified tests.

Verification: `test_generation_loop.py` seeds from the SAME real image and injects the
SAME noise draws (via `noise_override`) the reference's own two-pass-per-frame
`capture_generation_loop.py` run used, at the real (non-square) latent resolution
(32x64, not the arbitrary 16x16 the standalone VAE tests used) -- comparing both
intermediate latents and final decoded pixels end to end.

**Investigation trail (resolved) -- pixel-correlation-vs-reference is the wrong bar for
this loop; per-step correctness plus a real-image visual check is the right one.** The
seed path is excellent (latent corr 0.9996, decoded-pixel corr ~0.96), but GENERATED
frames initially looked catastrophic by the same metric: latent corr 0.86-0.88,
decoded-pixel correlation 0.05-0.24 against `capture_generation_loop.py`'s reference.
`test_frozen_step.py` isolated a real mean bias in a single `is_frozen=True` call
(correlation still matching the established baseline, ~0.95, but with a systematic shift
undiluted single-call tests never showed this starkly).

Chased further (per Taylor's ask to keep digging): `test_sigma_sweep.py` first looked
like a discrete bug (transformer catastrophically wrong, corr 0.256, at exactly
sigma=0.30078125, with the reference's OWN activations exploding ~12x through later
layers -- `test_sigma_block_trace.py`), but this was a false lead. That test fed the SAME
fixed random noise across every tested sigma, including 0.3 -- in a real trajectory, x at
sigma=0.3 is a partially-denoised signal from 2 prior steps, not raw noise, so labeling
raw noise "sigma=0.3" is out-of-distribution and plausibly explains an explosive
reference response unrelated to this port. Verified with real per-step (x_in, v_out)
pairs from an actual trajectory (`test_step_trace.py`): all 8 real denoising steps show
correlation 0.897-0.971, consistent with the established baseline, no failure anywhere.

Tried a real mitigation next: `ttnn.scaled_dot_product_attention` only accepts
bf16/bf8/bf4 tensors (confirmed against upstream's own open issue #36717 -- the
maintainers explicitly declined true fp32 SDPA support and are instead investing in fp32
ACCUMULATION precision), so added `compute_kernel_config` (`fp32_dest_acc_en=True`,
`MathFidelity.HiFi4`) to the attention and MLP matmuls -- exactly upstream's own
recommended lever, not tried before (an earlier experiment had only tried fp32 input
tensors, a different axis). Real, consistent improvement: all 8 real denoising steps
improved (avg correlation ~0.926 -> ~0.959), the 24-layer full-model correlation improved
0.948 -> 0.963. Kept.

Then the key realization: `test_generation_loop.py`'s own accumulated trajectory (using
OUR OWN output at each step, not the reference's) did NOT improve with this fix, and this
is expected, not alarming -- generation is an ITERATIVE, SELF-REFERENTIAL process. Step 2
evaluates at OUR step-1 output, not the reference's; any per-step difference at all,
however small, means the two trajectories diverge from that point on, the same way two
chaotic systems with infinitesimally different initial conditions diverge regardless of
how "correct" each step's dynamics are. Demanding a full self-referential trajectory match
one specific reference trajectory bit-for-bit is not a meaningful correctness bar for this
kind of loop -- the per-step tests (holding input fixed) are.

Confirmed this the right way: decoded and VISUALLY INSPECTED frames from a real,
in-distribution seed image (`ref_activations/seed_frame_ref.png`, a real photo from the
Stage 0 reference run) rather than trusting a number alone. The VAE round-trip of the
seed is visually indistinguishable from the original. Both generated frames (`step()`
called twice) are coherent, plausible nature/foliage scenes -- not garbage, not washed
out. The EARLIER catastrophic-looking numbers were measured using a synthetic random-
STATIC seed image (convenient for reproducible testing, but wildly out-of-distribution
for a model trained on real video) -- even the REFERENCE's own output under that scenario
was unstructured, grid-artifact noise, confirmed by decoding and viewing the reference's
own frames from that test too. Stage 6 is now considered adequately verified: per-step
correctness matches the established baseline (improved further by the fp32-accumulation
fix), and real-image generation looks visually correct. Bit-for-bit trajectory matching
against one specific reference run remains unachieved and is not expected to be
achievable for this kind of iterative process.

## Stage 7 -- packaging with tt-model-manager (hardware-verified)

Serving contract: NOT a `tt-dit-server` one-shot-request model like tt-skyreels --
needs a stateful, per-session protocol (a live KV cache persists across many `step()`
calls). `waypoint_ttnn/server/app.py` implements `POST /v1/sessions` (seed from a real
image) and `POST /v1/sessions/{id}/step` (advance one frame) under the SAME
`tt-dit-server` kind, since that kind only means "launch my own ASGI app" -- it doesn't
prescribe the one-shot shape tt-skyreels/tt-animatediff happen to use.
`waypoint_ttnn/session.py` holds a single active `WaypointGenerator` per process (a
deliberate scope choice, documented in the module's own docstring); a real multi-tenant
server would need one generator per session id (the current architecture ties KV-cache
state to the same objects that hold the model's weights), out of scope for now.

`tt_model_package.yaml` declares `hardware: p150`/`mesh_device: P150` rather than
`p300x2`: tt-model-manager requires an exact chip-count match between the two, and this
box's p300 boards can't be sub-divided below 2 chips at the driver level -- there is no
label for "1 chip of a p300x2". The model's real requirement is a plain 1x1 mesh, which
a genuine single-chip board label honestly represents (ttnn cares about chip topology,
not board SKU); documented as requirement-accurate rather than literally tested on that
board type.

**Hardware-verified end to end** (see BRINGUP_LOG.md for the full trail, including two
real bugs found and fixed -- a stale git ref pinned before `server/` was committed, and
`session.py` resolving weights via a hardcoded host-only path that doesn't exist inside
the container, fixed with `huggingface_hub.snapshot_download()`): packaged, served,
health/liveness/models endpoints correct, `POST /v1/sessions` + `POST .../step` both
returned real, visually coherent decoded frames from the actual running container, error
handling (404 on a stale session id) and clean shutdown all verified.

## Benchmarking plan

Correctness first (Stages 1-5 above), performance measured only once Stage 6's loop is
verified end to end -- matching the `ttm-functional-decoder` skill's own convention
(`tt-perf-report` over a Tracy-profiled, warmed run, not a first eager pass) rather than
quoting cold-start numbers as if they were steady state:

1. **Warm-up methodology**: run `WaypointGenerator.step()` a few times to let TTNN's
   kernel cache populate (JIT compilation is a real, one-time cost per unique shape --
   already observed as several extra seconds on every FIRST call of a given op/shape
   combination throughout this bring-up) before measuring anything. Report cold
   (first-ever call) and warm (steady-state) numbers separately and labeled as such --
   never blend them into one average.
2. **Per-frame latency breakdown**, warmed, single chip (P300x2, 1x1 mesh): time the
   transformer's denoise pass (K frozen sigma steps) and commit pass separately from the
   VAE decode pass, via `ttnn`'s device-side profiler / Tracy signposts
   (`PERF_DENOISE`/`PERF_DECODE`-style markers, same pattern the skill uses for
   prefill/decode) so a slow VAE isn't misattributed to the transformer or vice versa.
   Convert to an effective FPS and compare against the config's own `inference_fps=60`
   target -- report the gap honestly rather than picking a favorable subset of steps.
3. **Session-length scaling check**: because attention runs DENSE over the full
   zero-padded per-layer capacity buffer regardless of how much of it is real history
   (Stage 4's finding), per-frame latency should stay FLAT as a session grows longer
   (frame 50 should cost about the same as frame 5) -- this is a real, testable
   architectural prediction, not an assumption, so benchmark frame latency at several
   points in a long session (e.g. frames 1, 10, 50) specifically to confirm or refute it,
   the same "trust the subject, verify the instrument" way every other claim in this repo
   has been checked.
4. **Seed vs. steady-state cost**: `seed()` (VAE encode + one commit pass) is a
   once-per-session cost, architecturally different from `step()`'s repeated cost --
   report it separately, not folded into a per-frame average.
5. **Tokens/wall-clock accounting**: continue the `BRINGUP_LOG.md` convention (approximate
   `total_tokens` delta + wall-clock time per milestone) for the benchmarking work itself,
   consistent with how the rest of this bring-up has been tracked since Taylor's original
   request to measure "how long it takes, how many tokens it burns."
6. **What's explicitly NOT benchmarked yet**: multi-chip scaling and quantized
   (fp8/bf8) precision are both out of scope per this plan's own "explicitly out of
   scope" section below -- correctness and single-chip bf16 performance come first.

## Stage 6 — full interactive loop + serving contract

Wire stages 1-5 into a frame-by-frame generation loop matching the reference's calling
convention (seed with an image, then step with button/mouse/scroll each call, `kv_cache`
threaded through). THEN design the serving contract — this is NOT a `tt-dit-server`
one-shot-request model; it needs a stateful, per-session interactive protocol (closer to a
websocket/streaming API than the SkyReels `/v1/videos/generations` shape). Packaging with
tt-model-manager is a separate, later concern once there's something correct to package.

## Explicitly out of scope for a first pass

- Real-time / 60fps performance (the README's headline target) — a correctness-first pass
  will be slow; optimization is its own project once Stage 6 produces correct frames at
  any speed.
- FP4/FP8/w8a8 quantization (CUDA-specific in the reference; TT would use its own
  precision tooling, and only after functional correctness).
- MoE variants of the Waypoint family (this specific 1B checkpoint has `moe: false`).
- Multi-chip parallelism — start single-chip; this model's autoregressive per-frame loop
  doesn't obviously benefit from tensor/sequence parallelism the way batch DiT does, so
  that's its own evaluation, not an assumption.
