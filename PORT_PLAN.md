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
- Still unverified: the ring-buffer write/indexing logic across MULTIPLE frames (this
  first pass only proved frame 0's empty-buffer case). Test frame 1+ next, where the local
  ring actually starts holding real history and the global (dilated) layers' write_step
  logic first kicks in.
- Performance note for later: dense attention over an 8704-long (or larger, for global
  layers) buffer is far more compute than actually needed — a real optimization pass
  should gather just the written blocks before running attention, once correctness is
  established across multiple frames. Not needed for a first correctness pass.

## Stage 5 — VAE (ChunkedStreamingTAEHV)

Small CNN (Conv2d + `MemBlock` residual-with-memory + `TPool`/`TGrow` for temporal
pooling/growing). No attention. Lower risk than Stage 4, but genuinely new TTNN code
(nothing in tt_dit is a CNN autoencoder like this) — mainly `ttnn.conv2d` composition plus
getting the streaming state machine (`_sequential_single_step`'s work-queue) right.

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
