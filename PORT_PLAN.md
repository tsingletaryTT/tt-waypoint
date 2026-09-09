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

## Stage 4 — the hard part: local/global block-sparse causal attention + KV cache

**This is where the real risk and the majority of the effort lives.** No existing tt-metal
kernel implements this pattern. Needs its own design spike before writing code:

- Reference semantics (from `modular_blocks.py`'s `LayerKVCache`/`StaticKVCache`): each
  layer gets EITHER a small sliding local window (16 frames = 8192 tokens) OR, every
  `global_attn_period` (4) layers, a much larger window (128 frames) that only keeps every
  `global_pinned_dilation`-th (8) frame — a dilated long-context memory. Each layer's cache
  is an independent fixed-capacity ring buffer; the full cache set across all 24 layers IS
  the interactive "world state" (explicitly save/restorable).
- Options to evaluate: (a) two attention "modes" composed from existing TTNN sliding-window
  + full-attention primitives (if `tt_transformers` has a sliding-window/landmark-attention
  precedent worth checking first) with a strided gather for the dilated global cache: (b) a
  custom kernel via `ttnn.generic_op` + `KernelDescriptor` (see the "custom kernels via
  generic_op" convention already established for other TT-Lang-adjacent work on this box).
- Whichever approach: validate against the reference's ring-buffer/bucket indexing
  (`bucket = (frame_idx + dilation - 1) // dilation`, `slot = bucket % num_buckets`) on a
  short multi-frame sequence before trusting it on the full 512-frame context.

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
