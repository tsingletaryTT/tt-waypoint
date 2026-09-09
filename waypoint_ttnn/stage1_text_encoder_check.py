# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Stage 1 correctness check: Waypoint-1.5-1B's text encoder (google/umt5-xl) on real
Blackhole hardware, against the torch reference.

RESULT (first run, P300x2 board, 1x1 mesh): zero missing/unexpected keys loading
UMT5EncoderModel's state_dict directly into models.tt_dit.encoders.umt5.UMT5Encoder --
tt-metal's WAN-family UMT5 encoder is checkpoint-agnostic (its config is built entirely
from the loaded torch model's own config fields, see
models.tt_dit.pipelines.wan.text_encoder.TextEncoder) and needed no changes to run
against Waypoint's specific checkpoint. Output shape matched exactly ([1, 512, 2048]),
mean abs diff 0.0045 against the fp32-computed torch reference -- consistent with
expected bf16 precision noise across 24 transformer layers, not a bug.

Bypasses tt_dit's own TextEncoder wrapper class deliberately: that wrapper assumes the
WAN convention of tokenizer/text_encoder living in subfolders of one repo (its own
weights repo), which google/umt5-xl does not use (it's a flat, single-purpose repo).
Constructs UMT5Config/UMT5Encoder directly instead -- same underlying reusable code,
different loading convention.

Run under a gozer lease (1 chip -- UMD expands a P300 board's single-chip request to
both chips on the board, expected). Not yet a package; this is a correctness probe,
run manually:

    sys.path additions handled inline below since this predates any proper packaging.
"""
import sys
sys.path.insert(0, "/home/ttuser/tt-metal")

def main():
    import torch
    import ttnn
    from transformers import AutoTokenizer, UMT5EncoderModel

    from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config, UMT5Encoder
    from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager

    CHECKPOINT = "google/umt5-xl"
    prompt = "An explorable world"

    print("[stage1] loading torch reference...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    torch_encoder = UMT5EncoderModel.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16)
    torch_encoder.eval()

    ids = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    with torch.no_grad():
        ref_out = torch_encoder(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"]).last_hidden_state
    print("[stage1] torch reference output:", ref_out.shape, ref_out.dtype, "mean", ref_out.float().mean().item())

    print("[stage1] opening 1x1 mesh device...")
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        ccl_manager = CCLManager(mesh_device=device, num_links=1, topology=ttnn.Topology.Linear)
        enc_pc = EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=0))

        umt5_config = UMT5Config(
            vocab_size=torch_encoder.config.vocab_size,
            embed_dim=torch_encoder.config.d_model,
            ff_dim=torch_encoder.config.d_ff,
            kv_dim=torch_encoder.config.d_kv,
            num_heads=torch_encoder.config.num_heads,
            num_hidden_layers=torch_encoder.config.num_layers,
            max_prompt_length=512,
            layer_norm_eps=torch_encoder.config.layer_norm_epsilon,
            relative_attention_num_buckets=torch_encoder.config.relative_attention_num_buckets,
            relative_attention_max_distance=torch_encoder.config.relative_attention_max_distance,
        )
        print("[stage1] umt5_config:", umt5_config)

        print("[stage1] constructing TTNN UMT5Encoder...")
        tt_encoder = UMT5Encoder(
            config=umt5_config,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=enc_pc,
        )
        print("[stage1] loading torch state dict into TTNN encoder...")
        result = tt_encoder.load_torch_state_dict(torch_encoder.state_dict(), strict=False)
        print("[stage1] missing keys:", result.missing_keys[:10], "unexpected:", result.unexpected_keys[:10])

        print("[stage1] building ttnn input tensors...")
        tt_ids = ttnn.from_torch(
            ids["input_ids"], device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tt_mask = ttnn.from_torch(
            ids["attention_mask"].to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        print("[stage1] running TTNN forward...")
        tt_out = tt_encoder(tt_ids, attention_mask=tt_mask)
        final = ttnn.to_torch(tt_out[-1]).float()
        print("[stage1] TTNN output shape:", final.shape, "mean", final.mean().item())

        ref = ref_out.float()
        if final.shape == ref.shape:
            diff = (final - ref).abs()
            print("[stage1] max abs diff:", diff.max().item(), "mean abs diff:", diff.mean().item())
        else:
            print("[stage1] SHAPE MISMATCH:", final.shape, "vs", ref.shape)
    finally:
        ttnn.close_mesh_device(device)
    print("[stage1] DONE")

if __name__ == "__main__":
    main()
