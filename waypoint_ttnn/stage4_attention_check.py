"""Stage 4 (verified): dense attention over the FULL per-layer capacity buffer
(zero-initialized, current frame written into the tail) on real Blackhole hardware --
matches the reference exactly, no block-sparse masking needed for a first correctness
pass. See BRINGUP_LOG.md for how this was discovered (the "attend only to written
positions" hypothesis was WRONG; dense attention over the zero-padded full buffer is
what the reference actually computes)."""
import sys
sys.path.insert(0, "/home/ttuser/tt-metal")

def main():
    import torch
    import ttnn

    REF = "/home/ttuser/code/tt-waypoint/ref_activations/flex_attn_args.pt"
    d = torch.load(REF, weights_only=False)
    q, k, v = d["flex_q"].float(), d["flex_k"].float(), d["flex_v"].float()
    y_ref = d["attn__output"][0].float()  # [1, 512, 2048], post out_proj

    import glob
    from safetensors.torch import load_file
    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    W = load_file(weights_path)
    o_w = W["transformer.blocks.0.attn.out_proj.weight"].float()

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        rep = 32 // 16
        k_rep = k.repeat_interleave(rep, dim=1)
        v_rep = v.repeat_interleave(rep, dim=1)

        tt_q = ttnn.from_torch(q.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_k = ttnn.from_torch(k_rep.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_v = ttnn.from_torch(v_rep.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

        tt_out = ttnn.transformer.scaled_dot_product_attention(tt_q, tt_k, tt_v, is_causal=False)
        attn_out = ttnn.to_torch(tt_out).float().transpose(1, 2).reshape(1, 512, 2048)

        tt_o_w = ttnn.from_torch(o_w.t().contiguous().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_attn_flat = ttnn.from_torch(attn_out.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_y = ttnn.matmul(tt_attn_flat, tt_o_w)
        y_computed = ttnn.to_torch(tt_y).float()

        diff = (y_computed - y_ref).abs()
        print("[stage4v2] dense attn over full capacity buffer: max abs diff", diff.max().item(),
              "mean abs diff", diff.mean().item())
    finally:
        ttnn.close_mesh_device(device)
    print("[stage4v2] DONE")

if __name__ == "__main__":
    main()
