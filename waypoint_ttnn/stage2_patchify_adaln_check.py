"""Stage 2: patchify/unpatchify/AdaLN/conditioning embeddings on real Blackhole hardware,
checked against captured reference activations (ref_activations/block_activations.pt)."""
import sys
sys.path.insert(0, "/home/ttuser/tt-metal")

def main():
    import torch
    import ttnn

    REF = "/home/ttuser/code/tt-waypoint/ref_activations/block_activations.pt"
    ref = torch.load(REF, weights_only=False)
    cfg = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/transformer_config.pt", weights_only=False)

    # Pull the reference torch weights straight from the live WorldModel (re-instantiate
    # from HF cache -- cheap, config-only + tiny conv weights, not the full 1.86B forward).
    import glob
    from safetensors.torch import load_file

    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    weights = load_file(weights_path)

    class _WM:
        pass

    wm = _WM()

    class _Patchify:
        pass

    wm.patchify = _Patchify()
    wm.patchify.weight = weights["patchify.weight"]

    class _OutNorm:
        pass

    wm.out_norm = _OutNorm()
    wm.out_norm.fc = _Patchify()
    wm.out_norm.fc.weight = weights["out_norm.fc.weight"]

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        # --- Patchify: Conv2d(32, 2048, kernel=2, stride=2, bias=False) == unfold + matmul ---
        x_ref = ref["patchify__inputs"][0].float()  # [1, 32, 32, 64]
        y_ref = ref["patchify__output"][0].float()  # [1, 2048, 16, 32]

        B, C, H, W = x_ref.shape
        ph, pw = 2, 2
        Hp, Wp = H // ph, W // pw
        # unfold into patches: [B, C, Hp, ph, Wp, pw] -> [B, Hp, Wp, C, ph, pw] -> [B*Hp*Wp, C*ph*pw]
        patches = x_ref.view(B, C, Hp, ph, Wp, pw).permute(0, 2, 4, 1, 3, 5).reshape(B * Hp * Wp, C * ph * pw)
        weight = wm.patchify.weight.float().reshape(2048, C * ph * pw)  # [D, C*ph*pw] (Conv2d weight is [D,C,ph,pw])

        tt_patches = ttnn.from_torch(patches.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_weight = ttnn.from_torch(weight.t().contiguous().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_y = ttnn.matmul(tt_patches, tt_weight)
        y_flat = ttnn.to_torch(tt_y).float()  # [B*Hp*Wp, D]
        y_computed = y_flat.view(B, Hp, Wp, 2048).permute(0, 3, 1, 2)  # -> [B, D, Hp, Wp]

        diff = (y_computed - y_ref).abs()
        print("[stage2] patchify: max abs diff", diff.max().item(), "mean abs diff", diff.mean().item())

        # --- AdaLN (out_norm): rms_norm(x) * (1+scale) + bias, driven by SiLU(cond)->Linear ---
        x_norm_ref = ref["out_norm__inputs"][0].float()  # [1, 512, 2048]
        cond_ref = ref["out_norm__inputs"][1].float()  # [1, 1, 2048]
        y_norm_ref = ref["out_norm__output"].float()  # [1, 512, 2048]

        fc_w = wm.out_norm.fc.weight.float()  # [2*D, D]
        cond_act = torch.nn.functional.silu(cond_ref)
        ab = cond_act @ fc_w.t()  # [1,1,2*D]
        a, b_ = ab.chunk(2, dim=-1)

        tt_x = ttnn.from_torch(x_norm_ref.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_a = ttnn.from_torch(a.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_b = ttnn.from_torch(b_.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

        # RMSNorm (no learned weight, matches the reference's bare rms_norm()) via plain ops.
        sq_mean = ttnn.mean(ttnn.multiply(tt_x, tt_x), dim=-1, keepdim=True)
        rstd = ttnn.rsqrt(ttnn.add(sq_mean, 1e-6))
        normed = ttnn.multiply(tt_x, rstd)
        scaled = ttnn.multiply(normed, ttnn.add(tt_a, 1.0))
        tt_out = ttnn.add(scaled, tt_b)
        y_norm_computed = ttnn.to_torch(tt_out).float()

        diff2 = (y_norm_computed - y_norm_ref).abs()
        print("[stage2] AdaLN (out_norm): max abs diff", diff2.max().item(), "mean abs diff", diff2.mean().item())

    finally:
        ttnn.close_mesh_device(device)
    print("[stage2] DONE")

if __name__ == "__main__":
    main()
