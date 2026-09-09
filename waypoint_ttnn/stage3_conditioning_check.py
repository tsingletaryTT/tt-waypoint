"""Stage 3: NoiseConditioner (Fourier features -> MLP) and ControllerInputEmbedding
(plain MLP) on real Blackhole hardware, checked against captured reference activations."""
import sys
sys.path.insert(0, "/home/ttuser/tt-metal")

def main():
    import glob
    import math
    import torch
    import ttnn
    from safetensors.torch import load_file

    REF = "/home/ttuser/code/tt-waypoint/ref_activations/block_activations.pt"
    ref = torch.load(REF, weights_only=False)

    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    W = load_file(weights_path)

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        # --- NoiseConditioner ---
        sigma = ref["denoise_step_emb__inputs"][0].float()  # [1,1]
        y_ref = ref["denoise_step_emb__output"].float()  # [1,1,2048]

        fourier_dim = 512
        half = fourier_dim // 2
        freq = torch.logspace(0, -1, steps=half, base=10_000.0, dtype=torch.float32)
        s = sigma.reshape(-1).float() * 1000
        phase = s[:, None] * freq[None, :]
        emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1) * (2 ** 0.5)  # [1, 512]

        fc1_w = W["denoise_step_emb.mlp.fc1.weight"].float()  # [8192, 512]
        fc2_w = W["denoise_step_emb.mlp.fc2.weight"].float()  # [2048, 8192]

        tt_emb = ttnn.from_torch(emb.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_fc1 = ttnn.from_torch(fc1_w.t().contiguous().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_fc2 = ttnn.from_torch(fc2_w.t().contiguous().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

        h = ttnn.matmul(tt_emb, tt_fc1)
        h = ttnn.silu(h)
        out = ttnn.matmul(h, tt_fc2)
        y_computed = ttnn.to_torch(out).float().view(1, 1, -1)

        diff = (y_computed - y_ref).abs()
        print("[stage3] NoiseConditioner: max abs diff", diff.max().item(), "mean abs diff", diff.mean().item())

        # --- ControllerInputEmbedding ---
        mouse, button, scroll = [x.float() for x in ref["ctrl_emb__inputs"]]
        y_ref2 = ref["ctrl_emb__output"].float()

        x_cat = torch.cat((mouse, button, scroll), dim=-1)  # [1,1,259]
        fc1_w2 = W["ctrl_emb.mlp.fc1.weight"].float()
        fc2_w2 = W["ctrl_emb.mlp.fc2.weight"].float()

        tt_x = ttnn.from_torch(x_cat.to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_fc1b = ttnn.from_torch(fc1_w2.t().contiguous().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)
        tt_fc2b = ttnn.from_torch(fc2_w2.t().contiguous().to(torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT)

        h2 = ttnn.matmul(tt_x, tt_fc1b)
        h2 = ttnn.silu(h2)
        out2 = ttnn.matmul(h2, tt_fc2b)
        y2_computed = ttnn.to_torch(out2).float()

        diff2 = (y2_computed - y_ref2).abs()
        print("[stage3] ControllerInputEmbedding: max abs diff", diff2.max().item(), "mean abs diff", diff2.mean().item())
    finally:
        ttnn.close_mesh_device(device)
    print("[stage3] DONE")

if __name__ == "__main__":
    main()
