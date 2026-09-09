# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Localizes WHERE the sigma=0.30078125-specific catastrophic failure
(test_sigma_sweep.py: corr 0.256 vs 0.96 for a control sigma) originates, by comparing
our own blocks 0-3's outputs against the reference's for both the good control sigma
(0.75) and the bad one (0.30078125). If block 0 already diverges specifically for the
bad sigma, the bug is in something sigma-dependent within a single block (AdaLN
modulation, ctrl fusion, etc.); if blocks 0-3 all match and only later blocks diverge,
it's likely compounding instead.

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "sigma block trace"
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from waypoint_ttnn.tt.full_model import WaypointWorldModel
from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder
from waypoint_ttnn.tt.rope import compute_rope_angles


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    import ttnn

    ref = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/sigma_sweep.pt", weights_only=False)
    cfg_dict = torch.load("/home/ttuser/code/tt-waypoint/ref_activations/transformer_config.pt", weights_only=False)
    hf_config = _Cfg(cfg_dict)

    tf_weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    vae_weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/vae/diffusion_pytorch_model.safetensors"
    )[0]
    tf_state_dict = load_file(tf_weights_path)
    vae_state_dict = load_file(vae_weights_path)

    seed_image = ref["seed_image"]
    x_input = ref["x_input"].float()
    ts_mult = ref["ts_mult"]
    results = ref["results"]

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=21760)
    try:
        world_model = WaypointWorldModel.from_state_dict(tf_state_dict, hf_config=hf_config, mesh_device=device)
        vae_encoder = WaypointVAEEncoder.from_state_dict(vae_state_dict, mesh_device=device)

        mouse = torch.zeros(1, 1, 2)
        button = torch.zeros(1, 1, 256)
        scroll = torch.zeros(1, 1, 1)
        patch_size = 2

        t_down = vae_encoder.t_downscale
        seed_rgb = seed_image.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255)
        seed_rgb = seed_rgb.permute(0, 3, 1, 2)
        seed_patchified = F.pixel_unshuffle(seed_rgb, patch_size)
        seed_frames_nhwc = [
            ttnn.from_torch(
                seed_patchified[t : t + 1].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
                device=device, layout=ttnn.TILE_LAYOUT,
            )
            for t in range(t_down)
        ]
        enc_h = hf_config.height * hf_config.patch[0] * vae_encoder.spatial_downscale
        enc_w = hf_config.width * hf_config.patch[1] * vae_encoder.spatial_downscale
        seed_latent_tt = vae_encoder.encode(seed_frames_nhwc, enc_h, enc_w)
        seed_latent = ttnn.to_torch(seed_latent_tt).float().permute(0, 3, 1, 2).unsqueeze(1)

        rope0 = compute_rope_angles(hf_config, 0, ts_mult)
        zero_sigma = torch.zeros(1, 1)
        world_model.forward(seed_latent, zero_sigma, rope0, mouse, button, scroll, frame_idx=0, is_frozen=False)

        rope1 = compute_rope_angles(hf_config, 1, ts_mult)

        B, N, C, H, W = x_input.shape
        for sig_val in [0.75, 0.30078125]:
            r = results[sig_val]
            print(f"[test] === sigma={sig_val} ===")
            cond = world_model._noise_conditioner(torch.full((1, 1), float(sig_val)))
            ctrl_emb = world_model._ctrl_embedding(mouse, button, scroll)
            tokens, (hp, wp) = world_model._patchify(x_input.view(B * N, C, H, W))

            v1 = None
            h = tokens
            for i, layer in enumerate(world_model.layers):
                h, v1 = layer.decode_forward(h, rope1, cond, ctrl_emb, frame_idx=1, is_frozen=True, v1=v1)
                if i in r["block_outputs"]:
                    ref_block = r["block_outputs"][i]
                    diff = (h - ref_block).abs()
                    print(f"[test]   block {i:2d} (is_global={layer.layer_cfg.is_global} "
                          f"dilation={layer.layer_cfg.dilation}): max {diff.max().item():.4f} "
                          f"mean {diff.mean().item():.5f} corr {pearson_corr(h, ref_block):.6f}  "
                          f"(computed std/mean {h.std().item():.4f}/{h.mean().item():.4f}, "
                          f"ref std/mean {ref_block.std().item():.4f}/{ref_block.mean().item():.4f})")
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
