# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Layer-0 correctness check: FunctionalDecoder.prefill_forward against the real
HF reference, on real Blackhole hardware.

Reference activations come from calling WorldModel.forward directly with a synthetic
frame-0 input and a freshly-constructed StaticKVCache (full control over cache state --
see BRINGUP_LOG.md in ~/code/tt-waypoint for why the pipeline's own multi-frame image
seeding makes a naive first-call capture unreliable). Captured via:
~/code/tt-waypoint/waypoint_ttnn/capture_synthetic_frame0.py

Run under a gozer lease:
  gozer acquire --chips 1 --who "claude:tt-waypoint-bringup" --reason "functional_decoder layer0 PCC check"
"""
import sys
from pathlib import Path

# tt-metal for models.common.lightweightmodule / ttnn; this repo's own root for
# waypoint_ttnn itself. See CLAUDE.md's "Bring-up vs. packaging" note: functional_decoder.py
# uses the ttm-functional-decoder skill's conventions (LightweightModule, from_state_dict,
# prefill_forward/decode_forward, PCC-based verification) but lives in OUR repo rather
# than tt-metal's models/autoports/ tree, matching tt-skyreels' decoupled-repo pattern.
sys.path.insert(0, "/home/ttuser/tt-metal")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import torch
from safetensors.torch import load_file

from waypoint_ttnn.tt.functional_decoder import FunctionalDecoder


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    import ttnn

    ref = torch.load(
        "/home/ttuser/code/tt-waypoint/ref_activations/synthetic_frame0.pt", weights_only=False
    )
    cfg_dict = torch.load(
        "/home/ttuser/code/tt-waypoint/ref_activations/transformer_config.pt", weights_only=False
    )
    hf_config = _Cfg(cfg_dict)

    weights_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
    )[0]
    state_dict = load_file(weights_path)

    x_in = ref["block0__inputs"][0].float()
    pos_ids = ref["block0__inputs"][1]
    cond = ref["block0__inputs"][3].float()
    ctx = ref["block0__inputs"][4]
    ctrl_emb = ctx["ctrl_emb"].float() if isinstance(ctx, dict) and "ctrl_emb" in ctx else None
    y_ref = ref["block0__output"][0].float()

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder = FunctionalDecoder.from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=0, mesh_device=device
        )

        # rope_angles: the shared venv's diffusers predates AutoModel (same gap hit
        # during tt-skyreels' bring-up), so use the attn-level capture already loaded
        # above -- ref["attn__inputs"][2] is the exact same rope_angles tuple
        # WorldDiT.forward computed once and passed to every block.
        rope_angles = ref["attn__inputs"][2]
        rope_angles = (rope_angles[0].float(), rope_angles[1].float())

        y_computed, _ = decoder.prefill_forward(x_in, rope_angles, cond, ctrl_emb)

        diff = (y_computed - y_ref).abs()
        corr = pearson_corr(y_computed, y_ref)
        print(f"[test] layer0 prefill_forward vs reference: max abs diff {diff.max().item():.6f}, "
              f"mean abs diff {diff.mean().item():.6f}, correlation {corr:.6f}")
        assert corr > 0.99, f"correlation too low: {corr}"
        print("[test] y_ref std/mean:", y_ref.std().item(), y_ref.mean().item())
        print("[test] y_computed std/mean:", y_computed.std().item(), y_computed.mean().item())
        print("[test] norm ratio:", y_computed.norm().item() / y_ref.norm().item())
        worst = diff.argmax()
        idx = torch.unravel_index(worst, diff.shape)
        print("[test] worst position:", idx, "ref", y_ref[idx].item(), "computed", y_computed[idx].item())
        # per-token max diff, to see if it is spread or concentrated in a few tokens
        per_token = diff.amax(dim=-1)
        print("[test] per-token max diff (top 5):", per_token.flatten().topk(5))
    finally:
        ttnn.close_mesh_device(device)
    print("[test] DONE")


if __name__ == "__main__":
    main()
