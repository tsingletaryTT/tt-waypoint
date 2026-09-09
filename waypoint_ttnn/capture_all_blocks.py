"""Same synthetic frame-0 scenario as capture_synthetic_frame0.py, but hooks EVERY
transformer block (not just block 0) so per-layer divergence in the full model can be
localized precisely instead of guessed at from code reading. Run with a venv that has
diffusers>=0.38 (AutoModel support) -- the shared .tenstorrent-venv predates it; the
tt-skyreels venv works:
  /home/ttuser/code/tt-skyreels/.venv/bin/python3 waypoint_ttnn/capture_all_blocks.py
"""
import torch
from diffusers import AutoModel

OUT = "/home/ttuser/code/tt-waypoint/ref_activations"

wm = AutoModel.from_pretrained(
    "Overworld/Waypoint-1.5-1B", subfolder="transformer", trust_remote_code=True, torch_dtype=torch.bfloat16
)
wm.eval()

import sys, glob
sys.path.insert(0, "/home/ttuser/.cache/huggingface/modules")
mod_dir = glob.glob(
    "/home/ttuser/.cache/huggingface/modules/diffusers_modules/local/Overworld--Waypoint-1.5-1B/*/"
)[0]
sys.path.insert(0, mod_dir)
from modular_blocks import StaticKVCache

kv_cache = StaticKVCache(wm.config, batch_size=1, dtype=torch.bfloat16)
kv_cache.set_frozen(False)

B, C, H, W = 1, 32, 32, 64
torch.manual_seed(0)
x = torch.randn(B, 1, C, H, W, dtype=torch.bfloat16) * 0.5
sigma = torch.tensor([[0.5]], dtype=torch.bfloat16)
frame_timestamp = torch.tensor([[0]], dtype=torch.long)
mouse = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
button = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
scroll = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 2048, dtype=torch.bfloat16) * 0.1
prompt_pad_mask = torch.ones(1, 512, dtype=torch.bfloat16)

captured = {}
def hook(name):
    def _hook(module, inputs, output):
        captured[name] = (inputs, output)
    return _hook

n_layers = len(wm.transformer.blocks)
print(f"[capture_all] n_layers={n_layers}")
for i, blk in enumerate(wm.transformer.blocks):
    blk.register_forward_hook(hook(f"block{i}"))

with torch.no_grad():
    out = wm(
        x=x, sigma=sigma, frame_timestamp=frame_timestamp,
        prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
        mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache,
    )
print("[capture_all] output shape:", out.shape)

to_save = {"wm_output": out, "x_input": x}
for i in range(n_layers):
    inputs, output = captured[f"block{i}"]
    to_save[f"block{i}__output"] = output[0] if isinstance(output, tuple) else output
torch.save(to_save, f"{OUT}/all_blocks_frame0.pt")
print("[capture_all] saved. DONE")
