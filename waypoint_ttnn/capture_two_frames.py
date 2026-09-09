"""Two consecutive frames through the SAME kv_cache (frame 0 then frame 1, no reset in
between) -- ground truth for testing decode_forward's ring-buffer write/read logic across
multiple frames (Stage 4's still-unverified item per PORT_PLAN.md: "this first pass only
proved frame 0's empty-buffer case"). Run with a venv that has diffusers>=0.38 (AutoModel):
  /home/ttuser/code/tt-skyreels/.venv/bin/python3 waypoint_ttnn/capture_two_frames.py
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
x0 = torch.randn(B, 1, C, H, W, dtype=torch.bfloat16) * 0.5
x1 = torch.randn(B, 1, C, H, W, dtype=torch.bfloat16) * 0.5
sigma = torch.tensor([[0.5]], dtype=torch.bfloat16)
mouse = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
button = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
scroll = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 2048, dtype=torch.bfloat16) * 0.1
prompt_pad_mask = torch.ones(1, 512, dtype=torch.bfloat16)

captured = {}
def hook(name):
    def _hook(module, inputs, output):
        captured[name] = output[0] if isinstance(output, tuple) else output
    return _hook

n_layers = len(wm.transformer.blocks)
for i, blk in enumerate(wm.transformer.blocks):
    blk.register_forward_hook(hook(f"block{i}"))

attn_calls = []
def attn_hook(module, inputs, output):
    attn_calls.append(inputs)
wm.transformer.blocks[0].attn.register_forward_hook(attn_hook)

with torch.no_grad():
    out0 = wm(
        x=x0, sigma=sigma, frame_timestamp=torch.tensor([[0]], dtype=torch.long),
        prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
        mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache,
    )
    frame0_blocks = {f"block{i}__output": captured[f"block{i}"] for i in range(n_layers)}
    frame0_rope = attn_calls[0][2]

    out1 = wm(
        x=x1, sigma=sigma, frame_timestamp=torch.tensor([[1]], dtype=torch.long),
        prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
        mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache,
    )
    frame1_blocks = {f"block{i}__output": captured[f"block{i}"] for i in range(n_layers)}
    frame1_rope = attn_calls[1][2]

print("[capture2] out0 shape", out0.shape, "out1 shape", out1.shape)

to_save = {
    "x0": x0, "x1": x1,
    "wm_output_frame0": out0, "wm_output_frame1": out1,
    "rope_angles_frame0": frame0_rope, "rope_angles_frame1": frame1_rope,
}
for k, v in frame0_blocks.items():
    to_save[f"frame0_{k}"] = v
for k, v in frame1_blocks.items():
    to_save[f"frame1_{k}"] = v
torch.save(to_save, f"{OUT}/two_frames.pt")
print("[capture2] saved. DONE")
