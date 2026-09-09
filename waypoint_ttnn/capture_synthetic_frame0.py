"""Directly call WorldModel.forward with a synthetic, fully-controlled frame 0 and a
freshly-constructed KV cache -- guarantees the empty-cache, pure self-attention scenario,
bypassing the pipeline's own multi-frame image-seeding logic entirely."""
import torch
from diffusers import AutoModel

OUT = "/home/ttuser/code/tt-waypoint/ref_activations"

wm = AutoModel.from_pretrained(
    "Overworld/Waypoint-1.5-1B", subfolder="transformer", trust_remote_code=True, torch_dtype=torch.bfloat16
)
wm.eval()

# Build a StaticKVCache exactly as modular_blocks.py does, fresh/empty.
import sys
sys.path.insert(0, "/home/ttuser/.cache/huggingface/modules")
from diffusers_modules.local import __path__ as _dmp  # noqa: F401 -- ensure package path exists
import glob
mod_dir = glob.glob(
    "/home/ttuser/.cache/huggingface/modules/diffusers_modules/local/Overworld--Waypoint-1.5-1B/*/"
)[0]
sys.path.insert(0, mod_dir)
from modular_blocks import StaticKVCache  # noqa: E402

kv_cache = StaticKVCache(wm.config, batch_size=1, dtype=torch.bfloat16)
kv_cache.set_frozen(False)  # commit writes, matches a "real" (non-speculative) step

B, C, H, W = 1, 32, 32, 64
# Fixed seed: this capture's whole point is to be a reproducible ground truth that other
# scripts reconstruct the SAME inputs against later (a real bug once -- the original
# version of this script had no seed at all, so "reconstruct x with manual_seed(0)" in a
# downstream test silently produced a DIFFERENT x than the one the reference was run on,
# and a full-model correctness check correctly reported near-zero correlation because the
# inputs genuinely differed, not because the model was wrong). Seed immediately before the
# first random draw and do not draw anything else beforehand.
torch.manual_seed(0)
x = torch.randn(B, 1, C, H, W, dtype=torch.bfloat16) * 0.5  # fake noisy latent, N=1 frame
sigma = torch.tensor([[0.5]], dtype=torch.bfloat16)
frame_timestamp = torch.tensor([[0]], dtype=torch.long)  # frame_idx = 0, the real target
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

wm.transformer.blocks[0].attn.register_forward_hook(hook("attn"))
wm.transformer.blocks[0].register_forward_hook(hook("block0"))

with torch.no_grad():
    out = wm(
        x=x, sigma=sigma, frame_timestamp=frame_timestamp,
        prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
        mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache,
    )
print("[synthetic] output shape:", out.shape)
print("[synthetic] captured:", list(captured.keys()))

to_save = {}
for name, (inputs, output) in captured.items():
    to_save[f"{name}__inputs"] = inputs
    to_save[f"{name}__output"] = output
to_save["wm_output"] = out
to_save["x_input"] = x
torch.save(to_save, f"{OUT}/synthetic_frame0.pt")
print("[synthetic] saved. DONE")
