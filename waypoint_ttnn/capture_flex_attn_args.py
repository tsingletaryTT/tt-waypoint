"""Monkeypatch flex_attention itself to capture its EXACT arguments for layer 0's call --
100% ground truth, zero reimplementation risk."""
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
import modular_blocks as mb_mod

_mbm_captured = {}
_orig_make_block_mask = mb_mod.make_block_mask
def _patched_make_block_mask(T, L, written):
    if "written" not in _mbm_captured:
        _mbm_captured["written"] = written.clone()
        _mbm_captured["T"] = T
        _mbm_captured["L"] = L
    return _orig_make_block_mask(T, L, written)
mb_mod.make_block_mask = _patched_make_block_mask

kv_cache = StaticKVCache(wm.config, batch_size=1, dtype=torch.bfloat16)
kv_cache.set_frozen(False)

captured = {}
import torch.nn.attention.flex_attention as fa_mod_orig
orig_flex = fa_mod_orig.flex_attention
call_count = [0]
def patched_flex(q, k, v, block_mask=None, enable_gqa=False):
    if call_count[0] == 0:
        captured["q"] = q.clone()
        captured["k"] = k.clone()
        captured["v"] = v.clone()
        captured["enable_gqa"] = enable_gqa
        # Extract which KV positions the block mask actually marks visible for query 0.
        try:
            captured["kv_num_blocks"] = block_mask.kv_num_blocks.clone()
            captured["kv_indices"] = block_mask.kv_indices.clone()
        except Exception as e:
            captured["block_mask_repr_error"] = str(e)
    call_count[0] += 1
    return orig_flex(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)

# Patch the name flex_attention is bound to inside model.py's module namespace, since it's
# imported locally inside Attn.forward via `from torch.nn.attention.flex_attention import
# flex_attention` -- patch the global torch function itself instead, which that local import
# will resolve to at call time.
import torch.nn.attention.flex_attention as fa_mod
fa_mod.flex_attention = patched_flex

x = torch.randn(1, 1, 32, 32, 64, dtype=torch.bfloat16) * 0.5
sigma = torch.tensor([[0.5]], dtype=torch.bfloat16)
frame_timestamp = torch.tensor([[0]], dtype=torch.long)
mouse = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
button = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
scroll = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 2048, dtype=torch.bfloat16) * 0.1
prompt_pad_mask = torch.ones(1, 512, dtype=torch.bfloat16)

hooks = {}
def hook(name):
    def _hook(module, inputs, output):
        hooks[name] = (inputs, output)
    return _hook
wm.transformer.blocks[0].attn.register_forward_hook(hook("attn"))

with torch.no_grad():
    out = wm(
        x=x, sigma=sigma, frame_timestamp=frame_timestamp,
        prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
        mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache,
    )

print("[flex] q shape", captured["q"].shape, "k shape", captured["k"].shape, "v shape", captured["v"].shape)
print("[flex] enable_gqa", captured["enable_gqa"])
if "kv_num_blocks" in captured:
    print("[flex] kv_num_blocks (per q block):", captured["kv_num_blocks"].flatten()[:10])
    print("[flex] kv_indices (first q block's visible kv blocks):", captured["kv_indices"][0,0,0,:10])
else:
    print("[flex] block_mask_repr_error:", captured.get("block_mask_repr_error"))

to_save = {"flex_q": captured["q"], "flex_k": captured["k"], "flex_v": captured["v"],
           "attn__inputs": hooks["attn"][0], "attn__output": hooks["attn"][1]}
if "kv_num_blocks" in captured:
    to_save["kv_num_blocks"] = captured["kv_num_blocks"]
    to_save["kv_indices"] = captured["kv_indices"]
torch.save(to_save, f"{OUT}/flex_attn_args.pt")
print("[mbm] captured written sum:", _mbm_captured["written"].sum().item(), "/", _mbm_captured["written"].numel())
print("[mbm] T", _mbm_captured["T"], "L", _mbm_captured["L"])
print("[mbm] written True positions:", _mbm_captured["written"].nonzero().flatten())
print("[flex] DONE")
