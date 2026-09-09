"""Same synthetic frame-0 test, but monkeypatch LayerKVCache.upsert to capture its exact
(k, v, block_mask) return for layer 0, plus q after RoPE -- ground truth for what the
reference attention actually computes over, no reimplementation guessing."""
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
from modular_blocks import StaticKVCache, LayerKVCache

kv_cache = StaticKVCache(wm.config, batch_size=1, dtype=torch.bfloat16)
kv_cache.set_frozen(False)

captured = {}
orig_upsert = LayerKVCache.upsert
def patched_upsert(self, kv, pos_ids, is_frozen):
    k, v, bm = orig_upsert(self, kv, pos_ids, is_frozen)
    if "layer0" not in captured:
        captured["layer0"] = {"kv_in": kv.clone(), "k_out": k.clone(), "v_out": v.clone(),
                                "written": self.written.clone(), "capacity": self.capacity, "L": self.L}
    return k, v, bm
LayerKVCache.upsert = patched_upsert

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

layer0 = captured["layer0"]
print("[v2] kv_in shape", layer0["kv_in"].shape)
print("[v2] k_out shape", layer0["k_out"].shape, "capacity", layer0["capacity"], "L", layer0["L"])
print("[v2] written.sum()", layer0["written"].sum().item(), "/ ", layer0["written"].numel())
print("[v2] written True positions (first 20):", layer0["written"].nonzero().flatten()[:20])

# Compare k_out (full capacity buffer) at the "written" positions against kv_in (the fresh
# k/v for the current frame) -- confirms whether written positions == tail == kv_in.
tpf = 512
tail_k = layer0["k_out"][:, :, -tpf:]
fresh_k = layer0["kv_in"][0]  # [2,B,H,T,Dh][0] = k
diff_tail_vs_fresh = (tail_k.float() - fresh_k.float()).abs().max().item()
print("[v2] tail region vs fresh k, max abs diff (should be 0):", diff_tail_vs_fresh)

to_save = {"attn__inputs": hooks["attn"][0], "attn__output": hooks["attn"][1],
           "layer0_k_out": layer0["k_out"], "layer0_v_out": layer0["v_out"],
           "layer0_written": layer0["written"], "layer0_capacity": layer0["capacity"], "layer0_L": layer0["L"]}
torch.save(to_save, f"{OUT}/synthetic_frame0_v2.pt")
print("[v2] DONE")
