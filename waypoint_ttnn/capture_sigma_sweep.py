"""Investigates WHERE the frozen-step mean bias comes from by capturing, at several
sigma values (same x, same post-seed cache state, only sigma varies): (1) the raw
NoiseConditioner output `cond` in isolation (hooked directly, no transformer attention
involved at all -- isolates "is _noise_conditioner itself wrong at some sigmas" from
"is it an attention/cache issue"), and (2) the full transformer output `v`. If `cond`
matches at every sigma but `v`'s bias grows with sigma, the noise conditioner is cleared
and the bias is downstream (attention/MLP/AdaLN modulation). If `cond` itself diverges
at high sigma, that's the direct cause.
Run with a venv that has diffusers>=0.38 (AutoModel):
  /home/ttuser/code/tt-skyreels/.venv/bin/python3 waypoint_ttnn/capture_sigma_sweep.py
"""
import torch
from diffusers import AutoModel

OUT = "/home/ttuser/code/tt-waypoint/ref_activations"

wm = AutoModel.from_pretrained(
    "Overworld/Waypoint-1.5-1B", subfolder="transformer", trust_remote_code=True, torch_dtype=torch.bfloat16
)
wm.eval()
vae = AutoModel.from_pretrained(
    "Overworld/Waypoint-1.5-1B", subfolder="vae", trust_remote_code=True, torch_dtype=torch.float32
)
vae.eval()

import sys, glob
sys.path.insert(0, "/home/ttuser/.cache/huggingface/modules")
mod_dir = glob.glob(
    "/home/ttuser/.cache/huggingface/modules/diffusers_modules/local/Overworld--Waypoint-1.5-1B/*/"
)[0]
sys.path.insert(0, mod_dir)
from modular_blocks import StaticKVCache

kv_cache = StaticKVCache(wm.config, batch_size=1, dtype=torch.bfloat16)

C, latent_H, latent_W = wm.config.channels, wm.config.height * wm.config.patch[0], wm.config.width * wm.config.patch[1]
vae_scale_factor = 16
pixel_H, pixel_W = latent_H * vae_scale_factor, latent_W * vae_scale_factor

base_fps = getattr(wm.config, "base_fps", 60)
inference_fps = getattr(wm.config, "inference_fps", base_fps)
latent_fps = inference_fps / wm.config.temporal_compression
ts_mult = int(base_fps) // int(latent_fps)

torch.manual_seed(0)
mouse = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
button = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
scroll = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 2048, dtype=torch.bfloat16) * 0.1
prompt_pad_mask = torch.ones(1, 512, dtype=torch.bfloat16)
frame_timestamp = torch.tensor([[0]], dtype=torch.long)

seed_image = torch.randint(0, 256, (pixel_H, pixel_W, 3), dtype=torch.uint8)
t_down = vae.t_downscale
seed_frames = seed_image.unsqueeze(0).expand(t_down, -1, -1, -1)

cond_calls = []
def cond_hook(module, inputs, output):
    cond_calls.append((inputs[0].clone(), output.clone()))
wm.denoise_step_emb.register_forward_hook(cond_hook)

#: blocks 0-3 alone showed comparable (expected) degradation for both sigmas -- the
#: sigma=0.30078125 catastrophe emerges later, so hook everything this round.
HOOKED_BLOCKS = list(range(24))
block_calls = {i: [] for i in HOOKED_BLOCKS}
def make_block_hook(i):
    def _hook(module, inputs, output):
        block_calls[i].append(output[0].clone() if isinstance(output, tuple) else output.clone())
    return _hook
for i in HOOKED_BLOCKS:
    wm.transformer.blocks[i].register_forward_hook(make_block_hook(i))

with torch.no_grad():
    seed_latent = vae.encode(seed_frames).unsqueeze(1).to(torch.bfloat16)

    kv_cache.set_frozen(False)
    wm(x=seed_latent, sigma=seed_latent.new_zeros((1, 1)),
       frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
       prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
       mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)
    frame_timestamp = frame_timestamp + 1

    x = torch.randn((1, 1, C, latent_H, latent_W), dtype=torch.bfloat16)
    kv_cache.set_frozen(True)

    results = {}
    for sig_val in [0.75, 0.30078125]:  # one known-good control, one known-bad target
        sigma = x.new_full((1, 1), sig_val)
        v = wm(x=x, sigma=sigma,
               frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
               prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
               mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)
        cond_input, cond_output = cond_calls[-1]
        block_outputs = {i: block_calls[i][-1].float() for i in HOOKED_BLOCKS}
        results[sig_val] = {
            "v": v.float(), "cond_input": cond_input.float(), "cond_output": cond_output.float(),
            "block_outputs": block_outputs,
        }
        print(f"[sweep] sigma={sig_val}: v std/mean {v.float().std().item():.4f}/{v.float().mean().item():.4f}")

to_save = {"seed_image": seed_image, "x_input": x, "ts_mult": ts_mult, "results": results}
torch.save(to_save, f"{OUT}/sigma_sweep.pt")
print("[sweep] saved. DONE")
