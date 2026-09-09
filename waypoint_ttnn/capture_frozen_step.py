"""Isolates the FIRST frozen denoising forward call of frame 1 (is_frozen=True,
frame_idx=1, sigma=1.0) with a KNOWN x input, immediately after the real seed commit --
the one new code path test_generation_loop.py's multi-step loop exercises that no prior
test covers (test_two_frame_decode.py only ever used is_frozen=False). Isolates "is a
single frozen call correct" from "does 4-step rectified-flow accumulation amplify
already-expected per-call noise".
Run with a venv that has diffusers>=0.38 (AutoModel):
  /home/ttuser/code/tt-skyreels/.venv/bin/python3 waypoint_ttnn/capture_frozen_step.py
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
sigmas = torch.tensor(wm.config.scheduler_sigmas, dtype=torch.bfloat16)

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
with torch.no_grad():
    seed_latent = vae.encode(seed_frames).unsqueeze(1).to(torch.bfloat16)

    kv_cache.set_frozen(False)
    wm(x=seed_latent, sigma=seed_latent.new_zeros((1, 1)),
       frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
       prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
       mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)
    frame_timestamp = frame_timestamp + 1

    # ONE frozen call, frame_idx=1, sigma=sigmas[0]=1.0, known x.
    x = torch.randn((1, 1, C, latent_H, latent_W), dtype=torch.bfloat16)
    kv_cache.set_frozen(True)
    sigma = x.new_full((1, 1), sigmas[0].item())
    v = wm(x=x, sigma=sigma,
           frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
           prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
           mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)

to_save = {"seed_image": seed_image, "x_input": x, "sigma_value": sigmas[0].float(), "v_output": v,
           "rope_frame_idx": 1, "ts_mult": ts_mult}
torch.save(to_save, f"{OUT}/frozen_step.pt")
print("[frozen_step] v shape", v.shape, "saved. DONE")
