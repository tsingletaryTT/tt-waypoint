"""Ground truth for the real multi-frame interactive generation protocol: seed a session
from a real image (VAE encode -> commit pass, frame_idx=0, no denoising), then generate 2
real frames via the actual rectified-flow denoise loop (K frozen sigma steps + 1 unfrozen
commit pass each, matching WorldEngineDenoiseLoop/_denoise_pass/_cache_pass in
modular_blocks.py exactly -- replicated directly here rather than going through the full
ModularPipeline, for the same reason capture_synthetic_frame0.py bypasses the pipeline's
own multi-frame image-seeding: full control, full visibility, no ~10min-per-call cost).
Run with a venv that has diffusers>=0.38 (AutoModel):
  /home/ttuser/code/tt-skyreels/.venv/bin/python3 waypoint_ttnn/capture_generation_loop.py
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
print(f"[gen] latent shape (1,1,{C},{latent_H},{latent_W}), seed image {pixel_H}x{pixel_W}")

base_fps = getattr(wm.config, "base_fps", 60)
inference_fps = getattr(wm.config, "inference_fps", base_fps)
latent_fps = inference_fps / wm.config.temporal_compression
ts_mult = int(base_fps) // int(latent_fps)
sigmas = torch.tensor(wm.config.scheduler_sigmas, dtype=torch.bfloat16)
print(f"[gen] ts_mult={ts_mult} sigmas={sigmas.tolist()}")

torch.manual_seed(0)
mouse = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
button = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
scroll = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
prompt_emb = torch.randn(1, 512, 2048, dtype=torch.bfloat16) * 0.1
prompt_pad_mask = torch.ones(1, 512, dtype=torch.bfloat16)
frame_timestamp = torch.tensor([[0]], dtype=torch.long)

# --- Seed frame: real random "image" (uint8, repeated t_downscale times), VAE-encode, commit ---
seed_image = torch.randint(0, 256, (pixel_H, pixel_W, 3), dtype=torch.uint8)
t_down = vae.t_downscale
seed_frames = seed_image.unsqueeze(0).expand(t_down, -1, -1, -1)
with torch.no_grad():
    seed_latent = vae.encode(seed_frames).unsqueeze(1).to(torch.bfloat16)  # [1, 1, C, H, W]

    kv_cache.set_frozen(False)
    wm(x=seed_latent, sigma=seed_latent.new_zeros((1, 1)),
       frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
       prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
       mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)
    frame_timestamp = frame_timestamp + 1

    frame_latents = [seed_latent]
    noise_draws = []
    step_trace = []  # per (frame, step): real, properly-evolved x/v/sigma, not synthetic
    for frame_i in range(2):
        x = torch.randn((1, 1, C, latent_H, latent_W), dtype=torch.bfloat16)
        noise_draws.append(x.clone())
        kv_cache.set_frozen(True)
        sigma = x.new_empty((x.size(0), x.size(1)))
        for step_i, (step_sig, step_dsig) in enumerate(zip(sigmas, sigmas.diff())):
            v = wm(x=x, sigma=sigma.fill_(step_sig),
                   frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
                   prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
                   mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)
            step_trace.append({
                "frame": frame_i, "step": step_i, "sigma": float(step_sig), "dsigma": float(step_dsig),
                "x_in": x.float().clone(), "v_out": v.float().clone(),
            })
            x = x + step_dsig * v
        x = x.clone()

        kv_cache.set_frozen(False)
        wm(x=x, sigma=x.new_zeros((1, 1)),
           frame_timestamp=frame_timestamp * ts_mult, frame_idx=frame_timestamp,
           prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
           mouse=mouse, button=button, scroll=scroll, kv_cache=kv_cache)
        frame_timestamp = frame_timestamp + 1
        frame_latents.append(x)

    decoded = [vae.decode(lat.squeeze(1)) for lat in frame_latents]

print("[gen] frame_latents shapes:", [t.shape for t in frame_latents])
print("[gen] decoded shapes:", [t.shape for t in decoded])

to_save = {
    "seed_image": seed_image,
    "frame_latents": torch.stack(frame_latents),
    "noise_draws": torch.stack(noise_draws),
    "decoded_frames": decoded,
    "ts_mult": ts_mult,
    "sigmas": sigmas.float(),
    "step_trace": step_trace,
}
torch.save(to_save, f"{OUT}/generation_loop.pt")
print("[gen] saved. DONE")
