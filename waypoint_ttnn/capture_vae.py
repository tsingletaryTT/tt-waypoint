"""VAE (ChunkedStreamingTAEHV) reference capture: encode a 4-frame RGB chunk to one
latent, then decode 3 consecutive latents to RGB frames, keeping the SAME model instance
(streaming state) across calls -- exactly how the real pipeline uses it (seed with an
image via encode, then decode each generated latent as it arrives).
Run with a venv that has diffusers>=0.38 (AutoModel):
  /home/ttuser/code/tt-skyreels/.venv/bin/python3 waypoint_ttnn/capture_vae.py
"""
import torch
from diffusers import AutoModel

OUT = "/home/ttuser/code/tt-waypoint/ref_activations"

vae = AutoModel.from_pretrained(
    "Overworld/Waypoint-1.5-1B", subfolder="vae", trust_remote_code=True, torch_dtype=torch.float32
)
vae.eval()
print("[vae] t_downscale", vae.t_downscale, "t_upscale", vae.t_upscale,
      "frames_to_trim", vae.frames_to_trim, "latent_channels", vae.config.latent_channels)

torch.manual_seed(0)
# encode() wants [T, H, W, C] uint8, T == t_downscale
T_in = vae.t_downscale
H, W = 256, 256
frames_uint8 = torch.randint(0, 256, (T_in, H, W, 3), dtype=torch.uint8)

with torch.no_grad():
    latent0 = vae.encode(frames_uint8)  # [B, C, h, w]
    print("[vae] latent0 shape", latent0.shape)

    # Decode 3 consecutive latents through the SAME streaming decoder state.
    latents = [latent0]
    for _ in range(2):
        latents.append(torch.randn_like(latent0) * 0.5)

    decoded_frames = []
    for lat in latents:
        frames = vae.decode(lat)  # [T, H, W, C] uint8 -- may be empty for early latents (frames_to_trim)
        decoded_frames.append(frames)
        print("[vae] decode step -> frames shape", frames.shape)

to_save = {
    "frames_uint8_in": frames_uint8,
    "latent0": latent0,
    "latents": torch.stack(latents),
    "decoded_frames": decoded_frames,
    "t_downscale": vae.t_downscale,
    "t_upscale": vae.t_upscale,
    "frames_to_trim": vae.frames_to_trim,
}
torch.save(to_save, f"{OUT}/vae_capture.pt")

# Also dump the VAE config for the port.
import json
with open(f"{OUT}/vae_config.json", "w") as f:
    json.dump(dict(vae.config), f, indent=2, default=str)

print("[vae] saved. DONE")
