#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gradio UI for tt-waypoint (Waypoint-1.5-1B, interactive world model).

Local (Blackhole hardware, single chip -- a 1x1 mesh):
    pip install gradio
    python app.py

Unlike tt-skyreels' one-shot "type a prompt, get a video" app, Waypoint-1.5-1B is
STATEFUL and interactive: seed a session from a starting image, then step it forward one
generated frame at a time, each conditioned on a mouse/button/scroll control and the
session's accumulated history (a live KV cache -- see waypoint_ttnn/session.py, which
holds exactly one active session per process, matching this being a demo app rather than
a multi-tenant server).

Every `Step` click runs a REAL forward pass: K frozen rectified-flow denoising steps
through the 24-layer transformer, one unfrozen commit pass, then a VAE decode -- all on
real Blackhole hardware, no simulation/CPU fallback. This is a first CORRECTNESS pass
(PORT_PLAN.md's "Explicitly out of scope" section is explicit that real-time performance
is a separate, later project), so each step can take a while, especially on a cold
kernel cache -- the button disables and shows a status message while a step runs rather
than looking frozen.

Button-tensor mapping is a known, documented simplification, not a discovered fact: the
real model's `button_tensor` is a 256-wide one-hot vector without published semantics
for a specific index. This UI leaves it all-zero (idle) and only drives `mouse_tensor`
(one of 8 fixed directions) and `scroll_tensor` (zoom in/out) -- enough to show the model
actually responding to different inputs, without claiming to reverse-engineer a control
scheme nobody has documented.
"""

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

#: 8-directional mouse deltas, matching mouse_tensor's [dx, dy] contract.
_DIRECTIONS = {
    "Forward": (0.0, -1.0),
    "Back": (0.0, 1.0),
    "Left": (-1.0, 0.0),
    "Right": (1.0, 0.0),
    "Forward-left": (-0.7, -0.7),
    "Forward-right": (0.7, -0.7),
    "Back-left": (-0.7, 0.7),
    "Back-right": (0.7, 0.7),
}

_DESCRIPTION = """
**Waypoint-1.5-1B on Tenstorrent Blackhole** — an interactive world model: seed a session
from a starting image, then step it forward, one generated frame at a time, steered by
direction and zoom. Runs on a single Blackhole chip. First correctness pass, not yet
optimized for speed — see the repo's PORT_PLAN.md for the benchmarking plan.
"""


def start_session(image):
    if image is None:
        raise gr.Error("Upload a starting image first.")

    from waypoint_ttnn import session
    import torch
    import torch.nn.functional as F
    import numpy as np

    yield None, "Opening the mesh device (first call only; loads weights + compiles kernels)…", gr.update(interactive=False)

    try:
        gen = session.start_session()
    except Exception as exc:
        raise gr.Error(f"Device/model setup failed: {exc}") from exc

    yield None, "Encoding the seed image…", gr.update(interactive=False)

    import ttnn

    hf_config = gen.hf_config
    ph, pw = hf_config.patch
    vae_scale_factor = 16
    pixel_h, pixel_w = gen.latent_h * vae_scale_factor, gen.latent_w * vae_scale_factor

    img = torch.from_numpy(np.array(image.convert("RGB").resize((pixel_w, pixel_h))))  # [H,W,3] uint8
    t_down = gen.vae_encoder.t_downscale
    rgb = img.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255).permute(0, 3, 1, 2)
    patchified = F.pixel_unshuffle(rgb, gen.vae_decoder.patch_size)
    seed_frames = [
        ttnn.from_torch(
            patchified[t : t + 1].permute(0, 2, 3, 1).contiguous().to(torch.bfloat16),
            device=gen.mesh_device, layout=ttnn.TILE_LAYOUT,
        )
        for t in range(t_down)
    ]

    mouse = torch.zeros(1, 1, 2)
    button = torch.zeros(1, 1, 256)
    scroll = torch.zeros(1, 1, 1)

    try:
        latent = gen.seed(seed_frames, mouse, button, scroll)
        frame_img = _decode_and_show(gen, latent)
    except Exception as exc:
        raise gr.Error(f"Seeding failed: {exc}") from exc

    yield frame_img, "Session started. Pick a direction and click Step.", gr.update(interactive=True)


def step_session(direction: str, zoom: float):
    from waypoint_ttnn import session

    gen = session.current_session()
    if gen is None:
        raise gr.Error("Start a session first (upload an image above).")

    import torch

    dx, dy = _DIRECTIONS[direction]
    mouse = torch.tensor([[[dx, dy]]], dtype=torch.float32)
    button = torch.zeros(1, 1, 256)  # see module docstring: no documented button semantics
    scroll = torch.tensor([[[float(zoom)]]], dtype=torch.float32)

    try:
        latent = gen.step(mouse, button, scroll)
        frame_img = _decode_and_show(gen, latent)
    except Exception as exc:
        raise gr.Error(f"Step failed: {exc}") from exc

    return frame_img, f"Frame {gen.frame_timestamp}."


def _decode_and_show(gen, latent):
    """Decodes one latent to a displayable PIL image -- shows the LAST of the VAE's
    t_upscale output frames (the most temporally-recent one), since decode() can return
    more than one frame per call (see vae_decoder.py's own streaming-priming note)."""
    import torch.nn.functional as F
    import ttnn
    from PIL import Image

    out_frames = gen.decode_frame(latent)
    last = out_frames[-1]
    torch_frame = ttnn.to_torch(last).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
    rgb = F.pixel_shuffle(torch_frame, gen.vae_decoder.patch_size).clamp(0, 1)
    rgb_uint8 = (rgb[0] * 255).round().to(dtype=__import__("torch").uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(rgb_uint8)


with gr.Blocks(title="tt-waypoint") as demo:
    gr.Markdown("# tt-waypoint")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            seed_image = gr.Image(label="Starting image", type="pil")
            start_btn = gr.Button("Start session", variant="primary")
            gr.Markdown("---")
            direction = gr.Radio(
                list(_DIRECTIONS.keys()), value="Forward", label="Direction",
            )
            zoom = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Scroll / zoom")
            step_btn = gr.Button("Step", interactive=False)

        with gr.Column(scale=1):
            frame_view = gr.Image(label="Current frame", interactive=False)
            status_label = gr.Textbox(label="Status", value="", interactive=False, lines=1)

    start_btn.click(
        fn=start_session,
        inputs=[seed_image],
        outputs=[frame_view, status_label, step_btn],
    )
    step_btn.click(
        fn=step_session,
        inputs=[direction, zoom],
        outputs=[frame_view, status_label],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)
