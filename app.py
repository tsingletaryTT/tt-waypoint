#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gradio UI for tt-waypoint (Waypoint-1.5-1B, interactive world model).

Local (Blackhole hardware, single chip -- a 1x1 mesh), via the dedicated Gradio venv
(NOT the shared .tenstorrent-venv -- see "Why a separate venv" below):
    python3 -m venv ~/tt-gradio-venv && ~/tt-gradio-venv/bin/pip install gradio requests
    ~/tt-gradio-venv/bin/python app.py

This talks HTTP to the model server (waypoint_ttnn/server/app.py's ASGI app, run under
the SHARED venv where ttnn/torch actually live) rather than importing ttnn/torch itself.
Start that server first, e.g.:
    WAYPOINT_MESH_SHAPE=1x1 gozer run --chips 1 --who claude:tt-waypoint-gradio \
        --reason "serve waypoint_ttnn for the Gradio UI" -- \
        /home/ttuser/.tenstorrent-venv/bin/python -m uvicorn \
        waypoint_ttnn.server.app:app --host 0.0.0.0 --port 8001
Then point this app at it with WAYPOINT_SERVER_URL (default http://localhost:8001).

WHY A SEPARATE VENV, AND WHY HTTP INSTEAD OF IN-PROCESS IMPORT
----------------------------------------------------------------
The shared `.tenstorrent-venv` pins an old `gradio==4.44.1` alongside much newer
`starlette`/`jinja2`/`fastapi`/`pydantic` than that gradio version was built against --
real, reproducible crashes on EVERY page load. Upgrading gradio IN the shared venv looked
like the fix but isn't: gradio >=5 needs `huggingface-hub>=1.16`, which breaks
`transformers`/`tokenizers`/`datasets` (pinned `<1.0`) for every OTHER tool in that shared
venv -- confirmed by actually trying it, then reverting. A dedicated venv with a modern
gradio avoids that fight.

The first version of this file tried to bridge the two venvs by adding the shared venv's
site-packages (and the tt-metal checkout) to `sys.path` from within the dedicated venv's
interpreter, then importing `ttnn`/`torch` directly. That does NOT work for `ttnn`: it is
an EDITABLE install (`pip install -e`), and editable-install machinery (the `.pth` file's
import hook that registers ttnn's real finder in `sys.meta_path`) only runs during `site`
module processing at interpreter STARTUP -- i.e. only when `.tenstorrent-venv/bin/python`
itself is the interpreter that started. Retroactively appending that site-packages dir to
`sys.path` from an already-running DIFFERENT interpreter never triggers it, so `import
ttnn` silently resolves to a bare namespace package (no compiled bindings, no
`open_mesh_device`) instead of the real thing -- confirmed by reproducing it directly.
Talking HTTP to the already-verified `waypoint_ttnn/server/app.py` ASGI server (run
normally, under the venv that actually owns ttnn) sidesteps this entirely: this file
never imports ttnn/torch, so it has no ABI or editable-install concerns at all.

Unlike tt-skyreels' one-shot "type a prompt, get a video" app, Waypoint-1.5-1B is
STATEFUL and interactive: seed a session from a starting image, then step it forward one
generated frame at a time, each conditioned on a mouse/button/scroll control and the
session's accumulated history (a live KV cache, held server-side).

Every `Step` click runs a REAL forward pass on the server: K frozen rectified-flow
denoising steps through the 24-layer transformer, one unfrozen commit pass, then a VAE
decode -- all on real Blackhole hardware, no simulation/CPU fallback. This is a first
CORRECTNESS pass (PORT_PLAN.md's "Explicitly out of scope" section is explicit that
real-time performance is a separate, later project), so each step can take a while,
especially on a cold kernel cache -- the button disables and shows a status message while
a step runs rather than looking frozen.

Button-tensor mapping is a known, documented simplification, not a discovered fact: the
real model's `button_tensor` is a 256-wide one-hot vector without published semantics for
a specific index. This UI leaves it all-zero (idle) and only drives `mouse_tensor` (one of
8 fixed directions) and `scroll_tensor` (zoom in/out) -- enough to show the model actually
responding to different inputs, without claiming to reverse-engineer a control scheme
nobody has documented.
"""

import base64
import io
import os

import gradio as gr
import requests

SERVER_URL = os.environ.get("WAYPOINT_SERVER_URL", "http://localhost:8001")

_DIRECTIONS = ["Forward", "Back", "Left", "Right", "Forward-left", "Forward-right", "Back-left", "Back-right"]


def _direction_param(label: str) -> str:
    """UI label -> the server's DIRECTIONS key (waypoint_ttnn/server/app.py)."""
    return label.lower().replace("-", "_")


_DESCRIPTION = """
**Waypoint-1.5-1B on Tenstorrent Blackhole** — an interactive world model: seed a session
from a starting image, then step it forward, one generated frame at a time, steered by
direction and zoom. Runs on a single Blackhole chip. First correctness pass, not yet
optimized for speed — see the repo's PORT_PLAN.md for the benchmarking plan.
"""

#: Session id from the last successful /v1/sessions call, held per Gradio process --
#: matches the server's own single-active-session contract (see server/app.py docstring).
_session_id = None


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_image(b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64)))


def start_session(image):
    global _session_id
    if image is None:
        raise gr.Error("Upload a starting image first.")

    yield None, "Contacting the model server (first call: device open + weight load + kernel compile)…", gr.update(interactive=False)

    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/sessions",
            json={"image_b64": _image_to_b64(image)},
            timeout=1800,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise gr.Error(f"Seeding failed: {exc}") from exc

    data = resp.json()
    _session_id = data["session_id"]
    frame_img = _b64_to_image(data["frame_b64"])
    yield frame_img, f"Session started (frame {data['frame_index']}). Pick a direction and click Step.", gr.update(interactive=True)


def step_session(direction: str, zoom: float):
    if _session_id is None:
        raise gr.Error("Start a session first (upload an image above).")

    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/sessions/{_session_id}/step",
            json={"direction": _direction_param(direction), "zoom": float(zoom)},
            timeout=600,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise gr.Error(f"Step failed: {exc}") from exc

    data = resp.json()
    frame_img = _b64_to_image(data["frame_b64"])
    return frame_img, f"Frame {data['frame_index']}."


with gr.Blocks(title="tt-waypoint") as demo:
    gr.Markdown("# tt-waypoint")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            seed_image = gr.Image(label="Starting image", type="pil")
            start_btn = gr.Button("Start session", variant="primary")
            gr.Markdown("---")
            direction = gr.Radio(
                _DIRECTIONS, value="Forward", label="Direction",
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
