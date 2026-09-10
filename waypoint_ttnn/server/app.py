# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""ASGI app serving tt-waypoint (Waypoint-1.5-1B, interactive world model), for
tt-model-manager's ``tt-dit-server`` kind.

Modeled on tt-skyreels'/tt-animatediff's ``server/app.py`` (the reference
implementations for this kind), but the SHAPE of the API is necessarily different: this
kind only means "a diffusion-family model behind a small HTTP surface launched with
uvicorn" -- it does not prescribe a request/response contract, and Waypoint-1.5-1B isn't
a one-shot "type a prompt, get a video" model. It's STATEFUL: a session holds a live
per-layer KV cache across many steps, seeded once from a real image and then advanced
one generated frame at a time by mouse/scroll controls (see generation_loop.py's module
docstring for the exact seed/step protocol this wraps).

SINGLE ACTIVE SESSION PER PROCESS -- A DELIBERATE SCOPE CHOICE, NOT AN OVERSIGHT
------------------------------------------------------------------------------
The current architecture ties a session's KV-cache state to the SAME `FunctionalDecoder`
instances that hold the model's own loaded weights (see functional_decoder.py -- `self.
cache` is constructed once in `__init__`, alongside `self.w`). True concurrent
multi-session serving would need those separated (weights shared, cache state threaded
per-call) -- a real refactor of already-hardware-verified code, not attempted here.
Starting a new session discards any session already in progress, matching
`waypoint_ttnn/session.py`'s own `start_session()` contract. This mirrors tt-skyreels'
own "one denoise loop at a time, the pipeline owns the mesh" simplicity -- appropriate
for a first packaging pass of a showcase/demo model, not a claim that this is how a
high-throughput production service should work.

TWO CONTRACTS THIS FILE HAS TO HONOUR (same as tt-skyreels/tt-animatediff)
----------------------------------------------------------------------------
**1. Readiness is the lifespan.** tt-model-manager decides the server is up when uvicorn
logs "Application startup complete", printed after ASGI lifespan startup returns. Device
open and weight load belong there, nowhere else.

**2. Importing this module must not touch hardware.** `verify` lines in the manifest
import the ASGI attribute at image-build time, on a machine with no card. Every ttnn
import stays inside a function.
"""
from __future__ import annotations

import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

#: The env var carrying the resolved mesh shape (this kind's default, FLUX2_MESH_SHAPE,
#: means nothing to this model -- see tt_model_package.yaml's runtime.mesh_shape_env).
MESH_SHAPE_ENV = "WAYPOINT_MESH_SHAPE"

#: Only ever verified on a single chip (see BRINGUP_LOG.md) -- no tensor/sequence
#: parallelism has been exercised for this model's autoregressive per-frame loop.
SUPPORTED_MESH_SHAPES = {(1, 1)}

#: 8-directional mouse deltas, matching mouse_tensor's [dx, dy] contract -- same mapping
#: as app.py's Gradio UI, kept in exactly one place would be nicer, but the UI imports
#: gradio and this module must not (verify_lines imports this at build time, no card,
#: and gradio is not a runtime.packages dependency here).
DIRECTIONS = {
    "forward": (0.0, -1.0),
    "back": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "forward_left": (-0.7, -0.7),
    "forward_right": (0.7, -0.7),
    "back_left": (-0.7, 0.7),
    "back_right": (0.7, 0.7),
}


class CreateSessionRequest(BaseModel):
    #: base64-encoded PNG/JPEG bytes -- decoded and resized to the model's required
    #: pixel resolution (see generation_loop.py's latent_h/latent_w * vae_scale_factor).
    image_b64: str = Field(min_length=1)


class SessionFrameResponse(BaseModel):
    session_id: str
    frame_index: int
    #: base64-encoded PNG of the newest decoded frame.
    frame_b64: str


class StepRequest(BaseModel):
    direction: str = Field(default="forward")
    #: scroll_tensor's scalar value -- zoom in/out. No documented semantics beyond sign
    #: (see app.py's own docstring on the button_tensor mapping caveat, which applies
    #: here too: button_tensor is left all-zero, no published control-scheme mapping
    #: exists for its 256-wide one-hot vocabulary).
    zoom: float = 0.0


def mesh_shape_from_env(env: Optional[dict] = None) -> tuple:
    """Parse WAYPOINT_MESH_SHAPE ("RxC") into (rows, cols). Raises on anything not in
    SUPPORTED_MESH_SHAPES rather than silently opening whatever the string describes --
    same reasoning as tt-skyreels' mesh_shape_from_env."""
    raw = (env if env is not None else os.environ).get(MESH_SHAPE_ENV, "1x1").strip()
    try:
        rows, cols = (int(p) for p in raw.lower().split("x", 1))
    except ValueError as exc:
        raise ValueError(f"{MESH_SHAPE_ENV}={raw!r} is not a mesh shape like '1x1'") from exc
    if (rows, cols) not in SUPPORTED_MESH_SHAPES:
        raise ValueError(
            f"{MESH_SHAPE_ENV}={raw!r} is not a supported shape -- Waypoint has only "
            f"been exercised on {sorted('x'.join(map(str, s)) for s in SUPPORTED_MESH_SHAPES)}"
        )
    return (rows, cols)


def _decode_image_b64(image_b64: str, pixel_h: int, pixel_w: int):
    """base64 PNG/JPEG bytes -> [pixel_h, pixel_w, 3] uint8 torch tensor."""
    import numpy as np
    import torch
    from PIL import Image

    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((pixel_w, pixel_h))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image_b64: {exc}") from exc
    return torch.from_numpy(np.array(img))


def _frame_to_png_b64(rgb_uint8_hwc) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_uint8_hwc.numpy()).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _seed_session(state: dict, req: CreateSessionRequest) -> tuple:
    """Blocking; called in a worker thread under the device lock. Returns (session_id,
    frame_index, rgb_uint8_hwc)."""
    import torch
    import torch.nn.functional as F

    from waypoint_ttnn import session as session_module

    gen = session_module.start_session()
    vae_scale_factor = 16
    pixel_h = gen.latent_h * vae_scale_factor
    pixel_w = gen.latent_w * vae_scale_factor
    img_t = _decode_image_b64(req.image_b64, pixel_h, pixel_w)

    import ttnn

    t_down = gen.vae_encoder.t_downscale
    rgb = img_t.unsqueeze(0).expand(t_down, -1, -1, -1).contiguous().float().div(255).permute(0, 3, 1, 2)
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
    latent = gen.seed(seed_frames, mouse, button, scroll)
    rgb_out = _decode_latent(gen, latent)

    session_id = uuid.uuid4().hex
    state["session_id"] = session_id
    return session_id, gen.frame_timestamp, rgb_out


def _step_session(state: dict, req: StepRequest):
    """Blocking; called in a worker thread under the device lock."""
    import torch

    from waypoint_ttnn import session as session_module

    gen = session_module.current_session()
    if gen is None:
        raise HTTPException(status_code=409, detail="no active session -- POST /v1/sessions first")

    if req.direction not in DIRECTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"direction {req.direction!r} not in {sorted(DIRECTIONS)}",
        )
    dx, dy = DIRECTIONS[req.direction]
    mouse = torch.tensor([[[dx, dy]]], dtype=torch.float32)
    button = torch.zeros(1, 1, 256)
    scroll = torch.tensor([[[req.zoom]]], dtype=torch.float32)

    latent = gen.step(mouse, button, scroll)
    rgb_out = _decode_latent(gen, latent)
    return gen.frame_timestamp, rgb_out


def _decode_latent(gen, latent):
    """latent -> the newest of decode_frame()'s output frames, as [H,W,3] uint8."""
    import torch.nn.functional as F
    import ttnn

    out_frames = gen.decode_frame(latent)
    last = out_frames[-1]
    torch_frame = ttnn.to_torch(last).float().permute(0, 3, 1, 2)  # NHWC -> NCHW
    rgb = F.pixel_shuffle(torch_frame, gen.vae_decoder.patch_size).clamp(0, 1)
    import torch

    return (rgb[0] * 255).round().to(torch.uint8).permute(1, 2, 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Claim the mesh and load weights BEFORE the server reports ready. A failure here
    is deliberately fatal (same reasoning as tt-skyreels' lifespan)."""
    from waypoint_ttnn import session as session_module

    shape = mesh_shape_from_env()
    device, world_model, vae_encoder, vae_decoder, hf_config = await asyncio.to_thread(
        session_module.ensure_waypoint_models
    )
    app.state.engine = {
        "device": device,
        "mesh_shape": shape,
        "hf_model": os.environ.get("HF_MODEL", "unknown"),
        "mesh_device": os.environ.get("MESH_DEVICE", "unknown"),
    }
    app.state.device_lock = asyncio.Lock()
    app.state.session_state = {"session_id": None}
    try:
        yield
    finally:
        session_module.close()


app = FastAPI(title="tt-waypoint", lifespan=lifespan)


def _readiness() -> dict:
    engine = getattr(app.state, "engine", None)
    return {
        "model_ready": engine is not None,
        "model": (engine or {}).get("hf_model"),
        "mesh_device": (engine or {}).get("mesh_device"),
        "mesh_shape": "x".join(str(n) for n in (engine or {}).get("mesh_shape", ())),
    }


@app.get("/tt-liveness")
async def tt_liveness() -> dict:
    try:
        return {"status": "alive", **_readiness()}
    except Exception as exc:  # noqa: BLE001 - the one case that IS unrecoverable
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {exc}")


@app.get("/health")
async def health() -> dict:
    if not _readiness()["model_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {}


@app.get("/v1/models")
async def models() -> dict:
    engine = getattr(app.state, "engine", None)
    name = (engine or {}).get("hf_model", "unknown")
    return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "tenstorrent"}]}


@app.post("/v1/sessions", response_model=SessionFrameResponse)
async def create_session(req: CreateSessionRequest) -> SessionFrameResponse:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model is still starting")
    # Starting a new session discards any session in progress -- see module docstring.
    async with app.state.device_lock:
        session_id, frame_idx, rgb = await asyncio.to_thread(_seed_session, app.state.session_state, req)
    b64 = await asyncio.to_thread(_frame_to_png_b64, rgb)
    return SessionFrameResponse(session_id=session_id, frame_index=frame_idx, frame_b64=b64)


@app.post("/v1/sessions/{session_id}/step", response_model=SessionFrameResponse)
async def step_session(session_id: str, req: StepRequest) -> SessionFrameResponse:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model is still starting")
    current_id = app.state.session_state.get("session_id")
    if current_id is None or session_id != current_id:
        raise HTTPException(
            status_code=404,
            detail="session not found or superseded -- POST /v1/sessions to start a new one",
        )
    async with app.state.device_lock:
        frame_idx, rgb = await asyncio.to_thread(_step_session, app.state.session_state, req)
    b64 = await asyncio.to_thread(_frame_to_png_b64, rgb)
    return SessionFrameResponse(session_id=session_id, frame_index=frame_idx, frame_b64=b64)


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    current_id = app.state.session_state.get("session_id")
    if current_id == session_id:
        app.state.session_state["session_id"] = None
    return {"deleted": True}
