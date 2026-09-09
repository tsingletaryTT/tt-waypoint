# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Module-level mesh-device + model singleton, modeled on tt-skyreels'
`skyreels_ttnn/session.py` (itself modeled on tt-animatediff's). Manages the TTNN mesh
device, the three loaded models (`WaypointWorldModel`, `WaypointVAEEncoder`,
`WaypointVAEDecoder`), and the SINGLE active interactive `WaypointGenerator` session, so
the Gradio app doesn't pay weight-load cost more than once per process.

Unlike tt-skyreels (stateless: every call is an independent one-shot generation), this
model is inherently STATEFUL -- an interactive session holds a live KV cache across many
`step()` calls. This module therefore holds at most ONE active `WaypointGenerator` at a
time; starting a new session (a new seed image) discards the previous one's cache state.
A real multi-tenant server would need one generator per session id, not a module global --
out of scope for this first demo app (see PORT_PLAN.md Stage 6's serving-contract note).
"""

import threading
from typing import Optional

_lock = threading.Lock()
_device = None
_world_model = None
_vae_encoder = None
_vae_decoder = None
_hf_config = None
_generator = None

MESH_SHAPE = (1, 1)
L1_SMALL_SIZE = 21760  # conv2d's halo op needs this set; see vae_decoder.py's own note.


class _Cfg:
    def __init__(self, d):
        self.__dict__.update(d)


def ensure_waypoint_models():
    """Open the mesh device and load all three TTNN models (once per process).

    Returns:
        (device, world_model, vae_encoder, vae_decoder, hf_config)
    """
    global _device, _world_model, _vae_encoder, _vae_decoder, _hf_config
    if _device is not None:
        return _device, _world_model, _vae_encoder, _vae_decoder, _hf_config

    with _lock:
        if _device is not None:
            return _device, _world_model, _vae_encoder, _vae_decoder, _hf_config

        import glob

        import ttnn
        from safetensors.torch import load_file

        from waypoint_ttnn.tt.full_model import WaypointWorldModel
        from waypoint_ttnn.tt.vae_decoder import WaypointVAEDecoder
        from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder

        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), l1_small_size=L1_SMALL_SIZE)
        try:
            hf_config = _load_config()
            tf_weights = glob.glob(
                "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
                "snapshots/*/transformer/diffusion_pytorch_model.safetensors"
            )[0]
            vae_weights = glob.glob(
                "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
                "snapshots/*/vae/diffusion_pytorch_model.safetensors"
            )[0]
            tf_state_dict = load_file(tf_weights)
            vae_state_dict = load_file(vae_weights)

            world_model = WaypointWorldModel.from_state_dict(tf_state_dict, hf_config=hf_config, mesh_device=device)
            vae_encoder = WaypointVAEEncoder.from_state_dict(vae_state_dict, mesh_device=device)
            vae_decoder = WaypointVAEDecoder.from_state_dict(vae_state_dict, mesh_device=device)
        except Exception:
            _close_device(device)
            raise

        _world_model = world_model
        _vae_encoder = vae_encoder
        _vae_decoder = vae_decoder
        _hf_config = hf_config
        _device = device
        return _device, _world_model, _vae_encoder, _vae_decoder, _hf_config


def start_session():
    """Ends any current interactive session and returns a fresh WaypointGenerator
    (models loaded lazily via ensure_waypoint_models() if not already). Call `.seed()`
    on the result with a real image before the first `.step()`."""
    global _generator
    device, world_model, vae_encoder, vae_decoder, hf_config = ensure_waypoint_models()

    from waypoint_ttnn.tt.generation_loop import WaypointGenerator

    with _lock:
        _generator = WaypointGenerator(world_model, vae_encoder, vae_decoder, hf_config, device)
        return _generator


def current_session() -> Optional["WaypointGenerator"]:  # noqa: F821 -- forward ref, avoids importing ttnn at module load
    return _generator


def _load_config():
    """Loads the real transformer config the way every test in this repo does --
    `AutoConfig`-free, since the shared venv's diffusers predates it (see CLAUDE.md)."""
    import glob
    import json

    cfg_path = glob.glob(
        "/home/ttuser/.cache/huggingface/hub/models--Overworld--Waypoint-1.5-1B/"
        "snapshots/*/transformer/config.json"
    )[0]
    with open(cfg_path) as f:
        return _Cfg(json.load(f))


def close() -> None:
    """Closes the TTNN mesh device. Rarely needed -- process exit reclaims it."""
    global _device, _world_model, _vae_encoder, _vae_decoder, _hf_config, _generator
    with _lock:
        if _device is None:
            return
        _close_device(_device)
        _device = None
        _world_model = None
        _vae_encoder = None
        _vae_decoder = None
        _hf_config = None
        _generator = None


def _close_device(device) -> None:
    try:
        import ttnn

        ttnn.close_mesh_device(device)
    except Exception:
        pass
