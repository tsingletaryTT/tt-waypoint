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

# --- Weights: which repo, which exact revision, which files --------------------------
#
#: Upstream weights repo. Never embedded in this package; fetched into the HF cache.
WEIGHTS_REPO_ID = "Overworld/Waypoint-1.5-1B"

#: The upstream commit this port was brought up, benchmarked and hardware-verified
#: against (HF API `sha` on 2026-09-27; upstream lastModified 2026-07-16, i.e. before
#: the 2026-09-09 bring-up started, so this is the revision every number in
#: BRINGUP_LOG.md was measured on). Pinned so a later upstream push can't silently
#: change what a served bundle loads.
PINNED_WEIGHTS_REVISION = "391f92827075edcf4a8b3c8a2ddae010698f8636"

#: tt-model-manager's v6 `run.sh` exports this when the bundle manifest carries a
#: weights revision, so the manifest and the code can't disagree. When it is unset
#: (dev checkout, bare `uvicorn`), the pin above applies. An empty string counts as
#: unset -- `or`, not a `.get()` default -- so `TT_MODEL_WEIGHTS_REVISION=` can't
#: accidentally resolve `main`.
WEIGHTS_REVISION_ENV = "TT_MODEL_WEIGHTS_REVISION"

#: The ONLY files the served path reads -- see `ensure_waypoint_models()` and
#: `_load_config()` below, which open exactly these three paths and nothing else under
#: the snapshot (verified by grepping the served closure -- session.py, server/app.py,
#: tt/*.py -- for every file open / from_pretrained / hub call; none exist outside this
#: module). The upstream repo is ~11.4 GB; this filter skips the root
#: `model.safetensors` (3.72 GB, an alternative packaging of the same model that this
#: port never loads), `assets/` (~250 MB of demo media), and the upstream Python
#: sources (`transformer/model.py`, `vae/ae_model.py`, `modular_*`), which this port
#: replaces rather than imports. `vae/config.json` is deliberately NOT included: the
#: VAE's shape is hardcoded in tt/vae_{encoder,decoder}.py, not read from config.
#: Keep this list and the two `os.path.join(...)` reads below in lockstep --
#: `tests/test_weights_pin.py` asserts they match. A repackage should pass the same
#: list to `tt-model package-thin` so the bundle's own pull fetches the same subset.
WEIGHTS_ALLOW_PATTERNS = (
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)


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

        import os

        import ttnn
        from safetensors.torch import load_file

        from waypoint_ttnn.tt.full_model import WaypointWorldModel
        from waypoint_ttnn.tt.vae_decoder import WaypointVAEDecoder
        from waypoint_ttnn.tt.vae_encoder import WaypointVAEEncoder

        device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), l1_small_size=L1_SMALL_SIZE)
        try:
            snapshot_dir = _resolve_snapshot_dir()
            hf_config = _load_config(snapshot_dir)
            # Both paths must be covered by WEIGHTS_ALLOW_PATTERNS, or the filtered
            # snapshot won't contain them.
            tf_weights = os.path.join(snapshot_dir, "transformer", "diffusion_pytorch_model.safetensors")
            vae_weights = os.path.join(snapshot_dir, "vae", "diffusion_pytorch_model.safetensors")
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


def _resolve_snapshot_dir() -> str:
    """Downloads (or reuses an already-cached) local snapshot of the weights repo,
    returning its directory. Uses huggingface_hub's own resolution rather than a
    hardcoded `~/.cache/huggingface/...` path -- that path doesn't exist inside the
    served container, which mounts the HF cache at `/hf` (`HF_HOME=/hf`, set by
    tt-model-manager's own container.py) instead. A real bug hit on the first real
    `tt-model serve` attempt: `glob.glob(...)[0]` raised `IndexError` because the
    hardcoded host path was simply absent in the container.

    Fetches a pinned revision (`weights_revision()`), and only the files the served
    path reads (`WEIGHTS_ALLOW_PATTERNS`). Before 0.1.1 this downloaded `main` in full,
    ~4 GB more than needed, and would have silently followed any upstream push."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=WEIGHTS_REPO_ID,
        revision=weights_revision(),
        allow_patterns=list(WEIGHTS_ALLOW_PATTERNS),
    )


def weights_revision() -> str:
    """The weights revision to load: `$TT_MODEL_WEIGHTS_REVISION` if set and non-empty
    (exported by tt-model-manager's v6 run.sh from the bundle manifest), else the
    revision this port was verified on. Pure function of the environment -- no hub call,
    no device -- so it is unit-testable without hardware."""
    import os

    return os.environ.get(WEIGHTS_REVISION_ENV) or PINNED_WEIGHTS_REVISION


def _load_config(snapshot_dir: str):
    """Loads the real transformer config the way every test in this repo does --
    `AutoConfig`-free, since the shared venv's diffusers predates it (see CLAUDE.md)."""
    import json
    import os

    with open(os.path.join(snapshot_dir, "transformer", "config.json")) as f:
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
