# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Pins the weights revision and download filter in `waypoint_ttnn/session.py`.
Pure Python, no hardware: it never imports ttnn and never opens a device.
`session.py` imports ttnn lazily inside `ensure_waypoint_models()`, so importing the
module is safe, and `snapshot_download` is replaced with a recorder, so no network
access happens either.

What these tests guard, and at which layer:

* `test_snapshot_download_receives_pin_and_filter` checks the *call* that reaches
  huggingface_hub. It does not re-check the arithmetic of `weights_revision()`. A
  pin that exists as a constant but never reaches the hub call is exactly the
  failure this is meant to catch.
* `test_allow_patterns_cover_every_file_the_loader_reads` parses session.py's own
  source for the `os.path.join(snapshot_dir, ...)` reads. If someone adds a new read
  (say `vae/config.json`) without widening the filter, a served bundle would crash
  with FileNotFoundError at load time. This test fails first.

Run: `python -m pytest waypoint_ttnn/tests/test_weights_pin.py` from the repo root.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from waypoint_ttnn import session  # noqa: E402  (after sys.path setup)

SESSION_SRC = REPO_ROOT / "waypoint_ttnn" / "session.py"


@pytest.fixture
def recorded_download(monkeypatch):
    """Swap `huggingface_hub.snapshot_download` for a stub that records its kwargs.
    `_resolve_snapshot_dir()` imports it at call time (`from huggingface_hub import
    snapshot_download`), so patching the attribute on the module object is enough.
    A stub module is injected when huggingface_hub isn't installed."""
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/nonexistent/snapshot"

    hub = sys.modules.get("huggingface_hub")
    if hub is None:
        try:
            import huggingface_hub as hub  # noqa: F811
        except ImportError:
            hub = types.ModuleType("huggingface_hub")
            monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot_download, raising=False)
    return calls


def test_pin_is_a_full_sha():
    # A short or branch-like value would let the hub resolve something that moves.
    assert len(session.PINNED_WEIGHTS_REVISION) == 40
    int(session.PINNED_WEIGHTS_REVISION, 16)


def test_revision_defaults_to_pin(monkeypatch):
    monkeypatch.delenv(session.WEIGHTS_REVISION_ENV, raising=False)
    assert session.weights_revision() == session.PINNED_WEIGHTS_REVISION


def test_empty_env_var_still_uses_pin(monkeypatch):
    # `TT_MODEL_WEIGHTS_REVISION=` (exported but empty) must not become revision="".
    monkeypatch.setenv(session.WEIGHTS_REVISION_ENV, "")
    assert session.weights_revision() == session.PINNED_WEIGHTS_REVISION


def test_env_var_overrides_pin(monkeypatch):
    monkeypatch.setenv(session.WEIGHTS_REVISION_ENV, "deadbeef" * 5)
    assert session.weights_revision() == "deadbeef" * 5


def test_snapshot_download_receives_pin_and_filter(monkeypatch, recorded_download):
    monkeypatch.delenv(session.WEIGHTS_REVISION_ENV, raising=False)
    assert session._resolve_snapshot_dir() == "/nonexistent/snapshot"
    assert recorded_download == [
        {
            "repo_id": "Overworld/Waypoint-1.5-1B",
            "revision": session.PINNED_WEIGHTS_REVISION,
            "allow_patterns": list(session.WEIGHTS_ALLOW_PATTERNS),
        }
    ]


def test_snapshot_download_honours_env_override(monkeypatch, recorded_download):
    monkeypatch.setenv(session.WEIGHTS_REVISION_ENV, "cafe" * 10)
    session._resolve_snapshot_dir()
    assert recorded_download[0]["revision"] == "cafe" * 10


def _snapshot_relative_reads():
    """Every `os.path.join(snapshot_dir, "a", "b", ...)` in session.py, as "a/b/..."."""
    tree = ast.parse(SESSION_SRC.read_text())
    reads = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "join" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Name) and first.id == "snapshot_dir"):
            continue
        parts = [a.value for a in node.args[1:] if isinstance(a, ast.Constant)]
        assert len(parts) == len(node.args) - 1, "non-literal path segment; update this test"
        reads.add("/".join(parts))
    return reads


def test_allow_patterns_cover_every_file_the_loader_reads():
    reads = _snapshot_relative_reads()
    # Sanity: the parser really found the loader's reads (a parser that finds
    # nothing would make the subset check below pass vacuously).
    assert reads == {
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    }
    assert reads == set(session.WEIGHTS_ALLOW_PATTERNS)
