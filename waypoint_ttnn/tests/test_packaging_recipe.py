# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Guards for the v6 thin packaging recipe in `packaging/`.

Pure Python. It never imports ttnn, never opens a device and needs no network. Run from the
repo root:  python -m pytest waypoint_ttnn/tests/test_packaging_recipe.py

Each test is placed at the layer that can fail:

* The closure tests run `packaging/closure.py` on real and doctored source trees.
* The guard tests run the exact `--verify` statements that `install.sh` will run, against
  bundle layouts they must accept and layouts they must reject.
* `test_script_invokes_package_thin_as_documented` runs the real `package-thin.sh` with a stub
  `tt-model` that records its argv. It checks the call that would reach the tool: no repo id
  (which would upload), the weights pin and file filter read from session.py, the guards, and
  no whole-tree models wheel.
* `test_real_tt_model_stages_the_canonical_shape` runs the real `tt-model` (skipped when it is
  not installed) and applies `inspect_stage.py` to the output. It checks the staged files.

Each test here was seen to fail when the behaviour it covers was broken; see CLAUDE.md.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "packaging"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PKG))

import build_deps_wheel  # noqa: E402
import closure  # noqa: E402
import guards  # noqa: E402
import inspect_stage  # noqa: E402
from waypoint_ttnn import session  # noqa: E402

# The files the server can import, and nothing else.
EXPECTED_CLOSURE = {
    "waypoint_ttnn/__init__.py",
    "waypoint_ttnn/server/__init__.py",
    "waypoint_ttnn/server/app.py",
    "waypoint_ttnn/session.py",
    "waypoint_ttnn/tt/__init__.py",
    "waypoint_ttnn/tt/full_model.py",
    "waypoint_ttnn/tt/functional_decoder.py",
    "waypoint_ttnn/tt/generation_loop.py",
    "waypoint_ttnn/tt/lightweightmodule.py",
    "waypoint_ttnn/tt/rope.py",
    "waypoint_ttnn/tt/vae_decoder.py",
    "waypoint_ttnn/tt/vae_encoder.py",
}


# --------------------------------------------------------------------------- closure
def _copy_package(tmp_path: Path) -> Path:
    """A throwaway copy of the waypoint_ttnn package that a test may damage."""
    shutil.copytree(REPO_ROOT / "waypoint_ttnn", tmp_path / "waypoint_ttnn",
                    ignore=shutil.ignore_patterns("__pycache__", "tests", "ref_activations"))
    return tmp_path


def test_closure_is_exactly_the_served_modules():
    files = set(closure.served_closure(REPO_ROOT))
    assert files == EXPECTED_CLOSURE
    # The bring-up scripts and tests must never ship.
    assert not [f for f in files if "capture_" in f or "stage" in Path(f).name
                or "benchmark" in f or "/tests/" in f]


def test_closure_has_no_tt_metal_models_dependency():
    """The old bundle needed a wheel for `models.common.lightweightmodule`. The model code
    must not import any `models` package any more."""
    for rel, src in closure.served_closure(REPO_ROOT).items():
        text = src.read_text()
        assert "from models" not in text and "import models" not in text, rel


def test_closure_rejects_a_models_import(tmp_path):
    root = _copy_package(tmp_path)
    rope = root / "waypoint_ttnn/tt/rope.py"
    rope.write_text(rope.read_text() + "\nfrom models.common.lightweightmodule import LightweightModule\n")
    with pytest.raises(closure.ClosureError, match="models"):
        closure.served_closure(root)


def test_closure_rejects_a_dynamic_import(tmp_path):
    root = _copy_package(tmp_path)
    rope = root / "waypoint_ttnn/tt/rope.py"
    rope.write_text(rope.read_text() + "\nimport importlib\nimportlib.import_module('json')\n")
    with pytest.raises(closure.ClosureError, match="dynamic"):
        closure.served_closure(root)


def test_closure_rejects_an_unresolved_package_import(tmp_path):
    root = _copy_package(tmp_path)
    (root / "waypoint_ttnn/tt/rope.py").write_text(
        (root / "waypoint_ttnn/tt/rope.py").read_text() + "\nimport waypoint_ttnn.tt.not_there\n")
    with pytest.raises(closure.ClosureError, match="not_there"):
        closure.served_closure(root)


# --------------------------------------------------------------------------- guards
def _bundle(tmp_path: Path, *, with_app=True, with_models=False) -> Path:
    """A fake installed bundle: `<b>/venv` is sys.prefix, `<b>/extra_code` holds the code."""
    b = tmp_path / "bundle"
    (b / "venv").mkdir(parents=True)
    if with_app:
        shutil.copytree(REPO_ROOT / "waypoint_ttnn", b / "extra_code" / "waypoint_ttnn",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))
    if with_models:
        (b / "extra_code" / "models").mkdir(parents=True, exist_ok=True)
    return b


def _run_guard(name: str, bundle: Path, monkeypatch, versions: dict | None = None,
               installed: tuple = (), cwd_first: bool = False) -> None:
    """Run one guard statement in-process against a fake layout.

    `installed` names top-level packages the fake venv "has". `find_spec` answers None for the
    packages the guards probe (`models`, `tt_transformers`) unless they are listed, because the
    test machine may have a tt-metal checkout on its path that says nothing about the bundle.
    """
    import importlib.util as util
    real_find_spec = util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("models", "tt_transformers"):
            return object() if name in installed else None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(util, "find_spec", fake_find_spec)
    monkeypatch.setattr(sys, "prefix", str(bundle / "venv"))
    monkeypatch.setattr(sys, "path", list(sys.path))   # a guard may rebind sys.path; restore it after
    monkeypatch.syspath_prepend(str(bundle / "extra_code"))
    if cwd_first:
        sys.path.insert(0, "")   # `python -c` puts the current directory ahead of PYTHONPATH
    # Forget any earlier import of the package so the guard resolves it from this layout.
    for mod in [m for m in sys.modules if m == "waypoint_ttnn" or m.startswith("waypoint_ttnn.")]:
        monkeypatch.delitem(sys.modules, mod)
    if versions is not None:
        monkeypatch.setattr(importlib.metadata, "version", lambda n: versions[n])
    exec(guards.GUARDS[name], {})


def test_guards_are_single_line_and_never_import_ttnn():
    for name, stmt in guards.GUARDS.items():
        assert "\n" not in stmt, name
        assert "import ttnn" not in stmt.replace("'ttnn' not in sys.modules", ""), name


def test_guard_own_code_accepts_the_canonical_layout(tmp_path, monkeypatch):
    _run_guard("own-code-in-bundle", _bundle(tmp_path), monkeypatch)


def test_guard_own_code_rejects_missing_code(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="extra_code/waypoint_ttnn missing"):
        _run_guard("own-code-in-bundle", _bundle(tmp_path, with_app=False), monkeypatch)


def test_guard_own_code_rejects_an_installed_tt_transformers(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="tt_transformers is installed"):
        _run_guard("own-code-in-bundle", _bundle(tmp_path), monkeypatch, installed=("tt_transformers",))


def test_guard_own_code_rejects_an_installed_models_package(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="top-level models package"):
        _run_guard("own-code-in-bundle", _bundle(tmp_path), monkeypatch, installed=("models",))


def test_guard_own_code_rejects_a_shipped_models_tree(tmp_path, monkeypatch):
    with pytest.raises(AssertionError, match="extra_code/models"):
        _run_guard("own-code-in-bundle", _bundle(tmp_path, with_models=True), monkeypatch)


def test_guard_imports_without_device_accepts_and_resolves_from_extra_code(tmp_path, monkeypatch):
    # `ttnn` must not be loaded for this to be a fair test; the test process never imports it.
    assert "ttnn" not in sys.modules
    _run_guard("imports-without-device", _bundle(tmp_path), monkeypatch)


def test_guard_imports_without_device_rejects_a_top_level_ttnn_import(tmp_path, monkeypatch):
    b = _bundle(tmp_path)
    # A module-level `import ttnn` in the served code would open the device runtime at install.
    # A stand-in module named ttnn on the path shows the guard sees it without touching hardware.
    (b / "extra_code" / "ttnn.py").write_text("")
    app = b / "extra_code" / "waypoint_ttnn" / "server" / "app.py"
    app.write_text(app.read_text() + "\nimport ttnn\n")
    try:
        with pytest.raises(AssertionError, match="imported ttnn"):
            _run_guard("imports-without-device", b, monkeypatch)
    finally:
        sys.modules.pop("ttnn", None)


def test_guard_imports_without_device_ignores_the_current_directory(tmp_path, monkeypatch):
    """`python -c` puts '' (the cwd) first on sys.path. If the cwd holds a checkout of this repo,
    the guard must check the shipped copy."""
    b = _bundle(tmp_path)
    monkeypatch.chdir(REPO_ROOT)               # cwd contains waypoint_ttnn/
    _run_guard("imports-without-device", b, monkeypatch, cwd_first=True)


def test_guard_stock_ttnn(tmp_path, monkeypatch):
    b = _bundle(tmp_path)
    _run_guard("stock-ttnn", b, monkeypatch, {"ttnn": guards.TTNN_VERSION})
    with pytest.raises(AssertionError, match="not the stock"):
        _run_guard("stock-ttnn", b, monkeypatch, {"ttnn": guards.TTNN_VERSION + "+local.abc"})


def test_guard_current_network_stack(tmp_path, monkeypatch):
    b = _bundle(tmp_path)
    good = {"certifi": "2026.7.22", "urllib3": "2.8.0", "requests": "2.34.2"}
    _run_guard("current-network-stack", b, monkeypatch, good)
    # The versions the PyTorch CPU index would install.
    stale = {"certifi": "2022.12.7", "urllib3": "1.26.13", "requests": "2.28.1"}
    for pkg in stale:
        with pytest.raises(AssertionError, match=f"stale {pkg}"):
            _run_guard("current-network-stack", b, monkeypatch, {**good, pkg: stale[pkg]})


# --------------------------------------------------------------------------- recipe inputs
def test_ttnn_pin_is_consistent_everywhere():
    requirements = (PKG / "requirements.txt").read_text()
    constraints = (PKG / "constraints.txt").read_text()
    assert f"\nttnn=={guards.TTNN_VERSION}" in "\n" + requirements
    assert f"\nttnn=={guards.TTNN_VERSION}\n" in "\n" + constraints
    # No `+local` tag for ttnn anywhere: the engine is the stock PyPI wheel.
    assert "ttnn==" + guards.TTNN_VERSION + "+" not in requirements + constraints


def test_requirements_has_no_index_options_and_no_models_tree():
    lines = [ln.split("#", 1)[0].strip() for ln in (PKG / "requirements.txt").read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    assert not [ln for ln in lines if ln.startswith("-")]
    # huggingface_hub reaches the install through the serve-deps wheel (a later step).
    assert not [ln for ln in lines if ln.startswith(("huggingface", "safetensors", "requests"))]
    assert not [ln for ln in lines if "tt-metal-models" in ln or "tt_transformers" in ln.replace("-", "_")]


def test_constraints_are_exact_pins():
    pins = [ln for ln in (PKG / "constraints.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert len(pins) > 30
    assert all("==" in ln and not any(c in ln for c in "<>~!") for ln in pins)


def test_deps_wheel_is_metadata_only_and_reproducible(tmp_path):
    first = build_deps_wheel.build(tmp_path / "a", "9.9.9")
    second = build_deps_wheel.build(tmp_path / "b", "9.9.9")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as z:
        names = z.namelist()
        assert all(".dist-info/" in n for n in names), names   # no importable module inside
        meta = z.read("tt_waypoint_serve_deps-9.9.9.dist-info/METADATA").decode()
    for requirement in build_deps_wheel.REQUIRES:
        assert f"Requires-Dist: {requirement}" in meta


def test_package_version_was_bumped_past_the_published_bundle():
    text = (REPO_ROOT / "pyproject.toml").read_text()
    # 0.1.1 is the wheel-based bundle on episod/tt-waypoint. The canonical-shape bundle is newer.
    assert 'version = "0.1.2"' in text


def test_card_does_not_name_the_lease_tool():
    # House rule: cards and tt-model-manager docs never mention it.
    assert "gozer" not in (PKG / "README.hf.md").read_text().lower()


# --------------------------------------------------------------------------- the script
def _stub_tt_model(tmp_path: Path) -> Path:
    """A `tt-model` that writes its argv, one token per line, and stages nothing."""
    log = tmp_path / "argv.txt"
    stub = tmp_path / "tt-model"
    stub.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{log}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


def test_script_invokes_package_thin_as_documented(tmp_path):
    stub = _stub_tt_model(tmp_path)
    out = tmp_path / "bundle"
    env = {**os.environ, "TT_MODEL": str(stub), "SKIP_INSPECT": "1"}
    subprocess.run([str(PKG / "package-thin.sh"), "--out", str(out)], check=True, env=env,
                   capture_output=True, text=True)
    argv = (tmp_path / "argv.txt").read_text().split("\n")[:-1]

    # No positional repo id: with one, package-thin uploads. Every flag takes one value here, so
    # after the subcommand the tokens must alternate flag, value.
    assert argv[0] == "package-thin"
    flags, values = argv[1::2], argv[2::2]
    assert all(f.startswith("--") for f in flags), f"a positional token slipped in: {argv}"
    assert not [v for v in values if v.startswith("--")], argv
    pairs = list(zip(flags, values))
    get = lambda flag: [v for f, v in pairs if f == flag]  # noqa: E731

    assert get("--out") == [str(out)]
    assert get("--kind") == ["tt-dit-server"]
    assert get("--app") == ["waypoint_ttnn.server.app:app"]
    # Canonical shape: stock ttnn pins, constraints, own code as source, no whole-tree wheel.
    assert not get("--models-wheel")
    assert len(get("--ops-wheel")) == 1
    assert Path(get("--ops-wheel")[0]).name.startswith("tt_waypoint_serve_deps-")
    assert Path(get("--constraints")[0]) == PKG / "constraints.txt"
    assert Path(get("--requirements")[0]) == PKG / "requirements.txt"
    assert (Path(get("--extra-code")[0]) / "waypoint_ttnn" / "session.py").is_file()
    assert not (Path(get("--extra-code")[0]) / "models").exists()
    # The manifest's weights block comes from the code, so it cannot disagree with session.py.
    assert get("--weights") == [session.WEIGHTS_REPO_ID]
    assert get("--weights-revision") == [session.PINNED_WEIGHTS_REVISION]
    assert get("--weights-allow") == list(session.WEIGHTS_ALLOW_PATTERNS)
    # Every install guard is passed, verbatim.
    assert get("--verify") == list(guards.GUARDS.values())
    # --provenance is a path to a JSON file that exists and names the commit.
    prov = json.loads(Path(get("--provenance")[0]).read_text())
    assert prov["source"]["commit"] and prov["weights"]["revision"] == session.PINNED_WEIGHTS_REVISION
    assert Path(get("--readme")[0]) == PKG / "README.hf.md"
    # No parsers: the server has no tokens or reasoning channel.
    assert not get("--tool-parser") and not get("--reasoning-parser")


@pytest.mark.skipif(shutil.which("tt-model") is None, reason="tt-model is not installed")
def test_real_tt_model_stages_the_canonical_shape(tmp_path):
    out = tmp_path / "bundle"
    done = subprocess.run([str(PKG / "package-thin.sh"), "--out", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert inspect_stage.inspect(out, REPO_ROOT) == []

    # The inspector must reject a bundle that is not in the shape. Each change below is one of
    # the wrong answers the canonical-shape docs list.
    models_dir = out / "extra_code" / "models"
    models_dir.mkdir()
    assert any("extra_code/models" in p or "import closure" in p for p in inspect_stage.inspect(out, REPO_ROOT))
    models_dir.rmdir()
    (out / "wheels" / "tt_transformers-1.0-py3-none-any.whl").write_bytes(b"")
    assert any("wheels/" in p for p in inspect_stage.inspect(out, REPO_ROOT))
    (out / "wheels" / "tt_transformers-1.0-py3-none-any.whl").unlink()
    req = out / "requirements.txt"
    req.write_text("--extra-index-url https://example.invalid/simple\n" + req.read_text())
    assert any("option line" in p for p in inspect_stage.inspect(out, REPO_ROOT))
