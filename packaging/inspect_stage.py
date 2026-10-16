# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Inspect a staged tt-waypoint bundle and fail if it is not in the canonical shape.

`packaging/package-thin.sh` runs this on the folder `tt-model package-thin --out` produced.
It reads files only: it never runs `install.sh`, never imports ttnn and never opens a device.
The one subprocess is `bash`, which expands run.sh's `CMD=(...)` line to show the argument
tokens the server will be started with.

Usage: python3 packaging/inspect_stage.py <bundle_dir> [--repo-root <repo>]
Exit status 0 when every check passes, 1 otherwise. Each failure is printed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import closure  # noqa: E402
import guards  # noqa: E402


def inspect(bundle: Path, repo_root: Path) -> List[str]:
    """Return a list of problems; an empty list means the bundle is in the canonical shape."""
    problems: List[str] = []

    def check(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    # The code is the source of truth for the weights pin and the file filter.
    sys.path.insert(0, str(repo_root))
    from waypoint_ttnn import session  # safe: session.py imports ttnn lazily
    assert "ttnn" not in sys.modules

    manifest = json.loads((bundle / "tt_kernel_manifest.json").read_text())
    deps = manifest["deps"]

    # --- manifest ---------------------------------------------------------------------
    check(manifest["schema_version"] == "6", "schema_version is not 6")
    check(deps["kind"] == "tt-dit-server", f"deps.kind is {deps['kind']!r}")
    check(deps["app"] == "waypoint_ttnn.server.app:app", f"deps.app is {deps['app']!r}")
    check(deps["extra_code_dir"] == "extra_code", "deps.extra_code_dir is not 'extra_code'")
    check(deps["vllm"] is None, "a vLLM block is present; this bundle serves with uvicorn")
    check(manifest["tt_metal_version"] == guards.TTNN_VERSION,
          f"tt_metal_version {manifest['tt_metal_version']!r} != ttnn pin {guards.TTNN_VERSION}")
    check(manifest["weights"]["repo_id"] == session.WEIGHTS_REPO_ID, "weights repo differs from session.py")
    check(manifest["weights"]["revision"] == session.PINNED_WEIGHTS_REVISION,
          "manifest weights revision differs from session.PINNED_WEIGHTS_REVISION")
    check(manifest["weights"]["allow_patterns"] == list(session.WEIGHTS_ALLOW_PATTERNS),
          "manifest allow_patterns differ from session.WEIGHTS_ALLOW_PATTERNS")
    # No model-library wheel, and no custom op: this model has no generic_op wheel.
    check(deps["models_wheels"] == [], f"models_wheels is {deps['models_wheels']} (a whole-tree wheel?)")
    check(deps["custom_ops"] == [] if "custom_ops" in deps else True,
          f"deps.custom_ops lists {deps.get('custom_ops')}; only generic_op wheels belong there")
    check(len(deps["wheels"]) == 1 and re.fullmatch(
        r"wheels/tt_waypoint_serve_deps-[0-9.]+-py3-none-any\.whl", deps["wheels"][0]) is not None,
        f"deps.wheels is {deps['wheels']}; expected only the serve-deps wheel")
    check(deps["verify"] == list(guards.GUARDS.values()), "deps.verify differs from packaging/guards.py")
    check(manifest.get("capabilities") is None,
          "capabilities are set; this server has no tool or reasoning parser")

    # --- wheels/ ---------------------------------------------------------------------
    shipped = sorted(p.name for p in (bundle / "wheels").iterdir())
    check(len(shipped) == 1, f"wheels/ holds {shipped}")
    for name in shipped:
        check(not re.match(r"(ttnn|tt_transformers|tt_metal_models|tt_waypoint_models_closure|waypoint_ttnn)-", name),
              f"wheels/{name} is an engine, tooling or whole-tree model wheel")

    # --- requirements.txt and constraints.txt ---------------------------------------------
    req_lines = [ln.split("#", 1)[0].strip() for ln in (bundle / "requirements.txt").read_text().splitlines()]
    req_lines = [ln for ln in req_lines if ln]
    check(not any(ln.startswith("-") for ln in req_lines), "requirements.txt has an option line (index?)")
    check(f"ttnn=={guards.TTNN_VERSION}" in req_lines, "requirements.txt does not pin stock ttnn")
    check(not any("+" in ln and not ln.startswith("torch==") for ln in req_lines),
          "a +local version tag in requirements.txt")
    check(not any(re.match(r"(tt[-_]transformers|tt[-_]metal[-_]models)", ln) for ln in req_lines),
          "requirements.txt names a tt-transformers or models-tree package")
    pins = {}
    for ln in (bundle / "constraints.txt").read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            m = re.fullmatch(r"([A-Za-z0-9_.-]+)==(\S+)", ln)
            check(m is not None, f"constraints.txt line is not an exact pin: {ln!r}")
            if m:
                pins[re.sub(r"[-_.]+", "-", m.group(1).lower())] = m.group(2)
    for needed in ("ttnn", "torch", "huggingface-hub", "safetensors", "requests", "certifi", "urllib3", "fastapi"):
        check(needed in pins, f"constraints.txt does not pin {needed}")
    check(pins.get("ttnn") == guards.TTNN_VERSION, "constraints.txt ttnn pin differs from the guard's version")
    check([n for n, v in pins.items() if "+" in v] == ["torch"], "unexpected +local pins in constraints.txt")

    # --- shipped code ---------------------------------------------------------------------
    expected = set(closure.served_closure(repo_root))
    on_disk = {p.relative_to(bundle / "extra_code").as_posix()
               for p in (bundle / "extra_code").rglob("*") if p.is_file()}
    check(on_disk == expected, f"extra_code differs from the import closure: "
          f"extra={sorted(on_disk - expected)} missing={sorted(expected - on_disk)}")
    check(not (bundle / "extra_code" / "models").exists(), "extra_code/models exists")
    for rel in expected:
        check((bundle / "extra_code" / rel).read_bytes() == (repo_root / rel).read_bytes(),
              f"extra_code/{rel} differs from the repo file")

    # --- run.sh -------------------------------------------------------------------------
    run_sh = (bundle / "run.sh").read_text()
    pp = re.search(r'^export PYTHONPATH="([^"]*)"', run_sh, re.M)
    check(pp is not None and pp.group(1).startswith("$HERE/extra_code:"),
          f"run.sh PYTHONPATH does not start with extra_code/: {pp.group(1) if pp else None!r}")
    check(f'TT_MODEL_WEIGHTS_REVISION:-{session.PINNED_WEIGHTS_REVISION}}}' in run_sh,
          "run.sh does not export the pinned weights revision")
    cmd = re.search(r"^CMD=\(.*\)$", run_sh, re.M)
    check(cmd is not None, "run.sh has no CMD=(...) line")
    if cmd:
        # Let bash expand the array. Each element is one argv token.
        script = f'PYBIN=/venv/bin/python; PORT=1234; set --; {cmd.group(0)}; printf "%s\\n" "${{CMD[@]}}"'
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout.split("\n")[:-1]
        check(out == ["/venv/bin/python", "-m", "uvicorn", "--host", "0.0.0.0", "--port", "1234",
                      "--lifespan", "on", "waypoint_ttnn.server.app:app"],
              f"run.sh argv tokens are {out}")
        print("run.sh argv:", out)

    # --- install.sh -----------------------------------------------------------------------
    install_sh = (bundle / "install.sh").read_text()
    check('--constraint "$HERE/constraints.txt"' in install_sh, "install.sh does not apply constraints.txt")
    check(install_sh.count("--extra-index-url") == 1, "install.sh uses an index option other than the single torch one")
    check("tt_waypoint_serve_deps" in install_sh, "install.sh does not install the serve-deps wheel")
    check(install_sh.count('"$VENV/bin/python" -c') == len(guards.GUARDS), "install.sh does not run every --verify guard")
    return problems


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--repo-root", type=Path, default=HERE.parent)
    args = ap.parse_args(argv)
    problems = inspect(args.bundle, args.repo_root)
    for p in problems:
        print(f"FAIL  {p}", file=sys.stderr)
    if not problems:
        print(f"inspect_stage: {args.bundle} is in the canonical shape")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
