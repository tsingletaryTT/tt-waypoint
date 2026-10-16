#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Stage the tt-waypoint v6 thin bundle, locally, in the canonical shape.
#
#     packaging/package-thin.sh [--out DIR] [--regen-constraints]
#
# This script STAGES ONLY. It runs `tt-model package-thin --out DIR` without a repo id, so
# nothing is uploaded. Publishing is a separate, deliberate step: re-run the `tt-model
# package-thin` command that this script prints, with `<org>/<name>` added (see the end).
# The script needs no card and never imports ttnn.
#
# THE SHAPE (tt-model-manager docs/E2E_RECIPE_V6.md, "The canonical shape"):
#
#   <bundle>/
#     model.py            re-exports the ASGI app (kind=tt-dit-server has no vLLM adapter)
#     extra_code/waypoint_ttnn/   the model's OWN code, as source. run.sh puts extra_code/ first
#                                 on PYTHONPATH. Only the files the server can import are
#                                 shipped (packaging/closure.py derives the list).
#     requirements.txt    ttnn==0.78.0 (stock, from PyPI), torch CPU, HTTP stack. No index options.
#     constraints.txt     every package pinned (packaging/constraints.txt)
#     wheels/tt_waypoint_serve_deps-*.whl
#                         an empty wheel that pulls huggingface_hub and safetensors in a later
#                         install step, away from the PyTorch CPU index (see its docstring)
#     install.sh run.sh tt_kernel_manifest.json README.md PROVENANCE.json
#   weights: a pointer in the manifest (Overworld/Waypoint-1.5-1B at a pinned revision, three files)
#
# WHAT IS NOT IN THE SHAPE, AND WHY (decided from the actual imports):
#   * No tt-transformers wheel. The served code imports nothing from tt-transformers. Its only
#     tt-metal dependency was `models.common.lightweightmodule.LightweightModule` (ten lines, no
#     imports). That class now lives in waypoint_ttnn/tt/lightweightmodule.py. The old bundle
#     shipped it as a separate `tt_waypoint_models_closure` wheel; that wheel is gone.
#   * No --ops-wheel for custom kernels. The model uses stock ttnn ops only, so there is no
#     generic_op wheel. `deps.custom_ops` is correctly empty, and model.py has no ops import.
#     (The one --ops-wheel below is the serve-deps wheel, which is an install-order device.)
#   * No --tool-parser / --reasoning-parser. The server has no tokens, no chat and no reasoning
#     channel. It is a session API (POST /v1/sessions, POST .../step).
#   * No vLLM wheels. kind=tt-dit-server serves the app with uvicorn.
#
# WHERE THE INPUTS COME FROM (nothing large is committed to git):
#   * Model code: this repo, at the commit recorded in PROVENANCE.json.
#   * serve-deps wheel: built here by packaging/build_deps_wheel.py (stdlib, byte-reproducible).
#   * ttnn, torch, vLLM-free HTTP stack, huggingface_hub: installed from PyPI / the PyTorch CPU
#     index by the staged bundle's install.sh. Nothing is downloaded by this script.
#   * `tt-model`: the CLI from tt-model-manager (PATH, or TT_MODEL=/path/to/tt-model).
#
# FLAGS / ENVIRONMENT
#   --out DIR            where to stage (default: <repo>/build/tt-waypoint-bundle; replaced)
#   --regen-constraints  re-resolve packaging/constraints.txt with uv first (needs network)
#   TT_MODEL             the tt-model executable (default: tt-model)
#   PYTHON_PIN           interpreter the bundle's venv uses (default 3.12.12)
#   SKIP_INSPECT=1       skip step 9 (used by the tests, which swap in a stub `tt-model`
#                        that stages nothing)
#
# Re-running with the same inputs gives the same bundle. Only manifest `created_at` and
# PROVENANCE.json `built_at` change.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <repo>/packaging
REPO="$(dirname "$HERE")"
TT_MODEL="${TT_MODEL:-tt-model}"
# 3.12.12 is what the box's uv can provision and what the staged install was tested with. The
# docs' example is 3.12.14; raise this when the producer's uv knows that release.
PYTHON_PIN="${PYTHON_PIN:-3.12.12}"
OUT="$REPO/build/tt-waypoint-bundle"
REGEN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --regen-constraints) REGEN=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | head -n -1; exit 0 ;;
    *) echo "package-thin.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Isolated python for the helper scripts (-I: ignore PYTHON* env vars and the cwd).
py() { python3 -I "$@"; }

# --- 1. Facts the bundle must agree with the code on --------------------------------------
# Read them from the code instead of retyping them, so the manifest cannot drift from what
# session.py pins. Importing waypoint_ttnn.session is safe: it imports ttnn lazily, inside
# functions, and the helper below asserts that ttnn was not loaded.
read -r PKG_VERSION < <(py - "$REPO" <<'PY'
import re, sys, pathlib
text = (pathlib.Path(sys.argv[1]) / "pyproject.toml").read_text()
print(re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1))
PY
)
mapfile -t WEIGHTS < <(py - "$REPO" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from waypoint_ttnn import session
assert "ttnn" not in sys.modules, "importing session.py must not import ttnn"
print(session.WEIGHTS_REPO_ID)
print(session.PINNED_WEIGHTS_REVISION)
for pattern in session.WEIGHTS_ALLOW_PATTERNS:
    print(pattern)
PY
)
WEIGHTS_REPO="${WEIGHTS[0]}"
WEIGHTS_REV="${WEIGHTS[1]}"
WEIGHTS_ALLOW=("${WEIGHTS[@]:2}")
TTNN_VERSION="$(py -c "import sys; sys.path.insert(0, sys.argv[1]); import guards; print(guards.TTNN_VERSION)" "$HERE")"
echo "waypoint_ttnn $PKG_VERSION  weights $WEIGHTS_REPO@${WEIGHTS_REV:0:7} (${#WEIGHTS_ALLOW[@]} files)  ttnn $TTNN_VERSION  python $PYTHON_PIN"

# --- 2. Working area ----------------------------------------------------------------------
WORK="$REPO/build/work"
rm -rf "$WORK"
mkdir -p "$WORK"

# --- 3. The model's own code -> extra_code/ ---------------------------------------------
# closure.py walks the imports from the ASGI entry point and fails if the code imports a
# package the bundle does not install (this is what would catch a leftover `models.*` import).
py "$HERE/closure.py" --repo-root "$REPO" --stage "$WORK/extra_code"
# `.pyc` files are not part of the closure; none exist in a fresh copy.

# --- 4. serve-deps wheel ------------------------------------------------------------------
py "$HERE/build_deps_wheel.py" "$WORK/wheels" "$PKG_VERSION" >/dev/null
DEPS_WHEEL="$(ls "$WORK"/wheels/tt_waypoint_serve_deps-*.whl)"

# --- 5. Constraints -----------------------------------------------------------------------
# constraints.txt pins every package the install resolves. It is resolved in two passes that
# mirror install.sh's two steps, so each package gets the version install.sh will actually find:
#   pass 1: requirements.txt WITH the PyTorch CPU index (that is how install.sh runs step 1)
#   pass 2: the serve-deps wheel's requirements WITHOUT it, constrained by pass 1
# Packages that pass 1 already pinned keep that pin. Everything new in pass 2 is the current
# PyPI release. Known residue: pass 1 takes idna 3.4 from the PyTorch index, because fastapi ->
# anyio -> idna sits in step 1. That is a tt-model-manager install-order limit; see CLAUDE.md.
if [ "$REGEN" = 1 ]; then
  command -v uv >/dev/null || { echo "package-thin.sh: --regen-constraints needs uv" >&2; exit 1; }
  RESOLVE=(--python-version 3.12 --python-platform x86_64-manylinux_2_34 --no-annotate --no-header)
  uv pip compile "$HERE/requirements.txt" "${RESOLVE[@]}" \
      --extra-index-url https://download.pytorch.org/whl/cpu -o "$WORK/pass1.txt" >/dev/null
  py -c "import sys; sys.path.insert(0,'$HERE'); import build_deps_wheel as b; print('\n'.join(b.REQUIRES))" \
      > "$WORK/deps-in.txt"
  uv pip compile "$WORK/deps-in.txt" "${RESOLVE[@]}" -c "$WORK/pass1.txt" -o "$WORK/pass2.txt" >/dev/null
  {
    echo "# Fully resolved pins for the tt-waypoint bundle. Generated by"
    echo "#   packaging/package-thin.sh --regen-constraints   (do not edit by hand)"
    echo "# Linux x86_64, Python 3.12. Pass 1 = requirements.txt with the PyTorch CPU index;"
    echo "# pass 2 = the serve-deps wheel's requirements from PyPI. See that script for why."
    py - "$WORK/pass1.txt" "$WORK/pass2.txt" <<'PY'
import re, sys
seen, rows = set(), []
for path in sys.argv[1:]:
    for line in open(path):
        line = line.strip()
        m = re.match(r"([A-Za-z0-9_.-]+)==", line)
        if not m:
            continue
        key = re.sub(r"[-_.]+", "-", m.group(1).lower())
        if key not in seen:          # pass 1 wins over pass 2
            seen.add(key)
            rows.append(line)
print("\n".join(sorted(rows, key=str.lower)))
PY
  } > "$HERE/constraints.txt"
  echo "constraints.txt regenerated ($(grep -c '==' "$HERE/constraints.txt") pins)"
fi
[ -s "$HERE/constraints.txt" ] || { echo "package-thin.sh: packaging/constraints.txt is missing; run with --regen-constraints" >&2; exit 1; }

# --- 6. Install guards (`--verify`) ---------------------------------------------------------
# One statement per line, defined in guards.py so the tests can run them against good and bad
# layouts. install.sh runs each one with the venv's python; a failure fails the install.
mapfile -t GUARDS < <(py "$HERE/guards.py")
VERIFY_ARGS=()
for g in "${GUARDS[@]}"; do VERIFY_ARGS+=(--verify "$g"); done

# --- 7. Provenance ------------------------------------------------------------------------
# A JSON file whose PATH goes to --provenance. It is staged verbatim as PROVENANCE.json. No
# hostname or user name goes in.
py - "$REPO" "$HERE" "$WORK" "$PKG_VERSION" "$WEIGHTS_REPO" "$WEIGHTS_REV" "$TTNN_VERSION" "$PYTHON_PIN" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
repo, here, work, ver, wrepo, wrev, ttnn_v, pyv = sys.argv[1:9]
repo, here, work = map(pathlib.Path, (repo, here, work))
def git(*a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
files = sorted(p.relative_to(work / "extra_code").as_posix() for p in (work / "extra_code").rglob("*") if p.is_file())
json.dump({
    "bundle": "tt-waypoint",
    "shape": "v6 thin, kind=tt-dit-server, canonical (extra_code + stock ttnn + constraints)",
    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "recipe": "packaging/package-thin.sh",
    "source": {"repo": "https://github.com/tsingletaryTT/tt-waypoint", "commit": git("rev-parse", "HEAD"),
               "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
    "waypoint_ttnn_version": ver,
    "ttnn": ttnn_v, "python": pyv,
    "weights": {"repo": wrepo, "revision": wrev},
    "extra_code_files": {f: sha(work / "extra_code" / f) for f in files},
    "constraints_sha256": sha(here / "constraints.txt"),
    "requirements_sha256": sha(here / "requirements.txt"),
}, open(work / "provenance.json", "w"), indent=2)
PY

# --- 8. Stage -----------------------------------------------------------------------------
# NO positional repo id: with one, package-thin stages AND uploads.
CMD=("$TT_MODEL" package-thin --out "$OUT" --name tt-waypoint
  --kind tt-dit-server --app waypoint_ttnn.server.app:app
  --model-py "$HERE/model.py" --extra-code "$WORK/extra_code"
  --requirements "$HERE/requirements.txt" --constraints "$HERE/constraints.txt"
  --python "$PYTHON_PIN"
  --ops-wheel "$DEPS_WHEEL"
  --arch blackhole --mesh P150 --device-count 1
  --weights "$WEIGHTS_REPO" --weights-revision "$WEIGHTS_REV"
  --readme "$HERE/README.hf.md" --provenance "$WORK/provenance.json"
  "${VERIFY_ARGS[@]}")
for pattern in "${WEIGHTS_ALLOW[@]}"; do CMD+=(--weights-allow "$pattern"); done

echo "+ ${CMD[*]:0:3} ... (${#CMD[@]} words)"
"${CMD[@]}"

# --- 9. Inspect what was staged -----------------------------------------------------------
# The checks that catch a bad bundle cheaply: argv tokenisation of run.sh's CMD, env, the
# manifest's deps block, PYTHONPATH order, requirements/constraints hygiene, and that the
# shipped tree equals the import closure.
if [ "${SKIP_INSPECT:-0}" != 1 ]; then
  py "$HERE/inspect_stage.py" "$OUT" --repo-root "$REPO"
fi

cat <<EOF

Staged at: $OUT
Nothing was uploaded. To publish (only when asked; look at the target repo first):
  re-run the tt-model command above with the positional repo id, e.g. episod/tt-waypoint.
EOF
