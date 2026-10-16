# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Static import closure of the served Waypoint app (stdlib only, no ttnn, no hardware).

`packaging/package-thin.sh` ships the model's own code inside the bundle (`--extra-code`).
It must ship exactly the modules the server can reach, and nothing the server cannot:
no `capture_*.py`, `benchmark.py`, `stage*_check.py` or `tests/`. This module computes that
set by reading the source, so the list is derived from the code and nobody keeps it by hand. It fails loudly
when an import cannot be resolved.

What it does, starting from `ENTRY`:

1. Parses every file with `ast` and follows each `import waypoint_ttnn...` and
   `from waypoint_ttnn... import ...` it finds, including imports written inside functions
   (the served code imports ttnn and its own heavy modules lazily).
2. Includes the `__init__.py` of every package on the way.
3. Collects every other top-level import. Each one must be stdlib or in
   `THIRD_PARTY_OK`, the packages the bundle's pins install. Anything else is an error.
   That catches a `models.*` import (the old tt-metal dependency) and a typo alike.
4. Rejects dynamic imports (`importlib`, `__import__`), because a static walk cannot see them.

Usage (from the repo root):
    python3 packaging/closure.py --list              # print the files, one per line
    python3 packaging/closure.py --stage <dir>       # copy the closure to <dir>/waypoint_ttnn/...
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

#: The ASGI module the bundle serves (`--app waypoint_ttnn.server.app:app`).
ENTRY = "waypoint_ttnn.server.app"

#: The package this repo ships as source.
PACKAGE = "waypoint_ttnn"

#: Third-party top-level names the served code may import. Each is installed by the bundle:
#: ttnn and torch by requirements.txt, fastapi/uvicorn/pydantic/pillow by the HTTP stack,
#: numpy by ttnn, safetensors and huggingface_hub by the serve-deps wheel, starlette by fastapi.
THIRD_PARTY_OK = frozenset({
    "ttnn", "torch", "numpy", "PIL", "fastapi", "starlette", "pydantic", "uvicorn",
    "safetensors", "huggingface_hub",
})


class ClosureError(Exception):
    """The served code imports something the bundle would not provide."""


def _module_file(root: Path, dotted: str) -> Path | None:
    """The source file for a dotted `waypoint_ttnn...` module name, or None if there is none."""
    rel = Path(*dotted.split("."))
    as_module = root / rel.with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = root / rel / "__init__.py"
    return as_package if as_package.is_file() else None


def _imports(path: Path) -> Tuple[Set[str], Set[str], Set[str], List[str]]:
    """(modules, maybe-modules, third-party top-level names, dynamic-import findings) of a file.

    `modules` are names that must be files: `import waypoint_ttnn.x` and the `M` in
    `from M import ...`. `maybe-modules` are the names in `from waypoint_ttnn import a`, which
    are a submodule when a file exists for them and a plain attribute otherwise.
    Relative imports are rejected: the served code uses absolute imports only, and the
    closure walk would otherwise need to track package context.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    own: Set[str] = set()
    maybe: Set[str] = set()
    other: Set[str] = set()
    dynamic: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                (own if top == PACKAGE else other).add(alias.name if top == PACKAGE else top)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ClosureError(f"{path}: relative import (line {node.lineno}); use absolute imports")
            top = (node.module or "").split(".", 1)[0]
            if top == PACKAGE:
                own.add(node.module)
                maybe.update(f"{node.module}.{a.name}" for a in node.names)
            else:
                other.add(top)
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name in ("__import__", "import_module"):
                dynamic.append(f"{path}:{node.lineno} {name}()")
    return own, maybe, other, dynamic


def served_closure(repo_root: Path) -> Dict[str, Path]:
    """Map of repo-relative posix path -> file, for every module the server can import.

    Raises ClosureError for an unresolved import, a non-allowed third-party import, or a
    dynamic import.
    """
    repo_root = Path(repo_root)
    found: Dict[str, Path] = {}
    third_party: Dict[str, str] = {}
    dynamic: List[str] = []
    todo = [ENTRY]
    seen: Set[str] = set()
    while todo:
        dotted = todo.pop()
        if dotted in seen:
            continue
        seen.add(dotted)
        src = _module_file(repo_root, dotted)
        if src is None:
            continue  # e.g. `from waypoint_ttnn import session` also reports an attribute name
        found[src.relative_to(repo_root).as_posix()] = src
        # Every parent package's __init__.py ships too, because Python imports it.
        parts = dotted.split(".")
        for i in range(1, len(parts)):
            init = _module_file(repo_root, ".".join(parts[:i]))
            if init is not None:
                found[init.relative_to(repo_root).as_posix()] = init
        own, maybe, other, dyn = _imports(src)
        dynamic += dyn
        for top in other:
            third_party.setdefault(top, src.relative_to(repo_root).as_posix())
        # A module the code imports by name must be a file. A "maybe" name (from M import a)
        # is followed only when it is a file; otherwise it is an attribute of M.
        for m in own:
            if _module_file(repo_root, m) is None:
                raise ClosureError(f"{src.relative_to(repo_root).as_posix()}: imports {m!r}, "
                                   f"which is not a module under {repo_root}")
        todo.extend(m for m in own | {x for x in maybe if _module_file(repo_root, x)} if m not in seen)

    stdlib = set(sys.stdlib_module_names)
    bad = {t: where for t, where in third_party.items()
           if t not in stdlib and t not in THIRD_PARTY_OK and t != "__future__"}
    if bad:
        lines = "\n".join(f"  {t}  (first seen in {w})" for t, w in sorted(bad.items()))
        raise ClosureError(
            "the served code imports packages the bundle does not install:\n" + lines +
            "\nAdd the package to packaging/requirements.txt and THIRD_PARTY_OK, or remove the import.")
    if dynamic:
        raise ClosureError("dynamic imports cannot be checked statically:\n  " + "\n  ".join(dynamic))
    return dict(sorted(found.items()))


def stage(repo_root: Path, dest: Path) -> List[str]:
    """Copy the closure into `dest` (which is created, and must not already exist)."""
    files = served_closure(repo_root)
    if dest.exists():
        raise ClosureError(f"{dest} already exists; refusing to mix files into it")
    for rel, src in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return list(files)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print the closure, one path per line")
    group.add_argument("--stage", type=Path, metavar="DIR", help="copy the closure into DIR")
    args = ap.parse_args(argv)
    try:
        if args.list:
            print("\n".join(served_closure(args.repo_root)))
        else:
            files = stage(args.repo_root, args.stage)
            print(f"staged {len(files)} files into {args.stage}", file=sys.stderr)
    except ClosureError as exc:
        print(f"closure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
