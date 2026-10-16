# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The `--verify` statements for the tt-waypoint bundle.

`install.sh` runs each statement with `<bundle>/venv/bin/python -c` after the venv is built,
with `<bundle>/extra_code` on PYTHONPATH. A failing statement fails the install, so a broken
bundle stops at install time and not at the first request.

Rules the statements follow:
* One line each, so `package-thin.sh` can pass them as separate `--verify` arguments.
* No `import ttnn`. Importing ttnn opens the device runtime. Versions are read with
  `importlib.metadata`, presence with `importlib.util.find_spec`.
* The bundle folder is found from `sys.prefix`, because the venv lives at `<bundle>/venv` and
  `install.sh` does not export `$HERE`.

`python3 packaging/guards.py` prints the statements, one per line. The tests in
`waypoint_ttnn/tests/test_packaging_recipe.py` run every statement against a layout it should
accept and layouts it should reject.
"""

#: The ttnn release the bundle was built and verified against.
TTNN_VERSION = "0.78.0"

GUARDS = {
    # The engine is the stock PyPI release: exact version, and no "+local" tag of a fork.
    "stock-ttnn": (
        "import importlib.metadata as m; v=m.version('ttnn'); "
        f"assert v=='{TTNN_VERSION}', f'ttnn {{v}} is not the stock {TTNN_VERSION}'"
    ),
    # The model's own code ships in the bundle. This guard checks that it is there, that it is
    # the copy Python resolves, and that no tt-metal `models` tree or tt-transformers wheel came
    # along with it (this model uses neither).
    "own-code-in-bundle": (
        "import sys,pathlib,importlib.util as u; b=pathlib.Path(sys.prefix).parent; "
        "assert (b/'extra_code'/'waypoint_ttnn'/'server'/'app.py').is_file(), 'extra_code/waypoint_ttnn missing'; "
        "assert not (b/'extra_code'/'models').exists(), 'extra_code/models must not ship'; "
        "assert u.find_spec('models') is None, 'a top-level models package is installed'; "
        "assert u.find_spec('tt_transformers') is None, 'tt_transformers is installed but unused'"
    ),
    # The served code imports without opening a device, from extra_code/, and never loads ttnn
    # at import time. A top-level `import ttnn` added later would touch the device at install.
    # `python -c` puts the current directory first on sys.path, so a checkout of this repo in the
    # directory install.sh runs from would shadow the shipped copy. Drop it first. (This caught
    # a real shadowing when install.sh was run from the repo root.)
    "imports-without-device": (
        "import sys,pathlib; sys.path=[p for p in sys.path if p not in ('','.')]; "
        "b=pathlib.Path(sys.prefix).parent; "
        "import waypoint_ttnn.session as s, waypoint_ttnn.server.app as a; "
        "assert pathlib.Path(a.__file__).resolve().is_relative_to((b/'extra_code').resolve()), a.__file__; "
        "assert 'ttnn' not in sys.modules, 'importing the served code imported ttnn'"
    ),
    # The network stack must be a current one. A 2022 certifi/urllib3 is what the PyTorch CPU
    # index would install if huggingface_hub came in through the wrong step.
    "current-network-stack": (
        "import importlib.metadata as m; "
        "assert int(m.version('certifi').split('.')[0])>=2025, 'stale certifi: '+m.version('certifi'); "
        "assert int(m.version('urllib3').split('.')[0])>=2, 'stale urllib3: '+m.version('urllib3'); "
        "assert int(m.version('requests').split('.')[1])>=31, 'stale requests: '+m.version('requests')"
    ),
}

if __name__ == "__main__":
    for statement in GUARDS.values():
        assert "\n" not in statement
        print(statement)
