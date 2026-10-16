# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Bundle entry point for the tt-waypoint v6 thin bundle (`kind: tt-dit-server`).

`tt-model package-thin` requires a runner file (`--model-py`). For a vLLM model it names the
adapter class. This model has no vLLM adapter: `run.sh` serves the ASGI app directly with
uvicorn (`waypoint_ttnn.server.app:app`, set with `--app`). So this file only re-exports that
app. It lets `import model` prove, in one step, that the shipped code under `extra_code/`
imports.

There is no `import <ops wheel>` line here, and that is deliberate. The canonical shape asks
`model.py` to import the model's `generic_op` wheel so that `deps.custom_ops` records it.
Waypoint has no custom kernels: every op is a stock `ttnn` op, so there is no ops wheel to
import and `deps.custom_ops` is correctly empty.

Importing this module must not open a device or import ttnn. `server/app.py` imports ttnn
inside functions only, and `packaging/guards.py` asserts that after install.
"""

from waypoint_ttnn.server.app import app

__all__ = ["app"]
