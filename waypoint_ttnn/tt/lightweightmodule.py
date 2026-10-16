# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""`LightweightModule`: the tiny base class every TTNN module in this package derives from.

This is a copy of tt-metal's `models/common/lightweightmodule.py` (identical to the
v0.78.0 file, which is the release `ttnn==0.78.0` is built from). It lives here so the
served code has no dependency on any tt-metal `models/` tree. The class is ten lines and
has no imports, so a vendored copy is cheaper than a wheel that carries it. The bundle
therefore ships `waypoint_ttnn` as plain source (`--extra-code`) and needs no
`models` package at all.
"""


class LightweightModule:
    """Torch modules add a surprising amount of host overhead for attribute
    access and method calls. This class is a lightweight alternative that
    just wraps a forward function for now."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
