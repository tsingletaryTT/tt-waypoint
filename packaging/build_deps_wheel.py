# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build the empty "serve-deps" wheel (stdlib only, byte-for-byte reproducible).

Why this exists. `install.sh` installs `requirements.txt` with the PyTorch CPU index ranked
above PyPI. That index carries one years-old copy each of certifi, urllib3, requests, idna and
charset-normalizer, and uv then takes those packages only from it. `huggingface_hub` (which
`session.py` needs for `snapshot_download`) depends on `requests`, so listing it in
`requirements.txt` would install a 2022 CA bundle. `install.sh` installs wheels given with
`--ops-wheel` in a later step, by path, without the PyTorch index. This wheel holds no code. It
only declares the packages that must come in through that step, so they resolve from PyPI
against `constraints.txt`.

The wheel is metadata only: `Requires-Dist` lines and no importable module. It is not a custom
op and `model.py` does not import it, so `deps.custom_ops` stays empty.

Usage: python3 packaging/build_deps_wheel.py <out_dir> <version>
"""

import base64
import hashlib
import sys
import zipfile
from pathlib import Path

NAME = "tt-waypoint-serve-deps"
DIST = NAME.replace("-", "_")

#: Packages that must not be resolved against the PyTorch CPU index.
REQUIRES = (
    "huggingface_hub>=0.20,<1",   # session.py: snapshot_download of the pinned weights
    "safetensors>=0.4.0,<1",      # session.py: load_file for the transformer and VAE weights
)

#: A fixed timestamp keeps the wheel identical from run to run (ZIP epoch is 1980-01-01).
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _record_line(path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(data)}"


def build(out_dir: Path, version: str) -> Path:
    info = f"{DIST}-{version}.dist-info"
    metadata = "\n".join(
        ["Metadata-Version: 2.1", f"Name: {NAME}", f"Version: {version}",
         "Summary: Dependencies of the tt-waypoint server that must not resolve against the PyTorch CPU index. No code.",
         "Requires-Python: >=3.10",
         *[f"Requires-Dist: {r}" for r in REQUIRES], ""])
    wheel = "\n".join(["Wheel-Version: 1.0", "Generator: tt-waypoint packaging/build_deps_wheel.py",
                       "Root-Is-Purelib: true", "Tag: py3-none-any", ""])
    files = {f"{info}/METADATA": metadata.encode(), f"{info}/WHEEL": wheel.encode()}
    record = "\n".join([_record_line(p, d) for p, d in files.items()] + [f"{info}/RECORD,,", ""])
    files[f"{info}/RECORD"] = record.encode()

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{DIST}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path, data in files.items():
            zi = zipfile.ZipInfo(path, _ZIP_TIME)
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
    return target


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(build(Path(sys.argv[1]), sys.argv[2]))
