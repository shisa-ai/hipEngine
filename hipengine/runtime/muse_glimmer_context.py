"""Disk snapshots of Muse Glimmer's contiguous KV (target and DFlash drafter).

A snapshot is a directory:

* ``context.json`` -- format tag, cached token count, KV geometry, and the
  source model file's name and size (recorded, warned on, never refused).
* ``tokens.npy`` -- the token id at every cached position.
* ``target.kv`` -- for each target layer, its K rows then its V rows
  (``tokens * kv_dim`` BF16 values each).
* ``drafter.kv`` -- the same for the DFlash drafter, present when the drafter
  held the same prefix when the snapshot was taken.

Loading restores the KV and the token history, so the next request whose
prompt starts with those tokens reuses them through the prefix cache and
prefills only the suffix. A 256K-token context is ~13.8 GB of target KV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_array_to_device,
)

FORMAT = "hipengine.muse_glimmer.context/1"


def write_kv(
    path: Path, kv: Sequence[tuple[DeviceBuffer, DeviceBuffer]], rows: int,
    row_bytes: int, *, runtime: Any,
) -> None:
    """Write the first ``rows`` rows of every K and V buffer, layer by layer."""

    nbytes = int(rows) * int(row_bytes)
    host = np.empty(nbytes, dtype=np.uint8)
    with open(path, "wb") as fh:
        for pair in kv:
            for buf in pair:
                if nbytes:
                    copy_device_to_host(host.ctypes.data, buf, nbytes, runtime=runtime)
                fh.write(memoryview(host))


def read_kv(
    path: Path, kv: Sequence[tuple[DeviceBuffer, DeviceBuffer]], rows: int,
    row_bytes: int, *, runtime: Any,
) -> None:
    """Fill the first ``rows`` rows of every K and V buffer from ``write_kv`` output."""

    nbytes = int(rows) * int(row_bytes)
    expected = nbytes * 2 * len(kv)
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} holds {actual} bytes; this geometry needs {expected}")
    host = np.empty(nbytes, dtype=np.uint8)
    with open(path, "rb") as fh:
        for pair in kv:
            for buf in pair:
                if fh.readinto(memoryview(host)) != nbytes:
                    raise ValueError(f"{path} ended early")
                if nbytes:
                    copy_host_array_to_device(DeviceBuffer(buf.ptr, nbytes), host, runtime=runtime)


def write_meta(directory: Path, meta: dict[str, Any], tokens: np.ndarray) -> None:
    np.save(directory / "tokens.npy", np.asarray(tokens, dtype=np.int64))
    (directory / "context.json").write_text(json.dumps({"format": FORMAT, **meta}, indent=2) + "\n")


def read_meta(directory: Path) -> tuple[dict[str, Any], np.ndarray]:
    meta = json.loads((directory / "context.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{directory} is not a {FORMAT} snapshot (format={meta.get('format')!r})")
    tokens = np.load(directory / "tokens.npy")
    if tokens.ndim != 1 or tokens.size != int(meta["tokens"]):
        raise ValueError(f"{directory}/tokens.npy does not match context.json")
    return meta, tokens.astype(np.int64, copy=False)
