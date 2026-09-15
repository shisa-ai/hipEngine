"""Opt-in launch census for GGUF quantized matmuls.

Answers "which tensor, at which shape, through which kernel variant" for a real
prefill. The kernel symbol alone is not enough: one variant serves several
tensors, so a profile that only sees ``..._coltile8_rowbatch4_f32_f32_out``
cannot say whether the time went to an attention projection, a hyper-connection
projection or an expert matmul.

Off unless ``HIPENGINE_KERNEL_CENSUS`` is set, and it does nothing but a cached
string check when off, so it can sit on the launch path. It records shape and
role only - never pointers, tensors or data - and aggregates by
``(role, quant, symbol, rows, in_features, out_features)`` so a long run cannot
grow memory without bound.

The role comes from :func:`push_role`, which a driver script sets around the
module-level entry points it wants to attribute. When no role is pushed the
record is attributed to ``"<unattributed>"`` rather than guessed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ENV_VAR = "HIPENGINE_KERNEL_CENSUS"

_state: bool | None = None
_records: dict[tuple[str, str, str, int, int, int, int], int] = {}
_order: list[tuple[str, str, str, int, int, int, int]] = []
_roles: list[str] = []


def enabled() -> bool:
    """Whether the census is on. Cached: this is called on every launch."""
    global _state
    if _state is None:
        _state = bool(os.environ.get(ENV_VAR))
    return _state


def reset() -> None:
    _records.clear()
    _order.clear()
    _roles.clear()


def push_role(role: str) -> None:
    _roles.append(role)


def pop_role() -> None:
    if _roles:
        _roles.pop()


def current_role() -> str:
    return _roles[-1] if _roles else "<unattributed>"


def record_launch(
    quant: str,
    symbol: str,
    rows: int,
    in_features: int,
    out_features: int,
    num_experts: int = 1,
) -> None:
    """Record one launch. Cheap enough to call unconditionally.

    ``num_experts`` is 1 for a dense projection and the expert count for a
    routed one. It is recorded rather than folded into ``rows`` because the
    routed kernels tile over experts and the two are not interchangeable.
    """
    if not enabled():
        return
    key = (
        current_role(), str(quant), str(symbol), int(rows), int(in_features),
        int(out_features), int(num_experts),
    )
    if key not in _records:
        _order.append(key)
    _records[key] = _records.get(key, 0) + 1


def snapshot() -> dict[str, Any]:
    """Aggregate the census, with per-role subtotals and the launch total."""
    rows = [
        {
            "role": key[0],
            "quant": key[1],
            "symbol": key[2],
            "rows": key[3],
            "in_features": key[4],
            "out_features": key[5],
            "num_experts": key[6],
            "launches": _records[key],
        }
        for key in _order
    ]
    rows.sort(key=lambda row: (-row["launches"], row["role"], row["symbol"]))
    by_role: dict[str, int] = {}
    for row in rows:
        by_role[row["role"]] = by_role.get(row["role"], 0) + row["launches"]
    return {
        "schema": 1,
        "kind": "gguf_launch_census",
        "total_launches": sum(_records.values()),
        "distinct_shapes": len(rows),
        "launches_by_role": dict(sorted(by_role.items(), key=lambda kv: -kv[1])),
        "rows": rows,
    }


def dump(path: str | Path | None = None) -> dict[str, Any]:
    payload = snapshot()
    target = path or os.environ.get(ENV_VAR)
    if target and target != "1":
        Path(target).write_text(json.dumps(payload, indent=1) + "\n")
    return payload
