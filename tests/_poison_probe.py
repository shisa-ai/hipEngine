"""Poison probes for device buffers that are read before they are written.

``hipMalloc`` returns zeroed pages for a *fresh* allocation, but a recycled block
keeps its previous contents at small sizes (measured on gfx1151: a 1 MiB block
comes back holding what was written to it, 4 MiB and above come back zeroed).
Any per-call buffer whose unread region is assumed to be zero is therefore a
latent full-suite failure: the test that allocates it first passes, and the one
that recycles another test's NaN or garbage does not.  ``docs/KERNELS.md``
"Device-memory hygiene" states the rule; this module is the probe that enforces
it.

The probe is deliberately end-to-end rather than white-box.  It runs the
workload once for a reference, overwrites every per-call device buffer the
runner owns with ``0xFF`` (a quiet NaN in fp32), runs the workload again, and
requires the second result to be *bit-identical* to the first.  A runner that
reads a buffer before writing it, or "clears" state with a scale-by-zero kernel
(``NaN * 0 == NaN``), fails.  Weights and once-uploaded metadata are excluded:
they are written by the loader before the first call and are not per-call state.

Poisoning one group at a time localizes the offender, which is what turned the
Surya all-NaN full-suite failure into the conv/GDN state re-zero.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from hipengine.core.memory import DeviceBuffer

# 0xFF bytes are a quiet NaN in fp32 and an invalid value in every integer
# width, so one poison byte serves every dtype.
POISON_BYTE = 0xFF

# Attribute names that are never per-call state, matched as case-insensitive
# substrings: the runtime itself, the loaded weights, the library handles, and
# the once-uploaded position/frequency tables.  Walking into them is either
# useless or unbounded, and poisoning them would corrupt the model rather than
# test it.  A test that expects a specific buffer set asserts it explicitly, so
# an over-broad token here shows up as a missing buffer rather than a silent
# pass.
_DENY_TOKENS = (
    "runtime",
    "hip",
    "lib",
    "rocblas",
    "hipblaslt",
    "weight",
    "local",
    "loaded",
    "model",
    "spec",
    "config",
    "tokenizer",
    "scheduler",
    "host",
    "stream",
    "handle",
    "timescale",
    "rope",
    "inv_freq",
)


def _denied(name: str, deny: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in deny)

# Types the walker never descends into, by class-name substring.  These hold
# library handles and model state, not per-call scratch.
_DENY_TYPES = ("Runtime", "Library", "Rocblas", "Hipblaslt", "Module", "Path", "type")

# A hard cap on visited objects: a runner that exposes a large object graph must
# not turn a probe into a multi-second walk, and a walk that hits the cap is a
# probe bug rather than a clean result.
_MAX_VISITED = 20_000


def collect_device_buffers(
    root: Any,
    *,
    max_depth: int = 5,
    deny: Sequence[str] = _DENY_TOKENS,
) -> list[tuple[str, DeviceBuffer]]:
    """Return every ``DeviceBuffer`` reachable from ``root`` as (path, buffer).

    Walks dataclass fields, mappings, sequences, and plain attributes, skipping
    names matching :data:`_DENY_TOKENS`. Incomplete traversal raises instead
    of silently certifying a partial buffer set.
    Arena views are reported through their owning allocation, so a poison write
    covers the whole arena rather than one view of it.
    """

    found: list[tuple[str, DeviceBuffer]] = []
    seen: set[int] = set()
    budget = [_MAX_VISITED]

    def visit(value: Any, path: str, depth: int) -> None:
        if id(value) in seen:
            return
        if isinstance(value, (str, bytes, bytearray, int, float, bool, type(None))):
            return
        if any(token in type(value).__name__ for token in _DENY_TYPES):
            return
        if budget[0] <= 0:
            raise ValueError(f"poison traversal budget exhausted at {path}")
        budget[0] -= 1
        seen.add(id(value))
        if isinstance(value, DeviceBuffer):
            found.append((path, value))
            return
        # ``DeviceMemoryArena`` is not imported here to keep this helper usable
        # without a device; its owning allocation is the poisonable range.
        owner = getattr(value, "owner", None)
        if isinstance(owner, DeviceBuffer) and type(value).__name__.endswith("Arena"):
            found.append((f"{path}.owner", owner))
            return
        if depth >= max_depth:
            raise ValueError(f"poison traversal depth limit reached at {path}")
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                if _denied(field.name, deny):
                    continue
                visit(getattr(value, field.name, None), f"{path}.{field.name}", depth + 1)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}[{key!r}]", depth + 1)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in attributes.items():
                if name.startswith("__") or _denied(name, deny):
                    continue
                visit(item, f"{path}.{name}", depth + 1)

    visit(root, type(root).__name__, 0)
    # Deduplicate by pointer: one allocation reachable by two paths is one write.
    unique: dict[int, tuple[str, DeviceBuffer]] = {}
    for path, buffer in found:
        unique.setdefault(int(buffer.ptr), (path, buffer))
    return sorted(unique.values(), key=lambda item: (item[0], item[1].ptr))


def poison(
    runtime: Any,
    buffers: Sequence[DeviceBuffer],
    *,
    byte: int = POISON_BYTE,
    synchronize: bool = True,
) -> int:
    """Overwrite every buffer with ``byte`` and return the bytes written."""

    written = 0
    for buffer in buffers:
        if buffer.nbytes <= 0:
            continue
        runtime.memset(int(buffer.ptr), int(byte), int(buffer.nbytes))
        written += int(buffer.nbytes)
    if synchronize:
        runtime.device_synchronize()
    return written


def _flatten(value: Any, *, path: str = "") -> list[tuple[str, np.ndarray]]:
    """Flatten a run's return value into named arrays.

    A workload returns whatever it returns -- a single array, a tuple of
    arrays, a mapping -- and the probe only needs comparable leaves.
    """

    if isinstance(value, Mapping):
        out: list[tuple[str, np.ndarray]] = []
        for key, item in value.items():
            out.extend(_flatten(item, path=f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(value, (tuple, list)):
        out = []
        for index, item in enumerate(value):
            out.extend(_flatten(item, path=f"{path}[{index}]"))
        return out
    return [(path, np.asarray(value))]


def _diff(a: Any, b: Any) -> tuple[str, str]:
    left = _flatten(a)
    right = _flatten(b)
    if len(left) != len(right):
        return (f"{len(left)} outputs -> {len(right)}", "shape")
    for (left_name, left_arr), (right_name, right_arr) in zip(left, right, strict=True):
        if left_arr.shape != right_arr.shape:
            return (f"{left_name}: shape {left_arr.shape} -> {right_arr.shape}", "shape")
        if not np.isfinite(right_arr).all():
            return (
                f"{right_name}: {int((~np.isfinite(right_arr)).sum())} of "
                f"{right_arr.size} values are not finite",
                "nonfinite",
            )
        if not np.array_equal(left_arr, right_arr):
            delta = np.abs(left_arr.astype(np.float64) - right_arr.astype(np.float64))
            return (
                f"{right_name}: max|delta| {delta.max():.3e} over "
                f"{int((delta != 0).sum())} values",
                "mismatch",
            )
    return ("bit-identical", "ok")


def _snapshot(value: Any) -> dict[str, np.ndarray]:
    # Runners may return views into reusable host output buffers. Own the
    # reference bytes before a later run can overwrite them, also in localization.
    return {name: array.copy() for name, array in _flatten(value)}


def _checked_groups(collect) -> dict[str, Sequence[DeviceBuffer]]:
    groups = dict(collect())
    if not groups or any(not any(b.nbytes > 0 for b in group) for group in groups.values()):
        raise ValueError("poison probe has empty buffer coverage")
    return groups


def assert_poison_invariant(
    runtime: Any,
    collect: Callable[[], Mapping[str, Sequence[DeviceBuffer]]],
    run: Callable[[], Any],
    *,
    label: str,
    reset: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run, poison every per-call buffer, run again, and require bit-identity.

    ``collect`` returns the buffer groups to poison and is called again after
    every ``reset``, so a workload that reallocates its buffers is probed with
    the *current* pointers rather than stale ones.  ``run`` performs the
    workload and returns whatever is comparable: an array, a tuple of arrays, a
    mapping of arrays.

    Returns a small report (buffer count, bytes, comparison) so a caller can
    assert what it proved.  On a mismatch, ``reset`` localizes the offender by
    re-running the workload once per group, which is the diagnostic that
    matters: the alternative is a whole-run failure with no attribution.

    ``reset`` restores a clean, unpoisoned workload state and must re-create any
    buffer the probe can poison (a fresh runner, or one whose per-call buffers
    have been released).  It is required for localization because a poison is
    not undoable: without it the second group is measured against an
    already-poisoned reference and every later group looks guilty.  A caller
    that cannot reset still gets the all-groups verdict, with the localization
    reported as skipped.
    """

    reference = _snapshot(run())
    groups = _checked_groups(collect)
    all_buffers = [buffer for group in groups.values() for buffer in group]
    written = poison(runtime, all_buffers)
    poisoned = run()
    message, verdict = _diff(reference, poisoned)
    report: dict[str, Any] = {
        "label": label,
        "buffers": len(all_buffers),
        "bytes": written,
        "verdict": verdict,
        "message": message,
    }
    if verdict == "ok":
        return report

    if reset is None:
        report["localization"] = "skipped: no reset callback supplied"
    else:
        offenders: list[tuple[str, str]] = []
        for name in groups:
            reset()
            fresh = _snapshot(run())
            current = _checked_groups(collect)
            buffers = current.get(name)
            if not buffers:
                offenders.append((name, "group absent after reset"))
                continue
            poison(runtime, buffers)
            after = run()
            group_message, group_verdict = _diff(fresh, after)
            if group_verdict != "ok":
                offenders.append((name, group_message))
        report["offenders"] = offenders
        reset()

    detail = ", ".join(f"{name}: {msg}" for name, msg in report.get("offenders", []))
    raise AssertionError(
        f"{label}: poison probe changed the result ({message})"
        + (f"; offender groups: {detail}" if detail else "")
        + (f"; {report['localization']}" if "localization" in report else "")
        + f"; poisoned {len(all_buffers)} buffers, {written} bytes"
    )


def group_by_prefix(
    buffers: Sequence[tuple[str, DeviceBuffer]],
    *,
    depth: int = 2,
) -> dict[str, list[DeviceBuffer]]:
    """Bucket ``collect_device_buffers`` output into coarse groups.

    ``depth`` is how many trailing path components name a group, with sequence
    indices stripped, so ``..._buffers[(2, 16, 20)].caches_v[7]`` and
    ``...caches_v[19]`` land in one group named ``_buffers.caches_v``.  That is
    the granularity the Surya probe localized with: the scratch family, the KV
    family, the state family.
    """

    groups: dict[str, list[DeviceBuffer]] = {}
    for path, buffer in buffers:
        parts = [re.sub(r"\[.*\]$", "", part) for part in path.split(".")]
        name = ".".join(parts[-depth:]) if len(parts) >= depth else ".".join(parts)
        groups.setdefault(name, []).append(buffer)
    return groups


def iter_buffers(root: Any, **kwargs: Any) -> Iterator[tuple[str, DeviceBuffer]]:
    """Convenience iterator over :func:`collect_device_buffers`."""

    yield from collect_device_buffers(root, **kwargs)
