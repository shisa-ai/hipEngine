"""Cold-path physical target row policy shared by profile and direct owners."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hipengine.kernels.backends import backend_package_capability


MTP2_MAX_CANDIDATE_DEPTH = 7
MTP2_MAX_PHYSICAL_REQUESTS = 8
DEFAULT_PHYSICAL_WIDTH_DEPTHS = tuple(
    (width, depth) for width in range(1, 5) for depth in range(1, 4)
)


def resolve_physical_width_depths(
    backend: str, profile: str, *, capability=backend_package_capability,
) -> tuple[tuple[int, int], ...]:
    """Validate and snapshot package width/depth cells without capacity clamping."""

    if not backend:
        return DEFAULT_PHYSICAL_WIDTH_DEPTHS
    table = capability(backend, "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS", {})
    if not isinstance(table, Mapping):
        raise RuntimeError("backend MTP2 physical width/depth policies must be a mapping")
    raw = table.get(profile, DEFAULT_PHYSICAL_WIDTH_DEPTHS)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise RuntimeError("backend MTP2 physical width/depth policy must be a sequence")
    cells: set[tuple[int, int]] = set()
    for item in raw:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)):
            raise RuntimeError("backend MTP2 physical width/depth cells must be pairs")
        values = tuple(item)
        if len(values) != 2:
            raise RuntimeError("backend MTP2 physical width/depth cells must be pairs")
        try:
            width, depth = (int(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "backend MTP2 physical width/depth cells must be integer pairs"
            ) from exc
        if not 1 <= width <= MTP2_MAX_PHYSICAL_REQUESTS:
            raise RuntimeError("backend MTP2 physical widths must be within [1, 8]")
        if not 1 <= depth <= MTP2_MAX_CANDIDATE_DEPTH:
            raise RuntimeError(
                "backend MTP2 candidate depths must be within "
                f"[1, {MTP2_MAX_CANDIDATE_DEPTH}]"
            )
        cells.add((width, depth))
    if not cells:
        raise RuntimeError("backend MTP2 physical width/depth policy must not be empty")
    return tuple(sorted(cells))


@dataclass(frozen=True, slots=True)
class PhysicalTargetRowsPlan:
    pad_counts: tuple[int, ...] = ()
    exact_counts: tuple[int, ...] = ()
    mixed_q6_rowtiles: bool = False
    extra_q4_rowtiles: bool = False
    q5_rowtiles: bool = False
    q6_rowtiles: bool = False


def resolve_physical_target_rows(backend: str, profile: str) -> PhysicalTargetRowsPlan:
    """Snapshot row capabilities and explicit diagnostic rollbacks once."""

    if profile != "production":
        return PhysicalTargetRowsPlan()

    def enabled(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}

    padding = tuple(int(value) for value in backend_package_capability(
        backend, "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS", (),
    ))
    exact: set[int] = set()
    if enabled("HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS", True):
        exact.update(int(value) for value in backend_package_capability(
            backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS", (),
        ))
        for capability in (
            "GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS_POLICY",
            "GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS_POLICY",
        ):
            policy = backend_package_capability(backend, capability, {})
            env = policy.get("enabled_env")
            default = bool(policy.get("enabled_default", False))
            if policy and (default if env is None else enabled(str(env), default)):
                exact.update(int(value) for value in policy.get("rows", ()))
    extra_q4 = bool(backend_package_capability(
        backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXTRA_ROWTILE_SHAPES", (),
    ))
    q5 = bool(backend_package_capability(
        backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q5_ROWTILE_ROWS", (),
    ))
    q6 = bool(backend_package_capability(
        backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS", (),
    ))
    mixed_q6 = bool(
        q6
        and backend_package_capability(
            backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS", {},
        )
        and enabled("HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", True)
    )
    return PhysicalTargetRowsPlan(
        padding, tuple(sorted(exact)), mixed_q6, extra_q4, q5, q6,
    )
