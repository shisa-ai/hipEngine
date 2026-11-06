"""Cold-path physical target row policy shared by profile and direct owners."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hipengine.kernels.backends import backend_package_capability


@dataclass(frozen=True, slots=True)
class PhysicalTargetRowsPlan:
    pad_counts: tuple[int, ...] = ()
    exact_counts: tuple[int, ...] = ()
    mixed_q6_rowtiles: bool = False


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
    mixed_q6 = bool(
        backend_package_capability(
            backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS", (),
        )
        and backend_package_capability(
            backend, "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS", {},
        )
        and enabled("HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES", True)
    )
    return PhysicalTargetRowsPlan(padding, tuple(sorted(exact)), mixed_q6)
