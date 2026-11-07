"""One explicit speculative candidate-depth setting.

``HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET`` is the setting shared by the server
and direct generator callers. The former generator-local name
``HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET`` has been removed; setting it raises an
error naming its replacement, so an old configuration is never silently ignored.
"""

from __future__ import annotations

import os

SPECULATIVE_CANDIDATE_BUDGET_ENV = "HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET"
REMOVED_GGUF_MTP_CANDIDATE_BUDGET_ENV = "HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"


def _read_positive_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{name} must be a positive integer, got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(f"{name} value must be >= 1")
    return value


def resolve_explicit_candidate_budget_env() -> int | None:
    """Return the operator-pinned depth, or ``None`` when it is not set.

    An explicitly configured depth is returned unchanged; capability checks
    belong to the owner that executes it. The removed legacy name is rejected
    rather than ignored, because ignoring it would run a different depth from
    the one the operator configured.
    """

    removed = os.environ.get(REMOVED_GGUF_MTP_CANDIDATE_BUDGET_ENV)
    if removed is not None and removed.strip():
        raise ValueError(
            f"{REMOVED_GGUF_MTP_CANDIDATE_BUDGET_ENV} has been removed; set "
            f"{SPECULATIVE_CANDIDATE_BUDGET_ENV}={removed.strip()} instead"
        )
    return _read_positive_int(SPECULATIVE_CANDIDATE_BUDGET_ENV)
