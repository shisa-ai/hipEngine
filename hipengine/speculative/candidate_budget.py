"""One explicit speculative candidate-depth setting.

``HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET`` is the canonical setting shared by
the server and direct generator callers. ``HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET``
is a deprecated compatibility alias: a legacy-only setting still resolves its
exact requested depth and emits a ``FutureWarning``. When both names are present
they must agree.
"""

from __future__ import annotations

import os
import warnings

SPECULATIVE_CANDIDATE_BUDGET_ENV = "HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET"
LEGACY_GGUF_MTP_CANDIDATE_BUDGET_ENV = "HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"


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
    """Return the operator-pinned depth, or ``None`` when neither name is set.

    The canonical name is read first, but the legacy alias is not ignored: if
    both are present and disagree, that is a configuration error rather than a
    silent precedence choice. A legacy-only setting emits ``FutureWarning`` and
    still resolves its exact depth. An explicitly configured depth is returned
    unchanged; capability checks belong to the owner that executes it.
    """

    canonical = _read_positive_int(SPECULATIVE_CANDIDATE_BUDGET_ENV)
    legacy = _read_positive_int(LEGACY_GGUF_MTP_CANDIDATE_BUDGET_ENV)
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError(
            f"{SPECULATIVE_CANDIDATE_BUDGET_ENV}={canonical} conflicts with "
            f"{LEGACY_GGUF_MTP_CANDIDATE_BUDGET_ENV}={legacy}"
        )
    if canonical is None and legacy is not None:
        warnings.warn(
            f"{LEGACY_GGUF_MTP_CANDIDATE_BUDGET_ENV} is deprecated; use "
            f"{SPECULATIVE_CANDIDATE_BUDGET_ENV} instead. The legacy name "
            "still resolves the requested candidate depth.",
            FutureWarning,
            stacklevel=2,
        )
        return legacy
    return canonical
