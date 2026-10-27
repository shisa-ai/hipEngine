"""Independent token-serial Laguna prefill for correctness and benchmark controls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from hipengine.runtime.laguna_gguf_runner import (
        LagunaEagerTokenResult,
        LagunaGGUFResidentSession,
        LagunaHiddenCaptureTargets,
    )


def serial_prefill(
    session: LagunaGGUFResidentSession,
    token_ids: Sequence[int],
    *,
    capture_last: LagunaHiddenCaptureTargets | None = None,
    stream: int = 0,
) -> LagunaEagerTokenResult:
    """Advance the scalar primitive, capturing only the final prompt token."""
    tokens = tuple(int(value) for value in token_ids)
    if not tokens:
        raise ValueError("Laguna serial prefill requires at least one token")
    if capture_last is not None and capture_last.rows != 1:
        raise ValueError("Laguna serial prefill capture_last requires one-row targets")
    result = None
    for index, token in enumerate(tokens):
        result = session.forward_token(
            token,
            captures=capture_last if index == len(tokens) - 1 else None,
            stream=stream,
        )
    assert result is not None
    return result
