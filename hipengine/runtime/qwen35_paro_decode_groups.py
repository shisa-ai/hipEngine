"""Execute PARO decode width groups without owning scheduling or sampling policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ParoDecodeGroupResult:
    results: tuple[Any, ...]
    effective_groups: tuple[tuple[str, int], ...]
    native_fallback_rows: int


def execute_paro_decode_groups(
    session: Any,
    token_ids: Sequence[int],
    *,
    positions: Sequence[int],
    slots: Sequence[int],
    groups: Sequence[tuple[str, int]],
    native_kwargs: Mapping[str, Any] | None = None,
    check_control: Callable[[], None] | None = None,
) -> ParoDecodeGroupResult:
    """Run an ordered native/serial cover, retrying only native capability misses."""
    row_count = len(token_ids)
    if len(positions) != row_count or len(slots) != row_count:
        raise ValueError("PARO decode tokens, positions and slots must have equal lengths")
    if any(mode not in {"native", "serial"} or width <= 0 for mode, width in groups):
        raise ValueError("PARO decode groups require native/serial modes and positive widths")
    if sum(width for _mode, width in groups) != row_count:
        raise ValueError("PARO decode width partition must consume every live row")
    native_options = dict(native_kwargs or {})
    results: list[Any] = []
    effective_groups: list[tuple[str, int]] = []
    fallback_rows = 0
    cursor = 0
    for mode, width in groups:
        end = cursor + width
        tokens = [int(token) for token in token_ids[cursor:end]]
        options = {
            "positions": [int(position) for position in positions[cursor:end]],
            "slots": [int(slot) for slot in slots[cursor:end]],
            "sample": True,
        }
        effective_mode = mode
        if check_control is not None:
            check_control()
        if mode == "native":
            try:
                group_results = session.step_batch_native(tokens, **options, **native_options)
            except NotImplementedError:
                effective_mode = "serial"
                fallback_rows += width
                if check_control is not None:
                    check_control()
                group_results = session.step_batch_serial(tokens, **options)
        else:
            group_results = session.step_batch_serial(tokens, **options)
        if check_control is not None:
            check_control()
        if len(group_results) != width:
            raise RuntimeError(
                f"decode group returned {len(group_results)} results for {width} requests"
            )
        results.extend(group_results)
        effective_groups.append((effective_mode, width))
        cursor = end
    return ParoDecodeGroupResult(tuple(results), tuple(effective_groups), fallback_rows)
