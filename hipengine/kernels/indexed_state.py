"""Host admission for independent-request indexed state kernels."""
from __future__ import annotations

import operator
import numpy as np


def indexed_state_indices(state_indices: object, *, rows: int, state_slots: int,
                          max_rows: int, max_state_slots: int,
                          operation: str, state_name: str) -> np.ndarray:
    """Return bounded distinct int64 indices without mutating caller storage."""
    try:
        values = [int(operator.index(value)) for value in state_indices]
    except TypeError as error:
        raise TypeError("state_indices must be a sequence of integers") from error
    count = int(rows)
    slots = int(state_slots)
    if not 1 <= count <= max_rows:
        raise ValueError(f"indexed {operation} decode rows must be in 1..{max_rows}")
    if slots <= 0:
        raise ValueError("state_slots must be positive")
    if slots > max_state_slots:
        raise ValueError(f"state_slots must be in 1..{max_state_slots}")
    if len(values) != count:
        raise ValueError("state_indices must contain one index per row")
    seen: dict[int, int] = {}
    for row, slot in enumerate(values):
        if slot < 0:
            raise ValueError("state_indices must be non-negative")
        if slot >= slots:
            raise ValueError(
                f"state_indices[{row}]={slot} is outside the declared state slab "
                f"of {slots} slots"
            )
        if slot in seen:
            raise ValueError(
                f"state_indices duplicate slot {slot}: rows {seen[slot]} and "
                f"{row} name the same {state_name}; independent rows must "
                "select distinct slots"
            )
        seen[slot] = row
    return np.ascontiguousarray(values, dtype=np.int64)
