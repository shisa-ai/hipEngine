"""Common slot-reset contract for independent-request batch state owners."""
from __future__ import annotations

import operator
from typing import Protocol, Sequence


class BatchSlotResetOwner(Protocol):
    def reset_slot(self, slot: int, *, blocks: Sequence[int] = ()) -> None: ...


def checked_slot(slot: int, state_slots: int) -> int:
    parsed = operator.index(slot)
    if not 0 <= parsed < state_slots:
        raise ValueError(f"slot {parsed} is outside [0, {state_slots})")
    return parsed


def zero_block_runs(runtime, buffers, blocks: Sequence[int], block_bytes: int) -> None:
    """Validate every destination before zeroing contiguous physical block runs."""
    indices = [operator.index(block) for block in blocks]
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    if len(set(indices)) != len(indices):
        raise ValueError("reset blocks must be distinct")
    for buffer in buffers:
        if any(block < 0 or (block + 1) * block_bytes > buffer.nbytes for block in indices):
            raise ValueError("reset block is outside the owned buffer")
    runs = []
    for block in sorted(indices):
        if runs and block == runs[-1][0] + runs[-1][1]:
            first, count = runs[-1]
            runs[-1] = (first, count + 1)
        else:
            runs.append((block, 1))
    for first, count in runs:
        for buffer in buffers:
            runtime.memset(buffer.ptr + first * block_bytes, 0, count * block_bytes)
