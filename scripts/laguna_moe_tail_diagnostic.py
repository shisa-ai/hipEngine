"""Owner-scoped Laguna MoE-tail comparison without a production selector."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator


@contextmanager
def unfused_moe_tail(session) -> Iterator[None]:
    """Use the registered add/add/RMSNorm chain on one diagnostic session."""
    original = session.kernel_plan
    session.kernel_plan = replace(original, moe_tail_next_rmsnorm=None)
    try:
        yield
    finally:
        session.kernel_plan = original
