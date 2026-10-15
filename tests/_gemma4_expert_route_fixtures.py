"""Shared fixture doubles for the Gemma 4 routed-expert route tests.

``_ResidentWeight`` models the attributes the grouped and selected routes read
off a resident quantized weight. CPU tests must stub every reachable kernel;
device tests supply real allocated pointers.

Both the CPU route-decision tests in ``test_unit_gemma4_expert_route.py`` and
the device-execution tests in ``test_gpu_gemma4_expert_route.py`` build on it,
so it lives here instead of in either collected test module.
"""

from __future__ import annotations

from types import SimpleNamespace


class _ResidentWeight:
    """The attributes the grouped route reads off a resident quantized weight."""

    def __init__(self, *, backend: str, quant_key: str, ptr: int = 0x5A0000) -> None:
        self.backend = backend
        self.spec = SimpleNamespace(quant_key=quant_key)
        self.ptr = ptr

    def allocation(self, name: str | None = None) -> SimpleNamespace:
        # ``None`` selects the primary allocation, which for this double is the
        # raw blocks it models; the selected-route walk passes ``None`` for every
        # raw candidate and names ``tiles`` only for the tile sibling.
        assert name in (None, "raw"), "the grouped route reads the artifact's raw bytes"
        return SimpleNamespace(buffer=SimpleNamespace(ptr=self.ptr))

    def has_allocation(self, name: str) -> bool:
        # This double models a weight carrying only the raw blocks, which is what
        # a rank-2 tensor or a non-Q4_K expert tensor plans. The pack8 route must
        # decline it, so the ladder falls through to the raw selected route
        # rather than looking up a packed allocation that was never planned.
        return name == "raw"
