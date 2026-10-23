"""Unit contract for the request-local MTP draft-head token-index map.

``remap_prefix_tail_i64`` rewrites one int64 draft-head row index in place:
prefix rows pass through, tail rows resolve through the request's own extra-id
table, and every other row becomes the ``-1`` sentinel. These cases pin the
Python argument contract, the launch arguments, and the registry keys. The
device arithmetic is covered by ``tests/test_gpu_runtime_subset_token_map.py``.
"""

from __future__ import annotations

import ctypes
import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.runtime.state import (
    register_runtime_state_kernels,
    remap_prefix_tail_i64,
)
from hipengine.kernels.registry import KernelKey, resolve

KEY = KernelKey(
    "hip_gfx1100",
    "mtp_draft_token_index",
    "w4_paro",
    "remap_prefix_tail_i64",
)
ALIAS_KEY = KernelKey(
    "hip_gfx1151",
    "mtp_draft_token_index",
    "w4_paro",
    "remap_prefix_tail_i64",
)


class FakeFunction:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.calls.append(args)
        return 0


def _fake_library() -> tuple[SimpleNamespace, FakeFunction]:
    fn = FakeFunction()
    return SimpleNamespace(hipengine_remap_prefix_tail_i64=fn), fn


def test_runtime_state_registers_the_draft_token_index_map() -> None:
    register_runtime_state_kernels()

    assert resolve(
        backend=KEY.backend,
        layer=KEY.layer,
        quant=KEY.quant,
        variant=KEY.variant,
    ) is remap_prefix_tail_i64


def test_gfx1151_aliases_the_primitive_without_a_backend_branch() -> None:
    from hipengine.kernels import hip_gfx1151

    hip_gfx1151.register_gfx1151_kernels()

    assert resolve(
        backend=ALIAS_KEY.backend,
        layer=ALIAS_KEY.layer,
        quant=ALIAS_KEY.quant,
        variant=ALIAS_KEY.variant,
    ) is remap_prefix_tail_i64
    source = inspect.getsource(remap_prefix_tail_i64)
    assert "hip_gfx" not in source
    assert "w4_paro" not in source


def test_launch_forwards_index_table_and_counts() -> None:
    library, fn = _fake_library()

    remap_prefix_tail_i64(
        0x1000,
        0x2000,
        128,
        8,
        stream=0x30,
        library=library,
        runtime=object(),
    )

    assert fn.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    assert len(fn.calls) == 1
    index_ptr, extra_ptr, prefix_size, extra_count, stream = fn.calls[0]
    assert index_ptr.value == 0x1000
    assert extra_ptr.value == 0x2000
    assert prefix_size.value == 128
    assert extra_count.value == 8
    assert stream.value == 0x30


def test_zero_extra_count_passes_a_null_table() -> None:
    library, fn = _fake_library()

    remap_prefix_tail_i64(0x1000, None, 6, 0, library=library, runtime=object())

    index_ptr, extra_ptr, prefix_size, extra_count, stream = fn.calls[0]
    assert index_ptr.value == 0x1000
    assert extra_ptr.value is None
    assert prefix_size.value == 6
    assert extra_count.value == 0
    # The default stream is the null stream pointer.
    assert stream.value is None


def test_extra_only_geometry_is_accepted() -> None:
    library, fn = _fake_library()

    remap_prefix_tail_i64(0x1000, 0x2000, 0, 3, library=library, runtime=object())

    _, extra_ptr, prefix_size, extra_count, _ = fn.calls[0]
    assert extra_ptr.value == 0x2000
    assert prefix_size.value == 0
    assert extra_count.value == 3


def test_invalid_counts_and_missing_table_are_rejected_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="prefix_size"):
        remap_prefix_tail_i64(0, 0, -1, 4)
    with pytest.raises(ValueError, match="extra_count"):
        remap_prefix_tail_i64(0, 0, 4, -1)
    with pytest.raises(ValueError, match="must be positive"):
        remap_prefix_tail_i64(0, None, 0, 0)
    with pytest.raises(ValueError, match="extra_ids_i64_ptr"):
        remap_prefix_tail_i64(0, None, 4, 2)
