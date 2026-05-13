from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType, dtype_itemsize
from hipengine.runtime import RuntimeWorkspace, tensor_nbytes


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x8000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1) + 0x100
        self.allocations[ptr] = nbytes
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.allocations.pop(ptr, None)


def test_dtype_itemsize_and_tensor_nbytes_are_fixed_for_workspace_dtypes() -> None:
    assert dtype_itemsize("bf16") == 2
    assert dtype_itemsize(DType.FP32) == 4
    assert tensor_nbytes((2, 3), "int16") == 12
    assert tensor_nbytes((0, 3), "fp16") == 0
    with pytest.raises(ValueError, match="fixed element byte size"):
        dtype_itemsize("int4_paro")


def test_runtime_workspace_reuses_exact_tensor_spec() -> None:
    runtime = FakeRuntime()
    workspace = RuntimeWorkspace(device=Device("hip", 0), runtime=runtime)

    first = workspace.reserve_tensor("attn", (16, 256), "fp32")
    second = workspace.reserve_tensor("attn", (16, 256), "fp32")

    assert first is second
    assert first.ptr == 0x8000
    assert workspace.allocation("attn").buffer.nbytes == 16 * 256 * 4
    assert runtime.freed == []
    assert workspace.names == ("attn",)


def test_runtime_workspace_replaces_changed_tensor_spec_and_frees() -> None:
    runtime = FakeRuntime()
    workspace = RuntimeWorkspace(runtime=runtime)

    first = workspace.reserve_tensor("scratch", (4,), "fp32")
    second = workspace.reserve_tensor("scratch", (8,), "fp32")

    assert second.ptr != first.ptr
    assert runtime.freed == [first.ptr]
    assert workspace.allocation("scratch").buffer.nbytes == 8 * 4


def test_runtime_workspace_free_releases_in_reverse_order() -> None:
    runtime = FakeRuntime()
    workspace = RuntimeWorkspace(runtime=runtime)
    a = workspace.reserve_tensor("a", (1,), "int32")
    b = workspace.reserve_tensor("b", (2,), "int32")

    workspace.free()

    assert runtime.freed == [b.ptr, a.ptr]
    assert workspace.names == ()


def test_runtime_workspace_validates_names_and_shapes() -> None:
    workspace = RuntimeWorkspace(runtime=FakeRuntime())
    with pytest.raises(ValueError, match="name"):
        workspace.reserve_tensor("", (1,), "fp32")
    with pytest.raises(ValueError, match="non-negative"):
        workspace.reserve_tensor("bad", (-1,), "fp32")
