"""Owner lifetime and bounds at the synchronous array upload boundary."""
import ctypes
import ast
from pathlib import Path
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core import memory


@pytest.mark.parametrize("module", ["evie", "timesfm_decode", "timesfm3_decode", "qwen35_gguf_runner"])
def test_reviewed_runners_do_not_erase_temporary_array_owners(module):
    """Bounded syntax guard, not a proof of arbitrary pointer lifetimes."""
    path = Path(__file__).parents[1] / "hipengine" / "runtime" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = [node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "host_array_ptr"
               and node.args and isinstance(node.args[0], ast.Call)]
    assert not hazards, f"{path.name}: retain array owners at lines {hazards}"


def test_upload_retains_temporary_through_copy():
    refs = []

    def source():
        array = np.arange(8, dtype=np.int32)
        refs.append(weakref.ref(array))
        return array

    def memcpy(dst, src, nbytes, kind):
        assert refs[0]() is not None
        assert nbytes == 32
        assert list((ctypes.c_int32 * 8).from_address(src)) == list(range(8))

    memory.copy_host_array_to_device(
        memory.DeviceBuffer(1234, 64), source(),
        runtime=SimpleNamespace(memcpy=memcpy),
    )
    assert refs[0]() is None


@pytest.mark.parametrize("case", ["source_bounds", "destination_bounds", "strided"])
def test_upload_rejects_invalid_range_before_runtime(case):
    array = np.arange(8, dtype=np.int32)
    buffer = memory.DeviceBuffer(1234, 32)
    nbytes = 36 if case == "source_bounds" else None
    if case == "destination_bounds":
        buffer = memory.DeviceBuffer(1234, 16)
    if case == "strided":
        array = array[::2]
    with pytest.raises(ValueError):
        memory.copy_host_array_to_device(buffer, array, nbytes, runtime=object())


def test_upload_accepts_owned_contiguous_view_and_explicit_prefix():
    array = np.arange(8, dtype=np.int32)[2:]
    seen = []
    memory.copy_host_array_to_device(
        memory.DeviceBuffer(1234, 8), array, 8,
        runtime=SimpleNamespace(memcpy=lambda dst, src, size, kind: seen.append(
            list((ctypes.c_int32 * 2).from_address(src)))),
    )
    assert seen == [[2, 3]]


def test_evie_pointer_cache_uploads_only_new_content(monkeypatch):
    from hipengine.runtime import evie

    uploads = []
    allocations = []

    def allocate(nbytes):
        buffer = memory.DeviceBuffer(4096 + len(allocations) * 4096, nbytes)
        allocations.append(buffer)
        return buffer

    def upload(buffer, ptr, nbytes=None, **kwargs):
        uploads.append(list((ctypes.c_uint64 * (nbytes // 8)).from_address(ptr)))

    monkeypatch.setattr(evie, "_malloc_committed", allocate)
    monkeypatch.setattr(evie, "copy_host_to_device", upload)
    runner = object.__new__(evie.EvieRunner)
    a = runner._dev_ptr_array([123, 456])
    assert runner._dev_ptr_array([123, 456]) == a
    assert runner._dev_ptr_array([456, 123]) != a
    assert uploads == [[123, 456], [456, 123]]
