"""What actually goes wrong with an inline H2D source, and what does not.

Two claims about ``copy_host_to_device(buf, host_array_ptr(...))`` were made in
this repo, and only one of them survives measurement:

1. *The inline temporary is a use-after-free.* ``copy_host_to_device`` takes a
   bare integer address, so the array has to keep itself alive across the call.
   CPython drops the temporary's last reference when ``host_array_ptr`` returns,
   so by the time the copy function is entered the array is already gone and its
   block is available to the next same-size allocation.  No device-side timing is
   involved.  This is what broke
   ``tests/test_gpu_surya_kv_spans.py::test_scatter_f32_spans_honors_page_table_and_eviction``
   depending on which test ran before it.

2. *The transfer reads the source after the copy returns.* This is **false** on
   gfx1151: overwriting the source in place immediately after
   ``copy_host_to_device`` returns leaves the destination untouched at 1, 16, 64,
   and 128 MiB.  A named local is therefore sufficient, and a per-call
   ``device_synchronize()`` buys nothing for source lifetime.  The runners that
   keep one do so on load-time or cache-miss paths, defensively.

Test 1 is pure host-side lifetime and needs no GPU; test 3 is the device-side
measurement and skips without ROCm.
"""

from __future__ import annotations

import ctypes
import gc
import weakref

import numpy as np
import pytest

from hipengine.core.memory import host_array_ptr


def _fresh(size: int) -> np.ndarray:
    """A factory, so the allocation is not visible to the AST guard.

    ``host_array_ptr(_fresh())`` is an inline temporary like any other, but the
    guard in ``tests/test_gpu_device_memory_hygiene.py`` sees a call to an unknown
    name and cannot know it allocates.  That is a real limit of a source-level
    lint, and this test is the demonstration that the limit is not theoretical.
    """

    return np.zeros(size, dtype=np.uint8)


def test_host_array_ptr_is_just_the_data_address() -> None:
    """The helper keeps no reference, which is the whole problem."""

    array = np.zeros(16, dtype=np.uint8)
    assert host_array_ptr(array) == int(array.ctypes.data)


def test_inline_temporary_is_dead_before_the_copy_is_entered() -> None:
    """The load-bearing assertion: dead at call entry, not after return."""

    watched: list[weakref.ref] = []
    alive_at_entry: list[bool] = []

    def host_pointer(array: object) -> int:
        watched.append(weakref.ref(array))
        return host_array_ptr(array)

    def copy_host_to_device(buffer: object, host_ptr: int, nbytes: int) -> None:
        # First statement of the copy that receives the address.
        alive_at_entry.append(any(ref() is not None for ref in watched))

    copy_host_to_device(None, host_pointer(_fresh(4096)), 4096)

    assert watched and watched[0]() is None, (
        "the inline source outlived host_array_ptr; if this ever changes, the "
        "use-after-free argument in docs/KERNELS.md no longer applies"
    )
    assert alive_at_entry == [False], (
        "the inline source was still alive when the copy was entered"
    )


def test_the_freed_block_is_available_to_the_next_allocation() -> None:
    """The consequence: what the copy would read is somebody else's data."""

    pointer = host_array_ptr(_fresh(4096))
    gc.collect()
    reused = None
    for _ in range(8):
        candidate = np.full(4096, 0xBB, dtype=np.uint8)
        if int(candidate.ctypes.data) == pointer:
            reused = candidate
            break
    if reused is None:
        pytest.skip("allocator did not recycle the address; weakref test proves lifetime independently")
    # Reading through the stale address now yields the new owner's bytes.
    stale = np.frombuffer(ctypes.string_at(pointer, 4096), dtype=np.uint8)
    assert (stale == 0xBB).all()


def test_a_named_local_is_alive_for_the_whole_call() -> None:
    """The fix, stated as the property the guard enforces."""

    watched: list[weakref.ref] = []
    alive_at_entry: list[bool] = []

    def copy_host_to_device(buffer: object, host_ptr: int, nbytes: int) -> None:
        alive_at_entry.append(any(ref() is not None for ref in watched))

    source = _fresh(4096)
    watched.append(weakref.ref(source))
    copy_host_to_device(None, host_array_ptr(source), 4096)
    assert alive_at_entry == [True]


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime not available")
@pytest.mark.parametrize("mib", [1, 16, 64, 128])
def test_transfer_is_complete_when_the_copy_returns(mib: int) -> None:
    """Independent justification (or refutation) of the synchronization.

    Overwrite the source *in place* -- same memory, no free -- the instant the
    copy returns.  A transfer that were still reading the host buffer would pick
    up the new bytes.  All-original bytes means the source is not needed after
    the call, so a named local needs no ``device_synchronize()``.
    """

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        malloc,
    )

    runtime = get_hip_runtime()
    n = mib * 1024 * 1024
    buffer = malloc(n)
    try:
        source = np.full(n, 0xAA, dtype=np.uint8)
        copy_host_to_device(buffer, host_array_ptr(source), n)
        source[:] = 0xBB
        runtime.device_synchronize()
        back = np.empty(n, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(back), buffer, n)
        assert (back == 0xAA).all(), (
            f"{mib} MiB: the copy was still reading the host source after it "
            "returned; a named local is not sufficient and the runners need a "
            "synchronize after every upload"
        )
    finally:
        free(buffer)
