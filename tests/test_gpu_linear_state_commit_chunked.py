"""Raw-byte oracle for selected-state copies around the 64 KiB chunk size."""

import ctypes

import numpy as np
import pytest


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.mark.parametrize("conv_bytes,recurrent_bytes", [
    (63, 96), (65535, 65536), (65536, 131072), (65537, 131103),
    (0, 65537), (65537, 0),
])
@pytest.mark.parametrize("row", [0, 1, 3])
@pytest.mark.parametrize("offset", [0, 3])
def test_chunked_linear_commit_copies_selected_bytes_only(conv_bytes, recurrent_bytes, row, offset):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer, copy_device_to_host, copy_host_array_to_device,
        free, host_array_ptr, malloc,
    )
    from hipengine.kernels.hip_gfx1100.speculative.dflash_commit import (
        build_dflash_commit, linear_state_pair_commit_chunked_i32,
    )

    runtime = get_hip_runtime()
    buffers = []
    expected = []

    def upload(array):
        buffer = malloc(array.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_array_to_device(buffer, array, array.nbytes, runtime=runtime)
        return buffer

    try:
        tables = []
        for family, nbytes in enumerate((conv_bytes, recurrent_bytes)):
            sources, destinations = [], []
            for layer in range(3):
                indices = np.arange(4 * nbytes, dtype=np.uint64).reshape(4, nbytes)
                values = indices * 13 + np.arange(4, dtype=np.uint64)[:, None] * 47 + layer * 31 + family * 17
                source = (values % 256).astype(np.uint8).reshape(-1)
                # Include guards and optionally misalign the base addresses.
                source_storage = np.full(offset + source.size + 16, 0x5A, dtype=np.uint8)
                source_storage[offset:offset + source.size] = source
                src = upload(source_storage)
                host = np.full(offset + nbytes + 16, 0xA5, dtype=np.uint8)
                dst = upload(host)
                if nbytes:
                    host[offset:offset + nbytes] = source[row * nbytes:(row + 1) * nbytes]
                expected.append((dst, host))
                expected.append((src, source_storage))
                sources.append(src.ptr + offset)
                destinations.append(dst.ptr + offset)
            tables.extend((upload(np.asarray(sources, dtype=np.uint64)), upload(np.asarray(destinations, dtype=np.uint64))))
        selected = upload(np.asarray([row], dtype=np.int32))
        linear_state_pair_commit_chunked_i32(
            tables[0].ptr, tables[1].ptr, conv_bytes,
            tables[2].ptr, tables[3].ptr, recurrent_bytes,
            selected.ptr, 3, library=build_dflash_commit(load=True), runtime=runtime,
        )
        runtime.device_synchronize()
        for buffer, host in expected:
            actual = np.empty_like(host)
            copy_device_to_host(host_array_ptr(actual), DeviceBuffer(buffer.ptr, actual.nbytes), runtime=runtime)
            np.testing.assert_array_equal(actual, host)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
