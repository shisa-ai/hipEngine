"""GPU byte oracle for gfx1151 journal snapshot, serial restore and rollback."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free, copy_host_array_to_device, copy_device_to_host, host_array_ptr
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime.qwen35_gguf_mtp import _StateJournal


def available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


@pytest.mark.skipif(not available(), reason='HIP runtime required')
@pytest.mark.parametrize('initial_only', [False, True])
@pytest.mark.parametrize('conv_bytes,recurrent_bytes', [(63, 65537), (163840, 3145728)])
def test_gfx1151_journal_restores_initial_and_serial_rows(initial_only, conv_bytes, recurrent_bytes):
    register_gfx1151_kernels()
    runtime = get_hip_runtime()
    buffers = []
    journal = None
    rng = np.random.default_rng(17)
    def upload(nbytes):
        data = rng.integers(0, 256, nbytes, dtype=np.uint8)
        buf = malloc(nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_array_to_device(buf, data, runtime=runtime)
        return buf, data
    def read(buf):
        result = np.empty(buf.nbytes, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(result), buf, runtime=runtime)
        return result
    try:
        conv = [upload(conv_bytes) for _ in range(3)]
        rec = [upload(recurrent_bytes) for _ in range(3)]
        hidden, hidden_data = upload(8)
        destination, _ = upload(8)
        target = SimpleNamespace(
            backend='hip_gfx1151', runtime=runtime, runner=SimpleNamespace(hidden_size=4),
            scratch=SimpleNamespace(layer_conv_states=tuple(b for b,_ in conv),
                                    layer_recurrent_states=tuple(b for b,_ in rec)),
            _target_scratch_owner=object(), last_target_hidden=hidden, _hidden_a=destination,
            _dflash_commit_library=None, compiler_version=None, require_cached_build=False,
        )
        journal = _StateJournal.allocate(target, max_rows=4, initial_state_only=initial_only)
        assert journal.initial_state_copy is not None
        journal.capture_initial()
        for buf,_ in conv+rec:
            runtime.memset_async(buf.ptr, 0xA5, buf.nbytes, 0)
        if not initial_only:
            journal.capture_row(2)
            for buf,_ in conv+rec:
                runtime.memset_async(buf.ptr, 0, buf.nbytes, 0)
            journal.restore_row(2)
            runtime.device_synchronize()
            for buf,_ in conv+rec:
                np.testing.assert_array_equal(read(buf), np.full(buf.nbytes, 0xA5, dtype=np.uint8))
        journal.restore_initial()
        runtime.device_synchronize()
        for buf,data in conv+rec:
            np.testing.assert_array_equal(read(buf), data)
        np.testing.assert_array_equal(read(destination), hidden_data)
    finally:
        runtime.device_synchronize()
        if journal is not None:
            journal.close()
        for buf in reversed(buffers):
            free(buf, runtime=runtime)
