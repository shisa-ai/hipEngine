"""Selected subset scores preserve full-head logits, including padded tables."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import malloc, free, host_array_ptr, copy_host_to_device, copy_device_to_host
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.loading.materialize import DeviceTensorAllocation
from hipengine.loading.qwen4_exp_materialize import Qwen4ExpDeviceWeight
from hipengine.runtime.gguf_linear import launch_gguf_linear
from hipengine.runtime.mtp_draft_head import MTPSubsetHead


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason='HIP runtime unavailable')


@pytest.mark.parametrize('extra_count', [0, 1, 7, 513, 1024])
def test_subset_scores_and_global_token_mapping_are_exact(extra_count):
    runtime = get_hip_runtime()
    load_backend_kernel_package('hip_gfx1100')
    vocab, hidden, prefix = 4096, 2560, 1024
    rng = np.random.default_rng(117 + extra_count)
    weights = np.empty((vocab, hidden // 32, 34), dtype=np.uint8)
    scales = rng.uniform(.001, .02, (vocab, hidden // 32)).astype(np.float16)
    weights[:, :, :2] = scales.view(np.uint8).reshape(vocab, hidden // 32, 2)
    weights[:, :, 2:] = rng.integers(-100, 101, (vocab, hidden // 32, 32), dtype=np.int8).view(np.uint8)
    buffers = []
    head = None

    def upload(host):
        host = np.ascontiguousarray(host)
        buffer = malloc(host.nbytes, runtime=runtime)
        buffers.append(buffer)
        copy_host_to_device(buffer, host_array_ptr(host), host.nbytes, runtime=runtime)
        return buffer

    try:
        raw = upload(weights)
        spec = SimpleNamespace(layout='raw_gguf', quant_key='gguf_q8_0', slot_path='root.lm_head')
        allocation = DeviceTensorAllocation('raw', None, raw,
            Tensor.from_handle(raw.ptr, (raw.nbytes,), DType.INT8, 'hip'))
        weight = Qwen4ExpDeviceWeight(spec, 'hip_gfx1100', {'raw': allocation})
        x = upload(rng.standard_normal(hidden).astype(np.float32))
        full = upload(np.zeros(vocab, dtype=np.float32))
        compact = upload(np.full(vocab, np.nan, dtype=np.float32))
        token = upload(np.zeros(1, dtype=np.int64))
        ids = rng.choice(np.arange(prefix, vocab), extra_count, replace=False).tolist()
        head = MTPSubsetHead(weight, vocab_size=vocab, hidden_size=hidden,
                             special_ids=ids[:1], runtime=runtime, prefix_size=prefix)
        head.prepare(ids[1:] + [0] * 40, anchor=0)
        launch_gguf_linear(weight, x.ptr, full.ptr, 1, hidden, vocab,
                           activation_dtype='f32', output_dtype='f32', runtime=runtime)
        count = head.project(x.ptr, compact.ptr)
        assert count == prefix + extra_count
        expected = np.empty(vocab, dtype=np.float32)
        actual = np.empty(count, dtype=np.float32)
        copy_device_to_host(host_array_ptr(expected), full, expected.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(actual), compact, actual.nbytes, runtime=runtime)
        selected = list(range(prefix)) + ids
        np.testing.assert_array_equal(actual, expected[selected])
        for compact_id in (0, prefix - 1, count - 1):
            host = np.asarray([compact_id], dtype=np.int64)
            copy_host_to_device(token, host_array_ptr(host), host.nbytes, runtime=runtime)
            head.remap(token.ptr)
            copy_device_to_host(host_array_ptr(host), token, host.nbytes, runtime=runtime)
            assert host[0] == selected[compact_id]
    finally:
        if head is not None:
            head.close()
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
