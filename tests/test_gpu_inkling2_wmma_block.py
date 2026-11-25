"""Prompt blocks of the packed Inkling GGUF expert path through the grouped WMMA projection.

A block of 192 or more token rows goes through the grouped WMMA kernels instead of the FP32 tile
kernels; narrower blocks and tails stay on the tiles. The reference is the strict owner: FP32 tile projection, NumPy activation, ordered mix.
"""

import ctypes
from contextlib import ExitStack
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind as Copy, get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.cpu_reference.inkling2_mini import _sigmoid
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections

from tests.test_unit_inkling2_gguf import _file

def _block(legacy, rows, seed):
    rng = np.random.default_rng(seed)
    reader = Mock()
    reader.info = _file(legacy)
    weights = Inkling2GGUFWeights(reader)
    mapping = weights.mapping
    layer = mapping.layers[2]
    raw = {}
    for t in (layer.gate_up.weight, layer.down.weight):
        blocks = rng.integers(0, 256, t.byte_shape, dtype=np.uint8)
        blocks.reshape(-1, 36)[:, :4] = rng.integers(16, 63, (blocks.size // 36, 4), dtype=np.uint8)
        raw[t.name] = blocks
    raw[layer.gate_up.scale.name] = np.array([0.63, 0.33], np.float32)
    raw[layer.down.scale.name] = np.array([0.41, 0.72], np.float32)
    reader.tensor_data.side_effect = raw.__getitem__
    ids = rng.integers(0, 2, (rows, 8)).astype(np.int32)
    mix = rng.uniform(0.0, 1.0, (rows, 8)).astype(np.float32)
    hidden = rng.normal(0, 0.1, (rows, mapping.layers[0].gate_up.weight.shape[2])).astype(np.float32)
    return weights, mapping, ids, mix, hidden

def _reference(owner, visit, mapping, ids, mix, hidden):
    expanded = np.repeat(hidden, ids.shape[-1], axis=0)
    packed = owner.project_selected(visit, "gate_up", ids.ravel(), expanded)
    gate, up = np.split(packed, 2, axis=-1)
    a, b = mapping.gate_cap, mapping.up_cap
    activated = (a * np.tanh(gate / a) * _sigmoid(gate)) * (b * np.tanh(up / b))
    return owner.project_down_mixed(visit, ids, mix, activated)

@pytest.mark.parametrize("legacy", [True, False])
@pytest.mark.parametrize("rows", [192, 200, 257])
def test_wide_prompt_block_matches_the_fp32_tile_path(legacy, rows):
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    weights, mapping, ids, mix, hidden = _block(legacy, rows, 211 + rows)
    visit = mapping.plan.visits[2]
    blas = Rocblas.load()
    options = dict(runtime=get_hip_runtime(), blas=blas, backend=backend)
    with Inkling2GGUFDeviceProjections(weights, **options) as owner:
        want = _reference(owner, visit, mapping, ids, mix, hidden)
        with Inkling2GGUFDeviceProjections(weights, fast_experts=True, **options) as fast:
            assert fast._wmma is not None and fast._wmma_rows(visit, ((ids, mix, hidden, None),))
            # Narrow blocks and tails stay on the FP32 tile kernels.
            assert not fast._wmma_rows(visit, ((ids[:64], mix[:64], hidden[:64], None),))
            assert not fast._wmma_rows(visit, ((ids[:160], mix[:160], hidden[:160], None),))
            launches = fast._wmma
            fast._wmma = (Mock(side_effect=launches[0]), Mock(side_effect=launches[1]))
            uploads = fast._transfer.enqueue_upload = Mock(wraps=fast._transfer.enqueue_upload)
            downloads = fast._transfer.download = Mock(wraps=fast._transfer.download)
            consumer = Mock()
            actual = fast.project_experts_mixed(visit, ids, mix, hidden, transform_output=consumer)
            assert fast._wmma[0].call_count == 2 and fast._wmma[1].call_count == 2
            uploads.assert_called_once()
            downloads.assert_called_once()
            consumer.assert_called_once()
            assert consumer.call_args.args[1] == want.shape
            fast._wmma = launches
    # FP32 accumulation in another order: small outputs that cancel carry the same absolute error
    # as large ones, so the tolerance follows the largest output of the block.
    np.testing.assert_allclose(actual, want, rtol=1e-4, atol=1e-5 * np.abs(want).max())
    np.testing.assert_array_equal(actual.argmax(axis=-1), want.argmax(axis=-1))

@pytest.mark.parametrize("legacy", [True, False])
def test_wide_prompt_block_leaves_its_result_resident(legacy):
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    weights, mapping, ids, mix, hidden = _block(legacy, 200, 307)
    visit = mapping.plan.visits[2]
    rt = get_hip_runtime()
    blas = Rocblas.load()
    options = dict(runtime=rt, blas=blas, backend=backend)
    width = mapping.layers[0].down.weight.shape[1]
    host_result = np.full((200, width), np.nan, np.float32)
    with Inkling2GGUFDeviceProjections(weights, **options) as owner:
        want = _reference(owner, visit, mapping, ids, mix, hidden)
    with ExitStack() as owned:
        hidden_buffer = rt.malloc(hidden.nbytes)
        result_buffer = rt.malloc(host_result.nbytes)
        owned.callback(rt.free, hidden_buffer)
        owned.callback(rt.free, result_buffer)
        rt.memcpy(hidden_buffer, hidden.ctypes.data, hidden.nbytes, Copy.HOST_TO_DEVICE)
        rt.memcpy(result_buffer, host_result.ctypes.data, host_result.nbytes, Copy.HOST_TO_DEVICE)
        with Inkling2GGUFDeviceProjections(weights, fast_experts=True, **options) as fast:
            assert fast._wmma is not None
            downloads = fast._transfer.download = Mock(wraps=fast._transfer.download)
            out = fast.project_experts_mixed_rows(visit, ((ids, mix, hidden, None),),
                                                  device=(hidden_buffer, result_buffer))
            assert out == ()
            downloads.assert_not_called()
        rt.memcpy(host_result.ctypes.data, result_buffer, host_result.nbytes, Copy.DEVICE_TO_HOST)
    assert np.isfinite(host_result).all()
    np.testing.assert_allclose(host_result, want, rtol=1e-4, atol=1e-5 * np.abs(want).max())
    np.testing.assert_array_equal(host_result.argmax(axis=-1), want.argmax(axis=-1))
