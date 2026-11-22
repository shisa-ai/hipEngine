import ctypes
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections
from hipengine.quant.gguf import GGMLQuantizationType as Q
from tests.test_unit_inkling2_gguf import _file


@pytest.mark.parametrize("legacy", [True, False])
def test_inkling2_gguf_packed_projections_preserve_scale_and_route_order(legacy):
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")

    rng = np.random.default_rng(1983)
    reader = Mock()
    reader.info = _file(legacy)
    weights = Inkling2GGUFWeights(reader)
    mapping = weights.mapping
    matrices = [mapping.root["output.weight"], mapping.layers[2].gate_up.weight,
                mapping.layers[2].down.weight, mapping.layers[4].gate_up.weight, mapping.layers[4].down.weight]
    raw = {}
    for t in matrices:
        blocks = rng.integers(0, 256, t.byte_shape, dtype=np.uint8)
        if t.ggml_type == Q.NVFP4:
            blocks.reshape(-1, 36)[:, :4] = rng.integers(16, 63, (blocks.size // 36, 4), dtype=np.uint8)
        elif t.ggml_type == Q.Q8_0:
            blocks.reshape(-1, 34)[:, :2] = np.array([0.003], dtype="<f2").view(np.uint8)
        else:
            blocks.reshape(-1, 176)[:, :4] = np.array([0.002, 0.001], dtype="<f2").view(np.uint8)
        raw[t.name] = blocks
    raw[mapping.layers[2].gate_up.scale.name] = np.array([0.63, 0.33], np.float32)
    raw[mapping.layers[2].down.scale.name] = np.array([0.41, 0.72], np.float32)
    reader.tensor_data.side_effect = raw.__getitem__
    x = rng.normal(0, 0.1, (3, 256)).astype(np.float32)
    expert_ids = np.array([1, 0, 1], np.int32)
    blas = Rocblas.load()
    try:
        with Inkling2GGUFDeviceProjections(weights, runtime=get_hip_runtime(), blas=blas, backend=backend) as owner:
            head = matrices[0]
            np.testing.assert_allclose(owner.project(head.name, x, limit=mapping.vocab_size),
                                       x @ weights.decode(head, slice(mapping.vocab_size)).T, rtol=1e-5, atol=2e-6)
            for visit in (mapping.plan.visits[2], mapping.plan.visits[4], mapping.plan.visits[6]):
                for projection in ("gate_up", "down"):
                    observed = owner.project_selected(visit, projection, expert_ids, x)
                    expected = np.stack([weights.project_expert(visit, projection, expert, h)
                                         for expert, h in zip(expert_ids, x, strict=True)])
                    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=2e-5)
            assert owner.packed_bank_bytes == sum(t.nbytes for t in matrices[1:])
            # The final physical banks have different formats in the two layouts.
            visit = mapping.plan.visits[6]
            layer = mapping.layers[visit.weight_layer]
            for stem in ("gate_up", "down"):
                bank = getattr(layer, stem)
                quant = Q(bank.weight.ggml_type)
                launch = owner._selected_launches[quant]
                spy = Mock(wraps=launch)
                owner._selected_launches[quant] = spy
                try:
                    grouped = owner.project_selected(visit, stem, expert_ids, x)
                    assert spy.call_count == 1
                    owner._selected_launches.pop(quant)
                    scalar = owner.project_selected(visit, stem, expert_ids, x)
                finally:
                    owner._selected_launches[quant] = launch
                np.testing.assert_array_equal(grouped.view(np.uint32), scalar.view(np.uint32))
    finally:
        blas.close()
