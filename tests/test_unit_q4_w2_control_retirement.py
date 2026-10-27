"""Two-wave Q4 physical admission follows backend geometry, not old env."""
import pytest
from hipengine.kernels.hip_gfx1100.quant import gguf_k_t16_selected_prefill as kernels


@pytest.mark.parametrize('value', ['0', 'invalid'])
def test_retired_w2_flag_preserves_admitted_leaf(monkeypatch, value):
    monkeypatch.setenv('HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2', value)
    if hasattr(kernels, '_Q4_ROWTILE16_W2_RESOLVED'):
        monkeypatch.setattr(kernels, '_Q4_ROWTILE16_W2_RESOLVED', None)
    assert kernels._q4_physical_rowtile(6, (6144, 5120)) is (
        kernels.gguf_q4_k_t16_dense_rowtile16_w2_bf16_bf16_out)
    assert kernels._q4_physical_rowtile(5, (6144, 5120)) is (
        kernels.gguf_q4_k_t16_dense_rowtile_bf16_bf16_out)
