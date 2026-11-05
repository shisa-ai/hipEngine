"""Shape and registration misses preserve strict pair fallback in both roles."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as linear


@pytest.mark.parametrize('bulk', (False, True))
@pytest.mark.parametrize('quant,abi,width,outputs,registered', (
    ('gguf_q4_k_t16_v1', 't16', 5120, 17408, False),
    ('gguf_q4_k_t16_v1', 't16', 4096, 17408, True),
    ('gguf_q4_k_t16_v1', 't16', 5120, 16384, True),
    ('gguf_q6_k_t16_v1', 't16', 5120, 17408, True),
    ('gguf_q4_k_t16_v1', 'raw', 5120, 17408, True),
))
def test_bulk_ffn_policy_keeps_misses(monkeypatch, bulk, quant, abi, width, outputs, registered):
    monkeypatch.setattr(linear, 'physical_exact_rowtiles_enabled', lambda: False)
    monkeypatch.setattr(linear, '_q4_t16_dual_silu_retile_enabled', lambda: False)
    monkeypatch.setattr(linear, '_ensure_linear_kernel_registered', lambda key: None)
    monkeypatch.setattr(linear, 'is_registered', lambda key: registered)
    dispatch = SimpleNamespace(
        key=linear.KernelKey('hip_gfx1100', 'linear', quant, 'test'), abi=abi,
    )
    with linear.bulk_prefill_ffn_session(bulk):
        assert linear._q4_t16_dual_wmma_silu_dispatch(
            dispatch, dispatch, rows=57, in_features=width, out_features=outputs,
        ) is None
