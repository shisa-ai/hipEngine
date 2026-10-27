"""Shared admission preserves input and primitive-specific error context."""
import numpy as np
import pytest
from hipengine.kernels.indexed_state import indexed_state_indices
from hipengine.kernels.hip_gfx1100.linear_attn import conv, qwen4_exp_gdn
from hipengine.kernels.hip_gfx1100.fused import qwen4_exp_ple


@pytest.mark.parametrize('module,name', [
    (conv, 'qwen4_exp_conv_indexed_state_indices'),
    (qwen4_exp_gdn, 'qwen4_exp_gdn_indexed_state_indices'),
    (qwen4_exp_ple, 'qwen4_exp_ple_indexed_state_indices'),
])
def test_primitives_delegate_to_shared_admission(module, name, monkeypatch):
    calls = []
    expected = object()
    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return expected
    monkeypatch.setattr(module, 'indexed_state_indices', record)
    assert getattr(module, name)((2, 0), rows=2, state_slots=3) is expected
    assert calls[0][1]['rows'] == 2
    assert calls[0][1]['state_slots'] == 3


def test_shared_admission_does_not_mutate_input_on_success_or_refusal():
    for values in ([2, 0], [1, 1]):
        original = np.asarray(values, dtype=np.int32)
        before = original.copy()
        kwargs = dict(rows=2, state_slots=3, max_rows=8, max_state_slots=8,
                      operation='test', state_name='test state')
        if values == [1, 1]:
            with pytest.raises(ValueError, match='duplicate slot 1'):
                indexed_state_indices(original, **kwargs)
        else:
            result = indexed_state_indices(original, **kwargs)
            assert result.dtype == np.int64 and result.flags.c_contiguous
        np.testing.assert_array_equal(original, before)
