"""Scoped cumulative-semantic comparisons must restore on nested failure."""
import pytest

from hipengine.runtime import gguf_linear as linear


def test_cumulative_diagnostic_restores_nested_fields_on_exception():
    with linear.cumulative_prefill_diagnostic(dense_bulk=False, q8_pair=False):
        outer = linear._cumulative_prefill_override.get()
        assert outer == (False, None, False)
        with pytest.raises(RuntimeError):
            with linear.cumulative_prefill_diagnostic(dense_residual=False):
                assert linear._cumulative_prefill_override.get() == (False, False, False)
                raise RuntimeError('comparison failed')
        assert linear._cumulative_prefill_override.get() == outer
    assert linear._cumulative_prefill_override.get() == (None, None, None)


@pytest.mark.parametrize('index,name', [
    (0, 'HIPENGINE_GGUF_DENSE_WMMA_BULK'),
    (1, 'HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL'),
    (2, 'HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL'),
])
def test_cumulative_diagnostic_ignores_retired_environment(monkeypatch, index, name):
    monkeypatch.setenv(name, '0')
    assert linear._cumulative_prefill_enabled(index)
    fields = ('dense_bulk', 'dense_residual', 'q8_pair')
    with linear.cumulative_prefill_diagnostic(**{fields[index]: False}):
        assert not linear._cumulative_prefill_enabled(index)
    assert linear._cumulative_prefill_enabled(index)
