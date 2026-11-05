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


def test_cumulative_diagnostic_overrides_ambient_rollback(monkeypatch):
    name = 'HIPENGINE_GGUF_DENSE_WMMA_BULK'
    monkeypatch.setenv(name, '0')
    assert not linear._cumulative_prefill_enabled(0, name)
    with linear.cumulative_prefill_diagnostic(dense_bulk=True):
        assert linear._cumulative_prefill_enabled(0, name)
    assert not linear._cumulative_prefill_enabled(0, name)
