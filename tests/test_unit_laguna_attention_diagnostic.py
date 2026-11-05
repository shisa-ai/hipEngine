"""Laguna comparison controls belong to one resident owner."""
import pytest

from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


def test_attention_comparison_scope_restores_after_nested_failure():
    owner = LagunaGGUFResidentSession.__new__(LagunaGGUFResidentSession)
    owner.prefill_swa_attention_hipblaslt = True
    owner.prefill_dense_contiguous_cache = False
    with owner.prefill_attention_diagnostic(swa=False):
        assert owner.prefill_swa_attention_hipblaslt is False
        with pytest.raises(RuntimeError):
            with owner.prefill_attention_diagnostic(dense_contiguous=True):
                assert owner.prefill_swa_attention_hipblaslt is False
                assert owner.prefill_dense_contiguous_cache is True
                raise RuntimeError('failed')
        assert owner.prefill_dense_contiguous_cache is False
    assert owner.prefill_swa_attention_hipblaslt is True
    assert owner.prefill_dense_contiguous_cache is False
