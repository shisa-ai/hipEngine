"""Retile parent diagnostics restore their selector after failure."""

import pytest

from hipengine.runtime import gguf_linear
from scripts.qwen38_gfx1151_prefill_compare import candidate_scope


def test_retile_parent_scope_restores_nested_failure():
    original = gguf_linear._q4_t16_dual_silu_retile_enabled
    assert original()
    with pytest.raises(RuntimeError, match="comparison failed"):
        with candidate_scope("no_retiles"):
            assert not gguf_linear._q4_t16_dual_silu_retile_enabled()
            with candidate_scope("no_retiles"):
                assert not gguf_linear._q4_t16_dual_silu_retile_enabled()
            assert not gguf_linear._q4_t16_dual_silu_retile_enabled()
            raise RuntimeError("comparison failed")
    assert gguf_linear._q4_t16_dual_silu_retile_enabled is original
