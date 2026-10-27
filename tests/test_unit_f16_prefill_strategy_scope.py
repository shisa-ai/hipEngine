from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from hipengine.runtime import f16_weight_linear as projection


def _selected(rows=32, *, eligible=True, dtype="bf16", backend="hip_gfx1151"):
    return projection._prefill_strategy(
        rows=rows, activation_dtype=dtype, backend=backend,
        compensated_wmma_eligible=eligible,
    )


def test_prefill_strategy_scope_restores_nested_and_failed_comparisons(monkeypatch):
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "gemv")
    assert projection.current_f16_prefill_strategy() == "auto"
    automatic = _selected()
    with projection.f16_prefill_strategy("gemv"):
        assert _selected() is None
        with pytest.raises(RuntimeError, match="comparison failed"):
            with projection.f16_prefill_strategy("tiled"):
                assert _selected() == "tiled"
                raise RuntimeError("comparison failed")
        assert projection.current_f16_prefill_strategy() == "gemv"
    assert _selected() == automatic
    assert projection.current_f16_prefill_strategy() == "auto"


def test_prefill_strategy_contexts_and_threads_do_not_leak():
    context = copy_context()
    with projection.f16_prefill_strategy("gemv"):
        assert context.run(projection.current_f16_prefill_strategy) == "auto"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(projection.current_f16_prefill_strategy).result() == "auto"
        assert projection.current_f16_prefill_strategy() == "gemv"


def test_compensated_scope_preserves_shape_and_activation_fallbacks():
    with projection.f16_prefill_strategy("wmma_comp_swa"):
        for rows in (1, 3, 15, 16, 17, 53):
            expected = None if rows == 1 else "tiled" if rows < 16 else "wmma_comp"
            assert _selected(rows) == expected
            assert _selected(rows, eligible=False) == (None if rows == 1 else "tiled")
            assert _selected(rows, dtype="f32") is None


@pytest.mark.parametrize("mode", ["wmma", "invalid"])
def test_prefill_strategy_rejects_unknown_modes_without_changing_owner(mode):
    with pytest.raises(ValueError, match="F16 prefill strategy"):
        with projection.f16_prefill_strategy(mode):
            pytest.fail("invalid strategy entered")
    assert projection.current_f16_prefill_strategy() == "auto"
