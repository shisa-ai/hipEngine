"""Host retention budgets distinguish dense state from DMS cold prompt planes."""
import pytest

from hipengine.generation.qwen35_gguf import _gguf_prefix_retained_state_bytes_limit


@pytest.mark.parametrize("dms, expected", [(False, 1 << 30), (True, 4 << 30)])
def test_default_cache_budget(monkeypatch, dms, expected):
    monkeypatch.delenv("HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES", raising=False)
    assert _gguf_prefix_retained_state_bytes_limit(dms=dms) == expected


@pytest.mark.parametrize("dms", [False, True])
@pytest.mark.parametrize("budget", [1 << 20, 768 << 20, 2 << 30, 6 << 30])
def test_explicit_cache_budget_overrides_either_default(monkeypatch, dms, budget):
    monkeypatch.setenv("HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES", str(budget))
    assert _gguf_prefix_retained_state_bytes_limit(dms=dms) == budget


def test_dense_callers_keep_existing_default(monkeypatch):
    monkeypatch.delenv("HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES", raising=False)
    assert _gguf_prefix_retained_state_bytes_limit() == 1 << 30
