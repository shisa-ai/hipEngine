"""CPU tests for invocation-level Gemma attention route observation."""
from dataclasses import dataclass

import pytest

from scripts.gemma4_public_route_probe import observe_attention_launches, summarize_attention_launches


def test_single_row_prefill_tail_is_not_claimed_as_decode():
    records = [{"variant": "gemma4_staged", "tokens": width} for width in (512, 1, 1)]
    assert summarize_attention_launches(records) == [
        {"variant": "gemma4_staged", "query_width": "multi_token", "invocations": 1},
        {"variant": "gemma4_staged", "query_width": "singleton", "invocations": 2},
    ]



@dataclass(frozen=True)
class Selection:
    variant: str
    launcher: object
    reason: str = "capability match"


def test_observe_counts_calls_not_only_selection(monkeypatch):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    calls = []
    def launch(*args, **kwargs):
        calls.append(kwargs)
        return "launched"
    monkeypatch.setattr(layer, "select_prefill_attention", lambda **kwargs: Selection("wmma_full", launch))
    with observe_attention_launches() as records:
        selection = layer.select_prefill_attention(tokens=16, keys=777, head_dim=512,
                                                  num_heads=16, num_kv_heads=2)
        assert records == []
        assert selection.launcher(1, tokens=16, keys=777) == "launched"
    assert len(calls) == len(records) == 1
    assert records[0] == {"variant": "wmma_full", "tokens": 16, "keys": 777,
                          "head_dim": 512, "num_heads": 16, "num_kv_heads": 2,
                          "reason": "capability match"}


def test_observer_restores_selector_after_failure(monkeypatch):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    def fail(*args, **kwargs):
        raise RuntimeError("launch failed")
    original = lambda **kwargs: Selection("strict", fail)
    monkeypatch.setattr(layer, "select_prefill_attention", original)
    with pytest.raises(RuntimeError, match="launch failed"):
        with observe_attention_launches() as records:
            selection = layer.select_prefill_attention(tokens=1, keys=7, head_dim=512,
                                                      num_heads=16, num_kv_heads=2)
            selection.launcher()
    assert records == []
    assert layer.select_prefill_attention is original
