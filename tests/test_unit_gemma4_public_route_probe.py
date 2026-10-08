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



@pytest.mark.parametrize('name,variant', [
    ('gemma4_attention_prefill_tiled', 'tiled'),
    ('gemma4_attention_prefill_aotriton', 'aotriton'),
    ('gemma4_attention_prefill_bf16', 'exact'),
])
def test_observe_native_launches_without_variant_selection(monkeypatch, name, variant):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    def launch(*args, **kwargs):
        return 'native launch'

    monkeypatch.setattr(layer, name, launch)
    with observe_attention_launches() as records:
        assert getattr(layer, name)(tokens=3, keys=19, head_dim=256,
                                    num_heads=16, num_kv_heads=2) == 'native launch'
    assert len(records) == 1
    assert records[0]['variant'] == variant
    assert records[0]['tokens'] == 3
    assert getattr(layer, name) is launch


def test_native_failure_is_not_counted_and_wrapper_is_restored(monkeypatch):
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    def fail(**kwargs):
        raise RuntimeError('native launch failed')

    monkeypatch.setattr(layer, 'gemma4_attention_prefill_bf16', fail)
    with pytest.raises(RuntimeError, match='native launch failed'):
        with observe_attention_launches() as records:
            layer.gemma4_attention_prefill_bf16(tokens=1)
    assert records == []
    assert layer.gemma4_attention_prefill_bf16 is fail


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


def test_observe_decode_launches_report_the_registered_variant(monkeypatch):
    """Decode reaches the device through _launch_prefill, not the layer functions.

    Without this hook the probe reports prefill launches only and looks like it
    has proved that a decode scope was never reached.
    """
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    def launch(*args, **kwargs):
        return "launched"
    monkeypatch.setattr(attention, "_launch_prefill", launch)
    monkeypatch.setattr(attention, "decode_selection", lambda library=None: 3)
    with observe_attention_launches() as records:
        wrapped = attention._launch_prefill
        assert wrapped(tokens=1, keys=16384, head_dim=512, num_heads=16,
                       num_kv_heads=2) == "launched"
        assert wrapped(tokens=512, keys=16384, head_dim=512, num_heads=16,
                       num_kv_heads=2) == "launched"
    assert records == [{"variant": attention.DECODE_ATTENTION_FLASH,
                        "reason": "launcher selection counter",
                        "launcher_selection_sticky": True, "tokens": 1,
                        "keys": 16384, "head_dim": 512, "num_heads": 16,
                        "num_kv_heads": 2}]
    assert attention._launch_prefill is launch
    assert summarize_attention_launches(records) == [
        {"variant": attention.DECODE_ATTENTION_FLASH, "query_width": "singleton",
         "invocations": 1},
    ]


@pytest.mark.parametrize("selection,constant", [
    (1, "DECODE_ATTENTION_PLAIN"),
    (2, "DECODE_ATTENTION_STAGED"),
    (3, "DECODE_ATTENTION_FLASH"),
])
def test_observe_decode_launches_name_every_selection(monkeypatch, selection, constant):
    """The reported name is the registered variant, not the raw selection id."""
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    monkeypatch.setattr(attention, "_launch_prefill", lambda *a, **k: None)
    monkeypatch.setattr(attention, "decode_selection", lambda library=None: selection)
    with observe_attention_launches() as records:
        attention._launch_prefill(tokens=1, keys=2048, head_dim=512)
    assert records[0]["variant"] == getattr(attention, constant)


def test_observe_records_the_scoped_route_the_layer_resolved(monkeypatch):
    """The resolver's route is the per-call answer; the counter is only a hint."""
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer

    def resolver(*args, **kwargs):
        return "launcher", "gemma4_flash"
    monkeypatch.setattr(layer, "_select_exact_decode_attention", resolver)
    with observe_attention_launches() as records:
        assert layer._select_exact_decode_attention(tokens=1, keys=16384, head_dim=512,
                                                    num_heads=16, num_kv_heads=2,
                                                    sliding_window=None) == ("launcher", "gemma4_flash")
        layer._select_exact_decode_attention(tokens=512, keys=16384, head_dim=512,
                                             num_heads=16, num_kv_heads=2,
                                             sliding_window=None)
    assert records == [{"variant": "gemma4_flash", "reason": "scoped decode resolution",
                        "tokens": 1, "keys": 16384, "head_dim": 512, "num_heads": 16,
                        "num_kv_heads": 2, "sliding_window": None}]
    assert layer._select_exact_decode_attention is resolver


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
