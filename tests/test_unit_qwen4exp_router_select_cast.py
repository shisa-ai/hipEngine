"""Routing uses the registered composite; retired environment values are ignored."""
from types import SimpleNamespace
import pytest
import hipengine.runtime.qwen4_exp_runner as runner


@pytest.mark.parametrize("rows", [1, 3, 4, 17])
@pytest.mark.parametrize("setting", [None, "1", "0", "false"])
def test_router_select_cast_resolves_composite(monkeypatch, rows, setting):
    name = "HIPENGINE_QWEN4_EXP_ROUTER_SELECT_CAST"
    if setting is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, setting)
    runtime = object()
    scratch = SimpleNamespace(closed=False, runtime=runtime,
        router_logits=SimpleNamespace(ptr=10), selected=SimpleNamespace(ptr=20),
        routing=SimpleNamespace(ptr=30), hidden_bf16=SimpleNamespace(ptr=40))
    weight = SimpleNamespace(backend="test")
    weights = dict.fromkeys(("router", "expert_gate", "expert_up", "expert_down",
        "shared_gate", "shared_up", "shared_down", "shared_gate_weight"), weight)
    calls = []

    class Selected(Exception):
        pass

    monkeypatch.setattr(runner, "_ensure_qwen4_exp_moe_backend", lambda backend: None)
    monkeypatch.setattr(runner, "_qwen4_exp_router_producer_variant", lambda rows: None)
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **k: None)

    def combined(*args, **kwargs):
        calls.append((args, kwargs))
        raise Selected

    def resolve(**kwargs):
        assert kwargs == dict(backend="test", layer="router_select+cast", quant="fp32")
        return combined

    monkeypatch.setattr(runner, "resolve", resolve)
    with pytest.raises(Selected):
        runner.run_qwen4_exp_moe(50, weights, scratch=scratch, rows=rows,
            hidden=2560, ffn=640, experts=512, top_k=10, stream=7, runtime=runtime)
    assert calls == [((10, 20, 30, 50, 40, rows, 512, 512, 10, 2560),
                     {"stream": 7, "runtime": runtime})]
