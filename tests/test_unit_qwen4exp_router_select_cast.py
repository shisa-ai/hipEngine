"""Default expert routing fuses the exact activation cast; opt-out is strict."""
from types import SimpleNamespace
import pytest
import hipengine.runtime.qwen4_exp_runner as runner


@pytest.mark.parametrize('rows', [1, 3, 4, 17])
@pytest.mark.parametrize('setting,fused', [(None, True), ('1', True), ('0', False), ('false', False)])
def test_router_select_cast_default_and_rollback(monkeypatch, rows, setting, fused):
    name = 'HIPENGINE_QWEN4_EXP_ROUTER_SELECT_CAST'
    if setting is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, setting)
    runtime = object()
    scratch = SimpleNamespace(closed=False, runtime=runtime,
        router_logits=SimpleNamespace(ptr=10), selected=SimpleNamespace(ptr=20),
        routing=SimpleNamespace(ptr=30), hidden_bf16=SimpleNamespace(ptr=40))
    weight = SimpleNamespace(backend='test')
    weights = dict.fromkeys(('router', 'expert_gate', 'expert_up', 'expert_down',
        'shared_gate', 'shared_up', 'shared_down', 'shared_gate_weight'), weight)
    calls = []
    class Selected(Exception):
        pass
    monkeypatch.setattr(runner, '_ensure_qwen4_exp_moe_backend', lambda backend: None)
    monkeypatch.setattr(runner, '_qwen4_exp_router_producer_variant', lambda rows: None)
    monkeypatch.setattr(runner, 'launch_gguf_linear', lambda *a, **k: None)
    def select(*args, **kwargs):
        calls.append(('select', args, kwargs))
    def cast(*args, **kwargs):
        calls.append(('cast', args, kwargs))
        raise Selected
    def combined(*args, **kwargs):
        calls.append(('fused', args, kwargs))
        raise Selected
    monkeypatch.setattr(runner, 'qwen35_router_select', select)
    monkeypatch.setattr(runner, 'f32_to_bf16', cast)
    monkeypatch.setattr(runner, 'qwen35_router_select_cast_bf16', combined)
    with pytest.raises(Selected):
        runner.run_qwen4_exp_moe(50, weights, scratch=scratch, rows=rows,
            hidden=2560, ffn=640, experts=512, top_k=10, stream=7, runtime=runtime)
    assert [c[0] for c in calls] == (['fused'] if fused else ['select', 'cast'])
    if fused:
        assert calls[0][1] == (10, 20, 30, 50, 40, rows, 512, 512, 10, 2560)
    else:
        assert calls[0][1] == (10, 20, 30, rows, 512, 512, 10)
        assert calls[1][1] == (50, 40, rows * 2560)
    assert all(c[2] == {'stream': 7, 'runtime': runtime} for c in calls)
