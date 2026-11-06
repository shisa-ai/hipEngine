"""Surya unsupported fused geometry retains the span-aware unfused chain."""
from types import SimpleNamespace


def test_runner_fallback_passes_live_span_owner(monkeypatch):
    from hipengine.runtime import surya as module

    runner = object.__new__(module.SuryaGpuRunner)
    runner.spec = SimpleNamespace(num_attention_heads=6, num_key_value_heads=2, head_dim=32)
    runner.max_seq = 512
    runner._seq_len = 18
    runner._span_plan = SimpleNamespace(block_size=256)
    runner.runtime = object()
    runner.surya_library = object()
    spans = object()
    monkeypatch.setattr(runner, "_spans", lambda: spans)
    monkeypatch.setattr(runner, "_buf", lambda name, nbytes: SimpleNamespace(ptr=99))
    calls = []
    monkeypatch.setattr(module, "surya_attention_decode_unfused_f32_spans",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    runner._attention_decode_spans(1, 2, 3, 4)
    assert calls[0][0] == (1, 2, 3, 4, 99, spans, 256, 6, 2, 32)
    assert calls[0][1]["runtime"] is runner.runtime
