"""Shared physical NextN transition preserves launch order and caller options."""

from types import SimpleNamespace

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_nextn as module


def test_shared_batch_transition_preserves_order_and_options(monkeypatch):
    executor = object.__new__(module.Qwen35GGUFNextNExecutor)
    calls = []
    executor.hidden_size = 4
    executor.vocab_size = 16
    executor.runtime = SimpleNamespace(memcpy_async=lambda *args: calls.append(("copy", args)))
    weight = SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=90)))
    executor.weights = SimpleNamespace(
        fallback=lambda name: name,
        nextn=lambda name: weight if name in {"enorm", "hnorm"} else name,
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    for index, name in enumerate(("_token_buf", "_embedding_buf", "_enorm_buf", "_hnorm_buf",
                                  "_fusion_buf", "_fused_buf", "_final_hidden_buf", "_logits_buf")):
        setattr(executor, name, SimpleNamespace(ptr=1000 + index * 100))
    monkeypatch.setattr(module, "launch_gguf_embedding", lambda *args, **kw: calls.append(("embed", kw)))
    monkeypatch.setattr(module, "gguf_rmsnorm_bf16_f32_weight", lambda *args, **kw: calls.append(("norm", args)))
    monkeypatch.setattr(module, "launch_gguf_linear", lambda *args, **kw: calls.append(("fusion", kw)))
    executor._batch_session = SimpleNamespace(
        step_hidden_batch_native=lambda *args, **kw: calls.append(("backbone", kw)))
    sessions = (object(), object())
    stride = executor._enqueue_batch_hidden_transition(
        SimpleNamespace(ptr=42), sessions, (3, 7), score_output=False,
        kv_write_only=True, synchronize=False,
    )
    assert stride == executor.hidden_size * DType.BF16.itemsize
    assert [name for name, _ in calls] == ["embed", "norm", "norm", "copy", "copy", "copy", "copy", "fusion", "backbone"]
    assert [args[:3] for name, args in calls if name == "copy"] == [
        (1400, 1200, stride), (1400 + stride, 1300, stride),
        (1400 + 2 * stride, 1200 + stride, stride),
        (1400 + 3 * stride, 1300 + stride, stride),
    ]
    options = calls[-1][1]
    assert options["sessions"] is sessions
    assert options["positions"] == (3, 7)
    assert options["score_output"] is False
    assert options["kv_write_only"] is True
    assert options["synchronize"] is False
