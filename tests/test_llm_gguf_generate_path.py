from __future__ import annotations

from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.models import resolve_model

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
MOE_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_model_plugin_resolves_architecture() -> None:
    plugin = resolve_model("qwen35")

    assert plugin.name == "qwen3_5_gguf"
    assert plugin.default_quant == "gguf_q4_k_m"


def test_qwen35moe_gguf_model_plugin_resolves_architecture() -> None:
    plugin = resolve_model("qwen35moe")

    assert plugin.name == "qwen3_5_moe_gguf"
    assert plugin.default_quant == "gguf_q4_k_m"


def test_llm_generate_gguf_path_uses_resident_session(monkeypatch) -> None:
    import hipengine.generation.qwen35_gguf as qwen35_gguf

    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_exit", exc_type is None))

        def replay(self, steps):
            calls.append(("graph_replay", int(steps)))

        def read_generated_token_ids(self, count):
            calls.append(("graph_read", int(count)))
            return [16]

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(int(token) for token in token_ids), bool(return_logits)))
            return type("Result", (), {"token_id": 220, "logit": 4.5})()

        def capture_decode_graph(self, *, position, steps_per_replay, max_replay_steps, record_steps):
            calls.append(("capture_decode_graph", position, steps_per_replay, max_replay_steps, record_steps))
            return FakeGraph()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    llm = LLM(str(MODEL), backend="hip_gfx1100", quant="gguf_q4_k_m")
    assert llm.generate("The answer is", SamplingParams(max_tokens=2)) == [" 1"]

    assert calls == [
        ("init", str(MODEL.resolve())),
        ("enter",),
        ("prefill", (760, 4087, 369), False),
        ("capture_decode_graph", 3, 1, 1, 1),
        ("graph_enter",),
        ("graph_replay", 1),
        ("graph_read", 1),
        ("graph_exit", True),
        ("exit", True),
    ]


def test_llm_generate_qwen35moe_gguf_path_uses_resident_session(monkeypatch) -> None:
    if not MOE_MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MOE_MODEL}")
    import hipengine.generation.qwen35_gguf as qwen35_gguf

    calls = []

    class FakeSession:
        def __init__(self, model_path):
            calls.append(("init", str(model_path)))

        def __enter__(self):
            calls.append(("enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("exit", exc_type is None))

        def prefill(self, token_ids, *, return_logits=True):
            calls.append(("prefill", tuple(int(token) for token in token_ids), bool(return_logits)))
            return type("Result", (), {"token_id": 220, "logit": 4.5})()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFResidentSession", FakeSession)

    llm = LLM(str(MOE_MODEL), backend="hip_gfx1100", quant="gguf_q4_k_m")
    assert llm.generate("The answer is", SamplingParams(max_tokens=1)) == [" "]

    assert calls == [
        ("init", str(MOE_MODEL.resolve())),
        ("enter",),
        ("prefill", (760, 4087, 369), False),
        ("exit", True),
    ]
