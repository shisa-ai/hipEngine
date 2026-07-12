from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100 import GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS
from scripts.gguf_true_ar_category_bench import run_prompt_true_ar


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class _Graph:
    def __init__(self, session: "_Session") -> None:
        self.session = session
        self.replays = 0
        self.closed = False

    def replay(self, steps: int) -> None:
        assert steps == 1
        self.replays += 1
        self.session.position += 1

    def read_sample(self, *, return_logits: bool):
        return SimpleNamespace(
            token_id=1000 + self.session.position,
            logits=np.asarray([0.0, 1.0], dtype=np.float32) if return_logits else None,
        )

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, *, minimum: int | None) -> None:
        self.minimum = minimum
        self.position = 0
        self.step_calls = 0
        self.capture_kwargs = None
        self.graph: _Graph | None = None

    def reset(self) -> None:
        self.position = 0

    def prefill(self, prompt_tokens, **kwargs):
        self.position = len(prompt_tokens)
        return SimpleNamespace(token_id=7, logits=None)

    def step(self, token_id: int, *, return_logits: bool):
        self.step_calls += 1
        self.position += 1
        return SimpleNamespace(
            token_id=int(token_id) + 1,
            logits=np.asarray([0.0, 1.0], dtype=np.float32) if return_logits else None,
        )

    def decode_graph_min_replay_steps(self) -> int | None:
        return self.minimum

    def capture_decode_graph(self, **kwargs):
        self.capture_kwargs = kwargs
        self.graph = _Graph(self)
        return self.graph


def test_gfx1100_admits_measured_24_transition_decode_graph() -> None:
    assert GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS == 24
    assert backend_package_capability(
        "hip_gfx1100", "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS"
    ) == 24


def test_true_ar_uses_state_bound_graph_when_horizon_is_admitted() -> None:
    session = _Session(minimum=24)

    row = run_prompt_true_ar(
        session=session,
        tokenizer=_Tokenizer(),
        prompt_row={"id": "p", "category": "code", "prompt": "x"},
        decode_tokens=24,
        warmup_decode_tokens=0,
        use_bulk_prefill=True,
        bulk_attention_mode="bulk",
        graph_replay_decode=True,
        graph_steps_per_replay=1,
    )

    assert row["graph_replay_decode"] is True
    assert row["graph_steps_per_replay"] == 1
    assert row["graph_capture_ms_included"] >= 0.0
    assert row["graph_capture_ms_excluded"] == 0.0
    assert row["finite_final_logits"] is True
    assert session.step_calls == 0
    assert session.capture_kwargs == {
        "position": session.capture_kwargs["position"],
        "steps_per_replay": 1,
        "max_replay_steps": 24,
        "attention_max_context_len": session.capture_kwargs["position"] + 24,
    }
    assert session.graph is not None
    assert session.graph.replays == 24
    assert session.graph.closed is True


def test_true_ar_falls_back_to_eager_below_admitted_horizon() -> None:
    session = _Session(minimum=24)

    row = run_prompt_true_ar(
        session=session,
        tokenizer=_Tokenizer(),
        prompt_row={"id": "p", "category": "code", "prompt": "x"},
        decode_tokens=23,
        warmup_decode_tokens=0,
        use_bulk_prefill=True,
        bulk_attention_mode="bulk",
        graph_replay_decode=True,
        graph_steps_per_replay=1,
    )

    assert row["graph_replay_decode"] is False
    assert row["graph_replay_min_steps"] == 24
    assert session.step_calls == 23
    assert session.graph is None
