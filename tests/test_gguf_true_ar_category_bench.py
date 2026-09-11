from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100 import GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS
from hipengine.kernels.policy import (
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)
from hipengine.core.dtype import DType
from scripts.gguf_true_ar_category_bench import (
    resolve_kv_selection,
    run_prompt_true_ar,
)


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
    policies = backend_package_capability(
        "hip_gfx1100", "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES"
    )
    assert policies == {
        (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M"): {
            "transport": "pm4",
            "min_replay_steps_by_physical_rows": {1: 160, 2: 64, 4: 96, 8: 80},
        },
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            "transport": "pm4",
            "min_replay_steps_by_physical_rows": {1: 128},
        },
    }
    resolve = gguf_runner._resolve_gguf_decode_graph_submission_transport
    common = {
        "geometry": QWEN35_MOE_H2048_E256_GEOMETRY,
        "file_type_name": "MOSTLY_Q4_K_M",
    }
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=1,
        replay_steps=159,
        env={},
    ) == "hipgraph"
    assert resolve(
        "hip_gfx1100",
        **common,
        physical_rows=1,
        replay_steps=160,
        env={},
    ) == "pm4"
    dense = {
        "geometry": QWEN35_DENSE_H5120_GEOMETRY,
        "file_type_name": "MOSTLY_Q4_K_M",
    }
    assert resolve(
        "hip_gfx1100",
        **dense,
        physical_rows=1,
        replay_steps=127,
        env={},
    ) == "hipgraph"
    assert resolve(
        "hip_gfx1100",
        **dense,
        physical_rows=1,
        replay_steps=128,
        env={},
    ) == "pm4"


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


def _sequence_sha(tokens) -> str:
    return hashlib.sha256(
        ",".join(str(int(token)) for token in tokens).encode("ascii")
    ).hexdigest()


def test_true_ar_records_a_full_generated_sequence_hash() -> None:
    """The graph/eager identity check must cover every generated token."""

    session = _Session(minimum=24)
    row = run_prompt_true_ar(
        session=session,
        tokenizer=_Tokenizer(),
        prompt_row={"id": "p", "category": "code", "prompt": "x"},
        decode_tokens=3,
        warmup_decode_tokens=0,
        use_bulk_prefill=True,
        bulk_attention_mode="bulk",
        graph_replay_decode=True,
        graph_steps_per_replay=1,
    )

    generated = row["generated_token_ids"]
    # The prefill token plus one entry per decode step.
    assert len(generated) == 4
    assert row["generated_sha256"] == _sequence_sha(generated)
    # A sequence differing only in its LAST token must hash differently, so the
    # digest covers the whole sequence rather than a prefix or a preview.
    assert row["generated_sha256"] != _sequence_sha(generated[:-1] + [generated[-1] + 1])


def test_true_ar_graph_and_eager_paths_share_the_hash_schema() -> None:
    """Both decode paths must record the same full-sequence fields."""

    def run(minimum: int | None, decode_tokens: int) -> dict:
        return run_prompt_true_ar(
            session=_Session(minimum=minimum),
            tokenizer=_Tokenizer(),
            prompt_row={"id": "p", "category": "code", "prompt": "x"},
            decode_tokens=decode_tokens,
            warmup_decode_tokens=0,
            use_bulk_prefill=True,
            bulk_attention_mode="bulk",
            graph_replay_decode=True,
            graph_steps_per_replay=1,
        )

    eager = run(minimum=None, decode_tokens=4)
    graph = run(minimum=4, decode_tokens=4)
    assert eager["graph_replay_decode"] is False
    assert graph["graph_replay_decode"] is True
    for row in (eager, graph):
        assert len(row["generated_token_ids"]) == 5
        assert row["generated_sha256"] == _sequence_sha(row["generated_token_ids"])


def test_true_ar_kv_selection_defaults_to_bf16() -> None:
    """The harness must not silently change the KV layout it measures."""

    assert resolve_kv_selection("bf16") == (None, None, None)

    policy, scale_dtype, granularity = resolve_kv_selection("int8_per_token_head")
    assert policy.storage_dtype == DType.INT8_PER_TOKEN_HEAD
    assert scale_dtype == "fp32"
    assert granularity == "per_token_head"
