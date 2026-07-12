from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import qwen35_paro_int8_kv_quality_sweep as sweep


def _run_payload(token_ids: list[int], logits: list[list[float]]) -> dict[str, object]:
    return {
        "generated_token_ids": token_ids,
        "logits": [np.asarray(row, dtype=np.float32) for row in logits],
    }


def test_compare_logits_reports_reference_token_fidelity() -> None:
    reference = [
        np.asarray([0.0, 4.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 5.0], dtype=np.float32),
    ]
    candidate = [
        np.asarray([0.0, 3.9, 1.0], dtype=np.float32),
        np.asarray([6.0, 1.0, 4.8], dtype=np.float32),
    ]

    comparison = sweep._compare_logits(reference, candidate)

    assert comparison["position_labels"] == ["prefill_seed", "decode_0"]
    assert comparison["reference_top1"] == [1, 2]
    assert comparison["candidate_top1"] == [1, 0]
    assert comparison["candidate_reference_top1_rank"] == [1, 2]
    assert comparison["top1_agreement"] == 0.5
    assert comparison["first_top1_mismatch"] == {"index": 1, "reference": 2, "candidate": 0}
    assert len(comparison["candidate_reference_top1_logprob_delta"]) == 2
    assert comparison["candidate_reference_top1_logprob_delta"][0] < 0.0


def test_free_running_diagnostics_separates_rollout_cascade() -> None:
    reference = _run_payload(
        [10, 11, 12],
        [[4.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 0.0]],
    )
    candidate = _run_payload(
        [10, 99, 13],
        [[4.0, 0.0], [3.9, 0.0], [3.8, 0.0], [0.0, 4.0]],
    )

    diagnostic = sweep._free_running_diagnostics(reference, candidate)

    assert diagnostic["generated_first_mismatch"] == {"index": 1, "left": 11, "right": 99}
    assert diagnostic["first_context_divergent_logit_position"] == 3
    assert diagnostic["matched_history_logit_positions"] == 3
    assert diagnostic["matched_history_logit_gate"]["positions"] == 3
    assert diagnostic["matched_history_logit_gate"]["top1_agreement"] == 1.0
    assert diagnostic["full_rollout_logit_gate"]["top1_agreement"] == 0.75


def test_run_case_uses_reference_teacher_forced_inputs(monkeypatch) -> None:
    class FakeResult:
        def __init__(self, token_id: int) -> None:
            self.token_id = token_id
            self.token_text = str(token_id)
            self.logit = float(token_id)

    class FakeSession:
        vocab_size = 3
        runtime = object()
        prefill_config = SimpleNamespace(
            linear_chunk_size=1,
            moe_chunk_size=2,
            full_attn_query_chunk_size=3,
            full_attn_post_chunk_size=4,
            full_attn_rope_chunk_size=5,
        )
        prefill_chunk_tuning = {"reason": "test"}

        def __init__(self) -> None:
            self.inputs: list[tuple[int, int]] = []

        def reset(self) -> None:
            self.inputs.clear()

        def _resolve_prefill_config_for_length(self, prompt_length: int) -> None:
            assert prompt_length == 2

        def prefill_native(self, prompt_tokens, *, sample: bool):
            assert prompt_tokens == [1, 2]
            assert sample is True
            return FakeResult(5)

        def step(self, token_id: int, *, position: int, sample: bool):
            assert sample is True
            self.inputs.append((token_id, position))
            return FakeResult(token_id + 100)

        def owned_buffer_summary(self):
            return {}

    session = FakeSession()
    monkeypatch.setattr(sweep, "_prompt_tokens", lambda *_args: [1, 2])
    monkeypatch.setattr(
        sweep,
        "_read_logits",
        lambda _session: np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
    )

    result = sweep._run_case(
        session=session,
        model=Path("/unused"),
        prompt="unused",
        token_id=1,
        prompt_length=2,
        decode_steps=2,
        forced_decode_input_ids=[7, 8],
    )

    assert result["decode_input_mode"] == "reference_teacher_forced"
    assert result["decode_input_token_ids"] == [7, 8]
    assert result["generated_token_ids"] == [107, 108]
    assert session.inputs == [(7, 2), (8, 3)]


def test_run_case_rejects_incomplete_forced_input_sequence(monkeypatch) -> None:
    class FakeSession:
        prefill_config = SimpleNamespace(
            linear_chunk_size=1,
            moe_chunk_size=1,
            full_attn_query_chunk_size=1,
            full_attn_post_chunk_size=1,
            full_attn_rope_chunk_size=1,
        )
        prefill_chunk_tuning = None
        vocab_size = 2
        runtime = object()

        def reset(self) -> None:
            pass

        def _resolve_prefill_config_for_length(self, _prompt_length: int) -> None:
            pass

        def prefill_native(self, _prompt_tokens, *, sample: bool):
            assert sample
            return SimpleNamespace(token_id=1, token_text="1", logit=1.0)

        def owned_buffer_summary(self):
            return {}

    monkeypatch.setattr(sweep, "_prompt_tokens", lambda *_args: [1])
    monkeypatch.setattr(sweep, "_read_logits", lambda _session: np.asarray([0.0, 1.0], dtype=np.float32))

    try:
        sweep._run_case(
            session=FakeSession(),
            model=Path("/unused"),
            prompt="unused",
            token_id=1,
            prompt_length=1,
            decode_steps=2,
            forced_decode_input_ids=[1],
        )
    except ValueError as exc:
        assert "exactly decode_steps" in str(exc)
    else:
        raise AssertionError("expected incomplete forced input sequence to fail")
