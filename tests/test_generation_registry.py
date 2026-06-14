from __future__ import annotations

import subprocess
import sys

from hipengine.generation import DecodePhase, DecodeState, FinishDetails, GenerationOutput


def test_decode_state_stream_snapshot_normalizes_json_payload() -> None:
    state = DecodeState.from_stream_tokens(
        phase=DecodePhase.THINK,
        tokens={
            "prompt_tokens": 7,
            "completion_tokens": 5,
            "streamed_tokens": 3,
            "reasoning_tokens": 3,
        },
    )

    assert state.to_json_dict() == {
        "row_index": 0,
        "step_index": 3,
        "prompt_tokens": 7,
        "generated_tokens": 5,
        "phase": "think",
        "continuation_eligible": False,
        "reasoning_tokens": 3,
    }


def test_generation_output_accepts_telemetry_mapping() -> None:
    output = GenerationOutput(
        text="answer",
        telemetry={
            "event": "done",
            "decode_state": {
                "phase": "answer",
                "prompt_tokens": 4,
                "generated_tokens": 2,
                "answer_tokens": 2,
                "sampler_mode": "greedy_fast",
                "active_processors": "logit_bias",
                "sampler_fast_path_blockers": ["logit_bias"],
            },
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    )

    assert output.telemetry is not None
    assert output.telemetry.to_json_dict() == {
        "decode_state": {
            "row_index": 0,
            "step_index": 0,
            "prompt_tokens": 4,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 2,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_mode": "greedy_fast",
        },
        "event": "done",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }


def test_generation_output_accepts_finish_details_mapping() -> None:
    output = GenerationOutput(
        text="answer",
        finish_details={
            "reason": "eos",
            "eos_token_id": "151645",
            "stop_sequence": ["42", "43"],
            "length_limit": "7",
            "deadline_exceeded": True,
            "forced_close": True,
            "synthetic_tokens": 2,
            "reasoning_tokens": 3,
            "answer_tokens": 4,
            "tool_call_tokens": 5,
            "structured_tokens": 6,
            "budget_pressure": "hard_close",
            "cache_action": "append_prompt_only",
            "sampler_mode": "processed_argmax",
            "phase": "answer",
            "continuation_eligible": False,
        },
    )

    assert output.finish_details == FinishDetails(
        reason="eos",
        eos_token_id=151645,
        stop_sequence=(42, 43),
        length_limit=7,
        deadline_exceeded=True,
        forced_close=True,
        synthetic_tokens=2,
        reasoning_tokens=3,
        answer_tokens=4,
        tool_call_tokens=5,
        structured_tokens=6,
        budget_pressure="hard_close",
        cache_action="append_prompt_only",
        sampler_mode="processed_argmax",
        phase="answer",
        continuation_eligible=False,
    )
    assert output.finish_details.to_json_dict(reason="stop") == {
        "reason": "stop",
        "eos_token_id": 151645,
        "stop_sequence": [42, 43],
        "length_limit": 7,
        "deadline_exceeded": True,
        "forced_close": True,
        "synthetic_tokens": 2,
        "reasoning_tokens": 3,
        "answer_tokens": 4,
        "tool_call_tokens": 5,
        "structured_tokens": 6,
        "budget_pressure": "hard_close",
        "cache_action": "append_prompt_only",
        "sampler_mode": "processed_argmax",
        "phase": "answer",
        "continuation_eligible": False,
    }


def test_decode_state_mapping_accepts_json_nulls() -> None:
    state = DecodeState.from_value(
        {
            "row_index": None,
            "step_index": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "forced_tokens_pending": None,
        }
    )

    assert state.to_json_dict() == {
        "row_index": 0,
        "step_index": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "phase": "done",
        "continuation_eligible": False,
    }


def test_generation_telemetry_import_is_torch_free() -> None:
    code = (
        "import sys\n"
        "from hipengine.generation import DecodeState, GenerationTelemetry\n"
        "GenerationTelemetry(decode_state=DecodeState()).to_json_dict()\n"
        "print('torch' in sys.modules)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.stdout.strip() == "False"
