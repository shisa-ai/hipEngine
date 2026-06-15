from __future__ import annotations

import subprocess
import sys

from hipengine.generation import (
    DecodePhase,
    DecodeState,
    FinishDetails,
    GenerationOutput,
    GenerationStreamChunk,
    GenerationTelemetry,
    TokenLogprob,
)


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
                "sampler_fallback_reason": "processed_logits_required",
                "post_thinking_forced_tokens_pending": ["88", "89"],
                "post_thinking_forced_token_reason": "tool_choice_required",
                "force_sequence_completion_token_sequences": [["90", "91"]],
                "force_sequence_completion_reason": "tool_call_close_repair",
                "full_vocab_logits_d2h": False,
                "logits_d2h_bytes": "0",
                "execution_path": "scheduler_native_packed_prefill_serial_decode",
                "native_compact_prefill": True,
                "native_caware_decode": False,
                "serial_decode_fallback": True,
                "native_sampler_rows": False,
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
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "greedy_fast",
            "post_thinking_forced_tokens_pending": [88, 89],
            "post_thinking_forced_token_reason": "tool_choice_required",
            "force_sequence_completion_token_sequences": [[90, 91]],
            "force_sequence_completion_reason": "tool_call_close_repair",
            "full_vocab_logits_d2h": False,
            "logits_d2h_bytes": 0,
            "execution_path": "scheduler_native_packed_prefill_serial_decode",
            "native_compact_prefill": True,
            "native_caware_decode": False,
            "serial_decode_fallback": True,
            "native_sampler_rows": False,
        },
        "event": "done",
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }


def test_generation_telemetry_decode_counts_accept_phase_metadata() -> None:
    telemetry = GenerationTelemetry.from_decode_counts(
        prompt_tokens=5,
        generated_tokens=3,
        phase=DecodePhase.ANSWER,
        reasoning_tokens=2,
        answer_tokens=1,
        forced_tokens_pending=(42,),
        post_thinking_forced_tokens_pending=(77, 78),
        post_thinking_forced_token_reason="tool_choice_required",
        force_sequence_completion_token_sequences=((90, 91),),
        force_sequence_completion_reason="tool_call_close_repair",
        budget_pressure="hard_close",
        sampler_mode="processed_argmax",
        sampler_fallback_reason="processed_logits_required",
        full_vocab_logits_d2h=True,
        logits_d2h_bytes=1024,
        execution_path="scheduler_native_packed_prefill_native_decode",
        native_compact_prefill=True,
        native_caware_decode=True,
        serial_decode_fallback=False,
        native_sampler_rows=True,
        timing={"prefill_ms": 12.5, "decode_ms": 3},
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )

    payload = telemetry.to_json_dict()
    assert payload["decode_state"] == {
        "row_index": 0,
        "step_index": 3,
        "prompt_tokens": 5,
        "generated_tokens": 3,
        "phase": "answer",
        "continuation_eligible": False,
        "reasoning_tokens": 2,
        "answer_tokens": 1,
        "forced_tokens_pending": [42],
        "post_thinking_forced_tokens_pending": [77, 78],
        "post_thinking_forced_token_reason": "tool_choice_required",
        "force_sequence_completion_token_sequences": [[90, 91]],
        "force_sequence_completion_reason": "tool_call_close_repair",
        "sampler_fallback_reason": "processed_logits_required",
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 1024,
        "execution_path": "scheduler_native_packed_prefill_native_decode",
        "native_compact_prefill": True,
        "native_caware_decode": True,
        "serial_decode_fallback": False,
        "native_sampler_rows": True,
    }
    assert payload["timing"] == {"prefill_ms": 12.5, "decode_ms": 3.0}
    assert payload["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


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


def test_generation_stream_chunk_preserves_token_logprobs() -> None:
    chunk = GenerationStreamChunk.from_value(
        {
            "text": "answer",
            "token_logprobs": (
                TokenLogprob(token_id=7, token_text="answer", logprob=-0.25),
            ),
            "telemetry": {"decode_state": {"phase": "answer", "generated_tokens": 1}},
        }
    )

    assert chunk.text == "answer"
    assert chunk.token_logprobs == (
        TokenLogprob(token_id=7, token_text="answer", logprob=-0.25),
    )
    assert chunk.telemetry is not None
    assert chunk.telemetry.decode_state is not None
    assert chunk.telemetry.decode_state.phase == "answer"


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
