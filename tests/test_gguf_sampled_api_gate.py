from __future__ import annotations

import copy

from scripts.gguf_sampled_api_gate import (
    _route_delta_gate,
    _validate_blocking_completion,
    _validate_finish_case,
    _validate_sse_completion,
    _validate_structured_response,
    _validate_tool_response,
)


def _decode_state(*, request_id: str = "7", serial: bool = False) -> dict[str, object]:
    return {
        "request_id": request_id,
        "generated_tokens": 1,
        "phase": "done",
        "sampler_mode": "host_logits_sample",
        "sampler_fallback_reason": "host_sampling_required",
        "full_vocab_logits_d2h": True,
        "execution_path": "gguf_packed_ar_host_sampler_decode",
        "native_caware_decode": True,
        "serial_decode_fallback": serial,
        "native_sampler_rows": False,
    }


def _blocking_payload(*, choices: int = 2) -> dict[str, object]:
    rows = []
    for index in range(choices):
        token = chr(ord("A") + index)
        rows.append(
            {
                "text": token,
                "index": index,
                "logprobs": {
                    "tokens": [token],
                    "token_logprobs": [-0.25 - index],
                    "top_logprobs": [{token: -0.25 - index}],
                    "text_offset": [0],
                },
                "finish_reason": "length",
                "finish_details": {"reason": "length", "length_limit": 1},
                "hipengine": {
                    "generated_token_ids": [10 + index],
                    "generated_tokens": 1,
                    "decode_state": _decode_state(request_id=str(7 + index)),
                },
            }
        )
    return {
        "object": "text_completion",
        "choices": rows,
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": choices,
            "total_tokens": 8 + choices,
        },
    }


def test_blocking_completion_validator_requires_exact_sampled_metadata_and_logprobs() -> None:
    summary = _validate_blocking_completion(
        _blocking_payload(),
        expected_choices=2,
        require_logprobs=True,
    )

    assert summary["passed"] is True
    assert summary["failure_reasons"] == []
    assert [row["generated_token_ids"] for row in summary["choices"]] == [[10], [11]]
    assert [row["text"] for row in summary["choices"]] == ["A", "B"]

    serial = _blocking_payload()
    serial["choices"][0]["hipengine"]["decode_state"]["serial_decode_fallback"] = True
    serial_summary = _validate_blocking_completion(
        serial,
        expected_choices=2,
        require_logprobs=True,
    )
    assert serial_summary["passed"] is False
    assert "choice_0_serial_model_fallback" in serial_summary["failure_reasons"]

    missing_ids = _blocking_payload()
    del missing_ids["choices"][1]["hipengine"]["generated_token_ids"]
    missing_summary = _validate_blocking_completion(
        missing_ids,
        expected_choices=2,
        require_logprobs=True,
    )
    assert missing_summary["passed"] is False
    assert "choice_1_generated_token_ids_missing" in missing_summary["failure_reasons"]


def test_sse_completion_validator_reconstructs_each_choice_and_matches_blocking() -> None:
    reference = _validate_blocking_completion(
        _blocking_payload(),
        expected_choices=2,
        require_logprobs=True,
    )
    payloads = [
        {
            "object": "text_completion",
            "choices": [
                {
                    "text": "A",
                    "index": 0,
                    "logprobs": {
                        "tokens": ["A"],
                        "token_logprobs": [-0.25],
                        "top_logprobs": [{"A": -0.25}],
                        "text_offset": [0],
                    },
                    "finish_reason": None,
                    "hipengine": {"decode_state": _decode_state(request_id="7")},
                }
            ],
        },
        {
            "object": "text_completion",
            "choices": [
                {
                    "text": "B",
                    "index": 1,
                    "logprobs": {
                        "tokens": ["B"],
                        "token_logprobs": [-1.25],
                        "top_logprobs": [{"B": -1.25}],
                        "text_offset": [0],
                    },
                    "finish_reason": None,
                    "hipengine": {"decode_state": _decode_state(request_id="8")},
                }
            ],
        },
        {
            "object": "text_completion",
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length",
                    "finish_details": {"reason": "length", "length_limit": 1},
                    "hipengine": {"decode_state": _decode_state(request_id="7")},
                }
            ],
        },
        {
            "object": "text_completion",
            "choices": [
                {
                    "text": "",
                    "index": 1,
                    "logprobs": None,
                    "finish_reason": "length",
                    "finish_details": {"reason": "length", "length_limit": 1},
                    "hipengine": {"decode_state": _decode_state(request_id="8")},
                }
            ],
        },
        {
            "object": "text_completion",
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        },
    ]

    summary = _validate_sse_completion(
        payloads,
        done_seen=True,
        expected_choices=2,
        reference=reference,
        require_logprobs=True,
    )

    assert summary["passed"] is True
    assert summary["failure_reasons"] == []
    assert [row["text"] for row in summary["choices"]] == ["A", "B"]

    no_done = _validate_sse_completion(
        payloads,
        done_seen=False,
        expected_choices=2,
        reference=reference,
        require_logprobs=True,
    )
    assert no_done["passed"] is False
    assert "sse_done_sentinel_missing" in no_done["failure_reasons"]


def test_sse_validator_rejects_text_or_logprob_drift() -> None:
    reference = _validate_blocking_completion(
        _blocking_payload(choices=1),
        expected_choices=1,
        require_logprobs=True,
    )
    payloads = [
        {
            "choices": [
                {
                    "text": "X",
                    "index": 0,
                    "logprobs": {
                        "tokens": ["X"],
                        "token_logprobs": [-0.25],
                        "top_logprobs": [{"X": -0.25}],
                        "text_offset": [0],
                    },
                    "finish_reason": None,
                    "hipengine": {"decode_state": _decode_state()},
                }
            ]
        },
        {
            "choices": [
                {
                    "text": "",
                    "index": 0,
                    "finish_reason": "length",
                    "finish_details": {"reason": "length", "length_limit": 1},
                    "hipengine": {"decode_state": _decode_state()},
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}},
    ]

    summary = _validate_sse_completion(
        payloads,
        done_seen=True,
        expected_choices=1,
        reference=reference,
        require_logprobs=True,
    )

    assert summary["passed"] is False
    assert "choice_0_text_mismatch_vs_blocking" in summary["failure_reasons"]
    assert "choice_0_logprob_tokens_mismatch_vs_blocking" in summary["failure_reasons"]


def test_route_delta_gate_requires_packed_sampled_work_and_zero_serial_fallback() -> None:
    before = {
        "counts": {
            "host_sampler_requests": 2,
            "native_sampled_prefill_rows": 2,
            "native_packed_decode_steps": 3,
            "serial_decode_fallback_steps": 0,
            "resident_fallback_requests": 0,
        },
        "physical_width_decode_steps": {"1": 2, "2": 1, "4": 0, "8": 0},
    }
    after = copy.deepcopy(before)
    after["counts"].update(
        {
            "host_sampler_requests": 6,
            "native_sampled_prefill_rows": 6,
            "native_packed_decode_steps": 8,
        }
    )
    after["physical_width_decode_steps"].update({"1": 4, "2": 4, "4": 1})

    summary = _route_delta_gate(before, after, minimum_sampled_rows=4, require_packed=True)

    assert summary["passed"] is True
    assert summary["counts"]["host_sampler_requests"] == 4
    assert summary["physical_width_decode_steps"]["4"] == 1

    failed_after = copy.deepcopy(after)
    failed_after["counts"]["serial_decode_fallback_steps"] = 1
    failed = _route_delta_gate(before, failed_after, minimum_sampled_rows=4, require_packed=True)
    assert failed["passed"] is False
    assert "serial_model_fallback_observed" in failed["failure_reasons"]


def test_finish_tool_and_structured_validators_fail_closed() -> None:
    eos = _blocking_payload(choices=1)
    eos_choice = eos["choices"][0]
    eos_choice["finish_reason"] = "stop"
    eos_choice["finish_details"] = {"reason": "eos", "eos_token_id": 10}
    eos_summary = _validate_finish_case(
        eos,
        expected_backend_reason="eos",
        expected_generated_ids=[10],
        expected_eos_token_id=10,
    )
    assert eos_summary["passed"] is True

    stop = _blocking_payload(choices=1)
    stop_choice = stop["choices"][0]
    stop_choice["text"] = ""
    stop_choice["finish_reason"] = "stop"
    stop_choice["finish_details"] = {"reason": "stop", "stop_sequence": [10]}
    stop_summary = _validate_finish_case(
        stop,
        expected_backend_reason="stop",
        expected_generated_ids=[10],
        expected_text="",
    )
    assert stop_summary["passed"] is True

    tool_payload = {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "finish_details": {"reason": "tool_calls", "phase": "tool_call"},
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"README.md"}'},
                        }
                    ],
                },
                "hipengine": {
                    "generated_token_ids": [20, 21],
                    "decode_state": _decode_state(),
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    }
    tool = _validate_tool_response(
        tool_payload,
        expected_name="lookup",
        expected_arguments={"key": "README.md"},
    )
    assert tool["passed"] is True

    structured_payload = {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": {"reason": "stop", "phase": "structured"},
                "message": {"role": "assistant", "content": '{"status":"ok"}'},
                "hipengine": {
                    "generated_token_ids": [30, 31],
                    "decode_state": _decode_state(),
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    }
    structured = _validate_structured_response(
        structured_payload,
        expected_value={"status": "ok"},
    )
    assert structured["passed"] is True

    bad_tool = copy.deepcopy(tool_payload)
    bad_tool["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "not-json"
    assert _validate_tool_response(
        bad_tool,
        expected_name="lookup",
        expected_arguments={"key": "README.md"},
    )["passed"] is False

    bad_structured = copy.deepcopy(structured_payload)
    bad_structured["choices"][0]["message"]["content"] = "not-json"
    assert _validate_structured_response(
        bad_structured,
        expected_value={"status": "ok"},
    )["passed"] is False
