#!/usr/bin/env python3
"""Validate sampled gfx11 GGUF concurrency through real OpenAI HTTP APIs.

The packet starts one prepared model behind a real localhost Uvicorn socket.
It repeats concurrent seeded blocking and SSE completion waves with exact token
IDs/logprobs, validates n>1 derived-row sampling, exercises stop and explicit
EOS retirement, and checks forced tools plus JSON-schema output.  Route deltas
must show packed sampled model ticks and zero serial model fallback.  This is a
correctness/serving-path gate, not a throughput benchmark.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import http.client
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from hipengine import LLM, SamplingParams
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.server import ServerConfig, create_app
from scripts.gguf_live_server_bench import (
    _artifact_backend_scope,
    _memory_snapshot,
    _read_compiler_version,
)
from scripts.gguf_production_load_gate import (
    _LocalUvicorn,
    _http_json,
    _memory_recovery_gate,
    _parse_sse_line,
    _wait_for_idle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
_SUPPORTED_BACKENDS = ("hip_gfx1100", "hip_gfx1151")
_SAMPLED_EXECUTION_PATH = "gguf_packed_ar_host_sampler_decode"
_SAMPLE_PROMPTS = (
    "Write a Python function that returns the first prime larger than n.",
    "In one sentence, name a practical benefit of unit tests.",
    "日本語で、コードレビューの利点を一文で説明してください。",
    "Give one concise reason to measure latency percentiles, not only averages.",
)
_PROVENANCE_ENV_KEYS = (
    "HIPENGINE_BACKEND",
    "HIPENGINE_HIP_ARCH",
    "HIPENGINE_COMPILER_VERSION_FILE",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_MAX_HW_QUEUES",
)
_EXACT_ENV: dict[str, str | None] = {
    "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN": "1",
    "HIPENGINE_GGUF_GDN_PREFILL_MODE": "exact",
    "HIPENGINE_GGUF_AR_STREAM_DECODE": "0",
    "HIPENGINE_GGUF_AR_PACKED_DECODE": "1",
    # Validate the package registry default rather than an external override.
    "HIPENGINE_PREFILL_DECODE_POLICY": None,
    "HIPENGINE_MAX_PREFILL_CHUNK_TOKENS": None,
}


def _failure_summary(failures: Sequence[str], **payload: Any) -> dict[str, Any]:
    unique = sorted(set(str(item) for item in failures))
    return {"passed": not unique, "failure_reasons": unique, **payload}


def _choice_sampled_failures(choice: Mapping[str, Any], index: int) -> list[str]:
    failures: list[str] = []
    hipengine = choice.get("hipengine")
    if not isinstance(hipengine, Mapping):
        return [f"choice_{index}_hipengine_metadata_missing"]
    generated_ids = hipengine.get("generated_token_ids")
    if not isinstance(generated_ids, list) or not generated_ids or not all(
        isinstance(token, int) for token in generated_ids
    ):
        failures.append(f"choice_{index}_generated_token_ids_missing")
    decode = hipengine.get("decode_state")
    if not isinstance(decode, Mapping):
        failures.append(f"choice_{index}_decode_state_missing")
        return failures
    if decode.get("request_id") is None:
        failures.append(f"choice_{index}_request_id_missing")
    if decode.get("sampler_fallback_reason") != "host_sampling_required":
        failures.append(f"choice_{index}_host_sampler_reason_missing")
    if decode.get("serial_decode_fallback") is not False:
        failures.append(f"choice_{index}_serial_model_fallback")
    if decode.get("native_sampler_rows") is not False:
        failures.append(f"choice_{index}_native_sampler_metadata_inexact")
    if decode.get("full_vocab_logits_d2h") is not True:
        failures.append(f"choice_{index}_full_vocab_logits_metadata_missing")
    if decode.get("execution_path") != _SAMPLED_EXECUTION_PATH:
        failures.append(f"choice_{index}_sampled_execution_path_inexact")
    return failures


def _logprob_signature(
    logprobs: Any,
    *,
    choice_index: int,
    failures: list[str],
) -> dict[str, Any]:
    if not isinstance(logprobs, Mapping):
        failures.append(f"choice_{choice_index}_logprobs_missing")
        return {
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
        }
    tokens = logprobs.get("tokens")
    selected = logprobs.get("token_logprobs")
    top = logprobs.get("top_logprobs")
    offsets = logprobs.get("text_offset")
    lengths = [
        len(value)
        for value in (tokens, selected, top, offsets)
        if isinstance(value, list)
    ]
    if (
        not isinstance(tokens, list)
        or not isinstance(selected, list)
        or not isinstance(top, list)
        or not isinstance(offsets, list)
        or len(lengths) != 4
        or len(set(lengths)) != 1
        or not tokens
    ):
        failures.append(f"choice_{choice_index}_logprob_shape_inexact")
        return {
            "tokens": list(tokens) if isinstance(tokens, list) else [],
            "token_logprobs": list(selected) if isinstance(selected, list) else [],
            "top_logprobs": copy.deepcopy(top) if isinstance(top, list) else [],
        }
    if any(not isinstance(token, str) for token in tokens):
        failures.append(f"choice_{choice_index}_logprob_token_type_inexact")
    if any(
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in selected
    ):
        failures.append(f"choice_{choice_index}_selected_logprob_nonfinite")
    for row in top:
        if not isinstance(row, Mapping) or not row:
            failures.append(f"choice_{choice_index}_top_logprobs_missing")
            continue
        if any(
            not isinstance(token, str)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for token, value in row.items()
        ):
            failures.append(f"choice_{choice_index}_top_logprob_nonfinite")
    extension = logprobs.get("hipengine")
    if isinstance(extension, Mapping) and extension.get("omitted_token_logprobs"):
        failures.append(f"choice_{choice_index}_logprob_omission")
    return {
        "tokens": [str(token) for token in tokens],
        "token_logprobs": [None if value is None else float(value) for value in selected],
        "top_logprobs": [
            {
                str(token): float(value)
                for token, value in row.items()
            }
            if isinstance(row, Mapping)
            else None
            for row in top
        ],
    }


def _validate_blocking_completion(
    payload: Any,
    *,
    expected_choices: int,
    require_logprobs: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return _failure_summary(["response_not_json_object"], choices=[], usage=None)
    if payload.get("object") != "text_completion":
        failures.append("response_object_inexact")
    raw_choices = payload.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) != int(expected_choices):
        failures.append("choice_count_inexact")
        raw_choices = raw_choices if isinstance(raw_choices, list) else []
    indexes = [choice.get("index") for choice in raw_choices if isinstance(choice, Mapping)]
    if indexes != list(range(int(expected_choices))):
        failures.append("choice_indexes_inexact")
    normalized: list[dict[str, Any]] = []
    for fallback_index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, Mapping):
            failures.append(f"choice_{fallback_index}_not_object")
            continue
        index = int(raw_choice.get("index", fallback_index))
        failures.extend(_choice_sampled_failures(raw_choice, index))
        text = raw_choice.get("text")
        if not isinstance(text, str):
            failures.append(f"choice_{index}_text_missing")
            text = ""
        finish_reason = raw_choice.get("finish_reason")
        finish_details = raw_choice.get("finish_details")
        if finish_reason not in {"stop", "length"}:
            failures.append(f"choice_{index}_finish_reason_inexact")
        if not isinstance(finish_details, Mapping):
            failures.append(f"choice_{index}_finish_details_missing")
            finish_details = {}
        logprob = (
            _logprob_signature(
                raw_choice.get("logprobs"),
                choice_index=index,
                failures=failures,
            )
            if require_logprobs
            else {"tokens": [], "token_logprobs": [], "top_logprobs": []}
        )
        if require_logprobs and "".join(logprob["tokens"]) != text:
            failures.append(f"choice_{index}_logprob_text_mismatch")
        hipengine = raw_choice.get("hipengine")
        generated_ids = (
            list(hipengine.get("generated_token_ids", []))
            if isinstance(hipengine, Mapping)
            else []
        )
        normalized.append(
            {
                "index": index,
                "text": text,
                "generated_token_ids": generated_ids,
                "finish_reason": finish_reason,
                "finish_details": copy.deepcopy(dict(finish_details)),
                "logprob_tokens": logprob["tokens"],
                "token_logprobs": logprob["token_logprobs"],
                "top_logprobs": logprob["top_logprobs"],
            }
        )
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        failures.append("usage_missing")
        usage = {}
    expected_completion_tokens = sum(len(row["generated_token_ids"]) for row in normalized)
    if int(usage.get("completion_tokens", -1)) != expected_completion_tokens:
        failures.append("completion_token_accounting_inexact")
    return _failure_summary(
        failures,
        choices=normalized,
        usage=copy.deepcopy(dict(usage)),
    )


def _sse_choice_decode_failures(
    choice: Mapping[str, Any],
    *,
    index: int,
) -> list[str]:
    hipengine = choice.get("hipengine")
    decode = hipengine.get("decode_state") if isinstance(hipengine, Mapping) else None
    if not isinstance(decode, Mapping):
        return [f"choice_{index}_stream_decode_state_missing"]
    failures: list[str] = []
    if decode.get("request_id") is None:
        failures.append(f"choice_{index}_stream_request_id_missing")
    if decode.get("sampler_fallback_reason") != "host_sampling_required":
        failures.append(f"choice_{index}_stream_host_sampler_reason_missing")
    if decode.get("serial_decode_fallback") is not False:
        failures.append(f"choice_{index}_stream_serial_model_fallback")
    if decode.get("execution_path") != _SAMPLED_EXECUTION_PATH:
        failures.append(f"choice_{index}_stream_execution_path_inexact")
    return failures


def _validate_sse_completion(
    payloads: Sequence[Any],
    *,
    done_seen: bool,
    expected_choices: int,
    reference: Mapping[str, Any] | None,
    require_logprobs: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    if not done_seen:
        failures.append("sse_done_sentinel_missing")
    rows = {
        index: {
            "index": index,
            "text_parts": [],
            "logprob_tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
            "finish_reason": None,
            "finish_details": None,
            "done_count": 0,
        }
        for index in range(int(expected_choices))
    }
    usage: dict[str, Any] | None = None
    for raw_payload in payloads:
        if not isinstance(raw_payload, Mapping):
            failures.append("sse_payload_not_object")
            continue
        raw_error = raw_payload.get("error")
        if isinstance(raw_error, Mapping):
            failures.append(f"sse_error_{raw_error.get('code', 'unknown')}")
        if isinstance(raw_payload.get("usage"), Mapping):
            usage = copy.deepcopy(dict(raw_payload["usage"]))
        choices = raw_payload.get("choices")
        if not isinstance(choices, list):
            continue
        for raw_choice in choices:
            if not isinstance(raw_choice, Mapping):
                failures.append("sse_choice_not_object")
                continue
            raw_index = raw_choice.get("index")
            if not isinstance(raw_index, int) or raw_index not in rows:
                failures.append("sse_choice_index_inexact")
                continue
            row = rows[raw_index]
            finish_reason = raw_choice.get("finish_reason")
            text = raw_choice.get("text")
            if finish_reason is None:
                if not isinstance(text, str):
                    failures.append(f"choice_{raw_index}_stream_text_missing")
                    continue
                if text:
                    row["text_parts"].append(text)
                    failures.extend(
                        _sse_choice_decode_failures(raw_choice, index=raw_index)
                    )
                    if require_logprobs:
                        logprob = _logprob_signature(
                            raw_choice.get("logprobs"),
                            choice_index=raw_index,
                            failures=failures,
                        )
                        if "".join(logprob["tokens"]) != text:
                            failures.append(f"choice_{raw_index}_stream_logprob_text_mismatch")
                        row["logprob_tokens"].extend(logprob["tokens"])
                        row["token_logprobs"].extend(logprob["token_logprobs"])
                        row["top_logprobs"].extend(logprob["top_logprobs"])
            else:
                row["done_count"] += 1
                row["finish_reason"] = str(finish_reason)
                details = raw_choice.get("finish_details")
                row["finish_details"] = (
                    copy.deepcopy(dict(details)) if isinstance(details, Mapping) else None
                )
    normalized: list[dict[str, Any]] = []
    for index in range(int(expected_choices)):
        row = rows[index]
        if row["done_count"] != 1:
            failures.append(f"choice_{index}_stream_done_count_inexact")
        if row["finish_reason"] not in {"stop", "length"}:
            failures.append(f"choice_{index}_stream_finish_reason_inexact")
        if not isinstance(row["finish_details"], Mapping):
            failures.append(f"choice_{index}_stream_finish_details_missing")
        normalized.append(
            {
                "index": index,
                "text": "".join(row["text_parts"]),
                "finish_reason": row["finish_reason"],
                "finish_details": row["finish_details"],
                "logprob_tokens": list(row["logprob_tokens"]),
                "token_logprobs": list(row["token_logprobs"]),
                "top_logprobs": copy.deepcopy(row["top_logprobs"]),
            }
        )
    if usage is None:
        failures.append("sse_usage_missing")
        usage = {}
    if reference is not None:
        reference_choices = reference.get("choices")
        if not isinstance(reference_choices, list) or len(reference_choices) != len(normalized):
            failures.append("blocking_reference_shape_inexact")
        else:
            for row, expected in zip(normalized, reference_choices, strict=True):
                index = int(row["index"])
                if row["text"] != expected.get("text"):
                    failures.append(f"choice_{index}_text_mismatch_vs_blocking")
                if row["finish_reason"] != expected.get("finish_reason"):
                    failures.append(f"choice_{index}_finish_mismatch_vs_blocking")
                if require_logprobs:
                    if row["logprob_tokens"] != expected.get("logprob_tokens"):
                        failures.append(f"choice_{index}_logprob_tokens_mismatch_vs_blocking")
                    if row["token_logprobs"] != expected.get("token_logprobs"):
                        failures.append(f"choice_{index}_selected_logprobs_mismatch_vs_blocking")
                    if row["top_logprobs"] != expected.get("top_logprobs"):
                        failures.append(f"choice_{index}_top_logprobs_mismatch_vs_blocking")
            expected_tokens = sum(
                len(row.get("generated_token_ids", ())) for row in reference_choices
            )
            if int(usage.get("completion_tokens", -1)) != expected_tokens:
                failures.append("sse_completion_token_accounting_inexact")
    return _failure_summary(failures, choices=normalized, usage=usage)


def _validate_finish_case(
    payload: Any,
    *,
    expected_backend_reason: str,
    expected_generated_ids: Sequence[int],
    expected_eos_token_id: int | None = None,
    expected_text: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return _failure_summary(["finish_response_not_object"], choice=None)
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return _failure_summary(["finish_choice_shape_inexact"], choice=None)
    choice = choices[0]
    failures.extend(_choice_sampled_failures(choice, 0))
    if choice.get("finish_reason") != "stop":
        failures.append("finish_reason_not_stop")
    details = choice.get("finish_details")
    if not isinstance(details, Mapping):
        failures.append("finish_details_missing")
        details = {}
    if details.get("reason") != str(expected_backend_reason):
        failures.append("backend_finish_reason_inexact")
    if expected_eos_token_id is not None and int(details.get("eos_token_id", -1)) != int(
        expected_eos_token_id
    ):
        failures.append("eos_token_id_inexact")
    hipengine = choice.get("hipengine")
    generated_ids = (
        list(hipengine.get("generated_token_ids", []))
        if isinstance(hipengine, Mapping)
        else []
    )
    if generated_ids != [int(token) for token in expected_generated_ids]:
        failures.append("finish_generated_ids_inexact")
    if expected_text is not None and choice.get("text") != expected_text:
        failures.append("finish_visible_text_inexact")
    usage = payload.get("usage")
    if not isinstance(usage, Mapping) or int(usage.get("completion_tokens", -1)) != len(
        generated_ids
    ):
        failures.append("finish_usage_inexact")
    return _failure_summary(
        failures,
        choice={
            "text": choice.get("text"),
            "generated_token_ids": generated_ids,
            "finish_reason": choice.get("finish_reason"),
            "finish_details": copy.deepcopy(dict(details)),
        },
    )


def _validate_tool_response(
    payload: Any,
    *,
    expected_name: str,
    expected_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return _failure_summary(["tool_response_not_object"], tool_call=None)
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return _failure_summary(["tool_choice_shape_inexact"], tool_call=None)
    choice = choices[0]
    failures.extend(_choice_sampled_failures(choice, 0))
    if choice.get("finish_reason") != "tool_calls":
        failures.append("tool_finish_reason_inexact")
    details = choice.get("finish_details")
    if not isinstance(details, Mapping) or details.get("reason") != "tool_calls":
        failures.append("tool_finish_details_inexact")
    message = choice.get("message")
    calls = message.get("tool_calls") if isinstance(message, Mapping) else None
    parsed_arguments: Any = None
    tool_call: dict[str, Any] | None = None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        failures.append("tool_call_shape_inexact")
    else:
        raw_call = calls[0]
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            failures.append("tool_function_missing")
        else:
            if function.get("name") != str(expected_name):
                failures.append("tool_name_inexact")
            try:
                parsed_arguments = json.loads(str(function.get("arguments", "")))
            except Exception:
                failures.append("tool_arguments_not_json")
            if parsed_arguments != dict(expected_arguments):
                failures.append("tool_arguments_inexact")
            tool_call = {
                "id": raw_call.get("id"),
                "type": raw_call.get("type"),
                "name": function.get("name"),
                "arguments": parsed_arguments,
            }
    hipengine = choice.get("hipengine")
    generated_ids = (
        list(hipengine.get("generated_token_ids", []))
        if isinstance(hipengine, Mapping)
        else []
    )
    usage = payload.get("usage")
    if not isinstance(usage, Mapping) or int(usage.get("completion_tokens", -1)) != len(
        generated_ids
    ):
        failures.append("tool_usage_inexact")
    return _failure_summary(
        failures,
        tool_call=tool_call,
        generated_token_ids=generated_ids,
    )


def _validate_structured_response(
    payload: Any,
    *,
    expected_value: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return _failure_summary(
            ["structured_response_not_object"],
            outcome=None,
            value=None,
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return _failure_summary(
            ["structured_choice_shape_inexact"],
            outcome=None,
            value=None,
        )
    choice = choices[0]
    failures.extend(_choice_sampled_failures(choice, 0))
    details = choice.get("finish_details")
    if not isinstance(details, Mapping):
        failures.append("structured_finish_details_missing")
        details = {}
    message = choice.get("message")
    text = message.get("content") if isinstance(message, Mapping) else None
    value: Any = None
    if not isinstance(text, str):
        failures.append("structured_text_missing")
    else:
        try:
            value = json.loads(text)
        except Exception:
            value = None
    if value == dict(expected_value):
        outcome = "schema_valid"
        if choice.get("finish_reason") != "stop" or details.get("reason") != "stop":
            failures.append("structured_valid_finish_inexact")
    else:
        outcome = "schema_violation_rejected"
        if choice.get("finish_reason") != "length":
            failures.append("structured_rejection_finish_reason_inexact")
        if details.get("reason") != "schema_violation":
            failures.append("structured_rejection_reason_missing")
        if details.get("phase") != "structured":
            failures.append("structured_rejection_phase_inexact")
        if details.get("continuation_eligible") is not False:
            failures.append("structured_rejection_continuation_inexact")
    hipengine = choice.get("hipengine")
    generated_ids = (
        list(hipengine.get("generated_token_ids", []))
        if isinstance(hipengine, Mapping)
        else []
    )
    usage = payload.get("usage")
    if not isinstance(usage, Mapping) or int(usage.get("completion_tokens", -1)) != len(
        generated_ids
    ):
        failures.append("structured_usage_inexact")
    return _failure_summary(
        failures,
        outcome=outcome,
        value=value,
        text=text,
        generated_token_ids=generated_ids,
        finish_reason=choice.get("finish_reason"),
        finish_details=copy.deepcopy(dict(details)),
    )


def _route_delta_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    minimum_sampled_rows: int,
    require_packed: bool,
) -> dict[str, Any]:
    before_counts = before.get("counts") if isinstance(before.get("counts"), Mapping) else {}
    after_counts = after.get("counts") if isinstance(after.get("counts"), Mapping) else {}
    count_names = sorted(set(before_counts) | set(after_counts))
    counts = {
        str(name): int(after_counts.get(name, 0)) - int(before_counts.get(name, 0))
        for name in count_names
    }
    before_widths = (
        before.get("physical_width_decode_steps")
        if isinstance(before.get("physical_width_decode_steps"), Mapping)
        else {}
    )
    after_widths = (
        after.get("physical_width_decode_steps")
        if isinstance(after.get("physical_width_decode_steps"), Mapping)
        else {}
    )
    width_names = sorted(set(before_widths) | set(after_widths), key=lambda item: int(item))
    widths = {
        str(name): int(after_widths.get(name, 0)) - int(before_widths.get(name, 0))
        for name in width_names
    }
    failures: list[str] = []
    if counts.get("host_sampler_requests", 0) < int(minimum_sampled_rows):
        failures.append("host_sampler_request_count_inexact")
    if counts.get("native_sampled_prefill_rows", 0) < int(minimum_sampled_rows):
        failures.append("sampled_prefill_count_inexact")
    if counts.get("serial_decode_fallback_steps", 0) != 0:
        failures.append("serial_model_fallback_observed")
    if counts.get("resident_fallback_requests", 0) != 0:
        failures.append("resident_request_fallback_observed")
    packed_width_steps = sum(value for width, value in widths.items() if int(width) > 1)
    if require_packed and counts.get("native_packed_decode_steps", 0) <= 0:
        failures.append("packed_sampled_step_missing")
    if require_packed and packed_width_steps <= 0:
        failures.append("physical_c_gt_1_step_missing")
    return _failure_summary(
        failures,
        counts=counts,
        physical_width_decode_steps=widths,
        packed_width_steps=packed_width_steps,
    )


def _routes_snapshot(runner: Any) -> dict[str, Any]:
    snapshot = runner.observability_snapshot()
    routes = snapshot.get("routes") if isinstance(snapshot, Mapping) else None
    if not isinstance(routes, Mapping):
        raise RuntimeError("GGUF runner did not expose route observability")
    return copy.deepcopy(dict(routes))


def _response_signature(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = summary.get("choices")
    if not isinstance(choices, list):
        return []
    keys = (
        "index",
        "text",
        "generated_token_ids",
        "finish_reason",
        "finish_details",
        "logprob_tokens",
        "token_logprobs",
        "top_logprobs",
    )
    return [
        {key: copy.deepcopy(row.get(key)) for key in keys if key in row}
        for row in choices
        if isinstance(row, Mapping)
    ]


def _http_post_json(
    host: str,
    port: int,
    path: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[int, Any]:
    connection = http.client.HTTPConnection(host, int(port), timeout=float(timeout_seconds))
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload, separators=(",", ":")),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        status = int(response.status)
        raw = response.read()
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw.decode("utf-8", errors="replace")}
        return status, body
    finally:
        connection.close()


def _http_post_sse(
    host: str,
    port: int,
    path: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[int, list[dict[str, Any]], bool, Any | None]:
    connection = http.client.HTTPConnection(host, int(port), timeout=float(timeout_seconds))
    payloads: list[dict[str, Any]] = []
    done_seen = False
    error_body: Any | None = None
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(payload, separators=(",", ":")),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        status = int(response.status)
        if status != 200:
            raw = response.read()
            try:
                error_body = json.loads(raw)
            except Exception:
                error_body = {"raw": raw.decode("utf-8", errors="replace")}
            return status, payloads, done_seen, error_body
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            item = _parse_sse_line(raw_line)
            if item is None:
                continue
            if item == "[DONE]":
                done_seen = True
            elif isinstance(item, dict):
                payloads.append(item)
        return status, payloads, done_seen, error_body
    finally:
        connection.close()


def _run_concurrent(
    jobs: Sequence[tuple[str, Callable[[], Any]]],
) -> dict[str, Any]:
    start = threading.Event()

    def invoke(fn: Callable[[], Any]) -> Any:
        if not start.wait(timeout=30.0):
            raise TimeoutError("concurrent HTTP start barrier timed out")
        return fn()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs))) as executor:
        futures = {label: executor.submit(invoke, fn) for label, fn in jobs}
        start.set()
        return {label: futures[label].result() for label, _fn in jobs}


@contextmanager
def _temporary_environment(updates: Mapping[str, str | None]) -> Iterator[None]:
    prior = {str(key): os.environ.get(str(key)) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(str(key), None)
            else:
                os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _sample_payload(
    served_model_name: str,
    prompt: str,
    *,
    stream: bool,
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": served_model_name,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 1.2,
        "top_p": 0.92,
        "top_k": 64,
        "repetition_penalty": 1.08,
        "seed": 17,
        "ignore_eos": True,
        "logprobs": 3,
        "stream": bool(stream),
    }
    if stream:
        payload["stream_options"] = {"include_hipengine": True, "include_usage": True}
    return payload


def _run_blocking_wave(
    host: str,
    port: int,
    *,
    served_model_name: str,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    jobs = [
        (
            f"row_{index}",
            lambda prompt=prompt: _http_post_json(
                host,
                port,
                "/v1/completions",
                _sample_payload(
                    served_model_name,
                    prompt,
                    stream=False,
                    max_tokens=max_tokens,
                ),
                timeout_seconds=timeout_seconds,
            ),
        )
        for index, prompt in enumerate(_SAMPLE_PROMPTS)
    ]
    exchanges = _run_concurrent(jobs)
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for label, (status, payload) in exchanges.items():
        summary = _validate_blocking_completion(
            payload,
            expected_choices=1,
            require_logprobs=True,
        )
        if status != 200:
            failures.append(f"{label}_http_{status}")
        if not summary["passed"]:
            failures.extend(f"{label}_{reason}" for reason in summary["failure_reasons"])
        rows[label] = {"status_code": status, "validation": summary}
    return _failure_summary(failures, rows=rows)


def _run_sse_wave(
    host: str,
    port: int,
    *,
    served_model_name: str,
    max_tokens: int,
    timeout_seconds: float,
    blocking_reference: Mapping[str, Any],
) -> dict[str, Any]:
    jobs = [
        (
            f"row_{index}",
            lambda prompt=prompt: _http_post_sse(
                host,
                port,
                "/v1/completions",
                _sample_payload(
                    served_model_name,
                    prompt,
                    stream=True,
                    max_tokens=max_tokens,
                ),
                timeout_seconds=timeout_seconds,
            ),
        )
        for index, prompt in enumerate(_SAMPLE_PROMPTS)
    ]
    exchanges = _run_concurrent(jobs)
    rows: dict[str, Any] = {}
    failures: list[str] = []
    reference_rows = blocking_reference.get("rows")
    for label, (status, payloads, done_seen, error) in exchanges.items():
        reference = None
        if isinstance(reference_rows, Mapping):
            row = reference_rows.get(label)
            if isinstance(row, Mapping):
                candidate = row.get("validation")
                if isinstance(candidate, Mapping):
                    reference = candidate
        summary = _validate_sse_completion(
            payloads,
            done_seen=done_seen,
            expected_choices=1,
            reference=reference,
            require_logprobs=True,
        )
        if status != 200:
            failures.append(f"{label}_http_{status}")
        if error is not None:
            failures.append(f"{label}_sse_error_body")
        if not summary["passed"]:
            failures.extend(f"{label}_{reason}" for reason in summary["failure_reasons"])
        rows[label] = {
            "status_code": status,
            "done_seen": done_seen,
            "validation": summary,
            **({"error": error} if error is not None else {}),
        }
    return _failure_summary(failures, rows=rows)


def _wave_signature(wave: Mapping[str, Any]) -> dict[str, Any]:
    rows = wave.get("rows")
    if not isinstance(rows, Mapping):
        return {}
    return {
        str(label): _response_signature(row.get("validation", {}))
        for label, row in sorted(rows.items())
        if isinstance(row, Mapping)
    }


def _feature_payloads(
    served_model_name: str,
    *,
    max_tokens: int,
) -> dict[str, tuple[str, dict[str, Any]]]:
    common = {
        "model": served_model_name,
        "max_tokens": int(max_tokens),
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 64,
        "seed": 17,
        "enable_thinking": False,
    }
    tool = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": "Call lookup exactly once with key README.md. Do not add prose.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up one repository key.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string", "enum": ["README.md"]}},
                        "required": ["key"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
    }
    structured = {
        **common,
        "messages": [
            {
                "role": "user",
                "content": 'Return exactly the JSON object {"status":"ok"}.',
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "status_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "const": "ok"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        },
    }
    return {
        "tool": ("/v1/chat/completions", tool),
        "structured": ("/v1/chat/completions", structured),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    if int(args.max_active_requests) < len(_SAMPLE_PROMPTS):
        raise ValueError("max-active-requests must cover the four-row sampled wave")
    compiler_version = _read_compiler_version(args.compiler_version_file)
    if args.require_cached_build and compiler_version is None:
        raise ValueError("require-cached-build requires compiler-version-file")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    max_sequence_length = int(args.max_sequence_length)
    max_pages_per_request = max(1, math.ceil(max_sequence_length / 256))
    env: dict[str, str | None] = {
        **_EXACT_ENV,
        "HIPENGINE_MAX_ACTIVE_REQUESTS": str(int(args.max_active_requests)),
        "HIPENGINE_MAX_PENDING_REQUESTS": str(int(args.max_pending_requests)),
        "HIPENGINE_KV_POOL_INITIAL_PAGES": str(max_pages_per_request),
        "HIPENGINE_KV_POOL_LOW_WATER_PAGES": str(max_pages_per_request),
        "HIPENGINE_KV_POOL_HIGH_WATER_PAGES": str(
            int(args.max_active_requests) * max_pages_per_request
        ),
        "HIPENGINE_KV_POOL_CHUNK_PAGES": str(max_pages_per_request),
        "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS": "0",
        **(
            {"HIPENGINE_COMPILER_VERSION_FILE": str(args.compiler_version_file)}
            if args.compiler_version_file is not None
            else {}
        ),
    }
    served_model_name = "qwen35-sampled-api-gate"
    started_at = time.perf_counter()
    cases: dict[str, Any] = {}
    route_gates: dict[str, Any] = {}
    final_idle: dict[str, Any] = {}
    final_metrics: Any = None
    baseline_memory: dict[str, Any] = {}
    final_memory: dict[str, Any] = {}
    capabilities: Any = None
    with _temporary_environment(env):
        llm = LLM(
            model,
            backend=str(args.backend),
            max_active_requests=int(args.max_active_requests),
        )
        try:
            adapter = llm._get_text_generator()
            llm.prepare(
                max_sequence_length=max_sequence_length,
                sampling_params=SamplingParams(
                    max_tokens=int(args.feature_max_tokens),
                    temperature=1.2,
                    top_p=0.92,
                    top_k=64,
                    repetition_penalty=1.08,
                    logprobs=True,
                    top_logprobs=3,
                ),
            )
            runner = adapter._runner
            tokenizer = runner.generator.tokenizer
            baseline_memory = _memory_snapshot("prepared_baseline", runner)
            app = create_app(
                ServerConfig(
                    model=str(model),
                    backend=str(args.backend),
                    quant=str(args.quant),
                    served_model_name=served_model_name,
                    eager_load=False,
                    startup_chat_smoke=False,
                    startup_scratch_probe=False,
                    metrics="prometheus",
                    generation_batch_window_ms=float(args.batch_window_ms),
                    max_context_tokens=max_sequence_length,
                    max_active_requests=int(args.max_active_requests),
                    max_queued_requests=int(args.max_queued_requests),
                    stream_queue_max_chunks=int(args.stream_queue_max_chunks),
                    shutdown_grace_seconds=10.0,
                ),
                llm=llm,
            )
            batcher = app.state.hipengine_generation_batcher
            with _LocalUvicorn(app) as server:
                host = "127.0.0.1"
                ready = _http_json(host, server.port, "GET", "/ready")
                if not bool(ready.get("ready")):
                    raise RuntimeError(f"server readiness failed: {ready}")
                capabilities = _http_json(
                    host,
                    server.port,
                    "GET",
                    "/v1/hipengine/capabilities",
                )
                logprob_capability = (
                    capabilities.get("features", {}).get("logprobs", {})
                    if isinstance(capabilities, Mapping)
                    else {}
                )
                cases["capabilities"] = _failure_summary(
                    []
                    if bool(logprob_capability.get("live_chunk_metadata"))
                    else ["live_stream_logprobs_not_advertised"],
                    logprobs=copy.deepcopy(logprob_capability),
                )

                wave_routes_before = _routes_snapshot(runner)
                blocking_first = _run_blocking_wave(
                    host,
                    server.port,
                    served_model_name=served_model_name,
                    max_tokens=int(args.sample_max_tokens),
                    timeout_seconds=float(args.request_timeout_seconds),
                )
                blocking_second = _run_blocking_wave(
                    host,
                    server.port,
                    served_model_name=served_model_name,
                    max_tokens=int(args.sample_max_tokens),
                    timeout_seconds=float(args.request_timeout_seconds),
                )
                blocking_repeat = _wave_signature(blocking_first) == _wave_signature(
                    blocking_second
                )
                cases["sampled_blocking_c4"] = _failure_summary(
                    [
                        *blocking_first["failure_reasons"],
                        *blocking_second["failure_reasons"],
                        *([] if blocking_repeat else ["seeded_blocking_repeat_mismatch"]),
                    ],
                    first=blocking_first,
                    second=blocking_second,
                    repeat_exact=blocking_repeat,
                )

                sse_first = _run_sse_wave(
                    host,
                    server.port,
                    served_model_name=served_model_name,
                    max_tokens=int(args.sample_max_tokens),
                    timeout_seconds=float(args.request_timeout_seconds),
                    blocking_reference=blocking_first,
                )
                sse_second = _run_sse_wave(
                    host,
                    server.port,
                    served_model_name=served_model_name,
                    max_tokens=int(args.sample_max_tokens),
                    timeout_seconds=float(args.request_timeout_seconds),
                    blocking_reference=blocking_first,
                )
                sse_repeat = _wave_signature(sse_first) == _wave_signature(sse_second)
                cases["sampled_sse_c4"] = _failure_summary(
                    [
                        *sse_first["failure_reasons"],
                        *sse_second["failure_reasons"],
                        *([] if sse_repeat else ["seeded_sse_repeat_mismatch"]),
                    ],
                    first=sse_first,
                    second=sse_second,
                    repeat_exact=sse_repeat,
                )
                wave_routes_after = _routes_snapshot(runner)
                route_gates["sampled_blocking_sse_waves"] = _route_delta_gate(
                    wave_routes_before,
                    wave_routes_after,
                    minimum_sampled_rows=4 * 4,
                    require_packed=True,
                )

                n_payload = _sample_payload(
                    served_model_name,
                    _SAMPLE_PROMPTS[0],
                    stream=False,
                    max_tokens=int(args.sample_max_tokens),
                )
                n_payload["n"] = int(args.n_choices)
                n_routes_before = _routes_snapshot(runner)
                n_status_first, n_body_first = _http_post_json(
                    host,
                    server.port,
                    "/v1/completions",
                    n_payload,
                    timeout_seconds=float(args.request_timeout_seconds),
                )
                n_status_second, n_body_second = _http_post_json(
                    host,
                    server.port,
                    "/v1/completions",
                    n_payload,
                    timeout_seconds=float(args.request_timeout_seconds),
                )
                n_first = _validate_blocking_completion(
                    n_body_first,
                    expected_choices=int(args.n_choices),
                    require_logprobs=True,
                )
                n_second = _validate_blocking_completion(
                    n_body_second,
                    expected_choices=int(args.n_choices),
                    require_logprobs=True,
                )
                n_repeat = _response_signature(n_first) == _response_signature(n_second)
                cases["n_gt_1"] = _failure_summary(
                    [
                        *([] if n_status_first == 200 else [f"first_http_{n_status_first}"]),
                        *([] if n_status_second == 200 else [f"second_http_{n_status_second}"]),
                        *n_first["failure_reasons"],
                        *n_second["failure_reasons"],
                        *([] if n_repeat else ["derived_row_seed_repeat_mismatch"]),
                    ],
                    first=n_first,
                    second=n_second,
                    repeat_exact=n_repeat,
                )
                n_routes_after = _routes_snapshot(runner)
                route_gates["n_gt_1"] = _route_delta_gate(
                    n_routes_before,
                    n_routes_after,
                    minimum_sampled_rows=2 * int(args.n_choices),
                    require_packed=True,
                )

                first_reference = blocking_first["rows"]["row_0"]["validation"]["choices"][0]
                reference_ids = [int(token) for token in first_reference["generated_token_ids"]]
                reference_token_texts = [str(token) for token in first_reference["logprob_tokens"]]
                if not reference_ids:
                    raise RuntimeError("sampled reference produced no token IDs")
                eos_payload = _sample_payload(
                    served_model_name,
                    _SAMPLE_PROMPTS[0],
                    stream=False,
                    max_tokens=int(args.sample_max_tokens),
                )
                eos_payload["ignore_eos"] = False
                eos_payload["eos_token_id"] = int(reference_ids[0])
                eos_status, eos_body = _http_post_json(
                    host,
                    server.port,
                    "/v1/completions",
                    eos_payload,
                    timeout_seconds=float(args.request_timeout_seconds),
                )
                eos_validation = _validate_finish_case(
                    eos_body,
                    expected_backend_reason="eos",
                    expected_generated_ids=reference_ids[:1],
                    expected_eos_token_id=reference_ids[0],
                )
                cases["explicit_eos"] = _failure_summary(
                    [
                        *([] if eos_status == 200 else [f"http_{eos_status}"]),
                        *eos_validation["failure_reasons"],
                    ],
                    status_code=eos_status,
                    validation=eos_validation,
                )

                stop_index: int | None = None
                stop_text: str | None = None
                for index, (token_id, token_text) in enumerate(
                    zip(reference_ids, reference_token_texts, strict=True)
                ):
                    if not token_text:
                        continue
                    encoded = tuple(int(token) for token in tokenizer.encode(token_text))
                    if encoded == (int(token_id),):
                        stop_index = index
                        stop_text = token_text
                        break
                if stop_index is None or stop_text is None:
                    cases["stop_sequence"] = _failure_summary(
                        ["no_single_token_stop_fixture_in_reference"],
                        status_code=None,
                        validation=None,
                    )
                else:
                    stop_payload = _sample_payload(
                        served_model_name,
                        _SAMPLE_PROMPTS[0],
                        stream=False,
                        max_tokens=int(args.sample_max_tokens),
                    )
                    stop_payload["stop"] = stop_text
                    stop_status, stop_body = _http_post_json(
                        host,
                        server.port,
                        "/v1/completions",
                        stop_payload,
                        timeout_seconds=float(args.request_timeout_seconds),
                    )
                    stop_validation = _validate_finish_case(
                        stop_body,
                        expected_backend_reason="stop",
                        expected_generated_ids=reference_ids[: stop_index + 1],
                        expected_text="".join(reference_token_texts[:stop_index]),
                    )
                    cases["stop_sequence"] = _failure_summary(
                        [
                            *([] if stop_status == 200 else [f"http_{stop_status}"]),
                            *stop_validation["failure_reasons"],
                        ],
                        status_code=stop_status,
                        stop_text=stop_text,
                        stop_token_id=reference_ids[stop_index],
                        validation=stop_validation,
                    )

                feature_jobs = [
                    (
                        label,
                        lambda path=path, payload=payload: _http_post_json(
                            host,
                            server.port,
                            path,
                            payload,
                            timeout_seconds=float(args.request_timeout_seconds),
                        ),
                    )
                    for label, (path, payload) in _feature_payloads(
                        served_model_name,
                        max_tokens=int(args.feature_max_tokens),
                    ).items()
                ]
                feature_exchanges = _run_concurrent(feature_jobs)
                tool_status, tool_body = feature_exchanges["tool"]
                tool_validation = _validate_tool_response(
                    tool_body,
                    expected_name="lookup",
                    expected_arguments={"key": "README.md"},
                )
                cases["tool_forcing"] = _failure_summary(
                    [
                        *([] if tool_status == 200 else [f"http_{tool_status}"]),
                        *tool_validation["failure_reasons"],
                    ],
                    status_code=tool_status,
                    validation=tool_validation,
                )
                structured_status, structured_body = feature_exchanges["structured"]
                structured_validation = _validate_structured_response(
                    structured_body,
                    expected_value={"status": "ok"},
                )
                cases["structured_json_schema"] = _failure_summary(
                    [
                        *(
                            []
                            if structured_status == 200
                            else [f"http_{structured_status}"]
                        ),
                        *structured_validation["failure_reasons"],
                    ],
                    status_code=structured_status,
                    validation=structured_validation,
                )

                final_idle = _wait_for_idle(
                    llm,
                    batcher,
                    timeout_seconds=float(args.idle_timeout_seconds),
                )
                final_metrics = _http_json(host, server.port, "GET", "/metrics")
            final_memory = _memory_snapshot("final", runner)
            resolved_backend = str(runner.generator.backend)
            target_arch = str(runner._shared_runner.target_arch)
        finally:
            llm.close()

    memory_recovery = _memory_recovery_gate(
        baseline_memory,
        final_memory,
        tracked_tolerance_bytes=int(args.tracked_memory_tolerance_mib) * 1024 * 1024,
    )
    case_passed = bool(cases) and all(bool(row.get("passed")) for row in cases.values())
    routes_passed = bool(route_gates) and all(
        bool(row.get("passed")) for row in route_gates.values()
    )
    ownership = final_idle.get("snapshot", {})
    ownership_loop = ownership.get("loop", {}).get("requests", {})
    ownership_runner = ownership.get("runner", {}).get("model_runner", {})
    final_ownership_passed = bool(
        int(ownership_loop.get("active", -1)) == 0
        and int(ownership_loop.get("pending", -1)) == 0
        and int(ownership_runner.get("active_requests", -1)) == 0
        and int(final_idle.get("generation_queue_depth", -1)) == 0
        and int(final_idle.get("generation_active_requests", -1)) == 0
    )
    clean_source_passed = not source_dirty
    passed = bool(
        case_passed
        and routes_passed
        and final_ownership_passed
        and memory_recovery["passed"]
        and clean_source_passed
    )
    command = [sys.executable, "scripts/gguf_sampled_api_gate.py", *sys.argv[1:]]
    scope = _artifact_backend_scope(resolved_backend, target_arch)
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant=str(args.quant),
        kv_dtype="bf16",
        command=command,
        environment={
            **{key: os.environ.get(key) for key in _PROVENANCE_ENV_KEYS},
            **env,
        },
        build_profile=f"{scope}_gguf_sampled_openai_api_gate",
        timing_protocol=(
            "one prepared model; real localhost Uvicorn; repeated concurrent seeded "
            "blocking and SSE c4; exact generated IDs and logprobs; repeated n>1; "
            "stop/EOS; forced tool; JSON schema; packed-route and ownership gates"
        ),
        warmups=0,
        repetitions=2,
        profiler={"used": False, "reason": "OpenAI correctness and route-observability gate"},
        hipcc_version=compiler_version,
    )
    return {
        "schema": 1,
        "kind": f"{scope}_gguf_sampled_openai_api_gate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if passed else "failed",
        "passed": passed,
        "performance_claim": False,
        "source": {"commit": source_commit, "dirty": source_dirty},
        "provenance": provenance,
        "configuration": {
            "model": str(model),
            "backend": str(args.backend),
            "quant": str(args.quant),
            "kv_dtype": "bf16",
            "max_sequence_length": max_sequence_length,
            "max_active_requests": int(args.max_active_requests),
            "generation_batch_window_ms": float(args.batch_window_ms),
            "sampled_rows_per_wave": len(_SAMPLE_PROMPTS),
            "sample_max_tokens": int(args.sample_max_tokens),
            "sampling": {
                "temperature": 1.2,
                "top_p": 0.92,
                "top_k": 64,
                "repetition_penalty": 1.08,
                "seed": 17,
                "top_logprobs": 3,
            },
            "n_choices": int(args.n_choices),
            "package_default_scheduler": True,
        },
        "capabilities": capabilities,
        "cases": cases,
        "route_gates": route_gates,
        "gates": {
            "cases_passed": case_passed,
            "routes_passed": routes_passed,
            "final_ownership_passed": final_ownership_passed,
            "memory_recovery": memory_recovery,
            "clean_source_passed": clean_source_passed,
        },
        "baseline_memory": baseline_memory,
        "final_memory": final_memory,
        "final_ownership": final_idle,
        "final_metrics_sha256": hashlib.sha256(
            str(final_metrics).encode("utf-8")
        ).hexdigest(),
        "command": shlex.join(command),
        "elapsed_seconds": time.perf_counter() - started_at,
        "limitations": [
            "This is a correctness and serving-path claim, not a throughput comparison.",
            "The retained scope is Qwen GGUF Q4_K_M with BF16 KV on the stated backend.",
            "Tool and structured cases validate one bounded schema each; they do not imply broad agentic quality.",
            "Prefix reuse, long-context memory pressure, and external-engine comparisons are separate gates.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=_SUPPORTED_BACKENDS, default="hip_gfx1151")
    parser.add_argument("--quant", default="gguf_q4_k_m")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--max-active-requests", type=int, default=8)
    parser.add_argument("--max-pending-requests", type=int, default=16)
    parser.add_argument("--max-queued-requests", type=int, default=16)
    parser.add_argument("--stream-queue-max-chunks", type=int, default=16)
    parser.add_argument("--batch-window-ms", type=float, default=100.0)
    parser.add_argument("--sample-max-tokens", type=int, default=6)
    parser.add_argument("--feature-max-tokens", type=int, default=64)
    parser.add_argument("--n-choices", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--idle-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--tracked-memory-tolerance-mib", type=int, default=64)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
