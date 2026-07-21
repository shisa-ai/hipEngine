"""Live OpenAI chat normalization for the coding-agent benchmark."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from hipengine.benchmark.agentic import AgenticBenchmarkError, AgenticWorkloadSuite
from hipengine.tokenization.identity import token_ids_sha256


_SYSTEM_POLICY = (
    "You are measuring a local coding-agent server against a synthetic repository. "
    "Use the specifically requested tool exactly once. Return only the tool call, with "
    "arguments matching the user request. Do not expose reasoning or raw tool markers.\n\n"
)
_RAW_MARKERS = ("<think", "</think", "<tool_call", "</tool_call")


@dataclass(frozen=True)
class RenderedWorkloadPrefix:
    workload_id: str
    target_tokens: int
    text: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str


@dataclass(frozen=True)
class ChatToolOracle:
    generated_token_ids: tuple[int, ...]
    name: str
    arguments: Mapping[str, Any]
    finish_reason: str


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticBenchmarkError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AgenticBenchmarkError(f"{label} must be an array")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgenticBenchmarkError(f"{label} must be a non-empty string")
    return value


def _token_row(value: Any, *, label: str) -> tuple[int, ...]:
    raw = _sequence(value, label=label)
    tokens: list[int] = []
    for index, token in enumerate(raw):
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise AgenticBenchmarkError(f"{label}[{index}] must be a non-negative integer")
        tokens.append(int(token))
    if not tokens:
        raise AgenticBenchmarkError(f"{label} must not be empty")
    return tuple(tokens)


def _finite_time(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AgenticBenchmarkError(f"{label} must be a finite timestamp")
    return float(value)


def render_workload_prefix(
    suite: AgenticWorkloadSuite,
    workload_id: str,
    *,
    tokenize: Callable[[str], Sequence[int]],
    detokenize: Callable[[Sequence[int]], str],
) -> RenderedWorkloadPrefix:
    """Repeat synthetic repository blocks and trim to an exact tokenizer target."""

    if workload_id not in suite.workloads:
        raise AgenticBenchmarkError(f"unknown workload_id {workload_id!r}")
    workload = suite.workloads[workload_id]
    target = int(workload["target_prefix_tokens"])
    repository = _mapping(suite.payload["repository_context"], label="repository_context")
    blocks = [str(item) for item in _sequence(repository["expansion_blocks"], label="blocks")]
    pieces = [f"Repository: {repository['name']}\n{repository['base']}\n"]
    tokens = tuple(int(token) for token in tokenize("".join(pieces)))
    block_index = 0
    while len(tokens) < target:
        pieces.append(f"\nContext block {block_index + 1}: {blocks[block_index % len(blocks)]}\n")
        block_index += 1
        tokens = tuple(int(token) for token in tokenize("".join(pieces)))
        if block_index > target * 2:
            raise AgenticBenchmarkError("repository prefix expansion made no tokenizer progress")
    selected = tokens[:target]
    text = str(detokenize(selected))
    roundtrip = tuple(int(token) for token in tokenize(text))
    if roundtrip != selected:
        raise AgenticBenchmarkError("detokenized repository prefix does not roundtrip exactly")
    return RenderedWorkloadPrefix(
        workload_id=str(workload_id),
        target_tokens=target,
        text=text,
        token_ids=selected,
        token_ids_sha256=token_ids_sha256(selected),
    )


def build_openai_tools(suite: AgenticWorkloadSuite) -> list[dict[str, Any]]:
    """Translate the fixture tool declarations into OpenAI function envelopes."""

    return [
        {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool["description"]),
                "strict": True,
                "parameters": copy.deepcopy(dict(tool["parameters"])),
            },
        }
        for tool in suite.tools.values()
    ]


def _stable_call_id(agent_id: str, turn_index: int) -> str:
    return f"call_{agent_id}_{int(turn_index)}"


def build_canonical_turn_messages(
    suite: AgenticWorkloadSuite,
    workload_id: str,
    *,
    turn_index: int,
    agent_id: str,
    prefix_text: str,
    system_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Build one turn from a stable fixture transcript, independent of random server IDs."""

    if workload_id not in suite.workloads:
        raise AgenticBenchmarkError(f"unknown workload_id {workload_id!r}")
    workload = suite.workloads[workload_id]
    turns = list(_sequence(workload["turns"], label=f"{workload_id}.turns"))
    if turn_index < 0 or turn_index >= len(turns):
        raise AgenticBenchmarkError("turn_index is out of range")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (system_policy if system_policy is not None else _SYSTEM_POLICY)
            + str(prefix_text),
        }
    ]
    for index in range(turn_index):
        prior = _mapping(turns[index], label=f"turns[{index}]")
        call_id = _stable_call_id(agent_id, index)
        messages.extend(
            [
                {"role": "user", "content": str(prior["user"])},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": str(prior["expected_tool"]),
                                "arguments": json.dumps(
                                    prior["expected_arguments"],
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(prior["tool_result"]),
                },
            ]
        )
    current = _mapping(turns[turn_index], label=f"turns[{turn_index}]")
    messages.append({"role": "user", "content": str(current["user"])})
    return messages


def _parse_tool_arguments(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, str):
        raise AgenticBenchmarkError(f"{label} must be a JSON string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AgenticBenchmarkError(f"{label} are not valid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise AgenticBenchmarkError(f"{label} must decode to an object")
    return dict(parsed)


def normalize_chat_oracle(
    suite: AgenticWorkloadSuite,
    workload_id: str,
    turn_index: int,
    payload: Mapping[str, Any],
) -> ChatToolOracle:
    """Validate one independent non-streaming exact-token tool oracle."""

    if workload_id not in suite.workloads:
        raise AgenticBenchmarkError(f"unknown workload_id {workload_id!r}")
    expected = suite.workloads[workload_id]["turns"][turn_index]
    choices = _sequence(payload.get("choices"), label="oracle.choices")
    if len(choices) != 1:
        raise AgenticBenchmarkError("oracle must contain exactly one choice")
    choice = _mapping(choices[0], label="oracle.choices[0]")
    if choice.get("finish_reason") != "tool_calls":
        raise AgenticBenchmarkError("oracle did not finish with tool_calls")
    message = _mapping(choice.get("message"), label="oracle message")
    visible = str(message.get("content") or "")
    if any(marker in visible for marker in _RAW_MARKERS):
        raise AgenticBenchmarkError("oracle leaked raw model markup")
    if visible.strip():
        raise AgenticBenchmarkError("oracle emitted content alongside a tool-only response")
    calls = _sequence(message.get("tool_calls"), label="oracle tool_calls")
    if len(calls) != 1:
        raise AgenticBenchmarkError("oracle must contain exactly one tool call")
    function = _mapping(
        _mapping(calls[0], label="oracle tool call").get("function"), label="oracle function"
    )
    name = _nonempty(function.get("name"), label="oracle tool name")
    arguments = _parse_tool_arguments(function.get("arguments"), label="oracle tool arguments")
    if name != expected["expected_tool"] or arguments != expected["expected_arguments"]:
        raise AgenticBenchmarkError("oracle tool call differs from deterministic fixture")
    hipengine = _mapping(choice.get("hipengine"), label="oracle hipengine metadata")
    generated = _token_row(hipengine.get("generated_token_ids"), label="oracle generated_token_ids")
    usage = _mapping(payload.get("usage"), label="oracle usage")
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens != len(generated):
        raise AgenticBenchmarkError("oracle completion token accounting is inexact")
    return ChatToolOracle(
        generated_token_ids=generated,
        name=name,
        arguments=arguments,
        finish_reason="tool_calls",
    )


def _tool_event_parts(
    choice: Mapping[str, Any],
    *,
    observed_at: float,
    calls: dict[int, dict[str, Any]],
) -> bool:
    delta = choice.get("delta")
    if not isinstance(delta, Mapping):
        return False
    raw_calls = delta.get("tool_calls")
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes, bytearray)):
        return False
    emitted = False
    for raw_call in raw_calls:
        call = _mapping(raw_call, label="SSE tool call delta")
        index = int(call.get("index", 0))
        row = calls.setdefault(index, {"id": None, "name": None, "parts": [], "times": []})
        if call.get("id") is not None:
            row["id"] = str(call["id"])
        function = call.get("function")
        if isinstance(function, Mapping):
            if function.get("name") is not None:
                row["name"] = str(function["name"])
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                row["parts"].append(arguments)
                row["times"].append(float(observed_at))
                emitted = True
    return emitted


def _prefix_record_from_telemetry(
    telemetry: Mapping[str, Any],
    *,
    cache_mode: str,
    prompt_tokens: int,
) -> dict[str, Any]:
    diagnostics = telemetry.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    raw = diagnostics.get("prefix_cache")
    if cache_mode == "radix" and not isinstance(raw, Mapping):
        raise AgenticBenchmarkError("SSE tool turn is missing radix prefix telemetry")
    if not isinstance(raw, Mapping):
        return {
            "lookup": False,
            "hit": False,
            "reused_tokens": 0,
            "cache_bytes": 0,
        }
    if raw.get("mode") != cache_mode:
        raise AgenticBenchmarkError("SSE prefix telemetry mode differs from the server mode")

    def boolean(name: str) -> bool:
        value = raw.get(name)
        if not isinstance(value, bool):
            raise AgenticBenchmarkError(f"SSE prefix telemetry {name} must be boolean")
        return value

    def count(name: str) -> int:
        value = raw.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AgenticBenchmarkError(
                f"SSE prefix telemetry {name} must be a non-negative integer"
            )
        return int(value)

    block_size_tokens = count("block_size_tokens")
    eligible = boolean("eligible")
    lookup = boolean("lookup")
    hit = boolean("hit")
    snapshot_hit = boolean("snapshot_hit")
    admission_fallback = boolean("admission_fallback")
    matched_tokens = count("matched_tokens")
    reused_tokens = count("reused_tokens")
    avoided_tokens = count("avoided_prefill_tokens")
    executed_tokens = count("executed_prefill_tokens")
    reused_pages = count("reused_pages")
    reused_page_bytes = count("reused_page_bytes")
    state_clone_bytes = count("state_clone_bytes")
    cache_entries = count("cache_resident_entries")
    cache_pages = count("cache_resident_pages")
    cache_bytes = count("cache_resident_bytes")
    source = raw.get("source")
    if source not in {None, "active_current", "completed_snapshot"}:
        raise AgenticBenchmarkError("SSE prefix telemetry source is unsupported")
    fallback_reason = raw.get("fallback_reason")
    if fallback_reason is not None and (
        not isinstance(fallback_reason, str) or not fallback_reason
    ):
        raise AgenticBenchmarkError("SSE prefix telemetry fallback_reason is invalid")
    if block_size_tokens <= 0 or matched_tokens > int(prompt_tokens):
        raise AgenticBenchmarkError("SSE prefix telemetry boundary is invalid")
    if avoided_tokens != reused_tokens or executed_tokens + reused_tokens != int(
        prompt_tokens
    ):
        raise AgenticBenchmarkError("SSE prefix telemetry prefill token accounting is inexact")
    if reused_tokens != reused_pages * block_size_tokens:
        raise AgenticBenchmarkError("SSE prefix telemetry page accounting is inexact")
    if (reused_pages == 0) != (reused_page_bytes == 0):
        raise AgenticBenchmarkError("SSE prefix telemetry page-byte accounting is inexact")
    if reused_pages and reused_page_bytes % reused_pages:
        raise AgenticBenchmarkError("SSE prefix telemetry page bytes are not uniform")
    if (cache_entries == 0) != (cache_bytes == 0) or (
        cache_pages and not cache_entries
    ):
        raise AgenticBenchmarkError("SSE prefix cache residency telemetry is inconsistent")
    if hit:
        if not (eligible and lookup and reused_tokens > 0 and source is not None):
            raise AgenticBenchmarkError("SSE prefix hit telemetry is inconsistent")
        if (
            matched_tokens != reused_tokens
            or matched_tokens >= int(prompt_tokens)
            or state_clone_bytes <= 0
        ):
            raise AgenticBenchmarkError("SSE prefix hit boundary telemetry is inexact")
        if snapshot_hit != (source == "completed_snapshot"):
            raise AgenticBenchmarkError("SSE prefix snapshot source telemetry is inconsistent")
        if admission_fallback or fallback_reason is not None:
            raise AgenticBenchmarkError("SSE prefix hit cannot also report a fallback")
    elif (
        reused_tokens
        or reused_pages
        or reused_page_bytes
        or state_clone_bytes
        or source is not None
        or snapshot_hit
    ):
        raise AgenticBenchmarkError("SSE prefix miss reports reused state")
    if cache_mode == "radix":
        supported_fallbacks = {
            "sampling_unsupported",
            "prompt_too_short",
            "miss",
            "full_prompt_boundary_requires_suffix",
            "state_source_unavailable",
            "shared_admission_capacity",
        }
        if not hit and fallback_reason not in supported_fallbacks:
            raise AgenticBenchmarkError("SSE radix miss has no explicit fallback reason")
        if eligible != lookup:
            raise AgenticBenchmarkError("SSE radix eligibility/lookup telemetry is inconsistent")
        if admission_fallback != (fallback_reason == "shared_admission_capacity"):
            raise AgenticBenchmarkError("SSE radix admission fallback telemetry is inconsistent")
        if fallback_reason in {"sampling_unsupported", "prompt_too_short", "miss"} and matched_tokens:
            raise AgenticBenchmarkError("SSE radix fallback unexpectedly reports a boundary")
        if fallback_reason == "full_prompt_boundary_requires_suffix" and matched_tokens != int(
            prompt_tokens
        ):
            raise AgenticBenchmarkError("SSE full-prompt fallback boundary is inexact")
        if fallback_reason in {"state_source_unavailable", "shared_admission_capacity"} and not (
            0 < matched_tokens < int(prompt_tokens)
        ):
            raise AgenticBenchmarkError("SSE radix matched fallback boundary is inexact")
    elif (
        eligible
        or lookup
        or hit
        or matched_tokens
        or cache_bytes
        or fallback_reason != "cache_off"
    ):
        raise AgenticBenchmarkError("SSE prefix telemetry is incompatible with cache_mode=off")
    return {
        "block_size_tokens": block_size_tokens,
        "eligible": eligible,
        "lookup": lookup,
        "hit": hit,
        "source": source,
        "matched_tokens": matched_tokens,
        "reused_tokens": reused_tokens,
        "avoided_prefill_tokens": avoided_tokens,
        "executed_prefill_tokens": executed_tokens,
        "reused_pages": reused_pages,
        "reused_page_bytes": reused_page_bytes,
        "state_clone_bytes": state_clone_bytes,
        "snapshot_hit": snapshot_hit,
        "admission_fallback": admission_fallback,
        "fallback_reason": fallback_reason,
        "cache_entries": cache_entries,
        "cache_pages": cache_pages,
        "cache_bytes": cache_bytes,
    }


def normalize_chat_sse_turn(
    suite: AgenticWorkloadSuite,
    *,
    workload_id: str,
    turn_index: int,
    run_id: str,
    agent_id: str,
    session_id: str,
    request_id: str,
    prompt_token_ids: Sequence[int],
    submitted_at_s: float,
    tool_result_submitted_at_s: float,
    oracle: ChatToolOracle,
    events: Sequence[tuple[float, Mapping[str, Any] | str]],
    cache_mode: str,
) -> dict[str, Any]:
    """Normalize one measured SSE tool turn against its exact c1 oracle."""

    if cache_mode not in {"off", "radix"}:
        raise AgenticBenchmarkError("cache_mode must be off or radix")
    expected = suite.workloads[workload_id]["turns"][turn_index]
    submitted = _finite_time(submitted_at_s, label="submitted_at_s")
    result_submitted = _finite_time(tool_result_submitted_at_s, label="tool_result_submitted_at_s")
    prompt_ids = _token_row(prompt_token_ids, label="prompt_token_ids")
    calls: dict[int, dict[str, Any]] = {}
    first_public_token: float | None = None
    tool_ready: float | None = None
    response_done: float | None = None
    done_seen = False
    finish_reason: str | None = None
    usage: Mapping[str, Any] = {}
    telemetry: Mapping[str, Any] = {}
    response_generated_ids: tuple[int, ...] | None = None
    public_text: list[str] = []

    for raw_time, payload in events:
        observed_at = _finite_time(raw_time, label="SSE event timestamp")
        if payload == "[DONE]":
            done_seen = True
            continue
        event = _mapping(payload, label="SSE event")
        if event.get("error") is not None:
            raise AgenticBenchmarkError("SSE tool turn returned an error")
        if isinstance(event.get("usage"), Mapping):
            usage = event["usage"]
        raw_choices = event.get("choices", ())
        if not isinstance(raw_choices, Sequence) or isinstance(
            raw_choices, (str, bytes, bytearray)
        ):
            continue
        for raw_choice in raw_choices:
            choice = _mapping(raw_choice, label="SSE choice")
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                for field in ("content", "reasoning_content"):
                    value = delta.get(field)
                    if isinstance(value, str) and value:
                        public_text.append(value)
                        if first_public_token is None:
                            first_public_token = observed_at
                if _tool_event_parts(choice, observed_at=observed_at, calls=calls):
                    if first_public_token is None:
                        first_public_token = observed_at
                    tool_ready = observed_at
            choice_telemetry = choice.get("hipengine")
            if isinstance(choice_telemetry, Mapping) and choice_telemetry:
                telemetry = choice_telemetry
                if choice_telemetry.get("generated_token_ids") is not None:
                    observed_ids = _token_row(
                        choice_telemetry.get("generated_token_ids"),
                        label="SSE generated_token_ids",
                    )
                    if (
                        response_generated_ids is not None
                        and response_generated_ids != observed_ids
                    ):
                        raise AgenticBenchmarkError(
                            "SSE generated token IDs changed within the response"
                        )
                    response_generated_ids = observed_ids
                    generated_count = choice_telemetry.get("generated_tokens")
                    if generated_count not in {None, len(observed_ids)}:
                        raise AgenticBenchmarkError(
                            "SSE generated token ID accounting is inexact"
                        )
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
                response_done = observed_at

    if not done_seen:
        raise AgenticBenchmarkError("SSE stream is missing [DONE]")
    if finish_reason != "tool_calls" or response_done is None:
        raise AgenticBenchmarkError("SSE tool turn did not finish with tool_calls")
    if len(calls) != 1 or 0 not in calls:
        raise AgenticBenchmarkError("SSE tool turn must contain exactly one tool call")
    call = calls[0]
    name = _nonempty(call.get("name"), label="SSE tool name")
    arguments_text = "".join(str(part) for part in call["parts"])
    try:
        arguments = _parse_tool_arguments(arguments_text, label="SSE tool arguments")
    except AgenticBenchmarkError as exc:
        raise AgenticBenchmarkError("SSE tool arguments are not valid JSON") from exc
    if (
        name != oracle.name
        or arguments != oracle.arguments
        or name != expected["expected_tool"]
        or arguments != expected["expected_arguments"]
    ):
        raise AgenticBenchmarkError("SSE tool call differs from oracle or deterministic fixture")
    visible_text = "".join(public_text)
    if any(marker in visible_text for marker in _RAW_MARKERS):
        raise AgenticBenchmarkError("SSE tool turn leaked raw model markup")
    if visible_text.strip():
        raise AgenticBenchmarkError("SSE emitted content alongside a tool-only response")
    if (
        response_generated_ids is not None
        and response_generated_ids != oracle.generated_token_ids
    ):
        raise AgenticBenchmarkError("SSE generated token IDs differ from oracle")
    generated_ids = (
        oracle.generated_token_ids
        if response_generated_ids is None
        else response_generated_ids
    )
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and completion_tokens != len(generated_ids):
        raise AgenticBenchmarkError("SSE completion token accounting differs from exact IDs")
    if first_public_token is None or tool_ready is None:
        raise AgenticBenchmarkError("SSE tool turn emitted no public tool arguments")
    if result_submitted < response_done:
        raise AgenticBenchmarkError("tool result was submitted before the SSE response completed")

    decode = telemetry.get("decode_state")
    decode = decode if isinstance(decode, Mapping) else {}
    timing_scope = str(telemetry.get("timing_scope") or "choice")
    if timing_scope not in {"choice", "batch", "request"}:
        raise AgenticBenchmarkError("SSE tool turn has unsupported timing_scope")
    batch_id = telemetry.get("batch_id")
    if timing_scope == "batch" and (batch_id is None or not str(batch_id).strip()):
        raise AgenticBenchmarkError("SSE tool turn is missing backend batch_id")
    if batch_id is None or not str(batch_id).strip():
        batch_id = f"{timing_scope}:{request_id}"
    timing_owner = telemetry.get("timing_owner")
    if timing_owner is None and timing_scope != "batch":
        timing_owner = True
    if not isinstance(timing_owner, bool):
        raise AgenticBenchmarkError("SSE tool turn is missing timing_owner")
    group_rows = telemetry.get("group_rows", 1)
    if not isinstance(group_rows, int) or isinstance(group_rows, bool) or group_rows <= 0:
        raise AgenticBenchmarkError("SSE tool turn has invalid group_rows")
    sampler_mode = _nonempty(decode.get("sampler_mode"), label="SSE sampler_mode")
    raw_d2h = decode.get("logits_d2h_bytes", 0)
    if not isinstance(raw_d2h, int) or isinstance(raw_d2h, bool) or raw_d2h < 0:
        raise AgenticBenchmarkError("SSE logits_d2h_bytes is invalid")
    serial = decode.get("serial_decode_fallback", False)
    if not isinstance(serial, bool):
        raise AgenticBenchmarkError("SSE serial_decode_fallback is invalid")
    prefix_record = _prefix_record_from_telemetry(
        telemetry,
        cache_mode=cache_mode,
        prompt_tokens=len(prompt_ids),
    )
    backend_timing = telemetry.get("timing")
    backend_timing = backend_timing if isinstance(backend_timing, Mapping) else {}
    raw_prefill_ms = backend_timing.get("prefill_ms")
    if raw_prefill_ms is not None and (
        not isinstance(raw_prefill_ms, (int, float))
        or isinstance(raw_prefill_ms, bool)
        or not math.isfinite(float(raw_prefill_ms))
        or float(raw_prefill_ms) < 0.0
    ):
        raise AgenticBenchmarkError("SSE backend prefill_ms is invalid")

    generated = list(generated_ids)
    generated_source = (
        "matched_nonstreaming_oracle"
        if response_generated_ids is None
        else "response"
    )
    return {
        "workload_id": str(workload_id),
        "workload_sha256": suite.workload_sha256(workload_id),
        "run_id": str(run_id),
        "agent_id": str(agent_id),
        "session_id": str(session_id),
        "turn_index": int(turn_index),
        "request_id": str(request_id),
        "prompt": {
            "token_count": len(prompt_ids),
            "token_ids_sha256": token_ids_sha256(prompt_ids),
        },
        "output": {
            "generated_token_ids": generated,
            "generated_token_ids_sha256": token_ids_sha256(generated),
            "generated_token_ids_source": generated_source,
            "sse_exact_ids_observed": response_generated_ids is not None,
            "raw_markup_leaked": False,
        },
        "tool": {
            "expected_name": str(expected["expected_tool"]),
            "name": name,
            "declared_schema_sha256": suite.tool_schema_sha256(name),
            "call_id": _nonempty(call.get("id"), label="SSE tool call id"),
            "arguments": copy.deepcopy(dict(arguments)),
            "arguments_json_valid": True,
            "schema_valid": True,
            "result_linked": True,
        },
        "timing": {
            "submitted_at_s": submitted,
            "first_token_at_s": first_public_token,
            "token_observed_at_s": [first_public_token],
            "token_event_token_counts": [len(generated)],
            "token_timing_mode": "buffered_public",
            "tool_call_ready_at_s": tool_ready,
            "response_done_at_s": response_done,
            "tool_result_submitted_at_s": result_submitted,
        },
        "backend": {
            "batch_id": str(batch_id),
            "timing_scope": timing_scope,
            "timing_owner": timing_owner,
            "sampler_mode": sampler_mode,
            "logits_d2h_bytes": int(raw_d2h) * len(generated),
            "physical_width": int(group_rows),
            "serial_fallback": serial,
            **(
                {}
                if raw_prefill_ms is None
                else {"prefill_ms": float(raw_prefill_ms)}
            ),
        },
        "prefix": prefix_record,
        "finish": {"reason": "tool_calls", "retry_count": 0},
    }


def final_ownership_from_server(
    ready_payload: Mapping[str, Any],
    sessions_payload: Mapping[str, Any],
    *,
    cache_mode: str,
    allowed_cache_bytes: int = 0,
) -> dict[str, int]:
    """Map public readiness/session state into the A0 final-ownership envelope."""

    if ready_payload.get("ready") is not True:
        raise AgenticBenchmarkError("server is not ready after agentic run")
    queue = _mapping(ready_payload.get("queue"), label="ready.queue")
    pool = _mapping(
        _mapping(ready_payload.get("kv_capacity"), label="ready.kv_capacity").get("pool"),
        label="ready.kv_capacity.pool",
    )
    sessions = sessions_payload.get("sessions", ())
    sessions = _sequence(sessions, label="sessions.sessions")
    continuations = sessions_payload.get("continuations", {})
    continuations = continuations if isinstance(continuations, Mapping) else {}
    pending_sessions = int(sessions_payload.get("pending_creations", 0))
    active_continuations = int(continuations.get("active", 0))
    pending = int(queue.get("depth", 0))
    active = int(queue.get("active_requests", 0))
    worker_active = bool(queue.get("worker_active", False))
    refs = int(pool.get("refcounted_pages", 0))
    pins = int(pool.get("pinned_pages", 0))
    cache_pages = 0
    cache_entries = 0
    cache_bytes = 0
    allowed = int(allowed_cache_bytes)
    if allowed < 0:
        raise AgenticBenchmarkError("allowed_cache_bytes must be non-negative")
    if cache_mode == "off":
        allowed = 0
    elif cache_mode == "radix":
        prefix = _mapping(ready_payload.get("prefix_cache"), label="ready.prefix_cache")
        if prefix.get("mode") != "radix":
            raise AgenticBenchmarkError("ready prefix cache mode is not radix")
        cache_entries = int(
            prefix.get("retained_snapshot_entries", prefix.get("snapshot_entries", 0))
        )
        snapshot_entries = int(prefix.get("snapshot_entries", 0))
        snapshot_limit = int(prefix.get("snapshot_limit", -1))
        snapshot_bytes = int(prefix.get("snapshot_bytes", 0))
        cache_pages = int(prefix.get("retained_kv_pages", 0))
        retained_kv_bytes = int(prefix.get("retained_kv_bytes", 0))
        cache_bytes = int(prefix.get("resident_bytes", 0))
        server_limit = int(prefix.get("resident_limit_bytes", -1))
        if any(
            value < 0
            for value in (
                cache_entries,
                snapshot_entries,
                snapshot_limit,
                snapshot_bytes,
                cache_pages,
                retained_kv_bytes,
                cache_bytes,
                server_limit,
            )
        ):
            raise AgenticBenchmarkError("ready prefix cache ownership is invalid")
        if (
            snapshot_entries > snapshot_limit
            or cache_entries != snapshot_entries
            or (cache_entries == 0) != (cache_pages == 0)
            or (cache_pages == 0) != (retained_kv_bytes == 0)
        ):
            raise AgenticBenchmarkError("ready prefix cache entry/page ownership is inconsistent")
        if cache_bytes != snapshot_bytes + retained_kv_bytes:
            raise AgenticBenchmarkError("ready prefix cache byte ownership is inconsistent")
        if cache_pages > refs or cache_bytes > server_limit:
            raise AgenticBenchmarkError("ready prefix cache exceeds its residency bound")
        allowed = server_limit
    else:
        raise AgenticBenchmarkError("cache_mode must be off or radix")
    request_refs = refs - cache_pages
    if any(
        (
            pending,
            active,
            worker_active,
            request_refs,
            pins,
            len(sessions),
            pending_sessions,
            active_continuations,
        )
    ):
        raise AgenticBenchmarkError("server is not idle after agentic run")
    return {
        "pending_requests": pending,
        "active_requests": active,
        "stream_producers": int(worker_active),
        "model_active_requests": active,
        "session_count": len(sessions),
        "kv_refcounted_pages": request_refs,
        "kv_pinned_pages": pins,
        "graph_owners": pins,
        "workspace_owners": active,
        "cache_resident_entries": cache_entries,
        "cache_resident_pages": cache_pages,
        "cache_resident_bytes": cache_bytes,
        "allowed_cache_bytes": allowed,
    }


__all__ = [
    "ChatToolOracle",
    "RenderedWorkloadPrefix",
    "build_canonical_turn_messages",
    "build_openai_tools",
    "final_ownership_from_server",
    "normalize_chat_oracle",
    "normalize_chat_sse_turn",
    "render_workload_prefix",
]
