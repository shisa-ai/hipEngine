"""Fail-closed workload and rollup helpers for coding-agent server benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.benchmark.prompts import file_sha256
from hipengine.tokenization.identity import token_ids_sha256


AGENTIC_WORKLOAD_KIND = "hipengine.agentic_coding_workloads"
AGENTIC_RECORDS_KIND = "hipengine_agentic_coding_records"
AGENTIC_ARTIFACT_KIND = "hipengine_agentic_coding_benchmark"
AGENTIC_SCHEMA_VERSION = 1
DEFAULT_AGENTIC_WORKLOADS = Path("benchmarks/prompts/agentic-coding-v1.json")
_LANES = frozenset({"deterministic", "sampled", "auto_tool"})
_CACHE_MODES = frozenset({"off", "radix"})
_TOKEN_TIMING_MODES = frozenset({"live_exact", "buffered_public"})
_ZERO_OWNERSHIP_FIELDS = (
    "pending_requests",
    "active_requests",
    "stream_producers",
    "model_active_requests",
    "session_count",
    "kv_refcounted_pages",
    "kv_pinned_pages",
    "graph_owners",
    "workspace_owners",
)


class AgenticBenchmarkError(ValueError):
    """Raised when agentic benchmark input cannot support an exact claim."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgenticBenchmarkError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AgenticBenchmarkError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgenticBenchmarkError(f"{label} must be a non-negative integer")
    return int(value)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgenticBenchmarkError(f"{label} must be an object")
    return value


def _token_ids(value: Any, *, label: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not _is_sequence(value):
        raise AgenticBenchmarkError(f"{label} must be a token-ID array")
    result: list[int] = []
    for index, token in enumerate(value):
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise AgenticBenchmarkError(f"{label}[{index}] must be a non-negative integer")
        result.append(int(token))
    if not result and not allow_empty:
        raise AgenticBenchmarkError(f"{label} must not be empty")
    return tuple(result)


def _sha256_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise AgenticBenchmarkError(f"{label} must be a lowercase SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise AgenticBenchmarkError(f"{label} must be a lowercase SHA-256 string") from exc
    if value != value.lower():
        raise AgenticBenchmarkError(f"{label} must be a lowercase SHA-256 string")
    return value


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return _is_finite_number(value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return _is_sequence(value)
    if expected == "null":
        return value is None
    return False


def _validate_fixture_arguments(value: Any, schema: Mapping[str, Any], *, label: str) -> None:
    expected = schema.get("type")
    if isinstance(expected, str) and not _schema_type_matches(value, expected):
        raise AgenticBenchmarkError(f"{label} must have JSON type {expected}")
    enum = schema.get("enum")
    if _is_sequence(enum) and value not in enum:
        raise AgenticBenchmarkError(f"{label} is not in the allowed enum")
    if isinstance(value, str):
        minimum = schema.get("minLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise AgenticBenchmarkError(f"{label} is shorter than minLength")
    if isinstance(value, Mapping):
        required = schema.get("required", ())
        if _is_sequence(required):
            for key in required:
                if key not in value:
                    raise AgenticBenchmarkError(f"{label}.{key} is required")
        properties = schema.get("properties", {})
        properties = properties if isinstance(properties, Mapping) else {}
        if schema.get("additionalProperties") is False:
            extra = sorted(str(key) for key in value if key not in properties)
            if extra:
                raise AgenticBenchmarkError(f"{label} has undeclared properties: {extra}")
        for key, item in value.items():
            subschema = properties.get(key)
            if isinstance(subschema, Mapping):
                _validate_fixture_arguments(item, subschema, label=f"{label}.{key}")


@dataclass(frozen=True)
class AgenticWorkloadSuite:
    """Validated immutable identity for one checked-in agent workload suite."""

    path: Path
    payload: Mapping[str, Any]
    file_sha256: str
    canonical_sha256: str
    tools: Mapping[str, Mapping[str, Any]]
    workloads: Mapping[str, Mapping[str, Any]]

    @property
    def kind(self) -> str:
        return str(self.payload["kind"])

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    def workload_sha256(self, workload_id: str) -> str:
        try:
            workload = self.workloads[str(workload_id)]
        except KeyError as exc:
            raise AgenticBenchmarkError(f"unknown workload_id {workload_id!r}") from exc
        return _canonical_sha256(workload)

    def tool_schema_sha256(self, tool_name: str) -> str:
        try:
            tool = self.tools[str(tool_name)]
        except KeyError as exc:
            raise AgenticBenchmarkError(f"unknown tool name {tool_name!r}") from exc
        return _canonical_sha256(tool["parameters"])

    @staticmethod
    def token_ids_sha256(token_ids: Sequence[int]) -> str:
        return token_ids_sha256(token_ids)

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "schema_version": self.schema_version,
            "suite": str(self.payload["suite"]),
            "file_sha256": self.file_sha256,
            "canonical_sha256": self.canonical_sha256,
            "workload_sha256": {
                workload_id: self.workload_sha256(workload_id)
                for workload_id in sorted(self.workloads)
            },
        }


def load_agentic_workload_suite(
    path: str | Path = DEFAULT_AGENTIC_WORKLOADS,
) -> AgenticWorkloadSuite:
    """Load and validate a checked-in synthetic coding-agent workload suite."""

    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    root = _mapping(payload, label=str(fixture_path))
    if root.get("kind") != AGENTIC_WORKLOAD_KIND:
        raise AgenticBenchmarkError(f"{fixture_path} has unsupported workload kind")
    if root.get("schema_version") != AGENTIC_SCHEMA_VERSION:
        raise AgenticBenchmarkError(f"{fixture_path} has unsupported schema_version")
    _nonempty_string(root.get("suite"), label="suite")
    _nonempty_string(root.get("description"), label="description")

    raw_repository = _mapping(root.get("repository_context"), label="repository_context")
    _nonempty_string(raw_repository.get("name"), label="repository_context.name")
    _nonempty_string(raw_repository.get("base"), label="repository_context.base")
    expansion = raw_repository.get("expansion_blocks")
    if not _is_sequence(expansion) or not expansion:
        raise AgenticBenchmarkError("repository_context.expansion_blocks must be non-empty")
    for index, block in enumerate(expansion):
        _nonempty_string(block, label=f"repository_context.expansion_blocks[{index}]")

    raw_tools = root.get("tools")
    if not _is_sequence(raw_tools) or not raw_tools:
        raise AgenticBenchmarkError("tools must be a non-empty array")
    tools: dict[str, Mapping[str, Any]] = {}
    for index, raw_tool in enumerate(raw_tools):
        tool = _mapping(raw_tool, label=f"tools[{index}]")
        name = _nonempty_string(tool.get("name"), label=f"tools[{index}].name")
        if name in tools:
            raise AgenticBenchmarkError(f"duplicate tool name {name!r}")
        if tool.get("strict") is not True:
            raise AgenticBenchmarkError(f"tools[{index}].strict must be true")
        parameters = _mapping(tool.get("parameters"), label=f"tools[{index}].parameters")
        if parameters.get("type") != "object":
            raise AgenticBenchmarkError(f"tools[{index}].parameters.type must be object")
        tools[name] = copy.deepcopy(dict(tool))

    raw_workloads = root.get("workloads")
    if not _is_sequence(raw_workloads) or not raw_workloads:
        raise AgenticBenchmarkError("workloads must be a non-empty array")
    workloads: dict[str, Mapping[str, Any]] = {}
    for workload_index, raw_workload in enumerate(raw_workloads):
        workload = _mapping(raw_workload, label=f"workloads[{workload_index}]")
        workload_id = _nonempty_string(workload.get("id"), label=f"workloads[{workload_index}].id")
        if workload_id in workloads:
            raise AgenticBenchmarkError(f"duplicate workload id {workload_id!r}")
        _positive_int(
            workload.get("target_prefix_tokens"),
            label=f"workloads[{workload_index}].target_prefix_tokens",
        )
        if workload.get("history_mode") not in {"stable_prefix", "growing_tool_results"}:
            raise AgenticBenchmarkError(f"workloads[{workload_index}].history_mode is unsupported")
        turns = workload.get("turns")
        if not _is_sequence(turns) or not turns:
            raise AgenticBenchmarkError(f"workloads[{workload_index}].turns must be non-empty")
        for turn_index, raw_turn in enumerate(turns):
            turn = _mapping(raw_turn, label=f"workloads[{workload_index}].turns[{turn_index}]")
            _nonempty_string(turn.get("user"), label=f"{workload_id}.turns[{turn_index}].user")
            expected_tool = _nonempty_string(
                turn.get("expected_tool"),
                label=f"{workload_id}.turns[{turn_index}].expected_tool",
            )
            if expected_tool not in tools:
                raise AgenticBenchmarkError(
                    f"{workload_id}.turns[{turn_index}] references undeclared tool {expected_tool!r}"
                )
            arguments = _mapping(
                turn.get("expected_arguments"),
                label=f"{workload_id}.turns[{turn_index}].expected_arguments",
            )
            _validate_fixture_arguments(
                arguments,
                _mapping(tools[expected_tool]["parameters"], label=f"tool {expected_tool}"),
                label=f"{workload_id}.turns[{turn_index}].expected_arguments",
            )
            _nonempty_string(
                turn.get("tool_result"),
                label=f"{workload_id}.turns[{turn_index}].tool_result",
            )
        workloads[workload_id] = copy.deepcopy(dict(workload))

    exact_file_hash = file_sha256(fixture_path)
    return AgenticWorkloadSuite(
        path=fixture_path,
        payload=copy.deepcopy(dict(root)),
        file_sha256=exact_file_hash,
        # The committed byte stream is the canonical suite identity. Individual
        # workload hashes use canonical JSON so formatting-only fixture edits do
        # not alter per-workload identity accidentally.
        canonical_sha256=exact_file_hash,
        tools=tools,
        workloads=workloads,
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile values must not be empty")
    if not _is_finite_number(q) or not 0.0 <= float(q) <= 100.0:
        raise ValueError("percentile q must be between 0 and 100")
    normalized = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("percentile values must be finite")
    if len(normalized) == 1:
        return normalized[0]
    position = (len(normalized) - 1) * float(q) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return normalized[lower]
    fraction = position - lower
    return normalized[lower] + fraction * (normalized[upper] - normalized[lower])


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "p50": percentile(normalized, 50.0),
        "p95": percentile(normalized, 95.0),
        "p99": percentile(normalized, 99.0),
        "min": min(normalized),
        "max": max(normalized),
    }


def _required_time(timing: Mapping[str, Any], key: str, *, record_index: int) -> float:
    value = timing.get(key)
    if not _is_finite_number(value):
        raise AgenticBenchmarkError(f"record[{record_index}].timing.{key} is required")
    return float(value)


def _validate_final_ownership(value: Any) -> dict[str, int]:
    ownership = _mapping(value, label="final_ownership")
    normalized: dict[str, int] = {}
    for field in _ZERO_OWNERSHIP_FIELDS:
        count = _nonnegative_int(ownership.get(field), label=f"final_ownership.{field}")
        if count != 0:
            raise AgenticBenchmarkError(f"final_ownership.{field} must be zero")
        normalized[field] = count
    resident = _nonnegative_int(
        ownership.get("cache_resident_bytes"), label="final_ownership.cache_resident_bytes"
    )
    allowed = _nonnegative_int(
        ownership.get("allowed_cache_bytes"), label="final_ownership.allowed_cache_bytes"
    )
    if resident > allowed:
        raise AgenticBenchmarkError(
            "final_ownership.cache_resident_bytes exceeds allowed_cache_bytes"
        )
    normalized["cache_resident_bytes"] = resident
    normalized["allowed_cache_bytes"] = allowed
    return normalized


def _validate_turn_records(
    suite: AgenticWorkloadSuite,
    records_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    if records_payload.get("kind") != AGENTIC_RECORDS_KIND:
        raise AgenticBenchmarkError("records kind is unsupported")
    if records_payload.get("schema_version") != AGENTIC_SCHEMA_VERSION:
        raise AgenticBenchmarkError("records schema_version is unsupported")
    configuration = dict(_mapping(records_payload.get("configuration"), label="configuration"))
    _nonempty_string(configuration.get("id"), label="configuration.id")
    lane = configuration.get("lane")
    if lane not in _LANES:
        raise AgenticBenchmarkError(f"configuration.lane must be one of {sorted(_LANES)}")
    concurrency = _positive_int(configuration.get("concurrency"), label="configuration.concurrency")
    if configuration.get("cache_mode") not in _CACHE_MODES:
        raise AgenticBenchmarkError(
            f"configuration.cache_mode must be one of {sorted(_CACHE_MODES)}"
        )
    _nonempty_string(configuration.get("backend"), label="configuration.backend")
    _nonempty_string(configuration.get("model"), label="configuration.model")
    if configuration.get("performance_claim", False) is not False:
        raise AgenticBenchmarkError(
            "A0 normalized-record artifacts cannot set performance_claim=true"
        )

    raw_records = records_payload.get("turn_records")
    if not _is_sequence(raw_records) or not raw_records:
        raise AgenticBenchmarkError("turn_records must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    agents: set[tuple[str, str]] = set()
    agent_sessions: dict[tuple[str, str], str] = {}
    agent_workloads: dict[tuple[str, str], str] = {}
    agent_turns: dict[tuple[str, str], list[int]] = {}

    for record_index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, label=f"record[{record_index}]")
        workload_id = _nonempty_string(
            record.get("workload_id"), label=f"record[{record_index}].workload_id"
        )
        if workload_id not in suite.workloads:
            raise AgenticBenchmarkError(f"record[{record_index}] has unknown workload_id")
        if record.get("workload_sha256") != suite.workload_sha256(workload_id):
            raise AgenticBenchmarkError(f"record[{record_index}] workload hash mismatch")
        run_id = _nonempty_string(record.get("run_id"), label=f"record[{record_index}].run_id")
        agent_id = _nonempty_string(
            record.get("agent_id"), label=f"record[{record_index}].agent_id"
        )
        session_id = _nonempty_string(
            record.get("session_id"), label=f"record[{record_index}].session_id"
        )
        turn_index = _nonnegative_int(
            record.get("turn_index"), label=f"record[{record_index}].turn_index"
        )
        turns = suite.workloads[workload_id]["turns"]
        if turn_index >= len(turns):
            raise AgenticBenchmarkError(f"record[{record_index}].turn_index is out of range")
        request_id = _nonempty_string(
            record.get("request_id"), label=f"record[{record_index}].request_id"
        )
        if request_id in request_ids:
            raise AgenticBenchmarkError(f"duplicate request_id {request_id!r}")
        request_ids.add(request_id)
        agent_key = (run_id, agent_id)
        agents.add(agent_key)
        if agent_key in agent_sessions and agent_sessions[agent_key] != session_id:
            raise AgenticBenchmarkError(f"agent {run_id}/{agent_id} changed session_id")
        if agent_key in agent_workloads and agent_workloads[agent_key] != workload_id:
            raise AgenticBenchmarkError(f"agent {run_id}/{agent_id} changed workload_id")
        agent_sessions[agent_key] = session_id
        agent_workloads[agent_key] = workload_id
        agent_turns.setdefault(agent_key, []).append(turn_index)

        prompt = _mapping(record.get("prompt"), label=f"record[{record_index}].prompt")
        _positive_int(prompt.get("token_count"), label=f"record[{record_index}].prompt.token_count")
        _sha256_string(
            prompt.get("token_ids_sha256"),
            label=f"record[{record_index}].prompt.token_ids_sha256",
        )

        output = _mapping(record.get("output"), label=f"record[{record_index}].output")
        generated = _token_ids(
            output.get("generated_token_ids"),
            label=f"record[{record_index}].output.generated_token_ids",
        )
        generated_hash = _sha256_string(
            output.get("generated_token_ids_sha256"),
            label=f"record[{record_index}].output.generated_token_ids_sha256",
        )
        id_source = output.get("generated_token_ids_source")
        if id_source not in {"response", "matched_nonstreaming_oracle"}:
            raise AgenticBenchmarkError(
                f"record[{record_index}].output.generated_token_ids_source is unsupported"
            )
        observed_in_sse = output.get("sse_exact_ids_observed")
        if not isinstance(observed_in_sse, bool):
            raise AgenticBenchmarkError(
                f"record[{record_index}].output.sse_exact_ids_observed must be boolean"
            )
        if (id_source == "response") != observed_in_sse:
            raise AgenticBenchmarkError(
                f"record[{record_index}] exact-ID source/observation metadata is inconsistent"
            )
        if generated_hash != token_ids_sha256(generated):
            raise AgenticBenchmarkError(f"record[{record_index}] generated token hash mismatch")
        if output.get("raw_markup_leaked") is not False:
            raise AgenticBenchmarkError(f"record[{record_index}] leaked raw model markup")

        expected_turn = turns[turn_index]
        tool = _mapping(record.get("tool"), label=f"record[{record_index}].tool")
        expected_name = str(expected_turn["expected_tool"])
        if tool.get("expected_name") != expected_name or tool.get("name") != expected_name:
            raise AgenticBenchmarkError(
                f"record[{record_index}].tool.name does not match expected tool"
            )
        if tool.get("declared_schema_sha256") != suite.tool_schema_sha256(expected_name):
            raise AgenticBenchmarkError(
                f"record[{record_index}].tool declared schema hash mismatch"
            )
        _nonempty_string(tool.get("call_id"), label=f"record[{record_index}].tool.call_id")
        arguments = _mapping(tool.get("arguments"), label=f"record[{record_index}].tool.arguments")
        if arguments != expected_turn["expected_arguments"] and lane == "deterministic":
            raise AgenticBenchmarkError(
                f"record[{record_index}].tool.arguments differ from deterministic fixture"
            )
        schema = _mapping(
            suite.tools[expected_name]["parameters"], label=f"tool {expected_name} parameters"
        )
        _validate_fixture_arguments(
            arguments, schema, label=f"record[{record_index}].tool.arguments"
        )
        for field in ("arguments_json_valid", "schema_valid", "result_linked"):
            if tool.get(field) is not True:
                raise AgenticBenchmarkError(f"record[{record_index}].tool.{field} must be true")

        timing = _mapping(record.get("timing"), label=f"record[{record_index}].timing")
        submitted = _required_time(timing, "submitted_at_s", record_index=record_index)
        first = _required_time(timing, "first_token_at_s", record_index=record_index)
        ready = _required_time(timing, "tool_call_ready_at_s", record_index=record_index)
        done = _required_time(timing, "response_done_at_s", record_index=record_index)
        result_submit = _required_time(
            timing, "tool_result_submitted_at_s", record_index=record_index
        )
        timing_mode = timing.get("token_timing_mode")
        if timing_mode not in _TOKEN_TIMING_MODES:
            raise AgenticBenchmarkError(
                f"record[{record_index}].timing.token_timing_mode must be one of "
                f"{sorted(_TOKEN_TIMING_MODES)}"
            )
        observed = timing.get("token_observed_at_s")
        event_counts = timing.get("token_event_token_counts")
        if not _is_sequence(observed) or not observed:
            raise AgenticBenchmarkError(
                f"record[{record_index}].timing.token_observed_at_s must be non-empty"
            )
        if not _is_sequence(event_counts) or len(event_counts) != len(observed):
            raise AgenticBenchmarkError(
                f"record[{record_index}].timing.token_event_token_counts must match token events"
            )
        observed_times = [float(value) for value in observed if _is_finite_number(value)]
        if len(observed_times) != len(observed):
            raise AgenticBenchmarkError(
                f"record[{record_index}].timing.token_observed_at_s must be finite"
            )
        normalized_event_counts = [
            _positive_int(
                value,
                label=f"record[{record_index}].timing.token_event_token_counts[{event_index}]",
            )
            for event_index, value in enumerate(event_counts)
        ]
        if sum(normalized_event_counts) != len(generated):
            raise AgenticBenchmarkError(
                f"record[{record_index}] token event counts do not match generated IDs"
            )
        if timing_mode == "live_exact" and any(count != 1 for count in normalized_event_counts):
            raise AgenticBenchmarkError(
                f"record[{record_index}] live_exact timing requires one token per event"
            )
        ordered = [submitted, first, *observed_times, ready, done, result_submit]
        if any(right < left for left, right in zip(ordered, ordered[1:])):
            raise AgenticBenchmarkError(f"record[{record_index}] timing is not monotonic")
        if not math.isclose(first, observed_times[0], rel_tol=0.0, abs_tol=1e-9):
            raise AgenticBenchmarkError(
                f"record[{record_index}].timing.first_token_at_s does not match first token"
            )

        backend = _mapping(record.get("backend"), label=f"record[{record_index}].backend")
        _nonempty_string(backend.get("batch_id"), label=f"record[{record_index}].backend.batch_id")
        if backend.get("timing_scope") not in {"choice", "batch", "request"}:
            raise AgenticBenchmarkError(
                f"record[{record_index}].backend.timing_scope is unsupported"
            )
        if not isinstance(backend.get("timing_owner"), bool):
            raise AgenticBenchmarkError(
                f"record[{record_index}].backend.timing_owner must be boolean"
            )
        _nonempty_string(
            backend.get("sampler_mode"), label=f"record[{record_index}].backend.sampler_mode"
        )
        _nonnegative_int(
            backend.get("logits_d2h_bytes"),
            label=f"record[{record_index}].backend.logits_d2h_bytes",
        )
        _positive_int(
            backend.get("physical_width"),
            label=f"record[{record_index}].backend.physical_width",
        )
        if not isinstance(backend.get("serial_fallback"), bool):
            raise AgenticBenchmarkError(
                f"record[{record_index}].backend.serial_fallback must be boolean"
            )

        prefix = _mapping(record.get("prefix"), label=f"record[{record_index}].prefix")
        if not isinstance(prefix.get("lookup"), bool) or not isinstance(prefix.get("hit"), bool):
            raise AgenticBenchmarkError(f"record[{record_index}].prefix lookup/hit must be boolean")
        reused_tokens = _nonnegative_int(
            prefix.get("reused_tokens"), label=f"record[{record_index}].prefix.reused_tokens"
        )
        _nonnegative_int(
            prefix.get("cache_bytes"), label=f"record[{record_index}].prefix.cache_bytes"
        )
        if prefix.get("hit") and (not prefix.get("lookup") or reused_tokens == 0):
            raise AgenticBenchmarkError(
                f"record[{record_index}].prefix hit requires lookup and reused tokens"
            )
        if not prefix.get("hit") and reused_tokens != 0:
            raise AgenticBenchmarkError(
                f"record[{record_index}].prefix reused tokens require a hit"
            )
        if configuration["cache_mode"] == "off" and (
            prefix.get("lookup") or prefix.get("hit") or reused_tokens or int(prefix["cache_bytes"])
        ):
            raise AgenticBenchmarkError(
                f"record[{record_index}].prefix activity is incompatible with cache_mode=off"
            )

        finish = _mapping(record.get("finish"), label=f"record[{record_index}].finish")
        if finish.get("reason") != "tool_calls":
            raise AgenticBenchmarkError(f"record[{record_index}].finish.reason must be tool_calls")
        _nonnegative_int(
            finish.get("retry_count"), label=f"record[{record_index}].finish.retry_count"
        )
        normalized.append(copy.deepcopy(dict(record)))

    agents_by_run: dict[str, set[str]] = {}
    for run_id, agent_id in agents:
        agents_by_run.setdefault(run_id, set()).add(agent_id)
    invalid_runs = {
        run_id: len(run_agents)
        for run_id, run_agents in agents_by_run.items()
        if len(run_agents) != concurrency
    }
    if invalid_runs:
        raise AgenticBenchmarkError(
            f"configuration.concurrency={concurrency} but per-run agent counts are {invalid_runs}"
        )
    if configuration.get("require_complete_workloads") is True:
        for agent_key, turn_indexes in agent_turns.items():
            workload_id = agent_workloads[agent_key]
            expected = list(range(len(suite.workloads[workload_id]["turns"])))
            if sorted(turn_indexes) != expected:
                raise AgenticBenchmarkError(
                    f"agent {agent_key[0]}/{agent_key[1]} does not contain every workload turn"
                )

    batches: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in normalized:
        backend = _mapping(record["backend"], label="backend")
        key = (str(record["run_id"]), str(backend["batch_id"]))
        batches.setdefault(key, []).append(backend)
    for (run_id, batch_id), rows in sorted(batches.items()):
        owners = sum(1 for row in rows if row["timing_owner"] is True)
        if owners != 1:
            raise AgenticBenchmarkError(
                f"batch {run_id}/{batch_id} has {owners} timing owners; expected 1"
            )

    ownership = _validate_final_ownership(records_payload.get("final_ownership"))
    return configuration, normalized, ownership


def _rollup(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    ttft_ms: list[float] = []
    ready_ms: list[float] = []
    complete_ms: list[float] = []
    inter_token_ms: list[float] = []
    submitted_times: list[float] = []
    completed_times: list[float] = []
    generated_tokens = 0
    lookups = 0
    hits = 0
    reused_tokens = 0
    max_cache_bytes = 0
    logits_d2h_bytes = 0
    physical_width_turns: dict[str, int] = {}
    sampler_modes: dict[str, int] = {}
    serial_fallback_turns = 0
    token_timing_modes: dict[str, int] = {}
    generated_id_sources: dict[str, int] = {}
    agents: set[tuple[str, str]] = set()
    workloads: set[str] = set()
    batches: set[tuple[str, str]] = set()

    for record in records:
        timing = _mapping(record["timing"], label="timing")
        submitted = float(timing["submitted_at_s"])
        first = float(timing["first_token_at_s"])
        ready = float(timing["tool_call_ready_at_s"])
        complete = float(timing["tool_result_submitted_at_s"])
        observed = [float(value) for value in timing["token_observed_at_s"]]
        timing_mode = str(timing["token_timing_mode"])
        token_timing_modes[timing_mode] = token_timing_modes.get(timing_mode, 0) + 1
        ttft_ms.append(1000.0 * (first - submitted))
        ready_ms.append(1000.0 * (ready - submitted))
        complete_ms.append(1000.0 * (complete - submitted))
        if timing_mode == "live_exact":
            inter_token_ms.extend(
                1000.0 * (right - left) for left, right in zip(observed, observed[1:])
            )
        submitted_times.append(submitted)
        completed_times.append(complete)
        output = _mapping(record["output"], label="output")
        generated_tokens += len(output["generated_token_ids"])
        id_source = str(output["generated_token_ids_source"])
        generated_id_sources[id_source] = generated_id_sources.get(id_source, 0) + 1
        agents.add((str(record["run_id"]), str(record["agent_id"])))
        workloads.add(str(record["workload_id"]))

        prefix = _mapping(record["prefix"], label="prefix")
        lookups += int(prefix["lookup"])
        hits += int(prefix["hit"])
        reused_tokens += int(prefix["reused_tokens"])
        max_cache_bytes = max(max_cache_bytes, int(prefix["cache_bytes"]))

        backend = _mapping(record["backend"], label="backend")
        batches.add((str(record["run_id"]), str(backend["batch_id"])))
        logits_d2h_bytes += int(backend["logits_d2h_bytes"])
        width = str(int(backend["physical_width"]))
        physical_width_turns[width] = physical_width_turns.get(width, 0) + 1
        sampler = str(backend["sampler_mode"])
        sampler_modes[sampler] = sampler_modes.get(sampler, 0) + 1
        serial_fallback_turns += int(backend["serial_fallback"])

    workload_wall = max(completed_times) - min(submitted_times)
    if workload_wall <= 0.0:
        raise AgenticBenchmarkError("workload wall must be positive")
    run_agents: dict[str, set[str]] = {}
    for run_id, agent_id in agents:
        run_agents.setdefault(run_id, set()).add(agent_id)
    coverage = {
        "workloads": sorted(workloads),
        "runs": len(run_agents),
        "concurrency": max((len(items) for items in run_agents.values()), default=0),
        "agents": len(agents),
        "turns": len(records),
        "tool_calls": len(records),
        "generated_tokens": generated_tokens,
        "batches": len(batches),
    }
    rollup = {
        "latency_ms": {
            "ttft": _latency_summary(ttft_ms),
            "tool_call_ready": _latency_summary(ready_ms),
            "inter_token": _latency_summary(inter_token_ms),
            "complete_turn": _latency_summary(complete_ms),
        },
        "workload_wall_s": workload_wall,
        "exact_generated_tok_s": generated_tokens / workload_wall,
        "validated_tool_calls_s": len(records) / workload_wall,
        "prefix": {
            "lookups": lookups,
            "hits": hits,
            "hit_rate": (hits / lookups) if lookups else 0.0,
            "reused_tokens": reused_tokens,
            "max_cache_bytes": max_cache_bytes,
        },
        "backend": {
            "sampler_mode_turns": dict(sorted(sampler_modes.items())),
            "full_vocab_logits_d2h_bytes": logits_d2h_bytes,
            "physical_width_turns": dict(
                sorted(physical_width_turns.items(), key=lambda item: int(item[0]))
            ),
            "serial_fallback_turns": serial_fallback_turns,
            "token_timing_mode_turns": dict(sorted(token_timing_modes.items())),
            "generated_token_id_source_turns": dict(sorted(generated_id_sources.items())),
        },
    }
    return coverage, rollup


def validate_agentic_benchmark_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact artifact envelope emitted by this module."""

    root = _mapping(payload, label="agentic benchmark artifact")
    if root.get("kind") != AGENTIC_ARTIFACT_KIND:
        raise AgenticBenchmarkError("agentic benchmark artifact kind is invalid")
    if root.get("schema_version") != AGENTIC_SCHEMA_VERSION:
        raise AgenticBenchmarkError("agentic benchmark artifact schema_version is invalid")
    if root.get("performance_claim") not in {True, False}:
        raise AgenticBenchmarkError("agentic benchmark artifact performance_claim must be boolean")
    validation = _mapping(root.get("validation"), label="validation")
    if validation.get("passed") is not True or validation.get("failure_reasons") != []:
        raise AgenticBenchmarkError("agentic benchmark artifact validation did not pass")
    _mapping(root.get("workload_suite"), label="workload_suite")
    _mapping(root.get("configuration"), label="configuration")
    coverage = _mapping(root.get("coverage"), label="coverage")
    _positive_int(coverage.get("turns"), label="coverage.turns")
    _positive_int(coverage.get("tool_calls"), label="coverage.tool_calls")
    _positive_int(coverage.get("generated_tokens"), label="coverage.generated_tokens")
    _mapping(root.get("rollup"), label="rollup")
    _validate_final_ownership(root.get("final_ownership"))
    records = root.get("turn_records")
    if not _is_sequence(records) or len(records) != int(coverage["turns"]):
        raise AgenticBenchmarkError("turn_records do not match coverage.turns")
    record_hash = _sha256_string(root.get("turn_records_sha256"), label="turn_records_sha256")
    if record_hash != _canonical_sha256(records):
        raise AgenticBenchmarkError("turn_records_sha256 does not match turn_records")
    generated_tokens = 0
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, label=f"turn_records[{index}]")
        output = _mapping(record.get("output"), label=f"turn_records[{index}].output")
        generated_tokens += len(
            _token_ids(
                output.get("generated_token_ids"),
                label=f"turn_records[{index}].output.generated_token_ids",
            )
        )
    if generated_tokens != int(coverage["generated_tokens"]):
        raise AgenticBenchmarkError("turn_records do not match coverage.generated_tokens")
    return {"passed": True, "failure_reasons": []}


def build_agentic_benchmark_artifact(
    suite: AgenticWorkloadSuite,
    records_payload: Mapping[str, Any],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Validate normalized turn records and build their compact A0 artifact."""

    configuration, records, ownership = _validate_turn_records(suite, records_payload)
    coverage, rollup = _rollup(records)
    artifact = {
        "kind": AGENTIC_ARTIFACT_KIND,
        "schema_version": AGENTIC_SCHEMA_VERSION,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "performance_claim": bool(configuration.get("performance_claim", False)),
        "workload_suite": suite.identity(),
        "configuration": copy.deepcopy(configuration),
        "coverage": coverage,
        "validation": {"passed": True, "failure_reasons": []},
        "rollup": rollup,
        "final_ownership": ownership,
        "turn_records_sha256": _canonical_sha256(records),
        "turn_records": records,
    }
    validate_agentic_benchmark_artifact(artifact)
    return artifact


__all__ = [
    "AGENTIC_ARTIFACT_KIND",
    "AGENTIC_RECORDS_KIND",
    "AGENTIC_SCHEMA_VERSION",
    "AGENTIC_WORKLOAD_KIND",
    "DEFAULT_AGENTIC_WORKLOADS",
    "AgenticBenchmarkError",
    "AgenticWorkloadSuite",
    "build_agentic_benchmark_artifact",
    "load_agentic_workload_suite",
    "percentile",
    "validate_agentic_benchmark_artifact",
]
