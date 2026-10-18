"""Session-replay benchmark fixtures built from Jouzu agent session transcripts.

A Jouzu session ``.jsonl`` is an append-only record log whose live conversation
is the ``parentId`` ancestry chain of its tip, not the file order (retries and
regenerated turns are siblings on dead branches).  The converter walks that
ancestry, drops dead branches and post-compaction stubs, skips assistant
thinking blocks (clients never resend them), and emits the OpenAI-style
message shapes the chat renderer accepts — the same shapes
``build_canonical_turn_messages`` builds for the synthetic coding lanes.

The resend contract mirrors a real agentic client: every request resends the
whole transcript, which grows by one assistant round plus its tool results,
or by one user message, between consecutive requests.  ``request_ends`` marks
exclusive entry indices where each request's prompt ends; the final assistant
reply in a slice is the target of the last request and is never resent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SESSION_REPLAY_KIND = "hipengine.session_replay_workloads"
SESSION_REPLAY_SCHEMA_VERSION = 1

_TRUNCATION_MARKER = "…[truncated]"
_INTERRUPT_RESULT = "[tool result unavailable: interrupted before the tool returned]"

_DEFAULT_SYSTEM_POLICY = (
    "You are a coding agent working inside a repository. Read the relevant files "
    "before editing them, keep changes scoped to one logical unit, and run the "
    "narrowest relevant check before claiming a task is done. Use the provided "
    "tools to inspect and modify the repository rather than guessing. Report "
    "concrete results with file paths, and distinguish measured results from "
    "inference. When a request is ambiguous, state your reading of it and proceed "
    "with the smallest coherent change."
)


class SessionReplayError(ValueError):
    """Raised for malformed sessions or invalid replay fixtures."""


# ---------------------------------------------------------------------------
# Converter: session .jsonl -> replay fixture payload
# ---------------------------------------------------------------------------


def _record_blocks(message: Mapping[str, Any], block_type: str) -> list[Mapping[str, Any]]:
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return []
    return [
        block
        for block in content
        if isinstance(block, Mapping) and block.get("type") == block_type
    ]


def _join_block_text(blocks: Sequence[Mapping[str, Any]]) -> str:
    parts = [str(block.get("text", "")) for block in blocks if block.get("text")]
    return "\n".join(parts)


def _is_content_bearing_assistant(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    if any(str(block.get("text", "")) for block in _record_blocks(message, "text")):
        return True
    return bool(_record_blocks(message, "toolCall"))


def _walk_active_chain(
    records: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    tip_id: str,
) -> list[Mapping[str, Any]]:
    """Return message records from the session root to ``tip_id`` inclusive.

    The walk passes through non-message records (flow-branch markers, model
    and thinking-level changes) because they sit inside the ancestry, and
    refuses to cross a compaction record: a post-compaction client resend
    starts from the compaction summary, not the raw history.
    """

    chain: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    current: Mapping[str, Any] | None = by_id.get(tip_id)
    while current is not None:
        record_id = str(current.get("id"))
        if record_id in seen:
            raise SessionReplayError(f"ancestry cycle at record {record_id!r}")
        seen.add(record_id)
        record_type = str(current.get("type"))
        if record_type == "compaction":
            raise SessionReplayError(
                "tip ancestry crosses a compaction record; pick a pre-compaction tip"
            )
        if record_type == "session":
            break
        if record_type == "message":
            chain.append(current)
        parent_id = current.get("parentId")
        if parent_id is not None and parent_id not in by_id:
            raise SessionReplayError(
                f"record {record_id!r} references missing parent {parent_id!r}"
            )
        current = by_id.get(parent_id) if parent_id is not None else None
    chain.reverse()
    if not chain:
        raise SessionReplayError(f"tip {tip_id!r} has no message ancestry")
    return chain


def _default_tip(records: Sequence[Mapping[str, Any]]) -> str:
    """Use the compaction parent's live chain, or the last assistant without one.

    When the session was compacted, the compaction record's own ``parentId``
    is the definitive live tip at that moment; dead retry branches appended
    after it are ignored.  Without a compaction record, the last
    content-bearing assistant in file order is the best available tip.
    """

    by_id = {str(record.get("id")): record for record in records if record.get("id")}
    for record in records:
        if record.get("type") == "compaction":
            parent_id = record.get("parentId")
            if not isinstance(parent_id, str) or parent_id not in by_id:
                raise SessionReplayError("compaction references a missing parent")
            chain = _walk_active_chain(records, by_id, parent_id)
            for ancestor in reversed(chain):
                message = ancestor["message"]
                if _is_content_bearing_assistant(message) or message.get("role") in {
                    "user", "toolResult"
                }:
                    return str(ancestor["id"])
            raise SessionReplayError("compaction ancestry has no replayable messages")
    for record in reversed(records):
        if record.get("type") == "message" and _is_content_bearing_assistant(
            record["message"]
        ):
            return str(record["id"])
    raise SessionReplayError("session has no content-bearing assistant message")


def _emit_entries(
    chain: Sequence[Mapping[str, Any]],
    *,
    max_tool_result_chars: int | None,
) -> tuple[list[dict[str, Any]], list[int], dict[str, int]]:
    entries: list[dict[str, Any]] = []
    request_ends: list[int] = []
    stats = {
        "skipped_empty_messages": 0,
        "synthesized_interrupt_results": 0,
        "truncated_tool_results": 0,
    }
    open_calls: dict[str, dict[str, Any]] = {}

    for record in chain:
        message = record["message"]
        role = message.get("role")

        if role == "user":
            text = _join_block_text(_record_blocks(message, "text"))
            if not text:
                stats["skipped_empty_messages"] += 1
                continue
            if open_calls:
                for call_id, call in open_calls.items():
                    entries.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _INTERRUPT_RESULT,
                        }
                    )
                    stats["synthesized_interrupt_results"] += 1
                open_calls = {}
            entries.append({"role": "user", "content": text})
            request_ends.append(len(entries))
            continue

        if role == "assistant":
            text = _join_block_text(_record_blocks(message, "text"))
            calls = []
            for block in _record_blocks(message, "toolCall"):
                arguments = block.get("arguments")
                if not isinstance(arguments, Mapping):
                    arguments = {"value": arguments}
                calls.append(
                    {
                        "id": str(block["id"]),
                        "type": "function",
                        "function": {
                            "name": str(block["name"]),
                            "arguments": json.dumps(
                                dict(arguments),
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            if not text and not calls:
                stats["skipped_empty_messages"] += 1
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = calls
                for call in calls:
                    if call["id"] in open_calls:
                        raise SessionReplayError(
                            f"duplicate tool-call id {call['id']!r} in one round"
                        )
                    open_calls[call["id"]] = call
            entries.append(entry)
            continue

        if role == "toolResult":
            call_id = message.get("toolCallId")
            if call_id is None:
                stats["skipped_empty_messages"] += 1
                continue
            call_id = str(call_id)
            if call_id not in open_calls:
                raise SessionReplayError(
                    f"tool result {record.get('id')!r} references unknown or "
                    f"already-closed tool call {call_id!r}"
                )
            del open_calls[call_id]
            text = _join_block_text(_record_blocks(message, "text"))
            if (
                max_tool_result_chars is not None
                and len(text) > int(max_tool_result_chars)
            ):
                text = text[: int(max_tool_result_chars)] + _TRUNCATION_MARKER
                stats["truncated_tool_results"] += 1
            entries.append(
                {"role": "tool", "tool_call_id": call_id, "content": text}
            )
            if not open_calls:
                request_ends.append(len(entries))
            continue

        stats["skipped_empty_messages"] += 1

    if open_calls:
        dangling = ", ".join(sorted(open_calls))
        raise SessionReplayError(
            f"tip round has unpaired tool calls ({dangling}); choose an earlier tip"
        )
    return entries, request_ends, stats


def _apply_tail_rounds(
    entries: Sequence[Mapping[str, Any]],
    request_ends: Sequence[int],
    *,
    tail_rounds: int,
) -> tuple[list[dict[str, Any]], list[int], str]:
    if int(tail_rounds) < 1:
        raise SessionReplayError("tail_rounds must be a positive integer")
    tail_rounds = int(tail_rounds)
    if tail_rounds >= len(request_ends):
        start = 0
        reason = "conversation_start"
    else:
        start = int(request_ends[-tail_rounds - 1])
        first = entries[start]
        reason = "user_message" if first.get("role") == "user" else "tail_rounds"
    kept_ends = [end - start for end in request_ends[len(request_ends) - tail_rounds :]]
    kept_entries = [dict(entry) for entry in entries[start:]]
    return kept_entries, kept_ends, reason


def _stub_tools(names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": (
                f"Tool {name!r} invoked during the replayed agent session "
                "(stub schema; the recorded session does not store tool schemas)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
        for name in sorted(set(str(name) for name in names))
    ]


def convert_session_jsonl(
    session_path: str | Path,
    *,
    workload_id: str = "session-tail",
    tip: str | None = None,
    tail_rounds: int | None = None,
    max_tool_result_chars: int | None = None,
    suite: str | None = None,
    description: str | None = None,
    system_policy: str | None = None,
) -> dict[str, Any]:
    """Convert one Jouzu session transcript into a replay-fixture payload."""

    path = Path(session_path)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionReplayError(
                    f"{path}:{line_number} is not valid JSON: {exc}"
                ) from exc
            records.append(record)

    by_id: dict[str, Mapping[str, Any]] = {}
    session_id = ""
    for record in records:
        record_id = record.get("id")
        if record_id is None:
            continue
        record_id = str(record_id)
        if record_id in by_id:
            raise SessionReplayError(f"duplicate record id {record_id!r}")
        by_id[record_id] = record
        if record.get("type") == "session":
            session_id = record_id

    if not session_id:
        raise SessionReplayError("session file has no session header record")

    tip_id = str(tip) if tip is not None else _default_tip(records)
    if tip_id not in by_id:
        raise SessionReplayError(f"tip record {tip_id!r} not found in {path}")
    if by_id[tip_id].get("type") != "message":
        raise SessionReplayError(f"tip record {tip_id!r} is not a message record")

    chain = _walk_active_chain(records, by_id, tip_id)
    entries, request_ends, stats = _emit_entries(
        chain, max_tool_result_chars=max_tool_result_chars
    )
    if not request_ends:
        raise SessionReplayError("converted chain contains no client requests")

    slice_info: dict[str, Any] = {
        "tail_rounds": tail_rounds,
        "tip_record": tip_id,
        "start_reason": "conversation_start",
    }
    if tail_rounds is not None:
        entries, request_ends, reason = _apply_tail_rounds(
            entries, request_ends, tail_rounds=tail_rounds
        )
        slice_info["start_reason"] = reason

    message_records = [r for r in records if r.get("type") == "message"]
    slice_info.update(stats)
    slice_info["chain_messages"] = len(chain)
    slice_info["excluded_off_chain_messages"] = len(message_records) - len(chain)

    tool_names = [
        call["function"]["name"]
        for entry in entries
        for call in entry.get("tool_calls", [])
    ]

    return {
        "kind": SESSION_REPLAY_KIND,
        "schema_version": SESSION_REPLAY_SCHEMA_VERSION,
        "suite": str(suite) if suite is not None else f"jouzu-session-{session_id[:8]}",
        "description": (
            str(description)
            if description is not None
            else (
                f"Replay slice of Jouzu agent session {session_id}: "
                f"{len(entries)} transcript entries, {len(request_ends)} client requests."
            )
        ),
        "source": {
            "record": {
                "kind": "jouzu-session-jsonl",
                "session_id": session_id,
                "path": str(path),
                "tip_record": tip_id,
            },
            "notes": [
                "The session log does not record the client system prompt or tool "
                "schemas; a representative coding-agent policy and stub tool "
                "schemas are substituted for token-count fidelity.",
                "Assistant thinking blocks are not replayed; agentic clients do "
                "not resend them.",
                "File order is not conversation order: entries follow the "
                "parentId ancestry of the tip, so dead retry branches are excluded.",
            ],
        },
        "tools": _stub_tools(tool_names),
        "workloads": [
            {
                "id": str(workload_id),
                "system_prompt": (
                    str(system_policy)
                    if system_policy is not None
                    else _DEFAULT_SYSTEM_POLICY
                ),
                "entries": entries,
                "request_ends": request_ends,
                "slice": slice_info,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Fixture loading and replay-turn construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionReplayFixture:
    path: Path
    file_sha256: str
    payload: Mapping[str, Any]
    suite: str
    description: str
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    workloads: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": SESSION_REPLAY_KIND,
            "schema_version": SESSION_REPLAY_SCHEMA_VERSION,
            "suite": self.suite,
            "file_sha256": self.file_sha256,
            "workload_sha256": {
                workload_id: hashlib.sha256(
                    json.dumps(workload, sort_keys=True).encode("utf-8")
                ).hexdigest()
                for workload_id, workload in sorted(self.workloads.items())
            },
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SessionReplayError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SessionReplayError(f"{label} must be a non-empty string")
    return value


def _validate_workload(raw: Any, *, label: str) -> tuple[str, Mapping[str, Any]]:
    workload = _mapping(raw, label=label)
    workload_id = _nonempty_string(workload.get("id"), label=f"{label}.id")
    _nonempty_string(
        workload.get("system_prompt"), label=f"{label}.system_prompt"
    )
    raw_entries = workload.get("entries")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise SessionReplayError(f"{label}.entries must be an array")
    entries: list[dict[str, Any]] = []
    open_calls: dict[str, str] = {}
    valid_request_ends: set[int] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry_label = f"{label}.entries[{index}]"
        entry = dict(_mapping(raw_entry, label=entry_label))
        role = entry.get("role")
        if role == "user":
            _nonempty_string(entry.get("content"), label=f"{entry_label}.content")
            if open_calls:
                raise SessionReplayError(
                    f"{entry_label} interrupts a round with pending tool calls"
                )
        elif role == "assistant":
            if not isinstance(entry.get("content"), str):
                raise SessionReplayError(f"{entry_label}.content must be a string")
            calls = entry.get("tool_calls", [])
            if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                raise SessionReplayError(f"{entry_label}.tool_calls must be an array")
            for call_index, raw_call in enumerate(calls):
                call_label = f"{entry_label}.tool_calls[{call_index}]"
                call = _mapping(raw_call, label=call_label)
                envelope = _mapping(call.get("function"), label=f"{call_label}.function")
                call_id = _nonempty_string(call.get("id"), label=f"{call_label}.id")
                if call.get("type") != "function":
                    raise SessionReplayError(f"{call_label}.type must be 'function'")
                _nonempty_string(
                    envelope.get("name"), label=f"{call_label}.function.name"
                )
                arguments = envelope.get("arguments")
                if not isinstance(arguments, str) or not arguments:
                    raise SessionReplayError(
                        f"{call_label}.function.arguments must be a JSON string"
                    )
                if call_id in open_calls:
                    raise SessionReplayError(f"duplicate tool-call id {call_id!r}")
                open_calls[call_id] = str(envelope["name"])
            if not entry["content"] and not calls:
                raise SessionReplayError(f"{entry_label} is empty")
        elif role == "tool":
            call_id = _nonempty_string(
                entry.get("tool_call_id"), label=f"{entry_label}.tool_call_id"
            )
            if call_id not in open_calls:
                raise SessionReplayError(
                    f"{entry_label} references unknown or already-closed "
                    f"tool call {call_id!r}"
                )
            del open_calls[call_id]
            _nonempty_string(entry.get("content"), label=f"{entry_label}.content")
        else:
            raise SessionReplayError(f"{entry_label} has unsupported role {role!r}")
        entries.append(entry)
        if role in {"user", "tool"} and not open_calls:
            valid_request_ends.add(len(entries))

    raw_ends = workload.get("request_ends")
    if not isinstance(raw_ends, Sequence) or isinstance(raw_ends, (str, bytes)):
        raise SessionReplayError(f"{label}.request_ends must be an array")
    previous_end = 0
    for index, raw_end in enumerate(raw_ends):
        end_label = f"{label}.request_ends[{index}]"
        if not isinstance(raw_end, int) or isinstance(raw_end, bool):
            raise SessionReplayError(f"{end_label} must be an integer")
        if raw_end <= previous_end:
            raise SessionReplayError(
                f"{end_label} must be strictly increasing"
            )
        if raw_end > len(entries):
            raise SessionReplayError(f"{end_label} exceeds entries length")
        if raw_end not in valid_request_ends:
            raise SessionReplayError(
                f"{end_label} must end a user message or completed tool round"
            )
        previous_end = raw_end
    if not raw_ends:
        raise SessionReplayError(f"{label}.request_ends must not be empty")
    if open_calls:
        dangling = ", ".join(sorted(open_calls))
        raise SessionReplayError(f"{label} ends with pending tool calls ({dangling})")

    normalized = dict(workload)
    normalized["entries"] = entries
    normalized["request_ends"] = [int(end) for end in raw_ends]
    return workload_id, normalized


def load_session_replay_fixture(path: str | Path) -> SessionReplayFixture:
    """Load and validate a session-replay fixture JSON file."""

    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionReplayError(f"{fixture_path} is not valid JSON: {exc}") from exc
    root = _mapping(payload, label=str(fixture_path))
    if root.get("kind") != SESSION_REPLAY_KIND:
        raise SessionReplayError(f"{fixture_path} has unsupported workload kind")
    if root.get("schema_version") != SESSION_REPLAY_SCHEMA_VERSION:
        raise SessionReplayError(f"{fixture_path} has unsupported schema_version")
    suite = _nonempty_string(root.get("suite"), label="suite")
    _nonempty_string(root.get("description"), label="description")

    raw_tools = root.get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise SessionReplayError("tools must be an array")
    tools: dict[str, dict[str, Any]] = {}
    for index, raw_tool in enumerate(raw_tools):
        tool = dict(_mapping(raw_tool, label=f"tools[{index}]"))
        name = _nonempty_string(tool.get("name"), label=f"tools[{index}].name")
        _nonempty_string(
            tool.get("description"), label=f"tools[{index}].description"
        )
        _mapping(
            tool.get("parameters"), label=f"tools[{index}].parameters"
        )
        tools[name] = tool

    raw_workloads = root.get("workloads")
    if not isinstance(raw_workloads, Sequence) or isinstance(raw_workloads, (str, bytes)):
        raise SessionReplayError("workloads must be an array")
    workloads: dict[str, Mapping[str, Any]] = {}
    for index, raw_workload in enumerate(raw_workloads):
        workload_id, workload = _validate_workload(
            raw_workload, label=f"workloads[{index}]"
        )
        if workload_id in workloads:
            raise SessionReplayError(f"duplicate workload id {workload_id!r}")
        workloads[workload_id] = workload

    return SessionReplayFixture(
        path=fixture_path,
        file_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        payload=root,
        suite=suite,
        description=str(root["description"]),
        tools=tools,
        workloads=workloads,
    )


def _require_workload(
    fixture: SessionReplayFixture, workload_id: str
) -> Mapping[str, Any]:
    if workload_id not in fixture.workloads:
        raise SessionReplayError(f"unknown session-replay workload {workload_id!r}")
    return fixture.workloads[workload_id]


def session_replay_turn_count(
    fixture: SessionReplayFixture, workload_id: str
) -> int:
    """Number of client requests (resends) in one replay workload."""

    workload = _require_workload(fixture, workload_id)
    return len(workload["request_ends"])


def build_session_replay_turn_messages(
    fixture: SessionReplayFixture,
    workload_id: str,
    *,
    turn_index: int,
    system_policy: str | None = None,
) -> list[dict[str, Any]]:
    """Build one request's cumulative resend, mirroring the canonical lanes."""

    workload = _require_workload(fixture, workload_id)
    ends = workload["request_ends"]
    if turn_index < 0 or turn_index >= len(ends):
        raise SessionReplayError("turn_index is out of range")
    system = (
        str(system_policy)
        if system_policy is not None
        else str(workload["system_prompt"])
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(
        dict(entry) for entry in workload["entries"][: int(ends[turn_index])]
    )
    return messages


def build_session_replay_tools(fixture: SessionReplayFixture) -> list[dict[str, Any]]:
    """Translate the fixture tool declarations into OpenAI function envelopes."""

    return [
        {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool["description"]),
                "strict": True,
                "parameters": dict(tool["parameters"]),
            },
        }
        for _, tool in sorted(fixture.tools.items())
    ]
