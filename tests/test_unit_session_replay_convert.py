"""Unit tests for the Jouzu session-replay converter and fixture loader.

The converter walks a session ``.jsonl`` by its ``parentId`` ancestry (file
order is not chain order), drops dead retry branches, skips assistant
thinking blocks, and emits the exact OpenAI-style message shapes the chat
renderer accepts (the same shapes ``build_canonical_turn_messages`` builds).
Requests are client resends: the transcript grows by one assistant round
(plus its tool results) or by one user message between consecutive requests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hipengine.benchmark.session_replay import (
    SESSION_REPLAY_KIND,
    SessionReplayError,
    build_session_replay_tools,
    build_session_replay_turn_messages,
    convert_session_jsonl,
    load_session_replay_fixture,
    session_replay_turn_count,
)


# ---------------------------------------------------------------------------
# Synthetic session builder
# ---------------------------------------------------------------------------


def _msg(record_id: str, parent: str | None, role: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": record_id,
        "parentId": parent,
        "timestamp": "2026-09-19T06:44:50.000Z",
        "type": "message",
        "message": {"role": role, "content": blocks, "timestamp": "2026-09-19T06:44:50.000Z"},
    }


def _text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def _thinking(value: str) -> dict[str, Any]:
    return {"type": "thinking", "thinking": value}


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"type": "toolCall", "id": call_id, "name": name, "arguments": arguments}


def _result(call_id: str, name: str, text: str) -> dict[str, Any]:
    return {
        "role": "toolResult",
        "toolCallId": call_id,
        "toolName": name,
        "isError": False,
        "details": None,
        "content": [{"type": "text", "text": text}],
        "timestamp": "2026-09-19T06:44:50.000Z",
    }


def _result_msg(
    record_id: str, parent: str, call_id: str, name: str, text: str
) -> dict[str, Any]:
    return {
        "id": record_id,
        "parentId": parent,
        "timestamp": "2026-09-19T06:44:50.000Z",
        "type": "message",
        "message": _result(call_id, name, text),
    }


def _write_session(path: Path, records: list[dict[str, Any]]) -> Path:
    payload = [
        {
            "id": "sess",
            "type": "session",
            "cwd": "/tmp",
            "timestamp": "2026-09-19T06:44:50.000Z",
            "version": 1,
        }
    ]
    payload.extend(records)
    path.write_text("\n".join(json.dumps(record) for record in payload) + "\n", encoding="utf-8")
    return path


def _session_records() -> list[dict[str, Any]]:
    """A session with a dead retry branch, mid-chain flow markers, and a
    corrupted post-compaction tail.  File order deliberately interleaves the
    dead branch after the live chain to prove the walk is by ancestry."""

    live = [
        _msg("u1", None, "user", [_text("Fix the scheduler admission bug.")]),
        # Round 1: thinking is skipped, one tool call.
        _msg(
            "a1",
            "u1",
            "assistant",
            [_thinking("Let me inspect the file first."), _call("c1", "bash", {"command": "grep admit src/"})],
        ),
        _result_msg("r1", "a1", "c1", "bash", "src/scheduler.py:41:def admit"),
        # Round 2: multi tool-call round.
        _msg(
            "a2",
            "r1",
            "assistant",
            [_call("c2", "read", {"path": "src/scheduler.py"}), _call("c3", "bash", {"command": "pytest tests/"})],
        ),
        _result_msg("r2", "a2", "c2", "read", "def admit(self, request): return ok"),
        # Sequential tool results chain linearly in the append-only log.
        _result_msg("r3", "r2", "c3", "bash", "3 passed"),
        # User replies, then a final text-only assistant round.
        _msg("u2", "r3", "user", [_text("Looks good, continue.")]),
        _msg("a3", "u2", "assistant", [_text("The admission check is repaired.")]),
    ]
    dead = [
        # A retry of round 2 that the live chain does not pass through,
        # written after the live records in file order.
        _msg(
            "a2b",
            "r1",
            "assistant",
            [_call("c2b", "bash", {"command": "cat src/scheduler.py"})],
        ),
        _result_msg("r2b", "a2b", "c2b", "bash", "dead branch result"),
    ]
    markers = [
        {
            "id": "flow1",
            "parentId": "r1",
            "timestamp": "2026-09-19T06:44:50.500Z",
            "type": "custom",
            "customType": "jouzu-flow-branch",
            "data": {"version": 1, "sessionId": "s", "branchId": "b"},
        },
    ]
    tail = [
        {
            "id": "comp",
            "parentId": "a3",
            "timestamp": "2026-09-19T07:00:00.000Z",
            "type": "compaction",
            "summary": "summary text",
            "tokensBefore": 9000,
            "fromHook": True,
            "firstKeptEntryId": "u3",
            "details": {},
        },
        _msg("u3", "comp", "user", [_text("continue?")]),
        _msg("a4", "u3", "assistant", []),
    ]
    # flow1 sits between r1 and a2 on the live ancestry: a2's parent must
    # pass through it.  Insert the marker into the parent chain.
    for record in live:
        if record["id"] == "a2":
            record["parentId"] = "flow1"
    return live[:4] + markers + live[4:] + dead + tail


@pytest.fixture()
def session_path(tmp_path: Path) -> Path:
    return _write_session(tmp_path / "session.jsonl", _session_records())


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


def test_chain_walk_follows_ancestry_not_file_order(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    entries = fixture["workloads"][0]["entries"]
    texts = [e.get("content") for e in entries if e["role"] != "system"]
    assert "dead branch result" not in json.dumps(fixture)
    assert "cat src/scheduler.py" not in json.dumps(fixture)
    # Every entry comes from the live chain, in ancestry order.
    assert texts == [
        "Fix the scheduler admission bug.",
        "",  # a1: text-free assistant round carrying tool_calls
        "src/scheduler.py:41:def admit",
        "",  # a2
        "def admit(self, request): return ok",
        "3 passed",
        "Looks good, continue.",
        "The admission check is repaired.",
    ]


def test_converter_emits_renderer_message_shapes(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    entries = fixture["workloads"][0]["entries"]
    by_tool_call_id = {
        call["id"]: (index, call)
        for index, entry in enumerate(entries)
        for call in entry.get("tool_calls", [])
    }
    # Assistant rounds carry OpenAI tool-call envelopes with JSON-string arguments.
    a1 = entries[1]
    assert a1["role"] == "assistant"
    assert set(a1.keys()) == {"role", "content", "tool_calls"}
    call = a1["tool_calls"][0]
    assert set(call.keys()) == {"id", "type", "function"}
    assert call["type"] == "function"
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "grep admit src/"}
    # Tool results are role "tool" messages keyed by tool_call_id.
    r1 = entries[2]
    assert r1 == {
        "role": "tool",
        "tool_call_id": by_tool_call_id["c1"][1]["id"],
        "content": "src/scheduler.py:41:def admit",
    }
    # Multi-call round keeps both calls in order with paired results.
    a2 = entries[3]
    assert [c["function"]["name"] for c in a2["tool_calls"]] == ["read", "bash"]
    assert [entries[i]["tool_call_id"] for i in (4, 5)] == [
        a2["tool_calls"][0]["id"],
        a2["tool_calls"][1]["id"],
    ]


def test_thinking_blocks_are_not_replayed(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    dumped = json.dumps(fixture["workloads"][0]["entries"])
    assert "Let me inspect the file first." not in dumped
    assert '"thinking"' not in dumped


def test_request_boundaries_match_client_resends(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    workload = fixture["workloads"][0]
    entries = workload["entries"]
    ends = workload["request_ends"]
    assert ends == [1, 3, 6, 7]
    loaded = load_session_replay_fixture(_fixture_file(fixture))
    assert session_replay_turn_count(loaded, workload["id"]) == 4
    first = build_session_replay_turn_messages(loaded, workload["id"], turn_index=0)
    assert [m["role"] for m in first] == ["system", "user"]
    third = build_session_replay_turn_messages(loaded, workload["id"], turn_index=2)
    assert [m["role"] for m in third] == ["system", "user", "assistant", "tool", "assistant", "tool", "tool"]
    last = build_session_replay_turn_messages(loaded, workload["id"], turn_index=3)
    assert [m["role"] for m in last] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "tool", "user",
    ]
    # The final assistant reply is the target of the last request, never resent.
    assert all(m.get("content") != "The admission check is repaired." for m in last)


def test_default_tip_is_last_content_bearing_assistant_before_compaction(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    # The post-compaction stub and the corrupted tail are excluded by default.
    assert "continue?" not in json.dumps(fixture)


def test_compaction_anchor_follows_parents_past_dead_assistant(tmp_path: Path) -> None:
    records = [
        _msg("u1", None, "user", [_text("go")]),
        _msg("live", "u1", "assistant", [_call("c1", "bash", {"command": "live"})]),
        _msg("dead", "u1", "assistant", [_text("dead retry")]),
        _result_msg("r1", "live", "c1", "bash", "live result"),
        {"id": "comp", "parentId": "r1", "type": "compaction"},
    ]
    fixture = convert_session_jsonl(_write_session(tmp_path / "branch.jsonl", records))
    assert fixture["source"]["record"]["tip_record"] == "r1"
    assert "dead retry" not in json.dumps(fixture["workloads"])
    assert fixture["workloads"][0]["request_ends"] == [1, 3]


def test_converter_rejects_missing_parent(tmp_path: Path) -> None:
    records = [
        _msg("u1", "missing", "user", [_text("go")]),
        _msg("a1", "u1", "assistant", [_text("done")]),
    ]
    with pytest.raises(SessionReplayError, match="parent"):
        convert_session_jsonl(_write_session(tmp_path / "broken.jsonl", records))


def test_explicit_tip_can_shorten_the_slice(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, tip="r3", workload_id="s")
    dumped = json.dumps(fixture)
    assert "Looks good, continue." not in dumped
    assert "The admission check is repaired." not in dumped
    assert "dead branch result" not in dumped
    # The chain ends on a completed tool round, so that final resend exists.
    assert fixture["workloads"][0]["request_ends"] == [1, 3, 6]


def test_tail_rounds_keep_only_the_last_n_rounds(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, tail_rounds=1, workload_id="s")
    workload = fixture["workloads"][0]
    # The slice starts at a user-message boundary: u2, then the final round.
    assert workload["slice"]["start_reason"] == "user_message"
    roles = [e["role"] for e in workload["entries"] if e["role"] != "system"]
    assert roles == ["user", "assistant"]
    # After u2 one request exists; the final assistant reply is its target.
    assert workload["request_ends"] == [1]


def test_tool_result_cap_truncates_long_outputs(tmp_path: Path) -> None:
    long_text = "x" * 500
    records = [
        _msg("u1", None, "user", [_text("go")]),
        _msg("a1", "u1", "assistant", [_call("c1", "bash", {"command": "ls"})]),
        _result_msg("r1", "a1", "c1", "bash", long_text),
        _msg("a2", "r1", "assistant", [_text("done")]),
    ]
    path = _write_session(tmp_path / "long.jsonl", records)
    fixture = convert_session_jsonl(path, max_tool_result_chars=100, workload_id="s")
    tool_entry = fixture["workloads"][0]["entries"][2]
    assert tool_entry["content"].startswith("xxxx")
    assert len(tool_entry["content"]) <= 120
    assert "truncated" in tool_entry["content"]


def test_unpaired_tip_tool_call_is_rejected_or_trimmed(tmp_path: Path) -> None:
    records = [
        _msg("u1", None, "user", [_text("go")]),
        _msg("a1", "u1", "assistant", [_call("c1", "bash", {"command": "ls"})]),
        # No tool result for c1: the session died mid-round.
        _msg("a2", "a1", "assistant", [_text("never reached")]),
    ]
    path = _write_session(tmp_path / "dangling.jsonl", records)
    with pytest.raises(SessionReplayError):
        convert_session_jsonl(path, tip="a1", workload_id="s")


# ---------------------------------------------------------------------------
# Fixture schema and loader
# ---------------------------------------------------------------------------


def _fixture_file(fixture: dict[str, Any]) -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(fixture, handle)
    handle.close()
    return Path(handle.name)


def test_fixture_round_trips_with_metadata(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    assert fixture["kind"] == SESSION_REPLAY_KIND
    assert fixture["schema_version"] == 1
    assert fixture["suite"]
    assert fixture["source"]["record"]["session_id"]
    assert fixture["source"]["record"]["tip_record"]
    path = _fixture_file(fixture)
    loaded = load_session_replay_fixture(path)
    assert loaded.workloads["s"]["id"] == "s"
    tools = build_session_replay_tools(loaded)
    names = {tool["function"]["name"] for tool in tools}
    assert names == {"bash", "read"}
    for tool in tools:
        assert tool["type"] == "function"
        assert isinstance(tool["function"]["parameters"], dict)


def test_loader_rejects_bad_request_ends(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    fixture["workloads"][0]["request_ends"] = [3, 1, 6, 7]
    with pytest.raises(SessionReplayError):
        load_session_replay_fixture(_fixture_file(fixture))


@pytest.mark.parametrize("boundary", [2, 4, 5, 8])
def test_loader_rejects_incomplete_or_assistant_request_boundary(
    session_path: Path, boundary: int
) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    fixture["workloads"][0]["request_ends"] = [boundary]
    with pytest.raises(SessionReplayError, match="request"):
        load_session_replay_fixture(_fixture_file(fixture))


def test_loader_rejects_unmatched_tool_results(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    fixture["workloads"][0]["entries"][2]["tool_call_id"] = "call_missing"
    with pytest.raises(SessionReplayError):
        load_session_replay_fixture(_fixture_file(fixture))


def test_loader_rejects_unknown_workload(session_path: Path) -> None:
    fixture = convert_session_jsonl(session_path, workload_id="s")
    loaded = load_session_replay_fixture(_fixture_file(fixture))
    with pytest.raises(SessionReplayError):
        build_session_replay_turn_messages(loaded, "nope", turn_index=0)
