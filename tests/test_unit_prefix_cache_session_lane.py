"""Unit tests for the session-replay lane of the prefix-cache harness.

Drives ``_run_lane`` for ``kind == "session_replay"`` with a stub ``run_turn``
and a whitespace tokenizer, so the cumulative-resend contract, tool plumbing,
turn limits, and LCP observation are pinned without a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hipengine.benchmark.session_replay import convert_session_jsonl
from scripts.prefix_cache_multiturn_bench import (
    LaneResult,
    _run_lane,
    _session_replay_lanes,
)


# ---------------------------------------------------------------------------
# Synthetic replay fixture
# ---------------------------------------------------------------------------


def _msg(record_id: str, parent: str | None, role: str, message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record_id,
        "parentId": parent,
        "timestamp": "2026-09-19T06:44:50.000Z",
        "type": "message",
        "message": message,
    }


def _session_file(path: Path) -> Path:
    records = [
        {
            "id": "sess",
            "type": "session",
            "cwd": "/tmp",
            "timestamp": "2026-09-19T06:44:50.000Z",
            "version": 1,
        },
        _msg(
            "u1",
            None,
            "user",
            {
                "role": "user",
                "content": [{"type": "text", "text": "Fix the failing scheduler test."}],
                "timestamp": "2026-09-19T06:44:50.000Z",
            },
        ),
        _msg(
            "a1",
            "u1",
            "assistant",
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "c1",
                        "name": "bash",
                        "arguments": {"command": "pytest tests/test_scheduler.py"},
                    }
                ],
                "timestamp": "2026-09-19T06:44:50.000Z",
            },
        ),
        _msg(
            "r1",
            "a1",
            "toolResult",
            {
                "role": "toolResult",
                "toolCallId": "c1",
                "toolName": "bash",
                "isError": False,
                "details": None,
                "content": [{"type": "text", "text": "1 failed in 2.1s"}],
                "timestamp": "2026-09-19T06:44:50.000Z",
            },
        ),
        _msg(
            "a2",
            "r1",
            "assistant",
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The test is repaired."}],
                "timestamp": "2026-09-19T06:44:50.000Z",
            },
        ),
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def fixture_path(tmp_path: Path) -> Path:
    session = _session_file(tmp_path / "session.jsonl")
    payload = convert_session_jsonl(session, workload_id="synthetic")
    out = tmp_path / "replay.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


class _StubLLM:
    def tokenize(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


def _recorder(calls: list[dict[str, Any]]) -> Any:
    def run_turn(**kwargs: Any) -> dict[str, Any]:
        record = {
            "text": "ok",
            "prompt_tokens": 0,
            "output_tokens": 0,
        }
        calls.append({"kwargs": kwargs, "record": record})
        return dict(record)

    return run_turn


def test_session_lanes_build_from_fixture(fixture_path: Path) -> None:
    lanes = _session_replay_lanes(fixture_path, workload_ids=[], max_tokens=64)
    assert list(lanes) == ["session-synthetic"]
    lane = lanes["session-synthetic"]
    assert lane["kind"] == "session_replay"
    assert lane["workload_id"] == "synthetic"
    assert [tool["function"]["name"] for tool in lane["tools"]] == ["bash"]


def test_session_lane_replays_cumulative_resends(fixture_path: Path) -> None:
    lanes = _session_replay_lanes(fixture_path, workload_ids=[], max_tokens=64)
    calls: list[dict[str, Any]] = []
    result = _run_lane(
        _StubLLM(),
        engine=None,
        lane_id="session-synthetic",
        lane=lanes["session-synthetic"],
        turn_limit=None,
        run_turn=_recorder(calls),
    )
    assert isinstance(result, LaneResult)
    assert result.lane_kind == "session_replay"
    # Requests: after u1, and after the completed tool round (r1). The final
    # assistant reply never triggers a resend, so there are two requests.
    assert len(result.turns) == 2
    prompts = [call["kwargs"]["messages"] for call in calls]
    assert [m["role"] for m in prompts[0]] == ["system", "user"]
    assert [m["role"] for m in prompts[1]] == ["system", "user", "assistant", "tool"]
    # The final assistant reply is never resent.
    resent = json.dumps(prompts[1])
    assert "The test is repaired." not in resent
    # Tools and output budget are forwarded every turn.
    for call in calls:
        assert call["kwargs"]["max_tokens"] == 64
        assert [t["function"]["name"] for t in call["kwargs"]["tools"]] == ["bash"]
    # Lane metadata is stamped on every record.
    for index, turn in enumerate(result.turns):
        assert turn["lane_id"] == "session-synthetic"
        assert turn["lane_kind"] == "session_replay"
        assert turn["turn_index"] == index


def test_session_lane_records_lcp_observation(fixture_path: Path) -> None:
    lanes = _session_replay_lanes(fixture_path, workload_ids=[], max_tokens=64)
    result = _run_lane(
        _StubLLM(),
        engine=None,
        lane_id="session-synthetic",
        lane=lanes["session-synthetic"],
        turn_limit=None,
        run_turn=_recorder([]),
    )
    # The whitespace tokenizer makes every earlier prompt a strict prefix of
    # the next one, so the reusable share equals the previous prompt length.
    tokens = [turn["prompt_tokens_local"] for turn in result.turns]
    assert all(value > 0 for value in tokens)
    assert tokens == sorted(tokens)
    for previous, turn in zip(result.turns, result.turns[1:]):
        assert turn["prompt_lcp_tokens"] == previous["prompt_tokens_local"]
        assert turn["previous_prompt_tokens"] == previous["prompt_tokens_local"]
        assert turn["prompt_lcp_reusable_tokens"] == (
            turn["prompt_lcp_tokens"] // 256
        ) * 256


def test_session_lane_respects_turn_limit(fixture_path: Path) -> None:
    lanes = _session_replay_lanes(fixture_path, workload_ids=[], max_tokens=64)
    calls: list[dict[str, Any]] = []
    result = _run_lane(
        _StubLLM(),
        engine=None,
        lane_id="session-synthetic",
        lane=lanes["session-synthetic"],
        turn_limit=1,
        run_turn=_recorder(calls),
    )
    assert len(result.turns) == 1
    assert len(calls) == 1


def test_session_lane_rejects_unknown_workload(fixture_path: Path) -> None:
    with pytest.raises(ValueError):
        _session_replay_lanes(fixture_path, workload_ids=["missing"], max_tokens=64)
