from __future__ import annotations

import json

import pytest

from scripts.gguf_mtp_long_context_task_gate import (
    CATEGORIES,
    TaskSuiteError,
    finalize_payload,
    load_tasks,
    score_task_output,
)


def _row(task_id: str, category: str, expected: str = "B") -> dict[str, object]:
    return {
        "id": task_id,
        "category": category,
        "target_context_tokens": 32,
        "prefix": "prefix",
        "filler": "filler",
        "evidence": [{"position": 0.5, "text": "fact"}],
        "suffix": "suffix",
        "expected": ["answer"],
        "scorer": "choice_exact",
        "choices": {"A": "wrong-a", "B": "answer", "C": "wrong-c", "D": "wrong-d"},
        "expected_choice": expected,
    }


def test_committed_task_fixture_covers_all_categories() -> None:
    rows = load_tasks(
        __import__("pathlib").Path("benchmarks/prompts/mtp-realworld-long-context.jsonl")
    )

    assert tuple(row["category"] for row in rows) == CATEGORIES
    assert all(row["target_context_tokens"] == 4096 for row in rows)
    assert len({row["id"] for row in rows}) == len(CATEGORIES)


def test_load_tasks_fails_closed_on_missing_category(tmp_path) -> None:
    path = tmp_path / "suite.jsonl"
    path.write_text(json.dumps(_row("one", "retrieval")) + "\n")

    with pytest.raises(TaskSuiteError, match="every RF1 category"):
        load_tasks(path)


def test_score_task_output_accepts_choice_or_declared_answer_text() -> None:
    task = _row("one", "retrieval")

    assert score_task_output("B", task)["passed"] is True
    assert score_task_output("The answer is answer.", task)["passed"] is True
    assert score_task_output("C", task)["passed"] is False


def test_finalize_payload_separates_rf1_binding_from_absolute_task_score() -> None:
    payload = {
        "rows": [
            {
                "id": "task",
                "output_ids_exact": True,
                "gpu_accept_match_cpu": True,
                "all_cycles_eager": True,
                "score": {"passed": False},
            }
        ],
        "summary": {"wall_seconds": 1.0},
    }

    finalized = finalize_payload(payload)

    assert finalized["passed"] is True
    assert finalized["rows"][0]["binding_passed"] is True
    assert finalized["rows"][0]["task_score_passed"] is False
    assert finalized["summary"]["absolute_task_quality_passed"] is False
    assert finalized["production_quality_claim"] is False


def test_disable_target_graph_forces_the_eager_binding() -> None:
    """A graph-eligible cycle cannot satisfy the eager-ownership binding.

    ``finalize_payload`` already refuses a row whose cycle submitted the cached
    native target graph, so a packet that measures the eager chain has to
    disable graph eligibility. The switch is what makes that reachable: without
    it the session is built with graph replay allowed and the binding depends on
    the graph being refused at the packet's context.
    """

    import inspect

    import scripts.gguf_mtp_long_context_task_gate as gate

    source = inspect.getsource(gate)
    assert "allow_graph=not bool(args.disable_target_graph)" in source
    assert '"disable_target_graph": bool(args.disable_target_graph)' in source

    graph_row = {
        "id": "task",
        "output_ids_exact": True,
        "gpu_accept_match_cpu": True,
        "all_cycles_eager": False,
        "score": {"passed": True},
    }
    finalized = finalize_payload({"rows": [graph_row], "summary": {}})
    assert finalized["passed"] is False
    assert finalized["rows"][0]["binding_passed"] is False


def test_decode_session_forwards_allow_graph_to_the_verifier() -> None:
    """The eager-route knob has to reach the verifier's own switch.

    ``Qwen35GGUFMTPDecodeSession`` used to call ``verifier.prepare`` without an
    ``allow_graph`` argument, so its cycles always allowed the cached native
    target graph and a packet that binds eager ownership could only pass if the
    graph happened to be ineligible at its context. The session now stores the
    caller's choice and forwards it.
    """

    import inspect

    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession

    signature = inspect.signature(Qwen35GGUFMTPDecodeSession.__init__)
    assert signature.parameters["allow_graph"].default is True
    source = inspect.getsource(Qwen35GGUFMTPDecodeSession)
    assert "self.allow_graph = bool(allow_graph)" in source
    assert "allow_graph=bool(self.allow_graph)," in source
