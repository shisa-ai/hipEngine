from __future__ import annotations

import pytest

from scripts.pm4_lifecycle_repro import (
    REPO_ROOT,
    ReproConfig,
    _JsonlEventWriter,
    _close_generation,
    _source_context,
    build_parser,
    plan_from_args,
)


def test_lifecycle_repro_defaults_are_safe_reuse_without_timestamps() -> None:
    args = build_parser().parse_args([])
    config = plan_from_args(args)

    assert config.transport == "pm4"
    assert config.cycles == 4
    assert config.queue_mode == "reuse"
    assert config.resource_mode == "reuse"
    assert config.allocation_mode == "hip"
    assert config.submit is True
    assert config.timestamps is False
    assert config.quarantine_generations == 0
    assert config.reset_risk_acknowledged is False
    assert config.destructive is False


def test_any_native_submit_resource_recreate_requires_reset_risk_acknowledgement() -> None:
    parser = build_parser()
    for cycles in (1, 128):
        args = parser.parse_args(["--resource-mode", "recreate", "--cycles", str(cycles)])
        with pytest.raises(ValueError, match="reset-risk"):
            plan_from_args(args)

    acknowledged = parser.parse_args(
        ["--resource-mode", "recreate", "--cycles", "1", "--ack-reset-risk"]
    )
    config = plan_from_args(acknowledged)
    assert config.destructive is True
    assert config.reset_risk_acknowledged is True


def test_generation_teardown_preserves_first_failure_and_skips_pointee_release() -> None:
    calls: list[str] = []

    class FailingExecutable:
        def close(self) -> None:
            calls.append("executable")
            raise RuntimeError("destroy failed")

    class Context:
        def close(self) -> None:
            calls.append("context")

    statuses = _close_generation(
        {
            "executable": FailingExecutable(),
            "context": Context(),
            "hip_buffers": [object()],
            "hsa_buffers": [],
            "free_hip_buffer": lambda buffer: calls.append("buffer"),
        }
    )

    assert calls == ["executable"]
    assert statuses == [
        {
            "operation": "executable.close",
            "status": "fail",
            "error_type": "RuntimeError",
            "error": "destroy failed",
        },
        {
            "operation": "generation.remaining_resources",
            "status": "skipped",
            "reason": "executable close did not prove packet-pointee release",
        },
    ]


def test_generation_teardown_closes_packet_owners_before_all_pointees() -> None:
    calls: list[str] = []

    class Owner:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    class Runtime:
        def graph_exec_destroy(self, handle: int) -> None:
            calls.append(f"graph_exec:{handle}")

        def graph_destroy(self, handle: int) -> None:
            calls.append(f"graph:{handle}")

    statuses = _close_generation(
        {
            "executable": Owner("executable"),
            "context": Owner("context"),
            "hsa_buffers": [Owner("hsa0"), Owner("hsa1")],
            "hip_buffers": ["hip0", "hip1"],
            "free_hip_buffer": lambda buffer: calls.append(buffer),
            "runtime": Runtime(),
            "graph_exec": 91,
            "graph": 81,
        }
    )

    assert calls == [
        "executable",
        "graph_exec:91",
        "graph:81",
        "hsa1",
        "hsa0",
        "hip1",
        "hip0",
        "context",
    ]
    assert all(status["status"] == "pass" for status in statuses)


def test_lifecycle_repro_rejects_incoherent_resource_and_quarantine_modes() -> None:
    with pytest.raises(ValueError, match="resource-mode=recreate"):
        ReproConfig(queue_mode="recreate", resource_mode="reuse").validated()
    with pytest.raises(ValueError, match="queue-mode=recreate"):
        ReproConfig(quarantine_generations=2).validated()
    with pytest.raises(ValueError, match="timestamps"):
        ReproConfig(transport="hipgraph", timestamps=True).validated()


def test_lifecycle_event_journal_fsyncs_complete_json_lines(tmp_path) -> None:
    journal = tmp_path / "lifecycle.jsonl"
    with _JsonlEventWriter(journal) as writer:
        writer({"event": "run_started", "cycle": None})
        writer({"event": "cycle_prepared", "cycle": 0})

    assert journal.read_text(encoding="utf-8").splitlines() == [
        '{"cycle":null,"event":"run_started"}',
        '{"cycle":0,"event":"cycle_prepared"}',
    ]


def test_lifecycle_source_context_is_pinned_to_this_repository() -> None:
    source = _source_context()
    assert source["repo_root"] == str(REPO_ROOT)
    assert source["import_root"] == source["repo_root"]
    assert len(source["hipengine_commit"]) == 40
    assert isinstance(source["dirty"], bool)


def test_no_submit_create_drop_plan_is_non_destructive() -> None:
    args = build_parser().parse_args(
        [
            "--queue-mode",
            "recreate",
            "--resource-mode",
            "recreate",
            "--cycles",
            "8",
            "--no-submit",
            "--allocation-mode",
            "hsa",
            "--quarantine-generations",
            "2",
        ]
    )
    config = plan_from_args(args)
    assert config.submit is False
    assert config.destructive is False
    assert config.allocation_mode == "hsa"
    assert config.quarantine_generations == 2
