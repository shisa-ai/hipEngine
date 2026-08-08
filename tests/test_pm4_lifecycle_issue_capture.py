from __future__ import annotations

from pathlib import Path

import pytest

from scripts.pm4_lifecycle_issue_capture import (
    DESTRUCTIVE_APPROVAL_TOKEN,
    build_parser,
    build_reproducer_invocation,
    classify_kernel_journal,
    discover_devcoredump_data,
    plan_from_args,
)


def _plan(*args: str):
    return plan_from_args(build_parser().parse_args(list(args)))


def test_capture_defaults_plan_a_safe_no_submit_recreate_arm(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    plan = _plan("--output-dir", str(output))

    assert plan.execute is False
    assert plan.submit_recreate is False
    assert plan.destructive is False
    assert plan.transport == "pm4"
    assert plan.cycles == 8
    assert plan.allocation_mode == "hip"
    assert plan.buffer_mode == "reuse"
    assert plan.devcoredump_reader == "sudo"

    command = build_reproducer_invocation(plan)
    assert "--no-submit" in command
    assert "--ack-reset-risk" not in command
    assert command[command.index("--queue-mode") + 1] == "recreate"
    assert command[command.index("--resource-mode") + 1] == "recreate"
    assert command[command.index("--journal-jsonl") + 1] == str(
        output / "lifecycle-events.jsonl"
    )


def test_capture_rejects_destructive_execution_without_both_guards(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="ack-reset-risk"):
        _plan("--output-dir", str(output), "--submit-recreate")

    with pytest.raises(ValueError, match="approval-token"):
        _plan(
            "--output-dir",
            str(output),
            "--submit-recreate",
            "--ack-reset-risk",
            "--execute",
        )

    plan_only = _plan(
        "--output-dir",
        str(output),
        "--submit-recreate",
        "--ack-reset-risk",
    )
    assert plan_only.destructive is True
    assert plan_only.execute is False
    assert plan_only.cycles == 1


def test_capture_builds_exact_approved_cached_pm4_command(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    compiler = tmp_path / "hipcc-version.txt"
    plan = _plan(
        "--output-dir",
        str(output),
        "--compiler-version-file",
        str(compiler),
        "--submit-recreate",
        "--ack-reset-risk",
        "--approval-token",
        DESTRUCTIVE_APPROVAL_TOKEN,
        "--execute",
        "--transport",
        "pm4",
        "--cycles",
        "128",
        "--timestamps",
    )

    command = build_reproducer_invocation(plan)
    assert "--submit" in command
    assert "--ack-reset-risk" in command
    assert "--require-cached-build" in command
    assert command[command.index("--cycles") + 1] == "128"
    assert "--timestamps" in command


def test_capture_hsa_allocation_forces_context_owned_buffer_recreation(tmp_path: Path) -> None:
    plan = _plan(
        "--output-dir",
        str(tmp_path / "evidence"),
        "--allocation-mode",
        "hsa",
    )
    assert plan.allocation_mode == "hsa"
    assert plan.buffer_mode == "recreate"
    command = build_reproducer_invocation(plan)
    assert command[command.index("--buffer-mode") + 1] == "recreate"


def test_capture_rejects_repo_local_evidence_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="outside the repository"):
        _plan("--repo-root", str(repo), "--output-dir", str(repo / "raw"))


def test_devcoredump_discovery_covers_class_and_drm_layouts(tmp_path: Path) -> None:
    devcoredump = tmp_path / "devcoredump"
    drm = tmp_path / "drm"
    class_data = devcoredump / "devcd0" / "data"
    drm_data = drm / "card0" / "device" / "devcoredump" / "data"
    class_data.parent.mkdir(parents=True)
    drm_data.parent.mkdir(parents=True)
    class_data.write_bytes(b"class")
    drm_data.write_bytes(b"drm")

    assert discover_devcoredump_data(devcoredump, drm) == [class_data, drm_data]


def test_kernel_journal_classifier_requires_the_complete_issue_tuple() -> None:
    classification = classify_kernel_journal(
        "amdgpu: address 0x0000000000000000; GPU reset begin!"
    )
    assert classification["issue_6529_fault_tuple"] is False
    assert classification["classification"] == "other_amdgpu_recovery_event"


def test_kernel_journal_classifier_matches_issue_6529_tuple() -> None:
    journal = """
    amdgpu: [gfxhub] page fault in page starting at address 0x0000000000000000 from client 10
    amdgpu: GCVM_L2_PROTECTION_FAULT_STATUS:0x00801431
    amdgpu: Faulty UTCL2 client ID: SQC (data) (0xa)
    amdgpu: MES failed to respond to msg=REMOVE_QUEUE
    amdgpu: GPU reset begin!. Source: 3
    amdgpu: VRAM is lost due to GPU reset!
    """
    classification = classify_kernel_journal(journal)

    assert classification["issue_6529_fault_tuple"] is True
    assert classification["remove_queue_failure"] is True
    assert classification["gpu_reset"] is True
    assert classification["vram_lost"] is True
    assert classification["classification"] == "reproduced_issue_6529_signature"
