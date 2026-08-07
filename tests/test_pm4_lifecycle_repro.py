from __future__ import annotations

import pytest

from scripts.pm4_lifecycle_repro import ReproConfig, build_parser, plan_from_args


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


def test_recreate_stress_requires_explicit_reset_risk_acknowledgement() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--queue-mode", "recreate", "--resource-mode", "recreate", "--cycles", "128"]
    )
    with pytest.raises(ValueError, match="reset-risk"):
        plan_from_args(args)

    acknowledged = parser.parse_args(
        [
            "--queue-mode",
            "recreate",
            "--resource-mode",
            "recreate",
            "--cycles",
            "128",
            "--ack-reset-risk",
        ]
    )
    config = plan_from_args(acknowledged)
    assert config.destructive is True
    assert config.reset_risk_acknowledged is True


def test_lifecycle_repro_rejects_incoherent_resource_and_quarantine_modes() -> None:
    with pytest.raises(ValueError, match="resource-mode=recreate"):
        ReproConfig(queue_mode="recreate", resource_mode="reuse").validated()
    with pytest.raises(ValueError, match="queue-mode=recreate"):
        ReproConfig(quarantine_generations=2).validated()
    with pytest.raises(ValueError, match="timestamps"):
        ReproConfig(transport="hipgraph", timestamps=True).validated()


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
