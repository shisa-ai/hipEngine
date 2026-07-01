from __future__ import annotations

import sys

import pytest

from scripts import gguf_ar_mtp_suite as suite


def test_suite_default_mtp_route_is_current_speed_row() -> None:
    assert suite.DEFAULT_MTP_ROUTE == "resident-b1-probe-block-direct-cap32k-minrows2-pmin05"
    assert suite.DEFAULT_MTP_ROUTE in suite.MTP_ROUTES


def test_suite_default_route_enables_b2_block_verify() -> None:
    # the default must carry --target-block-min-rows 2 so B2 (3-row) block verify
    # amortizes instead of falling to serial; that is part of the retained win.
    flags = suite.MTP_ROUTES[suite.DEFAULT_MTP_ROUTE]
    assert "--target-block-min-rows" in flags
    assert flags[flags.index("--target-block-min-rows") + 1] == "2"


def test_suite_default_route_uses_draft_confidence_gate() -> None:
    # the default must carry --draft-p-min 0.5: gating low-confidence draft cycles
    # (hit conf ~0.69 vs miss ~0.36) trims wasted block-verify and unlocks deeper
    # budgets (B5 best), the 2026-06-30 retained speed win (1.0399 -> 1.055x AR).
    flags = suite.MTP_ROUTES[suite.DEFAULT_MTP_ROUTE]
    assert "--draft-p-min" in flags
    assert flags[flags.index("--draft-p-min") + 1] == "0.5"


def test_suite_exposes_resident_strict_context_route() -> None:
    assert suite.MTP_ROUTES["resident-strict-context"] == [
        "--resident-mtp-draft",
        "--root-topk-accept",
        "1",
        "--sibling-topk-accept",
        "1",
        "--mtp-context-replay",
        "--mtp-device-kv-cache",
        "--no-target-block-verify",
    ]


def test_suite_exposes_resident_strict_context_block_pmin_route() -> None:
    assert suite.MTP_ROUTES["resident-strict-context-block-pmin08"] == [
        "--resident-mtp-draft",
        "--draft-p-min",
        "0.8",
        "--root-topk-accept",
        "1",
        "--sibling-topk-accept",
        "1",
        "--mtp-context-replay",
        "--mtp-device-kv-cache",
        "--target-block-verify",
        "--adaptive-ar-fallback",
        "--mtp-draft-vocab-cap",
        "32768",
    ]


def test_suite_exposes_cap32k_recovery_route() -> None:
    assert suite.MTP_ROUTES["resident-cap32k-recover"] == [
        "--resident-mtp-draft",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
        "--mtp-draft-vocab-cap",
        "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ]


def test_suite_exposes_cap32k_device_seed_route() -> None:
    assert suite.MTP_ROUTES["resident-cap32k-device-seed"] == [
        "--resident-mtp-draft",
        "--resident-mtp-device-seed",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
        "--mtp-draft-vocab-cap",
        "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ]


def test_suite_exposes_cap32k_device_seed_kv_route() -> None:
    assert suite.MTP_ROUTES["resident-cap32k-device-seed-kv"] == [
        "--resident-mtp-draft",
        "--resident-mtp-device-seed",
        "--mtp-device-kv-cache",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
        "--mtp-draft-vocab-cap",
        "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ]


def test_suite_exposes_context_cap32k_device_seed_route() -> None:
    assert suite.MTP_ROUTES["resident-context-cap32k-device-seed"] == [
        "--resident-mtp-draft",
        "--resident-mtp-device-seed",
        "--mtp-context-replay",
        "--mtp-device-kv-cache",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
        "--mtp-draft-vocab-cap",
        "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ]


def test_suite_exposes_b1_branch_safe_block_device_seed_route() -> None:
    assert suite.MTP_ROUTES["resident-b1-branch-safe-block-cap32k-device-seed"] == [
        "--resident-mtp-draft",
        "--resident-mtp-device-seed",
        "--target-block-verify",
        "--target-b1-branch-safe-block-verify",
        "--adaptive-ar-fallback",
        "--mtp-draft-vocab-cap",
        "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ]


def test_suite_exposes_hybrid_strict_block_cap32k_route() -> None:
    assert suite.MTP_ROUTES["resident-hybrid-strict-block-cap32k"] == [
        "--resident-mtp-draft",
        "--root-topk-accept",
        "1",
        "--sibling-topk-accept",
        "1",
        "--target-block-verify",
        "--adaptive-block-after-full-accept",
        "--adaptive-probe-draft-n-max",
        "3",
        "--adaptive-strict-block-probe",
        "--adaptive-strict-probe-cycles",
        "2",
        "--adaptive-strict-probe-min-accepted",
        "2",
        "--adaptive-strict-fallback-draft-n-max",
        "1",
        "--adaptive-strict-fallback-root-topk",
        "40",
        "--adaptive-ar-fallback",
        "--mtp-draft-vocab-cap",
        "32768",
        "--adaptive-full-vocab-after-cap-miss",
    ]


def test_suite_exposes_b1_probe_block_direct_cap32k_route() -> None:
    assert suite.MTP_ROUTES["resident-b1-probe-block-direct-cap32k"] == [
        "--resident-mtp-draft",
        "--root-topk-accept",
        "1",
        "--sibling-topk-accept",
        "1",
        "--target-block-verify",
        "--target-block-direct-state-commit",
        "--adaptive-block-after-full-accept",
        "--adaptive-probe-draft-n-max",
        "1",
        "--adaptive-ar-fallback",
        "--mtp-draft-vocab-cap",
        "32768",
    ]


def test_suite_exposes_llama_compat_routes() -> None:
    assert suite.MTP_ROUTES["llama-compat"] == ["--llama-compat"]
    assert suite.MTP_ROUTES["llama-compat-dp4a"] == ["--llama-compat", "--verify-dp4a"]
    assert suite.MTP_ROUTES["llama-compat-device-chain"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--selected-down-x8-repack",
        "q6",
        "--selected-gate-up-x8",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--selected-down-x8-repack",
        "q6",
        "--verify-dense-q8-dp4a",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--resident-mtp-draft-q6-top1-stage1-threads",
        "64",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--resident-mtp-draft-q6-top1-stage1-shape",
        "row",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--resident-mtp-draft-q6-top1-stage1-shape",
        "pack8_scalehoist",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-draftsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--verify-dp4a",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--selected-down-x8-repack",
        "q6",
        "--selected-gate-up-x8",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--selected-down-x8-repack",
        "q6",
        "--verify-dense-q8-dp4a",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--resident-mtp-draft-q6-top1-stage1-threads",
        "64",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--resident-mtp-draft-q6-top1-stage1-shape",
        "row",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist-allsync"] == [
        "--llama-compat",
        "--resident-mtp-device-chain",
        "--resident-mtp-draft-sync-stage-timings",
        "--target-block-sync-stage-timings",
        "--verify-dp4a",
        "--resident-mtp-draft-q6-top1-dp4a",
        "--resident-mtp-draft-q6-top1-stage1-shape",
        "pack8_scalehoist",
        "--selected-down-x8-repack",
        "q6",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-seed-chain"] == [
        "--llama-compat",
        "--resident-mtp-device-seed",
        "--resident-mtp-device-chain",
    ]
    assert suite.MTP_ROUTES["llama-compat-device-seed-chain-dp4a"] == [
        "--llama-compat",
        "--resident-mtp-device-seed",
        "--resident-mtp-device-chain",
        "--verify-dp4a",
    ]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-dp4a"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-q6top1dp4a"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-draftsync"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-chain-dp4a-allsync"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-allsync"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup-allsync"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8-allsync"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-allsync"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row-allsync"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS[
        "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist-allsync"
    ] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-seed-chain"] == [2]
    assert suite.MTP_ROUTE_DEFAULT_BUDGETS["llama-compat-device-seed-chain-dp4a"] == [2]


def test_suite_exposes_fused_b1_block_probe_route() -> None:
    assert suite.MTP_ROUTES["resident-fused-b1-block-direct-cap32k-minrows2-pmin05"] == [
        "--resident-mtp-draft",
        "--root-topk-accept",
        "1",
        "--sibling-topk-accept",
        "1",
        "--target-block-verify",
        "--target-block-direct-state-commit",
        "--target-block-min-rows",
        "2",
        "--adaptive-block-after-full-accept",
        "--fused-b1-block-probe",
        "--adaptive-probe-draft-n-max",
        "1",
        "--draft-p-min",
        "0.5",
        "--mtp-draft-vocab-cap",
        "32768",
    ]


def test_suite_llama_compat_dry_run_defaults_to_b2(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "full",
            "--mtp-route",
            "llama-compat-dp4a",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--budgets 2" in out
    assert "--extra-arg=--llama-compat" in out
    assert "--extra-arg=--verify-dp4a" in out
    assert '"budgets": [\n    2\n  ]' in out
    assert '"mtp_route_default_budgets": [\n    2\n  ]' in out


def test_suite_llama_compat_device_chain_dry_run_defaults_to_b2(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "full",
            "--mtp-route",
            "llama-compat-device-chain-dp4a",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--budgets 2" in out
    assert "--extra-arg=--llama-compat" in out
    assert "--extra-arg=--resident-mtp-device-chain" in out
    assert "--extra-arg=--verify-dp4a" in out
    assert '"budgets": [\n    2\n  ]' in out
    assert '"mtp_route_default_budgets": [\n    2\n  ]' in out


def test_suite_llama_compat_q6top1dp4a_x8q6_dry_run_defaults_to_b2(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "full",
            "--mtp-route",
            "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--budgets 2" in out
    assert "--extra-arg=--llama-compat" in out
    assert "--extra-arg=--resident-mtp-device-chain" in out
    assert "--extra-arg=--verify-dp4a" in out
    assert "--extra-arg=--resident-mtp-draft-q6-top1-dp4a" in out
    assert "--extra-arg=--selected-down-x8-repack" in out
    assert "--extra-arg=q6" in out
    assert '"mtp_route": "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6"' in out
    assert '"budgets": [\n    2\n  ]' in out
    assert '"mtp_route_default_budgets": [\n    2\n  ]' in out


def test_suite_llama_compat_q6top1dp4a_x8q6_x8gateup_dry_run_defaults_to_b2(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "full",
            "--mtp-route",
            "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--budgets 2" in out
    assert "--extra-arg=--llama-compat" in out
    assert "--extra-arg=--resident-mtp-device-chain" in out
    assert "--extra-arg=--verify-dp4a" in out
    assert "--extra-arg=--resident-mtp-draft-q6-top1-dp4a" in out
    assert "--extra-arg=--selected-down-x8-repack" in out
    assert "--extra-arg=q6" in out
    assert "--extra-arg=--selected-gate-up-x8" in out
    assert '"mtp_route": "llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup"' in out
    assert '"budgets": [\n    2\n  ]' in out
    assert '"mtp_route_default_budgets": [\n    2\n  ]' in out


def test_suite_llama_compat_device_chain_draftsync_dry_run_defaults_to_b2(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "full",
            "--mtp-route",
            "llama-compat-device-chain-dp4a-draftsync",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--budgets 2" in out
    assert "--extra-arg=--llama-compat" in out
    assert "--extra-arg=--resident-mtp-device-chain" in out
    assert "--extra-arg=--resident-mtp-draft-sync-stage-timings" in out
    assert "--extra-arg=--verify-dp4a" in out
    assert '"budgets": [\n    2\n  ]' in out
    assert '"mtp_route_default_budgets": [\n    2\n  ]' in out


def test_suite_llama_compat_device_seed_chain_dry_run_defaults_to_b2(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "full",
            "--mtp-route",
            "llama-compat-device-seed-chain-dp4a",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--budgets 2" in out
    assert "--extra-arg=--llama-compat" in out
    assert "--extra-arg=--resident-mtp-device-seed" in out
    assert "--extra-arg=--resident-mtp-device-chain" in out
    assert "--extra-arg=--verify-dp4a" in out
    assert '"budgets": [\n    2\n  ]' in out
    assert '"mtp_route_default_budgets": [\n    2\n  ]' in out


def test_suite_llama_compat_rejects_non_b2_budget_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "smoke",
            "--mtp-route",
            "llama-compat",
            "--budgets",
            "5",
            "--raw-root",
            str(tmp_path / "raw"),
            "--dry-run",
        ],
    )

    with pytest.raises(suite.SuiteError, match="fixed to budgets \\[2\\]"):
        suite.main()


def test_suite_dry_run_forwards_cycle_stage_timing_flag(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gguf_ar_mtp_suite.py",
            "--scope",
            "smoke",
            "--raw-root",
            str(tmp_path / "raw"),
            "--record-cycle-stage-timings",
            "--dry-run",
        ],
    )

    assert suite.main() == 0

    out = capsys.readouterr().out
    assert "--extra-arg=--record-cycle-stage-timings" in out
    assert '"record_cycle_stage_timings": true' in out
