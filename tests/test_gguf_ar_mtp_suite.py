from __future__ import annotations

from scripts import gguf_ar_mtp_suite as suite


def test_suite_default_mtp_route_is_current_speed_row() -> None:
    assert suite.DEFAULT_MTP_ROUTE == "resident-b1-probe-block-direct-cap32k-minrows2"
    assert suite.DEFAULT_MTP_ROUTE in suite.MTP_ROUTES


def test_suite_default_route_enables_b2_block_verify() -> None:
    # the default must carry --target-block-min-rows 2 so B2 (3-row) block verify
    # amortizes instead of falling to serial; that is the retained speed win.
    flags = suite.MTP_ROUTES[suite.DEFAULT_MTP_ROUTE]
    assert "--target-block-min-rows" in flags
    assert flags[flags.index("--target-block-min-rows") + 1] == "2"


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
