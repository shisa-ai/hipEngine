from __future__ import annotations

from scripts import gguf_ar_mtp_suite as suite


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


def test_suite_exposes_cap32k_recovery_route() -> None:
    assert suite.MTP_ROUTES["resident-cap32k-recover"] == [
        "--resident-mtp-draft",
        "--adaptive-ar-fallback",
        "--no-target-block-verify",
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
