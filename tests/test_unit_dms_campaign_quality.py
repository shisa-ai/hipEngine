"""Pure scoring and manifest tests for the frozen DMS selector campaign evaluator.

These tests cover the Phase A G0/G1/G2 gate arithmetic, per-category scope
enforcement, sequence-count requirements, mismatch diagnostics, and repeated
prefix correlation reporting.  No GPU, model, or capture work runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.dms_campaign import (
    CATEGORIES,
    GATE_THRESHOLDS,
    compare_row,
    evaluate_gate,
    load_manifest,
    sequence_correlations,
)
from scripts.qwen38_dms_campaign_quality import build_parser


CATEGORIES_LIST = list(CATEGORIES)


def _row(
    category: str = "code",
    *,
    sequence_id: str = "selector-qualification-code-00",
    step: int | None = 0,
    phase: str = "decode",
    kl: float = 0.0005,
    top1_agrees: bool = True,
    finite: bool = True,
) -> dict:
    return {
        "sequence_id": sequence_id,
        "category": category,
        "step": step,
        "phase": phase,
        "kl": kl if finite else float("nan"),
        "top1_agrees": top1_agrees,
        "finite_candidate_logits": finite,
        "teacher_top1": 1,
        "candidate_top1": 1 if top1_agrees else 2,
    }


def _prefill_row(category: str = "code", **kwargs) -> dict:
    return _row(category, phase="prefill", step=None, **kwargs)


def _balanced_decode_rows(kl: float = 0.0005, top1_agrees: bool = True, steps: int = 8, finite: bool = True) -> list[dict]:
    rows = []
    for category in CATEGORIES_LIST:
        for step in range(steps):
            rows.append(
                _row(
                    category,
                    sequence_id=f"selector-qualification-{category}-00",
                    step=step,
                    kl=kl,
                    top1_agrees=top1_agrees,
                    finite=finite,
                )
            )
    return rows


def _balanced_prefill_rows(**kwargs) -> list[dict]:
    return [_prefill_row(category, sequence_id=f"selector-qualification-{category}-00", **kwargs) for category in CATEGORIES_LIST]


def _verdict(gate: str, *, prefill: list[dict], decode: list[dict]) -> dict:
    return evaluate_gate(
        gate,
        prefill_rows=prefill,
        decode_rows=decode,
        categories=CATEGORIES_LIST,
        correlations=sequence_correlations(_manifest_sequences()),
    )


def _manifest_sequences() -> list[dict]:
    return [
        {
            "sequence_id": f"selector-qualification-{category}-00",
            "category": category,
            "split": "qualification",
            "token_ids": list(range(1000 + index, 1000 + index + 64)),
        }
        for index, category in enumerate(CATEGORIES_LIST)
    ]


# ---------------------------------------------------------------------------
# Gate G2


def test_g2_passes_when_all_scopes_meet_every_threshold() -> None:
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=_balanced_decode_rows())
    assert verdict["passed"] is True
    assert verdict["failures"] == []
    assert set(verdict["thresholds"]["decode"]) == {
        "mean_kl", "p95_kl", "p99_kl", "max_kl", "min_top1_global", "min_top1_per_category",
    }


def test_g2_decode_mean_kl_threshold_fails_jointly() -> None:
    decode = _balanced_decode_rows(kl=0.0011)
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "mean_kl" and f["scope"] == "global" for f in verdict["failures"])


def test_g2_each_tail_threshold_fails_individually() -> None:
    cases = {
        "mean_kl": [0.0011] * 32,
        "p95_kl": [0.0009] * 30 + [0.0052] * 2,
        "p99_kl": [0.0009] * 31 + [0.0201] * 1,
        "max_kl": [0.0009] * 31 + [0.0501] * 1,
    }
    for check, kls in cases.items():
        decode = []
        for category in CATEGORIES_LIST:
            for step, kl in enumerate(kls):
                decode.append(_row(category, step=step, kl=kl))
        verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=decode)
        assert verdict["passed"] is False, check
        assert any(f["check"] == check and f["scope"] == "global" for f in verdict["failures"]), check


def test_g2_global_top1_and_category_top1_floors_fail_separately() -> None:
    # 99% global floor: two mismatches out of 100 rows.
    decode = _balanced_decode_rows(steps=25)
    for index in (0, 25):
        decode[index] = {**decode[index], "top1_agrees": False, "candidate_top1": 2}
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "min_top1" and f["scope"] == "global" for f in verdict["failures"])
    # 97% per-category floor: two mismatches inside a single category.
    decode = _balanced_decode_rows(steps=8)
    for index in (0, 1):
        decode[index] = {**decode[index], "top1_agrees": False, "candidate_top1": 2}
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=decode)
    assert verdict["passed"] is False
    category_scope = [f for f in verdict["failures"] if f["check"] == "min_top1" and f["scope"] != "global"]
    assert category_scope, verdict["failures"]


def test_g2_per_category_scope_cannot_be_averaged_away() -> None:
    # One failing category with bad KL, other categories perfect.
    decode = _balanced_decode_rows(kl=0.0001)
    for step in range(8):
        decode[step] = _row("general_ja", step=step, kl=0.06)
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "max_kl" and f["scope"] == "category:general_ja" for f in verdict["failures"])


def test_g2_prefill_max_kl_and_finiteness_fail() -> None:
    decode = _balanced_decode_rows()
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(kl=0.051), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "max_kl" and f["phase"] == "prefill" for f in verdict["failures"])
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(finite=False), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "finite_logits" for f in verdict["failures"])


def test_g2_reports_rows_above_kl_0_02_with_identity() -> None:
    decode = _balanced_decode_rows()
    decode[3] = _row("general_en", sequence_id="selector-qualification-general_en-00", step=3, kl=0.03)
    verdict = _verdict("g2", prefill=_balanced_prefill_rows(), decode=decode)
    hot = verdict["rows_above_kl_0_02"]
    assert len(hot) == 1
    assert hot[0]["sequence_id"] == "selector-qualification-general_en-00"
    assert hot[0]["category"] == "general_en"
    assert hot[0]["step"] == 3
    assert hot[0]["phase"] == "decode"


def test_missing_category_rows_fail_every_gate() -> None:
    decode = [row for row in _balanced_decode_rows() if row["category"] != "mixed_ja_en"]
    prefill = [row for row in _balanced_prefill_rows() if row["category"] != "mixed_ja_en"]
    for gate in ("g0", "g1", "g2"):
        verdict = _verdict(gate, prefill=prefill, decode=decode)
        assert verdict["passed"] is False, gate
        assert any(f["check"] == "row_coverage" for f in verdict["failures"]), gate


# ---------------------------------------------------------------------------
# Gate G1


def test_g1_passes_and_fails_per_category() -> None:
    verdict = _verdict("g1", prefill=_balanced_prefill_rows(), decode=_balanced_decode_rows(kl=0.04, steps=4))
    assert verdict["passed"] is True
    decode = _balanced_decode_rows(kl=0.04, steps=4)
    for step in range(4):
        decode[step] = _row("code", step=step, kl=0.0501)
    verdict = _verdict("g1", prefill=_balanced_prefill_rows(), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "max_kl" and f["scope"] == "category:code" for f in verdict["failures"])
    decode = _balanced_decode_rows(kl=0.01, steps=10)
    for index in range(2):
        decode[index] = {**decode[index], "top1_agrees": False, "candidate_top1": 2}
    verdict = _verdict("g1", prefill=_balanced_prefill_rows(), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "min_top1" and f["scope"] == "category:code" for f in verdict["failures"])


def test_g1_prefill_quality_is_binding_per_category() -> None:
    prefill = _balanced_prefill_rows(kl=0.01)
    prefill[0] = {
        **prefill[0],
        "kl": 0.0501,
        "top1_agrees": False,
        "candidate_top1": 2,
    }
    verdict = _verdict("g1", prefill=prefill, decode=_balanced_decode_rows())
    assert verdict["passed"] is False
    assert any(
        failure["phase"] == "prefill"
        and failure["scope"] == "category:code"
        and failure["check"] in {"max_kl", "min_top1"}
        for failure in verdict["failures"]
    )


def test_g1_nonfinite_prefill_or_decode_fails() -> None:
    decode = _balanced_decode_rows()
    for finite_prefill in (True, False):
        verdict = _verdict("g1", prefill=_balanced_prefill_rows(finite=finite_prefill), decode=decode)
        assert verdict["passed"] is finite_prefill
    verdict = _verdict("g1", prefill=_balanced_prefill_rows(), decode=_balanced_decode_rows(finite=False))
    assert verdict["passed"] is False


# ---------------------------------------------------------------------------
# Gate G0


def test_g0_requires_tight_kl_and_perfect_top1() -> None:
    verdict = _verdict("g0", prefill=_balanced_prefill_rows(kl=0.0005), decode=_balanced_decode_rows(kl=0.0005))
    assert verdict["passed"] is True
    verdict = _verdict("g0", prefill=_balanced_prefill_rows(kl=0.0011), decode=_balanced_decode_rows(kl=0.0005))
    assert verdict["passed"] is False
    assert any(f["check"] == "max_kl" for f in verdict["failures"])
    decode = _balanced_decode_rows(kl=0.0005)
    decode[7] = {**decode[7], "top1_agrees": False, "candidate_top1": 2}
    verdict = _verdict("g0", prefill=_balanced_prefill_rows(kl=0.0005), decode=decode)
    assert verdict["passed"] is False
    assert any(f["check"] == "min_top1" for f in verdict["failures"])


# ---------------------------------------------------------------------------
# Mismatch diagnostics


def test_compare_row_emits_full_mismatch_diagnostics() -> None:
    teacher = np.zeros(64, dtype=np.float32)
    teacher[3] = 5.0
    teacher[7] = 4.0
    candidate = np.zeros(64, dtype=np.float32)
    candidate[7] = 5.0
    candidate[3] = 4.5
    candidate[11] = 3.0
    row = compare_row(teacher, candidate)
    assert row["top1_agrees"] is False
    assert row["teacher_top1"] == 3 and row["candidate_top1"] == 7
    assert row["winners"] == {"teacher": 3, "candidate": 7}
    assert row["strict_margin"] > 0.0
    assert row["candidate_rank_of_strict_winner"] == 1
    assert 0.0 < row["topk_overlap"] < 1.0
    assert row["max_abs_logit_delta"] == pytest.approx(3.0, abs=1e-5)
    assert row["finite_candidate_logits"] is True
    assert row["kl"] > 0.0


def test_compare_row_flags_nonfinite_candidates() -> None:
    teacher = np.zeros(8, dtype=np.float32)
    candidate = np.zeros(8, dtype=np.float32)
    candidate[4] = float("inf")
    row = compare_row(teacher, candidate)
    assert row["finite_candidate_logits"] is False
    assert not np.isfinite(row["kl"])


def test_every_top1_mismatch_is_diagnosed_in_the_verdict() -> None:
    decode = _balanced_decode_rows(steps=4)
    decode[2] = {**decode[2], "top1_agrees": False, "candidate_top1": 9,
                  "strict_margin": 0.3, "candidate_rank_of_strict_winner": 4,
                  "topk_overlap": 0.5, "max_abs_logit_delta": 1.2}
    verdict = _verdict("g1", prefill=_balanced_prefill_rows(), decode=decode)
    mismatches = verdict["top1_mismatches"]
    assert len(mismatches) == 1
    entry = mismatches[0]
    assert entry["sequence_id"] == decode[2]["sequence_id"]
    assert entry["category"] == decode[2]["category"]
    assert entry["step"] == decode[2]["step"]
    assert entry["phase"] == "decode"
    assert entry["diagnostics"]["candidate_rank_of_strict_winner"] == 4
    assert entry["diagnostics"]["strict_margin"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Manifest loading and sequence-count enforcement


def _campaign_manifest(path: Path, *, per_category: int = 2) -> Path:
    sequences = []
    for category in CATEGORIES_LIST:
        for index in range(per_category):
            sequences.append(
                {
                    "sequence_id": f"selector-qualification-{category}-{index:02d}",
                    "category": category,
                    "split": "qualification",
                    "token_ids": [index + 7] * 32,
                    "provenance": {
                        "source_id": f"source-{category}-{index}",
                        "normalized_text_sha256": f"{index:064d}",
                    },
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "hipengine_dms_selector_campaign_manifest",
                "split": "qualification",
                "length_tokens": 32,
                "count": len(sequences),
                "sequences": sequences,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_manifest_enforces_expected_sequences_per_category(tmp_path: Path) -> None:
    manifest = _campaign_manifest(tmp_path / "qualification-data.json", per_category=2)
    sequences = load_manifest(
        manifest, split="qualification", categories=CATEGORIES_LIST, expected_sequences_per_category=2
    )
    assert len(sequences) == 8
    assert {s["category"] for s in sequences} == set(CATEGORIES_LIST)
    assert all(s["split"] == "qualification" for s in sequences)
    with pytest.raises(ValueError, match="expected 3"):
        load_manifest(
            manifest, split="qualification", categories=CATEGORIES_LIST, expected_sequences_per_category=3
        )
    with pytest.raises(ValueError, match="split"):
        load_manifest(
            manifest, split="final-32k", categories=CATEGORIES_LIST, expected_sequences_per_category=2
        )
    with pytest.raises(ValueError, match="unsupported category"):
        load_manifest(manifest, split="qualification", categories=("code", "fictional"), expected_sequences_per_category=2)


def test_load_manifest_rejects_sequences_missing_token_ids(tmp_path: Path) -> None:
    manifest = _campaign_manifest(tmp_path / "m.json", per_category=1)
    payload = json.loads(manifest.read_text())
    del payload["sequences"][1]["token_ids"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="token_ids"):
        load_manifest(manifest, split="qualification", categories=CATEGORIES_LIST, expected_sequences_per_category=1)


# ---------------------------------------------------------------------------
# Correlated prefixes are not independent prompts


def test_repeated_prefix_rows_reported_as_correlated_not_independent() -> None:
    shared = [11, 12, 13, 14, 15, 16, 17, 18]
    sequences = _manifest_sequences()
    sequences[0] = {**sequences[0], "token_ids": shared + [91, 92]}
    sequences[1] = {**sequences[1], "token_ids": shared + [93]}
    report = sequence_correlations(sequences)
    assert report["sequence_count"] == 4
    assert report["independent_source_groups"] == 3
    assert len(report["correlated_groups"]) == 1
    group = report["correlated_groups"][0]
    assert set(group["members"]) == {sequences[0]["sequence_id"], sequences[1]["sequence_id"]}
    assert group["shared_prefix_tokens"] >= 8
    verdict = evaluate_gate(
        "g1",
        prefill_rows=_balanced_prefill_rows(),
        decode_rows=_balanced_decode_rows(),
        categories=CATEGORIES_LIST,
        correlations=report,
    )
    assert verdict["row_counts"]["independent_source_groups"] == 3
    assert verdict["row_counts"]["sequence_count"] == 4
    assert "not independent" in verdict["row_counts"]["correlation_note"]


def test_identical_text_hashes_are_correlated_even_with_different_tokens() -> None:
    sequences = _manifest_sequences()
    sequences[0] = {
        **sequences[0],
        "token_ids": [1, 2, 3],
        "normalized_text_sha256": "d" * 64,
    }
    sequences[1] = {
        **sequences[1],
        "token_ids": [9, 9, 9, 9],
        "normalized_text_sha256": "d" * 64,
    }
    report = sequence_correlations(sequences)
    assert report["independent_source_groups"] == 3


# ---------------------------------------------------------------------------
# CLI surface


def test_cli_supports_gate_and_expected_sequence_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model", "m.gguf",
            "--metadata", "meta.json",
            "--data-manifest", "data.json",
            "--split", "qualification",
            "--gate", "g2",
            "--expected-sequences-per-category", "2",
            "--decode-steps", "32",
            "--modes", "no_evict,sidecar",
            "--codec", "bf16",
            "--backend", "hip_gfx1151",
            "--output", "out.json",
        ]
    )
    assert args.gate == "g2"
    assert args.expected_sequences_per_category == 2
    assert args.modes == "no_evict,sidecar"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m.gguf", "--metadata", "meta.json",
                           "--data-manifest", "d.json", "--output", "o.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m.gguf", "--metadata", "meta.json",
                           "--data-manifest", "d.json", "--output", "o.json",
                           "--expected-sequences-per-category", "1", "--gate", "g9"])


def test_gate_thresholds_are_frozen_constants() -> None:
    g2 = GATE_THRESHOLDS["g2"]
    assert g2["decode"]["mean_kl"] == 0.001
    assert g2["decode"]["p95_kl"] == 0.005
    assert g2["decode"]["p99_kl"] == 0.02
    assert g2["decode"]["max_kl"] == 0.05
    assert g2["decode"]["min_top1_global"] == 0.99
    assert g2["decode"]["min_top1_per_category"] == 0.97
    assert g2["prefill"]["max_kl"] == 0.05
    g1 = GATE_THRESHOLDS["g1"]
    assert g1["decode"]["max_kl"] == 0.05
    assert g1["decode"]["min_top1_per_category"] == 0.9
    assert g1["prefill"] == {"max_kl": 0.05, "min_top1_per_category": 0.9}
    g0 = GATE_THRESHOLDS["g0"]
    assert g0["combined"]["max_kl"] == 0.001
    assert g0["combined"]["min_top1"] == 1.0
