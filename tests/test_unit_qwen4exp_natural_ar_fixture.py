#!/usr/bin/env python3
"""Validation tests for the PF-0 natural-text route-covering fixture.

Covers the PF-0 exit conditions from
``docs/QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md`` section 6:

- fixture committed with construction provenance and suite hashes,
- route-engagement coverage >= 50% recorded by
  ``scripts/qwen4exp_route_coverage_finding.py``,
- unique-token ratio recorded per case,
- no measurement claim.

These are pure data/JSON checks and require no GPU. They are additionally
guarded on fixture presence so no-fixture runners skip instead of failing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "natural_sources"
FIXTURE_PATH = REPO_ROOT / "benchmarks" / "fixtures" / "qwen4exp_natural_ar_pf0.json"
COVERAGE_ARTIFACT = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "2026-09-03-gfx1151-qwen38-flash-next-pf0-natural-fixture-route-coverage.json"
)

CANONICAL_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")

# Route-covering floor: the full chat prompt must reach the rows >= 64 Q8 MMQ
# policy threshold.
ROUTE_FLOOR = 512


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_ids_sha256(token_ids) -> str:
    return hashlib.sha256(
        np.asarray(token_ids, dtype="<i8").tobytes()
    ).hexdigest()


@pytest.fixture(scope="module")
def fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip("PF-0 natural fixture not built")
    return json.loads(FIXTURE_PATH.read_text())


def test_fixture_exists() -> None:
    assert FIXTURE_PATH.exists()


def test_schema_and_categories(fixture: dict) -> None:
    assert int(fixture["schema"]) == 1
    assert fixture["categories"] == list(CANONICAL_CATEGORIES)
    assert int(fixture["decode_transitions"]) == 128


def test_construction_provenance_and_suite_hashes(fixture: dict) -> None:
    source = fixture["source"]
    assert source["provenance"].endswith("PROVENANCE.md")
    assert (REPO_ROOT / source["provenance"]).exists()
    assert source["chat_prompt_builder"] == (
        "scripts.gguf_mtp_bench.build_chat_prompt(reasoning='off')"
    )
    # construction provenance: every case references a committed body file
    # whose on-disk hash matches the recorded suite hash.
    for case in fixture["cases"]:
        body_path = REPO_ROOT / case["source_file"]
        assert body_path.exists(), case["source_file"]
        assert _sha256_path(body_path) == case["source_sha256"]
        assert case["source_file"] in source_bodies(fixture)


def source_bodies(fixture: dict) -> dict:
    return fixture["source"]["bodies"]


def test_hashes_match_token_arrays(fixture: dict) -> None:
    for case in fixture["cases"]:
        token_ids = case["prompt_token_ids"]
        assert len(token_ids) == case["prompt_tokens"]
        assert _token_ids_sha256(token_ids) == case["prompt_token_ids_sha256"]


def test_all_cases_route_covering(fixture: dict) -> None:
    assert all(
        case["prompt_tokens"] >= ROUTE_FLOOR
        for case in fixture["cases"]
    )


def test_unique_token_ratio_recorded_per_case(fixture: dict) -> None:
    for case in fixture["cases"]:
        ratio = case["unique_token_ratio"]
        assert 0.0 < ratio <= 1.0
        ids = case["prompt_token_ids"]
        assert abs(ratio - len(set(ids)) / len(ids)) < 5e-4


def test_coverage_artifact_route_covering() -> None:
    if not COVERAGE_ARTIFACT.exists():
        pytest.skip("coverage artifact not generated yet")
    artifact = json.loads(COVERAGE_ARTIFACT.read_text())
    cov = artifact["coverage"]["single_fixture"]
    assert cov["cases"] == 12
    assert cov["route_engagement_coverage"] >= 0.5
    assert artifact["verdict"]["single_fixture_classification"] == "route_covering"


def test_no_measurement_claim(fixture: dict) -> None:
    text = FIXTURE_PATH.read_text()
    assert "tok/s" not in text
    assert "performance_claim" not in text or (
        json.loads(text).get("performance_claim", False) is False
    )
