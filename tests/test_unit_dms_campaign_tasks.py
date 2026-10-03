"""Pure tests for the frozen G3 long-context task generator/evaluator.

These tests cover the 24-case matrix, exact target lengths and recorded
positions, placement distance rules, answer uniqueness/absence, language and
category templates, correlated filler reporting, answer parsing, G3 scope
gating with dense-failure retention and identity rejection, and the
non-binding free-running metrics.  No GPU, model, or network access runs
here; tokenization is injected as a deterministic fake.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.benchmark.dms_campaign import CATEGORIES
from hipengine.benchmark.dms_tasks import (
    DEFAULT_FREE_RUNNING_TOKENS,
    DENSE_ARM,
    TASK_MANIFEST_KIND,
    TASK_RUN_KIND,
    case_coordinates,
    case_entities,
    case_texts,
    compare_g3,
    filler_correlations,
    free_running_metrics,
    parse_answer,
    validate_case,
)
from hipengine.benchmark.dms_tasks import build_task_manifest as _build_manifest
from scripts.qwen38_dms_campaign_tasks import (
    _greedy_generate,
    _validate_dense_reference,
    _validate_task_manifest,
    _write_output,
    build_parser,
)


TARGET = 4096
SEED = 0
SPLIT = "qualification"


def fake_tokenize(text: str) -> list[int]:
    """Deterministic word-level tokenizer stable across processes."""

    def word_id(word: str) -> int:
        return 1 + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % 999983

    return [word_id(word) for word in re.findall(r"\S+", str(text))]


def synthetic_pool(target_tokens: int = TARGET) -> list[dict]:
    """One filler source per category with room for a full case window."""
    pool = []
    for index, category in enumerate(CATEGORIES):
        length = target_tokens + 512
        pool.append(
            {
                "sequence_id": f"{SPLIT}-{category}-00",
                "source_id": f"src-{category}",
                "category": category,
                "normalized_text_sha256": f"norm-{category}",
                "token_ids": [2_000_000 + index * 200_000 + i for i in range(length)],
            }
        )
    return pool


def build_manifest(
    *, seed: int = SEED, target_tokens: int = TARGET, pool: list[dict] | None = None
) -> dict:
    return _build_manifest(
        pool=pool if pool is not None else synthetic_pool(target_tokens),
        suite="qualification",
        target_tokens=target_tokens,
        seed=seed,
        tokenize=fake_tokenize,
        data_manifest={"path": "/sealed/data.json", "sha256": "d" * 64, "split": SPLIT},
        model={"path": "/sealed/model.gguf", "sha256": "m" * 64},
    )


def make_result(
    manifest: dict,
    *,
    arm: str = "sidecar",
    metadata_sha: str | None = "e" * 64,
    incorrect_cases: set[str] = frozenset(),
    task_manifest_sha: str = "t" * 64,
    model_sha: str = "m" * 64,
    evaluator_sha: str = "l" * 64,
    kind: str = TASK_RUN_KIND,
) -> dict:
    return {
        "kind": kind,
        "arm": arm,
        "metadata": {"sha256": metadata_sha} if metadata_sha else None,
        "model": {"sha256": model_sha},
        "task_manifest": {"sha256": task_manifest_sha},
        "evaluator": {
            "library_sha256": evaluator_sha,
            "script_sha256": "s" * 64,
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "category": case["category"],
                "placement": case["placement"],
                "target_tokens": case["target_tokens"],
                "answer_token_ids_sha256": case["answer_token_ids_sha256"],
                "prompt_token_ids_sha256": case["token_ids_sha256"],
                "correct": case["case_id"] not in incorrect_cases,
                "parser": {"verdict": "correct" if case["case_id"] not in incorrect_cases else "incorrect"},
                "generated_text": "x" if case["case_id"] not in incorrect_cases else "",
            }
            for case in manifest["cases"]
        ],
    }


# ---------------------------------------------------------------------------
# Case matrix, lengths, and positions


def test_case_matrix_is_exactly_24_in_canonical_order() -> None:
    coordinates = case_coordinates()
    assert len(coordinates) == 24
    assert len(set(coordinates)) == 24
    assert set(family for family, _, _ in coordinates) == {
        "retrieval", "two_hop", "variable_state"
    }
    assert set(category for _, category, _ in coordinates) == set(CATEGORIES)
    assert set(placement for _, _, placement in coordinates) == {"early", "recent"}
    manifest = build_manifest()
    assert manifest["kind"] == TASK_MANIFEST_KIND
    assert manifest["case_count"] == 24
    assert [case["case_index"] for case in manifest["cases"]] == list(range(24))
    assert [case["case_id"] for case in manifest["cases"]] == [
        f"qualification-{TARGET}-{family}-{category}-{placement}"
        for family, category, placement in coordinates
    ]


def test_every_case_hits_exact_target_length_and_recorded_positions() -> None:
    manifest = build_manifest()
    for case in manifest["cases"]:
        assert len(case["token_ids"]) == TARGET
        assert validate_case(case) == []
        dep = case["dependency_position"]
        query = case["query_position"]
        assert case["token_ids"][dep["start"] : dep["end"] + 1] == case["dependency_token_ids"]
        assert case["token_ids"][query["start"] : query["end"] + 1] == case["query_token_ids"]
        assert query["end"] == TARGET - 1
        assert case["placement_check"]["passed"] is True


def test_placement_distances_respect_w256_and_recent_window() -> None:
    manifest = build_manifest()
    for case in manifest["cases"]:
        dep = case["dependency_position"]
        distance = TARGET - 1 - dep["end"]
        if case["placement"] == "early":
            assert distance > 256
            assert TARGET - 1 - dep["start"] >= 128
        else:
            assert TARGET - 1 - dep["start"] < 128
            assert 0 <= distance < 128


def test_manifest_at_32768_hits_exact_frozen_length() -> None:
    manifest = build_manifest(target_tokens=32768)
    assert manifest["target_tokens"] == 32768
    assert all(len(case["token_ids"]) == 32768 for case in manifest["cases"])
    assert all(validate_case(case) == [] for case in manifest["cases"])


# ---------------------------------------------------------------------------
# Determinism and answers


def test_build_is_deterministic_and_seed_sensitive() -> None:
    first = build_manifest(seed=0)
    second = build_manifest(seed=0)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    other = build_manifest(seed=1)
    assert first["cases"][0]["token_ids"] != other["cases"][0]["token_ids"]
    assert {case["answer_text"] for case in first["cases"]}.isdisjoint(
        {case["answer_text"] for case in other["cases"]}
    )


def test_answers_are_unique_nonces_absent_from_filler_and_queries() -> None:
    manifest = build_manifest()
    answers = [case["answer_text"] for case in manifest["cases"]]
    assert len(set(answers)) == 24
    nonce_re = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}$")
    for case in manifest["cases"]:
        assert nonce_re.match(case["answer_text"])
        assert case["answer_absent_from_filler"] is True
        assert case["answer_text"] in case["dependency_text"]
        assert case["answer_text"] not in case["query_text"]
        answer_ids = case["answer_token_ids"]
        haystack = case["filler_token_ids"]
        for start in range(len(haystack) - len(answer_ids) + 1):
            assert haystack[start : start + len(answer_ids)] != answer_ids


def test_validate_case_rejects_moved_dependency_and_leaks() -> None:
    manifest = build_manifest()
    case = json.loads(json.dumps(manifest["cases"][0]))
    assert validate_case(case) == []
    moved = json.loads(json.dumps(case))
    moved["dependency_position"] = {
        "start": TARGET - len(moved["query_token_ids"]) - len(moved["dependency_token_ids"]),
        "end": TARGET - len(moved["query_token_ids"]) - 1,
    }
    assert validate_case(moved) != []
    leaky = json.loads(json.dumps(case))
    leaky["query_text"] = leaky["query_text"] + " " + leaky["answer_text"]
    leaky["query_token_ids"] = list(case["query_token_ids"]) + list(case["answer_token_ids"])
    leaky["query_position"] = {
        "start": case["query_position"]["start"],
        "end": case["query_position"]["end"] + len(case["answer_token_ids"]),
    }
    leaky["token_ids"] = case["token_ids"][: case["query_position"]["end"] + 1] + list(
        case["answer_token_ids"]
    )
    problems = validate_case(leaky)
    assert any("answer nonce leaks" in problem or "answer token sequence" in problem for problem in problems)


# ---------------------------------------------------------------------------
# Language and category templates


def test_language_and_category_templates_are_explicit_and_distinct() -> None:
    manifest = build_manifest()
    seen_texts = set()
    for case in manifest["cases"]:
        dep, query = case["dependency_text"], case["query_text"]
        assert dep and query
        seen_texts.add((case["family"], case["category"], dep, query))
        if case["category"] == "general_ja":
            assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", dep + query)
        elif case["category"] == "mixed_ja_en":
            assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", dep + query)
            assert re.search(r"[A-Za-z]", dep + query)
        elif case["category"] == "code":
            assert "const" in dep or "//" in dep
        assert case["answer_text"] in dep
    # every (family, category) pair has its own frozen template
    assert len(seen_texts) == 24


def test_case_texts_reject_unknown_inputs_and_entities_are_deterministic() -> None:
    with pytest.raises(ValueError):
        case_texts("retrieval", "not_a_category", {"entity": "E", "answer": "A"})
    with pytest.raises(ValueError):
        case_texts("not_a_family", "code", {})
    entities = case_entities("two_hop", 12345)
    assert entities == case_entities("two_hop", 12345)
    assert entities["gateway"] != entities["answer"]


# ---------------------------------------------------------------------------
# Correlated filler reporting


def test_reused_filler_sources_are_reported_as_correlated() -> None:
    manifest = build_manifest()
    correlations = manifest["filler_correlations"]
    assert correlations["case_count"] == 24
    # all six cases of one category share the single synthetic source
    assert correlations["independent_case_groups"] < 24
    assert correlations["correlated_groups"]
    for group in correlations["correlated_groups"]:
        assert len(group["members"]) >= 2
        assert group["shared_source_ids"]
    # filler_slices record provenance against the assigned split only
    for case in manifest["cases"]:
        for slice_record in case["filler_slices"]:
            assert slice_record["sequence_id"].startswith(SPLIT)
            assert slice_record["source_id"].startswith("src-")
            assert slice_record["length"] > 0
            assert slice_record["end"] >= slice_record["start"]
    assert "correlated" in correlations["note"]


def test_filler_correlations_reports_shared_prefix_groups() -> None:
    cases = [
        {
            "case_id": f"case-{i}",
            "filler_slices": [{"sequence_id": "shared"}],
            "filler_token_ids": [7, 8, 9, 10, 11, 12] * 16,
        }
        for i in range(3)
    ] + [
        {
            "case_id": "case-lone",
            "filler_slices": [{"sequence_id": "other"}],
            "filler_token_ids": [99, 98, 97] * 16,
        }
    ]
    correlations = filler_correlations(cases)
    assert correlations["independent_case_groups"] == 2
    assert len(correlations["correlated_groups"]) == 1
    assert sorted(correlations["correlated_groups"][0]["members"]) == [
        "case-0", "case-1", "case-2",
    ]


# ---------------------------------------------------------------------------
# Answer parsing


def test_parse_answer_positives_and_negatives() -> None:
    verdict = parse_answer("  K7QF-2M9X\n", "K7QF-2M9X")
    assert verdict["correct"] is True
    assert verdict["verdict"] == "correct"
    assert verdict["answer_index"] == 2
    prose = parse_answer("The key is K7QF-2M9X exactly.", "K7QF-2M9X")
    assert prose["answer_found"] is True
    assert prose["correct"] is False
    wrong = parse_answer("The key is 0000-0000.", "K7QF-2M9X")
    assert wrong["correct"] is False
    assert wrong["verdict"] == "incorrect"
    assert wrong["answer_index"] is None
    partial = parse_answer("K7QF-2M9", "K7QF-2M9X")
    assert partial["correct"] is False


# ---------------------------------------------------------------------------
# G3 comparison


def _g3_verdict(dense_bad: set[str], candidate_bad: set[str], baseline_bad: set[str] = frozenset()):
    manifest = build_manifest()
    dense = make_result(manifest, arm=DENSE_ARM, metadata_sha=None, incorrect_cases=dense_bad)
    baseline = make_result(manifest, arm="sidecar", metadata_sha="b" * 64, incorrect_cases=baseline_bad)
    candidate = make_result(manifest, incorrect_cases=candidate_bad)
    return compare_g3(dense, baseline, candidate)


def test_runner_helpers_bind_identity_schedule_and_immutable_outputs(tmp_path: Path) -> None:
    class Session:
        def __init__(self) -> None:
            self.steps: list[int] = []

        def prefill(self, prompt, **kwargs):
            assert prompt == [9]
            return SimpleNamespace(token_id=10)

        def step(self, token):
            self.steps.append(token)
            return SimpleNamespace(token_id=token + 1)

    session = Session()
    assert _greedy_generate(session, [9], max_new_tokens=3, eos_token_id=None) == [10, 11, 12]
    assert session.steps == [10, 11]
    eos_session = Session()
    assert _greedy_generate(eos_session, [9], max_new_tokens=5, eos_token_id=11) == [10, 11]
    assert eos_session.steps == [10]

    manifest = build_manifest(target_tokens=32768)
    _validate_task_manifest(manifest, model_sha256="m" * 64)
    with pytest.raises(ValueError, match="model hash"):
        _validate_task_manifest(manifest, model_sha256="x" * 64)
    duplicate = json.loads(json.dumps(manifest))
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="case IDs"):
        _validate_task_manifest(duplicate, model_sha256="m" * 64)

    reference = {
        "kind": "hipengine_qwen38_dms_campaign_free_running_run",
        "arm": "dense",
        "model": {"sha256": "m" * 64},
        "data_manifest": {"sha256": "d" * 64},
        "schedule": {"free_running_tokens": 256},
    }
    _validate_dense_reference(
        reference,
        model_sha256="m" * 64,
        data_manifest_sha256="d" * 64,
        free_running_tokens=256,
    )
    with pytest.raises(ValueError, match="schedule"):
        _validate_dense_reference(
            reference,
            model_sha256="m" * 64,
            data_manifest_sha256="d" * 64,
            free_running_tokens=128,
        )

    output = tmp_path / "result.json"
    _write_output(output, {"ok": True})
    assert json.loads(output.read_text()) == {"ok": True}
    with pytest.raises(FileExistsError):
        _write_output(output, {"ok": False})


def test_g3_passes_when_candidate_meets_dense_everywhere() -> None:
    verdict = _g3_verdict(frozenset(), frozenset())
    assert verdict["passed"] is True
    assert verdict["status"] == "passed"
    assert verdict["errors"] == []
    assert verdict["failures"] == []
    assert verdict["dense_failures"] == []
    assert verdict["baseline_note"].startswith("frozen baseline DMS arm reported separately")
    assert verdict["tested_lengths"] == [TARGET]


def test_g3_fails_per_task_family_scope() -> None:
    manifest = build_manifest()
    family_case = next(
        case["case_id"] for case in manifest["cases"] if case["family"] == "two_hop"
    )
    verdict = _g3_verdict(frozenset(), {family_case})
    assert verdict["passed"] is False
    assert any(
        failure["scope"] == "family" and failure["key"] == "two_hop"
        for failure in verdict["failures"]
    )


def test_g3_fails_per_category_and_placement_scopes() -> None:
    manifest = build_manifest()
    category_case = next(
        case["case_id"] for case in manifest["cases"] if case["category"] == "general_ja"
    )
    verdict = _g3_verdict(frozenset(), {category_case})
    assert any(
        failure["scope"] == "category" and failure["key"] == "general_ja"
        for failure in verdict["failures"]
    )
    placement_case = next(
        case["case_id"] for case in manifest["cases"] if case["placement"] == "recent"
    )
    verdict = _g3_verdict(frozenset(), {placement_case})
    assert any(
        failure["scope"] == "placement" and failure["key"] == "recent"
        for failure in verdict["failures"]
    )


def test_g3_fails_per_tested_length_scope() -> None:
    manifest = build_manifest()
    case = manifest["cases"][0]
    dense = make_result(manifest, arm=DENSE_ARM, metadata_sha=None)
    baseline = make_result(manifest, arm="sidecar", metadata_sha="b" * 64)
    candidate = make_result(manifest, incorrect_cases={case["case_id"]})
    verdict = compare_g3(dense, baseline, candidate)
    assert verdict["passed"] is False
    assert any(
        failure["scope"] == "target_tokens" for failure in verdict["failures"]
    )


def test_g3_retains_dense_failures_and_paired_deltas() -> None:
    manifest = build_manifest()
    failing = {manifest["cases"][3]["case_id"], manifest["cases"][7]["case_id"]}
    verdict = _g3_verdict(failing, frozenset())
    # candidate >= dense everywhere still passes
    assert verdict["passed"] is True
    assert {entry["case_id"] for entry in verdict["dense_failures"]} == failing
    assert all(
        entry["parser_verdict"] == "incorrect" for entry in verdict["dense_failures"]
    )
    assert all(entry["generated_text"] == "" for entry in verdict["dense_failures"])
    deltas = {entry["case_id"]: entry for entry in verdict["paired_case_deltas"]}
    assert len(deltas) == 24
    assert deltas[manifest["cases"][3]["case_id"]]["delta"] == 1
    assert deltas[manifest["cases"][0]["case_id"]]["delta"] == 0
    assert deltas[manifest["cases"][3]["case_id"]]["baseline_correct"] is True


def test_g3_reports_baseline_separately_without_gating() -> None:
    # baseline fails everything; candidate matches dense; G3 still passes
    manifest = build_manifest()
    all_bad = {case["case_id"] for case in manifest["cases"]}
    verdict = _g3_verdict(frozenset(), frozenset(), baseline_bad=all_bad)
    assert verdict["passed"] is True
    assert verdict["baseline_counts"]["overall"]["all"]["correct"] == 0
    assert verdict["baseline_counts"]["overall"]["all"]["total"] == 24


def test_g3_rejects_identity_and_hash_mismatches() -> None:
    manifest = build_manifest()
    dense = make_result(manifest, arm=DENSE_ARM, metadata_sha=None)
    baseline = make_result(manifest, arm="sidecar", metadata_sha="b" * 64)
    # task-manifest hash mismatch
    candidate = make_result(manifest, task_manifest_sha="z" * 64)
    verdict = compare_g3(dense, baseline, candidate)
    assert verdict["passed"] is False
    assert verdict["status"] == "rejected_identity"
    assert any("task-manifest hash differs" in error for error in verdict["errors"])
    # model hash mismatch
    candidate = make_result(manifest, model_sha="z" * 64)
    verdict = compare_g3(dense, baseline, candidate)
    assert any("model hash differs" in error for error in verdict["errors"])
    # evaluator hash mismatch
    candidate = make_result(manifest, evaluator_sha="z" * 64)
    verdict = compare_g3(dense, baseline, candidate)
    assert any("library_sha256" in error for error in verdict["errors"])
    # wrong arm and missing metadata
    candidate = make_result(manifest, arm=DENSE_ARM, metadata_sha=None)
    verdict = compare_g3(dense, baseline, candidate)
    assert any("arm" in error for error in verdict["errors"])
    candidate = make_result(manifest, metadata_sha=None)
    verdict = compare_g3(dense, baseline, candidate)
    assert any("carries no metadata hash" in error for error in verdict["errors"])
    # dense arm carrying metadata
    dense_bad = make_result(manifest, arm=DENSE_ARM, metadata_sha="x" * 64)
    verdict = compare_g3(dense_bad, baseline, make_result(manifest))
    assert any("must not carry metadata" in error for error in verdict["errors"])
    # wrong result kind
    candidate = make_result(manifest, kind="something_else")
    verdict = compare_g3(dense, baseline, candidate)
    assert any("kind" in error for error in verdict["errors"])


def test_g3_rejects_wrong_arms_duplicate_cases_and_script_hashes() -> None:
    manifest = build_manifest()
    dense = make_result(manifest, arm=DENSE_ARM, metadata_sha=None)
    baseline = make_result(manifest)
    candidate = make_result(manifest)

    wrong_arm = make_result(manifest, arm="no_evict")
    verdict = compare_g3(dense, wrong_arm, candidate)
    assert verdict["status"] == "rejected_identity"
    assert any("required 'sidecar'" in error for error in verdict["errors"])

    duplicate = make_result(manifest)
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    verdict = compare_g3(dense, baseline, duplicate)
    assert any("duplicate case IDs" in error for error in verdict["errors"])

    wrong_case_identity = make_result(manifest)
    wrong_case_identity["cases"][0]["category"] = "general_ja"
    verdict = compare_g3(dense, baseline, wrong_case_identity)
    assert any("field category differs" in error for error in verdict["errors"])

    candidate["evaluator"]["script_sha256"] = "z" * 64
    verdict = compare_g3(dense, baseline, candidate)
    assert any("script_sha256" in error for error in verdict["errors"])


def test_g3_rejects_case_set_mismatch() -> None:
    manifest = build_manifest()
    dense = make_result(manifest, arm=DENSE_ARM, metadata_sha=None)
    baseline = make_result(manifest, arm="sidecar", metadata_sha="b" * 64)
    candidate = make_result(manifest)
    candidate["cases"] = candidate["cases"][:12]
    verdict = compare_g3(dense, baseline, candidate)
    assert verdict["passed"] is False
    assert any("case set differs" in error for error in verdict["errors"])


# ---------------------------------------------------------------------------
# Free-running diagnostic (non-binding)


def test_free_running_metrics_identical_and_divergent() -> None:
    reference = list(range(100, 100 + 256))
    same = free_running_metrics(reference, list(reference))
    assert same["first_divergence"] is None
    assert same["comparable_prefix_tokens"] == 256
    assert same["fixed_length_token_match_rate"] == 1.0
    assert same["non_binding"] is True

    divergent = list(reference)
    divergent[10] = -1
    divergent[200] = -1
    metrics = free_running_metrics(divergent, reference)
    assert metrics["first_divergence"] == 10
    assert metrics["comparable_prefix_tokens"] == 10
    assert metrics["fixed_length_matches"] == 254
    assert metrics["fixed_length_token_match_rate"] == pytest.approx(254 / 256)

    truncated = free_running_metrics(reference, reference[:40])
    assert truncated["first_divergence"] == 40
    assert truncated["comparable_prefix_tokens"] == 40
    assert truncated["fixed_length_token_match_rate"] == pytest.approx(40 / 256)

    short_fixed = free_running_metrics(reference[:10], reference[:10], fixed_length=8)
    assert short_fixed["first_divergence"] is None
    assert short_fixed["fixed_length"] == 8


# ---------------------------------------------------------------------------
# CLI parsers


def test_cli_parsers_accept_required_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "build",
            "--model", "/m.gguf",
            "--data-manifest", "/data.json",
            "--split", "qualification",
            "--suite", "qualification",
            "--target-tokens", "32768",
            "--seed", "0",
            "--output", "/tasks.json",
        ]
    )
    assert args.command == "build"
    assert args.target_tokens == 32768
    assert not hasattr(args, "metadata")

    args = parser.parse_args(
        [
            "run",
            "--model", "/m.gguf",
            "--arm", "sidecar",
            "--metadata", "/side.json",
            "--task-manifest", "/tasks.json",
            "--max-answer-tokens", "24",
            "--output", "/result.json",
        ]
    )
    assert args.command == "run"
    assert args.arm == "sidecar"
    assert args.diagnostic == "tasks"

    args = parser.parse_args(
        [
            "run",
            "--model", "/m.gguf",
            "--arm", "dense",
            "--diagnostic", "free-running",
            "--data-manifest", "/data.json",
            "--split", "qualification",
            "--free-running-tokens", "256",
            "--output", "/freerun.json",
        ]
    )
    assert args.arm == "dense"
    assert args.free_running_tokens == DEFAULT_FREE_RUNNING_TOKENS

    args = parser.parse_args(
        [
            "compare",
            "--dense", "/dense.json",
            "--baseline", "/base.json",
            "--candidate", "/cand.json",
            "--output", "/verdict.json",
        ]
    )
    assert args.command == "compare"
