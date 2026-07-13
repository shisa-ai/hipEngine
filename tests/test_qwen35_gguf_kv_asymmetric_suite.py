from __future__ import annotations

from types import SimpleNamespace

from scripts import qwen35_gguf_kv_asymmetric_suite as suite


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1000 + byte for byte in text.encode("utf-8")]


def _prompt_rows() -> list[dict[str, str]]:
    return [
        {"id": "code_a", "category": "code", "prompt": "write code"},
        {"id": "english_a", "category": "general_en", "prompt": "explain memory"},
        {"id": "japanese_a", "category": "general_ja", "prompt": "日本語で説明"},
    ]


def test_build_prompt_cases_preserves_exact_shape_identity_and_final_query() -> None:
    cases = suite._build_prompt_cases(
        _FakeTokenizer(),
        _prompt_rows(),
        prompt_length=256,
        include_mixed_v1=True,
        heldout_ids={"english_a"},
    )

    assert [case.prompt_id for case in cases] == ["code_a", "english_a", "japanese_a", "mixed_v1"]
    assert all(len(case.tokens) == 256 for case in cases)
    assert len({case.token_sha256 for case in cases}) == 4
    assert cases[0].split == "train"
    assert cases[1].split == "heldout"
    assert cases[-1].split == "control"
    assert cases[-1].profile == "mixed_v1"
    assert all(case.current_query_tokens > 0 for case in cases[:-1])


def test_aggregate_candidates_requires_every_prompt_gate_and_reports_scopes() -> None:
    prompt_results = [
        {
            "prompt": {"id": "train_a", "category": "code", "split": "train", "profile": "natural_corpus_v1"},
            "candidates": [
                {"name": "good", "quality_gate_passed": True, "logit_gate": {"kl": [0.01, 0.02], "top1_matches": [True, True], "mean_kl": 0.015, "max_kl": 0.02, "top1_agreement": 1.0}, "target_context_memory": {"total_bytes": 120}, "extra_bytes_over_baseline": 20},
                {"name": "heldout_fail", "quality_gate_passed": True, "logit_gate": {"kl": [0.01, 0.02], "top1_matches": [True, True], "mean_kl": 0.015, "max_kl": 0.02, "top1_agreement": 1.0}, "target_context_memory": {"total_bytes": 100}, "extra_bytes_over_baseline": 0},
            ],
        },
        {
            "prompt": {"id": "heldout_a", "category": "general_en", "split": "heldout", "profile": "natural_corpus_v1"},
            "candidates": [
                {"name": "good", "quality_gate_passed": True, "logit_gate": {"kl": [0.03, 0.04], "top1_matches": [True, True], "mean_kl": 0.035, "max_kl": 0.04, "top1_agreement": 1.0}, "target_context_memory": {"total_bytes": 120}, "extra_bytes_over_baseline": 20},
                {"name": "heldout_fail", "quality_gate_passed": False, "logit_gate": {"kl": [0.06, 0.08], "top1_matches": [False, True], "mean_kl": 0.07, "max_kl": 0.08, "top1_agreement": 0.5}, "target_context_memory": {"total_bytes": 100}, "extra_bytes_over_baseline": 0},
            ],
        },
        {
            "prompt": {"id": "mixed_v1", "category": "mixed_synthetic", "split": "control", "profile": "mixed_v1"},
            "candidates": [
                {"name": "good", "quality_gate_passed": True, "logit_gate": {"kl": [0.02, 0.01], "top1_matches": [True, True], "mean_kl": 0.015, "max_kl": 0.02, "top1_agreement": 1.0}, "target_context_memory": {"total_bytes": 120}, "extra_bytes_over_baseline": 20},
                {"name": "heldout_fail", "quality_gate_passed": True, "logit_gate": {"kl": [0.02, 0.01], "top1_matches": [True, True], "mean_kl": 0.015, "max_kl": 0.02, "top1_agreement": 1.0}, "target_context_memory": {"total_bytes": 100}, "extra_bytes_over_baseline": 0},
            ],
        },
    ]

    rows = suite._aggregate_candidates(prompt_results, extra_budget_bytes=32)
    by_name = {row["name"]: row for row in rows}

    assert by_name["good"]["transfer_eligible"]
    assert by_name["good"]["scopes"]["natural_full"]["prompt_count"] == 2
    assert by_name["good"]["scopes"]["heldout"]["all_prompt_gates_passed"]
    assert by_name["good"]["scopes"]["mixed_v1"]["mean_kl"] == 0.015
    assert not by_name["heldout_fail"]["transfer_eligible"]
    assert not by_name["heldout_fail"]["scopes"]["heldout"]["all_prompt_gates_passed"]
    assert by_name["heldout_fail"]["first_failed_prompt"] == "heldout_a"


def test_compact_prompt_result_drops_resident_arrays() -> None:
    screen = {
        "reference": {"seed_token_id": 1, "generated_token_ids": [2], "finite_logits": True, "elapsed_seconds": 0.1},
        "keys": [object()],
        "values": [object()],
        "full_layers": 10,
        "baseline_memory": {"total_bytes": 100},
        "forced_ids": [1],
        "recommendation": None,
        "rows": [{"name": "candidate"}],
    }
    case = SimpleNamespace(
        prompt_id="p",
        category="code",
        split="train",
        profile="natural_corpus_v1",
        tokens=(1, 2),
        token_sha256="abc",
        source_prompt_sha256="def",
        current_query_tokens=1,
    )

    compact = suite._compact_prompt_result(case, screen)

    assert compact["prompt"]["id"] == "p"
    assert compact["reference"]["seed_token_id"] == 1
    assert compact["candidates"] == [{"name": "candidate"}]
    assert "keys" not in compact
    assert "values" not in compact
