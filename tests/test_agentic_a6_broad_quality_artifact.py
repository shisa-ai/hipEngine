from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hipengine.benchmark.agentic_quality import validate_agentic_quality_artifact
from hipengine.tokenization.identity import token_ids_sha256


ARTIFACT = Path(
    "benchmarks/results/2026-07-22-w7900-agentic-a6-broad-quality.json"
)
WORKLOADS = Path("benchmarks/prompts/agentic-quality-v2.json")
ORACLE = Path("benchmarks/oracles/agentic-quality-v2.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_a6_broad_quality_retains_external_oracles_and_exact_ids() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert validate_agentic_quality_artifact(payload) == {
        "passed": True,
        "failure_reasons": [],
    }
    assert payload["kind"] == "hipengine_agentic_coding_quality_benchmark"
    assert payload["schema_version"] == 1
    assert payload["performance_claim"] is False
    assert payload["validation"] == {"passed": True, "failure_reasons": []}

    provenance = payload["hipengine_artifact_provenance"]
    assert provenance["hipengine_commit"] == (
        "878d07a9ba8d3cd24cf44bd88d359be7b4921c2e"
    )
    assert provenance["configured_backend"] == "hip_gfx1100"
    assert provenance["resolved_backend"] == "hip_gfx1100"
    assert provenance["target_arch"] == "gfx1100"
    assert provenance["device_name"] == "AMD Radeon Pro W7900"
    assert provenance["quant"] == "gguf_q4_k_m"
    assert provenance["kv_dtype"] == "bf16"
    assert provenance["dirty"] is False
    assert provenance["staged_dirty"] is False
    assert provenance["unstaged_dirty"] is False
    assert provenance["untracked_dirty"] is False
    assert provenance["repetitions"] == 2
    assert provenance["warmups"] == 0
    assert provenance["profiler"] == {
        "used": False,
        "reason": "non-performance quality lane",
    }
    assert provenance["model_fingerprint"]["value"] == (
        "936659d614707776d8e6ca1fb8595991159e78361bff2e3a3616aa91564c89fb"
    )
    assert provenance["environment"]["HIP_VISIBLE_DEVICES"] == "0"
    assert provenance["environment"]["ROCR_VISIBLE_DEVICES"] == "0"
    assert provenance["environment"]["HIPENGINE_PREFIX_CACHE"] == "off"
    assert provenance["environment"]["HIPENGINE_QWEN35_NATIVE_SAMPLER"] == "0"

    suite = payload["workload_suite"]
    assert suite["file_sha256"] == _sha256(WORKLOADS)
    assert suite["quality_oracle"]["file_sha256"] == _sha256(ORACLE)
    assert suite["quality_oracle"]["kind"] == "hipengine.agentic_quality_oracles"
    assert suite["quality_oracle"]["suite"] == "agentic-quality-v2"

    assert payload["configuration"]["tool_choice"] == "auto"
    assert payload["configuration"]["performance_claim"] is False
    assert payload["configuration"]["quality_system_policy"] == (
        "automatic_selection_without_expected_tool_name_hint"
    )
    assert payload["configuration"]["workloads"] == [
        "repository_scheduler_en",
        "repository_cache_en",
        "general_en_operations",
        "general_en_release",
        "general_ja_operations",
        "mixed_ja_en_release",
    ]

    assert payload["coverage"] == {
        "agents": 12,
        "concurrency": 1,
        "families": ["general_en", "general_ja", "mixed_ja_en", "repository"],
        "generated_tokens": 4538,
        "runs": 12,
        "turns": 48,
        "workloads": [
            "general_en_operations",
            "general_en_release",
            "general_ja_operations",
            "mixed_ja_en_release",
            "repository_cache_en",
            "repository_scheduler_en",
        ],
    }
    quality = payload["quality"]
    assert quality["attempts"] == 48
    assert quality["valid_calls"] == 18
    assert quality["correct_tools"] == 18
    assert quality["exact_arguments"] == 16
    assert quality["successes"] == 10
    assert quality["repair_attempts"] == 0
    assert quality["outcomes"] == {
        "content_alongside_tool_call": 6,
        "invalid_tool_call": 20,
        "no_tool_call": 10,
        "passed": 10,
        "wrong_arguments": 2,
    }
    assert quality["external_oracle"] == {
        "attempts": 48,
        "passes": 16,
        "pass_rate": 1 / 3,
        "patch_attempts": 6,
        "patch_successes": 0,
        "patch_success_rate": 0.0,
        "test_attempts": 8,
        "test_successes": 8,
        "test_success_rate": 1.0,
    }
    expected_families = {
        "general_en": (16, 6, 6, 4, 6),
        "general_ja": (8, 2, 0, 0, 0),
        "mixed_ja_en": (8, 4, 4, 4, 4),
        "repository": (16, 6, 6, 2, 6),
    }
    for family, (attempts, valid, exact, successes, oracle_passes) in (
        expected_families.items()
    ):
        row = quality["families"][family]
        assert row["attempts"] == attempts
        assert row["valid_calls"] == valid
        assert row["exact_arguments"] == exact
        assert row["successes"] == successes
        assert row["external_oracle"]["passes"] == oracle_passes
        assert row["repair_attempts"] == 0

    records = payload["turn_records"]
    assert len(records) == 48
    assert payload["turn_records_sha256"] == hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    paired: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in records:
        output = record["output"]
        assert output["generated_token_ids_source"] == "response"
        assert output["generated_token_ids_sha256"] == token_ids_sha256(
            output["generated_token_ids"]
        )
        assert output["raw_markup_leaked"] is False
        assert record["quality"]["external_oracle"]["evaluated"] is True
        paired.setdefault((record["workload_id"], record["turn_index"]), []).append(
            record
        )
    assert len(paired) == 24
    for pair in paired.values():
        assert len(pair) == 2
        first, second = pair
        assert first["finish"] == second["finish"]
        assert first["output"] == second["output"]
        first_quality = {k: v for k, v in first["quality"].items() if k != "call_id"}
        second_quality = {k: v for k, v in second["quality"].items() if k != "call_id"}
        assert first_quality == second_quality

    ownership = payload["final_ownership"]
    assert ownership["allowed_cache_bytes"] == 0
    assert ownership["cache_resident_bytes"] == 0
    assert all(value == 0 for value in ownership.values())

    forbidden = {
        "latency",
        "goodput",
        "tok_per_s",
        "tokens_per_second",
        "wall_seconds",
        "ttft",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
