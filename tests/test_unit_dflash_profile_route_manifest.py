from __future__ import annotations

from pathlib import Path

from scripts.dflash_build_profile_route_manifest import build_manifest
from scripts.dflash_chain_e2e_bench import _draft_budgets_for_prompt, _load_profile_route_manifest


def _row(prompt_id: str, speedup: float, *, passed: bool = True, exact: bool = True, proposal: str = "chain") -> dict:
    return {
        "prompt": {"id": prompt_id, "benchmark_group": "code"},
        "config": {"proposal_mode": proposal, "verify_mode": "verify_chain", "profile_route": None},
        "correctness": {"passed": passed, "exact_match_ar": exact},
        "spec": {"speedup_vs_ar": speedup, "decode_tok_s": 10.0 * speedup},
        "ar": {"decode_tok_s": 10.0},
    }


def test_build_profile_route_manifest_selects_only_exact_chain_winners() -> None:
    artifact = {
        "measurements": {
            "rows": [
                _row("code:fast", 1.08),
                _row("code:slow", 0.99),
                _row("code:mismatch", 1.25, exact=False),
                _row("code:tree", 1.30, proposal="tree"),
            ]
        }
    }

    manifest = build_manifest(artifact, source="artifact.json", min_chain_speedup=1.02)

    assert manifest["default"] == "ar"
    assert manifest["routes"] == {"code:fast": "chain"}
    assert manifest["summary"]["rows_with_prompt_id"] == 4
    assert manifest["summary"]["chain_routes"] == 1
    evidence = {row["prompt_id"]: row for row in manifest["row_evidence"]}
    assert evidence["code:mismatch"]["route"] == "ar"
    assert evidence["code:mismatch"]["exact_match_ar"] is False


def test_build_profile_route_manifest_can_use_tok_s_ratio() -> None:
    artifact = {
        "measurements": {
            "rows": [
                {
                    "prompt": {"id": "code:ratio"},
                    "config": {"proposal_mode": "chain", "verify_mode": "verify_chain", "profile_route": None},
                    "correctness": {"passed": True, "exact_match_ar": True},
                    "spec": {"decode_tok_s": 12.0},
                    "ar": {"decode_tok_s": 10.0},
                }
            ]
        }
    }

    manifest = build_manifest(artifact, source="artifact.json", min_chain_speedup=1.1)

    assert manifest["routes"] == {"code:ratio": "chain"}
    assert manifest["row_evidence"][0]["speedup_vs_ar"] == 1.2


def test_profile_route_manifest_can_override_draft_budget_by_prompt(tmp_path: Path) -> None:
    manifest_path = tmp_path / "route.json"
    manifest_path.write_text(
        """
        {
          "default": "ar",
          "routes": {"code:fast": "chain"},
          "draft_budgets": {"default": 4, "code:fast": [8, 15], "code": "2,4"}
        }
        """,
        encoding="utf-8",
    )

    default, routes, manifest = _load_profile_route_manifest(manifest_path)

    assert default == "ar"
    assert routes == {"code:fast": "chain"}
    assert manifest is not None
    assert _draft_budgets_for_prompt({"id": "code:fast", "benchmark_group": "code"}, default=[4], manifest=manifest) == [8, 15]
    assert _draft_budgets_for_prompt({"id": "code:other", "benchmark_group": "code"}, default=[4], manifest=manifest) == [2, 4]
    assert _draft_budgets_for_prompt({"id": "math:slow", "benchmark_group": "math"}, default=[4], manifest=manifest) == [4]
