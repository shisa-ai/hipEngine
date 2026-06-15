from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_oracle_source_map import (
    build_oracle_source_map,
    collect_python_symbols,
    main,
)


def test_collect_python_symbols_reports_top_level_and_class_methods(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "def top_level():\n"
        "    pass\n\n"
        "class Owner:\n"
        "    def method(self):\n"
        "        pass\n"
    )

    symbols = collect_python_symbols(source)

    assert symbols["top_level"] == 1
    assert symbols["Owner"] == 4
    assert symbols["Owner.method"] == 5


def test_stepfun_oracle_source_map_lists_in_tree_owners() -> None:
    report = build_oracle_source_map(artifact_date="2030-01-27")

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_oracle_source_map"
    assert report["date"] == "2030-01-27"
    assert report["status"] == "mapped"
    assert report["ready"] is True
    assert report["next_action_kind"] == "logits_backend_parity_investigation"
    assert report["unresolved_evidence_gap"] == "generated_text_matches_target"
    assert report["entry_count"] == 3
    assert [entry["key"] for entry in report["entries"]] == [
        "host_prompt_logit_smoke",
        "llamacpp_oracle_runner",
        "oracle_blocker_handoff",
    ]
    assert report["all_symbols_present"] is True
    assert report["missing_symbols"] == []
    assert report["missing_artifacts"] == []
    prompt_entry, oracle_entry, handoff_entry = report["entries"]
    assert prompt_entry["all_symbols_present"] is True
    assert oracle_entry["all_symbols_present"] is True
    assert handoff_entry["all_symbols_present"] is True
    assert any(
        source["path"] == "hipengine/runtime/stepfun_gguf_runner.py"
        and {
            item["symbol"] for item in source["present_symbols"]
        }
        >= {
            "StepFunResidentSession.embed_chat_prompt_bf16",
            "StepFunResidentSession.layer_prefix_prompt_logits_probe_bf16",
            "StepFunResidentSession.final_logits_probe_bf16",
        }
        for source in prompt_entry["sources"]
    )
    assert "scripts/stepfun_llamacpp_oracle.py" in {
        source["path"] for source in oracle_entry["sources"]
    }
    assert "scripts/stepfun_oracle_next_action_manifest.py" in {
        source["path"] for source in handoff_entry["sources"]
    }
    artifact_kinds = {
        record["artifact_kind"] for record in report["artifact_records"] if record["exists"]
    }
    assert "stepfun_oracle_next_action_manifest" in artifact_kinds
    assert report["source_map_summary"] == {
        "host_prompt_logits_owner": "scripts/stepfun_layer_prefix_smoke.py + StepFunResidentSession logits probes",
        "canonical_oracle_owner": "scripts/stepfun_llamacpp_oracle.py + token/rank diagnostics",
        "handoff_owner": "oracle diagnosis, consistency, and next-action manifest scripts",
    }
    assert report["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert report["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This source map only identifies in-tree owners for the next "
            "logits/backend parity investigation; generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_oracle_source_map_reports_missing_sources(tmp_path: Path) -> None:
    report = build_oracle_source_map(repo_root=tmp_path)

    assert report["status"] == "incomplete"
    assert report["ready"] is False
    assert report["all_symbols_present"] is False
    assert report["missing_symbols"]
    assert {item["entry_key"] for item in report["missing_symbols"]} == {
        "host_prompt_logit_smoke",
        "llamacpp_oracle_runner",
        "oracle_blocker_handoff",
    }


def test_stepfun_oracle_source_map_cli_writes_report(tmp_path: Path) -> None:
    output = tmp_path / "source-map.json"

    rc = main(["--artifact-date", "2030-01-28", "--output", str(output), "--pretty"])

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-28"
    assert payload["status"] == "mapped"
    assert payload["entry_count"] == 3
    assert payload["all_symbols_present"] is True


def test_stepfun_oracle_source_map_cli_compact_modes(tmp_path: Path) -> None:
    output = tmp_path / "compact.json"

    assert main(["--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "mapped"
    assert main(["--missing-symbols-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == []
    assert main(["--entry-keys-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "host_prompt_logit_smoke",
        "llamacpp_oracle_runner",
        "oracle_blocker_handoff",
    ]
