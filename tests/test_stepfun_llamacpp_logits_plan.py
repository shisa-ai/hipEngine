from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_llamacpp_logits_plan import build_llamacpp_logits_plan, main


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True))


def _write_inputs(tmp_path: Path, *, missing: list[str] | None = None) -> tuple[Path, Path, Path]:
    preflight = tmp_path / "preflight.json"
    source_map = tmp_path / "source-map.json"
    manifest = tmp_path / "manifest.json"
    missing_evidence = (
        [
            "llama_cpp_same_prompt_logits_artifact_present",
            "llama_cpp_logits_dump_entrypoint_identified",
        ]
        if missing is None
        else missing
    )
    _write_json(
        preflight,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_llamacpp_logits_preflight",
            "status": "blocked" if missing_evidence else "ready",
            "missing_evidence": missing_evidence,
            "target": {
                "canonical_backend": "vulkan",
                "expected_next_token_id": 369,
                "expected_next_token_text": " |",
                "generated_first_token_id": 671,
                "generated_text": "The\n\n",
                "host_top_token_ids": [369, 5, 15251, 223, 201],
                "host_top1_to_top2_margin": 0.8150444030761719,
            },
            "expected_llama_logits_artifact": {
                "path": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json",
                "exists": False,
                "sha256": None,
            },
            "llama_cli_probe": {
                "path": "/tmp/llama-cli",
                "exists": True,
                "executable": True,
            },
            "prerequisites": [
                {
                    "name": "next_action_manifest_investigation_ready",
                    "passed": True,
                },
                {"name": "source_map_mapped", "passed": True},
                {"name": "host_logit_margin_passed", "passed": True},
                {"name": "llama_cli_available", "passed": True},
            ],
        },
    )
    source_entries = [
        {
            "key": "host_prompt_logit_smoke",
            "sources": [
                {
                    "path": "scripts/stepfun_layer_prefix_smoke.py",
                    "present_symbols": [{"symbol": "main", "line": 1}],
                }
            ],
        },
        {
            "key": "llamacpp_oracle_runner",
            "sources": [
                {
                    "path": "scripts/stepfun_llamacpp_oracle.py",
                    "present_symbols": [{"symbol": "main", "line": 1}],
                }
            ],
        },
        {
            "key": "oracle_blocker_handoff",
            "sources": [
                {
                    "path": "scripts/stepfun_oracle_blocker_diagnosis.py",
                    "present_symbols": [{"symbol": "main", "line": 1}],
                }
            ],
        },
    ]
    _write_json(
        source_map,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_oracle_source_map",
            "status": "mapped",
            "ready": True,
            "entries": source_entries,
        },
    )
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "artifact_kind": "stepfun_oracle_next_action_manifest",
            "status": "blocked",
            "investigation_ready": True,
        },
    )
    return preflight, source_map, manifest


def test_stepfun_llamacpp_logits_plan_describes_pending_capture_work(tmp_path: Path) -> None:
    preflight, source_map, manifest = _write_inputs(tmp_path)

    report = build_llamacpp_logits_plan(
        preflight_artifact=preflight,
        source_map_artifact=source_map,
        next_action_manifest=manifest,
        artifact_date="2030-02-01",
    )

    assert report["schema_version"] == 1
    assert report["artifact_kind"] == "stepfun_llamacpp_logits_plan"
    assert report["date"] == "2030-02-01"
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert report["implementation_ready"] is True
    assert report["missing_evidence"] == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "llama_cpp_logits_dump_entrypoint_identified",
    ]
    assert report["target"] == {
        "canonical_backend": "vulkan",
        "readiness_gate": "oracle_parity",
        "unresolved_evidence_gap": "generated_text_matches_target",
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "generated_first_token_id": 671,
        "generated_text": "The\n\n",
        "host_top_token_ids": [369, 5, 15251, 223, 201],
        "host_top1_to_top2_margin": 0.8150444030761719,
    }
    assert report["llama_cli"] == "/tmp/llama-cli"
    assert report["expected_llamacpp_logits_artifact"] == {
        "path": "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json",
        "exists": False,
        "sha256": None,
    }
    assert report["step_keys"] == [
        "identify_or_add_logits_dump_entrypoint",
        "capture_same_prompt_llamacpp_logits",
        "compare_host_vs_llamacpp_logits",
        "refresh_oracle_handoff_artifacts",
    ]
    assert [step["status"] for step in report["implementation_steps"]] == [
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert report["implementation_steps"][0]["acceptance"] == [
        "a llama.cpp command or helper can emit same-prompt logits/top-k for the retained StepFun prompt",
        "scripts/stepfun_llamacpp_logits_preflight.py no longer reports llama_cpp_logits_dump_entrypoint_identified",
    ]
    assert report["implementation_steps"][1]["expected_output_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json"
    )
    assert report["required_preflight_to_clear_plan_blocker"] == [
        "scripts/stepfun_llamacpp_logits_preflight.py --status-only returns \"ready\"",
        "scripts/stepfun_llamacpp_logits_preflight.py --missing-evidence-only returns []",
        "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-probe.json exists and is retained",
    ]
    assert report["blocked_reason"] == (
        "same-prompt llama.cpp logits capture is not implemented or retained yet"
    )
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
            "This plan only describes the llama.cpp logits evidence work needed next; "
            "generated_text_matches_target remains unresolved."
        ),
    }


def test_stepfun_llamacpp_logits_plan_marks_entrypoint_step_completed_when_only_logits_artifact_missing(
    tmp_path: Path,
) -> None:
    preflight, source_map, manifest = _write_inputs(
        tmp_path,
        missing=["llama_cpp_same_prompt_logits_artifact_present"],
    )
    payload = json.loads(preflight.read_text())
    payload["llama_cli_probe"] = {
        "path": "/tmp/llama-debug",
        "exists": True,
        "executable": True,
        "obvious_logits_dump_flag_present": True,
        "matched_logits_dump_markers": ["--save-logits", "--logits-output-dir"],
    }
    _write_json(preflight, payload)

    report = build_llamacpp_logits_plan(
        preflight_artifact=preflight,
        source_map_artifact=source_map,
        next_action_manifest=manifest,
    )

    assert report["status"] == "blocked"
    assert report["implementation_ready"] is True
    assert report["missing_evidence"] == ["llama_cpp_same_prompt_logits_artifact_present"]
    assert report["llama_cli"] == "/tmp/llama-debug"
    assert [step["status"] for step in report["implementation_steps"]] == [
        "completed",
        "pending",
        "pending",
        "pending",
    ]
    assert report["implementation_steps"][0]["notes"] == [
        "current preflight found a llama.cpp probe binary with logits dump flags",
        "do not use --logit-bias as a substitute for logits capture",
    ]


def test_stepfun_llamacpp_logits_plan_requires_only_expected_missing_evidence(
    tmp_path: Path,
) -> None:
    preflight, source_map, manifest = _write_inputs(
        tmp_path,
        missing=[
            "llama_cpp_same_prompt_logits_artifact_present",
            "next_action_manifest_investigation_ready",
        ],
    )

    report = build_llamacpp_logits_plan(
        preflight_artifact=preflight,
        source_map_artifact=source_map,
        next_action_manifest=manifest,
    )

    assert report["status"] == "blocked"
    assert report["implementation_ready"] is False
    assert report["missing_evidence"] == [
        "llama_cpp_same_prompt_logits_artifact_present",
        "next_action_manifest_investigation_ready",
    ]


def test_stepfun_llamacpp_logits_plan_cli_writes_report(tmp_path: Path) -> None:
    preflight, source_map, manifest = _write_inputs(tmp_path)
    output = tmp_path / "plan.json"

    rc = main(
        [
            "--preflight-artifact",
            str(preflight),
            "--source-map-artifact",
            str(source_map),
            "--next-action-manifest",
            str(manifest),
            "--artifact-date",
            "2030-02-02",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-02-02"
    assert payload["status"] == "blocked"
    assert payload["implementation_ready"] is True
    assert payload["step_keys"] == [
        "identify_or_add_logits_dump_entrypoint",
        "capture_same_prompt_llamacpp_logits",
        "compare_host_vs_llamacpp_logits",
        "refresh_oracle_handoff_artifacts",
    ]


def test_stepfun_llamacpp_logits_plan_cli_compact_modes(tmp_path: Path) -> None:
    preflight, source_map, manifest = _write_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--preflight-artifact",
        str(preflight),
        "--source-map-artifact",
        str(source_map),
        "--next-action-manifest",
        str(manifest),
    ]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--implementation-ready-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) is True
    assert main([*base_args, "--step-keys-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "identify_or_add_logits_dump_entrypoint",
        "capture_same_prompt_llamacpp_logits",
        "compare_host_vs_llamacpp_logits",
        "refresh_oracle_handoff_artifacts",
    ]
