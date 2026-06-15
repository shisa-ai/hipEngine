from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_remaining_blockers_rollup import (
    build_remaining_blockers_rollup,
    main,
)
from test_stepfun_correctness_status import (  # type: ignore[import-not-found]
    _write_docs,
    _write_prompt_artifact,
    _write_resource_artifact,
)
from test_stepfun_oracle_backend_matrix import _write_inputs as _write_backend_inputs  # type: ignore[import-not-found]


def _write_rollup_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    prompt, vulkan, hip, model, fake_tokenize = _write_backend_inputs(tmp_path)
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    # The backend helper writes a Vulkan oracle pair suitable for oracle matrix tests.
    # The correctness-status/KV rollup fixtures expect the fuller canonical prompt
    # fixture shape, so overwrite only the prompt with the shared StepFun test fixture.
    _write_prompt_artifact(prompt)
    _write_docs(docs)
    _write_resource_artifact(resource)
    return prompt, vulkan, hip, docs, resource, fake_tokenize


def test_stepfun_remaining_blockers_rollup_links_oracle_and_kv(
    tmp_path: Path,
) -> None:
    prompt, oracle, hip, docs, resource, fake_tokenize = _write_rollup_inputs(tmp_path)

    rollup = build_remaining_blockers_rollup(
        prompt_artifact=prompt,
        oracle_artifact=oracle,
        hip_artifact=hip,
        resource_artifact=resource,
        docs=docs,
        llama_tokenize=fake_tokenize,
        tokenizer_model=tmp_path / "model.gguf",
        artifact_date="2030-01-06",
    )

    assert rollup["schema_version"] == 1
    assert rollup["artifact_kind"] == "stepfun_remaining_blockers_rollup"
    assert rollup["date"] == "2030-01-06"
    assert rollup["status"] == "blocked"
    assert rollup["remaining_blocker_count"] == 2
    assert rollup["readiness"] == {
        "status": "blocked",
        "oracle_parity": False,
        "kv_decode_dispatch_ready": True,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "blocked_gates": ["oracle_parity", "kv_backed_decode", "e2e_inference"],
        "blocker_kinds": ["oracle_parity_blocked", "kv_backed_decode_not_wired"],
    }
    assert [item["blocker_kind"] for item in rollup["remaining_blockers"]] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    oracle_blocker, kv_blocker = rollup["remaining_blockers"]
    assert oracle_blocker["readiness_gate"] == "oracle_parity"
    assert oracle_blocker["generator_command_kind"] == "oracle_backend_matrix"
    assert oracle_blocker["backend_outcomes"] == [
        {"backend": "vulkan", "outcome": "executed_token_mismatch"},
        {"backend": "hip", "outcome": "timeout"},
    ]
    assert oracle_blocker["canonical_backend"] == "vulkan"
    assert oracle_blocker["oracle_parity_ready"] is False
    assert oracle_blocker["rank_check_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-oracle-rank-check.json"
    )
    assert oracle_blocker["rank_check_generator_command"] == (
        "python3 scripts/stepfun_oracle_rank_check.py --default-output --pretty"
    )
    assert oracle_blocker["host_logit_margin_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-host-top-logit-margin.json"
    )
    assert oracle_blocker["host_logit_margin_generator_command"] == (
        "python3 scripts/stepfun_host_logit_margin.py --default-output --pretty"
    )
    assert oracle_blocker["top_token_roundtrip_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-host-top-token-roundtrip.json"
    )
    assert oracle_blocker["top_token_roundtrip_generator_command"] == (
        "python3 scripts/stepfun_top_token_roundtrip.py --default-output --pretty"
    )
    assert oracle_blocker["prompt_token_roundtrip_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-prompt-token-roundtrip.json"
    )
    assert oracle_blocker["prompt_token_roundtrip_generator_command"] == (
        "python3 scripts/stepfun_prompt_token_roundtrip.py --default-output --pretty"
    )
    assert oracle_blocker["diagnosis_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-blocker-diagnosis.json"
    )
    assert oracle_blocker["diagnosis_generator_command"] == (
        "python3 scripts/stepfun_oracle_blocker_diagnosis.py --default-output --pretty"
    )
    assert oracle_blocker["evidence_consistency_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-evidence-consistency-check.json"
    )
    assert oracle_blocker["evidence_consistency_generator_command"] == (
        "python3 scripts/stepfun_oracle_evidence_consistency_check.py --default-output --pretty"
    )
    assert oracle_blocker["next_action_manifest_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-next-action-manifest.json"
    )
    assert oracle_blocker["next_action_manifest_generator_command"] == (
        "python3 scripts/stepfun_oracle_next_action_manifest.py --default-output --pretty"
    )
    assert oracle_blocker["source_map_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-oracle-source-map.json"
    )
    assert oracle_blocker["source_map_generator_command"] == (
        "python3 scripts/stepfun_oracle_source_map.py --default-output --pretty"
    )
    assert oracle_blocker["llamacpp_logits_preflight_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-llamacpp-logits-preflight.json"
    )
    assert oracle_blocker["llamacpp_logits_preflight_generator_command"] == (
        "python3 scripts/stepfun_llamacpp_logits_preflight.py --default-output --pretty"
    )
    assert kv_blocker["readiness_gate"] == "kv_backed_decode"
    assert kv_blocker["generator_command_kind"] == "kv_blocker_status"
    assert kv_blocker["kv_decode_dispatch_ready"] is True
    assert kv_blocker["kv_backed_decode_ready"] is False
    assert kv_blocker["blocked_count"] == 2
    assert kv_blocker["missing_artifact_paths"] == [
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]
    assert kv_blocker["session_contract_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-kv-session-contract.json"
    )
    assert kv_blocker["session_contract_generator_command"] == (
        "python3 scripts/stepfun_kv_session_contract.py --default-output --pretty"
    )
    assert kv_blocker["evidence_preflight_artifact"] == (
        "benchmarks/results/2026-06-15-stepfun-q3kl-kv-evidence-preflight.json"
    )
    assert kv_blocker["evidence_preflight_generator_command"] == (
        "python3 scripts/stepfun_kv_evidence_preflight.py --default-output --pretty"
    )
    assert kv_blocker["streaming_runner_source_status"]["next_action"] == (
        "wire_streaming_decode_loop"
    )
    assert kv_blocker["runtime_wiring_symbol_validation"]["all_symbols_present"] is True
    assert rollup["generator_commands"] == {
        "oracle_backend_matrix": (
            "python3 scripts/stepfun_oracle_backend_matrix.py --default-output --pretty"
        ),
        "oracle_token_mismatch": (
            "python3 scripts/stepfun_oracle_token_mismatch.py --default-output --pretty"
        ),
        "oracle_rank_check": (
            "python3 scripts/stepfun_oracle_rank_check.py --default-output --pretty"
        ),
        "host_logit_margin": (
            "python3 scripts/stepfun_host_logit_margin.py --default-output --pretty"
        ),
        "top_token_roundtrip": (
            "python3 scripts/stepfun_top_token_roundtrip.py --default-output --pretty"
        ),
        "prompt_token_roundtrip": (
            "python3 scripts/stepfun_prompt_token_roundtrip.py --default-output --pretty"
        ),
        "oracle_blocker_diagnosis": (
            "python3 scripts/stepfun_oracle_blocker_diagnosis.py --default-output --pretty"
        ),
        "oracle_evidence_consistency_check": (
            "python3 scripts/stepfun_oracle_evidence_consistency_check.py --default-output --pretty"
        ),
        "oracle_next_action_manifest": (
            "python3 scripts/stepfun_oracle_next_action_manifest.py --default-output --pretty"
        ),
        "oracle_source_map": (
            "python3 scripts/stepfun_oracle_source_map.py --default-output --pretty"
        ),
        "llamacpp_logits_preflight": (
            "python3 scripts/stepfun_llamacpp_logits_preflight.py --default-output --pretty"
        ),
        "kv_blocker_status": (
            "python3 scripts/stepfun_kv_blocker_status.py --default-output --pretty"
        ),
        "kv_session_contract": (
            "python3 scripts/stepfun_kv_session_contract.py --default-output --pretty"
        ),
        "kv_evidence_preflight": (
            "python3 scripts/stepfun_kv_evidence_preflight.py --default-output --pretty"
        ),
        "status_refresh": (
            "python3 scripts/stepfun_correctness_status.py --pretty "
            "--output benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"
        ),
        "final_blocker_refresh": (
            "python3 scripts/stepfun_final_blocker_manifest.py --pretty "
            "--output benchmarks/results/2026-05-31-stepfun-q3kl-final-blocker-manifest.json"
        ),
        "handoff_refresh": (
            "python3 scripts/stepfun_handoff_check.py --pretty "
            "--output benchmarks/results/2026-05-31-stepfun-q3kl-handoff-check.json"
        ),
    }
    assert rollup["no_claim_policy"] == {
        "oracle_parity_claim_allowed": False,
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This rollup links retained blocker evidence and generator commands; "
            "it does not satisfy oracle parity or run KV-backed decode."
        ),
    }


def test_stepfun_remaining_blockers_rollup_cli_writes_artifact(tmp_path: Path) -> None:
    prompt, oracle, hip, docs, resource, fake_tokenize = _write_rollup_inputs(tmp_path)
    output = tmp_path / "rollup.json"

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--hip-artifact",
            str(hip),
            "--docs",
            str(docs),
            "--resource-artifact",
            str(resource),
            "--llama-tokenize",
            str(fake_tokenize),
            "--tokenizer-model",
            str(tmp_path / "model.gguf"),
            "--artifact-date",
            "2030-01-07",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-07"
    assert payload["status"] == "blocked"
    assert payload["remaining_blocker_count"] == 2
    assert [item["blocker_kind"] for item in payload["remaining_blockers"]] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]


def test_stepfun_remaining_blockers_rollup_cli_compact_modes(tmp_path: Path) -> None:
    prompt, oracle, hip, docs, resource, fake_tokenize = _write_rollup_inputs(tmp_path)
    output = tmp_path / "compact.json"
    base_args = [
        "--prompt-artifact",
        str(prompt),
        "--oracle-artifact",
        str(oracle),
        "--hip-artifact",
        str(hip),
        "--docs",
        str(docs),
        "--resource-artifact",
        str(resource),
        "--llama-tokenize",
        str(fake_tokenize),
        "--tokenizer-model",
        str(tmp_path / "model.gguf"),
    ]

    assert main([*base_args, "--open-count-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == 2
    assert main([*base_args, "--blocker-kinds-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert main([*base_args, "--generator-commands-only", "--output", str(output)]) == 0
    commands = json.loads(output.read_text())
    assert commands["oracle_backend_matrix"].endswith(
        "stepfun_oracle_backend_matrix.py --default-output --pretty"
    )
    assert commands["oracle_rank_check"].endswith(
        "stepfun_oracle_rank_check.py --default-output --pretty"
    )
    assert commands["host_logit_margin"].endswith(
        "stepfun_host_logit_margin.py --default-output --pretty"
    )
    assert commands["top_token_roundtrip"].endswith(
        "stepfun_top_token_roundtrip.py --default-output --pretty"
    )
    assert commands["prompt_token_roundtrip"].endswith(
        "stepfun_prompt_token_roundtrip.py --default-output --pretty"
    )
    assert commands["oracle_blocker_diagnosis"].endswith(
        "stepfun_oracle_blocker_diagnosis.py --default-output --pretty"
    )
    assert commands["oracle_evidence_consistency_check"].endswith(
        "stepfun_oracle_evidence_consistency_check.py --default-output --pretty"
    )
    assert commands["oracle_next_action_manifest"].endswith(
        "stepfun_oracle_next_action_manifest.py --default-output --pretty"
    )
    assert commands["oracle_source_map"].endswith(
        "stepfun_oracle_source_map.py --default-output --pretty"
    )
    assert commands["llamacpp_logits_preflight"].endswith(
        "stepfun_llamacpp_logits_preflight.py --default-output --pretty"
    )
    assert commands["kv_blocker_status"].endswith(
        "stepfun_kv_blocker_status.py --default-output --pretty"
    )
    assert commands["kv_session_contract"].endswith(
        "stepfun_kv_session_contract.py --default-output --pretty"
    )
    assert commands["kv_evidence_preflight"].endswith(
        "stepfun_kv_evidence_preflight.py --default-output --pretty"
    )
