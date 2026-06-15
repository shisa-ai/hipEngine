from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.stepfun_kv_session_contract import (
    build_kv_session_contract_artifact,
    main,
    stepfun_gguf_paths,
)


def _available_stepfun_gguf_dir() -> Path:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", "/models/gguf"))
    try:
        stepfun_gguf_paths(root)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return root


def test_stepfun_kv_session_contract_artifact_binds_session_contract() -> None:
    gguf_dir = _available_stepfun_gguf_dir()

    artifact = build_kv_session_contract_artifact(gguf_dir=gguf_dir)

    assert artifact["schema_version"] == 1
    assert artifact["artifact_kind"] == "stepfun_kv_session_streaming_decode_contract"
    assert artifact["status"] == "blocked"
    assert artifact["backend"] == "hip_gfx1151"
    assert artifact["prompt"] == "hello"
    assert artifact["reasoning_effort"] == "low"
    assert artifact["max_context"] == 512
    assert artifact["max_new_tokens"] == 1
    assert artifact["context_pages"] == 1
    assert artifact["page_size"] == 512
    assert len(artifact["gguf_paths"]) == 3
    contract = artifact["contract"]
    assert contract["schema_version"] == 1
    assert contract["source"] == "StepFunResidentSession.kv_streaming_decode_contract"
    assert contract["executable"] is False
    assert contract["ready"] is False
    assert contract["blocked_by"] == "streaming_decode_loop_not_wired"
    assert contract["next_action"] == "wire_streaming_decode_loop"
    assert contract["backend_matches"] is True
    assert contract["layer_count_matches"] is True
    assert contract["session_layer_count"] == 45
    assert contract["plan_layer_count"] == 45
    assert contract["pre_run_upload_entry_count"] == 6
    assert contract["pre_run_upload_checks_passed"] is True
    assert contract["launch_operation_count"] == 135
    assert contract["launch_per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert contract["all_launches_have_dispatch_keys"] is True
    assert contract["all_launches_ready"] is True
    assert contract["no_kernel_launches"] is True
    assert contract["required_artifacts"] == [
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-kernel-trace.json",
        "benchmarks/results/2026-05-31-stepfun-q3kl-kv-backed-next-token.json",
    ]
    decode_entrypoint = artifact["decode_entrypoint_blocker"]
    assert decode_entrypoint["schema_version"] == 1
    assert decode_entrypoint["source"] == "StepFunResidentSession.decode_one_token_kv_bf16"
    assert decode_entrypoint["executable"] is False
    assert decode_entrypoint["ready"] is False
    assert decode_entrypoint["blocked_by"] == "streaming_decode_loop_not_wired"
    assert decode_entrypoint["next_action"] == "implement_resident_kv_streaming_decode_loop"
    assert decode_entrypoint["contract_source"] == "StepFunResidentSession.kv_streaming_decode_contract"
    assert decode_entrypoint["session_backend"] == "hip_gfx1151"
    assert decode_entrypoint["session_layer_count"] == 45
    assert decode_entrypoint["cache_layer_count"] == 45
    assert decode_entrypoint["cache_layer_count_matches"] is True
    assert decode_entrypoint["kv_cache_context_pages"] == 1
    assert decode_entrypoint["kv_cache_page_size"] == 512
    assert decode_entrypoint["kv_cache_tokens"] == 512
    assert decode_entrypoint["kv_cache_nbytes"] == 94371840
    assert decode_entrypoint["kv_cache_buffer_count"] == 0
    assert decode_entrypoint["kv_dispatch_key_names"] == [
        "decode_attention",
        "decode_kv_write",
        "prompt_kv_write",
    ]
    assert decode_entrypoint["all_kv_dispatch_keys_bound"] is True
    assert decode_entrypoint["kv_dispatch_keys"] == {
        "decode_attention": {
            "backend": "hip_gfx1151",
            "layer": "paged_attn_decode",
            "quant": "gguf_step35",
            "variant": "bf16_split_k_gate_f32_spans",
        },
        "decode_kv_write": {
            "backend": "hip_gfx1151",
            "layer": "paged_kv_write",
            "quant": "gguf_step35",
            "variant": "mixed_bf16_spans",
        },
        "prompt_kv_write": {
            "backend": "hip_gfx1151",
            "layer": "paged_kv_write",
            "quant": "gguf_step35",
            "variant": "mixed_bf16_prompt_spans",
        },
    }
    assert decode_entrypoint["kv_dispatch_keys_sha256"] == (
        "9457815868810189bc25a1897c24380f85dcdd76ef1a4cf4edd75491e794cff5"
    )
    assert decode_entrypoint["kv_cache_layer_nbytes_match_expected"] is True
    assert decode_entrypoint["kv_cache_layer_nbytes_sha256"]
    assert decode_entrypoint["pre_run_upload_plan_sha256"]
    assert decode_entrypoint["pre_run_upload_entry_count"] == contract[
        "pre_run_upload_entry_count"
    ]
    assert decode_entrypoint["pre_run_upload_total_nbytes"] == contract[
        "pre_run_upload_total_nbytes"
    ]
    assert decode_entrypoint["pre_run_upload_order"] == contract["pre_run_upload_order"]
    assert decode_entrypoint["pre_run_cleanup_order"] == contract[
        "pre_run_cleanup_order"
    ]
    assert decode_entrypoint["pre_run_upload_checks_passed"] is True
    assert decode_entrypoint["launch_operation_sequence_sha256"] == contract[
        "launch_operation_sequence_sha256"
    ]
    assert decode_entrypoint["launch_operation_records_sha256"] == contract[
        "launch_operation_records_sha256"
    ]
    assert decode_entrypoint["planned_launch_operation_count"] == 135
    assert decode_entrypoint["all_planned_launches_ready"] is True
    assert decode_entrypoint["no_kernel_launches"] is True
    assert decode_entrypoint["required_artifacts"] == contract["required_artifacts"]
    assert decode_entrypoint["no_claim_policy"] == {
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This entrypoint validates the future resident KV streaming loop contract "
            "but does not launch kernels or generate a token."
        ),
    }
    assert artifact["no_claim_policy"] == {
        "kv_backed_decode_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
        "performance_claim_allowed": False,
        "reason": (
            "This artifact captures the resident-session streaming decode contract "
            "but does not materialize weights, launch kernels, or generate a token."
        ),
    }


def test_stepfun_kv_session_contract_cli_writes_artifact(tmp_path: Path) -> None:
    gguf_dir = _available_stepfun_gguf_dir()
    output = tmp_path / "kv-session-contract.json"

    rc = main(
        [
            "--gguf-dir",
            str(gguf_dir),
            "--artifact-date",
            "2030-01-08",
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload["date"] == "2030-01-08"
    assert payload["status"] == "blocked"
    assert payload["contract"]["blocked_by"] == "streaming_decode_loop_not_wired"
    assert payload["contract"]["launch_operation_count"] == 135
    assert payload["decode_entrypoint_blocker"]["blocked_by"] == (
        "streaming_decode_loop_not_wired"
    )
    assert payload["decode_entrypoint_blocker"]["executable"] is False


def test_stepfun_kv_session_contract_cli_compact_modes(tmp_path: Path) -> None:
    gguf_dir = _available_stepfun_gguf_dir()
    output = tmp_path / "compact.json"
    base_args = ["--gguf-dir", str(gguf_dir)]

    assert main([*base_args, "--status-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "blocked"
    assert main([*base_args, "--blocked-by-only", "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == "streaming_decode_loop_not_wired"
    assert main([*base_args, "--sha-only", "--output", str(output)]) == 0
    assert isinstance(json.loads(output.read_text()), str)
