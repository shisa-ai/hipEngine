from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.speculative import Qwen35GGUFMTPContext, Qwen35GGUFMTPPerformanceReadiness
from scripts import gguf_mtp_b1_prompt_suite as suite


EXPECTED_CLI_GATE_EXIT_CODES = {
    "blocked": 2,
    "partial_trace_budget": 3,
    "noncomparable_accepted_output": 4,
    "performance_unready": 5,
    "noncomparable_accepted_draft": 6,
    "native_runtime_missing": 7,
    "optimization_missing": 8,
    "kvlivespans_smoke_fail": 9,
    "exactness_failed": 10,
    "precheck_failed": 11,
    "metrics_contract_invalid": 12,
}


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prompt_suite(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": 1,
            "prompts": [
                {"name": "p0", "prompt": "hello"},
                {"name": "p1", "prompt": "world"},
            ],
        },
    )


def test_hipengine_metrics_contract_builder_self_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = suite._build_hipengine_metrics_contract(draft_max=2)
    suite.Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract(contract, candidate_budget=2)

    original = suite.Qwen35GGUFMTPAcceptStepMetrics.blocked_contract

    def fake_blocked_contract(
        cls,
        *,
        candidate_budget: int,
        blocked_until: str = "native_gguf_mtp_runtime",
    ) -> dict[str, object]:
        payload = original(candidate_budget=candidate_budget, blocked_until=blocked_until)
        payload["candidate_budget"] = int(candidate_budget) + 1
        return payload

    monkeypatch.setattr(
        suite.Qwen35GGUFMTPAcceptStepMetrics,
        "blocked_contract",
        classmethod(fake_blocked_contract),
    )
    with pytest.raises(ValueError, match="candidate_budget mismatch"):
        suite._build_hipengine_metrics_contract(draft_max=2)


def test_cli_gate_failures_report_invalid_metrics_contracts() -> None:
    single = {"status": "ready", "hipengine_metrics_contract_validation": {"passed": False}}
    matrix = {
        "status": "ready",
        "all_hipengine_metrics_contracts_valid": False,
        "hipengine_metrics_contract_validation_summary": {"failed_count": 1},
    }
    by_budget = {
        "status": "ready",
        "hipengine_metrics_contract_validation_by_budget": {"B2": {"passed": False}},
    }

    assert suite._has_invalid_hipengine_metrics_contracts(single) is True
    assert suite._has_invalid_hipengine_metrics_contracts(matrix) is True
    assert suite._has_invalid_hipengine_metrics_contracts(by_budget) is True
    assert "metrics_contract_invalid" in suite._cli_gate_failures(single)
    assert suite.CLI_GATE_EXIT_CODES["metrics_contract_invalid"] == 12


def test_b1_prompt_suite_cli_fail_on_metrics_contract_invalid_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "_hipengine_metrics_contract_validation",
        lambda contract, *, draft_max: {
            "passed": False,
            "validator": "synthetic.invalid",
            "candidate_budget": draft_max,
            "budget_label": f"B{draft_max}",
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-metrics-contract-invalid",
        ]
    )

    assert rc == 12
    artifact = json.loads(out.read_text())
    assert artifact["hipengine_metrics_contract_validation"]["passed"] is False
    assert "metrics_contract_invalid" in artifact["cli_gate_failures"]


def _token_inventory(path: Path, *, token_ids: list[int] | None = None) -> Path:
    tokens = [1, 2, 3] if token_ids is None else token_ids
    return _write_json(
        path,
        {
            "schema": 1,
            "kind": "hipengine_gguf_prompt_token_inventory",
            "prompts": [
                {
                    "name": "p0",
                    "token_ids": tokens,
                    "token_ids_sha256": "synthetic",
                    "rendered_sha256": "p0-hash",
                }
            ],
        },
    )


def _sampling(
    path: Path,
    *,
    draft_max: int = 1,
    draft_top_k: int = 10,
    draft_selection: str = "greedy_top1_from_topk",
    selected_index: int = 0,
) -> Path:
    return _write_json(
        path,
        {
            "schema": 1,
            "sampling": {
                "target": {"temperature": 0.0, "seed": 12345},
                "draft": {
                    "budget": f"B{draft_max}",
                    "draft_max": draft_max,
                    "selection": draft_selection,
                    "selected_index": selected_index,
                    "temperature": 0.0,
                    "top_k": draft_top_k,
                },
            },
        },
    )


def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        suite,
        "scan_gguf",
        lambda model: SimpleNamespace(
            architecture="qwen35moe",
            file_type_name="MOSTLY_Q4_K_M",
            tensor_count=753,
        ),
    )
    monkeypatch.setattr(
        suite,
        "validate_qwen35_gguf_mtp_blocks",
        lambda info: (
            SimpleNamespace(
                layer_id=40,
                tensor_names=tuple(f"tensor_{i}" for i in range(20)),
                nextn_tensor_names=(
                    "blk.40.nextn.eh_proj.weight",
                    "blk.40.nextn.enorm.weight",
                    "blk.40.nextn.hnorm.weight",
                    "blk.40.nextn.shared_head_norm.weight",
                ),
                optional_fallback_tensor_names={
                    "nextn.embed_tokens": "token_embd.weight",
                    "nextn.shared_head_head": "output.weight",
                },
            ),
        ),
    )
    draft_topk = {
        "kernel": ["cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h"],
        "top_k": 10,
        "selection": "greedy_top1_from_topk",
        "selected_index": 0,
    }
    call_spec = {
        "layer_id": 40,
        "cpu_reference_kernel": [
            "cpu_reference",
            "mtp_nextn_layer",
            "w4_gguf",
            "qwen35_dense_logits",
        ],
        "draft_topk": draft_topk,
        "dynamic_inputs": [
            {"argument": "hidden_seed", "required": True, "shape": ["tokens", 2048]},
            {"argument": "kv_base_offsets", "required": False, "shape": ["tokens", "logical_blocks"]},
            {"argument": "kv_live_counts", "required": False, "shape": ["tokens"]},
            {"argument": "kv_token_positions", "required": False, "shape": ["tokens"]},
            {"argument": "kv_evict_mask", "required": False, "shape": ["tokens", "max_live_count"]},
            {"argument": "block_size", "required": False, "shape": []},
        ],
    }
    monkeypatch.setattr(
        suite,
        "build_qwen35_gguf_mtp_draft_tensor_plans",
        lambda info, *, strict=True: (
            SimpleNamespace(
                layer_id=40,
                as_dict=lambda: {
                    "layer_id": 40,
                    "draft_topk": draft_topk,
                    "cpu_reference_call_spec": call_spec,
                },
                cpu_reference_call_spec=SimpleNamespace(as_dict=lambda: call_spec),
            ),
        ),
    )
    monkeypatch.setattr(
        suite,
        "run_oracle_gate",
        lambda fixture: {
            "passed": True,
            "fixture": str(fixture),
            "metrics": {"max_kl": 0.0, "top1_agreement": 1.0},
            "kvlivespans_paged_cache_smoke": {
                "passed": True,
                "max_abs_diff": 0.0,
                "dense_shape": [1, 4],
                "paged_shape": [1, 4],
                "cache_tokens": 2,
                "block_size": 2,
                "kv_base_offsets": [[0]],
                "kv_live_counts": [2],
                "kv_token_positions": [1],
            },
        },
    )


def _artifact_inputs(
    tmp_path: Path,
    *,
    mismatch: bool = False,
    draft_max: int = 1,
    hipengine_draft_top_k: int = 10,
    llamacpp_draft_top_k: int = 10,
) -> dict[str, Path]:
    return {
        "model": tmp_path / "model.gguf",
        "prompts_file": _prompt_suite(tmp_path / "prompts.json"),
        "hipengine_token_inventory": _token_inventory(tmp_path / "hip.json"),
        "llamacpp_token_inventory": _token_inventory(
            tmp_path / "llama.json",
            token_ids=[1, 7, 3] if mismatch else [1, 2, 3],
        ),
        "hipengine_sampling": _sampling(
            tmp_path / "hip-sampling.json",
            draft_max=draft_max,
            draft_top_k=hipengine_draft_top_k,
        ),
        "llamacpp_sampling": _sampling(
            tmp_path / "llama-sampling.json",
            draft_max=draft_max,
            draft_top_k=llamacpp_draft_top_k,
        ),
    }


def test_b1_prompt_suite_preflight_blocks_only_on_missing_runtime_when_preconditions_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    artifact = suite.build_b1_prompt_suite_artifact(**_artifact_inputs(tmp_path))

    assert artifact["kind"] == "hipengine_gguf_mtp_b1_prompt_suite"
    assert artifact["mode"] == "preflight"
    assert artifact["status"] == "blocked"
    assert artifact["cli_gate_exit_codes"] == EXPECTED_CLI_GATE_EXIT_CODES
    assert artifact["cli_gate_failures"] == [
        "blocked",
        "noncomparable_accepted_output",
        "native_runtime_missing",
        "performance_unready",
    ]
    assert artifact["cli_gate_failure_exit_codes"] == {
        "blocked": 2,
        "noncomparable_accepted_output": 4,
        "native_runtime_missing": 7,
        "performance_unready": 5,
    }
    assert suite.CLI_GATE_EXIT_CODES == EXPECTED_CLI_GATE_EXIT_CODES
    assert artifact["backend"] == "hip_gfx1100"
    assert artifact["budget"] == "B1"
    assert artifact["draft_max"] == 1
    assert artifact["prompt_names"] == ["p0", "p1"]
    assert artifact["validated_mtp_blocks"] == [
        {
            "layer_id": 40,
            "tensor_count": 20,
            "nextn_tensor_count": 4,
            "optional_fallback_tensor_names": {
                "nextn.embed_tokens": "token_embd.weight",
                "nextn.shared_head_head": "output.weight",
            },
        }
    ]
    assert artifact["parity_precheck"]["all_pass"] is True
    assert artifact["mtp_draft_tensor_plans"] == [
        {
            "layer_id": 40,
            "draft_topk": artifact["mtp_draft_call_specs"][0]["draft_topk"],
            "cpu_reference_call_spec": artifact["mtp_draft_call_specs"][0],
        }
    ]
    assert artifact["mtp_draft_call_specs"][0]["draft_topk"] == {
        "kernel": ["cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h"],
        "top_k": 10,
        "selection": "greedy_top1_from_topk",
        "selected_index": 0,
    }
    dynamic_args = [item["argument"] for item in artifact["mtp_draft_call_specs"][0]["dynamic_inputs"]]
    assert dynamic_args == [
        "hidden_seed",
        "kv_base_offsets",
        "kv_live_counts",
        "kv_token_positions",
        "kv_evict_mask",
        "block_size",
    ]
    assert artifact["draft_budget_precheck"] == {
        "checked": True,
        "passed": True,
        "expected": {"budget": "B1", "draft_max": 1},
        "observed": {
            "hipengine": {"budget": "B1", "draft_max": 1},
            "llamacpp": {"budget": "B1", "draft_max": 1},
        },
        "mismatches": [],
    }
    assert artifact["draft_sampling_contract_precheck"] == {
        "checked": True,
        "passed": True,
        "expected": {"top_k": 10, "selection": "greedy_top1_from_topk", "selected_index": 0},
        "observed": {
            "hipengine": {"top_k": 10, "selection": "greedy_top1_from_topk", "selected_index": 0},
            "llamacpp": {"top_k": 10, "selection": "greedy_top1_from_topk", "selected_index": 0},
        },
        "mismatches": [],
    }
    assert artifact["hidden_seed_contract_precheck"]["passed"] is True
    assert artifact["hidden_seed_contract_precheck"]["hidden_size"] == 2048
    assert artifact["hidden_seed_contract_precheck"]["required_contract"]["dtype"] == "FP32"
    assert artifact["hidden_seed_contract_precheck"]["required_contract"]["provenance"] == "post_output_norm"
    assert artifact["hidden_seed_contract_precheck"]["required_contract"]["ready_for_mtp"] is True
    assert artifact["hidden_seed_contract_precheck"]["default_ar_contract"]["dtype"] == "BF16"
    assert artifact["hidden_seed_contract_precheck"]["default_ar_contract"]["ready_for_mtp"] is False
    assert artifact["hidden_seed_contract_precheck"]["dynamic_input"] == {
        "argument": "hidden_seed",
        "required": True,
        "shape": ["tokens", 2048],
    }
    assert artifact["target_context_contract"] == Qwen35GGUFMTPContext.contract()
    assert artifact["target_context_contract"]["validator"] == "Qwen35GGUFMTPContext.validate_payload"
    assert artifact["runtime_kernel_precheck"]["backend"] == "hip_gfx1100"
    assert artifact["runtime_kernel_precheck"]["exactness_oracles_ready"] is True
    assert artifact["runtime_kernel_precheck"]["native_runtime_kernels_ready"] is False
    assert artifact["runtime_kernel_precheck"]["optimization_kernels_ready"] is True
    assert artifact["runtime_kernel_precheck"]["missing_exactness_oracle_keys"] == []
    assert artifact["runtime_kernel_precheck"]["missing_native_runtime_keys"] == [
        ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"],
        ["hip_gfx1100", "paged_kv_write", "w4_gguf", "mixed_bf16_spans"],
        ["hip_gfx1100", "paged_attn_decode", "w4_gguf", "bf16_context_spans"],
    ]
    assert artifact["runtime_kernel_precheck"]["missing_optimization_keys"] == []
    assert artifact["oracle_gate"]["passed"] is True
    assert artifact["llamacpp_trace_oracle"]["passed"] is True
    assert artifact["llamacpp_trace_oracle"]["selected_token_ids"] == [8068, 271]
    assert artifact["llamacpp_trace_oracle"]["observed_top_k"] == 3
    assert artifact["llamacpp_trace_oracle"]["requested_draft_max"] == 1
    assert artifact["llamacpp_trace_oracle"]["max_generated_per_call"] == 1
    assert (
        artifact["llamacpp_trace_oracle"]["budget_coverage"]
        == "full_requested_budget_exercised"
    )
    assert artifact["llamacpp_trace_oracle"]["denominator_metrics"] == {
        "accepted_draft_tokens": 0,
        "generated_draft_tokens": 2,
        "accepted_per_draft": 0.0,
        "accepted_per_draft_status": suite.ACCEPTED_DRAFT_COMPARABLE,
        "visible_output_token_count": None,
        "accepted_per_output": None,
        "accepted_per_output_status": "not_comparable_debug_trace_missing_visible_output_count",
        "denominators": {
            "accepted_per_draft": "accepted_draft_tokens / generated_draft_tokens",
            "accepted_per_output": "accepted_draft_tokens / visible_output_token_count",
        },
    }
    assert artifact["hipengine_metrics_contract"] == {
        "status": "not_run",
        "blocked_until": "native_gguf_mtp_runtime",
        "schema": 1,
        "kind": "hipengine_gguf_mtp_accept_step_metrics",
        "source": "Qwen35GGUFMTPAcceptStepMetrics",
        "result_source": "Qwen35GGUFMTPAcceptStep",
        "draft_max": 1,
        "candidate_budget": 1,
        "budget_label": "B1",
        "required_fields": list(suite.Qwen35GGUFMTPAcceptStepMetrics.required_fields()),
        "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_payload",
        "denominators": {
            "accepted_per_draft": "accepted_token_count / draft_token_count",
            "accepted_per_output": "accepted_token_count / output_token_count",
        },
    }
    suite.Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract(
        artifact["hipengine_metrics_contract"],
        candidate_budget=1,
    )
    assert artifact["hipengine_metrics_contract_validation"] == {
        "passed": True,
        "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
        "candidate_budget": 1,
        "budget_label": "B1",
    }
    assert artifact["execution"] == {
        "implemented": False,
        "exactness_gate": "passed",
        "accepted_output_metrics": "not_run",
        "next_action": "implement native GGUF MTP draft execution and re-run this harness for B1",
    }
    assert artifact["blockers"] == [
        {
            "code": "native_gguf_mtp_runtime_missing",
            "detail": (
                "Native GGUF MTP draft execution is not implemented yet; this harness "
                "stops after metadata/token/sampling/runtime-kernel preflight instead of reporting metrics."
            ),
            "missing_native_runtime_keys": [
                ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"],
                ["hip_gfx1100", "paged_kv_write", "w4_gguf", "mixed_bf16_spans"],
                ["hip_gfx1100", "paged_attn_decode", "w4_gguf", "bf16_context_spans"],
            ],
            "missing_optimization_keys": [],
        }
    ]


def test_b1_prompt_suite_preflight_can_request_b4_when_sampling_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    artifact = suite.build_b1_prompt_suite_artifact(
        **_artifact_inputs(tmp_path, draft_max=4),
        draft_max=4,
    )

    assert artifact["budget"] == "B4"
    assert artifact["draft_max"] == 4
    assert artifact["draft_budget_precheck"]["passed"] is True
    assert artifact["draft_budget_precheck"]["expected"] == {"budget": "B4", "draft_max": 4}
    assert artifact["blockers"] == [
        {
            "code": "native_gguf_mtp_runtime_missing",
            "detail": (
                "Native GGUF MTP draft execution is not implemented yet; this harness "
                "stops after metadata/token/sampling/runtime-kernel preflight instead of reporting metrics."
            ),
            "missing_native_runtime_keys": [
                ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"],
                ["hip_gfx1100", "paged_kv_write", "w4_gguf", "mixed_bf16_spans"],
                ["hip_gfx1100", "paged_attn_decode", "w4_gguf", "bf16_context_spans"],
            ],
            "missing_optimization_keys": [],
        }
    ]


def test_b1_prompt_suite_default_sampling_fixtures_cover_b1_to_b4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    for draft_max in (1, 2, 3, 4):
        fixture = suite.default_sampling_fixture(draft_max)
        artifact = suite.build_b1_prompt_suite_artifact(
            model=tmp_path / "model.gguf",
            prompts_file=_prompt_suite(tmp_path / f"prompts-b{draft_max}.json"),
            hipengine_token_inventory=_token_inventory(tmp_path / f"hip-b{draft_max}.json"),
            llamacpp_token_inventory=_token_inventory(tmp_path / f"llama-b{draft_max}.json"),
            hipengine_sampling=fixture,
            llamacpp_sampling=fixture,
            draft_max=draft_max,
        )

        assert artifact["budget"] == f"B{draft_max}"
        assert artifact["draft_budget_precheck"]["passed"] is True
        assert artifact["draft_sampling_contract_precheck"]["passed"] is True
        assert artifact["blockers"][0]["code"] == "native_gguf_mtp_runtime_missing"

    with pytest.raises(suite.B1PromptSuitePreflightError, match="draft_max"):
        suite.default_sampling_fixture(5)


def test_b1_prompt_suite_matrix_builds_budget_matched_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    matrix = suite.build_b1_b4_prompt_suite_matrix(
        model=tmp_path / "model.gguf",
        prompts_file=_prompt_suite(tmp_path / "prompts-matrix.json"),
        hipengine_token_inventory=_token_inventory(tmp_path / "hip-matrix.json"),
        llamacpp_token_inventory=_token_inventory(tmp_path / "llama-matrix.json"),
        prompt_limit=1,
    )

    assert matrix["kind"] == "hipengine_gguf_mtp_b1_b4_prompt_suite_matrix"
    assert matrix["status"] == "blocked"
    assert matrix["target_context_contract"] == Qwen35GGUFMTPContext.contract()
    assert matrix["target_context_contract"]["validator"] == "Qwen35GGUFMTPContext.validate_payload"
    assert matrix["cli_gate_exit_codes"] == EXPECTED_CLI_GATE_EXIT_CODES
    assert matrix["cli_gate_failures"] == [
        "blocked",
        "partial_trace_budget",
        "noncomparable_accepted_output",
        "native_runtime_missing",
        "performance_unready",
    ]
    assert matrix["cli_gate_failures_by_budget"] == {
        "B1": [
            "blocked",
            "noncomparable_accepted_output",
            "native_runtime_missing",
            "performance_unready",
        ],
        "B2": [
            "blocked",
            "partial_trace_budget",
            "noncomparable_accepted_output",
            "native_runtime_missing",
            "performance_unready",
        ],
        "B3": [
            "blocked",
            "partial_trace_budget",
            "noncomparable_accepted_output",
            "native_runtime_missing",
            "performance_unready",
        ],
        "B4": [
            "blocked",
            "partial_trace_budget",
            "noncomparable_accepted_output",
            "native_runtime_missing",
            "performance_unready",
        ],
    }
    assert matrix["budgets"] == ["B1", "B2", "B3", "B4"]
    assert matrix["hipengine_metrics_contract_by_budget"]["B1"]["budget_label"] == "B1"
    assert matrix["hipengine_metrics_contract_by_budget"]["B4"]["candidate_budget"] == 4
    for budget, contract in matrix["hipengine_metrics_contract_by_budget"].items():
        suite.Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract(
            contract,
            candidate_budget=int(budget[1:]),
        )
    assert matrix["hipengine_metrics_contract_validation_by_budget"] == {
        "B1": {
            "passed": True,
            "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
            "candidate_budget": 1,
            "budget_label": "B1",
        },
        "B2": {
            "passed": True,
            "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
            "candidate_budget": 2,
            "budget_label": "B2",
        },
        "B3": {
            "passed": True,
            "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
            "candidate_budget": 3,
            "budget_label": "B3",
        },
        "B4": {
            "passed": True,
            "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
            "candidate_budget": 4,
            "budget_label": "B4",
        },
    }
    assert matrix["hipengine_metrics_contract_validation_summary"] == {
        "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
        "total_budgets": 4,
        "passed_count": 4,
        "failed_count": 0,
        "passed_budgets": ["B1", "B2", "B3", "B4"],
        "failed_budgets": [],
    }
    assert matrix["all_hipengine_metrics_contracts_valid"] is True
    assert matrix["draft_max_values"] == [1, 2, 3, 4]
    assert matrix["artifact_count"] == 4
    assert matrix["artifacts_included"] is True
    assert len(matrix["artifacts"]) == 4
    assert matrix["all_parity_prechecks_pass"] is True
    assert matrix["all_budget_prechecks_pass"] is True
    assert matrix["all_sampling_contract_prechecks_pass"] is True
    assert matrix["all_hidden_seed_contract_prechecks_pass"] is True
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B1"] == matrix["artifacts"][0]["hidden_seed_contract_precheck"]
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B4"] == matrix["artifacts"][3]["hidden_seed_contract_precheck"]
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B1"]["passed"] is True
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B4"]["required_contract"]["dtype"] == "FP32"
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B4"]["required_contract"]["provenance"] == "post_output_norm"
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B4"]["required_contract"]["ready_for_mtp"] is True
    assert matrix["all_exactness_gates_pass"] is True
    assert matrix["oracle_gate_by_budget"]["B1"] == matrix["artifacts"][0]["oracle_gate"]
    assert matrix["oracle_gate_by_budget"]["B4"] == matrix["artifacts"][3]["oracle_gate"]
    assert matrix["oracle_gate_by_budget"]["B1"]["passed"] is True
    assert matrix["oracle_gate_by_budget"]["B4"]["metrics"]["max_kl"] == 0.0
    assert matrix["oracle_gate_by_budget"]["B4"]["kvlivespans_paged_cache_smoke"]["passed"] is True
    assert matrix["all_kvlivespans_paged_cache_smokes_pass"] is True
    assert matrix["kvlivespans_paged_cache_smoke_by_budget"] == {
        "B1": True,
        "B2": True,
        "B3": True,
        "B4": True,
    }
    assert matrix["kvlivespans_paged_cache_max_abs_diff_by_budget"] == {
        "B1": 0.0,
        "B2": 0.0,
        "B3": 0.0,
        "B4": 0.0,
    }
    assert matrix["all_llamacpp_trace_budgets_full"] is False
    assert matrix["llamacpp_trace_budget_coverage_by_budget"] == {
        "B1": "full_requested_budget_exercised",
        "B2": "partial_trace_did_not_exercise_full_budget",
        "B3": "partial_trace_did_not_exercise_full_budget",
        "B4": "partial_trace_did_not_exercise_full_budget",
    }
    assert matrix["llamacpp_trace_oracle_by_budget"]["B1"] == matrix["artifacts"][0]["llamacpp_trace_oracle"]
    assert matrix["llamacpp_trace_oracle_by_budget"]["B4"] == matrix["artifacts"][3]["llamacpp_trace_oracle"]
    assert matrix["llamacpp_trace_oracle_by_budget"]["B1"]["denominator_metrics"]["accepted_per_draft"] == 0.0
    assert matrix["llamacpp_trace_oracle_by_budget"]["B4"]["budget_coverage"] == (
        "partial_trace_did_not_exercise_full_budget"
    )
    assert matrix["partial_llamacpp_trace_budget_budgets"] == ["B2", "B3", "B4"]
    assert matrix["all_accepted_per_draft_metrics_comparable"] is True
    assert matrix["accepted_per_draft_status_by_budget"] == {
        "B1": suite.ACCEPTED_DRAFT_COMPARABLE,
        "B2": suite.ACCEPTED_DRAFT_COMPARABLE,
        "B3": suite.ACCEPTED_DRAFT_COMPARABLE,
        "B4": suite.ACCEPTED_DRAFT_COMPARABLE,
    }
    assert matrix["noncomparable_accepted_per_draft_budgets"] == []
    assert matrix["all_accepted_per_output_metrics_comparable"] is False
    assert matrix["accepted_per_output_status_by_budget"] == {
        "B1": suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
        "B2": suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
        "B3": suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
        "B4": suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
    }
    assert matrix["noncomparable_accepted_per_output_budgets"] == ["B1", "B2", "B3", "B4"]
    assert matrix["all_native_runtime_kernels_ready"] is False
    assert matrix["all_optimization_kernels_ready"] is True
    assert matrix["runtime_kernel_precheck_by_budget"]["B1"] == matrix["artifacts"][0]["runtime_kernel_precheck"]
    assert matrix["runtime_kernel_precheck_by_budget"]["B4"] == matrix["artifacts"][3]["runtime_kernel_precheck"]
    for precheck in matrix["runtime_kernel_precheck_by_budget"].values():
        suite.Qwen35GGUFMTPRuntimeKernelPlan.validate_payload(precheck)
    assert matrix["all_performance_comparisons_ready"] is False
    assert matrix["performance_readiness_contract"] == Qwen35GGUFMTPPerformanceReadiness.contract()
    assert matrix["performance_readiness_contract"]["validator"] == (
        "Qwen35GGUFMTPPerformanceReadiness.validate_payload"
    )
    assert "validator" in matrix["performance_readiness_contract"]["required_fields"]
    assert matrix["performance_comparison_ready_by_budget"] == {
        "B1": False,
        "B2": False,
        "B3": False,
        "B4": False,
    }
    assert matrix["performance_unready_budgets"] == ["B1", "B2", "B3", "B4"]
    assert matrix["performance_comparison_blockers_by_budget"]["B1"] == [
        "accepted_output_denominator_not_comparable",
        "native_runtime_kernels_missing",
        "hipengine_metrics_not_ready",
    ]
    assert matrix["performance_comparison_blockers_by_budget"]["B4"] == [
        "partial_llamacpp_trace_budget_coverage",
        "accepted_output_denominator_not_comparable",
        "native_runtime_kernels_missing",
        "hipengine_metrics_not_ready",
    ]
    assert matrix["performance_readiness_by_budget"]["B1"] == {
        "ready": False,
        "blockers": [
            "accepted_output_denominator_not_comparable",
            "native_runtime_kernels_missing",
            "hipengine_metrics_not_ready",
        ],
        "known_blockers": list(Qwen35GGUFMTPPerformanceReadiness.known_blockers()),
        "required_fields": list(Qwen35GGUFMTPPerformanceReadiness.required_fields()),
        "validator": "Qwen35GGUFMTPPerformanceReadiness.validate_payload",
    }
    assert matrix["performance_readiness_by_budget"]["B4"]["blockers"] == [
        "partial_llamacpp_trace_budget_coverage",
        "accepted_output_denominator_not_comparable",
        "native_runtime_kernels_missing",
        "hipengine_metrics_not_ready",
    ]
    for readiness in matrix["performance_readiness_by_budget"].values():
        Qwen35GGUFMTPPerformanceReadiness.validate_payload(readiness)
    assert matrix["readiness_by_budget"]["B1"] == {
        "status": "blocked",
        "draft_max": 1,
        "parity_precheck": True,
        "draft_budget_precheck": True,
        "draft_sampling_contract_precheck": True,
        "hidden_seed_contract_precheck": True,
        "exactness_gate": "passed",
        "kvlivespans_paged_cache_smoke": True,
        "kvlivespans_paged_cache_max_abs_diff": 0.0,
        "llamacpp_trace_budget_coverage": "full_requested_budget_exercised",
        "accepted_per_draft_status": suite.ACCEPTED_DRAFT_COMPARABLE,
        "accepted_per_output_status": suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE,
        "native_runtime_kernels_ready": False,
        "optimization_kernels_ready": True,
        "missing_native_runtime_keys": [
            ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"],
            ["hip_gfx1100", "paged_kv_write", "w4_gguf", "mixed_bf16_spans"],
            ["hip_gfx1100", "paged_attn_decode", "w4_gguf", "bf16_context_spans"],
        ],
        "missing_optimization_keys": [],
        "metrics_contract_status": "not_run",
        "blocker_codes": ["native_gguf_mtp_runtime_missing"],
        "performance_comparison_blockers": [
            "accepted_output_denominator_not_comparable",
            "native_runtime_kernels_missing",
            "hipengine_metrics_not_ready",
        ],
        "performance_comparison_ready": False,
    }
    assert matrix["readiness_by_budget"]["B4"]["draft_max"] == 4
    assert (
        matrix["readiness_by_budget"]["B4"]["llamacpp_trace_budget_coverage"]
        == "partial_trace_did_not_exercise_full_budget"
    )
    assert matrix["readiness_by_budget"]["B4"]["blocker_codes"] == ["native_gguf_mtp_runtime_missing"]
    assert matrix["blocker_codes_by_budget"] == {
        "B1": ["native_gguf_mtp_runtime_missing"],
        "B2": ["native_gguf_mtp_runtime_missing"],
        "B3": ["native_gguf_mtp_runtime_missing"],
        "B4": ["native_gguf_mtp_runtime_missing"],
    }
    assert [item["draft_budget_precheck"]["expected"] for item in matrix["artifacts"]] == [
        {"budget": "B1", "draft_max": 1},
        {"budget": "B2", "draft_max": 2},
        {"budget": "B3", "draft_max": 3},
        {"budget": "B4", "draft_max": 4},
    ]
    assert [item["hipengine_metrics_contract"]["draft_max"] for item in matrix["artifacts"]] == [1, 2, 3, 4]
    assert [item["hipengine_metrics_contract"]["candidate_budget"] for item in matrix["artifacts"]] == [
        1,
        2,
        3,
        4,
    ]
    assert [item["hipengine_metrics_contract"]["budget_label"] for item in matrix["artifacts"]] == [
        "B1",
        "B2",
        "B3",
        "B4",
    ]
    assert all(
        item["hipengine_metrics_contract"]["kind"] == "hipengine_gguf_mtp_accept_step_metrics"
        for item in matrix["artifacts"]
    )
    assert all(
        item["hipengine_metrics_contract"]["validator"]
        == "Qwen35GGUFMTPAcceptStepMetrics.validate_payload"
        for item in matrix["artifacts"]
    )
    for item in matrix["artifacts"]:
        suite.Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract(
            item["hipengine_metrics_contract"],
            candidate_budget=item["draft_max"],
        )
    assert all(
        "step_rows" in item["hipengine_metrics_contract"]["required_fields"]
        for item in matrix["artifacts"]
    )
    assert all(
        "steps" in item["hipengine_metrics_contract"]["required_fields"]
        for item in matrix["artifacts"]
    )


def test_b1_prompt_suite_matrix_can_omit_child_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    matrix = suite.build_b1_b4_prompt_suite_matrix(
        model=tmp_path / "model.gguf",
        prompts_file=_prompt_suite(tmp_path / "prompts-compact.json"),
        hipengine_token_inventory=_token_inventory(tmp_path / "hip-compact.json"),
        llamacpp_token_inventory=_token_inventory(tmp_path / "llama-compact.json"),
        prompt_limit=1,
        include_artifacts=False,
    )

    assert matrix["artifact_count"] == 4
    assert matrix["artifacts_included"] is False
    assert "artifacts" not in matrix
    assert matrix["target_context_contract"] == Qwen35GGUFMTPContext.contract()
    assert matrix["target_context_contract"]["validator"] == "Qwen35GGUFMTPContext.validate_payload"
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B1"]["passed"] is True
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B4"]["hidden_size"] == 2048
    assert matrix["hidden_seed_contract_precheck_by_budget"]["B4"]["dynamic_input"] == {
        "argument": "hidden_seed",
        "required": True,
        "shape": ["tokens", 2048],
    }
    assert "artifacts" not in matrix["hidden_seed_contract_precheck_by_budget"]["B1"]
    assert matrix["hipengine_metrics_contract_by_budget"]["B1"]["budget_label"] == "B1"
    assert matrix["hipengine_metrics_contract_by_budget"]["B4"]["candidate_budget"] == 4
    for budget, contract in matrix["hipengine_metrics_contract_by_budget"].items():
        suite.Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract(
            contract,
            candidate_budget=int(budget[1:]),
        )
    assert matrix["hipengine_metrics_contract_validation_by_budget"]["B1"] == {
        "passed": True,
        "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
        "candidate_budget": 1,
        "budget_label": "B1",
    }
    assert matrix["hipengine_metrics_contract_validation_by_budget"]["B4"] == {
        "passed": True,
        "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
        "candidate_budget": 4,
        "budget_label": "B4",
    }
    assert matrix["hipengine_metrics_contract_validation_summary"] == {
        "validator": "Qwen35GGUFMTPAcceptStepMetrics.validate_blocked_contract",
        "total_budgets": 4,
        "passed_count": 4,
        "failed_count": 0,
        "passed_budgets": ["B1", "B2", "B3", "B4"],
        "failed_budgets": [],
    }
    assert matrix["all_hipengine_metrics_contracts_valid"] is True
    assert matrix["oracle_gate_by_budget"]["B1"]["passed"] is True
    assert matrix["oracle_gate_by_budget"]["B4"]["metrics"]["top1_agreement"] == 1.0
    assert matrix["oracle_gate_by_budget"]["B4"]["kvlivespans_paged_cache_smoke"]["passed"] is True
    assert "artifacts" not in matrix["oracle_gate_by_budget"]["B1"]
    assert matrix["all_kvlivespans_paged_cache_smokes_pass"] is True
    assert matrix["kvlivespans_paged_cache_max_abs_diff_by_budget"]["B1"] == 0.0
    assert matrix["all_llamacpp_trace_budgets_full"] is False
    assert matrix["partial_llamacpp_trace_budget_budgets"] == ["B2", "B3", "B4"]
    assert matrix["llamacpp_trace_oracle_by_budget"]["B1"]["budget_coverage"] == (
        "full_requested_budget_exercised"
    )
    assert matrix["llamacpp_trace_oracle_by_budget"]["B4"]["denominator_metrics"][
        "accepted_per_output_status"
    ] == suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE
    assert "artifacts" not in matrix["llamacpp_trace_oracle_by_budget"]["B1"]
    assert matrix["all_accepted_per_draft_metrics_comparable"] is True
    assert matrix["noncomparable_accepted_per_draft_budgets"] == []
    assert matrix["all_accepted_per_output_metrics_comparable"] is False
    assert matrix["noncomparable_accepted_per_output_budgets"] == ["B1", "B2", "B3", "B4"]
    assert matrix["all_performance_comparisons_ready"] is False
    assert matrix["runtime_kernel_precheck_by_budget"]["B1"]["backend"] == "hip_gfx1100"
    assert matrix["runtime_kernel_precheck_by_budget"]["B4"]["native_runtime_kernels_ready"] is False
    assert matrix["runtime_kernel_precheck_by_budget"]["B4"]["missing_native_runtime_keys"] == [
        ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"],
        ["hip_gfx1100", "paged_kv_write", "w4_gguf", "mixed_bf16_spans"],
        ["hip_gfx1100", "paged_attn_decode", "w4_gguf", "bf16_context_spans"],
    ]
    for precheck in matrix["runtime_kernel_precheck_by_budget"].values():
        suite.Qwen35GGUFMTPRuntimeKernelPlan.validate_payload(precheck)
    assert matrix["performance_unready_budgets"] == ["B1", "B2", "B3", "B4"]
    assert matrix["performance_readiness_by_budget"]["B1"]["validator"] == (
        "Qwen35GGUFMTPPerformanceReadiness.validate_payload"
    )
    assert matrix["performance_readiness_by_budget"]["B4"]["ready"] is False
    assert "artifacts" not in matrix["performance_readiness_by_budget"]["B1"]
    assert matrix["cli_gate_failures_by_budget"]["B1"] == [
        "blocked",
        "noncomparable_accepted_output",
        "native_runtime_missing",
        "performance_unready",
    ]
    assert matrix["cli_gate_failures_by_budget"]["B4"] == [
        "blocked",
        "partial_trace_budget",
        "noncomparable_accepted_output",
        "native_runtime_missing",
        "performance_unready",
    ]
    assert matrix["cli_gate_failure_exit_codes_by_budget"]["B1"] == {
        "blocked": 2,
        "noncomparable_accepted_output": 4,
        "native_runtime_missing": 7,
        "performance_unready": 5,
    }
    assert matrix["cli_gate_failure_exit_codes_by_budget"]["B4"] == {
        "blocked": 2,
        "partial_trace_budget": 3,
        "noncomparable_accepted_output": 4,
        "native_runtime_missing": 7,
        "performance_unready": 5,
    }
    assert matrix["cli_gate_failure_exit_codes"] == {
        "blocked": 2,
        "partial_trace_budget": 3,
        "noncomparable_accepted_output": 4,
        "native_runtime_missing": 7,
        "performance_unready": 5,
    }
    assert matrix["readiness_by_budget"]["B1"]["blocker_codes"] == ["native_gguf_mtp_runtime_missing"]
    assert matrix["readiness_by_budget"]["B4"]["draft_max"] == 4
    assert (
        matrix["readiness_by_budget"]["B4"]["llamacpp_trace_budget_coverage"]
        == "partial_trace_did_not_exercise_full_budget"
    )


def test_b1_prompt_suite_cli_fail_on_partial_trace_budget_skips_b1_full_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-partial-trace-budget",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["llamacpp_trace_oracle"]["budget_coverage"] == suite.FULL_TRACE_BUDGET_COVERAGE


def test_b1_prompt_suite_cli_fail_on_partial_trace_budget_rejects_b4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path, draft_max=4)
    out = tmp_path / "b4-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--draft-max",
            "4",
            "--out",
            str(out),
            "--fail-on-partial-trace-budget",
        ]
    )

    assert rc == 3
    artifact = json.loads(out.read_text())
    assert artifact["budget"] == "B4"
    assert artifact["llamacpp_trace_oracle"]["budget_coverage"] == suite.PARTIAL_TRACE_BUDGET_COVERAGE


def test_b1_prompt_suite_cli_fail_on_partial_trace_budget_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-partial-trace-budget",
        ]
    )

    assert rc == 3
    matrix = json.loads(out.read_text())
    assert matrix["partial_llamacpp_trace_budget_budgets"] == ["B2", "B3", "B4"]


def test_b1_prompt_suite_cli_fail_on_precheck_allows_default_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-precheck-fail",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["draft_budget_precheck"]["passed"] is True


def test_b1_prompt_suite_cli_fail_on_precheck_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "_build_draft_budget_precheck",
        lambda **kwargs: {
            "passed": False,
            "expected": {"budget": "B1", "draft_max": 1},
            "observed": {"draft_max": 0},
            "mismatches": ["draft_max"],
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-precheck-fail",
        ]
    )

    assert rc == 11
    artifact = json.loads(out.read_text())
    assert artifact["draft_budget_precheck"]["passed"] is False
    assert artifact["blockers"][0]["code"] == "draft_budget_mismatch"


def test_b1_prompt_suite_cli_fail_on_precheck_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "_build_draft_budget_precheck",
        lambda **kwargs: {
            "passed": False,
            "expected": {"budget": "B1", "draft_max": 1},
            "observed": {"draft_max": 0},
            "mismatches": ["draft_max"],
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-precheck-fail",
        ]
    )

    assert rc == 11
    matrix = json.loads(out.read_text())
    assert matrix["all_budget_prechecks_pass"] is False


def test_b1_prompt_suite_cli_fail_on_exactness_allows_default_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-exactness-fail",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["execution"]["exactness_gate"] == "passed"


def test_b1_prompt_suite_cli_fail_on_exactness_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "run_oracle_gate",
        lambda fixture: {
            "passed": False,
            "fixture": str(fixture),
            "metrics": {"max_kl": 0.5, "top1_agreement": 0.0},
            "kvlivespans_paged_cache_smoke": {
                "passed": True,
                "max_abs_diff": 0.0,
            },
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-exactness-fail",
        ]
    )

    assert rc == 10
    artifact = json.loads(out.read_text())
    assert artifact["execution"]["exactness_gate"] == "failed"
    assert artifact["blockers"][0]["code"] == "oracle_gate_failed"


def test_b1_prompt_suite_cli_fail_on_exactness_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "run_oracle_gate",
        lambda fixture: {
            "passed": False,
            "fixture": str(fixture),
            "metrics": {"max_kl": 0.5, "top1_agreement": 0.0},
            "kvlivespans_paged_cache_smoke": {
                "passed": True,
                "max_abs_diff": 0.0,
            },
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-exactness-fail",
        ]
    )

    assert rc == 10
    matrix = json.loads(out.read_text())
    assert matrix["all_exactness_gates_pass"] is False


def test_b1_prompt_suite_cli_fail_on_kvlivespans_smoke_allows_default_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-kvlivespans-smoke-fail",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["oracle_gate"]["kvlivespans_paged_cache_smoke"]["passed"] is True


def test_b1_prompt_suite_cli_fail_on_kvlivespans_smoke_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "run_oracle_gate",
        lambda fixture: {
            "passed": True,
            "fixture": str(fixture),
            "metrics": {"max_kl": 0.0, "top1_agreement": 1.0},
            "kvlivespans_paged_cache_smoke": {
                "passed": False,
                "max_abs_diff": 1.0,
            },
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-kvlivespans-smoke-fail",
        ]
    )

    assert rc == 9
    artifact = json.loads(out.read_text())
    assert artifact["oracle_gate"]["kvlivespans_paged_cache_smoke"]["passed"] is False


def test_b1_prompt_suite_cli_fail_on_kvlivespans_smoke_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "run_oracle_gate",
        lambda fixture: {
            "passed": True,
            "fixture": str(fixture),
            "metrics": {"max_kl": 0.0, "top1_agreement": 1.0},
            "kvlivespans_paged_cache_smoke": {
                "passed": False,
                "max_abs_diff": 1.0,
            },
        },
    )
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-kvlivespans-smoke-fail",
        ]
    )

    assert rc == 9
    matrix = json.loads(out.read_text())
    assert matrix["all_kvlivespans_paged_cache_smokes_pass"] is False


def test_b1_prompt_suite_cli_fail_on_noncomparable_accepted_output_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-noncomparable-accepted-output",
        ]
    )

    assert rc == 4
    artifact = json.loads(out.read_text())
    assert (
        artifact["llamacpp_trace_oracle"]["denominator_metrics"]["accepted_per_output_status"]
        == suite.ACCEPTED_OUTPUT_NOT_COMPARABLE_DEBUG_TRACE
    )


def test_b1_prompt_suite_cli_fail_on_noncomparable_accepted_output_allows_comparable_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    trace_payload = json.loads(suite.DEFAULT_LLAMACPP_TRACE_FIXTURE.read_text())
    trace_payload["visible_output_token_count"] = 2
    trace_fixture = _write_json(tmp_path / "llamacpp-trace.json", trace_payload)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--llamacpp-trace-fixture",
            str(trace_fixture),
            "--out",
            str(out),
            "--fail-on-noncomparable-accepted-output",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert (
        artifact["llamacpp_trace_oracle"]["denominator_metrics"]["accepted_per_output_status"]
        == suite.ACCEPTED_OUTPUT_COMPARABLE
    )


def test_b1_prompt_suite_cli_fail_on_noncomparable_accepted_output_rejects_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-noncomparable-accepted-output",
        ]
    )

    assert rc == 4
    matrix = json.loads(out.read_text())
    assert matrix["noncomparable_accepted_per_output_budgets"] == ["B1", "B2", "B3", "B4"]


def test_b1_prompt_suite_cli_fail_on_noncomparable_accepted_draft_allows_default_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-noncomparable-accepted-draft",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert (
        artifact["llamacpp_trace_oracle"]["denominator_metrics"]["accepted_per_draft_status"]
        == suite.ACCEPTED_DRAFT_COMPARABLE
    )


def test_b1_prompt_suite_cli_fail_on_noncomparable_accepted_draft_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    trace_payload = json.loads(suite.DEFAULT_LLAMACPP_TRACE_FIXTURE.read_text())
    for call in trace_payload["calls"]:
        call["generated"] = 0
        call["accepted"] = 0
        call["accept_generated"] = 0
    for metrics in (trace_payload["summary"], trace_payload["llamacpp_timing_summary"]):
        metrics["draft_n"] = 0
        metrics["draft_n_accepted"] = 0
        metrics["draft_acceptance"] = None
    trace_fixture = _write_json(tmp_path / "llamacpp-zero-draft-trace.json", trace_payload)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--llamacpp-trace-fixture",
            str(trace_fixture),
            "--out",
            str(out),
            "--fail-on-noncomparable-accepted-draft",
        ]
    )

    assert rc == 6
    artifact = json.loads(out.read_text())
    assert (
        artifact["llamacpp_trace_oracle"]["denominator_metrics"]["accepted_per_draft_status"]
        == suite.ACCEPTED_DRAFT_NOT_COMPARABLE_DEBUG_TRACE
    )


def test_b1_prompt_suite_cli_fail_on_noncomparable_accepted_draft_allows_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-noncomparable-accepted-draft",
        ]
    )

    assert rc == 0
    matrix = json.loads(out.read_text())
    assert matrix["noncomparable_accepted_per_draft_budgets"] == []


def test_b1_prompt_suite_cli_fail_on_native_runtime_missing_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-native-runtime-missing",
        ]
    )

    assert rc == 7
    artifact = json.loads(out.read_text())
    assert artifact["runtime_kernel_precheck"]["native_runtime_kernels_ready"] is False


def test_b1_prompt_suite_cli_fail_on_native_runtime_missing_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-native-runtime-missing",
        ]
    )

    assert rc == 7
    matrix = json.loads(out.read_text())
    assert matrix["all_native_runtime_kernels_ready"] is False


def test_b1_prompt_suite_native_runtime_missing_helper_allows_ready_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    artifact = suite.build_b1_prompt_suite_artifact(**inputs)
    artifact["runtime_kernel_precheck"] = dict(artifact["runtime_kernel_precheck"])
    artifact["runtime_kernel_precheck"]["native_runtime_kernels_ready"] = True
    matrix = suite.build_b1_b4_prompt_suite_matrix(
        model=inputs["model"],
        prompts_file=inputs["prompts_file"],
        hipengine_token_inventory=inputs["hipengine_token_inventory"],
        llamacpp_token_inventory=inputs["llamacpp_token_inventory"],
        include_artifacts=False,
    )
    matrix["all_native_runtime_kernels_ready"] = True

    assert suite._has_missing_native_runtime_kernels(artifact) is False
    assert suite._has_missing_native_runtime_kernels(matrix) is False


def test_b1_prompt_suite_cli_fail_on_optimization_missing_allows_current_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-optimization-missing",
        ]
    )

    assert rc == 0
    artifact = json.loads(out.read_text())
    assert artifact["runtime_kernel_precheck"]["optimization_kernels_ready"] is True


def test_b1_prompt_suite_cli_fail_on_optimization_missing_rejects_b1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    original_precheck = suite._build_runtime_kernel_precheck

    def missing_optimization_precheck(**kwargs: object) -> dict[str, object]:
        payload = dict(original_precheck(**kwargs))
        checks = [dict(check) for check in payload["checks"]]
        for check in checks:
            if check["name"] == "native_draft_topk_device":
                check["registered"] = False
        payload["checks"] = checks
        payload["optimization_kernels_ready"] = False
        payload["missing_optimization_keys"] = [
            ["hip_gfx1100", "mtp_draft_topk", "w4_gguf", "topk_device"]
        ]
        suite.Qwen35GGUFMTPRuntimeKernelPlan.validate_payload(payload)
        return payload

    monkeypatch.setattr(suite, "_build_runtime_kernel_precheck", missing_optimization_precheck)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--out",
            str(out),
            "--fail-on-optimization-missing",
        ]
    )

    assert rc == 8
    artifact = json.loads(out.read_text())
    assert artifact["runtime_kernel_precheck"]["optimization_kernels_ready"] is False


def test_b1_prompt_suite_cli_fail_on_optimization_missing_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    original_precheck = suite._build_runtime_kernel_precheck

    def missing_optimization_precheck(**kwargs: object) -> dict[str, object]:
        payload = dict(original_precheck(**kwargs))
        checks = [dict(check) for check in payload["checks"]]
        for check in checks:
            if check["name"] == "native_draft_topk_device":
                check["registered"] = False
        payload["checks"] = checks
        payload["optimization_kernels_ready"] = False
        payload["missing_optimization_keys"] = [
            ["hip_gfx1100", "mtp_draft_topk", "w4_gguf", "topk_device"]
        ]
        suite.Qwen35GGUFMTPRuntimeKernelPlan.validate_payload(payload)
        return payload

    monkeypatch.setattr(suite, "_build_runtime_kernel_precheck", missing_optimization_precheck)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-optimization-missing",
        ]
    )

    assert rc == 8
    matrix = json.loads(out.read_text())
    assert matrix["all_optimization_kernels_ready"] is False


def test_b1_prompt_suite_cli_fail_on_performance_unready_rejects_b1_until_runtime_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    trace_payload = json.loads(suite.DEFAULT_LLAMACPP_TRACE_FIXTURE.read_text())
    trace_payload["visible_output_token_count"] = 2
    trace_fixture = _write_json(tmp_path / "llamacpp-trace.json", trace_payload)
    out = tmp_path / "b1-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--hipengine-sampling",
            str(inputs["hipengine_sampling"]),
            "--llamacpp-sampling",
            str(inputs["llamacpp_sampling"]),
            "--llamacpp-trace-fixture",
            str(trace_fixture),
            "--out",
            str(out),
            "--fail-on-performance-unready",
        ]
    )

    assert rc == 5
    artifact = json.loads(out.read_text())
    readiness = suite._matrix_budget_readiness(artifact)
    assert readiness["performance_comparison_blockers"] == [
        "native_runtime_kernels_missing",
        "hipengine_metrics_not_ready",
    ]


def test_b1_prompt_suite_cli_fail_on_performance_unready_rejects_compact_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    inputs = _artifact_inputs(tmp_path)
    out = tmp_path / "matrix-artifact.json"

    rc = suite.main(
        [
            "--model",
            str(inputs["model"]),
            "--prompts-file",
            str(inputs["prompts_file"]),
            "--hipengine-token-inventory",
            str(inputs["hipengine_token_inventory"]),
            "--llamacpp-token-inventory",
            str(inputs["llamacpp_token_inventory"]),
            "--all-budgets",
            "--compact-matrix",
            "--out",
            str(out),
            "--fail-on-performance-unready",
        ]
    )

    assert rc == 5
    matrix = json.loads(out.read_text())
    assert matrix["performance_unready_budgets"] == ["B1", "B2", "B3", "B4"]
    assert matrix["all_performance_comparisons_ready"] is False


def test_b1_prompt_suite_preflight_blocks_requested_budget_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    artifact = suite.build_b1_prompt_suite_artifact(**_artifact_inputs(tmp_path), draft_max=2)

    assert artifact["budget"] == "B2"
    assert artifact["draft_max"] == 2
    assert artifact["draft_budget_precheck"]["passed"] is False
    assert artifact["blockers"] == [
        {
            "code": "draft_budget_mismatch",
            "detail": "requested GGUF MTP draft budget must match sampling settings before metrics are comparable",
            "expected": {"budget": "B2", "draft_max": 2},
            "mismatches": [
                {"engine": "hipengine", "field": "budget", "expected": "B2", "actual": "B1"},
                {"engine": "hipengine", "field": "draft_max", "expected": 2, "actual": 1},
                {"engine": "llamacpp", "field": "budget", "expected": "B2", "actual": "B1"},
                {"engine": "llamacpp", "field": "draft_max", "expected": 2, "actual": 1},
            ],
        }
    ]


def test_b1_prompt_suite_preflight_blocks_stale_draft_topk_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    artifact = suite.build_b1_prompt_suite_artifact(
        **_artifact_inputs(tmp_path, hipengine_draft_top_k=1, llamacpp_draft_top_k=1)
    )

    assert artifact["draft_budget_precheck"]["passed"] is True
    assert artifact["draft_sampling_contract_precheck"]["passed"] is False
    assert artifact["blockers"] == [
        {
            "code": "draft_sampling_contract_mismatch",
            "detail": "sampling fixtures must match the GGUF MTP draft top-k contract before metrics are comparable",
            "expected": {"top_k": 10, "selection": "greedy_top1_from_topk", "selected_index": 0},
            "mismatches": [
                {"engine": "hipengine", "field": "top_k", "expected": 10, "actual": 1},
                {"engine": "llamacpp", "field": "top_k", "expected": 10, "actual": 1},
            ],
        }
    ]


def test_b1_prompt_suite_preflight_rejects_out_of_range_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    with pytest.raises(suite.B1PromptSuitePreflightError, match="draft_max"):
        suite.build_b1_prompt_suite_artifact(**_artifact_inputs(tmp_path), draft_max=5)


def test_b1_prompt_suite_preflight_blocks_hidden_seed_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    draft_topk = {
        "kernel": ["cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h"],
        "top_k": 10,
        "selection": "greedy_top1_from_topk",
        "selected_index": 0,
    }
    call_spec = {
        "layer_id": 40,
        "cpu_reference_kernel": [
            "cpu_reference",
            "mtp_nextn_layer",
            "w4_gguf",
            "qwen35_dense_logits",
        ],
        "draft_topk": draft_topk,
        "dynamic_inputs": [
            {"argument": "hidden_seed", "required": True, "shape": ["tokens", 1024]},
        ],
    }
    monkeypatch.setattr(
        suite,
        "build_qwen35_gguf_mtp_draft_tensor_plans",
        lambda info, *, strict=True: (
            SimpleNamespace(
                layer_id=40,
                as_dict=lambda: {
                    "layer_id": 40,
                    "hidden_size": 2048,
                    "draft_topk": draft_topk,
                    "cpu_reference_call_spec": call_spec,
                },
                cpu_reference_call_spec=SimpleNamespace(as_dict=lambda: call_spec),
            ),
        ),
    )

    artifact = suite.build_b1_prompt_suite_artifact(**_artifact_inputs(tmp_path))

    assert artifact["hidden_seed_contract_precheck"]["passed"] is False
    assert artifact["blockers"] == [
        {
            "code": "hidden_seed_contract_mismatch",
            "detail": "GGUF MTP hidden seed must be fp32 post-output_norm and match the call-spec hidden_seed input",
            "failed_checks": [
                {
                    "name": "hidden_seed_dynamic_input_shape",
                    "passed": False,
                    "detail": "MTP call spec must expose hidden_seed with shape [tokens, hidden_size]",
                }
            ],
        }
    ]


def test_b1_prompt_suite_preflight_reports_llamacpp_trace_blocker_before_runtime_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    trace = _write_json(
        tmp_path / "bad-trace.json",
        {
            "schema": 1,
            "kind": "llamacpp_mtp_draft_candidate_trace",
            "calls": [],
            "summary": {"candidate_count": 0, "draft_call_count": 0, "observed_top_k": 0},
            "metadata": {"server_command": []},
        },
    )

    artifact = suite.build_b1_prompt_suite_artifact(
        **_artifact_inputs(tmp_path),
        llamacpp_trace_fixture=trace,
    )

    assert artifact["status"] == "blocked"
    assert artifact["llamacpp_trace_oracle"]["passed"] is False
    assert artifact["execution"]["exactness_gate"] == "failed"
    assert artifact["blockers"][0]["code"] == "llamacpp_trace_oracle_failed"
    assert {item["name"] for item in artifact["blockers"][0]["failed_checks"]} >= {
        "calls_present",
        "observed_top_k",
        "debug_trace_not_benchmark",
    }


def test_b1_prompt_suite_preflight_reports_llamacpp_trace_denominator_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    trace = _write_json(
        tmp_path / "bad-denominator-trace.json",
        {
            "schema": 1,
            "kind": "llamacpp_mtp_draft_candidate_trace",
            "prompt_tokens": 3,
            "calls": [
                {
                    "generated": 1,
                    "accepted": 0,
                    "candidates": [{"rank": 0, "token_id": 42}],
                }
            ],
            "summary": {
                "candidate_count": 1,
                "draft_call_count": 1,
                "observed_top_k": 1,
                "draft_n": 2,
                "draft_n_accepted": 0,
                "draft_acceptance": 0.0,
            },
            "metadata": {"server_command": ["llama-server", "--no-spec-draft-backend-sampling"]},
        },
    )

    artifact = suite.build_b1_prompt_suite_artifact(
        **_artifact_inputs(tmp_path),
        llamacpp_trace_fixture=trace,
    )

    assert artifact["llamacpp_trace_oracle"]["passed"] is False
    assert artifact["llamacpp_trace_oracle"]["denominator_metrics"]["generated_draft_tokens"] == 1
    assert artifact["blockers"][0]["code"] == "llamacpp_trace_oracle_failed"
    assert {item["name"] for item in artifact["blockers"][0]["failed_checks"]} == {"draft_n"}


def test_b1_prompt_suite_preflight_reports_oracle_gate_blocker_before_runtime_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)
    monkeypatch.setattr(
        suite,
        "run_oracle_gate",
        lambda fixture: {
            "passed": False,
            "fixture": str(fixture),
            "metrics": {"max_kl": 0.25, "top1_agreement": 0.0},
            "kvlivespans_paged_cache_smoke": {
                "passed": True,
                "max_abs_diff": 0.0,
            },
        },
    )

    artifact = suite.build_b1_prompt_suite_artifact(**_artifact_inputs(tmp_path))

    assert artifact["status"] == "blocked"
    assert artifact["parity_precheck"]["all_pass"] is True
    assert artifact["oracle_gate"]["passed"] is False
    assert artifact["execution"]["exactness_gate"] == "failed"
    assert artifact["blockers"] == [
        {
            "code": "oracle_gate_failed",
            "detail": "CPU-reference GGUF MTP oracle KL/top-1 gate must pass before B1 metrics are comparable",
            "max_kl": 0.25,
            "top1_agreement": 0.0,
        }
    ]


def test_b1_prompt_suite_preflight_reports_parity_blocker_before_runtime_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_model(monkeypatch)

    artifact = suite.build_b1_prompt_suite_artifact(**_artifact_inputs(tmp_path, mismatch=True))

    assert artifact["status"] == "blocked"
    assert artifact["parity_precheck"]["all_pass"] is False
    assert artifact["blockers"] == [
        {
            "code": "parity_precheck_failed",
            "detail": "token-id and sampling parity must pass before B1 accepted/output metrics are comparable",
            "token_match": False,
            "sampling_match": True,
        }
    ]


def test_b1_prompt_suite_cli_emits_blocked_artifact_for_real_fixtures(tmp_path: Path) -> None:
    model = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    out = tmp_path / "artifact.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_mtp_b1_prompt_suite.py",
            "--model",
            str(model),
            "--prompt-limit",
            "1",
            "--out",
            str(out),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(out.read_text())
    assert artifact["status"] == "blocked"
    assert artifact["prompt_count"] == 1
    assert artifact["parity_precheck"]["all_pass"] is True
    assert artifact["blockers"][0]["code"] == "native_gguf_mtp_runtime_missing"


def test_b1_prompt_suite_cli_fail_on_blocked_returns_two(tmp_path: Path) -> None:
    model = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    if not model.exists():
        pytest.skip(f"local GGUF fixture not found: {model}")
    out = tmp_path / "artifact.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_mtp_b1_prompt_suite.py",
            "--model",
            str(model),
            "--prompt-limit",
            "1",
            "--out",
            str(out),
            "--fail-on-blocked",
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 2
    assert json.loads(out.read_text())["blockers"][0]["code"] == "native_gguf_mtp_runtime_missing"
