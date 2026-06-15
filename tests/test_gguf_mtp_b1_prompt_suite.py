from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gguf_mtp_b1_prompt_suite as suite


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
    assert artifact["runtime_kernel_precheck"]["backend"] == "hip_gfx1100"
    assert artifact["runtime_kernel_precheck"]["exactness_oracles_ready"] is True
    assert artifact["runtime_kernel_precheck"]["native_runtime_kernels_ready"] is False
    assert artifact["runtime_kernel_precheck"]["optimization_kernels_ready"] is False
    assert artifact["runtime_kernel_precheck"]["missing_exactness_oracle_keys"] == []
    assert artifact["runtime_kernel_precheck"]["missing_native_runtime_keys"] == [
        ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"]
    ]
    assert artifact["runtime_kernel_precheck"]["missing_optimization_keys"] == [
        ["hip_gfx1100", "mtp_draft_topk", "w4_gguf", "topk_device"]
    ]
    assert artifact["oracle_gate"]["passed"] is True
    assert artifact["llamacpp_trace_oracle"]["passed"] is True
    assert artifact["llamacpp_trace_oracle"]["selected_token_ids"] == [8068, 271]
    assert artifact["llamacpp_trace_oracle"]["observed_top_k"] == 3
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
                ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"]
            ],
            "missing_optimization_keys": [
                ["hip_gfx1100", "mtp_draft_topk", "w4_gguf", "topk_device"]
            ],
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
                ["hip_gfx1100", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits"]
            ],
            "missing_optimization_keys": [
                ["hip_gfx1100", "mtp_draft_topk", "w4_gguf", "topk_device"]
            ],
        }
    ]


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
