from __future__ import annotations

import json
from pathlib import Path

from scripts.gguf_hidden_precision_decision_audit import (
    audit_doc_hidden_seed_contract,
    audit_runner_hidden_precision,
    build_hidden_precision_decision_artifact,
    decide_precision_path,
)


def test_runner_audit_detects_bf16_activation_lane() -> None:
    audit = audit_runner_hidden_precision(_runner_source())
    facts = audit["facts"]

    assert facts["current_hidden_seed_contract_dtype"] == "BF16"
    assert facts["fp32_hidden_seed_contract_dtype"] == "FP32"
    assert facts["fp32_hidden_seed_contract_present"] is True
    assert facts["fp32_hidden_seed_populated_guard"] is True
    assert facts["current_hidden_seed_llama_compatible"] is False
    assert facts["current_hidden_seed_marks_llama_incompatible"] is True
    assert facts["session_hidden_buffer_nbytes_expression"] == "self.runner.hidden_size * 2"
    assert facts["session_hidden_buffers_allocated"] is True
    assert facts["prefill_hidden_buffers_allocated"] is True
    assert facts["run_prompt_hidden_returns_uint16_bits"] is True
    assert facts["capture_hidden_in_uses_bf16_copy"] is True
    assert facts["current_runtime_activation_lane"] == "bf16"


def test_doc_audit_detects_fp32_seed_contract() -> None:
    text = "Target hidden-row seed = POST output-norm hidden, at fp32\nuses GGML_TYPE_F32"

    audit = audit_doc_hidden_seed_contract(text)

    assert audit["requires_fp32_seed"] is True
    assert "GGML_TYPE_F32" in audit["matched_terms"]


def test_decide_requires_f32_lane_when_numeric_and_docs_agree() -> None:
    decision = decide_precision_path(
        runner_facts={"current_runtime_activation_lane": "bf16"},
        doc_contract={"requires_fp32_seed": True},
        numeric={
            "llamacpp_matches_raw_dequant": True,
            "hipengine_matches_bf16_round": True,
            "earliest_first_mismatch_layer": 0,
            "earliest_layer0_preceding_precision_contractions": 0,
        },
    )

    assert decision["status"] == "decided"
    assert decision["conclusion"] == (
        "exact_parity_requires_explicit_f32_activation_or_seed_lane"
    )
    assert decision["keep_default_runtime"] == "bf16_activation_buffers"


def test_decide_prefers_existing_fp32_seed_target_when_present() -> None:
    decision = decide_precision_path(
        runner_facts={
            "current_runtime_activation_lane": "bf16",
            "fp32_hidden_seed_contract_dtype": "FP32",
            "fp32_hidden_seed_contract_present": True,
            "fp32_hidden_seed_populated_guard": True,
        },
        doc_contract={"requires_fp32_seed": True},
        numeric={
            "llamacpp_matches_raw_dequant": True,
            "hipengine_matches_bf16_round": True,
            "earliest_first_mismatch_layer": 0,
            "earliest_layer0_preceding_precision_contractions": 0,
        },
    )

    assert decision["status"] == "decided"
    assert decision["conclusion"] == (
        "fp32_seed_target_exists_but_activation_lane_is_bf16"
    )
    assert decision["next_action"] == (
        "capture_fp32_hidden_seed_vs_llamacpp_post_output_norm_oracle"
    )


def test_decide_blocks_when_numeric_proof_is_missing() -> None:
    decision = decide_precision_path(
        runner_facts={"current_runtime_activation_lane": "bf16"},
        doc_contract={"requires_fp32_seed": True},
        numeric={
            "llamacpp_matches_raw_dequant": False,
            "hipengine_matches_bf16_round": True,
            "earliest_first_mismatch_layer": 0,
            "earliest_layer0_preceding_precision_contractions": 0,
        },
    )

    assert decision["status"] == "blocked"
    assert decision["next_action"] == "rerun_token_embedding_and_source_audits"


def test_build_artifact_from_synthetic_inputs(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    doc = tmp_path / "MTP-gguf.md"
    token = tmp_path / "token.json"
    earliest = tmp_path / "earliest.json"
    runner.write_text(_runner_source())
    doc.write_text("Target hidden-row seed = POST output-norm hidden, at fp32\nGGML_TYPE_F32")
    token.write_text(
        json.dumps(
            {
                "status": "explained",
                "conclusion": "layer0_drift_is_bf16_embedding_output",
                "comparisons": {
                    "llamacpp_vs_raw_dequant": {"exact_match": True},
                    "hipengine_vs_bf16_round": {"exact_match": True},
                    "llamacpp_vs_hipengine": {"rmse": 1.0e-6, "max_abs_diff": 2.0e-5},
                },
            }
        )
    )
    earliest.write_text(
        json.dumps(
            {
                "ranking": {"first_mismatch_layer": 0},
                "layer_results": [
                    {
                        "layer": 0,
                        "numeric_delta": {"rmse": 1.0e-6, "max_abs_diff": 2.0e-5},
                        "preceding_precision_contractions": {"count": 0},
                    }
                ],
            }
        )
    )

    artifact = build_hidden_precision_decision_artifact(
        runner_path=runner,
        doc_path=doc,
        token_audit_path=token,
        earliest_path=earliest,
    )

    assert artifact["status"] == "decided"
    assert artifact["conclusion"] == (
        "fp32_seed_target_exists_but_activation_lane_is_bf16"
    )
    assert artifact["numeric_evidence"]["llamacpp_matches_raw_dequant"] is True
    assert artifact["runner_abi"]["facts"]["current_runtime_activation_lane"] == "bf16"


def _runner_source() -> str:
    return '''def qwen35_gguf_fp32_hidden_seed_contract():
    return Qwen35GGUFHiddenSeedContract(
        dtype=DType.FP32,
        source_buffer="Qwen35GGUFResidentSession.scratch.hidden_seed_fp32",
    )

def qwen35_gguf_current_hidden_seed_contract():
    return Qwen35GGUFHiddenSeedContract(
        dtype=DType.BF16,
        llama_cpp_compatible=False,
    )

class Qwen35GGUFResidentSession:
    def __enter__(self):
        hidden_bytes = self.runner.hidden_size * 2
        self._hidden_a = malloc(hidden_bytes, runtime=runtime)
        self._hidden_b = malloc(hidden_bytes, runtime=runtime)
        self._prefill_hidden_a = malloc(prefill_capacity * hidden_bytes, runtime=runtime)
        self._prefill_hidden_b = malloc(prefill_capacity * hidden_bytes, runtime=runtime)

    def run_prompt_hidden(self, token_ids):
        hidden_bits = np.empty((1, self.hidden_size), dtype=np.uint16)
        return hidden_bits

    def _run_token_to_final_hidden(self, token_id):
        launch_gguf_embedding(
            self.runner.weights.root("token_embedding"),
            token,
            self._hidden_a.ptr,
        )

    def fp32_hidden_seed_contract(self):
        return Contract(ready_for_mtp=True)

    def fp32_hidden_seed_ptr(self):
        if not self.fp32_hidden_seed_contract().ready_for_mtp:
            raise RuntimeError("call step(..., capture_hidden_seed_fp32=True) first")

    def _run_output_norm_hidden(self):
        gguf_rmsnorm_bf16_f32_weight_out_f32()

    def capture_attention_layer(self):
        hidden_in_f32=_copy_bf16_ptr_to_host_f32(target_src_ptr, hidden_size)
'''
