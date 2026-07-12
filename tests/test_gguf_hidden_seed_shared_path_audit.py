from __future__ import annotations

import json
from pathlib import Path

from scripts.gguf_hidden_seed_shared_path_audit import (
    audit_hipengine_shared_seed_path,
    audit_llamacpp_h_nextn_path,
    build_shared_path_audit,
    decide_shared_path,
    summarize_mode_evidence,
)


def test_summarize_mode_evidence_extracts_serial_agreement_and_bulk_delta() -> None:
    summary = summarize_mode_evidence(_mode_sweep())

    assert summary["native_serial_step_exact"] is True
    assert summary["native_serial_step_sha256"] == "serial-sha"
    assert summary["bulk_differs_from_native"] is True
    assert summary["native_vs_llamacpp_rmse"] == 3.1
    assert summary["bulk_vs_native_rmse"] == 0.03


def test_hipengine_source_audit_detects_shared_bf16_output_norm_path() -> None:
    audit = audit_hipengine_shared_seed_path(_runner_source())
    facts = audit["facts"]

    assert audit["ready"] is True
    assert facts["prefill_serial_captures_only_final_token"] is True
    assert facts["step_serial_calls_token_to_final_hidden"] is True
    assert facts["serial_path_loops_all_layers_before_output_norm"] is True
    assert facts["bulk_native_and_bulk_share_output_norm_call"] is True
    assert facts["hidden_seed_f32_recomputed_from_same_bf16_source"] is True


def test_llamacpp_source_audit_detects_post_norm_h_nextn_row_semantics() -> None:
    audit = audit_llamacpp_h_nextn_path(_qwen_source(), _context_source())

    assert audit["ready"] is True
    assert audit["facts"]["qwen_trunk_h_nextn_after_output_norm"] is True
    assert audit["facts"]["context_unmasked_rows_by_raw_position"] is True
    assert audit["facts"]["decode_copies_unmasked_rows_with_token_offset"] is True


def test_decide_shared_path_points_to_pre_output_norm_bisect() -> None:
    decision = decide_shared_path(
        mode_evidence={
            "native_serial_step_exact": True,
            "native_vs_llamacpp_rmse": 3.1,
            "bulk_vs_native_rmse": 0.03,
        },
        hipengine_path={"ready": True},
        llamacpp_path={"ready": True},
    )

    assert decision["status"] == "audited"
    assert decision["conclusion"] == "shared_serial_path_mismatch_before_or_at_output_norm"
    assert decision["secondary_issue"] == "prefill_bulk_differs_from_serial"
    assert decision["next_action"] == (
        "capture_pre_output_norm_rows_in_llamacpp_and_hipengine_serial_path"
    )


def test_build_shared_path_audit_from_synthetic_inputs(tmp_path: Path) -> None:
    runner = tmp_path / "runner.py"
    qwen = tmp_path / "qwen35moe.cpp"
    context = tmp_path / "llama-context.cpp"
    sweep = tmp_path / "sweep.json"
    runner.write_text(_runner_source())
    qwen.write_text(_qwen_source())
    context.write_text(_context_source())
    sweep.write_text(json.dumps(_mode_sweep()))

    artifact = build_shared_path_audit(
        runner_path=runner,
        qwen35moe_path=qwen,
        context_path=context,
        mode_sweep_path=sweep,
    )

    assert artifact["status"] == "audited"
    assert artifact["conclusion"] == "shared_serial_path_mismatch_before_or_at_output_norm"
    assert artifact["mode_evidence"]["native_serial_step_exact"] is True
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def _mode_sweep() -> dict[str, object]:
    def row(mode: str, sha: str, rmse: float) -> dict[str, object]:
        return {
            "mode": mode,
            "hipengine_capture": {"summary": {"sha256": sha}},
            "numeric_delta": {"rmse": rmse, "max_abs_diff": rmse + 1.0},
        }

    return {
        "status": "mismatched",
        "conclusion": "hipengine_seed_modes_diverge_and_mismatch_llamacpp",
        "mode_results": [
            row("prefill-bulk", "bulk-sha", 3.2),
            row("prefill-native", "serial-sha", 3.1),
            row("prefill-serial", "serial-sha", 3.1),
            row("step-serial", "serial-sha", 3.1),
        ],
        "hipengine_pairwise": {
            "pairs": [
                {
                    "left": "prefill-bulk",
                    "right": "prefill-native",
                    "rmse": 0.03,
                    "max_abs_diff": 0.1,
                    "exact_match": False,
                },
                {
                    "left": "prefill-native",
                    "right": "prefill-serial",
                    "rmse": 0.0,
                    "max_abs_diff": 0.0,
                    "exact_match": True,
                },
            ]
        },
    }


def _runner_source() -> str:
    return '''class Session:
    def prefill(self, token_ids, *, capture_hidden_seed_fp32=False):
        final_index = len(token_ids) - 1
        for index, token_id in enumerate(token_ids):
            self._run_token_to_final_hidden(
                token_id,
                capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32) and index == final_index,
            )

    def _run_bulk_prefill_and_sample(self, *, bulk_attention_mode="bulk"):
        if bulk_attention_mode == "native":
            pass
        last_src_ptr = src.ptr
        self._run_output_norm_hidden(last_src_ptr, scratch.norm.ptr)

    def step(self, token_id):
        return self._run_token_to_final_hidden(token_id)

    def _run_token_to_final_hidden(self, token_id):
        self._set_full_attention_position_device(0)
        self._set_token_id_device(token_id)
        return self._run_current_hidden_to_final_hidden()

    def _run_current_hidden_to_final_hidden(self):
        for layer_id, layer_type in enumerate(layers):
            pass
        return self._run_output_norm_hidden(src.ptr, scratch.norm.ptr)

    def _run_output_norm_hidden(self, src_ptr, out_ptr):
        gguf_rmsnorm_bf16_f32_weight(src_ptr, weight, out_ptr)
        gguf_rmsnorm_bf16_f32_weight_out_f32(
            src_ptr,
            weight,
            self.scratch.hidden_seed_fp32.ptr,
        )
        return int(out_ptr)
'''


def _qwen_source() -> str:
    return '''// post-norm hidden state feeds both the LM head and the MTP seed below
cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
cb(cur, "h_nextn", -1);
res->t_h_nextn = cur;
// LM head
'''


def _context_source() -> str:
    return '''// unmasked: nextn rows are stored densely, indexed by raw token position.
return embd_nextn.data + (size_t) i * n_embd;
const bool masked    = cparams.embeddings_nextn_masked;
const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;
ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, bytes);
embd_nextn.size = (size_t) n_embd_out * n_batch;
'''
