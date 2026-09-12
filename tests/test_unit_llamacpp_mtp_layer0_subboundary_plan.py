from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_layer0_subboundary_plan import (
    ATTN_NORM_CAPTURE_LABEL,
    audit_hipengine_linear_boundary_capture,
    audit_llamacpp_attn_norm_patch,
    build_layer0_subboundary_plan,
    decide_plan,
)


def test_llamacpp_attn_norm_patch_audit_finds_unique_anchor() -> None:
    audit = audit_llamacpp_attn_norm_patch(_qwen_source(), _context_source())

    assert audit["ready"] is True
    assert audit["attn_norm_anchor_count"] == 1
    assert audit["capture_label"] == ATTN_NORM_CAPTURE_LABEL
    assert audit["post_output_preserve_state"] == "needs_patch"
    assert audit["input_capture_patch_count"] == 0
    assert audit["layer_capture_patch_count"] == 0
    assert audit["final_pre_output_patch_count"] == 0
    assert "res->t_h_nextn = cur" in audit["attn_norm_capture_new_text"]


def test_llamacpp_attn_norm_patch_allows_already_preserved_post_output() -> None:
    source = _qwen_source().replace(
        '    cb(cur, "h_nextn", -1);\n'
        "    res->t_h_nextn = cur;\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {",
        '    cb(cur, "h_nextn_post_output_norm", -1);\n'
        "    // PRE-output_norm diagnostic: keep res->t_h_nextn from before output_norm.\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {",
    )

    audit = audit_llamacpp_attn_norm_patch(source, _context_source())

    assert audit["ready"] is True
    assert audit["post_output_preserve_state"] == "already_patched"
    assert audit["post_output_old_count"] == 0
    assert audit["post_output_new_count"] == 1


def test_llamacpp_attn_norm_patch_records_existing_diagnostic_taps() -> None:
    source = _qwen_source().replace(
        '        cb(cur, "attn_norm", il);',
        '        cb(cur, "attn_norm", il);\n        cb(cur, "h_nextn_input_embed", il);',
    ).replace(
        '    cb(cur, "h_nextn", -1);',
        '    cb(cur, "h_nextn_layer_out", -1);\n'
        '    cb(cur, "h_nextn_pre_output_norm", -1);\n'
        '    cb(cur, "h_nextn", -1);',
    )

    audit = audit_llamacpp_attn_norm_patch(source, _context_source())

    assert audit["ready"] is False
    assert audit["attn_norm_anchor_count"] == 0
    assert audit["input_capture_patch_count"] == 1
    assert audit["layer_capture_patch_count"] == 1
    assert audit["final_pre_output_patch_count"] == 1


def test_hipengine_linear_boundary_audit_detects_attn_norm_capture() -> None:
    audit = audit_hipengine_linear_boundary_capture(_runner_source())
    facts = audit["facts"]

    assert audit["ready"] is True
    assert facts["requires_linear_attention_layer"] is True
    assert facts["runs_attn_only"] is True
    assert facts["captures_attn_norm_f32"] is True
    assert facts["attn_norm_copied_from_bf16_norm_scratch"] is True
    assert audit["first_value_field"] == "attn_norm_f32"
    assert "BF16-rounded" in audit["dtype_note"]


def test_decide_plan_ready_when_all_facts_present() -> None:
    decision = decide_plan(
        prior_input={"ready": True},
        prior_layer0={"ready": True},
        llama_patch={"ready": True},
        hip_capture={"ready": True},
    )

    assert decision["status"] == "ready"
    assert decision["conclusion"] == "layer0_attn_norm_capture_plan_ready"
    assert decision["next_action"] == "build_layer0_attn_norm_capture_and_compare"


def test_build_layer0_subboundary_plan_from_synthetic_inputs(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen35moe.cpp"
    context = tmp_path / "llama-context.cpp"
    runner = tmp_path / "runner.py"
    input_artifact = tmp_path / "input.json"
    layer0_artifact = tmp_path / "layer0.json"
    qwen.write_text(_qwen_source())
    context.write_text(_context_source())
    runner.write_text(_runner_source())
    input_artifact.write_text(json.dumps(_input_artifact()))
    layer0_artifact.write_text(json.dumps(_layer0_artifact()))

    artifact = build_layer0_subboundary_plan(
        qwen35moe_path=qwen,
        context_path=context,
        runner_path=runner,
        input_artifact_path=input_artifact,
        layer0_artifact_path=layer0_artifact,
    )

    assert artifact["status"] == "ready"
    assert artifact["conclusion"] == "layer0_attn_norm_capture_plan_ready"
    assert artifact["llamacpp_attn_norm_patch"]["ready"] is True
    assert artifact["hipengine_linear_boundary_capture"]["ready"] is True
    assert artifact["prior_input_embed_result"]["ready"] is True
    assert artifact["prior_layer0_result"]["ready"] is True
    assert artifact["comparison_plan"]["llamacpp_effective_tap"] == ATTN_NORM_CAPTURE_LABEL
    assert artifact["comparison_plan"]["hipengine_value_field"] == "attn_norm_f32"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def _qwen_source() -> str:
    return '''void graph() {
    for (int il = 0; il < n_layer; ++il) {
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);
        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        }
    }
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
}
'''


def _context_source() -> str:
    return '''// unmasked: nextn rows are stored densely, indexed by raw token position.
return embd_nextn.data + (size_t) i * n_embd;
'''


def _runner_source() -> str:
    return '''class Session:
    def capture_linear_attention_boundary(self):
        if layer_types[layer_id] != LINEAR_ATTENTION:
            raise ValueError(f"layer {layer_id} is not a linear_attention layer")
        self._set_token_id_device(token)
        self.runner._run_linear_attention_attn_only(layer, src, dst, scratch)
        return Capture(
            attn_norm_f32=_copy_bf16_ptr_to_host_f32(
                int(self.scratch.norm.ptr), hidden_size, runtime=runtime
            ),
            linear_qkv_f32=linear_qkv,
            linear_z_f32=linear_z,
            attn_out_f32=attn_out,
        )
'''


def _input_artifact() -> dict[str, object]:
    return {
        "status": "mismatched",
        "classification": "input_embed_matches_after_bf16_roundtrip",
        "numeric_delta": {
            "rmse": 4.3e-6,
            "max_abs_diff": 8.2e-5,
            "llamacpp_sha256": "llama-input",
            "hipengine_sha256": "hip-hidden-in",
        },
        "bf16_rounded_delta": {
            "exact_match": True,
            "rmse": 0.0,
            "max_abs_diff": 0.0,
            "llamacpp_sha256": "llama-bf16",
        },
    }


def _layer0_artifact() -> dict[str, object]:
    return {
        "status": "mismatched",
        "classification": "layer_boundary_mismatch",
        "layer_id": 0,
        "next_action": "inspect_initial_embedding_or_token_capture",
        "numeric_delta": {
            "rmse": 0.012114962719327636,
            "max_abs_diff": 0.045740051893517375,
            "mean_abs_diff": 0.009691874419019086,
        },
    }
