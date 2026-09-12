from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_layer_boundary_bisect_plan import (
    LAYER_CAPTURE_LABEL,
    audit_hipengine_layer_capture,
    audit_llamacpp_layer_patch,
    binary_probe_order,
    build_layer_boundary_bisect_plan,
    decide_plan,
    infer_model_layer_count,
)


def test_llamacpp_layer_patch_audit_finds_unique_l_out_anchor() -> None:
    audit = audit_llamacpp_layer_patch(_qwen_source(), _context_source())

    assert audit["ready"] is True
    assert audit["layer_anchor_count"] == 1
    assert audit["post_output_preserve_state"] == "needs_patch"
    assert audit["final_pre_output_patch_count"] == 0
    assert audit["final_pre_output_patch_must_be_absent"] is True
    assert LAYER_CAPTURE_LABEL in audit["layer_capture_new_text_template"]
    assert "res->t_h_nextn = cur" in audit["layer_capture_new_text_template"]


def test_llamacpp_layer_patch_allows_already_preserved_post_output() -> None:
    source = _qwen_source().replace(
        '    cb(cur, "h_nextn", -1);\n'
        "    res->t_h_nextn = cur;\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {",
        '    cb(cur, "h_nextn_post_output_norm", -1);\n'
        "    // PRE-output_norm diagnostic: keep res->t_h_nextn from before output_norm.\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {",
    )

    audit = audit_llamacpp_layer_patch(source, _context_source())

    assert audit["ready"] is True
    assert audit["post_output_preserve_state"] == "already_patched"
    assert audit["post_output_old_count"] == 0
    assert audit["post_output_new_count"] == 1


def test_llamacpp_layer_patch_records_final_pre_output_patch_as_disallowed() -> None:
    source = _qwen_source().replace(
        '    }\n    cb(cur, "h_nextn", -1);',
        '    }\n    cb(cur, "h_nextn_pre_output_norm", -1);\n    cb(cur, "h_nextn", -1);',
    )

    audit = audit_llamacpp_layer_patch(source, _context_source())

    assert audit["ready"] is True
    assert audit["final_pre_output_patch_count"] == 1
    assert "would overwrite the selected layer tap" in audit["notes"][0]


def test_hipengine_layer_capture_detects_existing_helper() -> None:
    audit = audit_hipengine_layer_capture(_runner_source())
    facts = audit["facts"]

    assert audit["ready"] is True
    assert facts["supports_run_preceding_layers"] is True
    assert facts["captures_layer_out_f32"] is True
    assert facts["supports_linear_attention"] is True
    assert facts["supports_full_attention"] is True
    assert audit["value_field"] == "layer_out_f32"


def test_binary_probe_order_is_breadth_first_midpoints() -> None:
    assert binary_probe_order(0, 6) == [3, 1, 5, 0, 2, 4, 6]
    assert binary_probe_order(0, -1) == []


def test_decide_plan_ready_when_all_facts_present() -> None:
    decision = decide_plan(
        prior={"ready": True},
        llama_patch={"ready": True},
        hip_capture={"ready": True},
        layer_count=40,
    )

    assert decision["status"] == "ready"
    assert decision["conclusion"] == "layer_boundary_bisect_plan_ready"
    assert decision["next_action"] == "build_layer_boundary_capture_for_layer_39_and_compare"


def test_build_layer_boundary_bisect_plan_from_synthetic_inputs(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen35moe.cpp"
    context = tmp_path / "llama-context.cpp"
    runner = tmp_path / "runner.py"
    compare = tmp_path / "compare.json"
    qwen.write_text(_qwen_source())
    context.write_text(_context_source())
    runner.write_text(_runner_source())
    compare.write_text(json.dumps(_compare_artifact()))

    artifact = build_layer_boundary_bisect_plan(
        qwen35moe_path=qwen,
        context_path=context,
        runner_path=runner,
        compare_artifact_path=compare,
    )

    assert artifact["status"] == "ready"
    assert artifact["selected_layer_count"] == 40
    assert artifact["conclusion"] == "layer_boundary_bisect_plan_ready"
    assert artifact["llamacpp_layer_patch"]["ready"] is True
    assert artifact["hipengine_layer_capture"]["ready"] is True
    assert artifact["prior_pre_output_result"]["ready"] is True
    assert artifact["bisection_targets"]["first_probe_layer"] == 39
    assert artifact["bisection_targets"]["recommended_first_batch"] == [39, 19, 9, 29]
    assert artifact["execution_plan"]["first_probe_effective_tap"] == LAYER_CAPTURE_LABEL
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def test_infer_model_layer_count_uses_model_name_before_source_cases() -> None:
    assert infer_model_layer_count("case 48: break;", {"model": "Qwen-35B.gguf"}) == 40
    assert infer_model_layer_count("case 48: break;", {"model": "unknown.gguf"}) == 48


def _qwen_source() -> str:
    return '''void graph() {
    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_35B_A3B; break;
    }
    for (int il = 0; il < n_layer; ++il) {
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
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
    def step(self):
        pass

    def capture_attention_layer(self, run_preceding_layers=False):
        self._set_token_id_device(token)
        if run_preceding_layers:
            self.runner._run_linear_attention_layer(layer, src, dst, scratch)
            self.runner._run_full_attention_layer(layer, src, dst, scratch)
        return Capture(
            hidden_in_f32=hidden,
            layer_out_f32=out,
        )
'''


def _compare_artifact() -> dict[str, object]:
    return {
        "status": "mismatched",
        "classification": "pre_output_mismatch_already_present",
        "model": "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "prompt_tokens": [1, 2, 3],
        "position": 2,
        "token_id": 3,
        "next_action": "bisect_final_decoder_layer_output_before_output_norm",
        "numeric_delta": {
            "rmse": 0.45158678625003046,
            "max_abs_diff": 6.707832336425781,
            "mean_abs_diff": 0.3372998724516947,
            "llamacpp_sha256": "llama-pre",
            "hipengine_sha256": "hip-pre",
        },
    }
