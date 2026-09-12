from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_initial_input_capture_plan import (
    INPUT_CAPTURE_LABEL,
    audit_hipengine_hidden_in_capture,
    audit_llamacpp_input_patch,
    build_initial_input_capture_plan,
    decide_plan,
)


def test_llamacpp_input_patch_audit_finds_unique_input_anchor() -> None:
    audit = audit_llamacpp_input_patch(_qwen_source(), _context_source())

    assert audit["ready"] is True
    assert audit["input_anchor_count"] == 1
    assert audit["capture_label"] == INPUT_CAPTURE_LABEL
    assert audit["post_output_preserve_state"] == "needs_patch"
    assert audit["layer_capture_patch_count"] == 0
    assert audit["final_pre_output_patch_count"] == 0
    assert audit["layer_capture_patch_must_be_absent"] is True
    assert audit["final_pre_output_patch_must_be_absent"] is True
    assert "res->t_h_nextn = inpL" in audit["input_capture_new_text"]


def test_llamacpp_input_patch_allows_already_preserved_post_output() -> None:
    source = _qwen_source().replace(
        '    cb(cur, "h_nextn", -1);\n'
        "    res->t_h_nextn = cur;\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {",
        '    cb(cur, "h_nextn_post_output_norm", -1);\n'
        "    // PRE-output_norm diagnostic: keep res->t_h_nextn from before output_norm.\n\n"
        "    if (!cparams.embeddings_nextn_masked && inp_out_ids) {",
    )

    audit = audit_llamacpp_input_patch(source, _context_source())

    assert audit["ready"] is True
    assert audit["post_output_preserve_state"] == "already_patched"
    assert audit["post_output_old_count"] == 0
    assert audit["post_output_new_count"] == 1


def test_llamacpp_input_patch_records_disallowed_existing_capture_patches() -> None:
    source = _qwen_source().replace(
        '    cb(inpL, "model.input_embed", -1);',
        '    cb(inpL, "model.input_embed", -1);\n    cb(cur, "h_nextn_layer_out", 0);',
    ).replace(
        '    cb(cur, "h_nextn", -1);',
        '    cb(cur, "h_nextn_pre_output_norm", -1);\n    cb(cur, "h_nextn", -1);',
    )

    audit = audit_llamacpp_input_patch(source, _context_source())

    assert audit["ready"] is False
    assert audit["input_anchor_count"] == 0
    assert audit["layer_capture_patch_count"] == 1
    assert audit["final_pre_output_patch_count"] == 1
    assert "cannot be overwritten" in audit["notes"][0]


def test_hipengine_hidden_in_capture_audit_detects_existing_path() -> None:
    audit = audit_hipengine_hidden_in_capture(_runner_source(), _embedding_source())
    facts = audit["facts"]

    assert audit["ready"] is True
    assert facts["captures_hidden_in_f32"] is True
    assert facts["copies_hidden_in_from_target_src_ptr"] is True
    assert facts["sets_token_id_device"] is True
    assert facts["embedding_output_bf16"] is True
    assert audit["value_field"] == "hidden_in_f32"
    assert "BF16-rounded" in audit["dtype_note"]


def test_decide_plan_ready_when_all_facts_present() -> None:
    decision = decide_plan(
        prior={"ready": True},
        llama_patch={"ready": True},
        hip_capture={"ready": True},
    )

    assert decision["status"] == "ready"
    assert decision["conclusion"] == "initial_input_capture_plan_ready"
    assert decision["next_action"] == "build_input_embed_capture_and_compare_hidden_in_f32"


def test_build_initial_input_capture_plan_from_synthetic_inputs(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen35moe.cpp"
    context = tmp_path / "llama-context.cpp"
    runner = tmp_path / "runner.py"
    embedding = tmp_path / "gguf_embedding.py"
    layer0 = tmp_path / "layer0.json"
    qwen.write_text(_qwen_source())
    context.write_text(_context_source())
    runner.write_text(_runner_source())
    embedding.write_text(_embedding_source())
    layer0.write_text(json.dumps(_layer0_artifact()))

    artifact = build_initial_input_capture_plan(
        qwen35moe_path=qwen,
        context_path=context,
        runner_path=runner,
        embedding_path=embedding,
        layer0_artifact_path=layer0,
    )

    assert artifact["status"] == "ready"
    assert artifact["conclusion"] == "initial_input_capture_plan_ready"
    assert artifact["llamacpp_input_patch"]["ready"] is True
    assert artifact["hipengine_hidden_in_capture"]["ready"] is True
    assert artifact["prior_layer0_result"]["ready"] is True
    assert artifact["comparison_plan"]["llamacpp_effective_tap"] == INPUT_CAPTURE_LABEL
    assert artifact["comparison_plan"]["hipengine_value_field"] == "hidden_in_f32"
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def _qwen_source() -> str:
    return '''void graph() {
    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();
    for (int il = 0; il < n_layer; ++il) {
        cb(cur, "l_out", il);
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
    def capture_attention_layer(self):
        self._set_token_id_device(token)
        src = self._hidden_a
        target_src_ptr = int(src.ptr)
        run_layer(target_src_ptr)
        return Capture(
            hidden_in_f32=_copy_bf16_ptr_to_host_f32(target_src_ptr, hidden_size),
            layer_out_f32=out,
        )
'''


def _embedding_source() -> str:
    return '''GGUF_EMBEDDING_OUTPUT_BF16 = "bf16"
KernelKey = object

def resolve(*args, **kwargs):
    pass

def launch_gguf_embedding():
    pass
'''


def _layer0_artifact() -> dict[str, object]:
    return {
        "status": "mismatched",
        "classification": "layer_boundary_mismatch",
        "layer_id": 0,
        "model": "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "prompt_tokens": [1, 2, 3],
        "position": 2,
        "token_id": 3,
        "next_action": "inspect_initial_embedding_or_token_capture",
        "numeric_delta": {
            "rmse": 0.012114962719327636,
            "max_abs_diff": 0.045740051893517375,
            "mean_abs_diff": 0.009691874419019086,
            "llamacpp_sha256": "llama-layer0",
            "hipengine_sha256": "hip-layer0",
        },
    }
