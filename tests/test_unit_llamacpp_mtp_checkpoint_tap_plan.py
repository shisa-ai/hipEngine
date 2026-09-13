from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_checkpoint_tap_plan import (
    analyze_source_tree,
    build_checkpoint_mapping,
    build_tap_plan_artifact,
    choose_preferred_arch,
    summarize_checkpoint_target,
)


def test_prefers_qwen35moe_for_a3b_moe_checkpoint(tmp_path: Path) -> None:
    root = _write_source_tree(tmp_path, qwen35moe_wires_layer_input=False)
    checkpoint = _checkpoint_summary()
    target = summarize_checkpoint_target(checkpoint)
    inventory = analyze_source_tree(root)

    assert target["has_moe_arrays"] is True
    assert choose_preferred_arch(target, inventory) == "qwen35moe"


def test_detects_existing_api_but_missing_qwen35moe_layer_input(tmp_path: Path) -> None:
    root = _write_source_tree(tmp_path, qwen35moe_wires_layer_input=False)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(_checkpoint_summary()))

    artifact = build_tap_plan_artifact(
        llamacpp_root=root,
        checkpoint_summary_path=checkpoint_path,
        trace_inventory_path=None,
    )

    support = artifact["extraction_support"]
    hidden_mapping = artifact["checkpoint_mapping"][0]
    assert support["existing_layer_input_api_present"] is True
    assert support["needs_preferred_arch_layer_input_patch"] is True
    assert hidden_mapping["hipengine_key"] == "hidden_in_f32"
    assert hidden_mapping["status"] == "source_patch_needed_for_layer_input"
    assert artifact["next_action"] == (
        "prepare_llamacpp_qwen35moe_layer_input_patch_then_capture_hidden_in"
    )


def test_maps_checkpoint_arrays_to_named_llamacpp_callbacks(tmp_path: Path) -> None:
    root = _write_source_tree(tmp_path, qwen35moe_wires_layer_input=True)
    checkpoint = _checkpoint_summary()
    target = summarize_checkpoint_target(checkpoint)
    inventory = analyze_source_tree(root)
    mapping = build_checkpoint_mapping(target, inventory["architectures"]["qwen35moe"])
    by_key = {item["hipengine_key"]: item for item in mapping}

    assert by_key["hidden_in_f32"]["status"] == "existing_layer_input_api_ready"
    assert by_key["attn_out_f32"]["llamacpp_tensor"] == "attn_output-3"
    assert by_key["residual_f32"]["llamacpp_tensor"] == "attn_residual-3"
    assert by_key["post_norm_f32"]["llamacpp_tensor"] == "attn_post_norm-3"
    assert by_key["moe_shared_out_f32"]["llamacpp_tensor"] == "ffn_shexp_gated-3"
    assert by_key["layer_out_f32"]["llamacpp_tensor"] == "l_out-3"


def _checkpoint_summary() -> dict[str, object]:
    return {
        "capture": {
            "model": "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "position": 16,
            "token_id": 271,
            "layer_id": 3,
            "layer_type": "full_attention",
            "run_preceding_layers": True,
            "preceding_layer_count": 3,
            "hidden_size": 2048,
        },
        "arrays": {
            "hidden_in_f32": {"sha256": "hidden"},
            "attn_out_f32": {"sha256": "attn"},
            "residual_f32": {"sha256": "residual"},
            "post_norm_f32": {"sha256": "post_norm"},
            "moe_shared_out_f32": {"sha256": "shared"},
            "layer_out_f32": {"sha256": "layer"},
        },
    }


def _write_source_tree(tmp_path: Path, *, qwen35moe_wires_layer_input: bool) -> Path:
    root = tmp_path / "llama.cpp-hip"
    models = root / "src" / "models"
    models.mkdir(parents=True)
    (root / "common").mkdir()
    layer_input_line = "res->t_layer_inp[il] = inpL;" if qwen35moe_wires_layer_input else ""
    (models / "qwen35moe.cpp").write_text(
        f"""
std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(
        const llm_graph_params & params) const {{
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {{
        return std::make_unique<graph_mtp>(*this, params);
    }}
    return std::make_unique<graph>(*this, params);
}}
llama_model_qwen35moe::graph::graph(
        const llama_model & model, const llm_graph_params & params) {{
    inpL = build_inp_embd(model.tok_embd);
    for (int il = 0; il < n_layer; ++il) {{
        {layer_input_line}
        ggml_tensor * inpSA = inpL;
        cb(cur, "attn_norm", il);
        cb(cur, "attn_output", il);
        cb(cur, "attn_residual", il);
        cb(cur, "attn_post_norm", il);
        cb(moe_out, "ffn_moe_out", il);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        cb(cur, "ffn_out", il);
        cb(cur, "post_moe", il);
        cb(cur, "l_out", il);
    }}
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;
}}
// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
llama_model_qwen35moe::graph_mtp::graph_mtp(
        const llama_model & model, const llm_graph_params & params) {{
    cb(cur, "mtp_attn_out", il);
    cb(cur, "mtp_ffn_out", il);
}}
"""
    )
    (models / "qwen35.cpp").write_text(
        "for (int il = 0; il < n_layer; ++il) { res->t_layer_inp[il] = inpL; }"
    )
    (root / "src" / "llama-ext.h").write_text(
        """
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx,
        uint32_t lid, bool value);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx,
        uint32_t lid);
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx,
        bool value, bool masked);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);
"""
    )
    (root / "src" / "llama-context.cpp").write_text(
        """
void set_embeddings_layer_inp(uint32_t lid, bool enable) {}
float * get_embeddings_layer_inp(uint32_t lid) { return nullptr; }
void extract_layer_inputs(const llm_graph_result * res, size_t off, size_t n) {}
llm_graph_cb llama_context::graph_get_cb() const { return {}; }
void f() { ggml_backend_tensor_get_async(nullptr, nullptr, nullptr, 0, 0); }
"""
    )
    (root / "src" / "llama-graph.cpp").write_text(
        """
void llm_graph_result::set_outputs(const llm_graph_params & params) {
    ggml_set_output(t_h_nextn);
    ggml_set_output(t_layer_inp[il]);
}
"""
    )
    (root / "common" / "speculative.cpp").write_text(
        """
llama_set_embeddings_layer_inp(ctx_tgt, 3, true);
const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, 3);
"""
    )
    return root
