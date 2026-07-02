from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_linear_attn_trace_patch import (
    build_linear_attn_trace_patch_text,
    build_patch_artifact,
    render_combined_diff,
)


def test_generates_linear_attn_trace_allowlist_patch() -> None:
    result = build_linear_attn_trace_patch_text(
        graph_text=_graph_source(wired=False),
        context_text=_context_source(wired=False),
    )
    diff = render_combined_diff(result)

    assert result.status == "patch_ready"
    assert "+             std::strncmp(name, \"linear_attn_qkv_mixed_\", 22) == 0 ||" in diff
    assert "+         std::strncmp(name, \"linear_attn_qkv_mixed_\", 22) == 0 ||" in diff
    assert "+             std::strcmp(name, \"q_conv_predelta\") == 0 ||" in diff
    assert "+    if (label.rfind(\"beta_\", 0) == 0 ||" in diff
    assert result.graph.patched_text.count("linear_attn_qkv_mixed_") == 2
    assert "b/src/llama-graph.cpp" in diff
    assert "b/src/llama-context.cpp" in diff


def test_reports_already_wired_trace_sources_without_patch() -> None:
    first = build_linear_attn_trace_patch_text(
        graph_text=_graph_source(wired=False),
        context_text=_context_source(wired=False),
    )
    second = build_linear_attn_trace_patch_text(
        graph_text=first.graph.patched_text,
        context_text=first.context.patched_text,
    )

    assert second.status == "already_wired"
    assert render_combined_diff(second) == ""


def test_reports_missing_graph_anchor_without_mutating_graph() -> None:
    result = build_linear_attn_trace_patch_text(
        graph_text="void unrelated() {}\n",
        context_text=_context_source(wired=False),
    )

    assert result.status == "graph_anchor_missing"
    assert result.graph.patched_text == "void unrelated() {}\n"


def test_reports_missing_context_anchor_without_mutating_context() -> None:
    result = build_linear_attn_trace_patch_text(
        graph_text=_graph_source(wired=False),
        context_text="void unrelated() {}\n",
    )

    assert result.status == "context_anchor_missing"
    assert result.context.patched_text == "void unrelated() {}\n"


def test_writes_patch_and_json_artifact(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp-hip"
    graph_path = root / "src" / "llama-graph.cpp"
    context_path = root / "src" / "llama-context.cpp"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(_graph_source(wired=False))
    context_path.write_text(_context_source(wired=False))
    patch_path = tmp_path / "out.patch"

    artifact = build_patch_artifact(llamacpp_root=root, patch_output=patch_path)

    assert artifact["status"] == "patch_ready"
    assert artifact["reference_basis"]["external_checkout_modified"] is False
    assert artifact["validation"]["graph_allows_projection_input"] is True
    assert artifact["validation"]["context_qkv_views_use_token_dim_2"] is True
    assert artifact["next_action"] == (
        "apply_patch_to_temporary_llamacpp_trace_tree_and_capture_layer0_pre_ssm_labels"
    )
    assert patch_path.exists()
    assert "linear_attn_qkv_mixed_" in patch_path.read_text()
    json.dumps(artifact)


def _graph_source(*, wired: bool) -> str:
    prefix = (
        '             std::strncmp(name, "linear_attn_qkv_mixed_", 22) == 0 ||\n'
        if wired
        else ""
    )
    rename = (
        '             std::strcmp(name, "linear_attn_qkv_mixed") == 0 ||\n'
        '             std::strcmp(name, "q_conv_predelta") == 0 ||\n'
        if wired
        else ""
    )
    return (
        "static bool llama_mtp_debug_tensor_trace_wants(const char * name) {\n"
        "    if (name != nullptr &&\n"
        '            (std::strncmp(name, "attn_norm_", 10) == 0 ||\n'
        f"{prefix}"
        '             std::strncmp(name, "final_output_", 13) == 0 ||\n'
        '             std::strncmp(name, "linear_attn_out_", 16) == 0)) {\n'
        "        return true;\n"
        "    }\n"
        "}\n"
        "void llm_graph_result::add_mtp_debug_tensor(const char * name, ggml_tensor * tensor) {\n"
        "    const bool target_tensor_trace =\n"
        "        name != nullptr &&\n"
        '        (std::strcmp(name, "h_nextn_pre_output_norm") == 0 ||\n'
        '         std::strcmp(name, "h_nextn") == 0 ||\n'
        '         std::strncmp(name, "attn_norm_", 10) == 0 ||\n'
        f"{prefix.replace('             ', '         ')}"
        '         std::strncmp(name, "final_output_", 13) == 0 ||\n'
        '         std::strncmp(name, "linear_attn_out_", 16) == 0);\n'
        "}\n"
        "void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {\n"
        "    if (res->get_params().gtype != LLM_GRAPH_TYPE_DECODER_MTP &&\n"
        "            name != nullptr &&\n"
        '            (std::strcmp(name, "attn_norm") == 0 ||\n'
        f"{rename}"
        '             std::strcmp(name, "final_output") == 0 ||\n'
        '             std::strcmp(name, "linear_attn_out") == 0) &&\n'
        "            il >= 0) {}\n"
        "}\n"
    )


def _context_source(*, wired: bool) -> str:
    token_dim = (
        '    if (label.rfind("beta_", 0) == 0 ||\n'
        '            label.rfind("q_conv_", 0) == 0 ||\n'
        '            label.rfind("k_conv_", 0) == 0 ||\n'
        '            label.rfind("v_conv_", 0) == 0) {\n'
        "        return 2;\n"
        "    }\n"
        if wired
        else ""
    )
    return (
        "static int llama_mtp_debug_tensor_token_dim(const std::string & label) {\n"
        '    if (label == "mtp_Qcur_normed" ||\n'
        '            label == "mtp_Vcur") {\n'
        "        return 2;\n"
        "    }\n"
        f"{token_dim}"
        '    if (label.rfind("ffn_moe_gate_up_", 0) == 0 ||\n'
        '            label.rfind("ffn_moe_gate_", 0) == 0) {\n'
        "        return 2;\n"
        "    }\n"
        "    return 1;\n"
        "}\n"
    )
