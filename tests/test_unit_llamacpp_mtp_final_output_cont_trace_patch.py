from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_final_output_cont_trace_patch import (
    build_final_output_cont_trace_patch_text,
    build_patch_artifact,
    render_combined_diff,
)


def test_generates_final_output_cont_patch() -> None:
    result = build_final_output_cont_trace_patch_text(
        graph_text=_graph_source(wired=False),
        qwen35moe_text=_qwen35moe_source(wired=False),
    )
    diff = render_combined_diff(result)

    assert result.status == "patch_ready"
    assert '+             std::strncmp(name, "final_output_cont_", 18) == 0 ||' in diff
    assert '+         std::strncmp(name, "final_output_cont_", 18) == 0 ||' in diff
    assert '+             std::strcmp(name, "final_output_cont") == 0 ||' in diff
    assert "+    ggml_tensor * final_output_cont = ggml_cont_3d" in diff
    assert '+    cb(final_output_cont, "final_output_cont", il);' in diff
    assert (
        "+    cur = build_lora_mm(model.layers[il].ssm_out, final_output_cont,"
        in diff
    )
    assert " \n" not in diff
    assert "b/src/llama-graph.cpp" in diff
    assert "b/src/models/qwen35moe.cpp" in diff


def test_reports_already_wired_without_patch() -> None:
    first = build_final_output_cont_trace_patch_text(
        graph_text=_graph_source(wired=False),
        qwen35moe_text=_qwen35moe_source(wired=False),
    )
    second = build_final_output_cont_trace_patch_text(
        graph_text=first.graph.patched_text,
        qwen35moe_text=first.qwen35moe.patched_text,
    )

    assert second.status == "already_wired"
    assert render_combined_diff(second) == ""


def test_reports_missing_graph_anchor_without_mutating_graph() -> None:
    result = build_final_output_cont_trace_patch_text(
        graph_text="void unrelated() {}\n",
        qwen35moe_text=_qwen35moe_source(wired=False),
    )

    assert result.status == "graph_anchor_missing"
    assert result.graph.patched_text == "void unrelated() {}\n"


def test_reports_missing_qwen35moe_anchor_without_mutating_qwen35moe() -> None:
    result = build_final_output_cont_trace_patch_text(
        graph_text=_graph_source(wired=False),
        qwen35moe_text="void unrelated() {}\n",
    )

    assert result.status == "qwen35moe_anchor_missing"
    assert result.qwen35moe.patched_text == "void unrelated() {}\n"


def test_writes_patch_and_json_artifact(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp-hip"
    graph_path = root / "src" / "llama-graph.cpp"
    qwen35moe_path = root / "src" / "models" / "qwen35moe.cpp"
    graph_path.parent.mkdir(parents=True)
    qwen35moe_path.parent.mkdir(parents=True)
    graph_path.write_text(_graph_source(wired=False))
    qwen35moe_path.write_text(_qwen35moe_source(wired=False))
    patch_path = tmp_path / "out.patch"

    artifact = build_patch_artifact(llamacpp_root=root, patch_output=patch_path)

    assert artifact["status"] == "patch_ready"
    assert artifact["validation"]["graph_renames_final_output_cont"] is True
    assert artifact["validation"]["graph_wants_final_output_cont_prefix"] is True
    assert artifact["validation"]["graph_adds_final_output_cont_prefix"] is True
    assert artifact["validation"]["qwen35moe_emits_final_output_cont"] is True
    assert artifact["validation"]["qwen35moe_ssm_out_uses_final_output_cont"] is True
    assert artifact["trace_labels_enabled"] == ["final_output_cont_0"]
    assert patch_path.exists()
    assert "final_output_cont" in patch_path.read_text()
    json.dumps(artifact)


def _graph_source(*, wired: bool) -> str:
    rename = (
        '             std::strcmp(name, "final_output_cont") == 0 ||\n'
        if wired
        else ""
    )
    wants = (
        '             std::strncmp(name, "final_output_cont_", 18) == 0 ||\n'
        if wired
        else ""
    )
    add_tensor = (
        '         std::strncmp(name, "final_output_cont_", 18) == 0 ||\n'
        if wired
        else ""
    )
    return (
        "static bool llama_mtp_debug_tensor_trace_wants(const char * name) {\n"
        "    return name != nullptr &&\n"
        '            (std::strncmp(name, "attn_norm_", 10) == 0 ||\n'
        f"{wants}"
        '             std::strncmp(name, "final_output_", 13) == 0 ||\n'
        '             std::strncmp(name, "linear_attn_out_", 16) == 0);\n'
        "}\n"
        "\n"
        "void llm_graph_result::add_mtp_debug_tensor(const char * name, ggml_tensor * tensor) {\n"
        "    const bool target_tensor_trace =\n"
        "        name != nullptr &&\n"
        '        (std::strncmp(name, "attn_norm_", 10) == 0 ||\n'
        f"{add_tensor}"
        '         std::strncmp(name, "final_output_", 13) == 0 ||\n'
        '         std::strncmp(name, "linear_attn_out_", 16) == 0);\n'
        "}\n"
        "\n"
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


def _qwen35moe_source(*, wired: bool) -> str:
    cont = (
        "\n"
        "    ggml_tensor * final_output_cont = ggml_cont_3d(ctx0, final_output,\n"
        "            head_v_dim * num_v_heads, n_seq_tokens, n_seqs);\n"
        '    cb(final_output_cont, "final_output_cont", il);\n'
        if wired
        else ""
    )
    ssm_out_input = "final_output_cont" if wired else "final_output"
    return (
        "    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]\n"
        "    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);\n"
        '    cb(final_output, "final_output", il);\n'
        f"{cont}\n"
        "    // Output projection\n"
        f"    cur = build_lora_mm(model.layers[il].ssm_out, {ssm_out_input}, model.layers[il].ssm_out_s);\n"
    )
