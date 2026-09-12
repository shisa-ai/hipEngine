from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_layer_input_patch import (
    LAYER_INPUT_ASSIGNMENT,
    build_layer_input_patch_text,
    build_patch_artifact,
    render_unified_diff,
)


def test_generates_minimal_qwen35moe_layer_input_patch() -> None:
    source = _qwen35moe_source(wired=False)

    result = build_layer_input_patch_text(source)
    diff = render_unified_diff(
        result.original_text,
        result.patched_text,
        relative_path=Path("x.cpp"),
    )

    assert result.status == "patch_ready"
    assert result.insertion_line == 4
    assert result.patched_text.count(LAYER_INPUT_ASSIGNMENT) == 1
    assert "+        res->t_layer_inp[il] = inpL;" in diff
    assert "ggml_tensor * inpSA = inpL;" in diff


def test_reports_already_wired_source_without_patch() -> None:
    source = _qwen35moe_source(wired=True)

    result = build_layer_input_patch_text(source)
    diff = render_unified_diff(
        result.original_text,
        result.patched_text,
        relative_path=Path("x.cpp"),
    )

    assert result.status == "already_wired"
    assert result.changed is False
    assert diff == ""
    assert result.insertion_line == 4


def test_reports_anchor_missing_without_mutating_text() -> None:
    source = "void unrelated() {}\n"

    result = build_layer_input_patch_text(source)

    assert result.status == "anchor_missing"
    assert result.changed is False
    assert result.patched_text == source
    assert result.insertion_line is None


def test_writes_patch_and_json_artifact(tmp_path: Path) -> None:
    root = tmp_path / "llama.cpp-hip"
    target = root / "src" / "models" / "qwen35moe.cpp"
    target.parent.mkdir(parents=True)
    target.write_text(_qwen35moe_source(wired=False))
    patch_path = tmp_path / "out.patch"

    artifact = build_patch_artifact(llamacpp_root=root, patch_output=patch_path)

    assert artifact["status"] == "patch_ready"
    assert artifact["source_exists"] is True
    assert artifact["validation"]["single_assignment_added"] is True
    assert artifact["validation"]["diff_has_expected_assignment"] is True
    assert artifact["reference_basis"]["external_checkout_modified"] is False
    assert patch_path.exists()
    assert "+        res->t_layer_inp[il] = inpL;" in patch_path.read_text()
    json.dumps(artifact)


def _qwen35moe_source(*, wired: bool) -> str:
    assignment = f"{LAYER_INPUT_ASSIGNMENT}\n" if wired else ""
    return (
        "void llama_model_qwen35moe::graph::build() {\n"
        "    // MTP/NextN layers are loaded as extra decoder blocks but "
        "not executed in the main pass.\n"
        "    for (int il = 0; il < n_layer; ++il) {\n"
        f"{assignment}"
        "        ggml_tensor * inpSA = inpL;\n"
        "        cur = build_norm(inpL, model.layers[il].attn_norm, "
        "nullptr, LLM_NORM_RMS, il);\n"
        "    }\n"
        "}\n"
    )
