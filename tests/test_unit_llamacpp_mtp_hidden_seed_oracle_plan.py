from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_hidden_seed_oracle_plan import (
    audit_doc_contract,
    audit_llamacpp_nextn_api,
    build_hidden_seed_oracle_plan,
    decide_readiness,
    parse_prompt_tokens,
)


def test_audit_llamacpp_nextn_api_detects_extension_contract() -> None:
    sources = _llamacpp_sources()

    audit = audit_llamacpp_nextn_api(sources)

    assert audit["ready"] is True
    assert audit["facts"]["declares_set_embeddings_nextn"] is True
    assert audit["facts"]["declares_get_embeddings_nextn_ith"] is True
    assert audit["facts"]["context_has_embd_nextn_buffer"] is True
    assert audit["anchors"]["llama_ext_set"] == 1


def test_doc_contract_detects_fp32_post_output_norm() -> None:
    doc = (
        "Target hidden-row seed = POST output-norm hidden, at fp32\n"
        "GGML_TYPE_F32\n"
        "ggml_backend_tensor_get_async\n"
    )

    audit = audit_doc_contract(doc)

    assert audit["ready"] is True
    assert audit["facts"]["requires_post_output_norm"] is True
    assert audit["facts"]["requires_fp32"] is True


def test_build_hidden_seed_oracle_plan_ready(tmp_path: Path) -> None:
    source_dir = _write_llamacpp_tree(tmp_path)
    runner = tmp_path / "runner.py"
    doc = tmp_path / "MTP-gguf.md"
    decision = tmp_path / "decision.json"
    runner.write_text(_runner_source())
    doc.write_text(
        "Target hidden-row seed = POST output-norm hidden, at fp32\n"
        "GGML_TYPE_F32\n"
        "ggml_backend_tensor_get_async\n"
    )
    decision.write_text(
        json.dumps(
            {
                "conclusion": "fp32_seed_target_exists_but_activation_lane_is_bf16",
                "numeric_evidence": {"earliest_layer0_rmse": 1.25e-6},
            }
        )
    )

    artifact = build_hidden_seed_oracle_plan(
        source_dir=source_dir,
        runner_path=runner,
        doc_path=doc,
        decision_path=decision,
        prompt_tokens=(10, 11, 12),
        position=2,
    )

    assert artifact["status"] == "ready"
    assert artifact["conclusion"] == "llamacpp_nextn_embedding_oracle_capture_ready"
    assert artifact["comparison_contract"]["token_id_at_position"] == 12
    assert artifact["comparison_contract"]["hidden_width"] == 2048
    assert artifact["harness_plan"]["required_symbols"] == [
        "llama_set_embeddings_nextn",
        "llama_get_embeddings_nextn_ith",
    ]
    assert artifact["next_action"] == (
        "compile_and_run_llamacpp_nextn_hidden_seed_capture_harness"
    )
    json.dumps(artifact)


def test_decide_readiness_blocks_missing_prior_decision() -> None:
    ready = {"ready": True}
    result = decide_readiness(
        llama_api=ready,
        qwen_graph=ready,
        context_extract=ready,
        hipengine_api=ready,
        doc_contract=ready,
        decision={"conclusion": "other"},
    )

    assert result["status"] == "blocked"
    assert result["next_action"] == "inspect_missing_oracle_plan_facts"


def test_parse_prompt_tokens_rejects_empty() -> None:
    assert parse_prompt_tokens("1, 2,3") == (1, 2, 3)
    try:
        parse_prompt_tokens(" , ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected ValueError")


def _write_llamacpp_tree(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    (root / "src" / "models").mkdir(parents=True)
    (root / "src" / "llama-ext.h").write_text(_llama_ext_h())
    (root / "src" / "llama-context.h").write_text(_llama_context_h())
    (root / "src" / "llama-context.cpp").write_text(_llama_context_cpp())
    (root / "src" / "models" / "qwen35moe.cpp").write_text(_qwen35moe_cpp())
    return root


def _llamacpp_sources() -> dict[str, str]:
    return {
        "llama-ext.h": _llama_ext_h(),
        "llama-context.h": _llama_context_h(),
        "llama-context.cpp": _llama_context_cpp(),
        "qwen35moe.cpp": _qwen35moe_cpp(),
    }


def _llama_ext_h() -> str:
    return (
        "LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, "
        "bool value, bool masked);\n"
        "LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);\n"
        "LLAMA_API float * llama_get_embeddings_nextn_ith("
        "struct llama_context * ctx, int32_t i);\n"
    )


def _llama_context_h() -> str:
    return """struct llama_context {
    buffer_view<float> embd_nextn = {nullptr, 0};
};
"""


def _llama_context_cpp() -> str:
    return (
        "auto * t_h_nextn = cparams.embeddings_nextn ? "
        "res->get_h_nextn() : nullptr;\n"
        "auto * t_h_nextn2 = cparams.embeddings_nextn ? "
        "res->get_h_nextn()  : nullptr;\n"
        "ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, "
        "0, n_tokens*n_embd*sizeof(float));\n"
        "// unmasked: nextn rows are stored densely\n"
        "return embd_nextn.data + (size_t) i * n_embd;\n"
        "embd_nextn.size = (size_t) n_embd_out * n_batch;\n"
        "void llama_set_embeddings_nextn("
        "llama_context * ctx, bool value, bool masked) {}\n"
        "float * llama_get_embeddings_nextn_ith("
        "llama_context * ctx, int32_t i) { return nullptr; }\n"
    )


def _qwen35moe_cpp() -> str:
    return """// post-norm hidden state feeds both the LM head and the MTP seed below
cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
cb(cur, "h_nextn", -1);
res->t_h_nextn = cur;
// LM head
res->t_h_nextn= cur;
"""


def _runner_source() -> str:
    return """class Qwen35GGUFResidentSession:
    def prefill(self, token_ids, *, capture_hidden_seed_fp32: bool = False):
        pass

    def step(self, token_id, *, capture_hidden_seed_fp32: bool = False):
        pass

    def fp32_hidden_seed_ptr(self):
        if not self.fp32_hidden_seed_contract().ready_for_mtp:
            raise RuntimeError("not ready")

    def mtp_draft_seed(self):
        return Seed(hidden_ptr=self.fp32_hidden_seed_ptr())

    def _run_output_norm_hidden(self):
        gguf_rmsnorm_bf16_f32_weight_out_f32()
"""
