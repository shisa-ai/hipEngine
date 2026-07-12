from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_pre_output_norm_capture_plan import (
    audit_hipengine_pre_output_capture_path,
    audit_llamacpp_pre_output_patch,
    build_pre_output_norm_capture_plan,
    decide_plan,
)


def test_llamacpp_patch_audit_finds_unique_pre_output_anchor() -> None:
    audit = audit_llamacpp_pre_output_patch(_qwen_source(), _context_source())

    assert audit["ready"] is True
    assert audit["anchor_count"] == 1
    assert "h_nextn_pre_output_norm" in audit["new_text"]
    assert "res->t_h_nextn = cur" in audit["new_text"]
    assert audit["post_norm_h_nextn_currently_present"] is True


def test_hipengine_pre_output_capture_path_detects_private_replay_strategy() -> None:
    audit = audit_hipengine_pre_output_capture_path(_runner_source())
    facts = audit["facts"]

    assert audit["ready"] is True
    assert facts["serial_loop_has_src_before_output_norm"] is True
    assert facts["output_norm_called_with_src_ptr"] is True
    assert facts["bf16_copy_helper_available"] is True
    assert facts["output_norm_fp32_seed_uses_same_src_ptr"] is True


def test_decide_plan_ready_when_all_facts_present() -> None:
    result = decide_plan(
        shared_audit={"conclusion": "shared_serial_path_mismatch_before_or_at_output_norm"},
        llama_patch={"ready": True},
        hip_capture={"ready": True},
        build_inputs={"ready": True},
    )

    assert result["status"] == "ready"
    assert result["conclusion"] == "pre_output_norm_capture_plan_ready"
    assert result["next_action"] == (
        "build_patched_llamacpp_pre_output_norm_harness_and_compare_serial_rows"
    )


def test_decide_plan_blocks_missing_patch() -> None:
    result = decide_plan(
        shared_audit={"conclusion": "shared_serial_path_mismatch_before_or_at_output_norm"},
        llama_patch={"ready": False},
        hip_capture={"ready": True},
        build_inputs={"ready": True},
    )

    assert result["status"] == "blocked"
    assert result["next_action"] == "inspect_pre_output_norm_plan_blockers"


def test_build_pre_output_norm_capture_plan_from_synthetic_inputs(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen35moe.cpp"
    context = tmp_path / "llama-context.cpp"
    runner = tmp_path / "runner.py"
    compile_artifact = tmp_path / "compile.json"
    shared_audit = tmp_path / "shared.json"
    source_dir = tmp_path / "src"
    build_dir = tmp_path / "build"
    lib_dir = tmp_path / "lib"
    exe = tmp_path / "capture"
    source_dir.mkdir()
    build_dir.mkdir()
    lib_dir.mkdir()
    exe.write_text("#!/bin/sh\n")
    qwen.write_text(_qwen_source())
    context.write_text(_context_source())
    runner.write_text(_runner_source())
    compile_artifact.write_text(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "build_dir": str(build_dir),
                "lib_dir": str(lib_dir),
                "outputs": {"executable": str(exe)},
            }
        )
    )
    shared_audit.write_text(
        json.dumps({"conclusion": "shared_serial_path_mismatch_before_or_at_output_norm"})
    )

    artifact = build_pre_output_norm_capture_plan(
        qwen35moe_path=qwen,
        context_path=context,
        runner_path=runner,
        compile_artifact_path=compile_artifact,
        shared_audit_path=shared_audit,
    )

    assert artifact["status"] == "ready"
    assert artifact["conclusion"] == "pre_output_norm_capture_plan_ready"
    assert artifact["llamacpp_pre_output_patch"]["anchor_count"] == 1
    assert artifact["hipengine_pre_output_capture"]["ready"] is True
    assert artifact["build_inputs"]["ready"] is True
    assert artifact["external_checkout_modified"] is False
    json.dumps(artifact)


def _qwen_source() -> str:
    return '''void graph() {
    cur = inpL;

    // post-norm hidden state feeds both the LM head and the MTP seed below
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;
}
'''


def _context_source() -> str:
    return '''// unmasked: nextn rows are stored densely, indexed by raw token position.
return embd_nextn.data + (size_t) i * n_embd;
'''


def _runner_source() -> str:
    return '''def _copy_bf16_ptr_to_host_f32():
    pass

class Session:
    def _run_current_hidden_to_final_hidden(self):
        src = self._hidden_a
        dst = self._hidden_b
        for layer_id, layer_type in enumerate(self.runner.weights.config.layer_types):
            pass
            src, dst = dst, src
        return self._run_output_norm_hidden(src.ptr, self.scratch.norm.ptr)

    def _run_output_norm_hidden(self, src_ptr, out_ptr):
        gguf_rmsnorm_bf16_f32_weight_out_f32(src_ptr, weight, self.scratch.hidden_seed_fp32.ptr)

    def _set_token_id_device(self):
        pass

    def _set_full_attention_position_device(self):
        pass
'''
