from __future__ import annotations

import json
from pathlib import Path

from scripts.stepfun_correctness_status import build_status, main


def _write_prompt_artifact(path: Path) -> None:
    selected_slots = ["root.token_embedding", "root.output_norm", "root.lm_head"]
    dense_slots = (
        "attn_norm",
        "attn_q_norm",
        "attn_k_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "ffn_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    )
    moe_slots = (
        "attn_norm",
        "attn_q_norm",
        "attn_k_norm",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "ffn_norm",
        "ffn_gate_inp",
        "exp_probs_bias",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    )
    for layer_id in range(45):
        suffixes = dense_slots if layer_id < 3 else moe_slots
        selected_slots.extend(f"layers.{layer_id}.{suffix}" for suffix in suffixes)
    path.write_text(
        json.dumps(
            {
                "status": "partial_prompt_smoke",
                "execution_mode": "chunked",
                "layer_count": 45,
                "skipped_layers": [],
                "selected_slot_count": len(selected_slots),
                "selected_slots": selected_slots,
                "no_vision_projector_mtp_slots": True,
                "backend": "hip_gfx1151",
                "next_token_id": 369,
                "next_token_text": " |",
                "peak_resident_weight_nbytes": 3_531_578_496,
                "memory_stats_after_free": {
                    "active_allocations": 0,
                    "current_allocated_bytes": 0,
                },
            }
        )
    )


def _write_oracle_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "executed",
                "returncode": 1,
                "text_matches_expected_exact": False,
                "elapsed_s": 62.4,
                "llama_cpp_version": "version: test (deadbeef)",
                "stdout": "",
                "stderr": "unknown model architecture: 'step35'",
                "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
                "oracle_blocker_detail": "local llama.cpp build reports unknown model architecture: 'step35'",
                "step35_supported": False,
            }
        )
    )


def _write_docs(path: Path) -> None:
    path.write_text(
        "# StepFun\n\n"
        "### P0 — setup\n\n"
        "- [x] Done setup.\n"
        "- [ ] Wire KV-backed decode.\n"
        "- [~] Compare oracle.\n"
        "### P13 — benchmark\n\n"
        "- [ ] Out-of-scope benchmark item.\n"
    )


def test_stepfun_correctness_status_reports_remaining_blockers(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs)

    assert status["status"] == "blocked"
    assert status["all_layer_prompt_smoke"] is True
    assert status["all_layer_prompt_next_token_id"] == 369
    assert status["oracle_parity"] is False
    assert status["oracle_status"] == "executed"
    assert status["oracle_elapsed_s"] == 62.4
    assert status["oracle_llama_cpp_version"] == "version: test (deadbeef)"
    assert status["oracle_stdout_len"] == 0
    assert status["oracle_stderr_len"] == len("unknown model architecture: 'step35'")
    assert status["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert status["step35_supported_by_local_llama_cpp"] is False
    projections = status["linear_projection_progress"]
    assert projections["source"] == "prompt_artifact.selected_slots"
    assert projections["execution_mode"] == "chunked"
    assert projections["layer_count"] == 45
    assert projections["selected_layer_count"] == 45
    assert projections["selected_slot_count"] == 753
    assert projections["root_lm_head_present"] is True
    assert projections["resident_linear_projection_slot_count"] == 487
    assert projections["host_reference_router_projection_slot_count"] == 42
    assert projections["attention"]["slot_count"] == 225
    assert projections["attention"]["expected_slot_count"] == 225
    assert projections["attention"]["complete_layer_count"] == 45
    assert projections["attention"]["all_selected_layers_complete"] is True
    assert projections["dense_mlp"]["slot_count"] == 9
    assert projections["dense_mlp"]["complete_layer_count"] == 3
    assert projections["moe_router"]["slot_count"] == 42
    assert projections["moe_router"]["complete_layer_count"] == 42
    assert projections["moe_expert"]["slot_count"] == 126
    assert projections["moe_expert"]["complete_layer_count"] == 42
    assert projections["moe_shared_expert"]["slot_count"] == 126
    assert projections["moe_shared_expert"]["complete_layer_count"] == 42
    assert status["kv_backed_decode_ready"] is False
    assert status["e2e_inference_ready"] is False
    assert {blocker["kind"] for blocker in status["blockers"]} == {
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    }
    assert {action["blocker_kind"] for action in status["next_actions"]} == {
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    }
    assert status["docs_checklist"]["open_or_partial_count_p0_p12"] == 2
    assert [item["state"] for item in status["docs_checklist"]["open_or_partial_items_p0_p12"]] == [
        "open",
        "partial",
    ]


def test_stepfun_correctness_status_writes_json(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    output = tmp_path / "status.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload["status"] == "blocked"
    assert payload["all_layer_prompt_smoke"] is True
    assert payload["e2e_inference_ready"] is False
    assert payload["linear_projection_progress"]["resident_linear_projection_slot_count"] == 487
    assert len(payload["next_actions"]) == 2
    assert payload["docs_checklist"]["open_or_partial_count_p0_p12"] == 2


def test_stepfun_correctness_status_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--docs",
            str(docs),
            "--fail-on-blocked",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["status"] == "blocked"
    assert payload["e2e_inference_ready"] is False
