from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.stepfun_correctness_status import _status_integrity, build_status, main


def _primary_command_fields(kind: str | None, command: str | None) -> dict[str, object]:
    return {
        "primary_command_kind": kind,
        "primary_command": command,
        "primary_command_nchars": len(command) if command is not None else 0,
        "primary_command_sha256": (
            hashlib.sha256(command.encode()).hexdigest() if command is not None else None
        ),
    }



def _stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()



def _oracle_helper_command(
    prompt: Path,
    oracle: Path,
    *,
    diagnostic_logs: bool = False,
) -> str:
    diagnostic_logs_arg = "--diagnostic-logs " if diagnostic_logs else ""
    return (
        "python3 scripts/stepfun_llamacpp_oracle.py "
        f"--artifact {prompt} --llama-cli /tmp/llama-cli "
        "--model /data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf "
        f"--n-predict 1 --timeout-s 60.0 {diagnostic_logs_arg}"
        "--llama-arg=--device --llama-arg=none "
        f"--llama-arg=--gpu-layers --llama-arg=0 --execute --pretty --output {oracle}"
    )



def _oracle_helper_fields(
    prompt: Path,
    oracle: Path,
    *,
    diagnostic_logs: bool = False,
) -> dict[str, object]:
    command = _oracle_helper_command(prompt, oracle, diagnostic_logs=diagnostic_logs)
    return {
        "helper_command_kind": "oracle_helper_refresh_command",
        "helper_command": command,
        "helper_command_nchars": len(command),
        "helper_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
    }


def _source_verify_command(
    status_artifact: Path = Path("benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"),
) -> str:
    return (
        "python3 scripts/stepfun_correctness_status.py "
        f"--verify-source-artifacts {status_artifact} --pretty"
    )



def _streaming_runner_blocker_names() -> list[str]:
    return [
        "streaming_decode_loop_not_wired",
        "kv_kernel_trace_artifact_missing",
        "kv_backed_next_token_artifact_missing",
    ]


def _streaming_runner_blocker_names_sha256() -> str:
    return _stable_json_sha256(_streaming_runner_blocker_names())


def _streaming_runner_blockers() -> list[dict[str, object]]:
    return [
        {
            "name": "streaming_decode_loop_not_wired",
            "ready": False,
            "required_evidence": "resident decode loop must launch KV writes and gated attention",
        },
        {
            "name": "kv_kernel_trace_artifact_missing",
            "ready": False,
            "required_evidence": "retained trace must show KV write and attention kernels",
        },
        {
            "name": "kv_backed_next_token_artifact_missing",
            "ready": False,
            "required_evidence": "KV-backed next-token artifact must be retained",
        },
    ]



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


def _write_oracle_artifact(path: Path, *, diagnostic_logs: bool = False) -> None:
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
                "generated_text": "",
                "llama_cli": "/tmp/llama-cli",
                "model": "/data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf",
                "prompt_length": 23,
                "n_predict": 1,
                "timeout_s": 60.0,
                "diagnostic_logs": diagnostic_logs,
                "extra_llama_args": ["--device", "none", "--gpu-layers", "0"],
                "command_shell": "/tmp/llama-cli --model stepfun.gguf --predict 1 --temp 0",
                "expected_next_token_id": 369,
                "expected_next_token_text": " |",
                "expected_next_token_logit": 19.158626556396484,
                "expected_top_tokens": [
                    {"rank": 1, "token_id": 369, "token_text": " |", "logit": 19.158626556396484},
                    {"rank": 2, "token_id": 5, "token_text": "#", "logit": 18.343582153320312},
                ],
                "comparison_policy": {
                    "generated_text_source": "llama-cli stdout with --no-display-prompt --simple-io",
                    "exact_text_match_field": "text_matches_expected_exact",
                    "stripped_text_match_field": "text_matches_expected_stripped",
                    "expected_text_field": "expected_next_token_text",
                },
                "text_matches_expected_stripped": False,
                "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
                "oracle_blocker_detail": "local llama.cpp build reports unknown model architecture: 'step35'",
                "step35_supported": False,
            }
        )
    )


def _write_resource_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "dry_run_plan",
                "text_decode_resource_plan": {
                    "backend": "hip_gfx1151",
                    "kv_decode_launch_schedule": {
                        "all_stage_dispatch_ready": True,
                        "first_layer_ops": [
                            "layers.0.prompt_kv_write",
                            "layers.0.decode_kv_write",
                            "layers.0.decode_attention",
                        ],
                        "last_layer_ops": [
                            "layers.44.prompt_kv_write",
                            "layers.44.decode_kv_write",
                            "layers.44.decode_attention",
                        ],
                        "layer_count": 45,
                        "operation_count": 135,
                        "per_layer_order": ["prompt_kv_write", "decode_kv_write", "decode_attention"],
                        "source": "text_decode_resource_plan",
                        "stages": [
                            {
                                "dispatch_key": "prompt_kv_write",
                                "layer_count": 45,
                                "name": "prompt_prefill_kv_write",
                                "ready": True,
                                "span_contract": "prompt_span",
                            },
                            {
                                "dispatch_key": "decode_kv_write",
                                "layer_count": 45,
                                "name": "one_token_decode_kv_write",
                                "ready": True,
                                "span_contract": "decode_span",
                            },
                            {
                                "dispatch_key": "decode_attention",
                                "layer_count": 45,
                                "name": "one_token_gated_attention_decode",
                                "ready": True,
                                "span_contract": "decode_span",
                            },
                        ],
                        "streaming_runner_ready": False,
                        "streaming_runner_blocker_count": 3,
                        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
                        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
                        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
                        "streaming_runner_blockers": _streaming_runner_blockers(),
                    },
                    "kv_decode_kernel_plan": {
                        "all_registered": True,
                        "attention_block_size": 256,
                        "attention_block_table_len": 2,
                        "attention_capacity_tokens": 512,
                        "backend": "hip_gfx1151",
                        "decode_attention_kind": "splitk_gate_f32",
                        "decode_span": {
                            "block_size": 256,
                            "block_table_len": 2,
                            "capacity_tokens": 512,
                            "live_counts_len": 1,
                            "max_live_count": 511,
                            "shape_compatible": True,
                        },
                        "decode_span_shape_compatible": True,
                        "max_context": 512,
                        "max_new_tokens": 1,
                        "max_prompt_rows": 511,
                        "prompt_span": {
                            "base_offsets_len_formula": "rows * 2",
                            "block_size": 256,
                            "block_table_len_per_row": 2,
                            "live_counts_len_formula": "rows",
                            "max_prompt_rows": 511,
                            "row_positions_required": True,
                            "shape_compatible": True,
                        },
                        "prompt_span_shape_compatible": True,
                        "span_shape_compatible": True,
                        "dispatch_keys": {
                            "decode_attention": {
                                "backend": "hip_gfx1151",
                                "layer": "paged_attn_decode",
                                "quant": "gguf_step35",
                                "variant": "bf16_split_k_gate_f32_spans",
                            },
                            "decode_kv_write": {
                                "backend": "hip_gfx1151",
                                "layer": "paged_kv_write",
                                "quant": "gguf_step35",
                                "variant": "mixed_bf16_spans",
                            },
                            "prompt_kv_write": {
                                "backend": "hip_gfx1151",
                                "layer": "paged_kv_write",
                                "quant": "gguf_step35",
                                "variant": "mixed_bf16_prompt_spans",
                            },
                        },
                        "kv_storage_dtype": "bf16",
                        "model_quant": "gguf_step35",
                        "registered": {
                            "decode_attention": True,
                            "decode_kv_write": True,
                            "prompt_kv_write": True,
                        },
                    },
                },
                "kv_decode_run_plan": {
                    "attention_block_size": 256,
                    "attention_block_table_len": 2,
                    "context_fits_resource_plan": True,
                    "decode_live_count": 23,
                    "decode_position": 23,
                    "decode_span_inputs": {
                        "attention_live_counts": [23],
                        "attention_live_counts_dtype": "int64",
                        "attention_live_counts_len": 1,
                        "attention_live_counts_nbytes": 8,
                        "base_offsets": [0, 1],
                        "base_offsets_dtype": "int32",
                        "base_offsets_len": 2,
                        "base_offsets_nbytes": 8,
                        "block_size": 256,
                        "block_table_len": 2,
                        "kv_write_position": 23,
                        "kv_write_position_dtype": "int64",
                        "kv_write_position_nbytes": 8,
                        "max_live_count": 23,
                        "total_span_input_nbytes": 16,
                    },
                    "input_id_count": 23,
                    "input_id_preview": list(range(100, 108)),
                    "input_ids": list(range(100, 123)),
                    "input_ids_dtype": "int32",
                    "input_ids_nbytes": 92,
                    "input_ids_sha256": "f" * 64,
                    "kv_decode_launch_operation_count": 135,
                    "kv_decode_launch_per_layer_order": [
                        "prompt_kv_write",
                        "decode_kv_write",
                        "decode_attention",
                    ],
                    "kv_dispatch_keys": {
                        "decode_attention": {
                            "backend": "hip_gfx1151",
                            "layer": "paged_attn_decode",
                            "quant": "gguf_step35",
                            "variant": "bf16_split_k_gate_f32_spans",
                        },
                        "decode_kv_write": {
                            "backend": "hip_gfx1151",
                            "layer": "paged_kv_write",
                            "quant": "gguf_step35",
                            "variant": "mixed_bf16_spans",
                        },
                        "prompt_kv_write": {
                            "backend": "hip_gfx1151",
                            "layer": "paged_kv_write",
                            "quant": "gguf_step35",
                            "variant": "mixed_bf16_prompt_spans",
                        },
                    },
                    "max_context": 512,
                    "max_new_tokens": 1,
                    "max_prompt_rows": 511,
                    "prompt_fits_resource_plan": True,
                    "prompt_length": 23,
                    "prompt_positions": list(range(23)),
                    "prompt_span_inputs": {
                        "base_offsets": [value for _ in range(23) for value in (0, 1)],
                        "base_offsets_dtype": "int32",
                        "base_offsets_len": 46,
                        "base_offsets_nbytes": 184,
                        "block_size": 256,
                        "block_table_len_per_row": 2,
                        "live_counts": list(range(23)),
                        "live_counts_dtype": "int64",
                        "live_counts_len": 23,
                        "live_counts_nbytes": 184,
                        "max_live_count": 22,
                        "position_tensor_role": "prompt_row_positions",
                        "rows": 23,
                        "total_span_input_nbytes": 368,
                    },
                    "rendered_prompt_nchars": 123,
                    "rendered_prompt_sha256": "0" * 64,
                    "required_context_tokens": 24,
                    "span_input_total_nbytes": 384,
                    "span_input_upload_manifest": {
                        "entries": [
                            {
                                "dtype": "int32",
                                "kernel_args": ["prompt_kv_write.base_offsets"],
                                "name": "prompt_base_offsets",
                                "nbytes": 184,
                                "shape": [23, 2],
                                "source": "prompt_span_inputs.base_offsets",
                            },
                            {
                                "dtype": "int64",
                                "kernel_args": ["prompt_kv_write.live_counts"],
                                "name": "prompt_live_counts",
                                "nbytes": 184,
                                "shape": [23],
                                "source": "prompt_span_inputs.live_counts",
                            },
                            {
                                "dtype": "int32",
                                "kernel_args": [
                                    "decode_kv_write.base_offsets",
                                    "decode_attention.base_offsets",
                                ],
                                "name": "decode_base_offsets",
                                "nbytes": 8,
                                "shape": [2],
                                "source": "decode_span_inputs.base_offsets",
                            },
                            {
                                "dtype": "int64",
                                "kernel_args": ["decode_kv_write.position"],
                                "name": "decode_kv_write_position",
                                "nbytes": 8,
                                "shape": [],
                                "source": "decode_span_inputs.kv_write_position",
                            },
                            {
                                "dtype": "int64",
                                "kernel_args": ["decode_attention.live_counts"],
                                "name": "decode_attention_live_counts",
                                "nbytes": 8,
                                "shape": [1],
                                "source": "decode_span_inputs.attention_live_counts",
                            },
                        ],
                        "entry_count": 5,
                        "note": "Host-side upload manifest for metadata-only StepFun KV decode planning.",
                        "total_nbytes": 392,
                    },
                    "span_input_host_payloads": {
                        "entries": [
                            {
                                "byte_order": "little",
                                "dtype": "int32",
                                "name": "prompt_base_offsets",
                                "nbytes": 184,
                                "preview_values": [0, 1, 0, 1, 0, 1, 0, 1],
                                "sha256": "a" * 64,
                                "source": "prompt_span_inputs.base_offsets",
                                "value_count": 46,
                            },
                            {
                                "byte_order": "little",
                                "dtype": "int64",
                                "name": "prompt_live_counts",
                                "nbytes": 184,
                                "preview_values": list(range(8)),
                                "sha256": "b" * 64,
                                "source": "prompt_span_inputs.live_counts",
                                "value_count": 23,
                            },
                            {
                                "byte_order": "little",
                                "dtype": "int32",
                                "name": "decode_base_offsets",
                                "nbytes": 8,
                                "preview_values": [0, 1],
                                "sha256": "c" * 64,
                                "source": "decode_span_inputs.base_offsets",
                                "value_count": 2,
                            },
                            {
                                "byte_order": "little",
                                "dtype": "int64",
                                "name": "decode_kv_write_position",
                                "nbytes": 8,
                                "preview_values": [23],
                                "sha256": "d" * 64,
                                "source": "decode_span_inputs.kv_write_position",
                                "value_count": 1,
                            },
                            {
                                "byte_order": "little",
                                "dtype": "int64",
                                "name": "decode_attention_live_counts",
                                "nbytes": 8,
                                "preview_values": [23],
                                "sha256": "e" * 64,
                                "source": "decode_span_inputs.attention_live_counts",
                                "value_count": 1,
                            },
                        ],
                        "entry_count": 5,
                        "note": "Deterministic little-endian host payload hashes for StepFun KV span inputs.",
                        "total_nbytes": 392,
                    },
                    "decode_input_upload_plan": {
                        "entries": [
                            {
                                "dtype": "int32",
                                "name": "input_ids",
                                "nbytes": 92,
                                "sha256": "f" * 64,
                                "shape": [23],
                                "source": "input_ids",
                                "upload_group": "input_tokens",
                            },
                            {
                                "dtype": "int32",
                                "name": "prompt_base_offsets",
                                "nbytes": 184,
                                "sha256": "a" * 64,
                                "shape": [23, 2],
                                "source": "prompt_span_inputs.base_offsets",
                                "upload_group": "kv_span_inputs",
                            },
                            {
                                "dtype": "int64",
                                "name": "prompt_live_counts",
                                "nbytes": 184,
                                "sha256": "b" * 64,
                                "shape": [23],
                                "source": "prompt_span_inputs.live_counts",
                                "upload_group": "kv_span_inputs",
                            },
                            {
                                "dtype": "int32",
                                "name": "decode_base_offsets",
                                "nbytes": 8,
                                "sha256": "c" * 64,
                                "shape": [2],
                                "source": "decode_span_inputs.base_offsets",
                                "upload_group": "kv_span_inputs",
                            },
                            {
                                "dtype": "int64",
                                "name": "decode_kv_write_position",
                                "nbytes": 8,
                                "sha256": "d" * 64,
                                "shape": [],
                                "source": "decode_span_inputs.kv_write_position",
                                "upload_group": "kv_span_inputs",
                            },
                            {
                                "dtype": "int64",
                                "name": "decode_attention_live_counts",
                                "nbytes": 8,
                                "sha256": "e" * 64,
                                "shape": [1],
                                "source": "decode_span_inputs.attention_live_counts",
                                "upload_group": "kv_span_inputs",
                            },
                        ],
                        "entry_count": 6,
                        "upload_order": [
                            "input_ids",
                            "prompt_base_offsets",
                            "prompt_live_counts",
                            "decode_base_offsets",
                            "decode_kv_write_position",
                            "decode_attention_live_counts",
                        ],
                        "cleanup_order": [
                            "decode_attention_live_counts",
                            "decode_kv_write_position",
                            "decode_base_offsets",
                            "prompt_live_counts",
                            "prompt_base_offsets",
                            "input_ids",
                        ],
                        "input_token_nbytes": 92,
                        "span_input_nbytes": 392,
                        "total_nbytes": 484,
                        "consistency_checks": {
                            "cleanup_order_reverses_upload_order": True,
                            "entry_count_matches_upload_order": True,
                            "entry_total_nbytes_matches": True,
                            "input_token_hash_matches": True,
                            "span_payload_hashes_match_manifest": True,
                        },
                        "all_consistency_checks_passed": True,
                        "streaming_runner_ready": False,
                        "note": "Metadata-only combined upload plan; no kernels are launched.",
                    },
                    "stop_token_ids": [1, 2, 128007],
                    "streaming_runner_ready": False,
                    "streaming_runner_blocker_count": 3,
                    "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
                    "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
                    "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
                    "streaming_runner_blockers": _streaming_runner_blockers(),
                },
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
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    assert status["schema_version"] == 1
    assert status["schema_versions"] == {
        "status": 1,
        "readiness_summary": 1,
        "handoff_summary": 1,
        "blocker_work_queue": 1,
        "first_blocker_work_item": 1,
    }
    assert status["status"] == "blocked"
    assert status["blocker_kinds"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert status["blocker_kinds_sha256"] == _stable_json_sha256(
        status["blocker_kinds"]
    )
    assert status["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert status["blocked_gates_sha256"] == _stable_json_sha256(
        status["blocked_gates"]
    )
    source_artifacts = status["source_artifacts"]
    assert source_artifacts["prompt"] == {
        "path": str(prompt),
        "exists": True,
        "size_bytes": len(prompt.read_bytes()),
        "sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
    }
    assert source_artifacts["oracle"] == {
        "path": str(oracle),
        "exists": True,
        "size_bytes": len(oracle.read_bytes()),
        "sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
    }
    assert source_artifacts["text_resource"] == {
        "path": str(resource),
        "exists": True,
        "size_bytes": len(resource.read_bytes()),
        "sha256": hashlib.sha256(resource.read_bytes()).hexdigest(),
    }
    assert source_artifacts["docs"] == {
        "path": str(docs),
        "exists": True,
        "size_bytes": len(docs.read_bytes()),
        "sha256": hashlib.sha256(docs.read_bytes()).hexdigest(),
    }
    assert status["source_artifacts_sha256"] == _stable_json_sha256(source_artifacts)
    assert status["handoff_summary_sha256"] == _stable_json_sha256(
        status["handoff_summary"]
    )
    assert status["readiness_gates_sha256"] == _stable_json_sha256(
        status["readiness_gates"]
    )
    assert status["next_action_commands_sha256"] == _stable_json_sha256(
        status["next_action_commands"]
    )
    assert status["blocker_kinds_sha256"] == _stable_json_sha256(
        status["blocker_kinds"]
    )
    assert status["blocked_gates_sha256"] == _stable_json_sha256(
        status["blocked_gates"]
    )
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
    oracle_progress = status["oracle_progress"]
    assert oracle_progress["source"] == "oracle_artifact"
    assert oracle_progress["status"] == "executed"
    assert oracle_progress["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert oracle_progress["llama_cli"] == "/tmp/llama-cli"
    assert oracle_progress["model"].endswith("Step-3.7-flash-Q3_K_L-00001-of-00003.gguf")
    assert oracle_progress["prompt_length"] == 23
    assert oracle_progress["n_predict"] == 1
    assert oracle_progress["timeout_s"] == 60.0
    assert oracle_progress["diagnostic_logs"] is False
    assert oracle_progress["elapsed_s"] == 62.4
    assert oracle_progress["extra_llama_args"] == ["--device", "none", "--gpu-layers", "0"]
    assert oracle_progress["command_shell"].startswith("/tmp/llama-cli")
    assert oracle_progress["expected_next_token_id"] == 369
    assert oracle_progress["expected_next_token_text"] == " |"
    assert oracle_progress["expected_next_token_logit"] == 19.158626556396484
    assert oracle_progress["expected_top_tokens"][0]["token_id"] == 369
    assert oracle_progress["generated_text_len"] == 0
    assert oracle_progress["stdout_len"] == 0
    assert oracle_progress["stderr_len"] == len("unknown model architecture: 'step35'")
    assert oracle_progress["text_matches_expected_exact"] is False
    assert oracle_progress["text_matches_expected_stripped"] is False
    assert oracle_progress["comparison_policy"]["expected_text_field"] == "expected_next_token_text"
    assert oracle_progress["step35_supported_by_oracle"] is False
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
    assert status["text_resource_artifact"] == str(resource)
    kv_dispatch = status["kv_decode_dispatch_progress"]
    assert kv_dispatch["source"] == "resource_artifact.text_decode_resource_plan.kv_decode_kernel_plan"
    assert kv_dispatch["resource_status"] == "dry_run_plan"
    assert kv_dispatch["backend"] == "hip_gfx1151"
    assert kv_dispatch["model_quant"] == "gguf_step35"
    assert kv_dispatch["kv_storage_dtype"] == "bf16"
    assert kv_dispatch["decode_attention_kind"] == "splitk_gate_f32"
    assert kv_dispatch["max_context"] == 512
    assert kv_dispatch["max_new_tokens"] == 1
    assert kv_dispatch["max_prompt_rows"] == 511
    assert kv_dispatch["attention_block_size"] == 256
    assert kv_dispatch["attention_block_table_len"] == 2
    assert kv_dispatch["attention_capacity_tokens"] == 512
    assert kv_dispatch["decode_span"] == {
        "block_size": 256,
        "block_table_len": 2,
        "capacity_tokens": 512,
        "live_counts_len": 1,
        "max_live_count": 511,
        "shape_compatible": True,
    }
    assert kv_dispatch["prompt_span"] == {
        "base_offsets_len_formula": "rows * 2",
        "block_size": 256,
        "block_table_len_per_row": 2,
        "live_counts_len_formula": "rows",
        "max_prompt_rows": 511,
        "row_positions_required": True,
        "shape_compatible": True,
    }
    assert kv_dispatch["decode_span_shape_compatible"] is True
    assert kv_dispatch["prompt_span_shape_compatible"] is True
    assert kv_dispatch["span_shape_compatible"] is True
    assert kv_dispatch["launch_schedule"]["layer_count"] == 45
    assert kv_dispatch["launch_schedule"]["operation_count"] == 135
    assert kv_dispatch["launch_schedule"]["per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert kv_dispatch["launch_schedule"]["first_layer_ops"] == [
        "layers.0.prompt_kv_write",
        "layers.0.decode_kv_write",
        "layers.0.decode_attention",
    ]
    assert kv_dispatch["launch_schedule"]["last_layer_ops"] == [
        "layers.44.prompt_kv_write",
        "layers.44.decode_kv_write",
        "layers.44.decode_attention",
    ]
    assert kv_dispatch["launch_schedule"]["stages"][0] == {
        "dispatch_key": "prompt_kv_write",
        "layer_count": 45,
        "name": "prompt_prefill_kv_write",
        "ready": True,
        "span_contract": "prompt_span",
    }
    assert kv_dispatch["launch_schedule"]["streaming_runner_ready"] is False
    assert kv_dispatch["run_plan"]["prompt_length"] == 23
    assert kv_dispatch["run_plan"]["input_id_count"] == 23
    assert kv_dispatch["run_plan"]["input_id_preview"] == list(range(100, 108))
    assert kv_dispatch["run_plan"]["input_ids"] == list(range(100, 123))
    assert kv_dispatch["run_plan"]["input_ids_dtype"] == "int32"
    assert kv_dispatch["run_plan"]["input_ids_nbytes"] == 92
    assert kv_dispatch["run_plan"]["input_ids_sha256"] == "f" * 64
    assert kv_dispatch["run_plan"]["rendered_prompt_nchars"] == 123
    assert kv_dispatch["run_plan"]["rendered_prompt_sha256"] == "0" * 64
    assert kv_dispatch["run_plan"]["attention_block_size"] == 256
    assert kv_dispatch["run_plan"]["attention_block_table_len"] == 2
    assert kv_dispatch["run_plan"]["prompt_span_inputs"]["base_offsets_len"] == 46
    assert kv_dispatch["run_plan"]["prompt_span_inputs"]["base_offsets_nbytes"] == 184
    assert kv_dispatch["run_plan"]["prompt_span_inputs"]["live_counts"] == list(range(23))
    assert kv_dispatch["run_plan"]["prompt_span_inputs"]["live_counts_nbytes"] == 184
    assert kv_dispatch["run_plan"]["prompt_span_inputs"]["total_span_input_nbytes"] == 368
    assert kv_dispatch["run_plan"]["decode_span_inputs"]["base_offsets"] == [0, 1]
    assert kv_dispatch["run_plan"]["decode_span_inputs"]["base_offsets_nbytes"] == 8
    assert kv_dispatch["run_plan"]["decode_span_inputs"]["kv_write_position"] == 23
    assert kv_dispatch["run_plan"]["decode_span_inputs"]["attention_live_counts_nbytes"] == 8
    assert kv_dispatch["run_plan"]["decode_span_inputs"]["total_span_input_nbytes"] == 16
    assert kv_dispatch["run_plan"]["span_input_total_nbytes"] == 384
    upload_manifest = kv_dispatch["run_plan"]["span_input_upload_manifest"]
    assert upload_manifest["entry_count"] == 5
    assert upload_manifest["total_nbytes"] == 392
    assert upload_manifest["entries"][0] == {
        "dtype": "int32",
        "kernel_args": ["prompt_kv_write.base_offsets"],
        "name": "prompt_base_offsets",
        "nbytes": 184,
        "shape": [23, 2],
        "source": "prompt_span_inputs.base_offsets",
    }
    assert upload_manifest["entries"][3]["source"] == "decode_span_inputs.kv_write_position"
    host_payloads = kv_dispatch["run_plan"]["span_input_host_payloads"]
    assert host_payloads["entry_count"] == 5
    assert host_payloads["total_nbytes"] == 392
    assert host_payloads["entries"][0]["sha256"] == "a" * 64
    assert host_payloads["entries"][0]["preview_values"] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert host_payloads["entries"][3]["preview_values"] == [23]
    decode_upload_plan = kv_dispatch["run_plan"]["decode_input_upload_plan"]
    assert decode_upload_plan["entry_count"] == 6
    assert decode_upload_plan["upload_order"] == [
        "input_ids",
        "prompt_base_offsets",
        "prompt_live_counts",
        "decode_base_offsets",
        "decode_kv_write_position",
        "decode_attention_live_counts",
    ]
    assert decode_upload_plan["cleanup_order"] == list(reversed(decode_upload_plan["upload_order"]))
    assert decode_upload_plan["input_token_nbytes"] == 92
    assert decode_upload_plan["span_input_nbytes"] == 392
    assert decode_upload_plan["total_nbytes"] == 484
    assert decode_upload_plan["entries"][0]["sha256"] == "f" * 64
    assert kv_dispatch["run_plan"]["prompt_positions"] == list(range(23))
    assert kv_dispatch["run_plan"]["decode_position"] == 23
    assert kv_dispatch["run_plan"]["decode_live_count"] == 23
    assert kv_dispatch["run_plan"]["required_context_tokens"] == 24
    assert kv_dispatch["run_plan"]["prompt_fits_resource_plan"] is True
    assert kv_dispatch["run_plan"]["context_fits_resource_plan"] is True
    assert kv_dispatch["run_plan"]["streaming_runner_ready"] is False
    assert kv_dispatch["run_plan_prompt_fits_resource_plan"] is True
    assert kv_dispatch["run_plan_context_fits_resource_plan"] is True
    assert kv_dispatch["all_registered"] is True
    assert kv_dispatch["registered"] == {
        "decode_attention": True,
        "decode_kv_write": True,
        "prompt_kv_write": True,
    }
    assert kv_dispatch["dispatch_keys"]["prompt_kv_write"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_kv_write",
        "quant": "gguf_step35",
        "variant": "mixed_bf16_prompt_spans",
    }
    assert status["kv_decode_dispatch_ready"] is True
    assert status["kv_backed_decode_ready"] is False
    assert status["e2e_inference_ready"] is False
    gates = status["readiness_gates"]
    assert gates["oracle_parity"]["ready"] is False
    assert gates["oracle_parity"]["blocked_by"] == "llama_cpp_missing_step35_architecture"
    assert gates["oracle_parity"]["expected_next_token_id"] == 369
    assert gates["oracle_parity"]["expected_next_token_text"] == " |"
    assert gates["oracle_parity"]["current_oracle_status"] == "executed"
    assert gates["oracle_parity"]["current_oracle_returncode"] == 1
    oracle_gap = status["oracle_gap_report"]
    assert gates["oracle_parity"]["gap_report"] == oracle_gap
    assert oracle_gap["status"] == "blocked"
    assert oracle_gap["precondition_count"] == 3
    assert oracle_gap["validated_precondition_count"] == 2
    assert oracle_gap["missing_preconditions"] == ["step35_not_rejected"]
    assert oracle_gap["first_missing_precondition"] == "step35_not_rejected"
    assert oracle_gap["missing_evidence"] == [
        "oracle_completed_successfully",
        "oracle_generated_comparable_text",
        "oracle_exact_text_match",
    ]
    assert oracle_gap["first_missing_evidence"] == "oracle_completed_successfully"
    assert oracle_gap["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert oracle_gap["returncode"] == 1
    assert gates["kv_backed_decode"]["ready"] is False
    assert gates["kv_backed_decode"]["blocked_by"] == "kv_backed_decode_not_wired"
    assert gates["kv_backed_decode"]["dispatch_ready"] is True
    assert gates["kv_backed_decode"]["current_evidence"] == {
        "decode_span_shape_compatible": True,
        "dispatch_ready": True,
        "launch_schedule_operation_count": 135,
        "launch_schedule_streaming_ready": False,
        "prompt_span_shape_compatible": True,
        "resident_prompt_smoke": "host_composed_layer_prefix",
        "run_plan_context_fits_resource_plan": True,
        "run_plan_decode_span_base_offsets_len": 2,
        "run_plan_input_id_count": 23,
        "run_plan_input_ids_nbytes": 92,
        "run_plan_input_ids_sha256": "f" * 64,
        "run_plan_prompt_fits_resource_plan": True,
        "run_plan_prompt_span_base_offsets_len": 46,
        "run_plan_rendered_prompt_sha256": "0" * 64,
        "run_plan_span_input_total_nbytes": 384,
        "run_plan_decode_input_upload_checks_passed": True,
        "run_plan_decode_input_upload_entry_count": 6,
        "run_plan_decode_input_upload_total_nbytes": 484,
        "run_plan_host_payload_entry_count": 5,
        "run_plan_host_payload_total_nbytes": 392,
        "run_plan_streaming_ready": False,
        "run_plan_upload_manifest_entry_count": 5,
        "run_plan_upload_manifest_total_nbytes": 392,
    }
    gap_report = status["kv_backed_decode_gap_report"]
    assert gates["kv_backed_decode"]["gap_report"] == gap_report
    assert gap_report["status"] == "blocked"
    assert gap_report["precondition_count"] == 5
    assert gap_report["validated_precondition_count"] == 5
    assert gap_report["missing_preconditions"] == []
    assert gap_report["missing_evidence"] == [
        "streaming_runner_ready_flags",
        "kv_kernel_launch_trace",
        "kv_backed_next_token_artifact",
    ]
    assert gap_report["first_missing_evidence"] == "streaming_runner_ready_flags"
    assert gap_report["missing_evidence_count"] == 3
    assert gap_report["operation_count"] == 135
    assert gap_report["streaming_runner_blocker_count"] == 3
    assert gap_report["streaming_runner_blocker_names"] == _streaming_runner_blocker_names()
    assert gap_report["streaming_runner_blocker_names_sha256"] == _streaming_runner_blocker_names_sha256()
    assert gap_report["computed_streaming_runner_blocker_names_sha256"] == _streaming_runner_blocker_names_sha256()
    assert gap_report["streaming_runner_blocker_names_sha256_match"] is True
    assert gap_report["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert gap_report["streaming_runner_blockers"] == _streaming_runner_blockers()
    assert gap_report["upload_entry_count"] == 6
    assert gap_report["upload_total_nbytes"] == 484
    assert gap_report["remaining_evidence"][0]["current"] == {
        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
        "launch_schedule_streaming_runner_blocker_count": 3,
        "launch_schedule_streaming_runner_ready": False,
        "run_plan_streaming_runner_blocker_count": 3,
        "run_plan_streaming_runner_ready": False,
        "streaming_runner_blocker_count": 3,
        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "computed_streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "streaming_runner_blocker_names_sha256_match": True,
        "streaming_runner_blockers": _streaming_runner_blockers(),
    }
    assert gates["e2e_inference"]["ready"] is False
    assert gates["e2e_inference"]["blocked_by"] == ["oracle_parity", "kv_backed_decode"]
    handoff = status["handoff_summary"]
    assert handoff["schema_version"] == 1
    assert handoff["status"] == "blocked"
    assert handoff["open_or_partial_items_p0_p12"] == 2
    assert handoff["open_blocker_count"] == 2
    assert handoff["open_blockers"] == ["oracle_parity_blocked", "kv_backed_decode_not_wired"]
    assert handoff["blocker_work_queue_schema_version"] == 1
    assert handoff["blocker_work_queue_count"] == 2
    assert handoff["blocker_work_queue"] == [
        {
            "blocker_kind": "oracle_parity_blocked",
            "work_item_schema_version": 1,
            "queue_index": 0,
            "is_first": True,
            "command_available": True,
            **_primary_command_fields(
                "rerun_command_shell",
                "/tmp/llama-cli --model stepfun.gguf --predict 1 --temp 0",
            ),
            **_oracle_helper_fields(prompt, oracle),
            "first_missing_evidence": "oracle_completed_successfully",
            "first_missing_precondition": "step35_not_rejected",
            "gap_report_status": "blocked",
            "current_status": "executed",
            "current_returncode": 1,
            "elapsed_s": 62.4,
            "timeout_s": 60.0,
            "diagnostic_logs": False,
            "gate": "oracle_parity",
            "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
        },
        {
            "blocker_kind": "kv_backed_decode_not_wired",
            "work_item_schema_version": 1,
            "queue_index": 1,
            "is_first": False,
            "command_available": True,
            **_primary_command_fields(
                "resource_plan_refresh_command",
                (
                    "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
                    f"--kv-context-pages 1 --kv-page-size 512 --pretty > {resource}"
                ),
            ),
            "first_missing_evidence": "streaming_runner_ready_flags",
            "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
            "gap_report_status": "blocked",
            "operation_count": 135,
            "streaming_runner_blocker_count": 3,
            "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
            "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
            "streaming_runner_blocker_names_sha256_match": True,
            "gate": "kv_backed_decode",
        },
    ]
    assert handoff["blocker_work_queue_sha256"] == _stable_json_sha256(
        handoff["blocker_work_queue"]
    )
    assert handoff["blocker_work_queue_meta"] == {
        "schema_version": 1,
        "count": 2,
        "sha256": handoff["blocker_work_queue_sha256"],
        "first_blocker_kind": "oracle_parity_blocked",
        "first_work_item_schema_version": 1,
        "first_work_item_sha256": handoff["first_blocker_work_item_sha256"],
    }
    assert handoff["first_blocker_work_item"] == handoff["blocker_work_queue"][0]
    assert handoff["first_blocker_work_item_sha256"] == _stable_json_sha256(
        handoff["first_blocker_work_item"]
    )
    assert handoff["exit_codes"] == {
        "ready": 0,
        "source_artifact_mismatch": 1,
        "blocked_when_fail_on_blocked": 2,
        "current_with_fail_on_blocked": 2,
    }
    assert handoff["compact_output_modes"] == {
        "summary_only": "handoff_summary",
        "handoff_summary_sha_only": "handoff_summary_sha256",
        "schema_versions_only": "schema_versions",
        "status_integrity_only": "status_integrity",
        "status_integrity_failures_only": "status_integrity.failed_checks",
        "readiness_summary_only": "readiness_summary",
        "readiness_summary_sha_only": "readiness_summary_sha256",
        "readiness_gates_only": "readiness_gates",
        "readiness_gates_sha_only": "readiness_gates_sha256",
        "blocked_gates_only": "blocked_gates",
        "blocked_gates_sha_only": "blocked_gates_sha256",
        "source_artifacts_sha_only": "source_artifacts_sha256",
        "next_action_commands_sha_only": "next_action_commands_sha256",
        "blocker_kinds_only": "blocker_kinds",
        "blocker_kinds_sha_only": "blocker_kinds_sha256",
        "kv_streaming_blockers_only": "kv_streaming_runner_blocker_names",
        "kv_streaming_blockers_sha_only": "kv_streaming_runner_blocker_names_sha256",
        "status_refresh_command_only": (
            "next_action_commands.oracle_parity_blocked.status_refresh_command"
        ),
        "status_refresh_command_sha_only": (
            "next_action_commands.oracle_parity_blocked.status_refresh_command_sha256"
        ),
        "source_verify_command_only": (
            "next_action_commands.handoff_integrity.source_artifacts_verify_command"
        ),
        "source_verify_command_sha_only": (
            "next_action_commands.handoff_integrity.source_artifacts_verify_command_sha256"
        ),
        "kv_resource_command_only": (
            "next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command"
        ),
        "kv_resource_command_sha_only": (
            "next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command_sha256"
        ),
        "oracle_helper_command_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command"
        ),
        "oracle_helper_command_sha_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command_sha256"
        ),
        "blocker_work_queue_only": "handoff_summary.blocker_work_queue",
        "blocker_work_queue_meta_only": "handoff_summary.blocker_work_queue_meta",
        "blocker_work_queue_sha_only": "handoff_summary.blocker_work_queue_sha256",
        "first_blocker_sha_only": "handoff_summary.first_blocker_work_item_sha256",
        "first_blocker_only": "handoff_summary.first_blocker_work_item",
        "fail_on_blocked_preserves_payload": True,
    }
    assert handoff["ready_gates"] == []
    assert handoff["blocked_gates"] == ["oracle_parity", "kv_backed_decode", "e2e_inference"]
    assert handoff["ready_signals"] == {
        "all_layer_prompt_smoke": True,
        "kv_decode_dispatch_ready": True,
        "kv_decode_input_upload_plan_recorded": True,
        "kv_decode_run_plan_recorded": True,
        "kv_launch_schedule_recorded": True,
        "oracle_target_recorded": True,
    }
    assert handoff["oracle_gap_report"] == {
        "elapsed_s": 62.4,
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "first_missing_evidence": "oracle_completed_successfully",
        "first_missing_precondition": "step35_not_rejected",
        "missing_evidence": [
            "oracle_completed_successfully",
            "oracle_generated_comparable_text",
            "oracle_exact_text_match",
        ],
        "missing_evidence_count": 3,
        "missing_precondition_count": 1,
        "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
        "oracle_status": "executed",
        "precondition_count": 3,
        "status": "blocked",
        "timeout_s": 60.0,
        "validated_precondition_count": 2,
    }
    assert handoff["kv_decode_input_upload_plan"] == {
        "entry_count": 6,
        "total_nbytes": 484,
        "upload_order": [
            "input_ids",
            "prompt_base_offsets",
            "prompt_live_counts",
            "decode_base_offsets",
            "decode_kv_write_position",
            "decode_attention_live_counts",
        ],
        "cleanup_order": [
            "decode_attention_live_counts",
            "decode_kv_write_position",
            "decode_base_offsets",
            "prompt_live_counts",
            "prompt_base_offsets",
            "input_ids",
        ],
        "all_consistency_checks_passed": True,
    }
    assert handoff["kv_backed_decode_gap_report"] == {
        "first_missing_evidence": "streaming_runner_ready_flags",
        "missing_evidence": [
            "streaming_runner_ready_flags",
            "kv_kernel_launch_trace",
            "kv_backed_next_token_artifact",
        ],
        "missing_evidence_count": 3,
        "missing_precondition_count": 0,
        "operation_count": 135,
        "precondition_count": 5,
        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
        "status": "blocked",
        "streaming_runner_blocker_count": 3,
        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "streaming_runner_blocker_names_sha256_match": True,
        "upload_total_nbytes": 484,
        "validated_precondition_count": 5,
    }
    assert handoff["blocked_signals"] == {
        "e2e_inference": True,
        "kv_backed_decode": True,
        "oracle_parity": True,
    }
    assert handoff["next_commands_available_for"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert handoff["no_claim_policy"]["performance_claim_allowed"] is False
    assert handoff["no_claim_policy"]["e2e_inference_claim_allowed"] is False
    assert {blocker["kind"] for blocker in status["blockers"]} == {
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    }
    oracle_blocker = next(blocker for blocker in status["blockers"] if blocker["kind"] == "oracle_parity_blocked")
    assert oracle_blocker["expected_next_token_id"] == 369
    assert oracle_blocker["expected_next_token_text"] == " |"
    assert oracle_blocker["elapsed_s"] == 62.4
    assert oracle_blocker["timeout_s"] == 60.0
    assert oracle_blocker["gap_report_status"] == "blocked"
    assert oracle_blocker["first_missing_precondition"] == "step35_not_rejected"
    assert oracle_blocker["missing_evidence"] == [
        "oracle_completed_successfully",
        "oracle_generated_comparable_text",
        "oracle_exact_text_match",
    ]
    assert oracle_blocker["first_missing_evidence"] == "oracle_completed_successfully"
    kv_blocker = next(blocker for blocker in status["blockers"] if blocker["kind"] == "kv_backed_decode_not_wired")
    assert kv_blocker["resource_artifact"] == str(resource)
    assert kv_blocker["kv_decode_dispatch_ready"] is True
    assert kv_blocker["gap_report_status"] == "blocked"
    assert kv_blocker["missing_evidence"] == [
        "streaming_runner_ready_flags",
        "kv_kernel_launch_trace",
        "kv_backed_next_token_artifact",
    ]
    assert kv_blocker["streaming_runner_blocker_count"] == 3
    assert kv_blocker["streaming_runner_blocker_names"] == _streaming_runner_blocker_names()
    assert kv_blocker["streaming_runner_blocker_names_sha256"] == _streaming_runner_blocker_names_sha256()
    assert kv_blocker["streaming_runner_blocker_names_sha256_match"] is True
    assert kv_blocker["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert {action["blocker_kind"] for action in status["next_actions"]} == {
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    }
    assert status["handoff_summary_sha256"] == _stable_json_sha256(handoff)
    commands = status["next_action_commands"]
    assert status["next_action_commands_sha256"] == _stable_json_sha256(commands)
    integrity_commands = commands["handoff_integrity"]
    assert integrity_commands["source_artifacts_verify_command"] == _source_verify_command()
    assert integrity_commands["source_artifacts_verify_command_nchars"] == len(
        integrity_commands["source_artifacts_verify_command"]
    )
    assert integrity_commands["source_artifacts_verify_command_sha256"] == hashlib.sha256(
        integrity_commands["source_artifacts_verify_command"].encode()
    ).hexdigest()
    assert integrity_commands["success_criteria"] == [
        "source artifact verification exits 0",
        "source artifact verification reports status=match",
        "source artifact verification reports all_match=true",
    ]
    oracle_commands = commands["oracle_parity_blocked"]
    assert oracle_commands["rerun_command_shell"].startswith("/tmp/llama-cli")
    assert oracle_commands["oracle_helper_refresh_command"] == _oracle_helper_command(
        prompt, oracle
    )
    assert oracle_commands["oracle_helper_refresh_command_nchars"] == len(
        oracle_commands["oracle_helper_refresh_command"]
    )
    assert oracle_commands["oracle_helper_refresh_command_sha256"] == hashlib.sha256(
        oracle_commands["oracle_helper_refresh_command"].encode()
    ).hexdigest()
    assert f"--prompt-artifact {prompt}" in oracle_commands["status_refresh_command"]
    assert f"--oracle-artifact {oracle}" in oracle_commands["status_refresh_command"]
    assert oracle_commands["status_refresh_command_nchars"] == len(
        oracle_commands["status_refresh_command"]
    )
    assert oracle_commands["status_refresh_command_sha256"] == hashlib.sha256(
        oracle_commands["status_refresh_command"].encode()
    ).hexdigest()
    assert oracle_commands["gap_report_status"] == "blocked"
    assert oracle_commands["first_missing_precondition"] == "step35_not_rejected"
    assert oracle_commands["missing_evidence"] == [
        "oracle_completed_successfully",
        "oracle_generated_comparable_text",
        "oracle_exact_text_match",
    ]
    assert oracle_commands["first_missing_evidence"] == "oracle_completed_successfully"
    assert oracle_commands["success_criteria"] == [
        "oracle_gap_report.status is ready",
        "oracle_gap_report.missing_preconditions is empty",
        "oracle_gap_report.missing_evidence is empty",
        "oracle_parity is true",
        "readiness_gates.oracle_parity.ready is true",
    ]
    kv_commands = commands["kv_backed_decode_not_wired"]
    assert kv_commands["resource_plan_refresh_command"] == (
        "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
        f"--kv-context-pages 1 --kv-page-size 512 --pretty > {resource}"
    )
    assert kv_commands["resource_plan_refresh_command_nchars"] == len(
        kv_commands["resource_plan_refresh_command"]
    )
    assert kv_commands["resource_plan_refresh_command_sha256"] == hashlib.sha256(
        kv_commands["resource_plan_refresh_command"].encode()
    ).hexdigest()
    assert kv_commands["status_refresh_command"] == oracle_commands["status_refresh_command"]
    assert kv_commands["status_refresh_command_nchars"] == oracle_commands[
        "status_refresh_command_nchars"
    ]
    assert kv_commands["status_refresh_command_sha256"] == oracle_commands[
        "status_refresh_command_sha256"
    ]
    assert kv_commands["gap_report_status"] == "blocked"
    assert kv_commands["missing_evidence"] == [
        "streaming_runner_ready_flags",
        "kv_kernel_launch_trace",
        "kv_backed_next_token_artifact",
    ]
    assert kv_commands["first_missing_evidence"] == "streaming_runner_ready_flags"
    assert kv_commands["streaming_runner_blocker_count"] == 3
    assert kv_commands["streaming_runner_blocker_names"] == _streaming_runner_blocker_names()
    assert kv_commands["streaming_runner_blocker_names_sha256"] == _streaming_runner_blocker_names_sha256()
    assert kv_commands["streaming_runner_blocker_names_sha256_match"] is True
    assert kv_commands["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert kv_commands["success_criteria"] == [
        "kv_backed_decode_gap_report.status is ready",
        "kv_backed_decode_gap_report.missing_evidence is empty",
        "kv_backed_decode_ready is true",
        "readiness_gates.kv_backed_decode.ready is true",
        "e2e_inference_ready is true only after oracle_parity is also true",
    ]
    assert status["docs_checklist"]["open_or_partial_count_p0_p12"] == 2
    assert [item["state"] for item in status["docs_checklist"]["open_or_partial_items_p0_p12"]] == [
        "open",
        "partial",
    ]


def test_stepfun_correctness_status_drops_kv_blocker_when_gap_report_is_ready(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    resource_payload = json.loads(resource.read_text())
    schedule = resource_payload["text_decode_resource_plan"]["kv_decode_launch_schedule"]
    schedule["streaming_runner_ready"] = True
    run_plan = resource_payload["kv_decode_run_plan"]
    run_plan["streaming_runner_ready"] = True
    run_plan["kv_kernel_trace_artifact"] = "benchmarks/results/test-kv-kernel-trace.json"
    run_plan["kv_backed_next_token_artifact"] = "benchmarks/results/test-kv-next-token.json"
    resource.write_text(json.dumps(resource_payload))

    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    assert status["kv_backed_decode_ready"] is True
    assert status["e2e_inference_ready"] is False
    assert status["kv_backed_decode_gap_report"]["status"] == "ready"
    assert status["kv_backed_decode_gap_report"]["missing_evidence"] == []
    assert status["kv_backed_decode_gap_report"]["first_missing_evidence"] is None
    assert status["readiness_gates"]["kv_backed_decode"]["ready"] is True
    assert status["handoff_summary"]["ready_gates"] == ["kv_backed_decode"]
    assert status["handoff_summary"]["blocked_gates"] == ["oracle_parity", "e2e_inference"]
    assert status["handoff_summary"]["blocked_signals"]["kv_backed_decode"] is False
    assert [blocker["kind"] for blocker in status["blockers"]] == ["oracle_parity_blocked"]
    assert [action["blocker_kind"] for action in status["next_actions"]] == [
        "oracle_parity_blocked"
    ]



def test_stepfun_correctness_status_drops_oracle_blocker_when_gap_report_is_ready(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    oracle_payload = json.loads(oracle.read_text())
    oracle_payload.update(
        {
            "status": "executed",
            "returncode": 0,
            "step35_supported": True,
            "oracle_blocker_kind": None,
            "oracle_blocker_detail": None,
            "stdout": " |",
            "generated_text": " |",
            "text_matches_expected_exact": True,
            "text_matches_expected_stripped": True,
        }
    )
    oracle.write_text(json.dumps(oracle_payload))

    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    assert status["oracle_parity"] is True
    assert status["e2e_inference_ready"] is False
    assert status["oracle_gap_report"]["status"] == "ready"
    assert status["oracle_gap_report"]["missing_preconditions"] == []
    assert status["oracle_gap_report"]["missing_evidence"] == []
    assert status["oracle_gap_report"]["first_missing_evidence"] is None
    assert status["readiness_gates"]["oracle_parity"]["ready"] is True
    assert status["handoff_summary"]["ready_gates"] == ["oracle_parity"]
    assert status["handoff_summary"]["blocked_gates"] == ["kv_backed_decode", "e2e_inference"]
    assert status["handoff_summary"]["blocked_signals"]["oracle_parity"] is False
    assert [blocker["kind"] for blocker in status["blockers"]] == [
        "kv_backed_decode_not_wired"
    ]
    assert [action["blocker_kind"] for action in status["next_actions"]] == [
        "kv_backed_decode_not_wired"
    ]



def test_stepfun_correctness_status_writes_json(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "status.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
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
    assert payload["schema_version"] == 1
    assert payload["blocker_kinds"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["blocker_kinds_sha256"] == _stable_json_sha256(
        payload["blocker_kinds"]
    )
    assert payload["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert payload["blocked_gates_sha256"] == _stable_json_sha256(
        payload["blocked_gates"]
    )
    assert payload["schema_versions"] == {
        "status": 1,
        "readiness_summary": 1,
        "handoff_summary": 1,
        "blocker_work_queue": 1,
        "first_blocker_work_item": 1,
    }
    assert payload["status"] == "blocked"
    assert payload["source_artifacts"]["prompt"]["sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert payload["source_artifacts"]["oracle"]["size_bytes"] == len(oracle.read_bytes())
    assert payload["source_artifacts"]["text_resource"]["exists"] is True
    assert payload["source_artifacts"]["docs"]["path"] == str(docs)
    assert payload["source_artifacts_sha256"] == _stable_json_sha256(
        payload["source_artifacts"]
    )
    assert payload["handoff_summary_sha256"] == _stable_json_sha256(
        payload["handoff_summary"]
    )
    assert payload["readiness_gates_sha256"] == _stable_json_sha256(
        payload["readiness_gates"]
    )
    assert payload["next_action_commands_sha256"] == _stable_json_sha256(
        payload["next_action_commands"]
    )
    assert payload["blocker_kinds_sha256"] == _stable_json_sha256(
        payload["blocker_kinds"]
    )
    assert payload["blocked_gates_sha256"] == _stable_json_sha256(
        payload["blocked_gates"]
    )
    assert payload["all_layer_prompt_smoke"] is True
    assert payload["e2e_inference_ready"] is False
    assert payload["readiness_summary_sha256"] == _stable_json_sha256(
        payload["readiness_summary"]
    )
    assert payload["readiness_summary"] == {
        "schema_version": 1,
        "status": "blocked",
        "oracle_parity": False,
        "kv_decode_dispatch_ready": True,
        "kv_backed_decode_ready": False,
        "e2e_inference_ready": False,
        "open_or_partial_items_p0_p12": 2,
        "open_blocker_count": 2,
        "handoff_summary_sha256": payload["handoff_summary_sha256"],
        "source_artifacts_sha256": payload["source_artifacts_sha256"],
        "readiness_gates_sha256": payload["readiness_gates_sha256"],
        "next_action_commands_sha256": payload["next_action_commands_sha256"],
        "blocker_kinds_sha256": payload["blocker_kinds_sha256"],
        "blocked_gates_sha256": payload["blocked_gates_sha256"],
        "first_blocker_kind": "oracle_parity_blocked",
        "first_blocker_work_item_sha256": payload["handoff_summary"][
            "first_blocker_work_item_sha256"
        ],
        "blocker_work_queue_count": 2,
        "blocker_work_queue_sha256": payload["handoff_summary"][
            "blocker_work_queue_sha256"
        ],
        "fail_on_blocked_exit_code": 2,
        "performance_claim_allowed": False,
        "e2e_inference_claim_allowed": False,
    }
    assert payload["oracle_progress"]["expected_next_token_id"] == 369
    assert payload["oracle_progress"]["returncode"] == 1
    assert payload["oracle_progress"]["timeout_s"] == 60.0
    assert payload["oracle_gap_report"]["first_missing_precondition"] == "step35_not_rejected"
    assert payload["oracle_gap_report"]["first_missing_evidence"] == "oracle_completed_successfully"
    assert payload["readiness_gates"]["oracle_parity"]["gap_report"] == payload[
        "oracle_gap_report"
    ]
    assert payload["linear_projection_progress"]["resident_linear_projection_slot_count"] == 487
    assert payload["kv_decode_dispatch_ready"] is True
    assert payload["readiness_gates"]["oracle_parity"]["ready"] is False
    assert payload["readiness_gates"]["kv_backed_decode"]["dispatch_ready"] is True
    assert payload["kv_backed_decode_gap_report"]["missing_evidence"] == [
        "streaming_runner_ready_flags",
        "kv_kernel_launch_trace",
        "kv_backed_next_token_artifact",
    ]
    assert payload["readiness_gates"]["kv_backed_decode"]["gap_report"] == payload[
        "kv_backed_decode_gap_report"
    ]
    assert payload["readiness_gates"]["e2e_inference"]["blocked_by"] == [
        "oracle_parity",
        "kv_backed_decode",
    ]
    assert payload["handoff_summary"]["open_blockers"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["handoff_summary"]["blocker_work_queue_schema_version"] == 1
    assert payload["handoff_summary"]["blocker_work_queue_count"] == 2
    assert payload["handoff_summary"]["blocker_work_queue_sha256"] == _stable_json_sha256(
        payload["handoff_summary"]["blocker_work_queue"]
    )
    assert payload["handoff_summary"]["blocker_work_queue_meta"] == {
        "schema_version": 1,
        "count": 2,
        "sha256": payload["handoff_summary"]["blocker_work_queue_sha256"],
        "first_blocker_kind": "oracle_parity_blocked",
        "first_work_item_schema_version": 1,
        "first_work_item_sha256": payload["handoff_summary"]["first_blocker_work_item_sha256"],
    }
    assert payload["handoff_summary"]["first_blocker_work_item_sha256"] == _stable_json_sha256(
        payload["handoff_summary"]["first_blocker_work_item"]
    )
    assert payload["handoff_summary"]["first_blocker_work_item"]["blocker_kind"] == (
        "oracle_parity_blocked"
    )
    assert payload["handoff_summary"]["blocker_work_queue"][1][
        "first_streaming_runner_blocker"
    ] == "streaming_decode_loop_not_wired"
    assert payload["handoff_summary"]["exit_codes"]["current_with_fail_on_blocked"] == 2
    assert payload["handoff_summary"]["compact_output_modes"]["first_blocker_only"] == (
        "handoff_summary.first_blocker_work_item"
    )
    assert payload["handoff_summary"]["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert payload["handoff_summary"]["no_claim_policy"]["performance_claim_allowed"] is False
    integrity_command = payload["next_action_commands"]["handoff_integrity"]
    assert integrity_command["source_artifacts_verify_command"] == _source_verify_command()
    assert integrity_command["source_artifacts_verify_command_sha256"] == hashlib.sha256(
        _source_verify_command().encode()
    ).hexdigest()
    oracle_command = payload["next_action_commands"]["oracle_parity_blocked"]
    assert oracle_command["rerun_command_shell"].startswith("/tmp/llama-cli")
    assert oracle_command["first_missing_precondition"] == "step35_not_rejected"
    assert oracle_command["first_missing_evidence"] == "oracle_completed_successfully"
    assert oracle_command["success_criteria"][0] == "oracle_gap_report.status is ready"
    kv_command = payload["next_action_commands"]["kv_backed_decode_not_wired"]
    assert kv_command["resource_plan_refresh_command"].endswith(f"> {resource}")
    assert kv_command["first_missing_evidence"] == "streaming_runner_ready_flags"
    assert kv_command["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert kv_command["streaming_runner_blocker_count"] == 3
    assert kv_command["success_criteria"][0] == "kv_backed_decode_gap_report.status is ready"
    assert len(payload["next_actions"]) == 2
    assert payload["docs_checklist"]["open_or_partial_count_p0_p12"] == 2


def test_stepfun_correctness_status_readiness_summary_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "readiness-summary.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["readiness_summary"]
    assert payload["status"] == "blocked"
    assert payload["oracle_parity"] is False
    assert payload["kv_decode_dispatch_ready"] is True
    assert payload["kv_backed_decode_ready"] is False
    assert payload["first_blocker_kind"] == "oracle_parity_blocked"
    assert payload["blocker_work_queue_sha256"] == status["handoff_summary"][
        "blocker_work_queue_sha256"
    ]
    assert payload["handoff_summary_sha256"] == status["handoff_summary_sha256"]
    assert payload["source_artifacts_sha256"] == status["source_artifacts_sha256"]
    assert payload["readiness_gates_sha256"] == status["readiness_gates_sha256"]
    assert payload["next_action_commands_sha256"] == status[
        "next_action_commands_sha256"
    ]
    assert payload["blocker_kinds_sha256"] == status["blocker_kinds_sha256"]
    assert payload["blocked_gates_sha256"] == status["blocked_gates_sha256"]
    assert payload["fail_on_blocked_exit_code"] == 2


def test_stepfun_correctness_status_readiness_summary_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "readiness-summary-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["readiness_summary_sha256"]
    assert payload == _stable_json_sha256(status["readiness_summary"])


def test_stepfun_correctness_status_readiness_gates_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "readiness-gates-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--readiness-gates-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["readiness_gates_sha256"]
    assert payload == _stable_json_sha256(status["readiness_gates"])


def test_stepfun_correctness_status_readiness_gates_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "readiness-gates.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--readiness-gates-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["readiness_gates"]
    assert payload["oracle_parity"]["ready"] is False
    assert payload["kv_backed_decode"]["ready"] is False
    assert payload["kv_backed_decode"]["dispatch_ready"] is True
    assert payload["e2e_inference"]["ready"] is False


def test_stepfun_correctness_status_blocked_gates_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocked-gates-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--blocked-gates-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["blocked_gates_sha256"]
    assert payload == _stable_json_sha256(status["blocked_gates"])


def test_stepfun_correctness_status_blocked_gates_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocked-gates.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--blocked-gates-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["blocked_gates"]
    assert payload == ["oracle_parity", "kv_backed_decode", "e2e_inference"]


def test_stepfun_correctness_status_kv_streaming_blockers_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-blockers-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--kv-streaming-blockers-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_runner_blocker_names_sha256"
    ]
    assert payload == _streaming_runner_blocker_names_sha256()


def test_stepfun_correctness_status_kv_streaming_blockers_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-blockers.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--kv-streaming-blockers-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"]["streaming_runner_blocker_names"]
    assert payload == _streaming_runner_blocker_names()


def test_stepfun_correctness_status_blocker_kinds_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-kinds-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--blocker-kinds-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["blocker_kinds_sha256"]
    assert payload == _stable_json_sha256(status["blocker_kinds"])


def test_stepfun_correctness_status_blocker_kinds_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-kinds.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--blocker-kinds-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["blocker_kinds"]
    assert payload == ["oracle_parity_blocked", "kv_backed_decode_not_wired"]


def test_stepfun_correctness_status_handoff_summary_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "handoff-summary-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--handoff-summary-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["handoff_summary_sha256"]
    assert payload == _stable_json_sha256(status["handoff_summary"])


def test_stepfun_correctness_status_schema_versions_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "schema-versions.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--handoff-summary-sha-only",
            "--schema-versions-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["schema_versions"]
    assert payload == {
        "status": 1,
        "readiness_summary": 1,
        "handoff_summary": 1,
        "blocker_work_queue": 1,
        "first_blocker_work_item": 1,
    }


def test_stepfun_correctness_status_status_integrity_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "status-integrity.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--handoff-summary-sha-only",
            "--schema-versions-only",
            "--status-integrity-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == _status_integrity(status)
    assert payload == {
        "all_match": True,
        "failed_checks": [],
        "checks": {
            "source_artifacts_sha256": True,
            "handoff_summary_sha256": True,
            "readiness_summary_sha256": True,
            "readiness_gates_sha256": True,
            "next_action_commands_sha256": True,
            "blocker_kinds_sha256": True,
            "blocked_gates_sha256": True,
            "kv_streaming_runner_blocker_names_sha256": True,
            "kv_streaming_runner_blocker_mirrors": True,
            "schema_versions": True,
        },
    }


def test_stepfun_correctness_status_status_integrity_failures_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "status-integrity-failures.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--status-integrity-only",
            "--status-integrity-failures-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == _status_integrity(status)["failed_checks"]
    assert payload == []


def test_stepfun_correctness_status_source_artifacts_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "source-artifacts-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--source-artifacts-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["source_artifacts_sha256"]
    assert payload == _stable_json_sha256(status["source_artifacts"])



def test_stepfun_correctness_status_next_action_commands_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "next-action-commands-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--readiness-summary-sha-only",
            "--source-artifacts-sha-only",
            "--next-action-commands-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands_sha256"]
    assert payload == _stable_json_sha256(status["next_action_commands"])



def test_stepfun_correctness_status_summary_only_writes_handoff(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "summary.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["status"] == "blocked"
    assert payload["open_or_partial_items_p0_p12"] == 2
    assert payload["open_blockers"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["blocker_work_queue_schema_version"] == 1
    assert payload["blocker_work_queue_count"] == 2
    assert payload["blocker_work_queue"] == [
        {
            "blocker_kind": "oracle_parity_blocked",
            "work_item_schema_version": 1,
            "queue_index": 0,
            "is_first": True,
            "command_available": True,
            **_primary_command_fields(
                "rerun_command_shell",
                "/tmp/llama-cli --model stepfun.gguf --predict 1 --temp 0",
            ),
            **_oracle_helper_fields(prompt, oracle),
            "first_missing_evidence": "oracle_completed_successfully",
            "first_missing_precondition": "step35_not_rejected",
            "gap_report_status": "blocked",
            "current_status": "executed",
            "current_returncode": 1,
            "elapsed_s": 62.4,
            "timeout_s": 60.0,
            "diagnostic_logs": False,
            "gate": "oracle_parity",
            "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
        },
        {
            "blocker_kind": "kv_backed_decode_not_wired",
            "work_item_schema_version": 1,
            "queue_index": 1,
            "is_first": False,
            "command_available": True,
            **_primary_command_fields(
                "resource_plan_refresh_command",
                (
                    "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
                    f"--kv-context-pages 1 --kv-page-size 512 --pretty > {resource}"
                ),
            ),
            "first_missing_evidence": "streaming_runner_ready_flags",
            "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
            "gap_report_status": "blocked",
            "operation_count": 135,
            "streaming_runner_blocker_count": 3,
            "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
            "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
            "streaming_runner_blocker_names_sha256_match": True,
            "gate": "kv_backed_decode",
        },
    ]
    assert payload["blocker_work_queue_sha256"] == _stable_json_sha256(
        payload["blocker_work_queue"]
    )
    assert payload["blocker_work_queue_meta"] == {
        "schema_version": 1,
        "count": 2,
        "sha256": payload["blocker_work_queue_sha256"],
        "first_blocker_kind": "oracle_parity_blocked",
        "first_work_item_schema_version": 1,
        "first_work_item_sha256": payload["first_blocker_work_item_sha256"],
    }
    assert payload["first_blocker_work_item"] == payload["blocker_work_queue"][0]
    assert payload["first_blocker_work_item_sha256"] == _stable_json_sha256(
        payload["first_blocker_work_item"]
    )
    assert payload["exit_codes"] == {
        "ready": 0,
        "source_artifact_mismatch": 1,
        "blocked_when_fail_on_blocked": 2,
        "current_with_fail_on_blocked": 2,
    }
    assert payload["compact_output_modes"] == {
        "summary_only": "handoff_summary",
        "handoff_summary_sha_only": "handoff_summary_sha256",
        "schema_versions_only": "schema_versions",
        "status_integrity_only": "status_integrity",
        "status_integrity_failures_only": "status_integrity.failed_checks",
        "readiness_summary_only": "readiness_summary",
        "readiness_summary_sha_only": "readiness_summary_sha256",
        "readiness_gates_only": "readiness_gates",
        "readiness_gates_sha_only": "readiness_gates_sha256",
        "blocked_gates_only": "blocked_gates",
        "blocked_gates_sha_only": "blocked_gates_sha256",
        "source_artifacts_sha_only": "source_artifacts_sha256",
        "next_action_commands_sha_only": "next_action_commands_sha256",
        "blocker_kinds_only": "blocker_kinds",
        "blocker_kinds_sha_only": "blocker_kinds_sha256",
        "kv_streaming_blockers_only": "kv_streaming_runner_blocker_names",
        "kv_streaming_blockers_sha_only": "kv_streaming_runner_blocker_names_sha256",
        "status_refresh_command_only": (
            "next_action_commands.oracle_parity_blocked.status_refresh_command"
        ),
        "status_refresh_command_sha_only": (
            "next_action_commands.oracle_parity_blocked.status_refresh_command_sha256"
        ),
        "source_verify_command_only": (
            "next_action_commands.handoff_integrity.source_artifacts_verify_command"
        ),
        "source_verify_command_sha_only": (
            "next_action_commands.handoff_integrity.source_artifacts_verify_command_sha256"
        ),
        "kv_resource_command_only": (
            "next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command"
        ),
        "kv_resource_command_sha_only": (
            "next_action_commands.kv_backed_decode_not_wired.resource_plan_refresh_command_sha256"
        ),
        "oracle_helper_command_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command"
        ),
        "oracle_helper_command_sha_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_refresh_command_sha256"
        ),
        "blocker_work_queue_only": "handoff_summary.blocker_work_queue",
        "blocker_work_queue_meta_only": "handoff_summary.blocker_work_queue_meta",
        "blocker_work_queue_sha_only": "handoff_summary.blocker_work_queue_sha256",
        "first_blocker_sha_only": "handoff_summary.first_blocker_work_item_sha256",
        "first_blocker_only": "handoff_summary.first_blocker_work_item",
        "fail_on_blocked_preserves_payload": True,
    }
    assert payload["ready_signals"]["kv_decode_dispatch_ready"] is True
    assert payload["ready_signals"]["kv_decode_run_plan_recorded"] is True
    assert payload["ready_signals"]["kv_decode_input_upload_plan_recorded"] is True
    assert payload["kv_decode_input_upload_plan"]["entry_count"] == 6
    assert payload["oracle_gap_report"] == {
        "elapsed_s": 62.4,
        "expected_next_token_id": 369,
        "expected_next_token_text": " |",
        "first_missing_evidence": "oracle_completed_successfully",
        "first_missing_precondition": "step35_not_rejected",
        "missing_evidence": [
            "oracle_completed_successfully",
            "oracle_generated_comparable_text",
            "oracle_exact_text_match",
        ],
        "missing_evidence_count": 3,
        "missing_precondition_count": 1,
        "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
        "oracle_status": "executed",
        "precondition_count": 3,
        "status": "blocked",
        "timeout_s": 60.0,
        "validated_precondition_count": 2,
    }
    assert payload["kv_backed_decode_gap_report"] == {
        "first_missing_evidence": "streaming_runner_ready_flags",
        "missing_evidence": [
            "streaming_runner_ready_flags",
            "kv_kernel_launch_trace",
            "kv_backed_next_token_artifact",
        ],
        "missing_evidence_count": 3,
        "missing_precondition_count": 0,
        "operation_count": 135,
        "precondition_count": 5,
        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
        "status": "blocked",
        "streaming_runner_blocker_count": 3,
        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "streaming_runner_blocker_names_sha256_match": True,
        "upload_total_nbytes": 484,
        "validated_precondition_count": 5,
    }
    assert payload["kv_decode_input_upload_plan"]["upload_order"][0] == "input_ids"
    assert payload["kv_decode_input_upload_plan"]["cleanup_order"][-1] == "input_ids"
    assert payload["kv_decode_input_upload_plan"]["all_consistency_checks_passed"] is True
    assert payload["blocked_gates"] == ["oracle_parity", "kv_backed_decode", "e2e_inference"]
    assert payload["next_commands_available_for"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["no_claim_policy"]["e2e_inference_claim_allowed"] is False
    assert "blockers" not in payload
    assert "docs_checklist" not in payload


def test_stepfun_correctness_status_blocker_work_queue_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-work-queue.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == [
        {
            "blocker_kind": "oracle_parity_blocked",
            "work_item_schema_version": 1,
            "queue_index": 0,
            "is_first": True,
            "command_available": True,
            **_primary_command_fields(
                "rerun_command_shell",
                "/tmp/llama-cli --model stepfun.gguf --predict 1 --temp 0",
            ),
            **_oracle_helper_fields(prompt, oracle),
            "first_missing_evidence": "oracle_completed_successfully",
            "first_missing_precondition": "step35_not_rejected",
            "gap_report_status": "blocked",
            "current_status": "executed",
            "current_returncode": 1,
            "elapsed_s": 62.4,
            "timeout_s": 60.0,
            "diagnostic_logs": False,
            "gate": "oracle_parity",
            "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
        },
        {
            "blocker_kind": "kv_backed_decode_not_wired",
            "work_item_schema_version": 1,
            "queue_index": 1,
            "is_first": False,
            "command_available": True,
            **_primary_command_fields(
                "resource_plan_refresh_command",
                (
                    "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
                    f"--kv-context-pages 1 --kv-page-size 512 --pretty > {resource}"
                ),
            ),
            "first_missing_evidence": "streaming_runner_ready_flags",
            "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
            "gap_report_status": "blocked",
            "operation_count": 135,
            "streaming_runner_blocker_count": 3,
            "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
            "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
            "streaming_runner_blocker_names_sha256_match": True,
            "gate": "kv_backed_decode",
        },
    ]


def test_stepfun_correctness_status_blocker_work_queue_meta_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-work-queue-meta.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--blocker-work-queue-meta-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == status["handoff_summary"][
        "blocker_work_queue_meta"
    ]


def test_stepfun_correctness_status_blocker_work_queue_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-work-queue-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--blocker-work-queue-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == status["handoff_summary"][
        "blocker_work_queue_sha256"
    ]


def test_stepfun_correctness_status_first_blocker_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "first-blocker-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--blocker-work-queue-meta-only",
            "--blocker-work-queue-sha-only",
            "--first-blocker-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == status["handoff_summary"][
        "first_blocker_work_item_sha256"
    ]


def test_stepfun_correctness_status_kv_resource_command_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-resource-command-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--kv-resource-command-only",
            "--kv-resource-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    command = status["next_action_commands"]["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command"
    ]
    assert payload == status["next_action_commands"]["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command_sha256"
    ]
    assert payload == hashlib.sha256(command.encode()).hexdigest()


def test_stepfun_correctness_status_kv_resource_command_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-resource-command.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--kv-resource-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command"
    ]
    assert payload.startswith("python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan")
    assert f"> {resource}" in payload



def test_stepfun_correctness_status_status_refresh_command_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "status-refresh-command-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--status-refresh-command-only",
            "--status-refresh-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    command = status["next_action_commands"]["oracle_parity_blocked"][
        "status_refresh_command"
    ]
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "status_refresh_command_sha256"
    ]
    assert payload == hashlib.sha256(command.encode()).hexdigest()


def test_stepfun_correctness_status_status_refresh_command_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "status-refresh-command.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--status-refresh-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "status_refresh_command"
    ]
    assert f"--prompt-artifact {prompt}" in payload
    assert f"--resource-artifact {resource}" in payload



def test_stepfun_correctness_status_source_verify_command_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "source-verify-command-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--source-verify-command-only",
            "--source-verify-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    command = status["next_action_commands"]["handoff_integrity"][
        "source_artifacts_verify_command"
    ]
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "source_artifacts_verify_command_sha256"
    ]
    assert payload == hashlib.sha256(command.encode()).hexdigest()



def test_stepfun_correctness_status_source_verify_command_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "source-verify-command.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--source-verify-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "source_artifacts_verify_command"
    ]
    assert payload == _source_verify_command()
    assert "--verify-source-artifacts" in payload



def test_stepfun_correctness_status_oracle_helper_command_sha_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-helper-command-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--readiness-summary-only",
            "--oracle-helper-command-only",
            "--oracle-helper-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_refresh_command_sha256"
    ]
    assert payload == hashlib.sha256(_oracle_helper_command(prompt, oracle).encode()).hexdigest()



def test_stepfun_correctness_status_oracle_helper_command_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-helper-command.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--oracle-helper-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_refresh_command"
    ]
    assert payload == _oracle_helper_command(prompt, oracle)



def test_stepfun_correctness_status_oracle_helper_command_preserves_diagnostic_logs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-helper-command-diagnostic.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle, diagnostic_logs=True)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--oracle-helper-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    expected = _oracle_helper_command(prompt, oracle, diagnostic_logs=True)
    assert status["oracle_progress"]["diagnostic_logs"] is True
    assert status["handoff_summary"]["first_blocker_work_item"]["diagnostic_logs"] is True
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_refresh_command"
    ]
    assert payload == expected
    assert "--diagnostic-logs" in payload
    assert hashlib.sha256(payload.encode()).hexdigest() == status["next_action_commands"][
        "oracle_parity_blocked"
    ]["oracle_helper_refresh_command_sha256"]



def test_stepfun_correctness_status_first_blocker_only(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "first-blocker.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--first-blocker-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == {
        "blocker_kind": "oracle_parity_blocked",
        "work_item_schema_version": 1,
        "queue_index": 0,
        "is_first": True,
        "command_available": True,
        **_primary_command_fields(
            "rerun_command_shell",
            "/tmp/llama-cli --model stepfun.gguf --predict 1 --temp 0",
        ),
        **_oracle_helper_fields(prompt, oracle),
        "first_missing_evidence": "oracle_completed_successfully",
        "first_missing_precondition": "step35_not_rejected",
        "gap_report_status": "blocked",
        "current_status": "executed",
        "current_returncode": 1,
        "elapsed_s": 62.4,
        "timeout_s": 60.0,
        "diagnostic_logs": False,
        "gate": "oracle_parity",
        "oracle_blocker_kind": "llama_cpp_missing_step35_architecture",
    }


def test_stepfun_correctness_status_verifies_source_artifact_provenance(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_output = tmp_path / "verify.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(status_output),
            "--pretty",
        ]
    )
    assert rc == 0

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(verify_output.read_text())
    assert payload["status"] == "match"
    assert payload["status_artifact"] == str(status_output)
    assert payload["all_match"] is True
    assert payload["source_artifacts_all_match"] is True
    assert payload["status_integrity"] == {
        "all_match": True,
        "failed_checks": [],
        "checks": {
            "source_artifacts_sha256": True,
            "handoff_summary_sha256": True,
            "readiness_summary_sha256": True,
            "readiness_gates_sha256": True,
            "next_action_commands_sha256": True,
            "blocker_kinds_sha256": True,
            "blocked_gates_sha256": True,
            "kv_streaming_runner_blocker_names_sha256": True,
            "kv_streaming_runner_blocker_mirrors": True,
            "schema_versions": True,
        },
    }
    assert payload["checked_count"] == 4
    assert payload["records"]["prompt"]["match"] is True
    assert payload["records"]["prompt"]["matches"] == {
        "exists": True,
        "sha256": True,
        "size_bytes": True,
    }
    assert payload["records"]["docs"]["current"]["sha256"] == hashlib.sha256(
        docs.read_bytes()
    ).hexdigest()


def test_stepfun_correctness_status_source_artifact_verify_detects_stale_inputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_output = tmp_path / "verify.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(status_output),
        ]
    )
    assert rc == 0
    prompt.write_text(prompt.read_text() + "\n")

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_output),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(verify_output.read_text())
    assert payload["status"] == "mismatch"
    assert payload["all_match"] is False
    assert payload["source_artifacts_all_match"] is False
    assert payload["status_integrity"]["all_match"] is True
    assert payload["records"]["prompt"]["match"] is False
    assert payload["records"]["prompt"]["matches"] == {
        "exists": True,
        "sha256": False,
        "size_bytes": False,
    }
    assert payload["records"]["oracle"]["match"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_status_digest_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_output = tmp_path / "verify.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(status_output),
        ]
    )
    assert rc == 0
    status_payload = json.loads(status_output.read_text())
    status_payload["next_action_commands_sha256"] = "stale"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_output),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(verify_output.read_text())
    assert payload["status"] == "mismatch"
    assert payload["all_match"] is False
    assert payload["source_artifacts_all_match"] is True
    assert payload["status_integrity"]["all_match"] is False
    assert payload["status_integrity"]["failed_checks"] == ["next_action_commands_sha256"]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "handoff_summary_sha256": True,
        "readiness_summary_sha256": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": False,
        "blocker_kinds_sha256": True,
        "blocked_gates_sha256": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_mirrors": True,
        "schema_versions": True,
    }


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_streaming_digest_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_output = tmp_path / "verify.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(status_output),
        ]
    )
    assert rc == 0
    status_payload = json.loads(status_output.read_text())
    status_payload["kv_backed_decode_gap_report"][
        "streaming_runner_blocker_names_sha256"
    ] = "stale"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_output),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(verify_output.read_text())
    assert payload["status"] == "mismatch"
    assert payload["all_match"] is False
    assert payload["source_artifacts_all_match"] is True
    assert payload["status_integrity"]["all_match"] is False
    assert payload["status_integrity"]["failed_checks"] == [
        "kv_streaming_runner_blocker_names_sha256",
        "kv_streaming_runner_blocker_mirrors",
    ]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "handoff_summary_sha256": True,
        "readiness_summary_sha256": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "blocker_kinds_sha256": True,
        "blocked_gates_sha256": True,
        "kv_streaming_runner_blocker_names_sha256": False,
        "kv_streaming_runner_blocker_mirrors": False,
        "schema_versions": True,
    }


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_streaming_mirror_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_output = tmp_path / "verify.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--output",
            str(status_output),
        ]
    )
    assert rc == 0
    status_payload = json.loads(status_output.read_text())
    status_payload["next_action_commands"]["kv_backed_decode_not_wired"][
        "streaming_runner_blocker_names"
    ] = ["stale_streaming_blocker"]
    status_payload["next_action_commands_sha256"] = _stable_json_sha256(
        status_payload["next_action_commands"]
    )
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_output),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(verify_output.read_text())
    assert payload["status"] == "mismatch"
    assert payload["all_match"] is False
    assert payload["source_artifacts_all_match"] is True
    assert payload["status_integrity"]["all_match"] is False
    assert payload["status_integrity"]["failed_checks"] == ["kv_streaming_runner_blocker_mirrors"]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "handoff_summary_sha256": True,
        "readiness_summary_sha256": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "blocker_kinds_sha256": True,
        "blocked_gates_sha256": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_mirrors": False,
        "schema_versions": True,
    }


def test_stepfun_correctness_status_kv_resource_command_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--kv-resource-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command"
    ]
    assert payload.startswith("python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan")


def test_stepfun_correctness_status_kv_resource_command_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--kv-resource-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command_sha256"
    ]
    assert payload == hashlib.sha256(
        status["next_action_commands"]["kv_backed_decode_not_wired"][
            "resource_plan_refresh_command"
        ].encode()
    ).hexdigest()


def test_stepfun_correctness_status_status_refresh_command_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-refresh-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "status_refresh_command"
    ]


def test_stepfun_correctness_status_status_refresh_command_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-refresh-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "status_refresh_command_sha256"
    ]



def test_stepfun_correctness_status_source_verify_command_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--source-verify-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "source_artifacts_verify_command"
    ]
    assert payload == _source_verify_command()



def test_stepfun_correctness_status_source_verify_command_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--source-verify-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "source_artifacts_verify_command_sha256"
    ]
    assert payload == hashlib.sha256(_source_verify_command().encode()).hexdigest()



def test_stepfun_correctness_status_oracle_helper_command_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--oracle-helper-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_refresh_command"
    ]
    assert payload == _oracle_helper_command(prompt, oracle)


def test_stepfun_correctness_status_oracle_helper_command_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--oracle-helper-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_refresh_command_sha256"
    ]
    assert payload == hashlib.sha256(_oracle_helper_command(prompt, oracle).encode()).hexdigest()



def test_stepfun_correctness_status_first_blocker_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--first-blocker-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["handoff_summary"]["first_blocker_work_item_sha256"]



def test_stepfun_correctness_status_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
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



def test_stepfun_correctness_status_readiness_summary_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--readiness-summary-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["readiness_summary_sha256"]
    assert payload == _stable_json_sha256(status["readiness_summary"])


def test_stepfun_correctness_status_readiness_gates_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--readiness-gates-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["readiness_gates_sha256"]
    assert payload == _stable_json_sha256(status["readiness_gates"])


def test_stepfun_correctness_status_readiness_gates_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--readiness-gates-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["readiness_gates"]
    assert payload["oracle_parity"]["ready"] is False
    assert payload["kv_backed_decode"]["ready"] is False
    assert payload["kv_backed_decode"]["dispatch_ready"] is True
    assert payload["e2e_inference"]["ready"] is False


def test_stepfun_correctness_status_blocked_gates_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--blocked-gates-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["blocked_gates_sha256"]
    assert payload == _stable_json_sha256(status["blocked_gates"])


def test_stepfun_correctness_status_blocked_gates_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--blocked-gates-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["blocked_gates"]
    assert payload == ["oracle_parity", "kv_backed_decode", "e2e_inference"]


def test_stepfun_correctness_status_kv_streaming_blockers_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--kv-streaming-blockers-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_runner_blocker_names_sha256"
    ]
    assert payload == _streaming_runner_blocker_names_sha256()


def test_stepfun_correctness_status_kv_streaming_blockers_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--kv-streaming-blockers-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["kv_backed_decode_gap_report"]["streaming_runner_blocker_names"]
    assert payload == _streaming_runner_blocker_names()


def test_stepfun_correctness_status_blocker_kinds_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--blocker-kinds-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["blocker_kinds_sha256"]
    assert payload == _stable_json_sha256(status["blocker_kinds"])


def test_stepfun_correctness_status_blocker_kinds_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--blocker-kinds-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["blocker_kinds"]
    assert payload == ["oracle_parity_blocked", "kv_backed_decode_not_wired"]


def test_stepfun_correctness_status_handoff_summary_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--handoff-summary-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["handoff_summary_sha256"]
    assert payload == _stable_json_sha256(status["handoff_summary"])



def test_stepfun_correctness_status_schema_versions_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--schema-versions-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["schema_versions"]
    assert payload == {
        "status": 1,
        "readiness_summary": 1,
        "handoff_summary": 1,
        "blocker_work_queue": 1,
        "first_blocker_work_item": 1,
    }


def test_stepfun_correctness_status_status_integrity_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-integrity-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == _status_integrity(status)
    assert payload["all_match"] is True
    assert payload["failed_checks"] == []
    assert all(payload["checks"].values())


def test_stepfun_correctness_status_status_integrity_failures_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--status-integrity-failures-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == _status_integrity(status)["failed_checks"]
    assert payload == []


def test_stepfun_correctness_status_source_artifacts_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--source-artifacts-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["source_artifacts_sha256"]
    assert payload == _stable_json_sha256(status["source_artifacts"])



def test_stepfun_correctness_status_next_action_commands_sha_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--next-action-commands-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands_sha256"]
    assert payload == _stable_json_sha256(status["next_action_commands"])



def test_stepfun_correctness_status_readiness_summary_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)
    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--readiness-summary-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["readiness_summary"]
    assert payload["status"] == "blocked"
    assert payload["fail_on_blocked_exit_code"] == 2
    assert payload["first_blocker_kind"] == "oracle_parity_blocked"



def test_stepfun_correctness_status_summary_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--summary-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["open_blocker_count"] == 2
    assert payload["exit_codes"]["current_with_fail_on_blocked"] == 2
    assert payload["compact_output_modes"]["summary_only"] == "handoff_summary"
    assert payload["first_blocker_work_item"]["blocker_kind"] == "oracle_parity_blocked"



def test_stepfun_correctness_status_blocker_work_queue_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--blocker-work-queue-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert [item["blocker_kind"] for item in payload] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload[0]["primary_command_kind"] == "rerun_command_shell"
    assert payload[0]["first_missing_evidence"] == "oracle_completed_successfully"
    assert payload[1]["primary_command_kind"] == "resource_plan_refresh_command"
    assert payload[1]["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"



def test_stepfun_correctness_status_first_blocker_fail_on_blocked_returns_nonzero(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    rc = main(
        [
            "--prompt-artifact",
            str(prompt),
            "--oracle-artifact",
            str(oracle),
            "--resource-artifact",
            str(resource),
            "--docs",
            str(docs),
            "--first-blocker-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["blocker_kind"] == "oracle_parity_blocked"
    assert payload["primary_command_kind"] == "rerun_command_shell"
    assert payload["first_missing_evidence"] == "oracle_completed_successfully"
