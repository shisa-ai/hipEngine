from __future__ import annotations

import hashlib
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
                "generated_text": "",
                "llama_cli": "/tmp/llama-cli",
                "model": "/data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf",
                "prompt_length": 23,
                "n_predict": 1,
                "timeout_s": 60.0,
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
                        "streaming_runner_ready": False,
                        "note": "Metadata-only combined upload plan; no kernels are launched.",
                    },
                    "stop_token_ids": [1, 2, 128007],
                    "streaming_runner_ready": False,
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

    assert status["status"] == "blocked"
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
        "run_plan_decode_input_upload_entry_count": 6,
        "run_plan_decode_input_upload_total_nbytes": 484,
        "run_plan_host_payload_entry_count": 5,
        "run_plan_host_payload_total_nbytes": 392,
        "run_plan_streaming_ready": False,
        "run_plan_upload_manifest_entry_count": 5,
        "run_plan_upload_manifest_total_nbytes": 392,
    }
    assert gates["e2e_inference"]["ready"] is False
    assert gates["e2e_inference"]["blocked_by"] == ["oracle_parity", "kv_backed_decode"]
    handoff = status["handoff_summary"]
    assert handoff["status"] == "blocked"
    assert handoff["open_or_partial_items_p0_p12"] == 2
    assert handoff["open_blocker_count"] == 2
    assert handoff["open_blockers"] == ["oracle_parity_blocked", "kv_backed_decode_not_wired"]
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
    kv_blocker = next(blocker for blocker in status["blockers"] if blocker["kind"] == "kv_backed_decode_not_wired")
    assert kv_blocker["resource_artifact"] == str(resource)
    assert kv_blocker["kv_decode_dispatch_ready"] is True
    assert {action["blocker_kind"] for action in status["next_actions"]} == {
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    }
    commands = status["next_action_commands"]
    oracle_commands = commands["oracle_parity_blocked"]
    assert oracle_commands["rerun_command_shell"].startswith("/tmp/llama-cli")
    assert f"--prompt-artifact {prompt}" in oracle_commands["status_refresh_command"]
    assert f"--oracle-artifact {oracle}" in oracle_commands["status_refresh_command"]
    assert oracle_commands["success_criteria"] == [
        "oracle_progress.status is executed",
        "oracle_parity is true",
        "readiness_gates.oracle_parity.ready is true",
    ]
    kv_commands = commands["kv_backed_decode_not_wired"]
    assert kv_commands["resource_plan_refresh_command"] == (
        "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
        f"--kv-context-pages 1 --kv-page-size 512 --pretty > {resource}"
    )
    assert kv_commands["status_refresh_command"] == oracle_commands["status_refresh_command"]
    assert kv_commands["success_criteria"][-1] == "e2e_inference_ready is true only after oracle_parity is also true"
    assert status["docs_checklist"]["open_or_partial_count_p0_p12"] == 2
    assert [item["state"] for item in status["docs_checklist"]["open_or_partial_items_p0_p12"]] == [
        "open",
        "partial",
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
    assert payload["status"] == "blocked"
    assert payload["source_artifacts"]["prompt"]["sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert payload["source_artifacts"]["oracle"]["size_bytes"] == len(oracle.read_bytes())
    assert payload["source_artifacts"]["text_resource"]["exists"] is True
    assert payload["source_artifacts"]["docs"]["path"] == str(docs)
    assert payload["all_layer_prompt_smoke"] is True
    assert payload["e2e_inference_ready"] is False
    assert payload["oracle_progress"]["expected_next_token_id"] == 369
    assert payload["oracle_progress"]["timeout_s"] == 60.0
    assert payload["linear_projection_progress"]["resident_linear_projection_slot_count"] == 487
    assert payload["kv_decode_dispatch_ready"] is True
    assert payload["readiness_gates"]["oracle_parity"]["ready"] is False
    assert payload["readiness_gates"]["kv_backed_decode"]["dispatch_ready"] is True
    assert payload["readiness_gates"]["e2e_inference"]["blocked_by"] == [
        "oracle_parity",
        "kv_backed_decode",
    ]
    assert payload["handoff_summary"]["open_blockers"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["handoff_summary"]["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert payload["handoff_summary"]["no_claim_policy"]["performance_claim_allowed"] is False
    assert payload["next_action_commands"]["oracle_parity_blocked"]["rerun_command_shell"].startswith(
        "/tmp/llama-cli"
    )
    assert payload["next_action_commands"]["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command"
    ].endswith(f"> {resource}")
    assert len(payload["next_actions"]) == 2
    assert payload["docs_checklist"]["open_or_partial_count_p0_p12"] == 2


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
    assert payload["status"] == "blocked"
    assert payload["open_or_partial_items_p0_p12"] == 2
    assert payload["open_blockers"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["ready_signals"]["kv_decode_dispatch_ready"] is True
    assert payload["ready_signals"]["kv_decode_run_plan_recorded"] is True
    assert payload["ready_signals"]["kv_decode_input_upload_plan_recorded"] is True
    assert payload["kv_decode_input_upload_plan"]["entry_count"] == 6
    assert payload["kv_decode_input_upload_plan"]["upload_order"][0] == "input_ids"
    assert payload["kv_decode_input_upload_plan"]["cleanup_order"][-1] == "input_ids"
    assert payload["blocked_gates"] == ["oracle_parity", "kv_backed_decode", "e2e_inference"]
    assert payload["next_commands_available_for"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert payload["no_claim_policy"]["e2e_inference_claim_allowed"] is False
    assert "blockers" not in payload
    assert "docs_checklist" not in payload


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
    assert payload["records"]["prompt"]["match"] is False
    assert payload["records"]["prompt"]["matches"] == {
        "exists": True,
        "sha256": False,
        "size_bytes": False,
    }
    assert payload["records"]["oracle"]["match"] is True


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
