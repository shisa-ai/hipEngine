from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from scripts import stepfun_correctness_status

_status_integrity = stepfun_correctness_status._status_integrity
build_status = stepfun_correctness_status.build_status
main = stepfun_correctness_status.main


def test_stepfun_correctness_status_emit_json_replaces_output_atomically(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "status.json"
    output.write_text('{"status":"old"}\n')
    observed: dict[str, object] = {}
    real_replace = os.replace

    def spy_replace(src: object, dst: object) -> None:
        observed["destination_before_replace"] = output.read_text()
        observed["temp_payload"] = Path(src).read_text()
        real_replace(src, dst)

    monkeypatch.setattr(stepfun_correctness_status.os, "replace", spy_replace)

    stepfun_correctness_status._emit_json(
        {"status": "blocked", "open_or_partial_items_p0_p12": 2},
        pretty=True,
        output=output,
    )

    assert observed == {
        "destination_before_replace": '{"status":"old"}\n',
        "temp_payload": (
            '{\n  "open_or_partial_items_p0_p12": 2,\n  "status": "blocked"\n}\n'
        ),
    }
    assert json.loads(output.read_text()) == {
        "open_or_partial_items_p0_p12": 2,
        "status": "blocked",
    }
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def _primary_command_fields(kind: str | None, command: str | None) -> dict[str, object]:
    return {
        "primary_command_kind": kind,
        "primary_command": command,
        "primary_command_nchars": len(command) if command is not None else 0,
        "primary_command_sha256": (
            hashlib.sha256(command.encode()).hexdigest() if command is not None else None
        ),
    }


def _recommended_command_fields(
    kind: str | None,
    command: str | None,
    *,
    reason: str | None,
) -> dict[str, object]:
    return {
        "recommended_command_kind": kind,
        "recommended_command": command,
        "recommended_command_nchars": len(command) if command is not None else 0,
        "recommended_command_sha256": (
            hashlib.sha256(command.encode()).hexdigest() if command is not None else None
        ),
        "recommended_command_reason": reason if command is not None else None,
    }



def _stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()



def _oracle_helper_command(
    prompt: Path,
    oracle: Path,
    *,
    diagnostic_logs: bool = False,
    timeout_s: float = 60.0,
) -> str:
    diagnostic_logs_arg = "--diagnostic-logs " if diagnostic_logs else ""
    return (
        "python3 scripts/stepfun_llamacpp_oracle.py "
        f"--artifact {prompt} --llama-cli /tmp/llama-cli "
        "--model /data/models/gguf/Step-3.7-flash-Q3_K_L-00001-of-00003.gguf "
        f"--n-predict 1 --timeout-s {timeout_s} {diagnostic_logs_arg}"
        "--llama-arg=--device --llama-arg=none "
        f"--llama-arg=--gpu-layers --llama-arg=0 --execute --pretty --output {oracle}"
    )



def _resource_plan_refresh_command(resource: Path) -> str:
    return (
        "python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan "
        f"--kv-context-pages 1 --kv-page-size 512 --pretty --output {resource}"
    )


def _status_refresh_atomic_fields() -> dict[str, object]:
    return {
        "status_refresh_writes_atomic_output": True,
        "status_refresh_output_path": (
            "benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"
        ),
        "status_refresh_output_overwrite_policy": "atomic_os_replace",
        "status_refresh_uses_shell_redirection": False,
        "status_refresh_output_arg_present": True,
    }


def _resource_refresh_atomic_fields(resource: Path) -> dict[str, object]:
    return {
        "recommended_command_writes_atomic_output": True,
        "atomic_output_path": str(resource),
        "atomic_output_overwrite_policy": "atomic_os_replace",
        "atomic_output_helper": "stepfun_gguf_load_smoke.py",
        "recommended_command_uses_shell_redirection": False,
        "recommended_command_output_arg_present": True,
    }


def _oracle_helper_fields(
    prompt: Path,
    oracle: Path,
    *,
    diagnostic_logs: bool = False,
) -> dict[str, object]:
    command = _oracle_helper_command(prompt, oracle, diagnostic_logs=diagnostic_logs)
    long_command = _oracle_helper_command(
        prompt,
        oracle,
        diagnostic_logs=diagnostic_logs,
        timeout_s=900.0,
    )
    return {
        "helper_command_kind": "oracle_helper_refresh_command",
        "helper_command": command,
        "helper_command_nchars": len(command),
        "helper_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "long_timeout_helper_command_kind": "oracle_helper_long_timeout_command",
        "long_timeout_helper_command": long_command,
        "long_timeout_helper_command_nchars": len(long_command),
        "long_timeout_helper_command_sha256": hashlib.sha256(long_command.encode()).hexdigest(),
        "long_timeout_s": 900.0,
        "recommended_command_writes_partial_output_before_launch": True,
        "partial_output_status": "running",
        "partial_output_path": str(oracle),
        "partial_output_overwrite_policy": "overwrite_on_execute_or_timeout",
        "partial_output_blocker_kind": "llama_cpp_oracle_in_progress",
    }


def _source_verify_command(
    status_artifact: Path = Path("benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"),
    *,
    extra_args: tuple[str, ...] = (),
) -> str:
    args = " ".join(extra_args)
    suffix = f" {args}" if args else ""
    return (
        "python3 scripts/stepfun_correctness_status.py "
        f"--verify-source-artifacts {status_artifact}{suffix} --pretty"
    )



def _streaming_runner_blocker_names() -> list[str]:
    return [
        "streaming_decode_loop_not_wired",
        "kv_kernel_trace_artifact_missing",
        "kv_backed_next_token_artifact_missing",
    ]


def _streaming_runner_blocker_names_sha256() -> str:
    return _stable_json_sha256(_streaming_runner_blocker_names())


def _first_streaming_runner_blocker() -> str:
    return "streaming_decode_loop_not_wired"


def _first_streaming_runner_blocker_sha256() -> str:
    return _stable_json_sha256(_first_streaming_runner_blocker())


def _last_streaming_runner_blocker() -> str:
    return "kv_backed_next_token_artifact_missing"


def _last_streaming_runner_blocker_sha256() -> str:
    return _stable_json_sha256(_last_streaming_runner_blocker())


def _kernel_trace_streaming_runner_blocker() -> str:
    return "kv_kernel_trace_artifact_missing"


def _kernel_trace_streaming_runner_blocker_sha256() -> str:
    return _stable_json_sha256(_kernel_trace_streaming_runner_blocker())


def _kv_loop_operation_sequence() -> list[str]:
    return [
        f"layers.{layer_id}.{op_name}"
        for layer_id in range(45)
        for op_name in ["prompt_kv_write", "decode_kv_write", "decode_attention"]
    ]


def _streaming_decode_loop_blueprint() -> dict[str, object]:
    upload_order = [
        "input_ids",
        "prompt_base_offsets",
        "prompt_live_counts",
        "decode_base_offsets",
        "decode_kv_write_position",
        "decode_attention_live_counts",
    ]
    operation_sequence = _kv_loop_operation_sequence()
    return {
        "source": "kv_decode_run_plan",
        "executable": False,
        "blocked_by": _first_streaming_runner_blocker(),
        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
        "streaming_runner_ready": False,
        "layer_count": 45,
        "operation_count": len(operation_sequence),
        "per_layer_order": ["prompt_kv_write", "decode_kv_write", "decode_attention"],
        "operation_sequence_sha256": _stable_json_sha256(operation_sequence),
        "first_layer_ops": operation_sequence[:3],
        "last_layer_ops": operation_sequence[-3:],
        "pre_run_upload_order": upload_order,
        "pre_run_cleanup_order": list(reversed(upload_order)),
        "pre_run_upload_checks_passed": True,
        "stages": [
            {
                "name": "upload_decode_inputs",
                "source": "decode_input_upload_plan",
                "ready": True,
                "entry_count": 6,
                "total_nbytes": 484,
            },
            {
                "name": "prompt_prefill_kv_write",
                "dispatch_key": "prompt_kv_write",
                "span_contract": "prompt_span",
                "layer_count": 45,
                "ready": True,
            },
            {
                "name": "one_token_decode_kv_write",
                "dispatch_key": "decode_kv_write",
                "span_contract": "decode_span",
                "layer_count": 45,
                "ready": True,
            },
            {
                "name": "one_token_gated_attention_decode",
                "dispatch_key": "decode_attention",
                "span_contract": "decode_span",
                "layer_count": 45,
                "ready": True,
            },
        ],
        "stage_count": 4,
    }


def _streaming_decode_loop_blueprint_summary() -> dict[str, object]:
    return {
        "recorded": True,
        "matches_launch_schedule": True,
        "upload_order_matches": True,
        "blocker_matches": True,
        "executable": False,
        "blocked_by": _first_streaming_runner_blocker(),
        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
        "operation_count": 135,
        "operation_sequence_sha256": _stable_json_sha256(
            _kv_loop_operation_sequence()
        ),
        "stage_count": 4,
        "pre_run_upload_checks_passed": True,
    }


def _streaming_decode_loop_blueprint_summary_sha256() -> str:
    return _stable_json_sha256(_streaming_decode_loop_blueprint_summary())


def _streaming_decode_loop_status_summary() -> dict[str, object]:
    return {
        "recorded": True,
        "matches_blueprint": True,
        "blocker_names_match": True,
        "ready": False,
        "executable": False,
        "blocked_by": _first_streaming_runner_blocker(),
        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
        "blocker_count": 3,
        "blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "blueprint_operation_count": 135,
        "blueprint_stage_count": 4,
        "blueprint_sha256": _stable_json_sha256(_streaming_decode_loop_blueprint()),
        "next_action": "wire_streaming_decode_loop",
        "next_action_sha256": _stable_json_sha256("wire_streaming_decode_loop"),
    }


def _streaming_decode_loop_status_summary_sha256() -> str:
    return _stable_json_sha256(_streaming_decode_loop_status_summary())


def _streaming_decode_launch_trace() -> dict[str, object]:
    per_layer_order = ["prompt_kv_write", "decode_kv_write", "decode_attention"]
    stage_names = {
        "prompt_kv_write": "prompt_prefill_kv_write",
        "decode_kv_write": "one_token_decode_kv_write",
        "decode_attention": "one_token_gated_attention_decode",
    }
    span_contracts = {
        "prompt_kv_write": "prompt_span",
        "decode_kv_write": "decode_span",
        "decode_attention": "decode_span",
    }
    kernel_keys = {
        "prompt_kv_write": {
            "backend": "hip_gfx1151",
            "layer": "paged_kv_write",
            "quant": "gguf_step35",
            "variant": "mixed_bf16_prompt_spans",
        },
        "decode_kv_write": {
            "backend": "hip_gfx1151",
            "layer": "paged_kv_write",
            "quant": "gguf_step35",
            "variant": "mixed_bf16_spans",
        },
        "decode_attention": {
            "backend": "hip_gfx1151",
            "layer": "paged_attn_decode",
            "quant": "gguf_step35",
            "variant": "bf16_split_k_gate_f32_spans",
        },
    }
    span_uploads_by_operation = {
        "prompt_kv_write": ["prompt_base_offsets", "prompt_live_counts"],
        "decode_kv_write": ["decode_base_offsets", "decode_kv_write_position"],
        "decode_attention": ["decode_base_offsets", "decode_attention_live_counts"],
    }
    runtime_inputs_by_operation = {
        "prompt_kv_write": ["layer_prompt_key", "layer_prompt_value"],
        "decode_kv_write": ["layer_decode_key", "layer_decode_value"],
        "decode_attention": ["layer_decode_query", "layer_decode_attention_gate"],
    }
    records: list[dict[str, object]] = []
    for layer_id in range(45):
        for name in per_layer_order:
            records.append(
                {
                    "op_index": len(records),
                    "operation": f"layers.{layer_id}.{name}",
                    "layer": layer_id,
                    "name": name,
                    "stage_name": stage_names[name],
                    "dispatch_key_name": name,
                    "kernel_key": kernel_keys[name],
                    "span_contract": span_contracts[name],
                    "pre_run_uploads": span_uploads_by_operation[name],
                    "expected_runtime_inputs": runtime_inputs_by_operation[name],
                    "launch_ready": True,
                    "execution_status": "not_launched_metadata_only",
                    "blocked_by": _first_streaming_runner_blocker(),
                }
            )
    operation_sequence = [str(record["operation"]) for record in records]
    return {
        "schema_version": 1,
        "source": "kv_decode_run_plan",
        "executable": False,
        "ready": False,
        "blocked_by": _first_streaming_runner_blocker(),
        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
        "layer_count": 45,
        "per_layer_order": per_layer_order,
        "operation_count": len(records),
        "operation_sequence_sha256": _stable_json_sha256(operation_sequence),
        "operation_records_sha256": _stable_json_sha256(records),
        "first_operation": records[0],
        "last_operation": records[-1],
        "span_uploads_by_operation": span_uploads_by_operation,
        "pre_run_upload_order": [
            "input_ids",
            "prompt_base_offsets",
            "prompt_live_counts",
            "decode_base_offsets",
            "decode_kv_write_position",
            "decode_attention_live_counts",
        ],
        "all_launches_have_dispatch_keys": True,
        "all_launches_ready": True,
        "no_kernel_launches": True,
        "operation_records": records,
        "note": (
            "Metadata-only per-layer launch trace for the future StepFun KV "
            "streaming decode loop; it records dispatch/span/upload contracts but "
            "does not launch kernels or produce a token."
        ),
    }


def _streaming_decode_launch_trace_summary() -> dict[str, object]:
    trace = _streaming_decode_launch_trace()
    return {
        "recorded": True,
        "matches_blueprint": True,
        "non_executable": True,
        "dispatch_ready": True,
        "executable": False,
        "ready": False,
        "blocked_by": _first_streaming_runner_blocker(),
        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
        "layer_count": 45,
        "operation_count": 135,
        "per_layer_order": ["prompt_kv_write", "decode_kv_write", "decode_attention"],
        "operation_sequence_sha256": _stable_json_sha256(_kv_loop_operation_sequence()),
        "operation_records_sha256": trace["operation_records_sha256"],
        "first_operation": "layers.0.prompt_kv_write",
        "last_operation": "layers.44.decode_attention",
        "all_launches_have_dispatch_keys": True,
        "all_launches_ready": True,
        "no_kernel_launches": True,
    }


def _streaming_decode_launch_trace_summary_sha256() -> str:
    return _stable_json_sha256(_streaming_decode_launch_trace_summary())


def _kv_decode_blocker_summary() -> dict[str, object]:
    artifacts_needed = [
        {
            "name": "kv_kernel_trace_artifact",
            "required_for": "kv_kernel_trace_artifact_missing",
            "evidence": (
                "rocprofv3 or equivalent trace showing prompt KV write, decode KV write, "
                "and gated decode-attention kernels for the canonical prompt"
            ),
        },
        {
            "name": "kv_backed_next_token_artifact",
            "required_for": "kv_backed_next_token_artifact_missing",
            "evidence": (
                "one-token decode artifact recording generated token/logit path from KV-backed "
                "runtime execution, not host-composed layer-prefix outputs"
            ),
        },
    ]
    return {
        "schema_version": 1,
        "source": "kv_decode_run_plan",
        "status": "blocked",
        "ready": False,
        "executable": False,
        "next_action": "wire_streaming_decode_loop",
        "blocker_count": 3,
        "blocker_names": _streaming_runner_blocker_names(),
        "blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "first_blocker": _streaming_runner_blockers()[0],
        "first_blocker_name": _first_streaming_runner_blocker(),
        "first_blocker_sha256": _stable_json_sha256(_streaming_runner_blockers()[0]),
        "last_blocker": _streaming_runner_blockers()[-1],
        "last_blocker_name": _last_streaming_runner_blocker(),
        "last_blocker_sha256": _stable_json_sha256(_streaming_runner_blockers()[-1]),
        "kernel_trace_blocker": _streaming_runner_blockers()[1],
        "kernel_trace_blocker_name": _kernel_trace_streaming_runner_blocker(),
        "kernel_trace_blocker_sha256": _stable_json_sha256(_streaming_runner_blockers()[1]),
        "kernel_trace_blocker_present": True,
        "upload_plan_ready": True,
        "upload_entry_count": 6,
        "upload_total_nbytes": 484,
        "launch_blueprint_ready": True,
        "launch_stage_count": 4,
        "launch_operation_count": 135,
        "per_layer_order": ["prompt_kv_write", "decode_kv_write", "decode_attention"],
        "artifacts_needed": artifacts_needed,
        "artifacts_needed_sha256": _stable_json_sha256(artifacts_needed),
        "artifact_count": len(artifacts_needed),
        "no_claim_policy": {
            "oracle_parity_claim_allowed": False,
            "kv_backed_decode_claim_allowed": False,
            "performance_claim_allowed": False,
            "reason": (
                "metadata-only KV decode planning is not a streaming decode execution and "
                "does not generate a token/logit artifact"
            ),
        },
    }


def _kv_decode_blocker_summary_sha256() -> str:
    return _stable_json_sha256(_kv_decode_blocker_summary())


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


def _streaming_runner_blockers_sha256() -> str:
    return _stable_json_sha256(_streaming_runner_blockers())


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
                        "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
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
                    "streaming_decode_loop_blueprint": _streaming_decode_loop_blueprint(),
                    "streaming_decode_loop_status": {
                        "source": "kv_decode_run_plan",
                        "ready": False,
                        "executable": False,
                        "blocked_by": _first_streaming_runner_blocker(),
                        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
                        "blocker_count": 3,
                        "blocker_names": _streaming_runner_blocker_names(),
                        "blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
                        "blueprint_operation_count": 135,
                        "blueprint_stage_count": 4,
                        "blueprint_sha256": _stable_json_sha256(
                            _streaming_decode_loop_blueprint()
                        ),
                        "next_action": "wire_streaming_decode_loop",
                        "note": (
                            "Metadata-only readiness summary for the future StepFun KV streaming "
                            "decode loop; no kernels are launched."
                        ),
                    },
                    "streaming_decode_launch_trace": _streaming_decode_launch_trace(),
                    "kv_decode_blocker_summary": _kv_decode_blocker_summary(),
                    "stop_token_ids": [1, 2, 128007],
                    "streaming_runner_ready": False,
                    "streaming_runner_blocker_count": 3,
                    "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
                    "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
                    "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
                    "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
                    "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
                    "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
                    "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
                    "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
                    "kernel_trace_streaming_runner_blocker_present": True,
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
    assert oracle_progress["timeout_termination"] is None
    assert oracle_progress["timeout_termination_recorded"] is False
    assert oracle_progress["timeout_termination_sha256"] is None
    assert status["oracle_progress_sha256"] == _stable_json_sha256(oracle_progress)
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
    assert kv_dispatch["streaming_decode_loop_blueprint"] == _streaming_decode_loop_blueprint()
    assert kv_dispatch["streaming_decode_loop_blueprint_recorded"] is True
    assert kv_dispatch["streaming_decode_loop_blueprint_matches_launch_schedule"] is True
    assert kv_dispatch["streaming_decode_loop_blueprint_upload_order_matches"] is True
    assert kv_dispatch["streaming_decode_loop_blueprint_blocker_matches"] is True
    assert kv_dispatch["streaming_decode_loop_status"]
    assert kv_dispatch["streaming_decode_loop_status_sha256"] == _stable_json_sha256(
        kv_dispatch["streaming_decode_loop_status"]
    )
    assert kv_dispatch["streaming_decode_loop_status_recorded"] is True
    assert kv_dispatch["streaming_decode_loop_status_matches_blueprint"] is True
    assert kv_dispatch["streaming_decode_loop_status_blocker_names_match"] is True
    assert kv_dispatch["streaming_decode_launch_trace"] == _streaming_decode_launch_trace()
    assert kv_dispatch["streaming_decode_launch_trace_sha256"] == _stable_json_sha256(
        _streaming_decode_launch_trace()
    )
    assert kv_dispatch["streaming_decode_launch_trace_recorded"] is True
    assert kv_dispatch["streaming_decode_launch_trace_matches_blueprint"] is True
    assert kv_dispatch["streaming_decode_launch_trace_non_executable"] is True
    assert kv_dispatch["streaming_decode_launch_trace_dispatch_ready"] is True
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
    assert oracle_gap["timeout_termination"] is None
    assert oracle_gap["timeout_termination_recorded"] is False
    assert oracle_gap["timeout_termination_sha256"] is None
    assert oracle_gap["remaining_evidence"][0]["current"]["timeout_termination_recorded"] is False
    assert oracle_gap["remaining_evidence"][0]["current"]["timeout_termination_sha256"] is None
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
    assert gap_report["precondition_count"] == 8
    assert gap_report["validated_precondition_count"] == 8
    assert gap_report["missing_preconditions"] == []
    assert gap_report["missing_evidence"] == [
        "streaming_runner_ready_flags",
        "kv_kernel_launch_trace",
        "kv_backed_next_token_artifact",
    ]
    assert gap_report["first_missing_evidence"] == "streaming_runner_ready_flags"
    assert gap_report["missing_evidence_count"] == 3
    assert gap_report["operation_count"] == 135
    assert gap_report["streaming_decode_loop_blueprint"] == _streaming_decode_loop_blueprint_summary()
    assert gap_report[
        "streaming_decode_loop_blueprint_sha256"
    ] == _streaming_decode_loop_blueprint_summary_sha256()
    assert gap_report["streaming_decode_launch_trace"] == _streaming_decode_launch_trace()
    assert gap_report["streaming_decode_launch_trace_sha256"] == _stable_json_sha256(
        _streaming_decode_launch_trace()
    )
    assert gap_report[
        "streaming_decode_launch_trace_summary"
    ] == _streaming_decode_launch_trace_summary()
    assert gap_report[
        "streaming_decode_launch_trace_summary_sha256"
    ] == _streaming_decode_launch_trace_summary_sha256()
    assert gap_report["streaming_runner_blocker_count"] == 3
    assert gap_report["streaming_runner_blockers_present"] is True
    assert gap_report["streaming_runner_blocker_names"] == _streaming_runner_blocker_names()
    assert gap_report["streaming_runner_blocker_names_sha256"] == _streaming_runner_blocker_names_sha256()
    assert gap_report["computed_streaming_runner_blocker_names_sha256"] == _streaming_runner_blocker_names_sha256()
    assert gap_report["streaming_runner_blocker_names_sha256_match"] is True
    assert gap_report["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert gap_report["first_streaming_runner_blocker_sha256"] == _first_streaming_runner_blocker_sha256()
    assert gap_report["last_streaming_runner_blocker"] == _last_streaming_runner_blocker()
    assert gap_report["last_streaming_runner_blocker_sha256"] == _last_streaming_runner_blocker_sha256()
    assert gap_report["kernel_trace_streaming_runner_blocker"] == _kernel_trace_streaming_runner_blocker()
    assert gap_report["kernel_trace_streaming_runner_blocker_sha256"] == _kernel_trace_streaming_runner_blocker_sha256()
    assert gap_report["kernel_trace_streaming_runner_blocker_present"] is True
    assert gap_report["streaming_runner_blockers"] == _streaming_runner_blockers()
    assert gap_report["streaming_runner_blockers_sha256"] == _streaming_runner_blockers_sha256()
    assert gap_report["upload_entry_count"] == 6
    assert gap_report["upload_total_nbytes"] == 484
    assert gap_report["remaining_evidence"][0]["current"] == {
        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
        "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
        "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
        "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
        "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
        "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
        "kernel_trace_streaming_runner_blocker_present": True,
        "launch_schedule_streaming_runner_blocker_count": 3,
        "launch_schedule_streaming_runner_ready": False,
        "run_plan_streaming_runner_blocker_count": 3,
        "run_plan_streaming_runner_ready": False,
        "streaming_runner_blocker_count": 3,
        "streaming_runner_blockers_present": True,
        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "computed_streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "streaming_runner_blocker_names_sha256_match": True,
        "streaming_runner_blocker_names_joined": "|".join(
            _streaming_runner_blocker_names()
        ),
        "streaming_runner_blocker_names_joined_sha256": _stable_json_sha256(
            "|".join(_streaming_runner_blocker_names())
        ),
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
            **_recommended_command_fields(
                "oracle_helper_long_timeout_command",
                _oracle_helper_command(prompt, oracle, timeout_s=900.0),
                reason="oracle_timeout_retry_with_longer_timeout",
            ),
            **_oracle_helper_fields(prompt, oracle),
            **_status_refresh_atomic_fields(),
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
                    f"--kv-context-pages 1 --kv-page-size 512 --pretty --output {resource}"
                ),
            ),
            **_recommended_command_fields(
                "resource_plan_refresh_command",
                _resource_plan_refresh_command(resource),
                reason="refresh_kv_resource_and_run_plan_artifact",
            ),
            **_resource_refresh_atomic_fields(resource),
            **_status_refresh_atomic_fields(),
            "first_missing_evidence": "streaming_runner_ready_flags",
            "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
            "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
            "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
            "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
            "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
            "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
            "kernel_trace_streaming_runner_blocker_present": True,
            "streaming_decode_loop_blueprint": _streaming_decode_loop_blueprint_summary(),
            "streaming_decode_loop_blueprint_sha256": _streaming_decode_loop_blueprint_summary_sha256(),
            "streaming_decode_loop_status": _streaming_decode_loop_status_summary(),
            "streaming_decode_loop_status_sha256": _streaming_decode_loop_status_summary_sha256(),
            "gap_report_status": "blocked",
            "operation_count": 135,
            "streaming_runner_blocker_count": 3,
            "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
            "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
            "streaming_runner_blocker_names_sha256_match": True,
            "streaming_runner_blockers": _streaming_runner_blockers(),
            "streaming_runner_blockers_sha256": _streaming_runner_blockers_sha256(),
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
        "first_recommended_command_kind": "oracle_helper_long_timeout_command",
        "first_recommended_command_sha256": hashlib.sha256(
            _oracle_helper_command(prompt, oracle, timeout_s=900.0).encode()
        ).hexdigest(),
        "recommended_commands_sha256": _stable_json_sha256(
            handoff["blocker_recommended_commands"]
        ),
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
        "schema_versions_sha_only": "schema_versions_sha256",
        "status_integrity_only": "status_integrity",
        "status_integrity_sha_only": "status_integrity_sha256",
        "status_integrity_failures_only": "status_integrity.failed_checks",
        "persisted_status_integrity_only": "persisted_status_integrity",
        "persisted_status_integrity_failures_only": (
            "persisted_status_integrity.failed_checks"
        ),
        "docs_checklist_only": "docs_checklist",
        "docs_checklist_sha_only": "docs_checklist_sha256",
        "docs_open_partial_count_only": (
            "docs_checklist.open_or_partial_count_p0_p12"
        ),
        "docs_open_partial_summary_only": (
            "docs_checklist.open_or_partial_summary_p0_p12"
        ),
        "docs_open_partial_summary_sha_only": (
            "docs_checklist.open_or_partial_summary_p0_p12_sha256"
        ),
        "docs_open_partial_state_counts_only": (
            "docs_checklist.open_or_partial_state_counts_p0_p12"
        ),
        "docs_open_partial_state_counts_sha_only": (
            "docs_checklist.open_or_partial_state_counts_p0_p12_sha256"
        ),
        "docs_open_partial_lines_only": (
            "docs_checklist.open_or_partial_lines_p0_p12"
        ),
        "docs_open_partial_lines_sha_only": (
            "docs_checklist.open_or_partial_lines_p0_p12_sha256"
        ),
        "docs_open_partial_texts_only": (
            "docs_checklist.open_or_partial_texts_p0_p12"
        ),
        "docs_open_partial_texts_sha_only": (
            "docs_checklist.open_or_partial_texts_p0_p12_sha256"
        ),
        "docs_open_partial_texts_joined_only": (
            "docs_checklist.open_or_partial_texts_joined_p0_p12"
        ),
        "docs_open_partial_texts_joined_sha_only": (
            "docs_checklist.open_or_partial_texts_joined_p0_p12_sha256"
        ),
        "docs_open_partial_line_texts_joined_only": (
            "docs_checklist.open_or_partial_line_texts_joined_p0_p12"
        ),
        "docs_open_partial_line_texts_joined_sha_only": (
            "docs_checklist.open_or_partial_line_texts_joined_p0_p12_sha256"
        ),
        "docs_open_partial_state_line_texts_joined_only": (
            "docs_checklist.open_or_partial_state_line_texts_joined_p0_p12"
        ),
        "docs_open_partial_state_line_texts_joined_sha_only": (
            "docs_checklist.open_or_partial_state_line_texts_joined_p0_p12_sha256"
        ),
        "docs_first_open_partial_item_only": (
            "docs_checklist.first_open_or_partial_item_p0_p12"
        ),
        "docs_first_open_partial_item_sha_only": (
            "docs_checklist.first_open_or_partial_item_p0_p12.sha256"
        ),
        "docs_last_open_partial_item_only": (
            "docs_checklist.last_open_or_partial_item_p0_p12"
        ),
        "docs_last_open_partial_item_sha_only": (
            "docs_checklist.last_open_or_partial_item_p0_p12.sha256"
        ),
        "readiness_summary_only": "readiness_summary",
        "readiness_summary_sha_only": "readiness_summary_sha256",
        "readiness_gates_only": "readiness_gates",
        "readiness_gates_sha_only": "readiness_gates_sha256",
        "blocked_gates_only": "blocked_gates",
        "blocked_gates_sha_only": "blocked_gates_sha256",
        "remaining_blockers_report_only": "remaining_blockers_report",
        "remaining_blockers_report_sha_only": "remaining_blockers_report_sha256",
        "first_remaining_blocker_report_only": "first_remaining_blocker_report",
        "first_remaining_blocker_report_sha_only": "first_remaining_blocker_report_sha256",
        "source_artifacts_sha_only": "source_artifacts_sha256",
        "oracle_wrapper_timeout_source_only": "source_artifacts.oracle_wrapper_timeout",
        "oracle_wrapper_timeout_source_sha_only": (
            "source_artifacts.oracle_wrapper_timeout.sha256"
        ),
        "text_resource_source_only": "source_artifacts.text_resource",
        "text_resource_source_sha_only": "source_artifacts.text_resource.sha256",
        "next_action_commands_only": "next_action_commands",
        "next_action_commands_sha_only": "next_action_commands_sha256",
        "atomic_output_handoff_only": "atomic_output_handoff",
        "atomic_output_handoff_sha_only": "atomic_output_handoff_sha256",
        "oracle_partial_output_handoff_only": "oracle_partial_output_handoff",
        "oracle_partial_output_handoff_sha_only": "oracle_partial_output_handoff_sha256",
        "blocker_kinds_only": "blocker_kinds",
        "blocker_kinds_sha_only": "blocker_kinds_sha256",
        "kv_streaming_blockers_only": "kv_streaming_runner_blocker_names",
        "kv_streaming_blockers_sha_only": "kv_streaming_runner_blocker_names_sha256",
        "kv_streaming_blockers_joined_only": (
            "kv_backed_decode_gap_report.streaming_runner_blocker_names_joined"
        ),
        "kv_streaming_blockers_joined_sha_only": (
            "kv_backed_decode_gap_report.streaming_runner_blocker_names_joined_sha256"
        ),
        "kv_streaming_blocker_count_only": (
            "kv_backed_decode_gap_report.streaming_runner_blocker_count"
        ),
        "kv_streaming_blockers_present_only": (
            "kv_backed_decode_gap_report.streaming_runner_blockers_present"
        ),
        "kv_streaming_blocker_records_only": (
            "kv_backed_decode_gap_report.streaming_runner_blockers"
        ),
        "kv_streaming_blocker_records_sha_only": (
            "kv_backed_decode_gap_report.streaming_runner_blockers_sha256"
        ),
        "kv_first_streaming_blocker_only": (
            "kv_backed_decode_gap_report.first_streaming_runner_blocker"
        ),
        "kv_first_streaming_blocker_sha_only": (
            "kv_backed_decode_gap_report.first_streaming_runner_blocker_sha256"
        ),
        "kv_last_streaming_blocker_only": (
            "kv_backed_decode_gap_report.last_streaming_runner_blocker"
        ),
        "kv_last_streaming_blocker_sha_only": (
            "kv_backed_decode_gap_report.last_streaming_runner_blocker_sha256"
        ),
        "kv_kernel_trace_streaming_blocker_only": (
            "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker"
        ),
        "kv_kernel_trace_streaming_blocker_sha_only": (
            "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker_sha256"
        ),
        "kv_kernel_trace_streaming_blocker_present_only": (
            "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker_present"
        ),
        "kv_streaming_blueprint_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_blueprint"
        ),
        "kv_streaming_blueprint_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_blueprint_sha256"
        ),
        "kv_streaming_loop_status_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status"
        ),
        "kv_streaming_loop_status_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status_sha256"
        ),
        "kv_streaming_loop_next_action_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status.next_action"
        ),
        "kv_streaming_loop_next_action_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status.next_action_sha256"
        ),
        "kv_streaming_launch_trace_only": (
            "kv_backed_decode_gap_report.streaming_decode_launch_trace"
        ),
        "kv_streaming_launch_trace_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_launch_trace_sha256"
        ),
        "kv_decode_blocker_summary_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary"
        ),
        "kv_decode_blocker_summary_sha_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary_sha256"
        ),
        "kv_required_artifacts_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary.artifacts_needed"
        ),
        "kv_required_artifacts_sha_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary.artifacts_needed_sha256"
        ),
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
        "verification_status_command_only": (
            "next_action_commands.handoff_integrity.verification_status_command"
        ),
        "verification_status_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_status_command_sha256"
        ),
        "verification_exit_code_command_only": (
            "next_action_commands.handoff_integrity.verification_exit_code_command"
        ),
        "verification_exit_code_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_exit_code_command_sha256"
        ),
        "verification_failures_command_only": (
            "next_action_commands.handoff_integrity.verification_failures_command"
        ),
        "verification_failures_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_failures_command_sha256"
        ),
        "verification_failures_sha_command_only": (
            "next_action_commands.handoff_integrity.verification_failures_sha_command"
        ),
        "verification_failures_sha_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_failures_sha_command_sha256"
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
        "oracle_progress_only": "oracle_progress",
        "oracle_progress_sha_only": "oracle_progress_sha256",
        "oracle_helper_long_timeout_command_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command"
        ),
        "oracle_helper_long_timeout_command_sha_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command_sha256"
        ),
        "oracle_timeout_termination_only": "oracle_gap_report.timeout_termination",
        "oracle_timeout_termination_sha_only": (
            "oracle_gap_report.timeout_termination_sha256"
        ),
        "blocker_work_queue_only": "handoff_summary.blocker_work_queue",
        "blocker_work_queue_meta_only": "handoff_summary.blocker_work_queue_meta",
        "blocker_work_queue_sha_only": "handoff_summary.blocker_work_queue_sha256",
        "blocker_recommended_commands_only": "handoff_summary.blocker_recommended_commands",
        "blocker_recommended_commands_sha_only": "handoff_summary.blocker_recommended_commands_sha256",
        "first_blocker_sha_only": "handoff_summary.first_blocker_work_item_sha256",
        "first_blocker_only": "handoff_summary.first_blocker_work_item",
        "first_blocker_recommended_command_only": (
            "handoff_summary.first_blocker_work_item.recommended_command"
        ),
        "first_blocker_recommended_command_sha_only": (
            "handoff_summary.first_blocker_work_item.recommended_command_sha256"
        ),
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
        "kv_streaming_decode_loop_blueprint_recorded": True,
        "kv_streaming_decode_loop_status_recorded": True,
        "kv_streaming_decode_launch_trace_recorded": True,
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
        "precondition_count": 8,
        "streaming_decode_loop_blueprint": {
            "recorded": True,
            "matches_launch_schedule": True,
            "upload_order_matches": True,
            "blocker_matches": True,
            "executable": False,
            "blocked_by": "streaming_decode_loop_not_wired",
            "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
            "operation_count": 135,
            "operation_sequence_sha256": _stable_json_sha256(
                _kv_loop_operation_sequence()
            ),
            "stage_count": 4,
            "pre_run_upload_checks_passed": True,
        },
        "streaming_decode_loop_blueprint_sha256": _streaming_decode_loop_blueprint_summary_sha256(),
        "streaming_decode_loop_status": _streaming_decode_loop_status_summary(),
        "streaming_decode_loop_status_sha256": _streaming_decode_loop_status_summary_sha256(),
        "streaming_decode_launch_trace_summary": _streaming_decode_launch_trace_summary(),
        "streaming_decode_launch_trace_summary_sha256": _streaming_decode_launch_trace_summary_sha256(),
        "streaming_decode_launch_trace_sha256": _stable_json_sha256(_streaming_decode_launch_trace()),
        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
        "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
        "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
        "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
        "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
        "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
        "kernel_trace_streaming_runner_blocker_present": True,
        "status": "blocked",
        "streaming_runner_blocker_count": 3,
        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "streaming_runner_blocker_names_sha256_match": True,
        "streaming_runner_blockers": _streaming_runner_blockers(),
        "streaming_runner_blockers_sha256": _streaming_runner_blockers_sha256(),
        "upload_total_nbytes": 484,
        "validated_precondition_count": 8,
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
    assert kv_blocker["first_streaming_runner_blocker_sha256"] == _first_streaming_runner_blocker_sha256()
    assert kv_blocker["last_streaming_runner_blocker"] == _last_streaming_runner_blocker()
    assert kv_blocker["last_streaming_runner_blocker_sha256"] == _last_streaming_runner_blocker_sha256()
    assert kv_blocker["kernel_trace_streaming_runner_blocker"] == _kernel_trace_streaming_runner_blocker()
    assert kv_blocker["kernel_trace_streaming_runner_blocker_sha256"] == _kernel_trace_streaming_runner_blocker_sha256()
    assert kv_blocker["kernel_trace_streaming_runner_blocker_present"] is True
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
    assert integrity_commands["verification_status_command"] == _source_verify_command(
        extra_args=("--verification-status-only",)
    )
    assert integrity_commands["verification_status_command_nchars"] == len(
        integrity_commands["verification_status_command"]
    )
    assert integrity_commands["verification_status_command_sha256"] == hashlib.sha256(
        integrity_commands["verification_status_command"].encode()
    ).hexdigest()
    assert integrity_commands["verification_exit_code_command"] == _source_verify_command(
        extra_args=("--verification-exit-code-only",)
    )
    assert integrity_commands["verification_exit_code_command_nchars"] == len(
        integrity_commands["verification_exit_code_command"]
    )
    assert integrity_commands["verification_exit_code_command_sha256"] == hashlib.sha256(
        integrity_commands["verification_exit_code_command"].encode()
    ).hexdigest()
    assert integrity_commands["verification_failures_command"] == _source_verify_command(
        extra_args=("--verification-failures-only",)
    )
    assert integrity_commands["verification_failures_command_nchars"] == len(
        integrity_commands["verification_failures_command"]
    )
    assert integrity_commands["verification_failures_command_sha256"] == hashlib.sha256(
        integrity_commands["verification_failures_command"].encode()
    ).hexdigest()
    assert integrity_commands["verification_failures_sha_command"] == _source_verify_command(
        extra_args=("--verification-failures-sha-only",)
    )
    assert integrity_commands["verification_failures_sha_command_nchars"] == len(
        integrity_commands["verification_failures_sha_command"]
    )
    assert integrity_commands["verification_failures_sha_command_sha256"] == hashlib.sha256(
        integrity_commands["verification_failures_sha_command"].encode()
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
    expected_long_timeout_command = _oracle_helper_command(prompt, oracle, timeout_s=900.0)
    assert oracle_commands["oracle_helper_long_timeout_command"] == expected_long_timeout_command
    assert oracle_commands["oracle_helper_long_timeout_s"] == 900.0
    assert oracle_commands["oracle_helper_long_timeout_command_nchars"] == len(
        expected_long_timeout_command
    )
    assert oracle_commands["oracle_helper_long_timeout_command_sha256"] == hashlib.sha256(
        expected_long_timeout_command.encode()
    ).hexdigest()
    assert "--timeout-s 900.0" in oracle_commands["oracle_helper_long_timeout_command"]
    assert oracle_commands["oracle_helper_writes_partial_output_before_launch"] is True
    assert oracle_commands["oracle_helper_partial_output_status"] == "running"
    assert oracle_commands["oracle_helper_partial_output_path"] == str(oracle)
    assert oracle_commands["oracle_helper_partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    assert oracle_commands["oracle_helper_partial_output_blocker_kind"] == (
        "llama_cpp_oracle_in_progress"
    )
    assert f"--prompt-artifact {prompt}" in oracle_commands["status_refresh_command"]
    assert f"--oracle-artifact {oracle}" in oracle_commands["status_refresh_command"]
    assert oracle_commands["status_refresh_command_nchars"] == len(
        oracle_commands["status_refresh_command"]
    )
    assert oracle_commands["status_refresh_command_sha256"] == hashlib.sha256(
        oracle_commands["status_refresh_command"].encode()
    ).hexdigest()
    assert oracle_commands["status_refresh_writes_atomic_output"] is True
    assert oracle_commands["status_refresh_output_helper"] == (
        "stepfun_correctness_status.py"
    )
    assert oracle_commands["status_refresh_output_path"] == (
        "benchmarks/results/2026-05-31-stepfun-q3kl-correctness-status.json"
    )
    assert oracle_commands["status_refresh_output_overwrite_policy"] == (
        "atomic_os_replace"
    )
    assert oracle_commands["status_refresh_uses_shell_redirection"] is False
    assert oracle_commands["status_refresh_output_arg_present"] is True
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
        f"--kv-context-pages 1 --kv-page-size 512 --pretty --output {resource}"
    )
    assert kv_commands["resource_plan_refresh_command_nchars"] == len(
        kv_commands["resource_plan_refresh_command"]
    )
    assert kv_commands["resource_plan_refresh_command_sha256"] == hashlib.sha256(
        kv_commands["resource_plan_refresh_command"].encode()
    ).hexdigest()
    assert kv_commands["resource_plan_refresh_writes_atomic_output"] is True
    assert kv_commands["resource_plan_refresh_output_helper"] == (
        "stepfun_gguf_load_smoke.py"
    )
    assert kv_commands["resource_plan_refresh_output_path"] == str(resource)
    assert kv_commands["resource_plan_refresh_output_overwrite_policy"] == (
        "atomic_os_replace"
    )
    assert kv_commands["resource_plan_refresh_uses_shell_redirection"] is False
    assert kv_commands["resource_plan_refresh_output_arg_present"] is True
    assert kv_commands["status_refresh_command"] == oracle_commands["status_refresh_command"]
    assert kv_commands["status_refresh_command_nchars"] == oracle_commands[
        "status_refresh_command_nchars"
    ]
    assert kv_commands["status_refresh_command_sha256"] == oracle_commands[
        "status_refresh_command_sha256"
    ]
    assert kv_commands["status_refresh_writes_atomic_output"] is True
    assert kv_commands["status_refresh_output_path"] == oracle_commands[
        "status_refresh_output_path"
    ]
    assert kv_commands["status_refresh_output_overwrite_policy"] == (
        "atomic_os_replace"
    )
    assert kv_commands["status_refresh_uses_shell_redirection"] is False
    assert kv_commands["status_refresh_output_arg_present"] is True
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
    assert kv_commands["streaming_runner_blockers"] == _streaming_runner_blockers()
    assert kv_commands["streaming_runner_blockers_sha256"] == _streaming_runner_blockers_sha256()
    assert kv_commands["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert kv_commands["first_streaming_runner_blocker_sha256"] == _first_streaming_runner_blocker_sha256()
    assert kv_commands["last_streaming_runner_blocker"] == _last_streaming_runner_blocker()
    assert kv_commands["last_streaming_runner_blocker_sha256"] == _last_streaming_runner_blocker_sha256()
    assert kv_commands["kernel_trace_streaming_runner_blocker"] == _kernel_trace_streaming_runner_blocker()
    assert kv_commands["kernel_trace_streaming_runner_blocker_sha256"] == _kernel_trace_streaming_runner_blocker_sha256()
    assert kv_commands["kernel_trace_streaming_runner_blocker_present"] is True
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


def test_stepfun_correctness_status_surfaces_oracle_timeout_termination(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle, diagnostic_logs=True)
    _write_resource_artifact(resource)
    _write_docs(docs)
    termination = {
        "timeout_reached": True,
        "timeout_s": 60.0,
        "process_group_started": True,
        "termination_method": "os.killpg",
        "termination_signal": "SIGKILL",
        "termination_signal_number": 9,
        "termination_path": "killpg_sigkill_then_communicate",
        "communicate_after_signal_timeout_s": 10.0,
        "process_exited_before_signal": False,
        "fallback_proc_kill_used": False,
    }
    oracle_payload = json.loads(oracle.read_text())
    oracle_payload.update(
        {
            "status": "timeout",
            "returncode": None,
            "stderr": "",
            "oracle_blocker_kind": "llama_cpp_oracle_timeout",
            "oracle_blocker_detail": "llama.cpp oracle timed out before producing a comparable token",
            "step35_supported": None,
            "timeout_termination": termination,
        }
    )
    oracle.write_text(json.dumps(oracle_payload))

    status = build_status(prompt, oracle, docs, resource_artifact=resource)

    expected_sha = _stable_json_sha256(termination)
    progress = status["oracle_progress"]
    assert progress["status"] == "timeout"
    assert progress["returncode"] is None
    assert progress["oracle_blocker_kind"] == "llama_cpp_oracle_timeout"
    assert progress["timeout_termination"] == termination
    assert progress["timeout_termination_recorded"] is True
    assert progress["timeout_termination_sha256"] == expected_sha
    gap = status["oracle_gap_report"]
    assert gap["timeout_termination"] == termination
    assert gap["timeout_termination_recorded"] is True
    assert gap["timeout_termination_sha256"] == expected_sha
    assert gap["remaining_evidence"][0]["current"] == {
        "status": "timeout",
        "returncode": None,
        "oracle_blocker_kind": "llama_cpp_oracle_timeout",
        "elapsed_s": 62.4,
        "timeout_s": 60.0,
        "timeout_termination_recorded": True,
        "timeout_termination_sha256": expected_sha,
    }


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
    run_plan["streaming_decode_loop_status"]["ready"] = True
    run_plan["streaming_decode_loop_status"]["next_action"] = None
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
        "first_recommended_command_kind": "oracle_helper_long_timeout_command",
        "first_recommended_command_sha256": hashlib.sha256(
            _oracle_helper_command(prompt, oracle, timeout_s=900.0).encode()
        ).hexdigest(),
        "recommended_commands_sha256": _stable_json_sha256(
            payload["handoff_summary"]["blocker_recommended_commands"]
        ),
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
    assert payload["handoff_summary"]["blocker_work_queue"][1][
        "last_streaming_runner_blocker"
    ] == _last_streaming_runner_blocker()
    assert payload["handoff_summary"]["blocker_work_queue"][1][
        "kernel_trace_streaming_runner_blocker"
    ] == _kernel_trace_streaming_runner_blocker()
    assert payload["handoff_summary"]["blocker_work_queue"][1][
        "kernel_trace_streaming_runner_blocker_present"
    ] is True
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
    assert integrity_command["verification_status_command"] == _source_verify_command(
        extra_args=("--verification-status-only",)
    )
    assert integrity_command["verification_status_command_sha256"] == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-status-only",)).encode()
    ).hexdigest()
    assert integrity_command["verification_exit_code_command"] == _source_verify_command(
        extra_args=("--verification-exit-code-only",)
    )
    assert integrity_command["verification_exit_code_command_sha256"] == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-exit-code-only",)).encode()
    ).hexdigest()
    assert integrity_command["verification_failures_command"] == _source_verify_command(
        extra_args=("--verification-failures-only",)
    )
    assert integrity_command["verification_failures_command_sha256"] == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-failures-only",)).encode()
    ).hexdigest()
    assert integrity_command["verification_failures_sha_command"] == _source_verify_command(
        extra_args=("--verification-failures-sha-only",)
    )
    assert integrity_command["verification_failures_sha_command_sha256"] == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-failures-sha-only",)).encode()
    ).hexdigest()
    oracle_command = payload["next_action_commands"]["oracle_parity_blocked"]
    assert oracle_command["rerun_command_shell"].startswith("/tmp/llama-cli")
    assert oracle_command["first_missing_precondition"] == "step35_not_rejected"
    assert oracle_command["first_missing_evidence"] == "oracle_completed_successfully"
    assert oracle_command["success_criteria"][0] == "oracle_gap_report.status is ready"
    kv_command = payload["next_action_commands"]["kv_backed_decode_not_wired"]
    assert kv_command["resource_plan_refresh_command"].endswith(f"--output {resource}")
    assert kv_command["first_missing_evidence"] == "streaming_runner_ready_flags"
    assert kv_command["first_streaming_runner_blocker"] == "streaming_decode_loop_not_wired"
    assert kv_command["last_streaming_runner_blocker"] == _last_streaming_runner_blocker()
    assert kv_command["kernel_trace_streaming_runner_blocker"] == _kernel_trace_streaming_runner_blocker()
    assert kv_command["kernel_trace_streaming_runner_blocker_present"] is True
    assert kv_command["streaming_runner_blocker_count"] == 3
    assert kv_command["success_criteria"][0] == "kv_backed_decode_gap_report.status is ready"
    assert len(payload["next_actions"]) == 2
    assert payload["docs_checklist"]["open_or_partial_count_p0_p12"] == 2
    assert payload["docs_checklist_sha256"] == _stable_json_sha256(
        payload["docs_checklist"]
    )


def test_stepfun_correctness_status_docs_checklist_outputs(capsys, tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "docs-checklist.json"
    sha_output = tmp_path / "docs-checklist-sha.json"
    count_output = tmp_path / "docs-open-partial-count.json"
    summary_output = tmp_path / "docs-open-partial-summary.json"
    summary_sha_output = tmp_path / "docs-open-partial-summary-sha.json"
    state_counts_output = tmp_path / "docs-open-partial-state-counts.json"
    state_counts_sha_output = tmp_path / "docs-open-partial-state-counts-sha.json"
    lines_output = tmp_path / "docs-open-partial-lines.json"
    lines_sha_output = tmp_path / "docs-open-partial-lines-sha.json"
    texts_output = tmp_path / "docs-open-partial-texts.json"
    texts_sha_output = tmp_path / "docs-open-partial-texts-sha.json"
    texts_joined_output = tmp_path / "docs-open-partial-texts-joined.json"
    texts_joined_sha_output = tmp_path / "docs-open-partial-texts-joined-sha.json"
    line_texts_joined_output = tmp_path / "docs-open-partial-line-texts-joined.json"
    line_texts_joined_sha_output = tmp_path / "docs-open-partial-line-texts-joined-sha.json"
    state_line_texts_joined_output = tmp_path / "docs-open-partial-state-line-texts-joined.json"
    state_line_texts_joined_sha_output = tmp_path / "docs-open-partial-state-line-texts-joined-sha.json"
    first_output = tmp_path / "docs-first-open-partial-item.json"
    first_sha_output = tmp_path / "docs-first-open-partial-item-sha.json"
    last_output = tmp_path / "docs-last-open-partial-item.json"
    last_sha_output = tmp_path / "docs-last-open-partial-item-sha.json"
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
            "--docs-checklist-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["docs_checklist"]
    assert payload["open_or_partial_count_p0_p12"] == 2
    assert [item["state"] for item in payload["open_or_partial_items_p0_p12"]] == [
        "open",
        "partial",
    ]
    assert payload["first_open_or_partial_item_p0_p12"] == payload[
        "open_or_partial_items_p0_p12"
    ][0]
    assert payload["first_open_or_partial_item_p0_p12"]["state"] == "open"
    assert payload["last_open_or_partial_item_p0_p12"] == payload[
        "open_or_partial_items_p0_p12"
    ][-1]
    assert payload["last_open_or_partial_item_p0_p12"]["state"] == "partial"
    expected_state_counts = {"open": 1, "partial": 1}
    expected_lines = [
        payload["first_open_or_partial_item_p0_p12"]["line"],
        payload["last_open_or_partial_item_p0_p12"]["line"],
    ]
    expected_texts = [
        payload["first_open_or_partial_item_p0_p12"]["text"],
        payload["last_open_or_partial_item_p0_p12"]["text"],
    ]
    expected_texts_joined = "|".join(expected_texts)
    expected_line_texts_joined = "|".join(
        f"{line}:{text}" for line, text in zip(expected_lines, expected_texts)
    )
    expected_state_line_texts_joined = "|".join(
        f"{state}:{line}:{text}"
        for state, line, text in zip(
            ["open", "partial"], expected_lines, expected_texts
        )
    )
    assert payload["open_or_partial_state_counts_p0_p12"] == expected_state_counts
    assert payload["open_or_partial_state_counts_p0_p12_sha256"] == (
        _stable_json_sha256(expected_state_counts)
    )
    assert payload["open_or_partial_lines_p0_p12"] == expected_lines
    assert payload["open_or_partial_lines_p0_p12_sha256"] == (
        _stable_json_sha256(expected_lines)
    )
    assert payload["open_or_partial_texts_p0_p12"] == expected_texts
    assert payload["open_or_partial_texts_p0_p12_sha256"] == (
        _stable_json_sha256(expected_texts)
    )
    assert payload["open_or_partial_texts_joined_p0_p12"] == expected_texts_joined
    assert payload["open_or_partial_texts_joined_p0_p12_sha256"] == (
        _stable_json_sha256(expected_texts_joined)
    )
    assert payload["open_or_partial_line_texts_joined_p0_p12"] == (
        expected_line_texts_joined
    )
    assert payload["open_or_partial_line_texts_joined_p0_p12_sha256"] == (
        _stable_json_sha256(expected_line_texts_joined)
    )
    assert payload["open_or_partial_state_line_texts_joined_p0_p12"] == (
        expected_state_line_texts_joined
    )
    assert payload["open_or_partial_state_line_texts_joined_p0_p12_sha256"] == (
        _stable_json_sha256(expected_state_line_texts_joined)
    )
    expected_summary = {
        "schema_version": 1,
        "docs_path": str(docs),
        "open_or_partial_count_p0_p12": 2,
        "open_or_partial_items_sha256": _stable_json_sha256(
            payload["open_or_partial_items_p0_p12"]
        ),
        "open_or_partial_states": ["open", "partial"],
        "open_or_partial_state_counts": expected_state_counts,
        "open_or_partial_state_counts_sha256": _stable_json_sha256(
            expected_state_counts
        ),
        "open_or_partial_lines": expected_lines,
        "open_or_partial_lines_sha256": _stable_json_sha256(expected_lines),
        "open_or_partial_texts": expected_texts,
        "open_or_partial_texts_sha256": _stable_json_sha256(expected_texts),
        "open_or_partial_texts_joined": expected_texts_joined,
        "open_or_partial_texts_joined_sha256": _stable_json_sha256(
            expected_texts_joined
        ),
        "open_or_partial_line_texts_joined": expected_line_texts_joined,
        "open_or_partial_line_texts_joined_sha256": _stable_json_sha256(
            expected_line_texts_joined
        ),
        "open_or_partial_state_line_texts_joined": (
            expected_state_line_texts_joined
        ),
        "open_or_partial_state_line_texts_joined_sha256": _stable_json_sha256(
            expected_state_line_texts_joined
        ),
        "first_open_or_partial_item_p0_p12": payload[
            "first_open_or_partial_item_p0_p12"
        ],
        "first_open_or_partial_item_sha256": _stable_json_sha256(
            payload["first_open_or_partial_item_p0_p12"]
        ),
        "last_open_or_partial_item_p0_p12": payload[
            "last_open_or_partial_item_p0_p12"
        ],
        "last_open_or_partial_item_sha256": _stable_json_sha256(
            payload["last_open_or_partial_item_p0_p12"]
        ),
    }
    assert payload["open_or_partial_summary_p0_p12"] == expected_summary
    assert payload["open_or_partial_summary_p0_p12_sha256"] == (
        _stable_json_sha256(expected_summary)
    )
    assert status["docs_checklist_sha256"] == _stable_json_sha256(payload)
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_checklist_only"
    ] == "docs_checklist"

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
            str(sha_output),
            "--summary-only",
            "--docs-checklist-only",
            "--docs-checklist-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(sha_output.read_text()) == status["docs_checklist_sha256"]

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
            str(count_output),
            "--summary-only",
            "--docs-checklist-only",
            "--docs-checklist-sha-only",
            "--docs-open-partial-count-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(count_output.read_text()) == 2
    assert json.loads(count_output.read_text()) == status["docs_checklist"][
        "open_or_partial_count_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_count_only"
    ] == "docs_checklist.open_or_partial_count_p0_p12"

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
            str(summary_output),
            "--summary-only",
            "--docs-open-partial-summary-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(summary_output.read_text()) == status["docs_checklist"][
        "open_or_partial_summary_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_summary_only"
    ] == "docs_checklist.open_or_partial_summary_p0_p12"

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
            str(summary_sha_output),
            "--summary-only",
            "--docs-open-partial-summary-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(summary_sha_output.read_text()) == status["docs_checklist"][
        "open_or_partial_summary_p0_p12_sha256"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_summary_sha_only"
    ] == "docs_checklist.open_or_partial_summary_p0_p12_sha256"

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
            str(state_counts_output),
            "--summary-only",
            "--docs-open-partial-state-counts-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(state_counts_output.read_text()) == status["docs_checklist"][
        "open_or_partial_state_counts_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_state_counts_only"
    ] == "docs_checklist.open_or_partial_state_counts_p0_p12"

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
            str(state_counts_sha_output),
            "--summary-only",
            "--docs-open-partial-state-counts-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(state_counts_sha_output.read_text()) == status["docs_checklist"][
        "open_or_partial_state_counts_p0_p12_sha256"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_state_counts_sha_only"
    ] == "docs_checklist.open_or_partial_state_counts_p0_p12_sha256"

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
            str(lines_output),
            "--summary-only",
            "--docs-open-partial-lines-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(lines_output.read_text()) == status["docs_checklist"][
        "open_or_partial_lines_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_lines_only"
    ] == "docs_checklist.open_or_partial_lines_p0_p12"

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
            str(lines_sha_output),
            "--summary-only",
            "--docs-open-partial-lines-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(lines_sha_output.read_text()) == status["docs_checklist"][
        "open_or_partial_lines_p0_p12_sha256"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_lines_sha_only"
    ] == "docs_checklist.open_or_partial_lines_p0_p12_sha256"

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
            str(texts_output),
            "--summary-only",
            "--docs-open-partial-texts-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(texts_output.read_text()) == status["docs_checklist"][
        "open_or_partial_texts_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_texts_only"
    ] == "docs_checklist.open_or_partial_texts_p0_p12"

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
            str(texts_sha_output),
            "--summary-only",
            "--docs-open-partial-texts-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(texts_sha_output.read_text()) == status["docs_checklist"][
        "open_or_partial_texts_p0_p12_sha256"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_texts_sha_only"
    ] == "docs_checklist.open_or_partial_texts_p0_p12_sha256"

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
            str(texts_joined_output),
            "--summary-only",
            "--docs-open-partial-texts-joined-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(texts_joined_output.read_text()) == status["docs_checklist"][
        "open_or_partial_texts_joined_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_texts_joined_only"
    ] == "docs_checklist.open_or_partial_texts_joined_p0_p12"

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
            str(texts_joined_sha_output),
            "--summary-only",
            "--docs-open-partial-texts-joined-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(texts_joined_sha_output.read_text()) == status[
        "docs_checklist"
    ]["open_or_partial_texts_joined_p0_p12_sha256"]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_texts_joined_sha_only"
    ] == "docs_checklist.open_or_partial_texts_joined_p0_p12_sha256"

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
            str(line_texts_joined_output),
            "--summary-only",
            "--docs-open-partial-line-texts-joined-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(line_texts_joined_output.read_text()) == status[
        "docs_checklist"
    ]["open_or_partial_line_texts_joined_p0_p12"]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_line_texts_joined_only"
    ] == "docs_checklist.open_or_partial_line_texts_joined_p0_p12"

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
            str(line_texts_joined_sha_output),
            "--summary-only",
            "--docs-open-partial-line-texts-joined-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(line_texts_joined_sha_output.read_text()) == status[
        "docs_checklist"
    ]["open_or_partial_line_texts_joined_p0_p12_sha256"]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_line_texts_joined_sha_only"
    ] == "docs_checklist.open_or_partial_line_texts_joined_p0_p12_sha256"

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
            str(state_line_texts_joined_output),
            "--summary-only",
            "--docs-open-partial-state-line-texts-joined-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(state_line_texts_joined_output.read_text()) == status[
        "docs_checklist"
    ]["open_or_partial_state_line_texts_joined_p0_p12"]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_state_line_texts_joined_only"
    ] == "docs_checklist.open_or_partial_state_line_texts_joined_p0_p12"

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
            str(state_line_texts_joined_sha_output),
            "--summary-only",
            "--docs-open-partial-state-line-texts-joined-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(state_line_texts_joined_sha_output.read_text()) == status[
        "docs_checklist"
    ]["open_or_partial_state_line_texts_joined_p0_p12_sha256"]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_open_partial_state_line_texts_joined_sha_only"
    ] == "docs_checklist.open_or_partial_state_line_texts_joined_p0_p12_sha256"

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
            str(first_output),
            "--summary-only",
            "--docs-first-open-partial-item-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(first_output.read_text()) == status["docs_checklist"][
        "first_open_or_partial_item_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_first_open_partial_item_only"
    ] == "docs_checklist.first_open_or_partial_item_p0_p12"

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
            str(first_sha_output),
            "--summary-only",
            "--docs-first-open-partial-item-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(first_sha_output.read_text()) == _stable_json_sha256(
        status["docs_checklist"]["first_open_or_partial_item_p0_p12"]
    )
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_first_open_partial_item_sha_only"
    ] == "docs_checklist.first_open_or_partial_item_p0_p12.sha256"

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
            str(last_output),
            "--summary-only",
            "--docs-last-open-partial-item-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(last_output.read_text()) == status["docs_checklist"][
        "last_open_or_partial_item_p0_p12"
    ]
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_last_open_partial_item_only"
    ] == "docs_checklist.last_open_or_partial_item_p0_p12"

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
            str(last_sha_output),
            "--summary-only",
            "--docs-last-open-partial-item-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(last_sha_output.read_text()) == _stable_json_sha256(
        status["docs_checklist"]["last_open_or_partial_item_p0_p12"]
    )
    assert status["handoff_summary"]["compact_output_modes"][
        "docs_last_open_partial_item_sha_only"
    ] == "docs_checklist.last_open_or_partial_item_p0_p12.sha256"


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


def test_stepfun_correctness_status_remaining_blockers_report_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    report_output = tmp_path / "remaining-blockers-report.json"
    sha_output = tmp_path / "remaining-blockers-report-sha.json"
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
            str(report_output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--blocked-gates-only",
            "--remaining-blockers-report-only",
            "--pretty",
        ]
    )
    assert rc == 0
    report = json.loads(report_output.read_text())
    assert report == status["remaining_blockers_report"]
    assert report["schema_version"] == 1
    assert report["status"] == "blocked"
    assert report["open_or_partial_items_p0_p12"] == 2
    assert report["remaining_blocker_count"] == 2
    assert report["remaining_blocker_kinds"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert report["blocked_gates"] == [
        "oracle_parity",
        "kv_backed_decode",
        "e2e_inference",
    ]
    assert report["no_claim_policy"]["performance_claim_allowed"] is False
    assert report["items"][0]["checklist_item"] == (
        "P11 llama.cpp greedy next-token/logit comparison"
    )
    assert report["items"][0]["readiness_gate"] == "oracle_parity"
    assert report["items"][0]["gate_ready"] is False
    assert report["items"][0]["recommended_command_kind"] == (
        "oracle_helper_long_timeout_command"
    )
    assert report["items"][0]["recommended_command_writes_partial_output_before_launch"] is True
    assert report["items"][0]["partial_output_status"] == "running"
    assert report["items"][0]["partial_output_path"] == str(oracle)
    assert report["items"][0]["partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    assert report["items"][0]["partial_output_blocker_kind"] == (
        "llama_cpp_oracle_in_progress"
    )
    assert "oracle_parity is true" in report["items"][0]["success_criteria"]
    assert report["items"][1]["checklist_item"] == (
        "P11 text-only c=1 KV-backed decode runner"
    )
    assert report["items"][1]["readiness_gate"] == "kv_backed_decode"
    assert report["items"][1]["gate_ready"] is False
    assert report["items"][1]["first_streaming_runner_blocker"] == (
        "streaming_decode_loop_not_wired"
    )
    assert report["items"][1]["last_streaming_runner_blocker"] == (
        _last_streaming_runner_blocker()
    )
    assert report["items"][1]["kernel_trace_streaming_runner_blocker"] == (
        _kernel_trace_streaming_runner_blocker()
    )
    assert report["items"][1]["kernel_trace_streaming_runner_blocker_present"] is True
    assert report["items"][1]["recommended_command_kind"] == (
        "resource_plan_refresh_command"
    )
    assert "kv_backed_decode_ready is true" in report["items"][1]["success_criteria"]

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
            str(sha_output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--readiness-summary-only",
            "--remaining-blockers-report-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(sha_output.read_text()) == status["remaining_blockers_report_sha256"]
    assert json.loads(sha_output.read_text()) == _stable_json_sha256(
        status["remaining_blockers_report"]
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_correctness_status_first_remaining_blocker_report_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    report_output = tmp_path / "first-remaining-blocker-report.json"
    sha_output = tmp_path / "first-remaining-blocker-report-sha.json"
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
            str(report_output),
            "--summary-only",
            "--blocker-work-queue-only",
            "--remaining-blockers-report-only",
            "--first-remaining-blocker-report-only",
            "--pretty",
        ]
    )
    assert rc == 0
    report = json.loads(report_output.read_text())
    assert report == status["first_remaining_blocker_report"]
    assert report["schema_version"] == 1
    assert report["status"] == "blocked"
    assert report["open_or_partial_items_p0_p12"] == 2
    assert report["remaining_blocker_count"] == 2
    assert report["remaining_blocker_kinds"] == [
        "oracle_parity_blocked",
        "kv_backed_decode_not_wired",
    ]
    assert report["blocker_kind"] == "oracle_parity_blocked"
    assert report["queue_index"] == 0
    assert report["readiness_gate"] == "oracle_parity"
    assert report["gate_ready"] is False
    assert report["recommended_command_kind"] == "oracle_helper_long_timeout_command"
    assert report["recommended_command"] == status["remaining_blockers_report"][
        "items"
    ][0]["recommended_command"]
    assert report["recommended_command_sha256"] == status["remaining_blockers_report"][
        "items"
    ][0]["recommended_command_sha256"]
    assert report["recommended_command_writes_partial_output_before_launch"] is True
    assert report["partial_output_status"] == "running"
    assert report["partial_output_path"] == str(oracle)
    assert report["partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    assert report["partial_output_blocker_kind"] == "llama_cpp_oracle_in_progress"
    assert "oracle_parity is true" in report["success_criteria"]
    assert report["item"] == status["remaining_blockers_report"]["items"][0]
    assert report["no_claim_policy"]["performance_claim_allowed"] is False

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
            str(sha_output),
            "--summary-only",
            "--remaining-blockers-report-sha-only",
            "--first-remaining-blocker-report-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(sha_output.read_text()) == status[
        "first_remaining_blocker_report_sha256"
    ]
    assert json.loads(sha_output.read_text()) == _stable_json_sha256(
        status["first_remaining_blocker_report"]
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


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
    joined_output = tmp_path / "kv-streaming-blockers-joined.json"
    joined_sha_output = tmp_path / "kv-streaming-blockers-joined-sha.json"
    count_output = tmp_path / "kv-streaming-blocker-count.json"
    present_output = tmp_path / "kv-streaming-blockers-present.json"
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
    expected_joined = "|".join(_streaming_runner_blocker_names())
    assert status["kv_backed_decode_gap_report"][
        "streaming_runner_blocker_names_joined"
    ] == expected_joined
    assert status["kv_backed_decode_gap_report"][
        "streaming_runner_blocker_names_joined_sha256"
    ] == _stable_json_sha256(expected_joined)
    assert status["kv_backed_decode_gap_report"][
        "streaming_runner_blocker_count"
    ] == len(_streaming_runner_blocker_names())
    assert status["kv_backed_decode_gap_report"][
        "streaming_runner_blockers_present"
    ] is True

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
            str(joined_output),
            "--summary-only",
            "--kv-streaming-blockers-joined-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(joined_output.read_text()) == expected_joined
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_streaming_blockers_joined_only"
    ] == "kv_backed_decode_gap_report.streaming_runner_blocker_names_joined"

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
            str(joined_sha_output),
            "--summary-only",
            "--kv-streaming-blockers-joined-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(joined_sha_output.read_text()) == _stable_json_sha256(
        expected_joined
    )
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_streaming_blockers_joined_sha_only"
    ] == "kv_backed_decode_gap_report.streaming_runner_blocker_names_joined_sha256"

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
            str(count_output),
            "--summary-only",
            "--kv-streaming-blocker-count-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(count_output.read_text()) == len(
        _streaming_runner_blocker_names()
    )
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_streaming_blocker_count_only"
    ] == "kv_backed_decode_gap_report.streaming_runner_blocker_count"

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
            str(present_output),
            "--summary-only",
            "--kv-streaming-blockers-present-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(present_output.read_text()) is True
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_streaming_blockers_present_only"
    ] == "kv_backed_decode_gap_report.streaming_runner_blockers_present"


def test_stepfun_correctness_status_kv_streaming_blocker_records_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-blocker-records.json"
    sha_output = tmp_path / "kv-streaming-blocker-records-sha.json"
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
            "--kv-streaming-blocker-records-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"]["streaming_runner_blockers"]
    assert payload == _streaming_runner_blockers()
    assert payload[0]["required_evidence"].startswith("resident decode loop")
    assert status["kv_backed_decode_gap_report"][
        "streaming_runner_blockers_sha256"
    ] == _streaming_runner_blockers_sha256()
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_streaming_blocker_records_only"
    ] == "kv_backed_decode_gap_report.streaming_runner_blockers"

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
            str(sha_output),
            "--summary-only",
            "--kv-streaming-blocker-records-only",
            "--kv-streaming-blocker-records-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(sha_output.read_text()) == _streaming_runner_blockers_sha256()


def test_stepfun_correctness_status_kv_first_streaming_blocker_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-first-streaming-blocker.json"
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
            "--kv-first-streaming-blocker-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"]["first_streaming_runner_blocker"]
    assert payload == "streaming_decode_loop_not_wired"


def test_stepfun_correctness_status_kv_first_streaming_blocker_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-first-streaming-blocker-sha.json"
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
            "--kv-first-streaming-blocker-only",
            "--kv-first-streaming-blocker-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "first_streaming_runner_blocker_sha256"
    ]
    assert payload == _first_streaming_runner_blocker_sha256()


def test_stepfun_correctness_status_kv_last_streaming_blocker_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-last-streaming-blocker.json"
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
            "--kv-last-streaming-blocker-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"]["last_streaming_runner_blocker"]
    assert payload == _last_streaming_runner_blocker()
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_last_streaming_blocker_only"
    ] == "kv_backed_decode_gap_report.last_streaming_runner_blocker"


def test_stepfun_correctness_status_kv_last_streaming_blocker_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-last-streaming-blocker-sha.json"
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
            "--kv-last-streaming-blocker-only",
            "--kv-last-streaming-blocker-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "last_streaming_runner_blocker_sha256"
    ]
    assert payload == _last_streaming_runner_blocker_sha256()
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_last_streaming_blocker_sha_only"
    ] == "kv_backed_decode_gap_report.last_streaming_runner_blocker_sha256"


def test_stepfun_correctness_status_kv_kernel_trace_streaming_blocker_compact_modes(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    blocker_output = tmp_path / "kv-kernel-trace-streaming-blocker.json"
    sha_output = tmp_path / "kv-kernel-trace-streaming-blocker-sha.json"
    present_output = tmp_path / "kv-kernel-trace-streaming-blocker-present.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    calls = [
        (
            blocker_output,
            "--kv-kernel-trace-streaming-blocker-only",
            _kernel_trace_streaming_runner_blocker(),
        ),
        (
            sha_output,
            "--kv-kernel-trace-streaming-blocker-sha-only",
            _kernel_trace_streaming_runner_blocker_sha256(),
        ),
        (present_output, "--kv-kernel-trace-streaming-blocker-present-only", True),
    ]
    for output, mode, expected in calls:
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
                mode,
                "--pretty",
            ]
        )
        assert rc == 0
        assert json.loads(output.read_text()) == expected

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    gap_report = status["kv_backed_decode_gap_report"]
    assert gap_report["kernel_trace_streaming_runner_blocker"] == (
        _kernel_trace_streaming_runner_blocker()
    )
    assert gap_report["kernel_trace_streaming_runner_blocker_sha256"] == (
        _kernel_trace_streaming_runner_blocker_sha256()
    )
    assert gap_report["kernel_trace_streaming_runner_blocker_present"] is True
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_kernel_trace_streaming_blocker_only"
    ] == "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker"
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_kernel_trace_streaming_blocker_sha_only"
    ] == "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker_sha256"
    assert status["handoff_summary"]["compact_output_modes"][
        "kv_kernel_trace_streaming_blocker_present_only"
    ] == "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker_present"


def test_stepfun_correctness_status_kv_streaming_blueprint_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-blueprint.json"
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
            "--kv-streaming-blockers-only",
            "--kv-first-streaming-blocker-only",
            "--kv-streaming-blueprint-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_decode_loop_blueprint"
    ]
    assert payload == {
        "recorded": True,
        "matches_launch_schedule": True,
        "upload_order_matches": True,
        "blocker_matches": True,
        "executable": False,
        "blocked_by": "streaming_decode_loop_not_wired",
        "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
        "operation_count": 135,
        "operation_sequence_sha256": _stable_json_sha256(
            _kv_loop_operation_sequence()
        ),
        "stage_count": 4,
        "pre_run_upload_checks_passed": True,
    }


def test_stepfun_correctness_status_kv_streaming_blueprint_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-blueprint-sha.json"
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
            "--kv-streaming-blueprint-only",
            "--kv-streaming-blueprint-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    expected = status["kv_backed_decode_gap_report"][
        "streaming_decode_loop_blueprint"
    ]
    assert payload == _stable_json_sha256(expected)


def test_stepfun_correctness_status_kv_streaming_loop_status_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-loop-status.json"
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
            "--kv-streaming-blueprint-only",
            "--kv-streaming-loop-status-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_decode_loop_status"
    ]
    assert payload == _streaming_decode_loop_status_summary()


def test_stepfun_correctness_status_kv_streaming_loop_status_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-loop-status-sha.json"
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
            "--kv-streaming-loop-status-only",
            "--kv-streaming-loop-status-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_decode_loop_status_sha256"
    ]
    assert payload == _streaming_decode_loop_status_summary_sha256()


def test_stepfun_correctness_status_kv_streaming_launch_trace_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-launch-trace.json"
    sha_output = tmp_path / "kv-streaming-launch-trace-sha.json"
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
            "--kv-streaming-launch-trace-only",
            "--pretty",
        ]
    )
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_decode_launch_trace"
    ]
    assert payload == _streaming_decode_launch_trace()
    assert payload["operation_count"] == 135
    assert payload["first_operation"]["operation"] == "layers.0.prompt_kv_write"
    assert payload["last_operation"]["operation"] == "layers.44.decode_attention"
    assert payload["all_launches_have_dispatch_keys"] is True
    assert payload["no_kernel_launches"] is True

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
            str(sha_output),
            "--summary-only",
            "--kv-streaming-launch-trace-only",
            "--kv-streaming-launch-trace-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    sha_payload = json.loads(sha_output.read_text())
    assert sha_payload == status["kv_backed_decode_gap_report"][
        "streaming_decode_launch_trace_sha256"
    ]
    assert sha_payload == _stable_json_sha256(_streaming_decode_launch_trace())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_correctness_status_kv_decode_blocker_summary_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    summary_output = tmp_path / "kv-decode-blocker-summary.json"
    sha_output = tmp_path / "kv-decode-blocker-summary-sha.json"
    artifacts_output = tmp_path / "kv-required-artifacts.json"
    artifacts_sha_output = tmp_path / "kv-required-artifacts-sha.json"
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
            str(summary_output),
            "--summary-only",
            "--kv-streaming-loop-status-only",
            "--kv-decode-blocker-summary-only",
            "--pretty",
        ]
    )
    assert rc == 0
    summary = json.loads(summary_output.read_text())
    assert summary == status["kv_backed_decode_gap_report"][
        "kv_decode_blocker_summary"
    ]
    assert summary == _kv_decode_blocker_summary()
    assert summary["status"] == "blocked"
    assert summary["first_blocker_name"] == "streaming_decode_loop_not_wired"
    assert summary["last_blocker_name"] == _last_streaming_runner_blocker()
    assert summary["kernel_trace_blocker_name"] == _kernel_trace_streaming_runner_blocker()
    assert summary["kernel_trace_blocker_present"] is True
    assert summary["last_blocker"] == _streaming_runner_blockers()[-1]
    assert summary["kernel_trace_blocker"] == _streaming_runner_blockers()[1]
    assert summary["upload_plan_ready"] is True
    assert summary["artifact_count"] == 2
    assert status["kv_backed_decode_gap_report"][
        "kv_decode_blocker_summary_recorded"
    ] is True
    assert status["kv_backed_decode_gap_report"][
        "kv_decode_blocker_summary_mirrors_run_plan"
    ] is True

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
            str(sha_output),
            "--summary-only",
            "--kv-decode-blocker-summary-only",
            "--kv-decode-blocker-summary-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    sha_payload = json.loads(sha_output.read_text())
    assert sha_payload == status["kv_backed_decode_gap_report"][
        "kv_decode_blocker_summary_sha256"
    ]
    assert sha_payload == _kv_decode_blocker_summary_sha256()

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
            str(artifacts_output),
            "--summary-only",
            "--kv-required-artifacts-only",
            "--pretty",
        ]
    )
    assert rc == 0
    artifacts_payload = json.loads(artifacts_output.read_text())
    assert artifacts_payload == _kv_decode_blocker_summary()["artifacts_needed"]
    assert artifacts_payload[0]["name"] == "kv_kernel_trace_artifact"
    assert artifacts_payload[1]["name"] == "kv_backed_next_token_artifact"

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
            str(artifacts_sha_output),
            "--summary-only",
            "--kv-required-artifacts-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    artifacts_sha_payload = json.loads(artifacts_sha_output.read_text())
    assert artifacts_sha_payload == _kv_decode_blocker_summary()["artifacts_needed_sha256"]

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_correctness_status_kv_streaming_loop_next_action_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-loop-next-action.json"
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
            "--kv-streaming-loop-status-only",
            "--kv-streaming-loop-next-action-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["kv_backed_decode_gap_report"][
        "streaming_decode_loop_status"
    ]["next_action"]
    assert payload == "wire_streaming_decode_loop"


def test_stepfun_correctness_status_kv_streaming_loop_next_action_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "kv-streaming-loop-next-action-sha.json"
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
            "--kv-streaming-loop-next-action-only",
            "--kv-streaming-loop-next-action-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    expected = status["kv_backed_decode_gap_report"]["streaming_decode_loop_status"]
    assert payload == expected["next_action_sha256"]
    assert payload == _stable_json_sha256(expected["next_action"])


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
    sha_output = tmp_path / "schema-versions-sha.json"
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
    assert status["schema_versions_sha256"] == _stable_json_sha256(payload)
    assert status["handoff_summary"]["compact_output_modes"][
        "schema_versions_sha_only"
    ] == "schema_versions_sha256"

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
            str(sha_output),
            "--summary-only",
            "--schema-versions-only",
            "--schema-versions-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(sha_output.read_text()) == status["schema_versions_sha256"]


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
            "source_artifacts_compact_output_modes": True,
            "handoff_summary_sha256": True,
            "status_compact_output_modes": True,
            "readiness_summary_sha256": True,
            "readiness_compact_output_modes": True,
            "docs_checklist_sha256": True,
            "docs_checklist_compact_output_modes": True,
            "docs_checklist_count_matches_items": True,
            "readiness_summary_docs_checklist_count_mirror": True,
            "kv_compact_output_modes": True,
            "readiness_gates_sha256": True,
            "next_action_commands_sha256": True,
            "handoff_integrity_command_metadata": True,
            "handoff_integrity_compact_output_modes": True,
            "oracle_progress_sha256": True,
            "oracle_partial_output_handoff_sha256": True,
            "oracle_partial_output_handoff_safe": True,
            "oracle_compact_output_modes": True,
            "blocker_kinds_sha256": True,
            "blocker_kinds_mirror_handoff": True,
            "blocker_kinds_mirror_work_queue": True,
            "blocker_kinds_mirror_remaining_report": True,
            "blocked_gates_sha256": True,
            "blocked_gates_mirror_handoff": True,
            "blocked_gates_mirror_remaining_report": True,
            "kv_streaming_runner_blocker_names_sha256": True,
            "kv_streaming_runner_blocker_names_joined_sha256": True,
            "kv_streaming_runner_blocker_count_present": True,
            "kv_streaming_runner_blocker_mirrors": True,
            "kv_streaming_runner_blockers_sha256": True,
            "kv_streaming_runner_blocker_records_mirrors": True,
            "first_kv_streaming_runner_blocker_sha256": True,
            "first_kv_streaming_runner_blocker_mirrors": True,
            "last_kv_streaming_runner_blocker_sha256": True,
            "last_kv_streaming_runner_blocker_mirrors": True,
            "kernel_trace_kv_streaming_runner_blocker_sha256": True,
            "kernel_trace_kv_streaming_runner_blocker_present": True,
            "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
            "kv_streaming_blueprint_sha256": True,
            "kv_streaming_blueprint_mirrors": True,
            "kv_streaming_loop_status_sha256": True,
            "kv_streaming_loop_status_mirrors": True,
            "kv_streaming_loop_next_action_sha256": True,
            "kv_decode_blocker_summary_sha256": True,
            "kv_decode_blocker_summary_recorded": True,
            "kv_decode_blocker_summary_mirrors_run_plan": True,
            "blocker_work_queue_sha256": True,
            "blocker_work_queue_meta_mirror": True,
            "blocker_work_queue_compact_output_modes": True,
            "first_blocker_work_item_sha256": True,
            "first_blocker_work_item_mirror": True,
            "blocker_recommended_commands_sha256": True,
            "blocker_recommended_commands_mirror_work_queue": True,
            "remaining_blockers_report_sha256": True,
            "first_remaining_blocker_report_sha256": True,
            "first_remaining_blocker_report_mirror": True,
            "blocker_work_queue_command_metadata": True,
            "blocker_recommended_commands_command_metadata": True,
            "blocker_recommended_commands_meta_mirror": True,
            "oracle_partial_output_command_metadata": True,
            "oracle_partial_output_handoff_mirrors": True,
            "status_refresh_atomic_output_command_metadata": True,
            "status_refresh_atomic_output_handoff_mirrors": True,
            "resource_refresh_atomic_output_command_metadata": True,
            "resource_refresh_atomic_output_handoff_mirrors": True,
            "schema_versions": True,
            "schema_versions_sha256": True,
            "schema_versions_compact_output_modes": True,
        },
    }
    assert status["status_integrity"] == payload
    assert status["status_integrity_sha256"] == _stable_json_sha256(payload)


def test_stepfun_correctness_status_status_integrity_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "status-integrity-sha.json"
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
            "--status-integrity-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["status_integrity_sha256"]
    assert payload == _stable_json_sha256(status["status_integrity"])


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


def test_stepfun_correctness_status_persisted_status_integrity_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "persisted-status-integrity.json"
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
            "--status-integrity-only",
            "--persisted-status-integrity-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == {
        "all_match": True,
        "failed_checks": [],
        "checks": {
            "status_integrity_payload": True,
            "status_integrity_sha256": True,
        },
    }


def test_stepfun_correctness_status_persisted_status_integrity_failures_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "persisted-status-integrity-failures.json"
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
            "--persisted-status-integrity-only",
            "--persisted-status-integrity-failures-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == []


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



def test_stepfun_correctness_status_text_resource_source_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    source_output = tmp_path / "text-resource-source.json"
    sha_output = tmp_path / "text-resource-source-sha.json"
    verify_source_output = tmp_path / "verify-text-resource-source.json"
    verify_sha_output = tmp_path / "verify-text-resource-source-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle)
    _write_resource_artifact(resource)
    _write_docs(docs)

    status = build_status(prompt, oracle, docs, resource_artifact=resource)
    expected = status["source_artifacts"]["text_resource"]
    assert isinstance(expected, dict)
    expected_sha = expected["sha256"]

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
            str(source_output),
            "--summary-only",
            "--readiness-summary-only",
            "--source-artifacts-sha-only",
            "--text-resource-source-only",
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(source_output.read_text()) == expected

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
            str(sha_output),
            "--summary-only",
            "--readiness-summary-only",
            "--source-artifacts-sha-only",
            "--text-resource-source-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(sha_output.read_text()) == expected_sha

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
            str(verify_source_output),
            "--text-resource-source-only",
            "--pretty",
        ]
    )
    assert rc == 0
    verified = json.loads(verify_source_output.read_text())
    assert verified["path"] == str(resource)
    assert verified["match"] is True
    assert verified["recorded"] == expected
    assert verified["current"] == expected

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_sha_output),
            "--text-resource-source-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    assert json.loads(verify_sha_output.read_text()) == expected_sha

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_correctness_status_next_action_commands_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "next-action-commands.json"
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
            "--next-action-commands-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]
    assert payload["oracle_parity_blocked"]["oracle_helper_refresh_command"].startswith(
        "python3 scripts/stepfun_llamacpp_oracle.py"
    )
    assert payload["kv_backed_decode_not_wired"][
        "resource_plan_refresh_command"
    ].startswith("python3 scripts/stepfun_gguf_load_smoke.py --dry-run-plan")
    assert payload["handoff_integrity"]["source_artifacts_verify_command"].startswith(
        "python3 scripts/stepfun_correctness_status.py --verify-source-artifacts"
    )


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


def test_stepfun_correctness_status_atomic_output_handoff_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "atomic-output-handoff.json"
    sha_output = tmp_path / "atomic-output-handoff-sha.json"
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
            "--next-action-commands-only",
            "--atomic-output-handoff-only",
            "--pretty",
        ]
    )
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload == status["atomic_output_handoff"]
    assert payload["schema_version"] == 1
    assert payload["status"] == "safe"
    assert payload["all_refresh_outputs_atomic"] is True
    assert payload["status_refresh"]["all_command_records_safe"] is True
    assert payload["status_refresh"]["all_work_queue_mirrors_safe"] is True
    assert payload["resource_plan_refresh"]["command_record_safe"] is True
    assert payload["resource_plan_refresh"]["work_queue_mirror_safe"] is True
    assert payload["status_refresh"]["command_records"][0]["output_helper"] == (
        "stepfun_correctness_status.py"
    )
    assert payload["status_refresh"]["command_records"][0]["uses_shell_redirection"] is False
    assert payload["resource_plan_refresh"]["command_record"]["output_helper"] == (
        "stepfun_gguf_load_smoke.py"
    )
    assert payload["resource_plan_refresh"]["command_record"]["output_path"] == str(
        resource
    )
    assert payload["resource_plan_refresh"]["command_record"][
        "uses_shell_redirection"
    ] is False
    assert payload["integrity_checks"] == [
        "status_refresh_atomic_output_command_metadata",
        "status_refresh_atomic_output_handoff_mirrors",
        "resource_refresh_atomic_output_command_metadata",
        "resource_refresh_atomic_output_handoff_mirrors",
    ]

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
            str(sha_output),
            "--summary-only",
            "--atomic-output-handoff-only",
            "--atomic-output-handoff-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    sha_payload = json.loads(sha_output.read_text())
    assert sha_payload == status["atomic_output_handoff_sha256"]
    assert sha_payload == _stable_json_sha256(status["atomic_output_handoff"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_stepfun_correctness_status_oracle_partial_output_handoff_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-partial-output-handoff.json"
    sha_output = tmp_path / "oracle-partial-output-handoff-sha.json"
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
            "--oracle-partial-output-handoff-only",
            "--pretty",
        ]
    )
    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload == status["oracle_partial_output_handoff"]
    assert payload["schema_version"] == 1
    assert payload["status"] == "safe"
    assert payload["all_partial_output_contracts_safe"] is True
    command_record = payload["command_record"]
    assert command_record["command_kind"] == "oracle_helper_long_timeout_command"
    assert command_record["writes_partial_output_before_launch"] is True
    assert command_record["partial_output_status"] == "running"
    assert command_record["partial_output_path"] == str(oracle)
    assert command_record["partial_output_overwrite_policy"] == (
        "overwrite_on_execute_or_timeout"
    )
    assert command_record["partial_output_blocker_kind"] == (
        "llama_cpp_oracle_in_progress"
    )
    assert command_record["partial_output_matches_oracle_source_path"] is True
    assert command_record["command_has_execute"] is True
    assert command_record["command_has_output_path"] is True
    assert len(payload["mirror_records"]) == 3
    assert payload["all_mirror_records_safe"] is True
    assert {record["source"] for record in payload["mirror_records"]} == {
        "handoff_summary.blocker_work_queue.oracle_parity_blocked",
        "remaining_blockers_report.items.oracle_parity_blocked",
        "first_remaining_blocker_report",
    }
    supervisor_contract = payload["supervisor_signal_timeout_contract"]
    assert supervisor_contract == {
        "handled_signals": ["SIGTERM", "SIGINT"],
        "handler_scope": "while_llama_cli_subprocess_is_running",
        "cleanup_method": "os.killpg",
        "cleanup_signal": "SIGKILL",
        "cleanup_path": "supervisor_signal_killpg_then_communicate",
        "timeout_status": "timeout",
        "timeout_blocker_kind": "llama_cpp_oracle_timeout",
        "timeout_termination_supervisor_signal_received": True,
        "partial_output_overwrite_policy": "overwrite_on_execute_or_timeout",
    }
    assert payload["supervisor_signal_contract_safe"] is True
    assert payload["integrity_checks"] == [
        "oracle_partial_output_command_metadata",
        "oracle_partial_output_handoff_mirrors",
    ]

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
            str(sha_output),
            "--summary-only",
            "--oracle-partial-output-handoff-only",
            "--oracle-partial-output-handoff-sha-only",
            "--pretty",
        ]
    )
    assert rc == 0
    sha_payload = json.loads(sha_output.read_text())
    assert sha_payload == status["oracle_partial_output_handoff_sha256"]
    assert sha_payload == _stable_json_sha256(status["oracle_partial_output_handoff"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


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
            **_recommended_command_fields(
                "oracle_helper_long_timeout_command",
                _oracle_helper_command(prompt, oracle, timeout_s=900.0),
                reason="oracle_timeout_retry_with_longer_timeout",
            ),
            **_oracle_helper_fields(prompt, oracle),
            **_status_refresh_atomic_fields(),
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
                    f"--kv-context-pages 1 --kv-page-size 512 --pretty --output {resource}"
                ),
            ),
            **_recommended_command_fields(
                "resource_plan_refresh_command",
                _resource_plan_refresh_command(resource),
                reason="refresh_kv_resource_and_run_plan_artifact",
            ),
            **_resource_refresh_atomic_fields(resource),
            **_status_refresh_atomic_fields(),
            "first_missing_evidence": "streaming_runner_ready_flags",
            "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
            "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
            "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
            "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
            "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
            "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
            "kernel_trace_streaming_runner_blocker_present": True,
            "streaming_decode_loop_blueprint": _streaming_decode_loop_blueprint_summary(),
            "streaming_decode_loop_blueprint_sha256": _streaming_decode_loop_blueprint_summary_sha256(),
            "streaming_decode_loop_status": _streaming_decode_loop_status_summary(),
            "streaming_decode_loop_status_sha256": _streaming_decode_loop_status_summary_sha256(),
            "gap_report_status": "blocked",
            "operation_count": 135,
            "streaming_runner_blocker_count": 3,
            "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
            "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
            "streaming_runner_blocker_names_sha256_match": True,
            "streaming_runner_blockers": _streaming_runner_blockers(),
            "streaming_runner_blockers_sha256": _streaming_runner_blockers_sha256(),
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
        "first_recommended_command_kind": "oracle_helper_long_timeout_command",
        "first_recommended_command_sha256": hashlib.sha256(
            _oracle_helper_command(prompt, oracle, timeout_s=900.0).encode()
        ).hexdigest(),
        "recommended_commands_sha256": _stable_json_sha256(
            payload["blocker_recommended_commands"]
        ),
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
        "schema_versions_sha_only": "schema_versions_sha256",
        "status_integrity_only": "status_integrity",
        "status_integrity_sha_only": "status_integrity_sha256",
        "status_integrity_failures_only": "status_integrity.failed_checks",
        "persisted_status_integrity_only": "persisted_status_integrity",
        "persisted_status_integrity_failures_only": (
            "persisted_status_integrity.failed_checks"
        ),
        "docs_checklist_only": "docs_checklist",
        "docs_checklist_sha_only": "docs_checklist_sha256",
        "docs_open_partial_count_only": (
            "docs_checklist.open_or_partial_count_p0_p12"
        ),
        "docs_open_partial_summary_only": (
            "docs_checklist.open_or_partial_summary_p0_p12"
        ),
        "docs_open_partial_summary_sha_only": (
            "docs_checklist.open_or_partial_summary_p0_p12_sha256"
        ),
        "docs_open_partial_state_counts_only": (
            "docs_checklist.open_or_partial_state_counts_p0_p12"
        ),
        "docs_open_partial_state_counts_sha_only": (
            "docs_checklist.open_or_partial_state_counts_p0_p12_sha256"
        ),
        "docs_open_partial_lines_only": (
            "docs_checklist.open_or_partial_lines_p0_p12"
        ),
        "docs_open_partial_lines_sha_only": (
            "docs_checklist.open_or_partial_lines_p0_p12_sha256"
        ),
        "docs_open_partial_texts_only": (
            "docs_checklist.open_or_partial_texts_p0_p12"
        ),
        "docs_open_partial_texts_sha_only": (
            "docs_checklist.open_or_partial_texts_p0_p12_sha256"
        ),
        "docs_open_partial_texts_joined_only": (
            "docs_checklist.open_or_partial_texts_joined_p0_p12"
        ),
        "docs_open_partial_texts_joined_sha_only": (
            "docs_checklist.open_or_partial_texts_joined_p0_p12_sha256"
        ),
        "docs_open_partial_line_texts_joined_only": (
            "docs_checklist.open_or_partial_line_texts_joined_p0_p12"
        ),
        "docs_open_partial_line_texts_joined_sha_only": (
            "docs_checklist.open_or_partial_line_texts_joined_p0_p12_sha256"
        ),
        "docs_open_partial_state_line_texts_joined_only": (
            "docs_checklist.open_or_partial_state_line_texts_joined_p0_p12"
        ),
        "docs_open_partial_state_line_texts_joined_sha_only": (
            "docs_checklist.open_or_partial_state_line_texts_joined_p0_p12_sha256"
        ),
        "docs_first_open_partial_item_only": (
            "docs_checklist.first_open_or_partial_item_p0_p12"
        ),
        "docs_first_open_partial_item_sha_only": (
            "docs_checklist.first_open_or_partial_item_p0_p12.sha256"
        ),
        "docs_last_open_partial_item_only": (
            "docs_checklist.last_open_or_partial_item_p0_p12"
        ),
        "docs_last_open_partial_item_sha_only": (
            "docs_checklist.last_open_or_partial_item_p0_p12.sha256"
        ),
        "readiness_summary_only": "readiness_summary",
        "readiness_summary_sha_only": "readiness_summary_sha256",
        "readiness_gates_only": "readiness_gates",
        "readiness_gates_sha_only": "readiness_gates_sha256",
        "blocked_gates_only": "blocked_gates",
        "blocked_gates_sha_only": "blocked_gates_sha256",
        "remaining_blockers_report_only": "remaining_blockers_report",
        "remaining_blockers_report_sha_only": "remaining_blockers_report_sha256",
        "first_remaining_blocker_report_only": "first_remaining_blocker_report",
        "first_remaining_blocker_report_sha_only": "first_remaining_blocker_report_sha256",
        "source_artifacts_sha_only": "source_artifacts_sha256",
        "oracle_wrapper_timeout_source_only": "source_artifacts.oracle_wrapper_timeout",
        "oracle_wrapper_timeout_source_sha_only": (
            "source_artifacts.oracle_wrapper_timeout.sha256"
        ),
        "text_resource_source_only": "source_artifacts.text_resource",
        "text_resource_source_sha_only": "source_artifacts.text_resource.sha256",
        "next_action_commands_only": "next_action_commands",
        "next_action_commands_sha_only": "next_action_commands_sha256",
        "atomic_output_handoff_only": "atomic_output_handoff",
        "atomic_output_handoff_sha_only": "atomic_output_handoff_sha256",
        "oracle_partial_output_handoff_only": "oracle_partial_output_handoff",
        "oracle_partial_output_handoff_sha_only": "oracle_partial_output_handoff_sha256",
        "blocker_kinds_only": "blocker_kinds",
        "blocker_kinds_sha_only": "blocker_kinds_sha256",
        "kv_streaming_blockers_only": "kv_streaming_runner_blocker_names",
        "kv_streaming_blockers_sha_only": "kv_streaming_runner_blocker_names_sha256",
        "kv_streaming_blockers_joined_only": (
            "kv_backed_decode_gap_report.streaming_runner_blocker_names_joined"
        ),
        "kv_streaming_blockers_joined_sha_only": (
            "kv_backed_decode_gap_report.streaming_runner_blocker_names_joined_sha256"
        ),
        "kv_streaming_blocker_count_only": (
            "kv_backed_decode_gap_report.streaming_runner_blocker_count"
        ),
        "kv_streaming_blockers_present_only": (
            "kv_backed_decode_gap_report.streaming_runner_blockers_present"
        ),
        "kv_streaming_blocker_records_only": (
            "kv_backed_decode_gap_report.streaming_runner_blockers"
        ),
        "kv_streaming_blocker_records_sha_only": (
            "kv_backed_decode_gap_report.streaming_runner_blockers_sha256"
        ),
        "kv_first_streaming_blocker_only": (
            "kv_backed_decode_gap_report.first_streaming_runner_blocker"
        ),
        "kv_first_streaming_blocker_sha_only": (
            "kv_backed_decode_gap_report.first_streaming_runner_blocker_sha256"
        ),
        "kv_last_streaming_blocker_only": (
            "kv_backed_decode_gap_report.last_streaming_runner_blocker"
        ),
        "kv_last_streaming_blocker_sha_only": (
            "kv_backed_decode_gap_report.last_streaming_runner_blocker_sha256"
        ),
        "kv_kernel_trace_streaming_blocker_only": (
            "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker"
        ),
        "kv_kernel_trace_streaming_blocker_sha_only": (
            "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker_sha256"
        ),
        "kv_kernel_trace_streaming_blocker_present_only": (
            "kv_backed_decode_gap_report.kernel_trace_streaming_runner_blocker_present"
        ),
        "kv_streaming_blueprint_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_blueprint"
        ),
        "kv_streaming_blueprint_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_blueprint_sha256"
        ),
        "kv_streaming_loop_status_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status"
        ),
        "kv_streaming_loop_status_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status_sha256"
        ),
        "kv_streaming_loop_next_action_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status.next_action"
        ),
        "kv_streaming_loop_next_action_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_loop_status.next_action_sha256"
        ),
        "kv_streaming_launch_trace_only": (
            "kv_backed_decode_gap_report.streaming_decode_launch_trace"
        ),
        "kv_streaming_launch_trace_sha_only": (
            "kv_backed_decode_gap_report.streaming_decode_launch_trace_sha256"
        ),
        "kv_decode_blocker_summary_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary"
        ),
        "kv_decode_blocker_summary_sha_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary_sha256"
        ),
        "kv_required_artifacts_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary.artifacts_needed"
        ),
        "kv_required_artifacts_sha_only": (
            "kv_backed_decode_gap_report.kv_decode_blocker_summary.artifacts_needed_sha256"
        ),
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
        "verification_status_command_only": (
            "next_action_commands.handoff_integrity.verification_status_command"
        ),
        "verification_status_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_status_command_sha256"
        ),
        "verification_exit_code_command_only": (
            "next_action_commands.handoff_integrity.verification_exit_code_command"
        ),
        "verification_exit_code_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_exit_code_command_sha256"
        ),
        "verification_failures_command_only": (
            "next_action_commands.handoff_integrity.verification_failures_command"
        ),
        "verification_failures_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_failures_command_sha256"
        ),
        "verification_failures_sha_command_only": (
            "next_action_commands.handoff_integrity.verification_failures_sha_command"
        ),
        "verification_failures_sha_command_sha_only": (
            "next_action_commands.handoff_integrity.verification_failures_sha_command_sha256"
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
        "oracle_progress_only": "oracle_progress",
        "oracle_progress_sha_only": "oracle_progress_sha256",
        "oracle_helper_long_timeout_command_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command"
        ),
        "oracle_helper_long_timeout_command_sha_only": (
            "next_action_commands.oracle_parity_blocked.oracle_helper_long_timeout_command_sha256"
        ),
        "oracle_timeout_termination_only": "oracle_gap_report.timeout_termination",
        "oracle_timeout_termination_sha_only": (
            "oracle_gap_report.timeout_termination_sha256"
        ),
        "blocker_work_queue_only": "handoff_summary.blocker_work_queue",
        "blocker_work_queue_meta_only": "handoff_summary.blocker_work_queue_meta",
        "blocker_work_queue_sha_only": "handoff_summary.blocker_work_queue_sha256",
        "blocker_recommended_commands_only": "handoff_summary.blocker_recommended_commands",
        "blocker_recommended_commands_sha_only": "handoff_summary.blocker_recommended_commands_sha256",
        "first_blocker_sha_only": "handoff_summary.first_blocker_work_item_sha256",
        "first_blocker_only": "handoff_summary.first_blocker_work_item",
        "first_blocker_recommended_command_only": (
            "handoff_summary.first_blocker_work_item.recommended_command"
        ),
        "first_blocker_recommended_command_sha_only": (
            "handoff_summary.first_blocker_work_item.recommended_command_sha256"
        ),
        "fail_on_blocked_preserves_payload": True,
    }
    assert payload["ready_signals"]["kv_decode_dispatch_ready"] is True
    assert payload["ready_signals"]["kv_decode_run_plan_recorded"] is True
    assert payload["ready_signals"]["kv_decode_input_upload_plan_recorded"] is True
    assert payload["ready_signals"]["kv_streaming_decode_loop_blueprint_recorded"] is True
    assert payload["ready_signals"]["kv_streaming_decode_loop_status_recorded"] is True
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
        "precondition_count": 8,
        "streaming_decode_loop_blueprint": {
            "recorded": True,
            "matches_launch_schedule": True,
            "upload_order_matches": True,
            "blocker_matches": True,
            "executable": False,
            "blocked_by": "streaming_decode_loop_not_wired",
            "blocked_by_sha256": _first_streaming_runner_blocker_sha256(),
            "operation_count": 135,
            "operation_sequence_sha256": _stable_json_sha256(
                _kv_loop_operation_sequence()
            ),
            "stage_count": 4,
            "pre_run_upload_checks_passed": True,
        },
        "streaming_decode_loop_blueprint_sha256": _streaming_decode_loop_blueprint_summary_sha256(),
        "streaming_decode_loop_status": _streaming_decode_loop_status_summary(),
        "streaming_decode_loop_status_sha256": _streaming_decode_loop_status_summary_sha256(),
        "streaming_decode_launch_trace_summary": _streaming_decode_launch_trace_summary(),
        "streaming_decode_launch_trace_summary_sha256": _streaming_decode_launch_trace_summary_sha256(),
        "streaming_decode_launch_trace_sha256": _stable_json_sha256(_streaming_decode_launch_trace()),
        "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
        "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
        "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
        "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
        "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
        "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
        "kernel_trace_streaming_runner_blocker_present": True,
        "status": "blocked",
        "streaming_runner_blocker_count": 3,
        "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
        "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
        "streaming_runner_blocker_names_sha256_match": True,
        "streaming_runner_blockers": _streaming_runner_blockers(),
        "streaming_runner_blockers_sha256": _streaming_runner_blockers_sha256(),
        "upload_total_nbytes": 484,
        "validated_precondition_count": 8,
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
            **_recommended_command_fields(
                "oracle_helper_long_timeout_command",
                _oracle_helper_command(prompt, oracle, timeout_s=900.0),
                reason="oracle_timeout_retry_with_longer_timeout",
            ),
            **_oracle_helper_fields(prompt, oracle),
            **_status_refresh_atomic_fields(),
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
                    f"--kv-context-pages 1 --kv-page-size 512 --pretty --output {resource}"
                ),
            ),
            **_recommended_command_fields(
                "resource_plan_refresh_command",
                _resource_plan_refresh_command(resource),
                reason="refresh_kv_resource_and_run_plan_artifact",
            ),
            **_resource_refresh_atomic_fields(resource),
            **_status_refresh_atomic_fields(),
            "first_missing_evidence": "streaming_runner_ready_flags",
            "first_streaming_runner_blocker": "streaming_decode_loop_not_wired",
            "first_streaming_runner_blocker_sha256": _first_streaming_runner_blocker_sha256(),
            "last_streaming_runner_blocker": _last_streaming_runner_blocker(),
            "last_streaming_runner_blocker_sha256": _last_streaming_runner_blocker_sha256(),
            "kernel_trace_streaming_runner_blocker": _kernel_trace_streaming_runner_blocker(),
            "kernel_trace_streaming_runner_blocker_sha256": _kernel_trace_streaming_runner_blocker_sha256(),
            "kernel_trace_streaming_runner_blocker_present": True,
            "streaming_decode_loop_blueprint": _streaming_decode_loop_blueprint_summary(),
            "streaming_decode_loop_blueprint_sha256": _streaming_decode_loop_blueprint_summary_sha256(),
            "streaming_decode_loop_status": _streaming_decode_loop_status_summary(),
            "streaming_decode_loop_status_sha256": _streaming_decode_loop_status_summary_sha256(),
            "gap_report_status": "blocked",
            "operation_count": 135,
            "streaming_runner_blocker_count": 3,
            "streaming_runner_blocker_names": _streaming_runner_blocker_names(),
            "streaming_runner_blocker_names_sha256": _streaming_runner_blocker_names_sha256(),
            "streaming_runner_blocker_names_sha256_match": True,
            "streaming_runner_blockers": _streaming_runner_blockers(),
            "streaming_runner_blockers_sha256": _streaming_runner_blockers_sha256(),
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


def test_stepfun_correctness_status_blocker_recommended_commands_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-recommended-commands.json"
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
            "--blocker-recommended-commands-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["handoff_summary"]["blocker_recommended_commands"]
    assert payload[0]["blocker_kind"] == "oracle_parity_blocked"
    assert payload[0]["recommended_command_kind"] == "oracle_helper_long_timeout_command"
    assert payload[1]["blocker_kind"] == "kv_backed_decode_not_wired"
    assert payload[1]["recommended_command_kind"] == "resource_plan_refresh_command"


def test_stepfun_correctness_status_blocker_recommended_commands_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "blocker-recommended-commands-sha.json"
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
            "--blocker-recommended-commands-only",
            "--blocker-recommended-commands-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == status["handoff_summary"][
        "blocker_recommended_commands_sha256"
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
    assert f"--output {resource}" in payload


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


def test_stepfun_correctness_status_verification_status_command_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "verification-status-command-sha.json"
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
            "--verification-status-command-only",
            "--verification-status-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    command = status["next_action_commands"]["handoff_integrity"][
        "verification_status_command"
    ]
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_status_command_sha256"
    ]
    assert payload == hashlib.sha256(command.encode()).hexdigest()


def test_stepfun_correctness_status_verification_status_command_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "verification-status-command.json"
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
            "--verification-status-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_status_command"
    ]
    assert payload == _source_verify_command(extra_args=("--verification-status-only",))
    assert "--verification-status-only" in payload


def test_stepfun_correctness_status_verification_exit_code_command_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "verification-exit-code-command-sha.json"
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
            "--verification-exit-code-command-only",
            "--verification-exit-code-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    command = status["next_action_commands"]["handoff_integrity"][
        "verification_exit_code_command"
    ]
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_exit_code_command_sha256"
    ]
    assert payload == hashlib.sha256(command.encode()).hexdigest()


def test_stepfun_correctness_status_verification_exit_code_command_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "verification-exit-code-command.json"
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
            "--verification-exit-code-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_exit_code_command"
    ]
    assert payload == _source_verify_command(extra_args=("--verification-exit-code-only",))
    assert "--verification-exit-code-only" in payload


def test_stepfun_correctness_status_verification_failures_command_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "verification-failures-command-sha.json"
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
            "--verification-failures-command-only",
            "--verification-failures-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    command = status["next_action_commands"]["handoff_integrity"][
        "verification_failures_command"
    ]
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_failures_command_sha256"
    ]
    assert payload == hashlib.sha256(command.encode()).hexdigest()


def test_stepfun_correctness_status_verification_failures_command_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "verification-failures-command.json"
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
            "--verification-failures-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_failures_command"
    ]
    assert payload == _source_verify_command(extra_args=("--verification-failures-only",))
    assert "--verification-failures-only" in payload


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



def test_stepfun_correctness_status_oracle_progress_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-progress.json"
    sha_output = tmp_path / "oracle-progress-sha.json"
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
            "--oracle-progress-only",
            "--pretty",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text())
    assert payload == status["oracle_progress"]
    assert payload["oracle_blocker_kind"] == "llama_cpp_missing_step35_architecture"
    assert payload["expected_next_token_id"] == 369
    assert payload["timeout_s"] == 60.0

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
            str(sha_output),
            "--summary-only",
            "--readiness-summary-only",
            "--oracle-progress-only",
            "--oracle-progress-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    sha_payload = json.loads(sha_output.read_text())
    assert sha_payload == status["oracle_progress_sha256"]
    assert sha_payload == _stable_json_sha256(status["oracle_progress"])



def test_stepfun_correctness_status_oracle_helper_long_timeout_command_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-helper-long-timeout-command.json"
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
            "--oracle-helper-long-timeout-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_long_timeout_command"
    ]
    assert payload == _oracle_helper_command(prompt, oracle, timeout_s=900.0)
    assert "--timeout-s 900.0" in payload



def test_stepfun_correctness_status_oracle_helper_long_timeout_command_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-helper-long-timeout-command-sha.json"
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
            "--oracle-helper-long-timeout-command-only",
            "--oracle-helper-long-timeout-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    expected = _oracle_helper_command(prompt, oracle, timeout_s=900.0)
    assert payload == status["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_long_timeout_command_sha256"
    ]
    assert payload == hashlib.sha256(expected.encode()).hexdigest()



def test_stepfun_correctness_status_oracle_timeout_termination_outputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "oracle-timeout-termination.json"
    sha_output = tmp_path / "oracle-timeout-termination-sha.json"
    _write_prompt_artifact(prompt)
    _write_oracle_artifact(oracle, diagnostic_logs=True)
    _write_resource_artifact(resource)
    _write_docs(docs)
    termination = {
        "timeout_reached": True,
        "timeout_s": 60.0,
        "process_group_started": True,
        "termination_method": "os.killpg",
        "termination_signal": "SIGKILL",
        "termination_signal_number": 9,
        "termination_path": "killpg_sigkill_then_communicate",
        "communicate_after_signal_timeout_s": 10.0,
        "process_exited_before_signal": False,
        "fallback_proc_kill_used": False,
    }
    oracle_payload = json.loads(oracle.read_text())
    oracle_payload.update(
        {
            "status": "timeout",
            "returncode": None,
            "stderr": "",
            "oracle_blocker_kind": "llama_cpp_oracle_timeout",
            "oracle_blocker_detail": "llama.cpp oracle timed out before producing a comparable token",
            "step35_supported": None,
            "timeout_termination": termination,
        }
    )
    oracle.write_text(json.dumps(oracle_payload))

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
            "--oracle-timeout-termination-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text()) == termination
    assert status["handoff_summary"]["compact_output_modes"][
        "oracle_timeout_termination_only"
    ] == "oracle_gap_report.timeout_termination"

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
            str(sha_output),
            "--summary-only",
            "--oracle-timeout-termination-only",
            "--oracle-timeout-termination-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(sha_output.read_text()) == status["oracle_gap_report"][
        "timeout_termination_sha256"
    ]
    assert json.loads(sha_output.read_text()) == _stable_json_sha256(termination)



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
        **_recommended_command_fields(
            "oracle_helper_long_timeout_command",
            _oracle_helper_command(prompt, oracle, timeout_s=900.0),
            reason="oracle_timeout_retry_with_longer_timeout",
        ),
        **_oracle_helper_fields(prompt, oracle),
        **_status_refresh_atomic_fields(),
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


def test_stepfun_correctness_status_first_blocker_recommended_command_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "first-blocker-recommended-command.json"
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
            "--first-blocker-only",
            "--first-blocker-recommended-command-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    expected = _oracle_helper_command(prompt, oracle, timeout_s=900.0)
    assert payload == status["handoff_summary"]["first_blocker_work_item"][
        "recommended_command"
    ]
    assert payload == expected
    assert "--timeout-s 900.0" in payload



def test_stepfun_correctness_status_first_blocker_recommended_command_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    output = tmp_path / "first-blocker-recommended-command-sha.json"
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
            "--first-blocker-recommended-command-only",
            "--first-blocker-recommended-command-sha-only",
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(output.read_text())
    expected = _oracle_helper_command(prompt, oracle, timeout_s=900.0)
    assert payload == status["handoff_summary"]["first_blocker_work_item"][
        "recommended_command_sha256"
    ]
    assert payload == hashlib.sha256(expected.encode()).hexdigest()



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
    assert payload["verification_exit_code"] == 0
    assert payload["all_match"] is True
    assert payload["source_artifacts_all_match"] is True
    assert payload["source_artifact_failed_records"] == []
    assert payload["verification_failures"] == {
        "source_artifact_failed_records": [],
        "status_integrity_failed_checks": [],
        "persisted_status_integrity_failed_checks": [],
    }
    assert payload["verification_failures_sha256"] == _stable_json_sha256(
        payload["verification_failures"]
    )
    assert payload["status_integrity"] == {
        "all_match": True,
        "failed_checks": [],
        "checks": {
            "source_artifacts_sha256": True,
            "source_artifacts_compact_output_modes": True,
            "handoff_summary_sha256": True,
            "status_compact_output_modes": True,
            "readiness_summary_sha256": True,
            "readiness_compact_output_modes": True,
            "docs_checklist_sha256": True,
            "docs_checklist_compact_output_modes": True,
            "docs_checklist_count_matches_items": True,
            "readiness_summary_docs_checklist_count_mirror": True,
            "kv_compact_output_modes": True,
            "readiness_gates_sha256": True,
            "next_action_commands_sha256": True,
            "handoff_integrity_command_metadata": True,
            "handoff_integrity_compact_output_modes": True,
            "oracle_progress_sha256": True,
            "oracle_partial_output_handoff_sha256": True,
            "oracle_partial_output_handoff_safe": True,
            "oracle_compact_output_modes": True,
            "blocker_kinds_sha256": True,
            "blocker_kinds_mirror_handoff": True,
            "blocker_kinds_mirror_work_queue": True,
            "blocker_kinds_mirror_remaining_report": True,
            "blocked_gates_sha256": True,
            "blocked_gates_mirror_handoff": True,
            "blocked_gates_mirror_remaining_report": True,
            "kv_streaming_runner_blocker_names_sha256": True,
            "kv_streaming_runner_blocker_names_joined_sha256": True,
            "kv_streaming_runner_blocker_count_present": True,
            "kv_streaming_runner_blocker_mirrors": True,
            "kv_streaming_runner_blockers_sha256": True,
            "kv_streaming_runner_blocker_records_mirrors": True,
            "first_kv_streaming_runner_blocker_sha256": True,
            "first_kv_streaming_runner_blocker_mirrors": True,
            "last_kv_streaming_runner_blocker_sha256": True,
            "last_kv_streaming_runner_blocker_mirrors": True,
            "kernel_trace_kv_streaming_runner_blocker_sha256": True,
            "kernel_trace_kv_streaming_runner_blocker_present": True,
            "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
            "kv_streaming_blueprint_sha256": True,
            "kv_streaming_blueprint_mirrors": True,
            "kv_streaming_loop_status_sha256": True,
            "kv_streaming_loop_status_mirrors": True,
            "kv_streaming_loop_next_action_sha256": True,
            "kv_decode_blocker_summary_sha256": True,
            "kv_decode_blocker_summary_recorded": True,
            "kv_decode_blocker_summary_mirrors_run_plan": True,
            "blocker_work_queue_sha256": True,
            "blocker_work_queue_meta_mirror": True,
            "blocker_work_queue_compact_output_modes": True,
            "first_blocker_work_item_sha256": True,
            "first_blocker_work_item_mirror": True,
            "blocker_recommended_commands_sha256": True,
            "blocker_recommended_commands_mirror_work_queue": True,
            "remaining_blockers_report_sha256": True,
            "first_remaining_blocker_report_sha256": True,
            "first_remaining_blocker_report_mirror": True,
            "blocker_work_queue_command_metadata": True,
            "blocker_recommended_commands_command_metadata": True,
            "blocker_recommended_commands_meta_mirror": True,
            "oracle_partial_output_command_metadata": True,
            "oracle_partial_output_handoff_mirrors": True,
            "status_refresh_atomic_output_command_metadata": True,
            "status_refresh_atomic_output_handoff_mirrors": True,
            "resource_refresh_atomic_output_command_metadata": True,
            "resource_refresh_atomic_output_handoff_mirrors": True,
            "schema_versions": True,
            "schema_versions_sha256": True,
            "schema_versions_compact_output_modes": True,
        },
    }
    assert payload["persisted_status_integrity"] == {
        "all_match": True,
        "failed_checks": [],
        "checks": {
            "status_integrity_payload": True,
            "status_integrity_sha256": True,
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


def test_stepfun_correctness_status_verify_source_detects_persisted_status_integrity_sha_drift(
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
    status_payload["status_integrity_sha256"] = "stale"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--output",
            str(verify_output),
            "--pretty",
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
    assert payload["status_integrity"]["all_match"] is True
    assert payload["persisted_status_integrity"] == {
        "all_match": False,
        "failed_checks": ["status_integrity_sha256"],
        "checks": {
            "status_integrity_payload": True,
            "status_integrity_sha256": False,
        },
    }
    assert payload["verification_failures"] == {
        "source_artifact_failed_records": [],
        "status_integrity_failed_checks": [],
        "persisted_status_integrity_failed_checks": ["status_integrity_sha256"],
    }


def test_stepfun_correctness_status_verify_source_persisted_status_integrity_failures_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "persisted-failures.json"
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
    status_payload["status_integrity_sha256"] = "stale"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--persisted-status-integrity-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == ["status_integrity_sha256"]


def test_stepfun_correctness_status_verify_source_status_integrity_failures_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "failures.json"
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

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--status-integrity-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == []


def test_stepfun_correctness_status_verify_source_status_integrity_failures_only_detects_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "failures.json"
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
            "--status-integrity-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == ["next_action_commands_sha256"]


def test_stepfun_correctness_status_verify_source_status_integrity_detects_kv_blocker_summary_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "failures.json"
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
    status_payload["kv_backed_decode_gap_report"]["kv_decode_blocker_summary"][
        "first_blocker_name"
    ] = "stale_streaming_blocker"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--status-integrity-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == [
        "kv_decode_blocker_summary_sha256",
        "kv_decode_blocker_summary_mirrors_run_plan",
    ]


def test_stepfun_correctness_status_verify_source_status_integrity_detects_oracle_partial_output_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "failures.json"
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
    status_payload["next_action_commands"]["oracle_parity_blocked"][
        "oracle_helper_partial_output_status"
    ] = "stale_running_status"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--status-integrity-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == [
        "next_action_commands_sha256",
        "oracle_partial_output_command_metadata",
        "oracle_partial_output_handoff_mirrors",
    ]


def test_stepfun_correctness_status_verify_source_status_integrity_detects_resource_atomic_output_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "failures.json"
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
        "resource_plan_refresh_output_arg_present"
    ] = False
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--status-integrity-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == [
        "next_action_commands_sha256",
        "resource_refresh_atomic_output_command_metadata",
        "resource_refresh_atomic_output_handoff_mirrors",
    ]


def test_stepfun_correctness_status_verify_source_artifact_failures_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "source-failures.json"
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

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--source-artifact-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == []


def test_stepfun_correctness_status_verify_source_artifact_failures_only_detects_stale_inputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "source-failures.json"
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
            "--source-artifact-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(failures_output.read_text()) == ["prompt"]


def test_stepfun_correctness_status_verify_status_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_status_output = tmp_path / "verification-status.json"
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

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--verification-status-only",
            "--output",
            str(verify_status_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(verify_status_output.read_text()) == "match"


def test_stepfun_correctness_status_verify_status_only_detects_stale_inputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    verify_status_output = tmp_path / "verification-status.json"
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
            "--verification-status-only",
            "--output",
            str(verify_status_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(verify_status_output.read_text()) == "mismatch"


def test_stepfun_correctness_status_verify_exit_code_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    exit_code_output = tmp_path / "verification-exit-code.json"
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

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--verification-exit-code-only",
            "--output",
            str(exit_code_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(exit_code_output.read_text()) == 0


def test_stepfun_correctness_status_verify_exit_code_only_detects_stale_inputs(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    exit_code_output = tmp_path / "verification-exit-code.json"
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
            "--verification-exit-code-only",
            "--output",
            str(exit_code_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(exit_code_output.read_text()) == 1


def test_stepfun_correctness_status_verify_failures_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "verification-failures.json"
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

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--verification-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(failures_output.read_text())
    assert payload == {
        "source_artifact_failed_records": [],
        "status_integrity_failed_checks": [],
        "persisted_status_integrity_failed_checks": [],
    }


def test_stepfun_correctness_status_verify_failures_sha_only(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    sha_output = tmp_path / "verification-failures-sha.json"
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

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--verification-failures-sha-only",
            "--output",
            str(sha_output),
            "--pretty",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(sha_output.read_text()) == _stable_json_sha256(
        {
            "source_artifact_failed_records": [],
            "status_integrity_failed_checks": [],
            "persisted_status_integrity_failed_checks": [],
        }
    )


def test_stepfun_correctness_status_verify_failures_only_detects_source_and_status_drift(
    capsys,
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.json"
    oracle = tmp_path / "oracle.json"
    docs = tmp_path / "STEPFUN.md"
    resource = tmp_path / "resource.json"
    status_output = tmp_path / "status.json"
    failures_output = tmp_path / "verification-failures.json"
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
    status_payload = json.loads(status_output.read_text())
    status_payload["next_action_commands_sha256"] = "stale"
    status_output.write_text(json.dumps(status_payload))

    rc = main(
        [
            "--verify-source-artifacts",
            str(status_output),
            "--verification-failures-only",
            "--output",
            str(failures_output),
            "--pretty",
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    payload = json.loads(failures_output.read_text())
    assert payload == {
        "source_artifact_failed_records": ["prompt"],
        "status_integrity_failed_checks": ["next_action_commands_sha256"],
        "persisted_status_integrity_failed_checks": [
            "status_integrity_payload",
            "status_integrity_sha256",
        ],
    }


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
    assert payload["source_artifact_failed_records"] == ["prompt"]
    assert payload["status_integrity"]["all_match"] is True
    assert payload["records"]["prompt"]["match"] is False
    assert payload["records"]["prompt"]["matches"] == {
        "exists": True,
        "sha256": False,
        "size_bytes": False,
    }
    assert payload["records"]["oracle"]["match"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_status_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["status_integrity_failures_only"] = "stale.status.failures"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == ["status_compact_output_modes"]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["status_compact_output_modes"] is False
    assert checks["readiness_summary_sha256"] is True
    assert checks["source_artifacts_compact_output_modes"] is True
    assert checks["schema_versions_compact_output_modes"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_source_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["text_resource_source_sha_only"] = "stale.text_resource.sha"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "source_artifacts_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["source_artifacts_sha256"] is True
    assert checks["source_artifacts_compact_output_modes"] is False
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["readiness_compact_output_modes"] is True
    assert checks["schema_versions_compact_output_modes"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_readiness_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["readiness_summary_sha_only"] = "stale.readiness.sha"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "readiness_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["readiness_compact_output_modes"] is False
    assert checks["source_artifacts_compact_output_modes"] is True
    assert checks["docs_checklist_compact_output_modes"] is True
    assert checks["schema_versions_compact_output_modes"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["kv_streaming_loop_status_sha_only"] = "stale.kv.loop.status.sha"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == ["kv_compact_output_modes"]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["kv_compact_output_modes"] is False
    assert checks["readiness_compact_output_modes"] is True
    assert checks["docs_checklist_compact_output_modes"] is True
    assert checks["schema_versions_compact_output_modes"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_schema_versions_digest_drift(
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
    status_payload["schema_versions_sha256"] = "stale"
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
    assert payload["status_integrity"]["failed_checks"] == ["schema_versions_sha256"]
    checks = payload["status_integrity"]["checks"]
    assert checks["source_artifacts_sha256"] is True
    assert checks["handoff_summary_sha256"] is True
    assert checks["schema_versions"] is True
    assert checks["schema_versions_sha256"] is False
    assert checks["schema_versions_compact_output_modes"] is True
    assert checks["readiness_summary_sha256"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_schema_versions_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["schema_versions_sha_only"] = "stale.schema.digest"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "schema_versions_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["schema_versions"] is True
    assert checks["schema_versions_sha256"] is True
    assert checks["schema_versions_compact_output_modes"] is False


def test_stepfun_correctness_status_source_artifact_verify_detects_docs_checklist_digest_drift(
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
    status_payload["docs_checklist_sha256"] = "stale"
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
    assert payload["status_integrity"]["failed_checks"] == ["docs_checklist_sha256"]
    checks = payload["status_integrity"]["checks"]
    assert checks["source_artifacts_sha256"] is True
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["docs_checklist_sha256"] is False
    assert checks["docs_checklist_compact_output_modes"] is True
    assert checks["docs_checklist_count_matches_items"] is True
    assert checks["readiness_summary_docs_checklist_count_mirror"] is True
    assert checks["readiness_gates_sha256"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_docs_checklist_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["docs_open_partial_count_only"] = "stale.docs.metric"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "docs_checklist_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["docs_checklist_sha256"] is True
    assert checks["docs_checklist_compact_output_modes"] is False
    assert checks["docs_checklist_count_matches_items"] is True
    assert checks["readiness_summary_docs_checklist_count_mirror"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_readiness_docs_metric_mirror_drift(
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
    status_payload["readiness_summary"]["open_or_partial_items_p0_p12"] = 3
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "readiness_summary_docs_checklist_count_mirror"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["readiness_summary_sha256"] is True
    assert checks["docs_checklist_sha256"] is True
    assert checks["docs_checklist_count_matches_items"] is True
    assert checks["readiness_summary_docs_checklist_count_mirror"] is False
    assert checks["readiness_gates_sha256"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_docs_checklist_count_drift(
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
    status_payload["docs_checklist"]["open_or_partial_count_p0_p12"] = 3
    status_payload["docs_checklist_sha256"] = _stable_json_sha256(
        status_payload["docs_checklist"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "docs_checklist_count_matches_items",
        "readiness_summary_docs_checklist_count_mirror",
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["readiness_summary_sha256"] is True
    assert checks["docs_checklist_sha256"] is True
    assert checks["docs_checklist_count_matches_items"] is False
    assert checks["readiness_summary_docs_checklist_count_mirror"] is False
    assert checks["readiness_gates_sha256"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_blocker_kind_mirror_drift(
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
    status_payload["blocker_kinds"] = [
        "kv_backed_decode_not_wired",
        "oracle_parity_blocked",
    ]
    status_payload["blocker_kinds_sha256"] = _stable_json_sha256(
        status_payload["blocker_kinds"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "blocker_kinds_mirror_handoff",
        "blocker_kinds_mirror_work_queue",
        "blocker_kinds_mirror_remaining_report",
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["blocker_kinds_sha256"] is True
    assert checks["blocker_kinds_mirror_handoff"] is False
    assert checks["blocker_kinds_mirror_work_queue"] is False
    assert checks["blocker_kinds_mirror_remaining_report"] is False
    assert checks["blocked_gates_sha256"] is True
    assert checks["blocked_gates_mirror_handoff"] is True
    assert checks["blocked_gates_mirror_remaining_report"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_blocked_gates_mirror_drift(
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
    status_payload["blocked_gates"] = ["oracle_parity"]
    status_payload["blocked_gates_sha256"] = _stable_json_sha256(
        status_payload["blocked_gates"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "blocked_gates_mirror_handoff",
        "blocked_gates_mirror_remaining_report",
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["blocker_kinds_sha256"] is True
    assert checks["blocker_kinds_mirror_handoff"] is True
    assert checks["blocker_kinds_mirror_work_queue"] is True
    assert checks["blocker_kinds_mirror_remaining_report"] is True
    assert checks["blocked_gates_sha256"] is True
    assert checks["blocked_gates_mirror_handoff"] is False
    assert checks["blocked_gates_mirror_remaining_report"] is False


def test_stepfun_correctness_status_source_artifact_verify_detects_oracle_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["oracle_timeout_termination_sha_only"] = "stale.oracle.timeout.sha"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "oracle_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["oracle_compact_output_modes"] is False
    assert checks["handoff_integrity_command_metadata"] is True
    assert checks["handoff_integrity_compact_output_modes"] is True
    assert checks["schema_versions_compact_output_modes"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_handoff_integrity_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["verification_failures_sha_command_sha_only"] = (
        "stale.verification.failures.sha.command.sha"
    )
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "handoff_integrity_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["handoff_integrity_command_metadata"] is True
    assert checks["handoff_integrity_compact_output_modes"] is False
    assert checks["oracle_compact_output_modes"] is True
    assert checks["status_compact_output_modes"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_handoff_integrity_command_metadata_drift(
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
    handoff_integrity = status_payload["next_action_commands"]["handoff_integrity"]
    handoff_integrity["verification_status_command_sha256"] = "stale"
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
    assert payload["status_integrity"]["failed_checks"] == [
        "handoff_integrity_command_metadata"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["next_action_commands_sha256"] is True
    assert checks["handoff_integrity_command_metadata"] is False
    assert checks["oracle_compact_output_modes"] is True
    assert checks["blocker_work_queue_command_metadata"] is True
    assert checks["blocker_recommended_commands_command_metadata"] is True
    assert checks["schema_versions_sha256"] is True


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
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": False,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": True,
        "first_kv_streaming_runner_blocker_mirrors": True,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": True,
        "kv_streaming_blueprint_mirrors": True,
        "kv_streaming_loop_status_sha256": True,
        "kv_streaming_loop_status_mirrors": True,
        "kv_streaming_loop_next_action_sha256": True,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": True,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
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
        "kv_decode_blocker_summary_mirrors_run_plan",
    ]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": False,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": False,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": True,
        "first_kv_streaming_runner_blocker_mirrors": True,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": True,
        "kv_streaming_blueprint_mirrors": True,
        "kv_streaming_loop_status_sha256": True,
        "kv_streaming_loop_status_mirrors": True,
        "kv_streaming_loop_next_action_sha256": True,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": False,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
    }


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_streaming_record_digest_drift(
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
        "streaming_runner_blockers_sha256"
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
    assert payload["status_integrity"]["failed_checks"] == [
        "kv_streaming_runner_blockers_sha256",
        "kv_streaming_runner_blocker_records_mirrors",
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["kv_streaming_runner_blocker_names_sha256"] is True
    assert checks["kv_streaming_runner_blocker_mirrors"] is True
    assert checks["kv_streaming_runner_blockers_sha256"] is False
    assert checks["kv_streaming_runner_blocker_records_mirrors"] is False
    assert checks["kv_decode_blocker_summary_mirrors_run_plan"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_streaming_record_mirror_drift(
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
        "streaming_runner_blockers"
    ][0]["required_evidence"] = "stale required evidence"
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
    assert payload["status_integrity"]["failed_checks"] == [
        "kv_streaming_runner_blocker_records_mirrors"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["next_action_commands_sha256"] is True
    assert checks["kv_streaming_runner_blocker_names_sha256"] is True
    assert checks["kv_streaming_runner_blocker_mirrors"] is True
    assert checks["kv_streaming_runner_blockers_sha256"] is True
    assert checks["kv_streaming_runner_blocker_records_mirrors"] is False
    assert checks["first_kv_streaming_runner_blocker_sha256"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_first_kv_streaming_digest_drift(
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
        "first_streaming_runner_blocker_sha256"
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
        "first_kv_streaming_runner_blocker_sha256",
        "first_kv_streaming_runner_blocker_mirrors",
    ]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": False,
        "first_kv_streaming_runner_blocker_mirrors": False,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": True,
        "kv_streaming_blueprint_mirrors": True,
        "kv_streaming_loop_status_sha256": True,
        "kv_streaming_loop_status_mirrors": True,
        "kv_streaming_loop_next_action_sha256": True,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": True,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
    }


def test_stepfun_correctness_status_source_artifact_verify_detects_recommended_commands_digest_drift(
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
    status_payload["handoff_summary"]["blocker_recommended_commands_sha256"] = "stale"
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
        "handoff_summary_sha256",
        "blocker_work_queue_meta_mirror",
        "blocker_recommended_commands_sha256",
        "blocker_recommended_commands_meta_mirror",
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["blocker_work_queue_sha256"] is True
    assert checks["blocker_work_queue_meta_mirror"] is False
    assert checks["blocker_work_queue_compact_output_modes"] is True
    assert checks["first_blocker_work_item_sha256"] is True
    assert checks["first_blocker_work_item_mirror"] is True
    assert checks["blocker_recommended_commands_sha256"] is False
    assert checks["blocker_recommended_commands_mirror_work_queue"] is True
    assert checks["blocker_work_queue_command_metadata"] is True
    assert checks["blocker_recommended_commands_command_metadata"] is True
    assert checks["blocker_recommended_commands_meta_mirror"] is False
    assert checks["schema_versions"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_blocker_queue_compact_mode_drift(
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
    compact_modes = status_payload["handoff_summary"]["compact_output_modes"]
    compact_modes["first_blocker_recommended_command_sha_only"] = "stale.first.command.sha"
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(
        status_payload["handoff_summary"]
    )
    status_payload["readiness_summary"]["handoff_summary_sha256"] = status_payload[
        "handoff_summary_sha256"
    ]
    status_payload["readiness_summary_sha256"] = _stable_json_sha256(
        status_payload["readiness_summary"]
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
    assert payload["status_integrity"]["failed_checks"] == [
        "blocker_work_queue_compact_output_modes"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["readiness_summary_sha256"] is True
    assert checks["blocker_work_queue_sha256"] is True
    assert checks["blocker_work_queue_meta_mirror"] is True
    assert checks["blocker_work_queue_compact_output_modes"] is False
    assert checks["first_blocker_work_item_sha256"] is True
    assert checks["first_blocker_work_item_mirror"] is True
    assert checks["blocker_recommended_commands_mirror_work_queue"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_recommended_command_metadata_drift(
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
    handoff = status_payload["handoff_summary"]
    queue = handoff["blocker_work_queue"]
    compact = handoff["blocker_recommended_commands"]
    queue[0]["recommended_command_sha256"] = "stale"
    compact[0]["recommended_command_sha256"] = "stale"
    handoff["first_blocker_work_item"] = queue[0]
    handoff["first_blocker_work_item_sha256"] = _stable_json_sha256(queue[0])
    handoff["blocker_work_queue_sha256"] = _stable_json_sha256(queue)
    handoff["blocker_recommended_commands_sha256"] = _stable_json_sha256(compact)
    meta = handoff["blocker_work_queue_meta"]
    meta["sha256"] = handoff["blocker_work_queue_sha256"]
    meta["first_work_item_sha256"] = handoff["first_blocker_work_item_sha256"]
    meta["first_recommended_command_sha256"] = "stale"
    meta["recommended_commands_sha256"] = handoff["blocker_recommended_commands_sha256"]
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(handoff)
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
    assert payload["status_integrity"]["failed_checks"] == [
        "blocker_work_queue_command_metadata",
        "blocker_recommended_commands_command_metadata",
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["blocker_work_queue_sha256"] is True
    assert checks["blocker_work_queue_meta_mirror"] is True
    assert checks["blocker_work_queue_compact_output_modes"] is True
    assert checks["first_blocker_work_item_sha256"] is True
    assert checks["first_blocker_work_item_mirror"] is True
    assert checks["blocker_recommended_commands_sha256"] is True
    assert checks["blocker_recommended_commands_mirror_work_queue"] is True
    assert checks["blocker_work_queue_command_metadata"] is False
    assert checks["blocker_recommended_commands_command_metadata"] is False
    assert checks["blocker_recommended_commands_meta_mirror"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_recommended_command_mirror_drift(
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
    stale_command = "python3 stale-oracle-command.py --execute"
    queue = status_payload["handoff_summary"]["blocker_work_queue"]
    queue[0]["recommended_command"] = stale_command
    queue[0]["recommended_command_nchars"] = len(stale_command)
    queue[0]["recommended_command_sha256"] = hashlib.sha256(
        stale_command.encode()
    ).hexdigest()
    first_item = queue[0]
    handoff = status_payload["handoff_summary"]
    handoff["first_blocker_work_item"] = first_item
    handoff["first_blocker_work_item_sha256"] = _stable_json_sha256(first_item)
    handoff["blocker_work_queue_sha256"] = _stable_json_sha256(queue)
    meta = handoff["blocker_work_queue_meta"]
    meta["sha256"] = handoff["blocker_work_queue_sha256"]
    meta["first_work_item_sha256"] = handoff["first_blocker_work_item_sha256"]
    meta["first_recommended_command_sha256"] = first_item["recommended_command_sha256"]
    status_payload["handoff_summary_sha256"] = _stable_json_sha256(handoff)
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
    assert payload["status_integrity"]["failed_checks"] == [
        "blocker_recommended_commands_mirror_work_queue"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["handoff_summary_sha256"] is True
    assert checks["blocker_work_queue_sha256"] is True
    assert checks["blocker_work_queue_meta_mirror"] is True
    assert checks["blocker_work_queue_compact_output_modes"] is True
    assert checks["first_blocker_work_item_sha256"] is True
    assert checks["first_blocker_work_item_mirror"] is True
    assert checks["blocker_recommended_commands_sha256"] is True
    assert checks["blocker_recommended_commands_mirror_work_queue"] is False
    assert checks["blocker_work_queue_command_metadata"] is True
    assert checks["blocker_recommended_commands_command_metadata"] is True
    assert checks["blocker_recommended_commands_meta_mirror"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_remaining_blockers_digest_drift(
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
    status_payload["remaining_blockers_report_sha256"] = "stale"
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
    assert payload["status_integrity"]["failed_checks"] == [
        "remaining_blockers_report_sha256"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["remaining_blockers_report_sha256"] is False
    assert checks["first_remaining_blocker_report_sha256"] is True
    assert checks["first_remaining_blocker_report_mirror"] is True
    assert checks["blocker_recommended_commands_mirror_work_queue"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_first_remaining_blocker_mirror_drift(
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
    first_report = status_payload["first_remaining_blocker_report"]
    first_report["recommended_command"] = "python3 stale-front-blocker.py --execute"
    status_payload["first_remaining_blocker_report_sha256"] = _stable_json_sha256(
        first_report
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
    assert payload["status_integrity"]["failed_checks"] == [
        "first_remaining_blocker_report_mirror"
    ]
    checks = payload["status_integrity"]["checks"]
    assert checks["remaining_blockers_report_sha256"] is True
    assert checks["first_remaining_blocker_report_sha256"] is True
    assert checks["first_remaining_blocker_report_mirror"] is False
    assert checks["oracle_partial_output_handoff_mirrors"] is True


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_blueprint_digest_drift(
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
        "streaming_decode_loop_blueprint_sha256"
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
        "kv_streaming_blueprint_sha256",
        "kv_streaming_blueprint_mirrors",
    ]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": True,
        "first_kv_streaming_runner_blocker_mirrors": True,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": False,
        "kv_streaming_blueprint_mirrors": False,
        "kv_streaming_loop_status_sha256": True,
        "kv_streaming_loop_status_mirrors": True,
        "kv_streaming_loop_next_action_sha256": True,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": True,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
    }


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_loop_status_digest_drift(
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
        "streaming_decode_loop_status_sha256"
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
        "kv_streaming_loop_status_sha256",
        "kv_streaming_loop_status_mirrors",
    ]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": True,
        "first_kv_streaming_runner_blocker_mirrors": True,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": True,
        "kv_streaming_blueprint_mirrors": True,
        "kv_streaming_loop_status_sha256": False,
        "kv_streaming_loop_status_mirrors": False,
        "kv_streaming_loop_next_action_sha256": True,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": True,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
    }


def test_stepfun_correctness_status_source_artifact_verify_detects_kv_loop_next_action_digest_drift(
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
    status_payload["kv_backed_decode_gap_report"]["streaming_decode_loop_status"][
        "next_action_sha256"
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
        "kv_streaming_loop_status_sha256",
        "kv_streaming_loop_status_mirrors",
        "kv_streaming_loop_next_action_sha256",
    ]
    assert payload["status_integrity"]["checks"] == {
        "source_artifacts_sha256": True,
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": True,
        "first_kv_streaming_runner_blocker_mirrors": True,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": True,
        "kv_streaming_blueprint_mirrors": True,
        "kv_streaming_loop_status_sha256": False,
        "kv_streaming_loop_status_mirrors": False,
        "kv_streaming_loop_next_action_sha256": False,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": True,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
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
        "source_artifacts_compact_output_modes": True,
        "handoff_summary_sha256": True,
        "status_compact_output_modes": True,
        "readiness_summary_sha256": True,
        "readiness_compact_output_modes": True,
        "docs_checklist_sha256": True,
        "docs_checklist_compact_output_modes": True,
        "docs_checklist_count_matches_items": True,
        "readiness_summary_docs_checklist_count_mirror": True,
        "kv_compact_output_modes": True,
        "readiness_gates_sha256": True,
        "next_action_commands_sha256": True,
        "handoff_integrity_command_metadata": True,
        "handoff_integrity_compact_output_modes": True,
        "oracle_progress_sha256": True,
        "oracle_partial_output_handoff_sha256": True,
        "oracle_partial_output_handoff_safe": True,
        "oracle_compact_output_modes": True,
        "blocker_kinds_sha256": True,
        "blocker_kinds_mirror_handoff": True,
        "blocker_kinds_mirror_work_queue": True,
        "blocker_kinds_mirror_remaining_report": True,
        "blocked_gates_sha256": True,
        "blocked_gates_mirror_handoff": True,
        "blocked_gates_mirror_remaining_report": True,
        "kv_streaming_runner_blocker_names_sha256": True,
        "kv_streaming_runner_blocker_names_joined_sha256": True,
        "kv_streaming_runner_blocker_count_present": True,
        "kv_streaming_runner_blocker_mirrors": False,
        "kv_streaming_runner_blockers_sha256": True,
        "kv_streaming_runner_blocker_records_mirrors": True,
        "first_kv_streaming_runner_blocker_sha256": True,
        "first_kv_streaming_runner_blocker_mirrors": True,
        "last_kv_streaming_runner_blocker_sha256": True,
        "last_kv_streaming_runner_blocker_mirrors": True,
        "kernel_trace_kv_streaming_runner_blocker_sha256": True,
        "kernel_trace_kv_streaming_runner_blocker_present": True,
        "kernel_trace_kv_streaming_runner_blocker_mirrors": True,
        "kv_streaming_blueprint_sha256": True,
        "kv_streaming_blueprint_mirrors": True,
        "kv_streaming_loop_status_sha256": True,
        "kv_streaming_loop_status_mirrors": True,
        "kv_streaming_loop_next_action_sha256": True,
        "kv_decode_blocker_summary_sha256": True,
        "kv_decode_blocker_summary_recorded": True,
        "kv_decode_blocker_summary_mirrors_run_plan": True,
        "blocker_work_queue_sha256": True,
        "blocker_work_queue_meta_mirror": True,
        "blocker_work_queue_compact_output_modes": True,
        "first_blocker_work_item_sha256": True,
        "first_blocker_work_item_mirror": True,
        "blocker_recommended_commands_sha256": True,
        "blocker_recommended_commands_mirror_work_queue": True,
        "remaining_blockers_report_sha256": True,
        "first_remaining_blocker_report_sha256": True,
        "first_remaining_blocker_report_mirror": True,
        "blocker_work_queue_command_metadata": True,
        "blocker_recommended_commands_command_metadata": True,
        "blocker_recommended_commands_meta_mirror": True,
        "oracle_partial_output_command_metadata": True,
        "oracle_partial_output_handoff_mirrors": True,
        "status_refresh_atomic_output_command_metadata": True,
        "status_refresh_atomic_output_handoff_mirrors": True,
        "resource_refresh_atomic_output_command_metadata": True,
        "resource_refresh_atomic_output_handoff_mirrors": True,
        "schema_versions": True,
        "schema_versions_sha256": True,
        "schema_versions_compact_output_modes": True,
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


def test_stepfun_correctness_status_verification_status_command_fail_on_blocked_returns_nonzero(
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
            "--verification-status-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_status_command"
    ]
    assert payload == _source_verify_command(extra_args=("--verification-status-only",))


def test_stepfun_correctness_status_verification_status_command_sha_fail_on_blocked_returns_nonzero(
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
            "--verification-status-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_status_command_sha256"
    ]
    assert payload == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-status-only",)).encode()
    ).hexdigest()


def test_stepfun_correctness_status_verification_exit_code_command_fail_on_blocked_returns_nonzero(
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
            "--verification-exit-code-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_exit_code_command"
    ]
    assert payload == _source_verify_command(extra_args=("--verification-exit-code-only",))


def test_stepfun_correctness_status_verification_exit_code_command_sha_fail_on_blocked_returns_nonzero(
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
            "--verification-exit-code-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_exit_code_command_sha256"
    ]
    assert payload == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-exit-code-only",)).encode()
    ).hexdigest()


def test_stepfun_correctness_status_verification_failures_command_fail_on_blocked_returns_nonzero(
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
            "--verification-failures-command-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_failures_command"
    ]
    assert payload == _source_verify_command(extra_args=("--verification-failures-only",))


def test_stepfun_correctness_status_verification_failures_command_sha_fail_on_blocked_returns_nonzero(
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
            "--verification-failures-command-sha-only",
            "--fail-on-blocked",
            "--pretty",
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload == status["next_action_commands"]["handoff_integrity"][
        "verification_failures_command_sha256"
    ]
    assert payload == hashlib.sha256(
        _source_verify_command(extra_args=("--verification-failures-only",)).encode()
    ).hexdigest()


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
