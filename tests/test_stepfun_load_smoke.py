from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.stepfun_gguf_load_smoke import main

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _stepfun_gguf_dir() -> Path:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return root


def test_stepfun_load_smoke_dry_run_plan_emits_resource_json(capsys: pytest.CaptureFixture[str]) -> None:
    root = _stepfun_gguf_dir()

    rc = main(
        [
            "--dry-run-plan",
            "--model-dir",
            str(root),
            "--kv-context-pages",
            "1",
            "--kv-page-size",
            "512",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "planned"
    assert payload["snapshots"] == []
    assert payload["loaded_weight_count"] == 0
    assert payload["loaded_nbytes"] == 0
    assert payload["tensor_count"] == 754
    assert payload["plan_total_nbytes"] == 102_499_149_312
    assert payload["kv_nbytes"] == 94_371_840
    plan = payload["text_decode_resource_plan"]
    assert plan["backend"] == "hip_gfx1151"
    assert plan["slot_count"] == 754
    assert plan["resident_weight_nbytes"] == payload["plan_total_nbytes"]
    assert plan["kv_buffer_count"] == 90
    assert plan["kv_layer_nbytes"][0] == {
        "layer": 0,
        "key_nbytes": 1_048_576,
        "value_nbytes": 1_048_576,
    }
    assert plan["kv_nbytes"] == payload["kv_nbytes"]
    kv_plan = plan["kv_decode_kernel_plan"]
    assert kv_plan["model_quant"] == "gguf_step35"
    assert kv_plan["kv_storage_dtype"] == "bf16"
    assert kv_plan["decode_attention_kind"] == "splitk_gate_f32"
    assert kv_plan["max_context"] == 512
    assert kv_plan["max_new_tokens"] == 1
    assert kv_plan["max_prompt_rows"] == 511
    assert kv_plan["attention_block_size"] == 256
    assert kv_plan["attention_block_table_len"] == 2
    assert kv_plan["attention_capacity_tokens"] == 512
    assert kv_plan["decode_span"]["max_live_count"] == 511
    assert kv_plan["prompt_span"]["block_table_len_per_row"] == 2
    assert kv_plan["prompt_span"]["base_offsets_len_formula"] == "rows * 2"
    assert kv_plan["decode_span_shape_compatible"] is True
    assert kv_plan["prompt_span_shape_compatible"] is True
    assert kv_plan["span_shape_compatible"] is True
    assert kv_plan["all_registered"] is True
    assert kv_plan["dispatch_keys"]["decode_attention"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_attn_decode",
        "quant": "gguf_step35",
        "variant": "bf16_split_k_gate_f32_spans",
    }
    launch_schedule = plan["kv_decode_launch_schedule"]
    assert launch_schedule["layer_count"] == 45
    assert launch_schedule["operation_count"] == 135
    assert launch_schedule["per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert launch_schedule["all_stage_dispatch_ready"] is True
    assert launch_schedule["streaming_runner_ready"] is False
    assert launch_schedule["stages"][2] == {
        "name": "one_token_gated_attention_decode",
        "dispatch_key": "decode_attention",
        "span_contract": "decode_span",
        "layer_count": 45,
        "ready": True,
    }
    run_plan = payload["kv_decode_run_plan"]
    assert run_plan["prompt_length"] > 0
    assert run_plan["input_id_count"] == run_plan["prompt_length"]
    assert len(run_plan["input_ids"]) == run_plan["input_id_count"]
    assert all(isinstance(token_id, int) for token_id in run_plan["input_ids"])
    assert run_plan["input_ids_dtype"] == "int32"
    assert run_plan["input_ids_nbytes"] == run_plan["prompt_length"] * 4
    assert len(run_plan["input_ids_sha256"]) == 64
    assert int(run_plan["input_ids_sha256"], 16) >= 0
    assert run_plan["input_id_preview"] == run_plan["input_ids"][:8]
    assert run_plan["rendered_prompt_nchars"] > 0
    assert len(run_plan["rendered_prompt_sha256"]) == 64
    assert int(run_plan["rendered_prompt_sha256"], 16) >= 0
    assert run_plan["max_new_tokens"] == 1
    assert run_plan["required_context_tokens"] == run_plan["prompt_length"] + 1
    assert run_plan["max_context"] == 512
    assert run_plan["max_prompt_rows"] == 511
    assert run_plan["prompt_positions"] == list(range(run_plan["prompt_length"]))
    assert run_plan["decode_position"] == run_plan["prompt_length"]
    assert run_plan["decode_live_count"] == run_plan["prompt_length"]
    assert run_plan["attention_block_size"] == 256
    assert run_plan["attention_block_table_len"] == 2
    assert run_plan["prompt_span_inputs"]["rows"] == run_plan["prompt_length"]
    assert run_plan["prompt_span_inputs"]["base_offsets_dtype"] == "int32"
    assert run_plan["prompt_span_inputs"]["base_offsets_len"] == run_plan["prompt_length"] * 2
    assert run_plan["prompt_span_inputs"]["base_offsets_nbytes"] == run_plan["prompt_length"] * 8
    assert run_plan["prompt_span_inputs"]["live_counts"] == run_plan["prompt_positions"]
    assert run_plan["prompt_span_inputs"]["live_counts_dtype"] == "int64"
    assert run_plan["prompt_span_inputs"]["live_counts_nbytes"] == run_plan["prompt_length"] * 8
    assert run_plan["prompt_span_inputs"]["total_span_input_nbytes"] == run_plan["prompt_length"] * 16
    assert run_plan["prompt_span_inputs"]["position_tensor_role"] == "prompt_row_positions"
    assert run_plan["decode_span_inputs"] == {
        "block_size": 256,
        "block_table_len": 2,
        "base_offsets": [0, 1],
        "base_offsets_dtype": "int32",
        "base_offsets_len": 2,
        "base_offsets_nbytes": 8,
        "kv_write_position": run_plan["prompt_length"],
        "kv_write_position_dtype": "int64",
        "kv_write_position_nbytes": 8,
        "attention_live_counts": [run_plan["prompt_length"]],
        "attention_live_counts_dtype": "int64",
        "attention_live_counts_len": 1,
        "attention_live_counts_nbytes": 8,
        "max_live_count": run_plan["prompt_length"],
        "total_span_input_nbytes": 16,
    }
    assert run_plan["span_input_total_nbytes"] == run_plan["prompt_length"] * 16 + 16
    upload_manifest = run_plan["span_input_upload_manifest"]
    assert upload_manifest["entry_count"] == 5
    assert upload_manifest["total_nbytes"] == run_plan["prompt_length"] * 16 + 24
    assert upload_manifest["entries"][0] == {
        "name": "prompt_base_offsets",
        "source": "prompt_span_inputs.base_offsets",
        "kernel_args": ["prompt_kv_write.base_offsets"],
        "dtype": "int32",
        "shape": [run_plan["prompt_length"], 2],
        "nbytes": run_plan["prompt_length"] * 8,
    }
    assert upload_manifest["entries"][3] == {
        "name": "decode_kv_write_position",
        "source": "decode_span_inputs.kv_write_position",
        "kernel_args": ["decode_kv_write.position"],
        "dtype": "int64",
        "shape": [],
        "nbytes": 8,
    }
    host_payloads = run_plan["span_input_host_payloads"]
    assert host_payloads["entry_count"] == 5
    assert host_payloads["total_nbytes"] == upload_manifest["total_nbytes"]
    assert host_payloads["entries"][0]["byte_order"] == "little"
    assert host_payloads["entries"][0]["value_count"] == run_plan["prompt_length"] * 2
    assert host_payloads["entries"][0]["preview_values"] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert len(host_payloads["entries"][0]["sha256"]) == 64
    assert host_payloads["entries"][3]["preview_values"] == [run_plan["prompt_length"]]
    decode_upload_plan = run_plan["decode_input_upload_plan"]
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
    assert decode_upload_plan["input_token_nbytes"] == run_plan["input_ids_nbytes"]
    assert decode_upload_plan["span_input_nbytes"] == upload_manifest["total_nbytes"]
    assert decode_upload_plan["total_nbytes"] == run_plan["input_ids_nbytes"] + upload_manifest["total_nbytes"]
    assert decode_upload_plan["entries"][0]["sha256"] == run_plan["input_ids_sha256"]
    assert decode_upload_plan["streaming_runner_ready"] is False
    assert run_plan["prompt_fits_resource_plan"] is True
    assert run_plan["context_fits_resource_plan"] is True
    assert run_plan["stop_token_ids"] == [1, 2, 128007]
    assert run_plan["kv_dispatch_keys"]["prompt_kv_write"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_kv_write",
        "quant": "gguf_step35",
        "variant": "mixed_bf16_prompt_spans",
    }
    assert run_plan["kv_decode_launch_operation_count"] == 135
    assert run_plan["kv_decode_launch_per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert run_plan["streaming_runner_ready"] is False
    assert plan["slot_paths"][:4] == [
        "root.token_embedding",
        "root.rope_freqs",
        "root.output_norm",
        "root.lm_head",
    ]
