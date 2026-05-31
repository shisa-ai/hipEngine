from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime.stepfun_gguf_runner import (
    STEPFUN_GGUF_KERNEL_QUANT,
    STEPFUN_KV_ATTENTION_BLOCK_SIZE,
    StepFunShortContextDecodePlanner,
    stepfun_kv_cache_nbytes,
    stepfun_kv_decode_kernel_plan,
    stepfun_text_decode_slot_paths,
)

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def test_stepfun_short_context_decode_plan_preserves_chat_prefix_and_multi_eos() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )

    plan = planner.plan_chat([{"role": "user", "content": "hello"}], reasoning_effort="low")

    assert plan.prompt_length + plan.max_new_tokens <= 512
    assert plan.max_new_tokens == 1
    assert plan.rendered_prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert plan.stop_token_ids == (1, 2, 128007)
    assert plan.should_stop(1)
    assert plan.should_stop(2)
    assert plan.should_stop(128007)
    assert not plan.should_stop(128006)
    assert plan.quant_dispatch_keys["gguf_q3_k"] == KernelKey(
        "hip_gfx1151", "linear", "gguf_q3_k", "gemv_bf16_bf16_out"
    )
    assert plan.quant_dispatch_keys["gguf_q5_k"] == KernelKey(
        "hip_gfx1151", "linear", "gguf_q5_k", "gemv_bf16_bf16_out"
    )
    assert plan.quant_dispatch_keys["gguf_q8_0"] == KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0", "gemv_bf16_bf16_out"
    )
    assert plan.kv_dispatch_keys["prompt_kv_write"] == KernelKey(
        "hip_gfx1151", "paged_kv_write", STEPFUN_GGUF_KERNEL_QUANT, "mixed_bf16_prompt_spans"
    )
    assert plan.kv_dispatch_keys["decode_kv_write"] == KernelKey(
        "hip_gfx1151", "paged_kv_write", STEPFUN_GGUF_KERNEL_QUANT, "mixed_bf16_spans"
    )
    assert plan.kv_dispatch_keys["decode_attention"] == KernelKey(
        "hip_gfx1151", "paged_attn_decode", STEPFUN_GGUF_KERNEL_QUANT, "bf16_split_k_gate_f32_spans"
    )


def test_stepfun_kv_decode_run_plan_binds_prompt_to_resource_spans() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )

    run_plan = planner.plan_kv_decode_chat(
        [{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        context_pages=1,
        page_size=512,
    )

    assert run_plan.prompt_length == run_plan.decode_plan.prompt_length > 0
    assert run_plan.input_ids == run_plan.decode_plan.input_ids
    assert run_plan.rendered_prompt_nchars == len(run_plan.decode_plan.rendered_prompt)
    assert run_plan.rendered_prompt_sha256 == hashlib.sha256(
        run_plan.decode_plan.rendered_prompt.encode("utf-8")
    ).hexdigest()
    assert run_plan.prompt_positions == tuple(range(run_plan.prompt_length))
    assert run_plan.decode_position == run_plan.prompt_length
    assert run_plan.decode_live_count == run_plan.prompt_length
    assert run_plan.required_context_tokens == run_plan.prompt_length + 1
    assert run_plan.max_prompt_rows == 511
    assert run_plan.attention_block_size == 256
    assert run_plan.attention_block_table_len == 2
    assert run_plan.prompt_span_base_offsets == tuple(
        value for _ in range(run_plan.prompt_length) for value in (0, 1)
    )
    assert run_plan.decode_span_base_offsets == (0, 1)
    assert run_plan.prompt_fits_resource_plan is True
    assert run_plan.context_fits_resource_plan is True
    assert run_plan.streaming_runner_ready is False
    payload = run_plan.to_dict()
    assert payload["prompt_length"] == run_plan.prompt_length
    assert payload["input_ids"] == list(run_plan.decode_plan.input_ids)
    assert payload["input_id_count"] == run_plan.prompt_length
    assert payload["rendered_prompt_nchars"] == len(run_plan.decode_plan.rendered_prompt)
    assert payload["rendered_prompt_sha256"] == hashlib.sha256(
        run_plan.decode_plan.rendered_prompt.encode("utf-8")
    ).hexdigest()
    assert payload["prompt_positions"] == list(range(run_plan.prompt_length))
    assert payload["decode_position"] == run_plan.prompt_length
    assert payload["decode_live_count"] == run_plan.prompt_length
    assert payload["required_context_tokens"] == run_plan.required_context_tokens
    assert payload["max_context"] == 512
    assert payload["max_prompt_rows"] == 511
    assert payload["attention_block_size"] == 256
    assert payload["attention_block_table_len"] == 2
    assert payload["prompt_span_inputs"] == {
        "rows": run_plan.prompt_length,
        "block_size": 256,
        "block_table_len_per_row": 2,
        "base_offsets": [value for _ in range(run_plan.prompt_length) for value in (0, 1)],
        "base_offsets_len": run_plan.prompt_length * 2,
        "live_counts": list(range(run_plan.prompt_length)),
        "live_counts_len": run_plan.prompt_length,
        "position_tensor_role": "prompt_row_positions",
        "max_live_count": run_plan.prompt_length - 1,
    }
    assert payload["decode_span_inputs"] == {
        "block_size": 256,
        "block_table_len": 2,
        "base_offsets": [0, 1],
        "base_offsets_len": 2,
        "kv_write_position": run_plan.prompt_length,
        "attention_live_counts": [run_plan.prompt_length],
        "attention_live_counts_len": 1,
        "max_live_count": run_plan.prompt_length,
    }
    assert payload["stop_token_ids"] == [1, 2, 128007]
    assert payload["kv_dispatch_keys"]["decode_attention"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_attn_decode",
        "quant": STEPFUN_GGUF_KERNEL_QUANT,
        "variant": "bf16_split_k_gate_f32_spans",
    }
    assert payload["kv_decode_launch_operation_count"] == 135
    assert payload["kv_decode_launch_per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert payload["streaming_runner_ready"] is False


def test_stepfun_kv_decode_run_plan_rejects_resource_span_too_small() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=512,
        max_new_tokens=1,
    )

    with pytest.raises(ValueError, match="KV prompt span"):
        planner.plan_kv_decode_chat(
            [{"role": "user", "content": "hello"}],
            reasoning_effort="low",
            context_pages=1,
            page_size=16,
        )


def test_stepfun_short_context_decode_plan_rejects_long_prompts() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(
        _stepfun_gguf_paths(),
        max_context=32,
        max_new_tokens=1,
    )

    with pytest.raises(ValueError, match="max_context"):
        planner.plan_chat([{"role": "user", "content": "hello " * 128}], reasoning_effort="low")


def test_stepfun_text_decode_slot_paths_cover_validated_text_model_without_extra_modal_slots() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())

    slots = stepfun_text_decode_slot_paths(planner.model_map)

    assert slots[:4] == (
        "root.token_embedding",
        "root.rope_freqs",
        "root.output_norm",
        "root.lm_head",
    )
    assert len(slots) == len(set(slots)) == planner.info.tensor_count
    assert "layers.0.ffn_down" in slots
    assert "layers.3.ffn_gate_inp" in slots
    assert "layers.44.ffn_down_shexp" in slots
    forbidden_fragments = ("vision", "projector", "mmproj", "mtp", "nextn")
    assert not any(fragment in slot for slot in slots for fragment in forbidden_fragments)

    tensor_names: list[str] = []
    for slot in slots:
        parts = slot.split(".")
        if parts[0] == "root":
            tensor_names.append(planner.model_map.root(parts[1]).name)
        else:
            assert parts[0] == "layers"
            tensor_names.append(planner.model_map.layer(int(parts[1])).tensor(parts[2]).name)
    assert set(tensor_names) == set(planner.model_map.tensor_names)


def test_stepfun_text_decode_resource_plan_estimates_weight_and_kv_bytes() -> None:
    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())

    plan = planner.text_decode_resource_plan(context_pages=1, page_size=512)

    assert plan.backend == "hip_gfx1151"
    assert plan.context_pages == 1
    assert plan.page_size == 512
    assert plan.slot_paths == stepfun_text_decode_slot_paths(planner.model_map)
    assert plan.slot_count == planner.info.tensor_count == 754
    assert plan.resident_weight_nbytes == planner.info.total_tensor_nbytes
    assert plan.resident_weight_gib == pytest.approx(planner.info.total_tensor_nbytes / 2**30)
    assert len(plan.kv_layer_nbytes) == planner.model_map.config.block_count == 45
    assert plan.kv_layer_nbytes[0] == (1_048_576, 1_048_576)
    assert plan.kv_nbytes == 94_371_840
    assert plan.kv_gib == pytest.approx(94_371_840 / 2**30)
    assert plan.kv_nbytes == stepfun_kv_cache_nbytes(
        planner.model_map.config,
        context_pages=1,
        page_size=512,
    )
    assert plan.total_nbytes == plan.resident_weight_nbytes + plan.kv_nbytes
    payload = plan.to_dict()
    assert payload["backend"] == "hip_gfx1151"
    assert payload["slot_count"] == 754
    assert payload["slot_paths"][:4] == [
        "root.token_embedding",
        "root.rope_freqs",
        "root.output_norm",
        "root.lm_head",
    ]
    assert payload["resident_weight_nbytes"] == 102_499_149_312
    assert payload["context_pages"] == 1
    assert payload["page_size"] == 512
    assert payload["max_new_tokens"] == 1
    assert payload["kv_buffer_count"] == 90
    assert payload["kv_layer_nbytes"][0] == {
        "layer": 0,
        "key_nbytes": 1_048_576,
        "value_nbytes": 1_048_576,
    }
    assert payload["kv_nbytes"] == 94_371_840
    assert payload["total_nbytes"] == plan.total_nbytes
    kv_kernel_plan = payload["kv_decode_kernel_plan"]
    assert kv_kernel_plan["model_quant"] == STEPFUN_GGUF_KERNEL_QUANT
    assert kv_kernel_plan["kv_storage_dtype"] == "bf16"
    assert kv_kernel_plan["decode_attention_kind"] == "splitk_gate_f32"
    assert kv_kernel_plan["max_context"] == 512
    assert kv_kernel_plan["max_new_tokens"] == 1
    assert kv_kernel_plan["max_prompt_rows"] == 511
    assert kv_kernel_plan["attention_block_size"] == STEPFUN_KV_ATTENTION_BLOCK_SIZE == 256
    assert kv_kernel_plan["attention_block_table_len"] == 2
    assert kv_kernel_plan["attention_capacity_tokens"] == 512
    assert kv_kernel_plan["decode_span"] == {
        "block_size": 256,
        "block_table_len": 2,
        "live_counts_len": 1,
        "max_live_count": 511,
        "capacity_tokens": 512,
        "shape_compatible": True,
    }
    assert kv_kernel_plan["prompt_span"] == {
        "block_size": 256,
        "max_prompt_rows": 511,
        "block_table_len_per_row": 2,
        "base_offsets_len_formula": "rows * 2",
        "live_counts_len_formula": "rows",
        "row_positions_required": True,
        "shape_compatible": True,
    }
    assert kv_kernel_plan["decode_span_shape_compatible"] is True
    assert kv_kernel_plan["prompt_span_shape_compatible"] is True
    assert kv_kernel_plan["span_shape_compatible"] is True
    assert kv_kernel_plan["all_registered"] is True
    assert kv_kernel_plan["dispatch_keys"]["prompt_kv_write"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_kv_write",
        "quant": STEPFUN_GGUF_KERNEL_QUANT,
        "variant": "mixed_bf16_prompt_spans",
    }
    assert kv_kernel_plan["dispatch_keys"]["decode_attention"] == {
        "backend": "hip_gfx1151",
        "layer": "paged_attn_decode",
        "quant": STEPFUN_GGUF_KERNEL_QUANT,
        "variant": "bf16_split_k_gate_f32_spans",
    }
    launch_schedule = payload["kv_decode_launch_schedule"]
    assert launch_schedule["source"] == "text_decode_resource_plan"
    assert launch_schedule["layer_count"] == 45
    assert launch_schedule["operation_count"] == 135
    assert launch_schedule["per_layer_order"] == [
        "prompt_kv_write",
        "decode_kv_write",
        "decode_attention",
    ]
    assert launch_schedule["first_layer_ops"] == [
        "layers.0.prompt_kv_write",
        "layers.0.decode_kv_write",
        "layers.0.decode_attention",
    ]
    assert launch_schedule["last_layer_ops"] == [
        "layers.44.prompt_kv_write",
        "layers.44.decode_kv_write",
        "layers.44.decode_attention",
    ]
    assert launch_schedule["stages"] == [
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
    ]
    assert launch_schedule["all_stage_dispatch_ready"] is True
    assert launch_schedule["streaming_runner_ready"] is False

    with pytest.raises(ValueError, match="context_pages"):
        planner.text_decode_resource_plan(context_pages=0, page_size=512)


def test_stepfun_kv_decode_kernel_plan_resolves_step35_registry_keys() -> None:
    plan = stepfun_kv_decode_kernel_plan(backend="hip_gfx1151")

    assert plan.model_quant == STEPFUN_GGUF_KERNEL_QUANT
    assert plan.kv_storage_dtype == "bf16"
    assert plan.decode_attention_kind == "splitk_gate_f32"
    assert plan.max_context == 512
    assert plan.max_new_tokens == 1
    assert plan.max_prompt_rows == 511
    assert plan.decode_max_live_count == 511
    assert plan.attention_block_size == STEPFUN_KV_ATTENTION_BLOCK_SIZE == 256
    assert plan.attention_block_table_len == 2
    assert plan.attention_capacity_tokens == 512
    assert plan.decode_span_contract == {
        "block_size": 256,
        "block_table_len": 2,
        "live_counts_len": 1,
        "max_live_count": 511,
        "capacity_tokens": 512,
        "shape_compatible": True,
    }
    assert plan.prompt_span_contract == {
        "block_size": 256,
        "max_prompt_rows": 511,
        "block_table_len_per_row": 2,
        "base_offsets_len_formula": "rows * 2",
        "live_counts_len_formula": "rows",
        "row_positions_required": True,
        "shape_compatible": True,
    }
    assert plan.decode_span_shape_compatible is True
    assert plan.prompt_span_shape_compatible is True
    assert plan.span_shape_compatible is True
    assert plan.all_registered is True
    assert plan.registered == {
        "prompt_kv_write": True,
        "decode_kv_write": True,
        "decode_attention": True,
    }
    assert plan.dispatch_keys["decode_attention"] == KernelKey(
        "hip_gfx1151", "paged_attn_decode", STEPFUN_GGUF_KERNEL_QUANT, "bf16_split_k_gate_f32_spans"
    )


def test_stepfun_kv_decode_kernel_plan_rounds_block_table_to_attention_block_size() -> None:
    plan = stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=513)

    assert plan.attention_block_size == STEPFUN_KV_ATTENTION_BLOCK_SIZE
    assert plan.attention_block_table_len == 3
    assert plan.max_prompt_rows == 512
    assert plan.decode_max_live_count == 512
    assert plan.attention_capacity_tokens == 768
    assert plan.prompt_span_contract["base_offsets_len_formula"] == "rows * 3"
    assert plan.span_shape_compatible is True

    with pytest.raises(ValueError, match="max_context"):
        stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=512, max_new_tokens=0)
    with pytest.raises(ValueError, match="at least one prompt token"):
        stepfun_kv_decode_kernel_plan(backend="hip_gfx1151", max_context=1, max_new_tokens=1)


def test_stepfun_decode_planner_does_not_import_torch() -> None:
    had_torch = "torch" in sys.modules

    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())
    planner.plan_chat([{"role": "user", "content": "hello"}], reasoning_effort="low")

    if not had_torch:
        assert "torch" not in sys.modules
