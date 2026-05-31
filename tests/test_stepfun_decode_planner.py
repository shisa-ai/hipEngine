from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime.stepfun_gguf_runner import (
    StepFunShortContextDecodePlanner,
    stepfun_kv_cache_nbytes,
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

    with pytest.raises(ValueError, match="context_pages"):
        planner.text_decode_resource_plan(context_pages=0, page_size=512)


def test_stepfun_decode_planner_does_not_import_torch() -> None:
    had_torch = "torch" in sys.modules

    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())
    planner.plan_chat([{"role": "user", "content": "hello"}], reasoning_effort="low")

    if not had_torch:
        assert "torch" not in sys.modules
