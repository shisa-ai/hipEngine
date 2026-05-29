from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime.stepfun_gguf_runner import StepFunShortContextDecodePlanner

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


def test_stepfun_decode_planner_does_not_import_torch() -> None:
    had_torch = "torch" in sys.modules

    planner = StepFunShortContextDecodePlanner.from_gguf_paths(_stepfun_gguf_paths())
    planner.plan_chat([{"role": "user", "content": "hello"}], reasoning_effort="low")

    if not had_torch:
        assert "torch" not in sys.modules
