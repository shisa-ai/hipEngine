from __future__ import annotations

import pytest

from hipengine.dispatch.fusion import FusionPlanner, resolve_plan
from hipengine.kernels.registry import (
    KernelKey,
    MissingKernelError,
    clear_registry_for_tests,
    register,
)
from hipengine.models import resolve_model


def dummy_kernel(*args, **kwargs):
    return args, kwargs


def setup_function() -> None:
    clear_registry_for_tests()


def test_toy_model_plans_primitives_and_missing_kernel_is_clean() -> None:
    model = resolve_model("HipEngineToyForCausalLM")
    planner = FusionPlanner(backend="hip_gfx1100", quant="fp16")

    plan = planner.plan(model.layer_sequence())

    assert tuple(step.layer for step in plan) == model.layer_sequence()
    with pytest.raises(MissingKernelError) as exc_info:
        resolve_plan(plan)

    message = str(exc_info.value)
    assert "no kernel implementation" in message
    assert "embed" in message
    assert "hip_gfx1100" in message
    assert "fp16" in message


def test_longest_match_fusion_uses_registered_composite() -> None:
    model = resolve_model("HipEngineToyForCausalLM")
    composite = "rmsnorm+rotate+qkv_proj"

    for layer in ("embed", composite, "attention_decode", "o_proj", "lm_head"):
        register(KernelKey("hip_gfx1100", layer, "fp16"), dummy_kernel)

    planner = FusionPlanner(backend="hip_gfx1100", quant="fp16", max_fusion_width=4)
    plan = planner.plan(model.layer_sequence())

    assert tuple(step.layer for step in plan) == (
        "embed",
        composite,
        "attention_decode",
        "o_proj",
        "lm_head",
    )
    assert plan[1].source_layers == ("rmsnorm", "rotate", "qkv_proj")

    bound = resolve_plan(plan)
    assert len(bound) == 5
    assert all(item.kernel is dummy_kernel for item in bound)
