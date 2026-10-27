"""The Q5_1 wave64 decode reduction is a registered variant, not a selector.

A Q5_1 selected expert projection uses the logical-256 t64 owner. The wave64
kernel, registrations and direct diagnostic consumers remain available.
"""
import inspect
from pathlib import Path
import re

import pytest

from hipengine.runtime import qwen4_exp_runner as runner

ROOT = Path(__file__).resolve().parents[1]

#: Direct consumers of the reduction kernel. They call it by name rather than
#: through a dispatch, so the retirement leaves them untouched.
DIRECT_CONSUMERS = (
    ROOT / "tests" / "test_gpu_qwen4_exp_q5_1_selected.py",
    ROOT / "scripts" / "gguf_q5_1_gemma_down_decode_screen.py",
)

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_Q5_1_WAVE64"
WAVE64_VARIANT = "selected_gemv_wave64_bf16_bf16_out"
DEFAULT_VARIANT = "selected_gemv_logical256_t64_bf16_bf16_out"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.delenv(RETIRED_ENV, raising=False)


def test_q5_1_selected_gemv_runs_the_logical256_t64_owner():
    assert runner._qwen4_exp_selected_gemv_variant("gguf_q5_1") == DEFAULT_VARIANT


@pytest.mark.parametrize("value", ["1", "true", "wave64", "all"])
def test_the_retired_selector_cannot_reach_the_wave64_reduction(monkeypatch, value):
    monkeypatch.setenv(RETIRED_ENV, value)
    assert runner._qwen4_exp_selected_gemv_variant("gguf_q5_1") == DEFAULT_VARIANT


def test_no_runner_dispatch_reads_the_retired_switch():
    """Retirement is a reader removal, not a default flip."""

    source = inspect.getsource(runner)
    assert not re.search(r"environ[^\n]*Q5_1_WAVE64", source)


@pytest.mark.parametrize("quant", ["gguf_q4_k", "gguf_q6_k", "gguf_q8_0"])
def test_other_quants_keep_the_plain_selected_gemv(quant):
    assert runner._qwen4_exp_selected_gemv_variant(quant) == (
        "selected_gemv_bf16_bf16_out"
    )


def test_the_wave64_reduction_stays_registered_on_both_linear_layers():
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out,
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_qwen4_exp_q5_1_kernels()
    for layer in ("linear", "moe_linear"):
        assert resolve(
            backend="hip_gfx1100", layer=layer, quant="gguf_q5_1",
            variant=WAVE64_VARIANT,
        ) is qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out


def test_the_default_owner_resolves_to_the_t64_leaf():
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out,
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_qwen4_exp_q5_1_kernels()
    for layer in ("linear", "moe_linear"):
        assert resolve(
            backend="hip_gfx1100", layer=layer, quant="gguf_q5_1",
            variant=runner._qwen4_exp_selected_gemv_variant("gguf_q5_1"),
        ) is qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out


@pytest.mark.parametrize("path", DIRECT_CONSUMERS, ids=lambda path: path.name)
def test_direct_consumers_still_call_the_reduction(path):
    source = path.read_text()
    assert "qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out" in source
    # And no harness reintroduces the retired selector to reach it.
    assert RETIRED_ENV not in source


def test_the_selected_projection_keeps_one_variant_decision():
    source = inspect.getsource(runner.run_qwen4_exp_moe)
    assert "_qwen4_exp_selected_gemv_variant(" in source
    # The inline per-quant default the opt-in used to override is gone, so the
    # strict selected gemv cannot come back as a second copy of the ladder.
    assert 'variant = "selected_gemv_bf16_bf16_out"' not in source


def test_q5_1_owner_is_distinct_from_the_generic_selected_leaf():
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_qwen4_exp_q5_1_kernels()
    assert resolve(
        backend="hip_gfx1100", layer="moe_linear", quant="gguf_q5_1",
        variant=runner._qwen4_exp_selected_gemv_variant("gguf_q5_1"),
    ) is not qwen4_exp_q5_1_selected_gemv_bf16_bf16_out
