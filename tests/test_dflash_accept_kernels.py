from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.linear import (
    argmax_f32_rows_i32,
    lm_head_fp16_argmax_bf16_rows_i32,
    plan_lm_head_build,
    register_lm_head_kernels,
)
from hipengine.kernels.hip_gfx1100.speculative import (
    dflash_accept_chain_i32,
    plan_dflash_accept_build,
    register_dflash_accept_kernels,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import resolve


def test_dflash_accept_and_row_argmax_build_plans_include_native_arch(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")

    lm_head = plan_lm_head_build(compiler_version="hipcc:test")
    accept = plan_dflash_accept_build(compiler_version="hipcc:test")

    assert "--offload-arch=gfx1151" in lm_head.command
    assert "--offload-arch=gfx1151" in accept.command
    assert lm_head.target_arch == "gfx1151"
    assert accept.target_arch == "gfx1151"


def test_dflash_accept_and_row_argmax_register_for_gfx1151_aliases() -> None:
    register_lm_head_kernels(replace=True)
    register_dflash_accept_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    assert (
        resolve(backend="hip_gfx1151", layer="argmax", quant="w4_paro", variant="f32_rows_i32")
        is argmax_f32_rows_i32
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="lm_head",
            quant="w4_paro",
            variant="fp16_argmax_bf16_rows_i32",
        )
        is lm_head_fp16_argmax_bf16_rows_i32
    )
    assert (
        resolve(backend="hip_gfx1151", layer="dflash_accept_chain", quant="w4_paro", variant="i32")
        is dflash_accept_chain_i32
    )


def test_row_argmax_and_dflash_accept_wrappers_validate_shapes_before_loading_hip() -> None:
    with pytest.raises(ValueError, match="rows"):
        argmax_f32_rows_i32(0, 0, 0, 0, None, rows=0, vocab_size=16)
    with pytest.raises(ValueError, match="vocab_size"):
        lm_head_fp16_argmax_bf16_rows_i32(0, 0, 0, 0, 0, 0, None, rows=1, hidden_size=8, vocab_size=0)
    with pytest.raises(ValueError, match="request_count"):
        dflash_accept_chain_i32(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rows=2,
            request_count=3,
            output_stride=2,
        )
    with pytest.raises(ValueError, match="output_stride"):
        dflash_accept_chain_i32(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            rows=2,
            request_count=1,
            output_stride=0,
        )
