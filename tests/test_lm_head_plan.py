from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
    plan_lm_head_build,
    register_lm_head_kernels,
)
from hipengine.kernels.registry import resolve


def test_lm_head_registers_w4_paro_variant() -> None:
    register_lm_head_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="lm_head",
            quant="w4_paro",
            variant="fp16_argmax_bf16",
        )
        is lm_head_fp16_argmax_bf16
    )


def test_lm_head_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_lm_head_build(
        cache_root=tmp_path,
        compiler_version="hipcc fake version",
        profile="decode",
    )

    assert artifact.family == "lm_head"
    assert artifact.output_path.name == "lm_head.so"
    assert any(str(path).endswith("lm_head.hip") for path in artifact.sources)
    assert "hipcc" in artifact.command[0]


def test_lm_head_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        lm_head_fp16_argmax_bf16(0, 0, 0, 0, 0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="vocab_size"):
        lm_head_fp16_argmax_bf16(0, 0, 0, 0, 0, 0, 0, 8, 0)
    with pytest.raises(ValueError, match="threads"):
        lm_head_fp16_argmax_bf16(0, 0, 0, 0, 0, 0, 0, 8, 16, threads=64)


def test_lm_head_stage1_block_count() -> None:
    assert lm_head_argmax_stage1_blocks(1, threads=256) == 1
    assert lm_head_argmax_stage1_blocks(1024, threads=256) == 1
    assert lm_head_argmax_stage1_blocks(1025, threads=256) == 2
