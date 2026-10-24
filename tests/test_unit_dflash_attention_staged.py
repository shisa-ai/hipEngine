"""Deterministic (no-HIP) tests for the staged DFlash GQA attention wrapper.

The staged route is three kernels -- score, softmax, value -- that reproduce
``dflash_gqa_attention_f32_bf16`` arithmetic exactly while exposing
``batch * query_len * num_q_heads * kv_len`` blocks for the score pass.  Byte
equality against the parent needs a GPU and lives in
``tests/test_gpu_dflash_attention_exact.py``; this file covers registration and
the argument/workspace validation that must happen before any HIP build.
"""

from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
    dflash_gqa_attention_f32_bf16,
    dflash_gqa_attention_f32_bf16_staged,
    dflash_gqa_attention_scores_f32,
    dflash_gqa_attention_scores_workspace_elements,
    dflash_gqa_attention_softmax_f32,
    dflash_gqa_attention_values_f32_bf16,
)
from hipengine.kernels.registry import resolve

_SHAPE = dict(
    batch_size=1,
    query_len=1,
    kv_len=8,
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=128,
)


def test_staged_attention_is_registered_as_an_exact_variant() -> None:
    staged = resolve(
        backend="hip_gfx1100",
        layer="dflash_gqa_attention",
        quant="w4_paro",
        variant="f32_bf16_staged",
    )
    parent = resolve(
        backend="hip_gfx1100",
        layer="dflash_gqa_attention",
        quant="w4_paro",
        variant="f32_bf16",
    )
    assert staged is dflash_gqa_attention_f32_bf16_staged
    # The parent stays registered as the strict reference/fallback.
    assert parent is dflash_gqa_attention_f32_bf16


def test_scores_workspace_elements_is_batch_query_head_kv() -> None:
    assert dflash_gqa_attention_scores_workspace_elements(1, 1, 4096, 32) == 131072
    assert dflash_gqa_attention_scores_workspace_elements(2, 3, 17, 6) == 612
    assert dflash_gqa_attention_scores_workspace_elements(1, 1, 1, 1) == 1


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"num_q_heads": 3}, "num_q_heads"),
        ({"kv_len": 0}, "kv_len"),
        ({"head_dim": 0}, "head_dim"),
        ({"query_len": -1}, "query_len"),
        ({"threads": 96}, "threads"),
    ],
)
def test_staged_aggregate_validates_shape_before_loading_hip(overrides, match) -> None:
    kwargs = dict(_SHAPE)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        dflash_gqa_attention_f32_bf16_staged(
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            0x5000,
            scores_workspace_elements=0x10000,
            **kwargs,
        )


def test_staged_aggregate_requires_a_score_workspace_pointer() -> None:
    with pytest.raises(ValueError, match="scores_f32_ptr"):
        dflash_gqa_attention_f32_bf16_staged(
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            0,
            scores_workspace_elements=1024,
            **_SHAPE,
        )


def test_staged_aggregate_rejects_an_undersized_score_workspace() -> None:
    required = dflash_gqa_attention_scores_workspace_elements(
        _SHAPE["batch_size"], _SHAPE["query_len"], _SHAPE["kv_len"], _SHAPE["num_q_heads"]
    )
    with pytest.raises(ValueError, match=f"at least .*{required}"):
        dflash_gqa_attention_f32_bf16_staged(
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            0x5000,
            scores_workspace_elements=required - 1,
            **_SHAPE,
        )


def test_each_stage_validates_shape_and_workspace_before_loading_hip() -> None:
    with pytest.raises(ValueError, match="num_q_heads"):
        dflash_gqa_attention_scores_f32(
            0x1000,
            0x2000,
            0x3000,
            scores_workspace_elements=64,
            **{**_SHAPE, "num_q_heads": 3},
        )
    with pytest.raises(ValueError, match="scores_f32_ptr"):
        dflash_gqa_attention_softmax_f32(0, scores_workspace_elements=64, **_SHAPE)
    with pytest.raises(ValueError, match="scores_workspace_elements"):
        dflash_gqa_attention_values_f32_bf16(
            0x3000,
            0x4000,
            0x5000,
            scores_workspace_elements=0,
            **_SHAPE,
        )


@pytest.mark.parametrize("name", ["batch_size", "query_len", "kv_len", "num_q_heads", "num_kv_heads", "head_dim"])
def test_staged_rejects_wrapping_shape_integers(name):
    with pytest.raises(ValueError, match="64-bit|grid|extent"):
        dflash_gqa_attention_f32_bf16_staged(
            0x100000, 0x200000, 0x300000, 0x400000, 0x500000,
            scores_workspace_elements=128, **{**_SHAPE, name: (1 << 64) + 8},
        )


def test_staged_rejects_tensor_byte_extent_overflow():
    with pytest.raises(ValueError, match="extent"):
        dflash_gqa_attention_f32_bf16_staged(
            0x100000, 0x200000, 0x300000, 0x400000, 0x500000,
            scores_workspace_elements=128, **{**_SHAPE, "head_dim": (1 << 62)},
        )


@pytest.mark.parametrize("pointer_index", [0, 1, 2, 3])
def test_staged_rejects_workspace_overlap_before_any_submission(pointer_index):
    pointers = [0x100000, 0x200000, 0x300000, 0x400000, 0x500000]
    pointers[pointer_index] = pointers[4] + 4
    with pytest.raises(ValueError, match="overlap"):
        dflash_gqa_attention_f32_bf16_staged(
            *pointers, scores_workspace_elements=128, **_SHAPE,
        )
