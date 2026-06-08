from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear.lm_head import (
    argmax_f32,
    batch_argmax_f32,
    build_lm_head,
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16,
    plan_lm_head_build,
    register_lm_head_kernels,
)
from hipengine.kernels.registry import resolve


def _require_hip_runtime() -> None:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP runtime unavailable: {exc}")


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
    assert resolve(backend="hip_gfx1100", layer="argmax", quant="w4_paro", variant="f32") is argmax_f32
    assert (
        resolve(backend="hip_gfx1100", layer="argmax", quant="w4_paro", variant="batch_f32")
        is batch_argmax_f32
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
    with pytest.raises(ValueError, match="vocab_size"):
        argmax_f32(0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="hidden_size"):
        batch_argmax_f32(0, 0, 0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="vocab_size"):
        batch_argmax_f32(0, 0, 0, 0, 0, 1, 0)


def test_lm_head_stage1_block_count() -> None:
    assert lm_head_argmax_stage1_blocks(1, threads=256) == 1
    assert lm_head_argmax_stage1_blocks(1024, threads=256) == 1
    assert lm_head_argmax_stage1_blocks(1025, threads=256) == 2


def test_batch_argmax_f32_matches_cpu_oracle() -> None:
    _require_hip_runtime()
    runtime = get_hip_runtime()
    library = build_lm_head(load=True)
    logits = np.asarray(
        [
            [0.0, 4.0, 3.0, 4.0, -1.0, 2.0, 0.5],
            [-2.0, -1.0, 8.0, 7.0, 8.0, 1.0, 3.0],
            [5.5, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rows, vocab_size = logits.shape
    threads = 128
    blocks = lm_head_argmax_stage1_blocks(vocab_size, threads=threads)
    block_values = np.empty((rows, blocks), dtype=np.float32)
    block_indices = np.empty((rows, blocks), dtype=np.int64)
    out_indices = np.empty((rows,), dtype=np.int64)
    out_values = np.empty((rows,), dtype=np.float32)
    buffers = [
        malloc(logits.nbytes, runtime=runtime),
        malloc(block_values.nbytes, runtime=runtime),
        malloc(block_indices.nbytes, runtime=runtime),
        malloc(out_indices.nbytes, runtime=runtime),
        malloc(out_values.nbytes, runtime=runtime),
    ]
    try:
        copy_host_to_device(buffers[0], host_array_ptr(logits), logits.nbytes, runtime=runtime)
        batch_argmax_f32(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            rows,
            vocab_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_indices), buffers[3], out_indices.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_values), buffers[4], out_values.nbytes, runtime=runtime)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)

    expected_indices = np.asarray([np.flatnonzero(row == row.max())[0] for row in logits], dtype=np.int64)
    assert out_indices.tolist() == expected_indices.tolist()
    np.testing.assert_allclose(out_values, logits[np.arange(rows), expected_indices], rtol=0, atol=0)
