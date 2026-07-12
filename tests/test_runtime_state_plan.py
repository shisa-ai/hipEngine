from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    advance_decode_positions_i64,
    embedding_lookup_batch_bf16_i64,
    embedding_lookup_batch_fp16_i64,
    embedding_lookup_batch_mapped_bf16_i64,
    embedding_lookup_batch_mapped_fp16_i64,
    embedding_lookup_bf16_i64,
    embedding_lookup_fp16_i64,
    plan_runtime_state_build,
    record_f32_row_indexed,
    record_i64_scalar_indexed,
    register_runtime_state_kernels,
    set_decode_position_i64,
    set_decode_positions_i64,
    set_i64_scalar,
    set_i64_vector,
    unpack_verify_chain_dynamic_metadata_i64,
)
from hipengine.kernels.registry import resolve


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_runtime_state_registers_graph_friendly_helpers() -> None:
    register_runtime_state_kernels()

    assert (
        resolve(backend="hip_gfx1100", layer="token_embedding", quant="w4_paro", variant="bf16_i64")
        is embedding_lookup_bf16_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="token_embedding", quant="w4_paro", variant="batch_bf16_i64")
        is embedding_lookup_batch_bf16_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="token_embedding", quant="w4_paro", variant="batch_mapped_bf16_i64")
        is embedding_lookup_batch_mapped_bf16_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="token_embedding", quant="w4_paro", variant="fp16_i64")
        is embedding_lookup_fp16_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="token_embedding", quant="w4_paro", variant="batch_fp16_i64")
        is embedding_lookup_batch_fp16_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="token_embedding", quant="w4_paro", variant="batch_mapped_fp16_i64")
        is embedding_lookup_batch_mapped_fp16_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="decode_position", quant="w4_paro", variant="set_i64")
        is set_decode_position_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="decode_position", quant="w4_paro", variant="set_vector_i64")
        is set_decode_positions_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="decode_position", quant="w4_paro", variant="advance_i64")
        is advance_decode_position_i64
    )
    assert (
        resolve(backend="hip_gfx1100", layer="decode_position", quant="w4_paro", variant="advance_vector_i64")
        is advance_decode_positions_i64
    )
    assert resolve(backend="hip_gfx1100", layer="scalar_state", quant="w4_paro", variant="set_i64") is set_i64_scalar
    assert resolve(backend="hip_gfx1100", layer="scalar_state", quant="w4_paro", variant="set_vector_i64") is set_i64_vector
    assert (
        resolve(backend="hip_gfx1100", layer="scalar_state", quant="w4_paro", variant="record_i64_indexed")
        is record_i64_scalar_indexed
    )
    assert (
        resolve(backend="hip_gfx1100", layer="scalar_state", quant="w4_paro", variant="record_f32_row_indexed")
        is record_f32_row_indexed
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="verify_metadata",
            quant="w4_paro",
            variant="unpack_chain_dynamic_i64",
        )
        is unpack_verify_chain_dynamic_metadata_i64
    )


def test_runtime_state_build_plan_is_dry_run_safe(tmp_path) -> None:
    artifact = plan_runtime_state_build(
        cache_root=tmp_path,
        compiler_version="hipcc fake version",
        profile="decode",
    )

    assert artifact.family == "runtime_state"
    assert artifact.output_path.name == "runtime_state.so"
    assert any(str(path).endswith("state.hip") for path in artifact.sources)
    assert "hipcc" in artifact.command[0]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_record_f32_row_indexed_copies_row_without_advancing_index() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    values = np.asarray([1.25, -2.5, 3.75, -4.0], dtype=np.float32)
    out = np.full((3, 4), -99.0, dtype=np.float32)
    index = np.asarray([1], dtype=np.int64)
    bufs = []
    try:
        d_values = malloc(values.nbytes, runtime=runtime)
        d_out = malloc(out.nbytes, runtime=runtime)
        d_index = malloc(index.nbytes, runtime=runtime)
        bufs.extend((d_values, d_out, d_index))
        copy_host_to_device(d_values, host_array_ptr(values), runtime=runtime)
        copy_host_to_device(d_out, host_array_ptr(out), runtime=runtime)
        copy_host_to_device(d_index, host_array_ptr(index), runtime=runtime)
        record_f32_row_indexed(d_values.ptr, d_out.ptr, d_index.ptr, cols=4, capacity=3, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), d_out, runtime=runtime)
        copy_device_to_host(host_array_ptr(index), d_index, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)

    np.testing.assert_array_equal(out[1], values)
    np.testing.assert_array_equal(out[0], np.full((4,), -99.0, dtype=np.float32))
    np.testing.assert_array_equal(out[2], np.full((4,), -99.0, dtype=np.float32))
    assert int(index[0]) == 1


def test_embedding_lookup_validates_shape_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        embedding_lookup_bf16_i64(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="vocab_size"):
        embedding_lookup_bf16_i64(0, 0, 0, 8, 0)
    with pytest.raises(ValueError, match="tokens"):
        embedding_lookup_batch_bf16_i64(0, 0, 0, 0, 8, 16)
    with pytest.raises(ValueError, match="rows"):
        embedding_lookup_batch_mapped_bf16_i64(0, 0, 0, 0, 8, 16, 1)
    with pytest.raises(ValueError, match="token_slots"):
        embedding_lookup_batch_mapped_bf16_i64(0, 0, 0, 1, 8, 16, 0)
    with pytest.raises(ValueError, match="hidden_size"):
        embedding_lookup_fp16_i64(0, 0, 0, 0, 8)
    with pytest.raises(ValueError, match="tokens"):
        embedding_lookup_batch_fp16_i64(0, 0, 0, 0, 8, 16)
    with pytest.raises(ValueError, match="rows"):
        embedding_lookup_batch_mapped_fp16_i64(0, 0, 0, 0, 8, 16, 1)
    with pytest.raises(ValueError, match="rows"):
        set_i64_vector(0, 0, 0)
    with pytest.raises(ValueError, match="rows"):
        set_decode_positions_i64(0, 0, 0, 0)
    with pytest.raises(ValueError, match="rows"):
        advance_decode_positions_i64(0, 0, 0)
    with pytest.raises(ValueError, match="capacity"):
        record_i64_scalar_indexed(0, 0, 0, 0)
    with pytest.raises(ValueError, match="cols"):
        record_f32_row_indexed(0, 0, 0, 0, 1)
    with pytest.raises(ValueError, match="capacity"):
        record_f32_row_indexed(0, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="rows"):
        unpack_verify_chain_dynamic_metadata_i64(0, 0, 0, 0, 0, 0, 0)
