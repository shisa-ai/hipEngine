from __future__ import annotations

import pytest

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
    record_i64_scalar_indexed,
    register_runtime_state_kernels,
    set_decode_position_i64,
    set_decode_positions_i64,
    set_i64_scalar,
    set_i64_vector,
    unpack_verify_chain_dynamic_metadata_i64,
)
from hipengine.kernels.registry import resolve


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
    with pytest.raises(ValueError, match="rows"):
        unpack_verify_chain_dynamic_metadata_i64(0, 0, 0, 0, 0, 0, 0)
