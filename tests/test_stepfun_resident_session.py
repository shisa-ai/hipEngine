from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kernels.cpu_reference import gguf_q3_k_gemv, gguf_q5_k_gemv, gguf_q8_0_embedding
from hipengine.loading.gguf import GGUFReader, scan_gguf_splits
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.stepfun_gguf import build_stepfun_gguf_tensor_map
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.stepfun_gguf_runner import StepFunResidentSession

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_embeds_real_q8_tokens() -> None:
    had_torch = "torch" in sys.modules
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    token_tensor = model_map.root("token_embedding")
    token_ids = np.asarray(
        [
            0,
            model_map.config.bos_token_id,
            model_map.config.eos_token_id,
            128007,
            model_map.config.vocab_size - 1,
            model_map.config.eos_token_id,
        ],
        dtype=np.int64,
    )
    raw = GGUFReader(token_tensor.source_path).tensor_data(token_tensor.name)
    assert token_tensor.ggml_type_name == "Q8_0"
    assert raw.shape[0] == model_map.config.vocab_size
    expected_bits = float_array_to_bf16_bits(gguf_q8_0_embedding(token_ids, raw))
    runtime = get_hip_runtime()
    reset_memory_stats()

    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("root.token_embedding",),
        runtime=runtime,
    )
    try:
        assert session.weights.allocated_nbytes == token_tensor.nbytes
        actual_bits = session.embed_token_ids_bf16(token_ids, runtime=runtime)
        assert actual_bits.shape == (token_ids.size, model_map.config.hidden_size)
        assert actual_bits.dtype == np.uint16
        np.testing.assert_array_equal(actual_bits, expected_bits)
        np.testing.assert_array_equal(actual_bits[2], actual_bits[5])
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == token_tensor.nbytes
        assert stats["active_allocations"] == 1
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0
    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_projects_real_q3_and_q5_layer_weights() -> None:
    had_torch = "torch" in sys.modules
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    q3_tensor = model_map.layer(0).tensor("attn_q")
    q5_tensor = model_map.layer(0).tensor("attn_output")
    assert q3_tensor.ggml_type_name == "Q3_K"
    assert q5_tensor.ggml_type_name == "Q5_K"
    q3_out_features, q3_in_features = (int(dim) for dim in q3_tensor.shape)
    q5_out_features, q5_in_features = (int(dim) for dim in q5_tensor.shape)
    runtime = get_hip_runtime()
    reset_memory_stats()
    session = StepFunResidentSession.from_gguf_paths(
        paths,
        selected_slots=("layers.0.attn_q", "layers.0.attn_output"),
        runtime=runtime,
    )
    try:
        x_q3 = (
            (np.arange(2 * q3_in_features, dtype=np.float32).reshape(2, q3_in_features) % 31)
            - 15
        ) / 128.0
        x_q3_bits = float_array_to_bf16_bits(x_q3)
        raw_q3 = GGUFReader(q3_tensor.source_path).tensor_data(q3_tensor.name)
        expected_q3 = gguf_q3_k_gemv(bf16_to_float32(x_q3_bits), raw_q3)
        actual_q3 = session.linear_slot_bf16("layers.0.attn_q", x_q3_bits, runtime=runtime)
        assert actual_q3.shape == expected_q3.shape == (2, q3_out_features)
        assert actual_q3.dtype == np.float32
        np.testing.assert_allclose(actual_q3, expected_q3, rtol=2.0e-3, atol=2.0e-3)

        x_q5 = (
            (np.arange(2 * q5_in_features, dtype=np.float32).reshape(2, q5_in_features) % 37)
            - 18
        ) / 96.0
        x_q5_bits = float_array_to_bf16_bits(x_q5)
        raw_q5 = GGUFReader(q5_tensor.source_path).tensor_data(q5_tensor.name)
        expected_q5 = gguf_q5_k_gemv(bf16_to_float32(x_q5_bits), raw_q5)
        actual_q5 = session.linear_slot_bf16("layers.0.attn_output", x_q5_bits, runtime=runtime)
        assert actual_q5.shape == expected_q5.shape == (2, q5_out_features)
        assert actual_q5.dtype == np.float32
        np.testing.assert_allclose(actual_q5, expected_q5, rtol=2.0e-3, atol=2.0e-3)

        expected_nbytes = q3_tensor.nbytes + q5_tensor.nbytes
        assert session.weights.allocated_nbytes == expected_nbytes
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == expected_nbytes
        assert stats["active_allocations"] == 2
    finally:
        session.free(runtime=runtime)

    stats = memory_stats()
    assert stats["current_allocated_bytes"] == 0
    assert stats["active_allocations"] == 0
    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_embedding_is_torch_free() -> None:
    had_torch = "torch" in sys.modules
    runtime = get_hip_runtime()
    session = StepFunResidentSession.from_gguf_paths(
        _stepfun_gguf_paths(),
        selected_slots=("root.token_embedding",),
        runtime=runtime,
    )
    try:
        session.embed_token_ids_bf16([0], runtime=runtime)
    finally:
        session.free(runtime=runtime)

    if not had_torch:
        assert "torch" not in sys.modules


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_resident_session_requires_resident_embedding_weight() -> None:
    runtime = get_hip_runtime()
    session = StepFunResidentSession.from_gguf_paths(
        _stepfun_gguf_paths(),
        selected_slots=("root.output_norm",),
        runtime=runtime,
    )
    try:
        with pytest.raises(RuntimeError, match="token_embedding"):
            session.embed_token_ids_bf16([0], runtime=runtime)
    finally:
        session.free(runtime=runtime)
