from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats, reset_memory_stats
from hipengine.kernels.cpu_reference import gguf_q8_0_embedding
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
    paths = _stepfun_gguf_paths()
    info = scan_gguf_splits(paths)
    model_map = build_stepfun_gguf_tensor_map(info)
    token_tensor = model_map.root("token_embedding")
    token_ids = np.asarray([0, 1, 128007], dtype=np.int64)
    raw = GGUFReader(token_tensor.source_path).tensor_data(token_tensor.name)
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
        assert actual_bits.shape == (3, model_map.config.hidden_size)
        assert actual_bits.dtype == np.uint16
        np.testing.assert_array_equal(actual_bits, expected_bits)
        stats = memory_stats()
        assert stats["current_allocated_bytes"] == token_tensor.nbytes
        assert stats["active_allocations"] == 1
    finally:
        session.free(runtime=runtime)

    assert memory_stats()["current_allocated_bytes"] == 0


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
