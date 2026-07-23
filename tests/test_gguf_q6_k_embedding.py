from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_q4_k_embedding, gguf_q6_k_embedding
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_embedding import (
    build_gguf_q6_k_embedding,
    gguf_q4_k_embedding_bf16_out,
    gguf_q5_k_embedding_bf16_out,
    gguf_q6_k_embedding_bf16_out,
    gguf_q8_0_embedding_bf16_out,
    plan_gguf_q6_k_embedding_build,
)
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import resolve
from hipengine.loading.gguf import GGUFReader
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32, dequantize_gguf_data
from tests.test_gguf_k_gemv import make_q5_k_weight, make_q6_k_weight
from tests.test_gguf_q4_k_gemv import make_q4_k_weight

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
Q8_MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q8_0.gguf")
LAGUNA_Q2_MODEL = Path("/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.parametrize(
    ("embedding", "make_weight"),
    [
        (gguf_q4_k_embedding, make_q4_k_weight),
        (gguf_q6_k_embedding, make_q6_k_weight),
    ],
)
def test_gguf_k_embedding_cpu_reference_selects_rows(embedding, make_weight) -> None:
    qweight = make_weight(5, 512)
    token_ids = np.asarray([3, 1, 3], dtype=np.int64)

    out = embedding(token_ids, qweight)

    assert out.shape == (3, 512)
    dense = embedding(np.arange(5, dtype=np.int64), qweight)
    np.testing.assert_array_equal(out[0], dense[3])
    np.testing.assert_array_equal(out[1], dense[1])
    np.testing.assert_array_equal(out[2], dense[3])
    with pytest.raises(ValueError, match="out-of-range"):
        embedding(np.asarray([5], dtype=np.int64), qweight)


def test_gguf_q6_k_embedding_registry_and_build_plan() -> None:
    assert resolve(
        backend="hip_gfx1100",
        layer="embedding",
        quant="gguf_q4_k",
        variant="lookup_bf16_out",
    ) is gguf_q4_k_embedding_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="embedding",
        quant="gguf_q5_k",
        variant="lookup_bf16_out",
    ) is gguf_q5_k_embedding_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="embedding",
        quant="gguf_q6_k",
        variant="lookup_bf16_out",
    ) is gguf_q6_k_embedding_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="embedding",
        quant="gguf_q8_0",
        variant="lookup_bf16_out",
    ) is gguf_q8_0_embedding_bf16_out
    load_backend_kernel_package("hip_gfx1151")
    assert resolve(
        backend="hip_gfx1151",
        layer="embedding",
        quant="gguf_q4_k",
        variant="lookup_bf16_out",
    ) is gguf_q4_k_embedding_bf16_out

    artifact = plan_gguf_q6_k_embedding_build(compiler_version="test-compiler")
    assert artifact.output_path.name == "gguf_q6_k_embedding.so"
    assert "gguf_q6_k_embedding" in str(artifact.output_path)
    assert any(path.name == "gguf_q6_k_embedding.hip" for path in artifact.sources)

    dry_run = build_gguf_q6_k_embedding(dry_run=True, compiler_version="test-compiler")
    assert dry_run.output_path == artifact.output_path


def test_gguf_q6_k_embedding_wrapper_validates_contract() -> None:
    with pytest.raises(ValueError, match="rows"):
        gguf_q4_k_embedding_bf16_out(1, 2, 3, rows=0, hidden_size=256, vocab_size=1)
    with pytest.raises(ValueError, match="rows"):
        gguf_q6_k_embedding_bf16_out(1, 2, 3, rows=0, hidden_size=256, vocab_size=1)
    with pytest.raises(ValueError, match="divisible"):
        gguf_q6_k_embedding_bf16_out(1, 2, 3, rows=1, hidden_size=255, vocab_size=1)
    with pytest.raises(ValueError, match="divisible"):
        gguf_q8_0_embedding_bf16_out(1, 2, 3, rows=1, hidden_size=31, vocab_size=1)
    with pytest.raises(ValueError, match="threads"):
        gguf_q6_k_embedding_bf16_out(
            1, 2, 3, rows=1, hidden_size=256, vocab_size=1, threads=96
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q4_k_embedding_hip_matches_cpu_reference_synthetic() -> None:
    qweight = make_q4_k_weight(7, 512)
    token_ids = np.asarray([0, 3, 6, 3], dtype=np.int64)
    _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=512,
        vocab_size=7,
        expected=gguf_q4_k_embedding(token_ids, qweight),
        kernel=gguf_q4_k_embedding_bf16_out,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q4_k_embedding_invalid_ids_preserve_output_rows() -> None:
    qweight = make_q4_k_weight(7, 256)
    token_ids = np.asarray([-1, 3, 7], dtype=np.int64)
    valid = gguf_q4_k_embedding(np.asarray([3], dtype=np.int64), qweight)
    initial = np.full((3, 256), 0x3F80, dtype=np.uint16)
    actual = _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=256,
        vocab_size=7,
        expected=None,
        kernel=gguf_q4_k_embedding_bf16_out,
        initial=initial,
    )
    np.testing.assert_array_equal(actual[0], initial[0])
    np.testing.assert_array_equal(actual[2], initial[2])
    np.testing.assert_array_equal(actual[1], float_array_to_bf16_bits(valid)[0])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q5_k_embedding_hip_matches_cpu_reference_synthetic() -> None:
    qweight = make_q5_k_weight(7, 512)
    token_ids = np.asarray([0, 3, 6, 3], dtype=np.int64)
    _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=512,
        vocab_size=7,
        expected=dequantize_gguf_data(qweight[token_ids], GGMLQuantizationType.Q5_K),
        kernel=gguf_q5_k_embedding_bf16_out,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q6_k_embedding_hip_matches_cpu_reference_synthetic() -> None:
    qweight = make_q6_k_weight(7, 512)
    token_ids = np.asarray([0, 3, 6, 3], dtype=np.int64)
    _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=512,
        vocab_size=7,
        expected=gguf_q6_k_embedding(token_ids, qweight),
        kernel=gguf_q6_k_embedding_bf16_out,
    )


@pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q6_k_embedding_hip_matches_real_token_embedding() -> None:
    reader = GGUFReader(MODEL)
    tensor = reader.tensor_info("token_embd.weight")
    qweight = np.asarray(reader.tensor_data("token_embd.weight"))
    token_ids = np.asarray([760, 4087, 369, 760], dtype=np.int64)
    _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=tensor.shape[1],
        vocab_size=tensor.shape[0],
        expected=gguf_q6_k_embedding(token_ids, qweight),
        kernel=gguf_q6_k_embedding_bf16_out,
    )


@pytest.mark.skipif(
    not LAGUNA_Q2_MODEL.exists(),
    reason=f"local GGUF fixture not found: {LAGUNA_Q2_MODEL}",
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q5_k_embedding_hip_matches_real_laguna_token_embedding() -> None:
    reader = GGUFReader(LAGUNA_Q2_MODEL)
    tensor = reader.tensor_info("token_embd.weight")
    qweight = np.asarray(reader.tensor_data("token_embd.weight"))
    token_ids = np.asarray([1, 2, 3, 1], dtype=np.int64)
    _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=tensor.shape[1],
        vocab_size=tensor.shape[0],
        expected=dequantize_gguf_data(qweight[token_ids], tensor.ggml_type),
        kernel=gguf_q5_k_embedding_bf16_out,
    )


@pytest.mark.skipif(not Q8_MODEL.exists(), reason=f"local GGUF fixture not found: {Q8_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gguf_q8_0_embedding_hip_matches_real_token_embedding() -> None:
    reader = GGUFReader(Q8_MODEL)
    tensor = reader.tensor_info("token_embd.weight")
    qweight = np.asarray(reader.tensor_data("token_embd.weight"))
    token_ids = np.asarray([760, 4087, 369, 760], dtype=np.int64)
    expected = dequantize_gguf_data(qweight[token_ids], tensor.ggml_type)
    _run_embedding_case(
        qweight,
        token_ids,
        hidden_size=tensor.shape[1],
        vocab_size=tensor.shape[0],
        expected=expected,
        kernel=gguf_q8_0_embedding_bf16_out,
    )


def _run_embedding_case(
    qweight: np.ndarray,
    token_ids: np.ndarray,
    *,
    hidden_size: int,
    vocab_size: int,
    expected: np.ndarray | None,
    kernel,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = build_gguf_q6_k_embedding(load=True)
    expected_bits = None if expected is None else float_array_to_bf16_bits(expected)
    out = (
        np.empty((token_ids.shape[0], hidden_size), dtype=np.uint16)
        if initial is None
        else np.ascontiguousarray(initial).copy()
    )
    bufs = []
    try:
        token_dev = malloc(token_ids.nbytes, runtime=runtime)
        qweight_dev = malloc(qweight.nbytes, runtime=runtime)
        out_dev = malloc(out.nbytes, runtime=runtime)
        bufs.extend((token_dev, qweight_dev, out_dev))
        copy_host_to_device(token_dev, host_array_ptr(np.ascontiguousarray(token_ids)), runtime=runtime)
        copy_host_to_device(qweight_dev, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime)
        if initial is not None:
            copy_host_to_device(out_dev, host_array_ptr(out), runtime=runtime)
        kernel(
            token_dev.ptr,
            qweight_dev.ptr,
            out_dev.ptr,
            token_ids.shape[0],
            hidden_size,
            vocab_size,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)
    if expected_bits is not None:
        # CPU and GPU can disagree on the sign bit of exact zero; compare BF16
        # numeric values rather than raw bits for this no-accumulation lookup.
        max_abs = float(np.max(np.abs(bf16_to_float32(out) - bf16_to_float32(expected_bits))))
        assert max_abs == 0.0
    return out
