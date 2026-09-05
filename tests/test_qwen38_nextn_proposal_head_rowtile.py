"""Actual-weight RED for Qwen3.8 physical NextN proposal-head row reuse."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest


MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"local GGUF fixture not found: {MODEL}",
)


def _f32_to_bf16_bits(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=np.float32)
    return (contiguous.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def test_qwen38_nextn_proposal_head_rowtile_matches_direct_parent(
    monkeypatch: pytest.MonkeyPatch,
    hip_test_target_arch: str,
) -> None:
    if hip_test_target_arch != "gfx1151":
        pytest.skip("Qwen3.8 proposal-head route is qualified only on gfx1151")
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")

    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1151")
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")

    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.linear.lm_head import argmax_f32_rows_i32
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
    )
    from hipengine.runtime.gguf_linear import GGUF_OUTPUT_F32, launch_gguf_linear
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    # The package route must own every wide proposal width (M2 admits rows5-8).
    assert hasattr(Qwen35GGUFResidentSession, "_proposal_lm_head_rowtile")

    runtime = get_hip_runtime()
    guard_words = 32
    guard_value = np.uint32(0x4A37C0DE)
    rng = np.random.default_rng(0xE1B)

    with Qwen35GGUFResidentSession(
        MODEL,
        backend="hip_gfx1151",
        max_batch_size=4,
        max_sequence_length=1024,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        assert session.runner is not None
        weight = session.runner.weights.root("lm_head")
        hidden_size = int(session.runner.hidden_size)
        vocab_size = int(session.runner.vocab_size)
        assert (hidden_size, vocab_size) == (5120, 248320)

        for rows in (2, 3, 4, 5, 6, 7, 8):
            hidden_host = _f32_to_bf16_bits(
                rng.normal(0.0, 0.2, size=(rows, hidden_size)).astype(np.float32)
            )
            hidden = malloc(hidden_host.nbytes, runtime=runtime)
            reference = malloc(rows * vocab_size * np.dtype(np.float32).itemsize, runtime=runtime)
            guarded_words = rows * vocab_size + 2 * guard_words
            candidate = malloc(guarded_words * np.dtype(np.uint32).itemsize, runtime=runtime)
            guarded_host = np.full((guarded_words,), guard_value, dtype=np.uint32)
            candidate_ptr = candidate.ptr + guard_words * np.dtype(np.uint32).itemsize
            try:
                runtime.memcpy(
                    hidden.ptr,
                    host_array_ptr(hidden_host),
                    hidden_host.nbytes,
                    HipMemcpyKind.HOST_TO_DEVICE,
                )
                runtime.memcpy(
                    candidate.ptr,
                    host_array_ptr(guarded_host),
                    guarded_host.nbytes,
                    HipMemcpyKind.HOST_TO_DEVICE,
                )
                launch_gguf_linear(
                    weight,
                    hidden.ptr,
                    reference.ptr,
                    rows=rows,
                    in_features=hidden_size,
                    out_features=vocab_size,
                    output_dtype=GGUF_OUTPUT_F32,
                    runtime=runtime,
                )
                candidate_functions = (
                    (
                        ("package", None),
                        (
                            "exact-primitive",
                            gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
                        ),
                    )
                )
                reference_host = np.empty((rows, vocab_size), dtype=np.float32)
                runtime.memcpy(
                    host_array_ptr(reference_host),
                    reference.ptr,
                    reference_host.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                for candidate_name, candidate_fn in candidate_functions:
                    runtime.memcpy(
                        candidate.ptr,
                        host_array_ptr(guarded_host),
                        guarded_host.nbytes,
                        HipMemcpyKind.HOST_TO_DEVICE,
                    )
                    if candidate_fn is None:
                        assert session._proposal_lm_head_rowtile(
                            hidden.ptr,
                            candidate_ptr,
                            rows,
                            runtime=runtime,
                        )
                    else:
                        candidate_fn(
                            hidden.ptr,
                            weight.allocation("tiles").tensor.ptr,
                            candidate_ptr,
                            rows,
                            hidden_size,
                            vocab_size,
                            runtime=runtime,
                        )
                    runtime.device_synchronize()
                    candidate_host = np.empty((rows, vocab_size), dtype=np.float32)
                    runtime.memcpy(
                        host_array_ptr(candidate_host),
                        candidate_ptr,
                        candidate_host.nbytes,
                        HipMemcpyKind.DEVICE_TO_HOST,
                    )
                    runtime.memcpy(
                        host_array_ptr(guarded_host),
                        candidate.ptr,
                        guarded_host.nbytes,
                        HipMemcpyKind.DEVICE_TO_HOST,
                    )
                    np.testing.assert_array_equal(
                        candidate_host,
                        reference_host,
                        err_msg=f"rows={rows} candidate={candidate_name}",
                    )
                    assert np.all(guarded_host[:guard_words] == guard_value)
                    assert np.all(guarded_host[-guard_words:] == guard_value)

                session._ensure_verify_lm_head_buffers(rows, runtime=runtime)
                assert session._verify_lm_block_values is not None
                assert session._verify_lm_block_indices_i32 is not None
                assert session._verify_lm_out_indices_i32 is not None
                assert session._verify_lm_out_values is not None
                argmax_f32_rows_i32(
                    candidate_ptr,
                    session._verify_lm_block_values.ptr,
                    session._verify_lm_block_indices_i32.ptr,
                    session._verify_lm_out_indices_i32.ptr,
                    session._verify_lm_out_values.ptr,
                    rows,
                    vocab_size,
                    threads=session._lm_head_threads,
                    library=session._lm_head_library,
                    runtime=runtime,
                )
                ids_host = np.empty((rows,), dtype=np.int32)
                values_host = np.empty((rows,), dtype=np.float32)
                runtime.memcpy(
                    host_array_ptr(ids_host),
                    session._verify_lm_out_indices_i32.ptr,
                    ids_host.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                runtime.memcpy(
                    host_array_ptr(values_host),
                    session._verify_lm_out_values.ptr,
                    values_host.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                expected_ids = np.argmax(reference_host, axis=1).astype(np.int32)
                np.testing.assert_array_equal(ids_host, expected_ids)
                np.testing.assert_array_equal(
                    values_host,
                    reference_host[np.arange(rows), expected_ids],
                )
            finally:
                for buffer in (hidden, reference, candidate):
                    free(buffer, runtime=runtime)

        # The unchanged generic GPU argmax must keep its lowest-ID tie contract.
        tie_rows = 2
        tie_logits = np.full((tie_rows, vocab_size), -10.0, dtype=np.float32)
        tie_logits[:, 7] = 42.0
        tie_logits[:, 11] = 42.0
        tie_device = malloc(tie_logits.nbytes, runtime=runtime)
        try:
            runtime.memcpy(
                tie_device.ptr,
                host_array_ptr(tie_logits),
                tie_logits.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            session._ensure_verify_lm_head_buffers(tie_rows, runtime=runtime)
            argmax_f32_rows_i32(
                tie_device.ptr,
                session._verify_lm_block_values.ptr,
                session._verify_lm_block_indices_i32.ptr,
                session._verify_lm_out_indices_i32.ptr,
                session._verify_lm_out_values.ptr,
                tie_rows,
                vocab_size,
                threads=session._lm_head_threads,
                library=session._lm_head_library,
                runtime=runtime,
            )
            ids_host = np.empty((tie_rows,), dtype=np.int32)
            runtime.memcpy(
                host_array_ptr(ids_host),
                session._verify_lm_out_indices_i32.ptr,
                ids_host.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            np.testing.assert_array_equal(ids_host, np.full((tie_rows,), 7, dtype=np.int32))
        finally:
            free(tie_device, runtime=runtime)
