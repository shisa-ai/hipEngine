"""Enqueue-only target verify must write aux-layer hidden rows into graph-owned
device buffers (no per-cycle D2H), bit-exact vs the eager host capture.

This is the part-(a) prerequisite for graph-captured generic-drafter verify:
``capture_layer_output_hidden`` was previously refused in enqueue-only mode,
which blocked EAGLE-4/DSpine from keeping their aux rows on device. Skips without
HIP or the local Qwen3.8-27B GGUF fixture.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")

SEED = [1, 2, 13, 14, 198, 264, 374, 11, 323, 279, 304, 369, 429, 1, 13, 198]
# EAGLE-4 aux tap layers for Qwen3.8-27B (post-layer depths l+1 are used by the
# drafter; verify captures the layer ids l). A small spread exercises the map.
AUX_LAYERS = [1, 23, 47]


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def test_enqueue_only_verify_captures_aux_rows_to_device_buffers(
    hip_test_target_arch: str,
) -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    import os
    os.environ["HIPENGINE_HIP_ARCH"] = hip_test_target_arch

    from hipengine.core.dtype import DType
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    block = [SEED[-1], 71093, 12305]  # rows=3 continuation block
    rows = len(block)
    failures = []
    with Qwen35GGUFResidentSession(
        str(MODEL), backend="hip_gfx1151", compiler_version=None,
        require_cached_build=False, max_sequence_length=4096, max_batch_size=1,
    ) as session:
        hidden_size = int(session.runner.hidden_size)
        layer_count = len(session.runner.weights.config.layer_types)
        aux_layers = [l for l in AUX_LAYERS if l < layer_count]
        session.prefill(SEED, return_logits=False)
        snap = session._linear_state_snapshot()
        try:
            # Bulk mode is the supported enqueue-only config without a reusable
            # graph scratch. The aux D2D capture is mode-independent (it runs in
            # the layer-stack output callback); native-mode graph verify needs the
            # reusable scratch + dynamic cursor + native row scratches that the
            # generic-cycle integration provides, so it is exercised there.
            for mode in ("bulk",):
                # Eager reference: host aux rows + greedy target tokens.
                session._restore_linear_state_snapshot(snap, position=len(SEED))
                ref = session.verify_target_block(
                    block, bulk_attention_mode=mode,
                    capture_layer_output_hidden=aux_layers,
                )
                ref_tokens = [int(t) for t in ref.token_ids]
                ref_aux = {l: np.asarray(ref.layer_output_hidden[l], dtype=np.float32) for l in aux_layers}

                # Enqueue-only: aux rows land in graph-owned device F32 buffers.
                session._restore_linear_state_snapshot(snap, position=len(SEED))
                stream = runtime.stream_create()
                token_buf = DeviceBuffer(ptr=runtime.malloc(rows * 8), nbytes=rows * 8)
                top1_buf = DeviceBuffer(ptr=runtime.malloc(rows * 4), nbytes=rows * 4)
                aux_bufs = {
                    l: DeviceBuffer(ptr=runtime.malloc(rows * hidden_size * 4),
                                    nbytes=rows * hidden_size * 4)
                    for l in aux_layers
                }
                try:
                    staged = np.asarray(block, dtype=np.int64)
                    from hipengine.core.memory import copy_host_array_to_device
                    copy_host_array_to_device(token_buf, staged, staged.nbytes, runtime=runtime)
                    session.verify_target_block(
                        block, bulk_attention_mode=mode, stream=stream,
                        capture_layer_output_hidden=aux_layers,
                        _pre_staged_token_ids_ptr=token_buf.ptr,
                        _target_top1_i32_ptr=top1_buf.ptr,
                        _graph_layer_output_hidden_bufs=aux_bufs,
                        _enqueue_only=True,
                    )
                    runtime.device_synchronize()
                    got_tokens_raw = np.empty(rows, dtype=np.int32)
                    copy_device_to_host(host_array_ptr(got_tokens_raw), top1_buf, got_tokens_raw.nbytes, runtime=runtime)
                    got_tokens = [int(t) for t in got_tokens_raw]
                    if got_tokens != ref_tokens:
                        failures.append(f"mode={mode}: tokens {got_tokens} != eager {ref_tokens}")
                    for l in aux_layers:
                        got_aux_raw = np.empty(rows * hidden_size, dtype=np.float32)
                        copy_device_to_host(host_array_ptr(got_aux_raw), aux_bufs[l], got_aux_raw.nbytes, runtime=runtime)
                        got_aux = got_aux_raw.reshape(rows, hidden_size)
                        if not np.array_equal(got_aux, ref_aux[l]):
                            max_diff = float(np.max(np.abs(got_aux - ref_aux[l])))
                            failures.append(f"mode={mode} layer={l}: aux mismatch max_diff={max_diff}")
                finally:
                    for b in [token_buf, top1_buf, *aux_bufs.values()]:
                        free(b, runtime=runtime)
                    runtime.stream_destroy(stream)
        finally:
            session._free_linear_state_snapshot(snap)
    assert not failures, "; ".join(failures)


def test_enqueue_only_verify_rejects_aux_capture_without_device_destination(
    hip_test_target_arch: str,
) -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    import os
    os.environ["HIPENGINE_HIP_ARCH"] = hip_test_target_arch
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, free
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    runtime = get_hip_runtime()
    block = [SEED[-1], 71093, 12305]
    rows = len(block)
    with Qwen35GGUFResidentSession(
        str(MODEL), backend="hip_gfx1151", compiler_version=None,
        require_cached_build=False, max_sequence_length=4096, max_batch_size=1,
    ) as session:
        session.prefill(SEED, return_logits=False)
        stream = runtime.stream_create()
        token_buf = DeviceBuffer(ptr=runtime.malloc(rows * 8), nbytes=rows * 8)
        top1_buf = DeviceBuffer(ptr=runtime.malloc(rows * 4), nbytes=rows * 4)
        try:
            from hipengine.core.memory import copy_host_array_to_device
            staged = np.asarray(block, dtype=np.int64)
            copy_host_array_to_device(token_buf, staged, staged.nbytes, runtime=runtime)
            with pytest.raises(ValueError, match="graph-owned"):
                session.verify_target_block(
                    block, bulk_attention_mode="native", stream=stream,
                    capture_layer_output_hidden=[1],
                    _pre_staged_token_ids_ptr=token_buf.ptr,
                    _target_top1_i32_ptr=top1_buf.ptr,
                    _enqueue_only=True,
                )
        finally:
            free(token_buf, runtime=runtime)
            free(top1_buf, runtime=runtime)
            runtime.stream_destroy(stream)
