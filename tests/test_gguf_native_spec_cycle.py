from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)

from hipengine.runtime.gguf_native_spec_cycle import (
    NativeSpecTargetGraphUnsupportedError,
    build_native_b2_target_batch,
    verify_qwen35_gguf_native_b2_target,
)


def test_build_native_b2_target_batch_uses_root_prefixed_chain_layout() -> None:
    batch = build_native_b2_target_batch(
        [101, 202, 303],
        start_position=17,
        request_id=9,
    )

    assert batch.request_ids == (9,)
    assert batch.tokens == (101, 202, 303)
    assert batch.positions == (17, 18, 19)
    assert batch.root_rows == (0,)
    assert batch.candidate_rows == (1, 2)
    assert batch.parent_rows == (-1, 0, 1)
    assert batch.draft_depths == (0, 1, 2)
    assert batch.row_to_request == (9, 9, 9)
    assert batch.active_mask == (True, True, True)
    assert batch.mode == "verify_chain"


class _FallbackSession:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    def verify_target_block(self, input_token_ids, **kwargs):
        self.calls.append((tuple(int(token) for token in input_token_ids), dict(kwargs)))
        return SimpleNamespace(token_ids=[7, 8])


def test_native_b2_target_uses_exact_python_fallback_for_unsupported_shape() -> None:
    session = _FallbackSession()

    result = verify_qwen35_gguf_native_b2_target(
        session,
        [1, 2],
        fallback=True,
        bulk_attention_mode="native",
        capture_linear_state_rows=True,
        defer_linear_state_commit=True,
    )

    assert result.token_ids == [7, 8]
    assert session.calls == [
        (
            (1, 2),
            {
                "bulk_attention_mode": "native",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": True,
                "defer_linear_state_commit": True,
            },
        )
    ]


def test_native_b2_target_can_make_unsupported_shape_a_hard_error() -> None:
    with pytest.raises(NativeSpecTargetGraphUnsupportedError, match="three rows"):
        verify_qwen35_gguf_native_b2_target(
            _FallbackSession(),
            [1, 2],
            fallback=False,
        )


def test_native_b2_target_falls_back_before_capture_when_provider_key_is_missing(
    monkeypatch,
) -> None:
    session = _FallbackSession()
    session.backend = "hip_gfx1151"
    session.runner = object()
    session.scratch = object()
    session.host_token_embedding_enabled = False
    session.use_expert_sidecar = False
    session.kv_storage_dtype = "bf16"
    session.position = 8
    monkeypatch.setattr(
        "hipengine.runtime.gguf_native_spec_cycle.resolve",
        lambda **_kwargs: None,
    )

    result = verify_qwen35_gguf_native_b2_target(session, [1, 2, 3])

    assert result.token_ids == [7, 8]
    assert session.calls == [
        (
            (1, 2, 3),
            {
                "bulk_attention_mode": "bulk",
                "use_wmma_prefill": False,
                "capture_linear_state_rows": False,
                "defer_linear_state_commit": False,
            },
        )
    ]
    assert "not registered" in session.last_native_spec_target_fallback_reason


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _read_buffer(runtime, buffer) -> np.ndarray:
    host = np.empty(int(buffer.nbytes), dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(buffer.ptr), int(buffer.nbytes)),
        host.nbytes,
        runtime=runtime,
    )
    return host


def _linear_state_row_buffers(session) -> tuple[object, ...]:
    return tuple(
        buffer
        for pair in zip(
            session._verify_linear_conv_state_rows,
            session._verify_linear_recurrent_state_rows,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )


def _read_linear_state_rows(session) -> tuple[np.ndarray, ...]:
    return tuple(
        _read_buffer(session.runtime, buffer)
        for buffer in _linear_state_row_buffers(session)
    )


def _read_resident_linear_state(session) -> tuple[np.ndarray, ...]:
    return tuple(
        _read_buffer(session.runtime, buffer)
        for pair in zip(
            session.scratch.layer_conv_states,
            session.scratch.layer_recurrent_states,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )


def _full_kv_buffers(session) -> tuple[object, ...]:
    return tuple(
        buffer
        for pair in zip(
            session.scratch.full_key_caches,
            session.scratch.full_value_caches,
            strict=True,
        )
        for buffer in pair
        if buffer is not None
    )


def _read_full_kv(session) -> tuple[np.ndarray, ...]:
    return tuple(
        _read_buffer(session.runtime, buffer)
        for buffer in _full_kv_buffers(session)
    )


def _write_full_kv(session, values: tuple[np.ndarray, ...]) -> None:
    buffers = _full_kv_buffers(session)
    assert len(buffers) == len(values)
    for buffer, value in zip(buffers, values, strict=True):
        restored = np.ascontiguousarray(value, dtype=np.uint8)
        assert restored.nbytes == int(buffer.nbytes)
        copy_host_to_device(
            buffer,
            host_array_ptr(restored),
            restored.nbytes,
            runtime=session.runtime,
        )


_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
def test_native_b2_target_graph_matches_eager_hidden_state_and_kv(monkeypatch) -> None:
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_PREFILL_DEVICE_METADATA", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    prompt = [760, 4087, 369, 220, 16, 17, 18, 19]

    with Qwen35GGUFResidentSession(
        _MODEL,
        max_sequence_length=256,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as session:
        first = session.prefill(
            prompt,
            use_bulk=True,
            return_logits=False,
            capture_hidden_seed_fp32=True,
        )
        block_inputs = [int(first.token_id), 1, 2]
        start_position = int(session.position)
        snapshot = session._linear_state_snapshot()
        kv_snapshot = _read_full_kv(session)
        try:
            eager = session.verify_target_block(
                block_inputs,
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            )
            eager_rows = _read_linear_state_rows(session)
            eager_state = _read_resident_linear_state(session)
            eager_kv = _read_full_kv(session)

            session._restore_linear_state_snapshot(snapshot, position=start_position)
            _write_full_kv(session, kv_snapshot)
            assert session._verify_hidden_seed_buf is not None
            session.runtime.memset(
                session._verify_hidden_seed_buf.ptr,
                0xA5,
                session._verify_hidden_seed_buf.nbytes,
            )
            for buffer in _linear_state_row_buffers(session):
                session.runtime.memset(buffer.ptr, 0xA5, buffer.nbytes)
            session.runtime.device_synchronize()
            with session.capture_native_spec_target_graph(
                block_inputs,
                cycle_id=31,
                transaction_id=41,
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            ) as graph:
                native = graph.launch()
                assert graph.launch_count == 1
                assert graph.native_result is not None
                assert graph.native_result.cycle_id == 31
                assert graph.native_result.transaction_id == 41
                with pytest.raises(RuntimeError, match="only once"):
                    graph.launch()

            native_rows = _read_linear_state_rows(session)
            native_state = _read_resident_linear_state(session)
            native_kv = _read_full_kv(session)
        finally:
            session._free_linear_state_snapshot(snapshot)

    assert eager.token_ids == native.token_ids
    assert np.all(np.isfinite(native.hidden_seeds))
    np.testing.assert_array_equal(native.hidden_seeds, eager.hidden_seeds)
    assert len(eager_rows) == len(native_rows) == 60
    assert all(np.array_equal(expected, actual) for expected, actual in zip(eager_rows, native_rows, strict=True))
    assert len(eager_state) == len(native_state) == 60
    assert all(np.array_equal(expected, actual) for expected, actual in zip(eager_state, native_state, strict=True))
    assert len(eager_kv) == len(native_kv) == 20
    assert all(np.array_equal(expected, actual) for expected, actual in zip(eager_kv, native_kv, strict=True))
    assert native.start_position == start_position
    assert session.position == start_position + 3
