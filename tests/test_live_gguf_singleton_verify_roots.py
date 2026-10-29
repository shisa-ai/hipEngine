"""Live borrowed-singleton verifier roots versus strict packed-state imports."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def owner(hip_test_target_arch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    model = Path(os.environ.get(
        "HIPENGINE_TEST_GGUF_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
    ))
    if not model.is_file():
        pytest.skip("requires a dense Qwen hybrid GGUF model")

    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND, hip_target_arch_environment
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    with pytest.MonkeyPatch.context() as patch, hip_target_arch_environment(hip_test_target_arch):
        patch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
        patch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", "0")
        with Qwen35GGUFResidentSession(
            model, backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch],
            max_batch_size=4, max_sequence_length=256,
            use_wmma_prefill=True, use_gemv_decode=True,
        ) as resident:
            sessions = [resident, *(resident.resident_slot_view(slot) for slot in (1, 2, 3))]
            try:
                for slot, session in enumerate(sessions):
                    session.prefill([101 + slot] * (5 + 3 * slot), use_bulk=True)
                yield resident, sessions
            finally:
                for session in reversed(sessions[1:]):
                    session.close()


def _read(buffer):
    from hipengine.core.memory import copy_device_to_host

    result = np.empty(int(buffer.nbytes), dtype=np.uint8)
    copy_device_to_host(result.ctypes.data, buffer, result.nbytes)
    return result


@pytest.mark.parametrize("slot,rows", ((0, 2), (1, 3), (3, 5)))
def test_singleton_roots_match_imported_state_without_mutating_peers(owner, monkeypatch, slot, rows):
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", "0")
    resident, sessions = owner
    slab = resident._target_scratch_owner
    buffers = [buffer for pair in zip(slab.layer_conv_states, slab.layer_recurrent_states)
               for buffer in pair if buffer is not None]
    before = [_read(buffer) for buffer in buffers]
    job = dict(session=sessions[slot], input_token_ids=tuple(range(221, 221 + rows)),
               use_wmma_prefill=False, capture_linear_state_rows=True,
               defer_linear_state_commit=True, defer_state_scatter=True)

    def capture():
        result = resident.verify_target_blocks_batch([job])[0]
        state_rows = [_read(buffer) for pair in zip(resident._verify_linear_conv_state_rows,
                      resident._verify_linear_recurrent_state_rows)
                      for buffer in pair if buffer is not None]
        return result.hidden_seeds.copy(), state_rows, tuple(result.token_ids)

    # The imported-state oracle has a one-slot packed owner and therefore no
    # resident-slab indexing dependency. Keep all kernel/row choices identical.
    with monkeypatch.context() as patch:
        patch.setattr(resident, "_direct_resident_linear_state", lambda *args, **kwargs: None)
        expected = capture()
    # Select the real mapping independently of automatic backend policy. The
    # test covers state addressing, not a new arithmetic/default promotion.
    original = resident._direct_resident_linear_state
    with monkeypatch.context() as patch:
        patch.setattr(resident, "_direct_resident_linear_state",
                      lambda selected: original(selected, allow_unqualified=True))
        actual = capture()
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[2] == expected[2]
    for actual_rows, expected_rows in zip(actual[1], expected[1], strict=True):
        np.testing.assert_array_equal(actual_rows, expected_rows)
    for buffer, expected_bytes in zip(buffers, before, strict=True):
        np.testing.assert_array_equal(_read(buffer), expected_bytes)
