"""Exactness gate for verify_target_block(advance_state_only=True).

Partial-accept rollback (task #4) replays the accepted prefix only to advance the
GDN linear / KV state; its target tokens are already known from the first
full-block pass and discarded. ``advance_state_only=True`` skips the replay's
per-row LM-head vocab GEMV + greedy sampling (~16% of the forward). This test
asserts that skip is EXACT: the committed linear state and the FP32 hidden rows
are bit-identical to a full replay, and the reused first-pass tokens match.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _read_linear_state(session) -> np.ndarray:
    """Concatenate all GDN conv + recurrent state buffers to a host array."""
    runtime = session.runtime
    chunks: list[np.ndarray] = []
    for conv, rec in zip(
        session.scratch.layer_conv_states,
        session.scratch.layer_recurrent_states,
        strict=True,
    ):
        for state in (conv, rec):
            if state is not None:
                host = np.empty(int(state.nbytes) // 4, dtype=np.float32)
                copy_device_to_host(
                    host_array_ptr(host), DeviceBuffer(state.ptr, host.nbytes), host.nbytes, runtime=runtime
                )
                chunks.append(host)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _read_hidden_seed(session) -> np.ndarray:
    runtime = session.runtime
    host = np.empty(int(session.runner.hidden_size), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(session.scratch.hidden_seed_fp32.ptr, host.nbytes),
        host.nbytes,
        runtime=runtime,
    )
    return host


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.skipif(not MODEL.exists(), reason=f"model {MODEL} not present")
def test_advance_state_only_matches_full_replay(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    prompt_ids = [760, 4087, 369, 220, 16, 17, 18, 19]
    block_rows, consumed = 5, 2

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=256) as session:
        first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
        prefix_position = int(session.position)
        snapshot = session._linear_state_snapshot()

        # Build a continuation block by greedy stepping, then roll back to prefix.
        block_inputs = [int(first.token_id)]
        current = int(first.token_id)
        for _ in range(block_rows - 1):
            step = session.step(current, return_logits=False)
            block_inputs.append(int(step.token_id))
            current = int(step.token_id)

        # Full-block pass: its first `consumed` target tokens are what the replay
        # would re-derive (verifier is causal / prefix-deterministic).
        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        full = session.verify_target_block(block_inputs)

        # Reference accepted-prefix replay (full LM-head sampling).
        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        ref = session.verify_target_block(block_inputs[:consumed])
        ref_state = _read_linear_state(session)

        # Fast accepted-prefix replay (state advance only, LM-head skipped).
        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        fast = session.verify_target_block(block_inputs[:consumed], advance_state_only=True)
        fast_state = _read_linear_state(session)

        session._free_linear_state_snapshot(snapshot)

    # Linear/KV state advance is bit-identical (same layer stack ran).
    assert ref_state.shape == fast_state.shape and ref_state.size > 0
    np.testing.assert_array_equal(ref_state, fast_state)
    # FP32 hidden rows (used for decode continuity) are bit-identical.
    np.testing.assert_array_equal(ref.hidden_seeds, fast.hidden_seeds)
    # Reusing the first-pass tokens for the accepted prefix is exact.
    assert [int(t) for t in full.token_ids[:consumed]] == [int(t) for t in ref.token_ids]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.skipif(not MODEL.exists(), reason=f"model {MODEL} not present")
def test_direct_block_state_commit_matches_replay(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    prompt_ids = [760, 4087, 369, 220, 16, 17, 18, 19]
    block_rows, consumed = 5, 2

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=256) as session:
        first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
        prefix_position = int(session.position)
        snapshot = session._linear_state_snapshot()

        block_inputs = [int(first.token_id)]
        current = int(first.token_id)
        for _ in range(block_rows - 1):
            step = session.step(current, return_logits=False)
            block_inputs.append(int(step.token_id))
            current = int(step.token_id)

        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        ref = session.verify_target_block(block_inputs[:consumed])
        ref_state = _read_linear_state(session)

        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        block = session.verify_target_block(block_inputs, capture_linear_state_rows=True)
        assert block.linear_state_rows_captured
        session._commit_verify_linear_state_row(consumed - 1, position=prefix_position + consumed)
        direct_state = _read_linear_state(session)

        session._free_linear_state_snapshot(snapshot)

    assert [int(t) for t in block.token_ids[:consumed]] == [int(t) for t in ref.token_ids]
    np.testing.assert_array_equal(block.hidden_seeds[:consumed], ref.hidden_seeds)
    np.testing.assert_array_equal(ref_state, direct_state)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.skipif(not MODEL.exists(), reason=f"model {MODEL} not present")
def test_branch_block_restore_replay_supports_corrective_step(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    prompt_ids = [760, 4087, 369, 220, 16, 17, 18, 19]

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=256) as session:
        first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
        prefix_position = int(session.position)
        snapshot = session._linear_state_snapshot()

        ref0 = session.step(int(first.token_id), return_logits=False, capture_hidden_seed_fp32=True)
        branch_token = int(ref0.token_id)
        wrong_draft_child = (branch_token + 1) % int(session.runner.vocab_size)
        ref1 = session.step(branch_token, return_logits=False, capture_hidden_seed_fp32=True)
        ref_state = _read_linear_state(session)

        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        block = session.verify_target_block(
            [int(first.token_id), wrong_draft_child],
            capture_linear_state_rows=True,
        )
        assert block.linear_state_rows_captured
        assert int(block.token_ids[0]) == branch_token
        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        replay0 = session.step(int(first.token_id), return_logits=False, capture_hidden_seed_fp32=True)
        assert int(replay0.token_id) == branch_token
        direct1 = session.step(branch_token, return_logits=False, capture_hidden_seed_fp32=True)
        direct_state = _read_linear_state(session)

        session._free_linear_state_snapshot(snapshot)

    assert int(direct1.token_id) == int(ref1.token_id)
    np.testing.assert_array_equal(ref_state, direct_state)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
@pytest.mark.skipif(not MODEL.exists(), reason=f"model {MODEL} not present")
def test_serial_exact_direct_commit_matches_wrong_branch(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    prompt_ids = [760, 4087, 369, 220, 16, 17, 18, 19]

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=256) as session:
        first = session.prefill(prompt_ids, use_bulk=True, return_logits=False)
        prefix_position = int(session.position)
        snapshot = session._linear_state_snapshot()

        ref0 = session.step(int(first.token_id), return_logits=False, capture_hidden_seed_fp32=True)
        branch_token = int(ref0.token_id)
        wrong_draft_child = (branch_token + 1) % int(session.runner.vocab_size)
        ref0_hidden = _read_hidden_seed(session)
        ref0_state = _read_linear_state(session)
        ref1 = session.step(branch_token, return_logits=False, capture_hidden_seed_fp32=True)
        ref1_state = _read_linear_state(session)

        session._restore_linear_state_snapshot(snapshot, position=prefix_position)
        block = session.verify_target_block_serial_exact(
            [int(first.token_id), wrong_draft_child],
            capture_linear_state_rows=True,
        )
        assert block.linear_state_rows_captured
        assert int(block.token_ids[0]) == branch_token
        session._commit_verify_linear_state_row(0, position=prefix_position + 1)
        direct0_state = _read_linear_state(session)
        direct1 = session.step(branch_token, return_logits=False, capture_hidden_seed_fp32=True)
        direct1_state = _read_linear_state(session)

        session._free_linear_state_snapshot(snapshot)

    np.testing.assert_array_equal(block.hidden_seeds[0], ref0_hidden)
    np.testing.assert_array_equal(ref0_state, direct0_state)
    assert int(direct1.token_id) == int(ref1.token_id)
    np.testing.assert_array_equal(ref1_state, direct1_state)
