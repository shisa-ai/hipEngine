"""RT-4: capacity-sensitive reset clearing.

``Qwen4ExpGGUFResidentModelRunner.reset()`` historically cleared the FULL K/V
capacity on every request.  The cost is proportional to allocated capacity, not to
work done, so at native capacity a short request pays for clearing storage it never
touched - the row measures 29.182 ms at 128K with 99.6% wasted.

The bound is exact rather than heuristic, and these tests pin the invariant:

* the cache is fully zeroed at allocation,
* writes only ever occur below ``position``,
* a bounded clear zeroes ``[0, round_up(position, block_size))`` and resets position,
* therefore everything at or above the bound was already zero and stays zero.

Rounding up to a whole block is REQUIRED, not conservative padding: the final
partial block does contain written bytes.

The load-bearing test is the equivalence one - a bounded clear must leave the
device in exactly the state a full clear would, which is what makes the saving
safe to take.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, host_array_ptr
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpDenseAttentionState


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def _read(runtime, buffer) -> np.ndarray:
    actual = np.empty(buffer.nbytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(actual), buffer, runtime=runtime)
    return actual


def _allocate(**overrides):
    from hipengine.core.hip import get_hip_runtime

    kwargs = dict(
        max_positions=16,
        block_size=4,
        kv_heads=2,
        head_dim=3,
        runtime=get_hip_runtime(),
    )
    kwargs.update(overrides)
    return Qwen4ExpDenseAttentionState.allocate(**kwargs), kwargs["runtime"]


def test_written_cache_bytes_bounds_to_whole_blocks() -> None:
    """The bound is the whole blocks that could hold written bytes."""
    state, _ = _allocate()
    try:
        blocks = len(state.block_host)
        bytes_per_block = state.key_cache.nbytes // blocks
        assert state.key_cache.nbytes == blocks * bytes_per_block

        # Nothing written yet: nothing to clear.
        assert state.written_cache_bytes() == 0

        # A single token still dirties its whole block.
        state.position_host[0] = 1
        assert state.written_cache_bytes() == bytes_per_block

        # Exactly one block written.
        state.position_host[0] = 4
        assert state.written_cache_bytes() == bytes_per_block

        # One past a block boundary dirties a second block.
        state.position_host[0] = 5
        assert state.written_cache_bytes() == 2 * bytes_per_block

        # The whole capacity.
        state.position_host[0] = kwargs_max = len(state.block_host) * state.block_size
        assert state.written_cache_bytes() == state.key_cache.nbytes

        # Never exceeds the buffer even if position overshoots.
        state.position_host[0] = kwargs_max + 1000
        assert state.written_cache_bytes() == state.key_cache.nbytes
    finally:
        state.close()


def test_bounded_clear_zeroes_the_written_range_and_leaves_the_rest() -> None:
    """Poison everything, clear bounded, and confirm only the bound moved."""
    state, runtime = _allocate()
    try:
        blocks = len(state.block_host)
        bytes_per_block = state.key_cache.nbytes // blocks
        poison = np.full(state.key_cache.nbytes, 0xAB, dtype=np.uint8)
        copy_host_to_device(state.key_cache, host_array_ptr(poison), runtime=runtime)
        copy_host_to_device(state.value_cache, host_array_ptr(poison), runtime=runtime)
        runtime.device_synchronize()

        # Two whole blocks plus one token written.
        state.position_host[0] = 2 * state.block_size + 1
        bound = state.written_cache_bytes()
        assert bound == 3 * bytes_per_block

        state.clear_written_cache()
        runtime.device_synchronize()
        state.set_position(0)

        key = _read(runtime, state.key_cache)
        value = _read(runtime, state.value_cache)
        np.testing.assert_array_equal(key[:bound], 0)
        np.testing.assert_array_equal(value[:bound], 0)
        # Above the bound the poison must survive - this is what proves the clear
        # is bounded rather than a full memset wearing a new name.
        np.testing.assert_array_equal(key[bound:], 0xAB)
        np.testing.assert_array_equal(value[bound:], 0xAB)
        assert state.position_host[0] == 0
    finally:
        state.close()


def test_bounded_clear_matches_full_clear_on_the_written_range() -> None:
    """A bounded clear must leave the same device state a full clear would.

    This is the load-bearing safety test: the saving is only safe because the
    region above the bound is already zero, so clearing it changes nothing.
    """
    state, runtime = _allocate()
    try:
        blocks = len(state.block_host)
        bytes_per_block = state.key_cache.nbytes // blocks

        # The invariant: allocation zeroes everything, and writes only ever land
        # below ``position``.  So the setup is "zero everywhere, then written bytes
        # below the bound" - exactly the state a bounded clear is allowed to assume.
        written = 2 * state.block_size + 1
        state.position_host[0] = written
        bound = state.written_cache_bytes()
        assert bound == 3 * bytes_per_block

        zeros = np.zeros(state.key_cache.nbytes, dtype=np.uint8)
        copy_host_to_device(state.key_cache, host_array_ptr(zeros), runtime=runtime)
        copy_host_to_device(state.value_cache, host_array_ptr(zeros), runtime=runtime)
        written_bytes = np.full(bound, 0x77, dtype=np.uint8)
        # ``nbytes`` must be passed explicitly: copy_host_to_device defaults it to
        # the DESTINATION buffer's size, which would overread this short marker.
        copy_host_to_device(
            state.key_cache, host_array_ptr(written_bytes), written_bytes.nbytes, runtime=runtime
        )
        copy_host_to_device(
            state.value_cache, host_array_ptr(written_bytes), written_bytes.nbytes, runtime=runtime
        )
        runtime.device_synchronize()

        state.clear_written_cache()
        runtime.device_synchronize()

        full = np.zeros(state.key_cache.nbytes, dtype=np.uint8)
        np.testing.assert_array_equal(_read(runtime, state.key_cache), full)
        np.testing.assert_array_equal(_read(runtime, state.value_cache), full)
    finally:
        state.close()


def test_bounded_clear_with_nothing_written_is_a_no_op() -> None:
    """An unwritten cache is untouched - the allocation memset already covers it."""
    state, runtime = _allocate()
    try:
        poison = np.full(state.key_cache.nbytes, 0xCD, dtype=np.uint8)
        copy_host_to_device(state.key_cache, host_array_ptr(poison), runtime=runtime)
        runtime.device_synchronize()
        assert state.written_cache_bytes() == 0
        state.clear_written_cache()
        runtime.device_synchronize()
        # Nothing was written, so nothing is claimed to be dirty; the buffer keeps
        # whatever it held.  This documents that the invariant is "above the bound
        # is already zero", which the allocation-time memset establishes.
        np.testing.assert_array_equal(_read(runtime, state.key_cache), 0xCD)
    finally:
        state.close()


def test_shorter_request_after_longer_sees_no_stale_bytes() -> None:
    """The classic stale-KV hazard: a long request, then a short one.

    After a bounded clear the region the long request wrote must be zero, so a
    following short request cannot observe it.
    """
    state, runtime = _allocate()
    try:
        blocks = len(state.block_host)
        bytes_per_block = state.key_cache.nbytes // blocks

        # Long request: fill everything with a marker.
        long_marker = np.full(state.key_cache.nbytes, 0x5A, dtype=np.uint8)
        copy_host_to_device(state.key_cache, host_array_ptr(long_marker), runtime=runtime)
        runtime.device_synchronize()
        state.position_host[0] = blocks * state.block_size

        state.clear_written_cache()
        runtime.device_synchronize()
        state.set_position(0)

        # Short request writes only the first block.
        short_marker = np.full(bytes_per_block, 0x11, dtype=np.uint8)
        copy_host_to_device(
            state.key_cache, host_array_ptr(short_marker), short_marker.nbytes, runtime=runtime
        )
        runtime.device_synchronize()

        key = _read(runtime, state.key_cache)
        np.testing.assert_array_equal(key[:bytes_per_block], 0x11)
        # Everything the long request wrote beyond the short one must be gone.
        np.testing.assert_array_equal(
            key[bytes_per_block:], np.zeros(state.key_cache.nbytes - bytes_per_block, dtype=np.uint8)
        )
    finally:
        state.close()


def test_bounded_reset_is_the_default_and_explicit_falsy_opts_out(monkeypatch) -> None:
    """Promoted 2026-09-30: the bounded clear is the default path.

    Exactness was proven by parent parity before promotion (identical
    ``candidate_logits_sha256`` between ``production_baseline`` and
    ``bounded_kv_reset`` on ``scripts/qwen4exp_layer2_profile_gate.py``), so the
    full-capacity clear is now an opt-out kept only for rollback/bisection.
    """
    from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_bounded_kv_reset_enabled

    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", raising=False)
    assert _qwen4_exp_bounded_kv_reset_enabled() is True

    for truthy in ("1", "true", "yes"):
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", truthy)
        assert _qwen4_exp_bounded_kv_reset_enabled() is True, truthy

    for falsy in ("0", "false", "False", ""):
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", falsy)
        assert _qwen4_exp_bounded_kv_reset_enabled() is False, falsy
