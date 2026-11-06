"""Device ownership checks for Laguna verifier prefix publication and cancellation."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("accepted", [0, 1, 3, 7])
def test_verifier_prefix_matches_control_cache_across_swa_wrap(accepted):
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import build_laguna_kv_attention
    from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
    from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache
    from hipengine.runtime.laguna_prefill_policy import LagunaCachedPrefillPolicy

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(load=True)
    config = SimpleNamespace(
        block_count=2, layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72), head_count_kv=8, key_length=128,
        value_length=128, sliding_window=512,
    )
    caches = []
    allocations = []
    seed_rows, staged_rows = 510, 7
    row_bytes = 8 * 128 * np.dtype(np.float32).itemsize

    def snapshot(cache):
        result = []
        for layer_id in range(2):
            state = cache.layer(layer_id)
            for tensor in (state.key_cache, state.value_cache, state.spans.base_offsets,
                           state.spans.live_counts, state.spans.token_positions,
                           state.spans.evict_mask):
                size = tensor.nbytes if hasattr(tensor, "nbytes") else tensor.numel * tensor.dtype.itemsize
                values = np.empty(size, dtype=np.uint8)
                runtime.memcpy(host_array_ptr(values), tensor.ptr, size, HipMemcpyKind.DEVICE_TO_HOST)
                result.append(values)
        return result

    try:
        for _ in range(2):
            cache = allocate_laguna_kv_cache(
                config, context_length=520, backend="hip_gfx1151", runtime=runtime,
                cached_prefill_policy=LagunaCachedPrefillPolicy(True, True, True, True),
            )
            caches.append(cache)
            for state in cache.layers:
                for tensor in (state.key_cache, state.value_cache):
                    runtime.memset(tensor.ptr, 0, tensor.nbytes)
        rng = np.random.default_rng(321)
        buffers = []
        for _ in range(2):
            host = rng.normal(0, 0.1, size=(seed_rows + staged_rows, 8, 128)).astype(np.float32)
            device = malloc(host.nbytes, runtime=runtime)
            allocations.append(device)
            copy_host_to_device(device, host_array_ptr(host), host.nbytes, runtime=runtime)
            buffers.append(device)
        key, value = buffers
        for cache in caches:
            cache.prepare_rows(tuple(range(seed_rows)))
            for layer_id in range(2):
                cache.append_rows(layer_id, key.ptr, value.ptr, seed_rows, library=library)
            cache.commit_rows()
            cache.evict_position(0, 200)
            cache.evict_position(1, 200)
        target, control = caches
        before = snapshot(target)
        session = object.__new__(LagunaGGUFResidentSession)
        session._closed = False
        session.weights = SimpleNamespace(config=config)
        session.kv_cache = target
        session.libraries = SimpleNamespace(kv_attention=library)
        session.position = seed_rows - 1
        session._staged_verifier_tokens = tuple(range(staged_rows))
        session.verifier_scratch = SimpleNamespace(
            key_ptr=lambda layer: key.ptr + seed_rows * row_bytes,
            value_ptr=lambda layer: value.ptr + seed_rows * row_bytes,
        )
        target.prepare_rows(tuple(range(seed_rows, seed_rows + staged_rows)))
        if accepted:
            session._commit_staged_verifier_rows(accepted, stream=0)
            control.prepare_rows(tuple(range(seed_rows, seed_rows + accepted)))
            for layer_id in range(2):
                control.append_rows(layer_id, key.ptr + seed_rows * row_bytes,
                                    value.ptr + seed_rows * row_bytes, accepted, library=library)
            control.commit_rows()
        else:
            session._discard_staged_verifier_rows()
        runtime.device_synchronize()
        assert session.position == target.position == control.position == seed_rows + accepted - 1
        assert target.pending_positions == ()
        assert session._staged_verifier_tokens is None
        actual = snapshot(target)
        for left, right in zip(actual, snapshot(control), strict=True):
            np.testing.assert_array_equal(left, right)
        if not accepted:
            for left, right in zip(actual, before, strict=True):
                np.testing.assert_array_equal(left, right)
    finally:
        for buffer in reversed(allocations):
            free(buffer, runtime=runtime)
        for cache in reversed(caches):
            cache.free()
