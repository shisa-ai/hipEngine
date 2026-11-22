import ctypes
from contextlib import closing

import numpy as np
import pytest

from hipengine.runtime.inkling2_device_attention import Inkling2DeviceAttention


@pytest.mark.parametrize("capacity", [3, 17, 49])
def test_committed_attention_block_sees_causal_kv_before_ring_commit(capacity):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    nq, nk, dim, extent = 4, 2, 67, 15
    rng = np.random.default_rng(48)
    n = capacity + 13
    args = [rng.standard_normal((n, *shape), dtype=np.float32) for shape in
            ((nq, dim), (nk, dim), (nk, dim), (nq, extent), (nq,))]
    owner = dict(runtime=get_hip_runtime(), geometry=[(capacity, nq, nk, dim, extent)],
                 launch=attn.launch_tiled_attention, write=attn.launch_cache_write, split=attn.launch_split_attention)
    staged_extents = []

    def block(*args, **kwargs):
        staged_extents.append(args[-4])
        attn.launch_attention_block(*args, **kwargs)

    with (
        closing(Inkling2DeviceAttention(**owner)) as serial,
        closing(Inkling2DeviceAttention(**owner, block=block)) as batch,
    ):
        reference = np.stack([serial.step(0, i, *row) for i, row in enumerate(zip(*args, strict=True))])
        actual = []
        gain = rng.uniform(-0.25, 1.0, (n, nq * dim)).astype(np.float32)
        blocks = ((0, 2), (2, capacity + 6), (capacity + 6, n - 1))
        for i, j in blocks:
            actual.extend(batch.step_block(0, i, *(a[i:j] for a in args), gain=gain[i:j]))
        actual.append(batch.step(0, n - 1, *(a[-1] for a in args)))
        assert staged_extents == [min(extent, capacity, end) for _, end in blocks]
        reference[:-1] *= gain[:-1].reshape(-1, nq, dim)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), reference.view(np.uint32))


@pytest.mark.parametrize("capacity", [3, 9, 33])
def test_target_attention_matches_grouped_reference_with_relative_bias_and_sink(capacity):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.attention.inkling2_target_attention import (
        launch_cache_write, launch_split_attention, launch_target_attention, launch_tiled_attention,
    )
    from hipengine.runtime.inkling2_undo import Inkling2UndoJournal

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    nq, nk, dim, extent = 6, 2, 128 if capacity != 9 else 65, 17
    runtime = get_hip_runtime()
    copies = []
    exchanges = []
    upload_sizes = []
    allocations = {}

    class TracedRuntime:
        def __getattr__(self, name):
            return getattr(runtime, name)

        def malloc(self, size):
            ptr = runtime.malloc(size)
            allocations[ptr] = size
            return ptr

        def free(self, ptr):
            runtime.free(ptr)
            del allocations[ptr]

        def memcpy(self, dst, src, size, kind):
            copies.append(kind)
            runtime.memcpy(dst, src, size, kind)

        def memcpy_async(self, dst, src, size, kind, stream):
            exchanges.append(kind)
            if kind == HipMemcpyKind.HOST_TO_DEVICE:
                upload_sizes.append(size)
            runtime.memcpy_async(dst, src, size, kind, stream)

        def stream_synchronize(self, stream):
            exchanges.append("wait")
            runtime.stream_synchronize(stream)

    geometry = [(capacity, nq, nk, dim, extent)]
    split_rows = []

    def split(*args, **kwargs):
        split_rows.append(args[-1])
        launch_split_attention(*args, **kwargs)

    owner = Inkling2DeviceAttention(runtime=TracedRuntime(), launch=launch_tiled_attention, write=launch_cache_write,
                                    geometry=geometry, split=split)
    strict = Inkling2DeviceAttention(runtime=runtime, launch=launch_target_attention, write=launch_cache_write,
                                     geometry=geometry)
    gated = Inkling2DeviceAttention(runtime=runtime, launch=launch_tiled_attention, write=launch_cache_write,
                                    geometry=geometry, split=launch_split_attention)
    rng = np.random.default_rng(14)
    keys, values, replay = [], [], []
    undo = Inkling2UndoJournal()
    try:
        for pos in range(capacity + 3):
            q, k, v, bias, sink = [rng.standard_normal(shape, dtype=np.float32) for shape in
                                   ((nq, dim), (nk, dim), (nk, dim), (nq, extent), (nq,))]
            keys.append(k)
            values.append(v)
            first = max(0, pos + 1 - capacity)
            kr, vr = (np.repeat(np.stack(a[first:]).astype(np.float64), nq // nk, axis=1) for a in (keys, values))
            scores = np.einsum("hd,thd->ht", q, kr) / np.sqrt(dim)
            for i in range(min(extent, pos + 1 - first)):
                scores[:, -1 - i] += bias[:, i]
            scores = np.concatenate((scores, sink[:, None]), axis=1)
            p = np.exp(scores - scores.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            ref = np.einsum("ht,thd->hd", p[:, :-1], vr)
            original = strict.step(0, pos, q, k, v, bias, sink)
            gain = np.linspace(-0.25, 1, nq * dim, dtype=np.float32)
            gated_out = gated.step(0, pos, q, k, v, bias, sink, gain=gain)
            np.testing.assert_array_equal(gated_out.view(np.uint32), (original * gain.reshape(nq, dim)).view(np.uint32))
            copies.clear()
            exchanges.clear()
            upload_sizes.clear()
            existing = set(allocations) if not pos else None
            out = owner.step(0, pos, q, k, v, bias, sink, undo=undo if pos >= 3 else None)
            if existing is not None:
                keys_ptr, values_ptr, spans = owner.banks[0]
                assert set(allocations) - existing == {keys_ptr}
                assert values_ptr >= keys_ptr + capacity * nk * dim * 4
                assert values_ptr % 16 == 0
                assert spans.base_offsets.ptr >= values_ptr + capacity * nk * dim * 4
                assert spans.token_positions.ptr >= spans.base_offsets.ptr + capacity * 4
                assert spans.evict_mask.ptr >= spans.token_positions.ptr + capacity * 4
                assert spans.evict_mask.ptr + capacity <= keys_ptr + allocations[keys_ptr]
            if pos:
                assert all(kind == HipMemcpyKind.DEVICE_TO_DEVICE for kind in copies)
            assert exchanges == [HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST, "wait"]
            live_extent = min(extent, capacity, pos + 1)
            assert upload_sizes == [8 + q.nbytes + k.nbytes + v.nbytes + sink.nbytes
                                    + bias[:, :live_extent].nbytes]
            if pos >= 3:
                replay.append((pos, (q, k, v, bias, sink), out))
            np.testing.assert_array_equal(out.view(np.uint32), original.view(np.uint32))
            np.testing.assert_allclose(out, ref, rtol=5e-6, atol=1e-6)
            np.testing.assert_array_equal(out.argmax(axis=-1), ref.argmax(axis=-1))
            rp = np.exp(ref - ref.max(axis=-1, keepdims=True)); rp /= rp.sum(axis=-1, keepdims=True)
            gp = np.exp(out.astype(np.float64) - out.max(axis=-1, keepdims=True)); gp /= gp.sum(axis=-1, keepdims=True)
            assert np.max(np.sum(rp * (np.log(rp) - np.log(gp)), axis=-1)) <= 0.05
        if capacity >= 16:
            assert {16, 17, capacity} <= set(split_rows)
        else:
            assert not split_rows and owner.scores == 0
        undo.finish(3)
        for pos, args, expected in replay:
            np.testing.assert_array_equal(owner.step(0, pos, *args).view(np.uint32), expected.view(np.uint32))
    finally:
        undo.finish(3)
        strict.close()
        gated.close()
        owner.close()
    assert not owner.banks
    assert not allocations


def test_chunked_target_attention_selects_by_live_span_and_replays_after_wrap():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from functools import partial

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
    from hipengine.kernels.hip_gfx1100.attention import inkling2_target_attention as attn
    from hipengine.runtime.inkling2_undo import Inkling2UndoJournal

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    arch = hip_target_arch_for_backend(backend)
    cap, n, nq, nk, d, extent = 519, 525, 4, 2, 67, 39
    rng = np.random.default_rng(77)
    q, k, v, bias, sink = (rng.standard_normal((n, *shape), dtype=np.float32) for shape in
                           ((nq, d), (nk, d), (nk, d), (nq, extent), (nq,)))
    options = dict(runtime=get_hip_runtime(), geometry=[(cap, nq, nk, d, extent)],
                   launch=partial(attn.launch_tiled_attention, target_arch=arch),
                   split=partial(attn.launch_split_attention, target_arch=arch),
                   write=partial(attn.launch_cache_write, target_arch=arch),
                   block=partial(attn.launch_attention_block, target_arch=arch))
    calls = []

    def chunked(*args, **kwargs):
        calls.append((args[-1], kwargs["partial"]))
        return attn.launch_split_chunked_attention(*args, target_arch=arch, **kwargs)

    arrays = q, k, v, bias, sink
    start = 510
    with (
        closing(Inkling2DeviceAttention(**options)) as ordered,
        closing(Inkling2DeviceAttention(**options, split_chunked=chunked)) as first,
        closing(Inkling2DeviceAttention(**options, split_chunked=chunked)) as second,
    ):
        for owner in (ordered, first, second):
            owner.step_block(0, 0, *(a[:start] for a in arrays))
        assert first.value_partials == second.value_partials == 0
        undo = Inkling2UndoJournal()
        replay = []
        try:
            for i in range(start, n):
                row = tuple(a[i] for a in arrays)
                ref = ordered.step(0, i, *row)
                got = first.step(0, i, *row, undo=undo if i >= cap else None)
                other = second.step(0, i, *row)
                if i < 512:
                    np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
                    assert not calls
                np.testing.assert_array_equal(got.view(np.uint32), other.view(np.uint32))
                np.testing.assert_allclose(got, ref, atol=2e-6, rtol=1e-5)
                if i >= cap:
                    replay.append((i, row, got))
        finally:
            undo.finish(cap)
        assert calls[0][0] == 513
        assert {ptr for _, ptr in calls} == {first.value_partials, second.value_partials}
        assert first.value_partials != second.value_partials
        for i, row, out in replay:
            np.testing.assert_array_equal(first.step(0, i, *row).view(np.uint32), out.view(np.uint32))
