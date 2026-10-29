"""Host contracts for LC-D3 geometry references and leaf resource ownership."""

import ctypes
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import laguna_global_fixedshape_reduce_leaf as leaf
from hipengine.core.dtype import DType
from hipengine.kernels.hip_gfx1100.attention import laguna_kv as primitives


_PREFIX = "laguna_global_attention_decode_"
_EXACT = "split_exact_gated_gqa6_"
_CTX = "split_gated_gqa6_"
_END = "_bf16_spans"
_GEOMETRIES = [
    ("split-gqa6-dim32-vstage64-t256", _EXACT+"dim32_vstage64_t256", _EXACT+"dim32_vstage64"),
    ("split-gqa6-deferrednorm-dim32-vstage64", _EXACT+"deferrednorm_dim32_vstage64", _EXACT+"dim32_vstage64"),
    ("split-gqa6-tile4-dim32-vstage64", _EXACT+"tile4_dim32_vstage64", _EXACT+"dim32_vstage64"),
    ("split-gqa6-dim64-vstage64", _EXACT+"dim64_vstage64", _EXACT+"dim32_vstage64"),
    ("split-gqa6-dim64-vstage32", _EXACT+"dim64_vstage32", _EXACT+"dim32_vstage64"),
    ("split-qhead-dim32-direct", "split_exact_gated_qhead_dim32_direct", _EXACT+"dim32_vstage64"),
    ("split-gqa6-dim32-vstage64-ctx4096", _CTX+"dim32_vstage64_ctx4096", "split_exact_gated"),
    ("split-gqa6-dim64-vstage64-ctx4096", _CTX+"dim64_vstage64_ctx4096", _CTX+"dim32_vstage64_ctx4096"),
    ("split-gqa6-dim64-vstage64-ctx4096-compensated", _CTX+"dim64_vstage64_ctx4096_compensated", _CTX+"dim32_vstage64_ctx4096_compensated"),
]


@pytest.mark.parametrize("candidate,symbol,control", _GEOMETRIES)
def test_lcd3_wrapper_reaches_distinct_c_symbol_and_complete_spans(candidate, symbol, control):
    captured = []
    class Function:
        def __call__(self, *args):
            captured.append(tuple(arg.value for arg in args))
            return 0
    def tensor(ptr, numel):
        return SimpleNamespace(ptr=ptr, numel=numel)
    spans = SimpleNamespace(spans_mode="uniform", storage_dtype=DType.BF16,
        base_offsets=tensor(9, 32), live_counts=tensor(10, 1),
        token_positions=tensor(11, 8192), evict_mask=tensor(12, 8192),
        row_positions=tensor(13, 1), max_live_count=8192)
    fn = getattr(primitives, _PREFIX+symbol+_END)
    library = SimpleNamespace(**{"hipengine_"+_PREFIX+symbol+_END: Function()})
    fn(1, 2, 3, 4, 5, 6, 7, 8, spans, 6001, 8192, 48, 8, 128,
        0.125, stream=17, library=library,
        runtime=SimpleNamespace(check=lambda err: pytest.fail(str(err))))
    assert captured == [tuple(range(1, 14))+(8192, 256, 32, 8192, 6001, 48, 8, 128, 0.125, 17)]


@pytest.fixture
def harness(monkeypatch):
    allocated, freed, calls, cache_calls = [], [], [], []
    runtime = SimpleNamespace(device_synchronize=lambda: None,
        memcpy=lambda dst, src, nbytes, kind: ctypes.memset(dst, 0, nbytes))
    cache = SimpleNamespace(
        prepare_rows=lambda rows: cache_calls.append("prepare"),
        append_rows=lambda *a, **kw: cache_calls.append("append"),
        commit_rows=lambda: cache_calls.append("commit"),
        prepare_position=lambda pos: cache_calls.append("position"),
        layer=lambda layer: SimpleNamespace(key_cache=SimpleNamespace(ptr=101),
            value_cache=SimpleNamespace(ptr=102), spans=object()),
        free=lambda: cache_calls.append("free"),
    )
    def malloc(nbytes, **kwargs):
        buffer = SimpleNamespace(ptr=len(allocated)+1, nbytes=nbytes)
        allocated.append(buffer)
        return buffer
    monkeypatch.setattr(leaf, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(leaf, "build_laguna_kv_attention", lambda **kw: object())
    monkeypatch.setattr(leaf, "allocate_laguna_kv_cache", lambda *a, **kw: cache)
    monkeypatch.setattr(leaf, "malloc", malloc)
    monkeypatch.setattr(leaf, "free", lambda buffer, **kw: freed.append(buffer.ptr))
    monkeypatch.setattr(leaf, "copy_host_to_device", lambda *a, **kw: None)
    monkeypatch.setattr(leaf, "_download", lambda runtime, buffer, dtype: np.zeros(leaf.Q_HEADS*leaf.HEAD_DIM, dtype=dtype))
    monkeypatch.setattr(leaf, "_tracked_status", lambda: [])
    monkeypatch.setattr(leaf, "_time_ms", lambda runtime, fn, burst: (fn(), 1.0)[1])
    for name in {name for _, candidate, control in _GEOMETRIES for name in (candidate, control)}:
        monkeypatch.setattr(leaf, _PREFIX+name+_END,
            lambda *a, _name=name, **kw: calls.append((_name, a, kw)))
    args = SimpleNamespace(samples=2, warmups=0, burst=1, capacity=8192,
        live_counts="513,4097,6001", allow_dirty=True, compiler_version_file=None,
        require_cached_build=True, candidate=_GEOMETRIES[0][0])
    return SimpleNamespace(args=args, runtime=runtime, cache=cache, allocated=allocated,
        freed=freed, calls=calls, cache_calls=cache_calls, malloc=malloc)


@pytest.mark.parametrize("candidate,symbol,control", _GEOMETRIES)
def test_lcd3_leaf_keeps_distinct_current_geometry_references(harness, monkeypatch, candidate, symbol, control):
    h = harness
    monkeypatch.setattr(sys, "argv", ["laguna_global_fixedshape_reduce_leaf.py", "--candidate", candidate])
    assert leaf._parse_args().candidate == candidate
    h.args.candidate = candidate
    result = leaf.run(h.args)
    assert [name for name, _, _ in h.calls] == [control, symbol, control, symbol, symbol, control]*3
    assert all(args[:3] == (3, 101, 102) for _, args, _ in h.calls)
    assert result["candidate_kind"] == candidate and result["performance_claim"] is False
    assert all(row["correctness"]["context_f32_byte_exact"] and row["correctness"]["gated_bf16_byte_exact"] for row in result["rows"])
    assert h.cache_calls == ["prepare", "append", "commit", "position", "free"]
    assert h.freed == list(reversed([buffer.ptr for buffer in h.allocated]))


@pytest.mark.parametrize("candidate", ["split-gqa6-dim32-vstage64-ctx4096", "split-gqa6-dim64-vstage64-ctx4096"])
def test_context_split_drift_policy_is_relative_to_its_named_parent(harness, monkeypatch, candidate):
    h = harness
    h.args.candidate = candidate
    h.args.live_counts = "4097"
    downloads = []
    def download(runtime, buffer, dtype):
        downloads.append(None)
        return np.full(leaf.Q_HEADS*leaf.HEAD_DIM, len(downloads)%2, dtype=dtype)
    monkeypatch.setattr(leaf, "_download", download)
    if "dim32" in candidate:
        result = leaf.run(h.args)
        assert not result["rows"][0]["correctness"]["context_f32_byte_exact"]
        assert result["performance_claim"] is False
    else:
        with pytest.raises(AssertionError, match="not byte-exact"):
            leaf.run(h.args)
    assert h.freed == list(reversed([buffer.ptr for buffer in h.allocated]))


@pytest.mark.parametrize("fail_at", [2, 5, 10])
def test_global_geometry_leaf_frees_partial_allocations(harness, monkeypatch, fail_at):
    h = harness
    def failing_malloc(nbytes, **kw):
        if len(h.allocated)+1 == fail_at:
            raise MemoryError("synthetic global geometry allocation failure")
        return h.malloc(nbytes, **kw)
    monkeypatch.setattr(leaf, "malloc", failing_malloc)
    with pytest.raises(MemoryError, match="synthetic global geometry"):
        leaf.run(h.args)
    assert h.cache_calls == ["free"]
    assert h.freed == list(reversed([buffer.ptr for buffer in h.allocated]))


@pytest.mark.parametrize("failure", ["body", "cache_close"])
def test_global_geometry_leaf_frees_after_kernel_or_cache_close_failure(harness, monkeypatch, failure):
    h = harness
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic global geometry failure")
    if failure == "body":
        monkeypatch.setattr(leaf, _PREFIX+_GEOMETRIES[0][2]+_END, fail)
    else:
        h.cache.free = fail
    with pytest.raises(RuntimeError, match="synthetic global geometry"):
        leaf.run(h.args)
    assert h.freed == list(reversed([buffer.ptr for buffer in h.allocated]))


@pytest.mark.parametrize("failure", ["stop_create", "body", "stop_destroy", None])
def test_global_leaf_timing_releases_events(failure):
    created, destroyed = [], []
    def create():
        if failure == "stop_create" and created:
            raise RuntimeError("synthetic global timing failure")
        created.append(len(created)+1)
        return created[-1]
    def destroy(event):
        destroyed.append(event)
        if failure == "stop_destroy" and event == 2:
            raise RuntimeError("synthetic global timing failure")
    def body():
        if failure == "body":
            raise RuntimeError("synthetic global timing failure")
    runtime = SimpleNamespace(event_create=create, event_destroy=destroy,
        event_record=lambda event: None, event_synchronize=lambda event: None,
        event_elapsed_time_ms=lambda *a: 4.0)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic global timing"):
            leaf._time_ms(runtime, body, 2)
    else:
        assert leaf._time_ms(runtime, body, 2) == 2.0
    assert destroyed == list(reversed(created))
