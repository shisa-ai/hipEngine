"""Host contracts for live SWA value-tail leaf comparisons and cleanup."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.attention import laguna_kv as primitives
from hipengine.kernels.registry import KernelKey, is_registered
from scripts import laguna_swa_fixed512_reduce_leaf as leaf


_PREFIX = "mixed40-"
_TAIL = "exp32-producer-max-gate-stage-pcache-tail-producer-"
_END = "idle-vec4-denom-probability-vstage64-vec16-direct-assume-exp"
_VALUE = _PREFIX + _TAIL + "value-tail-" + _END
_LOCAL = _PREFIX + "local512-" + _TAIL + "value-tail-" + _END
_STAGE128 = _LOCAL.replace("vstage64", "vstage128")
_DUAL = _STAGE128.replace("tail-producer", "dual-tail-producer")


def _wrapper(name):
    return "laguna_swa_attention_decode_fused_exact_gated_" + name.replace("-", "_") + "_fixed512_bf16_spans"


class _Runtime:
    def device_synchronize(self):
        pass


@pytest.fixture
def harness(monkeypatch):
    runtime = _Runtime()
    cache_calls = []
    cache = SimpleNamespace(
        prepare_rows=lambda rows: cache_calls.append(("prepare", rows)),
        append_rows=lambda *a, **kw: cache_calls.append(("append_rows",)),
        commit_rows=lambda: cache_calls.append(("commit",)),
        prepare_position=lambda pos: cache_calls.append(("position", pos)),
        append=lambda *a, **kw: cache_calls.append(("append",)),
        layer=lambda layer: SimpleNamespace(
            key_cache=SimpleNamespace(ptr=101), value_cache=SimpleNamespace(ptr=102),
            spans=object(),
        ),
        free=lambda: cache_calls.append(("free",)),
    )
    allocated = []
    freed = []
    calls = []

    def malloc(nbytes, **kw):
        buffer = SimpleNamespace(ptr=len(allocated) + 1, nbytes=nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(leaf, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(leaf, "build_laguna_kv_attention", lambda **kw: object())
    monkeypatch.setattr(leaf, "allocate_laguna_kv_cache", lambda *a, **kw: cache)
    monkeypatch.setattr(leaf, "malloc", malloc)
    monkeypatch.setattr(leaf, "free", lambda buffer, **kw: freed.append(buffer.ptr))
    monkeypatch.setattr(leaf, "copy_host_to_device", lambda *a, **kw: None)
    monkeypatch.setattr(leaf, "_download", lambda runtime, buffer, dtype: np.zeros(leaf.Q_HEADS * leaf.HEAD_DIM, dtype=dtype))
    monkeypatch.setattr(leaf, "_tracked_status", lambda: [])
    monkeypatch.setattr(leaf, "_time_ms", lambda runtime, fn, burst: (fn(), 1.0)[1])
    for name in (_VALUE, _LOCAL, _STAGE128, _DUAL, _VALUE.replace("value-tail-", ""), _LOCAL.replace("value-tail-", "")):
        monkeypatch.setattr(leaf, _wrapper(name), lambda *a, _name=name, **kw: calls.append((_name, a, kw)))
    args = SimpleNamespace(
        samples=1, warmups=0, burst=1, allow_dirty=True,
        compiler_version_file=None, require_cached_build=True, candidate=_VALUE,
    )
    return SimpleNamespace(args=args, cache=cache, cache_calls=cache_calls, allocated=allocated, freed=freed, calls=calls, malloc=malloc)


@pytest.mark.parametrize("candidate,control", [
    (_VALUE, _VALUE.replace("value-tail-", "")),
    (_LOCAL, _LOCAL.replace("value-tail-", "")),
    (_STAGE128, _LOCAL),
    (_DUAL, _STAGE128),
])
def test_value_tail_leaf_keeps_distinct_live_reference_pairs(harness, monkeypatch, candidate, control):
    monkeypatch.setattr(sys, "argv", ["laguna_swa_fixed512_reduce_leaf.py", "--candidate", candidate])
    assert leaf._parse_args().candidate == candidate
    harness.args.candidate = candidate
    result = leaf.run(harness.args)
    assert [call[0] for call in harness.calls] == [control, candidate, control, candidate]
    assert result["candidate_kind"] == candidate
    assert result["correctness"]["context_f32_byte_exact"]
    assert result["correctness"]["gated_bf16_byte_exact"]
    assert result["performance_claim"] is False
    assert result["protocol"] == {"samples": 1, "warmups": 0, "burst": 1}
    assert [call[0] for call in harness.cache_calls] == ["prepare", "append_rows", "commit", "position", "append", "free"]
    assert harness.freed == list(reversed([buffer.ptr for buffer in harness.allocated]))


@pytest.mark.parametrize("fail_at", [2, 5, 10])
def test_value_tail_leaf_frees_partial_allocations(harness, monkeypatch, fail_at):
    def failing_malloc(nbytes, **kw):
        if len(harness.allocated) + 1 == fail_at:
            raise MemoryError("synthetic leaf allocation failure")
        return harness.malloc(nbytes, **kw)

    monkeypatch.setattr(leaf, "malloc", failing_malloc)
    with pytest.raises(MemoryError, match="synthetic leaf allocation failure"):
        leaf.run(harness.args)
    assert harness.cache_calls == [("free",)]
    assert harness.freed == list(reversed([buffer.ptr for buffer in harness.allocated]))


def test_leaf_timing_frees_start_event_when_stop_creation_fails():
    allocated = []
    freed = []

    def create():
        if allocated:
            raise RuntimeError("synthetic stop creation failure")
        allocated.append(1)
        return 1

    runtime = SimpleNamespace(event_create=create, event_destroy=freed.append)
    with pytest.raises(RuntimeError, match="synthetic stop creation failure"):
        leaf._time_ms(runtime, lambda: None, 1)
    assert freed == allocated


@pytest.mark.parametrize("failure", ["kernel", "mismatch", "cache_close"])
def test_value_tail_leaf_frees_allocations_after_body_or_cache_failure(harness, monkeypatch, failure):
    if failure == "kernel":
        def fail(*a, **kw):
            raise RuntimeError("synthetic kernel failure")
        monkeypatch.setattr(leaf, _wrapper(_VALUE), fail)
    elif failure == "mismatch":
        downloads = []
        def download(runtime, buffer, dtype):
            downloads.append(buffer.ptr)
            return np.full(leaf.Q_HEADS * leaf.HEAD_DIM, len(downloads), dtype=dtype)
        monkeypatch.setattr(leaf, "_download", download)
    else:
        def close():
            harness.cache_calls.append(("free",))
            raise RuntimeError("synthetic cache close failure")
        harness.cache.free = close
    with pytest.raises((RuntimeError, AssertionError)):
        leaf.run(harness.args)
    assert harness.cache_calls[-1] == ("free",)
    assert harness.freed == list(reversed([buffer.ptr for buffer in harness.allocated]))


@pytest.mark.parametrize("failure", ["none", "body", "destroy_stop"])
def test_leaf_timing_releases_both_events(failure):
    created = []
    destroyed = []
    def create():
        created.append(len(created) + 1)
        return created[-1]
    def destroy(event):
        destroyed.append(event)
        if failure == "destroy_stop" and event == 2:
            raise RuntimeError("synthetic event destruction failure")
    def body():
        if failure == "body":
            raise RuntimeError("synthetic timed body failure")
    runtime = SimpleNamespace(
        event_create=create, event_destroy=destroy,
        event_record=lambda event: None, event_synchronize=lambda event: None,
        event_elapsed_time_ms=lambda start, stop: 8.0,
    )
    if failure == "none":
        assert leaf._time_ms(runtime, body, 4) == 2.0
    else:
        with pytest.raises(RuntimeError, match="synthetic"):
            leaf._time_ms(runtime, body, 4)
    assert destroyed == list(reversed(created))


@pytest.mark.parametrize("candidate", [_VALUE, _LOCAL, _STAGE128, _DUAL])
def test_value_tail_wrappers_preserve_distinct_registered_symbols_and_spans(candidate):
    def tensor(ptr, shape, dtype):
        return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))
    spans = KVLiveSpans.sliding_ring(
        base_offsets=tensor(0x1000, (512,), "int32"),
        live_counts=tensor(0x2000, (1,), "int64"),
        token_positions=tensor(0x3000, (512,), "int64"),
        evict_mask=tensor(0x4000, (512,), "bool"),
        row_positions=tensor(0x5000, (1,), "int64"),
        capacity=512, storage_dtype="bf16",
    )
    symbol = "hipengine_" + _wrapper(candidate)
    calls = []
    class Function:
        def __call__(self, *args):
            calls.append(args)
            return 0
    library = SimpleNamespace(**{symbol: Function()})
    fn = getattr(primitives, _wrapper(candidate))
    fn(1, 2, 3, 4, 5, 6, 7, 8, spans, 512, 72, 8, 128, 128 ** -0.5,
       stream=99, library=library, runtime=SimpleNamespace())
    assert len(calls) == 1
    assert [value.value for value in calls[0][:13]] == list(range(1, 9)) + [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
    assert [value.value for value in calls[0][13:20]] == [512, 512, 512, 512, 72, 8, 128]
    assert calls[0][-1].value == 99
    variant = "swa_context_fused_exact_gated_" + candidate.replace("-", "_") + "_fixed512_spans"
    primitives.register_laguna_kv_attention_kernels()
    assert is_registered(KernelKey("hip_gfx1100", "laguna_attention_decode", "bf16", variant))
    with pytest.raises(ValueError, match="requires"):
        fn(1, 2, 3, 4, 5, 6, 7, 8, spans, 511, 72, 8, 128, 128 ** -0.5)
