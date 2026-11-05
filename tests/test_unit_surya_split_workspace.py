"""Host launch/allocation contract for Surya split-K partial workspaces."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans


def _fake_spans(capacity):
    def tensor(ptr, shape, dtype):
        return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))

    return KVLiveSpans.paged_dense(
        block_table=tensor(0x1000, ((capacity + 255) // 256,), "int32"),
        live_counts=tensor(0x2000, (1,), "int64"),
        token_positions=tensor(0x3000, (capacity,), "int64"),
        evict_mask=tensor(0x4000, (capacity,), "bool"),
        row_positions=tensor(0x5000, (1,), "int64"),
        capacity=capacity, block_size=256, storage_dtype="fp32",
    )


@pytest.mark.parametrize("capacity,chunk", [(2048, 32), (2304, 32), (4096, 64), (16384, 256)])
def test_partial_capacity_is_checked_before_runtime_or_build(monkeypatch, capacity, chunk):
    import hipengine.kernels.hip_gfx1100.surya.surya_ops as module

    def fail_runtime():
        raise AssertionError("runtime reached")

    monkeypatch.setattr(module, "get_hip_runtime", fail_runtime)
    _, splits = module.plan_surya_decode_splits(capacity, chunk_size=chunk)
    with pytest.raises(ValueError, match="partial.*splits"):
        module.surya_full_attn_decode_f32_spans(
            *range(1, 8), _fake_spans(capacity), 256, 8, 2, 256, 256**-0.5,
            chunk_size=chunk, partial_num_splits=splits - 1,
        )


@pytest.mark.parametrize("capacity,chunk", [(2048, 32), (2304, 32), (4096, 64), (16384, 256)])
def test_producer_and_reducer_use_same_allocated_split_geometry(monkeypatch, capacity, chunk):
    import hipengine.kernels.hip_gfx1100.surya.surya_ops as module

    calls = []

    def fake_fn(_library, name, _types):
        def launch(*args):
            calls.append((name, args))
            return 0
        return launch

    monkeypatch.setattr(module, "_fn", fake_fn)
    _, splits = module.plan_surya_decode_splits(capacity, chunk_size=chunk)
    result = module.surya_full_attn_decode_f32_spans(
        *range(1, 8), _fake_spans(capacity), 256, 8, 2, 256, 256**-0.5,
        chunk_size=chunk, partial_num_splits=splits, library=object(), runtime=object(),
    )
    assert result == (chunk, splits)
    assert calls[0][1][11].value == capacity
    assert calls[0][1][12].value == chunk
    assert calls[0][1][13].value == calls[1][1][5].value == splits


@pytest.mark.parametrize("capacity,live", [(2048, 31), (2304, 257), (4096, 1023), (4096, 4096)])
def test_runner_keeps_allocation_plan_when_live_extent_grows(monkeypatch, capacity, live):
    import hipengine.runtime.surya as module
    from hipengine.kernels.hip_gfx1100.surya.surya_ops import plan_surya_dense_spans

    runner = object.__new__(module.SuryaGpuRunner)
    runner.spec = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2, head_dim=256)
    runner._span_plan = plan_surya_dense_spans(capacity)
    runner._span_view = _fake_spans(capacity)
    runner._span_partial_out = SimpleNamespace(ptr=5)
    runner._span_partial_m = SimpleNamespace(ptr=6)
    runner._span_partial_l = SimpleNamespace(ptr=7)
    runner.surya_library = object()
    runner.runtime = object()
    runner._seq_len = live - 1
    calls = []
    monkeypatch.setattr(module, "surya_full_attn_decode_f32_spans", lambda *a, **kw: calls.append(kw))
    runner._attention_decode_spans(1, 2, 3, 4)
    assert calls[0]["chunk_size"] == runner._span_plan.chunk_size
    assert calls[0]["partial_num_splits"] == runner._span_plan.num_splits
