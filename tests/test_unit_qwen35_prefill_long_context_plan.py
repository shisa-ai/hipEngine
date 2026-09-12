"""CPU launch contracts for bounded native-prefill score storage."""
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.kernels.hip_gfx1100.attention import paged_attn_decode as kernel


def _spans(rows, context):
    blocks = (context + 255) // 256
    def tensor(ptr, shape, dtype):
        return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))
    return KVLiveSpans.paged_uniform(
        block_table=tensor(0x1000, (rows, blocks), "int32"),
        live_counts=tensor(0x2000, (rows,), "int64"),
        row_positions=tensor(0x3000, (rows,), "int64"),
        max_live_count=context, storage_dtype="bf16", span_role="prefill",
    )


def _launch(context, rows=4, **overrides):
    kwargs = dict(
        split_partial_out_ptr=0x600000, split_partial_m_ptr=0x700000,
        split_partial_l_ptr=0x800000, split_batch_rows=2,
        split_count=(context + 255) // 256, library=object(), runtime=object(),
    )
    gate_stride1 = overrides.pop("gate_stride1", 256)
    kwargs.update(overrides)
    return kernel.qwen35_paged_full_attn_prefill_gqa_gate_bf16_spans(
        0x100000, 0x200000, 0x300000, 0x400000, 0x500000,
        _spans(rows, context), rows, context, 256, 24, 4, 256, gate_stride1, 1, 0.0625,
        **kwargs,
    )


@pytest.mark.parametrize("context", [15872, 16064, 16065, 16128, 81920])
def test_native_prefill_bounds_shared_memory_and_reuses_owned_workspace(monkeypatch, context):
    calls = []
    monkeypatch.setattr(kernel, "_launch_prefill_gqa_gate", lambda *args, **kwargs: calls.append((args, kwargs)))
    _launch(context)
    if context <= 16064:
        assert len(calls) == 1
        assert calls[0][0][0] == kernel._SYMBOL_PREFILL_GQA_GATE_BF16
        return
    assert len(calls) == 2
    for index, (args, kwargs) in enumerate(calls):
        assert args[0] == "hipengine_qwen35_paged_full_attn_prefill_gqa_gate_bf16_global_scores_spans"
        assert args[1] == 0x100000 + index * 2 * 24 * 256 * 4
        assert args[4] == 0x400000 + index * 2 * 24 * 256 * 2
        assert args[5] == 0x500000 + index * 2 * 24 * 256 * 2
        assert args[6].live_counts.ptr == 0x2000 + index * 2 * 8
        assert args[6].row_positions.ptr == 0x3000 + index * 2 * 8
        assert args[7:9] == (2, context)
        assert kwargs["score_workspace_ptr"] == 0x600000
        assert kwargs["score_stride"] == context


@pytest.mark.parametrize("overrides", [
    {"split_partial_out_ptr": 0},
    {"split_batch_rows": 0},
    {"split_count": 0},
    {"split_count": 1},
])
def test_native_prefill_long_context_rejects_missing_or_small_workspace(monkeypatch, overrides):
    monkeypatch.setattr(kernel, "_launch_prefill_gqa_gate", lambda *args, **kwargs: pytest.fail("invalid scratch must fail before launch"))
    with pytest.raises(ValueError, match="workspace"):
        _launch(16128, **overrides)


def test_global_scores_reduce_batch_size_to_actual_scratch_capacity(monkeypatch):
    calls = []
    monkeypatch.setattr(kernel, "_launch_prefill_gqa_gate", lambda *args, **kwargs: calls.append((args, kwargs)))
    _launch(16128, split_count=32, gate_stride1=512)
    assert len(calls) == 4
    for row, (args, _) in enumerate(calls):
        assert args[7] == 1
        assert args[4] == 0x400000 + row * 24 * 512 * 2
        assert args[5] == 0x500000 + row * 24 * 256 * 2


def test_global_score_variant_preserves_single_page_table_rank(monkeypatch):
    calls = []
    monkeypatch.setattr(kernel, "_launch_prefill_gqa_gate", lambda *args, **kwargs: calls.append((args, kwargs)))
    _launch(128, global_score_workspace=True)
    assert len(calls) == 2
    assert all(args[6].base_offsets.shape == (2, 1) for args, _ in calls)
