from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.core.dtype import DType
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str, device: Device | None = None) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, device or Device("hip", 0))


def test_kv_live_spans_accepts_uniform_paged_bridge() -> None:
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (4,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=3,
        storage_dtype="bf16",
    )

    assert spans.base_offsets.ptr == 0x1000
    assert spans.live_counts.ptr == 0x2000
    assert spans.max_live_count == 3
    assert spans.token_positions is None
    assert spans.evict_mask is None
    assert spans.storage_dtype.value == "bf16"
    assert spans.spans_mode == "uniform"
    assert spans.request_ids is None
    assert spans.row_positions is None
    assert spans.span_role == "decode"


def test_kv_scale_metadata_validates_int8_scale_tensors() -> None:
    metadata = KVScaleMetadata(
        k_scale=_tensor(0x3000, (2, 4, 1), "fp16"),
        v_scale=_tensor(0x4000, (2, 4, 1), "fp16"),
        scale_dtype=DType.FP16,
    )
    spans = KVLiveSpans.paged_uniform(
        block_table=_tensor(0x1000, (2,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        max_live_count=3,
        storage_dtype="int8_per_token_head",
        scale_metadata=metadata,
    )

    assert spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
    assert spans.scale_metadata is metadata
    assert spans.scale_metadata.granularity == "per_token_head"


def test_kv_scale_metadata_rejects_bad_shapes_and_dtypes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        KVScaleMetadata(
            k_scale=_tensor(0x3000, (2, 4, 1), "fp16"),
            v_scale=_tensor(0x4000, (2, 4, 2), "fp16"),
        )
    with pytest.raises(ValueError, match="scale tensor dtypes"):
        KVScaleMetadata(
            k_scale=_tensor(0x3000, (2, 4, 1), "fp32"),
            v_scale=_tensor(0x4000, (2, 4, 1), "fp32"),
            scale_dtype=DType.FP16,
        )
    with pytest.raises(ValueError, match="fp16 or fp32"):
        KVScaleMetadata(
            k_scale=_tensor(0x3000, (2, 4, 1), "int32"),
            v_scale=_tensor(0x4000, (2, 4, 1), "int32"),
            scale_dtype=DType.INT32,
        )
    with pytest.raises(ValueError, match="per_token_head"):
        KVScaleMetadata(
            k_scale=_tensor(0x3000, (2, 4, 1), "fp16"),
            v_scale=_tensor(0x4000, (2, 4, 1), "fp16"),
            granularity="per_channel",
        )


def test_kv_live_spans_requires_int8_scale_metadata() -> None:
    assert DType.parse("int8_per_token_head").itemsize == 1
    with pytest.raises(ValueError, match="require scale metadata"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(0x1000, (2,), "int32"),
            live_counts=_tensor(0x2000, (1,), "int64"),
            max_live_count=3,
            storage_dtype="int8_per_token_head",
        )
    with pytest.raises(ValueError, match="only valid"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(0x1000, (2,), "int32"),
            live_counts=_tensor(0x2000, (1,), "int64"),
            max_live_count=3,
            storage_dtype="bf16",
            scale_metadata=KVScaleMetadata(
                k_scale=_tensor(0x3000, (2, 4, 1), "fp16"),
                v_scale=_tensor(0x4000, (2, 4, 1), "fp16"),
            ),
        )


def test_kv_live_spans_validates_metadata_tensors() -> None:
    with pytest.raises(ValueError, match="base_offsets must be int32"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (4,), "int64"),
            live_counts=_tensor(2, (1,), "int64"),
            max_live_count=1,
            storage_dtype="bf16",
        )
    with pytest.raises(ValueError, match="live_counts must be int32 or int64"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (4,), "int32"),
            live_counts=_tensor(2, (1,), "bf16"),
            max_live_count=1,
            storage_dtype="bf16",
        )
    with pytest.raises(ValueError, match="same device"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (4,), "int32", Device("hip", 0)),
            live_counts=_tensor(2, (1,), "int64", Device("hip", 1)),
            max_live_count=1,
            storage_dtype="bf16",
        )
    with pytest.raises(ValueError, match="max_live_count"):
        KVLiveSpans.paged_uniform(
            block_table=_tensor(1, (4,), "int32"),
            live_counts=_tensor(2, (1,), "int64"),
            max_live_count=-1,
            storage_dtype="bf16",
        )
