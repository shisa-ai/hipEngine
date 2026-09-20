from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
from hipengine.runtime.gguf_native_spec_cycle import (
    NativeSpecTargetGraphUnsupportedError,
    _native_target_binding_signature,
    _validate_capture_admission,
)
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner


def _rows():
    device = Device("hip", 0)
    table = Tensor.from_handle(0x1000, (4,), DType.INT32, device)
    key = SimpleNamespace(ptr=0x2000)
    value = SimpleNamespace(ptr=0x3000)
    metadata = KVScaleMetadata(
        Tensor.from_handle(0x4000, (4, 256, 4), DType.FP32, device),
        Tensor.from_handle(0x8000, (4, 256, 4), DType.FP32, device),
        scale_dtype=DType.FP32,
    )
    rows = []
    for row in range(4):
        position = Tensor.from_handle(0x10000 + 8 * row, (1,), DType.INT64, device)
        count = Tensor.from_handle(0x11000 + 8 * row, (1,), DType.INT64, device)
        decode = KVLiveSpans.paged_uniform(
            block_table=table, live_counts=count, row_positions=position,
            max_live_count=1024, storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            scale_metadata=metadata, span_role="verify_chain",
        )
        append = replace(decode, live_counts=position)
        rows.append(SimpleNamespace(
            block_size=256, kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
            decode_spans=decode, append_spans=append,
            decode_spans_for_layer=lambda _layer, spans=decode: spans,
            append_spans_for_layer=lambda _layer, spans=append: spans,
            full_cache=lambda _layer: (key, value),
            full_bf16_mirror_cache=lambda _layer: None,
        ))
    return tuple(rows)


def test_int8_verifier_shared_spans_preserve_scale_planes():
    runner = object.__new__(Qwen35GGUFFullStackRunner)
    rows = _rows()
    result = runner._full_attn_shared_batch_spans(
        3, rows, rows=4, attention_context_limit=1024,
    )
    assert result is not None
    spans, key, value = result
    assert spans.storage_dtype == DType.INT8_PER_TOKEN_HEAD
    assert spans.scale_metadata is rows[0].decode_spans.scale_metadata
    assert spans.live_counts.shape == (4,)
    assert spans.base_offsets.shape == (4,)
    assert (key.ptr, value.ptr) == (0x2000, 0x3000)


def _session():
    return SimpleNamespace(
        backend="hip_gfx1151", position=10, runner=object(),
        scratch=SimpleNamespace(max_positions=1024),
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        kv_storage_layout="uniform", kv_scale_granularity="per_token_head",
        int8_kv_value_bf16=False, _dms_backend=None,
    )


def _admit(session, mode="native"):
    load_backend_kernel_package("hip_gfx1151")
    _validate_capture_admission(
        session, (1, 2, 3, 4), context_limit=1023,
        bulk_attention_mode=mode, use_wmma_prefill=False,
        capture_lm_head_logits=False, record_stage_timings=False,
        sync_stage_timings=False,
    )


def test_int8_native_graph_admits_uniform_storage():
    _admit(_session())


@pytest.mark.parametrize("field,value", [
    ("kv_storage_layout", "tail4_hadamard_group32"),
    ("kv_scale_granularity", "block16"),
    ("int8_kv_value_bf16", True),
    ("_dms_backend", object()),
])
def test_int8_native_graph_rejects_unimplemented_layouts_before_capture(field, value):
    session = _session()
    setattr(session, field, value)
    with pytest.raises(NativeSpecTargetGraphUnsupportedError):
        _admit(session)


def test_int8_native_graph_does_not_use_bf16_bulk_attention():
    with pytest.raises(NativeSpecTargetGraphUnsupportedError):
        _admit(_session(), mode="bulk")


def test_int8_graph_binding_signature_tracks_scale_plane_reallocation():
    session = _session()
    metadata = _rows()[0].decode_spans.scale_metadata
    session.scratch.full_kv_scale_metadata = (metadata,)
    before = _native_target_binding_signature(session)
    replacement = Tensor.from_handle(
        metadata.k_scale.ptr + 4096, metadata.k_scale.shape, metadata.k_scale.dtype,
        metadata.k_scale.device,
    )
    session.scratch.full_kv_scale_metadata = (replace(metadata, k_scale=replacement),)
    assert _native_target_binding_signature(session) != before
