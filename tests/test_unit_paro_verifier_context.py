"""Verifier attention selection must use live context, not allocation capacity."""
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime.qwen35_paro import _requires_full_attention_split_decode
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession


@pytest.mark.parametrize("position", [28, 116, 511, 1022, 1023, 1024, 1759])
def test_verifier_row_spans_select_attention_from_actual_context(monkeypatch, position):
    monkeypatch.setenv("HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1024")
    device = Device("hip", 0)
    session = SimpleNamespace(
        prefill_position_buf=DeviceBuffer(0x1000, 32),
        prefill_context_count_buf=DeviceBuffer(0x2000, 32),
        block_table=Tensor.from_handle(0x3000, (32,), DType.INT32, device),
        device=device, max_sequence_length=8192,
    )
    tensor, append, decode = Qwen35ParoResidentSession._verify_chain_row_spans(
        session, 2, position=position,
    )
    assert tensor.ptr == 0x1010
    assert append.live_counts.ptr == 0x1010
    assert decode.live_counts.ptr == 0x2010
    assert append.max_live_count == position
    assert decode.max_live_count == position + 1
    assert _requires_full_attention_split_decode(decode) == (position + 1 >= 1024)
