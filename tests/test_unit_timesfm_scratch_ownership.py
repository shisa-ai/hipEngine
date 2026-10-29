"""TimesFM scratch must contain only live attention operands in either precision."""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

from hipengine.runtime import timesfm_decode as runtime


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
@pytest.mark.parametrize("shape", [(1, 3, 19), (2, 7, 29)])
def test_attention_scratch_has_no_score_plane_and_frees_each_owner(monkeypatch, precision, shape):
    batch, patches, cache_size = shape
    spec = SimpleNamespace(
        hidden_size=16, num_attention_heads=2, head_dim=8, num_hidden_layers=2,
        tokenizer_input_dims=8, qkv_size=48, quantile_output_dims=40,
    )
    allocated, freed = [], []

    def malloc(nbytes):
        buffer = SimpleNamespace(ptr=1000 + len(allocated), nbytes=nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(runtime, "malloc", malloc)
    monkeypatch.setattr(runtime, "free", lambda buffer: freed.append(buffer.ptr))
    decoder = object.__new__(runtime.TimesFMGPUDecoder)
    decoder.precision = precision
    decoder.spec = spec
    decoder._buffers = {}
    buffers = decoder._buffers_for(*shape)

    itemsize = 2 if precision == "fp16" else 4
    elements = batch * patches * (
        spec.tokenizer_input_dims + 10 * spec.hidden_size + spec.qkv_size
        + 2 * spec.quantile_output_dims + 2 * spec.num_attention_heads * spec.head_dim
    )
    cache_elements = 2 * spec.num_hidden_layers * batch * cache_size * spec.num_attention_heads * spec.head_dim
    parameter_bytes = 4 * batch * (patches + 2)
    assert sum(b.nbytes for b in allocated) == (elements + cache_elements) * itemsize + parameter_bytes
    assert "scores" not in {field.name for field in fields(buffers)}
    assert decoder._buffers_for(*shape) is buffers
    count = len(allocated)
    buffers.free()
    assert len(freed) == count
    assert set(freed) == {buffer.ptr for buffer in allocated}


@pytest.mark.parametrize("patches,horizon,expected", [
    (1, 1, 1), (3, 1, 3), (5, 1, 5),
    (1, 160, 4), (3, 160, 4), (5, 160, 5),
])
def test_decode_sizes_scratch_for_largest_segment(patches, horizon, expected):
    import numpy as np

    class AllocationObserved(Exception):
        pass

    decoder = object.__new__(runtime.TimesFMGPUDecoder)
    decoder.precision = "fp16"
    decoder.spec = SimpleNamespace(patch_length=32, horizon_length=128)
    observed = []

    def buffers_for(batch, rows, cache_size):
        observed.append((batch, rows, cache_size))
        raise AllocationObserved

    decoder._buffers_for = buffers_for
    inputs = np.zeros((2, 32 * patches), dtype=np.float32)
    with pytest.raises(AllocationObserved):
        decoder.decode(horizon, inputs, np.zeros_like(inputs, dtype=bool))
    assert observed == [(2, expected, patches + (4 if horizon == 160 else 0))]
