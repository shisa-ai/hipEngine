"""The compact verifier's span contract and launch wiring, without a GPU.

``dms_compact_int8_verify_chain_gate_bf16_spans`` is the compact store's
counterpart to the paged verify-chain leaf. It reads a ``per_head_variable``
INT8 span set through the store's own ``[rows, kv_heads]`` extent planes and
finishes through the FP32-in/BF16-out gate multiply.

Two things are worth pinning without a device. The first is the refusals: a
paged span set, a BF16 span set, or a declared extent that does not cover every
(row, kv head) must be named as such rather than read as if it fit. The second is
the wiring: which pointers reach the split-K producer, and that the gate multiply
covers exactly ``rows * q_heads * head_dim`` elements of the result plane.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.kernels.hip_gfx1100.attention import dms_compact_int8 as module
from hipengine.kernels.hip_gfx1100.attention.dms_compact_int8 import (
    dms_compact_int8_verify_chain_gate_bf16_spans,
    register_dms_compact_int8_kernels,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve

_ROWS, _Q_HEADS, _KV_HEADS, _DIM = 3, 8, 2, 64
_CONTIGUOUS = _Q_HEADS * _DIM


def _spans(
    *,
    mode: str = "per_head_variable",
    storage: DType = DType.INT8_PER_TOKEN_HEAD,
    shape: tuple[int, int, int] = (_ROWS, 1, _KV_HEADS),
    base_shape: tuple[int, int, int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        spans_mode=mode,
        storage_dtype=storage,
        base_offsets=SimpleNamespace(shape=base_shape or shape, ptr=0x1000),
        live_counts=SimpleNamespace(shape=shape, ptr=0x2000),
    )


def _call(**overrides) -> None:
    kwargs = {
        "spans": _spans(),
        "query_ptr": 0x11,
        "key_slot_ptr": 0x12,
        "value_slot_ptr": 0x13,
        "k_scale_ptr": 0x14,
        "v_scale_ptr": 0x15,
        "gate_ptr": 0x16,
        "out_ptr": 0x17,
        "result_ptr": 0x18,
        "partial_out_ptr": 0x19,
        "partial_m_ptr": 0x1A,
        "partial_l_ptr": 0x1B,
        "base_ptr": 0x1C,
        "live_ptr": 0x1D,
        "rows": _ROWS,
        "chunk_size": 256,
        "num_splits": 2,
        "num_q_heads": _Q_HEADS,
        "num_kv_heads": _KV_HEADS,
        "head_dim": _DIM,
        "query_row_stride": _CONTIGUOUS,
        "gate_row_stride": _CONTIGUOUS,
        "gate_head_stride": _DIM,
        "gate_dim_stride": 1,
        "out_row_stride": _CONTIGUOUS,
        "out_head_stride": _DIM,
        "out_dim_stride": 1,
        "scale": float(_DIM) ** -0.5,
    }
    kwargs.update(overrides)
    dms_compact_int8_verify_chain_gate_bf16_spans(**kwargs)


def test_verify_chain_leaf_registers_under_its_own_layer_key() -> None:
    clear_registry_for_tests()
    register_dms_compact_int8_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_compact_attn_decode",
            quant="int8_per_token_head",
            variant="verify_chain_gate_bf16_spans",
        )
        is dms_compact_int8_verify_chain_gate_bf16_spans
    )
    # The single-row AR leaf must keep its own key rather than be shadowed.
    assert resolve(
        backend="hip_gfx1100",
        layer="dms_compact_attn_decode",
        quant="int8_per_token_head",
        variant="grouped_gqa_splitk",
    ) is not dms_compact_int8_verify_chain_gate_bf16_spans


@pytest.mark.parametrize(
    "spans,match",
    [
        (_spans(mode="uniform"), "per_head_variable"),
        (_spans(storage=DType.BF16), "int8_per_token_head"),
        # A paged reading of the same rows: one live count per row, no head axis.
        (_spans(shape=(_ROWS, 1, 1)), r"cover every"),
        (_spans(shape=(2, 1, _KV_HEADS)), r"cover every"),
        # Both declare a legal (rows, *, kv_heads) extent but disagree on layers.
        (_spans(shape=(_ROWS, 1, _KV_HEADS), base_shape=(_ROWS, 2, _KV_HEADS)), "same extent"),
        (_spans(shape=(_ROWS, _KV_HEADS)), r"\[rows, layers, kv_heads\]"),
    ],
)
def test_verify_chain_leaf_refuses_a_span_set_it_cannot_read(spans, match) -> None:
    with pytest.raises(ValueError, match=match):
        _call(spans=spans)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rows": 1, "spans": _spans(shape=(1, 1, _KV_HEADS))}, "more than one row"),
        ({"chunk_size": 512}, "chunk_size"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"num_q_heads": 7}, "GQA"),
        ({"scale": 0.0}, "positive scale"),
        ({"num_splits": 0}, "positive dimensions"),
        ({"query_row_stride": _CONTIGUOUS - 1}, "query_row_stride"),
        ({"gate_head_stride": _CONTIGUOUS}, "gate_head_stride"),
        ({"gate_dim_stride": 2}, "gate_dim_stride"),
        ({"out_dim_stride": 4}, "out_dim_stride"),
        ({"base_ptr": 0}, "non-null"),
        ({"result_ptr": 0}, "non-null"),
    ],
)
def test_verify_chain_leaf_refuses_geometry_it_cannot_launch(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _call(**overrides)


def test_verify_chain_leaf_wires_the_store_planes_and_the_gate(monkeypatch) -> None:
    """The split-K producer gets the store's planes; the gate covers every row."""

    calls: list[tuple] = []
    monkeypatch.setattr(
        module,
        "dms_compact_attn_decode_splitk_int8",
        lambda *args, **kwargs: calls.append(("splitk", args, kwargs)),
    )
    gated: list[tuple] = []

    def _gate(attn_ptr, gate_ptr, out_ptr, total, **kwargs):
        gated.append((attn_ptr, gate_ptr, out_ptr, total))

    import hipengine.kernels.hip_gfx1100.attention.paged_attn_decode as paged

    monkeypatch.setattr(paged, "qwen35_full_attn_gate_mul_bf16", _gate)

    spans = _spans()
    _call(spans=spans)

    assert len(calls) == 1 and len(gated) == 1
    _, args, kwargs = calls[0]
    assert args[0] == 0x11, "query pointer"
    assert args[1] == 0x12 and args[2] == 0x13, "store slot planes"
    # The extent planes are the store's own, not the span set's declared tensors.
    assert args[3] == 0x1C and args[4] == 0x1D
    assert args[5] == 0x19 and args[6] == 0x1A and args[7] == 0x1B, "split partials"
    assert args[8] == 0x18, "fp32 result plane"
    assert args[9:13] == (_ROWS, _Q_HEADS, _KV_HEADS, _DIM)
    assert kwargs["k_scale_ptr"] == 0x14 and kwargs["v_scale_ptr"] == 0x15
    assert kwargs["library"] is None
    # The gate multiply reads the FP32 result and writes BF16 over every element.
    assert gated[0] == (0x18, 0x16, 0x17, _ROWS * _Q_HEADS * _DIM)


def test_verify_chain_leaf_rejects_a_nonfinite_scale_but_accepts_a_small_one() -> None:
    """A positive scale is a capability question, not a magnitude one."""

    for scale in (1e-8, 1.0, float(_DIM) ** -0.5):
        with pytest.raises(ValueError) as excinfo:
            _call(scale=scale, query_ptr=0)
        # Refused for the null pointer, not for the scale.
        assert "non-null" in str(excinfo.value)
    for scale in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="positive scale"):
            _call(scale=scale)


def test_verify_chain_leaf_stride_check_is_exact_not_a_minimum() -> None:
    """A wider row stride is still refused: neither consumer reads a stride."""

    for name, value in (
        ("query_row_stride", _CONTIGUOUS * 2),
        ("gate_row_stride", _CONTIGUOUS * 2),
        ("out_row_stride", _CONTIGUOUS + _DIM),
        ("gate_head_stride", _DIM * 2),
        ("out_head_stride", _DIM * 2),
    ):
        with pytest.raises(ValueError, match=name):
            _call(**{name: value})


def test_verify_chain_leaf_scale_metadata_is_not_required_to_launch() -> None:
    """The guard is the declared layout; scale planes arrive as pointers."""

    spans = _spans()
    assert not hasattr(spans, "scale_metadata")
    with pytest.raises(ValueError, match="non-null"):
        _call(spans=spans, out_ptr=0)
