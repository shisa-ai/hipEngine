"""VibeVoice encoder attention has one span entry point and no dense raw entry.

``vv_prefill_attention_kernel`` takes the ``KVLiveSpans`` metadata ABI:
``(base_offsets, live_counts, token_positions, evict_mask, row_positions)``.
Production reaches it through ``vv_attention_spans``, which requires complete
uniform BF16 block-size-one span metadata.

The dense ``vv_prefill_attention_f32`` wrapper had no caller and passed four
nulls for that metadata. It is removed; these checks pin the span surface so a
future edit cannot quietly reintroduce a dense entry or drop the span one.

CPU only: the span wrapper is driven through a recording stand-in for the built
library, so no device, hipcc, or checkpoint is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.core.dtype import DType
from hipengine.kernels.hip_gfx1100.vibevoice import encoder

_SOURCE = Path(encoder.__file__).with_name("encoder.hip")
_SPAN_SYMBOL = "hipengine_vv_attention_spans"
_RAW_SYMBOL = "hipengine_vv_prefill_attention_f32"
_KERNEL = "vv_prefill_attention_kernel"


def _source() -> str:
    return _SOURCE.read_text(encoding="utf-8")


def _extern_body(source: str, symbol: str) -> str:
    """Return one ``extern "C"`` definition body, failing if it is not unique."""
    marker = f'extern "C" int {symbol}('
    assert source.count(marker) == 1, f"{symbol} must be declared exactly once"
    start = source.index(marker)
    return source[start: source.index("\n}\n", start)]


class _FakeTensor:
    def __init__(self, dtype: DType, numel: int, ptr: int) -> None:
        self.dtype = dtype
        self.numel = numel
        self.ptr = ptr


class _FakeSpans:
    """The subset of ``KVLiveSpans`` that ``_span_arguments`` reads."""

    def __init__(self, rows: int = 4, capacity: int = 6) -> None:
        self.spans_mode = "uniform"
        self.storage_dtype = DType.BF16
        self.max_live_count = capacity
        self.base_offsets = _FakeTensor(DType.INT32, capacity, 0x1000)
        self.live_counts = _FakeTensor(DType.INT64, rows, 0x2000)
        self.token_positions = _FakeTensor(DType.INT64, capacity, 0x3000)
        self.evict_mask = _FakeTensor(DType.BOOL, capacity, 0x4000)
        self.row_positions = _FakeTensor(DType.INT64, rows, 0x5000)


class _RecordingLibrary:
    """Stands in for the built ``ctypes.CDLL``; records symbol use and calls."""

    def __init__(self) -> None:
        self.requested: list[str] = []
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.functions: dict[str, object] = {}

    def __getattr__(self, symbol: str):
        if symbol.startswith("_"):
            raise AttributeError(symbol)
        self.requested.append(symbol)

        def _fn(*args):
            self.calls.append((symbol, args))
            return 0

        self.functions[symbol] = _fn
        return _fn


class _RecordingRuntime:
    def __init__(self) -> None:
        self.checked: list[int] = []

    def check(self, status: int) -> None:
        self.checked.append(status)


def _call_span_entry(spans, library, runtime=None, **kwargs):
    return encoder.vv_attention_spans(
        0xA0, 0xB0, 0xC0, 0xD0, spans, 4, 2, 1, 8, 0.25,
        stream=0xE0, library=library, runtime=runtime or _RecordingRuntime(), **kwargs,
    )


def test_span_entry_survives_and_the_dense_raw_entry_is_gone() -> None:
    assert callable(encoder.vv_attention_spans)
    assert callable(encoder.vv_kv_write_spans)
    assert not hasattr(encoder, "vv_prefill_attention_f32")
    assert not hasattr(encoder, "_ARGTYPES_PREFILL_ATTN")


def test_span_wrapper_passes_complete_span_metadata_to_the_shared_kernel() -> None:
    library = _RecordingLibrary()
    runtime = _RecordingRuntime()
    spans = _FakeSpans(rows=4, capacity=6)

    _call_span_entry(spans, library, runtime)

    assert library.requested == [_SPAN_SYMBOL]
    assert _RAW_SYMBOL not in library.requested
    [(symbol, args)] = library.calls
    assert symbol == _SPAN_SYMBOL
    assert args == (
        0xA0, 0xB0, 0xC0, 0xD0,
        spans.base_offsets.ptr,
        spans.live_counts.ptr,
        spans.token_positions.ptr,
        spans.evict_mask.ptr,
        spans.row_positions.ptr,
        4, 2, 1, 8, 6, 0.25, 0xE0,
    )
    assert runtime.checked == []


def test_span_wrapper_pins_the_span_abi_argument_types() -> None:
    import ctypes

    library = _RecordingLibrary()
    _call_span_entry(_FakeSpans(), library)

    argtypes = library.functions[_SPAN_SYMBOL].argtypes
    pointers = (ctypes.c_void_p,) * 9
    ints = (ctypes.c_int64,) * 5
    assert tuple(argtypes) == pointers + ints + (ctypes.c_float, ctypes.c_void_p)


def test_span_wrapper_rejects_incomplete_span_metadata() -> None:
    library = _RecordingLibrary()
    spans = _FakeSpans()
    spans.row_positions = None

    with pytest.raises(ValueError, match="VibeVoice requires uniform BF16"):
        _call_span_entry(spans, library)
    assert library.calls == []


def test_source_declares_the_span_entry_and_no_dense_attention_entry() -> None:
    source = _source()

    assert _RAW_SYMBOL not in source
    assert source.count(f'extern "C" int {_SPAN_SYMBOL}(') == 1
    assert source.count(f"__global__ void {_KERNEL}(") == 1

    body = _extern_body(source, _SPAN_SYMBOL)
    assert f"hipLaunchKernelGGL({_KERNEL}" in body
    for parameter in (
        "base_offsets", "live_counts", "token_positions", "evict_mask", "row_positions",
    ):
        assert parameter in body, f"{_SPAN_SYMBOL} must forward {parameter}"

    kernel = source[source.index(f"__global__ void {_KERNEL}("):]
    kernel = kernel[: kernel.index("\n}\n")]
    for parameter in ("base_offsets", "live_counts", "token_positions", "evict_mask"):
        assert parameter in kernel, f"{_KERNEL} must read {parameter}"


def test_build_plan_still_compiles_only_the_encoder_translation_unit() -> None:
    plan = encoder.plan_vibevoice_encoder_build()
    assert [Path(path).name for path in plan.sources] == ["encoder.hip"]
