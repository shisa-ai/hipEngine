"""The dense Q8_0 T16 prefill's owned f16 operand is an abi, not a pointer table.

A weight owner that can spare the memory keeps the wide-wave prefill's half
operand pre-converted as an ``f16w`` sidecar of the weight it was built from.
These tests pin the two things that make that operand safe to read: the
dispatch step that selects the registered two-operand variant keys on the
weight's own capability and on the operand's exact shape, and the ``t16_f16w``
abi marshals the operand it was dispatched with. A weight without the sidecar
-- including one whose owner's reserve check failed, a supported outcome --
keeps the decoding launch. Byte equality of the two launches is a GPU contract
(``tests/test_gpu_gguf_q8_0_t16_wmma_prefill.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gl
from hipengine.kernels.registry import KernelKey, _KERNELS, is_registered
from hipengine.loading.qwen35_gguf_consumer_surface import linear_weight_pointers
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _q8_t16_f16w_prefill_dispatch,
    launch_gguf_linear,
)

_IN = 2816
_OUT = 4224
_RAW_PTR = 10
_TILES_PTR = 14
_F16W_PTR = 21
_ROWS = 1024

_DECODING_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_wmma_prefill_bf16_bf16_out"
)
_F16W_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_wmma_prefill_f16w_bf16_bf16_out"
)
_DECODING = GGUFLinearDispatch(_DECODING_KEY, "t16")


class _Weight:
    """A dual-resident Q8_0 weight, with an optional owned f16 operand."""

    def __init__(
        self,
        *,
        tiles: bool = True,
        f16w: tuple[int, tuple[int, ...]] | None = None,
        in_features: int = _IN,
        out_features: int = _OUT,
    ) -> None:
        self.spec = SimpleNamespace(
            layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0", slot_path="layers.0.ffn_down"
        )
        self._tiles = tiles
        self._f16w = f16w
        self._shape = (out_features, in_features)

    def has_allocation(self, name: str) -> bool:
        if name == "tiles":
            return self._tiles
        if name == "f16w":
            return self._f16w is not None
        return False

    def allocation(self, name: str | None = None):
        out_features, in_features = self._shape
        if name == "tiles" and self._tiles:
            return SimpleNamespace(
                tensor=SimpleNamespace(
                    ptr=_TILES_PTR,
                    shape=(out_features // 16, in_features // 32, 544),
                )
            )
        if name == "f16w" and self._f16w is not None:
            ptr, shape = self._f16w
            return SimpleNamespace(tensor=SimpleNamespace(ptr=ptr, shape=shape))
        if name in (None, "raw"):
            return SimpleNamespace(tensor=SimpleNamespace(ptr=_RAW_PTR))
        raise KeyError(name)


def _operand(shape: tuple[int, ...] = (_OUT, _IN), ptr: int = _F16W_PTR):
    return (ptr, shape)


def _rewrite(dispatch, weight, *, rows: int = _ROWS, in_features: int = _IN,
             out_features: int = _OUT):
    return _q8_t16_f16w_prefill_dispatch(
        dispatch,
        weight=weight,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )


def test_the_two_operand_variant_is_registered() -> None:
    assert is_registered(_F16W_KEY)


def test_the_sidecar_selects_the_registered_two_operand_variant() -> None:
    got = _rewrite(_DECODING, _Weight(f16w=_operand()))
    assert got.key == _F16W_KEY
    assert got.abi == "t16_f16w"


def test_a_weight_without_the_operand_keeps_the_decoding_launch() -> None:
    assert _rewrite(_DECODING, _Weight()) is _DECODING


def test_a_weight_that_cannot_express_sidecars_keeps_the_decoding_launch() -> None:
    """The reserve shortfall's own shape: no ``has_allocation`` at all."""

    assert _rewrite(_DECODING, SimpleNamespace()) is _DECODING


@pytest.mark.parametrize(
    "shape",
    [(_OUT, _IN + 16), (_OUT + 16, _IN), (_OUT, _IN, 1), ()],
)
def test_a_mismatched_operand_shape_is_a_decline(shape: tuple[int, ...]) -> None:
    assert _rewrite(_DECODING, _Weight(f16w=_operand(shape))) is _DECODING


def test_an_unregistered_variant_is_a_decline(monkeypatch) -> None:
    monkeypatch.setattr(gl, "_ensure_linear_kernel_registered", lambda key: None)
    monkeypatch.delitem(_KERNELS, _F16W_KEY)
    assert _rewrite(_DECODING, _Weight(f16w=_operand())) is _DECODING


_FOREIGN_QUANT_DECODING_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_wmma_prefill_bf16_bf16_out"
)
_FOREIGN_QUANT_F16W_KEY = KernelKey(
    "hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_wmma_prefill_f16w_bf16_bf16_out"
)


def test_a_foreign_quant_without_its_own_variant_stays_unchanged() -> None:
    """The step supplies an operand; it does not move the launch to a quant.

    No family registers the two-operand variant for this quant, so the
    capability is absent and the call keeps the variant it resolved. Nothing
    here branches on the quant key to decide that.
    """

    source = GGUFLinearDispatch(_FOREIGN_QUANT_DECODING_KEY, "t16")
    assert _rewrite(source, _Weight(f16w=_operand())) is source
    assert not is_registered(_FOREIGN_QUANT_F16W_KEY)


def test_a_foreign_quant_that_registers_the_variant_keeps_its_own_quant(
    monkeypatch,
) -> None:
    """With the capability present, the candidate carries the dispatch's quant."""

    source = GGUFLinearDispatch(_FOREIGN_QUANT_DECODING_KEY, "t16")
    monkeypatch.setitem(_KERNELS, _FOREIGN_QUANT_F16W_KEY, lambda *args, **kwargs: None)
    got = _rewrite(source, _Weight(f16w=_operand()))
    assert got.key == _FOREIGN_QUANT_F16W_KEY
    assert got.abi == "t16_f16w"


@pytest.mark.parametrize(
    "source",
    [
        GGUFLinearDispatch(
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_bf16_bf16_out"),
            "wmma_raw",
        ),
        GGUFLinearDispatch(
            KernelKey(
                "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"
            ),
            "t16",
        ),
        GGUFLinearDispatch(
            KernelKey(
                "hip_gfx1100", "linear", "gguf_q4_k_t16_v1",
                "t16_wmma_prefill_bf16_bf16_out",
            ),
            "t16",
        ),
    ],
)
def test_only_the_wide_wave_t16_variant_is_rewritten(source) -> None:
    assert _rewrite(source, _Weight(f16w=_operand())) is source


def test_the_abi_marshals_the_operand_it_declares() -> None:
    weight = _Weight(f16w=_operand())
    calls = []
    gl._LAUNCH_ABI["t16_f16w"](
        lambda *args, **kwargs: calls.append((args, kwargs)),
        weight, 201, 202, _ROWS, _IN, _OUT, {"stream": 7},
    )
    assert calls == [((201, _TILES_PTR, _F16W_PTR, 202, _ROWS, _IN, _OUT), {"stream": 7})]
    assert linear_weight_pointers("t16_f16w", weight) == (_TILES_PTR, _F16W_PTR)


def _launch_args(calls) -> list[tuple[str, tuple[int, ...]]]:
    def plain(value):
        value = getattr(value, "value", value)
        return 0 if value is None else value

    return [(symbol, tuple(plain(value) for value in args)) for symbol, args in calls]


class _Library:
    """Records the symbol and plain argument values of each launch."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, symbol: str):
        def fn(*args):
            self.calls.append((symbol, args))
            return 0

        fn.argtypes = None
        fn.restype = None
        return fn


@pytest.fixture
def wide_wave_env(monkeypatch):
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE", raising=False)


def test_the_f16w_wrapper_runs_the_two_wave_schedule_over_its_operand(wide_wave_env) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        gguf_q8_0_t16_wmma_prefill_f16w_bf16_bf16_out,
    )

    library = _Library()
    gguf_q8_0_t16_wmma_prefill_f16w_bf16_bf16_out(
        101, _TILES_PTR, _F16W_PTR, 202, _ROWS, _IN, _OUT,
        library=library, runtime=SimpleNamespace(check=pytest.fail),
    )
    assert _launch_args(library.calls) == [
        (
            "hipengine_gguf_q8_0_t16_wmma_prefill_2wave_f16w_bf16_bf16_out",
            (101, _TILES_PTR, _F16W_PTR, 202, _ROWS, _IN, _OUT, 0),
        )
    ]


def test_the_gfx1151_f16w_wrapper_runs_the_four_wave_schedule(wide_wave_env) -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        gguf_q8_0_t16_wmma_prefill_auto_4wave_f16w_bf16_bf16_out,
    )

    library = _Library()
    gguf_q8_0_t16_wmma_prefill_auto_4wave_f16w_bf16_bf16_out(
        101, _TILES_PTR, _F16W_PTR, 202, _ROWS, _IN, _OUT,
        library=library, runtime=SimpleNamespace(check=pytest.fail),
    )
    assert _launch_args(library.calls) == [
        (
            "hipengine_gguf_q8_0_t16_wmma_prefill_4wave_f16w_bf16_bf16_out",
            (101, _TILES_PTR, _F16W_PTR, 202, _ROWS, _IN, _OUT, 0),
        )
    ]


def test_the_f16w_wrapper_falls_back_to_the_decoding_launch_below_the_schedule(
    wide_wave_env,
) -> None:
    """No f16w kernel exists for the narrow tile; the operand goes unread."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        gguf_q8_0_t16_wmma_prefill_f16w_bf16_bf16_out,
    )

    library = _Library()
    gguf_q8_0_t16_wmma_prefill_f16w_bf16_bf16_out(
        101, _TILES_PTR, _F16W_PTR, 202, 17, 256, 2048,
        library=library, runtime=SimpleNamespace(check=pytest.fail),
    )
    assert _launch_args(library.calls) == [
        (
            "hipengine_gguf_q8_0_t16_wmma_prefill_bf16_bf16_out",
            (101, _TILES_PTR, 202, 17, 256, 2048, 32, 16, 0),
        )
    ]


def test_the_decoding_wrapper_is_unchanged_by_the_two_operand_form(wide_wave_env) -> None:
    """The auto wrapper still decodes; only dispatch can hand it an operand."""

    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
        gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out,
    )

    library = _Library()
    gguf_q8_0_t16_wmma_prefill_auto_2wave_bf16_bf16_out(
        101, _TILES_PTR, 202, _ROWS, _IN, _OUT,
        library=library, runtime=SimpleNamespace(check=pytest.fail),
    )
    assert _launch_args(library.calls) == [
        (
            "hipengine_gguf_q8_0_t16_wmma_prefill_2wave_bf16_bf16_out",
            (101, _TILES_PTR, 202, _ROWS, _IN, _OUT, 64, 32, 0),
        )
    ]


def _record_route(monkeypatch, *, with_sidecar: bool) -> list[tuple]:
    """Run one wide-wave prefill launch, recording the abi and its arguments."""

    launched: list[tuple] = []
    monkeypatch.setitem(_KERNELS, _DECODING_KEY, lambda *args, **kwargs: launched.append(args))
    monkeypatch.setitem(
        _KERNELS, _F16W_KEY, lambda *args, **kwargs: launched.append(args)
    )
    gl.clear_gguf_linear_dispatch_cache()
    try:
        weight = _Weight(f16w=_operand()) if with_sidecar else _Weight()
        launch_gguf_linear(
            weight, x_ptr=100, out_ptr=200, rows=_ROWS,
            in_features=_IN, out_features=_OUT,
            use_wmma_prefill=True, runtime="fake-runtime", stream=7,
        )
    finally:
        gl.clear_gguf_linear_dispatch_cache()
    return launched


def test_a_sidecar_launch_passes_the_operand_and_a_bare_weight_does_not(
    monkeypatch, wide_wave_env,
) -> None:
    assert _record_route(monkeypatch, with_sidecar=True) == [
        (100, _TILES_PTR, _F16W_PTR, 200, _ROWS, _IN, _OUT)
    ]
    assert _record_route(monkeypatch, with_sidecar=False) == [
        (100, _TILES_PTR, 200, _ROWS, _IN, _OUT)
    ]


def test_the_dispatch_memo_separates_weights_by_their_operand(monkeypatch, wide_wave_env) -> None:
    """A shared shape must not replay one weight's abi for another.

    The memo is process-global and keyed on the launch shape, so a second
    runner whose owner skipped the operand would otherwise replay the
    two-operand dispatch and read an allocation it does not have.
    """

    gl.clear_gguf_linear_dispatch_cache()
    calls = {"n": 0}
    original = gl.resolve_gguf_linear_dispatch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gl, "resolve_gguf_linear_dispatch", counting)
    seen: list[str] = []
    monkeypatch.setitem(_KERNELS, _DECODING_KEY, lambda *args, **kwargs: seen.append("decoding"))
    monkeypatch.setitem(_KERNELS, _F16W_KEY, lambda *args, **kwargs: seen.append("f16w"))
    with_operand = _Weight(f16w=_operand())
    without = _Weight()
    try:
        for weight in (with_operand, without, with_operand, without):
            launch_gguf_linear(
                weight, x_ptr=100, out_ptr=200, rows=_ROWS,
                in_features=_IN, out_features=_OUT,
                use_wmma_prefill=True, runtime="fake-runtime", stream=7,
            )
    finally:
        gl.clear_gguf_linear_dispatch_cache()
    assert seen == ["f16w", "decoding", "f16w", "decoding"]
    assert calls["n"] == 2
