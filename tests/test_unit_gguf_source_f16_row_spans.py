"""Request-local source-F16 ownership for packed GGUF prompt prefill.

One packed slab carries several requests. Each request's changed-arithmetic
admission is decided by its *own* complete length and its *own* physical rows,
never by the slab's aggregate rows: an aggregate 1541-row slab must not run the
1024-2047 pair-only interval for a 517-row request, and two 600-row requests
must not become one 1200-row request.

The launch contract under test:

* an eligible span runs the ordinary launch for its own rows and pointer
  offsets, with its own owner bound;
* a slab where nothing is eligible runs exactly today's launch;
* a mixed slab runs the unchanged whole-slab strict plan first, then the
  admitted spans, so an ineligible neighbour stays exact without disabling its
  eligible neighbours;
* spans never leak into the strict plan, a nested projection, or a pair-SiLU /
  concat owner that does not consume them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv  # noqa: F401
import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q8_0_T16,
)
from hipengine.runtime.gguf_linear import (
    SourceF16RowSpan,
    SourceF16SlabSpans,
    T16F16RocblasPrefillSession,
    current_source_f16_slab_spans,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_pair_silu,
    source_f16_slab_spans_session,
    t16_f16_rocblas_prefill_session,
)

_BACKEND = "hip_gfx1100"
_X_PTR = 0x10000000
_OUT_PTR = 0x20000000
_OUT_B_PTR = 0x21000000

_Q6_EXACT = KernelKey(
    _BACKEND, "linear", "gguf_q6_k_t16_qmicro_planar_v1",
    "t16_wmma_prefill_bf16_bf16_out",
)
_Q6_SOURCE = KernelKey(
    _BACKEND, "linear", "gguf_q6_k_t16_qmicro_planar_v1",
    "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
)
_Q5_EXACT = KernelKey(
    _BACKEND, "linear", "gguf_q5_k_t16_v1", "t16_wmma_prefill_bf16_bf16_out"
)
_Q5_SOURCE = KernelKey(
    _BACKEND, "linear", "gguf_q5_k_t16_v1", "f16_rocblas_t16_bf16_bf16_out"
)
_Q4_PAIR = KernelKey(
    _BACKEND, "linear", "gguf_q4_k_t16_v1",
    "f16_rocblas_t16_pair_bf16_bf16_out",
)
_Q5_SILU = KernelKey(
    _BACKEND, "linear_pair_silu", "gguf_q5_k_t16_v1",
    "dense_dual_wmma_prefill_bf16_bf16_out",
)


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {
        name: SimpleNamespace(
            tensor=SimpleNamespace(ptr=10 + index, shape=(1, 1, 544)),
            buffer=SimpleNamespace(nbytes=1),
        )
        for index, name in enumerate(("raw", "qweight", "scales", "mins", "tiles"))
    }

    class Weight:
        spec = SimpleNamespace(
            layout=layout,
            quant_key=quant_key,
            slot_path="layers.0.test",
            source=None,
        )

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def _owner(
    *,
    q6_tiles: dict[tuple[int, int, int], int] | None = None,
    q5_tiles: dict[tuple[int, int, int], int] | None = None,
    q4_tiles: dict[tuple[int, int, int], int] | None = None,
    q5_inplace: frozenset[tuple[int, int, int]] = frozenset(),
    pair_only: dict | None = None,
    max_rows: int = 4096,
) -> T16F16RocblasPrefillSession:
    return T16F16RocblasPrefillSession(
        min_rows=512,
        max_rows=max_rows,
        x_f16_ptr=0x30000000,
        x_f16_nbytes=1 << 40,
        weight_f16_ptr=0x40000000,
        weight_f16_nbytes=1 << 40,
        out_f16_ptr=0x50000000,
        out_f16_nbytes=1 << 40,
        tile_out_features_by_shape=q6_tiles or {},
        q5_tile_out_features_by_shape=q5_tiles or {},
        q4_tile_out_features_by_shape=q4_tiles or {},
        q5_x_inplace_shapes=q5_inplace,
        pair_only_second_operand_policies=pair_only or {},
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )


class _Calls:
    """Capture registered leaf launches, restoring the registry afterwards."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._originals: dict[KernelKey, object] = {}

    def install(self, label: str, key: KernelKey) -> None:
        if key not in self._originals:
            self._originals[key] = resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            )

        def fake(*args, **kwargs):
            self.calls.append((label, args, kwargs))

        register(key, fake, replace=True)

    def restore(self) -> None:
        for key, original in self._originals.items():
            register(key, original, replace=True)
        self._originals.clear()
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    def labels(self) -> list[str]:
        return [label for label, _args, _kwargs in self.calls]


@pytest.fixture()
def calls():
    capture = _Calls()
    try:
        yield capture
    finally:
        capture.restore()


def _spans(slab_rows: int, *rows: tuple[int, int, int, object]) -> SourceF16SlabSpans:
    cursor = 0
    spans = []
    for row_start, span_rows, request_rows, owner in rows:
        assert row_start == cursor
        spans.append(
            SourceF16RowSpan(
                row_start=row_start,
                rows=span_rows,
                request_rows=request_rows,
                owner=owner,
            )
        )
        cursor += span_rows
    assert cursor == slab_rows
    return SourceF16SlabSpans(slab_rows=slab_rows, spans=tuple(spans))


def _launch(weight, rows: int, *, out_features: int, out_ptr: int = _OUT_PTR):
    return launch_gguf_linear(
        weight,
        x_ptr=_X_PTR,
        out_ptr=out_ptr,
        rows=rows,
        in_features=5120,
        out_features=out_features,
        backend=_BACKEND,
        use_wmma_prefill=True,
        stream=7,
        runtime="runtime-sentinel",
    )


def test_eligible_spans_use_their_own_rows_offsets_and_tiles(calls) -> None:
    """A 517+1024 slab must not launch either request at 1541 rows."""

    calls.install("exact", _Q6_EXACT)
    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner_a = _owner(q6_tiles={(512, 5120, 10240): 2048})
    owner_b = _owner(q6_tiles={(1024, 5120, 10240): 512})
    spans = _spans(1541, (0, 517, 517, owner_a), (517, 1024, 1024, owner_b))

    with source_f16_slab_spans_session(spans):
        _launch(weight, 1541, out_features=10240)

    # Every request is eligible, so the strict slab plan is skipped entirely.
    assert calls.labels() == ["source", "source"]
    first_args, first_kwargs = calls.calls[0][1], calls.calls[0][2]
    second_args, second_kwargs = calls.calls[1][1], calls.calls[1][2]
    assert first_args[0] == _X_PTR
    assert first_args[2] == _OUT_PTR
    assert first_args[6] == 517
    assert first_kwargs["tile_out_features"] == 2048
    assert second_args[0] == _X_PTR + 517 * 5120 * 2
    assert second_args[2] == _OUT_PTR + 517 * 10240 * 2
    assert second_args[6] == 1024
    assert second_kwargs["tile_out_features"] == 512


def test_pair_only_interval_follows_complete_request_length(calls) -> None:
    """517 and 1024 keep their own pair-only interval inside one 1541 slab."""

    calls.install("exact", _Q6_EXACT)
    calls.install("source", _Q6_SOURCE)
    calls.install("pair", _Q4_PAIR)
    first = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    second = _fake_weight(
        layout=LAYOUT_GGUF_Q4_K_T16, quant_key="gguf_q4_k_t16_v1"
    )
    pair_only = {
        (
            "gguf_q6_k_t16_qmicro_planar_v1",
            5120,
            10240,
            "gguf_q4_k_t16_v1",
            6144,
        ): {
            (512, 1023): (2048, "f16_rocblas_t16_pair_bf16_bf16_out", False),
            (1024, 2047): (512, "f16_rocblas_t16_pair_bf16_bf16_out", False),
        }
    }
    owner_a = _owner(
        q6_tiles={(512, 5120, 10240): 2048},
        pair_only=pair_only,
    )
    owner_b = _owner(
        q6_tiles={(1024, 5120, 10240): 512},
        pair_only=pair_only,
    )
    spans = _spans(1541, (0, 517, 517, owner_a), (517, 1024, 1024, owner_b))

    with source_f16_slab_spans_session(spans):
        fused = launch_gguf_linear_pair(
            first,
            second,
            x_ptr=_X_PTR,
            out_a_ptr=_OUT_PTR,
            out_b_ptr=_OUT_B_PTR,
            rows=1541,
            in_features=5120,
            out_features=10240,
            out_features_b=6144,
            backend=_BACKEND,
            use_wmma_prefill=True,
            stream=7,
            runtime="runtime-sentinel",
        )

    assert fused is True
    assert calls.labels() == ["source", "pair", "source", "pair"]
    first_operand = calls.calls[0][1], calls.calls[0][2]
    second_operand = calls.calls[1][1], calls.calls[1][2]
    assert first_operand[0][0] == _X_PTR
    assert first_operand[0][6] == 517
    assert first_operand[1]["tile_out_features"] == 2048
    assert first_operand[1]["cast_activation"] is True
    assert second_operand[0][0] == _X_PTR
    assert second_operand[0][2] == _OUT_B_PTR
    assert second_operand[0][6] == 517
    assert second_operand[1]["tile_out_features"] == 2048
    assert second_operand[1]["cast_activation"] is False
    # 1024 rows keeps the 1024-2047 interval, which the 1541-row slab would
    # have applied to both requests.
    assert calls.calls[2][2]["tile_out_features"] == 512
    assert calls.calls[3][2]["tile_out_features"] == 512
    assert calls.calls[2][1][0] == _X_PTR + 517 * 5120 * 2
    assert calls.calls[2][1][6] == 1024


def test_two_six_hundred_row_requests_keep_the_512_interval(calls) -> None:
    """600+600 must not be read as one 1200-row request."""

    calls.install("exact", _Q6_EXACT)
    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(1200, (0, 600, 600, owner), (600, 600, 600, owner))

    with source_f16_slab_spans_session(spans):
        _launch(weight, 1200, out_features=10240)

    assert calls.labels() == ["source", "source"]
    assert [call[2]["tile_out_features"] for call in calls.calls] == [2048, 2048]
    assert [call[1][6] for call in calls.calls] == [600, 600]


def test_q5_spans_use_their_own_anchor_not_the_aggregate_anchor(calls) -> None:
    """The gfx1151 Q5 declaration: 600 rows anchors at 512, 1200 at 4096."""

    calls.install("exact", _Q5_EXACT)
    calls.install("source", _Q5_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1"
    )
    owner = _owner(
        q5_tiles={(512, 6144, 5120): 1280, (1024, 6144, 5120): 1280,
                  (4096, 6144, 5120): 1024},
        q5_inplace=frozenset({(512, 6144, 5120)}),
    )
    spans = _spans(1200, (0, 600, 600, owner), (600, 600, 600, owner))

    with source_f16_slab_spans_session(spans):
        launch_gguf_linear(
            weight,
            x_ptr=_X_PTR,
            out_ptr=_OUT_PTR,
            rows=1200,
            in_features=6144,
            out_features=5120,
            backend=_BACKEND,
            use_wmma_prefill=True,
        )

    assert calls.labels() == ["source", "source"]
    assert [call[2]["tile_out_features"] for call in calls.calls] == [1280, 1280]
    assert [call[1][6] for call in calls.calls] == [600, 600]


def test_mixed_slab_runs_the_strict_slab_plan_before_admitted_spans(calls) -> None:
    """An ineligible neighbour stays exact and does not disable the other."""

    calls.install("exact", _Q6_EXACT)
    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(1021, (0, 517, 517, owner), (517, 504, 504, None))

    with source_f16_slab_spans_session(spans):
        _launch(weight, 1021, out_features=10240)

    assert calls.labels() == ["exact", "source"]
    strict_args, strict_kwargs = calls.calls[0][1], calls.calls[0][2]
    # The exact T16 ABI is (x, tiles, out, rows, K, N).
    assert strict_args[3] == 1021
    assert "tile_out_features" not in strict_kwargs
    span_args = calls.calls[1][1]
    assert span_args[0] == _X_PTR
    assert span_args[6] == 517


def test_no_admitted_request_keeps_todays_slab_launch(calls) -> None:
    """Below-anchor or opted-out requests run exactly one aggregate launch."""

    calls.install("exact", _Q6_EXACT)
    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    spans = _spans(1024, (0, 512, 512, None), (512, 512, 512, None))

    with source_f16_slab_spans_session(spans):
        _launch(weight, 1024, out_features=10240)

    assert calls.labels() == ["exact"]
    assert calls.calls[0][1][3] == 1024


def test_single_span_matches_the_ordinary_owner_launch(calls) -> None:
    """A one-request slab is the ordinary launch, span plumbing included."""

    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})

    with t16_f16_rocblas_prefill_session(owner):
        _launch(weight, 517, out_features=10240)
    ordinary = calls.calls[:]
    calls.calls.clear()

    spans = _spans(517, (0, 517, 517, owner))
    with source_f16_slab_spans_session(spans):
        _launch(weight, 517, out_features=10240)

    assert calls.labels() == ["source"]
    assert calls.calls == ordinary


def test_shared_activation_pair_keeps_one_cast_per_span(calls) -> None:
    """Both admitted operands reuse the first operand's FP16 cast per span."""

    calls.install("source", _Q5_SOURCE)
    first = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1"
    )
    second = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1"
    )
    owner = _owner(
        q5_tiles={(512, 6144, 5120): 1280, (1024, 6144, 5120): 1280},
        q5_inplace=frozenset({(512, 6144, 5120), (1024, 6144, 5120)}),
    )
    spans = _spans(1200, (0, 600, 600, owner), (600, 600, 600, owner))

    with source_f16_slab_spans_session(spans):
        fused = launch_gguf_linear_pair(
            first,
            second,
            x_ptr=_X_PTR,
            out_a_ptr=_OUT_PTR,
            out_b_ptr=_OUT_B_PTR,
            rows=1200,
            in_features=6144,
            out_features=5120,
            backend=_BACKEND,
            use_wmma_prefill=True,
        )

    assert fused is True
    assert calls.labels() == ["source", "source", "source", "source"]
    assert [call[2]["cast_activation"] for call in calls.calls] == [
        True,
        False,
        True,
        False,
    ]
    assert [call[1][6] for call in calls.calls] == [600, 600, 600, 600]
    assert [call[1][0] for call in calls.calls] == [
        _X_PTR,
        _X_PTR,
        _X_PTR + 600 * 6144 * 2,
        _X_PTR + 600 * 6144 * 2,
    ]


def test_pair_silu_decline_and_post_op_ignore_spans(calls) -> None:
    """Pair-SiLU owns its own fallback and fused post-op, spans or not."""

    calls.install("silu", _Q5_SILU)
    first = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1"
    )
    second = _fake_weight(
        layout=LAYOUT_GGUF_Q5_K_T16, quant_key="gguf_q5_k_t16_v1"
    )
    owner = _owner(q5_tiles={(512, 6144, 5120): 1280})
    spans = _spans(517, (0, 517, 517, owner))

    def _silu():
        return launch_gguf_linear_pair_silu(
            first,
            second,
            x_ptr=_X_PTR,
            out_ptr=_OUT_PTR,
            rows=517,
            in_features=6144,
            out_features=5120,
            backend=_BACKEND,
        )

    unbound_fused = _silu()
    unbound_calls = calls.calls[:]
    calls.calls.clear()
    with source_f16_slab_spans_session(spans):
        bound_fused = _silu()
    assert bound_fused == unbound_fused is True
    assert calls.calls == unbound_calls

    calls.calls.clear()
    # A shape the fused owner does not cover still declines, spans or not.
    def _declining_silu():
        return launch_gguf_linear_pair_silu(
            first,
            second,
            x_ptr=_X_PTR,
            out_ptr=_OUT_PTR,
            rows=1,
            in_features=6144,
            out_features=5120,
            backend=_BACKEND,
            registered_decode_variant="dense_dual_q8_1x2_dp4a_bf16_bf16_out",
        )

    unbound_declined = _declining_silu()
    with source_f16_slab_spans_session(spans):
        bound_declined = _declining_silu()
    assert bound_declined == unbound_declined is False
    assert calls.calls == []


def test_concat_pair_plan_is_not_span_owned() -> None:
    """The concat owner never consumes request spans."""

    first = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    second = _fake_weight(
        layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"
    )
    kwargs = dict(
        rows=517,
        in_features=5120,
        out_features=10240,
        backend=_BACKEND,
        use_wmma_prefill=True,
    )
    unbound_plan = gguf_linear_module._resolve_gguf_linear_pair_concat_plan(
        first, second, **kwargs
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(517, (0, 517, 517, owner))
    with source_f16_slab_spans_session(spans):
        bound_plan = gguf_linear_module._resolve_gguf_linear_pair_concat_plan(
            first, second, **kwargs
        )
        fused = launch_gguf_linear_pair_concat(
            first,
            second,
            x_ptr=_X_PTR,
            out_ptr=_OUT_PTR,
            **kwargs,
        )
    assert bound_plan == unbound_plan
    assert fused is False


def test_span_scope_does_not_leak_into_the_launch_body() -> None:
    """Inside any launch the span map is cleared, including a mismatched call."""

    seen: list[object] = []
    key = _Q6_EXACT
    original = resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    )

    def fake(*args, **kwargs):
        seen.append(current_source_f16_slab_spans())

    register(key, fake, replace=True)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(517, (0, 517, 517, owner))
    try:
        with source_f16_slab_spans_session(spans):
            # A row count the span map does not describe keeps today's launch,
            # and the body must not see the outer span map.
            _launch(weight, 1024, out_features=10240)
            assert current_source_f16_slab_spans() is spans
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert seen == [None]


def test_span_scope_restores_after_a_failed_launch() -> None:
    key = _Q6_SOURCE
    original = resolve(
        backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant
    )

    def fake(*args, **kwargs):
        raise RuntimeError("leaf failed")

    register(key, fake, replace=True)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(517, (0, 517, 517, owner))
    outer = _spans(517, (0, 517, 517, owner))
    try:
        with source_f16_slab_spans_session(outer):
            with pytest.raises(RuntimeError, match="leaf failed"):
                with source_f16_slab_spans_session(spans):
                    _launch(weight, 517, out_features=10240)
            assert current_source_f16_slab_spans() is outer
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()


def test_span_launch_refuses_to_substitute_an_exact_projection(
    monkeypatch, calls
) -> None:
    """A span that cannot reach the changed arithmetic fails loudly."""

    calls.install("exact", _Q6_EXACT)
    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(517, (0, 517, 517, owner))

    def decline(dispatch, **kwargs):
        return dispatch

    monkeypatch.setattr(
        gguf_linear_module, "_q6_t16_f16_rocblas_prefill_dispatch", decline
    )

    with source_f16_slab_spans_session(spans):
        with pytest.raises(
            RuntimeError, match="did not select the request-local owner"
        ):
            _launch(weight, 517, out_features=10240)


def test_span_map_must_tile_the_slab() -> None:
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    with pytest.raises(ValueError, match="tile the slab"):
        SourceF16SlabSpans(
            slab_rows=1024,
            spans=(
                SourceF16RowSpan(0, 512, 512, owner),
                SourceF16RowSpan(600, 424, 424, owner),
            ),
        )
    with pytest.raises(ValueError, match="cover every slab row"):
        SourceF16SlabSpans(
            slab_rows=1024,
            spans=(SourceF16RowSpan(0, 512, 512, owner),),
        )
    with pytest.raises(ValueError, match="rows must be positive"):
        SourceF16RowSpan(0, 0, 512, owner)
    with pytest.raises(ValueError, match="must not be empty"):
        SourceF16SlabSpans(slab_rows=1, spans=())


def test_spans_cannot_nest_inside_a_request_local_owner(calls) -> None:
    calls.install("source", _Q6_SOURCE)
    weight = _fake_weight(
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    owner = _owner(q6_tiles={(512, 5120, 10240): 2048})
    spans = _spans(517, (0, 517, 517, owner))

    with t16_f16_rocblas_prefill_session(owner):
        with source_f16_slab_spans_session(spans):
            with pytest.raises(ValueError, match="cannot nest"):
                _launch(weight, 517, out_features=10240)
