from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
from hipengine.kernels.registry import KernelKey

BACKEND = "hip_gfx1100"
Q8_1_VARIANT = "dense_dual_q8_1x2_dp4a_bf16_bf16_out"
QMICRO_ROWTILE = "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out"

Q4_T16_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_q4_k_t16_v1",
    "dense_single_local32_bf16_bf16_out",
)
Q4_QMICRO_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_q4_k_qmicro_t16_v1",
    "dense_single_local32_bf16_bf16_out",
)
Q5_T16_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_q5_k_t16_v1",
    "t16_gemv_decode_bf16_bf16_out",
)
Q5_TILE8_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_q5_k_t16_v1",
    "t16_gemv_decode_tile8_bf16_bf16_out",
)
IQ4_XS_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_iq4_xs",
    "local32_gemv_bf16_bf16_out",
)
IQ4_NL_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_iq4_nl",
    "local32_gemv_bf16_bf16_out",
)
Q4_PACK8_DECODE = KernelKey(
    BACKEND,
    "linear",
    "gguf_q4_k",
    "pack8_bf16_bf16_out",
)


def _dispatch(key: KernelKey, abi: str = "t16") -> SimpleNamespace:
    return SimpleNamespace(key=key, abi=abi)


class _Weight:
    def __init__(
        self,
        label: str,
        quant_key: str,
        *,
        tiles: bool = True,
        decode_tiles: bool = False,
        decode_tiles_dual: bool = False,
    ) -> None:
        base = 10 if label == "a" else 110
        allocations = {
            "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=base)),
            "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=base + 1)),
            "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=base + 2)),
            "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=base + 3)),
        }
        if tiles:
            allocations["tiles"] = SimpleNamespace(
                tensor=SimpleNamespace(ptr=base + 4)
            )
        if decode_tiles:
            allocations["decode_tiles"] = SimpleNamespace(
                tensor=SimpleNamespace(ptr=base + 5)
            )
        if decode_tiles_dual:
            allocations["decode_tiles_dual"] = SimpleNamespace(
                tensor=SimpleNamespace(ptr=base + 6)
            )
        self.label = label
        self.backend = BACKEND
        self.spec = SimpleNamespace(
            layout="raw",
            quant_key=quant_key,
            slot_path=f"layers.0.{label}",
            source=None,
        )
        self.allocations = allocations

    def allocation(self, name: str = "raw"):
        return self.allocations[name]


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dispatch_a: SimpleNamespace,
    dispatch_b: SimpleNamespace,
    registered=True,
    **overrides,
) -> None:
    def _dispatch_for(weight, **kwargs):
        return dispatch_a if weight.label == "a" else dispatch_b

    values = {
        "resolve_gguf_linear_dispatch": _dispatch_for,
        "_resolve_use_gemv_decode": lambda kwarg: (
            True if kwarg is None else bool(kwarg)
        ),
        "_resolve_use_wmma_prefill": lambda kwarg: True,
        "_resolve_use_q4k_rowtile": lambda kwarg: True,
        "_ensure_linear_kernel_registered": lambda key: None,
        "is_registered": lambda key: registered,
        "_target_verifier_production_q4_pair_key": lambda *args, **kwargs: None,
        "_target_verifier_production_q4_rowtile_scope_enabled": (
            lambda *args, **kwargs: False
        ),
        "_iq_dense_prefill_dispatch": lambda dispatch, **kwargs: dispatch,
        "_iq_dense_decode_dispatch": lambda dispatch, **kwargs: dispatch,
        "_t16_c1_variant_dispatch": lambda dispatch, **kwargs: dispatch,
        "_q5_t16_dense_pair_silu_variant": lambda rows: None,
        "_iq4_xs_admitted_prefill_variant": lambda backend: "admitted",
        "_pack8_dual_wmma_silu_dispatch": lambda *args, **kwargs: None,
        "_q4_t16_dual_wmma_silu_dispatch": lambda *args, **kwargs: None,
        "_q4_t16_dual_rowtile_silu_dispatch": lambda *args, **kwargs: None,
        "_pack8_dual_rowtile_silu_dispatch": lambda *args, **kwargs: None,
        "_native_batch_decode_session_enabled": False,
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setattr(gguf_linear_module, name, value)


def _resolve(weight_a, weight_b, *, rows, **kwargs):
    return gguf_linear_module._resolve_gguf_linear_pair_silu_plan(
        weight_a,
        weight_b,
        rows=rows,
        in_features=kwargs.pop("in_features", 5120),
        out_features=kwargs.pop("out_features", 17408),
        backend=BACKEND,
        **kwargs,
    )


def _execute(
    plan,
    weight_a,
    weight_b,
    *,
    events,
    monkeypatch: pytest.MonkeyPatch,
    x_ptr=100,
    out_ptr=200,
    **kwargs,
):
    def _resolve_key(**key):
        def _fn(*args, **fn_kwargs):
            events.append(
                (
                    "call",
                    (key["backend"], key["layer"], key["quant"], key["variant"]),
                    args,
                    fn_kwargs,
                )
            )

        return _fn

    monkeypatch.setattr(gguf_linear_module, "resolve", _resolve_key)
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **fn_kwargs: events.append(("quantize", args, fn_kwargs)),
    )
    stream = kwargs.pop("stream", 7)
    runtime = kwargs.pop("runtime", "runtime-sentinel")
    return gguf_linear_module._execute_gguf_linear_pair_silu_plan(
        plan,
        weight_a,
        weight_b,
        x_ptr,
        out_ptr,
        stream=stream,
        runtime=runtime,
        **kwargs,
    )


def test_pair_silu_plan_q8x2_workspace_boundary(monkeypatch):
    weight_a = _Weight("a", "gguf_q4_k_t16_v1")
    weight_b = _Weight("b", "gguf_q4_k_t16_v1")
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_T16_DECODE),
        dispatch_b=_dispatch(Q4_T16_DECODE),
    )

    missing = _resolve(
        weight_a,
        weight_b,
        rows=1,
        registered_decode_variant=Q8_1_VARIANT,
    )
    assert missing.operation is None
    assert missing.fallback_reason == "q8_1_workspace_missing"

    plan = _resolve(
        weight_a,
        weight_b,
        rows=1,
        registered_decode_variant=Q8_1_VARIANT,
        q8_1_workspace_ptr=900,
    )
    assert plan.operation == "q4_t16_q8_1x2"
    assert plan.operand_mode == "tiles_tiles"
    assert plan.q8_1_quantize is True
    assert plan.pair_key.variant == Q8_1_VARIANT

    events = []
    assert _execute(plan, weight_a, weight_b, events=events, monkeypatch=monkeypatch,
                    q8_1_workspace_ptr=900)
    assert events[0] == (
        "quantize",
        (100, 900, 1, 5120),
        {"stream": 7, "library": None, "runtime": "runtime-sentinel"},
    )
    assert events[1][0] == "call"
    assert events[1][2] == (900, 14, 114, 200, 1, 5120, 17408)
    assert events[1][3] == {"stream": 7, "runtime": "runtime-sentinel"}


def test_pair_silu_plan_local32_keeps_int_activation_pointer(monkeypatch):
    weight_a = _Weight("a", "gguf_q4_k_t16_v1")
    weight_b = _Weight("b", "gguf_q4_k_t16_v1")
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_T16_DECODE),
        dispatch_b=_dispatch(Q4_T16_DECODE),
    )

    plan = _resolve(weight_a, weight_b, rows=1)
    assert plan.operation == "q4_t16_pair"
    assert plan.q8_1_quantize is False

    events = []
    assert _execute(plan, weight_a, weight_b, events=events, monkeypatch=monkeypatch,
                    x_ptr=100.9)
    assert events[0][2][0] == 100
    assert events[0][2][1:3] == (14, 114)


def test_pair_silu_plan_expanded_metadata_boundary(monkeypatch):
    weight_a = _Weight("a", "gguf_q4_k_qmicro_t16_v1")
    weight_b = _Weight("b", "gguf_q4_k_qmicro_t16_v1")
    metadata_nbytes = (17408 // 16) * (5120 // 256) * 256
    pair_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k_qmicro_t16_v1",
        "dense_dual_wmma_bf16_bf16_out",
    )
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_QMICRO_DECODE),
        dispatch_b=_dispatch(Q4_QMICRO_DECODE),
        _q4_t16_dual_wmma_silu_dispatch=lambda *args, **kwargs: pair_key,
    )

    below = _resolve(
        weight_a,
        weight_b,
        rows=4095,
        pair_workspace_ptr=5000,
        pair_workspace_nbytes=2 * metadata_nbytes,
    )
    assert below.operation == "registered_pair"
    assert below.expanded_metadata is False

    at = _resolve(
        weight_a,
        weight_b,
        rows=4096,
        pair_workspace_ptr=5000,
        pair_workspace_nbytes=2 * metadata_nbytes,
    )
    assert at.operation == "t16_wmma_expanded"
    assert at.expanded_metadata is True
    assert at.metadata_nbytes == metadata_nbytes

    events = []
    assert _execute(
        at,
        weight_a,
        weight_b,
        events=events,
        monkeypatch=monkeypatch,
        pair_workspace_ptr=5000,
        pair_workspace_nbytes=2 * metadata_nbytes,
    )
    assert events[0][2] == (
        100,
        14,
        114,
        5000,
        5000 + metadata_nbytes,
        200,
        4096,
        5120,
        17408,
    )


@pytest.mark.parametrize(
    ("rows", "chunks"),
    (
        (8, ((8, 0),)),
        (9, ((7, 0), (2, 7))),
        (70, ((8, 0), (8, 8), (8, 16), (8, 24), (8, 32),
              (8, 40), (8, 48), (8, 56), (6, 64))),
    ),
)
def test_pair_silu_plan_qmicro_chunk_boundaries(rows, chunks, monkeypatch):
    weight_a = _Weight("a", "gguf_q4_k_qmicro_t16_v1")
    weight_b = _Weight("b", "gguf_q4_k_qmicro_t16_v1")
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_QMICRO_DECODE),
        dispatch_b=_dispatch(Q4_QMICRO_DECODE),
        _native_batch_decode_session_enabled=True,
    )

    plan = _resolve(
        weight_a,
        weight_b,
        rows=rows,
        registered_decode_variant=QMICRO_ROWTILE,
        q8_1_workspace_ptr=900,
    )
    assert plan.operation == "qmicro_q8x2_rowbatch"
    assert plan.chunks == chunks


def test_pair_silu_plan_prefill_chunk_boundary(monkeypatch):
    weight_a = _Weight("a", "gguf_q4_k_t16_v1")
    weight_b = _Weight("b", "gguf_q4_k_t16_v1")
    pack8_rowtile = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_T16_DECODE),
        dispatch_b=_dispatch(Q4_T16_DECODE),
        _target_verifier_production_q4_rowtile_scope_enabled=lambda *a, **k: True,
        _pack8_dual_rowtile_silu_dispatch=lambda *a, **k: pack8_rowtile,
    )

    below = _resolve(weight_a, weight_b, rows=8)
    assert below.operation == "registered_pair"
    assert below.pair_key == pack8_rowtile

    at = _resolve(weight_a, weight_b, rows=9)
    assert at.operation == "production_q4_chunk_groups"
    assert at.chunks == ((7, 0), (2, 7))


def test_pair_silu_plan_sidecar_fallthrough_uses_exact_resource(monkeypatch):
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_PACK8_DECODE, "raw"),
        dispatch_b=_dispatch(Q4_PACK8_DECODE, "raw"),
    )

    base = _resolve(
        _Weight("a", "gguf_q4_k"),
        _Weight("b", "gguf_q4_k"),
        rows=1,
    )
    assert base.operation == "registered_pair"
    assert base.operand_mode == "pack8_pack8"
    assert base.pair_key.variant == "pack8_dual_decode_bf16_bf16_out"

    dual = _resolve(
        _Weight("a", "gguf_q4_k", decode_tiles_dual=True),
        _Weight("b", "gguf_q4_k", decode_tiles_dual=True),
        rows=1,
    )
    assert dual.operand_mode == "decode_tiles_dual"
    assert dual.pair_key.variant == "t16_dual_interleaved_sidecar_decode_bf16_bf16_out"

    sidecar = _resolve(
        _Weight("a", "gguf_q4_k", decode_tiles=True),
        _Weight("b", "gguf_q4_k", decode_tiles=True),
        rows=1,
    )
    assert sidecar.operand_mode == "decode_tiles_pair"
    assert sidecar.pair_key.variant == "t16_sidecar_dual_decode_bf16_bf16_out"


@pytest.mark.parametrize(
    ("dispatch_a", "dispatch_b", "operand_mode"),
    (
        (IQ4_XS_DECODE, Q4_T16_DECODE, "raw_tiles"),
        (Q4_T16_DECODE, IQ4_XS_DECODE, "tiles_raw"),
        (IQ4_XS_DECODE, Q5_TILE8_DECODE, "raw_tiles"),
        (Q4_T16_DECODE, Q5_TILE8_DECODE, "tiles_tiles"),
        (Q5_TILE8_DECODE, Q4_T16_DECODE, "tiles_tiles"),
        (IQ4_NL_DECODE, Q5_TILE8_DECODE, "raw_tiles"),
        (IQ4_NL_DECODE, IQ4_NL_DECODE, "raw_raw"),
    ),
)
def test_pair_silu_plan_mixed_operand_modes(
    dispatch_a, dispatch_b, operand_mode, monkeypatch
):
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(dispatch_a, "raw"),
        dispatch_b=_dispatch(dispatch_b, "raw"),
    )
    plan = _resolve(
        _Weight("a", dispatch_a.quant),
        _Weight("b", dispatch_b.quant),
        rows=1,
    )
    assert plan.operation == "registered_pair"
    assert plan.operand_mode == operand_mode


@pytest.mark.parametrize(
    ("dispatch_a", "dispatch_b", "weight_kwargs", "expected_args"),
    (
        (
            Q4_T16_DECODE,
            IQ4_XS_DECODE,
            {},
            (100, 14, 110, 200, 1, 5120, 17408),
        ),
        (
            IQ4_XS_DECODE,
            Q4_T16_DECODE,
            {},
            (100, 10, 114, 200, 1, 5120, 17408),
        ),
        (
            Q4_PACK8_DECODE,
            Q4_PACK8_DECODE,
            {},
            (100, 11, 12, 13, 111, 112, 113, 200, 1, 5120, 17408),
        ),
        (
            Q4_PACK8_DECODE,
            Q4_PACK8_DECODE,
            {"decode_tiles_dual": True},
            (100, 16, 200, 1, 5120, 17408),
        ),
        (
            Q4_PACK8_DECODE,
            Q4_PACK8_DECODE,
            {"decode_tiles": True},
            (100, 15, 115, 200, 1, 5120, 17408),
        ),
    ),
)
def test_pair_silu_plan_executes_exact_operands(
    dispatch_a, dispatch_b, weight_kwargs, expected_args, monkeypatch
):
    weight_a = _Weight("a", dispatch_a.quant, **weight_kwargs)
    weight_b = _Weight("b", dispatch_b.quant, **weight_kwargs)
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(dispatch_a),
        dispatch_b=_dispatch(dispatch_b),
    )
    plan = _resolve(weight_a, weight_b, rows=1)
    events = []
    assert _execute(plan, weight_a, weight_b, events=events, monkeypatch=monkeypatch)
    assert events[0][2] == expected_args


def test_pair_silu_plan_declines_unrelated_decode(monkeypatch):
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_PACK8_DECODE, "raw"),
        dispatch_b=_dispatch(Q4_PACK8_DECODE, "raw"),
        registered=False,
    )
    plan = _resolve(
        _Weight("a", "gguf_q4_k"),
        _Weight("b", "gguf_q4_k"),
        rows=1,
    )
    assert plan.operation is None
    assert plan.fallback_reason == "unsupported_decode_pair"

    events = []
    assert not _execute(
        plan,
        _Weight("a", "gguf_q4_k"),
        _Weight("b", "gguf_q4_k"),
        events=events,
        monkeypatch=monkeypatch,
    )
    assert events == []


def test_public_pair_silu_q8x2_characterization(monkeypatch):
    weight_a = _Weight("a", "gguf_q4_k_t16_v1")
    weight_b = _Weight("b", "gguf_q4_k_t16_v1")
    _patch(
        monkeypatch,
        dispatch_a=_dispatch(Q4_T16_DECODE),
        dispatch_b=_dispatch(Q4_T16_DECODE),
    )
    events = []

    def _resolve_key(**key):
        def _fn(*args, **kwargs):
            events.append(
                (
                    "call",
                    (key["backend"], key["layer"], key["quant"], key["variant"]),
                    args,
                    kwargs,
                )
            )

        return _fn

    monkeypatch.setattr(gguf_linear_module, "resolve", _resolve_key)
    monkeypatch.setattr(
        gguf_linear_module,
        "gguf_q4_k_quantize_bf16_q8_1x2",
        lambda *args, **kwargs: events.append(("quantize", args, kwargs)),
    )
    assert gguf_linear_module.launch_gguf_linear_pair_silu(
        weight_a,
        weight_b,
        x_ptr=100,
        out_ptr=200,
        rows=1,
        in_features=5120,
        out_features=17408,
        backend=BACKEND,
        stream=7,
        runtime="runtime-sentinel",
        use_gemv_decode=True,
        registered_decode_variant=Q8_1_VARIANT,
        q8_1_workspace_ptr=900,
    )
    assert [event[0] for event in events] == ["quantize", "call"]
    assert events[0][1] == (100, 900, 1, 5120)
    assert events[1][2] == (900, 14, 114, 200, 1, 5120, 17408)
    assert events[1][3] == {"stream": 7, "runtime": "runtime-sentinel"}
