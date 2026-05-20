from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# Import built-ins so the registry has real kernels to restore after overrides.
import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
)
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    GGUF_OUTPUT_FP16,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_triple,
    resolve_gguf_linear_dispatch,
    set_wmma_prefill_enabled,
    wmma_prefill_session,
)
from hipengine.runtime.prefill import PrefillConfig


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
    }

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def test_resolve_gguf_linear_dispatch_uses_weight_quant_for_raw_layouts() -> None:
    q4 = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    q5 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k")
    q6 = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k")
    q41 = _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1")

    assert resolve_gguf_linear_dispatch(q4).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q5).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q5_k", "gemv_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q6, output_dtype=GGUF_OUTPUT_F32).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q6_k", "gemv_bf16_f32_out"
    )
    assert resolve_gguf_linear_dispatch(q4, rows=4).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q5, rows=4, output_dtype=GGUF_OUTPUT_FP16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_fp16_out"
    )
    assert resolve_gguf_linear_dispatch(q41, rows=4).key == KernelKey(
        "hip_gfx1100", "dense_gemv", "bf16", "prefill_out"
    )
    q8_t16 = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    assert resolve_gguf_linear_dispatch(q8_t16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"
    )
    assert resolve_gguf_linear_dispatch(q8_t16, output_dtype=GGUF_OUTPUT_FP16).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_fp16_fp16_out"
    )
    assert resolve_gguf_linear_dispatch(q8_t16, activation_dtype=GGUF_ACTIVATION_F32).key == KernelKey(
        "hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_f32_bf16_out"
    )


@pytest.mark.parametrize(
    ("weight", "output_dtype", "key", "expected_args"),
    [
        (
            _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"),
            (100, 11, 12, 13, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q5_k"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_bf16_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k"),
            GGUF_OUTPUT_F32,
            KernelKey("hip_gfx1100", "linear", "gguf_q6_k", "prefill_bf16_f32_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_DENSE_BF16, quant_key="gguf_q4_1"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "dense_gemv", "bf16", "prefill_out"),
            (100, 10, 200, 2, 1024, 2048),
        ),
        (
            _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1"),
            GGUF_OUTPUT_BF16,
            KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
            (100, 14, 200, 2, 1024, 2048),
        ),
    ],
)
def test_launch_gguf_linear_calls_registry_kernel_with_expected_abi(
    weight, output_dtype: str, key: KernelKey, expected_args: tuple[int, ...]
) -> None:
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    register(key, fake_kernel, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=2,
            in_features=1024,
            out_features=2048,
            output_dtype=output_dtype,
            threads=128,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        register(key, original, replace=True)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == expected_args
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel", "threads": 128}


def test_gguf_linear_dispatch_rejects_unsupported_dtype() -> None:
    weight = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    with pytest.raises(ValueError, match="unsupported GGUF linear dispatch"):
        resolve_gguf_linear_dispatch(weight, output_dtype="int8")


# ---------------------------------------------------------------------------
# P8: WMMA batched prefill opt-in dispatch (docs/GGUF.md "P8: real batched
# prefill GEMM").
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_wmma_prefill_state(monkeypatch):
    """Clear the env var + session toggle before/after every test in this file.

    Without this, a test that flips ``HIPENGINE_GGUF_WMMA_PREFILL`` or calls
    ``set_wmma_prefill_enabled`` would silently leak into the next test
    case (and the next test module, since pytest runs the file in-process).
    """

    monkeypatch.delenv("HIPENGINE_GGUF_WMMA_PREFILL", raising=False)
    set_wmma_prefill_enabled(None)
    yield
    set_wmma_prefill_enabled(None)


_WMMA_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_bf16_bf16_out")
_PREFILL_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out")
_DECODE_PACK8_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
_Q4_WMMA_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "wmma_prefill_bf16_bf16_out")
_Q4_PREFILL_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_bf16_out")
_Q4_GEMV_BF16 = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "gemv_bf16_bf16_out")
_Q4_PACK8_PREFILL_BF16 = KernelKey(
    "hip_gfx1100", "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out"
)


def _capture_launch(
    *,
    rows: int,
    in_features: int = 1024,
    out_features: int = 2048,
    use_wmma_prefill: bool | None = None,
    quant_key: str = "gguf_q8_0",
    layout: str = LAYOUT_RAW_GGUF,
    output_dtype: str = GGUF_OUTPUT_BF16,
    threads: int = 0,
    extra_keys: tuple[KernelKey, ...] = (),
) -> tuple[KernelKey, tuple, dict]:
    """Drive ``launch_gguf_linear`` against a fake kernel + capture the call.

    Returns ``(key, args, kwargs)`` for the kernel that fired.
    """

    weight = _fake_weight(layout=layout, quant_key=quant_key)
    captured: dict[str, object] = {"key": None, "args": None, "kwargs": None}
    # Pre-resolve which key the dispatch should pick so we can register a
    # fake kernel under exactly that key (and the alternates we care about,
    # so the dispatch doesn't fall through to the real .so kernel).
    keys = (
        _WMMA_BF16,
        _PREFILL_BF16,
        _DECODE_PACK8_BF16,
        _Q4_WMMA_BF16,
        _Q4_PREFILL_BF16,
        _Q4_GEMV_BF16,
        _Q4_PACK8_PREFILL_BF16,
    ) + extra_keys
    originals = {k: resolve(backend=k.backend, layer=k.layer, quant=k.quant, variant=k.variant) for k in keys}

    def make_fake(key: KernelKey):
        def fake(*args, **kwargs):
            captured["key"] = key
            captured["args"] = args
            captured["kwargs"] = kwargs

        return fake

    try:
        for k in keys:
            register(k, make_fake(k), replace=True)
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_dtype=output_dtype,
            threads=threads,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=use_wmma_prefill,
        )
    finally:
        for k, fn in originals.items():
            register(k, fn, replace=True)

    return captured["key"], captured["args"], captured["kwargs"]  # type: ignore[return-value]


def test_prefill_config_exposes_wmma_prefill_field() -> None:
    """PrefillConfig.use_wmma_prefill is a real field with safe default."""

    cfg = PrefillConfig()
    assert cfg.use_wmma_prefill is False  # safe default: opt-in only
    on = PrefillConfig(use_wmma_prefill=True)
    assert on.use_wmma_prefill is True
    # Coercion: non-bool truthy values become True
    coerced = PrefillConfig(use_wmma_prefill=1)  # type: ignore[arg-type]
    assert coerced.use_wmma_prefill is True


def test_wmma_prefill_off_by_default_for_q8_0_rows_gt_1() -> None:
    """Without any opt-in, rows>1 Q8_0 still goes through the decode-shaped alias."""

    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_BF16  # decode-shaped prefill alias, NOT WMMA


def test_wmma_prefill_kwarg_opts_in_q8_0_rows_gt_1() -> None:
    """Passing ``use_wmma_prefill=True`` rewrites to the WMMA family for Q8_0 rows>1."""

    key, args, kwargs = _capture_launch(rows=4, use_wmma_prefill=True)
    assert key == _WMMA_BF16
    # WMMA ABI is the raw-pointer signature: (x, qweight, out, rows, in_f, out_f)
    assert args == (100, 10, 200, 4, 1024, 2048)
    # threads should NOT be present on the WMMA path (it takes tile_m / tile_n)
    assert "threads" not in kwargs
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_wmma_prefill_kwarg_can_force_off_even_with_session_on() -> None:
    """Per-call ``use_wmma_prefill=False`` wins over an enabled session."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=4, use_wmma_prefill=False)
    assert key == _PREFILL_BF16


def test_wmma_prefill_env_var_opts_in(monkeypatch) -> None:
    """``HIPENGINE_GGUF_WMMA_PREFILL=1`` enables the rewrite without any kwarg."""

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "1")
    key, _, _ = _capture_launch(rows=4)
    assert key == _WMMA_BF16


def test_wmma_prefill_env_var_accepts_common_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "On"):
        monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", value)
        key, _, _ = _capture_launch(rows=4)
        assert key == _WMMA_BF16, f"env value {value!r} should enable WMMA"


def test_wmma_prefill_env_var_falsy_values_keep_decode_path(monkeypatch) -> None:
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", value)
        key, _, _ = _capture_launch(rows=4)
        assert key == _PREFILL_BF16, f"env value {value!r} should keep decode path"


def test_wmma_prefill_session_toggle_persists_until_cleared() -> None:
    """``set_wmma_prefill_enabled(True)`` enables until cleared with ``None``."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=4)
    assert key == _WMMA_BF16
    set_wmma_prefill_enabled(False)
    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_BF16
    set_wmma_prefill_enabled(None)
    # back to env default (unset in this fixture) -> decode path
    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_BF16


def test_wmma_prefill_session_context_manager_restores_previous_state() -> None:
    set_wmma_prefill_enabled(False)  # baseline: explicit off
    with wmma_prefill_session(True):
        key, _, _ = _capture_launch(rows=4)
        assert key == _WMMA_BF16
    # Restored to the previous explicit-off session state
    key, _, _ = _capture_launch(rows=4)
    assert key == _PREFILL_BF16


def test_wmma_prefill_decode_path_unaffected_by_opt_in() -> None:
    """rows==1 never gets the WMMA rewrite, regardless of opt-in state."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=1)
    # rows==1 Q8_0 raw resolves to the pack8 decode GEMV alias (out_features %
    # 8 == 0 path), never to WMMA prefill.
    assert key == _DECODE_PACK8_BF16


def test_wmma_prefill_raw_q4_k_off_by_default_rows_gt_1() -> None:
    """Raw Q4_K rows>1 keeps the decode-shaped prefill alias unless opted in."""

    key, _, _ = _capture_launch(rows=4, quant_key="gguf_q4_k", layout=LAYOUT_RAW_GGUF)
    assert key == _Q4_PREFILL_BF16


def test_wmma_prefill_kwarg_opts_in_raw_q4_k_rows_gt_1() -> None:
    """Per-call opt-in routes raw Q4_K rows>1 to the new P8.2 WMMA family."""

    key, args, kwargs = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=True,
        threads=128,
    )
    assert key == _Q4_WMMA_BF16
    assert args == (100, 10, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


@pytest.mark.parametrize(
    ("output_dtype", "prefill_key", "wmma_key"),
    [
        (
            GGUF_OUTPUT_FP16,
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_fp16_out"),
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "wmma_prefill_bf16_fp16_out"),
        ),
        (
            GGUF_OUTPUT_F32,
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "prefill_bf16_f32_out"),
            KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "wmma_prefill_bf16_f32_out"),
        ),
    ],
)
def test_wmma_prefill_raw_q4_k_output_dtype_variants_route(
    output_dtype: str, prefill_key: KernelKey, wmma_key: KernelKey
) -> None:
    """Raw Q4_K FP16/F32 output variants also rewrite to matching WMMA keys."""

    key, _, _ = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        output_dtype=output_dtype,
        use_wmma_prefill=False,
        extra_keys=(prefill_key, wmma_key),
    )
    assert key == prefill_key
    key, args, kwargs = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        output_dtype=output_dtype,
        use_wmma_prefill=True,
        extra_keys=(prefill_key, wmma_key),
    )
    assert key == wmma_key
    assert args == (100, 10, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_wmma_prefill_env_var_opts_in_raw_q4_k(monkeypatch) -> None:
    """The env opt-in applies to raw Q4_K as well as Q8_0."""

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "1")
    key, _, _ = _capture_launch(rows=4, quant_key="gguf_q4_k", layout=LAYOUT_RAW_GGUF)
    assert key == _Q4_WMMA_BF16


def test_wmma_prefill_session_opts_in_raw_q4_k() -> None:
    """The session toggle applies to raw Q4_K and can be forced off per call."""

    set_wmma_prefill_enabled(True)
    key, _, _ = _capture_launch(rows=4, quant_key="gguf_q4_k", layout=LAYOUT_RAW_GGUF)
    assert key == _Q4_WMMA_BF16
    key, _, _ = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        use_wmma_prefill=False,
    )
    assert key == _Q4_PREFILL_BF16


def test_wmma_prefill_raw_q4_k_decode_path_unaffected_by_opt_in() -> None:
    """rows==1 raw Q4_K stays on the scalar raw GEMV path."""

    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        use_wmma_prefill=True,
    )
    assert key == _Q4_GEMV_BF16


def test_wmma_prefill_q4_k_pack8_layout_keeps_pack8_fallback_under_opt_in() -> None:
    """Dense 2D Q4_K pack8 materialization is not silently reinterpreted as raw."""

    key, args, kwargs = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_Q4_K_PACK8,
        use_wmma_prefill=True,
    )
    assert key == _Q4_PACK8_PREFILL_BF16
    assert args == (100, 11, 12, 13, 200, 4, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_wmma_prefill_raw_q4_k_requires_256_aligned_in_features() -> None:
    """Raw Q4_K WMMA requires in_features % 256 == 0; otherwise fallback."""

    key, _, _ = _capture_launch(
        rows=4,
        quant_key="gguf_q4_k",
        layout=LAYOUT_RAW_GGUF,
        in_features=1000,
        use_wmma_prefill=True,
    )
    assert key == _Q4_PREFILL_BF16


def test_wmma_prefill_q5_k_not_yet_supported_keeps_decode_path() -> None:
    """Q5_K does not yet ship a WMMA prefill family (lands in P8.5)."""

    q5_prefill = KernelKey("hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_bf16_out")
    key, _, _ = _capture_launch(
        rows=4, quant_key="gguf_q5_k", use_wmma_prefill=True, extra_keys=(q5_prefill,)
    )
    assert key == q5_prefill


def test_wmma_prefill_unaligned_in_features_falls_back_to_decode_path() -> None:
    """Q8_0 requires in_features % 32 == 0; unaligned shapes skip the WMMA path."""

    key, _, _ = _capture_launch(rows=4, in_features=1000, use_wmma_prefill=True)
    assert key == _PREFILL_BF16


def test_wmma_prefill_threads_silently_dropped_on_wmma_path() -> None:
    """The caller's ``threads`` value applies to the decode path only."""

    key, _, kwargs = _capture_launch(rows=4, use_wmma_prefill=True, threads=128)
    assert key == _WMMA_BF16
    assert "threads" not in kwargs
    # And confirm threads still flows through on the decode path:
    key2, _, kwargs2 = _capture_launch(rows=4, threads=128)
    assert key2 == _PREFILL_BF16
    assert kwargs2.get("threads") == 128


def test_t16_pair_concat_fuses_q8_shared_gate_up() -> None:
    """Resident Q8T16 shared gate/up can use the concatenated dual ABI."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair_concat(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=200,
            rows=1,
            in_features=2048,
            out_features=512,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out = original  # type: ignore[assignment]

    assert fused is True
    assert pair_calls == [
        ((100, 14, 14, 200, 1, 2048, 512, 512), {"stream": 7, "runtime": "runtime-sentinel"})
    ]


def test_t16_pair_fuses_q8_separate_outputs() -> None:
    """Resident Q8T16 same-input pairs can share one split-output launch."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out
    gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused_equal = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=1,
            in_features=2048,
            out_features=512,
            stream=7,
            runtime="runtime-sentinel",
        )
        fused_unequal = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=101,
            out_a_ptr=201,
            out_b_ptr=301,
            rows=1,
            in_features=2048,
            out_features=1536,
            out_features_b=512,
            stream=8,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out = original  # type: ignore[assignment]

    assert fused_equal is True
    assert fused_unequal is True
    assert pair_calls == [
        ((100, 14, 14, 200, 300, 1, 2048, 512, 512), {"stream": 7, "runtime": "runtime-sentinel"}),
        ((101, 14, 14, 201, 301, 1, 2048, 1536, 512), {"stream": 8, "runtime": "runtime-sentinel"}),
    ]


def test_t16_triple_fuses_q8_separate_outputs() -> None:
    """Resident Q8T16 full-attention Q/K/V can share one split-output launch."""

    weight_a = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_b = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    weight_c = _fake_weight(layout=LAYOUT_GGUF_Q8_0_T16, quant_key="gguf_q8_0_t16_v1")
    import hipengine.runtime.gguf_linear as gl

    triple_calls: list[tuple] = []

    def fake_triple(*args, **kwargs):
        triple_calls.append((args, kwargs))

    original = gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out
    gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out = fake_triple  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_triple(
            weight_a,
            weight_b,
            weight_c,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            out_c_ptr=400,
            rows=1,
            in_features=2048,
            out_features=1024,
            out_features_b=512,
            out_features_c=512,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        gl.gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out = original  # type: ignore[assignment]

    assert fused is True
    assert triple_calls == [
        ((100, 14, 14, 14, 200, 300, 400, 1, 2048, 1024, 512, 512), {"stream": 7, "runtime": "runtime-sentinel"})
    ]


def test_wmma_prefill_pair_declines_fusion_when_q8_0_rows_gt_1() -> None:
    """Pair fast paths defer to two singletons when Q8_0 rows>1 + WMMA opt-in.

    There is no Q8_0 dual WMMA prefill yet (follow-up P8 step). The pair
    function must return ``False`` so the caller falls back to two
    ``launch_gguf_linear`` calls that each hit the WMMA family.
    """

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    fused = launch_gguf_linear_pair(
        weight_a,
        weight_b,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        rows=4,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=True,
    )
    assert fused is False


def test_wmma_prefill_pair_fuses_raw_q4_k_dual_prefill_when_opted_in() -> None:
    """Raw Q4_K gate+up pair routes to the P8.2 dual WMMA path."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out
    gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=4,
            in_features=1024,
            out_features=2048,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        )
    finally:
        gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = original  # type: ignore[assignment]
    assert fused is True
    assert pair_calls == [
        ((100, 10, 10, 200, 300, 4, 1024, 2048), {"stream": 7, "runtime": "runtime-sentinel"})
    ]


def test_wmma_prefill_pair_raw_q4_k_requires_opt_in() -> None:
    """Raw Q4_K pair has no default-off pair fast path; callers fall back."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    fused = launch_gguf_linear_pair(
        weight_a,
        weight_b,
        x_ptr=100,
        out_a_ptr=200,
        out_b_ptr=300,
        rows=4,
        in_features=1024,
        out_features=2048,
        use_wmma_prefill=False,
    )
    assert fused is False


def test_wmma_prefill_pair_raw_q4_k_unaligned_falls_back() -> None:
    """Raw Q4_K dual WMMA pair requires the 256-wide Q4_K block alignment."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q4_k")
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out
    gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=4,
            in_features=1000,
            out_features=2048,
            use_wmma_prefill=True,
        )
    finally:
        gl.gguf_q4_k_wmma_prefill_dual_bf16_bf16_out = original  # type: ignore[assignment]
    assert fused is False
    assert pair_calls == []


def test_wmma_prefill_pair_still_fuses_q4_k_pack8_dual_prefill() -> None:
    """WMMA opt-in does NOT poison the Q4_K pack8 dual prefill fast path."""

    weight_a = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    weight_b = _fake_weight(layout=LAYOUT_Q4_K_PACK8, quant_key="gguf_q4_k")
    # Stub out the actual pair kernel so we don't touch the GPU.
    import hipengine.runtime.gguf_linear as gl

    pair_calls: list[tuple] = []

    def fake_pair(*args, **kwargs):
        pair_calls.append((args, kwargs))

    original = gl.gguf_q4_k_pack8_dual_prefill_bf16_bf16_out
    gl.gguf_q4_k_pack8_dual_prefill_bf16_bf16_out = fake_pair  # type: ignore[assignment]
    try:
        fused = launch_gguf_linear_pair(
            weight_a,
            weight_b,
            x_ptr=100,
            out_a_ptr=200,
            out_b_ptr=300,
            rows=4,
            in_features=1024,
            out_features=2048,
            use_wmma_prefill=True,
        )
    finally:
        gl.gguf_q4_k_pack8_dual_prefill_bf16_bf16_out = original  # type: ignore[assignment]
    assert fused is True
    assert len(pair_calls) == 1
