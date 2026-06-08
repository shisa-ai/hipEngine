"""Routing tests for the P9.B6 GGUF pack8 GEMV decode opt-in dispatch.

Mirrors :mod:`tests/test_gguf_linear_dispatch.py` but focused on the
``rows == 1`` decode rewrite added in P9.B6: ``pack8_gemv_*`` ->
``pack8_gemv_decode_*`` for the matching ``(quant, layer)``, controlled by
``HIPENGINE_GGUF_GEMV_DECODE`` env var, the ``gemv_decode_session(...)``
context manager, and per-call ``use_gemv_decode`` kwarg with the same
precedence as the WMMA prefill toggle.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Real kernel module imports keep the registry populated across tests.
import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv  # noqa: F401
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.kernels.registry import _KERNELS
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import (
    GGUF_OUTPUT_BF16,
    gemv_decode_session,
    gguf_gemv_decode_enabled,
    launch_gguf_linear,
    launch_gguf_linear_pair_concat,
    set_gemv_decode_enabled,
)


def _fake_weight(*, layout: str, quant_key: str):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "qweight": SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
        "scales": SimpleNamespace(tensor=SimpleNamespace(ptr=12)),
        "mins": SimpleNamespace(tensor=SimpleNamespace(ptr=13)),
    }

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(layout=layout, quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


_Q8_DECODE_PACK8 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
_Q8_DECODE_PACK8_F32 = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_bf16_f32_out")
_Q8_GEMV_DECODE = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "pack8_gemv_decode_bf16_bf16_out"
)
_Q8_PREFILL = KernelKey("hip_gfx1100", "linear", "gguf_q8_0", "prefill_bf16_bf16_out")
_Q8_WMMA_DUAL_PREFILL = KernelKey(
    "hip_gfx1100", "linear", "gguf_q8_0", "wmma_prefill_dual_gate_up_bf16_bf16_out"
)


@pytest.fixture(autouse=True)
def _reset_gemv_decode_state(monkeypatch):
    monkeypatch.delenv("HIPENGINE_GGUF_GEMV_DECODE", raising=False)
    set_gemv_decode_enabled(None)
    yield
    set_gemv_decode_enabled(None)


def _capture_launch(
    *,
    rows: int,
    in_features: int = 1024,
    out_features: int = 2048,
    use_gemv_decode: bool | None = None,
    quant_key: str = "gguf_q8_0",
    layout: str = LAYOUT_RAW_GGUF,
    output_dtype: str = GGUF_OUTPUT_BF16,
    extra_keys: tuple[KernelKey, ...] = (),
    remove_keys: tuple[KernelKey, ...] = (),
):
    weight = _fake_weight(layout=layout, quant_key=quant_key)
    captured: dict[str, object] = {"key": None, "args": None, "kwargs": None}
    keys = (
        _Q8_DECODE_PACK8,
        _Q8_DECODE_PACK8_F32,
        _Q8_GEMV_DECODE,
        _Q8_PREFILL,
    ) + extra_keys
    originals = {
        k: resolve(
            backend=k.backend,
            layer=k.layer,
            quant=k.quant,
            variant=k.variant,
            missing="none",
        )
        for k in keys
    }

    def make_fake(key: KernelKey):
        def fake(*args, **kwargs):
            captured["key"] = key
            captured["args"] = args
            captured["kwargs"] = kwargs

        return fake

    try:
        for k in keys:
            register(k, make_fake(k), replace=True)
        for k in remove_keys:
            # Simulate a missing kernel by clearing the registry entry. The
            # registry stores ``None`` and the resolver treats it the same
            # as an unregistered key (see ``_KERNELS.get`` short-circuit).
            register(k, None, replace=True)
        launch_gguf_linear(
            weight,
            x_ptr=100,
            out_ptr=200,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_dtype=output_dtype,
            stream=7,
            runtime="runtime-sentinel",
            use_gemv_decode=use_gemv_decode,
        )
    finally:
        for k, fn in originals.items():
            if fn is None:
                # The key was unregistered before the test ran; leave it
                # unregistered so we don't poison the global registry with
                # a ``None`` entry that later tests would observe.
                _KERNELS.pop(k, None)
            else:
                register(k, fn, replace=True)

    return captured["key"], captured["args"], captured["kwargs"]


# ---------------------------------------------------------------------------
# Default off + opt-in precedence.
# ---------------------------------------------------------------------------


def test_p9_c1_pair_concat_routes_q8_dual_wmma_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    """P9.C1 dispatch pin: Q8_0 shared gate+up prefill uses dual concat WMMA."""

    weight_a = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    weight_b = _fake_weight(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q8_0")
    captured: dict[str, object] = {}

    def fake_dual(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    original = resolve(
        backend=_Q8_WMMA_DUAL_PREFILL.backend,
        layer=_Q8_WMMA_DUAL_PREFILL.layer,
        quant=_Q8_WMMA_DUAL_PREFILL.quant,
        variant=_Q8_WMMA_DUAL_PREFILL.variant,
        missing="none",
    )
    try:
        register(_Q8_WMMA_DUAL_PREFILL, fake_dual, replace=True)
        monkeypatch.setattr(
            "hipengine.runtime.gguf_linear.gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out",
            fake_dual,
        )
        assert launch_gguf_linear_pair_concat(
            weight_a,
            weight_b,
            x_ptr=100,
            out_ptr=300,
            rows=512,
            in_features=2048,
            out_features=4096,
            stream=7,
            runtime="runtime-sentinel",
            use_wmma_prefill=True,
        ) is True
    finally:
        if original is None:
            _KERNELS.pop(_Q8_WMMA_DUAL_PREFILL, None)
        else:
            register(_Q8_WMMA_DUAL_PREFILL, original, replace=True)

    assert captured["args"] == (100, 10, 10, 300, 512, 2048, 4096, 4096)
    assert captured["kwargs"] == {
        "tile_m": 16,
        "tile_n": 32,
        "stream": 7,
        "runtime": "runtime-sentinel",
    }


def test_gemv_decode_off_by_default_routes_legacy_pack8_gemv() -> None:
    """Without any opt-in, rows==1 Q8_0 stays on the legacy pack8_gemv decoder."""

    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8


def test_gemv_decode_kwarg_opts_in_q8_0() -> None:
    """Per-call ``use_gemv_decode=True`` rewrites to the P9.B3 GEMV decoder."""

    key, args, kwargs = _capture_launch(rows=1, use_gemv_decode=True)
    assert key == _Q8_GEMV_DECODE
    # Raw ABI: (x, qweight, out, rows, in_f, out_f)
    assert args == (100, 10, 200, 1, 1024, 2048)
    assert kwargs == {"stream": 7, "runtime": "runtime-sentinel"}


def test_gemv_decode_env_var_opts_in(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", "1")
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_GEMV_DECODE


def test_gemv_decode_env_var_falsy_keeps_legacy(monkeypatch) -> None:
    for value in ("", "0", "false", "no", "off"):
        monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", value)
        key, _, _ = _capture_launch(rows=1)
        assert key == _Q8_DECODE_PACK8, f"env value {value!r} should keep legacy decoder"


def test_gemv_decode_env_var_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "On"):
        monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", value)
        key, _, _ = _capture_launch(rows=1)
        assert key == _Q8_GEMV_DECODE, f"env value {value!r} should enable GEMV decode"


def test_gemv_decode_kwarg_overrides_session() -> None:
    """Per-call ``use_gemv_decode=False`` wins over an enabled session."""

    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=1, use_gemv_decode=False)
    assert key == _Q8_DECODE_PACK8


def test_gemv_decode_session_toggle_persists_until_cleared() -> None:
    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_GEMV_DECODE
    set_gemv_decode_enabled(False)
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8
    set_gemv_decode_enabled(None)
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8  # env default is off


def test_gemv_decode_session_context_manager_restores_previous() -> None:
    set_gemv_decode_enabled(False)
    with gemv_decode_session(True):
        key, _, _ = _capture_launch(rows=1)
        assert key == _Q8_GEMV_DECODE
    key, _, _ = _capture_launch(rows=1)
    assert key == _Q8_DECODE_PACK8


def test_gguf_gemv_decode_enabled_resolver_precedence(monkeypatch) -> None:
    """The resolver mirrors :func:`gguf_wmma_prefill_enabled` precedence."""

    monkeypatch.delenv("HIPENGINE_GGUF_GEMV_DECODE", raising=False)
    set_gemv_decode_enabled(None)
    assert gguf_gemv_decode_enabled() is False
    assert gguf_gemv_decode_enabled(True) is True
    set_gemv_decode_enabled(True)
    assert gguf_gemv_decode_enabled() is True
    assert gguf_gemv_decode_enabled(False) is False
    set_gemv_decode_enabled(None)
    monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", "1")
    assert gguf_gemv_decode_enabled() is True


# ---------------------------------------------------------------------------
# Prefill path unaffected; fallback on missing key.
# ---------------------------------------------------------------------------


def test_gemv_decode_prefill_path_unaffected_by_opt_in() -> None:
    """rows>1 never gets the GEMV-decode rewrite, regardless of opt-in state."""

    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=4)
    assert key == _Q8_PREFILL


def test_gemv_decode_fallback_when_registry_key_missing() -> None:
    """If the P9.B3 kernel is not registered, the rewrite silently falls back.

    The default-off behaviour must be preserved when a runtime is built without
    the new GEMV decode kernels (e.g. partial build trees or older caches).
    """

    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(rows=1, remove_keys=(_Q8_GEMV_DECODE,))
    # Even though opt-in is on, the rewrite returns the original dispatch
    # because the rewritten registry key is missing.
    assert key == _Q8_DECODE_PACK8


# ---------------------------------------------------------------------------
# Q5_K / Q6_K dense decode opt-in (P9.B4b dense Q6_K kernel; legacy Q5_K
# stays on the existing decoder, no P9 dense Q5_K kernel exists).
# ---------------------------------------------------------------------------


def _q_decode_pack8(quant: str) -> KernelKey:
    return KernelKey("hip_gfx1100", "linear", quant, "pack8_gemv_bf16_bf16_out")


def _q_gemv_decode(quant: str) -> KernelKey:
    return KernelKey("hip_gfx1100", "linear", quant, "pack8_gemv_decode_bf16_bf16_out")


def test_gemv_decode_q6_k_opt_in_rewrites() -> None:
    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q6_k",
        layout=LAYOUT_RAW_GGUF,
        use_gemv_decode=True,
        extra_keys=(_q_decode_pack8("gguf_q6_k"), _q_gemv_decode("gguf_q6_k")),
    )
    assert key == _q_gemv_decode("gguf_q6_k")


def test_gemv_decode_q5_k_falls_back_without_registered_kernel() -> None:
    """Q5_K dense decode has no P9 kernel; opt-in is a no-op for it."""

    set_gemv_decode_enabled(True)
    key, _, _ = _capture_launch(
        rows=1,
        quant_key="gguf_q5_k",
        layout=LAYOUT_RAW_GGUF,
        extra_keys=(_q_decode_pack8("gguf_q5_k"), _q_gemv_decode("gguf_q5_k")),
        remove_keys=(_q_gemv_decode("gguf_q5_k"),),
    )
    assert key == _q_decode_pack8("gguf_q5_k")
