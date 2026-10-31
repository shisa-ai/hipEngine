"""Explicit R28 selector input for the physical Q4 dual-SiLU launch."""

from __future__ import annotations

import pytest

from hipengine.core.specdec2_scope import (
    physical_exact_rowtiles_session,
    q4_t16_physical_extra_rowtiles_session,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.runtime import gguf_linear as gguf_linear_module
from hipengine.runtime.gguf_linear import (
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear_pair_silu,
)
from tests._gguf_device_weight_fixtures import q4_k_t16_weight as _weight

R28_ENV = "HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28"
R28_VARIANT = "dense_dual_wmma_prefill_row32_bf16_bf16_out"
R28_ACCEPTED = (
    ("1", True),
    ("true", True),
    (" YES ", True),
    ("on", True),
    ("0", False),
    ("false", False),
    (" No ", False),
    ("off", False),
)
R28_REJECTED = ("2", "maybe", "", "truthy", "1.0")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(R28_ENV, raising=False)
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    clear_gguf_linear_dispatch_cache()
    yield
    gguf_linear_module._rowtile_variant_policy_env_cache.clear()
    clear_gguf_linear_dispatch_cache()


def _register_r28(calls: list[str]):
    key = KernelKey(
        "hip_gfx1100", "linear_pair_silu", "gguf_q4_k_t16_v1", R28_VARIANT
    )
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    register(key, lambda *args, **kwargs: calls.append(R28_VARIANT), replace=True)
    return key, original


def _launch(*, rows: int = 28, layer_id: int | None = 1, **kwargs) -> bool:
    weight_a = _weight(0x1000, in_features=5_120, out_features=17_408)
    weight_b = _weight(0x2000, in_features=5_120, out_features=17_408)
    with (
        q4_t16_physical_extra_rowtiles_session(True),
        physical_exact_rowtiles_session(True),
    ):
        return launch_gguf_linear_pair_silu(
            weight_a,
            weight_b,
            0x3000,
            0x4000,
            rows,
            5_120,
            17_408,
            layer_id=layer_id,
            **kwargs,
        )


def test_explicit_r28_true_bypasses_a_stale_false_ambient_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    key, original = _register_r28(calls)
    try:
        monkeypatch.setenv(R28_ENV, "1")
        gguf_linear_module._rowtile_variant_policy_env_cache[(R28_ENV, False)] = False
        assert _launch(r28_enabled=True)
    finally:
        register(key, original, replace=True)
    assert calls == [R28_VARIANT]


def test_explicit_r28_false_wins_over_ambient_true_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    key, original = _register_r28(calls)
    try:
        monkeypatch.setenv(R28_ENV, "1")
        gguf_linear_module._rowtile_variant_policy_env_cache[(R28_ENV, False)] = True
        assert not _launch(r28_enabled=False)
    finally:
        register(key, original, replace=True)
    assert calls == []


def test_explicit_r28_true_keeps_periodic_strict_layer_and_quant_shape_checks() -> None:
    calls: list[str] = []
    key, original = _register_r28(calls)
    try:
        assert _launch(layer_id=1, r28_enabled=True)
        assert not _launch(layer_id=8, r28_enabled=True)
        assert not _launch(layer_id=None, r28_enabled=True)
        assert not _launch(rows=27, r28_enabled=True)
    finally:
        register(key, original, replace=True)
    assert calls == [R28_VARIANT]


@pytest.mark.parametrize("raw,expected", R28_ACCEPTED)
def test_unbound_r28_uses_original_selector_boolean_syntax(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    calls: list[str] = []
    key, original = _register_r28(calls)
    try:
        monkeypatch.setenv(R28_ENV, raw)
        assert _launch(r28_enabled=None) is expected
    finally:
        register(key, original, replace=True)
    assert calls == ([R28_VARIANT] if expected else [])


@pytest.mark.parametrize("raw", R28_REJECTED)
def test_unbound_r28_rejects_non_selector_boolean_strings(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv(R28_ENV, raw)
    with pytest.raises(ValueError, match=f"{R28_ENV} must be a boolean value"):
        _launch(r28_enabled=None)


@pytest.mark.parametrize("populated", [True, False])
def test_unbound_r28_keeps_historical_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
    populated: bool,
) -> None:
    calls: list[str] = []
    key, original = _register_r28(calls)
    try:
        monkeypatch.setenv(R28_ENV, "1" if populated else "0")
        gguf_linear_module._rowtile_variant_policy_env_cache[(R28_ENV, False)] = populated
        assert _launch(r28_enabled=None) is populated
        assert (
            gguf_linear_module._rowtile_variant_policy_env_cache[(R28_ENV, False)]
            is populated
        )

        # The unbound path keeps its historical process cache: an ambient flip
        # does not retarget an already-populated entry.
        monkeypatch.setenv(R28_ENV, "0" if populated else "1")
        assert _launch(r28_enabled=None) is populated
        assert (
            gguf_linear_module._rowtile_variant_policy_env_cache[(R28_ENV, False)]
            is populated
        )

        # A bound owner bypasses the cache and honors its explicit decision.
        assert _launch(r28_enabled=not populated) is (not populated)
        assert _launch(r28_enabled=populated) is populated
        assert (
            gguf_linear_module._rowtile_variant_policy_env_cache[(R28_ENV, False)]
            is populated
        )
    finally:
        register(key, original, replace=True)
    expected_calls = [R28_VARIANT] if not populated else [R28_VARIANT] * 3
    assert calls == expected_calls
