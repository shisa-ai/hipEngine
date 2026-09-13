from __future__ import annotations

import pytest

import hipengine.kernels.registry as registry
from hipengine.kernels.registry import (
    DuplicateKernelError,
    KernelKey,
    MissingKernelError,
    clear_registry_for_tests,
    register,
    registered_keys,
    resolve,
)


def dummy_kernel(*args, **kwargs):
    return args, kwargs


def setup_function() -> None:
    clear_registry_for_tests()


def test_register_and_resolve_exact_key() -> None:
    key = KernelKey("hip_gfx1100", "rmsnorm", "fp16")
    register(key, dummy_kernel)

    assert resolve(backend="hip_gfx1100", layer="rmsnorm", quant="fp16") is dummy_kernel
    assert registered_keys() == (key,)


def test_duplicate_registration_is_an_error() -> None:
    key = KernelKey("hip_gfx1100", "rmsnorm", "fp16")
    register(key, dummy_kernel)

    with pytest.raises(DuplicateKernelError):
        register(key, dummy_kernel)


def test_resolve_falls_back_to_no_variant_then_fp16_then_cpu_reference() -> None:
    exact_no_variant = lambda: "same quant no variant"
    fp16_fallback = lambda: "fp16 fallback"
    cpu_fallback = lambda: "cpu fallback"

    register(KernelKey("hip_gfx1100", "attention_decode", "w8a16"), exact_no_variant)
    register(KernelKey("hip_gfx1100", "mlp", "fp16"), fp16_fallback)
    register(KernelKey("cpu_reference", "lm_head", "w8a16"), cpu_fallback)

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="attention_decode",
            quant="w8a16",
            variant="split_k",
        )
        is exact_no_variant
    )
    assert resolve(backend="hip_gfx1100", layer="mlp", quant="w8a16") is fp16_fallback
    assert resolve(backend="hip_gfx1100", layer="lm_head", quant="w8a16") is cpu_fallback


def test_missing_kernel_error_is_clean_and_specific() -> None:
    with pytest.raises(MissingKernelError) as exc_info:
        resolve(backend="hip_gfx1100", layer="not_registered", quant="w4_paro", variant="fast")

    message = str(exc_info.value)
    assert "no kernel implementation" in message
    assert "hip_gfx1100" in message
    assert "not_registered" in message
    assert "w4_paro" in message
    assert "cpu_reference" in message


def test_resolve_memoizes_candidates_and_invalidates_on_mutation(monkeypatch) -> None:
    key = KernelKey("hip_gfx1151", "linear", "gguf_q4_k", "decode")
    first = lambda: "first"
    second = lambda: "second"
    candidate_calls = 0
    original_candidate_keys = registry._candidate_keys

    def counting_candidate_keys(requested):
        nonlocal candidate_calls
        candidate_calls += 1
        return original_candidate_keys(requested)

    monkeypatch.setattr(registry, "_candidate_keys", counting_candidate_keys)
    register(key, first)

    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is first
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is first
    assert candidate_calls == 1

    register(key, second, replace=True)
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is second
    assert candidate_calls == 2


def test_restore_registry_for_tests_invalidates_cached_fallback() -> None:
    exact_key = KernelKey(
        "hip_gfx1151",
        "linear",
        "gguf_q6_k_t16_v1",
        "t16_gemv_decode_bf16_f32_out",
    )
    fallback_key = KernelKey("cpu_reference", "linear", "fp16")

    def exact(*args, **kwargs):
        return args, kwargs

    def fallback(*args):
        return args

    register(exact_key, exact)
    register(fallback_key, fallback)
    baseline = dict(registry._KERNELS)

    clear_registry_for_tests()
    register(fallback_key, fallback)
    assert (
        resolve(
            backend=exact_key.backend,
            layer=exact_key.layer,
            quant=exact_key.quant,
            variant=exact_key.variant,
        )
        is fallback
    )

    registry.restore_registry_for_tests(baseline)
    assert (
        resolve(
            backend=exact_key.backend,
            layer=exact_key.layer,
            quant=exact_key.quant,
            variant=exact_key.variant,
        )
        is exact
    )
