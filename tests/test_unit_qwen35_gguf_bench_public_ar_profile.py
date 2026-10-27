from __future__ import annotations

import argparse

import pytest

from hipengine.generation.qwen35_gguf import (
    _GGUF_PUBLIC_USE_GEMV_DECODE,
    _resident_session_wmma_prefill_default,
)
from scripts.qwen35_gguf_bench import apply_public_ar_profile, shipping_ar_route_mismatch


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "public_ar_profile": None,
        "use_wmma_prefill": None,
        "use_gemv_decode": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_resolves_the_public_session_selectors(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    monkeypatch.setenv("HIPENGINE_GGUF_GEMV_DECODE", "0")
    args = _args()
    assert apply_public_ar_profile(args) is True
    assert args.use_wmma_prefill is _resident_session_wmma_prefill_default()
    assert args.use_gemv_decode is _GGUF_PUBLIC_USE_GEMV_DECODE


def test_default_reads_public_diagnostic_at_call_time(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "0")
    args = _args()
    assert apply_public_ar_profile(args) is True
    assert args.use_wmma_prefill is False
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "1")
    args = _args()
    assert apply_public_ar_profile(args) is True
    assert args.use_wmma_prefill is True


@pytest.mark.parametrize("name", ("use_wmma_prefill", "use_gemv_decode"))
def test_explicit_diagnostic_selector_overrides_default(name) -> None:
    args = _args(**{name: False})
    assert apply_public_ar_profile(args) is True
    assert getattr(args, name) is False


def test_legacy_no_profile_preserves_low_level_env_resolution() -> None:
    args = _args(public_ar_profile=False)
    assert apply_public_ar_profile(args) is False
    assert args.use_wmma_prefill is None and args.use_gemv_decode is None


def test_legacy_profile_matches_shared_session_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", raising=False)
    args = _args(public_ar_profile=True)
    assert apply_public_ar_profile(args) is True
    assert args.use_wmma_prefill is _resident_session_wmma_prefill_default()
    assert args.use_gemv_decode is _GGUF_PUBLIC_USE_GEMV_DECODE


def test_public_ar_profile_rejects_an_explicit_contradiction() -> None:
    args = _args(public_ar_profile=True, use_wmma_prefill=False)
    with pytest.raises(ValueError, match="--public-ar-profile conflicts with --no-use-wmma-prefill"):
        apply_public_ar_profile(args)

    # An explicit agreement is not a conflict.
    args = _args(public_ar_profile=True, use_wmma_prefill=True, use_gemv_decode=True)
    assert apply_public_ar_profile(args) is True


def test_route_mismatch_uses_effective_selectors_not_legacy_flag(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", raising=False)
    for legacy in (False, True):
        assert shipping_ar_route_mismatch(legacy, [False, False], [True, True]) is True
        assert shipping_ar_route_mismatch(legacy, [True, True], [True, True]) is False
        assert shipping_ar_route_mismatch(legacy, [True], [False]) is True
        # A backend that reports no selector does not prove a mismatch.
        assert shipping_ar_route_mismatch(legacy, [None, None], [None, None]) is False


def test_route_mismatch_follows_public_diagnostic_default(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "0")
    assert shipping_ar_route_mismatch(False, [False], [True]) is False
    assert shipping_ar_route_mismatch(False, [True], [True]) is True


def test_invalid_public_diagnostic_default_fails_loudly(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL", "typo")
    with pytest.raises(ValueError, match="HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL"):
        apply_public_ar_profile(_args())
