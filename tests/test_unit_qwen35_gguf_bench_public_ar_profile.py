from __future__ import annotations

import argparse

import pytest

from scripts.qwen35_gguf_bench import (
    PUBLIC_AR_PROFILE_KWARGS,
    apply_public_ar_profile,
    shipping_ar_route_mismatch,
)


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "public_ar_profile": False,
        "use_wmma_prefill": None,
        "use_gemv_decode": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_public_ar_profile_is_opt_in_and_sets_the_shipping_selectors() -> None:
    args = _args()
    assert apply_public_ar_profile(args) is False
    assert args.use_wmma_prefill is None and args.use_gemv_decode is None

    args = _args(public_ar_profile=True)
    assert apply_public_ar_profile(args) is True
    for name, value in PUBLIC_AR_PROFILE_KWARGS.items():
        assert getattr(args, name) is value


def test_public_ar_profile_matches_the_session_path_selectors() -> None:
    # The product path is hipengine/generation/qwen35_gguf.py; if it ever stops
    # passing these selectors the bench alias must not keep claiming parity.
    source = (
        (__import__("pathlib").Path(__file__).resolve().parents[1] / "hipengine" / "generation" / "qwen35_gguf.py")
        .read_text(encoding="utf-8")
    )
    for name in PUBLIC_AR_PROFILE_KWARGS:
        assert f"{name}=True" in source, f"shipping AR session no longer passes {name}=True"


def test_public_ar_profile_rejects_an_explicit_contradiction() -> None:
    args = _args(public_ar_profile=True, use_wmma_prefill=False)
    with pytest.raises(ValueError, match="--public-ar-profile conflicts with --no-use-wmma-prefill"):
        apply_public_ar_profile(args)

    # An explicit agreement is not a conflict.
    args = _args(public_ar_profile=True, use_wmma_prefill=True, use_gemv_decode=True)
    assert apply_public_ar_profile(args) is True


def test_route_mismatch_flags_only_the_non_shipping_default_route() -> None:
    assert shipping_ar_route_mismatch(False, [False, False]) is True
    assert shipping_ar_route_mismatch(False, [True, True]) is False
    # A model/backend that resolves no WMMA route (None) is not a mismatch.
    assert shipping_ar_route_mismatch(False, [None, None]) is False
    assert shipping_ar_route_mismatch(True, [False]) is False
