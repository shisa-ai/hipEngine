"""Role isolation for the iu8-WMMA dense Q8_0 route.

The route is a single boolean that claims every eligible Q8_0 prefill linear.
Its two recorded rejections each closed an *arm*, not a role:
``20260915T015835`` rejected the unchanged generic switch and said so
explicitly ("not role-isolated non-GR work"), and ``20260915T022059`` rejected
the non-GR arm while recording that it "does not implicate every projection".
The ledger's reopen condition is role-isolated evidence, which the boolean
cannot express.

``HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES`` narrows the route to named roles. These
tests pin the scope's contract: unset must reproduce the whole-route shape that
already failed, a named role must claim, an unnamed role must decline, and a
weight the route cannot place must decline rather than be silently covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from hipengine.kernels.registry import KernelKey, register
from hipengine.runtime.gguf_linear import (
    GGUFLinearDispatch,
    _q8_iu8_role_cache_key,
    _q8_iu8_wmma_dispatch,
)

_CANDIDATE = KernelKey(
    "hip_gfx1151", "linear", "gguf_q8_0", "iu8_wmma_prefill_f32_f32_out"
)
_PARENT = GGUFLinearDispatch(
    KernelKey(
        "hip_gfx1151", "linear", "gguf_q8_0",
        "coltile8_rowbatch4_wave_scale_f32_f32_out",
    ),
    "raw",
)


@dataclass
class _Spec:
    slot_path: str


@dataclass
class _Weight:
    slot_path: str
    spec: _Spec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = _Spec(self.slot_path)


def _weight(slot_path: str) -> _Weight:
    return _Weight(slot_path)


def _dispatch(weight: object | None, *, rows: int = 512):
    return _q8_iu8_wmma_dispatch(
        _PARENT, rows=rows, in_features=10240, out_features=320, weight=weight
    )


@pytest.fixture(autouse=True)
def _route_enabled(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM", "1")
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", raising=False)
    register(_CANDIDATE, lambda *args, **kwargs: None, replace=True)


def test_unset_role_scope_keeps_the_whole_route(monkeypatch):
    """Unset must reproduce the shape that already failed, not a narrowed one."""

    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", raising=False)
    assert _dispatch(_weight("layers.5.attn_gate.weight")) == GGUFLinearDispatch(
        _CANDIDATE, "raw"
    )


@pytest.mark.parametrize("rows", [71, 256, 512, 1024])
def test_generic_dense_q8_can_cover_gr_down_without_its_dedicated_flag(
    rows, monkeypatch
):
    """The generic route claims GR-down shapes; the role scope is what narrows it."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN", "0")
    weight = _weight("layers.5.hc_ffn_down.weight")
    actual = _q8_iu8_wmma_dispatch(
        _PARENT, rows=rows, in_features=10240, out_features=320, weight=weight
    )
    assert actual == (GGUFLinearDispatch(_CANDIDATE, "raw") if rows > 256 else _PARENT)


def test_named_role_claims(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate")
    assert _dispatch(_weight("layers.5.attn_gate.weight")) == GGUFLinearDispatch(
        _CANDIDATE, "raw"
    )


def test_unnamed_role_declines(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate")
    assert _dispatch(_weight("layers.5.ffn_gate_shexp.weight")) == _PARENT


def test_multiple_roles_are_a_set(monkeypatch):
    monkeypatch.setenv(
        "HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate, ffn_down_shexp"
    )
    assert _dispatch(_weight("layers.5.ffn_down_shexp.weight")) == GGUFLinearDispatch(
        _CANDIDATE, "raw"
    )
    assert _dispatch(_weight("layers.5.attn_gate.weight")) == GGUFLinearDispatch(
        _CANDIDATE, "raw"
    )
    assert _dispatch(_weight("layers.5.hc_ffn_down.weight")) == _PARENT


def test_unplaceable_weight_declines_under_a_scope(monkeypatch):
    """A scoped route must not silently cover a weight it cannot place."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate")
    assert _dispatch(None) == _PARENT
    assert _dispatch(_weight("token_embd.weight")) == _PARENT


def test_role_scope_does_not_bypass_the_geometry_checks(monkeypatch):
    """The scope narrows the route; it does not widen it."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate")
    weight = _weight("layers.5.attn_gate.weight")
    # rows <= 256 is the retained sub-256-row path.
    assert _dispatch(weight, rows=256) == _PARENT
    # in_features must be a multiple of 32 for the dense iu8 kernel.
    assert (
        _q8_iu8_wmma_dispatch(
            _PARENT, rows=512, in_features=100, out_features=320, weight=weight
        )
        == _PARENT
    )


def test_role_match_is_segment_wise_not_substring(monkeypatch):
    """`attn` must not select `attn_gate`: the match is on whole segments."""

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn")
    assert _dispatch(_weight("layers.5.attn_gate.weight")) == _PARENT
    assert _dispatch(_weight("layers.5.attn_qkv.weight")) == _PARENT
    # The real prefix still selects.
    assert _dispatch(_weight("layers.5.attn.weight")) == GGUFLinearDispatch(
        _CANDIDATE, "raw"
    )


def test_role_scope_is_inert_when_the_route_is_off(monkeypatch):
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate")
    assert _dispatch(_weight("layers.5.attn_gate.weight")) == _PARENT


def test_role_scope_is_visible_to_the_resolution_cache(monkeypatch):
    """Two roles of identical shape must not share a cache key.

    The route resolves once per key inside ``launch_gguf_linear``. If the scope
    is absent from the key, the first role's decision is replayed for every
    later role of the same shape and the scope becomes an artifact of
    cache-fill order. This is the same silent-failure mode that produced a
    false pass for the WMMA scope (bit-identical arms, KL exactly zero,
    ``measurement_valid: true``), so it is pinned rather than assumed.
    """

    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "attn_gate")
    in_scope = _q8_iu8_role_cache_key(_weight("layers.5.attn_gate.weight"))
    out_of_scope = _q8_iu8_role_cache_key(_weight("layers.5.ffn_gate_shexp.weight"))
    assert in_scope != out_of_scope
    # And the scope itself is part of the key, not just the resolved role.
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM_ROLES", "ffn_gate_shexp")
    assert _q8_iu8_role_cache_key(
        _weight("layers.5.attn_gate.weight")
    ) != in_scope
    # An unbound route cannot let a stale scope entry decide anything.
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_IU8_WMM", "0")
    assert _q8_iu8_role_cache_key(_weight("layers.5.attn_gate.weight")) != in_scope
