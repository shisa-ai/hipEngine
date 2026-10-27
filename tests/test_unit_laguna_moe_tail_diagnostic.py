from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from types import SimpleNamespace

import pytest

from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession


@dataclass(frozen=True)
class _Plan:
    moe_tail_next_rmsnorm: object
    add: object


def test_unfused_tail_scope_replaces_only_composite_and_restores_after_failure():
    from scripts.laguna_moe_tail_diagnostic import unfused_moe_tail

    original = _Plan(object(), object())
    owner = SimpleNamespace(kernel_plan=original)
    with pytest.raises(RuntimeError, match="profile failed"):
        with unfused_moe_tail(owner):
            assert owner.kernel_plan.moe_tail_next_rmsnorm is None
            assert owner.kernel_plan.add is original.add
            outer = owner.kernel_plan
            with unfused_moe_tail(owner):
                assert owner.kernel_plan.moe_tail_next_rmsnorm is None
            assert owner.kernel_plan is outer
            raise RuntimeError("profile failed")
    assert owner.kernel_plan is original
    assert "use_moe_tail_next_rmsnorm" not in signature(LagunaGGUFResidentSession).parameters


def test_profile_tail_scope_is_applied_only_to_requested_owner():
    from contextlib import ExitStack
    from scripts.laguna_long_context_profile import _configure_moe_tail_comparison

    owner = SimpleNamespace(kernel_plan=_Plan(object(), object()))
    other = SimpleNamespace(kernel_plan=_Plan(object(), object()))
    original = owner.kernel_plan
    with ExitStack() as stack:
        _configure_moe_tail_comparison(stack, owner, enabled=False)
        assert owner.kernel_plan.moe_tail_next_rmsnorm is None
        assert other.kernel_plan.moe_tail_next_rmsnorm is not None
    assert owner.kernel_plan is original
    with ExitStack() as stack:
        _configure_moe_tail_comparison(stack, owner, enabled=True)
        assert owner.kernel_plan is original
