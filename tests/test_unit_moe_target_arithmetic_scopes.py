"""Isolation contracts for MoE target arithmetic scopes."""

from contextvars import Context

import pytest

from hipengine.core.specdec2_scope import (
    moe_physical_c2_exact_linear_enabled,
    moe_physical_c2_exact_linear_session,
    moe_physical_c2_f32_residual_disabled,
    moe_physical_c2_numerics_session,
)


@pytest.mark.parametrize("scope,enabled", [
    (moe_physical_c2_numerics_session, moe_physical_c2_f32_residual_disabled),
    (moe_physical_c2_exact_linear_session, moe_physical_c2_exact_linear_enabled),
])
def test_moe_arithmetic_scope_restores_and_isolates_contexts(scope, enabled):
    assert not enabled()
    independent = Context()
    with scope(True):
        assert enabled()
        assert not independent.run(enabled)
        with pytest.raises(RuntimeError, match="target failed"):
            with scope(False):
                assert not enabled()
                raise RuntimeError("target failed")
        assert enabled()
        with pytest.raises(RuntimeError, match="nested target failed"):
            with scope(True):
                assert enabled()
                raise RuntimeError("nested target failed")
        assert enabled()
    assert not enabled()
    assert not independent.run(enabled)
