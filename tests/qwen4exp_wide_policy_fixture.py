"""Explicit dense-wide session control for selector and launch fixtures."""
from contextlib import ExitStack
import pytest
from hipengine.runtime import gguf_linear as linear

_MISSING = object()


@pytest.fixture
def wide_policy():
    """Bind requested axes until teardown without writing process keys."""
    with ExitStack() as stack:
        def bind(*, enabled=None, layers=_MISSING):
            prior = linear.resolve_q8_dense_wide_policy(object())
            policy = linear.Q8DenseWidePolicy(
                prior.enabled if enabled is None else enabled,
                prior.layers if layers is _MISSING else layers)
            stack.enter_context(linear.q8_dense_wide_policy_session(policy))
            return policy
        yield bind
