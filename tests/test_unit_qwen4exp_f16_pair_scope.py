"""DP-2's loan is a request-local decision, never a process-wide switch."""

from contextvars import copy_context

import pytest

from hipengine.runtime import gguf_linear as linear
from hipengine.runtime.gguf_linear import (
    wide_f16_pair_diagnostic,
    wide_f16_pair_enabled,
    wide_f16_pair_session,
)

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR"


def test_the_unbound_default_is_the_parent_route():
    assert not wide_f16_pair_enabled()


def test_the_request_binds_and_restores_its_own_decision():
    with wide_f16_pair_session():
        assert wide_f16_pair_enabled()
    assert not wide_f16_pair_enabled()
    with wide_f16_pair_session(False):
        assert not wide_f16_pair_enabled()


def test_nested_scope_replaces_and_restores_the_enclosing_decision():
    with wide_f16_pair_session(True):
        with wide_f16_pair_session(False):
            assert not wide_f16_pair_enabled()
        assert wide_f16_pair_enabled()
        with wide_f16_pair_session(True):
            assert wide_f16_pair_enabled()
        assert wide_f16_pair_enabled()
    assert not wide_f16_pair_enabled()


def test_exception_inside_a_scope_restores_the_enclosing_decision():
    with wide_f16_pair_session(True):
        with pytest.raises(RuntimeError, match="boom"):
            with wide_f16_pair_session(False):
                assert not wide_f16_pair_enabled()
                raise RuntimeError("boom")
        assert wide_f16_pair_enabled()
    assert not wide_f16_pair_enabled()


def test_interleaved_contexts_keep_their_own_decision():
    """Two requests interleave: neither may read the other's lane."""

    def enabled_request():
        with wide_f16_pair_session(True):
            return wide_f16_pair_enabled()

    def strict_request():
        with wide_f16_pair_session(False):
            return wide_f16_pair_enabled()

    on, off = copy_context(), copy_context()
    assert on.run(enabled_request) is True
    assert off.run(strict_request) is False
    assert on.run(enabled_request) is True
    assert not wide_f16_pair_enabled()


def test_a_copied_context_cannot_widen_the_parent_request():
    with wide_f16_pair_session(False):
        child = copy_context()

        def widen():
            with wide_f16_pair_session(True):
                return wide_f16_pair_enabled()

        assert child.run(widen) is True
        assert not wide_f16_pair_enabled()


def test_diagnostic_control_outranks_the_request_decision():
    """A harness runs outside the request, so its control must win inside it."""

    with wide_f16_pair_session(True):
        with wide_f16_pair_diagnostic(False):
            assert not wide_f16_pair_enabled()
        assert wide_f16_pair_enabled()
        with wide_f16_pair_diagnostic(True):
            assert wide_f16_pair_enabled()
    with wide_f16_pair_session(False):
        with wide_f16_pair_diagnostic(True):
            assert wide_f16_pair_enabled()
        assert not wide_f16_pair_enabled()
    with wide_f16_pair_diagnostic(True):
        assert wide_f16_pair_enabled()
    assert not wide_f16_pair_enabled()


def test_the_retired_process_wide_switch_cannot_enable_the_loan(monkeypatch):
    """The switch is gone, not merely unused: no read may survive it."""

    assert not hasattr(linear, "WIDE_F16_PAIR_ENV")
    monkeypatch.setenv(RETIRED_ENV, "1")
    assert not wide_f16_pair_enabled()
    with wide_f16_pair_session(True):
        monkeypatch.setenv(RETIRED_ENV, "0")
        assert wide_f16_pair_enabled()
