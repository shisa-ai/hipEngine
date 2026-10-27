"""Common prefill scopes preserve entry-specific policies and unwind ownership."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as runner


@pytest.mark.parametrize("fail_at", (None, "q4", "f16", "wide", "q6", "iq", "body"))
def test_common_prefill_scopes_enter_in_order_and_unwind(monkeypatch, fail_at):
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.runner = object()
    events = []
    entered = []

    @contextmanager
    def scope(name):
        events.append(("enter", name))
        if fail_at == name:
            raise RuntimeError("injected scope failure")
        entered.append(name)
        try:
            yield
        finally:
            events.append(("exit", name))

    def eligible(owner):
        assert owner is session.runner
        return True

    def q4_scope(enabled):
        assert enabled is True
        return scope("q4")

    monkeypatch.setattr(runner, "_gguf_q4_t16_unequal_pair_prefill_applies", eligible)
    monkeypatch.setattr(runner, "q4_t16_unequal_pair_prefill_session", q4_scope)
    for name, method in (
        ("f16", "_prefill_f16_staging_context"),
        ("wide", "_wide_f16_activation_context"),
        ("q6", "_q6_integer_mmq_context"),
        ("iq", "_iq_dense_mmq_context"),
    ):
        setattr(session, method, lambda name=name: scope(name))

    def execute():
        with session._prefill_route_sessions():
            assert entered == ["q4", "f16", "wide", "q6", "iq"]
            events.append(("body", "body"))
            if fail_at == "body":
                raise RuntimeError("injected body failure")

    if fail_at is None:
        execute()
    else:
        with pytest.raises(RuntimeError, match="injected"):
            execute()
    assert [name for action, name in events if action == "exit"] == list(reversed(entered))
    assert [name for action, name in events if action == "enter"] == (
        ["q4", "f16", "wide", "q6", "iq"]
        if fail_at in (None, "body")
        else ["q4", "f16", "wide", "q6", "iq"][:len(entered) + 1]
    )


@pytest.mark.parametrize("fail", (False, True))
def test_ordinary_prefill_executes_inside_common_scope(monkeypatch, fail):
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=2)),
    )
    session.use_wmma_prefill = None
    session.use_gemv_decode = None
    session._bulk_prefill_scratch = object()
    events = []

    @contextmanager
    def common():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    session._prefill_route_sessions = common
    session._q8_mmq_prefill_context = nullcontext
    session._t16_f16_rocblas_prefill_context = lambda **kwargs: nullcontext()
    # Any duplicate common list bypasses the owner and fails instead of silently
    # using a second independent set of controls.
    monkeypatch.setattr(
        runner, "q4_t16_unequal_pair_prefill_session",
        lambda *args, **kwargs: pytest.fail("duplicate common scope"),
    )

    def execute(tokens, **kwargs):
        assert events == ["enter"]
        assert tuple(tokens) == (10, 11)
        events.append("execute")
        if fail:
            raise RuntimeError("injected prefill failure")
        return "result"

    session._run_bulk_prefill_and_sample = execute
    if fail:
        with pytest.raises(RuntimeError, match="injected prefill"):
            session.prefill([10, 11], use_bulk=True, return_logits=False)
    else:
        assert session.prefill([10, 11], use_bulk=True, return_logits=False) == "result"
    assert events == ["enter", "execute", "exit"]


@pytest.mark.parametrize("fail", (False, True))
def test_packed_prefill_common_scope_is_inside_guards_and_before_release(monkeypatch, fail):
    session = object.__new__(runner.Qwen35GGUFResidentSession)
    events = []
    session._packed_decode_state_dirty = True
    session._invalidate_live_packed_decode_graphs = lambda: events.append("invalidate")
    session.flush_packed_decode_state = lambda **kwargs: events.append("flush") or True
    session._release_int8_prefill_oracles_after_call = lambda sessions: events.append("release")

    @contextmanager
    def common():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    session._prefill_route_sessions = common
    monkeypatch.setattr(
        runner, "q4_t16_unequal_pair_prefill_session",
        lambda *args, **kwargs: pytest.fail("duplicate common scope"),
    )

    def execute(prompts, **kwargs):
        assert events == ["invalidate", "flush", "enter"]
        assert prompts == [[10, 11]]
        events.append("execute")
        if fail:
            raise RuntimeError("injected prefill failure")
        return ["result"]

    session._prefill_batch_native_impl = execute
    if fail:
        with pytest.raises(RuntimeError, match="injected prefill"):
            session.prefill_batch_native([[10, 11]])
    else:
        assert session.prefill_batch_native([[10, 11]]) == ["result"]
    assert events == ["invalidate", "flush", "enter", "execute", "exit", "release"]
