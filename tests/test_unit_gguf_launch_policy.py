from __future__ import annotations

from dataclasses import dataclass

from hipengine.runtime.gguf_launch_policy import LaunchDecisionOwner, launch_decision_session, own_launch_decision


@dataclass(frozen=True)
class Decision:
    registry_generation: int
    operation: str


def test_decisions_are_reused_only_inside_their_owner():
    first, second = LaunchDecisionOwner(), LaunchDecisionOwner()
    original = Decision(1, 'pair')
    equal = Decision(1, 'pair')
    assert own_launch_decision(original) is original
    with launch_decision_session(first):
        assert own_launch_decision(original) is original
        assert own_launch_decision(equal) is original
        with launch_decision_session(second):
            assert own_launch_decision(equal) is equal
        assert own_launch_decision(equal) is original
    assert own_launch_decision(equal) is equal


def test_generation_change_discards_old_decisions_and_capacity_is_bounded():
    owner = LaunchDecisionOwner(capacity=2)
    with launch_decision_session(owner):
        old = Decision(1, 'pair')
        assert own_launch_decision(old) is old
        new = Decision(2, 'pair')
        assert own_launch_decision(new) is new
        assert len(owner) == 1
        own_launch_decision(Decision(2, 'triple'))
        own_launch_decision(Decision(2, 'residual'))
        assert len(owner) == 2
        replacement = Decision(2, 'pair')
        assert own_launch_decision(replacement) is replacement


def test_dense_profile_binder_gives_each_generator_an_independent_owner():
    from types import SimpleNamespace
    from hipengine.generation.qwen35_dense_gfx1100_profiles import _strict_binder

    first, second = SimpleNamespace(), SimpleNamespace()
    _strict_binder(first, None)
    _strict_binder(second, None)
    assert isinstance(first.launch_decision_owner, LaunchDecisionOwner)
    assert first.launch_decision_owner is not second.launch_decision_owner


def test_async_decode_binds_profile_owner_and_restores_scope():
    from types import SimpleNamespace
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from contextlib import nullcontext

    session = object.__new__(Qwen35GGUFResidentSession)
    session.launch_decision_owner = LaunchDecisionOwner()
    session.use_gemv_decode = False
    session._position = 0
    session._refresh_dms_decode_owner_marker = lambda: None
    session._iq_dense_mmq_context = nullcontext
    decision = Decision(1, 'decode')
    def run(*args, **kwargs):
        assert own_launch_decision(decision) is decision
        assert own_launch_decision(Decision(1, 'decode')) is decision
        return 123
    session._run_token_to_final_hidden = run
    session._sample_device_from_hidden = lambda *args, **kwargs: None
    session.step_async_top1(7, stream=0)
    assert len(session.launch_decision_owner) == 1
    peer = Decision(1, 'decode')
    assert own_launch_decision(peer) is peer


def test_session_restores_owner_after_failure():
    owner = LaunchDecisionOwner()
    decision = Decision(1, 'pair')
    try:
        with launch_decision_session(owner):
            own_launch_decision(decision)
            raise RuntimeError('cancel')
    except RuntimeError:
        pass
    peer = Decision(1, 'pair')
    assert own_launch_decision(peer) is peer
