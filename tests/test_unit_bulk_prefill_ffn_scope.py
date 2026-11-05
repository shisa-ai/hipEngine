"""Execution-local FFN role does not leak across calls or threads."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as linear


def test_bulk_ffn_role_nested_error_restore():
    assert linear._bulk_prefill_ffn_enabled.get() is False
    with linear.bulk_prefill_ffn_session():
        assert linear._bulk_prefill_ffn_enabled.get() is True
        with pytest.raises(RuntimeError):
            with linear.bulk_prefill_ffn_session(False):
                assert linear._bulk_prefill_ffn_enabled.get() is False
                raise RuntimeError('injected')
        assert linear._bulk_prefill_ffn_enabled.get() is True
    assert linear._bulk_prefill_ffn_enabled.get() is False


@pytest.mark.parametrize('rows', (7, 28, 32, 33, 36, 57, 129))
def test_bulk_ffn_roles_preserve_dispatch_with_populated_caches(monkeypatch, rows):
    monkeypatch.setattr(linear, 'physical_exact_rowtiles_enabled', lambda: False)
    monkeypatch.setattr(linear, '_q4_t16_dual_silu_retile_enabled', lambda: False)
    monkeypatch.setattr(linear, '_ensure_linear_kernel_registered', lambda key: None)
    monkeypatch.setattr(linear, 'is_registered', lambda key: True)
    dispatch = SimpleNamespace(
        key=linear.KernelKey('hip_gfx1100', 'linear', 'gguf_q4_k_t16_v1', 'test'),
        abi='t16',
    )
    # These caches do not consume the role. Keep populated state across A/B/A;
    # the final fused decision must be resolved afresh, not read from them.
    with monkeypatch.context() as patch:
        patch.setattr(linear, '_DISPATCH_RESOLVE_CACHE', {'sentinel': dispatch})
        patch.setattr(linear, '_PAIR_DISPATCH_RESOLVE_CACHE', {'sentinel': dispatch})
        def resolve():
            return linear._q4_t16_dual_wmma_silu_dispatch(
                dispatch, dispatch, rows=rows, in_features=5120, out_features=17408,
            )
        control = resolve()
        with linear.bulk_prefill_ffn_session():
            assert resolve() == control
        assert resolve() == control
        assert (control is not None) is (rows >= 33)
        # Distinct floors are explicit policy, even though shipping values agree.
        patch.setattr(linear, '_Q4_T16_DUAL_WMMA_SILU_MIN_ROWS', 1)
        with linear.bulk_prefill_ffn_session():
            assert resolve() is not None
        assert resolve() == control


def test_resumable_segment_binds_ffn_role_and_restores_on_failure():
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    session = object.__new__(Qwen35GGUFResidentSession)
    events = []
    def execute(*args, **kwargs):
        assert linear._bulk_prefill_ffn_enabled.get() is True
        raise RuntimeError('injected segment failure')
    session._prefill_batch_native_layer_outer_segment_guarded = execute
    session._release_int8_prefill_oracles_after_call = lambda rows: events.append(rows)
    state = SimpleNamespace(sessions=(session,))
    with pytest.raises(RuntimeError, match='injected segment failure'):
        session.prefill_batch_native_layer_outer_segment(state)
    assert linear._bulk_prefill_ffn_enabled.get() is False
    assert events == [(session,)]


def test_bulk_ffn_role_thread_isolation():
    with ThreadPoolExecutor(max_workers=1) as pool:
        with linear.bulk_prefill_ffn_session():
            assert pool.submit(linear._bulk_prefill_ffn_enabled.get).result() is False
        assert pool.submit(linear._bulk_prefill_ffn_enabled.get).result() is False
