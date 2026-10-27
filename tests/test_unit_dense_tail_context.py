"""Dense-tail selection is bound to an execution context, not the process."""
import pytest
from hipengine.runtime import gguf_linear as linear
from hipengine.kernels.registry import KernelKey

PARENT = 'dense_wide256_f16in_f32_f32_out'
PADDED = 'dense_wide256_padded_f16in_f32_f32_out'


def test_tail_scope_restores_parent_after_exception(monkeypatch):
    key = KernelKey('hip_gfx1151', 'linear', 'gguf_q8_0', PARENT)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT', PADDED)
    monkeypatch.setattr(linear, 'is_registered', lambda key: True)
    def selected():
        return linear._dense_wide_f16_tail_key(key, rows=1024, in_features=10240, out_features=320)
    assert selected() == key
    with pytest.raises(RuntimeError):
        with linear.dense_wide_tail_session(PADDED):
            assert selected().variant == PADDED
            with linear.dense_wide_tail_session(PARENT):
                assert selected() == key
            assert selected().variant == PADDED
            raise RuntimeError('scope failed')
    assert selected() == key


def test_diagnostic_parent_scope_restores_runner_after_failure():
    from types import SimpleNamespace
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    owner = SimpleNamespace(dense_wide_tail_variant=PADDED)
    with pytest.raises(RuntimeError):
        with route_scope('dense-wide-padded', 'before', owner):
            assert owner.dense_wide_tail_variant == PARENT
            raise RuntimeError('arm failed')
    assert owner.dense_wide_tail_variant == PADDED
