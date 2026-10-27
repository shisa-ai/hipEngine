"""K320 specialization follows the selected DPP parent, not an ambient flag."""
import pytest
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner


@pytest.mark.parametrize('value', ['0', '1', 'false'])
def test_retired_k320_environment_does_not_override_shape_selection(monkeypatch, value):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS', value)
    monkeypatch.setattr(runner, 'is_registered', lambda key: True)
    parent = KernelKey('hip_gfx1151', 'linear+gr_gated_mean', 'gguf_q8_0',
                      'coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact')
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        parent, rows=512, branches=4, low_rank=320).variant.endswith('k320_full_rows_f32_exact')
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        parent, rows=512, branches=4, low_rank=320, hidden=128) == parent
    assert runner._qwen4_exp_gr_k320_full_rows_key(
        parent, rows=513, branches=4, low_rank=320) == parent


def test_scoped_k320_parent_control_restores_on_error(monkeypatch):
    from scripts._qwen4exp_k320_control import k320_selection

    monkeypatch.setattr(runner, 'is_registered', lambda key: True)
    original = runner._qwen4_exp_gr_k320_full_rows_key
    parent = KernelKey('hip_gfx1151', 'linear+gr_gated_mean', 'gguf_q8_0',
                      'coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact')
    with pytest.raises(RuntimeError, match='arm failed'):
        with k320_selection(False):
            assert runner._qwen4_exp_gr_k320_full_rows_key(
                parent, rows=512, branches=4, low_rank=320) == parent
            raise RuntimeError('arm failed')
    assert runner._qwen4_exp_gr_k320_full_rows_key is original
    strict = KernelKey(parent.backend, parent.layer, parent.quant, 'coltile2_branch4_rowbatch4_f32_exact')
    assert original(strict, rows=512, branches=4, low_rank=320) == strict
