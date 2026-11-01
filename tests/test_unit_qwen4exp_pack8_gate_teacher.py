"""Pack8 width gate teacher is narrow even when production is default-wide."""
import os
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_layer2_profile_gate as gate
from hipengine.runtime import gguf_linear as linear


@pytest.mark.parametrize('initial', [None, '1', '0'])
@pytest.mark.parametrize('raises', [False, True])
def test_teacher_width_binding_restores_on_all_exits(monkeypatch, initial, raises):
    key = linear._PACK8_GEMV_WIDE_ENV
    if initial is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, initial)
    args = SimpleNamespace(candidate='pack8_gemv_wide_256')
    def capture():
        with gate.pack8_width_teacher_scope(args):
            assert linear._pack8_gemv_decode_threads(
                rows=1, out_features=320, variant='pack8_gemv_f32_f32_out') == 0
            if raises:
                raise RuntimeError('capture failed')
    if raises:
        with pytest.raises(RuntimeError, match='capture failed'):
            capture()
    else:
        capture()
    assert os.environ.get(key) == initial


def test_public_gate_entry_binds_teacher_before_constructing(monkeypatch):
    monkeypatch.setenv(linear._PACK8_GEMV_WIDE_ENV, '1')
    args = SimpleNamespace(candidate='pack8_gemv_wide_256')
    def run(bound, *, command):
        assert bound is args
        assert command == ['gate']
        assert not linear._pack8_gemv_wide_threads_enabled()
        return {'observed_teacher': '128'}
    monkeypatch.setattr(gate, '_run', run)
    assert gate.run(args, command=['gate']) == {'observed_teacher': '128'}
    assert os.environ[linear._PACK8_GEMV_WIDE_ENV] == '1'


def test_exact_thread_owner_is_nested_and_restored(monkeypatch):
    monkeypatch.setenv(linear._PACK8_GEMV_WIDE_ENV, '1')
    def width():
        return linear._pack8_gemv_decode_threads(
            rows=1, out_features=320, variant='pack8_gemv_f32_f32_out')
    assert width() == 256
    with pytest.raises(RuntimeError, match='scope failed'):
        with linear.pack8_gemv_exact_threads_session(True):
            assert width() == 0
            with linear.pack8_gemv_exact_threads_session(False):
                assert width() == 256
            assert width() == 0
            raise RuntimeError('scope failed')
    assert width() == 256


def test_other_candidate_teacher_keeps_existing_width(monkeypatch):
    monkeypatch.setenv(linear._PACK8_GEMV_WIDE_ENV, '1')
    with gate.pack8_width_teacher_scope(SimpleNamespace(candidate='bounded_kv_reset')):
        assert linear._pack8_gemv_wide_threads_enabled()
