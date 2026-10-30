"""Residual-write deferral preserves row policy and pending-write ownership."""
import pytest
from hipengine.runtime import qwen4_exp_runner as runtime


@pytest.mark.parametrize('legacy', ['0', 'false', '1'])
@pytest.mark.parametrize('rows', [1, 512, 1023, 1024, 1025, 3073])
def test_pending_write_ignores_retired_switch(monkeypatch, legacy, rows):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_WRITE_NORM', legacy)
    expected = (11,22,33,False) if rows >= 1024 else None
    assert runtime._qwen4_exp_gr_pending_write(rows,11,22,33) == expected
    handoff=[]
    assert runtime._qwen4_exp_gr_defer_write(handoff,rows,11,22,33) is (rows >= 1024)
    assert handoff == ([(11,22,33,True)] if rows >= 1024 else [])
    assert runtime._qwen4_exp_gr_defer_write(None,rows,11,22,33) is False


@pytest.mark.parametrize('bf16', [False,True])
def test_flush_keeps_payload_dtype(monkeypatch,bf16):
    calls=[]
    monkeypatch.setattr(runtime,'qwen4_exp_gr_write_bf16_bf16',lambda *a,**kw:calls.append(('bf16',a,kw)))
    monkeypatch.setattr(runtime,'qwen4_exp_gr_write_bf16_f32',lambda *a,**kw:calls.append(('f32',a,kw)))
    runtime._qwen4_exp_gr_flush_write((11,22,33,bf16),44,1024,4,256,stream=7,runtime='hip')
    assert calls == [('bf16' if bf16 else 'f32',(11,22,33,44,1024,4,256),dict(stream=7,runtime='hip'))]
