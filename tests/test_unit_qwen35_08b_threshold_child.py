"""Threshold child applies diagnostics without leaking arguments or scope."""
import sys

import pytest

from hipengine.runtime import gguf_linear as linear
from scripts import qwen35_08b_threshold_child as child


@pytest.mark.parametrize('fail', [False, True])
def test_child_restores_scope_and_arguments(monkeypatch, fail):
    argv = ['child', '--diagnostic-dense-bulk', 'off', '--workloads', '513/1']
    monkeypatch.setattr(sys, 'argv', argv)

    def run():
        assert sys.argv == ['child', '--workloads', '513/1']
        assert linear._cumulative_prefill_override.get() == (False, True, True, True, True)
        if fail:
            raise RuntimeError('child failed')
        return 7

    monkeypatch.setattr(child.qwen35_readme_sweep, 'main', run)
    if fail:
        with pytest.raises(RuntimeError, match='child failed'):
            child.main()
    else:
        assert child.main() == 7
    assert sys.argv is argv
    assert linear._cumulative_prefill_override.get() == (None, None, None, None, None)
