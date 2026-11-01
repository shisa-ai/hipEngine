"""Journey generic WMMA baseline does not ride process snapshots."""
from types import SimpleNamespace
import os
import pytest
from scripts import qwen4exp_journey_localize as journey

ENV = 'HIPENGINE_GGUF_WMMA_PREFILL'


@pytest.mark.parametrize('published', [None, False, True])
def test_baseline_scope_restores_owner_without_process_write(monkeypatch, published):
    monkeypatch.setenv(ENV, '1')
    owner = SimpleNamespace(runner=SimpleNamespace(gguf_wmma_prefill_enabled=published))
    with pytest.raises(RuntimeError, match='restore'):
        with journey.generic_wmma_baseline_scope(owner):
            assert owner.gguf_wmma_prefill_enabled is False
            assert owner.runner.gguf_wmma_prefill_enabled is False
            assert os.environ[ENV] == '1'
            assert ENV not in journey.strict_environment(os.environ)
            raise RuntimeError('restore')
    assert not hasattr(owner, 'gguf_wmma_prefill_enabled')
    assert owner.runner.gguf_wmma_prefill_enabled is published


@pytest.mark.parametrize('arm', ['bound', 'all_flags_strict',
    'strict_prefill_production_decode', 'production_prefill_strict_decode'])
def test_split_baseline_is_off_in_both_phases(monkeypatch, arm):
    import numpy as np
    from hipengine.runtime import gguf_linear as linear
    monkeypatch.setenv(ENV, '1')
    seen = []
    result = SimpleNamespace(logits=np.array([1, 2], dtype=np.float32))
    runner = SimpleNamespace(gguf_wmma_prefill_enabled=True, reset=lambda: None)
    def observe(*args):
        with linear.wmma_prefill_session(runner.gguf_wmma_prefill_enabled):
            seen.append(linear.gguf_wmma_prefill_enabled())
        return result
    runner.prefill = observe
    runner.step = observe
    monkeypatch.setattr(journey, 'clear_arm_graphs', lambda owner: None)
    journey.split_trajectory(runner, [3], [5], {}, {}, arm=arm)
    assert seen == [False, False]
    assert runner.gguf_wmma_prefill_enabled is True
    assert os.environ[ENV] == '1'
