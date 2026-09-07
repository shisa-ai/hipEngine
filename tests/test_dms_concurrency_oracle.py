"""The concurrency oracle must reject finite but altered reference logits."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen38_dms_concurrency_probe as probe


@pytest.mark.parametrize('corrupt', [False, True])
def test_independent_c1_oracle_compares_logits(monkeypatch, corrupt):
    instances = []
    class Session:
        def __init__(self, *args, **kwargs):
            self.index = len(instances)
            instances.append(self)
            self._dms_backend = SimpleNamespace(observability_snapshot=lambda: {
                'capacity': {}, 'extent_pool': {'capacity_slots': 1, 'free_slots': 0, 'allocation_failures': 0},
                'ledger': {'active_reservations': 1}})
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def prefill(self, prompt, **kwargs):
            pass
        def step(self, token, **kwargs):
            logits = np.array([float(token), 1.0], dtype=np.float32)
            if corrupt and self.index >= 2:
                logits[1] += .01
            return SimpleNamespace(token_id=token+1, logits=logits)
    monkeypatch.setattr(probe, 'Qwen35GGUFResidentSession', Session)
    monkeypatch.setattr(probe, 'Qwen35GGUFFullStackRunner', lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(probe, 'memory_stats', lambda: {'current_allocated_bytes': 0})
    args = SimpleNamespace(model='fixture', metadata='fixture', backend='hip_gfx1100',
                           codec='int8_evaluation', verify_c1=True, cancel_after_steps=1)
    if corrupt:
        with pytest.raises(AssertionError, match='C1 logit mismatch'):
            probe._run_cycle(args, 0, [[1,2], [3,4]], [2,2], 3, '', [])
    else:
        result = probe._run_cycle(args, 0, [[1,2], [3,4]], [2,2], 3, '', [])
        assert len(result['decode']) == 4
        assert all(row['c1_logits_exact'] for row in result['decode'])
