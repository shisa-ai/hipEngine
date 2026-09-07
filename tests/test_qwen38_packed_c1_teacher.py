"""Strict-teacher schedule coordinates and owned logit buffers."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from scripts.qwen38_packed_c1_teacher import capture_teacher, teacher_windows


class Session:
    def __init__(self):
        self.position = 99
        self.logits = np.zeros(8, dtype=np.float32)
        self.calls = []

    def reset(self):
        self.position = 0
        self.calls.append('reset')

    def result(self, token):
        self.logits[:] = 0
        self.logits[token] = 2
        return NS(logits=self.logits, token_id=token)

    def prefill(self, ids, *, return_logits):
        assert return_logits
        self.position = len(ids)
        self.calls.append(tuple(ids))
        return self.result(2)

    def step(self, token, *, return_logits):
        assert return_logits
        self.calls.append(token)
        self.position += 1
        return self.result((token + 1) % 8)


def test_teacher_owns_rows_and_preserves_prompt_decode_boundary():
    session = Session()
    record = capture_teacher(session, [0, 1], steps=5)
    assert session.calls == ['reset', (0, 1), 2, 3, 4, 5, 6]
    assert record['inputs'] == (2, 3, 4, 5, 6)
    assert record['logits'].argmax(axis=1).tolist() == [3, 4, 5, 6, 7]
    session.logits[:] = -10
    assert record['prefill_logits'].argmax() == 2
    assert record['logits'].argmax(axis=1).tolist() == [3, 4, 5, 6, 7]


@pytest.mark.parametrize('budget', range(1, 8))
def test_windows_cover_teacher_rows_once_including_clipped_tail(budget):
    record = capture_teacher(Session(), [0, 1], steps=24)
    windows = teacher_windows(record, budget=budget)
    assert sum(len(w['tokens']) for w in windows) == 24
    offset = 0
    for window in windows:
        assert window['position'] == 2 + offset
        assert window['prefix'] == (0, 1) + record['inputs'][:offset]
        assert window['tokens'] == record['inputs'][offset:offset + budget + 1]
        np.testing.assert_array_equal(window['logits'], record['logits'][offset:offset + budget + 1])
        offset += len(window['tokens'])


def test_teacher_rejects_cursor_drift():
    class Broken(Session):
        def step(self, token, **kwargs):
            result = super().step(token, **kwargs)
            self.position += 1
            return result
    with pytest.raises(ValueError, match='cursor'):
        capture_teacher(Broken(), [0, 1], steps=2)


@pytest.mark.parametrize('bad', [np.array([np.nan]), np.zeros((2, 8)), np.array([])])
def test_teacher_rejects_invalid_full_logits(bad):
    class Broken(Session):
        def result(self, token):
            return NS(logits=bad, token_id=token)
    with pytest.raises(ValueError, match='finite full-vocabulary'):
        capture_teacher(Broken(), [0, 1], steps=2)


@pytest.mark.parametrize('budget', [0, 8])
def test_windows_reject_invalid_budget(budget):
    with pytest.raises(ValueError):
        teacher_windows(capture_teacher(Session(), [0, 1], steps=2), budget=budget)
