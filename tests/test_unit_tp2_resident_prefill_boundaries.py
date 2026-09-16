"""CPU-only boundary observer fixture and first-failure journal checks."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.tp2_resident_prefill_boundary_probe import Observer, BoundaryFailure
from scripts.tp2_xtx_tp1_eager_stage_probe import StageRecorder


@pytest.mark.parametrize('actual,expected', [
    (np.array([1., np.nan]), None), (np.array([np.inf]), None),
    (np.array([], dtype=np.float32), None),
    (np.array([0xA5, 0xA5], dtype=np.uint8), np.array([1, 2], dtype=np.uint8)),
])
def test_observer_saves_first_failure_and_stops(tmp_path, actual, expected):
    record = StageRecorder(tmp_path/'probe.json', {})
    observer = Observer(SimpleNamespace(runtime=None), record, tmp_path)
    with pytest.raises(BoundaryFailure):
        observer.check('embedding/H2D', lambda: observer.verify('embedding/H2D', actual, expected))
    fixture = np.load(tmp_path/'first-failure.npz')
    np.testing.assert_array_equal(fixture['actual'], actual)
    if expected is not None:
        np.testing.assert_array_equal(fixture['expected'], expected)
    with pytest.raises(BoundaryFailure):
        observer.check('layer-0', lambda: pytest.fail('launched after bad input'))
    assert json.loads((tmp_path/'probe.json').read_text())['first_bad_stage'] == 'embedding/H2D'


def test_observer_success_requires_exact_reference(tmp_path):
    observer = Observer(SimpleNamespace(runtime=None), StageRecorder(tmp_path/'p.json', {}), tmp_path)
    x = np.array([1., 2.], dtype=np.float32)
    result = observer.verify('copy', x, x.copy())
    assert result['exact_reference'] is True
    assert result['finite'] == 2
