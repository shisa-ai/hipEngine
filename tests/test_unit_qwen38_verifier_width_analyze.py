"""Reject missing or nested verifier profile ranges before trace attribution."""
import pytest

from scripts.qwen38_verifier_width_analyze import measured_windows, validate_coverage


def marker(label, start, end):
    return {'Function': label, 'Start_Timestamp': str(start), 'End_Timestamp': str(end)}


def test_only_measured_ranges_are_grouped():
    groups = measured_windows([marker('warm:c64:r2', 0, 5),
                               marker('verify:c64:r2', 5, 10),
                               marker('verify:c64:r2', 12, 17),
                               marker('ar_step:c64', 18, 20)])
    assert len(groups['verify:c64:r2']) == 2
    assert groups['ar_step:c64'][0] == {'start_ns': 18, 'end_ns': 20}


def test_coverage_rejects_partial_trace_or_failed_run():
    run = dict(protocol=dict(contexts=[64], widths=[2], repetitions=2),
               exact_all=True, rows=[{'exact': True}, {'exact': True}])
    windows = {'verify:c64:r2': [{}, {}], 'ar_step:c64': [{}, {}]}
    validate_coverage(run, windows)
    with pytest.raises(ValueError, match='coverage'):
        validate_coverage(run, {'verify:c64:r2': [{}, {}]})
    run['exact_all'] = False
    with pytest.raises(ValueError, match='non-exact'):
        validate_coverage(run, windows)


@pytest.mark.parametrize('rows', [[], [marker('verify:c64:r2', 5, 4)],
                                    [marker('verify:c64:r2', 0, 10), marker('verify:c64:r3', 1, 9)]])
def test_invalid_or_overlapping_windows_fail(rows):
    with pytest.raises(ValueError):
        measured_windows(rows)
