"""CPU protocol checks for the verifier-width diagnostic."""
import pytest

from scripts.qwen38_verifier_width_profile import width_schedule, summarize_rows


def test_width_schedule_is_balanced_and_reverses_order():
    widths = (1, 2, 3, 4, 6, 8)
    schedule = width_schedule(widths, 4)
    assert schedule[:6] == [(0, w) for w in widths]
    assert schedule[6:12] == [(1, w) for w in reversed(widths)]
    assert all(sum(width == w for _, width in schedule) == 4 for w in widths)


def test_summary_keeps_routes_and_uses_measured_wall():
    rows = [dict(context=64, width=2, wall_ms=t, graph=True, exact=True,
                 capture_ms=0.0, submit_ms=1.0, readback_ms=2.0) for t in (9, 11, 13)]
    summary = summarize_rows(rows)[0]
    assert summary['wall_ms_median'] == 11
    assert summary['wall_ms_min'] == 9
    assert summary['graph_calls'] == 3
    assert summary['exact_all']
    rows[-1]['exact'] = False
    assert not summarize_rows(rows)[0]['exact_all']


def test_saved_state_retains_live_destinations_and_owns_backup(monkeypatch):
    from types import SimpleNamespace
    from hipengine.core import memory
    from scripts.qwen38_verifier_width_profile import saved_linear_state
    state = memory.DeviceBuffer(10, 4)
    backup = memory.DeviceBuffer(20, 4)
    copy = memory.DeviceBuffer(30, 4)
    transfers, freed = [], []
    runtime = SimpleNamespace(memcpy=lambda *a: transfers.append(a), device_synchronize=lambda: None)
    session = SimpleNamespace(runtime=runtime, _linear_state_snapshot=lambda: [(state, backup)])
    monkeypatch.setattr(memory, 'malloc', lambda *a, **kw: copy)
    monkeypatch.setattr(memory, 'free', lambda b, **kw: freed.append(b))
    with saved_linear_state(session) as saved:
        assert saved == [(state, copy)]
        assert transfers[0][:3] == (30, 20, 4)
    assert freed == [copy]


@pytest.mark.parametrize('widths,reps', [((), 1), ((0,), 1), ((9,), 1), ((2, 2), 1), ((2,), 0)])
def test_bad_schedule_is_rejected(widths, reps):
    with pytest.raises(ValueError):
        width_schedule(widths, reps)
