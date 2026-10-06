"""Complete native DP-2 replay charges ordinary dispatch and pair ownership."""
from types import SimpleNamespace

import pytest

from scripts.qwen4exp_shared_f16_cast_owner import (
    Consumer, measure_operation, native_pair_operation, replay_operation,
)


def infos():
    rt = object()
    a = Consumer(24, "attn_qkv", 1024, 2560, 10240, 100, 200, 300, 7, rt)
    b = Consumer(24, "attn_gate", 1024, 2560, 6144, 100, 200, 400, 7, rt)
    return (a, object()), (b, object())


def test_native_pair_calls_real_helper_with_both_weights_and_outputs():
    left, right = infos()
    calls = []
    native_pair_operation(lambda *a, **kw: calls.append((a, kw)), left, right, [500, 600])
    assert calls == [((left[1], right[1], 100, 500, 600, 1024, 2560, 10240, 6144),
                     dict(stream=7, runtime=left[0].runtime))]


@pytest.mark.parametrize("arm", ["two_casts", "one_cast"])
def test_native_mode_does_not_call_prebound_primitive_chain(arm):
    calls = []
    def forbidden():
        pytest.fail("native timing must not use prebound primitive")
    replay_operation(arm, True, forbidden, forbidden, forbidden, lambda: calls.append("native"))
    assert calls == ["native"]


@pytest.mark.parametrize("arm,expected", [
    ("two_casts", ["cast", "a", "cast", "b"]),
    ("one_cast", ["cast", "a", "b"]),
])
def test_primitive_mode_preserves_old_boundary(arm, expected):
    calls = []
    replay_operation(arm, False, lambda: calls.append("cast"), lambda: calls.append("a"),
                     lambda: calls.append("b"), lambda: pytest.fail("primitive uses native"))
    assert calls == expected


def test_unknown_arm_never_launches_native():
    with pytest.raises(ValueError, match="arm"):
        replay_operation("bad", True, None, None, None, lambda: pytest.fail("launched"))


@pytest.mark.parametrize("final_sync", [False, True])
def test_measurement_charges_scope_teardown_and_final_sync(monkeypatch, final_sync):
    from scripts import qwen4exp_shared_f16_cast_owner as screen
    events = []
    times = iter([1.0, 1.125])
    monkeypatch.setattr(screen.time, "perf_counter", lambda: (events.append("clock"), next(times))[1])
    rt = SimpleNamespace(
        device_synchronize=lambda: events.append("device_sync"),
        event_record=lambda e: events.append(("record", e)),
        event_synchronize=lambda e: events.append(("event_sync", e)),
        event_elapsed_time_ms=lambda a, b: 42.0,
    )
    def operation():
        events.extend(["scope_enter", "dispatch_a", "dispatch_b", "scope_exit"])
    assert measure_operation(rt, "start", "stop", operation,
                             final_device_sync=final_sync) == (125.0, 42.0)
    assert events == ["device_sync", "clock", ("record", "start"), "scope_enter",
                      "dispatch_a", "dispatch_b", "scope_exit", ("record", "stop"),
                      ("event_sync", "stop"), *(["device_sync"] if final_sync else []), "clock"]
