"""CPU-only guards for the ordered X-5 scratch probe protocol."""
import ast
import inspect

from scripts import qwen4exp_decode_scratch_victim_probe as probe


class Runtime:
    def __init__(self):
        self.calls = []

    def device_synchronize(self):
        self.calls.append("device-sync")

    def event_record(self, event):
        self.calls.append(event)

    def event_synchronize(self, event):
        self.calls.append(("event-sync", event))

    def event_elapsed_time_ms(self, start, stop):
        self.calls.append(("elapsed", start, stop))
        return 0.125


def test_event_stop_precedes_host_synchronization():
    rt = Runtime()
    samples = probe.measure_window(
        rt, "start", "stop", lambda: rt.calls.append("launch"), 3
    )
    assert samples == [0.125] * 3
    assert rt.calls == [
        "device-sync", "start", "launch", "stop",
        ("event-sync", "stop"), ("elapsed", "start", "stop"),
    ] * 3


def test_dense_trigger_reaches_kernel_with_prepared_buffers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        probe.dense, "gguf_q8_0_dense_wide256_f32_f32_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    rt = Runtime()
    lib = object()
    probe.launch_dense_trigger(11, 22, 33, library=lib, runtime=rt)
    assert calls == [((11, 22, 33, 257, 256, 256), {"library": lib, "runtime": rt})]
    assert rt.calls == []  # The caller, not this asynchronous launcher, synchronizes.


def test_main_keeps_victim_launchers_async_and_executes_trigger():
    tree = ast.parse(inspect.getsource(probe.main))
    victims = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name in {"run_q4", "run_q8"}]
    assert len(victims) == 2
    for victim in victims:
        assert not any(isinstance(node, ast.Attribute) and node.attr == "device_synchronize"
                       for node in ast.walk(victim))
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "launch_dense_trigger" for node in ast.walk(tree))
    assert sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "measure_window" for node in ast.walk(tree)) == 2
