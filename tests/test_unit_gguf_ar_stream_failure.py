"""Compatibility stream launch failures propagate without serial replay."""
from types import SimpleNamespace

import pytest

from hipengine.generation.gguf_ar_batch_diagnostic import run_ar_serving_slots
from hipengine.generation.qwen35_gguf import Qwen35GGUFBringupGenerator


@pytest.mark.parametrize("phase", ["launch", "read"])
def test_stream_failure_never_replays_as_serial(phase):
    owner = object.__new__(Qwen35GGUFBringupGenerator)
    owner.ar_packed_decode = False
    owner.ar_stream_decode = True
    owner._flush_ar_packed_decode_owners = lambda slots: None
    owner._step_ar_serving_slot_serial = lambda *args: pytest.fail("must not replay launched work")
    owner._record_ar_serving_token = lambda *args: pytest.fail("must not publish failed group")
    calls = []
    runtime = SimpleNamespace(stream_create=lambda **kw: 7, stream_synchronize=lambda stream: None,
                              stream_destroy=lambda stream: None)
    def launch(*args, **kwargs):
        calls.append("launch")
        if phase == "launch":
            raise RuntimeError("launch failed")
    def read():
        raise RuntimeError("read failed")
    slots = [SimpleNamespace(done=False, decode_stream=0, prev_token=1, seq_position=3,
                             session=SimpleNamespace(runtime=runtime, step_async_top1=launch,
                                                     read_top1_sample=read)) for _ in range(2)]
    with pytest.raises(RuntimeError, match=phase):
        run_ar_serving_slots(owner, slots, object())
    assert calls
