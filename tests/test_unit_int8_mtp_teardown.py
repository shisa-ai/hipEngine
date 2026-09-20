from dataclasses import MISSING, fields
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.tensor import Tensor
from hipengine.runtime import gguf_native_spec_cycle as cycle
from hipengine.runtime import qwen35_gguf_runner as runner


@pytest.mark.parametrize("slot_view", [False, True])
def test_session_teardown_closes_retained_prefix_arenas(slot_view):
    cls = runner.Qwen35GGUFResidentSession
    session = object.__new__(cls)
    for descriptor in fields(cls):
        if descriptor.default is not MISSING:
            setattr(session, descriptor.name, descriptor.default)
        elif descriptor.default_factory is not MISSING:
            setattr(session, descriptor.name, descriptor.default_factory())
    session.runtime = SimpleNamespace(device_synchronize=lambda: None)
    session.runner = None
    session._moe_graph = None
    closed = []
    session._gguf_prefix_snapshot_arena_pool = SimpleNamespace(close=lambda: closed.append(True))
    if slot_view:
        session._resident_batch_owner = object()
        session._close_resident_slot_view_buffers(runtime=session.runtime)
        session._close_resident_slot_view_buffers(runtime=session.runtime)
    else:
        session.close()
        session.close()
    assert closed == [True]


def test_dynamic_verifier_scratch_growth_belongs_to_session_root(monkeypatch):
    device = Device("hip", 0)
    base = SimpleNamespace(
        rows=8, max_positions=1024, blocks=4,
        positions_tensor=Tensor.from_handle(0x1000, (8,), DType.INT64, device),
        block_table=DeviceBuffer(0x2000, 128),
        positions=DeviceBuffer(0x1000, 64),
        context_counts=DeviceBuffer(0x3000, 64),
        full_attn_split_root=None,
    )
    monkeypatch.setattr(cycle, "replace", lambda obj, **kwargs: SimpleNamespace(**(vars(obj) | kwargs)))
    _, dynamic = cycle._dynamic_target_scratch(
        SimpleNamespace(_bulk_prefill_scratch=base), SimpleNamespace(),
        rows=4, context_limit=1023,
    )
    assert dynamic.full_attn_split_root is base
