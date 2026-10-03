"""Capture must reserve the selected attention route on its own stream."""

from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_staged import staged_workspace_bytes
from hipengine.runtime import gemma4_decode_graph as graph


@pytest.mark.parametrize("position", [60, 120, 2501, 19999])
@pytest.mark.parametrize("variant", ["gemma4_staged", "gemma4_plain"])
def test_capture_reserves_selected_workspace_before_recording(monkeypatch, position, variant):
    geometries = (
        SimpleNamespace(head_dim=256, num_heads=32, num_kv_heads=16, sliding_window=1024),
        SimpleNamespace(head_dim=512, num_heads=16, num_kv_heads=2, sliding_window=None),
    )
    allocations = []
    capturing = False

    def allocate(nbytes, **kwargs):
        assert not capturing, "attention allocated inside graph capture"
        allocations.append(nbytes)
        return DeviceBuffer(ptr=1000 * len(allocations), nbytes=nbytes)

    def begin(*args, **kwargs):
        nonlocal capturing
        capturing = True

    def end(*args):
        nonlocal capturing
        capturing = False
        return 100

    runtime = SimpleNamespace(current_device=lambda: 0, stream_begin_capture=begin,
                              stream_end_capture=end, graph_instantiate=lambda _: 200)
    monkeypatch.setattr(attention, "malloc", allocate)
    monkeypatch.setattr(attention, "build_gemma4_attention", lambda **kwargs: object())
    monkeypatch.setattr(attention, "split_workspace_bytes", lambda t, h, d, k, s, **kw: t * h * k * 4)
    monkeypatch.setattr(attention, "flash_workspace_bytes", lambda t, h, d, s, **kw: t * h * d * s * 4)
    monkeypatch.setattr(graph, "get_hip_runtime", lambda: runtime)
    owner = attention.Gemma4AttentionScratch()
    warmed = owner.buffer(128, stream=0, runtime=runtime)
    runner = SimpleNamespace(position=position, prefill_attention_variants=(variant,),
                             weights=SimpleNamespace(layers=[None, None],
                                                     config=SimpleNamespace(geometry=lambda i: geometries[i])),
                             _scratches=[SimpleNamespace(attention=owner), SimpleNamespace(attention=owner)])
    session = object.__new__(graph.Gemma4DecodeGraphSession)
    session._runner_ref = lambda: runner
    session._stream, session._captures = 7, 0
    session._retire = lambda: None
    session._identify_appends = lambda *a, **kw: ()
    session._zero_pending_tail = lambda _: None
    bucket = graph._Bucket.for_position(position)

    def launch(*args, **kwargs):
        assert capturing
        for geometry in geometries:
            keys = bucket.end - session._frozen_key_begin(geometry, bucket)
            if variant == "gemma4_staged":
                need = staged_workspace_bytes(1, geometry.num_heads, keys)
            else:
                global_scores = attention._resident_attention_shared_bytes(
                    head_dim=geometry.head_dim, keys=keys) > 65536
                slices = 1 if global_scores else attention.decode_slices(keys, geometry.head_dim)
                if not global_scores and slices == 1:
                    continue
                need = geometry.num_heads * keys * 4
                if not global_scores and attention.flash_admits(
                    tokens=1, head_dim=geometry.head_dim, num_heads=geometry.num_heads,
                    num_kv_heads=geometry.num_kv_heads):
                    need = max(need, geometry.num_heads * geometry.head_dim * attention.flash_slices(keys) * 4)
            owner.buffer(need, stream=kwargs["stream"], runtime=runtime)

    runner._launch_block = launch
    session._capture(bucket, {}, {}, ("test",), token=1)
    assert session._captures == 1
    assert owner._current[0] is warmed
    if variant == "gemma4_staged" or position > 1024:
        assert owner._current[7] is not warmed
