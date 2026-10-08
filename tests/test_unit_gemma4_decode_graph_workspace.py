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
    # The reservation must ask the launcher's own count rather than re-derive
    # one: the global packing's count is not the occupancy target, and a
    # capture that reserved the smaller one would have the launch allocate
    # inside the capture. A sentinel that no key count maps to makes the
    # difference observable.
    requested_slices = 7
    flash_requests: list[int] = []

    def request_slices(**kwargs):
        flash_requests.append(kwargs["keys"])
        return requested_slices

    monkeypatch.setattr(attention, "flash_request_slices", request_slices)
    flash_needs: list[int] = []
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
            need = 0
            if variant == "gemma4_staged":
                need = staged_workspace_bytes(1, geometry.num_heads, keys)
            # The incumbent sizing, mirroring the reservation's own expression:
            # the split weights plus, when the admission can pick flash, the
            # flash partials at the launcher's count. A staged capture falls
            # through to it as well, because the launch it records still goes
            # through the same native call site.
            slices = attention.decode_slices(keys, geometry.head_dim)
            if slices > 1:
                need = max(need, geometry.num_heads * keys * 4)
                if attention.flash_admits(
                    tokens=1, head_dim=geometry.head_dim, num_heads=geometry.num_heads,
                    num_kv_heads=geometry.num_kv_heads):
                    count = attention.flash_request_slices(
                        keys=keys, num_heads=geometry.num_heads,
                        num_kv_heads=geometry.num_kv_heads,
                        head_dim=geometry.head_dim)
                    if count:
                        flash_needs.append(geometry.num_heads * geometry.head_dim * count * 4)
                        need = max(need, flash_needs[-1])
            if need:
                owner.buffer(need, stream=kwargs["stream"], runtime=runtime)

    runner._launch_block = launch
    session._capture(bucket, {}, {}, ("test",), token=1)
    assert session._captures == 1
    assert owner._current[0] is warmed
    if position > 1024:
        # The route the capture can pick is the launcher's own count, so the
        # reservation is at least the partials that count needs -- and asking
        # the shared helper is what makes the two agree.
        assert flash_requests, "the capture did not ask for the launcher's slice count"
        assert requested_slices not in (attention.flash_slices(k) for k in flash_requests)
        assert owner._current[7].nbytes >= max(flash_needs)
    if variant == "gemma4_staged" or position > 1024:
        assert owner._current[7] is not warmed
