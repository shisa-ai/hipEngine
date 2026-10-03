"""Cross-history graph capacity, replay metadata and drafter ownership."""
import ctypes
from types import SimpleNamespace

import pytest

from hipengine.runtime import gemma4_decode_graph as graph
from hipengine.generation.gemma4_mtp import Gemma4MTPTextProvider


@pytest.mark.parametrize('capacity,position', [(100,90), (100,99), (16640,16639), (17,4)])
def test_graph_capture_extent_never_exceeds_cache_and_replay_resets_metadata(monkeypatch, capacity, position):
    staged = []
    runner = SimpleNamespace(uses_int8_kv=False, position=position, capacity=capacity,
                             weights=SimpleNamespace(config=SimpleNamespace(vocab_size=10)),
                             _last_logits_rows=3, _normalized_hidden_rows=3)
    # SimpleNamespace is not weak-referenceable; retain the fake through this callable.
    session = object.__new__(graph.Gemma4DecodeGraphSession)
    session._runner_ref = lambda: runner
    session._stream, session._exec, session._key = 2, 7, ('reused',)
    runner._stage_block_content = lambda tokens, **kw: (staged.append(kw) or ({}, {}))
    runner._collect_block = lambda *a, **kw: (runner._last_logits_rows, runner._normalized_hidden_rows)
    session._capture_key = lambda *a: ('reused',)
    session._retarget_appends = lambda pos: None
    monkeypatch.setattr(graph, 'get_hip_runtime', lambda: SimpleNamespace(graph_launch=lambda *a: None))
    assert session.step(1) == (1, 0)
    assert position < staged[0]['keys_extent'] <= capacity


def test_cached_mtp_drafter_rebinds_before_reading_hidden_or_shared_kv():
    old = SimpleNamespace(closed=True)
    new = SimpleNamespace(closed=False)
    drafter = SimpleNamespace(runner=old)
    provider = Gemma4MTPTextProvider(target_generator=SimpleNamespace(_ensure_runner=lambda: new),
                                    config=SimpleNamespace(candidate_budget=1), drafter=drafter)
    assert provider._ensure_drafter() is drafter
    assert drafter.runner is new


def test_iu8_map_readback_and_risk_reset_follow_the_producer_stream(monkeypatch):
    from hipengine.core import memory
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts
    from hipengine.kernels.hip_gfx1100.moe import group_scatter
    from hipengine.kernels.hip_gfx1100.quant import gguf_q5_k_q8_1_selected_prefill as leaf

    events = []
    runtime = SimpleNamespace(
        stream_synchronize=lambda stream: events.append(('sync', stream)),
        memset_async=lambda ptr, value, count, stream: events.append(('reset', stream)),
        stream_is_capturing=lambda stream: False,
    )
    buffers = {name: SimpleNamespace(ptr=(i+1)*1000, nbytes=size)
               for i, (name, size) in enumerate([
                   ('wmma_expert_start',16), ('wmma_tile_expert',8), ('wmma_total',8),
                   ('mmq_risk_count',4), ('mmq_risk_indices',128)])}
    scratch = SimpleNamespace(buffer=lambda name: buffers[name])
    weight = SimpleNamespace(allocation=lambda name: SimpleNamespace(buffer=SimpleNamespace(ptr=6000)))
    monkeypatch.setattr(leaf, 'build_gguf_q5_k_q8_1_selected_prefill', lambda **kw: object())
    monkeypatch.setattr(group_scatter, 'qwen35_moe_wmma_tile_map',
                        lambda *a, **kw: events.append(('map', kw['stream'])))

    def readback(dst, src, count, **kw):
        assert events[-1] == ('sync', 7)
        events.append(('read', 7))
        ctypes.c_int64.from_address(dst).value = 16

    monkeypatch.setattr(memory, 'copy_device_to_host', readback)
    monkeypatch.setattr(leaf, 'gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out',
                        lambda *a, **kw: events.append(('risk', kw['stream'])))
    monkeypatch.setattr(leaf, 'gguf_q5_k_selected_dual_sparse_exact_repair_bf16',
                        lambda *a, **kw: events.append(('repair', kw['stream'])))
    assert experts._gemma4_project_experts_gate_up_wmma_iu8(
        weight, 10, 20, SimpleNamespace(ptr=30), 1, 1, 256, 16,
        scratch=scratch, stream=7, runtime=runtime)
    assert events == [('map',7), ('sync',7), ('read',7), ('reset',7), ('risk',7), ('repair',7)]


def test_iu8_capture_launches_the_bound_without_a_readback(monkeypatch):
    """Under capture the tile plan's row count is never read back to the host.

    The MMQ-family leaves size their grid from a device readback and synchronize
    the stream to make it visible. Both are illegal inside a HIP graph capture
    (HIP 900), so a capturing stream must launch the routing-independent upper
    bound instead. The map still fills every unused ``tile_expert`` slot with
    ``-1``, which the leaf rejects, so the bounded grid is the same output.
    """
    from hipengine.core import memory
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts
    from hipengine.kernels.hip_gfx1100.moe import group_scatter
    from hipengine.kernels.hip_gfx1100.quant import gguf_q5_k_q8_1_selected_prefill as leaf

    events = []
    checked = []
    runtime = SimpleNamespace(
        stream_synchronize=lambda stream: events.append(('sync', stream)),
        memset_async=lambda ptr, value, count, stream: events.append(('reset', stream)),
        stream_is_capturing=lambda stream: (checked.append(stream) or True),
    )
    buffers = {name: SimpleNamespace(ptr=(i+1)*1000, nbytes=size)
               for i, (name, size) in enumerate([
                   ('wmma_expert_start',16), ('wmma_tile_expert',8), ('wmma_total',8),
                   ('mmq_risk_count',4), ('mmq_risk_indices',128)])}
    scratch = SimpleNamespace(buffer=lambda name: buffers[name])
    weight = SimpleNamespace(allocation=lambda name: SimpleNamespace(buffer=SimpleNamespace(ptr=6000)))
    monkeypatch.setattr(leaf, 'build_gguf_q5_k_q8_1_selected_prefill', lambda **kw: object())
    monkeypatch.setattr(group_scatter, 'qwen35_moe_wmma_tile_map',
                        lambda *a, **kw: events.append(('map', kw['stream'])))

    def forbidden_readback(*a, **kw):
        raise AssertionError('capture must not read the tile total back to the host')

    monkeypatch.setattr(memory, 'copy_device_to_host', forbidden_readback)
    monkeypatch.setattr(leaf, 'gguf_q5_k_selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out',
                        lambda *a, **kw: events.append(('risk', kw['stream'], a[-1])))
    monkeypatch.setattr(leaf, 'gguf_q5_k_selected_dual_sparse_exact_repair_bf16',
                        lambda *a, **kw: events.append(('repair', kw['stream'])))
    assert experts._gemma4_project_experts_gate_up_wmma_iu8(
        weight, 10, 20, SimpleNamespace(ptr=30), 1, 1, 256, 16,
        scratch=scratch, stream=7, runtime=runtime)
    # No synchronize and no readback: the capture only maps, resets and launches.
    assert checked == [7]
    assert ('sync', 7) not in events and ('read', 7) not in events
    # 1 compact row over 1 expert bounds to one 16-row tile.
    assert events == [('map',7), ('reset',7), ('risk',7,16), ('repair',7)]


def test_tile_rows_bound_rounds_and_guards(monkeypatch):
    """``_tile_rows_or_bound`` under capture: aligned bound, capacity guard.

    The MMQ32 leaf requires a 32-aligned row count, so the routing bound is
    rounded up before it is launched. The buffer capacity is the hard ceiling;
    a bound past it is a real overflow and still raises.
    """
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts

    runtime = SimpleNamespace(
        stream_synchronize=lambda stream: (_ for _ in ()).throw(
            AssertionError('capture must not synchronize')),
        stream_is_capturing=lambda stream: True,
    )
    bound = ((9 + 31 * 4 + 31) // 32) * 32
    assert bound % 32 == 0 and bound == 160
    assert experts._tile_rows_or_bound(
        tile_total_ptr=1, tile_capacity=8, tile_rows=32, bound_rows=bound,
        stream=3, runtime=runtime, route='MMQ') == 160
    with pytest.raises(RuntimeError, match='tile row bound 160 is outside capacity 128'):
        experts._tile_rows_or_bound(
            tile_total_ptr=1, tile_capacity=4, tile_rows=32, bound_rows=bound,
            stream=3, runtime=runtime, route='MMQ')


def test_tile_rows_readback_is_the_non_capture_path(monkeypatch):
    """Outside capture the exact tile total is still read back and validated."""
    from hipengine.core import memory
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts

    events = []
    runtime = SimpleNamespace(
        stream_synchronize=lambda stream: events.append(('sync', stream)),
        stream_is_capturing=lambda stream: False,
    )

    def readback(dst, src, count, **kw):
        assert events[-1] == ('sync', 5)
        events.append(('read', 5))
        ctypes.c_int64.from_address(dst).value = 64

    monkeypatch.setattr(memory, 'copy_device_to_host', readback)
    assert experts._tile_rows_or_bound(
        tile_total_ptr=1, tile_capacity=8, tile_rows=16, bound_rows=999,
        stream=5, runtime=runtime, route='T16') == 64
    assert events == [('sync', 5), ('read', 5)]

    def over_capacity(dst, src, count, **kw):
        ctypes.c_int64.from_address(dst).value = 200

    monkeypatch.setattr(memory, 'copy_device_to_host', over_capacity)
    with pytest.raises(RuntimeError, match='tile row count 200 is outside capacity 128'):
        experts._tile_rows_or_bound(
            tile_total_ptr=1, tile_capacity=8, tile_rows=16, bound_rows=999,
            stream=5, runtime=runtime, route='T16')
