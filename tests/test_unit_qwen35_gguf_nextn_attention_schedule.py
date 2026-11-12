"""NextN graph attention schedules match eager selection at every depth."""
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_nextn as nextn
from hipengine.runtime import qwen35_gguf_runner as runner


def _executor():
    obj = object.__new__(nextn.Qwen35GGUFNextNExecutor)
    obj.backend = "hip_gfx1151"
    obj.weights = SimpleNamespace(config=SimpleNamespace(
        hidden_size=5120, block_count=64, head_count=24, head_count_kv=4,
        key_length=256, value_length=256,
    ))
    obj.runner = SimpleNamespace(weights=SimpleNamespace(config=SimpleNamespace(
        **{**vars(obj.weights.config), 'block_count': 1}
    )))
    obj.scratch = SimpleNamespace(max_positions=8192, block_size=256,
                                  full_attn_split_count=32)
    return obj


@pytest.mark.parametrize("position", [17, 767, 1000, 1021, 1023, 1024, 1249, 4093, 5831])
@pytest.mark.parametrize("budget", [1, 3, 7])
def test_schedule_matches_eager_at_each_depth(position, budget):
    obj = _executor()
    schedule = obj._proposal_attention_schedule(position, budget)
    assert len(schedule) == budget
    for depth, (cap, signature) in enumerate(schedule):
        context = position + depth + 1
        assert context <= cap <= obj.scratch.max_positions
        assert signature == obj._proposal_attention_signature(context)
        assert signature == obj._proposal_attention_signature(cap)


def test_schedule_captures_split_transition_per_depth():
    obj = _executor()
    schedule = obj._proposal_attention_schedule(1021, 3)
    assert [entry[1][0] for entry in schedule] == ["short_batch", "short_batch", "split"]
    assert [entry[0] for entry in schedule] == [1023, 1023, 1024]
    assert obj._proposal_attention_schedule(1300, 3) == obj._proposal_attention_schedule(1400, 3)
    assert obj._proposal_attention_schedule(1277, 3) != obj._proposal_attention_schedule(1278, 3)


def test_schedule_uses_adapted_single_layer_config(monkeypatch):
    obj = _executor()
    seen = []
    monkeypatch.setattr(runner, '_use_gguf_short_full_attention_split_decode',
                        lambda cfg, **kw: seen.append(cfg.block_count) or False)
    obj._proposal_attention_signature(17)
    assert seen == [1]


def test_schedule_tracks_policy_and_kernel_changes(monkeypatch):
    obj = _executor()
    before = obj._proposal_attention_schedule(1400, 3)
    monkeypatch.setattr(runner, "_gguf_full_attention_split_gate_bf16_fn", lambda *a, **kw: print)
    assert obj._proposal_attention_schedule(1400, 3) != before
    monkeypatch.setenv("HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT", "1537")
    schedule = obj._proposal_attention_schedule(1534, 3)
    assert [entry[1][0] for entry in schedule] == ["context", "context", "split"]
    assert [entry[0] for entry in schedule][:2] == [1536, 1536]


def test_capture_uses_each_depth_cap_and_restores_host_metadata(monkeypatch):
    import numpy as np
    from hipengine.core.memory import DeviceBuffer

    obj = _executor()
    position = np.array([1021], dtype=np.int64)
    context = np.array([1022], dtype=np.int64)
    scratch = SimpleNamespace(position_host=position, context_host=context,
                              position_buf=DeviceBuffer(100, 8), context_buf=DeviceBuffer(200, 8))
    obj.scratch.for_slot = lambda *a, **kw: scratch
    obj.weights.config.rms_norm_eps = 1e-6
    obj.weights.fallback = lambda name: SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=300)))
    obj.hidden_size = 8
    obj._proposal_target_hidden = DeviceBuffer(400, 16)
    obj._proposal_results = DeviceBuffer(500, 56)
    obj._proposal_history_hidden = DeviceBuffer(600, 112)
    obj._token_buf = DeviceBuffer(700, 8)
    obj._proposal_graph_runtime_library = object()
    obj.runtime = SimpleNamespace(
        stream_create=lambda: 1, device_synchronize=lambda: None,
        stream_begin_capture=lambda stream: None, stream_end_capture=lambda stream: 2,
        graph_instantiate=lambda graph: 3, event_create=lambda **kw: 4,
        memcpy_async=lambda *a: None,
    )
    calls = []
    def block(*args, **kwargs):
        calls.append(kwargs['attention_context_cap'])
        position[0] = args[2]
        context[0] = args[2] + 1
        return 800, 900
    obj._run_block = block
    obj._enqueue_exact_lm_head_top1 = lambda *a, **kw: True
    monkeypatch.setattr(nextn, 'gguf_rmsnorm_bf16_f32_weight', lambda *a, **kw: None)
    monkeypatch.setattr(nextn, 'copy_i32_to_i64', lambda *a, **kw: None)
    monkeypatch.setattr(nextn, 'advance_decode_position_i64', lambda *a, **kw: None)
    graph = obj._capture_exact_chain_graph(7, 0, 3, schedule=obj._proposal_attention_schedule(1021, 3))
    assert calls == [1023, 1023, 1024]
    assert (position[0], context[0]) == (1021, 1022)
    assert graph.graph_exec == 3


def test_schedule_rejects_out_of_capacity():
    obj = _executor()
    with pytest.raises(ValueError, match="capacity"):
        obj._proposal_attention_schedule(8190, 3)
