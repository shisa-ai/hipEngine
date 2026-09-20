from types import SimpleNamespace

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.speculative.prefix_checkpoint import PrefixCheckpointStore


def _adapter(released):
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter._prefix_checkpoint_store_instance = PrefixCheckpointStore(
        capacity=4, release=released.append,
    )
    adapter._prompt_streaming_sinks = {}
    adapter._states = {}
    adapter._intents = {}
    adapter._prompt_hidden_rows = {}
    adapter._disabled_requests = set()
    adapter._batch_accept_workspace = None
    adapter._target_pad_token_scratch = None
    adapter._close_cycle_workspace = lambda: None
    adapter._release_accept_staging = lambda: None
    return adapter


def test_adapter_close_releases_provider_checkpoints_exactly_once():
    released = []
    adapter = _adapter(released)
    adapter._prefix_checkpoint_store_instance.put((1, 2), (4,), "payload")
    adapter.close()
    adapter.close()
    assert released == ["payload"]
    assert len(adapter._prefix_checkpoint_store_instance) == 0


def test_target_prefix_eviction_also_releases_provider_checkpoint():
    released = []
    adapter = _adapter(released)
    adapter._prefix_checkpoint_store_instance.put((1, 2), (4,), "provider")
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._mtp2_adapter = adapter
    runner._prefix_state_snapshots = {
        (1, 2): SimpleNamespace(
            retained=False,
            snapshot=SimpleNamespace(close=lambda: released.append("target")),
        ),
    }
    runner._prefix_snapshot_evictions = 0
    assert runner._evict_prefix_snapshot((1, 2), reason="pool_pressure")
    assert sorted(released) == ["provider", "target"]
    assert len(adapter._prefix_checkpoint_store_instance) == 0
    assert not runner._evict_prefix_snapshot((1, 2), reason="pool_pressure")
    adapter.close()
    assert sorted(released) == ["provider", "target"]


def test_mtp_prefix_hit_without_provider_checkpoint_reprefills():
    reasons = []
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner._prefix_cache = SimpleNamespace(match=lambda tokens: SimpleNamespace(
        hit=True, matched_token_count=512, matched_tokens=(1,) * 512, block_ids=(3, 4),
    ))
    runner._prefix_reuse_supported = lambda row: True
    runner._flush_all_packed_owners = lambda: None
    runner._prefix_phase_add = lambda *args: None
    runner._note_prefix_fallback = lambda row, reason: reasons.append(reason)
    runner._rows = {}
    runner._prefix_unusable_hits = 0
    runner._mtp2_adapter = _adapter([])
    row = SimpleNamespace(request_id=9, prompt_ids=(1,) * 513, mtp2_candidate_budget=3)
    assert runner._prefix_source_for(row) is None
    assert reasons == ["provider_checkpoint_unavailable"]
    assert row.mtp2_candidate_budget == 3
