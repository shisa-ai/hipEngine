"""Prefix eviction drains independent owners after a provider release error."""
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner


def test_provider_discard_failure_does_not_leak_target_prefix():
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    events = []
    def discard(tokens):
        events.append("provider")
        raise RuntimeError("provider release failed")
    runner._mtp2_adapter = SimpleNamespace(discard_prefix_checkpoint=discard)
    runner._prefix_cache = SimpleNamespace(evict_entry=lambda tokens: events.append("cache"))
    runner._kv_pool = SimpleNamespace(release_blocks=lambda blocks: events.append("blocks"))
    runner._prefix_state_snapshots = {(1, 2): SimpleNamespace(
        retained=True, block_ids=(9,),
        snapshot=SimpleNamespace(close=lambda: events.append("state")),
    )}
    runner._prefix_snapshot_evictions = 0
    runner._prefix_retained_evictions_by_reason = {}
    with pytest.raises(RuntimeError, match="provider release failed"):
        runner._evict_prefix_snapshot((1, 2), reason="test")
    assert events == ["provider", "cache", "blocks", "state"]
    assert runner._prefix_state_snapshots == {}
    assert runner._prefix_snapshot_evictions == 1
