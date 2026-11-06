"""Checkpoint budgets count owning device allocations, never host KV or views."""
from types import SimpleNamespace

import numpy as np

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime.qwen35_gguf_prefix_state import Qwen35GGUFPrefixStateSnapshot
from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNPrefixCheckpoint


def test_target_inventory_counts_padded_arena_once():
    snapshot = object.__new__(Qwen35GGUFPrefixStateSnapshot)
    snapshot.closed = False
    snapshot.arena = SimpleNamespace(owner=DeviceBuffer(4096, 8192))
    snapshot.layer_conv_states = (DeviceBuffer(4096, 64),)
    snapshot.layer_recurrent_states = (DeviceBuffer(8192, 128),)
    assert snapshot.device_allocations == ((4096, 8192),)
    snapshot.closed = True
    assert snapshot.device_allocations == ()


def test_provider_inventory_excludes_host_kv_and_deduplicates_views():
    snapshot = object.__new__(Qwen35GGUFNextNPrefixCheckpoint)
    snapshot.released = False
    snapshot.state_backups = (DeviceBuffer(4096, 64), DeviceBuffer(4096, 64))
    snapshot.boundary_hidden = DeviceBuffer(8192, 128)
    snapshot.key_buffers = (np.zeros(1000, dtype=np.uint8),)
    snapshot.value_buffers = ()
    assert snapshot.device_allocations == ((4096, 64), (8192, 128))
    snapshot.released = True
    assert snapshot.device_allocations == ()
