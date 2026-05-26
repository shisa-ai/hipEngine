"""KV-cache policy and live-span scaffolding."""

from hipengine.kvcache.policy import (
    FixedPagedKVPolicy,
    KVPolicy,
    KVReservation,
    KVTransaction,
    KV_SCALE_DTYPE_CHOICES,
    KV_SCALE_GRANULARITY_CHOICES,
    KV_STORAGE_AUTO,
    KV_STORAGE_CHOICES,
    ResolvedKVPolicy,
    resolve_kv_policy,
)
from hipengine.kvcache.pool import ChunkedKVPool, KVPoolAllocation, KVPoolChunk, KVPoolStats
from hipengine.kvcache.spans import KVLiveSpans, KVScaleMetadata

__all__ = [
    "ChunkedKVPool",
    "FixedPagedKVPolicy",
    "KVLiveSpans",
    "KVPolicy",
    "KVPoolAllocation",
    "KVPoolChunk",
    "KVPoolStats",
    "KVReservation",
    "KVScaleMetadata",
    "KVTransaction",
    "KV_SCALE_DTYPE_CHOICES",
    "KV_SCALE_GRANULARITY_CHOICES",
    "KV_STORAGE_AUTO",
    "KV_STORAGE_CHOICES",
    "ResolvedKVPolicy",
    "resolve_kv_policy",
]
