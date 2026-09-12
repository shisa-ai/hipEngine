"""DMS compact pack must be layerwise against the dense BF16 pool.

Memory review target 1: the compact destination store allocates every
layer's payload lazily (first touch), and the dense BF16 chunk backing can
release one full-attention layer's planes after that layer has been packed,
so the eager compact-store/dense-pool overlap shrinks from the full compact
store to a single layer.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFKVChunkBacking

from tests.test_gpu_dms_streaming_pack_hip import _hip_available

_LAYERS = 4  # physical layers; two full-attention (1, 3), two linear (0, 2)


def _fake_layout():
    return SimpleNamespace(
        layer_storage_dtypes=(None, DType.BF16, None, DType.BF16),
        bf16_mirror_layer_indices=(),
        int8_kv_value_bf16=False,
    )


def _fake_backing(monkeypatch, pages: int = 4) -> tuple[Qwen35GGUFKVChunkBacking, list]:
    freed: list[DeviceBuffer] = []
    next_ptr = 0x20000000

    def fake_free(buffer, *, runtime=None):
        freed.append(buffer)

    monkeypatch.setattr(gguf_runner, "free", fake_free)
    layout = _fake_layout()
    layers = len(layout.layer_storage_dtypes)
    buffers: list[DeviceBuffer] = []
    planes: dict[str, list] = {name: [None] * layers for name in (
        "k", "v", "mk", "mv", "ks", "vs",
    )}

    def plane(name: str, layer: int, nbytes: int) -> DeviceBuffer:
        nonlocal next_ptr
        buffer = DeviceBuffer(ptr=next_ptr, nbytes=nbytes)
        next_ptr += ((nbytes + 255) // 256) * 256 + 256
        buffers.append(buffer)
        planes[name][layer] = buffer
        return buffer

    for layer, storage in enumerate(layout.layer_storage_dtypes):
        if storage is None:
            continue
        plane("k", layer, 4096)
        plane("v", layer, 4096)
    backing = Qwen35GGUFKVChunkBacking(
        layout=layout,
        start_block_id=0,
        pages=pages,
        full_key_caches=tuple(planes["k"]),
        full_value_caches=tuple(planes["v"]),
        full_bf16_mirror_key_caches=tuple(planes["mk"]),
        full_bf16_mirror_value_caches=tuple(planes["mv"]),
        full_k_scale_caches=tuple(planes["ks"]),
        full_v_scale_caches=tuple(planes["vs"]),
        full_kv_scale_metadata=(None,) * layers,
        buffers=tuple(buffers),
    )
    return backing, freed


def test_backing_releases_one_full_attention_layer(monkeypatch) -> None:
    backing, freed = _fake_backing(monkeypatch)
    total = backing.total_nbytes
    buffers_before = set(id(b) for b in backing.buffers)
    backing.release_full_attention_layer(1, runtime=SimpleNamespace())
    assert backing.full_key_caches[1] is None
    assert backing.full_value_caches[1] is None
    assert backing.full_key_caches[3] is not None, "other layers untouched"
    assert len(freed) == 2
    assert all(buffer.nbytes == 4096 for buffer in freed)
    assert set(id(b) for b in backing.buffers) == buffers_before - {
        id(b) for b in freed
    }
    assert backing.total_nbytes == total - 2 * 4096


def test_backing_release_rejects_linear_and_repeated_layers(monkeypatch) -> None:
    backing, freed = _fake_backing(monkeypatch)
    with pytest.raises(ValueError, match="linear-attention"):
        backing.release_full_attention_layer(0, runtime=SimpleNamespace())
    backing.release_full_attention_layer(3, runtime=SimpleNamespace())
    with pytest.raises(ValueError, match="already released"):
        backing.release_full_attention_layer(3, runtime=SimpleNamespace())
    assert len(freed) == 2


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")
def test_device_store_allocates_layers_lazily() -> None:
    from hipengine.kvcache.dms_device import DMSDevicePayloadStore

    retrofit = SimpleNamespace(
        num_layers=2, num_kv_heads=2, num_q_heads=8, head_dim=16, window_size=7
    )
    store = DMSDevicePayloadStore(
        retrofit=retrofit, slots_per_layer=64, max_pack_rows=16,
        codec="int8_per_token_head",
    )
    try:
        staging_bytes = store.resident_bytes
        assert staging_bytes > 0
        # Per-layer payloads are not allocated until first touch.
        store._ensure_layer(0)
        after_first = store.resident_bytes
        assert after_first > staging_bytes
        # Idempotent.
        store._ensure_layer(0)
        assert store.resident_bytes == after_first
        store._ensure_layer(1)
        assert store.resident_bytes > after_first
    finally:
        store.close()
