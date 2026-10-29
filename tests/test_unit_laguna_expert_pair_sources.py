"""Exact resident ownership for cached and raw Laguna expert T16 pairs."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf_materialize import (
    LagunaGGUFRepackedCache,
    _materialize_q4_t16_dual_pair,
    _spec_for_tensor,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import interleave_gguf_q4_k_tile16_dual, repack_gguf_q4_k_tile16
from tests._laguna_synthetic import tensor_info


class _Runtime:
    def __init__(self, failure=None):
        self.buffers = {}
        self.failure = failure
        self.copies = []

    def malloc(self, nbytes):
        if self.failure == "allocation":
            raise MemoryError("synthetic allocation failure")
        self.buffers[0x1000] = bytearray(nbytes)
        return 0x1000

    def free(self, ptr):
        del self.buffers[ptr]

    def memcpy(self, dst, src, nbytes, kind):
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        if self.failure == "upload":
            raise RuntimeError("synthetic upload failure")
        self.buffers[dst][:nbytes] = ctypes.string_at(src, nbytes)
        self.copies.append(nbytes)


@pytest.fixture
def pair():
    rng = np.random.default_rng(35)
    raw = {
        name: rng.integers(0, 256, size=(2, 16, 144), dtype=np.uint8)
        for name in ("gate", "up")
    }
    specs = [
        _spec_for_tensor(
            "layers.1.ffn_" + name + "_exps",
            tensor_info(name, (2, 16, 256), GGMLQuantizationType.Q4_K),
        ) for name in raw
    ]
    tiles = {name: repack_gguf_q4_k_tile16(array).tiles for name, array in raw.items()}
    reads = []
    def tensor_data(name):
        reads.append(name)
        return raw[name]
    return SimpleNamespace(specs=specs, tiles=tiles, reader=SimpleNamespace(tensor_data=tensor_data), reads=reads)


def _cache(pair, hits, path):
    entries = {}
    for spec in pair.specs:
        if spec.source.name not in hits:
            continue
        tiles = pair.tiles[spec.source.name]
        filename = spec.source.name + ".npy"
        np.save(path / filename, tiles, allow_pickle=False)
        entries[spec.slot_path] = {
            "layout": spec.layout,
            "allocations": {"tiles": {
                "file": filename, "shape": list(tiles.shape),
                "numpy_dtype": tiles.dtype.str, "runtime_dtype": "int8",
                "source_dtype": "I8",
            }},
        }
    return LagunaGGUFRepackedCache(path=path, manifest={"entries": entries}, plan_fingerprint="synthetic")


@pytest.mark.parametrize("hits,source_kind", [
    ((), "gguf_dual_interleaved"),
    (("gate",), "repacked_cache+gguf_dual_interleaved"),
    (("up",), "repacked_cache+gguf_dual_interleaved"),
    (("gate", "up"), "repacked_cache_dual_interleaved"),
])
def test_expert_pair_sources_preserve_bytes_ownership_and_profile(pair, hits, source_kind, tmp_path):
    runtime = _Runtime()
    profiles = []
    gate, up = _materialize_q4_t16_dual_pair(
        *pair.specs, pair.reader, device=None, runtime=runtime,
        backend="hip_gfx1151", repacked_cache=_cache(pair, hits, tmp_path), profile=profiles.append,
    )
    try:
        expected = interleave_gguf_q4_k_tile16_dual(pair.tiles["gate"], pair.tiles["up"])
        assert tuple(gate.allocations) == ("tiles_dual",) and not up.allocations
        assert runtime.buffers[gate.allocation("tiles_dual").buffer.ptr] == expected.tobytes()
        assert gate.resident_nbytes + up.resident_nbytes == sum(spec.resident_nbytes for spec in pair.specs)
        assert pair.reads == [name for name in ("gate", "up") if name not in hits]
        assert runtime.copies == [expected.nbytes]
        assert [profile.source_kind for profile in profiles] == [source_kind, source_kind]
        assert [profile.resident_nbytes for profile in profiles] == [expected.nbytes, 0]
        assert [profile.allocation_count for profile in profiles] == [1, 0]
    finally:
        gate.free(runtime=runtime)
        up.free(runtime=runtime)
    assert not runtime.buffers


@pytest.mark.parametrize("failure", ["allocation", "upload", "profile_gate", "profile_up"])
def test_expert_pair_failure_releases_single_owner(pair, failure, tmp_path):
    runtime = _Runtime(failure)
    profiles = []
    def profile(row):
        profiles.append(row)
        if failure == "profile_gate" or (failure == "profile_up" and len(profiles) == 2):
            raise RuntimeError("synthetic profile failure")
    with pytest.raises((MemoryError, RuntimeError), match="synthetic"):
        _materialize_q4_t16_dual_pair(
            *pair.specs, pair.reader, device=None, runtime=runtime,
            backend="hip_gfx1151", repacked_cache=_cache(pair, ("gate", "up"), tmp_path), profile=profile,
        )
    assert not runtime.buffers
