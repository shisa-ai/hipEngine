"""Layer-local finalization for the external DMS device prefill collector.

``finalize_layer(compact_layer_index, *, stream=0)`` turns one completely
captured compact layer's sidecar scores into that layer's eviction decisions,
uploads only that layer's decision plane, and caches the result. Later layers
may still be mid-capture; the global ``finalize`` keeps requiring full prompt
coverage and assembles ``[tokens, layers, heads]`` from the same per-layer
decisions.

The device is faked at the copy layer: every allocation owns real host bytes,
so projections and copies move actual data and a copy that names freed memory
fails loudly. The linear sidecar projector is replaced by the same NumPy
arithmetic, which keeps ``build_dms_exact_budget_eviction`` an independent CPU
oracle for the collector plumbing under test.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kvcache import dms_device as dms_device_module
from hipengine.kvcache.dms import build_dms_exact_budget_eviction
from hipengine.runtime import dms_prefill_cache
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import _ExternalDMSDevicePrefillCollector


class _FakeDevice:
    """Host-backed stand-in for the HIP allocator, memset, and copies."""

    def __init__(self) -> None:
        self.regions: dict[int, bytearray] = {}
        self.freed: set[int] = set()
        self.copies: list[tuple[str, int]] = []
        self.syncs: list[tuple[str, int | None]] = []
        self._next_ptr = 0x20000000

    # -- allocation ---------------------------------------------------
    def malloc(self, nbytes: int, *, runtime=None) -> DeviceBuffer:  # noqa: ARG002
        size = int(nbytes)
        ptr = self._next_ptr
        self._next_ptr += size + 0x1000
        self.regions[ptr] = bytearray(size)
        return DeviceBuffer(ptr, size)

    def free(self, buffer: DeviceBuffer, *, runtime=None) -> None:  # noqa: ARG002
        base = int(buffer.ptr)
        if base in self.freed:
            raise RuntimeError("fake device: double free")
        self.freed.add(base)

    # -- memory operations --------------------------------------------
    def _region(self, ptr: int, count: int) -> tuple[bytearray, int]:
        ptr = int(ptr)
        for base, data in self.regions.items():
            if base <= ptr and ptr + count <= base + len(data):
                if base in self.freed:
                    raise RuntimeError("fake device: operation names freed memory")
                return data, ptr - base
        raise RuntimeError(f"fake device: range {ptr:#x}+{count} is not allocated")

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        data, offset = self._region(ptr, int(nbytes))
        data[offset:offset + int(nbytes)] = bytes([value & 0xFF]) * int(nbytes)

    def copy_host_to_device(self, buffer, host_ptr, nbytes=None, *, runtime=None):  # noqa: ARG002
        count = buffer.nbytes if nbytes is None else int(nbytes)
        data, offset = self._region(buffer.ptr, count)
        data[offset:offset + count] = ctypes.string_at(int(host_ptr), count)
        self.copies.append(("h2d", count))

    def copy_device_to_host(self, host_ptr, buffer, nbytes=None, *, runtime=None):  # noqa: ARG002
        count = buffer.nbytes if nbytes is None else int(nbytes)
        data, offset = self._region(buffer.ptr, count)
        ctypes.memmove(int(host_ptr), bytes(data[offset:offset + count]), count)
        self.copies.append(("d2h", count))

    # -- raw access for the fake projector and for assertions ---------
    def write(self, ptr: int, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        data, offset = self._region(ptr, array.nbytes)
        data[offset:offset + array.nbytes] = array.tobytes()

    def read(self, ptr: int, nbytes: int, dtype=np.uint8) -> np.ndarray:
        data, offset = self._region(ptr, int(nbytes))
        return np.frombuffer(bytes(data[offset:offset + nbytes]), dtype=dtype).copy()

    # -- accounting ----------------------------------------------------
    def copy_bytes(self, kind: str) -> int:
        return sum(n for recorded, n in self.copies if recorded == kind)

    def reset_counters(self) -> None:
        self.copies.clear()


def _fake_scores(hidden, weights, bias, layer: int) -> np.ndarray:
    """The fake linear sidecar's scores.

    The fake projector and the expected-value oracle both call this, so the
    comparison isolates the collector's capture/finalize plumbing from the
    score source; ``build_dms_exact_budget_eviction`` stays independent.
    """

    return (hidden @ weights[layer].T + bias[layer]).astype(np.float32)


class _FakeSidecarProjector:
    """NumPy stand-in for ``DMSExternalLinearDeviceProjector``."""

    def __init__(self, device, weights, bias):
        self.device = device
        self.weights = weights
        self.bias = bias
        self.closed = 0
        self.calls: list[tuple[int, int, int]] = []

    def project(
        self,
        *,
        hidden_ptr: int,
        compact_layer_index: int,
        tokens: int,
        logits_ptr: int,
        evict_ptr: int,
        stream: int,
    ) -> None:
        if self.closed:
            raise RuntimeError("fake projector used after close")
        layer, rows = int(compact_layer_index), int(tokens)
        width = self.weights.shape[2]
        hidden = self.device.read(hidden_ptr, rows * width * 4, np.float32).reshape(rows, width)
        scores = _fake_scores(hidden, self.weights, self.bias, layer)
        self.calls.append((layer, rows, int(stream)))
        self.device.write(logits_ptr, scores)
        self.device.write(evict_ptr, (scores > 0).astype(np.uint8))

    def close(self) -> None:
        self.closed += 1


def _config(
    *,
    layers: int,
    kv_heads: int,
    hidden: int,
    window_size: int,
    target_compression_ratio: int,
    prefill_selection_mode: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_layers=layers,
        num_kv_heads=kv_heads,
        num_q_heads=4,
        head_dim=8,
        hidden_size=hidden,
        input_stage="attn_norm",
        physical_layer_ids=tuple(range(layers)),
        window_size=window_size,
        target_compression_ratio=target_compression_ratio,
        prefill_selection_mode=prefill_selection_mode,
    )


class _Harness:
    """A collector wired to fake device memory and a NumPy sidecar."""

    def __init__(
        self,
        monkeypatch,
        *,
        layers: int = 2,
        kv_heads: int = 2,
        tokens: int = 12,
        hidden: int = 6,
        window_size: int = 4,
        target_compression_ratio: int = 2,
        prefill_selection_mode: str = "exact_budget",
        decision_mode: str = "sidecar",
        diagnostic_injection=None,
        seed: int = 7,
        constant_scores: bool = False,
    ) -> None:
        self.device = _FakeDevice()
        self.config = _config(
            layers=layers,
            kv_heads=kv_heads,
            hidden=hidden,
            window_size=window_size,
            target_compression_ratio=target_compression_ratio,
            prefill_selection_mode=prefill_selection_mode,
        )
        self.tokens = int(tokens)
        self.layers = int(layers)
        self.kv_heads = int(kv_heads)
        rng = np.random.default_rng(seed)
        if constant_scores:
            # Every token/head ties, so exact-budget selection is decided
            # entirely by the oldest-position tie break.
            self.weights = np.zeros((layers, kv_heads, hidden), dtype=np.float32)
            self.bias = np.full((layers, kv_heads), 0.5, dtype=np.float32)
        else:
            self.weights = rng.standard_normal((layers, kv_heads, hidden)).astype(np.float32)
            self.bias = rng.standard_normal((layers, kv_heads)).astype(np.float32)
        self.hidden = rng.standard_normal((self.tokens, hidden)).astype(np.float32)
        self.runtime = SimpleNamespace(
            memset=self.device.memset,
            device_synchronize=lambda: self.device.syncs.append(("device", None)),
            stream_synchronize=lambda stream: self.device.syncs.append(("stream", int(stream))),
        )
        self.projector = _FakeSidecarProjector(self.device, self.weights, self.bias)
        monkeypatch.setattr(gguf_runner, "malloc", self.device.malloc)
        monkeypatch.setattr(gguf_runner, "free", self.device.free)
        monkeypatch.setattr(gguf_runner, "copy_host_to_device", self.device.copy_host_to_device)
        monkeypatch.setattr(gguf_runner, "copy_device_to_host", self.device.copy_device_to_host)
        # Prefix seeding writes through the cache module's own copy bindings.
        monkeypatch.setattr(dms_prefill_cache, "copy_host_to_device", self.device.copy_host_to_device)
        monkeypatch.setattr(dms_prefill_cache, "copy_device_to_host", self.device.copy_device_to_host)
        monkeypatch.setattr(
            dms_device_module,
            "DMSExternalLinearDeviceProjector",
            lambda source, *, backend: self.projector,
        )
        source = SimpleNamespace(
            config=self.config,
            compact_layer_index=lambda physical_layer: int(physical_layer),
        )
        self.collector = _ExternalDMSDevicePrefillCollector(
            source,
            token_count=self.tokens,
            backend="hip_gfx1100",
            runtime=self.runtime,
            decision_mode=decision_mode,
            diagnostic_injection=diagnostic_injection,
        )

    # -- capture -------------------------------------------------------
    def capture(self, layer: int, start: int = 0, rows: int | None = None, *, stream: int = 0) -> None:
        rows = self.tokens - start if rows is None else int(rows)
        chunk = np.ascontiguousarray(self.hidden[start:start + rows])
        buffer = self.device.malloc(chunk.nbytes)
        self.device.write(buffer.ptr, chunk)
        self.collector.capture_device_chunk(
            physical_layer_id=layer,
            compact_layer_index=layer,
            start=start,
            rows=rows,
            hidden_ptr=buffer.ptr,
            stream=stream,
        )

    def capture_all(self) -> None:
        for layer in range(self.layers):
            self.capture(layer)

    # -- independent CPU oracle ---------------------------------------
    def layer_logits(self, layer: int) -> np.ndarray:
        return _fake_scores(self.hidden, self.weights, self.bias, layer)

    def oracle(self) -> np.ndarray:
        """``build_dms_exact_budget_eviction`` over the whole prompt."""

        logits = np.stack(
            [self.layer_logits(layer) for layer in range(self.layers)], axis=1
        )
        return build_dms_exact_budget_eviction(
            logits,
            current_position=self.tokens - 1,
            window_size=self.config.window_size,
            target_compression_ratio=self.config.target_compression_ratio,
        )

    def decision_plane(self, layer: int) -> np.ndarray:
        """The layer's token-major device decision plane, read back."""

        nbytes = self.tokens * self.kv_heads
        raw = self.device.read(self.collector.decision_ptr(layer), nbytes)
        return raw.astype(np.bool_).reshape(self.tokens, self.kv_heads)

    def plane_bytes(self, layer: int) -> bytes:
        """Raw bytes of the layer's decision plane, for untouched-plane checks."""

        nbytes = self.tokens * self.kv_heads
        return self.device.read(self.collector.decision_ptr(layer), nbytes).tobytes()


def _assert_bool_decisions(decisions: np.ndarray, shape: tuple[int, ...]) -> None:
    assert decisions.shape == shape
    assert decisions.dtype == np.bool_


def test_finalize_layer_matches_exact_budget_oracle(monkeypatch) -> None:
    harness = _Harness(monkeypatch, kv_heads=3, tokens=13, window_size=5)
    # Two contiguous chunks for layer 0, one for layer 1.
    harness.capture(0, 0, 6)
    harness.capture(0, 6)
    harness.capture(1)
    other_plane = harness.plane_bytes(1)

    decisions = harness.collector.finalize_layer(0)

    _assert_bool_decisions(decisions, (13, 3))
    expected = harness.oracle()
    np.testing.assert_array_equal(decisions, expected[:, 0, :])
    # The layer's device plane carries the same decisions, token-major.
    np.testing.assert_array_equal(harness.decision_plane(0), expected[:, 0, :])
    # The later layer was never finalized, so its plane was not rewritten.
    assert harness.plane_bytes(1) == other_plane
    assert decisions.any(), "fixture must evict at least one token"


def test_finalize_layer_breaks_exact_budget_ties_by_oldest_position(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=12, window_size=4, constant_scores=True)
    harness.capture_all()

    decisions = harness.collector.finalize_layer(0)

    # 12 tokens, current position 11, window 4: positions 0..6 are eligible,
    # historical live = ceil(7 / 2) = 4, so the 3 oldest tied positions go.
    np.testing.assert_array_equal(decisions, harness.oracle()[:, 0, :])
    assert decisions[:3].all()
    assert not decisions[3:].any()


def test_finalize_layer_copies_only_its_own_planes(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=12, kv_heads=2)
    harness.capture_all()
    harness.device.reset_counters()
    other_plane = harness.plane_bytes(0)

    decisions = harness.collector.finalize_layer(1, stream=17)

    plane_bytes = 12 * 2
    assert harness.device.copy_bytes("d2h") == plane_bytes * 4, "one layer's scores"
    assert harness.device.copy_bytes("h2d") == plane_bytes, "one layer's decisions"
    assert harness.device.syncs == [("stream", 17)]
    np.testing.assert_array_equal(decisions, harness.oracle()[:, 1, :])
    # The other layer's plane was not written by this finalize.
    assert harness.plane_bytes(0) == other_plane

    harness.device.reset_counters()
    harness.collector.finalize_layer(0)
    assert harness.device.syncs[-1] == ("device", None), "stream 0 drains the device"
    np.testing.assert_array_equal(harness.decision_plane(0), harness.oracle()[:, 0, :])


def test_finalize_layer_allows_a_partially_captured_later_layer(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=12)
    harness.capture(0)
    harness.capture(1, 0, 4)

    decisions = harness.collector.finalize_layer(0)

    np.testing.assert_array_equal(decisions, harness.oracle()[:, 0, :])
    with pytest.raises(ValueError, match="coverage|incomplete|complete"):
        harness.collector.finalize_layer(1)
    with pytest.raises(ValueError, match="coverage|incomplete|complete"):
        harness.collector.finalize()


def test_finalize_layer_is_idempotent_and_global_finalize_reuses_it(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=12)
    harness.capture_all()
    first = harness.collector.finalize_layer(0)

    harness.device.reset_counters()
    harness.device.syncs.clear()
    again = harness.collector.finalize_layer(0)
    np.testing.assert_array_equal(again, first)
    assert harness.device.copies == [], "a cached layer is neither reread nor recomputed"
    assert harness.device.syncs == []

    harness.device.reset_counters()
    assembled = harness.collector.finalize()
    _assert_bool_decisions(assembled, (12, 2, 2))
    np.testing.assert_array_equal(assembled, harness.oracle())
    # Only the layer the cache did not already hold is computed.
    plane_bytes = 12 * 2
    assert harness.device.copy_bytes("d2h") == plane_bytes * 4
    assert harness.device.copy_bytes("h2d") == plane_bytes

    harness.device.reset_counters()
    np.testing.assert_array_equal(harness.collector.finalize(), assembled)
    assert harness.device.copies == []
    np.testing.assert_array_equal(harness.collector.finalize_layer(1), assembled[:, 1, :])
    assert harness.device.copies == []


def test_finalize_layer_includes_seeded_prefix_scores(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=10, window_size=3)
    prefix = 4
    # A high-score cached prefix wins eviction over every suffix token.
    prefix_scores = np.full((prefix, harness.kv_heads), 100.0, dtype=np.float32)
    predictor_logits = np.zeros((harness.layers, prefix, harness.kv_heads), np.float32)
    predictor_logits[0, :prefix] = prefix_scores
    cold = dms_prefill_cache.DMSColdPrefill(
        tuple(range(prefix)),
        tuple((np.zeros(1, np.uint8), np.zeros(1, np.uint8)) for _ in range(harness.layers)),
        predictor_logits,
        np.zeros((harness.layers, prefix, harness.kv_heads), np.uint8),
        np.zeros((prefix, 4), np.uint8),
        {},
    )
    seed = getattr(dms_prefill_cache, "seed_collector_prefix", None) or (
        dms_prefill_cache.seed_collector
    )
    seed(harness.collector, cold)
    harness.capture(0, prefix)

    decisions = harness.collector.finalize_layer(0)

    suffix_scores = _fake_scores(
        harness.hidden[prefix:], harness.weights, harness.bias, 0
    )
    timeline = np.concatenate([prefix_scores, suffix_scores], axis=0)[:, None, :]
    expected = build_dms_exact_budget_eviction(
        timeline,
        current_position=harness.tokens - 1,
        window_size=harness.config.window_size,
        target_compression_ratio=harness.config.target_compression_ratio,
    )
    np.testing.assert_array_equal(decisions, expected[:, 0, :])
    assert decisions[:prefix].any(), "seeded prefix scores must take part in selection"


@pytest.mark.parametrize(
    ("tokens", "evicts"),
    [(1, False), (2, False), (5, False), (6, False), (7, True), (9, True)],
)
def test_finalize_layer_short_prompts_off_the_window_boundary(
    monkeypatch, tokens: int, evicts: bool
) -> None:
    harness = _Harness(monkeypatch, tokens=tokens, window_size=4)
    harness.capture_all()

    decisions = harness.collector.finalize_layer(1)

    _assert_bool_decisions(decisions, (tokens, 2))
    np.testing.assert_array_equal(decisions, harness.oracle()[:, 1, :])
    assert bool(decisions.any()) is evicts


def test_finalize_layer_threshold_mode_reads_the_decision_plane(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=9, prefill_selection_mode="threshold")
    harness.capture_all()
    harness.device.reset_counters()

    decisions = harness.collector.finalize_layer(0)

    np.testing.assert_array_equal(decisions, harness.layer_logits(0) > 0)
    # Threshold mode publishes the projected flags; no scores are downloaded.
    assert harness.device.copy_bytes("d2h") == 9 * 2
    assert harness.device.copy_bytes("h2d") == 0
    np.testing.assert_array_equal(harness.collector.finalize(), np.stack(
        [harness.layer_logits(layer) > 0 for layer in range(harness.layers)], axis=1
    ))


def test_finalize_layer_no_evict_mode_publishes_all_false(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=7, decision_mode="no_evict")
    harness.capture_all()

    decisions = harness.collector.finalize_layer(1)

    _assert_bool_decisions(decisions, (7, 2))
    assert not decisions.any()
    assembled = harness.collector.finalize()
    _assert_bool_decisions(assembled, (7, 2, 2))
    assert not assembled.any()


def test_finalize_layer_diagnostic_mode_matches_sealed_injection(monkeypatch) -> None:
    # A sealed injection's mask is token-major [tokens, layers, heads].
    mask = np.zeros((6, 2, 2), dtype=bool)
    mask[(0, 3), 0, 1] = True
    mask[5, 1, :] = True
    harness = _Harness(
        monkeypatch,
        tokens=6,
        decision_mode="diagnostic",
        diagnostic_injection=SimpleNamespace(eviction_mask=mask),
    )
    harness.capture_all()

    np.testing.assert_array_equal(harness.collector.finalize_layer(0), mask[:, 0, :])
    np.testing.assert_array_equal(harness.collector.finalize_layer(1), mask[:, 1, :])
    np.testing.assert_array_equal(harness.collector.finalize(), mask)


def test_finalize_layer_rejects_out_of_range_layers(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=6)
    harness.capture_all()

    for layer in (-1, harness.layers):
        with pytest.raises((ValueError, IndexError)):
            harness.collector.finalize_layer(layer)


def test_capture_rejects_a_finalized_layer(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=6)
    harness.capture_all()
    harness.collector.finalize_layer(0)

    with pytest.raises(RuntimeError, match="collector|finali"):
        harness.capture(0, 0, 1)


def test_capture_rejects_released_inputs_and_closed_collector(monkeypatch) -> None:
    harness = _Harness(monkeypatch, tokens=6)
    harness.capture(0, 0, 5)
    harness.collector.release_decision_inputs()
    with pytest.raises(RuntimeError, match="collector|released"):
        harness.capture(0, 5, 1)

    closed = _Harness(monkeypatch, tokens=6)
    closed.collector.close()
    with pytest.raises(RuntimeError, match="collector|closed"):
        closed.capture(0, 0, 1)
