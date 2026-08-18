"""Strict RED gate for the HIP DMS streaming pack kernel (C2-7 U2).

The device pack scatters surviving prompt K/V rows straight into the compact
extents (count/rank/scatter, no retained dense sidecar). It must be bit-exact
against ``DMSCompactBackend.streaming_pack`` (the host parent implementation)
including extent geometry from ``estimate``/``reserve``. GPU cases skip
cleanly on no-ROCm runners.
"""

from __future__ import annotations

import ctypes
import types

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import clear_registry_for_tests, resolve
from hipengine.kvcache.dms import DMSCompactBackend, DMSRetrofitConfig


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_from_bits(bits: np.ndarray) -> np.ndarray:
    widened = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()
    return widened


def _backend(
    *,
    num_layers: int,
    heads: int,
    dim: int,
    window: int,
    slots: int,
) -> DMSCompactBackend:
    retrofit = DMSRetrofitConfig(
        artifact_fingerprint="fixture:dms-pack",
        model_family="qwen35",
        num_layers=num_layers,
        num_q_heads=heads * 4,
        num_kv_heads=heads,
        head_dim=dim,
        window_size=window,
        target_compression_ratio=2,
        alpha_scale=100.0,
        alpha_offset=5.0,
        borrowed_query_channel=dim - 1,
        corrected_mask=True,
        trained_checkpoint=True,
        evidence_source="unit fixture",
        source_path="tests/fixtures/dms_pack",
    )
    return DMSCompactBackend(
        retrofit=retrofit,
        codec="bf16",
        slots_per_layer=slots,
        max_request_rows=8,
        max_pack_rows=64,
    )


def _admit(backend: DMSCompactBackend, request_id: int, tokens: int) -> None:
    request = types.SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tuple(range(tokens)),
        max_new_tokens=0,
    )
    claims = backend.estimate(
        request, None, {"kind": "admission", "tokens": tokens, "max_new_tokens": 0}
    )
    backend.reserve(claims)


def _eviction_mask(tokens: int, heads: int, *, seed: int, window: int) -> np.ndarray:
    """Old tokens evicted per head, recent window kept; recent tokens random."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((tokens, heads), dtype=bool)
    if tokens > window + 1:
        mask[: tokens - (window + 1), :] = True
        jitter = rng.random((tokens - (window + 1), heads))
        mask[: tokens - (window + 1), :] = mask[: tokens - (window + 1), :] | (
            jitter > 0.5
        )
        # Un-evict a few old tokens to exercise the not-evicted retention path.
        if tokens - (window + 1) > 2:
            mask[1, :] = False
    return mask


def _case(
    *,
    rows_tokens: list[int],
    heads: int,
    dim: int,
    window: int,
    num_layers: int,
    seeds: list[int],
    row_starts: list[int],
) -> None:
    from hipengine.kernels.hip_gfx1100.attention import (
        build_dms_compact,
        dms_streaming_pack_bf16,
    )

    # per_head extent = min(tokens, ceil(tokens/2) + window) per admission;
    # size the layer slot pools to cover every row's reservation.
    slots = sum(rows_tokens) * 4 + window + 16
    backend = _backend(
        num_layers=num_layers, heads=heads, dim=dim, window=window, slots=slots
    )
    rng = np.random.default_rng(999)
    total_tokens = sum(rows_tokens)
    k_all = np.zeros((total_tokens, num_layers, heads, dim), dtype=np.float32)
    v_all = np.zeros_like(k_all)
    evict_all = np.zeros((total_tokens, num_layers, heads), dtype=bool)
    for r, (tokens, start) in enumerate(zip(rows_tokens, row_starts)):
        token_offset = sum(rows_tokens[:r])
        k_all[token_offset:token_offset + tokens] = (
            rng.standard_normal((tokens, num_layers, heads, dim))
        )
        v_all[token_offset:token_offset + tokens] = (
            rng.standard_normal((tokens, num_layers, heads, dim))
        )
        for layer in range(num_layers):
            evict_all[token_offset:token_offset + tokens, layer, :] = (
                _eviction_mask(
                    tokens, heads, seed=seeds[r] + layer, window=window
                )
            )
        _admit(backend, request_id=r, tokens=tokens)

    # Host parent implementation per row (per-request 0-based positions).
    for r in range(len(rows_tokens)):
        token_offset = sum(rows_tokens[:r])
        tokens = rows_tokens[r]
        backend.streaming_pack(
            request_id=r,
            k=k_all[token_offset:token_offset + tokens],
            v=v_all[token_offset:token_offset + tokens],
            eviction=evict_all[token_offset:token_offset + tokens],
        )

    # Device: per-layer contiguous [total_tokens, heads, dim] slices with the
    # rows concatenated in order. The device kernel consumes BF16 bits.
    k_layer = _bf16_bits(k_all[:, 0, :, :])
    v_layer = _bf16_bits(v_all[:, 0, :, :])
    evict_layer = np.ascontiguousarray(evict_all[:, 0, :].astype(np.uint8))

    states = [backend.state_for_request(r) for r in range(len(rows_tokens))]
    rows = len(rows_tokens)
    base = np.ascontiguousarray(
        np.stack([s.base_offsets[0] for s in states]), dtype=np.int32
    )
    capacity = np.ascontiguousarray(
        np.stack([s.range_capacity[0] for s in states]), dtype=np.int32
    )
    live_host = np.ascontiguousarray(
        np.stack([s.live_counts[0] for s in states]), dtype=np.int32
    )
    # Sanity: host pack must fit the reserved extents.
    assert (live_host <= capacity).all()

    total_capacity = int(capacity.sum())
    k_slot = np.zeros((total_capacity, dim), dtype=np.uint16)
    v_slot = np.zeros_like(k_slot)
    positions = np.full((total_capacity,), -1, dtype=np.int32)
    slot_evict = np.zeros(total_capacity, dtype=np.uint8)
    live_dev = np.zeros((rows, heads), dtype=np.int32)

    row_starts_arr = np.ascontiguousarray(row_starts, dtype=np.int32)
    row_tokens_arr = np.ascontiguousarray(rows_tokens, dtype=np.int32)

    buffers = {
        "k": malloc(k_layer.nbytes),
        "v": malloc(v_layer.nbytes),
        "evict": malloc(evict_layer.nbytes),
        "base": malloc(base.nbytes),
        "capacity": malloc(capacity.nbytes),
        "live": malloc(live_dev.nbytes),
        "row_starts": malloc(row_starts_arr.nbytes),
        "row_tokens": malloc(row_tokens_arr.nbytes),
        "k_slot": malloc(k_slot.nbytes),
        "v_slot": malloc(v_slot.nbytes),
        "positions": malloc(positions.nbytes),
        "slot_evict": malloc(slot_evict.nbytes),
    }
    uploads = {
        "k": k_layer,
        "v": v_layer,
        "evict": evict_layer,
        "base": base,
        "capacity": capacity,
        "row_starts": row_starts_arr,
        "row_tokens": row_tokens_arr,
        "positions": positions,
        "slot_evict": slot_evict,
    }
    try:
        for name, array in uploads.items():
            buf = buffers[name]
            copy_host_to_device(
                buf, host_array_ptr(np.ascontiguousarray(array)), array.nbytes
            )
        library = build_dms_compact(load=True)
        dms_streaming_pack_bf16(
            buffers["k"].ptr,
            buffers["v"].ptr,
            buffers["evict"].ptr,
            buffers["base"].ptr,
            buffers["capacity"].ptr,
            buffers["live"].ptr,
            buffers["row_starts"].ptr,
            buffers["row_tokens"].ptr,
            buffers["k_slot"].ptr,
            buffers["v_slot"].ptr,
            buffers["positions"].ptr,
            buffers["slot_evict"].ptr,
            rows,
            heads,
            dim,
            window,
            library=library,
        )
        readbacks = (
            ("k_slot", k_slot, buffers["k_slot"]),
            ("v_slot", v_slot, buffers["v_slot"]),
            ("positions", positions, buffers["positions"]),
            ("slot_evict", slot_evict, buffers["slot_evict"]),
            ("live", live_dev, buffers["live"]),
        )
        for name, array, buf in readbacks:
            copy_device_to_host(host_array_ptr(array), buf, array.nbytes)
    finally:
        for buf in buffers.values():
            free(buf)

    np.testing.assert_array_equal(
        live_dev, live_host, err_msg="live_counts differ from host parent"
    )
    for r, state in enumerate(states):
        for h in range(heads):
            start_slot = int(base[r, h])
            live = int(live_host[r, h])
            k_ref_bits = _bf16_bits(state.k_payload[(0, h)])
            v_ref_bits = _bf16_bits(state.v_payload[(0, h)])
            np.testing.assert_array_equal(
                k_slot[start_slot:start_slot + live],
                k_ref_bits,
                err_msg=f"row {r} head {h} K payload differs",
            )
            np.testing.assert_array_equal(
                v_slot[start_slot:start_slot + live],
                v_ref_bits,
                err_msg=f"row {r} head {h} V payload differs",
            )
            positions_ref = state.token_positions[0, h, :live] + row_starts[r]
            np.testing.assert_array_equal(
                positions[start_slot:start_slot + live],
                positions_ref,
                err_msg=f"row {r} head {h} token_positions differ",
            )
            evict_ref = state.evict_mask[0, h, :live].astype(np.uint8)
            np.testing.assert_array_equal(
                slot_evict[start_slot:start_slot + live],
                evict_ref,
                err_msg=f"row {r} head {h} evict_mask differs",
            )
            # Tails past live are untouched.
            np.testing.assert_array_equal(
                positions[start_slot + live:start_slot + int(capacity[r, h])],
                -1,
                err_msg=f"row {r} head {h} extent tail corrupted",
            )


def test_dms_streaming_pack_registers_and_build_plan() -> None:
    clear_registry_for_tests()
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_streaming_pack_bf16,
        plan_dms_compact_build,
        register_dms_compact_kernels,
    )

    register_dms_compact_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_streaming_pack",
            quant="bf16",
            variant="count_rank_scatter",
        )
        is dms_streaming_pack_bf16
    )
    artifact = plan_dms_compact_build(compiler_version="dms-test-version")
    assert artifact.family == "dms_compact"


def test_dms_streaming_pack_wrapper_validates_before_gpu_load() -> None:
    from hipengine.kernels.hip_gfx1100.attention import dms_streaming_pack_bf16

    with pytest.raises(ValueError, match="rows"):
        dms_streaming_pack_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 16
        )
    with pytest.raises(ValueError, match="window"):
        dms_streaming_pack_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, -1
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_streaming_pack_single_row_small_window_bit_exact() -> None:
    _case(
        rows_tokens=[7],
        heads=2,
        dim=16,
        window=1,
        num_layers=1,
        seeds=[1],
        row_starts=[0],
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_streaming_pack_batched_rows_bit_exact() -> None:
    _case(
        rows_tokens=[5, 9, 3],
        heads=4,
        dim=128,
        window=4,
        num_layers=2,
        seeds=[11, 22, 33],
        row_starts=[0, 57, 66],
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_streaming_pack_all_evicted_except_window_bit_exact() -> None:
    # Every old token evicted: only the retention window survives per head.
    _case(
        rows_tokens=[12],
        heads=2,
        dim=32,
        window=3,
        num_layers=1,
        seeds=[44],
        row_starts=[128],
    )
