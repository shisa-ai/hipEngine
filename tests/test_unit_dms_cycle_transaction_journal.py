"""A rejected compact-DMS cycle must undo its eviction and compaction exactly.

Task #10's acceptance: a cycle whose candidates are all rejected after an
eviction and compaction restores the exact pre-cycle payload, scales,
positions, and allocator ownership.

A compaction moves payload physically, so restoring the visible planes is not
enough on its own. The transaction therefore carries the per-head extents, the
range capacities, the extent pool's ownership and free ranges, and the resource
ledger's ownership and provisional reservations alongside the payload, and
``rollback`` puts all of them back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hipengine.kvcache.dms import (
    DMSCompactBackend,
    DMSCodecQualification,
    DMSRetrofitConfig,
)


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_from_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def _backend(
    *,
    codec: str = "int8_per_token_head",
    heads: int = 2,
    dim: int = 16,
    window: int = 4,
    slots: int = 128,
) -> DMSCompactBackend:
    retrofit = DMSRetrofitConfig(
        artifact_fingerprint="fixture:dms-cycle-journal",
        model_family="qwen35",
        num_layers=1,
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
        source_path="tests/fixtures/dms_cycle_journal",
    )
    qualification = None
    if codec == "int8_per_token_head":
        qualification = DMSCodecQualification(
            codec=codec,
            artifact_fingerprint=retrofit.artifact_fingerprint,
            kl_divergence=0,
            top1_agreement=1,
            no_dense_shadow=True,
            evidence_source="unit fixture; not model qualification",
        )
    return DMSCompactBackend(
        retrofit=retrofit,
        codec=codec,
        slots_per_layer=slots,
        max_request_rows=2,
        max_pack_rows=64,
        device_payloads=False,
        codec_qualification=qualification,
    )


def _admit(backend: DMSCompactBackend, request_id: int, tokens: int) -> None:
    request = SimpleNamespace(
        request_id=request_id,
        prompt_tokens=tuple(range(tokens)),
        max_new_tokens=0,
    )
    claims = backend.estimate(
        request, None, {"kind": "admission", "tokens": tokens, "max_new_tokens": 0}
    )
    backend.reserve(claims)


def _rows(rng: np.random.Generator, tokens: int, heads: int, dim: int) -> np.ndarray:
    """A decode-step row set: [rows, heads, dim]."""

    return _bf16_from_bits(_bf16_bits(rng.normal(size=(tokens, heads, dim)).astype(np.float32)))


def _pack_rows(rng: np.random.Generator, tokens: int, heads: int, dim: int) -> np.ndarray:
    """A packed prompt: [tokens, layers, heads, dim]."""

    return _bf16_from_bits(
        _bf16_bits(rng.normal(size=(tokens, 1, heads, dim)).astype(np.float32))
    )


def _observable(backend: DMSCompactBackend, request_id: int) -> dict[str, Any]:
    """Capture every piece of state a rejected cycle must put back."""

    state = backend.state_for_request(request_id)
    return {
        "live_counts": state.live_counts.copy(),
        "token_positions": state.token_positions.copy(),
        "evict_mask": state.evict_mask.copy(),
        "logical_tokens": int(state.logical_tokens),
        "range_capacity": state.range_capacity.copy(),
        "base_offsets": state.base_offsets.copy(),
        "extents": tuple(state.extents),
        "k_payload": {key: value.copy() for key, value in state.k_payload.items()},
        "v_payload": {key: value.copy() for key, value in state.v_payload.items()},
        "k_scales": {key: value.copy() for key, value in state.k_scales.items()},
        "v_scales": {key: value.copy() for key, value in state.v_scales.items()},
        "extent_pool": backend.extents.state_snapshot(),
        "ledger": backend.ledger.state_snapshot(),
        "evicted": int(backend.evicted_tokens),
        "counters": (
            int(backend.pack_calls),
            int(backend.decode_appends),
            int(backend.evicted_tokens),
            int(backend.released_provisional_slots),
        ),
    }


def _assert_plane_maps_equal(actual: dict, expected: dict, *, label: str) -> None:
    assert set(actual) == set(expected), label
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{label}[{key}]")


def _assert_ownership_equal(actual: dict, expected: dict, *, label: str) -> None:
    assert actual["owners"] == expected["owners"], label
    assert actual["free"] == expected["free"], label
    assert actual["allocation_failures"] == expected["allocation_failures"], label
    assert actual["high_water_slots"] == expected["high_water_slots"], label


def _assert_ledger_equal(actual: dict, expected: dict, *, label: str) -> None:
    assert actual["owners"] == expected["owners"], label
    assert actual["reservations"] == expected["reservations"], label
    assert actual["reservation_by_owner"] == expected["reservation_by_owner"], label
    assert actual["used"] == expected["used"], label
    assert actual["high_water"] == expected["high_water"], label
    assert actual["next_reservation_id"] == expected["next_reservation_id"], label


@pytest.mark.parametrize("codec", ["bf16", "int8_per_token_head"])
def test_rejected_cycle_restores_state_and_ownership(codec: str) -> None:
    backend = _backend(codec=codec)
    request_id = 0
    # The extent is sized by admission and filled exactly by the pack, so the
    # cycle's appended row is what forces an eviction and a compaction.
    packed = 9
    _admit(backend, request_id, 5)
    rng = np.random.default_rng(20260924)
    candidates = np.zeros((packed, 1, 2), dtype=bool)
    candidates[:5, :, :] = True
    backend.streaming_pack(
        request_id,
        _pack_rows(rng, packed, 2, 16),
        _pack_rows(rng, packed, 2, 16),
        candidates,
    )

    state = backend.state_for_request(request_id)
    before = _observable(backend, request_id)
    assert int(state.live_counts[0, 0]) == 5, "the pack must fill the extent exactly"

    operation = backend.begin_transaction(
        [SimpleNamespace(lease=state.lease)],
        None,
    )

    # The cycle writes one candidate row per head; the oldest retained token is
    # now outside the window, so the store evicts it and compacts payload in
    # place, which is what moves the plane contents.
    backend.append_decode(
        request_id,
        _rows(rng, 1, 2, 16),
        _rows(rng, 1, 2, 16),
        np.zeros((1, 2), dtype=bool),
        position=packed,
    )
    during = _observable(backend, request_id)

    assert not np.array_equal(during["token_positions"], before["token_positions"]), (
        "the cycle must actually evict and compact before the rollback proves anything"
    )
    assert int(during["evicted"]) > int(before["evicted"]), "the cycle must report an eviction"
    assert any(
        not np.array_equal(during["k_payload"][key], before["k_payload"][key])
        for key in before["k_payload"]
    ), "the compaction must physically move payload"

    backend.rollback(operation)
    after = _observable(backend, request_id)

    np.testing.assert_array_equal(after["live_counts"], before["live_counts"], err_msg="live counts")
    np.testing.assert_array_equal(
        after["token_positions"], before["token_positions"], err_msg="positions"
    )
    np.testing.assert_array_equal(after["evict_mask"], before["evict_mask"], err_msg="evict mask")
    np.testing.assert_array_equal(
        after["range_capacity"], before["range_capacity"], err_msg="range capacity"
    )
    np.testing.assert_array_equal(
        after["base_offsets"], before["base_offsets"], err_msg="base offsets"
    )
    assert after["logical_tokens"] == before["logical_tokens"]
    assert after["extents"] == before["extents"], "per-head extents"
    assert after["counters"] == before["counters"], "store counters"
    _assert_plane_maps_equal(after["k_payload"], before["k_payload"], label="k payload")
    _assert_plane_maps_equal(after["v_payload"], before["v_payload"], label="v payload")
    if codec == "int8_per_token_head":
        assert before["k_scales"], "the INT8 codec must own scale planes for this case to mean anything"
        _assert_plane_maps_equal(after["k_scales"], before["k_scales"], label="k scales")
        _assert_plane_maps_equal(after["v_scales"], before["v_scales"], label="v scales")
    _assert_ownership_equal(after["extent_pool"], before["extent_pool"], label="extent pool")
    _assert_ledger_equal(after["ledger"], before["ledger"], label="resource ledger")


def test_commit_keeps_the_cycle_and_releases_the_snapshot() -> None:
    """The accepted half of the transaction: commit leaves the cycle in place."""

    backend = _backend(codec="int8_per_token_head")
    request_id = 0
    packed = 9
    _admit(backend, request_id, 5)
    rng = np.random.default_rng(7)
    candidates = np.zeros((packed, 1, 2), dtype=bool)
    candidates[:5, :, :] = True
    backend.streaming_pack(
        request_id,
        _pack_rows(rng, packed, 2, 16),
        _pack_rows(rng, packed, 2, 16),
        candidates,
    )
    state = backend.state_for_request(request_id)
    operation = backend.begin_transaction([SimpleNamespace(lease=state.lease)], None)

    backend.append_decode(
        request_id,
        _rows(rng, 1, 2, 16),
        _rows(rng, 1, 2, 16),
        np.zeros((1, 2), dtype=bool),
        position=packed,
    )
    during = _observable(backend, request_id)
    delta = backend.commit(operation, None)

    assert delta.request_id == request_id
    after = _observable(backend, request_id)
    np.testing.assert_array_equal(after["token_positions"], during["token_positions"])
    np.testing.assert_array_equal(after["live_counts"], during["live_counts"])
    _assert_plane_maps_equal(after["k_payload"], during["k_payload"], label="committed payload")


def test_transaction_snapshot_restores_a_shrunk_allocation() -> None:
    """A shrink releases provisional slots and extents; rollback puts them back."""

    backend = _backend(codec="bf16")
    request_id = 0
    _admit(backend, request_id, 8)
    rng = np.random.default_rng(11)
    backend.streaming_pack(
        request_id,
        _pack_rows(rng, 8, 2, 16),
        _pack_rows(rng, 8, 2, 16),
        np.zeros((8, 1, 2), dtype=bool),
    )
    state = backend.state_for_request(request_id)
    before = _observable(backend, request_id)
    operation = backend.begin_transaction([SimpleNamespace(lease=state.lease)], None)

    # Re-pack fewer committed tokens, which is the store's shrink path: it
    # releases provisional slots and re-allocates the per-head extents.
    backend.streaming_pack(
        request_id,
        _pack_rows(rng, 3, 2, 16),
        _pack_rows(rng, 3, 2, 16),
        np.zeros((3, 1, 2), dtype=bool),
    )
    during = _observable(backend, request_id)

    backend.rollback(operation)
    after = _observable(backend, request_id)

    assert after["extent_pool"]["owners"] == before["extent_pool"]["owners"], "extent owners"
    assert after["extent_pool"]["free"] == before["extent_pool"]["free"], "free ranges"
    _assert_ledger_equal(after["ledger"], before["ledger"], label="resource ledger")
    assert after["extents"] == before["extents"], "per-head extents"
    np.testing.assert_array_equal(after["range_capacity"], before["range_capacity"])
    assert after["counters"] == before["counters"], "store counters"
    # The case is only meaningful if the shrink moved ownership in the first place.
    assert (
        during["extent_pool"]["free"] != before["extent_pool"]["free"]
        or during["ledger"]["reservations"] != before["ledger"]["reservations"]
        or during["extents"] != before["extents"]
    ), "the shrink path must move ownership for the rollback to prove anything"
