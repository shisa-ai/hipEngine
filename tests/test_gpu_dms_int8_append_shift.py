"""One expiring row shifts keepers; the whole compacted sequence must be right.

`append_layer`'s chunked keep-scan moves a retained row leftward when a row
before it was dropped. A thread's destination can then be another thread's
source, so the move stages each row before writing it. A test that checks only
the live count or the array length cannot see a duplicate-and-loss: the count is
unchanged, one position is duplicated and another disappears. This case asserts
every position, every quantized payload element, every scale, and every eviction
flag of every head, for a head that shifts and a head that does not.

The long parametrizations of `test_int8_pack_append_attention_and_restore` are
where the nondeterminism showed up (1, 1, 3 and 5 mismatched positions out of
8193 across four runs); this case pins the mechanism at a size that runs fast
enough to repeat.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kvcache.dms_device import DMSDevicePayloadStore
from hipengine.kernels.cpu_reference.dms import encode_dms_payload
from tests.test_gpu_dms_streaming_pack_hip import (
    _bf16_bits,
    _bf16_from_bits,
    _hip_available,
)

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime unavailable")

TOKENS = 600
DIM = 16
HEADS = 2
# The row must survive `pack_layer` and expire on the append, or the append's
# compaction never shifts anything. Packing keeps a slot when
# `tokens - 1 - t <= window`; the append keeps it when `tokens - t <= window`.
# At window 64 those disagree for exactly one position in this file, t = 535.
WINDOW = 64
EXPIRED_HEAD = 0
EXPIRED_ROW = 535


def _quantize(bits):
    return encode_dms_payload(_bf16_from_bits(bits), codec="int8_per_token_head")


def test_int8_append_with_exactly_one_expiry_keeps_the_whole_sequence():
    window = WINDOW  # the one evicted row survives packing and expires on append
    slots = (TOKENS + 8) * HEADS + 11
    retrofit = SimpleNamespace(
        num_layers=1,
        num_kv_heads=HEADS,
        num_q_heads=8,
        head_dim=DIM,
        window_size=window,
    )
    store = DMSDevicePayloadStore(
        retrofit=retrofit,
        slots_per_layer=slots,
        max_pack_rows=TOKENS,
        codec="int8_per_token_head",
    )
    rng = np.random.default_rng(2411)
    k = _bf16_bits(rng.normal(size=(TOKENS, HEADS, DIM)).astype(np.float32))
    v = _bf16_bits(rng.normal(size=(TOKENS, HEADS, DIM)).astype(np.float32))
    evict = np.zeros((TOKENS, HEADS), dtype=np.uint8)
    evict[EXPIRED_ROW, EXPIRED_HEAD] = 1
    base = np.array([3 + h * (TOKENS + 7) for h in range(HEADS)], dtype=np.int32)
    cap = np.full(HEADS, TOKENS + 3, dtype=np.int32)

    try:
        store.pack_layer(0, k, v, evict, base, cap)
        packed = store.layer_view(0)
        live = store.live_counts(0)
        # The packed prefix is the whole file for both heads; only the append
        # below drops a row, which is the shift under test.
        for h in range(HEADS):
            dst = slice(base[h], base[h] + TOKENS)
            np.testing.assert_array_equal(packed.positions[dst], np.arange(TOKENS))
            np.testing.assert_array_equal(live[h], TOKENS)

        kn = _bf16_bits(rng.normal(size=(HEADS, DIM)).astype(np.float32))
        vn = _bf16_bits(rng.normal(size=(HEADS, DIM)).astype(np.float32))
        store.append_layer(0, kn, vn, np.zeros(HEADS, np.uint8), TOKENS, base, cap, live)
        after = store.layer_view(0)
        new_live = store.live_counts(0)

        for h in range(HEADS):
            # Retention rule, host side: a packed slot survives unless it is
            # evicted and outside the window. Only the evicted row drops, and
            # only for the head that carries the eviction.
            slots_at = np.arange(TOKENS)
            retained = slots_at[(evict[slots_at, h] == 0) | (TOKENS - slots_at <= window)]
            if h == EXPIRED_HEAD:
                assert EXPIRED_ROW not in retained, "the case must exercise a shift"
                assert len(retained) == TOKENS - 1, len(retained)
            else:
                assert len(retained) == TOKENS, "the second head must not shift"

            span = len(retained) + 1
            assert int(new_live[h]) == span, (h, int(new_live[h]), span)
            dst = slice(base[h], base[h] + span)
            # Positions first: a duplicate-and-loss keeps the length and the live
            # count but moves one position, which is exactly the observed defect.
            np.testing.assert_array_equal(
                after.positions[dst], np.append(retained, TOKENS)
            )
            np.testing.assert_array_equal(
                after.evict[dst], np.append(evict[retained, h], np.uint8(0))
            )
            for bits, new_row, payload, scales in (
                (k, kn, after.k_bits, after.k_scales),
                (v, vn, after.v_bits, after.v_scales),
            ):
                expected_bits = np.concatenate([bits[retained, h], new_row[None, h]])
                expected_payload, expected_scales = _quantize(expected_bits)
                np.testing.assert_array_equal(payload[dst], expected_payload)
                np.testing.assert_array_equal(scales[dst], expected_scales)
    finally:
        store.close()


def test_int8_append_shift_is_repeatable_across_runs():
    """The same append must produce the same sequence on every run.

    The original defect was nondeterministic: the losing thread varied with
    scheduling, so repeated runs disagreed with each other. Repeating the append
    in one process is what makes a scheduling-dependent result visible.
    """

    window = 64
    tokens = 1025  # spans five chunks, so the shift crosses chunk boundaries
    slots = (tokens + 8) * HEADS + 11
    retrofit = SimpleNamespace(
        num_layers=1,
        num_kv_heads=HEADS,
        num_q_heads=8,
        head_dim=32,
        window_size=window,
    )
    rng = np.random.default_rng(99)
    k = _bf16_bits(rng.normal(size=(tokens, HEADS, 32)).astype(np.float32))
    v = _bf16_bits(rng.normal(size=(tokens, HEADS, 32)).astype(np.float32))
    evict = np.zeros((tokens, HEADS), dtype=np.uint8)
    evict[7, 1] = 1
    evict[tokens - 1, 0] = 1
    base = np.array([3 + h * (tokens + 7) for h in range(HEADS)], dtype=np.int32)
    cap = np.full(HEADS, tokens + 3, dtype=np.int32)
    kn = _bf16_bits(rng.normal(size=(HEADS, 32)).astype(np.float32))
    vn = _bf16_bits(rng.normal(size=(HEADS, 32)).astype(np.float32))

    signatures = []
    for _ in range(4):
        store = DMSDevicePayloadStore(
            retrofit=retrofit,
            slots_per_layer=slots,
            max_pack_rows=tokens,
            codec="int8_per_token_head",
        )
        try:
            store.pack_layer(0, k, v, evict, base, cap)
            live = store.live_counts(0)
            store.append_layer(0, kn, vn, np.zeros(HEADS, np.uint8), tokens, base, cap, live)
            view = store.layer_view(0)
            signatures.append(
                tuple(
                    (
                        int(store.live_counts(0)[h]),
                        view.positions[base[h]:base[h] + tokens].tobytes(),
                        view.k_bits[base[h]:base[h] + tokens].tobytes(),
                        view.v_bits[base[h]:base[h] + tokens].tobytes(),
                        view.k_scales[base[h]:base[h] + tokens].tobytes(),
                        view.evict[base[h]:base[h] + tokens].tobytes(),
                    )
                    for h in range(HEADS)
                )
            )
        finally:
            store.close()

    first = signatures[0]
    for index, signature in enumerate(signatures[1:], start=1):
        assert signature == first, f"run {index} disagreed with run 0"
