"""Strict RED gate for the HIP DMS append/decode kernel (C2-7 U3).

Each decode step drops the single row at position ``p - window - 1`` when it
is evicted (steady-state parity with the host parent's full keep-recompute,
the same incremental invariant FastDMS uses), appends the new token row, and
fails closed (status=1, state untouched) when appending would overflow the
extent — where the host parent raises MemoryError without mutating state.
GPU cases skip cleanly on no-ROCm runners.
"""

from __future__ import annotations

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
from hipengine.kvcache.dms import DMSCompactBackend
from tests.test_dms_streaming_pack_hip import (
    _admit,
    _backend,
    _bf16_bits,
    _hip_available,
)


def test_dms_append_decode_registers_and_build_plan() -> None:
    clear_registry_for_tests()
    from hipengine.kernels.hip_gfx1100.attention import (
        dms_append_decode_bf16,
        plan_dms_compact_build,
        register_dms_compact_kernels,
    )

    register_dms_compact_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="dms_append_decode",
            quant="bf16",
            variant="compact_append_evict",
        )
        is dms_append_decode_bf16
    )
    artifact = plan_dms_compact_build(compiler_version="dms-test-version")
    assert artifact.family == "dms_compact"


def test_dms_append_decode_wrapper_validates_before_gpu_load() -> None:
    from hipengine.kernels.hip_gfx1100.attention import dms_append_decode_bf16

    with pytest.raises(ValueError, match="rows"):
        dms_append_decode_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 16
        )
    with pytest.raises(ValueError, match="window"):
        dms_append_decode_bf16(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, -1
        )


def _never_set_mask(tokens: int, heads: int, never: set[int]) -> np.ndarray:
    """Evict every prompt token except the never-evicted set."""
    mask = np.ones((tokens, heads), dtype=bool)
    for t in never:
        mask[t, :] = False
    return mask


def _build_case_data(
    prompts: list[int],
    heads: int,
    dim: int,
    window: int,
    never_sets: list[set[int]],
    steps: list[tuple[list[bool], ...]],
):
    """Prompt + decode-step data shared by the host oracle and device run.

    ``steps`` is a list of per-step tuples of per-head bool flags (new-row
    eviction decisions); the step position for row r is prompt_r + step_i.
    """
    prompt_data = []
    for r, (tokens, never) in enumerate(zip(prompts, never_sets)):
        k_rows = [
            np.random.default_rng(10_000 * (r + 1) + t).standard_normal(
                (1, heads, dim)
            )
            for t in range(tokens)
        ]
        v_rows = [
            np.random.default_rng(15_000 * (r + 1) + t).standard_normal(
                (1, heads, dim)
            )
            for t in range(tokens)
        ]
        evict = _never_set_mask(tokens, heads, never)
        prompt_data.append((np.stack(k_rows), np.stack(v_rows), evict))
    step_data: list[tuple] = []
    for index, flags in enumerate(steps):
        flags = np.asarray(flags, dtype=bool)
        rng = np.random.default_rng(70_000 + index)
        k_row = rng.standard_normal((heads, dim))
        v_row = rng.standard_normal((heads, dim))
        step_data.append((flags, k_row, v_row))
    return prompt_data, step_data


def _host_run(
    backend: DMSCompactBackend,
    rows: list[int],
    prompts: list[int],
    prompt_data: list[tuple],
    step_data: list[tuple],
) -> None:
    for r, (k_prompt, v_prompt, evict_prompt) in zip(rows, prompt_data):
        backend.streaming_pack(r, k_prompt, v_prompt, evict_prompt[:, None, :])
    for step_index, (flags, k_row, v_row) in enumerate(step_data):
        for r in rows:
            backend.append_decode(
                r,
                k_row[None],
                v_row[None],
                flags.reshape(1, -1),
                position=prompts[r] + step_index,
            )


def _device_run(
    backend: DMSCompactBackend,
    *,
    rows: list[int],
    prompts: list[int],
    prompt_data: list[tuple],
    step_data: list[tuple],
    heads: int,
    dim: int,
    window: int,
    row_starts: list[int],
):
    from hipengine.kernels.hip_gfx1100.attention import (
        build_dms_compact,
        dms_append_decode_bf16,
        dms_streaming_pack_bf16,
    )

    n_rows = len(rows)
    states = [backend.state_for_request(r) for r in rows]
    base = np.ascontiguousarray(
        np.stack([s.base_offsets[0] for s in states]), dtype=np.int32
    )
    capacity = np.ascontiguousarray(
        np.stack([s.range_capacity[0] for s in states]), dtype=np.int32
    )
    # Host streaming_pack shrinks provisional extents in place; released gaps
    # remain in the global slot address space, so size the raw device fixture to
    # the pool rather than to the sum of committed capacities.
    total_capacity = int(backend.slots_per_layer)
    k_slot = np.zeros((total_capacity, dim), dtype=np.uint16)
    v_slot = np.zeros_like(k_slot)
    positions = np.full((total_capacity,), -1, dtype=np.int32)
    slot_evict = np.zeros(total_capacity, dtype=np.uint8)
    live_dev = np.zeros((n_rows, heads), dtype=np.int32)
    status = np.zeros((n_rows, heads), dtype=np.int32)

    buffers: dict[str, object] = {}
    try:

        def upload(name: str, array: np.ndarray) -> None:
            array = np.ascontiguousarray(array)
            previous = buffers.get(name)
            if previous is not None:
                free(previous)
            buf = malloc(array.nbytes)
            buffers[name] = buf
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes)

        upload("base", base)
        upload("capacity", capacity)
        upload("positions", positions)
        upload("slot_evict", slot_evict)
        upload("live", live_dev)
        upload("k_slot", k_slot)
        upload("v_slot", v_slot)
        upload("status", np.zeros_like(status))
        library = build_dms_compact(load=True)

        total_tokens = sum(prompts)
        k_layer = np.zeros((total_tokens, heads, dim), dtype=np.uint16)
        v_layer = np.zeros_like(k_layer)
        evict_layer = np.zeros((total_tokens, heads), dtype=np.uint8)
        offset = 0
        for k_prompt, v_prompt, evict_prompt in prompt_data:
            tokens = int(k_prompt.shape[0])
            k_layer[offset:offset + tokens] = _bf16_bits(k_prompt[:, 0, :, :])
            v_layer[offset:offset + tokens] = _bf16_bits(v_prompt[:, 0, :, :])
            evict_layer[offset:offset + tokens] = evict_prompt.astype(np.uint8)
            offset += tokens
        upload("k", k_layer)
        upload("v", v_layer)
        upload("evict", evict_layer)
        upload("row_starts", np.ascontiguousarray(row_starts, dtype=np.int32))
        upload("row_tokens", np.ascontiguousarray(prompts, dtype=np.int32))

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
            n_rows,
            heads,
            dim,
            window,
            library=library,
        )
        for step_index, (flags, k_row, v_row) in enumerate(step_data):
            upload("k_new", _bf16_bits(k_row))
            upload("v_new", _bf16_bits(v_row))
            upload("evict_new", np.asarray(flags, dtype=np.uint8))
            upload(
                "row_positions",
                np.ascontiguousarray(
                    [row_start + prompt + step_index for row_start, prompt in zip(row_starts, prompts)],
                    dtype=np.int32,
                ),
            )
            dms_append_decode_bf16(
                buffers["k_new"].ptr,
                buffers["v_new"].ptr,
                buffers["evict_new"].ptr,
                buffers["row_positions"].ptr,
                buffers["base"].ptr,
                buffers["capacity"].ptr,
                buffers["live"].ptr,
                buffers["k_slot"].ptr,
                buffers["v_slot"].ptr,
                buffers["positions"].ptr,
                buffers["slot_evict"].ptr,
                buffers["status"].ptr,
                n_rows,
                heads,
                dim,
                window,
                library=library,
            )
        for name, array in (
            ("k_slot", k_slot),
            ("v_slot", v_slot),
            ("positions", positions),
            ("slot_evict", slot_evict),
            ("live", live_dev),
            ("status", status),
        ):
            copy_device_to_host(host_array_ptr(array), buffers[name], array.nbytes)
    finally:
        for buf in buffers.values():
            free(buf)
    return k_slot, v_slot, positions, slot_evict, live_dev, status


def _case(
    *,
    prompts: list[int],
    heads: int,
    dim: int,
    window: int,
    never_sets: list[set[int]],
    steps: list[tuple[list[bool], ...]],
    row_starts: list[int],
) -> None:
    rows = list(range(len(prompts)))
    slots = sum(prompts) * 4 + window + 16
    backend = _backend(
        num_layers=1, heads=heads, dim=dim, window=window, slots=slots
    )
    for r, tokens in zip(rows, prompts):
        _admit(backend, request_id=r, tokens=tokens)
    prompt_data, step_data = _build_case_data(
        prompts, heads, dim, window, never_sets, steps
    )

    _host_run(backend, rows, prompts, prompt_data, step_data)
    (
        k_slot,
        v_slot,
        positions,
        slot_evict,
        live_dev,
        status,
    ) = _device_run(
        backend,
        rows=rows,
        prompts=prompts,
        prompt_data=prompt_data,
        step_data=step_data,
        heads=heads,
        dim=dim,
        window=window,
        row_starts=row_starts,
    )
    assert (status == 0).all(), f"no-evictable-slot status: {status}"

    states = [backend.state_for_request(r) for r in rows]
    for r, state in enumerate(states):
        for h in range(heads):
            start_slot = int(state.base_offsets[0, h])
            live = int(state.live_counts[0, h])
            k_ref_bits = _bf16_bits(state.k_payload[(0, h)])
            v_ref_bits = _bf16_bits(state.v_payload[(0, h)])
            np.testing.assert_array_equal(
                live_dev[r, h], live, err_msg=f"row {r} head {h} live count"
            )
            np.testing.assert_array_equal(
                k_slot[start_slot:start_slot + live],
                k_ref_bits,
                err_msg=f"row {r} head {h} K payload",
            )
            np.testing.assert_array_equal(
                v_slot[start_slot:start_slot + live],
                v_ref_bits,
                err_msg=f"row {r} head {h} V payload",
            )
            np.testing.assert_array_equal(
                positions[start_slot:start_slot + live],
                state.token_positions[0, h, :live] + row_starts[r],
                err_msg=f"row {r} head {h} token_positions",
            )
            np.testing.assert_array_equal(
                slot_evict[start_slot:start_slot + live],
                state.evict_mask[0, h, :live].astype(np.uint8),
                err_msg=f"row {r} head {h} evict_mask",
            )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_append_decode_grows_with_compaction_bit_exact() -> None:
    # Evicted in-window rows age out and are dropped each step while never
    # rows and appends keep the extent growing, all within capacity.
    _case(
        prompts=[8],
        heads=2,
        dim=16,
        window=3,
        never_sets=[{0, 1}],
        steps=[
            [False, False],
            [True, True],
            [False, False],
        ],
        row_starts=[0],
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_append_decode_expire_compaction_bit_exact() -> None:
    _case(
        prompts=[9],
        heads=2,
        dim=16,
        window=2,
        never_sets=[{0, 1}],
        steps=[
            [False, False],
            [True, False],
        ],
        row_starts=[31],
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_append_decode_overflow_fails_closed() -> None:
    # Synthetic steady-state violation: the extent is full of never-evicted
    # rows at capacity. Appending must fail closed (status=1, extent and
    # live count untouched) exactly where the host parent raises MemoryError
    # without mutating state.
    from hipengine.kernels.hip_gfx1100.attention import (
        build_dms_compact,
        dms_append_decode_bf16,
    )

    heads, dim, window = 2, 16, 1
    backend = _backend(num_layers=1, heads=heads, dim=dim, window=window, slots=32)
    _admit(backend, request_id=0, tokens=5)  # provisional full-prompt extent
    state = backend.state_for_request(0)
    live = int(state.range_capacity[0, 0])
    assert live == 5
    fill_rng = np.random.default_rng(4242)
    k_fill = _bf16_bits(fill_rng.standard_normal((live, heads, dim)))
    v_fill = _bf16_bits(fill_rng.standard_normal((live, heads, dim)))
    k_new = _bf16_bits(fill_rng.standard_normal((heads, dim)))
    v_new = _bf16_bits(fill_rng.standard_normal((heads, dim)))
    ev_new = np.zeros(heads, dtype=np.uint8)
    base = np.asarray([int(state.base_offsets[0, h]) for h in range(heads)], dtype=np.int32)

    # Host parent on the same synthetic state: MemoryError, state untouched.
    state.k_payload[(0, 0)] = _bf16_roundtrip(k_fill[:, 0])
    state.k_payload[(0, 1)] = _bf16_roundtrip(k_fill[:, 1])
    state.v_payload[(0, 0)] = _bf16_roundtrip(v_fill[:, 0])
    state.v_payload[(0, 1)] = _bf16_roundtrip(v_fill[:, 1])
    state.token_positions[0, :, :live] = np.arange(live, dtype=np.int32)
    state.token_positions[0, :, live:] = -1
    state.live_counts[0, :] = live
    state.evict_mask[0, :, :live] = False
    with pytest.raises(MemoryError):
        backend.append_decode(
            0,
            _bf16_roundtrip(k_new).reshape(1, heads, dim),
            _bf16_roundtrip(v_new).reshape(1, heads, dim),
            ev_new.reshape(1, heads),
            position=4,
        )
    # The host parent must not have mutated the state on failure.
    np.testing.assert_array_equal(state.live_counts[0, :], live)
    np.testing.assert_array_equal(
        state.token_positions[0, 0, :live], np.arange(live)
    )

    # Device: same synthetic state -> status=1, extent untouched.
    total = int(backend.slots_per_layer)
    k_slot = np.zeros((total, dim), dtype=np.uint16)
    v_slot = np.zeros_like(k_slot)
    pos_buf = np.full((total,), -1, dtype=np.int32)
    ev_buf = np.zeros(total, dtype=np.uint8)
    live_buf = np.zeros((1, heads), dtype=np.int32)
    status = np.zeros((1, heads), dtype=np.int32)
    for h in range(heads):
        k_slot[base[h]:base[h] + live] = k_fill[:, h]
        v_slot[base[h]:base[h] + live] = v_fill[:, h]
        pos_buf[base[h]:base[h] + live] = np.arange(live)
    live_buf[0, :] = live
    row_positions = np.asarray([4], dtype=np.int32)

    buffers: dict[str, object] = {}

    def upload(name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes)
        buffers[name] = buf
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)

    try:
        upload("k_new", k_new)
        upload("v_new", v_new)
        upload("evict_new", ev_new)
        upload("row_positions", row_positions)
        upload("base", base)
        upload("capacity", np.ascontiguousarray(state.range_capacity[0]))
        upload("live", live_buf)
        upload("k_slot", k_slot)
        upload("v_slot", v_slot)
        upload("positions", pos_buf)
        upload("slot_evict", ev_buf)
        upload("status", status)
        library = build_dms_compact(load=True)
        dms_append_decode_bf16(
            buffers["k_new"].ptr,
            buffers["v_new"].ptr,
            buffers["evict_new"].ptr,
            buffers["row_positions"].ptr,
            buffers["base"].ptr,
            buffers["capacity"].ptr,
            buffers["live"].ptr,
            buffers["k_slot"].ptr,
            buffers["v_slot"].ptr,
            buffers["positions"].ptr,
            buffers["slot_evict"].ptr,
            buffers["status"].ptr,
            1,
            heads,
            dim,
            window,
            library=library,
        )
        out_status = np.zeros_like(status)
        out_live = np.zeros_like(live_buf)
        out_k = np.zeros_like(k_slot)
        out_v = np.zeros_like(v_slot)
        out_pos = np.full_like(pos_buf, -1)
        out_ev = np.zeros_like(ev_buf)
        for name, array in (
            ("status", out_status),
            ("live", out_live),
            ("k_slot", out_k),
            ("v_slot", out_v),
            ("positions", out_pos),
            ("slot_evict", out_ev),
        ):
            copy_device_to_host(host_array_ptr(array), buffers[name], array.nbytes)
    finally:
        for buf in buffers.values():
            free(buf)

    np.testing.assert_array_equal(
        out_status, np.ones_like(status), err_msg="overflow must set status=1"
    )
    np.testing.assert_array_equal(out_live, live_buf, err_msg="live count mutated")
    np.testing.assert_array_equal(out_pos, pos_buf, err_msg="positions mutated")
    np.testing.assert_array_equal(out_ev, ev_buf, err_msg="evict flags mutated")
    np.testing.assert_array_equal(out_k, k_slot, err_msg="K payload mutated")
    np.testing.assert_array_equal(out_v, v_slot, err_msg="V payload mutated")


def _bf16_roundtrip(bits: np.ndarray) -> np.ndarray:
    """BF16 bits (uint16) -> FP32 with the same BF16 mantissa (oracle view)."""
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_append_decode_multi_drop_recompute_bit_exact() -> None:
    # Corrupted steady state (two evicted+out-of-window rows at once, which
    # admission never produces): the parent's full keep-recompute drops both
    # in one append where the steady-state single-drop would fail closed.
    # Synthetic device extent + host parent on the identical state.
    from hipengine.kernels.hip_gfx1100.attention import (
        build_dms_compact,
        dms_append_decode_bf16,
    )

    heads, dim, window = 2, 16, 2
    backend = _backend(num_layers=1, heads=heads, dim=dim, window=window, slots=32)
    _admit(backend, request_id=0, tokens=4)  # per_head = min(4, 2+2) = 4
    state = backend.state_for_request(0)
    live = 4
    p_new = 5
    fill_rng = np.random.default_rng(5150)
    k_fill = _bf16_bits(fill_rng.standard_normal((live, heads, dim)))
    v_fill = _bf16_bits(fill_rng.standard_normal((live, heads, dim)))
    k_new = _bf16_bits(fill_rng.standard_normal((heads, dim)))
    v_new = _bf16_bits(fill_rng.standard_normal((heads, dim)))
    ev_new = np.zeros(heads, dtype=np.uint8)
    positions = np.arange(live, dtype=np.int32)  # 0..3
    evict = np.zeros((live, heads), dtype=bool)
    evict[1, :] = True  # rows 1 and 2 are evicted+out-of-window at p=5, w=2
    evict[2, :] = True
    base = np.asarray([int(state.base_offsets[0, h]) for h in range(heads)], dtype=np.int32)

    # Host parent on the same state: full keep-recompute drops rows 1 and 2.
    state.k_payload[(0, 0)] = _bf16_roundtrip(k_fill[:, 0])
    state.k_payload[(0, 1)] = _bf16_roundtrip(k_fill[:, 1])
    state.v_payload[(0, 0)] = _bf16_roundtrip(v_fill[:, 0])
    state.v_payload[(0, 1)] = _bf16_roundtrip(v_fill[:, 1])
    state.token_positions[0, :, :live] = positions
    state.token_positions[0, :, live:] = -1
    state.live_counts[0, :] = live
    state.evict_mask[0, :, :live] = evict.T
    state.evict_mask[0, :, live:] = False
    backend.append_decode(
        0,
        _bf16_roundtrip(k_new).reshape(1, heads, dim),
        _bf16_roundtrip(v_new).reshape(1, heads, dim),
        ev_new.reshape(1, heads),
        position=p_new,
    )

    # Device: same synthetic state.
    total = int(backend.slots_per_layer)
    k_slot = np.zeros((total, dim), dtype=np.uint16)
    v_slot = np.zeros_like(k_slot)
    pos_buf = np.full((total,), -1, dtype=np.int32)
    ev_buf = np.zeros(total, dtype=np.uint8)
    for h in range(heads):
        k_slot[base[h]:base[h] + live] = k_fill[:, h]
        v_slot[base[h]:base[h] + live] = v_fill[:, h]
        pos_buf[base[h]:base[h] + live] = positions
        ev_buf[base[h]:base[h] + live] = evict[:, h].astype(np.uint8)
    live_buf = np.full((1, heads), live, dtype=np.int32)
    status = np.zeros((1, heads), dtype=np.int32)
    row_positions = np.asarray([p_new], dtype=np.int32)

    buffers: dict[str, object] = {}

    def upload(name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        buf = malloc(array.nbytes)
        buffers[name] = buf
        copy_host_to_device(buf, host_array_ptr(array), array.nbytes)

    try:
        upload("k_new", k_new)
        upload("v_new", v_new)
        upload("evict_new", ev_new)
        upload("row_positions", row_positions)
        upload("base", base)
        upload("capacity", np.ascontiguousarray(state.range_capacity[0]))
        upload("live", live_buf)
        upload("k_slot", k_slot)
        upload("v_slot", v_slot)
        upload("positions", pos_buf)
        upload("slot_evict", ev_buf)
        upload("status", status)
        library = build_dms_compact(load=True)
        dms_append_decode_bf16(
            buffers["k_new"].ptr,
            buffers["v_new"].ptr,
            buffers["evict_new"].ptr,
            buffers["row_positions"].ptr,
            buffers["base"].ptr,
            buffers["capacity"].ptr,
            buffers["live"].ptr,
            buffers["k_slot"].ptr,
            buffers["v_slot"].ptr,
            buffers["positions"].ptr,
            buffers["slot_evict"].ptr,
            buffers["status"].ptr,
            1,
            heads,
            dim,
            window,
            library=library,
        )
        outs = {
            "status": np.zeros_like(status),
            "live": np.zeros_like(live_buf),
            "k_slot": np.zeros_like(k_slot),
            "v_slot": np.zeros_like(v_slot),
            "positions": np.full_like(pos_buf, -1),
            "slot_evict": np.zeros_like(ev_buf),
        }
        for name, array in outs.items():
            copy_device_to_host(host_array_ptr(array), buffers[name], array.nbytes)
    finally:
        for buf in buffers.values():
            free(buf)

    (out_status, out_live, out_k, out_v, out_pos, out_ev) = (
        outs[n] for n in ("status", "live", "k_slot", "v_slot", "positions", "slot_evict")
    )
    assert (out_status == 0).all(), f"status: {out_status}"

    for h in range(heads):
        s = int(state.base_offsets[0, h])
        n = int(state.live_counts[0, h])
        np.testing.assert_array_equal(out_live[0, h], n)
        np.testing.assert_array_equal(
            out_pos[s:s + n],
            state.token_positions[0, h, :n],
            err_msg=f"head {h} positions",
        )
        np.testing.assert_array_equal(
            out_ev[s:s + n],
            state.evict_mask[0, h, :n].astype(np.uint8),
            err_msg=f"head {h} evict flags",
        )
        np.testing.assert_array_equal(
            out_k[s:s + n], _bf16_bits(state.k_payload[(0, h)]), err_msg=f"head {h} K"
        )
        np.testing.assert_array_equal(
            out_v[s:s + n], _bf16_bits(state.v_payload[(0, h)]), err_msg=f"head {h} V"
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime not available")
def test_dms_append_decode_batched_rows_bit_exact() -> None:
    _case(
        prompts=[8, 6],
        heads=4,
        dim=128,
        window=3,
        never_sets=[{0, 1}, {0}],
        steps=[
            [False, False, True, False],
            [True, True, False, False],
            [False, True, False, False],
        ],
        row_starts=[0, 100],
    )
