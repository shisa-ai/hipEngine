"""Unit tests for the Q8_0 grouped WMMA down tile-map prerequisite.

The Q8_0 grouped WMMA down owner consumes a 16-row WMMA tile map. The shared
``scratch.group_wmma_total`` buffer is written by ``qwen35_moe_group_prefix``
with the compact padded row total before the gate/up leg; the exact Q4/Q5
gate/up owners overwrite it with a tile map, but the strict Q4 owner does not.
A Q8_0 down owner that reads the shared buffer after strict Q4 therefore sees
the group-prefix total (10230 for a 1023-row chunk with top-k 10), which is not
a multiple of 16.

These tests mock the kernel launches on CPU and assert that
``_run_qwen4_exp_q8_wmma_down`` builds the current map, passes the static
16-row tile upper bound to the owner, and refuses an invalid bound without
calling the owner. The final test pins the exact stale arguments that raised
the production ValueError.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _runner():
    from hipengine.runtime import qwen4_exp_runner

    return qwen4_exp_runner


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.tile_map_calls: list[dict] = []
        self.owner_calls: list[dict] = []


def _install(monkeypatch, recorder: _Recorder):
    module = _runner()

    def fake_tile_map(expert_start, wmma_start, tile_expert, wmma_total,
                      num_experts, *, tile_capacity=0, stream=0, runtime=None):
        recorder.events.append(("tile_map", num_experts, tile_capacity, stream))
        recorder.tile_map_calls.append(
            {
                "expert_start": expert_start,
                "wmma_start": wmma_start,
                "tile_expert": tile_expert,
                "wmma_total": wmma_total,
                "num_experts": num_experts,
                "tile_capacity": tile_capacity,
                "stream": stream,
            }
        )

    def fake_owner(*args, **kwargs):
        recorder.events.append(("owner", args, kwargs))
        recorder.owner_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(module, "qwen35_moe_wmma_tile_map", fake_tile_map)
    return SimpleNamespace(), fake_owner


def _call(
    module,
    runtime,
    owner,
    *,
    compact_rows: int = 64,
    num_experts: int = 512,
    risk_kwargs=None,
    stream: int = 77,
    tile_capacity: int = 4096,
):
    return module._run_qwen4_exp_q8_wmma_down(
        input_ptr=11,
        expert_start_ptr=12,
        wmma_expert_start_ptr=13,
        tile_expert_ptr=14,
        weights_ptr=15,
        output_ptr=16,
        wmma_total_buffer=SimpleNamespace(ptr=0xDEAD, nbytes=8),
        compact_rows=compact_rows,
        num_experts=num_experts,
        in_features=640,
        out_features=2560,
        tile_capacity=tile_capacity,
        stream=stream,
        runtime=runtime,
        owner=owner,
        risk_kwargs=risk_kwargs or {},
    )


def _actual_tiles(counts: list[int]) -> int:
    return sum((count + 15) // 16 for count in counts)


def test_q8_wmma_down_builds_current_map_with_static_bound(monkeypatch) -> None:
    """Map first, then the static bound; no D2H readback or stream sync."""

    module = _runner()
    recorder = _Recorder()
    runtime, owner = _install(monkeypatch, recorder)

    rows = _call(module, runtime, owner, stream=77)

    # compact=64, experts=512 -> active=64, static tiles=64.
    assert rows == 64 * 16
    assert [event[0] for event in recorder.events] == ["tile_map", "owner"]

    tile = recorder.tile_map_calls[0]
    assert tile["expert_start"] == 12
    assert tile["wmma_start"] == 13
    assert tile["tile_expert"] == 14
    assert tile["num_experts"] == 512
    assert tile["tile_capacity"] == 64
    assert tile["stream"] == 77
    assert tile["wmma_total"] == 0xDEAD

    call = recorder.owner_calls[0]
    # Positional ABI: (..., compact_rows, num_experts, in_features,
    # out_features, wmma_total_rows) at indices 6-10.
    assert call["args"][6:11] == (64, 512, 640, 2560, 64 * 16)
    assert call["kwargs"]["stream"] == 77
    assert call["kwargs"]["runtime"] is runtime


def test_q8_wmma_down_static_bound_covers_full_chunk(monkeypatch) -> None:
    """The 1023-row/top-k-10 chunk uses 1119 static tiles, not 10230 rows."""

    module = _runner()
    recorder = _Recorder()
    runtime, owner = _install(monkeypatch, recorder)

    rows = _call(module, runtime, owner, compact_rows=10230, num_experts=512)

    assert rows == 1119 * 16
    assert recorder.tile_map_calls[0]["tile_capacity"] == 1119
    assert recorder.owner_calls[0]["args"][10] == 1119 * 16


def test_q8_wmma_down_passes_guarded_risk_arguments(monkeypatch) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime, owner = _install(monkeypatch, recorder)

    risk = dict(risk_count_ptr=101, risk_indices_ptr=102, risk_capacity=64 * 2560)
    _call(module, runtime, owner, risk_kwargs=risk)

    call = recorder.owner_calls[0]
    assert call["kwargs"]["risk_count_ptr"] == 101
    assert call["kwargs"]["risk_indices_ptr"] == 102
    assert call["kwargs"]["risk_capacity"] == 64 * 2560


def test_q8_wmma_down_rejects_zero_compact_without_launching(monkeypatch) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime, owner = _install(monkeypatch, recorder)

    with pytest.raises(RuntimeError):
        _call(module, runtime, owner, compact_rows=0)

    assert recorder.events == []
    assert recorder.owner_calls == []


def test_q8_wmma_down_rejects_capacity_below_static_bound(monkeypatch) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime, owner = _install(monkeypatch, recorder)

    with pytest.raises(RuntimeError):
        _call(module, runtime, owner, tile_capacity=63)

    assert recorder.events == []
    assert recorder.owner_calls == []


def test_q8_wmma_down_has_no_sync_without_a_stream(monkeypatch) -> None:
    module = _runner()
    recorder = _Recorder()
    runtime, owner = _install(monkeypatch, recorder)

    rows = _call(module, runtime, owner, stream=0)

    assert rows == 64 * 16
    assert [event[0] for event in recorder.events] == ["tile_map", "owner"]
    assert recorder.owner_calls[0]["kwargs"]["stream"] == 0


@pytest.mark.parametrize(
    ("compact_rows", "num_experts", "expected"),
    [
        (64, 512, 64),
        (100, 3, 9),
        (10230, 512, 1119),
        (1024, 512, 544),
    ],
)
def test_q8_static_tile_count_is_an_upper_bound(
    compact_rows: int, num_experts: int, expected: int
) -> None:
    module = _runner()
    assert module._qwen4_exp_q8_static_tile_count(compact_rows, num_experts) == expected


@pytest.mark.parametrize(
    "counts",
    [
        [1] * 100,
        [50, 50],
        [17, 17],
        [33, 33, 34],
        [1023] * 10,
        [1] * 511 + [9719],
    ],
)
def test_q8_static_tile_count_covers_actual_tiles(counts: list[int]) -> None:
    module = _runner()
    static = module._qwen4_exp_q8_static_tile_count(sum(counts), len(counts))
    assert _actual_tiles(counts) <= static


def test_q8_guarded_wrapper_rejects_stale_group_prefix_count() -> None:
    """The exact stale arguments from the strict-Q4 full-context failure.

    The group-prefix total is compact padded rows (1023 * top-k 10 = 10230),
    not a 16-row tile count. The guarded wrapper rejects it before any kernel
    launch; this test does not need HIP.
    """

    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
        gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out,
    )

    with pytest.raises(ValueError, match="invalid grouped block-scale shape or risk capacity"):
        gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out(
            1, 2, 3, 4, 5, 6,
            rows=10230,
            experts=512,
            hidden=640,
            outputs=2560,
            total_rows=10230,
            risk_count_ptr=7,
            risk_indices_ptr=8,
            risk_capacity=10230 * 2560,
        )
