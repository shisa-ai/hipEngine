from __future__ import annotations

import pytest

from scripts.gguf_packed_ar_bench import (
    CONFIGURATIONS,
    _configuration_groups,
    _cross_configuration_correctness,
    _graph_manifest_matches_configuration,
    _occupancy_event,
    _parse_configurations,
    _stats,
)


def test_packed_ar_bench_parses_honest_native_and_chunked_widths() -> None:
    names = _parse_configurations("c1,c2,c4,chunked_c8,serial_c4")

    assert names == ("c1", "c2", "c4", "chunked_c8", "serial_c4")
    assert CONFIGURATIONS["c4"].native_group_width == 4
    assert CONFIGURATIONS["c4"].native_group_count == 1
    assert CONFIGURATIONS["chunked_c8"].logical_rows == 8
    assert CONFIGURATIONS["chunked_c8"].native_group_width == 4
    assert CONFIGURATIONS["chunked_c8"].native_group_count == 2
    assert CONFIGURATIONS["serial_c4"].native_group_width == 1
    assert CONFIGURATIONS["serial_c4"].native_group_count == 4

    with pytest.raises(ValueError, match="unknown"):
        _parse_configurations("c1,native_c8")
    with pytest.raises(ValueError, match="unique"):
        _parse_configurations("c1,c1")


def test_packed_ar_bench_builds_declared_group_boundaries() -> None:
    assert _configuration_groups(CONFIGURATIONS["c1"]) == ((0,),)
    assert _configuration_groups(CONFIGURATIONS["c2"]) == ((0, 1),)
    assert _configuration_groups(CONFIGURATIONS["c4"]) == ((0, 1, 2, 3),)
    assert _configuration_groups(CONFIGURATIONS["chunked_c8"]) == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )
    assert _configuration_groups(CONFIGURATIONS["serial_c4"]) == (
        (0,),
        (1,),
        (2,),
        (3,),
    )


def test_packed_ar_bench_records_static_occupancy_without_inventing_native_c8() -> None:
    chunked = _occupancy_event(
        CONFIGURATIONS["chunked_c8"],
        phase="decode_complete",
        elapsed_seconds=1.25,
    )
    serial = _occupancy_event(
        CONFIGURATIONS["serial_c4"],
        phase="admitted",
        elapsed_seconds=0.0,
    )

    assert chunked == {
        "phase": "decode_complete",
        "elapsed_seconds": 1.25,
        "logical_active_rows": 8,
        "native_group_width": 4,
        "native_group_count": 2,
        "physical_bucket_widths": [4, 4],
        "active_masks": [[True] * 4, [True] * 4],
    }
    assert serial["logical_active_rows"] == 4
    assert serial["native_group_width"] == 1
    assert serial["native_group_count"] == 4


def test_packed_ar_bench_stats_report_latency_distribution_and_variance() -> None:
    stats = _stats([1.0, 2.0, 3.0, 4.0])

    assert stats["samples"] == [1.0, 2.0, 3.0, 4.0]
    assert stats["median"] == pytest.approx(2.5)
    assert stats["p95"] == pytest.approx(4.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["stdev"] is not None
    assert stats["stdev_pct_of_median"] is not None


def test_packed_ar_bench_allows_explicit_c1_controls_but_not_c4_row_loops() -> None:
    manifest = {
        "mode": "decode_graph_replay",
        "graph": {"captured": True, "replay_count": 2, "replayed_steps": 2},
        "model_step": {
            "complete_c1_session_replays": 0,
            "complete_c1_layer_replays": 0,
            "host_model_row_loop_sites": 30,
        },
        "host_device_movement": {
            "host_to_device_total_copies": 0,
            "device_to_host_vector_copies": 0,
        },
    }

    assert _graph_manifest_matches_configuration(
        CONFIGURATIONS["c1"], manifest, decode_steps=2
    )
    assert _graph_manifest_matches_configuration(
        CONFIGURATIONS["serial_c4"], manifest, decode_steps=2
    )
    assert not _graph_manifest_matches_configuration(
        CONFIGURATIONS["c4"], manifest, decode_steps=2
    )


def _summary(*hashes: str) -> dict[str, object]:
    return {
        "measured_trajectory_hashes": [list(hashes)],
        "repeatable_trajectories": True,
    }


def test_packed_ar_bench_cross_configuration_gate_uses_serial_and_chunked_controls() -> None:
    summaries = {
        "c1": _summary("a"),
        "c2": _summary("a", "b"),
        "c4": _summary("a", "b", "c", "d"),
        "chunked_c8": _summary("a", "b", "c", "d", "a", "b", "c", "d"),
        "serial_c4": _summary("a", "b", "c", "d"),
    }

    result = _cross_configuration_correctness(summaries)

    assert result["passed"] is True
    assert result["c1_c2_c4_prefix_exact"] is True
    assert result["c4_matches_serial_c4"] is True
    assert result["chunked_c8_groups_match_c4"] is True

    summaries["serial_c4"] = _summary("a", "bad", "c", "d")
    failed = _cross_configuration_correctness(summaries)
    assert failed["passed"] is False
    assert failed["c4_matches_serial_c4"] is False
