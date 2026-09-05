from __future__ import annotations

import os

import pytest

from scripts.gguf_packed_ar_bench import (
    CONFIGURATIONS,
    SUPPORTED_BACKENDS,
    _PRODUCTION_ENV,
    _PROVENANCE_ENV_KEYS,
    _configuration_groups,
    _cross_configuration_correctness,
    _graph_bucket_shape_sha256,
    _graph_manifest_matches_configuration,
    _occupancy_event,
    _parse_configurations,
    _scaling_summary,
    _stats,
    _temporary_environment,
    build_parser,
)


def test_packed_ar_bench_accepts_both_gfx11_backends() -> None:
    assert SUPPORTED_BACKENDS == ("hip_gfx1100", "hip_gfx1151")
    parser = build_parser()

    assert parser.parse_args(["--backend", "hip_gfx1100"]).backend == "hip_gfx1100"
    assert parser.parse_args(["--backend", "hip_gfx1151"]).backend == "hip_gfx1151"
    assert parser.parse_args([]).route == "exact"
    assert parser.parse_args(["--route", "production"]).route == "production"


def test_packed_ar_production_route_unsets_exact_overrides() -> None:
    keys = tuple(_PRODUCTION_ENV)
    prior = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "forced-exact"
        with _temporary_environment(_PRODUCTION_ENV):
            assert all(key not in os.environ for key in keys)
        assert all(os.environ[key] == "forced-exact" for key in keys)
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_packed_ar_bench_records_visible_device_provenance_keys() -> None:
    assert _PROVENANCE_ENV_KEYS == (
        "HIPENGINE_BACKEND",
        "HIPENGINE_HIP_ARCH",
        "HIPENGINE_COMPILER_VERSION_FILE",
        "HIPENGINE_SUBMISSION_TRANSPORT",
        "HIPENGINE_GGUF_FP16_RECURRENT_STATE",
        "HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_MAX_HW_QUEUES",
    )


def test_packed_ar_bench_parses_honest_native_and_chunked_widths() -> None:
    canonical = tuple(CONFIGURATIONS)
    names = _parse_configurations(",".join(canonical))

    assert names == canonical
    assert build_parser().parse_args([]).configurations == ",".join(canonical)
    assert CONFIGURATIONS["c4"].native_group_width == 4
    assert CONFIGURATIONS["c4"].native_group_count == 1
    assert CONFIGURATIONS["native_c8"].logical_rows == 8
    assert CONFIGURATIONS["native_c8"].native_group_width == 8
    assert CONFIGURATIONS["native_c8"].native_group_count == 1
    assert CONFIGURATIONS["chunked_c8"].logical_rows == 8
    assert CONFIGURATIONS["chunked_c8"].native_group_width == 4
    assert CONFIGURATIONS["chunked_c8"].native_group_count == 2
    assert CONFIGURATIONS["serial_c4"].native_group_width == 1
    assert CONFIGURATIONS["serial_c4"].native_group_count == 4

    with pytest.raises(ValueError, match="unknown"):
        _parse_configurations("c1,native_c16")
    with pytest.raises(ValueError, match="unique"):
        _parse_configurations("c1,c1")
    with pytest.raises(ValueError, match="canonical"):
        _parse_configurations(",".join(reversed(canonical)))


def test_packed_ar_bench_builds_declared_group_boundaries() -> None:
    assert _configuration_groups(CONFIGURATIONS["c1"]) == ((0,),)
    assert _configuration_groups(CONFIGURATIONS["c2"]) == ((0, 1),)
    assert _configuration_groups(CONFIGURATIONS["c3"]) == ((0, 1, 2),)
    assert _configuration_groups(CONFIGURATIONS["c4"]) == ((0, 1, 2, 3),)
    assert _configuration_groups(CONFIGURATIONS["c5"]) == ((0, 1, 2, 3, 4),)
    assert _configuration_groups(CONFIGURATIONS["c6"]) == ((0, 1, 2, 3, 4, 5),)
    assert _configuration_groups(CONFIGURATIONS["c7"]) == ((0, 1, 2, 3, 4, 5, 6),)
    assert _configuration_groups(CONFIGURATIONS["native_c8"]) == (
        (0, 1, 2, 3, 4, 5, 6, 7),
    )
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


def test_packed_ar_bench_records_honest_native_and_chunked_c8_occupancy() -> None:
    native = _occupancy_event(
        CONFIGURATIONS["native_c8"],
        phase="decode_complete",
        elapsed_seconds=1.0,
    )
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

    assert native == {
        "phase": "decode_complete",
        "elapsed_seconds": 1.0,
        "logical_active_rows": 8,
        "native_group_width": 8,
        "native_group_count": 1,
        "physical_bucket_widths": [8],
        "active_masks": [[True] * 8],
    }
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


def test_packed_ar_bench_shape_key_excludes_pointer_bound_instance_identity() -> None:
    first = {
        "physical_rows": 4,
        "active_rows": 4,
        "state_generations": [512, 512, 512, 512],
        "context_bucket": 768,
        "buffer_identity_sha256": "first-pointers",
        "key_sha256": "first-instance",
    }
    next_instance = {
        **first,
        "buffer_identity_sha256": "next-pointers",
        "key_sha256": "next-instance",
    }
    next_shape = {**first, "context_bucket": 1024}

    assert _graph_bucket_shape_sha256(first) == _graph_bucket_shape_sha256(next_instance)
    assert _graph_bucket_shape_sha256(first) != _graph_bucket_shape_sha256(next_shape)


def test_packed_ar_bench_allows_explicit_c1_controls_but_not_c4_row_loops() -> None:
    manifest = {
        "mode": "decode_graph_replay",
        "physical_rows": 1,
        "active_rows": 1,
        "active_mask": [True],
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
    native_manifest = {
        **manifest,
        "physical_rows": 8,
        "active_rows": 8,
        "active_mask": [True] * 8,
        "model_step": {**manifest["model_step"], "host_model_row_loop_sites": 0},
    }
    assert _graph_manifest_matches_configuration(
        CONFIGURATIONS["native_c8"], native_manifest, decode_steps=2
    )
    assert not _graph_manifest_matches_configuration(
        CONFIGURATIONS["native_c8"], manifest, decode_steps=2
    )


def _summary(*hashes: str) -> dict[str, object]:
    return {
        "measured_trajectory_hashes": [list(hashes)],
        "repeatable_trajectories": True,
    }


def test_packed_ar_bench_native_c8_scaling_gate_uses_honest_controls() -> None:
    def rates(aggregate: float, rows: int) -> dict[str, object]:
        return {
            "rates": {
                "decode_tok_s_aggregate": {"median": aggregate},
                "decode_tok_s_per_request": {"median": aggregate / rows},
            }
        }

    scaling = _scaling_summary(
        {
            "c1": rates(80.0, 1),
            "c4": rates(180.0, 4),
            "native_c8": rates(250.0, 8),
            "chunked_c8": rates(175.0, 8),
            "serial_c4": rates(78.0, 4),
        }
    )

    assert scaling["native_c8_scaling_gate_passed"] is True
    assert scaling["direct_c1_c8_decode_tok_s_aggregate"]["1"] == 80.0
    assert scaling["direct_c1_c8_decode_tok_s_aggregate"]["4"] == 180.0
    assert scaling["direct_c1_c8_decode_tok_s_aggregate"]["8"] == 250.0
    assert scaling["direct_c1_c8_scaling_vs_c1"]["8"] == pytest.approx(3.125)
    assert scaling["native_c8_decode_tok_s_aggregate"] == 250.0
    assert scaling["ratios"]["native_c8_aggregate_vs_c1"] == pytest.approx(3.125)
    assert scaling["ratios"]["native_c8_aggregate_vs_chunked_c8"] == pytest.approx(
        250.0 / 175.0
    )
    assert scaling["ratios"]["native_c8_aggregate_vs_serial_c4"] == pytest.approx(
        250.0 / 78.0
    )


def test_packed_ar_bench_cross_configuration_gate_uses_serial_and_chunked_controls() -> None:
    summaries = {
        "c1": _summary("a"),
        "c2": _summary("a", "b"),
        "c3": _summary("a", "b", "c"),
        "c4": _summary("a", "b", "c", "d"),
        "c5": _summary("a", "b", "c", "d", "a"),
        "c6": _summary("a", "b", "c", "d", "a", "b"),
        "c7": _summary("a", "b", "c", "d", "a", "b", "c"),
        "native_c8": _summary("a", "b", "c", "d", "a", "b", "c", "d"),
        "chunked_c8": _summary("a", "b", "c", "d", "a", "b", "c", "d"),
        "serial_c4": _summary("a", "b", "c", "d"),
    }

    result = _cross_configuration_correctness(summaries)

    assert result["passed"] is True
    assert result["all_direct_c1_c8_exact"] is True
    assert result["c1_c2_c3_c4_prefix_exact"] is True
    assert all(result["direct_c1_c8_match_c4_repeating_fixture"].values())
    assert result["c4_matches_serial_c4"] is True
    assert result["native_c8_rows_match_c4"] is True
    assert result["chunked_c8_groups_match_c4"] is True

    summaries["serial_c4"] = _summary("a", "bad", "c", "d")
    failed = _cross_configuration_correctness(summaries)
    assert failed["passed"] is False
    assert failed["c4_matches_serial_c4"] is False
