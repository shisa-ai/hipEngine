from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.gguf_arbitrary_c_lifecycle import (
    _all_native,
    _capture_compaction_group_graph,
    _expected_dense_group_masks,
    _expected_hole_group_masks,
    _group_masks,
    _lifecycle_environment_snapshot,
    _load_quality_gate,
    _resolve_widths,
    _state_kv_accepted,
    _tracked_memory_recovery,
    build_parser,
    run,
)


def test_gguf_arbitrary_c_lifecycle_defaults_cover_both_physical_windows() -> None:
    args = build_parser().parse_args(["--model", "/tmp/model.gguf"])

    assert args.rows == 13
    assert tuple(args.cancel_slots) == (2, 10)
    assert args.original_max_tokens == 5
    assert args.newcomer_max_tokens == 3
    assert args.prefill_chunk_size == 256
    assert args.gdn_prefill_mode == "exact"
    assert args.compact_after_middle_hole is False
    assert args.backend == "hip_gfx1100"

    compact = build_parser().parse_args(
        ["--model", "/tmp/model.gguf", "--compact-after-middle-hole"]
    )
    assert compact.compact_after_middle_hole is True

    production = build_parser().parse_args(
        ["--model", "/tmp/model.gguf", "--gdn-prefill-mode", "auto"]
    )
    assert production.gdn_prefill_mode == "auto"


def test_gguf_arbitrary_c_lifecycle_records_production_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", "1")
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "2")
    monkeypatch.setenv("HIPENGINE_GPU_MAX_HW_QUEUES_POLICY", "explicit")
    monkeypatch.setenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", "exact")
    args = build_parser().parse_args(
        ["--model", "/tmp/model.gguf", "--gdn-prefill-mode", "auto"]
    )

    environment = _lifecycle_environment_snapshot(args)

    assert environment["HIPENGINE_GGUF_FP16_RECURRENT_STATE"] == "1"
    assert environment["GPU_MAX_HW_QUEUES"] == "2"
    assert environment["HIPENGINE_GPU_MAX_HW_QUEUES_POLICY"] == "explicit"
    assert environment["HIPENGINE_GGUF_GDN_PREFILL_MODE"] == "auto"


def test_gguf_arbitrary_c_lifecycle_summarizes_declared_packed_masks() -> None:
    plan = {
        "group_count": 2,
        "groups": [
            {
                "physical_rows": 8,
                "active_mask": [True, True, False, True, True, True, True, True],
                "execution_path": "packed_native",
            },
            {
                "physical_rows": 8,
                "active_mask": [True, True, False, True, True, False, False, False],
                "execution_path": "packed_native",
            },
        ],
    }

    assert _all_native(plan)
    assert _group_masks(plan) == ["11011111", "11011000"]

    c1_plan = {
        "group_count": 1,
        "groups": [
            {
                "physical_rows": 1,
                "active_mask": [True],
                "execution_path": "native_c1_eager",
            }
        ],
    }
    assert _all_native(c1_plan)
    c1_plan["groups"][0]["execution_path"] = "native_c1_graph"
    assert _all_native(c1_plan)
    c1_plan["groups"][0]["physical_rows"] = 2
    assert not _all_native(c1_plan)

    plan["groups"][1]["execution_path"] = "serial_fallback"
    assert not _all_native(plan)


def test_gguf_arbitrary_c_lifecycle_pins_actual_graph_kind_for_compaction() -> None:
    calls: list[tuple[str, object]] = []
    c1_graph = object()
    c3_graph = object()

    class FakeSession:
        def capture_decode_graph(self, **kwargs):
            calls.append(("c1", kwargs))
            return c1_graph

        def capture_packed_decode_graph(self, token_ids, **kwargs):
            calls.append(("wrong_slot_owner", (tuple(token_ids), kwargs)))
            return object()

    class FakePackedOwner:
        def capture_packed_decode_graph(self, token_ids, **kwargs):
            calls.append(("packed", (tuple(token_ids), kwargs)))
            return c3_graph

    sessions = [FakeSession() for _ in range(3)]
    packed_owner = FakePackedOwner()
    rows = {
        request_id: SimpleNamespace(
            lease=SimpleNamespace(session=sessions[request_id]),
            slot=SimpleNamespace(
                session=sessions[request_id],
                prev_token=100 + request_id,
                seq_position=20 + request_id,
                c1_decode_graph=None,
            ),
        )
        for request_id in range(3)
    }
    runner = SimpleNamespace(
        _rows=rows,
        _packed_execution_owner=lambda _session: packed_owner,
    )

    result = _capture_compaction_group_graph(
        runner,
        {
            "request_ids": [2],
            "physical_rows": 1,
            "active_slot_indices": [0],
        },
    )
    assert result is c1_graph
    assert rows[2].slot.c1_decode_graph is c1_graph
    assert calls == [
        (
            "c1",
            {
                "position": 22,
                "steps_per_replay": 1,
                "max_replay_steps": 1,
                "attention_max_context_len": 23,
                "input_token_id": 102,
            },
        )
    ]

    calls.clear()
    result = _capture_compaction_group_graph(
        runner,
        {
            "request_ids": [0, 1, 2],
            "physical_rows": 3,
            "active_slot_indices": [0, 1, 2],
        },
    )
    assert result is c3_graph
    assert calls == [
        (
            "packed",
            (
                (100, 101, 102),
                {
                    "sessions": tuple(sessions),
                    "physical_rows": 3,
                    "active_slot_indices": (0, 1, 2),
                    "steps_per_replay": 1,
                    "max_replay_steps": 1,
                    "record_steps": 1,
                },
            ),
        )
    ]


def test_gguf_arbitrary_c_lifecycle_accepts_single_physical_group_shape() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "/missing/model.gguf",
            "--rows",
            "8",
            "--cancel-slots",
            "2",
            "6",
        ]
    )
    with pytest.raises(ValueError, match="model does not exist"):
        run(args)


def test_gguf_arbitrary_c_lifecycle_derives_single_and_cross_group_masks() -> None:
    assert _expected_dense_group_masks(8) == ["11111111"]
    assert _expected_dense_group_masks(13) == ["11111111", "11111"]
    assert _expected_hole_group_masks(8, (2, 6), compact=True) == ["111111"]
    assert _expected_hole_group_masks(13, (2, 10), compact=False) == [
        "11011111",
        "11011",
    ]


def test_gguf_arbitrary_c_lifecycle_derives_direct_width_masks() -> None:
    widths = (1, 2, 3, 4, 5, 6, 7, 8)
    assert _expected_dense_group_masks(3, widths) == ["111"]
    assert _expected_dense_group_masks(5, widths) == ["11111"]
    assert _expected_dense_group_masks(6, widths) == ["111111"]
    assert _expected_dense_group_masks(7, widths) == ["1111111"]
    assert _expected_dense_group_masks(13, widths) == ["11111111", "11111"]
    assert _expected_hole_group_masks(5, (2,), compact=False, buckets=widths) == [
        "11011"
    ]
    assert _expected_hole_group_masks(13, (2, 10), compact=False, buckets=widths) == [
        "11011111",
        "11011",
    ]
    # The masked default remains unchanged from the production owner.
    assert _expected_dense_group_masks(5, (1, 2, 4, 8)) == ["11111000"]


def test_gguf_arbitrary_c_lifecycle_resolves_active_width_set(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", raising=False)
    assert _resolve_widths() == (1, 2, 3, 4, 5, 6, 7, 8)
    monkeypatch.setenv(
        "HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1,2,4,8"
    )
    assert _resolve_widths() == (1, 2, 4, 8)
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1 2 4 8")
    assert _resolve_widths() == (1, 2, 4, 8)
    monkeypatch.setenv("HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS", "1 3 2")
    with pytest.raises(ValueError, match="sorted unique widths starting at c1"):
        _resolve_widths()


def test_gguf_arbitrary_c_lifecycle_requires_tracked_memory_recovery() -> None:
    before = {
        "current_allocated_bytes": 100,
        "peak_allocated_bytes": 150,
        "active_allocations": 2,
    }
    after = {
        "current_allocated_bytes": 100,
        "peak_allocated_bytes": 300,
        "active_allocations": 2,
    }

    recovered = _tracked_memory_recovery(before, after)

    assert recovered["passed"] is True
    assert recovered["current_allocated_delta_bytes"] == 0
    assert recovered["active_allocation_delta"] == 0
    assert recovered["peak_allocated_bytes"] == 300

    after["active_allocations"] = 3
    assert _tracked_memory_recovery(before, after)["passed"] is False


def test_gguf_arbitrary_c_lifecycle_requires_explicit_arithmetic_drift_policy(
    tmp_path: Path,
) -> None:
    assert _state_kv_accepted(bit_exact=True, allow_c1_arithmetic_drift=False)
    assert not _state_kv_accepted(bit_exact=False, allow_c1_arithmetic_drift=False)
    assert _state_kv_accepted(bit_exact=False, allow_c1_arithmetic_drift=True)

    model = tmp_path / "model.gguf"
    model.write_bytes(b"fixture")
    artifact = tmp_path / "quality.json"
    artifact.write_text(
        json.dumps(
            {
                "kind": "hipengine_execution_profile_gguf_batch_route_requalification_capture",
                "measurement_valid": True,
                "quality": {
                    "hard_gates_passed": True,
                    "summary": {
                        "kl_max": 0.0,
                        "top1_agreement": 1.0,
                        "rows": 1950,
                    },
                },
                "protocol": {
                    "route_profile": "current_package_direct",
                    "static_widths": [3, 5, 6, 7],
                },
                "provenance": {
                    "dirty": False,
                    "resolved_backend": "hip_gfx1151",
                    "model_path": str(model),
                    "hipengine_commit": "abc123",
                    "host_name": "zbook",
                },
            }
        ),
        encoding="utf-8",
    )
    gate = _load_quality_gate(artifact, model=model, backend="hip_gfx1151")
    assert gate["rows"] == 1950
    assert gate["hard_gate_passed"] is True

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["protocol"] = {"route_profile": "q8t16_candidate", "static_widths": [4, 8]}
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="valid matching hard gate"):
        _load_quality_gate(artifact, model=model, backend="hip_gfx1151")


def test_gguf_arbitrary_c_lifecycle_rejects_too_small_shape_before_model_io() -> None:
    args = build_parser().parse_args(["--model", "/missing/model.gguf", "--rows", "2"])
    with pytest.raises(ValueError, match="at least 3"):
        run(args)

    args = build_parser().parse_args(
        [
            "--model",
            "/missing/model.gguf",
            "--cancel-slots",
            "2",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="unique"):
        run(args)
