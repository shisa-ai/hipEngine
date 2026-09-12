from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine.benchmark.matrix import (
    MatrixError,
    build_benchmark_matrix,
    validate_benchmark_matrix,
)
from hipengine.benchmark.prompts import token_ids_sha256


def _provenance(*, quant: str) -> dict[str, object]:
    return {
        "kind": "hipengine_artifact_provenance",
        "schema_version": 1,
        "collected_at": "2026-07-11T00:00:00+00:00",
        "repo_root": "/repo",
        "hipengine_commit": "a" * 40,
        "git_branch": "main",
        "dirty": False,
        "staged_dirty": False,
        "unstaged_dirty": False,
        "untracked_dirty": False,
        "untracked_count": 0,
        "configured_backend": "hip_gfx1151",
        "resolved_backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "device_name": "Radeon 8060S Graphics",
        "model_path": "/models/model",
        "model_revision": "b" * 40,
        "model_fingerprint": {
            "algorithm": "sha256-directory-manifest-v1",
            "value": "c" * 64,
            "exists": True,
            "path_type": "directory",
            "size_bytes": 123,
            "file_count": 1,
            "sampled_bytes": 123,
        },
        "quant": quant,
        "kv_dtype": "bf16",
        "command": ["python3", "bench.py"],
        "environment": {},
        "build_profile": None,
        "timing_protocol": "exact-token-test",
        "warmups": 1,
        "repetitions": 3,
        "rocm_version": "test",
        "hipcc_version": "test",
        "profiler": {"enabled": False, "kind": None, "command": None},
    }


def _exact_artifact(
    *,
    mode: str,
    generated_rows: tuple[tuple[int, ...], ...],
    quant: str,
    wall_s: float,
    telemetry: list[dict[str, object]],
    server_shape: bool,
) -> dict[str, object]:
    prompt_rows = ((10, 11, 12), (20, 21, 22))
    response_metadata: dict[str, object] | None = None
    if server_shape:
        response_metadata = {
            "hipengine": {
                "generation_shape": {
                    "schema_version": 1,
                    "route": "default",
                    "route_cap": {
                        "scope": "queue_requests",
                        "value": 2,
                        "applied": True,
                    },
                    "queue_group": {
                        "id": "queue-1",
                        "request_count": 1,
                        "prompt_rows": 2,
                        "item_index": 0,
                        "item_prompt_offset": 0,
                        "item_prompt_rows": 2,
                    },
                    "backend_groups": [
                        {
                            "id": "backend-1",
                            "call_index": 0,
                            "prompt_offset": 0,
                            "input_rows": 2,
                            "actual_group_rows": [2],
                            "max_actual_group_rows": 2,
                            "verifier_rows": 0,
                        }
                    ],
                    "verifier_rows": 0,
                }
            },
            "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
        }
    return {
        "kind": "hipengine_exact_token_oracle",
        "schema_version": 1,
        "mode": mode,
        "shape": {"prompt_count": 2, "prompt_length": 3, "max_tokens": 2},
        "prompt_token_ids": [list(row) for row in prompt_rows],
        "prompt_token_ids_sha256": [token_ids_sha256(row) for row in prompt_rows],
        "generated_token_ids": [list(row) for row in generated_rows],
        "generated_token_ids_sha256": [token_ids_sha256(row) for row in generated_rows],
        "performance_claim": True,
        "request": {
            "route": mode,
            "model": "model",
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
        },
        "measurement": {
            "wall_s": wall_s,
            "timing_scope": "direct_call" if mode == "direct" else "client_e2e",
            "eligible": True,
            "reason": "test",
        },
        "exact_token_parity": {
            "passed": None if mode == "direct" else True,
            "status": "oracle" if mode == "direct" else "matched",
        },
        "generation_telemetry": telemetry,
        "response_metadata": response_metadata,
        "provenance": _provenance(quant=quant),
    }


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _choice_timing(value: float) -> list[dict[str, object]]:
    return [
        {
            "decode_state": {"execution_path": "direct-eager"},
            "timing": {"decode_ms": value},
            "timing_scope": "choice",
            "group_rows": 1,
            "timing_owner": True,
        },
        {
            "decode_state": {"execution_path": "direct-eager"},
            "timing": {"decode_ms": value},
            "timing_scope": "choice",
            "group_rows": 1,
            "timing_owner": True,
        },
    ]


def _batch_timing(value: float) -> list[dict[str, object]]:
    common = {
        "decode_state": {"execution_path": "scheduler-native"},
        "timing": {"batch_decode_ms": value},
        "timing_scope": "batch",
        "batch_id": "batch-1",
        "group_rows": 2,
    }
    return [{**common, "timing_owner": True}, {**common, "timing_owner": False}]


def _manifest(tmp_path: Path) -> dict[str, object]:
    generated = {
        "paro": ((101, 102), (201, 202)),
        "gguf": ((301, 302), (401, 402)),
    }
    rows: list[dict[str, object]] = []
    for engine, quant in (("paro", "w4_paro"), ("gguf", "gguf_q4_k_m")):
        for surface, mode in (("direct", "direct"), ("server", "http")):
            row_id = f"{engine}-{surface}"
            artifact_path = tmp_path / f"{row_id}.json"
            memory_path = tmp_path / f"{row_id}-memory.json"
            profiler_path = tmp_path / f"{row_id}-profiler.json"
            _write(
                artifact_path,
                _exact_artifact(
                    mode=mode,
                    generated_rows=generated[engine],
                    quant=quant,
                    wall_s=2.0 if surface == "direct" else 1.0,
                    telemetry=_choice_timing(5.0) if surface == "direct" else _batch_timing(8.0),
                    server_shape=surface == "server",
                ),
            )
            _write(
                memory_path,
                {
                    "result": {
                        "scope": "hipengine_tracked_process",
                        "current_allocated_bytes": 10,
                        "peak_allocated_bytes": 20,
                    }
                },
            )
            _write(
                profiler_path,
                {
                    "summary": {
                        "kind": "rocprofv3_kernel_summary",
                        "kernel_calls": 7,
                        "total_gpu_ms": 3.5,
                        "families": [{"name": "linear", "calls": 7, "total_gpu_ms": 3.5}],
                    }
                },
            )
            rows.append(
                {
                    "id": row_id,
                    "case_id": f"{engine}-ar",
                    "engine": engine,
                    "surface": surface,
                    "path_variant": "ar",
                    "artifact": artifact_path.name,
                    "memory": {"artifact": memory_path.name, "pointer": "/result"},
                    "profiler": {"artifact": profiler_path.name, "pointer": "/summary"},
                }
            )
    return {
        "kind": "hipengine_benchmark_matrix_manifest",
        "schema_version": 1,
        "name": "paro-gguf-direct-server",
        "required_engines": ["paro", "gguf"],
        "required_surfaces": ["direct", "server"],
        "requirements": {
            "performance_claim": True,
            "clean_provenance": True,
            "scoped_timing": True,
            "memory": True,
            "profiler": True,
            "server_generation_shape": True,
        },
        "rows": rows,
    }


def test_matrix_joins_four_surfaces_and_repairs_denominators(tmp_path: Path) -> None:
    matrix = build_benchmark_matrix(
        _manifest(tmp_path),
        base_dir=tmp_path,
        report_provenance=_provenance(quant="mixed"),
    )

    assert matrix["kind"] == "hipengine_benchmark_matrix"
    assert matrix["schema_version"] == 1
    assert matrix["coverage"] == {
        "engines": ["gguf", "paro"],
        "surfaces": ["direct", "server"],
        "path_variants": ["ar"],
        "required_engines": ["gguf", "paro"],
        "required_surfaces": ["direct", "server"],
        "required_grid_complete": True,
    }
    assert matrix["eligibility"] == {"passed": True, "blockers": []}
    assert matrix["performance_claim"] is True

    rows = {row["id"]: row for row in matrix["rows"]}
    assert rows["paro-direct"]["latency"]["total_generated_tokens"] == 4
    assert rows["paro-direct"]["latency"]["generated_tokens_per_second"] == 2.0
    assert rows["paro-direct"]["timing"]["owned_totals"] == {"decode_ms": 10.0}
    assert rows["paro-server"]["timing"]["owned_totals"] == {"batch_decode_ms": 8.0}
    assert rows["paro-server"]["timing"]["dedup"] == {
        "batch_ids": ["batch-1"],
        "batch_payloads_counted": 1,
        "choice_payloads_counted": 0,
        "non_owner_copies_ignored": 1,
    }
    assert rows["paro-server"]["path"]["backend_group_rows"] == [2]
    assert rows["paro-server"]["path"]["max_backend_group_rows"] == 2
    assert rows["paro-server"]["memory"]["peak_allocated_bytes"] == 20
    assert rows["paro-server"]["profiler"]["summary"]["kernel_calls"] == 7

    pairs = {row["case_id"]: row for row in matrix["comparisons"]["direct_server"]}
    assert pairs["paro-ar"]["exact_generated_ids_equal"] is True
    assert pairs["paro-ar"]["rates"] == {"direct": 2.0, "server": 4.0}
    assert pairs["paro-ar"]["rate_ratio"] is None
    assert pairs["paro-ar"]["ratio_reason"] == "timing scopes differ: direct_call vs client_e2e"


def test_matrix_rejects_direct_server_generated_id_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    server_path = tmp_path / "paro-server.json"
    payload = json.loads(server_path.read_text(encoding="utf-8"))
    payload["generated_token_ids"][0][0] = 999
    payload["generated_token_ids_sha256"][0] = token_ids_sha256((999, 102))
    _write(server_path, payload)

    with pytest.raises(MatrixError, match="generated token IDs differ"):
        build_benchmark_matrix(
            manifest,
            base_dir=tmp_path,
            report_provenance=_provenance(quant="mixed"),
        )


def test_matrix_rejects_multiple_batch_timing_owners(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    server_path = tmp_path / "paro-server.json"
    payload = json.loads(server_path.read_text(encoding="utf-8"))
    payload["generation_telemetry"][1]["timing_owner"] = True
    _write(server_path, payload)

    with pytest.raises(MatrixError, match="requires exactly one timing owner"):
        build_benchmark_matrix(
            manifest,
            base_dir=tmp_path,
            report_provenance=_provenance(quant="mixed"),
        )


def test_matrix_marks_incomplete_required_grid_ineligible(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest["rows"] = [
        row for row in manifest["rows"] if row["id"] != "gguf-server"
    ]

    matrix = build_benchmark_matrix(
        manifest,
        base_dir=tmp_path,
        report_provenance=_provenance(quant="mixed"),
    )

    assert matrix["coverage"]["required_grid_complete"] is False
    assert matrix["eligibility"]["passed"] is False
    assert "required matrix row is missing: engine=gguf surface=server" in matrix["eligibility"]["blockers"]
    assert matrix["performance_claim"] is False


def test_matrix_json_schemas_track_kinds_and_closed_top_level_contracts() -> None:
    matrix_schema = json.loads(
        Path("benchmarks/schemas/benchmark-matrix.schema.json").read_text(encoding="utf-8")
    )
    manifest_schema = json.loads(
        Path("benchmarks/schemas/benchmark-matrix-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert matrix_schema["properties"]["kind"] == {
        "const": "hipengine_benchmark_matrix"
    }
    assert matrix_schema["properties"]["schema_version"] == {"const": 1}
    assert matrix_schema["additionalProperties"] is False
    assert set(matrix_schema["required"]) == set(matrix_schema["properties"])
    assert manifest_schema["properties"]["kind"] == {
        "const": "hipengine_benchmark_matrix_manifest"
    }
    assert manifest_schema["additionalProperties"] is False


def test_matrix_validation_rejects_forged_generated_token_denominator(
    tmp_path: Path,
) -> None:
    matrix = build_benchmark_matrix(
        _manifest(tmp_path),
        base_dir=tmp_path,
        report_provenance=_provenance(quant="mixed"),
    )
    matrix["rows"][0]["latency"]["generated_tokens_per_second"] = 999.0

    with pytest.raises(MatrixError, match="rate denominator is forged"):
        validate_benchmark_matrix(matrix)
