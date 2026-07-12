#!/usr/bin/env python3
"""Build the compact SOL-G6 GGUF replacement-layout residency audit.

The input is a clean persistent-session ``qwen35_gguf_bench.py`` run with the
production graph explicitly enabled. Raw benchmark JSON stays under ``/tmp``;
this tool retains the allocation census, graph-close deltas, 24 GiB-class gate,
and a cryptographic link to the accepted SOL-G5 exact/performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KIND = "hipengine_gguf_residency_sol_g6_audit"
SCHEMA_VERSION = 1
GIB = 1 << 30


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _storage_class(layout: str) -> str:
    value = str(layout)
    if value == "raw_gguf":
        return "raw_gguf"
    if value in {"dense_f32", "dense_bf16"}:
        return "dense"
    if value == "q4_k_pack8" or value.endswith("_t16_v1") or value.endswith("_x8_v1"):
        return "replacement"
    return "other"


def _plan_rows(model: Path) -> list[dict[str, Any]]:
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.loading.qwen35_gguf_materialize import plan_qwen35_gguf_materialization

    reader = GGUFReader(model)
    plan = plan_qwen35_gguf_materialization(
        build_qwen35_gguf_tensor_map(reader.info),
        decode_repack=True,
    )
    return [
        {
            "slot_path": str(spec.slot_path),
            "source_name": str(spec.source.name),
            "source_nbytes": int(spec.source.nbytes),
            "quant_key": str(spec.quant_key),
            "layout": str(spec.layout),
            "allocation_names": [str(name) for name in spec.allocation_names],
        }
        for spec in plan.specs
    ]


def _audit_plan(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_bytes_by_class: dict[str, int] = defaultdict(int)
    seen_source_class: set[tuple[str, str]] = set()
    sidecars: list[dict[str, Any]] = []
    for row in normalized:
        source = str(row["source_name"])
        layout = str(row["layout"])
        storage_class = _storage_class(layout)
        by_source[source].append(row)
        key = (source, storage_class)
        if key not in seen_source_class:
            seen_source_class.add(key)
            source_bytes_by_class[storage_class] += int(row["source_nbytes"])
        allocations = {str(name) for name in row.get("allocation_names", ())}
        raw_plus_replacement = "raw" in allocations and bool(
            allocations.intersection({"tiles", "x8", "qweight", "scales", "mins"})
        )
        parallel_replacements = "tiles" in allocations and "x8" in allocations
        if raw_plus_replacement or parallel_replacements:
            sidecars.append(
                {
                    "slot_path": str(row["slot_path"]),
                    "source_name": source,
                    "layout": layout,
                    "allocation_names": sorted(allocations),
                }
            )

    duplicate_sources: list[dict[str, Any]] = []
    multi_layout_sources: list[dict[str, Any]] = []
    for source, source_rows in sorted(by_source.items()):
        layouts = sorted({str(row["layout"]) for row in source_rows})
        classes = sorted({_storage_class(layout) for layout in layouts})
        if len(layouts) > 1:
            multi_layout_sources.append({"source_name": source, "layouts": layouts})
        if "raw_gguf" in classes and "replacement" in classes:
            duplicate_sources.append(
                {
                    "source_name": source,
                    "source_nbytes": int(source_rows[0]["source_nbytes"]),
                    "layouts": layouts,
                }
            )
    return {
        "spec_count": len(normalized),
        "unique_source_count": len(by_source),
        "planned_source_bytes_by_storage_class": dict(sorted(source_bytes_by_class.items())),
        "raw_plus_replacement_duplicate_count": len(duplicate_sources),
        "raw_plus_replacement_duplicates": duplicate_sources,
        "optional_sidecar_count": len(sidecars),
        "optional_sidecars": sidecars,
        "multi_layout_source_count": len(multi_layout_sources),
        "multi_layout_sources": multi_layout_sources,
    }


def _snapshot(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        snapshot = payload["persistent_session_memory"]["snapshots"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"source benchmark is missing persistent snapshot {name!r}") from exc
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"persistent snapshot {name!r} must be an object")
    return snapshot


def _current_bytes(snapshot: Mapping[str, Any]) -> int:
    return int(snapshot.get("tracked", {}).get("current_allocated_bytes", 0))


def _hip_used_bytes(snapshot: Mapping[str, Any]) -> int | None:
    hip = snapshot.get("hip", {})
    if not isinstance(hip, Mapping) or not hip.get("available"):
        return None
    return int(hip["used_bytes"])


def _signed_delta(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return int(left) - int(right)


def _actual_weight_census(weight: Mapping[str, Any]) -> dict[str, Any]:
    by_layout = {str(key): int(value) for key, value in weight["by_layout_bytes"].items()}
    by_class: dict[str, int] = defaultdict(int)
    for layout, nbytes in by_layout.items():
        by_class[_storage_class(layout)] += nbytes
    total = int(weight["total_bytes"])
    classified = sum(by_class.values())
    if classified != total:
        raise ValueError(f"weight census mismatch: classified={classified}, total={total}")
    return {
        "total_bytes": total,
        "total_gib": total / GIB,
        "allocation_count": int(weight["allocation_count"]),
        "by_storage_class_bytes": dict(sorted(by_class.items())),
        "by_layout_bytes": by_layout,
        "by_quant_key_bytes": {
            str(key): int(value) for key, value in weight["by_quant_key_bytes"].items()
        },
        "by_allocation_name_bytes": {
            str(key): int(value) for key, value in weight["by_allocation_name_bytes"].items()
        },
    }


def _validate_g5(g5: Mapping[str, Any]) -> dict[str, Any]:
    stable = g5.get("correctness", {}).get("stable_key_relaunch", {})
    classification = g5.get("classification", {})
    comparisons = stable.get("comparisons", [])
    valid = bool(
        g5.get("status") == "accepted"
        and g5.get("performance_claim") is True
        and classification.get("decision") == "promote_state_bound_graph_relaunch"
        and classification.get("candidate_speedup_vs_eager", 0.0) > 1.0
        and stable.get("passed") is True
        and stable.get("first_failing_launch") is None
        and len(comparisons) >= 128
        and int(stable.get("third_and_later_launches_checked", 0)) >= 126
    )
    return {
        "valid": valid,
        "hipengine_commit": g5.get("provenance", {}).get("hipengine_commit"),
        "launches_checked": len(comparisons),
        "third_and_later_launches_checked": int(stable.get("third_and_later_launches_checked", 0)),
        "eager_median_ms_per_token": classification.get("eager_median_ms_per_token"),
        "graph_median_ms_per_token": classification.get("candidate_median_ms_per_token"),
        "speedup_vs_eager": classification.get("candidate_speedup_vs_eager"),
    }


def _build_artifact(
    source: Mapping[str, Any],
    *,
    plan_rows: Iterable[Mapping[str, Any]],
    g5: Mapping[str, Any],
    source_sha256: str,
    g5_sha256: str,
    source_path: str,
    g5_path: str,
    postprocess_command: list[str],
) -> dict[str, Any]:
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("source benchmark is missing canonical provenance")
    from hipengine.benchmark.provenance import validate_artifact_provenance

    validate_artifact_provenance(dict(provenance), require_model=True)
    runs = source.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("source benchmark has no runs")
    measured = [row for row in runs if row.get("measured")]
    if not measured:
        raise ValueError("source benchmark has no measured run")

    live_graph = _snapshot(source, "before_close")
    graph_closed = _snapshot(source, "after_graph_close")
    before_load = _snapshot(source, "before_load")
    session_closed = _snapshot(source, "after_close")
    breakdown = live_graph.get("owned_session_breakdown")
    if not isinstance(breakdown, Mapping):
        raise ValueError("source benchmark is missing owned-session breakdown")
    families = breakdown.get("families", {})
    weight_census = _actual_weight_census(families["weights"])
    decode_scratch = dict(families["decode_scratch"])
    session_buffers = dict(families["session_buffers"])
    plan_audit = _audit_plan(plan_rows)
    g5_link = _validate_g5(g5)

    graph_tracked_delta = _signed_delta(_current_bytes(live_graph), _current_bytes(graph_closed))
    graph_hip_delta = _signed_delta(_hip_used_bytes(live_graph), _hip_used_bytes(graph_closed))
    close_tracked_delta = _signed_delta(_current_bytes(session_closed), _current_bytes(before_load))
    close_hip_delta = _signed_delta(_hip_used_bytes(session_closed), _hip_used_bytes(before_load))
    owned_bytes = int(breakdown["total_bytes"])
    tracked_peak = int(source["persistent_session_memory"]["summary"]["tracked_peak_allocated_bytes"])
    budget_bytes = 24 * GIB

    graph_runs_exact = all(
        bool(row.get("effective_graph_replay_decode"))
        and row.get("correctness_sanity", {}).get("finite_final_logits") is True
        and int(row.get("correctness_sanity", {}).get("final_token_id", -1)) == 9707
        for row in measured
    )
    checks = {
        "source_provenance_clean": provenance.get("dirty") is False,
        "source_backend_is_gfx1151": provenance.get("resolved_backend") == "hip_gfx1151"
        and provenance.get("target_arch") == "gfx1151",
        "production_graph_exercised": graph_runs_exact,
        "no_default_raw_plus_replacement_duplicate": plan_audit[
            "raw_plus_replacement_duplicate_count"
        ]
        == 0,
        "no_default_optional_replacement_sidecar": plan_audit["optional_sidecar_count"] == 0,
        "owned_session_within_24gib": owned_bytes <= budget_bytes,
        "tracked_peak_within_24gib": tracked_peak <= budget_bytes,
        "graph_has_no_tracked_close_leak": graph_tracked_delta is not None
        and graph_tracked_delta >= 0,
        "session_returns_tracked_bytes_to_baseline": close_tracked_delta is not None
        and close_tracked_delta <= 0,
        "linked_g5_exact_performance_gate_valid": g5_link["valid"],
    }
    accepted = all(checks.values())
    kv_components = decode_scratch.get("by_component_bytes", {})
    graph_memory = {
        "method": "live production graph snapshot minus post-graph-close snapshot",
        "tracked_live_minus_closed_bytes": graph_tracked_delta,
        "tracked_retained_bytes_lower_bound": None
        if graph_tracked_delta is None
        else max(0, graph_tracked_delta),
        "hip_used_live_minus_closed_bytes": graph_hip_delta,
        "hip_retained_bytes_lower_bound": None if graph_hip_delta is None else max(0, graph_hip_delta),
        "owned_session_live_bytes": owned_bytes,
        "owned_session_closed_graph_bytes": int(
            graph_closed.get("owned_session_breakdown", {}).get("total_bytes", 0)
        ),
        "note": (
            "Production record_steps=0 allocates no graph-owned DeviceBuffer; HIP graph/exec "
            "internals are estimated from synchronized hipMemGetInfo phase deltas."
        ),
    }
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if accepted else "rejected",
        "performance_claim": False,
        "correctness_claim": True,
        "memory_claim": True,
        "classification": {
            "status": "accepted" if accepted else "rejected",
            "decision": (
                "retain_replacement_only_default_residency"
                if accepted
                else "reject_residency_or_evidence_gate"
            ),
            "checks": checks,
        },
        "workload": {
            "model": source.get("model"),
            "quant": source.get("quant"),
            "kv_storage_dtype": source.get("kv_storage_dtype"),
            "prompt_source": source.get("prompt_source"),
            "prompt_token_id": source.get("token_id"),
            "prompt_length": source.get("prompt_length"),
            "decode_tokens": source.get("decode_tokens"),
            "max_sequence_length": source.get("max_sequence_length"),
            "graph_replay_decode": source.get("graph_replay_decode"),
            "persistent_session": source.get("persistent_session"),
        },
        "plan_audit": plan_audit,
        "allocation_census": {
            "owned_session_total_bytes": owned_bytes,
            "owned_session_total_gib": owned_bytes / GIB,
            "resident_weights": weight_census,
            "decode_scratch": decode_scratch,
            "session_buffers": session_buffers,
            "kv_bytes": int(kv_components.get("full_attention_kv_cache", 0)),
            "kv_scale_bytes": int(kv_components.get("full_attention_kv_scales", 0)),
            "graph": graph_memory,
        },
        "capacity_gate_24gib": {
            "budget_bytes": budget_bytes,
            "budget_gib": 24.0,
            "owned_session_bytes": owned_bytes,
            "owned_session_gib": owned_bytes / GIB,
            "owned_margin_bytes": budget_bytes - owned_bytes,
            "owned_margin_gib": (budget_bytes - owned_bytes) / GIB,
            "tracked_peak_bytes": tracked_peak,
            "tracked_peak_gib": tracked_peak / GIB,
            "tracked_margin_bytes": budget_bytes - tracked_peak,
            "tracked_margin_gib": (budget_bytes - tracked_peak) / GIB,
        },
        "close_audit": {
            "tracked_after_close_minus_before_load_bytes": close_tracked_delta,
            "hip_used_after_close_minus_before_load_bytes": close_hip_delta,
        },
        "performance_non_regression": {
            "method": (
                "cryptographic link to the clean accepted production SOL-G5 gate; "
                "G6 changes only census/reporting"
            ),
            "artifact_path": g5_path,
            "artifact_sha256": g5_sha256,
            **g5_link,
        },
        "source": {
            "raw_benchmark_path": source_path,
            "raw_benchmark_sha256": source_sha256,
            "benchmark_command": source.get("argv"),
            "postprocess_command": postprocess_command,
        },
        "provenance": dict(provenance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--g5-artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_json.resolve()
    g5_path = args.g5_artifact.resolve()
    source = _load_json(source_path)
    g5 = _load_json(g5_path)
    model = Path(args.model or source.get("model", "")).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    artifact = _build_artifact(
        source,
        plan_rows=_plan_rows(model),
        g5=g5,
        source_sha256=_sha256_file(source_path),
        g5_sha256=_sha256_file(g5_path),
        source_path=str(source_path),
        g5_path=str(g5_path),
        postprocess_command=[str(part) for part in sys.argv],
    )
    text = json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if artifact["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
