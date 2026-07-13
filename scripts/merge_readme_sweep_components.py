#!/usr/bin/env python3
"""Merge one-workload README sweeps into a provenance-complete topline artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import (  # noqa: E402
    collect_artifact_provenance,
    validate_artifact_provenance,
)


STANDARD_WORKLOADS = ("512/128", "1K/128", "4K/128", "32K/128", "64K/128", "128K/128")
SUPPORTED_PLATFORMS = ("gfx1100", "gfx1151")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_identity(provenance: dict[str, Any]) -> tuple[Any, ...]:
    fingerprint = provenance.get("model_fingerprint") or {}
    return (
        provenance.get("hipengine_commit"),
        provenance.get("resolved_backend"),
        provenance.get("target_arch"),
        provenance.get("model_path"),
        fingerprint.get("algorithm"),
        fingerprint.get("value"),
        provenance.get("quant"),
        provenance.get("kv_dtype"),
        provenance.get("hipcc_version"),
    )


def _expected_protocol(*, engine: str, platform: str) -> tuple[int, int]:
    """Return the calibrated warmup/measured contract for one sweep lane."""

    if engine == "gguf" and platform == "gfx1151":
        return 1, 3
    return 2, 5


def _metric_gate(stats: dict[str, Any], *, expected_count: int) -> bool:
    count = stats.get("count")
    median = stats.get("median")
    stdev = stats.get("stdev")
    if type(count) is not int or count != expected_count:
        return False
    if not isinstance(median, (int, float)) or not math.isfinite(float(median)) or float(median) <= 0.0:
        return False
    if not isinstance(stdev, (int, float)) or not math.isfinite(float(stdev)):
        return False
    return float(stdev) <= 0.05 * float(median)


def _finite_final_logit_passed(correctness_sanity: dict[str, Any]) -> bool:
    """Accept the established PARO singular and GGUF plural field names."""

    return (
        correctness_sanity.get("finite_final_logit") is True
        or correctness_sanity.get("finite_final_logits") is True
    )


def _merge_component_payloads(
    components: Sequence[tuple[Path, dict[str, Any]]],
    *,
    engine: str,
    provenance: dict[str, Any],
    platform: str = "gfx1151",
) -> dict[str, Any]:
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unsupported README sweep platform {platform!r}")
    by_workload: dict[str, tuple[Path, dict[str, Any]]] = {}
    identities: set[tuple[Any, ...]] = set()
    component_rows: list[dict[str, Any]] = []
    all_finite = True
    all_ids_stable = True
    all_variance_ok = True
    all_clean = True
    expected_warmups, expected_repetitions = _expected_protocol(
        engine=engine, platform=platform
    )

    for path, payload in components:
        if payload.get("engine") != engine:
            raise ValueError(f"{path}: engine {payload.get('engine')!r} != {engine!r}")
        workloads = payload.get("workloads")
        if not isinstance(workloads, list) or len(workloads) != 1:
            raise ValueError(f"{path}: component must contain exactly one workload")
        workload = str(workloads[0])
        if workload in by_workload:
            raise ValueError(f"duplicate component workload {workload}")
        component_provenance = validate_artifact_provenance(
            payload.get("provenance") or {}, require_model=True
        )
        identities.add(_stable_identity(component_provenance))
        all_clean = all_clean and not bool(component_provenance["dirty"])
        if (
            component_provenance.get("warmups") != expected_warmups
            or component_provenance.get("repetitions") != expected_repetitions
        ):
            raise ValueError(
                f"{path}: expected provenance warmups={expected_warmups} "
                f"and repetitions={expected_repetitions}"
            )

        summary = (payload.get("summary_by_workload") or {}).get(workload)
        runs = (payload.get("runs_by_workload") or {}).get(workload)
        if not isinstance(summary, dict) or not isinstance(runs, list):
            raise ValueError(f"{path}: missing summary/runs for {workload}")
        measured = [run for run in runs if run.get("measured")]
        finite = all(
            _finite_final_logit_passed(run.get("correctness_sanity", {}))
            for run in measured
        )
        ids_stable = summary.get("final_token_ids_stable") is True
        variance_ok = _metric_gate(
            summary.get("prefill_tok_s") or {},
            expected_count=expected_repetitions,
        ) and _metric_gate(
            summary.get("decode_tok_s") or {},
            expected_count=expected_repetitions,
        )
        all_finite = all_finite and finite
        all_ids_stable = all_ids_stable and ids_stable
        all_variance_ok = all_variance_ok and variance_ok
        by_workload[workload] = (path, payload)
        component_rows.append(
            {
                "workload": workload,
                "source_name": path.name,
                "source_sha256": _sha256(path),
                "source_bytes": path.stat().st_size,
                "finite_final_logits": finite,
                "final_token_ids_stable": ids_stable,
                "variance_gate_passed": variance_ok,
                "command": component_provenance["command"],
            }
        )

    if tuple(by_workload) != STANDARD_WORKLOADS:
        missing = [item for item in STANDARD_WORKLOADS if item not in by_workload]
        extra = [item for item in by_workload if item not in STANDARD_WORKLOADS]
        raise ValueError(f"component workloads are incomplete/out of order: missing={missing}, extra={extra}")
    if len(identities) != 1:
        raise ValueError("component source/model/backend identities do not match")

    target_matches = provenance.get("target_arch") == platform
    accepted = bool(
        all_clean and all_finite and all_ids_stable and all_variance_ok and target_matches
    )
    first = by_workload[STANDARD_WORKLOADS[0]][1]
    return {
        "schema": 1,
        "kind": f"{platform}_readme_model_sweep_rollup",
        "status": "accepted_topline" if accepted else "rejected_topline_gate",
        "performance_claim": accepted,
        "performance_claim_scope": f"{platform} six-shape per-workload resident sweep",
        "engine": engine,
        "model": first["model"],
        "quant": first["quant"],
        "workloads": list(STANDARD_WORKLOADS),
        "session_scope": "one resident session per workload; reset between repetitions",
        "warmup_runs": expected_warmups,
        "measured_runs": expected_repetitions,
        "summary_by_workload": {
            workload: by_workload[workload][1]["summary_by_workload"][workload]
            for workload in STANDARD_WORKLOADS
        },
        "runs_by_workload": {
            workload: by_workload[workload][1]["runs_by_workload"][workload]
            for workload in STANDARD_WORKLOADS
        },
        "max_sequence_length_by_workload": {
            workload: by_workload[workload][1]["max_sequence_length"]
            for workload in STANDARD_WORKLOADS
        },
        "persistent_session_load_seconds_by_workload": {
            workload: by_workload[workload][1]["persistent_session_load_seconds"]
            for workload in STANDARD_WORKLOADS
        },
        "persistent_session_memory_by_workload": {
            workload: by_workload[workload][1]["persistent_session_memory"]
            for workload in STANDARD_WORKLOADS
        },
        "extra": first["extra"],
        "correctness": {
            "passed": bool(all_finite and all_ids_stable),
            "all_measured_final_logits_finite": all_finite,
            "all_workload_final_ids_stable": all_ids_stable,
            "all_workload_variance_gates_passed": all_variance_ok,
            "all_component_provenance_clean": all_clean,
            "target_arch_matches_platform": target_matches,
        },
        "components": component_rows,
        "provenance": provenance,
        "notes": [
            "Each shape uses its own right-sized resident session so short-row memory is not inflated by a 128K allocation.",
            "Load and graph capture are excluded from phase throughput; load is reported per workload.",
            f"The component artifacts remain diagnostic until this rollup verifies clean provenance, {expected_repetitions} measured samples, finite logits, stable final IDs, and <=5% stdev/median.",
        ],
    }


def _finalize_rollup(
    output: dict[str, Any], *, assembly_provenance: dict[str, Any]
) -> dict[str, Any]:
    """Attach assembly identity without replacing the measured source identity."""

    finalized = dict(output)
    finalized["rollup_assembly_provenance"] = assembly_provenance
    correctness = dict(finalized.get("correctness") or {})
    assembly_clean = assembly_provenance.get("dirty") is False
    correctness["rollup_assembly_provenance_clean"] = assembly_clean
    finalized["correctness"] = correctness
    if not assembly_clean:
        finalized["status"] = "rejected_topline_gate"
        finalized["performance_claim"] = False
    return finalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("paro", "gguf"), required=True)
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default="gfx1151")
    parser.add_argument("--components", nargs="+", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    components = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.components]
    first_provenance = validate_artifact_provenance(
        components[0][1].get("provenance") or {}, require_model=True
    )
    expected_warmups, expected_repetitions = _expected_protocol(
        engine=args.engine, platform=args.platform
    )
    assembly_provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(first_provenance["configured_backend"]),
        resolved_backend=str(first_provenance["resolved_backend"]),
        target_arch=str(first_provenance["target_arch"]),
        device_name=first_provenance.get("device_name"),
        model_path=str(first_provenance["model_path"]),
        model_revision=first_provenance.get("model_revision"),
        quant=str(first_provenance["quant"]),
        kv_dtype=str(first_provenance["kv_dtype"]),
        command=tuple(sys.argv),
        build_profile="readme_per_workload_rollup",
        timing_protocol=(
            f"{args.platform} six independent right-sized resident sessions; "
            f"{expected_warmups} warmup(s) plus {expected_repetitions} measured runs per shape"
        ),
        warmups=expected_warmups,
        repetitions=expected_repetitions,
        profiler={"enabled": False, "reason": "topline host-wall sweep"},
        rocm_version=first_provenance.get("rocm_version"),
        hipcc_version=first_provenance.get("hipcc_version"),
    )
    output = _merge_component_payloads(
        components,
        engine=args.engine,
        provenance=first_provenance,
        platform=args.platform,
    )
    output = _finalize_rollup(
        output,
        assembly_provenance=assembly_provenance,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.json),
                "status": output["status"],
                "performance_claim": output["performance_claim"],
                "workloads": output["workloads"],
            },
            indent=2,
        )
    )
    return 0 if output["performance_claim"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
