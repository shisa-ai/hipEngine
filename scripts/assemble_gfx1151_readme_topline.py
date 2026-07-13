#!/usr/bin/env python3
"""Gate and assemble a four-column gfx1100/gfx1151 README model sweep."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import validate_artifact_provenance  # noqa: E402
from scripts.merge_readme_sweep_components import (  # noqa: E402
    STANDARD_WORKLOADS,
    _expected_protocol,
)


COLUMNS = (
    "hipengine_paro",
    "hipengine_gguf",
    "llamacpp_hip",
    "llamacpp_vulkan",
)
COLUMN_LABELS = {
    "hipengine_paro": "hipEngine PARO",
    "hipengine_gguf": "hipEngine GGUF",
    "llamacpp_hip": "llama.cpp HIP",
    "llamacpp_vulkan": "llama.cpp Vulkan",
}
PLATFORM_CONFIG = {
    "gfx1100": {
        "hardware": "AMD Radeon Pro W7900, gfx1100",
        "device_marker": "w7900",
        "memory_domain": "vram",
        "default_correctness": {
            "gguf_external_and_state_oracle": "benchmarks/results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json",
            "paro_fixture_gate": "benchmarks/results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json",
        },
    },
    "gfx1151": {
        "hardware": "AMD Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151",
        "device_marker": "8060s",
        "memory_domain": "gtt",
        "default_correctness": {
            "gguf_external_and_state_oracle": "benchmarks/results/2026-07-11-sol-g1-gfx1151-gguf-eager-p512-d4.json",
            "paro_fixture_gate": "benchmarks/results/2026-07-11-sol-p1-gfx1151-paro-c1-c8-exact-catalog.json",
        },
    },
}


def _platform_config(platform: str) -> Mapping[str, Any]:
    try:
        return PLATFORM_CONFIG[platform]
    except KeyError as exc:
        raise ValueError(f"unsupported README topline platform {platform!r}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _metric_stats_pass(stats: Mapping[str, Any], *, expected_count: int) -> bool:
    count = stats.get("count")
    median = stats.get("median")
    stdev = stats.get("stdev")
    return bool(
        type(count) is int
        and count == expected_count
        and _positive_finite(median)
        and isinstance(stdev, (int, float))
        and math.isfinite(float(stdev))
        and float(stdev) <= 0.05 * float(median)
    )


def _llama_sample_stats(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = (record.get("llamacpp_record") or {}).get("samples_ts")
    samples = list(raw) if isinstance(raw, list) else []
    valid = len(samples) == 5 and all(_positive_finite(value) for value in samples)
    if not valid:
        return {
            "count": len(samples),
            "median": None,
            "mean": None,
            "stdev": None,
            "variance_gate_passed": False,
        }
    values = [float(value) for value in samples]
    median = statistics.median(values)
    stdev = statistics.stdev(values)
    return {
        "count": len(values),
        "median": median,
        "mean": statistics.mean(values),
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "variance_gate_passed": stdev <= 0.05 * median,
    }


def _model_identity(provenance: Mapping[str, Any]) -> tuple[Any, ...]:
    fingerprint = provenance.get("model_fingerprint") or {}
    return (
        fingerprint.get("algorithm"),
        fingerprint.get("value"),
        fingerprint.get("size_bytes"),
    )


def _validate_hipengine_rollup(
    payload: Mapping[str, Any], *, engine: str, platform: str = "gfx1151"
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, bool]]:
    _platform_config(platform)
    expected_warmups, expected_repetitions = _expected_protocol(
        engine=engine, platform=platform
    )
    provenance = validate_artifact_provenance(payload.get("provenance") or {}, require_model=True)
    structural = bool(
        payload.get("kind") == f"{platform}_readme_model_sweep_rollup"
        and payload.get("engine") == engine
        and payload.get("status") == "accepted_topline"
        and payload.get("performance_claim") is True
        and payload.get("warmup_runs") == expected_warmups
        and payload.get("measured_runs") == expected_repetitions
        and tuple(payload.get("workloads") or ()) == STANDARD_WORKLOADS
    )
    correctness = payload.get("correctness") or {}
    correctness_passed = bool(
        correctness.get("passed") is True
        and correctness.get("all_measured_final_logits_finite") is True
        and correctness.get("all_workload_final_ids_stable") is True
        and correctness.get("all_workload_variance_gates_passed") is True
        and correctness.get("all_component_provenance_clean") is True
    )
    rows: dict[str, dict[str, float]] = {}
    stats_ok = True
    summaries = payload.get("summary_by_workload") or {}
    for workload in STANDARD_WORKLOADS:
        summary = summaries.get(workload) or {}
        prefill = summary.get("prefill_tok_s") or {}
        decode = summary.get("decode_tok_s") or {}
        peak = summary.get("tracked_peak_allocated_gib") or {}
        row_ok = (
            _metric_stats_pass(prefill, expected_count=expected_repetitions)
            and _metric_stats_pass(decode, expected_count=expected_repetitions)
            and _metric_stats_pass(peak, expected_count=expected_repetitions)
            and summary.get("final_token_ids_stable") is True
        )
        stats_ok = stats_ok and row_ok
        rows[workload] = {
            "prefill_tok_s": float(prefill.get("median") or 0.0),
            "decode_tok_s": float(decode.get("median") or 0.0),
            "peak_gib": float(peak.get("median") or 0.0),
        }
    gates = {
        "structural": structural,
        "correctness": correctness_passed,
        "statistics": stats_ok,
        "clean": provenance.get("dirty") is False,
        "target_platform": provenance.get("target_arch") == platform,
    }
    return provenance, rows, gates


def _validate_llamacpp(
    payload: Mapping[str, Any], *, backend: str, platform: str = "gfx1151"
) -> tuple[dict[str, Any], dict[str, dict[str, float]], list[dict[str, Any]], dict[str, bool]]:
    config = _platform_config(platform)
    provenance = validate_artifact_provenance(payload.get("provenance") or {}, require_model=True)
    expected_backend = f"llamacpp_{backend}"
    common = payload.get("common_args") or {}
    structural = bool(
        payload.get("backend") == expected_backend
        and tuple(payload.get("workloads_requested") or ()) == STANDARD_WORKLOADS
        and common.get("ngl") == 99
        and common.get("flash_attn") == 1
        and common.get("cache_type_k") == "f16"
        and common.get("cache_type_v") == "f16"
        and common.get("repetitions") == 5
        and common.get("no_warmup") is False
        and provenance.get("repetitions") == 5
        and provenance.get("warmups") == 1
    )
    memory_scope = bool(
        payload.get("memory_domain") == config["memory_domain"]
        and isinstance(payload.get("poll_ms"), (int, float))
        and 0.0 < float(payload["poll_ms"]) <= 10.0
    )

    record_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    returncodes_ok = True
    phase_stats: list[dict[str, Any]] = []
    variance_ok = True
    for record in payload.get("phase_records") or []:
        key = (str(record.get("workload")), str(record.get("phase")))
        if key in record_map:
            structural = False
        record_map[key] = record
        returncodes_ok = returncodes_ok and record.get("returncode") == 0
        stats = _llama_sample_stats(record)
        variance_ok = variance_ok and stats["variance_gate_passed"]
        phase_stats.append({"workload": key[0], "phase": key[1], **stats})

    expected_keys = {
        (workload, phase)
        for workload in STANDARD_WORKLOADS
        for phase in ("prefill", "decode")
    }
    structural = structural and set(record_map) == expected_keys
    row_map = {
        str(row.get("workload")): row for row in payload.get("rows") or []
    }
    structural = structural and tuple(row_map) == STANDARD_WORKLOADS

    rows: dict[str, dict[str, float]] = {}
    values_ok = True
    for workload in STANDARD_WORKLOADS:
        prefill_stats = _llama_sample_stats(record_map.get((workload, "prefill"), {}))
        decode_stats = _llama_sample_stats(record_map.get((workload, "decode"), {}))
        row = row_map.get(workload) or {}
        peak = row.get("peak_vram_gib")
        row_ok = (
            prefill_stats["variance_gate_passed"]
            and decode_stats["variance_gate_passed"]
            and _positive_finite(peak)
        )
        values_ok = values_ok and row_ok
        rows[workload] = {
            "prefill_tok_s": float(prefill_stats["median"] or 0.0),
            "decode_tok_s": float(decode_stats["median"] or 0.0),
            "peak_gib": float(peak or 0.0),
        }

    gpu_info = str(payload.get("gpu_info") or "").lower()
    gates = {
        "structural": structural,
        "returncodes": returncodes_ok,
        "variance": variance_ok and values_ok,
        "clean": provenance.get("dirty") is False,
        "target_platform": provenance.get("target_arch") == platform,
        "device_identity": str(config["device_marker"]) in gpu_info,
        "memory_scope": memory_scope,
        "build_identity": bool(payload.get("build_commit") and payload.get("build_number")),
    }
    return provenance, rows, phase_stats, gates


def _assemble_topline(
    sources: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    platform: str = "gfx1151",
    linked_correctness: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    config = _platform_config(platform)
    if set(sources) != set(COLUMNS):
        raise ValueError(f"expected sources {list(COLUMNS)}, received {sorted(sources)}")

    paro_prov, paro_rows, paro_gates = _validate_hipengine_rollup(
        sources["hipengine_paro"][1], engine="paro", platform=platform
    )
    gguf_prov, gguf_rows, gguf_gates = _validate_hipengine_rollup(
        sources["hipengine_gguf"][1], engine="gguf", platform=platform
    )
    llama_hip_prov, llama_hip_rows, llama_hip_stats, llama_hip_gates = _validate_llamacpp(
        sources["llamacpp_hip"][1], backend="hip", platform=platform
    )
    llama_vk_prov, llama_vk_rows, llama_vk_stats, llama_vk_gates = _validate_llamacpp(
        sources["llamacpp_vulkan"][1], backend="vulkan", platform=platform
    )
    provenances = (paro_prov, gguf_prov, llama_hip_prov, llama_vk_prov)

    hip_rollups_accepted = all(paro_gates.values()) and all(gguf_gates.values())
    llama_all_phase_returncodes_zero = (
        llama_hip_gates["returncodes"] and llama_vk_gates["returncodes"]
    )
    llama_all_phase_variance_passed = (
        llama_hip_gates["variance"] and llama_vk_gates["variance"]
    )
    llama_structural = llama_hip_gates["structural"] and llama_vk_gates["structural"]
    all_clean = all(provenance.get("dirty") is False for provenance in provenances)
    all_target_platform = all(
        provenance.get("target_arch") == platform for provenance in provenances
    )
    same_harness_commit = len(
        {provenance.get("hipengine_commit") for provenance in provenances}
    ) == 1
    gguf_model_identity = _model_identity(gguf_prov)
    gguf_identity_matches = (
        gguf_model_identity == _model_identity(llama_hip_prov)
        and gguf_model_identity == _model_identity(llama_vk_prov)
    )
    llama_memory_scope = (
        llama_hip_gates["memory_scope"] and llama_vk_gates["memory_scope"]
    )
    llama_builds_identified = (
        llama_hip_gates["build_identity"]
        and llama_vk_gates["build_identity"]
        and llama_hip_gates["device_identity"]
        and llama_vk_gates["device_identity"]
    )
    gates = {
        "hipengine_rollups_accepted": hip_rollups_accepted,
        "llamacpp_structural_contract_passed": llama_structural,
        "llamacpp_all_phase_returncodes_zero": llama_all_phase_returncodes_zero,
        "llamacpp_all_phase_variance_passed": llama_all_phase_variance_passed,
        "all_component_provenance_clean": all_clean,
        f"all_components_target_{platform}": all_target_platform,
        "all_components_same_harness_commit": same_harness_commit,
        "gguf_model_identity_matches": gguf_identity_matches,
        f"llamacpp_{config['memory_domain']}_memory_scope_passed": llama_memory_scope,
        "llamacpp_builds_and_device_identified": llama_builds_identified,
    }
    accepted = all(gates.values())
    normalized = {
        "hipengine_paro": paro_rows,
        "hipengine_gguf": gguf_rows,
        "llamacpp_hip": llama_hip_rows,
        "llamacpp_vulkan": llama_vk_rows,
    }

    tables: dict[str, list[dict[str, Any]]] = {}
    for metric in ("prefill_tok_s", "decode_tok_s", "peak_gib"):
        tables[metric] = [
            {
                "workload": workload,
                **{
                    column: normalized[column][workload][metric]
                    for column in COLUMNS
                },
            }
            for workload in STANDARD_WORKLOADS
        ]

    component_records = {}
    for column in COLUMNS:
        path, payload = sources[column]
        component_records[column] = {
            "path": str(path),
            "name": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "component_status": payload.get("status"),
            "component_performance_claim": payload.get("performance_claim"),
        }

    llama_hip = sources["llamacpp_hip"][1]
    llama_vk = sources["llamacpp_vulkan"][1]
    return {
        "schema": 1,
        "kind": f"{platform}_readme_four_engine_topline",
        "status": "accepted_topline" if accepted else "rejected_topline_gate",
        "performance_claim": accepted,
        "date": dt.date.today().isoformat(),
        "hardware": config["hardware"],
        "measured_hipengine_commit": paro_prov["hipengine_commit"],
        "workloads": list(STANDARD_WORKLOADS),
        "columns": [
            {"key": column, "label": COLUMN_LABELS[column]} for column in COLUMNS
        ],
        "protocol": {
            "hipengine": (
                "one right-sized resident session per workload; "
                + (
                    "PARO uses 2 discarded warmups + median of 5 measured repetitions; "
                    "GGUF uses 1 discarded warmup + median of 3 measured repetitions"
                    if platform == "gfx1151"
                    else "PARO and GGUF use 2 discarded warmups + median of 5 measured repetitions"
                )
            ),
            "llamacpp": "split prefill/decode llama-bench; 1 internal warmup + median of 5 samples per phase",
            "prefill_decode_units": "tokens per second",
            "hipengine_memory": "tracked allocator high-water GiB",
            "llamacpp_memory": f"whole-device amdgpu {str(config['memory_domain']).upper()}-used peak GiB sampled every 10 ms",
        },
        "models": {
            "hipengine_paro": {
                "quant": paro_prov["quant"],
                "fingerprint": paro_prov["model_fingerprint"],
            },
            "gguf_columns": {
                "quant": gguf_prov["quant"],
                "kv_dtype": {
                    "hipengine": gguf_prov["kv_dtype"],
                    "llamacpp": llama_hip_prov["kv_dtype"],
                },
                "fingerprint": gguf_prov["model_fingerprint"],
            },
        },
        "llamacpp_builds": {
            "hip": {
                "commit": llama_hip.get("build_commit"),
                "build_number": llama_hip.get("build_number"),
                "gpu_info": llama_hip.get("gpu_info"),
            },
            "vulkan": {
                "commit": llama_vk.get("build_commit"),
                "build_number": llama_vk.get("build_number"),
                "gpu_info": llama_vk.get("gpu_info"),
            },
        },
        "gates": {**gates, "all_passed": accepted},
        "tables": tables,
        "llamacpp_phase_statistics": {
            "hip": llama_hip_stats,
            "vulkan": llama_vk_stats,
        },
        "components": component_records,
        "component_provenance": {
            "hipengine_paro": paro_prov,
            "hipengine_gguf": gguf_prov,
            "llamacpp_hip": llama_hip_prov,
            "llamacpp_vulkan": llama_vk_prov,
        },
        "linked_correctness": dict(
            linked_correctness or config["default_correctness"]
        ),
        "notes": [
            "The llama.cpp component tools intentionally emit performance_claim=false; this top-level artifact is the promotion boundary after checking their samples, return codes, build/model/device identity, and clean harness provenance.",
            "The PARO column uses W4 PARO/BF16 KV. The other three columns use the same Q4_K_M GGUF; hipEngine uses BF16 KV and llama.cpp uses f16 KV.",
            f"Peak-memory numbers have different scopes: hipEngine is tracked owned allocation, while llama.cpp is absolute whole-device {str(config['memory_domain']).upper()} usage. Compare trends within a column; do not interpret small cross-column deltas as allocator efficiency.",
        ],
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    headings = (
        ("prefill_tok_s", "Prefill tok/s"),
        ("decode_tok_s", "Decode tok/s"),
        ("peak_gib", "Peak memory GiB"),
    )
    lines: list[str] = []
    for metric, heading in headings:
        if lines:
            lines.append("")
        lines.extend(
            [
                f"#### {heading}",
                "",
                "| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in payload["tables"][metric]:
            values = " | ".join(f"{float(row[column]):.3f}" for column in COLUMNS)
            lines.append(f"| {row['workload']} | {values} |")
    return "\n".join(lines) + "\n"


def _git_assembly_record(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=repo_root, text=True).strip()

    staged_dirty = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False
    ).returncode != 0
    unstaged_dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=repo_root, check=False
    ).returncode != 0
    untracked = run("git", "ls-files", "--others", "--exclude-standard").splitlines()
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "hipengine_commit": run("git", "rev-parse", "HEAD"),
        "staged_dirty": staged_dirty,
        "unstaged_dirty": unstaged_dirty,
        "untracked_count": len(untracked),
        "dirty": bool(staged_dirty or unstaged_dirty or untracked),
        "command": list(sys.argv),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=tuple(PLATFORM_CONFIG), default="gfx1151")
    parser.add_argument("--hipengine-paro", type=Path, required=True)
    parser.add_argument("--hipengine-gguf", type=Path, required=True)
    parser.add_argument("--llamacpp-hip", type=Path, required=True)
    parser.add_argument("--llamacpp-vulkan", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--gguf-correctness")
    parser.add_argument("--paro-correctness")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = {
        "hipengine_paro": args.hipengine_paro,
        "hipengine_gguf": args.hipengine_gguf,
        "llamacpp_hip": args.llamacpp_hip,
        "llamacpp_vulkan": args.llamacpp_vulkan,
    }
    sources = {
        column: (path, json.loads(path.read_text(encoding="utf-8")))
        for column, path in paths.items()
    }
    linked_correctness = None
    if args.gguf_correctness or args.paro_correctness:
        if not args.gguf_correctness or not args.paro_correctness:
            raise ValueError("both --gguf-correctness and --paro-correctness are required together")
        linked_correctness = {
            "gguf_external_and_state_oracle": args.gguf_correctness,
            "paro_fixture_gate": args.paro_correctness,
        }
    payload = _assemble_topline(
        sources,
        platform=args.platform,
        linked_correctness=linked_correctness,
    )
    assembly = _git_assembly_record(REPO_ROOT)
    payload["assembly"] = assembly
    if assembly["dirty"]:
        payload["gates"]["assembly_provenance_clean"] = False
        payload["gates"]["all_passed"] = False
        payload["status"] = "rejected_topline_gate"
        payload["performance_claim"] = False
    else:
        payload["gates"]["assembly_provenance_clean"] = True

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = _render_markdown(payload)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(
        json.dumps(
            {
                "output": str(args.json),
                "status": payload["status"],
                "performance_claim": payload["performance_claim"],
                "gates": payload["gates"],
            },
            indent=2,
        ),
        file=sys.stderr,
    )
    return 0 if payload["performance_claim"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
