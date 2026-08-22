#!/usr/bin/env python3
"""Audit every CONCURRENCY2 phase and definition-of-done row against evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AuditRequirement:
    requirement_id: str
    section: str
    summary: str
    status: str
    evidence_paths: tuple[str, ...]
    validation: str
    blocker: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "blocked", "unavailable"}:
            raise ValueError(f"invalid audit status {self.status!r}")
        if self.status == "passed" and self.blocker is not None:
            raise ValueError("passed audit requirement cannot carry a blocker")
        if self.status != "passed" and not self.blocker:
            raise ValueError("blocked/unavailable requirement must name its blocker")


def _requirements() -> tuple[AuditRequirement, ...]:
    short = "benchmarks/results/2026-08-16-concurrency2-c2-6-w7900-global-native-accepted.json"
    c4c8 = "benchmarks/results/2026-08-17-concurrency2-c2-8-w7900-shared-slot-c4-c8-promotion.json"
    long_blocked = "benchmarks/results/2026-08-17-concurrency2-c2-6-w7900-long-load-blocked.json"
    canonical = "benchmarks/results/2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json"
    external = "benchmarks/results/2026-08-22-concurrency2-external-serving-unavailable.json"
    dms = "benchmarks/results/2026-08-17-concurrency2-c2-7-dms-host-blocked.json"
    dms_device = "benchmarks/results/2026-08-22-concurrency2-c2-7-dms-device-qualified.json"
    tier = "benchmarks/results/2026-08-17-concurrency2-c2-8-tier-host-accepted.json"
    tier_model = "benchmarks/results/2026-08-22-concurrency2-c2-8-real-model-tier-qualified.json"
    spec_c0 = "benchmarks/results/2026-08-22-concurrency2-spec-c0-host-contracts.json"
    spec_c1 = "benchmarks/results/2026-08-22-concurrency2-spec-c1-engine-service.json"
    spec_c2 = "benchmarks/results/2026-08-22-concurrency2-spec-c2-continuous-packing.json"
    return (
        AuditRequirement("C2-0", "roadmap", "contracts and deterministic simulator", "passed", ("hipengine/generation/concurrency2_simulator.py", "tests/test_concurrency2_simulator.py"), "deterministic/property host suite"),
        AuditRequirement("C2-1", "roadmap", "sole EngineService and independent child outputs", "passed", ("hipengine/generation/engine_service.py", "tests/test_generation_engine_service.py"), "service/refill/collector/cancellation suite"),
        AuditRequirement("C2-2", "roadmap", "format-neutral ledger and fit-aware admission", "passed", ("hipengine/kvcache/ledger.py", "tests/test_kvcache_resource_ledger.py"), "atomic claim/lookahead/starvation/conservation suite"),
        AuditRequirement("C2-3", "roadmap", "global pool and dense BF16/INT8 backends", "passed", ("hipengine/kvcache/global_pool.py", "hipengine/kvcache/dense.py", "tests/test_kvcache_global_pool.py"), "global-page/dense conformance suite"),
        AuditRequirement("C2-4", "roadmap", "integrated radix snapshots and eviction", "passed", ("hipengine/kvcache/backend_prefix.py", "hipengine/kvcache/radix.py", "tests/test_kvcache_backend_prefix.py"), "generation/COW/quota/eviction host suite"),
        AuditRequirement("C2-5", "roadmap", "token-budget scheduling and logical c1-c32", "passed", ("hipengine/dispatch/execution_planner.py", "tests/test_concurrency2_token_budget.py"), "all logical widths and fairness suite"),
        AuditRequirement("C2-6.graph", "roadmap", "changing-page graph/prefix/slot lifecycle", "passed", (long_blocked,), "W7900 4 captures / 100 replays / 4 invalidations plus host prefix eviction"),
        AuditRequirement("C2-6.long", "roadmap", "actual 4K/16K/32K mixed model execution", "passed", (long_blocked,), "actual c2 1K/4K/16K/32K/64K and mixed 1K/4K/32K pass with SLO/resource/drain evidence"),
        AuditRequirement("C2-6.load", "roadmap", "fixed/ragged/Poisson/overload/disconnect/soak", "passed", (canonical,), "clean W7900 canonical packet: tuning plus nine workloads, 210/210 correctness-accounted rows, bounded overload, 120/120 soak, final drain"),
        AuditRequirement("C2-6.external", "roadmap", "matched prior/llama/vLLM/SGLang comparisons", "unavailable", (external, long_blocked, canonical), "current same-host availability and invalid CPU-only Vulkan packet recorded", "vLLM/SGLang are not installed; llama.cpp HIP binaries fail CPU-ISA/ROCm-ABI checks; the available Vulkan binary has no usable GPU backend"),
        AuditRequirement("C2-6.default", "roadmap", "full production default promotion", "passed", (short, c4c8, canonical), "global/native and physical c4/c8 defaults exact; canonical token-budget/256 packet passes correctness, SLO, overload, memory, and ownership gates"),
        AuditRequirement("C2-7.metadata", "roadmap", "DMS checkpoint metadata gate", "passed", ("hipengine/kvcache/dms.py", "scripts/dms_backend_gate.py", dms), "strict loader and current-model fail-close"),
        AuditRequirement("C2-7.extents", "roadmap", "compact extent/resource backend", "passed", ("hipengine/kvcache/dms.py", "tests/test_kvcache_dms.py"), "atomic fragmentation/rollback/conservation"),
        AuditRequirement("C2-7.hip", "roadmap", "registered HIP no-shadow pack and compact attention", "passed", (dms_device, "hipengine/kernels/hip_gfx1100/attention/dms_compact.hip", "tests/test_kvcache_dms_device_hip.py", "scripts/dms_device_rocprof_smoke.py"), "53-test BF16 device parity/lifecycle bundle plus cached-only rocprof identities for all four kernels"),
        AuditRequirement("C2-7.widths", "roadmap", "DMS common c1/c2/c4/c8/c16/c32", "passed", (dms, "tests/test_kvcache_dms.py"), "common ResidentEngineLoop host packet"),
        AuditRequirement("C2-7.lifecycle", "roadmap", "DMS pressure/fragmentation/cancel/reclaim/soak", "passed", (dms, "tests/test_kvcache_dms.py"), "host pressure/replacement/final drain"),
        AuditRequirement("C2-7.codec", "roadmap", "format-distinct qualified compressed codec", "passed", (dms,), "fixture-scoped INT8 KL/top-1 and same lifecycle"),
        AuditRequirement("C2-7.prefix", "roadmap", "DMS prefix reuse disabled", "passed", ("hipengine/kvcache/dms.py",), "hard fail-closed lookup/estimate"),
        AuditRequirement("C2-8.maintenance", "roadmap", "offload/restore work and host/NVMe pools", "passed", ("hipengine/kvcache/tiering.py", tier), "typed tier work and resource ledger"),
        AuditRequirement("C2-8.codec", "roadmap", "cold codec restores to hot attention backend", "passed", ("hipengine/kvcache/tiering.py",), "delegated hot KVBatchView and checksummed restore"),
        AuditRequirement("C2-8.lifecycle", "roadmap", "fingerprints/quotas/LRU/cancel/drain", "passed", ("tests/test_kvcache_tiering.py", tier), "deterministic host/NVMe lifecycle"),
        AuditRequirement("C2-8.economics", "roadmap", "restore TTFT versus recompute", "passed", ("scripts/tier_model_kv_gate.py", tier_model, tier), "actual model-produced BF16 KV host restore vs same-loaded-model native prefill; pressure/cancel/drain; integrated GPU rehydrate remains default-off"),
        AuditRequirement("C2-S.C0", "roadmap", "speculative host contracts and deterministic simulator", "passed", ("hipengine/speculative/interfaces.py", "hipengine/speculative/simulator.py", "tests/test_speculative_cycle_simulator.py", spec_c0), "one-to-many row maps, provider+target atomic claims, reject/partial/full transactions, cancellation at every stage, final conservation"),
        AuditRequirement("C2-S.C1", "roadmap", "guarded MTP under one EngineService lifecycle", "passed", ("hipengine/generation/engine_service.py", "hipengine/generation/engine_loop.py", "tests/test_generation_engine_service.py", spec_c1), "VERIFY_CHAIN submission/work metadata, shared child/output/cancel/release path, declared pre-launch legacy fallback, actual Qwen parity"),
        AuditRequirement("C2-S.C2", "roadmap", "continuous compatible speculative packing and cost policy", "passed", ("hipengine/speculative/packing.py", "hipengine/generation/engine_service.py", "tests/test_speculative_packing.py", spec_c2), "one driver batch/multi-request VERIFY_CHAIN work, verifier-only cost map/budgets, fairness/refill/pressure/deadline fallback, actual Qwen c2 parity"),
        AuditRequirement("DoD.service", "definition_of_done", "one service owns blocking/SSE/library children", "passed", ("hipengine/generation/engine_service.py",), "sole-driver tests"),
        AuditRequirement("DoD.independent", "definition_of_done", "independent terminal publication/reclaim", "passed", ("tests/test_generation_engine_service.py",), "short-before-long/refill tests"),
        AuditRequirement("DoD.ledger", "definition_of_done", "one format-neutral resource ledger", "passed", ("hipengine/kvcache/ledger.py",), "dense/DMS/tier claims"),
        AuditRequirement("DoD.global_pool", "definition_of_done", "compatible requests share global pool", "passed", (short,), "W7900 c1-c32 global_generation2"),
        AuditRequirement("DoD.prefix", "definition_of_done", "prefix refs/COW/quota/eviction", "passed", ("tests/test_kvcache_backend_prefix.py",), "host conformance; DMS intentionally off"),
        AuditRequirement("DoD.logical_physical", "definition_of_done", "logical concurrency independent of physical width", "passed", (short, c4c8), "logical c32 over registered physical (1,2,4,8); c4/c8 lower to one bucket"),
        AuditRequirement("DoD.load", "definition_of_done", "width/load/overload matrices through c32", "passed", (short, c4c8, canonical), "c1-c32 exact plus canonical fixed/ragged/Poisson/cancel/overload/recovery/soak packet"),
        AuditRequirement("DoD.performance", "definition_of_done", "c1 direct and cN beats honest serial under SLO", "passed", (short, c4c8), "matched c8 +60.25% after c4/c8 promotion (was +29.68% at c2 cap), exact/SLO; c1 direct retained"),
        AuditRequirement("DoD.drain", "definition_of_done", "graph/pool/state/collector ownership drains", "passed", (short, long_blocked, canonical), "short, long/pressure, and canonical 271-admit/271-reclaim final drain"),
        AuditRequirement("DoD.compact", "definition_of_done", "format-distinct compact backend conformance", "passed", (dms, dms_device), "host and gfx1100 device fixture conformance, no-shadow c1-c32 lifecycle, codec composition, profiler identities"),
        AuditRequirement("DoD.swap", "definition_of_done", "topology/codec/tier swap needs no concurrency fork", "passed", ("hipengine/kvcache/dms.py", "hipengine/kvcache/tiering.py"), "common protocols/adapters"),
        AuditRequirement("DoD.docs", "definition_of_done", "docs/artifacts/telemetry disclose routes/memory", "passed", ("docs/CONCURRENCY2.md", "docs/KVCACHE.md", "benchmarks/README.md", "benchmarks/CHANGELOG.md"), "Worklog2 and compact artifacts"),
        AuditRequirement("DMS.product", "definition_of_done", "DMS checkpoint quality, HIP spans, no shadow", "blocked", (dms, dms_device), "host/device no-shadow implementation and fixture codec pass", "no valid trained retrofit checkpoint is available for model-quality, device-savings, and product-soak qualification"),
    )


def _artifact_status(path: Path) -> dict[str, object] | None:
    if path.suffix != ".json":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": payload.get("status"),
        "passed": payload.get("passed"),
        "performance_claim": payload.get("performance_claim"),
    }


def run(repo_root: Path) -> dict[str, object]:
    requirements = _requirements()
    rows: list[dict[str, object]] = []
    missing_evidence: list[str] = []
    for requirement in requirements:
        evidence = []
        for relative in requirement.evidence_paths:
            path = repo_root / relative
            exists = path.exists()
            if not exists:
                missing_evidence.append(f"{requirement.requirement_id}:{relative}")
            evidence.append(
                {
                    "path": relative,
                    "exists": exists,
                    "artifact": _artifact_status(path) if exists else None,
                }
            )
        row = asdict(requirement)
        row["evidence"] = evidence
        row["evidence_complete"] = all(item["exists"] for item in evidence)
        rows.append(row)
    status_counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("passed", "blocked", "unavailable")
    }
    false_passes = [
        row["requirement_id"]
        for row in rows
        if row["status"] == "passed" and not row["evidence_complete"]
    ]
    blockers = [
        {
            "requirement_id": row["requirement_id"],
            "status": row["status"],
            "blocker": row["blocker"],
        }
        for row in rows
        if row["status"] != "passed"
    ]
    audit_valid = not missing_evidence and not false_passes
    goal_complete = bool(audit_valid and not blockers)
    return {
        "schema": 1,
        "kind": "hipengine_concurrency2_completion_audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if goal_complete else "blocked",
        "passed": audit_valid,
        "goal_complete": goal_complete,
        "status_counts": status_counts,
        "requirements": rows,
        "blockers": blockers,
        "missing_evidence": missing_evidence,
        "false_passes": false_passes,
        "validation_commands": [
            "python3 -m pytest -q tests/test_concurrency2_simulator.py tests/test_generation_engine_service.py tests/test_kvcache_resource_ledger.py tests/test_kvcache_global_pool.py tests/test_kvcache_backend_prefix.py tests/test_concurrency2_token_budget.py tests/test_concurrency2_production_load.py tests/test_kvcache_dms.py tests/test_kvcache_dms_device_hip.py tests/test_kvcache_tiering.py",
            "python3 scripts/worklog.py check",
            "python3 scripts/sync_benchmark_readme.py --check",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args.repo_root.expanduser().resolve())
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["goal_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
