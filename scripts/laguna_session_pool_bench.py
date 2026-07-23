#!/usr/bin/env python3
"""Measure fresh versus reset Laguna session setup and direct first-token wall."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import memory_stats
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_CACHE = Path(
    "/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.hipengine-repacked-v1"
)
DEFAULT_MODEL_SHA256 = "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
DEFAULT_TEMPLATE = ROOT / "tests/fixtures/laguna_poolside_v1_template.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--case", default="no_thinking")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--direct-gguf", action="store_true")
    parser.add_argument("--safety-reserve-gib", type=float, default=8.0)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prompt_case(path: Path, name: str) -> tuple[tuple[int, ...], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [case for case in payload["cases"] if str(case.get("name")) == name]
    if len(matches) != 1:
        raise ValueError(f"expected one prompt case {name!r}, found {len(matches)}")
    case = matches[0]
    token_ids = tuple(int(token) for token in case["token_ids"])
    if not token_ids:
        raise ValueError("session-pool benchmark prompt must contain tokens")
    return token_ids, case


def _progress(completed: int, total: int, spec) -> None:
    if completed == 1 or completed == total or completed % 100 == 0:
        print(
            f"load {completed}/{total}: {spec.source.name} ({spec.layout})",
            file=sys.stderr,
            flush=True,
        )


def _borrowing_session(
    owner: LagunaGGUFResidentSession,
    args: argparse.Namespace,
) -> LagunaGGUFResidentSession:
    assert owner.weights is not None
    return LagunaGGUFResidentSession(
        resident_weights=owner.weights,
        context_length=args.context_length,
        backend=args.backend,
        runtime=owner.runtime,
        compiler_version=_compiler_version(args.compiler_version_file),
        require_cached_build=args.require_cached_build,
        prefill_chunk_size=args.chunk_size,
    )


def _fresh_sample(
    owner: LagunaGGUFResidentSession,
    prompt_ids: tuple[int, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    session = _borrowing_session(owner, args)
    owner.runtime.device_synchronize()
    prepared_at = time.perf_counter()
    try:
        result = session.prefill(prompt_ids)
        first_token_at = time.perf_counter()
    finally:
        close_started = time.perf_counter()
        session.close()
        close_ms = (time.perf_counter() - close_started) * 1_000.0
    return {
        "mode": "fresh",
        "session_prepare_ms": (prepared_at - started) * 1_000.0,
        "prefill_ms": (first_token_at - prepared_at) * 1_000.0,
        "direct_ttft_ms": (first_token_at - started) * 1_000.0,
        "close_ms": close_ms,
        "first_token_id": int(result.next_token_id),
    }


def _pooled_sample(
    session: LagunaGGUFResidentSession,
    prompt_ids: tuple[int, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    session.reset_state()
    session.runtime.device_synchronize()
    prepared_at = time.perf_counter()
    result = session.prefill(prompt_ids)
    first_token_at = time.perf_counter()
    return {
        "mode": "pooled_reset",
        "session_prepare_ms": (prepared_at - started) * 1_000.0,
        "prefill_ms": (first_token_at - prepared_at) * 1_000.0,
        "direct_ttft_ms": (first_token_at - started) * 1_000.0,
        "close_ms": None,
        "first_token_id": int(result.next_token_id),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(rows)}
    for key in ("session_prepare_ms", "prefill_ms", "direct_ttft_ms"):
        values = [float(row[key]) for row in rows]
        result[key] = {
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "values": values,
        }
    close = [float(row["close_ms"]) for row in rows if row["close_ms"] is not None]
    if close:
        result["close_ms"] = {
            "median": statistics.median(close),
            "minimum": min(close),
            "maximum": max(close),
            "values": close,
        }
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmups < 1:
        raise ValueError("--warmups must be at least 1")
    if args.repetitions < 5:
        raise ValueError("--repetitions must be at least 5 for the promotion gate")
    prompt_ids, prompt_case = _prompt_case(args.template, args.case)
    if len(prompt_ids) > args.context_length:
        raise ValueError("prompt exceeds admitted context")

    provenance = collect_artifact_provenance(
        repo_root=ROOT,
        configured_backend=args.backend,
        resolved_backend=args.backend,
        target_arch=args.backend.removeprefix("hip_"),
        model_path=args.model,
        quant="Q4_K_M mixed GGUF v3",
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        build_profile="laguna_session_pool",
        timing_protocol="alternating_synchronized_fresh_session_vs_reset_session",
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    runtime = get_hip_runtime()
    tracked_before = memory_stats()
    owner: LagunaGGUFResidentSession | None = None
    pooled: LagunaGGUFResidentSession | None = None
    rows: list[dict[str, Any]] = []
    load_started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=args.context_length,
            backend=args.backend,
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            safety_reserve_nbytes=int(args.safety_reserve_gib * 2**30),
            progress=_progress,
            repacked_cache=None if args.direct_gguf else args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=args.chunk_size,
        )
        load_seconds = time.perf_counter() - load_started
        pooled = _borrowing_session(owner, args)
        runtime.device_synchronize()

        for _ in range(args.warmups):
            warm_fresh = _fresh_sample(owner, prompt_ids, args)
            warm_pooled = _pooled_sample(pooled, prompt_ids)
            if warm_fresh["first_token_id"] != warm_pooled["first_token_id"]:
                raise RuntimeError("fresh/reset warmup first token mismatch")

        for repetition in range(args.repetitions):
            actions: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
                (
                    ("fresh", lambda: _fresh_sample(owner, prompt_ids, args)),
                    ("pooled_reset", lambda: _pooled_sample(pooled, prompt_ids)),
                )
                if repetition % 2 == 0
                else (
                    ("pooled_reset", lambda: _pooled_sample(pooled, prompt_ids)),
                    ("fresh", lambda: _fresh_sample(owner, prompt_ids, args)),
                )
            )
            for expected_mode, action in actions:
                sample = action()
                sample["repetition"] = repetition
                sample["order_index"] = len(rows)
                if sample["mode"] != expected_mode:
                    raise RuntimeError("session-pool benchmark action/mode mismatch")
                rows.append(sample)
                print(
                    f"rep={repetition} mode={sample['mode']} "
                    f"prepare={sample['session_prepare_ms']:.3f} ms "
                    f"prefill={sample['prefill_ms']:.3f} ms "
                    f"ttft={sample['direct_ttft_ms']:.3f} ms",
                    file=sys.stderr,
                    flush=True,
                )
        owner_resident_nbytes = int(owner.resident_nbytes)
        pooled_resident_nbytes = int(pooled.resident_nbytes)
    finally:
        if pooled is not None:
            pooled.close()
        if owner is not None:
            owner.close()
    tracked_after = memory_stats()
    recovered = bool(
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )

    fresh = [row for row in rows if row["mode"] == "fresh"]
    pooled_rows = [row for row in rows if row["mode"] == "pooled_reset"]
    fresh_by_rep = {int(row["repetition"]): row for row in fresh}
    pooled_by_rep = {int(row["repetition"]): row for row in pooled_rows}
    paired = [
        {
            "repetition": repetition,
            "fresh_prepare_ms": float(fresh_by_rep[repetition]["session_prepare_ms"]),
            "pooled_prepare_ms": float(pooled_by_rep[repetition]["session_prepare_ms"]),
            "prepare_saved_ms": float(
                fresh_by_rep[repetition]["session_prepare_ms"]
                - pooled_by_rep[repetition]["session_prepare_ms"]
            ),
            "fresh_ttft_ms": float(fresh_by_rep[repetition]["direct_ttft_ms"]),
            "pooled_ttft_ms": float(pooled_by_rep[repetition]["direct_ttft_ms"]),
        }
        for repetition in sorted(fresh_by_rep)
    ]
    fresh_summary = _summary(fresh)
    pooled_summary = _summary(pooled_rows)
    exact = len({int(row["first_token_id"]) for row in rows}) == 1
    every_setup_improved = all(row["prepare_saved_ms"] > 0.0 for row in paired)
    median_fresh_ttft = float(fresh_summary["direct_ttft_ms"]["median"])
    median_pooled_ttft = float(pooled_summary["direct_ttft_ms"]["median"])
    ttft_non_regressive = median_pooled_ttft <= median_fresh_ttft
    passed = bool(exact and every_setup_improved and ttft_non_regressive and recovered)

    return {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_laguna_session_pool_benchmark",
        "status": "accepted" if passed else "rejected",
        "pass": passed,
        "performance_claim": passed,
        "performance_claim_scope": (
            "direct synchronized target first-token wall; resident model load excluded; "
            "fresh borrowing session versus reset borrowing session"
        ),
        "provenance": provenance,
        "model": {
            "path": str(args.model.resolve()),
            "sha256": args.model_sha256,
            "quant": "Q4_K_M mixed GGUF v3",
            "repacked_cache": (
                None if args.direct_gguf else str(args.repacked_cache.resolve())
            ),
        },
        "platform": {
            "backend": args.backend,
            "target_arch": args.backend.removeprefix("hip_"),
            "device_name": provenance["device_name"],
            "machine": platform.machine(),
        },
        "protocol": {
            "template": str(args.template.resolve()),
            "template_sha256": _sha256_bytes(args.template.read_bytes()),
            "case": args.case,
            "rendered_sha256": _sha256_bytes(
                str(prompt_case["rendered"]).encode("utf-8")
            ),
            "prompt_tokens": len(prompt_ids),
            "context_length": args.context_length,
            "prefill_chunk_size": args.chunk_size,
            "warmups_per_mode": args.warmups,
            "repetitions_per_mode": args.repetitions,
            "order": "fresh/reset alternating by repetition",
            "synchronization": "device synchronize after construction/reset",
            "model_load_excluded": True,
            "load_seconds": load_seconds,
        },
        "correctness": {
            "first_token_exact": exact,
            "first_token_id": int(rows[0]["first_token_id"]),
            "lifecycle_recovered": recovered,
        },
        "memory": {
            "owner_resident_nbytes": owner_resident_nbytes,
            "pooled_resident_nbytes": pooled_resident_nbytes,
            "tracked_before": tracked_before,
            "tracked_after": tracked_after,
        },
        "fresh": fresh_summary,
        "pooled_reset": pooled_summary,
        "paired": paired,
        "comparison": {
            "median_prepare_saved_ms": (
                float(fresh_summary["session_prepare_ms"]["median"])
                - float(pooled_summary["session_prepare_ms"]["median"])
            ),
            "median_prepare_speedup": (
                float(fresh_summary["session_prepare_ms"]["median"])
                / float(pooled_summary["session_prepare_ms"]["median"])
            ),
            "median_ttft_saved_ms": median_fresh_ttft - median_pooled_ttft,
            "median_ttft_speedup": median_fresh_ttft / median_pooled_ttft,
        },
        "gates": {
            "first_token_exact": exact,
            "every_paired_setup_improved": every_setup_improved,
            "median_direct_ttft_non_regressive": ttft_non_regressive,
            "lifecycle_recovered": recovered,
        },
        "samples": rows,
    }


def main() -> int:
    args = _parse_args()
    result = _run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
