#!/usr/bin/env python3
"""Repeated, interleaved fused-versus-chain GGUF GDN prefill wall gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance


KIND = "hipengine_gguf_gdn_prefill_interleaved_ab"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CORRECTNESS_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json"
)
MODES = ("fused", "chain")


class BenchmarkError(RuntimeError):
    """Raised when the A/B protocol cannot produce a valid measurement."""


def _parse_contexts(raw: str) -> tuple[int, ...]:
    try:
        contexts = tuple(int(part.strip()) for part in str(raw).split(","))
    except ValueError as exc:
        raise BenchmarkError("contexts must be comma-separated positive integers") from exc
    if not contexts or any(context <= 0 for context in contexts):
        raise BenchmarkError("contexts must be comma-separated positive integers")
    if len(set(contexts)) != len(contexts):
        raise BenchmarkError("contexts must not contain duplicates")
    return contexts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_correctness_gate(path: Path, *, contexts: Sequence[int]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkError(f"correctness artifact does not exist: {resolved}")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"could not read correctness artifact: {resolved}") from exc
    classification = artifact.get("classification")
    if not isinstance(classification, Mapping) or classification.get("passed") is not True:
        raise BenchmarkError("correctness artifact is not an accepted passing gate")
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise BenchmarkError("correctness artifact does not contain a case matrix")
    covered = {
        int(case.get("prompt", {}).get("length", -1))
        for case in cases
        if isinstance(case, Mapping) and case.get("passed") is True
    }
    missing = sorted(set(int(context) for context in contexts) - covered)
    if missing:
        raise BenchmarkError(f"correctness artifact does not cover contexts: {missing}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "kind": artifact.get("kind"),
        "source_revision": artifact.get("source_revision"),
        "status": classification.get("status"),
        "covered_contexts": sorted(covered),
        "passed": True,
    }


@contextlib.contextmanager
def _gdn_mode(mode: str) -> Iterator[None]:
    if mode not in MODES:
        raise BenchmarkError(f"unsupported GDN mode: {mode!r}")
    name = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
    previous = os.environ.get(name)
    os.environ[name] = mode
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _timed_prefill(session: Any, *, mode: str, prompt_ids: Sequence[int]) -> dict[str, Any]:
    runtime = session.runtime
    if runtime is None:
        raise BenchmarkError("GGUF session has no HIP runtime")
    runtime.device_synchronize()
    started_ns = time.perf_counter_ns()
    with _gdn_mode(mode):
        result = session.prefill(
            [int(token) for token in prompt_ids],
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=False,
            capture_hidden_seed_fp32=False,
        )
    runtime.device_synchronize()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
        raise BenchmarkError(f"invalid {mode} prefill wall: {elapsed_ms!r}")
    return {"wall_ms": float(elapsed_ms), "token_id": int(result.token_id)}


def _mode_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise BenchmarkError("timing samples must be finite and positive")
    return {
        "samples_ms": values,
        "count": len(values),
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "stdev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _summarize_context(
    measurements: Sequence[Mapping[str, Any]],
    *,
    expected_token_id: int,
) -> dict[str, Any]:
    if not measurements:
        raise BenchmarkError("a context must contain measured repetitions")
    mode_stats = {
        mode: _mode_statistics(
            [float(row["modes"][mode]["wall_ms"]) for row in measurements]
        )
        for mode in MODES
    }
    paired_chain_delta_ms = [
        float(row["modes"]["chain"]["wall_ms"])
        - float(row["modes"]["fused"]["wall_ms"])
        for row in measurements
    ]
    token_ids = {
        mode: [int(row["modes"][mode]["token_id"]) for row in measurements]
        for mode in MODES
    }
    tokens_exact = all(
        fused == chain == int(expected_token_id)
        for fused, chain in zip(token_ids["fused"], token_ids["chain"], strict=True)
    )
    fused_median = float(mode_stats["fused"]["median_ms"])
    chain_median = float(mode_stats["chain"]["median_ms"])
    chain_speedup = fused_median / chain_median
    chain_delta_percent = ((chain_median - fused_median) / fused_median) * 100.0
    return {
        "measurements": [dict(row) for row in measurements],
        "statistics": mode_stats,
        "paired_chain_minus_fused_ms": paired_chain_delta_ms,
        "paired_chain_minus_fused_median_ms": float(
            statistics.median(paired_chain_delta_ms)
        ),
        "token_ids": token_ids,
        "expected_token_id": int(expected_token_id),
        "tokens_exact": tokens_exact,
        "chain_speedup_vs_fused": float(chain_speedup),
        "chain_wall_delta_percent": float(chain_delta_percent),
        "chain_wins": bool(chain_median < fused_median),
    }


def _promotion_decision(
    contexts: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    correctness_gate_passed: bool,
) -> dict[str, Any]:
    clean = not bool(provenance.get("dirty"))
    tokens_exact = all(bool(context.get("tokens_exact")) for context in contexts)
    chain_wins_all = all(bool(context.get("chain_wins")) for context in contexts)
    measurement_valid = bool(clean and correctness_gate_passed and tokens_exact)
    if not measurement_valid:
        status = "invalid_measurement"
        default = "unchanged"
        conclusion = "The G3 measurement is not promotion-eligible."
    elif chain_wins_all:
        status = "promote_chain"
        default = "chain"
        conclusion = "The exact split chain wins both primary contexts."
    else:
        status = "retain_fused_reject_chain_promotion"
        default = "fused"
        conclusion = "The exact split chain does not win both primary contexts."
    return {
        "measurement_valid": measurement_valid,
        "clean_provenance": clean,
        "correctness_gate_passed": bool(correctness_gate_passed),
        "timed_tokens_exact": tokens_exact,
        "chain_wins_all_contexts": chain_wins_all,
        "status": status,
        "selected_default": default,
        "conclusion": conclusion,
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    contexts = _parse_contexts(args.contexts)
    repetitions = int(args.repetitions)
    warmups = int(args.warmups)
    if repetitions <= 0 or repetitions % 2:
        raise BenchmarkError("repetitions must be a positive even integer")
    if warmups < 0:
        raise BenchmarkError("warmups must be non-negative")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise BenchmarkError(f"model does not exist: {model}")
    correctness_gate = _load_correctness_gate(
        args.correctness_artifact,
        contexts=contexts,
    )
    compiler_version = None
    if args.compiler_version_file is not None:
        compiler_version = args.compiler_version_file.read_text(encoding="utf-8")

    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    context_records: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max(contexts) + 2,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=True,
    ) as session:
        if session.runner is None:
            raise BenchmarkError("GGUF resident session closed during setup")
        resolved_backend = str(session.runner.backend)
        target_arch = str(session.runner.target_arch)
        for context in contexts:
            prompt_ids = [int(args.prompt_token_id)] * int(context)
            warmup_records: list[dict[str, Any]] = []
            for warmup in range(warmups):
                order = MODES if warmup % 2 == 0 else tuple(reversed(MODES))
                row = {"warmup": warmup, "order": list(order), "modes": {}}
                for mode in order:
                    row["modes"][mode] = _timed_prefill(
                        session,
                        mode=mode,
                        prompt_ids=prompt_ids,
                    )
                warmup_records.append(row)
            measured: list[dict[str, Any]] = []
            for repetition in range(repetitions):
                order = MODES if repetition % 2 == 0 else tuple(reversed(MODES))
                row = {"repetition": repetition, "order": list(order), "modes": {}}
                for mode in order:
                    row["modes"][mode] = _timed_prefill(
                        session,
                        mode=mode,
                        prompt_ids=prompt_ids,
                    )
                measured.append(row)
            summary = _summarize_context(
                measured,
                expected_token_id=int(args.expected_token_id),
            )
            context_records.append(
                {
                    "context_tokens": int(context),
                    "prompt_token_id": int(args.prompt_token_id),
                    "prompt_sha256_i64": hashlib.sha256(
                        int(args.prompt_token_id).to_bytes(8, "little", signed=True)
                        * int(context)
                    ).hexdigest(),
                    "warmups": warmup_records,
                    **summary,
                }
            )

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": "interleaved explicit sweep",
            "HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD": os.environ.get(
                "HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD"
            ),
        },
        build_profile="gguf_gdn_prefill_interleaved_ab",
        timing_protocol="same_session_balanced_interleaved_host_wall_v1",
        warmups=warmups,
        repetitions=repetitions,
        profiler={
            "enabled": False,
            "kind": None,
            "command": None,
            "correctness_trace_in": correctness_gate["path"],
        },
    )
    decision = _promotion_decision(
        context_records,
        provenance=provenance,
        correctness_gate_passed=bool(correctness_gate["passed"]),
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": bool(decision["measurement_valid"]),
        "correctness_claim": True,
        "workload": {
            "model": str(model),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "contexts": list(contexts),
            "prompt_token_id": int(args.prompt_token_id),
            "expected_token_id": int(args.expected_token_id),
            "bulk_attention_mode": "bulk",
            "use_wmma_prefill": bool(args.use_wmma_prefill),
            "use_gemv_decode": True,
            "gdn_modes": list(MODES),
        },
        "protocol": {
            "wall_clock": "time.perf_counter_ns around synchronized session.prefill",
            "session_scope": "one resident model/session shared by both modes and contexts",
            "ordering": "balanced fused-chain / chain-fused by measured repetition",
            "warmups_per_context": warmups,
            "measured_repetitions_per_mode_context": repetitions,
            "state_reset": "production session.prefill reset before every measured prompt",
        },
        "correctness_gate": correctness_gate,
        "contexts": context_records,
        "decision": decision,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--contexts", default="512,4096")
    parser.add_argument("--prompt-token-id", type=int, default=9707)
    parser.add_argument("--expected-token-id", type=int, default=9707)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument(
        "--use-wmma-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--correctness-artifact",
        type=Path,
        default=DEFAULT_CORRECTNESS_ARTIFACT,
    )
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        artifact = run(args, command=command)
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.json)
    print(json.dumps(artifact["decision"], indent=2, sort_keys=True))
    return 0 if artifact["decision"]["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
