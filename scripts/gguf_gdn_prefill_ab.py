#!/usr/bin/env python3
"""Repeated, interleaved baseline-versus-candidate GGUF GDN prefill wall gate."""

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
SCHEMA_VERSION = 3
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_CORRECTNESS_ARTIFACT = (
    REPO_ROOT
    / "benchmarks/results/2026-07-11-sol-g2-gfx1151-gdn-prefill-exact-matrix.json"
)
BASELINE_MODE = "fused"
CANDIDATE_MODES = (
    "chain",
    "chain_peer_wave32",
    "chain_peer_cluster8",
    "chain_tile64",
    "chain_tile32",
    "chain_wave32",
    "chain_wave32_tree",
    "chain_lds64",
    "chain_lds32",
    "chain_lds32_direct",
)
SUPPORTED_MODES = (BASELINE_MODE, *CANDIDATE_MODES)


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


def _case_context(case: Mapping[str, Any]) -> int | None:
    prompt = case.get("prompt")
    if isinstance(prompt, Mapping):
        try:
            value = int(prompt.get("length", -1))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    if isinstance(prompt, str) and "/" in prompt:
        try:
            value = int(prompt.rsplit("/", 1)[1])
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    return None


def _load_correctness_gate(
    path: Path,
    *,
    contexts: Sequence[int],
    baseline_mode: str = BASELINE_MODE,
    candidate_mode: str = "chain",
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchmarkError(f"correctness artifact does not exist: {resolved}")
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"could not read correctness artifact: {resolved}") from exc
    classification = artifact.get("classification")
    correctness = artifact.get("correctness")
    if isinstance(classification, Mapping) and classification.get("passed") is True:
        modes = artifact.get("protocol", {}).get("modes")
        expected_oracle_modes = [BASELINE_MODE, candidate_mode]
        expected_ab_modes = [baseline_mode, candidate_mode]
        if modes not in (expected_oracle_modes, expected_ab_modes):
            raise BenchmarkError(
                "correctness artifact candidate does not match A/B candidate: "
                f"{modes!r} not in "
                f"{(expected_oracle_modes, expected_ab_modes)!r}"
            )
        cases = artifact.get("cases")
        if not isinstance(cases, list):
            raise BenchmarkError("correctness artifact does not contain a case matrix")
        covered = {
            context
            for case in cases
            if isinstance(case, Mapping) and case.get("passed") is True
            if (context := _case_context(case)) is not None
        }
        contract = "byte_exact"
        promotion_eligible = True
        status = classification.get("status")
        source_revision = artifact.get("source_revision")
        correctness_modes = list(modes)
    elif isinstance(correctness, Mapping):
        project_gate = correctness.get("project_gate")
        cases = correctness.get("cases")
        if not isinstance(project_gate, Mapping) or not isinstance(cases, list):
            raise BenchmarkError("correctness artifact is not an accepted passing gate")
        try:
            cases_passed = int(project_gate.get("cases_passed", -1))
            cases_total = int(project_gate.get("cases_total", -1))
            kl_threshold = float(project_gate["kl_threshold"])
            top1_threshold = float(project_gate["top1_threshold"])
            kl_max = max(float(value) for value in project_gate["kl_mean_range"])
            top1_min = float(project_gate["top1_agreement_min"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError("project correctness gate is malformed") from exc
        selector = str(artifact.get("protocol", {}).get("selector", ""))
        expected_selector = f"HIPENGINE_GGUF_GDN_PREFILL_MODE={candidate_mode}"
        if selector != expected_selector:
            raise BenchmarkError(
                "correctness artifact candidate does not match A/B candidate: "
                f"{selector!r} != {expected_selector!r}"
            )
        project_passed = bool(
            cases_total > 0
            and cases_passed == cases_total
            and project_gate.get("sampled_tokens_identical") is True
            and math.isfinite(kl_max)
            and kl_max <= kl_threshold
            and math.isfinite(top1_min)
            and top1_min >= top1_threshold
        )
        if not project_passed:
            raise BenchmarkError("correctness artifact project KL/top-1 gate did not pass")
        covered = {
            context
            for case in cases
            if isinstance(case, Mapping)
            if (context := _case_context(case)) is not None
            if float(case.get("kl_mean", math.inf)) <= kl_threshold
        }
        contract = "project_kl_top1_non_exact"
        promotion_eligible = False
        status = artifact.get("status")
        source_revision = artifact.get("software", {}).get("candidate_base_commit")
        correctness_modes = None
    else:
        raise BenchmarkError("correctness artifact is not an accepted passing gate")
    missing = sorted(set(int(context) for context in contexts) - covered)
    if missing:
        raise BenchmarkError(f"correctness artifact does not cover contexts: {missing}")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "kind": artifact.get("kind"),
        "source_revision": source_revision,
        "status": status,
        "contract": contract,
        "default_promotion_eligible": promotion_eligible,
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "correctness_modes": correctness_modes,
        "covered_contexts": sorted(covered),
        "passed": True,
    }


@contextlib.contextmanager
def _gdn_mode(mode: str) -> Iterator[None]:
    if mode not in SUPPORTED_MODES:
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
    baseline_mode: str = BASELINE_MODE,
    candidate_mode: str = "chain",
) -> dict[str, Any]:
    if not measurements:
        raise BenchmarkError("a context must contain measured repetitions")
    modes = (baseline_mode, candidate_mode)
    mode_stats = {
        mode: _mode_statistics(
            [float(row["modes"][mode]["wall_ms"]) for row in measurements]
        )
        for mode in modes
    }
    paired_candidate_delta_ms = [
        float(row["modes"][candidate_mode]["wall_ms"])
        - float(row["modes"][baseline_mode]["wall_ms"])
        for row in measurements
    ]
    token_ids = {
        mode: [int(row["modes"][mode]["token_id"]) for row in measurements]
        for mode in modes
    }
    tokens_exact = all(
        baseline == candidate == int(expected_token_id)
        for baseline, candidate in zip(
            token_ids[baseline_mode], token_ids[candidate_mode], strict=True
        )
    )
    baseline_median = float(mode_stats[baseline_mode]["median_ms"])
    candidate_median = float(mode_stats[candidate_mode]["median_ms"])
    candidate_speedup = baseline_median / candidate_median
    candidate_delta_percent = (
        (candidate_median - baseline_median) / baseline_median
    ) * 100.0
    return {
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "measurements": [dict(row) for row in measurements],
        "statistics": mode_stats,
        "paired_candidate_minus_baseline_ms": paired_candidate_delta_ms,
        "paired_candidate_minus_baseline_median_ms": float(
            statistics.median(paired_candidate_delta_ms)
        ),
        "token_ids": token_ids,
        "expected_token_id": int(expected_token_id),
        "tokens_exact": tokens_exact,
        "candidate_speedup_vs_baseline": float(candidate_speedup),
        "candidate_wall_delta_percent": float(candidate_delta_percent),
        "candidate_wins": bool(candidate_median < baseline_median),
    }


def _promotion_decision(
    contexts: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    correctness_gate: Mapping[str, Any],
    baseline_mode: str = BASELINE_MODE,
    candidate_mode: str = "chain",
) -> dict[str, Any]:
    clean = not bool(provenance.get("dirty"))
    correctness_gate_passed = correctness_gate.get("passed") is True
    contract_allows_promotion = bool(
        correctness_gate.get("default_promotion_eligible")
    )
    tokens_exact = all(bool(context.get("tokens_exact")) for context in contexts)
    candidate_wins_all = all(bool(context.get("candidate_wins")) for context in contexts)
    measurement_valid = bool(clean and correctness_gate_passed and tokens_exact)
    if not measurement_valid:
        status = "invalid_measurement"
        default = "unchanged"
        conclusion = "The G3 measurement is not promotion-eligible."
    elif candidate_wins_all and contract_allows_promotion:
        status = "promote_candidate"
        default = candidate_mode
        conclusion = f"The correctness-qualified {candidate_mode} wins every context."
    elif candidate_wins_all:
        status = "candidate_wins_pending_correctness_contract"
        default = "unchanged"
        conclusion = (
            f"The {candidate_mode} candidate wins every context, but its current "
            "correctness artifact is not sufficient by itself for default promotion."
        )
    else:
        status = "retain_baseline_reject_candidate_performance"
        default = baseline_mode
        conclusion = f"The {candidate_mode} candidate does not win every context."
    return {
        "measurement_valid": measurement_valid,
        "clean_provenance": clean,
        "correctness_gate_passed": bool(correctness_gate_passed),
        "correctness_contract": correctness_gate.get("contract"),
        "contract_allows_default_promotion": contract_allows_promotion,
        "timed_tokens_exact": tokens_exact,
        "baseline_mode": baseline_mode,
        "candidate_mode": candidate_mode,
        "candidate_wins_all_contexts": candidate_wins_all,
        "status": status,
        "selected_default": default,
        "conclusion": conclusion,
    }


def run(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    contexts = _parse_contexts(args.contexts)
    baseline_mode = str(args.baseline_mode)
    candidate_mode = str(args.candidate_mode)
    if baseline_mode not in SUPPORTED_MODES:
        raise BenchmarkError(f"unsupported baseline mode: {baseline_mode!r}")
    if candidate_mode not in CANDIDATE_MODES:
        raise BenchmarkError(f"unsupported candidate mode: {candidate_mode!r}")
    if baseline_mode == candidate_mode:
        raise BenchmarkError("baseline and candidate modes must differ")
    modes = (baseline_mode, candidate_mode)
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
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
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
                order = modes if warmup % 2 == 0 else tuple(reversed(modes))
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
                order = modes if repetition % 2 == 0 else tuple(reversed(modes))
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
                baseline_mode=baseline_mode,
                candidate_mode=candidate_mode,
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
            "HIPENGINE_GGUF_GDN_PREFILL_MODE": (
                f"interleaved {baseline_mode} versus {candidate_mode}"
            ),
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
        correctness_gate=correctness_gate,
        baseline_mode=baseline_mode,
        candidate_mode=candidate_mode,
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
            "baseline_mode": baseline_mode,
            "candidate_mode": candidate_mode,
            "gdn_modes": list(modes),
        },
        "protocol": {
            "wall_clock": "time.perf_counter_ns around synchronized session.prefill",
            "session_scope": "one resident model/session shared by both modes and contexts",
            "ordering": (
                f"balanced {baseline_mode}-{candidate_mode} / "
                f"{candidate_mode}-{baseline_mode} by measured repetition"
            ),
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
    parser.add_argument(
        "--baseline-mode",
        choices=SUPPORTED_MODES,
        default=BASELINE_MODE,
        help="explicit GDN prefill implementation used as the wall-time baseline",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=CANDIDATE_MODES,
        default="chain",
        help="explicit GDN prefill implementation to compare against the baseline",
    )
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
