#!/usr/bin/env python3
"""In-process Qwen4Exp MTP economics on the full mtp-bench category suite.

One resident target, one process: for every prompt of
``benchmarks/prompts/mtpbench-code-general-ja.jsonl`` (all four categories,
heldouts included) a true autoregressive arm (target prefill + greedy decode
steps, no MTP) and an MTP arm (``Qwen4ExpMTPTextProvider``) generate the same
number of tokens from the same chat-formatted prompt. Reports request wall,
tok/s, MTP/AR ratio by category and overall, draft acceptance, verify phase
cost, and per-prompt token agreement with the AR arm (a diagnostic: the MTP
verifier runs prefill-shaped kernels, so its greedy path may legitimately
diverge from serial decode under production arithmetic).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata, _write_json
from scripts.qwen4exp_mtp_phase_census import DEFAULT_MODEL, DEFAULT_PROMPTS, DEFAULT_SIDECAR


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def source_snapshot():
    """Detect source edits even when both observations are already dirty."""
    return {**_git_metadata(ROOT), "tracked_diff_sha256": hashlib.sha256(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    ).hexdigest()}


def run_ar(runner, prompt, max_tokens, *, capture_mode="compact"):
    """Run true AR with the public greedy capture policy, or explicit diagnostics."""
    if max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if capture_mode not in ("compact", "full"):
        raise ValueError("capture-mode must be compact or full")
    from hipengine.generation.qwen4_exp_gguf import (
        _qwen4_exp_compact_prefill_kwargs,
        _qwen4_exp_compact_step_kwargs,
    )

    prefill = _qwen4_exp_compact_prefill_kwargs() if capture_mode == "compact" else {}
    step = _qwen4_exp_compact_step_kwargs() if capture_mode == "compact" else {}
    runner.reset()
    result = runner.prefill(prompt, **prefill)
    tokens = [int(result.token_id)]
    while len(tokens) < max_tokens:
        result = runner.step(tokens[-1], **step)
        tokens.append(int(result.token_id))
    return tokens


def pair_order(case_index, repetition):
    """Alternate order within a prompt and balance odd repeats across prompts."""
    return ("ar", "mtp") if (case_index + repetition) % 2 == 0 else ("mtp", "ar")


def measure_pair(generator, request, *, case, repetition, order, capture_mode,
                 clock=time.perf_counter):
    if tuple(order) not in (("ar", "mtp"), ("mtp", "ar")):
        raise ValueError("a pair must measure AR and MTP once each")
    tokens, seconds, diagnostics = {}, {}, {}
    for arm in order:
        started = clock()
        if arm == "ar":
            ids = run_ar(generator.runner, request.prompts[0], request.max_tokens,
                         capture_mode=capture_mode)
        else:
            output = generator.generate_speculative_mtp_detailed(request)[0]
            ids = [int(token) for token in output.generated_token_ids]
            diagnostics = output.telemetry.diagnostics
        elapsed = clock() - started
        if len(ids) != request.max_tokens:
            raise ValueError(f"{arm.upper()} expected {request.max_tokens} output tokens, got {len(ids)}")
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"{arm} wall time must be finite and positive")
        tokens[arm], seconds[arm] = ids, elapsed
    agree = 0
    for a, b in zip(tokens["ar"], tokens["mtp"], strict=True):
        if a != b:
            break
        agree += 1
    census = diagnostics.get("phase_census", {})
    return {
        "id": case["id"], "category": case["category"],
        "prompt_tokens": len(request.prompts[0]), "prompt_token_ids": list(request.prompts[0]),
        "repetition": repetition, "order": list(order), "ar_capture_mode": capture_mode,
        "ar_seconds": seconds["ar"], "mtp_seconds": seconds["mtp"],
        "ar_output_tokens": len(tokens["ar"]), "mtp_output_tokens": len(tokens["mtp"]),
        "ar_token_ids": tokens["ar"], "mtp_token_ids": tokens["mtp"],
        "ar_tok_s": len(tokens["ar"]) / seconds["ar"],
        "mtp_tok_s": len(tokens["mtp"]) / seconds["mtp"],
        "mtp_over_ar": seconds["ar"] / seconds["mtp"],
        "proposed": diagnostics["proposed_draft_tokens"],
        "accepted": diagnostics["accepted_draft_tokens"],
        "acceptance": diagnostics["draft_acceptance"],
        "cycles": census.get("cycles"), "phase_census": census,
        "verify_ms": census.get("target_verify", {}).get("ms"),
        "proposal_ms": census.get("proposal", {}).get("ms"),
        "agree_prefix_with_ar": agree, "identical_to_ar": tokens["ar"] == tokens["mtp"],
        "target_verify": diagnostics.get("target_verify"),
    }


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["full"].append(row)
        groups["category:" + row["category"]].append(row)
    summary = {}
    for name, group in groups.items():
        ar = sum(row["ar_seconds"] for row in group)
        mtp = sum(row["mtp_seconds"] for row in group)
        proposed = sum(row["proposed"] for row in group)
        sequences = {arm: defaultdict(set) for arm in ("ar", "mtp")}
        for row in group:
            for arm in sequences:
                sequences[arm][row["id"]].add(tuple(row[arm + "_token_ids"]))
        summary[name] = {
            "prompts": len({row["id"] for row in group}), "samples": len(group),
            "ar_tok_s": sum(row["ar_output_tokens"] for row in group) / ar,
            "mtp_tok_s": sum(row["mtp_output_tokens"] for row in group) / mtp,
            "mtp_over_ar": ar / mtp,
            "acceptance": sum(row["accepted"] for row in group) / proposed if proposed else 0.0,
            "identical_to_ar": sum(int(row["identical_to_ar"]) for row in group),
            **{arm + "_deterministic": all(len(ids) == 1 for ids in cases.values())
               for arm, cases in sequences.items()},
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--execution-profile", default="production")
    parser.add_argument("--max-sequence-length", type=int, default=1_024)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--capture-mode", choices=("compact", "full"), default="compact",
                        help="AR capture: public greedy path (compact) or legacy full-logit diagnostic.")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_tokens <= 0 or args.repetitions <= 0 or args.warmups < 0:
        parser.error("max-tokens and repetitions must be positive; warmups must be non-negative")
    if args.prefill_chunk_size <= 0 or args.max_sequence_length <= 0 or not 1 <= args.budget <= 4:
        parser.error("chunk/context sizes must be positive and budget must be in 1..4")
    cases = _load(args.prompts)
    if not cases or len({row["id"] for row in cases}) != len(cases):
        parser.error("prompt suite must be nonempty with unique IDs")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine.benchmark.provenance import collect_artifact_provenance, collect_model_identity
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND, QWEN4_EXP_MODEL, QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.generation.registry import GenerationRequest
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from hipengine.speculative.registry import SpeculativeProviderConfig

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND, quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(args.execution_profile))
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    source = source_snapshot()
    host = _host_metadata()
    provenance = collect_artifact_provenance(
        repo_root=ROOT, configured_backend=QWEN4_EXP_BACKEND, resolved_backend=QWEN4_EXP_BACKEND,
        model_path=args.model_root, quant=QWEN4_EXP_QUANTS[1], kv_dtype="bf16",
        command=command, build_profile=args.execution_profile,
        timing_protocol="whole-request synchronized wall; true AR and MTP in alternating order",
        warmups=args.warmups, repetitions=args.repetitions,
        profiler={"status": "not_run"},
        environment={key: value for key, value in os.environ.items()
                     if key.startswith("HIPENGINE_") or key == "GPU_MAX_HW_QUEUES"},
        rocm_version=host.get("rocm_platform"), hipcc_version=host.get("hipcc_version"),
    )
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""), backend=QWEN4_EXP_BACKEND,
        max_sequence_length=args.max_sequence_length, prefill_chunk_size=args.prefill_chunk_size))
    rows = []
    try:
        provider = Qwen4ExpMTPTextProvider(
            target_generator=generator,
            config=SpeculativeProviderConfig(
                provider="qwen4_exp_mtp", draft_model=args.sidecar,
                candidate_budget=args.budget))
        generator.attach_speculative_provider(provider)
        for case_index, case in enumerate(cases):
            text = "\n".join(str(m["content"]) for m in case["messages"])
            prompt = tuple(int(t) for t in build_chat_prompt(generator.tokenizer, text))
            request = GenerationRequest(
                prompts=(prompt,), max_tokens=args.max_tokens,
                temperature=0.0, top_p=1.0, ignore_eos=True, top_k=1)
            for repetition in range(-args.warmups, args.repetitions):
                row = measure_pair(
                    generator, request, case=case, repetition=repetition,
                    order=pair_order(case_index, repetition), capture_mode=args.capture_mode)
                if repetition >= 0:
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in (
                        "id", "repetition", "order", "ar_tok_s", "mtp_tok_s", "acceptance")}), flush=True)
    finally:
        generator.close()

    summary = summarize_rows(rows)
    source_after = source_snapshot()
    checks = {
        "source_unchanged": source == source_after,
        "ar_repeat_determinism": summary["full"]["ar_deterministic"],
        "mtp_repeat_determinism": summary["full"]["mtp_deterministic"],
    }
    result = {
        "schema": 2, "kind": "qwen4exp_mtp_inprocess_bench",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "status": "measured_diagnostic" if all(checks.values()) else "invalid",
        "quality_scope": "repeat determinism only; numerical and task gates are separate",
        "source": source, "source_after": source_after, "host": host,
        "provenance": provenance, "sidecar_identity": collect_model_identity(args.sidecar),
        "command": command, "checks": checks,
        "variant_manifest_sha256": profile.manifest_sha256,
        "strict_manifest_sha256": profile.strict_manifest_sha256,
        "protocol": {
            "prompts": str(args.prompts),
            "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "max_tokens": args.max_tokens,
            "budget": args.budget, "execution_profile": args.execution_profile,
            "prefill_chunk_size": args.prefill_chunk_size, "max_sequence_length": args.max_sequence_length,
            "ar_capture_mode": args.capture_mode, "warmups": args.warmups,
            "repetitions": args.repetitions, "pair_order": "alternate by case index plus repetition",
            "warmup": "unmeasured complete AR/MTP pairs per prompt; excluded from rows",
            "timing_scope": "whole-request wall including target prefill and MTP prompt catch-up",
            "batched_verify_env": os.environ.get("HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY"),
        },
        "rows": rows, "summary": summary,
    }
    _write_json(args.output, result)
    print(json.dumps(summary, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
