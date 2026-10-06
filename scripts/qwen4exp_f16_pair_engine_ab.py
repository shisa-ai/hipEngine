#!/usr/bin/env python3
"""Tracked-clean Qwen4Exp engine A/B for DP-2 F16 cast ownership.

The parent arm disables ``HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR`` and the
candidate arm enables it.  Both arms run in one Python process and one resident
model, with one warmup per arm and three counterbalanced measured samples for
the canonical 12-case fixture plus the eight category-heldout prompts at
1024/4096 rows.  The candidate is admitted only by the runtime's bounded
pair contract; this harness does not replace or bypass production consumers.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    DEFAULT_FIXTURE,
    _git_metadata,
    _hipengine_case_sample,
    _host_metadata,
    _write_json,
    load_fixture,
    summarize_samples,
    token_ids_sha256,
)

PAIR_ENV = "HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR"
HELDOUTS = ROOT / "benchmarks" / "prompts" / "gdn-prefill-category-heldouts.jsonl"
ARM_SEQUENCE = ("candidate", "parent", "parent", "candidate", "candidate", "parent")
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected object rows")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heldout_cases(generator: Any) -> list[dict[str, Any]]:
    from scripts.gguf_mtp_bench import build_chat_prompt
    from scripts.gguf_mtp_category_bench import load_prompt_rows

    rows = load_prompt_rows(HELDOUTS)
    if len(rows) != 8 or {str(row.get("category")) for row in rows} != set(CATEGORIES):
        raise ValueError("heldout suite must contain eight rows across all four categories")
    cases: list[dict[str, Any]] = []
    for row in rows:
        tokens = [int(value) for value in build_chat_prompt(generator.tokenizer, str(row["prompt"]))]
        if not tokens:
            raise ValueError(f"empty heldout prompt: {row.get('id')}")
        for target in (1024, 4096):
            padded = (tokens * ((target - 1) // len(tokens) + 1))[:target]
            cases.append(
                {
                    "id": f"{row['id']}-p{target}",
                    "category": str(row["category"]),
                    "prompt_tokens": target,
                    "prompt_token_ids": padded,
                    "prompt_token_ids_sha256": token_ids_sha256(padded),
                    "source": "heldout",
                }
            )
    return cases


def _arm_order(index: int) -> tuple[str, ...]:
    return ARM_SEQUENCE if int(index) % 2 == 0 else tuple(reversed(ARM_SEQUENCE))


def _expected_pairs(tokens: int, chunk_size: int = 1024) -> int:
    """Count admitted pairs per executed chunk, not per complete prompt."""
    if tokens <= 0 or chunk_size <= 0:
        raise ValueError("positive tokens and chunk size required")
    full, tail = divmod(tokens, chunk_size)
    return 61 * (full * (chunk_size in (1024, 4096)) + (tail in (1024, 4096)))


def _select_cases(cases, prompt_tokens):
    """Keep full-suite indices when completing an unfinished shape subset."""
    selected = [(i, case) for i, case in enumerate(cases)
                if not prompt_tokens or int(case["prompt_tokens"]) in prompt_tokens]
    if not selected:
        raise ValueError("no cases match the requested prompt lengths")
    return selected


def summarize_engine_ab(samples, *, chunk_size: int) -> dict[str, Any]:
    """Validate paired timing evidence; never merge arms into a variance row."""
    by_case: dict[str, dict[str, list]] = {}
    for row in samples:
        if row["mode"] not in ("parent", "candidate"):
            raise ValueError("invalid arm")
        if token_ids_sha256(row["output_token_ids"]) != row["output_token_ids_sha256"]:
            raise ValueError("output token hash mismatch")
        if any(not math.isfinite(float(row[key])) or float(row[key]) <= 0
               for key in ("prefill_ms", "decode_ms")):
            raise ValueError("nonpositive or nonfinite timing")
        if len(row["output_token_ids"]) != int(row["decode_transitions"]) + 1:
            raise ValueError("wrong output token count")
        by_case.setdefault(str(row["case_id"]), {"parent": [], "candidate": []})[row["mode"]].append(row)
    if not by_case:
        raise ValueError("no measured cases")
    case_summary = {}
    for case_id, modes in by_case.items():
        rows = sorted(modes["parent"] + modes["candidate"], key=lambda r: r["sequence_slot"])
        if any(len(modes[mode]) != 3 for mode in ("parent", "candidate")):
            raise ValueError(f"{case_id}: incomplete measured arm counts")
        order = tuple(rows[0]["case_sequence"])
        if (order not in (ARM_SEQUENCE, tuple(reversed(ARM_SEQUENCE)))
                or [r["sequence_slot"] for r in rows] != list(range(6))
                or tuple(r["mode"] for r in rows) != order
                or any(tuple(r["case_sequence"]) != order for r in rows)):
            raise ValueError(f"{case_id}: invalid counterbalance")
        for key in ("prompt_tokens", "category", "source", "prompt_token_ids_sha256", "decode_transitions"):
            if len({r[key] for r in rows}) != 1:
                raise ValueError(f"{case_id}: inconsistent {key}")
        if len({r["output_token_ids_sha256"] for r in rows}) != 1:
            raise ValueError(f"{case_id}: output is not exact/repeat-deterministic")
        expected = _expected_pairs(int(rows[0]["prompt_tokens"]), chunk_size)
        for mode in ("parent", "candidate"):
            expected_mode = expected if mode == "candidate" else 0
            if any(r["engagement"]["pairs"] != expected_mode
                   or r["engagement"]["accepted"] != 2 * expected_mode for r in modes[mode]):
                raise ValueError(f"{case_id}: invalid {mode} pair engagement")
            if len({r["engagement"]["casts"] for r in modes[mode]}) != 1:
                raise ValueError(f"{case_id}: cast counts are not repeat-deterministic")
        if modes["parent"][0]["engagement"]["casts"] - modes["candidate"][0]["engagement"]["casts"] != expected:
            raise ValueError(f"{case_id}: wrong cast elimination count")
        means = {mode: {key: sum(float(r[key]) for r in modes[mode]) / 3
                        for key in ("prefill_ms", "decode_ms")}
                 for mode in ("parent", "candidate")}
        case_summary[case_id] = {
            "source": rows[0]["source"], "category": rows[0]["category"],
            "prompt_tokens": rows[0]["prompt_tokens"],
            "parent_prefill_ms_mean": means["parent"]["prefill_ms"],
            "candidate_prefill_ms_mean": means["candidate"]["prefill_ms"],
            "prefill_ratio_candidate_over_parent": means["parent"]["prefill_ms"] / means["candidate"]["prefill_ms"],
            "parent_decode_ms_mean": means["parent"]["decode_ms"],
            "candidate_decode_ms_mean": means["candidate"]["decode_ms"],
            "cross_mode_output_exact": True,
            "output_token_ids_sha256": [rows[0]["output_token_ids_sha256"]],
            "candidate_engagement": modes["candidate"][0]["engagement"],
            "parent_engagement": modes["parent"][0]["engagement"],
        }
    result = {
        "samples_by_arm": {mode: summarize_samples([r for r in samples if r["mode"] == mode])
                           for mode in ("parent", "candidate")},
        "correctness": {"all_within_arm_deterministic": True, "all_cross_arm_output_exact": True},
        "by_case": case_summary, "by_shape": {}, "by_category": {},
    }
    for grouping, field in (("by_shape", "prompt_tokens"), ("by_category", "category")):
        for group in sorted({str(r[field]) for r in samples}):
            rows = [r for r in samples if str(r[field]) == group]
            arms = {mode: [r for r in rows if r["mode"] == mode] for mode in ("parent", "candidate")}
            totals = {mode: sum(float(r["prefill_ms"]) for r in arm) for mode, arm in arms.items()}
            result[grouping][group] = {
                "case_count": len({r["case_id"] for r in rows}),
                "parent_prefill_tok_s_weighted": 1000 * sum(int(r["prompt_tokens"]) for r in arms["parent"]) / totals["parent"],
                "candidate_prefill_tok_s_weighted": 1000 * sum(int(r["prompt_tokens"]) for r in arms["candidate"]) / totals["candidate"],
                "candidate_over_parent_prefill": totals["parent"] / totals["candidate"],
                **{f"{mode}_decode_tok_s_weighted": 1000 * sum(int(r["decode_transitions"]) for r in arm) / sum(float(r["decode_ms"]) for r in arm)
                   for mode, arm in arms.items()},
            }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--warmups-per-arm", type=int, default=1)
    parser.add_argument("--repetitions-per-arm", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, choices=(512, 1024, 4096), action="append",
                        default=[], help="Complete only these shapes; preserve full-suite arm order")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prefill_chunk_size != 1024:
        raise SystemExit("DP-2 A/B requires --prefill-chunk-size 1024")
    if args.warmups_per_arm != 1 or args.repetitions_per_arm != 3:
        raise SystemExit("DP-2 A/B requires one warmup and three measured samples per arm")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ.pop(PAIR_ENV, None)

    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from hipengine.runtime import qwen4_exp_runner as runner_module
    from scripts.qwen4exp_framework_family_refresh import model_identity

    fixture, fixture_sha256 = load_fixture(args.fixture)
    model_root = args.model_root.resolve()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )
    cases = list(fixture["cases"])
    max_sequence_length = 4096 + int(fixture["decode_transitions"]) + 8

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    source = _git_metadata(ROOT)
    if not source or source.get("tracked_clean") is not True:
        raise SystemExit("DP-2 publication A/B requires a tracked-clean worktree")
    if not args.compiler_version_file.is_file():
        raise SystemExit(f"compiler version file does not exist: {args.compiler_version_file}")

    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_dp2_engine_ab",
        "status": "running",
        "performance_claim": False,
        "host": _host_metadata(),
        "source": source,
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "model_root": str(model_root),
        "model_identity": model_identity(model_root),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": fixture_sha256,
        "heldout_suite": {
            "path": str(HELDOUTS.resolve()),
            "sha256": _sha256(HELDOUTS),
            "prompt_count": 8,
            "rows": [1024, 4096],
            "policy": "uniform repeat/truncate for ownership and wall timing; not a task-quality score",
        },
        "profile": {
            "requested": "production",
            "manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "protocol": {
            "one_python_process": True,
            "one_model_residency": True,
            "canonical_cases": None,
            "heldout_cases": None,
            "case_count": None,
            "case_order": "sorted by rows/category/id; adjacent full-suite cases reverse the six-slot arm order",
            "requested_prompt_tokens": args.prompt_tokens,
            "warmups_per_arm_per_case": args.warmups_per_arm,
            "measured_repetitions_per_arm_per_case": args.repetitions_per_arm,
            "arm_sequence_even_case": list(ARM_SEQUENCE),
            "arm_sequence_odd_case": list(reversed(ARM_SEQUENCE)),
            "prefill_chunk_size": args.prefill_chunk_size,
            "decode_transitions": int(fixture["decode_transitions"]),
            "timing_boundary": "synchronized runner.prefill plus exactly 128 synchronized runner.step transitions",
            "candidate_scope": "gfx1151 production Q8_0 dense-wide layers13-47, rows1024/4096, immediate ordered pairs",
        },
        "arms": {
            "parent": {"env": PAIR_ENV, "value": "0", "description": "ordinary two-cast parent route"},
            "candidate": {"env": PAIR_ENV, "value": "1", "description": "bounded two-consumer F16 ownership route"},
        },
        "samples": [],
        "warmups": [],
    }
    _write_json(args.output, report)

    original_scope = runner_module.wide_f16_activation_pair
    original_cast = wide.f32_to_f16
    counts = {"pairs": 0, "accepted": 0, "casts": 0}
    generator = None

    @contextlib.contextmanager
    def counted_scope(*scope_args: Any, **scope_kwargs: Any):
        with original_scope(*scope_args, **scope_kwargs) as loan:
            yield loan
            if loan is not None:
                if loan.accepted != 2:
                    raise AssertionError(f"DP-2 loan accepted {loan.accepted} times")
                counts["pairs"] += 1
                counts["accepted"] += int(loan.accepted)

    def counted_cast(*cast_args: Any, **cast_kwargs: Any):
        if len(cast_args) >= 3 and int(cast_args[2]) in (1024 * 2560, 4096 * 2560):
            counts["casts"] += 1
        return original_cast(*cast_args, **cast_kwargs)

    def make_cases() -> list[dict[str, Any]]:
        return sorted(cases + _heldout_cases(generator),
                      key=lambda row: (int(row["prompt_tokens"]), str(row["category"]), str(row["id"])))

    def sample(case: Mapping[str, Any], mode: str, repetition: int) -> dict[str, Any]:
        if mode not in ("parent", "candidate"):
            raise ValueError(mode)
        os.environ[PAIR_ENV] = "1" if mode == "candidate" else "0"
        before = dict(counts)
        row = _hipengine_case_sample(
            generator.runner,
            case=case,
            repetition=repetition,
            transitions=int(fixture["decode_transitions"]),
        )
        delta = {key: counts[key] - before[key] for key in counts}
        expected = _expected_pairs(int(case["prompt_tokens"]), args.prefill_chunk_size)
        if mode == "candidate" and delta["pairs"] != expected:
            raise AssertionError(f"{case['id']}: candidate pairs {delta['pairs']} != {expected}")
        if mode == "parent" and delta["pairs"] != 0:
            raise AssertionError(f"{case['id']}: parent pairs {delta['pairs']} != 0")
        row.update({"mode": mode, "engagement": delta, "source": case.get("source", "canonical")})
        return row

    try:
        with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generator = resolved.construct_generator(factory)
            runner_module.wide_f16_activation_pair = counted_scope
            wide.f32_to_f16 = counted_cast
            cases = make_cases()
            selected_cases = _select_cases(cases, args.prompt_tokens)
            report["protocol"]["case_ids"] = [str(row["id"]) for _, row in selected_cases]
            report["protocol"]["canonical_case_ids"] = [str(row["id"]) for _, row in selected_cases if row.get("source", "canonical") == "canonical"]
            report["protocol"]["heldout_case_ids"] = [str(row["id"]) for _, row in selected_cases if row.get("source") == "heldout"]
            report["protocol"].update(
                case_count=len(selected_cases),
                canonical_cases=len(report["protocol"]["canonical_case_ids"]),
                heldout_cases=len(report["protocol"]["heldout_case_ids"]),
                full_suite_counterbalance_indices={row["id"]: i for i, row in selected_cases},
            )
            report["provenance"] = collect_artifact_provenance(
                repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
                target_arch="gfx1151", model_path=model_root,
                model_revision=report["model_identity"]["revision"], quant="UD-Q4_K_XL", kv_dtype="bf16",
                command=report["command"], build_profile="cached production",
                timing_protocol="one-residency counterbalanced parent/candidate engine A/B; 128 decode transitions",
                warmups=args.warmups_per_arm, repetitions=args.repetitions_per_arm,
            )
            for index_case, case in selected_cases:
                order = _arm_order(index_case)
                for warmup_index, mode in enumerate(order[:2]):
                    row = sample(case, mode, warmup_index)
                    report["warmups"].append({
                        "case_id": row["case_id"], "mode": mode,
                        "output_token_ids_sha256": row["output_token_ids_sha256"],
                        "engagement": row["engagement"],
                    })
                measured_counts = {"parent": 0, "candidate": 0}
                for slot, mode in enumerate(order):
                    row = sample(case, mode, measured_counts[mode])
                    measured_counts[mode] += 1
                    row.update({"sequence_slot": slot, "case_sequence": list(order)})
                    report["samples"].append(row)
                    _write_json(args.output, report)
                    print(f"[measure {measured_counts[mode]-1}] {mode} {row['case_id']} slot={slot} "
                          f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}", flush=True)
            report["summary"] = summarize_engine_ab(
                report["samples"], chunk_size=args.prefill_chunk_size,
            )
            report["status"] = "completed"
            report["memory_before_close"] = memory_stats()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        _write_json(args.output, report)
        raise
    finally:
        os.environ[PAIR_ENV] = "0"
        runner_module.wide_f16_activation_pair = original_scope
        wide.f32_to_f16 = original_cast
        if generator is not None:
            generator.close()
        report["memory_after_close"] = memory_stats()
        if any(report["memory_after_close"].get(key, 0) != 0 for key in ("active_allocations", "current_allocated_bytes")):
            report["status"] = "failed_teardown"
        _write_json(args.output, report)
    if report["status"] != "completed":
        raise SystemExit(report["status"])
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "cases": report["protocol"]["case_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
