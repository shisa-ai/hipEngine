#!/usr/bin/env python3
"""Counterbalanced same-process Qwen4Exp prefill route A/B.

The named production profile is constructed once. ``--override`` values define
one diagnostic route and are toggled against the values bound by the profile.
Each pair reverses its first route, so monotonic thermal drift cannot masquerade
as a route win. Override runs are never named-profile evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_profile_gap import _parse_overrides


_T95 = (
    0.0,
    12.706204736,
    4.302652730,
    3.182446305,
    2.776445105,
    2.570581836,
    2.446911851,
    2.364624252,
    2.306004135,
    2.262157163,
    2.228138852,
    2.200985160,
    2.178812830,
    2.160368656,
    2.144786688,
    2.131449546,
    2.119905299,
    2.109815578,
    2.100922040,
    2.093024054,
    2.085963447,
    2.079613845,
    2.073873068,
    2.068657610,
    2.063898562,
    2.059538553,
    2.055529439,
    2.051830516,
    2.048407142,
    2.045229642,
    2.042272456,
)


def _mean_ci95(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return [mean, mean]
    degrees = len(values) - 1
    critical = _T95[degrees] if degrees < len(_T95) else 1.959963985
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def _counterbalanced_order(pair: int) -> tuple[str, str]:
    return ("bound", "override") if pair % 2 == 0 else ("override", "bound")


def _cv_percent(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return 100.0 * statistics.stdev(values) / mean if mean else 0.0


def _route_summary(rows: list[dict[str, Any]], prompt_tokens: int) -> dict[str, Any]:
    seconds = [float(row["seconds"]) for row in rows]
    hashes = {str(row["logits_sha256"]) for row in rows}
    token_ids = {int(row["token_id"]) for row in rows}
    return {
        "count": len(rows),
        "seconds": seconds,
        "sum_seconds": sum(seconds),
        "mean_seconds": statistics.mean(seconds),
        "median_seconds": statistics.median(seconds),
        "cv_percent": _cv_percent(seconds),
        "tok_s": prompt_tokens * len(rows) / sum(seconds),
        "repeat_exact": len(hashes) == 1 and len(token_ids) == 1,
        "logits_sha256": sorted(hashes),
        "token_ids": sorted(token_ids),
    }


def _paired_summary(
    samples: list[dict[str, Any]], *, prompt_tokens: int
) -> dict[str, Any]:
    routes = {
        route: _route_summary(
            [row for row in samples if row["route"] == route], prompt_tokens
        )
        for route in ("bound", "override")
    }
    pairs: list[dict[str, Any]] = []
    for pair in sorted({int(row["pair"]) for row in samples}):
        pair_rows = [row for row in samples if int(row["pair"]) == pair]
        by_route = {str(row["route"]): row for row in pair_rows}
        if set(by_route) != {"bound", "override"} or len(pair_rows) != 2:
            raise ValueError(f"pair {pair} does not contain both routes exactly once")
        bound_seconds = float(by_route["bound"]["seconds"])
        override_seconds = float(by_route["override"]["seconds"])
        pairs.append(
            {
                "pair": pair,
                "order": [str(row["route"]) for row in pair_rows],
                "bound_seconds": bound_seconds,
                "override_seconds": override_seconds,
                "throughput_ratio_override_vs_bound": bound_seconds
                / override_seconds,
            }
        )
    ratios = [row["throughput_ratio_override_vs_bound"] for row in pairs]
    return {
        "prompt_tokens": int(prompt_tokens),
        "counterbalanced": True,
        "routes": routes,
        "pairs": pairs,
        "ratio": {
            "count": len(ratios),
            "mean": statistics.mean(ratios),
            "mean_95ci": _mean_ci95(ratios),
            "median": statistics.median(ratios),
            "min": min(ratios),
            "max": max(ratios),
        },
        "cross_route_logits_exact": (
            routes["bound"]["logits_sha256"]
            == routes["override"]["logits_sha256"]
            and routes["bound"]["token_ids"] == routes["override"]["token_ids"]
        ),
    }


def _set_environment(values: dict[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt-file", type=Path)
    input_group.add_argument("--fixture", type=Path)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Fixture case ID to run; repeat to select multiple cases",
    )
    parser.add_argument("--expected-prompt-tokens", type=int)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=768)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        required=True,
        metavar="HIPENGINE_KEY=VALUE",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        overrides = _parse_overrides(args.override)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)
    if args.compiler_version_file is not None:
        os.environ.setdefault(
            "HIPENGINE_COMPILER_VERSION_FILE",
            str(args.compiler_version_file.resolve()),
        )
    if args.require_cached_build:
        os.environ.setdefault("HIPENGINE_REQUIRE_CACHED_BUILD", "1")

    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model
    from scripts.qwen4exp_canonical_ar_bench import (
        _git_metadata,
        _host_metadata,
        _write_json,
        load_fixture,
        token_ids_sha256,
    )

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    model_root = args.model_root.resolve()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=args.max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    prompt_bytes = args.prompt_file.read_bytes() if args.prompt_file else None
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_prefill_route_counterbalanced_ab",
        "status": "running",
        "configuration_class": "diagnostic_counterbalanced_post_binder_override",
        "named_profile_intact": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": list(sys.argv),
        "model_root": str(model_root),
        "model_architecture": index.architecture,
        "prompt_file": str(args.prompt_file) if args.prompt_file else None,
        "prompt_file_sha256": (
            hashlib.sha256(prompt_bytes).hexdigest() if prompt_bytes else None
        ),
        "fixture": str(args.fixture) if args.fixture else None,
        "fixture_sha256": None,
        "selected_case_ids": list(args.case_id),
        "manifest_sha256": resolved.manifest_sha256,
        "strict_manifest_sha256": resolved.strict_manifest_sha256,
        "fell_back_to_strict": resolved.fell_back_to_strict,
        "protocol": {
            "pairs": int(args.pairs),
            "warmups_per_route": int(args.warmups),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "override_stage": "post_profile_binder_pre_warmup_and_measurement",
        },
        "overrides": dict(overrides),
        "bound_override_env": {},
        "effective_override_env": {},
        "warmups": [],
        "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, report)

    generator = resolved.construct_generator(factory)
    bound_values = {key: os.environ.get(key) for key in overrides}
    report["bound_override_env"] = dict(bound_values)
    _set_environment(overrides)
    report["effective_override_env"] = {
        key: os.environ.get(key) for key in overrides
    }
    if args.fixture is not None:
        if not args.case_id:
            _set_environment(bound_values)
            generator.close()
            raise RuntimeError("--fixture requires at least one --case-id")
        if args.expected_prompt_tokens is not None:
            _set_environment(bound_values)
            generator.close()
            raise RuntimeError("--expected-prompt-tokens applies only to --prompt-file")
        fixture, fixture_sha256 = load_fixture(args.fixture)
        fixture_cases = {str(row["id"]): row for row in fixture["cases"]}
        missing = [case_id for case_id in args.case_id if case_id not in fixture_cases]
        if missing:
            _set_environment(bound_values)
            generator.close()
            raise RuntimeError(f"fixture does not contain cases: {missing}")
        cases = [
            {
                "id": case_id,
                "category": str(fixture_cases[case_id]["category"]),
                "token_ids": [
                    int(value)
                    for value in fixture_cases[case_id]["prompt_token_ids"]
                ],
            }
            for case_id in args.case_id
        ]
        report["fixture_sha256"] = fixture_sha256
    else:
        if args.case_id:
            _set_environment(bound_values)
            generator.close()
            raise RuntimeError("--case-id requires --fixture")
        assert prompt_bytes is not None
        ids = generator.tokenizer.encode(prompt_bytes.decode("utf-8"))
        if (
            args.expected_prompt_tokens is not None
            and len(ids) != args.expected_prompt_tokens
        ):
            _set_environment(bound_values)
            generator.close()
            raise RuntimeError(
                f"expected {args.expected_prompt_tokens} prompt tokens, got {len(ids)}"
            )
        cases = [
            {
                "id": args.prompt_file.stem,
                "category": "diagnostic",
                "token_ids": ids,
            }
        ]
    report["cases"] = [
        {
            "id": case["id"],
            "category": case["category"],
            "prompt_tokens": len(case["token_ids"]),
            "prompt_token_ids_sha256": token_ids_sha256(case["token_ids"]),
        }
        for case in cases
    ]

    route_values = {"bound": bound_values, "override": overrides}
    try:
        for case_index, case in enumerate(cases):
            ids = case["token_ids"]
            for warmup in range(args.warmups):
                for route in _counterbalanced_order(warmup + case_index):
                    _set_environment(route_values[route])
                    result = generator.runner.prefill(ids)
                    generator.runner.runtime.device_synchronize()
                    report["warmups"].append(
                        {
                            "case_id": case["id"],
                            "category": case["category"],
                            "warmup": warmup,
                            "route": route,
                            "token_id": int(result.token_id),
                            "logits_sha256": hashlib.sha256(
                                result.logits.tobytes()
                            ).hexdigest(),
                        }
                    )
            for pair in range(args.pairs):
                for order_index, route in enumerate(
                    _counterbalanced_order(pair + case_index)
                ):
                    _set_environment(route_values[route])
                    started = time.perf_counter()
                    result = generator.runner.prefill(ids)
                    generator.runner.runtime.device_synchronize()
                    row = {
                        "case_id": case["id"],
                        "category": case["category"],
                        "prompt_tokens": len(ids),
                        "pair": pair,
                        "order_index": order_index,
                        "route": route,
                        "seconds": time.perf_counter() - started,
                        "token_id": int(result.token_id),
                        "logits_sha256": hashlib.sha256(
                            result.logits.tobytes()
                        ).hexdigest(),
                    }
                    report["samples"].append(row)
                    _write_json(args.output, report)
                    print(
                        f"case={case['id']} pair={pair} order={order_index} "
                        f"route={route} seconds={row['seconds']:.6f}",
                        flush=True,
                    )
        report["case_summaries"] = {
            case["id"]: _paired_summary(
                [row for row in report["samples"] if row["case_id"] == case["id"]],
                prompt_tokens=len(case["token_ids"]),
            )
            for case in cases
        }
        report["summary"] = {
            "case_count": len(cases),
            "pair_count": len(cases) * args.pairs,
            "all_bound_repeat_exact": all(
                summary["routes"]["bound"]["repeat_exact"]
                for summary in report["case_summaries"].values()
            ),
            "all_override_repeat_exact": all(
                summary["routes"]["override"]["repeat_exact"]
                for summary in report["case_summaries"].values()
            ),
            "all_pairs_override_faster": all(
                row["throughput_ratio_override_vs_bound"] > 1.0
                for summary in report["case_summaries"].values()
                for row in summary["pairs"]
            ),
            "all_cross_route_logits_exact": all(
                summary["cross_route_logits_exact"]
                for summary in report["case_summaries"].values()
            ),
        }
        report["status"] = "completed"
    finally:
        _set_environment(bound_values)
        generator.close()
    report["memory_after_close"] = memory_stats()
    _write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
