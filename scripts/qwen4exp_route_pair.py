#!/usr/bin/env python3
"""Counterbalanced same-process Qwen4Exp prefill or decode route A/B.

The named production profile is constructed once. ``--override`` values define
one diagnostic route and are toggled against the values bound by the profile.
A route that is request-local runner state is named with its own flag instead
(``--gdn-tile16-parent``), because a retired process switch would leave the
measured route running and report a self-comparison. Each pair reverses its
first route, so monotonic thermal drift cannot masquerade as a route win.
Decode mode excludes prefill from timing and hashes the complete generated token
sequence. Override runs are never named-profile evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
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
    samples: list[dict[str, Any]], *, prompt_tokens: int, unit_name: str = "prompt_tokens"
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
        unit_name: int(prompt_tokens),
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


RETIRED_OVERRIDES = {
    "HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL": "--gdn-tile16-parent",
}


def _reject_retired_overrides(overrides: dict[str, str | None]) -> None:
    """Refuse an override the runtime no longer reads.

    The retired GDN tile-16 switch is still set by old command lines and by
    archived artifact recipes. Accepting it would leave both arms on the
    production owner and publish a self-comparison as a route A/B, so the
    harness names its replacement instead.
    """

    for key in overrides:
        replacement = RETIRED_OVERRIDES.get(key)
        if replacement is not None:
            raise SystemExit(
                f"{key} is retired; the runtime no longer reads it. "
                f"Use {replacement} to measure the parent arm."
            )


def _require_arm_control(
    overrides: dict[str, str | None],
    *,
    gdn_tile16_parent: bool,
    mode: str = "prefill",
) -> None:
    """Both routes must differ: an arm with no control measures itself."""

    if gdn_tile16_parent and mode != "prefill":
        raise SystemExit("--gdn-tile16-parent applies to prefill mode only")
    if not overrides and not gdn_tile16_parent:
        raise SystemExit("--override or --gdn-tile16-parent is required")


def _arm_scope(route: str, runner: Any, *, gdn_tile16_parent: bool):
    """The runner-local control an arm needs; env overrides are already set."""

    if route != "override" or not gdn_tile16_parent:
        return contextlib.nullcontext()
    from hipengine.runtime.qwen4_exp_runner import gdn_tile16_prefill

    return gdn_tile16_prefill(runner, False)


# The GDN prefill owners the tile16 branch and the columnwarp parent reach. The
# branch calls a named variant through the registry and the default tile-16 body
# directly, and the parent directly, so both routes are wrapped: an arm then
# reports a counted launch instead of a status over a route that never ran.
GDN_PREFILL_OWNERS = {
    "qwen4exp_gdn_tiled16_prefill": "qwen4_exp_gdn_prefill_tiled16_f32",
    "qwen4exp_gdn_columnwarps_prefill": "qwen4_exp_gdn_prefill_columnwarps_f32",
    "qwen4exp_sigmoid_strict_prefill": "qwen4_exp_gdn_prefill_f32",
}
TILE16_FAMILY = {
    "qwen4exp_gdn_tiled16_prefill",
    "qwen4exp_gdn_tiled16_multi_prefill",
    "qwen4exp_gdn_tiled16_dpp_prefill",
    "qwen4exp_gdn_rowsplit_prefill",
}
COLUMNWARP_OWNER = "qwen4exp_gdn_columnwarps_prefill"


@contextlib.contextmanager
def _count_prefill_dispatches(counts: dict[str, int]):
    """Count the GDN prefill owner that actually launches inside the window.

    The registry resolver is keyword-only, so the owner is named from its own
    arguments. Every wrapper carries the owner's attributes through
    ``functools.wraps``: the mixer reads ``supports_any_rows``,
    ``publishes_f16_image`` and ``supports_state_rows`` off the resolved
    callable, and a bare wrapper would silently turn the row-split owner into a
    fixed-width one and change what this instrumentation measures.
    """

    from hipengine.runtime import qwen4_exp_runner as runner_module

    def counted(name: str, function: Any) -> Any:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] = counts.get(name, 0) + 1
            return function(*args, **kwargs)

        return wrapper

    saved = {
        attribute: getattr(runner_module, attribute)
        for attribute in GDN_PREFILL_OWNERS.values()
    }
    for name, attribute in GDN_PREFILL_OWNERS.items():
        setattr(runner_module, attribute, counted(name, saved[attribute]))
    original_resolve = runner_module.resolve

    def resolve(*args: Any, **kwargs: Any) -> Any:
        function = original_resolve(*args, **kwargs)
        if (
            kwargs.get("layer") == "gdn_recurrence_norm_gate"
            and callable(function)
        ):
            return counted(str(kwargs.get("variant") or "default"), function)
        return function

    runner_module.resolve = resolve
    try:
        yield counts
    finally:
        runner_module.resolve = original_resolve
        for attribute, function in saved.items():
            setattr(runner_module, attribute, function)


def _sum_prefill_dispatches(samples: list[dict[str, Any]], route: str) -> dict[str, int]:
    total: dict[str, int] = {}
    for row in samples:
        if row["route"] != route:
            continue
        for name, count in row.get("prefill_dispatches", {}).items():
            total[name] = total.get(name, 0) + int(count)
    return total


def _parent_arm_counted(dispatches: dict[str, dict[str, int]]) -> bool:
    """A parent arm is an arm only if the counted production owner moved."""

    return (
        bool(set(dispatches["bound"]) & TILE16_FAMILY)
        and not (set(dispatches["override"]) & TILE16_FAMILY)
        and dispatches["override"].get(COLUMNWARP_OWNER, 0) > 0
    )


def _measure_route(
    route: str,
    runner: Any,
    token_ids: list[int],
    *,
    mode: str,
    decode_transitions: int,
    gdn_tile16_parent: bool,
) -> dict[str, Any]:
    """Run one arm sample under its controls, counting the prefill owner."""

    counts: dict[str, int] = {}
    with (
        _arm_scope(route, runner, gdn_tile16_parent=gdn_tile16_parent),
        _count_prefill_dispatches(counts),
    ):
        sample = _run_route(
            runner, token_ids, mode=mode, decode_transitions=decode_transitions,
        )
    sample["prefill_dispatches"] = dict(counts)
    return sample


def _run_route(
    runner: Any,
    token_ids: list[int],
    *,
    mode: str,
    decode_transitions: int,
) -> dict[str, Any]:
    """Run one synchronized route sample and return its timing/identity."""

    prefill_kwargs = {
        "capture_logits": False,
        "capture_target_hidden": False,
    }
    step_kwargs = {
        "capture_logits": False,
        "capture_target_hidden": False,
        "token_id_resident": True,
    }
    if mode == "prefill":
        started = time.perf_counter()
        result = runner.prefill(token_ids, **prefill_kwargs)
        runner.runtime.device_synchronize()
        return {
            "seconds": time.perf_counter() - started,
            "token_id": int(result.token_id),
            "logits_sha256": (
                hashlib.sha256(result.logits.tobytes()).hexdigest()
                if result.logits is not None
                else hashlib.sha256(str(int(result.token_id)).encode()).hexdigest()
            ),
        }
    if mode != "decode":
        raise ValueError(f"unsupported route mode: {mode}")
    if decode_transitions <= 0:
        raise ValueError("decode_transitions must be positive")
    result = runner.prefill(token_ids, **prefill_kwargs)
    runner.runtime.device_synchronize()
    output_ids = [int(result.token_id)]
    current = int(result.token_id)
    started = time.perf_counter()
    for _ in range(decode_transitions):
        result = runner.step(current, **step_kwargs)
        current = int(result.token_id)
        output_ids.append(current)
    runner.runtime.device_synchronize()
    return {
        "seconds": time.perf_counter() - started,
        "token_id": current,
        "logits_sha256": hashlib.sha256(
            json.dumps(output_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "output_token_ids": output_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
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
    parser.add_argument("--decode-transitions", type=int, default=128)
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="HIPENGINE_KEY=VALUE",
    )
    parser.add_argument(
        "--gdn-tile16-parent",
        action="store_true",
        help=(
            "Pin the columnwarp GDN prefill parent on the owning runner for "
            "the override arm; replaces the retired "
            "HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL override"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        overrides = _parse_overrides(args.override)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _reject_retired_overrides(overrides)
    _require_arm_control(
        overrides, gdn_tile16_parent=args.gdn_tile16_parent, mode=args.mode
    )
    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.mode == "decode" and args.decode_transitions <= 0:
        raise SystemExit("--decode-transitions must be positive in decode mode")

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
        "kind": f"qwen4exp_{args.mode}_route_counterbalanced_ab",
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
            "decode_transitions": (
                int(args.decode_transitions) if args.mode == "decode" else None
            ),
            "timing_boundary": (
                "synchronized runner.prefill including first greedy output"
                if args.mode == "prefill"
                else "prefill excluded; synchronized autoregressive runner.step transitions"
            ),
            "override_stage": "post_profile_binder_pre_warmup_and_measurement",
        },
        "overrides": dict(overrides),
        "runner_controls": (
            {"gdn_tile16_prefill": False} if args.gdn_tile16_parent else {}
        ),
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
    invalid_arms = False
    try:
        for case_index, case in enumerate(cases):
            ids = case["token_ids"]
            for warmup in range(args.warmups):
                for route in _counterbalanced_order(warmup + case_index):
                    _set_environment(route_values[route])
                    sample = _measure_route(
                        route,
                        generator.runner,
                        ids,
                        mode=args.mode,
                        decode_transitions=int(args.decode_transitions),
                        gdn_tile16_parent=args.gdn_tile16_parent,
                    )
                    report["warmups"].append(
                        {
                            "case_id": case["id"],
                            "category": case["category"],
                            "warmup": warmup,
                            "route": route,
                            "token_id": sample["token_id"],
                            "logits_sha256": sample["logits_sha256"],
                        }
                    )
            for pair in range(args.pairs):
                for order_index, route in enumerate(
                    _counterbalanced_order(pair + case_index)
                ):
                    _set_environment(route_values[route])
                    sample = _measure_route(
                        route,
                        generator.runner,
                        ids,
                        mode=args.mode,
                        decode_transitions=int(args.decode_transitions),
                        gdn_tile16_parent=args.gdn_tile16_parent,
                    )
                    row = {
                        "case_id": case["id"],
                        "category": case["category"],
                        "prompt_tokens": len(ids),
                        "pair": pair,
                        "order_index": order_index,
                        "route": route,
                        **sample,
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
                prompt_tokens=(
                    len(case["token_ids"])
                    if args.mode == "prefill"
                    else int(args.decode_transitions)
                ),
                unit_name=(
                    "prompt_tokens" if args.mode == "prefill" else "decode_transitions"
                ),
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
        dispatches = {
            route: _sum_prefill_dispatches(report["samples"], route)
            for route in ("bound", "override")
        }
        report["summary"]["prefill_dispatches"] = dispatches
        if args.gdn_tile16_parent:
            counted = _parent_arm_counted(dispatches)
            report["summary"]["parent_arm_counted"] = counted
            invalid_arms = not counted
            if invalid_arms:
                print(
                    "the parent arm did not move the GDN prefill owner; "
                    f"bound={dispatches['bound']} "
                    f"override={dispatches['override']}",
                    flush=True,
                )
        report["status"] = "invalid_arms" if invalid_arms else "completed"
    finally:
        _set_environment(bound_values)
        generator.close()
    report["memory_after_close"] = memory_stats()
    _write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    return 1 if invalid_arms else 0


if __name__ == "__main__":
    raise SystemExit(main())
