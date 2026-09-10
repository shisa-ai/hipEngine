#!/usr/bin/env python3
"""Measure the named decode step at the P8 transition-graph probe's own context.

The retained P8 full-transition artifact reports **194.758 -> 61.910 ms/step
(3.15x)**, but its eager arm is probe-local: it forces
``HIPENGINE_QWEN4_EXP_MOE_GRAPH=0`` and drives every layer from a script-level
launch loop. Neither property belongs to the named decode path, so that ratio
cannot be used to rank the P8 graph against other owners.

This harness supplies the missing denominator. It uses the same strict profile,
the same prompt, and the same context window as the probe, but measures through
production ``Qwen4ExpDenseRunner.step()`` with:

* the per-layer MoE graph cache on (the named default) and off, so the existing
  graph win is separated from the proposed whole-transition win; and
* the device-argmax output boundary (``capture_logits=False``, matching the
  graph arm's token readback) and the full host-logit boundary.

Arms are counterbalanced, a first warm arm is discarded, and tracked device
memory is reported construct-to-close. Diagnostic only: this is a denominator
for an Amdahl row, not a promotion gate and not a canonical speed claim.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata  # noqa: E402
from scripts.qwen4exp_stateful_layer_graph_probe import _make_generator  # noqa: E402

# Retained rows from
# benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json
P8_EAGER_MEDIAN_MS = 194.75767650146736
P8_GRAPH_MEDIAN_MS = 61.909654999908525

MODES = (
    "graph_on_device_argmax",
    "graph_off_device_argmax",
    "graph_on_host_logits",
    "graph_off_host_logits",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--pair-repetitions", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace, *, command: list[str]) -> dict[str, Any]:
    if not args.model_root.is_dir() or not args.prompt_file.is_file():
        raise ValueError("model root and prompt file must exist")
    if args.samples < 3 or args.pair_repetitions < 1:
        raise ValueError("at least three samples and one repetition required")
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    reset_memory_stats()
    generator = None
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_p8_production_denominator",
        "status": "unknown",
        "model": str(args.model_root),
        "prompt_tokens": args.prompt_tokens,
        "samples": args.samples,
        "host": _host_metadata(),
        "source": _git_metadata(ROOT),
        "command": command,
    }
    try:
        generator, resolved = _make_generator(args)
        payload["profile"] = {
            "name": getattr(getattr(resolved, "profile", None), "value", "strict"),
            "manifest_sha256": getattr(resolved, "manifest_sha256", None),
        }
        runner = generator.runner
        runtime = runner.runtime
        cache = runner.moe_graph_cache
        payload["moe_graph_cache_available"] = cache is not None
        payload["moe_graph_cache_default_enabled"] = bool(cache is not None and cache.enabled)
        ids = [int(item) for item in generator.tokenizer.encode(args.prompt_file.read_text())]
        ids = ids[: args.prompt_tokens]
        if not ids:
            raise ValueError("prompt tokenization is empty")
        payload["prompt_ids"] = ids

        def arm(mode: str) -> dict[str, Any]:
            graph_on = mode.startswith("graph_on")
            device_argmax = mode.endswith("device_argmax")
            if cache is not None:
                cache._enabled = graph_on
            runner.reset()
            result = runner.prefill(list(ids))
            start_position = int(runner.position)
            token = int(result.token_id)
            for _ in range(args.warmup_steps):
                result = runner.step(
                    token, capture_logits=not device_argmax, capture_target_hidden=False
                )
                token = int(result.token_id)
            runtime.device_synchronize()
            measured_start = int(runner.position)
            walls: list[float] = []
            tokens: list[int] = []
            for _ in range(args.samples):
                started = time.perf_counter()
                result = runner.step(
                    token, capture_logits=not device_argmax, capture_target_hidden=False
                )
                token = int(result.token_id)
                runtime.device_synchronize()
                walls.append((time.perf_counter() - started) * 1e3)
                tokens.append(token)
            return {
                "mode": mode,
                "prefill_position": start_position,
                "measured_start_position": measured_start,
                "measured_end_position": int(runner.position),
                "median_ms": statistics.median(walls),
                "mean_ms": statistics.fmean(walls),
                "min_ms": min(walls),
                "max_ms": max(walls),
                "ms": walls,
                "tokens": tokens,
                "moe_graph_stats": dict(cache.stats) if cache is not None else None,
            }

        # The first arm after load pays first-touch weight paging and clock ramp;
        # discard it so no measured row carries that transient.
        payload["discarded_warm_arm"] = arm(MODES[0])
        allocations_after_warm = memory_stats()["active_allocations"]

        rows: list[dict[str, Any]] = []
        for repetition in range(args.pair_repetitions):
            order = MODES if repetition % 2 == 0 else tuple(reversed(MODES))
            for mode in order:
                row = arm(mode)
                row["repetition"] = repetition
                rows.append(row)
                print(
                    f"rep={repetition} mode={mode} median={row['median_ms']:.3f} ms "
                    f"min={row['min_ms']:.3f} max={row['max_ms']:.3f}",
                    flush=True,
                )
        payload["rows"] = rows
        payload["lifecycle"] = {
            "active_allocations_after_warm_arm": allocations_after_warm,
            "active_allocations_after_measurement": memory_stats()["active_allocations"],
            "steady_allocation_growth": (
                memory_stats()["active_allocations"] - allocations_after_warm
            ),
        }

        summary: dict[str, Any] = {
            "p8_probe_eager_median_ms": P8_EAGER_MEDIAN_MS,
            "p8_probe_graph_median_ms": P8_GRAPH_MEDIAN_MS,
        }
        for mode in MODES:
            medians = [row["median_ms"] for row in rows if row["mode"] == mode]
            median_of_medians = statistics.median(medians)
            summary[mode] = {
                "repetition_medians_ms": medians,
                "median_of_medians_ms": median_of_medians,
                "p8_graph_speedup_over_this_arm": median_of_medians / P8_GRAPH_MEDIAN_MS,
            }
        named = summary["graph_on_device_argmax"]["median_of_medians_ms"]
        summary["named_default_ms"] = named
        summary["p8_graph_named_speedup"] = named / P8_GRAPH_MEDIAN_MS
        summary["p8_graph_named_saving_ms"] = named - P8_GRAPH_MEDIAN_MS
        summary["moe_graph_cache_speedup"] = (
            summary["graph_off_device_argmax"]["median_of_medians_ms"] / named
        )
        summary["probe_eager_arm_overstatement"] = P8_EAGER_MEDIAN_MS / named
        payload["summary"] = summary
        payload["status"] = "measured"
        return payload
    finally:
        if generator is not None:
            generator.close()
        payload.setdefault("lifecycle", {})["after_close"] = memory_stats()


def main() -> None:
    args = build_parser().parse_args()
    command = [Path(sys.argv[0]).name, *sys.argv[1:]]
    payload = run(args, command=command)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps(payload.get("summary", {}), indent=1))


if __name__ == "__main__":
    main()
