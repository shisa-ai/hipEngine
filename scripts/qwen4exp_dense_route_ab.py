#!/usr/bin/env python3
"""Matched A/B for the layer-scoped Q8_0 dense prefill routes.

Three arms, one process, one model load, interleaved:

===============  ==================================================
``exact``        both selectors cleared -> the exact coltile chain
``wmma-16-47``   the named production default -> ``wmma_prefill_*``
``wide-16-47``   ``HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=1`` at 16-47
===============  ==================================================

The selectors are post-binder: the named production binder writes them during
its own pass, so they are flipped after the generator is constructed and a
pre-binder value is inert.

The arms are interleaved round-robin and rotated between repetitions, so clock,
thermal and allocator drift are shared across arms rather than assigned to one
of them. Each arm warms once after its flip, because a flip changes the
dispatch the first request after it resolves. Every arm records the logits
digest it produced and a launch census, so the run shows which kernel served
each arm instead of assuming it.

This is a route A/B, not a topline rate row: it measures one category at three
prompt lengths in one process on one host. Read it as "what the dense Q8_0
prefill route is worth on this shape", and pair it with the quality gate for the
arm it favours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.kernels import launch_census  # noqa: E402
from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    _host_metadata,
    load_fixture,
)

DEFAULT_FIXTURE = Path("benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json")
DEFAULT_CASES = ("code-p512", "code-p1024", "code-p4096")
DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
WMMA_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"
DENSE_WIDE_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"
DENSE_WIDE_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"
LAYER_SCOPE = ",".join(str(layer) for layer in range(16, 48))


@dataclass(frozen=True)
class Arm:
    name: str
    env: Mapping[str, str]
    chain: str


ARMS: tuple[Arm, ...] = (
    Arm(
        name="exact",
        env={WMMA_LAYERS_ENV: "", DENSE_WIDE_ENV: "0", DENSE_WIDE_LAYERS_ENV: ""},
        chain="coltile8_rowbatch4_wave_scale_f32_f32_out",
    ),
    Arm(
        name="wmma-16-47",
        env={
            WMMA_LAYERS_ENV: LAYER_SCOPE,
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
        },
        chain="wmma_prefill_f32_f32_out",
    ),
    Arm(
        name="wide-16-47",
        env={
            WMMA_LAYERS_ENV: LAYER_SCOPE,
            DENSE_WIDE_ENV: "1",
            DENSE_WIDE_LAYERS_ENV: LAYER_SCOPE,
        },
        chain="dense_wide256_f32_f32_out",
    ),
)
ARM_ENV_KEYS = (WMMA_LAYERS_ENV, DENSE_WIDE_ENV, DENSE_WIDE_LAYERS_ENV)


def _apply_arm(arm: Arm) -> None:
    os.environ.update(arm.env)


def _merge_census(target: dict[tuple, int], snapshot: Mapping[str, Any]) -> None:
    for row in snapshot.get("rows", ()):
        key = (
            row["role"],
            row["quant"],
            row["symbol"],
            int(row["rows"]),
            int(row["in_features"]),
            int(row["out_features"]),
        )
        target[key] = target.get(key, 0) + int(row["launches"])


def _census_rows(merged: Mapping[tuple, int]) -> list[dict[str, Any]]:
    rows = [
        {
            "role": key[0],
            "quant": key[1],
            "symbol": key[2],
            "rows": key[3],
            "in_features": key[4],
            "out_features": key[5],
            "launches": count,
        }
        for key, count in merged.items()
    ]
    rows.sort(key=lambda row: (-row["launches"], row["role"], row["symbol"]))
    return rows


def _summarize(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "n": len(ordered),
    }


def _digest(result: Any) -> str:
    logits = getattr(result, "logits", None)
    if logits is None:
        return ""
    return hashlib.sha256(logits.tobytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The census is what shows which kernel served each arm; it is off by
    # default and reads its state once.
    os.environ["HIPENGINE_KERNEL_CENSUS"] = "1"
    os.environ.setdefault("HIPENGINE_HIP_ARCH", args.hip_arch)

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

    fixture, fixture_sha256 = load_fixture(args.fixture)
    wanted = tuple(args.case_id) if args.case_id else DEFAULT_CASES
    cases = [row for row in fixture["cases"] if str(row["id"]) in wanted]
    if len(cases) != len(wanted):
        raise SystemExit(f"unknown fixture case in {wanted}")
    cases.sort(key=lambda row: int(row["prompt_tokens"]))
    max_sequence_length = max(int(row["prompt_tokens"]) for row in cases) + 8

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
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    bound_env = {key: os.environ.get(key) for key in ARM_ENV_KEYS}
    generator = resolved.construct_generator(factory)
    try:
        runner = generator.runner
        if runner is None:
            raise SystemExit("runner is not resident")
        measurements: dict[str, dict[str, list[float]]] = {
            arm.name: {str(row["id"]): [] for row in cases} for arm in ARMS
        }
        digests: dict[str, dict[str, str]] = {
            arm.name: {} for arm in ARMS
        }
        tokens: dict[str, dict[str, int]] = {arm.name: {} for arm in ARMS}
        censuses: dict[str, dict[tuple, int]] = {arm.name: {} for arm in ARMS}
        effective_env: dict[str, dict[str, str | None]] = {}

        for row in cases:
            case_id = str(row["id"])
            ids = [int(token) for token in row["prompt_token_ids"]]
            for arm in ARMS:
                _apply_arm(arm)
                for _ in range(int(args.warmups)):
                    runner.prefill(ids)
                    runner.runtime.device_synchronize()
            for repetition in range(int(args.repetitions)):
                order = ARMS[repetition % len(ARMS):] + ARMS[: repetition % len(ARMS)]
                for arm in order:
                    _apply_arm(arm)
                    launch_census.reset()
                    started = time.perf_counter()
                    result = runner.prefill(ids)
                    runner.runtime.device_synchronize()
                    elapsed = time.perf_counter() - started
                    measurements[arm.name][case_id].append(elapsed)
                    _merge_census(censuses[arm.name], launch_census.snapshot())
                    digests[arm.name][case_id] = _digest(result)
                    tokens[arm.name][case_id] = int(result.token_id)
                print(
                    f"{case_id} rep {repetition + 1}/{args.repetitions}: "
                    + "  ".join(
                        f"{arm.name}={measurements[arm.name][case_id][-1]:.3f}s"
                        for arm in ARMS
                    ),
                    flush=True,
                )
            effective_env[case_id] = {
                key: os.environ.get(key) for key in ARM_ENV_KEYS
            }
        for arm in ARMS:
            _apply_arm(arm)
            effective_env[arm.name] = {
                key: os.environ.get(key) for key in ARM_ENV_KEYS
            }
        provenance = collect_artifact_provenance(
            repo_root=ROOT,
            configured_backend="hip_gfx1151",
            resolved_backend=str(getattr(runner, "backend", "hip_gfx1151")),
            target_arch=str(os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151")),
            model_path=model_root,
            quant=QWEN4_EXP_QUANTS[1],
            kv_dtype="bf16",
            command=[Path(sys.argv[0]).name, *sys.argv[1:]],
            environment={
                "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
                "HIPENGINE_KERNEL_CENSUS": "1",
                **{key: bound_env[key] for key in ARM_ENV_KEYS},
            },
            build_profile="production",
            timing_protocol="interleaved_arm_prefill_wall_median_v1",
            warmups=int(args.warmups),
            repetitions=int(args.repetitions),
        )
    finally:
        for key, value in bound_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        generator.close()

    arms: dict[str, Any] = {}
    for arm in ARMS:
        per_case = {
            case_id: {
                "wall_seconds": measurements[arm.name][case_id],
                "summary": _summarize(measurements[arm.name][case_id]),
                "logits_sha256": digests[arm.name][case_id],
                "token_id": tokens[arm.name][case_id],
            }
            for case_id in measurements[arm.name]
        }
        rows = _census_rows(censuses[arm.name])
        arms[arm.name] = {
            "env": dict(arm.env),
            "chain": arm.chain,
            "cases": per_case,
            "census": {
                "total_launches": sum(int(r["launches"]) for r in rows),
                "distinct_shapes": len(rows),
                "rows": rows,
            },
        }

    deltas: dict[str, Any] = {}
    for case_id in measurements["exact"]:
        reference = arms["exact"]["cases"][case_id]["summary"]["median"]
        deltas[case_id] = {
            arm.name: {
                "median_seconds": arms[arm.name]["cases"][case_id]["summary"]["median"],
                "saved_seconds": reference - arms[arm.name]["cases"][case_id]["summary"]["median"],
                "speedup": reference / arms[arm.name]["cases"][case_id]["summary"]["median"],
                "same_logits_as_exact": (
                    arms[arm.name]["cases"][case_id]["logits_sha256"]
                    == arms["exact"]["cases"][case_id]["logits_sha256"]
                ),
            }
            for arm in ARMS
        }

    artifact = {
        "schema": 1,
        "kind": "qwen4exp_q8_dense_route_ab",
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "provenance": provenance,
        "host": _host_metadata(),
        "model_root": str(model_root),
        "model": QWEN4_EXP_MODEL,
        "quant": QWEN4_EXP_QUANTS[1],
        "profile": "production",
        "manifest_sha256": resolved.manifest_sha256,
        "strict_manifest_sha256": resolved.strict_manifest_sha256,
        "fell_back_to_strict": resolved.fell_back_to_strict,
        "fixture": str((ROOT / args.fixture).resolve()),
        "fixture_sha256": fixture_sha256,
        "protocol": {
            "mode": "prefill",
            "categories": sorted({str(row["category"]) for row in cases}),
            "cases": [str(row["id"]) for row in cases],
            "prompt_tokens": {
                str(row["id"]): int(row["prompt_tokens"]) for row in cases
            },
            "repetitions": int(args.repetitions),
            "warmups_per_arm_per_case": int(args.warmups),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "interleaving": "round-robin, arm order rotated per repetition",
            "statistic": "median of per-repetition prefill wall",
            "timer": "perf_counter around runner.prefill + device_synchronize",
            "selectors": "flipped post-binder, one process, one model load",
            "arms": {arm.name: dict(arm.env) for arm in ARMS},
            "layer_scope": LAYER_SCOPE,
        },
        "bound_arm_env": bound_env,
        "effective_arm_env": effective_env,
        "arms": arms,
        "deltas_vs_exact": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")

    print()
    header = f"{'case':<16}" + "".join(f"{arm.name:>16}" for arm in ARMS)
    print(header)
    print("-" * len(header))
    for case_id in measurements["exact"]:
        line = f"{case_id:<16}"
        for arm in ARMS:
            line += f"{arms[arm.name]['cases'][case_id]['summary']['median']:>15.3f}s"
        print(line)
    print()
    for case_id, row in deltas.items():
        for arm in ARMS:
            if arm.name == "exact":
                continue
            entry = row[arm.name]
            print(
                f"{case_id:<16}{arm.name:>12}  {entry['median_seconds']:.3f}s  "
                f"saved {entry['saved_seconds']:+.3f}s  {entry['speedup']:.3f}x vs exact  "
                f"logits_match_exact={entry['same_logits_as_exact']}"
            )
    print()
    for arm in ARMS:
        rows = arms[arm.name]["census"]["rows"]
        by_symbol: dict[str, int] = {}
        for row in rows:
            by_symbol[row["symbol"]] = by_symbol.get(row["symbol"], 0) + row["launches"]
        print(f"{arm.name:<16}" + "  ".join(
            f"{symbol.split('_kernel')[0][-28:]}x{count}" for symbol, count in
            sorted(by_symbol.items(), key=lambda kv: -kv[1])[:3]
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
