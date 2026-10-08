#!/usr/bin/env python3
"""Same-residency decode A/B for the build-identity fast read.

The decode launch path resolves its kernel libraries once per launch through
``build_hip(load=True)``, whose memo lookup first derives the build-identity
environment tuple on every call. This A/B measures the decode wall with the
identity derivation in its two forms - the historical str-view read and the
raw-mapping fast read - inside one process and one model residency, with
balanced arm order per repetition.

Both arms resolve the SAME libraries (the identity values agree exactly; only
the derivation cost differs), so outputs must be identical. The candidate
changes no arithmetic, no kernel selection and no default; it is a host-side
cost change only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    DEFAULT_FIXTURE,
    _capture_kwargs,
    _git_metadata,
    _host_metadata,
    load_fixture,
)

CHUNK_SIZE = 4096
TRANSITIONS = 128
REPS_PER_ARM = 6


def _slow_str_identity(compiler: str) -> tuple:
    """The historical str-view derivation, embedded verbatim for the A/B."""

    import hipengine.core.build as build

    environment = os.environ
    prefix = build._compiler_env_prefix(compiler)
    return (
        environment.get(f"{prefix}_VERSION_TEXT"),
        environment.get("HIPENGINE_COMPILER_VERSION_TEXT"),
        environment.get(f"{prefix}_VERSION_FILE"),
        environment.get("HIPENGINE_COMPILER_VERSION_FILE"),
        environment.get(build._ENV_HIP_ARCH),
        environment.get(build._ENV_HIP_OFFLOAD_ARCH),
        environment.get(build._ENV_BUILD_CACHE_ROOT),
        environment.get(build._ENV_REQUIRE_CACHED_BUILD),
    )


def _normalized(identity: tuple) -> tuple:
    return tuple(
        value.decode() if isinstance(value, bytes) else value for value in identity
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--reps-per-arm", type=int, default=REPS_PER_ARM)
    args = parser.parse_args(argv)

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"

    import hipengine.core.build as build

    fast_identity = build._resolution_environment_identity
    assert _normalized(fast_identity("hipcc")) == _normalized(
        _slow_str_identity("hipcc")
    ), "identity forms disagree on the live environment"

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )
    fixture, fixture_sha = load_fixture(DEFAULT_FIXTURE)
    case_ids = args.case_id or ["code-p512", "code-p4096"]
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in case_ids]
    index = load_gguf_index(discover_gguf_files(args.model_root.resolve())[0])
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root.resolve(),
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND,
        max_sequence_length=max(int(c["prompt_tokens"]) for c in cases) + TRANSITIONS + 8,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runner = generator.runner
    prefill_kwargs, step_kwargs = _capture_kwargs("compact")

    report: dict = {
        "schema": 1,
        "kind": "qwen4exp_build_identity_decode_ab",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "arms": {"slow": "historical str-view identity", "fast": "raw-mapping identity"},
        "chunk_size": CHUNK_SIZE,
        "transitions": TRANSITIONS,
        "reps_per_arm": args.reps_per_arm,
        "cases": {},
    }

    def decode_ms(case, arm: str) -> tuple[float, list[int]]:
        build._resolution_environment_identity = (
            _slow_str_identity if arm == "slow" else fast_identity
        )
        token_ids = [int(t) for t in case["prompt_token_ids"]]
        first = runner.prefill(token_ids, **prefill_kwargs)
        outputs = [int(first.token_id)]
        current = int(first.token_id)
        runner.runtime.device_synchronize()
        started = time.perf_counter()
        for _ in range(TRANSITIONS):
            row = runner.step(current, **step_kwargs)
            current = int(row.token_id)
            outputs.append(current)
        runner.runtime.device_synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return elapsed_ms, outputs

    try:
        samples: dict[str, dict[str, list[float]]] = {}
        digests: dict[str, dict[str, list[str]]] = {}
        for case_index, case in enumerate(cases):
            case_id = str(case["id"])
            samples[case_id] = {"slow": [], "fast": []}
            digests[case_id] = {"slow": [], "fast": []}
            for arm in ("slow", "fast"):
                decode_ms(case, arm)  # warmup
            for rep in range(args.reps_per_arm):
                order = ("slow", "fast") if (rep + case_index) % 2 == 0 else ("fast", "slow")
                for arm in order:
                    elapsed_ms, outputs = decode_ms(case, arm)
                    samples[case_id][arm].append(elapsed_ms)
                    digest = hashlib_digest(outputs)
                    digests[case_id][arm].append(digest)
            entry = {}
            for arm in ("slow", "fast"):
                vals = samples[case_id][arm]
                entry[arm] = {
                    "decode_ms_median": statistics.median(vals),
                    "decode_ms_samples": vals,
                    "decode_tok_s_median": 1000.0 * TRANSITIONS / statistics.median(vals),
                    "within_arm_deterministic": len(set(digests[case_id][arm])) == 1,
                }
            entry["cross_arm_output_exact"] = (
                set(digests[case_id]["slow"]) == set(digests[case_id]["fast"])
            )
            entry["median_delta_ms"] = entry["slow"]["decode_ms_median"] - entry["fast"]["decode_ms_median"]
            report["cases"][case_id] = entry
            print(json.dumps({
                "case": case_id,
                "slow_decode_ms": round(entry["slow"]["decode_ms_median"], 3),
                "fast_decode_ms": round(entry["fast"]["decode_ms_median"], 3),
                "delta_ms": round(entry["median_delta_ms"], 3),
                "slow_tok_s": round(entry["slow"]["decode_tok_s_median"], 3),
                "fast_tok_s": round(entry["fast"]["decode_tok_s_median"], 3),
                "cross_arm_output_exact": entry["cross_arm_output_exact"],
            }), flush=True)
        report["status"] = "completed"
    finally:
        build._resolution_environment_identity = fast_identity
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def hashlib_digest(outputs: list[int]) -> str:
    import hashlib

    return hashlib.sha256(",".join(map(str, outputs)).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
