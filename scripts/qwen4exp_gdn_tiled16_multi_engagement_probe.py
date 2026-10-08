#!/usr/bin/env python3
"""Engagement probe: tiled16_multi GDN prefill on the UD-Q4_K_XL lane.

The GD-4 terminal evidence shows the registered tiled16_multi kernel is
1.39-2.39x faster than the UD lane's production wave_norm route, gated off
solely by the arithmetic-recovery zero of GDN_COLWARPS_PREFILL. This probe
verifies through-the-runner engagement on the UD lane: the recovery flag and
the tile16 variant name are overridden after generator construction, a
counting registry hook on the multi variant records its calls, and one prefill
per canonical depth establishes the window (rows, heads, layers) in which the
kernel actually dispatches. Diagnostic only: no default changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from hipengine.kernels.registry import KernelKey, registered_keys, register, resolve
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    load_fixture,
)

VARIANT = "qwen4exp_gdn_tiled16_multi_prefill"
PARENT_VARIANT = "qwen4exp_sigmoid_wave_norm_prefill"
BACKEND = "hip_gfx1151"
CHUNK_SIZE = 4096


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    # The multi-column library may not be in this host's JIT cache yet (the
    # GD-4 timing session built it under its own identity); the probe builds it
    # on first use rather than failing closed, so the cache is warm for the
    # qualification stages that follow.
    os.environ.pop("HIPENGINE_REQUIRE_CACHED_BUILD", None)
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION,
    )
    fixture, fixture_sha = load_fixture(DEFAULT_FIXTURE)
    cases = [next(c for c in fixture["cases"] if c["id"] == cid)
             for cid in ("code-p512", "code-p1024", "code-p4096")]
    index = load_gguf_index(discover_gguf_files(args.model_root.resolve())[0])
    generator = profile.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root.resolve(),
        weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND,
        max_sequence_length=4352,
        prefill_chunk_size=CHUNK_SIZE,
    ))
    runner = generator.runner

    # Find the layer/quant key the multi variant is registered under; it is a
    # gdn prefill family key. Resolve through the alias mirror by probing the
    # registered keys for the variant name.
    multi_keys = [k for k in registered_keys() if k.variant == VARIANT]
    if not multi_keys:
        raise AssertionError(f"variant {VARIANT} is not registered")
    print("registered keys for the multi variant:", multi_keys)
    key = multi_keys[0]
    real_multi = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    state = {"multi_calls": 0, "shapes": []}

    def counting(*call_args, **kwargs):
        state["multi_calls"] += 1
        if len(call_args) >= 4:
            state["shapes"].append(tuple(int(v) for v in call_args[:4]))
        return real_multi(*call_args, **kwargs)

    report = {
        "schema": 1,
        "kind": "qwen4exp_gdn_tiled16_multi_engagement_probe",
        "status": "running",
        "performance_claim": False,
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "model_root": str(args.model_root),
        "fixture_sha256": fixture_sha,
        "manifest_sha256": profile.manifest_sha256,
        "variant": VARIANT,
        "registry_key": [key.backend, key.layer, key.quant, key.variant],
        "cases": [],
    }
    try:
        # Open the recovery gate and name the multi variant AFTER construction.
        os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] = "1"
        os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] = VARIANT
        register(key, counting, replace=True)
        for case in cases:
            state["multi_calls"] = 0
            state["shapes"] = []
            first = runner.prefill([int(t) for t in case["prompt_token_ids"]])
            report["cases"].append({
                "id": case["id"],
                "first_token": int(first.token_id),
                "multi_calls": state["multi_calls"],
                "first_shapes": state["shapes"][:4],
            })
            print(json.dumps(report["cases"][-1]), flush=True)
        os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] = "0"
        os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] = ""
        engaged = any(c["multi_calls"] > 0 for c in report["cases"])
        report["engaged"] = engaged
        report["status"] = "passed" if engaged else "no-engagement"
    finally:
        os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] = "0"
        os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] = ""
        register(key, real_multi, replace=True)
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
