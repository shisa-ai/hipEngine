#!/usr/bin/env python3
"""Run the canonical p512 AR fixture through hipEngine with a candidate route.

This mirrors ``scripts/qwen4exp_canonical_ar_bench.py``'s ``hipengine`` mode
but lets the caller force a candidate routing override (e.g.
``HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL=1``) that the production profile
binder would otherwise reset to 0. It is a diagnostic candidate runner only —
never a retained claim — used to measure whether a layer-2 routing change moves
the p512 ledger before deciding whether the full production gate is worth it.

Usage:
  python3 scripts/qwen4exp_candidate_ar.py hipengine \
      --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
      --fixture /tmp/qwen4exp_p512_fixture.json \
      --output /tmp/cand.json --warmups 1 --repetitions 1 \
      --override HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL=1 \
      --compiler-version-file /tmp/hipengine-hipcc-version.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    _host_metadata,
    _hipengine_case_sample,
    _measurement_order,
    _write_json,
    load_fixture,
    summarize_samples,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--compiler-version-file", type=Path)
    args = parser.parse_args()

    for raw in args.override:
        key, sep, value = raw.partition("=")
        if not sep:
            raise SystemExit(f"--override must be KEY=VALUE, got {raw!r}")
        os.environ[key] = value

    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

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

    fixture, _fixture_sha = load_fixture(args.fixture)
    cases = fixture["cases"]
    transitions = int(fixture["decode_transitions"])
    model_root = args.model_root.resolve()
    max_sequence_length = max(int(row["prompt_tokens"]) for row in cases) + transitions + 8

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )

    # The production binder resets candidate routing envs to their certified
    # values during construct_generator, so the override is reapplied on the
    # fully-constructed generator (the runner reads these envs at MoE-call time).
    overrides = dict(item.partition("=")[::2] for item in args.override)

    def construct(base_factory, **kwargs):
        generator = resolved.construct_generator(base_factory, **kwargs)
        for key, value in overrides.items():
            os.environ[key] = value
        return generator

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    artifact: dict = {
        "schema": 1,
        "kind": "qwen4exp_canonical_ar_engine_run",
        "engine": "hipengine_candidate",
        "surface": "synchronized_direct_runner",
        "status": "running",
        "host": _host_metadata(),
        "fixture": str(args.fixture.resolve()),
        "model_root": str(model_root),
        "source": _git_metadata_import(),
        "profile": {
            "requested": "production",
            "manifest_sha256": resolved.manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "overrides": dict(overrides),
        "protocol": {
            "warmups_per_case": int(args.warmups),
            "measured_repetitions": int(args.repetitions),
            "decode_transitions": transitions,
            "prefill_chunk_size": int(args.prefill_chunk_size),
        },
        "warmups": [],
        "samples": [],
    }
    _write_json(args.output, artifact)
    generator = construct(factory)
    try:
        for warmup in range(args.warmups):
            for case in _measurement_order(cases, warmup):
                row = _hipengine_case_sample(
                    generator.runner, case=case, repetition=warmup, transitions=transitions
                )
                artifact["warmups"].append(row["case_id"])
                print(
                    f"[warmup] hipengine_candidate {row['case_id']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        for repetition in range(args.repetitions):
            for case in _measurement_order(cases, repetition):
                row = _hipengine_case_sample(
                    generator.runner, case=case, repetition=repetition, transitions=transitions
                )
                artifact["samples"].append(row)
                artifact["summary"] = summarize_samples(artifact["samples"])
                _write_json(args.output, artifact)
                print(
                    f"[measure {repetition}] hipengine_candidate {row['case_id']} "
                    f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        artifact["status"] = "completed"
        artifact["memory_before_close"] = memory_stats()
        _write_json(args.output, artifact)
    finally:
        generator.close()
    print(
        json.dumps(
            {"kind": artifact["kind"], "status": artifact["status"], "output": str(args.output)},
            indent=2,
        )
    )
    return 0


def _git_metadata_import():
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata

    return _git_metadata(ROOT)


if __name__ == "__main__":
    import json

    raise SystemExit(main())
