#!/usr/bin/env python3
"""R5 T1 qualification: GR iu8 route teacher-forced prefill logits drift.

Runs the canonical fixture prompts through the production incumbent route
and the default-off iu8-WMMA GR up route, twice each (determinism check),
and reports the candidate's incremental logits drift versus the incumbent:
absolute/relative statistics, KL divergence of the candidate softmax from
the incumbent softmax, and top-1 agreement. Diagnostic only; no runtime
default changes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.models import resolve_model
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)

from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, _git_metadata, _host_metadata, load_fixture,
)

FLAGS = ("HIPENGINE_QWEN4_EXP_GR_IU8", "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN")


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _prefill_logits(generator, token_ids):
    result = generator.runner.prefill(token_ids)
    return np.ascontiguousarray(result.logits, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--case-id", action="append")
    p.add_argument("--route", choices=("up", "down"), default="up")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(a.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4352,
        prefill_chunk_size=1024))

    report = {
        "schema": 1,
        "kind": "qwen4exp_gr_iu8_logits_probe",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "arithmetic_class": "T1_production_candidate",
        "flag": FLAGS[0] if a.route == "up" else FLAGS[1],
        "runtime_default_changed": False,
        "fixture_sha256": digest,
        "manifest_sha256": resolved.manifest_sha256,
        "cases": [],
    }
    try:
        for case in fixture["cases"]:
            if a.case_id:
                if case["id"] not in a.case_id:
                    continue
            elif not (case["prompt_tokens"] in (512, 1024)
                      or case["id"] == "code-p4096"):
                continue
            rows = {}
            flag = FLAGS[0] if a.route == "up" else FLAGS[1]
            for label, enabled in (
                ("incumbent_a", "0"), ("incumbent_b", "0"),
                ("candidate_a", "1"), ("candidate_b", "1"),
            ):
                os.environ[flag] = enabled
                rows[label] = _prefill_logits(
                    generator, case["prompt_token_ids"])

            incumbent_b_ok = bool(
                np.array_equal(rows["incumbent_a"], rows["incumbent_b"]))
            candidate_deterministic = bool(
                np.array_equal(rows["candidate_a"], rows["candidate_b"]))

            ref = rows["incumbent_a"]
            cand = rows["candidate_a"]
            diff = np.abs(cand - ref)
            rel = diff / np.maximum(np.abs(ref), 1e-30)
            # The publication logits are a single final-position row over the
            # full vocabulary (flat array).
            r = ref.astype(np.float64)
            c = cand.astype(np.float64)
            pr = np.exp(r - r.max()); pr /= pr.sum()
            pc = np.exp(c - c.max()); pc /= pc.sum()
            kl = float(np.sum(pc * (np.log(pc) - np.log(pr))))
            # Logit-gap margin around the incumbent argmax: drift below the
            # gap cannot flip sampling.
            order = np.argsort(-r)
            top1, top2 = r[order[0]], r[order[1]]
            entry = {
                "id": case["id"],
                "prompt_tokens": case["prompt_tokens"],
                "incumbent_deterministic": incumbent_b_ok,
                "candidate_deterministic": candidate_deterministic,
                "logits_shape": list(ref.shape),
                "max_abs": float(diff.max()),
                "p99_abs": float(np.percentile(diff, 99)),
                "median_abs": float(np.median(diff)),
                "rel_p50": float(np.percentile(rel, 50)),
                "rel_p99": float(np.percentile(rel, 99)),
                "rel_max": float(rel.max()),
                "kl_vs_incumbent": kl,
                "top1_agreement": float(
                    (ref.argmax() == cand.argmax())),
                "argmax_gap": float(top1 - top2),
                "argmax_drift": float(abs(c[order[0]] - top1)),
            }
            report["cases"].append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        for flag in FLAGS:
            os.environ[flag] = "0"
        generator.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
