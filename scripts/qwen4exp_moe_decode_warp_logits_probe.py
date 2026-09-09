#!/usr/bin/env python3
"""#22 R11 qualification: warp256 MoE decode GEMV pair logits drift.

Runs the canonical fixture prompts through the production decode route
(dp4a dual + logical256_t64 weighted-sum) and the default-off warp256
GEMV pair (HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP), teacher-forced: chunked
prefill then 4 forced decode steps, comparing per-step logits. Reports
absolute/relative statistics, KL divergence of the candidate softmax from
the incumbent softmax, and top-1 agreement per decode step.
Diagnostic only; no runtime default changes.
"""

from __future__ import annotations

import argparse
import ctypes
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

FLAG = "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP"
DECODE_STEPS = 4


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _forced_logits(generator, token_ids):
    """Teacher-forced prefill + DECODE_STEPS decode; returns stacked logits."""

    first = generator.runner.prefill(token_ids)
    rows = [np.ascontiguousarray(first.logits, dtype=np.float32)]
    token = int(first.token_id)
    for _ in range(DECODE_STEPS):
        nxt = generator.runner.step(token)
        generator.runner.runtime.device_synchronize()
        token = int(nxt.token_id)
        rows.append(np.ascontiguousarray(nxt.logits, dtype=np.float32))
    return np.stack(rows)


def _kl(ref: np.ndarray, cand: np.ndarray) -> float:
    r = ref.astype(np.float64)
    c = cand.astype(np.float64)
    pr = np.exp(r - r.max())
    pr /= pr.sum()
    pc = np.exp(c - c.max())
    pc /= pc.sum()
    return float(np.sum(pc * (np.log(pc) - np.log(pr))))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--case-id", action="append")
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
        "kind": "qwen4exp_moe_decode_warp_logits_probe",
        "source": _git_metadata(ROOT),
        "host": _host_metadata(),
        "command": sys.argv,
        "arithmetic_class": "down bit-exact T0 / gate-up T1 (1 bf16 ulp, reduction order)",
        "flag": FLAG,
        "decode_steps": DECODE_STEPS,
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
            for label, enabled in (
                ("incumbent_a", "0"), ("incumbent_b", "0"),
                ("candidate_a", "1"), ("candidate_b", "1"),
            ):
                os.environ[FLAG] = enabled
                rows[label] = _forced_logits(generator, case["prompt_token_ids"])

            incumbent_b_ok = bool(
                np.array_equal(rows["incumbent_a"], rows["incumbent_b"]))
            candidate_deterministic = bool(
                np.array_equal(rows["candidate_a"], rows["candidate_b"]))

            ref = rows["incumbent_a"]
            cand = rows["candidate_a"]
            diff = np.abs(cand - ref)
            rel = diff / np.maximum(np.abs(ref), 1e-30)
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
                "kl_vs_incumbent_max": float(max(
                    _kl(ref[i], cand[i]) for i in range(ref.shape[0]))),
                "kl_vs_incumbent_last": float(_kl(ref[-1], cand[-1])),
                "top1_agreement_all": float(np.mean(
                    ref.argmax(axis=1) == cand.argmax(axis=1))),
                "top1_agreement_final": float(
                    ref[-1].argmax() == cand[-1].argmax()),
                "exact_equal": bool(np.array_equal(ref, cand)),
            }
            report["cases"].append(entry)
            print(json.dumps(entry), flush=True)
    finally:
        os.environ[FLAG] = "0"
    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
