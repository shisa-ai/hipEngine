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
FLAG_DOWN = "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN"
DECODE_STEPS = 4
# Arm configurations: both flags are set explicitly in EVERY arm so an
# externally-set DOWN env cannot leak into the incumbent. The packet
# selects what the candidate arms enable; incumbent arms are always
# both-off (production incumbent).
PACKETS = {
    "dual": ("1", "0"),          # warp256 gate/up only
    "fast-down": ("0", "fast"),  # warp256 T1 vectorized down only
    "pair": ("1", "fast"),       # combined packet
}


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _forced_logits(generator, token_ids, steps, forced=None):
    """Teacher-forced prefill + `steps` decode; returns stacked logits.

    With ``forced`` (the incumbent arm's token chain) every arm consumes
    the SAME token sequence, so each row is a same-context measurement of
    pure route drift (the calibration definition) - a flip cannot cascade
    into incomparable contexts.
    """

    first = generator.runner.prefill(token_ids)
    rows = [np.ascontiguousarray(first.logits, dtype=np.float32)]
    token = int(first.token_id)
    chain = [token]
    for i in range(steps):
        if forced is not None:
            token = forced[i]
        nxt = generator.runner.step(token)
        generator.runner.runtime.device_synchronize()
        token = int(nxt.token_id)
        chain.append(token)
        rows.append(np.ascontiguousarray(nxt.logits, dtype=np.float32))
    return np.stack(rows), chain


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
    p.add_argument("--decode-steps", type=int, default=4)
    p.add_argument("--packet", choices=sorted(PACKETS), default="dual")
    p.add_argument("--emit-teacher", type=Path,
                   help="run the STRICT arm only and save its chains and "
                        "logits rows to this .npz (pass one of the two-pass "
                        "strict-reference qualification)")
    p.add_argument("--teacher", type=Path,
                   help="force the strict teacher chains from this .npz into "
                        "every arm and reference the strict logits rows "
                        "(production-versus-strict, the 6.1 calibration "
                        "definition) instead of the production incumbent")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    dual_v, down_v = PACKETS[a.packet]
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    if a.emit_teacher and a.teacher:
        p.error("--emit-teacher and --teacher are mutually exclusive")
    profile = (ExecutionProfile.STRICT if a.emit_teacher
               else ExecutionProfile.PRODUCTION)
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=profile)
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
        "flag_down": FLAG_DOWN,
        "packet": a.packet,
        "arm_flags": {"incumbent": {"dual": "0", "down": "0"},
                      "candidate": {"dual": dual_v, "down": down_v}},
        "comparison": ("strict teacher (production-versus-strict, 6.1)"
                       if a.teacher else "production incumbent (both flags off)"),
        "execution_profile": str(profile),
        "decode_steps": a.decode_steps,
        "runtime_default_changed": False,
        "fixture_sha256": digest,
        "manifest_sha256": resolved.manifest_sha256,
        "cases": [],
    }
    try:
        if a.emit_teacher:
            # pass one: strict teacher only - free chains, save rows
            os.environ[FLAG] = "0"
            os.environ[FLAG_DOWN] = "0"
            teacher = {}
            for case in fixture["cases"]:
                if a.case_id:
                    if case["id"] not in a.case_id:
                        continue
                trows, chain = _forced_logits(
                    generator, case["prompt_token_ids"], a.decode_steps)
                teacher[case["id"]] = {"chain": np.array(chain, dtype=np.int64),
                                        "logits": trows}
                print(f"teacher {case['id']}: {len(chain)} tokens", flush=True)
            np.savez_compressed(
                a.emit_teacher,
                **{f"{cid}__{field}": payload
                    for cid, entry in teacher.items()
                    for field, payload in entry.items()})
            a.output.write_text(json.dumps({
                "schema": 1, "kind": "strict_teacher_emit",
                "source": _git_metadata(ROOT), "host": _host_metadata(),
                "command": sys.argv, "decode_steps": a.decode_steps,
                "cases": sorted(teacher),
                "manifest_sha256": resolved.manifest_sha256,
                "fixture_sha256": digest,
            }, indent=1) + "\n")
            print(f"wrote {a.output} and {a.emit_teacher}")
            return
        teacher_data = None
        if a.teacher:
            teacher_data = np.load(a.teacher)
        for case in fixture["cases"]:
            if a.case_id:
                if case["id"] not in a.case_id:
                    continue
            rows = {}
            forced_chain = None
            ref_strict = None
            if teacher_data is not None:
                forced_chain = teacher_data[f"{case['id']}__chain"].tolist()
                ref_strict = teacher_data[f"{case['id']}__logits"]
            for label, enabled in (
                ("incumbent_a", "0"), ("incumbent_b", "0"),
                ("candidate_a", "1"), ("candidate_b", "1"),
            ):
                os.environ[FLAG] = dual_v if enabled == "1" else "0"
                os.environ[FLAG_DOWN] = down_v if enabled == "1" else "0"
                rows[label], chain = _forced_logits(
                    generator, case["prompt_token_ids"], a.decode_steps,
                    forced=forced_chain)
                if forced_chain is None:
                    forced_chain = chain

            incumbent_b_ok = bool(
                np.array_equal(rows["incumbent_a"], rows["incumbent_b"]))
            candidate_deterministic = bool(
                np.array_equal(rows["candidate_a"], rows["candidate_b"]))

            ref = (ref_strict if ref_strict is not None
                   else rows["incumbent_a"])
            cand = rows["candidate_a"]
            diff = np.abs(cand - ref)
            rel = diff / np.maximum(np.abs(ref), 1e-30)
            step_kls = [
                _kl(ref[i], cand[i]) for i in range(ref.shape[0])
            ]
            inc_step_kls = ([_kl(ref[i], rows["incumbent_a"][i])
                             for i in range(ref.shape[0])]
                            if ref_strict is not None else None)
            inc_step_top1 = ([bool(ref[i].argmax()
                                   == rows["incumbent_a"][i].argmax())
                              for i in range(ref.shape[0])]
                             if ref_strict is not None else None)
            step_top1 = [
                bool(ref[i].argmax() == cand[i].argmax())
                for i in range(ref.shape[0])
            ]
            step_margins = []
            for i in range(ref.shape[0]):
                order = np.argsort(-ref[i])
                step_margins.append(
                    float(ref[i][order[0]] - ref[i][order[1]]))
            entry = {
                "id": case["id"],
                "category": case.get("category", "unknown"),
                "step_kls": step_kls,
                "incumbent_step_kls": inc_step_kls,
                "incumbent_step_top1": inc_step_top1,
                "step_top1": step_top1,
                "step_margins": step_margins,
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
                "reference": ("strict" if ref_strict is not None
                             else "production_incumbent"),
                "incumbent_kl_vs_reference_max": (
                    float(max(_kl(ref[i], rows["incumbent_a"][i])
                             for i in range(ref.shape[0])))
                    if ref_strict is not None else None),
                "incumbent_top1_agreement_all": (
                    float(np.mean(ref.argmax(axis=1)
                                 == rows["incumbent_a"].argmax(axis=1)))
                    if ref_strict is not None else None),
            }
            report["cases"].append(entry)
            print(json.dumps(entry), flush=True)
        # envelope aggregation over every teacher-forced row
        rows = []
        flipped = []
        for c in report["cases"]:
            for i, kl in enumerate(c["step_kls"]):
                rows.append({
                    "id": c["id"], "category": c["category"], "step": i,
                    "kl": kl, "top1": c["step_top1"][i],
                })
                if not c["step_top1"][i]:
                    flipped.append({
                        "id": c["id"], "step": i,
                        "kl": kl, "margin": c["step_margins"][i],
                    })
        kls = np.array([r["kl"] for r in rows])
        report["envelope"] = {
            "rows": len(rows),
            "kl_mean": float(kls.mean()) if len(rows) else None,
            "kl_p95": float(np.percentile(kls, 95)) if len(rows) else None,
            "kl_p99": float(np.percentile(kls, 99)) if len(rows) else None,
            "kl_max": float(kls.max()) if len(rows) else None,
            "bars": {"mean": 1e-3, "p95": 5e-3, "p99": 2e-2, "max": 5e-2,
                     "top1_overall": 0.99, "top1_per_scope": 0.97},
            "top1_overall": float(
                sum(1 for r in rows if r["top1"]) / len(rows)) if rows else None,
            "top1_by_category": {
                cat: float(sum(1 for r in rows if r["category"] == cat and r["top1"])
                           / sum(1 for r in rows if r["category"] == cat))
                for cat in sorted({r["category"] for r in rows})
            },
            "flipped_rows": flipped,
        }
        if teacher_data is not None:
            # control envelope: the qualified production incumbent
            # versus the same strict teacher (must be within bars too)
            inc_kls = np.array([kl for c in report["cases"]
                                for kl in c["incumbent_step_kls"]])
            report["incumbent_envelope_vs_strict"] = {
                "kl_mean": float(inc_kls.mean()),
                "kl_p95": float(np.percentile(inc_kls, 95)),
                "kl_p99": float(np.percentile(inc_kls, 99)),
                "kl_max": float(inc_kls.max()),
                "top1_overall": float(np.mean([
                    t for c in report["cases"]
                    for t in c["incumbent_step_top1"]])),
            }
    finally:
        os.environ[FLAG] = "0"
        os.environ[FLAG_DOWN] = "0"
    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
