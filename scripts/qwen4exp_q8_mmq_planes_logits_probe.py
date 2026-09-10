#!/usr/bin/env python3
"""#28 R12 d4x2 MMQ-plane candidate logits envelope probe.

The Q8 MMQ plane policy (HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES) resolves once
at generator construction, so arms cannot switch in-process. This probe
runs each arm in a SUBPROCESS (fixed env per process) and aggregates:

- arm ``incumbent``: planes=3 (retained d4x3), two in-process repeats,
  repeat A supplies the shared teacher chain;
- arm ``candidate``: planes=2 (d4x2 candidate), two repeats, both forced
  onto the incumbent chain (shared-chain teacher forcing: every row is a
  same-context measurement of pure route drift).

Run without --arm to orchestrate both subprocesses and emit the
996-row envelope report (canonical fixture, or --fixture for the
18-prompt admission suite).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.qwen4exp_moe_decode_warp_logits_probe import _kl, hip_available
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata,
)

PLANES_ENV = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES"
MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL")


def _forced_logits(generator, token_ids, steps, forced=None):
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


def run_arm(args) -> None:
    """One subprocess: fixed planes env, two repeats per case, save npz."""

    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    planes = "3" if args.arm == "incumbent" else "2"
    os.environ[PLANES_ENV] = planes

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
        QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    if args.fixture is None:
        fixture, digest = load_fixture(DEFAULT_FIXTURE)
    else:
        payload = json.loads(args.fixture.read_text())
        fixture = payload
        digest = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4352,
        prefill_chunk_size=1024))

    payload = {}
    for case in fixture["cases"]:
        if args.case_id and case["id"] not in args.case_id:
            continue
        rows_a, chain = _forced_logits(
            generator, case["prompt_token_ids"], args.decode_steps)
        rows_b, _ = _forced_logits(
            generator, case["prompt_token_ids"], args.decode_steps,
            forced=chain)
        deterministic = bool(np.array_equal(rows_a, rows_b))
        payload[case["id"]] = {
            "category": case.get("category", "unknown"),
            "chain": np.array(chain, dtype=np.int64),
            "logits_a": rows_a, "logits_b": rows_b,
        }
        print(json.dumps({"arm": args.arm, "id": case["id"],
                          "deterministic": deterministic}), flush=True)
    np.savez_compressed(args.arm_output,
                         **{f"{cid}__{field}": entry[field]
                            for cid, entry in payload.items() for field in
                            ("chain", "logits_a", "logits_b")})
    args.arm_output.with_suffix(".json").write_text(json.dumps({
        "arm": args.arm, "planes": planes, "cases": sorted(payload),
        "fixture_sha256": digest, "decode_steps": args.decode_steps,
        "source": _git_metadata(ROOT),
    }, indent=1))


def orchestrate(args) -> None:
    env = dict(os.environ)
    for arm in ("incumbent", "candidate"):
        cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
               "--model-root", str(args.model_root),
               "--compiler-version-file", str(args.compiler_version_file),
               "--decode-steps", str(args.decode_steps),
               "--arm-output", str(args.arm_output_pattern % arm)]
        if args.fixture is not None:
            cmd += ["--fixture", str(args.fixture)]
        for cid in args.case_id or ():
            cmd += ["--case-id", cid]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        Path(str(args.arm_output_pattern % arm) + ".log").write_text(
            result.stdout + result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"{arm} arm failed: {result.returncode}\n"
                               f"{result.stderr[-2000:]}")

    inc = np.load(args.arm_output_pattern % "incumbent")
    cand = np.load(args.arm_output_pattern % "candidate")
    inc_meta = json.loads((Path(str(args.arm_output_pattern % "incumbent"))
                           ).with_suffix(".json").read_text())
    cases = inc_meta["cases"]
    if args.fixture is None:
        fixture, _ = load_fixture(DEFAULT_FIXTURE)
    else:
        fixture = json.loads(args.fixture.read_text())
    categories = {c["id"]: c.get("category", "unknown")
                  for c in fixture["cases"]}
    report = {
        "schema": 1,
        "kind": "qwen4exp_q8_mmq_planes_logits_probe",
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "command": sys.argv,
        "route": "HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES 2 (d4x2) vs 3 (d4x3)",
        "comparison": "production incumbent (d4x3)",
        "decode_steps": args.decode_steps,
        "fixture_sha256": inc_meta.get("fixture_sha256"),
        "cases": [],
    }
    for cid in cases:
        case_category = categories[cid]
        chain = inc[f"{cid}__chain"]
        assert np.array_equal(chain, cand[f"{cid}__chain"]), cid
        ref = inc[f"{cid}__logits_a"]
        assert np.array_equal(ref, inc[f"{cid}__logits_b"]), ("inc det", cid)
        cand_a = cand[f"{cid}__logits_a"]
        cand_b = cand[f"{cid}__logits_b"]
        candidate_deterministic = bool(np.array_equal(cand_a, cand_b))
        step_kls = [_kl(ref[i], cand_a[i]) for i in range(ref.shape[0])]
        step_top1 = [bool(ref[i].argmax() == cand_a[i].argmax())
                     for i in range(ref.shape[0])]
        step_margins = []
        for i in range(ref.shape[0]):
            order = np.argsort(-ref[i])
            step_margins.append(float(ref[i][order[0]] - ref[i][order[1]]))
        top5 = []
        for i in range(ref.shape[0]):
            r5 = set(np.argsort(-ref[i])[:5].tolist())
            c5 = set(np.argsort(-cand_a[i])[:5].tolist())
            top5.append(len(r5 & c5))
        diff = np.abs(cand_a - ref)
        entry = {
            "id": cid,
            "category": case_category,
            "step_kls": step_kls, "step_top1": step_top1,
            "step_margins": step_margins, "step_top5_overlap": top5,
            "candidate_deterministic": candidate_deterministic,
            "max_abs": float(diff.max()),
            "exact_equal": bool(np.array_equal(ref, cand_a)),
        }
        report["cases"].append(entry)
        print(json.dumps({k: entry[k] for k in ("id", "candidate_deterministic",
                                                "exact_equal")}), flush=True)

    rows = []
    flipped = []
    for c in report["cases"]:
        for i, kl in enumerate(c["step_kls"]):
            rows.append({"id": c["id"], "category": c["category"],
                         "kl": kl, "top1": c["step_top1"][i]})
            if not c["step_top1"][i]:
                flipped.append({"id": c["id"], "step": i, "kl": kl,
                                "margin": c["step_margins"][i],
                                "top5": c["step_top5_overlap"][i]})
    kls = np.array([r["kl"] for r in rows])
    report["envelope"] = {
        "rows": len(rows),
        "kl_mean": float(kls.mean()),
        "kl_p95": float(np.percentile(kls, 95)),
        "kl_p99": float(np.percentile(kls, 99)),
        "kl_max": float(kls.max()),
        "bars": {"mean": 1e-3, "p95": 5e-3, "p99": 2e-2, "max": 5e-2,
                 "top1_overall": 0.99, "top1_per_scope": 0.97},
        "top1_overall": float(sum(1 for r in rows if r["top1"]) / len(rows)),
        "top1_by_category": {
            cat: float(sum(1 for r in rows if r["category"] == cat and r["top1"])
                       / sum(1 for r in rows if r["category"] == cat))
            for cat in sorted({r["category"] for r in rows})
        },
        "flipped_rows": flipped,
        "min_top5_overlap": int(min(min(c["step_top5_overlap"])
                                    for c in report["cases"])),
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    e = report["envelope"]
    b = e["bars"]
    ok = (e["kl_mean"] <= b["mean"] and e["kl_p95"] <= b["p95"]
          and e["kl_p99"] <= b["p99"] and e["kl_max"] <= b["max"]
          and e["top1_overall"] >= b["top1_overall"])
    print("envelope: mean %.2e p95 %.2e p99 %.2e max %.2e | top1 %.5f | "
          "flips %d/%d | min top5 %d/5 | %s" % (
              e["kl_mean"], e["kl_p95"], e["kl_p99"], e["kl_max"],
              e["top1_overall"], len(flipped), e["rows"],
              e["min_top5_overlap"], "ALL PASS" if ok else "FAIL"))
    print(f"wrote {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("incumbent", "candidate"), default=None)
    p.add_argument("--arm-output", type=Path, default=None,
                   help="npz path for this arm's chains+logits (subprocess mode)")
    p.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--case-id", action="append")
    p.add_argument("--fixture", type=Path, default=None)
    p.add_argument("--decode-steps", type=int, default=82)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.arm:
        if args.arm_output is None:
            p.error("--arm requires --arm-output")
        run_arm(args)
        return
    if not hip_available():
        p.error("HIP runtime unavailable")
    base = str(args.output)
    args.arm_output_pattern = base.replace(".json", ".%s.npz")
    orchestrate(args)


if __name__ == "__main__":
    main()
