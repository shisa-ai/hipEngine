#!/usr/bin/env python3
"""State/engagement gate for the #22 R11 MoE decode warp pair.

The pair (warp256 dp4a dual + warp256 T1 Q5_1 down) is a non-exact decode
route, so the exactness state gate does not apply byte-for-byte. This gate
verifies the applicable contract properties instead:

- engagement: candidate arms call the warp256 dual and warp256 fast down
  exactly once per MoE layer per decode step and never during prefill;
  incumbent arms never call them; the exact-tree warp down variant is
  never called in pair mode;
- isolation: the off->on->off->on arm sequence returns identical bytes
  for repeated same-flag arms (no cross-arm leakage);
- determinism: repeated candidate arms are bit-identical on full logits
  and the full decode-state digest;
- state drift: decode-state buffers stay finite and their max-abs drift
  versus the incumbent arm is recorded (bounded evidence, not asserted
  equal - logits drift is qualified separately by the envelope probe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.execution_profiles import (
    ExecutionProfile, resolve_runtime_profile,
)
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.generation.qwen4_exp_profiles import (
    register_qwen4_exp_gfx1151_profiles, QWEN4_EXP_MODEL,
    QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata,
)
from scripts.qwen4exp_layer2_profile_gate import _state_summary, _bf16_to_f32

import hipengine.runtime.qwen4_exp_runner as runner_module

FLAG = "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP"
FLAG_DOWN = "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN"

DUAL_WRAPPER = "gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out"
DOWN_FAST_WRAPPER = "qwen4_exp_q5_1_selected_weighted_sum_warp256_fast_bf16_bf16_out"
DOWN_EXACT_WRAPPER = "qwen4_exp_q5_1_selected_weighted_sum_warp256_bf16_bf16_out"


def _drift_stats(incumbent: dict, candidate: dict) -> dict:
    """Max-abs drift per shared decode-state buffer (bf16/f32 views)."""
    out = {}
    for name in sorted(incumbent):
        a = incumbent[name]
        b = candidate[name]
        if a.dtype == np.uint8 and a.nbytes % 2 == 0:
            av = _bf16_to_f32(a.view(np.uint16))
            bv = _bf16_to_f32(b.view(np.uint16))
        else:
            av = a.view(np.float32)
            bv = b.view(np.float32)
        out[str(name)] = float(np.max(np.abs(av - bv))) if av.size else 0.0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-id", action="append")
    p.add_argument("--all-cases", action="store_true")
    p.add_argument("--decode-steps", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.decode_steps <= 128:
        p.error("--decode-steps must be in 1..128")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ[FLAG] = "0"
    os.environ[FLAG_DOWN] = "0"

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4352,
        prefill_chunk_size=1024))

    counters = {"dual": 0, "down_fast": 0, "down_exact": 0}
    originals = {}

    for key, wrapper_name in (
        ("dual", DUAL_WRAPPER),
        ("down_fast", DOWN_FAST_WRAPPER),
        ("down_exact", DOWN_EXACT_WRAPPER),
    ):
        original = getattr(runner_module, wrapper_name)
        originals[key] = original
        setattr(runner_module, wrapper_name,
                _make_counter(counters, key, original))

    report = {
        "status": "running",
        "kind": "qwen4exp_moe_decode_warp_state_gate",
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "command": sys.argv,
        "fixture_sha256": digest,
        "manifest_sha256": resolved.manifest_sha256,
        "route_package": "moe-decode-warp-pair",
        "decode_steps": args.decode_steps,
        "scope": "engagement + isolation + same-arm determinism + state drift recording",
        "cases": [],
    }
    try:
        moe_layers = None
        for case in fixture["cases"]:
            if args.case_id:
                if case["id"] not in args.case_id:
                    continue
            elif not args.all_cases and case["prompt_tokens"] != 512 and case["id"] != "code-p4096":
                continue
            captures = []
            for enabled in ("0", "1", "0", "1"):
                os.environ[FLAG] = enabled
                os.environ[FLAG_DOWN] = "fast" if enabled == "1" else "0"
                for k in counters:
                    counters[k] = 0
                first = generator.runner.prefill(case["prompt_token_ids"])
                prefill_counts = dict(counters)
                logits = first.logits.copy()
                token = int(first.token_id)
                step_logits = []
                step_seconds = []
                for _ in range(args.decode_steps):
                    start = time.perf_counter()
                    next_row = generator.runner.step(token)
                    generator.runner.runtime.device_synchronize()
                    step_seconds.append(time.perf_counter() - start)
                    token = int(next_row.token_id)
                    step_logits.append(next_row.logits.copy())
                next_logits = np.stack(step_logits)
                state = _state_summary(generator.runner)
                total_counts = dict(counters)
                decode_counts = {
                    k: total_counts[k] - prefill_counts[k]
                    for k in total_counts}
                if enabled == "1":
                    assert prefill_counts == {"dual": 0, "down_fast": 0,
                                               "down_exact": 0}, (
                        "warp pair ran during prefill", prefill_counts)
                    assert decode_counts["down_exact"] == 0, (
                        "exact-tree down engaged in fast pair mode")
                    assert decode_counts["dual"] > 0, "dual not engaged"
                    assert decode_counts["dual"] == decode_counts["down_fast"], (
                        "dual/down call mismatch", decode_counts)
                    if moe_layers is None:
                        assert decode_counts["dual"] % args.decode_steps == 0
                        moe_layers = decode_counts["dual"] // args.decode_steps
                        assert moe_layers > 0
                    assert decode_counts["dual"] == (
                        moe_layers * args.decode_steps), (
                        "unexpected call count", decode_counts, moe_layers)
                else:
                    assert total_counts == {"dual": 0, "down_fast": 0,
                                            "down_exact": 0}, (
                        "incumbent arm engaged the warp pair", total_counts)
                assert state["finite"], ("non-finite state", case["id"])
                captures.append({
                    "enabled": enabled,
                    "prefill_logits_sha256": _sha(logits),
                    "step_logits_sha256": _sha(next_logits),
                    "state_sha256": state["state_sha256"],
                    "layout_sha256": state["layout_sha256"],
                    "decode_counts": decode_counts,
                    "step_seconds": step_seconds,
                    "state_buffers": {
                        str(name): np.ascontiguousarray(raw)
                        for name, raw in sorted(
                            generator.runner.snapshot().decode_state.buffers.items())
                    },
                })
            assert captures[0]["prefill_logits_sha256"] == captures[2]["prefill_logits_sha256"]
            assert captures[1]["prefill_logits_sha256"] == captures[3]["prefill_logits_sha256"]
            assert captures[0]["step_logits_sha256"] == captures[2]["step_logits_sha256"]
            assert captures[1]["step_logits_sha256"] == captures[3]["step_logits_sha256"]
            assert captures[0]["state_sha256"] == captures[2]["state_sha256"]
            assert captures[1]["state_sha256"] == captures[3]["state_sha256"]
            assert captures[1]["step_logits_sha256"] != captures[0]["step_logits_sha256"], (
                "candidate arm byte-identical to incumbent: engagement unproven")
            drift = _drift_stats(captures[0]["state_buffers"],
                                  captures[1]["state_buffers"])
            entry = {
                "id": case["id"],
                "moe_layers": moe_layers,
                "incumbent_deterministic": True,
                "candidate_deterministic": True,
                "decode_counts_candidate": captures[1]["decode_counts"],
                "decode_counts_incumbent": captures[0]["decode_counts"],
                "state_drift_max_abs": drift,
                "state_drift_overall_max": max(drift.values()) if drift else 0.0,
                "median_step_ms": float(np.median(captures[1]["step_seconds"]) * 1e3),
            }
            report["cases"].append(entry)
            print(json.dumps({k: v for k, v in entry.items()
                              if k != "state_drift_max_abs"}), flush=True)
        report["moe_layers"] = moe_layers
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for key, wrapper_name in (
            ("dual", DUAL_WRAPPER),
            ("down_fast", DOWN_FAST_WRAPPER),
            ("down_exact", DOWN_EXACT_WRAPPER),
        ):
            setattr(runner_module, wrapper_name, originals[key])
        os.environ[FLAG] = "0"
        os.environ[FLAG_DOWN] = "0"
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")


def _sha(array: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def _wrapper_name(key: str) -> str:
    return {"dual": DUAL_WRAPPER, "down_fast": DOWN_FAST_WRAPPER,
            "down_exact": DOWN_EXACT_WRAPPER}[key]


def _make_counter(counters: dict, key: str, original):
    def counted(*a, **kw):
        counters[key] += 1
        return original(*a, **kw)
    return counted


if __name__ == "__main__":
    main()
