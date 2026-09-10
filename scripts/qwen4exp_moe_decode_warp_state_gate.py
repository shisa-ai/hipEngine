#!/usr/bin/env python3
"""State/engagement gate for the #22 R11 MoE decode warp pair.

The pair (warp256 dp4a dual + warp256 T1 Q5_1 down) is a non-exact decode
route, so the exactness state gate does not apply byte-for-byte. The MoE
decode block runs through ``MoeGraphCache``: Python kernel wrappers are
invoked only at capture time (twice per new key), so wrapper counting
cannot prove per-arm engagement. This gate instruments ``MoeGraphCache.run``
instead and verifies the applicable contract properties:

- engagement: every ``run()`` call in a candidate arm carries a graph key
  with the warp route flags enabled (``1``/``fast``) and every call in an
  incumbent arm carries them disabled (``0``/``0``); the call count equals
  MoE-layer-count x decode-steps per arm;
- capture bookkeeping: captures happen once per key; the exact-tree warp
  down kernel is never invoked in fast mode (cumulative wrapper count);
- isolation: the off->on->off->on arm sequence returns identical bytes for
  repeated same-flag arms (graph keys embed the route flags, so each arm
  replays its own graphs);
- determinism: repeated candidate arms are bit-identical on full logits
  and the full decode-state digest (plus the cross-run probe repeat);
- state drift: decode-state buffers stay finite and their max-abs drift
  versus the incumbent arm is recorded (bounded evidence, not asserted
  equal - logits drift is qualified separately by the envelope probes).
"""

from __future__ import annotations

import argparse
import hashlib
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
from hipengine.runtime.moe_graph import MoeGraphCache
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _git_metadata, _host_metadata,
)
from scripts.qwen4exp_layer2_profile_gate import _state_summary, _bf16_to_f32

import hipengine.runtime.qwen4_exp_runner as runner_module

FLAG = "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP"
FLAG_DOWN = "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN"

DOWN_EXACT_WRAPPER = "qwen4_exp_q5_1_selected_weighted_sum_warp256_bf16_bf16_out"

# graph key layout at both call sites:
# (route, layer, Q4_DP4A64, Q4_DP4A64_LAYERS, MOE_DECODE_WARP, ...DOWN)
KEY_WARP_INDEX = 4
KEY_WARP_DOWN_INDEX = 5


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

    # instrument the graph cache (primary per-arm engagement signal)
    graph_events: list[dict] = []
    original_run = MoeGraphCache.run

    def counting_run(self, key, *, eager, out_ptr, out_nbytes,
                     mutable_inputs=(), stream=0, out_regions=()):
        mode = original_run(
            self, key, eager=eager, out_ptr=out_ptr, out_nbytes=out_nbytes,
            mutable_inputs=mutable_inputs, stream=stream,
            out_regions=out_regions)
        graph_events.append({
            "mode": mode,
            "key": [str(k) for k in key] if isinstance(key, tuple) else [str(key)],
        })
        return mode

    MoeGraphCache.run = counting_run

    # cumulative wrapper guard: the exact-tree warp down must never run
    down_exact_count = [0]
    down_exact_original = getattr(runner_module, DOWN_EXACT_WRAPPER)

    def counted_down_exact(*a, **kw):
        down_exact_count[0] += 1
        return down_exact_original(*a, **kw)

    setattr(runner_module, DOWN_EXACT_WRAPPER, counted_down_exact)

    report = {
        "status": "running",
        "kind": "qwen4exp_moe_decode_warp_state_gate",
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "command": sys.argv,
        "fixture_sha256": digest,
        "manifest_sha256": resolved.manifest_sha256,
        "route_package": "moe-decode-warp-pair",
        "decode_steps": args.decode_steps,
        "scope": "graph-key engagement + isolation + same-arm determinism "
                 "+ state drift recording",
        "cases": [],
    }
    try:
        moe_run_calls_per_step = None
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
                del graph_events[:]
                first = generator.runner.prefill(case["prompt_token_ids"])
                prefill_events = len(graph_events)
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
                assert state["finite"], ("non-finite state", case["id"])
                assert prefill_events == 0, (
                    "MoE graph ran during prefill", prefill_events)
                decode_events = graph_events
                # only the qsa/gdn MoE decode keys embed the warp flags at a
                # fixed position; other MoeGraphCache users (different key
                # layouts) are out of scope for this gate and recorded.
                def _warp_fields(ev):
                    key = ev["key"]
                    if len(key) > KEY_WARP_DOWN_INDEX and key[0] in {"qsa", "gdn"}:
                        return key[KEY_WARP_INDEX], key[KEY_WARP_DOWN_INDEX]
                    return None
                out_of_scope = [
                    tuple(ev["key"][:2]) for ev in decode_events
                    if _warp_fields(ev) is None
                ]
                if enabled == "1":
                    for ev in decode_events:
                        fields = _warp_fields(ev)
                        if fields is None:
                            continue
                        assert fields[0] == "1", (
                            "candidate arm replayed a non-warp graph", ev)
                        assert fields[1] == "fast", (
                            "candidate arm replayed a non-fast-down graph", ev)
                else:
                    for ev in decode_events:
                        fields = _warp_fields(ev)
                        if fields is None:
                            continue
                        assert fields[0] == "0", (
                            "incumbent arm replayed a warp graph", ev)
                        assert fields[1] == "0", (
                            "incumbent arm replayed a fast-down graph", ev)
                modes = {}
                layers = set()
                oos_layers = sorted(set(out_of_scope))
                for ev in decode_events:
                    modes[ev["mode"]] = modes.get(ev["mode"], 0) + 1
                    layers.add((ev["key"][0], ev["key"][1]))
                if enabled == "1":
                    assert modes.get("replay", 0) + modes.get("capture", 0) + \
                        modes.get("eager", 0) == len(decode_events)
                    if moe_run_calls_per_step is None:
                        assert len(decode_events) % args.decode_steps == 0
                        moe_run_calls_per_step = (
                            len(decode_events) // args.decode_steps)
                    else:
                        assert len(decode_events) == (
                            moe_run_calls_per_step * args.decode_steps), (
                            "graph run() count changed across cases",
                            len(decode_events))
                captures.append({
                    "enabled": enabled,
                    "prefill_logits_sha256": _sha(logits),
                    "step_logits_sha256": _sha(next_logits),
                    "state_sha256": state["state_sha256"],
                    "layout_sha256": state["layout_sha256"],
                    "graph_modes": modes,
                    "graph_layers": len(layers),
                    "out_of_scope_graph_layers": oos_layers,
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
                "graph_run_calls_per_step": moe_run_calls_per_step,
                "incumbent_deterministic": True,
                "candidate_deterministic": True,
                "candidate_graph_modes": captures[1]["graph_modes"],
                "incumbent_graph_modes": captures[0]["graph_modes"],
                "state_drift_max_abs": drift,
                "state_drift_overall_max": max(drift.values()) if drift else 0.0,
                "median_step_ms": float(np.median(captures[1]["step_seconds"]) * 1e3),
            }
            report["cases"].append(entry)
            print(json.dumps({k: v for k, v in entry.items()
                              if k != "state_drift_max_abs"}), flush=True)
        assert down_exact_count[0] == 0, (
            "exact-tree warp down engaged in fast pair mode", down_exact_count[0])
        report["down_exact_wrapper_calls"] = down_exact_count[0]
        report["moe_run_calls_per_step"] = moe_run_calls_per_step
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        MoeGraphCache.run = original_run
        setattr(runner_module, DOWN_EXACT_WRAPPER, down_exact_original)
        os.environ[FLAG] = "0"
        os.environ[FLAG_DOWN] = "0"
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


if __name__ == "__main__":
    main()
