#!/usr/bin/env python3
"""Localize the public long-C8 prefill failure; not a publication gate."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import execution_profile_gguf_fp16_state_batch_gate as gate
from scripts.execution_profile_gguf_batch_route_gate import BatchRouteCapture, build_batch_route_quality
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen38_q4_verifier_numerics import _environment
from hipengine.runtime.gguf_linear import PREFILL_F16_STAGING_ENV, Q6_INTEGER_MMQ_PREFILL_ENV

MODES = {
    "no_staging": {PREFILL_F16_STAGING_ENV: "0"},
    "no_mmq": {Q6_INTEGER_MMQ_PREFILL_ENV: "0"},
    "neither": {PREFILL_F16_STAGING_ENV: "0", Q6_INTEGER_MMQ_PREFILL_ENV: "0"},
}


def main():
    from hipengine.loading.gguf import scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--modes", choices=tuple(MODES), nargs="+", default=list(MODES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.backend = "hip_gfx1151"
    args.quant_label = "gguf_q4_k_m"
    rows = _load_suites(DEFAULT_PROMPTS)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
    tokens = {row["id"]: build_chat_prompt(tokenizer, str(row["prompt"])) for row in rows}
    scenario = gate.build_static_scenarios(
        rows, tokens, widths=(8,), decode_steps=128,
        long_prompt_tokens=512, long_decode_steps=128)[-1]
    stack, sessions, _, _ = gate._make_public_profile_sessions(
        args, candidate=False, max_sequence_length=644)
    with stack:
        strict, _ = gate._run_static_once(sessions, scenario, reference=None)
    results = {}
    for mode in args.modes:
        with ExitStack() as contexts:
            for name, value in MODES[mode].items():
                contexts.enter_context(_environment(name, value))
            stack, sessions, _, _ = gate._make_public_profile_sessions(
                args, candidate=True, max_sequence_length=644)
            with stack:
                actual_flags = {
                    "staging": sessions[0].use_prefill_f16_staging,
                    "mmq": sessions[0].use_q6_integer_mmq,
                }
                print(f"{mode}: {actual_flags}", flush=True)
                runs = []
                for repeat in range(3):
                    trajectory, _ = gate._run_static_once(sessions, scenario, reference=strict)
                    runs.append(trajectory)
                    print(f"{mode}: repeat {repeat + 1}", flush=True)
            captures = tuple(BatchRouteCapture(
                scenario_id=scenario.scenario_id, request_id=row["id"], category=row["category"],
                strict=strict[index], candidate_runs=tuple(run[index] for run in runs),
                shapes=("c8_prefill_p512",) + ("c8",) * 128,
                transitions=("long_prefill_to_c8",) + ("steady",) * 128,
                teacher_steps=tuple(range(129)),
            ) for index, row in enumerate(scenario.rows))
            quality = build_batch_route_quality(captures)
            results[mode] = {"flags": actual_flags, "quality": quality}
            print(json.dumps(quality["quality"]["summary"]), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "kind": "long_c8_prefill_composition_diagnostic",
            "performance_claim": False, "full_profile_qualification": False,
            "command": [sys.executable, *sys.argv], "results": results,
            "runtime_profiles": args.runtime_profiles,
        }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
