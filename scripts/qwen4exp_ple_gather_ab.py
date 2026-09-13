"""Counterbalanced complete-model PLE or GDN owner A/B.

Both methods run in one production residency. Gather and final-logit/state
equality are checked; the incumbent numerical certificate remains separate.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_ple_gather_screen import gather_sorted_unique
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _hipengine_case_sample, _host_metadata, _git_metadata
from scripts.qwen4exp_halo_box_campaign_ab import arm_sequence, summarize_campaign_ab
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_summary
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity


def select_gather_arm(table, original, *, mode, method, cache_mode):
    if mode not in ("before", "after") or cache_mode not in ("warm", "cold"):
        raise ValueError("invalid gather arm or cache mode")
    if cache_mode == "cold":
        table.advise_cache("cold")
    if method == "mmap_random":
        if not table.configure_mapping_access("random" if mode == "after" else "normal"):
            raise RuntimeError("mapping advice did not engage")
        table.gather_rows = original
    else:
        table.gather_rows = original if mode == "before" else (
            lambda ids: gather_sorted_unique(table, ids, method=method)
        )


def pair_sequence(case_index, pairs):
    if pairs < 1:
        raise ValueError("positive pair count required")
    result = []
    for pair in range(pairs):
        result.extend(("before", "after") if (case_index + pair) % 2 == 0
                      else ("after", "before"))
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("copy_elision", "sampled_dedup_elision", "mmap_random", "gdn_dpp"), required=True)
    parser.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    check_host()
    args.prefill_chunk_size = 1024
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixture, fixture_hash = load_fixture(args.fixture)
        case_indices = {c["id"]: i for i, c in enumerate(fixture["cases"])}
        cases = fixture["cases"]
        if args.case_id:
            if not set(args.case_id) <= set(case_indices):
                raise ValueError("unknown canonical case id")
            cases = [c for c in cases if c["id"] in args.case_id]
        pairs = 1 if args.screen_only else args.pairs
        if pairs < 1 or (not args.screen_only and pairs < 3):
            raise ValueError("use at least three pairs or explicit screen-only")
        transitions = fixture["decode_transitions"]
        args.max_sequence_length = max(c["prompt_tokens"] for c in cases) + transitions + 8
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        reset_memory_stats()
        identity = model_identity(args.model_root)
        report = {
            "kind": "qwen4exp_ple_complete_model_ab", "status": "running",
            "performance_claim": False, "host": _host_metadata(), "source": _git_metadata(ROOT),
            "command": sys.argv, "fixture_sha256": fixture_hash, "model_identity": identity,
            "method": args.method, "protocol": {
                "chunk": 1024, "kv": "BF16", "decode_transitions": transitions,
                "warmups_per_arm_case": 1, "repetitions_per_arm": pairs,
                "screen_only": args.screen_only, "counterbalanced": True,
                "complete_canonical_suite": not bool(args.case_id),
                "case_ids": [c["id"] for c in cases],
                "cache": args.cache_mode,
                "cache_scope": "file-scoped PLE only; no persistent row cache",
            },
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_script_sha256": hashlib.sha256((ROOT / "scripts/qwen4exp_ple_gather_screen.py").read_bytes()).hexdigest(),
            "samples": [],
        }
        generator, profile, _ = _make_generator(args, "production")
        import hipengine.runtime.qwen4_exp_runner as runner_module
        original_gdn = runner_module.qwen4_exp_gdn_prefill_tiled16_f32
        dpp_calls = 0
        dpp_library = None
        if args.method == "gdn_dpp":
            from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import build_qwen4_exp_gdn_dpp
            dpp_library = build_qwen4_exp_gdn_dpp()
            report["kind"] = "qwen4exp_gdn_complete_model_ab"
            report["candidate_library_sha256"] = hashlib.sha256(Path(dpp_library._name).read_bytes()).hexdigest()
        table = generator._resident.ple_table
        original_gather = table.gather_rows
        original_step = generator.runner.step
        last = {}

        def step(*a, **kw):
            result = original_step(*a, **kw)
            last["result"] = result
            return result

        generator.runner.step = step

        def dpp(*a, **kw):
            nonlocal dpp_calls
            dpp_calls += 1
            return original_gdn(*a, **{**kw, "library": dpp_library})

        def sample(mode, case, rep):
            if table.telemetry() is not None or table._random_access_requested_mode != "off":
                raise ValueError("unqualified telemetry/advice configuration")
            if args.method == "gdn_dpp":
                runner_module.qwen4_exp_gdn_prefill_tiled16_f32 = original_gdn if mode == "before" else dpp
            else:
                select_gather_arm(table, original_gather, mode=mode,
                                  method=args.method, cache_mode=args.cache_mode)
            previous_dpp_calls = dpp_calls
            row = _hipengine_case_sample(generator.runner, case=case, repetition=rep, transitions=transitions)
            if args.method == "gdn_dpp":
                row["dpp_calls"] = dpp_calls - previous_dpp_calls
                if (row["dpp_calls"] > 0) != (mode == "after"):
                    raise ValueError("DPP candidate did not engage")
            logits = np.asarray(last["result"].logits)
            if not logits.size or not np.isfinite(logits).all():
                raise ValueError("missing/nonfinite final logits")
            state = _state_summary(generator.runner)
            if not state["finite"]:
                raise ValueError("nonfinite final state")
            row.update(mode=mode, final_logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                       final_state_sha256=state["state_sha256"])
            return row

        try:
            for case in cases:
                index = case_indices[case["id"]]
                warm = [sample(mode, case, -1) for mode in arm_sequence(index)[:2]]
                for field in ("output_token_ids_sha256", "final_logits_sha256", "final_state_sha256"):
                    if warm[0][field] != warm[1][field]:
                        raise ValueError(f"warmup differs: {case['id']} {field}")
                counts = {"before": 0, "after": 0}
                order = pair_sequence(index, pairs)
                for slot, mode in enumerate(order):
                    row = sample(mode, case, counts[mode])
                    counts[mode] += 1
                    row["sequence_slot"] = slot
                    for field in ("output_token_ids_sha256", "final_logits_sha256", "final_state_sha256"):
                        if row[field] != warm[0][field]:
                            raise ValueError(f"measured state/output differs: {case['id']} {field}")
                    report["samples"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(case["id"], mode, slot, row["prefill_tok_s"], row["decode_tok_s"], flush=True)
            report["summary"] = summarize_campaign_ab(
                report["samples"], repetitions_per_mode=pairs,
            )
            report["exact_final_logits_and_state"] = True
            report["status"] = "completed"
        finally:
            runner_module.qwen4_exp_gdn_prefill_tiled16_f32 = original_gdn
            table.gather_rows = original_gather
            generator.runner.step = original_step
            generator.close()
            report["after_close"] = memory_stats()
            report["manifest_sha256"] = profile.manifest_sha256
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
