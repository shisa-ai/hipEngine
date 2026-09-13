"""Counterbalanced complete-model PLE gather A/B without a runtime selector.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("copy_elision", "sampled_dedup_elision"), required=True)
    parser.add_argument("--screen-only", action="store_true")
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
        cases = fixture["cases"]
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
                "warmups_per_arm_case": 1, "repetitions_per_arm": 1 if args.screen_only else 3,
                "screen_only": args.screen_only, "counterbalanced": True,
                "cache": "warm per case/arm; no persistent row cache",
            },
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_script_sha256": hashlib.sha256((ROOT / "scripts/qwen4exp_ple_gather_screen.py").read_bytes()).hexdigest(),
            "samples": [],
        }
        generator, profile, _ = _make_generator(args, "production")
        table = generator._resident.ple_table
        original_gather = table.gather_rows
        original_step = generator.runner.step
        last = {}

        def step(*a, **kw):
            result = original_step(*a, **kw)
            last["result"] = result
            return result

        generator.runner.step = step

        def sample(mode, case, rep):
            if table.telemetry() is not None or table._random_access_requested_mode != "off":
                raise ValueError("unqualified telemetry/advice configuration")
            table.gather_rows = original_gather if mode == "before" else (
                lambda ids: gather_sorted_unique(table, ids, method=args.method)
            )
            row = _hipengine_case_sample(generator.runner, case=case, repetition=rep, transitions=transitions)
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
            for index, case in enumerate(cases):
                warm = [sample(mode, case, -1) for mode in arm_sequence(index)[:2]]
                for field in ("output_token_ids_sha256", "final_logits_sha256", "final_state_sha256"):
                    if warm[0][field] != warm[1][field]:
                        raise ValueError(f"warmup differs: {case['id']} {field}")
                counts = {"before": 0, "after": 0}
                order = arm_sequence(index)[:2] if args.screen_only else arm_sequence(index)
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
                report["samples"], repetitions_per_mode=1 if args.screen_only else 3,
            )
            report["exact_final_logits_and_state"] = True
            report["status"] = "completed"
        finally:
            table.gather_rows = original_gather
            generator.runner.step = original_step
            generator.close()
            report["after_close"] = memory_stats()
            report["manifest_sha256"] = profile.manifest_sha256
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
