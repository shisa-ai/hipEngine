"""Counterbalanced same-residency candidate versus current production."""

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

from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE, load_fixture, _hipengine_case_sample, _host_metadata, _git_metadata,
)
from scripts.qwen4exp_conservative_cost import summarize_cost
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _state_summary, CANDIDATES
from scripts.qwen4exp_journey_localize import set_flags
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity


SHARED_DECODE_SAFE_FLAGS = frozenset(
    "HIPENGINE_QWEN4_EXP_" + flag for flag in (
        "Q8_0_SELECTED_WMMA_DOWN", "Q8_DOWN_VARIANT",
        "QSA_H256_WAVE_PREFILL", "QSA_HEAD_PAIR",
        "QSA_ORDERED_DECODE", "QSA_ORDERED_DECODE_V2",
    )
)


def validate_shared_graphs(environments):
    # These switches affect prefill or uncaptured QSA attention, not the
    # captured GDN and MoE decode units or their graph keys.
    changed = {key for env in environments for key in env}
    if not changed <= SHARED_DECODE_SAFE_FLAGS:
        raise ValueError("candidate can change captured decode units")


def round_order(case_index, arms, repetition):
    order = list(arms)
    if case_index % 2:
        order.reverse()
    offset = repetition % len(order)
    return order[offset:] + order[:offset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True, action="append")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--shared-decode-graphs", action="store_true")
    args = parser.parse_args()
    check_host()
    if args.pairs < 3:
        parser.error("at least three pairs required")
    if len(set(args.candidate)) != len(args.candidate):
        parser.error("duplicate candidates")
    candidate_envs = {name: dict(CANDIDATES[name].environment) for name in args.candidate}
    if args.shared_decode_graphs:
        validate_shared_graphs(candidate_envs.values())
    args.prefill_chunk_size = 1024
    args.max_sequence_length = 4096 + 128 + 8
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.runtime.gguf_linear import clear_gguf_linear_dispatch_cache
    from hipengine.runtime.moe_graph import MoeGraphCache

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        reset_memory_stats()
        fixture, digest = load_fixture(DEFAULT_FIXTURE)
        report = dict(status="running", command=sys.argv, source=_git_metadata(ROOT),
                      host=_host_metadata(), model=model_identity(args.model_root),
                      fixture_sha256=digest, samples=[], performance_claim=False,
                      candidates=args.candidate,
                      protocol=dict(pairs=args.pairs, warmups=1, chunk=1024, kv="BF16",
                                    shared_decode_graphs=args.shared_decode_graphs))
        generator, profile, _ = _make_generator(args, "production")
        runner = generator.runner
        controlled = {key for env in candidate_envs.values() for key in env}
        previous = {key: os.environ.get(key) for key in controlled}
        arm_envs = {"before": previous}
        arm_envs.update({name: {**previous, **env} for name, env in candidate_envs.items()})
        report["arm_overrides"] = arm_envs
        original_step = runner.step
        last = {}
        names = ("moe_graph_cache", "layer_graph_cache")
        caches = {"before": {name: getattr(runner, name) for name in names}}
        for candidate in args.candidate:
            caches[candidate] = caches["before"] if args.shared_decode_graphs else {
                name: MoeGraphCache(runner.runtime, enabled=cache.enabled)
                for name, cache in caches["before"].items()
            }

        def step(*a, **kw):
            result = original_step(*a, **kw)
            last["result"] = result
            return result

        runner.step = step

        def sample(arm, case, rep):
            runner.runtime.device_synchronize()
            for name, cache in caches[arm].items():
                setattr(runner, name, cache)
            clear_gguf_linear_dispatch_cache()
            set_flags(arm_envs[arm])
            if os.environ.get("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL") == "1":
                runner.configure_mmq_prefill_resources()
            row = _hipengine_case_sample(
                runner, case=case, repetition=rep, transitions=fixture["decode_transitions"])
            logits = np.asarray(last["result"].logits)
            state = _state_summary(runner)
            if not logits.size or not np.isfinite(logits).all() or not state["finite"]:
                raise ValueError("nonfinite logits/state")
            row.update(arm=arm, mode="before" if arm == "before" else "after", finite=True,
                       logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                       state_sha256=state["state_sha256"])
            return row

        try:
            for index, case in enumerate(fixture["cases"]):
                arms = tuple(arm_envs)
                for arm in arms:
                    sample(arm, case, -1)
                for repetition in range(args.pairs):
                    for slot, arm in enumerate(round_order(index, arms, repetition)):
                        row = sample(arm, case, repetition)
                        row["sequence_slot"] = repetition * len(arms) + slot
                        report["samples"].append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(case["id"], arm, row["prefill_tok_s"], row["decode_tok_s"], flush=True)
            report["comparisons"] = {
                candidate: summarize_cost(
                    [row for row in report["samples"] if row["arm"] in ("before", candidate)],
                    args.pairs)
                for candidate in args.candidate
            }
            for case in fixture["cases"]:
                for arm in arm_envs:
                    rows = [r for r in report["samples"] if
                            r["case_id"] == case["id"] and r["arm"] == arm]
                    for field in ("logits_sha256", "state_sha256"):
                        if len({row[field] for row in rows}) != 1:
                            raise ValueError(f"nonrepeatable {field}: {case['id']} {arm}")
            report["status"] = "completed"
        finally:
            set_flags(previous)
            runner.step = original_step
            runner.runtime.device_synchronize()
            report["graph_stats"] = {
                arm: {name: cache.stats for name, cache in arm_caches.items()}
                for arm, arm_caches in caches.items()
            }
            for mode_caches in caches.values():
                for name, cache in mode_caches.items():
                    if cache is not getattr(runner, name):
                        cache.close()
            generator.close()
            report["after_close"] = memory_stats()
            report["base_manifest"] = profile.manifest_sha256
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
