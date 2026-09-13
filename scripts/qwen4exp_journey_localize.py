"""Diagnose incumbent numerical drift with named-family ablations.

Full18 prefill-last rows plus the previously failing Japanese decode chain.
One numerical run per arm: not a promotion, determinism or task certificate.
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

from hipengine.benchmark.execution_profiles import RowDescriptor, compare_profile_logits
from scripts.gguf_gdn_semantic_gate import DEFAULT_PROMPTS, _load_suites
from scripts.gguf_mtp_bench import build_chat_prompt
from scripts.qwen4exp_layer2_profile_gate import _make_generator, _strict_trajectory, _candidate_trajectory
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

PREFIX = "HIPENGINE_QWEN4_EXP_"
ARMS = {
    "bound": {},
    "gdn_strict": {PREFIX + key: "0" for key in ("GDN_PEER_PREFILL", "GDN_COLWARPS_PREFILL")},
    "moe_strict": {PREFIX + key: "0" for key in ("PRODUCTION_MOE_PREFILL", "Q4_IU8_PREFILL")},
    "dense_q8_gr_strict": {PREFIX + key: "0" for key in ("Q8_MMQ_PREFILL", "Q8_IU8_WMM", "GR_IU8", "GR_IU8_DOWN")},
    "decode_strict": {PREFIX + "Q4_DP4A64": "0"},
}
ARMS["all_numerics_strict"] = {
    key: value for name, values in ARMS.items() if name != "bound"
    for key, value in values.items()
}
ARMS["all_flags_strict"] = {}
ARMS["gdn_flags_strict"] = {}
OUTLIER = "heldout_general_ja_speculative"


def clear_arm_graphs(runner):
    from hipengine.runtime.moe_graph import MoeGraphCache

    runner.runtime.device_synchronize()
    for name in ("moe_graph_cache", "layer_graph_cache"):
        cache = getattr(runner, name, None)
        if cache is not None:
            enabled = cache.enabled
            cache.close()
            setattr(runner, name, MoeGraphCache(runner.runtime, enabled=enabled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    args = parser.parse_args()
    check_host()
    args.max_sequence_length = 2051
    args.prefill_chunk_size = 1024
    os.environ["HIPENGINE_HIP_ARCH"] = "gfx1151"
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        prompt_rows = _load_suites(DEFAULT_PROMPTS)
        reset_memory_stats()
        identity = model_identity(args.model_root)
        strict_generator, strict_profile, _ = _make_generator(args, "strict")
        strict_flags = {key: value for key, value in os.environ.items()
                        if key.startswith(PREFIX) or key == "HIPENGINE_GGUF_WMMA_PREFILL"}
        strict = {}
        tokens = {}
        descriptors = []
        try:
            for row in prompt_rows:
                key = row["id"]
                tokens[key] = build_chat_prompt(strict_generator.tokenizer, row["prompt"])
                strict[key] = _strict_trajectory(strict_generator.runner, tokens[key], 32 if key == OUTLIER else 0)
                for step, sample in enumerate(strict[key]):
                    descriptors.append(RowDescriptor(
                        scenario_id="journey-family-localization", scenario_step=len(descriptors),
                        request_id=key, teacher_step=step, category=row["category"],
                        shape="prefill_last" if step == 0 else "c1",
                        transition="prefill_to_c1" if step == 0 else "steady",
                        teacher_token_id=sample["token_id"],
                    ))
                print("strict", key, flush=True)
        finally:
            strict_generator.close()
        strict_close = memory_stats()
        ARMS["all_flags_strict"] = strict_flags
        ARMS["gdn_flags_strict"] = {key: value for key, value in strict_flags.items()
                                   if key.startswith(PREFIX + "GDN_")}
        strict_logits = np.stack([r["logits"] for p in prompt_rows for r in strict[p["id"]]])
        report = {
            "kind": "journey_incumbent_family_localization", "status": "running",
            "performance_claim": False, "promotion_eligible": False,
            "command": sys.argv, "source": _git_metadata(ROOT), "host": _host_metadata(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_identity": identity,
            "strict_manifest_sha256": strict_profile.manifest_sha256,
            "protocol": {"prefill_prompts": 18, "outlier_decode_steps": 32,
                         "rows": len(descriptors), "chunk": 1024, "kv": "BF16",
                         "repeats": 1, "task_gate": "not_run",
                         "graph_cache_reset_per_arm": True},
            "arms": {}, "strict_after_close": strict_close,
        }
        generator, production_profile, _ = _make_generator(args, "production")
        controlled = set(key for values in ARMS.values() for key in values)
        bound = {key: os.environ.get(key) for key in controlled}
        try:
            for arm in args.arms:
                # Family overrides can affect captured nodes beyond the normal
                # production cache key. Never reuse an earlier arm's graphs.
                clear_arm_graphs(generator.runner)
                for key, value in bound.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                os.environ.update(ARMS[arm])
                candidate = []
                for row in prompt_rows:
                    key = row["id"]
                    run = _candidate_trajectory(generator.runner, tokens[key],
                                                [r["token_id"] for r in strict[key][:-1]])
                    candidate.extend(r["logits"] for r in run)
                quality = compare_profile_logits(strict_logits, np.stack(candidate), descriptors)
                report["arms"][arm] = {"overrides": ARMS[arm], "quality": quality}
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(arm, quality["summary"], flush=True)
            report["status"] = "completed"
        finally:
            for key, value in bound.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            generator.close()
            report["after_close"] = memory_stats()
            report["production_manifest_sha256"] = production_profile.manifest_sha256
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
