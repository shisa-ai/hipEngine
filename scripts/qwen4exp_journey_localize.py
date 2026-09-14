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
ARMS["base_plus_moe_flags"] = {}
ARMS["base_plus_q8_gr_flags"] = {}
ARMS["base_plus_gdn_flags"] = {}
ARMS["q8_selected_down_strict"] = {PREFIX + "Q8_0_SELECTED_WMMA_DOWN": "0"}
ARMS["base_plus_q8_selected_down_strict"] = {
    **ARMS["all_numerics_strict"], **ARMS["q8_selected_down_strict"],
}
for name in ("strict_plus_gdn", "strict_plus_gdn_serial",
             "strict_plus_gdn_multi", "strict_prefill_production_decode",
             "production_prefill_strict_decode"):
    ARMS[name] = {}
OUTLIER = "heldout_general_ja_speculative"
FAMILY_PREFIXES = {
    "matrix": ("Q4_", "Q51_", "Q5_", "GROUPED_", "FORKB_", "PROFILE_Q5_1_",
               "PRODUCTION_MOE_", "Q8_0_"),
    "dense": ("Q8_MMQ_", "Q8_IU8_", "GR_"),
    "qsa": ("QSA_",),
}
for family in FAMILY_PREFIXES:
    ARMS["strict_plus_" + family] = {}
    ARMS["q8down_off_without_" + family] = {}
for flag in ("Q8_MMQ_PREFILL", "Q8_IU8_WMM", "GR_IU8", "GR_IU8_DOWN"):
    ARMS["strict_plus_" + flag.lower()] = {}
    ARMS["q8down_off_without_" + flag.lower()] = {
        **ARMS["q8_selected_down_strict"], PREFIX + flag: "0"}


def clear_arm_graphs(runner):
    from hipengine.runtime.moe_graph import MoeGraphCache

    runner.runtime.device_synchronize()
    for name in ("moe_graph_cache", "layer_graph_cache"):
        cache = getattr(runner, name, None)
        if cache is not None:
            enabled = cache.enabled
            cache.close()
            setattr(runner, name, MoeGraphCache(runner.runtime, enabled=enabled))


def strict_family_overrides(strict_flags):
    families = {
        "base_plus_moe_flags": (
            "Q4_", "Q51_", "Q5_", "GROUPED_", "FORKB_", "PROFILE_Q5_1_",
            "Q8_0_SELECTED_WMMA_DOWN",
        ),
        "base_plus_q8_gr_flags": ("Q8_", "GR_"),
        "base_plus_gdn_flags": ("GDN_",),
    }
    return {
        name: {
            **ARMS["all_numerics_strict"],
            **{key: value for key, value in strict_flags.items()
               if any(key.startswith(PREFIX + prefix) for prefix in prefixes)},
        } for name, prefixes in families.items()
    }


def gdn_isolation_overrides(strict_flags, production_flags):
    production_gdn = {key: value for key, value in production_flags.items()
                      if key.startswith(PREFIX + "GDN_")}
    isolated = {**strict_flags, **production_gdn}
    return {
        "strict_plus_gdn": isolated,
        "strict_plus_gdn_serial": {**isolated, **ARMS["gdn_strict"]},
        "strict_plus_gdn_multi": {
            **isolated,
            PREFIX + "GDN_TILE16_VARIANT": "qwen4exp_gdn_tiled16_multi_prefill",
        },
    }


def set_flags(flags):
    for key, value in flags.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def split_trajectory(runner, prompt_ids, forced_ids, prefill_flags, decode_flags):
    # Reconstruct the complete prefix in one runner; snapshots omit KV data.
    clear_arm_graphs(runner)
    set_flags(prefill_flags)
    runner.reset()
    result = runner.prefill(prompt_ids)
    logits = [np.array(result.logits, dtype=np.float32, copy=True)]
    clear_arm_graphs(runner)
    set_flags(decode_flags)
    for token in forced_ids:
        result = runner.step(token)
        logits.append(np.array(result.logits, dtype=np.float32, copy=True))
    return logits


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
        ARMS.update(strict_family_overrides(strict_flags))
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
        ARMS.update(gdn_isolation_overrides(strict_flags, os.environ))
        for family, prefixes in FAMILY_PREFIXES.items():
            keys = [key for key in strict_flags
                    if any(key.startswith(PREFIX + prefix) for prefix in prefixes)]
            ARMS["strict_plus_" + family] = {
                **strict_flags, **{key: os.environ.get(key) for key in keys}}
            ARMS["q8down_off_without_" + family] = {
                **ARMS["q8_selected_down_strict"],
                **{key: strict_flags[key] for key in keys}}
        for flag in ("Q8_MMQ_PREFILL", "Q8_IU8_WMM", "GR_IU8", "GR_IU8_DOWN"):
            ARMS["strict_plus_" + flag.lower()] = {
                **strict_flags, PREFIX + flag: "1"}
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
                    forced = [r["token_id"] for r in strict[key][:-1]]
                    if arm in ("strict_prefill_production_decode",
                               "production_prefill_strict_decode"):
                        pre, dec = ({**bound, **strict_flags}, bound)
                        if arm == "production_prefill_strict_decode":
                            pre, dec = dec, pre
                        candidate.extend(split_trajectory(
                            generator.runner, tokens[key], forced, pre, dec))
                    else:
                        run = _candidate_trajectory(generator.runner, tokens[key], forced)
                        candidate.extend(r["logits"] for r in run)
                candidate_logits = np.stack(candidate)
                quality = compare_profile_logits(strict_logits, candidate_logits, descriptors)
                quality["all_rows"] = [
                    {**descriptor.to_dict(), **compare_profile_logits(
                        strict_logits[i:i + 1], candidate_logits[i:i + 1],
                        [descriptor])["summary"]}
                    for i, descriptor in enumerate(descriptors)
                ]
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
