#!/usr/bin/env python3
"""#29 R13 routing-distribution capture and tile-policy derivation.

Patches the MoE group-count wrapper to download the per-expert row counts
for every (layer, chunk) during real prefills, then derives the
row-count-based expert-tile policy from OUR routing distribution:

- the per-expert count histogram (mean/median/percentiles/max);
- padding waste at the current fixed 16-row tiles;
- the waste-minimizing assignment over candidate tile sizes {16, 32, 48}
  chosen per expert count (thresholds derived from the measured
  distribution, not copied from any external engine);
- the aggregate waste for each single-tile and the derived mixed policy.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from hipengine.core.hip import get_hip_runtime


def hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--case-id", action="append")
    p.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 48])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not hip_available():
        p.error("HIP runtime unavailable")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
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
    import hipengine.runtime.qwen4_exp_runner as runner_module

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
    fixture, digest = load_fixture(DEFAULT_FIXTURE)
    index = load_gguf_index(discover_gguf_files(a.model_root)[0])
    generator = resolved.construct_generator(lambda: Qwen4ExpGGUFTextGenerator(
        model_path=a.model_root, weight_index=index,
        model_plugin=resolve_model(index.architecture or ""),
        backend=QWEN4_EXP_BACKEND, max_sequence_length=4352,
        prefill_chunk_size=1024))

    runtime = get_hip_runtime()
    original = runner_module.qwen35_moe_group_count
    captures: list[np.ndarray] = []

    def counted(selected_ptr, counts_ptr, compact, experts, *,
                stream=0, runtime=None, **kw):
        result = original(selected_ptr, counts_ptr, compact, experts,
                          stream=stream, runtime=runtime or get_hip_runtime(),
                          **kw)
        host = np.empty(int(experts), dtype=np.int32)
        active = runtime or get_hip_runtime()
        active.device_synchronize()
        active.memcpy(host.ctypes.data, counts_ptr, host.nbytes, 2)  # D2H
        captures.append(host.copy())
        return result

    runner_module.qwen35_moe_group_count = counted
    try:
        for case in fixture["cases"]:
            if a.case_id and case["id"] not in a.case_id:
                continue
            generator.runner.prefill(case["prompt_token_ids"])
            generator.runner.runtime.device_synchronize()
            print(f"{case['id']}: {len(captures)} (layer,chunk) captures so far",
                  flush=True)
            generator.runner.reset()
    finally:
        runner_module.qwen35_moe_group_count = original

    all_counts = np.concatenate(captures) if captures else np.array([])
    positive = all_counts[all_counts > 0]
    hist = Counter(positive.tolist())

    def waste(counts, tile):
        return int((np.ceil(counts / tile) * tile - counts).sum())

    total = int(positive.sum())
    policies = {}
    for tile in a.tiles:
        policies[f"fixed{tile}"] = {
            "padded_rows": int(np.ceil(positive / tile).sum() * tile),
            "waste_rows": waste(positive, tile),
            "waste_frac": waste(positive, tile) / total,
        }
    # derived mixed policy: per expert-count, pick the tile minimizing waste
    # (ties prefer the smaller tile for occupancy granularity)
    tile_choice = {}
    mixed_waste = 0
    mixed_padded = 0
    for count_value, n in hist.items():
        best_tile, best_w = None, None
        for tile in a.tiles:
            w = (-(-count_value // tile)) * tile - count_value
            if best_w is None or w < best_w:
                best_tile, best_w = tile, w
        tile_choice[count_value] = best_tile
        mixed_waste += best_w * n
        mixed_padded += (-(-count_value // best_tile)) * best_tile * n
    policies["derived_mixed"] = {
        "padded_rows": int(mixed_padded),
        "waste_rows": int(mixed_waste),
        "waste_frac": mixed_waste / total,
        "tile_by_count": {str(k): v for k, v in sorted(tile_choice.items())},
        "thresholds": {
            "count_ranges_per_tile": {
                str(tile): [min([k for k, v in tile_choice.items() if v == tile], default=None),
                            max([k for k, v in tile_choice.items() if v == tile], default=None)]
                for tile in a.tiles
            }
        },
    }

    # pair-share by expert-count bucket (for sub-16 path break-even analysis)
    buckets = [(1, 2), (3, 4), (5, 8), (9, 12), (13, 16), (17, 24), (25, 32),
               (33, 48), (49, 64), (65, 128), (129, 1024)]
    bucket_pairs = {}
    bucket_instances = {}
    for lo, hi in buckets:
        mask = (positive >= lo) & (positive <= hi)
        bucket_pairs[f"{lo}-{hi}"] = int(positive[mask].sum())
        bucket_instances[f"{lo}-{hi}"] = int(mask.sum())

    report = {
        "schema": 1, "kind": "qwen4exp_moe_routing_distribution",
        "bucket_pairs": bucket_pairs,
        "bucket_instances": bucket_instances,
        "source": _git_metadata(ROOT), "host": _host_metadata(),
        "command": sys.argv, "fixture_sha256": digest,
        "captures": len(captures),
        "pairs_total": total,
        "count_percentiles": {
            "p50": float(np.percentile(positive, 50)),
            "p75": float(np.percentile(positive, 75)),
            "p90": float(np.percentile(positive, 90)),
            "p99": float(np.percentile(positive, 99)),
            "max": int(positive.max()),
        },
        "count_histogram_top": dict(hist.most_common(30)),
        "expert_saturation": {
            "nonzero_experts_per_capture_mean": float(
                np.mean([(c > 0).sum() for c in captures])),
            "pairs_per_capture_mean": float(np.mean([c.sum() for c in captures])),
        },
        "policies": policies,
        "note": "thresholds derived from this distribution; top_k=10, experts=512, chunk=1024 rows",
    }
    a.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({
        "captures": len(captures), "pairs": total,
        "percentiles": report["count_percentiles"],
        "policies": {k: round(v["waste_frac"], 4) for k, v in policies.items()},
        "derived_thresholds": policies["derived_mixed"]["thresholds"]["count_ranges_per_tile"],
    }, indent=1))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
