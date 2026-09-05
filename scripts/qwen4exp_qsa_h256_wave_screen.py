#!/usr/bin/env python3
"""Counterbalanced H256 sparse attention screen, no selection-policy changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_qwen4exp_qsa_h256_wave import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[16, 64, 512])
    parser.add_argument("--selected", type=int, default=2051)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1 or not 1 <= args.selected <= 4352 or any(r <= 0 for r in args.rows):
        parser.error("positive rows/pairs and selected in 1..4352 required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    report = {
        "schema": 1, "kind": "qwen4exp_h256_wave_attention_screen",
        "command": sys.argv, "source": _git_metadata(ROOT), "host": _host_metadata(),
        "arithmetic_class": "T0", "runtime_default_changed": False,
        "boundary": "selected paged BF16 K/V attention to F32 output; selection and KV publication excluded",
        "geometry": {"query_heads":24,"kv_heads":2,"head_dim":256,"capacity":4352,"block_size":256},
        "fixture": "seed506 random Q/BF16 KV, per-row page permutations and sorted unique selected positions",
        "cases": [],
    }
    for rows in args.rows:
        f = Fixture(rows, args.selected)
        try:
            f.run(False)
            f.run(True)
            timing = {"parent": [], "candidate": []}
            for pair in range(args.pairs):
                for candidate in ((False, True) if pair % 2 == 0 else (True, False)):
                    start = time.perf_counter()
                    f.run(candidate)
                    timing["candidate" if candidate else "parent"].append(time.perf_counter()-start)
                np.testing.assert_array_equal(f.download(True).view(np.uint32), f.download(False).view(np.uint32))
            report["cases"].append({
                "rows": rows, "selected_stride": args.selected,
                "selected_sha256": hashlib.sha256(f.selected).hexdigest(),
                "counts_sha256": hashlib.sha256(f.counts).hexdigest(),
                "seconds": timing, "all_pairs_exact": True,
                "speedup": statistics.median(timing["parent"])/statistics.median(timing["candidate"]),
            })
        finally:
            f.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
