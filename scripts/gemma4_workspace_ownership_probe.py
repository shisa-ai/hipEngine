#!/usr/bin/env python3
"""Compare Gemma public-generation workspace ownership with unchanged arithmetic.

Supply a native gemma4_campaign_bench.py JSON packet. Fresh runners execute its
exact prompt, capacity and greedy output count, first with separate per-layer
attention owners and then with the shipping owner arrangement. Full-logit hashes
and allocator counts are diagnostics; instrumentation is not a latency benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    from scripts import gemma4_campaign_bench as bench
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import Gemma4AttentionScratch
    from hipengine.llm import SamplingParams

    baseline = json.loads(args.baseline_json.read_text())
    ids = baseline["prompt_token_ids"]
    model = Path(baseline["provenance"]["artifact"]["path"])
    capacity = int(baseline["loading"]["runner_capacity"])
    outputs = int(baseline["workload"]["output_tokens"])
    expected = baseline["samples"][0]["generated_token_ids"]
    rows = []
    for mode in ("per_layer_control", "shared_default"):
        llm, runner, loading = bench._resolve_generator(model, capacity)
        try:
            if mode == "per_layer_control":
                for owner in {id(s.attention): s.attention for s in runner._scratches}.values():
                    owner.close()
                for scratch in runner._scratches:
                    scratch.attention = Gemma4AttentionScratch()
            resident = memory_stats()
            reset_memory_stats()
            original = runner.forward
            hashes, finite = [], []

            def observed(*a, **kw):
                logits = original(*a, **kw)
                array = np.asarray(logits, dtype=np.float32)
                finite.append(bool(np.isfinite(array).all()))
                hashes.append(hashlib.sha256(array.tobytes()).hexdigest())
                return logits

            runner.forward = observed
            output = llm.generate_detailed(
                ids, SamplingParams(max_tokens=outputs, temperature=0, ignore_eos=True)
            )[0]
            get_hip_runtime().device_synchronize()
            owners = list({id(s.attention): s.attention for s in runner._scratches}.values())
            row = {
                "mode": mode, "resident": resident, "after": memory_stats(),
                "owners": len(owners),
                "owned_buffers": sum(len(x._owned) for x in owners),
                "owned_bytes": sum(b.nbytes for x in owners for b in x._owned),
                "current_workspace_bytes": [sum(b.nbytes for b in x._current.values()) for x in owners],
                "full_logits_hashes": hashes, "finite_logits": all(finite),
                "generated_ids": list(output.generated_token_ids), "loading": loading,
            }
            assert len(hashes) == outputs and all(finite)
            assert row["generated_ids"] == expected
            rows.append(row)
            print(mode, row["owners"], row["owned_bytes"], flush=True)
        finally:
            llm.close()
        assert all(x._closed for x in owners)
    assert rows[0]["full_logits_hashes"] == rows[1]["full_logits_hashes"]
    assert rows[1]["owners"] == 1 and rows[1]["owned_bytes"] < rows[0]["owned_bytes"]
    packet = {
        "kind": "public_workspace_control", "status": "ownership_pass", "performance_claim": False,
        "command": shlex.join([sys.executable, *sys.argv]),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "baseline_provenance": baseline["provenance"],
        "candidate_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "candidate_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--", "hipengine/runtime/gemma4.py"])).hexdigest(),
        "workload": {"prompt_tokens": len(ids), "output_tokens": outputs, "context": capacity, "samples_per_arm": 1},
        "note": "instrumented public generate_detailed; full prefill and greedy decode; no latency claim",
        "correctness": {"full_logits_finite": True, "full_logit_hashes_bit_identical": True, "generated_ids_equal": True},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
