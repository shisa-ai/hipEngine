#!/usr/bin/env python3
"""D6 screen: does one fused multi-norm launch beat the chain it replaces?

The Gemma 4 decode layer runs three normalizations over the SAME residual row
(pre-FFN, pre-FFN-2, the weightless router norm) and two weighted normalizations
over different rows (dense vs expert branch outputs). At rows=1 every one of
those launches is a single 256-thread block with sub-microsecond real work, so
the norm family's cost is dispatch: 301 launches/token for 1.52-1.56 ms/token.

This screen measures, in one session at the decode geometry (rows=1,
hidden=2816):

  chain_g3   three standalone launches over one input (the G3 site)
  multi_g3   one multi_rmsnorm launch, count=3
  chain_g4   two standalone launches over two inputs (the G4 site)
  multi_g4   one multi_rmsnorm launch, count=2
  probe_*    the D5 fixed-cost probe: multi(count=1) vs one standalone launch

The probe exists because iteration-D5's rejected pack kernel carried an
unexplained ~20 us fixed cost at rows=1 while its chain ran 3.6 us. A fused
norm kernel that costs more per call than the launch it removes is a negative
result, however good its arithmetic looks.

Bitwise agreement between every fused output and its chain output is asserted
in-process and recorded in the artifact; the chain-vs-fused equality is also
covered by tests/test_unit_gemma4_gpu_kernels.py.

Run (GPU chosen by HIP_VISIBLE_DEVICES or --gpu):
    python3 scripts/gemma4_norm_fusion_screen.py --gpu 1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _f32_to_bf16_u16(arr) -> "np.ndarray":
    import numpy as np

    bits = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "2026-09-30-gemma4-norm-fusion-screen.json",
    )
    args = parser.parse_args()

    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.gpu))
    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        build_gemma4_norm,
        gemma4_branch_add_bf16,
        gemma4_dense_combine_rmsnorm_scale_bf16,
        gemma4_multi_rmsnorm_bf16,
        gemma4_rmsnorm_f32w_bf16,
        gemma4_rmsnorm_weightless_bf16,
        gemma4_add_rmsnorm_scale_bf16,
    )

    rt = get_hip_runtime()
    build_gemma4_norm(load=True)

    rows, hidden = args.rows, args.hidden
    rng = np.random.default_rng(63)
    _buffers: list = []

    def dev(array) -> int:
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes)
        copy_host_to_device(buffer, host_array_ptr(host), host.nbytes)
        _buffers.append(buffer)
        return buffer.ptr

    src = dev(_f32_to_bf16_u16(rng.standard_normal((rows, hidden))))
    dense = dev(_f32_to_bf16_u16(rng.standard_normal((rows, hidden))))
    expert = dev(_f32_to_bf16_u16(rng.standard_normal((rows, hidden))))
    resid = dev(_f32_to_bf16_u16(rng.standard_normal((rows, hidden))))
    wa = dev((rng.standard_normal(hidden) * 0.05).astype(np.float32))
    wb = dev((rng.standard_normal(hidden) * 0.05).astype(np.float32))
    wc = dev((rng.standard_normal(hidden) * 0.05).astype(np.float32))
    out = [dev(np.zeros((rows, hidden), dtype=np.uint16)) for _ in range(5)]
    eps = 1e-6

    def bench(fn) -> float:
        for _ in range(args.warmup):
            fn()
        rt.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        rt.device_synchronize()
        return (time.perf_counter() - t0) / args.iters * 1e6  # us/call

    arms = {
        "chain_g3": lambda: (
            gemma4_rmsnorm_f32w_bf16(src, wa, out[0], rows, hidden, eps),
            gemma4_rmsnorm_f32w_bf16(src, wb, out[1], rows, hidden, eps),
            gemma4_rmsnorm_weightless_bf16(src, out[2], rows, hidden, eps),
        ),
        "multi_g3": lambda: gemma4_multi_rmsnorm_bf16(
            src, src, src, wa, wb, 0, out[0], out[1], out[2], 3, rows, hidden, eps
        ),
        "chain_g4": lambda: (
            gemma4_rmsnorm_f32w_bf16(dense, wa, out[0], rows, hidden, eps),
            gemma4_rmsnorm_f32w_bf16(expert, wb, out[1], rows, hidden, eps),
        ),
        "multi_g4": lambda: gemma4_multi_rmsnorm_bf16(
            dense, expert, src, wa, wb, 0, out[0], out[1], out[2], 2, rows, hidden, eps
        ),
        "probe_single": lambda: gemma4_rmsnorm_f32w_bf16(src, wa, out[0], rows, hidden, eps),
        "probe_multi1": lambda: gemma4_multi_rmsnorm_bf16(
            src, 0, 0, wa, 0, 0, out[0], out[1], out[2], 1, rows, hidden, eps
        ),
        # The tail: post_ffw_norm_1 -> branch_add -> add_rmsnorm_scale. The
        # chain normalizes into a scratch buffer instead of in place so 5000
        # iterations do not feed each other's output (in-place re-normalization
        # would drift toward denormals and slow the chain artificially).
        "chain_tail": lambda: (
            gemma4_rmsnorm_f32w_bf16(dense, wa, out[3], rows, hidden, eps),
            gemma4_branch_add_bf16(out[3], expert, out[4], rows * hidden),
            gemma4_add_rmsnorm_scale_bf16(out[4], resid, wc, 0, out[0], rows, hidden, eps),
        ),
        "fold_tail": lambda: gemma4_dense_combine_rmsnorm_scale_bf16(
            dense, expert, resid, wa, wc, 0, out[0], rows, hidden, eps
        ),
    }

    results = {name: round(bench(fn), 3) for name, fn in arms.items()}

    # Bitwise agreement: one execution of each fused arm against its chain,
    # reading back raw bf16 bits (the RED tests assert this too; the artifact
    # records the proof at the exact screen geometry).
    def read(ptr) -> "np.ndarray":
        buf = np.empty((rows, hidden), dtype=np.uint16)
        copy_device_to_host(host_array_ptr(buf), DeviceBuffer(ptr=ptr, nbytes=buf.nbytes), buf.nbytes)
        return buf

    chain_g3_out = []
    gemma4_rmsnorm_f32w_bf16(src, wa, out[0], rows, hidden, eps)
    chain_g3_out.append(read(out[0]))
    gemma4_rmsnorm_f32w_bf16(src, wb, out[1], rows, hidden, eps)
    chain_g3_out.append(read(out[1]))
    gemma4_rmsnorm_weightless_bf16(src, out[2], rows, hidden, eps)
    chain_g3_out.append(read(out[2]))
    gemma4_multi_rmsnorm_bf16(src, src, src, wa, wb, 0, out[0], out[1], out[2], 3, rows, hidden, eps)
    bit_equal_g3 = all(np.array_equal(c, read(o)) for c, o in zip(chain_g3_out, out[:3]))
    chain_g4_a = None
    gemma4_rmsnorm_f32w_bf16(dense, wa, out[0], rows, hidden, eps)
    chain_g4_a = read(out[0])
    gemma4_rmsnorm_f32w_bf16(expert, wb, out[1], rows, hidden, eps)
    chain_g4_b = read(out[1])
    gemma4_multi_rmsnorm_bf16(dense, expert, src, wa, wb, 0, out[0], out[1], out[2], 2, rows, hidden, eps)
    bit_equal_g4 = np.array_equal(chain_g4_a, read(out[0])) and np.array_equal(chain_g4_b, read(out[1]))

    # Tail fold bitwise: chain into scratch, fold into out[1], compare.
    gemma4_rmsnorm_f32w_bf16(dense, wa, out[3], rows, hidden, eps)
    gemma4_branch_add_bf16(out[3], expert, out[4], rows * hidden)
    gemma4_add_rmsnorm_scale_bf16(out[4], resid, wc, 0, out[2], rows, hidden, eps)
    tail_chain = read(out[2])
    gemma4_dense_combine_rmsnorm_scale_bf16(dense, expert, resid, wa, wc, 0, out[1], rows, hidden, eps)
    bit_equal_tail = np.array_equal(tail_chain, read(out[1]))

    delta_g3 = results["chain_g3"] - results["multi_g3"]  # us saved per site
    delta_g4 = results["chain_g4"] - results["multi_g4"]
    delta_tail = results["chain_tail"] - results["fold_tail"]
    layers = 30
    projected_ms = (layers * 3 * delta_g3 + layers * delta_g4 + layers * delta_tail) / 1000.0
    probe_overhead = results["probe_multi1"] - results["probe_single"]
    probe_ratio = results["probe_multi1"] / max(results["probe_single"], 1e-9)
    wins = delta_g3 > 0 and delta_g4 > 0 and delta_tail > 0
    # The D5 fixed-cost alarm is ABSOLUTE: that pathology was a fused kernel
    # costing ~19.8 us where its chain ran 3.6 us (+16 us). At a 5.5 us single
    # launch the ratio is noise-dominated -- it read 1.083 and 1.107 on two
    # consecutive runs of the same binary -- so the gate is the absolute
    # overhead a fixed cost could impose on every site, with the ratio reported
    # for information.
    probe_ok = probe_overhead <= 1.0
    bits_ok = bit_equal_g3 and bit_equal_g4 and bit_equal_tail
    verdict = "wire" if (wins and probe_ok and bits_ok) else "record"

    artifact = {
        "schema": "gemma4-norm-fusion-screen/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": f"python3 scripts/gemma4_norm_fusion_screen.py --gpu {args.gpu} --rows {rows} --hidden {hidden} --iters {args.iters}",
        "git_commit": _git_head(),
        "arch": platform.machine(),
        "gpu_env": {"HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "")},
        "geometry": {"rows": rows, "hidden": hidden, "layers": layers, "iters": args.iters, "warmup": args.warmup},
        "arms_us_per_call": results,
        "bit_equal": {"g3": bool(bit_equal_g3), "g4": bool(bit_equal_g4), "tail": bool(bit_equal_tail)},
        "delta_us": {"g3": round(delta_g3, 3), "g4": round(delta_g4, 3), "tail": round(delta_tail, 3)},
        "projected_norm_family_ms_per_token": round(projected_ms, 4),
        "probe": {
            "multi1_minus_single_us": round(probe_overhead, 3),
            "limit_us": 1.0,
            "ratio": round(probe_ratio, 3),
            "ratio_note": "informational; the gate is the absolute overhead (see evidence_note)",
        },
        "verdict": verdict,
        "evidence_note": "Bitwise agreement asserted in-process at the screen geometry. Projected saving counts 3 G3 sites and 1 G4 site per layer x 30 layers plus the tail fold. The G3/G4 arms are kernel-level capability: at the real layer, pre_ffw runs on the main stream and pre_ffw_2 on moe_stream, so those sites need stream-aware wiring (recorded in the worklog entry); the tail fold is strictly sequential same-stream and wires directly. The shipped-path claim is the same-lane census A/B, not this projection.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")

    print(json.dumps(artifact, indent=2))
    for buffer in _buffers:
        free(buffer)
    return 0 if verdict == "wire" else 2


if __name__ == "__main__":
    sys.exit(main())