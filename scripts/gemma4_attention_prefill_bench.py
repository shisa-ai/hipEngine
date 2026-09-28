#!/usr/bin/env python3
"""Time the Gemma 4 prefill attention kernel in isolation, at its real geometry.

The gfx1151 prefill route ablation puts attention at 30.5% of a 2048-token
prefill (2.295 s of 7.536 s). The campaign's gfx1100 record for the same kernel
at the same prompt length is 133.02 ms, which is 17x lower than the CU count and
bandwidth ratio between the two parts explains. This probe removes every other
launch from the measurement so the kernel's own rate is unambiguous, and reports
it per layer type and per window setting.

Usage:
    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \
        python3 scripts/gemma4_attention_prefill_bench.py --tokens 2048
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument(
        "--keys",
        type=int,
        default=None,
        help="KV length, independent of the query token count. Defaults to --tokens, "
             "which is the dense-prefill geometry. Setting it separately is what "
             "separates a cost that tracks the number of key iterations from one "
             "that tracks the grid, since --tokens scales both and --keys scales "
             "only the loop.",
    )
    ap.add_argument(
        "--dtype",
        choices=("bf16", "f32"),
        default="bf16",
        help="Storage width for q/k/v. The two run identical arithmetic over the same "
             "shapes and differ only in bytes moved, so comparing them is a traffic "
             "ablation that does not also remove the work -- which the --mask zero "
             "ablation does, since the masked branch contains both the load and the "
             "dot. A time ratio near 2.0 means the kernel is paying for bytes; near "
             "1.0 means it is not.",
    )
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument(
        "--mask",
        choices=("keep", "zero"),
        default="keep",
        help="'zero' runs the same shapes with an all-zero keep-mask, which skips "
             "every K/V load while leaving the key loop and the reductions intact. "
             "The gap between the two modes is the load share of the cost, and it "
             "is what separates a traffic-bound kernel from a latency-bound one.",
    )
    ap.add_argument(
        "--compare-wmma",
        action="store_true",
        help="Additionally time the BF16 WMMA prefill candidate "
             "(gemma4_attention_prefill_wmma) on the same device buffers as the "
             "strict row above it, and print its rate and the ratio. Additive: "
             "the strict rows are unchanged and the candidate is not selected by "
             "anything in the production path. Geometry it does not implement "
             "(the head_dim-512 full layers, f32 storage) reports n/a.",
    )
    ap.add_argument(
        "--compare-wmma-full",
        action="store_true",
        help="Additionally time the BF16 WMMA full-layer prefill candidate "
             "(gemma4_attention_prefill_wmma_full) on the same device buffers as "
             "the strict row above it, and print its rate and the ratio. Additive "
             "and independent of --compare-wmma: the strict rows are unchanged, "
             "and the candidate is not selected by anything in the production "
             "path. It implements the head_dim-512, 16q/2kv full layers only; "
             "any other geometry or f32 storage reports n/a.",
    )
    args = ap.parse_args()

    import numpy as np

    from hipengine.core.memory import (
        copy_host_array_to_device,
        free,
        malloc,
    )
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as ga
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_prefill_wmma as gw
    from hipengine.kernels.hip_gfx1100.gemma4 import (
        gemma4_attention_prefill_wmma_full as gwf,
    )

    def time_launch(launch, iters: int, warmup: int) -> float:
        """Median wall time of one launch, device-synchronized, in milliseconds."""

        for _ in range(warmup):
            launch()
        runtime.device_synchronize()
        samples = []
        for _ in range(iters):
            started = time.perf_counter()
            launch()
            runtime.device_synchronize()
            samples.append(time.perf_counter() - started)
        return statistics.median(samples) * 1000.0

    runtime = get_hip_runtime()
    tokens = args.tokens

    # (label, num_heads, num_kv_heads, head_dim, window, layers, counts_toward_total)
    #
    # The two sliding-window rows are the same 25 layers under two window
    # settings, so only the artifact's own (window=1024) row is part of the
    # model's attention cost; the window=0 row is a probe that says what the
    # window is worth. Summing all three rows counts those 25 layers twice and
    # adds a diagnostic that does not exist in the model, which is what an
    # earlier revision of this script printed.
    cases = [
        ("swa  256d 16q/8kv window=1024", 16, 8, 256, 1024, 25, True),
        ("swa  256d 16q/8kv window=0", 16, 8, 256, 0, 25, False),
        ("full 512d 16q/2kv window=0", 16, 2, 512, 0, 5, True),
    ]

    print(f"tokens={tokens} iters={args.iters} mask={args.mask} dtype={args.dtype}")
    print(f"{'case':32s} {'ms/launch':>10s} {'GFLOP/s':>10s} {'all layers':>11s}")

    total_ms = 0.0
    total_layers = 0
    for label, n_head, n_kv, head_dim, window, layers, counts in cases:
        key_count = tokens if args.keys is None else args.keys
        rng = np.random.default_rng(20260930)
        q = rng.standard_normal((tokens, n_head, head_dim)).astype(np.float32)
        k = rng.standard_normal((key_count, n_kv, head_dim)).astype(np.float32)
        v = rng.standard_normal((key_count, n_kv, head_dim)).astype(np.float32)
        mask = np.ones((tokens, key_count), dtype=np.uint8)
        for t in range(tokens):
            lo = 0 if window <= 0 else max(0, t - window + 1)
            mask[t, :lo] = 0
            mask[t, t + 1 :] = 0
        if args.mask == "zero":
            mask[:] = 0

        def bf16(a):
            bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
            rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            return (rounded >> 16).astype(np.uint16)

        qb, kb, vb = bf16(q), bf16(k), bf16(v)
        out = np.zeros((tokens, n_head, head_dim), dtype=np.uint16)
        prefill = ga.gemma4_attention_prefill_bf16
        if args.dtype == "f32":
            qb = np.ascontiguousarray(q, dtype=np.float32)
            kb = np.ascontiguousarray(k, dtype=np.float32)
            vb = np.ascontiguousarray(v, dtype=np.float32)
            out = np.zeros((tokens, n_head, head_dim), dtype=np.float32)
            prefill = ga.gemma4_attention_prefill_f32

        bufs = [
            malloc(qb.nbytes),
            malloc(kb.nbytes),
            malloc(vb.nbytes),
            malloc(mask.nbytes),
            malloc(out.nbytes),
        ]
        wmma_ms = None
        wmma_full_ms = None
        try:
            for buf, arr in zip(bufs, (qb, kb, vb, mask, out)):
                copy_host_array_to_device(buf, arr)

            def once():
                prefill(
                    bufs[0].ptr,
                    bufs[1].ptr,
                    bufs[2].ptr,
                    bufs[3].ptr,
                    bufs[4].ptr,
                    tokens=tokens,
                    num_heads=n_head,
                    num_kv_heads=n_kv,
                    head_dim=head_dim,
                    scale=1.0,
                    keys=key_count,
                    window=window,
                    row_offset=0,
                )

            ms = time_launch(once, args.iters, args.warmup)

            # Same buffers, same shapes, same mask: the candidate's row is the
            # strict row's row with one kernel swapped, so the two rates are
            # directly comparable. The strict timing above is already complete,
            # so overwriting `out` here cannot change it.
            if args.compare_wmma and args.dtype == "bf16" and gw.gemma4_attention_prefill_wmma_supported(
                num_heads=n_head, num_kv_heads=n_kv, head_dim=head_dim
            ):
                def once_wmma():
                    gw.gemma4_attention_prefill_wmma_bf16(
                        bufs[0].ptr,
                        bufs[1].ptr,
                        bufs[2].ptr,
                        bufs[3].ptr,
                        bufs[4].ptr,
                        tokens=tokens,
                        num_heads=n_head,
                        num_kv_heads=n_kv,
                        head_dim=head_dim,
                        scale=1.0,
                        keys=key_count,
                        window=window,
                        row_offset=0,
                    )

                wmma_ms = time_launch(once_wmma, args.iters, args.warmup)

            # Same buffers, same shapes, same mask, same argument list: the
            # full-layer candidate's row is the strict row's row with one
            # kernel swapped. It is a separate flag from --compare-wmma because
            # the two candidates cover disjoint geometries.
            if (
                args.compare_wmma_full
                and args.dtype == "bf16"
                and gwf.gemma4_attention_prefill_wmma_full_supported(
                    num_heads=n_head, num_kv_heads=n_kv, head_dim=head_dim
                )
            ):
                def once_wmma_full():
                    gwf.gemma4_attention_prefill_wmma_full_bf16(
                        bufs[0].ptr,
                        bufs[1].ptr,
                        bufs[2].ptr,
                        bufs[3].ptr,
                        bufs[4].ptr,
                        tokens=tokens,
                        num_heads=n_head,
                        num_kv_heads=n_kv,
                        head_dim=head_dim,
                        scale=1.0,
                        keys=key_count,
                        window=window,
                        row_offset=0,
                    )

                wmma_full_ms = time_launch(once_wmma_full, args.iters, args.warmup)
        finally:
            for buf in bufs:
                free(buf)

        # Causal pairs, windowed where a window applies. The kept range is
        # [lo, min(t + 1, key_count)) -- the same range the mask above encodes --
        # so it must use key_count and not tokens. Using tokens here silently
        # reported the flops for a different geometry whenever --keys differed.
        pairs = 0
        for t in range(tokens):
            lo = 0 if window <= 0 else max(0, t - window + 1)
            hi = min(t + 1, key_count)
            pairs += max(0, hi - lo)
        flops = 4.0 * pairs * n_head * head_dim
        gflops = flops / (ms / 1000.0) / 1e9
        if counts:
            total_ms += ms * layers
            total_layers += layers
        print(
            f"{label:32s} {ms:10.2f} {gflops:10.1f} {ms * layers:9.1f} ms"
            + ("" if counts else "   (probe, not in the total)")
        )
        if args.compare_wmma:
            if wmma_ms is None:
                reason = "f32 storage" if args.dtype != "bf16" else "unsupported geometry"
                print(f"{'  + wmma (candidate)':32s} {'n/a':>10s} {reason:>10s}")
            else:
                wmma_gflops = flops / (wmma_ms / 1000.0) / 1e9
                print(
                    f"{'  + wmma (candidate)':32s} {wmma_ms:10.2f} {wmma_gflops:10.1f}"
                    f" {wmma_ms * layers:9.1f} ms   {ms / wmma_ms:6.2f}x vs strict"
                )
        if args.compare_wmma_full:
            if wmma_full_ms is None:
                reason = "f32 storage" if args.dtype != "bf16" else "unsupported geometry"
                print(f"{'  + wmma-full (candidate)':32s} {'n/a':>10s} {reason:>10s}")
            else:
                wmma_full_gflops = flops / (wmma_full_ms / 1000.0) / 1e9
                print(
                    f"{'  + wmma-full (candidate)':32s} {wmma_full_ms:10.2f}"
                    f" {wmma_full_gflops:10.1f}"
                    f" {wmma_full_ms * layers:9.1f} ms   {ms / wmma_full_ms:6.2f}x vs strict"
                )

    print(
        f"\nthe model's {total_layers} attention layers: {total_ms:.1f} ms  "
        f"(the 2048-token route ablation measured 2295 ms for the same route in situ)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
