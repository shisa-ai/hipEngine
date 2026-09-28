"""Are the plain and token-tiled router logits kernels bit-identical?

`qwen35_router_logits_bf16_f32w` launches `dim3(num_rows, tokens)` -- one block
per (expert-row, token) -- so each F32 weight row is read once per token.
`..._token_tile_8` launches `dim3(num_rows, ceil(tokens / 8))`, so each weight
row is read once per eight tokens. gfx1151 declares
`LAGUNA_ROUTER_LOGITS_MODE = "token_tile_8"` with the admission "Exact
eight-token router tiling preserves every token/expert's K traversal and
reduction tree while reusing each F32 weight row twice as long", and gemma4
calls the untiled variant.

The router is 17.5 ms of the prefill (`20260929T083000`). Before that is worth
changing, the two must agree exactly: the logits feed top-k selection, and a
changed logit can change which experts a token routes to, which is not a
rounding difference.
"""

from __future__ import annotations

import sys

import numpy as np


def check(tokens: int, hidden_size: int, num_rows: int) -> bool:
    from hipengine.core import memory as mem
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.moe import router as r

    runtime = get_hip_runtime()
    rng = np.random.default_rng(20260929)

    # BF16 hidden via float32 -> bf16 bit pattern, F32 weights.
    hidden = rng.standard_normal((tokens, hidden_size)).astype(np.float32)
    hidden_bf16 = (hidden.view(np.uint32) >> 16).astype(np.uint16)
    weight = (rng.standard_normal((num_rows, hidden_size)) * 0.02).astype(np.float32)

    hbuf = mem.malloc(hidden_bf16.nbytes)
    wbuf = mem.malloc(weight.nbytes)
    mem.copy_host_to_device(hbuf, mem.host_array_ptr(np.ascontiguousarray(hidden_bf16)))
    mem.copy_host_to_device(wbuf, mem.host_array_ptr(np.ascontiguousarray(weight)))

    def run(fn, threads: int) -> np.ndarray:
        lbuf = mem.malloc(tokens * num_rows * 4)
        fn(hbuf.ptr, wbuf.ptr, lbuf.ptr, tokens, hidden_size, num_rows,
           threads=threads, runtime=runtime)
        runtime.device_synchronize()
        out = np.zeros(tokens * num_rows, dtype=np.float32)
        mem.copy_device_to_host(mem.host_array_ptr(out), lbuf)
        return out

    plain = run(r.qwen35_router_logits_bf16_f32w, 512)
    ok = True
    for label, fn, threads in (
        ("token_tile_8", r.qwen35_router_logits_bf16_f32w_token_tile_8, 512),
        ("token_tile_16", r.qwen35_router_logits_bf16_f32w_token_tile_16, 256),
    ):
        tiled = run(fn, threads)
        if np.array_equal(plain, tiled):
            continue
        ok = False
        diff = np.flatnonzero(plain != tiled)
        print(f"    {label}: {diff.size} differing of {plain.size}; max abs delta "
              f"{np.max(np.abs(plain - tiled)):.3e}")
    return ok


def time_kernels(tokens: int, hidden_size: int, num_rows: int) -> None:
    """Time the untiled and tile-8 kernels in one session on the same inputs.

    The correctness check above needs the inputs identical, so this repeats the
    same shape rather than sharing its buffers. It exists because the whole
    justification for the tiling is that it reads each weight row once per eight
    tokens instead of once per token, and that is a timing claim.
    """

    from hipengine.core import memory as mem
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.moe import router as r

    runtime = get_hip_runtime()
    rng = np.random.default_rng(20260929)
    hidden = rng.standard_normal((tokens, hidden_size)).astype(np.float32)
    hidden_bf16 = (hidden.view(np.uint32) >> 16).astype(np.uint16)
    weight = (rng.standard_normal((num_rows, hidden_size)) * 0.02).astype(np.float32)

    hbuf = mem.malloc(hidden_bf16.nbytes)
    wbuf = mem.malloc(weight.nbytes)
    mem.copy_host_to_device(hbuf, mem.host_array_ptr(np.ascontiguousarray(hidden_bf16)))
    mem.copy_host_to_device(wbuf, mem.host_array_ptr(np.ascontiguousarray(weight)))
    lbuf = mem.malloc(tokens * num_rows * 4)

    def bench(fn, threads: int, iters: int = 20) -> float:
        for _ in range(3):
            fn(hbuf.ptr, wbuf.ptr, lbuf.ptr, tokens, hidden_size, num_rows,
               threads=threads, runtime=runtime)
        runtime.device_synchronize()
        start = runtime.event_create()
        stop = runtime.event_create()
        runtime.event_record(start)
        for _ in range(iters):
            fn(hbuf.ptr, wbuf.ptr, lbuf.ptr, tokens, hidden_size, num_rows,
               threads=threads, runtime=runtime)
        runtime.event_record(stop)
        runtime.device_synchronize()
        return runtime.event_elapsed_time_ms(start, stop) / iters

    untiled = bench(r.qwen35_router_logits_bf16_f32w, 512)
    tile8 = bench(r.qwen35_router_logits_bf16_f32w_token_tile_8, 512)
    print(f"\n  timing, tokens={tokens} hidden={hidden_size} rows={num_rows}:")
    print(f"    untiled      {untiled:8.4f} ms")
    print(f"    token_tile_8 {tile8:8.4f} ms   {untiled / tile8:.3f}x")


def main() -> int:
    # Sweep rather than one point: the prefill shape, a decode shape, a size not
    # a multiple of the 8-token tile, and a larger case.
    cases = [
        (1, 2816, 128),      # decode
        (512, 2816, 128),    # the prefill shape
        (777, 2816, 128),    # not a multiple of 8
        (4096, 2816, 128),   # large
        (512, 1024, 64),     # unrelated geometry
    ]
    bad = 0
    for tokens, hidden, rows in cases:
        ok = check(tokens, hidden, rows)
        print(f"  tokens={tokens:<5d} hidden={hidden:<5d} rows={rows:<4d} "
              f"{'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
        bad += 0 if ok else 1
    print("VERDICT:", "ALL IDENTICAL" if not bad else f"{bad} DIFFERENT")
    time_kernels(512, 2816, 128)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
