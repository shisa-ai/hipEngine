"""Do the serial and parallel MoE compactors produce identical output?

`qwen35_moe_group_compact_active` launches one block of 256 threads
(`dim3(1)`) to compact every lane; `..._parallel` launches one block per
expert (`dim3(num_experts)`). The expert forward uses the serial one and
pays 12.35 ms per prefill for it (`20260929T103000`).

Before that is worth changing, the two must agree. The parallel kernel is
described in-tree as *stable* compaction, which should mean it preserves
the serial lane order exactly -- in which case the swap is bit-exact and
needs no arithmetic gate. This checks that rather than assuming it.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np


def main() -> int:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core import memory as mem
    from hipengine.kernels.hip_gfx1100.moe import group_scatter as gs

    # Sweep rather than test one point: the prefill shape (512 tokens x top_k 8),
    # a decode shape (1 token), a size unrelated to either, and the largest case.
    # AGENTS.md's 'validate the space, not the gate point'.
    cases = [(1, 8), (512, 8), (4096, 8), (777, 4)]
    failures = 0
    for tokens, top_k in cases:
        if check_case(tokens, top_k):
            print(f"  tokens={tokens:<5d} top_k={top_k}  IDENTICAL")
        else:
            failures += 1
            print(f"  tokens={tokens:<5d} top_k={top_k}  DIFFERENT")
    print("VERDICT:", "ALL IDENTICAL" if not failures else f"{failures} DIFFERENT")
    return 0 if not failures else 1


def check_case(tokens: int, top_k: int) -> bool:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core import memory as mem
    from hipengine.kernels.hip_gfx1100.moe import group_scatter as gs

    runtime = get_hip_runtime()
    library = gs.build_qwen35_moe_group_scatter(load=True)

    rng = np.random.default_rng(20260929)
    num_experts = 128
    lanes = tokens * top_k

    selected = rng.integers(0, num_experts, size=(tokens, top_k)).astype(np.int64).reshape(-1)
    weights = rng.random(size=(tokens, top_k)).astype(np.float32).reshape(-1)
    total = int(selected.size)

    def dev(arr):
        buf = mem.malloc(arr.nbytes)
        mem.copy_host_to_device(buf, mem.host_array_ptr(np.ascontiguousarray(arr)))
        return buf

    sel_d = dev(selected)
    w_d = dev(weights)
    counts_d = mem.malloc(num_experts * 8)
    start_d = mem.malloc((num_experts + 1) * 8)
    active_d = mem.malloc(num_experts * 8)
    active_count_d = mem.malloc(8)

    def zero(buf):
        ctypes.memset(ctypes.c_void_p(buf.ptr), 0, buf.nbytes)

    for b in (counts_d, start_d, active_d, active_count_d):
        zero(b)

    gs.qwen35_moe_group_count(sel_d.ptr, counts_d.ptr, total, num_experts)
    gs.qwen35_moe_group_prefix_active(
        counts_d.ptr, start_d.ptr, active_d.ptr, active_count_d.ptr, num_experts
    )
    runtime.device_synchronize()

    def run(parallel: bool):
        sl = mem.malloc(total * 8)
        se = mem.malloc(total * 8)
        sw = mem.malloc(total * 4)
        for b in (sl, se, sw):
            zero(b)
        gs.qwen35_moe_group_compact_active(
            sel_d.ptr, w_d.ptr, start_d.ptr, active_d.ptr, active_count_d.ptr,
            sl.ptr, se.ptr, sw.ptr, total, num_experts,
            parallel=parallel, library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        out = {}
        for name, buf, dt in (("sorted_lanes", sl, np.int64),
                              ("sorted_experts", se, np.int64),
                              ("sorted_weights", sw, np.float32)):
            host = np.zeros(total, dtype=dt)
            mem.copy_device_to_host(mem.host_array_ptr(host), buf)
            out[name] = host
        return out

    serial = run(False)
    parallel = run(True)
    ok = True
    for name in ("sorted_lanes", "sorted_experts", "sorted_weights"):
        if not np.array_equal(serial[name], parallel[name]):
            ok = False
            diff = np.flatnonzero(serial[name] != parallel[name])
            print(f"    {name}: {diff.size} differing, first at {diff[:6]}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
