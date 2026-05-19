"""GPU smoke for the tree-aware prefill GQA gate kernel.

Runs ``qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans`` (chain) and
``qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans`` (tree) with the
same Q/K/V/gate tensors but different ancestor masks:

  * ``chain_mask`` (lower-triangular):  ``ancestor[i, j] = 1`` iff ``j <= i``.
    With this mask the tree kernel MUST produce bit-equal output to the chain
    kernel because the chain kernel's ``row_positions`` causal limit is
    equivalent to the lower-triangular ancestor mask.

  * ``branching tree mask`` (depth-2 binary tree, 7 nodes): sibling rows
    should NOT attend to each other.  The tree output must differ from the
    chain output on the affected rows, and finite for every (row, dim) cell.

Run with::

  HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
    python3 scripts/dflash_tree_attn_kernel_smoke.py
"""

from __future__ import annotations

import argparse
import ctypes
import sys

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.attention import (
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans,
    qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    build_qwen35_paged_attn_decode,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.core.hip import HipRuntime, HipMemcpyKind, get_hip_runtime


def _fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    arr = x.astype(np.float32, copy=False)
    bits = arr.view(np.uint32)
    bf16 = (bits >> 16).astype(np.uint16)
    return bf16


def _bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    expanded = bits.astype(np.uint32) << 16
    return expanded.view(np.float32).reshape(bits.shape)


def _fp32_to_fp16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16, copy=False).view(np.uint16)


def _fp16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    return bits.view(np.float16).astype(np.float32)


def _alloc(runtime: HipRuntime, host: np.ndarray) -> tuple[int, np.ndarray]:
    nbytes = host.nbytes
    ptr = runtime.malloc(nbytes)
    runtime.memcpy(ptr, host.ctypes.data, nbytes, HipMemcpyKind.HOST_TO_DEVICE)
    return ptr, host


def _alloc_zero(runtime: HipRuntime, nbytes: int) -> int:
    ptr = runtime.malloc(nbytes)
    runtime.memset(ptr, 0, nbytes)
    return ptr


def _build_lower_triangular_mask(rows: int) -> np.ndarray:
    mask = np.zeros((rows, rows), dtype=np.uint8)
    for i in range(rows):
        mask[i, : i + 1] = 1
    return mask


def _build_binary_tree_mask(depth: int, branching: int) -> tuple[np.ndarray, list[int]]:
    """Return (ancestor_mask, parent_rows) for a depth-``depth`` ``branching``-ary tree.

    Root is row 0; each non-leaf has ``branching`` children appended in BFS order.
    Total nodes = (branching**(depth+1) - 1) / (branching - 1) for branching != 1.
    """
    parent_rows: list[int] = [-1]
    levels: list[list[int]] = [[0]]
    next_id = 1
    for _level in range(depth):
        new_level: list[int] = []
        for parent in levels[-1]:
            for _branch in range(branching):
                parent_rows.append(parent)
                new_level.append(next_id)
                next_id += 1
        levels.append(new_level)
    rows = len(parent_rows)
    mask = np.zeros((rows, rows), dtype=np.uint8)
    for i in range(rows):
        cursor = i
        while cursor >= 0:
            mask[i, cursor] = 1
            cursor = parent_rows[cursor] if cursor != 0 else -1
    return mask, parent_rows


def _run_chain(
    runtime: HipRuntime,
    *,
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    library: ctypes.CDLL,
) -> None:
    qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        gate_ptr,
        out_ptr,
        spans,
        rows,
        max_context_len,
        256,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_q_heads * head_dim,
        1,
        scale,
        library=library,
        runtime=runtime,
    )


def _run_tree(
    runtime: HipRuntime,
    *,
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    gate_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    rows: int,
    max_context_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    ancestor_mask_ptr: int,
    tree_committed_count: int,
    library: ctypes.CDLL,
) -> None:
    qwen35_paged_full_attn_prefill_gqa_gate_tree_fp16_spans(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        gate_ptr,
        out_ptr,
        spans,
        ancestor_mask_ptr,
        tree_committed_count,
        rows,
        max_context_len,
        256,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_q_heads * head_dim,
        1,
        scale,
        library=library,
        runtime=runtime,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-q-heads", type=int, default=4)
    ap.add_argument("--num-kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--committed-context", type=int, default=24)
    ap.add_argument("--rows", type=int, default=5, help="Chain length / verifier rows for chain test")
    ap.add_argument("--tree-depth", type=int, default=2)
    ap.add_argument("--tree-branching", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    num_q_heads = args.num_q_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    committed = args.committed_context

    chain_rows = args.rows
    tree_mask, tree_parents = _build_binary_tree_mask(args.tree_depth, args.tree_branching)
    tree_rows = tree_mask.shape[0]

    total_verifier_rows = max(chain_rows, tree_rows)
    total_positions = committed + total_verifier_rows
    block_size = 256
    block_table_len = (total_positions + block_size - 1) // block_size

    runtime = get_hip_runtime()
    library = build_qwen35_paged_attn_decode(load=True)
    device = Device("hip", 0)

    # Random Q (fp32), K/V (bf16 bits), gate (fp16 bits)
    query_host = rng.standard_normal((total_verifier_rows, num_q_heads, head_dim), dtype=np.float32) * 0.1
    key_cache_host_f32 = rng.standard_normal((block_table_len, block_size, num_kv_heads, head_dim), dtype=np.float32) * 0.1
    value_cache_host_f32 = rng.standard_normal((block_table_len, block_size, num_kv_heads, head_dim), dtype=np.float32) * 0.1
    key_cache_host = _fp32_to_bf16_bits(key_cache_host_f32)
    value_cache_host = _fp32_to_bf16_bits(value_cache_host_f32)
    gate_host_f32 = rng.standard_normal((total_verifier_rows, num_q_heads, head_dim), dtype=np.float32)
    gate_host = _fp32_to_fp16_bits(gate_host_f32)
    block_table_host = np.tile(
        np.arange(block_table_len, dtype=np.int32), (total_verifier_rows, 1)
    ).reshape(total_verifier_rows, block_table_len)

    query_ptr, _ = _alloc(runtime, query_host)
    key_cache_ptr, _ = _alloc(runtime, key_cache_host)
    value_cache_ptr, _ = _alloc(runtime, value_cache_host)
    gate_ptr, _ = _alloc(runtime, gate_host)
    block_table_ptr, _ = _alloc(runtime, block_table_host)

    # ---- Chain test ----
    chain_positions = np.arange(committed, committed + chain_rows, dtype=np.int64)
    chain_context_counts = chain_positions + 1  # post-append context length per row
    chain_position_ptr, _ = _alloc(runtime, chain_positions)
    chain_context_ptr, _ = _alloc(runtime, chain_context_counts)

    chain_block_table_tensor = Tensor.from_handle(block_table_ptr, (chain_rows, block_table_len), DType.INT32, device)
    chain_live_tensor = Tensor.from_handle(chain_context_ptr, (chain_rows,), DType.INT64, device)
    chain_row_positions = Tensor.from_handle(chain_position_ptr, (chain_rows,), DType.INT64, device)
    chain_spans = KVLiveSpans.paged_uniform(
        block_table=chain_block_table_tensor,
        live_counts=chain_live_tensor,
        max_live_count=int(chain_context_counts.max()),
        storage_dtype=DType.BF16,
        row_positions=chain_row_positions,
        span_role="verify_chain",
    )

    chain_out_nbytes = chain_rows * num_q_heads * head_dim * 2  # fp16
    chain_out_chain_ptr = _alloc_zero(runtime, chain_out_nbytes)
    chain_out_tree_ptr = _alloc_zero(runtime, chain_out_nbytes)

    chain_lower_mask = _build_lower_triangular_mask(chain_rows)
    chain_ancestor_ptr, _ = _alloc(runtime, chain_lower_mask)

    scale = head_dim ** -0.5

    _run_chain(
        runtime,
        query_ptr=query_ptr,
        key_cache_ptr=key_cache_ptr,
        value_cache_ptr=value_cache_ptr,
        gate_ptr=gate_ptr,
        out_ptr=chain_out_chain_ptr,
        spans=chain_spans,
        rows=chain_rows,
        max_context_len=int(chain_context_counts.max()),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        library=library,
    )
    _run_tree(
        runtime,
        query_ptr=query_ptr,
        key_cache_ptr=key_cache_ptr,
        value_cache_ptr=value_cache_ptr,
        gate_ptr=gate_ptr,
        out_ptr=chain_out_tree_ptr,
        spans=chain_spans,
        rows=chain_rows,
        max_context_len=int(chain_context_counts.max()),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        ancestor_mask_ptr=chain_ancestor_ptr,
        tree_committed_count=committed,
        library=library,
    )
    runtime.device_synchronize()

    chain_chain_out_host = np.empty(chain_rows * num_q_heads * head_dim, dtype=np.uint16)
    chain_tree_out_host = np.empty(chain_rows * num_q_heads * head_dim, dtype=np.uint16)
    runtime.memcpy(chain_chain_out_host.ctypes.data, chain_out_chain_ptr, chain_out_nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    runtime.memcpy(chain_tree_out_host.ctypes.data, chain_out_tree_ptr, chain_out_nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    chain_chain_fp32 = _fp16_bits_to_fp32(chain_chain_out_host)
    chain_tree_fp32 = _fp16_bits_to_fp32(chain_tree_out_host)

    bit_equal = np.array_equal(chain_chain_out_host, chain_tree_out_host)
    max_abs_diff = float(np.max(np.abs(chain_chain_fp32 - chain_tree_fp32)))
    finite_chain = bool(np.isfinite(chain_chain_fp32).all())
    finite_tree = bool(np.isfinite(chain_tree_fp32).all())

    print(
        f"[chain reduction] rows={chain_rows} committed={committed} num_q_heads={num_q_heads} "
        f"num_kv_heads={num_kv_heads} head_dim={head_dim}: bit_equal={bit_equal} "
        f"max_abs_diff={max_abs_diff:.3e} finite_chain={finite_chain} finite_tree={finite_tree}"
    )

    chain_ok = bit_equal and finite_chain and finite_tree
    if not chain_ok:
        print("  FAIL: chain-shaped tree mask did not reproduce chain output bit-for-bit", file=sys.stderr)
        return 1

    # ---- Branching tree test ----
    print(
        f"[branching tree] depth={args.tree_depth} branching={args.tree_branching} rows={tree_rows} "
        f"parents={tree_parents}"
    )
    tree_positions = np.arange(committed, committed + tree_rows, dtype=np.int64)
    tree_context_counts = tree_positions + 1
    tree_position_ptr, _ = _alloc(runtime, tree_positions)
    tree_context_ptr, _ = _alloc(runtime, tree_context_counts)

    tree_block_table_tensor = Tensor.from_handle(block_table_ptr, (tree_rows, block_table_len), DType.INT32, device)
    tree_live_tensor = Tensor.from_handle(tree_context_ptr, (tree_rows,), DType.INT64, device)
    tree_row_positions = Tensor.from_handle(tree_position_ptr, (tree_rows,), DType.INT64, device)
    tree_spans = KVLiveSpans.paged_uniform(
        block_table=tree_block_table_tensor,
        live_counts=tree_live_tensor,
        max_live_count=int(tree_context_counts.max()),
        storage_dtype=DType.BF16,
        row_positions=tree_row_positions,
        span_role="verify_tree",
    )
    tree_out_nbytes = tree_rows * num_q_heads * head_dim * 2
    tree_out_chain_ptr = _alloc_zero(runtime, tree_out_nbytes)
    tree_out_tree_ptr = _alloc_zero(runtime, tree_out_nbytes)

    tree_ancestor_ptr, _ = _alloc(runtime, tree_mask)

    _run_chain(
        runtime,
        query_ptr=query_ptr,
        key_cache_ptr=key_cache_ptr,
        value_cache_ptr=value_cache_ptr,
        gate_ptr=gate_ptr,
        out_ptr=tree_out_chain_ptr,
        spans=tree_spans,
        rows=tree_rows,
        max_context_len=int(tree_context_counts.max()),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        library=library,
    )
    _run_tree(
        runtime,
        query_ptr=query_ptr,
        key_cache_ptr=key_cache_ptr,
        value_cache_ptr=value_cache_ptr,
        gate_ptr=gate_ptr,
        out_ptr=tree_out_tree_ptr,
        spans=tree_spans,
        rows=tree_rows,
        max_context_len=int(tree_context_counts.max()),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        ancestor_mask_ptr=tree_ancestor_ptr,
        tree_committed_count=committed,
        library=library,
    )
    runtime.device_synchronize()

    tree_chain_host = np.empty(tree_rows * num_q_heads * head_dim, dtype=np.uint16)
    tree_tree_host = np.empty(tree_rows * num_q_heads * head_dim, dtype=np.uint16)
    runtime.memcpy(tree_chain_host.ctypes.data, tree_out_chain_ptr, tree_out_nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    runtime.memcpy(tree_tree_host.ctypes.data, tree_out_tree_ptr, tree_out_nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    tree_chain_fp32 = _fp16_bits_to_fp32(tree_chain_host).reshape(tree_rows, num_q_heads * head_dim)
    tree_tree_fp32 = _fp16_bits_to_fp32(tree_tree_host).reshape(tree_rows, num_q_heads * head_dim)

    finite_tree_chain = bool(np.isfinite(tree_chain_fp32).all())
    finite_tree_tree = bool(np.isfinite(tree_tree_fp32).all())
    per_row_max_diff = np.max(np.abs(tree_chain_fp32 - tree_tree_fp32), axis=1)
    differing_rows = [int(i) for i, diff in enumerate(per_row_max_diff) if diff > 0]
    same_rows = [int(i) for i, diff in enumerate(per_row_max_diff) if diff == 0]
    expected_differ = [i for i in range(tree_rows) if any(
        # row i differs from chain if at least one row j in [0, i) is NOT an ancestor of i
        tree_mask[i, j] == 0 for j in range(i)
    )]
    expected_same = [i for i in range(tree_rows) if i not in expected_differ]

    print(
        f"  per-row max abs diff vs chain kernel: {per_row_max_diff.tolist()}\n"
        f"  rows that differ (mask filtered something): {differing_rows}\n"
        f"  rows that match chain (mask is full lower-triangular for that row): {same_rows}\n"
        f"  expected differ:  {expected_differ}\n"
        f"  expected same:    {expected_same}\n"
        f"  finite (chain kernel output): {finite_tree_chain}\n"
        f"  finite (tree  kernel output): {finite_tree_tree}"
    )

    tree_ok = (
        finite_tree_chain
        and finite_tree_tree
        and set(differing_rows) == set(expected_differ)
        and set(same_rows) == set(expected_same)
    )
    if not tree_ok:
        print("  FAIL: tree mask differs from chain output in unexpected rows", file=sys.stderr)
        return 2

    print("[OK] tree-aware GQA gate kernel: chain reduction is bit-equal AND branching mask filters siblings correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
