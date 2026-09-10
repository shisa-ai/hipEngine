"""PF-4 RED tests: M2 parallel top-10 routing-map compaction (prefill arm).

Declaration entry:
``worklog/entries/20260903T212627.415167Z-lhl-pf-4-m2-routing-compaction-declaration-4d93f0.md``

- **Arithmetic class:** T0. The candidate fuses the incumbent
  memset + ``group_count`` + ``group_prefix`` + memset + map-scatter chain
  into one per-expert-block kernel (halo-box ``a7ad7b7f`` ``mmid.cu``
  ``mm_ids_helper_top10_parallel`` mechanism: 8 warps/block, warp-local
  any-scan + ``shfl_up`` exclusive scan over 2 tokens x 16 padded top-k slots,
  cross-warp combine through shared per-warp counts/prefixes). All outputs are
  integer maps plus copied float weights; no float arithmetic is reordered or
  changed, so the binding contract is exact ``expert_start`` equality, a valid
  per-expert bijection over (token, slot) lanes with matching weight content,
  and identical gathered-row content. Compact-row ORDER within one expert is
  explicitly NOT the contract: the incumbent assigns it with atomics
  (``group_scatter_offsets``) and is therefore not a stable order oracle.
- **Registered strict fallback (named per loop contract):** the incumbent
  chain stays registered, production, and untouched:
  ``qwen35_moe_group_count`` + ``qwen35_moe_group_prefix`` +
  ``qwen35_moe_group_scatter_gather_lowp`` (module
  ``hipengine/kernels/hip_gfx1100/moe/group_scatter.py``), plus the
  gather-only ``qwen35_moe_gather_packed_hidden_lowp`` reused downstream.

RED semantics (unmodified path): the candidate function
``qwen35_moe_group_map_top10_parallel`` does not exist yet, so every test
below fails with ``ImportError`` before any GPU work runs. The candidate must
land under the NEW four-axis variant key
``KernelKey("hip_gfx1100", "moe_group_map", "generic", "top10_parallel_i64")``
with the incumbent chain still resolving as production.

End-to-end runner parity is deliberately NOT in this file: the grouped
prefill kernels consume (expert_start, packed_hidden, sorted maps) and every
compact row is computed independently, so chain-level parity below plus the
campaign whole-model A/B covers the end-to-end bit-exactness gate.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)

_NUM_EXPERTS = 512
_TOP_K = 10
_HIDDEN = 2560

# Incumbent chain keys that must stay bound to the untouched owners.
_INCUMBENT_COUNT_KEY = ("hip_gfx1100", "moe_group_count", "generic", "selected_experts")
_INCUMBENT_PREFIX_KEY = ("hip_gfx1100", "moe_group_prefix", "generic", "active_experts")
_INCUMBENT_SCATTER_KEY = ("hip_gfx1100", "moe_group_scatter_gather", "w4_paro", "qwen35_lowp")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _routed_lanes(rows: int, seed: int) -> np.ndarray:
    """Deterministic routed expert ids [rows * top_k], uneven with empty experts.

    Experts are sampled WITHOUT replacement within each token, matching the
    production top-k invariant (distinct experts per token) that the M2
    mechanism relies on.
    """
    rng = np.random.default_rng(seed)
    # Skew the distribution so many experts get zero lanes and active experts
    # get uneven counts (binding shape: ~330/512 active at ~7-10 lanes each).
    weights = rng.uniform(0.0, 1.0, size=_NUM_EXPERTS) ** 4
    p = weights / weights.sum()
    selected = np.stack(
        [rng.choice(_NUM_EXPERTS, size=_TOP_K, replace=False, p=p)
         for _ in range(rows)]
    ).astype(np.int64).reshape(rows * _TOP_K)
    return selected


def _routing_weights(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    weights = rng.uniform(0.001, 1.0, size=rows * _TOP_K).astype(np.float32)
    return weights


def test_pf4_m2_variant_key_registry_contract() -> None:
    """The M2 candidate key must resolve to the candidate function.

    RED on the unmodified path: the candidate does not exist, so the import
    fails before any GPU work runs. The incumbent keys must stay bound to the
    untouched owners.
    """
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_group_count,
        qwen35_moe_group_map_top10_parallel,
        qwen35_moe_group_prefix_active,
        qwen35_moe_group_scatter_gather_lowp,
        register_qwen35_moe_group_scatter_kernels,
    )
    from hipengine.kernels.registry import KernelKey, resolve

    register_qwen35_moe_group_scatter_kernels()

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_group_map",
            quant="generic",
            variant="top10_parallel_i64",
            missing="none",
        )
        is qwen35_moe_group_map_top10_parallel
    )
    assert (
        resolve(backend=_INCUMBENT_COUNT_KEY[0], layer=_INCUMBENT_COUNT_KEY[1],
                quant=_INCUMBENT_COUNT_KEY[2], variant=_INCUMBENT_COUNT_KEY[3], missing="none")
        is qwen35_moe_group_count
    )
    assert (
        resolve(backend=_INCUMBENT_PREFIX_KEY[0], layer=_INCUMBENT_PREFIX_KEY[1],
                quant=_INCUMBENT_PREFIX_KEY[2], variant=_INCUMBENT_PREFIX_KEY[3], missing="none")
        is qwen35_moe_group_prefix_active
    )
    assert (
        resolve(backend=_INCUMBENT_SCATTER_KEY[0], layer=_INCUMBENT_SCATTER_KEY[1],
                quant=_INCUMBENT_SCATTER_KEY[2], variant=_INCUMBENT_SCATTER_KEY[3], missing="none")
        is qwen35_moe_group_scatter_gather_lowp
    )


def _incumbent_chain(selected, routing, rows, runtime, allocations):
    """Run the incumbent memset+count+prefix+memset+scatter_gather chain."""
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_group_count,
        qwen35_moe_group_prefix,
        qwen35_moe_group_scatter_gather_lowp,
    )

    compact = rows * _TOP_K
    counts = np.zeros(_NUM_EXPERTS, dtype=np.int32)
    padded_counts = np.zeros(_NUM_EXPERTS, dtype=np.int32)
    expert_start = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
    total_padded = np.zeros(1, dtype=np.int64)
    scatter_offsets = np.zeros(_NUM_EXPERTS, dtype=np.int32)
    sorted_lanes = np.zeros(compact, dtype=np.int64)
    sorted_experts = np.zeros(compact, dtype=np.int64)
    sorted_weights = np.zeros(compact, dtype=np.float32)
    hidden = (np.arange(rows * _HIDDEN, dtype=np.uint16) * 7 + 13).astype(np.uint16)
    packed = np.zeros((compact, _HIDDEN), dtype=np.uint16)

    d_counts = _upload(counts, runtime, allocations)
    d_padded = _upload(padded_counts, runtime, allocations)
    d_expert_start = _upload(expert_start, runtime, allocations)
    d_total_padded = _upload(total_padded, runtime, allocations)
    d_offsets = _upload(scatter_offsets, runtime, allocations)
    d_sorted_lanes = _upload(sorted_lanes, runtime, allocations)
    d_sorted_experts = _upload(sorted_experts, runtime, allocations)
    d_sorted_weights = _upload(sorted_weights, runtime, allocations)
    d_hidden = _upload(hidden, runtime, allocations)
    d_packed = _upload(packed.reshape(-1), runtime, allocations)
    d_selected = _upload(selected, runtime, allocations)
    d_routing = _upload(routing, runtime, allocations)

    # Production chain includes both memsets (qwen4_exp_runner.py
    # grouped_prefill block); they are no-ops on fresh zero host uploads but
    # keep the fixture faithful for reuse.
    runtime.memset(d_counts.ptr, 0, d_counts.nbytes)
    qwen35_moe_group_count(
        d_selected.ptr, d_counts.ptr, compact, _NUM_EXPERTS,
        runtime=runtime,
    )
    qwen35_moe_group_prefix(
        d_counts.ptr, d_padded.ptr, d_expert_start.ptr, d_total_padded.ptr,
        _NUM_EXPERTS, 1, runtime=runtime,
    )
    runtime.memset(d_offsets.ptr, 0, d_offsets.nbytes)
    qwen35_moe_group_scatter_gather_lowp(
        d_hidden.ptr, d_selected.ptr, d_routing.ptr, d_expert_start.ptr,
        d_offsets.ptr, d_sorted_lanes.ptr, d_sorted_experts.ptr,
        d_sorted_weights.ptr, d_packed.ptr, compact, _NUM_EXPERTS, _TOP_K,
        _HIDDEN, runtime=runtime,
    )
    runtime.device_synchronize()
    return {
        "expert_start": _download(d_expert_start, (_NUM_EXPERTS + 1,), np.int64, runtime),
        "sorted_lanes": _download(d_sorted_lanes, (compact,), np.int64, runtime),
        "sorted_experts": _download(d_sorted_experts, (compact,), np.int64, runtime),
        "sorted_weights": _download(d_sorted_weights, (compact,), np.float32, runtime),
        "packed": _download(d_packed, (compact * _HIDDEN,), np.uint16, runtime),
        "hidden": hidden,
        "d_expert_start": d_expert_start,
        "d_selected": d_selected,
        "d_routing": d_routing,
        "d_hidden": d_hidden,
    }


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 64, 512])
def test_pf4_m2_group_map_expert_start_parity(rows: int) -> None:
    """Candidate expert_start must be int64-identical to the incumbent prefix.

    RED on the unmodified path: the candidate function does not exist, so the
    import fails before any GPU work runs.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_group_map_top10_parallel,
    )

    runtime = get_hip_runtime()
    selected = _routed_lanes(rows, seed=6100 + rows)
    routing = _routing_weights(rows, seed=6200 + rows)
    compact = rows * _TOP_K
    allocations = []
    try:
        incumbent = _incumbent_chain(selected, routing, rows, runtime, allocations)
        expert_start = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
        sorted_lanes = np.zeros(compact, dtype=np.int64)
        sorted_experts = np.zeros(compact, dtype=np.int64)
        sorted_weights = np.zeros(compact, dtype=np.float32)
        d_es = _upload(expert_start, runtime, allocations)
        d_sl = _upload(sorted_lanes, runtime, allocations)
        d_se = _upload(sorted_experts, runtime, allocations)
        d_sw = _upload(sorted_weights, runtime, allocations)
        qwen35_moe_group_map_top10_parallel(
            incumbent["d_selected"].ptr, incumbent["d_routing"].ptr,
            d_es.ptr, d_sl.ptr, d_se.ptr, d_sw.ptr,
            compact, _NUM_EXPERTS, _TOP_K, runtime=runtime,
        )
        runtime.device_synchronize()
        candidate_es = _download(d_es, (_NUM_EXPERTS + 1,), np.int64, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    # T0 contract: identical boundaries (deterministic prefix).
    np.testing.assert_array_equal(candidate_es, incumbent["expert_start"])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", [16, 64, 512])
def test_pf4_m2_group_map_bijection_and_content(rows: int) -> None:
    """Candidate maps must form the same per-expert bijections with matching
    weight content, and the gathered rows must equal the hidden rows named by
    the maps.

    Compact-row order within one expert is intentionally not compared: the
    incumbent assigns it with atomics and is not a stable order oracle.
    """
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.moe.group_scatter import (
        qwen35_moe_gather_packed_hidden_lowp,
        qwen35_moe_group_map_top10_parallel,
    )

    runtime = get_hip_runtime()
    selected = _routed_lanes(rows, seed=6300 + rows)
    routing = _routing_weights(rows, seed=6400 + rows)
    compact = rows * _TOP_K
    allocations = []
    try:
        incumbent = _incumbent_chain(selected, routing, rows, runtime, allocations)
        expert_start = np.zeros(_NUM_EXPERTS + 1, dtype=np.int64)
        sorted_lanes = np.zeros(compact, dtype=np.int64)
        sorted_experts = np.zeros(compact, dtype=np.int64)
        sorted_weights = np.zeros(compact, dtype=np.float32)
        d_es = _upload(expert_start, runtime, allocations)
        d_sl = _upload(sorted_lanes, runtime, allocations)
        d_se = _upload(sorted_experts, runtime, allocations)
        d_sw = _upload(sorted_weights, runtime, allocations)
        packed = np.zeros(compact * _HIDDEN, dtype=np.uint16)
        d_packed = _upload(packed, runtime, allocations)
        qwen35_moe_group_map_top10_parallel(
            incumbent["d_selected"].ptr, incumbent["d_routing"].ptr,
            d_es.ptr, d_sl.ptr, d_se.ptr, d_sw.ptr,
            compact, _NUM_EXPERTS, _TOP_K, runtime=runtime,
        )
        qwen35_moe_gather_packed_hidden_lowp(
            incumbent["d_hidden"].ptr, d_sl.ptr, d_packed.ptr,
            compact * _HIDDEN, rows, _TOP_K, _HIDDEN, runtime=runtime,
        )
        runtime.device_synchronize()
        cand_es = _download(d_es, (_NUM_EXPERTS + 1,), np.int64, runtime)
        cand_sl = _download(d_sl, (compact,), np.int64, runtime)
        cand_se = _download(d_se, (compact,), np.int64, runtime)
        cand_sw = _download(d_sw, (compact,), np.float32, runtime)
        cand_packed = _download(d_packed, (compact * _HIDDEN,), np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    # Every valid routed lane appears exactly once across the maps.
    np.testing.assert_array_equal(np.sort(cand_sl), np.arange(compact, dtype=np.int64))
    # Per-expert ranges: expert tags and boundaries agree with selected.
    for expert in range(_NUM_EXPERTS):
        lo, hi = int(cand_es[expert]), int(cand_es[expert + 1])
        lanes = cand_sl[lo:hi]
        assert np.all(cand_se[lo:hi] == expert)
        expected = np.sort(np.where(selected == expert)[0])
        np.testing.assert_array_equal(np.sort(lanes), expected)
        if lanes.size:
            np.testing.assert_array_equal(
                cand_sw[lo:hi], routing[lanes]
            )
    # Gathered row content: row r of the packed buffer equals the hidden row
    # of the token owning sorted_lanes[r] (content, not order).
    token_of_row = cand_sl // _TOP_K
    np.testing.assert_array_equal(
        cand_packed.reshape(compact, _HIDDEN),
        incumbent["hidden"].reshape(rows, _HIDDEN)[token_of_row],
    )
    # Same lane sets per expert as the incumbent (order-free comparison).
    for expert in range(_NUM_EXPERTS):
        lo, hi = int(cand_es[expert]), int(cand_es[expert + 1])
        inc_lo, inc_hi = int(incumbent["expert_start"][expert]), int(incumbent["expert_start"][expert + 1])
        np.testing.assert_array_equal(
            np.sort(cand_sl[lo:hi]), np.sort(incumbent["sorted_lanes"][inc_lo:inc_hi])
        )


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
