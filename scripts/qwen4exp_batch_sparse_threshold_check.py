#!/usr/bin/env python3
"""Sparse-threshold correctness check for the N-row Qwen4Exp batch runner.

The short-prompt smoke (``scripts/qwen4exp_batch_runner_real_smoke.py``) never
leaves the dense-equivalence region: every row it advances is far below the
index budget, so the per-row top-k selection that defines the sparse path is
never dispatched. This script exercises exactly that boundary on the real
checkpoint.

What it does:

1. **Long distinct prompts.** Each prompt is encoded and truncated to an exact
   token length, so the first continuation step lands on a chosen side of
   ``dense_equivalent_limit = block_budget * ratio + ratio - 1`` (2051 for the
   production artifact: 512 blocks x 4-token compression). The default ladder
   is 2050 (dense, then crosses during the continuation), 2052 (sparse at the
   first step), and 2100 (sparse, several blocks past the bound).
2. **c1 prefill, shared weights, batch import.** One weight residency, one c1
   reference per prompt, imported into a batch slot. The import is audited
   against the source: per-layer paged K/V, raw index keys, GDN conv and
   recurrent rows, PLE history, and the index cursors.
3. **Forced steps.** The teacher token at every step is the c1 reference's own
   greedy id, so both paths see the same token stream.
4. **Selection comparison, per layer and per row.** The c1 mixer and the batch
   mixed mixer are wrapped with host-side taps that read, immediately after the
   layer returns: the index query bytes, the pooled index keys, the block
   starts, the raw device scores, and the selected positions/count. Because both
   paths use the same ``qwen4_exp_qsa_score_f32`` and the same
   ``qwen4_exp_qsa_topk_expand_f32_i64`` kernel body (the rows entry point is
   the same kernel with a per-row position vector), identical inputs must give
   identical selections. Any difference is therefore separated into:
     - *input drift* (query / pooled key / score bytes differ), and
     - *policy drift* (the selection disagrees with the documented host rule in
       ``hipengine.kernels.cpu_reference.qwen4_exp.qsa_select_positions`` on its
       own score bytes).
5. **Ownership.** Each row's raw keys and pooled keys are read through that
   slot's own page table, so a slot that selected from another slot's keys
   cannot match its own reference. C2 repeats two prompts side by side and
   compares every row against the C1 result.
6. **Logits.** KL both directions, top-1, and max/mean absolute error per step.

This is a correctness diagnostic at depth, not a capacity probe, not a speed
measurement, and not a production-quality claim. It evaluates no calibrated
gate: the production envelope in ``docs/EXECUTION-PROFILES.md`` is measured
against the strict teacher on the full row suite, which this script does not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    malloc,
    free,
    memory_stats,
)
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.kernels.cpu_reference.qwen4_exp import qsa_select_positions
from hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa import (
    qwen4_exp_qsa_topk_expand_f32_i64,
)
from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
from hipengine.models import resolve_model
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_runner as c1_module
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from scripts.qwen4_exp_compare_logits import compare_logits

MIB = 1024**2

# Three distinct long sources. Each is tiled with a changing marker so the token
# stream has no long exact repeat, which keeps the index scores from collapsing
# into exact ties and makes a selection difference meaningful.
SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "code",
        "def reverse_list(head):\n    previous = None\n    while head is not None:\n"
        "        following = head.next\n        head.next = previous\n"
        "        previous = head\n        head = following\n    return previous\n\n",
        "The function walks the chain once, rewiring each link before advancing. "
        "Its running time is linear in the number of nodes and it allocates no "
        "auxiliary structure. A recursive variant reverses the tail first and "
        "then points the successor back at the caller, which costs stack depth "
        "equal to the list length. Callers that need the old head kept intact "
        "must copy the list before invoking either form, because both mutate the "
        "links in place. Edge cases are an empty list, a single node, and a "
        "cycle, and only the last one requires a separate detector. ",
    ),
    (
        "general_en",
        "Sunlight entering the atmosphere is scattered by molecules much smaller "
        "than its wavelength. That scattering grows sharply toward short "
        "wavelengths, so blue light is redirected across the whole sky while red "
        "light continues closer to a straight line. ",
        "An observer looking away from the sun therefore sees predominantly blue "
        "scattered light arriving from every direction. Near sunset the path "
        "through the atmosphere is long enough that most blue light has already "
        "been scattered out of the direct beam, which leaves the transmitted "
        "light reddened. Suspended particles and water droplets scatter more "
        "broadly and add a whitish haze, which is why humid air looks paler than "
        "dry air at the same elevation. ",
    ),
    (
        "general_ja",
        "富士山の標高は三千七百七十六メートルで、日本で最も高い山です。",
        "山頂は一年の大半を雪に覆われ、七月から九月にかけてのみ登山道が開きます。"
        "周辺には五つの湖があり、それぞれ異なる角度から山容を望むことができます。"
        "気温は標高とともに下がり、麓と山頂の差は夏でも十度を超えることがあります。"
        "登山者は水分と防寒具を十分に用意し、天候の変化に注意する必要があります。",
    ),
)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="GGUF model directory")
    parser.add_argument(
        "--lengths",
        default="2050,2052,2100",
        help="exact prompt token lengths, one prompt each (default: 2050,2052,2100)",
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--context", type=int, default=0, help="override max_positions")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-c2", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


# -- profile and mapping helpers --------------------------------------------

# c1's decode is graph-captured as a single-request optimization while the batch
# runner is eager. Captured and eager execution are different profiles, so a
# comparison that leaves capture on measures the profile difference too.
EAGER_PROFILE_ENV = {
    "HIPENGINE_QWEN4_EXP_MOE_GRAPH": "0",
    "HIPENGINE_QWEN4_EXP_LAYER_GRAPHS": "0",
}


def _apply_eager_profile(environ: Any | None = None) -> dict[str, str]:
    """Force the eager profile on both arms and return what was set."""

    target = os.environ if environ is None else environ
    target.update(EAGER_PROFILE_ENV)
    return dict(EAGER_PROFILE_ENV)


def _sparse_subset_row(sparse_rows: Sequence[int], row: int) -> int:
    """Position of mixer ``row`` inside the sparse group that was scored.

    ``select_active`` scores only the sparse rows, in that group's ascending
    order, so its score slab is indexed by subset position. A mixed dense/sparse
    step therefore needs this mapping: for rows ``(1, 3)`` the sparse row 3 owns
    score row 1.
    """

    return tuple(int(value) for value in sparse_rows).index(int(row))


def _selection_baseline_key(prompt: int, layer: int, position: int) -> tuple[int, int, int]:
    """Key a single-row selection by the prompt that produced it.

    Every C1-phase record carries slot 0, so a slot-keyed baseline would make the
    second prompt's side-by-side rows compare against the first prompt's
    selections.
    """

    return (int(prompt), int(layer), int(position))


def _baseline_for(
    baselines: dict[tuple[int, int, int], Any],
    *,
    slot: int,
    layer: int,
    position: int,
) -> Any:
    """Look a C1-phase selection up by the slot that holds that prompt now."""

    return baselines.get(_selection_baseline_key(slot, layer, position))


def _slot_window_offset(slot: int, blocks_per_request: int, block_bytes: int) -> int:
    """Byte offset of one slot's window in the shared batch KV slab.

    Slot ``s`` owns blocks ``[s*bpr, (s+1)*bpr)``, so a digest that reads from the
    slab base reports a neighbour's writes as its own.
    """

    return int(slot) * int(blocks_per_request) * int(block_bytes)


# -- device reads -----------------------------------------------------------


def _read(runtime, ptr: int, dtype, count: int) -> np.ndarray:
    """Synchronous device read into a fresh host array."""

    itemsize = np.dtype(dtype).itemsize
    nbytes = int(count) * itemsize
    if nbytes == 0:
        return np.empty(0, dtype=dtype)
    host = np.empty(int(count), dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(ptr), nbytes),
        nbytes,
        runtime=runtime,
    )
    return host


def _digest(runtime, ptr: int, nbytes: int) -> str:
    if nbytes == 0:
        return hashlib.sha256(b"").hexdigest()[:16]
    host = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(ptr), nbytes),
        nbytes,
        runtime=runtime,
    )
    return hashlib.sha256(host.tobytes()).hexdigest()[:16]


def _raw_slab_positions(index_state) -> int:
    """Total physical positions the batch index slab covers, all slots included."""

    return int(index_state.state_slots) * int(index_state.slot_capacity_tokens)


def _physical_positions(block_table: np.ndarray, block_size: int, count: int) -> np.ndarray:
    """Logical position -> physical row, through one request's page table."""

    logical = np.arange(int(count), dtype=np.int64)
    table = np.asarray(block_table, dtype=np.int64)
    return table[logical // int(block_size)] * int(block_size) + logical % int(block_size)


# -- prompts ----------------------------------------------------------------


def _long_prompt(tokenizer, source: tuple[str, str, str], length: int) -> list[int]:
    """Build exactly ``length`` tokens from a source, with changing markers."""

    category, head, body = source
    text = head
    index = 0
    while True:
        index += 1
        text += f"[{category} {index:04d}] " + body
        tokens = [int(token) for token in tokenizer.encode(text)]
        if len(tokens) >= length:
            if len(tokens) == length:
                return tokens
            # Trim on a whole marker boundary when possible so the tail is not a
            # half-token sentence; the exact length is what the ladder needs.
            return tokens[:length]


# -- taps -------------------------------------------------------------------


class Taps:
    """Host-side records of one QSA layer's index selection on either path."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.enabled = False
        self.c1_slot = 0
        self.runtime = None

    def clear(self) -> None:
        self.records = []

    def _c1_record(self, weights, kwargs) -> dict[str, Any]:
        scratch = kwargs["scratch"].qsa
        index_state = kwargs.get("index_state")
        position = int(kwargs["position"])
        heads = int(kwargs.get("index_heads") or 0)
        dim = int(kwargs.get("index_dim") or 0)
        record: dict[str, Any] = {
            "side": "c1",
            "layer": int(weights.layer_id),
            "qsa_state_index": int(weights.qsa_state_index),
            "slot": int(self.c1_slot),
            "position": position,
            "sparse": False,
        }
        if index_state is None:
            return record
        count = int(index_state.count)
        ratio = int(index_state.compression_ratio)
        blocks = count // ratio
        record.update({"count": count, "blocks": blocks})
        if count <= int(index_state.dense_equivalent_limit):
            return record
        selected_count = int(_read(self.runtime, index_state.selected_count.ptr, np.int32, 1)[0])
        record.update(
            {
                "sparse": True,
                "selected_count": selected_count,
                "query": _read(self.runtime, scratch.index_query.ptr, np.float32, heads * dim),
                "pooled": _read(
                    self.runtime, index_state.pooled_keys.ptr, np.float32, blocks * dim
                ),
                "starts": _read(self.runtime, index_state.block_starts.ptr, np.int64, blocks),
                "scores": _read(self.runtime, index_state.scores.ptr, np.float32, blocks),
                "selected": _read(
                    self.runtime,
                    index_state.selected_positions.ptr,
                    np.int64,
                    selected_count,
                ),
                # The raw slab is indexed by physical position, so read the
                # whole capacity and gather through this request's own map.
                "raw_prefix": _read(
                    self.runtime,
                    index_state.raw_keys.ptr,
                    np.float32,
                    int(index_state.capacity) * dim,
                )
                .reshape(-1, dim)[
                    np.asarray(index_state.physical_positions_host[:count], dtype=np.int64)
                ],
            }
        )
        return record

    def _batch_records(self, weights, kwargs) -> list[dict[str, Any]]:
        scratch = kwargs["scratch"].qsa
        index_state = kwargs["index_state"]
        selection = kwargs["selection"]
        rows = int(kwargs["rows"])
        active = index_state.active
        if active is None:
            return []
        slots = [int(value) for value in active.slots]
        cursors = [int(index_state.cursor_host[slot]) for slot in slots]
        # ``select_active`` scores only the sparse group, in that group's own
        # ascending order, so its score rows are subset rows -- not mixer rows.
        # ``partition_rows`` re-derives the same split from the same cursors.
        _, sparse_group = index_state.partition_rows()
        ratio = int(index_state.compression_ratio)
        dim = int(index_state.index_dim)
        heads = int(index_state.index_heads)
        records = []
        slab = _read(
            self.runtime,
            index_state.raw_keys.ptr,
            np.float32,
            _raw_slab_positions(index_state) * dim,
        ).reshape(-1, dim)
        for row in range(rows):
            record: dict[str, Any] = {
                "side": "batch",
                "layer": int(weights.layer_id),
                "qsa_state_index": int(weights.qsa_state_index),
                "slot": slots[row],
                "row": row,
                "position": cursors[row] - 1,
                "count": cursors[row],
                "blocks": cursors[row] // ratio,
                "sparse": False,
            }
            if record["blocks"] > int(index_state.block_budget):
                blocks = record["blocks"]
                staged = int(selection.counts_host[row])
                pooled_stride = int(index_state.complete_blocks) * dim
                score_stride = int(index_state.complete_blocks)
                record.update(
                    {
                        "sparse": True,
                        "selected_count": staged,
                        "query": _read(
                            self.runtime,
                            scratch.index_query.ptr + row * heads * dim * 4,
                            np.float32,
                            heads * dim,
                        ),
                        "pooled": _read(
                            self.runtime,
                            index_state.pooled_keys.ptr + slots[row] * pooled_stride * 4,
                            np.float32,
                            blocks * dim,
                        ),
                        "starts": _read(
                            self.runtime, index_state.block_starts.ptr, np.int64, blocks
                        ),
                        "scores": _read(
                            self.runtime,
                            index_state.scores.ptr
                            + _sparse_subset_row(sparse_group, row) * score_stride * 4,
                            np.float32,
                            blocks,
                        ),
                        "selected": np.asarray(
                            selection.positions_host[row, :staged], dtype=np.int64
                        ).copy(),
                    }
                )
                # This slot's own raw keys, read through its own page table.
                physical = _physical_positions(
                    index_state.attention.block_host[slots[row]],
                    int(index_state.block_size),
                    record["count"],
                )
                record["raw_prefix"] = slab[physical]
                record["physical"] = physical
            records.append(record)
        return records

    def install(self) -> None:
        taps = self
        original_c1 = c1_module.run_qwen4_exp_dense_qsa_layer
        original_batch = batch_module.run_qwen4_exp_batch_qsa_layer

        def c1_wrapper(residual_ptr, weights, **kwargs):
            output = original_c1(residual_ptr, weights, **kwargs)
            if taps.enabled:
                taps.records.append(taps._c1_record(weights, kwargs))
            return output

        def batch_wrapper(residual_ptr, weights, **kwargs):
            output = original_batch(residual_ptr, weights, **kwargs)
            if taps.enabled:
                taps.records.extend(taps._batch_records(weights, kwargs))
            return output

        c1_module.run_qwen4_exp_dense_qsa_layer = c1_wrapper
        batch_module.run_qwen4_exp_batch_qsa_layer = batch_wrapper


# -- comparisons ------------------------------------------------------------


def _oracle(record: dict[str, Any], ratio: int, budget: int) -> np.ndarray:
    """Documented host selection over this side's own device score bytes."""

    selection = qsa_select_positions(
        record["scores"][None, :],
        record["starts"],
        query_positions=[record["position"]],
        available_positions=np.arange(record["count"], dtype=np.int64),
        compression_ratio=ratio,
        block_budget=budget,
    )
    return np.asarray(selection.selected_positions[0], dtype=np.int64)


def _block_margin(scores: np.ndarray, budget: int) -> float:
    """Score gap between the last selected block and the best rejected one."""

    if scores.size <= budget:
        return float("inf")
    ordered = np.sort(scores)[::-1]
    return float(ordered[budget - 1]) - float(ordered[budget])


class CrossFeed:
    """Run the c1 scalar top-k entry point on the batch's own score bytes.

    Both entry points launch the same kernel body, so this turns "the two
    selection paths are interchangeable" from a source reading into a
    measurement: identical score bytes must give the batch's own selection.
    """

    def __init__(self, runtime, *, stride: int) -> None:
        self.runtime = runtime
        self.stride = int(stride)
        self.scores = malloc(self.stride * 4, runtime=runtime)
        self.positions = malloc(self.stride * 8, runtime=runtime)
        self.count = malloc(4, runtime=runtime)

    def select(self, record: dict[str, Any], ratio: int, budget: int) -> np.ndarray:
        scores = np.ascontiguousarray(record["scores"], dtype=np.float32)
        host = np.frombuffer(scores.tobytes(), dtype=np.uint8).copy()
        copy_host_to_device(
            self.scores, host_array_ptr(host), host.nbytes, runtime=self.runtime
        )
        qwen4_exp_qsa_topk_expand_f32_i64(
            self.scores.ptr,
            self.positions.ptr,
            self.count.ptr,
            int(record["blocks"]),
            int(record["position"]),
            int(ratio),
            int(budget),
            runtime=self.runtime,
        )
        count = int(_read(self.runtime, self.count.ptr, np.int32, 1)[0])
        return _read(self.runtime, self.positions.ptr, np.int64, count)

    def close(self) -> None:
        free(self.scores, runtime=self.runtime)
        free(self.positions, runtime=self.runtime)
        free(self.count, runtime=self.runtime)


def _compare_selection(
    c1: dict[str, Any],
    batch: dict[str, Any],
    *,
    ratio: int,
    budget: int,
) -> dict[str, Any]:
    """One (layer, slot) sparse comparison: inputs, policy, and the difference."""

    result: dict[str, Any] = {
        "layer": c1["layer"],
        "slot": c1["slot"],
        "position": c1["position"],
        "count": c1["count"],
        "blocks": c1["blocks"],
        "routing_agrees": bool(
            c1["sparse"] == batch["sparse"]
            and c1["position"] == batch["position"]
            and c1["count"] == batch["count"]
        ),
    }
    if not (c1["sparse"] and batch["sparse"]):
        result["mode"] = "dense"
        return result
    result["mode"] = "sparse"
    query_equal = bool(np.array_equal(c1["query"], batch["query"]))
    pooled_equal = bool(np.array_equal(c1["pooled"], batch["pooled"]))
    starts_equal = bool(np.array_equal(c1["starts"], batch["starts"]))
    scores_equal = bool(np.array_equal(c1["scores"], batch["scores"]))
    raw_equal = bool(np.array_equal(c1["raw_prefix"], batch["raw_prefix"]))
    selected_equal = bool(np.array_equal(c1["selected"], batch["selected"]))
    oracle_c1 = _oracle(c1, ratio, budget)
    oracle_batch = _oracle(batch, ratio, budget)
    c1_policy = bool(np.array_equal(oracle_c1, c1["selected"]))
    batch_policy = bool(np.array_equal(oracle_batch, batch["selected"]))
    delta = np.abs(c1["scores"].astype(np.float64) - batch["scores"].astype(np.float64))
    query_delta = np.abs(
        c1["query"].astype(np.float64) - batch["query"].astype(np.float64)
    )
    raw_delta = np.abs(
        c1["raw_prefix"].astype(np.float64) - batch["raw_prefix"].astype(np.float64)
    )
    pooled_delta = np.abs(
        c1["pooled"].astype(np.float64) - batch["pooled"].astype(np.float64)
    )
    result.update(
        {
            "selected_count_c1": int(c1["selected_count"]),
            "selected_count_batch": int(batch["selected_count"]),
            "selected_equal": selected_equal,
            "selected_differing_positions": int(
                np.setxor1d(c1["selected"], batch["selected"]).size
            ),
            "query_equal": query_equal,
            "raw_keys_equal": raw_equal,
            "pooled_keys_equal": pooled_equal,
            "block_starts_equal": starts_equal,
            "scores_equal": scores_equal,
            "query_max_abs_delta": float(query_delta.max()) if query_delta.size else 0.0,
            "query_scale": float(np.abs(c1["query"]).max()),
            "raw_keys_max_abs_delta": float(raw_delta.max()) if raw_delta.size else 0.0,
            "raw_keys_scale": float(np.abs(c1["raw_prefix"]).max()),
            # Only positions inside a complete compression block are pooled, so
            # this separates accumulated key drift from the current step's key.
            "raw_keys_differing_rows": int(np.count_nonzero(raw_delta.max(axis=1))),
            "raw_keys_last_differing_position": int(
                np.nonzero(raw_delta.max(axis=1))[0].max()
            )
            if raw_delta.size and np.count_nonzero(raw_delta.max(axis=1))
            else -1,
            "pooled_blocks": int(c1["blocks"]),
            "pooled_keys_max_abs_delta": float(pooled_delta.max()) if pooled_delta.size else 0.0,
            "pooled_keys_scale": float(np.abs(c1["pooled"]).max()),
            "score_max_abs_delta": float(delta.max()) if delta.size else 0.0,
            "score_scale": float(np.abs(c1["scores"]).max()),
            "score_differing_elements": int(np.count_nonzero(delta)),
            "policy_c1_matches_host": c1_policy,
            "policy_batch_matches_host": batch_policy,
            "block_margin_c1": _block_margin(c1["scores"], budget),
            "block_margin_batch": _block_margin(batch["scores"], budget),
        }
    )
    result["cut_is_near_tie"] = bool(
        result["block_margin_c1"] <= 4.0 * result["score_max_abs_delta"]
        or result["block_margin_batch"] <= 4.0 * result["score_max_abs_delta"]
    )
    if not selected_equal:
        # The attribution is already complete without a heuristic: each side's
        # selection is checked against the documented host rule *on its own
        # score bytes*. If both match, the difference is a consequence of the
        # score bytes alone. These two numbers say whether the cut was a near
        # tie (margin comparable to the score delta) or a wide one.
        differing_blocks = sorted(
            {
                position // ratio
                for position in np.setxor1d(c1["selected"], batch["selected"]).tolist()
            }
        )
        result["differing_blocks"] = len(differing_blocks)
    return result


def _align(
    c1: Sequence[dict[str, Any]],
    batch: Sequence[dict[str, Any]],
    *,
    slots: Sequence[int],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair records by (layer, slot) and fail loudly on a routing disagreement."""

    def key(record: dict[str, Any]) -> tuple[int, int]:
        return int(record["layer"]), int(record["slot"])

    c1_by_key = {key(record): record for record in c1}
    batch_by_key = {key(record): record for record in batch}
    missing_batch = sorted(set(c1_by_key) - set(batch_by_key))
    missing_c1 = sorted(set(batch_by_key) - set(c1_by_key))
    if missing_batch or missing_c1:
        raise RuntimeError(
            "c1 and the batch disagree on which layers ran a sparse selection: "
            f"c1-only {missing_batch}, batch-only {missing_c1}"
        )
    pairs = []
    for slot_id in slots:
        for layer in sorted({item[0] for item in c1_by_key}):
            pairs.append((c1_by_key[(layer, slot_id)], batch_by_key[(layer, slot_id)]))
    return pairs


def _slot_state_digest(batch, slot: int, runtime) -> dict[str, Any]:
    """Hash one slot's whole persistent state through its own page tables."""

    import hashlib

    attention_states = batch.attention_states
    block_size = int(batch.block_size)
    row_bytes = int(attention_states[0].kv_heads) * int(attention_states[0].head_dim) * 2
    block_bytes = block_size * row_bytes
    kv = []
    for layer_state in attention_states:
        # Slot s owns blocks [s*bpr, (s+1)*bpr) of the shared slab.
        used = (int(batch.position(slot)) + block_size - 1) // block_size
        offset = _slot_window_offset(
            slot, int(layer_state.blocks_per_request), block_bytes
        )
        for cache in (layer_state.key_cache, layer_state.value_cache):
            host = _read(runtime, cache.ptr + offset, np.uint8, used * block_bytes)
            kv.append(hashlib.sha256(host.tobytes()).digest())
    index = []
    cursors = []
    pooled = []
    for index_state in batch.index_states:
        count = int(index_state.cursor_host[slot])
        cursors.append(count)
        pooled.append(int(index_state.pooled_count_host[slot]))
        dim = int(index_state.index_dim)
        physical = _physical_positions(
            np.asarray(index_state.attention.block_host[slot]).reshape(-1),
            block_size,
            max(count, 1),
        )
        slab = _read(
            runtime,
            index_state.raw_keys.ptr,
            np.float32,
            _raw_slab_positions(index_state) * dim,
        ).reshape(-1, dim)
        index.append(hashlib.sha256(slab[physical].tobytes()).digest())
    gdn = []
    for gdn_state in batch.gdn_states:
        gdn.append(
            hashlib.sha256(
                _read(
                    runtime,
                    gdn_state.conv_slot_ptr(slot),
                    np.uint8,
                    int(gdn_state.conv_bytes_per_slot),
                ).tobytes()
                + _read(
                    runtime,
                    gdn_state.recurrent_slot_ptr(slot),
                    np.uint8,
                    int(gdn_state.recurrent_bytes_per_slot),
                ).tobytes()
            ).digest()
        )
    ple = b""
    if batch.ple_state is not None:
        ple = hashlib.sha256(
            _read(
                runtime,
                batch.ple_state.history_slot_ptr(slot),
                np.uint8,
                int(batch.ple_state.history_bytes_per_slot),
            ).tobytes()
        ).digest()
    return {
        "kv": hashlib.sha256(b"".join(kv)).hexdigest(),
        "index": hashlib.sha256(b"".join(index)).hexdigest(),
        "gdn": hashlib.sha256(b"".join(gdn)).hexdigest(),
        "ple": ple.hex(),
        "cursors": cursors,
        "pooled_counts": pooled,
        "position": int(batch.position(slot)),
    }


# -- import audit -----------------------------------------------------------


def _import_audit(reference, batch, slot: int, runtime) -> dict[str, Any]:
    """Compare every imported buffer by logical position, through each page table."""

    config = batch.config
    position = int(reference.position)
    qsa_layers = int(config.qsa_layer_count)
    row_bytes = int(config.attention_kv_head_count) * int(config.attention_key_length) * 2
    block_size = int(batch.block_size)
    used_blocks = (position + block_size - 1) // block_size
    mismatches: list[str] = []

    source_attention = reference.attention_states
    dest_attention = batch.attention_states
    for layer in range(qsa_layers):
        source_table = np.asarray(source_attention[layer].block_host).reshape(-1)
        dest_table = np.asarray(dest_attention[layer].block_host[slot]).reshape(-1)
        source_physical = _physical_positions(source_table, block_size, position)
        dest_physical = _physical_positions(dest_table, block_size, position)
        span = used_blocks * block_size
        for name, source, dest in (
            ("key", source_attention[layer].key_cache, dest_attention[layer].key_cache),
            ("value", source_attention[layer].value_cache, dest_attention[layer].value_cache),
        ):
            source_rows = _read(runtime, source.ptr, np.uint8, span * row_bytes)
            source_rows = source_rows.reshape(span, row_bytes)[source_physical]
            # A batch slot owns a disjoint global block window, so read only the
            # row range this slot's page table actually addresses.
            low = int(dest_physical.min())
            high = int(dest_physical.max()) + 1
            dest_rows = _read(
                runtime, dest.ptr + low * row_bytes, np.uint8, (high - low) * row_bytes
            )
            dest_rows = dest_rows.reshape(high - low, row_bytes)[dest_physical - low]
            if not np.array_equal(source_rows, dest_rows):
                differing = int(np.count_nonzero(source_rows != dest_rows))
                mismatches.append(f"qsa{layer} {name} ({differing} bytes)")

    source_index = reference.index_states
    dest_index = batch.index_states
    for layer in range(qsa_layers):
        dim = int(source_index[layer].index_dim)
        count = int(source_index[layer].count)
        dest_count = int(dest_index[layer].cursor_host[slot])
        if count != dest_count:
            mismatches.append(f"index{layer} cursor {dest_count} != {count}")
        source_physical = np.asarray(source_index[layer].physical_positions_host[:count])
        dest_physical = _physical_positions(
            np.asarray(dest_index[layer].attention.block_host[slot]).reshape(-1),
            block_size,
            count,
        )
        # Both slabs are physical-position indexed; the last block is only
        # partly used, so read the whole span and gather through each map.
        source_rows = _read(
            runtime,
            source_index[layer].raw_keys.ptr,
            np.float32,
            int(source_index[layer].capacity) * dim,
        ).reshape(-1, dim)[source_physical]
        dest_rows = _read(
            runtime,
            dest_index[layer].raw_keys.ptr,
            np.float32,
            _raw_slab_positions(dest_index[layer]) * dim,
        ).reshape(-1, dim)[dest_physical]
        if not np.array_equal(source_rows, dest_rows):
            differing = int(np.count_nonzero(source_rows != dest_rows))
            mismatches.append(f"index{layer} raw_keys ({differing} floats)")
        if int(dest_index[layer].pooled_count_host[slot]) != 0:
            mismatches.append(
                f"index{layer} pooled_count {int(dest_index[layer].pooled_count_host[slot])} "
                "was not reset on import"
            )

    state = reference.state
    gdn_layers = len(batch.gdn_states)
    conv_bytes = int(batch.gdn_states[0].conv_bytes_per_slot)
    recurrent_bytes = int(batch.gdn_states[0].recurrent_bytes_per_slot)
    for layer in range(gdn_layers):
        conv = _read(
            runtime,
            state.gdn_conv.ptr + layer * (state.gdn_conv.nbytes // gdn_layers),
            np.uint8,
            conv_bytes,
        )
        dest_conv = _read(
            runtime, batch.gdn_states[layer].conv_slot_ptr(slot), np.uint8, conv_bytes
        )
        if not np.array_equal(conv, dest_conv):
            mismatches.append(f"gdn{layer} conv")
        recurrent = _read(
            runtime,
            state.gdn_matrix.ptr + layer * (state.gdn_matrix.nbytes // gdn_layers),
            np.uint8,
            recurrent_bytes,
        )
        dest_recurrent = _read(
            runtime,
            batch.gdn_states[layer].recurrent_slot_ptr(slot),
            np.uint8,
            recurrent_bytes,
        )
        if not np.array_equal(recurrent, dest_recurrent):
            mismatches.append(f"gdn{layer} recurrent")

    if batch.ple_state is not None:
        history_bytes = int(batch.ple_state.history_bytes_per_slot)
        source_history = _read(runtime, state.ple_conv.ptr, np.uint8, history_bytes)
        dest_history = _read(
            runtime, batch.ple_state.history_slot_ptr(slot), np.uint8, history_bytes
        )
        if not np.array_equal(source_history, dest_history):
            mismatches.append("ple history")

    return {
        "slot": slot,
        "position": position,
        "used_blocks": used_blocks,
        "mismatches": mismatches,
        "exact": not mismatches,
        "kv_digest": _digest(
            runtime,
            batch.attention_states[0].key_cache.ptr,
            used_blocks * block_size * row_bytes,
        ),
    }


# -- run --------------------------------------------------------------------


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    kl = np.asarray([row["kl_c1_to_batch"] for row in records], dtype=np.float64)
    kl_back = np.asarray([row["kl_batch_to_c1"] for row in records], dtype=np.float64)
    top1 = np.asarray([1.0 if row["top1_agreement"] else 0.0 for row in records])
    max_abs = np.asarray([row["max_absolute_logit_error"] for row in records])
    return {
        "rows": len(records),
        "kl_c1_to_batch": {
            "mean": float(kl.mean()),
            "p95": float(np.percentile(kl, 95)),
            "max": float(kl.max()),
        },
        "kl_batch_to_c1": {"mean": float(kl_back.mean()), "max": float(kl_back.max())},
        "top1_agreement": float(top1.mean()),
        "max_absolute_logit_error": {
            "mean": float(max_abs.mean()),
            "max": float(max_abs.max()),
        },
    }


def _selection_summary(comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sparse = [item for item in comparisons if item.get("mode") == "sparse"]
    summary = {
        "sparse_layers": 0,
        "routing_disagreements": 0,
        "selections_equal": 0,
        "selections_differing": 0,
        "differing_positions_total": 0,
        "inputs_bit_identical": 0,
        "policy_c1_matches_host": 0,
        "policy_batch_matches_host": 0,
        "selected_count_mismatches": 0,
        "cross_feed_checks": 0,
        "cross_feed_equal": 0,
    }
    if not sparse:
        return summary
    return {
        "sparse_layers": len(sparse),
        "routing_disagreements": int(
            sum(1 for item in sparse if not item["routing_agrees"])
        ),
        "selections_equal": int(sum(1 for item in sparse if item["selected_equal"])),
        "selections_differing": int(
            sum(1 for item in sparse if not item["selected_equal"])
        ),
        "differing_positions_total": int(
            sum(item["selected_differing_positions"] for item in sparse)
        ),
        "inputs_bit_identical": int(
            sum(
                1
                for item in sparse
                if item["query_equal"]
                and item["pooled_keys_equal"]
                and item["scores_equal"]
            )
        ),
        "policy_c1_matches_host": int(
            sum(1 for item in sparse if item["policy_c1_matches_host"])
        ),
        "policy_batch_matches_host": int(
            sum(1 for item in sparse if item["policy_batch_matches_host"])
        ),
        "selected_count_mismatches": int(
            sum(
                1
                for item in sparse
                if item["selected_count_c1"] != item["selected_count_batch"]
            )
        ),
        "cross_feed_checks": int(
            sum(
                1
                for item in sparse
                if "batch_scores_through_c1_entry_point_equal" in item
            )
        ),
        "cross_feed_equal": int(
            sum(
                1
                for item in sparse
                if item.get("batch_scores_through_c1_entry_point_equal")
            )
        ),
        "differing_on_near_tie_cut": int(
            sum(
                1
                for item in sparse
                if not item["selected_equal"] and item["cut_is_near_tie"]
            )
        ),
        "differing_on_wide_cut": int(
            sum(
                1
                for item in sparse
                if not item["selected_equal"] and not item["cut_is_near_tie"]
            )
        ),
        "block_margin_min": float(
            min(item["block_margin_c1"] for item in sparse)
        ),
        "block_margin_median": float(
            np.median([item["block_margin_c1"] for item in sparse])
        ),
        "score_delta_max": float(
            max(item["score_max_abs_delta"] for item in sparse)
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    model_path = args.model.expanduser().resolve()
    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    if len(lengths) < 1 or args.steps < 1:
        raise ValueError("at least one length and one step are required")
    context = args.context or (max(lengths) + args.steps + 8)
    started = perf_counter()
    report: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "lengths": lengths,
        "steps": args.steps,
        "context": context,
    }

    _apply_eager_profile()

    generator = None
    batch = None
    references: list[Qwen4ExpGGUFResidentModelRunner] = []
    taps = Taps()
    cross = None
    try:
        index = load_gguf_index(discover_gguf_files(model_path)[0])
        plugin = resolve_model(index.architecture or "")
        runtime = get_hip_runtime()
        taps.runtime = runtime
        load_started = perf_counter()
        generator = Qwen4ExpGGUFTextGenerator(
            model_path=model_path,
            weight_index=index,
            model_plugin=plugin,
            backend=args.backend,
            max_sequence_length=context,
            resident_capacity=1,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        resident = generator.runner.resident
        report["load_seconds"] = perf_counter() - load_started
        config = resident.plan.config
        ratio = int(config.qsa_compression_ratio)
        budget = int(config.qsa_block_budget)
        limit = budget * ratio + ratio - 1
        report["geometry"] = {
            "qsa_layers": int(config.qsa_layer_count),
            "compression_ratio": ratio,
            "block_budget": budget,
            "dense_equivalent_limit": limit,
            "index_heads": int(config.indexer_head_count),
            "index_dim": int(config.indexer_key_length),
            "block_size": int(batch_module.QWEN4_EXP_BATCH_BLOCK_SIZE),
        }
        print(
            f"loaded in {report['load_seconds']:.1f}s; dense_equivalent_limit "
            f"{limit}; lengths {lengths}",
            flush=True,
        )
        prompts = [
            {
                "category": source[0],
                "length": length,
                "tokens": _long_prompt(generator.tokenizer, source, length),
            }
            for source, length in zip(SOURCES, lengths)
        ]
        for row in prompts:
            if len(row["tokens"]) != row["length"]:
                raise RuntimeError(
                    f"prompt built with {len(row['tokens'])} tokens, wanted {row['length']}"
                )
        report["prompts"] = [
            {"category": row["category"], "length": row["length"]} for row in prompts
        ]

        references = [
            Qwen4ExpGGUFResidentModelRunner(
                resident,
                max_sequence_length=context,
                prefill_chunk_size=args.prefill_chunk_size,
                backend=args.backend,
                runtime=runtime,
            )
            for _ in prompts
        ]
        batch = Qwen4ExpGGUFBatchRunner(
            resident,
            state_slots=max(1, len(prompts) if not args.skip_c2 else 1),
            max_positions=context,
            backend=args.backend,
            runtime=runtime,
        )
        owners = [batch.admit(slot) for slot in range(batch.state_slots)]
        report["state_slots"] = int(batch.state_slots)
        report["device_after_owner"] = {"owned_bytes": memory_stats()}
        cross = CrossFeed(runtime, stride=budget * ratio + ratio - 1)
        taps.install()

        c1_phase: list[dict[str, Any]] = []
        c1_selection_sets: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
        for index_row, row in enumerate(prompts):
            print(
                f"[C1] prompt {index_row} ({row['category']}, {row['length']} tokens)",
                flush=True,
            )
            reference = references[index_row]
            reference.reset()
            prefill_started = perf_counter()
            result = reference.prefill_serial(row["tokens"])
            reference.last_logits = np.asarray(result.logits, dtype=np.float32).copy()
            prefill_seconds = perf_counter() - prefill_started
            if int(reference.position) != row["length"]:
                raise RuntimeError(
                    f"c1 prefill landed at {int(reference.position)}, wanted {row['length']}"
                )
            print(
                f"  prefill {prefill_seconds:.1f}s ({row['length'] / prefill_seconds:.1f} tok/s), "
                f"position {int(reference.position)}",
                flush=True,
            )
            if index_row:
                batch.reset_slot(0, owners[0])
            imported = batch.import_prefilled(0, owners[0], source=reference)
            audit = _import_audit(reference, batch, 0, runtime)
            print(
                f"  import at {imported}; exact={audit['exact']}"
                + ("" if audit["exact"] else f" {audit['mismatches']}"),
                flush=True,
            )
            row["prefill_seconds"] = prefill_seconds
            row["import"] = {"position": imported, "audit": audit}

            # Isolation at depth: a second slot holds an identical copy of this
            # request and never advances. Its whole persistent state must stay
            # byte-identical while slot 0 steps past the threshold.
            idle: dict[str, Any] | None = None
            if index_row == 1 and batch.state_slots > 1:
                batch.reset_slot(1, owners[1])
                idle_position = batch.import_prefilled(1, owners[1], source=reference)
                if idle_position != row["length"]:
                    raise RuntimeError(
                        f"idle slot imported at {idle_position}, wanted {row['length']}"
                    )
                idle = {
                    "slot": 1,
                    "position": idle_position,
                    "before": _slot_state_digest(batch, 1, runtime),
                    "after_steps": [],
                }

            step_records: list[dict[str, Any]] = []
            comparisons: list[dict[str, Any]] = []
            for step in range(args.steps):
                token = int(np.argmax(reference.last_logits))
                taps.clear()
                taps.c1_slot = 0
                taps.enabled = True
                c1_result = reference.step(token)
                batch_result = batch.step([token], slots=[0], owners=[owners[0]])
                taps.enabled = False
                if idle is not None:
                    idle["after_steps"].append(_slot_state_digest(batch, 1, runtime))
                if int(batch_result.positions[0]) != int(reference.position) - 1:
                    raise RuntimeError(
                        f"step {step} batch position {int(batch_result.positions[0])} "
                        f"but c1 says {int(reference.position) - 1}"
                    )
                c1_records = [r for r in taps.records if r["side"] == "c1"]
                batch_records = [r for r in taps.records if r["side"] == "batch"]
                layer_pairs = _align(c1_records, batch_records, slots=[0])
                for c1_record, batch_record in layer_pairs:
                    item = _compare_selection(
                        c1_record, batch_record, ratio=ratio, budget=budget
                    )
                    if item.get("mode") == "sparse":
                        c1_selection_sets[
                            _selection_baseline_key(
                                index_row, item["layer"], item["position"]
                            )
                        ] = {
                            "c1": c1_record["selected"].copy(),
                            "batch": batch_record["selected"].copy(),
                        }
                        if not item["selected_equal"]:
                            fed = cross.select(batch_record, ratio, budget)
                            item["batch_scores_through_c1_entry_point_equal"] = bool(
                                np.array_equal(fed, batch_record["selected"])
                            )
                    comparisons.append(item)
                metrics = compare_logits(c1_result.logits, batch_result.logits[0])
                back = compare_logits(batch_result.logits[0], c1_result.logits)
                entry = {
                    "step": step,
                    "position": int(batch_result.positions[0]),
                    "token": token,
                    "context": int(batch_result.positions[0]) + 1,
                    "sparse_layers": int(
                        sum(1 for record in batch_records if record["sparse"])
                    ),
                    "dense_layers": int(
                        sum(1 for record in batch_records if not record["sparse"])
                    ),
                    "c1_top1": int(metrics["teacher_top1"]),
                    "batch_top1": int(metrics["hipengine_top1"]),
                    "top1_agreement": bool(metrics["top1_agreement"]),
                    "kl_c1_to_batch": float(metrics["kl_teacher_to_hipengine"]),
                    "kl_batch_to_c1": float(back["kl_teacher_to_hipengine"]),
                    "max_absolute_logit_error": float(metrics["max_absolute_logit_error"]),
                    "mean_absolute_logit_error": float(
                        metrics["mean_absolute_logit_error"]
                    ),
                }
                step_records.append(entry)
                reference.last_logits = np.asarray(
                    c1_result.logits, dtype=np.float32
                ).copy()
                if args.progress:
                    sparse_diff = sum(
                        1
                        for item in comparisons
                        if item.get("mode") == "sparse" and not item["selected_equal"]
                    )
                    print(
                        f"  step {step}: context {entry['context']}, sparse layers "
                        f"{entry['sparse_layers']}, selections differing {sparse_diff}, "
                        f"kl {entry['kl_c1_to_batch']:.3e}, "
                        f"max|dlogit| {entry['max_absolute_logit_error']:.3e}",
                        flush=True,
                    )
            if idle is not None:
                idle["unchanged"] = bool(
                    all(item == idle["before"] for item in idle["after_steps"])
                )
                row["idle_slot_isolation"] = idle
                batch.reset_slot(1, owners[1])
            c1_phase.append(
                {
                    "prompt": index_row,
                    "length": row["length"],
                    "prefill_seconds": row["prefill_seconds"],
                    "import": row["import"],
                    "idle_slot_isolation": row.get("idle_slot_isolation"),
                    "steps": step_records,
                    "selection": comparisons,
                    "selection_summary": _selection_summary(comparisons),
                    "logit_summary": _summary(step_records),
                }
            )
            summary = c1_phase[-1]["selection_summary"]
            print(
                f"  selection: {summary['sparse_layers']} sparse layers, "
                f"{summary['selections_equal']} equal, "
                f"{summary['selections_differing']} differing "
                f"({summary['differing_positions_total']} positions), "
                f"inputs bit-identical {summary['inputs_bit_identical']}, "
                f"policy host-match c1 {summary['policy_c1_matches_host']} / "
                f"batch {summary['policy_batch_matches_host']}",
                flush=True,
            )
        report["c1_phase"] = c1_phase
        _write(args.json_out, report)

        if not args.skip_c2 and len(prompts) >= 2:
            # Two prompts side by side, advanced together. Each row must still
            # match its own C1 result: the per-row selections, its own raw keys
            # read through its own page table, and its logits.
            print("[C2] two rows, both prompts re-prefilled and imported", flush=True)
            c2_imports: list[dict[str, Any]] = []
            for slot in (0, 1):
                batch.reset_slot(slot, owners[slot])
                reference = references[slot]
                reference.reset()
                result = reference.prefill_serial(prompts[slot]["tokens"])
                reference.last_logits = np.asarray(result.logits, dtype=np.float32).copy()
                imported = batch.import_prefilled(slot, owners[slot], source=reference)
                if imported != prompts[slot]["length"]:
                    raise RuntimeError(f"slot {slot} imported at {imported}")
                c2_imports.append(
                    {"slot": slot, "audit": _import_audit(reference, batch, slot, runtime)}
                )
            comparisons_c2: list[dict[str, Any]] = []
            step_records_c2: list[dict[str, Any]] = []
            ownership_checks: list[dict[str, Any]] = []
            invariance: list[dict[str, Any]] = []
            for step in range(args.steps):
                tokens = [int(np.argmax(reference.last_logits)) for reference in references[:2]]
                taps.clear()
                taps.enabled = True
                c1_results = []
                for slot, token in enumerate(tokens):
                    taps.c1_slot = slot
                    c1_results.append(references[slot].step(token))
                batch_result = batch.step(
                    tokens, slots=[0, 1], owners=[owners[0], owners[1]]
                )
                taps.enabled = False
                if batch_result.slot_ids != (0, 1):
                    raise RuntimeError(
                        f"step {step} published slots {batch_result.slot_ids}"
                    )
                for slot in (0, 1):
                    if int(batch_result.positions[slot]) != int(references[slot].position) - 1:
                        raise RuntimeError(
                            f"step {step} slot {slot} position "
                            f"{int(batch_result.positions[slot])} but c1 says "
                            f"{int(references[slot].position) - 1}"
                        )
                c1_records = [r for r in taps.records if r["side"] == "c1"]
                batch_records = [r for r in taps.records if r["side"] == "batch"]
                for c1_record, batch_record in _align(
                    c1_records, batch_records, slots=[0, 1]
                ):
                    other = next(
                        (
                            record
                            for record in c1_records
                            if record["layer"] == c1_record["layer"]
                            and record["slot"] != c1_record["slot"]
                        ),
                        None,
                    )
                    if (
                        c1_record["sparse"]
                        and batch_record["sparse"]
                        and other is not None
                        and "raw_prefix" in other
                    ):
                        # The two prompts have different lengths at a given
                        # step, so compare over the shared prefix.
                        span = min(
                            len(c1_record["raw_prefix"]),
                            len(other["raw_prefix"]),
                            len(batch_record["raw_prefix"]),
                        )
                        batch_raw = batch_record["raw_prefix"][:span].astype(np.float64)
                        own = float(
                            np.abs(
                                c1_record["raw_prefix"][:span].astype(np.float64)
                                - batch_raw
                            ).max()
                        )
                        foreign = float(
                            np.abs(other["raw_prefix"][:span].astype(np.float64) - batch_raw).max()
                        )
                        ownership_checks.append(
                            {
                                "step": step,
                                "slot": c1_record["slot"],
                                "layer": c1_record["layer"],
                                "own_max_abs_delta": own,
                                "foreign_max_abs_delta": foreign,
                                "nearer_own_reference": bool(own < foreign),
                            }
                        )
                    elif c1_record["sparse"] and batch_record["sparse"]:
                        ownership_checks.append(
                            {
                                "step": step,
                                "slot": c1_record["slot"],
                                "layer": c1_record["layer"],
                                "skipped": "neighbour is dense at this layer",
                            }
                        )
                    item = _compare_selection(
                        c1_record, batch_record, ratio=ratio, budget=budget
                    )
                    if item.get("mode") == "sparse":
                        # Composition invariance: the same row, same position,
                        # same teacher tokens, with a neighbour present.
                        baseline = _baseline_for(
                            c1_selection_sets,
                            slot=item["slot"],
                            layer=item["layer"],
                            position=item["position"],
                        )
                        if baseline is not None:
                            invariance.append(
                                {
                                    "slot": item["slot"],
                                    "layer": item["layer"],
                                    "position": item["position"],
                                    "c2_equal_c1_phase_batch": bool(
                                        np.array_equal(
                                            baseline["batch"], batch_record["selected"]
                                        )
                                    ),
                                    "c2_equal_c1_phase_c1": bool(
                                        np.array_equal(
                                            baseline["c1"], batch_record["selected"]
                                        )
                                    ),
                                }
                            )
                        if not item["selected_equal"]:
                            fed = cross.select(batch_record, ratio, budget)
                            item["batch_scores_through_c1_entry_point_equal"] = bool(
                                np.array_equal(fed, batch_record["selected"])
                            )
                    comparisons_c2.append(item)
                for slot in (0, 1):
                    metrics = compare_logits(c1_results[slot].logits, batch_result.logits[slot])
                    back = compare_logits(batch_result.logits[slot], c1_results[slot].logits)
                    step_records_c2.append(
                        {
                            "step": step,
                            "slot": slot,
                            "position": int(batch_result.positions[slot]),
                            "context": int(batch_result.positions[slot]) + 1,
                            "sparse": bool(
                                sum(
                                    1
                                    for record in batch_records
                                    if record["slot"] == slot and record["sparse"]
                                )
                            ),
                            "top1_agreement": bool(metrics["top1_agreement"]),
                            "kl_c1_to_batch": float(metrics["kl_teacher_to_hipengine"]),
                            "kl_batch_to_c1": float(back["kl_teacher_to_hipengine"]),
                            "max_absolute_logit_error": float(
                                metrics["max_absolute_logit_error"]
                            ),
                        }
                    )
                    references[slot].last_logits = np.asarray(
                        c1_results[slot].logits, dtype=np.float32
                    ).copy()
                if args.progress:
                    print(
                        f"  step {step}: slots {tuple(batch_result.slot_ids)}, "
                        f"selections differing "
                        f"{sum(1 for item in comparisons_c2 if item.get('mode') == 'sparse' and not item['selected_equal'])}",
                        flush=True,
                    )
            report["c2_phase"] = {
                "imports": c2_imports,
                "steps": step_records_c2,
                "selection": comparisons_c2,
                "selection_summary": _selection_summary(comparisons_c2),
                "logit_summary": _summary(step_records_c2),
                "composition_invariance": invariance,
                "ownership": ownership_checks,
                "ownership_checks": len(ownership_checks),
                "ownership_skipped": int(
                    sum(1 for item in ownership_checks if "skipped" in item)
                ),
                "ownership_all_nearer_own": (
                    bool(
                        all(
                            item["nearer_own_reference"]
                            for item in ownership_checks
                            if "nearer_own_reference" in item
                        )
                    )
                    if any("nearer_own_reference" in item for item in ownership_checks)
                    else None
                ),
            }
            summary = report["c2_phase"]["selection_summary"]
            print(
                f"  selection: {summary['sparse_layers']} sparse layers, "
                f"{summary['selections_equal']} equal, "
                f"{summary['selections_differing']} differing, "
                f"policy host-match c1 {summary['policy_c1_matches_host']} / "
                f"batch {summary['policy_batch_matches_host']}, imports exact "
                f"{[item['audit']['exact'] for item in c2_imports]}, ownership "
                f"nearer-own {report['c2_phase']['ownership_all_nearer_own']}",
                flush=True,
            )

        _write(args.json_out, report)
        report["final"] = {
            "positions": [batch.position(slot) for slot in range(batch.state_slots)],
            "poisoned_slots": list(batch.poisoned_slots),
        }
        report["elapsed_seconds"] = perf_counter() - started
        _write(args.json_out, report)
        print(f"done in {report['elapsed_seconds']:.1f}s", flush=True)
        return 0
    finally:
        if cross is not None:
            cross.close()
        if batch is not None:
            batch.close()
        for reference in references:
            reference.close()
        if generator is not None:
            generator.close()


def _write(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=_json_default) + "\n", "utf-8")
    print(f"wrote {path}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


if __name__ == "__main__":
    raise SystemExit(main())
