#!/usr/bin/env python3
"""Gemma 4 D12 diagnostic: same-artifact teacher-forced INT8 vs BF16 KV.

This is a bounded diagnostic evaluation, not a promotion gate and not a
performance campaign. It runs the *same* frozen prompt chain through two KV
storages on the same artifact, host and GPU, and compares the teacher-forced
full-vocabulary next-token distributions:

* ``bf16``  -- the comparison default (per-layer BF16 K/V caches);
* ``int8_per_token_head`` -- the D12 BF16-source per-token/head INT8 writer and
  the strict FP32-reconstruction decode/prefill consumers over the owned INT8
  cache (no persistent BF16 shadow).

The evaluator is reused, not reimplemented: ``evaluate`` from
``scripts/gemma4_teacher_forced_gate.py`` supplies the full-vocabulary float64
KL, the nearest-rank tails, the binding ``docs/EXECUTION-PROFILES.md`` limits,
and the small-chain zero-flip screening rule. The KL direction is
``kl(bf16 || int8)``: BF16 is the reference arm.

What the run records, all in one compact JSON artifact:

* the frozen workload (exact prompt ids, chain hashes, prefill widths) written
  *before* any logits are captured;
* per-case and combined verdicts against the frozen thresholds, with the
  overall pass/fail requiring every category verdict;
* literal-repeatability controls for both arms: exact shape/dtype/raw-byte
  equality, kept separate from the KL repeat diagnostic, with both captures of
  every arm preserved in the external ``.npz`` so the parent can recheck;
* execution controls that compare the **entire ordered** ``begin_block``
  schedule and route-event row sequence against the schedule the attention
  geometry demands, validate every observed writer/consumer key against its row
  shape, and require the above-block case to cross into a nonzero prefill
  offset. A workload whose frozen widths cannot place a prefill across
  ``max_block`` is rejected before capture.
* owned resident allocations read from live buffers after construction, kept
  separate from external device usage;
* a short real ``hipengine.LLM.generate_detailed`` request whose route is
  observed around the call and whose generator runner is reacquired afterwards;
* the overall verdict as the conjunction of the combined envelope, every
  category verdict, both repeatability controls, the execution controls, and
  the public witness.

Full logits are written to an ``.npz`` *outside* the repository (default under
``/mnt/nvme1/gemma4-eval``) and only their path and sha256 go in the artifact.

Example::

    env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=0 PYTHONPATH=. \\
      HIP_PATH=/opt/rocm ROCM_PATH=/opt/rocm \\
      .venv/bin/python scripts/gemma4_int8_kv_teacher_forced_eval.py run \\
      --out benchmarks/results/<diagnostic>.json \\
      --npz /mnt/nvme1/gemma4-eval/d12-int8-kv-teacher-forced.npz
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
import argparse
import json
import os
import socket
import sys
import time

import numpy as np

# The binding evaluator, thresholds and chain utilities. Importing rather than
# copying is the point: the metric, percentile convention and screening rule
# must be the ones the campaign already froze.
from scripts.gemma4_teacher_forced_gate import (
    THRESHOLDS,
    capture_chain,
    chain_sha256,
    evaluate,
    sha256_file,
    _provenance,
)

#: Which contract the frozen thresholds come from. This is a diagnostic screen
#: against the production numerical gate; a KV-representation change is a T3
#: representation change (docs/EXECUTION-PROFILES.md section 5), so a pass here
#: is a screening result, not an automatic-admission qualification.
CONTRACT_SOURCE = (
    "docs/EXECUTION-PROFILES.md section 6.1 calibrated production envelope "
    "(mean KL <= 1e-3, p95 <= 5e-3, p99 <= 2e-2, max <= 5e-2, top-1 >= 99%), "
    "with the September 10 2026 small-chain rule (a <500-row screen passes "
    "only with zero top-1 flips)"
)

#: The registered writer/consumer keys the INT8 layer must select. Selection is
#: by declared capability and launch shape, so these are witnesses, not inputs.
EXPECTED_WRITER_PROMPT = "int8_per_token_head/per_token_head_bf16_prompt_spans"
EXPECTED_WRITER_DECODE = "int8_per_token_head/per_token_head_bf16_spans"
EXPECTED_CONSUMER_PREFILL = "paged_attn_prefill/int8_per_token_head/gemma4_direct_spans"
EXPECTED_CONSUMER_DECODE = "paged_attn_decode/int8_per_token_head/gemma4_direct_spans"

DEFAULT_ARTIFACT = Path(
    "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
DEFAULT_CONTEXT = 8192
DEFAULT_SCORED_ROWS = 32

#: Three materially different prompt categories. Cycled by ``exact_prompt_ids``
#: to the exact target width, so the category is the content, not the length.
CORPUS_EN_PROSE: tuple[str, ...] = (
    "The capital of France is Paris, and the city has been a center of art and "
    "science for centuries. Its museums hold works that span the medieval "
    "period through the present day.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose "
    "and oxygen inside the chloroplasts of green plants.",
    "Ocean currents distribute heat around the planet; the Gulf Stream alone "
    "carries water warmer than the air above much of northern Europe.",
)
CORPUS_PY_CODE: tuple[str, ...] = (
    "def quicksort(items):\n"
    "    if len(items) <= 1:\n"
    "        return items\n"
    "    pivot = items[len(items) // 2]\n"
    "    left = [x for x in items if x < pivot]\n"
    "    right = [x for x in items if x > pivot]\n"
    "    return quicksort(left) + [pivot] + quicksort(right)\n",
    "class Cache:\n"
    "    def __init__(self, capacity):\n"
    "        self.capacity = capacity\n"
    "        self._store = {}\n"
    "    def get(self, key, default=None):\n"
    "        return self._store.get(key, default)\n",
)
CORPUS_JA_PROSE: tuple[str, ...] = (
    "京都の寺社は季節ごとに異なる表情を見せ、春の桜と秋の紅葉では観光客の数も変わる。",
    "海洋の循環は地球の熱を運び、メキシコ湾流は北ヨーロッパの多くの地域の"
    "気温を周囲の空気より暖かく保っている。",
    "産業革命がゆるやかな構造変化だったのか、それとも過去との突然の断絶"
    "だったのかについては、歴史家の見解が分かれている。",
)

#: (name, category, corpus, prompt width). The width is an absolute length; the
#: widths that sit on either side of the shipping prefill block are computed
#: from the resolved block, so the workload is frozen against what actually
#: ships rather than a guessed constant.
_UNRELATED_PROMPT = 96


@dataclass(frozen=True)
class FrozenCase:
    """One frozen prompt chain, its category, and its prefill split."""

    name: str
    category: str
    prompt_ids: tuple[int, ...]
    prefill: int
    scored_rows: int
    chain_sha256: str
    prompt_tokens: int


def freeze_workload(
    tokenize: Callable[[str], Sequence[int]],
    *,
    max_block: int,
    scored_rows: int = DEFAULT_SCORED_ROWS,
) -> list[FrozenCase]:
    """Tokenize the frozen category corpora into the exact screening workload.

    Three cases, each a distinct content category, with prompt widths on both
    sides of the shipping prefill block plus one width unrelated to it:

    * ``prose_en_short`` -- 96 tokens, unrelated to the block threshold;
    * ``code_py_below_block`` -- ``max_block - 32`` tokens, so the prefill is a
      single block strictly below the shipping width;
    * ``prose_ja_above_block`` -- ``max_block + 96`` tokens, so the prefill
      spans two blocks and the second block's append offset is nonzero.

    ``prefill = prompt_tokens - scored_rows - 1``, so exactly ``scored_rows``
    positions are teacher-forced per case (``capture_chain`` scores
    ``prompt_tokens - 1 - prefill`` rows). Token ids and chain hashes are
    deterministic; they are frozen here before any logits are captured.

    The ``prose_ja_above_block`` case must cross ``max_block`` into a second
    prefill block; a ``scored_rows`` wide enough to pull its prefill back to
    ``max_block`` or below (95 for the default widths) is rejected here, before
    capture, rather than silently producing a workload with no crossing.
    """

    from scripts.gemma4_campaign_bench import exact_prompt_ids

    if int(max_block) < 128:
        raise ValueError(
            f"max_block {max_block} is too small to place cases on both sides "
            "of the shipping prefill block"
        )
    if int(scored_rows) < 1:
        raise ValueError("scored_rows must be positive")
    scored = int(scored_rows)
    specs = (
        ("prose_en_short", "english-prose", CORPUS_EN_PROSE, _UNRELATED_PROMPT, False),
        ("code_py_below_block", "python-code", CORPUS_PY_CODE, int(max_block) - 32, False),
        ("prose_ja_above_block", "japanese-prose", CORPUS_JA_PROSE, int(max_block) + 96, True),
    )
    cases: list[FrozenCase] = []
    for name, category, corpus, prompt_tokens, must_cross in specs:
        prompt_tokens = int(prompt_tokens)
        prefill = prompt_tokens - scored - 1
        if prefill < 0 or prompt_tokens < 2:
            raise ValueError(
                f"case {name} cannot place {scored} scored rows in "
                f"{prompt_tokens} tokens"
            )
        if must_cross and prefill <= int(max_block):
            raise ValueError(
                f"case {name}: scored_rows {scored} leaves prefill {prefill}, "
                f"which does not cross max_block {max_block}; the workload "
                "requires a case that crosses into a second prefill block "
                "(reduce scored_rows)"
            )
        ids = [int(t) for t in exact_prompt_ids(tokenize, prompt_tokens, corpus=corpus)]
        if len(ids) != prompt_tokens:
            raise ValueError(
                f"case {name}: tokenizer produced {len(ids)} ids, expected "
                f"{prompt_tokens}"
            )
        cases.append(
            FrozenCase(
                name=name,
                category=category,
                prompt_ids=tuple(ids),
                prefill=prefill,
                scored_rows=scored,
                chain_sha256=chain_sha256(ids),
                prompt_tokens=prompt_tokens,
            )
        )
    return cases


def evaluate_cases(
    bf16_by_case: dict[str, np.ndarray],
    int8_by_case: dict[str, np.ndarray],
    *,
    order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Judge each case against the frozen evaluator and the combined chain.

    Baseline is the BF16 arm and candidate is the INT8 arm, so the reported KL
    is ``kl(bf16 || int8)``. The combined verdict concatenates every case's rows
    in ``order``; per-case verdicts are retained so a single category cannot be
    hidden by the aggregate.
    """

    names = list(order) if order is not None else sorted(bf16_by_case)
    if set(names) != set(int8_by_case):
        raise ValueError("the two arms do not cover the same cases")
    per_case: dict[str, Any] = {}
    base_rows: list[np.ndarray] = []
    cand_rows: list[np.ndarray] = []
    for name in names:
        per_case[name] = evaluate(bf16_by_case[name], int8_by_case[name])
        base_rows.append(np.asarray(bf16_by_case[name]))
        cand_rows.append(np.asarray(int8_by_case[name]))
    combined = evaluate(np.concatenate(base_rows, axis=0), np.concatenate(cand_rows, axis=0))
    return {"cases": names, "per_case": per_case, "combined": combined}


def quality_passed(quality: dict[str, Any]) -> bool:
    """Whether every category verdict and the combined verdict pass.

    The combined rows are a summary, not the gate: a small category that fails
    the zero-flip screening rule can be hidden by a large passing category in
    the aggregate, so every per-case verdict must pass too.
    """

    return bool(
        quality["combined"]["passed"]
        and all(verdict["passed"] for verdict in quality["per_case"].values())
    )


def check_repeatability(
    arm_a: dict[str, np.ndarray],
    arm_b: dict[str, np.ndarray],
    *,
    order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Exact repeatability of one arm captured twice: shape, dtype, raw bytes.

    The calibrated KL envelope is a quality gate, not an equality gate: two
    captures can differ in the low bits with unchanged top-1 and a KL far below
    the limits. A literal-determinism control therefore compares the arrays
    themselves. The KL comparison is kept separately as a diagnostic.
    """

    names = list(order) if order is not None else sorted(arm_a)
    if set(names) != set(arm_b):
        raise ValueError("the two captures do not cover the same cases")
    per_case: dict[str, Any] = {}
    for name in names:
        a = np.asarray(arm_a[name])
        b = np.asarray(arm_b[name])
        shape_match = a.shape == b.shape
        dtype_match = a.dtype == b.dtype
        raw_bytes_equal = bool(shape_match and dtype_match and a.tobytes() == b.tobytes())
        max_abs_diff = None
        if shape_match:
            max_abs_diff = float(
                np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
            )
        failed = [] if raw_bytes_equal else ["not_raw_byte_equal"]
        per_case[name] = {
            "a_shape": list(a.shape),
            "b_shape": list(b.shape),
            "a_dtype": str(a.dtype),
            "b_dtype": str(b.dtype),
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "raw_bytes_equal": raw_bytes_equal,
            "max_abs_diff": max_abs_diff,
            "failed": failed,
            "passed": raw_bytes_equal,
        }
    all_equal = all(verdict["passed"] for verdict in per_case.values())
    return {
        "cases": names,
        "per_case": per_case,
        "all_raw_bytes_equal": all_equal,
        "passed": all_equal,
    }


def owned_allocation_bytes(runner: Any) -> dict[str, Any]:
    """Bytes held by this runner's live buffers, read after construction.

    This is allocator arithmetic over the objects the runner actually holds, not
    a total-VRAM figure. For the BF16 arm the per-layer K/V caches are a subset
    of ``runner._buffers``, so they are reported once. For the INT8 arm the
    owner's buffers are a separate list, so its total is added to the runner's.
    Staging and layer-scratch buffers are taken lazily on the first forward and
    are therefore *not* in this after-construction number; they are covered by
    the external device-usage rows instead.
    """

    runner_buffers = sum(int(buffer.nbytes) for buffer in runner._buffers)
    if runner.uses_int8_kv:
        owner = runner.kv_cache
        kv_owned = int(owner.allocated_bytes)
        kv_scope = (
            "int8_per_token_head owner live buffers "
            "(INT8 payload + scale planes + identity page table + positions/"
            "counts + FP32 query/context scratch); a separate list from the "
            "runner's own buffers"
        )
        total = runner_buffers + kv_owned
    else:
        kv_owned = sum(int(buffer.nbytes) for buffer in runner._caches)
        kv_scope = (
            "bf16 per-layer K/V caches; these buffers are a subset of the "
            "runner's own buffer list, so they are not added again"
        )
        total = runner_buffers
    return {
        "runner_owned_bytes": runner_buffers,
        "kv_owned_bytes": kv_owned,
        "owned_total_bytes": total,
        "kv_scope": kv_scope,
        "scope": (
            "allocator arithmetic over live buffers after runner construction "
            "(before the first forward, so lazy staging/scratch are excluded); "
            "not a measured total VRAM"
        ),
    }


def device_usage_bytes() -> dict[str, int]:
    """External device free/total/used from the HIP runtime, for context only."""

    from hipengine.core.hip import get_hip_runtime

    free, total = get_hip_runtime().mem_get_info()
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "used_bytes": int(total) - int(free),
    }


def summarize_route_witness(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the registered writer/consumer selections the layer made."""

    writer_counts: dict[str, int] = {}
    consumer_counts: dict[str, int] = {}
    prefill = 0
    decode = 0
    histogram: dict[str, int] = {}
    for event in events:
        writer = str(event.get("writer"))
        consumer = str(event.get("consumer"))
        writer_counts[writer] = writer_counts.get(writer, 0) + 1
        consumer_counts[consumer] = consumer_counts.get(consumer, 0) + 1
        rows = int(event.get("rows", 0))
        if rows > 1:
            prefill += 1
            histogram[str(rows)] = histogram.get(str(rows), 0) + 1
        elif rows == 1:
            decode += 1
    return {
        "calls": len(events),
        "prefill_calls": prefill,
        "decode_calls": decode,
        "prefill_row_histogram": histogram,
        "writer_counts": writer_counts,
        "consumer_counts": consumer_counts,
    }


def summarize_block_witness(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate every ``begin_block`` offset, so nonzero offsets are witnessed."""

    prefill_blocks = [event for event in events if int(event["rows"]) > 1]
    decode_blocks = [event for event in events if int(event["rows"]) == 1]
    offsets = [int(event["write_offset"]) for event in events]
    return {
        "calls": len(events),
        "prefill_blocks": [
            {"write_offset": int(event["write_offset"]), "rows": int(event["rows"])}
            for event in prefill_blocks
        ],
        "decode_block_count": len(decode_blocks),
        "nonzero_offsets": sum(1 for offset in offsets if offset > 0),
        "max_write_offset": max(offsets) if offsets else None,
    }


def expected_block_plan(case: FrozenCase, max_block: int) -> dict[str, Any]:
    """The complete ordered block schedule ``capture_chain`` must produce.

    The runner forwards a prefill wider than ``max_block`` as consecutive
    blocks, so the prefill becomes ``ceil(prefill / max_block)`` blocks at
    offsets ``0, max_block, ...`` (the last may carry a single row) and the
    decode loop becomes ``scored_rows`` one-row blocks at strictly increasing
    absolute offsets ``prefill, prefill+1, ...``. ``schedule`` is the full
    ordered concatenation; ``prefill_blocks`` and ``decode_blocks`` are its two
    slices. This is the expected execution schedule, not a numerical oracle: it
    witnesses that chunking and absolute positions are what the frozen workload
    demands.
    """

    max_block = int(max_block)
    prefill = int(case.prefill)
    full, remainder = divmod(prefill, max_block)
    prefill_blocks = [
        {"write_offset": index * max_block, "rows": max_block} for index in range(full)
    ]
    if remainder:
        prefill_blocks.append({"write_offset": full * max_block, "rows": remainder})
    decode_blocks = [
        {"write_offset": prefill + index, "rows": 1}
        for index in range(int(case.scored_rows))
    ]
    return {
        "schedule": [dict(block) for block in prefill_blocks + decode_blocks],
        "prefill_blocks": prefill_blocks,
        "decode_blocks": decode_blocks,
        "num_prefill_blocks": len(prefill_blocks),
        "num_decode_blocks": len(decode_blocks),
        "crosses_block_boundary": prefill > max_block,
    }


def expected_route_rows(plan: dict[str, Any], num_layers: int) -> list[int]:
    """The expected route-event row sequence: each block's rows, repeated.

    The layer runs once per block per attention layer, so the expected shape
    sequence is ``[block.rows] * num_layers`` for every block, in schedule
    order. A one-row prefill chunk therefore appears as a one-row route event,
    indistinguishable by shape from a decode step; only its position in the
    schedule marks it as prefill.
    """

    return [
        int(block["rows"])
        for block in plan["schedule"]
        for _ in range(int(num_layers))
    ]


def evaluate_controls(
    workload: Sequence[FrozenCase],
    *,
    max_block: int,
    num_layers: int,
    route_events: dict[str, Sequence[dict[str, Any]]],
    block_events: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Judge the observed INT8 execution against the expected block schedule.

    The entire ordered ``begin_block`` schedule is compared, not filtered
    sublists: a swapped block order, a same-total schedule with swapped widths,
    or an extra block all fail. The observed route-event row sequence must equal
    the expected ``[block.rows] * num_layers`` concatenation, and each event's
    writer/consumer must match the key its row shape selects -- a prefill block
    (``rows > 1``) the prompt writer and prefill consumer, a decode step *or a
    one-row prefill chunk* (``rows == 1``) the decode pair. A case whose prefill
    crosses ``max_block`` must show a nonzero prefill offset. Registration and
    offsets witness execution and scheduling; they are not a numerical-continuity
    oracle.
    """

    per_case: dict[str, Any] = {}
    for case in workload:
        plan = expected_block_plan(case, max_block)
        events = list(route_events.get(case.name, ()))
        blocks = list(block_events.get(case.name, ()))
        expected_schedule = [dict(block) for block in plan["schedule"]]
        observed_schedule = [
            {
                "write_offset": int(block.get("write_offset", -1)),
                "rows": int(block.get("rows", -1)),
            }
            for block in blocks
        ]
        expected_rows = expected_route_rows(plan, num_layers)
        observed_rows = [int(event.get("rows", -1)) for event in events]
        expected_writers = [
            EXPECTED_WRITER_PROMPT if rows > 1 else EXPECTED_WRITER_DECODE
            for rows in expected_rows
        ]
        expected_consumers = [
            EXPECTED_CONSUMER_PREFILL if rows > 1 else EXPECTED_CONSUMER_DECODE
            for rows in expected_rows
        ]
        observed_writers = [str(event.get("writer")) for event in events]
        observed_consumers = [str(event.get("consumer")) for event in events]
        failed: list[str] = []
        if observed_schedule != expected_schedule:
            failed.append("begin_block_schedule")
        if observed_rows != expected_rows:
            failed.append("route_shape_sequence")
        if observed_writers != expected_writers:
            failed.append("route_writer_sequence")
        if observed_consumers != expected_consumers:
            failed.append("route_consumer_sequence")
        if plan["crosses_block_boundary"] and not any(
            block["write_offset"] > 0
            for block in observed_schedule[: plan["num_prefill_blocks"]]
        ):
            failed.append("above_block_case_no_nonzero_prefill_offset")
        per_case[case.name] = {
            "crosses_block_boundary": plan["crosses_block_boundary"],
            "num_prefill_blocks": plan["num_prefill_blocks"],
            "num_decode_blocks": plan["num_decode_blocks"],
            "expected_schedule": expected_schedule,
            "observed_schedule": observed_schedule,
            "expected_route_rows": expected_rows,
            "observed_route_rows": observed_rows,
            "failed": failed,
            "passed": not failed,
        }
    crossing_cases = [
        case.name for case in workload if int(case.prefill) > int(max_block)
    ]
    aggregate_failed = [
        f"case_controls:{name}" for name, verdict in per_case.items() if not verdict["passed"]
    ]
    if not crossing_cases:
        aggregate_failed.append("workload_has_no_above_block_case")
    elif not all(per_case[name]["passed"] for name in crossing_cases):
        aggregate_failed.append("above_block_case_controls_failed")
    return {
        "num_layers": int(num_layers),
        "crossing_cases": crossing_cases,
        "per_case": per_case,
        "failed": aggregate_failed,
        "passed": not aggregate_failed,
    }


@contextmanager
def observe_int8_routes(events: list[dict[str, Any]]) -> Iterator[None]:
    """Record the registered writer/consumer the layer selects on each call."""

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer

    original = gemma4_layer._run_int8_attention

    def observed(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        route = gemma4_layer.last_int8_kv_route() or {}
        events.append({"rows": int(kwargs["rows"]), **route})
        return result

    gemma4_layer._run_int8_attention = observed
    try:
        yield
    finally:
        gemma4_layer._run_int8_attention = original


@contextmanager
def observe_int8_blocks(owner: Any, events: list[dict[str, Any]]) -> Iterator[None]:
    """Record every ``begin_block`` offset the runner stages for this owner."""

    original = owner.begin_block

    def observed(*args: Any, **kwargs: Any) -> Any:
        block = original(*args, **kwargs)
        events.append({"write_offset": int(block.write_offset), "rows": int(block.rows)})
        return block

    owner.begin_block = observed
    try:
        yield
    finally:
        owner.begin_block = original


def _capture_cases(
    runner: Any,
    workload: Sequence[FrozenCase],
    *,
    owner: Any | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, list[dict[str, Any]]]]]:
    """Teacher-force every case, returning rows and raw route/block events.

    Raw events are returned rather than summaries so ``evaluate_controls`` can
    check every per-call selection, not just whether an expected key appeared
    somewhere. The caller summarizes them for the artifact.
    """

    rows: dict[str, np.ndarray] = {}
    raw: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for case in workload:
        route_events: list[dict[str, Any]] = []
        block_events: list[dict[str, Any]] = []
        route_ctx = observe_int8_routes(route_events) if owner is not None else nullcontext()
        block_ctx = observe_int8_blocks(owner, block_events) if owner is not None else nullcontext()
        with route_ctx, block_ctx:
            rows[case.name] = capture_chain(
                runner, case.prompt_ids, case.prefill
            )
        if rows[case.name].shape[0] != case.scored_rows:
            raise RuntimeError(
                f"case {case.name}: captured {rows[case.name].shape[0]} rows, "
                f"workload froze {case.scored_rows}"
            )
        raw[case.name] = {"route_events": route_events, "block_events": block_events}
    return rows, raw


def _save_npz(
    path: Path,
    bf16_rows: dict[str, np.ndarray],
    bf16_repeat_rows: dict[str, np.ndarray],
    int8_rows: dict[str, np.ndarray],
    int8_repeat_rows: dict[str, np.ndarray],
    workload: Sequence[FrozenCase],
    provenance: dict[str, Any],
) -> str:
    """Write every arm's full-vocabulary logits outside the repository.

    Both captures of each arm are preserved, so the parent can independently
    check shape, dtype and raw-byte equality of the repeat pairs rather than
    trusting the recorded verdict.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for case in workload:
        arrays[f"bf16.{case.name}.logits"] = np.asarray(bf16_rows[case.name], dtype=np.float32)
        arrays[f"bf16_repeat.{case.name}.logits"] = np.asarray(
            bf16_repeat_rows[case.name], dtype=np.float32
        )
        arrays[f"int8.{case.name}.logits"] = np.asarray(int8_rows[case.name], dtype=np.float32)
        arrays[f"int8_repeat.{case.name}.logits"] = np.asarray(
            int8_repeat_rows[case.name], dtype=np.float32
        )
        arrays[f"{case.name}.prompt_ids"] = np.asarray(case.prompt_ids, dtype=np.int32)
    arrays["provenance"] = np.frombuffer(
        json.dumps(provenance, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez(path, **arrays)
    return sha256_file(path)


def _public_route_key_mismatches(
    route_events: Sequence[dict[str, Any]],
) -> dict[str, list[int]]:
    """Indices of route events whose keys disagree with their row shape."""

    writer_mismatches: list[int] = []
    consumer_mismatches: list[int] = []
    for index, event in enumerate(route_events):
        rows = int(event.get("rows", -1))
        if rows > 1:
            expected_writer, expected_consumer = EXPECTED_WRITER_PROMPT, EXPECTED_CONSUMER_PREFILL
        elif rows == 1:
            expected_writer, expected_consumer = EXPECTED_WRITER_DECODE, EXPECTED_CONSUMER_DECODE
        else:
            expected_writer = expected_consumer = None
        if expected_writer is None or str(event.get("writer")) != expected_writer:
            writer_mismatches.append(index)
        if expected_consumer is None or str(event.get("consumer")) != expected_consumer:
            consumer_mismatches.append(index)
    return {"writer": writer_mismatches, "consumer": consumer_mismatches}


def public_witness_verdict(
    route_events: Sequence[dict[str, Any]],
    runner_after: Any,
    *,
    runner_before: Any | None = None,
    prompt: str | None = None,
    max_tokens: int | None = None,
    generated_token_ids: Sequence[int] | None = None,
    generated_text: str | None = None,
    wall_s: float | None = None,
) -> dict[str, Any]:
    """Judge the runner state left by one real public request.

    The runner is reacquired from the generator *after* the request, so the
    verdict describes what the public call actually used rather than a runner
    captured before it. Every captured route event is validated against the key
    its row shape selects, not merely checked for the presence of an expected
    key, and the request must return the requested number of tokens with
    ``ignore_eos=True``. The generated text is a route witness, not an
    output-quality pass.
    """

    writers = sorted({str(event.get("writer")) for event in route_events})
    consumers = sorted({str(event.get("consumer")) for event in route_events})
    storage = getattr(runner_after, "kv_storage_resolved", None)
    uses_int8 = bool(getattr(runner_after, "uses_int8_kv", False))
    owner = getattr(runner_after, "kv_cache", None)
    generated = list(generated_token_ids or [])
    mismatches = _public_route_key_mismatches(route_events)
    decode_events = [event for event in route_events if int(event.get("rows", -1)) == 1]
    failed: list[str] = []
    if runner_after is None:
        failed.append("no_runner_after_request")
    if storage != "int8_per_token_head":
        failed.append("post_request_storage_not_int8")
    if not uses_int8:
        failed.append("post_request_runner_has_no_int8_owner")
    if owner is None:
        failed.append("post_request_owner_missing")
    if not route_events:
        failed.append("no_int8_route_observed")
    if not decode_events:
        failed.append("no_decode_route_observed")
    if mismatches["writer"]:
        failed.append("public_route_writer_mismatch")
    if mismatches["consumer"]:
        failed.append("public_route_consumer_mismatch")
    if max_tokens is not None and len(generated) != int(max_tokens):
        failed.append("public_generated_token_count")
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "generated_token_ids": generated,
        "generated_token_count": len(generated),
        "generated_text": generated_text,
        "wall_s": wall_s,
        "route_calls": len(route_events),
        "decode_route_calls": len(decode_events),
        "writers_observed": writers,
        "consumers_observed": consumers,
        "writer_mismatch_indices": mismatches["writer"],
        "consumer_mismatch_indices": mismatches["consumer"],
        "post_request_storage": storage,
        "post_request_uses_int8_kv": uses_int8,
        "post_request_owner_present": owner is not None,
        "same_runner_as_before": (
            None if runner_before is None else runner_after is runner_before
        ),
        "output_quality_claim": False,
        "failed": failed,
        "passed": not failed,
    }


def observe_public_witness(
    llm: Any,
    generator: Any,
    prompt: str,
    *,
    scale_dtype: str,
    max_tokens: int = 4,
    runner_before: Any | None = None,
) -> dict[str, Any]:
    """Run one real ``LLM.generate_detailed`` and judge the route it used."""

    from hipengine import SamplingParams

    params = SamplingParams(
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        kv_storage="int8_per_token_head",
        kv_scale_dtype=str(scale_dtype),
        kv_scale_granularity="per_token_head",
    )
    route_events: list[dict[str, Any]] = []
    started = time.perf_counter()
    with observe_int8_routes(route_events):
        outputs = llm.generate_detailed(prompt, params)
    wall_s = time.perf_counter() - started
    inner = getattr(generator, "_inner", generator)
    runner_after = getattr(inner, "_runner", None)
    first = outputs[0] if outputs else None
    return public_witness_verdict(
        route_events,
        runner_after,
        runner_before=runner_before,
        prompt=prompt,
        max_tokens=int(max_tokens),
        generated_token_ids=list(first.generated_token_ids) if first is not None else [],
        generated_text=first.text if first is not None else None,
        wall_s=round(wall_s, 3),
    )


def overall_verdict(
    quality: dict[str, Any],
    repeat_bf16: dict[str, Any],
    repeat_int8: dict[str, Any],
    controls: dict[str, Any],
    public: dict[str, Any],
) -> dict[str, Any]:
    """Combine every control and every category verdict into one pass/fail.

    The overall result is the conjunction of the combined quality envelope,
    every per-case category verdict, both literal-repeatability controls, the
    route/block execution controls, and the public-surface witness. A passing
    aggregate cannot mask a failed category, an inexact repeat, a wrong route,
    or a public request that used a different storage.
    """

    failed: list[str] = []
    if not quality["combined"]["passed"]:
        failed.append("combined_quality")
    for name, verdict in quality["per_case"].items():
        if not verdict["passed"]:
            failed.append(f"case_quality:{name}")
    if not repeat_bf16["passed"]:
        failed.append("repeat_bf16_not_byte_equal")
    if not repeat_int8["passed"]:
        failed.append("repeat_int8_not_byte_equal")
    if not controls["passed"]:
        failed.append("execution_controls")
    if not public["passed"]:
        failed.append("public_witness")
    return {"failed": failed, "passed": not failed}


def _source_hashes() -> dict[str, Any]:
    """SHA-256 of the INT8 writer, consumers, owner and runtime sources."""

    root = Path(__file__).resolve().parents[1]
    paths = {
        "int8_writer_hip": "hipengine/kernels/hip_gfx1100/attention/paged_kv_write.hip",
        "int8_writer_py": "hipengine/kernels/hip_gfx1100/attention/paged_kv_write.py",
        "int8_consumer_hip": "hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention_int8.hip",
        "int8_consumer_py": "hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention_int8.py",
        "int8_owner": "hipengine/runtime/gemma4_int8_kv.py",
        "runtime": "hipengine/runtime/gemma4.py",
        "layer": "hipengine/kernels/hip_gfx1100/gemma4/gemma4_layer.py",
        "generator": "hipengine/generation/gemma4_gguf.py",
    }
    return {
        name: {
            "path": relative,
            "sha256": sha256_file(root / relative) if (root / relative).exists() else None,
        }
        for name, relative in paths.items()
    }


def _json_safe(value: Any) -> Any:
    """Keep only JSON-representable values; stringify anything else."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _execution_identity(llm: Any, generator: Any, runner: Any) -> dict[str, Any]:
    """Record the implemented public LLM profile API without inventing identity.

    An omitted constructor/environment selector can resolve to production.
    An exposed API returning None identifies migration; an absent API is an
    unavailable observation, not proof of migration. This manifest describes
    the resolved model execution plan, not qualification of the INT8 KV layout.
    """

    missing = object()
    requested = getattr(llm, "execution_profile", missing)
    resolved = getattr(llm, "resolved_execution_profile", missing)
    manifest = getattr(llm, "execution_profile_manifest", missing)
    resolved_exposed = resolved is not missing
    manifest_exposed = manifest is not missing
    return {
        "requested_profile": None if requested is missing else _json_safe(requested),
        "requested_profile_status": (
            "unavailable: no public constructor/environment selector"
            if requested is missing else "constructor/environment selector"
        ),
        "resolved_profile": (
            {"attribute": "resolved_execution_profile", "value": _json_safe(resolved)}
            if resolved_exposed and resolved is not None else None
        ),
        "resolved_profile_status": (
            "unavailable: no public resolved_execution_profile property"
            if not resolved_exposed else
            "migration: no resolved named profile" if resolved is None else "exposed"
        ),
        "manifest": (
            {"attribute": "execution_profile_manifest", "value": _json_safe(manifest)}
            if manifest_exposed and manifest is not None else None
        ),
        "manifest_status": (
            "unavailable: no public execution_profile_manifest property"
            if not manifest_exposed else
            "migration: no resolved variant manifest" if manifest is None else "exposed"
        ),
        "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
        "strict_manifest_sha256": getattr(llm, "execution_profile_strict_manifest_sha256", None),
        "fell_back_to_strict": getattr(llm, "execution_profile_fell_back_to_strict", None),
        "legacy_path": (resolved is None) if resolved_exposed else None,
        "scope": "resolved model execution plan; not INT8 KV numerical qualification",
    }


def _arm_runner(generator: Any, storage: str, scale_dtype: str) -> Any:
    if storage == "bf16":
        return generator._ensure_runner(storage="bf16")
    return generator._ensure_runner(
        storage="int8_per_token_head",
        scale_dtype=str(scale_dtype),
        granularity="per_token_head",
    )


def run(args: argparse.Namespace) -> int:
    from scripts.gemma4_campaign_bench import _resolve_generator

    artifact = Path(args.artifact)
    started = time.time()
    llm, runner0, loading = _resolve_generator(artifact, args.context)
    generator = llm._get_text_generator()
    try:
        max_block = int(runner0.max_block)
        config = runner0.weights.config
        attention_geometry = [
            {
                "layer_type": str(attention.layer_type),
                "num_heads": int(attention.num_heads),
                "num_kv_heads": int(attention.num_kv_heads),
                "head_dim": int(attention.head_dim),
                "sliding_window": (
                    None if attention.sliding_window is None else int(attention.sliding_window)
                ),
                "k_eq_v": bool(attention.k_eq_v),
            }
            for attention in config.attention
        ]
        workload = freeze_workload(
            generator.tokenize, max_block=max_block, scored_rows=args.scored_rows
        )
        workload_manifest = {
            "kind": "gemma4_d12_int8_kv_workload",
            "frozen_before_capture": True,
            "max_block": max_block,
            "scored_rows_per_case": int(args.scored_rows),
            "thresholds": dict(THRESHOLDS),
            "contract_source": CONTRACT_SOURCE,
            "cases": [
                {
                    "name": case.name,
                    "category": case.category,
                    "prompt_tokens": case.prompt_tokens,
                    "prefill": case.prefill,
                    "scored_rows": case.scored_rows,
                    "chain_sha256": case.chain_sha256,
                    "prompt_ids": list(case.prompt_ids),
                }
                for case in workload
            ],
        }
        if args.workload_out:
            args.workload_out.parent.mkdir(parents=True, exist_ok=True)
            args.workload_out.write_text(json.dumps(workload_manifest, indent=1, sort_keys=True) + "\n")

        # --- BF16 reference arm -------------------------------------------
        names = [case.name for case in workload]
        bf16_runner = _arm_runner(generator, "bf16", args.scale_dtype)
        bf16_owned = owned_allocation_bytes(bf16_runner)
        bf16_device_after_build = device_usage_bytes()
        bf16_rows, _ = _capture_cases(bf16_runner, workload)
        bf16_rows_repeat, _ = _capture_cases(bf16_runner, workload)
        bf16_device_after_capture = device_usage_bytes()
        repeat_bf16 = check_repeatability(bf16_rows, bf16_rows_repeat, order=names)
        repeat_bf16_kl = evaluate_cases(bf16_rows, bf16_rows_repeat, order=names)

        # --- INT8 arm -----------------------------------------------------
        int8_runner = _arm_runner(generator, "int8_per_token_head", args.scale_dtype)
        assert int8_runner.uses_int8_kv, "the INT8 arm did not take the INT8 owner"
        int8_owned = owned_allocation_bytes(int8_runner)
        int8_device_after_build = device_usage_bytes()
        int8_rows, int8_raw = _capture_cases(
            int8_runner, workload, owner=int8_runner.kv_cache
        )
        int8_rows_repeat, _ = _capture_cases(
            int8_runner, workload, owner=int8_runner.kv_cache
        )
        int8_device_after_capture = device_usage_bytes()
        repeat_int8 = check_repeatability(int8_rows, int8_rows_repeat, order=names)
        repeat_int8_kl = evaluate_cases(int8_rows, int8_rows_repeat, order=names)

        route_witness = {
            name: summarize_route_witness(int8_raw[name]["route_events"])
            for name in names
        }
        block_witness = {
            name: summarize_block_witness(int8_raw[name]["block_events"])
            for name in names
        }
        controls = evaluate_controls(
            workload,
            max_block=max_block,
            num_layers=len(config.attention),
            route_events={name: int8_raw[name]["route_events"] for name in names},
            block_events={name: int8_raw[name]["block_events"] for name in names},
        )

        quality = evaluate_cases(bf16_rows, int8_rows, order=names)

        # --- public-surface witness --------------------------------------
        public = observe_public_witness(
            llm,
            generator,
            args.witness_prompt,
            scale_dtype=args.scale_dtype,
            runner_before=int8_runner,
        )

        overall = overall_verdict(quality, repeat_bf16, repeat_int8, controls, public)

        provenance = _provenance(artifact, loading)
        provenance["metric_evaluator_source_sha256"] = provenance.get(
            "evaluator_source_sha256"
        )
        provenance["wrapper_source_sha256"] = sha256_file(Path(__file__))
        provenance["git_head"] = provenance.get("git_commit")
        provenance["command"] = " ".join([sys.executable, *sys.argv])
        provenance["physical_host"] = {
            "hostname": socket.gethostname(),
            "platform_node": os.uname().nodename,
        }
        provenance["gpu_binding"] = {
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "device0_name": provenance.get("gpu_name_device0"),
            "note": (
                "the device binding is the environment recorded above; logical "
                "device 0 is whichever physical GPU ROCR_VISIBLE_DEVICES / "
                "HIP_VISIBLE_DEVICES select on this host, and its name is "
                "device0_name"
            ),
        }
        artifact_bytes = (provenance.get("artifact") or {}).get("bytes")
        provenance["artifact_hash_scope"] = (
            "first 64 MiB SHA-256 only; the full "
            f"{int(artifact_bytes)}-byte artifact is not hashed"
            if artifact_bytes is not None
            else "first 64 MiB SHA-256 only; full-file byte length unavailable"
        )
        provenance["int8_source_sha256"] = _source_hashes()
        provenance["execution_identity"] = _execution_identity(
            llm, generator, int8_runner
        )

        npz_sha = None
        if args.npz:
            npz_sha = _save_npz(
                Path(args.npz),
                bf16_rows,
                bf16_rows_repeat,
                int8_rows,
                int8_rows_repeat,
                workload,
                provenance,
            )

        record = {
            "kind": "gemma4_d12_int8_kv_teacher_forced",
            "performance_claim": False,
            "diagnostic_only": True,
            "promotion_qualified": False,
            "int8_versus": "bf16",
            "kl_direction": "kl(bf16 || int8)",
            "contract_source": CONTRACT_SOURCE,
            "thresholds": dict(THRESHOLDS),
            "artifact": str(artifact),
            "max_block": max_block,
            "attention_geometry": attention_geometry,
            "mask_semantics": (
                "causal (key <= query) plus the per-layer sliding window; the "
                "INT8 route rejects a non-causal keep mask and reads no evict "
                "mask"
            ),
            "scale_dtype": str(args.scale_dtype),
            "workload": workload_manifest,
            "quality": quality,
            "quality_passed": quality_passed(quality),
            "repeatability_bf16": repeat_bf16,
            "repeatability_int8": repeat_int8,
            "kl_repeat_bf16": repeat_bf16_kl,
            "kl_repeat_int8": repeat_int8_kl,
            "execution_controls": controls,
            "route_witness": route_witness,
            "block_witness": block_witness,
            "public_witness": public,
            "overall": overall,
            "allocations": {
                "bf16": {
                    **bf16_owned,
                    "device_after_build": bf16_device_after_build,
                    "device_after_capture": bf16_device_after_capture,
                },
                "int8_per_token_head": {
                    **int8_owned,
                    "device_after_build": int8_device_after_build,
                    "device_after_capture": int8_device_after_capture,
                },
                "device_scope": (
                    "external hipMemGetInfo used/free, which includes the ~17 GB "
                    "resident weights and every other allocation on the device; "
                    "reported for context only, never as owned bytes"
                ),
            },
            "npz_path": str(args.npz) if args.npz else None,
            "npz_sha256": npz_sha,
            "passed": overall["passed"],
            "run_seconds": round(time.time() - started, 1),
            "provenance": provenance,
        }
        text = json.dumps(record, indent=1, sort_keys=True)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        return 0 if overall["passed"] else 1
    finally:
        llm.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="freeze, capture both arms, and judge")
    run_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    run_parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    run_parser.add_argument("--scored-rows", type=int, default=DEFAULT_SCORED_ROWS)
    run_parser.add_argument(
        "--scale-dtype", choices=("fp16", "fp32"), default="fp16",
        help="INT8 KV scale plane dtype (both are registered)",
    )
    run_parser.add_argument(
        "--witness-prompt", default="The capital of France is",
        help="short prompt for the real LLM.generate public witness",
    )
    run_parser.add_argument("--out", type=Path, help="compact diagnostic JSON to write")
    run_parser.add_argument(
        "--workload-out", type=Path, help="frozen workload manifest (token ids) to write"
    )
    run_parser.add_argument(
        "--npz", type=Path,
        help="full-logits npz path outside the repository (recorded by sha256)",
    )
    args = parser.parse_args(argv)
    if args.scored_rows < 1:
        parser.error("--scored-rows must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
