#!/usr/bin/env python3
"""Qualify host-materialized, eager-native GGUF MTP beyond the short graph.

The direct gate compares the production eager-native target verifier with the
serial-exact teacher at identical root/candidate rows.  It checks full logits,
all Conv/GDN state surfaces, touched BF16 K/V rows, the selected hidden row,
cursors, rollback, and long-context split-K ownership.  ``--generation-contexts``
adds a real NextN provider run with cycle logits enabled; that setting forces a
host proposal and prevents proposal/target graph submission.

This is a correctness/oracle gate, not a performance benchmark.  The reusable
N1/N2 graph remains deliberately capped below 1024 until RF2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-27B-Q4_K_M.gguf")
DEFAULT_SEED = (
    7734,
    264,
    12654,
    709,
    421,
    4523,
    279,
    307,
    7324,
    76938,
    1324,
    1608,
    20781,
    1954,
    13,
)
_REQUIRED_RESULT_BOOLEANS = (
    "target_logits_exact",
    "linear_state_exact",
    "kv_rows_exact",
    "hidden_exact",
    "cursor_exact",
    "commit_exact",
    "rollback_exact",
)

ProgressCallback = Callable[[str, dict[str, Any]], None]
ResultCallback = Callable[[str, dict[str, Any]], None]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one checkpoint/artifact without leaving partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _notify_progress(
    callback: ProgressCallback | None,
    event: str,
    details: dict[str, Any],
) -> None:
    if callback is not None:
        callback(str(event), dict(details))


@dataclass(frozen=True, slots=True)
class EagerMTPCase:
    """One target-verifier shape keyed by its cycle end, not prompt length."""

    cycle_end: int
    candidate_budget: int
    expected_accepted_count: int | None = None

    def __post_init__(self) -> None:
        if int(self.candidate_budget) not in {1, 2, 3}:
            raise ValueError("candidate budget must be 1, 2, or 3")
        if int(self.cycle_end) <= self.rows:
            raise ValueError("cycle end must leave a positive verifier start position")
        if self.expected_accepted_count is not None and not (
            0 <= int(self.expected_accepted_count) <= int(self.candidate_budget)
        ):
            raise ValueError("expected accepted count must be within the candidate budget")

    @property
    def rows(self) -> int:
        return int(self.candidate_budget) + 1

    @property
    def start_position(self) -> int:
        return int(self.cycle_end) - self.rows

    @property
    def case_id(self) -> str:
        base = f"end{int(self.cycle_end)}-b{int(self.candidate_budget)}"
        if self.expected_accepted_count is None:
            return base
        return f"{base}-a{int(self.expected_accepted_count)}"


def _parse_scaled_int(raw: str) -> int:
    text = raw.strip().lower().replace("_", "")
    scale = 1
    if text.endswith("k"):
        text = text[:-1]
        scale = 1024
    if not text or not text.isdigit():
        raise ValueError(f"invalid token count {raw!r}")
    value = int(text) * scale
    if value <= 0:
        raise ValueError("token counts must be positive")
    return value


def parse_token_spec(raw: str) -> tuple[int, ...]:
    """Parse ordered CSV token counts, inclusive ranges, and binary K suffixes."""

    values: list[int] = []
    seen: set[int] = set()
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid token range {item!r}")
            start = _parse_scaled_int(parts[0])
            end = _parse_scaled_int(parts[1])
            if end < start:
                raise ValueError("token ranges must be ascending")
            expanded: Iterable[int] = range(start, end + 1)
        else:
            expanded = (_parse_scaled_int(item),)
        for value in expanded:
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise ValueError("at least one positive token count is required")
    return tuple(values)


def build_cycle_cases(
    *,
    cycle_ends: Sequence[int],
    candidate_budgets: Sequence[int],
) -> tuple[EagerMTPCase, ...]:
    cases: list[EagerMTPCase] = []
    for cycle_end in cycle_ends:
        for budget in candidate_budgets:
            cases.append(
                EagerMTPCase(
                    cycle_end=int(cycle_end),
                    candidate_budget=int(budget),
                )
            )
    return tuple(cases)


def build_acceptance_cases(
    *,
    cycle_ends: Sequence[int],
    candidate_budget: int = 3,
) -> tuple[EagerMTPCase, ...]:
    budget = int(candidate_budget)
    return tuple(
        EagerMTPCase(
            cycle_end=int(cycle_end),
            candidate_budget=budget,
            expected_accepted_count=accepted,
        )
        for cycle_end in cycle_ends
        for accepted in range(budget + 1)
    )


def gate_passed(
    results: Sequence[dict[str, Any]],
    *,
    require_target_graph: bool = False,
) -> bool:
    """Fail closed unless rows prove the requested owner and exact surfaces."""

    if not results:
        return False
    for result in results:
        if not bool(result.get("passed", False)):
            return False
        graph_submitted = bool(result.get("target_native_graph_submitted", False))
        if graph_submitted is not bool(require_target_graph):
            return False
        if not bool(result.get("host_target_batch_materialized", False)):
            return False
        if result.get("target_verify_route") != "eager_native":
            return False
        required_fields = (
            tuple(
                field
                for field in _REQUIRED_RESULT_BOOLEANS
                if field not in {"kv_rows_exact", "cursor_exact"}
            )
            if require_target_graph
            else _REQUIRED_RESULT_BOOLEANS
        )
        if any(not bool(result.get(field, False)) for field in required_fields):
            return False
        expected_accepted = result.get("expected_accepted_count")
        if expected_accepted is not None and int(result.get("accepted_count", -1)) != int(
            expected_accepted
        ):
            return False
        if (
            int(result.get("cycle_end", 0)) >= 1024
            and int(result.get("split_k_calls", 0)) <= 0
            and not graph_submitted
        ):
            return False
    return True


def _target_batch(root: int, position: int, candidates: Sequence[int]):
    from hipengine.speculative import DraftBatch, TargetVerifyBatch

    request_id = 17
    tokens = tuple(int(token) for token in candidates)
    draft = DraftBatch(
        request_ids=(request_id,),
        candidate_tokens=tokens,
        parent_positions=tuple(int(position) + depth for depth in range(len(tokens))),
        draft_depths=tuple(range(1, len(tokens) + 1)),
        row_to_request=(request_id,) * len(tokens),
        mode="verify_chain",
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(int(root),),
        root_positions=(int(position),),
    )


def _commit_plan(prepared: Any):
    from hipengine.speculative import TargetCommitPlan

    summary = prepared.summary
    batch = prepared.batch
    return TargetCommitPlan(
        transaction_id=int(prepared.buffers.transaction_id),
        request_ids=batch.request_ids,
        accepted_counts=summary.accepted_counts,
        commit_rows=summary.commit_rows,
        commit_tokens=summary.commit_tokens,
        commit_positions=summary.commit_positions,
        next_tokens=summary.next_tokens,
        candidate_counts=batch.candidate_counts,
        draft_depth=batch.draft_depth,
        tree_shape=batch.tree_shape,
        mode=batch.mode,
    )


def _wrong_token(token: int, vocab_size: int) -> int:
    return (int(token) + 1) % int(vocab_size)


def _candidate_tokens(
    case: EagerMTPCase,
    *,
    teacher_candidates: Sequence[int] | None = None,
    vocab_size: int | None = None,
) -> tuple[int, ...]:
    if case.expected_accepted_count is None:
        offset = int(case.start_position) % len(DEFAULT_SEED)
        return tuple(
            int(DEFAULT_SEED[(offset + index) % len(DEFAULT_SEED)])
            for index in range(int(case.candidate_budget))
        )
    if teacher_candidates is None or len(teacher_candidates) != int(
        case.candidate_budget
    ):
        raise ValueError("acceptance-controlled cases require complete teacher candidates")
    candidates = [int(token) for token in teacher_candidates]
    accepted = int(case.expected_accepted_count)
    if accepted < int(case.candidate_budget):
        if vocab_size is None:
            raise ValueError("acceptance-controlled rejection requires vocab size")
        candidates[accepted] = _wrong_token(candidates[accepted], int(vocab_size))
    return tuple(candidates)


def _teacher_candidates(
    verifier: Any,
    *,
    root_token: int,
    case: EagerMTPCase,
    transaction_base: int,
) -> tuple[int, ...]:
    candidates: list[int] = []
    for depth in range(1, int(case.candidate_budget) + 1):
        trial = (*candidates, int(DEFAULT_SEED[depth % len(DEFAULT_SEED)]))
        batch = _target_batch(root_token, case.start_position, trial)
        prepared = verifier.prepare(
            batch,
            transaction_id=int(transaction_base) + depth,
            graph_bucket=verifier.graph_bucket(
                ("rf1-teacher", case.start_position, case.candidate_budget, depth),
                batch,
            ),
            remaining_decode=(batch.rows,),
            return_logits=False,
        )
        try:
            candidates.append(int(prepared.target_top1[depth - 1]))
        finally:
            verifier.rollback(prepared)
    return tuple(candidates)


def _prompt(length: int) -> tuple[int, ...]:
    return tuple(int(DEFAULT_SEED[index % len(DEFAULT_SEED)]) for index in range(int(length)))


def _buffer_nbytes(buffer: Any) -> int:
    declared = getattr(buffer, "nbytes", None)
    if declared is not None:
        return int(declared)
    return int(buffer.numel) * int(buffer.dtype.itemsize)


def _copy_bytes(buffer: Any, *, offset: int = 0, nbytes: int | None = None) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    available = _buffer_nbytes(buffer)
    count = available - int(offset) if nbytes is None else int(nbytes)
    if count < 0 or int(offset) < 0 or int(offset) + count > available:
        raise ValueError("device copy slice is outside its buffer")
    out = np.empty((count,), dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(int(buffer.ptr) + int(offset), count),
        count,
    )
    return out


def _linear_state_bytes(session: Any) -> tuple[np.ndarray | None, ...]:
    owner = session._target_scratch_owner
    if owner is None:
        raise RuntimeError("resident target scratch owner is closed")
    return tuple(
        None if state is None else _copy_bytes(state)
        for state in (*owner.layer_conv_states, *owner.layer_recurrent_states)
    )


def _hidden_bytes(session: Any) -> np.ndarray:
    hidden = session.last_target_hidden
    return _copy_bytes(hidden, nbytes=int(hidden.shape[1]) * int(hidden.dtype.itemsize))


def _state_mismatch_indices(left: Any, right: Any) -> list[int]:
    left_states = _linear_state_bytes(left)
    right_states = _linear_state_bytes(right)
    return [
        index
        for index, (a, b) in enumerate(zip(left_states, right_states, strict=True))
        if not (
            (a is None and b is None)
            or (a is not None and b is not None and np.array_equal(a, b))
        )
    ]


def _states_equal(left: Any, right: Any) -> bool:
    left_states = _linear_state_bytes(left)
    right_states = _linear_state_bytes(right)
    return all(
        (a is None and b is None)
        or (a is not None and b is not None and np.array_equal(a, b))
        for a, b in zip(left_states, right_states, strict=True)
    )


def _state_snapshot_equal(session: Any, snapshot: Sequence[np.ndarray | None]) -> bool:
    current = _linear_state_bytes(session)
    return all(
        (a is None and b is None)
        or (a is not None and b is not None and np.array_equal(a, b))
        for a, b in zip(current, snapshot, strict=True)
    )


def _first_kv_row_mismatch(
    left: Any,
    right: Any,
    positions: Sequence[int],
) -> dict[str, int] | None:
    left_owner = left._target_scratch_owner
    right_owner = right._target_scratch_owner
    if left_owner is None or right_owner is None:
        raise RuntimeError("resident target scratch owner is closed")
    unique_positions = tuple(dict.fromkeys(int(position) for position in positions))
    left_caches = (*left_owner.full_key_caches, *left_owner.full_value_caches)
    right_caches = (*right_owner.full_key_caches, *right_owner.full_value_caches)
    key_count = len(left_owner.full_key_caches)
    for cache_index, (left_cache, right_cache) in enumerate(
        zip(left_caches, right_caches, strict=True)
    ):
        if left_cache is None or right_cache is None:
            if left_cache is not None or right_cache is not None:
                return {"cache_index": cache_index, "position": -1}
            continue
        row_nbytes = int(left_cache.nbytes) // int(left_owner.max_positions)
        if row_nbytes != int(right_cache.nbytes) // int(right_owner.max_positions):
            return {"cache_index": cache_index, "position": -1}
        for position in unique_positions:
            offset = position * row_nbytes
            if not np.array_equal(
                _copy_bytes(left_cache, offset=offset, nbytes=row_nbytes),
                _copy_bytes(right_cache, offset=offset, nbytes=row_nbytes),
            ):
                return {
                    "cache_index": cache_index,
                    "layer": cache_index if cache_index < key_count else cache_index - key_count,
                    "plane": 0 if cache_index < key_count else 1,
                    "position": int(position),
                }
    return None


def _kv_rows_equal(left: Any, right: Any, positions: Sequence[int]) -> bool:
    left_owner = left._target_scratch_owner
    right_owner = right._target_scratch_owner
    if left_owner is None or right_owner is None:
        raise RuntimeError("resident target scratch owner is closed")
    unique_positions = tuple(dict.fromkeys(int(position) for position in positions))
    for left_cache, right_cache in zip(
        (*left_owner.full_key_caches, *left_owner.full_value_caches),
        (*right_owner.full_key_caches, *right_owner.full_value_caches),
        strict=True,
    ):
        if left_cache is None or right_cache is None:
            if left_cache is not None or right_cache is not None:
                return False
            continue
        left_row_nbytes = int(left_cache.nbytes) // int(left_owner.max_positions)
        right_row_nbytes = int(right_cache.nbytes) // int(right_owner.max_positions)
        if left_row_nbytes != right_row_nbytes:
            return False
        for position in unique_positions:
            if position < 0 or position >= int(left_owner.max_positions):
                return False
            offset = position * left_row_nbytes
            if not np.array_equal(
                _copy_bytes(left_cache, offset=offset, nbytes=left_row_nbytes),
                _copy_bytes(right_cache, offset=offset, nbytes=right_row_nbytes),
            ):
                return False
    return True


@contextmanager
def _count_split_k_calls() -> Iterator[list[str]]:
    import hipengine.runtime.qwen35_gguf_runner as runner_module

    names = (
        "qwen35_paged_full_attn_decode_split_k_gate_bf16_spans",
        "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_spans",
        "qwen35_paged_full_attn_decode_split_k_gqa_gate_bf16_parallel_reduce_spans",
        "qwen35_paged_full_attn_decode_split_k_warp_gate_bf16_spans",
    )
    originals = {name: getattr(runner_module, name) for name in names}
    calls: list[str] = []

    def wrapper(name: str):
        original = originals[name]

        def counted(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return counted

    try:
        for name in names:
            setattr(runner_module, name, wrapper(name))
        yield calls
    finally:
        for name, original in originals.items():
            setattr(runner_module, name, original)


def _run_direct_cases(
    model: Path,
    cases: Sequence[EagerMTPCase],
    *,
    max_sequence_length: int,
    require_cached_build: bool,
    require_target_graph: bool = False,
    direct_remaining_decode: int | None = None,
    progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
) -> list[dict[str, Any]]:
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFTransactionalVerifier
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    grouped: dict[int, list[EagerMTPCase]] = defaultdict(list)
    for case in cases:
        grouped[int(case.start_position)].append(case)
    results: list[dict[str, Any]] = []
    max_budget = max(int(case.candidate_budget) for case in cases)
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=int(max_sequence_length),
        require_cached_build=bool(require_cached_build),
    ) as native:
        native.select_prefill_quant("gguf_q4_k_m")
        if native.runner is None:
            raise RuntimeError("native session did not retain its runner")
        with Qwen35GGUFResidentSession(
            model,
            max_sequence_length=int(max_sequence_length),
            shared_runner=native.runner,
            require_cached_build=bool(require_cached_build),
        ) as strict, Qwen35GGUFTransactionalVerifier(
            native,
            max_candidate_budget=max_budget,
            quant="gguf_q4_k_m",
            target_verify_mode="native",
        ) as native_verifier, Qwen35GGUFTransactionalVerifier(
            strict,
            max_candidate_budget=max_budget,
            quant="gguf_q4_k_m",
            target_verify_mode="serial_exact",
        ) as strict_verifier:
            for start_position in sorted(grouped):
                group_cases = grouped[start_position]
                _notify_progress(
                    progress,
                    "direct_prefill_start",
                    {
                        "start_position": int(start_position),
                        "case_ids": [case.case_id for case in group_cases],
                    },
                )
                native.reset()
                strict.reset()
                prompt = _prompt(start_position)
                native_root = int(
                    native.prefill(prompt, use_bulk=True, return_logits=False).token_id
                )
                strict_root = int(
                    strict.prefill(prompt, use_bulk=True, return_logits=False).token_id
                )
                if native_root != strict_root:
                    raise RuntimeError(
                        f"prefill root mismatch at {start_position}: "
                        f"native={native_root} strict={strict_root}"
                    )
                _notify_progress(
                    progress,
                    "direct_prefill_complete",
                    {
                        "start_position": int(start_position),
                        "case_ids": [case.case_id for case in group_cases],
                    },
                )
                initial_state = _linear_state_bytes(native)
                initial_hidden = _hidden_bytes(native)
                teacher_cache: dict[int, tuple[int, ...]] = {}
                for case in group_cases:
                    _notify_progress(
                        progress,
                        "direct_case_start",
                        {
                            "case_id": case.case_id,
                            "cycle_end": int(case.cycle_end),
                            "candidate_budget": int(case.candidate_budget),
                        },
                    )
                    started = time.perf_counter()
                    result: dict[str, Any] = {
                        **asdict(case),
                        "case_id": case.case_id,
                        "rows": case.rows,
                        "start_position": case.start_position,
                        "host_target_batch_materialized": True,
                        "host_proposal_materialized": False,
                        "target_verify_route": "eager_native",
                        "target_native_graph_submitted": False,
                    }
                    native_prepared = None
                    strict_prepared = None
                    try:
                        teacher_candidates = None
                        if case.expected_accepted_count is not None:
                            teacher_candidates = teacher_cache.get(
                                int(case.candidate_budget)
                            )
                            if teacher_candidates is None:
                                teacher_candidates = _teacher_candidates(
                                    strict_verifier,
                                    root_token=strict_root,
                                    case=case,
                                    transaction_base=30_000 + 100 * len(results),
                                )
                                teacher_cache[int(case.candidate_budget)] = (
                                    teacher_candidates
                                )
                        batch = _target_batch(
                            native_root,
                            case.start_position,
                            _candidate_tokens(
                                case,
                                teacher_candidates=teacher_candidates,
                                vocab_size=int(native.runner.vocab_size),
                            ),
                        )
                        with _count_split_k_calls() as split_calls:
                            native_prepared = native_verifier.prepare(
                                batch,
                                transaction_id=10_000 + len(results),
                                graph_bucket=native_verifier.graph_bucket(
                                    ("rf1-native", case.case_id), batch
                                ),
                                remaining_decode=(
                                    case.rows
                                    if direct_remaining_decode is None
                                    else int(direct_remaining_decode),
                                ),
                                return_logits=not bool(require_target_graph),
                            )
                        strict_prepared = strict_verifier.prepare(
                            batch,
                            transaction_id=20_000 + len(results),
                            graph_bucket=strict_verifier.graph_bucket(
                                ("rf1-strict", case.case_id), batch
                            ),
                            remaining_decode=(
                                case.rows
                                if direct_remaining_decode is None
                                else int(direct_remaining_decode),
                            ),
                            return_logits=not bool(require_target_graph),
                        )
                        touched_positions = tuple(int(value) for value in batch.positions)
                        result.update(
                            {
                                "split_k_calls": len(split_calls),
                                "split_k_kernels": sorted(set(split_calls)),
                                "target_native_graph_submitted": bool(
                                    native_prepared.native_graph_submitted
                                ),
                                "target_native_graph_fallback_reason": (
                                    native_prepared.native_graph_fallback_reason
                                ),
                                "target_native_graph_capture_ms": float(
                                    native_prepared.native_graph_capture_ms
                                ),
                                "target_native_graph_submit_ms": float(
                                    native_prepared.native_graph_submit_ms
                                ),
                                "target_native_graph_readback_ms": float(
                                    native_prepared.native_graph_readback_ms
                                ),
                                "target_logits_exact": bool(
                                    np.array_equal(
                                        native_prepared.target_logits,
                                        strict_prepared.target_logits,
                                    )
                                ),
                                "target_top1_exact": bool(
                                    native_prepared.target_top1
                                    == strict_prepared.target_top1
                                ),
                                "accept_summary_exact": bool(
                                    native_prepared.summary.accepted_counts
                                    == strict_prepared.summary.accepted_counts
                                    and native_prepared.summary.commit_rows
                                    == strict_prepared.summary.commit_rows
                                    and native_prepared.summary.commit_tokens
                                    == strict_prepared.summary.commit_tokens
                                    and native_prepared.summary.commit_positions
                                    == strict_prepared.summary.commit_positions
                                ),
                                # Native deferred-state ownership intentionally
                                # leaves resident Conv/GDN at the initial row until
                                # commit; compare selected state/hidden after both
                                # verifiers commit their independently captured row.
                                "kv_rows_exact": _kv_rows_equal(
                                    native, strict, touched_positions
                                ),
                                "cursor_exact": bool(
                                    int(native.position)
                                    == int(strict.position)
                                    == int(case.cycle_end)
                                ),
                            }
                        )
                        scratch = native.scratch
                        if scratch is None:
                            raise RuntimeError("native scratch closed during gate")
                        required_splits = (
                            int(case.cycle_end) + int(scratch.block_size) - 1
                        ) // int(scratch.block_size)
                        result.update(
                            {
                                "block_size": int(scratch.block_size),
                                "required_split_count": required_splits,
                                "workspace_split_count": int(
                                    scratch.full_attn_split_count
                                ),
                                "split_workspace_sufficient": bool(
                                    int(scratch.full_attn_split_count)
                                    >= required_splits
                                ),
                            }
                        )
                        native_verifier.commit(
                            native_prepared,
                            _commit_plan(native_prepared),
                        )
                        strict_verifier.commit(
                            strict_prepared,
                            _commit_plan(strict_prepared),
                        )
                        committed_position = int(
                            native_prepared.summary.commit_positions[0]
                        ) + 1
                        committed_position_exact = bool(
                            int(native.position)
                            == int(strict.position)
                            == committed_position
                        )
                        state_mismatches = _state_mismatch_indices(native, strict)
                        committed_linear_exact = not state_mismatches
                        first_kv_mismatch = _first_kv_row_mismatch(
                            native, strict, touched_positions
                        )
                        committed_kv_exact = first_kv_mismatch is None
                        native_hidden = _hidden_bytes(native)
                        strict_hidden = _hidden_bytes(strict)
                        committed_hidden_exact = np.array_equal(
                            native_hidden, strict_hidden
                        )
                        result.update(
                            {
                                "accepted_count": int(
                                    native_prepared.summary.accepted_counts[0]
                                ),
                                "acceptance_exact": bool(
                                    case.expected_accepted_count is None
                                    or int(
                                        native_prepared.summary.accepted_counts[0]
                                    )
                                    == int(case.expected_accepted_count)
                                ),
                                "committed_position": committed_position,
                                "commit_position_exact": committed_position_exact,
                                "linear_state_exact": committed_linear_exact,
                                "state_mismatch_indices": state_mismatches[:8],
                                "native_state_matches_initial": _state_snapshot_equal(
                                    native, initial_state
                                ),
                                "commit_kv_exact": committed_kv_exact,
                                "first_kv_mismatch": first_kv_mismatch,
                                "hidden_exact": committed_hidden_exact,
                                "hidden_mismatch_bytes": int(
                                    np.count_nonzero(native_hidden != strict_hidden)
                                ),
                                "native_hidden_matches_initial": bool(
                                    np.array_equal(native_hidden, initial_hidden)
                                ),
                                "commit_exact": bool(
                                    committed_position_exact
                                    and committed_linear_exact
                                    and committed_kv_exact
                                    and committed_hidden_exact
                                ),
                            }
                        )
                    except Exception as exc:
                        result["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        if native_prepared is not None:
                            native_verifier.rollback(native_prepared)
                        if strict_prepared is not None:
                            strict_verifier.rollback(strict_prepared)
                    result["rollback_exact"] = bool(
                        int(native.position)
                        == int(strict.position)
                        == int(case.start_position)
                        and _state_snapshot_equal(native, initial_state)
                        and np.array_equal(_hidden_bytes(native), initial_hidden)
                        and _states_equal(native, strict)
                    )
                    result["wall_seconds"] = time.perf_counter() - started
                    result["passed"] = bool(
                        "error" not in result
                        and bool(result["target_native_graph_submitted"])
                        is bool(require_target_graph)
                        and result.get("target_logits_exact", False)
                        and result.get("target_top1_exact", False)
                        and result.get("accept_summary_exact", False)
                        and result.get("linear_state_exact", False)
                        and (
                            bool(require_target_graph)
                            or result.get("kv_rows_exact", False)
                        )
                        and result.get("hidden_exact", False)
                        and (
                            bool(require_target_graph)
                            or result.get("cursor_exact", False)
                        )
                        and result.get("commit_exact", False)
                        and result.get("acceptance_exact", False)
                        and result.get("rollback_exact", False)
                        and result.get("split_workspace_sufficient", False)
                        and (
                            int(case.cycle_end) < 1024
                            or int(result.get("split_k_calls", 0)) > 0
                            or bool(result.get("target_native_graph_submitted", False))
                        )
                    )
                    results.append(result)
                    if on_result is not None:
                        on_result("direct", result)
                    _notify_progress(
                        progress,
                        "direct_case_complete",
                        {
                            "case_id": case.case_id,
                            "passed": bool(result["passed"]),
                            "wall_seconds": float(result["wall_seconds"]),
                        },
                    )
    return results


def _run_generation_contexts(
    model: Path,
    contexts: Sequence[int],
    *,
    candidate_budget: int,
    max_new_tokens: int,
    max_sequence_length: int,
    require_cached_build: bool,
    require_target_graph: bool = False,
    progress: ProgressCallback | None = None,
    on_result: ResultCallback | None = None,
) -> list[dict[str, Any]]:
    from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession
    from hipengine.runtime.qwen35_gguf_nextn import (
        Qwen35GGUFNextNDraftProvider,
        borrow_qwen35_gguf_nextn_fallback_weights,
    )
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    results: list[dict[str, Any]] = []
    with Qwen35GGUFResidentSession(
        model,
        max_sequence_length=int(max_sequence_length),
        require_cached_build=bool(require_cached_build),
    ) as target:
        target.select_prefill_quant("gguf_q4_k_m")
        borrowed = borrow_qwen35_gguf_nextn_fallback_weights(target)
        provider = Qwen35GGUFNextNDraftProvider.from_model(
            model,
            max_positions=int(max_sequence_length),
            max_requests=1,
            runtime=target.runtime,
            require_cached_build=bool(require_cached_build),
            borrowed_fallback_weights=borrowed,
        )
        try:
            with Qwen35GGUFMTPDecodeSession(
                target,
                provider,
                candidate_budget=int(candidate_budget),
                quant="gguf_q4_k_m",
                target_verify_mode="native",
            ) as decoder:
                for context in contexts:
                    _notify_progress(
                        progress,
                        "generation_case_start",
                        {
                            "context_tokens": int(context),
                            "candidate_budget": int(candidate_budget),
                            "max_new_tokens": int(max_new_tokens),
                        },
                    )
                    started = time.perf_counter()
                    prompt = _prompt(int(context))
                    target.reset()
                    ar_started = time.perf_counter()
                    root = int(
                        target.prefill(prompt, use_bulk=True, return_logits=False).token_id
                    )
                    expected = [root]
                    while len(expected) < int(max_new_tokens):
                        expected.append(
                            int(
                                target.step(
                                    expected[-1], return_logits=False
                                ).token_id
                            )
                        )
                    ar_wall_seconds = time.perf_counter() - ar_started
                    target.reset()
                    mtp_started = time.perf_counter()
                    with _count_split_k_calls() as split_calls:
                        actual = decoder.generate(
                            prompt,
                            max_new_tokens=int(max_new_tokens),
                            request_id=40_000 + len(results),
                            return_cycle_logits=not bool(require_target_graph),
                            use_bulk_prefill=True,
                        )
                    # The server releases each NextN request slot after the
                    # owned cycle; the harness must do the same or a second
                    # generation context fails with "no free request slot".
                    mtp_wall_seconds = time.perf_counter() - mtp_started
                    provider.release_request(40_000 + len(results))
                    cycles = tuple(actual.cycle_records)
                    result = {
                        "context_tokens": int(context),
                        "candidate_budget": int(candidate_budget),
                        "max_new_tokens": int(max_new_tokens),
                        "host_proposal_materialized": True,
                        "target_verify_route": "eager_native",
                        "target_native_graph_submitted": any(
                            bool(row["target_native_graph_submitted"])
                            for row in cycles
                        ),
                        "output_ids_exact": tuple(actual.token_ids)
                        == tuple(expected),
                        "gpu_accept_match_cpu": bool(
                            actual.gpu_accept_match_cpu
                        ),
                        "all_cycle_logits_recorded": bool(cycles)
                        and all(
                            bool(row.get("candidate_logits_recorded", False))
                            for row in cycles
                        ),
                        "all_cycles_eager": bool(cycles)
                        and all(
                            not bool(row["target_native_graph_submitted"])
                            and not bool(row["proposal_target_device_chained"])
                            for row in cycles
                        ),
                        "all_cycles_target_graph": bool(cycles)
                        and all(
                            bool(row["target_native_graph_submitted"])
                            for row in cycles
                        ),
                        "target_graph_capture_ms": sum(
                            float(row.get("target_native_graph_capture_ms", 0.0))
                            for row in cycles
                        ),
                        "target_graph_submit_ms": sum(
                            float(row.get("target_native_graph_submit_ms", 0.0))
                            for row in cycles
                        ),
                        "draft_tail_advance_count": sum(
                            bool(row.get("draft_tail_advanced", False))
                            for row in cycles
                        ),
                        "accepted_counts": [
                            int(row["accepted"]) for row in cycles
                        ],
                        "split_k_calls": len(split_calls),
                        "split_k_kernels": sorted(set(split_calls)),
                        "cycle_count": len(cycles),
                        "ar_wall_seconds": ar_wall_seconds,
                        "mtp_wall_seconds": mtp_wall_seconds,
                        "mtp_vs_true_ar_ratio": (
                            0.0
                            if mtp_wall_seconds <= 0.0
                            else ar_wall_seconds / mtp_wall_seconds
                        ),
                        "wall_seconds": time.perf_counter() - started,
                    }
                    result["passed"] = bool(
                        result["output_ids_exact"]
                        and result["gpu_accept_match_cpu"]
                        and (
                            (require_target_graph and not result["all_cycle_logits_recorded"])
                            or (
                                not require_target_graph
                                and result["all_cycle_logits_recorded"]
                            )
                        )
                        and (
                            result["all_cycles_target_graph"]
                            if require_target_graph
                            else result["all_cycles_eager"]
                        )
                        and bool(result["target_native_graph_submitted"])
                        is bool(require_target_graph)
                        and (
                            int(context) + int(max_new_tokens) < 1024
                            or int(result["split_k_calls"]) > 0
                            or result["all_cycles_target_graph"]
                        )
                    )
                    results.append(result)
                    if on_result is not None:
                        on_result("generation", result)
                    _notify_progress(
                        progress,
                        "generation_case_complete",
                        {
                            "context_tokens": int(context),
                            "passed": bool(result["passed"]),
                            "wall_seconds": float(result["wall_seconds"]),
                        },
                    )
        finally:
            provider.close()
    return results


def _git(args: Sequence[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(model: Path, *, hash_model: bool) -> dict[str, Any]:
    from hipengine.kernels.backends import resolve_backend

    status = _git(("status", "--short"))
    stat = model.stat()
    uname = os.uname()
    tracked_environment = (
        "HIPENGINE_BACKEND",
        "HIPENGINE_GGUF_AOTRITON_PREFILL_ENABLE",
        "HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT",
        "HIPENGINE_REQUIRE_CACHED_BUILD",
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _git(("rev-parse", "HEAD")),
        "repo_branch": _git(("rev-parse", "--abbrev-ref", "HEAD")),
        "repo_dirty": bool(status),
        "model_path": str(model.resolve()),
        "model_size_bytes": int(stat.st_size),
        "model_sha256": _sha256(model) if hash_model else None,
        "backend_requested": os.environ.get("HIPENGINE_BACKEND", "auto"),
        "backend_resolved": resolve_backend("auto"),
        "host": uname.nodename,
        "system": uname.sysname,
        "kernel_release": uname.release,
        "machine": uname.machine,
        "environment": {
            name: os.environ.get(name) for name in tracked_environment
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--cycle-ends",
        default="1016-1032",
        help="direct eager/strict cycle ends; accepts CSV, inclusive ranges, K suffix",
    )
    parser.add_argument("--candidate-budgets", default="1,2,3")
    parser.add_argument(
        "--direct-repeats",
        type=int,
        default=1,
        help="Repeat each direct case in the same session to prove graph replay.",
    )
    parser.add_argument(
        "--direct-remaining-decode",
        type=int,
        help="Override direct remaining-decode to diagnose N1 versus N2 ownership.",
    )
    parser.add_argument(
        "--acceptance-cycle-ends",
        default="1024",
        help="cycle ends for reject/every-partial/full controlled B3 cases; empty disables",
    )
    parser.add_argument("--acceptance-budget", type=int, default=3)
    parser.add_argument(
        "--generation-contexts",
        default="1024",
        help="real host-proposal NextN prompt lengths (for example 1024,4K); empty disables",
    )
    parser.add_argument(
        "--generation-require-target-graph",
        action="store_true",
        help="Require every real-generation cycle to use a cached native target graph.",
    )
    parser.add_argument("--generation-budget", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument(
        "--require-target-graph",
        action="store_true",
        help="Require direct cases to use a native target graph; disables diagnostic logits.",
    )
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    cycle_ends = (
        () if not str(args.cycle_ends).strip() else parse_token_spec(args.cycle_ends)
    )
    budgets = (
        ()
        if not str(args.candidate_budgets).strip()
        else parse_token_spec(args.candidate_budgets)
    )
    acceptance_cycle_ends = (
        ()
        if not str(args.acceptance_cycle_ends).strip()
        else parse_token_spec(args.acceptance_cycle_ends)
    )
    generation_contexts = (
        ()
        if not str(args.generation_contexts).strip()
        else parse_token_spec(args.generation_contexts)
    )
    if int(args.acceptance_budget) not in {1, 2, 3}:
        raise SystemExit("--acceptance-budget must be 1, 2, or 3")
    if int(args.generation_budget) not in {1, 2, 3}:
        raise SystemExit("--generation-budget must be 1, 2, or 3")
    if int(args.max_new_tokens) <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if int(args.direct_repeats) <= 0:
        raise SystemExit("--direct-repeats must be positive")
    if args.direct_remaining_decode is not None and int(args.direct_remaining_decode) < 0:
        raise SystemExit("--direct-remaining-decode must be non-negative")
    if bool(args.require_target_graph) and generation_contexts:
        raise SystemExit("--require-target-graph supports direct cases only")
    if cycle_ends and not budgets:
        raise SystemExit("--candidate-budgets is required when --cycle-ends is set")
    base_cases = (
        *build_cycle_cases(
            cycle_ends=cycle_ends,
            candidate_budgets=budgets,
        ),
        *build_acceptance_cases(
            cycle_ends=acceptance_cycle_ends,
            candidate_budget=int(args.acceptance_budget),
        ),
    )
    cases = tuple(
        case
        for case in base_cases
        for _repeat in range(int(args.direct_repeats))
    )
    if not cases and not generation_contexts:
        raise SystemExit("at least one direct or generation scenario is required")
    maximum = max(
        max(cycle_ends, default=0),
        max(acceptance_cycle_ends, default=0),
        max(generation_contexts, default=0) + int(args.max_new_tokens),
    )
    max_sequence_length = (
        int(args.max_sequence_length)
        if args.max_sequence_length is not None
        else maximum + 8
    )
    if max_sequence_length < maximum:
        raise SystemExit("--max-sequence-length does not cover the requested cycles")

    configuration = {
        "cycle_ends": list(cycle_ends),
        "candidate_budgets": list(budgets),
        "direct_repeats": int(args.direct_repeats),
        "direct_remaining_decode": (
            None
            if args.direct_remaining_decode is None
            else int(args.direct_remaining_decode)
        ),
        "acceptance_cycle_ends": list(acceptance_cycle_ends),
        "acceptance_budget": int(args.acceptance_budget),
        "generation_contexts": list(generation_contexts),
        "generation_require_target_graph": bool(
            args.generation_require_target_graph
        ),
        "generation_budget": int(args.generation_budget),
        "max_new_tokens": int(args.max_new_tokens),
        "max_sequence_length": max_sequence_length,
        "require_cached_build": bool(args.require_cached_build),
        "require_target_graph": bool(args.require_target_graph),
    }
    checkpoint_direct: list[dict[str, Any]] = []
    checkpoint_generation: list[dict[str, Any]] = []
    started = time.perf_counter()

    def write_checkpoint(event: str, details: dict[str, Any]) -> None:
        if args.out is None:
            return
        _atomic_write_json(
            args.out,
            {
                "schema": 1,
                "kind": "gguf_mtp_eager_long_context_correctness_checkpoint",
                "status": "running",
                "verdict": None,
                "command": [sys.executable, *sys.argv],
                "configuration": configuration,
                "active_event": str(event),
                "active_details": dict(details),
                "direct_results": checkpoint_direct,
                "generation_results": checkpoint_generation,
                "summary": {
                    "direct_cases_completed": len(checkpoint_direct),
                    "generation_cases_completed": len(checkpoint_generation),
                    "wall_seconds": time.perf_counter() - started,
                },
            },
        )

    def progress(event: str, details: dict[str, Any]) -> None:
        print(
            json.dumps({"event": str(event), **details}, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        write_checkpoint(event, details)

    def record_result(kind: str, result: dict[str, Any]) -> None:
        destination = (
            checkpoint_direct if kind == "direct" else checkpoint_generation
        )
        destination.append(dict(result))
        write_checkpoint(
            f"{kind}_result_checkpointed",
            {
                "case_id": result.get("case_id"),
                "context_tokens": result.get("context_tokens"),
                "passed": bool(result.get("passed", False)),
            },
        )

    write_checkpoint("run_start", {})
    try:
        direct_results = (
            _run_direct_cases(
                args.model,
                cases,
                max_sequence_length=max_sequence_length,
                require_cached_build=bool(args.require_cached_build),
                require_target_graph=bool(args.require_target_graph),
                direct_remaining_decode=(
                    None
                    if args.direct_remaining_decode is None
                    else int(args.direct_remaining_decode)
                ),
                progress=progress,
                on_result=record_result,
            )
            if cases
            else []
        )
        generation_results = (
            _run_generation_contexts(
                args.model,
                generation_contexts,
                candidate_budget=int(args.generation_budget),
                max_new_tokens=int(args.max_new_tokens),
                max_sequence_length=max_sequence_length,
                require_cached_build=bool(args.require_cached_build),
                require_target_graph=bool(args.generation_require_target_graph),
                progress=progress,
                on_result=record_result,
            )
            if generation_contexts
            else []
        )
    except Exception as exc:
        progress(
            "run_error",
            {"error": f"{type(exc).__name__}: {exc}"},
        )
        if args.out is not None:
            failure = json.loads(args.out.read_text())
            failure["status"] = "error"
            failure["verdict"] = "error"
            failure["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(args.out, failure)
        raise
    direct_passed = (
        gate_passed(
            direct_results,
            require_target_graph=bool(args.require_target_graph),
        )
        if direct_results
        else True
    )
    generation_passed = (
        all(bool(result.get("passed", False)) for result in generation_results)
        if generation_results
        else True
    )
    overall_passed = bool(direct_passed and generation_passed)
    payload = {
        "schema": 1,
        "kind": "gguf_mtp_eager_long_context_correctness",
        "status": "passed" if overall_passed else "failed",
        "verdict": "pass" if overall_passed else "fail",
        "performance_claim": False,
        "profile_contract": "strict_teacher_parity",
        "profile_manifest_hash": None,
        "model_quant": "gguf_q4_k_m",
        "kv_storage": "bf16",
        "command": [sys.executable, *sys.argv],
        "proposal_mode": (
            "host_materialized" if generation_results else "host_teacher_rows"
        ),
        "target_verify_mode": "eager_native",
        "reusable_target_graph_context_limit": 1023,
        "provenance": _provenance(args.model, hash_model=bool(args.hash_model)),
        "configuration": configuration,
        "direct_results": direct_results,
        "generation_results": generation_results,
        "summary": {
            "direct_passed": direct_passed,
            "direct_cases": len(direct_results),
            "generation_passed": generation_passed,
            "generation_cases": len(generation_results),
            "wall_seconds": time.perf_counter() - started,
        },
        "passed": overall_passed,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        _atomic_write_json(args.out, payload)
        print(
            f"wrote {args.out}: passed={payload['passed']} "
            f"direct={len(direct_results)} generation={len(generation_results)}",
            flush=True,
        )
    if args.fail_on_fail and not payload["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
