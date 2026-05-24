"""Speculative-decoding benchmark artifact helpers.

The native DFlash/MTP fast path is not implemented yet, but the benchmark
contract needs to be stable before the kernels land.  This module keeps the
contract torch-free and ingest-oriented: future runners can hand it row-level
measurements and receive a compact schema-2 benchmark artifact with the same
fields used by the parent ``~/amd-gpu-tuning`` DFlash/MTP harnesses.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_TARGET_MODEL = "shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed"
DEFAULT_DFLASH_DRAFTER = "z-lab/Qwen3.6-35B-A3B-DFlash"

PORTED_PARENT_BENCHMARKS = (
    "../amd-gpu-tuning/nano-vllm-amd/scripts/sweep_qwen35_mtp_real_acceptance.py",
    "../amd-gpu-tuning/nano-vllm-amd/scripts/bench_qwen35_dflash_acceptance.py",
    "../amd-gpu-tuning/nano-vllm-amd/scripts/eval_qwen35_dflash_acceptance_suite.py",
    "../amd-gpu-tuning/scripts/bench_dense27_dflash_smoke.py",
    "../amd-gpu-tuning/scripts/sweep_dense27_dflash_prediction.py",
)


@dataclass(frozen=True)
class SpeculativeBenchmarkModels:
    """Target/drafter identity recorded in every DFlash/MTP artifact."""

    target_name: str = DEFAULT_TARGET_MODEL
    target_path: str | None = None
    target_revision: str | None = None
    target_quant: str = "w4_paro_packed"
    drafter_name: str = DEFAULT_DFLASH_DRAFTER
    drafter_path: str | None = None
    drafter_revision: str | None = None
    drafter_dtype: str = "bf16"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "target": {
                "name": self.target_name,
                "path": self.target_path,
                "snapshot_revision": self.target_revision,
                "quant": self.target_quant,
            },
            "drafter": {
                "name": self.drafter_name,
                "path": self.drafter_path,
                "snapshot_revision": self.drafter_revision,
                "dtype": self.drafter_dtype,
            },
        }


@dataclass(frozen=True)
class D2HCounts:
    """Host readback counters for a speculative decode cycle/row."""

    scalar_reads: int = 0
    vector_reads: int = 0
    scalar_values: int = 0
    vector_values: int = 0
    full_logits_readbacks: int = 0
    notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "D2HCounts":
        if raw is None:
            return cls()
        return cls(
            scalar_reads=_int(raw.get("scalar_reads", raw.get("scalar", 0))),
            vector_reads=_int(raw.get("vector_reads", raw.get("vector", 0))),
            scalar_values=_int(raw.get("scalar_values", 0)),
            vector_values=_int(raw.get("vector_values", 0)),
            full_logits_readbacks=_int(raw.get("full_logits_readbacks", raw.get("logit_readbacks", 0))),
            notes=tuple(str(x) for x in raw.get("notes", ()) or ()),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "scalar_reads": self.scalar_reads,
            "vector_reads": self.vector_reads,
            "scalar_values": self.scalar_values,
            "vector_values": self.vector_values,
            "full_logits_readbacks": self.full_logits_readbacks,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SpeculativeGraphStatus:
    """Graph-capture status for a fixed speculative verification bucket."""

    status: str = "not_captured"
    replay_steps: int = 0
    bucket_key: dict[str, Any] | None = None
    validation_passed: bool | None = None
    fallback_reason: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "SpeculativeGraphStatus":
        if raw is None:
            return cls()
        return cls(
            status=str(raw.get("status", raw.get("graph_status", "not_captured"))),
            replay_steps=_int(raw.get("replay_steps", raw.get("graph_steps_per_replay", 0))),
            bucket_key=_dict_or_none(raw.get("bucket_key", raw.get("shape_key"))),
            validation_passed=_optional_bool(raw.get("validation_passed", raw.get("decode_step_graph_validation"))),
            fallback_reason=_optional_str(raw.get("fallback_reason")),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "replay_steps": self.replay_steps,
            "bucket_key": self.bucket_key,
            "validation_passed": self.validation_passed,
            "fallback_reason": self.fallback_reason,
        }


def first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    """Return the first mismatch index for two generated-token sequences."""

    for idx, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def acceptance_histogram(lengths: Iterable[int]) -> dict[str, int]:
    """Return a stable string-keyed histogram of accepted output lengths."""

    hist = Counter(int(value) for value in lengths)
    return {str(key): int(hist[key]) for key in sorted(hist)}


def expand_histogram(histogram: Mapping[str, Any] | None) -> list[int]:
    """Expand a string-keyed histogram back into a list of accepted lengths."""

    if not histogram:
        return []
    out: list[int] = []
    for key, count in histogram.items():
        out.extend([int(key)] * _int(count))
    return out


def acceptance_summary(lengths: Sequence[int], *, max_positions: int = 10) -> dict[str, Any]:
    """Summarize accepted lengths using the parent DFlash/MTP metric shape."""

    values = [int(value) for value in lengths]
    steps = len(values)
    hist = acceptance_histogram(values)
    denom = max(1, steps)
    exact = {str(k): sum(1 for value in values if value == k) / denom for k in range(0, max_positions + 1)}
    ge = {str(k): sum(1 for value in values if value >= k) / denom for k in range(1, max_positions + 1)}
    accepted_tokens = sum(values)
    return {
        "steps": steps,
        "accepted_output_tokens": accepted_tokens,
        "avg_accept_length": _safe_div(accepted_tokens, steps),
        "accept_histogram": hist,
        "exact_rate_0_to_n": exact,
        "ge_rate_1_to_n": ge,
        "multi_token_acceptance_rate": ge.get("2"),
    }


def normalize_speculative_row(raw: Mapping[str, Any], *, row_index: int = 0) -> dict[str, Any]:
    """Normalize one raw row from a future DFlash/MTP runner.

    The accepted input shape intentionally matches the parent harnesses: rows may
    contain top-level ``ar`` and either ``spec`` or ``dflash`` sections.  The
    normalizer computes exact equality, finite-logit gates, acceptance
    histograms, rows/output, verify ETA, split timing, D2H counters, graph
    status, and memory peaks so future runners do not all reinvent that logic.
    """

    prompt = dict(raw.get("prompt", {}) or {})
    config = dict(raw.get("config", {}) or {})
    ar = dict(raw.get("ar", raw.get("autoregressive", {})) or {})
    spec = dict(raw.get("spec", raw.get("dflash", raw.get("mtp", {}))) or {})
    quality = dict(raw.get("quality_gate", raw.get("correctness", {})) or {})
    memory = dict(raw.get("memory", {}) or {})

    ar_tokens = _int_list(ar.get("generated_ids", ar.get("output_tokens", ar.get("generated_sample", ()))) or ())
    spec_tokens = _int_list(spec.get("generated_ids", spec.get("output_tokens", spec.get("generated_sample", ()))) or ())

    explicit_exact = spec.get("exact_match_ar", quality.get("exact_match_ar", raw.get("exact_match_ar")))
    mismatch = first_mismatch(ar_tokens, spec_tokens) if ar_tokens and spec_tokens else None
    exact_match = (mismatch is None) if ar_tokens and spec_tokens else _optional_bool(explicit_exact)
    if exact_match is False and mismatch is None:
        mismatch = _optional_int(spec.get("first_mismatch_index", quality.get("first_mismatch_index")))

    decode_tokens = _first_positive_int(
        raw.get("decode_tokens"),
        prompt.get("decode_tokens"),
        ar.get("decode_tokens"),
        spec.get("decode_tokens"),
        len(ar_tokens),
        len(spec_tokens),
    )
    prompt_tokens = _first_nonnegative_int(
        prompt.get("prompt_tokens"),
        raw.get("prompt_tokens"),
        raw.get("prompt_length"),
    )

    ar_seconds = _optional_float(ar.get("decode_seconds", ar.get("seconds")))
    spec_seconds = _optional_float(spec.get("decode_seconds", spec.get("seconds")))
    ar_tok_s = _optional_float(ar.get("decode_tok_s", ar.get("tok_s")))
    if ar_tok_s is None and ar_seconds and decode_tokens:
        ar_tok_s = _safe_div(decode_tokens, ar_seconds)
    spec_tok_s = _optional_float(spec.get("verified_target_tok_s", spec.get("decode_tok_s", spec.get("tok_s"))))
    if spec_tok_s is None and spec_seconds and decode_tokens:
        spec_tok_s = _safe_div(decode_tokens, spec_seconds)

    accepted_lengths = _accepted_lengths_from_spec(spec)
    accepted = acceptance_summary(accepted_lengths)
    target_verify_rows = _first_nonnegative_int(
        spec.get("target_verify_rows"),
        spec.get("verify_rows"),
        spec.get("rows"),
    )
    rows_per_output = _safe_div(target_verify_rows, decode_tokens)
    verify_seconds = _optional_float(spec.get("target_verify_seconds", spec.get("verify_seconds")))
    verify_eta = _safe_div(verify_seconds, target_verify_rows)
    if verify_eta is not None and ar_tok_s is not None:
        verify_eta *= ar_tok_s

    draft_seconds = _optional_float(spec.get("draft_seconds"))
    draft_context_full_rebuild_seconds = _optional_float(spec.get("draft_context_full_rebuild_seconds"))
    draft_context_append_seconds = _optional_float(spec.get("draft_context_append_seconds"))
    draft_query_seconds = _optional_float(spec.get("draft_query_seconds"))
    commit_seconds = _optional_float(
        spec.get("commit_seconds", spec.get("commit_install_seconds", spec.get("commit_replay_seconds")))
    )
    split = _phase_split(
        total_seconds=spec_seconds,
        draft_seconds=draft_seconds,
        verify_seconds=verify_seconds,
        commit_seconds=commit_seconds,
    )

    finite_ar = _optional_bool(ar.get("finite_logits", quality.get("finite_ar_logits")))
    finite_draft = _optional_bool(spec.get("finite_draft_logits", quality.get("finite_dflash_draft_logits")))
    finite_verify = _optional_bool(spec.get("finite_verify_logits", quality.get("finite_dflash_verify_logits")))
    finite_all = all(value is True for value in (finite_ar, finite_draft, finite_verify))

    d2h = D2HCounts.from_mapping(spec.get("d2h", raw.get("d2h")))
    graph = SpeculativeGraphStatus.from_mapping(spec.get("graph", raw.get("graph")))
    peak_memory = _peak_memory_summary(memory)

    draft_budget = _first_positive_int(
        config.get("draft_budget"),
        config.get("block_size"),
        spec.get("draft_budget"),
        spec.get("block_size"),
    )
    proposed_tokens = _first_nonnegative_int(
        spec.get("draft_tokens_proposed"),
        spec.get("draft_tokens"),
        spec.get("candidate_tokens"),
        len(accepted_lengths) * (draft_budget or 0),
    )
    accepted_tokens = _first_nonnegative_int(
        spec.get("accepted_draft_tokens"),
        spec.get("accepted_tokens"),
        accepted["accepted_output_tokens"],
    )

    row = {
        "row_index": int(row_index),
        "prompt": {
            "id": _optional_str(prompt.get("id", prompt.get("prompt_id", f"row-{row_index}"))),
            "dataset": _optional_str(prompt.get("dataset")),
            "category": _optional_str(prompt.get("category", prompt.get("kind"))),
            "prompt_tokens": prompt_tokens,
            "prompt_ids_sha256": _optional_str(prompt.get("prompt_ids_sha256")),
            "prompt_text_sha256": _optional_str(prompt.get("prompt_text_sha256")),
            "prompt_preview": _optional_str(prompt.get("prompt_preview", prompt.get("text_preview"))),
            "representative": _optional_bool(prompt.get("representative")),
        },
        "config": {
            "name": _optional_str(config.get("name")),
            "provider": _optional_str(config.get("provider", "dflash")),
            "proposal_mode": _optional_str(config.get("proposal_mode", "chain")),
            "verify_mode": _optional_str(config.get("verify_mode", config.get("target_verify_mode", "verify_chain"))),
            "draft_budget": draft_budget,
            "topk": _optional_int(config.get("topk", config.get("ddtree_topk"))),
            "tree_budget": _optional_int(config.get("tree_budget", config.get("ddtree_budget"))),
            "tree_mode": _optional_str(config.get("tree_mode")),
            "profile_route": _optional_str(config.get("profile_route")),
        },
        "ar": {
            "same_session_control": _optional_bool(ar.get("same_session_control", ar.get("same_session_control_reused_for_prompt", True))),
            "same_process_control": _optional_bool(ar.get("same_process_control")),
            "decode_seconds": ar_seconds,
            "decode_tok_s": ar_tok_s,
            "finite_logits": finite_ar,
            "generated_sample": ar_tokens[:32],
        },
        "spec": {
            "decode_seconds": spec_seconds,
            "decode_tok_s": spec_tok_s,
            "speedup_vs_ar": _safe_div(spec_tok_s, ar_tok_s),
            "draft_seconds": draft_seconds,
            "draft_context_full_rebuild_seconds": draft_context_full_rebuild_seconds,
            "draft_context_append_seconds": draft_context_append_seconds,
            "draft_query_seconds": draft_query_seconds,
            "draft_context_phase_seconds": {
                "full_context_rebuild": draft_context_full_rebuild_seconds,
                "append_materialize": draft_context_append_seconds,
                "query_only_drafter": draft_query_seconds,
            },
            "draft_native_phase_seconds": _float_dict(spec.get("draft_native_phase_seconds")),
            "draft_graph": _dict_or_none(spec.get("draft_graph")),
            "draft_fusion": _dict_or_none(spec.get("draft_fusion")),
            "adaptive_budget": _dict_or_none(spec.get("adaptive_budget")),
            "drafter_context_mode": _optional_str(spec.get("drafter_context_mode")),
            "draft_phase_timing_mode": _optional_str(spec.get("draft_phase_timing_mode")),
            "proposal_trace_sample": _json_list(spec.get("proposal_trace_sample")),
            "proposal_trace_count": _optional_int(spec.get("proposal_trace_count")),
            "draft_kv_bytes": _optional_int(spec.get("draft_kv_bytes", spec.get("draft_context_kv_bytes"))),
            "draft_kv_capacity_tokens": _optional_int(spec.get("draft_kv_capacity_tokens", spec.get("draft_context_capacity_tokens"))),
            "target_verify_seconds": verify_seconds,
            "commit_seconds": commit_seconds,
            "phase_split": split,
            "target_verify_rows": target_verify_rows,
            "target_forward_calls": _optional_int(spec.get("target_forward_calls")),
            "target_bulk_forward_calls": _optional_int(spec.get("target_bulk_forward_calls")),
            "target_serial_forward_calls": _optional_int(spec.get("target_serial_forward_calls")),
            "target_bulk_rows": _optional_int(spec.get("target_bulk_rows")),
            "canonical_commit_replay_rows": _optional_int(spec.get("canonical_commit_replay_rows")),
            "target_forwards_per_draft_call": _optional_float(spec.get("target_forwards_per_draft_call")),
            "gpu_accept_match_cpu": _optional_bool(spec.get("gpu_accept_match_cpu")),
            "verifier_graph": _dict_or_none(spec.get("verifier_graph")),
            "target_verify_rows_per_output_token": rows_per_output,
            "verify_eta_vs_ar_per_row": verify_eta,
            "draft_tokens_proposed": proposed_tokens,
            "accepted_draft_tokens": accepted_tokens,
            "tree_active_nodes_total": _optional_int(spec.get("tree_active_nodes_total")),
            "tree_top_k": _optional_int(spec.get("tree_top_k")),
            "draft_top_k": _optional_int(spec.get("draft_top_k")),
            "drafter_query_mode": _optional_str(spec.get("drafter_query_mode")),
            "drafter_query_rows": _optional_int(spec.get("drafter_query_rows")),
            "drafter_block_size": _optional_int(spec.get("drafter_block_size")),
            "tree_compiler": _optional_str(spec.get("tree_compiler")),
            "draft_calls": _optional_int(spec.get("draft_calls")),
            "commit_rows": _optional_int(spec.get("commit_rows", spec.get("commit_replay_rows"))),
            "generated_sample": spec_tokens[:32],
            "same_session_control": _optional_bool(spec.get("same_session_control")),
            "same_process_control": _optional_bool(spec.get("same_process_control")),
            "verifier_mode": _optional_str(spec.get("verifier_mode")),
            "verifier_tree_mode": _optional_str(spec.get("verifier_tree_mode")),
            "verifier_state_strategy": _optional_str(spec.get("verifier_state_strategy")),
            "canonical_commit_mode": _optional_str(spec.get("canonical_commit_mode")),
            "verifier_state_copies_per_cycle": _optional_float(spec.get("verifier_state_copies_per_cycle")),
            "verifier_state_copies_total": _optional_int(spec.get("verifier_state_copies_total")),
            "native_bulk_verifier": _optional_bool(spec.get("native_bulk_verifier")),
            "backend": _optional_str(spec.get("backend")),
            "target_arch": _optional_str(spec.get("target_arch", spec.get("arch"))),
        },
        "acceptance": accepted,
        "correctness": {
            "exact_match_ar": exact_match,
            "first_mismatch_index": mismatch,
            "finite_ar_logits": finite_ar,
            "finite_draft_logits": finite_draft,
            "finite_verify_logits": finite_verify,
            "finite_all_logits": finite_all,
            "passed": bool(exact_match is True and finite_all),
        },
        "d2h": d2h.to_json_dict(),
        "graph": graph.to_json_dict(),
        "memory": peak_memory,
        "decode_tokens": decode_tokens,
    }
    return row


def aggregate_speculative_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate normalized speculative rows for artifact-level summaries."""

    normalized = [dict(row) for row in rows]
    decode_tokens = 0
    ar_seconds = 0.0
    spec_seconds = 0.0
    target_verify_rows = 0
    accepted_lengths: list[int] = []
    exact_count = 0
    correctness_count = 0
    finite_count = 0
    d2h_scalar = 0
    d2h_vector = 0
    d2h_scalar_values = 0
    d2h_vector_values = 0
    full_logits_readbacks = 0
    speedups: list[float] = []

    for row in normalized:
        prompt_tokens = _optional_int(row.get("prompt", {}).get("prompt_tokens"))
        _ = prompt_tokens  # prompt count is carried per-row; no aggregate needed yet.
        ar = row.get("ar", {})
        spec = row.get("spec", {})
        correctness = row.get("correctness", {})
        acceptance = row.get("acceptance", {})
        token_count = _decode_tokens_from_row(row)
        decode_tokens += token_count
        ar_seconds += float(ar.get("decode_seconds") or 0.0)
        spec_seconds += float(spec.get("decode_seconds") or 0.0)
        target_verify_rows += int(spec.get("target_verify_rows") or 0)
        accepted_lengths.extend(expand_histogram(acceptance.get("accept_histogram") or {}))
        if correctness.get("exact_match_ar") is True:
            exact_count += 1
        if correctness.get("passed") is True:
            correctness_count += 1
        if correctness.get("finite_all_logits") is True:
            finite_count += 1
        d2h = row.get("d2h", {})
        d2h_scalar += int(d2h.get("scalar_reads") or 0)
        d2h_vector += int(d2h.get("vector_reads") or 0)
        d2h_scalar_values += int(d2h.get("scalar_values") or 0)
        d2h_vector_values += int(d2h.get("vector_values") or 0)
        full_logits_readbacks += int(d2h.get("full_logits_readbacks") or 0)
        speedup = _optional_float(spec.get("speedup_vs_ar"))
        if speedup is not None and math.isfinite(speedup):
            speedups.append(speedup)

    ar_tok_s = _safe_div(decode_tokens, ar_seconds)
    spec_tok_s = _safe_div(decode_tokens, spec_seconds)
    aggregate_speedup = _safe_div(spec_tok_s, ar_tok_s)
    return {
        "rows": len(normalized),
        "decode_tokens": decode_tokens,
        "exact_match_rows": exact_count,
        "correctness_pass_rows": correctness_count,
        "finite_all_logits_rows": finite_count,
        "all_exact_match_ar": exact_count == len(normalized) if normalized else False,
        "all_correctness_passed": correctness_count == len(normalized) if normalized else False,
        "all_finite_logits": finite_count == len(normalized) if normalized else False,
        "ar_decode_tok_s": ar_tok_s,
        "spec_decode_tok_s": spec_tok_s,
        "speedup_vs_ar": aggregate_speedup,
        "median_row_speedup_vs_ar": statistics.median(speedups) if speedups else None,
        "target_verify_rows": target_verify_rows,
        "target_verify_rows_per_output_token": _safe_div(target_verify_rows, decode_tokens),
        "acceptance": acceptance_summary(accepted_lengths),
        "d2h": {
            "scalar_reads": d2h_scalar,
            "vector_reads": d2h_vector,
            "scalar_values": d2h_scalar_values,
            "vector_values": d2h_vector_values,
            "full_logits_readbacks": full_logits_readbacks,
        },
        "speed_gate_gt_1p10": bool(aggregate_speedup is not None and aggregate_speedup > 1.10),
    }


def build_speculative_artifact(
    *,
    run_tag: str,
    summary: str,
    rows: Sequence[Mapping[str, Any]],
    models: SpeculativeBenchmarkModels | None = None,
    status: str = "diagnostic",
    timestamp: str | None = None,
    hardware: Mapping[str, Any] | None = None,
    software: Mapping[str, Any] | None = None,
    workload: Mapping[str, Any] | None = None,
    commands: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
    synthetic_schema_fixture: bool = False,
    decision_reason: str | None = None,
) -> dict[str, Any]:
    """Build a compact schema-2 artifact for DFlash/MTP benchmark rows."""

    normalized_rows = [
        normalize_speculative_row(row, row_index=idx) if not _looks_normalized(row) else dict(row)
        for idx, row in enumerate(rows)
    ]
    aggregate = aggregate_speculative_rows(normalized_rows)
    performance_claim = bool(
        status == "accepted"
        and not synthetic_schema_fixture
        and aggregate["all_correctness_passed"]
        and aggregate["speed_gate_gt_1p10"]
    )
    accepted = bool(status == "accepted" and performance_claim)
    if decision_reason is None:
        if synthetic_schema_fixture:
            decision_reason = "schema fixture only; no performance claim"
        elif not normalized_rows:
            decision_reason = "no speculative rows recorded yet"
        elif not aggregate["all_correctness_passed"]:
            decision_reason = "one or more rows failed exact/finite correctness gates"
        elif not aggregate["speed_gate_gt_1p10"]:
            decision_reason = "speed gate >1.10x AR not met"
        else:
            decision_reason = "correctness and speed gate passed"

    baseline_delta_percent = None
    if aggregate["speedup_vs_ar"] is not None:
        baseline_delta_percent = (float(aggregate["speedup_vs_ar"]) - 1.0) * 100.0

    return {
        "schema": 2,
        "speculative_schema": 1,
        "status": status,
        "performance_claim": performance_claim,
        "synthetic_schema_fixture": bool(synthetic_schema_fixture),
        "timestamp": timestamp,
        "run_tag": run_tag,
        "summary": summary,
        "decision_reason": decision_reason,
        "correctness_gate": {
            "oracle": "same-session greedy AR equality plus finite AR/draft/verify logits",
            "passed": aggregate["all_correctness_passed"],
            "all_exact_match_ar": aggregate["all_exact_match_ar"],
            "all_finite_logits": aggregate["all_finite_logits"],
            "correctness_pass_rows": aggregate["correctness_pass_rows"],
            "rows": aggregate["rows"],
        },
        "baseline": {
            "type": "same_session_ar_control",
            "ar_decode_tok_s": aggregate["ar_decode_tok_s"],
            "spec_decode_tok_s": aggregate["spec_decode_tok_s"],
            "speedup_vs_ar": aggregate["speedup_vs_ar"],
            "delta_vs_ar_percent": baseline_delta_percent,
        },
        "source_lineage": {
            "ported_metric_shape_from": list(PORTED_PARENT_BENCHMARKS),
            "notes": [
                "Metric contract only; PyTorch/HF parent hot loops are not production hipEngine runtime code.",
                "Rows require same-session AR controls before any speedup can be promoted.",
            ],
        },
        "hardware": dict(hardware or {}),
        "software": dict(software or {}),
        "models": (models or SpeculativeBenchmarkModels()).to_json_dict(),
        "workload": dict(workload or {}),
        "commands": dict(commands or {}),
        "speculative_contract": {
            "providers": ["dflash", "mtp"],
            "verify_modes": ["verify_chain", "verify_tree"],
            "required_row_metrics": [
                "same_session_ar_decode_tok_s",
                "spec_decode_tok_s",
                "exact_match_ar",
                "finite_ar_logits",
                "finite_draft_logits",
                "finite_verify_logits",
                "acceptance_histogram",
                "target_verify_rows_per_output_token",
                "draft_seconds",
                "target_verify_seconds",
                "commit_seconds",
                "scalar_d2h_reads",
                "vector_d2h_reads",
                "graph_status",
                "peak_memory",
            ],
        },
        "measurements": {
            "rows": normalized_rows,
            "aggregate": aggregate,
        },
        "decision": {
            "accepted": accepted,
            "reason": decision_reason,
            "promotion_rule": "accepted only when every row is exact/finite and aggregate speedup_vs_ar > 1.10",
        },
        "notes": list(notes),
    }


def schema_fixture_row() -> dict[str, Any]:
    """Return one synthetic row that exercises every speculative metric field."""

    return {
        "prompt": {
            "id": "schema:humaneval_add",
            "dataset": "stable/code",
            "category": "code",
            "prompt_tokens": 64,
            "prompt_ids_sha256": "sha256:synthetic-token-fixture",
            "prompt_text_sha256": "sha256:synthetic-text-fixture",
            "prompt_preview": "def add(a: int, b: int) -> int: ...",
            "representative": True,
        },
        "config": {
            "name": "chain_b4",
            "provider": "dflash",
            "proposal_mode": "chain",
            "verify_mode": "verify_chain",
            "draft_budget": 4,
            "topk": 1,
        },
        "ar": {
            "same_session_control": True,
            "same_process_control": True,
            "decode_seconds": 4.0,
            "finite_logits": True,
            "generated_ids": [101, 102, 103, 104, 105, 106, 107, 108],
        },
        "spec": {
            "decode_seconds": 3.0,
            "draft_seconds": 0.35,
            "draft_context_full_rebuild_seconds": 0.20,
            "draft_context_append_seconds": 0.05,
            "draft_query_seconds": 0.10,
            "draft_native_phase_seconds": {"context_projection": 0.02, "decoder_layers": 0.25, "lm_head": 0.03, "topk_and_readback": 0.01},
            "drafter_context_mode": "append_kv_query_only",
            "draft_phase_timing_mode": "synchronized",
            "proposal_trace_sample": [
                {"cycle": 1, "root_token": 101, "draft_candidates": [102, 103], "target_top1_path": [102, 103], "accepted": 2}
            ],
            "proposal_trace_count": 4,
            "draft_kv_bytes": 576,
            "draft_kv_capacity_tokens": 6,
            "target_verify_seconds": 2.25,
            "commit_seconds": 0.40,
            "target_verify_rows": 16,
            "target_forward_calls": 4,
            "target_bulk_forward_calls": 4,
            "target_serial_forward_calls": 0,
            "target_bulk_rows": 20,
            "target_forwards_per_draft_call": 1.0,
            "gpu_accept_match_cpu": True,
            "verifier_graph": {"mode": "auto", "validation_passed": True},
            "draft_tokens": 16,
            "accepted_draft_tokens": 8,
            "accepted_lengths": [2, 1, 3, 2],
            "finite_draft_logits": True,
            "finite_verify_logits": True,
            "generated_ids": [101, 102, 103, 104, 105, 106, 107, 108],
            "same_session_control": True,
            "same_process_control": True,
            "verifier_mode": "native_bulk",
            "native_bulk_verifier": True,
            "backend": "hip_gfx1151",
            "target_arch": "gfx1151",
            "d2h": {
                "scalar_reads": 4,
                "vector_reads": 1,
                "scalar_values": 8,
                "vector_values": 8,
                "full_logits_readbacks": 0,
            },
            "graph": {
                "status": "not_captured",
                "replay_steps": 0,
                "bucket_key": {"mode": "verify_chain", "draft_budget": 4, "active_c": 1},
                "validation_passed": None,
            },
        },
        "memory": {
            "allocated_after_load_bytes": 20_000_000_000,
            "peak_allocated_bytes": 22_000_000_000,
            "peak_reserved_bytes": 23_000_000_000,
            "hip_used_peak_sampled_bytes": 24_000_000_000,
        },
        "decode_tokens": 8,
    }


def _accepted_lengths_from_spec(spec: Mapping[str, Any]) -> list[int]:
    for key in ("accepted_lengths", "accepted_output_lengths", "accepted_draft_counts", "accepted_draft_streak_lengths"):
        value = spec.get(key)
        if value:
            return _int_list(value)
    hist = spec.get("acceptance_histogram", spec.get("accepted_draft_streak_histogram"))
    return expand_histogram(hist if isinstance(hist, Mapping) else None)


def _phase_split(
    *,
    total_seconds: float | None,
    draft_seconds: float | None,
    verify_seconds: float | None,
    commit_seconds: float | None,
) -> dict[str, Any]:
    values = {
        "draft_seconds": draft_seconds,
        "target_verify_seconds": verify_seconds,
        "commit_seconds": commit_seconds,
    }
    measured = sum(float(value) for value in values.values() if value is not None)
    if total_seconds is None and measured > 0:
        total_seconds = measured
    other = None
    if total_seconds is not None:
        other = max(0.0, float(total_seconds) - measured)
    return {
        **values,
        "other_seconds": other,
        "total_seconds": total_seconds,
        "draft_fraction": _safe_div(draft_seconds, total_seconds),
        "target_verify_fraction": _safe_div(verify_seconds, total_seconds),
        "commit_fraction": _safe_div(commit_seconds, total_seconds),
        "other_fraction": _safe_div(other, total_seconds),
    }


def _peak_memory_summary(memory: Mapping[str, Any]) -> dict[str, Any]:
    peak_allocated = _first_nonnegative_int(
        memory.get("peak_allocated_bytes"),
        memory.get("allocator_reserved_peak_bytes"),
        memory.get("tracked_peak_allocated_bytes"),
    )
    peak_reserved = _first_nonnegative_int(memory.get("peak_reserved_bytes"), memory.get("reserved_peak_bytes"))
    after_load = _first_nonnegative_int(memory.get("allocated_after_load_bytes"), memory.get("after_load_bytes"))
    hip_peak = _first_nonnegative_int(memory.get("hip_used_peak_sampled_bytes"), memory.get("hip_peak_used_bytes"))
    return {
        "allocated_after_load_bytes": after_load,
        "allocated_after_load_gib": _bytes_to_gib(after_load),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_gib": _bytes_to_gib(peak_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": _bytes_to_gib(peak_reserved),
        "hip_used_peak_sampled_bytes": hip_peak,
        "hip_used_peak_sampled_gib": _bytes_to_gib(hip_peak),
    }


def _decode_tokens_from_row(row: Mapping[str, Any]) -> int:
    ar = row.get("ar", {}) or {}
    spec = row.get("spec", {}) or {}
    ar_sample = ar.get("generated_sample", ()) or ()
    spec_sample = spec.get("generated_sample", ()) or ()
    return _first_positive_int(
        row.get("decode_tokens"),
        spec.get("decode_tokens"),
        ar.get("decode_tokens"),
        len(ar_sample),
        len(spec_sample),
    ) or 0


def _looks_normalized(row: Mapping[str, Any]) -> bool:
    return all(key in row for key in ("prompt", "config", "ar", "spec", "acceptance", "correctness", "d2h", "graph"))


def _safe_div(numer: Any, denom: Any) -> float | None:
    n = _optional_float(numer)
    d = _optional_float(denom)
    if n is None or d is None or d == 0.0:
        return None
    return n / d


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    parsed = _optional_int(value)
    return int(parsed or 0)


def _int_list(value: Iterable[Any]) -> list[int]:
    return [int(item) for item in value]


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "pass", "passed"}:
            return True
        if lowered in {"false", "0", "no", "n", "fail", "failed"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        parsed = _optional_float(raw)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _first_positive_int(*values: Any) -> int | None:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _first_nonnegative_int(*values: Any) -> int | None:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / float(1 << 30)
