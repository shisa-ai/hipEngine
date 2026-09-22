"""Frozen multi-sequence G0/G1/G2 distribution evaluator for the DMS selector campaign.

This module owns the campaign gate arithmetic defined in
``docs/campaigns/DMS-SELECTOR-IMPROVEMENT.md`` Section 4.  It is deliberately
pure: numpy plus the standard library, no model, GPU, capture, or sidecar
dependency, and no knowledge of any candidate implementation.  The evaluator
score is ``KL(dense teacher || candidate)`` over the full vocabulary; a
candidate mode supplies logits and this module decides pass/fail.

Frozen thresholds (do not retune to rescue a failed candidate):

- **G0** integrity/control — compact no-evict versus dense on the same
  evaluated rows: max KL <= 0.001 and 100% top-1, prefill and decode, plus
  finiteness.
- **G1** diagnostic screen — decode max KL <= 0.05 and top-1 >= 0.90 in every
  category, finite prefill and decode.
- **G2** finalist distribution — decode mean <= 0.001, p95 <= 0.005,
  p99 <= 0.02, max <= 0.05 globally and per category; top-1 >= 0.99 globally
  and >= 0.97 per category; prefill max KL <= 0.05 and finite globally and per
  category.  No scope may be averaged away by another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence as SequenceType

import numpy as np

CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")

DIAGNOSTIC_KL_FLOOR = 0.02
_TOPK = 10

GATE_THRESHOLDS: dict[str, dict[str, Any]] = {
    "g0": {
        "name": "integrity/control: compact no-evict versus dense",
        "combined": {"max_kl": 0.001, "min_top1": 1.0},
    },
    "g1": {
        "name": "diagnostic screen",
        "decode": {"max_kl": 0.05, "min_top1_per_category": 0.9},
    },
    "g2": {
        "name": "finalist distribution",
        "decode": {
            "mean_kl": 0.001,
            "p95_kl": 0.005,
            "p99_kl": 0.02,
            "max_kl": 0.05,
            "min_top1_global": 0.99,
            "min_top1_per_category": 0.97,
        },
        "prefill": {"max_kl": 0.05},
    },
}


# ---------------------------------------------------------------------------
# Row-level comparison


def log_softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - float(np.max(array))
    return shifted - float(np.log(np.exp(shifted).sum()))


def compare_row(
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    topk: int = _TOPK,
) -> dict[str, Any]:
    """Compare one dense-teacher row against one candidate row over the full vocabulary."""
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if teacher.shape != candidate.shape or teacher.ndim != 1 or teacher.size == 0:
        raise ValueError("teacher/candidate logits must be non-empty matching 1-D arrays")
    finite = bool(np.isfinite(candidate).all() and np.isfinite(teacher).all())
    teacher_top1 = int(np.argmax(teacher))
    candidate_top1 = int(np.argmax(candidate))
    row: dict[str, Any] = {
        "teacher_top1": teacher_top1,
        "candidate_top1": candidate_top1,
        "top1_agrees": teacher_top1 == candidate_top1,
        "finite_candidate_logits": finite,
        "winners": {"teacher": teacher_top1, "candidate": candidate_top1},
    }
    if not finite:
        row["kl"] = float("inf")
        row["strict_margin"] = float("inf")
        row["candidate_rank_of_strict_winner"] = -1
        row["topk_overlap"] = 0.0
        row["max_abs_logit_delta"] = float("inf")
        return row
    teacher_logp = log_softmax(teacher)
    candidate_logp = log_softmax(candidate)
    row["kl"] = float(np.sum(np.exp(teacher_logp) * (teacher_logp - candidate_logp)))
    # Strict margin: how far the dense teacher's winner leads the candidate's
    # winner under the teacher distribution.
    row["strict_margin"] = float(teacher_logp[teacher_top1] - teacher_logp[candidate_top1])
    # Rank of the teacher's winner in the candidate's descending logit order.
    order = np.argsort(-candidate)
    rank = int(np.where(order == teacher_top1)[0][0])
    row["candidate_rank_of_strict_winner"] = rank
    k = min(int(topk), teacher.size)
    teacher_top = set(np.argsort(-teacher)[:k].tolist())
    candidate_top = set(np.argsort(-candidate)[:k].tolist())
    row["topk_overlap"] = float(len(teacher_top & candidate_top) / k)
    row["max_abs_logit_delta"] = float(np.max(np.abs(teacher - candidate)))
    return row


def summarize_rows(rows: SequenceType[dict[str, Any]]) -> dict[str, Any]:
    """Summarize KL and top-1 statistics over a scope's rows."""
    if not rows:
        return {
            "rows": 0,
            "mean_kl": float("inf"),
            "p95_kl": float("inf"),
            "p99_kl": float("inf"),
            "max_kl": float("inf"),
            "top1_agreement": 0.0,
            "finite_logits": False,
        }
    kls = np.asarray([float(row["kl"]) for row in rows], dtype=np.float64)
    finite = all(bool(row.get("finite_candidate_logits")) for row in rows)
    return {
        "rows": len(rows),
        "mean_kl": float(np.mean(kls)),
        "p95_kl": float(np.quantile(kls, 0.95)),
        "p99_kl": float(np.quantile(kls, 0.99)),
        "max_kl": float(np.max(kls)),
        "top1_agreement": float(np.mean([bool(row.get("top1_agrees")) for row in rows])),
        "finite_logits": finite,
    }


# ---------------------------------------------------------------------------
# Manifest loading


def load_manifest(
    path: Path,
    *,
    split: str | None = None,
    categories: Iterable[str] | None = None,
    expected_sequences_per_category: int | None = None,
    prompt_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Load one evaluated manifest, filtering by split and category.

    Every selected sequence is returned as its own record so callers evaluate
    sequences separately; concatenation is not supported by design.  When
    ``expected_sequences_per_category`` is declared, each requested category
    must supply exactly that many sequences or the run refuses to score.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_sequences = payload.get("sequences")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError(f"manifest {path} contains no sequences")
    manifest_split = payload.get("split")
    selected_categories = tuple(categories) if categories is not None else CATEGORIES
    if any(category not in CATEGORIES for category in selected_categories):
        raise ValueError(
            "unsupported category in filter: "
            + ",".join(c for c in selected_categories if c not in CATEGORIES)
        )
    if split is not None and manifest_split is not None and str(manifest_split) != str(split):
        raise ValueError(
            f"manifest split {manifest_split!r} does not match requested split {split!r}"
        )
    records: list[dict[str, Any]] = []
    for raw in raw_sequences:
        sequence_split = str(raw.get("split", manifest_split if manifest_split is not None else split))
        if split is not None and sequence_split != str(split):
            continue
        category = str(raw.get("category"))
        if category not in selected_categories:
            continue
        token_ids = raw.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(
                f"sequence {raw.get('sequence_id')!r} has no token_ids; manifests without "
                "per-sequence tokens cannot be evaluated per sequence"
            )
        tokens = [int(token) for token in token_ids]
        if prompt_tokens is not None:
            if len(tokens) < int(prompt_tokens):
                raise ValueError(
                    f"sequence {raw.get('sequence_id')!r} has {len(tokens)} tokens, "
                    f"fewer than prompt-tokens {int(prompt_tokens)}"
                )
            tokens = tokens[: int(prompt_tokens)]
        provenance = raw.get("provenance") or {}
        records.append(
            {
                "sequence_id": str(raw.get("sequence_id")),
                "category": category,
                "split": sequence_split,
                "token_ids": tokens,
                "source_id": str(provenance.get("source_id", raw.get("source_id", ""))),
                "normalized_text_sha256": str(
                    provenance.get("normalized_text_sha256", raw.get("normalized_text_sha256", ""))
                ),
            }
        )
    if not records:
        raise ValueError("data manifest filters select no prompt sequences")
    if expected_sequences_per_category is not None:
        expected = int(expected_sequences_per_category)
        for category in selected_categories:
            actual = sum(1 for record in records if record["category"] == category)
            if actual != expected:
                raise ValueError(
                    f"category {category}: manifest supplies {actual} sequences, "
                    f"expected {expected}; refusing to score a partial suite"
                )
    return records


# ---------------------------------------------------------------------------
# Correlated prefixes are not independent prompts


def _common_prefix_length(a: SequenceType[int], b: SequenceType[int]) -> int:
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def sequence_correlations(
    sequences: SequenceType[dict[str, Any]],
    *,
    min_shared_fraction: float = 0.5,
) -> dict[str, Any]:
    """Group sequences whose token prefixes repeat or whose normalized text hashes match.

    Repeated prefixes at different lengths are correlated samples, not new
    sources; the evaluator reports group structure instead of counting every
    sequence as an independent prompt.
    """
    count = len(sequences)
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(count):
        for j in range(i + 1, count):
            a, b = sequences[i], sequences[j]
            text_a = str(a.get("normalized_text_sha256", ""))
            text_b = str(b.get("normalized_text_sha256", ""))
            if text_a and text_a == text_b:
                union(i, j)
                continue
            tokens_a = a.get("token_ids") or []
            tokens_b = b.get("token_ids") or []
            if not tokens_a or not tokens_b:
                continue
            shared = _common_prefix_length(tokens_a, tokens_b)
            threshold = min_shared_fraction * min(len(tokens_a), len(tokens_b))
            if shared >= max(1, threshold):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    correlated = []
    for members in groups.values():
        if len(members) < 2:
            continue
        prefix = sequences[members[0]].get("token_ids") or []
        shared = len(prefix)
        for member in members[1:]:
            shared = _common_prefix_length(prefix, sequences[member].get("token_ids") or [])
            prefix = (sequences[members[0]].get("token_ids") or [])[:shared]
        correlated.append(
            {
                "members": sorted(str(sequences[m]["sequence_id"]) for m in members),
                "shared_prefix_tokens": int(shared),
            }
        )
    return {
        "sequence_count": count,
        "independent_source_groups": len(groups),
        "correlated_groups": correlated,
        "note": (
            "rows from correlated groups are repeated-prefix samples and are "
            "reported as correlated, not independent prompts"
        ),
    }


# ---------------------------------------------------------------------------
# Gate evaluation


def _check(
    checks: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    scope: str,
    phase: str,
    check: str,
    value: Any,
    limit: Any,
    direction: str,
) -> None:
    if direction == "<=":
        passed = bool(value <= limit)
    else:
        passed = bool(value >= limit)
    entry = {
        "scope": scope,
        "phase": phase,
        "check": check,
        "value": float(value),
        "limit": float(limit),
        "passed": passed,
    }
    checks.append(entry)
    if not passed:
        failures.append(entry)


def _mismatch_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence_id": row.get("sequence_id"),
        "category": row.get("category"),
        "step": row.get("step"),
        "phase": row.get("phase", "decode"),
        "teacher_top1": row.get("teacher_top1"),
        "candidate_top1": row.get("candidate_top1"),
        "diagnostics": {
            "winners": row.get("winners"),
            "strict_margin": row.get("strict_margin"),
            "candidate_rank_of_strict_winner": row.get("candidate_rank_of_strict_winner"),
            "topk_overlap": row.get("topk_overlap"),
            "max_abs_logit_delta": row.get("max_abs_logit_delta"),
        },
    }


def evaluate_gate(
    gate: str,
    *,
    prefill_rows: SequenceType[dict[str, Any]],
    decode_rows: SequenceType[dict[str, Any]],
    categories: Iterable[str] | None = None,
    correlations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one frozen gate to the scored prefill and decode rows.

    Every row must carry ``sequence_id``, ``category``, ``phase``
    (``prefill`` or ``decode``), ``step`` (``None`` for prefill), ``kl``,
    ``top1_agrees``, and ``finite_candidate_logits``, plus the
    :func:`compare_row` diagnostics.
    """
    if gate not in GATE_THRESHOLDS:
        raise ValueError(f"unknown gate {gate!r}; expected one of {sorted(GATE_THRESHOLDS)}")
    prefill = list(prefill_rows)
    decode = list(decode_rows)
    evaluated_categories = tuple(categories) if categories is not None else CATEGORIES
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    scopes: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = [
        ("global", prefill, decode)
    ]
    for category in evaluated_categories:
        scopes.append(
            (
                f"category:{category}",
                [row for row in prefill if row.get("category") == category],
                [row for row in decode if row.get("category") == category],
            )
        )

    # Row coverage: every declared category must supply prefill and decode rows.
    coverage = {
        "scope": "global",
        "phase": "coverage",
        "check": "row_coverage",
        "categories": list(evaluated_categories),
        "prefill_rows": len(prefill),
        "decode_rows": len(decode),
        "per_category": {
            category: {
                "prefill": sum(1 for row in prefill if row.get("category") == category),
                "decode": sum(1 for row in decode if row.get("category") == category),
            }
            for category in evaluated_categories
        },
    }
    checks.append(coverage)
    missing = [
        category
        for category in evaluated_categories
        if not any(row.get("category") == category for row in prefill)
        or not any(row.get("category") == category for row in decode)
    ]
    if missing:
        failure = dict(coverage)
        failure["passed"] = False
        failure["missing_categories"] = missing
        failures.append(failure)

    if gate == "g0":
        thresholds = GATE_THRESHOLDS["g0"]["combined"]
        for scope, scope_prefill, scope_decode in scopes:
            rows = scope_prefill + scope_decode
            summary = summarize_rows(rows)
            _check(checks, failures, scope=scope, phase="combined", check="finite_logits",
                   value=summary["finite_logits"], limit=1.0, direction=">=")
            _check(checks, failures, scope=scope, phase="combined", check="max_kl",
                   value=summary["max_kl"], limit=thresholds["max_kl"], direction="<=")
            _check(checks, failures, scope=scope, phase="combined", check="min_top1",
                   value=summary["top1_agreement"], limit=thresholds["min_top1"], direction=">=")
    elif gate == "g1":
        thresholds = GATE_THRESHOLDS["g1"]["decode"]
        for scope, _scope_prefill, scope_decode in scopes:
            summary = summarize_rows(scope_decode)
            _check(checks, failures, scope=scope, phase="decode", check="finite_logits",
                   value=summary["finite_logits"], limit=1.0, direction=">=")
            _check(checks, failures, scope=scope, phase="decode", check="max_kl",
                   value=summary["max_kl"], limit=thresholds["max_kl"], direction="<=")
            _check(checks, failures, scope=scope, phase="decode", check="min_top1",
                   value=summary["top1_agreement"], limit=thresholds["min_top1_per_category"], direction=">=")
        prefill_summary = summarize_rows(prefill)
        _check(checks, failures, scope="global", phase="prefill", check="finite_logits",
               value=prefill_summary["finite_logits"], limit=1.0, direction=">=")
    elif gate == "g2":
        decode_thresholds = GATE_THRESHOLDS["g2"]["decode"]
        prefill_thresholds = GATE_THRESHOLDS["g2"]["prefill"]
        for scope, scope_prefill, scope_decode in scopes:
            decode_summary = summarize_rows(scope_decode)
            _check(checks, failures, scope=scope, phase="decode", check="finite_logits",
                   value=decode_summary["finite_logits"], limit=1.0, direction=">=")
            _check(checks, failures, scope=scope, phase="decode", check="mean_kl",
                   value=decode_summary["mean_kl"], limit=decode_thresholds["mean_kl"], direction="<=")
            _check(checks, failures, scope=scope, phase="decode", check="p95_kl",
                   value=decode_summary["p95_kl"], limit=decode_thresholds["p95_kl"], direction="<=")
            _check(checks, failures, scope=scope, phase="decode", check="p99_kl",
                   value=decode_summary["p99_kl"], limit=decode_thresholds["p99_kl"], direction="<=")
            _check(checks, failures, scope=scope, phase="decode", check="max_kl",
                   value=decode_summary["max_kl"], limit=decode_thresholds["max_kl"], direction="<=")
            min_top1 = (
                decode_thresholds["min_top1_global"]
                if scope == "global"
                else decode_thresholds["min_top1_per_category"]
            )
            _check(checks, failures, scope=scope, phase="decode", check="min_top1",
                   value=decode_summary["top1_agreement"], limit=min_top1, direction=">=")
            prefill_summary = summarize_rows(scope_prefill)
            _check(checks, failures, scope=scope, phase="prefill", check="finite_logits",
                   value=prefill_summary["finite_logits"], limit=1.0, direction=">=")
            _check(checks, failures, scope=scope, phase="prefill", check="max_kl",
                   value=prefill_summary["max_kl"], limit=prefill_thresholds["max_kl"], direction="<=")

    all_rows = prefill + decode
    hot_rows = [
        {
            "sequence_id": row.get("sequence_id"),
            "category": row.get("category"),
            "step": row.get("step"),
            "phase": row.get("phase", "decode"),
            "kl": float(row["kl"]),
        }
        for row in all_rows
        if np.isfinite(float(row["kl"])) and float(row["kl"]) > DIAGNOSTIC_KL_FLOOR
    ]
    mismatches = [_mismatch_entry(row) for row in all_rows if not bool(row.get("top1_agrees"))]
    correlation = dict(correlations) if correlations else {
        "sequence_count": len({row.get("sequence_id") for row in all_rows}),
        "independent_source_groups": len({row.get("sequence_id") for row in all_rows}),
        "correlated_groups": [],
        "note": "no correlation input supplied",
    }
    by_sequence: dict[str, int] = {}
    for row in decode:
        key = str(row.get("sequence_id"))
        by_sequence[key] = by_sequence.get(key, 0) + 1
    row_counts = {
        "prefill_rows": len(prefill),
        "decode_rows": len(decode),
        "decode_rows_by_sequence": by_sequence,
        "decode_rows_by_category": {
            category: sum(1 for row in decode if row.get("category") == category)
            for category in evaluated_categories
        },
        "sequence_count": correlation.get("sequence_count"),
        "independent_source_groups": correlation.get("independent_source_groups"),
        "correlated_groups": correlation.get("correlated_groups"),
        "correlation_note": correlation.get(
            "note",
            "rows from correlated groups are repeated-prefix samples and are "
            "reported as correlated, not independent prompts",
        ),
    }
    return {
        "gate": gate,
        "gate_name": GATE_THRESHOLDS[gate]["name"],
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "thresholds": GATE_THRESHOLDS[gate],
        "rows_above_kl_0_02": hot_rows,
        "top1_mismatches": mismatches,
        "row_counts": row_counts,
        "scope_summaries": {
            scope: {
                "prefill": summarize_rows(scope_prefill),
                "decode": summarize_rows(scope_decode),
            }
            for scope, scope_prefill, scope_decode in scopes
        },
    }


# ---------------------------------------------------------------------------
# DMS observability digest


def dms_digest(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact capacity/mask/count digest sufficient for route verification."""
    backend = dict(snapshot.get("backend", {}))
    capacity = dict(snapshot.get("capacity", {}))
    digest = hashlib.sha256(
        json.dumps({"backend": backend, "capacity": capacity}, sort_keys=True).encode()
    ).hexdigest()
    return {
        "backend": {
            "topology": backend.get("topology"),
            "codec": backend.get("codec"),
            "codec_evaluation_only": backend.get("codec_evaluation_only"),
            "decision_source": backend.get("decision_source"),
            "device_payloads": backend.get("device_payloads"),
            "physical_layer_ids": backend.get("physical_layer_ids"),
        },
        "capacity": {
            "logical_token_rows": capacity.get("logical_token_rows"),
            "live_token_rows": capacity.get("live_token_rows"),
            "actual_compression_ratio": capacity.get("actual_compression_ratio"),
            "target_compression_ratio": capacity.get("target_compression_ratio"),
            "max_live_count": capacity.get("max_live_count"),
            "payload_bytes": capacity.get("payload_bytes"),
        },
        "digest_sha256": digest,
    }
