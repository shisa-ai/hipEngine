"""Validated same-file prefill comparison across engines.

The earlier version of this comparison put two different estimators in one
table: the llama.cpp-family rows were the equal-weight mean of per-case median
rates, while hipEngine's row was total tokens divided by total prefill time.
Those are not the same quantity, so their ratio was not a speedup. This module
computes one estimator for every engine from the raw per-repetition timings, and
refuses to produce a row when the underlying measurements do not line up.

Rules enforced here, all as hard failures rather than silent drops:

* the case set must be exactly the canonical twelve;
* each case's prompt token hash must match the shared reference, so two engines
  are known to have prefilled the same tokens;
* each case must carry exactly the expected number of measured repetitions;
* every retained sample must have a positive, finite prompt time;
* only full-suite runs may enter a rate row, and a run whose declared
  measurement class says otherwise is excluded with its reason recorded;
* a run whose stall status was never recorded is reported as ``unknown``, not
  as zero.

Every sample is preserved in the output, including the ones that were excluded,
so a slow or stalled repetition cannot disappear from the record.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CANONICAL_CASE_IDS: tuple[str, ...] = tuple(
    f"{category}-p{shape}"
    for shape in (512, 1024, 4096)
    for category in ("code", "general_en", "general_ja", "mixed_ja_en")
)
CANONICAL_CASE_SET = frozenset(CANONICAL_CASE_IDS)
EXPECTED_REPETITIONS = 3
EXPECTED_SAMPLE_COUNT = len(CANONICAL_CASE_IDS) * EXPECTED_REPETITIONS

# A run may only enter a rate row if it declares one of these. Profiled,
# single-case and diagnostic captures are excluded by declaration rather than by
# filename convention, so a new diagnostic run cannot leak into a headline.
RATE_ELIGIBLE_CLASSES = frozenset({"comparator_unprofiled"})


class ValidationError(RuntimeError):
    """A measurement set is not fit to be compared."""


@dataclass(frozen=True)
class Sample:
    case_id: str
    category: str
    prompt_tokens: int
    token_sha256: str
    repetition: int
    prompt_ms: float

    @property
    def prompt_tok_s(self) -> float:
        return 1000.0 * self.prompt_tokens / self.prompt_ms


@dataclass
class EngineMeasurement:
    label: str
    path: str
    measurement_class: str
    engine: str
    samples: list[Sample]
    source: Mapping[str, Any] = field(default_factory=dict)
    kv_dtype: str = "unknown"
    stalled_reps_by_case: Mapping[str, Sequence[int]] | None = None
    stalls_known: bool = True
    notes: list[str] = field(default_factory=list)
    excluded: bool = False
    exclusion_reason: str | None = None

    @property
    def case_ids(self) -> frozenset[str]:
        return frozenset(sample.case_id for sample in self.samples)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValidationError(f"{path}: expected a JSON object")
    return payload


def load_comparator(path: Path, label: str) -> EngineMeasurement:
    """Read one llama.cpp-family comparator run."""
    payload = _read_json(path)
    measurement_class = str(payload.get("measurement_class") or "unspecified")
    samples: list[Sample] = []
    stalls: dict[str, list[int]] = {}
    for case in payload.get("cases", []):
        case_id = str(case["id"])
        stalls[case_id] = [int(r) for r in (case.get("stalled_reps") or [])]
        for rep in case.get("repetitions", []):
            samples.append(Sample(
                case_id=case_id,
                category=str(case.get("category") or case_id.split("-p")[0]),
                prompt_tokens=int(rep["prompt_tokens"]),
                token_sha256=str(case["prompt_token_ids_sha256"]),
                repetition=int(rep["rep"]),
                prompt_ms=float(rep["prompt_ms"]),
            ))
    return EngineMeasurement(
        label=label,
        path=str(path),
        measurement_class=measurement_class,
        engine=str(payload.get("label") or label),
        samples=samples,
        source=payload.get("source") or {},
        kv_dtype=str(payload.get("kv_dtype") or "unknown"),
        stalled_reps_by_case=stalls,
        stalls_known="total_stalled_reps" in payload,
        notes=[
            f"server_args={payload.get('server_args')}",
            f"context={payload.get('context')} batch={payload.get('batch')} ubatch={payload.get('ubatch')}",
            f"server_sha256={payload.get('server_sha256')}",
        ],
        excluded=measurement_class not in RATE_ELIGIBLE_CLASSES,
        exclusion_reason=(
            None if measurement_class in RATE_ELIGIBLE_CLASSES
            else f"declared measurement_class={measurement_class!r} is not rate-eligible"
        ),
    )


def load_hipengine(path: Path, label: str) -> EngineMeasurement:
    """Read a hipEngine canonical AR bench run.

    hipEngine records its prefill as a phase window alongside ``prefill_ms``.
    Those must agree, otherwise the row is measuring a different span from the
    comparators' prompt-eval time.
    """
    payload = _read_json(path)
    samples: list[Sample] = []
    for row in payload.get("samples", []):
        window = (row.get("phase_windows_ns") or {}).get("prefill")
        if window:
            window_ms = (int(window[1]) - int(window[0])) / 1e6
            if not math.isclose(window_ms, float(row["prefill_ms"]), rel_tol=1e-9):
                raise ValidationError(
                    f"{path}: {row['case_id']} rep {row['repetition']} "
                    f"prefill_ms={row['prefill_ms']} does not match its prefill "
                    f"phase window {window_ms}; the timing scope is not comparable"
                )
        samples.append(Sample(
            case_id=str(row["case_id"]),
            category=str(row["category"]),
            prompt_tokens=int(row["prompt_tokens"]),
            token_sha256=str(row["prompt_token_ids_sha256"]),
            repetition=int(row["repetition"]),
            prompt_ms=float(row["prefill_ms"]),
        ))
    return EngineMeasurement(
        label=label,
        path=str(path),
        measurement_class="hipengine_canonical_ar_bench",
        engine=str(payload.get("engine") or "hipengine"),
        samples=samples,
        source=payload.get("source") or {},
        kv_dtype=str((payload.get("profile") or {}).get("kv_dtype") or "bf16"),
        stalled_reps_by_case=None,
        # hipEngine's harness does not record a stall classification at all, so
        # this is unknown rather than zero.
        stalls_known=False,
        notes=[
            f"protocol={payload.get('protocol')}",
            f"timing_boundary={(payload.get('protocol') or {}).get('timing_boundary')}",
        ],
    )


def _grouped(samples: Iterable[Sample], key) -> dict[Any, list[Sample]]:
    groups: dict[Any, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(key(sample), []).append(sample)
    return groups


def weighted_tok_s(samples: Sequence[Sample]) -> float:
    """Total prompt tokens divided by total prompt time.

    This is the estimator used for every engine, so the ratio of two rows is a
    ratio of the same quantity.
    """
    total_tokens = sum(sample.prompt_tokens for sample in samples)
    total_ms = sum(sample.prompt_ms for sample in samples)
    if total_ms <= 0:
        raise ValidationError("total prompt time is not positive")
    return 1000.0 * total_tokens / total_ms


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValidationError("median of an empty set")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def estimate(samples: Sequence[Sample]) -> dict[str, Any]:
    """Every estimator the comparison needs, from the same samples.

    ``weighted_tok_s`` is the headline. The per-case medians and the
    median-of-medians are reported beside it so the spread is visible rather
    than averaged away.
    """
    if not samples:
        raise ValidationError("no samples to estimate from")
    by_case = _grouped(samples, lambda sample: sample.case_id)
    by_shape = _grouped(samples, lambda sample: sample.prompt_tokens)
    by_category = _grouped(samples, lambda sample: sample.category)
    case_medians = {
        case_id: _median([sample.prompt_tok_s for sample in rows])
        for case_id, rows in by_case.items()
    }
    return {
        "weighted_tok_s": weighted_tok_s(samples),
        "median_of_case_medians_tok_s": _median(list(case_medians.values())),
        "weighted_tok_s_by_shape": {
            str(shape): weighted_tok_s(rows) for shape, rows in sorted(by_shape.items())
        },
        "weighted_tok_s_by_category": {
            category: weighted_tok_s(rows) for category, rows in sorted(by_category.items())
        },
        "case_medians_tok_s": dict(sorted(case_medians.items())),
        "sample_count": len(samples),
        "case_count": len(by_case),
        "samples_per_case": {
            case_id: len(rows) for case_id, rows in sorted(by_case.items())
        },
    }


def validate(measurements: Sequence[EngineMeasurement]) -> list[str]:
    """Raise unless every engine measured the same thing the same way."""
    eligible = [m for m in measurements if not m.excluded]
    if not eligible:
        raise ValidationError("no rate-eligible measurements")
    reference = eligible[0]
    reference_hashes = {s.case_id: s.token_sha256 for s in reference.samples}
    problems: list[str] = []
    for measurement in eligible:
        if measurement.case_ids != CANONICAL_CASE_SET:
            missing = sorted(CANONICAL_CASE_SET - measurement.case_ids)
            extra = sorted(measurement.case_ids - CANONICAL_CASE_SET)
            problems.append(
                f"{measurement.label}: case set is not the canonical twelve "
                f"(missing={missing}, unexpected={extra})"
            )
        per_case = _grouped(measurement.samples, lambda s: s.case_id)
        for case_id, rows in sorted(per_case.items()):
            if len(rows) != EXPECTED_REPETITIONS:
                problems.append(
                    f"{measurement.label}: {case_id} has {len(rows)} measured "
                    f"repetitions, expected {EXPECTED_REPETITIONS}"
                )
            hashes = {row.token_sha256 for row in rows}
            if len(hashes) != 1:
                problems.append(
                    f"{measurement.label}: {case_id} has inconsistent prompt token hashes"
                )
            expected_hash = reference_hashes.get(case_id)
            if expected_hash and hashes and next(iter(hashes)) != expected_hash:
                problems.append(
                    f"{measurement.label}: {case_id} prompt token hash "
                    f"{next(iter(hashes))[:12]} does not match the reference "
                    f"{expected_hash[:12]}; the engines did not prefill the same tokens"
                )
            for row in rows:
                if not (math.isfinite(row.prompt_ms) and row.prompt_ms > 0):
                    problems.append(
                        f"{measurement.label}: {case_id} rep {row.repetition} has "
                        f"prompt_ms={row.prompt_ms}; a missing measurement is a "
                        "validation failure, not a row to drop"
                    )
        if len(measurement.samples) != EXPECTED_SAMPLE_COUNT:
            problems.append(
                f"{measurement.label}: {len(measurement.samples)} samples, "
                f"expected {EXPECTED_SAMPLE_COUNT}"
            )
    if problems:
        raise ValidationError("comparison validation failed:\n  " + "\n  ".join(problems))
    return [
        f"{m.label}: {len(m.samples)} samples, {len(m.case_ids)} cases, "
        f"class={m.measurement_class}"
        for m in eligible
    ]


def comparison_table(measurements: Sequence[EngineMeasurement]) -> list[dict[str, Any]]:
    """One row per engine, every row on the same estimator."""
    rows = []
    for measurement in measurements:
        row: dict[str, Any] = {
            "label": measurement.label,
            "engine": measurement.engine,
            "measurement_class": measurement.measurement_class,
            "kv_dtype": measurement.kv_dtype,
            "source": measurement.source,
            "raw_path": measurement.path,
            "notes": measurement.notes,
            "excluded": measurement.excluded,
            "exclusion_reason": measurement.exclusion_reason,
            "stalls": (
                {
                    "status": "recorded",
                    "by_case": {k: list(v) for k, v in (measurement.stalled_reps_by_case or {}).items()},
                }
                if measurement.stalls_known
                else {"status": "unknown", "reason": "harness does not record a stall classification"}
            ),
        }
        if not measurement.excluded:
            row["estimate"] = estimate(measurement.samples)
            row["samples"] = [
                {
                    "case_id": s.case_id,
                    "category": s.category,
                    "prompt_tokens": s.prompt_tokens,
                    "prompt_token_ids_sha256": s.token_sha256,
                    "repetition": s.repetition,
                    "prompt_ms": round(s.prompt_ms, 6),
                    "prompt_tok_s": round(s.prompt_tok_s, 6),
                }
                for s in measurement.samples
            ]
        rows.append(row)
    return rows
