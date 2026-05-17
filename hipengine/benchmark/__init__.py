"""Benchmark and correctness helpers."""

from hipengine.benchmark.correctness import LogitCorrectness, evaluate_logits
from hipengine.benchmark.speculative import (
    D2HCounts,
    SpeculativeBenchmarkModels,
    SpeculativeGraphStatus,
    acceptance_summary,
    aggregate_speculative_rows,
    build_speculative_artifact,
    normalize_speculative_row,
)

__all__ = [
    "D2HCounts",
    "LogitCorrectness",
    "SpeculativeBenchmarkModels",
    "SpeculativeGraphStatus",
    "acceptance_summary",
    "aggregate_speculative_rows",
    "build_speculative_artifact",
    "evaluate_logits",
    "normalize_speculative_row",
]
