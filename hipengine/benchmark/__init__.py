"""Benchmark and correctness helpers."""

from hipengine.benchmark.correctness import LogitCorrectness, evaluate_logits
from hipengine.benchmark.exact_tokens import (
    DEFAULT_EXACT_TOKEN_FIXTURE,
    EXACT_TOKEN_ORACLE_KIND,
    EXACT_TOKEN_ORACLE_SCHEMA_VERSION,
    ExactTokenFixture,
    ExactTokenOracle,
    load_exact_token_fixture,
    validate_exact_token_parity,
)
from hipengine.benchmark.matrix import (
    BENCHMARK_MATRIX_KIND,
    BENCHMARK_MATRIX_MANIFEST_KIND,
    BENCHMARK_MATRIX_SCHEMA_VERSION,
    MatrixError,
    build_benchmark_matrix,
    validate_benchmark_matrix,
)
from hipengine.benchmark.prompts import (
    DEFAULT_STABLE_PROMPT_FIXTURE,
    STABLE_PROMPT_SPECS,
    StablePromptSpec,
    build_prompt_records,
    load_prompt_records,
    token_ids_sha256,
    validate_prompt_records,
)
from hipengine.benchmark.provenance import (
    ARTIFACT_PROVENANCE_KIND,
    ARTIFACT_PROVENANCE_SCHEMA_VERSION,
    collect_artifact_provenance,
    collect_model_identity,
    collect_repo_state,
    validate_artifact_provenance,
)
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
    "ARTIFACT_PROVENANCE_KIND",
    "ARTIFACT_PROVENANCE_SCHEMA_VERSION",
    "BENCHMARK_MATRIX_KIND",
    "BENCHMARK_MATRIX_MANIFEST_KIND",
    "BENCHMARK_MATRIX_SCHEMA_VERSION",
    "DEFAULT_STABLE_PROMPT_FIXTURE",
    "DEFAULT_EXACT_TOKEN_FIXTURE",
    "D2HCounts",
    "EXACT_TOKEN_ORACLE_KIND",
    "EXACT_TOKEN_ORACLE_SCHEMA_VERSION",
    "ExactTokenFixture",
    "ExactTokenOracle",
    "LogitCorrectness",
    "MatrixError",
    "STABLE_PROMPT_SPECS",
    "SpeculativeBenchmarkModels",
    "SpeculativeGraphStatus",
    "StablePromptSpec",
    "acceptance_summary",
    "aggregate_speculative_rows",
    "build_prompt_records",
    "build_benchmark_matrix",
    "build_speculative_artifact",
    "collect_artifact_provenance",
    "collect_model_identity",
    "collect_repo_state",
    "evaluate_logits",
    "load_prompt_records",
    "normalize_speculative_row",
    "token_ids_sha256",
    "load_exact_token_fixture",
    "validate_exact_token_parity",
    "validate_artifact_provenance",
    "validate_benchmark_matrix",
    "validate_prompt_records",
]
