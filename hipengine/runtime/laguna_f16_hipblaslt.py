"""Session-owned hipBLASLt route for Laguna's resident F16 projections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from hipengine.core.hipblaslt import (
    HipblasLt,
    HipblasLtHeuristicResult,
    HipblasLtProblem,
)
from hipengine.kernels.backends import backend_package_capability

_MODES = frozenset(
    {
        "retained",
        "hipblaslt_scaled",
        "hipblaslt_norm_direct",
        "hipblaslt_range_direct",
    }
)
_BASELINE_MODE = "retained"
_BF16_RNE_MAX_FACTOR = 257.0 / 256.0
_FP32_UNIT_ROUNDOFF = 2.0**-24
_ATTENTION_NUMERIC_SAFETY_FACTOR = 2.0
_QUALITY_ALGORITHM_BY_KN = {
    # The default heuristic-4 schedule misses the cumulative all-exact quality
    # gate at 0.05350 KL.  Restricting heuristic 2 to the tiny sliding-attention
    # gate projection restores the gate without slowing the wide projections.
    (3072, 72): 2,
}


def resolve_laguna_f16_prefill_mode(
    backend: str,
    requested: str | None = None,
) -> str:
    """Resolve the retained default or an explicit scaled hipBLASLt candidate."""

    selected = (
        backend_package_capability(
            backend,
            "LAGUNA_F16_PREFILL_MODE",
            _BASELINE_MODE,
        )
        if requested is None
        else str(requested)
    )
    parsed = str(selected)
    if parsed not in _MODES:
        raise ValueError("unsupported Laguna F16 prefill mode")
    return parsed


def laguna_attention_norm_fp16_bound(
    hidden_size: int,
    source_abs_maxima: Iterable[float | None],
) -> float:
    """Return the finite RMSNorm output bound from resident norm metadata."""

    hidden = int(hidden_size)
    if hidden <= 0:
        raise ValueError("Laguna attention norm hidden_size must be positive")
    parsed: list[float] = []
    for value in source_abs_maxima:
        if value is None:
            raise ValueError("Laguna attention norm source abs-max metadata is missing")
        item = float(value)
        if not math.isfinite(item):
            raise ValueError("Laguna attention norm source abs-max metadata must be finite")
        if item < 0.0:
            raise ValueError("Laguna attention norm source abs-max metadata must be nonnegative")
        parsed.append(item)
    if not parsed:
        raise ValueError("Laguna attention norm source abs-max metadata is missing")
    return math.sqrt(hidden) * max(parsed)


def laguna_attention_gated_fp16_bound(
    hidden_size: int,
    layer_metadata: Iterable[
        tuple[float | None, float | None, float | None]
    ],
) -> float:
    """Bound the BF16 gated-attention producer consumed by output projection.

    Each tuple is ``(attention_norm_abs_max, value_row_l2_max,
    gate_row_l2_max)`` for one layer. The bound applies Cauchy-Schwarz to the
    normed F16 projections, includes FP32 dot accumulation and BF16 rounding,
    and reserves 2x for the online-softmax reduction and gate multiply.
    """

    hidden = int(hidden_size)
    if hidden <= 0:
        raise ValueError("Laguna attention hidden_size must be positive")
    dot_denominator = 1.0 - hidden * _FP32_UNIT_ROUNDOFF
    if dot_denominator <= 0.0:
        raise ValueError("Laguna attention hidden_size exceeds FP32 dot bound")
    dot_factor = 1.0 / dot_denominator
    bounds: list[float] = []
    for metadata in layer_metadata:
        if len(metadata) != 3:
            raise ValueError("Laguna attention range metadata must have three fields")
        parsed: list[float] = []
        for value in metadata:
            if value is None:
                raise ValueError("Laguna attention source range metadata is missing")
            item = float(value)
            if not math.isfinite(item):
                raise ValueError("Laguna attention source range metadata must be finite")
            if item < 0.0:
                raise ValueError(
                    "Laguna attention source range metadata must be nonnegative"
                )
            parsed.append(item)
        norm_abs_max, value_row_l2_max, gate_row_l2_max = parsed
        norm_l2_bound = (
            math.sqrt(hidden) * norm_abs_max * _BF16_RNE_MAX_FACTOR
        )
        value_bound = (
            norm_l2_bound
            * value_row_l2_max
            * dot_factor
            * _BF16_RNE_MAX_FACTOR
        )
        gate_bound = norm_l2_bound * gate_row_l2_max * dot_factor
        softplus_bound = (
            gate_bound
            if gate_bound > 20.0
            else math.log1p(math.exp(gate_bound))
        )
        bounds.append(
            value_bound
            * softplus_bound
            * _ATTENTION_NUMERIC_SAFETY_FACTOR
            * _BF16_RNE_MAX_FACTOR
        )
    if not bounds:
        raise ValueError("Laguna attention source range metadata is missing")
    return max(bounds)


@dataclass(frozen=True)
class _CachedProblem:
    problem: HipblasLtProblem
    algorithm: HipblasLtHeuristicResult


class LagunaF16HipblasLt:
    """Cache quality-qualified zero-workspace hipBLASLt descriptors."""

    def __init__(
        self,
        *,
        library_path: str = "libhipblaslt.so",
        preferred_algorithm_index: int = 4,
    ) -> None:
        self.owner = HipblasLt(library_path)
        self.preferred_algorithm_index = int(preferred_algorithm_index)
        self._problems: dict[tuple[int, int, int], _CachedProblem] = {}
        self._closed = False

    def launch(
        self,
        x_ptr: int,
        weight_ptr: int,
        out_ptr: int,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        stream: int = 0,
    ) -> None:
        if self._closed:
            raise RuntimeError("Laguna hipBLASLt route is closed")
        shape = (int(rows), int(in_features), int(out_features))
        cached = self._problems.get(shape)
        if cached is None:
            problem = self.owner.problem(*shape)
            algorithm_index = _QUALITY_ALGORITHM_BY_KN.get(
                (shape[1], shape[2]),
                self.preferred_algorithm_index,
            )
            algorithm = problem.algorithm(algorithm_index)
            if int(algorithm.workspace_size) != 0:
                raise RuntimeError("Laguna hipBLASLt route requires a zero-workspace algorithm")
            cached = _CachedProblem(problem=problem, algorithm=algorithm)
            self._problems[shape] = cached
        cached.problem.launch(
            cached.algorithm,
            int(x_ptr),
            int(weight_ptr),
            int(out_ptr),
            stream=int(stream),
        )

    @property
    def cached_shape_count(self) -> int:
        return len(self._problems)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._problems.clear()
        self.owner.close()


__all__ = [
    "LagunaF16HipblasLt",
    "laguna_attention_norm_fp16_bound",
    "resolve_laguna_f16_prefill_mode",
]
