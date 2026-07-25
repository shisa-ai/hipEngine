"""Session-owned hipBLASLt route for Laguna's resident F16 projections."""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.hipblaslt import (
    HipblasLt,
    HipblasLtHeuristicResult,
    HipblasLtProblem,
)
from hipengine.kernels.backends import backend_package_capability

_MODES = frozenset({"retained", "hipblaslt_scaled"})
_BASELINE_MODE = "retained"
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
    "resolve_laguna_f16_prefill_mode",
]
