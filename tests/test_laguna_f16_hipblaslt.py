from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.laguna_f16_hipblaslt as route_module
from hipengine.runtime.laguna_f16_hipblaslt import (
    LagunaF16HipblasLt,
    resolve_laguna_f16_prefill_mode,
)


def test_laguna_f16_hipblaslt_mode_is_explicit_and_validated() -> None:
    assert resolve_laguna_f16_prefill_mode("hip_gfx1100") == "retained"
    assert resolve_laguna_f16_prefill_mode("hip_gfx1151") == "hipblaslt_scaled"
    assert (
        resolve_laguna_f16_prefill_mode("hip_gfx1151", "retained")
        == "retained"
    )
    assert (
        resolve_laguna_f16_prefill_mode("hip_gfx1151", "hipblaslt_scaled")
        == "hipblaslt_scaled"
    )
    with pytest.raises(ValueError, match="unsupported"):
        resolve_laguna_f16_prefill_mode("hip_gfx1151", "unknown")


def test_laguna_f16_hipblaslt_caches_shape_descriptors(monkeypatch) -> None:
    events: list[tuple] = []

    class Problem:
        def algorithm(self, preferred_index):
            events.append(("algorithm", preferred_index))
            return SimpleNamespace(workspace_size=0)

        def launch(self, algorithm, x_ptr, weight_ptr, out_ptr, *, stream):
            events.append(
                ("launch", algorithm.workspace_size, x_ptr, weight_ptr, out_ptr, stream)
            )

    class Owner:
        def __init__(self, path):
            events.append(("owner", path))

        def problem(self, rows, in_features, out_features):
            events.append(("problem", rows, in_features, out_features))
            return Problem()

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(route_module, "HipblasLt", Owner)
    route = LagunaF16HipblasLt(preferred_algorithm_index=4)
    route.launch(10, 20, 30, 512, 3072, 9216, stream=7)
    route.launch(11, 21, 31, 512, 3072, 9216, stream=8)
    assert route.cached_shape_count == 1
    route.close()
    route.close()

    assert events == [
        ("owner", "libhipblaslt.so"),
        ("problem", 512, 3072, 9216),
        ("algorithm", 4),
        ("launch", 0, 10, 20, 30, 7),
        ("launch", 0, 11, 21, 31, 8),
        ("close",),
    ]
