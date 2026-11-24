"""Device route-tile builder must match build_expert_route_tiles exactly."""

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_OK = True
except OSError:
    _HIP_OK = False

pytestmark = pytest.mark.skipif(not _HIP_OK, reason="HIP runtime not available")

from hipengine.kernels.backends import hip_target_arch_for_backend, resolve_backend
from hipengine.kernels.hip_gfx1100.moe.inkling2_route_tiles import build_expert_route_tiles_device
from hipengine.loading.inkling2_device_experts import build_expert_route_tiles


@pytest.fixture(scope="module", autouse=True)
def require_gfx():
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip(f"requires gfx1100/gfx1151, got {backend}")


def _host(ids, width):
    return np.asarray(build_expert_route_tiles(ids, width), dtype=np.int32).reshape(-1, width + 1)


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize(
    "case_ids",
    [
        [0],
        [7, 7, 7],
        [3, 1, 3, 1, 3, 1, 3],
        list(range(8)),
        [5, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 5, 3, 7],
        list(np.random.default_rng(1234).integers(0, 8, size=64)),
        list(np.random.default_rng(987).integers(0, 128, size=512)),
        list(np.random.default_rng(4242).integers(0, 128, size=1024)),
    ],
    ids=["single", "one-expert", "two-experts-interleaved", "eight-distinct", "mixed-15",
         "random-64", "random-512", "random-1024"],
)
def test_tiles_match_host_builder(width, case_ids):
    ids = np.asarray(case_ids, dtype=np.int64)
    expected = _host(ids, width)
    records, scales = build_expert_route_tiles_device(ids, 128, width=width)
    assert scales is None
    assert records.shape == expected.shape
    assert records.tolist() == expected.tolist()


@pytest.mark.parametrize("width", [2, 4])
def test_padding_and_route_order(width):
    ids = np.asarray([5, 5, 5, 3, 7, 5], dtype=np.int64)
    records, _ = build_expert_route_tiles_device(ids, 16, width=width)
    expected = _host(ids, width)
    assert records.tolist() == expected.tolist()
    # Every padding slot is -1 and every real slot points at a route whose
    # id matches the record's expert.
    for expert, *slots in records.tolist():
        for route in slots:
            assert route == -1 or ids[route] == expert


def test_gathered_scales_match_host_table():
    rng = np.random.default_rng(7)
    ids = rng.integers(0, 128, size=64).astype(np.int64)
    table = rng.normal(size=128).astype(np.float32)
    records, scales = build_expert_route_tiles_device(ids, 128, width=4, scale_table=table)
    assert scales is not None
    np.testing.assert_array_equal(scales, table[ids])
    assert records.tolist() == _host(ids, 4).tolist()


def test_empty_routes_return_empty():
    records, scales = build_expert_route_tiles_device(np.empty(0, dtype=np.int64), 8, width=4,
                                                      scale_table=np.ones(8, dtype=np.float32))
    assert records.shape == (0, 5)
    assert scales is not None and scales.shape == (0,)
