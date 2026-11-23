import numpy as np
import pytest

from hipengine.loading.inkling2_device_experts import build_expert_route_tiles


@pytest.mark.parametrize("count", [1, 16, 24, 31, 32, 33, 64])
@pytest.mark.parametrize("width", [2, 4])
def test_expert_route_tiles_keep_stable_order_and_padding(count, width):
    rng = np.random.default_rng(1130 + count)
    for pool in (1, 3, 16, 96):
        ids = rng.integers(0, pool, size=count, dtype=np.int32)
        expected = []
        for expert in sorted(set(ids.tolist())):
            routes = [i for i, chosen in enumerate(ids) if chosen == expert]
            for start in range(0, len(routes), width):
                tile = routes[start:start + width]
                expected.append([expert, *tile, *([-1] * (width - len(tile)))])
        actual = build_expert_route_tiles(ids, width)
        assert actual.dtype == np.int32
        np.testing.assert_array_equal(actual, expected)
