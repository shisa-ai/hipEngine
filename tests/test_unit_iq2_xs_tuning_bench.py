from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "iq2_xs_tuning_bench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("iq2_xs_tuning_bench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_defaults_and_raw_byte_accounting() -> None:
    module = _load_module()
    args = module.parse_args([])
    assert (args.experts, args.in_features, args.out_features, args.top_k) == (
        256,
        3072,
        1024,
        10,
    )
    assert module.parse_decode_threads(args.decode_threads) == (256,)
    assert args.include_mmq32 is False
    assert module.parse_args(["--include-mmq32"]).include_mmq32 is True
    assert module.parse_decode_threads("64,128,256") == (64, 128, 256)
    with pytest.raises(ValueError, match="decode threads"):
        module.parse_decode_threads("96,256")
    single = module.raw_weight_bytes_per_dispatch(
        rows=10,
        in_features=3072,
        out_features=1024,
        matrices=1,
    )
    assert single == 10 * 1024 * 12 * 74
    assert module.raw_weight_bytes_per_dispatch(
        rows=10,
        in_features=3072,
        out_features=1024,
        matrices=2,
    ) == 2 * single
    assert [module.adaptive_row_batch(count) for count in (0, 1, 2, 3, 4, 7, 8)] == [
        1,
        1,
        2,
        4,
        4,
        4,
        4,
    ]
    counts = np.asarray([0, 1, 2, 3, 4, 7, 8], dtype=np.int64)
    adaptive = module.adaptive_grouped_raw_weight_bytes_per_dispatch(
        counts,
        in_features=3072,
        out_features=1024,
        matrices=2,
    )
    expected_visits = 0 + 1 + 1 + 1 + 1 + 2 + 2
    assert adaptive == module.raw_weight_bytes_per_dispatch(
        rows=expected_visits,
        in_features=3072,
        out_features=1024,
        matrices=2,
    )


@pytest.mark.parametrize("assignments", [1, 160, 320, 1280, 5120])
def test_balanced_counts_are_deterministic_and_conserve_assignments(
    assignments: int,
) -> None:
    module = _load_module()
    counts = module.build_expert_counts(
        num_experts=256,
        assignments=assignments,
        distribution="balanced",
        seed=123,
    )
    assert counts.dtype == np.int64
    assert counts.shape == (256,)
    assert int(counts.sum()) == assignments
    assert int(counts.max() - counts.min()) <= 1
    np.testing.assert_array_equal(
        counts,
        module.build_expert_counts(
            num_experts=256,
            assignments=assignments,
            distribution="balanced",
            seed=999,
        ),
    )


def test_hot_and_zipf_counts_are_skewed_and_conserve_assignments() -> None:
    module = _load_module()
    hot = module.build_expert_counts(
        num_experts=256,
        assignments=5120,
        distribution="hot",
        seed=7,
    )
    zipf = module.build_expert_counts(
        num_experts=256,
        assignments=5120,
        distribution="zipf",
        seed=7,
    )
    assert int(hot.sum()) == 5120
    assert int(hot[:8].sum()) >= 2560
    assert int(zipf.sum()) == 5120
    assert zipf[0] > zipf[1] > zipf[7] > zipf[-1]
    assert np.all(zipf[:-1] >= zipf[1:])


def test_decode_route_sets_cover_cold_hot_repeated_and_boundaries() -> None:
    module = _load_module()
    rotating = module.build_decode_route_sets(
        num_experts=256,
        top_k=10,
        route_sets=32,
        pattern="rotating",
        seed=11,
    )
    hot = module.build_decode_route_sets(
        num_experts=256,
        top_k=10,
        route_sets=4,
        pattern="hot",
        seed=11,
    )
    repeated = module.build_decode_route_sets(
        num_experts=256,
        top_k=10,
        route_sets=4,
        pattern="repeated",
        seed=11,
    )
    assert len(rotating) == 32
    assert all(row.dtype == np.int64 and row.shape == (10,) for row in rotating)
    assert all(len(set(row.tolist())) == 10 for row in rotating)
    assert len({tuple(row.tolist()) for row in rotating}) == 32
    assert 0 in np.concatenate(rotating)
    assert 255 in np.concatenate(rotating)
    assert all(np.array_equal(row, hot[0]) for row in hot)
    assert all(np.all(row == row[0]) for row in repeated)


def test_counterbalanced_order_is_a_rotating_permutation() -> None:
    module = _load_module()
    names = ("base", "rowbatch4", "wmma")
    orders = [module.counterbalanced_order(names, repeat) for repeat in range(6)]
    assert all(sorted(order) == sorted(names) for order in orders)
    assert {order[0] for order in orders} == set(names)
    assert orders[0] == names
    assert orders[1] != orders[0]


def test_invalid_distribution_and_shape_are_rejected() -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="distribution"):
        module.build_expert_counts(
            num_experts=256,
            assignments=10,
            distribution="bad",
            seed=0,
        )
    with pytest.raises(ValueError, match="divisible by 256"):
        module.raw_weight_bytes_per_dispatch(
            rows=10,
            in_features=3000,
            out_features=1024,
            matrices=1,
        )
