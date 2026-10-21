"""CPU checks for the matched-partition DMS cache oracle."""
import pytest
from scripts.dms_prefix_smoke import compose_partition, validate_reuse_count


def test_partition_restores_only_the_matched_prefix():
    assert compose_partition(((0, 127), (127, 128), (128, 1152), (1152, 1200)),
                             128, ((0, 1002),), 1130) == ((0, 127), (127, 128), (128, 1130))


def test_exact_hit_reuses_the_full_partition():
    assert compose_partition(((0, 127), (127, 128)), 128, (), 128) == ((0, 127), (127, 128))


@pytest.mark.parametrize('prefix,reused,suffix,total', [
    (((0, 127), (0, 127)), 127, (), 127),
    (((0, 127),), 128, ((0, 10),), 138),
    ((), 0, ((0, 4), (5, 8)), 8),
    ((), 0, ((0, 4),), 5),
])
def test_invalid_partition_fails_closed(prefix, reused, suffix, total):
    with pytest.raises(ValueError):
        compose_partition(prefix, reused, suffix, total)


@pytest.mark.parametrize('actual', [0, 8194, 10003])
def test_return_to_source_allows_budget_eviction(actual):
    assert validate_reuse_count(
        cache='radix', scenario='return-source', actual=actual, expected=10003,
        lengths=[8191, 8194, 10003], branch_at=8500,
    ) == actual


@pytest.mark.parametrize('cache,scenario,actual,expected', [
    ('radix', 'return-source', -1, 10003),
    ('radix', 'return-source', 10004, 10003),
    ('radix', 'repeat-10003', 8194, 10003),
    ('radix', 'extend-10003', 0, 8194),
    ('off', 'return-source', 8194, 10003),
    ('matched', 'return-source', 8194, 10003),
    ('radix', 'branch', 10003, 0),
])
def test_budget_eviction_does_not_relax_other_reuse_contracts(cache, scenario, actual, expected):
    with pytest.raises(AssertionError):
        validate_reuse_count(
            cache=cache, scenario=scenario, actual=actual, expected=expected,
            lengths=[8191, 8194, 10003], branch_at=8500,
        )


@pytest.mark.parametrize('actual', [256, 8191, 8194])
def test_branch_still_accepts_only_its_recorded_boundary_candidates(actual):
    assert validate_reuse_count(
        cache='radix', scenario='branch', actual=actual, expected=0,
        lengths=[8191, 8194, 10003], branch_at=8500,
    ) == actual
