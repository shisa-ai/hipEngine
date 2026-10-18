"""CPU checks for the matched-partition DMS cache oracle."""
import pytest
from scripts.dms_prefix_smoke import compose_partition


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
