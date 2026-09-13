"""EOS limits acceptance before state commit, not only output formatting."""
import pytest


@pytest.mark.parametrize('depth', range(1, 8))
def test_eos_at_every_reachable_chain_position(depth):
    from hipengine.speculative.streaming import greedy_chain_eos_limit
    candidates = tuple(range(101, 101 + depth))
    for rejected in range(depth + 1):
        target = list(candidates) + [900]
        target[rejected] = 900
        visible = candidates[:rejected] + (900,)
        for index, eos in enumerate(visible):
            for remaining in range(1, depth + 3):
                assert greedy_chain_eos_limit(candidates, target,
                    remaining_decode=remaining, eos_token_id=eos) == min(remaining, index + 1)
        # Candidate IDs after the first rejected edge are not visible, even if
        # another target row predicts EOS there.
        for eos in candidates[rejected:]:
            assert greedy_chain_eos_limit(candidates, target,
                remaining_decode=depth + 2, eos_token_id=eos) == depth + 2


@pytest.mark.parametrize('remaining', [0, 1, 8])
def test_no_eos_preserves_decode_room(remaining):
    from hipengine.speculative.streaming import greedy_chain_eos_limit
    assert greedy_chain_eos_limit((2, 3), (2, 3, 4), remaining_decode=remaining,
                                  eos_token_id=None) == remaining


@pytest.mark.parametrize('candidates,target,remaining', [((2,), (), 3), ((2,), (2,3), -1)])
def test_invalid_chain_limit_fails_closed(candidates,target,remaining):
    from hipengine.speculative.streaming import greedy_chain_eos_limit
    with pytest.raises(ValueError):
        greedy_chain_eos_limit(candidates,target,remaining_decode=remaining,eos_token_id=3)
