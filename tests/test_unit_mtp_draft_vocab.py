"""Unit contract for the request-local MTP draft-vocabulary policy.

The policy owns the per-request draft-head row table (prefix, special tokens,
and the request's own out-of-prefix tokens) and the decision whether a cycle
may score the small head. It reads only the request's own committed tokens, so
these cases pin table order, capacity behaviour, the recent-miss window, and
the append-only catch-up contract.
"""

from __future__ import annotations

import pytest

from hipengine.speculative.draft_vocab import (
    DEFAULT_EXTRA_CAPACITY,
    DEFAULT_PREFIX_SIZE,
    DEFAULT_WINDOW,
    MTPDraftVocabulary,
)

VOCAB = 4096
PREFIX = 100


def _policy(
    *,
    vocab: int = VOCAB,
    prefix: int = PREFIX,
    capacity: int = 8,
    specials: tuple[int, ...] = (),
    window: int = DEFAULT_WINDOW,
) -> MTPDraftVocabulary:
    return MTPDraftVocabulary(
        vocab,
        prefix_size=prefix,
        extra_capacity=capacity,
        special_ids=specials,
        window=window,
    )


def _history(length: int, misses: tuple[int, ...], *, prefix: int = PREFIX) -> list[int]:
    """Prefix-token history with distinct out-of-prefix tokens at ``misses``."""
    tokens = [0] * length
    for offset, position in enumerate(misses):
        tokens[position] = prefix + 1 + offset
    return tokens


def test_default_geometry_matches_the_draft_head_contract() -> None:
    policy = MTPDraftVocabulary(248_320)
    assert policy.vocab_size == 248_320
    assert policy.prefix_size == DEFAULT_PREFIX_SIZE == 98_304
    assert policy.extra_capacity == DEFAULT_EXTRA_CAPACITY == 1024
    assert policy.window == DEFAULT_WINDOW == 32
    assert policy.scanned == 0
    assert policy.extra_ids == ()


@pytest.mark.parametrize(
    ("vocab", "prefix", "expected"),
    [
        (1000, DEFAULT_PREFIX_SIZE, 1000),
        (1000, 500, 500),
        (1000, 1000, 1000),
        (1000, 0, 0),
        (1000, 1, 1),
    ],
)
def test_prefix_is_clipped_to_the_vocabulary_size(vocab: int, prefix: int, expected: int) -> None:
    policy = _policy(vocab=vocab, prefix=prefix, capacity=4)
    assert policy.prefix_size == expected


def test_prefix_rows_need_no_extra_slots() -> None:
    policy = _policy(vocab=1000, prefix=1000, capacity=0)
    assert policy.extra_ids == ()
    assert policy.fits(0)
    assert policy.fits(999)
    assert policy.catch_up([0, 999, 500]) == ()
    assert policy.scanned == 3
    assert policy.fits(500)


def test_in_prefix_specials_do_not_consume_extra_rows() -> None:
    policy = _policy(capacity=1, specials=(5, 7))
    assert policy.special_ids == (5, 7)
    assert policy.extra_ids == ()
    assert policy.fits(5)
    assert policy.fits(7)
    # A special already inside the prefix is never tabled as a rare token.
    assert policy.catch_up([5, 7]) == ()
    assert policy.extra_ids == ()


def test_out_of_prefix_specials_lead_the_extra_table() -> None:
    policy = _policy(capacity=4, specials=(PREFIX + 3, PREFIX + 1))
    assert policy.extra_ids == (PREFIX + 3, PREFIX + 1)
    assert policy.fits(PREFIX + 1)
    assert policy.fits(PREFIX + 3)
    assert policy.catch_up([PREFIX + 9]) == (PREFIX + 9,)
    assert policy.extra_ids == (PREFIX + 3, PREFIX + 1, PREFIX + 9)


def test_duplicate_specials_collapse_in_declared_order() -> None:
    policy = _policy(capacity=4, specials=(PREFIX + 2, PREFIX + 2, PREFIX + 1, PREFIX + 2))
    assert policy.special_ids == (PREFIX + 2, PREFIX + 1)
    assert policy.extra_ids == (PREFIX + 2, PREFIX + 1)


def test_rare_ids_are_tabled_in_first_use_order() -> None:
    policy = _policy()
    added = policy.catch_up([PREFIX + 7, 3, PREFIX + 4, PREFIX + 7, PREFIX + 2])
    assert added == (PREFIX + 7, PREFIX + 4, PREFIX + 2)
    assert policy.extra_ids == (PREFIX + 7, PREFIX + 4, PREFIX + 2)
    assert policy.scanned == 5


def test_duplicate_committed_ids_are_tabled_once() -> None:
    policy = _policy()
    assert policy.catch_up([PREFIX + 5, PREFIX + 5, PREFIX + 5]) == (PREFIX + 5,)
    assert policy.catch_up([PREFIX + 5, PREFIX + 5, PREFIX + 5, PREFIX + 5]) == ()
    assert policy.extra_ids == (PREFIX + 5,)
    assert policy.scanned == 4


def test_specials_are_never_tabled_as_rare_tokens() -> None:
    policy = _policy(capacity=4, specials=(PREFIX + 6,))
    assert policy.catch_up([PREFIX + 6, PREFIX + 6]) == ()
    assert policy.extra_ids == (PREFIX + 6,)
    # A declared special is a represented anchor even before it is committed.
    assert policy.fits(PREFIX + 6)


def test_special_table_overflow_is_rejected_before_construction() -> None:
    with pytest.raises(ValueError, match="extra_capacity"):
        _policy(capacity=2, specials=(PREFIX + 1, PREFIX + 2, PREFIX + 3))
    # Exactly at capacity is legal and leaves no room for rare tokens.
    policy = _policy(capacity=2, specials=(PREFIX + 1, PREFIX + 2))
    assert policy.extra_ids == (PREFIX + 1, PREFIX + 2)
    assert policy.catch_up([PREFIX + 3]) == ()
    assert policy.extra_ids == (PREFIX + 1, PREFIX + 2)


def test_extra_capacity_overflow_keeps_unknown_tokens_as_misses() -> None:
    policy = _policy(capacity=1, window=32)
    assert policy.catch_up([PREFIX + 1, PREFIX + 2]) == (PREFIX + 1,)
    assert policy.extra_ids == (PREFIX + 1,)
    # The tabled token is a usable anchor; the untabled one stays a miss.
    assert not policy.fits(PREFIX + 2)
    assert not policy.fits(0)  # two misses inside the window
    # Repeating the untabled token is still a miss, not a silent admission.
    assert policy.catch_up([PREFIX + 1, PREFIX + 2, PREFIX + 2]) == ()
    assert policy.extra_ids == (PREFIX + 1,)


def test_zero_extra_capacity_tables_nothing() -> None:
    policy = _policy(capacity=0)
    assert policy.catch_up([PREFIX + 1, PREFIX + 2]) == ()
    assert policy.extra_ids == ()
    assert not policy.fits(PREFIX + 1)
    assert policy.scanned == 2


def test_fits_requires_a_represented_anchor() -> None:
    policy = _policy(capacity=4)
    assert policy.fits(0)
    assert policy.fits(PREFIX - 1)
    assert not policy.fits(PREFIX)
    assert policy.catch_up([PREFIX])
    assert policy.fits(PREFIX)
    assert not policy.fits(PREFIX + 1)


@pytest.mark.parametrize("anchor", [-1, VOCAB, VOCAB + 1000])
def test_out_of_vocabulary_anchor_is_a_miss(anchor: int) -> None:
    policy = _policy()
    assert not policy.fits(anchor)


@pytest.mark.parametrize("window", [2, 4, 32])
@pytest.mark.parametrize("offset", [0, 1, 2])
def test_one_recent_miss_still_allows_the_small_head(window: int, offset: int) -> None:
    length = window + 2
    position = length - 1 - offset
    policy = _policy(window=window)
    assert policy.catch_up(_history(length, (position,))) == (PREFIX + 1,)
    assert policy.fits(0)


@pytest.mark.parametrize("window", [2, 4, 32])
def test_two_recent_misses_leave_the_small_head(window: int) -> None:
    length = window + 2
    policy = _policy(window=window)
    policy.catch_up(_history(length, (length - 2, length - 1)))
    assert not policy.fits(0)


@pytest.mark.parametrize(
    ("window", "length"),
    [(2, 5), (2, 12), (4, 9), (4, 33), (32, 33), (32, 40), (32, 97)],
)
def test_the_miss_window_holds_exactly_at_its_boundary(window: int, length: int) -> None:
    quiet_position = length - window - 1
    loud_position = length - window
    assert 0 <= quiet_position < loud_position < length - 1

    quiet = _policy(window=window)
    quiet.catch_up(_history(length, (quiet_position, length - 1)))
    assert quiet.fits(0)

    loud = _policy(window=window)
    loud.catch_up(_history(length, (loud_position, length - 1)))
    assert not loud.fits(0)


@pytest.mark.parametrize("length", [3, 7, 31, 32, 33, 64, 129])
def test_a_lone_miss_anywhere_in_the_history_stays_quiet(length: int) -> None:
    for position in (0, length // 2, length - 1):
        policy = _policy()
        policy.catch_up(_history(length, (position,)))
        assert policy.fits(0), (length, position)


def test_the_window_slides_as_history_grows() -> None:
    policy = _policy(window=4)
    policy.catch_up(_history(4, (0, 1)))
    assert not policy.fits(0)
    policy.catch_up(_history(5, (0, 1)))
    assert policy.fits(0)


def test_catch_up_is_append_only_and_reports_only_new_ids() -> None:
    policy = _policy()
    assert policy.catch_up([]) == ()
    assert policy.scanned == 0
    assert policy.catch_up([1, 2]) == ()
    assert policy.scanned == 2
    assert policy.catch_up([1, 2]) == ()
    assert policy.catch_up([1, 2, PREFIX + 4]) == (PREFIX + 4,)
    assert policy.scanned == 3


def test_chunked_catch_up_matches_one_shot() -> None:
    tokens = [1, PREFIX + 3, 2, PREFIX + 3, PREFIX + 9, 3, PREFIX + 1, PREFIX + 9, PREFIX + 4]
    anchors = [0, 1, PREFIX - 1, PREFIX + 1, PREFIX + 3, PREFIX + 4, PREFIX + 9]

    chunked = _policy(capacity=8)
    chunked_added: list[int] = []
    for end in (1, 3, 4, 6, 7, 9):
        chunked_added.extend(chunked.catch_up(tokens[:end]))
        # The chunked policy must equal a fresh policy handed the same history
        # in one call: catch-up carries no state beyond the committed prefix.
        reference = _policy(capacity=8)
        reference_added = reference.catch_up(tokens[:end])
        assert chunked.scanned == reference.scanned == end
        assert tuple(chunked_added) == reference_added
        assert chunked.extra_ids == reference.extra_ids
        assert [chunked.fits(anchor) for anchor in anchors] == [
            reference.fits(anchor) for anchor in anchors
        ]

    one_shot = _policy(capacity=8)
    one_shot_added = one_shot.catch_up(tokens)
    assert tuple(chunked_added) == one_shot_added
    assert chunked.extra_ids == one_shot.extra_ids
    assert chunked.scanned == one_shot.scanned == len(tokens)
    assert [chunked.fits(anchor) for anchor in anchors] == [
        one_shot.fits(anchor) for anchor in anchors
    ]


def test_shortened_history_is_rejected_without_mutation() -> None:
    policy = _policy()
    assert policy.catch_up([1, PREFIX + 2, 3]) == (PREFIX + 2,)
    with pytest.raises(ValueError, match="reset"):
        policy.catch_up([1, PREFIX + 2])
    assert policy.scanned == 3
    assert policy.extra_ids == (PREFIX + 2,)
    assert policy.catch_up([1, PREFIX + 2, 3]) == ()
    assert policy.catch_up([1, PREFIX + 2, 3, PREFIX + 5]) == (PREFIX + 5,)


def test_invalid_committed_ids_are_rejected_without_partial_mutation() -> None:
    policy = _policy()
    assert policy.catch_up([1, PREFIX + 2]) == (PREFIX + 2,)

    with pytest.raises(ValueError):
        policy.catch_up([1, PREFIX + 2, PREFIX + 5, VOCAB])
    with pytest.raises(ValueError):
        policy.catch_up([1, PREFIX + 2, -1])
    with pytest.raises(TypeError):
        policy.catch_up([1, PREFIX + 2, 1.5])

    assert policy.scanned == 2
    assert policy.extra_ids == (PREFIX + 2,)
    # The rejected tail is not consumed: the next valid call tables it.
    assert policy.catch_up([1, PREFIX + 2, PREFIX + 5]) == (PREFIX + 5,)
    assert policy.scanned == 3


def test_reset_restores_specials_only() -> None:
    policy = _policy(capacity=4, specials=(PREFIX + 1,))
    assert policy.catch_up([PREFIX + 1, PREFIX + 2, PREFIX + 3]) == (PREFIX + 2, PREFIX + 3)
    assert policy.extra_ids == (PREFIX + 1, PREFIX + 2, PREFIX + 3)
    assert policy.scanned == 3

    policy.reset()

    assert policy.extra_ids == (PREFIX + 1,)
    assert policy.scanned == 0
    assert policy.fits(PREFIX + 1)
    assert not policy.fits(PREFIX + 2)
    # The reset history is empty, so the same tokens table again in order.
    assert policy.catch_up([PREFIX + 2, PREFIX + 3]) == (PREFIX + 2, PREFIX + 3)
    assert policy.extra_ids == (PREFIX + 1, PREFIX + 2, PREFIX + 3)


def test_reset_clears_the_miss_window() -> None:
    policy = _policy(window=32)
    policy.catch_up(_history(33, (32,)))
    assert policy.fits(0)
    policy.catch_up(_history(34, (32, 33)))
    assert not policy.fits(0)

    policy.reset()

    assert policy.scanned == 0
    assert policy.fits(0)
    assert policy.catch_up([PREFIX + 4]) == (PREFIX + 4,)
    assert policy.fits(0)


def test_request_instances_are_isolated() -> None:
    shared = [1, PREFIX + 2, 3]
    first = _policy(capacity=4)
    second = _policy(capacity=4)
    third = _policy(capacity=4)

    assert first.catch_up(shared) == (PREFIX + 2,)
    assert second.scanned == 0
    assert second.extra_ids == ()
    assert not second.fits(PREFIX + 2)

    assert third.catch_up(shared) == (PREFIX + 2,)
    assert third.extra_ids == first.extra_ids
    assert third.scanned == first.scanned
    assert third.fits(PREFIX + 2) == first.fits(PREFIX + 2)

    first.reset()
    assert second.scanned == 0
    assert third.scanned == 3
    assert third.extra_ids == (PREFIX + 2,)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"vocab_size": 0}, ValueError),
        ({"vocab_size": -1}, ValueError),
        ({"vocab_size": "10"}, TypeError),
        ({"vocab_size": 10.0}, TypeError),
        ({"vocab_size": 10, "prefix_size": -1}, ValueError),
        ({"vocab_size": 10, "prefix_size": 1.5}, TypeError),
        ({"vocab_size": 10, "extra_capacity": -1}, ValueError),
        ({"vocab_size": 10, "extra_capacity": 2.5}, TypeError),
        ({"vocab_size": 10, "window": 0}, ValueError),
        ({"vocab_size": 10, "window": -4}, ValueError),
        ({"vocab_size": 10, "special_ids": (10,)}, ValueError),
        ({"vocab_size": 10, "special_ids": (-1,)}, ValueError),
        ({"vocab_size": 10, "special_ids": (1.5,)}, TypeError),
        (
            {"vocab_size": 10, "prefix_size": 4, "extra_capacity": 1, "special_ids": (5, 6)},
            ValueError,
        ),
    ],
)
def test_constructor_rejects_invalid_geometry(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        MTPDraftVocabulary(**kwargs)
