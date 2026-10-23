from __future__ import annotations

import dataclasses

import pytest

from hipengine.speculative.prompt_lookup import (
    PromptLookup,
    PromptLookupMatch,
    PromptLookupSnapshot,
)


def _span(first: int, count: int) -> tuple[int, ...]:
    return tuple(range(first, first + count))


def _prompt(
    source_count: int = 40, *, noise_first: int = 5000, noise_count: int = 7
) -> tuple[int, ...]:
    """A unique "file" of tokens followed by unique filler."""

    return (*_span(1000, source_count), *_span(noise_first, noise_count))


def _first_step(
    lookup: PromptLookup, prompt: tuple[int, ...], root: int, max_tokens: int = 8
) -> None:
    """Run the request's first proposal step, which never copies."""

    assert lookup.propose((*prompt, root), max_tokens) == ()
    assert lookup.last_match is None


def _verified_copy(
    lookup: PromptLookup,
    prompt: tuple[int, ...],
    source: tuple[int, ...],
    accepted: int,
) -> tuple[int, ...]:
    """Quote ``source[5:20]``; the copy starts at 20, then record ``accepted``."""

    quoting = (*prompt, *source[5:20])
    assert lookup.propose(quoting, 8) == source[20:28]
    lookup.observe(accepted)
    return quoting


def _resumed_request(
    source: tuple[int, ...], prompt: tuple[int, ...]
) -> tuple[PromptLookup, tuple[int, ...]]:
    """A request whose last copy was accepted and continues at index 24."""

    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])
    quoting = _verified_copy(lookup, prompt, source, accepted=4)
    assert lookup.resume_point == 24
    return lookup, quoting


# --------------------------------------------------------------------------- #
# Gate widths
# --------------------------------------------------------------------------- #


def test_prompt_lookup_first_step_copies_nothing() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()

    # The prompt already contains the passage the answer is about to retype.
    assert lookup.propose((*prompt, source[5]), 8) == ()
    assert lookup.last_match is None
    assert lookup.request_start == len(prompt)
    assert lookup.diagnostics()["calls"] == 1

    # The next step may copy: the boundary no longer hides the match.
    assert lookup.propose((*prompt, *source[5:20]), 8) == source[20:28]


def test_prompt_lookup_greedy_needs_twelve_matched_tokens() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5], 16)

    eleven = (*prompt, source[5], *source[6:16])
    assert lookup.propose(eleven, 16) == ()
    assert lookup.last_match is None

    twelve = (*prompt, source[5], *source[6:17])
    assert lookup.propose(twelve, 16) == source[17:33]
    assert lookup.last_match == PromptLookupMatch(start=17, length=12)


def test_prompt_lookup_sampled_needs_twenty_four_matched_tokens() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    greedy = PromptLookup()
    sampled = PromptLookup(sampled=True)
    _first_step(greedy, prompt, source[5], 16)
    _first_step(sampled, prompt, source[5], 16)

    twenty_three = (*prompt, source[5], *source[6:28])
    assert greedy.propose(twenty_three, 8) == source[28:36]
    assert sampled.propose(twenty_three, 8) == ()
    assert sampled.last_match is None

    twenty_four = (*twenty_three, source[28])
    assert sampled.propose(twenty_four, 8) == source[29:37]
    assert sampled.last_match == PromptLookupMatch(start=29, length=24)


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #


def test_full_width_reservation_declines_short_source_without_counting_a_copy():
    source = _span(1000, 30)
    lookup = PromptLookup(sampled=True)
    lookup.propose((*source, source[0]), 3, require_full=True)
    history = (*source, *source[:29])
    assert lookup.propose(history, 3, require_full=True) == ()
    assert lookup.diagnostics()['hit_cycles'] == 0
    assert lookup.propose(history, 3) == source[29:]


def test_prompt_lookup_prefers_the_longest_backwards_match() -> None:
    block = _span(100, 14)
    prompt = (*block, *_span(5000, 5), 999, *block[11:14])
    lookup = PromptLookup()
    _first_step(lookup, prompt, 888)

    # The newest occurrence of (111, 112, 113) has a different history, so the
    # older occurrence that matches 14 tokens back wins.
    assert lookup.propose((*prompt, 888, *block), 8) == prompt[14:22]
    assert lookup.last_match == PromptLookupMatch(start=14, length=14)


def test_prompt_lookup_breaks_equal_length_ties_with_the_newest_occurrence() -> None:
    block = _span(100, 14)
    prompt = (*block, *_span(5000, 5), 999, *block)
    lookup = PromptLookup()
    _first_step(lookup, prompt, 888)

    proposal = lookup.propose((*prompt, 888, *block), 8)

    # Both occurrences match 14 tokens back; the newest one's continuation wins.
    assert lookup.last_match == PromptLookupMatch(start=34, length=14)
    assert proposal == (888, *block[:7])


def _repeated_key_history(
    key: tuple[int, ...], *, deep: tuple[int, ...], short: int
) -> tuple[int, ...]:
    history = (*deep, *key)
    for index in range(short):
        history = (*history, 3000 + index, *key)
    return history


def test_prompt_lookup_examines_only_the_newest_sixteen_occurrences() -> None:
    key = (777, 778, 779)
    deep = _span(2000, 14)

    # The one long-matching occurrence is the twentieth newest.
    twenty = _repeated_key_history(key, deep=deep, short=19)
    lookup = PromptLookup()
    _first_step(lookup, twenty, 4000)
    assert lookup.propose((*twenty, 4000, *deep, *key), 8) == ()
    assert lookup.last_match is None

    # As the sixteenth newest it is examined and wins.
    sixteen = _repeated_key_history(key, deep=deep, short=15)
    lookup = PromptLookup()
    _first_step(lookup, sixteen, 4000)
    assert lookup.propose((*sixteen, 4000, *deep, *key), 8) == sixteen[17:25]
    assert lookup.last_match == PromptLookupMatch(start=17, length=17)


def test_prompt_lookup_caps_the_match_at_sixty_four_tokens() -> None:
    source = _span(1000, 200)
    prompt = (*source, *_span(5000, 7))
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[100])

    proposal = lookup.propose((*prompt, source[100], *source[101:180]), 8)

    assert lookup.last_match == PromptLookupMatch(start=180, length=64)
    assert proposal == source[180:188]


# --------------------------------------------------------------------------- #
# Copy bounds
# --------------------------------------------------------------------------- #


def test_prompt_lookup_bounds_the_copy_by_budget_and_source_end() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    sampled = PromptLookup(sampled=True)
    greedy = PromptLookup()
    _first_step(sampled, prompt, source[5], 64)
    _first_step(greedy, prompt, source[5], 4)

    history = (*prompt, source[5], *source[6:29])

    # A sampled request copies only from its starting context: the source ends
    # at the prompt's last token, before the root.
    assert sampled.indexed_tokens == len(prompt)
    assert sampled.propose(history, 64) == (*source[29:40], *_span(5000, 7))
    assert len(sampled.propose(history, 64)) == sampled.indexed_tokens - 29

    # A greedy request indexes everything it committed, so its source is longer.
    assert greedy.propose(history, 4) == source[29:33]
    assert greedy.indexed_tokens == len(history)


def test_prompt_lookup_sampled_indexes_only_its_starting_context() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    passage = _span(9000, 40)
    sampled = PromptLookup(sampled=True)
    greedy = PromptLookup()
    _first_step(sampled, prompt, source[5], 16)
    _first_step(greedy, prompt, source[5], 16)

    # A new answer generates the same passage twice.
    answer = (*passage, 90_000, *passage[:23])
    history = (*prompt, source[5], *answer)

    assert sampled.propose(history, 8) == ()
    assert sampled.last_match is None
    assert greedy.propose(history, 8) == passage[23:31]
    assert greedy.last_match == PromptLookupMatch(start=71, length=23)

    # The starting context still supplies copies, bounded by its last token.
    quoted = (*history, *source[14:38])
    proposal = sampled.propose(quoted, 64)
    assert proposal == (*source[38:40], *_span(5000, 7))
    assert len(proposal) == sampled.indexed_tokens - 38


def test_prompt_lookup_new_request_copies_the_previous_answer() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    passage = _span(9000, 40)
    previous = (*prompt, source[5], *passage, 90_000, *passage[:23], *source[14:38])

    # A new request is a new instance; its starting context is the old history.
    next_request = PromptLookup(sampled=True)
    _first_step(next_request, previous, source[0])

    proposal = next_request.propose((*previous, source[0], *passage[:24]), 8)

    assert next_request.indexed_tokens == len(previous)
    assert proposal == passage[24:32]
    assert next_request.last_match == PromptLookupMatch(start=72, length=24)


def test_prompt_lookup_match_is_bounded_by_the_first_proposal_boundary() -> None:
    source = _span(1000, 80)
    # The prompt ends with twenty tokens of shared framing.
    prompt = (*source, *_span(5000, 7), *source[20:40])
    sampled = PromptLookup(sampled=True)
    _first_step(sampled, prompt, source[40])

    # Those framing tokens match the answer's own start, so only the boundary
    # keeps a four-token answer from counting as a full 24-token match.
    assert sampled.propose((*prompt, *source[40:44]), 8) == ()
    assert sampled.last_match is None

    assert sampled.propose((*prompt, *source[40:64]), 8) == source[64:72]
    assert sampled.last_match == PromptLookupMatch(start=64, length=24)


# --------------------------------------------------------------------------- #
# History shapes
# --------------------------------------------------------------------------- #


def test_prompt_lookup_proposes_nothing_without_a_repeated_key() -> None:
    prompt = _span(10_000, 200)
    lookup = PromptLookup()
    _first_step(lookup, prompt, 20_000)

    assert lookup.propose((*prompt, 20_000, 20_001, 20_002, 20_003), 8) == ()
    assert lookup.last_match is None
    assert lookup.diagnostics()["hit_cycles"] == 0

    # Held-out ids that appear once never become a proposal.
    proposal = lookup.propose((*prompt, 20_000, 20_001, 20_002, 20_003, *prompt[:20]), 8)
    assert proposal == prompt[20:28]
    assert not set(proposal) & {20_000, 20_001, 20_002, 20_003}


def test_prompt_lookup_copies_repetitive_code_like_history() -> None:
    line_a = (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    line_b = (30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43)
    prompt = (*line_a, *_span(60, 3), *line_b, *_span(70, 3))
    lookup = PromptLookup()
    _first_step(lookup, prompt, line_a[0], 6)

    # The answer retypes line_a; the copy continues with what followed it.
    proposal = lookup.propose((*prompt, *line_a), 6)

    assert proposal == (*_span(60, 3), *line_b[:3])
    assert lookup.last_match == PromptLookupMatch(start=14, length=14)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_prompt_lookup_needs_no_copy_to_resume() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])

    # The answer copied source[5:20], edited two tokens, then rejoined at 22.
    rejoin = (*prompt, *source[5:20], 7001, 7002, *source[22:27])
    assert lookup.propose(rejoin, 8) == ()
    assert lookup.last_match is None
    assert lookup.resume_point is None


def test_prompt_lookup_resumes_a_copy_at_five_matched_tokens() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])
    quoting = _verified_copy(lookup, prompt, source, accepted=4)

    rejoin = (*quoting, 7001, 7002, *source[22:27])
    assert lookup.propose(rejoin, 8) == source[27:35]
    assert lookup.last_match == PromptLookupMatch(start=27, length=5)


def test_prompt_lookup_needs_half_of_a_copy_to_keep_resuming() -> None:
    source = _span(1000, 40)
    prompt = _prompt()

    for accepted in (0, 3):
        lookup = PromptLookup()
        _first_step(lookup, prompt, source[5])
        quoting = _verified_copy(lookup, prompt, source, accepted=accepted)
        assert lookup.resume_point is None
        rejoin = (*quoting, 7001, 7002, *source[22:27])
        assert lookup.propose(rejoin, 8) == ()

    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])
    quoting = _verified_copy(lookup, prompt, source, accepted=4)
    rejoin = (*quoting, 7001, 7002, *source[22:27])

    # A resumed copy that keeps half its tokens still resumes.
    assert lookup.propose(rejoin, 8) == source[27:35]
    lookup.observe(4)
    assert lookup.resume_point == 31
    assert lookup.propose(rejoin, 8) == source[27:35]

    # One that keeps fewer has left the text until a full match copies again.
    lookup.observe(3)
    assert lookup.resume_point is None
    assert lookup.propose(rejoin, 8) == ()


def test_prompt_lookup_resume_window_is_minus_eight_plus_512() -> None:
    source = _span(1000, 600)
    prompt = (*source, *_span(5000, 7))

    def _propose_rejoin(end: int) -> tuple[int, ...]:
        lookup, quoting = _resumed_request(source, prompt)
        return lookup.propose((*quoting, 7001, 7002, *source[end - 5 : end]), 8)

    # The resume point is 24: eight tokens before it, 512 after it.
    assert _propose_rejoin(24) == source[24:32]
    assert _propose_rejoin(16) == source[16:24]
    assert _propose_rejoin(15) == ()
    assert _propose_rejoin(536) == source[536:544]
    assert _propose_rejoin(537) == ()


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #


def test_prompt_lookup_observe_counts_once_and_validates() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])

    quoting = _verified_copy(lookup, prompt, source, accepted=5)
    assert lookup.diagnostics()["proposed_tokens"] == 8
    assert lookup.diagnostics()["accepted_tokens"] == 5
    assert lookup.resume_point == 25

    # The pending copy is spent: a second observation cannot count again.
    with pytest.raises(ValueError):
        lookup.observe(5)
    assert lookup.diagnostics()["accepted_tokens"] == 5

    # A cycle that proposed nothing clears pending but keeps the resume point.
    assert lookup.propose((*quoting, 7001, 7002), 8) == ()
    assert lookup.resume_point == 25
    lookup.observe(0)
    assert lookup.diagnostics()["accepted_tokens"] == 5

    with pytest.raises(ValueError):
        lookup.observe(1)
    with pytest.raises(ValueError):
        lookup.observe(-1)


def test_prompt_lookup_observe_rejects_more_than_was_proposed() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])
    quoting = (*prompt, *source[5:20])
    assert lookup.propose(quoting, 8) == source[20:28]

    with pytest.raises(ValueError):
        lookup.observe(9)

    # The rejected call did not consume the pending proposal.
    lookup.observe(8)
    assert lookup.resume_point == 28


def test_prompt_lookup_diagnostics_counts_only_nonempty_proposals() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup(sampled=True)
    _first_step(lookup, prompt, source[5])

    diagnostics = lookup.diagnostics()
    assert isinstance(diagnostics, dict)
    assert diagnostics == {
        "calls": 1,
        "hit_cycles": 0,
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "sampled": True,
    }

    quoting = (*prompt, *source[5:29])
    proposal = lookup.propose(quoting, 8)
    assert proposal == source[29:37]
    diagnostics = lookup.diagnostics()
    assert diagnostics["calls"] == 2
    assert diagnostics["hit_cycles"] == 1
    assert diagnostics["proposed_tokens"] == len(proposal)
    lookup.observe(len(proposal))
    assert lookup.diagnostics()["accepted_tokens"] == len(proposal)

    # A cycle with no proposal is a call, not a hit.
    assert lookup.propose((*quoting, 7001, 7002), 8) == ()
    diagnostics = lookup.diagnostics()
    assert diagnostics["calls"] == 3
    assert diagnostics["hit_cycles"] == 1
    assert diagnostics["proposed_tokens"] == len(proposal)


# --------------------------------------------------------------------------- #
# Ownership, validation, and rollback
# --------------------------------------------------------------------------- #


def test_prompt_lookup_rejects_shortened_or_rewritten_history() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])
    quoting = (*prompt, *source[5:20])
    assert lookup.propose(quoting, 8) == source[20:28]

    with pytest.raises(ValueError):
        lookup.propose((*prompt, *source[5:10]), 8)
    with pytest.raises(ValueError):
        lookup.propose((*quoting[:-1], 7001), 8)
    with pytest.raises(ValueError):
        lookup.propose((7001, *quoting[1:]), 8)
    with pytest.raises(ValueError):
        lookup.propose((*quoting[:10], 7001, *quoting[11:], 7002), 8)

    # The rejected calls left the request intact.
    assert lookup.propose(quoting, 8) == source[20:28]


def test_prompt_lookup_state_is_request_local() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    other = _span(9000, 40)
    first = PromptLookup()
    second = PromptLookup()
    _first_step(first, prompt, source[5])
    _first_step(second, other, other[5])

    assert first.propose((*prompt, *source[5:20]), 8) == source[20:28]
    # The second request's prompt never contained the first request's tokens.
    assert second.propose((*other, other[5], *source[6:20]), 8) == ()
    assert first.diagnostics()["hit_cycles"] == 1
    assert second.diagnostics()["hit_cycles"] == 0

    # Two instances that saw the same history propose the same tokens.
    twin = PromptLookup()
    _first_step(twin, prompt, source[5])
    assert twin.propose((*prompt, *source[5:20]), 8) == source[20:28]


def test_prompt_lookup_validates_tokens_and_budget() -> None:
    lookup = PromptLookup()

    with pytest.raises(ValueError):
        lookup.propose((), 8)
    with pytest.raises(ValueError):
        lookup.propose((1, 2, 3), 0)
    with pytest.raises(ValueError):
        lookup.propose((1, 2, 3), -2)
    with pytest.raises(TypeError):
        lookup.propose((1, 2, 3), 8.0)
    with pytest.raises(TypeError):
        lookup.propose((1, 2, True), 8)
    with pytest.raises(TypeError):
        lookup.propose((1, 2, "3"), 8)
    with pytest.raises(ValueError):
        lookup.propose((1, 2, -3), 8)
    with pytest.raises(ValueError):
        lookup.propose((1, 2, 2**31), 8)
    with pytest.raises(TypeError):
        lookup.propose(5, 8)
    with pytest.raises(TypeError):
        PromptLookup(sampled="yes")

    # Nothing above started the request.
    assert lookup.request_start is None
    assert lookup.diagnostics()["calls"] == 0


def test_prompt_lookup_snapshot_replays_a_cycle() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])
    quoting = (*prompt, *source[5:20])

    before = lookup.snapshot()
    before_diagnostics = lookup.diagnostics()
    assert isinstance(before, PromptLookupSnapshot)

    proposal = lookup.propose(quoting, 8)
    assert proposal == source[20:28]
    after_propose = lookup.diagnostics()
    assert after_propose["proposed_tokens"] == 8
    lookup.observe(8)
    assert lookup.resume_point == 28
    assert lookup.diagnostics()["accepted_tokens"] == 8

    lookup.restore(before)
    assert lookup.diagnostics() == before_diagnostics
    assert lookup.resume_point is None
    assert lookup.last_match is None

    replay = lookup.propose(quoting, 8)
    assert replay == proposal
    assert lookup.diagnostics() == after_propose
    assert lookup.last_match == PromptLookupMatch(start=20, length=15)
    # The index kept its derived entries; the replay added nothing new.
    assert lookup.indexed_tokens == len(quoting)


def test_prompt_lookup_snapshot_restores_the_first_step() -> None:
    source = _span(1000, 40)
    prompt = _prompt()
    lookup = PromptLookup()
    fresh = lookup.snapshot()
    assert fresh.request_start is None

    first = (*prompt, source[5])
    assert lookup.propose(first, 8) == ()
    assert lookup.request_start == len(prompt)

    lookup.restore(fresh)
    assert lookup.request_start is None
    assert lookup.diagnostics() == {
        "calls": 0,
        "hit_cycles": 0,
        "proposed_tokens": 0,
        "accepted_tokens": 0,
        "sampled": False,
    }

    # The aborted first step repeats: the boundary is captured again.
    assert lookup.propose(first, 8) == ()
    assert lookup.last_match is None
    assert lookup.request_start == len(prompt)
    assert lookup.propose((*first, *source[6:20]), 8) == source[20:28]

    # A restored first step cannot index a different history.
    lookup.restore(fresh)
    with pytest.raises(ValueError):
        lookup.propose((*prompt, 7001), 8)


def test_prompt_lookup_snapshot_carries_no_token_history() -> None:
    source = _span(1000, 20_000)
    prompt = (*source, *_span(50_000, 7))
    lookup = PromptLookup()
    _first_step(lookup, prompt, source[5])

    state = lookup.snapshot()
    assert isinstance(state, PromptLookupSnapshot)
    assert {field.name for field in dataclasses.fields(state)} == {
        "request_start",
        "resume",
        "pending",
        "match",
        "calls",
        "hit_cycles",
        "proposed_tokens",
        "accepted_tokens",
        "sampled",
    }
    for field in dataclasses.fields(state):
        value = getattr(state, field.name)
        if isinstance(value, tuple):
            assert len(value) <= 2
        else:
            assert value is None or isinstance(value, (bool, int, PromptLookupMatch))
    # The index still covers the whole committed history.
    assert lookup.indexed_tokens == len(prompt) + 1


def test_prompt_lookup_rejects_a_foreign_snapshot() -> None:
    greedy = PromptLookup()
    sampled = PromptLookup(sampled=True)

    with pytest.raises(ValueError):
        sampled.restore(greedy.snapshot())
    with pytest.raises(TypeError):
        greedy.restore(object())  # type: ignore[arg-type]
