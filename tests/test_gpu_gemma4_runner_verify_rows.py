"""Multi-row logits and rewind on the synthetic fixture artifact.

A speculative verify pass reads the target distribution at every drafted
position and gives back rejected tail tokens. These tests require exact
single-row versus multi-row logits, trailing-row selection, and rewind
recomputation on this fixture. The default exact expert-down owner now satisfies
these assertions; this fixture does not certify other artifacts or profiles.
"""

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(
    not hip_runtime_available(), reason="HIP runtime unavailable"
)

# A five-token block: the four-token draft the drafter proposes plus the token
# it was seeded from, which is the widest verify the default max_drafts asks for.
_TOKENS = [3, 17, 42, 8, 11]


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights
    from tests._gemma4_gguf_fixture import write_default_gemma4_gguf

    path = write_default_gemma4_gguf(tmp_path_factory.mktemp("verify") / "fixture.gguf")
    reader = GGUFReader(path)
    weights = load_gemma4_device_weights(reader)
    # max_logits_rows is what a speculative generator would pass: the draft
    # width plus the bonus row. The default of one is what keeps a 512-wide
    # prefill from sizing its logits scratch for the whole block.
    runner = Gemma4Runner(weights=weights, capacity=64, max_logits_rows=5)
    try:
        yield runner
    finally:
        runner.close()
        weights.free()


def test_verification_phase_keeps_decode_attention_without_changing_prefill(runner, monkeypatch):
    from hipengine.runtime import gemma4 as module
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        PREFILL_ATTENTION_PLAIN, PREFILL_ATTENTION_WMMA_FLASH,
    )

    requested = (PREFILL_ATTENTION_WMMA_FLASH,)
    requests = []
    original = module.gemma4_layer_forward_bf16

    def record(*args, **kwargs):
        requests.append(kwargs["prefill_attention_variants"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "gemma4_layer_forward_bf16", record)
    previous = runner.prefill_attention_variants
    runner.prefill_attention_variants = requested
    try:
        for verification in (False, True, False):
            runner.reset()
            requests.clear()
            runner.forward(_TOKENS, logits_rows=5, verification=verification)
            # None requests automatic attention and can select a tiled route;
            # verification must explicitly request the decode's strict owner.
            expected = (PREFILL_ATTENTION_PLAIN,) if verification else requested
            assert requests and all(value == expected for value in requests)
            assert runner.prefill_attention_variants == requested
    finally:
        runner.prefill_attention_variants = previous


def test_one_row_matches_a_one_row_forward(runner) -> None:
    """At width one, both public forwarding forms are the same computation."""

    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = runner.forward([_TOKENS[1]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    in_one_call = runner.forward([_TOKENS[1]], logits_rows=1)

    assert in_one_call.shape == one_at_a_time.shape
    np.testing.assert_array_equal(in_one_call, one_at_a_time)


def test_multi_row_logits_are_bit_equal_to_one_row_forwards(runner) -> None:
    """Require identical logits at identical absolute positions and contexts."""

    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = np.stack([runner.forward([t]) for t in _TOKENS[1:]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    in_one_call = runner.forward(_TOKENS[1:], logits_rows=len(_TOKENS) - 1)

    assert in_one_call.shape == one_at_a_time.shape == (4, one_at_a_time.shape[1])
    np.testing.assert_array_equal(in_one_call, one_at_a_time)


@pytest.mark.parametrize("exact", [True, False])
def test_verify_block_requests_row_exact_launches_only_when_exact(runner, monkeypatch, exact):
    """``verify_dense_exact`` decides the dense schedule and the per-row attention."""

    from hipengine.runtime import gemma4 as module

    seen = []
    original = module.gemma4_layer_forward_bf16

    def record(*args, **kwargs):
        seen.append((kwargs["dense_wmma_prefill"], kwargs["row_decode_attention"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "gemma4_layer_forward_bf16", record)
    previous = runner.verify_dense_exact
    runner.verify_dense_exact = exact
    try:
        runner.reset()
        runner.forward(_TOKENS[:1])
        seen.clear()
        runner.forward(_TOKENS[1:], logits_rows=4, verification=True)
        assert seen
        for dense_wmma, rows in seen:
            assert dense_wmma is (not exact)
            if exact:
                # Row t of the block sits at position 1 + t and attends from key 0.
                assert rows.key_begins == (0, 0, 0, 0)
            else:
                assert rows is None
        # A prefill block never asks for either, whatever the setting.
        runner.reset()
        seen.clear()
        runner.forward(_TOKENS, logits_rows=5)
        assert seen and all(entry == (True, None) for entry in seen)
    finally:
        runner.verify_dense_exact = previous


def test_exact_verify_block_is_bit_equal_to_one_row_forwards(runner) -> None:
    """The verify phase's rows are the one-token steps' rows, bit for bit."""

    assert runner.verify_dense_exact
    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = np.stack([runner.forward([t]) for t in _TOKENS[1:]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    block = runner.forward(_TOKENS[1:], logits_rows=4, verification=True)

    np.testing.assert_array_equal(block, one_at_a_time)


def test_multi_row_logits_agree_on_the_sampled_token(runner) -> None:
    """Require greedy acceptance to see the same winners on this fixture."""

    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = np.stack([runner.forward([t]) for t in _TOKENS[1:]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    in_one_call = runner.forward(_TOKENS[1:], logits_rows=len(_TOKENS) - 1)

    assert [int(np.argmax(row)) for row in in_one_call] == [
        int(np.argmax(row)) for row in one_at_a_time
    ]


def test_a_partial_multi_row_request_returns_the_trailing_rows(runner) -> None:
    """``logits_rows=k`` means the last k rows, which is what acceptance reads.

    A verify pass forwards the seed token and the draft together, then accepts a
    prefix of the draft. The row that decides the first drafted token is the one
    the seed token produced, so the rows the caller wants are the trailing ones
    and the mapping has to be by offset rather than by row index.

    Both shape and values must match the trailing rows of the wider request.
    """

    runner.reset()
    runner.forward(_TOKENS[:1])
    all_rows = runner.forward(_TOKENS[1:], logits_rows=4)

    runner.reset()
    runner.forward(_TOKENS[:3])
    trailing = runner.forward(_TOKENS[3:], logits_rows=2)

    assert trailing.shape == (2, all_rows.shape[1])
    np.testing.assert_array_equal(trailing, all_rows[2:])


def test_rewind_gives_back_a_rejected_tail(runner) -> None:
    """A rewound append leaves no trace in the next forward's output.

    This is the speculative step's exact shape: append a draft, reject part of
    it, then append the one token that was accepted. The comparison sequence
    never appended the rejected tokens at all.
    """

    runner.reset()
    runner.forward(_TOKENS[:3])
    rejected = runner.forward(_TOKENS[3:])
    assert runner.position == 5
    runner.rewind(3)
    assert runner.position == 3
    after_rewind = runner.forward([_TOKENS[4]])

    runner.reset()
    runner.forward(_TOKENS[:3])
    never_appended = runner.forward([_TOKENS[4]])

    np.testing.assert_array_equal(after_rewind, never_appended)
    assert runner.position == 4


def test_rewind_refuses_positions_outside_the_sequence(runner) -> None:
    runner.reset()
    runner.forward(_TOKENS[:2])
    runner.rewind(0)
    assert runner.position == 0
    runner.forward(_TOKENS[:2])
    with pytest.raises(ValueError, match="cannot rewind"):
        runner.rewind(3)
    with pytest.raises(ValueError, match="cannot rewind"):
        runner.rewind(-1)


def test_logits_rows_beyond_the_buffer_bound_is_refused(runner) -> None:
    """The bound is the constructor's, so a caller cannot overrun the scratch.

    The projection buffers are sized from ``max_logits_rows``; asking for more
    rows than that would write past them rather than fail.
    """

    runner.reset()
    with pytest.raises(ValueError, match="logits_rows 6 is outside"):
        runner.forward(_TOKENS, logits_rows=6)
    with pytest.raises(ValueError, match="logits_rows 0 is outside"):
        runner.forward(_TOKENS, logits_rows=0)


def test_max_logits_rows_defaults_to_one_and_is_bounded(runner) -> None:
    from hipengine.runtime.gemma4 import Gemma4Runner

    assert runner.max_logits_rows == 5
    with pytest.raises(ValueError, match="max_logits_rows must be positive"):
        Gemma4Runner(weights=runner.weights, capacity=64, max_logits_rows=0)
    with pytest.raises(ValueError, match="must not exceed max_block"):
        Gemma4Runner(weights=runner.weights, capacity=64, max_block=4, max_logits_rows=5)
