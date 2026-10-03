"""Multi-row logits and rewind, on the synthetic fixture artifact.

A speculative verify pass forwards a whole draft in one call and reads the
target's own distribution at every drafted position, then gives back the
rejected tail. Greedy output survives that only if both operations are exact: a
row's logits must not depend on how many rows accompany it, and a rewound cache
must recompute what it would have computed had the rejected tokens never been
appended.

**The first of those does not hold today, and this file records it rather than
asserting it.** A one-row block is bit-exact against a one-row forward, but a
block of two or more rows is not: the hidden state after the layers already
differs, at every row including the first, and the difference is the same
whatever the block width is, so each row is computed independently of its
companions and the divergence is between a one-row block and a wider one. That
is a route selected by batch shape somewhere in the layers, the same class of
defect ``tests/test_gpu_gemma4_attention_decode_parity.py`` and
``test_incremental_decode_matches_a_dense_prefill`` already catch. It predates
the ``logits_rows`` plumbing here, which only changes the final norm and head.

The consequence for speculative decoding is that a verify pass is an arithmetic
change and not a free one, so it needs the ``docs/EXECUTION-PROFILES.md`` gate
before it can be a default. The top-1 token still agrees at every position on
this fixture, which is what greedy acceptance actually reads, but one fixture is
not that gate.
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
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import PREFILL_ATTENTION_WMMA_FLASH

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
            expected = ("gemma4_plain",) if verification else requested
            assert requests and all(value == expected for value in requests)
            assert runner.prefill_attention_variants == requested
    finally:
        runner.prefill_attention_variants = previous


def test_one_row_matches_a_one_row_forward(runner) -> None:
    """The control: at width one the two paths are the same computation.

    This is what makes the wider-row failure below a route difference rather
    than a bug in the multi-row plumbing. The hidden state is compared too,
    because that is where the divergence starts.
    """

    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = runner.forward([_TOKENS[1]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    in_one_call = runner.forward([_TOKENS[1]], logits_rows=1)

    assert in_one_call.shape == one_at_a_time.shape
    np.testing.assert_array_equal(in_one_call, one_at_a_time)


@pytest.mark.xfail(
    reason="a two-or-wider block takes a different layer route than a one-row "
    "forward; the divergence is in the hidden state, not in logits_rows",
    strict=False,
)
def test_multi_row_logits_are_bit_equal_to_one_row_forwards(runner) -> None:
    """The verify's premise, which does not hold yet.

    The mask is built from absolute positions, so row ``i`` of a five-row block
    should attend over exactly what the ``i``-th one-row forward attended over.
    It does not: the hidden state after the layers already differs, so some
    kernel in the layers picks its route from the block width. Fixing that is
    what would make a speculative verify bit-exact; until then it is an
    arithmetic change.
    """

    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = np.stack([runner.forward([t]) for t in _TOKENS[1:]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    in_one_call = runner.forward(_TOKENS[1:], logits_rows=len(_TOKENS) - 1)

    assert in_one_call.shape == one_at_a_time.shape == (4, one_at_a_time.shape[1])
    np.testing.assert_array_equal(in_one_call, one_at_a_time)


def test_multi_row_logits_agree_on_the_sampled_token(runner) -> None:
    """What greedy acceptance actually reads, and it holds today.

    Acceptance compares the target's argmax at each drafted position against
    the draft. The values differ by up to 2.5 on this fixture while the top-1
    token agrees everywhere, which is why the route difference is a promotion
    question and not an immediate correctness failure -- but it is only one
    fixture, so it is not the gate either.
    """

    runner.reset()
    runner.forward(_TOKENS[:1])
    one_at_a_time = np.stack([runner.forward([t]) for t in _TOKENS[1:]])

    runner.reset()
    runner.forward(_TOKENS[:1])
    in_one_call = runner.forward(_TOKENS[1:], logits_rows=len(_TOKENS) - 1)

    assert [int(np.argmax(row)) for row in in_one_call] == [
        int(np.argmax(row)) for row in one_at_a_time
    ]


@pytest.mark.xfail(
    reason="the trailing rows come from a narrower block than the full set, and "
    "a narrower block takes a different layer route",
    strict=False,
)
def test_a_partial_multi_row_request_returns_the_trailing_rows(runner) -> None:
    """``logits_rows=k`` means the last k rows, which is what acceptance reads.

    A verify pass forwards the seed token and the draft together, then accepts a
    prefix of the draft. The row that decides the first drafted token is the one
    the seed token produced, so the rows the caller wants are the trailing ones
    and the mapping has to be by offset rather than by row index.

    The shape is the plumbing contract and holds; the values are subject to the
    same block-width route difference as the bit-equality test above, because a
    two-row block and a four-row block are not the same computation today.
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
