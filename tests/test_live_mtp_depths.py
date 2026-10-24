"""Real-device parity for the packed PARO MTP candidate chain at depths 1..3.

What this pins down, on real packed PARO MTP weights:

* ``NativeMtpChainProposer.advance_chain_step`` (device token hand-off) produces
  the same bytes as the host-read scalar ``advance(input_token=...)`` it
  replaces: candidate int32 winner, published int64 token, final hidden, live
  K/V bytes, position and cache length, at every proposed prefix.
* ``publish_chain_token`` widens the *latest candidate row's* int32 winner into
  the int64 token the fuse kernel reads. There is no scratch re-reduction, so
  the published token must equal that row's candidate exactly.
* ``save_state``/``restore_state`` carry the accepted-prefix candidate identity:
  the row the candidate was written to, that row's scalar, the published/stale
  int64 flags, and a restore that puts the saved scalar back on its own row. A
  restore followed by the correction/bonus reseed must equal replaying the
  accepted prefix from the prompt with host scalars.

Fixtures are synthetic where the real route supplies a target: the W8A16
full-vocabulary scorer is a shared borrowed head with a deterministic
nonconstant pattern, and target hidden rows are seeded noise. Prompt priming
itself is covered by ``tests/test_live_mtp_prompt_kv.py``; this file starts from
a primed provider.

The scorer is built so its answers are not degenerate: every vocab row outside
``_VARIED_ROWS`` is zero, and each of the ``_VARIED_ROWS // 2`` seeded patterns
appears together with its exact negation. Because a dot product with a negated
row is the exact negation of the dot product with the row, at least one varied
row has a non-negative logit for any hidden, and a varied row (lower index) wins
every zero-logit tie. So a winner is always a varied row, whatever the hidden
is.

Run: ``uv run pytest --suite live tests/test_live_mtp_depths.py -v``
Model: ``HIPENGINE_TEST_PARO_MODEL`` (default below).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.speculative import mtp_native
from hipengine.speculative.mtp_native import NativeMtpChainProposer, _f32_to_bf16_bits

_MODEL_ENV = "HIPENGINE_TEST_PARO_MODEL"
_DEFAULT_MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"

_DEPTHS = (1, 2, 3)
_VARIED_ROWS = 128
_PROMPT_TOKENS = (101, 211, 307)
_SEED_TOKEN = 409
_BONUS_TOKEN = 977
_TARGET_ROWS = 16
_INT32 = np.dtype(np.int32).itemsize
_INT64 = np.dtype(np.int64).itemsize


@pytest.fixture(scope="module")
def live():
    """Two real proposers sharing one borrowed nonconstant full-vocab W8 head."""

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is unavailable")
    model = Path(os.environ.get(_MODEL_ENV, _DEFAULT_MODEL))
    if not (model / "config.json").is_file():
        pytest.skip("PARO model fixture is unavailable")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.loading import load_weight_index, qwen35_paro_config_from_hf
    from hipengine.speculative.mtp_native import NativeMtpW8A16Head

    config = qwen35_paro_config_from_hf(load_weight_index(model).config)
    runtime = get_hip_runtime()
    hidden, vocab = int(config.hidden_size), int(config.vocab_size)
    weight = malloc(vocab * hidden, runtime=runtime)
    scale = malloc(vocab * np.dtype(np.float32).itemsize, runtime=runtime)
    target_rows = malloc(_TARGET_ROWS * hidden * DType.BF16.itemsize, runtime=runtime)
    owners: list[NativeMtpChainProposer] = []
    try:
        # Zero the whole vocabulary, then give the first _VARIED_ROWS rows a
        # seeded pattern and its exact negation (see the module docstring for
        # why that makes every winner a varied row).
        runtime.memset(weight.ptr, 0, weight.nbytes)
        rng = np.random.default_rng(20261024)
        base = rng.integers(1, 8, size=(_VARIED_ROWS // 2, hidden), dtype=np.int8)
        varied = np.ascontiguousarray(np.concatenate((base, -base), axis=0))
        assert len({row.tobytes() for row in base}) == _VARIED_ROWS // 2, (
            "W8 patterns are not distinct"
        )
        copy_host_to_device(weight, host_array_ptr(varied), varied.nbytes, runtime=runtime)
        scales = np.zeros((vocab,), dtype=np.float32)
        scales[:_VARIED_ROWS] = np.float32(1.0) / np.float32(hidden)
        copy_host_to_device(scale, host_array_ptr(scales), scales.nbytes, runtime=runtime)
        head = NativeMtpW8A16Head(weight.ptr, scale.ptr, vocab)
        rows = _f32_to_bf16_bits(rng.normal(0, 0.4, (_TARGET_ROWS, hidden)).astype(np.float32))
        copy_host_to_device(target_rows, host_array_ptr(rows), rows.nbytes, runtime=runtime)
        with pytest.MonkeyPatch.context() as patch:
            # Keep the unused fp16 lm-head scratch small; the W8A16 path scores
            # the full vocabulary regardless of the draft cap.
            patch.setattr(mtp_native, "_draft_vocab_from_env", lambda _vocab: 64)
            for _ in range(2):
                owners.append(
                    NativeMtpChainProposer(
                        model,
                        max_positions=1024,
                        max_mtp_tokens=512,
                        max_candidate_rows=len(_DEPTHS),
                        scoring_head=head,
                    )
                )
        yield SimpleNamespace(
            reference=owners[0],
            chain=owners[1],
            runtime=runtime,
            hidden=hidden,
            vocab=vocab,
            target_rows=target_rows,
        )
    finally:
        for owner in reversed(owners):
            owner.close()
        for buffer in reversed((weight, scale, target_rows)):
            free(buffer, runtime=runtime)


def _read(owner, ptr, shape, dtype):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(
        host_array_ptr(host), DeviceBuffer(ptr, host.nbytes), host.nbytes, runtime=owner.runtime
    )
    return host


def _settle(live, stream=0):
    if int(stream) == 0:
        live.runtime.device_synchronize()
    else:
        live.runtime.stream_synchronize(int(stream))


def _prime(owner, live, *, stream=0):
    """Reset and replay the shared prompt rows; return the first candidate id."""

    owner.reset()
    result = owner.current
    for index in range(len(_PROMPT_TOKENS)):
        token = _PROMPT_TOKENS[index + 1] if index + 1 < len(_PROMPT_TOKENS) else _SEED_TOKEN
        result = owner.advance_with_target_hidden(
            input_token=int(token),
            target_hidden_ptr=live.target_rows.ptr + index * live.hidden * DType.BF16.itemsize,
            position=index + 1,
            read_expert_topk=False,
            read_lm_head_value=False,
            stream=stream,
        )
    return int(result.token)


def _assert_live_equal(live, *, rows, chain_stream=0, scored=True):
    """Every live byte a chain step can move, compared across both owners."""

    reference, chain = live.reference, live.chain
    _settle(live, chain_stream)
    assert chain.cache_len == reference.cache_len
    assert chain.position == reference.position
    for name, dtype in (("key_cache_buf", np.float32), ("value_cache_buf", np.uint16)):
        shape = (reference.cache_len, reference.kv_features)
        a = _read(reference, getattr(reference, name).ptr, shape, dtype)
        b = _read(chain, getattr(chain, name).ptr, shape, dtype)
        assert a.tobytes() == b.tobytes(), name
        if dtype is np.float32:
            assert np.isfinite(a).all(), name
    assert (
        reference.read_final_hidden_bf16().tobytes() == chain.read_final_hidden_bf16().tobytes()
    ), "final hidden"
    a = _read(reference, reference.candidate_ids_buf.ptr, (int(rows),), np.int32)
    b = _read(chain, chain.candidate_ids_buf.ptr, (int(rows),), np.int32)
    assert a.tobytes() == b.tobytes(), "candidate rows"
    if scored:
        a = _read(reference, reference.out_value_buf.ptr, (1,), np.float32)
        b = _read(chain, chain.out_value_buf.ptr, (1,), np.float32)
        assert a.tobytes() == b.tobytes(), "top-1 logit"


def _assert_published(live, *, row, stream=0):
    """The published int64 token must be the candidate winner of ``row``.

    ``publish_chain_token`` widens one candidate row's int32 winner into the
    int64 token the fuse kernel reads, so this compares exactly those two
    values: no scratch re-reduction, no value buffer.
    """

    chain = live.chain
    _settle(live, stream)
    winner = int(_read(chain, chain.candidate_ids_buf.ptr + int(row) * _INT32, (1,), np.int32)[0])
    token = int(_read(chain, chain.token_buf.ptr, (1,), np.int64)[0])
    assert token == winner, f"published int64 token {token} != candidate row {row} winner {winner}"
    return winner


def _propose_chain(live, *, depth, chain_stream=0):
    """Mirror ``Qwen35ParoMTP2Adapter.propose_batch`` and compare every prefix.

    The reference owner runs the host-read scalar protocol: it reads each
    candidate back to the host and feeds that int64 into the next advance. The
    chain owner publishes the candidate on device and consumes it from device.
    """

    reference, chain = live.reference, live.chain
    first = _prime(reference, live)
    assert _prime(chain, live, stream=chain_stream) == first
    _assert_live_equal(live, rows=1, chain_stream=chain_stream)
    chain.ensure_chain_token(stream=chain_stream)
    assert _assert_published(live, row=0, stream=chain_stream) == first
    snapshots = [chain.save_state(0, stream=chain_stream)]
    winners = [first]
    for step in range(1, int(depth)):
        result = reference.advance(
            input_token=winners[-1],
            target_hidden_ptr=reference.final_hidden_buf.ptr,
            position=reference.position + 1,
            candidate_row=step,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        chain.advance_chain_step(candidate_row=step, stream=chain_stream)
        chain.publish_chain_token(stream=chain_stream)
        winner = _assert_published(live, row=step, stream=chain_stream)
        assert winner == int(result.token), f"depth {depth} step {step}: {winner} != {result.token}"
        _assert_live_equal(live, rows=step + 1, chain_stream=chain_stream)
        winners.append(winner)
        snapshots.append(chain.save_state(step, stream=chain_stream))
    return winners, snapshots


def test_the_synthetic_head_produces_distinct_actual_winners(live):
    """Guard the gate itself: a degenerate head would make parity vacuous."""

    reference = live.reference
    winners = [_prime(reference, live)]
    for step in range(1, len(_DEPTHS)):
        result = reference.advance(
            input_token=winners[-1],
            target_hidden_ptr=reference.final_hidden_buf.ptr,
            position=reference.position + 1,
            candidate_row=step,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        winners.append(int(result.token))
    assert all(0 <= winner < _VARIED_ROWS for winner in winners), winners
    assert len(set(winners)) >= 2, (
        "the synthetic W8 head produced identical winners "
        f"{winners}; the chain is not discriminating and this gate would be vacuous"
    )


@pytest.mark.parametrize("depth", _DEPTHS)
def test_device_chain_matches_host_scalar_advance_at_every_prefix(live, depth):
    reference = live.reference
    winners, snapshots = _propose_chain(live, depth=depth)
    assert len(winners) == depth
    assert len(snapshots) == depth
    if depth == 1:
        # Depth 1 must not advance the provider at all: proposal is the primed
        # prefix plus one published scalar.
        assert snapshots[0].cache_len == reference.cache_len
        assert snapshots[0].position == reference.position
    else:
        assert snapshots[-1].cache_len == snapshots[0].cache_len + depth - 1
        assert snapshots[-1].position == snapshots[0].position + depth - 1
    _assert_live_equal(live, rows=depth)
    assert _assert_published(live, row=depth - 1) == winners[-1]


def test_unsynchronized_proposals_match_the_synchronized_scalar_path(live):
    """No host read, no D2H and no sync between chain steps."""

    reference, chain = live.reference, live.chain
    winners = [_prime(reference, live)]
    for step in range(1, len(_DEPTHS)):
        result = reference.advance(
            input_token=winners[-1],
            target_hidden_ptr=reference.final_hidden_buf.ptr,
            position=reference.position + 1,
            candidate_row=step,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        winners.append(int(result.token))
    assert _prime(chain, live) == winners[0]
    live.runtime.device_synchronize()
    # The reference path is fully drained above; the chain below submits
    # everything and drains once, at the end.
    chain.ensure_chain_token()
    chain.advance_chain_step(candidate_row=1)
    chain.publish_chain_token()
    chain.advance_chain_step(candidate_row=2)
    chain.publish_chain_token()
    _assert_live_equal(live, rows=len(_DEPTHS))
    assert int(_read(chain, chain.token_buf.ptr, (1,), np.int64)[0]) == winners[-1]


def test_private_stream_proposals_match_the_default_stream_scalar_path(live):
    stream = live.runtime.stream_create()
    if not stream:
        pytest.skip("HIP private stream creation is unavailable")
    try:
        # One ordering point so the fixture's default-stream uploads precede the
        # private stream's first copy. Everything after it is stream-ordered.
        live.runtime.device_synchronize()
        winners, snapshots = _propose_chain(live, depth=len(_DEPTHS), chain_stream=stream)
        _assert_live_equal(live, rows=len(_DEPTHS), chain_stream=stream)
        assert _assert_published(live, row=len(_DEPTHS) - 1, stream=stream) == winners[-1]
        assert snapshots[-1].cache_len == snapshots[0].cache_len + len(_DEPTHS) - 1
    finally:
        live.runtime.stream_destroy(stream)


def test_prefix_snapshots_restore_the_candidate_identity_and_flags(live):
    chain = live.chain
    winners, snapshots = _propose_chain(live, depth=len(_DEPTHS))
    assert [snapshot.slot for snapshot in snapshots] == list(range(len(_DEPTHS)))
    # Overwrite every candidate row with a value no scoring pass can produce, so
    # a restore that always writes row 0 cannot pass by leaving the chain's own
    # rows in place. Restoring prefix i must put the saved scalar back on row i.
    sentinels = np.array([_VARIED_ROWS + 7 + row for row in range(len(_DEPTHS))], dtype=np.int32)
    copy_host_to_device(
        DeviceBuffer(chain.candidate_ids_buf.ptr, sentinels.nbytes),
        host_array_ptr(sentinels),
        sentinels.nbytes,
        runtime=chain.runtime,
    )
    for prefix, snapshot in enumerate(snapshots):
        chain.restore_state(snapshot)
        assert chain.cache_len == snapshot.cache_len
        assert chain.position == snapshot.position
        assert snapshot.candidate_ready is True
        assert snapshot.chain_token_current is True
        assert chain._candidate_ready is True
        assert chain._chain_token_current is True
        assert _assert_published(live, row=prefix) == winners[prefix]
    # A restored, published prefix needs no republish before the next chain step.
    chain.restore_state(snapshots[1])
    chain.advance_chain_step(candidate_row=2)
    assert chain._chain_token_current is False


def _unpublished_candidate(live, *, depth=len(_DEPTHS)):
    """Leave the chain owner on an unpublished prefix above row 0.

    Returns ``(row, winner, input_token)`` where ``row >= 1``, row ``row`` holds
    ``winner``, and ``winner`` differs from the row-0 candidate, so a republish
    that reads row 0 instead of the latest row cannot pass by luck.
    """

    chain = live.chain
    winners, snapshots = _propose_chain(live, depth=depth)
    for row in range(depth - 1, 0, -1):
        if winners[row] != winners[0]:
            # Re-derive the same prefix without publishing, so the snapshot's
            # int64 is the advance's input token rather than this row's winner.
            chain.restore_state(snapshots[row - 1])
            chain.advance_chain_step(candidate_row=row)
            return row, winners[row], winners[row - 1]
    # Every natural candidate matched row 0: score the prefix-0 candidate against
    # a different target hidden row until the winner separates from row 0.
    for target_row in range(len(_PROMPT_TOKENS) + 2, _TARGET_ROWS):
        chain.restore_state(snapshots[0])
        chain.advance(
            input_token=winners[0],
            target_hidden_ptr=live.target_rows.ptr + target_row * live.hidden * DType.BF16.itemsize,
            position=chain.position + 1,
            candidate_row=1,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
        winner = int(_read(chain, chain.candidate_ids_buf.ptr + _INT32, (1,), np.int32)[0])
        if winner != winners[0]:
            return 1, winner, winners[0]
    raise AssertionError(
        "could not stage an unpublished candidate that differs from the row-0 candidate"
    )


def test_restored_unpublished_prefix_republishes_its_own_row(live):
    """An unpublished snapshot must republish its own row, not row 0."""

    chain = live.chain
    row, winner, input_token = _unpublished_candidate(live)
    assert row >= 1
    # The staging is only discriminating if row 0 still holds something else.
    assert int(_read(chain, chain.candidate_ids_buf.ptr, (1,), np.int32)[0]) != winner
    assert chain._candidate_ready is True
    assert chain._chain_token_current is False
    unpublished = chain.save_state(row)
    assert unpublished.candidate_ready is True
    assert unpublished.chain_token_current is False
    # The saved int64 is the advance's input token, not this row's candidate.
    assert int(_read(chain, chain.token_buf.ptr, (1,), np.int64)[0]) == input_token
    with pytest.raises(RuntimeError, match="published chain token"):
        chain.advance_chain_step(candidate_row=1)
    chain.restore_state(unpublished)
    assert chain._chain_token_current is False
    with pytest.raises(RuntimeError, match="published chain token"):
        chain.advance_chain_step(candidate_row=1)
    # The republish must read the latest candidate row. Row 0 still holds a
    # different winner, so a row-0 read fails this assertion.
    chain.ensure_chain_token()
    assert chain._chain_token_current is True
    assert _assert_published(live, row=row) == winner


def test_publish_reads_the_latest_candidate_row(live):
    """A publish after a new scoring pass reports that row's winner, not a cached one."""

    chain = live.chain
    first = _prime(chain, live)
    chain.publish_chain_token()
    stale = int(_read(chain, chain.token_buf.ptr, (1,), np.int64)[0])
    assert stale == first
    # The next scoring pass writes row 1 and moves the latest-candidate row.
    chain.advance_chain_step(candidate_row=1)
    assert chain._chain_token_current is False
    winner = int(_read(chain, chain.candidate_ids_buf.ptr + _INT32, (1,), np.int32)[0])
    chain.publish_chain_token()
    token = int(_read(chain, chain.token_buf.ptr, (1,), np.int64)[0])
    assert token == winner
    if winner != stale:
        assert token != stale


@pytest.mark.parametrize("accepted", (0, 1, 2, 3))
@pytest.mark.parametrize("depth", _DEPTHS)
def test_acceptance_and_reseed_restore_equals_prompt_replay(live, depth, accepted):
    """Restore-to-accepted-prefix must equal replaying that prefix from the prompt."""

    if accepted > depth:
        pytest.skip(f"depth {depth} cannot accept {accepted} drafts")
    reference, chain = live.reference, live.chain
    winners, snapshots = _propose_chain(live, depth=depth)
    chain_end = depth - 1
    prefix = min(accepted, chain_end)
    # Provider side: the adapter's restore, then the repair only when the last
    # draft was accepted (that advance was never materialized by the chain).
    if prefix != chain_end:
        chain.restore_state(snapshots[prefix])
    if accepted == depth:
        chain.advance_with_previous_hidden(
            input_token=winners[chain_end],
            position=chain.position + 1,
            need_result=False,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
    # Reference side: reach the same prefix by replaying it from the prompt with
    # host scalars, mirroring the chain's row assignment exactly (drafts 1..n in
    # rows 1..n, the repair in row 0).
    _prime(reference, live)
    for index in range(prefix):
        reference.advance(
            input_token=winners[index],
            target_hidden_ptr=reference.final_hidden_buf.ptr,
            position=reference.position + 1,
            candidate_row=index + 1,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
    if accepted == depth:
        reference.advance_with_previous_hidden(
            input_token=winners[chain_end],
            position=reference.position + 1,
            need_result=False,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
    # Both owners now stand at the same accepted prefix with the same candidate
    # rows: restore-from-snapshot must equal replay-from-prompt byte for byte.
    # Scoring diagnostics are transient scratch, not provider state. A restore
    # need not restore out_value_buf; the reseed below must recompute an exact
    # value before any result-producing continuation consumes it.
    _assert_live_equal(live, rows=prefix + 1, scored=False)
    # Correction/bonus reseed on the target's own hidden row.
    target_hidden_ptr = (
        live.target_rows.ptr + (len(_PROMPT_TOKENS) + 1) * live.hidden * DType.BF16.itemsize
    )
    for owner in (reference, chain):
        owner.advance_with_target_hidden(
            input_token=_BONUS_TOKEN,
            target_hidden_ptr=target_hidden_ptr,
            position=owner.position + 1,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
    _assert_live_equal(live, rows=1)
    # Following cycle: the reseed's scoring pass rewrote row 0 and the scratch,
    # so the next publish must report the reseeded candidate, not a stale one.
    following = int(_read(reference, reference.candidate_ids_buf.ptr, (1,), np.int32)[0])
    chain.ensure_chain_token()
    assert _assert_published(live, row=0) == following
    result = reference.advance(
        input_token=following,
        target_hidden_ptr=reference.final_hidden_buf.ptr,
        position=reference.position + 1,
        candidate_row=1,
        read_expert_topk=False,
        read_lm_head_value=False,
    )
    chain.advance_chain_step(candidate_row=1)
    chain.publish_chain_token()
    assert _assert_published(live, row=1) == int(result.token)
    _assert_live_equal(live, rows=2)
