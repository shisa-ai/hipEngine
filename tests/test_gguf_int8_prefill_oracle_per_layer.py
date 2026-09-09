"""Per-layer BF16 oracle ownership for multi-chunk packed INT8 prefill.

2026-09-10 P1 regression tests. The packed slot-local INT8 prefill route
(``int8_direct`` sessions are forced onto it above the mirror threshold)
shared one BF16 oracle pair across every INT8 layer under a chunk-outer
executor: chunks outside, layers inside. From the second chunk on, every
layer attended over the previous chunk's *last* layer's K/V, so any prompt
spanning more than one prefill chunk produced wrong output. The fix keys the
oracle per INT8 layer whenever ``_prefill_batch_native_impl`` plans more than
one chunk, sets that ownership before the first slab, and clears it in
``prefill_batch_native``'s finally alongside the oracle release.

Contracts under test (CPU, fake device):

- ``_int8_prefill_oracle_cache_for_layer`` shares one pair only under a
  ``layer_outer_shared_oracle`` plan *without* the per-layer override; the
  override keys one pair per layer; any other plan keys per layer regardless;
- ``_prefill_batch_native_impl`` sets ownership on every session before the
  first slab, for any multi-chunk plan including tail chunks, and leaves it
  unset for single-chunk calls;
- ``prefill_batch_native``'s finally clears the flag on every session even
  when a slab raises, and releases the oracle buffers, so a later
  single-chunk call cannot inherit per-layer keying;
- ``_int8_prefill_oracle_capacity_positions`` stays sized by the whole
  physical backing pool, not by ``max_positions`` or the prompt length,
  because the oracle rows are addressed through physical page addresses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _plan_packed_ar_prefill_chunks,
)

from tests.test_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)


def _oracle_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str | None = "layer_outer_shared_oracle",
    per_layer: bool = False,
) -> Qwen35GGUFResidentSession:
    _install_fake_device(monkeypatch)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.__dict__.update(
        runner=_fake_dense_qwen36_runner(),
        runtime=SimpleNamespace(),
        scratch=SimpleNamespace(max_positions=1_024, block_size=256),
        _device_kv_allocation=None,
        _int8_prefill_oracle_buffers={},
        _int8_prefill_oracle_per_layer=bool(per_layer),
        _int8_prefill_lifetime_plan=(
            None if mode is None else SimpleNamespace(mode=mode)
        ),
    )
    return session


# ---------------------------------------------------------------------------
# Oracle cache keying
# ---------------------------------------------------------------------------


def test_shared_plan_without_override_shares_one_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _oracle_session(monkeypatch)
    assert session._int8_prefill_oracle_cache_for_layer(3) is (
        session._int8_prefill_oracle_cache_for_layer(5)
    )
    assert set(session._int8_prefill_oracle_buffers) == {-1}


def test_per_layer_override_keys_one_pair_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _oracle_session(monkeypatch, per_layer=True)
    pair_3 = session._int8_prefill_oracle_cache_for_layer(3)
    pair_5 = session._int8_prefill_oracle_cache_for_layer(5)
    assert pair_3 is session._int8_prefill_oracle_cache_for_layer(3)
    assert pair_3 is not pair_5
    assert set(session._int8_prefill_oracle_buffers) == {3, 5}


def test_shared_plan_with_override_does_not_touch_the_shared_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override must not poison the shared key for later readers."""

    session = _oracle_session(monkeypatch, per_layer=True)
    session._int8_prefill_oracle_cache_for_layer(3)
    session._int8_prefill_oracle_per_layer = False
    assert -1 not in session._int8_prefill_oracle_buffers
    assert session._int8_prefill_oracle_cache_for_layer(5) is (
        session._int8_prefill_oracle_cache_for_layer(7)
    )
    assert set(session._int8_prefill_oracle_buffers) == {3, -1}


@pytest.mark.parametrize("mode", ["chunk_outer_direct_int8", None])
def test_non_shared_plans_key_per_layer_without_the_override(
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
) -> None:
    session = _oracle_session(monkeypatch, mode=mode, per_layer=False)
    assert session._int8_prefill_oracle_cache_for_layer(3) is not (
        session._int8_prefill_oracle_cache_for_layer(5)
    )
    assert set(session._int8_prefill_oracle_buffers) == {3, 5}


def test_release_frees_every_per_layer_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _oracle_session(monkeypatch, per_layer=True)
    freed: list[object] = []
    monkeypatch.setattr(gguf_runner, "free", lambda buffer, *, runtime: freed.append(buffer))
    for layer_id in (3, 5, 9):
        session._int8_prefill_oracle_cache_for_layer(layer_id)
    live = tuple(session._int8_prefill_oracle_buffers.values())
    session._release_int8_prefill_oracle_buffers()
    assert session._int8_prefill_oracle_buffers == {}
    assert sorted(freed, key=id) == sorted(
        (buffer for pair in live for buffer in pair), key=id
    )
    # Re-acquisition after release starts from an empty cache, never a stale
    # pair left behind by a partially released per-layer prefix.
    session._int8_prefill_oracle_cache_for_layer(3)
    assert set(session._int8_prefill_oracle_buffers) == {3}


# ---------------------------------------------------------------------------
# Ownership lifecycle across packed prefill calls
# ---------------------------------------------------------------------------


def _flag_recorder(monkeypatch: pytest.MonkeyPatch) -> list[list[bool]]:
    """Record every session's per-layer flag at each slab entry."""

    observed: list[list[bool]] = []

    def fake_single_slab(self, prompt_token_ids, *, sessions, **kwargs):
        del self, prompt_token_ids, kwargs
        observed.append(
            [
                bool(getattr(session, "_int8_prefill_oracle_per_layer", False))
                for session in sessions
            ]
        )
        for session in sessions:
            session.position += 1
        return [SimpleNamespace(token_id=1) for _ in sessions]

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        fake_single_slab,
        raising=False,
    )
    return observed


def _bare_owner(row_capacity: int) -> Qwen35GGUFResidentSession:
    owner = object.__new__(Qwen35GGUFResidentSession)
    owner._bulk_prefill_scratch = SimpleNamespace(rows=int(row_capacity))
    return owner


def test_multi_chunk_call_sets_per_layer_ownership_before_the_first_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _flag_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = tuple(SimpleNamespace(position=0) for _ in range(2))
    prompts = tuple(tuple(range(8)) for _ in range(2))

    results = owner.prefill_batch_native(prompts, sessions=sessions)

    assert len(results) == 2
    assert owner.last_packed_prefill_plan["chunk_count"] == 2
    # Both slabs must observe the flag on every session, so it was set once,
    # before the first slab, and stayed set for the whole call.
    assert observed == [[True, True], [True, True]]
    # The finally clears it on every session after the call.
    assert all(
        session._int8_prefill_oracle_per_layer is False for session in sessions
    )


def test_single_chunk_call_keeps_shared_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _flag_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = (SimpleNamespace(position=0),)

    owner.prefill_batch_native((tuple(range(8)),), sessions=sessions)

    assert owner.last_packed_prefill_plan["chunk_count"] == 1
    assert observed == [[False]]
    assert sessions[0]._int8_prefill_oracle_per_layer is False


def test_tail_chunk_plans_still_set_per_layer_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny final chunk is still a second chunk: wrong to share a pair."""

    observed = _flag_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = (SimpleNamespace(position=0),)
    prompt = tuple(range(17))

    chunks = _plan_packed_ar_prefill_chunks((prompt,), row_capacity=8)
    assert [chunk.rows for chunk in chunks] == [8, 8, 1]

    owner.prefill_batch_native((prompt,), sessions=sessions)

    assert owner.last_packed_prefill_plan["chunk_count"] == 3
    assert owner.last_packed_prefill_plan["chunk_rows"] == [8, 8, 1]
    assert observed == [[True], [True], [True]]


def test_multi_chunk_then_single_chunk_reuse_keeps_shared_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later single-chunk call must not inherit per-layer keying."""

    observed = _flag_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = (SimpleNamespace(position=0),)

    owner.prefill_batch_native((tuple(range(16)),), sessions=sessions)
    owner.prefill_batch_native((tuple(range(4)),), sessions=sessions)

    assert observed == [[True], [True], [False]]


def test_prefill_batch_native_clears_the_flag_and_releases_when_a_slab_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding_single_slab(self, prompt_token_ids, *, sessions, **kwargs):
        del self, prompt_token_ids, kwargs, sessions
        raise RuntimeError("slab exploded")

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        exploding_single_slab,
        raising=False,
    )
    freed: list[object] = []
    monkeypatch.setattr(gguf_runner, "free", lambda buffer, *, runtime: freed.append(buffer))
    _install_fake_device(monkeypatch)
    owner = object.__new__(Qwen35GGUFResidentSession)
    owner.__dict__.update(
        runtime=SimpleNamespace(),
        _bulk_prefill_scratch=SimpleNamespace(rows=8),
        _int8_prefill_oracle_buffers={3: (object(), object())},
        _int8_prefill_retained_block_table=None,
        _int8_prefill_oracle_per_layer=True,
    )

    with pytest.raises(RuntimeError, match="slab exploded"):
        owner.prefill_batch_native((tuple(range(16)),))

    assert owner._int8_prefill_oracle_per_layer is False
    assert owner._int8_prefill_oracle_buffers == {}
    assert len(freed) == 2


# ---------------------------------------------------------------------------
# Oracle capacity addressing
# ---------------------------------------------------------------------------


def test_oracle_capacity_covers_the_whole_backing_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oracle is addressed through physical page rows, not logical ones.

    A session's block table can point at any physical page the backing pool
    owns (shared prefixes, shifted COW suffixes), so the oracle must span
    ``backing_pages * block_size`` even when that exceeds ``max_positions``.
    Shrinking it to the prompt length would strand those addresses.
    """

    session = _oracle_session(monkeypatch)
    session.scratch = SimpleNamespace(max_positions=1_024, block_size=256)
    session._device_kv_allocation = SimpleNamespace(
        backing=SimpleNamespace(pages=16)
    )
    assert session._int8_prefill_oracle_capacity_positions() == 16 * 256
    # A pool smaller than the declared context must not shrink the oracle.
    session._device_kv_allocation = SimpleNamespace(
        backing=SimpleNamespace(pages=2)
    )
    assert session._int8_prefill_oracle_capacity_positions() == 1_024


@pytest.mark.parametrize(
    "allocation",
    [None, SimpleNamespace(backing=None), SimpleNamespace(backing=SimpleNamespace(pages=0))],
)
def test_oracle_capacity_without_backing_pages_uses_max_positions(
    monkeypatch: pytest.MonkeyPatch,
    allocation,
) -> None:
    session = _oracle_session(monkeypatch)
    session._device_kv_allocation = allocation
    assert session._int8_prefill_oracle_capacity_positions() == 1_024
