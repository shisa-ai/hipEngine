"""Layer-outer packed AR prefill: eligibility gates and route selection.

2026-09-10 P3 regression tests. The chunk-outer packed slot-local INT8
executor realizes one transient BF16 oracle pair per INT8 layer (16 pairs on
the 27B) whenever a prompt spans multiple prefill rounds, because rounds
interleave layers. The layer-outer executor reorders the loops - every chunk
of a layer completes before the next layer - so one shared oracle pair per
session is sound again (the positions of different rounds never overlap
inside one layer).

Contracts under test (CPU, fake device):

- ``_prefill_batch_native_layer_outer`` fails closed (NotImplementedError,
  no device work) when the chunk rounds are not slot-stable, when the
  session's lifetime plan is not ``layer_outer_shared_oracle``, or when the
  plan's hidden capacity cannot hold every prompt row at once;
- ``_prefill_batch_native_impl`` dispatches to the layer-outer executor only
  when the feature flag is enabled and the request shape fits (no hidden
  seeds, no layer captures, no target-hidden sinks), and falls back to the
  corrected chunk-outer executor whenever the executor declines;
- the executor resets ``_int8_prefill_oracle_per_layer`` on every session
  before any layer runs, so the shared key is used - packed execution cannot
  silently realize per-layer pairs while the plan promises one;
- the feature flag defaults ON (promoted 2026-09-10 after the parity, wall
  A/B, trace-identity, and server-probe gates passed); the env var rolls
  back to the corrected chunk-outer executor.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _gguf_packed_layer_outer_enabled,
    _plan_packed_ar_prefill_chunks,
)

from tests.test_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)


@pytest.fixture(autouse=True)
def _reset_layer_outer_flag_cache():
    """The module-level flag cache must never leak between tests."""

    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    yield
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None


def _fake_runner_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan_mode: str | None = "layer_outer_shared_oracle",
    hidden_capacity: int = 1_024,
    per_layer: bool = True,
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
        _prefill_token_buf=SimpleNamespace(ptr=0),
        _prefill_hidden_a=SimpleNamespace(ptr=0),
        _prefill_hidden_b=SimpleNamespace(ptr=0),
        _int8_prefill_lifetime_plan=(
            None
            if plan_mode is None
            else SimpleNamespace(
                mode=plan_mode,
                required_hidden_capacity=int(hidden_capacity),
            )
        ),
    )
    return session


def _layer_outer_owner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan_mode: str | None = "layer_outer_shared_oracle",
    hidden_capacity: int = 1_024,
) -> Qwen35GGUFResidentSession:
    _install_fake_device(monkeypatch)
    owner = object.__new__(Qwen35GGUFResidentSession)
    owner.__dict__.update(
        _int8_prefill_lifetime_plan=(
            None
            if plan_mode is None
            else SimpleNamespace(
                mode=plan_mode,
                required_hidden_capacity=int(hidden_capacity),
            )
        ),
        _int8_prefill_oracle_per_layer=True,
        _int8_prefill_oracle_buffers={},
        last_packed_prefill_plan={},
    )
    return owner


def _chunks_for(
    prompts: tuple[tuple[int, ...], ...],
    *,
    row_capacity: int,
) -> tuple:
    return _plan_packed_ar_prefill_chunks(prompts, row_capacity=row_capacity)


# ---------------------------------------------------------------------------
# Eligibility gates (fail closed, no device work)
# ---------------------------------------------------------------------------


def test_slot_unstable_rounds_decline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _layer_outer_owner(monkeypatch)
    # 8 rows each, capacity 8: the shorter prompt finishes and later rounds
    # address only the surviving slot - not slot-stable.
    prompts = (tuple(range(16)), tuple(range(6)))
    chunks = _chunks_for(prompts, row_capacity=8)
    assert len(chunks) > 1
    with pytest.raises(NotImplementedError, match="slot-stable"):
        owner._prefill_batch_native_layer_outer(
            prompts,
            sessions=tuple(
                SimpleNamespace(position=0) for _ in range(len(prompts))
            ),
            chunks=chunks,
        )


@pytest.mark.parametrize("plan_mode", ["chunk_outer_layer_local_oracles", None])
def test_non_shared_plans_decline(
    monkeypatch: pytest.MonkeyPatch,
    plan_mode: str | None,
) -> None:
    owner = _layer_outer_owner(monkeypatch, plan_mode=plan_mode)
    prompts = (tuple(range(16)),)
    chunks = _chunks_for(prompts, row_capacity=8)
    with pytest.raises(NotImplementedError, match="shared-oracle"):
        owner._prefill_batch_native_layer_outer(
            prompts,
            sessions=(SimpleNamespace(position=0),),
            chunks=chunks,
        )


def test_insufficient_hidden_capacity_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _layer_outer_owner(monkeypatch, hidden_capacity=8)
    prompts = (tuple(range(16)),)
    chunks = _chunks_for(prompts, row_capacity=8)
    with pytest.raises(NotImplementedError, match="hidden capacity"):
        owner._prefill_batch_native_layer_outer(
            prompts,
            sessions=(SimpleNamespace(position=0),),
            chunks=chunks,
        )


def test_single_chunk_call_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _layer_outer_owner(monkeypatch)
    prompts = (tuple(range(4)),)
    chunks = _chunks_for(prompts, row_capacity=8)
    assert len(chunks) == 1
    with pytest.raises(ValueError, match="multiple chunks"):
        owner._prefill_batch_native_layer_outer(
            prompts,
            sessions=(SimpleNamespace(position=0),),
            chunks=chunks,
        )


# ---------------------------------------------------------------------------
# Route selection in _prefill_batch_native_impl
# ---------------------------------------------------------------------------


def _dispatch_recorder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decline: bool = False,
) -> tuple[list, list]:
    """Record layer-outer dispatches and single-slab fallbacks."""

    layer_outer_calls: list = []
    slab_calls: list = []

    def fake_layer_outer(self, prompt_token_ids, *, sessions, chunks, **kwargs):
        layer_outer_calls.append(
            (
                tuple(tuple(prompt) for prompt in prompt_token_ids),
                tuple(sessions),
                tuple(chunks),
            )
        )
        if decline:
            raise NotImplementedError("declined for the test")
        return [SimpleNamespace(token_id=7) for _ in sessions]

    def fake_single_slab(self, prompt_token_ids, *, sessions, **kwargs):
        slab_calls.append(tuple(sessions))
        for session in sessions:
            session.position += 1
        return [
            gguf_runner.Qwen35GGUFPackedPrefillResult(
                input_token_ids=[int(token) for token in prompt],
                token_id=1,
                hidden_seeds=np.empty((len(prompt), 4), dtype=np.float32),
                start_position=0,
            )
            for prompt, session in zip(prompt_token_ids, sessions, strict=True)
        ]

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_layer_outer",
        fake_layer_outer,
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_prefill_batch_native_single_slab",
        fake_single_slab,
        raising=False,
    )
    return layer_outer_calls, slab_calls


def _bare_owner(row_capacity: int) -> Qwen35GGUFResidentSession:
    owner = object.__new__(Qwen35GGUFResidentSession)
    owner._bulk_prefill_scratch = SimpleNamespace(rows=int(row_capacity))
    return owner


def test_impl_dispatches_layer_outer_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "1")
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    layer_outer_calls, slab_calls = _dispatch_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = tuple(SimpleNamespace(position=0) for _ in range(2))
    prompts = tuple(tuple(range(8)) for _ in range(2))

    results = owner.prefill_batch_native(prompts, sessions=sessions)

    assert [result.token_id for result in results] == [7, 7]
    assert len(layer_outer_calls) == 1
    assert layer_outer_calls[0][0] == prompts
    assert len(layer_outer_calls[0][2]) == 2
    assert slab_calls == []


def test_impl_falls_back_when_the_executor_declines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "1")
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    layer_outer_calls, slab_calls = _dispatch_recorder(monkeypatch, decline=True)
    owner = _bare_owner(row_capacity=8)
    sessions = tuple(SimpleNamespace(position=0) for _ in range(2))
    prompts = tuple(tuple(range(8)) for _ in range(2))

    results = owner.prefill_batch_native(prompts, sessions=sessions)

    assert len(layer_outer_calls) == 1
    assert len(slab_calls) == 2
    assert [result.token_id for result in results] == [1, 1]


def test_impl_keeps_chunk_outer_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "0")
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    layer_outer_calls, slab_calls = _dispatch_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = tuple(SimpleNamespace(position=0) for _ in range(2))
    prompts = tuple(tuple(range(8)) for _ in range(2))

    results = owner.prefill_batch_native(prompts, sessions=sessions)

    assert layer_outer_calls == []
    assert len(slab_calls) == 2
    assert [result.token_id for result in results] == [1, 1]


def test_impl_skips_layer_outer_for_hidden_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "1")
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    layer_outer_calls, slab_calls = _dispatch_recorder(monkeypatch)
    owner = _bare_owner(row_capacity=8)
    sessions = tuple(SimpleNamespace(position=0) for _ in range(2))
    prompts = tuple(tuple(range(8)) for _ in range(2))

    results = owner.prefill_batch_native(
        prompts, sessions=sessions, return_hidden_seeds=True
    )

    assert layer_outer_calls == []
    assert len(slab_calls) == 2
    assert all(result is not None for result in results)


# ---------------------------------------------------------------------------
# Oracle binding and the feature flag
# ---------------------------------------------------------------------------


def test_executor_resets_per_layer_ownership_before_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor must clear per-layer keying before any layer work runs."""

    class _StopBeforeLayers(Exception):
        pass

    observed: list[list[bool]] = []

    def fake_sync(self, session_tuple, layout, packed_state, **kwargs):
        # This runs after the executor's reset loop and before any layer
        # work, so it observes the binding the layers will actually use.
        observed.append(
            [
                bool(getattr(session, "_int8_prefill_oracle_per_layer", False))
                for session in session_tuple
            ]
        )
        raise _StopBeforeLayers()

    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_sync_packed_decode_initial_state",
        fake_sync,
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_packed_ar_kv_layout_for_sessions",
        lambda self, sessions, **kwargs: SimpleNamespace(
            layer_storage_dtypes=("int8_per_token_head",),
            bf16_mirror_layer_indices=(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_device_kv_contiguous_base_row",
        lambda session: 0,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_ensure_bulk_prefill_workspace",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner.Qwen35GGUFResidentSession,
        "_ensure_packed_verify_workspace",
        lambda self, **kwargs: (
            SimpleNamespace(slot_count=2, blocks_per_slot=1, page_ids=[0, 1]),
            SimpleNamespace(),
        ),
        raising=False,
    )

    owner = _fake_runner_session(monkeypatch, per_layer=True)
    sessions = (
        _fake_runner_session(monkeypatch, per_layer=True),
        _fake_runner_session(monkeypatch, per_layer=True),
    )
    prompts = (tuple(range(8)), tuple(range(8)))
    chunks = _chunks_for(prompts, row_capacity=8)
    # Every session starts with the wrapper's chunk-outer default (True).
    with pytest.raises(_StopBeforeLayers):
        owner._prefill_batch_native_layer_outer(
            prompts, sessions=sessions, chunks=chunks
        )
    # The executor cleared per-layer keying before the layer work.
    assert observed == [[False, False]]
    assert all(
        session._int8_prefill_oracle_per_layer is False for session in sessions
    )


def test_feature_flag_defaults_on_and_rolls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Promoted to default ON 2026-09-10; the env var rolls back to chunk-outer."""

    monkeypatch.delenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", raising=False)
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    assert _gguf_packed_layer_outer_enabled() is True
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "0")
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    assert _gguf_packed_layer_outer_enabled() is False


def test_shared_oracle_binding_records_executor_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan and the realized executor mode must be observable together."""

    owner = _fake_runner_session(monkeypatch, per_layer=True)
    # The binding contract: under the shared-oracle plan the executor clears
    # per-layer keying, so the oracle cache returns one shared pair even
    # though the wrapper's chunk-outer default set the flag before dispatch.
    assert owner._int8_prefill_oracle_per_layer is True
    owner._int8_prefill_oracle_per_layer = False
    pair_a = owner._int8_prefill_oracle_cache_for_layer(3)
    pair_b = owner._int8_prefill_oracle_cache_for_layer(9)
    assert pair_a is pair_b
    assert set(owner._int8_prefill_oracle_buffers) == {-1}


def test_env_flag_cache_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", "0")
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    assert _gguf_packed_layer_outer_enabled() is False
    monkeypatch.delenv("HIPENGINE_GGUF_PACKED_LAYER_OUTER", raising=False)
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    assert _gguf_packed_layer_outer_enabled() is True
