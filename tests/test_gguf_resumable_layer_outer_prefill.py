"""P6 resumable layer-outer INT8 prefill: checkpoint and yield contract.

Roadmap F5 / reviewer packet item 1-2. The compact INT8 ``_prefill_native_chunk``
fallback used to defer all model work to the final scheduler chunk, so a long
prompt blocked admission, cancellation, and interleaved decode for its whole
duration: an internal slab is not a service yield unless control actually
returns to the service driver.

These CPU tests pin the contract of the replacement:

- ``_prefill_batch_native_layer_outer_segment`` runs at most ``layer_budget``
  layers, advances ``next_layer``, and returns the checkpoint instead of
  results while layers remain;
- the ping-pong phase is derived from the segment's start layer, so a resume
  reads the plane the previous segment wrote;
- setup (initial packed-state sync and the embedding launches) runs once, on
  the first segment only - a resume must not re-run it;
- the sampling tail and the packed-decode scatter run only on the segment that
  reaches the last layer, and only that segment returns results;
- the public resumable entry releases the transient BF16 oracle on every
  segment boundary and leaves the session reusable after a mid-segment failure;
- the generator's scheduler-facing budget computation spreads the layers over
  the remaining polls and lets the final chunk finish the remainder.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFResidentSession,
    _GGUFResumablePrefillState,
    _plan_packed_ar_prefill_chunks,
)

from tests.test_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)

_LAYER_TYPES = (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)


@pytest.fixture(autouse=True)
def _reset_layer_outer_flag_cache():
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None
    yield
    gguf_runner._gguf_packed_layer_outer_enabled_cache = None


class _Recorder:
    """Records the layer ids the executor actually ran, per segment."""

    def __init__(self) -> None:
        self.layers: list[int] = []
        self.embeddings = 0
        self.syncs = 0
        self.scatters = 0


def _install_layer_outer_fakes(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _Recorder,
    *,
    layers: tuple[str, ...] = _LAYER_TYPES,
) -> None:
    """Stub the device surface the layer-outer executor touches.

    The real executor's device work is out of scope here (the GPU gates cover
    it); these fakes make the *control flow* - which layers run in which
    segment, when setup runs, when the tail runs - observable on CPU.
    """

    _install_fake_device(monkeypatch)
    monkeypatch.setattr(
        gguf_runner,
        "replace",
        lambda obj, **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        gguf_runner,
        "wmma_prefill_session",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        gguf_runner,
        "gemv_decode_session",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(
        gguf_runner,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_rebind_packed_verify_layout_pages",
        lambda layout, state: layout,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_slot_local_prefill_cache_views",
        lambda session, scratch, **kwargs: scratch,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_slot_local_prefill_allow_aotriton",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "_gguf_device_kv_contiguous_base_row",
        lambda session: 0,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_packed_ar_kv_layout_for_sessions",
        lambda self, sessions, **kwargs: SimpleNamespace(
            layer_storage_dtypes=("int8_per_token_head",) * len(layers),
            bf16_mirror_layer_indices=(),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_ensure_bulk_prefill_workspace",
        lambda self: None,
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_ensure_packed_verify_workspace",
        lambda self, **kwargs: (
            SimpleNamespace(
                slot_count=1,
                blocks_per_slot=1,
                page_ids=[0],
                layer_conv_states=SimpleNamespace(),
                layer_recurrent_states=SimpleNamespace(),
            ),
            SimpleNamespace(
                for_packed_verify_layout=lambda *a, **k: SimpleNamespace(
                    norm=SimpleNamespace(ptr=0x9000)
                )
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_device_token_embedding_weight",
        lambda self, **kwargs: SimpleNamespace(ptr=0x8000),
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_packed_full_kv_row_nbytes",
        lambda self: 0,
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_full_attention_prefill_scratch_for_layer",
        lambda self, scratch, layer_id: scratch,
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_scatter_packed_decode_state",
        lambda self, *a, **k: recorder.__setattr__(
            "scatters", recorder.scatters + 1
        ),
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_enqueue_target_block_rows_from_hidden",
        lambda self, *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_read_target_block_row_tokens",
        lambda self, rows, **kwargs: np.arange(rows, dtype=np.int64),
        raising=False,
    )

    def fake_sync(self, session_tuple, layout, packed_state, **kwargs):
        recorder.syncs += 1

    monkeypatch.setattr(
        Qwen35GGUFResidentSession,
        "_sync_packed_decode_initial_state",
        fake_sync,
        raising=False,
    )
    monkeypatch.setattr(
        gguf_runner,
        "launch_gguf_embedding",
        lambda *a, **k: recorder.__setattr__(
            "embeddings", recorder.embeddings + 1
        ),
    )


def _resumable_owner(
    monkeypatch: pytest.MonkeyPatch,
    recorder: _Recorder,
    *,
    layers: tuple[str, ...] = _LAYER_TYPES,
    rows: int = 8,
) -> Qwen35GGUFResidentSession:
    _install_layer_outer_fakes(monkeypatch, recorder, layers=layers)
    runner = _fake_dense_qwen36_runner()
    runner.weights.config.layer_types = tuple(layers)
    runner.vocab_size = 32
    runner.weights.config.rms_norm_eps = 1e-6
    runner.weights.root = lambda name: SimpleNamespace(
        allocation=lambda: SimpleNamespace(
            tensor=SimpleNamespace(ptr=0x7000)
        )
    )

    # Instance attributes: plain functions, so no ``self`` binding.
    def run_full_attention(layer_id, *args, **kwargs):
        recorder.layers.append(int(layer_id))

    def run_linear(layer_id, *args, **kwargs):
        recorder.layers.append(int(layer_id))

    monkeypatch.setattr(runner, "_run_full_attention_prefill_layer_aotriton", run_full_attention, raising=False)
    monkeypatch.setattr(runner, "_run_linear_attention_prefill_layer_rows", run_linear, raising=False)

    owner = object.__new__(Qwen35GGUFResidentSession)
    owner.__dict__.update(
        runner=runner,
        runtime=SimpleNamespace(
            memcpy_async=lambda *a, **k: None,
            device_synchronize=lambda: None,
            stream_synchronize=lambda stream=0: None,
            free=lambda ptr: None,
        ),
        scratch=SimpleNamespace(
            max_positions=1024,
            cos_table=SimpleNamespace(ptr=0x4000),
            sin_table=SimpleNamespace(ptr=0x5000),
        ),
        position=0,
        _device_kv_allocation=None,
        _int8_prefill_oracle_buffers={},
        _int8_prefill_oracle_per_layer=True,
        _prefill_token_buf=SimpleNamespace(ptr=0x1000),
        _prefill_hidden_a=SimpleNamespace(ptr=0x2000),
        _prefill_hidden_b=SimpleNamespace(ptr=0x3000),
        _int8_prefill_lifetime_plan=SimpleNamespace(
            mode="layer_outer_shared_oracle",
            required_hidden_capacity=1024,
        ),
        last_packed_prefill_plan={"output_norm_rows": 0, "lm_head_sample_rows": 0},
        _bulk_prefill_scratch=SimpleNamespace(
            rows=int(rows),
            for_chunk=lambda *a, **k: SimpleNamespace(retained_key_cache=None),
        ),
        _packed_decode_state_dirty=False,
        use_wmma_prefill=False,
        use_gemv_decode=False,
        _release_int8_prefill_oracle_buffers=lambda: None,
    )
    return owner


def _layers_run(recorder: _Recorder) -> list[int]:
    """Distinct layers, in first-run order (each layer runs once per round)."""

    seen: list[int] = []
    for layer_id in recorder.layers:
        if layer_id not in seen:
            seen.append(layer_id)
    return seen


def _prompt_rounds(prompt: tuple[int, ...], *, rows: int) -> tuple:
    chunks = _plan_packed_ar_prefill_chunks((prompt,), row_capacity=int(rows))
    assert len(chunks) > 1
    return chunks


# ---------------------------------------------------------------------------
# Segment contract
# ---------------------------------------------------------------------------


def test_first_segment_runs_budget_layers_and_returns_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    chunks = _prompt_rounds(prompt, rows=8)

    result = owner._prefill_batch_native_layer_outer(
        (prompt,),
        sessions=(owner,),
        chunks=chunks,
        layer_budget=3,
    )
    assert isinstance(result, _GGUFResumablePrefillState)
    assert result.next_layer == 3
    assert _layers_run(recorder) == [0, 1, 2]
    # Setup ran once: the initial sync and one embedding launch per round.
    assert recorder.syncs == 1
    assert recorder.embeddings == len(chunks)
    # No sampling tail yet.
    assert recorder.scatters == 0
    # The checkpoint exposes the derived phase for the next segment.
    assert result.phase == 1


def test_resume_continues_layers_without_re_running_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    chunks = _prompt_rounds(prompt, rows=8)

    state = owner._prefill_batch_native_layer_outer(
        (prompt,),
        sessions=(owner,),
        chunks=chunks,
        layer_budget=3,
    )
    assert isinstance(state, _GGUFResumablePrefillState)
    recorder.layers.clear()

    resumed = owner._prefill_batch_native_layer_outer(
        None,
        sessions=None,
        chunks=None,
        resume_state=state,
        layer_budget=2,
    )
    assert isinstance(resumed, _GGUFResumablePrefillState)
    assert resumed.next_layer == 5
    assert _layers_run(recorder) == [3, 4]
    # A resume must not re-sync packed state or re-launch embeddings.
    assert recorder.syncs == 1
    assert recorder.embeddings == len(chunks)


def test_final_segment_runs_tail_and_returns_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    chunks = _prompt_rounds(prompt, rows=8)

    state = owner._prefill_batch_native_layer_outer(
        (prompt,),
        sessions=(owner,),
        chunks=chunks,
        layer_budget=3,
    )
    assert isinstance(state, _GGUFResumablePrefillState)
    recorder.layers.clear()

    results = owner._prefill_batch_native_layer_outer(
        None,
        sessions=None,
        chunks=None,
        resume_state=state,
        layer_budget=None,
    )
    assert isinstance(results, list) and len(results) == 1
    assert _layers_run(recorder) == list(range(3, len(_LAYER_TYPES)))
    # The tail ran exactly once, on the completing segment.
    assert recorder.scatters == 1


def test_layer_budget_is_bounded_by_remaining_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget larger than the remaining layers must not overrun the model."""

    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    chunks = _prompt_rounds(prompt, rows=8)

    results = owner._prefill_batch_native_layer_outer(
        (prompt,),
        sessions=(owner,),
        chunks=chunks,
        layer_budget=10_000,
    )
    assert isinstance(results, list)
    assert _layers_run(recorder) == list(range(len(_LAYER_TYPES)))


def test_segment_resume_after_completion_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    prompt = tuple(range(32))
    chunks = _prompt_rounds(prompt, rows=8)

    state = owner._prefill_batch_native_layer_outer(
        (prompt,),
        sessions=(owner,),
        chunks=chunks,
        layer_budget=len(_LAYER_TYPES),
    )
    # A full-length budget completes in one call.
    assert isinstance(state, list)
    assert _layers_run(recorder) == list(range(len(_LAYER_TYPES)))


# ---------------------------------------------------------------------------
# Public entry: guard behaviour
# ---------------------------------------------------------------------------


def test_public_entry_releases_oracle_on_every_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    releases: list[int] = []

    def release() -> None:
        releases.append(1)
        owner._int8_prefill_oracle_buffers = {}

    owner._release_int8_prefill_oracle_buffers = release
    owner._int8_prefill_oracle_buffers = {
        -1: (
            SimpleNamespace(ptr=1, nbytes=2048),
            SimpleNamespace(ptr=2, nbytes=2048),
        )
    }
    prompt = tuple(range(32))

    state = owner.prefill_batch_native_layer_outer_resumable(
        (prompt,),
        sessions=(owner,),
        layer_budget=3,
    )
    assert isinstance(state, _GGUFResumablePrefillState)
    assert len(releases) == 1
    assert owner._int8_prefill_oracle_per_layer is False

    owner._int8_prefill_oracle_buffers = {
        -1: (
            SimpleNamespace(ptr=3, nbytes=2048),
            SimpleNamespace(ptr=4, nbytes=2048),
        )
    }
    results = owner.prefill_batch_native_layer_outer_resumable(
        state=state,
        layer_budget=None,
    )
    assert isinstance(results, list)
    assert len(releases) == 2


def test_public_entry_releases_oracle_after_mid_segment_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    releases: list[int] = []

    class _Boom(Exception):
        pass

    def release() -> None:
        releases.append(1)
        owner._int8_prefill_oracle_buffers = {}

    owner._release_int8_prefill_oracle_buffers = release

    def boom(self, layer_id, *args, **kwargs):
        raise _Boom()

    owner.runner._run_full_attention_prefill_layer_aotriton = boom
    owner._int8_prefill_oracle_buffers = {
        -1: (
            SimpleNamespace(ptr=1, nbytes=2048),
            SimpleNamespace(ptr=2, nbytes=2048),
        )
    }
    prompt = tuple(range(32))

    with pytest.raises(_Boom):
        owner.prefill_batch_native_layer_outer_resumable(
            (prompt,),
            sessions=(owner,),
            layer_budget=3,
        )
    assert len(releases) == 1
    assert owner._int8_prefill_oracle_buffers == {}
    assert owner._int8_prefill_oracle_per_layer is False


def test_public_entry_declines_single_round_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    with pytest.raises(NotImplementedError, match="multi-round"):
        owner.prefill_batch_native_layer_outer_resumable(
            (tuple(range(4)),),
            sessions=(owner,),
            layer_budget=2,
        )


# ---------------------------------------------------------------------------
# Generator wiring: scheduler-facing budget and fallback
# ---------------------------------------------------------------------------


class _WiringRow:
    """Minimal row surface for ``_prefill_resumable_int8_chunk``."""

    def __init__(self, prompt: tuple[int, ...]) -> None:
        self.prompt_ids = tuple(prompt)
        self.prefill_tokens_seen = 0
        self.prefill_ms = 0.0
        self.prefill_chunk_count = 0
        self.resumable_prefill = None
        self.prefix_reused_tokens = 0
        self.lease = SimpleNamespace(session=SimpleNamespace())
        self.slot = None
        self.request = SimpleNamespace()
        self.request_id = 1
        self.native_greedy = True
        self.samples = []
        self.first_token_emitted = False


class _WiringHost:
    """Stands in for the generator around the resumable chunk helper."""

    def __init__(self, owner) -> None:
        self._resident_batch_owner = owner
        self._fallback_reasons: dict[str, int] = {}
        self._route_counts: dict[str, int] = {}
        self.finished: list[object] = []

    def _packed_execution_owner(self, fallback):
        return self._resident_batch_owner

    def _refresh_prefix_cache(self, row):
        return None

    def _finish_native_prefill(self, row, result, *, native_compact_prefill):
        self.finished.append(result)
        row.first_token_emitted = True


class _Counter(dict):
    """Counter-like bookkeeping dict for the wiring host."""

    def __missing__(self, key):
        self[key] = 0
        return 0


def _wiring_host(owner) -> "_WiringHost":
    host = _WiringHost(owner)
    host._fallback_reasons = _Counter()
    host._route_counts = _Counter()
    return host


def test_budget_spreads_layers_over_remaining_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight layers over four polls is two layers per poll."""

    from hipengine.generation import qwen35_gguf as gen

    monkeypatch.setattr(gen, "_gguf_packed_layer_outer_enabled", lambda: True)
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    host = _wiring_host(owner)

    prompt = tuple(range(32))
    row = _WiringRow(prompt)
    budgets: list[int | None] = []

    def fake_resume(prompt_token_ids=None, *, sessions=None, state=None, layer_budget=None, stream=0):
        budgets.append(layer_budget)
        if state is None:
            state = _GGUFResumablePrefillState(
                prompts=(prompt,),
                sessions=(owner,),
                chunk_plans=[],
                packed_state=None,
                packed_scratch_base=None,
                layer_types=tuple(_LAYER_TYPES),
                next_layer=0,
                total_rows=len(prompt),
            )
        if layer_budget is None:
            return [SimpleNamespace(token_id=7)]
        state.next_layer = min(
            len(_LAYER_TYPES), int(state.next_layer) + int(layer_budget)
        )
        if state.next_layer >= len(_LAYER_TYPES):
            return [SimpleNamespace(token_id=7)]
        return state

    owner.prefill_batch_native_layer_outer_resumable = fake_resume

    helper = gen.Qwen35GGUFResidentModelRunner._prefill_resumable_int8_chunk
    # Four chunks of eight tokens: poll 1 has three polls left after it.
    for index, start in enumerate(range(0, 32, 8)):
        row.prefill_tokens_seen = start + 8
        chunk = prompt[start : start + 8]
        final = row.prefill_tokens_seen == len(prompt)
        handled = helper(host, row, chunk, final_chunk=final)
        assert handled is True
        if final:
            break
    # Non-final polls spread the layers; the final poll finishes the rest.
    assert budgets[0] == 2
    assert budgets[-1] is None
    assert host.finished and host.finished[0].token_id == 7
    assert row.resumable_prefill is gen._RESUMABLE_PREFILL_DONE


def test_wiring_declines_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.generation import qwen35_gguf as gen

    monkeypatch.setattr(gen, "_gguf_packed_layer_outer_enabled", lambda: False)
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    host = _wiring_host(owner)

    row = _WiringRow(tuple(range(32)))
    row.prefill_tokens_seen = 8
    helper = gen.Qwen35GGUFResidentModelRunner._prefill_resumable_int8_chunk
    assert helper(host, row, tuple(range(8)), final_chunk=False) is False


def test_wiring_declines_prefix_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    from hipengine.generation import qwen35_gguf as gen

    monkeypatch.setattr(gen, "_gguf_packed_layer_outer_enabled", lambda: True)
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    host = _wiring_host(owner)

    row = _WiringRow(tuple(range(32)))
    row.prefix_reused_tokens = 8
    row.prefill_tokens_seen = 8
    helper = gen.Qwen35GGUFResidentModelRunner._prefill_resumable_int8_chunk
    assert helper(host, row, tuple(range(8)), final_chunk=False) is False


def test_wiring_declines_on_executor_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen35_gguf as gen

    monkeypatch.setattr(gen, "_gguf_packed_layer_outer_enabled", lambda: True)
    recorder = _Recorder()
    owner = _resumable_owner(monkeypatch, recorder)
    host = _wiring_host(owner)

    def decline(*args, **kwargs):
        raise NotImplementedError("slot-stable")

    owner.prefill_batch_native_layer_outer_resumable = decline
    row = _WiringRow(tuple(range(32)))
    row.prefill_tokens_seen = 8
    helper = gen.Qwen35GGUFResidentModelRunner._prefill_resumable_int8_chunk
    assert helper(host, row, tuple(range(8)), final_chunk=False) is False
    assert host._fallback_reasons["resumable_int8_prefill_declined"] == 1


def test_finished_row_ignores_later_scheduler_chunks() -> None:
    """A row that already emitted its first token must no-op, not re-prefill."""

    from hipengine.generation import qwen35_gguf as gen

    row = _WiringRow(tuple(range(32)))
    row.resumable_prefill = gen._RESUMABLE_PREFILL_DONE
    row.slot = SimpleNamespace()  # would trip the "prefilled more than once" guard
    # No exception: the done marker short-circuits before the slot guard.
    gen.Qwen35GGUFResidentModelRunner._prefill_native_chunk(
        SimpleNamespace(), row, tuple(range(8)), final_chunk=False
    )
