"""CPU unit: opt-in full-model AR batch decode inside the Qwen4Exp generator.

Every test drives fake c1 runners and a fake batch runner, so the assertions are
about the wiring contracts and never about device arithmetic:

* one N-row batch call per decode tick, never N c1 ``step`` calls;
* physical slots follow the request, not the scheduler's slot order, and a
  scheduler compaction moves no state;
* the c1 prefill is imported exactly once per request, at its own position;
* cancel/discard/admission-rollback release the pinned slot, and a reused slot
  gets a fresh owner token;
* a poisoned step releases every participating slot, fails the tick, and leaves
  nothing that could silently continue on the c1 path;
* MTP rows never hold a batch slot and never join a batch step.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.dispatch import SlotMove, WorkItem, WorkKind
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_ar_batch import (
    AR_BATCH_ENV,
    AR_BATCH_SLOTS_ENV,
    Qwen4ExpARBatchOwner,
    Qwen4ExpARBatchStepError,
    ar_batch_state_slots,
)
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpResidentServingRunner
from hipengine.generation.registry import GenerationRequest
from tests._qwen4exp_ar_batch_fixtures import (
    FakeBatchRunner,
    FakeC1Runner,
    FakeDraft,
    FakeMTPProvider,
    make_generator,
)


def _request(prompt, *, max_tokens=3, **kwargs):
    return GenerationRequest(
        prompts=(prompt,), max_tokens=max_tokens, temperature=0.0, top_p=1.0,
        ignore_eos=True, **kwargs,
    )


def _register(native, request_id, prompt, *, max_tokens=3, **kwargs):
    native.register_batch(
        (request_id,), _request(prompt, max_tokens=max_tokens, **kwargs),
        prompt_rows=(prompt,),
    )


def _prefill_work(request_id, prompt):
    return WorkItem(
        kind=WorkKind.PREFILL, request_ids=(request_id,),
        row_to_request=(request_id,) * len(prompt), token_rows=(prompt,),
    )


def _prefill(native, request_id, prompt, *, max_tokens=3, **kwargs):
    _register(native, request_id, prompt, max_tokens=max_tokens, **kwargs)
    native.reserve_admission(SimpleNamespace(request_id=request_id))
    native.prefill_batch(_prefill_work(request_id, prompt), commit=True)


def _decode(native, request_ids, *, slot_ids=()):
    return native.decode_batch(
        WorkItem(
            kind=WorkKind.DECODE, request_ids=tuple(request_ids),
            row_to_request=tuple(request_ids),
            token_rows=tuple((0,) for _ in request_ids),
            slot_ids=tuple(slot_ids),
            active_mask=tuple(True for _ in request_ids) if slot_ids else (),
        ),
        commit=True,
    )


@pytest.fixture
def batch_env(monkeypatch):
    monkeypatch.setenv(AR_BATCH_ENV, "1")
    monkeypatch.delenv(AR_BATCH_SLOTS_ENV, raising=False)
    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_batch_runner.Qwen4ExpGGUFBatchRunner",
        FakeBatchRunner,
    )


def _patch_extra_runners(monkeypatch):
    """Only the primary runner is real; the c2 pool grows with fakes."""

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: FakeC1Runner(bias=10),
    )


def _native(monkeypatch):
    _patch_extra_runners(monkeypatch)
    generator = make_generator()
    native = generator.create_resident_model_runner(capacity=2)
    assert isinstance(native, Qwen4ExpResidentServingRunner)
    return generator, native


def _batch(native):
    return native.decode_telemetry()["ar_batch"]


def test_default_off_keeps_the_request_owned_c1_step(monkeypatch) -> None:
    """The committed path is unchanged: no batch owner, one c1 step per row."""

    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        _prefill(native, 2, (3, 4))
        generated = _decode(native, (1, 2))
        assert [item.request_id for item in generated] == [1, 2]
        assert native.decode_telemetry() == {
            "decode_path": "qwen4exp_c1_request_owned_step", "ar_batch": None,
        }
        for request_id in (1, 2):
            row = native._row(request_id)
            assert [name for name, _ in row.runner.calls] == ["prefill_append", "step"]
            assert row.runner.position == len(row.prompt_ids) + 1
    finally:
        native.close()


def test_one_shared_step_per_tick_replaces_n_c1_steps(batch_env, monkeypatch) -> None:
    """Two rows advance in one 2-row call, and no c1 runner is stepped."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        _prefill(native, 2, (3, 4))
        fake = native._ar_batch.runner
        assert _batch(native)["imports"] == 2
        assert fake.import_calls == [(0, 1, 2), (1, 2, 2)]
        first = _decode(native, (1, 2))
        assert [item.request_id for item in first] == [1, 2]
        assert len(fake.step_calls) == 1
        slots, owners, tokens, capture = fake.step_calls[0]
        assert slots == (0, 1)
        assert owners == fake.owners_of(slots)
        assert capture is False
        assert _batch(native)["step_widths"] == {"2": 1}
        # The sampled token is the c1 prefill's argmax; the batch step only
        # produces the next tick's logits, so it shows up on the second tick.
        second = _decode(native, (1, 2))
        assert second[0].token_id == tokens[0] + FakeBatchRunner.token_bias
        assert second[1].token_id == tokens[1] + 2 * FakeBatchRunner.token_bias
        assert len(fake.step_calls) == 2
        assert _batch(native)["step_widths"] == {"2": 2}
        for request_id in (1, 2):
            row = native._row(request_id)
            assert [name for name, _ in row.runner.calls] == ["prefill_append"]
            assert row.runner.position == len(row.prompt_ids)
    finally:
        native.close()


def test_driver_decodes_two_rows_with_one_step_per_tick(batch_env, monkeypatch) -> None:
    """End to end through the scheduler: 3 tokens each, two shared steps."""

    _patch_extra_runners(monkeypatch)
    generator = make_generator()
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(
            max_active_requests=2, max_prefill_chunk_tokens=8,
            # Complete both prefills before decoding so the shared step's width
            # is the batch's, not the TTFT policy's.
            prefill_decode_policy="protect_ttft",
        ),
    )
    try:
        outputs = driver.generate_detailed(
            GenerationRequest(
                prompts=((1, 2, 3), (4, 5)), max_tokens=3, temperature=0.0,
                top_p=1.0, ignore_eos=True,
            )
        )
        native = generator._resident_model_runner
        assert native is not None
        fake = native._ar_batch.runner
        # Tick 1 samples the prefill logits and steps; tick 2 samples and steps;
        # tick 3 samples the final token, which finishes on length.
        assert [call[0] for call in fake.step_calls] == [(0, 1), (0, 1)]
        assert [len(output.generated_token_ids) for output in outputs] == [3, 3]
        assert outputs[0].generated_token_ids != outputs[1].generated_token_ids
        assert native.decode_telemetry()["decode_path"] == "qwen4exp_full_model_ar_batch"
        assert _batch(native)["step_widths"] == {"2": 2}
        assert _batch(native)["imports"] == 2
        # Each response names the physical width its own steps ran at, so a
        # caller cannot read a per-request "1" and conclude it was serial.
        for output in outputs:
            batch = (output.telemetry.diagnostics or {})["decode_batch"]
            assert batch["execution_path"] == "qwen4exp_full_model_ar_batch"
            assert batch["step_widths"] == {"2": 2}
            assert batch["max_width"] == 2
            assert batch["steps"] == 2
            assert output.telemetry.decode_state.execution_path == (
                "qwen4exp_full_model_ar_batch"
            )
        # No c1 runner took a decode step: every c1 position still equals the
        # prompt it prefilled.
        assert sorted(int(runner.position) for runner in native._all_runners) == sorted(
            len(prompt) for prompt in ((1, 2, 3), (4, 5))
        )
    finally:
        driver.close()


def test_chunked_prefill_imports_once_at_the_final_position(batch_env, monkeypatch) -> None:
    """A multi-chunk c1 prefill imports one complete state, never a partial one."""

    generator, native = _native(monkeypatch)
    try:
        _register(native, 1, (1, 2, 3, 4))
        native.reserve_admission(SimpleNamespace(request_id=1))
        fake = native._ar_batch.runner
        native.prefill_batch(
            WorkItem(
                kind=WorkKind.PREFILL, request_ids=(1,),
                row_to_request=(1, 1), token_rows=((1, 2),),
            ),
            commit=True,
        )
        assert fake.import_calls == []
        assert native._ar_batch.owns(1)
        native.prefill_batch(
            WorkItem(
                kind=WorkKind.PREFILL, request_ids=(1,),
                row_to_request=(1, 1), token_rows=((3, 4),),
            ),
            commit=True,
        )
        assert fake.import_calls == [(fake.admit_calls[0], 1, 4)]
        assert native._row(1).next_result is not None
    finally:
        native.close()


def test_physical_slots_follow_the_request_not_the_scheduler_order(batch_env, monkeypatch) -> None:
    """Descending free slots and shuffled scheduler slots keep row identity."""

    generator, native = _native(monkeypatch)
    try:
        native.prepare()
        native._ar_batch.runner.free_slot_order = "descending"
        _prefill(native, 1, (1, 2))
        _prefill(native, 2, (3, 4))
        assert native._ar_batch.lease(1) == (1, 1)
        assert native._ar_batch.lease(2) == (0, 2)
        fake = native._ar_batch.runner
        _decode(native, (1, 2))
        slots, _, tokens, _ = fake.step_calls[0]
        assert slots == (1, 0)
        _decode(native, (2, 1), slot_ids=(0, 1))
        assert fake.step_calls[1][0] == (0, 1)
        # Row 2's state lives in slot 0 and row 1's in slot 1, whatever order
        # the work item names them in.
        assert native._row(2).generated_ids[1] == tokens[1] + FakeBatchRunner.token_bias
        assert native._row(1).generated_ids[1] == tokens[0] + 2 * FakeBatchRunner.token_bias
    finally:
        native.close()


def test_compaction_moves_no_state(batch_env, monkeypatch) -> None:
    """Scheduler slot moves leave the pinned physical slot and owner alone."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        _prefill(native, 2, (3, 4))
        before = {rid: native._ar_batch.lease(rid) for rid in (1, 2)}
        native.compact_batch((SlotMove(request_id=1, old_slot=1, new_slot=0),))
        assert {rid: native._ar_batch.lease(rid) for rid in (1, 2)} == before
        fake = native._ar_batch.runner
        _decode(native, (1, 2), slot_ids=(0, 1))
        assert fake.step_calls[0][0] == tuple(lease[0] for lease in before.values())
        assert fake.step_calls[0][1] == tuple(lease[1] for lease in before.values())
    finally:
        native.close()


def test_discard_releases_the_slot_and_a_reused_slot_gets_a_new_owner(batch_env, monkeypatch) -> None:
    """A cancelled row frees its slot; the old owner token can no longer step."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        _prefill(native, 2, (3, 4))
        fake = native._ar_batch.runner
        slot, owner = native._ar_batch.lease(1)
        native.discard((1,))
        assert native._ar_batch.lease(1) is None
        assert fake.release_calls == [(slot, owner)]
        assert fake.owners_of((slot,)) == (0,)

        _prefill(native, 3, (5, 6))
        reused = native._ar_batch.lease(3)
        assert reused[0] == slot and reused[1] != owner
        with pytest.raises(RuntimeError, match="owned by request"):
            fake.step([7], slots=[slot], owners=[owner])
        _decode(native, (2, 3))
        assert len(fake.step_calls) == 1
    finally:
        native.close()


def test_cancel_between_sampling_and_the_shared_step_publishes_nothing(batch_env, monkeypatch) -> None:
    """A row that cancels while the step runs is not published, and is released."""

    token = GenerationCancellationToken()
    generator, native = _native(monkeypatch)
    try:
        _register(native, 1, (1, 2), cancellation_token=token)
        native.reserve_admission(SimpleNamespace(request_id=1))
        native.prefill_batch(_prefill_work(1, (1, 2)), commit=True)
        fake = native._ar_batch.runner
        slot, owner = native._ar_batch.lease(1)
        original_step = fake.step

        def cancel_then_step(*args, **kwargs):
            token.cancel()
            return original_step(*args, **kwargs)

        fake.step = cancel_then_step
        assert _decode(native, (1,)) == ()
        assert native._row(1).generated_ids == []
        native.discard((1,))
        assert fake.release_calls == [(slot, owner)]
    finally:
        native.close()


def test_admission_rollback_returns_the_pinned_slot(batch_env, monkeypatch) -> None:
    """A failed admission reservation leaves no lease and no lost c1 runner."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        assert native._ar_batch.live_request_ids == (1,)
        available = len(native._available)
        _register(native, 2, (3, 4))
        native.reserve_admission(SimpleNamespace(request_id=2))
        assert native._ar_batch.live_request_ids == (1, 2)
        native.rollback_admission(SimpleNamespace(request_id=2))
        assert native._ar_batch.live_request_ids == (1,)
        assert len(native._available) == available
    finally:
        native.close()


def test_no_free_slot_fails_admission_instead_of_falling_back(batch_env, monkeypatch) -> None:
    """A narrower width refuses admission; it never decodes that row on c1."""

    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, "1")
    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        _register(native, 2, (3, 4))
        with pytest.raises(RuntimeError, match="no free state slot"):
            native.reserve_admission(SimpleNamespace(request_id=2))
        assert native._ar_batch.live_request_ids == (1,)
        assert native._row(2).runner is None
        assert native._available
        _decode(native, (1,))
        assert native._ar_batch.runner.step_calls[0][0] == (native._ar_batch.lease(1)[0],)
    finally:
        native.close()


def test_narrow_width_keeps_the_extra_request_pending(batch_env, monkeypatch) -> None:
    """``plan_admission`` holds the surplus row pending instead of failing a tick."""

    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, "1")
    _patch_extra_runners(monkeypatch)
    generator = make_generator()
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(
            max_active_requests=2, max_prefill_chunk_tokens=8,
            # Complete both prefills before decoding so the shared step's width
            # is the batch's, not the TTFT policy's.
            prefill_decode_policy="protect_ttft",
        ),
    )
    try:
        outputs = driver.generate_detailed(
            GenerationRequest(
                prompts=((1, 2, 3), (4, 5, 6)), max_tokens=2, temperature=0.0,
                top_p=1.0, ignore_eos=True,
            )
        )
        assert [len(output.generated_token_ids) for output in outputs] == [2, 2]
        native = generator._resident_model_runner
        fake = native._ar_batch.runner
        assert fake.state_slots == 1
        assert [len(call[0]) for call in fake.step_calls] == [1, 1]
        assert _batch(native)["step_widths"] == {"1": 2}
    finally:
        driver.close()


def test_poisoned_step_releases_every_slot_and_disables_the_path(batch_env, monkeypatch) -> None:
    """A failed shared step fails the tick and leaves no stale owner to step."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        _prefill(native, 2, (3, 4))
        fake = native._ar_batch.runner
        leases = {rid: native._ar_batch.lease(rid) for rid in (1, 2)}
        fake.fail_next_step = RuntimeError("device state lost")
        with pytest.raises(Qwen4ExpARBatchStepError) as excinfo:
            _decode(native, (1, 2))
        assert excinfo.value.request_ids == (1, 2)
        assert sorted(fake.release_calls) == sorted(leases.values())
        assert native._ar_batch.live_request_ids == ()
        assert _batch(native)["failure"]
        assert fake.step_calls[-1][0] == "failed"
        steps_before = len(fake.step_calls)
        # No stale-owner stepping, and no silent continuation on the c1 path.
        with pytest.raises(RuntimeError, match="lost its AR batch slot"):
            _decode(native, (1,))
        assert len(fake.step_calls) == steps_before
        # A new admission fails closed instead of mixing decode paths, once the
        # engine has reclaimed the failed rows' c1 runners.
        native.discard((1, 2))
        _register(native, 3, (5, 6))
        with pytest.raises(RuntimeError, match="disabled after a failed call"):
            native.reserve_admission(SimpleNamespace(request_id=3))
        assert native._row(3).runner is None
    finally:
        native.close()


def test_failed_import_releases_the_slot_and_fails_closed(batch_env, monkeypatch) -> None:
    """A prefill import that cannot copy stops the row instead of approximating."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2))
        fake = native._ar_batch.runner
        fake.fail_next_import = ValueError("index count disagrees with position")
        with pytest.raises(ValueError, match="index count"):
            _prefill(native, 2, (3, 4))
        assert native._ar_batch.live_request_ids == (1,)
        assert len(fake.release_calls) == 1
        assert _batch(native)["failure"]
        assert _batch(native)["imports"] == 1
    finally:
        native.close()


def test_mtp_row_never_holds_a_batch_slot(batch_env, monkeypatch) -> None:
    """An MTP row stays on its request-owned c1 runner and draft at width 1."""

    _patch_extra_runners(monkeypatch)
    generator = make_generator()
    native = generator.create_resident_model_runner(capacity=1)
    provider = FakeMTPProvider()
    generator._speculative_provider = provider
    native._prepare_mtp()
    try:
        draft = FakeDraft()
        native._available_drafts.append(draft)
        _register(native, 1, (1, 2))
        native._row(1).mtp_budget = 1
        native.reserve_admission(SimpleNamespace(request_id=1))
        assert native._ar_batch.live_request_ids == ()
        native.prefill_batch(_prefill_work(1, (1, 2)), commit=True)
        assert native._ar_batch.live_request_ids == ()
        generated = _decode(native, (1,))
        assert [item.token_id for item in generated] == [provider.root_token]
        assert native._ar_batch.runner.step_calls == []
        assert native._ar_batch.runner.import_calls == []
        assert native._row(1).draft is draft
    finally:
        native.close()


def test_owner_reservation_checks_the_position_bound() -> None:
    """An oversized request is refused before it takes a slot."""

    built = []

    def factory():
        built.append(FakeBatchRunner(state_slots=2, max_positions=8))
        return built[-1]

    owner = Qwen4ExpARBatchOwner(batch_factory=factory, state_slots=2, max_positions=8)
    try:
        with pytest.raises(ValueError, match="needs 9 positions"):
            owner.reserve(1, positions=9)
        assert built == []
        owner.reserve(1, positions=8)
        assert owner.telemetry()["live_slots"] == 1
        with pytest.raises(ValueError, match="already owns an AR batch slot"):
            owner.reserve(1, positions=1)
    finally:
        owner.close()


def test_state_slots_default_to_capacity_and_reject_a_bad_bound(monkeypatch) -> None:
    monkeypatch.delenv(AR_BATCH_SLOTS_ENV, raising=False)
    assert ar_batch_state_slots(2) == 2
    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, "1")
    assert ar_batch_state_slots(2) == 1
    for bad in ("0", "3", "-1"):
        monkeypatch.setenv(AR_BATCH_SLOTS_ENV, bad)
        with pytest.raises(ValueError, match="outside the serving capacity"):
            ar_batch_state_slots(2)


def test_controlled_row_requests_the_logits_readback(batch_env, monkeypatch) -> None:
    """A row with host-side sampling takes the full-logits batch step."""

    generator, native = _native(monkeypatch)
    try:
        _prefill(native, 1, (1, 2), max_tokens=2, logit_bias={5: 50.0})
        generated = _decode(native, (1,))
        fake = native._ar_batch.runner
        assert fake.step_calls[0][3] is True
        assert generated[0].token_id == 5
    finally:
        native.close()


def _standalone_owner(*, state_slots=2, max_positions=8):
    """An owner over a fake batch, without a generator or serving runner."""

    return Qwen4ExpARBatchOwner(
        batch_factory=lambda: FakeBatchRunner(
            state_slots=state_slots, max_positions=max_positions
        ),
        state_slots=state_slots,
        max_positions=max_positions,
    )


def test_close_keeps_the_runner_reachable_until_the_free_succeeds() -> None:
    """A failed free is retryable, and work is refused while it is outstanding."""

    owner = _standalone_owner(state_slots=1)
    try:
        owner.reserve(1, positions=2)
        owner.import_prefilled(1, SimpleNamespace(position=2))
        fake = owner.runner
        fake.fail_next_close = RuntimeError("hipFree failed")
        with pytest.raises(RuntimeError, match="hipFree failed"):
            owner.close()
        # The handle and the leases survive a partial free so close can retry.
        assert owner.runner is fake
        assert fake.closed is False
        assert owner.live_request_ids == (1,)
        assert "close failed" in (owner.failure or "")
        for refused in (
            owner.ensure,
            lambda: owner.reserve(2, positions=1),
            lambda: owner.step([(1, 3)], capture_logits=False),
            lambda: owner.import_prefilled(1, SimpleNamespace(position=2)),
        ):
            with pytest.raises(RuntimeError, match="closing"):
                refused()
        assert fake.step_calls == []
        owner.close()
        assert fake.closed is True
        assert owner.runner is None
        assert owner.live_request_ids == ()
        owner.close()
        assert fake.close_calls == 2
    finally:
        owner.close()


def test_step_requires_an_imported_lease() -> None:
    """A reserved slot with no imported prefill is refused before any step."""

    owner = _standalone_owner(state_slots=1)
    try:
        owner.reserve(1, positions=2)
        fake = owner.runner
        with pytest.raises(RuntimeError, match="no imported prefilled state"):
            owner.step([(1, 5)], capture_logits=False)
        assert fake.step_calls == []
        assert owner.live_request_ids == (1,)
        assert owner.failure is None
    finally:
        owner.close()


def test_repeated_import_is_refused_and_keeps_the_live_history() -> None:
    """A second import cannot silently reset a slot that already has history."""

    owner = _standalone_owner(state_slots=1)
    try:
        owner.reserve(1, positions=2)
        source = SimpleNamespace(position=2)
        assert owner.import_prefilled(1, source) == 2
        assert owner.imported(1) is True
        fake = owner.runner
        with pytest.raises(RuntimeError, match="already imported"):
            owner.import_prefilled(1, source)
        assert fake.import_calls == [(0, 1, 2)]
        assert owner.failure is None
        assert owner.imported(1) is True
        # The imported history is still the slot's, and it can be stepped.
        assert owner.step([(1, 5)], capture_logits=False)[1].token_id == (
            5 + FakeBatchRunner.token_bias
        )
        assert owner.telemetry()["imports"] == 1
    finally:
        owner.close()


@pytest.mark.parametrize("bad", [3.7, True, "5", None, np.float32(2.0)])
def test_step_rejects_a_non_integer_token_before_touching_the_slot(bad) -> None:
    """A float or bool token is refused instead of being truncated to an int."""

    owner = _standalone_owner(state_slots=1)
    try:
        owner.reserve(1, positions=2)
        owner.import_prefilled(1, SimpleNamespace(position=2))
        fake = owner.runner
        with pytest.raises(ValueError, match="token id"):
            owner.step([(1, bad)], capture_logits=False)
        with pytest.raises(ValueError, match="request id"):
            owner.step([(bad, 5)], capture_logits=False)
        assert fake.step_calls == []
        assert owner.failure is None
        assert owner.imported(1) is True
        decoded = owner.step([(1, 5)], capture_logits=False)
        assert decoded[1].token_id == 5 + FakeBatchRunner.token_bias
    finally:
        owner.close()


@pytest.mark.parametrize(
    "mode", ["short_tokens", "wrong_logits_rows", "no_logits", "nan_logits"]
)
def test_malformed_step_output_fails_the_rows_and_blocks_a_retry(mode) -> None:
    """A step that advances state but returns an unusable row set fails closed."""

    owner = _standalone_owner()
    try:
        for request_id, position in ((1, 2), (2, 3)):
            owner.reserve(request_id, positions=position)
            owner.import_prefilled(request_id, SimpleNamespace(position=position))
        fake = owner.runner
        fake.malformed = mode
        with pytest.raises(Qwen4ExpARBatchStepError) as excinfo:
            owner.step([(1, 5), (2, 6)], capture_logits=True)
        assert excinfo.value.request_ids == (1, 2)
        assert owner.live_request_ids == ()
        assert owner.failed(1) and owner.failed(2)
        assert "malformed result" in (owner.failure or "")
        assert len(fake.step_calls) == 1
        # The rows cannot be re-stepped or re-attributed.
        with pytest.raises(RuntimeError, match="disabled after a failed call"):
            owner.step([(1, 5)], capture_logits=True)
        assert len(fake.step_calls) == 1
    finally:
        owner.close()
