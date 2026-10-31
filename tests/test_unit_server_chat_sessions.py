"""Application-state contract for the server chat-session owner.

The store is a pure state owner: it receives explicit policy values and
per-operation data, and raises typed ``SessionStateError`` values instead of
building HTTP responses. These checks pin reservation, transcript commit,
continuation scope, cap asymmetry and deep-copy boundaries.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from hipengine.server.chat_sessions import (
    ChatSessionRecord,
    ChatSessionStore,
    ContinuationRecord,
    SessionStateError,
)


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _run(coro):
    return asyncio.run(coro)


def _store(
    *,
    max_sessions: int | None = None,
    ttl: int = 900,
    retry_after: int = 1,
    clock: _Clock | None = None,
) -> tuple[ChatSessionStore, _Clock, list[str]]:
    ids: list[str] = []

    def next_id() -> str:
        ids.append(f"gen_test_{len(ids) + 1}")
        return ids[-1]

    current = clock or _Clock()
    return (
        ChatSessionStore(
            max_sessions=max_sessions,
            continuation_ttl_seconds=ttl,
            retry_after_seconds=retry_after,
            clock=current,
            continuation_id_factory=next_id,
        ),
        current,
        ids,
    )


_UNSET = object()


def _assert_error(
    error: SessionStateError,
    *,
    status: int,
    code: str,
    param: str | None | object = _UNSET,
    message: str | None = None,
) -> None:
    assert error.status_code == status
    assert error.code == code
    if param is not _UNSET:
        assert error.param == param
    if message is not None:
        assert error.message == message


def test_reserve_pending_cap_release_and_bypasses() -> None:
    store, _, _ = _store(max_sessions=1)

    assert _run(
        store.reserve(
            None,
            cache_action="append_visible_only",
            continuation_id=None,
            requested_model="fake-model",
        )
    ) is None
    assert _run(
        store.reserve(
            "sess",
            cache_action="append_none",
            continuation_id=None,
            requested_model="fake-model",
        )
    ) is None
    assert _run(
        store.reserve(
            "sess",
            cache_action="append_visible_only",
            continuation_id="gen_existing",
            requested_model="fake-model",
        )
    ) is None

    assert (
        _run(
            store.reserve(
                "sess",
                cache_action="append_visible_only",
                continuation_id=None,
                requested_model="fake-model",
            )
        )
        == "sess"
    )
    assert store.pending == {"sess"}

    with pytest.raises(SessionStateError) as pending:
        _run(
            store.reserve(
                "sess",
                cache_action="append_visible_only",
                continuation_id=None,
                requested_model="fake-model",
            )
        )
    _assert_error(
        pending.value,
        status=429,
        code="engine_busy",
        message="chat session is being created",
    )
    assert pending.value.error_type == "rate_limit_error"
    assert pending.value.details == {"overload_source": "chat_session_pending"}
    assert pending.value.headers == {"Retry-After": "1"}
    assert pending.value.requested_model == "fake-model"
    assert pending.value.record_rejection is True

    _run(store.release("sess"))
    assert store.pending == set()

    assert (
        _run(
            store.reserve(
                "sess",
                cache_action="append_visible_only",
                continuation_id=None,
                requested_model="fake-model",
            )
        )
        == "sess"
    )
    with pytest.raises(SessionStateError) as capped:
        _run(
            store.reserve(
                "other",
                cache_action="append_visible_only",
                continuation_id=None,
                requested_model="fake-model",
            )
        )
    _assert_error(
        capped.value,
        status=429,
        code="engine_busy",
        message="chat session limit is full",
    )
    assert capped.value.details == {
        "overload_source": "chat_session_cap",
        "max_active_chat_sessions": 1,
    }
    assert capped.value.headers == {"Retry-After": "1"}
    assert capped.value.requested_model == "fake-model"
    assert capped.value.record_rejection is True

    _run(store.release(None))
    _run(store.release("missing"))


def test_commit_modes_reset_base_and_deep_copy_boundaries() -> None:
    store, clock, _ = _store()

    prompt = {"role": "user", "content": "hello", "nested": {"value": 1}}
    visible = {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [{"id": "call_1", "type": "function"}],
    }
    _run(
        store.commit(
            "sess",
            cache_action="append_visible_only",
            request_messages=(prompt,),
            raw_output="raw answer",
            visible_message=visible,
        )
    )
    record = store.sessions["sess"]
    assert record.messages == (
        {"role": "user", "content": "hello", "nested": {"value": 1}},
        {
            "role": "assistant",
            "content": "answer",
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
    )
    prompt["nested"]["value"] = 99
    visible["tool_calls"][0]["id"] = "mutated"
    assert record.messages[0]["nested"]["value"] == 1
    assert record.messages[1]["tool_calls"][0]["id"] == "call_1"

    clock.now = 1001.0
    _run(
        store.commit(
            "sess",
            cache_action="append_all",
            request_messages=({"role": "user", "content": "again"},),
            raw_output="raw again",
            visible_message={"role": "assistant", "content": "ignored"},
        )
    )
    record = store.sessions["sess"]
    assert record.messages[-1] == {"role": "assistant", "content": "raw again"}
    assert record.created == 1000.0
    assert record.updated == 1001.0

    clock.now = 1002.0
    _run(
        store.commit(
            "sess",
            cache_action="append_prompt_only",
            request_messages=({"role": "user", "content": "base turn"},),
            raw_output="",
            visible_message={"role": "assistant", "content": ""},
            reset_session=True,
            commit_base_messages=({"role": "system", "content": "base"},),
        )
    )
    record = store.sessions["sess"]
    assert record.messages == (
        {"role": "system", "content": "base"},
        {"role": "user", "content": "base turn"},
    )
    assert record.created == 1002.0
    assert record.updated == 1002.0

    _run(
        store.commit(
            None,
            cache_action="append_all",
            request_messages=(),
            raw_output="",
            visible_message={},
        )
    )
    _run(
        store.commit(
            "sess",
            cache_action="append_none",
            request_messages=(),
            raw_output="",
            visible_message={},
        )
    )
    _run(
        store.commit(
            "sess",
            cache_action="unknown",
            request_messages=(),
            raw_output="",
            visible_message={},
        )
    )
    assert store.sessions["sess"].messages == record.messages


def test_prefix_messages_copies_and_missing_session_is_empty() -> None:
    store, _, _ = _store()
    _run(
        store.commit(
            "sess",
            cache_action="append_all",
            request_messages=({"role": "user", "content": "hello", "nested": {"x": 1}},),
            raw_output="answer",
            visible_message={},
        )
    )
    prefix = _run(store.prefix_messages("sess"))
    assert prefix[0]["nested"] == {"x": 1}
    prefix[0]["nested"]["x"] = 2
    assert store.sessions["sess"].messages[0]["nested"]["x"] == 1
    assert _run(store.prefix_messages(None)) == ()
    assert _run(store.prefix_messages("missing")) == ()


def test_continuation_scope_expiry_consume_once_and_provider_order() -> None:
    store, clock, ids = _store(ttl=10)
    record = _run(
        store.store_continuation(
            endpoint="chat",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id="sess",
            tokenizer_metadata={"name": "Tokenizer", "tokenize": True},
            prompts=("prompt",),
            generated_texts=("generated",),
            response_format={"type": "json_object"},
            guided_json={"type": "object"},
            guided_regex=None,
            guided_choice=None,
            guided_patch=None,
            guided_diff=None,
        )
    )
    assert isinstance(record, ContinuationRecord)
    assert record.id == "gen_test_1"
    assert ids == ["gen_test_1"]
    assert record.expires_at == 1010.0
    assert record.resume_prompts() == ("promptgenerated",)
    _run(
        store.put_if_capacity(
            ChatSessionRecord(id="sess", messages=(), created=0.0, updated=0.0)
        )
    )

    calls: list[int] = []

    def tokenizer() -> Mapping[str, Any] | None:
        calls.append(1)
        return {"name": "OtherTokenizer", "tokenize": True}

    with pytest.raises(SessionStateError) as wrong_endpoint:
        _run(
            store.pop_continuation(
                "gen_test_1",
                endpoint="completion",
                model_id="fake-model",
                auth_principal="anonymous",
                session_id="sess",
                current_tokenizer_metadata=tokenizer,
            )
        )
    _assert_error(wrong_endpoint.value, status=400, code="invalid_continuation")
    assert calls == []
    assert "gen_test_1" in store.continuations

    for kwargs, message in (
        ({"model_id": "other-model"}, "different served model"),
        ({"auth_principal": "other"}, "different auth principal"),
        ({"session_id": "other"}, "different session"),
    ):
        scope = {
            "endpoint": "chat",
            "model_id": "fake-model",
            "auth_principal": "anonymous",
            "session_id": "sess",
        }
        scope.update(kwargs)
        with pytest.raises(SessionStateError) as scoped:
            _run(
                store.pop_continuation(
                    "gen_test_1",
                    current_tokenizer_metadata=lambda: None,
                    **scope,
                )
            )
        _assert_error(scoped.value, status=400, code="invalid_continuation")
        assert message in scoped.value.message
        assert "gen_test_1" in store.continuations

    with pytest.raises(SessionStateError) as tokenizer_error:
        _run(
            store.pop_continuation(
                "gen_test_1",
                endpoint="chat",
                model_id="fake-model",
                auth_principal="anonymous",
                session_id="sess",
                current_tokenizer_metadata=tokenizer,
            )
        )
    _assert_error(tokenizer_error.value, status=400, code="invalid_continuation")
    assert "different tokenizer" in tokenizer_error.value.message
    assert calls == [1]
    assert "gen_test_1" in store.continuations

    popped = _run(
        store.pop_continuation(
            "gen_test_1",
            endpoint="chat",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id="sess",
            current_tokenizer_metadata=lambda: {"name": "Tokenizer", "tokenize": True},
        )
    )
    assert popped is not None and popped.id == "gen_test_1"
    assert "gen_test_1" not in store.continuations

    with pytest.raises(SessionStateError) as consumed:
        _run(
            store.pop_continuation(
                "gen_test_1",
                endpoint="chat",
                model_id="fake-model",
                auth_principal="anonymous",
                session_id="sess",
                current_tokenizer_metadata=lambda: None,
            )
        )
    _assert_error(consumed.value, status=400, code="invalid_continuation")

    _run(
        store.store_continuation(
            endpoint="completion",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id=None,
            tokenizer_metadata={"name": "Tokenizer", "tokenize": True},
            prompts=("prompt",),
            generated_texts=("generated",),
            response_format=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            guided_patch=None,
            guided_diff=None,
        )
    )
    clock.now = 1011.0
    with pytest.raises(SessionStateError) as expired:
        _run(
            store.pop_continuation(
                "gen_test_2",
                endpoint="completion",
                model_id="fake-model",
                auth_principal="anonymous",
                session_id=None,
                current_tokenizer_metadata=lambda: None,
            )
        )
    _assert_error(expired.value, status=410, code="continuation_expired")
    assert "gen_test_2" not in store.continuations

    assert _run(
        store.pop_continuation(
            None,
            endpoint="completion",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id=None,
            current_tokenizer_metadata=lambda: None,
        )
    ) is None
    with pytest.raises(SessionStateError) as empty:
        _run(
            store.pop_continuation(
                "   ",
                endpoint="completion",
                model_id="fake-model",
                auth_principal="anonymous",
                session_id=None,
                current_tokenizer_metadata=lambda: None,
            )
        )
    _assert_error(empty.value, status=400, code="invalid_request", param="continuation_id")


def test_continuation_chat_requires_live_session_but_completion_does_not() -> None:
    store, _, _ = _store()
    _run(
        store.store_continuation(
            endpoint="chat",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id="deleted",
            tokenizer_metadata={"name": "Tokenizer"},
            prompts=("prompt",),
            generated_texts=("generated",),
            response_format=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            guided_patch=None,
            guided_diff=None,
        )
    )
    with pytest.raises(SessionStateError) as missing:
        _run(
            store.pop_continuation(
                "gen_test_1",
                endpoint="chat",
                model_id="fake-model",
                auth_principal="anonymous",
                session_id="deleted",
                current_tokenizer_metadata=lambda: None,
            )
        )
    _assert_error(missing.value, status=400, code="invalid_continuation")
    assert "deleted session" in missing.value.message
    assert "gen_test_1" in store.continuations

    _run(
        store.store_continuation(
            endpoint="completion",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id="deleted",
            tokenizer_metadata={"name": "Tokenizer"},
            prompts=("prompt",),
            generated_texts=("generated",),
            response_format=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            guided_patch=None,
            guided_diff=None,
        )
    )
    popped = _run(
        store.pop_continuation(
            "gen_test_2",
            endpoint="completion",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id="deleted",
            current_tokenizer_metadata=lambda: None,
        )
    )
    assert popped is not None


def test_fork_rollback_delete_listing_and_summary() -> None:
    store, clock, _ = _store(max_sessions=2)
    _run(
        store.commit(
            "source",
            cache_action="append_all",
            request_messages=(
                {"role": "user", "content": "one", "nested": {"x": 1}},
                {"role": "user", "content": "two"},
            ),
            raw_output="answer",
            visible_message={},
        )
    )
    clock.now = 1001.0
    forked = _run(store.fork("source", "branch"))
    assert forked.id == "branch"
    assert forked.created == 1001.0
    assert forked.updated == 1001.0
    assert forked.messages[0]["nested"] == {"x": 1}
    forked.messages[0]["nested"]["x"] = 2
    assert store.sessions["source"].messages[0]["nested"]["x"] == 1

    with pytest.raises(SessionStateError) as missing_source:
        _run(store.fork("missing", "other"))
    _assert_error(missing_source.value, status=404, code="invalid_request", param="session_id")

    with pytest.raises(SessionStateError) as existing_target:
        _run(store.fork("source", "branch"))
    _assert_error(existing_target.value, status=400, code="invalid_request", param="id")

    with pytest.raises(SessionStateError) as fork_cap:
        _run(store.fork("source", "third"))
    _assert_error(fork_cap.value, status=429, code="engine_busy")
    assert fork_cap.value.details == {
        "overload_source": "chat_session_cap",
        "max_active_chat_sessions": 2,
    }
    assert fork_cap.value.record_rejection is True

    clock.now = 1002.0
    result = _run(store.rollback("source", 1))
    assert result.previous_message_count == 3
    assert result.rolled_back is True
    assert result.record.messages == ({"role": "user", "content": "one", "nested": {"x": 1}},)
    assert result.record.created == 1000.0
    assert result.record.updated == 1002.0

    unchanged = _run(store.rollback("source", 1))
    assert unchanged.previous_message_count == 1
    assert unchanged.rolled_back is False
    assert unchanged.record.updated == 1002.0

    with pytest.raises(SessionStateError) as rollback_missing:
        _run(store.rollback("missing", 0))
    _assert_error(rollback_missing.value, status=404, code="invalid_request", param="session_id")

    with pytest.raises(SessionStateError) as rollback_range:
        _run(store.rollback("source", 2))
    _assert_error(rollback_range.value, status=400, code="invalid_request", param="message_count")

    _run(
        store.store_continuation(
            endpoint="completion",
            model_id="fake-model",
            auth_principal="anonymous",
            session_id=None,
            tokenizer_metadata={"name": "Tokenizer"},
            prompts=("prompt",),
            generated_texts=("generated",),
            response_format=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            guided_patch=None,
            guided_diff=None,
        )
    )
    listing = _run(store.listing())
    assert [record.id for record in listing.records] == ["source", "branch"]
    assert listing.active_continuations == 1
    assert listing.pending_creations == 0
    assert _run(store.listing()).active_continuations == 1

    summary = store.summary()
    assert summary == {
        "resident_context": True,
        "active": 2,
        "pending_creations": 0,
        "max_active": 2,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "total_messages": 4,
        "continuations": {"active": 1, "ttl_seconds": 900},
    }

    assert _run(store.delete("branch")) is True
    assert _run(store.delete("branch")) is False
    assert set(store.sessions) == {"source"}


def test_put_if_capacity_preserves_existing_and_rejects_new() -> None:
    store, _, _ = _store(max_sessions=1)
    first = ChatSessionRecord(
        id="first",
        messages=({"role": "user", "content": "one"},),
        created=1.0,
        updated=1.0,
    )
    second = ChatSessionRecord(
        id="second",
        messages=({"role": "user", "content": "two"},),
        created=2.0,
        updated=2.0,
    )
    assert _run(store.put_if_capacity(first)) is first
    assert _run(store.put_if_capacity(first)) is first

    with pytest.raises(SessionStateError) as capped:
        _run(store.put_if_capacity(second))
    _assert_error(capped.value, status=429, code="engine_busy")
    assert capped.value.details == {
        "overload_source": "chat_session_cap",
        "max_active_chat_sessions": 1,
    }
    assert capped.value.record_rejection is True
    assert set(store.sessions) == {"first"}



def test_snapshot_record_lease_blocks_delete_and_commit_until_payload_exits() -> None:
    store, _, _ = _store()
    _run(
        store.commit(
            "sess",
            cache_action="append_all",
            request_messages=({"role": "user", "content": "one"},),
            raw_output="answer",
            visible_message={},
        )
    )
    original = store.sessions["sess"]

    async def scenario() -> None:
        async with store.snapshot_record("sess") as record:
            assert record is original
            delete_task = asyncio.create_task(store.delete("sess"))
            commit_task = asyncio.create_task(
                store.commit(
                    "sess",
                    cache_action="append_all",
                    request_messages=({"role": "user", "content": "two"},),
                    raw_output="next",
                    visible_message={},
                )
            )
            await asyncio.sleep(0)
            assert not delete_task.done()
            assert not commit_task.done()
            # Stand in for tokenizer and payload construction while the lease
            # is still held.
            await asyncio.sleep(0)
        assert await delete_task is True
        await commit_task

    _run(scenario())


def test_snapshot_record_lease_failure_releases_lock_and_preserves_transcript() -> None:
    store, _, _ = _store()
    _run(
        store.commit(
            "sess",
            cache_action="append_all",
            request_messages=({"role": "user", "content": "one"},),
            raw_output="answer",
            visible_message={},
        )
    )
    original = store.sessions["sess"]

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="payload failed"):
            async with store.snapshot_record("sess") as record:
                assert record is original
                delete_task = asyncio.create_task(store.delete("sess"))
                commit_task = asyncio.create_task(
                    store.commit(
                        "sess",
                        cache_action="append_all",
                        request_messages=({"role": "user", "content": "two"},),
                        raw_output="next",
                        visible_message={},
                    )
                )
                await asyncio.sleep(0)
                assert not delete_task.done()
                assert not commit_task.done()
                raise RuntimeError("payload failed")

        # The lease released synchronously on the exception; the blocked tasks
        # have not run yet, so the transcript is still the original record.
        assert store.sessions["sess"] is original
        assert store.sessions["sess"].messages == original.messages
        assert await delete_task is True
        await commit_task

    _run(scenario())
