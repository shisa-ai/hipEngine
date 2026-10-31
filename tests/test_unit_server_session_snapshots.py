"""Contract for the pure session-snapshot owner and its route boundary.

The snapshot envelope and transcript validation stay outside the HTTP
application. These checks pin validation order, typed errors, the narrow
tokenizer-metadata operation, and the lease timing the GET route must preserve.
"""

from __future__ import annotations

import ast
import inspect
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from hipengine.server.chat_sessions import (
    ChatSessionRecord,
    ChatSessionStore,
    SessionStateError,
)
from hipengine.server.session_snapshot_routes import create_session_snapshot_router
from hipengine.server.session_snapshots import (
    CHAT_SESSION_SNAPSHOT_SCHEMA,
    SessionSnapshotModel,
    build_chat_session_snapshot,
    restore_chat_session_from_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_OWNER = REPO_ROOT / "hipengine" / "server" / "session_snapshots.py"
SNAPSHOT_ROUTES = REPO_ROOT / "hipengine" / "server" / "session_snapshot_routes.py"

_MODEL = SessionSnapshotModel(
    id="fake-model",
    backend="test-backend",
    quant="test-quant",
)
_TOKENIZER_METADATA = {
    "name": "FakeTokenizer",
    "tokenize": True,
    "detokenize": True,
    "count_tokens": True,
}


def _metadata() -> dict[str, Any]:
    return dict(_TOKENIZER_METADATA)


def _snapshot() -> dict[str, Any]:
    return {
        "object": "hipengine.session.snapshot",
        "schema": CHAT_SESSION_SNAPSHOT_SCHEMA,
        "model": {
            "id": "fake-model",
            "backend": "test-backend",
            "quant": "test-quant",
        },
        "tokenizer": dict(_TOKENIZER_METADATA),
        "session": {
            "id": "sess",
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "includes_transcript": True,
            "created": 10,
            "updated": 20,
            "message_count": 1,
        },
        "resident_state_reuse": False,
        "messages": [{"role": "user", "content": "hi"}],
    }


def _restore(
    snapshot: Mapping[str, Any],
    *,
    current_tokenizer_metadata=_metadata,
) -> ChatSessionRecord:
    return restore_chat_session_from_snapshot(
        "sess",
        snapshot,
        schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
        model=_MODEL,
        current_tokenizer_metadata=current_tokenizer_metadata,
    )


def _corrupt(snapshot: dict[str, Any], case: str) -> None:
    if case == "object":
        snapshot["object"] = "hipengine.session.other"
        snapshot["schema"] = "bad-schema"
    elif case == "schema":
        snapshot["schema"] = "bad-schema"
    elif case == "resident_state_reuse":
        snapshot["resident_state_reuse"] = True
    elif case == "model":
        snapshot["model"] = "not-an-object"
    elif case == "model_quant":
        snapshot["model"]["quant"] = "other-quant"
        snapshot["tokenizer"] = "not-an-object"
    elif case == "tokenizer":
        snapshot["tokenizer"] = "not-an-object"
    elif case == "tokenizer_name":
        snapshot["tokenizer"]["name"] = "OtherTokenizer"
        snapshot["session"] = "not-an-object"
    elif case == "session":
        snapshot["session"] = "not-an-object"
    elif case == "session_storage":
        snapshot["session"]["storage"] = "other-storage"
        snapshot["messages"] = "not-an-array"
    elif case == "session_resident_state_reuse":
        snapshot["session"]["resident_state_reuse"] = True
    elif case == "session_includes_transcript":
        snapshot["session"]["includes_transcript"] = False
    elif case == "session_id":
        snapshot["session"]["id"] = "other-session"
    elif case == "messages":
        snapshot["messages"] = "not-an-array"
        snapshot["session"]["created"] = True
    elif case == "message_not_object":
        snapshot["messages"] = [7]
    elif case == "message_extra":
        snapshot["messages"][0]["unexpected"] = True
    elif case == "message_role":
        snapshot["messages"][0]["role"] = "critic"
    elif case == "message_content":
        snapshot["messages"][0]["content"] = {"text": "not an accepted content shape"}
    elif case == "created":
        snapshot["session"]["created"] = True
        snapshot["session"]["updated"] = -1
    elif case == "updated":
        snapshot["session"]["updated"] = -1
    else:
        raise AssertionError(f"unhandled corruption case: {case}")


@pytest.mark.parametrize(
    ("case", "param", "message"),
    [
        ("object", "object", "session snapshot object must be hipengine.session.snapshot"),
        (
            "schema",
            "schema",
            "session snapshot schema must be 'hipengine.chat_session_snapshot.v1'",
        ),
        (
            "resident_state_reuse",
            "resident_state_reuse",
            "session snapshot resident_state_reuse must be false",
        ),
        ("model", "model", "session snapshot model must be an object"),
        (
            "model_quant",
            "model.quant",
            "session snapshot model.quant is incompatible with this server",
        ),
        ("tokenizer", "tokenizer", "session snapshot tokenizer must be an object"),
        (
            "tokenizer_name",
            "tokenizer.name",
            "session snapshot tokenizer.name is incompatible with this server",
        ),
        ("session", "session", "session snapshot session must be an object"),
        (
            "session_storage",
            "session.storage",
            "session snapshot storage must be app_local_transcript",
        ),
        (
            "session_resident_state_reuse",
            "session.resident_state_reuse",
            "session snapshot resident_state_reuse must be false",
        ),
        (
            "session_includes_transcript",
            "session.includes_transcript",
            "session snapshot includes_transcript must be true",
        ),
        (
            "session_id",
            "session.id",
            "session snapshot id must match the restore path",
        ),
        ("messages", "messages", "session snapshot messages must be an array"),
        (
            "message_not_object",
            "messages[0]",
            "session snapshot messages[0] must be an object",
        ),
        (
            "message_extra",
            "messages[0].unexpected",
            "session snapshot messages[0].unexpected is not supported",
        ),
        (
            "message_role",
            "messages[0].role",
            "messages[0].role must be one of: assistant, developer, system, tool, user",
        ),
        (
            "message_content",
            "messages[0].content",
            "session snapshot messages[0].content must be text, null, or text content parts",
        ),
        (
            "created",
            "session.created",
            "session snapshot session.created must be a non-negative timestamp",
        ),
        (
            "updated",
            "session.updated",
            "session snapshot session.updated must be a non-negative timestamp",
        ),
    ],
)
def test_restore_preserves_validation_order_and_typed_errors(
    case: str,
    param: str,
    message: str,
) -> None:
    snapshot = _snapshot()
    _corrupt(snapshot, case)

    with pytest.raises(SessionStateError) as caught:
        _restore(snapshot)

    error = caught.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert error.param == param
    assert error.message == message


def test_restore_skips_tokenizer_metadata_when_snapshot_has_no_tokenizer() -> None:
    snapshot = _snapshot()
    snapshot["tokenizer"] = None
    calls: list[str] = []

    record = _restore(
        snapshot,
        current_tokenizer_metadata=lambda: calls.append("read") or _metadata(),
    )

    assert calls == []
    assert record.messages == ({"role": "user", "content": "hi"},)


def test_restore_reads_tokenizer_metadata_after_model_and_before_session() -> None:
    snapshot = _snapshot()
    snapshot["session"]["storage"] = "other-storage"
    events: list[str] = []

    def current_metadata() -> dict[str, Any]:
        events.append("tokenizer")
        return _metadata()

    with pytest.raises(SessionStateError) as caught:
        _restore(snapshot, current_tokenizer_metadata=current_metadata)

    assert events == ["tokenizer"]
    assert caught.value.param == "session.storage"


def test_restore_allows_mismatched_tokenizer_when_no_model_is_loaded() -> None:
    snapshot = _snapshot()
    snapshot["tokenizer"]["name"] = "OtherTokenizer"

    record = _restore(snapshot, current_tokenizer_metadata=lambda: None)

    assert record.messages == ({"role": "user", "content": "hi"},)


def test_build_snapshot_uses_explicit_model_metadata_and_deep_copies_messages() -> None:
    record = ChatSessionRecord(
        id="sess",
        messages=({"role": "user", "content": "hi"},),
        created=10.0,
        updated=20.0,
    )
    calls: list[str] = []

    payload = build_chat_session_snapshot(
        record,
        schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
        model=_MODEL,
        tokenizer_metadata=lambda: calls.append("read") or _metadata(),
    )

    assert calls == ["read"]
    assert payload == {
        "object": "hipengine.session.snapshot",
        "schema": CHAT_SESSION_SNAPSHOT_SCHEMA,
        "model": {
            "id": "fake-model",
            "backend": "test-backend",
            "quant": "test-quant",
        },
        "tokenizer": _TOKENIZER_METADATA,
        "session": {
            "id": "sess",
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "message_count": 1,
            "created": 10,
            "updated": 20,
            "includes_transcript": True,
        },
        "resident_state_reuse": False,
        "messages": [{"role": "user", "content": "hi"}],
    }
    payload["messages"][0]["content"] = "changed"
    assert record.messages[0]["content"] == "hi"


class _LeaseObservingStore(ChatSessionStore):
    def __init__(self) -> None:
        super().__init__(
            max_sessions=None,
            continuation_ttl_seconds=900,
            retry_after_seconds=1,
        )
        self.lease_active = False
        self.sessions["sess"] = ChatSessionRecord(
            id="sess",
            messages=({"role": "user", "content": "hi"},),
            created=10.0,
            updated=20.0,
        )

    @asynccontextmanager
    async def snapshot_record(self, session_id: str):
        self.lease_active = True
        try:
            async with super().snapshot_record(session_id) as record:
                yield record
        finally:
            self.lease_active = False


def test_snapshot_route_reads_tokenizer_metadata_inside_session_lease() -> None:
    store = _LeaseObservingStore()
    observed: list[bool] = []

    def snapshot_tokenizer_metadata() -> dict[str, Any]:
        observed.append(store.lease_active)
        return _metadata()

    async def auth() -> str:
        return "test"

    def translate(exc: SessionStateError) -> Exception:
        return HTTPException(status_code=exc.status_code, detail=exc.message)

    app = FastAPI()
    app.include_router(
        create_session_snapshot_router(
            store=store,
            auth_dependency=auth,
            error_translator=translate,
            model=_MODEL,
            schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
            snapshot_tokenizer_metadata=snapshot_tokenizer_metadata,
            current_tokenizer_metadata=lambda: None,
        )
    )

    response = TestClient(app).get("/v1/hipengine/sessions/sess/snapshot")

    assert response.status_code == 200
    assert observed == [True]
    assert store.lease_active is False


def test_snapshot_route_restore_uses_the_explicit_store_and_metadata_operation() -> None:
    store = _LeaseObservingStore()
    store.sessions.clear()
    current_calls: list[str] = []

    async def auth() -> str:
        return "test"

    def translate(exc: SessionStateError) -> Exception:
        return HTTPException(status_code=exc.status_code, detail=exc.message)

    app = FastAPI()
    app.include_router(
        create_session_snapshot_router(
            store=store,
            auth_dependency=auth,
            error_translator=translate,
            model=_MODEL,
            schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
            snapshot_tokenizer_metadata=_metadata,
            current_tokenizer_metadata=lambda: current_calls.append("read") or _metadata(),
        )
    )

    response = TestClient(app).post(
        "/v1/hipengine/sessions/sess/snapshot",
        json=_snapshot(),
    )

    assert response.status_code == 200
    assert current_calls == ["read"]
    assert store.sessions["sess"].messages == ({"role": "user", "content": "hi"},)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_snapshot_owner_has_no_http_application_or_engine_dependency() -> None:
    source = SNAPSHOT_OWNER.read_text(encoding="utf-8")
    modules = _imported_modules(SNAPSHOT_OWNER)

    assert "app.state" not in source
    assert "ServerConfig" not in source
    assert "hipengine.server.api" not in modules
    assert "fastapi" not in modules
    assert "hipengine.llm" not in modules
    assert "hipengine.server.config" not in modules


def test_snapshot_route_owner_receives_only_explicit_dependencies() -> None:
    source = SNAPSHOT_ROUTES.read_text(encoding="utf-8")
    modules = _imported_modules(SNAPSHOT_ROUTES)

    assert "app.state" not in source
    assert "ServerConfig" not in source
    assert "hipengine.server.api" not in modules
    assert "hipengine.llm" not in modules
    assert "hipengine.server.config" not in modules
    assert set(inspect.signature(create_session_snapshot_router).parameters) == {
        "store",
        "auth_dependency",
        "error_translator",
        "model",
        "schema",
        "snapshot_tokenizer_metadata",
        "current_tokenizer_metadata",
    }


def test_snapshot_route_missing_session_translates_404_inside_lease() -> None:
    store = _LeaseObservingStore()
    store.sessions.clear()
    captured: list[SessionStateError] = []

    async def auth() -> str:
        return "test"

    def translate(exc: SessionStateError) -> Exception:
        captured.append(exc)
        return HTTPException(status_code=exc.status_code, detail=exc.message)

    app = FastAPI()
    app.include_router(
        create_session_snapshot_router(
            store=store,
            auth_dependency=auth,
            error_translator=translate,
            model=_MODEL,
            schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
            snapshot_tokenizer_metadata=_metadata,
            current_tokenizer_metadata=lambda: None,
        )
    )

    response = TestClient(app).get("/v1/hipengine/sessions/missing/snapshot")

    assert response.status_code == 404
    assert store.lease_active is False
    assert len(captured) == 1
    assert captured[0].status_code == 404
    assert captured[0].code == "invalid_request"
    assert captured[0].param == "session_id"
    assert captured[0].message == "chat session does not exist"


def test_snapshot_route_metadata_failure_releases_lease_and_preserves_transcript() -> None:
    store = _LeaseObservingStore()
    original = store.sessions["sess"]

    def failing_metadata() -> dict[str, Any]:
        raise RuntimeError("tokenizer metadata failed")

    async def auth() -> str:
        return "test"

    def translate(exc: SessionStateError) -> Exception:
        return HTTPException(status_code=exc.status_code, detail=exc.message)

    app = FastAPI()
    app.include_router(
        create_session_snapshot_router(
            store=store,
            auth_dependency=auth,
            error_translator=translate,
            model=_MODEL,
            schema=CHAT_SESSION_SNAPSHOT_SCHEMA,
            snapshot_tokenizer_metadata=failing_metadata,
            current_tokenizer_metadata=lambda: None,
        )
    )

    with pytest.raises(RuntimeError, match="tokenizer metadata failed"):
        TestClient(app).get("/v1/hipengine/sessions/sess/snapshot")

    assert store.lease_active is False
    assert store.sessions["sess"] is original
    assert store.sessions["sess"].messages == original.messages
