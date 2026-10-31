"""HTTP composition for chat-session snapshot export and restore.

The route owner receives the state store, the application's actual auth
dependency, the application's existing session-error translator, explicit model
identity, and narrow tokenizer-metadata operations. It does not read
application state, an engine or the server config.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import APIRouter, Depends

from hipengine.server.chat_sessions import ChatSessionStore, SessionStateError
from hipengine.server.session_snapshots import (
    SessionSnapshotModel,
    build_chat_session_snapshot,
    restore_chat_session_from_snapshot,
)


def create_session_snapshot_router(
    *,
    store: ChatSessionStore,
    auth_dependency: Callable[..., Any],
    error_translator: Callable[[SessionStateError], Exception],
    model: SessionSnapshotModel,
    schema: str,
    snapshot_tokenizer_metadata: Callable[[], Mapping[str, Any]],
    current_tokenizer_metadata: Callable[[], Mapping[str, Any] | None],
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/hipengine/sessions/{session_id}/snapshot")
    async def export_session_snapshot(
        session_id: str,
        _auth: None = Depends(auth_dependency),
    ) -> dict[str, Any]:
        async with store.snapshot_record(session_id) as record:
            if record is None:
                raise error_translator(
                    SessionStateError(
                        404,
                        "chat session does not exist",
                        code="invalid_request",
                        param="session_id",
                    )
                )
            return build_chat_session_snapshot(
                record,
                schema=schema,
                model=model,
                tokenizer_metadata=snapshot_tokenizer_metadata,
            )

    @router.post("/v1/hipengine/sessions/{session_id}/snapshot")
    async def restore_session_snapshot(
        session_id: str,
        snapshot: dict[str, Any],
        _auth: None = Depends(auth_dependency),
    ) -> dict[str, Any]:
        try:
            record = restore_chat_session_from_snapshot(
                session_id,
                snapshot,
                schema=schema,
                model=model,
                current_tokenizer_metadata=current_tokenizer_metadata,
            )
            await store.put_if_capacity(record)
        except SessionStateError as exc:
            raise error_translator(exc) from exc
        return {
            "object": "hipengine.session.restored",
            "id": session_id,
            "restored": True,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "message_count": len(record.messages),
        }

    return router
