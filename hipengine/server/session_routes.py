"""HTTP composition for the core app-local chat-session routes.

The route owner receives the state store and the application's actual auth
dependency. It builds success payloads and translates typed state failures
through the supplied HTTP error operation; it does not read ``app.state``, the
server config or an engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends

from hipengine.server.chat_sessions import (
    ChatSessionStore,
    SessionStateError,
    session_metadata,
)
from hipengine.server.schemas import SessionForkRequest, SessionRollbackRequest


def create_session_router(
    *,
    store: ChatSessionStore,
    auth_dependency: Callable[..., Any],
    error_translator: Callable[[SessionStateError], Exception],
) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/hipengine/sessions")
    async def list_sessions(_auth: str = Depends(auth_dependency)) -> dict[str, Any]:
        listing = await store.listing()
        return {
            "object": "hipengine.sessions",
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "includes_transcript": False,
            "active": len(listing.records),
            "pending_creations": listing.pending_creations,
            "max_active": (
                None if store.max_sessions is None else int(store.max_sessions)
            ),
            "sessions": [session_metadata(record) for record in listing.records],
            "continuations": {
                "active": listing.active_continuations,
                "ttl_seconds": store.continuation_ttl_seconds,
            },
        }

    @router.delete("/v1/hipengine/sessions/{session_id}")
    async def delete_session(
        session_id: str,
        _auth: str = Depends(auth_dependency),
    ) -> dict[str, Any]:
        deleted = await store.delete(session_id)
        return {
            "object": "hipengine.session.deleted",
            "id": session_id,
            "deleted": deleted,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
        }

    @router.post("/v1/hipengine/sessions/{session_id}/fork")
    async def fork_session(
        session_id: str,
        request: SessionForkRequest,
        _auth: str = Depends(auth_dependency),
    ) -> dict[str, Any]:
        target_id = str(request.id).strip()
        if not target_id:
            raise error_translator(
                SessionStateError(
                    400,
                    "target session id must be a non-empty string",
                    code="invalid_request",
                    param="id",
                )
            )
        if target_id == session_id:
            raise error_translator(
                SessionStateError(
                    400,
                    "target session id must differ from source session id",
                    code="invalid_request",
                    param="id",
                )
            )
        try:
            forked = await store.fork(session_id, target_id)
        except SessionStateError as exc:
            raise error_translator(exc) from exc
        return {
            "object": "hipengine.session.forked",
            "source_id": session_id,
            "id": target_id,
            "forked": True,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "message_count": len(forked.messages),
        }

    @router.post("/v1/hipengine/sessions/{session_id}/rollback")
    async def rollback_session(
        session_id: str,
        request: SessionRollbackRequest,
        _auth: str = Depends(auth_dependency),
    ) -> dict[str, Any]:
        target_count = int(request.message_count)
        try:
            result = await store.rollback(session_id, target_count)
        except SessionStateError as exc:
            raise error_translator(exc) from exc
        return {
            "object": "hipengine.session.rolled_back",
            "id": session_id,
            "rolled_back": result.rolled_back,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "previous_message_count": result.previous_message_count,
            "message_count": target_count,
        }

    return router
