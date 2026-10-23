"""Temporary resource ownership before a generation service is published."""

from __future__ import annotations

from typing import Any


class ConstructionOwner:
    """Close the current owner on failure, until ownership is handed onward.

    ``take`` replaces an earlier owner only after the new object has successfully
    assumed responsibility for it. ``release`` transfers responsibility to the
    caller. A factory that raises before returning still owns its own cleanup.
    """

    def __init__(self) -> None:
        self._resource: Any = None

    def take(self, resource: Any) -> Any:
        self._resource = resource
        return resource

    def release(self) -> Any:
        resource, self._resource = self._resource, None
        return resource

    def __enter__(self) -> ConstructionOwner:
        return self

    def __exit__(self, exc_type, error, traceback) -> None:
        resource = self.release()
        if error is None or resource is None:
            return
        try:
            closer = getattr(resource, "close", None)
            if callable(closer):
                closer()
        except BaseException as cleanup_error:
            error.add_note(
                f"Construction cleanup of {type(resource).__name__} failed: "
                f"{cleanup_error!r}"
            )
