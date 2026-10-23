"""Device uploads shared by draft runners belonging to one model owner."""

from __future__ import annotations

from typing import Any

import numpy as np


class WeightUploadCache:
    """Own immutable named weights, separated by allocating runtime.

    Names identify immutable tensors within this owner, never across models.
    Borrowing draft runners do not close this cache; their model owner does so
    after closing the runners and graphs that reference its buffers.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, str], tuple[Any, Any]] = {}
        self._failed_uploads: list[tuple[Any, Any]] = []

    def upload(self, name: str, data: np.ndarray, *, runtime: Any) -> Any:
        from hipengine.core.memory import copy_host_array_to_device, free, malloc

        key = (id(runtime), name)
        existing = self._entries.get(key)
        if existing is not None:
            buffer = existing[1]
            if int(buffer.nbytes) != int(data.nbytes):
                raise ValueError(f"immutable weight {name!r} changed its byte size")
            return buffer
        array = np.ascontiguousarray(data)
        buffer = malloc(array.nbytes, runtime=runtime)
        try:
            copy_host_array_to_device(buffer, array, runtime=runtime)
        except BaseException as error:
            try:
                free(buffer, runtime=runtime)
            except BaseException as cleanup_error:
                # Keep a failed release reachable for owner shutdown to retry.
                self._failed_uploads.append((runtime, buffer))
                error.add_note(f"Weight upload cleanup failed: {cleanup_error!r}")
            raise
        self._entries[key] = (runtime, buffer)
        return buffer

    def close(self) -> None:
        from hipengine.core.memory import free

        error: BaseException | None = None
        for key, (runtime, buffer) in tuple(self._entries.items()):
            try:
                free(buffer, runtime=runtime)
            except BaseException as exc:
                if error is None:
                    error = exc
            else:
                del self._entries[key]
        for index in range(len(self._failed_uploads) - 1, -1, -1):
            runtime, buffer = self._failed_uploads[index]
            try:
                free(buffer, runtime=runtime)
            except BaseException as exc:
                if error is None:
                    error = exc
            else:
                del self._failed_uploads[index]
        if error is not None:
            raise error
