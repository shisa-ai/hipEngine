"""Request-local ownership of the reusable dense DMS prefill oracle."""
from dataclasses import dataclass
from typing import Callable


@dataclass
class DMSPrefillOracleLease:
    buffers: tuple | None

    @classmethod
    def acquire(cls, nbytes: int, *, runtime, allocate: Callable, release: Callable):
        key = allocate(nbytes, runtime=runtime)
        try:
            value = allocate(nbytes, runtime=runtime)
        except BaseException:
            release(key, runtime=runtime)
            raise
        return cls((key, value))

    def close(self, *, runtime, release: Callable) -> None:
        if self.buffers is None:
            return
        key, value = self.buffers
        self.buffers = None
        failure = None
        for buffer in (value, key):
            try:
                release(buffer, runtime=runtime)
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"DMS prefill oracle cleanup also failed: {error!r}")
        if failure is not None:
            raise failure
