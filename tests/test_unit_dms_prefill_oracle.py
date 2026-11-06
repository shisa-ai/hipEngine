"""Transient oracle acquisition has one rollback owner."""
from types import SimpleNamespace

import pytest

from hipengine.runtime.dms_prefill_oracle import DMSPrefillOracleLease


def test_second_plane_failure_releases_first_once():
    key = object()
    calls = []

    def allocate(nbytes, *, runtime):
        calls.append(nbytes)
        if len(calls) == 2:
            raise MemoryError('value allocation')
        return key

    released = []
    with pytest.raises(MemoryError, match='value allocation'):
        DMSPrefillOracleLease.acquire(
            128, runtime=SimpleNamespace(), allocate=allocate,
            release=lambda buffer, **kwargs: released.append(buffer),
        )
    assert released == [key]
