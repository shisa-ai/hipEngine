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


def test_partial_acquisition_preserves_allocation_error_when_release_fails():
    calls = []
    primary = MemoryError('value allocation')

    def allocate(nbytes, *, runtime):
        calls.append(nbytes)
        if len(calls) == 2:
            raise primary
        return object()

    def release(buffer, *, runtime):
        raise RuntimeError('key release failed')

    with pytest.raises(MemoryError, match='value allocation') as caught:
        DMSPrefillOracleLease.acquire(
            128, runtime=SimpleNamespace(), allocate=allocate, release=release,
        )
    assert caught.value is primary
    assert any('key release failed' in note for note in primary.__notes__)
