from __future__ import annotations

from types import SimpleNamespace

import pytest


def _execute(*args, **kwargs):
    from hipengine.runtime.qwen35_paro_decode_groups import execute_paro_decode_groups

    return execute_paro_decode_groups(*args, **kwargs)


class _Session:
    def __init__(self, *, native_missing=False, error=None):
        self.calls = []
        self.native_missing = native_missing
        self.error = error

    def step_batch_native(self, tokens, **kwargs):
        self.calls.append(("native", tuple(tokens), kwargs))
        if self.error:
            raise self.error
        if self.native_missing:
            raise NotImplementedError("native shape unavailable")
        return tuple(SimpleNamespace(token_id=token + 1) for token in tokens)

    def step_batch_serial(self, tokens, **kwargs):
        self.calls.append(("serial", tuple(tokens), kwargs))
        return tuple(SimpleNamespace(token_id=token + 1) for token in tokens)


def test_decode_groups_preserve_sparse_slot_order_and_control_boundaries():
    session = _Session()
    checks = []
    result = _execute(
        session, [10, 20, 30], positions=[500, 600, 700], slots=[1, 4, 9],
        groups=(("native", 2), ("serial", 1)),
        native_kwargs={"device_resident": False},
        check_control=lambda: checks.append("check"),
    )
    assert [r.token_id for r in result.results] == [11, 21, 31]
    assert result.effective_groups == (("native", 2), ("serial", 1))
    assert result.native_fallback_rows == 0
    assert checks == ["check"] * 4
    assert session.calls == [
        ("native", (10, 20), dict(positions=[500, 600], slots=[1, 4], sample=True, device_resident=False)),
        ("serial", (30,), dict(positions=[700], slots=[9], sample=True)),
    ]


def test_decode_groups_fall_back_only_for_native_capability_miss():
    session = _Session(native_missing=True)
    result = _execute(
        session, [10, 20], positions=[5, 7], slots=[2, 6],
        groups=(("native", 2),), native_kwargs={"exact_hybrid": True},
    )
    assert result.effective_groups == (("serial", 2),)
    assert result.native_fallback_rows == 2
    assert "exact_hybrid" not in session.calls[-1][-1]
    session = _Session(error=RuntimeError("kernel failed"))
    with pytest.raises(RuntimeError, match="kernel failed"):
        _execute(session, [1, 2], positions=[5, 7], slots=[2, 6], groups=(("native", 2),))
    assert len(session.calls) == 1


@pytest.mark.parametrize("groups", [(("native", 1),), (("serial", 3),), (("bad", 2),), (("serial", 0), ("native", 2))])
def test_decode_groups_validate_full_cover_before_launch(groups):
    session = _Session()
    with pytest.raises(ValueError):
        _execute(session, [1, 2], positions=[5, 7], slots=[2, 6], groups=groups)
    assert session.calls == []


def test_decode_groups_stop_after_control_failure_without_serial_retry():
    session = _Session()
    def expired():
        raise TimeoutError("deadline")
    with pytest.raises(TimeoutError, match="deadline"):
        _execute(session, [1, 2], positions=[5, 7], slots=[2, 6], groups=(("native", 2),), check_control=expired)
    assert session.calls == []


def test_decode_groups_check_deadline_after_native_before_next_group():
    session = _Session()
    count = 0

    def check_control():
        nonlocal count
        count += 1
        if count == 2:
            raise TimeoutError("deadline after native")

    with pytest.raises(TimeoutError, match="deadline after native"):
        _execute(
            session, [1, 2, 3], positions=[5, 7, 9], slots=[2, 6, 8],
            groups=(("native", 2), ("serial", 1)), check_control=check_control,
        )
    assert [call[0] for call in session.calls] == ["native"]


@pytest.mark.parametrize(
    ("tokens", "positions", "slots"),
    [([1], [], [2]), ([1, 2], [5, 7], [2])],
)
def test_decode_groups_validate_row_metadata_lengths(tokens, positions, slots):
    session = _Session()
    with pytest.raises(ValueError, match="equal lengths"):
        _execute(session, tokens, positions=positions, slots=slots, groups=(("serial", len(tokens)),))
    assert session.calls == []


def test_decode_groups_empty_cover_performs_no_launch():
    session = _Session()
    result = _execute(session, [], positions=[], slots=[], groups=())
    assert result.results == ()
    assert result.effective_groups == ()
    assert session.calls == []


def test_decode_groups_reject_result_count_mismatch():
    session = _Session()
    session.step_batch_native = lambda *args, **kwargs: (None,)
    with pytest.raises(RuntimeError, match="results for 2"):
        _execute(session, [1, 2], positions=[5, 7], slots=[2, 6], groups=(("native", 2),))
