from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.dispatch import NativeBatchWidthProfile
from scripts import qwen35_batch_retained_bench as retained_bench


class _FakeActiveBatch:
    def __init__(self, rows: int) -> None:
        self.requests = {
            request_id: SimpleNamespace(context_len=512)
            for request_id in range(rows)
        }

    @staticmethod
    def slot_for(request_id: int) -> int:
        return request_id


class _FakeScheduler:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.active_batch = _FakeActiveBatch(rows)
        self.recorded = []

    def next_decode_work(self, **_kwargs):
        return SimpleNamespace(request_ids=tuple(range(self.rows)))

    def record_generated(self, generated) -> None:
        self.recorded.extend(generated)


class _FakeSession:
    def __init__(self) -> None:
        self.calls = []
        self.profile = NativeBatchWidthProfile(
            source_artifact="benchmarks/results/gfx1151-profile.json",
            native_step_ms=(
                (2, 25.465),
                (3, 34.310),
                (4, 40.158),
                (5, 48.927),
                (6, 54.568),
                (7, 63.905),
                (8, 69.254),
            ),
            serial_row_step_ms=14.969,
            min_position=512,
            max_position=647,
        )

    def native_batch_width_profile(self) -> NativeBatchWidthProfile:
        return self.profile

    def step_batch_native(self, token_ids, *, positions, slots, sample, device_resident=False):
        self.calls.append(("native", tuple(slots), device_resident))
        return tuple(
            SimpleNamespace(token_id=token_id + 1, token_text=str(token_id + 1), logit=1.0)
            for token_id in token_ids
        )

    def step_batch_serial(self, token_ids, *, positions, slots, sample):
        self.calls.append(("serial", tuple(slots), False))
        return tuple(
            SimpleNamespace(token_id=token_id + 1, token_text=str(token_id + 1), logit=1.0)
            for token_id in token_ids
        )


@pytest.mark.parametrize(
    ("rows", "expected_calls", "native_complete", "signature"),
    [
        (9, [("native", tuple(range(8)), False), ("serial", (8,), False)], False, "native:8+serial:1"),
        (
            10,
            [("native", tuple(range(8)), False), ("native", (8, 9), False)],
            True,
            "native:8+native:2",
        ),
        (
            16,
            [("native", tuple(range(8)), False), ("native", tuple(range(8, 16)), False)],
            True,
            "native:8+native:8",
        ),
    ],
)
def test_profile_partitioned_retained_decode_uses_exact_native_cover(
    rows: int,
    expected_calls: list[tuple[str, tuple[int, ...], bool]],
    native_complete: bool,
    signature: str,
) -> None:
    session = _FakeSession()
    scheduler = _FakeScheduler(rows)
    next_tokens = {request_id: 100 + request_id for request_id in range(rows)}
    generated = {request_id: [] for request_id in range(rows)}

    count, native, metadata = retained_bench._decode_scheduler_step_native(
        session,
        scheduler,
        next_tokens,
        generated,
        count_output=True,
        execution_mode="profile_partitioned",
    )

    assert count == rows
    assert native is native_complete
    assert session.calls == expected_calls
    assert metadata["signature"] == signature
    assert metadata["requested_plan"]["requested_rows"] == rows
    assert sum(group["width"] for group in metadata["requested_plan"]["groups"]) == rows
    assert [next_tokens[row] for row in range(rows)] == [101 + row for row in range(rows)]


@pytest.mark.parametrize(
    ("execution_mode", "expected_call", "native"),
    [
        ("direct_native", ("native", tuple(range(9)), False), True),
        ("serial", ("serial", tuple(range(9)), False), False),
    ],
)
def test_retained_decode_execution_controls(
    execution_mode: str,
    expected_call: tuple[str, tuple[int, ...], bool],
    native: bool,
) -> None:
    session = _FakeSession()
    scheduler = _FakeScheduler(9)
    next_tokens = {request_id: 100 + request_id for request_id in range(9)}
    generated = {request_id: [] for request_id in range(9)}

    count, native_complete, metadata = retained_bench._decode_scheduler_step_native(
        session,
        scheduler,
        next_tokens,
        generated,
        count_output=False,
        execution_mode=execution_mode,
    )

    assert count == 9
    assert native_complete is native
    assert session.calls == [expected_call]
    assert metadata["signature"] == ("native:9" if native else "serial:9")
