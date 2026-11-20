"""CPU ownership contracts for the R0471 opt-in native-row resident-scheduler route.

These pin the dispatch + capability contract for connecting the resident
scheduler to the compact native-row owner (``step_rows_native``, the
``ar_decode_native_rows`` DP4A-capable route) without disturbing the
packed/serial default:

- the dispatch tries the route only when ``native_batch_decode`` is requested,
  and falls back to packed/serial when the route declines;
- the route itself engages only for a compact physical group (active lanes ==
  physical width, >= 2 rows) where every row is greedy;
- a workspace refusal falls back to the packed/serial route, not a failed request.

Host-only: no HIP runtime, no device buffers. Synthetic sessions mirror
``tests/test_unit_gguf_ar_decode_control.py``.
"""

from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf


def _runner(monkeypatch, *, native_batch_decode, packed_available=True):
    runner = object.__new__(qwen35_gguf.Qwen35GGUFResidentModelRunner)
    runner._resident_batch_owner = object()
    runner.generator = object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)
    runner.generator.ar_packed_decode = True
    runner.generator.native_batch_decode = native_batch_decode
    runner._shared_runner = SimpleNamespace(backend="hip_gfx1100", vocab_size=32)
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    runner._route_counts = {}
    runner._fallback_reasons = {}
    calls = {"compact": [], "packed": [], "serial": []}

    def compact(rows, **kwargs):
        calls["compact"].extend(row.request_id for row in rows)
        return True

    def chunk(rows, **kwargs):
        calls["packed"].extend(row.request_id for row in rows)
        return packed_available

    def fallback(rows, *, fallback_reason):
        calls["serial"].extend(row.request_id for row in rows)

    runner._step_native_rows_compact = compact
    runner._step_native_chunk = chunk
    runner._step_native_serial = fallback
    monkeypatch.setattr(
        qwen35_gguf, "_gguf_ar_physical_widths", lambda *a, **kw: (1, 2, 4, 8)
    )
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1")
    return runner, calls


def _rows(count, *, greedy=True):
    rows = []
    for index in range(count):
        rows.append(
            SimpleNamespace(
                request_id=index,
                native_greedy=greedy,
                native_sampled=not greedy,
                native_sampler=False,
                slot=SimpleNamespace(
                    session=SimpleNamespace(
                        kv_attention_source="bf16", packed_decode_max_rows=8
                    ),
                    c1_decode_graph=None,
                    seq_position=10 + index,
                    prev_token=1,
                    generated_ids=[],
                    native_decode_steps=0,
                    packed_decode_owner=None,
                    done=False,
                ),
            )
        )
    return rows


# --- dispatch contract (route mocked) -------------------------------------


def test_dispatch_tries_native_row_route_when_requested(monkeypatch):
    runner, calls = _runner(monkeypatch, native_batch_decode=True)
    runner._step_native_rows(_rows(4))
    assert calls["compact"] == [0, 1, 2, 3]
    assert calls["packed"] == []
    assert calls["serial"] == []


def test_dispatch_preserves_packed_default_when_not_requested(monkeypatch):
    runner, calls = _runner(monkeypatch, native_batch_decode=False)
    runner._step_native_rows(_rows(4))
    assert calls["compact"] == []
    assert sorted(calls["packed"] + calls["serial"]) == [0, 1, 2, 3]


def test_dispatch_falls_back_when_native_row_route_declines(monkeypatch):
    runner, calls = _runner(monkeypatch, native_batch_decode=True)
    runner._step_native_rows_compact = lambda rows, **kwargs: False
    runner._step_native_rows(_rows(4))
    assert calls["compact"] == []
    assert sorted(calls["packed"] + calls["serial"]) == [0, 1, 2, 3]


# --- route capability contract (real method) ------------------------------


def _capability_runner(monkeypatch, *, native_batch_decode=True):
    runner = object.__new__(qwen35_gguf.Qwen35GGUFResidentModelRunner)
    runner.generator = object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)
    runner.generator.native_batch_decode = native_batch_decode
    runner._route_counts = {}
    runner._fallback_reasons = {}
    return runner


@pytest.mark.parametrize("requested, expected", [(True, [0, 1]), (False, [3, 2])])
def test_resident_leases_prioritize_compact_slots_only_for_native_rows(monkeypatch, requested, expected):
    runner = _capability_runner(monkeypatch, native_batch_decode=requested)
    runner._resident_batch_owner = object()
    runner._available = [
        SimpleNamespace(session=SimpleNamespace(_resident_slot_index=i))
        for i in range(4)
    ]
    assert [runner._acquire_lease().session._resident_slot_index for _ in expected] == expected


def test_route_declines_when_not_requested(monkeypatch):
    runner = _capability_runner(monkeypatch, native_batch_decode=False)
    assert runner._step_native_rows_compact(
        _rows(4), physical_rows=4, active_slot_indices=(0, 1, 2, 3)
    ) is False


def test_route_declines_non_greedy_rows(monkeypatch):
    runner = _capability_runner(monkeypatch)
    assert runner._step_native_rows_compact(
        _rows(4, greedy=False), physical_rows=4, active_slot_indices=(0, 1, 2, 3)
    ) is False


def test_route_declines_singleton(monkeypatch):
    runner = _capability_runner(monkeypatch)
    assert runner._step_native_rows_compact(
        _rows(1), physical_rows=1, active_slot_indices=(0,)
    ) is False


def test_route_declines_non_compact_lanes(monkeypatch):
    runner = _capability_runner(monkeypatch)
    # 2 active rows in a 4-wide physical group with a gap -> not compact.
    assert runner._step_native_rows_compact(
        _rows(2), physical_rows=4, active_slot_indices=(0, 2)
    ) is False


def test_route_declines_when_owner_lacks_step_rows_native(monkeypatch):
    runner = _capability_runner(monkeypatch)
    runner._resident_batch_owner = SimpleNamespace()  # no step_rows_native
    assert runner._step_native_rows_compact(
        _rows(4), physical_rows=4, active_slot_indices=(0, 1, 2, 3)
    ) is False