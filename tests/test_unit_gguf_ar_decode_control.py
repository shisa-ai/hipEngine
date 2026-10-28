"""CPU contracts for packed/serial resident AR diagnostic selection."""

from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf


@pytest.mark.parametrize("count", [2, 3, 5, 8, 9, 13])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("packed_available", [False, True])
def test_resident_decode_preserves_explicit_control_and_capability_fallback(
    monkeypatch, count, enabled, packed_available
):
    runner = object.__new__(qwen35_gguf.Qwen35GGUFResidentModelRunner)
    runner._resident_batch_owner = object()
    runner._shared_runner = SimpleNamespace(backend="hip_gfx1151")
    runner._last_execution_manifest = {}
    runner._last_physical_group_plan = {}
    packed = []
    serial = []

    def chunk(rows, **kwargs):
        packed.extend(row.request_id for row in rows)
        return packed_available

    def fallback(rows, *, fallback_reason):
        serial.extend(row.request_id for row in rows)

    runner._step_native_chunk = chunk
    runner._step_native_serial = fallback
    monkeypatch.setattr(qwen35_gguf, "_gguf_ar_physical_widths", lambda *a, **kw: (1, 2, 4, 8))
    monkeypatch.setenv("HIPENGINE_GGUF_AR_PACKED_DECODE", "1" if enabled else "0")
    rows = [
        SimpleNamespace(
            request_id=index,
            slot=SimpleNamespace(
                session=SimpleNamespace(kv_attention_source="bf16", packed_decode_max_rows=8),
                c1_decode_graph=None,
            ),
        )
        for index in range(count)
    ]

    runner._step_native_rows(rows)

    assert (bool(packed)) is enabled
    if not enabled or not packed_available:
        assert serial == list(range(count))
    else:
        assert sorted(packed + serial) == list(range(count))
        assert len(set(packed + serial)) == count
    plan = runner._last_physical_group_plan
    assert plan["logical_c"] == count
    if not enabled:
        assert all(group["execution_path"] != "packed_native" for group in plan["groups"])
    elif packed_available:
        assert any(group["execution_path"] == "packed_native" for group in plan["groups"])
