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


@pytest.mark.parametrize("control", [None, "0", "1"])
@pytest.mark.parametrize("count", [1, 2, 7, 8, 9, 13])
def test_compat_packed_chunks_preserve_stream_control_and_singleton_fallback(
    monkeypatch, control, count,
):
    generator = object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)
    if control is None:
        monkeypatch.delenv("HIPENGINE_GGUF_AR_STREAM_DECODE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", control)
    rows = [SimpleNamespace(request_id=index) for index in range(count)]
    observed = []
    generator._try_step_ar_serving_slot_chunks_streams = lambda chunks, request: (
        observed.append(("stream", tuple(len(chunk) for chunk in chunks))) or False
    )
    generator._step_ar_serving_slot_chunk_packed = lambda chunk, request: (
        observed.append(("packed", tuple(row.request_id for row in chunk))) or True
    )
    generator._flush_ar_packed_decode_owners = lambda chunk: None
    generator._step_ar_serving_slot_serial = lambda row, request: observed.append(
        ("serial", row.request_id)
    )

    handled = generator._try_step_ar_serving_slots_batch(rows, object())

    assert handled is (count > 1)
    assert any(kind == "stream" for kind, _ in observed) is (
        count > 8 and control != "0"
    )
    if count > 1:
        packed_ids = [rid for kind, ids in observed if kind == "packed" for rid in ids]
        assert packed_ids == list(range(count))
        assert all(len(chunk) <= 8 for chunk in generator._ar_serving_slot_chunks(rows))
    else:
        assert observed == []


@pytest.mark.parametrize("control", [None, "0", "1"])
@pytest.mark.parametrize("missing", [None, "step_async_top1", "stream_synchronize"])
def test_compat_slot_streams_preserve_explicit_serial_and_prelaunch_capability_fallback(
    monkeypatch, control, missing,
):
    generator = object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)
    if control is None:
        monkeypatch.delenv("HIPENGINE_GGUF_AR_STREAM_DECODE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_AR_STREAM_DECODE", control)
    calls = []
    generator._flush_ar_packed_decode_owners = lambda rows: None
    generator._record_ar_serving_token = lambda row, token, request: calls.append(("token", token))
    rows = []
    for index in range(2):
        runtime = SimpleNamespace(
            stream_create=lambda **kwargs: 100,
            stream_synchronize=lambda stream: calls.append(("sync", stream)),
            stream_destroy=lambda stream: None,
        )
        session = SimpleNamespace(
            runtime=runtime,
            step_async_top1=lambda *args, **kwargs: calls.append(("launch", kwargs["stream"])),
            read_top1_sample=lambda: SimpleNamespace(token_id=7),
        )
        if missing == "step_async_top1" and index == 1:
            session.step_async_top1 = None
        if missing == "stream_synchronize" and index == 1:
            runtime.stream_synchronize = None
        rows.append(SimpleNamespace(session=session, decode_stream=0, prev_token=1,
                                    seq_position=10, timing={}, native_decode_steps=0))

    handled = generator._try_step_ar_serving_slots_streams(rows, object())

    expected = control != "0" and missing is None
    assert handled is expected
    if expected:
        assert [kind for kind, _ in calls] == ["launch", "launch", "sync", "sync", "token", "token"]
        assert [row.native_decode_steps for row in rows] == [1, 1]
    else:
        assert calls == []
