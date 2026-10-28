"""Pin the existing legacy MTP chunk policy without opening a device session."""

from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf


def _generator():
    return object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)


@pytest.mark.parametrize(
    "count,widths",
    [
        (0, ()),
        (1, ()),
        (2, (2,)),
        (3, (3,)),
        (4, (4,)),
        (5, (3, 2)),
        (6, (4, 2)),
        (7, (4, 3)),
        (8, (4, 4)),
        (9, (4, 3, 2)),
        (13, (4, 4, 3, 2)),
    ],
)
def test_verifier_chunks_preserve_order_without_singleton_tail(monkeypatch, count, widths):
    generator = _generator()
    calls = []
    owner = object()
    monkeypatch.setattr(qwen35_gguf, "_gguf_mtp_server_stream_verify_enabled", lambda: False)

    def verify(chunk, *, verify_owner_session):
        assert verify_owner_session is owner
        calls.append(tuple(chunk))
        return list(chunk)

    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)
    cycles = list(range(count))
    result = generator._try_verify_mtp_serving_cycles_batch(cycles, verify_owner_session=owner)

    assert tuple(map(len, calls)) == widths
    assert [row for chunk in calls for row in chunk] == (cycles if count > 1 else [])
    assert result == (cycles if count > 1 else None)


@pytest.mark.parametrize("stream_available", [False, True])
def test_verifier_uses_stream_chunks_or_serial_capability_fallback(monkeypatch, stream_available):
    generator = _generator()
    streamed = []
    serial = []
    monkeypatch.setattr(qwen35_gguf, "_gguf_mtp_server_stream_verify_enabled", lambda: True)

    def streams(chunks):
        streamed.append([list(chunk) for chunk in chunks])
        return list(range(9)) if stream_available else None

    def verify(chunk, *, verify_owner_session):
        serial.append(list(chunk))
        return list(chunk)

    monkeypatch.setattr(generator, "_try_verify_mtp_serving_cycle_chunks_streams", streams)
    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)

    assert generator._try_verify_mtp_serving_cycles_batch(list(range(9))) == list(range(9))
    expected = [list(range(4)), list(range(4, 7)), list(range(7, 9))]
    assert streamed == [expected]
    assert serial == ([] if stream_available else expected)


@pytest.mark.parametrize("failure_index", [0, 1])
def test_verifier_declines_only_before_any_chunk_advances(monkeypatch, failure_index):
    generator = _generator()
    calls = []
    monkeypatch.setattr(qwen35_gguf, "_gguf_mtp_server_stream_verify_enabled", lambda: False)

    def verify(chunk, *, verify_owner_session):
        index = len(calls)
        calls.append(list(chunk))
        return None if index == failure_index else list(chunk)

    monkeypatch.setattr(generator, "_verify_mtp_serving_cycle_chunk", verify)
    if failure_index:
        with pytest.raises(RuntimeError, match="prior chunk advanced state"):
            generator._try_verify_mtp_serving_cycles_batch(list(range(8)))
    else:
        assert generator._try_verify_mtp_serving_cycles_batch(list(range(8))) is None
    assert len(calls) == failure_index + 1


@pytest.mark.parametrize("count", [5, 7, 8, 13])
def test_prefill_above_cap_declines_before_acquiring_slots(monkeypatch, count):
    generator = _generator()
    monkeypatch.setattr(qwen35_gguf, "_gguf_mtp_server_packed_prefill_enabled", lambda: True)

    def unexpected(*args, **kwargs):
        pytest.fail("the over-cap packed route must decline before acquiring sessions")

    monkeypatch.setattr(generator, "_acquire_shared_session", unexpected)
    assert generator._try_open_mtp_serving_slots_batch_prefill(
        None,
        None,
        {},
        SimpleNamespace(prompts=tuple(range(count))),
        pool_sessions=True,
    ) is None
