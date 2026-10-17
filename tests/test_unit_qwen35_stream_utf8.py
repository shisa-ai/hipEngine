"""Streaming must assemble tokenizer bytes across cycles without sharing rows."""

from types import SimpleNamespace

import pytest

from hipengine.generation.batch_scheduler import GeneratedTokenEvent
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.registry import GenerationStreamChunk
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer, bytes_to_unicode


@pytest.fixture
def runner():
    byte_map = bytes_to_unicode()
    tokenizer = Qwen35GGUFTokenizer(
        tokens=tuple(byte_map[i] for i in range(256)) + ("<eos>", "<think>", "</think>", "漢"),
        token_types=(1,) * 256 + (3, 4, 4, 4), merges=(), eos_token_id=256,
    )
    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = SimpleNamespace(tokenizer=tokenizer)
    runner._cycle_token_samples = {}
    return runner


def event(rid, token, *, finished=False):
    return GeneratedTokenEvent(
        request_id=rid, token_id=token, finished=finished,
        stream_chunk=GenerationStreamChunk(text="", generated_token_ids=(token,)),
    )


def decorate(runner, *events):
    result = runner.decorate_speculative_stream_events(events)
    assert [e.stream_chunk.generated_token_ids for e in result] == [
        e.stream_chunk.generated_token_ids for e in events
    ]
    return [e.stream_chunk.text for e in result]


@pytest.mark.parametrize("text", ["┌", "日", "🙂", "café"])
def test_speculative_stream_assembles_byte_fragments_across_cycles(runner, text):
    data = text.encode()
    chunks = []
    for i, token in enumerate(data):
        chunks.extend(decorate(runner, event(1, token, finished=i == len(data) - 1)))
    assert "".join(chunks) == text
    assert "\ufffd" not in "".join(chunks)


def test_interleaved_requests_do_not_share_pending_utf8(runner):
    # One request emits a three-byte box glyph; another emits a two-byte accent.
    assert decorate(runner, event(1, 0xE2), event(2, 0xC3)) == ["", ""]
    assert decorate(runner, event(1, 0x94), event(2, 0xA9, finished=True)) == ["", "é"]
    assert decorate(runner, event(1, 0x8C, finished=True)) == ["┌"]


def test_special_token_suppression_is_request_local(runner):
    assert decorate(runner, event(1, 256), event(2, ord("B"))) == ["", "B"]
    assert decorate(runner, event(1, ord("X"), finished=True)) == [""]


def test_terminal_incomplete_utf8_matches_blocking_replacement(runner):
    assert decorate(runner, event(1, 0xE2, finished=True)) == ["\ufffd"]
    assert runner._stream_decoders == {}


@pytest.mark.parametrize("skip_special", [False, True])
@pytest.mark.parametrize("ids", [
    (0xE2, 0x94, 0x8C), (0xE2,), (0xFF, 0xA9, 65),
    (0xE2, 256, 0x94, 0x8C), (0xE2, 259, 65), (257, 0xC3, 0xA9, 258),
])
def test_tokenizer_incremental_matches_full_decode(runner, ids, skip_special):
    tokenizer = runner.generator.tokenizer
    decoder = tokenizer.incremental_decoder(skip_special=skip_special)
    parts = [decoder.feed((token,)) for token in ids]
    parts.append(decoder.feed((), final=True))
    assert "".join(parts) == tokenizer.decode(ids, skip_special=skip_special)


@pytest.mark.parametrize("token", [-1, 260])
def test_tokenizer_incremental_checks_vocabulary_bounds(runner, token):
    with pytest.raises(ValueError, match="outside vocabulary"):
        runner.generator.tokenizer.incremental_decoder().feed((token,))


def test_thinking_delimiters_and_terminal_byte_flush(runner):
    assert decorate(runner, event(1, 257), event(1, 0xC3)) == ["<think>", ""]
    assert decorate(runner, event(1, 0xA9), event(1, 258)) == ["é", "</think>"]
    assert decorate(runner, event(1, 0xE2), event(1, 256, finished=True)) == ["", "\ufffd"]
    assert runner._stream_decoders == {}


@pytest.fixture
def resident_row(runner, monkeypatch):
    import hipengine.generation.qwen35_gguf as gguf
    from hipengine.generation.registry import GenerationRequest

    monkeypatch.setattr(gguf, "_gguf_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(gguf, "_gguf_finish_details", lambda *args, **kwargs: None)
    runner._request_diagnostics = lambda *args, **kwargs: {}
    row = gguf._GGUFResidentLoopRow(
        request_id=1, batch_id=1, row_index=0, prompt_ids=(65,),
        request=GenerationRequest(prompts=("A",), max_tokens=8, temperature=0.7, logprobs=True,
                                  top_p=1.0, ignore_eos=False),
        native_greedy=False, native_sampled=True, native_sampler=True, submitted_at=0.0,
    )
    row.slot = SimpleNamespace(
        generated_ids=[], timing={}, done=False, native_compact_prefill=True,
        native_decode_steps=1, serial_decode_steps=0,
    )
    return row


def test_ar_and_mtp_share_byte_state_in_both_directions(runner, resident_row):
    row = resident_row
    row.slot.generated_ids.append(0xE2)
    assert runner._native_stream_chunk(row).text == ""
    row.slot.generated_ids.append(0x94)
    assert decorate(runner, event(1, 0x94)) == [""]
    row.slot.generated_ids.append(0x8C)
    row.slot.done = True
    assert runner._native_stream_chunk(row).text == "┌"
    assert runner._stream_decoders == {}


def test_ignore_eos_keeps_streaming_after_a_hidden_eos(runner, resident_row):
    from dataclasses import replace
    row = resident_row
    row.request = replace(row.request, ignore_eos=True)
    runner._rows = {1: row}
    row.slot.generated_ids.append(256)
    assert runner._native_stream_chunk(row).text == ""
    assert decorate(runner, event(1, ord("B"), finished=True)) == ["B"]


def test_sampled_blocking_and_streaming_decode_ids_not_logprob_text(runner, resident_row):
    row = resident_row
    parts = []
    for token in "日本🙂".encode():
        row.slot.generated_ids.append(token)
        row.samples.append(SimpleNamespace(
            token_id=token, token_text="\ufffd", logprob=-0.5, top_logprobs=(),
        ))
        parts.append(runner._native_stream_chunk(row).text)
    output = runner._native_output(row, SimpleNamespace(finish_reason="length"))
    assert output.text == "".join(parts) == "日本🙂"
    assert output.generated_token_ids == tuple(row.slot.generated_ids)
    assert all(token.token_text == "\ufffd" for token in output.token_logprobs)


def test_sampled_direct_output_decodes_committed_ids(runner):
    from hipengine.generation.qwen35_gguf import _gguf_generation_output
    samples = [SimpleNamespace(token_id=i, logprob=-0.5, top_logprobs=()) for i in "café".encode()]
    output = _gguf_generation_output(runner.generator.tokenizer, samples, finish_details=None)
    assert output.text == "café"
    assert output.generated_token_ids == tuple("café".encode())


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("text", ["┌日🙂", "\ufffd"])
def test_direct_generator_stream_assembles_and_flushes_bytes(runner, monkeypatch, sampled, text):
    import hipengine.generation.qwen35_gguf as gguf
    from hipengine.generation.registry import GenerationRequest

    ids = tuple(text.encode()) if text != "\ufffd" else (0xE2,)
    generated = iter(ids)
    session = SimpleNamespace(
        prefill=lambda *args, **kwargs: SimpleNamespace(token_id=next(generated)),
        step=lambda *args, **kwargs: SimpleNamespace(token_id=next(generated)),
    )
    generator = object.__new__(gguf.Qwen35GGUFBringupGenerator)
    generator.tokenizer = runner.generator.tokenizer
    monkeypatch.setattr(gguf, "_gguf_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(gguf, "_gguf_logits_d2h_metadata", lambda *args: (False, 0))
    monkeypatch.setattr(gguf, "_select_from_gguf_logits", lambda result, *args: SimpleNamespace(
        token_id=result.token_id, logprob=-0.5, top_logprobs=(),
    ))
    request = GenerationRequest(prompts=("A",), max_tokens=len(ids), temperature=0.7 if sampled else 0.0,
                                top_p=1.0, ignore_eos=False)
    stream = generator._stream_sampled if sampled else generator._stream_greedy
    chunks = list(stream(session, [65], request, tokenize_ms=0.0, **({"row_index": 0} if sampled else {})))
    assert "".join(c.text for c in chunks) == text
    assert chunks[-1].generated_token_ids == ids
    assert chunks[-1].finish_details is not None


@pytest.mark.parametrize("cleanup", ["take_outputs", "discard"])
def test_completion_and_cancellation_release_decoder_state(runner, cleanup):
    assert decorate(runner, event(1, 0xE2)) == [""]
    runner._outputs = {1: "done"}
    runner._rows = {}
    runner._completed_metadata = {}
    getattr(runner, cleanup)((1,))
    assert runner._stream_decoders == {}
    assert decorate(runner, event(1, ord("B"), finished=True)) == ["B"]
