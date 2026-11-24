"""CPU tests for scripts/inkling2_decode_bench.py (no model, no GPU)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inkling2_decode_bench.py"
spec = importlib.util.spec_from_file_location("inkling2_decode_bench", MODULE_PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class _Stream:
    """One chunk per token; the token counter advances as each chunk is handed out."""

    def __init__(self, chunks, tokens):
        self._chunks = list(chunks)
        self._tokens = tokens

    def __iter__(self):
        for index, chunk in enumerate(self._chunks, 1):
            self._tokens[0] = index
            yield chunk

    def close(self):
        pass


class _LLM:
    def __init__(self, chunks):
        self._chunks = chunks
        self.requests = 0
        self.tokens = [0]

    def stream_detailed(self, ids, sampling):
        self.requests += 1
        return _Stream(self._chunks, self.tokens)


def chunk(text, ids, diagnostics=None, reason=None):
    telemetry = None if diagnostics is None else SimpleNamespace(diagnostics=diagnostics)
    finish = None if reason is None else SimpleNamespace(reason=reason)
    return SimpleNamespace(text=text, generated_token_ids=None if ids is None else tuple(ids),
                           telemetry=telemetry, finish_details=finish)


def test_prompt_ids_repeats_the_seed_to_the_exact_length():
    assert bench.prompt_ids([3, 5, 7], 7) == (3, 5, 7, 3, 5, 7, 3)
    assert bench.prompt_ids([4], 3) == (4, 4, 4)
    with pytest.raises(ValueError):
        bench.prompt_ids([], 3)
    with pytest.raises(ValueError):
        bench.prompt_ids([1], 0)


def test_install_token_counter_counts_what_the_generator_yields():
    class Generator:
        def _iterate_row(self, request, index, row):
            self.seen = (request, index, row)
            yield from (7, 8, 9)

    original = Generator._iterate_row
    generator = Generator()
    tokens = bench.install_token_counter(Generator)
    assert tokens == [0]
    steps = generator._iterate_row("request", 0, "row")
    assert next(steps) == 7 and tokens == [1]
    next(steps)
    assert tokens == [2]
    assert list(steps) == [9] and tokens == [3]
    assert generator.seen == ("request", 0, "row")
    Generator._iterate_row = original


def test_stream_once_times_from_the_first_text_delta_to_close():
    # The first delta carries no text, so the decode window opens at the second token and
    # holds one decode step for three generated tokens.
    llm = _LLM([chunk("", [10]), chunk("a", [10, 11]),
                chunk("b", [10, 11, 12], {"execution_path": "x"}, "stop")])
    llm.tokens[0] = 500  # a count left over from an earlier request
    sample = bench.stream_once(llm, (1, 2, 3), None, tokens=llm.tokens)
    assert llm.requests == 1
    assert sample["prompt_tokens"] == 3
    assert sample["decode_intervals"] == 1
    assert sample["decode_tok_s"] == pytest.approx(1 / sample["decode_s"])
    assert sample["first_token_s"] >= 0
    assert sample["output_ids"] == [10, 11, 12]
    assert sample["diagnostics"] == {"execution_path": "x"}
    assert sample["finish"] == "stop"
    assert sample["calls"] is None


def test_stream_once_rejects_a_window_without_a_decode_step():
    first_only = _LLM([chunk("a", [10])])
    with pytest.raises(RuntimeError):
        bench.stream_once(first_only, (1,), None, tokens=first_only.tokens)
    no_text = _LLM([chunk("", [10, 11])])
    with pytest.raises(RuntimeError):
        bench.stream_once(no_text, (1,), None, tokens=no_text.tokens)


def _sample(tok, digest, calls=None):
    return {"first_token_s": 0.5, "decode_s": 1.0 / tok, "decode_intervals": 10, "decode_tok_s": tok,
            "output_ids_sha256": digest, "diagnostics": {"k": 1}, "calls": calls}


def test_summarize_reports_medians_and_output_identity():
    same = bench.summarize([_sample(10, "a"), _sample(12, "a"), _sample(11, "a")])
    assert same["reps"] == 3
    assert same["decode_intervals"] == 10
    assert same["decode_tok_s"] == {"median": 11, "min": 10, "max": 12}
    assert same["decode_ms_per_token"] == pytest.approx(1e3 / 11 / 10)
    assert same["identical_output"] and same["output_ids_sha256"] == ["a"]
    mixed = bench.summarize([_sample(10, "a"), _sample(10, "b")])
    assert not mixed["identical_output"] and mixed["output_ids_sha256"] == ["a", "b"]


def test_summarize_averages_hip_calls_per_request_and_per_token():
    calls = {"stream_synchronize": {"calls": 40, "seconds": 0.4, "bytes": 8}}
    summary = bench.summarize([_sample(10, "a", calls), _sample(10, "a", None)])
    assert summary["calls_per_request"] == {
        "stream_synchronize": {"calls": 20, "seconds": pytest.approx(0.2), "bytes": 4}}
    assert summary["calls_per_token"] == {
        "stream_synchronize": {"calls": 2, "seconds": pytest.approx(0.02), "bytes": pytest.approx(0.4)}}


def test_provenance_records_repo_model_and_protocol():
    args = SimpleNamespace(backend="hip_gfx1100", quant="UD-Q4_K_M", warmup=1, reps=5)
    block = bench.provenance("model.gguf", args)
    assert block["kind"] == "hipengine_artifact_provenance"
    assert block["configured_backend"] == "hip_gfx1100" and block["resolved_backend"] != "auto"
    assert block["kv_dtype"] == "fp32" and block["quant"] == "UD-Q4_K_M"
    assert block["warmups"] == 1 and block["repetitions"] == 5
    assert block["timing_protocol"] == bench.PROTOCOL
    assert block["model_fingerprint"]["exists"] is False


def test_environment_lists_hipengine_variables(monkeypatch):
    monkeypatch.setenv("HIPENGINE_EXAMPLE", "1")
    recorded = bench.environment()
    assert recorded["hipengine_env"]["HIPENGINE_EXAMPLE"] == "1"
    assert set(recorded) == {"host", "platform", "python", "hipengine_env", "rocm_path"}


class _Kind:
    HOST_TO_DEVICE = "host_to_device"
    DEVICE_TO_HOST = "device_to_host"


class _Runtime:
    def __init__(self):
        self.waits = 0

    def stream_synchronize(self, stream):
        self.waits += 1

    def memcpy(self, dst, src, nbytes, kind):
        return nbytes

    def memcpy_async(self, dst, src, nbytes, kind, stream):
        return nbytes


def test_call_counter_counts_copies_by_direction_and_restores_the_class():
    original_sync = _Runtime.stream_synchronize
    original_copy = _Runtime.memcpy
    original_async = _Runtime.memcpy_async
    runtime = _Runtime()
    counter = bench.CallCounter(_Runtime, names=("stream_synchronize",), copies=("memcpy", "memcpy_async"))
    runtime.stream_synchronize(0)
    counter.start()
    runtime.stream_synchronize(0)
    runtime.stream_synchronize(0)
    runtime.memcpy(1, 2, 64, _Kind.HOST_TO_DEVICE)
    runtime.memcpy(3, 4, 32, _Kind.DEVICE_TO_HOST)
    runtime.memcpy_async(5, 6, 16, _Kind.HOST_TO_DEVICE, 0)
    with pytest.raises(RuntimeError):
        counter.start()
    tally = counter.stop()
    assert runtime.waits == 3
    assert tally["stream_synchronize"]["calls"] == 2
    assert tally["stream_synchronize"]["seconds"] >= 0
    assert tally["memcpy:host_to_device"]["calls"] == 1
    assert tally["memcpy:host_to_device"]["bytes"] == 64
    assert tally["memcpy:device_to_host"]["bytes"] == 32
    assert tally["memcpy_async:host_to_device"]["bytes"] == 16
    assert _Runtime.stream_synchronize is original_sync
    assert _Runtime.memcpy is original_copy
    assert _Runtime.memcpy_async is original_async
    runtime.stream_synchronize(0)
    assert tally["stream_synchronize"]["calls"] == 2
