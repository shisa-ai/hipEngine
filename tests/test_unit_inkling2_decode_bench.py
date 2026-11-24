"""CPU tests for scripts/inkling2_decode_bench.py (no model, no GPU)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inkling2_decode_bench.py"
spec = importlib.util.spec_from_file_location("inkling2_decode_bench", MODULE_PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def chunk(text, ids, diagnostics=None, reason=None):
    telemetry = None if diagnostics is None else SimpleNamespace(diagnostics=diagnostics)
    finish = None if reason is None else SimpleNamespace(reason=reason)
    return SimpleNamespace(text=text, generated_token_ids=None if ids is None else tuple(ids),
                           telemetry=telemetry, finish_details=finish)


def detokenize(ids, *, skip_special):
    """Token 10 decodes to nothing, like a leading special token."""
    return "" if ids == [10] else "text"


class _Stream:
    """One chunk per token. The engine side runs step ``index + 1`` while chunk ``index`` is delivered."""

    def __init__(self, chunks, steps, calls):
        self._chunks = list(chunks)
        self._steps = steps
        self._calls = calls

    def __iter__(self):
        for index, piece in enumerate(self._chunks):
            self._steps.started.append(float(index))
            self._steps.finished.append(index + 1.0)
            if self._calls:
                self._steps.calls.append({"stream_synchronize": {"calls": index + 1, "seconds": 0.5, "bytes": 0}})
            yield piece

    def close(self):
        pass


class _LLM:
    def __init__(self, chunks):
        self._chunks = chunks
        self.steps = bench.DecodeSteps()
        self.calls = False
        self.requests = 0

    def stream_detailed(self, ids, sampling):
        self.requests += 1
        return _Stream(self._chunks, self.steps, self.calls)


def test_prompt_ids_repeats_the_seed_to_the_exact_length():
    assert bench.prompt_ids([3, 5, 7], 7) == (3, 5, 7, 3, 5, 7, 3)
    assert bench.prompt_ids([4], 3) == (4, 4, 4)
    with pytest.raises(ValueError):
        bench.prompt_ids([], 3)
    with pytest.raises(ValueError):
        bench.prompt_ids([1], 0)


def test_first_text_tokens_counts_up_to_the_first_token_with_text():
    assert bench.first_text_tokens([10, 10, 11], detokenize) == 3
    assert bench.first_text_tokens([11, 10], detokenize) == 1
    with pytest.raises(RuntimeError):
        bench.first_text_tokens([10, 10], detokenize)


def test_stream_once_times_the_decode_steps_in_the_window():
    # Token 10 has no text, so the window opens at token 2 and holds one step: the step that
    # produces token 3.
    llm = _LLM([chunk("", [10]), chunk("a", [10, 11]), chunk("b", [10, 11, 12], {"execution_path": "x"}, "stop")])
    llm.steps.started, llm.steps.finished = [99.0], [100.0]  # left over from an earlier request
    sample = bench.stream_once(llm, (1, 2, 3), None, steps=llm.steps, detokenize=detokenize)
    assert llm.requests == 1
    assert sample["prompt_tokens"] == 3
    assert sample["decode_intervals"] == 1
    assert sample["decode_tok_s"] == pytest.approx(1 / sample["decode_s"])
    assert sample["engine_decode_s"] == pytest.approx(2.0 - 1.0)
    assert sample["engine_decode_tok_s"] == pytest.approx(1.0)
    assert sample["first_token_s"] >= 0
    assert sample["output_ids"] == [10, 11, 12]
    assert sample["diagnostics"] == {"execution_path": "x"}
    assert sample["finish"] == "stop"
    assert sample["calls"] is None


def test_stream_once_counts_hip_calls_of_the_steps_in_the_window_only():
    llm = _LLM([chunk("", [10]), chunk("a", [10, 11]), chunk("b", [10, 11, 12], None, "stop")])
    llm.calls = True
    sample = bench.stream_once(llm, (1, 2, 3), None, steps=llm.steps, detokenize=detokenize, counts=True)
    # Three steps ran; only the second lies in the window.
    assert sample["calls"] == {"stream_synchronize": {"calls": 2, "seconds": pytest.approx(0.5), "bytes": 0}}


def test_stream_once_rejects_a_window_without_a_decode_step():
    single = _LLM([chunk("a", [11], None, "stop")])
    with pytest.raises(RuntimeError):
        bench.stream_once(single, (1,), None, steps=single.steps, detokenize=detokenize)
    silent = _LLM([chunk("", [10, 11], None, "length")])
    with pytest.raises(RuntimeError):
        bench.stream_once(silent, (1,), None, steps=silent.steps, detokenize=detokenize)
    blank = _LLM([chunk("a", [10, 10], None, "stop")])
    with pytest.raises(RuntimeError):
        bench.stream_once(blank, (1,), None, steps=blank.steps, detokenize=detokenize)
    stalled = _LLM([chunk("a", [11]), chunk("b", [11, 12], None, "stop")])
    stalled.steps.count_calls = False
    stalled.steps.started, stalled.steps.finished = [], []

    class _NoSteps(_Stream):
        def __iter__(self):
            self._steps.started, self._steps.finished = [], []
            yield from self._chunks

    stalled.stream_detailed = lambda ids, sampling: _NoSteps(stalled._chunks, stalled.steps, False)
    with pytest.raises(RuntimeError):
        bench.stream_once(stalled, (1,), None, steps=stalled.steps, detokenize=detokenize)


def test_install_step_timer_records_real_steps_and_skips_the_first_tick():
    class Model:
        def decode_rows(self, rows):
            return [len(rows)]

    steps = bench.DecodeSteps()
    model = Model()
    assert bench.install_step_timer(Model, steps) is steps
    assert model.decode_rows([]) == [0]
    assert steps.prompt_done_at is not None and steps.started == [] and steps.finished == []
    assert model.decode_rows([(1, 2)]) == [1]
    assert len(steps.started) == 1 and steps.finished[0] >= steps.started[0]
    assert steps.calls == []
    steps.count_calls = True
    model.decode_rows([(1, 2)])
    assert steps.calls == [{}]
    from hipengine.core.hip import HipRuntime
    assert "wrapped" not in HipRuntime.stream_synchronize.__name__


def _sample(tok, digest, calls=None):
    return {"first_token_s": 0.5, "decode_s": 1.0 / tok, "decode_intervals": 10, "decode_tok_s": tok,
            "engine_decode_tok_s": tok, "output_ids_sha256": digest, "diagnostics": {"k": 1}, "calls": calls}


def test_summarize_reports_medians_and_output_identity():
    same = bench.summarize([_sample(10, "a"), _sample(12, "a"), _sample(11, "a")])
    assert same["reps"] == 3
    assert same["decode_intervals"] == 10
    assert same["decode_tok_s"] == {"median": 11, "min": 10, "max": 12}
    assert same["engine_decode_tok_s"]["median"] == 11
    assert same["decode_ms_per_token"] == pytest.approx(1e3 / 11 / 10)
    assert same["identical_output"] and same["output_ids_sha256"] == ["a"]
    assert same["calls_per_token"] == {}
    mixed = bench.summarize([_sample(10, "a"), _sample(10, "b")])
    assert not mixed["identical_output"] and mixed["output_ids_sha256"] == ["a", "b"]


def test_summarize_averages_hip_calls_over_the_counted_repetitions():
    calls = {"stream_synchronize": {"calls": 40, "seconds": 0.4, "bytes": 8}}
    summary = bench.summarize([_sample(10, "a", calls), _sample(10, "a", None)])
    assert summary["calls_per_request"] == {
        "stream_synchronize": {"calls": 40, "seconds": pytest.approx(0.4), "bytes": 8}}
    assert summary["calls_per_token"] == {
        "stream_synchronize": {"calls": 4, "seconds": pytest.approx(0.04), "bytes": pytest.approx(0.8)}}


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
