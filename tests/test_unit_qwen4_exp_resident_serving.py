from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_gguf import (
    Qwen4ExpGGUFTextGenerator,
    Qwen4ExpResidentServingRunner,
)
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.registry import GenerationRequest


class _Tokenizer:
    eos_token_id = 99

    def encode(self, text):
        return [ord(char) % 32 + 1 for char in str(text)]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(value) for value in ids)


class _FakeRunner:
    max_sequence_length = 64
    prefill_chunk_size = 8
    runtime = None

    def __init__(self, bias=0):
        self.bias = bias
        self.position = 0
        self.closed = False
        self.calls = []
        self.prompt_ids = []

    def reset(self):
        self.position = 0
        self.prompt_ids.clear()

    def prefill_append(self, tokens, **kwargs):
        self.prompt_ids.extend(tokens)
        self.position += len(tokens)
        self.calls.append(("prefill_append", dict(kwargs)))
        return SimpleNamespace(token_id=sum(self.prompt_ids) % 50 + self.bias)

    def prefill(self, tokens, **kwargs):
        self.position = len(tokens)
        self.calls.append(("prefill", dict(kwargs)))
        return SimpleNamespace(token_id=sum(tokens) % 50 + self.bias)

    def step(self, token, **kwargs):
        self.position += 1
        self.calls.append(("step", dict(kwargs)))
        return SimpleNamespace(token_id=int(token) + 1)

    def close(self):
        self.closed = True


def _generator():
    runner = _FakeRunner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), runner=runner,
    )
    generator._resident = SimpleNamespace(close=lambda: None)
    return generator


def test_qwen4_exp_resident_c2_preserves_request_owned_state(monkeypatch) -> None:
    generator = _generator()
    created = []

    def make_runner(*args, **kwargs):
        del args, kwargs
        runner = _FakeRunner(bias=10)
        created.append(runner)
        return runner

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        make_runner,
    )
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=2, max_prefill_chunk_tokens=8),
    )
    try:
        request = GenerationRequest(
            prompts=("alpha", "beta"), max_tokens=3, temperature=0.0,
            top_p=1.0, ignore_eos=True
        )
        native = generator._resident_model_runner
        assert isinstance(native, Qwen4ExpResidentServingRunner)
        outputs = driver.generate_detailed(request)
        assert len(outputs) == 2
        assert all(len(output.generated_token_ids or ()) == 3 for output in outputs)
        assert outputs[0].generated_token_ids != outputs[1].generated_token_ids
        assert len(native._all_runners) == 2
        assert len(created) == 1
    finally:
        driver.close()


def test_qwen4_exp_resident_c2_uses_compact_output_by_default(monkeypatch) -> None:
    generator = _generator()
    created = []

    def make_runner(*args, **kwargs):
        del args, kwargs
        runner = _FakeRunner(bias=10)
        created.append(runner)
        return runner

    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        make_runner,
    )
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=2, max_prefill_chunk_tokens=8),
    )
    try:
        outputs = driver.generate_detailed(
            GenerationRequest(
                prompts=("alpha", "beta"), max_tokens=3, temperature=0.0,
                top_p=1.0, ignore_eos=True,
            )
        )
        assert len(outputs) == 2
        native = generator._resident_model_runner
        assert native is not None
        runners = native._all_runners
        assert len(runners) == 2
        assert len(created) == 1
        assert all(runner.calls for runner in runners)
        assert all(
            kwargs
            == (
                {"capture_logits": False, "capture_target_hidden": False}
                if method == "prefill_append"
                else {
                    "capture_logits": False,
                    "capture_target_hidden": False,
                    "token_id_resident": True,
                }
            )
            for runner in runners
            for method, kwargs in runner.calls
        )
    finally:
        driver.close()


def test_qwen4_exp_resident_admission_rollback_and_cancellation(monkeypatch) -> None:
    generator = _generator()
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: _FakeRunner(bias=10),
    )
    native = generator.create_resident_model_runner(capacity=2)
    request = GenerationRequest(
        prompts=("a", "b", "c"), max_tokens=2, temperature=0.0,
        top_p=1.0, ignore_eos=True,
    )
    native.register_batch((1, 2, 3), request, prompt_rows=((1,), (2,), (3,)))
    native.reserve_admission(SimpleNamespace(request_id=1))
    native.reserve_admission(SimpleNamespace(request_id=2))
    assert native._row(1).runner is not native._row(2).runner
    with pytest.raises(RuntimeError, match="no free request runner"):
        native.reserve_admission(SimpleNamespace(request_id=3))
    native.rollback_admission(SimpleNamespace(request_id=1))
    native.reserve_admission(SimpleNamespace(request_id=3))
    assert native._row(3).runner is not None
    native.discard((1, 2, 3))
    native.close()

    generator = _generator()
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=1, max_prefill_chunk_tokens=8),
    )
    token = GenerationCancellationToken()
    token.cancel()
    cancelled = GenerationRequest(
        prompts=("cancel",), max_tokens=2, temperature=0.0, top_p=1.0,
        ignore_eos=True, cancellation_token=token,
    )
    try:
        outputs = driver.generate_detailed(cancelled)
        assert len(outputs) == 1
        assert outputs[0].generated_token_ids == ()
        assert outputs[0].finish_details.reason == "cancelled"
        native = generator._resident_model_runner
        assert native is not None
        assert all(row.runner is None for row in native._rows.values())
    finally:
        driver.close()


def test_native_prefill_executes_each_chunk_without_replaying_the_prompt():
    generator = _generator()
    native = generator.create_resident_model_runner(capacity=1)
    request = GenerationRequest(
        prompts=((1, 2, 3, 4),), max_tokens=2, temperature=0.0,
        top_p=1.0, ignore_eos=True,
    )
    try:
        native.register_batch((1,), request, prompt_rows=request.prompts)
        native.reserve_admission(SimpleNamespace(request_id=1))
        for chunk, position in [((1, 2), 2), ((3, 4), 4)]:
            native.prefill_batch(WorkItem(
                kind=WorkKind.PREFILL, request_ids=(1,),
                row_to_request=(1, 1), token_rows=(chunk,),
            ), commit=True)
            assert generator.runner.position == position
            assert native._row(1).prefill_tokens_seen == position
        assert generator.runner.prompt_ids == [1, 2, 3, 4]
        assert [method for method, _ in generator.runner.calls] == [
            "prefill_append", "prefill_append",
        ]
        assert native._row(1).next_result.token_id == 10
    finally:
        native.close()


def test_native_prefill_cancellation_stops_work_and_reuses_clean_owner():
    generator = _generator()
    driver = SubmitPollTextGenerator(
        generator, config=EngineLoopConfig(
            max_active_requests=1, max_prefill_chunk_tokens=2,
        ),
    )
    request = GenerationRequest(
        prompts=((1, 2, 3, 4, 5, 6),), max_tokens=2, temperature=0.0,
        top_p=1.0, ignore_eos=True,
    )
    try:
        submission = driver.submit_detailed(request)
        assert generator.runner.calls == []
        driver.poll(max_ticks=1)
        assert generator.runner.prompt_ids == [1, 2]
        assert not driver.generation_complete(submission)
        driver.cancel_submission(submission)
        output = driver.take_result(submission)[0]
        assert output.generated_token_ids == ()
        assert output.finish_details.cancelled
        assert generator.runner.prompt_ids == [1, 2]
        recovered = driver.generate_detailed(GenerationRequest(
            prompts=((7, 8),), max_tokens=2, temperature=0.0,
            top_p=1.0, ignore_eos=True,
        ))[0]
        assert recovered.generated_token_ids == (15, 16)
        assert generator.runner.prompt_ids == [7, 8]
    finally:
        driver.close()


def test_qwen4_exp_generator_advertises_bounded_native_capacity() -> None:
    generator = _generator()
    try:
        assert generator.server_plain_ar_max_active_requests == 2
        runner = generator.create_resident_model_runner(capacity=2)
        assert runner.capacity == 2
    finally:
        generator.close()


def test_qwen4_exp_narrow_scratch_probe_reuses_live_wider_runner(monkeypatch) -> None:
    # A server started without --max-active-requests probes scratch at width 1
    # after the engine loop has already built the runner at the plugin's
    # default width 2. The probe must reuse that runner instead of trying to
    # rebuild it at width 1, which failed startup.
    generator = _generator()
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: _FakeRunner(bias=10),
    )
    try:
        live = generator.create_resident_model_runner(capacity=None)
        assert live.capacity == 2
        report = generator.prepare_request_scratch(max_prompt_tokens=32, max_batch_size=1)
        assert generator._resident_model_runner is live
        assert report["max_batch_size"] == 1
        assert len(live._all_runners) == 2
        report = generator.prepare_request_scratch(max_prompt_tokens=32, max_batch_size=2)
        assert report["max_batch_size"] == 2
    finally:
        generator.close()
