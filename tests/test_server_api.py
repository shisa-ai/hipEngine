from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS, FinishDetails, GenerationOutput, TokenLogprob
from hipengine.server import ServerConfig, create_app, render_chat_prompt
from hipengine.server.__main__ import build_parser
from hipengine.server.api import ChatCompletionRequest, _GenerationBatcher


class FakeLLM:
    def __init__(
        self,
        outputs: list[str] | None = None,
        stream_chunks: list[str] | None = None,
        token_map: dict[str, list[int]] | None = None,
        detailed_outputs: list[GenerationOutput] | None = None,
    ) -> None:
        self.outputs = outputs
        self.detailed_outputs = detailed_outputs
        self.stream_chunks = stream_chunks
        self.token_map = token_map
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []
        self.prepares: list[tuple[int | None, SamplingParams]] = []
        self.tokenize_calls: list[str] = []
        self.max_sequence_length: int | None = None
        self.kv_capacity_estimate = None
        self.kv_capacity_int8_estimate = None

    def prepare(self, *, max_sequence_length: int | None = None, sampling_params: SamplingParams) -> int:
        self.prepares.append((None if max_sequence_length is None else int(max_sequence_length), sampling_params))
        requested = 262144 if max_sequence_length is None else int(max_sequence_length)
        selected = min(262144, 131072) if max_sequence_length is None else requested
        self.max_sequence_length = selected
        self.kv_capacity_estimate = _fake_kv_estimate(
            max_sequence_length=selected,
            storage="bf16" if sampling_params.kv_storage == "auto" else sampling_params.kv_storage,
        )
        self.kv_capacity_int8_estimate = _fake_kv_estimate(
            max_sequence_length=selected,
            storage="int8_per_token_head",
        )
        return selected

    def generate(self, prompts, sampling_params: SamplingParams) -> list[str]:
        return [output.text for output in self.generate_detailed(prompts, sampling_params)]

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(prompts)
        self.calls.append((prompts, sampling_params))
        if self.detailed_outputs is not None:
            return self.detailed_outputs[: len(prompts)]
        if self.outputs is not None:
            return [GenerationOutput(text=output) for output in self.outputs[: len(prompts)]]
        return [GenerationOutput(text=f"generated:{prompt}") for prompt in prompts]

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self.calls.append(((prompt,), sampling_params))
        if self.stream_chunks is not None:
            yield from self.stream_chunks
        elif self.outputs is not None:
            yield self.outputs[0]
        else:
            yield f"generated:{prompt}"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def tokenize(self, text: str) -> tuple[int, ...]:
        self.tokenize_calls.append(str(text))
        if self.token_map is None:
            raise NotImplementedError("fake tokenization is not configured")
        return tuple(self.token_map[str(text)])


class DetailedGenerateFakeLLM(FakeLLM):
    def generate(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        return self.generate_detailed(prompts, sampling_params)


def _fake_kv_estimate(*, max_sequence_length: int, storage: str):
    bytes_per_token = 8192
    rounded_tokens = ((int(max_sequence_length) + 255) // 256) * 256
    return SimpleNamespace(
        requested_context_tokens=int(max_sequence_length),
        model_max_context_tokens=262144,
        allocatable_context_tokens=131072,
        requested_kv_bytes=rounded_tokens * bytes_per_token,
        bytes_per_token=bytes_per_token,
        usable_bytes=4 * 1024**3,
        reserve_bytes=512 * 1024**2,
        kv_storage_dtype=storage,
        kv_scale_dtype="fp16" if storage == "int8_per_token_head" else None,
        fits_model_max=False,
    )


def test_models_endpoint_reports_served_model_name_and_auth() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret"),
        llm=fake,
    )
    client = TestClient(app)

    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"

    response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "fake-model",
                "object": "model",
                "created": app.state.hipengine_config.created,
                "owned_by": "hipengine",
            }
        ],
    }


def test_server_eager_loads_model_on_startup(caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = FakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        eager_load_prompt="one two three four",
        eager_load_max_tokens=2,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "Config: model=fake-path" in caplog.text
    assert "max_context_tokens=131072" in caplog.text
    assert "chat_default_max_tokens=4096" in caplog.text
    assert "kv_storage=auto" in caplog.text
    assert "KVCache: storage=bf16" in caplog.text
    assert "model_max_context_tokens=262144" in caplog.text
    assert "WARMUP: prompt_tokens<=131072 max_tokens=2" in caplog.text
    assert "LOAD_TIMING: phase=startup resident_prepare_s=" in caplog.text
    assert "LOAD_TIMING: model=fake-model engine_create_s=" in caplog.text
    assert "warmup_s=" in caplog.text
    assert "hipEngine is ready." in caplog.text
    assert fake.prepares == [
        (
            None,
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
        )
    ]
    assert fake.calls == [
        (
            ("one two three four",),
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
        )
    ]


def test_chat_default_max_tokens_is_dynamic_when_omitted() -> None:
    request = ChatCompletionRequest(
        model="fake-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert request.max_tokens is None


def test_generation_batcher_coalesces_compatible_submissions() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two", "three"), sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two", "generated:three"]
        assert fake.calls == [(("one", "two", "three"), sampling)]

    asyncio.run(run())


def test_generation_batcher_default_zero_window_queues_without_lifetime_lock() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two",), sampling),
        )

        streamed = [chunk async for chunk in batcher.stream(("three",), sampling)]

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert streamed == ["generated:three"]
        assert fake.calls == [(("one", "two"), sampling), (("three",), sampling)]

    asyncio.run(run())


def test_generation_batcher_stream_uses_per_request_queue_and_coalesces() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        async def collect_stream() -> list[str]:
            return [chunk async for chunk in batcher.stream(("stream",), sampling)]

        streamed, submitted = await asyncio.gather(
            collect_stream(),
            batcher.submit(("batch",), sampling),
        )

        assert streamed == ["generated:stream"]
        assert submitted == ["generated:batch"]
        assert len(fake.calls) == 1
        assert set(fake.calls[0][0]) == {"stream", "batch"}
        assert fake.calls[0][1] == sampling

    asyncio.run(run())


def test_generation_batcher_keeps_incompatible_sampling_separate() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_sampling = SamplingParams(max_tokens=1)
        second_sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls == [(("one",), first_sampling), (("two",), second_sampling)]

    asyncio.run(run())


def test_completions_endpoint_calls_llm_and_applies_stop() -> None:
    fake = FakeLLM(outputs=["alpha<stop>tail", "beta<stop>tail"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": ["one", "two"],
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": "<stop>",
            "kv_storage": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "fake-model"
    assert [choice["text"] for choice in body["choices"]] == ["alpha", "beta"]
    assert [choice["finish_reason"] for choice in body["choices"]] == ["stop", "stop"]
    assert [choice["finish_details"] for choice in body["choices"]] == [{"reason": "stop"}, {"reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert fake.calls == [
        (
            ("one", "two"),
            SamplingParams(
                max_tokens=3,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=False,
                kv_storage="int8_per_token_head",
                kv_scale_dtype="fp32",
            ),
        )
    ]


def test_completions_preserve_structured_finish_details() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="greedy_fast"),
            ),
            GenerationOutput(
                text="beta",
                finish_details=FinishDetails(reason="length", length_limit=2, budget_pressure="answer_budget"),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 2},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert [choice["finish_reason"] for choice in choices] == ["stop", "length"]
    assert choices[0]["finish_details"] == {
        "reason": "eos",
        "eos_token_id": 151645,
        "sampler_mode": "greedy_fast",
    }
    assert choices[1]["finish_details"] == {
        "reason": "length",
        "length_limit": 2,
        "budget_pressure": "answer_budget",
    }


def test_server_lowers_single_token_stop_strings_to_stop_token_ids() -> None:
    fake = FakeLLM(outputs=["alpha!tail"], token_map={"!": [99], "two tokens": [10, 11]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 4,
            "stop": ["!", "two tokens"],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "alpha"
    sampling = fake.calls[0][1]
    assert sampling.stop_token_ids == (99,)
    assert sampling.stop_token_sequences == ((10, 11),)
    assert fake.tokenize_calls == ["!", "two tokens"]


def test_chat_completion_uses_bounded_default_max_tokens() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=100,
            chat_default_max_tokens=7,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][1].max_tokens == 7


def test_chat_completion_auto_default_max_tokens_uses_remaining_context() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=12,
            chat_default_max_tokens=None,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[0][0][0]
    assert fake.calls[0][1].max_tokens == 12 - fake.count_tokens(prompt) - 1


def test_completions_endpoint_plumbs_sampling_parameters() -> None:
    fake = FakeLLM(outputs=["sampled"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 2,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "logit_bias": {"12": -1.5},
            "seed": 123,
        },
    )

    assert response.status_code == 200
    sampling = fake.calls[0][1]
    assert sampling.temperature == 0.8
    assert sampling.top_p == 0.9
    assert sampling.top_k == 40
    assert sampling.min_p == 0.05
    assert sampling.repetition_penalty == 1.1
    assert sampling.presence_penalty == 0.2
    assert sampling.frequency_penalty == 0.3
    assert sampling.logit_bias == ((12, -1.5),)
    assert sampling.seed == 123



def test_completions_endpoint_returns_openai_logprobs() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha beta",
                token_logprobs=(
                    TokenLogprob(
                        token_id=1,
                        token_text="alpha",
                        logprob=-0.25,
                        top_logprobs=((1, "alpha", -0.25), (2, "omega", -1.5)),
                    ),
                    TokenLogprob(
                        token_id=3,
                        token_text=" beta",
                        logprob=-0.5,
                        top_logprobs=((3, " beta", -0.5),),
                    ),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 2, "logprobs": 2},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "alpha beta"
    assert choice["logprobs"]["tokens"] == ["alpha", " beta"]
    assert choice["logprobs"]["token_logprobs"] == [-0.25, -0.5]
    assert choice["logprobs"]["top_logprobs"][0] == {"alpha": -0.25, "omega": -1.5}
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 2


def test_completions_endpoint_echo_logprobs_shift_generated_offsets() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text=" world",
                token_logprobs=(
                    TokenLogprob(token_id=7, token_text=" world", logprob=-0.125),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "echo": True, "logprobs": 0},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "hello world"
    assert choice["logprobs"]["tokens"] == ["hello", " world"]
    assert choice["logprobs"]["token_logprobs"] == [None, -0.125]
    assert choice["logprobs"]["text_offset"] == [0, 5]


def test_streaming_completion_returns_logprobs_from_buffered_path() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                token_logprobs=(
                    TokenLogprob(token_id=1, token_text="alpha", logprob=-0.25, top_logprobs=((1, "alpha", -0.25),)),
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "alpha"
    assert payloads[0]["choices"][0]["logprobs"]["token_logprobs"] == [-0.25]
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "stop"}
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_chat_completion_returns_openai_logprobs() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                token_logprobs=(
                    TokenLogprob(
                        token_id=4,
                        token_text="assistant",
                        logprob=-0.1,
                        top_logprobs=((4, "assistant", -0.1),),
                    ),
                    TokenLogprob(token_id=5, token_text=" reply", logprob=-0.2),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "assistant reply"
    assert choice["logprobs"]["content"][0]["token"] == "assistant"
    assert choice["logprobs"]["content"][0]["top_logprobs"] == [
        {"token": "assistant", "logprob": -0.1, "bytes": None}
    ]
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 1


def test_streaming_chat_completion_returns_logprobs_from_buffered_path() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="assistant",
                token_logprobs=(
                    TokenLogprob(token_id=4, token_text="assistant", logprob=-0.1),
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content_chunks = [payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")]
    assert content_chunks[0]["choices"][0]["delta"] == {"content": "assistant"}
    assert content_chunks[0]["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.1
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "stop"}
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_chat_completion_renders_messages_to_prompt() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "assistant reply"},
            "finish_reason": "stop",
            "finish_details": {"reason": "stop"},
        }
    ]
    assert fake.calls[0][0] == (
        "<|im_start|>system\nbe concise<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n",
    )
    assert fake.calls[0][1].max_tokens == 4


def test_chat_completion_segregates_reasoning_content() -> None:
    fake = FakeLLM(outputs=["<think>scratch pad</think>assistant reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": "assistant reply",
        "reasoning_content": "scratch pad",
    }


def test_chat_completion_accepts_qwen_no_think_controls() -> None:
    fake = FakeLLM(outputs=["direct answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer directly"}],
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][0][0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_completion_accepts_reasoning_effort_controls() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    low = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think briefly"}],
            "reasoning_effort": "low",
        },
    )
    assert low.status_code == 200
    assert "keep it very brief" in fake.calls[-1][0][0]
    assert not fake.calls[-1][0][0].endswith("<think>\n\n</think>\n\n")

    none = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "do not think"}],
            "reasoning_effort": "none",
        },
    )
    assert none.status_code == 200
    assert fake.calls[-1][0][0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_render_chat_prompt_includes_qwen_tool_blocks() -> None:
    prompt = render_chat_prompt(
        [
            {"role": "developer", "content": "Use tools carefully."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "read"}},
    )

    assert "<tools>" in prompt
    assert '"name":"read"' in prompt
    assert "You must call the function named 'read'." in prompt
    assert "<|im_start|>system\nUse tools carefully.<|im_end|>" in prompt
    assert '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>' in prompt
    assert "<tool_response>\nfile text\n</tool_response>" in prompt


def test_chat_completion_returns_openai_tool_calls() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {"reason": "tool_calls"}
    message = choice["message"]
    assert message["content"] == ""
    tool_call = message["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<tools>" in fake.calls[0][0][0]


def test_streaming_chat_completion_returns_tool_call_deltas() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["index"] == 0
    assert tool_call["id"].startswith("call_")
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "tool_calls"}


def test_streaming_chat_completion_returns_token_sse_and_usage() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=["<think>scratch pad</think>streamed ", "reply"],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    deltas = [payload["choices"][0]["delta"] for payload in payloads if payload.get("choices")]
    assert deltas[:4] == [
        {"role": "assistant"},
        {"reasoning_content": "scratch pad"},
        {"content": "streamed "},
        {"content": "reply"},
    ]
    assert len(fake.stream_calls) == 1
    prompt = fake.calls[0][0][0]
    completion_tokens = fake.count_tokens("<think>scratch pad</think>streamed reply")
    assert payloads[-1]["usage"] == {
        "prompt_tokens": fake.count_tokens(prompt),
        "completion_tokens": completion_tokens,
        "total_tokens": fake.count_tokens(prompt) + completion_tokens,
    }
    assert fake.calls[0][1].max_tokens == 4096


def test_streaming_completion_uses_engine_stream_and_usage() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    text_chunks = [payload["choices"][0]["text"] for payload in payloads if payload.get("choices")]
    assert text_chunks == ["alpha", " beta", ""]
    assert payloads[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert fake.stream_calls == [("hello", SamplingParams(max_tokens=2))]


def test_metrics_prefix_cache_and_generation_batch_cli_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_DEBUG", raising=False)
    monkeypatch.delenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", raising=False)
    default_args = build_parser().parse_args(["--model", "fake-path"])
    assert default_args.generation_batch_window_ms == 0.0
    assert default_args.debug is False
    assert default_args.chat_default_max_tokens == 4096

    monkeypatch.setenv("HIPENGINE_METRICS", "prometheus")
    monkeypatch.setenv("HIPENGINE_PREFIX_CACHE", "radix")
    monkeypatch.setenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", "3.5")
    monkeypatch.setenv("HIPENGINE_DEBUG", "1")
    monkeypatch.setenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", "auto")
    env_args = build_parser().parse_args(["--model", "fake-path"])
    assert env_args.metrics == "prometheus"
    assert env_args.prefix_cache == "radix"
    assert env_args.generation_batch_window_ms == 3.5
    assert env_args.debug is True
    assert env_args.chat_default_max_tokens is None

    cli_args = build_parser().parse_args(
        [
            "--model",
            "fake-path",
            "--metrics",
            "off",
            "--prefix-cache",
            "off",
            "--generation-batch-window-ms",
            "0",
            "--chat-default-max-tokens",
            "123",
            "--no-debug",
        ]
    )
    assert cli_args.metrics == "off"
    assert cli_args.prefix_cache == "off"
    assert cli_args.generation_batch_window_ms == 0.0
    assert cli_args.chat_default_max_tokens == 123
    assert cli_args.debug is False

    app = create_app(ServerConfig(model="fake-path", eager_load=False, prefix_cache="radix"), llm=FakeLLM())
    assert app.state.hipengine_prefix_cache_mode == "radix"


def test_debug_mode_logs_full_request_and_response_payloads(caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = FakeLLM(outputs=["debug reply"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, debug=True),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1},
    )

    assert response.status_code == 200
    assert "DEBUG_PAYLOAD REQUEST POST /v1/completions" in caplog.text
    assert '"prompt":"hello"' in caplog.text
    assert "DEBUG_PAYLOAD RESPONSE POST /v1/completions status=200" in caplog.text
    assert '"text":"debug reply"' in caplog.text


def test_metrics_endpoint_is_opt_in_and_additive() -> None:
    disabled = create_app(ServerConfig(model="fake-path", eager_load=False), llm=FakeLLM())
    assert TestClient(disabled).get("/metrics").status_code == 404

    fake = FakeLLM(outputs=["alpha beta"])
    fake.kv_pool_stats = SimpleNamespace(
        current_bytes=4096,
        high_water_observed_bytes=8192,
        grow_events=2,
        grow_failures=1,
        shrink_events=3,
        free_pages=4,
        refcounted_pages=5,
    )
    fake.graph_bucket_stats = SimpleNamespace(
        entries=6,
        hits=7,
        misses=8,
        replay_hit_rate=0.0,
        miss_reasons={"cache_absent": 5, "shape_changed": 3, "bool_bad": True, "nan_bad": float("nan")},
        kernel_time_histogram_ns={
            "le_10us": 2,
            "le_100us": 4,
            "le_1ms": True,
            "le_10ms": float("inf"),
            "gt_10ms": -1,
            "lt_1us": 9,
        },
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    before = client.get("/metrics")
    assert before.status_code == 200
    assert _metric_value(before.text, "hipengine_requests_total") == 0

    for prompt in ["one", "two three"]:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": prompt, "max_tokens": 2},
        )
        assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert _metric_value(metrics.text, "hipengine_requests_total") == 2
    assert _metric_value(metrics.text, "hipengine_request_completed_total") == 2
    assert _metric_value(metrics.text, "hipengine_request_failed_total") == 0
    assert _metric_value(metrics.text, "hipengine_prompt_tokens_total") == 3
    assert _metric_value(metrics.text, "hipengine_completion_tokens_total") == 4
    assert _metric_value(metrics.text, "hipengine_kv_pool_current_bytes") == 4096
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_events_total") == 2
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_failures_total") == 1
    assert _metric_value(metrics.text, "hipengine_kv_pool_shrink_events_total") == 3
    assert _metric_value(metrics.text, "hipengine_kv_pool_free_pages") == 4
    assert _metric_value(metrics.text, "hipengine_kv_pool_refcounted_pages") == 5
    assert _metric_value(metrics.text, "hipengine_graph_bucket_entries") == 6
    assert _metric_value(metrics.text, "hipengine_graph_bucket_hits_total") == 7
    assert _metric_value(metrics.text, "hipengine_graph_bucket_misses_total") == 8
    assert _metric_value(metrics.text, "hipengine_graph_bucket_replay_hit_rate") == 7 / 15
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_miss_reason_total", reason="cache_absent") == 5
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_miss_reason_total", reason="shape_changed") == 3
    assert 'hipengine_graph_bucket_miss_reason_total{reason="bool_bad"}' not in metrics.text
    assert 'hipengine_graph_bucket_miss_reason_total{reason="nan_bad"}' not in metrics.text
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_10us") == 2
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_100us") == 4
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_1ms") == 0
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_10ms") == 0
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="gt_10ms") == 0
    for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS:
        assert f'hipengine_graph_bucket_kernel_time_bucket_total{{bucket="{bucket}"}}' in metrics.text
    assert 'hipengine_graph_bucket_kernel_time_bucket_total{bucket="lt_1us"}' not in metrics.text


def test_metrics_endpoint_filters_malformed_graph_bucket_scalars() -> None:
    fake = FakeLLM()
    fake.graph_bucket_stats = SimpleNamespace(
        entries=True,
        hits=float("nan"),
        misses=float("inf"),
        replay_hit_rate=1.0,
        miss_reasons={},
        kernel_time_histogram_ns={},
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    malformed = client.get("/metrics")

    assert malformed.status_code == 200
    assert _metric_value(malformed.text, "hipengine_graph_bucket_entries") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_hits_total") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_misses_total") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_replay_hit_rate") == 0

    fake.graph_bucket_stats = SimpleNamespace(
        entries=-1,
        hits=3,
        misses=-4,
        replay_hit_rate=0.0,
        miss_reasons={},
        kernel_time_histogram_ns={},
    )
    partially_valid = client.get("/metrics")

    assert partially_valid.status_code == 200
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_entries") == 0
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_hits_total") == 3
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_misses_total") == 0
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_replay_hit_rate") == 1


def test_metrics_endpoint_filters_malformed_kv_pool_scalars() -> None:
    fake = FakeLLM()
    fake.kv_pool_stats = SimpleNamespace(
        current_bytes=True,
        high_water_observed_bytes=float("nan"),
        grow_events=float("inf"),
        grow_failures=-1,
        shrink_events="bad",
        free_pages=3,
        refcounted_pages=-4,
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert _metric_value(metrics.text, "hipengine_kv_pool_current_bytes") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_high_water_observed_bytes") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_events_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_failures_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_shrink_events_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_free_pages") == 3
    assert _metric_value(metrics.text, "hipengine_kv_pool_refcounted_pages") == 0


def test_streaming_chat_completion_lowers_n_to_seeded_rows() -> None:
    fake = FakeLLM(outputs=["alpha", "beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "n": 2,
            "seed": 5,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    indices = [payload["choices"][0]["index"] for payload in payloads]
    assert 0 in indices and 1 in indices
    assert "data: [DONE]" in response.text
    assert fake.calls[0][0] == (fake.calls[0][0][0], fake.calls[0][0][0])
    assert len(fake.calls[0][1].row_seeds) == 2
    assert len(set(fake.calls[0][1].row_seeds)) == 2


def test_server_rejects_requests_beyond_preallocated_context() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=5,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one two three four", "max_tokens": 2},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert fake.calls == []


def test_server_rejects_request_kv_policy_mismatch() -> None:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            kv_storage="int8_per_token_head",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "kv_storage": "bf16"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_kv_policy"


def test_server_rejects_wrong_model_and_unsupported_options(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=FakeLLM())
    client = TestClient(app)

    wrong_model = client.post(
        "/v1/completions",
        json={"model": "other", "prompt": "hello"},
    )
    assert wrong_model.status_code == 404
    assert wrong_model.json()["error"]["code"] == "model_not_found"

    unsupported_chat_top_logprobs = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "top_logprobs": 1,
        },
    )
    assert unsupported_chat_top_logprobs.status_code == 400
    assert unsupported_chat_top_logprobs.json()["error"]["param"] == "top_logprobs"

    unsupported_extra = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "typical_p": 0.9},
    )
    assert unsupported_extra.status_code == 400
    assert unsupported_extra.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported_extra.json()["error"]["param"] == "typical_p"
    assert "REQUEST_FAILED: POST /v1/completions status=404 code=model_not_found" in caplog.text
    assert "REQUEST_FAILED: POST /v1/completions status=400 code=unsupported_parameter" in caplog.text
    assert "param=typical_p" in caplog.text


def test_completions_endpoint_lowers_n_to_distinct_seeded_rows() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 1, "n": 2, "seed": 123},
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["text"] for choice in body["choices"]] == [
        "generated:one",
        "generated:one",
        "generated:two",
        "generated:two",
    ]
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2, 3]
    assert len({choice["request_id"] for choice in body["choices"]}) == 4
    assert fake.calls[0][0] == ("one", "one", "two", "two")
    assert len(fake.calls[0][1].row_seeds) == 4
    assert len(set(fake.calls[0][1].row_seeds)) == 4


def test_chat_endpoint_lowers_n_to_distinct_seeded_rows() -> None:
    fake = FakeLLM(outputs=["first", "second"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}], "n": 2, "seed": 9},
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["index"] for choice in body["choices"]] == [0, 1]
    assert [choice["message"]["content"] for choice in body["choices"]] == ["first", "second"]
    assert len({choice["request_id"] for choice in body["choices"]}) == 2
    assert fake.calls[0][0] == (fake.calls[0][0][0], fake.calls[0][0][0])
    assert len(fake.calls[0][1].row_seeds) == 2
    assert len(set(fake.calls[0][1].row_seeds)) == 2


def test_render_chat_prompt_accepts_plain_message_mappings() -> None:
    assert render_chat_prompt([{"role": "user", "content": "hello"}]) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_chat_endpoint_rejects_non_text_content_parts() -> None:
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=FakeLLM())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_content_type"


def _metric_value(text: str, name: str) -> float:
    prefix = f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    raise AssertionError(f"metric {name} not found in:\n{text}")


def _labeled_metric_value(text: str, name: str, **labels: str) -> int:
    encoded_labels = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    prefix = f"{name}{{{encoded_labels}}} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return int(float(line.removeprefix(prefix)))
    raise AssertionError(f"metric {name} with labels {labels} not found in:\n{text}")


def _sse_payloads(text: str) -> list[dict]:
    payloads = []
    for line in text.splitlines():
        if line == "data: [DONE]" or not line.startswith("data: "):
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads
