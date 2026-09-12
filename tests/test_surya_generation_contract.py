"""Request/resource contract tests for the Surya OCR generators.

Pure unit tests over ``hipengine.generation.surya_contract`` — no model
checkpoint, GPU, or torch required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.deadline import GenerationDeadlineExceeded
from hipengine.generation.registry import GenerationRequest
from hipengine.generation.surya_contract import (
    SuryaGreedySettings,
    SuryaRequestError,
    check_prompt_capacity,
    greedy_decode_tokens,
    resolve_surya_greedy_settings,
)


def _spec() -> SimpleNamespace:
    return SimpleNamespace(eos_token_id=2, vision_spatial_merge_size=2)


def _request(**overrides) -> SimpleNamespace:
    base = dict(
        max_tokens=4,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logit_bias=(),
        suppress_token_ids=(),
        min_tokens=0,
        eos_token_id=None,
        stop_token_ids=(),
        stop_token_sequences=(),
        forced_tokens_pending=(),
        post_thinking_forced_tokens_pending=(),
        force_sequence_completion_token_sequences=(),
        thinking_close_token_ids=(),
        thinking_hard_token_cap=None,
        thinking_soft_close_window=0,
        json_object_close_forcing=False,
        tool_call_constraint=None,
        grammar=None,
        logprobs=False,
        top_logprobs=0,
        kv_storage="auto",
        kv_scale_dtype="fp16",
        kv_scale_granularity="per_token_head",
        ignore_eos=False,
        deadline_at=None,
        cancellation_token=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_defaults_are_accepted() -> None:
    settings = resolve_surya_greedy_settings(_request(), _spec())
    assert settings.max_tokens == 4
    assert settings.ignore_eos is False
    assert settings.eos_token_ids == frozenset({2})
    assert settings.stop_token_ids == frozenset()


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 40),
        ("min_p", 0.1),
        ("repetition_penalty", 1.1),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("logit_bias", ((1, 2.0),)),
        ("suppress_token_ids", (5,)),
        ("min_tokens", 3),
        ("grammar", {"type": "json"}),
        ("json_object_close_forcing", True),
        ("logprobs", True),
        ("kv_storage", "quantized"),
    ],
)
def test_unsupported_controls_are_rejected(field: str, value: object) -> None:
    with pytest.raises(SuryaRequestError, match=field):
        resolve_surya_greedy_settings(_request(**{field: value}), _spec())


def test_supported_controls_are_honored() -> None:
    settings = resolve_surya_greedy_settings(
        _request(ignore_eos=True, eos_token_id=7, stop_token_ids=(9, 11)),
        _spec(),
    )
    assert settings.ignore_eos is True
    assert settings.eos_token_ids == frozenset({7})
    assert settings.stop_token_ids == frozenset({9, 11})


def test_generation_request_roundtrip() -> None:
    request = GenerationRequest(
        prompts=("hi",),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    settings = resolve_surya_greedy_settings(request, _spec())
    assert settings.max_tokens == 8 and settings.ignore_eos is True


def test_capacity_validation() -> None:
    check_prompt_capacity(10, 4, 16)  # exactly fits
    with pytest.raises(SuryaRequestError, match="exceeds"):
        check_prompt_capacity(10, 7, 16)
    with pytest.raises(SuryaRequestError, match="no tokens"):
        check_prompt_capacity(0, 1, 16)


# ---------------------------------------------------------------------------
# budget contract: the registered HIP generator must honor what it advertises
# ---------------------------------------------------------------------------


_MIB = 1024 * 1024


def test_vision_tile_plan_stays_inside_the_budget() -> None:
    """The vision score tile must never exceed the configured memory budget.

    The dense path allocated ``heads * n^2 * 4`` bytes: 56.5 GB for a 300-DPI
    A4 page (220x156 grid, 34320 patches) and 206 GB at the checkpoint's
    ``max_pixels`` ceiling (256x256, 65536 patches). Tiling by query rows makes
    the live tile ``heads * n * block * 4`` instead, and this pins the bound.
    """

    from hipengine.runtime.surya import plan_vision_attention

    heads = 12
    budget = 512 * _MIB
    for n in (256, 4096, 34320, 65536):
        block, scratch = plan_vision_attention(n, heads, budget)
        assert 1 <= block <= n, (n, block)
        assert scratch == heads * n * block * 4, (n, block, scratch)
        assert scratch <= budget, f"grid of {n} patches needs {scratch} > {budget}"

    # the two grids the dense path could not run are now well inside a
    # half-gigabyte tile budget
    assert plan_vision_attention(34320, heads, budget)[1] <= budget
    assert plan_vision_attention(65536, heads, budget)[1] <= budget


def test_vision_tile_plan_uses_one_tile_when_it_fits() -> None:
    """Small grids must keep the single dense tile, not be split for nothing."""

    from hipengine.runtime.surya import plan_vision_attention

    heads = 12
    # 4096 patches x 12 heads x 4 B = 196 KiB per query row; the whole grid is
    # 805 MiB, so a 1 GiB budget keeps it in one tile and a 512 MiB budget must
    # split it
    dense = heads * 4096 * 4096 * 4
    block, scratch = plan_vision_attention(4096, heads, 1024 * _MIB)
    assert block == 4096
    assert scratch == dense

    block, scratch = plan_vision_attention(4096, heads, 512 * _MIB)
    assert 1 <= block < 4096
    assert scratch == heads * 4096 * block * 4 <= 512 * _MIB


def test_vision_tile_plan_honors_a_disabled_budget() -> None:
    from hipengine.runtime.surya import plan_vision_attention

    block, scratch = plan_vision_attention(4096, 12, None)
    assert block == 4096
    assert scratch == 12 * 4096 * 4096 * 4


def test_vision_tile_plan_shrinks_to_one_row_under_a_tiny_budget() -> None:
    from hipengine.runtime.surya import plan_vision_attention

    heads, n = 12, 4096
    block, scratch = plan_vision_attention(n, heads, heads * n * 4)
    assert block == 1
    assert scratch == heads * n * 4
    # below a single row there is nothing left to give: the plan still returns
    # a valid tile and the caller's admission check is what rejects the grid
    block, scratch = plan_vision_attention(n, heads, 1)
    assert block == 1
    assert scratch == heads * n * 4


def test_prefill_tile_plan_stays_inside_the_budget() -> None:
    """The causal prefill score tile must never exceed the configured budget.

    The dense path allocated ``nq * tokens^2 * 4`` bytes: 2.36 GB for the 8580
    image tokens of a 300-DPI A4 page and 8.59 GB at the default 16384-token
    context. Tiling by query rows makes the live tile ``nq * tokens * block *
    4`` instead, and this pins the bound on the token counts the envelope was
    measured at.
    """

    from hipengine.runtime.surya import plan_score_tiles

    heads = 8
    budget = 512 * _MIB
    for tokens in (256, 1024, 8580, 16384):
        block, scratch = plan_score_tiles(tokens, heads, budget)
        assert 1 <= block <= tokens, (tokens, block)
        assert scratch == heads * tokens * block * 4, (tokens, block, scratch)
        assert scratch <= budget, f"{tokens} tokens need {scratch} > {budget}"
        # a prompt whose dense matrix already fits keeps the single dense tile,
        # so the tiled tile can only ever be smaller
        assert scratch <= heads * tokens * tokens * 4

    # the two measured peaks: the 8580-token page and the full 16384-token
    # context, against the dense matrices they used to materialize
    for tokens, dense in ((8580, 2_355_724_800), (16384, 8_589_934_592)):
        _, scratch = plan_score_tiles(tokens, heads, budget)
        assert heads * tokens * tokens * 4 == dense
        assert scratch <= budget < dense


def test_prefill_and_vision_share_one_planner() -> None:
    """Both score matrices are planned by the same arithmetic.

    The vision tower and the text prefill tile the same ``heads * rows * rows``
    matrix, so a fix or a tuning change to one must not leave the other behind.
    """

    from hipengine.runtime.surya import plan_score_tiles, plan_vision_attention

    for rows, heads, budget in (
        (256, 12, 64 * _MIB),
        (4096, 12, 512 * _MIB),
        (16384, 8, 512 * _MIB),
        (16384, 8, None),
        (4096, 12, 1),
    ):
        assert plan_vision_attention(rows, heads, budget) == plan_score_tiles(
            rows, heads, budget
        ), (rows, heads, budget)


def test_prefill_tile_plan_uses_one_tile_when_it_fits() -> None:
    """A prompt whose dense matrix fits the budget must not be split.

    ``block >= tokens`` is what makes the tiled path a strict superset of the
    dense one: one tile with ``bq == tokens`` issues exactly the dense calls.
    """

    from hipengine.runtime.surya import plan_score_tiles

    heads = 8
    tokens = 1024
    dense = heads * tokens * tokens * 4
    block, scratch = plan_score_tiles(tokens, heads, dense)
    assert block == tokens
    assert scratch == dense
    # and one byte less must split it
    block, scratch = plan_score_tiles(tokens, heads, dense - 1)
    assert 1 <= block < tokens
    assert scratch == heads * tokens * block * 4 <= dense - 1


def test_gpu_factory_declares_the_budgets_it_honors() -> None:
    """The registered factory must not swallow capacity configuration.

    ``LLM._factory_capacity_kwargs`` forwards a limit only to a factory that
    declares the parameter by name. ``make_surya_generator_gpu`` used to take
    ``**_kwargs``, so ``LLM(max_sequence_length=...)`` was silently dropped and
    the runner stayed at its 2048-token default — below the 8580 image tokens
    a 300-DPI A4 page needs.
    """

    import inspect

    from hipengine.generation.surya_gpu import make_surya_generator_gpu
    from hipengine.llm import _factory_capacity_kwargs

    parameters = inspect.signature(make_surya_generator_gpu).parameters
    assert "max_sequence_length" in parameters
    assert "vision_max_scratch_bytes" in parameters
    assert "prefill_max_scratch_bytes" in parameters

    forwarded = _factory_capacity_kwargs(
        make_surya_generator_gpu,
        max_sequence_length=16384,
        resident_capacity=None,
        vision_max_scratch_bytes=256 * _MIB,
        prefill_max_scratch_bytes=128 * _MIB,
    )
    assert forwarded["max_sequence_length"] == 16384
    assert forwarded["vision_max_scratch_bytes"] == 256 * _MIB
    assert forwarded["prefill_max_scratch_bytes"] == 128 * _MIB

    # an unset prefill budget is left to the runner default rather than
    # forwarded as an explicit ``None`` (which the runner reads as "no budget")
    forwarded = _factory_capacity_kwargs(
        make_surya_generator_gpu,
        max_sequence_length=None,
        resident_capacity=None,
    )
    assert forwarded == {}


def test_llm_rejects_a_non_positive_score_tile_budget() -> None:
    """``LLM`` validates both score-tile budgets before it loads anything.

    A zero or negative budget would otherwise be read as "no budget" and
    silently restore the quadratic score matrix, which is the failure mode the
    budgets exist to prevent.
    """

    from hipengine.llm import LLM

    with pytest.raises(ValueError, match="prefill_max_scratch_bytes"):
        LLM("datalab-to/surya-ocr-2", prefill_max_scratch_bytes=0)
    with pytest.raises(ValueError, match="vision_max_scratch_bytes"):
        LLM("datalab-to/surya-ocr-2", vision_max_scratch_bytes=-1)


def _settings(**overrides) -> SuryaGreedySettings:
    base = dict(
        max_tokens=4,
        ignore_eos=False,
        eos_token_ids=frozenset({2}),
        stop_token_ids=frozenset(),
    )
    base.update(overrides)
    return SuryaGreedySettings(**base)


def test_greedy_stops_at_eos_without_appending() -> None:
    calls: list[int] = []

    def step_fn(token_id: int, step: int) -> np.ndarray:
        calls.append(token_id)
        return np.array([0.0, 0.0, 5.0, 0.0])  # token 2 = eos

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 0.0, 5.0, 0.0]), _settings(), step_fn, _request()
    )
    assert ids == []
    assert reason == "eos"
    assert calls == []


def test_ignore_eos_continues_past_eos() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:
        return np.array([0.0, 0.0, 5.0, 0.0])

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 0.0, 5.0, 0.0]),
        _settings(ignore_eos=True, max_tokens=3),
        step_fn,
        _request(ignore_eos=True),
    )
    assert ids == [2, 2, 2]
    assert reason == "length"


def test_stop_token_ids_finish_reason() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 5.0])  # token 3

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 0.0, 0.0, 5.0]),
        _settings(stop_token_ids=frozenset({3})),
        step_fn,
        _request(),
    )
    assert ids == []
    assert reason == "stop"


def test_no_trailing_decode_after_final_token() -> None:
    """The last requested token must not trigger an extra decode step."""

    calls: list[int] = []

    def step_fn(token_id: int, step: int) -> np.ndarray:
        calls.append(token_id)
        return np.array([0.0, 5.0, 0.0, 0.0])  # token 1

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 5.0, 0.0, 0.0]), _settings(max_tokens=2), step_fn, _request()
    )
    assert ids == [1, 1]
    assert reason == "length"
    # one step between the two tokens, none after the second
    assert calls == [1]


def test_deadline_is_checked() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:
        return np.array([0.0, 5.0, 0.0, 0.0])

    request = _request(deadline_at=0.0)  # monotonic clock is always past 0
    with pytest.raises(GenerationDeadlineExceeded):
        greedy_decode_tokens(
            np.array([0.0, 5.0, 0.0, 0.0]), _settings(max_tokens=3), step_fn, request
        )


def test_zero_max_tokens_returns_empty() -> None:
    def step_fn(token_id: int, step: int) -> np.ndarray:  # pragma: no cover
        raise AssertionError("must not decode when max_tokens is 0")

    ids, reason = greedy_decode_tokens(
        np.array([0.0, 5.0, 0.0, 0.0]), _settings(max_tokens=0), step_fn, _request()
    )
    assert ids == [] and reason == "length"


# -- Stage-boundary cancellation and deadline checks ------------------------
#
# Both generators must observe an abandoned request before the expensive
# stages, not only per generated token. The vision tower and the prefill are
# the two costs that matter: an already-cancelled request must not pay for
# either, and a cancel that lands during preprocessing or during vision must
# not be paid for by the next stage.


class _StubRunner:
    """Records the device stages a Surya generator asks for."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.max_seq = 4096
        self.max_vision_scratch_bytes = 0
        self.max_prefill_scratch_bytes = 0

    def check_vision_capacity(self, grid_thw: object) -> None:
        self.calls.append("check_vision_capacity")

    def vision_forward(self, pixel_rows: object, grid_thw: object) -> np.ndarray:
        self.calls.append("vision_forward")
        return np.zeros((1, 4), dtype=np.float32)

    def prefill(self, input_ids: object, positions: object, visual_features: object = None) -> np.ndarray:
        self.calls.append("prefill")
        return np.zeros(8, dtype=np.float32)

    def decode_step(self, token_id: int, position: int) -> np.ndarray:
        self.calls.append("decode_step")
        return np.zeros(8, dtype=np.float32)


def _cpu_weights() -> dict[str, np.ndarray]:
    """Minimal weights for the CPU decode tail, so a missing stage check fails
    on the intended assertion rather than on a lookup error."""

    return {
        "model.language_model.embed_tokens.weight": np.zeros((8, 4), dtype=np.float32)
    }


def _gpu_generator(calls: list[str]) -> object:
    """A GPU generator wired to a stub runner, without a checkpoint."""

    from hipengine.generation import surya_gpu

    generator = surya_gpu.SuryaOCRGeneratorGPU.__new__(surya_gpu.SuryaOCRGeneratorGPU)
    generator.runner = _StubRunner(calls)
    generator.spec = _spec()
    generator.tokenizer = None
    generator.max_seq = generator.runner.max_seq
    generator.max_vision_scratch_bytes = 0
    generator.max_prefill_scratch_bytes = 0
    return generator


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, module: object, calls: list[str],
                   *, during_preprocess: object = None,
                   during_vision: object = None) -> None:
    """Replace preprocessing and prompt rendering with recording stubs."""

    def fake_preprocess(image: object) -> tuple[np.ndarray, tuple[int, int, int]]:
        calls.append("preprocess")
        if during_preprocess is not None:
            during_preprocess()
        return np.zeros((4, 1536), dtype=np.float32), (1, 4, 4)

    def fake_render(tokenizer: object, prompt: str, n_image_tokens: int | None = None):
        calls.append("render_prompt")
        return [1, 2, 3], [0, 1, 1]

    monkeypatch.setattr(module, "preprocess_image_surya", fake_preprocess)
    monkeypatch.setattr(module, "render_chat_prompt", fake_render)
    monkeypatch.setattr(
        module,
        "compute_mrope_positions",
        lambda mm, grid, merge=None: np.zeros((3, 3), dtype=np.int64),
    )


def _multimodal_call(generator: object, request: object) -> None:
    generator.generate_multimodal_detailed("prompt", "page.png", request)


def test_gpu_ocr_skips_every_stage_when_already_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import surya_gpu
    from hipengine.generation.deadline import GenerationCancelled, GenerationCancellationToken

    calls: list[str] = []
    _stub_pipeline(monkeypatch, surya_gpu, calls)
    token = GenerationCancellationToken()
    token.cancel()

    with pytest.raises(GenerationCancelled):
        _multimodal_call(
            _gpu_generator(calls), _request(cancellation_token=token)
        )
    assert calls == [], f"a cancelled request still ran {calls}"


def test_gpu_ocr_skips_every_stage_when_the_deadline_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import surya_gpu

    calls: list[str] = []
    _stub_pipeline(monkeypatch, surya_gpu, calls)

    with pytest.raises(GenerationDeadlineExceeded):
        _multimodal_call(_gpu_generator(calls), _request(deadline_at=0.0))
    assert calls == [], f"an expired request still ran {calls}"


def test_gpu_ocr_does_not_start_vision_after_a_cancel_during_preprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import surya_gpu
    from hipengine.generation.deadline import GenerationCancelled, GenerationCancellationToken

    calls: list[str] = []
    token = GenerationCancellationToken()
    _stub_pipeline(monkeypatch, surya_gpu, calls, during_preprocess=token.cancel)

    with pytest.raises(GenerationCancelled):
        _multimodal_call(_gpu_generator(calls), _request(cancellation_token=token))
    assert "preprocess" in calls
    assert "vision_forward" not in calls
    assert "prefill" not in calls


def test_gpu_ocr_does_not_prefill_after_a_cancel_during_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import surya_gpu
    from hipengine.generation.deadline import GenerationCancelled, GenerationCancellationToken

    calls: list[str] = []
    token = GenerationCancellationToken()
    generator = _gpu_generator(calls)
    _stub_pipeline(monkeypatch, surya_gpu, calls)
    original = generator.runner.vision_forward

    def cancel_then_run(pixel_rows: object, grid_thw: object) -> np.ndarray:
        token.cancel()
        return original(pixel_rows, grid_thw)

    monkeypatch.setattr(generator.runner, "vision_forward", cancel_then_run)

    with pytest.raises(GenerationCancelled):
        _multimodal_call(generator, _request(cancellation_token=token))
    assert "vision_forward" in calls
    assert "prefill" not in calls
    assert "decode_step" not in calls


def test_cpu_ocr_skips_every_stage_when_already_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import surya as surya_cpu
    from hipengine.generation.deadline import GenerationCancelled, GenerationCancellationToken

    calls: list[str] = []
    _stub_pipeline(monkeypatch, surya_cpu, calls)
    monkeypatch.setattr(
        "hipengine.kernels.cpu_reference.surya.vision_forward",
        lambda *a, **k: (calls.append("vision_forward"), None, None, np.zeros((1, 4), dtype=np.float32))[1:],
    )
    monkeypatch.setattr(
        surya_cpu, "text_prefill",
        lambda *a, **k: (calls.append("prefill") or np.zeros((1, 1, 4), dtype=np.float32), None),
    )
    generator = surya_cpu.SuryaOCRGenerator.__new__(surya_cpu.SuryaOCRGenerator)
    generator.spec = _spec()
    generator.tokenizer = None
    generator.weights = _cpu_weights()
    token = GenerationCancellationToken()
    token.cancel()

    with pytest.raises(GenerationCancelled):
        _multimodal_call(generator, _request(cancellation_token=token))
    assert calls == [], f"a cancelled request still ran {calls}"


def test_cpu_ocr_does_not_prefill_after_a_cancel_during_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import surya as surya_cpu
    from hipengine.generation.deadline import GenerationCancelled, GenerationCancellationToken

    calls: list[str] = []
    token = GenerationCancellationToken()
    _stub_pipeline(monkeypatch, surya_cpu, calls)

    def fake_vision(weights: object, spec: object, pixel_rows: object, grid_thw: object):
        calls.append("vision_forward")
        token.cancel()
        return None, None, np.zeros((1, 4), dtype=np.float32)

    monkeypatch.setattr(
        "hipengine.kernels.cpu_reference.surya.vision_forward", fake_vision
    )
    monkeypatch.setattr(
        surya_cpu, "text_prefill",
        lambda *a, **k: (calls.append("prefill") or np.zeros((1, 1, 4), dtype=np.float32), None),
    )
    generator = surya_cpu.SuryaOCRGenerator.__new__(surya_cpu.SuryaOCRGenerator)
    generator.spec = _spec()
    generator.tokenizer = None
    generator.weights = _cpu_weights()

    with pytest.raises(GenerationCancelled):
        _multimodal_call(generator, _request(cancellation_token=token))
    assert "vision_forward" in calls
    assert "prefill" not in calls
