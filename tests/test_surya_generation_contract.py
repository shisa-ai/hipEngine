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
    """A grid the envelope covers in one tile keeps the single dense tile."""

    from hipengine.runtime.surya import plan_vision_attention

    heads = 12
    # 128 patches x 12 heads x 4 B = 6 KiB per query row; the whole grid is
    # 768 KiB, so a 1 MiB budget keeps it in one tile and a 512 KiB budget must
    # split it
    dense = heads * 128 * 128 * 4
    block, scratch = plan_vision_attention(128, heads, 1024 * 1024)
    assert block == 128
    assert scratch == dense

    block, scratch = plan_vision_attention(128, heads, 512 * 1024)
    assert 1 <= block < 128
    assert scratch == heads * 128 * block * 4 <= 512 * 1024


def test_vision_tile_plan_is_bounded_by_the_measured_shape_envelope() -> None:
    """The block is a shape choice, not just the widest tile that fits.

    ``scripts/surya_vision_tiling_cost.py`` sweeps the query block on gfx1151.
    At 4096 patches the byte budget alone picks a 2730-row tile (1123.42 ms)
    where a 128-row tile runs 874.38 ms, and at 1024 patches the budget admits
    the whole dense matrix (193.54 ms) where a 128-row tile runs 135.52 ms. The
    envelope that recovers that caps the block at ``max(128, rows/32)`` rows, so
    the budget stays the binding constraint where it matters: at the 34320-patch
    A4 page the cap is 1073 rows and the 512 MiB budget still chooses the shape.
    """

    from hipengine.runtime.surya import plan_vision_attention

    heads = 12
    dense = heads * 4096 * 4096 * 4
    # a budget that admits the whole dense matrix is still split to 32 tiles
    block, scratch = plan_vision_attention(4096, heads, dense)
    assert block == 128
    assert scratch == heads * 4096 * 128 * 4 == dense // 32
    # the 128-row bound binds on the smaller grids rather than rows/32
    assert plan_vision_attention(1024, heads, 512 * _MIB)[0] == 128
    assert plan_vision_attention(256, heads, 512 * _MIB)[0] == 128
    # and rows/32 never binds at page scale, where the budget is narrower
    block, scratch = plan_vision_attention(34320, heads, 512 * _MIB)
    assert block == 320
    assert scratch <= 512 * _MIB


def test_tile_plan_keeps_the_block_on_a_wavefront_multiple() -> None:
    """The tile's query width is a whole number of 32-lane wavefronts.

    On the 34320-patch A4 page every measured block that is a multiple of 32
    runs 43862-44668 ms (256/288/320/352/384/448/480/512) and every block that
    is not runs 45050-48496 ms (272/300/304/325/336/400), so the best
    non-multiple is slower than the worst multiple. The 512 MiB budget derives
    325 rows there, which is the slow side of that line.
    """

    from hipengine.runtime.surya import plan_score_tiles, plan_vision_attention

    for rows, heads, budget in (
        (34320, 12, 512 * _MIB),
        (65536, 12, 512 * _MIB),
        (6400, 12, 512 * _MIB),
        (8580, 8, 512 * _MIB),
        (1000, 8, 512 * _MIB),
        (34320, 12, 700 * _MIB),
    ):
        block, scratch = plan_score_tiles(rows, heads, budget)
        assert block % 32 == 0, (rows, heads, budget, block)
        assert scratch == heads * rows * block * 4 <= budget
        assert plan_vision_attention(rows, heads, budget) == (block, scratch)
    # a budget too small for a whole wavefront still gets a usable tile
    assert plan_score_tiles(4096, 12, 20 * 12 * 4096 * 4)[0] == 20


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
    """A prompt the envelope covers in one tile keeps the single dense tile."""

    from hipengine.runtime.surya import plan_score_tiles

    heads = 8
    tokens = 128
    dense = heads * tokens * tokens * 4
    block, scratch = plan_score_tiles(tokens, heads, dense)
    assert block == tokens
    assert scratch == dense
    # and one byte less must split it
    block, scratch = plan_score_tiles(tokens, heads, dense - 1)
    assert 1 <= block < tokens
    assert scratch == heads * tokens * block * 4 <= dense - 1


def test_prefill_tile_plan_is_bounded_by_the_measured_shape_envelope() -> None:
    """The text prefill shares the vision envelope and wavefront multiple."""

    from hipengine.runtime.surya import plan_score_tiles

    heads = 8
    # 1024 tokens need 32 MiB dense, well inside the default budget, and the
    # envelope still splits them into 8 tiles
    block, scratch = plan_score_tiles(1024, heads, 512 * _MIB)
    assert block == 128
    assert scratch == heads * 1024 * 128 * 4
    # the full 16384-token context: the cap is 512 rows, inside the budget
    block, scratch = plan_score_tiles(16384, heads, 512 * _MIB)
    assert block == 512
    assert scratch == heads * 16384 * 512 * 4 <= 512 * _MIB
    # 8580 tokens: the cap is 269 rows, rounded down to 256
    block, _ = plan_score_tiles(8580, heads, 512 * _MIB)
    assert block == 256
    # and a budget narrower than the envelope still wins
    assert plan_score_tiles(16384, heads, 64 * _MIB)[0] == 128


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
    assert "vision_max_seconds" in parameters

    forwarded = _factory_capacity_kwargs(
        make_surya_generator_gpu,
        max_sequence_length=16384,
        resident_capacity=None,
        vision_max_scratch_bytes=256 * _MIB,
        prefill_max_scratch_bytes=128 * _MIB,
        vision_max_seconds=45.0,
    )
    assert forwarded["max_sequence_length"] == 16384
    assert forwarded["vision_max_scratch_bytes"] == 256 * _MIB
    assert forwarded["prefill_max_scratch_bytes"] == 128 * _MIB
    assert forwarded["vision_max_seconds"] == 45.0

    # an unset prefill budget is left to the runner default rather than
    # forwarded as an explicit ``None`` (which the runner reads as "no budget")
    forwarded = _factory_capacity_kwargs(
        make_surya_generator_gpu,
        max_sequence_length=None,
        resident_capacity=None,
    )
    assert forwarded == {}


def test_llm_rejects_a_non_positive_score_tile_budget() -> None:
    """``LLM`` validates the score-tile and vision time budgets up front.

    A zero or negative score-tile budget would otherwise be read as "no
    budget" and silently restore the quadratic score matrix, which is the
    failure mode the budgets exist to prevent. A zero vision time budget would
    reject every page, including the ones the default admits.
    """

    from hipengine.llm import LLM

    with pytest.raises(ValueError, match="prefill_max_scratch_bytes"):
        LLM("datalab-to/surya-ocr-2", prefill_max_scratch_bytes=0)
    with pytest.raises(ValueError, match="vision_max_scratch_bytes"):
        LLM("datalab-to/surya-ocr-2", vision_max_scratch_bytes=-1)
    with pytest.raises(ValueError, match="vision_max_seconds"):
        LLM("datalab-to/surya-ocr-2", vision_max_seconds=0)
    with pytest.raises(ValueError, match="vision_max_seconds"):
        LLM("datalab-to/surya-ocr-2", vision_max_seconds=-1.0)


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


# -- vision compute-time admission -------------------------------------------
#
# The byte budget bounds the score tile, not the time. At the checkpoint's
# ``max_pixels`` ceiling (a 256x256 patch grid, 65536 patches) one vision
# forward is 1.7e14 FLOPs and about 2.7 minutes on gfx1151, so memory-only
# admission let a page through that nothing downstream could serve.


def _admission_runner(**overrides):
    """A ``SuryaGpuRunner`` carrying only the admission state.

    ``check_vision_capacity`` reads the spec, the two declared budgets, the
    scratch cache, and free device memory; nothing else. Building the runner
    through ``__new__`` keeps these contract tests free of a checkpoint, a
    device, and HIP.
    """

    from types import SimpleNamespace

    from hipengine.kernels.cpu_reference.surya import SuryaSpec
    from hipengine.runtime.surya import SuryaGpuRunner

    runner = SuryaGpuRunner.__new__(SuryaGpuRunner)
    runner.spec = SuryaSpec()
    runner.max_vision_scratch_bytes = 512 * _MIB
    runner.max_vision_seconds = 120.0
    runner._scratch = {}
    runner.runtime = SimpleNamespace(mem_get_info=lambda: (1 << 40, 1 << 40))
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def test_vision_time_estimate_tracks_the_measured_page_costs() -> None:
    """The estimate reproduces the measured production rows on gfx1151.

    Calibration rows are the retained vision-tiling artifacts: the 256/1024/
    4096/6400-patch pages at the 512 MiB budget and the 34320-patch A4 page at
    its production shape (``benchmarks/results/2026-09-13-gfx1151-surya-vision-
    tiling-cost.json`` and ``...-a4-v2.json``). The estimate gates admission
    rather than reporting a benchmark, so the contract is a stated band.
    """

    from hipengine.kernels.cpu_reference.surya import SuryaSpec
    from hipengine.runtime.surya import vision_forward_seconds

    spec = SuryaSpec()
    for patches, tiles, measured in (
        (256, 2, 0.02807),
        (1024, 8, 0.13600),
        (4096, 32, 0.88159),
        (6400, 34, 2.13113),
        (34320, 108, 43.31830),
    ):
        estimate = vision_forward_seconds(spec, patches, tiles)
        assert 0.85 * measured <= estimate <= 1.15 * measured, (
            patches,
            estimate,
            measured,
        )


def test_vision_time_estimate_follows_the_tile_plan() -> None:
    """The estimate is a plan model: more query tiles means more key re-reads.

    On the A4 page the measured curve is 43.3 s at 108 tiles against 49.6 s at
    358 tiles, so the estimate has to move with ``max_vision_scratch_bytes``
    rather than being a function of the patch count alone. It is also monotone
    in the patch count, which is what lets the budget be inverted.
    """

    from hipengine.kernels.cpu_reference.surya import SuryaSpec
    from hipengine.runtime.surya import plan_vision_attention, vision_forward_seconds

    spec = SuryaSpec()
    heads = spec.vision_num_heads
    patches = 34320
    wide = plan_vision_attention(patches, heads, 512 * _MIB)[0]
    # the sweep's 96-row row, whose budget is exactly one 96-row tile
    narrow = plan_vision_attention(patches, heads, 96 * heads * patches * 4)[0]
    assert (wide, narrow) == (320, 96)
    wide_s = vision_forward_seconds(spec, patches, -(-patches // wide))
    narrow_s = vision_forward_seconds(spec, patches, -(-patches // narrow))
    assert narrow_s > wide_s, (narrow_s, wide_s)
    # measured: 43.32 s at 108 tiles, 49.60 s at 358 tiles
    assert 40.0 <= wide_s <= 46.0
    assert 47.0 <= narrow_s <= 52.0
    # monotone in the patch count at a fixed plan
    estimates = [
        vision_forward_seconds(spec, n, -(-n // 128)) for n in (256, 1024, 4096, 16384, 65536)
    ]
    assert estimates == sorted(estimates)


def test_vision_time_budget_rejects_the_checkpoint_ceiling() -> None:
    """The 65536-patch ceiling passes the byte budget but not the time budget.

    ``SURYA_MAX_PIXELS`` admits a 256x256 patch grid on memory grounds (502 MB
    of score tile at the default budget) and the estimate puts it at ~161 s, so
    the default budget has to reject it while still admitting the 34320-patch
    A4 page that the benchmark suite measures.
    """

    from hipengine.runtime.surya import DEFAULT_MAX_VISION_SECONDS

    runner = _admission_runner()
    assert runner.max_vision_seconds == DEFAULT_MAX_VISION_SECONDS
    a4 = runner.vision_time_seconds([(1, 220, 156)])
    ceiling = runner.vision_time_seconds([(1, 256, 256)])
    assert a4 < DEFAULT_MAX_VISION_SECONDS < ceiling, (a4, ceiling)
    assert 40.0 <= a4 <= 47.0
    assert 150.0 <= ceiling <= 170.0


def test_check_vision_capacity_rejects_a_page_over_the_time_budget() -> None:
    """An over-large page is rejected by admission, with the estimate named."""

    from hipengine.runtime.surya import SuryaGpuRuntimeError

    runner = _admission_runner()
    runner.check_vision_capacity([(1, 220, 156)])  # A4: ~43 s, inside 120 s
    with pytest.raises(SuryaGpuRuntimeError, match="vision time budget"):
        runner.check_vision_capacity([(1, 256, 256)])
    with pytest.raises(SuryaGpuRuntimeError, match="161"):
        runner.check_vision_capacity([(1, 256, 256)])
    with pytest.raises(SuryaGpuRuntimeError, match="max_vision_seconds"):
        runner.check_vision_capacity([(1, 256, 256)])
    # the rejection is still free: nothing was allocated
    assert runner._scratch == {}


def test_check_vision_capacity_time_budget_can_be_raised_or_disabled() -> None:
    """``max_vision_seconds`` is a declared budget, not a hard ceiling."""

    runner = _admission_runner(max_vision_seconds=200.0)
    runner.check_vision_capacity([(1, 256, 256)])
    runner = _admission_runner(max_vision_seconds=None)
    runner.check_vision_capacity([(1, 256, 256)])
    runner = _admission_runner(max_vision_seconds=1.0)
    with pytest.raises(Exception, match="vision time budget"):
        runner.check_vision_capacity([(1, 64, 64)])


def test_vision_patch_ceiling_inverts_the_time_budget() -> None:
    """The rejection can name the largest page the budget admits."""

    from hipengine.kernels.cpu_reference.surya import SuryaSpec
    from hipengine.runtime.surya import plan_vision_attention, vision_forward_seconds

    runner = _admission_runner()
    spec = SuryaSpec()
    heads = spec.vision_num_heads
    ceiling = runner.vision_patch_ceiling()
    assert ceiling is not None and 0 < ceiling < 65536

    def estimate(n: int) -> float:
        block = plan_vision_attention(n, heads, runner.max_vision_scratch_bytes)[0]
        return vision_forward_seconds(spec, n, -(-n // block))

    assert estimate(ceiling) <= runner.max_vision_seconds
    assert estimate(ceiling + 1) > runner.max_vision_seconds
    # an unbounded budget has no ceiling; a millisecond admits one tiny tile
    assert _admission_runner(max_vision_seconds=None).vision_patch_ceiling() is None
    assert runner.vision_patch_ceiling(seconds=0.0) == 0
    tiny = _admission_runner(max_vision_seconds=0.001).vision_patch_ceiling()
    assert tiny is not None and 0 < tiny < 64

    # Below the shape envelope's crossing the tile is envelope-chosen, the
    # estimate dips by under 2% at each tile-count step, and the bisection can
    # stop slightly short of the true maximum -- but it is never unsound, and
    # never short by more than the documented band.
    small = _admission_runner(max_vision_seconds=1.0)
    short = small.vision_patch_ceiling()
    assert short is not None and estimate(short) <= 1.0
    exact = max(n for n in range(1, 20001) if estimate(n) <= 1.0)
    assert 0.98 * exact <= short <= exact


def test_llm_forwards_the_vision_time_budget() -> None:
    """The public budget reaches a factory that declares it, and only then.

    Same opt-in rule as the score-tile budgets: a factory that does not declare
    ``vision_max_seconds`` keeps its own default, and an unset value is not
    forwarded at all so the generator default (not "unbounded") applies.
    """

    from hipengine.llm import _factory_capacity_kwargs

    def old(*, model_path, weight_index, model_plugin):
        pass

    def new(
        *,
        model_path,
        weight_index,
        model_plugin,
        max_sequence_length=None,
        vision_max_scratch_bytes=None,
        vision_max_seconds=None,
    ):
        pass

    base = {"max_sequence_length": 8192, "resident_capacity": 1}
    assert _factory_capacity_kwargs(old, **base) == {}
    assert _factory_capacity_kwargs(new, **base, vision_max_seconds=30.0) == {
        "max_sequence_length": 8192,
        "vision_max_seconds": 30.0,
    }
    assert _factory_capacity_kwargs(new, **base, vision_max_seconds=None) == {
        "max_sequence_length": 8192
    }
    # `math.inf` is how a caller disables the budget through the public API
    forwarded = _factory_capacity_kwargs(new, **base, vision_max_seconds=float("inf"))
    assert forwarded["vision_max_seconds"] == float("inf")
