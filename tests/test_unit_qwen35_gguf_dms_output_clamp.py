"""Compact DMS serving sizes its store for the context and clamps outputs to it.

An exact-budget DMS store keeps the newest window plus one in ``ratio`` older
rows. Sizing that store for the served context, instead of full retention,
roughly doubles the 24 GB context; a request's output is then clamped so its
retained prompt plus output still fits.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import engine_loop, qwen35_gguf
from hipengine.kvcache.dms import build_dms_exact_budget_eviction


@pytest.mark.parametrize("tokens", [1, 9, 10, 11, 12, 37, 100])
@pytest.mark.parametrize("window, ratio", [(8, 2), (0, 3), (16, 1)])
def test_retained_rows_match_the_exact_budget_eviction(tokens, window, ratio) -> None:
    logits = np.random.default_rng(tokens).normal(size=(tokens, 1, 1)).astype(np.float32)
    evict = build_dms_exact_budget_eviction(
        logits, current_position=tokens - 1, window_size=window, target_compression_ratio=ratio,
    )
    kept = tokens - int(evict.sum())
    assert qwen35_gguf._dms_exact_budget_retained_rows(tokens, window=window, ratio=ratio) == kept


def _generator(*, retention=(8192, 2), context=229_376, dms=True):
    generator = qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )
    generator.dms_metadata_path = "/tmp/dms_metadata.json" if dms else None
    generator._prepared_max_sequence_length = context
    generator.__dict__["_dms_retention_cache"] = (retention,)
    return generator


def test_output_cap_is_the_store_left_after_the_retained_prompt() -> None:
    generator = _generator()
    rows = lambda n: qwen35_gguf._dms_exact_budget_retained_rows(n, window=8192, ratio=2)
    assert generator.dms_output_token_cap(0) == rows(229_376)
    assert generator.dms_output_token_cap(151_823) == rows(229_376) - rows(151_823)
    # A prompt filling the context still gets at least one output token.
    assert generator.dms_output_token_cap(229_376) == 1


def test_unbounded_retention_does_not_clamp() -> None:
    assert _generator(retention=None).dms_output_token_cap(0) is None


@pytest.mark.parametrize("requested, prompt, expected_cap", [(64, 1000, False), (200_000, 1000, True)])
def test_resident_runner_clamps_only_dms_requests(requested, prompt, expected_cap) -> None:
    generator = _generator()
    runner = SimpleNamespace(generator=generator)
    hook = qwen35_gguf.Qwen35GGUFResidentModelRunner.scheduler_max_new_tokens_for_prompt
    runner.scheduler_max_new_tokens = lambda request: int(request.max_tokens)
    request = SimpleNamespace(max_tokens=requested)
    limit = hook(runner, request, tuple(range(prompt)))
    cap = generator.dms_output_token_cap(prompt)
    assert limit == (cap if expected_cap else requested)
    generator.dms_metadata_path = None
    assert hook(runner, request, tuple(range(prompt))) == requested


def test_engine_loop_asks_for_a_per_prompt_limit_when_offered() -> None:
    seen = []
    runner = SimpleNamespace(
        scheduler_max_new_tokens=lambda request: 100,
        scheduler_max_new_tokens_for_prompt=lambda request, row: seen.append(len(row)) or 7,
    )
    assert engine_loop._runner_max_new_tokens(runner, object(), (1, 2, 3)) == 7
    assert seen == [3]
    plain = SimpleNamespace(scheduler_max_new_tokens=lambda request: 100)
    assert engine_loop._runner_max_new_tokens(plain, object(), (1, 2, 3)) == 100


def test_banner_announces_the_clamp() -> None:
    from hipengine.server.api import _dms_output_limit_text

    generator = _generator()
    text = _dms_output_limit_text(SimpleNamespace(dms_output_token_cap=generator.dms_output_token_cap), 229_376)
    assert "max_tokens is clamped" in text
    assert f"{generator.dms_output_token_cap(0):,} tokens after a short prompt" in text
    assert _dms_output_limit_text(SimpleNamespace(dms_output_token_cap=lambda n: None), 4096) == ""
    assert _dms_output_limit_text(SimpleNamespace(), 4096) == ""
