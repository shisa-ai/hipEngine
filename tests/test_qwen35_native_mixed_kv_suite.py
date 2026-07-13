from __future__ import annotations

import numpy as np

from hipengine.kvcache import resolve_kv_policy
from scripts.qwen35_gguf_kv_asymmetric_suite import PromptCase
from scripts.qwen35_native_mixed_kv_suite import (
    PolicyRun,
    PromptRun,
    _engine_summary,
    _teacher_inputs,
)


def _case(prompt_id: str = "p0") -> PromptCase:
    return PromptCase(
        prompt_id=prompt_id,
        category="code",
        split="train",
        profile="natural_corpus_v1",
        tokens=(1, 2, 3),
        token_sha256="abc",
        source_prompt_sha256="def",
        current_query_tokens=2,
    )


def _prompt_run(prompt_id: str, logits: list[list[float]], generated: list[int]) -> PromptRun:
    return PromptRun(
        prompt_id=prompt_id,
        logits=np.asarray(logits, dtype=np.float32),
        generated_token_ids=generated,
        decode_input_ids=generated[:-1],
        elapsed_seconds=1.0,
    )


def test_teacher_inputs_use_reference_seed_and_prior_decode_tokens() -> None:
    run = _prompt_run("p0", [[0.0, 1.0]] * 4, [10, 11, 12, 13])

    assert _teacher_inputs(run, 3) == [10, 11, 12]


def test_engine_summary_requires_every_prompt_and_layout_audit() -> None:
    cases = [_case("p0"), _case("p1")]
    reference_policy = resolve_kv_policy("bf16")
    candidate_policy = resolve_kv_policy("tail4_hadamard_group32")
    reference = PolicyRun(
        policy=reference_policy,
        prompts={
            "p0": _prompt_run("p0", [[0.0, 4.0], [0.0, 3.0]], [1, 1]),
            "p1": _prompt_run("p1", [[4.0, 0.0], [3.0, 0.0]], [0, 0]),
        },
        elapsed_seconds=2.0,
        layout_audit={"passed": True},
    )
    candidate = PolicyRun(
        policy=candidate_policy,
        prompts={
            "p0": _prompt_run("p0", [[0.0, 3.9], [0.0, 2.9]], [1, 1]),
            "p1": _prompt_run("p1", [[3.9, 0.0], [2.9, 0.0]], [0, 0]),
        },
        elapsed_seconds=2.0,
        layout_audit={"passed": True},
    )

    summary = _engine_summary(
        engine="paro",
        cases=cases,
        reference=reference,
        candidate=candidate,
        kl_threshold=0.05,
        top1_threshold=0.90,
    )

    assert summary["passed"] is True
    assert summary["summary"]["prompt_count"] == 2
    assert summary["summary"]["positions"] == 4
    assert summary["summary"]["failing_prompt_ids"] == []

    candidate.layout_audit = {"passed": False}
    failed = _engine_summary(
        engine="paro",
        cases=cases,
        reference=reference,
        candidate=candidate,
        kl_threshold=0.05,
        top1_threshold=0.90,
    )
    assert failed["passed"] is False
