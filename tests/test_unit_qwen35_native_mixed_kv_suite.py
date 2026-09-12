from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.kvcache import resolve_kv_policy
from scripts.qwen35_gguf_kv_asymmetric_suite import PromptCase
from scripts.qwen35_native_mixed_kv_suite import (
    PolicyRun,
    PromptRun,
    _build_parser,
    _engine_summary,
    _gguf_layout_audit,
    _parse_layer_indices,
    _select_prompt_cases,
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


def test_parser_accepts_gfx1151_uniform_int8_candidate() -> None:
    args = _build_parser().parse_args(
        [
            "--engine",
            "gguf",
            "--backend",
            "hip_gfx1151",
            "--candidate-kv-storage",
            "int8_per_token_head",
            "--kv-scale-dtype",
            "fp32",
            "--prompt-id",
            "mixed_v1",
        ]
    )

    assert args.backend == "hip_gfx1151"
    assert args.candidate_kv_storage == "int8_per_token_head"
    assert args.kv_scale_dtype == "fp32"
    assert args.prompt_id == ["mixed_v1"]


def test_parser_accepts_forced_long_uniform_int8_layout_audit() -> None:
    args = _build_parser().parse_args(
        [
            "--engine",
            "gguf",
            "--backend",
            "hip_gfx1151",
            "--candidate-kv-storage",
            "int8_per_token_head",
            "--max-sequence-length",
            "65792",
            "--require-no-bf16-mirror",
            "--expected-bf16-full-layers",
            "0-7",
            "--expected-int8-full-layers",
            "8,9",
        ]
    )

    assert args.max_sequence_length == 65_792
    assert args.require_no_bf16_mirror is True
    assert _parse_layer_indices(args.expected_bf16_full_layers) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert _parse_layer_indices(args.expected_int8_full_layers) == [8, 9]


def test_select_prompt_cases_requires_known_unique_ids() -> None:
    cases = [_case("p0"), _case("mixed_v1")]

    assert [case.prompt_id for case in _select_prompt_cases(cases, [])] == ["p0", "mixed_v1"]
    assert [case.prompt_id for case in _select_prompt_cases(cases, ["mixed_v1"])] == [
        "mixed_v1"
    ]
    for prompt_ids in (["missing"], ["p0", "p0"]):
        try:
            _select_prompt_cases(cases, prompt_ids)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid prompt selection: {prompt_ids!r}")


def test_parse_layer_indices_rejects_duplicates_and_descending_ranges() -> None:
    assert _parse_layer_indices("0-2,4,6-7") == [0, 1, 2, 4, 6, 7]
    assert _parse_layer_indices("none") == []

    for value in ("2,2", "7-3", "-1", "1,,2"):
        try:
            _parse_layer_indices(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid layer expression: {value!r}")


def test_uniform_int8_layout_audit_requires_fixed_layers_and_no_mirror() -> None:
    buffer = SimpleNamespace(nbytes=1024)
    per_token = SimpleNamespace(granularity="per_token_head")
    scratch = SimpleNamespace(
        kv_storage_layout="uniform",
        full_key_caches=[buffer] * 10,
        full_value_caches=[buffer] * 10,
        full_kv_scale_metadata=[None] * 8 + [per_token] * 2,
        full_k_scale_caches=[None] * 8 + [buffer] * 2,
        full_v_scale_caches=[None] * 8 + [buffer] * 2,
        full_bf16_mirror_key_caches=[None] * 10,
        full_bf16_mirror_value_caches=[None] * 10,
    )
    session = SimpleNamespace(scratch=scratch, _int8_prefill_oracle_buffers={})
    policy = resolve_kv_policy("int8_per_token_head", scale_dtype="fp16")

    audit = _gguf_layout_audit(
        session,
        policy=policy,
        expected_bf16_layers=range(8),
        expected_int8_layers=(8, 9),
        require_no_bf16_mirror=True,
    )

    assert audit["passed"] is True
    assert audit["fixed_layer_policy_passed"] is True
    assert audit["bf16_mirror_requirement_passed"] is True
    assert audit["bf16_full_attention_indices"] == list(range(8))
    assert audit["int8_full_attention_indices"] == [8, 9]

    scratch.full_bf16_mirror_key_caches[-1] = buffer
    mirrored = _gguf_layout_audit(
        session,
        policy=policy,
        expected_bf16_layers=range(8),
        expected_int8_layers=(8, 9),
        require_no_bf16_mirror=True,
    )
    assert mirrored["passed"] is False
    assert mirrored["bf16_mirror_requirement_passed"] is False


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
