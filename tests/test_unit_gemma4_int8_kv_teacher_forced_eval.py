"""Unit tests for the D12 INT8-vs-BF16 teacher-forced evaluation harness.

The harness reuses the frozen evaluator in
``scripts/gemma4_teacher_forced_gate.py`` (full-vocabulary float64 KL,
nearest-rank tails, the binding ``docs/EXECUTION-PROFILES.md`` limits) rather
than inventing a second metric. These tests pin the reused evaluator's input
contracts through the harness module, then cover the harness's own logic: the
frozen workload is deterministic and spans both sides of the shipping prefill
block, a small screening chain requires zero top-1 flips, owned allocations are
read from live buffers rather than inferred, route witnesses aggregate the
registered selections, and a repeated capture is a literal replay.

No GPU is required; the module keeps its device imports inside functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.gemma4_int8_kv_teacher_forced_eval import (
    CONTRACT_SOURCE,
    EXPECTED_CONSUMER_DECODE,
    EXPECTED_CONSUMER_PREFILL,
    EXPECTED_WRITER_DECODE,
    EXPECTED_WRITER_PROMPT,
    THRESHOLDS,
    FrozenCase,
    _execution_identity,
    check_repeatability,
    evaluate,
    evaluate_cases,
    evaluate_controls,
    expected_block_plan,
    freeze_workload,
    owned_allocation_bytes,
    public_witness_verdict,
    quality_passed,
    summarize_route_witness,
)


def _fake_tokenize(text: str) -> list[int]:
    """Deterministic, corpus-sensitive tokenizer with no model dependency."""

    return [ord(character) % 251 for character in text]


class _Buffer:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = int(nbytes)


class _Owner:
    def __init__(self, allocated: int) -> None:
        self.allocated_bytes = int(allocated)


class _FakeRunner:
    def __init__(self, buffers, caches=(), int8_owner=None) -> None:
        self._buffers = list(buffers)
        self._caches = list(caches)
        self._int8_owner = int8_owner

    @property
    def uses_int8_kv(self) -> bool:
        return self._int8_owner is not None

    @property
    def kv_cache(self):
        return self._int8_owner


class TestExecutionIdentity:
    def test_records_actual_public_profile_and_manifest_properties(self) -> None:
        from types import SimpleNamespace

        manifest = {"execution_profile": "production", "variants": []}
        llm = SimpleNamespace(
            execution_profile=None,
            resolved_execution_profile="production",
            execution_profile_manifest=manifest,
            execution_profile_manifest_sha256="selected-hash",
            execution_profile_strict_manifest_sha256="strict-hash",
            execution_profile_fell_back_to_strict=False,
        )
        result = _execution_identity(llm, object(), object())
        assert result["requested_profile"] is None
        assert result["resolved_profile"] == {
            "attribute": "resolved_execution_profile", "value": "production"
        }
        assert result["manifest"]["value"] == manifest
        assert result["manifest_sha256"] == "selected-hash"
        assert result["strict_manifest_sha256"] == "strict-hash"
        assert result["fell_back_to_strict"] is False
        assert result["legacy_path"] is False

    def test_migration_is_distinct_from_unavailable_profile_api(self) -> None:
        from types import SimpleNamespace

        migration = SimpleNamespace(
            execution_profile=None,
            resolved_execution_profile=None,
            execution_profile_manifest=None,
            execution_profile_manifest_sha256=None,
            execution_profile_strict_manifest_sha256=None,
            execution_profile_fell_back_to_strict=None,
        )
        result = _execution_identity(migration, object(), object())
        assert result["legacy_path"] is True
        assert result["resolved_profile_status"] == "migration: no resolved named profile"
        assert result["manifest_status"] == "migration: no resolved variant manifest"
        unavailable = _execution_identity(object(), object(), object())
        assert unavailable["legacy_path"] is None
        assert unavailable["resolved_profile_status"].startswith("unavailable:")


class TestEvaluatorReuse:
    """The harness must expose the binding evaluator unchanged."""

    def test_thresholds_are_the_binding_production_limits(self) -> None:
        assert THRESHOLDS == {
            "kl_mean": 1e-3,
            "kl_p95": 5e-3,
            "kl_p99": 2e-2,
            "kl_max": 5e-2,
            "top1_rate": 0.99,
        }
        assert "EXECUTION-PROFILES" in CONTRACT_SOURCE

    def test_non_finite_logits_are_rejected(self) -> None:
        base = np.zeros((4, 8), dtype=np.float32)
        cand = base.copy()
        cand[1, 2] = np.inf
        with pytest.raises(ValueError, match="finite"):
            evaluate(base, cand)

    def test_row_and_vocabulary_mismatch_are_rejected(self) -> None:
        base = np.zeros((4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="row"):
            evaluate(base, base[:3])
        with pytest.raises(ValueError, match="vocab"):
            evaluate(base, base[:, :4])


class TestFreezeWorkload:
    def test_workload_is_deterministic_and_hashes_are_stable(self) -> None:
        first = freeze_workload(_fake_tokenize, max_block=1024)
        second = freeze_workload(_fake_tokenize, max_block=1024)
        assert [case.prompt_ids for case in first] == [
            case.prompt_ids for case in second
        ]
        assert [case.chain_sha256 for case in first] == [
            case.chain_sha256 for case in second
        ]
        assert len({case.name for case in first}) == len(first)
        assert len({case.category for case in first}) == len(first)

    def test_cases_span_both_sides_of_the_shipping_prefill_block(self) -> None:
        max_block = 1024
        cases = {case.name: case for case in freeze_workload(_fake_tokenize, max_block=max_block)}
        prompts = sorted(case.prompt_tokens for case in cases.values())
        assert any(case.prompt_tokens < max_block for case in cases.values())
        assert any(case.prompt_tokens > max_block for case in cases.values())
        # One unrelated length, not adjacent to the block boundary.
        assert any(abs(case.prompt_tokens - max_block) > 128 for case in cases.values())
        assert prompts[0] != prompts[-1]

    def test_scored_rows_equal_prompt_minus_prefill(self) -> None:
        for case in freeze_workload(_fake_tokenize, max_block=512):
            assert case.prompt_tokens == len(case.prompt_ids)
            # ``capture_chain`` scores prompt_tokens - 1 - prefill rows.
            assert case.scored_rows == case.prompt_tokens - 1 - case.prefill
            assert case.scored_rows > 0

    def test_a_block_too_small_for_the_design_is_refused(self) -> None:
        with pytest.raises(ValueError):
            freeze_workload(_fake_tokenize, max_block=32)


class TestEvaluateCases:
    def _rows(self, rows: int = 16, vocab: int = 12, seed: int = 3) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(rows, vocab)).astype(np.float32)

    def test_identical_arms_pass_with_zero_flips(self) -> None:
        order = ("a", "b")
        base = {name: self._rows(seed=i) for i, name in enumerate(order)}
        report = evaluate_cases(base, {k: v.copy() for k, v in base.items()}, order=order)
        assert report["combined"]["passed"] is True
        assert report["combined"]["top1_flips"] == 0
        for name in order:
            assert report["per_case"][name]["passed"] is True
        assert report["cases"] == list(order)

    def test_a_single_flip_in_a_small_screen_fails_even_at_high_agreement(self) -> None:
        base = self._rows(rows=100, vocab=4, seed=5)
        cand = base.copy()
        # Force exactly one argmax flip.
        cand[0] = np.roll(cand[0], 1)
        cand[0, 0] += 50.0
        report = evaluate_cases({"case": base}, {"case": cand}, order=("case",))
        case = report["per_case"]["case"]
        assert case["top1_flips"] == 1
        assert case["top1_rate"] == pytest.approx(0.99)
        assert case["passed"] is False
        assert "screen_top1_flips" in case["failed"]

    def test_combined_rows_concatenate_in_order(self) -> None:
        order = ("first", "second")
        base = {"first": self._rows(rows=5, seed=1), "second": self._rows(rows=7, seed=2)}
        cand = {k: v.copy() for k, v in base.items()}
        report = evaluate_cases(base, cand, order=order)
        assert report["combined"]["rows"] == 12


class TestOwnedAllocationBytes:
    def test_bf16_caches_are_reported_once(self) -> None:
        runner = _FakeRunner(
            buffers=[_Buffer(100), _Buffer(200)],
            caches=[_Buffer(100), _Buffer(200)],
        )
        report = owned_allocation_bytes(runner)
        assert report["runner_owned_bytes"] == 300
        assert report["kv_owned_bytes"] == 300
        # The BF16 caches are a subset of the runner buffers, never double counted.
        assert report["owned_total_bytes"] == 300

    def test_int8_owner_is_added_to_the_runner_buffers(self) -> None:
        runner = _FakeRunner(buffers=[_Buffer(100)], int8_owner=_Owner(4096))
        report = owned_allocation_bytes(runner)
        assert report["runner_owned_bytes"] == 100
        assert report["kv_owned_bytes"] == 4096
        assert report["owned_total_bytes"] == 4196
        assert "owner" in report["kv_scope"]

    def test_scope_names_allocator_arithmetic_not_total_vram(self) -> None:
        runner = _FakeRunner(buffers=[_Buffer(8)], int8_owner=_Owner(16))
        report = owned_allocation_bytes(runner)
        scope = report["scope"].lower()
        assert "allocator" in scope
        # The scope must explicitly deny being a total-VRAM figure.
        assert "not a measured total vram" in scope


class TestRouteWitness:
    def test_counts_separate_prefill_and_decode_selections(self) -> None:
        events = [
            {"rows": 1024, "writer": "w/prompt", "consumer": "c/prefill"},
            {"rows": 64, "writer": "w/prompt", "consumer": "c/prefill"},
            {"rows": 1, "writer": "w/decode", "consumer": "c/decode"},
            {"rows": 1, "writer": "w/decode", "consumer": "c/decode"},
        ]
        report = summarize_route_witness(events)
        assert report["calls"] == 4
        assert report["prefill_calls"] == 2
        assert report["decode_calls"] == 2
        assert report["prefill_row_histogram"] == {"1024": 1, "64": 1}
        assert report["writer_counts"] == {"w/prompt": 2, "w/decode": 2}

    def test_empty_witness_is_reported_honestly(self) -> None:
        report = summarize_route_witness([])
        assert report["calls"] == 0
        assert report["prefill_calls"] == 0
        assert report["decode_calls"] == 0


# --- repaired controls -------------------------------------------------------

_MAX_BLOCK = 256
_NUM_LAYERS = 4


def _frozen_case(name: str, prefill: int, scored_rows: int) -> FrozenCase:
    prompt_tokens = prefill + scored_rows + 1
    return FrozenCase(
        name=name,
        category="test",
        prompt_ids=tuple(range(prompt_tokens)),
        prefill=prefill,
        scored_rows=scored_rows,
        chain_sha256="0" * 64,
        prompt_tokens=prompt_tokens,
    )


def _control_events(case, *, drop_last_prefill_block=False, wrong_prefill_writer=None,
                    swap_block_order=False, swap_block_widths=False, extra_zero_row=False,
                    max_block=_MAX_BLOCK):
    plan = expected_block_plan(case, max_block)
    schedule = [dict(block) for block in plan["schedule"]]
    if drop_last_prefill_block and plan["num_prefill_blocks"] > 1:
        cut = plan["num_prefill_blocks"] - 1
        schedule = schedule[:cut] + schedule[cut + 1:]
    if swap_block_order and len(schedule) > 1:
        schedule[0], schedule[1] = schedule[1], schedule[0]
    if swap_block_widths and len(schedule) > 1:
        schedule[0]["rows"], schedule[1]["rows"] = schedule[1]["rows"], schedule[0]["rows"]
    route_events = []
    block_events = []
    for block in schedule:
        block_events.append({"write_offset": block["write_offset"], "rows": block["rows"]})
        if block["rows"] > 1:
            writer, consumer = EXPECTED_WRITER_PROMPT, EXPECTED_CONSUMER_PREFILL
        else:
            writer, consumer = EXPECTED_WRITER_DECODE, EXPECTED_CONSUMER_DECODE
        for _ in range(_NUM_LAYERS):
            route_events.append({"rows": block["rows"], "writer": writer, "consumer": consumer})
    if wrong_prefill_writer is not None:
        route_events[wrong_prefill_writer]["writer"] = EXPECTED_WRITER_DECODE
    if extra_zero_row:
        route_events.append(
            {"rows": 0, "writer": EXPECTED_WRITER_DECODE, "consumer": EXPECTED_CONSUMER_DECODE}
        )
    return route_events, block_events


def _controls(workload, per_case_events):
    route_events = {case.name: per_case_events[case.name][0] for case in workload}
    block_events = {case.name: per_case_events[case.name][1] for case in workload}
    return evaluate_controls(
        workload, max_block=_MAX_BLOCK, num_layers=_NUM_LAYERS,
        route_events=route_events, block_events=block_events,
    )


class TestExpectedBlockPlan:
    def test_above_block_case_crosses_into_a_nonzero_prefill_offset(self) -> None:
        case = _frozen_case("above", prefill=300, scored_rows=2)
        plan = expected_block_plan(case, _MAX_BLOCK)
        assert plan["crosses_block_boundary"] is True
        assert plan["prefill_blocks"] == [
            {"write_offset": 0, "rows": 256},
            {"write_offset": 256, "rows": 44},
        ]
        assert plan["decode_blocks"] == [
            {"write_offset": 300, "rows": 1},
            {"write_offset": 301, "rows": 1},
        ]
        assert plan["schedule"] == plan["prefill_blocks"] + plan["decode_blocks"]

    def test_below_block_case_is_a_single_prefill_block(self) -> None:
        case = _frozen_case("below", prefill=100, scored_rows=2)
        plan = expected_block_plan(case, _MAX_BLOCK)
        assert plan["crosses_block_boundary"] is False
        assert plan["prefill_blocks"] == [{"write_offset": 0, "rows": 100}]

    def test_scored_rows_94_leaves_a_one_row_second_prefill_chunk(self) -> None:
        # 1025 = 1024 + 1: the second prefill block carries a single row, which
        # is a decode-shaped launcher that is still part of prefill scheduling.
        case = _frozen_case("above94", prefill=1025, scored_rows=94)
        plan = expected_block_plan(case, 1024)
        assert plan["crosses_block_boundary"] is True
        assert plan["prefill_blocks"] == [
            {"write_offset": 0, "rows": 1024},
            {"write_offset": 1024, "rows": 1},
        ]
        assert plan["num_prefill_blocks"] == 2
        assert plan["num_decode_blocks"] == 94
        assert plan["schedule"][0] == {"write_offset": 0, "rows": 1024}
        assert plan["schedule"][1] == {"write_offset": 1024, "rows": 1}
        assert plan["schedule"][2] == {"write_offset": 1025, "rows": 1}


class TestFreezeWorkloadBoundary:
    def test_scored_rows_94_still_places_an_above_block_case(self) -> None:
        cases = {
            case.name: case
            for case in freeze_workload(_fake_tokenize, max_block=1024, scored_rows=94)
        }
        above = cases["prose_ja_above_block"]
        assert above.prefill > 1024
        plan = expected_block_plan(above, 1024)
        assert plan["crosses_block_boundary"] is True
        assert plan["num_prefill_blocks"] == 2

    def test_scored_rows_95_is_rejected_before_capture(self) -> None:
        with pytest.raises(ValueError, match="does not cross max_block"):
            freeze_workload(_fake_tokenize, max_block=1024, scored_rows=95)


class TestEvaluateControls:
    def _workload(self):
        return [
            _frozen_case("below", prefill=100, scored_rows=2),
            _frozen_case("above", prefill=300, scored_rows=2),
        ]

    def test_a_correct_plan_passes(self) -> None:
        workload = self._workload()
        events = {case.name: _control_events(case) for case in workload}
        verdict = _controls(workload, events)
        assert verdict["passed"] is True, verdict["failed"]
        assert verdict["crossing_cases"] == ["above"]

    def test_scored_rows_94_one_row_chunk_passes_the_full_schedule(self) -> None:
        case = _frozen_case("above94", prefill=1025, scored_rows=94)
        routes, blocks = _control_events(case, max_block=1024)
        verdict = evaluate_controls(
            [case], max_block=1024, num_layers=_NUM_LAYERS,
            route_events={case.name: routes}, block_events={case.name: blocks},
        )
        assert verdict["passed"] is True, verdict["failed"]
        # The one-row prefill chunk uses the decode keys but stays a prefill block.
        one_row_prefill = verdict["per_case"]["above94"]["expected_schedule"][1]
        assert one_row_prefill == {"write_offset": 1024, "rows": 1}

    def test_a_wrong_writer_choice_fails_even_though_the_key_is_present(self) -> None:
        workload = self._workload()
        events = {
            "below": _control_events(workload[0], wrong_prefill_writer=0),
            "above": _control_events(workload[1]),
        }
        verdict = _controls(workload, events)
        assert verdict["passed"] is False
        assert "route_writer_sequence" in verdict["per_case"]["below"]["failed"]

    def test_a_missing_route_call_fails_the_shape_sequence(self) -> None:
        workload = self._workload()
        below_routes, below_blocks = _control_events(workload[0])
        below_routes = below_routes[:-1]
        events = {"below": (below_routes, below_blocks), "above": _control_events(workload[1])}
        verdict = _controls(workload, events)
        assert verdict["passed"] is False
        assert "route_shape_sequence" in verdict["per_case"]["below"]["failed"]

    def test_missing_chunk_continuity_fails_the_block_schedule(self) -> None:
        workload = self._workload()
        events = {
            "below": _control_events(workload[0]),
            "above": _control_events(workload[1], drop_last_prefill_block=True),
        }
        verdict = _controls(workload, events)
        assert verdict["passed"] is False
        assert "begin_block_schedule" in verdict["per_case"]["above"]["failed"]

    def test_a_swapped_block_order_fails(self) -> None:
        workload = self._workload()
        events = {
            "below": _control_events(workload[0], swap_block_order=True),
            "above": _control_events(workload[1]),
        }
        verdict = _controls(workload, events)
        assert verdict["passed"] is False
        assert "begin_block_schedule" in verdict["per_case"]["below"]["failed"]

    def test_same_total_routes_with_swapped_widths_fails(self) -> None:
        workload = self._workload()
        events = {
            "below": _control_events(workload[0]),
            "above": _control_events(workload[1], swap_block_widths=True),
        }
        verdict = _controls(workload, events)
        assert verdict["passed"] is False
        failed = verdict["per_case"]["above"]["failed"]
        assert "begin_block_schedule" in failed
        assert "route_shape_sequence" in failed

    def test_an_extra_zero_row_event_fails_the_shape_sequence(self) -> None:
        workload = self._workload()
        events = {
            "below": _control_events(workload[0], extra_zero_row=True),
            "above": _control_events(workload[1]),
        }
        verdict = _controls(workload, events)
        assert verdict["passed"] is False
        assert "route_shape_sequence" in verdict["per_case"]["below"]["failed"]


class TestRepeatability:
    def test_identical_arrays_are_byte_equal(self) -> None:
        rows = {"c": np.arange(16, dtype=np.float32).reshape(4, 4)}
        verdict = check_repeatability(rows, {k: v.copy() for k, v in rows.items()})
        assert verdict["passed"] is True
        assert verdict["per_case"]["c"]["raw_bytes_equal"] is True

    def test_a_tiny_change_with_unchanged_top1_fails_repeatability(self) -> None:
        base = np.zeros((4, 8), dtype=np.float32)
        base[:, 0] = 1.0
        candidate = base.copy()
        candidate[0, 1] = 1e-6  # below any rounding that could flip the argmax
        quality = evaluate(base, candidate)
        assert quality["passed"] is True
        assert quality["top1_flips"] == 0
        assert quality["kl_max"] < THRESHOLDS["kl_max"]
        verdict = check_repeatability({"c": base}, {"c": candidate})
        assert verdict["passed"] is False
        assert verdict["per_case"]["c"]["raw_bytes_equal"] is False
        assert verdict["per_case"]["c"]["shape_match"] is True


class TestQualityPassed:
    def test_a_failed_category_is_not_hidden_by_combined_rows(self) -> None:
        # 1000 identical rows plus 2 rows with one near-tie top-1 flip: the
        # combined verdict passes (1/1002 flips, negligible KL) while the small
        # case fails the <500-row zero-flip screen.
        base_big = np.zeros((1000, 8), dtype=np.float32)
        base_big[:, 0] = 1.0
        small = np.tile([1.0, 0.99999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32)
        small_candidate = small.copy()
        small_candidate[0] = [0.99999, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        quality = evaluate_cases(
            {"big": base_big, "small": small},
            {"big": base_big.copy(), "small": small_candidate},
            order=["big", "small"],
        )
        assert quality["combined"]["passed"] is True
        assert quality["per_case"]["small"]["passed"] is False
        assert quality_passed(quality) is False

    def test_all_cases_passing_is_required_and_sufficient(self) -> None:
        rows = {"a": np.zeros((4, 4), dtype=np.float32)}
        quality = evaluate_cases(rows, {k: v.copy() for k, v in rows.items()}, order=["a"])
        assert quality_passed(quality) is True


class _WitnessOwner:
    pass


class _WitnessRunner:
    def __init__(self, storage: str, uses_int8: bool) -> None:
        self.kv_storage_resolved = storage
        self.uses_int8_kv = uses_int8
        self.kv_cache = _WitnessOwner() if uses_int8 else None


def _expected_public_events():
    return [
        {"rows": 4, "writer": EXPECTED_WRITER_PROMPT, "consumer": EXPECTED_CONSUMER_PREFILL},
        {"rows": 1, "writer": EXPECTED_WRITER_DECODE, "consumer": EXPECTED_CONSUMER_DECODE},
    ]


class TestPublicWitnessVerdict:
    def _int8(self) -> _WitnessRunner:
        return _WitnessRunner("int8_per_token_head", True)

    def test_a_correct_int8_runner_with_expected_keys_passes(self) -> None:
        verdict = public_witness_verdict(
            _expected_public_events(), self._int8(),
            max_tokens=1, generated_token_ids=[563],
        )
        assert verdict["passed"] is True, verdict["failed"]
        assert verdict["post_request_uses_int8_kv"] is True
        assert verdict["post_request_owner_present"] is True
        assert verdict["generated_token_count"] == 1

    def test_a_rebuild_to_the_wrong_storage_fails(self) -> None:
        verdict = public_witness_verdict(
            _expected_public_events(), _WitnessRunner("bf16", False),
            max_tokens=1, generated_token_ids=[563],
        )
        assert verdict["passed"] is False
        assert "post_request_storage_not_int8" in verdict["failed"]
        assert "post_request_runner_has_no_int8_owner" in verdict["failed"]

    def test_a_missing_route_observation_fails(self) -> None:
        verdict = public_witness_verdict([], self._int8(), max_tokens=1, generated_token_ids=[563])
        assert verdict["passed"] is False
        assert "no_int8_route_observed" in verdict["failed"]

    def test_a_wrong_decode_key_fails(self) -> None:
        events = [{"rows": 1, "writer": EXPECTED_WRITER_PROMPT, "consumer": EXPECTED_CONSUMER_PREFILL}]
        verdict = public_witness_verdict(events, self._int8(), max_tokens=1, generated_token_ids=[563])
        assert verdict["passed"] is False
        assert "public_route_writer_mismatch" in verdict["failed"]
        assert "public_route_consumer_mismatch" in verdict["failed"]

    def test_a_wrong_route_beside_the_expected_route_fails(self) -> None:
        events = _expected_public_events() + [
            {"rows": 4, "writer": EXPECTED_WRITER_DECODE, "consumer": EXPECTED_CONSUMER_DECODE}
        ]
        verdict = public_witness_verdict(events, self._int8(), max_tokens=1, generated_token_ids=[563])
        assert verdict["passed"] is False
        assert "public_route_writer_mismatch" in verdict["failed"]
        assert "public_route_consumer_mismatch" in verdict["failed"]
        assert verdict["writer_mismatch_indices"] == [2]

    def test_a_short_generation_fails_the_token_count(self) -> None:
        verdict = public_witness_verdict(
            _expected_public_events(), self._int8(),
            max_tokens=4, generated_token_ids=[563, 563],
        )
        assert verdict["passed"] is False
        assert "public_generated_token_count" in verdict["failed"]
