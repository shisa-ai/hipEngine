#!/usr/bin/env python3
"""Measure the retained halo-box campaign routes in one model residency.

The ``before`` arm restores the two exact owners selected at campaign commit
``0436e138b``. The ``after`` arm selects the current PF-3 Q5_1 M1 owner and
PF-1 grouped Q8_0 down owner. Every case receives three measurements per arm
in a balanced six-slot order. Adjacent cases reverse that order so each arm
occupies every timing slot equally across the four canonical categories.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    DEFAULT_FIXTURE,
    _git_metadata,
    _hipengine_case_sample,
    _host_metadata,
    _write_json,
    load_fixture,
    summarize_samples,
)


BEFORE_COMMIT = "0436e138b5fe6a43b1b1bae5df6c33fff2110148"
FORKB_ENV = "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN"
Q5_M1_ENV = "HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1"
ROW4_ENV = "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"
QSA_H256_ENV = "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"
Q4_BUNDLE_ENV = "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL"
Q51_PAIR_ENV = "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL"
MMQ_PREFILL_ENV = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"
MMQ_PREPACK_ENV = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"
MMQ_VECTOR_ROUTE_PACKAGES = ("q8-mmq-raw-vector", "q8-mmq-vec4")
_BASE_SEQUENCE = ("before", "after", "after", "before", "before", "after")


def arm_sequence(case_index: int) -> tuple[str, ...]:
    """Return a three-per-arm order, reversed for adjacent cases."""

    if int(case_index) % 2:
        return tuple(reversed(_BASE_SEQUENCE))
    return _BASE_SEQUENCE


def fixture_case_index(cases: Sequence[Mapping[str, Any]], case: Mapping[str, Any]) -> int:
    """Keep the full-fixture counterbalance when running a diagnostic subset."""
    return next(index for index, row in enumerate(cases) if row["id"] == case["id"])


def _weighted_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    prefill_ms = sum(float(row["prefill_ms"]) for row in rows)
    transitions = sum(int(row["decode_transitions"]) for row in rows)
    decode_ms = sum(float(row["decode_ms"]) for row in rows)
    if prefill_ms <= 0.0 or decode_ms <= 0.0:
        raise ValueError("sample timing must be positive")
    return {
        "prefill_tok_s_weighted": 1000.0 * prompt_tokens / prefill_ms,
        "decode_tok_s_weighted": 1000.0 * transitions / decode_ms,
    }


def _comparison(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    before_rates = _weighted_rates(before)
    after_rates = _weighted_rates(after)
    return {
        "before_prefill_tok_s_weighted": before_rates["prefill_tok_s_weighted"],
        "after_prefill_tok_s_weighted": after_rates["prefill_tok_s_weighted"],
        "after_over_before_prefill": (
            after_rates["prefill_tok_s_weighted"]
            / before_rates["prefill_tok_s_weighted"]
        ),
        "before_decode_tok_s_weighted": before_rates["decode_tok_s_weighted"],
        "after_decode_tok_s_weighted": after_rates["decode_tok_s_weighted"],
        "after_over_before_decode": (
            after_rates["decode_tok_s_weighted"]
            / before_rates["decode_tok_s_weighted"]
        ),
    }


def staged_slots(case_index: int, stage: int) -> list[tuple[int, str]]:
    if stage not in (0,1):
        raise ValueError("invalid screen stage")
    slots=list(enumerate(arm_sequence(case_index)))
    return slots[:2] if stage==0 else slots[2:]


def clear_screen_losses(samples, wall_increase):
    if not 0 < wall_increase < 1:
        raise ValueError("invalid screening threshold")
    summary=summarize_campaign_ab(samples,repetitions_per_mode=1)
    if len(summary["by_case"])!=12:
        raise ValueError("staged screen requires all12 cases")
    losses=[]
    for case in sorted(summary["by_case"]):
        rows={r["mode"]:r for r in samples if r["case_id"]==case}
        before,after=rows["before"],rows["after"]
        if (after["prefill_ms"]>before["prefill_ms"]*(1+wall_increase)
                and after["prefill_ms"]+after["decode_ms"]>
                (before["prefill_ms"]+before["decode_ms"])*(1+wall_increase)):
            losses.append(case)
    return losses if len(losses)>=2 else []


def measurement_sequence(case_index: int, repetitions: int) -> tuple[str, ...]:
    if repetitions not in (1,3):
        raise ValueError("only one-pair screen or three-pair publication supported")
    sequence=arm_sequence(case_index)
    return sequence[:2] if repetitions==1 else sequence


def summarize_campaign_ab(
    samples: Sequence[Mapping[str, Any]], *, repetitions_per_mode: int,
    allow_cross_mode_drift: bool = False,
) -> dict[str, Any]:
    """Validate the balanced A/B packet and summarize complete timing walls."""

    expected = int(repetitions_per_mode)
    if expected <= 0:
        raise ValueError("repetitions_per_mode must be positive")
    by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        mode = row.get("mode")
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        by_case[str(row["case_id"])].append(row)
    if not by_case:
        raise ValueError("campaign A/B produced no samples")

    case_summary: dict[str, Any] = {}
    mismatches: list[str] = []
    within_mode_deterministic = True
    for case_id, rows in sorted(by_case.items()):
        modes = {
            mode: [row for row in rows if row["mode"] == mode]
            for mode in ("before", "after")
        }
        counts = {mode: len(mode_rows) for mode, mode_rows in modes.items()}
        if any(count != expected for count in counts.values()):
            raise ValueError(
                f"{case_id}: expected {expected} samples per mode, got {counts}"
            )
        digests = {
            mode: {str(row["output_token_ids_sha256"]) for row in mode_rows}
            for mode, mode_rows in modes.items()
        }
        deterministic = all(len(values) == 1 for values in digests.values())
        within_mode_deterministic = within_mode_deterministic and deterministic
        exact = deterministic and digests["before"] == digests["after"]
        if not exact:
            mismatches.append(case_id)
        case_summary[case_id] = {
            "category": str(rows[0]["category"]),
            "prompt_tokens": int(rows[0]["prompt_tokens"]),
            "samples_per_mode": expected,
            **_comparison(modes["before"], modes["after"]),
            "within_mode_deterministic": deterministic if expected>1 else None,
            "cross_mode_output_exact": exact,
            "output_token_ids_sha256": {
                mode: sorted(values) for mode, values in digests.items()
            },
        }
    if not within_mode_deterministic:
        raise ValueError("within-mode output mismatch in campaign A/B")
    if mismatches and not allow_cross_mode_drift:
        raise ValueError(
            "cross-mode output mismatch in campaign A/B: " + ", ".join(mismatches)
        )
    by_shape: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_mode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in samples:
        by_shape[int(row["prompt_tokens"])].append(row)
        by_category[str(row["category"])].append(row)
        by_mode[str(row["mode"])].append(row)

    def compare_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        before = [row for row in rows if row["mode"] == "before"]
        after = [row for row in rows if row["mode"] == "after"]
        return {
            "case_count": len({str(row["case_id"]) for row in rows}),
            "samples_per_mode": len(before),
            **_comparison(before, after),
        }

    result = {
        "correctness": {
            "within_mode_deterministic": True if expected > 1 else None,
            "cross_mode_output_exact": not mismatches,
            "mismatched_case_ids": mismatches,
        },
        "by_case": case_summary,
        "by_shape": {
            str(shape): compare_group(rows)
            for shape, rows in sorted(by_shape.items())
        },
        "by_category": {
            category: compare_group(rows)
            for category, rows in sorted(by_category.items())
        },
        "before": summarize_samples(by_mode["before"]),
        "after": summarize_samples(by_mode["after"]),
    }
    if expected==1:
        for mode in ("before","after"):
            for case in result[mode]["cases"].values():
                for metric in case.values():
                    if isinstance(metric,dict):
                        for key in ("stddev","stdev","standard_deviation","coefficient_of_variation"):
                            if key in metric:
                                metric[key]=None
        result["uncertainty"]="Within-case repeatability/variance unmeasured; one sample per arm."
    return result


def q8_down_row4_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 4 * (full * (chunk_size >= 512) + (tail >= 512))


def q51_fold128_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 25 * (full * (chunk_size >= 64) + (tail >= 64))


def q51_fold_pair_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 25 * (full * (chunk_size >= 512) + (tail >= 512))


def q8_mmq_vec4_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full, tail = divmod(prompt_tokens, chunk_size)
    return 72 * (full * (chunk_size >= 64) + (tail >= 64))


def q8_mmq_raw_vector_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full,tail = divmod(prompt_tokens,chunk_size)
    return 242*(full*(chunk_size>=64)+(tail>=64))


def q8_bundle_call_in_scope(route_package: str, call_args: Sequence[object]) -> bool:
    if route_package not in {"q8-mapped-down","q8-down-bundle"}:
        return True
    if len(call_args) < 3:
        raise ValueError("Q8 bundle call requires a lane-map argument")
    mapped = bool(call_args[2])
    return mapped if route_package == "q8-mapped-down" else not mapped


def q8_mapped_down_expected_calls(prompt_tokens: int, chunk_size: int) -> int:
    if prompt_tokens < 1 or chunk_size < 1:
        raise ValueError("positive prompt and chunk sizes required")
    full,tail = divmod(prompt_tokens,chunk_size)
    return full*(chunk_size>=512)+(tail>=512)


def q8_down_register_expected_calls(tokens: int,chunk: int) -> int:
    return q8_down_row4_expected_calls(tokens,chunk)+q8_mapped_down_expected_calls(tokens,chunk)


def gdn_wave_norm_expected_calls(tokens: int, chunk: int) -> int:
    """Serial wave-norm calls: every GDN layer once per chunk of 2+ rows.

    ``route_scope`` closes the tiled GDN branch for both arms of a serial-owner
    package, so all 36 GDN layers dispatch the serial leaf; the 21-layer prefix
    this counted before was the tiled branch's own window, not a serial arm.
    """
    full, tail = divmod(tokens, chunk)
    return 36 * (full * (chunk >= 2) + (tail >= 2))

def mmq_token64_expected_calls(tokens: int, chunk: int) -> int:
    full,tail=divmod(tokens,chunk)
    return 12*(full*(chunk>=512)+(tail>=512))


def gr_dpp_expected_calls(tokens: int, chunk: int) -> int:
    """Two GR reads per layer, only for prefill chunks above 256 rows."""
    if tokens < 1 or chunk < 1:
        raise ValueError("positive token and chunk sizes required")
    full, tail = divmod(tokens, chunk)
    return 96 * (full * (chunk > 256) + (tail > 256))


def router_dpp_expected_calls(tokens: int, chunk: int) -> int:
    """One router per layer for multirow chunks; single-row decode is excluded."""
    if tokens < 1 or chunk < 1:
        raise ValueError("positive token and chunk sizes required")
    full, tail = divmod(tokens, chunk)
    return 48 * (full * (chunk >= 2) + (tail >= 2))


def apply_chunk_mode(runner: Any, mode: str, *, allocated_chunk_size: int) -> None:
    if mode not in {"before","after"}:
        raise ValueError("invalid chunk mode")
    chunk = 512 if mode=="before" else 1024
    if allocated_chunk_size < chunk:
        raise ValueError("chunk exceeds preallocated capacity")
    runner.prefill_chunk_size = chunk


def validate_chunk_coverage(chunks: Sequence[int], tokens: int, size: int) -> None:
    if tokens < 1 or size < 1:
        raise ValueError("positive token and chunk sizes required")
    expected = [min(size,tokens-start) for start in range(0,tokens,size)]
    if list(chunks) != expected:
        raise ValueError(f"chunk coverage {list(chunks)} != {expected}")


#: Route packages whose arm is request-local runner state rather than an env key.
RUNNER_SCOPED_ROUTE_PACKAGES = frozenset(
    {"mmq-token64", "qsa-head-pair", "qsa-head-quad", "q8-mapped-down"})
#: Route packages whose counted leaf is the scalar sparse-prefill rows owner.
HEAD_ROUTE_PACKAGES = frozenset({"qsa-head-pair", "qsa-head-quad"})
QSA_WMMA_ENV = "HIPENGINE_QWEN4_EXP_QSA_WMMA"


def head_route_configuration(route_package, *, environment=os.environ):
    """Disable WMMA for a head-grouping arm; return the record that restores it.

    Production binds ``QSA_WMMA=1`` and the mixer launches the grouped rows
    owner only when WMMA attention does not own the chunk, so an arm that leaves
    WMMA on counts zero head launches and measures the WMMA kernel instead. Both
    arms disable it, and the profile's own value is restored from the record.
    """

    if route_package not in HEAD_ROUTE_PACKAGES:
        return None
    previous = environment.get(QSA_WMMA_ENV)
    environment[QSA_WMMA_ENV] = "0"
    return {
        "environment": QSA_WMMA_ENV,
        "value": "0",
        "previous": previous,
        "arms": "both",
        "reason": (
            "the head-grouped rows owner is the scalar fallback: with "
            "QSA_WMMA=1 the mixer launches the WMMA attention kernel and the "
            "counted head leaf never engages"),
    }


def restore_route_configuration(record, *, environment=os.environ):
    if record is None:
        return
    previous = record["previous"]
    if previous is None:
        environment.pop(record["environment"], None)
    else:
        environment[record["environment"]] = previous


#: Route packages whose counted leaf is the serial GDN recurrence.
GDN_SERIAL_ROUTE_PACKAGES = frozenset({"gdn-narrow", "gdn-wave-norm"})
GDN_COLWARPS_ENV = "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"
#: Owners that preempt the grouped pair2 Q5_1 down chain - and with it the
#: row-publish selection - on the shipped lane: the routed F16 WMMA down (the
#: actual production owner), the exact iu8-risk+repair down, grouped WMMA,
#: grouped MoE prefill, the ds4 MMQ suffix and the T16 expert-major down. A
#: row-publish arm holds every one of them off for both
#: arms; the profile's own values come back on exit.
Q51_PAIR2_PREEMPTING_ENV = (
    "HIPENGINE_QWEN4_EXP_ROUTED_F16",
    "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT",
    "HIPENGINE_QWEN4_EXP_Q5_1_WMMA",
    "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL",
    "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL",
    "HIPENGINE_QWEN4_EXP_T16_DOWN",
)
#: The pair2 chain's own gate. It must stay on for the counted leaf to run, so
#: the arm records it instead of forcing it.
Q51_PAIR2_GATE_ENV = "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN"
#: Kept for callers that name the single exact owner.
Q51_IU8_EXACT_ENV = Q51_PAIR2_PREEMPTING_ENV[1]


def q51_pair2_control_record(*, environment=os.environ):
    """The recorded control: the value both arms force, and what it replaced.

    The forced value is what the scope writes, not what the environment held
    before it, so a reader cannot mistake the production binding for the arm's
    own state. The previous values are recorded beside it.
    """

    return {
        "preempting_owners_closed": {
            key: "0" for key in Q51_PAIR2_PREEMPTING_ENV
        },
        "preempting_owners_previous": {
            key: environment.get(key) for key in Q51_PAIR2_PREEMPTING_ENV
        },
        "pair2_gate": {Q51_PAIR2_GATE_ENV: environment.get(Q51_PAIR2_GATE_ENV)},
    }


def q51_register_cache_control_record(runner=None, *, environment=os.environ):
    """The recorded control for the register-cache arm.

    The arm moves one decision - the register-cache fallback - and holds two
    others fixed: the preempting owners are closed and the downstream
    row-publish sibling is bound off, because that sibling replaces the leaf the
    arm counts. Both are recorded with the value the arm forced and the value it
    replaced, so a reader cannot mistake the production binding for the arm's
    own state.
    """

    return {
        "runner_field": "q51_register_cache",
        "before": False,
        "after": True,
        "scope": "per measured run; restored on exit",
        **q51_pair2_control_record(environment=environment),
        "downstream_row_publish": {
            "runner_field": "q51_row_publish",
            "bound_value": False,
            "published_value": (
                None if runner is None
                else getattr(runner, "q51_row_publish", None)),
            "reason": (
                "the row-publish sibling replaces the register-cache leaf this "
                "arm counts, so both arms hold it off; the plan's value comes "
                "back on exit"),
        },
        "published_value": (
            None if runner is None
            else getattr(runner, "q51_register_cache", None)),
        "reason": (
            "the register-cache leaf is the fallback parent of the row-publish "
            "sibling, and the routed F16 WMMA down plus the exact iu8, "
            "grouped-WMMA/grouped-MoE, MMQ and T16 owners preempt the pair2 "
            "chain that contains it; both arms close those owners, hold the "
            "downstream sibling off and record the pair2 gate because it must "
            "stay on"),
    }


@contextlib.contextmanager
def q51_pair2_scope(*, environment=os.environ):
    """Close the owners that preempt the grouped pair2 Q5_1 down chain.

    The row-publish selection only exists inside that chain, so an arm that
    leaves any preemptor open counts zero launches and measures the other
    owner. Every key is read per dispatch, so closing them after construction
    takes effect; all prior values (including absent keys) come back on exit.
    The pair2 chain's own gate is not one of the preemptors and must already be
    on: without it the chain is skipped entirely, so the scope fails loudly
    rather than reporting an engagement it cannot produce.
    """

    if environment.get(Q51_PAIR2_GATE_ENV, "") in {"", "0", "false", "False"}:
        raise ValueError(
            f"{Q51_PAIR2_GATE_ENV} must be on for the grouped pair2 Q5_1 down "
            "chain; without it the counted row-publish leaf cannot launch")
    previous = {key: environment.get(key) for key in Q51_PAIR2_PREEMPTING_ENV}
    for key in Q51_PAIR2_PREEMPTING_ENV:
        environment[key] = "0"
    try:
        yield previous
    finally:
        for key, value in previous.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value


#: Owners that preempt the Q8_0 expert-down chains - and with them the register
#: selection - on the shipped lane: the routed F16 WMMA down (which owns the
#: grouped chain in production) and the Q8_0 WMMA down, which sits in front of
#: the sorted chain. A down-register arm closes both for both arms; the
#: profile's own values come back on exit.
Q8_DOWN_PREEMPTING_ENV = (
    "HIPENGINE_QWEN4_EXP_ROUTED_F16",
    "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN",
)
#: The chains' own gates. They must stay on for the counted leaf to run, so the
#: arm records them instead of forcing them: ``FORKB_GROUPED_DOWN`` and the
#: row4/bundle gates carry the sorted path. The token-major mapped path's own
#: admission is the runner's ``q8_mapped_down`` decision, recorded in the mapped
#: arm's control, not an env key.
Q8_DOWN_CHAIN_GATES = (
    "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN",
    "HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL",
    "HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL",
)
#: The token-major mapped path's own prerequisite: the grouped row4 gate/up map
#: this invocation builds. The arm fails loudly without it instead of measuring
#: a neutral A/B.
Q8_MAPPED_DOWN_MAP_GATE = "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL"
#: Owners that preempt the token-major mapped path. The routed F16 WMMA down and
#: the Q8_0 WMMA down preempt the expert-down chains the same arm counts; the
#: grouped Q5_K IU8 gate/up owner additionally sets ``grouped_prefill`` and keeps
#: the token-major Q5_K map producer from running, so the mapped arm closes it
#: in both arms too. The register/bundle arms must not: there the grouped chain
#: is the one they count.
Q8_MAPPED_DOWN_PREEMPTING_ENV = Q8_DOWN_PREEMPTING_ENV + (
    "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT",
)


def q8_down_control_record(*, environment=os.environ):
    """The recorded control: the value both arms force, and what it replaced.

    The forced value is what the scope writes, not what the environment held
    before it, so a reader cannot mistake the production binding for the arm's
    own state. The previous values are recorded beside it, and the chain's own
    gates are recorded as found.
    """

    return {
        "preempting_owners_closed": {
            key: "0" for key in Q8_DOWN_PREEMPTING_ENV
        },
        "preempting_owners_previous": {
            key: environment.get(key) for key in Q8_DOWN_PREEMPTING_ENV
        },
        "chain_gates": {
            key: environment.get(key) for key in Q8_DOWN_CHAIN_GATES
        },
    }


@contextlib.contextmanager
def q8_down_scope(*, environment=os.environ):
    """Close the owners that preempt the Q8_0 expert-down chains.

    The register selection only exists inside those chains, so an arm that
    leaves a preemptor open counts zero launches and measures the other owner.
    Every key is read per dispatch, so closing them after construction takes
    effect; all prior values (including absent keys) come back on exit. The
    chains' own gates are not preemptors and must already be on: without them
    the counted leaf is skipped entirely, so the scope fails loudly rather than
    reporting an engagement it cannot produce.
    """

    closed = {"", "0", "false", "False"}
    for key in Q8_DOWN_CHAIN_GATES:
        if environment.get(key, "") in closed:
            raise ValueError(
                f"{key} must be on for the Q8_0 expert-down chain; without it "
                "the counted Q8_0 down leaf cannot launch")
    previous = {key: environment.get(key) for key in Q8_DOWN_PREEMPTING_ENV}
    for key in Q8_DOWN_PREEMPTING_ENV:
        environment[key] = "0"
    try:
        yield previous
    finally:
        for key, value in previous.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value


def q8_mapped_down_control_record(*, environment=os.environ):
    """The mapped arm's control: forced preemptors, prior values and gates.

    The forced value is what the scope writes, not what the environment held
    before it, so a reader cannot mistake the production binding for the arm's
    own state. The map gate is recorded, never forced.
    """

    return {
        "preempting_owners_closed": {
            key: "0" for key in Q8_MAPPED_DOWN_PREEMPTING_ENV
        },
        "preempting_owners_previous": {
            key: environment.get(key) for key in Q8_MAPPED_DOWN_PREEMPTING_ENV
        },
        "chain_gates": {
            key: environment.get(key) for key in Q8_DOWN_CHAIN_GATES
        },
        "mapped_gate_up_map": environment.get(Q8_MAPPED_DOWN_MAP_GATE),
    }


@contextlib.contextmanager
def q8_mapped_down_scope(*, environment=os.environ):
    """Close the owners that preempt the token-major mapped path.

    The mapped selector needs the current call's row4 gate/up map, which only
    the token-major path builds. The routed F16 WMMA down, the Q8_0 WMMA down
    and the grouped Q5_K IU8 gate/up owner all keep that path from running, so
    the arm closes them for both arms and restores every prior value on exit.
    The map gate itself is a prerequisite, never forced: without it the arm
    cannot produce the owner it counts, so the scope fails loudly rather than
    reporting a neutral A/B.
    """

    closed = {"", "0", "false", "False"}
    for key in Q8_DOWN_CHAIN_GATES:
        if environment.get(key, "") in closed:
            raise ValueError(
                f"{key} must be on for the Q8_0 expert-down chain; without it "
                "the counted Q8_0 down leaf cannot launch")
    if environment.get(Q8_MAPPED_DOWN_MAP_GATE, "") in closed:
        raise ValueError(
            f"{Q8_MAPPED_DOWN_MAP_GATE} must be on for the token-major mapped "
            "down arm; without it the current call builds no row4 map and the "
            "mapped owner cannot engage")
    previous = {key: environment.get(key) for key in Q8_MAPPED_DOWN_PREEMPTING_ENV}
    for key in Q8_MAPPED_DOWN_PREEMPTING_ENV:
        environment[key] = "0"
    try:
        yield previous
    finally:
        for key, value in previous.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value


@contextlib.contextmanager
def mmq_vector_prerequisite(runner, route_package, *, environment=os.environ):
    """Admit the Q8 MMQ parent both vector arms measure inside.

    The child is request-local runner state, but the parent it rides is not on
    the shipped UD lane: the profile's recovery zeroes the parent switch after
    the bind, and a prefill session with no workspace launches nothing. The arm
    therefore admits the parent, provisions its workspace (and, for the vec4
    arm, the prepacked mapping), restores the caller's switches on exit and
    fails loudly when the prerequisite cannot be met - an arm that binds the
    child without the parent measures the scalar owner at zero calls and would
    report it as the candidate.

    The raw arm counts every raw-weight launch, so the prepacked mapping must
    stay unpublished for its window; the vec4 arm counts only the prepared
    weights, so it needs the mapping and its sidecars. The allocations are
    runner-owned and shared by both arms, not an arm difference.
    """

    if route_package not in MMQ_VECTOR_ROUTE_PACKAGES:
        raise ValueError(f"{route_package} is not an MMQ vector arm")
    raw = route_package == "q8-mmq-raw-vector"
    forced = {
        MMQ_PREFILL_ENV: "1",
        MMQ_PREPACK_ENV: "0" if raw else "1",
    }
    previous = {key: environment.get(key) for key in forced}
    for key, value in forced.items():
        environment[key] = value
    try:
        configure = getattr(runner, "configure_mmq_prefill_resources", None)
        if not callable(configure):
            raise ValueError(
                f"{route_package} needs the owning runner's MMQ provisioner; "
                "binding the child alone would measure the scalar parent")
        configure()
        if not getattr(runner, "_q8_mmq_buffers", ()):
            raise ValueError(
                "the Q8 MMQ parent did not provision a workspace, so the arm "
                "would measure the scalar owner at zero calls; admit it with "
                f"{MMQ_PREFILL_ENV}=1 and configure_mmq_prefill_resources()")
        if not raw:
            sidecars = getattr(runner, "_q8_mmq_weight_sidecars", None)
            if sidecars is None or not getattr(sidecars, "mapping", None):
                raise ValueError(
                    "the prepacked MMQ mapping did not provision, so the vec4 "
                    "arm would measure the raw owner at zero calls; it needs "
                    f"{MMQ_PREPACK_ENV}=1 and prepared sidecars")
        yield {"forced": forced, "previous": previous}
    finally:
        for key, value in previous.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value


def mmq_vector_control_record(route_package, runner=None, *, environment=os.environ):
    """The recorded control: the arm's field, and the prerequisite it forces."""

    raw = route_package == "q8-mmq-raw-vector"
    field = "dense_mmq_raw_vector_enabled" if raw else "dense_mmq_vec4_enabled"
    forced = {MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0" if raw else "1"}
    record = {
        "runner_field": field,
        "before": False,
        "after": True,
        "scope": "per measured run; restored on exit",
        "prerequisite_environment": forced,
        "prerequisite_previous": {key: environment.get(key) for key in forced},
        "published_value": (
            None if runner is None else getattr(runner, field, None)),
        "provisioning": (
            "runner.configure_mmq_prefill_resources(); the workspace and any "
            "prepacked sidecars are runner-owned and shared by both arms"),
        "reason": (
            "the child is request-local runner state, but the MMQ parent it "
            "rides is recovery-disabled on the shipped UD lane and the "
            "prepacked mapping decides which arm can count launches, so each "
            "arm admits and provisions its own prerequisite and restores it"),
    }
    if raw:
        # A sibling decision can move this arm's own count, so it is pinned in
        # both arms and recorded next to the field the arm does move.
        record["sibling_pins"] = {
            "dense_mmq_token64_enabled": {
                "bound_value": False,
                "published_value": (
                    None if runner is None
                    else getattr(runner, "dense_mmq_token64_enabled", None)),
                "reason": (
                    "the token64 leg remaps a matching raw-vec4 parent "
                    "(K2560/N12288, rows>=512) onto its own owner, so an "
                    "inherited token64 decision would count the token64 owner "
                    "inside this arm's raw-vec4 window and would replace the "
                    "scalar teacher the before arm measures; both arms pin it "
                    "off"),
            },
        }
    return record


def mmq_token64_control_record(runner=None, *, environment=os.environ):
    """The recorded control for the token64 leg and the parent it rides."""

    forced = {MMQ_PREFILL_ENV: "1", MMQ_PREPACK_ENV: "0"}
    return {
        "runner_field": "dense_mmq_token64_enabled",
        "before": False,
        "after": True,
        "scope": "per measured run; restored on exit",
        "parent_field": "dense_mmq_raw_vector_enabled",
        "parent_bound_value": True,
        "prerequisite_environment": forced,
        "prerequisite_previous": {key: environment.get(key) for key in forced},
        "published_value": (
            None if runner is None
            else getattr(runner, "dense_mmq_token64_enabled", None)),
        "provisioning": (
            "runner.configure_mmq_prefill_resources(); the raw workspace is "
            "runner-owned and shared by both arms"),
        "reason": (
            "the leg only exists on the raw-vec4 parent, and the shipped UD "
            "lane zeroes the MMQ parent switch after the bind, so an arm that "
            "binds the child alone measures the scalar owner at zero calls"),
    }


@contextlib.contextmanager
def gdn_serial_scope(*, environment=os.environ):
    """Close the tiled GDN branch for the measured window of a serial arm.

    Production binds ``GDN_COLWARPS_PREFILL=1`` with the row-split recurrence on
    every GDN layer (``GDN_COLWARPS_LAYERS=0..47``), and that owner serves any
    row count, so the serial wave-norm / narrow leaf these arms count never
    launches while the tiled branch is open. Both arms close it; the profile's
    own value comes back on exit. The key is read per dispatch, not cached at
    construction, so closing it after the generator exists still takes effect.
    """

    previous = environment.get(GDN_COLWARPS_ENV)
    environment[GDN_COLWARPS_ENV] = "0"
    try:
        yield
    finally:
        if previous is None:
            environment.pop(GDN_COLWARPS_ENV, None)
        else:
            environment[GDN_COLWARPS_ENV] = previous


@contextlib.contextmanager
def _combined_scope(*scopes):
    """Enter several diagnostic scopes together, and only when entered.

    Building a scope must not change state: the campaign calls ``route_scope``
    once per arm and enters it with ``with``, so an eagerly entered
    ``ExitStack`` would leave the arm bound before the measured run starts and
    would keep it bound if the caller never entered at all.
    """

    with contextlib.ExitStack() as stack:
        for scope in scopes:
            stack.enter_context(scope)
        yield


def route_scope(route_package: str, mode: str, runner: Any = None, *,
                environment: MutableMapping[str, str] = os.environ):
    """The live control a route package needs around its measured run.

    Most packages are selected through profile-owned env keys, which
    ``_apply_mode`` binds before the run. A package whose route is request-local
    state on the runner (the Q8 MMQ lane children, the QSA prefill head
    grouping) is selected here instead, so neither arm writes a process-wide
    switch and the profile's own decision comes back on exit. The Q8 MMQ vector
    arms also admit and provision the parent lane they ride, because the child
    alone measures nothing. The serial GDN arms are the exception that needs
    both: the leaf is runner state, but whether it serves the chunk at all is
    the profile's tiled branch, so the scope closes that branch for the measured
    window and restores it.
    """

    from hipengine.runtime.gguf_linear import q8_mmq_lane_diagnostic
    from hipengine.runtime.qwen4_exp_runner import (
        GDN_PREFILL_NARROW_VARIANT, GDN_PREFILL_WIDE_VARIANT, gdn_prefill_variant,
        qsa_prefill_head_mode,
    )

    if route_package == "mmq-token64":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # The leg only exists on the raw-vec4 parent, which the shipped UD
            # lane does not leave provisioned, so an arm without the owning
            # runner would measure the scalar owner at zero calls.
            raise ValueError(
                "mmq-token64 needs the owning runner; the leg only exists on "
                "the raw-vec4 parent and that parent is recovery-disabled on "
                "the shipped UD lane")
        # The parent the leg rides is not provisioned on the shipped lane: the
        # recovery zeroes the parent switch after the bind, and a prefill
        # session with no workspace launches nothing. Both arms admit the same
        # raw prerequisite the raw-vector package uses - the prepacked mapping
        # must stay unpublished or the raw-vec4 parents the leg remaps never
        # appear - and then move only the token64 decision.
        return _combined_scope(
            mmq_vector_prerequisite(
                runner, "q8-mmq-raw-vector", environment=environment),
            q8_mmq_lane_diagnostic(token64=mode == "after", raw_vector=True))
    if route_package in MMQ_VECTOR_ROUTE_PACKAGES:
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Binding nothing would measure whatever the plan published and
            # report it as the arm, so the diagnostic must name its runner.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no selection "
                "would measure the plan's own MMQ owner")
        raw = route_package == "q8-mmq-raw-vector"
        # The parent and the prepacked mapping decide whether the counted child
        # can launch at all, so the arm admits them and restores the caller's
        # values on exit. The diagnostic moves exactly one decision: the raw arm
        # leaves the prepacked child at the plan's value and the vec4 arm leaves
        # the raw child at it, because the two paths are exclusive per weight.
        # The raw arm additionally pins the token64 leg off in both arms: that
        # leg remaps a matching raw-vec4 parent onto its own owner, which would
        # count a different kernel inside this arm's window and would replace
        # the scalar teacher the before arm measures. The vec4 arm counts the
        # prepared weights, which the token64 leg never remaps, so it keeps the
        # plan's token64 value.
        return _combined_scope(
            mmq_vector_prerequisite(
                runner, route_package, environment=environment),
            q8_mmq_lane_diagnostic(
                token64=False if raw else None,
                raw_vector=(mode == "after") if raw else None,
                vec4=None if raw else (mode == "after")))
    if route_package in {"qsa-head-pair", "qsa-head-quad"}:
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Silently binding nothing would measure the parent and report it
            # as the head arm, so an explicit head diagnostic must name its
            # runner. Non-head packages keep their env-owned scope.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no head mode "
                "would measure the single-head parent")
        return qsa_prefill_head_mode(
            runner,
            ("pair" if route_package == "qsa-head-pair" else "quad")
            if mode == "after"
            else "parent",
        )
    if route_package in {"gdn-narrow", "gdn-wave-norm"}:
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Binding nothing would measure the profile's own GDN owner and
            # report it as the arm, so an explicit GDN diagnostic must name its
            # runner. Non-GDN packages keep their env-owned scope.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no variant "
                "would measure the profile's own GDN owner")
        if route_package == "gdn-narrow":
            # Both arms retain the wave-norm admission; the variant names which
            # index width runs.
            variant = gdn_prefill_variant(
                runner,
                GDN_PREFILL_NARROW_VARIANT if mode == "after"
                else GDN_PREFILL_WIDE_VARIANT,
            )
        else:
            variant = gdn_prefill_variant(
                runner, GDN_PREFILL_WIDE_VARIANT if mode == "after" else None)
        # The variant is request-local runner state, but which owner serves the
        # chunk is the profile's process-wide tiled branch. Close it for both
        # arms so the counted serial leaf is the one that runs.
        return _combined_scope(
            gdn_serial_scope(environment=environment), variant)
    if route_package == "q51-register-cache":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Binding nothing would measure whatever the plan published and
            # report it as the arm, so the diagnostic must name its runner.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no selection "
                "would measure the plan's own Q5_1 down owner")
        from hipengine.runtime.qwen4_exp_runner import (
            q51_register_cache, q51_row_publish)

        # The same owners that preempt the pair2 chain preempt this leaf, and
        # the downstream row-publish sibling replaces it outright, so both arms
        # close the preemptors and hold that sibling off; the profile's own
        # values come back on exit. ``Q51_PAIR2_GATE_ENV`` must stay on and is
        # recorded, not forced.
        return _combined_scope(
            q51_pair2_scope(environment=environment),
            q51_register_cache(runner, mode == "after"),
            q51_row_publish(runner, False))
    if route_package == "q51-row-publish":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Binding nothing would measure whatever the plan published and
            # report it as the arm, so the diagnostic must name its runner.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no selection "
                "would measure the plan's own Q5_1 down owner")
        from hipengine.runtime.qwen4_exp_runner import q51_row_publish

        # The routed F16 WMMA down, the exact iu8 down and the grouped-WMMA/
        # grouped-MoE/MMQ/T16 owners all preempt the pair2 chain the arm counts,
        # so both arms close them and the profile's own values come back on
        # exit. ``Q51_PAIR2_GATE_ENV`` must stay on and is recorded, not forced.
        return _combined_scope(
            q51_pair2_scope(environment=environment),
            q51_row_publish(runner, mode == "after"))
    if route_package == "q8-down-register":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Binding nothing would measure whatever the plan published and
            # report it as the arm, so the diagnostic must name its runner.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no selection "
                "would measure the plan's own Q8_0 down owner")
        from hipengine.runtime.qwen4_exp_runner import q8_down_register

        # The routed F16 WMMA down and the Q8_0 WMMA down preempt the two chains
        # the arm counts, so both arms close them and the profile's own values
        # come back on exit. The chains' own gates must stay on and are
        # recorded, not forced.
        return _combined_scope(
            q8_down_scope(environment=environment),
            q8_down_register(runner, mode == "after"))
    if route_package == "q8-mapped-down":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        if runner is None:
            # Binding nothing would measure whatever the plan published and
            # report it as the arm, so the diagnostic must name its runner.
            raise ValueError(
                f"{route_package} needs the owning runner; binding no selection "
                "would measure the plan's own token-major Q8_0 down owner")
        from hipengine.runtime.qwen4_exp_runner import q8_mapped_down

        # The routed F16 WMMA down, the Q8_0 WMMA down and the grouped Q5_K IU8
        # gate/up owner all keep the token-major map producer from running, so
        # both arms close them and the profile's own values come back on exit.
        # The map gate is a prerequisite, never forced: the scope fails loudly
        # without it rather than measuring a neutral A/B.
        return _combined_scope(
            q8_mapped_down_scope(environment=environment),
            q8_mapped_down(runner, mode == "after"))
    return contextlib.nullcontext()


def _apply_mode(
    mode: str,
    *,
    environment: MutableMapping[str, str] = os.environ,
    route_package: str = "pf13",
) -> None:
    if route_package == "norm-f16-publish":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        from scripts.qwen4exp_norm_f16_gate_support import ENV
        environment[ENV] = "1" if mode == "after" else "0"
        return
    if route_package == "dense-wide-padded":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        # Both diagnostic arms enter through the parent registry hook, even
        # after the named production plan starts selecting the padded leaf.
        from hipengine.runtime.gguf_linear import DENSE_WIDE_TAIL_VARIANT_ENV
        from scripts.qwen4exp_dense_wide_padded_gate_support import PARENT
        environment[DENSE_WIDE_TAIL_VARIANT_ENV] = PARENT
        return
    if route_package == "gdn-narrow":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        # Request-local on the runner; ``route_scope`` binds the arm.
        return
    if route_package == "router-dpp":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        environment["HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"] = (
            "f32_hidden_token_tile4_dpp_exact" if mode == "after"
            else "f32_hidden_token_tile4_dense_exact"
        )
        return
    if route_package == "gr-dpp-reduction":
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        environment["HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION"] = "1" if mode == "after" else "0"
        return
    if route_package == "chunk1024":
        if mode not in {"before","after"}:
            raise ValueError("invalid chunk mode")
        return
    if route_package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2", "gr-iu8", "gr-iu8-down", "q8-wmma-down"}:
        if mode not in {"before", "after"}:
            raise ValueError(f"invalid campaign A/B mode {mode!r}")
        flag = ROW4_ENV if route_package == "q5k-row4" else QSA_H256_ENV
        if route_package == "q4-bundle":
            flag = Q4_BUNDLE_ENV
        if route_package == "q51-pair":
            flag = Q51_PAIR_ENV
        if route_package == "gdn-register":
            flag = "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL"
        if route_package == "gdn-wave-norm":
            # Request-local on the runner; ``route_scope`` binds the arm.
            return
        if route_package == "mmq-token64":
            # Request-local on the runner; ``route_scope`` binds the arm.
            return
        if route_package in MMQ_VECTOR_ROUTE_PACKAGES:
            # Request-local on the runner; ``route_scope`` admits the parent
            # prerequisite and binds the arm.
            return
        if route_package in {"qsa-head-pair", "qsa-head-quad"}:
            # Request-local on the runner; ``route_scope`` binds the arm.
            return
        if route_package == "q4-pair":
            flag = "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL"
        if route_package == "q4-iu8-exact":
            flag = "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT"
        if route_package == "q51-iu8-exact":
            flag = "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT"
        if route_package == "q5k-iu8-exact":
            flag = "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT"
        if route_package == "qsa-ordered-v2":
            flag = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2"
        if route_package == "gr-iu8":
            flag = "HIPENGINE_QWEN4_EXP_GR_IU8"
        if route_package == "gr-iu8-down":
            flag = "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"
        if route_package == "q8-wmma-down":
            flag = "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"
        if route_package == "q8-wave-scale":
            flag = "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE"
        if route_package == "gr-wave-scale":
            flag = "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE"
        if route_package == "q8-mmq-prepack":
            flag = "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK"
        if route_package == "q8-down-row4":
            flag = "HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL"
        if route_package == "q51-fold128":
            flag = "HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL"
        if route_package == "q8-down-bundle":
            flag = "HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL"
        if route_package == "q8-mapped-down":
            # Request-local on the runner; ``route_scope`` binds the arm.
            return
        if route_package == "q8-down-register":
            # Request-local on the runner; ``route_scope`` binds the arm.
            return
        if route_package == "q51-fold-pair":
            flag = "HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL"
        if route_package == "q51-register-cache":
            # Request-local on the runner; ``route_scope`` binds the arm, closes
            # the preemptors and holds the downstream row-publish sibling off.
            return
        if route_package == "q51-row-publish":
            # Request-local on the runner; ``route_scope`` binds the arm.
            return
        environment[flag] = "1" if mode == "after" else "0"
        if route_package == "q8-mmq-planes2":
            environment["HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES"] = (
                "2" if mode == "after" else "3")
        if route_package == "qsa-h256-page256" and mode == "after":
            environment[flag] = "page256"
        return
    if mode == "before":
        environment[Q5_M1_ENV] = "0"
        environment[FORKB_ENV] = "0"
        return
    if mode == "after":
        environment[Q5_M1_ENV] = "1"
        environment[FORKB_ENV] = "1"
        return
    raise ValueError(f"invalid campaign A/B mode {mode!r}")


def validate_row4_engagement(mode: str, calls: int) -> None:
    if (mode == "before" and calls != 0) or (mode == "after" and calls <= 0):
        raise ValueError(f"invalid row4 engagement: {mode} calls={calls}")


def validate_qsa_h256_engagement(mode: str, calls: int, prompt_tokens: int) -> None:
    if prompt_tokens not in {512, 1024, 4096} or mode not in {"before", "after"}:
        raise ValueError("QSA engagement check requires a canonical shape and arm")
    expected = mode == "after" and prompt_tokens == 4096
    if (calls > 0) != expected:
        raise ValueError(f"invalid QSA engagement: {mode} p{prompt_tokens} calls={calls}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--warmups-per-mode", type=int, default=1)
    parser.add_argument("--repetitions-per-mode", type=int, default=3)
    parser.add_argument("--allow-cross-mode-drift", action="store_true",
                        help="non-exact route packages: record cross-mode "
                             "output drift instead of failing the protocol "
                             "(within-mode determinism still required)")
    parser.add_argument("--screen-only", action="store_true",
                        help="Permit one pair/case; diagnostic only, no promotion evidence")
    parser.add_argument("--staged-screen",action="store_true",
                        help="One full-suite pair, then stop for multiple clear losses or finish remaining pairs in-residency")
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--route-package", choices=("pf13", "q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "chunk1024", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2", "gr-iu8", "gr-iu8-down", "q8-wmma-down", "q8-mmq-planes2", "gr-dpp-reduction", "router-dpp", "gdn-narrow", "dense-wide-padded", "norm-f16-publish"), default="pf13")
    parser.add_argument("--case-id", action="append", help="Diagnostic subset; omitted for full gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmups_per_mode < 1:
        raise SystemExit("--warmups-per-mode must be at least 1")
    if args.screen_only and args.repetitions_per_mode != 1:
        raise SystemExit("screen-only requires --repetitions-per-mode 1")
    if not args.screen_only and args.repetitions_per_mode != 3:
        raise SystemExit("publication protocol requires --repetitions-per-mode 3")
    if args.staged_screen and (args.screen_only or args.repetitions_per_mode!=3 or args.case_id):
        raise SystemExit("staged-screen requires full suite,three repeats,no screen-only")
    if args.route_package == "chunk1024" and args.prefill_chunk_size != 1024:
        raise SystemExit("chunk1024 requires --prefill-chunk-size1024 for shared allocation")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file.resolve()
        )
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.core.memory import memory_stats
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill import (
        register_gguf_q5_k_q8_1_selected_prefill_kernels,
    )
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    fixture, fixture_sha256 = load_fixture(args.fixture)
    cases = fixture["cases"]
    if args.case_id:
        cases = [case for case in cases if case["id"] in args.case_id]
        if {case["id"] for case in cases} != set(args.case_id):
            raise SystemExit("unknown --case-id")
    transitions = int(fixture["decode_transitions"])
    model_root = args.model_root.resolve()
    max_sequence_length = max(int(row["prompt_tokens"]) for row in cases) + transitions + 8

    register_gguf_q5_k_q8_1_selected_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile.PRODUCTION,
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=args.prefill_chunk_size,
        )

    source = _git_metadata(ROOT)
    if not source or not source.get("tracked_clean"):
        raise SystemExit("campaign publication A/B requires a tracked-clean worktree")
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_halo_box_campaign_same_residency_ab",
        "status": "running",
        "performance_claim": False,
        "host": _host_metadata(),
        "source": source,
        "commands": {"argv": command, "shell": shlex.join(command)},
        "model_root": str(model_root),
        "model_identity": fixture.get("model"),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": fixture_sha256,
        "profile": {
            "requested": "production",
            "manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
            "fell_back_to_strict": resolved.fell_back_to_strict,
        },
        "arms": {
            "before": {
                "reference_commit": BEFORE_COMMIT,
                "q5_1_down": "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
                "q8_0_down": "selected_gemv_bf16_bf16_out",
            },
            "after": {
                "q5_1_down": "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
                "q8_0_down": "selected_grouped_gemv_bf16_bf16_out",
            },
        },
        "protocol": {
            "one_model_residency": True,
            "one_python_process": True,
            "case_ids": [str(row["id"]) for row in cases],
            "case_order": "fixture order",
            "case_counterbalance_indices": {
                str(row["id"]): fixture_case_index(fixture["cases"], row) for row in cases
            },
            "arm_order_even_case": list(measurement_sequence(0,args.repetitions_per_mode)),
            "arm_order_odd_case": list(measurement_sequence(1,args.repetitions_per_mode)),
            "qualification": "screen-one-pair" if args.screen_only else "canonical-three-pair",
            "promotion_eligible_protocol": not args.screen_only,
            "warmups_per_mode_per_case": int(args.warmups_per_mode),
            "measured_repetitions_per_mode_per_case": int(args.repetitions_per_mode),
            "decode_transitions": transitions,
            "visible_output_tokens": transitions + 1,
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "timing_boundary": (
                "synchronized runner.prefill including first greedy output, then "
                "exactly 128 runner.step transitions and a final device synchronize"
            ),
        },
        "warmups": [],
        "samples": [],
    }
    _write_json(args.output, artifact)

    generator = resolved.construct_generator(factory)
    if args.route_package == "q8-wmma-down":
        # Registered after construction: profile construction re-registers
        # the base families and would otherwise clobber the counting hook.
        from hipengine.kernels.registry import KernelKey, register, resolve
        row4_key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0",
                             "selected_grouped_wmma_prefill_bf16_bf16_out")
        q8_wmma_original = resolve(
            backend=row4_key.backend, layer=row4_key.layer,
            quant=row4_key.quant, variant=row4_key.variant)

        def counted_q8_wmma_down(*call_args, **call_kwargs):
            row4_calls[0] += 1
            return q8_wmma_original(*call_args, **call_kwargs)

        register(row4_key, counted_q8_wmma_down, replace=True)
        artifact["arms"] = {
            "before": {"q8_down":
                "selected_grouped_gemv row4 chain (exact grouped)"},
            "after": {"q8_down":
                "selected_grouped_wmma_prefill (T1 f16-WMMA dequant)"},
        }
    observed_chunks = []
    original_chunk = None
    if args.route_package == "chunk1024":
        original_chunk = generator.runner._prefill_chunk
        def counted_chunk(token_ids, **kwargs):
            observed_chunks.append(len(token_ids))
            return original_chunk(token_ids,**kwargs)
        generator.runner._prefill_chunk = counted_chunk
        artifact["arms"] = {"before":{"prefill_chunk_size":512},
                            "after":{"prefill_chunk_size":1024}}
        artifact["protocol"]["allocated_chunk_size"] = args.prefill_chunk_size
        artifact["protocol"]["prefill_chunk_size"] = "arm_specific_512_or_1024"
        artifact["protocol"]["memory_scope"] = (
            "Both arms share1024-capacity PLE/prefill/MMQ scratch, allocated before timing. "
            "Does not measure512-sized allocation vs1024-sized allocation.")
        artifact["protocol"]["correctness_scope"] = (
            "Exact generated trajectories are required by this screening harness. "
            "A mismatch requires full declared production numerical/state gates, not relaxed checks.")
    row4_calls = [0]
    register_mapped_calls = [0]
    mapped_owner_calls = {"bundle": 0, "register": 0}
    original_row4 = None
    original_register = None
    register_key = None
    narrow_original = None
    narrow_key = None
    narrow_enabled = [False]
    if args.route_package == "gdn-narrow":
        from hipengine.kernels.registry import KernelKey, register, resolve
        from scripts.qwen4exp_gdn_narrow_gate_support import expected_calls as gdn_narrow_expected_calls, validate_call
        row4_key = KernelKey("hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                             "qwen4exp_sigmoid_wave_norm_prefill")
        narrow_key = KernelKey(row4_key.backend, row4_key.layer, row4_key.quant,
                               "qwen4exp_sigmoid_wave_norm_narrow_prefill")
        original_row4 = resolve(backend=row4_key.backend, layer=row4_key.layer,
                                quant=row4_key.quant, variant=row4_key.variant)
        narrow_original = resolve(backend=narrow_key.backend, layer=narrow_key.layer,
                                  quant=narrow_key.quant, variant=narrow_key.variant)

        def counted_wide(*call_args, **call_kwargs):
            validate_call(call_args, max_rows=args.prefill_chunk_size)
            return original_row4(*call_args, **call_kwargs)

        def counted_narrow(*call_args, **call_kwargs):
            validate_call(call_args, max_rows=args.prefill_chunk_size)
            row4_calls[0] += 1
            return narrow_original(*call_args, **call_kwargs)

        # The arm selects one leaf through the runner's published variant, so
        # engagement is observed on the leaf that actually ran rather than
        # substituted into the parent's registry slot.
        register(row4_key, counted_wide, replace=True)
        register(narrow_key, counted_narrow, replace=True)
        artifact["arms"] = {
            "before": {"gdn": row4_key.variant},
            "after": {"gdn": narrow_key.variant},
        }
        artifact["diagnostic_registry_substitution"] = False
        artifact["diagnostic_runner_variant"] = True
        artifact["profile"]["scope"] = (
            "named profile; diagnostic arms override its GDN selection on the runner")
    if args.route_package == "norm-f16-publish":
        from hipengine.kernels.registry import register, resolve
        from scripts.qwen4exp_norm_f16_gate_support import (
            PRODUCER_KEY, expected_calls as publication_expected_calls, validate_producer_call,
        )
        row4_key = PRODUCER_KEY
        original_row4 = resolve(backend=row4_key.backend, layer=row4_key.layer,
                                quant=row4_key.quant, variant=row4_key.variant)
        def counted_publication(*call_args, **call_kwargs):
            validate_producer_call(call_args)
            row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)
        register(row4_key, counted_publication, replace=True)
        artifact["arms"] = {
            "before": {"hc_down": "parent norm + staging cast + native padded GEMM"},
            "after": {"hc_down": "dual-publication norm + native padded GEMM"},
        }
        artifact["diagnostic_registry_substitution"] = False
        artifact["profile"]["scope"] = "named profile; arms toggle bounded publication only"
    if args.route_package == "dense-wide-padded":
        from hipengine.kernels.registry import KernelKey, register, resolve
        from scripts.qwen4exp_dense_wide_padded_gate_support import (
            PARENT, CANDIDATE, call_in_scope, expected_calls as tail_expected_calls,
        )
        row4_key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", PARENT)
        original_row4 = resolve(backend=row4_key.backend, layer=row4_key.layer,
                                quant=row4_key.quant, variant=PARENT)
        tail_candidate = resolve(backend=row4_key.backend, layer=row4_key.layer,
                                 quant=row4_key.quant, variant=CANDIDATE)
        def counted_tail(*call_args, **call_kwargs):
            if narrow_enabled[0] and call_in_scope(call_args):
                row4_calls[0] += 1
                return tail_candidate(*call_args, **call_kwargs)
            return original_row4(*call_args, **call_kwargs)
        register(row4_key, counted_tail, replace=True)
        artifact["arms"] = {"before": {"dense_tail": PARENT},
                            "after": {"dense_tail": CANDIDATE}}
        artifact["diagnostic_registry_substitution"] = True
        artifact["profile"]["scope"] = "named profile; diagnostic arms override only scoped HC tail leaf"
    if args.route_package == "router-dpp":
        from hipengine.kernels.registry import KernelKey, register, resolve
        router_key = KernelKey("hip_gfx1151", "router_logits", "f32",
                               "f32_hidden_token_tile4_dpp_exact")
        router_original = resolve(backend=router_key.backend, layer=router_key.layer,
                                  quant=router_key.quant, variant=router_key.variant)
        def counted_router(*call_args, **call_kwargs):
            row4_calls[0] += 1
            return router_original(*call_args, **call_kwargs)
        register(router_key, counted_router, replace=True)
        artifact["arms"] = {
            "before": {"router_logits": "f32_hidden_token_tile4_dense_exact"},
            "after": {"router_logits": router_key.variant},
        }
    if args.route_package == "gr-dpp-reduction":
        from hipengine.kernels.registry import KernelKey, register, resolve
        row4_key = KernelKey("hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                             "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact")
        original_row4 = resolve(backend=row4_key.backend, layer=row4_key.layer,
                                quant=row4_key.quant, variant=row4_key.variant)

        def counted_gr_dpp(*call_args, **call_kwargs):
            row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)

        register(row4_key, counted_gr_dpp, replace=True)
        artifact["arms"] = {
            "before": {"gr_up": "coltile2_branch4_rowbatch4_wave_scale_f32_exact"},
            "after": {"gr_up": row4_key.variant},
        }
    gr_iu8_module = None
    if args.route_package in {"gr-iu8", "gr-iu8-down"}:
        import hipengine.runtime.qwen4_exp_runner as _gr_runner_module
        gr_iu8_module = _gr_runner_module
        # The up (320->10240) and down (10240->320) legs share one kernel
        # entry; the production profile may already bind the other leg, so
        # each package counts only its own geometry.
        down_leg = args.route_package == "gr-iu8-down"

        def counted_gr_iu8(*call_args, **call_kwargs):
            in_features = call_args[4]
            out_features = call_args[5]
            if (in_features > out_features) == down_leg:
                row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)

        original_row4 = _gr_runner_module.gguf_q8_0_iu8_wmma_prefill_f32_f32
        _gr_runner_module.gguf_q8_0_iu8_wmma_prefill_f32_f32 = counted_gr_iu8
        if down_leg:
            artifact["arms"] = {
                "before": {"gr_down":
                    "gguf_k_prefill_out_coltile_rowbatch f32 (exact coltile)"},
                "after": {"gr_down":
                    "q8_0_iu8_wmma_prefill_f32_f32 (T1 3-plane)"},
            }
        else:
            artifact["arms"] = {
                "before": {"gr_up":
                    "q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32 (fused exact)"},
                "after": {"gr_up":
                    "q8_0_iu8_wmma_prefill_f32_f32 + sigmoid + gated_mean (T1 3-plane)"},
            }
    if args.route_package in {"q5k-row4", "qsa-h256-wave", "qsa-h256-page256", "q4-bundle", "q51-pair", "gdn-register", "q4-pair", "q8-wave-scale", "gr-wave-scale", "q8-mmq-prepack", "q8-down-row4", "q51-fold128", "q8-down-bundle", "q51-fold-pair", "q8-mmq-vec4", "q51-register-cache", "q8-mmq-raw-vector", "q8-mapped-down", "q8-down-register", "q51-row-publish", "gdn-wave-norm", "mmq-token64", "qsa-head-pair", "qsa-head-quad", "q4-iu8-exact", "q51-iu8-exact", "q5k-iu8-exact", "qsa-ordered-v2"}:
        from hipengine.kernels.registry import KernelKey, register, resolve
        row4_key = (KernelKey(
            "hip_gfx1151", "linear", "gguf_q5_k",
            "selected_grouped_row4_gemv_bf16_bf16_out")
            if args.route_package == "q5k-row4" else KernelKey(
                "hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                "strict_h256_page256_wave_rows_spans" if args.route_package == "qsa-h256-page256"
                else "strict_h256_wave_rows_spans"))
        if args.route_package == "q4-bundle":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q4_k",
                "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out")
        if args.route_package == "q51-pair":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_bf16_bf16_out")
        if args.route_package == "gdn-register":
            row4_key = KernelKey(
                "hip_gfx1151", "gdn_recurrence_norm_gate", "f32_state",
                "qwen4exp_sigmoid_register_prefill")
        if args.route_package == "q4-pair":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q4_k",
                "selected_dual_grouped_pair2_bf16_bf16_out")
        if args.route_package == "q4-iu8-exact":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q4_k",
                "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
        if args.route_package == "q51-iu8-exact":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_wmma_iu8_risk_prefill_bf16_bf16_out")
        if args.route_package == "q5k-iu8-exact":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_k",
                "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out")
        if args.route_package == "qsa-ordered-v2":
            row4_key = KernelKey(
                "hip_gfx1151", "qsa_sparse_attention", "bf16_kv",
                "strict_ordered_three_pass_v2_spans")
        if args.route_package == "q8-wave-scale":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "coltile8_rowbatch4_wave_scale_f32_f32_out")
        if args.route_package == "gr-wave-scale":
            row4_key = KernelKey(
                "hip_gfx1151", "linear+gr_gated_mean", "gguf_q8_0",
                "coltile2_branch4_rowbatch4_wave_scale_f32_exact")
        if args.route_package == "q8-mmq-prepack":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package == "q8-mmq-vec4":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package == "q8-mmq-raw-vector":
            row4_key = KernelKey("hip_gfx1151","linear","gguf_q8_0",
                                "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package == "q8-down-row4":
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "selected_grouped_row4_gemv_bf16_bf16_out")
        if args.route_package == "q51-fold128":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_fold128_bf16_bf16_out")
        if args.route_package in {"q8-down-bundle","q8-mapped-down"}:
            row4_key = KernelKey(
                "hip_gfx1151", "linear", "gguf_q8_0",
                "selected_grouped_row4_bundle_gemv_bf16_bf16_out")
        if args.route_package == "q51-fold-pair":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out")
        if args.route_package == "q51-register-cache":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out")
        if args.route_package == "q51-row-publish":
            row4_key = KernelKey(
                "hip_gfx1151", "moe_linear", "gguf_q5_1",
                "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out")
        if args.route_package=="q8-down-register":
            row4_key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                              "selected_grouped_row4_register_gemv_bf16_bf16_out")
        if args.route_package == "gdn-wave-norm":
            row4_key = KernelKey("hip_gfx1151","gdn_recurrence_norm_gate",
                                "f32_state","qwen4exp_sigmoid_wave_norm_prefill")
        if args.route_package == "mmq-token64":
            row4_key=KernelKey("hip_gfx1151","linear","gguf_q8_0",
                              "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out")
        if args.route_package=="qsa-head-pair":
            row4_key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                              "strict_h256_head_pair_rows_spans")
        if args.route_package=="qsa-head-quad":
            row4_key=KernelKey("hip_gfx1151","qsa_sparse_attention","bf16_kv",
                              "strict_h256_head_quad_rows_spans")
        original_row4 = resolve(
            backend=row4_key.backend, layer=row4_key.layer,
            quant=row4_key.quant, variant=row4_key.variant)

        def counted_row4(*call_args, **call_kwargs):
            if args.route_package=="q8-down-register" and call_args[2]:
                register_mapped_calls[0]+=1
            if q8_bundle_call_in_scope(args.route_package,call_args):
                row4_calls[0] += 1
            return original_row4(*call_args, **call_kwargs)

        if args.route_package == "q8-mapped-down":
            # The mapped path admits the bundle key and the published register
            # decision may then move the call to the register sibling, so the
            # arm instruments both owners and counts only the mapped path's
            # map-pointer calls. The split is recorded per sample so a reader
            # sees which owner actually served the arm.
            register_key = KernelKey(
                row4_key.backend, row4_key.layer, row4_key.quant,
                "selected_grouped_row4_register_gemv_bf16_bf16_out")
            original_register = resolve(
                backend=register_key.backend, layer=register_key.layer,
                quant=register_key.quant, variant=register_key.variant)

            def counted_bundle(*call_args, **call_kwargs):
                if q8_bundle_call_in_scope("q8-mapped-down", call_args):
                    row4_calls[0] += 1
                    mapped_owner_calls["bundle"] += 1
                return original_row4(*call_args, **call_kwargs)

            def counted_register(*call_args, **call_kwargs):
                if q8_bundle_call_in_scope("q8-mapped-down", call_args):
                    row4_calls[0] += 1
                    mapped_owner_calls["register"] += 1
                return original_register(*call_args, **call_kwargs)

            register(row4_key, counted_bundle, replace=True)
            register(register_key, counted_register, replace=True)
        else:
            register(row4_key, counted_row4, replace=True)
        artifact["arms"] = {
            "before": {"q5_k_gate_up": "selected_gemv_bf16_bf16_out"},
            "after": {"q5_k_gate_up": row4_key.variant},
        }
        if args.route_package.startswith("qsa-h256-"):
            artifact["arms"] = {
                "before": {"sparse_prefill": "strict_rows_spans"},
                "after": {"sparse_prefill": row4_key.variant},
            }
        elif args.route_package == "q4-bundle":
            artifact["arms"] = {
                "before": {"q4_gate_up": "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out"},
                "after": {"q4_gate_up": row4_key.variant},
            }
        elif args.route_package == "q51-pair":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
        elif args.route_package == "gdn-register":
            artifact["arms"] = {
                "before": {"serial_gdn": "qwen4exp_sigmoid_strict_prefill"},
                "after": {"serial_gdn": row4_key.variant},
            }
        elif args.route_package == "q4-pair":
            artifact["arms"] = {
                "before": {"q4_gate_up": "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out"},
                "after": {"q4_gate_up": row4_key.variant},
            }
        elif args.route_package == "q8-wave-scale":
            artifact["arms"] = {
                "before": {"exact_q8_prefill": "coltile8_rowbatch4_f32_f32_out"},
                "after": {"exact_q8_prefill": row4_key.variant},
            }
        elif args.route_package == "gr-wave-scale":
            artifact["arms"] = {
                "before": {"gr_up": "coltile2_branch4_rowbatch4_f32_exact"},
                "after": {"gr_up": row4_key.variant},
            }
        elif args.route_package == "q8-mmq-prepack":
            artifact["arms"] = {
                "before": {"qkv_ssm_mmq": "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"},
                "after": {"qkv_ssm_mmq": row4_key.variant},
            }
        elif args.route_package == "q8-mmq-vec4":
            artifact["arms"] = {
                "before": {"qkv_ssm_mmq": "mmq128_prepacked_q8_1_d4x3_guarded_f32_f32_out"},
                "after": {"qkv_ssm_mmq": row4_key.variant},
            }
        elif args.route_package == "q8-mmq-raw-vector":
            artifact["arms"] = {
                "before": {"raw_mmq":"mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out"},
                "after": {"raw_mmq":row4_key.variant},
            }
        elif args.route_package == "q8-down-row4":
            artifact["arms"] = {
                "before": {"grouped_q8_down": "selected_grouped_gemv_bf16_bf16_out"},
                "after": {"grouped_q8_down": row4_key.variant},
            }
        elif args.route_package == "q51-fold128":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_pair2_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
        elif args.route_package == "q8-down-bundle":
            artifact["arms"] = {
                "before": {"q8_down": "selected_grouped_row4_gemv_bf16_bf16_out"},
                "after": {"q8_down": row4_key.variant},
            }
        elif args.route_package == "q8-mapped-down":
            artifact["arms"] = {
                "before": {"token_major_q8_down":"selected_gemv_bf16_bf16_out"},
                "after": {"token_major_q8_down":row4_key.variant},
            }
        elif args.route_package == "q51-fold-pair":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_pair2_fold128_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
        elif args.route_package == "q51-register-cache":
            artifact["arms"] = {
                "before": {"q51_down": "selected_grouped_prefill_pair2_fold128_pair_bf16_bf16_out"},
                "after": {"q51_down": row4_key.variant},
            }
    artifact["route_package"] = args.route_package
    if args.route_package=="q8-down-register":
        artifact["arms"] = {
            "before":{"q8_down":"selected_grouped_row4_bundle_gemv_bf16_bf16_out"},
            "after":{"q8_down":"selected_grouped_row4_register_gemv_bf16_bf16_out"}}
        artifact["selection_control"] = {
            "runner_field": "q8_down_register",
            "before": False,
            "after": True,
            "scope": "per measured run; restored on exit",
            **q8_down_control_record(),
            "reason": (
                "the routed F16 WMMA down owns the grouped chain and the Q8_0 "
                "WMMA down sits in front of the sorted chain, so neither "
                "counted path reaches the register leaf in production; both "
                "arms close those owners and the chain gates (FORKB grouped "
                "down, row4/bundle) are recorded because they must stay on"),
            "mapped_owner_runner_field": bool(
                getattr(generator.runner, "q8_mapped_down", False)),
            "published_value": bool(
                getattr(generator.runner, "q8_down_register", False)),
        }
    if args.route_package=="q8-mapped-down":
        artifact["selection_control"] = {
            "runner_field": "q8_mapped_down",
            "before": False,
            "after": True,
            "scope": "per measured run; restored on exit",
            **q8_mapped_down_control_record(),
            "counted_owners": ["bundle", "register"],
            "counted_owner_rule": (
                "the mapped path admits the bundle key and the published "
                "register decision may rewrite it; the arm counts the owner "
                "that actually launched, filtered to the map pointer"),
            "published_value": bool(
                getattr(generator.runner, "q8_mapped_down", False)),
            "reason": (
                "the token-major mapped path admits the bundle leaf only from "
                "the grouped row4 gate/up map this invocation builds; the "
                "routed F16 WMMA down, the Q8_0 WMMA down and the grouped "
                "Q5_K IU8 gate/up owner all preempt that map producer, so "
                "both arms close them and the chain gates and map gate are "
                "recorded because they must stay on"),
        }
    artifact["diagnostic_subset"] = bool(args.case_id)
    artifact["screen_only"] = args.screen_only
    if args.route_package in MMQ_VECTOR_ROUTE_PACKAGES:
        artifact["selection_control"] = mmq_vector_control_record(
            args.route_package, generator.runner)
    if args.route_package == "mmq-token64":
        artifact["selection_control"] = mmq_token64_control_record(
            generator.runner)
    if args.route_package=="qsa-head-quad":
        artifact["arms"]={
            "before":{"sparse_attention":"strict_h256_page256_wave_rows_spans"},
            "after":{"sparse_attention":"strict_h256_head_quad_rows_spans"}}
    if args.route_package=="qsa-head-pair":
        artifact["arms"]={
            "before":{"sparse_attention":"strict_h256_page256_wave_rows_spans"},
            "after":{"sparse_attention":"strict_h256_head_pair_rows_spans"}}
    if args.route_package=="qsa-ordered-v2":
        artifact["arms"]={
            "before":{"sparse_decode":"strict_ordered_three_pass_spans"},
            "after":{"sparse_decode":"strict_ordered_three_pass_v2_spans"}}
    if args.staged_screen:
        artifact["protocol"].update(
            qualification="staged-one-then-three",
            case_order="fixture order per stage; warmup once per case in first stage",
            early_stop_wall_increase=0.05,
            early_stop_minimum_cases=2,
            early_promotion=False)
    if args.route_package=="mmq-token64":
        artifact["arms"]={
            "before":{"raw_q":"mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out"},
            "after":{"raw_q":"mmq128_token64_q8_1_d4x3_guarded_f32_f32_out"}}
    if args.route_package == "gdn-wave-norm":
        artifact["arms"] = {
            "before":{"gdn":"qwen4exp_sigmoid_register_prefill"},
            "after":{"gdn":"qwen4exp_sigmoid_wave_norm_prefill"}}
    if args.route_package in GDN_SERIAL_ROUTE_PACKAGES:
        artifact["serial_route_control"] = {
            "environment": GDN_COLWARPS_ENV,
            "value": "0",
            "arms": "both",
            "scope": "per measured run; restored on exit",
            "reason": (
                "production binds the row-split tiled recurrence on every GDN "
                "layer, so the counted serial leaf only launches with the "
                "tiled branch closed"),
        }
    if args.route_package == "q51-register-cache":
        artifact["selection_control"] = q51_register_cache_control_record(
            generator.runner)
    if args.route_package == "q51-row-publish":
        artifact["arms"] = {
            "before": {"q51_down": "selected_grouped_prefill_pair2_register_cache_bf16_bf16_out"},
            "after": {"q51_down": "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out"}}
        artifact["selection_control"] = {
            "runner_field": "q51_row_publish",
            "before": False,
            "after": True,
            "scope": "per measured run; restored on exit",
            **q51_pair2_control_record(),
            "reason": (
                "the routed F16 WMMA down is the production owner of this "
                "scope and, behind it, the exact iu8, grouped-WMMA/grouped-MoE, "
                "MMQ and T16 owners also preempt the pair2 chain the arm "
                "counts; both arms close all of them so the counted leaf "
                "engages, and the pair2 gate is recorded because it must stay "
                "on"),
        }

    def sample(mode, case, repetition):
        narrow_enabled[0] = mode == "after"
        _apply_mode(mode, route_package=args.route_package)
        if args.route_package == "chunk1024":
            apply_chunk_mode(generator.runner,mode,allocated_chunk_size=args.prefill_chunk_size)
            observed_chunks.clear()
        start_calls = row4_calls[0]
        start_mapped = register_mapped_calls[0]
        start_owners = dict(mapped_owner_calls)
        with route_scope(args.route_package, mode, generator.runner):
            row = _hipengine_case_sample(
                generator.runner, case=case, repetition=repetition, transitions=transitions)
        if args.route_package == "chunk1024":
            validate_chunk_coverage(observed_chunks,int(case["prompt_tokens"]),
                                    generator.runner.prefill_chunk_size)
            row["executed_prefill_chunks"] = list(observed_chunks)
            row["active_chunk_size"] = generator.runner.prefill_chunk_size
        if args.route_package == "router-dpp":
            calls = row4_calls[0] - start_calls
            expected = (router_dpp_expected_calls(int(case["prompt_tokens"]),
                                                   generator.runner.prefill_chunk_size)
                        if mode == "after" else 0)
            if calls != expected:
                raise AssertionError(f"router DPP calls {calls} != {expected}")
            row["candidate_calls"] = calls
        if original_row4 is not None:
            calls = row4_calls[0] - start_calls
            if args.route_package == "norm-f16-publish":
                expected = (publication_expected_calls(int(case["prompt_tokens"]),
                                                      args.prefill_chunk_size)
                            if mode == "after" else 0)
                assert calls == expected, (calls, expected)
            elif args.route_package == "dense-wide-padded":
                expected = (tail_expected_calls(int(case["prompt_tokens"]),
                                                args.prefill_chunk_size)
                            if mode == "after" else 0)
                assert calls == expected, (calls, expected)
            elif args.route_package == "gdn-narrow":
                expected = (gdn_narrow_expected_calls(int(case["prompt_tokens"]),
                                                     args.prefill_chunk_size)
                            if mode == "after" else 0)
                assert calls == expected, (calls, expected)
            elif args.route_package=="mmq-token64":
                expected=mmq_token64_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                assert calls==expected,(calls,expected)
            if args.route_package == "gdn-wave-norm":
                expected=gdn_wave_norm_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                assert calls==expected,(calls,expected)
            if args.route_package=="q8-down-register":
                expected=q8_down_register_expected_calls(int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                mapped=register_mapped_calls[0]-start_mapped
                assert calls==expected,(calls,expected)
                assert mapped==(q8_mapped_down_expected_calls(int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0)
                row["mapped_candidate_calls"]=mapped
                row["compact_candidate_calls"]=calls-mapped
            elif args.route_package == "q8-mapped-down":
                expected_calls = q8_mapped_down_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Mapped Q8 down calls {calls} != {expected_calls}")
                owners = {name: mapped_owner_calls[name] - start_owners[name]
                          for name in mapped_owner_calls}
                if sum(owners.values()) != calls:
                    raise AssertionError(
                        f"Mapped Q8 down owner split {owners} does not cover "
                        f"{calls} calls")
                row["mapped_owner_calls"] = owners
            elif args.route_package == "q8-mmq-raw-vector":
                expected_calls = q8_mmq_raw_vector_expected_calls(
                    int(case["prompt_tokens"]),args.prefill_chunk_size) if mode=="after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Raw MMQ vector calls {calls} != {expected_calls}")
            elif args.route_package == "q8-mmq-vec4":
                expected_calls = q8_mmq_vec4_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q8 MMQ vec4 calls {calls} != {expected_calls}")
            elif args.route_package in {"q8-down-row4", "q8-down-bundle"}:
                expected_calls = q8_down_row4_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q8 down row4 calls {calls} != {expected_calls}")
            elif args.route_package == "q51-fold128":
                expected_calls = q51_fold128_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q51 fold128 calls {calls} != {expected_calls}")
            elif args.route_package in {"q51-fold-pair", "q51-register-cache", "q51-row-publish"}:
                expected_calls = q51_fold_pair_expected_calls(
                    int(case["prompt_tokens"]), args.prefill_chunk_size) if mode == "after" else 0
                if calls != expected_calls:
                    raise AssertionError(f"Q51 folded pair calls {calls} != {expected_calls}")
            elif args.route_package.startswith("qsa-h256-") or args.route_package in {"qsa-head-pair","qsa-head-quad"}:
                validate_qsa_h256_engagement(mode, calls, int(case["prompt_tokens"]))
            elif args.route_package == "qsa-ordered-v2":
                expected = (transitions * 12
                            if mode == "after" and int(case["prompt_tokens"]) == 4096
                            else 0)
                if calls != expected:
                    raise AssertionError(
                        f"ordered v2 decode calls {calls} != {expected}")
            elif args.route_package == "gr-dpp-reduction":
                expected = (gr_dpp_expected_calls(int(case["prompt_tokens"]),
                                                 args.prefill_chunk_size)
                            if mode == "after" else 0)
                if calls != expected:
                    raise AssertionError(f"GR DPP calls {calls} != {expected}")
            elif args.route_package == "gr-iu8-down":
                prompt_tokens = int(case["prompt_tokens"])
                full_chunks, tail = divmod(prompt_tokens, args.prefill_chunk_size)
                routed_chunks = full_chunks + (1 if tail > 256 else 0)
                expected = 96 * routed_chunks if mode == "after" else 0
                if calls != expected:
                    raise AssertionError(
                        f"GR iu8 down calls {calls} != {expected}")
            elif args.route_package == "gr-iu8":
                # 96 GR up sites (48 layers x attention+ffn), one call per
                # prefill chunk with rows > 256.
                prompt_tokens = int(case["prompt_tokens"])
                full_chunks, tail = divmod(prompt_tokens, args.prefill_chunk_size)
                routed_chunks = full_chunks + (1 if tail > 256 else 0)
                expected = 96 * routed_chunks if mode == "after" else 0
                if calls != expected:
                    raise AssertionError(
                        f"GR iu8 calls {calls} != {expected}")
            elif args.route_package not in {
                "norm-f16-publish", "dense-wide-padded", "gdn-narrow", "mmq-token64",
                "gdn-wave-norm", "q8-down-register", "q8-mapped-down",
                "q8-mmq-raw-vector", "q8-mmq-vec4", "q8-down-row4", "q8-down-bundle",
                "q51-fold128", "q51-fold-pair", "q51-register-cache", "q51-row-publish",
                "qsa-ordered-v2", "gr-dpp-reduction", "gr-iu8-down", "gr-iu8",
            }:
                validate_row4_engagement(mode, calls)
            row["candidate_calls"] = calls
        return row

    head_configuration = None
    try:
        # Applied inside the guarded block so the finally restores it even when
        # prefill or validation fails.
        head_configuration = head_route_configuration(args.route_package)
        if head_configuration is not None:
            artifact["qsa_wmma"] = head_configuration
        if args.route_package == "q8-mmq-prepack":
            _apply_mode("after",route_package=args.route_package)
            prepare_start = time.perf_counter()
            generator.runner.configure_mmq_prefill_resources()
            owner = generator.runner._q8_mmq_weight_sidecars
            artifact["sidecar"] = {
                "prepare_seconds": time.perf_counter()-prepare_start,
                "bytes": owner.nbytes,
                "count": len(owner.mapping),
                "protocol": "Prepared before either timing arm; raw weights remain resident.",
            }
        stages = (0,1) if args.staged_screen else (None,)
        measured_repetitions=args.repetitions_per_mode
        for stage in stages:
          if stage==1:
            try:
                losses=clear_screen_losses(artifact["samples"],0.05)
            except ValueError as error:
                artifact["status"]="failed_correctness_or_protocol"
                artifact["error"]=str(error)
                _write_json(args.output,artifact)
                return 2
            artifact["screen_decision"]={
                "clear_loss_cases":losses,
                "action":"stop_diagnostic" if losses else "continue_same_residency",
                "first_stage_sample_count":len(artifact["samples"])}
            _write_json(args.output,artifact)
            if losses:
                artifact["screen_only"]=True
                artifact["protocol"]["promotion_eligible_protocol"]=False
                artifact["protocol"]["measured_repetitions_per_mode_per_case"]=1
                artifact["protocol"]["arm_order_even_case"]=list(measurement_sequence(0,1))
                artifact["protocol"]["arm_order_odd_case"]=list(measurement_sequence(1,1))
                measured_repetitions=1
                break
          for case in cases:
            case_index = fixture_case_index(fixture["cases"], case)
            warmup_modes = ("before", "after") if case_index % 2 == 0 else ("after", "before")
            for warmup in range(0 if stage==1 else args.warmups_per_mode):
                for mode in warmup_modes:
                    row = sample(mode, case, warmup)
                    artifact["warmups"].append(
                        {"case_id": row["case_id"], "mode": mode}
                    )
                    print(
                        f"[warmup] {mode} {row['case_id']} "
                        f"pp={row['prefill_tok_s']:.3f} tg={row['decode_tok_s']:.3f}",
                        flush=True,
                    )
            mode_counts = {"before": int(stage==1), "after": int(stage==1)}
            sequence = measurement_sequence(case_index,args.repetitions_per_mode)
            slots=staged_slots(case_index,stage) if stage is not None else enumerate(sequence)
            for slot, mode in slots:
                row = sample(mode, case, mode_counts[mode])
                mode_counts[mode] += 1
                row.update(
                    {
                        "mode": mode,
                        "sequence_slot": slot,
                        "case_sequence": list(sequence),
                        "measurement_stage":stage,
                    }
                )
                artifact["samples"].append(row)
                _write_json(args.output, artifact)
                print(
                    f"[measure {mode_counts[mode] - 1}] {mode} {row['case_id']} "
                    f"slot={slot} pp={row['prefill_tok_s']:.3f} "
                    f"tg={row['decode_tok_s']:.3f}",
                    flush=True,
                )
        try:
            artifact["summary"] = summarize_campaign_ab(
                artifact["samples"],
                repetitions_per_mode=measured_repetitions,
                allow_cross_mode_drift=args.allow_cross_mode_drift,
            )
        except ValueError as error:
            artifact["status"] = "failed_correctness_or_protocol"
            artifact["error"] = str(error)
            _write_json(args.output, artifact)
            return 2
        artifact["status"] = "completed"
        artifact["memory_before_close"] = memory_stats()
        _write_json(args.output, artifact)
        return 0
    finally:
        if args.route_package == "gr-iu8":
            os.environ["HIPENGINE_QWEN4_EXP_GR_IU8"] = "0"
        else:
            _apply_mode("after", route_package=args.route_package)
        if original_chunk is not None:
            generator.runner._prefill_chunk = original_chunk
            generator.runner.prefill_chunk_size = args.prefill_chunk_size
        if gr_iu8_module is not None:
            gr_iu8_module.gguf_q8_0_iu8_wmma_prefill_f32_f32 = original_row4
            if args.route_package == "gr-iu8-down":
                os.environ["HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"] = "0"
        elif original_row4 is not None:
            register(row4_key, original_row4, replace=True)
        if original_register is not None:
            register(register_key, original_register, replace=True)
        if narrow_original is not None:
            register(narrow_key, narrow_original, replace=True)
        if args.route_package == "router-dpp":
            register(router_key, router_original, replace=True)
            _apply_mode("before", route_package=args.route_package)
        restore_route_configuration(head_configuration)
        generator.close()
        artifact["memory_after_close"] = memory_stats()
        _write_json(args.output, artifact)


if __name__ == "__main__":
    raise SystemExit(main())
