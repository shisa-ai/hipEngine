#!/usr/bin/env python3
"""Whole-model gate for a layer-scoped f16 Q8_0 dense prefill route.

Evaluates one of the layer-scoped f16 WMMA-class dense Q8_0 prefill selectors
against the exact coltile chain on the Qwen4Exp UD-Q4_K_XL canonical
exact-token fixture: full-vocabulary logits trajectories with the candidate
consuming the strict generated prefix at every compared transition, evaluated
with the calibrated mean/tail/max KL and top-1 thresholds, plus same-schedule
repeat determinism. Schema v2 distinguishes complete canonical numerical
coverage from profile admission: partial runs are screens, and this collector
alone never claims full profile qualification. ``--route`` picks which route is measured; see
:data:`SELECTORS`.

The teacher is the named production profile with the route's selectors cleared,
so the measured drift is that route's own contribution and not the rest of the
production stack. That is the same candidate-local isolation the Q8 MMQ plane
gate uses.

Every route here is a post-binder selector: the named production binder writes
its env keys during its own pass, so setting them before construction is
silently inert. This gate constructs the generator first and flips the
selectors per arm.

Phases per canonical case:
- teacher: selectors cleared -> exact ``coltile8_rowbatch4`` chain
- candidate x N: selectors set to ``--layers`` -> the route's candidate chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.execution_profiles import EvaluationThresholds
from hipengine.benchmark.provenance import collect_artifact_provenance
from hipengine.execution_profiles import (
    ExecutionProfile,
    VariantSelection,
    build_variant_manifest,
    resolve_runtime_profile,
)
from scripts.execution_profile_gdn_calibration import (
    CalibrationError,
    PromptCalibrationCapture,
    build_candidate_quality,
)

KIND = "hipengine_execution_profile_q8_wmma_prefill_layers_gate"
SCHEMA_VERSION = 2
CANONICAL_FIXTURE_SHA256 = "42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1"
WMMA_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"
DENSE_WIDE_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"
DENSE_WIDE_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"
STAGED_F32_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS"
# The certified f16 WMMA scope, written as the comma-joined list these selectors
# actually parse. A "16-47" range string is accepted by ``--layers`` (which runs
# it through ``_parse_layers``) but **not** by the selectors themselves, which
# split on commas and call ``int``. Passing the range form here fails at the
# first prefill with ``invalid literal for int() with base 10: '16-47'``.
WMMA_CERTIFIED_SCOPE = ",".join(str(layer) for layer in range(16, 48))
DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_FIXTURE = Path(
    "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"
)
# The category-heldout suite: 18 natural-length prompts across the same four
# categories, with the same per-case schema. This is the fixture that can supply
# the canonical protocol's first missing admission certificate.
ADMISSION_FIXTURE = Path(
    "benchmarks/fixtures/qwen4exp_admission_suite_18prompt.json"
)
ADMISSION_FIXTURE_SHA256 = (
    "005196cbf06f747350b0f35a02febb4aad3cd85a1251af1bfe9c152f95aadbfb"
)
# Both layer-scoped routes decline at ``rows <= 256``, so a fixture whose longest
# prompt is at or below this cannot exercise either one. Recorded here because
# the heldout suite is *short*: its 18 prompts run 39-71 tokens. Pointing this
# gate at it produces a bit-identical candidate/teacher pair and a KL of exactly
# zero - the same false-pass shape as the engagement defects recorded in
# ``benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/WMMA-ROUTE-FALSE-PASS.md``.
# The ledger's X-11 says the same thing from the other side: "39-71-token task
# prompts never enter >256-row candidates. Inactive short-task passes are
# controls, not qualification." So the row floor is a *check*, not a note.
ROUTE_ROW_FLOOR = 256


def _fixture_kind(fixture_sha256: str) -> str:
    """Classify a fixture so the two protocols can never be confused."""

    if fixture_sha256 == CANONICAL_FIXTURE_SHA256:
        return "canonical"
    if fixture_sha256 == ADMISSION_FIXTURE_SHA256:
        return "category_heldout"
    return "unknown"


def qualification_coverage(
    *,
    captures,
    fixture_sha256: str,
    prefill_chunk_size: int,
    fixture_rows=(),
    heldout_decode_steps: int | None = None,
):
    """Account for actual trajectories, not requested CLI counts.

    This is the frozen canonical D128 numerical protocol, not the full profile
    admission protocol. A changed fixture requires explicit protocol review.
    A complete run at one chunk size never qualifies another chunk size.

    A category-heldout fixture produces a **separate** record and never sets
    ``complete`` here: the two protocols are not interchangeable, and a run of
    one must not be readable as the other.
    """
    expected = {
        f"{category}-p{shape}": category
        for category in ("code", "general_en", "general_ja", "mixed_ja_en")
        for shape in (512, 1024, 4096)
    }
    ids = [capture.prompt_id for capture in captures]
    kind = _fixture_kind(fixture_sha256)
    checks = {
        "canonical_fixture_identity": kind == "canonical",
        "canonical_cases_once_each": len(ids) == len(expected) and set(ids) == set(expected),
        "canonical_categories": all(expected.get(c.prompt_id) == c.category for c in captures),
        "teacher_d128": bool(captures) and all(len(c.strict) == 129 for c in captures),
        "candidate_d128_three_repeats": bool(captures) and all(
            len(c.candidate_runs.get("candidate", ())) >= 3
            and all(len(run) == 129 for run in c.candidate_runs.get("candidate", ()))
            for c in captures
        ),
        "positive_chunk_size": prefill_chunk_size > 0,
    }
    record = {
        "protocol": "canonical_12_case_d128_three_repeat_numerical_v1",
        "complete": all(checks.values()), "checks": checks,
        "required_fixture_sha256": CANONICAL_FIXTURE_SHA256,
        "observed_fixture_sha256": fixture_sha256,
        "fixture_kind": kind,
        "observed_case_ids": ids,
        "observed_teacher_rows": sum(len(c.strict) for c in captures),
        "prefill_chunk_size": prefill_chunk_size,
        "scope": "this route/layer set, manifest and chunk size only",
    }
    heldout = _heldout_coverage(
        captures,
        fixture_sha256=fixture_sha256,
        fixture_rows=fixture_rows,
        decode_steps=heldout_decode_steps,
    )
    if heldout is not None:
        record["category_heldout"] = heldout
    return record


def _heldout_coverage(
    captures, *, fixture_sha256: str, fixture_rows=(), decode_steps: int | None = None
):
    """Coverage for the category-heldout protocol, or ``None`` if not run.

    The engagement check is the load-bearing one. A heldout suite that cannot
    reach the route's row floor produces a flawless-looking comparison between
    two runs of the *same* kernel - bit-identical logits, KL exactly zero,
    ``measurement_valid: true`` - so ``complete`` must stay ``False`` and the
    blocker must stay open. That is what the current suite does: 18 prompts of
    39-71 tokens against a 256-row floor.
    """
    if fixture_sha256 != ADMISSION_FIXTURE_SHA256:
        return None
    expected_ids = {
        "code_merge_intervals", "code_topological_sort", "code_lru_cache",
        "code_markdown_table", "general_en_plan", "general_en_explain",
        "general_ja_plan", "general_ja_explain", "mixed_ja_en_translate",
        "mixed_ja_en_review", "heldout_code_rate_limiter",
        "heldout_code_interval_schedule", "heldout_general_en_incident",
        "heldout_general_en_quant", "heldout_general_ja_migration",
        "heldout_general_ja_speculative", "heldout_mixed_summary",
        "heldout_mixed_review",
    }
    ids = [capture.prompt_id for capture in captures]
    # Taken from the fixture rather than the capture: ``PromptCalibrationCapture``
    # is shared by ten other gates and carries no row count, and adding a
    # required field to it would change all of them for one gate's check.
    rows = [int(value) for value in fixture_rows]
    longest = max(rows) if rows else 0
    # The heldout protocol is **not** the canonical 128-step one: the admission
    # suite declares 24 transitions. Hardcoding 129 here made the trajectory
    # checks fail and left the blocker open for the wrong reason - a misleading
    # accounting defect rather than the honest row-floor one.
    expected_rows = (int(decode_steps) + 1) if decode_steps else None
    checks = {
        "heldout_fixture_identity": fixture_sha256 == ADMISSION_FIXTURE_SHA256,
        "heldout_cases_once_each": len(ids) == len(expected_ids) and set(ids) == set(expected_ids),
        "heldout_categories": len({c.category for c in captures}) == 4,
        "heldout_declared_transitions": expected_rows is not None,
        "heldout_teacher_rows": bool(captures) and all(
            len(c.strict) == expected_rows for c in captures
        ),
        "heldout_candidate_three_repeats": bool(captures) and all(
            len(c.candidate_runs.get("candidate", ())) >= 3
            and all(
                len(run) == expected_rows
                for run in c.candidate_runs.get("candidate", ())
            )
            for c in captures
        ),
        # Not a formality: without it this protocol certifies a route that
        # never ran. See the module note above ROUTE_ROW_FLOOR.
        "heldout_prompts_exceed_route_row_floor": longest > ROUTE_ROW_FLOOR,
    }
    return {
        "protocol": "category_heldout_natural_length_d128_three_repeat_v1",
        "complete": all(checks.values()),
        "checks": checks,
        "required_fixture_sha256": ADMISSION_FIXTURE_SHA256,
        "observed_fixture_sha256": fixture_sha256,
        "observed_case_ids": ids,
        "observed_prompt_rows": rows,
        "longest_prompt_rows": longest,
        "decode_steps": decode_steps,
        "observed_rows_per_trajectory": (
            len(captures[0].strict) if captures else 0
        ),
        "route_row_floor": ROUTE_ROW_FLOOR,
        "blocked_reason": (
            None
            if longest > ROUTE_ROW_FLOOR
            else (
                f"longest heldout prompt is {longest} rows, at or below the "
                f"{ROUTE_ROW_FLOOR}-row floor both layer-scoped routes decline "
                "at, so the route cannot have engaged and this run is a "
                "control rather than qualification"
            )
        ),
        "scope": "this route/layer set, manifest and chunk size only",
    }


def qualification_verdict(coverage, *, quality_passed, deterministic, clean):
    checks = {
        "complete canonical numerical coverage": coverage["complete"],
        "calibrated quality thresholds": quality_passed,
        "same-schedule repeat determinism": deterministic,
        "execution-affecting worktree provenance": clean,
    }
    numerical = all(checks.values())
    heldout = coverage.get("category_heldout")
    heldout_complete = bool(heldout and heldout["complete"])
    # Each entry is emitted only when this collector has not produced that
    # certificate. A blocker is dropped by producing evidence, never by
    # asserting it - and the heldout entry in particular is dropped only when
    # the run both completed and reached the route's row floor.
    missing = []
    if not heldout_complete:
        missing.append("category-heldout numerical evidence")
    missing += [
        "strict-teacher and applicable BF16-relative evidence",
        "exact control/isolation/graph/lifecycle evidence",
        "task-quality evidence",
        "candidate route engagement and repair-oracle coverage",
    ]
    return {
        "status": "complete" if numerical else "invalid_or_screen_only",
        "measurement_valid": numerical,
        "canonical_numerical_qualification_claim": numerical,
        "category_heldout_qualification_claim": heldout_complete,
        "profile_qualification_claim": False,
        "qualification_blockers": [name for name, ok in checks.items() if not ok] + missing,
    }


def _load_fixture(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _trajectory_rows(result: Any) -> dict[str, Any]:
    logits = result.logits
    if logits is None:
        raise CalibrationError("gate requires full-vocabulary logits")
    return {
        "token_id": int(result.token_id),
        "logits": np.ascontiguousarray(logits, dtype=np.float32),
    }


def _strict_trajectory(runner: Any, prompt_ids: Sequence[int], decode_steps: int):
    runner.reset()
    result = runner.prefill([int(t) for t in prompt_ids], capture_logits=True)
    trajectory = [_trajectory_rows(result)]
    for _ in range(int(decode_steps)):
        result = runner.step(int(result.token_id), capture_logits=True)
        trajectory.append(_trajectory_rows(result))
    return tuple(trajectory)


def _forced_trajectory(
    runner: Any,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int],
):
    runner.reset()
    result = runner.prefill([int(t) for t in prompt_ids], capture_logits=True)
    trajectory = [_trajectory_rows(result)]
    for token_id in forced_input_ids:
        result = runner.step(int(token_id), capture_logits=True)
        trajectory.append(_trajectory_rows(result))
    return tuple(trajectory)


def _warm(runner: Any, warmup_tokens: Sequence[int]) -> None:
    """Absorb a first-request dispatch before the measured trajectory.

    The dispatch resolution is memoized on the selector state, so a flip is
    expected to take effect immediately; the discarded request also warms the
    JIT so the measured trajectory is not the first launch of a new variant.
    """

    runner.reset()
    runner.prefill([int(t) for t in warmup_tokens[:32]], capture_logits=False)


def _parse_layers(raw: str) -> str:
    """Normalize ``--layers`` to the selector's comma-joined form."""

    values: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, _, end = item.partition("-")
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(item))
    if not values:
        raise CalibrationError("--layers must select at least one layer")
    unique = sorted(set(values))
    if any(layer < 0 or layer > 63 for layer in unique):
        raise CalibrationError("--layers must be within 0..63")
    return ",".join(str(layer) for layer in unique)


@dataclass(frozen=True)
class Route:
    """One layer-scoped Q8_0 dense prefill route this gate can measure.

    Both routes are post-binder selectors: the named production binder writes
    their env keys during its own pass, so a value set before construction is
    inert and only a post-binder flip is effective.

    ``dense_wide`` clears the WMMA selector in **both** arms. The production
    default binds WMMA to the same layer window this route is gated at, so
    leaving it bound would make the comparison carry the WMMA route's
    arithmetic as well as the candidate's, and the two routes would contend for
    the window under test.
    """

    kind: str
    scenario_prefix: str
    candidate_chain: str
    candidate_env: Mapping[str, str]
    teacher_env: Mapping[str, str]
    surface: str

    def env_for(self, *, candidate: bool, layers: str) -> dict[str, str]:
        bindings = self.candidate_env if candidate else self.teacher_env
        return {
            key: value.format(layers=layers) for key, value in bindings.items()
        }

    @property
    def env_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.candidate_env, *self.teacher_env)))


# ``--route`` -> the route to gate. The teacher arm is the exact coltile chain
# in every case, so the measured drift is the route's own contribution and the
# figures are comparable across routes.
SELECTORS: dict[str, Route] = {
    "wmma": Route(
        kind=KIND,
        scenario_prefix="qwen4exp-ud-q4-k-xl-q8-wmma-dense-prefill-layers",
        candidate_chain="wmma_prefill_f32_f32_out",
        # The wide route must be cleared in **both** arms, for the reciprocal
        # reason the ``dense_wide`` route clears this one: the certified plan
        # binds ``dense_wide`` to 16-47, the same window this route is gated at,
        # and ``_q8_dense_wide_claims_parent`` accepts the WMMA-rewritten parent
        # as well as the exact parent. So with the wide route left bound it
        # claims the dispatch whichever way this selector is set, both arms
        # resolve to ``dense_wide256_f32_f32_out``, and the gate reports
        # bit-identical candidate/teacher logits with a KL of exactly zero and
        # ``measurement_valid: true`` - a green status covering a route that
        # never ran. Measured, not theoretical: the certified 16-47
        # configuration returned exactly that at HEAD, while the 2026-09-16
        # certification of the same route had a genuine exact-chain teacher and
        # mean KL 3.796e-4. See
        # ``benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/WMMA-ROUTE-FALSE-PASS.md``.
        candidate_env={
            WMMA_LAYERS_ENV: "{layers}",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
        },
        teacher_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
        },
        surface=(
            f"post-binder {WMMA_LAYERS_ENV}; "
            f"{DENSE_WIDE_ENV} cleared in both arms"
        ),
    ),
    "dense_wide": Route(
        kind="hipengine_execution_profile_q8_dense_wide_prefill_layers_gate",
        scenario_prefix=(
            "qwen4exp-ud-q4-k-xl-q8-dense-wide256-dense-prefill-layers"
        ),
        candidate_chain="dense_wide256_f32_f32_out",
        candidate_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "1",
            DENSE_WIDE_LAYERS_ENV: "{layers}",
        },
        teacher_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
        },
        surface=(
            f"post-binder {DENSE_WIDE_ENV} + {DENSE_WIDE_LAYERS_ENV}; "
            f"{WMMA_LAYERS_ENV} cleared in both arms"
        ),
    ),
    # The LDS-staged f32 tile. Unlike the other two routes this one does not
    # claim a dispatch the wide or WMMA route would otherwise take: it replaces
    # the leaf of the *exact* coltile chain, and its dispatch declines any
    # variant the other routes have already rewritten. So both wide selectors
    # must be cleared in both arms, or the candidate arm would resolve to a
    # wide/WMMA variant and the staged tile would never be selected at all -
    # the run would silently measure the exact chain against itself.
    "staged_f32": Route(
        kind="hipengine_execution_profile_q8_staged_f32_prefill_layers_gate",
        scenario_prefix=(
            "qwen4exp-ud-q4-k-xl-q8-staged-f32-dense-prefill-layers"
        ),
        candidate_chain="staged_f32_bm128_bn64_bk32",
        candidate_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
            STAGED_F32_LAYERS_ENV: "{layers}",
        },
        teacher_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
            STAGED_F32_LAYERS_ENV: "",
        },
        surface=(
            f"post-binder {STAGED_F32_LAYERS_ENV}; "
            f"{WMMA_LAYERS_ENV} and {DENSE_WIDE_ENV} cleared in both arms"
        ),
    ),
    # The composition that promotion would actually create, which the plain
    # ``staged_f32`` route cannot measure because it clears WMMA in both arms.
    #
    # In production the f16 WMMA route owns layers 16-47, so a staged scope of
    # 13-15 does not compete with it: the exact chain runs on 13-15, WMMA takes
    # over from 16, and the staged tile only ever replaces the exact chain on
    # 13-15. The teacher here is therefore **current production** (WMMA on
    # 16-47), not the exact chain, and the measured drift is the marginal cost
    # of adding the staged tile at 13-15 to a composition that is already
    # certified.
    #
    # This matters because the plain route's numbers cannot be reused for it.
    # Summing WMMA's 16-47 drift with the staged 13-15 increment would be a
    # summed-KL budget, which the scope ladders in this route's own packet show
    # to be invalid: 8-47's max KL exceeds 0-47's while its mean is lower, and
    # 0-7's mean exceeds 0-47's, so scope contributions are neither additive nor
    # monotonic.
    # The end-to-end promotion certificate for the composition above.
    #
    # ``staged_f32_over_wmma`` measures the marginal cost against current
    # production, which is the right question for "what does adding this cost"
    # but not the question the production envelope asks. The envelope is defined
    # against the exact chain, so the full promoted composition has to be scored
    # against the exact chain directly. Summing the marginal cost onto WMMA's own
    # 16-47 drift would be a summed-KL budget, which this packet's ladders show
    # is invalid.
    #
    # So: candidate = staged on the named scope **with WMMA still bound at
    # 16-47**, exactly as promotion would leave it; teacher = the exact chain.
    "staged_f32_composed_vs_exact": Route(
        kind=(
            "hipengine_execution_profile_q8_staged_f32_composed_vs_exact_gate"
        ),
        scenario_prefix=(
            "qwen4exp-ud-q4-k-xl-q8-staged-f32-composed-dense-prefill-layers"
        ),
        candidate_chain="wmma_prefill_f32_f32_out + staged_f32_bm128_bn64_bk32",
        candidate_env={
            WMMA_LAYERS_ENV: WMMA_CERTIFIED_SCOPE,
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
            STAGED_F32_LAYERS_ENV: "{layers}",
        },
        teacher_env={
            WMMA_LAYERS_ENV: "",
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
            STAGED_F32_LAYERS_ENV: "",
        },
        surface=(
            f"post-binder {STAGED_F32_LAYERS_ENV} with {WMMA_LAYERS_ENV}=16-47 "
            f"held in the candidate arm only; teacher is the exact chain; "
            f"{DENSE_WIDE_ENV} cleared in both arms"
        ),
    ),
    "staged_f32_over_wmma": Route(
        kind=(
            "hipengine_execution_profile_q8_staged_f32_over_wmma_prefill_layers_gate"
        ),
        scenario_prefix=(
            "qwen4exp-ud-q4-k-xl-q8-staged-f32-over-wmma-dense-prefill-layers"
        ),
        candidate_chain="wmma_prefill_f32_f32_out + staged_f32_bm128_bn64_bk32",
        candidate_env={
            WMMA_LAYERS_ENV: WMMA_CERTIFIED_SCOPE,
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
            STAGED_F32_LAYERS_ENV: "{layers}",
        },
        teacher_env={
            WMMA_LAYERS_ENV: WMMA_CERTIFIED_SCOPE,
            DENSE_WIDE_ENV: "0",
            DENSE_WIDE_LAYERS_ENV: "",
            STAGED_F32_LAYERS_ENV: "",
        },
        surface=(
            f"post-binder {STAGED_F32_LAYERS_ENV} over a {WMMA_LAYERS_ENV}=16-47 "
            f"teacher; {DENSE_WIDE_ENV} cleared in both arms"
        ),
    ),
}


def _select_route(name: str) -> Route:
    try:
        return SELECTORS[str(name)]
    except KeyError as exc:
        raise CalibrationError(f"unknown Q8 prefill route {name!r}") from exc


def _apply_route_env(route: Route, *, candidate: bool, layers: str) -> None:
    """Bind every selector the route owns. Empty layers means the exact chain."""

    os.environ.update(route.env_for(candidate=candidate, layers=layers))


def _restore_env(bound: Mapping[str, str | None]) -> None:
    for key, value in bound.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        last_prebinder_conflicts,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    if int(args.repeat_runs) < 3:
        raise CalibrationError("the gate requires at least three candidate repeats")

    fixture, fixture_sha256 = _load_fixture(REPO_ROOT / args.fixture)
    cases = list(fixture["cases"])
    if args.case_id:
        selected = {str(cid) for cid in args.case_id}
        cases = [row for row in cases if str(row["id"]) in selected]
        if {str(row["id"]) for row in cases} != selected:
            raise CalibrationError("unknown canonical case id in --case-id")
    if args.limit is not None:
        if int(args.limit) <= 0:
            raise CalibrationError("--limit must be positive")
        cases = cases[: int(args.limit)]
    if not cases:
        raise CalibrationError("selected case set is empty")
    decode_steps = int(args.decode_steps)
    if decode_steps < 0 or int(args.prefill_chunk_size) <= 0:
        raise CalibrationError("decode steps must be nonnegative and chunk size positive")
    transitions = int(fixture.get("decode_transitions", 128))
    if decode_steps > transitions:
        raise CalibrationError(
            f"decode steps {decode_steps} exceed fixture transitions {transitions}"
        )
    max_sequence_length = (
        max(int(row["prompt_tokens"]) for row in cases) + transitions + 8
    )
    candidate_layers = _parse_layers(args.layers)
    route = _select_route(str(args.route))

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    model_root = args.model_root.resolve()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(str(args.execution_profile)),
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    # Construct first: the named binder writes the selector as "" during its own
    # pass, so only a post-binder flip is effective.
    generator = resolved.construct_generator(factory)
    runner = generator.runner
    if runner is None:
        raise CalibrationError("runner is not resident")
    bound_route_env = {key: os.environ.get(key) for key in route.env_keys}
    # A caller-set value the binder discarded means the intended route never
    # ran and the gate would report a clean null. Fail rather than measure it.
    discarded = last_prebinder_conflicts()
    if discarded:
        raise CalibrationError(
            "the profile binder discarded pre-binder route values, so the "
            f"measured route is not the requested one: {sorted(discarded)}"
        )

    captures: list[PromptCalibrationCapture] = []
    prompt_manifest: list[dict[str, Any]] = []
    command = ["python3", str(Path(__file__).resolve().relative_to(REPO_ROOT)), *sys.argv[1:]]
    try:
        for position, row in enumerate(cases):
            prompt_id = str(row["id"])
            tokens = [int(t) for t in row["prompt_token_ids"]]
            _apply_route_env(route, candidate=False, layers="")
            _warm(runner, tokens)
            teacher = _strict_trajectory(runner, tokens, decode_steps)
            forced = [step["token_id"] for step in teacher[:-1]]
            _apply_route_env(route, candidate=True, layers=candidate_layers)
            _warm(runner, tokens)
            runs = tuple(
                _forced_trajectory(runner, tokens, forced)
                for _ in range(int(args.repeat_runs))
            )
            captures.append(
                PromptCalibrationCapture(
                    prompt_id=prompt_id,
                    category=str(row["category"]),
                    strict=teacher,
                    candidate_runs={"candidate": runs},
                )
            )
            prompt_manifest.append(
                {
                    "id": prompt_id,
                    "category": str(row["category"]),
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256")
                    or hashlib.sha256(
                        np.asarray(tokens, dtype="<i8").tobytes()
                    ).hexdigest(),
                }
            )
            print(
                f"{position + 1}/{len(cases)} {prompt_id}: teacher + "
                f"candidate(layers={candidate_layers}) x {args.repeat_runs}",
                flush=True,
            )
    finally:
        _restore_env(bound_route_env)

    evaluated = build_candidate_quality(
        captures,
        candidate_mode="candidate",
        scenario_id=(
            f"{route.scenario_prefix}-"
            f"{candidate_layers.replace(',', '_')}-c1-teacher-forced"
        ),
        thresholds=EvaluationThresholds(),
    )

    manifest = build_variant_manifest(
        profile="production",
        backend=QWEN4_EXP_BACKEND,
        model=QWEN4_EXP_MODEL,
        quant=QWEN4_EXP_QUANTS[1],
        kv_policy="paged_bf16_qsa_index_f32",
        graph_policy="request_owned_exact_moe_graph_c1",
        selections=(
            VariantSelection(
                layer="linear",
                scope="prefill_dense_q8_wmma_layers",
                selected_variant=route.candidate_chain,
                strict_fallback_variant="coltile8_rowbatch4_f32_f32_out",
                registry_quant="gguf_q8_0",
            ),
        ),
    )
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1151",
        resolved_backend=str(getattr(runner, "backend", "hip_gfx1151")),
        target_arch=str(os.environ.get("HIPENGINE_HIP_ARCH", "gfx1151")),
        model_path=model_root,
        quant="gguf_ud_q4_k_xl",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            **route.env_for(candidate=True, layers=candidate_layers),
        },
        build_profile=route.kind,
        timing_protocol="none_full_logits_only_v1",
        warmups=1,
        repetitions=int(args.repeat_runs),
        profiler={"enabled": False, "kind": None, "command": None},
    )
    generator.close()

    quality = evaluated["quality"]
    quality_passed = bool(
        quality.get("hard_gates_passed")
        and quality.get("finite")
        and not quality.get("requires_outlier_review", False)
    )
    deterministic = bool(evaluated["repeat_determinism"]["passed"])
    # Validity turns on whether the dirty worktree can change what executed,
    # not on whether any file anywhere is dirty: another agent's untracked
    # notes cannot alter dispatch or arithmetic.
    clean = not provenance.get("execution_affecting_dirty", provenance.get("dirty"))
    coverage = qualification_coverage(
        captures=captures, fixture_sha256=fixture_sha256,
        prefill_chunk_size=int(args.prefill_chunk_size),
        fixture_rows=[int(row["prompt_tokens"]) for row in cases],
        heldout_decode_steps=decode_steps,
    )
    verdict = qualification_verdict(
        coverage, quality_passed=quality_passed, deterministic=deterministic, clean=clean,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": route.kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        **verdict,
        "qualification_coverage": coverage,
        "route": {
            "name": str(args.route),
            "surface": route.surface,
            "candidate_layers": candidate_layers,
            "teacher_selector": "cleared",
            "teacher_chain": "exact coltile8_rowbatch4_f32_f32_out",
            "candidate_chain": route.candidate_chain,
            "warmup_request_after_selector_flip": True,
            "selector_restored_after_capture": True,
        },
        "protocol": {
            "model_root": str(model_root),
            "fixture": str((REPO_ROOT / args.fixture).resolve()),
            "fixture_sha256": fixture_sha256,
            "case_count": len(cases),
            "decode_steps": decode_steps,
            "candidate_repeat_runs": int(args.repeat_runs),
            "execution_profile": str(args.execution_profile),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "same_context_rule": (
                "candidate consumes the strict generated token prefix at every "
                "compared transition"
            ),
            "teacher_scope": (
                "production profile with the route's selectors cleared; "
                "candidate-local isolation of the changed route"
            ),
            "thresholds_evaluated": EvaluationThresholds().to_dict(),
            "profile_manifest_sha256": resolved.manifest_sha256,
            "strict_manifest_sha256": resolved.strict_manifest_sha256,
        },
        "quality": quality,
        "repeat_determinism": evaluated["repeat_determinism"],
        "strict_logits_sha256": evaluated["strict_logits_sha256"],
        "candidate_logits_sha256": evaluated["candidate_logits_sha256"],
        "variant_manifests": {"candidate_evidentiary": manifest},
        "prompts": prompt_manifest,
        "provenance": provenance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--route",
        choices=tuple(SELECTORS),
        default="wmma",
        help=(
            "which layer-scoped Q8_0 dense prefill route to gate "
            "(default: wmma)"
        ),
    )
    parser.add_argument(
        "--layers",
        default="0-47",
        help="candidate layer set, e.g. 0-47 or 8,9,10 (default: every layer)",
    )
    parser.add_argument(
        "--decode-steps", type=int, default=24,
        help="decode depth (default: 24, screen only; canonical numerical coverage requires 128)",
    )
    parser.add_argument("--repeat-runs", type=int, default=3)
    parser.add_argument("--execution-profile", choices=("production",), default="production")
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1, sort_keys=False) + "\n")
    quality = artifact["quality"]
    # The evaluator nests its figures under "summary"; reading them from the
    # top level printed nulls for every metric.
    summary = quality.get("summary", {})
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "measurement_valid": artifact["measurement_valid"],
                "hard_gates_passed": quality.get("hard_gates_passed"),
                "mean_kl": summary.get("kl_mean"),
                "p95_kl": summary.get("kl_p95"),
                "max_kl": summary.get("kl_max"),
                "top1_rate": summary.get("top1_agreement"),
                "repeat_deterministic": artifact["repeat_determinism"]["passed"],
                "blockers": artifact["qualification_blockers"],
                "output": str(args.output),
            },
            indent=1,
        )
    )
    return 0 if artifact["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
