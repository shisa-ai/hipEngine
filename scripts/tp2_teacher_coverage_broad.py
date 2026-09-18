"""Broadened AR teacher-forced coverage: TP2 vs both per-GPU TP1 controls.

Compares the production TP2 runner against the matched TP1 control on each
physical card over the engine's canonical multi-category prompt suite plus its
heldout-only rows, using FULL teacher-forced trajectories (every position of
every prompt), not only the final position.

The harness is fail-closed. ``all_gates_passed`` is False unless *all* of the
following hold, and every failure is named in ``gate_failures``:

* the intended suite is complete (all canonical + heldout prompt ids) unless
  ``--allow-partial`` is given, in which case the run is labelled a diagnostic
  subset and still does not certify anything;
* both controls (``tp1-d0`` and ``tp1-d1``) were scored against TP2;
* every declared category (code / general_en / general_ja / mixed_ja_en) has
  rows and passes the production KL envelope and the 97% top-1 bar, and the
  same holds within each scope (canonical and heldout) so an aggregate cannot
  mask a localized failure;
* the global and canonical-scope production envelope passes for each control,
  and the heldout scope passes its KL envelope and the global 99% top-1 bar;
* every teacher/student trajectory matches the expected prompt length and
  vocabulary width and is finite;
* every reported metric is finite (a NaN cannot pass a comparison);
* the TP2 arm is deterministic across at least three identical sweeps
  (docs/EXECUTION-PROFILES.md 6.4);
* reset/reuse boundaries hold: a fresh teacher-forced call after the sweep, an
  intervening different prompt, and a `generate()` call are all bit-identical
  to prompt 0's first occurrence. Because `teacher_forced_logits` zeroes state
  itself, these verify reset/reuse, not warm-state continuation.

The row count only chooses the ``coverage_scale`` label; it never turns a run
into a "complete" or "qualified" result. The gates above are the verdict.

Usage::

    python scripts/tp2_teacher_coverage_broad.py \
        --repeat-tp2 3 \
        --json benchmarks/results/2026-09-16-w7900-tp2-teacher-coverage-broad.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tp2_mlp_generate_e2e import (  # noqa: E402
    PRODUCTION_GATE,
    _gate_passes,
    _softmax,
)

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
CANONICAL_SUITE = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
HELDOUT_SUITE = REPO_ROOT / "benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl"
CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
CATEGORY_TOP1 = 0.97
#: docs/EXECUTION-PROFILES.md 6.4 requires at least three fixed-seed runs.
MIN_DETERMINISM_SWEEPS = 3
#: A row count below this cannot resolve the 99% global / 97% per-category
#: top-1 bars (docs/EXECUTION-PROFILES.md "Teacher-forced probe row-count
#: standard"); it only labels the probe scale, it does not certify anything.
QUALIFICATION_ROWS = 500


def qualification_label(rows: int) -> str:
    """Probe-scale label only: ``short_probe`` or ``extended_probe``.

    This is deliberately not called "full" or "complete": a row count cannot
    certify correctness. The enforced gates are the verdict.
    """

    return "extended_probe" if rows >= QUALIFICATION_ROWS else "short_probe"


# -- suite / rendering ------------------------------------------------------


def load_prompt_suite(
    canonical_path: Path = CANONICAL_SUITE,
    heldout_path: Path = HELDOUT_SUITE,
) -> list[dict[str, object]]:
    """Canonical suite plus heldout-only rows, deduplicated by id."""

    canonical: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in (canonical_path, heldout_path):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in seen:
                continue
            seen.add(row_id)
            row["heldout"] = path == heldout_path
            canonical.append(row)
    return canonical


def full_suite_ids(
    canonical_path: Path = CANONICAL_SUITE,
    heldout_path: Path = HELDOUT_SUITE,
) -> set[str]:
    ids: set[str] = set()
    for path in (canonical_path, heldout_path):
        for line in path.read_text().splitlines():
            if line.strip():
                ids.add(str(json.loads(line)["id"]))
    return ids


def render_chat(tokenizer: object, messages: list[dict[str, str]]) -> tuple[int, ...]:
    parts = []
    for message in messages:
        parts.append(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        )
    parts.append("<|im_start|>assistant\n")
    return tuple(int(t) for t in tokenizer.encode("".join(parts)))  # type: ignore[attr-defined]


def _load_tokenizer() -> object:
    """Seam for mocked tests; returns the GGUF tokenizer."""

    import hipengine.loading as _loading
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    return Qwen35GGUFTokenizer.from_gguf_info(_loading.load_gguf_index(MODEL))


def _session_factory(
    model: str, *, devices: tuple[int, ...], mode: str,
    resident_control: bool = False, capacity: int = 2048, row_hook=None,
) -> object:
    """Seam for mocked tests; constructs one resident session."""

    if resident_control:
        from scripts.tp2_resident_control import create_coverage_session
        return create_coverage_session(model, devices=devices, mode=mode,
                                       capacity=capacity, row_hook=row_hook)
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    return MlpTP2GenerationSession(model, devices=devices, mode=mode)


# -- metrics ----------------------------------------------------------------


def _kl_rows(teacher: np.ndarray, student: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row KL(P||Q) and top-1 match flags; shapes must agree."""

    if teacher.shape != student.shape:
        raise ValueError(f"shape mismatch: teacher {teacher.shape} vs student {student.shape}")
    p = _softmax(teacher)
    q = _softmax(student)
    kl = (p * (np.log(p + 1e-45) - np.log(q + 1e-45))).sum(axis=-1)
    return kl, teacher.argmax(axis=-1) == student.argmax(axis=-1)


def _aggregate(kl: np.ndarray, top1: np.ndarray) -> dict[str, float]:
    if kl.size == 0:
        return {
            "rows": 0,
            "mean_kl": float("nan"),
            "p95_kl": float("nan"),
            "p99_kl": float("nan"),
            "max_kl": float("nan"),
            "top1_agreement": float("nan"),
            "flipped_rows": 0,
        }
    return {
        "rows": int(kl.size),
        "mean_kl": float(kl.mean()),
        "p95_kl": float(np.percentile(kl, 95)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(kl.max()),
        "top1_agreement": float(top1.mean()),
        "flipped_rows": int((~top1).sum()),
    }


def _envelope_gate(summary: dict[str, float], *, top1_bar: float) -> dict[str, object]:
    """Production KL envelope plus a caller-supplied top-1 bar.

    Any non-finite metric fails: a NaN cannot pass a comparison.
    """

    failures: list[str] = []
    if summary["rows"] == 0:
        failures.append("no rows")
    else:
        for key in ("mean_kl", "p95_kl", "p99_kl", "max_kl", "top1_agreement"):
            if not math.isfinite(float(summary[key])):
                failures.append(f"{key} non-finite ({summary[key]})")
        for key in ("mean_kl", "p95_kl", "p99_kl", "max_kl"):
            if math.isfinite(float(summary[key])) and summary[key] > PRODUCTION_GATE[key]:
                failures.append(f"{key} {summary[key]:.6g} > {PRODUCTION_GATE[key]}")
        if (
            math.isfinite(float(summary["top1_agreement"]))
            and summary["top1_agreement"] < top1_bar
        ):
            failures.append(f"top1_agreement {summary['top1_agreement']:.6f} < {top1_bar}")
    return {"passed": not failures, "failures": failures, "top1_bar": top1_bar}


def _row_hash(logits: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(logits, dtype=np.float32).tobytes()).hexdigest()


def _trajectory_digest(rows: list[np.ndarray]) -> str:
    """Order-sensitive digest over the exact float32 bytes of a trajectory.

    Each row's byte length is prefixed so two different row partitions cannot
    collide on the same concatenation.
    """

    digest = hashlib.sha256()
    for row in rows:
        raw = np.ascontiguousarray(row, dtype=np.float32).tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def cross_teacher_check(
    teacher_a: list[np.ndarray],
    teacher_b: list[np.ndarray],
    *,
    expected_positions: list[int] | None = None,
    vocab_size: int | None = None,
) -> dict[str, object]:
    """Direct control-vs-control comparison, including byte equality.

    Two controls can produce *identical aggregate metrics* against a shared
    student while their logits differ (for example a softmax-invariant shift),
    so aggregate equality is not evidence of bit-identity. Only the digest and
    byte comparison below are. This is a **diagnostic**: bit-identity across a
    physical hardware boundary is not a promotion requirement, and the
    production numerical contract (envelope + same-schedule determinism) is
    what controls. ``byte_identical`` is recorded, never gated on.
    """

    shape_mismatches: list[dict[str, object]] = []
    nonfinite_rows: list[int] = []
    if expected_positions is None:
        expected_positions = [int(row.shape[0]) for row in teacher_a]
    if len(teacher_a) != len(teacher_b) or len(teacher_a) != len(expected_positions):
        shape_mismatches.append(
            {
                "kind": "count",
                "a_rows": len(teacher_a),
                "b_rows": len(teacher_b),
                "expected_rows": len(expected_positions),
            }
        )
    row_kl: list[np.ndarray] = []
    row_top1: list[np.ndarray] = []
    differing_rows = 0
    max_abs_diff = 0.0
    for index, (a_row, b_row) in enumerate(zip(teacher_a, teacher_b)):
        expected = expected_positions[index] if index < len(expected_positions) else None
        row_invalid = False
        for label, row in (("a", a_row), ("b", b_row)):
            if row.ndim != 2:
                shape_mismatches.append({"kind": "ndim", "index": index, "which": label})
                row_invalid = True
                continue
            if expected is not None and row.shape[0] != expected:
                shape_mismatches.append(
                    {
                        "kind": "positions",
                        "index": index,
                        "which": label,
                        "shape": list(row.shape),
                        "expected_positions": expected,
                    }
                )
                row_invalid = True
            if vocab_size is not None and row.shape[1] != vocab_size:
                shape_mismatches.append(
                    {
                        "kind": "vocab",
                        "index": index,
                        "which": label,
                        "cols": int(row.shape[1]),
                        "expected_vocab": vocab_size,
                    }
                )
                row_invalid = True
        if a_row.shape != b_row.shape:
            shape_mismatches.append(
                {
                    "kind": "row",
                    "index": index,
                    "a_shape": list(a_row.shape),
                    "b_shape": list(b_row.shape),
                }
            )
            row_invalid = True
        if row_invalid:
            continue
        if not (np.isfinite(a_row).all() and np.isfinite(b_row).all()):
            nonfinite_rows.append(index)
            continue
        a32 = np.ascontiguousarray(a_row, dtype=np.float32)
        b32 = np.ascontiguousarray(b_row, dtype=np.float32)
        if a32.tobytes() != b32.tobytes():
            differing_rows += 1
            max_abs_diff = max(
                max_abs_diff,
                float(np.abs(a32.astype(np.float64) - b32.astype(np.float64)).max()),
            )
        kl, top1 = _kl_rows(a32, b32)
        row_kl.append(kl)
        row_top1.append(top1)
    metrics = _aggregate(
        np.concatenate(row_kl) if row_kl else np.empty(0),
        np.concatenate(row_top1) if row_top1 else np.empty(0, dtype=bool),
    )
    return {
        "rows": len(teacher_a),
        "scored_rows": len(row_kl),
        "byte_identical": (
            differing_rows == 0 and not shape_mismatches and not nonfinite_rows
        ),
        "differing_rows": differing_rows,
        "max_abs_diff": max_abs_diff,
        "digest": {"tp1-d0": _trajectory_digest(teacher_a), "tp1-d1": _trajectory_digest(teacher_b)},
        "metrics": metrics,
        "shape_mismatches": shape_mismatches,
        "nonfinite_rows": nonfinite_rows,
    }


# -- arms -------------------------------------------------------------------


def run_teacher_arm(session: object, token_rows: list[tuple[int, ...]]) -> list[np.ndarray]:
    """Full-trajectory logits for one arm (one resident session)."""

    return [
        np.asarray(session.teacher_forced_logits(tokens), dtype=np.float32)  # type: ignore[attr-defined]
        for tokens in token_rows
    ]


def score_arm(
    teacher: list[np.ndarray],
    student: list[np.ndarray],
    categories: list[str],
    heldout: list[bool],
    *,
    expected_positions: list[int] | None = None,
    vocab_size: int | None = None,
) -> dict[str, object]:
    """Per-position KL/top-1 for one control, aggregated globally, per category,
    per scope, and per category-within-scope.

    Trajectory completeness is checked against the *expected* prompt lengths and
    vocabulary width (not merely teacher vs student), so two identically
    truncated trajectories do not pass.
    """

    shape_mismatches: list[dict[str, object]] = []
    nonfinite_rows: list[int] = []
    if expected_positions is None:
        expected_positions = [int(row.shape[0]) for row in teacher]
    if len(teacher) != len(student) or len(teacher) != len(expected_positions):
        shape_mismatches.append(
            {
                "kind": "count",
                "teacher_rows": len(teacher),
                "student_rows": len(student),
                "expected_rows": len(expected_positions),
            }
        )
    row_kl: list[np.ndarray] = []
    row_top1: list[np.ndarray] = []
    row_index: list[int] = []
    for index, (t_row, s_row) in enumerate(zip(teacher, student)):
        expected = expected_positions[index] if index < len(expected_positions) else None
        row_invalid = False
        for label, row in (("teacher", t_row), ("student", s_row)):
            if row.ndim != 2:
                shape_mismatches.append({"kind": "ndim", "index": index, "which": label})
                row_invalid = True
                continue
            if expected is not None and row.shape[0] != expected:
                shape_mismatches.append(
                    {
                        "kind": "positions",
                        "index": index,
                        "which": label,
                        "shape": list(row.shape),
                        "expected_positions": expected,
                    }
                )
                row_invalid = True
            if vocab_size is not None and row.shape[1] != vocab_size:
                shape_mismatches.append(
                    {
                        "kind": "vocab",
                        "index": index,
                        "which": label,
                        "cols": int(row.shape[1]),
                        "expected_vocab": vocab_size,
                    }
                )
                row_invalid = True
        if t_row.shape != s_row.shape:
            shape_mismatches.append(
                {
                    "kind": "row",
                    "index": index,
                    "teacher_shape": list(t_row.shape),
                    "student_shape": list(s_row.shape),
                }
            )
            row_invalid = True
        if row_invalid:
            continue
        if not (np.isfinite(t_row).all() and np.isfinite(s_row).all()):
            nonfinite_rows.append(index)
            continue
        kl, top1 = _kl_rows(t_row, s_row)
        row_kl.append(kl)
        row_top1.append(top1)
        row_index.append(index)

    def agg(indices: list[int]) -> dict[str, float]:
        kl = np.concatenate([row_kl[i] for i in indices]) if indices else np.empty(0)
        top1 = (
            np.concatenate([row_top1[i] for i in indices])
            if indices
            else np.empty(0, dtype=bool)
        )
        return _aggregate(kl, top1)

    all_indices = list(range(len(row_index)))
    per_category = {
        category: agg(
            [pos for pos, src in enumerate(row_index) if categories[src] == category]
        )
        for category in CATEGORIES
    }
    per_scope = {
        "canonical": agg([pos for pos, src in enumerate(row_index) if not heldout[src]]),
        "heldout": agg([pos for pos, src in enumerate(row_index) if heldout[src]]),
    }
    per_category_scope: dict[str, dict[str, dict[str, float]]] = {}
    for category in CATEGORIES:
        per_category_scope[category] = {}
        for scope, want_heldout in (("canonical", False), ("heldout", True)):
            per_category_scope[category][scope] = agg(
                [
                    pos
                    for pos, src in enumerate(row_index)
                    if categories[src] == category and heldout[src] == want_heldout
                ]
            )
    return {
        "global": agg(all_indices),
        "categories": per_category,
        "scopes": per_scope,
        "category_scopes": per_category_scope,
        "shape_mismatches": shape_mismatches,
        "nonfinite_rows": nonfinite_rows,
        "scored_rows": len(row_index),
    }


# -- gate evaluation (pure, unit-tested) ------------------------------------


def evaluate_gates(
    *,
    comparisons: dict[str, dict[str, object]],
    expected_controls: tuple[str, ...] = ("tp1-d0", "tp1-d1"),
    categories_present: set[str],
    expected_categories: tuple[str, ...] = CATEGORIES,
    suite_has_heldout: bool,
    suite_complete: bool,
    require_suite_complete: bool = True,
    nonfinite_rows: list[str],
    determinism: dict[str, object] | None,
    state_boundaries: dict[str, object] | None,
    teacher_cross_check: dict[str, object] | None = None,
) -> tuple[bool, list[str]]:
    """Fail-closed verdict over every required gate."""

    failures: list[str] = []
    if require_suite_complete and not suite_complete:
        failures.append("intended prompt suite incomplete (diagnostic subset)")
    for control in expected_controls:
        data = comparisons.get(control)
        if data is None:
            failures.append(f"missing control {control}")
            continue
        global_gate = _envelope_gate(
            data["global"], top1_bar=PRODUCTION_GATE["top1_agreement"]  # type: ignore[arg-type]
        )
        if not global_gate["passed"]:
            failures.append(f"{control} global: {'; '.join(global_gate['failures'])}")
        canonical = data["scopes"]["canonical"]  # type: ignore[index]
        canon_gate = _envelope_gate(canonical, top1_bar=PRODUCTION_GATE["top1_agreement"])
        if not canon_gate["passed"]:
            failures.append(f"{control} canonical: {'; '.join(canon_gate['failures'])}")
        heldout = data["scopes"]["heldout"]  # type: ignore[index]
        heldout_gate = _envelope_gate(
            heldout, top1_bar=PRODUCTION_GATE["top1_agreement"]
        )
        if not heldout_gate["passed"]:
            failures.append(f"{control} heldout: {'; '.join(heldout_gate['failures'])}")
        for category in expected_categories:
            summary = data["categories"].get(category)  # type: ignore[union-attr]
            if summary is None:
                failures.append(f"{control} missing category {category}")
                continue
            gate = _envelope_gate(summary, top1_bar=CATEGORY_TOP1)
            if not gate["passed"]:
                failures.append(
                    f"{control} category {category}: {'; '.join(gate['failures'])}"
                )
            # Per category-within-scope: only require a scope the suite covers.
            for scope in ("canonical", "heldout"):
                scope_summary = data["category_scopes"][category][scope]  # type: ignore[index]
                if scope_summary["rows"] == 0:
                    continue
                scope_gate = _envelope_gate(scope_summary, top1_bar=CATEGORY_TOP1)
                if not scope_gate["passed"]:
                    failures.append(
                        f"{control} {category}/{scope}: {'; '.join(scope_gate['failures'])}"
                    )
        mismatches = data.get("shape_mismatches") or []
        if mismatches:
            failures.append(f"{control} incomplete trajectories: {len(mismatches)}")
        bad_rows = data.get("nonfinite_rows") or []
        if bad_rows:
            failures.append(f"{control} non-finite trajectories: {len(bad_rows)}")

    missing_categories = set(expected_categories) - set(categories_present)
    if missing_categories:
        failures.append(f"suite missing categories: {sorted(missing_categories)}")
    if not suite_has_heldout:
        failures.append("suite has no heldout rows")

    if teacher_cross_check is None:
        failures.append("teacher cross-check not measured")
    else:
        cross_mismatches = teacher_cross_check.get("shape_mismatches") or []
        if cross_mismatches:
            failures.append(f"teacher cross-check incomplete: {len(cross_mismatches)}")
        cross_nonfinite = teacher_cross_check.get("nonfinite_rows") or []
        if cross_nonfinite:
            failures.append(f"teacher cross-check non-finite: {len(cross_nonfinite)}")
        cross_gate = _envelope_gate(
            teacher_cross_check["metrics"],  # type: ignore[arg-type]
            top1_bar=PRODUCTION_GATE["top1_agreement"],
        )
        if not cross_gate["passed"]:
            failures.append(
                f"teacher cross-check envelope: {'; '.join(cross_gate['failures'])}"
            )
        # ``byte_identical`` is recorded as a diagnostic only. Two physical GPUs
        # crossing a hardware boundary are not required to be bit-exact; the
        # production numerical contract above is what controls. Same-schedule
        # repeatability is enforced separately by the determinism gate.
        if not teacher_cross_check.get("byte_identical", False):
            pass  # diagnostic: differing bytes are expected and do not fail the gate

    if nonfinite_rows:
        failures.append(f"non-finite logit rows: {len(nonfinite_rows)}")

    if determinism is None:
        failures.append("determinism not measured")
    else:
        if int(determinism.get("sweeps", 0)) < MIN_DETERMINISM_SWEEPS:
            failures.append(
                f"determinism needs >={MIN_DETERMINISM_SWEEPS} identical sweeps"
            )
        if not determinism.get("per_row_match", False):
            failures.append("determinism per-row mismatch")

    if state_boundaries is None:
        failures.append("state boundaries not measured")
    else:
        if not state_boundaries.get("reset_reuse_bit_exact", False):
            failures.append("reset/reuse boundary not bit-exact")
        if not state_boundaries.get("intervening_prompt_reuse_bit_exact", False):
            failures.append("intervening-prompt reuse boundary not bit-exact")
        if not state_boundaries.get("reset_after_generation_bit_exact", False):
            failures.append("reset-after-generation boundary not bit-exact")

    return (not failures), failures


# -- provenance -------------------------------------------------------------


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return True


def _model_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Identity fields that describe host scheduling rather than the arithmetic or
#: the host model. This environment re-nices long-running shells, so one host
#: reports different values in different processes (16 and -4 have both been
#: observed inside a single shell command). Comparing them would fail a
#: numerical provenance check at random, so equality uses ``numerical_identity``
#: while every artifact keeps the raw value for the reader.
_VOLATILE_IDENTITY_FIELDS = ("nice",)


def numerical_identity(identity: dict[str, object] | None) -> dict[str, object]:
    """Identity for provenance equality: host scheduling fields removed."""

    out = dict(identity or {})
    host = dict(out.get("host") or {})  # type: ignore[arg-type]
    for field in _VOLATILE_IDENTITY_FIELDS:
        host.pop(field, None)
    if host or "host" in out:
        out["host"] = host
    return out


def _host_identity() -> dict[str, object]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "nice": os.nice(0),
    }


def _device_identities(session: object) -> dict[str, object]:
    out: dict[str, object] = {}
    for device in session.devices:  # type: ignore[attr-defined]
        info = session.runtime.device_info(int(device))  # type: ignore[attr-defined]
        out[str(device)] = {
            "name": info.name,
            "uuid": info.uuid,
            "pci_bus_id": info.pci_bus_id,
        }
    return out


def _resolved_route(session: object) -> dict[str, object]:
    return {
        "mode": session.mode,  # type: ignore[attr-defined]
        "schedule": session.schedule,  # type: ignore[attr-defined]
        "prefill_schedule": getattr(session, "prefill_schedule", None),
        "driver": session.driver,  # type: ignore[attr-defined]
        "reduce_mode": session.reduce_mode,  # type: ignore[attr-defined]
        "head_shard": session.head_shard,  # type: ignore[attr-defined]
        "max_sequence_length": session.max_sequence_length,  # type: ignore[attr-defined]
    }


# -- driver -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument(
        "--arms",
        default="tp1-d0,tp1-d1,tp2",
        help="comma-separated subset of tp1-d0,tp1-d1,tp2",
    )
    parser.add_argument(
        "--repeat-tp2",
        type=int,
        default=MIN_DETERMINISM_SWEEPS,
        help=f"identical TP2 sweeps; >={MIN_DETERMINISM_SWEEPS} required for determinism",
    )
    parser.add_argument(
        "--model-hash",
        choices=("full", "none"),
        default="full",
        help="full sha256 of the model artifact, or 'none' for a fast probe",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow a diagnostic subset of the suite; the run still cannot pass "
        "the suite-completeness gate, it is labelled diagnostic",
    )
    parser.add_argument('--capture-resident-arm', choices=('tp1-d0', 'tp1-d1', 'tp2'))
    parser.add_argument('--resident-results', type=Path, nargs=3)
    parser.add_argument('--max-sequence-length', type=int, default=71)
    parser.add_argument('--execution-profile', choices=('strict', 'production'), default='production')
    parser.add_argument('--sustained-arm', choices=('tp1-d0','tp1-d1','tp2'))
    parser.add_argument('--tp2-bulk-prefill', action='store_true',
        help='drive the tp2 sustained arm through the session rank-local bulk '
             'prefill candidate instead of the committed token-serial route; '
             'the artifact records the schedule it measured')
    parser.add_argument('--teacher-source', type=Path)
    parser.add_argument('--sustained-report', type=Path, nargs=3)
    parser.add_argument('--horizon', type=int, default=None,
        help='score only the first D teacher-forced rows of each prompt. The '
             'captures always hold 128 rows; D must come from a declared '
             'authorization (docs/EXECUTION-PROFILES.md 6.5), and the '
             'full-horizon envelope is always recorded as a diagnostic so a '
             'shorter horizon cannot make a tail disappear from the artifact')
    args = parser.parse_args(argv)
    if args.sustained_arm or args.sustained_report:
        if not args.json or args.repeat_tp2 < 3:
            parser.error('sustained gate requires --json and >=3 repeats')
        return capture_sustained_arm(args) if args.sustained_arm else report_sustained(args)
    if args.capture_resident_arm or args.resident_results:
        if not args.json:
            parser.error('resident capture/report requires --json')
        if args.repeat_tp2 < MIN_DETERMINISM_SWEEPS:
            parser.error('resident controls require at least three sweeps per arm')
        if args.capture_resident_arm:
            return capture_resident_arm(args)
        return report_resident_coverage(args)
    if args.repeat_tp2 < 1:
        parser.error("--repeat-tp2 must be >= 1")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    suite = load_prompt_suite()
    if args.limit_rows is not None:
        suite = suite[: args.limit_rows]
    tokenizer = _load_tokenizer()
    token_rows = [render_chat(tokenizer, row["messages"]) for row in suite]  # type: ignore[arg-type]
    categories = [str(row["category"]) for row in suite]
    heldout = [bool(row["heldout"]) for row in suite]
    total_positions = sum(len(t) for t in token_rows)
    category_counts = {c: categories.count(c) for c in CATEGORIES}
    categories_present = {c for c in CATEGORIES if category_counts.get(c, 0) > 0}
    expected_ids = full_suite_ids()
    suite_complete = {str(row["id"]) for row in suite} == expected_ids

    t0 = time.perf_counter()
    print(
        f"suite: {len(suite)} prompts ({total_positions} teacher-forced positions) "
        f"categories={category_counts} heldout={sum(heldout)} complete={suite_complete}",
        flush=True,
    )

    teacher_logits: dict[str, list[np.ndarray]] = {}
    arm_build_s: dict[str, float] = {}
    device_identities: dict[str, object] = {}
    resolved_routes: dict[str, object] = {}
    vocab_size: int | None = None
    student: list[np.ndarray] | None = None
    repeats: list[list[np.ndarray]] = []
    reset_logits: np.ndarray | None = None
    intervening_logits: np.ndarray | None = None
    after_generation_logits: np.ndarray | None = None
    for arm in arms:
        if arm not in {"tp1-d0", "tp1-d1", "tp2"}:
            raise SystemExit(f"unknown arm {arm!r}")
        devices = {"tp1-d0": (0,), "tp1-d1": (1,), "tp2": (0, 1)}[arm]
        mode = "tp2" if arm == "tp2" else "tp1"
        session: object | None = None
        try:
            build0 = time.perf_counter()
            session = _session_factory(MODEL, devices=devices, mode=mode)
            arm_build_s[arm] = time.perf_counter() - build0
            device_identities[arm] = _device_identities(session)
            resolved_routes[arm] = _resolved_route(session)
            vocab_size = int(session.vocab_size)  # type: ignore[attr-defined]
            print(
                f"{arm}: session built in {arm_build_s[arm]:.0f}s "
                f"route={resolved_routes[arm]} vocab={vocab_size}",
                flush=True,
            )
            logits = run_teacher_arm(session, token_rows)
            if arm in {"tp1-d0", "tp1-d1"}:
                teacher_logits[arm] = logits
            else:
                student = logits
                repeats = [student]
                for rep in range(1, max(args.repeat_tp2, 1)):
                    print(f"tp2: determinism repeat {rep}", flush=True)
                    repeats.append(run_teacher_arm(session, token_rows))
                # State boundaries. teacher_forced_logits zeroes all state
                # itself, so these are RESET/REUSE boundaries - they verify a
                # fresh call does not inherit prior state. They do NOT certify
                # warm-state continuation.
                print("tp2: reset/reuse re-run of prompt 0", flush=True)
                reset_logits = run_teacher_arm(session, token_rows[:1])[0]
                if len(token_rows) > 1:
                    print("tp2: intervening-prompt reuse re-run", flush=True)
                    run_teacher_arm(session, token_rows[1:2])
                    intervening_logits = run_teacher_arm(session, token_rows[:1])[0]
                print("tp2: reset-after-generation re-run", flush=True)
                session.generate(token_rows[0], max_new_tokens=2)  # type: ignore[attr-defined]
                after_generation_logits = run_teacher_arm(session, token_rows[:1])[0]
        finally:
            if session is not None:
                session.close()  # type: ignore[attr-defined]
        print(f"{arm}: swept; session closed", flush=True)

    # Finite checks (teacher and student).
    nonfinite: list[str] = []
    for arm_name, arm_logits in teacher_logits.items():
        for index, row in enumerate(arm_logits):
            if not np.isfinite(row).all():
                nonfinite.append(f"{arm_name}:{index}")
    for index, row in enumerate(student or []):
        if not np.isfinite(row).all():
            nonfinite.append(f"tp2:{index}")

    expected_positions = [len(t) for t in token_rows]
    comparisons: dict[str, dict[str, object]] = {}
    for control in ("tp1-d0", "tp1-d1"):
        if control not in teacher_logits or student is None:
            continue
        summary = score_arm(
            teacher_logits[control],
            student,
            categories,
            heldout,
            expected_positions=expected_positions,
            vocab_size=vocab_size,
        )
        comparisons[control] = summary
        print(f"{control}: global={summary['global']}", flush=True)
        print(f"{control}: scopes={summary['scopes']}", flush=True)
        for c in CATEGORIES:
            print(f"  {c}: {summary['categories'][c]}", flush=True)

    # Direct control-vs-control comparison. Identical aggregate metrics against
    # the shared student do NOT prove the two teacher arrays are bit-identical;
    # this is the only measurement that does.
    teacher_cross_check: dict[str, object] | None = None
    if all(control in teacher_logits for control in ("tp1-d0", "tp1-d1")):
        teacher_cross_check = cross_teacher_check(
            teacher_logits["tp1-d0"],
            teacher_logits["tp1-d1"],
            expected_positions=expected_positions,
            vocab_size=vocab_size,
        )
        print(
            "controls: byte_identical="
            f"{teacher_cross_check['byte_identical']} "
            f"differing_rows={teacher_cross_check['differing_rows']} "
            f"max_abs_diff={teacher_cross_check['max_abs_diff']:.6g} "
            f"digests={teacher_cross_check['digest']}",
            flush=True,
        )

    determinism: dict[str, object] | None = None
    if student is not None:
        hashes = [_row_hash(row) for row in student]
        determinism = {
            "sweeps": max(args.repeat_tp2, 1),
            "per_row_match": True,
            "mismatched_sweeps": [],
        }
        for rep, sweep in enumerate(repeats[1:], start=1):
            rep_hashes = [_row_hash(row) for row in sweep]
            if rep_hashes != hashes:
                determinism["per_row_match"] = False
                determinism["mismatched_sweeps"].append(rep)  # type: ignore[union-attr]
    state_boundaries: dict[str, object] | None = None
    if student is not None:
        first_hash = _row_hash(student[0])
        state_boundaries = {
            "prompt_index": 0,
            "certifies_warm_continuation": False,
            "note": (
                "teacher_forced_logits zeroes state per call; these verify "
                "reset/reuse, not warm-state continuation"
            ),
            "reset_reuse_bit_exact": bool(
                reset_logits is not None and _row_hash(reset_logits) == first_hash
            ),
            "intervening_prompt_reuse_bit_exact": bool(
                intervening_logits is not None
                and _row_hash(intervening_logits) == first_hash
            ),
            "reset_after_generation_bit_exact": bool(
                after_generation_logits is not None
                and _row_hash(after_generation_logits) == first_hash
            ),
        }

    all_pass, gate_failures = evaluate_gates(
        comparisons=comparisons,
        categories_present=categories_present,
        suite_has_heldout=any(heldout),
        suite_complete=suite_complete,
        require_suite_complete=not args.allow_partial,
        nonfinite_rows=nonfinite,
        determinism=determinism,
        state_boundaries=state_boundaries,
        teacher_cross_check=teacher_cross_check,
    )

    result: dict[str, object] = {
        "protocol": (
            "full teacher-forced trajectories, canonical suite + heldout-only rows, "
            "matched per-GPU TP1 controls, direct control-vs-control byte comparison, "
            "sequential arms, fail-closed gates"
        ),
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_revision": _git_revision(),
        "source_dirty": _git_dirty(),
        "host": _host_identity(),
        "model": {
            "path": MODEL,
            "size_bytes": os.path.getsize(MODEL),
            "sha256": _model_sha256(MODEL) if args.model_hash == "full" else None,
            "hash_mode": args.model_hash,
        },
        "suite": {
            "canonical": str(CANONICAL_SUITE),
            "heldout": str(HELDOUT_SUITE),
            "prompts": len(suite),
            "positions": total_positions,
            "categories": category_counts,
            "heldout_prompts": int(sum(heldout)),
            "prompt_ids": [str(row["id"]) for row in suite],
            "complete": suite_complete,
            "diagnostic_scope": not suite_complete,
        },
        "arms_run": arms,
        "arm_build_s": arm_build_s,
        "device_identities": device_identities,
        "resolved_routes": resolved_routes,
        "thresholds": {
            **PRODUCTION_GATE,
            "category_top1_agreement": CATEGORY_TOP1,
            "min_determinism_sweeps": MIN_DETERMINISM_SWEEPS,
        },
        "coverage_scale": qualification_label(total_positions),
        "certification": "none (probe only; gates are the verdict)",
        "nonfinite_rows": nonfinite,
        "comparison": comparisons,
        "teacher_cross_check": teacher_cross_check,
        "determinism": determinism,
        "state_boundaries": state_boundaries,
        "all_gates_passed": all_pass,
        "gate_failures": gate_failures,
        "wall_s": time.perf_counter() - t0,
    }

    if args.json:
        _write_artifact(result, args.json)
    print(
        f"done in {result['wall_s']:.0f}s; coverage_scale={result['coverage_scale']} "
        f"all_gates_passed={all_pass} failures={gate_failures}",
        flush=True,
    )
    return 0 if all_pass else 1


def _product_suite():
    from scripts.gguf_mtp_bench import build_chat_prompt
    suite = load_prompt_suite()
    tokenizer = _load_tokenizer()
    if any(len(row['messages']) != 1 or row['messages'][0]['role'] != 'user' for row in suite):
        raise ValueError('product coverage requires the declared single-user prompt suite')
    tokens = [tuple(build_chat_prompt(tokenizer, row['messages'][0]['content'])) for row in suite]
    return suite, tokens


def capture_resident_arm(args) -> int:
    """One bounded fresh-process arm, reusing the session factory and suite logic."""
    from scripts.tp2_resident_control import bind_resident_profile, resolved_scope_manifest
    from scripts.tp2_xtx_tp1_eager_stage_probe import StageRecorder
    from scripts.tp2_teacher_child import validate_logits
    import uuid
    arm = args.capture_resident_arm
    expected_visibility = {'tp1-d0': '0', 'tp1-d1': '1'}.get(arm)
    if expected_visibility is not None and os.environ.get('HIP_VISIBLE_DEVICES') != expected_visibility:
        raise ValueError('resident TP1 controls require fresh physical-device visibility')
    if arm == 'tp2' and any(os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')):
        raise ValueError('TP2 worker requires the unfiltered two-device process')
    os.environ['HIPENGINE_GGUF_DECODE_REPACK'] = '1'
    suite, token_rows = _product_suite()
    if max(map(len, token_rows)) > args.max_sequence_length or len(token_rows[0]) + 2 > args.max_sequence_length:
        raise ValueError('declared capacity does not cover the suite and lifecycle control')
    path = Path(args.json)
    root = path.parent / (path.stem + '-arrays')
    root.mkdir(parents=True, exist_ok=True)
    record = StageRecorder(path, {'kind': 'resident_tp_ar_capture', 'arm': arm,
        'run_id': uuid.uuid4().hex, 'source_revision': _git_revision(), 'host': _host_identity(),
        'command': shlex.join([sys.executable, *sys.argv]),
        'model_sha256': _model_sha256(MODEL), 'performance_claim': False,
        'suite': {'ids': [str(r['id']) for r in suite], 'categories': [r['category'] for r in suite],
                  'heldout': [r['heldout'] for r in suite], 'tokens': token_rows,
                  'positions': sum(map(len, token_rows)), 'renderer': 'product build_chat_prompt'},
        'arrays': [], 'profile': bind_resident_profile(args.execution_profile)})
    control_path = root / 'controls.jsonl'
    current = {'sweep': -1, 'prompt_id': 'build'}
    handle = control_path.open('w')
    def row_hook(value):
        handle.write(json.dumps({**current, **value}) + '\n')
        handle.flush()
    state = {}
    def build():
        devices = (0, 1) if arm == 'tp2' else (0,)
        session = _session_factory(MODEL, devices=devices, mode='tp2' if arm == 'tp2' else 'tp1',
            resident_control=True, capacity=args.max_sequence_length, row_hook=row_hook)
        state['session'] = session
        if arm == 'tp2':
            session._ensure_graph_schedule()
        record.artifact['devices'] = _device_identities(session)
        record.artifact['route'] = _resolved_route(session)
        record.artifact['scope_manifest'] = resolved_scope_manifest(session)
        record.artifact['vocab_size'] = int(session.vocab_size)
        return {'vocab_size': int(session.vocab_size)}
    hashes = []
    def capture(index, save=False):
        session = state['session']
        logits = run_teacher_arm(session, token_rows[index:index+1])[0]
        ok, detail = validate_logits(logits, positions=len(token_rows[index]), vocab_size=session.vocab_size)
        if not ok:
            raise ValueError(detail)
        digest = _row_hash(logits)
        if save:
            out = root / f'prompt-{index}.npy'
            np.save(out, logits)
            record.artifact['arrays'].append({'path': str(out), 'logit_sha256': digest,
                                              'shape': list(logits.shape)})
            hashes.append(digest)
        elif digest != hashes[index]:
            raise ValueError(f'same-schedule/reset mismatch: prompt {index}')
        return {'prompt_id': suite[index]['id'], 'shape': list(logits.shape), 'logit_sha256': digest}
    if record.guard('build', build):
        for sweep in range(args.repeat_tp2):
            for index, row in enumerate(suite):
                current.update(sweep=sweep, prompt_id=str(row['id']))
                if not record.guard(f'sweep-{sweep}/{row["id"]}', lambda i=index, s=sweep: capture(i, save=s == 0)):
                    break
            if record.exit_code:
                break
        if not record.exit_code:
            current.update(sweep=-1, prompt_id=str(suite[0]['id']))
            record.guard('reset-reuse', lambda: capture(0))
            current['prompt_id'] = str(suite[1]['id'])
            record.guard('intervening-prompt', lambda: capture(1))
            current['prompt_id'] = str(suite[0]['id'])
            record.guard('isolation-after-neighbor', lambda: capture(0))
            def generation():
                result = state['session'].generate(token_rows[0], max_new_tokens=2)
                control = getattr(state['session'], 'generation_control', None)
                if control is None:
                    control = {'sampled_sequence': list(result.token_ids),
                               'positions': [t.position for t in result.step_traces],
                               'decode_transitions': sum(t.kind == 'decode' for t in result.step_traces)}
                record.artifact['generation_control'] = control
                return control
            record.guard('generation', generation)
            record.guard('reset-after-generation', lambda: capture(0))
        if not record.exit_code:
            record.artifact['determinism'] = {'sweeps': args.repeat_tp2, 'per_row_match': True}
            record.artifact['state_boundaries'] = {
                'reset_reuse_bit_exact': True, 'intervening_prompt_reuse_bit_exact': True,
                'reset_after_generation_bit_exact': True,
                'scope': 'c1 sequential request isolation/reset/reuse, not concurrent cN serving'}
            record.guard('teardown', state['session'].close)
    handle.close()
    record.artifact['control_log'] = {'path': str(control_path),
        'sha256': hashlib.sha256(control_path.read_bytes()).hexdigest()}
    record.artifact['natural_teardown'] = bool(not record.exit_code)
    record.finish()
    if record.exit_code:
        sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    return 0


def load_resident_capture(path: Path):
    """Reject incomplete, forged-profile, truncated, or nonfinite capture data."""
    from hipengine.execution_profiles import manifest_sha256
    data = json.loads(path.read_text())
    if data.get('status') != 'complete' or data.get('natural_teardown') is not True:
        raise ValueError('capture did not complete natural teardown')
    if data.get('first_bad_stage') is not None or not all(s.get('ok') is True for s in data['stages']):
        raise ValueError('capture contains a failed/incomplete stage')
    if manifest_sha256(data['profile']['manifest']) != data['profile']['manifest_sha256']:
        raise ValueError('profile manifest hash mismatch')
    scope = data['scope_manifest']
    encoded = json.dumps(scope['manifest'], sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(encoded).hexdigest() != scope['sha256']:
        raise ValueError('scope manifest hash mismatch')
    control = Path(data['control_log']['path'])
    if hashlib.sha256(control.read_bytes()).hexdigest() != data['control_log']['sha256']:
        raise ValueError('control log hash mismatch')
    control_rows = [json.loads(line) for line in control.read_text().splitlines() if line.strip()]
    for sweep in range(int(data['determinism']['sweeps'])):
        for prompt_id, tokens in zip(data['suite']['ids'], data['suite']['tokens'], strict=True):
            selected = [r for r in control_rows if r['sweep'] == sweep and r['prompt_id'] == prompt_id]
            phase = 'tp2-output' if data['arm'] == 'tp2' else 'head-complete'
            outputs = [r for r in selected if r['phase'] == phase]
            if [r['position'] for r in outputs] != list(range(len(tokens))):
                raise ValueError('control log lacks complete ordered output positions')
            if [r['input_token'] for r in outputs] != tokens:
                raise ValueError('control log has mismatched teacher inputs')
            if data['arm'] == 'tp2':
                for rank in (0, 1):
                    inputs = [r for r in selected if r['phase'] == 'tp2-input' and r['logical_device'] == rank]
                    if [r['input_token'] for r in inputs] != tokens or [r['position_context'] for r in inputs] != [[i, i+1] for i in range(len(tokens))]:
                        raise ValueError('rank input/position control mismatch')
            else:
                states = [r for r in selected if r['phase'] == 'resident-state']
                if len(states) != 1 or states[0]['device_token_rows'] != tokens or states[0]['position_context'] != [len(tokens), len(tokens)+1]:
                    raise ValueError('resident input/state control mismatch')
    entries = data['arrays']
    if len(entries) != len(data['suite']['ids']):
        raise ValueError('capture prompt array count mismatch')
    arrays = []
    for entry, tokens in zip(entries, data['suite']['tokens'], strict=True):
        array = np.load(entry['path'], mmap_mode='r')
        if array.dtype != np.float32 or array.shape != (len(tokens), data['vocab_size']) or not np.isfinite(array).all():
            raise ValueError('capture trajectory shape/finiteness mismatch')
        if _row_hash(array) != entry['logit_sha256']:
            raise ValueError('capture logit hash mismatch')
        arrays.append(array)
    return data, arrays


def report_resident_coverage(args) -> int:
    """CPU aggregation through the existing category/scope numerical evaluator."""
    captures = [load_resident_capture(path) for path in args.resident_results]
    by_arm = {data['arm']: (data, arrays) for data, arrays in captures}
    if set(by_arm) != {'tp1-d0', 'tp1-d1', 'tp2'} or len(by_arm) != len(captures):
        raise ValueError('need exactly three distinct capture arms')
    if len({d['run_id'] for d, _ in captures}) != 3:
        raise ValueError('capture run identity reused')
    first = by_arm['tp1-d0'][0]
    for data, _ in captures:
        for key in ('suite', 'model_sha256', 'host', 'source_revision', 'vocab_size', 'profile'):
            if data[key] != first[key]:
                raise ValueError(f'cross-arm {key} mismatch')
    suite, tokens = _product_suite()
    if first['suite']['ids'] != [r['id'] for r in suite] or first['suite']['tokens'] != [list(t) for t in tokens]:
        raise ValueError('captures do not match current product prompt suite')
    uuid0 = by_arm['tp1-d0'][0]['devices']['0']['uuid']
    uuid1 = by_arm['tp1-d1'][0]['devices']['0']['uuid']
    if uuid0 == uuid1 or {uuid0, uuid1} != {v['uuid'] for v in by_arm['tp2'][0]['devices'].values()}:
        raise ValueError('physical control devices do not match distinct TP2 ranks')
    categories = [str(r['category']) for r in suite]
    heldout = [bool(r['heldout']) for r in suite]
    expected = [len(t) for t in tokens]
    comparisons = {arm: score_arm(by_arm[arm][1], by_arm['tp2'][1], categories, heldout,
        expected_positions=expected, vocab_size=first['vocab_size']) for arm in ('tp1-d0', 'tp1-d1')}
    cross = cross_teacher_check(by_arm['tp1-d0'][1], by_arm['tp1-d1'][1],
                               expected_positions=expected, vocab_size=first['vocab_size'])
    passed, failures = evaluate_gates(comparisons=comparisons, categories_present=set(categories),
        suite_has_heldout=any(heldout), suite_complete=True, nonfinite_rows=[],
        determinism=by_arm['tp2'][0]['determinism'], state_boundaries=by_arm['tp2'][0]['state_boundaries'],
        teacher_cross_check=cross)
    for data, _ in captures:
        if data['determinism']['sweeps'] < 3 or data['determinism']['per_row_match'] is not True:
            failures.append(f'{data["arm"]} determinism incomplete')
        for key in ('reset_reuse_bit_exact', 'intervening_prompt_reuse_bit_exact', 'reset_after_generation_bit_exact'):
            if data['state_boundaries'][key] is not True:
                failures.append(f'{data["arm"]} {key} failed')
    output = {'kind': 'optimized_resident_tp1_tp2_coverage', 'performance_claim': False,
        'production_qualified': False, 'all_gates_passed': passed and not failures,
        'gate_failures': failures, 'suite': first['suite'], 'coverage_scale': 'extended_probe',
        'population_note': 'Natural causal prompt prefixes, not a sustained generated-token or full task population.',
        'model_sha256': first['model_sha256'], 'host': first['host'], 'profile': first['profile'],
        'comparison': comparisons, 'teacher_cross_check': cross,
        'controls': {data['arm']: {k: data[k] for k in ('run_id', 'devices', 'route', 'scope_manifest',
            'determinism', 'state_boundaries', 'generation_control', 'control_log', 'command', 'natural_teardown')} for data, _ in captures},
        'captures': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.resident_results},
        'thresholds': {**PRODUCTION_GATE, 'category_top1': CATEGORY_TOP1}}
    _write_artifact(output, args.json)
    return 0 if output['all_gates_passed'] else 1


def sustained_trajectory(adapter, tokens, *, forced=None, steps=128, progress=None, reference=None, failure=None, gate_rows=None):
    """Untimed AR graph/native trajectory; input positions are P..P+D-1.

    ``gate_rows`` limits the absolute-ceiling check to the first N rows while
    still running and returning all ``steps`` of them, so a capture can complete
    past a declared comparison horizon (docs/EXECUTION-PROFILES.md 6.5) with the
    rows beyond it preserved as a diagnostic. ``None`` gates every row.
    """
    from scripts.tp2_xtx_tp1_eager_stage_probe import validate_result
    if forced is not None and len(forced) != steps: raise ValueError('forced trajectory length mismatch')
    if gate_rows is not None and not 0 < gate_rows <= steps: raise ValueError('gate_rows outside 1..steps')
    next_token = adapter.prefill(tokens)
    adapter.begin_decode(steps)
    inputs, output, controls = [], [], []
    for i in range(steps):
        token = int(next_token if forced is None else forced[i])
        position = len(tokens)+i
        if progress: progress({'phase':'start','position':position,'input_token':token})
        adapter.force_input(token)
        result = adapter.transition(token,return_logits=True)
        validate_result(result,adapter.vocab_size)
        row = np.asarray(result.logits,dtype=np.float32).reshape(-1).copy()
        control = adapter.check_transition(token,position)
        if control.get('position') != position or control.get('input_token') != token:
            raise ValueError('sustained control position/input mismatch')
        if reference is not None and (gate_rows is None or i < gate_rows):
            kl,top1 = _kl_rows(reference[i:i+1],row[None,:])
            if float(kl[0]) > PRODUCTION_GATE['max_kl']:
                detail = {'position':position,'decode_index':i,'input_token':token,
                          'kl':float(kl[0]),'top1':bool(top1[0]),
                          'reference_top1':int(np.argmax(reference[i])), 'candidate_top1':int(np.argmax(row))}
                if failure: failure(detail,reference[i],row)
                raise ValueError(f'sustained absolute KL ceiling failed: {detail}')
        inputs.append(token); output.append(row); controls.append(control)
        next_token = int(result.token_id)
        if progress: progress({'phase':'complete',**control})
    adapter.end_decode()
    return np.stack(output),inputs,controls


def load_sustained(path):
    data = json.loads(Path(path).read_text())
    if data.get('status')!='complete' or data.get('natural_teardown') is not True or data.get('first_bad_stage') is not None:
        raise ValueError('incomplete sustained capture')
    if data.get('determinism',{}).get('sweeps',0)<3 or data['determinism'].get('per_row_match') is not True:
        raise ValueError('missing sustained determinism')
    if not all(data['state_boundaries'].values()): raise ValueError('sustained lifecycle failure')
    arrays=[]
    for entry,inputs in zip(data['arrays'],data['forced_inputs'],strict=True):
        a=np.load(entry['path'],mmap_mode='r')
        if a.dtype!=np.float32 or a.shape!=(128,data['vocab_size']) or len(inputs)!=128 or not np.isfinite(a).all() or _row_hash(a)!=entry['sha256']:
            raise ValueError('sustained logit evidence invalid')
        arrays.append(a)
    if len(arrays)!=18: raise ValueError('sustained suite incomplete')
    control=Path(data['control_log']['path'])
    if hashlib.sha256(control.read_bytes()).hexdigest()!=data['control_log']['sha256']:
        raise ValueError('sustained controls hash mismatch')
    records=[json.loads(line) for line in control.read_text().splitlines()]
    for sweep in range(data['determinism']['sweeps']):
        for i,prompt_id in enumerate(data['suite']['ids']):
            out=[r for r in records if r['sweep']==sweep and r['prompt_id']==prompt_id and r['phase']=='complete']
            if [r['position'] for r in out]!=list(range(len(data['suite']['tokens'][i]),len(data['suite']['tokens'][i])+128)) or [r['input_token'] for r in out]!=data['forced_inputs'][i]:
                raise ValueError('sustained position/input controls mismatch')
    return data,arrays


def capture_sustained_arm(args):
    from scripts.tp2_resident_control import create_native_adapter, bind_resident_profile, resolved_scope_manifest
    from scripts.tp2_matched_ar_baseline import product_identity
    from scripts.tp2_xtx_tp1_eager_stage_probe import StageRecorder
    import uuid
    arm=args.sustained_arm
    if args.max_sequence_length!=200: raise ValueError('D128 gate requires declared capacity 200')
    expected={'tp1-d0':'0','tp1-d1':'1'}.get(arm)
    if os.environ.get('HIP_VISIBLE_DEVICES')!=expected: raise ValueError('sustained physical visibility mismatch')
    suite,tokens=_product_suite()
    identity=product_identity(MODEL)
    reference_data,reference=(load_sustained(args.teacher_source) if args.teacher_source else (None,None))
    if reference_data is None and arm!='tp1-d0': raise ValueError('only tp1-d0 may choose teacher trajectories')
    if reference_data is not None and numerical_identity(reference_data['identity'])!=numerical_identity(identity):
        raise ValueError('teacher identity mismatch')
    path=Path(args.json); root=path.parent/(path.stem+'-arrays'); root.mkdir(parents=True,exist_ok=True)
    record=StageRecorder(path,{'kind':'tp2_sustained_d128','arm':arm,'run_id':uuid.uuid4().hex,
        'identity':identity,'profile':bind_resident_profile('production'),'command':shlex.join([sys.executable,*sys.argv]),
        'suite':{'ids':[r['id'] for r in suite], 'categories':[r['category'] for r in suite],
                 'heldout':[r['heldout'] for r in suite], 'tokens':tokens},
        'forced_inputs':[] if reference_data is None else reference_data['forced_inputs'], 'arrays':[]})
    control_path=root/'controls.jsonl'; handle=control_path.open('w'); current={}
    def progress(value):
        handle.write(json.dumps({**current,**value})+'\n'); handle.flush()
    def failure(detail,teacher,candidate):
        fixture=root/'first-numerical-failure.npz'; np.savez(fixture,teacher=teacher,candidate=candidate)
        record.artifact['numerical_failure']={**current,**detail,'fixture':str(fixture)}
    state={}; hashes=[]; first_arrays=[]
    bulk=bool(getattr(args,'tp2_bulk_prefill',False))
    if bulk and arm!='tp2': raise ValueError('bulk prefill is a tp2-arm candidate')
    horizon=getattr(args,'horizon',None)
    if horizon is not None and not 0 < int(horizon) <= 128: raise ValueError('horizon outside 1..128 captured rows')
    record.artifact['horizon']=int(horizon) if horizon is not None else 128
    record.artifact['horizon_declared_by']=('docs/EXECUTION-PROFILES.md 6.5' if horizon is not None else None)
    def build():
        a=create_native_adapter(MODEL,arm,capacity=200,bulk_prefill=bulk); state['adapter']=a; a.prepare()
        record.artifact.update(vocab_size=a.vocab_size,devices=_device_identities(a.owner),
            route=_resolved_route(a.owner),scope_manifest=resolved_scope_manifest(a.owner),
            prefill_schedule=a.prefill_schedule,tp2_bulk_prefill_requested=bulk)
        return {'vocab_size':a.vocab_size}
    def capture(i,save=False):
        forced=(None if reference_data is None and save else record.artifact['forced_inputs'][i])
        logits,inputs,controls=sustained_trajectory(state['adapter'],tokens[i],forced=forced,
            progress=progress,reference=None if reference is None else reference[i],failure=failure,
            gate_rows=None if horizon is None else int(horizon))
        digest=_row_hash(logits)
        if save:
            out=root/f'prompt-{i}.npy'; np.save(out,logits)
            record.artifact['arrays'].append({'path':str(out),'sha256':digest,'shape':list(logits.shape)})
            hashes.append(digest); first_arrays.append(np.load(out,mmap_mode='r'))
            if reference_data is None: record.artifact['forced_inputs'].append(inputs)
        elif digest!=hashes[i]: raise ValueError('sustained same-schedule/reset mismatch')
        return {'shape':list(logits.shape),'sha256':digest,'start_position':len(tokens[i]),'end_position':len(tokens[i])+127}
    if record.guard('build',build):
        for sweep in range(args.repeat_tp2):
            for i,row in enumerate(suite):
                current.update(sweep=sweep,prompt_id=row['id'])
                if not record.guard(f'sweep-{sweep}/{row["id"]}',lambda i=i,s=sweep:capture(i,save=s==0)): break
            if record.exit_code: break
            if sweep==0 and reference is not None:
                def gate():
                    # The declared horizon bounds this check exactly as it bounds
                    # the per-row ceiling above; the rows past it stay in the
                    # artifact and are scored again by the report as a diagnostic.
                    scored=slice(None) if horizon is None else slice(0,int(horizon))
                    summary=score_arm([r[scored] for r in reference],[a[scored] for a in first_arrays],
                                      [r['category'] for r in suite], [r['heldout'] for r in suite],
                                      expected_positions=[len(reference[0]) if horizon is None else int(horizon)]*len(suite),
                                      vocab_size=state['adapter'].vocab_size)
                    record.artifact['comparison']=summary
                    checks=[_envelope_gate(summary['global'],top1_bar=.99)]
                    checks.extend(_envelope_gate(v,top1_bar=.99) for v in summary['scopes'].values())
                    checks.extend(_envelope_gate(v,top1_bar=.97) for v in summary['categories'].values())
                    checks.extend(_envelope_gate(v,top1_bar=.97) for scopes in summary['category_scopes'].values() for v in scopes.values())
                    if not all(c['passed'] for c in checks): raise ValueError(f'sustained envelope failure: {checks}')
                    return {'envelope_passed':True}
                if not record.guard('sustained-numerical-gate',gate): break
        if not record.exit_code:
            current.update(sweep=-1,prompt_id=suite[0]['id']); record.guard('reset-reuse',lambda:capture(0))
            current['prompt_id']=suite[1]['id']; record.guard('neighbor',lambda:capture(1))
            current['prompt_id']=suite[0]['id']; record.guard('isolation',lambda:capture(0))
            def generation():
                a=state['adapter']; t=a.prefill(tokens[0]); a.begin_decode(2)
                for _ in range(2): t=int(a.transition(t,return_logits=True).token_id)
                a.end_decode()
            record.guard('generation',generation)
            record.guard('reset-after-generation',lambda:capture(0))
        if not record.exit_code:
            record.artifact['determinism']={'sweeps':args.repeat_tp2,'per_row_match':True}
            record.artifact['state_boundaries']={'reset':True,'isolation':True,'reset_after_generation':True}
            record.guard('graph-destroy',state['adapter'].destroy_graphs)
            record.guard('teardown',state['adapter'].close)
    handle.close()
    record.artifact['control_log']={'path':str(control_path),'sha256':hashlib.sha256(control_path.read_bytes()).hexdigest()}
    record.artifact['natural_teardown']=not bool(record.exit_code); record.finish()
    if record.exit_code: sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
    return 0


def _rows_per_prompt(arrays) -> int:
    """Captured teacher-forced rows per prompt; 128 is the capture standard.

    ``load_sustained`` already rejects any capture whose arrays are not
    ``(128, vocab)``, so the fallback only applies to synthetic inputs.
    """

    shape = getattr(arrays[0], 'shape', ())
    return int(shape[0]) if shape else 128


def report_sustained(args):
    captures=[load_sustained(p) for p in args.sustained_report]
    arms={d['arm']:(d,a) for d,a in captures}
    if set(arms)!={'tp1-d0','tp1-d1','tp2'}: raise ValueError('sustained arms missing/duplicated')
    schedules={d['arm']:d.get('prefill_schedule') for d,_ in captures}
    missing=[arm for arm,schedule in schedules.items() if schedule is None]
    if missing: raise ValueError(f'sustained missing prefill_schedule provenance for arms {missing}')
    rows_per_prompt=_rows_per_prompt(captures[0][1])
    requested=getattr(args,'horizon',None)
    horizon=rows_per_prompt if requested is None else int(requested)
    if not 1 <= horizon <= rows_per_prompt:
        raise ValueError(f'horizon {horizon} outside 1..{rows_per_prompt} captured rows')
    # Different native prefill algorithms are legitimate product paths. They are
    # not required to be identical; each arm must still satisfy the unchanged
    # production numerical envelope against the shared teacher. Record which
    # schedules were compared so a mixed comparison is explicit, never silent.
    mixed=len(set(schedules.values()))>1
    comparison_scope='mixed-prefill-schedules' if mixed else 'matched-prefill-schedule'
    teacher,reference=arms['tp1-d0']; comparisons={}; full_horizon={}
    for arm,(data,arrays) in arms.items():
        if numerical_identity(data['identity'])!=numerical_identity(teacher['identity']) or data['suite']!=teacher['suite'] or data['forced_inputs']!=teacher['forced_inputs']:
            raise ValueError('sustained shared teacher provenance mismatch')
        if arm != 'tp1-d0':  # The teacher is the reference, not a candidate to score against itself.
            if horizon < rows_per_prompt:
                scored_reference=[r[:horizon] for r in reference]
                scored_arrays=[a[:horizon] for a in arrays]
            else:
                scored_reference, scored_arrays = reference, arrays
            comparisons[arm]=score_arm(scored_reference,scored_arrays,
                teacher['suite']['categories'],teacher['suite']['heldout'],
                expected_positions=[horizon]*18,vocab_size=teacher['vocab_size'])
            if horizon < rows_per_prompt:
                # The declared horizon bounds the claim, not the record: the rows
                # past it stay in the artifact as an unscored diagnostic.
                full_horizon[arm]=score_arm(reference,arrays,teacher['suite']['categories'],
                    teacher['suite']['heldout'],expected_positions=[rows_per_prompt]*18,
                    vocab_size=teacher['vocab_size'])
    passed=all(_envelope_gate(v,top1_bar=.99)['passed'] for c in comparisons.values() for v in [c['global'],*c['scopes'].values()])
    passed &= all(_envelope_gate(v,top1_bar=.97)['passed'] for c in comparisons.values() for v in [*c['categories'].values(),*(v for s in c['category_scopes'].values() for v in s.values())])
    result={'kind':'tp2_sustained_d128_gate','all_gates_passed':bool(passed),'production_qualified':False,
        'population':(f'18 product prompts, first {horizon} aligned generated decode transitions each; '
                      f'shared tp1-d0 chosen trajectories' if horizon<rows_per_prompt else
                      '18 product prompts, 128 aligned generated decode transitions each; shared tp1-d0 chosen trajectories'),
        'horizon':horizon,'horizon_declared_by':'docs/EXECUTION-PROFILES.md 6.5',
        'captured_rows_per_prompt':rows_per_prompt,
        'reference_arm':'tp1-d0', 'prefill_schedules':schedules,
        'mixed_prefill_schedules':bool(mixed),'comparison_scope':comparison_scope,
        'positions':18*horizon,'suite':teacher['suite'],'identity':teacher['identity'],'profile':teacher['profile'],'comparison':comparisons,
        'beyond_horizon_diagnostic':full_horizon or None,
        'thresholds':PRODUCTION_GATE,'captures':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.sustained_report},
        'controls':{d['arm']:{k:d[k] for k in ('run_id','devices','route','scope_manifest','determinism','state_boundaries','control_log','natural_teardown')} for d,a in captures},
        'identity_host_scheduling':{d['arm']:{f:(d['identity'].get('host') or {}).get(f) for f in _VOLATILE_IDENTITY_FIELDS} for d,a in captures},
        'identity_equality_scope':('host scheduling fields excluded: '
            + ', '.join(_VOLATILE_IDENTITY_FIELDS) + ' are recorded per arm but not compared'),
        'not_qualified':['task quality','BF16-relative','public distributed profile','performance promotion']}
    _write_artifact(result,args.json)
    return 0 if passed else 1


def _write_artifact(result: dict[str, object], path: str | None) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"artifact: {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
