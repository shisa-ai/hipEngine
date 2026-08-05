#!/usr/bin/env python3
"""Screen 1,024 versus 4,096 GGUF full-attention prefill query rows.

The parent mode launches one right-sized, cached-only benchmark process for
both query-row policies at 512/4K/32K/64K and wraps every process in a 10-ms
whole-device GTT sampler.  All five prefill chunk surfaces are set explicitly;
changing only the query field would disable autotuning and accidentally leave
other layer families unchunked.

Two additional state-child processes reuse the largest required capacity and
compare full-model byte fingerprints at every context: prefill logits, final
prompt hidden/layer/Conv/GDN/KV state, a fixed-input decode-logit trajectory,
and final decode state.  Timings from the state children are not performance
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import collect_artifact_provenance  # noqa: E402
from hipengine.util.amdgpu_vram import VramSampler, select_card  # noqa: E402


KIND = "hipengine_gguf_sh_m1_query_rows_screen"
STATE_KIND = "hipengine_gguf_sh_m1_state_child"
SCHEMA_VERSION = 1
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_WORKLOADS = (512, 4096, 32768, 65536)
BASELINE_QUERY_ROWS = 4096
CANDIDATE_QUERY_ROWS = 1024
FIXED_LAYER_ROWS = 1024
MODES = ("q4096", "q1024")
_GIB = 1 << 30


class ScreenError(RuntimeError):
    """Raised when the SH-M1 screen cannot produce fail-closed evidence."""


def _parse_length(text: str) -> int:
    value = str(text).strip().lower()
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1024
        value = value[:-1]
    try:
        parsed = int(value) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workload lengths must be positive integers") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("workload lengths must be positive")
    return parsed


def mode_for_query_rows(query_rows: int) -> str:
    rows = int(query_rows)
    if rows == BASELINE_QUERY_ROWS:
        return "q4096"
    if rows == CANDIDATE_QUERY_ROWS:
        return "q1024"
    raise ScreenError(f"unsupported SH-M1 query-row policy: {rows}")


def chunk_sizes(query_rows: int) -> dict[str, int]:
    mode_for_query_rows(query_rows)
    return {
        "linear": FIXED_LAYER_ROWS,
        "moe": FIXED_LAYER_ROWS,
        "full_attn_query": int(query_rows),
        "full_attn_post": FIXED_LAYER_ROWS,
        "full_attn_rope": FIXED_LAYER_ROWS,
    }


def _rounded_capacity(
    prompt_length: int,
    *,
    decode_tokens: int,
    warmup_decode_tokens: int,
) -> int:
    requested = int(prompt_length) + int(decode_tokens) + int(warmup_decode_tokens) + 1
    return ((requested + 255) // 256) * 256


def _expected_scratch_rows(
    prompt_length: int,
    *,
    query_rows: int,
    decode_tokens: int,
    warmup_decode_tokens: int,
) -> int:
    capacity = _rounded_capacity(
        prompt_length,
        decode_tokens=decode_tokens,
        warmup_decode_tokens=warmup_decode_tokens,
    )
    return min(capacity, max(FIXED_LAYER_ROWS, int(query_rows)))


def build_benchmark_command(
    *,
    python: str,
    model: Path,
    prompt_length: int,
    query_rows: int,
    decode_tokens: int,
    warmup_decode_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    compiler_version_file: Path,
    output: Path,
) -> list[str]:
    chunks = chunk_sizes(query_rows)
    return [
        str(python),
        str(REPO_ROOT / "scripts" / "qwen35_gguf_bench.py"),
        "--model",
        str(model),
        "--quant",
        "gguf_q4_k_m",
        "--token-id",
        "9707",
        "--prompt-length",
        str(int(prompt_length)),
        "--decode-tokens",
        str(int(decode_tokens)),
        "--warmup-decode-tokens",
        str(int(warmup_decode_tokens)),
        "--warmup-runs",
        str(int(warmup_runs)),
        "--measured-runs",
        str(int(measured_runs)),
        "--persistent-session",
        "--force-bulk-prefill",
        "--bulk-prefill-attention-mode",
        "bulk",
        "--use-wmma-prefill",
        "--use-gemv-decode",
        "--no-graph-replay-decode",
        "--prefill-linear-chunk-size",
        str(chunks["linear"]),
        "--prefill-moe-chunk-size",
        str(chunks["moe"]),
        "--prefill-full-attn-query-chunk-size",
        str(chunks["full_attn_query"]),
        "--prefill-full-attn-post-chunk-size",
        str(chunks["full_attn_post"]),
        "--prefill-full-attn-rope-chunk-size",
        str(chunks["full_attn_rope"]),
        "--prefill-chunk-autotune",
        "--compiler-version-file",
        str(compiler_version_file),
        "--require-cached-build",
        "--json",
        str(output),
    ]


def _session_buffers_from_run(run: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshots = run.get("memory_snapshots")
    if not isinstance(snapshots, Mapping):
        raise ScreenError("benchmark run has no memory snapshots")
    for label in ("after_load", "after_prefill", "before_close"):
        snapshot = snapshots.get(label)
        if not isinstance(snapshot, Mapping):
            continue
        breakdown = snapshot.get("owned_session_breakdown")
        if not isinstance(breakdown, Mapping):
            continue
        families = breakdown.get("families")
        if not isinstance(families, Mapping):
            continue
        session_buffers = families.get("session_buffers")
        if isinstance(session_buffers, Mapping):
            return session_buffers
    raise ScreenError("benchmark run has no owned session-buffer census")


def _mapping_ints(payload: Mapping[str, Any], keys: Sequence[str]) -> dict[str, int]:
    try:
        return {str(key): int(payload[key]) for key in keys}
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenError(f"malformed chunk-size record: {payload!r}") from exc


def validate_benchmark_leg(
    payload: Mapping[str, Any],
    *,
    query_rows: int,
    prompt_length: int,
    decode_tokens: int,
    warmup_decode_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    expected_token_id: int = 9707,
) -> None:
    expected_chunks = chunk_sizes(query_rows)
    checks = {
        "persistent_session": payload.get("persistent_session") is True,
        "prompt_length": int(payload.get("prompt_length", -1)) == int(prompt_length),
        "decode_tokens": int(payload.get("decode_tokens", -1)) == int(decode_tokens),
        "warmup_decode_tokens": int(payload.get("warmup_decode_tokens", -1))
        == int(warmup_decode_tokens),
        "warmup_runs": int(payload.get("warmup_runs", -1)) == int(warmup_runs),
        "measured_runs": int(payload.get("measured_runs", -1)) == int(measured_runs),
        "resolved_backend": payload.get("resolved_backend") == "hip_gfx1151",
        "target_arch": payload.get("target_arch") == "gfx1151",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ScreenError(f"benchmark leg failed protocol fields: {failed}")
    requested = payload.get("requested_prefill_chunk_sizes")
    if not isinstance(requested, Mapping) or _mapping_ints(requested, tuple(expected_chunks)) != expected_chunks:
        raise ScreenError("benchmark leg did not request every SH-M1 chunk size explicitly")

    all_chunks = payload.get("prefill_chunk_sizes_all")
    expected_run_count = int(warmup_runs) + int(measured_runs)
    if not isinstance(all_chunks, list) or len(all_chunks) != expected_run_count:
        raise ScreenError("benchmark leg has incomplete resolved chunk-size rows")
    for resolved in all_chunks:
        if not isinstance(resolved, Mapping) or _mapping_ints(resolved, tuple(expected_chunks)) != expected_chunks:
            raise ScreenError("benchmark leg resolved a chunk policy different from the requested SH-M1 leg")

    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != expected_run_count:
        raise ScreenError("benchmark leg has an incomplete warmup/measurement grid")
    expected_scratch = _expected_scratch_rows(
        prompt_length,
        query_rows=query_rows,
        decode_tokens=decode_tokens,
        warmup_decode_tokens=warmup_decode_tokens,
    )
    for run in runs:
        if not isinstance(run, Mapping):
            raise ScreenError("benchmark run is malformed")
        if run.get("resolved_backend") != "hip_gfx1151" or run.get("target_arch") != "gfx1151":
            raise ScreenError("benchmark run left the gfx1151 backend")
        if run.get("effective_use_wmma_prefill") is not True:
            raise ScreenError("benchmark run did not keep qualified WMMA prefill active")
        if run.get("effective_use_gemv_decode") is not True:
            raise ScreenError("benchmark run did not keep qualified GEMV decode active")
        correctness = run.get("correctness_sanity")
        if not isinstance(correctness, Mapping) or correctness.get("finite_final_logits") is not True:
            raise ScreenError("benchmark run did not report finite final logits")
        if int(correctness.get("final_token_id", -1)) != int(expected_token_id):
            raise ScreenError("benchmark run changed the exact expected token")
        buffers = _session_buffers_from_run(run)
        try:
            observed_rows = int(buffers["bulk_prefill_scratch_rows"])
            census_rows = int(buffers["bulk_prefill_scratch_census"]["rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScreenError("benchmark run has malformed scratch rows") from exc
        if observed_rows != expected_scratch or census_rows != expected_scratch:
            raise ScreenError(
                f"benchmark scratch rows {observed_rows}/{census_rows} != expected {expected_scratch}"
            )

    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("finite_final_logits_all") is not True:
        raise ScreenError("benchmark summary did not preserve finite logits")
    final_ids = summary.get("final_token_ids")
    if not isinstance(final_ids, list) or len(final_ids) != int(measured_runs):
        raise ScreenError("benchmark summary has incomplete measured final tokens")
    if any(int(token) != int(expected_token_id) for token in final_ids):
        raise ScreenError("benchmark measured trajectory changed the exact expected token")


def _context_map(payload: Mapping[str, Any], *, label: str) -> dict[int, Mapping[str, Any]]:
    rows = payload.get("contexts")
    if not isinstance(rows, list):
        raise ScreenError(f"{label} state child has no context rows")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ScreenError(f"{label} state child has a malformed context row")
        try:
            context = int(row["prompt_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScreenError(f"{label} state child has a malformed context length") from exc
        if context in result:
            raise ScreenError(f"{label} state child has duplicate context {context}")
        result[context] = row
    return result


def _validate_state_child(
    payload: Mapping[str, Any],
    *,
    query_rows: int,
    expected_contexts: Sequence[int],
) -> dict[int, Mapping[str, Any]]:
    if payload.get("kind") != STATE_KIND or int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ScreenError("state child kind/schema mismatch")
    if payload.get("mode") != mode_for_query_rows(query_rows) or int(payload.get("query_rows", -1)) != int(query_rows):
        raise ScreenError("state child mode/query rows mismatch")
    if payload.get("resolved_backend") != "hip_gfx1151" or payload.get("target_arch") != "gfx1151":
        raise ScreenError("state child left the gfx1151 backend")
    chunks = payload.get("prefill_chunk_sizes")
    if not isinstance(chunks, Mapping) or _mapping_ints(chunks, tuple(chunk_sizes(query_rows))) != chunk_sizes(query_rows):
        raise ScreenError("state child did not resolve the explicit SH-M1 chunk policy")
    capacity = int(payload.get("bulk_prefill_scratch_capacity", 0))
    expected_scratch = min(capacity, max(FIXED_LAYER_ROWS, int(query_rows)))
    if capacity <= 0 or int(payload.get("bulk_prefill_scratch_rows", -1)) != expected_scratch:
        raise ScreenError("state child resolved unexpected scratch rows")
    rows = _context_map(payload, label=mode_for_query_rows(query_rows))
    expected = tuple(int(value) for value in expected_contexts)
    if tuple(sorted(rows)) != tuple(sorted(expected)):
        raise ScreenError("state child context grid is incomplete")
    for context, row in rows.items():
        if row.get("finite") is not True:
            raise ScreenError(f"state child context {context} contains non-finite values")
    return rows


def compare_state_children(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_contexts: Sequence[int],
) -> dict[str, Any]:
    baseline_rows = _validate_state_child(
        baseline,
        query_rows=BASELINE_QUERY_ROWS,
        expected_contexts=expected_contexts,
    )
    candidate_rows = _validate_state_child(
        candidate,
        query_rows=CANDIDATE_QUERY_ROWS,
        expected_contexts=expected_contexts,
    )
    rows: list[dict[str, Any]] = []
    for context in expected_contexts:
        baseline_row = baseline_rows[int(context)]
        candidate_row = candidate_rows[int(context)]
        comparison = {
            "prompt_length": int(context),
            "prefill_logits_exact": baseline_row.get("prefill_logits")
            == candidate_row.get("prefill_logits"),
            "prefill_state_exact": baseline_row.get("prefill_state")
            == candidate_row.get("prefill_state"),
            "trajectory_exact": baseline_row.get("trajectory")
            == candidate_row.get("trajectory"),
            "final_state_exact": baseline_row.get("final_state")
            == candidate_row.get("final_state"),
        }
        comparison["passed"] = all(
            bool(comparison[key])
            for key in (
                "prefill_logits_exact",
                "prefill_state_exact",
                "trajectory_exact",
                "final_state_exact",
            )
        )
        rows.append(comparison)
    return {
        "passed": all(bool(row["passed"]) for row in rows),
        "contexts": rows,
        "baseline_scratch_rows": int(baseline["bulk_prefill_scratch_rows"]),
        "candidate_scratch_rows": int(candidate["bulk_prefill_scratch_rows"]),
        "protocol": (
            "byte-exact FP32 logits plus hidden/layer/Conv/GDN/live-BF16-KV "
            "fingerprints after prefill and the fixed-input decode trajectory"
        ),
    }


def _median(summary: Mapping[str, Any], field: str) -> float:
    row = summary.get(field)
    if not isinstance(row, Mapping):
        raise ScreenError(f"benchmark summary has no {field}")
    try:
        value = float(row["median"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenError(f"benchmark summary has malformed {field}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ScreenError(f"benchmark summary has invalid {field}: {value!r}")
    return value


def _tracked_reclamation(payload: Mapping[str, Any]) -> dict[str, int | bool]:
    memory = payload.get("persistent_session_memory")
    if not isinstance(memory, Mapping):
        raise ScreenError("benchmark leg has no persistent-session memory record")
    snapshots = memory.get("snapshots")
    if not isinstance(snapshots, Mapping):
        raise ScreenError("benchmark leg has no lifecycle memory snapshots")
    try:
        before = int(snapshots["before_load"]["tracked"]["current_allocated_bytes"])
        after = int(snapshots["after_close"]["tracked"]["current_allocated_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ScreenError("benchmark leg has malformed close-to-baseline memory snapshots") from exc
    return {
        "before_load_bytes": before,
        "after_close_bytes": after,
        "delta_bytes": after - before,
        "within_1_mib": abs(after - before) <= 1 * 1024 * 1024,
    }


def _scratch_census(payload: Mapping[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ScreenError("benchmark leg has no runs for scratch census")
    buffers = _session_buffers_from_run(runs[0])
    census = buffers.get("bulk_prefill_scratch_census")
    if not isinstance(census, Mapping):
        raise ScreenError("benchmark leg has no bulk-prefill scratch census")
    return {
        "rows": int(buffers["bulk_prefill_scratch_rows"]),
        "physical_owner_bytes": int(census["physical_owner_bytes"]),
        "physical_owner_gib": float(census.get("physical_owner_gib", int(census["physical_owner_bytes"]) / _GIB)),
    }


def summarize_context(
    *,
    prompt_length: int,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_gtt: Mapping[str, Any],
    candidate_gtt: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_summary = baseline.get("summary")
    candidate_summary = candidate.get("summary")
    if not isinstance(baseline_summary, Mapping) or not isinstance(candidate_summary, Mapping):
        raise ScreenError("benchmark leg has no summary")
    baseline_prefill = _median(baseline_summary, "prefill_tok_s")
    candidate_prefill = _median(candidate_summary, "prefill_tok_s")
    baseline_decode = _median(baseline_summary, "decode_tok_s")
    candidate_decode = _median(candidate_summary, "decode_tok_s")
    baseline_tracked = _median(baseline_summary, "tracked_peak_allocated_gib")
    candidate_tracked = _median(candidate_summary, "tracked_peak_allocated_gib")
    baseline_owned = _median(baseline_summary, "owned_session_peak_gib")
    candidate_owned = _median(candidate_summary, "owned_session_peak_gib")
    baseline_peak_gtt = float(baseline_gtt["peak_gib"])
    candidate_peak_gtt = float(candidate_gtt["peak_gib"])
    if not all(math.isfinite(value) and value >= 0.0 for value in (baseline_peak_gtt, candidate_peak_gtt)):
        raise ScreenError("whole-GTT sampler returned invalid peaks")
    baseline_scratch = _scratch_census(baseline)
    candidate_scratch = _scratch_census(candidate)
    return {
        "prompt_length": int(prompt_length),
        "baseline": {
            "prefill_tok_s": baseline_prefill,
            "decode_tok_s": baseline_decode,
            "tracked_peak_gib": baseline_tracked,
            "owned_session_peak_gib": baseline_owned,
            "whole_gtt_peak_gib": baseline_peak_gtt,
            "scratch": baseline_scratch,
            "reclamation": _tracked_reclamation(baseline),
        },
        "candidate": {
            "prefill_tok_s": candidate_prefill,
            "decode_tok_s": candidate_decode,
            "tracked_peak_gib": candidate_tracked,
            "owned_session_peak_gib": candidate_owned,
            "whole_gtt_peak_gib": candidate_peak_gtt,
            "scratch": candidate_scratch,
            "reclamation": _tracked_reclamation(candidate),
        },
        "comparison": {
            "prefill_speedup": baseline_prefill and candidate_prefill / baseline_prefill,
            "prefill_loss_pct": 100.0 * (baseline_prefill - candidate_prefill) / baseline_prefill,
            "decode_speedup": baseline_decode and candidate_decode / baseline_decode,
            "decode_loss_pct": 100.0 * (baseline_decode - candidate_decode) / baseline_decode,
            "tracked_savings_gib": baseline_tracked - candidate_tracked,
            "owned_session_savings_gib": baseline_owned - candidate_owned,
            "whole_gtt_savings_gib": baseline_peak_gtt - candidate_peak_gtt,
            "bulk_scratch_savings_gib": (
                int(baseline_scratch["physical_owner_bytes"])
                - int(candidate_scratch["physical_owner_bytes"])
            )
            / _GIB,
        },
        "controls_exact": True,
    }


def classify_screen(
    rows: Sequence[Mapping[str, Any]],
    *,
    state_comparison: Mapping[str, Any],
    provenance: Mapping[str, Any],
    long_context_min: int,
    min_tracked_savings_gib: float,
    max_prefill_loss_pct: float,
    max_decode_loss_pct: float,
) -> dict[str, Any]:
    if not rows:
        raise ScreenError("classification requires context rows")
    long_rows = [row for row in rows if int(row["prompt_length"]) >= int(long_context_min)]
    if not long_rows:
        raise ScreenError("classification requires at least one 4K+ row")
    clean = not bool(provenance.get("dirty"))
    controls_exact = all(row.get("controls_exact") is True for row in rows)
    state_exact = state_comparison.get("passed") is True
    lifecycle_exact = all(
        row[mode]["reclamation"]["within_1_mib"] is True
        for row in rows
        for mode in ("baseline", "candidate")
    )
    prefill_ok = all(
        float(row["comparison"]["prefill_loss_pct"]) <= float(max_prefill_loss_pct)
        for row in rows
    )
    decode_ok = all(
        float(row["comparison"]["decode_loss_pct"]) <= float(max_decode_loss_pct)
        for row in rows
    )
    memory_ok = all(
        float(row["comparison"]["tracked_savings_gib"])
        >= float(min_tracked_savings_gib)
        for row in long_rows
    )
    measurement_valid = bool(clean and controls_exact and state_exact and lifecycle_exact)
    if not clean:
        status = "invalid_measurement"
        conclusion = "The SH-M1 process matrix has dirty tracked-source provenance."
    elif not controls_exact:
        status = "invalid_controls"
        conclusion = "A benchmark leg did not preserve the explicit five-surface chunk policy."
    elif not state_exact:
        status = "reject_correctness"
        conclusion = "The 1,024-row candidate changes exact prefill/decode logits or resident state."
    elif not lifecycle_exact:
        status = "reject_lifecycle"
        conclusion = "A benchmark leg did not reclaim tracked allocations close to its process baseline."
    elif not prefill_ok:
        status = "reject_prefill_regression"
        conclusion = "The 1,024-row candidate loses more than the admitted prefill threshold."
    elif not decode_ok:
        status = "reject_decode_regression"
        conclusion = "The 1,024-row candidate loses more than the admitted decode-neutrality threshold."
    elif not memory_ok:
        status = "reject_memory_savings"
        conclusion = "The 1,024-row candidate does not save at least 1 GiB tracked peak at every 4K+ context."
    else:
        status = "promote_q1024"
        conclusion = (
            "The 1,024-row candidate is byte-exact, decode/prefill neutral, lifecycle-clean, "
            "and saves the required tracked peak at every 4K+ context."
        )
    promotion_passed = status == "promote_q1024"
    return {
        "status": status,
        "selected_default": "q1024" if promotion_passed else "q4096",
        "conclusion": conclusion,
        "measurement_valid": measurement_valid,
        "promotion_passed": promotion_passed,
        "clean_provenance": clean,
        "controls_exact": controls_exact,
        "state_exact": state_exact,
        "lifecycle_exact": lifecycle_exact,
        "prefill_within_threshold": prefill_ok,
        "decode_within_threshold": decode_ok,
        "tracked_memory_gate_passed": memory_ok,
        "thresholds": {
            "long_context_min": int(long_context_min),
            "min_tracked_savings_gib": float(min_tracked_savings_gib),
            "max_prefill_loss_pct": float(max_prefill_loss_pct),
            "max_decode_loss_pct": float(max_decode_loss_pct),
            "tracked_reclamation_tolerance_mib": 1.0,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    print(f"[sh-m1] {' '.join(command)}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


def _run_with_gtt_sampler(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    card_name: str | None,
    interval_ms: float,
) -> dict[str, Any]:
    card = select_card(card_name=card_name) if card_name else select_card()
    sampler = VramSampler(
        card,
        interval_ms=float(interval_ms),
        memory_domain="gtt",
        keep_samples=False,
    )
    sampler.start()
    try:
        _run_logged(
            command,
            env=env,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    finally:
        sampler.stop()
    return sampler.result().to_dict(include_samples=False)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreenError(f"could not read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ScreenError(f"{label} must be a JSON object")
    return payload


def _state_command(
    *,
    python: str,
    model: Path,
    contexts: Sequence[int],
    query_rows: int,
    decode_steps: int,
    max_sequence_length: int,
    compiler_version_file: Path,
    output: Path,
) -> list[str]:
    return [
        str(python),
        str(Path(__file__).resolve()),
        "state-child",
        "--model",
        str(model),
        "--contexts",
        *(str(int(value)) for value in contexts),
        "--query-rows",
        str(int(query_rows)),
        "--decode-steps",
        str(int(decode_steps)),
        "--max-sequence-length",
        str(int(max_sequence_length)),
        "--backend",
        "hip_gfx1151",
        "--compiler-version-file",
        str(compiler_version_file),
        "--require-cached-build",
        "--json",
        str(output),
    ]


def _state_context(
    session: Any,
    *,
    prompt_length: int,
    prompt_token_id: int,
    decode_steps: int,
) -> dict[str, Any]:
    from scripts.gguf_eager_teacher_forced_oracle import (
        _capture_checkpoint,
        _fingerprint_array,
    )

    if session.runner is None or session.runner.weights is None:
        raise ScreenError("state child session is closed")
    layer_ids = tuple(range(len(session.runner.weights.config.layer_types)))
    prompt = [int(prompt_token_id)] * int(prompt_length)
    session.reset()
    first = session.prefill(
        prompt,
        use_bulk=True,
        bulk_attention_mode="bulk",
        return_logits=True,
        capture_hidden_seed_fp32=True,
        capture_layer_output_hidden=layer_ids,
    )
    prefill_logits = _fingerprint_array(
        np.ascontiguousarray(first.logits, dtype=np.float32)
    )
    prefill_state = _capture_checkpoint(
        session,
        current_token_id=int(prompt[-1]),
        predicted_token_id=int(first.token_id),
    )
    trajectory: list[dict[str, Any]] = []
    final = first
    for transition in range(int(decode_steps)):
        final = session.step(
            int(prompt_token_id),
            return_logits=True,
            capture_hidden_seed_fp32=True,
            capture_layer_output_hidden=layer_ids,
        )
        trajectory.append(
            {
                "transition": int(transition),
                "input_token_id": int(prompt_token_id),
                "predicted_token_id": int(final.token_id),
                "logits": _fingerprint_array(
                    np.ascontiguousarray(final.logits, dtype=np.float32)
                ),
            }
        )
    final_state = (
        prefill_state
        if not trajectory
        else _capture_checkpoint(
            session,
            current_token_id=int(prompt_token_id),
            predicted_token_id=int(final.token_id),
        )
    )
    fingerprints = [
        prefill_logits,
        *(row["logits"] for row in trajectory),
    ]
    return {
        "prompt_length": int(prompt_length),
        "prompt_token_id": int(prompt_token_id),
        "prefill_logits": prefill_logits,
        "prefill_state": prefill_state,
        "trajectory": trajectory,
        "final_state": final_state,
        "finite": bool(
            all(bool(row.get("finite")) for row in fingerprints)
            and prefill_state.get("finite") is True
            and final_state.get("finite") is True
        ),
    }


def run_state_child(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    contexts = tuple(int(value) for value in args.contexts)
    if not contexts or len(set(contexts)) != len(contexts) or any(value <= 0 for value in contexts):
        raise ScreenError("state-child contexts must be unique positive integers")
    query_rows = int(args.query_rows)
    chunks = chunk_sizes(query_rows)
    if int(args.decode_steps) < 0:
        raise ScreenError("state-child decode steps must be non-negative")
    if int(args.max_sequence_length) < max(contexts) + int(args.decode_steps):
        raise ScreenError("state-child max sequence length does not cover its trajectory")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise ScreenError(f"model does not exist: {model}")
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8")

    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    config = PrefillConfig(
        linear_chunk_size=chunks["linear"],
        moe_chunk_size=chunks["moe"],
        full_attn_query_chunk_size=chunks["full_attn_query"],
        full_attn_post_chunk_size=chunks["full_attn_post"],
        full_attn_rope_chunk_size=chunks["full_attn_rope"],
        auto_tune_chunk_sizes=True,
    )
    with Qwen35GGUFResidentSession(
        model,
        backend=str(args.backend),
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=int(args.max_sequence_length),
        use_wmma_prefill=True,
        use_gemv_decode=True,
        prefill_config=config,
    ) as session:
        if session.runner is None or session._bulk_prefill_scratch is None:
            raise ScreenError("state-child session closed during setup")
        session.select_prefill_quant("gguf_q4_k_m")
        resolved_backend = str(session.backend)
        target_arch = str(session.runner.target_arch)
        resolved_chunks = {
            name: int(getattr(session.prefill_config, {
                "linear": "linear_chunk_size",
                "moe": "moe_chunk_size",
                "full_attn_query": "full_attn_query_chunk_size",
                "full_attn_post": "full_attn_post_chunk_size",
                "full_attn_rope": "full_attn_rope_chunk_size",
            }[name]))
            for name in chunks
        }
        scratch_rows = int(session._bulk_prefill_scratch.rows)
        scratch_capacity = int(session._bulk_prefill_scratch.max_positions)
        rows = [
            _state_context(
                session,
                prompt_length=context,
                prompt_token_id=int(args.prompt_token_id),
                decode_steps=int(args.decode_steps),
            )
            for context in contexts
        ]

    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(args.backend),
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_BACKEND": os.environ.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": os.environ.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_DECODE_REPACK": os.environ.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "GPU_MAX_HW_QUEUES": os.environ.get("GPU_MAX_HW_QUEUES"),
        },
        build_profile="gguf_sh_m1_state_child",
        timing_protocol="correctness-only byte fingerprints; no performance claim",
        warmups=0,
        repetitions=1,
        profiler={"enabled": False, "reason": "correctness-only state child"},
    )
    return {
        "kind": STATE_KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "correctness_claim": True,
        "mode": mode_for_query_rows(query_rows),
        "query_rows": query_rows,
        "resolved_backend": resolved_backend,
        "target_arch": target_arch,
        "prefill_chunk_sizes": resolved_chunks,
        "bulk_prefill_scratch_rows": scratch_rows,
        "bulk_prefill_scratch_capacity": scratch_capacity,
        "contexts": rows,
        "provenance": provenance,
    }


def run_screen(args: argparse.Namespace, *, command: Sequence[str]) -> dict[str, Any]:
    workloads = tuple(int(value) for value in args.workloads)
    if not workloads or len(set(workloads)) != len(workloads):
        raise ScreenError("workloads must be unique and non-empty")
    for name in ("decode_tokens", "measured_runs"):
        if int(getattr(args, name)) <= 0:
            raise ScreenError(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmup_decode_tokens", "warmup_runs", "state_decode_steps"):
        if int(getattr(args, name)) < 0:
            raise ScreenError(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.gtt_interval_ms) <= 0.0:
        raise ScreenError("--gtt-interval-ms must be positive")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise ScreenError(f"model does not exist: {model}")
    if args.raw_root.exists():
        shutil.rmtree(args.raw_root)
    args.raw_root.mkdir(parents=True)
    if not args.compiler_version_file.is_file():
        compiler = subprocess.run(
            ["hipcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        args.compiler_version_file.write_text(compiler.stdout, encoding="utf-8")

    env = os.environ.copy()
    env["HIPENGINE_BACKEND"] = "hip_gfx1151"
    env["HIPENGINE_HIP_ARCH"] = "gfx1151"
    env["HIPENGINE_GGUF_DECODE_REPACK"] = "1"
    env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    env["GPU_MAX_HW_QUEUES"] = "1"

    legs: dict[int, dict[str, dict[str, Any]]] = {}
    for index, context in enumerate(workloads):
        context_root = args.raw_root / str(context)
        context_root.mkdir()
        order = (
            (BASELINE_QUERY_ROWS, CANDIDATE_QUERY_ROWS)
            if index % 2 == 0
            else (CANDIDATE_QUERY_ROWS, BASELINE_QUERY_ROWS)
        )
        legs[context] = {}
        for query_rows in order:
            mode = mode_for_query_rows(query_rows)
            output = context_root / f"{mode}.json"
            child_command = build_benchmark_command(
                python=sys.executable,
                model=model,
                prompt_length=context,
                query_rows=query_rows,
                decode_tokens=int(args.decode_tokens),
                warmup_decode_tokens=int(args.warmup_decode_tokens),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                compiler_version_file=args.compiler_version_file,
                output=output,
            )
            gtt = _run_with_gtt_sampler(
                child_command,
                env=env,
                stdout_path=context_root / f"{mode}.stdout.log",
                stderr_path=context_root / f"{mode}.stderr.log",
                card_name=args.card_name,
                interval_ms=float(args.gtt_interval_ms),
            )
            payload = _load_json(output, label=f"{context}/{mode} benchmark")
            validate_benchmark_leg(
                payload,
                query_rows=query_rows,
                prompt_length=context,
                decode_tokens=int(args.decode_tokens),
                warmup_decode_tokens=int(args.warmup_decode_tokens),
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                expected_token_id=int(args.expected_token_id),
            )
            legs[context][mode] = {
                "payload": payload,
                "gtt": gtt,
                "command": child_command,
                "json": str(output),
                "json_sha256": _sha256(output),
            }

    state_payloads: dict[str, dict[str, Any]] = {}
    state_capacity = max(workloads) + int(args.decode_tokens) + int(args.warmup_decode_tokens) + 1
    state_root = args.raw_root / "state"
    state_root.mkdir()
    for query_rows in (BASELINE_QUERY_ROWS, CANDIDATE_QUERY_ROWS):
        mode = mode_for_query_rows(query_rows)
        output = state_root / f"{mode}.json"
        state_command = _state_command(
            python=sys.executable,
            model=model,
            contexts=workloads,
            query_rows=query_rows,
            decode_steps=int(args.state_decode_steps),
            max_sequence_length=state_capacity,
            compiler_version_file=args.compiler_version_file,
            output=output,
        )
        _run_logged(
            state_command,
            env=env,
            stdout_path=state_root / f"{mode}.stdout.log",
            stderr_path=state_root / f"{mode}.stderr.log",
        )
        state_payloads[mode] = _load_json(output, label=f"{mode} state child")

    state_comparison = compare_state_children(
        state_payloads["q4096"],
        state_payloads["q1024"],
        expected_contexts=workloads,
    )
    rows = [
        summarize_context(
            prompt_length=context,
            baseline=legs[context]["q4096"]["payload"],
            candidate=legs[context]["q1024"]["payload"],
            baseline_gtt=legs[context]["q4096"]["gtt"],
            candidate_gtt=legs[context]["q1024"]["gtt"],
        )
        for context in workloads
    ]
    resolved_backend = str(legs[workloads[0]]["q4096"]["payload"]["resolved_backend"])
    target_arch = str(legs[workloads[0]]["q4096"]["payload"]["target_arch"])
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend="hip_gfx1151",
        resolved_backend=resolved_backend,
        target_arch=target_arch,
        model_path=model,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        command=command,
        environment={
            "HIPENGINE_BACKEND": env.get("HIPENGINE_BACKEND"),
            "HIPENGINE_HIP_ARCH": env.get("HIPENGINE_HIP_ARCH"),
            "HIPENGINE_GGUF_DECODE_REPACK": env.get("HIPENGINE_GGUF_DECODE_REPACK"),
            "GPU_MAX_HW_QUEUES": env.get("GPU_MAX_HW_QUEUES"),
        },
        build_profile="gguf_sh_m1_query_rows_screen",
        timing_protocol="independent right-sized process per context/mode; one warmup plus three measurements",
        warmups=int(args.warmup_runs),
        repetitions=int(args.measured_runs),
        profiler={"enabled": False, "reason": "host wall plus whole-GTT memory screen"},
    )
    child_dirty = any(
        bool(legs[context][mode]["payload"].get("provenance", {}).get("dirty"))
        for context in workloads
        for mode in MODES
    ) or any(
        bool(state_payloads[mode].get("provenance", {}).get("dirty"))
        for mode in MODES
    )
    decision_provenance = dict(provenance)
    decision_provenance["dirty"] = bool(provenance.get("dirty") or child_dirty)
    decision = classify_screen(
        rows,
        state_comparison=state_comparison,
        provenance=decision_provenance,
        long_context_min=int(args.long_context_min),
        min_tracked_savings_gib=float(args.min_tracked_savings_gib),
        max_prefill_loss_pct=float(args.max_prefill_loss_pct),
        max_decode_loss_pct=float(args.max_decode_loss_pct),
    )
    raw_legs = {
        str(context): {
            mode: {
                "command": legs[context][mode]["command"],
                "child_json": legs[context][mode]["json"],
                "child_json_sha256": legs[context][mode]["json_sha256"],
                "whole_gtt_10ms": legs[context][mode]["gtt"],
            }
            for mode in MODES
        }
        for context in workloads
    }
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": decision["status"],
        "performance_claim": bool(decision["promotion_passed"]),
        "correctness_claim": bool(state_comparison["passed"]),
        "workload": {
            "model": str(model),
            "quant": "gguf_q4_k_m",
            "kv_dtype": "bf16",
            "backend": "hip_gfx1151",
            "prompt_source": "repeated_token_id",
            "prompt_token_id": int(args.expected_token_id),
            "prompt_lengths": list(workloads),
            "decode_tokens": int(args.decode_tokens),
            "warmup_decode_tokens": int(args.warmup_decode_tokens),
            "modes": {
                "q4096": chunk_sizes(BASELINE_QUERY_ROWS),
                "q1024": chunk_sizes(CANDIDATE_QUERY_ROWS),
            },
        },
        "protocol": {
            "benchmark": "independent persistent session per context/mode; one discarded run plus three measured runs",
            "whole_gtt": f"whole-device amdgpu GTT sampled every {float(args.gtt_interval_ms):g} ms",
            "correctness": state_comparison["protocol"],
            "promotion_rule": (
                "clean provenance, exact state/logits/trajectory, <=1% prefill and decode loss, "
                ">=1.0 GiB tracked-peak saving at every 4K+ context, and <=1 MiB tracked close delta"
            ),
        },
        "rows": rows,
        "state_comparison": state_comparison,
        "decision": decision,
        "provenance": decision_provenance,
        "raw": {
            "root": str(args.raw_root),
            "legs": raw_legs,
            "state_children": {
                mode: {
                    "json": str(state_root / f"{mode}.json"),
                    "json_sha256": _sha256(state_root / f"{mode}.json"),
                }
                for mode in MODES
            },
        },
        "notes": [
            "All five chunk fields are explicit in both legs; only full_attn_query differs.",
            "State-child timings are excluded and make no performance claim.",
            "Whole-GTT is descriptive; the binding memory gate uses tracked allocator peak.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    state = subparsers.add_parser("state-child", help="emit one correctness-only state fingerprint child")
    state.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    state.add_argument("--contexts", nargs="+", type=_parse_length, required=True)
    state.add_argument("--query-rows", type=int, choices=(BASELINE_QUERY_ROWS, CANDIDATE_QUERY_ROWS), required=True)
    state.add_argument("--prompt-token-id", type=int, default=9707)
    state.add_argument("--decode-steps", type=int, default=4)
    state.add_argument("--max-sequence-length", type=int, required=True)
    state.add_argument("--backend", choices=("hip_gfx1151",), default="hip_gfx1151")
    state.add_argument("--compiler-version-file", type=Path, required=True)
    state.add_argument("--require-cached-build", action="store_true")
    state.add_argument("--json", type=Path, required=True)

    screen = subparsers.add_parser("screen", help="run the complete SH-M1 process matrix")
    screen.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    screen.add_argument("--workloads", nargs="+", type=_parse_length, default=list(DEFAULT_WORKLOADS))
    screen.add_argument("--expected-token-id", type=int, default=9707)
    screen.add_argument("--decode-tokens", type=int, default=128)
    screen.add_argument("--warmup-decode-tokens", type=int, default=1)
    screen.add_argument("--warmup-runs", type=int, default=1)
    screen.add_argument("--measured-runs", type=int, default=3)
    screen.add_argument("--state-decode-steps", type=int, default=4)
    screen.add_argument("--gtt-interval-ms", type=float, default=10.0)
    screen.add_argument("--card-name")
    screen.add_argument("--compiler-version-file", type=Path, default=Path("/tmp/hipengine-hipcc-version.txt"))
    screen.add_argument("--raw-root", type=Path, default=Path("/tmp/hipengine-sh-m1"))
    screen.add_argument("--long-context-min", type=int, default=4096)
    screen.add_argument("--min-tracked-savings-gib", type=float, default=1.0)
    screen.add_argument("--max-prefill-loss-pct", type=float, default=1.0)
    screen.add_argument("--max-decode-loss-pct", type=float, default=1.0)
    screen.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    command = [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *raw_argv]
    try:
        if args.command == "state-child":
            artifact = run_state_child(args, command=command)
            output = args.json
            decision = None
        else:
            artifact = run_screen(args, command=command)
            output = args.out
            decision = artifact["decision"]
    except (ScreenError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if decision is not None:
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if decision["measurement_valid"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
