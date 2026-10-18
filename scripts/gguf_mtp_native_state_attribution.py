#!/usr/bin/env python3
"""Attribute one GGUF MTP divergence on the singleton native target cycle.

Two divergences are on record for the native GGUF target verifier: a
Qwen3.6-35B-A3B MoE gfx1151 request first differed from its autoregressive (AR)
reference at generated token 35, and a Qwen3.8 dense Q4_K_S gfx1151 request
first differed at token 118. Both requests were served by the public server with
``--max-active-requests 4``. That capacity does not decide the lane for a lone
request: a single speculative request keeps its per-request
``Qwen35GGUFTransactionalVerifier``, which launches
``Qwen35GGUFResidentSession.verify_target_block_native_cycle`` with
``device_accept_commit=True``. The N2 native target graph then decides
acceptance, copies the selected Conv/GDN state row into the resident
destination, and advances the cursor on device. ``verify_target_blocks_batch``
plus ``_commit_deferred_packed_verify_states_batch`` run only for a packed
request group, and the one-request packed target exists only under an explicit
``packed_c1_target`` evidence row.

This diagnostic reproduces the singleton native cycle. It records the lane it
resolved and refuses a lane it does not reproduce, so a packed capacity>1
reading can never be reported as the lone-request lane.

Four contracts are reported separately. A single generated-token comparison
cannot express any of them.

``strict_state_copy``
    The committed destination bytes equal the captured verifier row the device
    accept selected. This byte comparison is exact; it never compares native
    state against AR state.

``ownership``
    The selected row index, committed token, commit position, resulting cursor,
    and visible tokens are the ones the accept decided:
    ``commit_row == accepted_draft_tokens``,
    ``commit_token == input_token_ids[commit_row]``,
    ``commit_position == start_position + commit_row``,
    ``end_position == commit_position + 1``, the session cursor equals
    ``end_position``, and the visible tokens are the accepted draft prefix plus
    the selected row's top-1 correction. A failure here is a proven commit
    defect: a wrong row, a wrong offset, or a wrong destination.

``repeatability``
    Repeated native runs at the same prefix produce identical row top-1, logit
    bytes, commit metadata, and committed state. It is checked only when
    ``--repeat`` is greater than one.

``numerical_drift``
    How far the native rows are from the AR reference at the same prefix:
    per-row top-1, KL, and logit deltas, plus committed Conv/GDN state deltas
    against AR decoding the committed tokens. This block is a measurement with
    ``gating: false``. A top-1 or generated-ID difference alone is a diagnostic
    signal, never a correctness failure: the native row kernels may carry
    different numerics than the token-serial path, and the MTP contract is exact
    ownership and repeatability, not bit-identical AR arithmetic.

Diagnostic only: it never runs a server, never selects a production route, and
makes no performance claim. It needs a GPU plus the recorded HTTP matrix row
that carries the prompt and the AR reference ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gguf_mtp_bench import GGUF_PATH, build_chat_prompt  # noqa: E402
from scripts.gguf_mtp_forced_target_probe import (  # noqa: E402
    ProbeError,
    _bytes_digest,
    _compare_lifecycle_fingerprints,
    _copy_device_bytes,
    _hip_available,
    _llama_accept_count,
    _numeric_summary_from_raw,
    _prefill_probe_session,
    _read_json,
    _repo_provenance,
    _session_lifecycle_fingerprint,
    _top_k_rows,
    diagnostic_env_payload,
    exact_command_payload,
)
from scripts.quant_quality.metrics import compare_logits  # noqa: E402


SCHEMA = "gguf_mtp_native_state_attribution.v3"

LANE_AUTO = "auto"
LANE_SINGLETON_REQUEST = "singleton"
LANE_PACKED_REQUEST = "packed"
LANE_SINGLETON = "singleton_native_cycle"
LANE_PACKED = "packed_capacity_gt_1"

VERDICT_CLEAN = "no_divergence_at_this_cycle"
VERDICT_OWNERSHIP_DEFECT = "commit_ownership_defect"
VERDICT_REPEATABILITY = "repeatability_mismatch"
VERDICT_NUMERICAL_DRIFT = "numerical_drift_measured"
VERDICT_OWNERSHIP_UNVERIFIED = "inconclusive_ownership_unverified"
VERDICT_SERIAL_CONTROL = "inconclusive_serial_control_mismatch"

VERDICT_CLASS_CLEAN = "clean"
VERDICT_CLASS_OWNERSHIP = "ownership_defect"
VERDICT_CLASS_REPEATABILITY = "repeatability_defect"
VERDICT_CLASS_NUMERICAL = "numerical_diagnostic"
VERDICT_CLASS_INCONCLUSIVE = "inconclusive"

# The native graph refuses these two diagnostics together; reporting the
# conflict up front is clearer than a capture-time capability miss.
NATIVE_REFUSED_ENV_PAIR = (
    "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL",
    "HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING",
)

# The direct Q6 top-1 verifier path skips the full-logit readback this
# diagnostic needs; a recorded run that enabled it cannot supply row logits.
NATIVE_REFUSED_LOGIT_ENV = "HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A"

_NATIVE_LAUNCH = (
    "Qwen35GGUFResidentSession.verify_target_block_native_cycle"
)


# ---------------------------------------------------------------------------
# Lane resolution
# ---------------------------------------------------------------------------


def resolve_lane(
    *,
    requested: str = LANE_AUTO,
    request_count: int = 1,
    packed_c1_target: bool = False,
    max_active_requests: int = 1,
) -> dict[str, Any]:
    """Resolve which production target-verifier lane a request group runs.

    ``auto`` follows the request group. Two or more speculative requests in one
    group, or the explicit one-request ``packed_c1_target`` route, are packed
    work and resolve to the packed lane; only a lone request with no packed C1
    target resolves to the singleton native cycle. ``max_active_requests`` is
    recorded, never used to pick the lane: serving capacity does not move a lone
    request onto the packed batch lane. An explicit singleton claim on packed
    work raises instead of relabelling it.
    """

    count = int(request_count)
    capacity = int(max_active_requests)
    if count <= 0:
        raise ProbeError("lane resolution requires at least one speculative request")
    if capacity <= 0:
        raise ProbeError("max_active_requests must be positive")
    requested_lane = str(requested)
    if requested_lane not in {
        LANE_AUTO,
        LANE_SINGLETON_REQUEST,
        LANE_PACKED_REQUEST,
    }:
        raise ProbeError(f"unknown lane request {requested_lane!r}")
    packed_group = count >= 2
    packed_c1 = bool(packed_c1_target)
    if requested_lane == LANE_SINGLETON_REQUEST:
        if packed_group:
            raise ProbeError(
                f"a request group of {count} speculative requests runs the "
                "packed target lane, so an explicit singleton claim would "
                "relabel packed work as the per-request native cycle. Request "
                "the packed lane and report it from the packed harness"
            )
        if packed_c1:
            raise ProbeError(
                "packed_c1_target selects the one-request packed target, not "
                "the singleton native cycle"
            )
    if requested_lane == LANE_PACKED_REQUEST and not packed_group and not packed_c1:
        raise ProbeError(
            "the packed batch lane needs a packed request group; a lone "
            "speculative request runs the per-request "
            "Qwen35GGUFTransactionalVerifier. The one-request packed target "
            "also needs the backend's GGUF_SPECDEC2_MTP2_PHYSICAL_C1 "
            "capability (gfx1100 declares it, gfx1151 does not) plus an "
            "explicit packed_c1_target evidence row"
        )
    packed = bool(packed_group or packed_c1)
    lane = LANE_PACKED if packed else LANE_SINGLETON
    if packed_group and packed_c1:
        reason = (
            f"a request group of {count} speculative requests on the explicit "
            "packed_c1_target route runs the packed target lane"
        )
    elif packed_group:
        reason = (
            f"a request group of {count} speculative requests runs the packed "
            "target lane, never the per-request singleton verifier"
        )
    elif packed_c1:
        reason = (
            "the explicit packed_c1_target evidence row selects the "
            "one-request packed target"
        )
    else:
        reason = (
            "a lone speculative request runs the per-request "
            "Qwen35GGUFTransactionalVerifier, which launches "
            "verify_target_block_native_cycle with device_accept_commit=True. "
            f"--max-active-requests {capacity} does not put a lone request on "
            "the packed batch lane, and the one-request packed target needs the "
            "backend's GGUF_SPECDEC2_MTP2_PHYSICAL_C1 capability plus an "
            "explicit packed_c1_target evidence row"
        )
    return {
        "lane": lane,
        "requested": requested_lane,
        "request_count": count,
        "max_active_requests": capacity,
        "packed_c1_target": bool(packed_c1_target),
        "reproduced_by_this_diagnostic": lane == LANE_SINGLETON,
        "launch": _NATIVE_LAUNCH if lane == LANE_SINGLETON else "verify_target_blocks_batch",
        "reason": reason,
    }


def require_reproducible_lane(lane: Mapping[str, Any]) -> None:
    """Raise unless the resolved lane is the one this diagnostic reproduces."""

    if str(lane.get("lane")) == LANE_SINGLETON:
        return
    raise ProbeError(
        "this diagnostic reproduces the singleton native cycle only. The packed "
        "lane is verify_target_blocks_batch over a TargetVerifyBatch plus a "
        "device accept payload and _commit_deferred_packed_verify_states_batch"
        "[_device]; reproducing it needs the packed request group and its "
        "device accept buffers, which this script does not build. Report the "
        "packed lane from that harness instead of relabelling a singleton run."
    )


# ---------------------------------------------------------------------------
# Logit comparison (shared quant-quality metrics)
# ---------------------------------------------------------------------------


def _float32_rows(values: Any, *, label: str) -> np.ndarray:
    rows = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if rows.ndim != 2:
        raise ProbeError(f"{label} logits must be rank-2, got shape {rows.shape}")
    return rows


def logits_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    rows: int | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare two ``(rows, vocab)`` logit matrices row by row.

    ``rows`` truncates the comparison, which is what makes an accepted-prefix
    restriction explicit rather than implicit. Distribution metrics come from
    ``scripts/quant_quality/metrics.compare_logits`` with the reference top-1 as
    the teacher label; the result is a measurement and carries ``gating: false``.
    """

    ref = _float32_rows(reference, label="reference")
    cand = _float32_rows(candidate, label="candidate")
    if ref.shape != cand.shape:
        raise ProbeError(
            f"logit matrices disagree in shape: {ref.shape} vs {cand.shape}"
        )
    compared = int(ref.shape[0]) if rows is None else int(rows)
    if compared <= 0:
        raise ProbeError("logits_agreement requires at least one compared row")
    if compared > int(ref.shape[0]):
        raise ProbeError(
            f"compared rows {compared} exceed available rows {ref.shape[0]}"
        )
    ref = np.ascontiguousarray(ref[:compared])
    cand = np.ascontiguousarray(cand[:compared])
    vocab = int(ref.shape[1])
    ref_top1 = np.argmax(ref, axis=1).astype(np.int64)
    cand_top1 = np.argmax(cand, axis=1).astype(np.int64)
    mismatch_rows = [
        int(index)
        for index in range(compared)
        if int(ref_top1[index]) != int(cand_top1[index])
    ]
    shared = compare_logits(
        ref,
        cand,
        ref_top1,
        top_k=max(1, min(int(top_k), vocab)),
    )
    return {
        "compared_rows": compared,
        "vocab_size": vocab,
        "metric_source": "scripts/quant_quality/metrics.compare_logits",
        "labels_source": "reference_top1",
        "reference_top1": [int(value) for value in ref_top1],
        "candidate_top1": [int(value) for value in cand_top1],
        "top1_mismatch_rows": mismatch_rows,
        "top1_agreement": float(compared - len(mismatch_rows)) / float(compared),
        "max_abs_logit_delta": float(shared["max_abs_logit_delta"]),
        "mean_kl_nats": float(shared["mean_kl_nats"]),
        "gating": False,
        "shared_metrics": shared,
    }


def _top1_only_agreement(
    reference: Sequence[int],
    candidate: Sequence[int],
    *,
    rows: int | None,
) -> dict[str, Any]:
    ref = [int(value) for value in reference]
    cand = [int(value) for value in candidate]
    if len(ref) != len(cand):
        raise ProbeError(
            f"top-1 rows disagree in length: {len(ref)} vs {len(cand)}"
        )
    compared = len(ref) if rows is None else int(rows)
    if compared <= 0 or compared > len(ref):
        raise ProbeError(f"invalid compared row count {compared} for {len(ref)} rows")
    mismatch_rows = [
        index for index in range(compared) if ref[index] != cand[index]
    ]
    return {
        "compared_rows": compared,
        "reference_top1": ref[:compared],
        "candidate_top1": cand[:compared],
        "top1_mismatch_rows": mismatch_rows,
        "top1_agreement": float(compared - len(mismatch_rows)) / float(compared),
        "max_abs_logit_delta": None,
        "mean_kl_nats": None,
        "logits_available": False,
        "gating": False,
    }


# ---------------------------------------------------------------------------
# Commit provenance
# ---------------------------------------------------------------------------


def _copy_byte_mismatch(
    component: str,
    source_raw: Any,
    destination_raw: Any,
    *,
    layer: int | None = None,
    other: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a strict-copy mismatch for one committed buffer, or None."""

    source = _as_u8(source_raw)
    destination = _as_u8(destination_raw)
    base: dict[str, Any] = {
        "contract": "strict_state_copy",
        "component": str(component),
    }
    if layer is not None:
        base["layer"] = int(layer)
    if source.size != destination.size:
        return {
            **base,
            "reason": "size",
            "source_nbytes": int(source.size),
            "destination_nbytes": int(destination.size),
        }
    if np.array_equal(source, destination):
        return None
    differing = np.flatnonzero(source != destination)
    mismatch: dict[str, Any] = {
        **base,
        "reason": "bytes",
        "nbytes": int(source.size),
        "first_diff_byte": int(differing[0]),
        "differing_bytes": int(differing.size),
    }
    if other is not None:
        other_bytes = _as_u8(other["bytes"])
        if other_bytes.size == destination.size and np.array_equal(
            other_bytes, destination
        ):
            mismatch["destination_matches_other_row"] = int(other["index"])
    return mismatch


def commit_ownership_check(record: Mapping[str, Any]) -> dict[str, Any]:
    """Check the selected-state commit against the accept that selected it.

    ``record`` carries the device accept metadata and, per linear layer, the
    captured source row the accept selected plus the destination bytes after the
    commit. The contract is the one the N2 native graph implements:
    ``linear_commit_kernel`` copies source row ``commit_rows[0]`` into the
    resident destination, and the accept kernel sets the cursor to
    ``commit_position + 1`` with ``commit_position == start + commit_row``.
    AR state plays no part in either check.
    """

    accepted = int(record["accepted"])
    rows = int(record["rows"])
    commit_row = int(record["commit_row"])
    commit_token = int(record["commit_token"])
    commit_position = int(record["commit_position"])
    start_position = int(record["start_position"])
    end_position = int(record["end_position"])
    cursor_after = int(record["cursor_after"])
    input_tokens = [int(token) for token in record["input_token_ids"]]
    visible = [int(token) for token in record["visible_tokens"]]
    drafts = [int(token) for token in record["draft_tokens"]]
    target_top1 = [int(token) for token in record["target_top1"]]
    if rows <= 0:
        raise ProbeError("commit ownership requires at least one verifier row")
    if len(input_tokens) != rows:
        raise ProbeError(
            f"commit record carries {len(input_tokens)} input tokens for {rows} rows"
        )
    if accepted < 0 or accepted >= rows:
        raise ProbeError(
            f"accepted count {accepted} is outside the verifier rows {rows}"
        )

    commit_row_in_range = 0 <= commit_row < rows
    ownership_mismatches: list[dict[str, Any]] = []
    if not commit_row_in_range:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "accept.commit_row",
                "expected": f"0 <= commit_row < {rows}",
                "actual": int(commit_row),
                "out_of_range": True,
            }
        )
    elif commit_row != accepted:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "accept.commit_row",
                "expected": int(accepted),
                "actual": int(commit_row),
            }
        )
    if len(target_top1) != rows:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "accept.target_top1_rows",
                "expected": int(rows),
                "actual": len(target_top1),
            }
        )
    if commit_row_in_range:
        expected_token = input_tokens[commit_row]
        if commit_token != expected_token:
            ownership_mismatches.append(
                {
                    "contract": "ownership",
                    "component": "accept.commit_token",
                    "expected": int(expected_token),
                    "actual": int(commit_token),
                }
            )
    expected_position = start_position + commit_row
    if commit_position != expected_position:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "cursor.commit_position",
                "expected": int(expected_position),
                "actual": int(commit_position),
            }
        )
    expected_end = expected_position + 1
    if end_position != expected_end:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "cursor.end_position",
                "expected": int(expected_end),
                "actual": int(end_position),
            }
        )
    if cursor_after != expected_end:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "cursor.session_position",
                "expected": int(expected_end),
                "actual": int(cursor_after),
            }
        )
    if len(drafts) < accepted:
        ownership_mismatches.append(
            {
                "contract": "ownership",
                "component": "accept.draft_tokens",
                "expected": f"at least {int(accepted)}",
                "actual": len(drafts),
            }
        )
    elif commit_row_in_range and commit_row < len(target_top1):
        expected_visible = [*drafts[:accepted], target_top1[commit_row]]
        if visible != expected_visible:
            ownership_mismatches.append(
                {
                    "contract": "ownership",
                    "component": "accept.committed_tokens",
                    "expected": expected_visible,
                    "actual": visible,
                }
            )

    copy_mismatches: list[dict[str, Any]] = []
    layers_checked = 0
    for layer in record.get("layers", ()):
        layers_checked += 1
        for part in ("conv", "recurrent"):
            mismatch = _copy_byte_mismatch(
                f"linear_state_{part}",
                layer[part]["source_row"],
                layer[part]["destination"],
                layer=int(layer["layer"]),
                other=layer[part].get("other_row"),
            )
            if mismatch is not None:
                copy_mismatches.append(mismatch)
    missing_captures: list[dict[str, Any]] = []
    expected_layers = [int(layer) for layer in (record.get("expected_layers") or ())]
    captured_layers = {
        int(layer["layer"]) for layer in (record.get("layers") or ())
    }
    for layer_id in expected_layers:
        if layer_id in captured_layers:
            continue
        missing_captures.append(
            {
                "contract": "strict_state_copy",
                "component": "linear_state_capture",
                "layer": int(layer_id),
                "reason": "not_captured",
            }
        )
    hidden = record.get("hidden_row")
    hidden_checked = hidden is not None
    if not hidden_checked and record.get("hidden_required", False):
        missing_captures.append({"component": "hidden_seed", "reason": "not_captured"})
    if hidden is not None:
        mismatch = _copy_byte_mismatch(
            "hidden_seed",
            hidden["source_row"],
            hidden["destination"],
            other=hidden.get("other_row"),
        )
        if mismatch is not None:
            copy_mismatches.append(mismatch)

    checked = bool(layers_checked) or bool(hidden_checked)
    mismatches = [*ownership_mismatches, *copy_mismatches]
    if mismatches:
        status = "broken"
    elif not checked or missing_captures:
        status = "not_checked"
    else:
        status = "exact"
    return {
        "status": status,
        "selected_row": int(commit_row),
        "start_position": int(start_position),
        "expected_commit_position": int(start_position + commit_row),
        "expected_end_position": int(start_position + commit_row + 1),
        "actual_commit_position": int(commit_position),
        "actual_end_position": int(end_position),
        "actual_cursor_after": int(cursor_after),
        "commit_row_in_range": bool(commit_row_in_range),
        "layers_checked": layers_checked,
        "layers_expected": len(expected_layers),
        "hidden_row_checked": bool(hidden_checked),
        "contracts": {
            "strict_state_copy": {
                "status": (
                    "broken" if copy_mismatches else
                    "not_checked" if not checked or missing_captures else "exact"
                ),
                "mismatches": copy_mismatches,
                "missing_captures": missing_captures,
            },
            "ownership": {
                "status": "broken" if ownership_mismatches else "exact",
                "mismatches": ownership_mismatches,
            },
        },
        "mismatches": mismatches,
        "contract": (
            "the committed destination equals the captured source row "
            "(commit_rows[0]) byte for byte, commit_row is the accepted row, "
            "commit_token is that row's input token, and the cursor is "
            "start_position + commit_row + 1"
        ),
    }


# ---------------------------------------------------------------------------
# Committed-state numerical delta
# ---------------------------------------------------------------------------


def _as_u8(raw: Any) -> np.ndarray:
    """Return raw device bytes as one contiguous uint8 vector."""

    if isinstance(raw, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(raw), dtype=np.uint8).copy()
    return np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)).reshape(-1)


def _fp32_view(raw: Any) -> np.ndarray | None:
    data = _as_u8(raw)
    if data.size == 0 or data.size % np.dtype(np.float32).itemsize != 0:
        return None
    return data.view(np.float32)


def _state_part_delta(native_raw: Any, ar_raw: Any, *, label: str) -> dict[str, Any]:
    native = _as_u8(native_raw)
    ar = _as_u8(ar_raw)
    result: dict[str, Any] = {
        "label": str(label),
        "nbytes": int(native.size),
    }
    if native.size != ar.size:
        result.update(
            {
                "comparable": False,
                "reason": "size_mismatch",
                "ar_control_nbytes": int(ar.size),
            }
        )
        return result
    result["bitwise_equal"] = bool(np.array_equal(native, ar))
    native_values = _fp32_view(native)
    ar_values = _fp32_view(ar)
    if native_values is None or ar_values is None:
        result.update({"comparable": False, "reason": "not_fp32_aligned"})
        return result
    delta = np.abs(
        native_values.astype(np.float64) - ar_values.astype(np.float64)
    )
    result.update(
        {
            "comparable": True,
            "elements": int(delta.size),
            "changed_elements": int(np.count_nonzero(delta)),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "native_summary": _numeric_summary_from_raw(
                native, dtype="fp32", label=f"native_{label}"
            ),
            "ar_control_summary": _numeric_summary_from_raw(
                ar, dtype="fp32", label=f"ar_control_{label}"
            ),
        }
    )
    return result


def state_numeric_delta(
    native_layers: Mapping[int, Mapping[str, Any]],
    ar_control_layers: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure native committed Conv/GDN state against AR at the same cursor.

    Both inputs map ``layer -> {"conv": bytes, "recurrent": bytes}``. The result
    keeps the measured magnitude and both numeric summaries; it is a
    measurement, not a verdict.
    """

    if not native_layers or not ar_control_layers:
        return {
            "checked": False,
            "reason": "no linear-state layers were captured in both arms",
        }
    layers: list[dict[str, Any]] = []
    all_identical = True
    for layer_id in sorted(set(native_layers) & set(ar_control_layers)):
        entry: dict[str, Any] = {"layer": int(layer_id)}
        for part in ("conv", "recurrent"):
            delta = _state_part_delta(
                native_layers[layer_id][part],
                ar_control_layers[layer_id][part],
                label=f"linear_{layer_id}_{part}",
            )
            entry[part] = delta
            if not delta.get("comparable") or not delta.get("bitwise_equal"):
                all_identical = False
        layers.append(entry)
    return {
        "checked": True,
        "compared_layers": len(layers),
        "all_identical": bool(all_identical),
        "layers": layers,
        "note": (
            "elementwise native committed state minus AR state after decoding "
            "the same committed tokens; measured, never gating"
        ),
    }


# ---------------------------------------------------------------------------
# Repeatability
# ---------------------------------------------------------------------------


def repeatability_check(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare repeated native runs of one schedule."""

    ordered = list(runs)
    if len(ordered) < 2:
        return {
            "checked": False,
            "runs": len(ordered),
            "identical": None,
            "differences": [],
            "reason": "the native arm ran once; pass --repeat 2 or more to check determinism",
        }
    fields: dict[str, Any] = {
        "row_top1": lambda arm: [int(token) for token in arm["row_top1"]],
        "logits_digest": lambda arm: str(arm["logits_digest"]),
        "state_digest": lambda arm: str(arm["state_digest"]),
        "visible_tokens": lambda arm: [int(token) for token in arm["visible"]],
        "commit": lambda arm: {
            key: int(arm["commit"][key])
            for key in (
                "accepted",
                "commit_row",
                "commit_token",
                "commit_position",
                "end_position",
                "cursor_after",
            )
        },
    }
    differences: list[dict[str, Any]] = []
    baseline = ordered[0]
    for index, run in enumerate(ordered[1:], start=2):
        for name, getter in fields.items():
            expected = getter(baseline)
            actual = getter(run)
            if expected != actual:
                differences.append(
                    {
                        "field": name,
                        "run": int(index),
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {
        "checked": True,
        "runs": len(ordered),
        "identical": not differences,
        "differences": differences,
        "reason": (
            "repeated native runs at the same prefix must produce identical row "
            "top-1, logit bytes, commit metadata, and committed state"
        ),
    }


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def attribute_divergence(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    lane: Mapping[str, Any] | None = None,
    state_delta: Mapping[str, Any] | None = None,
    repeatability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the verdict, its contract breakdown, and the measured drift.

    ``arms`` carries ``ar``, ``native``, ``serial``, and optionally
    ``ar_control``. Only a broken state copy, broken commit ownership, or a
    repeatability mismatch is decisive. A native-versus-AR numerical difference
    is reported as a measurement.
    """

    missing = [name for name in ("ar", "native", "serial") if name not in arms]
    if missing:
        raise ProbeError(f"attribution requires arms {missing}")
    ar = arms["ar"]
    native = arms["native"]
    serial = arms["serial"]
    ar_control = arms.get("ar_control")

    accepted = int(native["accepted"])
    verifier_rows = len(native["row_top1"])
    if verifier_rows <= 0:
        raise ProbeError("native arm carries no verifier rows")
    consumed_rows = accepted + 1
    if consumed_rows > verifier_rows:
        raise ProbeError(
            f"native arm consumed {consumed_rows} rows but scored only "
            f"{verifier_rows}"
        )

    row_arithmetic = logits_agreement(
        ar["row_logits"], native["row_logits"], rows=consumed_rows
    )
    full_row_arithmetic = logits_agreement(ar["row_logits"], native["row_logits"])
    serial_row_arithmetic = _top1_only_agreement(
        ar["row_top1"], serial["row_top1"], rows=None
    )

    ownership = native.get("ownership") or {
        "status": "not_checked",
        "contracts": {
            "strict_state_copy": {"status": "not_checked", "mismatches": []},
            "ownership": {"status": "not_checked", "mismatches": []},
        },
        "mismatches": [],
        "layers_checked": 0,
    }
    copy_contract = dict(
        (ownership.get("contracts") or {}).get("strict_state_copy")
        or {"status": "not_checked", "mismatches": []}
    )
    index_contract = dict(
        (ownership.get("contracts") or {}).get("ownership")
        or {"status": "not_checked", "mismatches": []}
    )

    drafts = [int(token) for token in native["verifier_drafts"]]
    ar_top1 = [int(token) for token in ar["row_top1"]]
    ar_accepted = _llama_accept_count(drafts, ar_top1[: len(drafts)])
    visible = [int(token) for token in native["visible"]]
    accepted_decision = {
        "native_accepted": int(accepted),
        "ar_accepted": int(ar_accepted),
        "accepted_count_matches_ar": bool(int(accepted) == int(ar_accepted)),
        "native_visible": visible,
        "ar_top1_at_selected_row": int(ar_top1[accepted]),
        "correction_token_matches_ar": bool(
            visible[accepted] == ar_top1[accepted]
        ),
    }

    state_vs_ar_control = (
        None
        if ar_control is None
        else _compare_lifecycle_fingerprints(ar_control["state"], native["state"])
    )
    ar_vs_serial = _compare_lifecycle_fingerprints(ar["state"], serial["state"])
    positions = {
        "native": int(native["state"]["position"]),
        "ar_control": (
            None if ar_control is None else int(ar_control["state"]["position"])
        ),
        "ar_full_block": int(ar["state"]["position"]),
        "serial": int(serial["state"]["position"]),
    }

    state_drift = list(state_vs_ar_control or ())
    numeric_state = dict(state_delta or {"checked": False})
    logit_drift = bool(
        full_row_arithmetic["top1_mismatch_rows"]
        or float(full_row_arithmetic["max_abs_logit_delta"]) > 0.0
    )
    state_differs = bool(state_drift) or bool(
        numeric_state.get("checked") and not numeric_state.get("all_identical", True)
    )
    numerical_differs = bool(logit_drift or state_differs)
    repeat = dict(
        repeatability
        or {
            "checked": False,
            "runs": 1,
            "identical": None,
            "differences": [],
            "reason": "the native arm ran once",
        }
    )

    if ar_vs_serial:
        verdict = VERDICT_SERIAL_CONTROL
        verdict_class = VERDICT_CLASS_INCONCLUSIVE
        decisive = False
        reason = (
            "the serial-exact control does not match AR state at the same "
            "cursor, so the resident-state comparison basis is broken; fix or "
            "re-run the control before reading the native arm"
        )
    elif repeat.get("checked") and repeat.get("identical") is False:
        verdict = VERDICT_REPEATABILITY
        verdict_class = VERDICT_CLASS_REPEATABILITY
        decisive = True
        reason = (
            "repeated native runs at this prefix disagree, so the same-schedule "
            "determinism contract fails and the ownership and drift readings "
            "below are not attributable until the repeats agree"
        )
    elif (
        copy_contract.get("status") == "broken"
        or index_contract.get("status") == "broken"
    ):
        verdict = VERDICT_OWNERSHIP_DEFECT
        verdict_class = VERDICT_CLASS_OWNERSHIP
        decisive = True
        broken = [
            name
            for name, contract in (
                ("strict_state_copy", copy_contract),
                ("ownership", index_contract),
            )
            if contract.get("status") == "broken"
        ]
        reason = (
            "the device commit broke "
            + " and ".join(broken)
            + ": the committed destination, the selected row, the committed "
            "token, the cursor, or the visible tokens disagree with the accept "
            "that selected them. This is proven from the commit provenance and "
            "does not depend on how the native rows score"
        )
    elif (
        copy_contract.get("status") != "exact"
        or index_contract.get("status") != "exact"
    ):
        verdict = VERDICT_OWNERSHIP_UNVERIFIED
        verdict_class = VERDICT_CLASS_INCONCLUSIVE
        decisive = False
        reason = (
            "the commit provenance was not captured, so a wrong row, wrong "
            "offset, or wrong copy is neither proven nor excluded. Equal top-1 "
            "does not establish equal arithmetic state"
        )
    elif numerical_differs:
        verdict = VERDICT_NUMERICAL_DRIFT
        verdict_class = VERDICT_CLASS_NUMERICAL
        decisive = False
        reason = (
            "ownership and the state copy are exact and the repeats agree, but "
            "the native rows and committed state differ numerically from AR at "
            "the same prefix. That is measured drift to evaluate against a "
            "profile envelope, not a correctness failure: a top-1 or "
            "generated-ID difference alone is diagnostic"
        )
    else:
        verdict = VERDICT_CLEAN
        verdict_class = VERDICT_CLASS_CLEAN
        decisive = False
        reason = (
            "ownership and the state copy are exact, the repeats agree, and the "
            "native rows and committed state are bit-identical to AR at this "
            "cycle"
        )

    numerical = {
        "gating": False,
        "decisive": False,
        "top1_divergence_at_consumed_rows": list(
            row_arithmetic["top1_mismatch_rows"]
        ),
        "top1_divergence_at_any_verifier_row": list(
            full_row_arithmetic["top1_mismatch_rows"]
        ),
        "row_arithmetic_accepted_prefix": row_arithmetic,
        "row_arithmetic_full_block": full_row_arithmetic,
        "serial_row_arithmetic": serial_row_arithmetic,
        "state_drift_vs_ar_control": state_drift,
        "state_numeric_delta_vs_ar_control": numeric_state,
        "note": (
            "native-versus-AR measurements at one prefix. A top-1 or "
            "generated-ID difference alone is a diagnostic signal, never a "
            "correctness failure; ownership and repeatability are the binding "
            "contracts here"
        ),
    }

    return {
        "verdict": verdict,
        "verdict_class": verdict_class,
        "verdict_is_decisive": bool(decisive),
        "reason": reason,
        "lane": dict(lane) if lane is not None else None,
        "consumed_rows": consumed_rows,
        "verifier_rows": verifier_rows,
        "contracts": {
            "strict_state_copy": copy_contract,
            "ownership": index_contract,
            "repeatability": repeat,
            "numerical_drift": {
                "status": "measured" if numerical_differs else "identical",
                "gating": False,
                "decisive": False,
                "note": numerical["note"],
            },
        },
        "numerical": numerical,
        "accepted_decision": accepted_decision,
        "accepted_state": {
            "ar_control_vs_native": state_vs_ar_control,
            "ar_vs_serial": ar_vs_serial,
            "positions": positions,
        },
        "commit_ownership": ownership,
    }


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _row_ids(row: Mapping[str, Any]) -> list[int]:
    ids = row.get("ids")
    if isinstance(ids, list) and ids:
        return [int(token) for token in ids]
    response = row.get("response")
    if isinstance(response, dict):
        hipengine = response.get("hipengine")
        if isinstance(hipengine, dict):
            accounting = hipengine.get("token_accounting")
            if isinstance(accounting, dict):
                choices = accounting.get("choice_generated_token_ids")
                if (
                    isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], list)
                    and choices[0]
                ):
                    return [int(token) for token in choices[0]]
    return []


def _row_prompt_token_count(row: Mapping[str, Any]) -> int | None:
    response = row.get("response")
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
            return int(usage["prompt_tokens"])
    return None


def _record_reference_row(
    record: Mapping[str, Any],
    *,
    row_label: str | None,
) -> dict[str, Any]:
    """Read the prompt text and AR reference ids out of an HTTP matrix record.

    Records in ``/tmp/mtp-repair`` come in two shapes: a row-level ``ids`` list
    and an OpenAI response whose ids sit under
    ``response.hipengine.token_accounting.choice_generated_token_ids``. Without
    an explicit label the first row that did not use MTP wins, because that row
    is the AR reference trajectory.
    """

    rows = record.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("record has no rows[]")
    candidates: list[tuple[int, Mapping[str, Any], list[int]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row_label is not None and str(row.get("label")) != str(row_label):
            continue
        if int(row.get("status", 0)) != 200:
            continue
        ids = _row_ids(row)
        if len(ids) < 2:
            continue
        mtp = row.get("mtp") if isinstance(row.get("mtp"), dict) else {}
        request = row.get("request") if isinstance(row.get("request"), dict) else {}
        uses_mtp = bool(mtp.get("used")) or bool(request.get("speculative_mtp"))
        candidates.append((0 if not uses_mtp else 1, row, ids))
    if not candidates:
        raise SystemExit(
            f"record has no row matching label={row_label!r} with status 200 and "
            "at least two generated ids"
        )
    candidates.sort(key=lambda item: item[0])
    row = candidates[0][1]
    ids = candidates[0][2]
    request = row.get("request")
    if not isinstance(request, dict):
        raise SystemExit("record row has no request{}")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise SystemExit("record request has no messages[]")
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise SystemExit("record request carries a non-object message")
        role = str(message.get("role", "user"))
        if role not in {"user", "system"}:
            raise SystemExit(
                f"record request carries a {role!r} message; this diagnostic "
                "renders a single-user-message chat prompt only"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise SystemExit(
                "record request message content is not plain text; this "
                "diagnostic cannot render it"
            )
        contents.append(content)
    if len(contents) > 1:
        raise SystemExit(
            f"record request carries {len(contents)} messages; this diagnostic "
            "renders a single-user-message chat prompt only"
        )
    return {
        "label": str(row.get("label")),
        "prompt": contents[0],
        "reference_ids": ids,
        "expected_prompt_tokens": _row_prompt_token_count(row),
        "mtp": dict(row.get("mtp") or {}),
    }


# ---------------------------------------------------------------------------
# Runtime arms
# ---------------------------------------------------------------------------


def _new_session(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    max_sequence_length: int,
):
    from hipengine.runtime.prefill import PrefillConfig
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

    return Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=args.use_gemv_decode,
        prefill_config=PrefillConfig(),
    )


def _advance_prefix(
    session: Any,
    *,
    prompt_tokens: list[int],
    prefill_attention_mode: str,
    initial_prev_token: int,
    initial_prev_position: int,
    prefix_advance_tokens: list[int],
) -> int:
    """Prefill, then advance the prefix to the cycle boundary with exact AR."""

    current_prev = _prefill_probe_session(
        session,
        prompt_tokens=prompt_tokens,
        prefill_attention_mode=prefill_attention_mode,
        initial_prev_token=int(initial_prev_token),
        initial_prev_position=int(initial_prev_position),
    )
    if prefix_advance_tokens:
        session.verify_target_block_serial_exact(
            prefix_advance_tokens,
            advance_state_only=True,
        )
        current_prev = int(prefix_advance_tokens[-1])
    expected = int(initial_prev_position) + len(prefix_advance_tokens)
    if int(session.position) != expected:
        raise ProbeError(
            f"prefix cursor {session.position} does not match expected {expected}"
        )
    return current_prev


def _fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def ar_control_consumed_tokens(commit: Mapping[str, Any]) -> list[int]:
    """Return the verifier inputs the native accept consumed, in order.

    The N2 graph processes and commits rows ``0..commit_row``: the root input
    plus the accepted drafts. ``commit["token_ids"]`` is
    ``drafts[:accepted] + [target_top1[commit_row]]``, whose last element is the
    bonus prediction at the committed row and was never consumed. An empty list
    means the device reported a commit row outside the captured inputs.
    """

    input_tokens = [int(token) for token in (commit.get("input_token_ids") or ())]
    commit_row = int(commit["commit_row"])
    if not input_tokens or commit_row < 0 or commit_row >= len(input_tokens):
        return []
    return input_tokens[: commit_row + 1]


def _expected_linear_layers(session: Any) -> list[int]:
    """Return the layer ids whose Conv/GDN state the commit must copy."""

    from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION

    if session.runner is None:
        raise ProbeError("resident session is closed")
    config = session.runner.weights.config
    return [
        int(layer_id)
        for layer_id, layer_type in enumerate(config.layer_types)
        if layer_type == LINEAR_ATTENTION
    ]


def _capture_hidden_commit_row(
    session: Any,
    *,
    hidden_rows: Any,
    commit_row: int,
) -> dict[str, Any] | None:
    """Read the selected hidden tap and the committed hidden seed.

    The N2 graph commits the selected row of the captured hidden taps into
    ``scratch.hidden_seed_fp32`` with the same commit-row buffer the linear
    commit uses. Comparing the two proves the hidden side of the state commit
    for every model, including artifacts with no linear-attention layers.
    """

    if session.scratch is None or session.runner is None:
        raise ProbeError("resident session is closed")
    if hidden_rows is None:
        return None
    hidden_size = int(session.runner.hidden_size)
    row_nbytes = hidden_size * 4
    rows = np.ascontiguousarray(np.asarray(hidden_rows, dtype=np.float32))
    if (
        rows.ndim != 2
        or int(rows.shape[1]) != hidden_size
        or int(commit_row) >= int(rows.shape[0])
    ):
        return None
    return {
        "source_row": np.ascontiguousarray(rows[int(commit_row)]).view(np.uint8),
        "destination": _copy_device_bytes(
            session,
            int(session.scratch.hidden_seed_fp32.ptr),
            row_nbytes,
        ),
        "selected_row": int(commit_row),
    }


def _hidden_row_summary(
    hidden_pair: Mapping[str, Any] | None,
    *,
    commit_row: int,
) -> dict[str, Any] | None:
    if hidden_pair is None:
        return None
    source = _as_u8(hidden_pair["source_row"])
    destination = _as_u8(hidden_pair["destination"])
    return {
        "selected_row": int(commit_row),
        "nbytes": int(destination.size),
        "source_row_digest": _bytes_digest(source),
        "destination_digest": _bytes_digest(destination),
        "copy_exact": bool(np.array_equal(source, destination)),
        "source_row_summary": (
            _numeric_summary_from_raw(
                source, dtype="fp32", label=f"selected_hidden_row_{int(commit_row)}"
            )
            if _fp32_view(source) is not None
            else None
        ),
        "destination_summary": (
            _numeric_summary_from_raw(
                destination,
                dtype="fp32",
                label=f"committed_hidden_seed_{int(commit_row)}",
            )
            if _fp32_view(destination) is not None
            else None
        ),
    }


def _capture_commit_rows(
    session: Any,
    *,
    commit_row: int,
    rows: int | None = None,
) -> list[dict[str, Any]]:
    """Read the selected captured row and the committed destination per layer.

    The source rows are the verifier's own captured linear-state rows, the same
    buffers ``_ensure_verify_linear_state_commit_tables`` hands to
    ``linear_commit_kernel``. Row 0 is also read when it is not the selected
    row, so a wrong-row copy can be named instead of only detected.

    A ``commit_row`` outside the captured verifier rows reads nothing: the row
    offset would address memory the verifier never captured. The ownership check
    reports that out-of-range commit row instead.
    """

    if session.runner is None or session.scratch is None:
        raise ProbeError("resident session is closed")
    if rows is not None and not (0 <= int(commit_row) < int(rows)):
        return []
    layers: list[dict[str, Any]] = []
    for layer_id in _expected_linear_layers(session):
        pair = session._verify_linear_state_row_pair(layer_id)
        destination_conv = session.scratch.layer_conv_states[layer_id]
        destination_recurrent = session.scratch.layer_recurrent_states[layer_id]
        if pair is None or destination_conv is None or destination_recurrent is None:
            continue
        source_conv_rows, source_recurrent_rows = pair
        entry: dict[str, Any] = {"layer": int(layer_id)}
        for part, source_rows, destination in (
            ("conv", source_conv_rows, destination_conv),
            ("recurrent", source_recurrent_rows, destination_recurrent),
        ):
            row_nbytes = int(destination.nbytes)
            entry[part] = {
                "source_row": _copy_device_bytes(
                    session,
                    int(source_rows.ptr) + int(commit_row) * row_nbytes,
                    row_nbytes,
                ),
                "destination": _copy_device_bytes(
                    session, int(destination.ptr), row_nbytes
                ),
            }
            if int(commit_row) != 0:
                entry[part]["other_row"] = {
                    "index": 0,
                    "bytes": _copy_device_bytes(
                        session, int(source_rows.ptr), row_nbytes
                    ),
                }
        layers.append(entry)
    return layers


def _commit_rows_summary(
    layers: Sequence[Mapping[str, Any]], *, commit_row: int
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for layer in layers:
        entry: dict[str, Any] = {
            "layer": int(layer["layer"]),
            "selected_row": int(commit_row),
        }
        for part in ("conv", "recurrent"):
            source = _as_u8(layer[part]["source_row"])
            destination = _as_u8(layer[part]["destination"])
            item: dict[str, Any] = {
                "nbytes": int(destination.size),
                "source_row_digest": _bytes_digest(source),
                "destination_digest": _bytes_digest(destination),
                "copy_exact": bool(np.array_equal(source, destination)),
                "source_row_summary": (
                    _numeric_summary_from_raw(
                        source,
                        dtype="fp32",
                        label=f"selected_row_{int(commit_row)}_{part}",
                    )
                    if _fp32_view(source) is not None
                    else None
                ),
                "destination_summary": (
                    _numeric_summary_from_raw(
                        destination,
                        dtype="fp32",
                        label=f"destination_{int(commit_row)}_{part}",
                    )
                    if _fp32_view(destination) is not None
                    else None
                ),
            }
            other = layer[part].get("other_row")
            if other is not None:
                other_bytes = _as_u8(other["bytes"])
                item["other_row_index"] = int(other["index"])
                item["other_row_digest"] = _bytes_digest(other_bytes)
                item["destination_matches_other_row"] = bool(
                    np.array_equal(other_bytes, destination)
                )
            entry[part] = item
        summary.append(entry)
    return summary


def _capture_linear_state_bytes(session: Any) -> dict[int, dict[str, Any]]:
    captured: dict[int, dict[str, Any]] = {}
    if session.scratch is None:
        raise ProbeError("resident session is closed")
    for layer_id, (conv_state, recurrent_state) in enumerate(
        zip(
            session.scratch.layer_conv_states,
            session.scratch.layer_recurrent_states,
            strict=True,
        )
    ):
        if conv_state is None or recurrent_state is None:
            continue
        captured[int(layer_id)] = {
            "conv": _copy_device_bytes(
                session, int(conv_state.ptr), int(conv_state.nbytes)
            ),
            "recurrent": _copy_device_bytes(
                session, int(recurrent_state.ptr), int(recurrent_state.nbytes)
            ),
        }
    return captured


def _run_native_arm(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    prompt_tokens: list[int],
    prefix_tokens: list[int],
    initial_prev_token: int,
    initial_prev_position: int,
    verifier_inputs: list[int],
    drafts: list[int],
    max_sequence_length: int,
    remaining_decode: int,
    keep_state_bytes: bool,
) -> dict[str, Any]:
    """Run the singleton native cycle and capture its commit provenance."""

    session = _new_session(
        args=args,
        compiler_version=compiler_version,
        max_sequence_length=max_sequence_length,
    )
    try:
        _advance_prefix(
            session,
            prompt_tokens=prompt_tokens,
            prefill_attention_mode=str(args.prefill_attention_mode),
            initial_prev_token=initial_prev_token,
            initial_prev_position=initial_prev_position,
            prefix_advance_tokens=prefix_tokens,
        )
        start_position = int(session.position)
        from hipengine.runtime.gguf_native_spec_cycle import (
            NativeSpecTargetGraphUnsupportedError,
        )

        try:
            result = session.verify_target_block_native_cycle(
                list(verifier_inputs),
                fallback=False,
                cycle_id=0,
                transaction_id=1,
                request_id=0,
                bulk_attention_mode=str(args.native_bulk_attention_mode),
                use_wmma_prefill=False,
                capture_linear_state_rows=True,
                capture_pre_output_norm_hidden=True,
                capture_lm_head_logits=True,
                defer_linear_state_commit=True,
                device_accept_commit=True,
                remaining_decode=int(remaining_decode),
            )
        except NativeSpecTargetGraphUnsupportedError as error:
            raise ProbeError(
                "the singleton native cycle is unavailable for this session and "
                f"this diagnostic does not substitute the eager fallback: {error}"
            ) from error
        session.runtime.device_synchronize()
        fallback_reason = getattr(session, "last_native_spec_target_fallback_reason", None)
        if not bool(getattr(session, "last_native_spec_target_submitted", False)):
            raise ProbeError(
                "the native target graph did not launch; the eager fallback ran "
                f"instead (fallback_reason={fallback_reason!r}). A fallback run "
                "is not this lane"
            )
        logits = getattr(result, "lm_head_logits_f32", None)
        if logits is None:
            raise ProbeError(
                "the native target graph returned no full logits; "
                "capture_lm_head_logits needs the resident verifier logits "
                "buffer, which this session did not provide"
            )
        logits = np.ascontiguousarray(np.asarray(logits, dtype=np.float32))
        if logits.shape[0] != len(verifier_inputs):
            raise ProbeError(
                f"native cycle returned {logits.shape[0]} logit rows for "
                f"{len(verifier_inputs)} inputs"
            )
        target_top1 = [int(token) for token in (getattr(result, "target_top1", ()) or ())]
        if len(target_top1) != len(verifier_inputs):
            raise ProbeError(
                f"native cycle returned {len(target_top1)} top-1 rows for "
                f"{len(verifier_inputs)} inputs"
            )
        accepted = int(result.accepted_draft_tokens)
        commit_row = int(result.commit_row)
        input_token_ids = [int(token) for token in result.input_token_ids]
        expected_layers = _expected_linear_layers(session)
        layers = _capture_commit_rows(
            session,
            commit_row=commit_row,
            rows=len(input_token_ids),
        )
        hidden_pair = _capture_hidden_commit_row(
            session,
            hidden_rows=getattr(result, "hidden_seeds", None),
            commit_row=commit_row,
        )
        ownership_record = {
            "accepted": accepted,
            "rows": len(input_token_ids),
            "commit_row": commit_row,
            "commit_token": int(result.commit_token),
            "commit_position": int(result.commit_position),
            "start_position": int(result.start_position),
            "end_position": int(result.end_position),
            "cursor_after": int(session.position),
            "input_token_ids": input_token_ids,
            "visible_tokens": [int(token) for token in result.token_ids],
            "draft_tokens": [int(token) for token in drafts],
            "target_top1": target_top1,
            "layers": layers,
            "expected_layers": expected_layers,
            "hidden_required": True,
            "hidden_row": hidden_pair,
        }
        ownership = commit_ownership_check(ownership_record)
        commit_rows = _commit_rows_summary(layers, commit_row=commit_row)
        hidden_row_summary = _hidden_row_summary(hidden_pair, commit_row=commit_row)
        # ``current_prev`` names the token processed at ``position - 1``: the
        # committed input ``input_token_ids[commit_row]``. ``result.token_ids``
        # ends with the bonus prediction ``target_top1[commit_row]``, which was
        # never consumed, so it cannot describe the committed cursor.
        state = _session_lifecycle_fingerprint(
            session,
            current_prev=int(result.commit_token),
            label="native",
        )
        linear_state_bytes = None
        if keep_state_bytes:
            linear_state_bytes = {
                int(layer["layer"]): {
                    "conv": layer["conv"]["destination"],
                    "recurrent": layer["recurrent"]["destination"],
                }
                for layer in layers
            }
        return {
            "arm": "native",
            "start_position": start_position,
            "verifier_drafts": [int(token) for token in drafts],
            "row_top1": target_top1,
            "row_logits": logits,
            "logits_digest": _bytes_digest(logits.view(np.uint8)),
            "accepted": accepted,
            "visible": [int(token) for token in result.token_ids],
            "commit": {
                "accepted": accepted,
                "commit_row": commit_row,
                "commit_token": int(result.commit_token),
                "input_token_ids": input_token_ids,
                "consumed_input_tokens": ar_control_consumed_tokens(
                    {
                        "input_token_ids": input_token_ids,
                        "commit_row": commit_row,
                    }
                ),
                "commit_position": int(result.commit_position),
                "end_position": int(result.end_position),
                "cursor_after": int(session.position),
                "full_accept": bool(result.full_accept),
                "next_token": int(result.next_token),
            },
            "ownership": ownership,
            "commit_rows": commit_rows,
            "hidden_row": hidden_row_summary,
            "state": state,
            "state_digest": _fingerprint_digest(state),
            "linear_state_bytes": linear_state_bytes,
            "native_launch": {
                "launch": _NATIVE_LAUNCH,
                "device_accept_commit": True,
                "fallback_reason": fallback_reason,
                "capture_ms": float(
                    getattr(session, "last_native_spec_target_capture_ms", 0.0)
                ),
                "submit_ms": float(
                    getattr(session, "last_native_spec_target_submit_ms", 0.0)
                ),
                "readback_ms": float(
                    getattr(session, "last_native_spec_target_readback_ms", 0.0)
                ),
            },
            "notes": [
                "singleton native cycle: verify_target_block_native_cycle with "
                "device_accept_commit=True, capture_linear_state_rows=True, and "
                "capture_lm_head_logits=True",
                "ownership is checked against the captured source row, the "
                "committed token, and the cursor, never against AR state",
            ],
        }
    finally:
        session.close()


def _run_ar_arm(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    prompt_tokens: list[int],
    prefix_tokens: list[int],
    initial_prev_token: int,
    initial_prev_position: int,
    verifier_inputs: list[int],
    max_sequence_length: int,
    session_factory: Callable[..., Any] | None = None,
    fingerprint: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Token-serial AR over the whole verifier block.

    ``session_factory`` and ``fingerprint`` exist so the CPU tests can run this
    arm against a recording session; the GPU path leaves both at their
    defaults.
    """

    session = (session_factory or _new_session)(
        args=args,
        compiler_version=compiler_version,
        max_sequence_length=max_sequence_length,
    )
    try:
        _advance_prefix(
            session,
            prompt_tokens=prompt_tokens,
            prefill_attention_mode=str(args.prefill_attention_mode),
            initial_prev_token=initial_prev_token,
            initial_prev_position=initial_prev_position,
            prefix_advance_tokens=prefix_tokens,
        )
        start_position = int(session.position)
        row_top1: list[int] = []
        row_logits: list[np.ndarray] = []
        for token in verifier_inputs:
            step = session.step(
                int(token), return_logits=True, capture_hidden_seed_fp32=True
            )
            row_top1.append(int(step.token_id))
            logits = _float32_rows(np.atleast_2d(step.logits), label="AR step")
            if logits.shape[0] != 1:
                raise ProbeError(f"AR step returned {logits.shape[0]} logit rows")
            row_logits.append(logits[0])
        # The block consumed every input row, so the token processed at
        # ``position - 1`` is the last verifier input, not its prediction.
        state = (fingerprint or _session_lifecycle_fingerprint)(
            session,
            current_prev=int(verifier_inputs[-1]),
            label="ar",
        )
        return {
            "arm": "ar",
            "start_position": start_position,
            "row_top1": row_top1,
            "row_logits": np.stack(row_logits, axis=0),
            "state": state,
            "state_digest": _fingerprint_digest(state),
            "notes": [
                "token-serial AR over [root] + drafts; reference for row "
                "decisions and for resident state at start + rows",
            ],
        }
    finally:
        session.close()


def _run_ar_control_arm(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    prompt_tokens: list[int],
    prefix_tokens: list[int],
    initial_prev_token: int,
    initial_prev_position: int,
    consumed_input_tokens: Sequence[int],
    max_sequence_length: int,
    session_factory: Callable[..., Any] | None = None,
    fingerprint: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay the verifier inputs the native accept consumed.

    The N2 graph processes and commits ``input_token_ids[:commit_row + 1]``: the
    root row plus the accepted drafts. ``result.token_ids`` is
    ``drafts[:accepted] + [target_top1[commit_row]]``; its last element is the
    bonus prediction at the committed row, which the graph never fed back.
    Stepping that stream would drop the root input and treat the prediction as
    consumed, so the control would describe a different token sequence at the
    same cursor.

    ``session_factory`` and ``fingerprint`` exist so the CPU tests can run this
    arm against a recording session; the GPU path leaves both at their defaults.
    """

    consumed = [int(token) for token in consumed_input_tokens]
    if not consumed:
        raise ProbeError(
            "the AR control needs at least the root input token; an empty "
            "consumed list means the commit row was outside the captured rows"
        )
    session = (session_factory or _new_session)(
        args=args,
        compiler_version=compiler_version,
        max_sequence_length=max_sequence_length,
    )
    try:
        _advance_prefix(
            session,
            prompt_tokens=prompt_tokens,
            prefill_attention_mode=str(args.prefill_attention_mode),
            initial_prev_token=initial_prev_token,
            initial_prev_position=initial_prev_position,
            prefix_advance_tokens=prefix_tokens,
        )
        for token in consumed:
            session.step(token, return_logits=False, capture_hidden_seed_fp32=True)
        state = (fingerprint or _session_lifecycle_fingerprint)(
            session,
            current_prev=consumed[-1],
            label="ar_control",
        )
        return {
            "arm": "ar_control",
            "consumed_input_tokens": consumed,
            "state": state,
            "state_digest": _fingerprint_digest(state),
            "linear_state_bytes": _capture_linear_state_bytes(session),
            "notes": [
                "token-serial AR over exactly the verifier inputs the native "
                "cycle consumed (root plus accepted drafts), never over its "
                "output predictions; state baseline at the committed cursor",
            ],
        }
    finally:
        session.close()


def _run_serial_arm(
    *,
    args: argparse.Namespace,
    compiler_version: str | None,
    prompt_tokens: list[int],
    prefix_tokens: list[int],
    initial_prev_token: int,
    initial_prev_position: int,
    verifier_inputs: list[int],
    max_sequence_length: int,
    session_factory: Callable[..., Any] | None = None,
    fingerprint: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Strict serial-exact control over the whole verifier block."""

    session = (session_factory or _new_session)(
        args=args,
        compiler_version=compiler_version,
        max_sequence_length=max_sequence_length,
    )
    try:
        _advance_prefix(
            session,
            prompt_tokens=prompt_tokens,
            prefill_attention_mode=str(args.prefill_attention_mode),
            initial_prev_token=initial_prev_token,
            initial_prev_position=initial_prev_position,
            prefix_advance_tokens=prefix_tokens,
        )
        result = session.verify_target_block_serial_exact(list(verifier_inputs))
        row_top1 = [int(token) for token in result.token_ids]
        # As in the AR arm: the last processed token is the last input row.
        state = (fingerprint or _session_lifecycle_fingerprint)(
            session,
            current_prev=int(verifier_inputs[-1]),
            label="serial",
        )
        return {
            "arm": "serial",
            "row_top1": row_top1,
            "row_logits": None,
            "state": state,
            "state_digest": _fingerprint_digest(state),
            "notes": [
                "verify_target_block_serial_exact control; a state mismatch "
                "against AR makes the comparison basis inconclusive",
            ],
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# CLI and report
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute one GGUF MTP divergence on the singleton native target "
            "cycle: commit ownership, same-schedule repeatability, and measured "
            "numerical drift are reported separately. Diagnostic only."
        )
    )
    parser.add_argument("--model", type=Path, default=Path(GGUF_PATH))
    parser.add_argument(
        "--record",
        type=Path,
        required=True,
        help=(
            "HTTP matrix record (for example /tmp/mtp-repair/"
            "moe-hostproposal-native.json) holding the prompt text and the AR "
            "reference ids"
        ),
    )
    parser.add_argument(
        "--record-row",
        default=None,
        help=(
            "rows[] label to read the reference ids from; default is the first "
            "row that did not use MTP, which is the AR arm"
        ),
    )
    parser.add_argument(
        "--output-index",
        type=int,
        required=True,
        help="generated token index to attribute (for example 35 or 118)",
    )
    parser.add_argument(
        "--lane",
        choices=(LANE_AUTO, LANE_SINGLETON_REQUEST, LANE_PACKED_REQUEST),
        default=LANE_AUTO,
        help=(
            "target-verifier lane; auto resolves a lone request to the "
            "singleton native cycle"
        ),
    )
    parser.add_argument(
        "--packed-c1-target",
        action="store_true",
        help=(
            "the request runs under an explicit packed_c1_target evidence row; "
            "this diagnostic does not reproduce that lane and reports it as such"
        ),
    )
    parser.add_argument(
        "--max-active-requests",
        type=int,
        default=1,
        help=(
            "recorded serving capacity. It is not used to select the lane: a "
            "lone speculative request keeps its per-request verifier"
        ),
    )
    parser.add_argument("--draft-budget", type=int, default=2)
    parser.add_argument(
        "--force-partial-accept",
        action="store_true",
        help=(
            "perturb the last draft so the accept is partial; exercises the "
            "selected-row ownership path rather than the full-accept path"
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="native-arm runs at the same prefix; 2 or more checks repeatability",
    )
    parser.add_argument(
        "--remaining-decode",
        type=int,
        default=None,
        help="output room handed to the device accept; default is the row count",
    )
    parser.add_argument(
        "--prompt-reasoning",
        choices=("off", "none", "open"),
        default="none",
        help=(
            "chat-prompt reasoning mode. none reproduces both recorded runs "
            "(512 and 22 prompt tokens); off and open each add tokens, and the "
            "prompt-length check rejects them. The record's thinking_policy is a "
            "sampler-level MTP policy and does not change the rendered prompt"
        ),
    )
    parser.add_argument(
        "--expect-prompt-tokens",
        type=int,
        default=None,
        help=(
            "expected rendered prompt length; default is the record row's "
            "usage.prompt_tokens when present"
        ),
    )
    parser.add_argument(
        "--prefill-attention-mode",
        choices=("bulk", "native"),
        default="bulk",
    )
    parser.add_argument(
        "--native-bulk-attention-mode",
        choices=("native", "bulk"),
        default="native",
        help=(
            "bulk_attention_mode the production native verifier passes; long "
            "context buckets require native split-K attention"
        ),
    )
    parser.add_argument(
        "--use-wmma-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-gemv-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="decode GEMV selection; default lets the session resolve it",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bytes, bytearray)):
        return {"nbytes": len(value), "blake2b_128": _bytes_digest(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _distributions(row_logits: np.ndarray | None, *, top_k: int) -> list[Any]:
    if row_logits is None:
        return []
    return [_top_k_rows(row, top_k=int(top_k)) for row in row_logits]


def _env_contract() -> dict[str, Any]:
    import os

    enabled = {
        name: os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in (*NATIVE_REFUSED_ENV_PAIR, NATIVE_REFUSED_LOGIT_ENV)
    }
    if all(enabled[name] for name in NATIVE_REFUSED_ENV_PAIR):
        raise ProbeError(
            "the native target graph refuses the F32 residual and F32 token "
            f"embedding diagnostics together ({', '.join(NATIVE_REFUSED_ENV_PAIR)}); "
            "unset one of them and re-run"
        )
    if enabled[NATIVE_REFUSED_LOGIT_ENV]:
        raise ProbeError(
            f"{NATIVE_REFUSED_LOGIT_ENV}=1 selects the direct Q6 top-1 verifier "
            "path, which does not leave full row logits for the drift "
            "measurement. Unset it and re-run to match a recorded run that did "
            "not enable it"
        )
    return {
        "refused_pair": list(NATIVE_REFUSED_ENV_PAIR),
        "refused_logit_path": NATIVE_REFUSED_LOGIT_ENV,
        "enabled": enabled,
        "native_capture_env": {
            name: os.environ.get(name) for name in enabled
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        lane = resolve_lane(
            requested=str(args.lane),
            request_count=1,
            packed_c1_target=bool(args.packed_c1_target),
            max_active_requests=int(args.max_active_requests),
        )
        require_reproducible_lane(lane)
        if args.repeat < 1:
            raise ProbeError("--repeat must be positive")
        if args.draft_budget < 1 or args.draft_budget > 7:
            raise ProbeError("--draft-budget must be between 1 and 7")
        env_contract = _env_contract()
        if not _hip_available():
            raise ProbeError(
                "ROCm/HIP runtime is unavailable; this diagnostic needs a GPU"
            )
        if not args.model.exists():
            raise ProbeError(f"model not found: {args.model}")

        record = _read_json(args.record)
        reference = _record_reference_row(record, row_label=args.record_row)
        reference_ids = [int(token) for token in reference["reference_ids"]]
        output_index = int(args.output_index)
        if output_index < 1 or output_index >= len(reference_ids):
            raise ProbeError(
                f"--output-index {output_index} is outside the reference "
                f"trajectory of {len(reference_ids)} ids"
            )
        budget = int(args.draft_budget)
        if output_index + budget >= len(reference_ids):
            raise ProbeError(
                f"--output-index {output_index} + budget {budget} exceeds the "
                "reference trajectory"
            )

        from hipengine.loading.gguf import scan_gguf
        from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

        compiler_version = None
        if args.compiler_version_file is not None:
            compiler_version = args.compiler_version_file.read_text(
                encoding="utf-8"
            )
        tokenizer = Qwen35GGUFTokenizer.from_gguf_info(scan_gguf(args.model))
        prompt_tokens = build_chat_prompt(
            tokenizer,
            str(reference["prompt"]),
            reasoning=str(args.prompt_reasoning),
        )
        if not prompt_tokens:
            raise ProbeError("prompt tokenization produced no token ids")
        expected_prompt_tokens = (
            int(args.expect_prompt_tokens)
            if args.expect_prompt_tokens is not None
            else reference.get("expected_prompt_tokens")
        )
        if expected_prompt_tokens is not None and int(expected_prompt_tokens) != len(
            prompt_tokens
        ):
            raise ProbeError(
                f"rendered prompt has {len(prompt_tokens)} tokens but the record "
                f"reported {int(expected_prompt_tokens)}; the prompt template or "
                "reasoning mode does not reproduce the recorded request"
            )

        initial_prev_position = len(prompt_tokens)
        initial_prev_token = int(reference_ids[0])
        prefix_tokens = [int(token) for token in reference_ids[: output_index - 1]]
        root_token = int(reference_ids[output_index - 1])
        drafts = [
            int(token)
            for token in reference_ids[output_index : output_index + budget]
        ]
        if args.force_partial_accept:
            replacement_index = output_index + budget
            if replacement_index >= len(reference_ids):
                raise ProbeError(
                    "--force-partial-accept needs one more reference id than "
                    f"--output-index {output_index} + budget {budget}"
                )
            replacement = int(reference_ids[replacement_index])
            if replacement == int(drafts[-1]):
                raise ProbeError(
                    "--force-partial-accept could not find a distinct "
                    "replacement token"
                )
            drafts[-1] = replacement
        verifier_inputs = [root_token, *drafts]
        remaining_decode = (
            int(args.remaining_decode)
            if args.remaining_decode is not None
            else len(verifier_inputs)
        )
        if remaining_decode < 1:
            raise ProbeError("--remaining-decode must be at least 1")
        max_sequence_length = len(prompt_tokens) + len(reference_ids) + budget + 8

        shared = {
            "args": args,
            "compiler_version": compiler_version,
            "prompt_tokens": prompt_tokens,
            "prefix_tokens": prefix_tokens,
            "initial_prev_token": initial_prev_token,
            "initial_prev_position": initial_prev_position,
            "max_sequence_length": max_sequence_length,
        }
        native_runs = [
            _run_native_arm(
                **shared,
                verifier_inputs=verifier_inputs,
                drafts=drafts,
                remaining_decode=remaining_decode,
                keep_state_bytes=index == 0,
            )
            for index in range(int(args.repeat))
        ]
        native_arm = native_runs[0]
        ar_arm = _run_ar_arm(**shared, verifier_inputs=verifier_inputs)
        consumed_input_tokens = ar_control_consumed_tokens(native_arm["commit"])
        ar_control_arm = (
            None
            if not consumed_input_tokens
            else _run_ar_control_arm(
                **shared,
                consumed_input_tokens=consumed_input_tokens,
            )
        )
        serial_arm = _run_serial_arm(**shared, verifier_inputs=verifier_inputs)

        native_state_bytes = native_arm.pop("linear_state_bytes", None)
        ar_control_state_bytes = (
            None
            if ar_control_arm is None
            else ar_control_arm.pop("linear_state_bytes", None)
        )
        state_delta = state_numeric_delta(
            native_state_bytes or {}, ar_control_state_bytes or {}
        )
        repeatability = repeatability_check(native_runs)

        attribution = attribute_divergence(
            {
                "ar": ar_arm,
                "native": native_arm,
                "ar_control": ar_control_arm,
                "serial": serial_arm,
            },
            lane=lane,
            state_delta=state_delta,
            repeatability=repeatability,
        )

        report = {
            "schema": SCHEMA,
            "status": "complete",
            "performance_claim": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repo": _repo_provenance(),
            "model": str(args.model),
            "source_record": str(args.record),
            "record_row": str(reference["label"]),
            "command": exact_command_payload(sys.argv),
            "diagnostic_env": diagnostic_env_payload(),
            "native_env_contract": env_contract,
            "lane": {
                **lane,
                "native_kwargs": {
                    "bulk_attention_mode": str(args.native_bulk_attention_mode),
                    "use_wmma_prefill": False,
                    "capture_linear_state_rows": True,
                    "capture_pre_output_norm_hidden": True,
                    "capture_lm_head_logits": True,
                    "defer_linear_state_commit": True,
                    "device_accept_commit": True,
                    "remaining_decode": int(remaining_decode),
                },
            },
            "inputs": {
                "output_index": output_index,
                "reference_ids": len(reference_ids),
                "prompt_tokens": len(prompt_tokens),
                "expected_prompt_tokens": (
                    None
                    if expected_prompt_tokens is None
                    else int(expected_prompt_tokens)
                ),
                "prompt_reasoning": str(args.prompt_reasoning),
                "start_position": int(native_arm["start_position"]),
                "verifier_inputs": verifier_inputs,
                "draft_tokens": drafts,
                "draft_source": "record_ar_trajectory",
                "draft_budget": budget,
                "force_partial_accept": bool(args.force_partial_accept),
                "repeat": int(args.repeat),
                "remaining_decode": int(remaining_decode),
                "prefill_attention_mode": str(args.prefill_attention_mode),
                "max_sequence_length": int(max_sequence_length),
                "use_gemv_decode": args.use_gemv_decode,
            },
            "attribution": attribution,
            "ar_control_skipped": (
                None
                if ar_control_arm is not None
                else (
                    "the device commit row falls outside the captured verifier "
                    "inputs, so no consumed prefix exists to replay"
                )
            ),
            "arms": {
                "ar": {
                    "row_top1": ar_arm["row_top1"],
                    "top_k": _distributions(
                        ar_arm["row_logits"], top_k=int(args.top_k)
                    ),
                    "state": ar_arm["state"],
                    "notes": ar_arm["notes"],
                },
                "native": {
                    "row_top1": native_arm["row_top1"],
                    "top_k": _distributions(
                        native_arm["row_logits"], top_k=int(args.top_k)
                    ),
                    "accepted": int(native_arm["accepted"]),
                    "visible": native_arm["visible"],
                    "commit": native_arm["commit"],
                    "commit_rows": native_arm["commit_rows"],
                    "hidden_row": native_arm["hidden_row"],
                    "ownership": native_arm["ownership"],
                    "state": native_arm["state"],
                    "native_launch": native_arm["native_launch"],
                    "notes": native_arm["notes"],
                },
                "ar_control": (
                    None
                    if ar_control_arm is None
                    else {
                        "consumed_input_tokens": ar_control_arm[
                            "consumed_input_tokens"
                        ],
                        "state": ar_control_arm["state"],
                        "notes": ar_control_arm["notes"],
                    }
                ),
                "serial": {
                    "row_top1": serial_arm["row_top1"],
                    "state": serial_arm["state"],
                    "notes": serial_arm["notes"],
                },
            },
            "notes": [
                "Diagnostic only: attributes one recorded position, not performance.",
                (
                    "Lane: the singleton native cycle. A lone speculative request "
                    "runs the per-request verifier at any --max-active-requests; "
                    "the packed batch lane is not reproduced here."
                ),
                (
                    "Contracts are separate: strict state-copy equality and commit "
                    "ownership are decisive; repeatability is decisive when "
                    "--repeat > 1; numerical drift is measured and never gating."
                ),
                (
                    "A top-1 or generated-ID difference between the native rows and "
                    "AR is a diagnostic signal, not a correctness failure. The "
                    "native row kernels may carry different numerics than the "
                    "token-serial path."
                ),
                (
                    "The AR control replays the verifier inputs the native "
                    "accept consumed (root plus accepted drafts). It is skipped, "
                    "and the ownership contract decides the verdict, when the "
                    "device commit row falls outside the captured inputs."
                ),
                (
                    "Draft tokens come from the record's AR trajectory because the "
                    "HTTP record carries no per-cycle provider drafts. The root row "
                    "is the production row; deeper rows use AR candidates, so their "
                    "tokens can differ from the production proposal."
                ),
            ],
        }
        text = json.dumps(_json_safe(report), indent=2, sort_keys=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        else:
            print(text)
        return 0
    except ProbeError as error:
        raise SystemExit(f"gguf_mtp_native_state_attribution: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
