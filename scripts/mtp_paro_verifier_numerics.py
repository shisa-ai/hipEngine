#!/usr/bin/env python3
"""B1 target-verifier numerical diagnostic for the PARO MTP fast verifier.

Runs strict and candidate target sessions on the same prompt and strict-owned
B1 proposal/commit schedule. Full-vocabulary verifier logits are compared for
every aligned c2 row. Candidate commits are forced back to the strict-selected
row after each comparison so downstream contexts remain teacher-forced even if
a discrete candidate decision differs.

Scope. This is a B1 host-candidate, synchronized post-commit target diagnostic.
It is not a public adapter, rollback, asynchronous-publication, B2/B3 depth or
quality certification, and its strict teacher is a B1 verifier trajectory, not
an independent serial-AR oracle. A clean run here is a numerical result on one
prompt, not full qualification.

Verdicts. This is a one-prompt screen, so it issues no policy verdict at any row
count. Two named envelopes are reported as observed sample comparisons.
``current_sparse_moe_policy`` (``docs/EXECUTION-PROFILES.md`` 2.11) is the
binding envelope for this sparse top-k MoE model class on median/p90 row KL, a
flip-damage budget and top-1 agreement. ``legacy_mean_tail_kl_envelope`` is the
2026-08-16 mean/p95/p99/max KL plus 99%/97% top-1 envelope; its thresholds and
observed comparisons are reported unchanged so saved historical rejections stay
readable, but it is not binding for this model class. Per 6.1, a declared scope
binds on its top-1 floor only and its scope KL is reported as an advisory.

Per 6.4 a reduced run reports every metric but decides no rate gate (top-1,
flip damage) at any sample size, and its continuous metrics cannot be read
against a full-protocol limit without a control scope and a screen-to-full
ratio interval. A one-prompt capture contains neither, so ``policy_decision`` is
always ``inconclusive`` and ``qualification.full_qualification`` is always
false. ``status`` therefore reports only whether the capture itself ran cleanly:
``failed`` for an independent execution failure (prefill-root mismatch,
nonfinite logits, route drift, decision inconsistency, manifest drift), and
``inconclusive`` otherwise.

Backend scope. The registered strict/production plans, their route binder and
the capture manifest binding exist only for ``hip_gfx1100``, so any other
``--backend`` is refused with :class:`DiagnosticScopeError` instead of producing
a report whose manifest and route attestation describe a different backend.
That refusal is a diagnostic-scope limit, not model, quantization or runtime
admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.execution_profiles import (
    EXECUTION_PROFILE_ENV,
    VariantSelection,
    build_variant_manifest,
    manifest_sha256,
    resolve_runtime_profile,
)
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from hipengine.speculative.mtp_native import NativeMtpChainProposer, NativeMtpW8A16Head
from hipengine.speculative.paro_mtp_profiles import (
    FAST_VERIFIER_CANDIDATE_VARIANT,
    PARO_MTP_BACKEND,
    PARO_MTP_CHAIN_ATTN_MODE_ENV,
    PARO_MTP_CONTRACT_ENV,
    PARO_MTP_MODEL,
    PARO_MTP_MODEL_QUANT,
    PARO_MTP_REGISTRY_QUANT,
    PARO_MTP_ROUTE_ENV,
    PROPOSER_LAYER,
    STRICT_VERIFIER_VARIANT,
    TARGET_CONTRACT_VARIANT,
    VERIFIER_LAYER,
)
from scripts.mtp_prompt_suite_economics import _load_prompt_encoder, _load_prompt_suite
from scripts.quant_quality.metrics import per_row_metrics

_ROUTE_FLAGS = (
    "HIPENGINE_GDN_TLOOP_C1_EXACT",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX",
)
_STRICT_FLAGS = {
    "HIPENGINE_GDN_TLOOP_C1_EXACT": "1",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "1",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": "1",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX": "0",
}
_FAST_FLAGS = {
    "HIPENGINE_GDN_TLOOP_C1_EXACT": "0",
    "HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS": "0",
    "HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT": "0",
    "HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX": "0",
}

# 2026-08-16 mean/tail envelope (``docs/EXECUTION-PROFILES.md`` 6.1). Binding for
# model classes without sparse top-k expert routing; for this sparse MoE model
# class 2.11 supersedes it. Thresholds and observed comparisons are reported
# unchanged so saved historical rejections keep their recorded meaning.
_THRESHOLDS = {
    "mean_kl_max": 1.0e-3,
    "p95_kl_max": 5.0e-3,
    "p99_kl_max": 2.0e-2,
    "max_kl_max": 5.0e-2,
    "top1_min": 0.99,
    "per_scope_top1_min": 0.97,
}

# Router-routing-scoped production KL envelope (``docs/EXECUTION-PROFILES.md``
# 2.11). Binding for this model class; its limits name the observed sample
# comparisons. No envelope decides status (6.4).
_CURRENT_SPARSE_MOE_THRESHOLDS = {
    "median_kl_max": 1.0e-3,
    "p90_kl_max": 2.0e-2,
    "large_drift_kl": 5.0e-2,
    "large_drift_rate_max_global": 3.0e-2,
    "large_drift_rate_max_category": 8.0e-2,
    "top1_min_global": 0.96,
    "top1_min_category": 0.94,
}

# ``docs/EXECUTION-PROFILES.md`` 6.4: the canonical full-protocol fixture caps at
# 12 cases x 129 transitions = 1548 rows across the prompt suite. Reported for
# context only: one prompt's row count never constitutes that protocol, so this
# is not a decision threshold and no gate is decided by reaching it.
_FULL_PROTOCOL_FIXTURE_ROWS = 1548

# Env names this harness or the registered profile binder writes. A pre-existing
# value is recorded so the capture's route provenance is unambiguous.
_ROUTE_ENV_NAMES = _ROUTE_FLAGS + (
    PARO_MTP_ROUTE_ENV,
    PARO_MTP_CHAIN_ATTN_MODE_ENV,
    PARO_MTP_CONTRACT_ENV,
    EXECUTION_PROFILE_ENV,
)


class DiagnosticScopeError(ValueError):
    """Named refusal: this diagnostic's provenance covers one backend only.

    Raised instead of running a capture whose registered manifests and route
    attestation would describe a different backend than the one requested. It is
    a diagnostic-scope limit, not model, quantization or runtime admission: the
    model still runs on other backends through the normal engine surface.
    """


def _require_diagnostic_backend(backend: str) -> str:
    """Return ``backend`` or refuse it as outside this diagnostic's scope."""

    if backend != PARO_MTP_BACKEND:
        raise DiagnosticScopeError(
            "diagnostic scope error: this B1 PARO MTP target-verifier diagnostic is "
            f"restricted to backend {PARO_MTP_BACKEND!r}, but --backend {backend!r} "
            "was requested. The registered strict/production profile plans, their "
            "route binder and the capture manifest binding exist only for "
            f"{PARO_MTP_BACKEND!r}, so this run would report a manifest and route "
            "attestation that do not describe it. This is a diagnostic-scope limit, "
            "not model, quantization or runtime admission."
        )
    return backend


def _backend_scope(backend: str) -> dict[str, Any]:
    """Record why the manifest and route attestation describe one backend."""

    return {
        "backend": backend,
        "required_backend": PARO_MTP_BACKEND,
        "manifest_backend": PARO_MTP_BACKEND,
        "restriction": "gfx1100_only_diagnostic",
        "refusal": "DiagnosticScopeError",
        "note": (
            "The registered PARO MTP strict/production plans, their route binder and "
            "the capture manifest binding exist only for this backend. A run on "
            "another backend is refused rather than reporting a manifest and route "
            "attestation that do not describe it. This is a diagnostic-scope limit, "
            "not model, quantization or runtime admission."
        ),
    }


def _set_route(flags: dict[str, str]) -> None:
    os.environ.update(flags)


def _observed_route_flags() -> dict[str, str | None]:
    """Read the route flags actually in effect, rather than the intended ones."""

    return {key: os.environ.get(key) for key in _ROUTE_FLAGS}


def _ambient_env_snapshot() -> dict[str, str]:
    """Record ambient ``HIPENGINE_*`` selectors present before any arm is bound."""

    return {key: value for key, value in sorted(os.environ.items()) if key.startswith("HIPENGINE_")}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _capture_sha256(
    *,
    prompt: dict[str, Any],
    candidate_manifest_sha256: str,
    rows: Sequence[dict[str, Any]],
    cycles: Sequence[dict[str, Any]],
) -> str:
    payload = {
        "prompt": prompt,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "rows": list(rows),
        "cycles": list(cycles),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _plain_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a resolved variant manifest."""

    payload = dict(manifest)
    payload["selections"] = [dict(row) for row in payload["selections"]]
    return payload


def _review_manifests() -> dict[str, Any]:
    """Report the registered strict/production manifests and the candidate labels.

    The registered plans are the authority: the candidate's registration, its
    ``certified`` flag and its production selection are read from the kernel
    registry and the resolved production manifest, never asserted as constants.
    The pre-2026-10-24 hand-built review manifest is retained under
    ``legacy_review_manifest`` only so saved capture hashes stay identifiable;
    it describes no execution and its scope is not a registered selection.
    """

    strict = resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile="strict",
    )
    production = resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile="production",
    )
    production_selections = [dict(row) for row in production.manifest["selections"]]
    verifier = next(row for row in production_selections if row["layer"] == VERIFIER_LAYER)
    candidate_variant = FAST_VERIFIER_CANDIDATE_VARIANT
    key = KernelKey(
        backend=PARO_MTP_BACKEND,
        layer=VERIFIER_LAYER,
        quant=PARO_MTP_REGISTRY_QUANT,
        variant=candidate_variant,
    )
    registered = is_registered(key)
    route = None
    if registered:
        resolved_route = resolve(
            backend=PARO_MTP_BACKEND,
            layer=VERIFIER_LAYER,
            quant=PARO_MTP_REGISTRY_QUANT,
            variant=candidate_variant,
            missing="none",
        )
        route = resolved_route() if callable(resolved_route) else resolved_route
    strict_verifier_variants = {
        str(row["selected_variant"])
        for row in strict.manifest["selections"]
        if row["layer"] == VERIFIER_LAYER
    }
    legacy = build_variant_manifest(
        profile="production",
        backend=PARO_MTP_BACKEND,
        model=PARO_MTP_MODEL,
        quant=PARO_MTP_MODEL_QUANT,
        kv_policy="paged_bf16_kv_live_spans",
        graph_policy="off_b1_fixed_chain",
        selections=(
            VariantSelection(
                layer=PROPOSER_LAYER,
                scope="b1_graph_off_fixed_chain",
                selected_variant=TARGET_CONTRACT_VARIANT,
                strict_fallback_variant=TARGET_CONTRACT_VARIANT,
                registry_quant=PARO_MTP_REGISTRY_QUANT,
            ),
            VariantSelection(
                layer=VERIFIER_LAYER,
                scope="b1_graph_off_fixed_chain",
                selected_variant=FAST_VERIFIER_CANDIDATE_VARIANT,
                strict_fallback_variant=STRICT_VERIFIER_VARIANT,
                registry_quant=PARO_MTP_REGISTRY_QUANT,
            ),
        ),
    )
    legacy_sha256 = manifest_sha256(legacy)
    return {
        "source": "hipengine.speculative.paro_mtp_profiles registered runtime profile plans",
        "registered_strict_sha256": strict.manifest_sha256,
        "registered_production_sha256": production.manifest_sha256,
        "registered_production_fell_back_to_strict": bool(production.fell_back_to_strict),
        "registered_production_qualification_failure": production.qualification_failure,
        "registered_strict_manifest": _plain_manifest(strict.manifest),
        "registered_production_manifest": _plain_manifest(production.manifest),
        "candidate_verifier_variant": candidate_variant,
        "registered_production_verifier_variant": str(verifier["selected_variant"]),
        "candidate_registered": bool(registered),
        "candidate_certified": bool(getattr(route, "certified", False)),
        "candidate_verifier_profile": getattr(route, "verifier_profile", None),
        "candidate_selected_by_production": candidate_variant
        == str(verifier["selected_variant"]),
        "candidate_selected_by_strict": candidate_variant in strict_verifier_variants,
        # The capture hash binds the registered production manifest, which is the
        # identity the compared routes claim.
        "capture_manifest_sha256": production.manifest_sha256,
        "legacy_review_manifest_sha256": legacy_sha256,
        "legacy_review_manifest": {
            **legacy,
            "describes_execution": False,
            "scope_note": "review_only_scope_not_a_registered_selection",
            "note": (
                "Pre-2026-10-24 hand-built reporting manifest. Its "
                "b1_graph_off_fixed_chain scope is not a registered selection, and "
                "its uncertified/unselected labels were stale. Kept only so saved "
                "v3 capture hashes remain identifiable."
            ),
        },
        # Retained field name: sibling aggregation scripts compare it across
        # captures. Its meaning is unchanged (the legacy review manifest identity).
        "candidate_review_sha256": legacy_sha256,
    }


def _binder_flags_for_profile(profile: str) -> dict[str, str | None]:
    """Return the route flags the registered profile binder sets, sandboxed."""

    resolved = resolve_runtime_profile(
        model=PARO_MTP_MODEL,
        backend=PARO_MTP_BACKEND,
        quant=PARO_MTP_MODEL_QUANT,
        profile=profile,
    )
    if resolved.binder is None:
        raise RuntimeError(f"registered {profile!r} profile has no route binder")
    saved = dict(os.environ)
    try:
        for key in _ROUTE_ENV_NAMES:
            os.environ.pop(key, None)
        resolved.binder(None, resolved)
        return {key: os.environ.get(key) for key in _ROUTE_FLAGS}
    finally:
        for key in set(os.environ) - set(saved):
            del os.environ[key]
        os.environ.update(saved)


def _route_attestation() -> dict[str, Any]:
    """Attest the harness arm flags against the registered profile binders.

    The harness pins these selectors directly instead of constructing a
    generator through ``ResolvedRuntimeProfile.construct_generator``. Comparing
    each arm against what the registered binder writes makes arm drift from the
    registry visible instead of silently mislabelling the capture.
    """

    arms: dict[str, Any] = {}
    matches = True
    for arm, profile, bound in (
        ("strict", "strict", _STRICT_FLAGS),
        ("fast", "production", _FAST_FLAGS),
    ):
        registered = _binder_flags_for_profile(profile)
        mismatches = {
            key: {"bound": bound[key], "registered": registered.get(key)}
            for key in bound
            if registered.get(key) is not None and registered.get(key) != bound[key]
        }
        pins = {key: bound[key] for key in bound if registered.get(key) is None}
        if mismatches:
            matches = False
        arms[arm] = {
            "profile": profile,
            "bound_flags": dict(bound),
            "registered_binder_flags": {
                key: value for key, value in registered.items() if value is not None
            },
            "mismatches": mismatches,
            "pins_outside_binder": pins,
        }
    return {
        "source": "registered runtime profile binder (paro_mtp_profiles._bind_route)",
        "arms": arms,
        "harness_pins_outside_binder": sorted(
            {
                key
                for arm in arms.values()
                for key in arm["pins_outside_binder"]
            }
        ),
        "matches_registered_binder": matches,
        "note": (
            "Agreement means the harness writes the same selectors the registered "
            "binder writes. It does not establish that every selector has a live "
            "consumer in the runtime, nor that ambient selectors outside this list "
            "are inert; ambient HIPENGINE_* values are recorded separately."
        ),
    }


def _target_batch(root: int, context: int, candidate: int) -> TargetVerifyBatch:
    return TargetVerifyBatch.from_draft(
        compile_mtp_chain(
            [MtpDraftRequest(request_id=0, root_position=context, candidate_tokens=(candidate,), active_count=1)],
            candidate_budget=1,
        ),
        root_tokens=(root,),
        root_positions=(context,),
    )


def _copy_logits(session: Qwen35ParoResidentSession, *, rows: int = 2) -> np.ndarray:
    host = np.empty((rows, int(session.vocab_size)), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(session.verify_lm_logits.ptr), int(host.nbytes)),
        host.nbytes,
        runtime=session.runtime,
    )
    return host


def _prefill(
    session: Qwen35ParoResidentSession,
    tokens: Sequence[int],
    *,
    capture: Tensor | None = None,
) -> int:
    result = None
    for position, token in enumerate(tokens):
        if capture is None:
            result = session.step(int(token), position=position, sample=position == len(tokens) - 1)
        else:
            result = session.step_with_hidden_taps(
                int(token),
                position=position,
                capture_layer_ids=(int(session.layer_limit) - 1,),
                capture_hidden_concat=capture,
                capture_row=position,
                sample=position == len(tokens) - 1,
                capture_final_hidden_bf16=capture,
            )
    if result is None:
        raise RuntimeError("prompt prefill produced no root token")
    return int(result.token_id)


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation")
    if successes < 0 or successes > total:
        raise ValueError("Wilson successes must be in [0, total]")
    proportion = float(successes) / float(total)
    z2 = float(z) * float(z)
    denominator = 1.0 + z2 / float(total)
    center = (proportion + z2 / (2.0 * float(total))) / denominator
    radius = (
        float(z)
        * math.sqrt(
            proportion * (1.0 - proportion) / float(total)
            + z2 / (4.0 * float(total) * float(total))
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _summary(values: np.ndarray, top1: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or top1.size != values.size:
        raise ValueError("summary needs non-empty aligned KL/top-1 vectors")
    matches = int(np.count_nonzero(top1))
    low, high = _wilson_interval(matches, int(values.size))
    large_drift_rows = int(
        np.count_nonzero(values > _CURRENT_SPARSE_MOE_THRESHOLDS["large_drift_kl"])
    )
    return {
        "rows": int(values.size),
        "mean_kl": float(np.mean(values)),
        "median_kl": float(np.median(values)),
        "p90_kl": float(np.percentile(values, 90.0)),
        "p95_kl": float(np.percentile(values, 95.0)),
        "p99_kl": float(np.percentile(values, 99.0)),
        "max_kl": float(np.max(values)),
        "large_drift_rows": large_drift_rows,
        "large_drift_rate": float(large_drift_rows) / float(values.size),
        "top1_agreement": float(np.mean(top1)),
        "top1_matches": matches,
        "top1_mismatches": int(values.size) - matches,
        "top1_wilson95_low": low,
        "top1_wilson95_high": high,
    }


def _stable_topk(row: np.ndarray, k: int) -> list[int]:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    if k <= 0 or k > values.size:
        raise ValueError("top-k must be in [1, vocab]")
    candidate_ids = np.argpartition(values, -k)[-k:]
    ordered = candidate_ids[np.lexsort((candidate_ids, -values[candidate_ids]))]
    return [int(token) for token in ordered]


def _token_probability(row: np.ndarray, token_id: int) -> float:
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    maximum = float(np.max(values))
    denominator = float(np.exp(values - maximum).sum())
    return float(math.exp(float(values[int(token_id)]) - maximum) / denominator)


def _row_review_diagnostic(
    strict_row: np.ndarray,
    candidate_row: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    strict_values = np.asarray(strict_row, dtype=np.float32).reshape(-1)
    candidate_values = np.asarray(candidate_row, dtype=np.float32).reshape(-1)
    if strict_values.shape != candidate_values.shape or strict_values.size < 2:
        raise ValueError("row diagnostics need aligned vocab vectors with at least two tokens")
    strict_top = _stable_topk(strict_values, max(2, top_k))
    candidate_top = _stable_topk(candidate_values, max(2, top_k))
    strict_top1 = strict_top[0]
    candidate_top1 = candidate_top[0]
    strict_margin = float(strict_values[strict_top1] - strict_values[strict_top[1]])
    candidate_margin = float(
        candidate_values[candidate_top1] - candidate_values[candidate_top[1]]
    )
    return {
        "strict_topk": strict_top[:top_k],
        "candidate_topk": candidate_top[:top_k],
        "strict_margin": strict_margin,
        "candidate_margin": candidate_margin,
        "strict_top1_candidate_rank": int(
            1 + np.count_nonzero(candidate_values > candidate_values[strict_top1])
        ),
        "candidate_top1_strict_rank": int(
            1 + np.count_nonzero(strict_values > strict_values[candidate_top1])
        ),
        "strict_gap_to_candidate_top1": float(
            strict_values[strict_top1] - strict_values[candidate_top1]
        ),
        "candidate_gap_to_strict_top1": float(
            candidate_values[candidate_top1] - candidate_values[strict_top1]
        ),
        "strict_top1_probability": _token_probability(strict_values, strict_top1),
        "candidate_probability_of_strict_top1": _token_probability(
            candidate_values, strict_top1
        ),
        "strict_probability_of_candidate_top1": _token_probability(
            strict_values, candidate_top1
        ),
        "candidate_top1_probability": _token_probability(
            candidate_values, candidate_top1
        ),
    }


def _scope_summaries(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-scope summaries with the 6.1 binding rule applied.

    A declared scope binds on its top-1 floor only (``passed``). Its mean/p95/p99
    KL against the same global limits is reported as ``kl_within_global_limits``
    and flagged ``advisory`` when it is exceeded; it does not decide admission.
    The maximum-row ceiling is a per-row limit, so it holds in every scope
    whenever it holds globally.
    """

    if not rows:
        raise ValueError("scope summaries need at least one row")
    result: dict[str, dict[str, Any]] = {}
    for dimension in (
        "category",
        "shape",
        "transition",
        "row_role",
        "decision_role",
        "strict_selected_for_commit",
    ):
        groups: dict[str, Any] = {}
        for value in sorted({str(row[dimension]) for row in rows}):
            selected = [row for row in rows if str(row[dimension]) == value]
            summary = _summary(
                np.asarray([row["kl"] for row in selected], dtype=np.float64),
                np.asarray([row["top1_equal"] for row in selected], dtype=np.bool_),
            )
            summary["top5_overlap_mean"] = float(
                np.mean([row["top5_overlap"] for row in selected])
            )
            summary["strict_margin_min"] = float(
                min(row["strict_margin"] for row in selected)
            )
            binding = dimension in {"category", "shape", "transition"}
            kl_within_global_limits = bool(
                summary["mean_kl"] <= _THRESHOLDS["mean_kl_max"]
                and summary["p95_kl"] <= _THRESHOLDS["p95_kl_max"]
                and summary["p99_kl"] <= _THRESHOLDS["p99_kl_max"]
            )
            summary["global_limits"] = {
                "mean_kl_max": _THRESHOLDS["mean_kl_max"],
                "p95_kl_max": _THRESHOLDS["p95_kl_max"],
                "p99_kl_max": _THRESHOLDS["p99_kl_max"],
                "max_kl_max": _THRESHOLDS["max_kl_max"],
                "top1_min": _THRESHOLDS["per_scope_top1_min"],
            }
            summary["kl_within_global_limits"] = kl_within_global_limits
            summary["max_kl_within_global_ceiling"] = bool(
                summary["max_kl"] <= _THRESHOLDS["max_kl_max"]
            )
            summary["binding"] = binding
            summary["passed"] = bool(
                summary["top1_agreement"] >= _THRESHOLDS["per_scope_top1_min"]
            )
            summary["advisory"] = (
                "scope_kl_advisory"
                if binding and not kl_within_global_limits
                else None
            )
            groups[value] = summary
        result[dimension] = groups
    return result


def _envelope_metadata() -> dict[str, Any]:
    """Named envelope identities: which one binds, and what each one reports."""

    return {
        "current_sparse_moe_policy": {
            "source": "docs/EXECUTION-PROFILES.md 2.11",
            "binding": True,
            "applies_to": (
                "sparse top-k expert routing; this harness targets "
                f"{PARO_MTP_MODEL}"
            ),
            "binding_basis": (
                "Router-routing-scoped production KL envelope for models with sparse "
                "top-k expert routing. It is the envelope this report's observed "
                "median/p90 KL, flip-damage and top-1 comparisons are named against."
            ),
            "decides": "nothing; observed sample metrics only",
            "policy_decision": "inconclusive",
            "thresholds": dict(_CURRENT_SPARSE_MOE_THRESHOLDS),
        },
        "legacy_mean_tail_kl_envelope": {
            "source": (
                "2026-08-16 calibration recorded in docs/EXECUTION-PROFILES.md 6.1"
            ),
            "binding": False,
            "binding_basis": (
                "Superseded for this model class by 2.11. Thresholds and observed "
                "comparisons are reported unchanged so saved historical rejections "
                "keep their recorded meaning; they do not decide admission here. For "
                "a model class without sparse top-k expert routing these thresholds "
                "remain the binding 6.1 envelope."
            ),
            "decides": "nothing; observed sample metrics only",
            "policy_decision": "inconclusive",
            "thresholds": dict(_THRESHOLDS),
        },
    }


def _observed_comparison(
    *,
    observed: float,
    limit: float,
    at_least: bool,
    **extra: Any,
) -> dict[str, Any]:
    """One observed sample value beside its limit; never a policy verdict.

    ``sample_within_limit`` states only what this sample did. ``decides_policy``
    is always false: ``docs/EXECUTION-PROFILES.md`` 6.4 forbids deciding a rate
    gate from a reduced run, and forbids reading a screen's continuous metric
    against a full-protocol limit without a calibrated control and a ratio
    interval.
    """

    value = float(observed)
    bound = float(limit)
    within = value >= bound if at_least else value <= bound
    return {
        "observed": value,
        "limit": bound,
        "direction": "at_least" if at_least else "at_most",
        "sample_within_limit": bool(within),
        "decides_policy": False,
        **extra,
    }


def _rate_gate_policy() -> dict[str, Any]:
    """Why no rate gate is decided here, at any row count."""

    return {
        "gates": ["top1_agreement", "large_drift_rate"],
        "decided": False,
        "reason": (
            "docs/EXECUTION-PROFILES.md 6.4 rule 1: top-1 agreement and the "
            "flip-damage rate are rate gates. A reduced run reports them as counts "
            "with intervals and never as a pass/fail, at any row count. This "
            "single-prompt capture is not the full protocol, so no rate gate is "
            "decided here - including one whose interval lies entirely on the "
            "failing side of its limit."
        ),
    }


def _sample_scope(
    rows: Sequence[Mapping[str, Any]], *, prompts: int = 1
) -> dict[str, Any]:
    """What this capture's sample can and cannot decide (6.4)."""

    return {
        "prompts": int(prompts),
        "rows": len(rows),
        "full_protocol_rows": _FULL_PROTOCOL_FIXTURE_ROWS,
        "full_protocol_reached": False,
        "control_scopes": [],
        "screen_to_full_ratio_interval": None,
        "continuous_gate_policy": (
            "docs/EXECUTION-PROFILES.md 6.4 rules 1-3: median and p90 KL are reported "
            "as observed sample values. Reading a sample against a full-protocol "
            "limit needs a control scope with a known full-gate verdict in the same "
            "batch and a screen-to-full ratio interval; a one-prompt capture has "
            "neither, so no continuous gate is decided and no ratio correction is "
            "applied."
        ),
        "rate_gate_policy": _rate_gate_policy()["reason"],
        "reason": (
            "The canonical full protocol is 12 cases x 129 transitions across the "
            "prompt suite (1548 rows), not a row count from one prompt. A "
            "single-prompt B1 capture is a screen at every row count, so it never "
            "becomes the full protocol by accumulating rows."
        ),
        "policy_decision": "inconclusive",
    }


def _robust_scope_summary(
    rows: Sequence[dict[str, Any]], *, scope_kind: str
) -> dict[str, Any]:
    """2.11 statistics for one scope: median/p90 KL, flip damage, top-1."""

    if scope_kind not in {"global", "category"}:
        raise ValueError("scope_kind must be 'global' or 'category'")
    summary = _summary(
        np.asarray([row["kl"] for row in rows], dtype=np.float64),
        np.asarray([row["top1_equal"] for row in rows], dtype=np.bool_),
    )
    damage_max = _CURRENT_SPARSE_MOE_THRESHOLDS[
        "large_drift_rate_max_global"
        if scope_kind == "global"
        else "large_drift_rate_max_category"
    ]
    top1_min = _CURRENT_SPARSE_MOE_THRESHOLDS[
        "top1_min_global" if scope_kind == "global" else "top1_min_category"
    ]
    summary["limits"] = {
        "median_kl_max": _CURRENT_SPARSE_MOE_THRESHOLDS["median_kl_max"],
        "p90_kl_max": _CURRENT_SPARSE_MOE_THRESHOLDS["p90_kl_max"],
        "large_drift_kl": _CURRENT_SPARSE_MOE_THRESHOLDS["large_drift_kl"],
        "large_drift_rate_max": damage_max,
        "top1_min": top1_min,
    }
    summary["median_kl_within_limit"] = bool(
        summary["median_kl"] <= _CURRENT_SPARSE_MOE_THRESHOLDS["median_kl_max"]
    )
    summary["p90_kl_within_limit"] = bool(
        summary["p90_kl"] <= _CURRENT_SPARSE_MOE_THRESHOLDS["p90_kl_max"]
    )
    summary["large_drift_rate_within_limit"] = bool(
        summary["large_drift_rate"] <= damage_max
    )
    summary["top1_within_limit"] = bool(summary["top1_agreement"] >= top1_min)
    return summary


def _current_policy_envelope(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The binding 2.11 envelope, reported as observed sample comparisons."""

    categories = sorted({str(row["category"]) for row in rows})
    aggregate = _robust_scope_summary(rows, scope_kind="global")
    per_category = {
        category: _robust_scope_summary(
            [row for row in rows if str(row["category"]) == category],
            scope_kind="category",
        )
        for category in categories
    }

    def comparisons(summary: Mapping[str, Any]) -> dict[str, Any]:
        limits = summary["limits"]
        damage_low, damage_high = _wilson_interval(
            int(summary["large_drift_rows"]), int(summary["rows"])
        )
        return {
            "median_kl": _observed_comparison(
                observed=summary["median_kl"],
                limit=limits["median_kl_max"],
                at_least=False,
            ),
            "p90_kl": _observed_comparison(
                observed=summary["p90_kl"],
                limit=limits["p90_kl_max"],
                at_least=False,
            ),
            "top1_agreement": _observed_comparison(
                observed=summary["top1_agreement"],
                limit=limits["top1_min"],
                at_least=True,
                rows=int(summary["rows"]),
                mismatches=int(summary["top1_mismatches"]),
                wilson95=[summary["top1_wilson95_low"], summary["top1_wilson95_high"]],
            ),
            "large_drift_rate": _observed_comparison(
                observed=summary["large_drift_rate"],
                limit=limits["large_drift_rate_max"],
                at_least=False,
                large_drift_kl=limits["large_drift_kl"],
                rows=int(summary["rows"]),
                rows_above=int(summary["large_drift_rows"]),
                wilson95=[damage_low, damage_high],
            ),
        }

    global_comparisons = comparisons(aggregate)
    category_comparisons = {
        category: comparisons(summary) for category, summary in per_category.items()
    }
    exceeded = [
        f"global.{name}"
        for name, entry in global_comparisons.items()
        if not entry["sample_within_limit"]
    ] + [
        f"per_category.{category}.{name}"
        for category, entries in category_comparisons.items()
        for name, entry in entries.items()
        if not entry["sample_within_limit"]
    ]
    return {
        **_envelope_metadata()["current_sparse_moe_policy"],
        "aggregate": aggregate,
        "per_category": per_category,
        "observed_comparisons": {
            "global": global_comparisons,
            "per_category": category_comparisons,
        },
        "sample_limits_exceeded": exceeded,
        "rate_gate_policy": _rate_gate_policy(),
        "status": "inconclusive",
    }


def _legacy_envelope(
    aggregate: dict[str, Any],
    *,
    scopes: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, Any]:
    """The 2026-08-16 mean/tail envelope, reported as observed comparisons."""

    binding_scopes = [
        (dimension, value, summary)
        for dimension, groups in scopes.items()
        for value, summary in groups.items()
        if bool(summary["binding"])
    ]
    if not binding_scopes:
        raise ValueError("legacy envelope needs at least one binding scope")
    worst_dimension, worst_value, worst_summary = min(
        binding_scopes, key=lambda item: item[2]["top1_agreement"]
    )
    comparisons = {
        "mean_kl": _observed_comparison(
            observed=aggregate["mean_kl"],
            limit=_THRESHOLDS["mean_kl_max"],
            at_least=False,
        ),
        "p95_kl": _observed_comparison(
            observed=aggregate["p95_kl"],
            limit=_THRESHOLDS["p95_kl_max"],
            at_least=False,
        ),
        "p99_kl": _observed_comparison(
            observed=aggregate["p99_kl"],
            limit=_THRESHOLDS["p99_kl_max"],
            at_least=False,
        ),
        "max_kl": _observed_comparison(
            observed=aggregate["max_kl"],
            limit=_THRESHOLDS["max_kl_max"],
            at_least=False,
        ),
        "top1_agreement": _observed_comparison(
            observed=aggregate["top1_agreement"],
            limit=_THRESHOLDS["top1_min"],
            at_least=True,
            rows=int(aggregate["rows"]),
            mismatches=int(aggregate["top1_mismatches"]),
            wilson95=[
                aggregate["top1_wilson95_low"],
                aggregate["top1_wilson95_high"],
            ],
        ),
        "per_scope_top1_floor": _observed_comparison(
            observed=worst_summary["top1_agreement"],
            limit=_THRESHOLDS["per_scope_top1_min"],
            at_least=True,
            worst_scope={"dimension": worst_dimension, "value": worst_value},
            scopes_below_floor=[
                {
                    "dimension": dimension,
                    "value": value,
                    "top1_agreement": summary["top1_agreement"],
                }
                for dimension, value, summary in binding_scopes
                if summary["top1_agreement"] < _THRESHOLDS["per_scope_top1_min"]
            ],
        ),
    }
    advisories = [
        {
            "dimension": dimension,
            "value": value,
            "mean_kl": summary["mean_kl"],
            "p95_kl": summary["p95_kl"],
            "p99_kl": summary["p99_kl"],
        }
        for dimension, groups in scopes.items()
        for value, summary in groups.items()
        if bool(summary["binding"]) and not bool(summary["kl_within_global_limits"])
    ]
    return {
        **_envelope_metadata()["legacy_mean_tail_kl_envelope"],
        "aggregate": aggregate,
        "observed_comparisons": comparisons,
        "sample_limits_exceeded": [
            name
            for name, entry in comparisons.items()
            if not entry["sample_within_limit"]
        ],
        "scope_kl_advisories": advisories,
        "status": "inconclusive",
    }


def _qualification_scope(*, rows: int) -> dict[str, Any]:
    """State what this capture does and does not certify."""

    return {
        "status": "not_qualified",
        "full_qualification": False,
        "policy_decision": "inconclusive",
        "binding_envelope": "current_sparse_moe_policy",
        "scope": "b1_target_verifier_teacher_forced_diagnostic",
        "rows": int(rows),
        "prompts": 1,
        "full_protocol_reached": False,
        "rate_gates_decided": False,
        "covers": [
            (
                "B1 fixed-chain strict-versus-candidate target verifier "
                "full-vocabulary logits on a strict-owned teacher-forced schedule"
            ),
            "host candidate handoff and synchronized post-commit state",
        ],
        "does_not_cover": [
            "the public adapter's route selection, rollback and asynchronous publication",
            "B2/B3 draft depths, including the depth-2 to depth-1 output-tail transition",
            (
                "isolated kernel attribution: candidate recurrent/KV state is "
                "carried, not reset to strict values per cycle"
            ),
            "serial-AR equality: the strict teacher here is a B1 verifier trajectory",
            (
                "any admission or rate-gate decision: a single-prompt screen issues "
                "none at any row count"
            ),
        ],
        "full_qualification_requires": [
            (
                "task non-inferiority on the applicable prompt suite and "
                "heldouts under a predeclared paired criterion"
            ),
            "three bit-identical fixed-seed repeats of the same manifest and execution schedule",
            "single-slot isolation and unrelated-request replay",
            (
                "the full protocol across the prompt suite, whose top-1 and "
                "flip-damage rate gates a single-prompt capture cannot decide "
                "(EXECUTION-PROFILES 6.4)"
            ),
            "B2/B3 captures through the public adapter",
        ],
    }


def _numerical_report(
    row_metrics: Sequence[dict[str, Any]],
    *,
    prefill_root_equal: bool,
    finite_logits: bool,
    decision_mismatches: Sequence[dict[str, Any]],
    route_flags_attested: bool,
    extra_checks: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Build the report from captured rows (no device access).

    ``status`` says whether the capture itself ran cleanly: ``failed`` when an
    independent execution check failed, ``inconclusive`` otherwise. It is never
    ``passed``, because a one-prompt screen decides no policy gate.

    ``scope_failures`` lists binding scopes whose sample top-1 missed the 6.1
    floor. The sibling aggregation script consumes it across prompts; on one
    prompt it is an observation, not a decision, and no envelope reads it.
    """

    if not row_metrics:
        raise ValueError("numerical report needs at least one row")
    kl = np.asarray([row["kl"] for row in row_metrics], dtype=np.float64)
    top1 = np.asarray([row["top1_equal"] for row in row_metrics], dtype=np.bool_)
    aggregate = _summary(kl, top1)
    aggregate["sample_resolution"] = 1.0 / float(aggregate["rows"])
    aggregate["top5_overlap_mean"] = float(
        np.mean([row["top5_overlap"] for row in row_metrics])
    )
    aggregate["strict_margin_min"] = float(
        min(row["strict_margin"] for row in row_metrics)
    )
    scopes = _scope_summaries(row_metrics)
    scope_failures = [
        {"dimension": dimension, "value": value}
        for dimension, groups in scopes.items()
        for value, summary in groups.items()
        if bool(summary["binding"]) and not bool(summary["passed"])
    ]
    current = _current_policy_envelope(row_metrics)
    legacy = _legacy_envelope(aggregate, scopes=scopes)
    execution_checks = {
        "prefill_root_equal": bool(prefill_root_equal),
        "finite_logits": bool(finite_logits),
        "finite_kl": bool(np.isfinite(kl).all()),
        "task_decisions": not decision_mismatches,
        "route_flags_attested": bool(route_flags_attested),
        **{key: bool(value) for key, value in (extra_checks or {}).items()},
    }
    execution_failures = sorted(
        name for name, passed in execution_checks.items() if not passed
    )
    return {
        "status": "failed" if execution_failures else "inconclusive",
        "policy_decision": "inconclusive",
        "execution_checks": execution_checks,
        "execution_failures": execution_failures,
        "aggregate": aggregate,
        "scopes": scopes,
        "scope_failures": scope_failures,
        "scope_kl_advisories": legacy["scope_kl_advisories"],
        "sample_scope": _sample_scope(row_metrics),
        "envelopes": {
            "current_sparse_moe_policy": current,
            "legacy_mean_tail_kl_envelope": legacy,
        },
        "envelope_disagreement": {
            "current_sample_limits_exceeded": current["sample_limits_exceeded"],
            "legacy_sample_limits_exceeded": legacy["sample_limits_exceeded"],
            "disagree": (
                bool(current["sample_limits_exceeded"])
                != bool(legacy["sample_limits_exceeded"])
            ),
            "note": (
                "Both envelopes are reported as observed sample comparisons and "
                "neither decides admission or a rate gate. Where the 2026-08-16 "
                "mean/tail sample exceeds a limit the binding 2.11 sample does not, "
                "2.11 is the applicable envelope for this sparse top-k MoE model "
                "class; the legacy numbers stay reported so saved rejections remain "
                "readable."
            ),
        },
        "qualification": _qualification_scope(rows=int(aggregate["rows"])),
    }


def run_sequential(
    *,
    model: Path,
    prompts_file: Path,
    prompt_name: str,
    prompt_render: str,
    decode_tokens: int,
    backend: str,
    prompt_tokens_file: Path | None = None,
) -> dict[str, Any]:
    """Run strict capture and fast replay sequentially to bound W7900 VRAM."""

    _require_diagnostic_backend(backend)
    suite = _load_prompt_suite(prompts_file)
    prompt = next((row for row in suite["prompts"] if row["name"] == prompt_name), None)
    if prompt is None:
        raise ValueError(f"unknown prompt {prompt_name!r}")
    if prompt_tokens_file is None:
        encoder = _load_prompt_encoder(model, prompt_render)
        prompt_tokens = [int(token) for token in encoder.encode(prompt["prompt"]).token_ids]
    else:
        prompt_tokens = [
            int(token)
            for token in prompt_tokens_file.read_text(encoding="utf-8")
            .replace(",", " ")
            .split()
        ]
        if not prompt_tokens:
            raise ValueError("prompt token fixture is empty")
    if decode_tokens < 2:
        raise ValueError("decode_tokens must be at least 2")

    saved_env = {key: os.environ.get(key) for key in _ROUTE_FLAGS}
    started = time.perf_counter()
    # Provenance recorded before any arm is bound: ambient selectors cannot be
    # inferred from the report later, and the registered manifests must describe
    # the run rather than a post-capture construction.
    ambient_env = _ambient_env_snapshot()
    manifests_before = _review_manifests()
    attestation = _route_attestation()
    observed_flags: dict[str, dict[str, str | None]] = {}
    runner = Qwen35ParoNextTokenRunner(model, backend=backend)
    max_sequence = len(prompt_tokens) + decode_tokens + 8
    schedule: list[dict[str, Any]] = []
    strict_logits_by_cycle: list[np.ndarray] = []
    strict_generated: list[int] = []
    row_metrics: list[dict[str, Any]] = []
    replay_cycles: list[dict[str, Any]] = []
    decision_mismatch = None
    strict_root = -1
    candidate_root = -1
    capture_buf = None
    logits_finite_all = True
    try:
        print("[verifier-numerics] loading strict session", flush=True)
        _set_route(_STRICT_FLAGS)
        observed_flags["strict"] = _observed_route_flags()
        with Qwen35ParoResidentSession(
            runner, max_sequence_length=max_sequence, max_batch_size=2
        ) as strict_session:
            hidden = int(strict_session.config.hidden_size)
            capture_buf = malloc(
                len(prompt_tokens) * hidden * DType.BF16.itemsize,
                runtime=strict_session.runtime,
            )
            capture = Tensor.from_handle(
                capture_buf.ptr,
                (len(prompt_tokens), hidden),
                DType.BF16,
                Device("hip", 0),
            )
            no_capture = Tensor.from_handle(0, (2, 0), DType.BF16, Device("hip", 0))
            strict_root = _prefill(strict_session, prompt_tokens, capture=capture)
            scoring_head = NativeMtpW8A16Head(
                weight_int8_ptr=int(strict_session.lm_head_weight.tensor.ptr),
                scale_f32_ptr=int(strict_session.lm_head_scale.tensor.ptr),
                vocab_size=int(strict_session.vocab_size),
                threads=int(strict_session.lm_head_threads),
                owner=strict_session,
            )
            with NativeMtpChainProposer(
                model,
                max_positions=max_sequence,
                max_mtp_tokens=len(prompt_tokens) + 2 * decode_tokens + 8,
                runtime=strict_session.runtime,
                compiler_version=strict_session.compiler_version,
                scoring_head=scoring_head,
            ) as proposer:
                proposer.prefill_from_target_hidden_rows(
                    prompt_tokens,
                    capture_base_ptr=capture_buf.ptr,
                    seed_token=strict_root,
                    read_expert_topk=False,
                    read_lm_head_value=False,
                )
                context = len(prompt_tokens)
                root = strict_root
                cycle = 0
                while len(strict_generated) < decode_tokens:
                    cycle += 1
                    candidate = int(proposer.current.token)
                    batch = _target_batch(root, context, candidate)
                    strict = strict_session.verify_chain_bulk_and_commit(
                        batch,
                        base_slot=0,
                        capture_layer_ids=(),
                        capture_hidden_concat=no_capture,
                        capture_row_start=0,
                        chain_attn_mode="c1_loop",
                        graph_mode="off",
                        canonicalize_after=False,
                    )
                    strict_logits = _copy_logits(strict_session)
                    strict_logits_by_cycle.append(strict_logits)
                    strict_top1 = np.argmax(strict_logits, axis=1).astype(np.int64)
                    strict_bonus = int(
                        strict.next_token
                        if strict.next_token is not None
                        else strict_top1[strict.accepted_count]
                    )
                    committed = [root] + ([candidate] if int(strict.accepted_count) else [])
                    schedule.append(
                        {
                            "cycle": cycle,
                            "output_offset": len(strict_generated),
                            "context": context,
                            "root": root,
                            "candidate": candidate,
                            "strict_accepted": int(strict.accepted_count),
                            "strict_bonus": strict_bonus,
                            "strict_commit_row": int(strict.commit_row),
                            "strict_commit_position": int(strict.commit_position),
                            "strict_target_top1": [int(token) for token in strict_top1.tolist()],
                            "committed": committed,
                        }
                    )
                    strict_generated.extend(committed)
                    if len(strict_generated) >= decode_tokens:
                        break
                    if int(strict.accepted_count) >= 1:
                        proposer.advance_with_previous_hidden(
                            input_token=candidate,
                            position=proposer.position + 1,
                            need_result=False,
                            read_expert_topk=False,
                            read_lm_head_value=False,
                        )
                    proposer.advance_with_target_hidden(
                        input_token=strict_bonus,
                        target_hidden_ptr=int(strict.selected_target_hidden_ptr),
                        position=proposer.position + 1,
                        read_expert_topk=False,
                        read_lm_head_value=False,
                    )
                    context += len(committed)
                    root = strict_bonus
            free(capture_buf, runtime=strict_session.runtime)
            capture_buf = None
        print(
            f"[verifier-numerics] strict capture complete cycles={len(schedule)}; loading fast session",
            flush=True,
        )

        _set_route(_FAST_FLAGS)
        observed_flags.setdefault("fast", _observed_route_flags())
        with Qwen35ParoResidentSession(
            runner, max_sequence_length=max_sequence, max_batch_size=2
        ) as candidate_session:
            no_capture = Tensor.from_handle(0, (2, 0), DType.BF16, Device("hip", 0))
            candidate_root = _prefill(candidate_session, prompt_tokens)
            output_offset = 0
            for record, strict_logits in zip(schedule, strict_logits_by_cycle, strict=True):
                batch = _target_batch(
                    int(record["root"]),
                    int(record["context"]),
                    int(record["candidate"]),
                )
                fast = candidate_session.verify_chain_bulk_and_commit(
                    batch,
                    base_slot=0,
                    capture_layer_ids=(),
                    capture_hidden_concat=no_capture,
                    capture_row_start=0,
                    chain_attn_mode="decode_batched",
                    graph_mode="off",
                    canonicalize_after=False,
                )
                fast_logits = _copy_logits(candidate_session)
                fast_top1 = np.argmax(fast_logits, axis=1).astype(np.int64)
                labels = np.argmax(strict_logits, axis=1).astype(np.int64)
                metrics = per_row_metrics(strict_logits, fast_logits, labels, top_k=5)
                cycle_logits_finite = bool(
                    np.isfinite(strict_logits).all() and np.isfinite(fast_logits).all()
                )
                logits_finite_all = logits_finite_all and cycle_logits_finite
                fast_bonus = int(
                    fast.next_token
                    if fast.next_token is not None
                    else fast_top1[fast.accepted_count]
                )
                cycle_decision_mismatch = bool(
                    int(fast.accepted_count) != int(record["strict_accepted"])
                    or fast_bonus != int(record["strict_bonus"])
                )
                if decision_mismatch is None and cycle_decision_mismatch:
                    decision_mismatch = {
                        "cycle": int(record["cycle"]),
                        "output_offset": int(record["output_offset"]),
                        "strict_accepted": int(record["strict_accepted"]),
                        "candidate_accepted": int(fast.accepted_count),
                        "strict_bonus": int(record["strict_bonus"]),
                        "candidate_bonus": fast_bonus,
                    }
                for row in range(2):
                    diagnostic = _row_review_diagnostic(
                        strict_logits[row], fast_logits[row], top_k=5
                    )
                    row_metrics.append(
                        {
                            "cycle": int(record["cycle"]),
                            "output_offset": int(record["output_offset"]),
                            "row": row,
                            "row_role": "root" if row == 0 else "draft_candidate",
                            "decision_role": (
                                "draft_acceptance_or_reject_correction"
                                if row == 0
                                else "full_accept_bonus"
                            ),
                            "category": str(prompt.get("category", "unknown")),
                            "shape": "c2_b1",
                            "transition": (
                                "prefill_to_verify"
                                if int(record["cycle"]) == 1
                                else "verify_to_verify"
                            ),
                            "context": int(record["context"]),
                            "position": int(record["context"]) + row,
                            "strict_selected_for_commit": bool(
                                row == int(record["strict_commit_row"])
                            ),
                            "candidate_selected_for_commit": bool(
                                row == int(fast.commit_row)
                            ),
                            "cycle_task_decision_mismatch": cycle_decision_mismatch,
                            "strict_top1": int(record["strict_target_top1"][row]),
                            "candidate_top1": int(fast_top1[row]),
                            "kl": float(metrics["kl_nats"][row]),
                            "top1_equal": bool(metrics["top1_equal"][row]),
                            "top5_overlap": float(metrics["topk_set_overlap"][row]),
                            "max_abs_logit_delta": float(metrics["max_abs_logit_delta"][row]),
                            **diagnostic,
                        }
                    )
                if int(fast.commit_row) != int(record["strict_commit_row"]):
                    candidate_session._commit_bulk_linear_states(
                        int(record["strict_commit_row"]), base_slot=0
                    )
                    candidate_session._set_slot_position(
                        int(record["strict_commit_position"]), slot=0
                    )
                    candidate_session.runtime.device_synchronize()
                replay_cycles.append(
                    {
                        **record,
                        "candidate_accepted": int(fast.accepted_count),
                        "candidate_bonus": fast_bonus,
                        "candidate_commit_row": int(fast.commit_row),
                        "task_decision_mismatch": cycle_decision_mismatch,
                        "strict_and_candidate_logits_finite": cycle_logits_finite,
                        # Recorded, not verified here: the runner defaults these on
                        # the trusted-GPU path without reading the CPU oracle or
                        # every full-vocabulary logit.
                        "runner_reported_finite_logits": bool(fast.finite_logits),
                        "runner_reported_gpu_accept_match_cpu": bool(
                            fast.gpu_accept_match_cpu
                        ),
                    }
                )
                output_offset += len(record["committed"])
        print("[verifier-numerics] fast replay complete", flush=True)
    finally:
        if capture_buf is not None:
            free(capture_buf)
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    decision_mismatches = _decision_mismatches(replay_cycles)
    prompt_metadata = {
        "name": prompt_name,
        "category": prompt.get("category"),
        "split": prompt.get("split"),
        "render": prompt_render,
        "prompt_tokens": len(prompt_tokens),
        "prompt_tokens_sha256": _sha256_bytes(
            np.asarray(prompt_tokens, dtype=np.int64).tobytes()
        ),
        "prompts_file": str(prompts_file),
        "prompts_file_sha256": _sha256_bytes(prompts_file.read_bytes()),
        "decode_tokens": decode_tokens,
    }
    return _capture_result(
        row_metrics=row_metrics,
        replay_cycles=replay_cycles,
        decision_mismatch=decision_mismatch,
        decision_mismatches=decision_mismatches,
        manifests_before=manifests_before,
        attestation=attestation,
        observed_flags=observed_flags,
        ambient_env=ambient_env,
        prompt_metadata=prompt_metadata,
        strict_root=int(strict_root),
        candidate_root=int(candidate_root),
        logits_finite_all=logits_finite_all,
        model=str(model),
        backend=backend,
        seconds=time.perf_counter() - started,
    )


def _decision_mismatches(replay_cycles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cycle-level accept/bonus disagreements between the two arms."""

    return [
        {
            "cycle": int(cycle["cycle"]),
            "output_offset": int(cycle["output_offset"]),
            "strict_accepted": int(cycle["strict_accepted"]),
            "candidate_accepted": int(cycle["candidate_accepted"]),
            "strict_bonus": int(cycle["strict_bonus"]),
            "candidate_bonus": int(cycle["candidate_bonus"]),
        }
        for cycle in replay_cycles
        if bool(cycle["task_decision_mismatch"])
    ]


def _capture_result(
    *,
    row_metrics: Sequence[dict[str, Any]],
    replay_cycles: Sequence[dict[str, Any]],
    decision_mismatch: dict[str, Any] | None,
    decision_mismatches: Sequence[dict[str, Any]],
    manifests_before: Mapping[str, Any],
    attestation: Mapping[str, Any],
    observed_flags: Mapping[str, Mapping[str, str | None]],
    ambient_env: Mapping[str, str],
    prompt_metadata: Mapping[str, Any],
    strict_root: int,
    candidate_root: int,
    logits_finite_all: bool,
    model: str,
    backend: str,
    seconds: float,
) -> dict[str, Any]:
    """Assemble one capture's report from captured rows (no device access)."""

    _require_diagnostic_backend(backend)
    top1_mismatch_rows = [row for row in row_metrics if not row["top1_equal"]]
    arm_bound = {"strict": _STRICT_FLAGS, "fast": _FAST_FLAGS}
    observed_matches_bound = {
        arm: observed_flags.get(arm)
        == {key: arm_bound[arm].get(key) for key in _ROUTE_FLAGS}
        for arm in ("strict", "fast")
    }
    route_flags_attested = bool(attestation["matches_registered_binder"]) and all(
        observed_matches_bound.values()
    )
    attested = {
        **attestation,
        "observed_flags": dict(observed_flags),
        "observed_matches_bound": observed_matches_bound,
        "attested": route_flags_attested,
    }
    manifests = _review_manifests()
    manifests_stable = (
        manifests["registered_production_sha256"]
        == manifests_before["registered_production_sha256"]
        and manifests["registered_strict_sha256"]
        == manifests_before["registered_strict_sha256"]
    )
    report = _numerical_report(
        row_metrics,
        prefill_root_equal=int(strict_root) == int(candidate_root),
        finite_logits=logits_finite_all,
        decision_mismatches=decision_mismatches,
        route_flags_attested=route_flags_attested,
        extra_checks={"manifests_stable_during_capture": manifests_stable},
    )
    capture_sha256 = _capture_sha256(
        prompt=dict(prompt_metadata),
        candidate_manifest_sha256=manifests["capture_manifest_sha256"],
        rows=row_metrics,
        cycles=replay_cycles,
    )
    aggregate = report["aggregate"]
    return {
        "schema": "hipengine.paro_mtp_verifier_numerics.v5",
        "status": report["status"],
        "policy_decision": report["policy_decision"],
        "execution_checks": report["execution_checks"],
        "execution_failures": report["execution_failures"],
        "qualification": report["qualification"],
        "performance_claim": False,
        "model": model,
        "backend": backend,
        "backend_scope": _backend_scope(backend),
        "capture_mode": "sequential_strict_then_fast_replay",
        "capture_sha256": capture_sha256,
        "capture_binding": "registered_production_manifest_sha256",
        "manifests": manifests,
        "manifests_stable_during_capture": manifests_stable,
        "route_attestation": attested,
        "ambient_env": {
            "present": bool(ambient_env),
            "count": len(ambient_env),
            "hipengine_env_at_start": dict(ambient_env),
            "route_env_at_start": {
                name: ambient_env[name]
                for name in _ROUTE_ENV_NAMES
                if name in ambient_env
            },
            "note": (
                "Recorded before any arm was bound. The harness overwrites its own "
                "route flags; any other HIPENGINE_* value listed here was ambient "
                "for this capture and must be reviewed before the capture is "
                "compared with another run."
            ),
        },
        "runner_reported_metadata": {
            "all_cycles_logits_finite": logits_finite_all,
            "all_cycles_runner_reported_finite_logits": all(
                bool(cycle["runner_reported_finite_logits"]) for cycle in replay_cycles
            ),
            "all_cycles_runner_reported_gpu_accept_match_cpu": all(
                bool(cycle["runner_reported_gpu_accept_match_cpu"])
                for cycle in replay_cycles
            ),
            "note": (
                "The runner defaults finite_logits and gpu_accept_match_cpu on its "
                "trusted-GPU path without reading every full-vocabulary logit or the "
                "CPU accept oracle. They are recorded here, not used as evidence; "
                "this harness's own finiteness check reads the captured logits."
            ),
        },
        "candidate": {
            "source_class": "T2",
            "chain_attn_mode": "decode_batched",
            "environment": dict(_FAST_FLAGS),
            "strict_fallback": {
                "chain_attn_mode": "c1_loop",
                "environment": dict(_STRICT_FLAGS),
            },
        },
        "prompt": dict(prompt_metadata),
        "teacher_forcing": "strict-owned proposals, commit rows, tokens, positions, and contexts",
        # Legacy 2026-08-16 thresholds, retained with unchanged meaning so saved
        # captures stay comparable. `envelopes` carries the observed comparisons.
        "thresholds": dict(_THRESHOLDS),
        "thresholds_envelope": "legacy_mean_tail_kl_envelope",
        "envelopes": report["envelopes"],
        "aggregate": aggregate,
        "scopes": report["scopes"],
        "scope_failures": report["scope_failures"],
        "scope_failures_note": (
            "Binding scopes whose sample top-1 missed the 6.1 floor. On a "
            "single-prompt capture this is an observation, not a rate-gate "
            "decision; the sibling aggregation script consumes it across prompts."
        ),
        "scope_kl_advisories": report["scope_kl_advisories"],
        "sample_scope": report["sample_scope"],
        "envelope_disagreement": report["envelope_disagreement"],
        "review": {
            "binding_envelope": "current_sparse_moe_policy",
            "policy_decision": report["policy_decision"],
            "current_sample_limits_exceeded": report["envelopes"][
                "current_sparse_moe_policy"
            ]["sample_limits_exceeded"],
            "legacy_sample_limits_exceeded": report["envelopes"][
                "legacy_mean_tail_kl_envelope"
            ]["sample_limits_exceeded"],
            "legacy_envelope_thresholds_unchanged": True,
            "rate_gates_decided": report["envelopes"][
                "current_sparse_moe_policy"
            ]["rate_gate_policy"]["decided"],
            "one_mismatch_point_estimate": (
                float(aggregate["rows"] - 1) / float(aggregate["rows"])
            ),
            "top1_mismatch_rows": top1_mismatch_rows,
            "task_decision_mismatches": list(decision_mismatches),
        },
        "first_decision_mismatch": decision_mismatch,
        "first_top1_mismatch": top1_mismatch_rows[0] if top1_mismatch_rows else None,
        "rows": list(row_metrics),
        "cycles": list(replay_cycles),
        "seconds": seconds,
    }


def _summary_line(result: Mapping[str, Any]) -> dict[str, Any]:
    """The compact stdout summary the CLI prints for one capture."""

    current = result["envelopes"]["current_sparse_moe_policy"]
    legacy = result["envelopes"]["legacy_mean_tail_kl_envelope"]
    return {
        "status": result["status"],
        "policy_decision": result["policy_decision"],
        "qualification_status": result["qualification"]["status"],
        "binding_envelope": "current_sparse_moe_policy",
        "current_sample_limits_exceeded": current["sample_limits_exceeded"],
        "legacy_sample_limits_exceeded": legacy["sample_limits_exceeded"],
        "rate_gates_decided": current["rate_gate_policy"]["decided"],
        "execution_checks": result["execution_checks"],
        "execution_failures": result["execution_failures"],
        "rows": result["aggregate"]["rows"],
        "median_kl": result["aggregate"]["median_kl"],
        "p90_kl": result["aggregate"]["p90_kl"],
        "top1_agreement": result["aggregate"]["top1_agreement"],
        "large_drift_rate": result["aggregate"]["large_drift_rate"],
        "route_flags_attested": result["route_attestation"]["attested"],
        "first_decision_mismatch": result["first_decision_mismatch"],
    }


def _describe_scope() -> dict[str, Any]:
    """Machine-readable scope metadata for this diagnostic (no device access)."""

    return {
        "schema": "hipengine.paro_mtp_verifier_numerics.scope.v2",
        "diagnostic": "B1 strict-versus-fast PARO MTP target verifier numerics",
        "note": (
            "No capture was performed; row counts, observed comparisons and "
            "execution checks are reported per capture."
        ),
        "backend_scope": _backend_scope(PARO_MTP_BACKEND),
        "qualification": _qualification_scope(rows=0),
        "sample_scope": _sample_scope((), prompts=0),
        "envelopes": _envelope_metadata(),
        "manifests": _review_manifests(),
        "route_attestation": _route_attestation(),
        "exit_codes": {
            "0": (
                "capture completed with no execution failure. Overall status is "
                "'inconclusive': this one-prompt screen issues no policy decision, "
                "no rate-gate verdict and no qualification."
            ),
            "1": (
                "capture completed but an independent execution check failed "
                "(prefill-root mismatch, nonfinite logits or KL, route drift, "
                "decision inconsistency, manifest drift). Overall status is 'failed'."
            ),
            "2": "usage error.",
            "3": (
                "diagnostic scope error: --backend is not hip_gfx1100, so the "
                "registered manifest and route attestation would not describe the run."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scope: this CLI captures one prompt's B1 target-verifier trajectory and "
            "reports observed sample comparisons under two named envelopes. It is "
            "not a public adapter, rollback, asynchronous-publication, B2/B3 or "
            "quality certification, and it never decides admission: overall status "
            "is 'inconclusive' unless an independent execution check failed, in "
            "which case it is 'failed'. Exit codes: 0 inconclusive, 1 failed, "
            "2 usage error, 3 diagnostic scope error (--backend is not "
            f"{PARO_MTP_BACKEND}). --describe-scope prints the scope, envelope "
            "identities, registered manifests and route attestation without "
            "touching a GPU."
        ),
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--prompts-file", type=Path, default=Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl"))
    parser.add_argument("--prompt-name")
    parser.add_argument("--prompt-render", choices=("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on"), default="qwen_chat_thinking_off")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prompt-tokens-file", type=Path)
    parser.add_argument(
        "--backend",
        default=PARO_MTP_BACKEND,
        help=(
            "backend to run; restricted to "
            f"{PARO_MTP_BACKEND!r} by diagnostic scope"
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--describe-scope",
        action="store_true",
        help="print this diagnostic's scope, envelope identities and route attestation, then exit",
    )
    args = parser.parse_args()
    if args.describe_scope:
        print(json.dumps(_describe_scope(), indent=2, sort_keys=True))
        return 0
    missing = [
        name
        for name, value in (
            ("--model", args.model),
            ("--prompt-name", args.prompt_name),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error("capture runs require " + ", ".join(missing))
    try:
        _require_diagnostic_backend(args.backend)
    except DiagnosticScopeError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    result = run_sequential(
        model=args.model,
        prompts_file=args.prompts_file,
        prompt_name=args.prompt_name,
        prompt_render=args.prompt_render,
        decode_tokens=int(args.decode_tokens),
        backend=args.backend,
        prompt_tokens_file=args.prompt_tokens_file,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_summary_line(result), sort_keys=True))
    return 0 if result["status"] == "inconclusive" else 1


if __name__ == "__main__":
    raise SystemExit(main())
