#!/usr/bin/env python3
"""Capture crossed public PARO verifier state (diagnostic only).

At each capture checkpoint this tool restores the resident carried state and
records four kinds of comparison, all from that one state:

``verifier-carried``
    The public verifier forward versus a serial ``step()`` replay of the same
    rows. This is the only comparison that speaks to carried-state numerical
    drift, and only while the AR and MTP arms still share the token prefix.
``ar-carried``
    The same replay from the AR arm's checkpoint at that context.
``--cross-widths``
    Verifier widths 2/3/4 on their common row prefix. A width wider than the
    public cycle is padded with fixed AR-teacher tokens, and those extra rows
    are labelled diagnostic.
``--cross-profiles``
    The strict and production route flags on that same state and those rows.

Then the incoming route environment and the pre-cycle state are restored, and
the untouched public verifier call must reproduce its logits and committed state
bit for bit. That repeat is what keeps every comparison above honest.

Each capture carries the token history it actually shares. An AR checkpoint
taken after the free-running arms diverge does not share the public history, so
its logits comparison is reported as ``cross_history_carried_state``, is not a
kernel arithmetic result, and is excluded from ``numerical_rows``. Width and
profile comparisons are valid observations from the same checkpoint and forced
tokens, but their rows overlap the carried comparison and each other, so they are
reported as ``diagnostic_observations`` and never pooled with it.

Diagnostic only: no threshold, no admission, no verdict. Exit codes: 0 complete,
1 incomplete, 2 usage, 3 failed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.core.tensor import Tensor
from hipengine.speculative import MtpDraftRequest, TargetVerifyBatch, compile_mtp_chain
from hipengine.speculative.paro_mtp_profiles import (
    PARO_MTP_CHAIN_ATTN_MODE_ENV,
    PARO_MTP_CONTRACT_ENV,
    PARO_MTP_ROUTE_ENV,
)

SCHEMA = 1
KIND = "paro_crossed_state_capture"
DEFAULT_MODEL = "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"
DEFAULT_PROMPTS = Path("benchmarks/prompts/mtpbench-code-general-ja.jsonl")
DEFAULT_PROMPT = "code_merge_intervals"
DEFAULT_FOCUS = "85:92"
DEFAULT_OUTPUTS = 129
DEFAULT_MAX_SEQUENCE = 8192
AR_CONTEXT_MARGIN = 3
CROSS_WIDTHS = (2, 3, 4)
CROSS_PROFILES = ("strict", "production")
# The only comparison whose rows may be pooled as independent numerical samples.
# ``ar-carried`` is deliberately excluded: when the AR and verifier checkpoints are
# byte-identical it repeats these rows, and when they differ it still covers the
# same positions, so pooling both would double-count the same tokens.
POOLABLE_SCOPES = ("same_history_carried_state",)
HOOKS = ("_run_layers", "_sample_verify_rows_from_hidden", "verify_chain_bulk_and_commit")
PINNED_ENV = {
    "HIPENGINE_VERIFY_GPU_ACCEPT": "validate",
    "HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD": "off",
}
# Selectors the PARO MTP profile binder writes at LLM construction, after the
# declared route snapshot is taken. They have to be tracked to record the route
# the public call actually ran under.
ROUTE_BINDER_ENV = (PARO_MTP_CONTRACT_ENV, PARO_MTP_ROUTE_ENV, PARO_MTP_CHAIN_ATTN_MODE_ENV)
PROVENANCE_FILES = (
    "hipengine/generation/qwen35_paro_mtp2.py",
    "hipengine/speculative/paro_mtp_profiles.py",
    "hipengine/runtime/qwen35_paro_runner.py",
    "hipengine/speculative/mtp_native.py",
    "scripts/mtp_paro_crossed_state.py",
    "scripts/mtp_paro_verifier_numerics.py",
    "scripts/quant_quality/metrics.py",
    "scripts/specdec2_perf_gfx1100_child.py",
)
EXIT_COMPLETE, EXIT_INCOMPLETE, EXIT_FAILED = 0, 1, 3


class DiagnosticError(RuntimeError):
    """A capture could not be taken, or did not prove what it claims to."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


# --- window, parsing, and pure comparisons ---------------------------------


@dataclass(frozen=True)
class FocusWindow:
    """Inclusive token-index window, relative to the end of the prompt."""

    low: int
    high: int

    def wants_ar_state(self, index: int) -> bool:
        return index == 0 or self.low - AR_CONTEXT_MARGIN <= index <= self.high + AR_CONTEXT_MARGIN

    def wants_capture(self, index: int, rows: int) -> bool:
        return index == 0 or (index <= self.high and index + rows >= self.low)


def parse_focus(text: str) -> FocusWindow:
    parts = str(text).strip().split(":")
    if len(parts) != 2:
        raise DiagnosticError(f"--focus must be LO:HI, got {text!r}")
    try:
        low, high = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise DiagnosticError(f"--focus bounds must be integers, got {text!r}") from exc
    if low < 0 or high < 0:
        raise DiagnosticError(f"--focus bounds must be non-negative, got {text!r}")
    if high < low:
        raise DiagnosticError(f"--focus HI must be >= LO, got {text!r}")
    return FocusWindow(low=low, high=high)


def parse_choice_list(text: str, choices: Sequence[Any], flag: str) -> tuple[Any, ...]:
    """Parse a comma-separated opt-in list, refusing unknown or repeated values."""

    raw = [part.strip() for part in str(text).split(",") if part.strip()]
    if not raw:
        return ()
    try:
        values: tuple[Any, ...] = tuple(int(part) for part in raw)
    except ValueError:
        values = tuple(raw)
    unknown = [value for value in values if value not in choices]
    require(not unknown, f"{flag} accepts {list(choices)}, got {unknown}")
    require(len(set(values)) == len(values), f"{flag} repeats a value: {raw}")
    return tuple(sorted(values))


def state_digest(state: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(state[name]).tobytes())
    return digest.hexdigest()


def compare_states(expected: Mapping[str, np.ndarray],
                   observed: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compare two snapshots view by view; a differing view set is not comparable."""

    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    require(not (missing or unexpected), f"state views disagree: missing={missing} extra={unexpected}")
    unequal = [name for name in sorted(expected) if not np.array_equal(expected[name], observed[name])]
    return {"equal": not unequal, "unequal": unequal,
            "sha256_expected": state_digest(expected), "sha256_observed": state_digest(observed)}


def exact_rows(left: np.ndarray, right: np.ndarray) -> list[bool]:
    return [bool(np.array_equal(a, b)) for a, b in zip(left, right, strict=True)]


def compare_logits(reference: np.ndarray, candidate: np.ndarray, *, top_k: int = 5) -> list[dict]:
    """Per-row logit metrics and margins between two verifier-shaped matrices."""

    # Deferred so --help and the CPU test do not import the numerics harness.
    from scripts.mtp_paro_verifier_numerics import _row_review_diagnostic
    from scripts.quant_quality.metrics import per_row_metrics

    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    require(ref.ndim == 2 and cand.ndim == 2 and ref.shape == cand.shape,
            f"logit comparison needs aligned rank-2 matrices, got {ref.shape!r} and {cand.shape!r}")
    require(bool(np.isfinite(ref).all() and np.isfinite(cand).all()),
            "logit comparison requires finite logits on both sides")
    metrics = per_row_metrics(ref, cand, np.argmax(ref, axis=1).astype(np.int64), top_k=top_k)
    return [
        {
            **_row_review_diagnostic(ref_row, cand_row, top_k=top_k),
            "kl": float(metrics["kl_nats"][index]),
            "top1_equal": bool(metrics["top1_equal"][index]),
            "max_abs_logit_delta": float(metrics["max_abs_logit_delta"][index]),
            "bit_exact": bool(np.array_equal(ref_row, cand_row)),
        }
        for index, (ref_row, cand_row) in enumerate(zip(ref, cand, strict=True))
    ]


def build_token_batch(tokens: Sequence[int], context: int) -> TargetVerifyBatch:
    """Build the chain verifier batch: row 0 is the committed root, then candidates."""

    row_tokens = tuple(int(token) for token in tokens)
    require(len(row_tokens) >= 2, "a verifier row set needs a root token plus one candidate")
    require(all(token >= 0 for token in row_tokens), "verifier token ids must be non-negative")
    require(int(context) >= 0, "verifier context position must be non-negative")
    draft = compile_mtp_chain(
        [MtpDraftRequest(request_id=0, root_position=int(context),
                         candidate_tokens=row_tokens[1:], active_count=len(row_tokens) - 1)],
        candidate_budget=len(row_tokens) - 1,
    )
    return TargetVerifyBatch.from_draft(
        draft, root_tokens=(row_tokens[0],), root_positions=(int(context),)
    )


# --- device state views ----------------------------------------------------


def state_spans(session: Any, context: int) -> dict[str, tuple[int, int]]:
    """Device spans holding slot 0's carried state at ``context``.

    Linear-attention Conv/GDN state is fixed-size; full-attention K/V spans only
    the ``context`` committed cells, so that span depends on the position.
    """

    require(int(context) > 0, f"state spans need a positive context position, got {context}")
    spans: dict[str, tuple[int, int]] = {}
    for layers, slot_state, fixed in (
        (session.linear_states, session._slot_linear_state, True),
        (session.full_caches, session._slot_full_cache, False),
    ):
        for layer in sorted(layers):
            views = slot_state(layer, 0)
            for label, tensor in zip(("conv", "recurrent") if fixed else ("k", "v"), views,
                                     strict=True):
                cell = int(np.prod(tensor.shape if fixed else tensor.shape[2:]))
                size = cell * tensor.dtype.itemsize
                spans[f"{layer}.{label}"] = (int(tensor.ptr), size if fixed else size * int(context))
    return spans


def snapshot(session: Any, context: int) -> dict[str, np.ndarray]:
    """Copy the carried state at ``context`` to host memory as raw bytes."""

    session.runtime.device_synchronize()
    state: dict[str, np.ndarray] = {}
    for name, (ptr, nbytes) in state_spans(session, context).items():
        host = np.empty((nbytes,), dtype=np.uint8)
        copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), host.nbytes),
                            runtime=session.runtime)
        state[name] = host
    return state


def restore(session: Any, state: Mapping[str, np.ndarray], context: int) -> None:
    """Write a snapshot back and prove the device actually took it."""

    for name, (ptr, nbytes) in state_spans(session, context).items():
        require(name in state, f"snapshot is missing the {name} view needed at context {context}")
        value = np.ascontiguousarray(state[name])
        require(value.dtype == np.uint8, f"{name}: snapshots must be raw uint8 bytes, got {value.dtype}")
        require(value.nbytes == nbytes,
                f"{name}: snapshot holds {value.nbytes} bytes, the view at {context} needs {nbytes}")
        copy_host_to_device(DeviceBuffer(int(ptr), nbytes), host_array_ptr(value),
                            runtime=session.runtime)
    session._set_slot_position(int(context) - 1, slot=0)
    session.runtime.device_synchronize()
    comparison = compare_states(state, snapshot(session, context))
    require(comparison["equal"],
            f"restore did not land at context {context}: unequal views {comparison['unequal']}")


# --- environment and provenance --------------------------------------------


def set_env(values: Mapping[str, str | None]) -> None:
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def route_env_keys() -> tuple[str, ...]:
    """Every selector the verifier route and its profile binder can read or write."""

    return (*verifier_flags(), *ROUTE_BINDER_ENV)


def route_env_snapshot(keys: Sequence[str]) -> dict[str, str | None]:
    """The current value of every tracked selector (``None`` when unset)."""

    return {key: os.environ.get(key) for key in keys}


def apply_route_env(values: Mapping[str, str | None], keys: Sequence[str]) -> None:
    """Make ``values`` the complete route environment: a missing key is cleared."""

    set_env({key: values.get(key) for key in keys})


@contextmanager
def route_env(values: Mapping[str, str | None], keys: Sequence[str]) -> Iterator[None]:
    """Apply a complete route-env snapshot, then restore the previous values.

    Every tracked key is set or cleared, so a crossed call cannot inherit an
    ambient selector the public call did not have, and a failure inside the block
    cannot leave the route environment changed.
    """

    saved = route_env_snapshot(keys)
    try:
        apply_route_env(values, keys)
        yield
    finally:
        apply_route_env(saved, keys)


def verifier_flags(profile: str | None = None) -> Any:
    """Route-flag names, or one profile's bound values, from the numerics helper."""

    from scripts.mtp_paro_verifier_numerics import _FAST_FLAGS, _ROUTE_FLAGS, _STRICT_FLAGS

    if profile is None:
        return tuple(_ROUTE_FLAGS)
    require(profile in CROSS_PROFILES, f"unknown verifier profile {profile!r}")
    return dict(_STRICT_FLAGS if profile == "strict" else _FAST_FLAGS)


def ambient_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {name: source[name] for name in sorted(source) if name.startswith("HIPENGINE_")}


def source_hashes(sources: Mapping[str, Any], *,
                  relative_to: Path | None = None) -> dict[str, dict[str, str]]:
    """Hash every source file whose behavior the capture depends on."""

    hashes: dict[str, dict[str, str]] = {}
    for label, target in sorted(sources.items()):
        path = target if isinstance(target, Path) else Path(str(inspect.getsourcefile(target) or ""))
        require(path.is_file(), f"cannot hash {label}: {path} is not a file")
        recorded = path.relative_to(relative_to) if relative_to is not None else path
        hashes[label] = {"path": str(recorded),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return hashes


def provenance_hashes() -> dict[str, dict[str, str]]:
    """Hash the tool and every runtime file whose behavior a capture depends on.

    ``hipengine/generation/qwen35_paro_mtp2.py`` owns the verifier invocation and
    its attention-mode rule; ``hipengine/speculative/paro_mtp_profiles.py`` owns
    the profile binder that writes the route selectors.
    """

    root = Path(__file__).resolve().parents[1]
    return source_hashes({rel: root / rel for rel in PROVENANCE_FILES}, relative_to=root)


# --- history alignment -----------------------------------------------------


def first_mismatch(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate, strict=False)):
        if left != right:
            return index
    return None


def history_alignment(root_index: int, mismatch: int | None, *, known: bool = True) -> dict[str, Any]:
    """Whether the carried state at ``root_index`` shares the public token history.

    The state at ``root_index`` covers tokens ``0..root_index-1``, so the two arms
    share it exactly while they still agree at or beyond ``root_index``.
    """

    root = int(root_index)
    if not known:
        return {"aligned": None, "root_index": root, "first_mismatch": None,
                "shared_history_tokens": None, "reason": "first_mismatch_not_recorded"}
    aligned = mismatch is None or root <= int(mismatch)
    return {"aligned": aligned, "root_index": root, "first_mismatch": mismatch,
            "shared_history_tokens": root if aligned else int(mismatch),
            "reason": "shared_prefix" if aligned else "free_running_divergence"}


def annotate_alignment(report: dict[str, Any]) -> dict[str, Any]:
    """Label every comparison with the history it actually shares.

    Runs after the MTP arm because ``first_mismatch`` is only known then. Raw
    comparisons stay untouched; the label decides whether a row may be read as
    same-history numerical drift.
    """

    known = "first_mismatch" in report
    mismatch = report.get("first_mismatch")
    canonical = ar_checkpoint = cross = 0
    for capture in report.get("captures") or ():
        alignment = history_alignment(int(capture["root_index"]), mismatch, known=known)
        capture["history_alignment"] = alignment
        state = capture.get("carried_ar_vs_verifier_state")
        if isinstance(state, dict):
            state.update(scope="state_only_carried_state", numerical_pool_eligible=False,
                         expected_to_differ=alignment["aligned"] is False)
        for item in capture.get("comparisons") or ():
            if item.get("kind") == "carried":
                is_canonical = item.get("label") == "verifier-carried"
                aligned = True if is_canonical else alignment["aligned"]
                if is_canonical:
                    item["scope"] = ("same_history_carried_state" if aligned
                                     else "cross_history_carried_state" if aligned is False
                                     else "unknown_history_carried_state")
                    canonical += 1
                else:
                    ar_checkpoint += 1
                    item["scope"] = ("same_history_ar_checkpoint" if aligned
                                     else "cross_history_carried_state" if aligned is False
                                     else "unknown_history_carried_state")
                    if aligned is False:
                        cross += 1
                        item["alignment_note"] = (
                            f"the AR checkpoint at root index {alignment['root_index']} follows a "
                            f"token history that diverges from the public route at index "
                            f"{mismatch}; this comparison mixes two histories, is not "
                            "same-history numerical drift, and is not a kernel arithmetic result"
                        )
                    elif isinstance(state, dict) and state.get("equal") is True:
                        item["duplicate_of_canonical"] = True
                        item["note"] = (
                            "the AR checkpoint is byte-identical to the verifier checkpoint at this "
                            "context, so this comparison repeats the canonical rows and is not an "
                            "independent sample"
                        )
                item["history_alignment"] = alignment
            item.setdefault("scope", "same_state_variant")
            item["numerical_pool_eligible"] = item["scope"] in POOLABLE_SCOPES
    report["alignment"] = {
        "first_mismatch": mismatch,
        "first_mismatch_known": known,
        "canonical_comparisons": canonical,
        "ar_checkpoint_comparisons": ar_checkpoint,
        "cross_history_comparisons": cross,
        "canonical_scope": POOLABLE_SCOPES[0],
        "poolable_scopes": list(POOLABLE_SCOPES),
        "policy": ("numerical_rows holds only the verifier-carried comparison, the one row set "
                   "that may be pooled as independent samples; the AR-carried cross-check, the "
                   "width crosses, the profile crosses, and cross-history rows are diagnostic "
                   "observations, because they repeat or overlap the same token positions and "
                   "would double-count them"),
    }
    report["numerical_rows"] = numerical_rows(report)
    report["diagnostic_observations"] = diagnostic_observations(report)
    return report


def _comparison_rows(report: Mapping[str, Any], *, poolable: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capture in report.get("captures") or ():
        for item in capture.get("comparisons") or ():
            if item.get("skipped") or bool(item.get("numerical_pool_eligible")) != poolable:
                continue
            rows.extend(
                {"context": capture.get("context"), "root_index": capture.get("root_index"),
                 "label": item.get("label"), "scope": item.get("scope"), "row": index,
                 "kl": values["kl"], "top1_equal": values["top1_equal"],
                 "max_abs_logit_delta": values["max_abs_logit_delta"]}
                for index, values in enumerate(item.get("logits") or ())
            )
    return rows


def numerical_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Same-history carried rows: the only rows that may be pooled as samples."""

    return _comparison_rows(report, poolable=True)


def diagnostic_observations(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Width, profile, and cross-history rows, which must never be pooled.

    Width 3 and width 4 share the same prefix rows, and those rows also appear in
    the carried comparison, so counting them as independent samples would
    over-weight the same tokens. They are observations, not verdict inputs.
    """

    return _comparison_rows(report, poolable=False)


# --- report completeness ---------------------------------------------------


def mtp_accounting(output: Any) -> dict[str, Any]:
    telemetry = getattr(output, "telemetry", None)
    diagnostics = getattr(telemetry, "diagnostics", None) if telemetry is not None else None
    block = diagnostics.get("specdec2_mtp2") if isinstance(diagnostics, Mapping) else None
    return dict(block) if isinstance(block, Mapping) else {}


def capture_line(capture: Mapping[str, Any]) -> str:
    comparisons = capture.get("comparisons") or []
    carried = next((item for item in comparisons if item.get("label") == "verifier-carried"), {})
    pair = next((item for item in comparisons if item.get("kind") == "profile"), {})
    kls = lambda item: [round(float(row["kl"]), 9) for row in item.get("logits") or ()]  # noqa: E731
    state = capture.get("carried_ar_vs_verifier_state") or {}
    return (
        f"capture context={capture.get('context')} root_index={capture.get('root_index')} "
        f"tokens={capture.get('tokens')} ar_state_equal={state.get('equal')} "
        f"public_repeat_exact={capture.get('public_repeat_exact')} same_state_kl={kls(carried)} "
        f"comparisons={[item.get('label') for item in comparisons]}"
        + (f" profile_kl={kls(pair)}" if pair.get("logits") else "")
    )


def _item_problems(item: Mapping[str, Any], where: str, exists: Callable[[str], bool]) -> list[str]:
    if item.get("skipped"):
        return [f"{where}: not run ({item['skipped']})"]
    problems: list[str] = []
    if not item.get("logits"):
        problems.append(f"{where}: the logit comparison is missing")
    if not isinstance(item.get("selected_state"), Mapping) or "comparable" not in item[
        "selected_state"
    ]:
        problems.append(f"{where}: the selected-state comparison is missing")
    if item.get("kind") == "width" and int((item.get("rows") or {}).get("shared") or 0) < 2:
        problems.append(f"{where}: fewer than two rows share the public prefix")
    if not item.get("raw"):
        problems.append(f"{where}: no raw comparison artifact was written")
    elif not exists(str(item["raw"])):
        problems.append(f"{where}: raw comparison artifact {item['raw']} is missing")
    for name, arm in ((item.get("extra") or {}).get("arms") or {}).items():
        if not arm.get("vs_public_logits"):
            problems.append(f"{where}/{name}: the logit comparison is missing")
        if not arm.get("raw") or not exists(str(arm["raw"])):
            problems.append(f"{where}/{name}: the raw comparison artifact is missing")
    return problems


def _capture_problems(capture: Mapping[str, Any], exists: Callable[[str], bool],
                      widths: set[int], profiles: set[str]) -> list[str]:
    label = f"capture at context {capture.get('context')} (root index {capture.get('root_index')})"
    problems = [f"{label} is missing {key}" for key in
                ("tokens", "public_result", "public_hidden_sha256", "post_state_repeat", "comparisons")
                if key not in capture]
    if capture.get("history_alignment") is None:
        problems.append(f"{label} carries no token-history alignment label")
    if capture.get("public_repeat_exact") is not True:
        problems.append(f"{label}: repeating the untouched public verifier call did not reproduce its "
                        "logits, so the instrumentation perturbed the public call")
    repeat = capture.get("post_state_repeat") or {}
    if repeat.get("equal") is not True:
        problems.append(f"{label}: repeating the public verifier call did not reproduce the committed "
                        f"state (unequal views: {repeat.get('unequal')})")
    if capture.get("env_restored") is not True:
        problems.append(f"{label}: the incoming route environment was not restored before the replay")
    comparisons = capture.get("comparisons") or []
    if not comparisons:
        problems.append(f"{label} has no crossed replay")
    if capture.get("ar_state_available") and not any(
        item.get("label") == "ar-carried" for item in comparisons
    ):
        problems.append(f"{label} had an AR state at this context but no AR-carried replay")
    for item in comparisons:
        problems.extend(_item_problems(item, f"{label}/{item.get('label')}", exists))
        if item.get("kind") == "carried":
            absent = sorted({"serial_vs_public_logits", "same_state_hidden_exact"}
                            - set(item.get("extra") or ()))
            problems.extend(f"{label}/{item.get('label')}: {key} is missing" for key in absent)
    seen = {int(item["width"]) for item in comparisons if item.get("kind") == "width"}
    if widths - seen:
        problems.append(f"{label}: requested width(s) {sorted(widths - seen)} were silently skipped")
    seen_profiles = {name for item in comparisons if item.get("kind") == "profile"
                     for name in item.get("profiles") or ()}
    if profiles - seen_profiles:
        problems.append(f"{label}: requested profile(s) {sorted(profiles - seen_profiles)} were "
                        "silently skipped")
    return problems


def report_problems(report: Mapping[str, Any], *,
                    exists: Callable[[str], bool] | None = None) -> list[str]:
    """Every reason this report does not describe a complete capture."""

    path_exists = exists or (lambda path: Path(path).is_file())
    inputs = report.get("inputs") or {}
    outputs = int(inputs.get("outputs") or 0)
    problems = [str(item) for item in report.get("capture_problems") or ()]
    problems += [f"capture hook {failure.get('hook')!r} raised {failure.get('error')}: "
                 f"{failure.get('detail')}" for failure in report.get("hook_failures") or ()]
    if not report.get("prompt_ids"):
        problems.append("no prompt token ids were recorded")
    for arm in ("ar_ids", "mtp_ids"):
        ids = report.get(arm) or []
        if not ids:
            problems.append(f"{arm} is empty: the public call produced no tokens")
        elif outputs and len(ids) != outputs:
            problems.append(f"{arm} holds {len(ids)} tokens, the run asked for {outputs}")
    schedule = report.get("schedule") or []
    captures = report.get("captures") or []
    if not schedule:
        problems.append("the verifier hook never ran: no public verifier cycle was observed")
    if not captures:
        problems.append("no crossed-state capture was taken; widen --focus or raise --outputs")
    focus = inputs.get("focus") or []
    if len(focus) == 2 and outputs and int(focus[0]) >= outputs:
        problems.append(f"focus window starts at {focus[0]}, beyond the {outputs}-token budget, so no "
                        "capture inside it was reachable")
    widths = {int(width) for width in inputs.get("cross_widths") or ()}
    profiles = {str(name) for name in inputs.get("cross_profiles") or ()}
    for capture in captures:
        problems.extend(_capture_problems(capture, path_exists, widths, profiles))
    route = report.get("route") or {}
    if route.get("serial_decode_fallback"):
        problems.append("the public route fell back to serial decode, so the verifier cycles it "
                        "describes are not the ones the crossed replay assumes")
    if "specdec2_mtp2_used" in route and not route["specdec2_mtp2_used"]:
        problems.append("the public route did not run speculative MTP at all")
    accounting = report.get("mtp_accounting") or {}
    recoverable = accounting.get("recoverable_failures")
    if recoverable:
        problems.append(f"the public route recovered from {recoverable} failed cycle(s); a hook "
                        "exception swallowed by that recovery is indistinguishable from a clean run")
    problems += [f"recoverable cycle failure {reason.get('category')}: {reason.get('detail')}"
                 for reason in accounting.get("failure_reasons") or ()]
    cycles = accounting.get("cycles")
    if cycles is not None and int(cycles) != len(schedule):
        problems.append(f"the public accounting reports {cycles} cycles but {len(schedule)} verifier "
                        "calls were observed")
    if "first_mismatch" not in report:
        problems.append("the AR-versus-MTP first mismatch was not recorded")
    return problems


def finish_report(report: dict[str, Any], *, failed: bool = False) -> dict[str, Any]:
    problems = report_problems(report)
    report["problems"] = problems
    report["status"] = "failed" if failed else "incomplete" if problems else "complete"
    report["complete"] = report["status"] == "complete"
    return report


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


# --- capture ---------------------------------------------------------------


class CrossedStateCapture:
    """Install the session hooks and record one crossed-state report.

    Every hook body runs inside :meth:`_guard`, which records the exception in
    ``report['hook_failures']`` before re-raising it: the verifier graph path
    catches exceptions and silently re-launches the forward directly, so a hook
    failure visible only as a raised exception can be swallowed.
    """

    def __init__(self, *, session_cls: Any, focus: FocusWindow, out: Path, report: dict[str, Any],
                 declared_env: Mapping[str, str | None], widths: Sequence[int] = (),
                 profiles: Sequence[str] = (), chain_attn_mode: str = "c1_loop",
                 outputs: int = 0, max_sequence_length: int = 0) -> None:
        self._cls, self._focus, self._out, self._report = session_cls, focus, Path(out), report
        # The declared snapshot is taken before the LLM profile binder runs, so it is
        # recorded for comparison only; the authoritative route environment is read
        # at each public verifier hook entry.
        self.declared_env = dict(declared_env)
        self._keys = route_env_keys()
        self.chain_attn_mode = str(chain_attn_mode)
        self.widths, self.profiles = tuple(int(w) for w in widths), tuple(profiles)
        self._originals: dict[str, Any] = {}
        self._no_capture: Tensor | None = None
        self.prompt_length = 0
        self.ar_ids: list[int] = []
        self.max_rows = 0
        self.active = "ar"
        self.ar_states: dict[int, dict[str, np.ndarray]] = {}
        self.serial_hidden: np.ndarray | None = None
        self.serial_epoch = 0
        self.verify_hidden: np.ndarray | None = None
        self.verify_epoch = 0
        self._env_recorded = False
        self.outputs = int(outputs)
        self.max_sequence_length = int(max_sequence_length)
        self.public_rows = 0
        self._prepared_rows = 0

    def _route_env(self) -> dict[str, str | None]:
        return route_env_snapshot(self._keys)

    def _apply_route_env(self, values: Mapping[str, str | None]) -> None:
        apply_route_env(values, self._keys)

    def _capture_token(self, rows: int) -> Tensor:
        """A zero-width capture tensor that still covers ``rows`` verifier rows."""

        if self._no_capture is None or int(self._no_capture.shape[0]) < int(rows):
            self._no_capture = Tensor.from_handle(0, (int(rows), 0), DType.BF16, Device("hip", 0))
        return self._no_capture

    def _scratch_context(self, rows: int) -> int:
        """The verifier context bucket the public route reserves for ``rows`` rows."""

        bucket = self.prompt_length + self.outputs + (int(rows) - 2)
        return max(1, min(bucket, self.max_sequence_length or bucket))

    def _ensure_scratch(self, session: Any, rows: int) -> dict[str, Any] | None:
        """Reserve verifier scratch for a crossed call wider than the public cycle.

        The public route sizes the C1 verifier scratch for its own row count, so a
        wider crossed width must reserve its own before the call.
        """

        if int(rows) <= self._prepared_rows:
            return None
        prepare = getattr(session, "prepare_specdec2_verify_scratch", None)
        require(prepare is not None,
                f"the resident session cannot reserve verifier scratch for {rows} rows")
        context = self._scratch_context(rows)
        prepare(rows=int(rows), chain_attn_mode=self.chain_attn_mode, max_context_tokens=context)
        self._prepared_rows = int(rows)
        return {"rows": int(rows), "chain_attn_mode": self.chain_attn_mode,
                "max_context_tokens": int(context)}

    def _restore_scratch(self, session: Any) -> None:
        """Return the scratch to the public row count after a wider crossed call."""

        if not self.public_rows or self._prepared_rows <= self.public_rows:
            return
        prepare = getattr(session, "prepare_specdec2_verify_scratch", None)
        if prepare is None:
            return
        prepare(rows=int(self.public_rows), chain_attn_mode=self.chain_attn_mode,
                max_context_tokens=self._scratch_context(self.public_rows))
        self._prepared_rows = int(self.public_rows)

    def install(self) -> None:
        hooks = {
            "_run_layers": lambda s, *a, **k: self._layers_hook(s, *a, **k),
            "_sample_verify_rows_from_hidden": lambda s, *a, **k: self._head_hook(s, *a, **k),
            "verify_chain_bulk_and_commit": lambda s, *a, **k: self._verify_hook(s, *a, **k),
        }
        for name in HOOKS:
            require(hasattr(self._cls, name), f"{self._cls.__name__} has no {name}; the hooks moved")
            self._originals[name] = getattr(self._cls, name)
            setattr(self._cls, name, hooks[name])

    def uninstall(self) -> None:
        for name, original in self._originals.items():
            setattr(self._cls, name, original)
        self._originals.clear()

    def write(self) -> None:
        self._out.parent.mkdir(parents=True, exist_ok=True)
        self._out.write_text(json.dumps(self._report, indent=2, default=json_default) + "\n",
                             encoding="utf-8")

    @contextmanager
    def _guard(self, hook: str) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:
            self._report.setdefault("hook_failures", []).append(
                {"hook": hook, "error": f"{type(exc).__name__}: {exc}", "detail": str(exc),
                 "traceback": "".join(traceback.format_exception(
                     type(exc), exc, exc.__traceback__))[-4000:]}
            )
            with suppress(Exception):  # never replace the hook exception with a write error
                self.write()
            raise

    def _layers_hook(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        with self._guard("_run_layers"):
            position = int(kwargs["position"])
            if self.active == "ar" and self._focus.wants_ar_state(position - self.prompt_length):
                self.ar_states[position] = snapshot(session, position)
            hidden = self._originals["_run_layers"](session, *args, **kwargs)
            if self.active == "cross":
                self.serial_hidden = self._read(session, hidden.ptr,
                                                (int(session.config.hidden_size),), np.uint16)
                self.serial_epoch += 1
            return hidden

    def _head_hook(self, session: Any, hidden: Any, rows: int, **kwargs: Any) -> Any:
        with self._guard("_sample_verify_rows_from_hidden"):
            self.verify_hidden = self._read(
                session, hidden.ptr, (int(rows), int(session.config.hidden_size)), np.uint16
            )
            self.verify_epoch += 1
            return self._originals["_sample_verify_rows_from_hidden"](session, hidden, rows, **kwargs)

    def _verify_hook(self, session: Any, batch: Any, **kwargs: Any) -> Any:
        with self._guard("verify_chain_bulk_and_commit"):
            return self._verify(session, batch, **kwargs)

    @staticmethod
    def _read(session: Any, ptr: int, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        host = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(host), DeviceBuffer(int(ptr), host.nbytes),
                            runtime=session.runtime)
        return host

    def _logits(self, session: Any, rows: int) -> np.ndarray:
        return self._read(session, session.verify_lm_logits.ptr, (rows, int(session.vocab_size)),
                          np.float32)

    def _fresh(self, value: np.ndarray | None, epoch: int, previous: int, where: str,
               hook: str) -> np.ndarray:
        require(epoch > previous and value is not None,
                f"the {hook} hook did not record a hidden state for {where}")
        return value

    # -- verifier driving --------------------------------------------------

    def _run_verifier(self, session: Any, batch: Any, kwargs: Mapping[str, Any],
                      context: int) -> tuple[Any, np.ndarray]:
        """Run one verifier batch and prove its accept row and logits are usable."""

        result = self._originals["verify_chain_bulk_and_commit"](session, batch, **kwargs)
        require(result.gpu_accept_match_cpu,
                f"the GPU accept row and the CPU oracle disagree at context {context}")
        logits = self._logits(session, int(batch.rows))
        require(bool(np.isfinite(logits).all()), f"verifier logits are not finite at context {context}")
        return result, logits

    def _verifier_kwargs(self, rows: int) -> dict[str, Any]:
        """The public route's exact keyword set, with only the tested dimension changed."""

        return {
            "base_slot": 0,
            "capture_layer_ids": (),
            "capture_hidden_concat": self._capture_token(rows),
            "capture_row_start": 0,
            "chain_attn_mode": self.chain_attn_mode,
            "graph_mode": "off",
            "canonicalize_after": False,
            # The public route commits without waiting; the diagnostic reads the
            # logits and the committed state immediately, so it must synchronize.
            "synchronize_after_commit": True,
            "candidate_token_ids_i32": None,
        }

    @staticmethod
    def _invocation(kwargs: Mapping[str, Any],
                    env: Mapping[str, str | None]) -> dict[str, Any]:
        """Exactly what one diagnostic verifier call was given."""

        return {
            "base_slot": int(kwargs["base_slot"]),
            "capture_layer_ids": [int(layer) for layer in kwargs["capture_layer_ids"]],
            "capture_row_start": int(kwargs["capture_row_start"]),
            "chain_attn_mode": str(kwargs["chain_attn_mode"]),
            "graph_mode": str(kwargs["graph_mode"]),
            "canonicalize_after": bool(kwargs["canonicalize_after"]),
            "synchronize_after_commit": bool(kwargs["synchronize_after_commit"]),
            "candidate_token_ids_i32": kwargs["candidate_token_ids_i32"] is not None,
            "env": dict(env),
        }

    def _verify_once(self, session: Any, tokens: Sequence[int], context: int, *,
                     env: Mapping[str, str | None]) -> tuple[Any, Any, np.ndarray, dict, dict]:
        """Run one verifier batch on already-restored state and read its logits."""

        batch = build_token_batch(tokens, context)
        self.max_rows = max(self.max_rows, int(getattr(session, "max_batch_size", 0) or 0))
        scratch = self._ensure_scratch(session, int(batch.rows))
        kwargs = self._verifier_kwargs(int(batch.rows))
        with route_env(env, self._keys):
            result, logits = self._run_verifier(session, batch, kwargs, context)
        return batch, result, logits, kwargs, {"reserved": scratch}

    def _variant(self, session: Any, tokens: Sequence[int], context: int,
                 source: Mapping[str, np.ndarray], label: str, *,
                 env: Mapping[str, str | None]) -> dict[str, Any]:
        """Restore ``source``, run one verifier batch, and snapshot its commit."""

        restore(session, source, context)
        epoch = self.verify_epoch
        batch, result, logits, kwargs, scratch = self._verify_once(session, tokens, context, env=env)
        variant = {
            "batch": batch, "result": result, "logits": logits,
            "hidden": self._fresh(self.verify_hidden, self.verify_epoch, epoch,
                                  f"the {label} replay at context {context}",
                                  "_sample_verify_rows_from_hidden"),
            "selected": snapshot(session, int(result.commit_position) + 1),
            "invocation": {**self._invocation(kwargs, env), **scratch},
        }
        restore(session, source, context)
        return variant

    def _serial(self, session: Any, tokens: Sequence[int], context: int) -> dict[str, Any]:
        """Replay rows through the public serial path, snapshotting every row."""

        logits, hidden, states = [], [], []
        for index, token in enumerate(tokens):
            epoch = self.serial_epoch
            session.step(int(token), position=context + index, sample=True)
            logits.append(self._read(session, session.lm_logits.ptr, (int(session.vocab_size),),
                                     np.float32))
            hidden.append(self._fresh(self.serial_hidden, self.serial_epoch, epoch,
                                      f"serial row {index}", "_run_layers").copy())
            states.append(snapshot(session, context + index + 1))
        return {"logits": np.stack(logits), "hidden": np.stack(hidden), "states": states}

    def _raw(self, label: str, **arrays: Any) -> str:
        path = self._out.with_name(self._out.stem + f"-{label}.npz")
        np.savez_compressed(path, **arrays)
        return str(path)

    @staticmethod
    def _selected(reference: Mapping[str, np.ndarray] | None, observed: Mapping[str, np.ndarray],
                  row: int, limit: int, *,
                  reference_row: int | None = None) -> dict[str, Any]:
        if reference is None:
            return {"comparable": False, "row": int(row),
                    "reason": f"commit row {row} is beyond the {limit} rows the reference owns"}
        if reference_row is not None and int(reference_row) != int(row):
            return {"comparable": False, "row": int(row), "reference_row": int(reference_row),
                    "reason": "the two runs committed different rows, so their committed states "
                              "sit at different positions and cannot be compared"}
        return {"comparable": True, "row": int(row), **compare_states(reference, observed)}

    # -- capture -----------------------------------------------------------

    def _verify(self, session: Any, batch: Any, **kwargs: Any) -> Any:
        # The route environment the public call actually runs under is only settled
        # once the LLM profile binder has written its selectors, so it is read here
        # rather than at tool start.
        incoming = self._route_env()
        # Profile guesses cannot override an explicit public verifier invocation.
        self.chain_attn_mode = str(kwargs.get("chain_attn_mode", "c1_loop"))
        if not self._env_recorded:
            self._env_recorded = True
            env_block = self._report.setdefault("env", {})
            env_block["route_flags_at_public_call"] = dict(incoming)
            env_block["route_flags_declared_before_binder"] = dict(self.declared_env)
        context = int(batch.positions[0])
        index = context - self.prompt_length
        rows = int(batch.rows)
        # The public route reserved its own verifier scratch for exactly these rows.
        self.public_rows = rows
        self._prepared_rows = max(self._prepared_rows, rows)
        actual = batch
        candidates = kwargs.get("candidate_token_ids_i32")
        if candidates is not None:
            ids = self._read(session, candidates.ptr, (len(batch.tokens) - 1,), np.int32).tolist()
            actual = replace(batch, tokens=(batch.tokens[0], *ids))
        capture_here = self._focus.wants_capture(index, rows)
        before = snapshot(session, context) if capture_here else None
        epoch = self.verify_epoch
        result, logits = self._run_verifier(session, batch, kwargs, context)
        self._report["schedule"].append(
            {"context": context, "root_index": index, "tokens": list(actual.tokens), "rows": rows,
             "accepted": int(result.accepted_count), "commit_row": int(result.commit_row),
             "next_token": result.next_token, "target_top1": list(result.target_top1)}
        )
        if before is None:
            return result
        public_hidden = self._fresh(self.verify_hidden, self.verify_epoch, epoch,
                                    f"the cycle at context {context}",
                                    "_sample_verify_rows_from_hidden").copy()
        return self._capture(session, batch, actual, kwargs, result, logits, public_hidden,
                             before, context, index, incoming)

    def _capture(self, session: Any, batch: Any, actual: Any, kwargs: Mapping[str, Any],
                 result: Any, logits: np.ndarray, public_hidden: np.ndarray,
                 before: Mapping[str, np.ndarray], context: int, index: int,
                 incoming: Mapping[str, str | None]) -> Any:
        tokens = tuple(actual.tokens)
        post = snapshot(session, int(result.commit_position) + 1)
        public = {"logits": logits, "hidden": public_hidden, "post": post, "result": result,
                  "rows": int(batch.rows)}
        capture: dict[str, Any] = {
            "context": context, "root_index": index, "tokens": list(tokens),
            "rows": int(batch.rows), "public_result": result.to_json_dict(),
            "public_hidden_sha256": hashlib.sha256(public_hidden.tobytes()).hexdigest(),
            "ar_state_available": context in self.ar_states, "comparisons": [],
            "env_incoming": dict(incoming),
            "public_invocation": self._invocation({
                **self._verifier_kwargs(int(batch.rows)),
                "canonicalize_after": True,
                **kwargs,
            }, incoming),
        }
        if context in self.ar_states:
            capture["carried_ar_vs_verifier_state"] = compare_states(self.ar_states[context], before)
        self.active = "cross"
        restore(session, before, context)
        public["serial"] = self._serial(session, tokens, context)
        capture["comparisons"].append(
            self._carried(session, tokens, context, public, before, "verifier-carried",
                          incoming=incoming))
        if context in self.ar_states:
            capture["comparisons"].append(
                self._carried(session, tokens, context, public, self.ar_states[context],
                              "ar-carried", incoming=incoming))
        for width in self.widths:
            capture["comparisons"].append(
                self._width(session, tokens, context, public, before, width, incoming=incoming))
        if self.profiles:
            capture["comparisons"].append(
                self._profile(session, tokens, context, public, before, incoming=incoming))

        # The public replay runs under exactly the incoming route environment, and a
        # failure inside it must not leave the route environment changed.
        self._restore_scratch(session)
        restore(session, before, context)
        self._apply_route_env(incoming)
        try:
            replay = self._originals["verify_chain_bulk_and_commit"](session, batch, **kwargs)
            require(replay.gpu_accept_match_cpu,
                    f"the repeated public verifier call disagrees with the CPU oracle at context "
                    f"{context}")
            capture["public_repeat_exact"] = bool(
                np.array_equal(logits, self._logits(session, int(batch.rows))))
            capture["post_state_repeat"] = compare_states(
                post, snapshot(session, int(replay.commit_position) + 1))
            # Measured after the replay: neither the crossed calls nor the public
            # call may leave a route selector behind.
            capture["env_restored"] = self._route_env() == {key: incoming.get(key)
                                                            for key in self._keys}
        finally:
            self._apply_route_env(incoming)
        self._report["captures"].append(capture)
        self.write()
        print(capture_line(capture), flush=True)
        require(capture["public_repeat_exact"] and capture["post_state_repeat"]["equal"],
                f"instrumentation changed the public call at context {context}: "
                f"logits_exact={capture['public_repeat_exact']} "
                f"state_equal={capture['post_state_repeat']['equal']}")
        self.active = "mtp"
        return replay

    def _carried(self, session: Any, tokens: Sequence[int], context: int,
                 public: Mapping[str, Any], source: Mapping[str, np.ndarray], label: str, *,
                 incoming: Mapping[str, str | None]) -> dict:
        restore(session, source, context)
        serial = self._serial(session, tokens, context)
        variant = self._variant(session, tokens, context, source, label, env=incoming)
        commit_row = int(variant["result"].commit_row)
        return {
            "kind": "carried", "label": label,
            "rows": {"batch": len(tokens), "shared": len(tokens)},
            "logits": compare_logits(serial["logits"], variant["logits"]),
            "selected_state": self._selected(serial["states"][commit_row], variant["selected"],
                                             commit_row, len(tokens)),
            "result": variant["result"].to_json_dict(),
            "invocation": variant["invocation"],
            "extra": {
                "serial_vs_public_logits": compare_logits(serial["logits"], public["logits"]),
                "serial_vs_public_hidden_exact": exact_rows(serial["hidden"], public["hidden"]),
                "same_state_hidden_exact": exact_rows(serial["hidden"], variant["hidden"]),
                "commit_row_agrees_with_public": commit_row == int(public["result"].commit_row),
            },
            "raw": self._raw(
                f"{context}-{label}",
                serial_logits=serial["logits"], verifier_logits=variant["logits"],
                public_logits=public["logits"], serial_hidden=serial["hidden"],
                verifier_hidden=variant["hidden"], public_hidden=public["hidden"],
            ),
        }

    def _skip(self, label: str, width: int, reason: str, context: int) -> dict[str, Any]:
        print(f"{reason}; not run at context {context}", file=sys.stderr)
        return {"kind": "width", "label": label, "width": width, "skipped": reason}

    def _width(self, session: Any, tokens: Sequence[int], context: int, public: Mapping[str, Any],
               before: Mapping[str, np.ndarray], width: int, *,
               incoming: Mapping[str, str | None]) -> dict[str, Any]:
        label = f"width{width}"
        shared = min(width, len(tokens))
        if width > self.max_rows:
            return self._skip(label, width, f"width {width} exceeds the resident verifier row "
                                            f"capacity {self.max_rows}", context)
        extra: tuple[int, ...] = ()
        indices: list[int] = []
        if width > shared:
            first = context + shared - self.prompt_length
            extra = tuple(int(token) for token in self.ar_ids[first:first + (width - shared)])
            if len(extra) != width - shared:
                return self._skip(label, width,
                                  f"width {width} needs {width - shared} fixed teacher token(s) from "
                                  f"index {first}; the AR arm recorded {len(self.ar_ids)} tokens",
                                  context)
            indices = list(range(first, first + len(extra)))
        variant = self._variant(session, tuple(tokens[:shared]) + extra, context, before, label,
                                env=incoming)
        commit_row = int(variant["result"].commit_row)
        return {
            "kind": "width", "label": label, "width": width,
            "scope": "same_state_shared_prefix",
            "rows": {"batch": int(variant["batch"].rows), "shared": shared},
            "logits": compare_logits(variant["logits"][:shared], public["logits"][:shared]),
            "selected_state": self._selected(
                public["serial"]["states"][commit_row] if commit_row < shared else None,
                variant["selected"], commit_row, shared),
            "invocation": variant["invocation"],
            "suffix": None if not extra else {
                "tokens": list(extra), "indices": indices, "source": "ar_greedy_teacher",
                "diagnostic_only": True,
                "note": "these rows have no public reference; their logits are in the raw artifact only",
            },
            "result": variant["result"].to_json_dict(),
            "raw": self._raw(f"{context}-{label}", verifier_logits=variant["logits"],
                             public_logits=public["logits"]),
        }

    def _profile(self, session: Any, tokens: Sequence[int], context: int,
                 public: Mapping[str, Any], before: Mapping[str, np.ndarray], *,
                 incoming: Mapping[str, str | None]) -> dict[str, Any]:
        arms: dict[str, Any] = {}
        logits: dict[str, np.ndarray] = {}
        selected: dict[str, Mapping[str, np.ndarray]] = {}
        commit_rows: dict[str, int] = {}
        public_row = int(public["result"].commit_row)
        for name in self.profiles:
            flags = verifier_flags(name)
            # Only the profile dimension changes; every other route selector, and the
            # verifier attention mode, stay exactly as the public call had them.
            variant = self._variant(session, tokens, context, before, f"profile:{name}",
                                    env={**incoming, **flags})
            logits[name] = variant["logits"]
            selected[name] = variant["selected"]
            commit_rows[name] = int(variant["result"].commit_row)
            arms[name] = {
                "flags": flags,
                "result": variant["result"].to_json_dict(),
                "invocation": variant["invocation"],
                "vs_public_logits": compare_logits(variant["logits"], public["logits"]),
                "selected_state": self._selected(
                    public["post"], variant["selected"], commit_rows[name], len(tokens),
                    reference_row=public_row),
                "raw": self._raw(f"{context}-profile-{name}", verifier_logits=variant["logits"],
                                 public_logits=public["logits"]),
            }
        pair = (compare_logits(logits["strict"], logits["production"])
                if "strict" in logits and "production" in logits else None)
        if pair is not None:
            # Logits are comparable across a different commit row; committed states
            # are only comparable when both arms committed the same row.
            selected_state = self._selected(selected["strict"], selected["production"],
                                            commit_rows["production"], len(tokens),
                                            reference_row=commit_rows["strict"])
        else:
            selected_state = arms[self.profiles[0]]["selected_state"]
        return {
            "kind": "profile",
            "label": "strict-vs-production" if pair is not None else f"profile:{self.profiles[0]}",
            "profiles": list(self.profiles),
            "scope": "same_state_same_rows_env_flags",
            "rows": {"batch": len(tokens), "shared": len(tokens)},
            "logits": pair if pair is not None else arms[self.profiles[0]]["vs_public_logits"],
            "selected_state": selected_state,
            "invocation": arms[self.profiles[0]]["invocation"],
            "raw": self._raw(f"{context}-profiles", public_logits=public["logits"],
                             **{f"{name}_logits": value for name, value in logits.items()}),
            "extra": {"arms": arms},
            "limits": ("only the profile env flags differ per arm; every other route selector, the "
                       "verifier attention mode, and the row set are held at the public route's "
                       "values. Logits stay comparable across arms; committed states are compared "
                       "only when both arms committed the same row."),
        }


# --- command line ----------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Diagnostic only: the report carries observations, not thresholds, and no run of "
                "this tool admits or rejects a model, artifact, or quantization."),
    )
    parser.add_argument("--out", required=True, help="report path (raw .npz files land beside it)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="resident PARO MTP model directory")
    parser.add_argument("--backend", default="hip_gfx1100", help="kernel backend name")
    parser.add_argument("--profile", default="strict", help="execution profile of the public call")
    parser.add_argument("--budget", type=int, default=3, help="public candidate budget (rows = +1)")
    parser.add_argument("--prompts-file", default=str(DEFAULT_PROMPTS), help="prompt suite to read")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt id inside the suite")
    parser.add_argument("--focus", default=DEFAULT_FOCUS,
                        help="inclusive LO:HI token-index window relative to the prompt")
    parser.add_argument("--outputs", type=int, default=DEFAULT_OUTPUTS, help="generated tokens")
    parser.add_argument("--pad", type=int, default=0, help="prepend this many filler tokens")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE,
                        help="resident sequence capacity; must cover prompt plus --outputs")
    parser.add_argument("--cross-widths", default="",
                        help="opt-in verifier widths to compare from one state, e.g. 2,3,4")
    parser.add_argument("--cross-profiles", default="",
                        help="opt-in verifier flag profiles to compare, e.g. strict,production")
    args = parser.parse_args(argv)
    for name, minimum in (("budget", 1), ("outputs", 1), ("pad", 0), ("max_sequence_length", 2)):
        if getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    try:
        args.focus_window = parse_focus(args.focus)
        args.cross_widths = parse_choice_list(args.cross_widths, CROSS_WIDTHS, "--cross-widths")
        args.cross_profiles = parse_choice_list(args.cross_profiles, CROSS_PROFILES, "--cross-profiles")
    except DiagnosticError as exc:
        parser.error(str(exc))
    return args


def _inputs(args: argparse.Namespace, focus: FocusWindow, widths: Sequence[int],
            profiles: Sequence[str], out: Path, prompts_file: Path) -> dict[str, Any]:
    keys = ("model", "backend", "profile", "budget", "prompt", "outputs", "pad",
            "max_sequence_length")
    return {**{key: getattr(args, key) for key in keys}, "prompts_file": str(prompts_file),
            "focus": [focus.low, focus.high], "cross_widths": list(widths),
            "cross_profiles": list(profiles), "out": str(out)}


def teacher_output_budget(outputs: int, widths: Sequence[int]) -> int:
    """Reserve fixed AR teacher rows beyond the public output-tail boundary."""

    return int(outputs) + max(widths, default=1) - 1


def reserved_candidate_budget(budget: int, widths: Sequence[int]) -> int:
    """Candidates to reserve so the widest crossed verifier fits: rows = budget + 1."""

    return max(int(budget), max(widths, default=1) - 1)


def route_chain_attn_mode(execution_profile: str) -> str:
    """The verifier attention mode the public MTP2 adapter binds for this profile.

    ``hipengine/generation/qwen35_paro_mtp2.py`` selects ``c1_loop`` for the strict
    profile and ``decode_batched`` otherwise; the profile binder publishes the same
    choice as ``HIPENGINE_MTP_CHAIN_ATTN_MODE``.
    """

    return "c1_loop" if str(execution_profile) == "strict" else "decode_batched"


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    """Run both public arms and return the finished report."""

    focus = getattr(args, "focus_window", None) or parse_focus(args.focus)
    widths = tuple(getattr(args, "cross_widths", ()) or ())
    profiles = tuple(getattr(args, "cross_profiles", ()) or ())
    out = Path(args.out).expanduser()
    prompts_file = Path(args.prompts_file).expanduser()
    require(prompts_file.is_file(), f"prompt suite {prompts_file} does not exist")
    require(Path(args.model).exists(), f"model directory {args.model} does not exist")
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(PINNED_ENV)

    # Deferred: --help, argument validation, and the CPU test avoid the serving stack.
    from hipengine import LLM, SamplingParams
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
    from hipengine.speculative import paro_mtp_profiles_registered
    from scripts.gguf_mtp_category_bench import load_prompt_rows
    from scripts.specdec2_perf_gfx1100_child import _latest_route_metadata

    # Reserve the candidates the widest crossed verifier needs (rows = budget + 1),
    # then narrow the public request to the CLI budget, so the width crosses cannot
    # be refused by resident row capacity.
    reserved = reserved_candidate_budget(args.budget, widths)
    declared_env = route_env_snapshot(route_env_keys())
    report: dict[str, Any] = {
        "schema": SCHEMA, "kind": KIND, "diagnostic_only": True, "performance_claim": False,
        "status": "running", "complete": False, "problems": [], "hook_failures": [],
        "inputs": _inputs(args, focus, widths, profiles, out, prompts_file),
        "capacity": {"reserved_candidate_budget": reserved,
                     "public_candidate_budget": args.budget,
                     "public_verifier_rows": args.budget + 1,
                     "widest_crossed_rows": max(widths, default=1),
                     "ar_teacher_outputs": teacher_output_budget(args.outputs, widths),
                     "resident_max_rows": None},
        "env": {"pinned": dict(PINNED_ENV),
                "route_flags_declared_before_binder": dict(declared_env),
                "route_flags_at_public_call": None,
                "ambient": ambient_env()},
        "source_hashes": provenance_hashes(),
        "schedule": [], "captures": [],
    }
    capture = CrossedStateCapture(
        session_cls=Qwen35ParoResidentSession, focus=focus, out=out, report=report,
        declared_env=declared_env, widths=widths, profiles=profiles,
        chain_attn_mode=route_chain_attn_mode(args.profile), outputs=args.outputs,
        max_sequence_length=args.max_sequence_length)
    llm = None
    try:
        capture.install()
        llm = LLM(args.model, backend=args.backend, execution_profile=args.profile,
                  max_active_requests=1, max_sequence_length=args.max_sequence_length,
                  speculative_candidate_budget=reserved)
        llm.prepare(max_sequence_length=args.max_sequence_length)
        # The binder runs during construction, so the mode is re-derived from the
        # profile the live object actually carries.
        observed_profile = str(getattr(llm, "execution_profile", None)
                               or getattr(getattr(llm, "generator", None),
                                          "execution_profile", None)
                               or args.profile)
        capture.chain_attn_mode = route_chain_attn_mode(observed_profile)
        report["invocation"] = {
            "execution_profile": observed_profile,
            "chain_attn_mode": capture.chain_attn_mode,
            "chain_attn_mode_source": ("hipengine/generation/qwen35_paro_mtp2.py verifier rule "
                                       "(strict -> c1_loop, otherwise decode_batched)"),
            "base_slot": 0, "capture_layer_ids": [], "capture_row_start": 0,
            "graph_mode": "off", "canonicalize_after": False,
            "synchronize_after_commit": True,
            "public_synchronize_after_commit": False,
            "note": ("crossed calls reuse the public keyword set and change only the tested "
                     "dimension; they synchronize because the logits are read immediately"),
        }
        report["profile_binding"] = {
            "execution_profile": observed_profile,
            "profiles_registered": bool(paro_mtp_profiles_registered()),
            "binder_env": {key: os.environ.get(key) for key in ROUTE_BINDER_ENV},
        }
        llm.speculative_candidate_budget_requested = args.budget
        llm.speculative_candidate_budget = args.budget
        rows = load_prompt_rows(prompts_file)
        case = next((row for row in rows if row.get("id") == args.prompt), None)
        require(case is not None,
                f"prompt {args.prompt!r} is not in {prompts_file}; known ids: "
                f"{sorted(str(row.get('id')) for row in rows)}")
        prompt_ids = list(llm.tokenize(case["prompt"]))
        if args.pad:
            filler = list(llm.tokenize("This is background context for the task. "))
            needed = max(0, args.pad - len(prompt_ids))
            require(bool(filler), "--pad needs a tokenizer that produces at least one token")
            prompt_ids = (filler * ((needed + len(filler) - 1) // len(filler)))[:needed] + prompt_ids
        teacher_budget = teacher_output_budget(args.outputs, widths)
        require(len(prompt_ids) + teacher_budget <= args.max_sequence_length,
                f"prompt ({len(prompt_ids)} tokens) plus AR teacher budget ({teacher_budget}) exceeds "
                f"--max-sequence-length ({args.max_sequence_length}); width crosses need "
                "fixed teacher tokens beyond the public output tail")
        report["prompt_ids"] = prompt_ids
        capture.prompt_length = len(prompt_ids)
        params = SamplingParams(max_tokens=args.outputs, temperature=0.0, ignore_eos=True)

        capture.active = "ar"
        teacher_params = SamplingParams(max_tokens=teacher_budget, temperature=0.0, ignore_eos=True)
        autoregressive = llm.generate_detailed((tuple(prompt_ids),), teacher_params)[0]
        capture.ar_ids = list(autoregressive.generated_token_ids or ())
        report["ar_teacher_ids"] = list(capture.ar_ids)
        report["ar_ids"] = capture.ar_ids[:args.outputs]
        require(len(capture.ar_ids) == teacher_budget,
                f"AR teacher produced {len(capture.ar_ids)} tokens, expected {teacher_budget}")
        print(f"AR arm captured {len(capture.ar_states)} state snapshots", flush=True)

        capture.active = "mtp"
        speculative = llm.generate_speculative_mtp_detailed((tuple(prompt_ids),), params)[0]
        report["mtp_ids"] = list(speculative.generated_token_ids or ())
        report["first_mismatch"] = first_mismatch(report["ar_ids"], report["mtp_ids"])
        report["route"] = _latest_route_metadata(llm.live_loop_snapshot())
        report["mtp_accounting"] = mtp_accounting(speculative)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": "".join(traceback.format_exception(
                               type(exc), exc, exc.__traceback__))}
        raise
    finally:
        capture.uninstall()
        if llm is not None:
            with suppress(Exception):
                llm.close()
        report["capacity"]["resident_max_rows"] = capture.max_rows or None
        annotate_alignment(report)
        finish_report(report, failed="error" in report)
        capture.write()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_capture(args)
    except DiagnosticError as exc:
        print(f"crossed-state capture failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # noqa: BLE001 - reported, then mapped to an exit code
        traceback.print_exc()
        print(f"crossed-state capture failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    rows = len(report["numerical_rows"])
    observations = len(report.get("diagnostic_observations") or ())
    if report["status"] == "complete":
        print(f"complete: {len(report['captures'])} captures, {rows} poolable rows, "
              f"{observations} diagnostic observations -> {args.out}", flush=True)
        return EXIT_COMPLETE
    print(f"{report['status']}: {args.out} does not describe a complete capture", file=sys.stderr)
    for problem in report["problems"]:
        print(f"  - {problem}", file=sys.stderr)
    return EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
