#!/usr/bin/env python3
"""Common true-AR / legacy-native / staged SPECDEC2 performance bridge.

The bridge is the shared P1 measurement owner from ``docs/SPECDEC2-PERF.md``
and ``docs/SPECDEC2-PERF-GFX1100.md``. It loads one backend-qualified dense
service per candidate budget, runs true AR and Generation-2 SPECDEC2 under one
request boundary, and keeps the current-source direct dense
MTP route as a C1 execution-efficiency control.  Dense direct C>1 is
request-serial, so those legacy cells are explicitly skipped rather than being
misreported as physical evidence.

The script writes an atomic checkpoint after every arm.  ``--profile-child`` is
the final leaf mode for rocprofv3: one C/K/prompt/repetition, cached builds, and
ROCTX stage ranges.  Profile this child directly; never profile a parent that
launches it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.provenance import (  # noqa: E402
    collect_artifact_provenance,
    collect_repo_state,
)
from hipengine.core.memory import memory_stats  # noqa: E402
from hipengine.generation.registry import GenerationRequest  # noqa: E402
from hipengine.speculative.serving import (  # noqa: E402
    SpeculativeMTPStaticEligibility,
    SpeculativeMTPStaticState,
)
from hipengine.kernels.backends import HIP_BACKEND_TARGET_ARCH  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
_HELDOUT_IDS = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)
_REQUIRED_CATEGORIES = frozenset({"code", "general_en", "general_ja", "mixed_ja_en"})
FULL_PROMPT_IDS = (
    "code_merge_intervals",
    "code_topological_sort",
    "code_lru_cache",
    "code_markdown_table",
    "general_en_plan",
    "general_en_explain",
    "general_ja_plan",
    "general_ja_explain",
    "mixed_ja_en_translate",
    "mixed_ja_en_review",
)
ARMS = ("true_ar", "legacy_native", "specdec2")
ROCTX_PREFIX = "specdec2_perf_"
_ALLOWED_UNTRACKED_ROOT = "benchmarks/results/"


class BridgeContractError(ValueError):
    """Raised when an artifact cannot support the common bridge claim."""


def _render_prompt_suite_messages(messages: object) -> str:
    """Render the canonical simple-message fixture without server extras."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt messages must be a non-empty list")
    rendered: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"prompt message {index} must be a mapping")
        role = str(message.get("role", "")).strip()
        content = message.get("content")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"prompt message {index} has unsupported role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"prompt message {index} content must be text")
        rendered_role = "system" if role == "developer" else role
        rendered.append(f"<|im_start|>{rendered_role}\n{content}<|im_end|>")
    rendered.append("<|im_start|>assistant\n")
    return "\n".join(rendered)


def load_prompt_suite(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        prompt_id = payload.get("id")
        category = payload.get("category")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt line {line_number} has no strict id")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"prompt {prompt_id} has no strict category")
        rendered = _render_prompt_suite_messages(payload.get("messages"))
        rows.append(
            {
                "id": prompt_id,
                "category": category,
                "messages": payload["messages"],
                "rendered_prompt": rendered,
                "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            }
        )
    ids = tuple(row["id"] for row in rows)
    if ids != FULL_PROMPT_IDS:
        raise ValueError("canonical SPECDEC2 prompt IDs/order are incomplete")
    if {row["category"] for row in rows} != _REQUIRED_CATEGORIES:
        raise ValueError("canonical SPECDEC2 categories are incomplete")
    return tuple(rows)


def _parse_unique_ints(value: str, *, allowed: frozenset[int], label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a comma-separated integer list") from exc
    if not parsed or any(item not in allowed for item in parsed):
        supported = ",".join(str(item) for item in sorted(allowed))
        raise ValueError(f"{label} must be a comma-separated subset of {supported}")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{label} must not contain duplicate values")
    return parsed


def parse_concurrencies(value: str) -> tuple[int, ...]:
    return _parse_unique_ints(
        value,
        allowed=frozenset(range(1, 9)),
        label="concurrency",
    )


def parse_budgets(value: str) -> tuple[int, ...]:
    return _parse_unique_ints(
        value,
        allowed=frozenset({1, 2, 3}),
        label="candidate budget",
    )


def bridge_service_capacity(
    concurrencies: Sequence[int],
    *,
    requested_capacity: int | None = None,
) -> int:
    """Return one honest service capacity or require a separate C1 packet.

    A focused physical-width profile may keep the resident owner at a larger
    qualified capacity (for example realized C6/C7 under the capacity-8 owner).
    The requested capacity can never be smaller than the measured width.
    """

    selected = tuple(int(value) for value in concurrencies)
    supported = frozenset(range(1, 9))
    if not selected or any(value not in supported for value in selected):
        raise ValueError("bridge concurrency must be a non-empty subset of 1..8")
    if 1 in selected and any(value > 1 for value in selected):
        raise ValueError(
            "strict C1 and physical C2-C8 require separate bridge invocations "
            "with independent service capacities"
        )
    capacity = max(selected) if requested_capacity is None else int(requested_capacity)
    if capacity not in supported:
        raise ValueError("service capacity must be in 1..8")
    if capacity < max(selected):
        raise ValueError("service capacity is smaller than realized concurrency")
    return capacity


def resolve_platform(
    *,
    backend: str,
    target_arch: str | None,
    quant_label: str,
    gpu_max_hw_queues: int | None,
    environ: Mapping[str, str],
) -> dict[str, str | None]:
    """Resolve backend-local evidence identity without cross-lane defaults."""

    selected_backend = str(backend)
    expected_arch = HIP_BACKEND_TARGET_ARCH.get(selected_backend)
    if expected_arch is None:
        raise ValueError(f"unsupported HIP bridge backend: {selected_backend!r}")
    selected_arch = expected_arch if target_arch is None else str(target_arch)
    if selected_arch != expected_arch:
        raise ValueError(
            f"target arch {selected_arch!r} does not match backend "
            f"{selected_backend!r} ({expected_arch!r})"
        )
    selected_quant = str(quant_label).strip()
    if not selected_quant:
        raise ValueError("quant label must be non-empty")
    if gpu_max_hw_queues is not None:
        if int(gpu_max_hw_queues) <= 0:
            raise ValueError("gpu_max_hw_queues must be positive when set")
        queue = str(int(gpu_max_hw_queues))
        queue_source = "explicit_cli"
    elif str(environ.get("GPU_MAX_HW_QUEUES", "")).strip():
        queue = str(environ["GPU_MAX_HW_QUEUES"]).strip()
        queue_source = "environment"
    elif selected_backend == "hip_gfx1151":
        queue = "2"
        queue_source = "gfx1151_campaign_default"
    else:
        queue = None
        queue_source = "unset"
    return {
        "backend": selected_backend,
        "target_arch": selected_arch,
        "quant_label": selected_quant,
        "gpu_max_hw_queues": queue,
        "queue_source": queue_source,
    }


def arm_order(prompt_index: int) -> tuple[str, str, str]:
    """Counterbalance true AR and staged SPECDEC2 using only row index."""

    return ARMS if int(prompt_index) % 2 == 0 else tuple(reversed(ARMS))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _float_mapping(value: object, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise BridgeContractError(f"{label} must be a timing mapping")
    result: dict[str, float] = {}
    for key, raw in value.items():
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise BridgeContractError(f"{label}.{key} must be finite and non-negative")
        result[str(key)] = number
    return result


def normalize_timing_payloads(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate ownership and sum each physical/request timing exactly once."""

    if not rows:
        raise BridgeContractError("arm has no timing payloads")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        scope = str(row.get("timing_scope") or "choice")
        if scope not in {"choice", "request", "batch", "client"}:
            raise BridgeContractError(f"timing payload {index} has unsupported scope {scope!r}")
        if "timing_owner" not in row or not isinstance(row["timing_owner"], bool):
            raise BridgeContractError(f"timing payload {index} must declare boolean timing_owner")
        group_rows = int(row.get("group_rows", 1))
        if group_rows <= 0:
            raise BridgeContractError(f"timing payload {index} has invalid group_rows")
        batch_id = row.get("batch_id")
        if scope == "batch" and (batch_id is None or not str(batch_id).strip()):
            raise BridgeContractError(f"timing payload {index} has batch scope without batch_id")
        normalized.append(
            {
                "scope": scope,
                "batch_id": None if batch_id is None else str(batch_id),
                "group_rows": group_rows,
                "owner": bool(row["timing_owner"]),
                "timing": _float_mapping(row.get("timing"), label=f"timing payload {index}"),
            }
        )

    owners: list[dict[str, Any]] = []
    ignored = 0
    batch_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        if row["scope"] == "batch":
            batch_groups[str(row["batch_id"])].append(row)
        elif row["owner"]:
            owners.append(row)
        else:
            raise BridgeContractError("non-batch timing payload cannot be a non-owner")
    for batch_id, batch_rows in batch_groups.items():
        widths = {int(row["group_rows"]) for row in batch_rows}
        if len(widths) != 1:
            raise BridgeContractError(f"batch {batch_id!r} has inconsistent group_rows")
        width = next(iter(widths))
        if len(batch_rows) != width:
            raise BridgeContractError(
                f"batch {batch_id!r} expected {width} timing payloads; found {len(batch_rows)}"
            )
        batch_owners = [row for row in batch_rows if row["owner"]]
        if len(batch_owners) != 1:
            raise BridgeContractError(
                f"batch {batch_id!r} requires exactly one timing owner; found {len(batch_owners)}"
            )
        owners.extend(batch_owners)
        ignored += len(batch_rows) - 1

    totals: dict[str, float] = defaultdict(float)
    for row in owners:
        for key, value in row["timing"].items():
            totals[key] += float(value)
    return {
        "owners": len(owners),
        "ignored_nonowners": ignored,
        "owned_totals_ms": dict(sorted(totals.items())),
        "batch_ids": sorted(batch_groups),
    }


class _Roctx:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._push = None
        self._pop = None
        if not self.enabled:
            return
        try:
            library = ctypes.CDLL("librocprofiler-sdk-roctx.so.1")
        except OSError:
            library = ctypes.CDLL("libroctx64.so")
        self._push = library.roctxRangePushA
        self._pop = library.roctxRangePop
        self._push.argtypes = [ctypes.c_char_p]
        self._push.restype = ctypes.c_int
        self._pop.argtypes = []
        self._pop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        if self._push is not None:
            self._push(name.encode("utf-8"))

    def pop(self) -> None:
        if self._pop is not None:
            self._pop()


class _StageLedger:
    """Low-frequency host stage timers installed on one loaded service."""

    def __init__(self, *, roctx: bool) -> None:
        self._roctx = _Roctx(roctx)
        self._lock = threading.Lock()
        self._active_arm: str | None = None
        self._counter = 0
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._allocation_samples: dict[str, list[dict[str, int]]] = defaultdict(list)
        self._markers: dict[str, list[str]] = defaultdict(list)
        self._restores: list[tuple[Any, str, Any]] = []

    def install(self, owner: Any, method_name: str, phase: str) -> bool:
        original = getattr(owner, method_name, None)
        if not callable(original):
            return False

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.measure(phase):
                return original(*args, **kwargs)

        setattr(owner, method_name, wrapped)
        self._restores.append((owner, method_name, original))
        return True

    @contextmanager
    def arm(self, arm: str) -> Iterator[None]:
        with self._lock:
            if self._active_arm is not None:
                raise RuntimeError("stage ledger arms cannot overlap")
            self._active_arm = str(arm)
            self._counter = 0
            self._samples.clear()
            self._allocation_samples.clear()
            self._markers.clear()
        try:
            with self.measure("arm_complete"):
                yield
        finally:
            with self._lock:
                self._active_arm = None

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        with self._lock:
            arm = self._active_arm
            if arm is None:
                enabled = False
                marker_name = ""
            else:
                enabled = True
                self._counter += 1
                marker_name = (
                    f"{ROCTX_PREFIX}{arm}_{str(phase)}_{self._counter:06d}"
                )
                self._markers[str(phase)].append(marker_name)
        if not enabled:
            yield
            return
        allocation_before = memory_stats() if str(phase) == "cycle_total" else None
        self._roctx.push(marker_name)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            allocation_after = (
                memory_stats() if allocation_before is not None else None
            )
            self._roctx.pop()
            with self._lock:
                self._samples[str(phase)].append(elapsed)
                if allocation_before is not None and allocation_after is not None:
                    self._allocation_samples[str(phase)].append(
                        {
                            "allocated_bytes": int(
                                allocation_after["total_allocated_bytes"]
                                - allocation_before["total_allocated_bytes"]
                            ),
                            "freed_bytes": int(
                                allocation_after["total_freed_bytes"]
                                - allocation_before["total_freed_bytes"]
                            ),
                            "active_delta": int(
                                allocation_after["active_allocations"]
                                - allocation_before["active_allocations"]
                            ),
                            "current_bytes_delta": int(
                                allocation_after["current_allocated_bytes"]
                                - allocation_before["current_allocated_bytes"]
                            ),
                        }
                    )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "totals_seconds": {
                    phase: sum(samples)
                    for phase, samples in sorted(self._samples.items())
                },
                "samples_seconds": {
                    phase: list(samples)
                    for phase, samples in sorted(self._samples.items())
                },
                "call_counts": {
                    phase: len(samples)
                    for phase, samples in sorted(self._samples.items())
                },
                "allocation_samples": {
                    phase: [dict(sample) for sample in samples]
                    for phase, samples in sorted(self._allocation_samples.items())
                },
                "marker_names": {
                    phase: list(names)
                    for phase, names in sorted(self._markers.items())
                },
            }

    def close(self) -> None:
        for owner, method_name, original in reversed(self._restores):
            setattr(owner, method_name, original)
        self._restores.clear()


def _install_stage_ledger(service: Any, ledger: _StageLedger) -> dict[str, bool]:
    driver = service.inner
    runner = driver._runner
    loop = driver._loop
    installed = {
        "cycle_total": ledger.install(loop, "_run_staged_speculative_cycle", "cycle_total"),
        "target_prefill": ledger.install(runner, "prefill_batch", "target_prefill"),
        "ar_decode": ledger.install(runner, "decode_batch", "ar_decode"),
        "provider_k0_attach": ledger.install(
            runner, "prepare_speculative_k0", "provider_k0_attach"
        ),
        "provider_open": ledger.install(
            runner, "prepare_speculative_requests", "provider_open"
        ),
        "proposal": ledger.install(runner, "propose_speculative_batch", "proposal"),
        "target_accept_commit_provider": ledger.install(
            runner,
            "execute_target_frontier",
            "target_accept_commit_provider",
        ),
        "resident_owner_transition": ledger.install(
            runner, "_flush_row_owner", "resident_owner_transition"
        ),
        "graph_close": ledger.install(
            runner, "_close_packed_decode_graphs", "graph_close"
        ),
        "terminal_reclaim": ledger.install(runner, "reclaim", "terminal_reclaim"),
    }
    resolve_adapter = getattr(runner, "_resolved_mtp2_adapter", None)
    adapter = resolve_adapter() if callable(resolve_adapter) else None
    installed["nextn_prompt_prime_c1"] = bool(
        adapter is not None
        and ledger.install(adapter, "_catch_up_provider", "nextn_prompt_prime")
    )
    installed["nextn_prompt_prime_batch"] = bool(
        adapter is not None
        and ledger.install(adapter, "_catch_up_provider_batch", "nextn_prompt_prime")
    )
    begin_streaming = None if adapter is None else getattr(
        adapter, "begin_prompt_streaming", None
    )
    installed["provider_streaming_open"] = callable(begin_streaming)
    if callable(begin_streaming):
        wrapped_sink_ids: set[int] = set()

        def wrapped_begin_streaming(*args: Any, **kwargs: Any):
            with ledger.measure("provider_streaming_open"):
                sinks = begin_streaming(*args, **kwargs)
            if sinks is not None:
                for sink in sinks:
                    if id(sink) in wrapped_sink_ids:
                        continue
                    wrapped_sink_ids.add(id(sink))
                    ledger.install(sink, "consume", "nextn_prompt_prime")
            return sinks

        setattr(adapter, "begin_prompt_streaming", wrapped_begin_streaming)
        ledger._restores.append(
            (adapter, "begin_prompt_streaming", begin_streaming)
        )
    return installed


def _close_preserving_primary(
    closer: Any,
    *,
    primary_failure_active: bool,
) -> dict[str, str] | None:
    """Close an owner without replacing an in-flight measurement failure."""

    try:
        closer()
    except BaseException as exc:
        if not primary_failure_active:
            raise
        return {"type": type(exc).__name__, "message": str(exc)}
    return None


def _memory_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    delta_keys = (
        "total_allocated_bytes",
        "total_freed_bytes",
        "current_allocated_bytes",
        "active_allocations",
    )
    return {
        **{f"delta_{key}": int(after.get(key, 0)) - int(before.get(key, 0)) for key in delta_keys},
        "peak_allocated_bytes_after": int(after.get("peak_allocated_bytes", 0)),
        "peak_allocations_after": int(after.get("peak_allocations", 0)),
    }


def _telemetry_payload(output: Any) -> dict[str, Any]:
    telemetry = getattr(output, "telemetry", None)
    if telemetry is None or getattr(telemetry, "timing", None) is None:
        raise BridgeContractError("generation output is missing timing telemetry")
    decode_state = getattr(telemetry, "decode_state", None)
    return {
        "timing_scope": getattr(telemetry, "timing_scope", None) or "choice",
        "batch_id": getattr(telemetry, "batch_id", None),
        "group_rows": getattr(telemetry, "group_rows", None) or 1,
        "timing_owner": (
            True
            if getattr(telemetry, "timing_owner", None) is None
            else bool(telemetry.timing_owner)
        ),
        "timing": dict(telemetry.timing),
        "execution_path": getattr(decode_state, "execution_path", None),
        "request_id": getattr(decode_state, "request_id", None),
        "prompt_tokens": getattr(decode_state, "prompt_tokens", None),
        "generated_tokens": getattr(decode_state, "generated_tokens", None),
        "diagnostics": (
            None
            if getattr(telemetry, "diagnostics", None) is None
            else dict(telemetry.diagnostics)
        ),
    }


def _diagnostic_static_eligibility(budget: int, *, max_realized_group_rows: int = 8) -> SpeculativeMTPStaticEligibility:
    """Screening-only eligibility mirroring the c1c8 bench diagnostic.

    Explicitly unqualified (never automatic), bounded to the requested
    candidate budget and the diagnostic max width. This is the only mechanism
    the better-MTP campaign uses to measure sub-capacity cells (C3/K3, C5/K3)
    through the bridge; it is not production admission.
    """

    key = {
        "candidate_budget": int(budget),
        "max_realized_group_rows": int(max_realized_group_rows),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return SpeculativeMTPStaticEligibility(
        state=SpeculativeMTPStaticState.SPECULATIVE_CAPABLE,
        reason="diagnostic_physical_gguf_mtp",
        max_candidate_count=int(budget),
        max_realized_group_rows=int(max_realized_group_rows),
        automatic_eligible=False,
        strict_fallback_key="gguf_target_ar",
        evidence_key=f"gguf-c1-c{max_realized_group_rows}-generation2-diagnostic",
        evidence_fingerprint=f"sha256:{digest}",
    )


def _request(
    prompt: str,
    max_tokens: int,
    eligibility: SpeculativeMTPStaticEligibility | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        prompts=(str(prompt),),
        max_tokens=int(max_tokens),
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        speculative_mtp_static_eligibility=eligibility,
    )


def _run_service_route(service: Any, request: GenerationRequest, concurrency: int, *, staged: bool):
    requests = tuple(request for _ in range(int(concurrency)))
    handles = (
        service.submit_speculative_children(requests)
        if staged
        else service.submit_children(requests)
    )
    return tuple(handle.result(timeout=900) for handle in handles)


def _run_legacy_native(
    direct_generator: Any,
    direct_config: Any,
    request: GenerationRequest,
):
    direct_request = GenerationRequest(
        prompts=request.prompts,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        ignore_eos=request.ignore_eos,
    )
    if bool(getattr(direct_config, "is_moe", False)):
        return tuple(direct_generator.generate_speculative_mtp_detailed(direct_request))
    return tuple(
        direct_generator._generate_dense_speculative_mtp_detailed(
            direct_request,
            config=direct_config,
        )
    )


def _recent_routes(service: Any, concurrency: int) -> tuple[dict[str, Any], ...]:
    snapshot = service.live_loop_snapshot()
    recent = snapshot.get("runner", {}).get("routes", {}).get("recent_completed", [])
    return tuple(dict(row) for row in recent[-int(concurrency) :])


def _route_identity(
    arm: str,
    timing_payloads: Sequence[Mapping[str, Any]],
    recent_routes: Sequence[Mapping[str, Any]],
) -> str:
    paths = {
        str(row.get("execution_path"))
        for row in timing_payloads
        if row.get("execution_path") is not None
    }
    if arm == "specdec2":
        if not recent_routes or not all(bool(row.get("specdec2_mtp2_used")) for row in recent_routes):
            return "not_specdec2"
        if paths and paths != {"gguf_specdec2_mtp2"}:
            return "not_specdec2"
        return "specdec2"
    if arm == "true_ar":
        if any(bool(row.get("specdec2_mtp2_used")) for row in recent_routes):
            return "specdec2"
        if any("mtp" in path.lower() for path in paths):
            return "legacy_native"
        return "true_ar"
    if arm == "legacy_native":
        if paths and not any("mtp" in path.lower() for path in paths):
            return "not_legacy_native"
        return "legacy_native"
    return "unknown"


def _decode_only_seconds(
    arm: str,
    stage_ledger: Mapping[str, Any],
    timing_summary: Mapping[str, Any],
) -> float:
    stage_totals = stage_ledger.get("totals_seconds", {})
    if arm == "true_ar":
        value = float(stage_totals.get("ar_decode", 0.0))
    elif arm == "specdec2":
        value = float(stage_totals.get("cycle_total", 0.0)) + float(
            stage_totals.get("ar_decode", 0.0)
        )
    else:
        owned = timing_summary.get("owned_totals_ms", {})
        value = float(owned.get("decode_ms", 0.0)) / 1000.0
        if value <= 0.0:
            value = max(
                0.0,
                float(owned.get("mtp_run_total_ms", 0.0))
                - float(owned.get("prefill_ms", 0.0)),
            ) / 1000.0
    if not math.isfinite(value) or value <= 0.0:
        raise BridgeContractError(f"{arm} has no positive decode-only timing owner")
    return value


def _run_arm(
    *,
    arm: str,
    service: Any,
    direct_generator: Any,
    direct_config: Any,
    request: GenerationRequest,
    concurrency: int,
    ledger: _StageLedger,
    legacy_native_supported: bool = True,
) -> dict[str, Any]:
    if arm == "legacy_native" and (
        int(concurrency) != 1 or not bool(legacy_native_supported)
    ):
        return {
            "status": "skipped",
            "reason": (
                "dense_direct_legacy_requires_strict_fp32_state"
                if int(concurrency) == 1
                else "dense_direct_legacy_is_request_serial_for_c_gt_1"
            ),
            "realized_route": None,
        }
    before_memory = memory_stats()
    started = time.perf_counter()
    with ledger.arm(arm):
        if arm == "true_ar":
            outputs = _run_service_route(service, request, concurrency, staged=False)
        elif arm == "specdec2":
            outputs = _run_service_route(service, request, concurrency, staged=True)
        elif arm == "legacy_native":
            outputs = _run_legacy_native(direct_generator, direct_config, request)
        else:  # pragma: no cover - parser and caller own this
            raise ValueError(f"unknown bridge arm: {arm}")
    complete_wall = time.perf_counter() - started
    stage_snapshot = ledger.snapshot()
    after_memory = memory_stats()
    timing_payloads = tuple(_telemetry_payload(output) for output in outputs)
    timing_summary = normalize_timing_payloads(timing_payloads)
    recent_routes = (
        _recent_routes(service, concurrency)
        if arm in {"true_ar", "specdec2"}
        else ()
    )
    generated = tuple(
        tuple(int(token) for token in (getattr(output, "generated_token_ids", None) or ()))
        for output in outputs
    )
    if len(generated) != int(concurrency) or any(not row for row in generated):
        raise BridgeContractError(f"{arm} did not return one non-empty ID row per request")
    return {
        "status": "complete",
        "realized_route": _route_identity(arm, timing_payloads, recent_routes),
        "complete_wall_seconds": complete_wall,
        "decode_only_seconds": _decode_only_seconds(
            arm,
            stage_snapshot,
            timing_summary,
        ),
        "generated_token_ids": [list(row) for row in generated],
        "generated_tokens": sum(len(row) for row in generated),
        "timing_payloads": list(timing_payloads),
        "timing_ownership": timing_summary,
        "stage_ledger": stage_snapshot,
        "recent_routes": list(recent_routes),
        "memory": {
            "before": before_memory,
            "after": after_memory,
            "delta": _memory_delta(before_memory, after_memory),
            "scope": "hipengine-owned allocations only; library-internal allocations excluded",
        },
    }


def _expected_grid(workload: Mapping[str, Any]) -> set[tuple[str, int, int, int]]:
    return {
        (str(prompt_id), int(run), int(concurrency), int(budget))
        for prompt_id in workload.get("prompt_ids", ())
        for run in range(int(workload.get("runs", 0)))
        for concurrency in workload.get("concurrency", ())
        for budget in workload.get("candidate_budgets", ())
    }


def validate_bridge_artifact(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema", 0)) != 1 or payload.get("kind") != "specdec2_perf_bridge":
        raise BridgeContractError("artifact schema/kind is not the SPECDEC2 bridge contract")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BridgeContractError("artifact has no provenance")
    if bool(provenance.get("staged_dirty")) or bool(provenance.get("unstaged_dirty")):
        raise BridgeContractError("performance bridge requires a tracked-clean worktree")
    if provenance.get("unexpected_untracked"):
        raise BridgeContractError("performance bridge has unexpected untracked files")
    workload = payload.get("workload")
    if not isinstance(workload, Mapping):
        raise BridgeContractError("artifact has no workload contract")
    prompt_ids = tuple(str(value) for value in workload.get("prompt_ids", ()))
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise BridgeContractError("artifact has no model contract")
    execution_profile = str(model.get("execution_profile", "strict"))
    if (
        workload.get("scope") == "full"
        and bool(workload.get("selection_complete", True))
        and prompt_ids != FULL_PROMPT_IDS
    ):
        raise BridgeContractError("full bridge has an incomplete canonical prompt suite")

    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise BridgeContractError("artifact cells must be a list")
    expected = _expected_grid(workload)
    actual: set[tuple[str, int, int, int]] = set()
    repeat_ids: dict[
        tuple[str, int, int, str],
        list[tuple[int, tuple[tuple[int, ...], ...]]],
    ] = {}
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise BridgeContractError(f"cell {cell_index} is malformed")
        key = (
            str(cell.get("prompt_id")),
            int(cell.get("run", -1)),
            int(cell.get("concurrency", 0)),
            int(cell.get("candidate_budget", 0)),
        )
        if key in actual:
            raise BridgeContractError(f"bridge grid contains duplicate cell {key}")
        actual.add(key)
        arms = cell.get("arms")
        if not isinstance(arms, Mapping):
            raise BridgeContractError(f"cell {key} has no arms")
        complete_ids: list[tuple[tuple[int, ...], ...]] = []
        for arm in ARMS:
            row = arms.get(arm)
            if not isinstance(row, Mapping):
                raise BridgeContractError(f"cell {key} has no {arm} arm")
            if row.get("status") == "skipped":
                allowed_legacy_skip = bool(
                    arm == "legacy_native"
                    and (key[2] != 1 or execution_profile == "production")
                )
                if not allowed_legacy_skip:
                    raise BridgeContractError(f"cell {key} illegally skipped {arm}")
                continue
            if row.get("status") != "complete":
                raise BridgeContractError(f"cell {key} has incomplete {arm} arm")
            if row.get("realized_route") != arm:
                raise BridgeContractError(
                    f"cell {key} {arm} realized route {row.get('realized_route')!r}"
                )
            wall = float(row.get("complete_wall_seconds", 0.0))
            if arm == "true_ar" and (not math.isfinite(wall) or wall <= 0.0):
                raise BridgeContractError(f"cell {key} has invalid true AR denominator")
            decode = row.get("decode_only_seconds")
            if decode is None or not math.isfinite(float(decode)) or float(decode) <= 0.0:
                raise BridgeContractError(f"cell {key} {arm} has invalid decode-only timing")
            timing_rows = row.get("timing_payloads")
            if not isinstance(timing_rows, list):
                raise BridgeContractError(f"cell {key} {arm} has no timing payloads")
            normalize_timing_payloads(timing_rows)
            ids = tuple(
                tuple(int(token) for token in tokens)
                for tokens in row.get("generated_token_ids", ())
            )
            if len(ids) != key[2] or any(not tokens for tokens in ids):
                raise BridgeContractError(f"cell {key} {arm} generated IDs are malformed")
            complete_ids.append(ids)
            repeat_key = (key[0], key[2], key[3], arm)
            repeat_ids.setdefault(repeat_key, []).append((key[1], ids))
        cross_arm_exact = bool(complete_ids) and len(set(complete_ids)) == 1
        if bool(cell.get("exact")) != cross_arm_exact:
            raise BridgeContractError(
                f"cell {key} generated IDs conflict with cross-arm exactness diagnostic"
            )
        if execution_profile != "production" and not cross_arm_exact:
            raise BridgeContractError(f"cell {key} generated IDs are not exact across arms")
    if execution_profile == "production":
        for repeat_key, rows in repeat_ids.items():
            ordered = sorted(rows, key=lambda item: item[0])
            if len({ids for _, ids in ordered}) != 1:
                raise BridgeContractError(
                    f"production arm {repeat_key} generated IDs are not repeatable"
                )
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BridgeContractError(
            f"bridge grid is incomplete or unexpected: missing={missing[:5]} extra={extra[:5]}"
        )


def _repo_provenance() -> dict[str, Any]:
    state = collect_repo_state(REPO_ROOT)
    completed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    untracked = tuple(sorted(path for path in completed.stdout.split("\0") if path))
    expected = tuple(path for path in untracked if path.startswith(_ALLOWED_UNTRACKED_ROOT))
    unexpected = tuple(path for path in untracked if path not in expected)
    return {
        **state,
        "shared_untracked_files_excluded": bool(expected),
        "excluded_untracked_count": len(expected),
        "excluded_untracked_sha256": hashlib.sha256(
            "\n".join(expected).encode("utf-8")
        ).hexdigest(),
        "unexpected_untracked": list(unexpected),
    }


def bridge_speed_claim_eligible(
    *,
    scope: str,
    prompt_ids: Sequence[str],
    runs: int,
    concurrencies: Sequence[int],
    tracked_clean: bool,
    unexpected_untracked: Sequence[str],
    all_exact: bool,
) -> bool:
    """Gate the retained P1 packet without requiring diagnostic K cells."""

    return bool(
        str(scope) == "full"
        and tuple(str(value) for value in prompt_ids) == FULL_PROMPT_IDS
        and int(runs) >= 3
        and set(int(value) for value in concurrencies) == {1, 2, 4}
        and bool(tracked_clean)
        and not tuple(unexpected_untracked)
        and bool(all_exact)
    )


def _arm_ratios(cell: Mapping[str, Any]) -> dict[str, float | None]:
    arms = cell["arms"]
    ar = arms["true_ar"]
    denominator = float(ar["complete_wall_seconds"])
    result: dict[str, float | None] = {}
    for arm in ("legacy_native", "specdec2"):
        row = arms[arm]
        result[f"{arm}_over_true_ar_wall"] = (
            None
            if row.get("status") != "complete"
            else float(row["complete_wall_seconds"]) / denominator
        )
    return result


def _summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    ratios: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for cell in cells:
        concurrency = int(cell["concurrency"])
        budget = int(cell["candidate_budget"])
        for arm, row in cell["arms"].items():
            if row.get("status") != "complete":
                continue
            grouped[(concurrency, budget, str(arm))].append(row)
        for label, value in cell.get("ratios", {}).items():
            if value is not None:
                ratios[(concurrency, budget, str(label))].append(float(value))
    rows: dict[str, Any] = {}
    for (concurrency, budget, arm), samples in sorted(grouped.items()):
        walls = [float(row["complete_wall_seconds"]) for row in samples]
        decode = [float(row["decode_only_seconds"]) for row in samples]
        generated = sum(int(row["generated_tokens"]) for row in samples)
        key = f"c{concurrency}_k{budget}_{arm}"
        rows[key] = {
            "samples": len(samples),
            "complete_wall_seconds": walls,
            "complete_wall_median_seconds": statistics.median(walls),
            "decode_only_seconds": decode,
            "decode_only_median_seconds": statistics.median(decode),
            "generated_tokens": generated,
            "aggregate_generated_tok_s": generated / sum(walls),
        }
    ratio_rows = {
        f"c{concurrency}_k{budget}_{label}": {
            "samples": values,
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        for (concurrency, budget, label), values in sorted(ratios.items())
    }
    return {"arms": rows, "ratios": ratio_rows}


@contextmanager
def _temporary_environment(updates: Mapping[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _selected_prompts(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    rows = load_prompt_suite(Path(args.prompts).resolve())
    if args.scope == "train":
        rows = tuple(row for row in rows if row["id"] not in _HELDOUT_IDS)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if not rows:
        raise ValueError("selected prompt suite is empty")
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        choices=tuple(sorted(HIP_BACKEND_TARGET_ARCH)),
        default="hip_gfx1151",
    )
    parser.add_argument("--target-arch")
    parser.add_argument("--quant-label", default="Q4_K_S")
    parser.add_argument(
        "--execution-profile",
        choices=("strict", "production"),
        default="strict",
    )
    parser.add_argument("--gpu-max-hw-queues", type=int)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--scope", choices=("train", "full"), default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=parse_concurrencies, default=(1, 2, 4))
    parser.add_argument(
        "--service-capacity",
        type=int,
        help="Resident owner capacity; defaults to the largest measured concurrency.",
    )
    parser.add_argument("--budgets", type=parse_budgets, default=(2,))
    parser.add_argument("--max-tokens", type=int, default=25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--roctx-markers", action="store_true")
    parser.add_argument("--profile-child", action="store_true")
    parser.add_argument(
        "--diagnostic-plan",
        action="store_true",
        help=(
            "Install the fail-closed diagnostic serving-plan resolver (same "
            "semantics as gguf_mtp_c1c8_server_bench --generation2-diagnostic) "
            "so explicitly unqualified sub-capacity screening cells can be "
            "profiled. Screening-only; never production admission."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-fail", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not Path(args.model).is_file():
        raise ValueError(f"model not found: {args.model}")
    if int(args.max_tokens) <= 1:
        raise ValueError("--max-tokens must exceed one")
    if int(args.runs) <= 0:
        raise ValueError("--runs must be positive")
    if args.limit is not None and int(args.limit) <= 0:
        raise ValueError("--limit must be positive")
    if int(args.max_sequence_length) <= 0:
        raise ValueError("--max-sequence-length must be positive")
    bridge_service_capacity(
        args.concurrency,
        requested_capacity=args.service_capacity,
    )
    if args.require_cached_build and args.compiler_version_file is None:
        raise ValueError("--require-cached-build requires --compiler-version-file")
    if args.compiler_version_file is not None and not Path(args.compiler_version_file).is_file():
        raise ValueError(f"compiler version file not found: {args.compiler_version_file}")
    resolve_platform(
        backend=args.backend,
        target_arch=args.target_arch,
        quant_label=args.quant_label,
        gpu_max_hw_queues=args.gpu_max_hw_queues,
        environ=os.environ,
    )
    if args.profile_child:
        if not args.roctx_markers or not args.require_cached_build:
            raise ValueError("--profile-child requires ROCTX markers and cached builds")
        if (
            len(args.concurrency) != 1
            or len(args.budgets) != 1
            or int(args.runs) != 1
            or int(args.limit or 0) != 1
        ):
            raise ValueError(
                "--profile-child requires one concurrency, one budget, one run, and --limit 1"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    from hipengine import LLM
    from hipengine.generation.qwen35_gguf import _gguf_mtp_required_tensor_names

    prompts = _selected_prompts(args)
    repo_gate = _repo_provenance()
    compiler_version = (
        None
        if args.compiler_version_file is None
        else Path(args.compiler_version_file).read_text(encoding="utf-8")
    )
    platform_config = resolve_platform(
        backend=args.backend,
        target_arch=args.target_arch,
        quant_label=args.quant_label,
        gpu_max_hw_queues=args.gpu_max_hw_queues,
        environ=os.environ,
    )
    recurrent_state = "fp16" if args.execution_profile == "production" else "fp32"
    environment = {
        "HIPENGINE_HIP_ARCH": platform_config["target_arch"],
        "HIPENGINE_GGUF_FP16_RECURRENT_STATE": (
            "1" if args.execution_profile == "production" else "0"
        ),
        "GPU_MAX_HW_QUEUES": platform_config["gpu_max_hw_queues"],
        "HIPENGINE_REQUIRE_CACHED_BUILD": (
            "1" if args.require_cached_build else os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD")
        ),
        "HIPENGINE_COMPILER_VERSION_FILE": (
            None
            if args.compiler_version_file is None
            else str(Path(args.compiler_version_file).resolve())
        ),
    }
    provenance = collect_artifact_provenance(
        repo_root=REPO_ROOT,
        configured_backend=str(platform_config["backend"]),
        resolved_backend=str(platform_config["backend"]),
        target_arch=str(platform_config["target_arch"]),
        model_path=Path(args.model).resolve(),
        quant=str(platform_config["quant_label"]),
        kv_dtype="bf16",
        command=(str(Path(sys.executable).resolve()), *sys.argv),
        environment={
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            **environment,
        },
        build_profile="specdec2_perf_p1_bridge",
        timing_protocol=(
            "complete request plus post-activation decode owner; index-only "
            "counterbalanced true_ar/legacy_native/specdec2"
        ),
        warmups=1 if args.warmup else 0,
        repetitions=int(args.runs),
        profiler={
            "enabled": bool(args.roctx_markers),
            "profile_child": bool(args.profile_child),
            "marker_prefix": ROCTX_PREFIX,
        },
        hipcc_version=compiler_version,
    )
    workload = {
        "scope": str(args.scope),
        "selection_complete": args.limit is None,
        "prompt_file": str(Path(args.prompts).resolve()),
        "prompt_file_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "prompt_ids": [str(row["id"]) for row in prompts],
        "categories": sorted({str(row["category"]) for row in prompts}),
        "heldout_ids": sorted(_HELDOUT_IDS),
        "concurrency": list(args.concurrency),
        "service_capacity": bridge_service_capacity(
            args.concurrency,
            requested_capacity=args.service_capacity,
        ),
        "candidate_budgets": list(args.budgets),
        "max_tokens": int(args.max_tokens),
        "runs": int(args.runs),
        "warmup": bool(args.warmup),
        "max_sequence_length": int(args.max_sequence_length),
        "sampling": "raw greedy; temperature=0, top_p=1",
        "counterbalance": (
            "even prompt+run index true_ar→legacy_native→specdec2; "
            "odd index reverse"
        ),
        "legacy_native_scope": (
            "C1 current-source direct dense transactional control only; "
            "C>1 skipped because direct dense ownership is request-serial"
        ),
    }
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "specdec2_perf_bridge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "performance_claim": False,
        "speed_claim_eligible": False,
        "provenance": repo_gate,
        "canonical_provenance": provenance,
        "host": {
            "hostname": platform.node(),
            "device": provenance.get("device_name"),
            "backend": platform_config["backend"],
            "target_arch": platform_config["target_arch"],
            "gpu_max_hw_queues": platform_config["gpu_max_hw_queues"],
            "queue_source": platform_config["queue_source"],
        },
        "model": {
            "path": str(Path(args.model).resolve()),
            "fingerprint": provenance.get("model_fingerprint"),
            "quant": platform_config["quant_label"],
            "kv": "bf16",
            "execution_profile": str(args.execution_profile),
            "recurrent_state": recurrent_state,
        },
        "workload": workload,
        "timing_contract": {
            "complete_wall": (
                "host route entry through terminal GenerationOutput/reclaim; "
                "includes tokenize/admission/activation/decode/publication/reclaim"
            ),
            "decode_only": (
                "sum of Generation-2 staged cycle/ar-tail owners after prefill; "
                "legacy uses direct decoder decode_ms"
            ),
            "stage_nesting": {
                "nextn_prompt_prime": (
                    "nested inside initial provider_k0_attach or a later "
                    "provider_open/refill attachment"
                ),
                "proposal_and_target": "nested inside cycle_total",
                "terminal_reclaim": "may include resident_owner_transition",
            },
            "memory": (
                "process-local hipEngine malloc/free byte counters; library-internal "
                "allocations and exact call counts deferred to API trace/P3"
            ),
        },
        "warmups": [],
        "loads": [],
        "cells": [],
        "completed_arms": 0,
    }
    atomic_write_json(args.output, payload)

    try:
        with _temporary_environment(environment):
            for budget in args.budgets:
                budget_started = time.perf_counter()
                os.environ["HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET"] = str(int(budget))
                llm = LLM(
                    str(Path(args.model).resolve()),
                    backend=str(platform_config["backend"]),
                    execution_profile=str(args.execution_profile),
                    max_active_requests=bridge_service_capacity(
                        args.concurrency,
                        requested_capacity=args.service_capacity,
                    ),
                    max_sequence_length=int(args.max_sequence_length),
                    speculative_candidate_budget=int(budget),
                )
                diag_eligibility = (
                    _diagnostic_static_eligibility(int(budget))
                    if args.diagnostic_plan
                    else None
                )
                ledger: _StageLedger | None = None
                load_row: dict[str, Any] | None = None
                try:
                    llm.prepare(
                        max_sequence_length=int(args.max_sequence_length)
                    )
                    service = llm._get_text_generator()
                    driver = service.inner
                    direct_generator = driver.inner
                    direct_config, _block_id, _required = _gguf_mtp_required_tensor_names(
                        direct_generator.weight_index
                    )
                    actual_fp16_state = os.environ.get(
                        "HIPENGINE_GGUF_FP16_RECURRENT_STATE",
                        "0",
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    payload["model"]["recurrent_state"] = (
                        "fp16" if actual_fp16_state else "fp32"
                    )
                    payload["canonical_provenance"]["environment"][
                        "HIPENGINE_GGUF_FP16_RECURRENT_STATE"
                    ] = "1" if actual_fp16_state else "0"
                    ledger = _StageLedger(roctx=bool(args.roctx_markers))
                    installed = _install_stage_ledger(service, ledger)
                    load_row = {
                        "candidate_budget": int(budget),
                        "load_seconds": time.perf_counter() - budget_started,
                        "execution_profile_manifest_sha256": getattr(
                            llm, "execution_profile_manifest_sha256", None
                        ),
                        "execution_profile_strict_manifest_sha256": getattr(
                            llm, "execution_profile_strict_manifest_sha256", None
                        ),
                        "stage_instrumentation": installed,
                    }
                    payload["loads"].append(load_row)
                    atomic_write_json(args.output, payload)

                    if args.warmup:
                        warm_prompt = _render_prompt_suite_messages(
                            [{"role": "user", "content": "Write one short greeting."}]
                        )
                        warm_request = _request(
                            warm_prompt,
                            min(int(args.max_tokens), 5),
                            diag_eligibility,
                        )
                        for concurrency in args.concurrency:
                            for arm in ARMS:
                                warm = _run_arm(
                                    arm=arm,
                                    service=service,
                                    direct_generator=direct_generator,
                                    direct_config=direct_config,
                                    request=warm_request,
                                    concurrency=int(concurrency),
                                    ledger=ledger,
                                    legacy_native_supported=(
                                        args.execution_profile == "strict"
                                    ),
                                )
                                payload["warmups"].append(
                                    {
                                        "candidate_budget": int(budget),
                                        "concurrency": int(concurrency),
                                        "arm": arm,
                                        "status": warm["status"],
                                        "reason": warm.get("reason"),
                                        "complete_wall_seconds": warm.get(
                                            "complete_wall_seconds"
                                        ),
                                        "realized_route": warm.get("realized_route"),
                                    }
                                )
                                print(
                                    json.dumps(
                                        {
                                            "phase": "warmup",
                                            "budget": int(budget),
                                            "concurrency": int(concurrency),
                                            "arm": arm,
                                            "status": warm["status"],
                                        },
                                        sort_keys=True,
                                    ),
                                    flush=True,
                                )
                                atomic_write_json(args.output, payload)

                    for run_index in range(int(args.runs)):
                        for prompt_index, prompt in enumerate(prompts):
                            request = _request(
                                str(prompt["rendered_prompt"]),
                                int(args.max_tokens),
                                diag_eligibility,
                            )
                            prompt_tokens = tuple(
                                int(token)
                                for token in service.tokenize(
                                    str(prompt["rendered_prompt"])
                                )
                            )
                            for concurrency in args.concurrency:
                                cell: dict[str, Any] = {
                                    "prompt_id": str(prompt["id"]),
                                    "category": str(prompt["category"]),
                                    "heldout": str(prompt["id"]) in _HELDOUT_IDS,
                                    "prompt_sha256": str(prompt["prompt_sha256"]),
                                    "prompt_token_count": len(prompt_tokens),
                                    "prompt_token_sha256": hashlib.sha256(
                                        json.dumps(prompt_tokens).encode("ascii")
                                    ).hexdigest(),
                                    "run": int(run_index),
                                    "concurrency": int(concurrency),
                                    "candidate_budget": int(budget),
                                    "execution_order": list(
                                        arm_order(prompt_index + run_index)
                                    ),
                                    "arms": {},
                                    "exact": False,
                                }
                                payload["cells"].append(cell)
                                for arm in cell["execution_order"]:
                                    result = _run_arm(
                                        arm=str(arm),
                                        service=service,
                                        direct_generator=direct_generator,
                                        direct_config=direct_config,
                                        request=request,
                                        concurrency=int(concurrency),
                                        ledger=ledger,
                                        legacy_native_supported=(
                                            args.execution_profile == "strict"
                                        ),
                                    )
                                    cell["arms"][str(arm)] = result
                                    payload["completed_arms"] = int(
                                        payload["completed_arms"]
                                    ) + 1
                                    print(
                                        json.dumps(
                                            {
                                                "phase": "measure",
                                                "run": int(run_index),
                                                "prompt": prompt["id"],
                                                "concurrency": int(concurrency),
                                                "budget": int(budget),
                                                "arm": arm,
                                                "status": result["status"],
                                                "wall_seconds": result.get(
                                                    "complete_wall_seconds"
                                                ),
                                                "route": result.get("realized_route"),
                                            },
                                            sort_keys=True,
                                        ),
                                        flush=True,
                                    )
                                    atomic_write_json(args.output, payload)
                                complete_ids = [
                                    tuple(
                                        tuple(int(token) for token in tokens)
                                        for tokens in row["generated_token_ids"]
                                    )
                                    for row in cell["arms"].values()
                                    if row.get("status") == "complete"
                                ]
                                cell["exact"] = bool(
                                    complete_ids and len(set(complete_ids)) == 1
                                )
                                cell["ratios"] = _arm_ratios(cell)
                                atomic_write_json(args.output, payload)
                finally:
                    primary_failure_active = sys.exc_info()[0] is not None
                    if ledger is not None:
                        ledger.close()
                    close_error = _close_preserving_primary(
                        llm.close,
                        primary_failure_active=primary_failure_active,
                    )
                    if load_row is not None:
                        load_row["memory_after_close"] = memory_stats()
                        if close_error is not None:
                            load_row["secondary_close_error"] = close_error
                    elif close_error is not None:
                        payload.setdefault("secondary_close_errors", []).append(
                            close_error
                        )
                    atomic_write_json(args.output, payload)

        payload["summary"] = _summarize(payload["cells"])
        payload["status"] = "complete"
        validate_bridge_artifact(payload)
        payload["speed_claim_eligible"] = bridge_speed_claim_eligible(
            scope=str(args.scope),
            prompt_ids=tuple(str(row["id"]) for row in prompts),
            runs=int(args.runs),
            concurrencies=args.concurrency,
            tracked_clean=(
                not bool(repo_gate["staged_dirty"])
                and not bool(repo_gate["unstaged_dirty"])
            ),
            unexpected_untracked=repo_gate["unexpected_untracked"],
            all_exact=all(bool(cell["exact"]) for cell in payload["cells"]),
        )
        payload["performance_claim"] = bool(payload["speed_claim_eligible"])
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(args.output, payload)
        raise
    atomic_write_json(args.output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = run(args)
    except (BridgeContractError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "speed_claim_eligible": payload["speed_claim_eligible"],
                "cells": len(payload["cells"]),
                "output": str(Path(args.output).resolve()),
            },
            sort_keys=True,
        )
    )
    if args.fail_on_fail and payload["status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
