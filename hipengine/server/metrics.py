"""Prometheus metrics ownership for the OpenAI-compatible server.

This module owns the additive request counters, the recursively immutable
snapshot of normalized metric inputs, the normalization helpers, and the
Prometheus text exposition.

It takes explicit data only: ``_ServerMetrics`` counters and a
``_MetricsSnapshot`` carrying the normalized resident/pool/graph/queue/session/
MTP/scheduler values. The application passes the graph histogram bucket names to
``_graph_bucket_metric_values`` before constructing the snapshot. It imports
only the standard library. Engine capability discovery, model lifecycle,
live-snapshot reads, queue attribute reads and service-health attribution stay
in ``hipengine.server.api``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any


_EMPTY_METRICS_MAPPING: Mapping[str, Any] = MappingProxyType({})
_IMMUTABLE_METRICS_LEAVES = (str, bytes, int, float, bool, type(None))


class _FrozenList(tuple):
    """An immutable list-shaped sequence, thawed back to ``list`` when needed."""

    __slots__ = ()


def _freeze_metrics_value(value: Any, path: str = "$") -> Any:
    """Recursively close one snapshot value over immutable containers.

    Mappings become ``MappingProxyType`` and sequences become tuples. Lists
    keep their list identity through ``_FrozenList`` so a later thaw restores
    the collected value type. Primitive leaves are returned unchanged. This
    freezer is applied to normalized metric output, not the raw live snapshot:
    the application normalizes known fields first and drops unknown extras, so
    any other leaf is a boundary bug and is refused.
    """

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_metrics_value(item, f"{path}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, _FrozenList)):
        return _FrozenList(
            _freeze_metrics_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, tuple):
        return tuple(
            _freeze_metrics_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, _IMMUTABLE_METRICS_LEAVES):
        return value
    raise TypeError(
        f"metrics snapshot value at {path} is not an immutable primitive: "
        f"{type(value).__name__}"
    )


def _freeze_metrics_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a recursively read-only mapping, or the shared empty mapping."""

    if not isinstance(value, Mapping):
        return _EMPTY_METRICS_MAPPING
    return _freeze_metrics_value(value)


def _thaw_metrics_value(value: Any) -> Any:
    """Return a mutable, JSON-shaped copy of a frozen snapshot value."""

    if isinstance(value, Mapping):
        return {key: _thaw_metrics_value(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw_metrics_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw_metrics_value(item) for item in value)
    return value


@dataclass(frozen=True)
class _MetricsSnapshot:
    """One scrape's explicit, recursively immutable normalized inputs.

    The application collects the raw live data, normalizes the fields the
    exposition consumes, and passes only that output here. Unknown live
    snapshot extras never reach this boundary. ``graph`` is the normalized
    graph metric mapping; the application resolves the raw graph stats and
    bucket names before constructing the snapshot.
    """

    resident: Mapping[str, Any] = field(
        default_factory=lambda: _resident_loop_metric_values(None)
    )
    pool: Mapping[str, Any] = field(
        default_factory=lambda: _kv_pool_metric_values(None)
    )
    graph: Mapping[str, Any] = field(
        default_factory=lambda: _graph_bucket_metric_values(
            {}, None, kernel_time_buckets=()
        )
    )
    queue: Mapping[str, Any] = field(
        default_factory=lambda: _generation_queue_metric_values(None)
    )
    chat_sessions_active: int | None = None
    chat_sessions_pending: int | None = None
    chat_sessions_max_active: int | None = None
    mtp_serving: float = 0.0
    scheduler_fairness: Mapping[str, Any] = field(
        default_factory=lambda: _scheduler_metric_values(None)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "resident", _freeze_metrics_mapping(self.resident))
        object.__setattr__(self, "pool", _freeze_metrics_mapping(self.pool))
        object.__setattr__(self, "graph", _freeze_metrics_mapping(self.graph))
        object.__setattr__(self, "queue", _freeze_metrics_mapping(self.queue))
        object.__setattr__(
            self,
            "scheduler_fairness",
            _freeze_metrics_mapping(self.scheduler_fairness),
        )


_KV_POOL_METRIC_DEFAULTS = {
    "current_bytes": 0.0,
    "high_water_observed_bytes": 0.0,
    "current_pages": 0.0,
    "high_water_observed_pages": 0.0,
    "grow_events": 0.0,
    "grow_failures": 0.0,
    "shrink_events": 0.0,
    "free_pages": 0.0,
    "refcounted_pages": 0.0,
    "pinned_pages": 0.0,
    "max_pages": 0.0,
    "budget_bytes": 0.0,
    "retired_pages": 0.0,
    "retired_bytes": 0.0,
}


@dataclass
class _ServerMetrics:
    """Additive server counters rendered by the opt-in Prometheus endpoint."""

    request_total: int = 0
    request_completed_total: int = 0
    request_failed_total: int = 0
    request_rejected_total: int = 0
    request_cancelled_total: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    mtp_requests_total: int = 0
    mtp_draft_tokens_accepted_total: int = 0
    mtp_draft_tokens_rejected_total: int = 0

    def record_success(self, usage: Mapping[str, Any]) -> None:
        self.request_total += 1
        self.request_completed_total += 1
        self.prompt_tokens_total += int(usage.get("prompt_tokens", 0))
        self.completion_tokens_total += int(usage.get("completion_tokens", 0))
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, Mapping) and (
            "accepted_prediction_tokens" in completion_details
            or "rejected_prediction_tokens" in completion_details
        ):
            accepted = int(completion_details.get("accepted_prediction_tokens", 0) or 0)
            rejected = int(completion_details.get("rejected_prediction_tokens", 0) or 0)
            self.mtp_requests_total += 1
            self.mtp_draft_tokens_accepted_total += accepted
            self.mtp_draft_tokens_rejected_total += rejected

    def record_failure(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1

    def record_rejected(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1
        self.request_rejected_total += 1

    def record_cancelled(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1
        self.request_cancelled_total += 1


def _nested_mapping(owner: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if owner is None:
        return {}
    value = owner.get(key)
    return value if isinstance(value, Mapping) else {}


def _non_negative_metric_value(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return default
    return numeric


def _string_value_counts(value: Any) -> dict[str, float]:
    """Count occurrences of each distinct string in a list of optional strings.

    Used for diagnostics that report *which* path ran rather than how much of
    something was used. ``None`` entries become ``"none"`` so an unset field is
    visible in the export instead of silently disappearing, and an absent or
    non-list value yields an empty mapping so the metric is simply not emitted.
    """

    if not isinstance(value, (list, tuple)):
        return {}
    counts: dict[str, float] = {}
    for item in value:
        key = "none" if item is None else str(item)
        counts[key] = counts.get(key, 0.0) + 1.0
    return counts


def _non_negative_metric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        numeric = _non_negative_metric_value(raw, default=-1.0)
        if numeric < 0:
            continue
        metrics[str(key)] = numeric
    return metrics


def _prefix_cache_metric_values(
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runner = _nested_mapping(snapshot, "runner")
    prefix = _nested_mapping(runner, "prefix_cache")
    if not prefix:
        return {
            "mode": "off",
            "block_size_tokens": 256,
            "snapshot_entries": 0,
            "snapshot_limit": 0,
            "retained_snapshot_entries": 0,
            "snapshot_bytes": 0,
            "retained_kv_pages": 0,
            "retained_kv_bytes": 0,
            "resident_bytes": 0,
            "resident_limit_bytes": 0,
        }
    return _thaw_metrics_value(prefix)


def _resident_loop_metric_values(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    loop = _nested_mapping(snapshot, "loop")
    requests = _nested_mapping(loop, "requests")
    bucket = _nested_mapping(loop, "physical_bucket")
    work = _nested_mapping(loop, "work_counts")
    policy = _nested_mapping(loop, "scheduler_policy")
    latency = _nested_mapping(loop, "latency_seconds")
    runner = _nested_mapping(snapshot, "runner")
    model_runner = _nested_mapping(runner, "model_runner")
    routes = _nested_mapping(runner, "routes")

    active_mask_value = bucket.get("active_mask")
    if isinstance(active_mask_value, Sequence) and not isinstance(active_mask_value, (str, bytes)):
        active_mask = "".join("1" if value is True else "0" for value in active_mask_value)
    else:
        active_mask = ""
    latency_rows: dict[str, dict[str, float]] = {}
    for kind in ("queue", "time_to_first_token", "inter_token", "service", "completion"):
        row = _nested_mapping(latency, kind)
        latency_rows[kind] = {
            "count": _non_negative_metric_value(row.get("count")),
            "sum": _non_negative_metric_value(row.get("sum")),
            "max": _non_negative_metric_value(row.get("max")),
            "p50": _non_negative_metric_value(row.get("p50")),
            "p95": _non_negative_metric_value(row.get("p95")),
        }
    return {
        "requests": {
            "pending": _non_negative_metric_value(requests.get("pending")),
            "admitted": _non_negative_metric_value(requests.get("admitted_current")),
            "active": _non_negative_metric_value(requests.get("active")),
            "admitted_total": _non_negative_metric_value(requests.get("admitted_total")),
            "reclaimed_total": _non_negative_metric_value(requests.get("reclaimed_total")),
        },
        "bucket": {
            "capacity": _non_negative_metric_value(bucket.get("capacity")),
            "active_rows": _non_negative_metric_value(bucket.get("active_c")),
            "occupied_slots": _non_negative_metric_value(bucket.get("occupied_slots")),
            "free_slots": _non_negative_metric_value(bucket.get("free_slots")),
            "occupancy_ratio": _non_negative_metric_value(bucket.get("occupancy_ratio")),
            "active_mask": active_mask,
        },
        "work": {
            "prefill": _non_negative_metric_value(work.get("prefill")),
            "decode": _non_negative_metric_value(work.get("decode")),
            "reclaim": _non_negative_metric_value(work.get("reclaim")),
        },
        "packed_workspace": {
            "current_bytes": _non_negative_metric_value(
                model_runner.get("packed_workspace_current_bytes")
            ),
            "owner_sessions": _non_negative_metric_value(
                model_runner.get("packed_workspace_owner_sessions")
            ),
            "leased_pool_bytes": _non_negative_metric_value(
                model_runner.get("packed_workspace_leased_pool_bytes")
            ),
            "release_events": _non_negative_metric_value(
                model_runner.get("packed_workspace_release_events")
            ),
            "released_bytes": _non_negative_metric_value(
                model_runner.get("packed_workspace_released_bytes")
            ),
        },
        "prefill_transients": {
            "oracle_owner_bytes": _non_negative_metric_value(
                (model_runner.get("prefill_transients") or {}).get("oracle_owner_bytes")
            ),
            "oracle_owner_count_total": _non_negative_metric_value(
                (model_runner.get("prefill_transients") or {}).get("oracle_owner_count_total")
            ),
            "oracle_observed_peak_bytes": _non_negative_metric_value(
                (model_runner.get("prefill_transients") or {}).get("oracle_observed_peak_bytes")
            ),
            "oracle_observed_peak_owners": _non_negative_metric_value(
                (model_runner.get("prefill_transients") or {}).get("oracle_observed_peak_owners")
            ),
            "hidden_and_bulk_owner_bytes": _non_negative_metric_value(
                (model_runner.get("prefill_transients") or {}).get("hidden_and_bulk_owner_bytes")
            ),
            # Which packed-prefill executor actually ran, per resident session.
            # ``route`` alone cannot distinguish the layer-outer executor from the
            # chunk-outer fallback (qwen35_gguf_runner.py:212-214), so without this
            # a run that silently took the fallback is indistinguishable from one
            # that took the deliverable route. String-valued, hence not a gauge.
            "executor_modes": _string_value_counts(
                (model_runner.get("prefill_transients") or {}).get(
                    "last_packed_executor_modes"
                )
            ),
            # The session's own KV layout, which gates whether the resumable
            # executor is attempted at all. Distinct from the route manifest's
            # kv_attention_source label, which describes the last decode.
            "kv_attention_sources": _string_value_counts(
                (model_runner.get("prefill_transients") or {}).get(
                    "kv_attention_sources"
                )
            ),
        },
        "persistent_kv": {
            "int8_payload_bytes": _non_negative_metric_value(
                model_runner.get("persistent_int8_payload_bytes")
            ),
            "bf16_payload_bytes": _non_negative_metric_value(
                model_runner.get("persistent_bf16_payload_bytes")
            ),
            "scale_bytes": _non_negative_metric_value(
                model_runner.get("persistent_scale_bytes")
            ),
            "bf16_mirror_bytes": _non_negative_metric_value(
                model_runner.get("persistent_bf16_mirror_bytes")
            ),
            "total_bytes": _non_negative_metric_value(
                model_runner.get("persistent_kv_total_bytes")
            ),
        },
        "policy": {
            "name": str(policy.get("prefill_decode_policy") or "unavailable"),
            "prefill_chunk_tokens": _non_negative_metric_value(
                policy.get("prefill_chunk_tokens")
            ),
            "fair_prefill_burst_chunks": max(
                1.0,
                _non_negative_metric_value(policy.get("fair_prefill_burst_chunks", 1)),
            ),
            "consecutive_prefill_chunks": _non_negative_metric_value(
                policy.get("consecutive_prefill_chunks")
            ),
            "last_work_kind": str(policy.get("last_work_kind") or "none"),
        },
        "latency": latency_rows,
        "route_counts": _non_negative_metric_mapping(routes.get("counts")),
        "fallback_reasons": _non_negative_metric_mapping(routes.get("fallback_reasons")),
        "last_execution_manifest": _nested_mapping(routes, "last_execution_manifest"),
    }


def _execution_manifest_metric_values(
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize the manifest fields the exposition consumes.

    The raw manifest is an optional, engine-owned mapping. Selecting only the
    consumed fields here keeps unknown extras (including arbitrary objects)
    out of the immutable snapshot, while calling ``str``/``bool`` on the known
    fields exactly where the pre-move renderer did. A malformed known field
    therefore keeps its old label/flag semantics instead of becoming a new
    ``/metrics`` failure.
    """

    data = manifest if isinstance(manifest, Mapping) else {}
    if not data:
        return {}
    return {
        "claim_level": str(data.get("claim_level") or "unavailable"),
        "kind": str(data.get("kind") or "unavailable"),
        "kv_attention_source": str(
            data.get("kv_attention_source") or "unavailable"
        ),
        "logical_c": str(data.get("logical_c") or data.get("rows") or 0),
        "mode": str(data.get("mode") or "unavailable"),
        "physical_execution_width": str(
            data.get("physical_execution_width") or data.get("physical_rows") or 0
        ),
        "rows": str(data.get("rows") or 0),
        "serial_decode_fallback": bool(data.get("serial_decode_fallback", False)),
        "throughput_claim_eligible": bool(
            data.get("throughput_claim_eligible", False)
        ),
    }


def _mtp_acceptance_rate_metric(metrics: _ServerMetrics) -> float:
    """Cumulative MTP draft acceptance rate across recorded requests."""

    total = metrics.mtp_draft_tokens_accepted_total + metrics.mtp_draft_tokens_rejected_total
    if total <= 0:
        return 0.0
    return metrics.mtp_draft_tokens_accepted_total / total


def _escape_prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_metric_value(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return repr(numeric)


def _append_resident_latency_metrics(
    lines: list[str],
    values: Mapping[str, Mapping[str, float]],
) -> None:
    name = "hipengine_resident_request_latency_seconds"
    lines.append(f"# HELP {name} Bounded resident request latency summaries by timing kind.")
    lines.append(f"# TYPE {name} summary")
    max_name = "hipengine_resident_request_latency_max_seconds"
    lines.append(f"# HELP {max_name} Maximum bounded resident request latency by timing kind.")
    lines.append(f"# TYPE {max_name} gauge")
    for kind, row in sorted(values.items()):
        label = _escape_prometheus_label_value(str(kind))
        lines.append(
            f'{name}{{kind="{label}",quantile="0.5"}} '
            f'{_format_metric_value(row.get("p50", 0.0))}'
        )
        lines.append(
            f'{name}{{kind="{label}",quantile="0.95"}} '
            f'{_format_metric_value(row.get("p95", 0.0))}'
        )
        lines.append(
            f'{name}_sum{{kind="{label}"}} '
            f'{_format_metric_value(row.get("sum", 0.0))}'
        )
        lines.append(
            f'{name}_count{{kind="{label}"}} '
            f'{_format_metric_value(row.get("count", 0.0))}'
        )
        lines.append(
            f'{max_name}{{kind="{label}"}} '
            f'{_format_metric_value(row.get("max", 0.0))}'
        )


def _append_labeled_gauge_metrics(
    lines: list[str],
    name: str,
    help_text: str,
    label: str,
    values: Mapping[str, float],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    for key, value in sorted(values.items()):
        lines.append(f'{name}{{{label}="{_escape_prometheus_label_value(key)}"}} {_format_metric_value(value)}')


def _append_labeled_counter_metrics(lines: list[str], name: str, help_text: str, label: str, values: Mapping[str, float]) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    for key, value in sorted(values.items()):
        lines.append(f'{name}{{{label}="{_escape_prometheus_label_value(key)}"}} {_format_metric_value(value)}')


def _kv_pool_metric_values(stats: Mapping[str, Any] | None) -> dict[str, float]:
    """Normalize one collected KV-pool stats mapping, or report defaults."""

    values = dict(_KV_POOL_METRIC_DEFAULTS)
    data = stats if isinstance(stats, Mapping) else {}
    for key in values:
        values[key] = _non_negative_metric_value(data.get(key))
    return values


def _generation_queue_metric_values(queue: Mapping[str, Any] | None) -> dict[str, float]:
    """Normalize the collected generation-batcher queue values."""

    data = queue if isinstance(queue, Mapping) else {}
    depth = _non_negative_metric_value(data.get("depth"))
    max_depth_raw = data.get("max_depth")
    max_depth = 0.0 if max_depth_raw is None else _non_negative_metric_value(max_depth_raw)
    active = data.get("worker_active")
    active_requests = _non_negative_metric_value(data.get("active_requests"))
    max_active_raw = data.get("max_active_requests")
    max_active = 0.0 if max_active_raw is None else _non_negative_metric_value(max_active_raw)
    return {
        "depth": depth,
        "max_depth": max_depth,
        "worker_active": 1.0 if bool(active) else 0.0,
        "active_requests": active_requests,
        "max_active_requests": max_active,
    }


def _graph_bucket_metric_values(
    live_graph: Mapping[str, Any] | None,
    graph_stats: Mapping[str, Any] | None,
    *,
    kernel_time_buckets: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize a live graph-bucket snapshot or a collected stats mapping.

    ``graph_stats`` is ``None`` when the engine exposed no stats object; an
    empty mapping means the object existed but reported no recognized fields.
    """

    values: dict[str, Any] = {
        "entries": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "replay_hit_rate": 0.0,
        "captures": 0.0,
        "replays": 0.0,
        "invalidations": 0.0,
        "buckets": {},
        "miss_reasons": {},
        "kernel_time_histogram_ns": {},
    }
    live = live_graph if isinstance(live_graph, Mapping) else {}
    if live:
        values["entries"] = _non_negative_metric_value(live.get("entries"))
        values["hits"] = _non_negative_metric_value(live.get("hits_total"))
        values["captures"] = _non_negative_metric_value(live.get("captures_total"))
        values["misses"] = values["captures"]
        values["replays"] = _non_negative_metric_value(live.get("replays_total"))
        values["invalidations"] = _non_negative_metric_value(
            live.get("invalidations_total")
        )
        lookups = values["hits"] + values["misses"]
        values["replay_hit_rate"] = values["hits"] / lookups if lookups > 0.0 else 0.0
        bucket_rows = _nested_mapping(live, "buckets")
        values["buckets"] = {
            str(bucket): {
                field: _non_negative_metric_value(row.get(field))
                for field in ("entries", "captures", "hits", "replays", "invalidations")
            }
            for bucket, row in bucket_rows.items()
            if isinstance(row, Mapping)
        }
        return values
    if graph_stats is None:
        return values
    data = graph_stats if isinstance(graph_stats, Mapping) else {}
    for key in ("entries", "hits", "misses"):
        values[key] = _non_negative_metric_value(data.get(key))
    lookups = values["hits"] + values["misses"]
    values["replay_hit_rate"] = values["hits"] / lookups if lookups > 0.0 else 0.0
    values["miss_reasons"] = _non_negative_metric_mapping(data.get("miss_reasons"))
    kernel_time_histogram = _non_negative_metric_mapping(data.get("kernel_time_histogram_ns"))
    known_kernel_time_histogram = {
        str(bucket): 0.0 for bucket in kernel_time_buckets
    }
    bucket_set = frozenset(known_kernel_time_histogram)
    for bucket, value in kernel_time_histogram.items():
        if bucket in bucket_set:
            known_kernel_time_histogram[bucket] = value
    values["kernel_time_histogram_ns"] = known_kernel_time_histogram
    return values


def _scheduler_metric_values(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the scheduler-fairness capability the application collected."""

    data = raw if isinstance(raw, Mapping) else {}
    policy = data.get("policy")
    return {
        "policy": "unavailable" if policy is None else str(policy),
        "compatible_sampling_coalescing": bool(
            data.get("compatible_sampling_coalescing", False)
        ),
        "continuous_decode": bool(data.get("continuous_decode", False)),
        "preemptive_fairness": bool(data.get("preemptive_fairness", False)),
    }


def _render_prometheus_metrics(
    metrics: _ServerMetrics,
    *,
    snapshot: _MetricsSnapshot,
) -> str:
    """Render the Prometheus text exposition for one collected snapshot."""

    resident = snapshot.resident
    pool = snapshot.pool
    graph = snapshot.graph
    queue = snapshot.queue
    values = {
        "hipengine_requests_total": metrics.request_total,
        "hipengine_request_completed_total": metrics.request_completed_total,
        "hipengine_request_failed_total": metrics.request_failed_total,
        "hipengine_request_rejected_total": metrics.request_rejected_total,
        "hipengine_request_cancelled_total": metrics.request_cancelled_total,
        "hipengine_prompt_tokens_total": metrics.prompt_tokens_total,
        "hipengine_completion_tokens_total": metrics.completion_tokens_total,
        "hipengine_resident_requests_pending": resident["requests"]["pending"],
        "hipengine_resident_requests_admitted": resident["requests"]["admitted"],
        "hipengine_resident_requests_active": resident["requests"]["active"],
        "hipengine_resident_requests_admitted_total": resident["requests"]["admitted_total"],
        "hipengine_resident_requests_reclaimed_total": resident["requests"]["reclaimed_total"],
        "hipengine_resident_bucket_capacity": resident["bucket"]["capacity"],
        "hipengine_resident_bucket_active_rows": resident["bucket"]["active_rows"],
        "hipengine_resident_bucket_occupied_slots": resident["bucket"]["occupied_slots"],
        "hipengine_resident_bucket_free_slots": resident["bucket"]["free_slots"],
        "hipengine_resident_bucket_occupancy_ratio": resident["bucket"]["occupancy_ratio"],
        "hipengine_resident_work_prefill_total": resident["work"]["prefill"],
        "hipengine_resident_work_decode_total": resident["work"]["decode"],
        "hipengine_resident_work_reclaim_total": resident["work"]["reclaim"],
        "hipengine_resident_prefill_chunk_tokens": resident["policy"]["prefill_chunk_tokens"],
        "hipengine_resident_fair_prefill_burst_chunks": resident["policy"]["fair_prefill_burst_chunks"],
        "hipengine_resident_consecutive_prefill_chunks": resident["policy"]["consecutive_prefill_chunks"],
        "hipengine_resident_packed_workspace_current_bytes": resident["packed_workspace"]["current_bytes"],
        "hipengine_resident_packed_workspace_owner_sessions": resident["packed_workspace"]["owner_sessions"],
        "hipengine_resident_packed_workspace_leased_pool_bytes": resident["packed_workspace"]["leased_pool_bytes"],
        "hipengine_resident_prefill_oracle_owner_bytes": resident["prefill_transients"]["oracle_owner_bytes"],
        "hipengine_resident_prefill_oracle_owners": resident["prefill_transients"]["oracle_owner_count_total"],
        "hipengine_resident_prefill_oracle_observed_peak_bytes": resident["prefill_transients"]["oracle_observed_peak_bytes"],
        "hipengine_resident_prefill_oracle_observed_peak_owners": resident["prefill_transients"]["oracle_observed_peak_owners"],
        "hipengine_resident_prefill_hidden_owner_bytes": resident["prefill_transients"]["hidden_and_bulk_owner_bytes"],
        "hipengine_resident_packed_workspace_release_events_total": resident["packed_workspace"]["release_events"],
        "hipengine_resident_packed_workspace_released_bytes_total": resident["packed_workspace"]["released_bytes"],
        "hipengine_resident_kv_int8_payload_bytes": resident["persistent_kv"]["int8_payload_bytes"],
        "hipengine_resident_kv_bf16_payload_bytes": resident["persistent_kv"]["bf16_payload_bytes"],
        "hipengine_resident_kv_scale_bytes": resident["persistent_kv"]["scale_bytes"],
        "hipengine_resident_kv_bf16_mirror_bytes": resident["persistent_kv"]["bf16_mirror_bytes"],
        "hipengine_resident_kv_total_bytes": resident["persistent_kv"]["total_bytes"],
        "hipengine_kv_pool_current_bytes": pool["current_bytes"],
        "hipengine_kv_pool_high_water_observed_bytes": pool["high_water_observed_bytes"],
        "hipengine_kv_pool_grow_events_total": pool["grow_events"],
        "hipengine_kv_pool_grow_failures_total": pool["grow_failures"],
        "hipengine_kv_pool_shrink_events_total": pool["shrink_events"],
        "hipengine_kv_pool_current_pages": pool["current_pages"],
        "hipengine_kv_pool_high_water_observed_pages": pool["high_water_observed_pages"],
        "hipengine_kv_pool_free_pages": pool["free_pages"],
        "hipengine_kv_pool_refcounted_pages": pool["refcounted_pages"],
        "hipengine_kv_pool_pinned_pages": pool["pinned_pages"],
        "hipengine_kv_pool_max_pages": pool["max_pages"],
        "hipengine_kv_pool_budget_bytes": pool["budget_bytes"],
        "hipengine_kv_pool_retired_pages_total": pool["retired_pages"],
        "hipengine_kv_pool_retired_bytes_total": pool["retired_bytes"],
        "hipengine_graph_bucket_entries": graph["entries"],
        "hipengine_graph_bucket_hits_total": graph["hits"],
        "hipengine_graph_bucket_misses_total": graph["misses"],
        "hipengine_graph_bucket_replay_hit_rate": graph["replay_hit_rate"],
        "hipengine_graph_bucket_captures_total": graph["captures"],
        "hipengine_graph_bucket_replays_total": graph["replays"],
        "hipengine_graph_bucket_invalidations_total": graph["invalidations"],
        "hipengine_generation_queue_depth": queue["depth"],
        "hipengine_generation_queue_max_depth": queue["max_depth"],
        "hipengine_generation_worker_active": queue["worker_active"],
        "hipengine_generation_requests_active": queue["active_requests"],
        "hipengine_generation_requests_max_active": queue["max_active_requests"],
        "hipengine_chat_sessions_active": (
            0.0 if snapshot.chat_sessions_active is None else float(snapshot.chat_sessions_active)
        ),
        "hipengine_chat_sessions_pending": (
            0.0 if snapshot.chat_sessions_pending is None else float(snapshot.chat_sessions_pending)
        ),
        "hipengine_chat_sessions_max_active": (
            0.0
            if snapshot.chat_sessions_max_active is None
            else float(snapshot.chat_sessions_max_active)
        ),
        "hipengine_mtp_requests_total": metrics.mtp_requests_total,
        "hipengine_mtp_draft_tokens_accepted_total": metrics.mtp_draft_tokens_accepted_total,
        "hipengine_mtp_draft_tokens_rejected_total": metrics.mtp_draft_tokens_rejected_total,
        "hipengine_mtp_draft_acceptance_rate": _mtp_acceptance_rate_metric(metrics),
        "hipengine_mtp_serving": float(snapshot.mtp_serving),
    }
    help_text = {
        "hipengine_requests_total": "Total generation requests observed by the server.",
        "hipengine_request_completed_total": "Generation requests that completed successfully.",
        "hipengine_request_failed_total": "Generation requests that failed after reaching generation validation.",
        "hipengine_request_rejected_total": "Generation requests rejected by admission/backpressure.",
        "hipengine_request_cancelled_total": "Generation requests cancelled after client disconnect or backend cancellation.",
        "hipengine_prompt_tokens_total": "Prompt tokens counted for successful requests.",
        "hipengine_completion_tokens_total": "Completion tokens counted for successful requests.",
        "hipengine_resident_requests_pending": "Current requests pending resident-loop admission.",
        "hipengine_resident_requests_admitted": "Current requests admitted to physical resident slots.",
        "hipengine_resident_requests_active": "Current active resident-loop requests.",
        "hipengine_resident_requests_admitted_total": "Cumulative resident-loop admissions.",
        "hipengine_resident_requests_reclaimed_total": "Cumulative resident-loop reclaims.",
        "hipengine_resident_bucket_capacity": "Physical resident bucket capacity.",
        "hipengine_resident_bucket_active_rows": "Current active rows in the physical resident bucket.",
        "hipengine_resident_bucket_occupied_slots": "Current occupied physical resident slots.",
        "hipengine_resident_bucket_free_slots": "Current free physical resident slots.",
        "hipengine_resident_bucket_occupancy_ratio": "Current occupied/capacity ratio for the resident bucket.",
        "hipengine_resident_work_prefill_total": "Executed resident prefill work items.",
        "hipengine_resident_work_decode_total": "Executed resident decode work items.",
        "hipengine_resident_work_reclaim_total": "Executed resident reclaim transitions.",
        "hipengine_resident_prefill_chunk_tokens": "Configured maximum tokens in one resident prefill work item.",
        "hipengine_resident_fair_prefill_burst_chunks": "Configured maximum consecutive prefill chunks while fair scheduling also has decode work.",
        "hipengine_resident_consecutive_prefill_chunks": "Current consecutive resident prefill chunks since the last decode work item.",
        "hipengine_resident_packed_workspace_current_bytes": "Current owner-only packed GGUF workspace bytes (owner-deduplicated: shared slot views counted once, split-growth owners included).",
        "hipengine_resident_packed_workspace_owner_sessions": "Resident sessions whose views contributed to the packed workspace inventory (aliasing diagnostic).",
        "hipengine_resident_packed_workspace_leased_pool_bytes": "Pool-plane bytes pinned by workspace leases (a distinct accounting domain from workspace allocation bytes).",
        "hipengine_resident_prefill_oracle_owner_bytes": "Live BF16 prefill oracle pair bytes across resident sessions (per-layer during multi-slab packed calls).",
        "hipengine_resident_prefill_oracle_owners": "Live BF16 prefill oracle owners (pairs) across resident sessions.",
        "hipengine_resident_prefill_oracle_observed_peak_bytes": "Monotonic while-live peak of BF16 prefill oracle pair bytes, sampled at release time (scrapes cannot observe mid-call).",
        "hipengine_resident_prefill_oracle_observed_peak_owners": "Monotonic while-live peak of BF16 prefill oracle owners, sampled at release time.",
        "hipengine_resident_prefill_hidden_owner_bytes": "Bulk prefill hidden/scratch/token owner bytes across resident sessions.",
        "hipengine_resident_packed_workspace_release_events_total": "Reclaimed packed GGUF workspace owners.",
        "hipengine_resident_packed_workspace_released_bytes_total": "Cumulative owner-only packed GGUF workspace bytes reclaimed.",
        "hipengine_resident_kv_int8_payload_bytes": "Current resident request-owned INT8 KV payload bytes.",
        "hipengine_resident_kv_bf16_payload_bytes": "Current resident request-owned BF16 KV payload bytes.",
        "hipengine_resident_kv_scale_bytes": "Current resident request-owned KV scale bytes.",
        "hipengine_resident_kv_bf16_mirror_bytes": "Current resident request-owned persistent BF16 mirror bytes.",
        "hipengine_resident_kv_total_bytes": "Current resident request-owned persistent KV bytes.",
        "hipengine_kv_pool_current_bytes": "Current dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_high_water_observed_bytes": "Peak observed dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_grow_events_total": "Dynamic KV pool grow events, or 0 when unavailable.",
        "hipengine_kv_pool_grow_failures_total": "Dynamic KV pool grow failures, or 0 when unavailable.",
        "hipengine_kv_pool_shrink_events_total": "Dynamic KV pool shrink events, or 0 when unavailable.",
        "hipengine_kv_pool_current_pages": "Current dynamic KV pool pages, or 0 when unavailable.",
        "hipengine_kv_pool_high_water_observed_pages": "Peak observed dynamic KV pool pages, or 0 when unavailable.",
        "hipengine_kv_pool_free_pages": "Current dynamic KV pool free pages, or 0 when unavailable.",
        "hipengine_kv_pool_refcounted_pages": "Current dynamic KV pool refcounted pages, or 0 when unavailable.",
        "hipengine_kv_pool_pinned_pages": "Current graph-pinned dynamic KV pages, or 0 when unavailable.",
        "hipengine_kv_pool_max_pages": "Configured maximum dynamic KV pool pages, or 0 when unavailable.",
        "hipengine_kv_pool_budget_bytes": "Configured maximum dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_retired_pages_total": "Cumulative dynamic KV pool pages returned to the device allocator, or 0 when unavailable.",
        "hipengine_kv_pool_retired_bytes_total": "Cumulative dynamic KV pool bytes returned to the device allocator, or 0 when unavailable.",
        "hipengine_graph_bucket_entries": "Current graph bucket cache entries, or 0 when unavailable.",
        "hipengine_graph_bucket_hits_total": "Graph bucket cache hits, or 0 when unavailable.",
        "hipengine_graph_bucket_misses_total": "Graph bucket cache misses, or 0 when unavailable.",
        "hipengine_graph_bucket_replay_hit_rate": "Graph bucket replay hit rate, or 0 when unavailable.",
        "hipengine_graph_bucket_captures_total": "Resident graph captures across all physical buckets.",
        "hipengine_graph_bucket_replays_total": "Resident graph replay calls across all physical buckets.",
        "hipengine_graph_bucket_invalidations_total": "Resident graph invalidations across all physical buckets.",
        "hipengine_generation_queue_depth": "Current generation-batcher queue depth.",
        "hipengine_generation_queue_max_depth": "Configured generation-batcher queue cap, or 0 when unset.",
        "hipengine_generation_worker_active": "Whether the generation-batcher worker is active, as 0 or 1.",
        "hipengine_generation_requests_active": "Current number of HTTP requests in the active backend batch.",
        "hipengine_generation_requests_max_active": "Configured active backend request cap, or 0 when unset.",
        "hipengine_chat_sessions_active": "Current app-local chat transcript session count.",
        "hipengine_chat_sessions_pending": "Current app-local chat session creations in flight.",
        "hipengine_chat_sessions_max_active": "Configured app-local chat session cap, or 0 when unset.",
        "hipengine_mtp_requests_total": "Completed requests that served through the speculative MTP route.",
        "hipengine_mtp_draft_tokens_accepted_total": "Draft tokens proposed by MTP that the target accepted.",
        "hipengine_mtp_draft_tokens_rejected_total": "Draft tokens proposed by MTP that the target rejected.",
        "hipengine_mtp_draft_acceptance_rate": "Cumulative MTP draft acceptance rate (accepted/(accepted+rejected)).",
        "hipengine_mtp_serving": "Whether the server's effective MTP serving route is enabled (1) or off (0).",
    }
    counter_names = {
        "hipengine_requests_total",
        "hipengine_request_completed_total",
        "hipengine_request_failed_total",
        "hipengine_request_rejected_total",
        "hipengine_request_cancelled_total",
        "hipengine_prompt_tokens_total",
        "hipengine_completion_tokens_total",
        "hipengine_resident_requests_admitted_total",
        "hipengine_resident_requests_reclaimed_total",
        "hipengine_resident_work_prefill_total",
        "hipengine_resident_work_decode_total",
        "hipengine_resident_work_reclaim_total",
        "hipengine_resident_packed_workspace_release_events_total",
        "hipengine_resident_packed_workspace_released_bytes_total",
        "hipengine_kv_pool_grow_events_total",
        "hipengine_kv_pool_grow_failures_total",
        "hipengine_kv_pool_shrink_events_total",
        "hipengine_graph_bucket_hits_total",
        "hipengine_graph_bucket_misses_total",
        "hipengine_graph_bucket_captures_total",
        "hipengine_graph_bucket_replays_total",
        "hipengine_graph_bucket_invalidations_total",
    }
    lines: list[str] = []
    for name, value in values.items():
        lines.append(f"# HELP {name} {help_text[name]}")
        lines.append(f"# TYPE {name} {'counter' if name in counter_names else 'gauge'}")
        lines.append(f"{name} {_format_metric_value(value)}")
    scheduler = snapshot.scheduler_fairness
    lines.append("# HELP hipengine_generation_scheduler_fairness_policy_info Generation scheduler fairness policy.")
    lines.append("# TYPE hipengine_generation_scheduler_fairness_policy_info gauge")
    lines.append(
        "hipengine_generation_scheduler_fairness_policy_info{"
        f'policy="{_escape_prometheus_label_value(str(scheduler["policy"]))}",'
        f'compatible_sampling_coalescing="{str(bool(scheduler["compatible_sampling_coalescing"])).lower()}",'
        f'continuous_decode="{str(bool(scheduler["continuous_decode"])).lower()}",'
        f'preemptive_fairness="{str(bool(scheduler["preemptive_fairness"])).lower()}"'
        "} 1"
    )
    lines.append("# HELP hipengine_resident_bucket_info Resident physical bucket and scheduler policy.")
    lines.append("# TYPE hipengine_resident_bucket_info gauge")
    lines.append(
        "hipengine_resident_bucket_info{"
        f'active_mask="{_escape_prometheus_label_value(str(resident["bucket"]["active_mask"]))}",'
        f'fair_prefill_burst_chunks="{int(resident["policy"]["fair_prefill_burst_chunks"])}",'
        f'last_work_kind="{_escape_prometheus_label_value(str(resident["policy"]["last_work_kind"]))}",'
        f'policy="{_escape_prometheus_label_value(str(resident["policy"]["name"]))}"'
        "} 1"
    )
    _append_resident_latency_metrics(lines, resident["latency"])
    graph_buckets = graph.get("buckets", {})
    for graph_stat, metric_name in (
        ("entries", "hipengine_graph_bucket_entries_by_bucket"),
        ("captures", "hipengine_graph_bucket_captures_by_bucket_total"),
        ("hits", "hipengine_graph_bucket_hits_by_bucket_total"),
        ("replays", "hipengine_graph_bucket_replays_by_bucket_total"),
        ("invalidations", "hipengine_graph_bucket_invalidations_by_bucket_total"),
    ):
        bucket_values = {
            str(bucket): _non_negative_metric_value(row.get(graph_stat))
            for bucket, row in graph_buckets.items()
            if isinstance(row, Mapping)
        }
        if graph_stat == "entries":
            _append_labeled_gauge_metrics(
                lines,
                metric_name,
                "Current resident graph entries by stable physical bucket.",
                "bucket",
                bucket_values,
            )
        else:
            _append_labeled_counter_metrics(
                lines,
                metric_name,
                f"Resident graph {graph_stat} by stable physical bucket.",
                "bucket",
                bucket_values,
            )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_prefill_executor_mode_total",
        "Packed-prefill executor that ran per resident session, by executor mode.",
        "mode",
        resident["prefill_transients"]["executor_modes"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_kv_attention_source_total",
        "Resident session KV attention source, which gates the resumable prefill.",
        "source",
        resident["prefill_transients"]["kv_attention_sources"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_route_total",
        "Resident GGUF execution transitions by declared route.",
        "route",
        resident["route_counts"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_fallback_total",
        "Resident GGUF explicit fallbacks by reason.",
        "reason",
        resident["fallback_reasons"],
    )
    manifest = resident["last_execution_manifest"]
    if manifest:
        lines.append("# HELP hipengine_resident_route_manifest_info Last resident execution manifest identity.")
        lines.append("# TYPE hipengine_resident_route_manifest_info gauge")
        lines.append(
            "hipengine_resident_route_manifest_info{"
            f'claim_level="{_escape_prometheus_label_value(str(manifest.get("claim_level") or "unavailable"))}",'
            f'kind="{_escape_prometheus_label_value(str(manifest.get("kind") or "unavailable"))}",'
            f'kv_attention_source="{_escape_prometheus_label_value(str(manifest.get("kv_attention_source") or "unavailable"))}",'
            f'logical_c="{_escape_prometheus_label_value(str(manifest.get("logical_c") or manifest.get("rows") or 0))}",'
            f'mode="{_escape_prometheus_label_value(str(manifest.get("mode") or "unavailable"))}",'
            f'physical_execution_width="{_escape_prometheus_label_value(str(manifest.get("physical_execution_width") or manifest.get("physical_rows") or 0))}",'
            f'rows="{_escape_prometheus_label_value(str(manifest.get("rows") or 0))}",'
            f'serial_decode_fallback="{str(bool(manifest.get("serial_decode_fallback", False))).lower()}",'
            f'throughput_claim_eligible="{str(bool(manifest.get("throughput_claim_eligible", False))).lower()}"'
            "} 1"
        )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_graph_bucket_miss_reason_total",
        "Graph bucket cache misses by reason, or empty when unavailable.",
        "reason",
        graph["miss_reasons"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_graph_bucket_kernel_time_bucket_total",
        "Graph bucket kernel-time observations by duration bucket, or empty when unavailable.",
        "bucket",
        graph["kernel_time_histogram_ns"],
    )
    return "\n".join(lines) + "\n"
