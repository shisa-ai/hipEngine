#!/usr/bin/env python3
"""Replay/qprofile qwen35moe selected-MoE kernels outside full-model noise.

P9.C2 tooling: this script runs one real qwen35moe bulk prefill, intercepts the
compact selected-MoE WMMA helper, records the real expert routing/tile maps, and
replays the two heavy selected kernels per MoE layer:

* Q4_K dual gate+up selected WMMA
* Q5_K/Q6_K down selected WMMA

The replay uses the exact resident raw GGUF-K weight pointers and compact
scheduler buffers produced by the live prefill, so timings are representative of
current runtime dispatch without requiring a separate fixture format.  It emits a
compact JSON report that downstream P9.C3/P9.C4 work can use to choose hot-expert
thresholds and validate microbench/prototype correlation with rocprof buckets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_prefill import (
    gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out,
    selected_wmma_prefill_compact_default_tiles,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill import (
    gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out,
    gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out,
    q4_k_predecode_scale_min_sidemeta,
    selected_dual_wmma_prefill_compact_default_tiles,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_t16_selected_prefill import (
    gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out,
)
import hipengine.runtime.qwen35_gguf_runner as qgr
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_REFERENCE_ARTIFACT = Path(
    "benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c1-wmma-tile-sweep-blocked.json"
)
THRESHOLDS = (16, 32, 64, 128)


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text().strip()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _git_status() -> str | None:
    try:
        return subprocess.check_output(["git", "status", "-sb"], text=True).strip()
    except Exception:
        return None


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def summarize_counts(counts: np.ndarray, thresholds: tuple[int, ...] = THRESHOLDS) -> dict[str, Any]:
    """Summarize one layer's per-expert compact row counts."""

    counts = np.asarray(counts, dtype=np.int64)
    nonzero = counts[counts > 0]
    summary: dict[str, Any] = {
        "experts": int(counts.size),
        "compact_rows": int(counts.sum()),
        "nonzero_experts": int(nonzero.size),
        "nonzero_fraction": float(nonzero.size / counts.size) if counts.size else 0.0,
        "max_rows_per_expert": int(nonzero.max()) if nonzero.size else 0,
        "nonzero_p50": _percentile(nonzero, 50),
        "nonzero_p90": _percentile(nonzero, 90),
        "nonzero_p99": _percentile(nonzero, 99),
    }
    for threshold in thresholds:
        summary[f"experts_ge_{threshold}"] = int((counts >= threshold).sum())
        summary[f"rows_in_experts_ge_{threshold}"] = int(counts[counts >= threshold].sum())
    return summary


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate replay records by kernel/bucket and quant."""

    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"layers": 0, "total_ms": 0.0})
    for record in records:
        gate_ms = float(record["timings_ms"]["gate_up_avg"])
        down_ms = float(record["timings_ms"]["down_avg"])
        totals["q4_dual_gate_up"]["layers"] += 1
        totals["q4_dual_gate_up"]["total_ms"] += gate_ms
        down_key = f"{record['down_quant']}_down"
        totals[down_key]["layers"] += 1
        totals[down_key]["total_ms"] += down_ms
    for value in totals.values():
        layers = int(value["layers"])
        value["avg_ms"] = float(value["total_ms"] / layers) if layers else 0.0
    selected_total = sum(float(value["total_ms"]) for value in totals.values())
    return {"by_component": dict(totals), "selected_moe_total_ms": selected_total}


def _load_reference_buckets(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text())
    try:
        buckets = data["measurements"]["rocprof_summary_512_0"]["phases"]["prefill"]["buckets"]
    except KeyError:
        return {}
    out: dict[str, float] = {}
    for bucket in buckets:
        name = bucket.get("bucket")
        if name:
            out[str(name)] = float(bucket.get("total_ms", 0.0))
    return out


def _correlate_with_reference(aggregate: dict[str, Any], reference_buckets: dict[str, float]) -> dict[str, Any]:
    q4_ref = reference_buckets.get("moe_q4_k_selected_dual_wmma_prefill")
    q5_ref = reference_buckets.get("moe_q5_k_selected_wmma_prefill")
    q6_ref = reference_buckets.get("moe_q6_k_selected_wmma_prefill")
    components = aggregate.get("by_component", {})
    pairs = {
        "q4_dual_gate_up": (components.get("q4_dual_gate_up", {}).get("total_ms"), q4_ref),
        "gguf_q5_k_down": (components.get("gguf_q5_k_down", {}).get("total_ms"), q5_ref),
        "gguf_q6_k_down": (components.get("gguf_q6_k_down", {}).get("total_ms"), q6_ref),
    }
    details: dict[str, Any] = {}
    replay_selected = 0.0
    ref_selected = 0.0
    for name, (replay, ref) in pairs.items():
        if replay is None or ref is None:
            continue
        replay_f = float(replay)
        ref_f = float(ref)
        replay_selected += replay_f
        ref_selected += ref_f
        details[name] = {
            "replay_ms": replay_f,
            "reference_rocprof_ms": ref_f,
            "delta_pct": ((replay_f / ref_f) - 1.0) * 100.0 if ref_f else None,
        }
    return {
        "components": details,
        "selected_moe_replay_ms": replay_selected,
        "selected_moe_reference_rocprof_ms": ref_selected,
        "selected_moe_delta_pct": ((replay_selected / ref_selected) - 1.0) * 100.0 if ref_selected else None,
    }


@dataclass
class ReplayRecorder:
    warmup_iters: int
    replay_iters: int
    sample_groups: int
    records: list[dict[str, Any]] = field(default_factory=list)

    def time_kernel(self, runtime: HipRuntime, stream: int, fn: Callable[[], None]) -> dict[str, Any]:
        if self.replay_iters <= 0:
            raise ValueError("replay_iters must be positive")
        for _ in range(max(0, self.warmup_iters)):
            fn()
        if stream:
            runtime.stream_synchronize(stream)
        else:
            runtime.device_synchronize()
        samples: list[float] = []
        for _ in range(max(1, self.sample_groups)):
            t0 = time.perf_counter()
            for _ in range(self.replay_iters):
                fn()
            if stream:
                runtime.stream_synchronize(stream)
            else:
                runtime.device_synchronize()
            samples.append(((time.perf_counter() - t0) * 1000.0) / self.replay_iters)
        return {
            "samples": samples,
            "avg": float(statistics.fmean(samples)),
            "median": float(statistics.median(samples)),
            "min": float(min(samples)),
            "max": float(max(samples)),
            "sample_groups": int(max(1, self.sample_groups)),
            "launches_per_group": int(self.replay_iters),
        }


def _copy_i64_buffer(buffer, count: int, *, runtime: HipRuntime, stream: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative")
    host = np.empty((count,), dtype=np.int64)
    if stream:
        runtime.stream_synchronize(stream)
    copy_device_to_host(host_array_ptr(host), buffer, host.nbytes, runtime=runtime)
    return host


def _layer_index_from_name(name: str) -> int | None:
    match = re.search(r"(?:blk|layers?)\.(\d+)", name)
    return int(match.group(1)) if match else None


def _install_replay_helper(recorder: ReplayRecorder):
    original = qgr._try_run_post_attention_moe_rows_compact_wmma

    def replay_helper(
        runner,
        layer,
        gate_weight,
        up_weight,
        down_weight,
        out_ptr,
        scratch,
        *,
        rows: int,
        selected_rows: int,
        top_k: int,
        stream: int,
        runtime: HipRuntime,
    ) -> bool:
        if not qgr.gguf_wmma_prefill_enabled(None):
            return False
        if not qgr._scratch_has_compact_moe_fields(scratch):
            return False
        cfg = runner.weights.config if runner.weights is not None else None
        if cfg is None:
            return False
        kernels = qgr._resolve_compact_moe_wmma_kernels(gate_weight, up_weight, down_weight)
        if kernels is None:
            return False
        gate_up_fn, down_fn = kernels
        num_experts = int(cfg.expert_count)
        hidden_size = int(runner.hidden_size)
        expert_ffn = int(cfg.expert_feed_forward_length)
        if selected_rows <= 0 or selected_rows > int(getattr(scratch, "moe_selected_rows_capacity", selected_rows)):
            return False
        if hidden_size % 256 != 0 or expert_ffn % 256 != 0 or expert_ffn % 16 != 0:
            return False
        qgr._validate_raw_rank3_expert_weight(
            gate_weight, num_experts=num_experts, in_features=hidden_size, out_features=expert_ffn
        )
        qgr._validate_raw_rank3_expert_weight(
            up_weight, num_experts=num_experts, in_features=hidden_size, out_features=expert_ffn
        )
        qgr._validate_raw_rank3_expert_weight(
            down_weight, num_experts=num_experts, in_features=expert_ffn, out_features=hidden_size
        )

        qgr._zero(runtime, scratch.moe_group_counts, scratch.moe_group_counts_zero)
        qgr.qwen35_moe_group_count(
            scratch.moe_selected_experts.ptr,
            scratch.moe_group_counts.ptr,
            selected_rows,
            num_experts,
            stream=stream,
            runtime=runtime,
        )
        qgr.qwen35_moe_group_prefix(
            scratch.moe_group_counts.ptr,
            scratch.moe_padded_counts.ptr,
            scratch.moe_expert_start_compact.ptr,
            scratch.moe_total_compact.ptr,
            num_experts,
            1,
            stream=stream,
            runtime=runtime,
        )
        qgr._zero(runtime, scratch.moe_scatter_offsets, scratch.moe_scatter_offsets_zero)
        qgr.qwen35_moe_group_scatter_gather_lowp(
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            scratch.moe_expert_start_compact.ptr,
            scratch.moe_scatter_offsets.ptr,
            scratch.moe_sorted_lanes.ptr,
            scratch.moe_sorted_experts.ptr,
            scratch.moe_sorted_weights.ptr,
            scratch.moe_down_out.ptr,
            selected_rows,
            num_experts,
            top_k,
            hidden_size,
            stream=stream,
            runtime=runtime,
        )
        qgr.qwen35_moe_wmma_tile_map(
            scratch.moe_expert_start_compact.ptr,
            scratch.moe_expert_start_wmma.ptr,
            scratch.moe_tile_expert.ptr,
            scratch.moe_wmma_total.ptr,
            num_experts,
            stream=stream,
            runtime=runtime,
        )
        wmma_total_rows = qgr._read_i64_device_scalar(
            scratch.moe_wmma_total,
            scratch.moe_wmma_total_host,
            stream=stream,
            runtime=runtime,
        )
        if wmma_total_rows <= 0 or wmma_total_rows > int(getattr(scratch, "moe_wmma_rows_capacity", wmma_total_rows)):
            return False

        compact_starts = _copy_i64_buffer(scratch.moe_expert_start_compact, num_experts + 1, runtime=runtime, stream=stream)
        wmma_starts = _copy_i64_buffer(scratch.moe_expert_start_wmma, num_experts + 1, runtime=runtime, stream=stream)
        tile_count = int(wmma_total_rows // 16)
        tile_expert = _copy_i64_buffer(scratch.moe_tile_expert, tile_count, runtime=runtime, stream=stream)
        counts = np.diff(compact_starts)
        padded_counts = np.diff(wmma_starts)
        compact_rows = int(compact_starts[-1])
        padding_rows = int(wmma_total_rows - compact_rows)

        gate_tile_m, gate_tile_n = selected_dual_wmma_prefill_compact_default_tiles()
        down_tile_m, down_tile_n = selected_wmma_prefill_compact_default_tiles(down_weight.spec.quant_key)
        use_hot_q4 = int(getattr(recorder, "q4_hot_fulltile_threshold", 0)) > 0
        hot_threshold = int(getattr(recorder, "q4_hot_fulltile_threshold", 0))
        layer_order = len(recorder.records)
        use_q5_opt = bool(getattr(recorder, "q5_opt", False)) and down_weight.spec.quant_key == "gguf_q5_k"
        use_sidemeta_q4 = int(getattr(recorder, "q4_sidemeta_layers", 0)) > layer_order
        sidemeta_bufs = []
        tile16_bufs = []
        tile16_materialized_bytes = 0
        use_tile16_wmma = int(getattr(recorder, "q4_tile16_wmma_layers", 0)) > layer_order
        use_tile16_materialize = use_tile16_wmma or int(getattr(recorder, "q4_tile16_materialize_layers", 0)) > layer_order
        if use_sidemeta_q4:
            reader = getattr(recorder, "q4_sidemeta_reader")
            for weight in (gate_weight, up_weight):
                raw = reader.tensor_data(weight.spec.source.name)
                side = q4_k_predecode_scale_min_sidemeta(raw)
                buf = malloc(side.nbytes, runtime=runtime)
                copy_host_to_device(buf, host_array_ptr(side), side.nbytes, runtime=runtime)
                sidemeta_bufs.append(buf)
        if use_tile16_materialize:
            reader = getattr(recorder, "q4_tile16_reader")
            for weight in (gate_weight, up_weight):
                raw = reader.tensor_data(weight.spec.source.name)
                packed = repack_gguf_q4_k_tile16(raw)
                tile16_materialized_bytes += int(packed.tiles.nbytes)
                buf = malloc(packed.tiles.nbytes, runtime=runtime)
                copy_host_to_device(buf, host_array_ptr(packed.tiles), packed.tiles.nbytes, runtime=runtime)
                tile16_bufs.append(buf)

        def launch_gate_up() -> None:
            if use_tile16_wmma:
                gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_bf16_bf16_out(
                    scratch.moe_down_out.ptr,
                    scratch.moe_expert_start_compact.ptr,
                    scratch.moe_expert_start_wmma.ptr,
                    scratch.moe_tile_expert.ptr,
                    tile16_bufs[0].ptr,
                    tile16_bufs[1].ptr,
                    scratch.ffn_gate_up.ptr,
                    selected_rows,
                    hidden_size,
                    expert_ffn,
                    expert_ffn,
                    num_experts,
                    wmma_total_rows,
                    stream=stream,
                    runtime=runtime,
                )
                return
            if use_sidemeta_q4:
                gguf_q4_k_selected_dual_wmma_prefill_compact_sidemeta_bf16_bf16_out(
                    scratch.moe_down_out.ptr,
                    scratch.moe_expert_start_compact.ptr,
                    scratch.moe_expert_start_wmma.ptr,
                    scratch.moe_tile_expert.ptr,
                    gate_weight.allocation("raw").tensor.ptr,
                    up_weight.allocation("raw").tensor.ptr,
                    sidemeta_bufs[0].ptr,
                    sidemeta_bufs[1].ptr,
                    scratch.ffn_gate_up.ptr,
                    selected_rows,
                    hidden_size,
                    expert_ffn,
                    expert_ffn,
                    num_experts,
                    wmma_total_rows,
                    stream=stream,
                    runtime=runtime,
                )
                return
            if use_hot_q4:
                gguf_q4_k_selected_dual_wmma_prefill_compact_hot_fulltile_bf16_bf16_out(
                    scratch.moe_down_out.ptr,
                    scratch.moe_expert_start_compact.ptr,
                    scratch.moe_expert_start_wmma.ptr,
                    scratch.moe_tile_expert.ptr,
                    gate_weight.allocation("raw").tensor.ptr,
                    up_weight.allocation("raw").tensor.ptr,
                    scratch.ffn_gate_up.ptr,
                    selected_rows,
                    hidden_size,
                    expert_ffn,
                    expert_ffn,
                    num_experts,
                    wmma_total_rows,
                    hot_threshold=hot_threshold,
                    stream=stream,
                    runtime=runtime,
                )
                return
            gate_up_fn(
                scratch.moe_down_out.ptr,
                scratch.moe_expert_start_compact.ptr,
                scratch.moe_expert_start_wmma.ptr,
                scratch.moe_tile_expert.ptr,
                gate_weight.allocation("raw").tensor.ptr,
                up_weight.allocation("raw").tensor.ptr,
                scratch.ffn_gate_up.ptr,
                selected_rows,
                hidden_size,
                expert_ffn,
                expert_ffn,
                num_experts,
                wmma_total_rows,
                stream=stream,
                runtime=runtime,
            )

        gate_stats = recorder.time_kernel(runtime, stream, launch_gate_up)
        if sidemeta_bufs or tile16_bufs:
            if stream:
                runtime.stream_synchronize(stream)
            for buf in reversed(sidemeta_bufs):
                free(buf, runtime=runtime)
            for buf in reversed(tile16_bufs):
                free(buf, runtime=runtime)
        qgr.silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.ffn_intermediate.ptr,
            rows=selected_rows,
            features=expert_ffn,
            stream=stream,
            runtime=runtime,
        )

        def launch_down() -> None:
            fn = gguf_q5_k_selected_wmma_prefill_compact_opt_bf16_bf16_out if use_q5_opt else down_fn
            fn(
                scratch.ffn_intermediate.ptr,
                scratch.moe_expert_start_compact.ptr,
                scratch.moe_expert_start_wmma.ptr,
                scratch.moe_tile_expert.ptr,
                down_weight.allocation("raw").tensor.ptr,
                scratch.moe_down_out.ptr,
                selected_rows,
                expert_ffn,
                hidden_size,
                num_experts,
                wmma_total_rows,
                stream=stream,
                runtime=runtime,
            )

        down_stats = recorder.time_kernel(runtime, stream, launch_down)

        qgr.weighted_lanes_sum_out_bf16_f32w(
            scratch.moe_down_out.ptr,
            scratch.moe_sorted_weights.ptr,
            scratch.moe_sorted_lanes.ptr,
            scratch.moe_lane_to_row.ptr,
            scratch.ffn_down.ptr,
            rows,
            top_k,
            hidden_size,
            stream=stream,
            runtime=runtime,
        )

        shared_ffn = int(cfg.expert_shared_feed_forward_length)
        if qgr.launch_gguf_linear_pair_concat(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr,
            rows=rows,
            in_features=hidden_size,
            out_features=shared_ffn,
            stream=stream,
            runtime=runtime,
        ):
            qgr.silu_mul_dual_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=rows,
                features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
        else:
            if not qgr.launch_gguf_linear_pair(
                layer.weight("ffn_gate_shexp"),
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                rows=rows,
                in_features=hidden_size,
                out_features=shared_ffn,
                stream=stream,
                runtime=runtime,
            ):
                qgr.launch_gguf_linear(
                    layer.weight("ffn_gate_shexp"),
                    scratch.post_norm.ptr,
                    scratch.moe_shared_gate.ptr,
                    rows=rows,
                    in_features=hidden_size,
                    out_features=shared_ffn,
                    stream=stream,
                    runtime=runtime,
                )
                qgr.launch_gguf_linear(
                    layer.weight("ffn_up_shexp"),
                    scratch.post_norm.ptr,
                    scratch.moe_shared_up.ptr,
                    rows=rows,
                    in_features=hidden_size,
                    out_features=shared_ffn,
                    stream=stream,
                    runtime=runtime,
                )
            qgr.silu_mul_separate_out_bf16(
                scratch.moe_shared_gate.ptr,
                scratch.moe_shared_up.ptr,
                scratch.moe_shared_intermediate.ptr,
                rows=rows,
                features=shared_ffn,
                stream=stream,
                runtime=runtime,
            )
        qgr.launch_gguf_linear(
            layer.weight("ffn_down_shexp"),
            scratch.moe_shared_intermediate.ptr,
            scratch.moe_shared_out.ptr,
            rows=rows,
            in_features=shared_ffn,
            out_features=hidden_size,
            stream=stream,
            runtime=runtime,
        )
        qgr.shared_gate_combine_residual_batch_out_bf16(
            scratch.ffn_down.ptr,
            scratch.moe_shared_out.ptr,
            scratch.moe_shared_gate_logits.ptr,
            scratch.residual.ptr,
            out_ptr,
            rows,
            hidden_size,
            1,
            stream=stream,
            runtime=runtime,
        )

        gate_name = str(gate_weight.spec.source.name)
        record = {
            "order": len(recorder.records),
            "layer_index": _layer_index_from_name(gate_name),
            "gate_weight": gate_name,
            "gate_quant": gate_weight.spec.quant_key,
            "up_quant": up_weight.spec.quant_key,
            "down_quant": down_weight.spec.quant_key,
            "rows": int(rows),
            "selected_rows": int(selected_rows),
            "top_k": int(top_k),
            "num_experts": int(num_experts),
            "hidden_size": int(hidden_size),
            "expert_ffn": int(expert_ffn),
            "compact_rows": compact_rows,
            "wmma_rows": int(wmma_total_rows),
            "wmma_tiles": tile_count,
            "padding_rows": padding_rows,
            "padding_pct": ((float(wmma_total_rows) / compact_rows) - 1.0) * 100.0 if compact_rows else 0.0,
            "tile_decisions": {
                "gate_up": {
                    "tile_m": int(gate_tile_m),
                    "tile_n": int(gate_tile_n),
                    "hot_fulltile_threshold": hot_threshold if use_hot_q4 else None,
                    "sidemeta": bool(use_sidemeta_q4),
                    "tile16_materialized": bool(use_tile16_materialize),
                    "tile16_wmma": bool(use_tile16_wmma),
                    "tile16_materialized_bytes": tile16_materialized_bytes,
                },
                "down": {"tile_m": int(down_tile_m), "tile_n": int(down_tile_n), "q5_opt": use_q5_opt},
            },
            "counts_summary": summarize_counts(counts),
            "padded_counts_summary": summarize_counts(padded_counts),
            "counts": counts.astype(int).tolist(),
            "padded_counts": padded_counts.astype(int).tolist(),
            "tile_expert_histogram": np.bincount(tile_expert.clip(min=0), minlength=num_experts).astype(int).tolist(),
            "timings_ms": {
                "gate_up_samples": gate_stats["samples"],
                "gate_up_avg": gate_stats["avg"],
                "gate_up_median": gate_stats["median"],
                "down_samples": down_stats["samples"],
                "down_avg": down_stats["avg"],
                "down_median": down_stats["median"],
            },
        }
        recorder.records.append(record)
        return True

    qgr._try_run_post_attention_moe_rows_compact_wmma = replay_helper
    return original


def _restore_replay_helper(original) -> None:
    qgr._try_run_post_attention_moe_rows_compact_wmma = original


def run(args: argparse.Namespace) -> dict[str, Any]:
    compiler_version = _read_compiler_version(args.compiler_version_file)
    runtime = get_hip_runtime()
    prompt_tokens = [int(args.token_id)] * int(args.prompt_length)
    recorder = ReplayRecorder(
        warmup_iters=int(args.warmup_iters),
        replay_iters=int(args.replay_iters),
        sample_groups=int(args.sample_groups),
    )
    recorder.q4_hot_fulltile_threshold = int(args.q4_hot_fulltile_threshold)
    recorder.q4_sidemeta_layers = int(args.q4_sidemeta_layers)
    recorder.q4_sidemeta_reader = GGUFReader(args.model) if args.q4_sidemeta_layers else None
    recorder.q4_tile16_materialize_layers = int(args.q4_tile16_materialize_layers)
    recorder.q4_tile16_wmma_layers = int(args.q4_tile16_wmma_layers)
    recorder.q4_tile16_reader = GGUFReader(args.model) if (args.q4_tile16_materialize_layers or args.q4_tile16_wmma_layers) else None
    recorder.q5_opt = bool(args.q5_opt)
    original = _install_replay_helper(recorder)
    start = time.perf_counter()
    try:
        session = Qwen35GGUFResidentSession(
            args.model,
            runtime=runtime,
            compiler_version=compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=len(prompt_tokens) + 1,
            use_wmma_prefill=True,
        )
        try:
            sample = session.prefill(
                prompt_tokens,
                use_bulk=True,
                bulk_attention_mode=args.bulk_prefill_attention_mode,
                return_logits=False,
            )
            runtime.device_synchronize()
        finally:
            session.close()
    finally:
        _restore_replay_helper(original)
    elapsed = time.perf_counter() - start

    aggregate = aggregate_records(recorder.records)
    reference = _load_reference_buckets(args.reference_artifact)
    correlation = _correlate_with_reference(aggregate, reference)
    all_counts = np.asarray([count for rec in recorder.records for count in rec["counts"]], dtype=np.int64)
    all_padded = np.asarray([count for rec in recorder.records for count in rec["padded_counts"]], dtype=np.int64)
    hot_summary = summarize_counts(all_counts) if all_counts.size else {}
    padded_summary = summarize_counts(all_padded) if all_padded.size else {}
    total_compact = int(sum(int(rec["compact_rows"]) for rec in recorder.records))
    total_wmma = int(sum(int(rec["wmma_rows"]) for rec in recorder.records))

    return {
        "schema": "p9_c2_qwen35_moe_replay_v1",
        "model": str(args.model),
        "prompt_source": "repeated_token_id",
        "token_id": int(args.token_id),
        "prompt_length": int(args.prompt_length),
        "bulk_prefill_attention_mode": args.bulk_prefill_attention_mode,
        "compiler_version_file": str(args.compiler_version_file) if args.compiler_version_file else None,
        "require_cached_build": bool(args.require_cached_build),
        "warmup_iters": int(args.warmup_iters),
        "replay_iters": int(args.replay_iters),
        "sample_groups": int(args.sample_groups),
        "q4_hot_fulltile_threshold": int(args.q4_hot_fulltile_threshold),
        "q4_sidemeta_layers": int(args.q4_sidemeta_layers),
        "q4_tile16_materialize_layers": int(args.q4_tile16_materialize_layers),
        "q4_tile16_wmma_layers": int(args.q4_tile16_wmma_layers),
        "q5_opt": bool(args.q5_opt),
        "git_commit": _git_commit(),
        "git_status": _git_status(),
        "elapsed_seconds_including_model_load": elapsed,
        "sample_token_id": int(sample.token_id),
        "layer_count": len(recorder.records),
        "aggregate": aggregate,
        "reference_correlation": correlation,
        "routing_summary": {
            "total_compact_rows": total_compact,
            "total_wmma_rows": total_wmma,
            "total_padding_rows": total_wmma - total_compact,
            "total_padding_pct": ((total_wmma / total_compact) - 1.0) * 100.0 if total_compact else 0.0,
            "compact_counts_all_layers": hot_summary,
            "padded_counts_all_layers": padded_summary,
        },
        "records": recorder.records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--bulk-prefill-attention-mode", choices=("bulk", "native"), default="bulk")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--replay-iters", type=int, default=5, help="Kernel launches per timing sample group")
    parser.add_argument("--sample-groups", type=int, default=3, help="Number of timing sample groups per kernel")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--reference-artifact", type=Path, default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument(
        "--q4-hot-fulltile-threshold",
        type=int,
        default=0,
        help="Use the P9.C4 Q4 hot/full-tile prototype for gate+up when >0.",
    )
    parser.add_argument(
        "--q4-sidemeta-layers",
        type=int,
        default=0,
        help="Use the P9.C5 scale/min side-metadata prototype for the first N MoE layers.",
    )
    parser.add_argument(
        "--q5-opt",
        action="store_true",
        help="Use the P9.C7 optimized Q5_K selected-down prototype.",
    )
    parser.add_argument(
        "--q4-tile16-materialize-layers",
        type=int,
        default=0,
        help="Build/copy the P9.C13 Q4T16 repack prototype for the first N MoE layers without using it for compute.",
    )
    parser.add_argument(
        "--q4-tile16-wmma-layers",
        type=int,
        default=0,
        help="Use the P9.C14 Q4T16 selected-dual WMMA prototype for gate+up in the first N MoE layers.",
    )
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_length <= 1:
        raise ValueError("--prompt-length must be > 1 for selected-MoE replay")
    if args.warmup_iters < 0 or args.replay_iters <= 0 or args.sample_groups <= 0:
        raise ValueError("--warmup-iters must be >=0 and --replay-iters/--sample-groups must be >0")
    if args.q4_hot_fulltile_threshold < 0:
        raise ValueError("--q4-hot-fulltile-threshold must be >=0")
    if args.q4_sidemeta_layers < 0:
        raise ValueError("--q4-sidemeta-layers must be >=0")
    if args.q4_tile16_materialize_layers < 0:
        raise ValueError("--q4-tile16-materialize-layers must be >=0")
    if args.q4_tile16_wmma_layers < 0:
        raise ValueError("--q4-tile16-wmma-layers must be >=0")
    report = run(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    agg = report["aggregate"]
    corr = report.get("reference_correlation", {})
    print(f"wrote {args.json}")
    print(f"layers={report['layer_count']} selected_moe_replay_ms={agg['selected_moe_total_ms']:.3f}")
    if corr.get("selected_moe_reference_rocprof_ms"):
        print(
            "correlation: "
            f"replay={corr['selected_moe_replay_ms']:.3f} ms "
            f"reference={corr['selected_moe_reference_rocprof_ms']:.3f} ms "
            f"delta={corr['selected_moe_delta_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
