#!/usr/bin/env python3
"""Ingest qwen35 GGUF ``rocprofv3 --kernel-trace`` CSVs and emit a compact JSON.

P9.E1 tooling for the GGUF performance row format. The script is intentionally
read-only: it does **not** run a benchmark. Future P9 perf rows (tasks #19
P9.A3 and #26 P9.B7) point ``--csv`` (or ``--prefill-csv`` + ``--decode-csv``)
at the CSVs they already generate during their bench run; this helper turns
those CSVs into the canonical JSON shape the retained artifact format
expects.

Outputs (per phase):

* Per-kernel rankings (top-N): ``total_ms``, ``dispatches``,
  ``avg_dispatch_ms``, share of phase total, and rocprof resource metadata
  (VGPR/SGPR/scratch/LDS/workgroup size) when present.
* Per-bucket rollup: GGUF-aware bucket classifier (P8/P9 selected WMMA vs
  P9.B GEMV decode, legacy ``*_prefill_out_*`` family, GDN, router, full
  attention, etc.).
* Optional **back-calculated effective GB/s** per bucket, using the
  methodology in ``docs/ROOFLINE.md`` 12.4. Footprints come from a small
  built-in dict tuned for Qwen3.6-35B-A3B-UD-Q4_K_M; override or extend via
  ``--config-json``. Buckets without a known per-dispatch footprint emit
  ``None`` for ``effective_gb_s`` to make missing data visible.

Single-CSV mode produces only a ``"prefill"`` phase block. Paired mode
produces both ``"prefill"`` and ``"decode"`` blocks (the typical 512/0 vs
512/128 split). The decode CSV is expected to include the prefill prefix
too (rocprofv3 traces both phases when the bench runs end-to-end); per-token
metrics divide by ``--tokens-decode`` only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "p9_gguf_rocprof_summary_v1"

# Default per-dispatch byte footprints for Qwen3.6-35B-A3B-UD-Q4_K_M, used for
# back-calculated effective GB/s. ``None`` means "no known per-dispatch
# footprint -- omit GB/s". Override via ``--config-json``.
#
# The values come from the model config (hidden_size=2048,
# expert_ffn=4096, top_k=8, shared_ffn=4096) crossed with the GGUF block
# bytes-per-weight density: Q4_K ~0.5625 B/w (144 B / 256 K), Q5_K ~0.6875
# B/w, Q6_K ~0.8203 B/w, Q8_0 ~1.0625 B/w. Per-block headers (d/dmin/scales)
# are amortised over 256 K's so the effective average dominates.
_QWEN36_35B_A3B_DEFAULT_FOOTPRINTS_PER_DISPATCH: dict[str, int | None] = {
    # ------------------------------------------------------------------ MoE
    # Compact selected dual gate+up (P9.B1) and the matching WMMA prefill
    # (P8.4). One dispatch processes one compact tile across all active
    # experts in that tile; for c=1 decode, compact_rows == top_k == 8 and
    # one dispatch covers two HxF weight tensors per active expert.
    "moe_q4_k_selected_dual_wmma_prefill": int(8 * 2 * 2048 * 4096 * 0.5625),
    "moe_q4_k_selected_dual_pack8_gemv_decode_p9": int(8 * 2 * 2048 * 4096 * 0.5625),
    "moe_q4_k_selected_dual_t16_gemv_decode_p9": int(8 * 2 * 2048 * 512 * 0.5625),
    "moe_q4_k_selected_legacy_decode": int(8 * 2 * 2048 * 4096 * 0.5625),
    # Selected down (P8.5 WMMA prefill / P9.B2 GEMV decode / legacy decode).
    "moe_q5_k_selected_wmma_prefill": int(8 * 4096 * 2048 * 0.6875),
    "moe_q5_k_selected_pack8_gemv_decode_p9": int(8 * 4096 * 2048 * 0.6875),
    "moe_q5_k_selected_t16_gemv_decode_p9": int(8 * 512 * 2048 * 0.6875),
    "moe_q5_k_selected_legacy_decode": int(8 * 4096 * 2048 * 0.6875),
    "moe_q6_k_selected_wmma_prefill": int(8 * 4096 * 2048 * 0.8203),
    "moe_q6_k_selected_pack8_gemv_decode_p9": int(8 * 4096 * 2048 * 0.8203),
    "moe_q6_k_selected_t16_gemv_decode_p9": int(8 * 512 * 2048 * 0.8203),
    "moe_q6_k_selected_legacy_decode": int(8 * 4096 * 2048 * 0.8203),
    # ------------------------------------------------------ Dense Q8_0 attn
    # Each per-layer attention or shared-expert projection touches one HxF
    # Q8_0 weight tensor. The per-dispatch shape depends on which projection
    # (Q/K/V/O/shexp-gate/shexp-up/shexp-down); 2048*2048 is a representative
    # midpoint for QKV/O at hidden_size=2048.
    "dense_q8_0_wmma_prefill": int(2048 * 2048 * 1.0625),
    "dense_q8_0_pack8_gemv_decode_p9": int(2048 * 2048 * 1.0625),
    "dense_q8_0_t16_gemv_decode_p9": int(2048 * 2048 * 1.0625),
    "dense_q8_0_legacy_decode": int(2048 * 2048 * 1.0625),
    # ------------------------------------------------------ Dense Q4_K attn
    # Same midpoint shape (2048x2048); Q4_K density 0.5625 B/w.
    "dense_q4_k_prefill": int(2048 * 2048 * 0.5625),
    "dense_q4_k_pack8_gemv_decode_p9": int(2048 * 2048 * 0.5625),
    "dense_q4_k_legacy_decode": int(2048 * 2048 * 0.5625),
    # ----------------------------------------------------- Dense Q6_K lm-head
    # Vocab assumed 151_936 (Qwen3.6 default). Override via config-json for
    # other tokenisers.
    "dense_q6_k_pack8_gemv_decode_p9": int(2048 * 151_936 * 0.8203),
    "dense_q6_k_t16_gemv_decode_p9": int(2048 * 151_936 * 0.8203),
    "dense_q6_k_legacy_decode": int(2048 * 151_936 * 0.8203),
    # ---------------------------------------------------------- Non-weight
    "gdn_prefill_recurrent": None,
    "gdn_prefill_rmsnorm_gate": None,
    "gdn_prepare": None,
    "gdn_decode": None,
    "linear_attn_conv": None,
    "full_attention_prefill": None,
    "full_attention_decode": None,
    "router": None,
    "moe_scheduler": None,
    "silu_mul": None,
    "moe_combine": None,
    "rmsnorm": None,
    "kv_write": None,
    "copy": None,
    "other": None,
}


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


@dataclass
class _Kernel:
    name: str
    duration_ns: int
    vgpr: int | None = None
    accum_vgpr: int | None = None
    sgpr: int | None = None
    scratch: int | None = None
    lds: int | None = None
    workgroup_size: int | None = None
    grid_size: int | None = None


def _int_or_none(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _product_or_none(*values: int | None) -> int | None:
    product = 1
    for value in values:
        if value is None:
            return None
        product *= value
    return product


def _row_product_or_none(row: dict[str, str], *columns: str) -> int | None:
    return _product_or_none(*(_int_or_none(row.get(column)) for column in columns))


def read_kernel_trace(path: Path) -> list[_Kernel]:
    """Read a rocprofv3 ``--kernel-trace`` CSV into a list of ``_Kernel``."""

    rows: list[_Kernel] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start = int(float(row["Start_Timestamp"]))
                end = int(float(row["End_Timestamp"]))
            except (KeyError, ValueError):
                continue
            if end < start:
                continue
            name = (
                row.get("Kernel_Name")
                or row.get("KernelName")
                or row.get("Name")
                or ""
            ).strip()
            if not name:
                continue
            rows.append(
                _Kernel(
                    name=name,
                    duration_ns=end - start,
                    vgpr=_int_or_none(row.get("VGPR_Count")),
                    accum_vgpr=_int_or_none(row.get("Accum_VGPR_Count")),
                    sgpr=_int_or_none(row.get("SGPR_Count")),
                    scratch=_int_or_none(row.get("Scratch_Size")),
                    lds=_int_or_none(row.get("LDS_Block_Size")),
                    workgroup_size=_row_product_or_none(
                        row,
                        "Workgroup_Size_X",
                        "Workgroup_Size_Y",
                        "Workgroup_Size_Z",
                    ),
                    grid_size=_row_product_or_none(
                        row,
                        "Grid_Size_X",
                        "Grid_Size_Y",
                        "Grid_Size_Z",
                    ),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Bucket classifier
# ---------------------------------------------------------------------------


_TEMPLATE_RE = re.compile(r"<[^>]*>")


def _normalise_kernel_name(name: str) -> str:
    """Strip template arguments for bucket classification."""

    return _TEMPLATE_RE.sub("", name).strip()


def classify_kernel(name: str) -> str:
    """Map a raw rocprof kernel symbol to a GGUF-aware bucket name.

    The classifier inspects the raw symbol (including its template arguments
    for Q-quant template numbers like ``<5>`` / ``<6>`` / ``<8>``) and falls
    back to ``other`` on unknown kernels. Order matters: more specific
    patterns must be tested before broader ones.
    """

    lower = name.lower()
    base = _normalise_kernel_name(lower)
    # ------------------------------------------------------ GGUF Q4_K MoE
    if (
        "gguf_q4_k_selected_dual_wmma_prefill_compact" in base
        or "gguf_q4_k_t16_selected_dual_wmma_prefill_compact" in base
    ):
        return "moe_q4_k_selected_dual_wmma_prefill"
    if "gguf_q4_k_selected_dual_pack8_gemv_decode_compact" in base:
        return "moe_q4_k_selected_dual_pack8_gemv_decode_p9"
    if "q4_k_t16_selected_dual" in base and "gemv" in base:
        return "moe_q4_k_selected_dual_t16_gemv_decode_p9"
    if "gguf_q4_k_selected_dual_prefill_out" in base or "gguf_q4_k_selected_pack8_prefill_out" in base:
        return "moe_q4_k_selected_legacy_decode"
    # ------------------------------------------------- GGUF Q5_K / Q6_K MoE
    if (
        "gguf_k_selected_wmma_prefill_compact" in base
        or "gguf_k_t16_selected_wmma_prefill_compact" in base
    ):
        if ", 5" in name or ",5" in name:
            return "moe_q5_k_selected_wmma_prefill"
        if ", 6" in name or ",6" in name:
            return "moe_q6_k_selected_wmma_prefill"
    if "gguf_k_selected_pack8_gemv_decode_compact" in base:
        if ", 5" in name or ",5" in name:
            return "moe_q5_k_selected_pack8_gemv_decode_p9"
        if ", 6" in name or ",6" in name:
            return "moe_q6_k_selected_pack8_gemv_decode_p9"
    if "qk_t16_selected" in base and "gemv" in base:
        if ", 5" in name or ",5" in name:
            return "moe_q5_k_selected_t16_gemv_decode_p9"
        if ", 6" in name or ",6" in name:
            return "moe_q6_k_selected_t16_gemv_decode_p9"
    if "gguf_k_selected_pack8_prefill_out" in base:
        if ", 5" in name or ",5" in name:
            return "moe_q5_k_selected_legacy_decode"
        if ", 6" in name or ",6" in name:
            return "moe_q6_k_selected_legacy_decode"
    # ---------------------------------------------- Dense Q8_0 / Q4_K / Q6_K
    if (
        "gguf_q8_0_prefill_wmma" in base
        or "gguf_q8_0_prefill_dual_wmma" in base
        or "gguf_q8_0_t16_prefill_wmma" in base
    ):
        return "dense_q8_0_wmma_prefill"
    if "gguf_q8_0_pack8_gemv_decode" in base or "gguf_q8_0_pack8_dual_gate_up_gemv_decode" in base:
        return "dense_q8_0_pack8_gemv_decode_p9"
    if (
        "q8_0_t16_gemv" in base
        or "q8_0_t16_dual_gemv" in base
        or "q8_0_t16_dual_split_gemv" in base
        or "q8_0_t16_triple" in base
    ):
        return "dense_q8_0_t16_gemv_decode_p9"
    if "gguf_k_pack8_prefill_out" in base and (", 8" in name or ",8" in name):
        return "dense_q8_0_legacy_decode"
    if "gguf_q4_k_pack8_gemv_decode" in base:
        return "dense_q4_k_pack8_gemv_decode_p9"
    if "gguf_q4_k_prefill" in base:
        # Includes _wmma, _dual_wmma, and _out single + dual.
        return "dense_q4_k_prefill"
    if "gguf_q4_k_pack8" in base or "gguf_q4_k_dual_pack8_prefill_out" in base:
        return "dense_q4_k_legacy_decode"
    if "gguf_q6_k_pack8_gemv_decode" in base:
        return "dense_q6_k_pack8_gemv_decode_p9"
    if "q6_k_t16_gemv" in base:
        return "dense_q6_k_t16_gemv_decode_p9"
    if "gguf_k_pack8_prefill_out" in base and (", 6" in name or ",6" in name):
        return "dense_q6_k_legacy_decode"
    # ------------------------------------------------------ GDN / linear-attn
    if "qwen35_gdn_prefill_recurrent_rmsnorm_gate" in base or "qwen35_gdn_prefill_recurrent_k2" in base or "qwen35_gdn_prefill_recurrent_segments" in base or "qwen35_gdn_prefill_recurrent" in base:
        return "gdn_prefill_recurrent"
    if "qwen35_gdn_prefill_rmsnorm_gate" in base:
        return "gdn_prefill_rmsnorm_gate"
    if "qwen35_linear_attn_prefill_prepare" in base:
        return "gdn_prepare"
    if "qwen35_gdn_recurrent" in base:
        return "gdn_decode"
    if "qwen35_linear_attn_conv" in base:
        return "linear_attn_conv"
    # ------------------------------------------------------ Full attention
    if "qwen35_paged_full_attn_prefill" in base or "causal_gqa" in base or "attn_fwd" in base:
        return "full_attention_prefill"
    if "qwen35_paged_full_attn_decode" in base or "full_attn_decode" in base:
        return "full_attention_decode"
    # ----------------------------------------------------------- Router / MoE meta
    if "qwen35_router" in base:
        return "router"
    if (
        "qwen35_moe_group_count" in base
        or "qwen35_moe_group_prefix" in base
        or "qwen35_moe_group_scatter_gather" in base
        or "qwen35_moe_wmma_tile_map" in base
    ):
        return "moe_scheduler"
    # ------------------------------------------------------------- Combine + SiLU
    if "silu" in base:
        return "silu_mul"
    if "weighted_lanes" in base or "weighted_sum" in base or "shared_gate_combine" in base or "combine_residual" in base:
        return "moe_combine"
    # --------------------------------------------------------------- RMSNorm
    if "rmsnorm" in base or "qwen35_head_rmsnorm" in base:
        return "rmsnorm"
    # --------------------------------------------------------------- KV write
    if "write_paged_kv" in base or "paged_kv" in base:
        return "kv_write"
    # --------------------------------------------------------------- Runtime
    if (
        "copybuffer" in base
        or "fillbuffer" in base
        or base.startswith("hipmemcpy")
        or base.startswith("hipmemset")
        or "memset" in base
    ):
        return "copy"
    return "other"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


_RESOURCE_FIELDS = {
    "vgpr_count": "vgpr",
    "accum_vgpr_count": "accum_vgpr",
    "sgpr_count": "sgpr",
    "scratch_bytes": "scratch",
    "lds_bytes": "lds",
    "workgroup_size": "workgroup_size",
    "grid_size": "grid_size",
}


def _new_kernel_stats() -> dict[str, Any]:
    return {"ns": 0, "n": 0, "resources": defaultdict(set)}


def _add_resources(resource_sets: dict[str, set[int]], kernel: _Kernel) -> None:
    for public_name, attr in _RESOURCE_FIELDS.items():
        value = getattr(kernel, attr)
        if value is not None:
            resource_sets[public_name].add(value)


def _resource_summary(resource_sets: dict[str, set[int]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for public_name in _RESOURCE_FIELDS:
        values = sorted(resource_sets.get(public_name, ()))
        summary[public_name] = {
            "min": values[0] if values else None,
            "max": values[-1] if values else None,
            "values": values,
        }
    return summary


@dataclass
class BucketStats:
    bucket: str
    total_ns: int = 0
    dispatches: int = 0
    kernels: dict[str, dict[str, Any]] = field(default_factory=lambda: defaultdict(_new_kernel_stats))
    resources: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))

    def add(self, kernel: _Kernel) -> None:
        self.total_ns += kernel.duration_ns
        self.dispatches += 1
        _add_resources(self.resources, kernel)
        ent = self.kernels[kernel.name]
        ent["ns"] += kernel.duration_ns
        ent["n"] += 1
        _add_resources(ent["resources"], kernel)


def _summarise_phase(
    kernels: Iterable[_Kernel],
    *,
    tokens: int | None,
    footprints: dict[str, int | None],
    top: int,
) -> dict[str, Any]:
    buckets: dict[str, BucketStats] = {}
    per_kernel: dict[str, dict[str, Any]] = defaultdict(_new_kernel_stats)
    total_ns = 0
    for k in kernels:
        total_ns += k.duration_ns
        bucket = classify_kernel(k.name)
        buckets.setdefault(bucket, BucketStats(bucket=bucket)).add(k)
        ent = per_kernel[k.name]
        ent["ns"] += k.duration_ns
        ent["n"] += 1
        _add_resources(ent["resources"], k)

    bucket_rows: list[dict[str, Any]] = []
    for stats in buckets.values():
        total_ms = stats.total_ns / 1e6
        share = stats.total_ns / total_ns if total_ns else 0.0
        avg_ms = (stats.total_ns / stats.dispatches) / 1e6 if stats.dispatches else 0.0
        footprint = footprints.get(stats.bucket)
        if footprint is not None and stats.total_ns > 0:
            total_bytes = footprint * stats.dispatches
            effective_gb_s = total_bytes / (stats.total_ns / 1e9) / 1e9
        else:
            effective_gb_s = None
        bucket_rows.append(
            {
                "bucket": stats.bucket,
                "total_ms": total_ms,
                "dispatches": stats.dispatches,
                "avg_dispatch_ms": avg_ms,
                "share_of_phase": share,
                "footprint_bytes_per_dispatch": footprint,
                "effective_gb_s": effective_gb_s,
                "resource_summary": _resource_summary(stats.resources),
                "kernel_names": sorted(stats.kernels.keys()),
                "per_kernel_ms": {
                    name: ent["ns"] / 1e6 for name, ent in stats.kernels.items()
                },
                "per_kernel_dispatches": {
                    name: ent["n"] for name, ent in stats.kernels.items()
                },
                "per_kernel_resource_summary": {
                    name: _resource_summary(ent["resources"])
                    for name, ent in stats.kernels.items()
                },
                "ms_per_token": None if not tokens else total_ms / tokens,
            }
        )
    bucket_rows.sort(key=lambda x: x["total_ms"], reverse=True)

    per_kernel_rows: list[dict[str, Any]] = []
    for name, ent in per_kernel.items():
        total_ms = ent["ns"] / 1e6
        share = ent["ns"] / total_ns if total_ns else 0.0
        avg_ms = total_ms / ent["n"] if ent["n"] else 0.0
        per_kernel_rows.append(
            {
                "kernel": name,
                "total_ms": total_ms,
                "dispatches": ent["n"],
                "avg_dispatch_ms": avg_ms,
                "share_of_phase": share,
                "bucket": classify_kernel(name),
                "resource_summary": _resource_summary(ent["resources"]),
                "ms_per_token": None if not tokens else total_ms / tokens,
            }
        )
    per_kernel_rows.sort(key=lambda x: x["total_ms"], reverse=True)

    return {
        "total_kernel_ms": total_ns / 1e6,
        "total_dispatches": sum(b["dispatches"] for b in bucket_rows),
        "tokens": tokens,
        "ms_per_token": None if not tokens else (total_ns / 1e6) / tokens,
        "buckets": bucket_rows,
        "top_kernels": per_kernel_rows[:top],
        "per_kernel": per_kernel_rows,
    }


# ---------------------------------------------------------------------------
# Phase split for paired CSV
# ---------------------------------------------------------------------------


def split_prefill_decode(
    kernels: list[_Kernel], *, prefill_dispatches: int
) -> tuple[list[_Kernel], list[_Kernel]]:
    """Split a paired CSV into prefill + decode using the dispatch count from
    a prefill-only reference run.

    ``prefill_dispatches`` is the number of kernels logged during the
    prefill phase of an otherwise identical bench run (i.e. with
    ``--decode-tokens 0``). The same prefill prefix is expected at the start
    of the paired trace, so we slice off that many leading kernels.
    """

    if prefill_dispatches <= 0:
        return [], kernels
    return kernels[:prefill_dispatches], kernels[prefill_dispatches:]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def build_summary(
    *,
    prefill_csv: Path | None,
    decode_csv: Path | None,
    single_csv: Path | None,
    tokens_prefill: int | None,
    tokens_decode: int | None,
    footprints: dict[str, int | None],
    top: int,
    prefill_dispatches_from_single: bool,
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    notes: list[str] = []

    if single_csv is not None:
        kernels = read_kernel_trace(single_csv)
        phases["prefill"] = _summarise_phase(
            kernels, tokens=tokens_prefill, footprints=footprints, top=top
        )
        inputs = {"csv": str(single_csv)}
        notes.append(
            "single-csv mode: only the prefill phase is reported. Pass --decode-csv "
            "(paired with --prefill-csv) to also report a decode phase."
        )
    else:
        assert prefill_csv is not None and decode_csv is not None
        prefill_kernels = read_kernel_trace(prefill_csv)
        decode_kernels_all = read_kernel_trace(decode_csv)
        phases["prefill"] = _summarise_phase(
            prefill_kernels, tokens=tokens_prefill, footprints=footprints, top=top
        )
        prefill_dispatches_used: int
        if prefill_dispatches_from_single:
            prefill_dispatches_used = phases["prefill"]["total_dispatches"]
        else:
            prefill_dispatches_used = 0
            notes.append(
                "decode phase reported over the full --decode-csv (prefill prefix "
                "not subtracted). Pass --strip-prefill-prefix to subtract the "
                "leading prefill dispatch count from the prefill-only CSV."
            )
        prefill_prefix, decode_kernels = split_prefill_decode(
            decode_kernels_all, prefill_dispatches=prefill_dispatches_used
        )
        if prefill_dispatches_used:
            notes.append(
                f"stripped {len(prefill_prefix)} leading dispatch(es) from the "
                "decode CSV as the prefill prefix."
            )
        phases["decode"] = _summarise_phase(
            decode_kernels, tokens=tokens_decode, footprints=footprints, top=top
        )
        inputs = {
            "prefill_csv": str(prefill_csv),
            "decode_csv": str(decode_csv),
            "decode_prefill_prefix_dispatches": prefill_dispatches_used,
        }

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "tokens_prefill": tokens_prefill,
        "tokens_decode": tokens_decode,
        "footprints_used": footprints,
        "phases": phases,
        "notes": notes,
    }


def _load_footprint_overrides(path: Path | None) -> dict[str, int | None]:
    footprints = dict(_QWEN36_35B_A3B_DEFAULT_FOOTPRINTS_PER_DISPATCH)
    if path is None:
        return footprints
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("config JSON must be an object mapping bucket -> bytes-per-dispatch (or null)")
    for bucket, value in raw.items():
        if value is None:
            footprints[str(bucket)] = None
        else:
            footprints[str(bucket)] = int(value)
    return footprints


def _print_human(summary: dict[str, Any]) -> None:
    for phase_name, phase in summary["phases"].items():
        print(
            f"\n=== {phase_name.upper()} === total kernel {phase['total_kernel_ms']:.3f} ms / "
            f"{phase['total_dispatches']} dispatches"
            + (
                f" / {phase['ms_per_token']:.3f} ms/token"
                if phase.get("ms_per_token") is not None
                else ""
            )
        )
        print("buckets:")
        for b in phase["buckets"]:
            line = (
                f"  {b['total_ms']:10.3f} ms "
                f"{b['dispatches']:7d} disp "
                f"{b['avg_dispatch_ms']:8.4f} ms/disp "
                f"{b['share_of_phase'] * 100:6.2f}% "
                f"{b['bucket']}"
            )
            if b["effective_gb_s"] is not None:
                line += f"  ~{b['effective_gb_s']:7.2f} GB/s"
            print(line)
        print("top kernels:")
        for k in phase["top_kernels"]:
            print(
                f"  {k['total_ms']:10.3f} ms {k['dispatches']:7d} disp "
                f"{k['avg_dispatch_ms']:8.4f} ms/disp {k['kernel'][:120]}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None, help="Single rocprofv3 --kernel-trace CSV.")
    parser.add_argument(
        "--prefill-csv",
        type=Path,
        default=None,
        help="Prefill-only rocprofv3 CSV (paired mode).",
    )
    parser.add_argument(
        "--decode-csv",
        type=Path,
        default=None,
        help="Full prefill+decode rocprofv3 CSV (paired mode). Prefill prefix is "
        "stripped using the dispatch count from --prefill-csv when "
        "--strip-prefill-prefix is set.",
    )
    parser.add_argument(
        "--strip-prefill-prefix",
        action="store_true",
        help="Strip the leading prefill dispatches from --decode-csv using the "
        "dispatch count from --prefill-csv. Default: False (decode CSV is "
        "summarised as a single phase including any prefill prefix).",
    )
    parser.add_argument(
        "--tokens-prefill",
        type=int,
        default=None,
        help="Number of prefill tokens. Enables ms/token in the prefill phase report.",
    )
    parser.add_argument(
        "--tokens-decode",
        type=int,
        default=None,
        help="Number of decode tokens. Enables ms/token in the decode phase report.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=None,
        help="Optional JSON config that overrides the built-in footprints "
        "(bucket -> bytes-per-dispatch, null to mark unknown).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write the JSON report to this path. If omitted, JSON is printed to stdout.",
    )
    parser.add_argument(
        "--top", type=int, default=30, help="Top-N kernels per phase (default 30)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )
    args = parser.parse_args(argv)

    if args.csv is None and (args.prefill_csv is None or args.decode_csv is None):
        parser.error("specify either --csv (single) or both --prefill-csv and --decode-csv")
    if args.csv is not None and (args.prefill_csv is not None or args.decode_csv is not None):
        parser.error("cannot combine --csv with --prefill-csv / --decode-csv")
    if args.csv is not None and not args.csv.is_file():
        parser.error(f"--csv {args.csv} does not exist or is not a file")
    if args.prefill_csv is not None and not args.prefill_csv.is_file():
        parser.error(f"--prefill-csv {args.prefill_csv} does not exist or is not a file")
    if args.decode_csv is not None and not args.decode_csv.is_file():
        parser.error(f"--decode-csv {args.decode_csv} does not exist or is not a file")

    footprints = _load_footprint_overrides(args.config_json)

    summary = build_summary(
        prefill_csv=args.prefill_csv,
        decode_csv=args.decode_csv,
        single_csv=args.csv,
        tokens_prefill=args.tokens_prefill,
        tokens_decode=args.tokens_decode,
        footprints=footprints,
        top=args.top,
        prefill_dispatches_from_single=args.strip_prefill_prefix,
    )

    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    else:
        print(json.dumps(summary, indent=2))

    if not args.quiet:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
