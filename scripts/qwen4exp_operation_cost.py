#!/usr/bin/env python3
"""Operation-level prefill cost for one Qwen4Exp case, with real shapes.

Answers "which operation costs what, at which shape" for a full prefill, rather
than "which kernel symbol costs what". A symbol cannot answer it: one variant
serves attention, hyper-connection and shared-expert projections, and the MoE
role groups its gate matmul with routing, scatter and repair passes.

Three inputs, all from the same run and the same model file:

* ``--trace-kernels``: per (role, kernel) milliseconds and launch count from a
  role-marked ``rocprofv3`` capture.
* ``--census``: the GGUF launch census, giving ``K``, ``N`` and the row count
  behind the matmuls that go through the GGUF quant path.
* ``--model-shapes``: tensor geometry and the expert routing config read from
  the GGUF itself, so operations outside that path (MoE, QSA, GR, GDN, indexer)
  still get a real shape instead of being dropped.

Rows whose shape cannot be resolved are reported as ``unresolved`` with their
milliseconds intact, so the table always sums to the measured window.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

BYTES_PER_ELEMENT = {
    "gguf_q8_0": 34 / 32, "gguf_q4_k": 144 / 256, "gguf_q5_k": 176 / 256,
    "gguf_q6_k": 210 / 256, "gguf_q5_1": 24 / 32,
}
GGML_BYTES_PER_ELEMENT = {
    "Q8_0": 34 / 32, "Q4_K": 144 / 256, "Q5_K": 176 / 256,
    "Q6_K": 210 / 256, "Q5_1": 24 / 32, "Q4_0": 18 / 32, "Q6_0": 26 / 32,
}
# Roofline constants. These are per-machine, and the machine matters: the run
# behind these artifacts executed on gfx1151 (Radeon 8060S, Strix Halo APU), not
# on the gfx1100 W7900 that docs/ROOFLINE.md documents. Using the W7900 figures
# here would overstate the FP32 ceiling by about 2x and the memory bandwidth by
# about 3.4x, so the hardware identity travels with the report.
HARDWARE = {
    "gfx1151": {
        "name": "AMD Radeon 8060S (Strix Halo APU)",
        "compute_units": 40,
        "clock_ghz": 2.9,
        # 40 CU x 128 lanes x 2 FLOP x 2.9 GHz
        "fp32_fma_peak_gflops": 29_696.0,
        "dram_bytes_per_second": 256e9,
        "mall_bytes": 32 * 1024 * 1024,
        "l2_bytes": 2 * 1024 * 1024,
    },
    "gfx1100": {
        "name": "AMD Radeon Pro W7900 (RDNA3)",
        "compute_units": 96,
        "clock_ghz": 2.5,
        "fp32_fma_peak_gflops": 61_300.0,
        "dram_bytes_per_second": 864e9,
        "mall_bytes": 96 * 1024 * 1024,
        "l2_bytes": 6 * 1024 * 1024,
    },
}
DEFAULT_HARDWARE = "gfx1151"

BF16_WMMA_SPEC_GFLOPS = 123_000.0
BF16_WMMA_MEASURED_GFLOPS = 84_800.0

# Resolved from HARDWARE below. The dense projections here accumulate in scalar
# FP32, so the FP32 rate is the honest denominator for them; matrix-core
# families are measured against the WMMA rate and labelled per row.
FP32_FMA_PEAK_GFLOPS = HARDWARE[DEFAULT_HARDWARE]["fp32_fma_peak_gflops"]
DRAM_BYTES_PER_SECOND = HARDWARE[DEFAULT_HARDWARE]["dram_bytes_per_second"]
PEAK_BASIS = "scalar FP32 FMA on " + HARDWARE[DEFAULT_HARDWARE]["name"]

MATRIX_CORE_FAMILIES = ("wmma", "iu8", "dp4a", "mmq", "matmul")
_NORMALIZE = re.compile(r"layers\.\d+\.")


def bytes_per_element(quant: str) -> float:
    """Bytes per weight element, accepting both ``Q8_0`` and ``gguf_q8_0``."""
    key = quant.upper()
    if key.startswith("GGUF_"):
        key = key[5:]
    return GGML_BYTES_PER_ELEMENT.get(key, 0.0)


def normalize(role: str) -> str:
    return _NORMALIZE.sub("layers.*.", role)


# Kernel families that are a quantized matmul, mapped to the weight they read
# and what their row count means. "routed" = tokens x experts_used (full MoE
# work); "token" = the dense token count; "partial" = only the rows flagged by
# the risk heuristic, so the full-shape FLOP count would overstate the work and
# no achieved rate is reported for it.
MATMUL_FAMILIES: tuple[tuple[str, str, str], ...] = (
    # --- MoE full-work matmuls (rows are tokens x experts_used) -----------
    (r"gguf_q4_k_selected_dual_wmma_iu8_risk_prefill", "ffn_gate_exps", "routed"),
    (r"gguf_q4_k_selected_dual_wmma_iu8_prefill", "ffn_gate_exps", "routed"),
    (r"q5_1_selected_wmma_iu8_risk_prefill", "ffn_down_exps", "routed"),
    (r"q8_0_selected_grouped_wmma_prefill", "ffn_down_exps", "routed"),
    # --- Risk / repair passes (rows are only the flagged subset) ----------
    (r"gguf_q4_k_selected_dual_wmma_iu8_risk_p2_prefill", "ffn_gate_exps", "partial"),
    (r"gguf_q4_k_selected_dual_sparse_exact_repair", "ffn_gate_exps", "partial"),
    (r"q5_1_selected_sparse_exact_repair", "ffn_down_exps", "partial"),
    (r"q8_0_selected_sparse_repair", "ffn_down_exps", "partial"),
    # --- Dense projections (rows are tokens) ------------------------------
    (r"gguf_k_prefill_out_coltile_rowbatch", None, "token"),
    (r"gguf_k_prefill_out_rowbatch", None, "token"),
    (r"gguf_k_prefill_out_rowtile", None, "token"),
)


def _classify(kernel: str) -> tuple[str, str] | None:
    for pattern, tensor, row_mode in MATMUL_FAMILIES:
        if re.search(pattern, kernel):
            return tensor or "", row_mode
    return None


def load_inputs(
    trace_kernels: Path, census: Path, model_shapes: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    per_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(trace_kernels.read_text())["exact_role_kernels"]:
        per_role[normalize(row["role"])].append(row)

    # Census shapes keyed by tensor suffix, so a role can borrow the geometry of
    # any other role that launched the same tensor.
    census_shapes: dict[str, dict[str, Any]] = {}
    for row in json.loads(census.read_text())["rows"]:
        suffix = row["role"].split(":", 1)[-1]
        suffix = suffix.split(".", 1)[-1] if suffix.startswith("layers.") else suffix
        suffix = re.sub(r"^root\.", "", suffix)
        key = suffix.split(".")[-1] if "." in suffix else suffix
        entry = census_shapes.setdefault(key, {
            "quant": row["quant"], "in_features": row["in_features"],
            "out_features": row["out_features"], "launches_per_layer": 0,
            "rows_per_launch": row["rows"],
        })
        entry["launches_per_layer"] += int(row["launches"])

    model = json.loads(model_shapes.read_text())
    return per_role, census_shapes, model


def _tensor_geometry(model: dict[str, Any], name: str) -> dict[str, Any] | None:
    tensors = model["tensors"]
    for key in (f"{name}.weight", name):
        if key in tensors:
            return tensors[key]
    return None


def build_rows(
    per_role: dict[str, list[dict[str, Any]]],
    census_shapes: dict[str, dict[str, Any]],
    model: dict[str, Any],
    chunk_tokens: int,
    total_tokens: int,
) -> list[dict[str, Any]]:
    experts_used = int(model["config"].get("qwen4exp.expert_used_count", 1))
    chunks = max(1, total_tokens // chunk_tokens)
    out: list[dict[str, Any]] = []

    for role, kernels in per_role.items():
        tensor_suffix = role.split(":", 1)[-1].rsplit(".", 1)[-1]
        for row in kernels:
            ms = float(row["ms"])
            classified = _classify(row["kernel"])
            entry: dict[str, Any] = {
                "role": role,
                "kernel": row["kernel"][:120],
                "ms": round(ms, 1),
            }
            if classified is None:
                entry["operation"] = "non-matmul"
                out.append(entry)
                continue

            explicit_tensor, row_mode = classified
            weight = explicit_tensor or tensor_suffix
            geometry = _tensor_geometry(model, weight)
            census_entry = census_shapes.get(tensor_suffix)

            if row_mode in ("routed", "partial"):
                if geometry is None:
                    entry["operation"] = "unresolved"
                    out.append(entry)
                    continue
                k = geometry["in_features"]
                n = geometry["out_features"]
                rows_per_launch = chunk_tokens * experts_used
                quant = str(geometry["quant"])
                source = "model_shapes+routing"
                # A tensor can be quantized differently across layers, so use
                # the layer-average bytes per element for the traffic estimate.
                bytes_per_element_override = float(
                    geometry.get("quant_mix_bytes_weighted")
                    or bytes_per_element(quant)
                )
            else:
                if census_entry is None:
                    entry["operation"] = "unresolved"
                    out.append(entry)
                    continue
                k = census_entry["in_features"]
                n = census_entry["out_features"]
                rows_per_launch = census_entry["rows_per_launch"]
                quant = census_entry["quant"]
                source = "launch_census"
                bytes_per_element_override = bytes_per_element(str(quant))

            # ``rows`` in the trace is the number of launches of this kernel
            # under this role, which is the count that scales the FLOPs.
            launches = int(row.get("rows") or 1)

            flops = 2.0 * rows_per_launch * k * n * launches
            bytes_moved = bytes_per_element_override * k * n * launches
            entry.update({
                "operation": "matmul" if row_mode != "partial" else "risk_or_repair",
                "kernel_family": _family_name(row["kernel"]),
                "shape_source": source,
                "quant": quant,
                "quant_mix": (
                    geometry.get("quant_mix") if row_mode in ("routed", "partial") else None
                ),
                "in_features": k,
                "out_features": n,
                "rows_per_launch": rows_per_launch,
                "launches": launches,
                "gflop": flops / 1e9,
                "gflop_basis": (
                    "full routed rows" if row_mode != "partial"
                    else "full-shape upper bound; actual rows are the flagged subset only"
                ),
                "weight_traffic_gb": bytes_moved / 1e9,
            })
            out.append(entry)
    return out


def _family_name(kernel: str) -> str:
    for pattern, _tensor, _mode in MATMUL_FAMILIES:
        if re.search(pattern, kernel):
            return pattern.replace("\\", "")
    return "other"


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-layer rows into one row per (role, operation, family)."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["role"], row["operation"], row.get("kernel_family", "-"))
        acc = grouped.get(key)
        if acc is None:
            acc = dict(row)
            acc.pop("kernel", None)
            grouped[key] = acc
            continue
        acc["ms"] = acc["ms"] + row["ms"]
        for field in ("gflop", "weight_traffic_gb"):
            if field in row:
                acc[field] = acc.get(field, 0.0) + row[field]
        for field in ("launches",):
            if field in row:
                acc[field] = acc.get(field, 0) + row[field]
    # Round only after every contribution is summed; rounding each layer first
    # drives small per-layer traffic values to zero.
    for acc in grouped.values():
        acc["ms"] = round(acc["ms"], 1)
        for field in ("gflop", "weight_traffic_gb"):
            if field in acc:
                acc[field] = round(acc[field], 2)
    return list(grouped.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-kernels", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--model-shapes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--total-tokens", type=int, default=4096)
    args = parser.parse_args()

    per_role, census_shapes, model = load_inputs(
        args.trace_kernels, args.census, args.model_shapes
    )
    rows = aggregate(build_rows(
        per_role, census_shapes, model, args.chunk_tokens, args.total_tokens
    ))
    rows.sort(key=lambda r: -r["ms"])

    for row in rows:
        if row["operation"] == "matmul" and row["ms"] > 0:
            flops = row["gflop"] * 1e9
            row["achieved_gflops"] = round(flops / (row["ms"] / 1000.0) / 1e9, 1)
            intensity = flops / (row["weight_traffic_gb"] * 1e9) if row.get(
                "weight_traffic_gb"
            ) else 0.0
            row["arithmetic_intensity_flops_per_byte"] = round(intensity, 1)
            ridge = FP32_FMA_PEAK_GFLOPS * 1e9 / DRAM_BYTES_PER_SECOND
            row["roofline_limit"] = (
                "compute-bound" if intensity > ridge else "bandwidth-bound"
            )
            family = str(row.get("kernel_family", "")).lower()
            uses_matrix_core = any(tag in family for tag in MATRIX_CORE_FAMILIES)
            row["peak_basis"] = (
                "BF16 WMMA spec (matrix-core kernel)" if uses_matrix_core
                else PEAK_BASIS
            )
            denominator = (
                BF16_WMMA_SPEC_GFLOPS if uses_matrix_core else FP32_FMA_PEAK_GFLOPS
            )
            row["share_of_peak_pct"] = round(
                100.0 * row["achieved_gflops"] / denominator, 1
            )

    total_ms = sum(r["ms"] for r in rows)
    matmul_ms = sum(r["ms"] for r in rows if r["operation"] == "matmul")
    print(f"{'role':30s} {'operation':11s} {'ms':>8s} {'%':>5s} {'K':>6s} {'N':>6s} "
          f"{'rows':>6s} {'n':>5s} {'GFLOP':>9s} {'GFLOP/s':>8s} {'%pk':>5s}")
    print("-" * 120)
    for row in rows:
        if row["ms"] < 20:
            continue
        pct = 100 * row["ms"] / total_ms
        if row["operation"] != "matmul":
            print(f"{row['role']:30s} {row['operation']:11s} {row['ms']:8.1f} "
                  f"{pct:4.1f}%")
            continue
        print(f"{row['role']:30s} {'matmul':11s} {row['ms']:8.1f} "
              f"{pct:4.1f}% {row['in_features']:6d} "
              f"{row['out_features']:6d} {row['rows_per_launch']:6d} {row['launches']:5d} "
              f"{row['gflop']:9.1f} {row['achieved_gflops']:8.1f} "
              f"{row['share_of_peak_pct']:4.1f}%")
    print("-" * 120)
    print(f"{'TOTAL':30s} {'':11s} {total_ms:8.1f} 100.0%")
    print(f"{'  of which matmul':30s} {'':11s} {matmul_ms:8.1f} "
          f"{100*matmul_ms/total_ms:4.1f}%")

    classes: dict[str, float] = defaultdict(float)
    gflops: dict[str, float] = defaultdict(float)
    for row in rows:
        classes[row["operation"]] += row["ms"]
        if row["operation"] == "matmul":
            gflops[row["operation"]] += row["gflop"]
    print("\nby operation class")
    print("-" * 120)
    for name, ms in sorted(classes.items(), key=lambda kv: -kv[1]):
        rate = (
            f"{gflops[name]/(ms/1000.0):8.1f} GFLOP/s" if name == "matmul" else ""
        )
        print(f"  {name:20s} {ms:9.1f} ms {100*ms/total_ms:5.1f}%  {rate}")

    # Roll up by the leading role family so the prefill can be described at a
    # granularity coarser than one tensor but finer than "everything".
    families: dict[str, float] = defaultdict(float)
    for row in rows:
        families[row["role"].split(":", 1)[0]] += row["ms"]
    print("\nby role family")
    print("-" * 120)
    for name, ms in sorted(families.items(), key=lambda kv: -kv[1]):
        if ms < 20:
            continue
        print(f"  {name:20s} {ms:9.1f} ms {100*ms/total_ms:5.1f}%")

    args.output.write_text(json.dumps({
        "schema": 1,
        "kind": "qwen4exp_operation_cost",
        "performance_claim": False,
        "case": {"chunk_tokens": args.chunk_tokens, "total_tokens": args.total_tokens},
        "peak_gflops": FP32_FMA_PEAK_GFLOPS,
        "peak_basis": PEAK_BASIS,
        "hardware": dict(HARDWARE[DEFAULT_HARDWARE], arch=DEFAULT_HARDWARE),
        "peaks": {
            "fp32_fma_gflops": FP32_FMA_PEAK_GFLOPS,
            "bf16_wmma_spec_gflops": BF16_WMMA_SPEC_GFLOPS,
            "bf16_wmma_measured_gflops": BF16_WMMA_MEASURED_GFLOPS,
            "dram_bytes_per_second": DRAM_BYTES_PER_SECOND,
        },
        "total_ms": round(total_ms, 1),
        "matmul_ms": round(matmul_ms, 1),
        "by_operation_class": {
            name: round(ms, 1)
            for name, ms in sorted(classes.items(), key=lambda kv: -kv[1])
        },
        "by_role_family": {
            name: round(ms, 1)
            for name, ms in sorted(families.items(), key=lambda kv: -kv[1])
        },
        "notes": [
            "Milliseconds from a role-marked rocprofv3 capture with 100% kernel attribution.",
            "Shapes from the launch census where the matmul goes through the GGUF quant path, otherwise from the GGUF tensor geometry plus the expert routing count.",
            "Achieved GFLOP/s excludes dequantization work.",
            "Percent-of-peak uses the scalar FP32 FMA rate for the scalar kernels and the BF16 WMMA spec rate for matrix-core kernels; the basis is recorded per row.",
            "Weight traffic counts per-chunk re-reads and can exceed DRAM bandwidth when a layer stays cache-resident.",
        ],
        "rows": rows,
    }, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
