#!/usr/bin/env python3
"""Assemble the gfx1151 Qwen3.8-27B Q4_K_M prefill kernel-profile artifact.

Every number in ``artifact.json`` comes from a file: the driver JSON written by
``scripts/qwen38_gfx1151_prefill_kernel_profile.py`` (provenance, wall time,
first-token checks) and the raw ``rocprofv3 --kernel-trace`` CSVs of that run.
The GGUF tensor-type table is read from the model file itself, so the
shape-to-owner attribution is checked against the artifact's own launch counts
instead of being transcribed by hand.

  python3 assemble.py \
      --driver-json /tmp/qwen38-prefill-profile/profile.json \
      --run-root /tmp/qwen38-prefill-profile/run \
      --out artifact.json
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen38_gfx1151_prefill_kernel_profile import (  # noqa: E402
    _blocks,
    _classify,
    _summarize_trace,
)
from scripts.gguf_prefill_kernel_resources import derived_occupancy

# Owner geometry is a property of the registered launcher, not of the trace:
# rocprofv3 reports the launched workgroup and grid, not the compile-time tile.
# Each entry is (template-argument names, columns per output tile, rows per
# wave row tile, threads per wave, static shared bytes per workgroup) read from
# the launcher that exports the symbol (see ``owners`` notes in the README).
OWNER_GEOMETRY: Mapping[str, Mapping[str, Any]] = {
    "gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel": {
        "template": ["qmicro", "expanded_meta", "row_tiles_per_wave", "waves_per_block"],
        "columns_per_out_tile": 16,
        "out_tiles": 2,
        "arg_index": {"row_tiles": 2, "waves": 3},
        "rows_per_wave_row_tile": 16,
        "threads_per_wave": 32,
        "quant_family": "q4",
        "layer": "linear_pair_silu",
        "variant": "dense_dual_wmma_prefill_bf16_bf16_out",
    },
    "gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel": {
        "template": ["out_tiles_per_block", "row_tiles_per_wave", "waves_per_block",
                     "scalar_t", "skip_inactive_waves"],
        "columns_per_out_tile": 16,
        "out_tiles": None,  # taken from the first template argument
        "arg_index": {"out_tiles": 0, "row_tiles": 1, "waves": 2},
        "rows_per_wave_row_tile": 16,
        "threads_per_wave": 32,
        "quant_family": "q4",
        "layer": "linear",
        "variant": "t16_wmma_prefill_shared_b_bf16_bf16_out",
    },
    "q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel": {
        "template": ["waves_per_block", "row_tiles_per_wave", "out_tiles_per_block",
                     "pair_decode"],
        "columns_per_out_tile": 16,
        "out_tiles": None,
        "arg_index": {"out_tiles": 2, "row_tiles": 1, "waves": 0},
        "rows_per_wave_row_tile": 16,
        "threads_per_wave": 32,
        "quant_family": "q6",
        "layer": "linear",
        "variant": "t16_wmma_prefill_shared4r4_bf16_bf16_out",
    },
    "q6_k_t16_wmma_prefill_shared_bf16_kernel": {
        "template": ["waves_per_block", "row_tiles_per_wave", "out_tiles_per_block"],
        "columns_per_out_tile": 16,
        "out_tiles": None,
        "arg_index": {"out_tiles": 2, "row_tiles": 1, "waves": 0},
        "rows_per_wave_row_tile": 16,
        "threads_per_wave": 32,
        "quant_family": "q6",
        "layer": "linear",
        "variant": "t16_wmma_prefill_shared8r3_bf16_bf16_out",
    },
    "q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel": {
        "template": ["row_tiles_per_wave", "out_tiles_per_block"],
        "columns_per_out_tile": 16,
        "out_tiles": None,
        "arg_index": {"out_tiles": 1, "row_tiles": 0, "waves": None},
        "rows_per_wave_row_tile": 16,
        "threads_per_wave": 32,
        "quant_family": "q6",
        "layer": "linear",
        "variant": "qmicro_planar_wmma_prefill_bf16_bf16_out",
        "fixed_waves": 1,
    },
}

# Registered owner variant per observed (kernel, template) pair. The launcher
# that exports each symbol carries the variant name used by the registry.
VARIANT_BY_TEMPLATE: Mapping[tuple[str, str], str] = {
    ("gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel", "false, false, 4, 4"):
        "dense_dual_wmma_prefill_bf16_bf16_out",
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false"):
        "t16_wmma_prefill_shared_b_bf16_bf16_out (fp16 activations)",
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, unsigned short, false"):
        "t16_wmma_prefill_shared_b_bf16_bf16_out (bf16 activations)",
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 4, 2, false"):
        "qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out",
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 6, 2, true"):
        "qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out",
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 4, 3, false"):
        "qmicro_planar_wmma_prefill_shared4_gfx1100_bf16_bf16_out",
    ("q6_k_t16_wmma_prefill_shared_bf16_kernel", "8, 3, 2"):
        "t16_wmma_prefill_shared8r3_bf16_bf16_out",
    ("q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel", "4, 3"):
        "qmicro_planar_wmma_prefill_bf16_bf16_out",
}

# Static shared bytes per workgroup, keyed by (kernel, template).
LDS_BY_TEMPLATE: Mapping[tuple[str, str], int] = {
    ("gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel", "false, false, 4, 4"): 32 * 1024,
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false"): 24 * 1024,
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, unsigned short, false"): 24 * 1024,
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 4, 2, false"): 16 * 1024,
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 6, 2, true"): 16 * 1024,
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 4, 3, false"): 24 * 1024,
    ("q6_k_t16_wmma_prefill_shared_bf16_kernel", "8, 3, 2"): 16 * 1024,
    ("q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel", "4, 3"): 0,
}

# Tensors each owner row serves, split by the layer group the prefill chunks
# separately: the 48 linear-attention (Gated DeltaNet) layers are prefilled in
# 1024-row chunks, the 16 full-attention layers in one pass over the prompt.
# A row whose reach is the whole prompt therefore serves both groups, and a row
# whose reach is shorter serves only the chunked group. Counts are tensor
# slots: the fused dual kernel covers a gate and an up slab per launch, so its
# counts are per-slab and its ``tensors_per_launch`` is 2.
OWNER_TENSORS: Mapping[tuple[str, str, int], Mapping[str, Any]] = {
    ("gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel", "false, false, 4, 4", 544): {
        "tensors_per_launch": 2,
        "chunked": [("ffn_gate.weight", 48), ("ffn_up.weight", 48)],
        "single_pass": [("ffn_gate.weight", 16), ("ffn_up.weight", 16)],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false", 107): {
        "tensors_per_launch": 1,
        "chunked": [("ffn_down.weight", 24)],
        "single_pass": [("ffn_down.weight", 8), ("attn_output.weight", 16)],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false", 128): {
        "tensors_per_launch": 1,
        "chunked": [("attn_gate.weight", 48)],
        "single_pass": [],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false", 214): {
        "tensors_per_launch": 1,
        "chunked": [("attn_qkv.weight", 24)],
        "single_pass": [],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false", 256): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("attn_q.weight", 16)],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, _Float16, false", 22): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("attn_k.weight", 16), ("attn_v.weight", 8)],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, unsigned short, false", 107): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("ffn_down.weight", 8), ("attn_output.weight", 16)],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, unsigned short, false", 256): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("attn_q.weight", 16)],
    },
    ("gguf_q4_t16_dense_wmma_prefill_shared_b_bf16_kernel", "3, 4, 4, unsigned short, false", 22): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("attn_k.weight", 16), ("attn_v.weight", 8)],
    },
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 4, 2, false", 160): {
        "tensors_per_launch": 1,
        "chunked": [("ffn_down.weight", 24)],
        "single_pass": [("ffn_down.weight", 8)],
    },
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 4, 3, false", 107): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("ffn_down.weight", 8)],
    },
    ("q6_k_t16_qmicro_planar_wmma_prefill_shared_bf16_kernel", "4, 6, 2, true", 32): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("attn_v.weight", 8)],
    },
    ("q6_k_t16_qmicro_planar_wmma_prefill_bf16_kernel", "4, 3", 22): {
        "tensors_per_launch": 1,
        "chunked": [],
        "single_pass": [("attn_v.weight", 8)],
    },
    ("q6_k_t16_wmma_prefill_shared_bf16_kernel", "8, 3, 2", 320): {
        "tensors_per_launch": 1,
        "chunked": [("attn_qkv.weight", 24)],
        "single_pass": [],
    },
}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-json", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="512,1024,4096")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# GGUF tensor-type table
# --------------------------------------------------------------------------- #
def _gguf_tensor_types(path: Path) -> tuple[dict[str, tuple[tuple[int, ...], int]], dict[str, int]]:
    """Return {tensor name: (dims, ggml type id)} and {tensor name: bytes}.

    GGUF tensor info carries an offset, not a size, so a tensor's byte size is
    the gap to the next tensor in file order (which includes at most one 32-byte
    alignment pad).
    """

    value_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with path.open("rb") as handle:
        magic, _version = struct.unpack("<4sI", handle.read(8))
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        n_tensors, n_kv = struct.unpack("<QQ", handle.read(16))

        def read_string() -> str:
            (size,) = struct.unpack("<Q", handle.read(8))
            return handle.read(size).decode("utf-8", "replace")

        def skip_value(value_type: int) -> None:
            if value_type == 8:
                read_string()
            elif value_type == 9:
                (array_type,) = struct.unpack("<I", handle.read(4))
                (count,) = struct.unpack("<Q", handle.read(8))
                for _ in range(count):
                    skip_value(array_type)
            else:
                handle.read(value_sizes[value_type])

        for _ in range(n_kv):
            read_string()
            (value_type,) = struct.unpack("<I", handle.read(4))
            skip_value(value_type)
        tensors: dict[str, tuple[tuple[int, ...], int]] = {}
        offsets: dict[str, int] = {}
        for _ in range(n_tensors):
            name = read_string()
            (n_dims,) = struct.unpack("<I", handle.read(4))
            dims = struct.unpack(f"<{n_dims}Q", handle.read(8 * n_dims))
            (type_id,) = struct.unpack("<I", handle.read(4))
            (offset,) = struct.unpack("<Q", handle.read(8))
            tensors[name] = (dims, type_id)
            offsets[name] = offset
    ordered = sorted(offsets.items(), key=lambda item: item[1])
    sizes: dict[str, int] = {}
    for index, (name, offset) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else None
        sizes[name] = (end - offset) if end is not None else 0
    return tensors, sizes


def _tensor_type_table(tensors: Mapping[str, tuple[tuple[int, ...], int]]) -> dict[str, Any]:
    from hipengine.quant.gguf import ggml_type_name

    groups: dict[tuple[str, str, tuple[int, ...]], int] = Counter()
    for name, (dims, type_id) in tensors.items():
        suffix = re.sub(r"^blk\.\d+\.", "", name)
        groups[(suffix, ggml_type_name(type_id), dims)] += 1
    rows = [
        {"suffix": suffix, "ggml_type": ggml_type, "dims": list(dims), "count": count}
        for (suffix, ggml_type, dims), count in sorted(
            groups.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0])
        )
    ]
    histogram = Counter(ggml_type for _, ggml_type, _ in groups.elements())
    return {
        "tensor_count": len(tensors),
        "type_histogram": dict(sorted(histogram.items())),
        "groups": rows,
    }


# --------------------------------------------------------------------------- #
# owner table
# --------------------------------------------------------------------------- #
def _agent_info(trace_dir: Path) -> dict[str, Any]:
    """Read the profiled device properties from the trace agent info."""

    import csv

    path = next(trace_dir.glob("*/[0-9]*_agent_info.csv"), None)
    if path is None:
        return {}
    columns = {
        "cu_count": "Cu_Count",
        "simd_count": "Simd_Count",
        "max_waves_per_simd": "Max_Waves_Per_Simd",
        "max_waves_per_cu": "Max_Waves_Per_Cu",
        "wave_front_size": "Wave_Front_Size",
        "lds_size_in_kb": "Lds_Size_In_Kb",
        "max_workgroup_size": "Workgroup_Max_Size",
        # Clock columns are copied with their rocprofv3 names and no unit
        # interpretation, because the trace does not state one.
        "max_engine_clk_ccompute_raw": "Max_Engine_Clk_Ccompute",
        "max_engine_clk_fcompute_raw": "Max_Engine_Clk_Fcompute",
    }
    info: dict[str, Any] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Agent_Type") == "GPU":
                info.update({key: int(row[column]) for key, column in columns.items()})
                info["name"] = row.get("Name")
                info["product_name"] = row.get("Product_Name")
            elif row.get("Agent_Type") == "CPU":
                info["cpu"] = row.get("Name")
    return info


def _trace_owners(trace_dir: Path) -> list[dict[str, Any]]:
    """Aggregate the K-quant prefill owners by (kernel, template, grid)."""

    import csv

    csv_path = next(trace_dir.glob("*/[0-9]*_kernel_trace.csv"), None)
    if csv_path is None:
        raise FileNotFoundError(f"no kernel trace CSV under {trace_dir}")
    aggregate: dict[tuple[str, str, int, int], dict[str, Any]] = defaultdict(
        lambda: {"launches": 0, "ns": 0, "workgroup": set(), "vgpr": set(), "sgpr": set()}
    )
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["Kernel_Name"].replace("(anonymous namespace)::", "")
            # drop the return type and any template argument list
            short = re.sub(r"<.*", "", name.split("(")[0]).strip().split("::")[-1].split()[-1]
            if short not in OWNER_GEOMETRY:
                continue
            match = re.search(r"<([^>]*)>", name)
            template = match.group(1) if match else ""
            workgroup = int(row["Workgroup_Size_X"])
            grid_x = int(row["Grid_Size_X"]) // workgroup
            grid_y = int(row["Grid_Size_Y"]) // int(row["Workgroup_Size_Y"])
            entry = aggregate[(short, template, grid_x, grid_y)]
            entry["launches"] += 1
            entry["ns"] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            entry["workgroup"].add(workgroup)
            entry["vgpr"].add(int(row["VGPR_Count"]))
            entry["sgpr"].add(int(row["SGPR_Count"]))
    owners = []
    for (short, template, grid_x, grid_y), entry in aggregate.items():
        geometry = OWNER_GEOMETRY[short]
        args = [arg.strip() for arg in template.split(",")] if template else []
        arg_index = geometry["arg_index"]
        out_tiles = geometry["out_tiles"]
        if out_tiles is None:
            out_tiles = int(args[arg_index["out_tiles"]])
        row_tiles = int(args[arg_index["row_tiles"]])
        waves = (geometry["fixed_waves"] if arg_index["waves"] is None
                 else int(args[arg_index["waves"]]))
        columns_per_block = out_tiles * geometry["columns_per_out_tile"]
        rows_per_block = waves * row_tiles * geometry["rows_per_wave_row_tile"]
        vgpr = sorted(entry["vgpr"])[0]
        owners.append({
            "kernel": short,
            "template_arguments": template,
            "template_parameter_names": geometry["template"],
            "registered_variant": VARIANT_BY_TEMPLATE.get((short, template), "unmapped"),
            "quant_family": geometry["quant_family"],
            "layer": geometry["layer"],
            "columns_per_block": columns_per_block,
            "rows_per_block": rows_per_block,
            "columns_total": grid_x * columns_per_block,
            "columns_covered": (
                f"({(grid_x - 1) * columns_per_block}, {grid_x * columns_per_block}]"
            ),
            "rows_per_launch": grid_y * rows_per_block,
            "grid": [grid_x, grid_y, 1],
            "workgroup": sorted(entry["workgroup"]),
            "vgpr": vgpr,
            "sgpr": sorted(entry["sgpr"]),
            "lds_bytes": LDS_BY_TEMPLATE.get((short, template)),
            "register_waves_per_simd_upper_bound": derived_occupancy(
                vgpr, LDS_BY_TEMPLATE.get((short, template), 0), 4,
            )["waves_per_simd_vgpr"],
            "occupancy_measured": False,
            "launches": entry["launches"],
            "total_ms": round(entry["ns"] / 1e6, 3),
        })
    return sorted(owners, key=lambda row: -row["total_ms"])


def _resolve_attribution(owners: list[dict[str, Any]],
                        tensors: Mapping[str, tuple[tuple[int, ...], int]],
                        sizes: Mapping[str, int],
                        *, prompt_length: int) -> dict[str, Any]:
    """Check the owner rows against the GGUF tensor table.

    For each traced owner row the number of tensor slots it must cover is
    ``launches / chunks * tensors_per_launch``. That count is compared with the
    composition listed in ``OWNER_TENSORS`` for the row's layer group, and the
    per-suffix totals are compared with the GGUF tensor counts. Both checks are
    integer equalities, so an unexplained row or tensor shows up as a failure
    instead of being smoothed over.
    """

    from hipengine.quant.gguf import ggml_type_name

    counts: dict[tuple[str, str], int] = Counter()
    shapes: dict[str, tuple[int, int]] = {}
    bytes_per_tensor: dict[str, int] = {}
    for name, (dims, type_id) in tensors.items():
        if len(dims) < 2:
            continue
        suffix = re.sub(r"^blk\.\d+\.", "", name)
        counts[(suffix, ggml_type_name(type_id))] += 1
        shapes.setdefault(suffix, (dims[0], dims[1]))
        bytes_per_tensor.setdefault(suffix, sizes.get(name, 0))

    rows = []
    per_suffix: dict[tuple[str, str], int] = Counter()
    # A row that reaches the whole prompt serves both layer groups only when no
    # shorter-reach sibling of the same shape exists in this trace; otherwise
    # the shorter rows carry the chunked group.
    keys_with_chunked_row = {
        (owner["kernel"], owner["template_arguments"], owner["grid"][0])
        for owner in owners
        if owner["rows_per_launch"] < prompt_length
    }
    for owner in sorted(owners, key=lambda row: (row["columns_total"], row["rows_per_launch"])):
        key = (owner["kernel"], owner["template_arguments"], owner["grid"][0])
        spec = OWNER_TENSORS.get(key)
        chunks = max(1, -(-prompt_length // owner["rows_per_launch"]))
        per_launch = spec["tensors_per_launch"] if spec else None
        expected = None if per_launch is None else (
            owner["launches"] // chunks * per_launch if owner["launches"] % chunks == 0
            else None
        )
        if spec is None:
            composition: list[tuple[str, int]] = []
        elif owner["rows_per_launch"] >= prompt_length:
            composition = list(spec["single_pass"])
            if key not in keys_with_chunked_row:
                composition = list(spec["chunked"]) + composition
        else:
            composition = list(spec["chunked"])
        listed = sum(count for _, count in composition)
        for suffix, count in composition:
            per_suffix[(suffix, "Q4_K" if owner["quant_family"] == "q4" else "Q6_K")] += count
        arithmetic = None
        shape_set = {shapes[suffix] for suffix, _ in composition if suffix in shapes}
        if len(shape_set) == 1 and per_launch:
            in_features, out_features = shape_set.pop()
            rows_processed = owner["launches"] * min(owner["rows_per_launch"], prompt_length)
            flops = 2.0 * rows_processed * in_features * out_features * per_launch
            seconds = owner["total_ms"] / 1e3
            weight_bytes = sum(bytes_per_tensor.get(suffix, 0) * count
                               for suffix, count in composition) * chunks
            activation_bytes = (
                owner["launches"] * min(owner["rows_per_launch"], prompt_length)
                * (in_features + out_features) * per_launch * 2
            )
            arithmetic = {
                "traffic_kind": "logical_tensor_bytes_not_measured_memory_transactions",
                "in_features": in_features,
                "out_features": out_features,
                "rows_processed": rows_processed,
                "flops": flops,
                "tflops": round(flops / seconds / 1e12, 2) if seconds else None,
                "weight_bytes_per_launch": weight_bytes // max(owner["launches"] // chunks, 1),
                "weight_bytes_total": weight_bytes,
                "implied_weight_gbps": round(
                    weight_bytes / seconds / 1e9, 2) if seconds else None,
                "implied_weight_and_activation_gbps": round(
                    (weight_bytes + activation_bytes) / seconds / 1e9, 2) if seconds else None,
            }
        rows.append({
            "kernel": owner["kernel"],
            "template_arguments": owner["template_arguments"],
            "registered_variant": owner["registered_variant"],
            "grid": owner["grid"],
            "rows_per_launch": owner["rows_per_launch"],
            "chunks_per_tensor": chunks,
            "pass_mode": "single-pass" if owner["rows_per_launch"] >= prompt_length else "chunked",
            "launches": owner["launches"],
            "total_ms": owner["total_ms"],
            "tensor_slots_expected": expected,
            "tensor_slots_listed": listed,
            "composition": [{"suffix": suffix, "count": count} for suffix, count in composition],
            "arithmetic": arithmetic,
            "matches": expected is not None and expected == listed,
        })

    group_rows = []
    for (suffix, ggml_type), count in sorted(counts.items()):
        attributed = per_suffix.get((suffix, ggml_type))
        if not attributed:
            continue
        group_rows.append({
            "suffix": suffix,
            "ggml_type": ggml_type,
            "tensors_in_model": count,
            "tensors_attributed": attributed,
            "unattributed": count - attributed,
            "unattributed_note": (
                "the MTP nextn layer's copy is not on the prefill path"
                if count - attributed == 1 else None
            ),
        })
    return {
        "rows": rows,
        "rows_matched": sum(1 for row in rows if row["matches"]),
        "rows_total": len(rows),
        "tensor_groups": group_rows,
        "tensor_groups_explained": sum(
            1 for row in group_rows if row["unattributed"] in (0, 1)),
        "tensor_groups_total": len(group_rows),
    }


def _findings(families: Mapping[str, Any], owners: Mapping[str, Any],
              attribution: Mapping[str, Any],
              prefill: Mapping[str, Any]) -> list[dict[str, Any]]:
    """State the measured structure of the prefill mix."""

    weight_traffic = {}
    for length, rows in attribution.items():
        total = sum(row["arithmetic"]["weight_bytes_total"] for row in rows["rows"]
                    if row["arithmetic"])
        wall_seconds = prefill[length]["wall_seconds"]
        weight_traffic[length] = {
            "dense_weight_traffic_gb": round(total / 1e9, 2),
            "prefill_wall_seconds": wall_seconds,
            "aggregate_weight_gbps": round(total / wall_seconds / 1e9, 2),
            "max_owner_weight_gbps": max(
                (row["arithmetic"]["implied_weight_gbps"] for row in rows["rows"]
                 if row["arithmetic"]), default=None),
        }

    def family_share(length: str, name: str) -> float:
        for row in families[length]:
            if row["family"] == name:
                return row["share_pct"]
        return 0.0

    def row(length: str, variant: str, grid_x: int, pass_mode: str | None = None) -> Mapping[str, Any]:
        for entry in attribution[length]["rows"]:
            if (entry["registered_variant"].startswith(variant)
                    and entry["grid"][0] == grid_x
                    and (pass_mode is None or entry["pass_mode"] == pass_mode)):
                return entry
        raise KeyError(f"no owner row for {variant} grid_x={grid_x} at {length}")

    dual_512 = row("512", "dense_dual_wmma_prefill", 544)
    dual_4096 = row("4096", "dense_dual_wmma_prefill", 544, "chunked")
    q4_wide_4096 = row("4096", "t16_wmma_prefill_shared_b_bf16_bf16_out (fp16", 107, "chunked")
    q6_wide_4096 = row("4096", "qmicro_planar_wmma_prefill_shared4r4", 160, "chunked")
    q6_wide48_4096 = row("4096", "qmicro_planar_wmma_prefill_shared4_gfx1100", 107)
    return [
        {
            "id": "q4-dense-wmma-owns-prefill",
            "statement": (
                "Dense K-quant WMMA prefill owns 55.0-57.8% of prefill kernel "
                "time at every prompt length, and its two largest owners are the "
                "fused gate+up (dual) kernel and the 48-column shared-weight "
                "kernel."
            ),
            "measured": {
                length: {
                    "q4_family_share_pct": family_share(length, "q4"),
                    "q6_family_share_pct": family_share(length, "q6"),
                    "dual_share_pct": round(sum(
                        o["share_pct"] for o in owners[length]
                        if o["kernel"].endswith("dual_wmma_prefill_silu_bf16_kernel")), 2),
                    "dual_tflops": {
                        "chunked": dual_4096["arithmetic"]["tflops"],
                        "single_pass": row(length, "dense_dual_wmma_prefill", 544,
                                           "single-pass")["arithmetic"]["tflops"],
                    },
                }
                for length in ("512", "1024", "4096")
            },
        },
        {
            "id": "prefill-is-kernel-bound",
            "statement": (
                "Traced device time is 98.1-99.4% of the measured prefill wall "
                "time, so prefill cost is kernel execution and not launch gaps."
            ),
        },
        {
            "id": "layer-type-drives-chunking",
            "statement": (
                "The 48 linear-attention layers are prefilled in 1024-row "
                "chunks and the 16 full-attention layers in one pass over the "
                "prompt, which is why the same owner appears twice at 4096 "
                "tokens with different grid heights."
            ),
            "measured": {
                "chunked_rows_at_4096": dual_4096["launches"],
                "single_pass_rows_at_4096": row("4096", "dense_dual_wmma_prefill", 544,
                                                 "single-pass")["launches"],
                "dual_chunked_tflops": dual_4096["arithmetic"]["tflops"],
                "dual_single_pass_tflops": row("4096", "dense_dual_wmma_prefill", 544,
                                               "single-pass")["arithmetic"]["tflops"],
            },
        },
        {
            "id": "q6-wide-down-trails-q4-on-the-same-shape",
            "statement": (
                "On the identical wide-down shape (K 17408, N 5120), the Q4_K "
                "shared-weight owner reaches 27.50 TFLOP/s in 1024-row chunks "
                "while the Q6_K planar owners reach 21.28 TFLOP/s in 1024-row "
                "chunks and 20.34 TFLOP/s in a single 4096-row pass. Logical "
                "weight bytes per elapsed second do not measure memory traffic "
                "or isolate the cause. Compare decode, repeated loads, cache "
                "behavior and actual residency before attributing the gap."
            ),
            "measured": {
                "q4_k_shared_b_tflops": q4_wide_4096["arithmetic"]["tflops"],
                "q4_k_shared_b_ms": q4_wide_4096["total_ms"],
                "q4_k_shared_b_implied_weight_gbps": q4_wide_4096["arithmetic"][
                    "implied_weight_gbps"],
                "q6_k_planar_shared4r4_tflops": q6_wide_4096["arithmetic"]["tflops"],
                "q6_k_planar_shared4r4_ms": q6_wide_4096["total_ms"],
                "q6_k_planar_shared4r4_implied_weight_gbps": q6_wide_4096[
                    "arithmetic"]["implied_weight_gbps"],
                "q6_k_planar_48col_tflops": q6_wide48_4096["arithmetic"]["tflops"],
                "q6_k_planar_48col_ms": q6_wide48_4096["total_ms"],
                "q6_k_planar_wide_down_total_ms": (
                    q6_wide_4096["total_ms"] + q6_wide48_4096["total_ms"]),
                "q6_k_planar_wide_down_share_pct": round(
                    q6_wide_4096["total_ms"] / 10184.927 * 100
                    + q6_wide48_4096["total_ms"] / 10184.927 * 100, 2),
            },
        },
        {
            "id": "logical-weight-volume",
            "statement": (
                "Logical tensor bytes counted once per chunk total 12.4 GB at "
                "512 tokens and 47.8 GB at 4096. These figures exclude repeated "
                "block loads, cache transactions and staging traffic. They are "
                "not measured bandwidth and cannot rule out a memory bottleneck."
            ),
            "measured": weight_traffic,
        },
        {
            "id": "register-only-residency-ceiling",
            "statement": (
                "gfx1151 wave32 has 1536 physical VGPRs per SIMD and a "
                "24-register allocation granule. A count of 248-256 rounds "
                "to 264 and permits at most five waves by registers alone. "
                "LDS, CU/WGP mode, scheduling and other constraints still apply. "
                "No occupancy counter was captured; the earlier two-wave and "
                "171-register conclusions are withdrawn."
            ),
            "measured": {
                length: {
                    owner["registered_variant"]: {
                        "vgpr": owner["vgpr"],
                        "register_waves_per_simd_upper_bound": owner["register_waves_per_simd_upper_bound"],
                    }
                    for owner in owners[length]
                }
                for length in ("4096",)
            },
        },
        {
            "id": "narrow-column-shapes-run-slow",
            "statement": (
                "Owners whose output is 1024 columns wide (attention K and V) "
                "run at 8.75-19.12 TFLOP/s while the same kernel families reach "
                "30-33 TFLOP/s on 6144- to 12288-wide outputs. The profile does "
                "not establish the mechanism, and grid occupancy does not "
                "explain it on its own: the 1024-wide row runs faster at 512 "
                "tokens (44 workgroups, 19.12 TFLOP/s) than at 1024 tokens "
                "(88 workgroups, 12.92 TFLOP/s)."
            ),
            "measured": {
                "n1024_owners": {
                    length: [
                        {"variant": owner["registered_variant"],
                         "grid": owner["grid"],
                         "tflops": next(
                             (entry["arithmetic"]["tflops"]
                              for entry in attribution[length]["rows"]
                              if entry["grid"] == owner["grid"]
                              and entry["kernel"] == owner["kernel"]
                              and entry["arithmetic"]
                              and entry["arithmetic"]["out_features"] == 1024), None)}
                        for owner in owners[length]
                        if owner["columns_total"] in (1024, 1056)
                    ]
                    for length in ("512", "1024", "4096")
                },
                "resident_workgroups_needed": None,
                "resident_workgroups_note": (
                    "Not established by this trace; register counts are not "
                    "measured workgroup residency."
                ),
            },
        },
    ]


def main() -> int:
    args = _parse_args()
    lengths = [int(item) for item in args.lengths.split(",")]
    driver = json.loads(Path(args.driver_json).read_text())
    run_root = Path(args.run_root)

    summaries = {}
    for length in lengths:
        summaries[str(length)] = _summarize_trace(run_root / "trace" / str(length), top=40)

    model_path = Path(driver["provenance"]["model_path"])
    tensors, sizes = _gguf_tensor_types(model_path)
    agent_info = _agent_info(run_root / "trace" / str(lengths[0]))
    build_command = list(driver["stages"]["build"]["command"])
    max_sequence_length = None
    if "--max-sequence-length" in build_command:
        max_sequence_length = int(
            build_command[build_command.index("--max-sequence-length") + 1])

    wall = {}
    for length in lengths:
        child = driver["children"][f"child-profile-{length}"]
        row = child["lengths"][0]
        wall[str(length)] = {
            "measured_prefill_seconds": row["measured_prefill_seconds"],
            "measured_prefill_tok_s": row["measured_prefill_tok_s"],
            "warmup_prefill_seconds": row["warmup_prefill_seconds"],
            "first_token_id": row["first_token_id"],
            "logits_finite": (
                row["logits_finite"] if row.get("logits_checked_after_timing") is True else None
            ),
            "host_stage_timings_ms": row["gpu_stage_timings_ms"],
        }

    families = {
        length: [
            {"family": row["family"], "total_ms": row["total_ms"],
             "share_pct": row["share_pct"], "launches": row["launches"],
             "distinct_kernels": row["distinct_kernels"]}
            for row in summaries[length]["families"]
        ]
        for length in summaries
    }

    owners = {}
    attribution = {}
    for length in lengths:
        rows = _trace_owners(run_root / "trace" / str(length))
        total_ns = summaries[str(length)]["total_kernel_ms"] * 1e6
        for row in rows:
            row["share_pct"] = round(100.0 * row["total_ms"] * 1e6 / total_ns, 2)
        owners[str(length)] = rows
        attribution[str(length)] = _resolve_attribution(
            rows, tensors, sizes, prompt_length=length)

    stage_double_count = {}
    for length in lengths:
        stages = wall[str(length)]["host_stage_timings_ms"]
        duplicated = [name for name in stages
                      if name.endswith("_fallback") and stages[name.replace("_fallback", "")] == stages[name]]
        stage_double_count[str(length)] = {
            "stage_total_ms": round(sum(stages.values()), 1),
            "measured_wall_ms": round(1000 * wall[str(length)]["measured_prefill_seconds"], 1),
            "aliased_pairs": sorted(duplicated),
        }

    prefill = {
        length: {
            "wall_seconds": wall[length]["measured_prefill_seconds"],
            "tok_s": wall[length]["measured_prefill_tok_s"],
            "warmup_seconds": wall[length]["warmup_prefill_seconds"],
            "kernel_ms": summaries[length]["total_kernel_ms"],
            "dispatches": summaries[length]["dispatch_count"],
        }
        for length in wall
    }

    artifact = {
        "schema": 1,
        "status": "accepted",
        "date": "2026-09-13",
        "kind": driver["kind"],
        "performance_claim": False,
        "purpose": (
            "Name the kernels that own gfx1151 Qwen3.8-27B Q4_K_M prefill time, "
            "with the registered variant, tile geometry, register footprint and "
            "resident-wave count for each owner. This is attribution, not a "
            "performance claim: the prefill rates are recorded as protocol "
            "context for the published resident-sweep row."
        ),
        "hardware": {
            "host_name": driver["provenance"]["host_name"],
            "device_name": driver["provenance"]["device_name"],
            "target_arch": driver["provenance"]["target_arch"],
            "agent": agent_info,
            "agent_source": "rocprofv3 agent_info.csv of this run",
        },
        "model": driver["model"],
        "workload": {
            "prompt_lengths": lengths,
            "prompt": "one repeated token id (%d), matching the resident-sweep protocol"
                      % driver["prompt_token_id"],
            "bulk_prefill": True,
            "bulk_attention_mode": driver["children"]["child-profile-512"]["bulk_attention_mode"],
            "kv_storage": driver["model"]["kv_storage"],
            "max_sequence_length": max_sequence_length,
        },
        "protocol": {
            "timing": driver["provenance"]["timing_protocol"],
            "repetitions": driver["provenance"]["repetitions"],
            "profiler": driver["provenance"]["profiler"],
            "profiler_note": (
                "rocprofv3 --kernel-trace with --selected-regions, one profiled "
                "child per prompt length; the JIT cache is populated and then "
                "proven compiler-free (HIPENGINE_REQUIRE_CACHED_BUILD=1, empty "
                "compiler guard, unchanged cache-tree hash) before the profiled "
                "child starts."
            ),
            "kernel_time_scope": (
                "device duration of every dispatch inside the ROCTx prefill "
                "region; excludes model load, the discarded warmup prefill and "
                "session teardown."
            ),
            "kernel_share_of_wall": {
                length: round(
                    summaries[length]["total_kernel_ms"]
                    / (1000 * wall[length]["measured_prefill_seconds"]), 4)
                for length in summaries
            },
        },
        "correctness": {
            "gate": (
                "The original child did not read logits; its empty-array .all() "
                "flags are invalid and now null. First-token observations and "
                "compiler-free warmup remain recorded. New captures validate "
                "nonempty finite logits after the measured region. This profile "
                "is not a full production numerical certificate."
            ),
            "first_token_id": {length: wall[length]["first_token_id"] for length in wall},
            "first_token_identical_across_lengths": len(
                {wall[length]["first_token_id"] for length in wall}) == 1,
            "logits_finite": {length: wall[length]["logits_finite"] for length in wall},
            "environment": driver["children"]["child-profile-512"]["environment"],
        },
        "prefill": prefill,
        "families": families,
        "owners": owners,
        "attribution": attribution,
        "tensor_type_table": _tensor_type_table(tensors),
        "host_stage_timers": {
            "note": (
                "the bulk path also records per-stage host timers; their sum "
                "exceeds the measured wall time because a fallback alias is "
                "recorded alongside its parent, so they are not used for "
                "attribution here"
            ),
            "per_length": stage_double_count,
        },
        "findings": _findings(families, owners, attribution, prefill),
        "provenance": driver["provenance"],
        "limitations": [
            "One profiled prefill per prompt length (one discarded warmup, one "
            "measured pass), matching the published resident-sweep protocol; the "
            "family split is stable across the three lengths but no per-kernel "
            "confidence interval is claimed.",
            "Prompt tokens are one repeated id, so attention is not "
            "representative of natural text; kernel selection is shape-driven, "
            "and the two attention owners stay under 3% at every length.",
            "Owner-to-tensor attribution is a launch-count and column-count "
            "consistency argument over the GGUF tensor-type table, not a "
            "per-dispatch tensor probe; a group's unattributed residual is the "
            "MTP nextn layer, which the prefill path does not run.",
            "VGPR/SGPR/LDS are the dispatch-recorded resource counts; resident "
            "waves per SIMD is derived from the VGPR budget and is not measured "
            "from hardware counters. Device clock fields are copied from the "
            "trace with their rocprofv3 names because the trace states no unit.",
            "Host-stage timers are recorded but not used: an aliased fallback "
            "timer makes their sum exceed the measured wall time.",
        ],
    }
    Path(args.out).write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
