#!/usr/bin/env python3
"""MX-1 first experiment: is the F16-operand routed-expert route admissible?

The competitor route for the routed experts (``mmb_routed_kernel``,
``RoutedGatedF16Gemm``) decodes the quantized expert weights to **F16** in LDS,
casts the activation to F16 once, and runs F16 WMMA with F32 accumulation. That
removes activation packing and the iu8 risk/repair machinery entirely. Our
exact path dequantizes to F32 and accumulates with F32 FMA plus a repair pass.

Before building that kernel this packet answers the numerical question only:
**does decoding the real expert weights to F16, and casting the real
activations to F16, stay inside the error the path already carries?**

The comparison basis is the existing BF16 publication floor, not an arbitrary
tolerance. The expert output is published to BF16 downstream, so an operand
change whose error is far below that rounding cannot move the model beyond what
the current path already does. Part A is activation-independent and covers every
expert quant in the model; Part B runs the full expert_down projection on real
natural-input activations captured from a live prefill.

Diagnostic only. No runtime default changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.loading.gguf import GGUFReader, discover_gguf_files  # noqa: E402
from hipengine.quant.gguf import (  # noqa: E402
    GGMLQuantizationType,
    dequantize_gguf_data,
)

DEFAULT_MODEL_ROOT = Path(
    "/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_CAPTURE = Path("/home/lhl/tmp/q51-down-input-general_en-p4096.npz")
F16_MAX = 65504.0
F16_MIN_NORMAL = 6.103515625e-05

QUANT_NAMES = {
    "Q4_K": GGMLQuantizationType.Q4_K,
    "Q5_K": GGMLQuantizationType.Q5_K,
    "Q5_1": GGMLQuantizationType.Q5_1,
    "Q8_0": GGMLQuantizationType.Q8_0,
}


def _bf16_round(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 and back, the publication floor of this path."""
    bits = values.astype(np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def _error_stats(exact: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = np.abs(candidate.astype(np.float64) - exact.astype(np.float64))
    scale = np.abs(exact.astype(np.float64))
    denom = np.where(scale > 0, scale, 1.0)
    return {
        "max_abs": float(diff.max()),
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "max_rel": float((diff / denom).max()),
        "rms_rel": float(np.sqrt(np.mean((diff / denom) ** 2))),
    }


def _f16_range_report(values: np.ndarray) -> dict[str, Any]:
    magnitude = np.abs(values.astype(np.float64))
    return {
        "max_magnitude": float(magnitude.max()),
        "overflow_count": int(np.count_nonzero(magnitude > F16_MAX)),
        "subnormal_count": int(
            np.count_nonzero((magnitude > 0) & (magnitude < F16_MIN_NORMAL))
        ),
        "zero_count": int(np.count_nonzero(magnitude == 0)),
        "finite_after_cast": bool(np.isfinite(values.astype(np.float16)).all()),
    }


def _reader_for(readers: list[GGUFReader], name: str) -> GGUFReader:
    for reader in readers:
        if any(tensor.name == name for tensor in reader.info.tensors):
            return reader
    raise KeyError(name)


def part_a(readers: list[GGUFReader], *, experts_sampled: int, seed: int) -> list[dict]:
    """Activation-independent F16 weight-decode error, per expert quant."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for layer, suffix, quant_name in _expert_tensor_inventory(readers):
        name = f"blk.{layer}.{suffix}.weight"
        reader = _reader_for(readers, name)
        info = reader.tensor_info(name)
        raw = reader.tensor_data(name)
        quant = QUANT_NAMES[quant_name]
        n_experts, per_expert_rows = _expert_shape(info, quant)
        block_bytes = raw.nbytes // (n_experts * per_expert_rows)
        data = raw.reshape(n_experts, per_expert_rows, block_bytes)
        picked = rng.choice(n_experts, size=min(experts_sampled, n_experts), replace=False)
        exact = dequantize_gguf_data(data[picked].reshape(-1, block_bytes), quant)
        exact = np.asarray(exact, dtype=np.float32)
        candidate = exact.astype(np.float16).astype(np.float32)
        stats = _error_stats(exact, candidate)
        rows.append(
            {
                "layer": layer,
                "suffix": suffix,
                "tensor": name,
                "quant": quant_name,
                "shape": [int(v) for v in info.shape],
                "experts_sampled": int(len(picked)),
                "weight_error_f16_vs_f32": stats,
                "weight_f16_range": _f16_range_report(exact),
            }
        )
    return rows


def _expert_tensor_inventory(readers: list[GGUFReader]) -> list[tuple[int, str, str]]:
    seen: dict[tuple[int, str], str] = {}
    for reader in readers:
        for tensor in reader.info.tensors:
            parts = tensor.name.split(".")
            if len(parts) != 4 or parts[0] != "blk" or not parts[2].endswith("_exps"):
                continue
            key = (int(parts[1]), parts[2])
            seen.setdefault(key, tensor.ggml_type_name)
    return [(layer, suffix, quant) for (layer, suffix), quant in sorted(seen.items())]


def _expert_shape(info: Any, quant: GGMLQuantizationType) -> tuple[int, int]:
    """(experts, output rows per expert) from the GGML tensor shape.

    Expert tensors are (K, N, experts) in GGML order, so the per-expert output
    row count is the middle dimension and K is the first.
    """
    ggml = tuple(int(v) for v in (getattr(info, "ggml_shape", None) or info.shape))
    return ggml[2], ggml[1]


def _expert_in_features(info: Any) -> int:
    ggml = tuple(int(v) for v in (getattr(info, "ggml_shape", None) or info.shape))
    return ggml[0]


def part_b(
    readers: list[GGUFReader],
    capture_path: Path,
    *,
    max_experts: int | None,
    layers: set[int] | None = None,
) -> dict[str, Any]:
    """Full expert_down projection on real activations, F16 operands vs F32."""
    capture = np.load(capture_path, allow_pickle=True)
    x_bits = capture["x_bits"]
    expert_start = capture["expert_start"]
    # The captured tensor is the BF16 activation of the MoE expert path
    # (``..._bf16_bf16_out``), so the stored uint16 are BF16 bits: the high half
    # of the float32 representation.
    if x_bits.dtype != np.uint16:
        raise ValueError(f"expected uint16 BF16 bits, got {x_bits.dtype}")
    x = (x_bits.astype(np.uint32) << 16).view(np.float32)
    case_id = str(capture["case_id"])
    out_features = int(capture["out_features"])
    n_experts = len(expert_start) - 1

    per_quant: dict[str, list[dict]] = {}
    for layer in _down_layers(readers):
        if layers is not None and layer not in layers:
            continue
        name = f"blk.{layer}.ffn_down_exps.weight"
        reader = _reader_for(readers, name)
        info = reader.tensor_info(name)
        quant_name = info.ggml_type_name
        quant = QUANT_NAMES[quant_name]
        raw = reader.tensor_data(name)
        n_exp, per_expert_rows = _expert_shape(info, quant)
        block_bytes = raw.nbytes // (n_exp * per_expert_rows)
        data = raw.reshape(n_exp, per_expert_rows, block_bytes)
        if per_expert_rows != out_features:
            continue
        limit = n_exp if max_experts is None else min(max_experts, n_exp)
        exact_parts, cand_parts, counts = [], [], []
        weight_stats, act_stats = [], []
        for expert in range(limit):
            start, stop = int(expert_start[expert]), int(expert_start[expert + 1])
            if stop <= start:
                continue
            rows = x[start:stop]
            weight = np.asarray(
                dequantize_gguf_data(data[expert], quant), dtype=np.float32
            )
            cand_weight = weight.astype(np.float16)
            cand_rows = rows.astype(np.float16)
            exact_parts.append(rows @ weight.T)
            cand_parts.append(
                (cand_rows.astype(np.float32) @ cand_weight.astype(np.float32).T)
            )
            counts.append(stop - start)
            weight_stats.append(_f16_range_report(weight))
            act_stats.append(_f16_range_report(rows))
        exact = np.concatenate(exact_parts) if exact_parts else np.zeros((0, out_features))
        cand = np.concatenate(cand_parts) if cand_parts else np.zeros((0, out_features))
        floor = _bf16_round(exact)
        entry = {
            "layer": layer,
            "tensor": name,
            "quant": quant_name,
            "experts_measured": len(counts),
            "rows_measured": int(sum(counts)),
            "rows_per_expert_min": int(min(counts)) if counts else 0,
            "rows_per_expert_max": int(max(counts)) if counts else 0,
            "output_error_f16_vs_f32": _error_stats(exact, cand),
            "output_error_bf16_floor_vs_f32": _error_stats(exact, floor),
            "weight_f16_range": _merge_range(weight_stats),
            "activation_f16_range": _merge_range(act_stats),
        }
        entry["candidate_below_bf16_floor"] = (
            entry["output_error_f16_vs_f32"]["rms_abs"]
            <= entry["output_error_bf16_floor_vs_f32"]["rms_abs"]
        )
        per_quant.setdefault(quant_name, []).append(entry)
    return {
        "capture": str(capture_path),
        "capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
        "case_id": case_id,
        "rows": int(x.shape[0]),
        "in_features": int(x.shape[1]),
        "out_features": out_features,
        "experts": n_experts,
        "activation_source": (
            "real natural-input prefill capture, BF16 bits decoded to float32"
        ),
        "activation_bf16_exact": bool(
            np.array_equal(_bf16_round(x), x)
        ),
        "per_quant": per_quant,
    }


def _down_layers(readers: list[GGUFReader]) -> list[int]:
    layers = set()
    for reader in readers:
        for tensor in reader.info.tensors:
            if tensor.name.endswith(".ffn_down_exps.weight"):
                layers.add(int(tensor.name.split(".")[1]))
    return sorted(layers)


def _merge_range(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats:
        return {}
    return {
        "max_magnitude": max(s["max_magnitude"] for s in stats),
        "overflow_count": int(sum(s["overflow_count"] for s in stats)),
        "subnormal_count": int(sum(s["subnormal_count"] for s in stats)),
        "finite_after_cast": all(s["finite_after_cast"] for s in stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts-sampled", type=int, default=8)
    parser.add_argument("--max-experts", type=int, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="comma-separated layer indices for part B; default is every down layer",
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--skip-part-b", action="store_true")
    args = parser.parse_args()

    readers = [GGUFReader(path) for path in discover_gguf_files(args.model_root)]
    report: dict[str, Any] = {
        "schema": 1,
        "kind": "qwen4exp_mx1_f16_operand_packet",
        "performance_claim": False,
        "model_root": str(args.model_root),
        "command": sys.argv,
        "comparison_basis": (
            "The BF16 publication floor already present in the path. An operand "
            "change whose error is below that floor cannot move the model beyond "
            "what the current path already does."
        ),
        "part_a_weight_decode": part_a(
            readers, experts_sampled=args.experts_sampled, seed=args.seed
        ),
    }
    if not args.skip_part_b:
        report["part_b_real_activation_down"] = part_b(
            readers,
            args.capture,
            max_experts=args.max_experts,
            layers=(
                {int(v) for v in args.layers.split(",") if v.strip()}
                if args.layers
                else None
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(_summary(report), indent=2))


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    quant_weight: dict[str, dict[str, float]] = {}
    for row in report["part_a_weight_decode"]:
        stats = row["weight_error_f16_vs_f32"]
        current = quant_weight.setdefault(row["quant"], dict(stats))
        for key, value in stats.items():
            current[key] = max(current[key], value)
    out: dict[str, Any] = {"weight_decode_error_worst_by_quant": quant_weight}
    part_b = report.get("part_b_real_activation_down")
    if part_b:
        out["part_b"] = {
            quant: [
                {
                    "layer": entry["layer"],
                    "rows": entry["rows_measured"],
                    "rms_abs": entry["output_error_f16_vs_f32"]["rms_abs"],
                    "bf16_floor_rms_abs": entry["output_error_bf16_floor_vs_f32"]["rms_abs"],
                    "below_floor": entry["candidate_below_bf16_floor"],
                }
                for entry in entries
            ]
            for quant, entries in part_b["per_quant"].items()
        }
    return out


if __name__ == "__main__":
    main()
