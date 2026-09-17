"""Offline FP32 reference check for the captured layer-0 dense-MLP half.

Diagnostics only, CPU-only. Consumes the ``.npz`` written by
``scripts/tp2_bulk_vs_resident_layer0.py --save-dir`` together with the source
GGUF, and compares the captured resident-TP1 and TP2-bulk layer-0 MLP values
against an independent numpy/float32 reference computed from the *dequantized*
layer-0 weights.

What this separates
-------------------

The two routes run different down-projection schedules:

* resident TP1: one ``(17408 -> 5120)`` bf16 down projection over the whole
  intermediate, then one bf16 residual add;
* TP2 bulk: two ``(8704 -> 5120)`` bf16 partials, summed in f32 by the staged
  exchange, cast back to bf16, then one bf16 residual add.

Everything else (post-attention norm, the activation, the residual add) is
structurally identical. So the reference lets each contribution be measured
instead of asserted:

* ``activation_kernel_error`` - the resident fused pair+SiLU vs the shard
  gate/up/SiLU chain against the same reference activation;
* ``shard_kernel_error`` - each rank's bf16 down partial against the exact
  float32 partial of the same rows and the same weight slice;
* ``partial_rounding`` - the bf16 rounding of the two partials, isolated by
  comparing the summed rounded partials against the summed exact partials;
* ``reassociation`` - the same comparison with f32 partials, i.e. the part
  that is *not* the bf16 boundary.

The reference is numpy sgemm, not a bit-exact oracle for either kernel; the
informative quantity is therefore each route's error *against the same
reference*, plus the reference-level partial-rounding term.

Usage::

    python3 scripts/tp2_layer0_mlp_reference_check.py \
        --capture /tmp/tp2_layer0/tp2_layer0.npz \
        --prompt mixed_ja_en_translate --json /tmp/tp2_layer0_mlp_reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.tp2_layer0_mlp_capture import (  # noqa: E402
    bf16_round,
    bf16_ulp,
    reference_exact_partials,
    reference_full_width_mlp,
    reference_sharded_mlp,
)

DEFAULT_MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
LAYER0_MLP_TENSORS = (
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
)

#: dtype of every captured MLP field, from its producer call site.
BF16_FIELDS = frozenset(
    {
        "post_norm",
        "residual",
        "ffn_intermediate",
        "ffn_down",
        "gate",
        "up",
        "act",
        "down_partial",
        "cast",
        "out",
    }
)
F32_FIELDS = frozenset({"reduced"})


def _decode(raw: np.ndarray, *, dtype: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1)
    if dtype == "f32":
        return raw.view("<f4").astype(np.float32)
    bits = raw.view("<u2").astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _stat(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "rel": None}
    diff = np.abs(a - b)
    scale = max(float(np.abs(b).max()), 1e-12)
    return {
        "shape": list(a.shape),
        "rel": float(diff.max() / scale),
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean((a - b) ** 2))),
        "bit_equal": bool(np.array_equal(a, b)),
        "reference_absmax": float(np.abs(b).max()),
        "captured_absmax": float(np.abs(a).max()),
    }


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """bf16 ULP distance between two bf16-valued arrays."""

    a_bits = np.asarray(a, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    b_bits = np.asarray(b, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    distance = np.abs(a_bits.astype(np.int64) - b_bits.astype(np.int64))
    return {
        "cells": int(distance.size),
        "max_ulp": int(distance.max()) if distance.size else 0,
        "nonzero_cells": int((distance != 0).sum()),
        "gt_one_ulp_cells": int((distance > 1).sum()),
    }


def _bf16_rounding_report(captured: np.ndarray, exact: np.ndarray) -> dict:
    """Per-cell deviation against the local bf16 half-ULP of the exact value."""

    captured = np.asarray(captured, dtype=np.float32)
    exact = np.asarray(exact, dtype=np.float32)
    if captured.shape != exact.shape:
        return {"shape": [list(captured.shape), list(exact.shape)]}
    deviation = np.abs(captured - exact)
    half_ulp = np.array(
        [float(bf16_ulp(np.float32(v))) / 2.0 for v in exact.ravel()], dtype=np.float32
    ).reshape(exact.shape)
    tolerance = half_ulp + np.float32(1e-6)
    return {
        "cells": int(deviation.size),
        "nonzero_cells": int((deviation != 0).sum()),
        "over_half_ulp_cells": int((deviation > tolerance).sum()),
        "max_abs": float(deviation.max()),
        "max_half_ulp_ratio": float((deviation / np.maximum(half_ulp, 1e-30)).max()),
    }


def _sum_f32(arrays: list) -> np.ndarray:
    """Sum arrays in float32, left to right, exactly as the exchange does."""

    total = np.zeros_like(np.asarray(arrays[0], dtype=np.float32))
    for item in arrays:
        total = total + np.asarray(item, dtype=np.float32)
    return total


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


#: Feature width of every captured MLP field: ``hidden``, the full ``ffn``
#: intermediate, or one rank's ``ffn`` slice.
WIDTH_KIND = {
    "post_norm": "hidden",
    "residual": "hidden",
    "ffn_intermediate": "ffn",
    "ffn_down": "hidden",
    "gate": "shard",
    "up": "shard",
    "act": "shard",
    "down_partial": "hidden",
    "reduced": "hidden",
    "cast": "hidden",
    "out": "hidden",
}


def _field(capture: dict, tag: str, name: str) -> np.ndarray | None:
    key = f"{tag}.mlp.{name}"
    if key not in capture:
        return None
    dtype = "f32" if name in F32_FIELDS else "bf16"
    if name not in BF16_FIELDS and name not in F32_FIELDS:
        raise ValueError(f"no recorded dtype for captured field {name!r}")
    return _decode(capture[key], dtype=dtype)


def _dequantized_weights(model: str) -> dict[str, np.ndarray]:
    from hipengine.loading.gguf import GGUFReader

    reader = GGUFReader(str(model))
    gate, up, down = (reader.dequantize_tensor(name) for name in LAYER0_MLP_TENSORS)
    return {
        "gate": np.ascontiguousarray(gate, dtype=np.float32),
        "up": np.ascontiguousarray(up, dtype=np.float32),
        "down": np.ascontiguousarray(down, dtype=np.float32),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args(argv)

    capture = _load(Path(args.capture))
    tags = sorted(
        {key.split(".", 1)[0] for key in capture if ".mlp." in key}
    )
    teacher_tag = "teacher"
    rank_tags = [tag for tag in tags if tag != teacher_tag]
    if teacher_tag not in tags or not rank_tags:
        raise SystemExit(f"capture has no teacher/rank MLP fields: {tags}")

    result: dict = {
        "kind": "tp2_layer0_mlp_reference_check",
        "capture": str(args.capture),
        "model": str(args.model),
        "tags": tags,
        "ranks": len(rank_tags),
    }

    post_norm = _field(capture, teacher_tag, "post_norm")
    residual = _field(capture, teacher_tag, "residual")
    if post_norm is None or residual is None:
        raise SystemExit("capture is missing the teacher post_norm/residual")

    weights = _dequantized_weights(args.model)
    hidden = int(weights["down"].shape[0])
    ffn = int(weights["down"].shape[1])
    shard_ffn = ffn // len(rank_tags)
    width_of = {"hidden": hidden, "ffn": ffn, "shard": shard_ffn}
    rows = int(post_norm.size // hidden)

    def field(tag: str, name: str) -> np.ndarray | None:
        values = _field(capture, tag, name)
        if values is None:
            return None
        width = width_of[WIDTH_KIND[name]]
        if values.size != rows * width:
            raise SystemExit(
                f"{tag}.{name} has {values.size} values, expected {rows * width}"
            )
        return values.reshape(rows, width)

    post_norm = post_norm.reshape(rows, hidden)
    residual = residual.reshape(rows, hidden)
    result["rows"] = rows
    result["hidden"] = hidden
    result["ffn"] = ffn
    result["per_rank_ffn"] = shard_ffn

    # 1. The MLP inputs have to be identical, otherwise nothing below is a
    #    controlled comparison of the MLP schedules.
    result["input_identity"] = {
        tag: {
            "post_norm": _stat(field(tag, "post_norm"), post_norm),
            "residual": _stat(field(tag, "residual"), residual),
        }
        for tag in rank_tags
    }
    result["weight_shapes"] = {name: list(w.shape) for name, w in weights.items()}
    result["weight_absmax"] = {name: float(np.abs(w).max()) for name, w in weights.items()}

    full = reference_full_width_mlp(
        post_norm, weights["gate"], weights["up"], weights["down"]
    )
    sharded = reference_sharded_mlp(
        post_norm,
        weights["gate"],
        weights["up"],
        weights["down"],
        ranks=len(rank_tags),
        partial_dtype="bf16",
    )
    f32_sharded = reference_sharded_mlp(
        post_norm,
        weights["gate"],
        weights["up"],
        weights["down"],
        ranks=len(rank_tags),
        partial_dtype="f32",
    )

    # 2. Activation boundary: the resident fused pair+SiLU vs the shard chain.
    activation: dict = {}
    teacher_intermediate = field(teacher_tag, "ffn_intermediate")
    if teacher_intermediate is not None:
        activation["teacher_vs_reference"] = _stat(
            teacher_intermediate, full["intermediate"]
        )
    rebuilt = []
    for index, tag in enumerate(rank_tags):
        act = field(tag, "act")
        if act is None:
            continue
        rebuilt.append(act)
        activation[f"{tag}_vs_reference_slice"] = _stat(
            act, sharded["intermediates"][index]
        )
    if teacher_intermediate is not None and len(rebuilt) == len(rank_tags):
        activation["rebuilt_vs_teacher"] = _stat(
            np.concatenate(rebuilt, axis=1), teacher_intermediate
        )
    result["activation"] = activation

    # 3. Down boundary, computed from the *captured* activation so the numpy
    #    activation reference cannot leak into the down-projection comparison.
    #    ``intermediate`` below is the measured (bit-identical) activation.
    intermediate = teacher_intermediate
    if intermediate is None and len(rebuilt) == len(rank_tags):
        intermediate = np.concatenate(rebuilt, axis=1)
    if intermediate is None:
        raise SystemExit("capture has no activation to key the down reference on")
    intermediate = np.ascontiguousarray(intermediate, dtype=np.float32)
    down_f = np.ascontiguousarray(weights["down"], dtype=np.float32)
    full_exact_f32 = intermediate @ down_f.T
    exact_partials = reference_exact_partials(intermediate, down_f, ranks=len(rank_tags))
    reference_full_bf16 = bf16_round(full_exact_f32)

    down: dict = {}
    down["reassociation_only_f32_partials"] = _stat(
        _sum_f32(exact_partials), full_exact_f32
    )
    teacher_down = field(teacher_tag, "ffn_down")
    if teacher_down is not None:
        down["teacher_ffn_down_vs_reference_bf16"] = _stat(
            teacher_down, reference_full_bf16
        )
        down["teacher_ffn_down_rounding"] = _bf16_rounding_report(
            teacher_down, full_exact_f32
        )
    for index, tag in enumerate(rank_tags):
        partial = field(tag, "down_partial")
        if partial is not None:
            down[f"{tag}_partial_vs_exact_slice"] = _stat(
                partial, exact_partials[index]
            )
            down[f"{tag}_partial_rounding"] = _bf16_rounding_report(
                partial, exact_partials[index]
            )
        cast = field(tag, "cast")
        if cast is not None:
            down[f"{tag}_cast_vs_reference_bf16"] = _stat(
                cast, reference_full_bf16
            )
            down[f"{tag}_cast_rounding"] = _bf16_rounding_report(
                cast, full_exact_f32
            )
        reduced = field(tag, "reduced")
        if reduced is not None:
            down[f"{tag}_reduced_vs_reference_f32_sum"] = _stat(
                reduced, full_exact_f32
            )
    result["down_boundary"] = down

    # 4. Isolate the bf16 partial boundary itself, inside the reference: the
    #    only difference between these two rows is rounding each rank's exact
    #    partial to bf16 before the f32 sum.
    rounded_sum = _sum_f32([bf16_round(p) for p in exact_partials])
    partial_scale = max(float(np.abs(p).max()) for p in exact_partials)
    result["partial_boundary"] = {
        "partial_dtype": "bf16",
        "partial_absmax": partial_scale,
        "partial_ulp": float(bf16_ulp(np.float32(partial_scale))),
        "f32_partials_vs_full_f32": _stat(_sum_f32(exact_partials), full_exact_f32),
        "bf16_partials_vs_full_f32": _stat(rounded_sum, full_exact_f32),
        "bf16_partials_vs_f32_partials": _stat(
            rounded_sum, _sum_f32(exact_partials)
        ),
        "bf16_partials_vs_f32_partials_rounding": _bf16_rounding_report(
            rounded_sum, _sum_f32(exact_partials)
        ),
    }

    # 5. Layer output: does the extra down-boundary rounding survive the
    #    residual add at the layer-output scale?
    layer_out: dict = {}
    teacher_out = field(teacher_tag, "out")
    if teacher_out is not None and teacher_down is not None:
        layer_out["teacher_out_is_bf16_residual_plus_own_down"] = _stat(
            teacher_out, bf16_round((residual + teacher_down).astype(np.float32))
        )
    layer_out["reference_schedule_difference_at_output"] = _stat(
        bf16_round((residual + rounded_sum).astype(np.float32)),
        bf16_round((residual + reference_full_bf16).astype(np.float32)),
    )
    for tag in rank_tags:
        rank_out = field(tag, "out")
        if teacher_out is not None and rank_out is not None:
            layer_out[f"{tag}_vs_teacher"] = _stat(rank_out, teacher_out)
        cast = field(tag, "cast")
        if rank_out is not None and cast is not None:
            host_add = bf16_round((residual + cast).astype(np.float32))
            layer_out[f"{tag}_vs_host_bf16_residual_add"] = _stat(
                rank_out, host_add
            )
    if teacher_out is not None:
        host_teacher_add = bf16_round((residual + full["down"]).astype(np.float32))
        layer_out["teacher_vs_host_bf16_reference_add"] = _stat(
            teacher_out, host_teacher_add
        )
        host_sharded_add = bf16_round(
            (residual + bf16_round(rounded_sum)).astype(np.float32)
        )
        layer_out["host_sharded_vs_host_reference_add"] = _stat(
            host_sharded_add, host_teacher_add
        )
        layer_out["residual_absmax"] = float(np.abs(residual).max())
    result["layer_output"] = layer_out

    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
