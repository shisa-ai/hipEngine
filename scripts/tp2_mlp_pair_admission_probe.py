#!/usr/bin/env python3
"""Admit the fused gate/up+SiLU pair route at a candidate shard width.

``GGUF_DENSE_PAIR_SILU_DECODE_POLICIES`` is an exact-shape allowlist, so an
uneven TP2 split moves the ranks off the admitted ``(1, hidden, ffn/world)``
shape and both ranks silently fall back to the unfused three-launch chain. The
kernel's own contract is wider than the allowlist
(``dense_t16_pair_decode_shape_error`` only requires ``rows == 1``,
``in_features % 256 == 0`` and ``out_features % 16 == 0``), so the question is
whether the fused candidate is still *arithmetically* the unfused chain at the
new width.

This probe answers exactly that, on real layer weights, per rank:

* build each rank's resident-layout payload from the shard manifest, for the
  even split or for an explicit per-rank boundary;
* run the unfused chain and the fused candidate on the same input bytes;
* compare ``gate``, ``up``, ``activated`` and the down partial byte for byte,
  and report the full-tensor and per-element differences.

A width is admitted to the policy table only when every compared field is
bit-identical on every rank. Timings are recorded for context, never as the
admission criterion.

Usage::

    python scripts/tp2_mlp_pair_admission_probe.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --fractions 0.417145/0.582855 \
        --json benchmarks/results/2026-09-18-w7900-tp2-pair-admission-uneven.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tp2_mlp_shard_plan_probe import resolve_incumbent_plan  # noqa: E402
from tp2_mlp_slice_e2e import (  # noqa: E402
    _ShardAllocation,
    _ShardWeight,
    rank_shard_payload,
    run_rank_chain,
    run_rank_chain_fused,
)

from hipengine.core.device import scoped_current_device  # noqa: E402
from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.loading.gguf import GGUFReader, scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import qwen35_gguf_config_from_metadata  # noqa: E402
from hipengine.loading.qwen35_gguf_admission import (  # noqa: E402
    build_qwen35_gguf_role_manifest,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    build_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_shards import (  # noqa: E402
    UnevenSplitPolicy,
    build_shard_manifest,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (  # noqa: E402
    dense_t16_pair_decode_shape_error,
)

MLP_ROLES = ("ffn_gate", "ffn_up", "ffn_down")
FUSED_VARIANT = "dense_dual_local32_bf16_bf16_out"
HIDDEN = 5120


def _parse_fractions(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    parts = [part.strip() for part in str(text).split("/")]
    if len(parts) < 2 or any(not part for part in parts):
        raise SystemExit(f"--fractions must be a '/' separated share per rank, got {text!r}")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as error:
        raise SystemExit(f"--fractions is not numeric: {text!r}") from error


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Byte-level comparison of two same-shape arrays.

    Bit-identity is decided on the raw bytes, so it does not depend on the
    dtype the download helper happened to return.
    """

    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.shape != b.shape:
        return {"shape_match": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    a_bytes = a.tobytes()
    b_bytes = b.tobytes()
    result: dict[str, Any] = {
        "shape_match": True,
        "bit_identical": a_bytes == b_bytes,
        "bytes": len(a_bytes),
    }
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub" and a.dtype == b.dtype:
        fa = a.astype(np.float64)
        fb = b.astype(np.float64)
        diff = np.abs(fa - fb)
        result.update(
            {
                "differing_elements": int(np.count_nonzero(a != b)),
                "elements": int(a.size),
                "max_abs_diff": float(diff.max()) if a.size else 0.0,
                "mean_abs_diff": float(diff.mean()) if a.size else 0.0,
            }
        )
    return result


def admission_verdict(rows: list[dict[str, Any]]) -> bool:
    """Whether every rank's fused candidate may be admitted to the policy table.

    A width is admitted only when the kernel's own shape contract accepts it
    *and* every compared field is bit-identical to the unfused chain. A rank
    whose fused launch raised, or whose comparison fields were missing, is not
    admitted: absence of a difference is not evidence of equality.
    """

    if not rows:
        return False
    for row in rows:
        if row.get("kernel_contract_error") is not None:
            return False
        if row.get("fused_error") is not None:
            return False
        fields = row.get("fields") or {}
        if not fields:
            return False
        for entry in fields.values():
            if not entry.get("present_in_both"):
                return False
            if not entry.get("shape_match"):
                return False
            if not entry.get("bit_identical"):
                return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=None,
        help="per-rank shard shares; unset means the even split",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    info = scan_gguf(str(args.model))
    config = qwen35_gguf_config_from_metadata(info)
    ffn = int(config.feed_forward_length)
    world_size = int(args.world_size)
    policy = (
        None if args.fractions is None else UnevenSplitPolicy(fractions=args.fractions)
    )

    materialization, plan_context = resolve_incumbent_plan(info)
    model_map = build_qwen35_gguf_tensor_map(info)
    fingerprint = build_qwen35_gguf_role_manifest(model_map).fingerprint
    manifest = build_shard_manifest(
        info,
        world_size=world_size,
        model_hash=fingerprint,
        uneven_split=policy,
    )

    reader = GGUFReader(str(args.model))
    runtime = get_hip_runtime()
    rng = np.random.default_rng(int(args.seed))
    x_f32 = rng.standard_normal(HIDDEN, dtype=np.float32) * np.float32(0.05)

    # The activation contract is bf16, so stage the same bf16 input both chains
    # see: round the f32 vector once, then use those bytes.
    from hipengine.kernels.cpu_reference.maple import f32_to_bf16_bits  # noqa: PLC0415

    x_bf16_bits = np.array([f32_to_bf16_bits(float(v)) for v in x_f32], dtype=np.uint16)
    x_bf16_bytes = np.frombuffer(x_bf16_bits.tobytes(), dtype=np.uint8)

    streams: list[int] = []
    device_names: list[str] = []
    for device in range(world_size):
        with scoped_current_device(runtime, device):
            streams.append(int(runtime.stream_create()))
            device_names.append(str(runtime.device_get_name(device)))

    rows: list[dict[str, Any]] = []
    allocations: list[Any] = []
    try:
        for rank in range(world_size):
            widths = {
                role: manifest.plan_for(f"blk.{int(args.layer)}.{role}.weight")
                .slice_for(rank)
                .local_shape
                for role in MLP_ROLES
            }
            per_rank_ffn = int(widths["ffn_gate"][0])
            contract = dense_t16_pair_decode_shape_error(
                rows=1, in_features=HIDDEN, out_features=per_rank_ffn
            )
            weights: dict[str, _ShardWeight] = {}
            for role in MLP_ROLES:
                name = f"blk.{int(args.layer)}.{role}.weight"
                plan = manifest.plan_for(name)
                payload, layout, quant_key = rank_shard_payload(
                    reader,
                    materialization,
                    plan,
                    plan.slice_for(rank),
                    layer=int(args.layer),
                )
                allocation = _ShardAllocation("tiles", runtime, rank, payload)
                allocations.append(allocation)
                weights[role] = _ShardWeight(layout, quant_key, allocation)

            started = time.perf_counter()
            unfused = run_rank_chain(
                runtime,
                device=rank,
                stream=streams[rank],
                weights=weights,
                x_bf16_bytes=x_bf16_bytes,
                hidden=HIDDEN,
                per_rank_ffn=per_rank_ffn,
            )
            unfused_ms = (time.perf_counter() - started) * 1e3

            started = time.perf_counter()
            try:
                fused = run_rank_chain_fused(
                    runtime,
                    device=rank,
                    stream=streams[rank],
                    weights=weights,
                    x_bf16_bytes=x_bf16_bytes,
                    hidden=HIDDEN,
                    per_rank_ffn=per_rank_ffn,
                    decode_variant=FUSED_VARIANT,
                )
                fused_error = None
            except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                fused = None
                fused_error = f"{type(error).__name__}: {error}"
            fused_ms = (time.perf_counter() - started) * 1e3

            fields: dict[str, Any] = {}
            if fused is not None:
                # The fused route produces the activation and the down partial;
                # gate/up exist only in the unfused chain, so comparing them
                # would compare a produced field against nothing.
                for field in ("activated", "down_partial"):
                    if field not in unfused or field not in fused:
                        fields[field] = {"present_in_both": False}
                        continue
                    entry = _compare(unfused[field], fused[field])
                    entry["present_in_both"] = True
                    fields[field] = entry
            rows.append(
                {
                    "rank": rank,
                    "device": device_names[rank],
                    "per_rank_ffn": per_rank_ffn,
                    "kernel_contract_error": contract,
                    "fused_error": fused_error,
                    "fields": fields,
                    "all_fields_bit_identical": bool(fields)
                    and all(
                        entry.get("present_in_both") and entry.get("bit_identical")
                        for entry in fields.values()
                    ),
                    "unfused_ms": round(unfused_ms, 4),
                    "fused_ms": round(fused_ms, 4),
                }
            )
    finally:
        for allocation in allocations:
            try:
                allocation.free(runtime=runtime)
            except Exception:  # noqa: BLE001 - teardown best effort
                pass

    admitted = admission_verdict(rows)
    artifact = {
        "schema": 1,
        "kind": "tp2-mlp-pair-admission",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "model": str(args.model),
        "layer": int(args.layer),
        "world_size": world_size,
        "hidden": HIDDEN,
        "ffn": ffn,
        "fused_variant": FUSED_VARIANT,
        "uneven_split": None if policy is None else policy.to_dict(),
        "plan_context": plan_context,
        "rows": rows,
        "admitted": bool(admitted),
        "note": (
            "Admission requires the kernel shape contract to admit the width and "
            "every compared field to be bit-identical between the fused candidate "
            "and the unfused chain on every rank. Timings are context only."
        ),
    }

    print(f"layer {args.layer}, world {world_size}, ffn {ffn}")
    for row in rows:
        widths = row["per_rank_ffn"]
        print(
            f"  rank {row['rank']} ({row['device']}): width {widths}, "
            f"contract {row['kernel_contract_error']}, "
            f"fused {'FAILED: ' + str(row['fused_error']) if row['fused_error'] else 'ok'}, "
            f"bit-identical {row['all_fields_bit_identical']}"
        )
        for field, entry in row["fields"].items():
            print(
                f"      {field}: identical={entry['bit_identical']} "
                f"differing={entry.get('differing_elements')}/{entry.get('elements')} "
                f"max_abs={entry.get('max_abs_diff')}"
            )
    print(f"admitted: {admitted}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(artifact, indent=2) + "\n")
        print("wrote", args.json)
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
