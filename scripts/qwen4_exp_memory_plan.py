#!/usr/bin/env python3
"""Report real Qwen4Exp residency and complete c1 memory admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hipengine.core.hip import get_hip_runtime
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import build_qwen4_exp_gguf_tensor_map
from hipengine.loading.qwen4_exp_materialize import (
    plan_qwen4_exp_memory_admission,
    plan_qwen4_exp_residency,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--context", type=int, action="append")
    parser.add_argument("--resident-capacity", type=int, default=1)
    parser.add_argument("--scratch-gib", type=float, default=4.0)
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    parser.add_argument("--available-bytes", type=int)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    parts = discover_gguf_files(args.model)
    readers = tuple(GGUFReader(path) for path in parts)
    model_map = build_qwen4_exp_gguf_tensor_map(tuple(reader.info for reader in readers))
    residency = plan_qwen4_exp_residency(model_map)
    if args.available_bytes is None:
        free_bytes, total_bytes = get_hip_runtime().mem_get_info()
        available = free_bytes
    else:
        available = int(args.available_bytes)
        free_bytes = available
        total_bytes = available
    contexts = args.context or [2051, 4096, 16384, 65536, model_map.config.context_length]
    scratch = int(args.scratch_gib * 1024**3)
    reserve = int(args.reserve_gib * 1024**3)
    admissions = []
    for context in contexts:
        plan = plan_qwen4_exp_memory_admission(
            residency,
            available_device_bytes=available,
            context_tokens=context,
            resident_capacity=args.resident_capacity,
            scratch_bytes=scratch,
            reserve_bytes=reserve,
        )
        admissions.append(
            {
                "context_tokens": context,
                "passed": plan.passed,
                "required_bytes": plan.required_bytes,
                "shortfall_bytes": plan.shortfall_bytes,
                "device_weight_bytes": plan.device_weight_bytes,
                "staging_bytes": plan.staging_bytes,
                "kv_bytes": plan.kv_bytes,
                "index_bytes": plan.index_bytes,
                "runtime_state_bytes": plan.runtime_state_bytes,
                "scratch_bytes": plan.scratch_bytes,
                "reserve_bytes": plan.reserve_bytes,
            }
        )
    report = {
        "schema": 1,
        "model": str(args.model.expanduser().resolve()),
        "part_paths": [str(path) for path in parts],
        "physical_device_free_bytes": free_bytes,
        "physical_device_total_bytes": total_bytes,
        "available_device_bytes": available,
        "resident_capacity": args.resident_capacity,
        "residency": {
            "raw_payload_bytes": residency.raw_payload_bytes,
            "device_weight_bytes": residency.device_weight_bytes,
            "replacement_payload_bytes": residency.replacement_payload_bytes,
            "alternate_layout_bytes": residency.alternate_layout_bytes,
            "ple_mmap_bytes": residency.ple_mmap_bytes,
            "staging_bytes": residency.staging_bytes,
            "tensor_bytes_by_type": dict(residency.tensor_bytes_by_type),
        },
        "admissions": admissions,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        output = args.json_out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
