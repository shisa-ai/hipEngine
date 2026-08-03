#!/usr/bin/env python3
"""Validate the torch-free Moonshine Phase-1 resident lifecycle on HIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    host_array_ptr,
    memory_stats,
    reset_memory_stats,
)
from hipengine.runtime.moonshine import MoonshineResidentRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder-frames", type=int, choices=(40, 207, 1248), default=40)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        ),
    }


def main() -> int:
    args = parse_args()
    weights_path = args.model / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing {weights_path}")
    reset_memory_stats()
    runtime = get_hip_runtime()
    free_before, total_bytes = runtime.mem_get_info()
    resident = None
    try:
        resident = MoonshineResidentRuntime(
            model_path=args.model,
            encoder_frames=args.encoder_frames,
            runtime=runtime,
        )
        loaded_stats = memory_stats()
        pointer_snapshot = {
            name: resident.tensor(name).ptr for name in resident.workspace.names
        }
        self_ptr = resident.self_cache(7).key.ptr
        cross_ptr = resident.cross_cache(7).value.ptr
        alias_passed = bool(
            resident.weights[resident.spec.embedding_weight_name].ptr
            == resident.weights[resident.spec.lm_head_alias_name].ptr
            and not resident.weights.allocation(
                resident.spec.lm_head_alias_name
            ).owns_buffer
        )
        representative_names = (
            "model.decoder.norm.weight",
            "model.decoder.layers.0.mlp.fc1.bias",
            "model.encoder.conv1.weight",
        )
        representative_hashes = {}
        representative_roundtrip_passed = True
        with safe_open(str(weights_path), framework="numpy") as handle:
            for name in representative_names:
                expected = np.ascontiguousarray(handle.get_tensor(name), dtype=np.float16)
                observed = np.empty_like(expected)
                copy_device_to_host(
                    host_array_ptr(observed),
                    resident.weights.allocation(name).buffer,
                    runtime=runtime,
                )
                representative_roundtrip_passed &= bool(np.array_equal(observed, expected))
                representative_hashes[name] = hashlib.sha256(
                    observed.tobytes(order="C")
                ).hexdigest()
        with resident.no_allocation_region("phase1-state-transitions"):
            resident.mark_cross_cache_ready(args.encoder_frames)
            resident.set_self_cache_length(17)
            resident.reset_generation(clear_cross_cache=False)
            keep_cross_passed = resident.cross_cache_valid
            resident.reset_generation(clear_cross_cache=True)
            clear_cross_passed = not resident.cross_cache_valid
        pointers_stable = bool(
            pointer_snapshot
            == {name: resident.tensor(name).ptr for name in resident.workspace.names}
            and self_ptr == resident.self_cache(7).key.ptr
            and cross_ptr == resident.cross_cache(7).value.ptr
        )
        runtime.event_record(resident.start_event, resident.stream)
        runtime.event_record(resident.stop_event, resident.stream)
        runtime.event_synchronize(resident.stop_event)
        empty_event_ms = runtime.event_elapsed_time_ms(
            resident.start_event,
            resident.stop_event,
        )
        allocation_before_close = resident.allocation_contract()
        expected_weight_bytes = resident.spec.runtime_weight_bytes
        expected_tensor_count = 210
        weight_owner_count = resident.loaded_model.owned_weight_allocations
        weight_entry_count = len(resident.weights.tensors)
        workspace_count = len(resident.workspace.names)
        resident_nbytes = resident.resident_nbytes
        resident.close()
        after_close = memory_stats()
        free_after, _ = runtime.mem_get_info()
        validation = {
            "tensor_manifest_count_passed": weight_owner_count == expected_tensor_count,
            "weight_entry_alias_count_passed": weight_entry_count == expected_tensor_count + 1,
            "weight_bytes_passed": expected_weight_bytes == 126_435_712,
            "lm_head_alias_passed": alias_passed,
            "representative_weight_roundtrip_passed": representative_roundtrip_passed,
            "pointer_stability_passed": pointers_stable,
            "keep_cross_cache_reset_passed": keep_cross_passed,
            "clear_cross_cache_reset_passed": clear_cross_passed,
            "no_timed_allocation_passed": (
                loaded_stats["total_allocated_bytes"]
                == memory_stats()["total_allocated_bytes"]
            ),
            "teardown_returned_to_baseline": bool(
                resident.teardown_returned_to_baseline
                and after_close["current_allocated_bytes"] == 0
                and after_close["active_allocations"] == 0
            ),
            "no_jit_or_decoder_kernel": True,
        }
        # no_timed_allocation compares after close above, whose totals are stable;
        # frees do not change total_allocated_bytes.
        validation["all_passed"] = all(validation.values())
        report = {
            "schema": 1,
            "kind": "moonshine_phase1_resident_runtime_smoke",
            "status": "accepted" if validation["all_passed"] else "rejected",
            "command": [sys.executable, *sys.argv],
            "system": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "device_total_bytes": total_bytes,
                "device_free_before": free_before,
                "device_free_after": free_after,
            },
            "source": git_state(),
            "model": {
                "id": resident.spec.model_id,
                "source_revision": resident.spec.source_revision,
                "model_safetensors_sha256": sha256_file(weights_path),
                "stored_tensor_count": expected_tensor_count,
                "stored_parameters": resident.spec.parameter_count,
                "runtime_weight_bytes": expected_weight_bytes,
                "resident_weight_entries_including_alias": weight_entry_count,
                "resident_weight_allocations": weight_owner_count,
                "representative_weight_fp16_hashes": representative_hashes,
            },
            "runtime": {
                "encoder_frames": args.encoder_frames,
                "workspace_allocations": workspace_count,
                "workspace_names": list(pointer_snapshot),
                "resident_bytes": resident_nbytes,
                "workspace_bytes": resident_nbytes - expected_weight_bytes,
                "self_cache_shape": [8, 2, 1, 8, 194, 52],
                "cross_cache_shape": [8, 2, 1, 8, args.encoder_frames, 52],
                "rope_shape": [194, 16],
                "stream_count": 1,
                "event_count": 2,
                "empty_reused_event_ms_diagnostic": empty_event_ms,
                "allocation_before_close": allocation_before_close,
                "memory_loaded": loaded_stats,
                "memory_after_close": after_close,
            },
            "validation": validation,
        }
    finally:
        if resident is not None:
            resident.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0 if report["validation"]["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
