#!/usr/bin/env python3
"""Compare dense split on natural activations without changing parent outputs.

Diagnostic only. Each eligible raw-F32 parent projection runs normally; a
private candidate buffer observes the same input and resident weight. F64 CPU
matmul uses evenly spaced rows and every output column. Projection softmax and
BF16 boundary crossings are localization aids, not model qualification gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hipengine.loading.materialize import float_array_to_bf16_bits


PARENTS = (
    "coltile8_rowbatch4_f32_f32_out",
    "coltile8_rowbatch4_wave_scale_f32_f32_out",
    "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out",
    "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out",
)


def projection_metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 2 or reference.shape != candidate.shape or not reference.size:
        raise ValueError("projection shape must match nonempty rank-two arrays")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("projection inputs must be finite")
    delta = candidate - reference
    norm = float(np.linalg.norm(reference))
    err = float(np.linalg.norm(delta))
    a = reference - reference.max(axis=1, keepdims=True)
    b = candidate - candidate.max(axis=1, keepdims=True)
    loga = a - np.log(np.exp(a).sum(axis=1, keepdims=True))
    logb = b - np.log(np.exp(b).sum(axis=1, keepdims=True))
    kl = np.maximum(np.sum(np.exp(loga) * (loga - logb), axis=1), 0.0)
    return {
        "finite": True, "elements": int(reference.size),
        "relative_l2": err / norm if norm else (0.0 if err == 0 else None),
        "max_abs": float(np.abs(delta).max()),
        "rms_abs": float(np.sqrt(np.mean(delta * delta))),
        "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
        "top1": float(np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1))),
        "bf16_flips": int(np.count_nonzero(
            float_array_to_bf16_bits(reference) != float_array_to_bf16_bits(candidate))),
    }


def sample_rows(rows, count):
    if rows < 1 or count < 1:
        raise ValueError("positive rows and sample count required")
    return np.linspace(0, rows - 1, min(rows, count), dtype=int).tolist()


def eligible_projection(fn, weight, *, rows, in_features, out_features, scope, parents):
    from hipengine.runtime.gguf_linear import _weight_layer_index
    return (
        fn in parents and getattr(weight.spec, "quant_key", None) == "gguf_q8_0"
        and _weight_layer_index(weight) in scope and rows > 256
        and in_features > 0 and in_features % 64 == 0 and out_features > 0
    )


def observe_parent_launch(parent, measure, *args, **kwargs):
    result = parent(*args, **kwargs)
    measure(*args, **kwargs)
    return result


@contextmanager
def projection_observer(report, *, case_id, scope, sample_count, capture_dir):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
    from hipengine.kernels.registry import resolve
    from hipengine.quant.gguf import dequantize_gguf_data, GGMLQuantizationType
    from hipengine.runtime import gguf_linear as linear

    runtime = get_hip_runtime()
    library = dw.build_gguf_q8_0_dense_wide(load=True)
    parents = {resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0", variant=v): v
               for v in PARENTS}
    candidate_fn = resolve(backend="hip_gfx1151", layer="linear", quant="gguf_q8_0",
                           variant="dense_split64x128_f32_f32_out")
    original = linear._LAUNCH_ABI["raw"]
    seen = set()

    def read(ptr, shape, dtype):
        array = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(array), DeviceBuffer(int(ptr), array.nbytes), runtime=runtime)
        return array

    def measure(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs):
        slot = str(weight.spec.slot_path)
        if slot in seen or not eligible_projection(
            fn, weight, rows=rows, in_features=in_features,
            out_features=out_features, scope=scope, parents=parents,
        ):
            return
        seen.add(slot)
        runtime.device_synchronize()
        x = read(x_ptr, (rows, in_features), np.float32)
        parent = read(out_ptr, (rows, out_features), np.float32)
        raw = read(weight.allocation("raw").tensor.ptr,
                   (out_features, in_features // 32 * 34), np.uint8)
        buffer = malloc(parent.nbytes, runtime=runtime)
        try:
            for repeat in range(2):
                candidate_fn(
                    x_ptr, weight.allocation("raw").tensor.ptr, buffer.ptr,
                    rows, in_features, out_features,
                    stream=kwargs.get("stream", 0), library=library, runtime=runtime,
                )
                runtime.device_synchronize()
                candidate = read(buffer.ptr, parent.shape, np.float32)
                digest = hashlib.sha256(candidate).hexdigest()
                if repeat == 0:
                    first_digest = digest
                elif digest != first_digest:
                    raise RuntimeError("candidate projection is not repeat deterministic")
            indices = sample_rows(rows, sample_count)
            weights = dequantize_gguf_data(raw, GGMLQuantizationType.Q8_0).reshape(
                out_features, in_features).astype(np.float64)
            reference = x[indices].astype(np.float64) @ weights.T
            record = {
                "case_id": case_id, "slot": slot, "shape": [rows, in_features, out_features],
                "parent": parents[fn], "input_sha256": hashlib.sha256(x).hexdigest(),
                "weight_sha256": hashlib.sha256(raw).hexdigest(),
                "input_abs_max": float(np.abs(x).max()),
                "input_abs_p99": float(np.percentile(np.abs(x), 99)),
                "weight_abs_max": float(np.abs(weights).max()),
                "sample_rows": indices, "candidate_deterministic": True,
                "parent_cpu_f64": projection_metrics(reference, parent[indices]),
                "candidate_cpu_f64": projection_metrics(reference, candidate[indices]),
                "candidate_parent": projection_metrics(parent, candidate),
            }
            if capture_dir is not None:
                capture_dir.mkdir(parents=True, exist_ok=True)
                path = capture_dir / (case_id + "-" + slot + ".npz")
                np.savez_compressed(path, x=x, raw_q8=raw, sample_rows=indices,
                                    parent_sample=parent[indices],
                                    candidate_sample=candidate[indices], reference=reference)
                record["local_capture"] = str(path)
            report["projections"].append(record)
            print(json.dumps(record), flush=True)
        finally:
            free(buffer, runtime=runtime)

    def hooked(*args, **kwargs):
        return observe_parent_launch(original, measure, *args, **kwargs)

    linear._LAUNCH_ABI["raw"] = hooked
    try:
        yield
        if not seen:
            raise RuntimeError("no eligible raw F32 parent projection was observed")
    finally:
        linear._LAUNCH_ABI["raw"] = original


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--case-id", action="append", required=True)
    p.add_argument("--layers", default="0-12")
    p.add_argument("--sample-rows", type=int, default=8)
    p.add_argument("--capture-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    command = [sys.argv[0], *(list(argv) if argv is not None else sys.argv[1:])]
    if args.sample_rows < 1:
        p.error("sample-rows must be positive")
    from scripts.execution_profile_q8_wmma_prefill_layers_gate import _parse_layers
    from scripts.qwen4exp_canonical_ar_bench import load_fixture, DEFAULT_FIXTURE, _git_metadata, _host_metadata
    from scripts.qwen4exp_layer2_profile_gate import _make_generator
    from hipengine.core.memory import memory_stats
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    import fcntl

    scope = {int(v) for v in _parse_layers(args.layers).split(",")}
    fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    cases = [c for c in fixture["cases"] if c["id"] in args.case_id]
    if {c["id"] for c in cases} != set(args.case_id):
        p.error("unknown case-id")
    report = {"kind": "dense_split_parent_trajectory_localization", "schema": 1,
              "status": "running", "performance_claim": False, "profile_qualification_claim": False,
              "source": _git_metadata(ROOT), "host": _host_metadata(), "model_root": str(args.model_root),
              "command": command, "fixture_sha256": fixture_hash,
              "protocol": {"profile": "production", "layers": sorted(scope), "chunk": 1024,
                           "observation": "first eligible projection per slot per case; split output never published",
                           "cpu_oracle": "F64 matmul, evenly spaced rows and every output column"},
              "projections": [], "parent_trajectory_checks": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        register_gfx1151_kernels(replace=True)
        register_qwen4_exp_gfx1151_profiles()
        gen_args = SimpleNamespace(model_root=args.model_root, max_sequence_length=4352, prefill_chunk_size=1024)
        generator, profile, _ = _make_generator(gen_args, "production")
        report["manifest_sha256"] = profile.manifest_sha256
        if os.environ.get("HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS"):
            generator.close()
            raise RuntimeError("parent trajectory requires split selector off")
        try:
            for case in cases:
                baseline = generator.runner.prefill(case["prompt_token_ids"]).logits.copy()
                with projection_observer(report, case_id=case["id"], scope=scope,
                                         sample_count=args.sample_rows, capture_dir=args.capture_dir):
                    observed = generator.runner.prefill(case["prompt_token_ids"]).logits
                unchanged = np.array_equal(baseline, observed)
                report["parent_trajectory_checks"].append({"case_id": case["id"], "logits_bit_identical": unchanged})
                if not unchanged:
                    raise RuntimeError("diagnostic observer changed parent trajectory")
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            report["status"] = "completed"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            generator.close()
            report["memory_after_close"] = memory_stats()
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
