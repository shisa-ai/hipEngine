#!/usr/bin/env python3
"""Order-balanced real-Q8 dense split WMMA screen against the live F32 parent.

This is a primitive diagnostic, not a full-model correctness/performance claim.
Real early-layer weight banks, seeded synthetic activations, full F32 output,
and synchronized prebound host-call wall. CPU-reference metrics use every
column of eight declared sample rows; GPU tests cover independent small inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kg
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as dw
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.quant.gguf import dequantize_gguf_data, GGMLQuantizationType
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata

TENSORS = ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
           "blk.0.ffn_down_shexp.weight", "blk.3.attn_q.weight"]
PARENT = "gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out"
CANDIDATE = "gguf_q8_0_dense_split64x128_f32_f32_out"


def distribution_metrics(reference, candidate):
    a = reference - reference.max(axis=1, keepdims=True)
    b = candidate - candidate.max(axis=1, keepdims=True)
    loga = a - np.log(np.exp(a).sum(axis=1, keepdims=True))
    logb = b - np.log(np.exp(b).sum(axis=1, keepdims=True))
    kl = np.sum(np.exp(loga) * (loga - logb), axis=1)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
            "top1": float(np.mean(reference.argmax(axis=1) == candidate.argmax(axis=1))),
            "relative_l2": float(np.linalg.norm(candidate-reference) / np.linalg.norm(reference)),
            "max_abs": float(np.max(np.abs(candidate-reference)))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--rows", nargs="+", type=int, default=[512, 1024, 4096])
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--tensor", action="append")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.pairs < 2 or args.pairs % 2 or any(r < 8 for r in args.rows):
        p.error("even pairs >=2 and rows >=8 required")
    runtime = get_hip_runtime()
    libs = [kg.build_gguf_k_gemv(load=True), dw.build_gguf_q8_0_dense_wide(load=True)]
    wrappers = [getattr(kg, PARENT), getattr(dw, CANDIDATE)]
    readers = [GGUFReader(f) for f in discover_gguf_files(args.model_root)]
    report = {"schema": 1, "kind": "qwen4exp_dense_split_primitive_screen", "status": "running",
              "performance_claim": False, "profile_qualification_claim": False,
              "model": "Qwen3.8-Flash-Next UD-Q4_K_XL", "model_root": str(args.model_root),
              "source": _git_metadata(ROOT), "host": _host_metadata(), "command": sys.argv,
              "parent": PARENT, "candidate": CANDIDATE, "pairs": args.pairs,
              "timing_boundary": "prebound raw-Q8 F32-input/output launch plus device synchronize",
              "input_kind": "real early-layer Q8_0 weights, seeded synthetic F32 activations",
              "correctness_boundary": "independent F64 CPU dequant/matmul on 8 evenly spaced rows; full-output parent comparison",
              "cases": []}
    for name in args.tensor or TENSORS:
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        assert info.ggml_type_name == "Q8_0"
        n, k = info.shape
        weight = np.ascontiguousarray(reader.tensor_data(name)).view(np.uint8).reshape(n, -1)
        w64 = dequantize_gguf_data(weight, GGMLQuantizationType.Q8_0).astype(np.float64)
        for rows in args.rows:
            rng = np.random.default_rng(5413 + rows)
            x = rng.normal(0, 0.2, (rows,k)).astype(np.float32)
            out = [np.zeros((rows,n), np.float32) for _ in range(2)]
            buffers = [malloc(x.nbytes), malloc(weight.nbytes), malloc(out[0].nbytes), malloc(out[1].nbytes)]
            try:
                copy_host_to_device(buffers[0],host_array_ptr(x),runtime=runtime)
                copy_host_to_device(buffers[1],host_array_ptr(weight),runtime=runtime)
                def run(arm):
                    wrappers[arm](buffers[0].ptr,buffers[1].ptr,buffers[arm+2].ptr,rows,k,n,
                                  threads=128 if arm == 0 else 256,library=libs[arm],runtime=runtime)
                    runtime.device_synchronize()
                for _ in range(3):
                    run(0); run(1)
                samples = [[],[]]
                first_hash = None
                for pair in range(args.pairs):
                    for arm in ((0,1) if pair%2 == 0 else (1,0)):
                        t=time.perf_counter_ns(); run(arm)
                        samples[arm].append((time.perf_counter_ns()-t)/1e6)
                    # Full-output repeat check once per order stratum.
                    if pair < 2 or pair == args.pairs-1:
                        for arm in range(2):
                            copy_device_to_host(host_array_ptr(out[arm]),buffers[arm+2],runtime=runtime)
                        digest=hashlib.sha256(out[1]).hexdigest()
                        if first_hash is None: first_hash=digest
                        assert digest == first_hash
                sample_rows = np.linspace(0,rows-1,8,dtype=int)
                reference = x[sample_rows].astype(np.float64) @ w64.T
                metrics = distribution_metrics(reference,out[1][sample_rows].astype(np.float64))
                parent_metrics = distribution_metrics(reference,out[0][sample_rows].astype(np.float64))
                assert metrics["kl_mean"] <= .05 and metrics["top1"] >= .90, metrics
                assert metrics["relative_l2"] <= 2e-6, metrics
                medians = [statistics.median(s) for s in samples]
                row = {"tensor":name,"shape":[rows,k,n],"weight_sha256":hashlib.sha256(weight).hexdigest(),
                       "parent_ms":medians[0],"candidate_ms":medians[1],"speedup":medians[0]/medians[1],
                       "speedup_by_order":{str(offset):statistics.median(samples[0][offset::2])/statistics.median(samples[1][offset::2]) for offset in (0,1)},
                       "samples_ms":samples,"candidate_cpu_f64":metrics,"parent_cpu_f64":parent_metrics,
                       "parent_candidate_relative_l2":float(np.linalg.norm(out[1]-out[0])/np.linalg.norm(out[0])),
                       "candidate_sha256":first_hash,"deterministic":True,"cpu_sample_rows":sample_rows.tolist()}
                report["cases"].append(row)
                print(json.dumps({key:row[key] for key in ("tensor","shape","parent_ms","candidate_ms","speedup","candidate_cpu_f64")}),flush=True)
            finally:
                for b in reversed(buffers): free(b,runtime=runtime)
    report["status"]="completed"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
