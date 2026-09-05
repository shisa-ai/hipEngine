#!/usr/bin/env python3
"""Actual-weight exact Q4 gate/up + SiLU publication A/B."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from tests.test_qwen4exp_q4_bundle import PARENT, CANDIDATE, PAIR
from tests.test_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download, _make_activation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--rows",type=int,nargs="+",default=[64,512])
    p.add_argument("--pairs",type=int,default=10)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--require-cached-build",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pair-reuse", action="store_true")
    p.add_argument("--routing", choices=("uniform", "skewed"), default="uniform")
    p.add_argument("--layer", type=int, default=0)
    a=p.parse_args()
    parent_name, candidate_name = (CANDIDATE, PAIR) if a.pair_reuse else (PARENT, CANDIDATE)
    if a.pairs<1 or any(r<1 for r in a.rows):
        p.error("positive rows and pairs required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"]=str(a.compiler_version_file)
    if a.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"]="1"
    runtime=get_hip_runtime()
    library=q4.build_gguf_q4_k_selected_prefill(load=True)
    readers=[GGUFReader(path) for path in discover_gguf_files(a.model_root)]
    weights=[]
    identities=[]
    for name in (f"blk.{a.layer}.ffn_gate_exps.weight",f"blk.{a.layer}.ffn_up_exps.weight"):
        reader=next(r for r in readers if any(t.name==name for t in r.info.tensors))
        info=reader.tensor_info(name)
        assert info.shape==(512,640,2560) and info.ggml_type_name=="Q4_K"
        raw=reader.tensor_data(name)
        identities.append({"tensor":name,"path":str(reader.path),"shape":info.shape,
                           "sha256":hashlib.sha256(raw).hexdigest()})
        weights.append(raw)
    report={"schema":1,"kind":"qwen4exp_q4_bundle_gate_up_silu",
            "source":_git_metadata(ROOT),"host":_host_metadata(),"command":sys.argv,
            "model":"Qwen3.8-Flash-Next UD-Q4_K_XL","weights":identities,
            "arithmetic_class":"T0","runtime_default_changed":False,
            "boundary":"grouped Q4_K gate/up plus unchanged BF16 SiLU, no routing/down/combine",
            "parent_variant": parent_name, "candidate_variant": candidate_name,
            "routing": a.routing, "layer": a.layer,
            "cases":[]}
    allocations=[]
    try:
        wa,wb=[_upload(w,runtime,allocations) for w in weights]
        for rows in a.rows:
            mark=len(allocations)
            rng=np.random.default_rng(1788+rows)
            scores = rng.random((rows,512))
            if a.routing == "skewed":
                priorities = np.exp(rng.normal(0, 1.5, 512))
                scores = -np.log(np.maximum(scores, np.finfo(np.float64).tiny)) / priorities
            selected=np.argsort(scores,axis=1)[:,:10]
            counts=np.bincount(selected.reshape(-1),minlength=512)
            starts=np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
            compact=rows*10
            x,_=_make_activation(compact,2560,3456+rows)
            dx,ds=[_upload(v,runtime,allocations) for v in (x,starts)]
            outputs=[_alloc((compact,640),np.uint16,runtime,allocations) for _ in range(6)]
            def run(candidate):
                offset=3 if candidate else 0
                getattr(q4,candidate_name if candidate else parent_name)(
                    dx.ptr,ds.ptr,wa.ptr,wb.ptr,outputs[offset].ptr,outputs[offset+1].ptr,
                    compact,512,2560,640,library=library,runtime=runtime)
                silu_mul_separate_out_bf16(
                    outputs[offset].ptr,outputs[offset+1].ptr,outputs[offset+2].ptr,
                    compact,640,runtime=runtime)
                runtime.device_synchronize()
            run(False)
            run(True)
            times={"parent":[],"candidate":[]}
            for pair in range(a.pairs):
                for candidate in ((False,True) if pair%2==0 else (True,False)):
                    start=time.perf_counter()
                    run(candidate)
                    times["candidate" if candidate else "parent"].append(time.perf_counter()-start)
                for i in range(3):
                    np.testing.assert_array_equal(
                        _download(outputs[i],(compact,640),np.uint16,runtime),
                        _download(outputs[i+3],(compact,640),np.uint16,runtime))
            report["cases"].append({"tokens":rows,"compact_rows":compact,
                "active_experts":int(np.count_nonzero(counts)),"seconds":times,
                "max_expert_rows":int(counts.max()),
                "median_active_expert_rows":float(np.median(counts[counts>0])),
                "weight_passes_by_row_batch":{
                    str(rb):int(np.sum((counts+rb-1)//rb)) for rb in (8,16,32)},
                "speedup":statistics.median(times["parent"])/statistics.median(times["candidate"]),
                "all_pairs_exact":True})
            for ptr in reversed(allocations[mark:]):
                free(ptr,runtime=runtime)
            del allocations[mark:]
    finally:
        for ptr in reversed(allocations):
            free(ptr,runtime=runtime)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
