#!/usr/bin/env python3
"""Actual rotating expert-down Q5_1 banks, exact M1 versus output-pair reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q5
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_qwen4exp_q51_pair import PARENT,CANDIDATE
from tests.test_qwen4_exp_pf3_moe_schedules import _upload,_alloc,_download,_make_activation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--rows",nargs="+",type=int,default=[64,512])
    p.add_argument("--pairs",type=int,default=10)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--require-cached-build",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.pairs<1 or any(r<1 for r in a.rows):
        p.error("positive rows/pairs required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"]=str(a.compiler_version_file)
    if a.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"]="1"
    runtime=get_hip_runtime()
    library=q5.build_qwen4_exp_q5_1(load=True)
    readers=[GGUFReader(f) for f in discover_gguf_files(a.model_root)]
    allocations=[]
    weights=[]
    identities=[]
    report={"schema":1,"kind":"qwen4exp_q51_pair_screen","source":_git_metadata(ROOT),
            "host":_host_metadata(),"command":sys.argv,"weights":identities,
            "model":"Qwen3.8-Flash-Next UD-Q4_K_XL","arithmetic_class":"T0",
            "boundary":"two rotating grouped Q5_1 down banks; maps supplied, no routing or combine",
            "runtime_default_changed":False,"cases":[]}
    try:
        for layer in (0,1):
            name=f"blk.{layer}.ffn_down_exps.weight"
            reader=next(r for r in readers if any(t.name==name for t in r.info.tensors))
            info=reader.tensor_info(name)
            assert info.ggml_type_name=="Q5_1" and info.shape==(512,2560,640)
            raw=reader.tensor_data(name)
            identities.append({"path":str(reader.path),"name":name,"shape":info.shape,
                               "sha256":hashlib.sha256(raw).hexdigest()})
            weights.append(_upload(raw,runtime,allocations))
        for rows in a.rows:
            mark=len(allocations)
            compact=rows*10
            rng=np.random.default_rng(3591+rows)
            selected=np.argsort(rng.random((rows,512)),axis=1)[:,:10]
            counts=np.bincount(selected.reshape(-1),minlength=512)
            starts=np.concatenate(([0],np.cumsum(counts))).astype(np.int64)
            x,_=_make_activation(compact,640,1256+rows)
            dx,ds=[_upload(v,runtime,allocations) for v in (x,starts)]
            out=[_alloc((compact,2560),np.uint16,runtime,allocations) for _ in range(4)]
            def run(candidate):
                fn=getattr(q5,CANDIDATE if candidate else PARENT)
                for i,w in enumerate(weights):
                    fn(dx.ptr,ds.ptr,w.ptr,out[(2 if candidate else 0)+i].ptr,
                       compact,512,640,2560,library=library,runtime=runtime)
                runtime.device_synchronize()
            run(False)
            run(True)
            times={"parent":[],"candidate":[]}
            for pair in range(a.pairs):
                for candidate in ((False,True) if pair%2==0 else (True,False)):
                    start=time.perf_counter()
                    run(candidate)
                    times["candidate" if candidate else "parent"].append(time.perf_counter()-start)
                for i in range(2):
                    np.testing.assert_array_equal(
                        _download(out[i],(compact,2560),np.uint16,runtime),
                        _download(out[i+2],(compact,2560),np.uint16,runtime))
            report["cases"].append({"tokens":rows,"compact_rows":compact,
                "seconds":times,"all_pairs_exact":True,
                "speedup":statistics.median(times["parent"])/statistics.median(times["candidate"])})
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
