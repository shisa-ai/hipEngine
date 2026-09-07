"""Actual F32 router weights, exact shared reduction versus wave-tail reduction."""
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
from hipengine.loading.gguf import GGUFReader,discover_gguf_files
from hipengine.kernels.hip_gfx1100.moe import router
from scripts.qwen4exp_framework_family_refresh import check_host,model_identity
from scripts.qwen4exp_canonical_ar_bench import _host_metadata,_git_metadata
from tests.test_qwen4_exp_pf3_moe_schedules import _upload,_alloc,_download
from tests.test_qwen4exp_router_shuffle_tail import PARENT,CANDIDATE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rows",type=int,nargs="+",default=[512,1024])
    p.add_argument("--layers",type=int,nargs="+",default=[0,27])
    p.add_argument("--pairs",type=int,default=20)
    a=p.parse_args()
    if a.pairs<2 or a.pairs%2 or any(r<1 for r in a.rows):
        p.error("positive rows and even pairs>=2 required")
    check_host()
    identity=model_identity(a.model_root)
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"]=str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"]="1"
    rt=get_hip_runtime()
    lib=router.build_qwen35_router(load=True,require_cached=True)
    readers=[GGUFReader(f) for f in discover_gguf_files(a.model_root)]
    report=dict(schema=1,source=_git_metadata(ROOT),host=_host_metadata(),
                model_identity=identity,command=sys.argv,default_changed=False,
                boundary="F32 router logits + synchronize;D2H excluded,selection not included",
                arithmetic="T0 parent F32 bits",cases=[])
    for layer in a.layers:
        name=f"blk.{layer}.ffn_gate_inp.weight"
        reader=next(r for r in readers if any(t.name==name for t in r.info.tensors))
        info=reader.tensor_info(name)
        assert info.ggml_type_name=="F32"
        n,k=info.shape
        raw=reader.tensor_data(name)
        for rows in a.rows:
            allocations=[]
            try:
                x=np.random.default_rng(1914+rows).normal(0,.2,(rows,k)).astype(np.float32)
                dx,dw=[_upload(v,rt,allocations) for v in (x,raw)]
                out=[_alloc((rows,n),np.float32,rt,allocations) for _ in range(2)]
                def run(i):
                    getattr(router,(PARENT,CANDIDATE)[i])(
                        dx.ptr,dw.ptr,out[i].ptr,rows,k,n,library=lib,runtime=rt)
                    rt.device_synchronize()
                run(0)
                run(1)
                times=[[],[]]
                for pair in range(a.pairs):
                    for mode in ((0,1) if pair%2==0 else (1,0)):
                        start=time.perf_counter_ns()
                        run(mode)
                        times[mode].append((time.perf_counter_ns()-start)/1e6)
                    np.testing.assert_array_equal(
                        _download(out[0],(rows,n),np.float32,rt).view(np.uint32),
                        _download(out[1],(rows,n),np.float32,rt).view(np.uint32))
                report["cases"].append(dict(tensor=name,shape=[rows,k,n],
                    weight_sha256=hashlib.sha256(raw).hexdigest(),ms=times,all_pairs_exact=True,
                    mean_ms=[statistics.mean(t) for t in times],
                    mean_speedup=statistics.mean(times[0])/statistics.mean(times[1]),
                    order_speedups=[statistics.mean(times[0][i::2])/statistics.mean(times[1][i::2])
                                    for i in (0,1)]))
            finally:
                for buf in reversed(allocations):
                    free(buf,runtime=rt)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
