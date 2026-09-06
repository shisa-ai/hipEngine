"""Screen reusable K-major Q8 weight packing against raw GGUF MMQ."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import free
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_mmq_prefill as mmq
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.benchmark.provenance import collect_artifact_provenance
from scripts.qwen4exp_canonical_ar_bench import _git_metadata, _host_metadata
from tests.test_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download
from tests.test_qwen4exp_mmq_prepack import PARENT, CANDIDATE, pack_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rows",type=int,nargs="+",default=[64,512])
    p.add_argument("--pairs",type=int,default=20)
    p.add_argument("--tensor",action="append")
    a = p.parse_args()
    if a.pairs < 2 or a.pairs % 2 or any(r < 1 for r in a.rows):
        p.error("positive rows and even pairs>=2 required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(a.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    library = mmq.build_gguf_q8_0_mmq_prefill(load=True,require_cached=True)
    readers = [GGUFReader(f) for f in discover_gguf_files(a.model_root)]
    report = dict(schema=1,source=_git_metadata(ROOT),host=_host_metadata(),
        command=sys.argv,model="Qwen3.8-Flash-Next UD-Q4_K_XL",default_changed=False,
        arithmetic="T0 parent F32 bits",cases=[],
        boundary="quantize + clear + matmul + raw-weight exact repair + synchronize; D2H excluded",
        layout="K256-major [K/256,ceil(N/128)*128,76 int32]:64 quant words +8 exact float scales +4 padding",
        resident_storage="raw and packed banks coexist; preprocessing outside request timing")
    report["hipengine_artifact_provenance"] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", device_name="AMD Radeon 8060S",
        model_path=a.model_root, model_revision="8bdc666649440e9bdc97e16f3f75782c98478ff5",
        quant="UD-Q4_K_XL", kv_dtype="BF16", command=[sys.executable,*sys.argv],
        timing_protocol=report["boundary"], warmups=1,repetitions=a.pairs)
    for name in a.tensor or ["blk.0.hc_attn_down.weight","blk.0.attn_qkv.weight"]:
        reader = next(r for r in readers if any(t.name==name for t in r.info.tensors))
        info = reader.tensor_info(name)
        n,k = info.shape
        assert info.ggml_type_name == "Q8_0" and k%256 == 0
        raw = reader.tensor_data(name)
        start = time.perf_counter_ns()
        packed = pack_reference(raw,n,k)
        pack_cpu_ms = (time.perf_counter_ns()-start)/1e6
        for rows in a.rows:
            allocations = []
            try:
                x = np.random.default_rng(3982+rows).normal(0,.2,(rows,k)).astype(np.float32)
                dx,dw = [_upload(v,runtime,allocations) for v in (x,raw)]
                start = time.perf_counter_ns()
                dp = _upload(packed,runtime,allocations)
                runtime.device_synchronize()
                packed_upload_ms = (time.perf_counter_ns()-start)/1e6
                d4 = _alloc((mmq.q8_mmq_d4x3_nbytes(rows,k),),np.uint8,runtime,allocations)
                count = _alloc((1,),np.int32,runtime,allocations)
                indices = _alloc((rows*n,),np.int32,runtime,allocations)
                outputs = [_alloc((rows,n),np.float32,runtime,allocations) for _ in range(2)]

                def run(candidate):
                    mmq.gguf_q8_0_mmq128_quantize_f32_d4x3(
                        dx.ptr,d4.ptr,rows,k,library=library,runtime=runtime)
                    runtime.memset(count.ptr,0,4)
                    getattr(mmq,CANDIDATE if candidate else PARENT)(
                        d4.ptr,(dp if candidate else dw).ptr,outputs[candidate].ptr,
                        count.ptr,indices.ptr,rows*n,0.,rows,k,n,library=library,runtime=runtime)
                    mmq.gguf_q8_0_mmq128_sparse_exact_correct_f32(
                        dx.ptr,dw.ptr,outputs[candidate].ptr,count.ptr,indices.ptr,
                        rows*n,rows,k,n,library=library,runtime=runtime)
                    runtime.device_synchronize()

                run(0)
                run(1)
                timings = [[],[]]
                for pair in range(a.pairs):
                    for mode in ((0,1) if pair%2==0 else (1,0)):
                        start = time.perf_counter_ns()
                        run(mode)
                        timings[mode].append((time.perf_counter_ns()-start)/1e6)
                    np.testing.assert_array_equal(
                        _download(outputs[0],(rows,n),np.float32,runtime).view(np.uint32),
                        _download(outputs[1],(rows,n),np.float32,runtime).view(np.uint32))
                report["cases"].append(dict(
                    tensor=name,shape=[rows,k,n],weight_sha256=hashlib.sha256(raw).hexdigest(),
                    raw_bytes=raw.nbytes,packed_bytes=packed.nbytes,pack_cpu_ms=pack_cpu_ms,
                    packed_upload_ms=packed_upload_ms,all_pairs_exact=True,
                    parent_ms=timings[0],candidate_ms=timings[1],
                    speedup=statistics.median(timings[0])/statistics.median(timings[1]),
                    mean_speedup=statistics.mean(timings[0])/statistics.mean(timings[1]),
                    order_speedups=[statistics.mean(timings[0][s::2])/statistics.mean(timings[1][s::2])
                                    for s in (0,1)]))
            finally:
                for ptr in reversed(allocations):
                    free(ptr,runtime=runtime)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
