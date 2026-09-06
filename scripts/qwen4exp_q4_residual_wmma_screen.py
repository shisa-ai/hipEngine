"""Screen T2 two-plane WMMA against exact Q4 pair2 and single-plane WMMA."""
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
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.kernels.hip_gfx1100.moe import group_scatter as group
from hipengine.kernels.hip_gfx1100.fused import paro_silu
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.quant.gguf import bf16_to_float32
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE,load_fixture,_host_metadata,_git_metadata
from scripts.qwen4exp_routing_capture import select_routing,validate_replay_identity
from scripts.qwen4exp_framework_family_refresh import check_host,model_identity
from tests.test_qwen4exp_q4_residual_wmma import tile_map,CANDIDATE
from tests.test_qwen4exp_q4_bundle import PAIR
from tests.test_qwen4_exp_pf3_moe_schedules import _upload,_alloc,_download,_make_activation


def metrics(reference,actual):
    a,b = bf16_to_float32(reference),bf16_to_float32(actual)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite projection/SiLU output")
    return dict(bf16_element_agreement=float(np.mean(reference==actual)),
                max_abs=float(np.max(np.abs(a-b))),
                relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),
                row_top1_agreement=float(np.mean(a.argmax(1)==b.argmax(1))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root",type=Path,required=True)
    p.add_argument("--compiler-version-file",type=Path,required=True)
    p.add_argument("--routing-capture",type=Path,required=True)
    p.add_argument("--case-id",default="code-p4096")
    p.add_argument("--chunk",type=int,default=0)
    p.add_argument("--layer",type=int,default=0)
    p.add_argument("--repetitions",type=int,default=12)
    p.add_argument("--tile-m",type=int,choices=(16,32,64),default=16)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.repetitions<3 or args.repetitions%3:
        p.error("repetitions must be a positive multiple of3")
    check_host()
    identity = model_identity(args.model_root)
    capture = json.loads(args.routing_capture.read_text())
    validate_replay_identity(capture,fixture_sha256=load_fixture(DEFAULT_FIXTURE)[1])
    counts = select_routing(capture,case_id=args.case_id,layer=args.layer,chunk=args.chunk,tokens=512)
    starts,padded,tiles = tile_map(counts)
    rows,k,n,experts = int(starts[-1]),2560,640,512
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    lib = q4.build_gguf_q4_k_selected_prefill(load=True,require_cached=True)
    group_lib = group.build_qwen35_moe_group_scatter(load=True,require_cached=True)
    silu_lib = paro_silu.build_paro_silu(load=True,require_cached=True)
    readers = [GGUFReader(f) for f in discover_gguf_files(args.model_root)]
    raw,weights = [],[]
    for name in (f"blk.{args.layer}.ffn_gate_exps.weight",f"blk.{args.layer}.ffn_up_exps.weight"):
        reader = next(r for r in readers if any(t.name==name for t in r.info.tensors))
        info = reader.tensor_info(name)
        assert info.ggml_type_name=="Q4_K" and info.shape==(experts,n,k)
        value = reader.tensor_data(name)
        raw.append(value)
        weights.append(dict(name=name,sha256=hashlib.sha256(value).hexdigest()))
    allocations = []
    report = dict(schema=1,source=_git_metadata(ROOT),host=_host_metadata(),command=sys.argv,
                  model_identity=identity,weights=weights,arithmetic="T2,not parent exact",
                  runtime_default_changed=False,shape=[rows,k,n,experts],tile_m=args.tile_m,
                  routing=dict(case_id=args.case_id,chunk=args.chunk,layer=args.layer,
                               sha256=hashlib.sha256(args.routing_capture.read_bytes()).hexdigest()),
                  boundary="WMMA arms: GPU tile-map + concatenated gate/up + BF16 SiLU; exact arm: separate gate/up + same BF16 SiLU. Routing counts/compact input supplied.",
                  limits="Synthetic activations; known fixture padded launch count avoids D2H,so this is not a model-path latency claim or production numerical qualification.")
    try:
        x,_ = _make_activation(rows,k,3456+512)
        dx,ds,da,db = [_upload(v,runtime,allocations) for v in (x,starts,*raw)]
        dp = _alloc(padded.shape,np.int64,runtime,allocations)
        dt = _alloc(tiles.shape,np.int64,runtime,allocations)
        total = _alloc((1,),np.int64,runtime,allocations)
        exact = [_alloc((rows,n),np.uint16,runtime,allocations) for _ in range(2)]
        joined = [_alloc((rows,2*n),np.uint16,runtime,allocations) for _ in range(2)]
        outputs = [_alloc((rows,n),np.uint16,runtime,allocations) for _ in range(3)]
        names = ["exact_pair2","wmma_f16","wmma_f16x2"]
        def run(mode):
            if mode==0:
                getattr(q4,PAIR)(dx.ptr,ds.ptr,da.ptr,db.ptr,exact[0].ptr,exact[1].ptr,
                                rows,experts,k,n,library=lib,runtime=runtime)
                paro_silu.silu_mul_separate_out_bf16(
                    exact[0].ptr,exact[1].ptr,outputs[0].ptr,rows,n,library=silu_lib,runtime=runtime)
            else:
                group.qwen35_moe_wmma_tile_map(ds.ptr,dp.ptr,dt.ptr,total.ptr,experts,
                    tile_capacity=len(tiles),library=group_lib,runtime=runtime)
                fn = (q4.gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out
                      if mode==1 else getattr(q4,CANDIDATE))
                fn(dx.ptr,ds.ptr,dp.ptr,dt.ptr,da.ptr,db.ptr,joined[mode-1].ptr,
                   rows,k,n,n,experts,int(padded[-1]),tile_m=args.tile_m,tile_n=16,library=lib,runtime=runtime)
                paro_silu.silu_mul_dual_out_bf16(
                    joined[mode-1].ptr,outputs[mode].ptr,rows,n,library=silu_lib,runtime=runtime)
            runtime.device_synchronize()
        initial = []
        initial_gate = []
        for mode in range(3):
            run(mode)
            initial.append(_download(outputs[mode],(rows,n),np.uint16,runtime))
            initial_gate.append(
                np.concatenate([_download(v,(rows,n),np.uint16,runtime) for v in exact],axis=1)
                if mode==0 else _download(joined[mode-1],(rows,2*n),np.uint16,runtime))
        np.testing.assert_array_equal(_download(dp,padded.shape,np.int64,runtime),padded)
        np.testing.assert_array_equal(_download(dt,tiles.shape,np.int64,runtime),tiles)
        assert int(_download(total,(1,),np.int64,runtime)[0]) == int(padded[-1])
        times = [[] for _ in range(3)]
        for rep in range(args.repetitions):
            for slot in range(3):
                mode = (rep+slot)%3
                start = time.perf_counter_ns()
                run(mode)
                times[mode].append((time.perf_counter_ns()-start)/1e6)
        for mode in range(3):
            np.testing.assert_array_equal(_download(outputs[mode],(rows,n),np.uint16,runtime),initial[mode])
        parent = np.concatenate([_download(v,(rows,n),np.uint16,runtime) for v in exact],axis=1)
        report["arms"] = {}
        for mode,name in enumerate(names):
            gate_up = parent if mode==0 else _download(joined[mode-1],(rows,2*n),np.uint16,runtime)
            np.testing.assert_array_equal(gate_up,initial_gate[mode])
            report["arms"][name] = dict(
                ms=times[mode],mean_ms=statistics.mean(times[mode]),median_ms=statistics.median(times[mode]),
                order_mean_ms=[statistics.mean(times[mode][r::3]) for r in range(3)],
                gate_up=metrics(parent,gate_up),silu=metrics(initial[0],initial[mode]),
                deterministic=True)
        report["status"] = "screen_complete_not_model_qualified"
    finally:
        for buf in reversed(allocations):
            free(buf,runtime=runtime)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
