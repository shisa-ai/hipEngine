import json, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, 'tests')
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_to_device, free, host_array_ptr, malloc
import hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv as G
from test_gpu_gguf_k_gemv import make_q8_0_weight

rt = get_hip_runtime(); lib = G.build_gguf_k_gemv(load=True)
in_features, out_features = 2560, 10240          # attn_qkv, DE-3's largest owner
W = 32                                            # total output cols each variant covers
weight = np.ascontiguousarray(make_q8_0_weight(out_features, in_features))
rng = np.random.default_rng(20260929)
x = np.ascontiguousarray(rng.normal(0,0.2,size=(1,in_features)).astype(np.float32))
bufs=[]
w=malloc(weight.nbytes,runtime=rt); copy_host_to_device(w,host_array_ptr(weight),runtime=rt); bufs.append(w)
xd=malloc(x.nbytes,runtime=rt); copy_host_to_device(xd,host_array_ptr(x),runtime=rt); bufs.append(xd)
out1=malloc(out_features*4,runtime=rt); bufs.append(out1)
outW=malloc(out_features*4,runtime=rt); bufs.append(outW)

def timeit(fn, reps=40, warm=8):
    for _ in range(warm): fn()
    rt.device_synchronize()
    s=[]
    for _ in range(reps):
        rt.device_synchronize(); t0=time.perf_counter_ns(); fn(); rt.device_synchronize()
        s.append((time.perf_counter_ns()-t0)/1e3)
    return statistics.median(s)

# Baseline: the actual decode kernel (8 output features per block, out_features>>3 blocks)
base = timeit(lambda: G.gguf_q8_0_pack8_gemv_f32_f32_out(
    xd.ptr, w.ptr, out1.ptr, 1, in_features, out_features, library=lib, runtime=rt))

variants = [("coltile4_rowbatch8", G.gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out, 4),
            ("coltile8_rowbatch4", G.gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out, 8),
            ("coltile8_rowbatch8", G.gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out, 8),
            ("coltile16_rowbatch2", G.gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out, 16),
            ("coltile16_rowbatch4", G.gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out, 16),
            ("coltile32_rowbatch1", G.gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out, 32)]
rows=[{"variant":"pack8_decode_baseline","out_cols_per_block":8,"us":base,"blocks":out_features//8}]
print(f"{'variant':<26} {'out/block':>9} {'us':>9} {'vs baseline':>12}")
print(f"{'pack8_decode_baseline':<26} {8:>9} {base:>9.2f} {'1.00x':>12}")
for name, fn, cols in variants:
    # SAME total work as the baseline: out_features output columns either way.
    t = timeit(lambda f=fn: f(xd.ptr, w.ptr, outW.ptr, 1, in_features, out_features, library=lib, runtime=rt))
    speedup = base / t
    rows.append({"variant":name,"out_cols_per_block":cols,"us":t,"speedup_vs_baseline":speedup})
    print(f"{name:<26} {cols:>9} {t:>9.2f} {speedup:>11.2f}x")
art={"schema":1,"kind":"qwen4exp_de3_output_pair_screen","performance_claim":True,
     "created_at":datetime.now(timezone.utc).isoformat(),
     "boundary":"q8_0 GEMV output-tile sweep at decode width (rows=1); kernel-level only, excludes attention and MoE",
     "shape":{"in_features":in_features,"out_features":out_features},"rows_measured":1,
     "reps":40,"warmups":8,"results":rows}
p=Path('benchmarks/results/2026-09-29-qwen4exp-de3-output-pair-screen'); p.mkdir(parents=True,exist_ok=True)
(p/'screen.json').write_text(json.dumps(art,indent=2)+"\n")
for b in reversed(bufs): free(b,runtime=rt)
