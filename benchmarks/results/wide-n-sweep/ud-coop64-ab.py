import sys, time
sys.path.insert(0, '/home/lhl/hipEngine-ud')
from pathlib import Path
import numpy as np
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_wmma_prefill as w4a16
from hipengine.loading.gguf import GGUFReader

reader = GGUFReader('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
runtime = get_hip_runtime()
lib = w4a16.build_gguf_iq_wmma_prefill(load=True, compiler_version=Path('/tmp/ud-hipcc-version.txt').read_text())
rng = np.random.default_rng(21)
iq_names = [t.name for t in reader.info.tensors if t.ggml_type_name == 'IQ4_XS']
names = [iq_names[0], 'blk.22.ffn_gate.weight', 'blk.24.ffn_down.weight', iq_names[len(iq_names)//2], iq_names[-1]]
for tname in names:
    t = reader.tensor_info(tname)
    n, k = int(t.shape[0]), int(t.shape[1])
    raw = np.asarray(reader.tensor_data(tname))
    x_bits = ((rng.normal(0, 0.2, (1024, k)).astype(np.float32).view(np.uint32) + 0x7FFF) >> 16).astype(np.uint16)
    a = np.zeros((1024, n), dtype=np.uint16); b = np.zeros_like(a); c = np.zeros_like(a)
    bufs = []
    try:
        def dev(arr):
            d = malloc(arr.nbytes, runtime=runtime); bufs.append(d)
            copy_host_to_device(d, host_array_ptr(arr), runtime=runtime); return d
        x_dev, w_dev = dev(x_bits), dev(raw)
        oa, ob, oc = dev(a), dev(b), dev(c)
        w4a16.launch(x_dev.ptr, w_dev.ptr, oa.ptr, 512, k, n, quant='gguf_iq4_xs', library=lib)
        w4a16.launch_coop(x_dev.ptr, w_dev.ptr, ob.ptr, 512, k, n, quant='gguf_iq4_xs', library=lib)
        w4a16.launch_coop64(x_dev.ptr, w_dev.ptr, oc.ptr, 512, k, n, quant='gguf_iq4_xs', library=lib)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(a), oa, a.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(b), ob, b.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(c), oc, c.nbytes, runtime=runtime)
        print(f"{tname} ({n}x{k}): coop32-vs-onewave mism={int(np.count_nonzero(a!=b))}  coop64-vs-onewave mism={int(np.count_nonzero(a!=c))}")
        for rows in (129, 512, 1024):
            ts = {}
            for name, fn in (("coop32", w4a16.launch_coop), ("coop64", w4a16.launch_coop64)):
                def run():
                    fn(x_dev.ptr, w_dev.ptr, ob.ptr, rows, k, n, quant='gguf_iq4_xs', library=lib)
                run(); runtime.device_synchronize()
                best = float('inf')
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(5): run()
                    runtime.device_synchronize()
                    best = min(best, (time.perf_counter() - t0) / 5)
                ts[name] = best
            print(f"  rows={rows}: coop32={ts['coop32']*1e3:.3f}ms coop64={ts['coop64']*1e3:.3f}ms ({ts['coop32']/ts['coop64']:.2f}x)  [{2*rows*n*k/ts['coop64']/1e12:.1f} TF]")
    finally:
        for d in reversed(bufs): free(d, runtime=runtime)
