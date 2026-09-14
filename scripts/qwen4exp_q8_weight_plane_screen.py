"""Real-weight Q8 down owner screen with explicit synthetic routing shapes."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--tensor", default="blk.2.ffn_down_exps.weight")
    parser.add_argument("--rows", nargs="+", type=int, default=[64, 512, 1024, 4096])
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_host()
    from hipengine.loading.gguf import GGUFReader, discover_gguf_files
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
        build_gguf_q8_0_prefill,
        gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out as parent,
        gguf_q8_0_selected_grouped_wmma_residual_prefill_bf16_bf16_out as residual,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        build_gguf_k_gemv, gguf_q8_0_selected_grouped_gemv_bf16_bf16_out as strict,
    )
    from tests.test_gpu_qwen4exp_q8_0_grouped_wmma_down import _tile_map, _f32_to_bf16_bits
    from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = model_identity(args.model_root)
        readers = [GGUFReader(p) for p in discover_gguf_files(args.model_root)]
        reader = next(r for r in readers if any(t.name == args.tensor for t in r.info.tensors))
        info = reader.tensor_info(args.tensor)
        if info.ggml_type_name != "Q8_0":
            raise ValueError("Q8_0 tensor required")
        experts, n, k = info.shape
        raw = np.ascontiguousarray(reader.tensor_data(args.tensor))
        runtime = get_hip_runtime()
        library, strict_library = build_gguf_q8_0_prefill(), build_gguf_k_gemv()
        allocations = []
        weight = _upload(raw, runtime, allocations)
        report = {
            "kind": "qwen4exp_q8_weight_plane_screen", "status": "running",
            "performance_claim": False, "source": _git_metadata(ROOT), "host": _host_metadata(),
            "model_identity": identity, "command": sys.argv,
            "tensor": args.tensor, "shape": list(info.shape),
            "tensor_sha256": hashlib.sha256(raw).hexdigest(),
            "library_sha256": hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
            "routing": "synthetic uniform/skew counts, not measured model routing",
            "cases": [],
        }
        try:
            for tokens in args.rows:
                for shape in ("uniform", "skew"):
                    count = tokens * 10
                    rng = np.random.default_rng(8713 + tokens)
                    probabilities = np.ones(experts)
                    if shape == "skew":
                        probabilities[:max(1, experts // 8)] = 32
                    counts = rng.multinomial(count, probabilities / probabilities.sum()).tolist()
                    starts, padded, tiles_count = _tile_map(counts)
                    tiles = np.repeat(np.arange(experts, dtype=np.int64), [(c + 15) // 16 for c in counts])
                    mark = len(allocations)
                    try:
                        x = _f32_to_bf16_bits(rng.normal(0, .2, (count, k)).astype(np.float32))
                        inputs = [_upload(a, runtime, allocations) for a in (x, starts, padded, tiles)]
                        outputs = {name: _alloc(count * n, np.uint16, runtime, allocations)
                                   for name in ("parent", "residual", "strict")}

                        def launch(name):
                            if name == "strict":
                                strict(inputs[0].ptr, inputs[1].ptr, 0, weight.ptr, outputs[name].ptr,
                                       count, count, experts, k, n, library=strict_library, runtime=runtime)
                            else:
                                fn = parent if name == "parent" else residual
                                fn(*(b.ptr for b in inputs), weight.ptr, outputs[name].ptr,
                                   count, experts, k, n, tiles_count * 16, library=library, runtime=runtime)

                        for _ in range(2):
                            for name in outputs:
                                launch(name)
                        runtime.device_synchronize()
                        results = {name: (_download(buf, (count, n), np.uint16, runtime).astype(np.uint32) << 16).view(np.float32)
                                   for name, buf in outputs.items()}
                        if not all(np.isfinite(a).all() for a in results.values()):
                            raise ValueError("nonfinite owner output")
                        samples = []
                        for pair in range(args.pairs):
                            for name in (tuple(outputs) if pair % 2 == 0 else tuple(reversed(outputs))):
                                begin, end = runtime.event_create(), runtime.event_create()
                                try:
                                    runtime.event_record(begin)
                                    launch(name)
                                    runtime.event_record(end)
                                    runtime.event_synchronize(end)
                                    samples.append({"arm": name, "pair": pair,
                                                    "ms": runtime.event_elapsed_time_ms(begin, end)})
                                finally:
                                    runtime.event_destroy(end)
                                    runtime.event_destroy(begin)
                        medians = {name: statistics.median(s["ms"] for s in samples if s["arm"] == name) for name in outputs}
                        report["cases"].append({
                            "tokens": tokens, "compact_rows": count, "routing": shape,
                            "expert_counts": counts, "median_ms": medians, "samples": samples,
                            "mse_vs_strict": {name: float(np.mean((results[name].astype(np.float64) - results["strict"]) ** 2))
                                              for name in ("parent", "residual")},
                        })
                        print(tokens, shape, medians, flush=True)
                    finally:
                        while len(allocations) > mark:
                            free(allocations.pop(), runtime=runtime)
            report["status"] = "completed"
        finally:
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
