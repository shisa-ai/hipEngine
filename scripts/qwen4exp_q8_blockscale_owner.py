"""Real Q8-down weights, seeded skewed routing, synthetic BF16 activations."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_canonical_ar_bench import _host_metadata, _git_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 512, 1024])
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--require-cached", action="store_true")
    args = parser.parse_args()
    check_host()
    if args.pairs < 2 or any(row <= 0 for row in args.rows):
        parser.error("positive rows and at least two pairs required")
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, memory_stats
    from hipengine.loading.gguf import GGUFReader, discover_gguf_files
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gemv
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_prefill as wmma
    from tests.test_gpu_qwen4_exp_pf3_moe_schedules import _upload, _alloc, _download
    from tests.test_gpu_qwen4exp_q8_blockscale import bf16

    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = get_hip_runtime()
        parent_lib = gemv.build_gguf_k_gemv(require_cached=args.require_cached)
        wmma_lib = wmma.build_gguf_q8_0_prefill(require_cached=args.require_cached)
        readers = [GGUFReader(path) for path in discover_gguf_files(args.model_root)]
        name = "blk.2.ffn_down_exps.weight"
        reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
        info = reader.tensor_info(name)
        if info.shape != (512, 2560, 640) or info.ggml_type_name != "Q8_0":
            raise ValueError("wrong Q8-down tensor")
        raw = reader.tensor_data(name)
        report = dict(command=sys.argv, source=_git_metadata(ROOT), host=_host_metadata(),
                      model=model_identity(args.model_root), tensor=name,
                      weight_sha256=hashlib.sha256(raw).hexdigest(),
                      protocol=__doc__, performance_claim=False, cases=[],
                      libraries={key: dict(path=lib._name, sha256=hashlib.sha256(
                          Path(lib._name).read_bytes()).hexdigest())
                          for key, lib in (("parent", parent_lib), ("wmma", wmma_lib))})
        for tokens in args.rows:
            rng = np.random.default_rng(8543 + tokens)
            compact = tokens * 10
            probabilities = rng.lognormal(0, 1.2, 512)
            counts = rng.multinomial(compact, probabilities / probabilities.sum())
            starts = np.r_[0, counts.cumsum()].astype(np.int64)
            padded = (counts + 15) // 16 * 16
            wmma_starts = np.r_[0, padded.cumsum()].astype(np.int64)
            tiles = np.repeat(np.arange(512), padded // 16).astype(np.int64)
            xb = bf16(rng.normal(0, .2, (compact, 640)).astype(np.float32))
            allocations = []
            try:
                x, s, ws, t, w = [_upload(a, runtime, allocations)
                                  for a in (xb, starts, wmma_starts, tiles, raw)]
                out = [_alloc((compact, 2560), np.uint16, runtime, allocations) for _ in range(4)]
                risk_count = _alloc(1, np.int32, runtime, allocations)
                risk_indices = _alloc(compact * 2560, np.int32, runtime, allocations)
                functions = (
                    gemv.gguf_q8_0_selected_grouped_row4_register_gemv_bf16_bf16_out,
                    wmma.gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
                    wmma.gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out,
                    wmma.gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out,
                )

                def launch(index):
                    if index == 0:
                        functions[index](x.ptr, s.ptr, 0, w.ptr, out[index].ptr,
                                         compact, compact, 512, 640, 2560,
                                         library=parent_lib, runtime=runtime)
                    else:
                        extra = dict(risk_count_ptr=risk_count.ptr,
                                     risk_indices_ptr=risk_indices.ptr,
                                     risk_capacity=compact * 2560) if index == 3 else {}
                        functions[index](x.ptr, s.ptr, ws.ptr, t.ptr, w.ptr, out[index].ptr,
                                         compact, 512, 640, 2560, int(wmma_starts[-1]),
                                         library=wmma_lib, runtime=runtime, **extra)

                for i in range(4):
                    launch(i)
                runtime.device_synchronize()
                reference_bits = _download(out[0], (compact, 2560), np.uint16, runtime)
                reference = (reference_bits.astype(np.uint32) << 16).view(np.float32)
                errors = []
                for i in (1, 2, 3):
                    bits = _download(out[i], (compact, 2560), np.uint16, runtime)
                    delta = (bits.astype(np.uint32) << 16).view(np.float32) - reference
                    if not np.isfinite(delta).all():
                        raise ValueError("nonfinite output")
                    errors.append(dict(changed=int(np.count_nonzero(bits != reference_bits)),
                                       elements=int(bits.size), mse=float(np.mean(delta ** 2)),
                                       max_abs=float(np.max(np.abs(delta)))))
                    if i == 3:
                        np.testing.assert_array_equal(bits, reference_bits)
                queued = int(_download(risk_count, (1,), np.int32, runtime)[0])
                times = [[], [], [], []]
                for pair in range(args.pairs):
                    for i in ((0, 1, 2, 3) if pair % 2 == 0 else (3, 2, 1, 0)):
                        start, stop = runtime.event_create(), runtime.event_create()
                        try:
                            runtime.event_record(start)
                            launch(i)
                            runtime.event_record(stop)
                            runtime.event_synchronize(stop)
                            times[i].append(runtime.event_elapsed_time_ms(start, stop))
                        finally:
                            runtime.event_destroy(stop)
                            runtime.event_destroy(start)
                case = dict(tokens=tokens, compact_rows=compact, counts=counts.tolist(),
                            repaired=queued, repair_fraction=queued / (compact * 2560),
                            milliseconds=times, medians=list(map(statistics.median, times)),
                            errors=errors)
                report["cases"].append(case)
                print(tokens, case["medians"], errors, flush=True)
            finally:
                for allocation in reversed(allocations):
                    free(allocation)
        report["after_close"] = memory_stats()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
