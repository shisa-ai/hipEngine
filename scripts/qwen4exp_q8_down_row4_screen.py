"""Actual Q8 down weights and same-layer captured routing, row1 versus row4."""
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
from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as gemv
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, _host_metadata, _git_metadata
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from scripts.qwen4exp_routing_capture import select_routing, validate_replay_identity
from tests.test_qwen4_exp_pf3_moe_schedules import _alloc, _upload, _download, _make_activation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--compiler-version-file", type=Path, required=True)
    p.add_argument("--routing-capture", type=Path, required=True)
    p.add_argument("--case-id", default="code-p4096")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--routing-layer", type=int)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", action="store_true")
    p.add_argument("--mapped-selected", action="store_true")
    args = p.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        p.error("use a positive even pair count")
    if args.mapped_selected and args.bundle:
        p.error("mapped-selected compares selected GEMV to bundled row4")
    check_host()
    identity = model_identity(args.model_root)
    capture = json.loads(args.routing_capture.read_text())
    validate_replay_identity(capture, fixture_sha256=load_fixture(DEFAULT_FIXTURE)[1])
    routing_layer = args.layer if args.routing_layer is None else args.routing_layer
    counts = select_routing(capture, case_id=args.case_id, layer=routing_layer,
                            chunk=args.chunk, tokens=512)
    if counts.shape != (512,):
        raise ValueError("requires512 experts")
    starts = np.concatenate(([0], counts.cumsum())).astype(np.int64)
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    runtime = get_hip_runtime()
    library = gemv.build_gguf_k_gemv(load=True, require_cached=True)
    readers = [GGUFReader(f) for f in discover_gguf_files(args.model_root)]
    name = f"blk.{args.layer}.ffn_down_exps.weight"
    reader = next(r for r in readers if any(t.name == name for t in r.info.tensors))
    info = reader.tensor_info(name)
    if info.shape != (512, 2560, 640) or info.ggml_type_name != "Q8_0":
        raise ValueError("not a binding Q8 down tensor")
    raw = reader.tensor_data(name)
    x, _ = _make_activation(5120, 640, 3471)
    allocations = []
    try:
        dx, ds, dw = [_upload(a, runtime, allocations) for a in (x, starts, raw)]
        mapping = selected = None
        if args.mapped_selected:
            lanes = np.random.default_rng(5421).permutation(5120).astype(np.int64)
            ids = np.empty(5120,dtype=np.int64)
            ids[lanes] = np.repeat(np.arange(512),counts)
            mapping,selected = [_upload(a,runtime,allocations) for a in (lanes,ids)]
        outputs = [_alloc((5120, 2560), np.uint16, runtime, allocations) for _ in range(2)]
        functions = [gemv.gguf_q8_0_selected_grouped_gemv_bf16_bf16_out,
                     gemv.gguf_q8_0_selected_grouped_row4_gemv_bf16_bf16_out]
        if args.bundle:
            functions = [gemv.gguf_q8_0_selected_grouped_row4_gemv_bf16_bf16_out,
                         gemv.gguf_q8_0_selected_grouped_row4_bundle_gemv_bf16_bf16_out]

        def run(i):
            if args.mapped_selected:
                if i == 0:
                    gemv.gguf_q8_0_selected_gemv_bf16_bf16_out(
                        dx.ptr,selected.ptr,dw.ptr,outputs[i].ptr,
                        5120,5120,512,640,2560,library=library,runtime=runtime)
                else:
                    gemv.gguf_q8_0_selected_grouped_row4_bundle_gemv_bf16_bf16_out(
                        dx.ptr,ds.ptr,mapping.ptr,dw.ptr,outputs[i].ptr,
                        5120,5120,512,640,2560,library=library,runtime=runtime)
            else:
                functions[i](dx.ptr, ds.ptr, 0, dw.ptr, outputs[i].ptr,
                             5120, 5120, 512, 640, 2560, library=library, runtime=runtime)
            runtime.device_synchronize()

        run(0)
        run(1)
        times = [[], []]
        for pair in range(args.pairs):
            for i in (0, 1) if pair % 2 == 0 else (1, 0):
                start = time.perf_counter()
                run(i)
                times[i].append(time.perf_counter() - start)
            np.testing.assert_array_equal(
                _download(outputs[0], (5120, 2560), np.uint16, runtime),
                _download(outputs[1], (5120, 2560), np.uint16, runtime))
        report = dict(
            schema=1, source=_git_metadata(ROOT), host=_host_metadata(),
            parent_variant=functions[0].__name__, candidate_variant=functions[1].__name__,
            model_identity=identity, quant="UD-Q4_K_XL", command=sys.argv,
            tensor=name, shape=[5120, 640, 2560],
            weight_sha256=hashlib.sha256(raw).hexdigest(),
            routing_capture_sha256=hashlib.sha256(args.routing_capture.read_bytes()).hexdigest(),
            case_id=args.case_id, chunk=args.chunk, routing_layer=routing_layer,
            counts=counts.tolist(),
            boundary="Pre-grouped BF16 activations -> Q8 down BF16 output. "
                     "Gate/up counts from explicitly named routing_layer; synthetic activations. "
                     "A different weight layer is a weight-bank screen, not its actual routing. "
                     "No routing/combine/default change.",
            seconds=times, exact=True,
            median_speedup=statistics.median(times[0]) / statistics.median(times[1]),
            mean_speedup=statistics.mean(times[0]) / statistics.mean(times[1]),
            order_speedups=[statistics.median(times[0][i::2]) / statistics.median(times[1][i::2])
                            for i in (0, 1)])
        if args.mapped_selected:
            report.update(
                parent_variant="gguf_q8_0_selected_gemv_bf16_bf16_out",
                candidate_variant="gguf_q8_0_selected_grouped_row4_bundle_gemv_bf16_bf16_out",
                boundary="Token-major BF16 activation/output; supplied counts and seeded random sorted-lane-to-original-row permutation. Both consume same expert selection. Mapping already exists after Q5_K row4 gate/up in target runtime; no map construction charged.",
                mapping_seed=5421)
    finally:
        for buf in reversed(allocations):
            free(buf, runtime=runtime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("counts", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
