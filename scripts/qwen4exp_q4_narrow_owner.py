#!/usr/bin/env python3
"""Pair checked/wide Q4 indexing on live natural routed gate/up inputs.

Times the complete count/prefix/scatter-gather/tile-map/risk/repair/SiLU
operation with private replay scratch. Router, down and combine are excluded.
Counterbalanced wall and HIP-event samples include the same work in both arms.
Compare outputs and risk membership in original routed-lane order so atomic
scatter order is not mistaken for a numerical or ownership difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.memory import free, malloc, memory_stats
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_out_bf16
from hipengine.kernels.hip_gfx1100.moe import group_scatter as group
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.runtime import qwen4_exp_runner as runner_module
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from tests.test_gpu_qwen4exp_q4_iu8_exact import _download

PARENT = "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"
CANDIDATE = "selected_dual_wmma_iu8_risk_narrow_prefill_bf16_bf16_out"


def canonical_risks(indices, lanes, cols):
    return np.sort(lanes[indices // cols] * cols + indices % cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two counterbalanced pairs required")
    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    if args.require_cached_build:
        os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile = gate.resolve_runtime_profile(
        model=gate.QWEN4_EXP_MODEL, backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1], profile=gate.ExecutionProfile.PRODUCTION)
    fixture, digest = gate.load_fixture(gate.DEFAULT_FIXTURE)
    ids = args.case_id or ["code-p512", "code-p1024", "code-p4096",
                          "general_en-p1024", "general_ja-p1024", "mixed_ja_en-p1024"]
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in ids]
    layers = set(args.layer or [0, 24, 47])
    if not layers <= (set(range(48)) - {2}):
        parser.error("layers must be Q4 gate/up layers in0..47, excluding Q5 layer2")
    index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=gate.resolve_model(index.architecture or ""),
        backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
    runtime = generator.runner.runtime
    library = q4.build_gguf_q4_k_selected_prefill(load=True)
    group_library = group.build_qwen35_moe_group_scatter(load=True)
    key = KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k", PARENT)
    parent = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=PARENT)
    candidate = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=CANDIDATE)
    original_moe = runner_module.run_qwen4_exp_moe
    current = {}
    captures = []
    start, stop = runtime.event_create(), runtime.event_create()
    report = dict(schema=1, source=gate._git_metadata(ROOT), host=gate._host_metadata(),
                  command=sys.argv, model_identity=model_identity(args.model_root),
                  fixture_sha256=digest, manifest_sha256=profile.manifest_sha256,
                  manifest_scope="named production with diagnostic replay hook, default unchanged",
                  parent_variant=PARENT, candidate_variant=CANDIDATE,
                  boundary="count/prefix/scatter-gather/tile-map/reset/risk/repair/SiLU; excludes router/down/combine",
                  chunk_size=4096, layers=sorted(layers), cases=[],
                  performance_claim=False, status="running")

    def download(ptr, shape, dtype):
        return _download(SimpleNamespace(ptr=ptr, nbytes=int(np.prod(shape)) * np.dtype(dtype).itemsize),
                         shape, dtype, runtime)

    def moe_hook(*a, **kw):
        slot = a[1]["expert_gate"].spec.slot_path
        current.update(scratch=kw["scratch"], layer=int(slot.split(".")[1]))
        try:
            return original_moe(*a, **kw)
        finally:
            current.clear()

    def hooked(*a, **kw):
        layer = current.get("layer")
        if layer not in layers:
            return parent(*a, **kw)
        rows, k, n, b, experts, padded = map(int, a[11:17])
        if int(kw.get("stream", 0)) != 0 or (k, n, b, experts) != (2560, 640, 640, 512):
            raise ValueError("natural replay requires default stream and pinned Q4 geometry")
        if not q4.q4_iu8_narrow_index_admissible(rows, k, n, b, experts, padded):
            raise ValueError("natural shape outside narrow domain")
        scratch = current["scratch"]
        topk = 10
        tokens = rows // topk
        capacity = rows * (n + b)
        tile_capacity = padded // 16
        sizes = dict(count=experts * 4, pc=experts * 4, starts=(experts + 1) * 8,
                     total=8, offsets=experts * 4, lanes=rows * 8, exps=rows * 8,
                     weights=rows * 4, x=rows * k * 2, ws=(experts + 1) * 8,
                     tiles=tile_capacity * 8, out=capacity * 2, silu=rows * n * 2,
                     count_risk=4, indices=capacity * 4)
        buffers = {}
        samples = {"parent": {"event_ms": [], "wall_ms": []},
                   "candidate": {"event_ms": [], "wall_ms": []}}
        ref = None
        input_hash = hashlib.sha256(download(scratch.hidden_bf16.ptr, (tokens, k), np.uint16)).hexdigest()
        route_hash = hashlib.sha256(download(scratch.selected.ptr, (rows,), np.int64)).hexdigest()
        immutable = [(a[4], (experts, n, k // 256 * 144), np.uint8),
                     (a[5], (experts, b, k // 256 * 144), np.uint8)]
        weight_hashes = [hashlib.sha256(download(p, shape, dtype)).hexdigest() for p, shape, dtype in immutable]
        try:
            for name, size in sizes.items():
                buffers[name] = malloc(size, runtime=runtime)
            def ptr(name):
                return buffers[name].ptr
            def operation(fn):
                runtime.memset(ptr("count"), 0, sizes["count"])
                group.qwen35_moe_group_count(scratch.selected.ptr, ptr("count"), rows, experts,
                                            library=group_library, runtime=runtime)
                group.qwen35_moe_group_prefix(ptr("count"), ptr("pc"), ptr("starts"), ptr("total"),
                                             experts, 1, library=group_library, runtime=runtime)
                runtime.memset(ptr("offsets"), 0, sizes["offsets"])
                group.qwen35_moe_group_scatter_gather_lowp(
                    scratch.hidden_bf16.ptr, scratch.selected.ptr, scratch.routing.ptr,
                    ptr("starts"), ptr("offsets"), ptr("lanes"), ptr("exps"), ptr("weights"),
                    ptr("x"), rows, experts, topk, k, library=group_library, runtime=runtime)
                group.qwen35_moe_wmma_tile_map(ptr("starts"), ptr("ws"), ptr("tiles"), ptr("total"),
                                             experts, tile_capacity=tile_capacity,
                                             library=group_library, runtime=runtime)
                runtime.memset(ptr("count_risk"), 0, 4)
                fn(ptr("x"), ptr("starts"), ptr("ws"), ptr("tiles"), a[4], a[5], ptr("out"),
                   ptr("count_risk"), ptr("indices"), capacity, a[10], rows, k, n, b, experts, padded,
                   library=library, runtime=runtime)
                q4.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
                    ptr("x"), ptr("starts"), a[4], a[5], ptr("out"), ptr("count_risk"), ptr("indices"),
                    capacity, rows, k, n, b, experts, library=library, runtime=runtime)
                silu_mul_dual_out_bf16(ptr("out"), ptr("silu"), rows, n, runtime=runtime)
            for pair in range(args.pairs + 1):
                for arm in ((0, 1) if (pair + len(captures)) % 2 == 0 else (1, 0)):
                    runtime.device_synchronize()
                    began = time.perf_counter()
                    runtime.event_record(start)
                    operation(parent if arm == 0 else candidate)
                    runtime.event_record(stop)
                    runtime.event_synchronize(stop)
                    wall = (time.perf_counter() - began) * 1e3
                    event = runtime.event_elapsed_time_ms(start, stop)
                    if pair:
                        samples["parent" if arm == 0 else "candidate"]["event_ms"].append(event)
                        samples["parent" if arm == 0 else "candidate"]["wall_ms"].append(wall)
                    lanes = download(ptr("lanes"), (rows,), np.int64)
                    np.testing.assert_array_equal(np.sort(lanes), np.arange(rows))
                    count = int(download(ptr("count_risk"), (1,), np.int32)[0])
                    assert 0 <= count <= capacity
                    indices = download(ptr("indices"), (capacity,), np.int32)[:count]
                    out = download(ptr("out"), (rows, n + b), np.uint16)
                    silu = download(ptr("silu"), (rows, n), np.uint16)
                    canonical_out = np.empty_like(out); canonical_out[lanes] = out
                    canonical_silu = np.empty_like(silu); canonical_silu[lanes] = silu
                    result = (canonical_out, canonical_silu, count, canonical_risks(indices, lanes, n + b))
                    if ref is None:
                        ref = result
                    else:
                        for old, new in zip(ref, result):
                            np.testing.assert_array_equal(old, new)
            assert hashlib.sha256(download(scratch.hidden_bf16.ptr, (tokens, k), np.uint16)).hexdigest() == input_hash
            assert hashlib.sha256(download(scratch.selected.ptr, (rows,), np.int64)).hexdigest() == route_hash
            for (p, shape, dtype), digest_w in zip(immutable, weight_hashes):
                assert hashlib.sha256(download(p, shape, dtype)).hexdigest() == digest_w
            medians = {arm: {metric: statistics.median(v) for metric, v in values.items()}
                       for arm, values in samples.items()}
            captures.append(dict(layer=layer, tokens=tokens, compact_rows=rows, padded_rows=padded,
                                 samples=samples, medians=medians,
                                 speedup=medians["parent"]["event_ms"] / medians["candidate"]["event_ms"],
                                 wall_speedup=medians["parent"]["wall_ms"] / medians["candidate"]["wall_ms"],
                                 risk_count=ref[2], risk_membership_sha256=hashlib.sha256(ref[3]).hexdigest(),
                                 output_sha256=hashlib.sha256(ref[0]).hexdigest(),
                                 silu_sha256=hashlib.sha256(ref[1]).hexdigest(),
                                 all_pairs_exact=True, inputs_immutable=True))
            print("layer", layer, "tokens", tokens, "speedup", captures[-1]["speedup"], flush=True)
        finally:
            for buffer in reversed(list(buffers.values())):
                free(buffer, runtime=runtime)
        return parent(*a, **kw)

    try:
        runner_module.run_qwen4_exp_moe = moe_hook
        register(key, hooked, replace=True)
        for case in cases:
            captures.clear()
            token = generator.runner.prefill(case["prompt_token_ids"])
            if {c["layer"] for c in captures} != layers or len(captures) != len(layers):
                raise AssertionError((case["id"], "engagement", len(captures)))
            totals = {arm: sum(c["medians"][arm]["event_ms"] for c in captures)
                      for arm in ("parent", "candidate")}
            report["cases"].append(dict(id=case["id"], first_token=token.token_id, calls=len(captures),
                                         totals_ms=totals, speedup=totals["parent"] / totals["candidate"],
                                         captures=list(captures)))
            print(case["id"], totals, flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        runner_module.run_qwen4_exp_moe = original_moe
        register(key, parent, replace=True)
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
