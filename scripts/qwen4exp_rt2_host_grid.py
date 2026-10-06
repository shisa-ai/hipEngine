#!/usr/bin/env python3
"""Screen host-sized Q4/Q5_1 routed grids on natural production inputs.

Both arms rebuild the same device-owned tile map, reset the risk counter, run
one unchanged iu8 owner plus exact repair, and (for Q4) publish SiLU. The host
arm downloads the map's 8-byte total inside the wall/event interval. Count,
prefix and scatter/gather are shared immutable inputs outside this boundary;
router/down/combine/model wall are not priced by this diagnostic screen.
Q8 guarded down is observed only: its runtime already reads the exact total.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.core.memory import (
    DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc, memory_stats,
)


def static_tile_bound(compact_rows: int, experts: int) -> int:
    """Maximum sum(ceil(count/16)) given rows and at most experts buckets."""
    if compact_rows <= 0 or experts <= 0:
        raise ValueError("rows and experts must be positive")
    active = min(compact_rows, experts)
    return active + (compact_rows - active) // 16


def checked_total_rows(total: int, *, compact_rows: int, static_rows: int) -> int:
    if not compact_rows <= total <= static_rows or total % 16:
        raise ValueError("device tile total is outside the complete launch bound")
    return int(total)


def launch_rows(arm, *, runtime, stream, total_buffer, host, compact_rows, static_rows):
    if arm == "static":
        return static_rows
    if arm != "host":
        raise ValueError("unknown grid arm")
    if stream:
        runtime.stream_synchronize(stream)
    copy_device_to_host(host_array_ptr(host), total_buffer, 8, runtime=runtime)
    return checked_total_rows(int(host[0]), compact_rows=compact_rows, static_rows=static_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--pairs", type=int, default=6)
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error("at least two counterbalanced pairs required")
    from scripts import qwen4exp_row4_state_gate as gate
    from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
    from hipengine.kernels.hip_gfx1100.moe import group_scatter as group
    from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill as q4
    from hipengine.kernels.hip_gfx1100.quant import qwen4_exp_q5_1 as q51
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_dual_out_bf16
    from hipengine.kernels.registry import KernelKey, register, resolve
    from hipengine.runtime import qwen4_exp_runner as runner_module
    from scripts.qwen4exp_ed3_mixed_quant_gate import Q8_GUARDED_KEY

    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile = gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL, backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1], profile=gate.ExecutionProfile.PRODUCTION)
    fixture, digest = gate.load_fixture(gate.DEFAULT_FIXTURE)
    ids = args.case_id or ["code-p512", "code-p1024", "code-p4096",
                          "general_en-p1024", "general_ja-p1024", "mixed_ja_en-p1024"]
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in ids]
    layers = set(args.layer or [0, 24, 47])
    if not layers <= (set(range(48)) - {2}):
        parser.error("screen Q4 layers 0..47 except Q5_K layer2")
    report = dict(schema=1, status="running", performance_claim=False, command=sys.argv,
        source=gate._git_metadata(ROOT), host=gate._host_metadata(), model_identity=model_identity(args.model_root),
        fixture_sha256=digest, manifest_sha256=profile.manifest_sha256,
        strict_manifest_sha256=profile.strict_manifest_sha256, execution_profile="production",
        chunk_size=4096, boundary=__doc__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layers=sorted(layers), pairs=args.pairs, cases=[])
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
        generator = profile.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root, weight_index=index, model_plugin=gate.resolve_model(index.architecture or ""),
            backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
        runtime = generator.runner.runtime
        libraries = {"group": group.build_qwen35_moe_group_scatter(require_cached=True),
                     "q4": q4.build_gguf_q4_k_selected_prefill(require_cached=True),
                     "q51": q51.build_qwen4_exp_q5_1(require_cached=True)}
        report["libraries"] = {name: dict(path=lib._name,
            sha256=hashlib.sha256(Path(lib._name).read_bytes()).hexdigest()) for name, lib in libraries.items()}
        original_moe = runner_module.run_qwen4_exp_moe
        current, captures, q8_observations = {}, [], []
        keys = {"q4": KernelKey("hip_gfx1151", "moe_linear", "gguf_q4_k",
                               "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out"),
                "q51": KernelKey("hip_gfx1151", "moe_linear", "gguf_q5_1",
                                "selected_wmma_iu8_risk_prefill_bf16_bf16_out"), "q8": Q8_GUARDED_KEY}
        originals = {name: resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
                     for name, key in keys.items()}
        start, stop = runtime.event_create(), runtime.event_create()
        def download(ptr, shape, dtype):
            array = np.empty(shape, dtype=dtype)
            copy_device_to_host(host_array_ptr(array), DeviceBuffer(ptr, array.nbytes), runtime=runtime)
            return array
        def sha(array):
            return hashlib.sha256(array.tobytes()).hexdigest()
        def moe_hook(*a, **kw):
            current.update(layer=int(a[1]["expert_gate"].spec.slot_path.split(".")[1]), scratch=kw["scratch"])
            try:
                return original_moe(*a, **kw)
            finally:
                current.clear()
        def q8_hook(*a, **kw):
            if current.get("layer") in layers:
                rows, experts, k, n, total = map(int, a[6:11])
                actual = int(download(a[2] + experts * 8, (1,), np.int64)[0])
                assert actual == total
                q8_observations.append(dict(layer=current["layer"], rows=rows, launch_rows=total,
                    actual_rows=actual, already_host_sized=True))
            return originals["q8"](*a, **kw)
        def hooked(route, *a, **kw):
            layer = current.get("layer")
            if layer not in layers:
                return originals[route](*a, **kw)
            dual = route == "q4"
            rows, k, n = map(int, a[11:14] if dual else a[10:13])
            experts, padded = map(int, a[15:17] if dual else a[13:15])
            cols = n + int(a[14]) if dual else n
            stream = int(kw.get("stream", 0))
            if stream != 0 or experts != 512:
                raise ValueError("screen requires default stream and512 experts")
            static_rows = static_tile_bound(rows, experts) * 16
            assert static_rows == padded, (route, padded, static_rows)
            tiles = static_rows // 16
            capacity = rows * cols
            before = memory_stats()
            allocations = []
            samples = {arm: dict(wall_ms=[], event_ms=[]) for arm in ("static", "host")}
            ref = None
            input_before = sha(download(a[0], (rows, k), np.uint16))
            starts_before = download(a[1], (experts + 1,), np.int64)
            lanes = download(current["scratch"].group_sorted_lanes.ptr, (rows,), np.int64)
            np.testing.assert_array_equal(np.sort(lanes), np.arange(rows))
            try:
                sizes = dict(ws=(experts + 1) * 8, tiles=tiles * 8, total=8,
                             out=capacity * 2, risk_count=4, indices=capacity * 4)
                if dual:
                    sizes["silu"] = rows * n * 2
                buffers = {}
                for name, size in sizes.items():
                    buffers[name] = malloc(size, runtime=runtime); allocations.append(buffers[name])
                def p(name):
                    return buffers[name].ptr
                total_host = np.empty(1, np.int64)
                def operation(arm):
                    group.qwen35_moe_wmma_tile_map(a[1], p("ws"), p("tiles"), p("total"), experts,
                        tile_capacity=tiles, library=libraries["group"], runtime=runtime)
                    actual_rows = launch_rows(arm, runtime=runtime, stream=stream,
                        total_buffer=buffers["total"], host=total_host, compact_rows=rows, static_rows=static_rows)
                    runtime.memset(p("risk_count"), 0, 4)
                    params = list(a)
                    params[2:4] = [p("ws"), p("tiles")]
                    out_idx, count_idx = (6, 7) if dual else (5, 6)
                    params[out_idx] = p("out")
                    params[count_idx:count_idx + 3] = [p("risk_count"), p("indices"), capacity]
                    params[-1] = actual_rows
                    originals[route](*params, library=libraries[route], runtime=runtime)
                    if dual:
                        q4.gguf_q4_k_selected_dual_sparse_exact_repair_bf16(
                            a[0], a[1], a[4], a[5], p("out"), p("risk_count"), p("indices"), capacity,
                            rows, k, n, int(a[14]), experts, library=libraries["q4"], runtime=runtime)
                        silu_mul_dual_out_bf16(p("out"), p("silu"), rows, n, runtime=runtime)
                    else:
                        q51.qwen4_exp_q5_1_selected_sparse_exact_repair_row_publish_bf16(
                            a[0], a[1], a[4], p("out"), p("risk_count"), p("indices"), capacity,
                            rows, k, n, experts, library=libraries["q51"], runtime=runtime)
                    return actual_rows
                observed_rows = set()
                for pair in range(args.pairs + 1):
                    order = ("static", "host") if (pair + len(captures)) % 2 == 0 else ("host", "static")
                    for arm in order:
                        runtime.device_synchronize()
                        began = time.perf_counter()
                        runtime.event_record(start)
                        launched = operation(arm)
                        runtime.event_record(stop)
                        runtime.event_synchronize(stop)
                        wall = (time.perf_counter() - began) * 1e3
                        event = runtime.event_elapsed_time_ms(start, stop)
                        if pair:
                            samples[arm]["wall_ms"].append(wall); samples[arm]["event_ms"].append(event)
                        if arm == "host":
                            observed_rows.add(launched)
                        count = int(download(p("risk_count"), (1,), np.int32)[0])
                        assert 0 <= count <= capacity
                        indices = download(p("indices"), (count,), np.int32) if count else np.empty(0, np.int32)
                        assert np.all((indices >= 0) & (indices < capacity))
                        out = download(p("out"), (rows, cols), np.uint16)
                        assert np.isfinite((out.astype(np.uint32) << 16).view(np.float32)).all()
                        canonical = np.empty_like(out); canonical[lanes] = out
                        membership = np.sort(lanes[indices // cols] * cols + indices % cols)
                        maps = [download(p("ws"), (experts + 1,), np.int64), download(p("tiles"), (tiles,), np.int64)]
                        expected_rows = sum((int(v) + 15) // 16 for v in np.diff(starts_before)) * 16
                        assert int(maps[0][-1]) == expected_rows
                        assert np.all(maps[1][expected_rows // 16:] == -1)
                        result = dict(out=sha(canonical), risk_count=count, membership=sha(membership), maps=[sha(m) for m in maps])
                        if dual:
                            silu = download(p("silu"), (rows, n), np.uint16)
                            canonical_silu = np.empty_like(silu); canonical_silu[lanes] = silu
                            result["silu"] = sha(canonical_silu)
                        if ref is None:
                            ref = result
                        else:
                            assert result == ref, (route, layer, arm, "output/risk/map mismatch")
                assert len(observed_rows) == 1
                assert sha(download(a[0], (rows, k), np.uint16)) == input_before
                np.testing.assert_array_equal(download(a[1], (experts + 1,), np.int64), starts_before)
                medians = {arm: {metric: statistics.median(v) for metric, v in values.items()} for arm, values in samples.items()}
                captures.append(dict(layer=layer, route=route, tokens=rows // 10, compact_rows=rows,
                    static_rows=static_rows, host_rows=observed_rows.pop(),
                    active_experts=int(np.count_nonzero(np.diff(starts_before))),
                    max_expert_rows=int(np.max(np.diff(starts_before))),
                    samples=samples, medians=medians,
                    wall_speedup=medians["static"]["wall_ms"] / medians["host"]["wall_ms"],
                    exact=ref, inputs_immutable=True))
                print(layer, route, "host/static wall speedup", captures[-1]["wall_speedup"], flush=True)
            finally:
                for buffer in reversed(allocations):
                    free(buffer, runtime=runtime)
            after = memory_stats()
            for field in ("active_allocations", "current_allocated_bytes"):
                assert after[field] == before[field], (route, field, "replay leak")
            return originals[route](*a, **kw)
        try:
            runner_module.run_qwen4_exp_moe = moe_hook
            for route in ("q4", "q51"):
                def wrapper(*a, _route=route, **kw):
                    return hooked(_route, *a, **kw)
                register(keys[route], wrapper, replace=True)
            register(keys["q8"], q8_hook, replace=True)
            for case in cases:
                captures.clear(); q8_observations.clear()
                first = generator.runner.prefill(case["prompt_token_ids"])
                expected = {(layer, "q4") for layer in layers} | {(layer, "q51") for layer in layers if layer not in (4, 30, 46, 47)}
                assert {(c["layer"], c["route"]) for c in captures} == expected
                assert len(captures) == len(expected)
                assert np.isfinite(first.logits).all()
                report["cases"].append(dict(id=case["id"], category=case["category"], prompt_tokens=case["prompt_tokens"],
                    captures=list(captures), q8_observations=list(q8_observations), first_token=int(first.token_id)))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"; report["error"] = repr(error)
            raise
        finally:
            runner_module.run_qwen4_exp_moe = original_moe
            for route, key in keys.items():
                register(key, originals[route], replace=True)
            runtime.event_destroy(start); runtime.event_destroy(stop)
            generator.close()
            report["memory_after_close"] = memory_stats()
            assert report["memory_after_close"]["active_allocations"] == 0
            assert report["memory_after_close"]["current_allocated_bytes"] == 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
