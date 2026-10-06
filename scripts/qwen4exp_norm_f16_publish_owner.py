#!/usr/bin/env python3
"""Pair complete norm/cast/HC-down on each natural production input.

Only layers13-47 HC-down sources are intercepted, by resident norm pointer
identity. Both arms use the promoted padded HC-down leaf; the named engine is
otherwise unchanged. This is a diagnostic owner screen, not a model gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hipengine.core.memory import free, malloc, memory_stats
from hipengine.kernels.hip_gfx1100.fused import qwen4_exp_gr as gr
from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
from hipengine.runtime import qwen4_exp_runner as runner
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
from tests.test_gpu_qwen4_exp_gr_hip import _download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    # Prebuild outside profiling; the owner launchers below then require cache.
    norm_lib = gr.build_qwen4_exp_gr(load=True)
    wide_lib = wide.build_gguf_q8_0_dense_wide(load=True, require_cached=True)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile = gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL, backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1], profile=gate.ExecutionProfile.PRODUCTION)
    fixture, digest = gate.load_fixture(gate.DEFAULT_FIXTURE)
    ids = args.case_id or ["code-p512", "code-p1024", "code-p4096",
                          "general_en-p4096", "general_ja-p4096", "mixed_ja_en-p4096"]
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in ids]
    index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=gate.resolve_model(index.architecture or ""), backend="hip_gfx1151",
        max_sequence_length=4352, prefill_chunk_size=4096))
    runtime = generator.runner.runtime
    mappings = {}
    for layer in range(13, 48):
        for part in ("attn", "ffn"):
            prefix = f"layers.{layer}.hc_{part}"
            wp = generator.runner.resident.weight(prefix + "_norm").allocation("raw").tensor.ptr
            down = generator.runner.resident.weight(prefix + "_down")
            mappings[int(wp)] = (prefix, int(down.allocation("raw").tensor.ptr))
    half = malloc(4096 * 10240 * 2, runtime=runtime)
    output = malloc(4096 * 320 * 4, runtime=runtime)
    start, stop = runtime.event_create(), runtime.event_create()
    parent = runner.qwen4_exp_grouped_rmsnorm_bf16_f32
    captures = []
    report = dict(source=gate._git_metadata(ROOT), host=gate._host_metadata(), command=sys.argv,
                  model_identity=model_identity(args.model_root), fixture_sha256=digest,
                  manifest_sha256=profile.manifest_sha256,
                  scope="named production;diagnostic complete-owner hook,not native runtime handoff",
                  boundary="HIP events: norm+cast+current padded HC-down vs dual norm+same HC-down",
                  performance_claim=False, status="running", cases=[])

    def download(ptr, shape, dtype):
        return _download(SimpleNamespace(ptr=ptr, nbytes=int(np.prod(shape))*np.dtype(dtype).itemsize),
                         shape, dtype, runtime)

    def hooked(x, gamma, norm, rows, branches, hidden, eps=1e-6, **kw):
        mapping = mappings.get(int(gamma))
        if mapping is None or branches != 4 or hidden != 2560 or not 512 <= rows <= 4096:
            return parent(x, gamma, norm, rows, branches, hidden, eps, **kw)
        if int(kw.get("stream", 0)) != 0:
            raise ValueError("natural diagnostic requires default stream")
        role, wptr = mapping
        runtime.device_synchronize()
        inputs = [download(x, (rows, 10240), np.uint16),
                  download(gamma, (10240,), np.float32)]
        input_hash = hashlib.sha256(b"".join(a.tobytes() for a in inputs)).hexdigest()
        samples, hashes = [[], []], [set(), set()]
        for rep in range(3):
            for arm in ((0, 1) if (rep + len(captures)) % 2 == 0 else (1, 0)):
                runtime.event_record(start)
                if arm:
                    gr.qwen4_exp_grouped_rmsnorm_bf16_f32_f16(x, gamma, norm, half.ptr,
                        rows, branches, hidden, eps, library=norm_lib, runtime=runtime)
                else:
                    parent(x, gamma, norm, rows, branches, hidden, eps,
                           library=norm_lib, runtime=runtime)
                    wide.f32_to_f16(norm, half.ptr, rows * 10240, library=wide_lib, runtime=runtime)
                wide._WRAPPERS["dense_wide256_padded_f16in_f32_f32_out"](
                    half.ptr, wptr, output.ptr, rows, 10240, 320, library=wide_lib, runtime=runtime)
                runtime.event_record(stop)
                runtime.event_synchronize(stop)
                if rep:
                    samples[arm].append(runtime.event_elapsed_time_ms(start, stop))
                parts = [download(norm, (rows, 10240), np.float32),
                         download(half.ptr, (rows, 10240), np.float16),
                         download(output.ptr, (rows, 320), np.float32)]
                if not all(np.isfinite(p).all() for p in parts):
                    raise AssertionError("nonfinite natural publication")
                hashes[arm].add(hashlib.sha256(b"".join(p.tobytes() for p in parts)).hexdigest())
        if len(hashes[0] | hashes[1]) != 1:
            raise AssertionError("natural publication/complete-owner mismatch;numerical review required")
        after = [download(x, (rows, 10240), np.uint16), download(gamma, (10240,), np.float32)]
        if hashlib.sha256(b"".join(p.tobytes() for p in after)).hexdigest() != input_hash:
            raise AssertionError("natural input changed")
        captures.append(dict(role=role, rows=rows, k=10240, n=320, samples_ms=samples,
                             output_sha256=next(iter(hashes[0])), output_exact=True, input_immutable=True))

    try:
        runner.qwen4_exp_grouped_rmsnorm_bf16_f32 = hooked
        for case in cases:
            captures.clear()
            token = generator.runner.prefill(case["prompt_token_ids"])
            if len(captures) != 70:
                raise AssertionError((case["id"], "expected70HC-down calls", len(captures)))
            totals = [sum(statistics.mean(c["samples_ms"][i]) for c in captures) for i in (0, 1)]
            row = dict(id=case["id"], calls=len(captures), first_token=token.token_id,
                       parent_ms=totals[0], candidate_ms=totals[1], speedup=totals[0]/totals[1],
                       captures=list(captures))
            report["cases"].append(row)
            print(case["id"], totals, row["speedup"], flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        runner.qwen4_exp_grouped_rmsnorm_bf16_f32 = parent
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        free(half, runtime=runtime)
        free(output, runtime=runtime)
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
