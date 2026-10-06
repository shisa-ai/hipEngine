#!/usr/bin/env python3
"""Pair padded-tail GEMMs on every intercepted natural model activation.

Named production composition is unchanged except for a diagnostic leaf hook.
The hook runs a warm pair and two counterbalanced measured pairs, compares all
output bytes and input bytes, and leaves the parity-checked output for the engine.
Only HIP-event GEMM intervals are timing evidence, not request wall.
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

from hipengine.core.memory import memory_stats
from hipengine.kernels.registry import KernelKey, register, resolve
from scripts import qwen4exp_row4_state_gate as gate
from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

from scripts.qwen4exp_dense_wide_padded_gate_support import PARENT, CANDIDATE, admitted_shape


def main():
    from tests.test_gpu_qwen4_exp_gdn_hip import _download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile = gate.resolve_runtime_profile(
        model=gate.QWEN4_EXP_MODEL, backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1], profile=gate.ExecutionProfile.PRODUCTION)
    fixture, digest = gate.load_fixture(gate.DEFAULT_FIXTURE)
    ids = args.case_id or ["code-p512", "code-p1024", "code-p4096",
                          "general_en-p4096", "general_ja-p4096", "mixed_ja_en-p4096"]
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in ids]
    index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
    generator = profile.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
        model_path=args.model_root, weight_index=index,
        model_plugin=gate.resolve_model(index.architecture or ""),
        backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
    # Keep both diagnostic launches inside the parent hook after promotion.
    from hipengine.runtime.gguf_linear import DENSE_WIDE_TAIL_VARIANT_ENV
    os.environ[DENSE_WIDE_TAIL_VARIANT_ENV] = PARENT
    runtime = generator.runner.runtime
    key = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", PARENT)
    parent = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=PARENT)
    candidate = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=CANDIDATE)
    start, stop = runtime.event_create(), runtime.event_create()
    report = dict(source=gate._git_metadata(ROOT), host=gate._host_metadata(),
                  command=sys.argv, model_identity=model_identity(args.model_root),
                  fixture_sha256=digest, manifest_sha256=profile.manifest_sha256,
                  manifest_scope="named profile with diagnostic leaf hook; no default change",
                  parent_variant=PARENT, candidate_variant=CANDIDATE,
                  chunk_size=4096, cases=[], performance_claim=False, status="running")
    captures = []

    def download(ptr, shape, dtype):
        return _download(SimpleNamespace(ptr=ptr, nbytes=int(np.prod(shape)) * np.dtype(dtype).itemsize),
                         shape, dtype, runtime)

    def hooked(*a, **kw):
        rows, k, n = map(int, a[3:6])
        if not admitted_shape(rows, k, n):
            return parent(*a, **kw)
        if int(kw.get("stream", 0)) != 0:
            raise ValueError("natural owner hook requires default stream")
        runtime.device_synchronize()
        x_before = hashlib.sha256(download(a[0], (rows, k), np.float16)).hexdigest()
        w_before = hashlib.sha256(download(a[1], (n, k // 32 * 34), np.uint8)).hexdigest()
        samples = [[], []]
        hashes = [set(), set()]
        for rep in range(3):
            for arm in ((0, 1) if (rep + len(captures)) % 2 == 0 else (1, 0)):
                runtime.event_record(start)
                (parent if arm == 0 else candidate)(*a, **kw)
                runtime.event_record(stop)
                runtime.event_synchronize(stop)
                if rep:
                    samples[arm].append(runtime.event_elapsed_time_ms(start, stop))
                out = download(a[2], (rows, n), np.float32)
                if not np.isfinite(out).all():
                    raise AssertionError("nonfinite natural output")
                hashes[arm].add(hashlib.sha256(out).hexdigest())
        if len(hashes[0] | hashes[1]) != 1:
            raise AssertionError("natural owner parity/repeat mismatch; numerical review required")
        assert hashlib.sha256(download(a[0], (rows, k), np.float16)).hexdigest() == x_before
        assert hashlib.sha256(download(a[1], (n, k // 32 * 34), np.uint8)).hexdigest() == w_before
        captures.append(dict(rows=rows, k=k, n=n, samples_ms=samples,
                             output_sha256=next(iter(hashes[0])), output_exact=True,
                             input_immutable=True))

    try:
        register(key, hooked, replace=True)
        for case in cases:
            captures.clear()
            token = generator.runner.prefill(case["prompt_token_ids"])
            if len(captures) != 70:
                raise AssertionError((case["id"], "expected70 natural HC tail calls", len(captures)))
            totals = [sum(statistics.mean(c["samples_ms"][i]) for c in captures) for i in (0, 1)]
            row = dict(id=case["id"], calls=len(captures), first_token=token.token_id,
                       parent_ms=totals[0], candidate_ms=totals[1], speedup=totals[0] / totals[1],
                       captures=list(captures))
            report["cases"].append(row)
            print(case["id"], totals, row["speedup"], flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        register(key, parent, replace=True)
        runtime.event_destroy(start)
        runtime.event_destroy(stop)
        generator.close()
        report["memory_after_close"] = memory_stats()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
