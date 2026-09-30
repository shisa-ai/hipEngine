#!/usr/bin/env python3
"""RT-4: locate the highest KV byte actually written, and compare arms.

The bounded reset is claimed exact because every byte at or above
``written_cache_bytes()`` is already zero.  Two things could break that:
the bound could be too SMALL (real data above it), or the bytes above the
bound could be non-zero for some other reason.

This measures the highest non-zero byte in each K/V cache after a real request
and prints it beside the bound.  ``last_nonzero < bound`` at every attention
state is the invariant holding; ``last_nonzero >= bound`` is the bug, and the
gap says how far off the bound is.

Run it twice - once with ``HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET=0`` and once
with ``=1`` - and diff the reported hashes.  Identical hashes mean the change
is byte-exact; the first differing attention index localizes the problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.uint8)).hexdigest()


def _read(buffer, runtime) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host

    # Synchronize BEFORE the copy, not after.  A D2H copy issued while the last
    # decode step is still in flight captures a partially-written buffer, which
    # shows up as a spuriously lower non-zero count and a different hash with no
    # arithmetic change.  This exact race produced a phantom arm difference in
    # the first version of this probe.
    runtime.device_synchronize()
    out = np.empty(buffer.nbytes, dtype=np.uint8)
    copy_device_to_host(out.ctypes.data, buffer, buffer.nbytes, runtime=runtime)
    runtime.device_synchronize()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-root", type=pathlib.Path, required=True)
    ap.add_argument("--fixture", type=pathlib.Path,
                    default=pathlib.Path(
                        "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json"))
    ap.add_argument("--case", default="code-p1024")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--requests", type=int, default=2,
                    help="Number of prefill+decode requests through one runner.  "
                         "The second is the one the bounded clear actually affects.")
    ap.add_argument("--capacity", type=int, default=None)
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    ap.add_argument("--dump-arrays", type=pathlib.Path, default=None,
                    help="Directory to save raw K/V buffers as .npy for byte-level diffing.")
    ap.add_argument("--tag", default=None,
                    help="Override the arm label recorded in the artifact, so the same flag can "
                         "be run twice under distinct names (determinism control).")
    args = ap.parse_args(argv)

    from dataclasses import replace as _replace

    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models.registry import resolve_model

    fixture = json.loads(args.fixture.read_text())
    cases = {str(row["id"]): row for row in fixture["cases"]}
    if args.case not in cases:
        raise SystemExit(f"unknown case {args.case!r}; have {sorted(cases)}")
    case = cases[args.case]
    token_ids = [int(t) for t in case["prompt_token_ids"]]
    prompt_tokens = int(case["prompt_tokens"])

    max_sequence_length = prompt_tokens + args.steps + 8
    if args.capacity is not None:
        max_sequence_length = int(args.capacity)

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    model_root = args.model_root.resolve()
    index = load_gguf_index(discover_gguf_files(model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile("production"),
    )

    def factory():
        return Qwen4ExpGGUFTextGenerator(
            model_path=model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=512,
        )

    flag = os.environ.get("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", "0")
    arm = args.tag or f"flag{flag}"
    print(f"[config] case={args.case} prompt_tokens={prompt_tokens} steps={args.steps} "
          f"requests={args.requests} capacity={max_sequence_length} "
          f"BOUNDED_KV_RESET={flag} arm={arm}", flush=True)

    generator = resolved.construct_generator(factory)
    record: dict = {
        "case": args.case,
        "arm": arm,
        "bounded_kv_reset": flag,
        "capacity": max_sequence_length,
        "prompt_tokens": prompt_tokens,
        "steps": args.steps,
        "requests": args.requests,
        "attention": [],
    }
    try:
        runner = generator.runner
        runtime = runner.runtime
        current = None
        for request in range(args.requests):
            result = runner.prefill(token_ids, capture_logits=False,
                                    capture_target_hidden=False)
            current = int(result.token_id)
            for _ in range(args.steps):
                result = runner.step(current, capture_logits=False,
                                     capture_target_hidden=False,
                                     token_id_resident=True)
                current = int(result.token_id)
            print(f"[request {request}] done, token={current}", flush=True)

        for i, attention in enumerate(runner.attention_states):
            key = _read(attention.key_cache, runtime)
            value = _read(attention.value_cache, runtime)
            bound = int(attention.written_cache_bytes())
            k_nz = np.nonzero(key)[0]
            v_nz = np.nonzero(value)[0]
            entry = {
                "index": i,
                "position": int(attention.position_host[0]),
                "bound": bound,
                "key_nbytes": int(key.nbytes),
                "key_last_nonzero": int(k_nz[-1]) if k_nz.size else -1,
                "key_nonzero": int(k_nz.size),
                "key_sha256": _sha(key),
                "value_last_nonzero": int(v_nz[-1]) if v_nz.size else -1,
                "value_nonzero": int(v_nz.size),
                "value_sha256": _sha(value),
            }
            if args.dump_arrays is not None:
                # Persist the raw bytes so two runs can be diffed offline - the
                # first differing offset is what says whether a difference is
                # above or below the bound.
                args.dump_arrays.mkdir(parents=True, exist_ok=True)
                np.save(args.dump_arrays / f"{arm}-att{i:02d}-key.npy", key)
                np.save(args.dump_arrays / f"{arm}-att{i:02d}-value.npy", value)
            # The invariant: nothing is written at or above the bound.
            entry["invariant_holds"] = bool(
                (entry["key_last_nonzero"] < bound) and (entry["value_last_nonzero"] < bound)
            )
            record["attention"].append(entry)
            print(f"[attention {i:2d}] pos={entry['position']:6d} bound={bound:10d} "
                  f"key_last_nz={entry['key_last_nonzero']:10d} "
                  f"value_last_nz={entry['value_last_nonzero']:10d} "
                  f"key_nz={entry['key_nonzero']:9d} value_nz={entry['value_nonzero']:9d} "
                  f"invariant={'OK' if entry['invariant_holds'] else 'VIOLATED'}",
                  flush=True)

        violations = [e["index"] for e in record["attention"] if not e["invariant_holds"]]
        record["invariant_violations"] = violations
        print(f"[summary] invariant violations at attention indices: {violations}", flush=True)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(record, indent=2) + "\n")
            print(f"[json] wrote {args.json_out}", flush=True)
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
