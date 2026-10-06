#!/usr/bin/env python3
"""Replay adjacent natural Q8 consumers with two casts versus one shared cast.

The default prebound primitive replay excludes dispatch and native ownership.
With --native-owner, the real runner pair helper charges scope admission,
both normal dispatches, casts/GEMMs, scope exit, events and final device sync.
Both modes validate native outputs and immutable inputs outside timing. Neither
mode changes production defaults or introduces a persistent activation cache.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

PAIRS = (("attn_qkv", "attn_gate"), ("shared_gate", "shared_up"))


@dataclass(frozen=True)
class Consumer:
    layer: int
    role: str
    rows: int
    k: int
    n: int
    x: int
    half: int
    out: int
    stream: int
    runtime: object


def checked_pair(a: Consumer, b: Consumer) -> str:
    if (a.role, b.role) not in PAIRS:
        raise ValueError("unexpected consumer order")
    if ((a.layer, a.rows, a.k, a.x, a.half, a.stream) !=
            (b.layer, b.rows, b.k, b.x, b.half, b.stream)
            or a.runtime is not b.runtime):
        raise ValueError("consumer source/owner identity mismatch")
    ranges = [(a.x, a.rows * a.k * 4), (a.half, a.rows * a.k * 2),
              (a.out, a.rows * a.n * 4), (b.out, b.rows * b.n * 4)]
    if any(ptr <= 0 or size <= 0 for ptr, size in ranges):
        raise ValueError("invalid pointer or extent")
    for i, (ptr, size) in enumerate(ranges):
        for other, count in ranges[i + 1:]:
            if ptr < other + count and other < ptr + size:
                raise ValueError("aliased activation/stage/output")
    return a.role + "+" + b.role


def paired_operation(arm, cast, first, second):
    if arm not in {"two_casts", "one_cast"}:
        raise ValueError("unknown cast-sharing arm")
    cast()
    first()
    if arm == "two_casts":
        cast()
    second()


def native_pair_operation(helper, left, right, outputs):
    """Invoke the runner's ordinary pair helper, not prebound kernel callables."""
    a, b = left[0], right[0]
    helper(left[1], right[1], a.x, outputs[0], outputs[1], a.rows, a.k, a.n, b.n,
           stream=a.stream, runtime=a.runtime)


def replay_operation(arm, native_owner, cast, first, second, native):
    if arm not in {"two_casts", "one_cast"}:
        raise ValueError("unknown cast-sharing arm")
    if native_owner:
        native()
    else:
        paired_operation(arm, cast, first, second)


def measure_operation(runtime, start, stop, operation, *, final_device_sync=False):
    """Charge the complete operation and completion; exclude the pre-run wait."""
    runtime.device_synchronize()
    began = time.perf_counter()
    runtime.event_record(start)
    operation()
    runtime.event_record(stop)
    runtime.event_synchronize(stop)
    if final_device_sync:
        runtime.device_synchronize()
    wall = (time.perf_counter() - began) * 1e3
    return wall, runtime.event_elapsed_time_ms(start, stop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--native-owner", action="store_true",
                        help="charge the actual runner pair helper and ownership admission")
    args = parser.parse_args()
    if args.pairs < 2 or args.pairs % 2:
        parser.error("an even number of at least two measured pairs is required")
    layers = set(args.layer or [13, 24, 46])
    if not layers <= (set(range(13, 48)) - set(range(15, 48, 4))):
        parser.error("select GDN layers within production wide scope13..47")
    from hipengine.benchmark.provenance import collect_artifact_provenance
    from hipengine.core.memory import (
        DeviceBuffer, copy_device_to_host, free, host_array_ptr, malloc, memory_stats,
    )
    from hipengine.kernels.hip_gfx1100.quant import gguf_q8_0_dense_wide as wide
    from hipengine.kernels.registry import KernelKey, resolve
    from hipengine.runtime import gguf_linear as linear, qwen4_exp_runner as runner
    from scripts import qwen4exp_row4_state_gate as gate
    from scripts.qwen4exp_framework_family_refresh import check_host, model_identity

    check_host()
    os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file.resolve())
    os.environ["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
    os.environ[linear.WIDE_F16_PAIR_ENV] = "0"
    gate.register_gfx1151_kernels(replace=True)
    gate.register_qwen4_exp_gfx1151_profiles()
    profile = gate.resolve_runtime_profile(model=gate.QWEN4_EXP_MODEL, backend=gate.QWEN4_EXP_BACKEND,
        quant=gate.QWEN4_EXP_QUANTS[1], profile=gate.ExecutionProfile.PRODUCTION)
    fixture, digest = gate.load_fixture(gate.DEFAULT_FIXTURE)
    ids = args.case_id or ([f"{category}-p{rows}" for rows in (1024, 4096)
                            for category in ("code", "general_en", "general_ja", "mixed_ja_en")]
                           if args.native_owner else
                           ["code-p512", "code-p1024", "code-p4096", "general_en-p4096",
                            "general_ja-p4096", "mixed_ja_en-p4096"])
    cases = [next(c for c in fixture["cases"] if c["id"] == name) for name in ids]
    if args.native_owner and any(c["prompt_tokens"] not in {1024, 4096} for c in cases):
        parser.error("native ownership screen admits only measured rows1024/4096")
    report = dict(schema=1, status="running", performance_claim=False, command=sys.argv,
        source=gate._git_metadata(ROOT), host=gate._host_metadata(), model_identity=model_identity(args.model_root),
        execution_profile="production", manifest_sha256=profile.manifest_sha256,
        fixture_sha256=digest, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        boundary=("actual runner pair helper: scope admission, both normal dispatches, "
                  "cast/GEMMs, scope exit, events and final device sync; allocation, "
                  "validation and surrounding model excluded" if args.native_owner else
                  "prebound casts/two GEMMs, events and end sync; allocation, validation, "
                  "dispatch selection, native ownership and surrounding model excluded"),
        native_owner=args.native_owner, chunk_size=4096, layers=sorted(layers), pairs=args.pairs, cases=[])
    report["hipengine_artifact_provenance"] = collect_artifact_provenance(
        repo_root=ROOT, configured_backend="hip_gfx1151", resolved_backend="hip_gfx1151",
        target_arch="gfx1151", model_path=args.model_root,
        model_revision=report["model_identity"]["revision"], quant="UD-Q4_K_XL", kv_dtype="bf16",
        command=sys.argv, build_profile="cached production",
        timing_protocol=report["boundary"] + "; alternating arms",
        warmups=1, repetitions=args.pairs,
    )
    with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = gate.load_gguf_index(gate.discover_gguf_files(args.model_root)[0])
        generator = profile.construct_generator(lambda: gate.Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root, weight_index=index, model_plugin=gate.resolve_model(index.architecture or ""),
            backend="hip_gfx1151", max_sequence_length=4352, prefill_chunk_size=4096))
        runtime = generator.runner.runtime
        library = wide.build_gguf_q8_0_dense_wide(require_cached=True)
        report["cast_library"] = dict(path=library._name,
            sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest())
        original = linear._launch_dense_wide_f16_staged
        start, stop = runtime.event_create(), runtime.event_create()
        pending, captures = [], []
        replaying = False

        def download(ptr, shape, dtype):
            out = np.empty(shape, dtype=dtype)
            copy_device_to_host(host_array_ptr(out), DeviceBuffer(ptr, out.nbytes), runtime=runtime)
            return out

        def sha(array):
            return hashlib.sha256(array.tobytes()).hexdigest()

        def replay(left, right):
            nonlocal replaying
            replaying = True
            prior_pair_env = os.environ.get(linear.WIDE_F16_PAIR_ENV)
            a, b = left[0], right[0]
            role = checked_pair(a, b)
            before = memory_stats()
            allocations = []
            inputs = download(a.x, (a.rows, a.k), np.float32)
            if not np.isfinite(inputs).all() or not np.isfinite(inputs.astype(np.float16)).all():
                raise AssertionError("nonfinite F16 staging input")
            input_hash = sha(inputs)
            weights_before = [sha(download(info[1].allocation("raw").tensor.ptr,
                (c.n * (c.k // 32) * 34,), np.uint8)) for c, info in ((a, left), (b, right))]
            native_outputs = [sha(download(c.out, (c.rows, c.n), np.float32)) for c in (a, b)]
            samples = {arm: [] for arm in ("two_casts", "one_cast")}
            try:
                if args.native_owner:
                    half = DeviceBuffer(a.half, a.rows * a.k * 2)
                else:
                    half = malloc(a.rows * a.k * 2, runtime=runtime); allocations.append(half)
                outs = []
                for c in (a, b):
                    out = malloc(c.rows * c.n * 4, runtime=runtime)
                    allocations.append(out); outs.append(out)
                checked_pair(Consumer(**{**a.__dict__, "half": half.ptr, "out": outs[0].ptr}),
                             Consumer(**{**b.__dict__, "half": half.ptr, "out": outs[1].ptr}))

                def cast():
                    wide.f32_to_f16(a.x, half.ptr, a.rows * a.k,
                                   library=library, runtime=runtime, stream=a.stream)

                def consumer(info, out):
                    c, weight, fn, kw, _ = info
                    linear._LAUNCH_ABI["raw"](fn, weight, half.ptr, out.ptr,
                                               c.rows, c.k, c.n, kw)

                output_ptrs = tuple(out.ptr for out in outs)

                def native():
                    native_pair_operation(runner._launch_qwen4_exp_dense_pair, left, right,
                                          output_ptrs)

                for pair in range(args.pairs + 1):
                    order = (("two_casts", "one_cast") if (pair + len(captures)) % 2 == 0
                             else ("one_cast", "two_casts"))
                    for position, arm in enumerate(order):
                        if args.native_owner:
                            os.environ[linear.WIDE_F16_PAIR_ENV] = "1" if arm == "one_cast" else "0"
                        wall, event = measure_operation(runtime, start, stop,
                            lambda: replay_operation(arm, args.native_owner, cast,
                                lambda: consumer(left, outs[0]),
                                lambda: consumer(right, outs[1]), native),
                            final_device_sync=args.native_owner)
                        if pair:
                            samples[arm].append(dict(pair=pair, position=position,
                                first_arm=order[0], wall_ms=wall, event_ms=event))
                        for c, out, expected in zip((a, b), outs, native_outputs):
                            actual = download(out.ptr, (c.rows, c.n), np.float32)
                            if not np.isfinite(actual).all() or sha(actual) != expected:
                                raise AssertionError((role, arm, "native output/repeat mismatch"))
                        staged = download(half.ptr, (a.rows, a.k), np.float16)
                        if sha(staged) != sha(inputs.astype(np.float16)):
                            raise AssertionError("F16 staging bytes differ from CPU conversion")
                engagement = {}
                if args.native_owner:
                    import contextlib
                    original_scope, original_cast = runner.wide_f16_activation_pair, wide.f32_to_f16
                    counts = dict(pairs=0, accepted=0, casts=0)

                    @contextlib.contextmanager
                    def counted_scope(*args, **kwargs):
                        with original_scope(*args, **kwargs) as loan:
                            yield loan
                            if loan is not None:
                                counts["pairs"] += 1
                                counts["accepted"] += loan.accepted

                    def counted_cast(*args, **kwargs):
                        counts["casts"] += 1
                        return original_cast(*args, **kwargs)

                    try:
                        runner.wide_f16_activation_pair, wide.f32_to_f16 = counted_scope, counted_cast
                        for arm in samples:
                            for field in counts:
                                counts[field] = 0
                            os.environ[linear.WIDE_F16_PAIR_ENV] = "1" if arm == "one_cast" else "0"
                            native()
                            runtime.device_synchronize()
                            expected = dict(pairs=1, accepted=2, casts=1) if arm == "one_cast" else dict(pairs=0, accepted=0, casts=2)
                            if counts != expected:
                                raise AssertionError((arm, "native engagement", counts, expected))
                            engagement[arm] = dict(counts)
                            for c, out, expected_hash in zip((a, b), outs, native_outputs):
                                if sha(download(out.ptr, (c.rows, c.n), np.float32)) != expected_hash:
                                    raise AssertionError("census output differs from native")
                    finally:
                        runner.wide_f16_activation_pair, wide.f32_to_f16 = original_scope, original_cast
                if sha(download(a.x, (a.rows, a.k), np.float32)) != input_hash:
                    raise AssertionError("source activation changed")
                for c, info, expected in zip((a, b), (left, right), weights_before):
                    assert sha(download(info[1].allocation("raw").tensor.ptr,
                        (c.n * (c.k // 32) * 34,), np.uint8)) == expected, "weight changed"
                medians = {arm: {metric: statistics.median(s[metric] for s in values)
                    for metric in ("wall_ms", "event_ms")} for arm, values in samples.items()}
                strata = {first: {metric: statistics.median(s[metric] for s in samples["two_casts"]
                    if s["first_arm"] == first) / statistics.median(s[metric] for s in samples["one_cast"]
                    if s["first_arm"] == first) for metric in ("wall_ms", "event_ms")}
                    for first in samples}
                captures.append(dict(layer=a.layer, role=role, rows=a.rows, k=a.k, n=[a.n, b.n],
                    variants=[left[4], right[4]], samples=samples, medians=medians,
                    wall_speedup=medians["two_casts"]["wall_ms"] / medians["one_cast"]["wall_ms"],
                    order_strata_speedup=strata, native_output_sha256=native_outputs, input_sha256=input_hash,
                    native_engagement=engagement, weight_sha256=weights_before, exact_native_outputs=True,
                    exact_f16_staging=True, source_weights_immutable=True))
                print(a.layer, role, "one/two cast wall speedup", captures[-1]["wall_speedup"], flush=True)
            finally:
                replaying = False
                if prior_pair_env is None:
                    os.environ.pop(linear.WIDE_F16_PAIR_ENV, None)
                else:
                    os.environ[linear.WIDE_F16_PAIR_ENV] = prior_pair_env
                for allocation in reversed(allocations):
                    free(allocation, runtime=runtime)
            after = memory_stats()
            for field in ("active_allocations", "current_allocated_bytes"):
                assert after[field] == before[field], (role, field, "replay leak")

        def hooked(fn, weight, x, out, rows, k, n, kwargs, **kw):
            result = original(fn, weight, x, out, rows, k, n, kwargs, **kw)
            if replaying:
                return result
            slot = str(weight.spec.slot_path).split(".")
            role = slot[-1]
            layer = int(slot[1]) if len(slot) > 2 and slot[1].isdigit() else -1
            if not result or layer not in layers or role not in {r for p in PAIRS for r in p}:
                pending.clear()
                return result
            if kw["quant"] != "gguf_q8_0" or rows not in {512, 1024, 4096}:
                raise ValueError("unexpected qualifying quant/shape")
            stage = linear._wide_f16_activation_stage_ptr(rows * k)
            target = linear._wide_f16_activation_target_variant(kw["variant"])
            key = linear._dense_wide_f16_tail_key(KernelKey(kw["backend"], kw["layer"], kw["quant"], target),
                                                rows=rows, in_features=k, out_features=n)
            sibling = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
            launch_kw = dict(kwargs)
            if kw.get("libraries") is not None:
                scoped = linear._variant_scoped_library(kw["libraries"], key)
                if scoped is not None:
                    launch_kw["library"] = scoped
            c = Consumer(layer, role, rows, k, n, x, stage, out, int(kwargs.get("stream", 0)), kw["runtime"])
            if c.stream != 0 or c.runtime is not runtime:
                raise ValueError("screen requires runner runtime and default stream")
            info = (c, weight, sibling, launch_kw, key.variant)
            if role in {p[0] for p in PAIRS}:
                if pending:
                    raise AssertionError("missing immediate second consumer")
                pending.append(info)
            else:
                if not pending:
                    raise AssertionError("second consumer without immediate first")
                replay(pending.pop(), info)
            return result

        try:
            linear._launch_dense_wide_f16_staged = hooked
            for case in cases:
                captures.clear(); pending.clear()
                token = generator.runner.prefill(case["prompt_token_ids"])
                assert not pending and len(captures) == 2 * len(layers), (case["id"], len(captures))
                totals = {arm: sum(c["medians"][arm]["wall_ms"] for c in captures)
                          for arm in ("two_casts", "one_cast")}
                report["cases"].append(dict(id=case["id"], first_token=token.token_id,
                    diagnostic_sum_wall_medians_ms=totals, captures=list(captures)))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"; report["error"] = repr(error)
            raise
        finally:
            linear._launch_dense_wide_f16_staged = original
            runtime.event_destroy(start); runtime.event_destroy(stop)
            generator.close()
            report["memory_after_close"] = memory_stats()
            if (report["memory_after_close"]["active_allocations"] != 0
                    or report["memory_after_close"]["current_allocated_bytes"] != 0):
                report["status"] = "failed"
                report["error"] = "nonzero final allocation/byte count"
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert report["status"] == "passed", report.get("error")


if __name__ == "__main__":
    main()
