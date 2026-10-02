#!/usr/bin/env python3
"""DE-5 re-derivation: split the clean decode wall into GPU-busy and host-stall time.

DE-5 records "serving-wrapper host time, ~3.6 ms/token above the runner loop" and
is marked "OPEN, re-derive".  The figure has never been re-derived, and DE-1's own
finding is the reason it must be: in the profiled harness the kernel sum is stable
to 1.0% while the step span varies 2.8x, so no gap figure taken from that harness
is an engine property.

This probe measures the split directly instead of inferring it:

* it runs the same decode protocol as ``qwen4exp_canonical_ar_bench.py`` -- the
  production profile, ``runner.prefill`` then exactly N ``runner.step`` calls,
  with no profiler attached; and
* a concurrent sampler reads the device's own ``gpu_busy_percent`` counter, which
  reports whether the GPU had work, so a gap between the wall and the kernel sum
  can be attributed rather than assumed.

The distinction matters.  A HIP event pair around the loop measures the *stream*
timeline and therefore includes idle gaps, so it cannot separate the two cases.
The device counter can: if the GPU is busy essentially all of the wall, then the
wall is kernel time and any shortfall against a profiled kernel sum is the
profiler's, not the host's.  If it is busy only part of the wall, the remainder is
real host-side stall and is DE-5's target.

Usage:
    python3 scripts/qwen4exp_de5_wall_split.py [--case code-p1024] [--steps 128]
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import os
import pathlib
import pstats
import statistics
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = pathlib.Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_FIXTURE = ROOT / "benchmarks" / "fixtures" / "qwen4exp_canonical_ar_p512_p1024_p4096.json"
BUSY_PATH = pathlib.Path("/sys/class/drm/card1/device/gpu_busy_percent")
SCLK_PATH = pathlib.Path("/sys/class/drm/card1/device/pp_dpm_sclk")
MCLK_PATH = pathlib.Path("/sys/class/drm/card1/device/pp_dpm_mclk")
CPU_AVG_FREQ_PATH = pathlib.Path(
    "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_avg_freq"
)
CPU_CORE_FREQ_TEMPLATE = "/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq"
CPU_CORE_COUNT = os.cpu_count() or 1


class BusySampler:
    """Poll the device's own busy counter and DPM clocks while the decode loop runs.

    The clocks matter as much as the busy share here: a step that is 3x slower at
    99% busy is either doing 3x the work or running at a third of the clock, and
    those have completely different fixes.  ``pp_dpm_sclk`` marks the active level
    with a trailing ``*``.
    """

    def __init__(self, path: pathlib.Path, interval_s: float = 0.005) -> None:
        self._path = path
        self._interval = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.sclk: list[float] = []
        self.mclk: list[float] = []
        self.errors = 0
        self.cpu_avg: list[float] = []
        self.cpu_max: list[float] = []
        self.cpu_mean: list[float] = []
        self._tick = 0

    @staticmethod
    def _active_mhz(path: pathlib.Path) -> float:
        text = path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if line.endswith("*"):
                token = line.split(":", 1)[1].strip().rstrip("*").strip()
                return float(token.lower().replace("mhz", ""))
        return float("nan")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(int(self._path.read_text().strip()))
            except Exception:
                self.errors += 1
            try:
                self.sclk.append(self._active_mhz(SCLK_PATH))
                self.mclk.append(self._active_mhz(MCLK_PATH))
            except Exception:
                self.errors += 1
            # Host CPU clocks.  The GPU clocks above are flat in this harness, but
            # the host thread is the bottleneck, so a package power limit shared
            # with the GPU would show up here and nowhere else.  ``cpuinfo_avg_freq``
            # is cpu0's own hardware average; the max across cores is what the busy
            # thread actually gets, and it is sampled every fifth tick because 32
            # sysfs reads per 5 ms would perturb the loop being measured.
            try:
                self.cpu_avg.append(
                    float(CPU_AVG_FREQ_PATH.read_text().strip()) / 1000.0
                )
            except Exception:
                self.errors += 1
            self._tick += 1
            if self._tick % 5 == 1:
                values = []
                for core in range(CPU_CORE_COUNT):
                    try:
                        values.append(
                            float(
                                pathlib.Path(
                                    CPU_CORE_FREQ_TEMPLATE.format(core=core)
                                ).read_text().strip()
                            )
                            / 1000.0
                        )
                    except Exception:
                        self.errors += 1
                if values:
                    self.cpu_max.append(max(values))
                    self.cpu_mean.append(sum(values) / len(values))
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-root", type=pathlib.Path, default=DEFAULT_MODEL)
    ap.add_argument("--fixture", type=pathlib.Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--case", default="code-p1024")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sample-interval-ms", type=float, default=5.0)
    ap.add_argument("--settle-ms", type=float, default=0.0,
                    help="Sleep after prefill before the decode loop; if a non-zero settle "
                         "removes the loop-start transient, the prefill tail was draining "
                         "rather than the decode doing real work.")
    ap.add_argument("--split-steps", action="store_true",
                    help="Synchronize after every step and record the call (host) and wait "
                         "(device) halves separately.  Adds a sync per step so it is an "
                         "attribution probe, not a throughput measurement.")
    ap.add_argument("--instrument", action="store_true",
                    help="Sample the graph-cache counters and the hipEngine allocation "
                         "counters after every step, so the loop-start transient can be "
                         "attributed to a mechanism (capture or churn) instead of inferred. "
                         "The earlier churn probe was void because it used the device "
                         "memory_info API, which returns None here; these are host-side counters.")
    ap.add_argument("--prefill-clocks", action="store_true",
                    help="Also sample the host CPU clock across the prefill window, so the clamp "
                         "can be attributed to the prefill burst rather than to the decode.")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="cProfile the first N steps and the last N steps of every decode loop "
                         "and print the top functions by self time for each window.  The two "
                         "windows do the same work, so a function that grows in the first "
                         "window is the transient's owner.")
    ap.add_argument("--no-gc", action="store_true",
                    help="Disable cyclic GC for the decode loop (re-enabled after it).  A causal "
                         "test: if the loop-start transient is the collection of the prefill's "
                         "unreachable objects, removing collections removes the transient.")
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from dataclasses import replace as _replace

    from hipengine.core.memory import memory_stats
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
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile

    fixture = json.loads(args.fixture.read_text())
    cases = {str(row["id"]): row for row in fixture["cases"]}
    if args.case not in cases:
        raise SystemExit(f"unknown case {args.case!r}; have {sorted(cases)}")
    case = cases[args.case]
    transitions = int(fixture["decode_transitions"])

    model_root = args.model_root.resolve()
    # Size the sequence from the steps actually requested, not the fixture's own
    # transition count: a longer loop must be able to hold its own context.
    max_sequence_length = int(case["prompt_tokens"]) + max(int(args.steps), transitions) + 8

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
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

    generator = resolved.construct_generator(factory)
    rows: list[dict] = []
    try:
        runner = generator.runner
        runtime = runner.runtime
        token_ids = [int(t) for t in case["prompt_token_ids"]]

        def counters() -> dict:
            """Host-side mechanism counters: graph cache state plus tracked allocations."""
            stats = memory_stats()
            sample = {
                "alloc_bytes": int(stats["total_allocated_bytes"]),
                "freed_bytes": int(stats["total_freed_bytes"]),
                "active_allocations": int(stats["active_allocations"]),
                "current_bytes": int(stats["current_allocated_bytes"]),
            }
            # Cyclic-GC counters.  A prefill leaves a large volume of unreachable
            # Python objects; the collections that reclaim them are host time and
            # land inside the first decode steps.
            for generation, generation_stats in enumerate(gc.get_stats()):
                sample[f"gc{generation}_collections"] = int(generation_stats["collections"])
                sample[f"gc{generation}_collected"] = int(generation_stats["collected"])
            sample["gc_tracked_objects"] = len(gc.get_objects()) if gc.isenabled() else -1
            for name, cache in (
                ("moe", runner.moe_graph_cache),
                ("layer", runner.layer_graph_cache),
            ):
                if cache is None:
                    continue
                cache_stats = cache.stats
                for field in ("capture", "replay", "eager", "reject"):
                    sample[f"{name}_{field}"] = int(cache_stats[field])
            return sample

        for rep in range(int(args.repeats) + 1):
            before_prefill = counters() if args.instrument else None
            prefill_sampler = None
            if args.prefill_clocks:
                prefill_sampler = BusySampler(BUSY_PATH, args.sample_interval_ms / 1000.0)
                prefill_sampler.start()
            prefill_started = time.perf_counter_ns()
            result = runner.prefill(
                token_ids,
                capture_logits=False,
                capture_target_hidden=False,
            )
            runtime.device_synchronize()
            prefill_ns = time.perf_counter_ns() - prefill_started
            # A second synchronize immediately after the first: if it absorbs a
            # large block of time, the outstanding prefill work was already queued
            # (a missing stream dependency).  If it returns instantly, the work had
            # not been issued yet (host-side async dispatch).
            resync_started = time.perf_counter_ns()
            runtime.device_synchronize()
            resync_ns = time.perf_counter_ns() - resync_started
            after_prefill = counters() if args.instrument else None
            prefill_clocks = None
            if prefill_sampler is not None:
                prefill_sampler.stop()
                prefill_clocks = {
                    "samples": len(prefill_sampler.cpu_max),
                    "cpu_max_freq_mhz": list(prefill_sampler.cpu_max),
                    "cpu_mean_freq_mhz": list(prefill_sampler.cpu_mean),
                    "gpu_busy_pct": list(prefill_sampler.samples),
                }
            settle_ms = float(args.settle_ms)
            if settle_ms > 0:
                time.sleep(settle_ms / 1000.0)
            current = int(result.token_id)
            if args.no_gc:
                gc.disable()

            sampler = BusySampler(BUSY_PATH, args.sample_interval_ms / 1000.0)
            sampler.start()
            started = time.perf_counter_ns()
            step_ns: list[int] = []
            call_ns: list[int] = []
            wait_ns: list[int] = []
            cpu_ns: list[int] = []
            step_counters: list[dict] = []
            split = bool(args.split_steps)
            window = max(1, int(args.profile_steps))
            profile_a = cProfile.Profile() if window > 1 else None
            profile_b = cProfile.Profile() if window > 1 else None
            if profile_a is not None:
                profile_a.enable()
            for step_index in range(int(args.steps)):
                if profile_b is not None and step_index == max(0, int(args.steps) - window):
                    if profile_a is not None:
                        profile_a.disable()
                    profile_b.enable()
                cpu_started = time.process_time_ns()
                t0 = time.perf_counter_ns()
                result = runner.step(
                    current,
                    capture_logits=False,
                    capture_target_hidden=False,
                    token_id_resident=True,
                )
                current = int(result.token_id)
                t1 = time.perf_counter_ns()
                step_ns.append(t1 - t0)
                cpu_ns.append(time.process_time_ns() - cpu_started)
                if args.instrument:
                    step_counters.append(counters())
                if split:
                    runtime.device_synchronize()
                    t2 = time.perf_counter_ns()
                    call_ns.append(t1 - t0)
                    wait_ns.append(t2 - t1)
            runtime.device_synchronize()
            ended = time.perf_counter_ns()
            sampler.stop()
            if profile_a is not None:
                profile_a.disable()
            if profile_b is not None:
                profile_b.disable()
            if args.no_gc:
                gc.enable()
                gc.collect()

            wall_ms = (ended - started) / 1_000_000.0
            per_token_ms = wall_ms / int(args.steps)
            s = sampler.samples
            busy_mean = statistics.fmean(s) if s else float("nan")
            busy_median = statistics.median(s) if s else float("nan")
            busy_min = min(s) if s else float("nan")
            busy_max = max(s) if s else float("nan")
            idle_share = max(0.0, 1.0 - busy_mean / 100.0)

            row = {
                "repetition": rep,
                "warmup": rep == 0,
                "steps": int(args.steps),
                "wall_ms": wall_ms,
                "prefill_ms": prefill_ns / 1e6,
                "resync_ms": resync_ns / 1e6,
                "settle_ms": settle_ms,
                "prefill_tok_s": int(case["prompt_tokens"]) / (prefill_ns / 1e9) if prefill_ns else None,
                "per_token_ms": per_token_ms,
                "decode_tok_s": 1000.0 / per_token_ms,
                "busy_samples": len(s),
                "sclk_mhz": list(sampler.sclk),
                "mclk_mhz": list(sampler.mclk),
                "sclk_mean_mhz": statistics.fmean(sampler.sclk) if sampler.sclk else None,
                "sclk_min_mhz": min(sampler.sclk) if sampler.sclk else None,
                "sclk_max_mhz": max(sampler.sclk) if sampler.sclk else None,
                "cpu_avg_freq_mhz": list(sampler.cpu_avg),
                "cpu_max_freq_mhz": list(sampler.cpu_max),
                "cpu_mean_freq_mhz": list(sampler.cpu_mean),
                "cpu_avg_freq_mean_mhz": (
                    statistics.fmean(sampler.cpu_avg) if sampler.cpu_avg else None
                ),
                "cpu_max_freq_mean_mhz": (
                    statistics.fmean(sampler.cpu_max) if sampler.cpu_max else None
                ),
                "cpu_max_freq_min_mhz": (
                    min(sampler.cpu_max) if sampler.cpu_max else None
                ),
                "prefill_clocks": prefill_clocks,
                "step_ms": [n / 1e6 for n in step_ns],
                "cpu_ms": [n / 1e6 for n in cpu_ns],
                "cpu_ms_mean": statistics.fmean(cpu_ns) / 1e6 if cpu_ns else None,
                "call_ms": [n / 1e6 for n in call_ns],
                "wait_ms": [n / 1e6 for n in wait_ns],
                "step_ms_mean": statistics.fmean(step_ns) / 1e6 if step_ns else None,
                "step_ms_min": min(step_ns) / 1e6 if step_ns else None,
                "step_ms_max": max(step_ns) / 1e6 if step_ns else None,
                "step_ms_median": statistics.median(step_ns) / 1e6 if step_ns else None,
                "busy_mean_pct": busy_mean,
                "busy_median_pct": busy_median,
                "busy_min_pct": busy_min,
                "busy_max_pct": busy_max,
                "sampler_errors": sampler.errors,
                "implied_idle_share": idle_share,
                "implied_gpu_ms_per_token": per_token_ms * busy_mean / 100.0,
                "implied_host_ms_per_token": per_token_ms * idle_share,
            }
            if args.instrument:
                series = [after_prefill, *step_counters]
                n = len(step_counters)
                q = max(1, n // 8)
                fields = (
                    "moe_capture", "layer_capture", "moe_eager", "layer_eager",
                    "alloc_bytes", "freed_bytes", "active_allocations",
                    "gc0_collections", "gc1_collections", "gc2_collections",
                    "gc0_collected", "gc1_collected", "gc2_collected",
                )
                bands: dict[str, list[int]] = {}
                for field in fields:
                    values = []
                    for i in range(0, n, q):
                        end = min(i + q, n)
                        values.append(
                            series[end].get(field, 0) - series[i].get(field, 0)
                        )
                    bands[field] = values
                row["instrument_bands"] = bands
                row["instrument_prefill"] = {
                    field: after_prefill.get(field, 0) - before_prefill.get(field, 0)
                    for field in fields
                }
                row["step_counters"] = step_counters
            for tag, profiler in (("transient", profile_a), ("steady", profile_b)):
                if profiler is None:
                    continue
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(14)
                text = stream.getvalue()
                row[f"profile_{tag}"] = text
                print(f"        --- {tag} window: top functions by self time ---", flush=True)
                for line in text.splitlines()[4:22]:
                    print(f"        {line}", flush=True)
            rows.append(row)
            tag = "warmup" if rep == 0 else f"rep{rep}"
            print(
                f"[{tag}] wall {per_token_ms:7.2f} ms/tok ({1000.0/per_token_ms:5.2f} tok/s) "
                f"busy mean {busy_mean:5.1f}% median {busy_median:5.1f}% "
                f"min {busy_min:3.0f}% max {busy_max:3.0f}%  "
                f"=> gpu {row['implied_gpu_ms_per_token']:6.2f} host {row['implied_host_ms_per_token']:5.2f} ms/tok"
                f"  prefill {prefill_ns/1e6:8.2f} ms ({row['prefill_tok_s']:7.1f} tok/s)"
                f"  resync {resync_ns/1e6:7.2f} ms",
                flush=True,
            )
            # A compact picture of WHEN the device had no work.  A uniform busy
            # profile means per-step host cost; a bursty one points at a periodic
            # synchronous event instead.
            stride = max(1, len(s) // 60)
            print(f"        busy timeline (every {stride}): {' '.join(f'{v:02d}' for v in s[::stride])}", flush=True)
            if sampler.sclk:
                q = max(1, len(sampler.sclk) // 16)
                bands = [statistics.fmean(sampler.sclk[i:i + q]) for i in range(0, len(sampler.sclk), q)]
                print(
                    "        sclk by sixteenth: " + " ".join(f"{v:5.0f}" for v in bands)
                    + f"   (mean {statistics.fmean(sampler.sclk):5.0f} min {min(sampler.sclk):5.0f} max {max(sampler.sclk):5.0f} MHz)",
                    flush=True,
                )
                if sampler.mclk:
                    q = max(1, len(sampler.mclk) // 16)
                    bands = [statistics.fmean(sampler.mclk[i:i + q]) for i in range(0, len(sampler.mclk), q)]
                    print(
                        "        mclk by sixteenth: " + " ".join(f"{v:5.0f}" for v in bands)
                        + f"   (mean {statistics.fmean(sampler.mclk):5.0f} MHz)",
                        flush=True,
                    )
            if prefill_clocks and prefill_clocks["cpu_max_freq_mhz"]:
                values = prefill_clocks["cpu_max_freq_mhz"]
                q = max(1, len(values) // 8)
                bands = [statistics.fmean(values[i:i + q]) for i in range(0, len(values), q)]
                busy = prefill_clocks["gpu_busy_pct"]
                bq = max(1, len(busy) // 8)
                busy_bands = [statistics.fmean(busy[i:i + bq]) for i in range(0, len(busy), bq)]
                print(
                    "        prefill host cpu MHz by eighth (max core): "
                    + " ".join(f"{v:5.0f}" for v in bands),
                    flush=True,
                )
                print(
                    "        prefill gpu busy by eighth:                "
                    + " ".join(f"{v:5.0f}" for v in busy_bands),
                    flush=True,
                )
            if sampler.cpu_max:
                q = max(1, len(sampler.cpu_max) // 16)
                max_bands = [
                    statistics.fmean(sampler.cpu_max[i:i + q])
                    for i in range(0, len(sampler.cpu_max), q)
                ]
                mean_bands = [
                    statistics.fmean(sampler.cpu_mean[i:i + q])
                    for i in range(0, len(sampler.cpu_mean), q)
                ]
                package = (
                    f"cpu0 avg {statistics.fmean(sampler.cpu_avg):5.0f} MHz"
                    if sampler.cpu_avg
                    else "cpu0 avg n/a"
                )
                print(
                    "        host cpu MHz by sixteenth (max core): "
                    + " ".join(f"{v:5.0f}" for v in max_bands),
                    flush=True,
                )
                print(
                    "        host cpu MHz by sixteenth (mean of "
                    + f"{CPU_CORE_COUNT}): "
                    + " ".join(f"{v:5.0f}" for v in mean_bands)
                    + f"   (max {max(sampler.cpu_max):5.0f} "
                    + f"min {min(sampler.cpu_max):5.0f} MHz; {package})",
                    flush=True,
                )
            if step_ns:
                n = len(step_ns)
                q = max(1, n // 8)
                bands = [statistics.fmean(step_ns[i:i + q]) / 1e6 for i in range(0, n, q)]
                print(
                    "        step ms by eighth: "
                    + " ".join(f"{v:6.2f}" for v in bands)
                    + f"   (mean {statistics.fmean(step_ns)/1e6:6.2f} min {min(step_ns)/1e6:6.2f} max {max(step_ns)/1e6:6.2f})",
                    flush=True,
                )
            if call_ns:
                n = len(call_ns)
                q = max(1, n // 8)
                cb = [statistics.fmean(call_ns[i:i + q]) / 1e6 for i in range(0, n, q)]
                wb = [statistics.fmean(wait_ns[i:i + q]) / 1e6 for i in range(0, n, q)]
                print("        call ms by eighth: " + " ".join(f"{v:6.2f}" for v in cb), flush=True)
                print("        wait ms by eighth: " + " ".join(f"{v:6.2f}" for v in wb), flush=True)
            if cpu_ns:
                n = len(cpu_ns)
                q = max(1, n // 8)
                ub = [statistics.fmean(cpu_ns[i:i + q]) / 1e6 for i in range(0, n, q)]
                print(
                    "        cpu ms by eighth:  " + " ".join(f"{v:6.2f}" for v in ub)
                    + "   (process CPU time; it stays flat when the wall is inflated by "
                      "blocking on a device-paced queue)",
                    flush=True,
                )
            if args.instrument:
                bands = row["instrument_bands"]
                print(
                    "        captures by eighth (moe/layer): "
                    + " ".join(
                        f"{a}/{b}" for a, b in zip(bands["moe_capture"], bands["layer_capture"])
                    ),
                    flush=True,
                )
                print(
                    "        eager by eighth (moe/layer):    "
                    + " ".join(
                        f"{a}/{b}" for a, b in zip(bands["moe_eager"], bands["layer_eager"])
                    ),
                    flush=True,
                )
                print(
                    "        alloc MiB by eighth:            "
                    + " ".join(f"{v / 1048576.0:6.2f}" for v in bands["alloc_bytes"]),
                    flush=True,
                )
                print(
                    "        live allocs by eighth:          "
                    + " ".join(f"{v:+5d}" for v in bands["active_allocations"])
                    + "   prefill delta "
                    + ", ".join(
                        f"{k}={v}" for k, v in sorted(row["instrument_prefill"].items())
                    ),
                    flush=True,
                )
                print(
                    "        gc collections by eighth (0/1/2): "
                    + " ".join(
                        f"{a}/{b}/{c}"
                        for a, b, c in zip(
                            bands["gc0_collections"],
                            bands["gc1_collections"],
                            bands["gc2_collections"],
                        )
                    ),
                    flush=True,
                )
                print(
                    "        gc collected by eighth (0/1/2):   "
                    + " ".join(
                        f"{a}/{b}/{c}"
                        for a, b, c in zip(
                            bands["gc0_collected"],
                            bands["gc1_collected"],
                            bands["gc2_collected"],
                        )
                    ),
                    flush=True,
                )
    finally:
        try:
            generator.close()
        except Exception:
            pass

    measured = [r for r in rows if not r["warmup"]]
    payload = {
        "schema": 1,
        "kind": "qwen4exp_de5_wall_split",
        "case": args.case,
        "steps": int(args.steps),
        "decode_boundary": "device-argmax (capture_logits=False, token_id_resident=True)",
        "busy_counter": str(BUSY_PATH),
        "sclk_counter": str(SCLK_PATH),
        "mclk_counter": str(MCLK_PATH),
        "sample_interval_ms": float(args.sample_interval_ms),
        "gc_disabled_in_decode": bool(args.no_gc),
        "instrumented": bool(args.instrument),
        "rows": rows,
        "measured": measured,
        "summary": {
            "wall_ms_per_token_median": statistics.median(r["per_token_ms"] for r in measured) if measured else None,
            "busy_mean_pct_median": statistics.median(r["busy_mean_pct"] for r in measured) if measured else None,
            "implied_host_ms_per_token_median": (
                statistics.median(r["implied_host_ms_per_token"] for r in measured) if measured else None
            ),
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
