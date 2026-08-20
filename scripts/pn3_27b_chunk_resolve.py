#!/usr/bin/env python3
"""27B dense (H5120) prefill chunk 512-vs-1024 resolution on the 140 W gfx1151 lane.

Deferred from the ZBook handoff (60 W lane, clock could not be pinned and the
~10% clock-band swing swamped the verdict). On this desktop gfx1151:

  * Clock pinning works: `power_dpm_force_performance_level=high` is honored
    and holds ~2620-2800 MHz under sustained load (measured,
    scripts/pn3_clock_probe.py).
  * Protocol still requires thermal pre-warmup + interleaved counter-rotated
    legs + per-leg sclk sampling, but the swing is ~+/-4% so effects around
    1% are resolvable with enough reps.

Improvements over the 60 W script (`pn3_27b_chunk_pinned.py`):

  * Single resident session; the chunk size is changed in place by mutating
    `session.prefill_config` between legs (the runner reads
    `prefill_config.linear_chunk_size` per call). No per-leg model reload, so
    the dominant reload-variance source is removed.
  * The 35B sanity leg in the 60 W script was a no-op: `_gguf_prefill_chunk_sizes_for`
    forces the 512-row geometry override for H2048-MoE, so both legs ran 512.
    Here we optionally bypass that override (`--no-override`) so the 35B leg is a
    true 512-vs-1024 positive control (known ~1.2% 512 win).
"""
import argparse
import ctypes
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import replace
from typing import Any

sys.path.insert(0, '/home/lhl/hipEngine-main')
import hipengine.runtime.qwen35_gguf_runner as rm  # noqa: E402
from hipengine.runtime.prefill import PrefillConfig  # noqa: E402

ctypes.CDLL('libamdhip64.so')

CARD = '/sys/class/drm/card1/device'
M35 = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
M27 = '/models/gguf/Qwen3.6-27B-Q4_K_M.gguf'


def force_high() -> None:
    try:
        subprocess.run(
            ['sudo', 'sh', '-c', f'echo high > {CARD}/power_dpm_force_performance_level'],
            check=True, capture_output=True, text=True, timeout=20,
        )
    except Exception as exc:  # pragma: no cover - best-effort
        print(f"[warn] could not force high: {exc}", flush=True)


def read_sclk() -> int:
    try:
        with open(f'{CARD}/pp_dpm_sclk') as f:
            for line in f.read().splitlines():
                if '*' in line:
                    return int(line.split(':')[1].replace('Mhz', '').strip().replace('*', ''))
    except Exception:
        pass
    return -1


class SclkSampler:
    def __init__(self) -> None:
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = read_sclk()
            if value > 0:
                self.samples.append(value)
            self._stop.wait(0.75)

    def __enter__(self) -> "SclkSampler":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def median(self) -> int:
        return int(statistics.median(self.samples)) if self.samples else -1


def timed_prefill(session: Any, toks: list[int], n: int) -> list[float]:
    results: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.prefill(toks, use_bulk=True, return_logits=False)
        session.runtime.device_synchronize()
        results.append((time.perf_counter() - t0) * 1e3)
    return results


def run_ab(model: str, label: str, length: int, *, reps: int, n_timed: int,
           bypass_override: bool, warmup_rounds: int = 2) -> None:
    print(f"\n== {label} (L={length}) — 140 W gfx1151 interleaved A/B ==", flush=True)
    if bypass_override:
        original = rm._gguf_prefill_chunk_sizes_for
        rm._gguf_prefill_chunk_sizes_for = lambda *a, **k: None  # type: ignore[assignment]
        print("  [control] geometry override bypassed for the 35B positive control", flush=True)
    try:
        with rm.Qwen35GGUFResidentSession(
            model, max_sequence_length=length, backend='hip_gfx1151',
            prefill_config=PrefillConfig(linear_chunk_size=512, moe_chunk_size=512),
        ) as session:
            toks = [9707] * length
            print("  thermal pre-warmup (discarded) ...", flush=True)
            for chunk in (512, 1024):
                session.prefill_config = replace(session.prefill_config, linear_chunk_size=chunk, moe_chunk_size=chunk)
                for _ in range(warmup_rounds):
                    timed_prefill(session, toks, 1)
            times: dict[int, list[float]] = {512: [], 1024: []}
            clocks: dict[int, list[int]] = {512: [], 1024: []}
            for rep in range(reps):
                order = [1024, 512] if rep % 2 == 0 else [512, 1024]
                for chunk in order:
                    session.prefill_config = replace(
                        session.prefill_config, linear_chunk_size=chunk, moe_chunk_size=chunk)
                    force_high()
                    with SclkSampler() as sampler:
                        leg = timed_prefill(session, toks, n_timed)
                    times[chunk].extend(leg)
                    clocks[chunk].append(sampler.median())
                    print(f"  rep{rep} chunk{chunk}: median={statistics.median(leg):.0f}ms "
                          f"legs={[f'{x:.0f}' for x in leg]} sclk_med={sampler.median()}MHz", flush=True)
    finally:
        if bypass_override:
            rm._gguf_prefill_chunk_sizes_for = original  # type: ignore[assignment]
    med_512 = statistics.median(times[512])
    med_1024 = statistics.median(times[1024])
    delta_pct = (med_512 / med_1024 - 1) * 100
    all_clocks = clocks[512] + clocks[1024]
    clock_swing = (max(all_clocks) - min(all_clocks)) / statistics.median(all_clocks) * 100 if all_clocks else 0.0
    print(f"  median 512={med_512:.0f}ms  1024={med_1024:.0f}ms  delta={(med_512-med_1024):+.0f}ms ({delta_pct:+.2f}%)  n={len(times[512])}", flush=True)
    print(f"  512 sclk medians: {clocks[512]}", flush=True)
    print(f"  1024 sclk medians: {clocks[1024]}", flush=True)
    print(f"  per-leg sclk swing across A/B: ~{clock_swing:.1f}% of median clock", flush=True)
    if abs(delta_pct) > clock_swing:
        winner = "1024" if delta_pct > 0 else "512"
        print(f"  VERDICT: {winner} wins beyond clock-band noise ({delta_pct:+.2f}% vs {clock_swing:.1f}% swing)", flush=True)
    else:
        print(f"  VERDICT: INCONCLUSIVE within clock-band noise ({delta_pct:+.2f}% vs {clock_swing:.1f}% swing)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=6)
    parser.add_argument('--n-timed', type=int, default=2)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--no-override', action='store_true',
                        help='bypass the H2048-MoE 512-row geometry override so the 35B sanity is a true 512-vs-1024')
    parser.add_argument('--skip-35b', action='store_true')
    parser.add_argument('--skip-27b', action='store_true')
    args = parser.parse_args()

    print(f"protocol: single resident session, in-place chunk mutation, interleaved counter-rotated legs, "
          f"per-leg sclk sampling (reps={args.reps}, n_timed={args.n_timed}, L={args.length})", flush=True)
    if not args.skip_35b:
        run_ab(M35, "35B-A3B sanity (expected ~1.2% 512 win; positive control)",
               args.length, reps=args.reps, n_timed=args.n_timed, bypass_override=args.no_override)
    if not args.skip_27b:
        run_ab(M27, "27B dense H5120 (the resolution)",
               args.length, reps=args.reps, n_timed=args.n_timed, bypass_override=False)
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
