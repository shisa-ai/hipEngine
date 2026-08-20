#!/usr/bin/env python3
"""Best-effort 27B (H5120 dense) prefill chunk A/B: 512 vs 1024 rows.

Clock pinning is impossible on this power-limited gfx1151 APU lane (sysfs
pp_dpm_sclk / pp_od_clk_voltage writes rejected; power_dpm_force_performance_level
accepted but ignored). Under sustained load the sclk settles to a thermal
equilibrium band of ~1180-1415 MHz (measured, scripts/pn3_clock_probe.py),
recovering to 2900 MHz whenever load drops.

Best-case protocol for a power-limited lane:
  1. Sustained thermal pre-warmup (~75 s of continuous prefill) until the sclk
     is inside the settled band (not still descending from 2900).
  2. Interleaved 512/1024 legs with alternating leg order per rep.
  3. A background sclk sampler records the per-leg clock distribution, so a
     512-vs-1024 delta can be checked for correlation with clock drift.
  4. Verdict rule: call a win only if the per-leg median delta exceeds the
     clock-band swing (proxy for run noise). Anything smaller is INCONCLUSIVE
     on this lane and deferred to the non-power-limited gfx1151 system.

The 35B-A3B known ~1.2% chunk-512 win is re-measured under the same protocol
as a sensitivity sanity check.
"""
import sys, os, time, ctypes, statistics, threading, subprocess
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

CARD = '/sys/class/drm/card1/device'

def force_high():
    try:
        subprocess.run(
            ['sudo', 'sh', '-c', f'echo high > {CARD}/power_dpm_force_performance_level'],
            check=True, capture_output=True, text=True, timeout=20,
        )
    except Exception as e:
        print(f"[warn] could not force high: {e}", flush=True)

def read_sclk():
    try:
        with open(f'{CARD}/pp_dpm_sclk') as f:
            for l in f.read().splitlines():
                if '*' in l:
                    return int(l.split(':')[1].replace('Mhz', '').strip().replace('*', ''))
    except Exception:
        pass
    return -1

class SclkSampler:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
    def _run(self):
        while not self._stop.is_set():
            v = read_sclk()
            if v > 0:
                self.samples.append(v)
            self._stop.wait(1.0)
    def __enter__(self):
        self._t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=3)
    def band(self):
        if not self.samples: return (-1, -1, -1)
        s = sorted(self.samples)
        return (s[0], statistics.median(s), s[-1])

def pref(model, linear, moe, L):
    cfg = PrefillConfig(linear_chunk_size=linear, moe_chunk_size=moe)
    with rm.Qwen35GGUFResidentSession(model,
            max_sequence_length=L, backend='hip_gfx1151', prefill_config=cfg) as s:
        toks = [9707]*L
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        t0 = time.perf_counter()
        s.prefill(toks, use_bulk=True, return_logits=False)
        s.runtime.device_synchronize()
        return (time.perf_counter() - t0) * 1e3

M35 = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
M27 = '/models/gguf/Qwen3.6-27B-Q4_K_M.gguf'

def thermal_warmup(model, L, seconds=75):
    print(f"  thermal pre-warmup {seconds}s (continuous prefill) ...", flush=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        pref(model, 1024, 1024, L)
    lo, med, hi = read_sclk(), read_sclk(), read_sclk()
    print(f"  warmup done, sclk now ~{med} MHz", flush=True)

def run_pair(label, model, L, reps=6):
    print(f"\n== {label} (L={L}) — best-effort protocol (no clock pin available) ==", flush=True)
    force_high()
    thermal_warmup(model, L)
    d, g = [], []          # per-leg times
    dc, gc = [], []        # per-leg median sclk
    for rep in range(reps):
        order = ((1024, 512), (512, 1024)) if rep % 2 == 0 else ((512, 1024), (1024, 512))
        for cfg in order:
            force_high()
            with SclkSampler() as samp:
                ms = pref(model, *cfg, L)
            lo, med, hi = samp.band()
            (d if cfg[0] == 1024 else g).append(ms)
            (dc if cfg[0] == 1024 else gc).append(med)
            print(f"  rep{rep} chunk{cfg[0]}: {ms:.0f}ms  sclk[med]={med} band={lo}-{hi}", flush=True)
    md, mg = statistics.median(d), statistics.median(g)
    delta_pct = (md/mg - 1) * 100
    clock_swing = (max(max(dc), max(gc)) - min(min(dc), min(gc))) / statistics.median(dc + gc) * 100
    print(f"  median 1024={md:.0f}ms  512={mg:.0f}ms  delta={(md-mg):+.0f}ms ({delta_pct:+.2f}%)  n_legs={len(d)}", flush=True)
    print(f"  legs 1024: {[f'{x:.0f}@{c}MHz' for x, c in zip(d, dc)]}", flush=True)
    print(f"  legs 512 : {[f'{x:.0f}@{c}MHz' for x, c in zip(g, gc)]}", flush=True)
    print(f"  per-leg sclk swing across A/B: ~{clock_swing:.1f}% of median clock", flush=True)
    if abs(delta_pct) > clock_swing:
        w = "512" if delta_pct > 0 else "1024"
        print(f"  VERDICT: {w} wins beyond clock-band noise ({delta_pct:+.2f}% vs {clock_swing:.1f}% swing)", flush=True)
    else:
        print(f"  VERDICT: INCONCLUSIVE on this power-limited lane ({delta_pct:+.2f}% within {clock_swing:.1f}% swing) — defer to non-power-limited gfx1151", flush=True)
    return md, mg

if __name__ == '__main__':
    print("best-effort (no pin): thermal warmup + interleaved legs + per-leg sclk sampling", flush=True)
    run_pair("35B-A3B sanity (expected ~1.2% 512 win)", M35, 2048)
    run_pair("27B dense H5120 (the resolution)", M27, 2048)
    print("\nDONE", flush=True)
