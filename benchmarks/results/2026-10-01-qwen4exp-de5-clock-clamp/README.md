# DE-5: the loop-start transient is a host CPU clock clamp, not engine host code

Date: 2026-10-01
Row: `DE-5` (Runtime) in `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md`
Harness: `scripts/qwen4exp_de5_wall_split.py`, `scripts/qwen4exp_de5_clock_burst_probe.py`
Model: `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, production profile, case `code-p1024`
Host: AMD Radeon 8060S (integrated, **gfx1151**), 32-thread APU, `amd-pstate-epp` / `powersave` / `boost=1`

## Question

DE-5's last open item was the reproducible cost it could not attribute: every
decode loop that follows a prefill spends its first ~30 steps far above the
steady-state step time (**~1.5-1.9 s of extra wall**), and a 5 s idle between
the prefill and the decode removes it entirely but costs more than it saves
(net **+3.3 s**). Four mechanisms had been excluded — unsynchronized prefill
tail, queued device work, GPU clock throttling, and pure host blocking — and the
remaining candidates were named as "a graph cache, a residency or page-table
effect, or allocation churn". The allocation-churn probe was **void**: it used
`runtime.memory_info()`, which returns `None` on this runtime.

## What was measured

The same production decode boundary (`prefill` then N `runner.step` calls,
`capture_logits=False`, `capture_target_hidden=False`, `token_id_resident=True`,
no profiler) with four new host-side instruments in the probe:

* `--instrument` — the graph cache's own capture/replay/eager counters, the
  hipEngine device-allocation counters, and the cyclic-GC counters, sampled after
  every step. The churn probe is host-side counters now, not the device API.
* `--split-steps` — a `device_synchronize()` per step, so the host call half and
  the device wait half are separated.
* process CPU time (`time.process_time_ns`) per step, which distinguishes real
  host CPU burn from blocking on a device-paced queue.
* host CPU clocks — `scaling_cur_freq` across all 32 cores (max and mean) and
  `cpuinfo_avg_freq`. This is the instrument the harness never had: every prior
  run sampled the **GPU** clocks (`pp_dpm_sclk`/`mclk`, flat at 2900/1000 MHz
  throughout) and never the CPU's.

## Result 1 — the mechanism is a host CPU package clock clamp

`code-p1024`, 96 decode steps, warmup plus one repeat, one run
(`de5-p1024-prefillclocks.json`), the same pattern in every arm below:

| loop | step ms by eighth | max CPU core MHz by sixteenth |
| --- | --- | --- |
| warmup (first prefill+decode after load) | 59.5 52.7 52.4 52.1 51.8 52.2 51.6 52.3 | 3267 3552 4724 **5117 5093 5110 5097 5086** 5031 4968 4968 4969 4864 4933 4780 4575 |
| rep1 (every later prefill+decode) | **162.1 80.1 56.5** 49.3 49.5 49.3 49.4 49.5 | **2000 2000 2000 2000 2000 2000** 2032 3940 4547 3934 3642 3972 3939 3543 3665 3724 |

The package sits at its **2.0 GHz base clock** — every one of 32 cores, so the
busy thread gets no more than that — for the first ~2 s of every repeat's
post-prefill decode, and the step time tracks it: 162 ms at the clamp onset (when
`cpuinfo_avg_freq` on cpu0 reads **593-600 MHz**, a deeper transient state),
80 ms as it steps up, and the 49.3 ms floor once boost returns. The warmup is the
shallower case, and the contrast below is what identifies the variable.

**The clamp is established during the prefill, not during the decode.** Sampling
across the prefill window itself:

| prefill | host CPU MHz by eighth of the prefill | decode step ms by eighth |
| --- | --- | --- |
| `code-p1024` (4.05 s) | 2446 **2000 2000 2000 2000 2000 2000 2000 2000** | 162.1 80.1 56.5 49.3 |
| `code-p512` (1.99 s) | 2871 **2000 2000 2000 2000 2000 2000 2000 2000** | 102.3 58.4 47.4 47.4 |

So the transient **scales with the prefill** (4.05 s prefill: ~2 s at 2000 MHz and
a 162 ms first step; 1.99 s: ~1.3 s and 102 ms), and the warmup's own prefill
(`3541 2438 2000 ...`) shows the clamp engaging about 1.3 s into it.

**How deep the decode starts tracks how much of the prefill ran clamped**, not
the prefill's length: the warmups' prefills are clamped for **0.65-0.78** of
their samples (their first third runs at ~3.5 GHz while the GPU load ramps up)
and their decodes start at **3.2-3.5 GHz** with 57-60 ms steps, while the
repeats' prefills are clamped for **0.97-0.98** and their decodes start at
exactly **2000 MHz** with 162 ms steps. Same run, same case, same model.

**The 5 s idle arm is the causal confirmation.** With `--settle-ms 5000` the
clock is already back at boost at the first decode step and the transient is
gone:

```
rep1  step ms by eighth:   49.35 49.14 49.22 49.42 49.37 49.39 49.41 49.41   (mean 49.34, min 48.97, max 51.26)
rep1  max CPU core MHz:    3553 3678 5077 5085 5097 5066 5060 4919 5037 5044 ...   (by sixteenth, ~0.35 s each)
```

That is the same arm DE-5 priced at **net +3.3 s**, now explained: the idle does
not remove work, it waits out the clamp.

## Result 2 — every named candidate is excluded by direct counters

The three candidates the row named — a graph cache, allocation churn, and a
residency or page-table effect — plus the two host-side suspects the counters
made cheap to check (cyclic GC and cgroup throttling):

| candidate | instrument | measurement |
| --- | --- | --- |
| graph cache (capture/re-capture) | `MoeGraphCache.stats` per step | **48 MoE captures in the warmup's first 16 steps, then 0 in every band of every later decode.** Capture is cheap here (~100 ms total) and cannot be the 1.5 s |
| device allocation churn | hipEngine allocation counters per step | **0.00 MiB in every band of every rep.** The warmup prefill allocates 167,772,176 B and frees 52,428,808 B (+5 live); **every later prefill allocates and frees 0 B** — the workspace is reused, and the transient appears in exactly those reps |
| cyclic GC collecting the prefill's garbage | `gc.get_stats()` per step, plus a GC-disabled arm | **0 collections in the transient bands**; the warmup's collections reclaim 45 objects. With `gc.disable()` for the decode the transient is unchanged (**151.1 / 84.3 / 58.0** against **152.4 / 78.2 / 54.8** ms) |
| cgroup CPU throttling | `/sys/fs/cgroup/cpu.stat` | `nr_periods 0`, `nr_throttled 0`, `throttled_usec 0` |

The **residency/page-table candidate is not directly measured** — it is
*unnecessary* rather than excluded, because the clock series predicts the step
series in every arm: the deep clamp (162 ms) and the shallow one (57 ms) differ
by the same ratio as their clocks, and in the settled arm the transient is absent
with no change to residency or page tables.

Two further candidates are excluded by the same runs:

* **Blocking on a device-paced queue.** With a sync per step the **device wait
  is 0.03-0.14 ms in every band** while the host call half carries the whole
  transient (**153.8 -> 49.4 ms**). The transient is inside the host call.
* **Extra host work.** Process CPU time tracks the wall (**193.0 -> 81.1 ms**
  per step), and `cProfile` over a 64-step window against the last 32 steps shows
  the **same function mix with no new path** — `_launch`, `launch_gguf_linear`,
  `os.environ` lookups — with per-call costs ~1.5x higher, i.e. the same work at a
  lower clock.

**A correction to the earlier reading of this row's own numbers.** The steady
state is **not** host-submission-bound: within one run the step is 49.3 ms while
the package is still at the 2.0 GHz clamp and 49.3 ms after it returns to
3.9-4.5 GHz, and 49.35 ms flat in the settled arm at 5.0-5.1 GHz — over a 2.5x
clock range the step time does not move, so the device is the limit above the
base clock and the host becomes the limit only in the deep clamp. `call_ms ~=
49.4` with `wait_ms ~= 0.03` is device-paced blocking inside the launch calls,
not 49 ms of host CPU. (The first decode after model load runs 52 ms at boost —
a different regime, and one that moves the *opposite* way for a clock effect,
so it is not part of this comparison.)

## Result 3 — the trigger is the prefill's own load, not any single load class

`scripts/qwen4exp_de5_clock_burst_probe.py` applies each load class in isolation
(5 s arms, burner in a subprocess so the sampler is not GIL-starved, 32 cores
sampled every 100 ms). Median max-core clock:

| arm | what it loads | max core MHz (median / min) |
| --- | --- | --- |
| idle | nothing | 3533 / 2018 |
| cpu | one thread at 100% | **5117** / 2007 |
| cpu_mem | one thread driving 64 MiB copies + a strided walk over 256 MiB | **5117** / 5108 |
| gpu | D2D copies, 474 GiB | 3536 / 3510 |
| cpu+gpu | both | **5118** / 3532 |
| mem_gpu | memory burner + D2D | 3555 / 3533 |
| release | after the arms | 3557 / 2000 |

**No synthetic arm reaches the 2.0 GHz base clamp**, so the trigger is not "a CPU
burst", "host memory traffic", "GPU DMA bandwidth", or a pair of them. The one
consistent trend is that **GPU load alone suppresses the CPU boost** (3536-3555
against 5117 MHz) without clamping to base, which is the shape of a shared
package power budget; the prefill's heavier load goes further. The machine has
background load, so the per-series *minimum* is noisy and the median is the
statistic to read.

## What this changes

* DE-5's transient is a **platform power-management response to the prefill**,
  not an engine host-code defect. Nothing in `reset()`, the wrapper, the graph
  cache, the allocator, or the GC is the owner, and there is no in-tree fix.
* The only mitigation measured remains the idle settle, and it is **net
  negative** (+3.3 s against a ~1.5-1.9 s transient) because it relocates the
  wait rather than removing the work.
* The row's original scope — the wrapper's ~3 ms/token, ~5-6% of a step — is
  unchanged and still not a lever.
* **Reusable method finding:** a host-bound loop measured on this host is
  measuring the package clock as much as the engine. Any before/after on this
  machine should sample `scaling_cur_freq`; a 2.5x clock swing is worth 3.3x on a
  host-bound step and is invisible in the GPU counters.

## Limits

One host, one model, one quant, one profile, one case family; wall medians
around a device synchronize, not a serving-path throughput measurement. The
clamp cannot be pinned for a direct causal test — writing
`scaling_min_freq` requires root and this session has none — so the causal link
rests on the settle arm plus the clock/step-time correspondence across four
independent repetitions, not on forcing the clock. `cpuinfo_avg_freq` is a
per-CPU attribute, not a package average (one thread pinned to cpu4 reads 5118
MHz there while cpu0 and cpu8 read 2000), so it is reported as cpu0's own value.
The burst control's negative result is bounded by the load classes it tried: it
does not include a compute-heavy GPU kernel loop, so "the prefill's GPU power
profile" remains the untested candidate trigger. No performance claim, no
promotion, no default changed.

## Files

* `artifact.json` — compact per-run summaries and series for every arm above.
* `scripts/qwen4exp_de5_wall_split.py` — `--instrument`, `--split-steps`,
  `--no-gc`, `--profile-steps`, `--prefill-clocks`, CPU clock sampling, process
  CPU time per step.
* `scripts/qwen4exp_de5_clock_burst_probe.py` — the load-class control.
