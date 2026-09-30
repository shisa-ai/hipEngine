# DE-5 re-derivation: the wrapper host cost is ~3 ms/token, and the loop-start cost is a per-prefill state-rewind transient

Date: 2026-09-30
Row: `DE-5` (Runtime) in `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md`
Harness: `scripts/qwen4exp_de5_wall_split.py`
Model: `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, production profile
Case: `code-p1024`, one prompt, one host (Radeon Pro W7900 / gfx1100)

## Question

DE-5's row scope is "serving-wrapper host time, ~3.6 ms/token above the runner
loop (logits D2H, sampling, bookkeeping) after the R7 one-H2D promotion". The
row's prior correction left one open follow-up: the anomaly it could not explain
was attributed to the harness, "most likely
`snapshot_restore_between_repeats: true` interacting with the QSA indexed-sparse
path's rewindable index state (hypothesis, untested)".

## What was measured

A decode loop that mirrors the production boundary (`prefill` then exactly N
`runner.step` calls, `capture_logits=False`, `capture_target_hidden=False`,
`token_id_resident=True`), with a device busy-counter sampler and per-step wall
timing. This probe does **not** use `snapshot_restore_between_repeats`; the only
state rewind is `runner.prefill()`, which calls `reset()`.

## Results

Steady state at `code-p1024` is flat and reproducible:

| run | steps | prefill | wall ms/token | tok/s | host ms/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| `wall-split-production.json` warmup | 512 | 4770.15 | 52.76 | 18.95 | 2.69 |
| `wall-split-production.json` rep1 | 512 | 4201.90 | 54.01 | 18.52 | 3.34 |
| `wall-split-production.json` rep2 | 512 | 4205.48 | 53.78 | 18.59 | 3.05 |
| `wall-split-settle.json` rep1 (5 s idle) | 512 | 4193.45 | 50.50 | 19.80 | 2.32 |

Per-step bands (`wall-split-production.json` rep1, 512 steps in eighths of 64):

```
79.18  49.76  49.97  50.24  50.39  50.60  50.82  51.10
```

Steady state is **49.8-50.5 ms/token (~19.8-20.1 tok/s)**, flat from step 32 on,
drifting +1.2 ms over 480 steps (context growth). The wrapper host cost is
**2.7-3.3 ms/token**, i.e. the row's own ~3.6 ms/token estimate, and ~5-6% of a
step.

## The finding: a per-prefill transient, not a per-token cost

Each repeat opens with a decaying transient over roughly the first 32 steps
(`wall-split-perstep.json`):

```
[rep1] step ms by eighth: 135.48  63.63  49.89  49.72  49.73  49.82  50.23  49.98
[rep2] step ms by eighth: 144.42  65.91  49.77  49.76  50.02  49.85  49.90  49.96
[rep3] step ms by eighth: 145.99  65.66  51.77  49.74  49.83  49.86  50.01  49.93
```

Raw first steps: `153 149 148 152 164 146 144 145 148 145 149 155 151 75 72 72 69 65 63 64`

That is ~1.5-1.8 s per prefill, concentrated in the first ~32 steps, and it is
**absent in the warmup** (the first loop after model load), whose bands are flat
from step 0. The row's untested hypothesis was state rewind; this reproduces the
signature on the `reset()` path, without the snapshot mechanism.

Four candidate mechanisms were tested and excluded:

| candidate | test | result |
| --- | --- | --- |
| unsynchronized prefill tail | time prefill across `device_synchronize()`, then re-sync immediately | prefill 4.2-4.8 s; **`resync` = 0.00-0.01 ms** (`wall-split-resync.json`) - nothing is queued |
| queued device work | same as above | excluded by `resync` = 0 |
| clock throttling | sample `pp_dpm_sclk` / `pp_dpm_mclk` per 5 ms as sixteenth-bands | **flat 2900 MHz / 1000 MHz in every band** (`wall-split-clocks.json`) |
| pure host blocking | busy counter during the transient | busy mean **82.0%** vs 95.4% with an idle; GPU idles ~11.5 ms/token, but pure host blocking over 32 steps would need ~37% idle, so the transient is a **mix** |

A 5 s idle between the synchronized prefill and the decode loop removes the
transient entirely (`wall-split-settle.json` rep1 `49.79 49.69 49.82 ...`), but
it is **not a fix**: same loop, no-settle `4.208 + 512x53.94/1000 = 31.8 s`
versus settle `4.193 + 5.0 + 512x50.50/1000 = 35.1 s`, a net **+3.3 s**. The
work is relocated, not removed, so idling must not be adopted as an optimization.

## Path-independence check

An earlier revision of this probe used the runner's default `capture_logits=True`,
which routes each step through `_finish_lm_head`'s `device_synchronize()`, a
full-vocab logits D2H copy and a host `np.argmax`. Every canonical bench and
probe instead uses the device-argmax boundary (`capture_logits=False`,
`token_id_resident=True`), so the probe was corrected and re-run. The two
boundaries agree within noise at this shape (production `54.01 / 53.78` versus
diagnostic `53.94 / 53.79`), and **the transient is present on both**, so it is a
property of the decode path rather than of the logits-capture boundary.

## Limits

One case (`code-p1024`), one prompt, one model, one quant, one host, one profile.
The busy counter's averaging window is not documented by the driver, so
busy-derived shares are approximate and the per-step wall series is the primary
evidence. The transient's magnitude is measured on 128- and 512-transition
windows; its behaviour at longer horizons is inferred from the 512-step run
holding steady after step 32. No mechanism inside `reset()` is isolated here -
`state.zero()`, the per-attention KV memsets, `index.reset()` and the
`_ple_hash_states` clear are all candidates and none is individually excluded.

## Status

**No performance claim and no default changed.** This is a re-derivation of an
existing diagnostic row. DE-5's premise of a large per-token wrapper host cost is
not supported: host time is 2.7-3.3 ms/token, matching the row's own ~3.6
ms/token. The actionable open item is the per-prefill transient, now localized to
the `reset()`-then-decode path rather than to the x4 harness's snapshot restore.
