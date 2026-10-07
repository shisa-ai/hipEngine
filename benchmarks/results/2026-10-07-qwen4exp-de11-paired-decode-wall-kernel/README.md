# DE-11: the decode wall and kernel sum re-derived in one interleaved session

`DE-11` in `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md` reports the decode non-kernel
remainder as `wall - kernel_sum`, but the wall came from one session and the
kernel sum from another, and a sequential `code-p4096` comparison on this host
drifts up to 8%. This packet re-derives the pair from **one interleaved session**
and finds that the recorded `7x` remainder growth is a cross-session artifact.

## Result

Production profile, `code-p512` / `code-p4096`, chunk 1024, warm PLE, 128 decode
transitions, one process per arm, counterbalanced `A/B/A/B -> B/A/B/A`, cooldown
to <=50 C before each arm, HEAD `7d97832d3`, model
`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL` UD-Q4_K_XL, host gfx1151 (Strix Halo,
Radeon 8060S, machine `55ea6c509d0b49eea8de7094a1023668`).

| case | wall ms/step | kernel sum ms/step | remainder ms/step | kernel share | launches/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `code-p512` | **47.700** (47.744 / 47.655) | **44.242** (44.228 / 44.256) | **3.457** | 92.8% | 1740 |
| `code-p4096` | **55.946** (55.844 / 56.048) | **45.050** (45.048 / 45.053) | **10.896** | 80.5% | 1815 |

Within-arm spread is 0.01-0.36%. The wall grows **+8.246 ms/step (+17.3%)**, the
kernel sum grows **+0.808 (+1.8%)**, and the remainder grows **+7.439 (3.15x)**.

**This corrects the p512 remainder, not the p4096 one.** The p4096 remainder
reproduces the ledger's `10.56` within 3%, but the p512 remainder is **3.46, not
1.47**, and the p512 step is **92.8% kernel, not 96.9%**. The same-session
remainder grows **3.15x**, not `7x`.

## Why the p512 kernel sum moved: the DE-3 lever

The p512 wall is unchanged from the ledger's (47.700 against 47.656, 0.09%), but
this session's p512 kernel sum is **1.95 ms/step lower** (44.242 against 46.193).
The dominant family explains it: `gguf_k_pack8_prefill_out_kernel` is **2851.6 ms**
over 128 steps at the default, and the ledger's is **3015.8 ms**.

`DE-3`'s `HIPENGINE_GGUF_PACK8_GEMV_WIDE` lever (default on since 2026-09-30) is
the difference. One extra profiled p512 capture with the lever off:

| arm | p512 kernel sum ms/step | `gguf_k_pack8_prefill_out_kernel` ms / 128 steps |
| --- | ---: | ---: |
| `HIPENGINE_GGUF_PACK8_GEMV_WIDE=1` (default) | 44.242 | 2851.6 |
| `HIPENGINE_GGUF_PACK8_GEMV_WIDE=0` | 45.758 | 3039.9 |
| ledger `bench-decode-rep-trace.json` | 46.193 | 3015.8 |

The lever-off capture reproduces the ledger's kernel time to 0.8% while the
default is 5.8% below it, and the ledger's **wall is the lever-on wall** (47.656;
the measured lever-off wall is 49.201). So the recorded `1.47 ms/step` p512
remainder is a **lever-on wall minus a lever-off kernel sum** - the same class of
apples-to-oranges subtraction the ledger's retracted third correction suspected,
but driven by the `DE-3` block-size lever rather than the `DE-1` promotion. The
p4096 kernel sum carries the same ~1.3 ms/step pack8 difference.

**Caveat.** The ledger's walls were captured at head `8c6f2bc94` with production
manifest `a55324f9...`; this session is head `7d97832d3` with manifest
`be32e719...`, so the production dispatch changed between them. The lever-off
reproduction is the best available explanation of the pack8 difference and not a
proof that the ledger capture had the lever off.

## Method

- **Wall.** `scripts/qwen4exp_canonical_ar_bench.py hipengine --case-id <case>
  --prefill-chunk-size 1024 --warmups 1 --repetitions 3`, unprofiled. The per-step
  wall is `samples[].decode_ms / 128` (the synchronized 128-step loop); the
  reported value is the median over three measured repetitions, and the two runs
  are the two counterbalanced arms.
- **Kernel sum.** The same command with `--warmups 1 --repetitions 1` under
  `rocprofv3 --kernel-trace -f csv`. `rocprofv3` kernel-dispatch timestamps are
  `CLOCK_MONOTONIC`, the same clock as the bench's `time.perf_counter_ns`, so the
  measured sample's own `phase_windows_ns["decode"]` slices the trace directly.
  The kernel sum is the total `DurationNs` of the dispatches contained in that
  window, divided by 128. The profiled arm's own wall (48.9 / 56.3 ms/step) is
  2.5% / 0.6% above the unprofiled wall, so the profiler inflation is small; the
  profiled wall is never reported as the engine wall.
- **Join.** `scripts/qwen4exp_de11_paired_decode_wall_kernel.py session <dir>`
  writes `paired-summary.json`. The driver is `run_interleaved.sh`.
- **Environment.** All arms pin `PYTHONPATH=/home/lhl/hipEngine` and a pinned
  `--compiler-version-file` with `--require-cached-build`. The `therock` env's
  editable `.pth` points `hipengine` at `/home/lhl/hipEngine-main`, so a run that
  does not pin `PYTHONPATH` silently imports the wrong tree; that trap was hit and
  corrected before any reported arm.

## Files

| file | what |
| --- | --- |
| `paired-summary.json` | the joined wall / kernel / remainder rows |
| `wall/*.json` | per-arm wall summaries (median over 3 reps) |
| `kernel/*.json` | per-arm decode-window kernel sums |
| `wall-raw/*.json` | the raw canonical-bench wall artifacts |
| `diag-wide0-p512-kernel.json`, `diag-wide0-p512-bench.json` | the lever-off diagnostic |
| `run_interleaved.sh` | the session driver |
| `compiler-version.txt` | the pinned `hipcc --version` text used by every arm |

Raw `rocprofv3` trace CSVs (114 MB each) are not retained; they are regenerable
from the driver.

## Limits

- One host (`gfx1151`, not the gfx1100 default hardware), one model, one quant,
  one execution profile, one category (`code`), two depths, chunk 1024, two wall
  runs and two kernel runs per depth.
- The kernel sum is a sum of dispatch `DurationNs` and assumes no inter-kernel
  overlap; it is a device floor, not a wall.
- The `DE-3` attribution is a **reproduction** (lever-off 3039.9 against the
  ledger's 3015.8, 0.8% apart) and not a record of the ledger capture's
  environment. The production manifest also differs between the two sessions
  (`a55324f9` against `be32e719`), so the tree changed; the residual ~0.4 ms/step
  between lever-off and the ledger total is within session variation.
- No performance claim and no default changed.
