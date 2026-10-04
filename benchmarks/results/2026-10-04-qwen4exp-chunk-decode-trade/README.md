# The Qwen4Exp prefill chunk trades +9.06% prefill for -1.15% decode

**Result: interleaved at `code-p4096`, chunk 1024 against chunk 512 is +9.06%
prefill and -1.15% decode, with every arm fully separated and within-arm spread
of 0.05-0.48%. The same comparison run sequentially reads -7.1% decode, because
a non-interleaved `code-p4096` A/B on this host drifts 8%.**

Date 2026-10-04. Host gfx1151 (Strix Halo, Radeon 8060S, machine
`55ea6c509d0b49eea8de7094a1023668`). Model
`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, UD-Q4_K_XL, production profile, warm
PLE, BF16 KV, 128 decode transitions, HEAD `e04260e5d`, under `uv run`.
Harness `scripts/qwen4exp_canonical_ar_bench.py hipengine --case-id code-p4096
--prefill-chunk-size <C>`, one process per arm, 1 warmup + 3 repetitions.

## The interleaved measurement

A/B/A/B/A/B in one session, 150 s cooldowns:

| chunk | prefill tok/s | within-arm spread | decode tok/s | within-arm spread |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 249.76 (249.70 / 249.76 / 249.83) | 0.051% | 18.120 (18.110 / 18.118 / 18.133) | 0.127% |
| 1024 | 272.38 (271.60 / 272.78 / 272.77) | 0.433% | 17.911 (17.857 / 17.935 / 17.942) | 0.479% |

**chunk 1024 against 512: prefill +9.06%, decode -1.15%.** No sample overlaps on
either column (prefill 249.70-249.83 against 271.60-272.78; decode 18.110-18.133
against 17.857-17.942).

## Why the sequential reading was wrong

Five sequential runs, 75-90 s cooldowns, same harness and host:

| # | chunk | prefill tok/s | decode tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 512 | 262.41 | 19.603 |
| 2 | 1024 | 291.18 | 18.930 |
| 3 | 2048 | 308.44 | 18.549 |
| 4 | 4096 | 295.59 | 18.222 |
| 5 | 512 (repeat of run 1) | 249.83 | 18.065 |

Runs 1-4 are monotone in decode and read as a -7.1% chunk cost. **Run 5 is a
repeat of run 1 and returns the session's lowest decode**, while the prefill
column keeps its chunk shape and peaks at chunk 2048. Decode tracked run order;
prefill tracked chunk.

The interleaved session's chunk-512 prefill (249.76) reproduces run 5 (249.83)
rather than run 1 (262.41), so the sequential session's **first** run is the
outlier. A fresh-process `code-p4096` run is the first measurement of a ~90 s
APU load, and this host's sustained-load behaviour is already recorded
(`benchmarks/results/2026-10-04-qwen4exp-thermal-protocol-and-llamacpp-basis`,
98 °C under sustained prefill, ~40% prefill loss, 98 -> 39 °C in about two
minutes). That packet's required protocol fixes the **start** temperature per
run; it does not fix the **between-arm ordering**, which is what went wrong here.

**Interleaving took within-arm repeatability from 8.0% to 0.05-0.48%.** A
non-interleaved `code-p4096` comparison on this host cannot resolve anything
below roughly 8%.

## Cross-session agreement where it should hold

The interleaved chunk-1024 prefill (272.38) sits 0.03% from the value recorded in
`benchmarks/CHANGELOG.md` for this tree (272.46). The sequential `code-p512` and
`code-p1024` runs at chunk 1024 (265.33 / 20.985 and 286.64 / 19.901) sit within
0.5-1.0% of the thermal packet's fresh-process values (263.96 / 20.98 and
287.05 / 19.70). The absolute `code-p4096` level in this session is lower than
that packet's (249.76 against 270.57), which is why this artifact claims the
within-session paired deltas and not absolute rates.

## What this changes

**`RT-1`'s promoted chunk default is validated on both axes.** `RT-1` promoted
chunk 1024 (serving) and qualified 2048/4096 on **prefill-only** evidence - its
own state cell and the scoreboard's chunk section both say so ("It is not a
decode-latency-under-concurrency claim"). The decode cost is now measured at
**-1.15%**, eight times smaller than the **+9.06%** prefill benefit. No default
changed and no metric row moved.

**`DE-11`'s decode non-kernel remainder should be re-derived from a paired
capture.** `DE-11` uses decode walls of 47.656 ms/step at p512 and 57.694 at
p4096 and subtracts a separately captured kernel sum to get a remainder that
"grows 7x". Those walls are this thermal packet's chunk-1024 readings (20.98 and
17.33 tok/s). The p512 value reproduces here to 0.02% (20.985) but p4096 reads
**18.930**, 8.7% higher - at or inside the 8% sequential drift. The remainder
inherits that drift; measuring wall and kernel sum in one interleaved capture is
the fix.

## Artifacts

| file | what |
| --- | --- |
| `ab-pass{1,2,3}-chunk{512,1024}.json` | the six interleaved arms |
| `sequential-chunk4096.json` | sequential run 4 |
| `sequential-code-p512-chunk1024.json` | sequential cross-check, p512 |
| `sequential-code-p1024-chunk1024.json` | sequential cross-check, p1024 |

The sequential chunk-512 / 1024 / 2048 runs and the chunk-512 repeat are the
entry's own; only the runs cited above were retained.

## Limits

One host, one model, one quant, one profile, one case (`code-p4096`), one
architecture, 3 repetitions per arm, chunk 512 against 1024 only. The 2048 and
4096 points exist only in the confounded sequential session and are **not**
carried as measured rates. `decode_tok_s` is the canonical harness's own
128-transition weighted rate; no kernel-level attribution is claimed here.
