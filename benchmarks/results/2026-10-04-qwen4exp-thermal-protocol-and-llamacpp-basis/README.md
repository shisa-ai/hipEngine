# The benchmark protocol on gfx1151, and the real llama.cpp comparison

Two findings that together change how this campaign must measure. Both are
methodology, not engine changes: **no kernel, default, flag or dispatch
changed, and no metric row moved.**

## 1. Sustained load thermally throttles this host, and long in-process suites measure the throttle

The Strix Halo APU reaches **98 °C** under sustained prefill load and loses
roughly **40% of prefill throughput**. It cools from 98 °C to 39 °C in about
**two minutes** once the load stops.

The evidence is inside a single 12-case suite run (chunk 1024, one process,
warmup + 3 repetitions per case, cases in fixture order). Throughput is tight
and consistent for the first ~36 samples and then steps down abruptly:

| sample | case | prefill tok/s | decode tok/s |
| --- | --- | ---: | ---: |
| `[measure 0]` | code-p512 | 264.88 | **20.98** |
| `[measure 0]` | code-p1024 | 262.09 | 19.46 |
| `[measure 1]` | code-p4096 | 249.14 | **17.68** |
| `[measure 1]` | mixed_ja_en-p4096 | 249.55 | **13.79** |
| `[measure 2]` | code-p4096 | 248.26 | **12.42** |
| `[measure 2]` | general_ja-p4096 | 254.19 | **10.14** |

The last three rows are the same case mix as the rows above them, minutes
apart, in the same process. Decode falls **17.68 → 10.14** and prefill falls
**~265 → ~248**.

**Temperature alone is not the trigger; sustained load is.** Three consecutive
`code-p4096` runs in separate processes, each starting at 48 °C and ending at
53-54 °C, are indistinguishable:

| run | temp start | temp end | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 39 °C | 54 °C | 269.66 / 269.76 / 269.77 | 17.46 / 17.48 / 17.57 |
| 2 | 48 °C | 53 °C | 270.45 / 270.21 / 270.10 | 17.48 / 17.41 / 17.53 |
| 3 | 48 °C | 54 °C | 270.97 / 270.65 / 270.99 | 17.45 / 17.41 / 17.47 |

A single-case process runs for ~90 s and never exceeds ~56 °C. The 12-case
suite runs 48 back-to-back passes and does.

**There is also a slower in-process effect that is not the 98 °C throttle.**
The same case measured in a fresh process and inside a suite that had already
run eight passes:

| case | inside the suite | fresh process | ratio |
| --- | ---: | ---: | ---: |
| code-p512 | 264.88 | 264.0 | 1.00 |
| code-p1024 | 262.09 | **287.24** | 0.91 |
| code-p4096 | 249.17 | **270.69** | 0.92 |

code-p512 is unaffected, code-p1024 and code-p4096 lose ~9%. **Prefill is the
sensitive metric and decode is not**, which is the signature of a
bandwidth-hungry phase losing memory throughput rather than a latency-bound
one - but that is an inference, not a measurement, and it is not established
here.

### Required protocol

- **One case per process**, with a cooldown to ≤50 °C before each run.
- Never compare a fresh-process number against an in-suite number.
- Record temperature at the start and end of every run.
- Any 12-case suite run in a single process is a **thermal trajectory**, not a
  measurement, and its later cases are not comparable with its earlier ones.

This also reframes earlier results in this project: the ledger's baseline row
(`239.5 / 253.4 / 241.9` at p512 / 1K / 4K) and any suite-shaped measurement
taken in one process carry an unknown thermal component.

## 2. The real llama.cpp comparison

The campaign's first milestone is to match **llama.cpp**. The ledger's
reference row (`6011c34ce`, 321.3 / 415.1 / 459.0) is **not reproducible**: that
commit is not present in the local llama.cpp checkout, whose HIP build is at
`002a12ad2`. Measured under the protocol above - same host, same model file,
same one-shape-per-process structure, `llama-bench -ngl 999 -fa 1 -r 3`:

| shape | llama.cpp `002a12ad2` | hipEngine in-tree | ratio |
| --- | ---: | ---: | ---: |
| pp512 | 385.97 ± 43.58 | 264.0 | **0.684x** |
| pp1024 | 421.60 ± 1.41 | 287.2 | **0.681x** |
| pp4096 | 421.87 ± 2.85 | 295.0 (chunk 4096) | **0.699x** |
| tg128 at p512 | 21.85 ± 0.12 | 20.98 | **0.960x** |
| tg128 at p1024 | 22.04 ± 0.07 | 19.40 | **0.880x** |
| tg128 at p4096 | 21.89 ± 0.14 | 17.40 | **0.795x** |

hipEngine rows are the thermal-controlled fresh-process measurements below.
llama.cpp's pp512 carries a ± 43.58 spread (11%) over three repetitions, which
is the same instability this host shows elsewhere; its pp1024 and pp4096 do not
(± 1.41 and ± 2.85).

**The gap is 0.68-0.70x on prefill and 0.80-0.96x on decode.** That is a
completely different campaign from the 5.78x gap to gufo: gufo is not
llama.cpp, and the objective names llama.cpp first.

### Decode: llama.cpp is flat across context, and hipEngine's context trend is not established

llama.cpp's decode is **flat** across prompt length (21.85 / 22.04 / 21.89) and
so is its prefill (385.97 / 421.60 / 421.87). hipEngine's decode, measured by
`qwen4exp_canonical_ar_bench.py`, **falls 17%** (20.98 → 19.40 → 17.40) and its
prefill is non-monotone (264.0 / 287.2 / 295.0).

**That hipEngine decode trend is reported here as measured but not as
established, because a second harness contradicts it.**
`qwen4exp_profile_gap.py --mode decode` at p512 and p4096, with an explicit
`--max-sequence-length` of 648 and 4232 and a cooldown before each, reports
**19.46 and 20.07 tok/s** - flat, and in the opposite direction. A kernel trace
of those two runs shows the device work is context-independent:

| quantity (16 decode steps) | p512 | p4096 |
| --- | ---: | ---: |
| decode launches | 44495 | 44482 |
| total kernel time | 1081.8 ms | 1075.9 ms |
| `qwen35_paged_full_attn_decode_context` | 1.790 ms (n=288) | 1.799 ms (n=288) |
| `gguf_k_pack8_prefill_out_kernel` | 563.3 ms | 557.2 ms |

The two harnesses measure decode differently - the canonical bench runs 128
`runner.step()` calls in one tight Python loop with a single
`device_synchronize()` at the end, while `profile_gap` runs 16 steps with 8
warm steps - and the canonical bench's p4096 decode rows were taken inside the
12-case in-process suite whose thermal trajectory is documented above. **Which
of the two is right is unresolved.** Until it is settled, the decode ratio to
llama.cpp should be read as 0.80-0.96x with an unknown trend, not as a
decode-context deficit.

What the trace does establish, as a same-protocol comparison, is that
**`gguf_k_pack8_prefill_out_kernel` is 52% of decode kernel time at 786 launches
per step**, and that a decode step issues **~2781 kernel launches**. Both are
context-independent and both are candidates for the decode gap.

> **Correction.** An earlier revision of this file claimed the 17% decode fall
> was context-driven and called it the most actionable finding. A follow-up
trace did not reproduce it. The claim is withdrawn; the measurement stands and
> its cause is open.

## Measured hipEngine baseline (thermal-controlled, in-tree, fresh process per case)

`uv run`, production profile, `code-p4096`/`code-p1024`/`code-p512` case,
warmup + 3 repetitions, `--require-cached-build`, cooldown to ≤50 °C before
each.

| case | chunk | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: |
| code-p512 | 512 | 265.18 / 265.40 / 265.17 | 21.014 / 21.012 / 21.011 |
| code-p512 | 1024 | 263.89 / 264.02 / 263.97 | 20.984 / 20.971 / 20.986 |
| code-p1024 | 1024 | 287.24 / 286.99 / 286.91 | 20.219 / 19.509 / 19.397 |
| code-p1024 | 2048 | 286.94 / 286.85 / 286.94 | 20.257 / 19.568 / 19.421 |
| code-p4096 | 1024 | 270.69 / 270.44 / 270.58 | 17.333 / 17.398 / 17.490 |
| code-p4096 | 4096 | **294.43 / 295.01 / 295.29** | 17.707 / 17.643 / 17.707 |

**The chunk effect is shape-dependent, and only the whole-prompt chunk wins:**

| case | smaller chunk | larger chunk | delta |
| --- | ---: | ---: | ---: |
| code-p512 | 265.18 (512) | 263.89 (1024) | **-0.5%** |
| code-p1024 | 287.24 (1024) | 286.94 (2048) | **-0.1%** |
| code-p4096 | 270.69 (1024) | 295.01 (4096) | **+9.0%** |

So a blanket larger chunk is not a win, and a blanket smaller one is not
either. The +9% appears only when the chunk covers the whole prompt. This
supersedes the chunk sweep in
`2026-10-04-qwen4exp-chunk-order-balanced`, whose arms ran against a different
code tree.

## Limits

- One host (`gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`), one
  model file, one quant, one execution profile, one category (`code`) per
  shape. The llama.cpp rows are one build (`002a12ad2`), not the ledger's
  `6011c34ce`, and `llama-bench` and the hipEngine harness do not share a
  tokenizer or a prompt, so this is same-shape rather than same-prompt.
- Temperature is read from the APU's `temp1_input` only. No per-domain
  (CPU/GPU/memory) thermal or clock telemetry was captured, so the mechanism
  behind the ~9% in-process prefill loss is inferred from which metric moves
  and is not established.
- The `-0.5%` and `-0.1%` chunk deltas are at or below the within-run spread of
  the arms they compare; they are reported as "no gain", not as regressions.
- hipEngine decode rows at p1024 and p4096 show a first-repetition outlier
  (20.22 then 19.51/19.40) that is not explained; the median is quoted.
