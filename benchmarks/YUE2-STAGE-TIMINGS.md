# YuE2 stage timing and tuning evidence

Detailed stage comparisons from the September 17, 2026 benchmark runs.
The [topline scoreboard](README.md#radeon-8060s-yue2-3b-end-to-end-product-path)
keeps the end-to-end results. The sections below preserve the original evidence;
references to the current build describe those recorded runs, not a fresh measurement.

### Radeon 8060S: YuE2 3B stage-by-stage matched timing

Every earlier stage comparison in this file put one side's warm measurement against the other side's
recorded product run, which mixes in per-request overheads and, for the decoder, an 8x first-use
cost. These three harnesses give both sides identical inputs and compare steady-state passes, timed
alternately in one session on `mandarin-off-s1234` at the product's 32 ODE steps:

| Stage | Inputs shared | hipEngine | torch reference | Reading |
| --- | --- | ---: | ---: | --- |
| AR decode, 1 297 steps | recorded token trajectory, both prefixes, 2 CFG branches | 54.94 ms/step | 30.88 ms/step | **reference 1.78x faster** |
| Acoustic solver, 32 steps | prefix, codes, seed and the reference's own noise | 27.35 s | 21.81 s | **reference 1.25x faster** |
| FP32 Oobleck decode, 1 297 frames | the case's recorded latents, tiled 1024/16 on both sides | 20.08 s | 20.15 s | reference 1.00x faster (level) |

Every row above is a same-session re-measurement of both sides at current HEAD, on
`mandarin-off-s1234` at the product's 32 ODE steps. The AR row is the fixed-token replay: both sides
drive the same recorded trajectory, so sampling is excluded by design (the token is recorded) and
the number is model work only. The replay uses the full-vocabulary head, which is what the
reference's own rows are, so the windowed head and the window-relative sampler do not appear in it;
the product loop below is where they show up.

The AR step was 58.30 ms/step before the paired-branch head (see the next section): the lm head runs
once for both branches through a two-row F32-output GEMV, which is bit-identical per row, leaves the
run's trajectory digest unchanged, and took the step to 54.88, against 54.94 on the re-measurement
above.

Each harness refuses to report when the two sides did not run the same work: the AR harness compares
prefix, negative-prefix and trajectory digests, and the solver harness compares the initial-noise
digest after being handed the reference's own whole-song draw (the two implementations' seeded draws
differ by design, and the first version of this comparison silently compared two different
trajectories). The solver's latent norm is 287.400 against 287.113 on that shared noise, which is
the arithmetic-parity cross-check the M4 gate makes on recorded fixtures. Our tiled decode is
bit-identical to our own full decode (max abs 0.0) while the reference's differ by 1.4e-06, its own
FP32 noise floor.

So on matched, steady-state work this runtime is behind on two stages and level on the third. What
the product path does differently is overhead: the reference moves its decoder off the device after
every request and back on for the next one, and its recorded product decode of these same latents is
69.09 s against its own warm 19.60 s, while this runtime pays about 8 ms per token of host sampling
that the reference does with device ops. Evidence:
[`matched AR timing`](results/yue2_ar_matched_timing_20260917.json),
[`matched solver timing`](results/yue2_nar_matched_timing_20260917.json),
[`matched decoder timing`](results/yue2_vae_matched_timing_20260917.json).

### Radeon 8060S: YuE2 3B paired-branch projections

The AR decode runs two CFG branches over one set of weights, and the profile of a step is 79%
projection GEMVs. The lm head alone streams 756 MB of BF16 weight per call, once per branch.
`dense_gemv_bf16_f32_out_rowtile2` puts two rows in one block so the weight is issued once for both
branches, keeping each row's K traversal and reduction order, which makes a row bit-identical to the
single-row kernel.

| Shape (in x out) | Serial, 2 calls | Two-row, 1 call | Speedup | Serial GB/s | Two-row GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| LM head 2048 x 184704 | 6.723 ms | **3.798 ms** | 1.77x | 112.5 | 199.2 |
| q 2048 x 2048 | 0.060 ms | 0.033 ms | 1.78x | 140.7 | 250.6 |
| o 2048 x 2048 | 0.054 ms | 0.031 ms | 1.77x | 154.3 | 272.7 |
| v 2048 x 1024 | 0.041 ms | 0.028 ms | 1.46x | 102.7 | 150.3 |
| k 2048 x 1024 | 0.028 ms | 0.022 ms | 1.30x | 149.0 | 193.1 |

Per-call latency with a synchronize per call, best of 30, after 5 warmups. In the runtime only the
head is paired so far, which is where the traffic is: the step goes from **58.30 ms to 54.88 ms**
(1.06x) on `mandarin-off-s1234` with the trajectory digest unchanged at `6a308700b3b949ac`, so the
paired head reproduces the serial head exactly. Each branch keeps its own KV spans and its own
positions (this case's positive prefix is 98 tokens against the negative branch's 12); only the
weight read is shared. Evidence: [`kernel A/B`](results/yue2_ar_rowtile_bench_20260917.json),
[`step timing`](results/yue2_ar_matched_timing_20260917.json), gate
`tests/test_unit_yue2_ar_gemv_rowtile2.py`.

### Radeon 8060S: YuE2 3B phase-windowed output head and sampler

`distribution` masks every row outside a phase's domain and the phase's end token to `-inf` before
any other arithmetic, so a phase can only ever select inside its own window: 32 769 rows for
`semantic` (the 32 768 codec tokens plus `MUSIC_END`, 17.7% of the vocabulary) and 151 849 for
`abc`. The head projects just that slice, and the sampler runs in window coordinates, so mask,
repetition penalty, top-k, top-p, softmax and the draw all work over the window instead of the full
vocabulary.

| Phase | Head, full 184 704 rows | Head, windowed | Speedup | Rows | Weight read |
| --- | ---: | ---: | ---: | ---: | ---: |
| `semantic` | 3.84 ms | **0.82 ms** | **4.68x** | 32 769 | 722 MiB → 128 MiB |
| `abc` | 4.13 ms | 3.44 ms | 1.20x | 151 849 | 722 MiB → 593 MiB |

Both head rows are medians of 15 interleaved pairs, with the two arms measured inside one repetition
and the order flipped each repetition; the paired ratio spans 4.41-5.01 for `semantic` and 1.18-1.39
for `abc`. A separate interleaved run gave 4.53x and 1.20x.

| Sampler stage, `semantic` | Full row | Window | Speedup |
| --- | ---: | ---: | ---: |
| mask, penalty, top-k, top-p, softmax, draw | 5.86 ms | **0.43 ms** | **13.5x** |
| the same for `abc` (82.2% of the vocabulary) | 3.37 ms | 2.56 ms | 1.32x |

The production session loop with two-branch CFG, temperature 1.0, top-p 0.95, top-k 100, repetition
penalty 1.2 over a 50-token window on `mandarin-off-s1234`, three arms measured inside one
repetition with the order rotated, 5 repetitions per arm:

| Arm | ms/token | Saved vs baseline |
| --- | ---: | ---: |
| full head, full-row sampler | 61.11 | — |
| windowed head, full-row sampler | 55.14 | 5.97 ms (1.11x) |
| windowed head, windowed sampler | **43.80** | **17.32 ms (1.40x)** |

A second interleaved run measured the same comparison as 60.84 → 43.75 ms/token, 17.10 ms saved,
within 1.3% of the first. The saving exceeds the two head calls' own 6.0 ms because the
device-to-host row and the host-side conversion shrink from 739 KB to 131 KB per branch as well. All
15 runs produced **identical tokens and an identical PCG64 state digest**.

The windowed row is bit-identical to the full projection inside the window and `-inf` outside it for
both phases, and the windowed sampler returns bit-identical scores and softmax probabilities to the
full-row sampler, with the same top-1 and the same drawn token, over 432 cases (both phases x 3
sampling settings x both arithmetic modes) on rows recorded from the pinned reference. Windowed
execution is covered by those checks and by the session gate below; the replay matrix keeps using
the unwindowed call, so it validates that fallback and the paired head rather than the window
itself. Evidence: [`head window validation`](results/yue2_ar_head_window_20260917.json),
[`sampler window validation`](results/yue2_sampler_window_20260917.json),
[`session gate`](results/yue2_session_gate_windowed_sampler_20260917.json).

### Radeon 8060S: YuE2 3B NAR attention packing and projection selection

The NAR attention kernel now owns eight query rows per block and reuses every loaded K/V element
across them, and the projection path selects by measured hipBLASLt index instead of the first
returned algorithm. On gfx1151 (zbook) the scalar attention kernel falls 177.22 ms to **19.33 ms**
per call (**9.17x**), the f16 WMMA tensor-core path runs the production shape at **2.13 ms** against
the scalar 19.0 ms (**9.3x**), and the product 32-step solve drops 443.4 s to **26.07 s**
(**17.01x**). The scalar kernel stays the strict fallback behind
`HIPENGINE_YUE2_NAR_ATTENTION=scalar`; the tensor-core path changes arithmetic by design and is held
to the production gates. Evidence:
[`M7 attention packing`](results/yue2_m7_attention_packed_20260917.json),
[`M7 tensor-core attention`](results/yue2_m7_attention_wmma_20260917.json). Profiling, block-geometry,
projection-ceiling, and parity detail is archived in [`HISTORY.md`](HISTORY.md#radeon-8060s-yue2-3b-nar-attention-packing-and-projection-selection).
