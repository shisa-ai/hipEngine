# Long-context ladder: hipEngine and llama.cpp-family at 8K-256K

Ledger row `X-8` asked for our own context-scaling curve before any long-context
kernel work. This is that curve, with a same-host, same-file, same-token
llama.cpp-family ladder beside it, and it comes back with a warning about its
own decode column: **the continuous-ladder decode numbers are not a depth curve.**
The same case family inside one rung reads 13.53 and 6.79 tok/s minutes apart at
the same prompt size, the step is 2.0-2.8x slower than a settled measurement at
the same prompt size, and the comparator's decode does not move at all across the
same host states.

Prefill is the part of this ladder that is a clean engine measurement, and it is
where the real long-context gap is: **3.8x behind the Strix Halo fork at 8K and
4.9x at 128K** because that fork's prefill is flat while ours falls 42.5% from 8K
to 256K, and **1.8x behind upstream llama.cpp at 8K narrowing to 1.3x at 128K**
because upstream's falls faster than ours.

## Result: prefill and decode, 8K to 256K

AMD Radeon 8060S (`gfx1151`), Framework Desktop, `gguf_ud_q4_k_xl`, production
execution profile, BF16 KV, warm PLE cache, 1024-token prefill chunks, 128
autoregressive decode transitions, one warmup and one measured repetition per
case, one process per rung. The fixture carries exact tokens for the canonical
category prompts, built by `scripts/build_qwen4exp_long_context_fixture.py`.

| Prompt tokens | Cases | Prefill tok/s | Decode tok/s |
| ---: | --- | ---: | ---: |
| 8,192 | 4 categories | 260.5 / 263.2 / 263.3 / 260.9 | 13.53 / 13.51 / 6.79 / 6.77 |
| 16,384 | 4 categories | 255.6 / 258.3 / 258.1 / 256.8 | 6.71 / 6.72 / 6.65 / 6.70 |
| 32,768 | 4 categories | 242.7 / 230.6 / 244.8 / 241.2 | 6.72 / 2.82 / 2.61 / 2.67 |
| 65,536 | code | 208.8 | 2.89 |
| 131,072 | code | 183.8 | 2.90 |
| 262,008 | code | 149.7 | 2.89 |

Order within a rung is `code`, `general_en`, `general_ja`, `mixed_ja_en`.

**Prefill is the clean column.** 260.5 -> 255.6 -> 242.7 -> 208.8 -> 183.8 ->
149.7 tok/s from 8K to 262,008 for the `code` case (the one category present at
every rung), **-42.5%**; across the rung means it is 262.0 -> 149.7, -42.9%. It
declines smoothly, the four categories agree within 6.2% at every rung that has
four, and the GPU is not the variable: in the settled windows below, GPU sclk sits at its 2,900 MHz
maximum in all three, including the one whose host is clamped. This is the number
to plan against.

**Decode is not a depth curve.** The 8,192-token rung contains the fastest decode
in the whole ladder (13.53 tok/s) and, four cases later at the same prompt size,
6.79 tok/s - a **2.0x** spread inside one rung. The 32,768-token rung contains
both 148.8 ms/step (`code`) and 382.5 ms/step (`general_ja`). The `code` category
alone falls 13.53 -> 6.72 -> 2.89 tok/s from 8K to 64K (**4.7x**) and then stops
falling (2.90 at 128K, 2.89 at 262,008), and
[Settled decode](#settled-decode-the-step-with-the-host-held-still) shows the
step at 8K and 32K is actually **identical** (51.6 against 52.4 ms) when the host
is held still. What the decode column measures is the state of the host package,
which the next two sections pin down.

## Settled decode: the step with the host held still

`scripts/qwen4exp_de5_wall_split.py`, device-argmax decode boundary, 512-token
prefill chunks, one 128-step window per depth with a 30 s idle before the window
(the harness records the settle before its warmup decode, so these rows carry
`"warmup": true` and no measured repetition). The sampler reads
`gpu_busy_percent` and every core's `scaling_cur_freq` **inside the measured
window**, so the host state is recorded rather than assumed.

| Prompt tokens | Max core clock in the window | Decode tok/s | Step ms (median) | GPU busy | Implied GPU ms | Implied host ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 3,535-5,149 MHz, mean 5,011 | 19.23 | **51.6** | 79.2% | 41.2 | 10.8 |
| 32,768 | 3,535-5,147 MHz, mean 4,943 | 18.93 | **52.4** | 79.6% | 42.0 | 10.8 |
| 65,536 | **2,000 MHz in all 512 samples** | 5.72 | **170.9** | 39.5% | 69.1 | 105.6 |

Three readings:

- **At 8K and 32K the step is flat: 51.6 against 52.4 ms, +1.5% for 4x the
  context.** The host is at boost for both (the max core averages 4.9-5.0 GHz,
  against a 3.5 GHz floor). This is the shape the GPU work predicts - at these
  depths the attention term is small against the dense weights.
- **The 64K window could not be settled.** A 310 s prefill leaves the package
  clamped for longer than a 30 s idle: the max core reads exactly 2,000 MHz - the
  base clock - in every one of the window's 512 samples, GPU busy falls to 39.5%,
  and the step is 3.3x the 32K step. Bounding it from that window's own
  decomposition (69.1 ms of device time plus the 10.8 ms of non-overlapped host
  time the boosted steps show) puts a boost-clock 64K step at **~80 ms, ~12.5
  tok/s** - still about 1.5x the 32K step, nothing like the 3.3x the clamped
  reading suggests. That is an estimate, not a measurement.
- **The settle is not a cure for the transient, only for the clamp.** Even in the
  boosted windows the first step costs 2.3-2.5x the steady state (121.1 and 130.9
  ms against 52-53 ms) and flattens by the third step; in the clamped 64K window
  the first step is 493.5 ms, 2.9x its own steady state.

**The same case on a second gfx1151 host, for scale:** the 2026-09-02 owner
profile for `code-p65536` (same quant, same production profile, 512-token chunks,
snapshot-restored, three repeats) reads a clean wall of **82.459 ms/token** on the
`zbook` lane. That is 2.1x faster than this host's clamped settled step and 4.2x
faster than this host's continuous-ladder step for the same shape. It is an
independent lane, not a same-host comparison, and it is cited only as the scale
of the host effect.

## Host state: what moved, and what did not

Two instruments, in two windows that together cover everything above.

### A. Requested clock during the hipEngine ladder

`clock_watch.py` samples all 32 cores' `scaling_cur_freq` every ~15 s;
`host-clock-watch.jsonl` holds 360 samples from 16:46:57 to 18:16:46 UTC, which
covers the **16,384 / 32,768 / 65,536 / 131,072 rungs of the ladder above**
(291 samples) and the ~17 minutes after it.

| Window | Samples | Max core clock | Median core clock | GPU busy (median) |
| --- | ---: | --- | --- | ---: |
| p16384 rung | 30 | **2,000 MHz in all 30** | 2,000 MHz | 96% |
| p32768 rung | 101 | **2,000 MHz in all 101** | 2,000 MHz | 95% |
| p65536 rung | 53 | **2,000 MHz in all 53** | 2,000 MHz | 90% |
| p131072 rung | 107 | **2,000 MHz in all 107** | 2,000 MHz | 91% |
| After the ladder | 68 | 2,000 MHz median, 3,530 max | 2,000 MHz | 96% |

**Every one of the 291 samples inside those four rungs reads the base clock.**
A dedicated 300 s recovery probe after the load agrees: the max core reaches
3,500 MHz or more in 16 of 60 samples, peaks at 3,603 MHz, and is still at
exactly 2,000 MHz in 19 of them.

### B. Delivered speed, and the comparator control

`scaling_cur_freq` is the *requested* P-state, so the delivered-speed instrument
is a fixed CPU-bound workload (dict and environment lookups, string work).
`cpu_speed_watch.py` runs one workload pass and samples the GPU busy counter per
interval; 423 samples cover 18:34-19:33 UTC, **the window of the comparator rungs
below**.

| Host state | Fixed workload | Max core clock | GPU busy |
| --- | ---: | ---: | ---: |
| Between runs | **116-118 ms** (median) | 3,543-3,586 MHz | 0-3% |
| During a comparator rung | **730-762 ms** (median) | exactly 2,000 MHz | 94-98% |
| During the 256K hipEngine rung | **5,800-7,400 ms** | exactly 2,000 MHz | 95% |

The identical workload moves **6.4x** between idle and a comparator rung and
**50-64x** during the 256K rung, where the package also reads 89.8 C (CPU) and
90.0 C (GPU) at 165 W. The requested clock moves by only 1.78x across the first
of those, so the clock reading understates the effect; the delivered measurement
is the one to use. There is no single "load" host state on this machine to
normalize against - only a range from 1x to 64x.

## Comparators: two llama.cpp-family builds, same host, same file, same tokens

Both run through the same harness against the same fixture token arrays,
`-ngl 999 -fa on -b 8192 -ub 2048 -t 4`, context sized to the rung, zero warmups
and one measured repetition per case.

pwilkin `strix-halo` `40a9f4d01` (f16 KV, as that fork requires):

| Prompt tokens | Cases | pwilkin prefill tok/s | pwilkin decode tok/s | pwilkin step ms |
| ---: | --- | ---: | ---: | ---: |
| 8,192 | 4 categories | 851.3 / 1034.3 / 1028.9 / 1016.4 | 23.32 / 17.56 / 21.18 / 20.88 | 42.9-56.9 |
| 32,768 | 4 categories | 919.1 / 1019.7 / 1014.7 / 1011.6 | 21.39 / 19.90 / 19.53 / 18.21 | 46.8-54.9 |
| 131,072 | 4 categories | 880.7 / 900.8 / 904.8 / 891.1 | 20.90 / 19.89 / 18.14 / 18.43 | 47.8-55.1 |

upstream llama.cpp `6011c34ce` (bf16 KV, the 2026-09-15 protocol):

| Prompt tokens | Cases | upstream prefill tok/s | upstream decode tok/s | upstream step ms |
| ---: | --- | ---: | ---: | ---: |
| 8,192 | 4 categories | 455.0 / 455.5 / 472.2 / 465.2 | 18.33 / 17.88 / 17.80 / 17.75 | 54.5-56.4 |
| 32,768 | 4 categories | 376.4 / 389.0 / 384.6 / 374.0 | 13.58 / 12.82 / 13.56 / 13.88 | 72.0-78.0 |
| 131,072 | 4 categories | 229.7 / 235.6 / 236.3 / 234.0 | 6.24 / 6.22 / 6.24 / 6.27 | 159.5-160.7 |

Side by side, with hipEngine's decode taken from the settled probes (51.6 and
52.4 ms) where they exist and from the continuous ladder (345.3 ms) at 128K:

| Prompt tokens | hipEngine prefill | upstream prefill | pwilkin prefill | hipEngine step | upstream step | pwilkin step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 262.0 | 462.0 | 982.7 | **51.6 ms** (settled) | 56.1 ms | 47.5 ms |
| 32,768 | 239.8 | 381.0 | 991.3 | **52.4 ms** (settled) | 73.7 ms | 50.7 ms |
| 131,072 | 183.8 | 233.9 | 894.4 | 345.3 ms (clamped) | 160.3 ms | 52.2 ms |
| 262,008 | 149.7 | not run | not run | 346.5 ms (clamped) | not run | not run |

Prefill ratio against us, by rung:

| Prompt tokens | upstream / hipEngine | pwilkin / hipEngine |
| ---: | ---: | ---: |
| 8,192 | 1.76x | **3.8x** |
| 32,768 | 1.59x | **4.1x** |
| 131,072 | 1.27x | **4.9x** |

Five things follow.

1. **The prefill gap has two very different sizes, and the larger one belongs to
   the fork tuned for this part.** We are 1.8x behind upstream llama.cpp at 8K
   and 1.3x behind it at 128K, against 3.8x and 4.9x behind pwilkin's
   `strix-halo`. Ours falls 42.5% from 8K to 262,008, upstream's falls 49.4%
   (462.0 -> 233.9) over the same span, and pwilkin's is flat (982.7 -> 894.4,
   -9.0%). So the prefill work is real but its size depends on which build is the
   bar, and pwilkin's flatness is the property to copy.
2. **pwilkin's decode is flat across every state measured** (42.9-56.9 ms at 8K,
   46.8-54.9 at 32K, 47.8-55.1 at 128K) - including the rungs whose own load
   windows the fixed-workload instrument recorded at 6.4x slower host speed and
   the 2,000 MHz floor. **A step that does not move across a 6.4x host-speed
   change is not host-bound; ours is, and that is the difference between the two
   decode columns.**
3. **Upstream's decode does have a real depth curve, which is the contrast that
   makes the previous point concrete:** 56.1 -> 73.7 -> 160.3 ms from 8K to 128K
   (**+186%**), against pwilkin's +10% over the same span and our settled +1.5%
   from 8K to 32K. A depth curve in this family is therefore measurable, and our
   continuous ladder's 4.7x from 8K to 64K is not one - it is 2.8x at 32K against
   the settled measurement at the same prompt size, and 2.0x at 64K.
4. **In the settled state our decode sits between the two builds**: 51.6 ms at
   8K and 52.4 ms at 32K, i.e. **1.09x faster than upstream and 1.09x / 1.04x
   slower than pwilkin** against the medians of their four cases. There is no
   long-context decode deficit to chase against these comparators at these
   depths; there is a host-state sensitivity to remove, and at 128K our clamped
   345.3 ms is 2.2x upstream's 160.3 ms and 6.6x pwilkin's 52.2 ms.
5. **A build-compatibility finding:** the pwilkin fork asserts F16 KV in its
   qwen4exp graph (`src/models/qwen4exp.cpp:1360`, `GGML_ASSERT(q->ne[0] == 256
   && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 ...)`) and aborts
   during `sched_reserve` on a bf16 KV configuration. The 2026-09-15 comparator
   protocol's bf16 KV is not usable for that fork; the f16 rows above are.

## Why our decode step is host-bound

`cProfile` of the steady decode window, from the same harness at
`code-p65536` (64 steps), against the 2026-10-01 `DE-5` capture at `code-p1024`
(32 steps). Counts are **per decode step**:

| Python calls per step | `code-p1024` | `code-p65536` |
| --- | ---: | ---: |
| `gguf_k_gemv._launch` (K-quant GEMV) | 353 | 353 |
| `launch_gguf_linear` | 533 | 545 |
| `build_hip` (JIT cache key resolution) | 919 | **958** |
| `os.environ.__getitem__` | 12,381 | 12,798 |
| `str.strip` | not in the top eight | 23,510 |
| `_io.open` | not in the top eight | 375 |
| total Python calls | - | 209,933 |

**The per-step call counts are the same at 1,024 and 65,536 tokens**, so depth
adds no host work per step. What changes is what each call costs:
`os.environ.__getitem__` - a frozen C-level dict lookup - goes from 0.338 us to
2.97 us of self time (**8.8x**), and `build_hip`'s cumulative time from 23.5 us
to 330 us per call (**14x**). Both profiles carry `cProfile`'s own per-call
overhead, so the ratios are relative statements, not absolute costs; the
overhead-free calibration is the 6.4x the fixed-workload instrument measures.

The absolute size of the dispatch path is a separate finding and does not depend
on the clock: **~350-545 kernel launches per decode step, each re-resolving a JIT
build cache key** (958 `build_hip` calls, ~12.8k environment lookups, ~375 file
opens and 23.5k `str.strip` calls per step, 209,933 Python calls in total). At
`code-p1024` the whole clean 49.3 ms step is that path with the device at 78.4%
busy behind it (`settle_5s` rep1 in the 2026-10-01 `DE-5` artifact).

## The 256K rung

A 262,144-token prompt cannot be run through this harness: the canonical bench
sizes its allocation at `max(prompt_tokens) + decode_transitions + 8`, so a
128-step decode window needs `prompt_tokens <= 262,008` inside the model's native
262,144 context. The 262,008-token rung is the largest prompt that fits and is
run in place of a literal 262,144.

It is the slowest point of the ladder on both axes: **prefill 149.7 tok/s**
(1,749.9 s for 256 chunks of 1,024 tokens) against 183.8 at 128,072 - **-18.6%
for the last doubling** - and **decode 2.89 tok/s (346.5 ms/step)**, in the same
band as 64K's 2.89 and 128K's 2.90. Its artifact carries `status: completed` with
`prefill_ms: 1749889.39`; the two over-sized rungs' skeletons
(`hipengine-p262144.json`, `hipengine-p262016.json`) record the capacity failure
that set this bound.

## Tuning implications

1. **Long-context decode is not an attention problem on this path.** The step is
   host-dispatch-bound at every depth measured. The existing owner profile prices
   the QSA index-selection family (compressed-key score plus top-k expand) at
   **2.490 ms/step, 3.02%** of that run's own 82.459 ms/token clean wall at 64K
   (1.116 ms, 1.35% at 16K), so QA-4's indexer-scorer epilogues and QA-7's
   selection pricing have a 3% target at the depth where they matter most.
2. **The concrete decode lever is the per-step dispatch path**, and the case for
   it grows with depth: 958 `build_hip` calls and 375 file opens per step, all of
   them re-deriving a build cache key that cannot have changed. Resolving it once
   per launch site, or extending the existing layer-graph cache (currently gated
   to `rows == 1` GDN layers) to the QSA decode layers, attacks the term that
   scales with the host state rather than with context.
3. **The comparator gap to close is prefill**, 3.8x-4.9x behind pwilkin and
   1.3x-1.8x behind upstream, on a GPU-bound path where the host clock does not
   matter. The 2026-09-15 per-role split already names the owners on a single
   `code-p4096` prefill (device time: dense quantized matmul **10,749.6 ms**
   against pwilkin's 1,485.2 on the same packet); this ladder adds the depth
   dimension and shows the gap is not a fixed constant - and that the build whose
   prefill stays flat with depth is the one to copy.

## Evidence

| File | What it is |
| --- | --- |
| `hipengine-p<shape>.json` | hipEngine canonical AR runs, one per rung, with per-case protocol, memory and token hashes |
| `pwilkin-40a9f4d01-p<shape>.json`, `upstream-6011c34ce-p<shape>.json` | the two comparator ladders through the same harness |
| `de5-settle30000-<case>.json` | settled decode probes with in-window host clock and GPU-busy sampling |
| `de5-code-p65536.json` | per-step call/wait split, per-eighth counters and the two cProfile windows |
| `host-clock-watch.jsonl` | requested clock, 360 samples covering the ladder rungs |
| `cpu-speed-watch.jsonl` | fixed-workload delivered speed, 423 samples covering the comparator rungs |
| `hipengine-p262144.json`, `hipengine-p262016.json` | the two over-sized rungs' `status: failed` skeletons, which record the capacity boundary |

The 262,008-token fixture is **not committed** (11 MB); it is deterministic from
the committed canonical fixture plus the model tokenizer:

```bash
python3 scripts/build_qwen4exp_long_context_fixture.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --shapes 8192 16384 32768 65536 131072 262144 --decode-transitions 128
```

The builder proves its own derivation by reconstructing all twelve committed
canonical cases byte-for-byte; the run above used
`sha256 7ee6f480ac147a9c9553974b500701480c23cfe777ac850673d8974aaf9e7f5d`.

Commands, as run (hermetic TheRock gfx1151 environment, `HIPENGINE_HIP_ARCH=gfx1151`,
`GPU_MAX_HW_QUEUES=2`, cached build; `-ctk/-ctv` differ per comparator as noted
above - pwilkin requires f16, upstream ran bf16):

```bash
# hipEngine rung (one per shape; four case ids at <=32K, code only above)
python3 scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --fixture /tmp/q4x-long/qwen4exp_long_context_p8192_p262144.json \
  --case-id code-p131072 --prefill-chunk-size 1024 \
  --execution-profile production --ple-cache-mode warm --ple-random-access off \
  --warmups 1 --repetitions 1 --output <out>.json

# comparator rung
python3 scripts/qwen4exp_canonical_ar_bench.py llamacpp \
  --server-bin /home/lhl/comparators-20260915/build-pwilkin/bin/llama-server \
  --model /home/lhl/models/gguf/.../Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --engine-label pwilkin-40a9f4d01 --fixture <per-rung fixture> \
  --server-arg=-ngl --server-arg=999 --server-arg=-fa --server-arg=on \
  --server-arg=-b --server-arg=8192 --server-arg=-ub --server-arg=2048 \
  --server-arg=-c --server-arg=<shape+512> \
  --server-arg=-ctk --server-arg=f16 --server-arg=-ctv --server-arg=f16 \
  --warmups 0 --repetitions 1 --output <out>.json

# settled decode probe
python3 scripts/qwen4exp_de5_wall_split.py \
  --fixture /tmp/q4x-long/qwen4exp_long_context_p8192_p262144.json \
  --case code-p32768 --steps 128 --repeats 0 --settle-ms 30000 --split-steps \
  --json-out <out>.json
```

## Limits

- **One host, one model, one quant, one execution profile.** The host is an
  integrated APU (not the gfx1100/W7900 default hardware) and its power
  management is part of every number here.
- **One measured repetition per case, no interleaving.** Every decode figure
  from the continuous protocol is a single 128-step window. The clock watcher's
  samples inside a rung are dominated by prefill time (a rung's decode windows
  are 20-45 s against prefills of 30-310 s), so "the rung ran clamped" is
  established for the rung as a whole rather than per decode window; the settled
  probes measure the boosted state directly.
- **The boost-clock 64K decode step is an estimate**, not a measurement: the
  settle arm could not restore boost after a 310 s prefill, so ~80 ms is derived
  from that window's own GPU-busy fraction and the boosted steps' host share.
- **Two comparators, not the whole family.** pwilkin `40a9f4d01` (f16 KV) and
  upstream `6011c34ce` (bf16 KV) at 8K/32K/128K with zero warmups; the halo-box
  forks are absent, so their own long-context changes (PR #82/#91) are not tested
  against, and gufo's 1292 tok/s at d131072 remains an author-reported figure.
  Neither comparator ran the 262,008 rung.
- **The comparators' per-case prefill rates within a rung vary by up to 21%**
  (pwilkin 851.3 against 1034.3 at 8K), which is why the ratio tables use means
  and the decode comparison uses medians of their four cases. The two comparators
  also use different KV types (f16 and bf16), which is their own requirement and
  not a controlled variable here.
- Prefill and decode are separate timings from one harness; no end-to-end request
  rate is claimed and no serving-path measurement was taken.
- No default changed. No performance claim, no promotion, and no kernel was built
  from this work.
