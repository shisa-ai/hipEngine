# Qwen3.8-27B Strix Halo External Report Requalification

- Created: 2026-08-28
- Scope: **normalize every public Qwen3.8-27B-on-Strix-Halo AR/MTP speed report
  onto invariants that survive differences in model file, quant, backend, and
  prompt class**, then state which external rows are valid comparison targets
  for hipEngine and which are not.
- Hardware lane: **AMD Ryzen AI MAX+ 395 / Radeon 8060S / `gfx1151`**
- Local anchor: **Qwen3.8-27B `Q4_K_M`**, BF16 KV
  (SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`)
- Supersedes: the intake table formerly at
  [`CONCURRENCY2-GFX1151-MTP-TUNING.md`](CONCURRENCY2-GFX1151-MTP-TUNING.md) §2
- Local evidence:
  [`row-scaling baseline`](../benchmarks/results/2026-08-28-gfx1151-qwen38-row-scaling-baseline.json)

This is a **survey and requalification doc**, not a campaign. It records how to
compare an external number to ours and does not itself admit or reject any
implementation candidate.

## 1. Why requalification was needed

Public Strix Halo Qwen3.8 numbers span 7.2 to 163 tok/s. The
[hogeheer499 guide](https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/QWEN38_STRIX_HALO.md)
correctly names eight variables that differ across them: model artifact,
quantization, framework, runtime build, speculation method, prompt class,
context state, and generation parameters. Its own conclusion is that
*"a screenshot with only 'tokens per second' does not establish a portable
buyer recommendation"* and that a matched ladder on one pinned artifact is the
missing work.

Raw tok/s is therefore not a comparison surface. Two invariants are.

## 2. Normalization method

### 2.1 AR: implied memory bandwidth

Dense decode at c1 is weight-bandwidth bound, so the invariant is

```text
implied GB/s = model_file_bytes x AR_tok_s
```

compared against the gfx1151 ceiling: **256 GB/s theoretical**, 221-234 GB/s
measured for large streams (see [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md)).
File size is used uniformly because most external reports publish nothing else;
it overstates the true figure by 4-6% because the token embedding is a row
lookup, not a stream. Our exact streamed budget for `Q4_K_M` is **16.091 GB**
(64 AR blocks 15.048 + `output.weight` 1.043; `token_embd` 0.715 excluded) out
of a 17.096 GB file.

Any AR row implying materially more than ~256 GB/s is not an AR row.

### 2.2 MTP: cycle efficiency

Speculative speedup is the product of a content property and an engine
property. Separate them:

```text
tokens_per_cycle  = 1 + K x acceptance          (content property)
cycle_efficiency  = achieved_speedup / tokens_per_cycle   (engine property)
```

`cycle_efficiency` is what fraction of the theoretical speculative win the
engine actually realizes. It is **only comparable at matched K**, because
verify cost grows with the frontier row count: the same engine scores lower at
K=7 than at K=3. Section 5 shows it is invariant to prompt class, which is what
makes it the correct cross-engine metric.

## 3. Local anchor (measured 2026-08-28, this host)

Direct packed AR graph, `scripts/gguf_packed_ar_bench.py`, p128/d8, three
samples per configuration, stdev < 0.12%:

| Verify rows | Aggregate tok/s | Step ms | Weight sweeps per step |
| ---: | ---: | ---: | ---: |
| 1 | 12.332 | 81.09 | 1.000 |
| 2 | 23.708 | 84.36 | 1.040 |
| 4 | 43.828 | 91.26 | 1.126 |
| 8 | **46.503** | **172.03** | **2.122** |

Rows 1-4 amortize weights almost perfectly. Rows 4-8 cost +88% wall for 2x
rows. This is the row-amortization cliff that the reopened D6 verifier-rowtile
work addresses; it is recorded here only because every ratio below is anchored
to the R=1 figure.

### 3.1 Same-host 1:1 against llama.cpp (2026-08-28) — binding

This supersedes every as-published comparison in §4 and §6 where the two
disagree. `/home/lhl/llama.cpp/llama.cpp-hip` **build 10438, commit
`9d57ce456`** (the same b10435-era build KyaniteLabs used) already carries
`--spec-type draft-mtp` and `src/models/qwen35.cpp`, so no fork build was
needed. Both arms ran the **identical GGUF**, the **identical 10-prompt
category suite**, `max_tokens=24`, `temperature=0`, `--no-cache-prompt`, and a
**fresh server per arm**; llama.cpp's MTP context is created against the same
file's built-in NextN block, which is the same provider hipEngine uses.

| Metric | llama.cpp HIP | hipEngine | Delta |
| --- | ---: | ---: | ---: |
| AR decode tok/s | 12.156 | 12.332 (direct) | **+1.4%** |
| AR complete-wall tok/s | 9.75 | 9.807 (served) | **+0.6%** |
| MTP decode tok/s | 24.897 | 21.158 (direct-leaf) | -15.0% |
| MTP complete-wall tok/s | 15.73 | 15.609 (served) | **-0.8%** |
| MTP / AR, complete wall | 1.613x | 1.5916x | -1.3% |
| **Draft acceptance, K=3** | **90.16%** | **78.57%** | **-11.6 pts** |
| Cycle efficiency, complete wall | 43.5% | **47.4%** | **+3.9 pts** |
| Cycle efficiency, decode basis | 55.1% | 53.9% | -1.2 pts |

Commands and raw output:
[`1:1 artifact`](../benchmarks/results/2026-08-28-gfx1151-qwen38-llamacpp-1to1.json).

**The result is a dead heat, and it inverts the as-published reading.** On this
model, host and suite hipEngine is at parity with llama.cpp on AR (both bases),
at parity on MTP complete wall, and converts each speculative cycle slightly
*better* than llama.cpp does. The single axis where hipEngine is behind is
**draft acceptance: 78.57% versus 90.16%**. llama.cpp needs ~61 cycles to emit
240 tokens where hipEngine needs ~70; our cycles are cheaper, theirs are fewer,
and the two effects nearly cancel.

## 4. AR requalified

| Source | Model / quant | File | Backend | AR tok/s | Implied GB/s | % of 256 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| **hipEngine (direct)** | Qwen3.8-27B `Q4_K_M` | 17.10 GB | HIP | **12.332** | **210.8** | 82% |
| julianmb, stock llama.cpp | Qwen3.8-27B `Q4_K_M` | ~17.1 GB | llama.cpp | 12.27 | 209.8 | 82% |
| julianmb | `ROCmFP4_FAST` 4.26 bpw | 13.55 GiB | Vulkan | 14.02 | 204.0 | 80% |
| julianmb | `Q3_K_M` 3.95 bpw | 12.56 GiB | Vulkan | 15.15 | 204.3 | 80% |
| julianmb | `Q3_K_S` 3.59 bpw | 11.40 GiB | Vulkan | 16.69 | 204.3 | 80% |
| julianmb | `ROCmFP8` 8.25 bpw | 26.25 GiB | Vulkan | 7.66 | 215.9 | 84% |
| MikeVeerman | Unsloth `UD-Q6_K_XL` | 25.9 GB | Vulkan | 8.43 | 218.3 | 85% |
| MikeVeerman | Unsloth `UD-Q8_K_XL` | 31.5 GB | Vulkan | 7.23 | 227.7 | 89% |
| LaurentZuijdwijk | FP4 (size unstated) | ~14.5 GB | Vulkan | 14.0 | ~203.7 | 80% |
| **hipEngine (served)** | Qwen3.8-27B `Q4_K_M` | 17.10 GB | HIP | **9.807** | **167.7** | 66% |
| Ollama official, per the guide | `qwen3.8:27b` `Q4_K_M` | 17.7 GB | Vulkan-RADV | 20.42 | **361.4** | **141%** |

Three conclusions:

1. **Every credible AR report lands in a 204-228 GB/s band (80-89% of peak).**
   Against julianmb's stock llama.cpp on the identical quant, hipEngine direct
   AR is 12.332 vs 12.27 tok/s — **+0.5%, i.e. parity**. hipEngine's AR decode
   is not slow on gfx1151; a dense 27B at Q4_K_M on 256 GB/s LPDDR5X is simply
   capped near 13-15 tok/s.
2. **Vulkan/RADV holds a real 4-9% edge over ROCm/HIP** on this part (218-228
   vs 204-211 GB/s), consistent with julianmb's Vulkan-Wave64 note. That is a
   backend gap, not an engine gap, and it is the only genuine absolute-speed
   deficit in the set.
3. **The guide's own baseline row is not an AR row.** 20.42 tok/s implies 141%
   of theoretical peak bandwidth, which a dense model cannot do. Qwen3.8 ships
   a NextN/MTP block and Ollama 0.32.13 evidently uses it, so that figure is
   speculation-on. It is the most widely quoted "Strix Halo is fast" number and
   it is routinely compared against other people's no-speculation rows.

hipEngine's one real AR deficit is the **served** row: 167.7 vs 210.8 GB/s, a
~20% serving-path overhead that is independent of kernels.

## 5. Does it generalize, or are the numbers overfit?

This is the decisive question for the whole survey, and it is answerable
because MikeVeerman and julianmb both publish acceptance **per prompt class**
at a fixed K on a fixed host and artifact.

### 5.1 MikeVeerman, `UD-Q8_K_XL`, K=3, AR 7.23 tok/s

| Prompt class | MTP tok/s | Acceptance | Tokens/cycle | Speedup | **Cycle eff.** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 16.46 | 74.3% | 3.229 | 2.273x | **70.4%** |
| JSON extraction | 16.18 | 73.2% | 3.196 | 2.238x | **70.0%** |
| Reasoning | 15.83 | 71.2% | 3.136 | 2.189x | **69.8%** |
| Prose | 12.72 | 50.8% | 2.524 | 1.759x | **69.7%** |
| Spread | | **23.5 pts** | | **29%** | **0.7 pts** |

### 5.2 julianmb, `ROCmFP4_FAST`, K=4, AR 14.02 tok/s

| Prompt class | MTP tok/s | Acceptance | Tokens/cycle | Speedup | **Cycle eff.** |
| --- | ---: | ---: | ---: | ---: | ---: |
| JSON extraction | 35.79 | 88.0% | 4.520 | 2.553x | **56.5%** |
| Code generation | 34.82 | 82.6% | 4.304 | 2.484x | **57.7%** |
| Technical explanation | 32.40 | 76.2% | 4.048 | 2.311x | **57.1%** |
| Reasoning / math | 30.56 | 71.4% | 3.856 | 2.180x | **56.5%** |
| Spread | | **16.6 pts** | | **17%** | **1.2 pts** |

### 5.3 The answer

**Raw MTP tok/s and MTP speedup are overfit to prompt class. Cycle efficiency
is not.** In two independent sources, on different quants, backends and K,
cycle efficiency is constant to within **0.7 and 1.2 percentage points** across
four prompt classes each, while acceptance moves 17-24 points and speedup moves
17-29%.

That decomposition holds across the whole survey:

- **AR is content-independent** — Laurent's own control is 14.0 t/s structured
  vs 14.1 t/s prose, identical as it must be.
- **Acceptance is a property of the content**, not the engine: repetition
  ~96%, JSON 73-88%, code 74-83%, reasoning 71-76%, prose **44-51%**.
- **Cycle efficiency is a property of the engine**, invariant to content and
  comparable only at matched K.

Practical consequences:

1. Any single-number MTP claim without a prompt class attached is
   uninterpretable. The gap between a source's own best and worst prompt class
   (1.76x-2.27x for MikeVeerman) exceeds most cross-engine gaps being argued
   about.
2. Headline peaks are the right tail of content overfit. Laurent's 65.6 t/s is
   structured output; his prose column for the same configuration is 26.1.
   KyaniteLabs' 148-163 t/s is a count-to-30 repetition task their own document
   labels an *"Ngram speculation artifact"*.
3. Our own 47-54% deficit versus llama.cpp's 70-77% is therefore a **real
   engine gap and not an artifact of our Japanese/mixed prompt suite** — and
   symmetrically, it cannot be closed by choosing friendlier prompts.

## 6. MTP requalified, at matched K

Comparable rows are grouped by K. Higher K structurally lowers cycle
efficiency, so cross-K rows are not rankable against each other.

| Source | K | Acceptance | Tokens/cycle | Speedup | **Cycle eff.** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **K = 3** | | | | | |
| LaurentZuijdwijk, fixed n=3, structured | 3 | 95% | 3.850 | 2.971x | **77.2%** |
| MikeVeerman `UD-Q6_K_XL` c1 | 3 | 65.4% | 2.962 | 2.220x | **74.9%** |
| MikeVeerman `UD-Q8_K_XL` c1 mean | 3 | 66.4% | 2.992 | 2.115x | **70.7%** |
| **hipEngine direct-leaf B3 `Q4_K_S`** | 3 | ~78.6% | 3.357 | 1.785x | **53.2%** |
| **hipEngine direct-leaf B3 `Q4_K_M`** | 3 | 78.6% | 3.357 | 1.810x | **53.9%** |
| **hipEngine served C1 B3 `Q4_K_M`** | 3 | 78.6% | 3.357 | 1.592x | **47.4%** |
| **K = 4** | | | | | |
| julianmb `ROCmFP4_FAST` (4 classes) | 4 | 71-88% | 3.86-4.52 | 2.18-2.55x | **56.5-57.7%** |
| **K = 7, adaptive** | | | | | |
| LaurentZuijdwijk adaptive n_max 7 | ~5.3 eff. | 96% | ~6.1 | 4.686x | not directly computable |

> **Corrected 2026-08-28 by the §3.1 same-host 1:1.** An earlier revision of
> this section read the K=3 block as showing a 47-54% versus 70-77%
> *engine* gap. **That inference does not survive measurement.** Run against
> the same GGUF, host and suite, llama.cpp scores 43.5% complete-wall / 55.1%
> decode-basis cycle efficiency versus hipEngine's 47.4% / 53.9% — parity, with
> hipEngine ahead on the complete-wall basis. The external 70-77% rows are not
> a higher-quality engine; they are a different **target model and timing
> boundary** (see §6.2). The real hipEngine deficit is **acceptance**, which
> §3.1 measures directly at -11.6 points.

Reading the K=3 block with that correction: the external rows are not rankable
against ours, because their targets are 25.9-31.5 GB Q6/Q8 files measured over
256-token generations while ours is a 17.1 GB Q4_K_M measured over 24. §3.1 is
the only row in this document that holds all of that fixed.

### 6.2 Why cycle efficiency is not model- or boundary-invariant

Cycle efficiency divides out acceptance, so it is invariant to **prompt class**
(§5). It is **not** invariant to two other things, and the earlier draft of
this doc wrongly treated it as if it were:

- **Target size.** The drafter cost per cycle is roughly fixed, so a larger
  target amortizes it better. MikeVeerman's 70% is on a 31.5 GB `UD-Q8_K_XL`
  target; the same engine on a 17.1 GB `Q4_K_M` target scores 55.1% in §3.1.
  Higher cycle efficiency on a fatter model is arithmetic, not engineering.
- **Timing boundary.** At `max_tokens=24` prefill dominates the complete wall
  and compresses the ratio. The same llama.cpp run is **2.04x on a decode basis
  and 1.613x on a complete-wall basis**. MikeVeerman's 2.11x is a 256-token
  decode-style figure and must never be set beside a 24-token complete-wall
  figure.

Compare cycle efficiency only at matched K, matched target file, and matched
timing boundary. In practice that means §3.1, not §6.1.

### 6.3 The "spectacular claim" configurations, run on our GGUF

Every headline configuration in §7 was executed against **our** Qwen3.8-27B
`Q4_K_M` and **our** mixed 10-prompt suite, `max_tokens=128` (not 24 — at 24
tokens prefill dominates and a depth-7 draft can never show its value, which
would rig the test against the deep-draft claims), `temperature=0`,
`--no-cache-prompt`, fresh server per arm.

`LaurentZuijdwijk/llama.cpp` was built from source at the pinned commit
**`c28d538df`** (build 10681) as a git worktree off our existing checkout. Its
`common/speculative.cpp` differs from mainline by **337 lines** and it adds a
real flag mainline does not have, `--spec-draft-adaptive` — *"size each draft
from measured acceptance rather than always drafting `--spec-draft-n-max`"*.
Mainline b10438 is the comparison build.

| Build / arm | agg tok/s | vs own AR | Acceptance | Drafts | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| mainline AR | 11.37 | 1.000x | — | 0 | 0 |
| mainline `n_max 3` | 16.02 | 1.409x | 63.41% | 1301 | 825 |
| mainline `n_max 7` fixed | 14.83 | 1.304x | 38.43% | 2376 | 913 |
| mainline `n_max 7 n_min 3` | 14.79 | 1.301x | 38.29% | 2382 | 912 |
| mainline `draft-mtp,ngram-mod n12` | 16.26 | 1.430x | 23.87% | 3917 | 935 |
| **fork AR** | **11.35** | 1.000x | — | 0 | 0 |
| **fork `n_max 3`** | **18.93** | **1.668x** | 63.86% | 1295 | 827 |
| fork `n_max 7` fixed | 14.87 | 1.310x | 38.58% | 2369 | 914 |
| **fork `n_max 7` + adaptive** | 17.66 | 1.556x | 61.70% | 1410 | 870 |
| fork `n_max 12` + adaptive | 17.62 | 1.553x | 60.16% | 1461 | 879 |

Four results, in descending order of importance:

1. **The headline claim does not generalize.** Laurent reports adaptive
   `n_max 7` at 65.6 t/s versus fixed `n=3` at 41.6 — a +58% win for adaptive.
   On our model and suite **adaptive loses to plain `n_max 3` by 6.7%**
   (17.66 vs 18.93), and `n_max 12` adaptive is no better. His 4.69x over bare
   decode does not appear at all: our best configuration of any kind is
   **1.668x**. The published numbers are FP4 plus 300-token *structured*
   output; on a Q4_K_M target and a mixed code/EN/JA suite the best policy is
   simply a shallow draft.
2. **The mechanism does generalize, and is real.** Fixed `n_max 7` collapses
   acceptance to 38.6% on both builds; the adaptive controller holds it at
   61.7% and recovers throughput 14.87 -> 17.66 (+18.8%). Laurent's diagnosis —
   that a fixed deep draft destroys acceptance and acceptance-sized drafts fix
   it — reproduces exactly. It simply never overtakes a shallow fixed draft
   here, because on this target depth was never the binding constraint.
3. **Mainline's `--spec-draft-n-min` is not an adaptive controller.**
   `n_max 7 n_min 3` (14.79, 38.29%) is statistically identical to fixed
   `n_max 7` (14.83, 38.43%). Anyone reading mainline's `n_min` as Laurent's
   mechanism is measuring nothing. This is why the fork had to be built.
4. **The genuinely transferable finding is unrelated to adaptive.** Fork
   `n_max 3` reaches **1.668x** where mainline `n_max 3` reaches **1.409x** —
   **+18.4% MTP throughput at identical AR (11.35 vs 11.37) and identical
   acceptance (63.86% vs 63.41%) and identical draft counts.** Same tokens,
   same acceptance, ~18% less wall: that is a pure MTP **cycle-cost** win
   somewhere in the fork's 337 changed lines or in upstream between b10438 and
   b10681. This is the one thing in the entire external survey worth reading
   the diff for, and §8 promotes it.

The KyaniteLabs ngram-mod stack behaves exactly as their own document admits.
It posts the best mainline aggregate (16.26) and the best single category
(`mixed_ja_en` 25.0 decode) but drafts **3917 tokens to accept 935** — a 23.87%
acceptance — and is worth only 1.07x on `general_en`. It is a repetition
exploit: excellent where the output echoes the prompt, near-worthless
otherwise. Our T4.1 rejection stands.

Per-category decode rates make the content dependence concrete
(AR: code 12.3, general_en 12.3, general_ja 12.2, mixed_ja_en 10.7):

| Arm | code | general_en | general_ja | mixed_ja_en |
| --- | ---: | ---: | ---: | ---: |
| fork `n_max 3` | 22.0 | 18.0 | 18.8 | 23.2 |
| fork `n_max 7` adaptive | 20.7 | 16.8 | 17.0 | 21.8 |
| mainline `n_max 3` | 19.2 | 13.4 | 18.4 | 20.2 |
| mainline mtp+ngram | 20.1 | 13.1 | 14.6 | **25.0** |

Even the best arm spans 1.46x (`general_en`) to 2.17x (`mixed_ja_en`) over AR.
No single-number claim survives that spread.

Two internal consistency checks: our `Q4_K_S` and `Q4_K_M` direct paths land at
53.2% and 53.9% independently, so ~54% is a stable property of our direct
route; and serving costs a further 6.5 points (54% -> 47%), matching the
independently measured ~20% serving overhead in §4.

Laurent's adaptive row is left uncomputed on purpose: adaptive draft length
means the mean realized depth is unknown, so tokens/cycle cannot be formed.
Back-solving at his own fixed-n=3 efficiency of 77% implies a mean depth near
5.3, which is consistent but not measured.

### 6.1 Concurrency

| Source | c1 | c2 | c3 | c4 | Crossover |
| --- | ---: | ---: | ---: | ---: | --- |
| MikeVeerman, MTP / AR aggregate | 2.19x | 1.28x | 1.04x | 0.78x | between c3 and c4 |
| hipEngine, pre-fix | 1.59x | 0.53x (K1) / 0.82x (K3) | — | 0.59x (K1) | between c1 and c2 |
| hipEngine, post planar-Q6 verifier rowtile | 1.59x | **0.92x (K3)** | — | — | approaching c2 |

The inverse comparison matters as much: MikeVeerman's **AR** batch scaling c1
to c4 is 7.10 -> 21.75 = **3.06x**, while hipEngine's direct packed AR is
12.332 -> 43.828 = **3.55x**. Our batching is better than llama.cpp's. The
width deficit was never a scheduler or batching problem; it was isolated to the
speculative verify path.

The [CIRU vLLM fork](https://recipes.vllm.ai/inclusionAI/Ling-3.0-flash) row
(MTP K1 scaling c1->c6, 26.79 -> 63.51 aggregate) remains the target *shape*
for c>1 scaling, but it is a 77 GB W4A16 **MoE**, so no rate transfers.

## 7. Rows struck as comparison targets

| Row | Why it is not a target |
| --- | --- |
| Ollama official 20.42 tok/s "generation" | Implies 141% of theoretical peak bandwidth; speculation-on, not AR (§4) |
| KyaniteLabs 148-163 tok/s "warm c30" | Count-to-30 repetition; their own doc calls it an *"Ngram speculation artifact"*. Cold c30 is 59.7 |
| KyaniteLabs "5x on repetition-heavy tasks" | Their own prose sweep: MTP+ngram 11.0, MTP solo 10.7, ngram solo 11.3, **speculation disabled 11.1** — speculation buys ~nothing on genuine prose |
| LaurentZuijdwijk 65.6 tok/s | FP4 + adaptive n_max 7 on 300-token *structured* output; his prose column is 26.1. The guide flags it "not reproducible end to end yet" |
| "Tuned Vulkan + ROCmFP4 + DFlash2 ~52 tok/s" (guide) | Guide's own status: *"different fork, quant, drafter, prompt, and request shape"*; unimported raw package |
| Stock b10503 Q8_0 with MTP, 7.3-22.4 tok/s (guide) | Range spans prompt classes with no per-class breakdown; guide states the raw package was never imported |
| Any cross-host absolute rate | Two 8060S hosts are independent lanes per `AGENTS.md`; the ZBook lane is 60 W power-limited |

**Valid comparison targets retained:** julianmb stock `Q4_K_M` AR (12.27
tok/s), MikeVeerman's full 12-configuration and concurrency ladders, and
Laurent's fixed-n=3 arm — all on the cycle-efficiency and implied-GB/s bases
above, never on raw tok/s.

## 8. Consequences for hipEngine campaigns

1. **The AR premise is settled.** hipEngine AR is at parity with stock
   llama.cpp on the same quant and inside the external 80-89% bandwidth band.
   No AR decode kernel campaign is justified by external comparison. The
   remaining AR item is the ~20% **serving** overhead (§4), which is a separate
   unit from kernels.
2. **There is no MTP cycle-cost gap versus llama.cpp at C1.** §3.1 measures a
   dead heat on the same file, host and suite: MTP complete wall 15.609 vs
   15.73 tok/s, and hipEngine converts each cycle slightly better (47.4% vs
   43.5%). Do not open a C1 verifier-dataflow campaign on the strength of the
   as-published external rows; they compare different targets and timing
   boundaries.
2a. **Draft acceptance is the one real C1 deficit: 78.57% vs 90.16%,
   -11.6 points on the identical suite.** That is worth roughly the whole
   remaining C1 gap — at llama.cpp's acceptance our 47.4% cycle efficiency
   would give 1.613x+ rather than 1.5916x, and acceptance compounds with any
   later depth increase. This is now the highest-value C1 item and it is a
   drafter/provider question, not a kernel one. Diff targets: NextN priming and
   cursor synchronization, `p_min` handling, and whether our K=3 proposal chain
   degrades at positions 2-3 the way llama.cpp's does not.
3. **T3 adaptive-K and the B4 clamp should be reopened** after the verifier
   rowtile work lands. `CONCURRENCY2-GFX1151-MTP-TUNING.md` T3.1 rejected the
   EMA controller for losing to fixed B3 by 0.58%, and T1.4 recorded that B4
   "clamps to B3". Both limits sat directly on top of the rows>4 amortization
   cliff, because depth K>3 requires R>4 verify rows. Laurent's fixed-vs-
   adaptive table is the reason this matters: his adaptive arm wins by going
   **deeper** (n_max 7 at 96% acceptance) where fixed n=7 collapses to 18%.
   Our controller was never able to test that regime.
4. **Target the c3 crossover.** llama.cpp holds MTP > AR through c3; the
   post-fix hipEngine c2/K3 is 0.92x. c2 then c3 are the ordered milestones.
5. **Read the `c28d538df` MTP cycle diff — highest-value external item.**
   §6.3 measures +18.4% MTP throughput at identical AR, identical acceptance,
   and identical draft counts versus mainline b10438. That is a pure cycle-cost
   win in a path we have just established we are at parity with (§3.1), so
   whatever it is likely transfers to us directly. Bisect b10438..b10681 first
   to separate Laurent's 337-line `common/speculative.cpp` change from upstream
   movement, then diff the winning commit against our cycle.
6. **Do not implement adaptive draft depth.** §6.3 measures it losing to a
   plain shallow draft by 6.7% on our target and suite. This independently
   re-confirms the T3.1 rejection on a second engine, and it means the "reopen
   T3 after the rowtile work" item in §8.3 should be re-scoped: the reason to
   reopen would be C>1 frontier economics, not draft depth at C1.

## 9. Open measurement gap

We do **not** currently publish per-category acceptance, only the 78.57%
aggregate over the four-category suite. §5 demonstrates that cycle efficiency
is content-invariant for two external engines; we have not verified it for
ours. Emitting per-category acceptance alongside per-category tok/s in the next
full-suite run would confirm that our ~54%/47% is a clean engine constant
rather than a blend, and would make every future cell directly comparable to
the tables above. This is the single cheapest addition to the benchmark
artifact schema.

Attempted 2026-08-28 and blocked. `scripts/mtp-bench.py --mode
hipengine-current` wraps `scripts/mtp_prompt_suite_economics.py`, whose
`_load_raw_tokenizer()` requires `<engine-model>/tokenizer.json`. This host
carries tokenizers only for `Qwen3.6-35B-A3B-PARO-packed-MTP-BF16` and
`laguna-s-2.1`, so the direct harness cannot encode the suite for
Qwen3.8-27B. Two ways round it, in preference order:

1. **`--mode server`** against a live hipEngine OpenAI server. Tokenization
   happens server-side from the GGUF, so the missing `tokenizer.json` is moot,
   and it is the same request shape llama.cpp's MTP bench uses — which would
   also let a llama.cpp server be measured through one entry point and convert
   §4 and §6 from as-published to same-host. This is how the D2-D7 campaigns
   measured ("blocking OpenAI complete wall"); note `run-server.sh` in the repo
   root is local-only and points at a different model.
2. Stage a Qwen3.8-27B `tokenizer.json` beside an engine-model directory and
   rerun `--mode hipengine-current` unchanged.

Reproducing the external rows on this host additionally requires building a
llama.cpp carrying `--spec-type draft-mtp`, which is not in mainline.

## 10. Sources

All URLs re-fetched **2026-08-28**. Where the prior intake table in
`CONCURRENCY2-GFX1151-MTP-TUNING.md` §2 pinned a commit, that pin is recorded;
the content below is current HEAD at the fetch date and was **not** re-read at
the pinned commit.

| Source | Prior pin | URL |
| --- | --- | --- |
| hogeheer499-commits/strix-halo-guide | `029320fb` | <https://github.com/hogeheer499-commits/strix-halo-guide/blob/main/QWEN38_STRIX_HALO.md> |
| MikeVeerman/qwen38-27-Strix-Halo-bench | `cc527064` | <https://github.com/MikeVeerman/qwen38-27-Strix-Halo-bench> |
| julianmb/q38rocm | `5d097740` | <https://github.com/julianmb/q38rocm> |
| KyaniteLabs/qwen38-27b-strix-halo | `7fa3ca81` | <https://github.com/KyaniteLabs/qwen38-27b-strix-halo> |
| LaurentZuijdwijk/llama.cpp | `c28d538` | <https://github.com/LaurentZuijdwijk/llama.cpp> |
| CIRU / Ling-3.0-Flash int4 Strix (vLLM) | `838616875` on vLLM `d35eb6c4` | <https://recipes.vllm.ai/inclusionAI/Ling-3.0-flash> — the `jcbtc/Ling-3.0-Flash-CIRU-int4-Strix-native` repo URL in the prior table now returns 404 |
| kyuz0 gfx1151 vLLM benchmark toolboxes | — | <https://kyuz0.github.io/amd-strix-halo-vllm-toolboxes/> |

Engine/config details captured at fetch time:

- **MikeVeerman:** llama.cpp build 9867 (`152d337fa`), Vulkan, Mesa 26.0.3 RADV
  STRIX_HALO, 128 GB. Server flags
  `-ngl 999 -fa on -b 2048 -ub 512 --no-mmap --cache-reuse 256`; MTP enabled
  with `--spec-type draft-mtp`, defaults `--spec-draft-n-max 3
  --spec-draft-p-min 0.00`. Generation prompt 256 tokens; prefill probe 10,863
  tokens; 3.2 mean draft tokens per step.
- **julianmb:** `ROCmFP4_FAST` 13.55 GiB / 4.26 bpw, SHA-256
  `fb89c78d2be91cdb68eaaaa45b1270710bf34aa721dc1f0b9e3aa7b98d2e1da9`;
  asymmetric TurboQuant KV (`-ctk q8_0 -ctv turbo4`); profiles via
  `run_server.sh --profile {speed,agent,cache,safe}`; sustained-decode profile
  `--draft-n 4 --draft-p 0.0 --ubatch 2048`. Notes Vulkan RADV Wave64 highest
  decode, ROCm/HIP lowest TTFT and highest prefill.
- **KyaniteLabs:** `Qwen3.8-27B-UD-Q4_K_XL`, ROCm/HIP, llama.cpp b10435-era
  (`9d57ce4`), 64 GB, KV `q4_0`, 262,144-token ceiling. Speculation via
  `--spec-type draft-mtp,ngram-mod --spec-draft-n-max 12
  --spec-ngram-mod-n-min 24`. Hang guards `HSA_ENABLE_SDMA=0 HSA_XNACK=1`.
- **LaurentZuijdwijk:** EMA adaptive draft length in
  `common/speculative.cpp`; per-sequence acceptance EMA sizes the next draft to
  `ema+1` with an additive probe on full accept.
- **hogeheer499 guide:** Ollama 0.32.13 / Vulkan-RADV on Beelink GTR9 Pro
  128 GB; 292.49 prompt t/s, 20.42 generation t/s, nine warm repeats, context
  validated to 50,059 prompt tokens.

## 11. Limitations

- External implied-GB/s uses **file size**, not streamed bytes, so those
  figures run ~4-6% high; the ranking and the band are unaffected.
- KyaniteLabs and LaurentZuijdwijk publish no model file size; their rows use
  size estimates and are marked as such.
- Cycle efficiency assumes the reported acceptance is per-draft-token and that
  the engine emits `1 + K x acceptance` tokens per cycle. Engines that resample
  or use tree drafts would violate this; none of the sources above appear to.
- The §3 row-scaling anchor is the direct packed AR graph path, not the
  speculative verify frontier, and is a fixed-token synthetic fixture. It is
  diagnostic for this survey and is not a retained performance claim.
- **llama.cpp HIP was reproduced locally (§3.1); the other external rows were
  not.** Every number in §4, §5 and §6.1 remains as-published and is therefore
  subject to the target-size and timing-boundary confounds set out in §6.2.
  Where §3.1 and the as-published tables disagree, §3.1 binds.
- §3.1 is C1 only, one run per arm, and reuses prior hipEngine artifacts rather
  than re-running both engines in one lifecycle. Its KV dtypes differ
  (BF16 vs F16).
