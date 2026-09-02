# Qwen3.8-Flash-Next gfx1151 Performance Campaign

Status: **active plan, fully impact-profiled 2026-09-01.** The retained
category-balanced screening baseline remains **83.70/83.16/69.10 tok/s**
prefill and **14.40/14.42/10.42 tok/s** decode at p512/p1024/p4096. A clean
current diagnostic packet measures **86.21/85.71/70.95** and
**14.70/14.65/10.59**, but strong repetition-order drift prevents silently
promoting those rates as the closure baseline. Exact matched profiling now
attributes **100%** of hipEngine device time at all three prefills and at live
513/1025/4097 decode; both hipEngine and pinned patched llama.cpp have zero
generic family remainder. The old **41.6% unattributed** row was incomplete
comparison coverage, not unidentified execution. The impact queue is now:
exact p4096 QSA decode, operation-complete prefill MoE, dense/GR boundaries,
p4096 QSA prefill, short-decode selected projections/Q8, then GDN. P8 returns
to admission-pending because its replacement 1.112x ratio mixes a strict
runner denominator with a graph result imported from another run; production
`O` and `s` remain unknown. Cold PLE and MTP remain separate lanes. This is not
section 6 closure: five thermal pairs, cold-PLE isolation, and category
heldouts remain open. The first retained post-profile unit replaces serialized
p4096 QSA decode with an exact ordered three-pass owner. Four-category tg128
improves **93.912→80.061 ms/token (1.173x)** across 12 counterbalanced pairs
with exact full logits and IDs; the named trace reduces the QSA operation role
to **20.913 ms/token**. This is a retained production improvement, not section
6 closure.

This document is the performance-specific plan and punchlist.
[`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md) remains the model/bring-up
authority; this file owns only the gap-closure campaign. Cross-engine speed,
static-logit, autoregressive-repeatability, MTP-equivalence, test-coverage, and
absolute-quality evidence is consolidated in
[`QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md`](QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md).

## 1. Objective and boundaries

Close the same-host, same-model, same-weight-quant performance gap to the best
current HIP comparator first, then to the best same-host Vulkan comparator.
Upstream llama.cpp remains the role-resolved attribution baseline; EngramHalo
HIP and Nathan Vulkan are additional competitor lanes, not inherited evidence.
A retained win must preserve the published execution-profile contract and must
come with a compact artifact, a worklog entry, and a benchmark rollup. The
campaign does not close merely because one microbenchmark, one prompt, one
backend, or one MTP budget wins.

### In scope

- Host: `zbook`, AMD Ryzen AI Max+ Pro 395, Radeon 8060S, `gfx1151`.
- Model: `Qwen/Qwen3.8-Flash-Next` through the pinned Unsloth
  `UD-Q4_K_XL` split GGUF.
- KV/cache policy: current BF16 baseline unless a row explicitly declares a
  different KV profile.
- Profiles: named `strict` and `production` manifests.
- Binding AR representation: the pinned target weights with BF16 K/V. Q8 K/V
  is a separately declared T3 product configuration after BF16 AR parity.
- Workloads: category-balanced exact matched p512, p1024, and p4096 prefill
  with 128 autoregressive transitions after each prefix, then the existing
  long-context and full-category MTP suites. Legacy p508/p1012/tg32 rows remain
  tail/continuity diagnostics, not closure targets.
- Comparator lanes: upstream llama.cpp HIP/Vulkan, EngramHalo HIP, and Nathan
  Vulkan, each pinned to an exact source and binary identity.

### Out of scope

- Changing model representation, quant recipe, or prompts to improve a score.
- Treating external EngramHalo/Nathan numbers as hipEngine results.
- Treating a different weight quant, KV type, prompt, cache state, or MTP policy
  as a same-configuration old-to-new comparison.
- Vulkan-specific code in this campaign except as design evidence for HIP work.
- New feature expansion that does not close a measured gap.
- Inferring W7900/gfx1100 performance from this host.

## 2. Current verified gap

Canonical screening evidence:
[`2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json).
It pins hipEngine commit `6fbcc721a`, current upstream `f1793c1c4`, every
external binary and patch hash, the exact 12-case fixture, all 180 measured
samples, host state, and output-repeatability verdicts. The current canonical
impact authority is
[`2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json).
It adds exact-token p512/p1024/p4096 prefill, live-513/1025/4097 decode,
allocation/lifecycle, broad-active-expert telemetry, and matched patched
llama.cpp role-family evidence. The older p508/tg32 artifact remains historical
continuity evidence only.

### Historical p508/tg32 attribution rows

| Workload | hipEngine production | llama.cpp HIP | llama.cpp Vulkan | HIP advantage | Vulkan advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| p508 prefill | **84.83 tok/s** | 272.83 | 331.03 | **3.22x** | **3.90x** |
| tg32 steady decode | **15.19 tok/s** | 16.64 | 24.22 | **1.10x** | **1.59x** |

### 2.1 Canonical exact-token screening

All rows use the same four category token arrays at each shape, BF16 K/V,
greedy sampling, disabled prompt reuse, one warmup per case, and three measured
requests per case. Each cell is prompt processing / 128-transition decode in
tokens per second.

| Engine | p512 | p1024 | p4096 | Repeatability |
| --- | ---: | ---: | ---: | --- |
| hipEngine production `61b1cef1b` | **83.70 / 14.40** | **83.16 / 14.42** | **69.10 / 10.42** | 12/12 exact |
| Upstream Vulkan `f1793c1c4`, queue/repack/fit-off | 200.01 / 24.39 | 241.84 / 21.33 | 266.58 / 18.98 | 12/12 exact; p512 and p1024 prefill noisy |
| Patched-upstream HIP `f1793c1c4` | 239.23 / 17.74 | 301.68 / 16.88 | 294.47 / 14.77 | 12/12 exact; non-stock loader |
| EngramHalo HIP `1423f689` | 234.84 / 17.44 | 314.98 / 17.04 | 381.17 / 15.99 | p512/p1024 exact; `code-p4096` alternates |
| Nathan Vulkan `ad914eb`, queue/repack/fit-off | 360.23 / 24.34 | 357.61 / 21.10 | 351.85 / 19.01 | 0/12 exact; diagnostic |

#### Starting-point correctness and accuracy contract

This screening measured exact greedy-continuation repeatability. It did **not**
collect logits, KL divergence, task scores, or human accuracy ratings, so it
cannot rank model accuracy. hipEngine's starting accuracy basis remains the
existing BF16-teacher
[`canonical text gate`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-text-bringup.json),
[`heldout logits gate`](../benchmarks/results/2026-08-27-gfx1151-qwen38-flash-next-heldout-logits.json),
and named-production execution-profile packet (production manifest
`9e27fec0...`, strict manifest `42509601...`). Any arithmetic change must re-run
the applicable gates; a repeat-exact speed row does not replace them.

The entitled Vulkan refresh uses graphics queue, repack, explicit fit-off, and
auto clocks. Upstream remains exact, but p512 prefill/decode and p1024 prefill
have coefficients of variation above 2%; these rows replace the
under-configured diagnostic screen but cannot freeze the section 6 target.
Nathan remains diagnostic. Evidence:
[`2026-09-02-gfx1151-qwen38-flash-next-entitled-vulkan-canonical-refresh.json`](../benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-entitled-vulkan-canonical-refresh.json).

The measured correctness status is:

- hipEngine production, pristine upstream Vulkan, and patched-upstream HIP are
  repeat-exact on all 12 cases.
- EngramHalo is repeat-exact at p512 and p1024, but `code-p4096` alternates
  between two continuation hashes with the first difference at output index 1
  (the second generated token). Its p4096 rates are diagnostic only.
- Nathan fails repeatability on all 12 cases. A follow-up repeated the identical
  p1024 token array 16 times and produced 16 distinct continuations, often
  diverging at output indices 1–9. All Nathan rates are diagnostic only.
- Full 129-token hashes do not establish cross-engine parity: hipEngine matches
  pristine upstream Vulkan on 1/12 cases, patched-upstream HIP on 0/12, and
  EngramHalo on 1/12. Compounding arithmetic differences make this diagnostic,
  not an accuracy verdict; the logits/KL/task gates remain binding. No external
  lane becomes an accuracy oracle merely by being deterministic or faster.

#### Classifying a non-exact continuation

A divergence is not self-explaining. Split it into one of two classes before
assigning blame; an unclassified divergence is neither noise nor a bug.

- **Tie class.** A last-ulp flip at a stable output position, with a logit
  margin at or below roughly 0.04 nats, reproducing at the same position across
  repeats. hipEngine tolerates this only where a declared production profile
  already permits the reassociation that produced it.
- **State class.** A divergence whose position moves between repeats, or one
  carrying a whole-nat logit shift. Recurrent-state corruption lives here, and
  it surfaces wherever the triggering event occurs: a rejected draft at 64K
  produces a state-class divergence thousands of positions past any fixed
  cutoff. Never classify by output index alone. The Pat1entZ3r0 program's
  "only a divergence before position 50 is a real bug" rule would not have
  caught the rollback-ring corruption that same program discovered.

A single capture from build A against a single capture from build B cannot
perform this split at all: it cannot separate "B is wrong" from "B is
nondeterministic". Every equivalence claim here therefore needs a repeat arm on
each build before any cross-build arm. That is what the 12-case canonical screen
supplies and what external token-parity gates generally omit; it is why Nathan
reads as 0/12 here and as an unremarkable pass elsewhere.

#### Required upstream HIP loader patch on this host

Pristine upstream HIP `f1793c1c4` produced zero samples after separate
1,800-second starts with default mmap and with `--no-host`. For this exact
111-GB model on `zbook`, the only successful current-upstream HIP configuration
tested applied both documented Strix Halo loader patches:

1. `llama-cpp-25992-rocm-host-buffer.patch`, SHA-256
   `aca70db134d0e65be7a250cf1eb4237bb739d9d586f5c5153ce972372f67b4de`;
2. `llama-cpp-qwen38-per-buffer-mmap.patch`, SHA-256
   `971d428de98ecdf59941946bb391c257e82501ce98b7c71cc1f34803181fe133`.

The combined source diff is
`a37fa3bb64cb693dbe26c98177c757bda60683de2aeb5bae56222bfbfe5783b1` and the
measured HIP server is
`bb41c7555c4ad6cd14df4d2a308d991ddb5d1b44ec21d294fa0fe24f5aeafa86`.
No single-patch ablation was run, so individual necessity is unknown. The
campaign treats the pair as one host/model-scoped startup requirement until an
ablation proves otherwise; this is not a claim that every upstream HIP
installation needs the patches. The result must remain labeled **patched
upstream**; pristine upstream HIP remains `startup_blocked` and has no numeric
row.

#### Decode-depth cliff

The depth loss is larger than for the other hipEngine models in the scoreboard
and is partly specific to Qwen4Exp. Throughput and derived synchronized latency
are:

| Lane | tg128 p512 | tg128 p1024 | tg128 p4096 | Added ms/token, p1024→p4096 | p512→p4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hipEngine production | 14.399 | 14.421 | 10.416 | **+26.64** | **-27.7%** |
| Upstream Vulkan | 24.385 | 21.327 | 18.975 | +5.81 | -22.2% |
| Patched-upstream HIP | 17.741 | 16.881 | 14.768 | +8.47 | -16.8% |
| EngramHalo HIP | 17.439 | 17.044 | 15.989 | +3.87 | -8.3% (p4096 diagnostic) |
| Nathan Vulkan | 24.342 | 21.098 | 19.012 | +5.20 | -21.9% (diagnostic) |

For context only, the same-host scoreboards show Qwen3.8-27B Dense at
13.069→13.038 tok/s (-0.2%) and Qwen3.6-35B-A3B at 54.330→54.798 tok/s
(+0.9%) from p512 to p4096. Those are different architectures and are not
comparison denominators, but they show that this is not a generic hipEngine
context penalty.

The target GGUF declares a 2,048-token QSA budget and compression ratio 4, so
`qsa_dense_equivalent_max_tokens == 2051`. p512 and p1024 decode remain on the
dense-equivalent path. p4096 crosses the boundary: each of 12 QSA layers adds
the index-query projection, normalization/RoPE, score over about context/4
pooled blocks, stable top-k expansion, and sparse attention over 2,048 selected
tokens plus the tail. The path transition is confirmed from dispatch and model
geometry. The retained P6 boundary profile now localizes the cliff: clean wall
adds **29.27 ms/token** at live 2,051→2,052, of which sparse attention owns
**27.47 ms** and score/top-k only **0.92 ms** in the profiled delta. Live
2,052→4,097 is flat, confirming fixed selected-budget cost rather than
context-length growth. This supersedes the earlier unlocalized diagnosis.

#### Repeatability-valid performance targets

The repeatability-valid screening targets and current hipEngine gaps are:

| Shape | HIP target and gap | Vulkan target and gap |
| --- | ---: | ---: |
| p512 / tg128 | patched upstream, **2.86x / 1.23x** | upstream, **2.39x / 1.69x** (noisy) |
| p1024 / tg128 | EngramHalo, **3.79x / 1.18x** | upstream, **2.91x / 1.48x** (prefill noisy) |
| p4096 / tg128 | patched upstream, **4.26x / 1.42x** | upstream, **3.86x / 1.82x** |

These are screening ratios, not match/loss verdicts. Section 6 still requires
five same-thermal counterbalanced pairs, per-row CV at or below 2% for a match,
and paired confidence intervals. Cross-engine generated-ID equality is
recorded but remains diagnostic for named production arithmetic; each lane's
repeatability and hipEngine's execution-profile gates are separate checks.

### 2.2 Wall-time gap budget and Amdahl ledger

Throughput ratios hide how much time an optimization must actually remove. The
following budget derives synchronized wall from the clean current diagnostic
packet at `3ddb748d4`; it is used for candidate admission despite thermal drift,
not promoted as the closure baseline. The HIP target is the fastest
repeatability-valid HIP lane for that row.
The final target is the fastest repeatability-valid HIP or Vulkan lane; invalid
Nathan/apepojken rows and EngramHalo p4096 are excluded.

| Workload | hipEngine wall | HIP target wall; reduction required | Final valid wall; reduction required |
| --- | ---: | ---: | ---: |
| p512 prefill | 5.939 s | 2.140 s; **3.799 s (64.0%)** | 2.140 s; **3.799 s (64.0%)** |
| p1024 prefill | 11.947 s | 3.251 s; **8.696 s (72.8%)** | 3.251 s; **8.696 s (72.8%)** |
| p4096 prefill | 57.730 s | 13.910 s; **43.821 s (75.9%)** | 13.910 s; **43.821 s (75.9%)** |
| p512 decode | 68.012 ms/token | 56.367 ms; **11.645 ms (17.1%)** | 41.008 ms; **27.004 ms (39.7%)** |
| p1024 decode | 68.250 ms/token | 58.672 ms; **9.578 ms (14.0%)** | 46.890 ms; **21.360 ms (31.3%)** |
| p4096 decode | 94.461 ms/token | 67.714 ms; **26.747 ms (28.3%)** | 52.700 ms; **41.761 ms (44.2%)** |

Every profile artifact and candidate decision now uses one Amdahl row with:

- `W`: current unprofiled complete wall for the exact workload;
- `C`: current same-host comparator wall under the same protocol;
- `O`: the current path's **exclusive** owner wall, normalized per request or
  token; and
- `s`: a locally measured owner speedup. A source or comparator ratio is
  recorded separately as a hypothesis until the local leaf screen runs.

The zero-cost ceiling is `O/W`; the maximum complete-wall speedup is
`W/(W-O)`. A realistic candidate projects `saved = O*(1-1/s)` and
`gap_coverage = saved/(W-C)`. `s` is `unknown` until measured; a competitor
kernel ratio is a hypothesis, not a local result. Kernel, submission, copy, and
host-stage buckets must be mutually exclusive before they are added. In
particular, graph savings and the kernels hidden by that graph cannot be summed
from separate traces. The unprofiled wall is always the denominator; profiler
API time is attribution evidence only.

#### Current impact queue

| Rank | Lane and measured owner | Amdahl interpretation | Next decision |
| ---: | --- | --- | --- |
| Blocked | Exact p4096 QSA decode: the retained ordered three-pass route reduces four-category complete wall **93.912→80.061 ms/token** and the traced QSA operation role from **36.304 to 20.913 ms/token**. | Local complete-wall `s=1.173`; the unit saves **13.851 ms/token** and covers **35.4%** of its measured final-valid gap. Exact four-column-per-thread value grouping loses its leaf screen and is removed. | Require a new exact data-reuse mechanism or fresh profile before more value-pass scheduling. Do not revisit partial-softmax merges. |
| Blocked | Operation-complete prefill MoE owns **3.408/6.307/25.398 s** at p512/p1024/p4096. Exact chunks activate a median **333/327/325 of 512 experts** with seven rows per active expert. | Exact worker/output/team schedules are exhausted. Early exact layers already use device maps. A worst-case guarded grid removes the remaining WMMA-suffix D2H but is neutral at **1.0008x**, 95% CI **0.9994–1.0022**. | A paying route needs new projection/activation reuse or device-sized indirect dispatch, neither currently exists in-tree. Continue the highest independent owner. |
| 1 | Dense linear + GR-read roles own **1.839/3.442/13.864 s**; aligned dense-projection delta reaches **1.131/2.079/8.429 s**. | No single projection closes prefill, but these uniform boundaries cover **48.3%/39.6%/31.6%** of the current final-valid gaps at zero cost. | Fuse shared layouts across `attn_qkv`, `attn_gate`, `ssm_out`, HC projection/read, and inject/publication boundaries. |
| 2 | p4096 QSA prefill attention is **10.229 s** versus llama's **0.526 s**; its share grows from **0.85%** at p512 to **18.68%** at p4096. | Its p4096 zero-cost role ceiling covers **23.8%** of the final-valid gap, but the short rows are only 1–2% owners. | Differential-profile the exact flash/index materialization and build a prefill-specific path; do not assume the decode design transfers. |
| 3 | Short decode selected Q4+Q5_1 projection deltas total **6.617 ms** at live 513; dense-Q8 adds **2.866 ms**. | Together they cover about **38.7%** of the p512 final-valid gap and most of its HIP gap after interactions; measure one exclusive operation-complete owner at a time. | Rank selected projection/data reuse before another submission-only campaign; preserve strict ordered weighting. |
| 4 | GDN owns **0.655/1.233/4.983 s** prefill versus llama **0.083/0.222/1.845 s**; live-513 is **2.670 vs 0.531 ms**. | Zero-cost GDN covers **17.2%/14.2%/11.4%** of final-valid prefill gaps and **8.7%** of the p512 final-valid decode gap. | Port the multi-column transposed-state-in-register mechanism as an operation-complete owner, not another transpose sidecar. |
| Admit | P8 whole-transition production graph. | Production `O` and `s` are **unknown**. The strict 68.855-ms runner denominator and 61.910-ms graph row came from different runs, so 1.112x/6.945 ms cannot rank this rung. | Admit only from a same-session named-production graph arm. Do not integrate from the current denominator. |
| Defer | MTP full draft head: 3.153 ms per proposal. | Even a free head saves only **0.97%** of retained suite wall and remains 0.964x AR. | Keep below device-output and target-verifier work until its complete-wall ceiling rises. |

These owners come from one current exact-token ledger and are mutually
exclusive within each hipEngine trace. Comparator deltas use a shared complete
symbol-family taxonomy; source ratios remain hypotheses, never local `s`.

#### P8 denominator audit: both published ratios are unrankable for production

The original P8 artifact's **194.758→61.910 ms/step (3.15x)** compared a graph
against a probe-local eager arm that disabled the shipped per-layer MoE graph
cache and drove 48 layers from a script loop. The follow-up denominator correctly
measures those effects, but its claim that **68.855→61.910 ms (1.112x)** is the
"honest" named-production graph speedup also overreaches:

| Row at the shallow probe context | Profile / process | Median | Valid conclusion |
| --- | --- | ---: | --- |
| Runner, MoE graphs on, device argmax | strict; denominator process | 68.855 ms | Valid strict runner diagnostic |
| Runner, MoE graphs off, device argmax | strict; denominator process | 143.989 ms | Shipped MoE cache is 2.091x versus this disabled route |
| Probe-local eager loop | strict; graph-probe process | 194.758 ms | Invalid named baseline |
| Whole-transition graph | strict; graph-probe process | 61.910 ms | Exact mechanism row; not counterbalanced against 68.855 ms |

The follow-up imports 61.910 ms from a different run, and its generator helper
hardcodes `ExecutionProfile.STRICT`. Therefore **1.112x and 6.945 ms are a
cross-run strict diagnostic, not a same-session named-production A/B**. They do
not define `O`, do not clear section 5.1, and cannot place P8 at rank 2. The
current decision is admission-pending.

The audit still retains two useful findings. First, the old 3.15x denominator
was inflated by work production already removed. Second, device argmax versus
host full-logit D2H differs by at most **0.35 ms/step** with a sign flip, so AR
output publication is not a measured decode lever; P11 must justify that work on
the MTP draft step. The denominator run also has zero steady allocation growth
and clean teardown. Evidence:
[`2026-09-01-gfx1151-qwen38-flash-next-p8-production-denominator.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-production-denominator.json)
and the superseding
[`canonical impact profile`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json).

**This is now a binding rule, not an anecdote.** A candidate's baseline arm must
be the named path with its shipped optimizations enabled. Disabling an existing
production optimization to construct a slower "before" is an invalid denominator
and its ratio is not retainable. Every `W/C/O/s` row states which routes were
active in the baseline arm.

### 2.3 Historical external-fork shape refresh

The 2026-08-30 refresh used the existing four-part `UD-Q4_K_XL`; no new
weight quant was downloaded. EngramHalo HEAD remained `1423f689...`. Its
locally built HIP binary also applied the two patches used by the fork's
published container (`#25992` host-buffer workaround and per-buffer mmap).
Nathan's toolbox HEAD was `a8631df...`, its source branch and v0.7.2 payload
were both `ad914eb...`, and a local source build reproduced the release within
1%. Compact evidence:
[`2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json).

| Same weights, BF16 K/V, `llama-bench` shape | p508 | p1012 | tg32 |
| --- | ---: | ---: | ---: |
| EngramHalo HIP HEAD + documented build patches | **296.12** | **362.72** | **17.62** |
| Nathan Vulkan v0.7.2 payload | **413.04** | **396.25** | **23.85** |
| Nathan Vulkan local build of the same source | 416.41 | 394.57 | 23.95 |

These are **historical shape diagnostics**, not current targets, a source-only
A/B, or a replacement for the frozen role profile. They use generated
`llama-bench` inputs, `-b 8192 -ub 2048 -t 4`, lazy mmap, and fork-recommended backend
settings. hipEngine's p508 row uses the committed text fixture, while its tg32
diagnostic starts from token `9707`; prompt-dependent MoE routing and PLE page
locality make exact-token matching material. P0 therefore adds one durable
cross-engine exact-prompt harness before closure comparisons.

Q8 K/V diagnostics were EngramHalo **301.07/361.75/17.85** and Nathan
**420.99/393.79/23.67** at p508/p1012/tg32. They do not replace the BF16-KV
campaign denominator. Nathan lazy mode partially reproduced its published
mechanism on this 128-GiB host: BF16 p508 averaged **329.23 off vs 413.04 on
(1.255x)**, but the off arm warmed from 271.87 to 396.97 tok/s inside its three
repetitions and p1012 was neutral (**399.67 off vs 396.25 on**). Decode was
also neutral. Cold/warm cache state must remain separate.

An EngramHalo MTP check on the full ten-prompt category+heldout suite reduced
complete request wall from **17.75 to 15.74 s (1.128x)** at 94.55% draft
acceptance, but AR/MTP message hashes matched only **9/10** prompts; the
`general_ja_plan` continuation differed while repeated AR was stable. That MTP
row is a correctness-failing diagnostic, not a speed target. hipEngine must
beat a true same-protocol AR denominator while retaining its own exact/full
profile gate; it must not copy a competitor's invalid speed row.

### Current exact-token device windows

| Window | hipEngine kernel sum | patched llama HIP kernel sum | llama advantage | hipEngine / llama rows |
| --- | ---: | ---: | ---: | ---: |
| p512 prefill | **5.972 s** | 1.926 s | **3.10x** | 3,135 / 9,944 |
| p1024 prefill | **11.196 s** | 2.990 s | **3.74x** | 6,210 / 9,984 |
| p4096 prefill | **54.762 s** | 10.838 s | **5.05x** | 24,900 / 14,904 |
| live-513 decode | **51.062 ms** | 41.009 ms | **1.25x** | 1,764 / 4,349 |
| live-1025 decode | **53.262 ms** | 41.110 ms | **1.30x** | 1,764 / 4,349 |
| live-4097 decode | **87.732 ms** | 44.005 ms | **1.99x** | 1,812 / 4,334 |

Every hipEngine row is inside an exact ROCTX operation boundary; role coverage
is **100%**. The comparator consumes the same code token arrays, and cached
decode evaluates exactly one appended root token. Both symbol-family ledgers
sum exactly to their device totals. Kernel rows are dispatch records, not host
launches.

### Prefill operation and aligned-delta ledger

hipEngine's exclusive operation roles are:

| Role | p512 | p1024 | p4096 |
| --- | ---: | ---: | ---: |
| Routed MoE | **3.408 s** | **6.307 s** | **25.398 s** |
| Dense linear projections | 1.401 s | 2.630 s | **10.589 s** |
| GDN mixer | 655 ms | 1.233 s | 4.983 s |
| GR read/tail | 437 ms | 812 ms | 3.275 s |
| QSA mixer/index/attention | 56 ms | 190 ms | **10.423 s** |
| Root/PLE boundary | 13 ms | 24 ms | 94 ms |

The matched aligned-family deltas explain the old remainder rather than leaving
it unknown:

| Family delta, hipEngine minus llama | p512 | p1024 | p4096 |
| --- | ---: | ---: | ---: |
| Dense projection compute | **1.131 s** | **2.079 s** | **8.429 s** |
| Selected Q4 gate/up | 668 ms | **1.677 s** | **8.327 s** |
| Selected Q5_1 down | 673 ms | **1.568 s** | **7.328 s** |
| Dense Q8 | 688 ms | **1.351 s** | **5.650 s** |
| GDN | 572 ms | **1.011 s** | **3.137 s** |
| QSA attention | 38 ms | 131 ms | **9.703 s** |
| Elementwise/materialization | 185 ms | 216 ms | 549 ms |
| MoE routing | 99 ms | 180 ms | 634 ms |

The earlier **41.6% unattributed remainder was a reporting artifact**: it
subtracted five selected families from the total and called every omitted
family unknown. It must not be used for prioritization. The current ledger also
shows why one prefill kernel cannot close parity: MoE leads overall, dense/GR
is a second portfolio, and QSA becomes a co-leading p4096 owner.

Exact telemetry further constrains MoE design. Each 512-row chunk selects ten
experts per token and activates a median **333/327/325 of 512 experts** at
p512/p1024/p4096, with a median seven rows per active expert. Compact-map work
must handle hundreds of active experts; tiny-active-set specialization would be
benchmark-shape mismatch.

### Decode aligned deltas per token

| Family delta, hipEngine minus llama | live 513 | live 1025 | live 4097 |
| --- | ---: | ---: | ---: |
| QSA attention | 1.963 ms | 3.985 ms | **34.996 ms** |
| Selected Q5_1 down | **3.379 ms** | **3.640 ms** | **4.919 ms** |
| Selected Q4 gate/up | **3.238 ms** | **3.231 ms** | **4.112 ms** |
| Dense Q8 | 2.866 ms | 3.058 ms | 3.035 ms |
| GDN | 2.139 ms | 2.166 ms | 1.932 ms |
| Other dense projection compute | 0.953 ms | 0.878 ms | 1.012 ms |

At live 4097, QSA alone explains more than the complete device gap because
llama is slower in some smaller routing/layout families. Candidate arithmetic
must therefore use hipEngine's exclusive operation owner and complete wall,
not sum positive family deltas as independent savings. At live 513 the selected
Q4+Q5_1 delta is 6.617 ms and dense Q8 adds 2.866 ms; these are the measured
short-decode kernel portfolio below QSA.

The unprofiled and profiled runs use the same exact fixture but are separate
processes, so wall-minus-kernel remains an order-of-magnitude residual, not
host-overhead proof. P8 receives no residual by subtraction; only a same-session
production graph arm can measure its exclusive saving.

### Invalid path removed

The previous "GDN colwarps decode all layers" row is invalid: its selector sat
below the `rows == 1` branch and compared the strict owner with itself. Wiring
the actual candidate costs **6.832 + 0.117 ms/token**, versus **2.454 ms/token**
for the retained serial-column owner, and lowers full decode. Commit
`15a436766` removed the dead route. The corrected evidence is
[`2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json).

## 3. Profiling pattern

The committed tools replace the earlier `/tmp` harnesses. Use the same sequence
for every campaign claim.

### 3.1 Freeze identity

1. Record repository, host, power, model, quant, profile manifests, and
   comparator revisions before measuring.
2. Confirm HIP and device visibility:

   ```bash
   python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
   rocminfo | grep -E 'Name:|gfx'
   ```

3. Prebuild hipEngine kernels outside the profiler and pass
   `--compiler-version-file` plus `--require-cached-build` to the profiled
   process. Do not let `rocprofv3` spawn `hipcc` or clang children.

### 3.2 Build the comparator once

The canonical 2026-08-30 refresh used current upstream HEAD
`f1793c1c4e586022efa0b1d3aa6e30ccd67f4e2d`. The pristine Release binaries are
HIP `020d0e94...` and Vulkan `c6c9dd2b...`; HIP uses
`AMDGPU_TARGETS=gfx1151`, `GGML_HIP_GRAPHS=ON`, and
`GGML_HIP_MMQ_MFMA=ON`, while Vulkan uses RADV/Mesa 26.2.1 in a separate build
tree. Pristine HIP is startup-blocked for this 111-GB artifact. A separately
labeled HIP binary `bb41c755...` applies only the documented host-buffer and
per-buffer-mmap patches (`aca70db1...` and `971d428d...`). EngramHalo HIP is
`0514f125...` at `1423f689...` plus the same patches; Nathan Vulkan is
`d3dbb492...` at `ad914eb...`. Refresh any comparator only as a separate
baseline event; do not report old and new absolute rows as an optimization A/B.

### 3.3 Collect canonical exact-token wall rows

The committed fixture
[`qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json)
has SHA-256 `42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1`.
It derives one code, English, Japanese, and mixed Japanese/English token array
at each canonical length from the ten-prompt MTP bench source. Every engine
consumes those exact IDs. Regenerate it only as an explicit protocol change:

```bash
MODEL_ROOT=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
uv run python scripts/qwen4exp_canonical_ar_bench.py fixture \
  --model-root "$MODEL_ROOT" \
  --output benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json
```

Run hipEngine from a clean worktree after the current compiler cache is warm:

```bash
HIPENGINE_HIP_ARCH=gfx1151 \
uv run python scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL_ROOT" \
  --compiler-version-file /tmp/hipengine-qwen4exp-hipcc-version.txt \
  --require-cached-build \
  --output /tmp/qwen4exp-canonical-hipengine.json
```

Run each llama.cpp-compatible lane with its exact binary/source identity and
backend settings. Arguments that begin with `-` use the `--server-arg=value`
form:

```bash
MODEL_PART="$MODEL_ROOT/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
uv run python scripts/qwen4exp_canonical_ar_bench.py llamacpp \
  --server-bin /path/to/llama-server \
  --source-root /path/to/source \
  --model "$MODEL_PART" \
  --engine-label ENGINE_LABEL \
  --server-arg=-ngl --server-arg=999 \
  --server-arg=-fa --server-arg=on \
  --server-arg=-ctk --server-arg=bf16 \
  --server-arg=-ctv --server-arg=bf16 \
  --server-arg=-c --server-arg=4352 \
  --server-arg=-b --server-arg=8192 \
  --server-arg=-ub --server-arg=2048 \
  --output /tmp/qwen4exp-canonical-ENGINE_LABEL.json \
  --server-log /tmp/qwen4exp-canonical-ENGINE_LABEL.server.log
```

The driver discards one warmup and records three measured requests per case.
The first sampled output belongs to prefill; it requests 129 visible outputs
from llama.cpp and reports exactly 128 post-first-output transitions, matching
hipEngine's 128 timed `runner.step()` calls. Prompt cache is disabled while OS
page-cache state is warm. Synthetic `llama-bench` p512/p1024/p4096/tg128 rows
remain a separate `shape_only` diagnostic class.

For a route-local prefill A/B, keep one model residency and reverse the first
route in each pair with `scripts/qwen4exp_route_pair.py`. Select exact fixture
cases explicitly and treat the override arm as diagnostic until its complete
profile gate passes:

```bash
uv run python scripts/qwen4exp_route_pair.py \
  --model-root "$MODEL_ROOT" \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --case-id code-p512 --case-id general_en-p512 \
  --case-id general_ja-p512 --case-id mixed_ja_en-p512 \
  --pairs 5 --warmups 1 --require-cached-build \
  --override HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL=1 \
  --output /tmp/qwen4exp-layer2-p512-pairs.json
```

### 3.4 Collect role-resolved device traces

The current ledger consumes exact fixture token IDs. Example p512 prefill:

```bash
TRACE=/tmp/qwen4exp-role-p512
rm -rf "$TRACE" && mkdir -p "$TRACE"
HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-qwen4exp-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 \
rocprofv3 --kernel-trace --hip-trace --marker-trace \
  --memory-copy-trace --memory-allocation-trace --output-format csv \
  -d "$TRACE" -o role-p512 -- \
  uv run python scripts/qwen4exp_profile_gap.py \
    --model-root "$MODEL_ROOT" --mode prefill \
    --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
    --case-id code-p512 --profile --role-markers --repetitions 1 \
    --compiler-version-file /tmp/hipengine-qwen4exp-hipcc-version.txt \
    --require-cached-build --output "$TRACE/child.json"
```

Use `scripts/qwen4exp_context_decode_profile.py --live-count 513 1025 4097
--repetitions 3 --profile --role-markers` for exact context-conditioned decode.
It restores one snapshot before every transition, hashes every mutable owner,
and scopes allocation growth separately to each warmed context bucket. A
cross-bucket resize is reported but is not relabeled as per-step growth.

To profile an isolated candidate that the named profile binder normally resets,
use a repeatable post-binder `--override`. The child records bound/effective
environments and `named_profile_intact: false`; that row remains diagnostic
until the complete profile gate passes.

For exact llama.cpp attribution, use
`scripts/qwen4exp_llamacpp_exact_profile.py`. It launches the pinned server
directly under `rocprofv3`, sends the same fixture arrays, and records monotonic
bounds for p512/p1024/p4096 prefill plus cached live-513/1025/4097 one-token
decode. Select rows with `qwen4exp_trace_analyze.py --start-ns ... --end-ns
...`. Dynamic attach is not supported by this packaged ROCm SDK; do not change
host ptrace policy. The retained comparator CSVs flushed completely, although
the rocprof wrapper required forced exit after flush; they remain diagnostic
attribution, not a promotion gate.

### 3.5 Analyze without conflating rows and launches

```bash
uv run python scripts/qwen4exp_trace_analyze.py \
  --trace-dir "$TRACE" --engine hipengine \
  --marker-prefix qwen4exp_prefill_p512_ \
  --output "$TRACE/summary.json"

uv run python scripts/qwen4exp_role_analyze.py \
  --trace-dir "$TRACE" \
  --measure-prefix qwen4exp_prefill_p512_ \
  --output "$TRACE/roles.json"
```

The analyzer reports the selected marker or explicit clock window, kernel
sum/span, row counts, complete family totals, HIP API launch correlations,
unmatched graph/copy rows, allocation events, and memory-copy rows. The role
analyzer correlates ROCTX ranges to HIP launch correlation IDs and kernel rows,
then reports attributed/unattributed time coverage, normalized/exact roles, and
a flat exact-role kernel/API breakdown. Collect actual per-layer expert-row
distributions in a **separate**, non-profiled instrumentation run so its D2H
telemetry cannot contaminate the role trace:

```bash
uv run python scripts/qwen4exp_profile_gap.py \
  --model-root "$MODEL_ROOT" --mode prefill \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --case-id code-p512 --repetitions 1 --moe-telemetry \
  --require-cached-build --output /tmp/qwen4exp-p512-moe-telemetry.json
```

`scripts/qwen4exp_perf_gap_report.py` renders the compact artifact as markdown
tables:

```bash
uv run python scripts/qwen4exp_perf_gap_report.py \
  benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json
```

### 3.6 Lifecycle profiling — required for every candidate

Steady-state wall is half of a measurement. A kernel or graph that is fast in a
warm loop but reallocates, leaks, or diverges on reuse is not retainable, and
these failures are invisible in a median. Every candidate collects the
following, and every compact artifact carries a `lifecycle` block. **A candidate
without one is not eligible for promotion**, regardless of its speed row.

1. **Tracked device memory, construct to close.** Wrap the run in
   `hipengine.core.memory.reset_memory_stats()` / `memory_stats()` and record
   `total_allocated_bytes`, `total_freed_bytes`, `peak_allocated_bytes`,
   `active_allocations`, and `peak_allocations`. Require
   `active_allocations == 0` and `current_allocated_bytes == 0` after close.
2. **Steady-state allocation growth.** Sample allocation count and current
   bytes after warmup and after the measured window; require **zero** growth.
   For multiple context buckets, gate each warmed bucket independently and
   report first-use cross-bucket resizing separately. A per-step allocation is
   a leak, not a fast path.
3. **Allocation timing under the profiler.** Add
   `rocprofv3 --memory-allocation-trace` beside the kernel/HIP traces. For any
   capture/replay unit require **zero device allocations at or after the first
   graph launch**, and confirm capture is non-executing.
4. **Per-step census, never per-run.** From the same trace, report kernel
   dispatches, direct `hipLaunchKernel` correlations, graph launches, and
   memcpy rows **per step** inside the marker window. Aggregate counts hide
   exactly the launch growth this campaign is trying to remove.
5. **Replay and state gates.** At least three consecutive replays,
   `reset → replay`, `graph → forced eager → graph` resumption, and
   snapshot/restore. Compare **every mutable owner by hash**, not only the
   output: K/V, index cursors, recurrent state, PLE history, position/context
   scalars. An output-only check passed on hardware that later failed a state
   check.
6. **First-arm and cold/warm separation.** Discard the first arm after model
   load; it pays first-touch weight paging and clock ramp, and it moved the
   named decode step by **35%** in the denominator harness above. Cold-cache
   rows use the isolated protocol, and one process's warming repetitions are
   never independent samples.
7. **Concurrency, cancellation, and teardown.** Physical c2 isolation,
   cancellation mid-step, and teardown with the tracked-memory check from
   item 1.
8. **Nested-process rule.** Never wrap a parent harness that spawns Python
   children in `rocprofv3`; profiler and JIT state propagate into the children.
   Profile the child directly, or use `scripts/mtp_verifier_rocprof.py`.
   Prebuild kernels and pass `--compiler-version-file` plus
   `--require-cached-build` so no profiled process spawns `hipcc`.

Use shared model residency, a discarded first arm, counterbalanced named-path
arms, per-arm route state, graph-cache census, and a lifecycle block. The P8
denominator harness demonstrates those mechanics for its runner arms, but its
imported graph value is not a counterbalanced arm and must not be copied as an
A/B pattern.

Interpretation rules:

- End-to-end unprofiled wall is the headline.
- Kernel sum ranks device dataflow.
- Kernel span minus sum exposes gaps, but profiler inflation is not Python
  overhead without a separate unprofiled event/control.
- Direct `hipLaunchKernel` correlations count host submissions; graph-expanded
  kernels and copy/fill kernels appear as trace rows without direct launches.
- A/B decisions use same-session counterbalanced orders and identical IDs or
  the applicable production-profile gate.

## 4. External evidence review

This section records useful external hypotheses and their evidentiary status.
None of these numbers are hipEngine results.

| Source | Mechanism or claim | Status for this campaign |
| --- | --- | --- |
| [Sleeping Robots, 2026-08-29](https://sleepingrobots.com/dreams/engramhalo-qwen38-flash-next-strix-halo/) | Independently tested EngramHalo on Strix Halo with a different quant. MTP reaches 28-38 tok/s at working depths; 26K MTP regresses to 15.0; kernel-only prefill improves up to about 35% at 26K. | Useful cross-check of the direction, not a same-quant baseline. Confidence: medium-high for the external fork, low for transfer magnitude. |
| [apepojken/llama.cpp `843d575`](https://github.com/apepojken/llama.cpp/commit/843d5750579a15ed4a42d73eb862855c271021ac) and local survey, 2026-08-31 | Vulkan rollback/MTP fixes, pooled QSA keys, gathered attention, radix top-k, GDN/inject dataflow, and epilog fusion. Matched Q4/BF16 reaches 291.73/23.21, 375.23/22.42, and 397.43/22.25 pp/tg128 at p512/p1024/p4096. Static logits remain 160/160 top-1 vs upstream Vulkan, but only 8/12 AR cases repeat and Q8-KV MTP is 9/10 AR-message exact. | Fast experimental lane only. Its published Q3 50.4 tok/s headline remains author-reported; the matched AR and MTP rows fail this campaign's exact-output contract. See the Strix Halo survey. |
| Local source/build refresh, 2026-08-30 | Existing `UD-Q4_K_XL` runs in both forks. EngramHalo BF16 reaches 296.12/362.72/17.62 and Nathan v0.7.2 reaches 413.04/396.25/23.85 at p508/p1012/tg32. Nathan lazy-on is 1.255x over the cache-cold-to-warm off average at p508 and neutral by p1012. Engram MTP is 1.128x complete-wall but only 9/10 AR-message exact. | Historical same-host shape evidence, superseded for AR targets by the exact-token screening in section 2.1. The MTP speed row fails correctness and remains diagnostic only. |
| [Pat1entZ3r0/strix-qwen-next-flash-optimization](https://github.com/Pat1entZ3r0/strix-qwen-next-flash-optimization) `413c33c`, source-reviewed 2026-09-02 | Single-commit program consolidating the three forks this campaign already tracks: `SOURCE_LOCK.json` pins base `c589f0ed1` + #27879 and reference forks apepojken `843d575`, Nathan `ad914eb`, EngramHalo `1423f689`. 40 patches in two lines (`hybrid-04` correctness/perf, `hybrid-03-mtp` draft head + rollback fixes), 58 claimed experiments, and a published nulls catalog. Headline 2.5-3x decode and 2.5x prefill on Vulkan with a custom IQ4_XS + dense-Q6K quant, Q8 K/V, `-ub 2048`, and an EasiiX Q8_0 MTP sidecar. | Author-reported; nothing locally reproduced. No rate binds: different quant, K/V, backend, and host instance. The headline moves four axes at once, and its `pr-27742-035e227` baseline predates #27879, so the denominator still carries the QSA block-selection bug that the program's own patch `0001` fixes. Its correctness gate has no repeat arm. The value is the rollback-ring mechanism, the nulls catalog, and the fact that its `MODEL_LOCK.json` pins this campaign's exact UD-Q4_K_XL shards (revision `c8b5954a`, all four shards LFS-verified) while publishing no row for them. Full review: survey section 6.7. |
| [Aristo94/EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp), refreshed at `1423f689986f670417128fd545a0aa1241166103` | Wide radix top-k (`33766da`), masked-slice FA skip (`bf8412d`), QSA top-k row gather (`2606d49`), MTP sidecar (`afb80ed` + `2ba3009`), PLE lazy row prefetch (`c911e6b`), and load-page drop-behind (`5486559`). Chunked GDN prefill exists (`62160a7`) but was explicitly not active in the published numbers. The published container additionally applies the tracked #25992 host-buffer and per-buffer-mmap patches. | Code and build mechanisms verified by source inspection and a local gfx1151 HIP build. hipEngine already covers the QSA selector/gather direction; PLE advice/prefetch, loader drop-behind, full-step graphing, and MTP economics remain open. |
| [Nathanw1014/strix-halo-llamacpp v0.7.2](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.7.2), toolbox HEAD `a8631dfbf0aeb6a4004866fce1fd7e5c10370049`, source `ad914eb6587d3da8b2bf50f0056cc20b3d3e91f5` | `TENSOR_READ_LAZY` + `MADV_RANDOM` alone loses; merged `WILLNEED` row prefetch is the paying half (`77362a8`). Qwen4Exp also adds host PLE gather and reusable decode topology (`631b9ff`), per-block QSA bias (`024b7ad`), and MTP graph/context (`3543908` + `39817c4`). Vulkan lineage includes MoE row lists (`212cca8`), route-scale epilogue, SiLU/mul fusion, transposed concat (`30d8bb0`), dense wave32 (`25c45fe`), and LDS padding (`baf6360`). | Source mechanisms verified; release and local source builds agree within 1% on this host. Vulkan shader topology is not portable to HIP, but the removed data movement, host synchronization, graph rebuild, and LDS-bank mechanisms are actionable. |
| [quimmedes/cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp), observed HEAD `2da84198eccb0aee59abba59e967dcc61f84ce07` | The fresh fork exposes pinned-host/CPU routed-expert placement, PLE n-gram SSD mmap or disable modes, and Qwen4Exp MTP trunk/combiner fixes. Commits `ba7bd23` and `7ee981d` add the PLE controls; `19aefd2` and `d98dc18` address MTP hidden export and mixer mapping. | Track as a source lead, not a measured comparator. SSD PLE and host-placement ownership may inform P9; `--no-ngram` changes the model and cannot close parity. No same-weight local rate or correctness packet has been verified. Confidence: high for repository/commit identity, medium for transfer applicability. |
| [omlx PR #3260](https://github.com/jundot/omlx/pull/3260), open head `3343e4414f75b9808d2d8a6de1950ad96ce8dac8` | Adds row-addressable SSD expert reads, fixed preallocated expert banks, manifest pins plus an evictable hot tier, learned route-frequency hotlists, expert-major overflow chunks, and checked/speculative miss handling with transactional KV/SSM restore. The author reports exact expert output at a 0% substitution threshold. | Track the fixed-bank, telemetry, hotlist, and transactional retry mechanisms for constrained-residency work. This is an unmerged Apple MLX/safetensors path with a dirty merge state, not direct HIP/GGUF code or local performance evidence. Confidence: high for PR state/design, low for transfer magnitude. |
| [exllamav3 PR #303](https://github.com/turboderp-org/exllamav3/pull/303), open head `5705f07b39671746af336bb004ad2e324410a654` | Builds a selected draft-only vocabulary head, keeps draft IDs on GPU through the block, and leaves full-vocabulary target verification unchanged. The author reports CUDA/SM89 Qwen3.8-27B MTP-4 **36.584→44.579 tok/s (+21.9%)** with a grouped 64K head; follow-up comments report separately quantized individual-row 8K/16K/32K heads at 0.032/0.165/0.329 ms versus 0.932 ms for grouped 64K, but acceptance varies with the proposal map. | Mechanism: high-confidence source match. Transfer magnitude: low confidence because backend, model, quant, draft depth, and verifier differ. On gfx1151 the current Qwen4Exp 248,320-row Q8_0 draft head is **3.153 ms / 41.3%** of a 7.639-ms draft step, but eliminating it entirely would improve the retained full-suite wall only **0.97%** and **0.955x→0.964x AR** because serial target verification dominates. P11 should first remove host draft outputs, then test individual Q8_0 rows—not EXL3's 128-token groups—after target verification improves. |
| [halo-box/strix-llama.cpp PR #11](https://github.com/halo-box/strix-llama.cpp/pull/11), head `a7ad7b7f` on base `6c84c7d5`, source-reviewed 2026-09-02 | gfx1151-gated ROCm prefill package: MMQ tile retune (256→128-wide Q4_K/Q5_K/Q6_K/Q8_0), parallel top-10 `mm_ids` compaction, device-built routed-compact J48 MMQ (Q6_K/Q8_0 only), fused weighted top-10 expert sum + shared mul-add-residual, 32-warp/tile-16 GDN, Q8_0-KV decode FA opt. Author-reported +44.8%/+25.6%/+15.3%/+13.5% PP2048 at depth 0/12K/32K/64K on UD-**IQ4_XS** with byte-identical-logits correctness claims; faster FA tiles were rejected for logit drift. | Author-reported; different quant, PP2048 shape, and a base older than pinned upstream `f1793c1c4`; nothing locally reproduced yet. The mechanisms map directly onto the blocked prefill-MoE, dense/GR, and GDN owners. Dedicated follow-up: [`QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-HALO-BOX-CAMPAIGN.md). |
| Upstream llama.cpp [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) | Qwen4Exp architecture support; merged at `6c84c7d5`. | Already represented in the fresh comparator. |
| Upstream [#27794](https://github.com/ggml-org/llama.cpp/pull/27794) | `TENSOR_READ_LAZY` plumbing; merged at `fac889fb`. Nathan's branch keeps the missing batched row-prefetch half. | Useful PLE hypothesis. |
| Upstream [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) | Qwen4Exp NextN/MTP draft head; open. Its key note is that the hyper-connection combiner must run per stream; mean pooling first destroys acceptance. | Matches our retained lesson; use it to audit, not re-derive, the Qwen4Exp MTP combiner. |
| Upstream [#26592](https://github.com/ggml-org/llama.cpp/pull/26592) and [#26388](https://github.com/ggml-org/llama.cpp/pull/26388) | hipCUB/CUB paths for top-k/argsort on HIP. | Superseded for our purposes by the in-tree GPU QSA selector; keep as lineage context only. |
| Upstream [#27466](https://github.com/ggml-org/llama.cpp/pull/27466) | ROCm radix top-k for long rows; open. | Confirms the long-context failure mode; hipEngine already uses a device selector. |
| Upstream [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | Chunked GDN prefill using tensor-core fragments; CUDA/NVIDIA-focused and open. | Hypothesis only. Our nearest unit is the retained colwarps owner plus the rejected decode correction. |
| Nathan's Vulkan evidence pack and branches | q8 KV dequant-once, contiguous K/V, MoE row-list prepass, scale epilogue, SiLU/mul fusion, concat transpose. | Vulkan/RADV-first; useful patterns, not HIP evidence. Map only the algorithmic dataflow, not shader specifics. |
| Upstream [#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | Vulkan q8_0 KV dequant-once for prefill; merged at `dc72703f`. | Not directly applicable to HIP; reinforces "dequantize/reorganize once, then attend". |
| Upstream [#26419](https://github.com/ggml-org/llama.cpp/pull/26419) | MMA FlashAttention at head-dim 256 on RDNA; open. | Relevant to QSA prefill geometry, but measured on RDNA4, not gfx1151. |
| Upstream [#27880](https://github.com/ggml-org/llama.cpp/pull/27880) | qwen4exp graph-split reduction; merged at `6fe74980`. | Already in the remote-HEAD comparator. |
| Upstream [#27925](https://github.com/ggml-org/llama.cpp/pull/27925) and [#26686](https://github.com/ggml-org/llama.cpp/pull/26686) | Vulkan MoE padding/row-ID changes that improve the Vulkan comparator. | Already in the remote-HEAD Vulkan comparator; no HIP action implied. |

#### Externally published claims this host does not reproduce

Three widely repeated external statements fail or do not transfer here. Record
them so no unit is planned or rejected on their authority.

- **"ROCm/HIP is 25-30% slower than Vulkan on gfx1151."** Pat1entZ3r0's
  `REJECTED.md`, from one containerized ROCm-10 experiment reported as four
  percentages with no rates. The matched exact-token screen in section 2.1 does
  not reproduce the prefill half: patched upstream HIP is level at p512 and
  ahead at p1024/p4096 (**301.68 vs 259.73** and **294.47 vs 266.98**). Only the
  decode half is directionally consistent (**17.74/16.88/14.77 vs
  22.97/20.11/18.07**, -23%/-16%/-18%). It is a decode-only, build-specific
  result, not a backend-level fact, and it does not bear on the HIP-first
  ordering in section 1.
- **"`RADV_PERFTEST=cswave32` is -4%."** A Vulkan shader-compilation knob that
  does not transfer to wave32-native HIP kernels. It is not evidence against the
  P3 dense retile and LDS-padding sweep.
- **"`-ub 2048` is +26-36% prefill."** Their ubatch selects backend GEMM/MMQ
  paths; hipEngine's chunk is host-loop granularity over its own kernels, where
  the P0 sweep measured chunk 1024 at +2.25/2.55% and chunk 2048 losing. Not
  transferable. Every comparator lane here already runs `-b 8192 -ub 2048`, so
  no lane is under-configured on this axis.

### 4.1 Mechanism transfer audit

The useful part of the external forks is the mechanism, not their headline rates.
This is how each mechanism maps to the current hipEngine implementation:

| Mechanism | Current hipEngine state | Campaign action |
| --- | --- | --- |
| Wide, graph-safe QSA radix top-k | Exact stable four-pass radix selection already lives in `qwen4_exp_qsa.hip`; no host sort is used. | No port. Keep it as the selector oracle and re-profile only its real long-context score width. |
| Persistent pooled QSA keys and per-block bias | Complete four-token blocks are pooled once into a persistent device buffer; selection is block-based rather than one bias value per KV cell. | Treat the Nathan fixes as corroboration. Audit metadata copies and append work, not the already-solved algorithm. |
| Gather only selected QSA K/V rows | Paged sparse QSA consumes explicit selected positions/counts rather than scanning a dense mask. | Audit selected-row count, sort, page locality, and head-dim-256 geometry at 16K+; do not rebuild llama's graph gather. |
| Host-side PLE row gather | `Qwen4ExpPLEMMapTable` gathers/dequantizes only requested rows, and a pinned two-buffer ring stages them. It still creates temporary arrays and performs synchronous gather/copy. | Preserve ownership, but add direct-to-ring dequantization, duplicate/page coalescing, telemetry, and prefill overlap in P9. |
| PLE random advice plus merged row prefetch | No equivalent `MADV_RANDOM` + page-aligned merged `WILLNEED` pair exists. | Implement both halves together with off/auto/on rollback. Never ship random advice alone; Nathan measured that half losing. |
| Load-page drop-behind | Hot weights are uploaded without an explicit per-tensor file-page release policy. | Measure transient free/available/swap/load wall, then add bounded unmap/fadvise only for copied tensors; never drop lazy PLE pages. |
| Fixed expert bank plus route hotlist | Current Qwen4Exp keeps the declared routed-expert representation resident; it has no SSD miss/promotion tier. | Keep omlx PR #3260 as a constrained-memory design lead. Any future port needs GGUF row indexing, exact 0%-substitution routing, request-safe bank ownership, transactional state restore, and cold/warm workload gates; it is not part of short-AR parity. |
| MoE row-list prepass | Device count/prefix/scatter and expert-sorted lanes already exist. | Do not port Vulkan topology. Remove the remaining D2H tile-count synchronization and the Q8 expert-start D2H + Python expert loop; use guarded fixed-capacity device grids. |
| Route-scale/weighted-down epilogue | c1 Q5 down can fuse ordered weighted sum; grouped prefill still writes expert outputs and runs a separate weighted lane reduction. | Add exact/T2 grouped down+route-weight+scatter/ordered-reduce candidates, with the current chain registered as fallback. |
| Shared-expert and router completion | Shared gate/up, SiLU, down, gate projection, cast, and combine remain separate; router writes all 512 logits before top-10. | P3 owns operation-complete shared-expert and exact stable router+top-10 candidates after role measurement. |
| Dense wave32 retile and LDS padding | gfx1151 HIP kernels are wave32-native, but each quant/shape has independent register/LDS behavior. Vulkan constants do not transfer. | Sweep actual rotating-weight shapes with compiler VGPR/LDS/scratch evidence. Port the bank-conflict/resource hypothesis, not constants. |
| Tiled transpose and SiLU/mul fusion | Exact bulk Conv and selected gate/up+SiLU cover analogous paths, but GDN state layout and shared-expert SiLU still expose traffic. | Use the transpose hypothesis in P7 and the activation hypothesis in P3/P6 only where a current trace names the traffic. |
| Reusable decode topology | 48 stateless per-layer MoE graphs are reused; the other 1,195 direct launches/token remain outside them. | P8 grows from one stateful layer to one transition and then a full step, with pointer/state/rollback/replay gates at every rung. |
| Device-owned decode boundary | Normal greedy text generation now runs exact registered F32 argmax on device and copies one int64 token; direct runner, MTP, numerical, and debug paths retain explicit full-logit output. | P5 still needs device-to-device token chaining and the complete blocking/async copy census before graph capture. |
| MTP selected-vocabulary draft head | The Qwen4Exp draft currently runs the full 675,430,400-byte Q8_0 `248320×2560` head and returns 993,280 logit bytes plus hidden state to the host for every proposal. A clean leaf diagnostic measures the head at **3.153 ms**, **41.3%** of a 7.639-ms draft step. GGUF Q8_0 output rows are independent, so EXL3's aligned Hadamard-group restriction does not apply. | First use existing device argmax and one compact candidate packet; full draft logits remain debug-only. Then gather individual Q8_0 rows into default-off 8K/16K/32K heads (21.25/42.5/85 MiB) with a local→global ID map. Build the map from unrestricted full-suite plus training-only telemetry, gate category-heldouts separately, and report acceptance/economics by category. Do not expect this alone to fix current MTP: even a free head projects only **0.964x AR** on the retained suite. |
| Per-stream MTP combiner and graph | The combiner is correct, but draft hidden/logits and target full logits cross the host; target verification is serial. | Keep the combiner. P11 makes proposal, hidden chaining, verification, acceptance, commit, and rollback device-resident before budget tuning. |
| q8_0 K/V, `-ub 2048`, and hipBLASLt | These are different llama.cpp representation/config knobs. Current hipEngine is BF16 K/V at chunk 512. | Chunk size is a P0 same-representation sweep. Q8 K/V remains a separately gated T3 profile after BF16 AR parity. |
| Quantized-KV dequant-once/contiguization | There is no quantized-QSA-KV owner in the binding campaign. | Backend-disjoint evidence only until P10; it cannot close a BF16-KV milestone. |
| Fully masked FA slice skip | Current sparse attention no longer scans a dense selected-token mask, while short prefill has separate dense flash geometry. | Test only against a trace-proven masked slice in P4/P10; reject if it optimizes work hipEngine does not execute. |
| Chunked GDN prefill | hipEngine has strict serial/prepare+peer/column-warp owners; Engram's chunked kernel was not active in its published rows. | Treat it as a design hypothesis in P4. Require local arithmetic classification, state parity, and whole-role evidence. |
| Recurrent rollback ring depth and bank coverage | hipEngine owns its own checkpoint/replay for GDN conv and SSM state; no equivalent audit against the two Pat1entZ3r0 EXP-016 failure cells has been run. | Untested hypothesis, highest MTP-correctness value. Their patch `hybrid-03-mtp/0006` (after apepojken `32af70900`) claims banks `[n_written, K)` are never written and keep stale content a rollback then restores, and that the spec ring must be `n_max + 1` deep because the verify batch holds the previously sampled token plus `n_max` drafts. Symptom: a ~4.8-nat post-rejection logit shift, invisible to any single-shot logit gate. P11 owns the rejection-depth RED sweep; their own SSM clamp is self-described as "best-effort deeper" than exact, so a correct diagnosis may still be an incomplete fix. |
| Depth-conditional draft budget | P11 currently sweeps budgets 1-6 at short context only, so a budget chosen there would be frozen for every depth. | Transfer the finding, not the constants. They report n-max 2 shallow and n-max 6 at >=32K (+37% at 64K, +41% at 128K versus plain) with code acceptance decaying 0.94-0.97 shallow to 0.75-0.88 at 128K. Fit a policy over measured acceptance rather than adopting two hand-chosen constants tuned on their prompts; a constant selected that way is not retainable under the anti-gaming rule. |
| Lazy PLE from local storage | P9 already owns sparse mmap ownership, cold/warm separation, and the advice+prefetch pair. | External corroboration for the premise and a correction to the expected payoff. They report a Q8_0 PLE splice served from NVMe at identical decode speed with page cache down to ~7 GiB and ~30 GiB RAM freed, validated on 128-token reps only. Complementary negative: the IQ4_NL-PLE "91 GB" quant loses at 128K per its own publisher (18.6 vs 26.9 tok/s). Direction is stream a large PLE, not shrink it. Raises P9's memory payoff and lowers its expected speed payoff. |
| Dense-versus-expert decode dominance | The impact queue already ranks short-decode selected projections and Q8; there is no in-tree quant-axis evidence separating dense from routed-expert decode cost. | Two independent external rows agree that dense tensors, not routed experts, bind decode bandwidth at this MoE shape: a dense-Q6K re-quant gives +5-13% decode at flat PPL, while Q3_K_XL, which shrinks only the experts, gives no meaningful speed. Supporting evidence for the dense/projection ranking. As a quant change it stays a separately gated T3 product configuration with its own denominator; it cannot move the pinned UD-Q4_K_XL AR baseline. |
| Per-dispatch versus per-submit cost | P8 contracts 48 MoE graphs plus 1,195 direct launches toward one submission. | Cheap external prior on where a P8 win can come from: their `GGML_VK_MAX_NODES_PER_SUBMIT` 200-800 sweep measured -1.3 to -2.4%, concluding cost is per-dispatch, not per-submit. Supports the launch-count premise and pre-rejects any variant that only bundles submissions without removing dispatches. |
| Indexer head-sum and pooling reassociation | The QSA selector and pooling path are exact and stable; no candidate currently reassociates the indexer head sum. | Two external exactness negatives worth banking before one does. Their bit-exact r=4 indexer-key pooling uses a tree of strided adds rather than transpose/mean/transpose (EXP-005), and the head-sum slice tree from #28023 flips bits at depth and had to be made opt-in (patch `0032`). Their pooled-key cache, the largest depth win they report (+38-40% at 32-64K), is explicitly non-bit-exact in QSA selection with no published KL or top-1. Any hipEngine analogue is a production-profile candidate that must clear the numerical gates, not a free win. |

### 4.2 Direct hipEngine versus pinned llama.cpp implementation audit

This audit compares the current in-tree runtime with the compute sources behind
the pinned upstream HIP lane. The local readable checkout is
`llama.cpp-hip@17252c769`; the measured comparator is `f1793c1c4`. The Qwen4Exp
model graph, scheduler reuse, GDN kernel, and MoE graph builder are unchanged
between those revisions. The relevant measured-revision delta specializes the
HIP `mm_ids_helper` for top-10 routing. Loader patches affect startup ownership,
not these compute conclusions.

| Boundary | hipEngine implementation | Pinned llama.cpp implementation | Gap-closing action |
| --- | --- | --- | --- |
| Decode execution topology | `Qwen4ExpGGUFResidentModelRunner.step()` executes embedding, optional host PLE staging, and 48 physical layers from Python. Production `MoeGraphCache` captures only each stateless MoE subgraph. The exact full-transition graph exists only in `scripts/qwen4exp_stateful_layer_graph_probe.py`. | `src/models/qwen4exp.cpp` constructs one declarative root→48-layer→head graph and reuses topology through the backend scheduler. | Keep P8 admission-pending until the same-residency harness has a named-production graph arm. The strict cross-run 1.112x ratio is not an effect measurement. Work the larger measured QSA/MoE/dense owners first. |
| QSA selected attention | `qsa_sparse_attention_paged_bf16_f32_kernel` gives one CTA to each query head and iterates about 2,048 selected tokens serially. Every token performs QK reduction, online-softmax update, weighted-V update, and several CTA barriers. Twelve calls own 35.88 ms/token above the QSA boundary. | `build_qsa_top_k()` builds selected visibility, then `build_attn_qsa()` turns it into an attention mask and delegates the QKV work to the backend MHA/flash-attention path. It does not run a barrier-per-selected-token Qwen-specific kernel. | Build strict ordered QK-score, online-softmax-coefficient, and weighted-V-recurrence passes in P6. Preserve the current stable selector and selected-position ABI; replace only the serialized attention owner. |
| GDN recurrence/state | The strict decode kernel assigns one CTA per value head, repeatedly reads/writes strict-layout state, and combines prepare/recurrence/norm-gate through separate ownership. The rejected transposed candidate assigned one wave to each output value, creating 6,144 blocks. | `ggml-cuda/gated_delta_net.cu` assigns four output columns to a CTA, stores transposed state shards in registers across the token loop, and writes each shard once. Qwen4Exp then applies its sigmoid-gated norm in the graph. | Implement a four-or-more-column operation-complete owner with persistent transposed state. Do not retry transpose as a sidecar around the current stages. |
| Routed MoE | `run_qwen4_exp_moe()` explicitly materializes router logits, top-10, BF16 activation, count/prefix/scatter maps, quant-specific gate/up, activation, down, weighting, and reduction. Some WMMA prefill routes read `group_wmma_total` back to the host to size launches. Three exact early-MoE tile/grid schedules have already lost. | `build_moe_ffn()` expresses routing, `MUL_MAT_ID` gate/up, GLU, down, weighting, and ordered adds in one graph. HIP `mm_ids_helper` builds expert bounds plus compact forward/inverse maps on device; `f1793c1c4` specializes top-10. | Stop changing isolated tile constants. Transfer device compact-map, activation reuse, and graph-fusion mechanisms against the current **3.408/6.307/25.398-s** operation owner and the measured broad-active-expert shape. |
| PLE publication | hipEngine keeps the 28.8-GB IQ4_NL table as a sparse mmap, hashes on host, gathers/dequantizes 16 rows into pinned staging, and publishes 10 KiB before the graph. This is memory-efficient but remains a host boundary. | llama computes PLE row IDs in `llm_graph_input_ple::set_input()` and presents them as reusable graph inputs; `ggml_get_rows` and downstream PLE operations remain in the graph. Its loader needs host-specific patches for this artifact. | Keep hipEngine's sparse ownership. Make the host stage request-owned/direct-to-ring and overlap or enclose everything after publication; do not copy llama's full table residency. |
| Prefill orchestration | hipEngine runs fixed chunks (512 default) through explicit Python-owned kernels. Chunk 1024 improved only 2.3–2.6% at longer shapes. | llama's reusable graph and backend selection use larger ubatches and generic MMQ/`MUL_MAT_ID` paths. Exact matched device sums are **3.10x/3.74x/5.05x** faster at p512/p1024/p4096. | Treat prefill as a MoE+dense/GR+long-QSA data-reuse portfolio. Graphing alone cannot erase the device-sum gap. |

This source comparison narrows the campaign to mechanisms visible in both code
and profiles. It does not prove the size of any unimplemented hipEngine win;
section 2.2's local Amdahl row remains the admission gate.

## 5. Plan

Phase numbers preserve evidence lineage; they are not the execution queue. The
impact queue in section 2.2 controls what runs next. After every retained unit,
collect a fresh role/launch/copy census, recompute overlapping Amdahl rows, and
re-rank the remaining work. If a phase is blocked, record the concrete blocker
and continue the highest-gap-coverage independent phase; a blocker is not
campaign closure.

### 5.1 Impact gate and reprofile cadence

Before implementation, a campaign-critical candidate must:

1. come from the current named-production whole-path trace for the exact shape;
2. name an exclusive owner `O`, current wall `W`, comparator wall `C`, plausible
   local owner speedup `s`, zero-cost ceiling, projected wall saving, and
   target-gap coverage;
3. identify overlap with graph, copy, synchronization, host-stage, and kernel
   buckets so savings are counted once;
4. cite the concrete hipEngine and comparator implementation difference that
   could produce `s`; and
5. define the cheapest falsifying leaf test plus the binding whole-model gate.

By default, do not spend a dedicated campaign iteration when either the
zero-cost ceiling is below **1% of complete wall** or the candidate can close
below **5% of the current comparator gap**. An exception must unblock a
higher-impact owner, repair correctness, or have negligible implementation
cost. Exact small sub-window wins remain retainable under repository policy,
but they do not
outrank a larger measured owner and they accumulate in a separate ledger until
the complete-wall effect resolves.

After a topology change such as whole-step graphing, rerun the complete affected
profile before using any old kernel share. After an ordinary retained kernel
change, refresh its role plus the launch/copy census immediately and refresh the
full canonical ledger when cumulative measured savings reach 3% of wall or the
owner order changes. Two losses in one scheduling family still require a new
mechanism or profile. A micro win whose whole-role effect is below timing
resolution is bundled into an operation-complete boundary rather than tuned
again in isolation.

### 5.2 Definition of done for one optimization unit

Every code or kernel unit follows the same loop:

1. Name `W/C/O/s`, zero-cost ceiling, projected wall saving, target-gap
   coverage, arithmetic class (T0-T3), affected layers/shapes, expected
   mechanism, overlap exclusions, and registered strict fallback.
2. Add the RED oracle before implementation. For a port, run
   `scripts/check_lineage.py`, inspect source drift, and cite source path plus
   commit.
3. Microbenchmark actual immutable weights/shapes. Rotate more than 64 MiB of
   weights for c1 tests so MALL does not fake DRAM throughput. Use warmups plus
   counterbalanced pairs and record grid, waves, VGPR, LDS, scratch, spills,
   duration, and effective bandwidth/throughput.
4. Trace the expected kernel name with `rocprofv3`; a fast helper that is not
   selected is not evidence.
5. Run the complete strict or production numerical/control/task/lifecycle gate
   before promotion, then same-session p512/p1024/p4096 and context-conditioned
   tg128 wall measurements.
6. Retain an exact non-regression or a fully gated production win; otherwise
   reject it and restore the registered incumbent. Do not tune the same family
   again after two measured losses without a new profile or mechanism.
7. Emit the compact accepted/rejected/blocked artifact, update rollups/catalogs,
   write the immutable worklog entry, and commit the validated unit immediately.

### Phase P0 — binding comparator and measurement contract

Goal: turn the shape-only external refresh into one exact, repeatable target
matrix before implementation claims parity.

- [x] Promote the wall/profile driver, trace analyzers, role attributor, sync
      diagnostic, report generator, and historical p508 fixture.
- [x] Refresh/build EngramHalo HIP and Nathan Vulkan at exact identities; retain
      same-weight BF16/Q8 shape diagnostics and the MTP correctness failure.
- [x] Commit the category-balanced p512/p1024/p4096 fixture and cross-engine
      driver that feed identical token IDs, sampler, K/V type, prompt-cache
      state, and 128-transition output horizon to hipEngine and every comparator.
      Keep generated `llama-bench` rows in a separate `shape_only` class.
- [x] Add at least one prompt from every code/general-English/general-Japanese/
      mixed category to the AR wall packet so one routed prompt cannot define
      parity. Category-heldout expansion remains a closure gate.
- [x] Freeze the current three-repeat BF16 screening target per row for
      upstream llama.cpp, EngramHalo HIP, and Nathan Vulkan, with deterministic
      rows separated from non-binding diagnostics. Refresh a comparator only as
      a separate baseline event with old and new binaries measured on the same
      host. A five-pair section 6 closure refresh remains open.
- [x] Record hostname/machine ID, source and binary hashes, compiler/driver,
      profile manifest, model-part hashes, exact command, CPU governor/TuneD,
      `amd_iommu`, power/clock samples, free/available/swap, and active GPU
      processes in the canonical screening artifact. Repeat the capture for the
      eventual closure artifact.
- [x] Add explicit warm-page-cache and isolated cold-PLE modes. Never average
      or compare them as one workload. Cold mode closes/remaps and applies
      DONTNEED only to the 28.8-GB PLE tensor range before every request; it
      never uses global `drop_caches`. Matched code-p512 is **91.676 warm vs
      56.214 cold pp/s (0.613x)** with deterministic equal outputs and zero
      teardown. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p0-ple-cache-modes.json`.
- [x] Sweep hipEngine prompt chunk 256/512/1024 (and 2048 where the prompt
      permits) at p512/p1024/p4096 with memory and correctness controls; select
      by model evidence rather than copying an external `ubatch` value. All
      routes are deterministic and tear down to zero. Chunk 1024 improves
      weighted p1024/p4096 **2.55%/2.25%**, but is **-2.05%** at p512 versus
      chunk 512 and adds **720 MiB** peak; the incumbent p512 arm is also noisy
      (>2% CV). Chunk 2048 loses to 1024. Retain default 512; no promotion from
      an ordered/noisy screen. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p0-canonical-chunk-sweep.json`.
- [x] Extend the gap report to carry per-layer role time, direct/graph launch
      APIs, blocking/async copies and bytes, synchronizations, compiler resource
      data, and unresolved wall-minus-device time. The current context report
      renders marker-scoped wall/kernel/residual time, kernel rows/families,
      direct/graph/memcpy call counts and API time; raw profile artifacts retain
      copy direction/bytes, sync calls, and kernel VGPR/LDS/scratch resources.
- [x] Emit the first current canonical Amdahl ledger: clean four-category wall,
      100%-attributed exact-token hipEngine p512/p1024/p4096 prefill,
      live-513/1025/4097 decode with complete state/lifecycle checks, broad-
      active-expert telemetry, and matched pinned llama.cpp HIP family traces.
      `s` remains unknown until a local candidate runs. The old 41.6% remainder
      is retired, and P8 returns to admission-pending. Evidence:
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json`.

- [x] Declare and hold one GPU clock policy across every arm of every paired
      row, and record it in host state next to the existing power/clock samples.
      Pat1entZ3r0 measures +3-7% interactive decode from pinning
      `power_dpm_force_performance_level=high`, which is the same magnitude as
      the entire section 6.1 match band. Either pin it for the serving window or
      explicitly declare `auto` and prove both arms ran under it;
      `scripts/pn3_clock_probe.py` already samples the control. An unpinned,
      undeclared clock policy invalidates the five thermal closure pairs before
      they are collected. The campaign declares `auto`; canonical host
      metadata now records every visible
      `power_dpm_force_performance_level` value, and paired campaign commands
      verify `auto` before launch.
- [x] Audit every comparator lane for configuration it is entitled to before the
      closure freeze. `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` measures +4.0% decode on
      RADV APUs externally and appears nowhere in this tree, so both Vulkan
      lanes may be under-configured on the exact axis milestone 3 binds to.
      A/B it, and A/B `--no-repack` and `-fit off` (expected neutral at
      `-ngl 999`, but they change loader behavior). Milestone 3 requires the
      *best* same-host Vulkan engine: a target frozen against an
      under-configured lane is invalid and would have to be re-frozen.
      Chunk/ubatch needs no action; every lane already runs `-b 8192 -ub 2048`.
      **Hybrid-04 subaudit:** on exact code-p512 with one warmup and three
      measured repetitions, graphics queue changes median pp/tok/s from
      **212.08/25.59** to **209.60/25.96** (+1.43% decode, -1.19% prefill).
      Relative to queue-on/repack/fit-off, no-repack is **+0.40%/-0.03%** and
      fit-on is **+0.17%/+0.02%**, both neutral. Every repetition has the same
      generated-ID hash. This does not extrapolate to or close the historical
      upstream/Nathan Vulkan lanes; their old temporary binaries are absent and
      must be rebuilt for separate A/B. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-hybrid04-vulkan-config-screen.json`.
      **Historical lanes complete:** exact-source rebuilds at upstream
      `f1793c1c4` and Nathan `ad914eb` show graphics queue improves decode by
      **2.23%** and **2.31%** respectively while changing prefill by -0.18% and
      +0.40% when disabled. Upstream no-repack is neutral; Nathan no-repack
      loses 1.02% prefill. Fit-on loses 8.86% upstream and 1.79% Nathan prefill.
      Therefore both refreshed canonical lanes use graphics queue, repack, and
      explicit fit-off. Upstream IDs are exact across all arms; Nathan repeats
      its same four position-dependent hashes in every arm but remains
      nondeterministic and diagnostic. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-historical-vulkan-config-audit.json`.
      **Canonical refresh complete, freeze blocked:** upstream reaches
      **200.01/24.39**, **241.84/21.33**, and **266.58/18.98 pp/tok/s** and is
      exact on 12/12 cases. Nathan reaches **360.23/24.34**,
      **357.61/21.10**, and **351.85/19.01** but remains 0/12 repeatable.
      Upstream p512 prefill/decode CVs are **3.74%/2.81%** and p1024 prefill CV
      is **2.46%**, so this screen replaces the old under-configured row but
      cannot freeze the section 6 target. Stabilize and repeat upstream in the
      final counterbalanced thermal window. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-entitled-vulkan-canonical-refresh.json`.
- [x] Build the Pat1entZ3r0 `hybrid-04` patch line on the pinned UD-Q4_K_XL
      shards with BF16 K/V and run it through the canonical 12-case screen. It
      is a patch series over a pinned base, so it is the cheapest new comparator
      lane available, and its `MODEL_LOCK.json` shows the program already held
      these exact shards while publishing no row for them. Use `hybrid-04`, the
      token-parity-clean default line, not the headline stack: patches `0032`
      and `0033` make the head-sum slice tree and the non-bit-exact pooled-key
      cache opt-in. Expect nothing; the point is to convert an author-reported
      program into a matched row or to close it. **Screen complete:** all 33
      patches apply cleanly to `c589f0ed1`; the opt-in non-bit-exact head-sum
      and pooled-key routes remain off. On Vulkan graphics queue, BF16 K/V,
      `-b 8192 -ub 2048`, and `fit off`, one 12-case repetition reaches
      **270.31/25.83**, **327.03/25.67**, and **364.04/21.26 pp/tok/s** at
      p512/p1024/p4096. It matches current hipEngine generated-ID hashes on only
      **2/12** cases, so it is correctness-invalid and cannot bind a parity
      target. Zero warmups/one repetition is screening, not final thermal
      evidence. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-hybrid04-canonical-screen.json`.

### Phase P1 — layer 2 and the Q8 expert-down family

Goal: remove the largest isolated miss and replace the only host-driven grouped
expert path.

The frozen MoE map is 43 layers of Q4_K/Q4_K/Q5_1, layer 2 of
Q5_K/Q5_K/Q8_0, and layers 4/30/46/47 of Q4_K/Q4_K/Q8_0.

- [x] Commit a generated quant/shape/owner inventory test so artifact drift
      fails before timing. (82f646979)
- [x] Write actual-weight RED fixtures for layer-2 Q5_K dual gate/up and Q8_0
      down, including compact row maps, empty experts, tails, and route order.
      (runner-level regression RED, 30a2fad9e)
- [x] Route the existing selected Q5_K WMMA body for layer 2; classify its
      arithmetic before timing and preserve the strict selected chain.
      **PROFILE-REJECTED; DEFAULT OFF.** A durable p508 role trace cuts
      layer-2 MoE 371.10→88.13 ms (4.21x) and Q5_K gate/up 279.86→16.66 ms;
      p508 improves 5.34%, and all 20 category-balanced p512 pairs win. The
      complete 450-row gate nevertheless fails the binding prefill-last /
      prefill-to-c1 mean-KL scope at 0.001179 > 0.001. Overall/category,
      repeat, state, and lifecycle checks pass, but no scope can be averaged
      away. The T2 route is rejected and remains default-off. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p1-layer2-grouped-profile-rejected.json`.
- [x] Replace the Q8 path's `group_expert_start` D2H copy and Python loop over
      512 experts with a device-driven grouped Q8 owner. Use a fixed-capacity
      grid guarded by device counts or an equivalent no-host-roundtrip design.
      (2a58aa1d8). **Perf-negative** (20260830T202256); strict stays default.
- [x] Extend the proven Q8 owner to layers 4/30/46/47 when their independent
      actual-shape and composition gates pass; do not hardcode only layer 2.
      (Disposition: owner is perf-negative, so extension is not warranted on
      present evidence.) The p4096 re-audit measures strict selected Q8_0 down
      at **3.086 s**: layer 2 **0.758 s** and layers 4/30/46/47
      **0.569/0.581/0.592/0.586 s**. This confirms impact but does not overturn
      the same-mechanism negative; a materially new owner is required. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json`.
- [x] Fuse route scaling/ordered accumulation into Q8 down only if the declared
      strict/T2 contract passes. **Blocked before implementation:** Q8 down has
      the same expert-major BF16 publication versus token-major ordered-FMA
      ownership boundary as Q5. Preweighting rounds early and atomics lose
      top-k order, so no strict candidate was admitted; no T2 contract was
      declared merely to close the checkbox. Keep the primitive chain. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-q5-down-route-blocked.json`
      and
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-dense-other-subowners.json`.
- [x] Run the complete 450-row/three-repeat packet, tasks, physical c2,
      lifecycle, paired p512/p1024, and the canonical p4096 gate. Bind only
      certified scopes. The 450-row/three-repeat numerical, state, task-screen,
      and lifecycle rung ran and rejected the candidate. Physical c2 and depth
      timing were not run because later gates cannot compensate for a binding
      numerical failure. A future materially new candidate restarts this rung.

Expected evidence: layer 2 falls from about 397.95 ms toward the comparator
role range; its maximum standalone p508 contribution is about 6.6%.

**Actual P1 status (2026-08-31):** the durable recheck supersedes the initial
unretained performance reading, but the complete profile gate rejects the same
T2 route. It closes the isolated Q5_K gate/up device gap and wins about 5% at
p512, yet prefill-last mean KL is 0.001179 versus the binding 0.001 ceiling.
The route remains default-off; no c2/depth/promotion work is warranted for this
unchanged arithmetic. A future attempt requires a materially different exact
or T1 Q5_K dataflow and restarts the profile gate. Otherwise proceed to the
larger P2 early-MoE owner. The Q8_0 down strict-fallback regression from the
earlier device-owner refactor remains fixed by 30a2fad9e.

### Phase P2 — early routed MoE layers 0-26

Goal: attack the **2.366 s** fresh early-MoE owner without repeating the failed
broad WMMA suffix experiment. The current p508 split is Q4/Q5_K gate/up
**1.200 s**, Q5_1/Q8 down **1.152 s**, and activation+routing/shared tails only
**13.25 ms**. Excluding rejected layer 2, layers 3–26 still own **1.849 s**.
Active experts span 166–298 with median 9 rows/active expert across layers 0–26.
Evidence:
[`2026-08-31 P2 profile`](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p2-early-moe-profile.json).

- [x] Split the role into Q4 gate/up, activation, Q5_1/Q8 down, route-weight
      reduction, and host synchronization by layer and actual active-row count.
      A clean role/API/copy trace plus separately instrumented routing census is
      retained; telemetry wall is excluded from performance evidence.
- [~] Remove the per-layer `group_wmma_total` stream sync/D2H read. Launch a
      safe maximum tile grid with a device count guard, or prove a different
      device-only submission scheme. The pinned llama `MUL_MAT_ID` path builds
      expert bounds and forward/inverse compact maps entirely on device and has
      a top-10 specialization at `f1793c1c4`; use that ownership pattern as a
      differential design reference, not as inherited performance evidence.
      **Blocked:** early exact layers already avoid this read. For the production
      WMMA suffix, a T0 maximum-grid candidate removes runner D2H calls and keeps
      all logits exact, but four-category p512 is neutral: aggregate **1.0008x**,
      95% CI **0.9994–1.0022**, with only 8/12 pair wins. Worst-case invalid
      CTAs offset the synchronization saving. The candidate is removed; a paying
      route needs device-sized indirect dispatch or compact graph submission,
      neither currently exists in-tree. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-moe-device-tile-grid-blocked.json`.
- [x] Optimize T0 exact association first: physical-lane contraction,
      multi-row weight reuse, coalesced metadata, and output grouping while
      preserving the strict reduction/publication tree. Gate/up and down are
      co-primary owners; start with an actual-weight counterbalanced leaf screen
      on layers 3–26 rather than the <0.6% routing/activation tail. A first
      exact 64→128 expert-worker-grid mechanism is rejected and removed:
      both-family/Q4-only/Q5_1-only paired ratios are 1.0016/1.0026/0.9981,
      all confidence intervals include 1.0. More workers alone do not improve
      data reuse. Exact wider output tiling is also rejected and removed:
      Q4 output8/Q5_1 output16/both ratios are 0.9968/0.9980/0.9944; the
      both-wide 95% CI is wholly below 1.0. Serial per-CTA work outweighs the
      smaller grid. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p2-expertgrid128-rejected.json`
      and
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p2-wider-output-tiles-rejected.json`.
      A two-team exact Q4 CTA that computes output columns concurrently is also
      rejected/removed at 0.9978x (95% CI 0.9927–1.0028). With worker count,
      serial width, and concurrent-team schedules exhausted, do not continue
      this scheduling family without a new reuse mechanism.
- [x] Add operation-complete grouped dual gate/up+SiLU and
      down+route-weight+scatter/ordered-reduce candidates. Keep primitive
      chains registered. The first T0 grouped-Q4+SiLU epilogue passed a small
      byte-exact RED but faulted the GPU before the first bound p512 warmup was
      recorded. Removing it restored named-production p512 to **5.517 s / 92.802
      tok/s** with exact lifecycle closure. It is rejected/removed; any retry
      requires an actual K2560/production-FFN owner oracle before whole-model
      execution. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-q4-grouped-silu-rejected.json`.
      **Exact down+route blocker:** grouped Q5 publishes expert-major BF16 rows,
      while the binding reducer gathers token-major lanes and evaluates ordered
      `fmaf(value, weight, accumulator)`. Preweighting rows rounds the product
      before addition; expert-major atomics cannot preserve top-k order; keeping
      BF16 publication leaves the reducer intact. Do not add a nominal epilogue.
      A retry needs cooperative token-major ownership or explicit T1/T2 gates.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-q5-down-route-blocked.json`.
- [x] Sweep wave32 ownership, workgroup size, row/output tiles, and LDS padding
      on rotating actual weights. Use Nathan's wave32/bank-conflict findings as
      hypotheses, never as transferable constants. The full-model p508 ladder
      already traverses actual resident early-layer tensors: expert-grid
      64→128 is neutral (**1.0026x**, 95% CI **0.9981–1.0070**), output4→8 is
      neutral/negative (**0.9968x**, **0.9922–1.0013**), and two concurrent
      128-thread teams are neutral (**0.9978x**, **0.9927–1.0028**). The retained
      128-thread kernel uses contiguous unique LDS metadata stores, wave-uniform
      metadata broadcasts, and one reduction slot per wave; there is no
      strided multi-lane LDS access for padding to repair. This closes the T0
      schedule family. A retry requires new concurrent data reuse or explicit
      T1/T2 gates. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-exact-geometry-closure.json`.
- [x] Test T1/T2 WMMA only in independently calibrated layer clusters. Every
      candidate must pass category, shape, transition, repeat, task, c2, BF16-
      relative, lifecycle, and manifest gates; no final-prompt or one-layer
      screen can promote it. The retained Q4/Q5 suffix 27–47 and Q8 suffix
      32–47 satisfy independent boundary screens, 450-row numerical/category/
      transition/repeat/state gates, 18-prompt task gates, physical c2,
      lifecycle, manifest, expected-kernel trace, and same-session p508/p1012
      whole-model gates. Their p508 speedups are **1.132x** and **1.234x**.
      Layer 2 demonstrates the fail-closed boundary: its **0.001179** mean KL
      exceeds **0.001**, so c2/depth work was skipped and the route remains
      default-off. BF16-relative comparison was inapplicable because no
      qualified full-BF16 Qwen4Exp runtime existed; same-quant strict remained
      binding. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-cluster-calibration-closure.json`.
- [x] Re-profile after each retained cluster and stop widening when the next
      boundary fails. Layers that fail remain on strict owners. The retained
      chronology is MoE suffix **32→28→27**, then Q8 suffix **32**, with p508
      speedups **1.096x→1.122x→1.132x→1.234x** and a complete superseding packet
      at every step. Widening stopped at failed MoE suffix 24, every individual
      layer 0–26, Q8 suffix 27, and grouped layer 2; all remain strict. The
      later 100%-attributed post-P4 p512 ledger refreshes the queue at Q4
      **1.265 s**, Q5 down **1.087 s**, dense-other **1.071 s**, Q8 **0.884 s**,
      GDN **0.599 s**, and GR **0.404 s**. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p2-reprofile-stop-boundary-closure.json`.

**Current P2 exact-scheduling outcome (2026-08-31):** three distinct T0
schedules are exhausted without a retainable whole-model win: more expert
workers, wider serial output ownership, and concurrent 128-thread Q4 teams. The
remaining **1.849 s** layers 3–26 projection gap is a kernel data-reuse/quality
problem, not a routing-tail or launch-grid problem. A future P2 attempt requires
a new reuse mechanism (or independently justified T1 arithmetic), plus fresh
actual-weight and complete-profile evidence. Continue P3 meanwhile; do not keep
mutating the same schedule family.

### Phase P3 — shared expert, router, dense projections, and GR prefill

Goal: account for the large non-routed prefill remainder that the earlier plan
left without an implementation phase. The fresh p508 split names **1.670 s**
of primary P3 roles: GR projection/read **709.32 ms**, GDN
`attn_qkv+attn_gate` **532.36 ms**, router **181.91 ms**, `ssm_out`
**137.84 ms**, and shared gate/up/down **121.61 ms**. Evidence:
[`2026-08-31 P3 profile`](../benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-prefill-profile.json).

- [x] Re-profile and separately name router, shared gate/up/down, attention and
      FFN GR reads/writes, `attn_qkv`, `attn_gate`, `ssm_out`, QSA projections,
      casts, and elementwise tails. Exact per-layer roles and kernel symbols are
      retained from the fresh current-production trace.
- [x] Evaluate an exact F32 router+stable-top-10 owner so 512 router logits are
      not written and reread when the public path only needs deterministic
      routing. Keep the full-logit primitive for diagnostics. The retained
      multirow producer now reuses each F32 weight row across four prompt rows,
      preserves the dense FMA/reduction order, and improves clean p508
      89.689→91.121 tok/s (1.0160x); c1 stays dense. The complete 450-row and
      18-prompt state/task gate is exact. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-router-f32-tile4.json`.
      A counter-last projection+stable-top-10 fusion preserves logits/IDs/weights
      exactly but regresses the rows508 operation-complete primitive
      1.877→2.128 ms (0.882x), even with four selector CTAs per tile. It is
      removed; do not retry without eliminating global coordination or the
      materialized logits. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-router-fused-select-rejected.json`.
- [x] Fuse shared gate/up+SiLU, then shared down+sigmoid gate+combine, preserving
      F32/BF16 boundaries and the strict shared-expert chain. Reusing one F32
      D4x3 activation pack across the two production-MMQ projections is exact
      and improves their GPU window 1.134→1.077 ms, but is rejected/removed:
      combined p508 is 0.9988x (95% CI 0.9971–1.0005) and code-p1024 is 0.9997x
      (95% CI 0.9980–1.0014). Pack reuse alone is below complete-wall
      resolution; require a larger gate/up+SiLU or down+gate+combine boundary.
      Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-shared-q8-pair-rejected.json`.
      An exact shared-down+BF16-boundary+sigmoid-combine composite is likewise
      rejected/removed: its rows508 GPU window improves 1.376→1.292 ms, but
      p508 is flat at 0.99974x (95% CI 0.99772–1.00176). Small shared-expert
      epilogue launch contractions are now exhausted; move to a larger data-
      reuse boundary. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-shared-down-combine-rejected.json`.
- [x] Fuse GR grouped RMSNorm + unequal down/inject where ownership permits;
      add down+scaled-SiLU and up+sigmoid+gated-mean epilogues. The exact
      sigmoid+gated-mean subunit is now retained for rows <=256: it removes one
      launch per GR read, improves clean counterbalanced p508+128-step decode
      14.162→15.111 tok/s, and passes 450/450 logits, 18/18 state/task prompts,
      and lifecycle exactly. The first all-row tail-only fusion lost at rows508;
      a replacement rows>256 Q8-up composite keeps each output's coltile8
      reduction but groups two hidden columns across four branches and emits
      sigmoid gates plus branch mean. Clean p508 improves 91.158→91.600 tok/s and
      code-p1024 88.754→89.239 tok/s; the complete 450-row/state/task gate is
      exact. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-sigmoid-mean.json`
      and
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-up-sigmoid-mean.json`.
      Exact Q8 down+scaled-SiLU publication is rejected/removed before
      whole-model timing: rows2 improves 1.118x, but binding rows508 regresses
      1.655→1.667 ms (0.9926x). Keep the separate scaled-SiLU fallback and do
      not add transcendental work to the register-heavy down publication.
      Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-gr-down-scaled-silu-rejected.json`.
- [x] Evaluate output-projection+GR-write composites for attention and MoE
      boundaries, including the exact inject ordering. Start with the 36-layer
      Q8 `attn_qkv+attn_gate` boundary: preserve the current MMQ qkv and exact
      coltile gate arithmetic while sharing input/activation quantization, or
      declare and fully gate a T1 pair. Registered singleton routes remain
      fallbacks. The real rows508/K2560/N6144 exact gate geometry is exhausted:
      incumbent c8r4 is **8.04 ms**, versus c4r8/c16r2/c8r8/c16r4/c32r1 at
      **10.59/9.69/12.46/18.06/37.92 ms**, all bit-exact. A qkv+gate win now
      requires true original-F32/MMQ ownership, not another tile constant.
      Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-attn-gate-exact-geometry-exhausted.json`.
      A final five-pair/category auto-clock screen rejects the current Q8 MMQ
      attention-gate route: English beats at **1.0309x** (95% CI
      **1.0113–1.0504**), Japanese/mixed match, but code remains noisy with
      bound/candidate CV **2.19%/2.30%** and interval **0.9862–1.0271**. All
      logits and repeats are exact and teardown is zero, but section 6 forbids
      averaging categories; keep the route default-off. The secondary GR
      down+inject boundary has a static heterogeneous-arithmetic blocker: both
      attention/FFN down
      weights are Q8_0 **[320,10240]**, while inject is F32 **[4,10240]**.
      Exact inject must preserve its original-F32 K/FMA/reduction tree; reusing
      Q8 activation is T1/T2, while a T0 single launch still performs both
      traversals and removes only the tiny four-output boundary. Do not add a
      nominal pair kernel without a new mixed-Q8/F32 RED and operation-complete
      win. This closes the evaluated composite ladder with strict chains intact.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p3-gr-composite-closure.json`.
- [~] Extend dense Q8 MMQ/WMMA scopes earlier only through the complete
      production packet. Optimize exact coltile/rowbatch fallbacks for layers
      that reject changed arithmetic. The first default-off extension adds the
      omitted K2560/N6144 attention-gate shape: same-process p508 improves 3.52%
      and all numerical scopes pass, but candidate state repeat 1 differs from
      repeats 2–3 on the first code prompt. Same-schedule state repeatability is
      binding; the scope is rejected/default-off. Ignoring the first run as
      warmup is diagnostic only, not a promotion rule. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-rejected.json`.
      **Current recheck blocked on performance, not correctness.** Later state
      fixes clear the first-repeat failure: the complete 450-row numerical,
      state-repeat, lifecycle, and semantic task review now passes. Current
      four-category p512 is only **1.0104x** aggregate (95% CI
      **1.0002–1.0206**, 9/12 wins), every per-category interval includes 1.0,
      and alternating first-pair drift makes three categories noisy. This is
      neither a clean win nor a loss under section 6; the route stays
      default-off. The subsequent five-pair/category screen confirms rejection:
      code remains above 2% CV in both arms and its interval includes 1.0,
      while the other categories cannot compensate. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-fivepair-rejected.json`.
- [x] Require each retained subunit to reduce its complete role and p512/p1024,
      not merely an isolated GEMM; re-run p4096 at the phase gate. A fresh
      stacked profile after the retained router/GR paths re-ranks P3 to GR
      projection/read **651.16 ms**, `attn_qkv+attn_gate` **536.26 ms**,
      `ssm_out` **137.62 ms**, shared expert **122.04 ms**, and router producer
      **94.77 ms**. The 36-layer qkv+gate boundary remains the largest uniform
      next target; GR down+inject is secondary. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p3-stacked-profile.json`.

### Phase P4 — GDN and QSA prefill parity

Goal: close the remaining **634.94 vs 92.34 ms** GDN and
**110.49 vs 13.91 ms** QSA gaps after P1-P3 are stacked.

- [x] Re-profile prepare, recurrence, norm/gate tail, projection, KV/index
      append, selection, attention, and output roles by admitted layer scope.
      After QSA fixed256, dense attention falls **82.33→29.18 ms** and total
      QSA role **103.32→50.61 ms**. Remaining GDN is 21 early strict recurrence
      calls **522.58 ms**, admitted columnwarps **63.11 ms** plus **5.16 ms**
      tail, and Conv **7.28 ms**. Early strict state layout is the P4 blocker;
      P5 may proceed independently. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p4-stacked-profile.json`.
- [x] For GDN, test exact early-layer column ownership, prepare+recurrence
      fusion, state residency, and bounded chunking. Engram's chunked kernel is
      a design reference only; it was not active in the fork's published rows.
      A strict-order prepared-QKV/scalar route is exact in output and recurrent
      state but rejected/removed at rows508: **12.716→12.757 ms (0.9967x)**.
      Prepared tensor traffic offsets removed norm/transcendental work; require
      direct column ownership or state-layout reuse instead. A direct four-block
      column candidate improves rows508 **12.628→12.227 ms (1.0328x)** and keeps
      recurrent state exact, but production output parity fails (about 25% of
      each 32-column quarter); the one-block reduced fixture masked it. It is
      rejected/removed before model timing. Future layout work requires a
      production-width output oracle from RED. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p4-gdn-prepared-strict-rejected.json`
      and
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p4-gdn-columnblocks4-rejected.json`.
      The retained register-sharded column-warp route is confined to admitted
      layers 27–47 and passes its full profile packet; early exact prepared
      traffic is neutral, production-width four-block columns fail output
      parity, and operation-complete transposed-state integration regresses
      **0.07535→0.08535 ms (0.883x)**. Early layers therefore remain strict;
      further widening is the separate T1/T2 item. Consolidated evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-gdn-exact-campaign-closure.json`.
- [x] Evaluate T1/T2 GDN suffix widening only with fresh all-category boundary
      packets; keep every rejected early layer strict. The calibrated ladder
      rejects all-layer widening at mean KL **0.0068** and admits suffix 27–47
      at **0.00099**. Its complete 450-row/three-repeat packet passes every
      category/shape/transition, deterministic state, 18-prompt task, physical
      c2, lifecycle, manifest, selected-kernel, and same-session p508/p1012
      gate; whole-model deltas are **-17.09%/-15.67%**. Layers 0–26 and shape
      misses retain `qwen4exp_sigmoid_strict_prefill`. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-gdn-suffix-boundary-closure.json`.
- [x] For QSA, compare current key-parallel head-dim-256 flash geometry with the
      selected llama kernel family, including key tiles, online-softmax merge,
      grid sufficiency, and register/LDS pressure. The non-flash multirow dense
      owner now selects the registered fixed256/precomputed-offset/vector2
      kernel: real primitive **6.846→2.485 ms (2.755x)**, clean p508
      **91.529→92.442 tok/s**, code-p1024 **89.150→90.634 tok/s**, and the
      complete 450-row/state/task gate is exact. Generic batch attention remains
      fallback. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p4-qsa-dense-fixed256.json`.
      The p4096 differential audit now isolates variable-selection sparse rows
      at **9.418 s** of the **10.229-s** attention role; dense attention is
      **0.638 s**, index score **0.129 s**, and top-k expand **0.023 s**.
      The next mechanism is therefore a bounded multirow ordered-attention path,
      not score/top-k tuning or a direct copy of the c1 scratch layout. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-qsa-prefill-subowner.json`.
      **Rejected follow-up:** the first bounded three-pass implementation was
      byte-exact at H256 and 2,049–2,051 selected tokens, but its QK grid exposed
      about **25.2 million CTAs per 512-row chunk**. The p4096 whole-model screen
      remained in its first warmup at 99% GPU use after more than 168 seconds,
      so it was terminated before collecting an invalid sample and removed.
      Any retry must tile rows or persist selected-token work rather than launch
      one CTA per `(row, head, selected token)`. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-qsa-multirow-grid-rejected.json`.
      **Exact-mechanism blocker:** the incumbent already exposes 12,288
      row/head CTAs per 512-row chunk. Each exact H256 row needs all 256 threads
      to preserve the QK tree and executes 11 block barriers per selected token.
      A four-row tile reaches the 1,024-thread block limit, preserves total wave
      work, couples variable row counts at block barriers, and cannot remove the
      binding recurrence. Wave reductions or partial-summary merging cross into
      T1/T2. Exact sparse-prefill QSA is therefore exhausted at this geometry;
      re-rank to an independent owner unless production-numerics widening is
      explicitly admitted. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-qsa-exact-rowtile-blocked.json`.
- [x] Confirm selected-position attention already removes dense-mask work.
      The current 100%-attributed p4096 trace contains explicit selected-position
      sparse-row attention and a separate dense-row owner; it contains no
      dense-mask or fully-masked-slice kernel. The conditional skip mechanism
      is therefore inapplicable and is not ported. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-qsa-prefill-subowner.json`.
- [x] Publish a fresh p512 device-role ledger after P4; no prefill phase may be
      called closed with an unexplained multi-x owner. The new named-production
      code-p512 trace is **100% role-attributed**: **5,483.08 ms** kernel sum in
      a **5,552.89-ms** window. Owners are Q4 gate/up **1,265.08 ms**, Q5 down
      **1,086.52 ms**, dense-other **1,071.12 ms**, dense Q8 **883.63 ms**, GDN
      **599.36 ms**, and GR **403.94 ms**; no unexplained multi-x bucket remains.
      Output/lifecycle pass, with zero steady growth and zero allocations after
      close. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p4-p512-ledger.json`.

### Phase P5 — device-owned AR output boundary

Goal: remove full-vocabulary host readback and synchronization before enlarging
graph scope.

- [x] Add a registered device argmax/greedy sampler after the lm head and copy
      only the token ID plus explicitly requested compact telemetry to the host.
      The exact two-stage `top1_i64` route is now the strict and production
      normal-greedy default.
- [x] Keep full logits/probabilities as an explicit API/debug path; do not
      silently change public response semantics. Direct runner, MTP, numerical,
      and debug calls retain full logits by default.
- [x] Feed the device-owned token into the next embedding path where possible.
      PLE hashing consumes the compact host token, while the next embedding
      lookup reuses the resident device token without an H2D round trip.
- [x] Reduce the current 28 blocking and 12 async memcpy calls/token to a
      role-explained minimum and record bytes/directions, synchronization count,
      first-token, steady-state, and exact-ID/logit controls. The final ledger is
      **26 blocking + 12 async**: no explicit sync, 10,440 blocking bytes/token,
      and no unused normal-AR hidden D2D. The remaining 24 scalar H2D calls are
      QSA position/context ownership and require a separate shared-state design.
- [x] Re-run natural multi-prompt decode, not only the repeated `9707` steady
      diagnostic. The complete 18-prompt/category-heldout T0 packet has 450/450
      logits at KL=0, 18/18 exact generated ID sequences/tasks, repeat-exact
      compact state, lifecycle closure, and a focused strict-repeat confirmation
      for the one initial strict-state warmup anomaly. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p5-device-argmax.json`.

### Phase P6 — decode GR, MoE, and dense operation completion

Goal: reduce the **48.63 vs 38.90 ms/token** device gap and the remaining
direct-launch surface before graph capture hides it.

- [x] Stack GR down+inject, down+scaled-SiLU, up+sigmoid+gated-mean, and GR-write
      composites one at a time with exact/T1/T2 declarations. Exact up+sigmoid+
      gated-mean row-scoped owners are retained; down+scaled-SiLU is rejected at
      **0.9926x** on binding rows508; qkv+gate is rejected by the five-pair
      category gate; and Q8-down/F32-inject is blocked from nominal T0 fusion by
      incompatible arithmetic trees. Primitive projection/epilogue/GR-write
      chains remain strict fallbacks. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p3-gr-composite-closure.json`.
- [x] Re-rank dense Q8, selected Q4 gate/up, selected Q5_1/Q8 down, shared
      expert, router, QSA, and lm-head kernels after P5. At live 2,052, profiled
      per-step owners are sparse QSA attention **35.88 ms**, dense Q8
      **25.84 ms**, Q4 gate/up **9.30 ms**, Q5 down **8.45 ms**, and remaining
      dense **4.18 ms**. QSA is the context-conditioned first target.
- [x] Profile decode immediately below/above the QSA transition (live counts
      2,051/2,052) and at p4096. Clean identical-transition medians are
      **66.61/95.88/96.02 ms**. The 2,051→2,052 kernel delta is **30.77 ms**:
      sparse attention adds **27.47 ms**, score/top-k adds **0.92 ms**, and the
      rest is launch/secondary-owner variance. The flat 2,052→4,097 result shows
      a fixed selected-budget activation cost, not O(context) growth. The first
      T1 wave8 H256 candidate improves the 2K-selected primitive **4.534x** and
      removes the cliff (**95.88→67.84 ms** at 2,052), but is correctness-
      rejected and removed: p4096 teacher top-1 is **98/100**, three category
      scopes fail, and only **2/4** free-generation tasks match strict. Future
      attention dataflow must preserve global selected-token softmax order or
      pass this same gate. The next exact design splits the current serialized
      body into three passes: parallel selected-token QK scores using the
      incumbent reduction tree; deterministic selected-order online-softmax
      coefficients; then one weighted-V recurrence per output column using those
      coefficients in the same selected order. This removes per-token CTA
      barriers without another partial-softmax merge. llama's selected-mask to
      backend MHA path is topology evidence only. A second contiguous-chunk
      merge lowers mean/p95/max KL to **0.000565/0.002551/0.004372** but worsens
      top-1 to **97/100** and
      free tasks to **1/4**; it is also removed. Partial-softmax merge schedules
      are exhausted. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p6-context-transition-profile.json`
      and
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p6-qsa-wave8-h256-rejected.json`
      and
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p6-qsa-contiguous-h256-rejected.json`.
      The exact successor is retained. It computes incumbent-order QK scores in
      parallel, runs one global selected-order online-softmax coefficient
      recurrence per query head, and applies those coefficients in independent
      output-column weighted-V recurrences. The actual 2,051-selected-token leaf
      improves **2.158→1.180 ms (1.829x)** and is bit-exact to strict.
      Four-category p4096 tg128 improves **93.912→80.061 ms/token (1.173x)**;
      all 12 counterbalanced pairs win, the aggregate 95% ratio interval is
      **1.170–1.176**, full logits/IDs are exact, and teardown is zero. A named
      `rocprofv3` trace records all three expected kernels and reduces the QSA
      role to **20.913 ms/token** with 100% attribution and no measured-window
      allocations. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json`.
      A follow-up exact four-column-per-thread weighted-V schedule is rejected
      and removed: after fixing a RED-caught reciprocal-multiply
      reassociation, it remains bit-exact but measures **1.469 ms** versus the
      retained **1.180-ms** leaf in separate cached-build runs. The magnitude
      is diagnostic because the runs used separate processes, but the candidate
      clearly fails leaf admission and does not warrant whole-model timing.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-value-col4-rejected.json`.
- [x] Tune Q4/Q5/Q8 c1 owners on rotating actual weights for coalescing,
      physical-lane contraction, occupancy, and operation-complete epilogues;
      do not force WMMA onto M=1. Actual-model retained packets select calibrated
      Q4 DP4A+SiLU on 43 layers (**1.120x** complete decode; five layers stay
      exact), exact physical64 Q5_1 weighted-down (**1.011x**), and exact Q8
      F32/F32 output-pack8 (**1.107x**). Each packet proves the expected kernel,
      full correctness controls, complete-model wall, and registered exact
      fallback. All are decode-shaped GEMV/DP4A/output-column owners; no WMMA is
      forced onto M=1. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p6-c1-quant-owner-closure.json`.
- [x] Preserve the exact fused Q5 weighted-down and Q4 dual+SiLU fallbacks;
      replace them only with same-role evidence. Current c1 dispatch still
      resolves Q5_1 `selected_weighted_sum_logical256_t64_bf16_bf16_out` with
      the exact selected-projection+ordered-weighted-sum chain on misses, while
      the production Q4 DP4A scope names
      `selected_dual_silu_logical128_t64_gemv_bf16_bf16_out` as strict
      fallback. Focused registry/profile and fused-vs-unfused bit tests pass
      **9/9**. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p6-c1-fallback-preservation.json`.
- [x] After every retained fusion, update direct launches, graph launches,
      kernel rows, API time, copy bytes, and context-conditioned tg128 wall.
      The retained ordered-QSA unit refreshes this census; the requirement
      remains binding for every later retained unit.

### Phase P7 — normalized/transposed GDN decode

Goal: replace the measured **2.659 vs 0.465 ms/token** recurrence with a
c1-shaped layout without reviving the rejected prefill-colwarps route.

- [x] Normalize Q/K once per head instead of once per output column. An exact
      t128 sibling that removes the t256 zero-only reduction half is rejected
      and removed: **0.06633→0.06754 ms (0.982x)** at production geometry.
      Thread count is not the state-traffic solution. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p7-gdn-t128-rejected.json`.
- [x] Keep recurrent state transposed in the decode-native layout across steps;
      define exact construction, snapshot, rollback, reset, and strict
      conversion boundaries. Exact strict↔transposed conversion and a
      default-unselected wave-per-value primitive now pass actual-shape
      output/state envelopes, but operation-complete prepare+recurrence+gate
      regresses **0.07535→0.08535 ms (0.883x)**. Runner layout integration is
      removed; a paying design must fuse stages or reduce 6,144 wave-block
      overhead. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p7-gdn-transposed-integration-rejected.json`.
- [ ] Port the relevant llama four-warp decode dataflow, not its prefill body or
      constants. `ggml-cuda/gated_delta_net.cu` assigns four output columns to
      one CTA, keeps each transposed state shard in registers, and writes state
      once after the token loop. The rejected hipEngine primitive instead used
      one wave per value (**6,144 blocks**) plus separate prepare/gate stages.
      A new candidate must combine four or more columns per CTA and fuse enough
      prepare/recurrence/norm-gate work to win the complete boundary.
- [ ] Prove CPU-reference state/output parity on reduced and actual fixtures,
      repeated steps, restore/replay, cancellation, and c2 isolation.
- [ ] Require expected kernel trace, complete profile packet, and canonical
      tg128 win. Never re-enable the invalid `GDN_COLWARPS_DECODE_LAYERS` route.

### Phase P8 — state-safe whole-transition graph

Goal: contract 48 small MoE graphs plus 1,195 direct launches/token toward one
request-owned transition submission.

- [x] Reproduce and localize the historical third-replay state corruption
      before changing capture scope. A faithful recurrent-subgraph probe now
      captures 36 independent production-shape Conv+GDN state pairs (72 kernels)
      and remains bit-exact through four replays, reset, capture non-execution,
      and teardown. The old fault is therefore **not** an isolated Conv/GDN
      relaunch hazard on current ROCm; localization moves to cross-kernel/full-
      layer composition. Do not widen production capture yet. Evidence:
      `benchmarks/results/2026-08-31-gfx1151-qwen38-flash-next-p8-gdn-graph-replay-probe.json`.
- [x] Capture in rungs: one stateful layer, one complete attention/FFN
      transition, a multi-layer segment, then the full token step. **Rung 1 is
      complete:** strict layer 0's full 34-kernel GR→Conv/GDN→MoE transition is
      output/state exact through four replays and measures **4.051→1.258 ms
      (3.22x)** over 30 synchronized samples; cached profiling confirms 34
      dispatches per `hipGraphLaunch` and no post-launch device allocation.
      A second rung chains complete GDN layers 0–2: all state/output remains
      exact through four replays and **9.801→3.896 ms (2.52x)** over 30 samples;
      profiling confirms 105 dispatches per graph launch and zero post-launch
      allocation. These are GDN-only research rungs, not a production binding;
      A fixed-position mixed GDN/GDN/GDN/QSA segment also passes all captured
      device owners and output through four replays and measures
      **12.160→4.955 ms (2.45x)**, with 142 dispatches/launch and no post-launch
      allocation. It is diagnostic only: scalar position/live-counts and the
      host QSA index cursor do not advance (`host_cursor_replay_safe=false`).
      Its device-owned successor now passes positions 8–11 exactly across
      position/context, K/V, raw index, GDN state, and output, and measures
      **13.882→4.974 ms (2.79x)** over 30 advancing samples. Each graph launch
      contains one device-position append plus one paired advance among 143
      dispatches and allocates nothing post-launch. This admits the dynamic
      four-layer research rung. The next eight-layer rung includes layer-1 PLE,
      six GDN states, and two QSA owners; it remains exact at positions 8–11 and
      measures **26.739→10.112 ms (2.64x)**, with 294 dispatches, two dynamic
      appends, two advances, and no post-launch allocation per graph replay.
      The all-physical-layer rung also passes: layer-1 PLE plus all 48 layers,
      all 12 QSA owners, and 136 hashed device-state owners are exact at
      positions 8–11; wall is **154.346→57.900 ms (2.67x)**. Profiling confirms
      1,697 dispatches, 12 dynamic appends, 12 advances, and no post-launch
      allocation per replay. The historical third-replay corruption therefore
      does not reproduce on the current full physical stack. This still is not
      production: token/PLE row publication, final mix/head/argmax, generated-
      token feedback, bucket transitions, fallback, c2, and lifecycle remain.
      Evidence:
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-stateful-layer-graph.json`,
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-gdn-segment3-graph.json`,
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-mixed-segment4-graph.json`,
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-mixed-segment4-graph.json`,
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-advancing-segment8-graph.json`,
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-all48-graph.json`.
      The root/head boundary now feeds generated device argmax tokens through
      embedding, active PLE, all 48 layers, full logits, and argmax. Exact PLE
      lookup cannot live inside the graph: its 320,001,446-row IQ4_NL table is a
      28.8-GB sparse mmap. The honest transition is host hash + 16-row mmap
      gather/dequant + 10-KiB H2D, then one graph launch and token readback.
      Across positions 8–11 the trajectory `3147→278→18407→2129→69422`, full
      logits, and 138 owners are exact; reset→replay and graph→forced-eager→
      graph resumption are also exact. Operation-complete wall including PLE
      and readback is **194.758→61.910 ms** over 30 samples, but that eager arm
      is probe-local. The later strict runner row of 68.855 ms came from another
      process, so the derived **1.112x is not a named-production A/B** (see the
      denominator audit in section 2.2). Profiling
      confirms 10 launches with 1,708 dispatches each and no post-launch device
      allocation.
      Production binding stays off pending request graph keys/lifecycle,
      multi-prompt generation, context buckets, fallback, cold PLE, and c2.
      Evidence:
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-full-transition-graph.json`.
- [x] Audit the probe denominator. The strict runner measures **68.855 ms**
      with shipped MoE graphs and **143.989 ms** without them, proving that the
      194.758-ms probe eager arm is invalid. However, 61.910 ms is imported from
      the separate graph-probe process, and both rows are strict rather than a
      named-production same-session A/B. Therefore **1.112x/6.945 ms is also
      unrankable**; production `O` and `s` remain unknown. Device argmax versus
      host full-logit D2H is not measurable. Evidence:
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-p8-production-denominator.json`
      and the canonical impact profile.
- [x] Add a named-production graph arm to the same-residency harness and run a
      counterbalanced p512/p1024/p4096 tg128 feasibility A/B. **Evaluated and
      rejected; default remains eager.** After discarding one complete
      production and graph trajectory per pair to remove PLE first-touch bias,
      the production-manifest arm wins all five exact pairs at p512
      (**57.312→55.907 ms, `O=1.404 ms`, `s=1.0251`, 8.6% of the measured
      comparator gap**) and p1024 (**58.962→57.868 ms, `O=1.094 ms`,
      `s=1.0189`, 9.1%**). The p4096 arm cannot execute: full-transition capture
      rejects crossing QSA's dense-equivalent limit before timing, exactly where
      sparse selection becomes active. Thus no p4096 `W/C/O/s` exists, no graph
      result is imported from the strict probe, and no production cache is
      admitted. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p8-production-feasibility-rejected.json`.
- [~] Keep token/PLE input buffers and every weight/state/scratch pointer stable;
      include profile-manifest hash, shape, context bucket, and fallback in the
      graph key. The strict research probe reuses one graph exec over stable
      resident weight/state/scratch allocations, updates token/PLE staging in
      place, records manifest
      `e93c8fa47bf2804a781bade0939c617563531ce5436e3281c2ab869209875dca`,
      and performs zero device allocations after first launch. It does not have
      a request-owned production graph cache or composite key, so this remains
      partial pending a context-bucket-safe sparse-QSA mechanism; the named-
      production A/B above rejected admission at p4096. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p8-pointer-key-audit.json`.
- [ ] Gate GDN, QSA K/V/index append, PLE history, sampler output, snapshot,
      rollback, reset, cancellation, c2 isolation, teardown, and at least three
      consecutive replays at every rung.
- [ ] Compare direct/graph API time, graph build/reuse counts, kernel rows/span,
      compact copies, first-token latency, and context-conditioned tg128. A graph
      that merely hides a slower kernel chain is not a retained win.
- [x] Target no per-layer graph launches and no unexplained direct launch in the
      steady transition; document any irreducible boundary. The research full
      transition uses exactly one graph launch/token containing **1,708** traced
      dispatches: embedding, active PLE consumer, all 48 layers, 12 QSA appends/
      advances, lm head, and argmax, with zero post-launch allocation. The
      explained external boundary is host PLE hash + 16-row IQ4_NL mmap gather/
      dequant + 10-KiB H2D, then compact token readback; the exact 28.8-GB sparse
      table is not graph-resident. This closes only the research prerequisite;
      named-production A/B/key/lifecycle/c2/cancellation remain open. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p8-single-launch-boundary-closure.json`.
- [x] Do not spend a rung on submission batching alone. The nearest external
      evidence (`GGML_VK_MAX_NODES_PER_SUBMIT` 200-800, -1.3 to -2.4%) concludes
      cost is per-dispatch, not per-submit. A P8 win must remove dispatches; a
      variant that only bundles more nodes into one submission is pre-rejected
      unless a local trace contradicts that prior.

### Phase P9 — PLE mmap, cold-cache, and load-memory lane

Goal: implement the reproduced Nathan/Engram mechanism without mixing it into
warm GPU-kernel claims.

- [x] Instrument `Qwen4ExpPLEMMapTable` and staging with requested/unique rows,
      unique/adjacent pages, bytes, prefetch ranges, faults or resident-page
      proxy, dequant/copy/H2D wall, and cache mode. Telemetry is opt-in and
      excluded from performance evidence. A real warm code-p512 request records
      10,240 requested / 4,216 unique rows, 4,297 unique pages with 86 adjacent
      pairs, 921,600 source bytes, 1,310,720 H2D bytes, process-fault proxies,
      cache advice/range, and **5.670 s gather+dequant / 0.825 ms staging copy /
      19.207 ms H2D** across prefill plus 128 transitions. Output hash remains
      canonical and teardown is zero. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-ple-telemetry.json`.
- [x] Add off/auto/on random-access advice only together with page-aligned,
      deduplicated and adjacent-range-merged `WILLNEED` prefetch. The default is
      `off`; `auto` selects random advice only when useful row bytes are below
      half the page-aligned fetch bytes. A real warm code-p512 route smoke selects
      `on`, emits 16 deduplicated 4-KiB ranges for the final 16-row gather, keeps
      the canonical output hash, and tears down to zero. This closes the mechanism,
      not a speed claim: the single telemetry-enabled request is excluded, and a
      future cold/warm paired gate must choose any production policy. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-random-prefetch.json`.
- [x] Dequantize directly into the active pinned ring where practical; remove
      temporary gather/value arrays and redundant copies. **Evaluated and
      rejected.** The exact candidate removes the multi-row gather/value arrays
      and final copy by dequantizing 16 IQ4_NL rows individually into the active
      ring, but loses every one of 20 four-category p512 prefill pairs:
      **111.698→113.594 s aggregate, 0.9833x**. The first exact tg128 pair also
      loses **7.348→7.452 s (0.9861x)**, so the remaining decode pairs were
      stopped as incapable of reversing rejection. The temporary flag and route
      are removed; vectorized multi-row dequantization plus its measured
      ~0.825-ms/request copy remains production. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-direct-staging-rejected.json`.
- [x] For prefill, overlap next-chunk prefetch/dequant with current GPU work
      using the existing two-buffer ownership plus explicit event/thread
      lifetime. Decode remains demand-driven unless a real lookahead exists.
      **Evaluated and rejected.** Because the complete prompt is known, exact PLE
      hash rows can be precomputed safely; a one-worker candidate stages chunk
      N+1 into the inactive pinned buffer while chunk N submits GPU work and
      joins before reuse/H2D. Exact screens are neutral: p1024
      **11.223→11.240 s (0.9985x)** and p4096 **54.487→54.396 s (1.0017x)**.
      The latter is below the 1% complete-wall floor and cannot justify new
      thread/future lifecycle. Candidate and flag removed; decode remains
      demand-driven. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-prefill-overlap-rejected.json`.
- [x] Add a safe isolated cold-cache protocol and a warm steady protocol. Do
      not use one process's warming repetitions as independent samples. The
      retained driver applies one initial file-scoped `WILLNEED` for warm steady;
      cold closes/remaps the PLE mapping and reapplies mapping+file `DONTNEED`
      before every warmup and measured request, only over the 28.8-GB PLE tensor
      range. It never uses global `drop_caches` or treats warming repetitions as
      independent samples. Existing code-p512 evidence is deterministic and
      output-equal at **91.676 warm versus 56.214 cold pp/s**, with zero teardown.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-cache-protocol-closure.json`.
- [x] Measure lazy on/off, cold/warm p512/p1024/p4096 and tg128, page reads,
      RSS/file versus anonymous memory, available/free/swap, and exact output
      hashes. The 18-row off/auto/on screening matrix records every requested
      field and is exact by shape. Across the three complete rows, warm auto is
      **0.9962x** off and warm on **0.9845x**, while isolated cold auto is
      **1.1603x** and cold on **1.1546x**. Cold physical reads/major faults fall
      from **55.15 GB / 29,271** off to **55.81 MB / 11** auto; file RSS also
      falls materially. These are one-repetition diagnostics, not a thermal
      promotion gate. Keep warm/off default and cold modes separate; auto/on
      remain explicit diagnostics pending a counterbalanced cold policy gate.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-economics-matrix.json`.
- [x] Add optional per-tensor load drop-behind only for data already copied to
      device ownership. Never invalidate lazy PLE pages or validation readers;
      include reload-heavy and one-shot serving controls. The default-off
      `HIPENGINE_QWEN4_EXP_LOAD_DROP_BEHIND=1` path applies file `DONTNEED` only
      after each of 1,223 hot tensor loaders returns successful device ownership;
      descriptors are reused by part and closed on success/failure. Lazy PLE and
      validation readers are excluded. One-shot and immediate reload controls
      remain exact and zero-teardown at **47.12/45.67 s** process wall, with
      ~**243/245 MB** RSS before request, but each request rereads **13.75/15.61
      GB** and incurs **6,093/6,212** major faults. Retain only as explicit
      one-shot serving control; reload-heavy services keep it off. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p9-load-drop-behind.json`.

### Phase P10 — long-context AR and optional KV profile

Goal: close depth-dependent competitor gaps only after short BF16 AR parity and
the existing escalation thresholds permit each rung.

- [x] Re-run natural 4K, 16K, then 64K retrieval with the current stacked
      production path and exact selected-position/CPU selector controls. Current
      chunk-512 rows reach **53.08 / 51.48 / 48.02 prompt tok/s** at 4K/16K/64K.
      Every depth retrieves `VIOLET-7391`, selects the needle in all 12 QSA
      layers, matches all 2,048 binding CPU-selected positions at layer 47,
      passes repeat/rollback isolation, and tears down to zero. Historical depth
      rows are not used as old→new comparisons. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-natural-retrieval-current.json`.
- [x] Profile QSA score/top-k O(context/4), selected-K/V page locality, sparse
      attention, PLE I/O, graph reuse, and KV bytes separately at each depth.
      The corrected natural 4K/16K/64K census records 1,024/4,096/16,384 score
      rows, fixed 2,048-token selected attention, a 2,048-byte physical K+V row,
      and 1,024 selected 4-KiB pages per layer at every depth. Selected spans are
      4,096/16,384/65,536 tokens with mean gaps 2.00/8.00/32.02 and maximum gaps
      41/481/5,081. Live all-QSA-layer KV grows from 103,615,572 to
      1,613,654,028 bytes. PLE publication remains 3,143,680 H2D bytes, and MoE
      graph reuse remains 48 captures plus 14,688 replays with zero eager or
      rejected routes. Every depth passes exact retrieval, CPU-selected-position
      parity, repeatability, transaction isolation, and zero-allocation teardown.
      The diagnostic adds no default-route work and makes no throughput claim.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-structural-depth-census.json`.
- [x] Compare exact matched same-weight/BF16-KV Engram/Nathan/upstream rows; do
      not compare against their IQ3/IQ4 or quantized-KV headlines. Audit result:
      the shared canonical fixture contains 512/1,024/4,096-token rows only.
      Upstream's refreshed 4,096 row is repeat-exact at **266.58/18.98 pp/tok/s**;
      EngramHalo's **381.17/15.99** row fails `code-p4096` repeatability, and
      Nathan's **351.85/19.01** row belongs to a 0/12-repeatable lane. No retained
      exact-token BF16-KV row exists for any named external lane at 16K or 64K.
      The current hipEngine natural-retrieval rates use a different single-prompt
      protocol and are not divided by these canonical rates. Published Q8-KV and
      different-weight-quant headlines remain excluded. This closes the evidence
      inventory, not the missing 16K/64K measurement. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-matched-comparator-audit.json`.
- [x] Optimize persistent compressed-key scoring, top-k, selected attention,
      and graph/context buckets only where the depth profile ranks them. Exact
      live-16K/64K clean decode is **82.617/82.459 ms/token**. Complete QSA is
      **20.834/22.132 ms/token**, while ordered selected attention remains the
      largest named P10 path at **19.642/19.584 ms/token**. Compressed-key
      score/top-k grows from **1.116 ms (1.35%)** at 16K to only **2.490 ms
      (3.02%)** at 64K. The fixed graph count and lifecycle remain clean. The
      selected-attention route was already promoted in P6 and its next exact
      col4 leaf was rejected, so no score/top-k or graph/context candidate opens.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-16k-owner-profile.json`,
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-64k-owner-profile.json`.
- [~] After BF16 AR parity, open Q8 K/V as a T3 product configuration with its
      own CPU/reference, BF16-relative, task/retrieval, memory, deterministic,
      lifecycle, and same-config competitor gates. It cannot close BF16 parity.
      Blocked at the declared prerequisite: the retained exact ordered route is
      **12.868 tok/s** on `code-p4096`, only **0.678x** the repeat-valid
      same-weight/BF16-KV upstream **18.975 tok/s**. Qwen4Exp also has no
      registered quantized-KV storage, write, or attention family; existing
      INT8-KV evidence belongs to other model/backend configurations. Do not
      bypass BF16 parity or substitute those rows. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p10-q8-kv-readiness.json`.

### Phase P11 — device-resident MTP economics

Goal: exceed true AR on the full suite, then match only correctness-valid
external MTP rows.

- [x] Reconfirm the per-stream hyper-connection combiner and sidecar tensor map;
      do not replace it with mean pooling. Focused GGUF-map and draft-runner
      tests pass **6/6**: all 34 sidecar tensors are pinned with shape/qtype
      drift rejection, the global-normalization reference pairs every branch
      independently, and the real local sidecar is deterministic and
      transactional. Per-stream hyper-connection ownership remains unchanged.
- [x] Add phase timing/census for target hidden export, draft input fusion,
      draft layer/head, sampler, target verify, acceptance, commit/rollback,
      copies, and graphs on every category. The four-category B2/tg8 census
      records **18 cycles, 34 proposals, 28 target rows**, and clean teardown.
      Serial target verification owns **2,454.264 ms total / 87.652 ms per
      row**, versus **320.013 ms** of proposal wall; host acceptance and cursor
      repair total only **1.306 ms**. Nested asynchronous draft launch buckets
      overlap proposal wall, and the synchronized bucket owns queued GPU work,
      so they are not added together. A cached `rocprofv3` trace proves target
      prefill/decode plus Q8 sidecar embedding/QSA execution and confirms that
      this provider has no graph route. This T0 instrumentation changes no
      arithmetic or dispatch and retains `target_ar` as strict fallback. The
      retained 10-prompt reference remains `W=51.711 s` MTP and `C=49.383 s`
      true AR; this census assigns no local speedup `s` and makes no promotion.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-mtp-phase-census.json`.
      The earlier isolated sidecar diagnostic remains
      `benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-mtp-hot-head-feasibility.json`.
- [~] Keep target hidden, draft hidden chaining, logits/top-k, and candidate IDs
      on device. Remove per-draft full-logit/hidden D2H and host reconstruction;
      read one compact candidate packet per cycle at most. The first T0 rung
      keeps draft hidden chaining resident and replaces full-logit/hidden D2H
      with device argmax plus one int64 read per draft; the explicit full-output
      path remains the default/fallback. Sidecar B2 median improves
      **16.945→16.639 ms (1.018x)** with exact IDs. A warmed four-category
      same-session AB/BA is also exact and improves aggregate wall only
      **6,585.101→6,537.548 ms (1.0073x)**; `general_en` and `mixed_ja_en`
      regress to **0.9435x/0.9745x**, so promotion is rejected. Keep
      `HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT=1` default-off until candidate IDs
      become one packet per cycle and target hidden export is resident; remove
      it if that complete route still fails category non-regression. Cached
      tracing proves the candidate's argmax stage1/stage2 route. A follow-up
      appends each device argmax ID to a four-int64 device packet and performs
      one D2H per cycle. Sidecar B2 is exact at **16.794→16.591 ms (1.012x)**;
      warmed whole-model wall is **6,588.589→6,521.622 ms (1.0103x)**, but
      `general_en` and `mixed_ja_en` still regress to **0.9344x/0.9845x**.
      Packet D2H synchronizes preceding proposal work, so promotion remains
      rejected and full output remains default. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-compact-draft-output-rejected.json`
      and
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-candidate-packet-rejected.json`. A shared-runtime D2D handoff then removes post-prefill target-hidden D2H and draft-hidden H2D. It is exact and traced, but warmed whole-model wall is only **6,590.224→6,530.592 ms (1.0091x)**; `general_en`/`mixed_ja_en` regress to **0.9292x/0.9847x**. Packet synchronization masks the removed round trip, so this rung is also rejected and default-off. Evidence: `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-target-hidden-d2d-rejected.json`.
- [~] After device-output cleanup and target-verifier progress, build
      default-off individual-row compact Q8_0 draft heads at 8K/16K/32K with a
      local→global token map. Do not port EXL3 block-group rules or FP8 constants.
      Train maps on unrestricted multi-prompt proposal telemetry plus general
      corpus frequency; keep category-heldouts disjoint and require full-suite
      AR-equivalence, acceptance, memory, and same-command economics. The current
      zero-cost-head upper bound is only **0.97% complete wall / 0.964x AR**, so
      reject unchanged hot-head work if verifier progress does not raise its
      Amdahl ceiling. Admission is currently blocked: compact output and D2D
      hidden handoff both failed category promotion, the rows<=8 verifier is not
      executable, and no disjoint map-training manifest exists. Even a free
      head cannot reach 1.0x AR on retained full-suite economics. Do not build or
      fit vocabulary maps until verifier/device transactions raise that ceiling.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-compact-head-admission.json`.
- [~] Build the rows<=8 batch-invariant target verifier before raising budget.
      Its per-row decode arithmetic, GDN/QSA/PLE state, and outputs must equal
      serial target verification under the declared contract. Readiness audit:
      Qwen4Exp has no `verify_target_block` API. Its `_prefill_chunk` commits all
      rows, scores only the final residual, has rows=1 head/logit capacity, and
      cannot defer accepted-prefix state. `Qwen4ExpRunnerSnapshot` also copies
      recurrent state through the host. Qwen3.5's mature rows<=8 verifier uses a
      different linear-state/layer/graph ABI and is not reusable unchanged.
      Split this into per-row output storage, device state transactions,
      deferred accepted-prefix commit, and rows2-8 serial-reference RED gates
      before provider wiring. The first subunit is complete:
      `Qwen4ExpTargetVerifyOutput` lazily owns bounded rows<=8 residual,
      per-row-logit/token, and head-scratch storage; rows 0/9 reject before
      allocation, HIP allocation/lifecycle passes, and serial dispatch is
      unchanged. A second subunit adds exact D2D snapshot/restore ownership for
      GDN matrix/conv, PLE conv, and residual state with idempotent cleanup.
      A third subunit adds all-or-nothing begin/commit/rollback ownership over
      that snapshot, every QSA KV/index cursor, runner position, and a copied
      host PLE-hash map; double finalization rejects. A fourth subunit adds the
      explicit rows1-8 serial oracle, returning every token, full-logit row, and
      target-hidden row through unchanged `step` arithmetic. The first rows=2
      bulk candidate reused prompt `_prefill_chunk` plus rows-capable head
      output. It matched top-1 IDs but failed RED: hidden max absolute difference
      **0.40625**, logit-row maxima **0.1803/1.3686**, and GDN matrix/conv, PLE
      conv, and residual state all differed. Position and PLE hashes matched.
      The uncommitted candidate was removed before timing/tracing; prompt-prefill
      arithmetic is not the verifier. Accepted-prefix row commit and a
      verifier-specific arithmetic candidate remain next. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-rows2-prefill-verifier-rejected.json`.
      A verifier-specific follow-up preserves serial body/state order and batches
      only final GR/head scoring. Rows2-8 pass bit-exact tokens, hidden, logits,
      GDN/PLE/residual state, position, and hashes. Head-only B2 is effectively
      neutral (**126.763→126.592 ms, 1.00136x**); including the required D2D
      begin/commit transaction regresses **127.693→132.354 ms (0.9648x)**. The
      uncommitted candidate was removed before provider wiring or tracing. A
      viable verifier must capture accepted-prefix state without a full
      pre-execution snapshot; final-head batching alone cannot pay for rollback.
      A retained follow-up removes fresh HIP allocation from that transaction:
      one request-owned 119.53-MB snapshot lease lowers begin/commit host wall
      **6.568→0.0268 ms**, with downstream readback owning copy completion. The
      exact complete B2 candidate then improves **128.464→127.560 ms
      (1.0071x)**. Accepted-prefix replay now passes the complete rejection grid,
      and cached B2 tracing proves GDN decode, QSA, router, and rows=2 final-head
      execution. Keep it internal/default-off until provider wiring and
      whole-model category gates pass. Default-off provider wiring then passes
      **10/10 exact IDs** with unchanged acceptance, but full-suite tg16 is
      **26.152 s** deferred versus **22.821 s** serial MTP (**0.8726x**) and
      **16.904 s** true AR (**0.6464x**); every category regresses. Full-width
      target execution plus early-mismatch replay overwhelms pooled/head savings.
      The uncommitted provider branch was removed. A viable verifier must batch
      operation-complete target bodies or avoid rejected-suffix work. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-public-deferred-verifier-rejected.json`,
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-deferred-head-verifier-rejected.json`
      and
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-pooled-verifier-transaction.json`.
      `W=51.711 s`, `C=49.383 s`, and the measured target-step owner is
      `O=87.652 ms/row`; internal B2 `s=1.0071`, but public/full-suite `s`
      remains unknown. Strict fallback is serial `target.step`. Device-output
      cleanup remains independent. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-target-verifier-readiness.json`.
- [x] Add a rejection-depth RED sweep before any budget work. Force a rejection
      at every draft depth `d` in `[1, n_max]`, at the first, middle, and last
      position of a verify batch, and with a batch shorter than the ring depth;
      require post-rollback GDN conv state, SSM state, and the next-token logits
      to be bit-equal to the no-MTP AR path at the same position. This is the
      grid that localizes the Pat1entZ3r0 EXP-016 class: unwritten ring banks
      beyond the written group, and a spec ring one entry too shallow for a
      verify batch that carries the previously sampled token plus `n_max`
      drafts. Passing at `n_max` only is not passing. The pooled verifier
      transaction passes **12/12** real B2/B3/B4 cases: rejection at every
      depth, full acceptance, and shorter widths. Committed tokens, hidden,
      logits, GDN matrix/conv, PLE conv, residual, position, and PLE hashes are
      bit-exact to serial; the following token, full logits, and hidden row are
      also bit-exact. Rejections before the last row rollback and replay only
      the consumed prefix; last-row rejection/full acceptance commit directly.
      Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-rejection-depth-grid.json`.
- [~] Falsify EXP-016 as the explanation for the measured external MTP failures
      with one build. The failure identities are already known per lane
      (EngramHalo 9/10 failing `general_ja_plan`; Nathan 8/10 with AR and MTP
      each self-repeating 9/10; apepojken 9/10). Rebuild one lane on
      `c589f0ed` + #27879 + EXP-016 and re-run the section 5.3 equivalence
      probe: if those specific prompts move, the mechanism is confirmed and the
      hipEngine audit above is urgent; if the same prompts fail, the hypothesis
      is closed for the cost of one build. Either result is worth recording.
      Blocked on immutable source: `patches/hybrid-03-mtp/0006`, program commit
      `413c33c`, and parent `32af70900` are absent from all local comparator
      object databases, and the cited repository returns HTTP 404. Three local
      comparator trees are clean and remain untouched; the modified
      `EngramHalo.cpp-patched` tree is excluded. Obtain the exact patch/commit,
      then build it in a disposable clone rather than reconstructing it from
      prose. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-exp016-readiness.json`.
- [~] Move acceptance, first-mismatch selection, commit, rollback, and cursor
      repair to device-owned transactional kernels/graphs with exact recovery
      and cancellation tests. Admission is blocked by current ownership: the
      four-category B2/tg8 census assigns only **0.150 ms** to host acceptance
      and **1.156 ms** to draft cursor repair across 18 cycles, **1.306 ms total
      / 0.020%** of 6,590.224-ms wall. The prerequisite pooled verifier adds
      **3,331.227 ms** on the ten-prompt suite, 2,550x this owner. Building an
      acceptance kernel cannot repair verifier economics. First produce an exact
      operation-complete target-body verifier that beats serial MTP, then
      re-profile this boundary. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-acceptance-kernel-admission.json`.
- [~] Sweep budget against context depth, not budget alone, on the full
      category+heldout suite against a true no-MTP AR denominator from the same
      command. Record acceptance, visible tokens/cycle, target rows, phase wall,
      and speed by category and context. Externally the optimum is
      depth-dependent (n-max 2 shallow, 6 at >=32K, acceptance decaying
      0.94-0.97 to 0.75-0.88 by 128K), so a single budget fitted at short
      context under-serves depth. Fit a policy over measured acceptance -
      raise the budget while marginal accepted tokens per cycle still increase -
      rather than freezing a constant; a constant tuned on a fixed prompt set is
      not retainable. Admission is blocked before fitting: the public provider
      and draft are capped at **1,024 tokens**, while the committed long-context
      suite has six **4,096-token** categories; local budgets are only 1-4, not
      the external hypothesis's 6. The exact public deferred verifier is
      **0.8726x serial MTP**, and serial MTP is only **0.7407x true AR** on the
      ten-prompt suite. Sweeping <=1K budgets would fit the wrong depth and fixed
      evaluation prompts. Extend draft/provider capacity to >=4K, add B6 RED,
      and beat serial verification before fitting disjoint telemetry. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-depth-budget-admission.json`.
- [~] Evaluate confidence thresholds as explicit provider policies over the
      full suite. No fixed prompt, token, or candidate-specific policy is
      retainable. Readiness is blocked: compact output exposes device argmax ID
      and value, but no top-1 probability, runner-up margin, log-sum-exp, or
      calibration metadata. The retained ten-row artifact calls its input
      category+heldout but provides no machine-readable fit/heldout partition,
      and there is no independent policy-fit corpus. A sweep now would restore
      full-logit host output or fit the fixed evaluation prompts. Add a compact
      calibrated confidence packet, immutable disjoint fit/heldout manifests,
      and the rejection-depth gate first. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-confidence-policy-readiness.json`.
      Deprioritize the `ngram-mod` combination: externally it is
      -12% on code when combined with an MTP draft head (low-quality drafts
      dilute the head), plain n-gram speculation is 0 to -33%, and two of its
      four types hang the server on this hybrid-recurrent architecture. Run it
      only if a local trace gives a reason to expect a different result.
- [x] Require exact greedy outputs where that is the provider contract. The
      public serial provider passes **10/10** category+heldout rows and all
      generated IDs against true target AR at B2/tg16, with **134/159 = 84.28%**
      accepted drafts. It remains **51.711 vs 49.383 s (0.955x AR)**, so this
      closes correctness policy only and does not promote performance. The
      refreshed Engram row is 1.128x but only 9/10 exact and is therefore not a
      valid target or promotion precedent. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-exact-greedy-policy.json`.
- [~] Promote at >=1.0x AR with every binding category non-regressive; continue
      toward >=1.5x and the best correctness-valid same-config competitor.
      Promotion is blocked: the current same-command full-suite row is exact
      **10/10** but serial MTP is **22.821 s** versus true AR **16.904 s
      (0.7407x)**. Every category regresses (**0.7581x code, 0.7354x English,
      0.7462x Japanese, 0.7111x mixed**). The exact pooled B2 internal 1.0071x
      does not transfer publicly; deferred full-suite verification is 0.8726x
      serial. Required 4K MTP is unavailable at the 1K provider cap. Keep
      `target_ar` default and do not promote from isolated or invalid external
      rows. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p11-promotion-readiness.json`.

### Phase P12 — final refresh, cleanup, and rollup

Goal: make the complete result reproducible, default, and reversible.

- [x] Refresh all comparator lanes once under P0's exact protocol and fixed host
      state; do not compare old absolute rows to new binaries. The campaign
      inventory binds source/binary hashes and exact-fixture attempts for
      upstream Vulkan, Nathan Vulkan, patched/pristine upstream HIP,
      EngramHalo HIP, and diagnostic hybrid-04. Upstream Vulkan is 12/12
      repeatable but short-shape variance blocks final freeze; Nathan is 0/12,
      Engram 4K alternates, hybrid-04 is 2/12 exact, and pristine HIP produced
      no sample in two 1,800-second starts. This closes refresh-once inventory,
      not section-6 five-pair target freeze. Temporary binaries are gone; rebuild
      eligible lanes from pinned sources for final pairs rather than comparing
      new binaries to these absolute rows. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p12-comparator-refresh-inventory.json`.
- [~] Run the final strict/production packet, task/BF16/control/state/c2/
      lifecycle gates, exact matched p512/p1024/p4096 with tg128, and every
      unlocked long-context/MTP milestone. Current short packet passes:
      **268** focused Qwen4Exp tests; strict and production each complete 36/36
      deterministic exact-fixture samples with zero teardown. Weighted
      strict pp/tg is **61.049/13.517, 60.322/13.427, 52.561/9.472** at
      p512/p1024/p4096; production is **83.352/14.180, 82.933/14.164,
      69.200/12.160**. Production manifest `37d59564…` does not fall back;
      its existing 450-row BF16/task/state/lifecycle gate and c2 exactness remain
      binding because later default-path changes are unselected T0 verifier
      primitives. Current 4K/16K/64K retrieval remains qualified. Final packet
      is still blocked by five-pair eligible comparator windows and 4K MTP at
      the provider's 1K capacity; neither is relabeled as passed. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-packet.json`.
- [~] Emit compact accepted/rejected/blocked artifacts, raw-log hashes, generated
      reports, benchmark README/changelog updates, and the model checkpoint.
      Current P12 packet, compact report, and README/changelog rollup are current;
      report SHA-256 is `7130500d…`. Strict/production raw result hashes are
      recorded without committing 148-KB raw files. Final-freeze comparator raw
      hashes and the final model checkpoint remain blocked with their five-pair
      and 4K-MTP inputs; do not publish a closure checkpoint early. Evidence:
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-packet.json`
      and
      `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p12-validation-report.md`.
- [x] Update `docs/KERNELS.md`, lineage metadata, `docs/REFACTOR.md`, and
      `docs/PLAN.md` if architecture moved. P11 added runtime-only verifier
      storage/transaction/oracle primitives, not a kernel or lineage entry;
      `docs/KERNELS.md` and `docs/source_lineage.json` therefore stay unchanged.
      `docs/PLAN.md` now records current manifest/rates and MTP/verifier bounds;
      `docs/REFACTOR.md` records selector closure and retained oracle lifetime.
      The optional lineage audit is currently environment-blocked because
      `/home/lhl/amd-gpu-tuning/reference/atlas` is absent, but no kernel port or
      source-parent claim was made in P11/P12.
- [~] Remove superseded experiment flags, dead selectors, duplicate fallback
      chains, and stale graph routes only after their replacements are default.
      Removed public `HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT` selection and its
      resident-hidden provider branch after three exact candidates failed
      category promotion. The explicit runner `compact_output=True` leaf remains
      only for focused rejected-path oracles/profiling. Retain strict serial MTP,
      verifier oracles/transactions, and graph fallbacks because no replacement
      is default. The broader campaign's rejected/default-off selectors remain
      tracked in `docs/REFACTOR.md`; final cleanup stays partial.
- [ ] Commit every validated logical unit; campaign closure is its own final
      decision/worklog commit.

## 6. Acceptance and closure

All comparisons bind to the same physical host, exact target parts, weight
quant, K/V type, prompt/token fixture, cache mode, sampler, output horizon,
power policy, and declared profile. A different representation gets a separate
milestone and denominator.

### 6.1 Match and beat rules

- Screening may use three counterbalanced repetitions. Closure uses at least
  five counterbalanced pairs after warmup, reports every sample and coefficient
  of variation, and keeps competitor/hipEngine runs in the same thermal window.
- **Beat** means the paired 95% confidence interval for the hipEngine/comparator
  throughput ratio is above 1.0.
- **Match** means the median ratio is at least 0.98, both arms have coefficient
  of variation at most 2%, and the paired 95% interval includes 1.0. A noisy
  result is neither a match nor a loss; stabilize and repeat.
- p512, p1024, p4096, and context-conditioned tg128 bind independently. A large
  prefill win cannot average away a decode loss, and one category cannot
  compensate for another.
- End-to-end unprofiled wall is binding. Kernel/sub-window/launch/copy wins are
  retainable evidence but do not by themselves close parity.

### 6.2 Milestones

1. **P0 measurement closure:** exact matched cross-engine harness, frozen source
   and binary identities, host state, cold/warm separation, and category
   heldouts are committed.
2. **HIP short-AR parity:** named hipEngine production matches or beats the best
   refreshed same-configuration HIP comparator on p512, p1024, p4096, and the
   corresponding tg128 rows.
3. **Same-host short-AR parity:** the same path matches or beats the best
   refreshed same-configuration HIP or Vulkan engine on every canonical row.
   Vulkan is a required second milestone, not an optional source of ideas.
4. **Role closure:** no layer-2/early-MoE, shared/router/GR, GDN, QSA, output,
   submission, copy, or synchronization bucket remains an unexplained multi-x
   outlier. Any residual >=1.25x names a measured architectural reason and an
   exhausted or blocked action.
5. **Correctness/product closure:** production manifest and strict fallbacks,
   complete numerical/BF16/task/category packet, same-schedule deterministic
   repeats, graph/eager and restore/replay, physical c2/isolation, cancellation,
   memory, and teardown all pass.
6. **Long-context parity:** every context rung unlocked by the benchmark ladder
   passes retrieval/control gates and matches the best exact same-config
   comparator at that depth. Historical or different-KV rows do not bind.
7. **MTP parity:** the full category+heldout suite is exact under the provider
   contract and non-regressive in every binding category versus true AR. It
   reaches at least 1.5x AR or matches/beats the best correctness-valid
   same-configuration external MTP row, whichever is higher. A competitor row
   that fails output equivalence is not a target.
8. **Evidence closure:** the final artifact contains exact commands,
   source/binary/model/manifest hashes, raw trace/result hashes, role and
   launch/API/copy census, the final non-overlapping Amdahl ledger, host state,
   all correctness verdicts, and a generated report from
   `scripts/qwen4exp_perf_gap_report.py`. Rollups, catalog, refactor ledger,
   worklog, and atomic commits are current.

A final deliberate comparator refresh freezes the closure target. A later
external commit does not retroactively invalidate the artifact; it is a new
baseline event and, if desired, a new campaign.

## 7. Copyable coder goal

```text
Execute docs/QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md as the active
performance authority for Qwen3.8-Flash-Next on zbook/gfx1151. Keep the pinned
Unsloth UD-Q4_K_XL target weights and BF16 K/V as the binding AR configuration.
Do not obtain or substitute a new weight quant. Treat Q8 K/V and MTP as separate
product configurations with their own gates.

Do not declare success after a microbenchmark, a single prompt, HIP-only parity,
or one accepted optimization. Execute section 2.2's impact queue, not phase
number order. Before coding, record W/C/O/s, overlap exclusions, zero-cost
ceiling, projected wall saving, and target-gap coverage. Re-profile after every
retained unit and re-rank the remaining work. The short-AR objective is complete
only when named hipEngine production matches or beats the final refreshed best
same-host HIP and Vulkan comparators on exact-matched p512, p1024, p4096, and
their tg128 rows under section 6's statistical rule, while the complete
execution-profile, task, determinism, state/isolation, lifecycle, and strict-
fallback gates pass. Then continue through every unlocked
long-context rung and the device-resident MTP milestone; MTP must beat true AR
on the full category+heldout suite and reach the section 6 target without
benchmark gaming.

Start with the exact ordered live-4097 QSA decode dataflow. Then execute
operation-complete prefill MoE, dense/GR, p4096 QSA prefill, short-decode
selected projections/Q8, and GDN in measured order. P8 is not rank 2: do not
integrate it until a same-session named-production arm supplies `O` and `s`.

Settle the P0 measurement gaps before the closure freeze: GPU clock policy must
be declared and identical across both arms of every paired row, and the Vulkan
lanes must be audited for configuration they are entitled to
(`GGML_VK_ALLOW_GRAPHICS_QUEUE` first). A milestone-3 target frozen against an
under-configured Vulkan lane is invalid and has to be re-frozen.

Treat every external mechanism in section 4 as an untested hypothesis with a
named falsification, never as a result. Nothing from an external program is
adopted because it was published; it is adopted after it reproduces here under
this campaign's gates, or closed with the evidence that killed it. The MTP
rejection-depth RED sweep in P11 runs before any budget tuning, and a
non-exact continuation is classified as tie or state class by the section 2.1
rule before it is reported as either noise or a bug.

For every implementation unit: declare the measured owner, complete-wall and
gap-coverage ceilings, arithmetic class, affected scope, mechanism, and strict
fallback; cite the concrete hipEngine-versus-comparator implementation delta;
add the RED oracle; inspect kernel lineage before a port; benchmark actual
rotating weights; prove the expected kernel ran; run the full applicable gate
and same-session whole-model A/B; retain or reject from evidence; update compact
artifacts, benchmark rollups, kernel/refactor docs, and the immutable worklog;
then commit the validated unit immediately. Never hardcode prompt/token/
candidate behavior, weaken a gate, relabel a different representation, add
torch to the hot path,
or add backend/quant branches outside the registry.

Do not stop because one hypothesis loses. Record the rejection and move to the
next measured owner. If blocked, write the exact blocker and continue every
independent phase. Stop only on explicit user pause or when all campaign
acceptance criteria are satisfied and the final comparator refresh, evidence,
cleanup, and commits are complete.
```

## 8. References

- Campaign authority: [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md)
- Benchmark policy: [`BENCHMARK.md`](BENCHMARK.md)
- Testing policy: [`TESTING.md`](TESTING.md)
- Profile contracts: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)
- Kernel catalog and port rules: [`KERNELS.md`](KERNELS.md)
- gfx1151 roofline: [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md)
- Canonical exact-token fixture and driver:
  [`qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json)
  and `scripts/qwen4exp_canonical_ar_bench.py`.
- Tracked implementation leads:
  [cafe-llama.cpp](https://github.com/quimmedes/cafe-llama.cpp) and
  [omlx PR #3260](https://github.com/jundot/omlx/pull/3260).
- Source-reviewed external program (no local reproduction):
  [Pat1entZ3r0/strix-qwen-next-flash-optimization](https://github.com/Pat1entZ3r0/strix-qwen-next-flash-optimization)
  at `413c33c`; full review in survey section 6.7.
- Canonical exact-token screening:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-canonical-ar-screening.json)
- Cross-engine Strix Halo speed/accuracy survey:
  [`QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md`](QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md)
- Current canonical impact profile:
  [`benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json`](../benchmarks/results/2026-09-01-gfx1151-qwen38-flash-next-canonical-impact-profile.json)
- Historical p508/tg32 profile:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json)
- External fork refresh:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-external-fork-refresh.json)
- Corrected invalid decode route:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json)
- Durable profiling tools: `scripts/qwen4exp_profile_gap.py`,
  `scripts/qwen4exp_context_decode_profile.py`,
  `scripts/qwen4exp_llamacpp_exact_profile.py`,
  `scripts/qwen4exp_trace_analyze.py`, `scripts/qwen4exp_role_analyze.py`,
  `scripts/qwen4exp_decode_sync_ab.py`, and
  `scripts/qwen4exp_perf_gap_report.py`.
