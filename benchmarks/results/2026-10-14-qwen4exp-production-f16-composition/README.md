# Qwen4Exp `production` with the F16 route composition: rates, gates and long-context sensitivity

Date: 2026-10-13 UTC. Host: AMD Radeon 8060S / gfx1151, machine
`55ea6c509d0b49eea8de7094a1023668`, ROCm 10.0.0 (TheRock). Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, BF16 KV, prefill chunk 4096.

The `production` plan for `qwen4_exp_gguf` / `hip_gfx1151` /
`gguf_ud_q4_k_xl` runs the routed experts, the dense projections at every
layer, the hyper-connection projections, the indexer projections, the router
and the prefill attention on F16-operand matrix kernels with F32
accumulation, and the tiled GDN recurrence at every GDN layer. Every route
keeps its registered strict fallback. Before this change the plan ran exact
or exact-repaired arithmetic for most of those families ("previous plan"
below).

## Rates

Canonical 12-case protocol
(`scripts/qwen4exp_canonical_ar_bench.py hipengine --prefill-chunk-size 4096
--capture-mode compact --ple-cache-mode warm --execution-profile production`),
one warmup and three measured repetitions per case, 128 decode transitions,
tracked-clean tree at `6d8306d1e`. Artifact:
[`hipengine-production.json`](hipengine-production.json). The previous plan
was measured on the same host and protocol earlier the same day
([artifact](../2026-10-14-qwen4exp-gdn-norm-gate-waves/hipengine-production.json)).

| Prompt tokens | Previous plan prefill tok/s | Prefill tok/s | Ratio | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 305.6 | **769.2** | 2.52x | 20.91 |
| 1024 | 331.6 | **932.9** | 2.81x | 20.15 |
| 4096 | 335.5 | **1108.4** | 3.30x | 20.57 |

Decode is unchanged (20.92 / 20.16 / 20.54 before). All 12 cases are
deterministic across repetitions.

| Case | Previous prefill tok/s | Prefill tok/s | Decode tok/s | 128-token output vs previous plan |
| --- | ---: | ---: | ---: | --- |
| code-p512 | 305.5 | 764.8 | 20.91 | differs |
| general_en-p512 | 306.0 | 769.1 | 20.91 | same |
| general_ja-p512 | 310.5 | 784.0 | 20.91 | same |
| mixed_ja_en-p512 | 304.3 | 760.3 | 20.91 | same |
| code-p1024 | 329.6 | 923.8 | 20.16 | differs |
| general_en-p1024 | 332.8 | 936.9 | 20.15 | differs |
| general_ja-p1024 | 333.3 | 945.4 | 20.16 | differs |
| mixed_ja_en-p1024 | 330.0 | 925.4 | 20.15 | differs |
| code-p4096 | 336.9 | 1106.2 | 20.57 | differs |
| general_en-p4096 | 337.4 | 1109.9 | 20.57 | differs |
| general_ja-p4096 | 337.4 | 1110.3 | 20.57 | differs |
| mixed_ja_en-p4096 | 337.0 | 1107.7 | 20.57 | differs |

Nine of twelve greedy continuations differ from the previous plan's.
Generated-ID equality across routes is a diagnostic; the long-context
section below explains why it is expected here.

## Default-suite gate (short prompts)

`scripts/qwen4exp_layer2_profile_gate.py --candidate production_baseline
--prefill-chunk-size 4096`: the unmodified `production` profile against the
`strict` teacher, 18 prompts of at most a few hundred tokens, 24
teacher-forced decode steps, full-vocabulary logits, 450 rows, three
candidate repeats. Tracked-clean at `6d8306d1e`. Artifact:
[`gate.json`](gate.json).

| Metric over 450 rows | Measured | Limit | |
| --- | ---: | ---: | --- |
| Mean KL | 6.493e-04 | 1e-3 | pass |
| p95 KL | 3.452e-03 | 5e-3 | pass |
| p99 KL | 7.889e-03 | 2e-2 | pass |
| Max KL | 1.186e-02 | 5e-2 | pass, no row over the 2e-2 review boundary |
| Top-1 agreement | 99.56% (448/450) | 99% | pass |
| Per-scope top-1 floor | no scope below 99% | 97% | pass |

Three scopes are listed as KL advisories, which do not bind:
`prefill_last` / `prefill_to_c1` (mean 1.206e-3, 18 rows) and `mixed_ja_en`
(mean 1.108e-3, p95 5.636e-3, 100 rows). Repeats are bit-identical and the
state gates pass. Status is `requires_task_review`: 15 of 18 free-running
tasks are strict-exact, the same count the previous plan measured.

## 18-prompt cross-engine battery

Raw-text continuation, 32 steps, top-20 log-probabilities, scored against
the strict teacher ([`quality-score.json`](quality-score.json)): 14/18
strict-exact, aligned top-1 agreement 1.00, top-20 mean KL 0.049,
deterministic. The previous plan scored 15/18 and mean KL 0.0052 on this
battery; halo-box scored 11/18 (mean KL 0.241) and gufo 13/18.

## Long-context gate and what it measures

The default suite's prompts are short, so it does not exercise the routes
that engage only on long prefill chunks, or the attention route's sparse
window. `benchmarks/prompts/qwen4exp-long-context-gate.jsonl` adds four
prompts, one per category, of 1,981 to 2,572 tokens built by repeating the
canonical category material; 100 teacher-forced rows. This is a small,
repetitive sample: four contexts, 25 correlated rows each.

`--candidate production_baseline --max-sequence-length 6144 --prompts
benchmarks/prompts/qwen4exp-long-context-gate.jsonl`, tracked-clean at
`517dcc661` ([`gate-long-context.json`](gate-long-context.json)), and
isolation arms that clear the composition after the binder and restore one
route ([`long-context-isolation/`](long-context-isolation/)):

| Arm against `strict` | Mean KL | p95 | Max | Top-1 | Strict-exact tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous plan (composition cleared) | 8.30e-04 | 5.17e-03 | 9.81e-03 | 99/100 | 3/4 |
| `production` (full composition) | 2.10e-03 | 1.28e-02 | 2.06e-02 | 99/100 | 3/4 |
| Previous plan + routed F16 GEMMs | 2.04e-03 | 1.06e-02 | 2.51e-02 | 100/100 | 4/4 |
| Previous plan + dense wide at layers 0-12 | 1.70e-03 | 1.05e-02 | 2.01e-02 | 100/100 | 4/4 |
| Previous plan + hyper-connection up on the wide route | 2.03e-03 | 8.62e-03 | 2.52e-02 | 99/100 | 3/4 |
| Previous plan + WMMA prefill attention | 2.11e-03 | 1.56e-02 | 2.86e-02 | 98/100 | 2/4 |
| Previous plan + indexer and router F16 GEMM | 2.30e-03 | 1.28e-02 | 3.86e-02 | 99/100 | 3/4 |
| Previous plan + tiled GDN at layers 0-26 (F32 reassociation) | 1.55e-03 | 7.90e-03 | 2.70e-02 | 98/100 | 2/4 |
| Control: previous plan + F32 summation-order change in one hyper-connection projection | 2.09e-03 | 1.05e-02 | 4.81e-02 | 99/100 | 3/4 |

Every arm is deterministic across three repeats and measurement-valid.

Reading:

- On these prompts `production` is outside the calibrated envelope (mean
  2.10e-3 against 1e-3, p95 1.28e-2 against 5e-3), and so is the previous
  plan on p95.
- The control arm changes no operand precision at all: one projection keeps
  F32 operands and F32 FMA and only reorders its summation. It lands at the
  same mean KL as the full composition and has the largest maximum.
- Single routes do not add: each one alone gives 1.6e-3 to 2.3e-3, and all
  of them together give 2.1e-3.

At 2,000 tokens and above, any perturbation of round-off size or larger moves
the distribution by about 2e-3 mean KL against `strict`. The envelope was
calibrated on prompts of tens to hundreds of tokens and cannot separate an
F16 route from a reassociation at this length, so this gate does not single
out a route, and no non-bit-identical plan can pass it. Top-1 agreement
(98-100%) and the task count are the same across arms within the resolution
of 100 rows and 4 tasks.

## Strict fix found on the way

`strict` could not prefill a chunk longer than 256 tokens: an entry point its
hyper-connection read resolves above 256 rows had been dropped from the
kernel source on 2026-10-06. Every earlier teacher-forced gate ran prompts
short enough to miss it. Restored in `517dcc661` with a GPU test that
launches the strict owner through the registry.
