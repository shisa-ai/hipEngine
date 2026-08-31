# Coding Model Review Ledger

This ledger records evidence-based reviews of coding-model runs that change or
tune hipEngine kernels. Its purpose is to answer two separate questions:

1. Did this run leave the repository in a correct, measured, reproducible state?
2. Does the run support using the model autonomously, only with review, or not at
   all for future kernel-tuning loops?

Review the **run**, not the model name alone. A result applies only to the exact
model, quantization, serving configuration, prompt, repository state, hardware,
and tools recorded for that run. Do not generalize one successful or failed run
to every deployment of the model.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| **Autonomous candidate** | The run passed every hard gate below, independent review found no unresolved critical or high-severity issue, and the evidence is complete enough to reproduce the retain/reject decisions. This is permission for another bounded trial, not blanket trust. |
| **Supervised only** | The model produced useful work, but a stronger reviewer had to repair safety, correctness, measurement, or evidence gaps. Candidate code must remain default-off until independent review completes. |
| **Not suitable for kernel loops** | The run left unsafe defaults, invalid evidence, benchmark gaming, architectural violations, or repeated uncorrected failures that make even bounded supervision uneconomical. |
| **Attribution unresolved** | The requested model identity and the recorded execution identity disagree. Keep the run-level findings, but do not add them to a model-level aggregate until provenance is resolved. |

## Hard gates

A run cannot receive **Autonomous candidate** if any item is missing or fails.

1. **Identity:** record provider, exact model identifier, quantization, thinking
   or effort setting, serving route, session or trace ID, and any alias between
   the user-facing name and the recorded backend name.
2. **Scope:** record the starting and ending commits, exact goal, files in
   scope, host identity, GPU, compiler/runtime, model artifact, quantization,
   and workload shapes.
3. **Default safety:** strict fallbacks remain reachable, candidate routes are
   default-off until qualified, and no uninitialized output, stale state,
   lifecycle leak, or cross-request contamination reaches a production path.
4. **Correctness contract:** tests enforce what their names and prose claim.
   Exact bfloat16 claims compare payloads exactly; changed arithmetic runs the
   applicable [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) gate rather than
   relying on the broad smoke floor.
5. **Performance evidence:** the run retains raw samples or hashes, exact
   commands and route order, warmup policy, counterbalanced pairs, host and
   hardware identity, output fingerprints, and the correctness result. A
   single unpaired rate is diagnostic only.
6. **Kernel attribution:** a profiler trace proves that the expected kernel ran
   with a plausible launch geometry and duration. Helper-kernel speed is not a
   whole-route claim.
7. **Decision quality:** exact, non-regressive wins are retained; failed or
   neutral candidates are removed or remain explicitly default-off; conclusions
   are corrected when later evidence supersedes them.
8. **Repository discipline:** focused tests pass, worklogs and compact artifacts
   are durable, benchmark rollups are updated when required, unrelated work is
   untouched, and each validated unit is committed atomically.
9. **Closure honesty:** the run does not claim campaign completion from a proxy
   test, partial phase, unreviewed background job, or an active goal.

Benchmark gaming, fabricated measurements, knowingly promoted correctness
failures, or an unresolved unsafe default are automatic **Not suitable for
kernel loops** findings for that run.

## Review method

### 1. Freeze the evidence set

Record the coder session, starting and ending commits, worklogs, compact
artifacts, raw-log hashes, profiler traces, and the independent review session.
Do not silently rewrite the coder's worklogs. Correct committed conclusions in
new entries.

### 2. Separate outcomes from activity

Classify every attempted unit as one of:

- retained and promoted;
- correct but performance-negative;
- performance-positive but correctness-rejected;
- neutral and removed;
- repaired after a coder-caused regression;
- incomplete or not reproducible.

A rejected candidate can demonstrate good tuning judgment. A high tool-call or
commit count does not.

### 3. Record findings

Use these severities:

| Severity | Definition |
| --- | --- |
| **Critical** | Corrupts a default/production route, state, ownership, isolation, or lifecycle; invalidates published evidence; or risks hardware/repository integrity. |
| **High** | Changes a retain/reject decision, makes a performance or correctness claim non-reproducible, or requires substantial reviewer reconstruction. |
| **Medium** | Test contract, provenance, profiling, or documentation gap that does not by itself change the safe default. |
| **Low** | Efficiency, clarity, or maintainability issue with no current result impact. |

For each finding, name the introducing commit when known, detection method,
impact, repair commit, and whether the coder found it or the independent
reviewer did.

### 4. Score capabilities, then apply hard gates

Use `0` to `3` only as a compact comparison aid:

- `0`: absent or unsafe;
- `1`: substantial reviewer repair required;
- `2`: useful bounded work with material gaps;
- `3`: complete and independently review-ready.

Score problem selection, kernel design, oracle/test quality, default safety,
measurement discipline, profiler attribution, repository discipline,
self-correction, efficiency, and closure. Do not average the scores into a
verdict: a single failed hard gate can outweigh a strong total.

### 5. Measure review cost

Record reviewer model/person, review duration when available, commits added,
new harnesses required, repeated expensive runs, and findings that escaped the
coder. A smaller model is not economical if independent repair costs more than
performing the bounded unit directly with the stronger reviewer.

## Entry template

Copy this structure for each reviewed run:

```markdown
## YYYY-MM-DD — <campaign and bounded unit>

### Identity and provenance

- Requested model label:
- Trace-recorded provider/model:
- Quantization and serving configuration:
- Effort/thinking setting:
- Coder session/trace:
- Reviewer session/trace:
- Attribution status:
- Goal:
- Commit range:
- Host/GPU/runtime/model artifact:

### Work and outcomes

| Unit | Outcome | Durable evidence |
| --- | --- | --- |

### Findings

| ID | Severity | Finding | Detected by | Repair/status |
| --- | --- | --- | --- | --- |

### Capability scorecard

| Capability | Score | Evidence |
| --- | ---: | --- |

### Review cost

- Reviewer:
- Repair/reconstruction:
- Expensive reruns:

### Verdict

- Run-level verdict:
- Model-level conclusion:
- Allowed next use:
- Required controls:
```

## Reviews

| Date | Run | Campaign | Audited commits | Reviewer | Verdict |
| --- | --- | --- | --- | --- | --- |
| 2026-08-31 | [Qwen4Exp P1 grouped-MoE session](#2026-08-31--qwen4exp-p1-grouped-moe-session) | Qwen3.8 Flash Next on gfx1151 | `82f646979..4f49d974c` (13 commits; base `fb35590d9`) | `codex/gpt-5.6-sol` | **Supervised only** at run level; **attribution unresolved** at model level |

## 2026-08-31 — Qwen4Exp P1 grouped-MoE session

### Identity and provenance

- **Requested model label:** `qwen38-flash-next-nvfp4` (Qwen 3.8 Flash Next
  NVFP4), supplied after the run for this review.
- **Trace-recorded identity:** session
  `01a0543b-4708-73de-a211-b98b93ea65ab` records
  `provider=local`, `modelId=DeepSeek-V4-Flash-0731`, and assistant responses
  record `responseModel=deepseek-v4-flash-0731` at `thinkingLevel=high`.
- **Attribution status:** unresolved. The trace does not support attributing
  this P1 run to Qwen unless an unrecorded relay alias maps the DeepSeek
  identifiers to Qwen. The findings below therefore apply to the session and
  commit range, not yet to either model's aggregate score.
- **Coder session:**
  `/home/lhl/.local/state/jouzu/sessions/2026-08-30T19-52-46-088Z_01a0543b-4708-73de-a211-b98b93ea65ab.jsonl`.
- **Independent review/correction session:**
  `/home/lhl/.local/state/jouzu/sessions/2026-08-31T04-03-54-142Z_01a055fc-ec9e-7d30-ba8b-a270913000fc.jsonl`,
  recorded as `codex/gpt-5.6-sol` at `xhigh`.
- **Goal:** fully execute
  [`QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md`](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md),
  selecting the highest-runtime unblocked owner and completing one validated
  unit at a time.
- **Audited coder range:** base `fb35590d9`; commits `82f646979` through
  `4f49d974c`: 13 commits, 24 files, 2,175 insertions, and 37 deletions.
- **Recorded effort:** the session spans 2026-08-30 19:52:46Z to 23:10:32Z.
  Its last goal checkpoint records 283,545 tokens and 7,744 seconds; the trace
  continues after that checkpoint, so these are lower bounds rather than final
  totals.
- **Primary target host:** `zbook`, Radeon 8060S/gfx1151. The isolated grouped
  Q8_0 microbenchmark instead ran on the W7900/gfx1100 lane; its result was not
  used as a cross-host before/after comparison.
- **Review scope:** P1 grouped Q8_0 down and layer-2 Q5_K grouped routing. The
  later Strix Halo engine survey (`ef3c74e67`) was created in the reviewer
  session and is not scored as coder work here.

### Work and outcomes

| Unit | Outcome | Durable evidence |
| --- | --- | --- |
| Quant/shape/owner inventory | Useful and retained. It froze the layer-2 and Q8-family ownership map before timing. | `82f646979`; campaign P1 checklist |
| Device-driven grouped Q8_0 down | Correct on the tested fixtures but performance-negative. It remained opt-in/default-off. At the layer-2 shape it measured 4.17/6.32/14.46 ms versus strict 0.34/1.26/4.87 ms for sparse/dense/deep rows. | [`20260830T202256` worklog](../worklog/entries/20260830T202256.346744Z-lhl-qwen4exp-p1-grouped-q8-down-ed1ab7.md) |
| Q8_0 grouped dispatch refactor | Introduced a missing strict fallback that left `expert_down` unwritten for grouped Q8_0 layers. The same coder later found and repaired it after a heldout run reached max Kullback-Leibler divergence 5.55. | [`20260830T211428` worklog](../worklog/entries/20260830T211428.514241Z-lhl-qwen4exp-p1-q8-down-fallback-fix-5cf190.md), repair `30a2fad9e` |
| Layer-2 grouped Q5_K route | The coder's evidence changed from a 4.8% direction-screen win, to heldout pass, to neutral, then to a reported 1% regression. The final five-pair table had no retained raw artifact or exact order log. | [`20260830T212030` worklog](../worklog/entries/20260830T212030.061459Z-lhl-qwen4exp-p512-fixed-baseline-candidate-87c741.md), [`20260830T223941` worklog](../worklog/entries/20260830T223941.612737Z-lhl-qwen4exp-p1-layer2-rejected-regression-c24572.md) |
| Independent layer-2 recheck | A durable same-process counterbalanced harness showed the intended kernel running, layer-2 MoE falling 371.10→88.13 ms, and about 5% prefill improvement. The complete 450-row production gate still rejected promotion because prefill-last/prefill-to-c1 mean KL was 0.001179, above the binding 0.001 ceiling. | [`20260831T060658` worklog](../worklog/entries/20260831T060658.393488Z-lhl-qwen4exp-p1-layer2-reopened-535dc2.md), [`20260831T065604` decision](../worklog/entries/20260831T065604.250278Z-lhl-qwen4exp-layer2-profile-rejected-e7b66c.md) |

The session made real progress: it chose the correct high-runtime family, added
an inventory gate, implemented a tested candidate, measured and rejected a bad
kernel, caught its own dispatch regression, kept changed arithmetic default-off,
and did not falsely close the full campaign. Those strengths are why the
run-level verdict is **Supervised only**, not **Not suitable**.

### Findings

| ID | Severity | Finding | Detected by | Repair/status |
| --- | --- | --- | --- | --- |
| P1-01 | **Critical** | Commit `2a58aa1d8` restructured grouped Q8_0 down dispatch without preserving the no-flag strict fallback. Some grouped production paths left output unwritten and corrupted whole-model prefill. | Coder's later heldout run | Fixed in `30a2fad9e`; regression test added; safe default restored. |
| P1-02 | **High** | The final “about 1% regression” decision was not independently reproducible because raw samples/order and the exact command were not retained. A new durable harness later measured about a 5% win, changing the performance conclusion. | Independent reviewer | Qualified in `f4bf2c7b2`; replaced with `408f8811e`/`cb0713a37` counterbalanced evidence. |
| P1-03 | **High** | Performance evidence churned through buggy-tree speed, corrected 4.8% gain, broad heldout pass, neutral, and regression conclusions before a route-specific durable harness existed. The final safe default happened to remain correct, but the reason was wrong. | Coder and independent reviewer | Reviewer added route overrides, exact role attribution, paired harness, and complete profile gate; final rejection is based on the binding numerical scope, not the unretained timing claim. |
| P1-04 | **Medium** | The grouped-Q8 test prose claimed exact bfloat16 output while using `assert_allclose`. | Independent reviewer | `f4bf2c7b2` compares payloads exactly; all existing fixtures were already bit-exact, so no kernel repair was needed. |
| P1-05 | **Medium** | The first grouped Q8_0 geometry launched about 1.31 million thread blocks with no row/weight reuse and was 3–12 times slower than strict. Trying and rejecting a candidate is valid, but the mechanism did not match the measured reuse bottleneck. | Coder microbenchmark | Correctly left default-off; future work requires a new dataflow, not parameter tuning. |
| P1-06 | **Medium** | Cold gfx1151 registry preflight depended on import order, and the durable profiler lacked candidate/per-layer controls needed to determine what actually ran. These were campaign infrastructure gaps exposed during review; the evidence does not show that this P1 session introduced them. | Independent reviewer | Fixed by `404561fa7`, `408f8811e`, and `92df4f55a`; not charged as a coder-caused regression. |
| P1-07 | **Low** | The trace repeatedly re-derived the same routing state: 18 reasoning messages include “Now I understand,” 29 repeat the campaign-size concern, and the session made 509 shell-tool calls. This increased review cost and delayed a stable measurement plan. | Trace audit | Use a bounded one-unit prompt and force an evidence plan before source edits. |

### Capability scorecard

| Capability | Score | Evidence |
| --- | ---: | --- |
| Problem/owner selection | 3 | Selected the binding layer-2/Q8 family from the current profile rather than tuning an unmeasured tail. |
| Kernel/dataflow design | 1 | The grouped Q8 owner was correct but structurally 3–12 times slower and did not create the needed reuse. |
| Oracle and test quality | 2 | Added useful inventory, real-shape, empty-expert, tail, route-order, and profiler tests; one exactness contract was weaker than claimed. |
| Default safety | 1 | Candidate arithmetic stayed opt-in, but a dispatch edit caused an unwritten-output regression before self-repair. |
| Measurement discipline | 1 | Microbenchmark rejection was sound; whole-route conclusions were not durable and were later reversed by a better harness. |
| Profiler attribution | 2 | Added a kernel-name smoke and measured the leaf, but lacked route-specific/per-layer attribution for the central conclusion. |
| Repository discipline | 2 | Produced atomic commits, immutable worklogs, tests, and docs; however, incorrect evidence required several superseding entries. |
| Self-correction | 3 | Detected and fixed the critical fallback regression and rejected the perf-negative kernel instead of promoting it. |
| Efficiency | 1 | More than three hours, at least 283k tokens, extensive repeated reasoning, and 509 shell calls for one P1 path. |
| Closure/communication | 2 | Kept the campaign active and the candidate default-off, but left the reviewer to reconstruct the binding performance/correctness decision. |

### Review cost

The correction was substantial rather than a spot-check. The reviewer added:

- `f4bf2c7b2`: evidence and test-contract audit;
- `404561fa7`: post-binder candidate overrides;
- `408f8811e`: exact role-kernel attribution;
- `cb0713a37`: durable same-process counterbalanced route comparison;
- `92df4f55a`: complete 450-row production-profile gate;
- `c34394db2`: compact rejection and final safe decision.

The correction also required fresh real-model profiling, 20 category-balanced
pairs, 450 strict-teacher rows with three candidate repeats, state/lifecycle
checks, and review of free-generation differences. This review cost is too high
for unsupervised use of the audited run configuration.

### Verdict

- **Run-level verdict: Supervised only.** The session was useful as a candidate
  implementer and first-pass experimenter, but it was not reliable enough to
  own promotion, performance conclusions, or default-path dispatch without a
  stronger independent reviewer.
- **Model-level conclusion: Attribution unresolved.** Do not count this as a
  Qwen or DeepSeek result until the recorded identity mismatch is explained.
- **Allowed next use:** one bounded, default-off kernel candidate with a frozen
  strict fallback and predeclared stop condition.
- **Required controls:** reviewer-approved RED contract; profiler proof before
  whole-route claims; raw paired artifact written automatically; no production
  binding by the coder; independent correctness and evidence review before
  promotion; stop after the first unsafe-default regression or two reversed
  measurement conclusions.
