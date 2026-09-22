---
status: current
owns: Bounded DMS selector improvement campaign, experiment gates, execution punchlist, and inline results pending human review.
---
# DMS selector improvement campaign

Campaign state: **running — Phase A preflight in progress**. Opened 2026-09-21;
execution started 2026-09-22 under run
`selector-improvement-run-20260922-062438`.

Terminology: Dynamic Memory Sparsification (DMS) selects key/value (KV) cache
entries to retain. Q, K and V are attention queries, keys and values. W is the
protected-window size; CR is the eligible-history compression ratio. KL means
Kullback–Leibler divergence in nats, computed as `KL(dense teacher || candidate)`
over the full vocabulary. BCE is binary cross-entropy; MSE is mean squared error.
Top-1 agreement compares the highest-probability tokens at identical inputs.

## 1. Objective and authority

Determine whether better selection can reduce protected-window size or increase
eligible-history compression without unacceptable quality loss. Compare policy,
training objective, label horizon, value-aware importance, training diversity,
query information, and predictor capacity. Produce a complete evidence packet
and recommendations; the human lead will decide the next product steps after
reviewing this document.

The coder is authorized to implement missing experiment tooling, run the bounded
matrix below, repair defects, and commit validated logical units. Do not stop at
the first successful candidate. Do not launch experiments merely to finish this
planning-document task; execution begins when the coder takes the campaign.

This is an explicit decision-policy experiment (T3), not a T1/T2 arithmetic
optimization. The gates below select research finalists; they do not impose
model-admission allowlists, change the general product default, or certify
concurrent serving. New usable tooling should work directly, not hide behind
unnecessary default-off flags. Experimental oracle access belongs in diagnostic
tooling, never a production prompt-dependent routing branch.

Authorities: [AGENTS.md](../../AGENTS.md), [optimization rules](../OPTIMIZATION.md),
[testing](../TESTING.md), [execution profiles](../EXECUTION-PROFILES.md), and
[exploration methodology](../reference/PROCESS-EXPLORATION.md). This campaign
owns its T3 research thresholds; it does not replace production arithmetic gates.
[DMS architecture/history](../reference/DMS.md) and
[DMS quality analysis](../reference/DMS-ANALYSIS.md) provide context, not proof
that this campaign has run. Historical default-off and hash-admission language
in those snapshots does not override AGENTS.md.

### Completion contract

- Run every unconditional row in Sections 6–10. Resolve every conditional row
  using its stated trigger, with linked evidence for a justified skip.
- Preserve failures, negative findings, and resource costs. A failed quality
  candidate is a completed experiment; broken tooling is not.
- Complete finalist validation if any candidate qualifies. If none does, close
  as **executed: no qualifying improvement**, with all required branches resolved.
- A resource cap, missing dependency, or unfinished implementation means
  **blocked/partial**, not fully executed. Record the exact unblock action.
- Do not retune after final-holdout results, silently expand the search, publish
  a winner, or change the default sidecar. Finish the report and return to the lead.

## 2. Starting evidence and local assets

Historical measurements below are Qwen3.8-27B Q4_K_M on zbook / Radeon 8060S /
`hip_gfx1151`, using a dense BF16-KV teacher. These are historical observations,
not baselines measured on the execution host. Commands and full provenance live
in the linked artifacts.

| Observation | Evidence | Interpretation |
| --- | --- | --- |
| Long-label linear fine-tune: internal accuracy 0.835995; Japanese development max KL 0.140074 | [long-training record](../../benchmarks/results/2026-08-23-gfx1151-qwen38-dms-long-trained-linear-rejected.json) | Every sixteenth row of four training sequences was withheld; not source-held-out task accuracy |
| Exact-budget W256: Japanese max KL 0.141799; effective CR 1.983899 | [rank-budget record](../../benchmarks/results/2026-08-23-gfx1151-qwen38-dms-exact-budget-32k-rejected.json) | Capacity fixed, quality not fixed |
| W8192 32K final: max KL 0.00343035; effective CR 1.599688 | [32K final](../../benchmarks/results/2026-08-23-gfx1151-qwen38-dms-w8192-32k-final-pass.json) | More protection and more retained KV together; not proof that selector quality is irrelevant |
| W8192 128K final: max KL 0.00106191; effective CR 1.882225 | [128K final](../../benchmarks/results/2026-08-24-gfx1151-qwen38-dms-w8192-128k-final-pass.json) | Four prompts with eight teacher-forced steps each, not long free-running task validation |

The existing full-query long-label fine-tune took 583.07 seconds internally,
plus 637.08 seconds of threshold calibration. The roughly 100-second figure is
from short-context training. Historical final-suite walls were 1,262 seconds at
32K and 10,271 seconds at 128K; budget by measured execution-host pilot costs,
not by the short trainer time.

### Reuse map

Use `~/dms-artifacts/` read-only for historical inputs. Create one unique output
root, such as `~/dms-artifacts/selector-improvement-<run-id>/`; never overwrite a
prior run. Store raw captures/logits/checkpoints outside Git.

| Asset relative to `~/dms-artifacts/` | Reuse | Limitation |
| --- | --- | --- |
| `qwen38-external-v1/captures-fp32-768/` | Tiny cross-checks and historical comparison | Not representative long-context training |
| `qwen38-external-v2-long/captures-f16-32k-train4/` | Hidden/Q/K capture, four training sequences, 512 shards | No V, no teacher continuation; Q/K storage is FP16 |
| `qwen38-external-v2-long/labels-cr2-w256-f16-32k-exactq/` | Hidden rows, continuous `future_attention_mass`, binary labels; 64 shards | W256 scores; exact query coverage does not mean unrounded Q/K |
| `qwen38-external-v2-long/long-data-manifest-32k.json` | Reused development validation sequences | Already exposed; not a new final holdout |
| `qwen38-external-v3-final/qualified-w8192/` | Frozen baseline weights and metadata | Preserve original bytes |
| v3 32K and v4 128K manifests/results | Regression evidence and source exclusions | Previously consumed finals, not untouched confirmation |

A planning-time inventory found all 512 long-capture and 64 label shards present
with matching recorded byte sizes. This was not a full checksum audit.

Local model: `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`. Expected historical SHA-256:
`7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`.
Frozen sidecar SHA-256:
`e52fc60a4e8f86f00719be107b44a73567795dfdd88dbbce15cf955409fcd764`.

Verify hashes for experimental comparability. A different model is not globally
unsupported; it requires separately named captures/baselines rather than mixing
provenance. No full BF16 checkpoint or model redownload is required by this plan.
Training from saved hidden rows and recomputing labels from Q/K need no model
inference. Fresh capture and integrated evaluation use the local GGUF.

## 3. Scope, budget, and measurement contract

- One physical execution host and GPU lane, recorded before running. Prefer a
  host that fits both dense BF16-KV teacher and candidate at 128K. Do not silently
  switch codecs or compare absolute performance across hosts to fit a run.
- Model geometry: hidden 5120; 16 compact layers, physical IDs 3 through 63 step
  4; 24 query heads, 4 KV heads, head dimension 256; input stage
  `post_attn_rmsnorm_pre_q_projection`.
- Primary scope: c1, autoregressive, BF16 KV, no MTP, no prefix reuse. Freeze
  execution profile, attention variants, sampler, tokenizer, and arithmetic
  across arms. Record any existing production arithmetic separately from T3.
- Primary objective: improve effective live-KV compression at 32K subject to
  quality gates. Secondary: 128K transfer, smaller-window quality, task accuracy,
  actual memory, and end-to-end latency. Do not conflate these objectives.
- Initial execution ceiling: **120 GPU-hours and 500 GiB of additional disk**.
  Pilot one representative capture/evaluation and estimate the complete matrix.
  Count training, capture, failed attempts, and repeats. If the projection exceeds
  the ceiling, stop for a revised budget; do not quietly reduce samples.
- Maximum two scientific revision attempts per experiment family beyond the
  prescribed matrix, with the hypothesis recorded before each attempt. Mechanical
  correctness repairs do not count as new hypotheses, but consume the same time
  ceiling. No unlisted hyperparameter search. Budget exhaustion is a handoff,
  not evidence against the scientific hypothesis.
- One GPU owner. Do independent CPU work while background GPU work runs; use
  completion notifications rather than polling. Do not kill another owner's jobs.
- No power/clock/firmware changes, base-model fine-tuning, c>N scheduler redesign,
  cross-quant study, full online recovery of evicted keys, or 232K capacity search.

At prefill cut position `T-1`, eligibility is `(T-1-position) > W`; for contiguous
positions the protected count is `P=min(T,W+1)`. The exact live target per head is
`P + ceil((T-P)/CR)`. Use actual runtime positions, not an approximate W formula,
and record appended decode rows separately. CR names eligible-history compression,
not total KV, recurrent-state, allocator, or model-residency compression.

### Frozen run card — fill before candidate evaluation

| Field | Recorded value |
| --- | --- |
| Run ID, coder, start/end dates, state | `selector-improvement-run-20260922-062438`; `dms-selector-campaign`; started 2026-09-22; running |
| Commit, scoped dirty diff, host/GPU/VRAM/backend | `f9451f6856772a5d624961f4dba15ca06c90eddf`; campaign paths initially clean, unrelated IQ-routing edits preserved; `zbook`; AMD Radeon 8060S unified-memory APU / 125 GiB host memory; `hip_gfx1151` |
| Driver, ROCm, compiler, Python, Torch, environment | Linux `6.18.52-1-cachyos-lts`; HIP `7.15.26333`; AMD clang `23.0.0git` (`8f497e0`); Python `3.13.13`; Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`; `uv 0.12.3` |
| Model/tokenizer/capture/label/baseline hashes | model `7e78da5d...e169`; sidecar `e52fc60a...d764`; capture manifest `57faa3a8...db88`; labels `2a47af11...cdfc`; every referenced capture/label shard verified; tokenizer hash pending split seal |
| Profile, resolved variant manifest, strict fallback, codec | T3 policy experiment; c1 autoregressive BF16 KV; resolved variants/fallback pending evaluator freeze |
| Evaluator/scorer/test hashes and approved edit boundary | pending Phase A tooling; candidate edits restricted to named DMS scripts/modules/tests/docs; evaluator freezes before P0-P7 |
| Data manifests, source exclusions, seeds, split identities | historical v1-v4 sources located; new split manifests and normalized-text exclusions pending; training seeds 0/1 |
| Output root, free disk, pilot wall, projected GPU-hours | `~/dms-artifacts/selector-improvement-run-20260922-062438/`; 119 GiB free at start versus 500 GiB campaign ceiling; use bounded streaming capture and verified transient cleanup or return for storage revision; pilot/projection pending |
| Actual cumulative GPU-hours/disk, faults/retries | 0 GPU-hours; 4 KiB preflight record; no faults/retries |

## 4. Evaluation firewall and gates

### Data allocation

Keep all four categories: code, general English, Japanese, mixed Japanese/English.
Never tune against a single known Japanese token. Categories cannot compensate
for one another.

| Split | Definition | Permitted use |
| --- | --- | --- |
| Training base | Existing four v2 32K train sequences | Training and within-training diagnostics |
| Training expansion | 12 fresh 32K sequences, three per category | Diversity comparison; no qualification/final sources |
| Development | Existing v2 validation, plus separately sourced development task instances | All listed candidate selection and diagnosis |
| Qualification | Eight fresh 32K sequences, two per category | Frozen candidates; choose at most two finalists |
| Boundary qualification | Four fresh 49,157-token sequences, one per category | Correlated prefix slices for the pre-final boundary suite; never drawn from final sources |
| Final 32K | Eight fresh 32K sequences, two per category | Finalist and baseline confirmation, no tuning |
| Final 128K | Four fresh 128K sequences, one per category | Transfer confirmation, no tuning |

Generate and hash all split manifests before selecting candidates. Exclude every
v1/v2/v3/v4 source ID/path and new training/development source from qualification
and final sets. Exclude new qualification and boundary-qualification sources
from finals, and keep the boundary sources disjoint from ordinary qualification.
Also record normalized-text hashes to detect duplicated content under different IDs.
Repeated prefixes at different lengths are correlated samples, not new sources.

The existing long-manifest builder emits one train and one validation sequence
per category. Extend or compose it with tested split/source handling to obtain
these counts; do not claim its default output already supplies this matrix.
Keep final contents out of training/diagnostic summaries until finalist freeze.
Failure on a final ends that candidate's campaign; do not tune and reuse that
final as independent evidence.

### Gate definitions

| Gate | Binding condition and action |
| --- | --- |
| G0 — integrity/control | Valid hashes/shapes/maps, finite values, no protected eviction, exact per-head prefill budget/ties, canonical positions, no owner leaks; same-seed repeat decisions/export stable. Compact-no-evict versus dense must have max KL ≤ 0.001 and 100% top-1 on the same evaluated rows. Stop affected experiments on failure and repair or return the numerical-control discrepancy to the lead; do not attribute it to selection. |
| G1 — diagnostic screen | Dense-teacher aligned full-vocabulary KL: max ≤ 0.05 and top-1 ≥ 90% in every category, no nonfinite state. Use all four v2 development prompts × 32 decode steps, plus prefill logits. A pass only advances research; a fail remains a recorded result. |
| G2 — finalist distribution | Qualification/final: mean ≤ 0.001, p95 ≤ 0.005, p99 ≤ 0.02, max ≤ 0.05; top-1 ≥ 99% overall and ≥ 97% per category. Apply KL limits globally and per category/context scope; no averaging away a failed scope. Diagnose every top-1 mismatch and every row above KL 0.02; such rows cannot automatically select a finalist without the task and control results. |
| G3 — task quality | Paired exact-answer tasks: candidate correct count must be at least dense teacher count overall and in each task family and language category, at each tested length. Also report baseline DMS. No substitution of top-1 for task scoring. |
| G4 — route/resources | Intended selection actually runs; observed masks/counts match candidate; no silent fallback; temporary dense prefill released before compact decode; tracked allocations return to baseline after close; same-schedule three-run determinism and public-surface smoke. |
| G5 — comparative result | G0/G2/G3/G4 pass. Record full quality/compression/memory/latency vector versus frozen baseline. Call an operating-point improvement only for higher compression at passing quality; same-compression quality improvement is a separate result. No required speedup and no claim that KV savings equal total-memory savings. |

G2 adopts explicit numerical bars for this campaign; it is not automatic product
promotion or a claim that the initial T1/T2 profile campaign covers DMS policy.
A larger protected window passing does not prove smaller windows will pass.

Qualification: eight prompts × 128 teacher-forced decode rows = 1,024 rows.
Final 32K: eight × 128 = 1,024. Final 128K: four × 256 = 1,024. Score prefill
separately and require it to satisfy the same max-KL and finiteness checks.
Report sample counts by prompt, source, category, context, and step; token rows
are not independent prompts. Record free-running 256-token generations separately,
including first divergence, comparable-prefix length, and fixed-length match rate;
these are diagnostics, not substitutes for G3.

G3 task matrix: exact retrieval, two-hop key/value lookup, and variable-state
tracking; four language categories; two placements (early evictable history and
recent protected region). This gives 24 cases per length at 32K and 128K, with
fixed unique seeds per case and distinct development/qualification/final source
and answer pools. Pad with source-disjoint realistic documents. Keep answer keys
inside the evaluator, not selector inputs. Validate tokenization, expected answers,
answer parsing, and actual dependency positions. Report any cases the dense
teacher also fails; do not delete or replace hard cases after seeing candidates.
Run this full matrix for the baseline and frozen finalist(s), not every training
seed. Include a short four-category task/control smoke below the window.

Freeze evaluator code, thresholds, task generators, counts, seeds, and manifest
hashes before the first candidate. An evaluator bug is repaired in a separate
commit, followed by rerunning affected controls and invalidated comparisons.
Do not amend the evaluator to rescue a failed candidate.

## 5. Phase A — inventory, prerequisites, and baseline

- [x] Record `git status -sb`; preserve unrelated work and coordinate shared-file edits.
- [x] Read the authorities in Section 1 and relevant current worklog entries.
- [x] Verify model, sidecar, capture/label manifest and every shard checksum;
      validate source/model/tokenizer/geometry consistency and contiguous positions.
      Evidence: external `preflight/integrity.json`, SHA-256
      `72f2cf6222d3946e67316dd301e08717644c4e0ef32f79bb95589f231938bb9f`.
- [x] Check ROCm liveness and available resources. Same-host historical 128K
      execution already establishes fit; no new Tier-1 probe was needed for
      preflight. Fresh captures are storage-constrained and follow the run-card
      streaming/cleanup condition.
- [ ] Fill the run card; build and seal split manifests; estimate full costs.
- [ ] Implement missing evaluation support below in separate validated units.
- [ ] Run same-host dense and compact-no-evict controls, then frozen W8192/CR2
      on development. Run controls again when backend/arithmetic changes.
- [ ] Reproduce baseline policy counts, quality, and teardown; record differences
      from historical data without treating cross-host timings as regressions.

### Tooling prerequisites — existing versus required

| Surface | Existing implementation | Required addition before its experiment |
| --- | --- | --- |
| Capture | `hipengine/kvcache/dms_capture.py`, `scripts/qwen38_dms_capture.py`: hidden/Q/K | Versioned optional V and continuation capture, semantic stage/dtype validation |
| Labels | `hipengine/kvcache/dms_labels.py`, `scripts/qwen38_dms_build_labels.py`: mass, tiled GPU, CPU reference | Continuous-score objectives, continuation queries against prefix keys, value-aware CPU oracle/GPU implementation |
| Trainer | `scripts/qwen38_dms_train_sidecar.py`: BCE + budget penalty, BF16 export | Regression/ranking/weighted objectives, correct score direction, held-out-sequence reporting |
| Integrated evaluator | `scripts/qwen38_dms_integrated_quality_suite.py`: only `no_evict,sidecar`, one assembled prompt/category | Multiple independent prompts, diagnostic mask/score injection, prefill gating, full-row capture, task/free-running arms, frozen G1/G2 verdicts |
| Selection | `hipengine/kvcache/dms.py`: exact-budget ranks and borrowed own-Q channel | Seeded-random/recency/diagnostic-oracle adapters without production fixture recognition |
| Candidate metadata | Existing external-linear schema and evidence hashes | Validated derivation tool; no hand-edited hashes or mutation of frozen packages |

Oracle evaluation requires captures for the evaluated sequences. The existing
v2 training labels cannot be transplanted onto v2 validation prompts. Capture
required development sequences once and reuse them with exact token/position maps.

Baseline results:

| Arm | Manifest/command/artifact | Rows | KL mean/p95/p99/max; top-1/category | Live CR; bytes; final allocations | Verdict |
| --- | --- | --- | --- | --- | --- |
| Dense teacher | `pilots/g0-no-evict-1k-d2.json` (teacher trajectory within the G0 pilot; artifact SHA-256 `f3a4a631bbd7aa39c87849dfe6f611fb9b7e71bb9b7e71bad98be27024f661ce87dfb70b`) | 12 scored rows across four categories; 4 prefill rows | Candidate comparison recorded separately; dense teacher is the reference | Teardown baseline restored; 108.47 s total pilot wall | Control pilot reference; not a baseline sweep result |
| Compact no-evict | Same G0 pilot; repeat artifact SHA-256 `e047f1ef4b69b5e8c48a65571dff0e382fd674f38fddd36b3c244cdc1285a58f` | 12 scored rows across four categories; 4 prefill rows | max KL `0.004982890284225346`; top-1 `11/12`; mixed Japanese/English decode step 1 mismatch reproduced | Device payloads present; dense prefill pool released; tracked allocations returned to baseline | **G0 rejected**; numerical-control repair required before P0–P7 |
| Frozen W8192/CR2 | not run | pending | pending | pending | Blocked by compact no-evict G0 control failure |

## 6. Phase B — policy and non-learned controls

Use frozen weights. Run every point below on the G1 development suite, without
training. Predeclare deterministic oldest-position tie handling. Use the same
learned decode rule for the production-like sweep; log prefill versus decode
budget and quality separately because exact-budget prefill does not imply exact
budget enforcement throughout decode.

| ID | Protected window W | Eligible-history CR | Purpose | Result/artifact; G1; effective CR |
| --- | ---: | ---: | --- | --- |
| P0 | 8192 | 2 | Frozen baseline | pending |
| P1 | 256 | 2 | Reproduce difficult low-protection regime | pending |
| P2 | 2048 | 2 | Smaller-window candidate | pending |
| P3 | 4096 | 2 | Intermediate window | pending |
| P4 | 8192 | 4 | Higher historical compression | pending |
| P5 | 4096 | 4 | Combined pressure | pending |
| P6 | 2048 | 4 | Aggressive combined pressure | pending |
| P7 | 8192 | 8 | Bounded stress point, not a promised target | pending |

- [ ] Run P0–P7 and fill all result cells, including failures.
- [ ] At P0, P2, and P4 compare learned rank, most-recent eligible keys, and
      uniform random eligible keys with seeds 0 and 1 at identical per-head budgets.
- [ ] For these attribution controls, freeze prompt masks after the prefill cut
      and retain all newly decoded tokens for 32 steps in every arm. This isolates
      prompt selection; label it diagnostic and do not substitute it for ordinary
      decode-policy validation.
- [ ] Record overlap of retained sets, discarded oracle mass, head/layer error
      concentration, and the actual distance of dependencies from the cut.
- [ ] Choose one training-test operating point by this fixed rule: P2 if P2
      fails G1; otherwise P4 if P4 fails; otherwise P6. Call it `training_point`.
      All later learned candidates also run P0, so gains cannot hide regression.

Control results:

| Point | Selector/seed | Kept-set overlap with learned | KL tails/category; G1 | CR/count equality | Interpretation/artifact |
| --- | --- | --- | --- | --- | --- |
| P0/P2/P4 (expand per arm) | not run; compact no-evict G0 control failed first | pending | pending | P0–P7 are blocked until the no-evict numerical-control discrepancy is repaired | pending |

## 7. Phase C — exact-score and query-information diagnostics

These experiments distinguish approximation of a proxy from the proxy itself.
They are not deployable oracle policies and not mathematical quality ceilings.

- [x] Run the sealed-label continuation-mass diagnostic at W256/CR2 with
      matched recency and seeded-random budgets. The 64 layer shards produced
      mean discarded mass `0.100237` for the continuation oracle,
      `0.676643` for recency, and `0.514188` for random seed 0. This is CPU-only
      label evidence, not integrated quality evidence.
- [ ] Inject the continuation oracle at P1/P2/P4/P6 or run the causal last-query
      control. The existing sealed labels contain no last-query Q/K attention
      capture; the diagnostic records this as blocked rather than substituting
      continuation mass for causal query information.
- [ ] Build a causal query-aware control at `training_point`: rank prefix keys
      by attention mass from the last 64 available prompt query rows, honoring
      causal masks and GQA grouping. No continuation queries in this control.
- [ ] Build an explicitly noncausal diagnostic using the next 128 dense-teacher
      continuation queries against prefix keys. It measures hindsight relevance,
      not information a deployed prefill selector possesses.
- [ ] Compare both query controls with within-prompt mass at the same point;
      include score-computation/capture costs separately from runtime cost.
- [ ] Verify oracle score polarity: larger mass means retain; larger existing
      sidecar logit means evict. Never feed mass directly as eviction logits.

Borrowed-query extraction uses a token's own Q projection; it is not access to
future questions. No borrowed-channel checkpoint conversion is required here.
No decode experiment may restore permanently evicted KV without counting storage
and recomputation; full online reselection/recovery is outside this campaign.

| Point | Learned / within-prompt oracle / causal last-64 Q / continuation oracle | Gate and KL tails | Mask overlap, critical-token diagnosis | Finding/artifact |
| --- | --- | --- | --- | --- |
| Continuation oracle vs recency/random (W256/CR2, 64 sealed label shards) | CPU label diagnostic only; continuation oracle discarded-mass mean `0.100237`, recency `0.676643`, random seed 0 `0.514188`; causal last-query not available | Integrated G0/G1/KL not run because compact no-evict G0 is blocked; no mask-overlap conclusion | Continuation mass is a stronger retained-mass ranking on these labels than simple controls, but this does not establish causal learnability or runtime quality. Artifact: `~/dms-artifacts/selector-improvement-run-20260922-062438/diagnostics/oracle-controls-w256-cr2.json` |

Decision tree:

1. Exact-score oracle passes, learned ranks fail: test learnability/objectives;
   retain label-horizon experiments to see whether the target can improve further.
2. Both fail: fitting this mass proxy more accurately is insufficient at that
   point. Continue continuation/value experiments; do not declare all selectors
   impossible or stop the campaign.
3. Causal query control improves: record evidence for end-of-prefill query-aware
   selection. Hindsight-only improvement is not deployable query evidence.
4. No-evict fails G0/control checks: invalidate policy attribution until repaired.
   It is not evidence of selector failure.

## 8. Phase D — training objective with no recapture

Reuse long hidden states and continuous W256 mass. Preserve these score semantics
for the matched comparison even when evaluated with a larger policy window.
Do not relabel W256 scores as W8192 scores. Recomputing a window changes which
queries contribute; changing CR alone can reuse continuous scores.

All objectives: linear geometry unchanged, same short-run initial sidecar,
20 epochs, batch 512, Adam settings matching the existing trainer, learning rate
0.0001, weight decay 0, max gradient norm 1; seeds 0 and 1. Use the existing v1
short CR2 candidate as initialization, not the long-trained baseline, for an
unbiased retraining comparison. Record exact initialization hash. Existing
within-sequence held-out rows remain training diagnostics only. For objectives
expressed as importance, initialize the output as the negative of the legacy
eviction logit and export back to eviction polarity; record any further affine
normalization and prove that folding it into the linear weights preserves ranks.
Use AdamW as in the current trainer, betas (0.9, 0.95), epsilon 1e-8.

| ID | Objective | Definition frozen before training | Result at P0 / training_point, both seeds |
| --- | --- | --- | --- |
| L0 | Existing BCE + budget | Binary labels, budget coefficient 0.1 | blocked before training: no valid `training_point`; compact no-evict G0 failed with max KL `0.004982890284225346` and top-1 `11/12` |
| L1 | Log-mass regression | MSE of log1p(mass), train-only per-layer/head mean/std; export score direction so high importance is retained | objective implemented/tested; training and integrated evaluation blocked by missing valid control/training point |
| L2 | Pairwise ranking | Mean softplus(score_discard - score_keep) on importance scores; deterministic same-head pairs; omit empty pair sets | objective implemented/tested; training and integrated evaluation blocked by missing valid control/training point |
| L3 | Importance-weighted BCE | Existing BCE + budget; retained-label examples weighted by train-only p95 mass, capped at 4 | objective implemented/tested; training and integrated evaluation blocked by missing valid control/training point |

- [ ] Implement and test each loss, pair sampling, zero-mass cases, score
      direction, train-only normalization, and exported inference equivalence.
- [ ] Run L0–L3 × two seeds. Use identical splits and training examples.
- [ ] Preserve ranking with export transforms folded into weights/bias where
      valid; test BF16 rank changes and tie handling. Do not introduce runtime Torch.
- [ ] Calibrate the new token's decode threshold using training data only for
      every candidate. Increasing/decreasing prefill rank scores does not by itself
      define a usable decode eviction rule. Record threshold calibration separately.
- [ ] Report accuracy/BCE only where meaningful; also report retained mass,
      weighted false-eviction cost, rank correlation, and recall at exact budget.
- [ ] Select one loss by G1-pass count across the two evaluated points/seeds,
      then lowest worst-category max KL, then mean KL, then fixed ID order.
      Do not select a lucky seed; use seed 0 for downstream comparisons.

Record training wall, calibration wall, peak device/disk bytes, parameters,
export hashes, train versus source-held-out metrics, and per-layer/head errors.
An objective that predicts labels better but worsens integrated KL is not a win.

## 9. Phase E — label horizon, V, and data diversity

Run this phase even if Phase D improves the baseline. It tests the proposed
causal explanation rather than assuming it from a Japanese failure.

### E1. Paired label ablation

Capture the same four base training sequences plus 128 dense-teacher continuation
steps, including Q/K/V and the declared attention-output stage. Keep the old
capture immutable. Use W256 and CR2 for all three label builders, so the horizon
and value comparisons do not also change the training budget. Prefix keys alone
are eviction candidates; continuation and protected keys still participate in
the attention softmax denominator.

For query q, dense output o, probability p_i and value v_i, the exact single-key
removal change with other scores/values fixed is
`delta_i = p_i / (1-p_i) * (o-v_i)`. This is not logit KL and not the joint effect
of many evictions. The campaign's value-aware score is the sum over relevant
queries and grouped query heads of `||delta_i||_2`; record the units, head
aggregation and window mask. Use FP64 CPU fixtures. For numerical saturation,
compute 1-p_i from the other softmax probabilities; define and test a bounded
floating-point fallback before running GPU labels. Never silently drop NaNs.

Use the same fresh capture precision for all E1 arms so V is the only difference
in the value ablation. First compare within-prompt mass from the fresh capture
with the historical FP16-Q/K scores and record rounding effects. Tile both key
and query axes; do not materialize a tokens-squared-by-head-dimension tensor.
Measure one shard before committing to the full label pass.

| ID | Label horizon | Importance | Required comparison/result |
| --- | --- | --- | --- |
| H0 | Within-prompt future queries after grace window | Mass | Matched recapture control; pending |
| H1 | Next 128 teacher continuation queries after prefix cut | Mass | H1 vs H0 isolates horizon; pending |
| H2 | Same continuation queries as H1 | Single-key value perturbation norm | H2 vs H1 isolates value criterion; pending |

- [ ] Implement versioned capture/label extensions and independent CPU fixtures.
- [ ] Reuse/reconstruct V only if its numerical stage is verified against direct
      capture on small real-model rows; otherwise recapture the bounded corpus.
- [ ] Test single-key analytic deletion and multi-key non-additivity explicitly;
      test causal masks, GQA grouping, window boundaries, and saturation.
- [ ] Run exact-score replay for H0/H1/H2 at `training_point` on separately
      captured development data. Continuation access is diagnostic only.
- [ ] Train H0/H1/H2 with the Phase-D selected loss, seeds 0 and 1, same inputs,
      initialization, optimizer and budget. For BCE arms derive matching labels;
      for regression/ranking use the continuous importance score.
- [ ] Evaluate P0 and `training_point` with normal deployable hidden-only
      inference; teacher continuation inputs must not reach the deployed selector.
- [ ] Select the label by the Phase-D tie-break rule with H0/H1/H2 ID order.

### E2. Diversity ablation

- [ ] Capture the 12 new training sequences with the selected label procedure.
      Keep all categories balanced; additionally report Japanese error patterns.
- [ ] Compare original four versus expanded sixteen sequences using the selected
      loss/label, seeds 0 and 1. Match total optimizer updates to the four-sequence
      20-epoch run; record sampling weights and exposure per source.
- [ ] Run the expanded-data fit with 20 epochs for both seeds to distinguish
      more optimization from more source diversity. Do not conflate the two.
- [ ] Evaluate P0 and `training_point`; preserve all failures. Select the data/
      update arm using Phase-D's rule across both seeds; fixed tie order is
      four-sequence, sixteen matched-updates, sixteen full-epochs. This defines
      the linear parent for Phase F.

| Arm | Label/loss/data/seeds/update count | Score-oracle quality | Learned quality/categories | Capture/label/train cost; artifact | Conclusion |
| --- | --- | --- | --- | --- | --- |
| H0/H1/H2 (expand) | pending | pending | pending | pending | pending |
| Four vs sixteen, matched updates | pending | not applicable | pending | pending | pending |
| Sixteen, 20 epochs | pending | not applicable | pending | pending | pending |

## 10. Phase F — predictor capacity, last

- [ ] Fit one fixed small multilayer perceptron (MLP) per compact layer: hidden
      5120 → bottleneck 32 → SiLU → 4 outputs, using the selected label/loss/data,
      seeds 0 and 1, matched optimizer updates and the same train-only calibration.
      Use deterministic fresh initialization for the new architecture; record it
      separately from the linear warm start. Output polarity follows the selected
      loss and is converted to eviction scores at export.
- [ ] Fit a fresh-initialized linear control for both seeds under those same
      settings. Compare MLP against both this control and the selected warm-start
      linear parent so architecture and initialization are not conflated.
- [ ] Report parameters, optimizer state, projection latency estimate, fit and
      source-held-out ranking metrics. Do not call it nearly free from FLOPs alone.
- [ ] Run diagnostic full-model mask replay at P0 and `training_point`, including
      per-token decode decisions with the same mathematical model. Host diagnostic
      costs are not shipping-device performance.
- [ ] Integrate a versioned nonlinear source only if both seeds improve worst-
      category max KL versus both linear controls at `training_point`, neither
      introduces a G1 failure at P0, and both seeds pass G1 at least at one common
      evaluated point. Otherwise record **integration
      skipped by capacity gate**, with all measured MLP results retained.
- [ ] If triggered, add an explicit source/schema contract, torch-free inference,
      registered CPU and device paths, compatibility/error tests and kernel
      lineage/correctness/profiler checks. Do not reinterpret linear schema bytes.
      Validate actual exported BF16 outputs and rankings, then rerun G1.

This conditional integration is a finite engineering branch, not permission to
leave a passing diagnostic model mislabeled as a shipped candidate. An integration
blocker leaves that arm diagnostic and must be reported before campaign closure.

| Arm | Parameters/bytes; seeds | Offline ranking | Integrated replay KL/category | Device route status/cost | Gate/artifact |
| --- | --- | --- | --- | --- | --- |
| Best linear parent | pending | pending | pending | pending | pending |
| Fresh-initialized linear control | pending | pending | pending | pending | pending |
| MLP bottleneck 32 | pending | pending | pending | pending | pending |

## 11. Phase G — qualification and untouched confirmation

- [ ] Assemble all G1-passing deployable candidates. Run qualification G2 on
      at most four: the best policy-only point, best objective-only arm, best
      label/data linear arm, and MLP if integrated. Deduplicate identical artifacts.
      Select each family representative by Phase-D's rule, with compression as
      the first tie-break between different operating points that both pass G1.
- [ ] Run the baseline on identical qualification rows. Choose at most two
      finalists: highest effective CR passing G2, then the best-quality distinct
      candidate at no worse CR than baseline. Break ties by worst-category max
      KL, mean KL, then candidate ID. G2-passing is provisional until G3/G4.
- [ ] For finalists and baseline run G3 qualification task matrix, resource
      checks, and nearby/boundary contexts. Freeze surviving finalists once.
- [ ] Boundary suite: for each finalist W, test W-1/W/W+1/W+2 and unrelated
      contexts 12,345 and 49,157, all four categories × 32 teacher-forced steps,
      plus a 128-token no-eviction control (below every listed W). Take one
      prefix at each length from each of the four sealed boundary-qualification
      sequences; report these as four independent sources, not independent
      samples for every length. Apply G0 and the G2 numerical limits separately
      at each length/category; with 32 decode rows, the 97% category bar permits
      no mismatch. This conservative boundary check is not a rate estimate.
      Score prefill separately. Failure disqualifies the candidate from final
      confirmation; controls failing G0 require repair rather than rejection
      of the scientific hypothesis. No-eviction checks are not compression evidence.
- [ ] Exercise at least 2W+32 decode steps on the first canonical v2 development
      code prompt for each finalist and baseline to cross the protected-window
      expiry boundary. Chunk teacher forcing and check counts/state every window
      boundary. Apply G0 and G2 over the full trajectory and separately before/
      after the first expiry; record 128-step quality/count summaries throughout.
      Failure disqualifies the candidate from final confirmation. Run this before
      final freeze; eight-step success does not test expiry.
- [ ] Freeze artifact hashes, policy, thresholds, selected variants and commands.
      Run final 32K/128K G2, final G3, and free-running diagnostics with no tuning.
- [ ] Run three same-schedule repeats on one qualification prompt per category
      and one expiry-boundary case; require identical candidate decisions and
      outputs under the frozen schedule, not identity to dense free-running IDs.
- [ ] Exercise each surviving finalist through `hipengine.LLM.generate()` or
      `hipengine serve` with actual sidecar selection and eviction beyond W.
      Record selected source/variant, window, CR, mask/count digest and fallback
      reason. A resident-session-only result is diagnostic, not public-route proof.
- [ ] Measure dense/baseline/finalist memory and complete request timings using
      same-host counterbalanced order, three repeats, warmup/JIT outside timing.
      Record prefill peak and post-pack residency separately, tracked and sampled
      memory, prediction/rank/pack overhead, prefill wall, first-token latency,
      decode latency and complete request wall. Use the same work/output counts.
- [ ] If no candidate passes, retain baseline and complete negative-result closure;
      do not quietly loosen quality bars or omit a failing category.

| Candidate/hash | W/CR; actual CR at 32K/128K | G2 by length/category | G3 task scores vs dense/baseline | G4 public route/expiry/repeats | Memory/latency artifact | Final verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | pending | pending | pending | pending | pending | pending |
| Finalist 1 | pending | pending | pending | pending | pending | pending |
| Finalist 2 or justified absence | pending | pending | pending | pending | pending | pending |

## 12. Commands and implementation checks

These examples use existing interfaces. They do not pretend the new objective,
oracle, multi-prompt, or V options already exist. Add each new interface with
parser tests and a documented exact command before freezing the evaluator.

```bash
export MODEL=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
export ARTIFACTS="$HOME/dms-artifacts"
export BASELINE="$ARTIFACTS/qwen38-external-v3-final/qualified-w8192"
export LABELS="$ARTIFACTS/qwen38-external-v2-long/labels-cr2-w256-f16-32k-exactq"
# Choose a unique run ID and the actual host backend before executing.
export RUN="$ARTIFACTS/selector-improvement-<run-id>"
export BACKEND=hip_gfx1100
mkdir -p "$RUN"
sha256sum "$MODEL" "$BASELINE/qwen38-27b-q4km-dms-sidecar.safetensors"

HIPENGINE_HIP_ARCH="${BACKEND#hip_}" uv run python \
  scripts/qwen38_dms_integrated_quality_suite.py \
  --model "$MODEL" --metadata "$BASELINE/dms_metadata.json" \
  --data-manifest "$ARTIFACTS/qwen38-external-v2-long/long-data-manifest-32k.json" \
  --prompt-tokens 32768 --prompt-split validation \
  --categories code,general_en,general_ja,mixed_ja_en --decode-steps 32 \
  --modes no_evict,sidecar --codec bf16 --backend "$BACKEND" \
  --max-kl 0.05 --min-top1 0.9 --output "$RUN/baseline-development.json" \
  --fail-on-fail

# Existing BCE control. The four-sequence label artifact has no source-held-out
# validation split; derived validation is internal diagnostics only.
python3 scripts/qwen38_dms_train_sidecar.py \
  --labels "$LABELS" --output-dir "$RUN/l0-seed0" --device cuda \
  --epochs 20 --batch-size 512 --learning-rate 0.0001 \
  --budget-weight 0.1 --weight-decay 0 --max-grad-norm 1 --seed 0 \
  --derive-validation-modulus 16 \
  --initial-sidecar "$ARTIFACTS/qwen38-external-v1/sidecar-cr2-qualified-candidate/qwen38-27b-q4km-dms-sidecar.safetensors"
```

Select a Python environment with compatible Torch/ROCm for training; do not load
two ROCm stacks in one process. The code paths used by generation stay torch-free.
The baseline command alone does not implement prefill gating, tasks, multiple
prompts, or the G2 gate; complete Phase A prerequisites before claiming those.
Record all actual commands and return codes, including failing quality runs.

Focused starting tests (extend with new tests under the repository tier naming):

```bash
uv run pytest -q tests/test_unit_dms_capture.py \
  tests/test_unit_dms_sidecar_metadata.py tests/test_unit_dms_sidecar_replay.py \
  tests/test_unit_kvcache_dms.py tests/test_unit_qwen38_dms_quality.py \
  tests/test_unit_qwen38_dms_capture_script.py
```

For training/label math, use `tests/test_gpu_dms_labels.py` and
`tests/test_gpu_dms_sidecar_training.py` in a compatible Torch process; add pure
CPU analytic tests independently. For changed device inference, use
`tests/test_gpu_dms_external_linear_hip.py`,
`tests/test_gpu_dms_external_runtime.py`, and affected compact pipeline tests.
Every GPU test needs the repository HIP/Torch availability guards.

Follow RED/GREEN and the applicable CPU deterministic bundle in TESTING.md.
Kernel changes additionally require lineage, independent numerical checks and
actual profiler identity. Audit surviving temporary paths and run
`python3 audit/audit.py check` for cleanup/flag/ledger work. Do not weaken audit
budgets. At campaign implementation milestone closure run the repository's full
`uv run pytest --suite all -v` once, applying its focused-repair rule if needed.
No such GPU/full-suite run is required for authoring this plan alone.

## 13. Inline experiment record and final decision packet

Expand the phase tables above as runs complete. Each experiment also needs this
compact record here; link large logs/arrays externally and compact JSON evidence
under `benchmarks/results/`. Record rejected attempts as well as successes.

### Experiment record template — copy for each logical comparison

- **ID/state:** pending / running / passed / failed-quality / failed-control /
  skipped-by-declared-branch / blocked. State exact reason, never just “unqualified.”
- **Hypothesis and parent:** expected observable; what is held constant; one
  changed factor. Include whether oracle information is causally available.
- **Identity:** commit/scoped diff, host/hardware/software, model/tokenizer,
  sidecar/metadata, capture/labels/splits, evaluator/profile/variant hashes.
- **Command and cost:** exact argv/environment, start/end, return code, warmups,
  repeat/seed, capture/label/train/calibration/evaluation times, peak disk/device.
- **Quality:** row/prompt/source counts; prefill and decode KL mean/p95/p99/max;
  top-1 overall/per category/head diagnostics; every mismatch's prompt/step,
  winners, margin, top-k overlap; free-running and task results where applicable.
- **Selection/resources:** configured versus actual budgets; per-head live-count
  range, protected violations, retained-set overlap, weighted discarded score,
  decode drift, payload/metadata/workspace/recurrent-state/total bytes, prefill
  peak, sampled memory, teardown and selected/fallback route.
- **Decision:** gate verdict and reason, alternate explanations, next prescribed
  branch, limitations, artifact path/hash and immutable worklog link.

### Completion punchlist

- [ ] Every phase checkbox is checked with evidence or an explicit branch skip;
      blocked work remains visibly unchecked and prevents “fully executed.”
- [ ] All listed policy, oracle, query-information, loss, label, diversity and
      capacity experiments have results, including negative results.
- [ ] Answer separately: does learned ranking beat simple controls; is mass
      oracle imitation limiting; does continuation horizon help; does V help;
      does extra data help at equal updates; does nonlinear capacity help?
- [ ] State whether improvement survives fresh sources, Japanese/mixed categories,
      window expiry, unrelated lengths, 128K, and public-route execution.
- [ ] Identify any hindsight-only win, unavailable runtime input, unimplemented
      device route, or unmeasured cost. Do not promote a diagnostic replay result.
- [ ] Clean rejected implementation debris; preserve reusable tested tooling and
      external evidence. Inventory surviving temporary paths in REFACTOR.md.
- [ ] Update this document inline and add immutable worklog entries for substantial
      units. Update PLAN.md only if architectural/phase plans change.
- [ ] Run docs/index/worklog gates and applicable implementation tests; explicitly
      stage scoped files and commit each validated logical unit immediately.
- [ ] For retained public performance claims, update benchmark artifacts/scoreboard/
      changelog under OPTIMIZATION.md. Do not export provisional campaign claims
      into the root README or silently change product defaults.
- [ ] Set final campaign state, record unresolved questions and return to the lead.

### Final synthesis — fill after execution

| Question | Answer and evidence |
| --- | --- |
| Best passing operating point versus frozen baseline | pending |
| Best quality at equal live budget | pending |
| Policy-only versus training contribution | pending |
| Label horizon versus value information contribution | pending |
| Query information versus model capacity contribution | pending |
| Data diversity versus optimization-step contribution | pending |
| Generalization and longest tested decode/expiry coverage | pending |
| Real memory/latency trade-off and implementation cost | pending |
| Negative results, blockers, and precise reopen conditions | pending |
| Total cost and reproducibility packet | pending |
| Recommended next product/research decision, not enacted | pending |

The useful outcomes include “existing ranks plus a policy change are sufficient,”
“the target is wrong,” “the target is useful but hard to predict causally,” and
“no improvement within this budget.” None can be inferred from a low maximum KL
at one protected-window setting alone.
