# Qwen3.6-35B ZBook Production-Numerics Tuning

Status: **ready to execute PN1 control-capture; strict/production registration landed**

Owner lane: physical host `zbook`, HP ZBook Ultra G1a, Radeon 8060S / `gfx1151`

Model lane: Qwen3.6-35B-A3B GGUF `UD-Q4_K_M`, BF16 KV, greedy autoregressive
inference

This is the active **PLAN and PUNCHLIST** for the next ZBook tuning cycle. It is
not a benchmark result and it does not authorize a public profile/default
change by itself.

PN1 registration status: strict/production `RuntimeProfilePlan`s are registered
for `(qwen3_5_moe_gguf, hip_gfx1151, gguf_q4_k_m)` with resolution, strict
fallback, duplicate/missing, batch-invariant fail-closed, and cold-path policy
binder tests green; batch_invariant remains unregistered (fail-closed to
strict) until the composition-metamorphic gate lands. The standardized
actual-control capture, expected-control fixtures, control RED fixtures, and
the no-change GPU smoke remain open.

Normative authorities, in precedence order:

1. [`PLAN.md`](PLAN.md) — architecture and plugin invariants.
2. [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — strict, production, and
   batch-invariant numerical contracts.
3. [`PRODUCTION-NUMERICS-CAMPAIGN.md`](PRODUCTION-NUMERICS-CAMPAIGN.md) — global
   calibration, evidence, and campaign policy.
4. [`TESTING.md`](TESTING.md), [`BENCHMARK.md`](BENCHMARK.md), and
   [`CONCURRENCY.md`](CONCURRENCY.md) — test, performance, and serving gates.
5. This document — the frozen ZBook workload, phase order, commands, and
   candidate punchlist.

The broader artifact/quant/runtime comparison remains in
[`QWEN36-35B-ZBOOK-GFX1151-CAMPAIGN.md`](QWEN36-35B-ZBOOK-GFX1151-CAMPAIGN.md).
That document supplies model provenance and cross-runtime context; this one owns
new same-quant hipEngine production-numerics tuning.

---

## 1. Objective and non-goals

### Objective

Improve complete-request true-AR and production SLO-goodput on this ZBook by
changing one measured implementation mechanism at a time while preserving:

- exact request, token, slot, position, mask, KV metadata, state ownership,
  route, graph, transaction, and lifecycle semantics;
- deterministic repeats for an identical resolved manifest and schedule;
- the calibrated same-quant production numerical envelope;
- a registered strict fallback for every production variant; and
- the complete prompt/category, dynamic-width, lifecycle, and server gates.

The campaign may retain a small exact cycle-wall/kernel/launch-count win in the
package path. Switching the public default to named `production` is a separate,
harder decision: the full server packet must pass and SLO-goodput must improve
at least 3% with no more than 1% c1 regression.

### Non-goals

- Do not compare absolute rates with the Framework Desktop or another gfx1151
  host.
- Do not retune model weights, quant format, KV dtype, sampling, MTP acceptance,
  or speculative policy in this AR lane.
- Do not reopen rejected DP4A, unsafe-math, broad-c2-rowtile, MoE-graph, or
  row-compact-GEMV routes without a materially new mechanism.
- Do not optimize one prompt, token ID, candidate ID, or benchmark fixture.
- Do not alter firmware memory carve-outs, TTM limits, power limits, IOMMU,
  clocks, fan policy, or thermal policy.
- Do not port a foreign runtime or scheduler wholesale. External repositories
  are read-only references; retained code lands in-tree behind the four-axis
  registry and a strict fallback.
- MTP is PN7, an independent speculative/economics lane. Its numbers never form
  an AR candidate denominator.

---

## 2. Frozen campaign identity

A retained attempt must match this table or be labeled a new campaign revision.

| Field | Frozen value |
| --- | --- |
| Host | `zbook`, HP ZBook Ultra G1a |
| CPU/APU | AMD Ryzen AI MAX+ PRO 395 |
| GPU | Radeon 8060S, 40 CUs, `gfx1151` |
| Power | STAPM 60 W / fast PPT 60 W / slow PPT 45 W, AC power |
| Scheduler policy | amdgpu `sched_policy=0`; one HIP hardware queue |
| Model | `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Model size / SHA-256 | `22,663,387,424` bytes / `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` |
| Sampled model SHA-256 | `936659d614707776d8e6ca1fb8595991159e78361bff2e3a3616aa91564c89fb` |
| Backend | `hip_gfx1151` |
| Quant / KV | `gguf_q4_k_m` / BF16 KV |
| Sampling | greedy top-1, reasoning off, EOS ignored where the benchmark fixes output length |
| Main prompt fixture | `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, SHA-256 `fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a` |
| Heldout fixture | `benchmarks/prompts/gdn-prefill-category-heldouts.jsonl`, SHA-256 `8d86154fdb52adf5085dbf278c4097dce78d9510d1a6d04ade29a327560e5280` |
| Categories | `code`, `general_en`, `general_ja`, `mixed_ja_en`; all train and heldout rows |
| Production GDN route | `chain_lds32_direct_nonvolatile` |
| Fixed decode timing shape | p512/d128 unless the candidate is explicitly long-context or prefill-only |
| Long-context transfer shapes | p4K/d128, p32K/d128, p64K/d128 when attention/context behavior is affected |
| Dynamic physical widths | c1/c2/c4/c8 plus 8→4→2→1 retirement and sparse physical-c8 |
| Compiler/profile rule | JIT outside profiler; compiler-version file plus `--require-cached-build` inside measured/profiled children |
| Raw output root | `/tmp/hipengine-zbook-production-numerics/<run-tag>/` |
| Compact evidence | `benchmarks/results/<date>-zbook-qwen36-<candidate>-<verdict>.json` |

Every retained block records the current commit, dirty state, full command,
compiler/runtime versions, model fingerprint, prompt hashes, `ryzenadj -i`
before/after, temperature/memory samples, and raw-artifact SHA-256 values.

### 2.1 Idle and thermal gate

Before each retained block:

- use a clean detached commit or a clean main worktree;
- stop unrelated model servers, profilers, builds, downloads, and GPU jobs;
- verify `/dev/kfd` has no unrelated owner;
- wait for CPU/GPU utilization and temperature to return to the declared idle
  band;
- record AC/power state and 60/60/45 W limits;
- prebuild every required JIT object outside timing and `rocprofv3`;
- use one model-owning process at a time; and
- counterbalance A/B order in one resident process where the adapter supports
  it.

Do not discard a slow sample after seeing it. A sample may be excluded only for
a predeclared mechanical invalidation (process error, power-state change,
thermal-limit transition, profiler/JIT contamination, wrong kernel route, or
incorrect output), and the rejected sample plus reason stays in the artifact.

---

## 3. Current baseline and closed work

The starting decision is
[`2026-08-16-zbook-qwen36-production-profile-cn-blocked.json`](../benchmarks/results/2026-08-16-zbook-qwen36-production-profile-cn-blocked.json).
It is evidence to reproduce, not an old→new denominator after source changes.

| Surface | Current result / disposition |
| --- | --- |
| c1 cooperative/persistent F32 router | Retained package route; 450/450 full-logit rows exact; natural pN/d128 `30.438 → 33.219 tok/s` (`+9.136%`, 18/18 wins) |
| Q8T16 c2 | Direct owner; bundled candidate is neutral (`1.00052x` paired median) |
| Q8T16 c4 | Package rowtile floor retained; `1.00475x`, 6/7 wins |
| Q8T16 c8 | Rowtile single/triple plus existing col8 pair retained; `1.01006x` paired median, 6/7 wins |
| Combined numerical matrix | 1,050/1,050 exact full-logit rows; KL mean/p95/p99/max `0`; top-1 100%; three repeats |
| Lifecycle | c8 cancel/re-admit/compact/drain passes exact ownership and same-run state/KV preservation; c1-vs-cN arithmetic bytes remain diagnostic |
| Server | Eight workloads pass; 60-second soak fails with 87 completed and 33 overloaded/rejected of 120 |
| Public profile default | Unchanged; named profile manifest/control/task/BF16 evidence and passing soak remain open |

Corrected c1 stage attribution at the prior checkpoint was:

1. `decode_linear_attn_qkv_gate`: about `6.022 ms/token`;
2. selected-expert MoE: about `5.949 ms/token`;
3. full-attention core/gate: about `2.736 ms/token`;
4. MoE combine: about `2.597 ms/token`;
5. full-attention QKV/norm/RoPE: about `2.311 ms/token`;
6. linear-attention SSM output: about `2.085 ms/token`; and
7. LM-head/sample: about `2.014 ms/token`.

These values nominate profiling questions only. PN2 must reproduce ownership on
the campaign commit before choosing a candidate.

### 3.1 Do-not-repeat ledger

Do not spend a tuning iteration on these unchanged mechanisms:

- cooperative/persistent c1 router — already retained;
- package-floor c4/c8 Q8T16 rowtiling — already retained;
- forced all-width/c2 Q8T16 rowtiling — rejected;
- Q8T16 64-thread verifier pair launch — slower than 128 threads;
- selected-MoE DP4A/Q8_1 routes that failed operation-complete quality or wall;
- one-plane Q8_1 activation — operation-complete SiLU KL failure;
- selective unsafe math — 7.67% slower at the actual leaf;
- c1 MoE graph — exact but about 0.84% complete-wall regression;
- row-compact selected-MoE GEMV — large verifier regression; and
- prompt/token/candidate-specific routing — prohibited benchmark gaming.

Consult [`REFACTOR.md`](REFACTOR.md) before reusing any diagnostic flag.

---

## 4. Binding numerical and semantic contract

### 4.1 Exact in every profile

A mismatch in any item below stops the candidate immediately:

- request ID ↔ scheduler slot ↔ physical row ↔ response mapping;
- prompt slice, current/generated token ownership, finish reason, and accounting;
- token/context/RoPE positions and causal visibility;
- active, causal, finish, sparse, eviction, rollback, and compaction masks;
- `KVLiveSpans`, block/page owner, live count, base offset, token position,
  append/commit/rollback/eviction metadata;
- Conv/GDN/recurrent/KV allocation ownership through admission, cancellation,
  retirement, compaction, and reuse;
- selected expert and row scatter/gather ownership (the numerical expert choice
  may differ only as downstream bounded arithmetic, never by cross-row data);
- graph bucket, variant manifest, fallback route, publication order, and stream;
- per-request RNG/sampler accounting when applicable; and
- teardown, reclaim, stale-pointer protection, final allocator state, and memory
  recovery.

State/KV **values** may differ between production and strict because of T1/T2
arithmetic. Same-run preservation through ownership/compaction remains exact;
value drift must be finite, in range, deterministic, and bound to the complete
strict-teacher distribution/task gate.

### 4.2 Production numerical envelope

At identical strict-teacher contexts, all limits bind globally and per declared
category/shape/transition scope:

| Metric | Automatic-admission requirement |
| --- | ---: |
| Mean KL, production vs strict | `<= 1e-3` |
| p95 row KL | `<= 5e-3` |
| p99 row KL | `<= 2e-2` |
| Maximum row KL | `<= 5e-2` |
| Overall top-1 agreement | `>= 99%` |
| Every category/shape/transition top-1 | `>= 97%` |
| Candidate repeats | at least 3 bit-stable fixed-manifest/schedule runs |
| Finiteness | every captured row/state/logit boundary finite |

A row above KL `2e-2` or any top-1 mismatch is emitted with prompt, step,
strict/candidate winners, strict margin, top-k overlap, teacher NLL/delta-p, and
maximum absolute logit delta. Such rows require explicit review and cannot be
silently averaged away.

The broad kernel floor (maximum KL `<=0.05`, top-1 `>=90%` versus the CPU or
parent oracle) is necessary for a new kernel but insufficient for whole-model
production admission.

### 4.3 Generated IDs, task quality, and BF16

- Strict and batch-invariant candidates bind generated-ID equality according to
  their declared fixture.
- Production cross-width IDs are diagnostic. Same-manifest/same-width repeats
  must be deterministic and neighbor substitution must not contaminate a row.
- The initial automatic ZBook lane requires all 18 greedy task-prompt outputs to
  match strict exactly. This is a stronger campaign-local shortcut, not a claim
  that production universally requires composition-invariant IDs.
- If a candidate changes any task output, stop before performance. Add and
  approve a blinded paired task artifact first: code rows must compile and pass
  frozen hidden tests; English/Japanese/mixed rows must pass a frozen rubric and
  language/format checks with no per-prompt regression. No category may
  compensate for another.
- If strict and candidate full logits are byte-identical, implementation drift
  consumes zero additional BF16-relative budget; attach the existing strict
  BF16/quant result plus the zero delta.
- If logits differ, regenerate aligned BF16 logits for the exact teacher rows
  and pass the predeclared paired BF16 non-inferiority margins through
  `execution_profile_gate.py`. Missing BF16 data cannot qualify a drifted route.

### 4.4 Performance and default rules

- Quality and controls run before final performance.
- Use at least seven counterbalanced A/B pairs for a retained fixed-shape wall
  claim. Natural-suite c1 uses all 18 prompts and alternates route order.
- Report all samples, median, paired ratios, wins, warmups, timing boundary,
  generated hashes, memory, and teardown.
- A small exact non-regressive kernel/cycle/launch or complete-wall win may be
  retained in its scoped package route with rollback evidence.
- A named/public `production` default additionally requires the complete server
  packet, at least 3% SLO-goodput improvement versus both strict and incumbent,
  and no more than 1% c1 regression.
- The frozen 60-second soak is not weakened after a failure. Changing offered
  load, queue limits, SLOs, prompt mix, or rejection policy creates a new
  protocol revision; it cannot rescue the current candidate.

---

## 5. PLAN

### PN0 — Start a clean campaign unit

Deliverables:

1. Create a unique immutable worklog entry for the campaign start.
2. Record clean commit, model/prompt hashes, host/power/thermal/memory identity,
   HIP/compiler versions, and current package route manifest.
3. Verify ROCm and cached-build tooling.
4. Run `scripts/check_lineage.py --kind kernel --diff stat` before any port or
   kernel-body change; inspect every relevant drift/evidence hit.
5. Create one run root and compiler-version file. Never compile inside
   `rocprofv3`.

Exit: identities match Section 2, worktree is clean, no unrelated process owns
the GPU, and all required scripts parse `--help`.

### PN1 — Instantiate named profiles and standardized controls

This is the first implementation unit. Do not tune arithmetic before it lands.

Deliverables:

1. Register Qwen3.6 GGUF `hip_gfx1151/gguf_q4_k_m` cold-path plans with
   `register_runtime_profile_plan()`:
   - `strict`: separate router chain, the registered strict Q8T16 selections
     (including the pre-existing c8 pair-col8 owner where it is the declared
     strict route), strict GDN, and all registered unfused fallbacks;
   - `production`: incumbent cooperative router, direct c2, package-floor c4/c8
     rowtile, and strict fallback keys for every scope;
   - `batch_invariant`: strict selections only after the composition-metamorphic
     gate passes; otherwise leave the plan unsupported and fail closed clearly.
2. Resolve once at model/session construction. No hot-path `if profile`,
   backend, or quant branch is allowed.
3. Emit the standardized actual-control capture for every teacher row:
   request/slot/row map; tokens/positions/lengths; masks; `KVLiveSpans` and
   transaction fields; state/KV owner hashes and finite summaries; selected
   expert/scatter maps; graph bucket/route/stream/fallback; lifecycle counters;
   profile/schema and selected/strict manifest hashes.
4. Build strict and production expected-control fixtures independently; never
   derive expected controls from the candidate capture being checked.
5. Add exact resolution/fallback/duplicate/missing tests and RED fixtures for a
   swapped row, wrong position, wrong mask, wrong KV owner, wrong expert
   scatter, stale graph bucket, and lifecycle leak.
6. Run a no-change strict/production/batch-invariant GPU smoke through
   `scripts/execution_profile_gate.py`.

Exit: explicit `HIPENGINE_EXECUTION_PROFILE=strict|production|batch_invariant`
resolves a clean immutable manifest, all control negatives fail as intended,
and the no-change production route reproduces the retained package quality.
Omitted profile behavior remains unchanged.

### PN2 — Refresh clean baselines on the profile commit

Run, in order:

1. corrected c1 device-stage/rocprof ownership profile;
2. full c1 route numerical gate and natural/fixed performance controls;
3. full bundled c1/cN numerical matrix;
4. c8 lifecycle/control gate;
5. c2/c4/c8 seven-pair graph wall controls; and
6. focused then complete production server packet.

The server baseline is expected to reproduce the current soak blocker. A
different result must be explained mechanically before candidate work.

Exit: one compact baseline artifact binds all raw hashes, the current manifest,
stage ownership, complete quality/control state, fixed/natural wall, and server
SLOs. It records soak occupancy, width exposure, queue/rejection timing, and the
throughput increase needed to clear the frozen offered load. The profile
accounts for at least 90% of complete decode wall or records a measured residual
explanation.

### PN3 — Select exactly one measured mechanism

Choose from the fresh PN2 profile, not historical intuition. Initial questions,
if the ranking reproduces, are:

1. linear-attention QKV/gate projection family;
2. selected-expert MoE gate/up/down and weight reuse at actual c1/c4/c8 rows;
3. full-attention core/gate at 4K/32K/64K contexts;
4. MoE combine/residual boundary;
5. full-attention QKV/head-norm/RoPE;
6. linear-attention SSM output; then
7. LM-head/sampler movement.

For the chosen candidate, predeclare:

- candidate ID and arithmetic class (`T0`, `T1`, or `T2`; T3 leaves this lane);
- exact layer/role/shape scope and affected stateful surfaces;
- source/lineage and why the mechanism should help gfx1151;
- operation-complete boundary, expected kernel/launch change, and measured
  complete-wall ceiling;
- strict fallback and rollback/removal trigger;
- whether downstream expert/top-1 decisions may change; and
- binding quality, task, dynamic, long-context, and performance rows.

External ports require a measured >=1% complete-request ceiling. In-tree exact
micro-optimizations may proceed below 1% only when the profile shows a real
cycle/launch/transfer ceiling; leaf-only wins with no plausible complete-wall
or structural benefit are logged and skipped.

Exit: one candidate declaration is committed with a RED fixture before its
implementation begins.

### PN4 — Candidate RED/GREEN and operation-complete gate

Run this sequence without skipping forward:

1. RED CPU/parent fixture fails for the intended bug/math boundary.
2. Implement one variant. Kernel candidates live in `kernels/hip_gfx1151/`,
   take raw pointers, and use the four-axis registry; scheduling/runtime
   candidates stay cold-path/profile-resolved and add no backend/quant/profile
   hot-path branch. Preserve the strict/unfused path.
3. Pass leaf bit/tolerance, edge-shape, finite/sentinel, registry, and fallback
   tests.
4. Capture a cached `rocprofv3 --kernel-trace` smoke showing the expected kernel,
   dimensions, resources, and plausible duration.
5. Pass the complete producer→consumer operation, not only the leaf.
6. Run a small screen only to catch gross failures; label it diagnostic.
7. Run the full 18-prompt/three-repeat numerical and exact-control gate.
8. Run isolation, c1/c2/c4/c8, retirement, sparse mask, cancellation,
   compaction, graph/eager, and long-context rows applicable to the candidate.
9. Pass task/BF16 rules in Section 4.3.

Any binding failure stops tuning. Preserve the failure fixture, revert/remove
the candidate implementation, write the rejected artifact/worklog, and move to
the next declared mechanism only after committing the clean rejection unit.

### PN5 — Counterbalanced performance and production SLO

Only a PN4-green candidate reaches PN5.

1. Measure the smallest operation-complete boundary with at least 15 paired
   samples where practical.
2. Measure p512/d128 complete graph/eager wall with seven counterbalanced pairs.
3. Measure the 18-prompt natural c1 suite when c1 is affected.
4. Measure c2/c4/c8 and dynamic/sparse rows when cN is affected.
5. Measure p4K/p32K/p64K when attention/context behavior is affected.
6. Re-profile the retained candidate and reconcile device stages plus host
   residual to within 10% of complete wall.
7. Run the focused server screen.
8. Run the unmodified complete canonical server packet, including the
   60-second soak.

Exit decisions:

- **Retain scoped package win:** all correctness gates pass and the exact scoped
  wall/cycle/launch result is positive and same-suite non-regressive.
- **Promote named production variant:** above plus complete profile manifest,
  task/BF16 evidence, and all production workloads pass.
- **Change public default:** above plus >=3% SLO-goodput and <=1% c1 regression.
- **Reject:** correctness fails, complete wall regresses, wrong kernel runs, or
  the supposed mechanism has no reconciled effect.
- **Blocked:** infrastructure/hardware/protocol prevents a valid decision; do
  not relabel it neutral or retained.

### PN6 — Evidence and cleanup per candidate

In the same atomic logical unit:

- write the compact accepted/rejected/blocked JSON and raw hashes;
- update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and date;
- update this punchlist and the immutable worklog entry;
- update `docs/KERNELS.md`/lineage if kernel ownership changed;
- add/update `docs/REFACTOR.md` for every flag, duplicate path, or rollback;
- remove transient code and flags for rejected candidates;
- run focused tests and the applicable milestone suite; and
- explicitly stage, inspect, and commit only the unit's files.

### PN7 — Separate exact MTP follow-up

MTP begins only after the AR candidate/default decision is recorded. Use the
same ZBook and exact MTP-bearing GGUF artifact, but a separate run tag and
artifact family.

- Re-run true no-MTP AR plus adaptive B1/B2/B3 and matched fixed controls on the
  full ten-prompt train/heldout/category suite.
- Use complete proposal + target verify + accept/commit + correction + scheduler
  wall.
- Require exact same-run AR output streams and GPU/CPU transaction agreement.
- Use clean repeated B2/B3 brackets before changing the cap.
- Do not compare with the Framework Desktop `80.10 tok/s` row or use it as a
  denominator.
- Keep speculative acceptance/economics and AR production-profile promotion as
  separate decisions.

### PN8 — Campaign closure

Close only when every attempted candidate has a durable verdict and no transient
runtime path is left untracked. Record either:

- at least one retained production-numerics win plus its default disposition; or
- a measured no-win/blocker conclusion and the next dominant owner.

---

## 6. Command book

These commands are the frozen starting protocol. A candidate may add explicit
selectors only through its declared manifest/adapter; it may not silently edit
the baseline arguments.

### 6.1 Run-root and preflight

```bash
set -euo pipefail
export PYTHONPATH=.
export HIPENGINE_HIP_ARCH=gfx1151
export GPU_MAX_HW_QUEUES=1
export HIPENGINE_GGUF_DECODE_REPACK=1
export HIPENGINE_GGUF_GDN_PREFILL_MODE=chain_lds32_direct_nonvolatile

PY=/home/lhl/hipEngine/.venv/bin/python
MODEL=/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PROMPTS=benchmarks/prompts/mtpbench-code-general-ja.jsonl
HELDOUTS=benchmarks/prompts/gdn-prefill-category-heldouts.jsonl
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short=12 HEAD)"
RUN_ROOT="/tmp/hipengine-zbook-production-numerics/${RUN_TAG}"
mkdir -p "${RUN_ROOT}"

# Clean source and immutable identities.
test -z "$(git status --porcelain)"
test "$(hostname)" = zbook
sha256sum "${MODEL}" "${PROMPTS}" "${HELDOUTS}" | tee "${RUN_ROOT}/sha256.txt"
git rev-parse HEAD | tee "${RUN_ROOT}/commit.txt"

# ROCm/hardware/power snapshot. These are read-only.
${PY} -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx' | tee "${RUN_ROOT}/rocminfo.txt"
rocm-smi --showproductname --showuse --showmemuse --showtemp \
  | tee "${RUN_ROOT}/rocm-smi-before.txt"
sudo ryzenadj -i | tee "${RUN_ROOT}/ryzenadj-before.txt"
cat /sys/module/amdgpu/parameters/sched_policy \
  | tee "${RUN_ROOT}/amdgpu-sched-policy.txt"
cat /sys/module/ttm/parameters/pages_limit \
  | tee "${RUN_ROOT}/ttm-pages-limit.txt"
tuned-adm active | tee "${RUN_ROOT}/tuned.txt"

# Compiler identity used by all require-cached children.
hipcc --version | tee "${RUN_ROOT}/hipcc-version.txt"

# Mandatory before a kernel port/body change.
${PY} scripts/check_lineage.py --kind kernel --diff stat \
  | tee "${RUN_ROOT}/kernel-lineage.txt"
```

### 6.2 Corrected c1 ownership profile

This profile is attribution only; it makes no quality or speed decision.

```bash
${PY} scripts/zbook_production_numerics_c1_profile.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --expected-host zbook \
  --prompt-token-id 9707 \
  --expected-token-id 9707 \
  --prompt-length 512 \
  --baseline-steps 128 \
  --baseline-warmup-steps 1 \
  --baseline-repetitions 5 \
  --profile-steps 24 \
  --profile-warmup-steps 4 \
  --gdn-mode chain_lds32_direct_nonvolatile \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --raw-root "${RUN_ROOT}/c1-profile-raw" \
  --output "${RUN_ROOT}/c1-profile.json"
```

### 6.3 Existing c1 route control

This reproduces the incumbent cooperative router evidence. New candidates must
first add a declared candidate policy/manifest and RED tests rather than
reusing the router label.

```bash
${PY} scripts/execution_profile_gguf_c1_route_gate.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --candidate router_f32w_coop_persistent \
  --decode-steps 24 \
  --repeat-runs 3 \
  --baseline-gdn-mode chain_lds32_direct_nonvolatile \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/c1-quality.json"

${PY} scripts/execution_profile_gguf_c1_route_perf.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --candidate router_f32w_coop_persistent \
  --quality-artifact "${RUN_ROOT}/c1-quality.json" \
  --prompt-token-id 9707 \
  --prompt-length 512 \
  --decode-steps 128 \
  --warmup-decode-steps 1 \
  --repetitions 5 \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/c1-fixed-perf.json"

${PY} scripts/execution_profile_gguf_c1_route_perf.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --candidate router_f32w_coop_persistent \
  --quality-artifact "${RUN_ROOT}/c1-quality.json" \
  --natural-suite \
  --decode-steps 128 \
  --warmup-decode-steps 1 \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/c1-natural-perf.json"
```

### 6.4 Existing c1/cN bundle, lifecycle, and graph wall

```bash
${PY} scripts/execution_profile_gguf_batch_route_gate.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --include-router-candidate \
  --widths 4,8 \
  --decode-steps 24 \
  --repeat-runs 3 \
  --dynamic \
  --sparse \
  --gdn-mode chain_lds32_direct_nonvolatile \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/cn-quality.json"

${PY} scripts/gguf_arbitrary_c_lifecycle.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --rows 8 \
  --cancel-slots 2 6 \
  --prompt-token-id 9707 \
  --prompt-length 16 \
  --original-max-tokens 5 \
  --newcomer-max-tokens 3 \
  --compact-after-middle-hole \
  --allow-c1-arithmetic-drift \
  --quality-artifact "${RUN_ROOT}/cn-quality.json" \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/cn-lifecycle.json"

${PY} scripts/q8t16_batch_route_perf.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --include-router-candidate \
  --configurations c2,c4,native_c8 \
  --pairs 7 \
  --prompt-tokens 512 \
  --decode-steps 128 \
  --gdn-mode chain_lds32_direct_nonvolatile \
  --quality-artifact "${RUN_ROOT}/cn-quality.json" \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --output "${RUN_ROOT}/cn-perf.json"
```

The candidate environment must leave
`HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` unset so the package `MIN_ROWS=4` policy
keeps c2 direct. A forced-all capture is invalid for the current bundle.

### 6.5 Generic named-profile evaluation

PN1 supplies these files. The generic gate is the binding named-profile verdict;
model-specific adapters may screen earlier but cannot replace it.

```bash
${PY} scripts/execution_profile_gate.py \
  --variant-manifest "${RUN_ROOT}/production-variant-manifest.json" \
  --strict-manifest "${RUN_ROOT}/strict-variant-manifest.json" \
  --strict-capture "${RUN_ROOT}/strict-capture.json" \
  --candidate-capture "${RUN_ROOT}/production-capture.json" \
  --expected-controls "${RUN_ROOT}/production-expected-controls.json" \
  --strict-expected-controls "${RUN_ROOT}/strict-expected-controls.json" \
  --comparison-controls "${RUN_ROOT}/isolation-expected-controls.json" \
  --repeat-capture "${RUN_ROOT}/production-repeat.json" \
  --isolation-capture "${RUN_ROOT}/production-isolation.json" \
  --task-results "${RUN_ROOT}/task-results.json" \
  --arithmetic-class T2 \
  --bf16-logits "${RUN_ROOT}/bf16-aligned-logits.npy" \
  --bf16-thresholds "${RUN_ROOT}/bf16-thresholds.json" \
  --output "${RUN_ROOT}/execution-profile-evaluation.json"
```

For a T0/bit-identical candidate, set `--arithmetic-class T0`; the task artifact
still records full-suite strict/candidate output hashes and the BF16 attachment
records zero implementation delta. Do not pass dummy controls, tasks, or BF16
files.

### 6.6 Production server screen and binding packet

The screen is diagnostic. The second command is the unchanged binding packet.
Run each named profile in a fresh process after PN1.

```bash
HIPENGINE_EXECUTION_PROFILE=production \
${PY} scripts/gguf_production_load_gate.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --gdn-mode chain_lds32_direct_nonvolatile \
  --workloads static_c1,static_c8,ragged_burst,continuous_fixed \
  --skip-tuning \
  --fixed-rate-per-second 20 \
  --batch-window-ms 100 \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/production-load-screen.json"

HIPENGINE_EXECUTION_PROFILE=production \
${PY} scripts/gguf_production_load_gate.py \
  --model "${MODEL}" \
  --backend hip_gfx1151 \
  --gdn-mode chain_lds32_direct_nonvolatile \
  --compiler-version-file "${RUN_ROOT}/hipcc-version.txt" \
  --require-cached-build \
  --json "${RUN_ROOT}/production-load-full.json"
```

Run the same complete command with `HIPENGINE_EXECUTION_PROFILE=strict` for the
strict SLO-goodput/default denominator. The incumbent-production denominator is
the clean PN2 production artifact. Do not use verifier-derived, cross-host,
historical, or omitted-profile rows as the public-default denominator.

### 6.7 Post-run snapshot

```bash
rocm-smi --showproductname --showuse --showmemuse --showtemp \
  | tee "${RUN_ROOT}/rocm-smi-after.txt"
sudo ryzenadj -i | tee "${RUN_ROOT}/ryzenadj-after.txt"
find "${RUN_ROOT}" -type f ! -name raw-sha256.txt -print0 \
  | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/raw-sha256.txt"
```

---

## 7. Candidate declaration template

Copy this into the candidate's new worklog entry before implementation:

```text
Candidate ID:
Execution profile / arithmetic class:
Model / quant / KV / backend / host:
Affected layer roles and physical shapes:
Stateful/control surfaces touched:
Fresh profile evidence and complete-wall ceiling:
Hypothesis and gfx1151 mechanism:
Source/lineage (path + commit), if any:
Strict/unfused fallback registry keys:
Expected kernel names / launch-count change:
Can logits, expert IDs, or generated IDs change?:
RED fixture and CPU/parent oracle:
Screen command (diagnostic only):
Full numerical/control/task/BF16 commands:
Fixed/natural/dynamic/long-context performance commands:
Server/SLO commands:
Retention rule:
Rollback/removal trigger:
Artifact path:
```

No field may be filled after looking at final performance except measured result
and verdict.

---

## 8. PUNCHLIST

### Inherited prerequisites

- [x] Host-specific model, prompt, power, and quant identities are pinned.
- [x] Execution-profile policy and calibrated numerical envelope are approved.
- [x] Generic schemas/evaluator and fail-closed profile plumbing exist.
- [x] Current c1 router and c4/c8 rowtile package routes have numerical and
      same-host performance evidence.
- [x] c8 lifecycle semantics distinguish exact ownership from arithmetic value
      identity.
- [x] Current complete server blocker is recorded: 87 completed / 33 rejected
      during soak.
- [x] Framework Desktop MTP and other-host gfx1151 numbers are excluded as
      denominators.

### PN0 — Campaign start

- [ ] Create campaign-start worklog entry and run tag.
- [ ] Confirm clean source and record commit.
- [ ] Confirm full model and both prompt hashes.
- [ ] Record `rocminfo`, ROCm/HIP/compiler, power, thermal, TTM, memory, and idle
      snapshots.
- [ ] Create compiler-version file and warm all required caches outside timing.
- [ ] Run and review kernel lineage report before kernel work.

### PN1 — Named profile/control foundation

- [x] Register strict Qwen3.6 GGUF gfx1151 plan (`hipengine/generation/\
      qwen36_gguf_profiles.py`).
- [x] Register incumbent production plan with strict fallbacks.
- [x] batch_invariant left unregistered; resolution fails closed to strict with
      `fell_back_to_strict` and strict selections.
- [x] Cold-path policy binder applies cooperative router + rowtile-floor env at
      generator construction (production) and disables both (strict).
- [x] Resolution/fallback/duplicate/missing/binder RED tests green.
- [ ] Emit selected and strict manifest hashes from direct and server paths.
- [x] Deterministic control-capture model + serializers + independent fixture
      builder (`hipengine/benchmark/control_capture.py`) with RED tests.
- [x] Live-producer harness (`scripts/execution_profile_gguf_control_smoke.py`)
      that runs strict/production/repeat/isolation teacher-forced schedules,
      reads live session position/token per row, emits captures + fixtures +
      variant manifests + task results, and invokes the gate. CPU gate dry-run
      validated end-to-end.
- [x] RED tests for row, position, mask, KV/state owner, expert scatter, graph
      route, transaction, and lifecycle failures (17 control-capture tests
      incl. swapped-slot, wrong-owner-key, position/context, KV base offset,
      stale graph bucket, and lifecycle-leak mismatch negatives).
- [x] Pass no-change strict/production GPU smoke through
      `execution_profile_gate.py` (2026-08-17, run root
      `/tmp/hipengine-zbook-production-numerics/20260817T031433Z-7b3c810a8525`):
      decision `passed`, `eligible_for_automatic_admission=True`; controls /
      determinism / isolation / tasks / generated all pass; strict-vs-production
      KL=0.0, top-1=1.0, 4/4 IDs equal (T2 cooperative router bit-exact on this
      schedule). batch_invariant fail-closed covered by resolution unit tests.
- [x] Commit the control-capture remainder of PN1 as one validated logical unit.
- [ ] Commit independent full-suite strict/production expected-control fixtures
      (mechanism validated by the smoke; frozen fixtures land with the full-suite
      task artifact).

### PN2 — Baseline refresh

- [ ] Capture corrected c1 wall plus device-stage/rocprof profile.
- [ ] Reproduce c1 full-logit/state gate.
- [ ] Reproduce c1 fixed and 18-prompt natural performance.
- [ ] Reproduce c1/cN 1,050-row static/dynamic/sparse gate.
- [ ] Reproduce c8 lifecycle/compaction gate.
- [ ] Reproduce seven-pair c2/c4/c8 graph wall.
- [ ] Run focused production-load screen.
- [ ] Run strict and incumbent complete server packets.
- [ ] Reconcile >=90% of complete wall or record measured residual.
- [ ] Record soak occupancy, physical-width exposure, rejection timing, and the
      throughput ceiling required to clear offered load.
- [ ] Publish compact PN2 baseline artifact and raw hashes.

### PN3 — Select candidate 1

- [ ] Confirm fresh stage ranking; do not rely on the prior 6.022/5.949 ms rows
      without reproduction.
- [ ] Check `REFACTOR.md` and lineage to avoid rejected/superseded paths.
- [ ] Select one operation family only.
- [ ] Fill every field in the candidate declaration template.
- [ ] Predeclare scope, numerical class, controls, task/BF16 rows, performance
      ceiling, strict fallback, and retention/removal rules.
- [ ] Add and observe RED before implementation.

### PN4 — Candidate correctness

- [ ] Pass leaf oracle and edge/sentinel tests.
- [ ] Pass operation-complete producer/consumer gate.
- [ ] Confirm expected kernel name/resources with cached `rocprofv3` smoke.
- [ ] Pass 18-prompt full-logit gate and three repeats.
- [ ] Pass exact control/ownership and same-width isolation.
- [ ] Pass applicable c1/c2/c4/c8, dynamic, sparse, cancellation, compaction,
      graph/eager, and long-context scenarios.
- [ ] Pass task and BF16 rules.
- [ ] Stop and commit a rejection immediately if any binding gate fails.

### PN5 — Candidate performance/SLO

- [ ] Run operation-complete paired micro/leaf measurements.
- [ ] Run seven-pair p512/d128 complete-wall A/B.
- [ ] Run 18-prompt natural c1 A/B if c1 is affected.
- [ ] Run c2/c4/c8 plus dynamic/sparse timing if cN is affected.
- [ ] Run 4K/32K/64K transfer when attention/context is affected.
- [ ] Re-profile and reconcile the measured mechanism.
- [ ] Run focused server screen.
- [ ] Run unchanged complete server packet.
- [ ] Compare SLO-goodput against same-commit strict and incumbent production.
- [ ] Record package-retain, named-production, public-default, reject, or blocked
      verdict without weakening the protocol.

### PN6 — Evidence/cleanup

- [ ] Write compact artifact with model/hardware/command/sample/control hashes.
- [ ] Update benchmark README/date and changelog for retained performance.
- [ ] Update worklog, this punchlist, kernel catalog/lineage, and refactor ledger.
- [ ] Remove rejected candidate code/flags or retain only an explicitly justified
      diagnostic.
- [ ] Run focused tests and applicable milestone suite.
- [ ] Stage explicit files, inspect staged diff, and commit immediately.

### PN7 — Optional exact MTP follow-up

- [ ] Start only after the AR verdict.
- [ ] Use clean same-host true AR denominator.
- [ ] Run full train/heldout/category suite and exact transaction oracle.
- [ ] Repeat adaptive B2/B3 brackets before changing any cap.
- [ ] Keep MTP/default and AR/default decisions separate.

### PN8 — Closure

- [ ] Every candidate has retained/rejected/blocked artifact and immutable
      worklog entry.
- [ ] No unexplained dirty state, temporary flag, duplicate dispatch path, or
      stale cache/profiler output remains.
- [ ] Current default disposition and next dominant owner are explicit.
- [ ] `PLAN.md`, campaign navigation, benchmark rollup, and handoff are current.
