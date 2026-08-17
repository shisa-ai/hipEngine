# Qwen3.6-35B ZBook Production-Numerics Tuning

Status: **closed 2026-08-17 — measured no-win for the c1 mechanism search; next dominant owner explicit**

AR punchlist PN0-PN8 executed: P3-LAQ1 / P3-LA2 / P3-LAQ1-B rejected with durable
evidence, host and recurrence A/Bs neutral, model ~94% GPU-bound at ~30.5 ms/token
sync'd eager. Next dominant owner: the `launch_gguf_linear` T16 GEMV path
(~8.9 ms/token, ~29% of wall); only a non-T0 (production-envelope) mechanism or a
launch restructure has real headroom. PN7 (MTP) is a separate deferred lane.

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

> Completed in the PN0 unit (2026-08-17); the checkboxes record the evidence.

- [x] Create campaign-start worklog entry and run tag.
- [x] Confirm clean source and record commit.
- [x] Confirm full model and both prompt hashes.
- [x] Record `rocminfo`, ROCm/HIP/compiler, power, thermal, TTM, memory, and idle
      snapshots.
- [x] Create compiler-version file and warm all required caches outside timing.
- [x] Run and review kernel lineage report before kernel work.

### PN1 — Named profile/control foundation

- [x] Register strict Qwen3.6 GGUF gfx1151 plan (`hipengine/generation/\
      qwen36_gguf_profiles.py`).
- [x] Register incumbent production plan with strict fallbacks.
- [x] batch_invariant left unregistered; resolution fails closed to strict with
      `fell_back_to_strict` and strict selections.
- [x] Cold-path policy binder applies cooperative router + rowtile-floor env at
      generator construction (production) and disables both (strict).
- [x] Resolution/fallback/duplicate/missing/binder RED tests green.
- [x] Emit selected and strict manifest hashes from direct and server paths:
      `ResolvedRuntimeProfile.strict_manifest_sha256`, forwarded to the
      generator (`execution_profile_strict_manifest_sha256`) and engine loop,
      LLM property, and the server `/v1/models` execution_profile block.
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
- [x] Full-suite task-results artifact: all 18 greedy task-prompt outputs equal
      strict exactly under the named production profile (2026-08-17, run root
      `/tmp/hipengine-zbook-production-numerics/20260817T035551Z-c26f05bba5e5`,
      decode_steps=128): `18/18` `greedy_outputs_equal_all=True`,
      `task passed=True`, per-prompt output hashes recorded; evidence
      `benchmarks/results/2026-08-17-zbook-qwen36-c1-fullsuite-task-18of18.json`.
- [x] Zero implementation delta (BF16): strict/candidate full logits are
      byte-identical (KL=0.0, top-1=1.0 on the no-change smoke and 18/18 on the
      full suite), so Section 4.3's zero-BF16-budget rule applies; recorded in
      the task artifact (`bf16_note`).
- [x] Commit independent full-suite strict/production expected-control fixtures
      (mechanism validated by the smoke and full suite; full ~3.4MB fixture
      JSONs land in the run root with the task artifact and their SHA256s are
      recorded in the committed evidence artifact).

### PN2 — Baseline refresh

- [x] Capture corrected c1 wall plus device-stage/rocprof profile
      (`zbook_production_numerics_c1_profile.py`, run root
      `/tmp/hipengine-zbook-production-numerics/20260817T040401Z-53be617b835b`):
      `status=complete`, `measurement_valid=True`, expected token 9707 exact;
      host wall 21.79 tok/s (45.9 ms/token); top device stages (ms/token):
      linear_attn_qkv_gate 7.74, ffn_moe_selected 6.96, ffn_moe_combine 4.57,
      ffn_moe_shared_down 4.18, ffn_moe_shared_gate_up 3.95, ffn_moe_router
      3.77, linear_attn_ssm_out 3.47, full_attn_qkv_head_norm_rope 3.04.
- [x] Reproduce c1 full-logit/state gate
      (2026-08-17 run root
      `/tmp/hipengine-zbook-production-numerics/20260817T040401Z-53be617b835b`,
      `execution_profile_gguf_c1_route_gate.py --decode-steps 24 --repeat-runs
      3`, clean tree): `status=passed`, `measurement_valid=True`, 450 rows,
      `kl_max=0`, `top1_agreement=1.0`, `teacher_nll_delta=0.0`,
      repeat-deterministic, hard_gates_passed=True.
- [x] Reproduce c1 fixed and 18-prompt natural performance
      (fixed p512/d128 x5: candidate 32.67 vs strict 30.24 tok/s, +8.02%;
      natural 18-prompt d128: candidate 32.90 vs strict 30.43 tok/s, +8.27%,
      18/18 paired wins, IDs equal; non-regressive vs retained 2026-08-16
      within ZBook thermal variance).
- [x] Reproduce c1/cN 1,050-row static/dynamic/sparse gate
      (`execution_profile_gguf_batch_route_gate.py --include-router-candidate
      --widths 4,8 --decode-steps 24 --repeat-runs 3 --dynamic --sparse`):
      `status=complete`, `measurement_valid=True`, 1,050 rows bit-exact
      (`kl_max=0`, `top1_agreement=1.0`, `max_abs_logit_delta=0`),
      repeat-deterministic, router candidate env set.
- [x] Reproduce c8 lifecycle/compaction gate
      (`gguf_arbitrary_c_lifecycle.py --rows 8 --cancel-slots 2 6
      --compact-after-middle-hole --allow-c1-arithmetic-drift
      --quality-artifact cn-quality.json`): `status=passed`, compaction
      passed with 6 real moves, ownership/same-run state preserved,
      c1-vs-cN arithmetic bytes accepted as diagnostic (bit-exact quality
      binding via the cn-quality gate); width-8 decode 4/5 packed steps.
- [x] Reproduce seven-pair c2/c4/c8 graph wall
      (`q8t16_batch_route_perf.py --include-router-candidate
      --configurations c2,c4,native_c8 --pairs 7 --prompt-tokens 512
      --decode-steps 128`): `status=complete`, all trajectories exact,
      candidate wins 3/7 (c2 direct, neutral), 7/7 (c4, paired median 1.0063),
      6/7 (native_c8, 1.0083); memory teardown exact.
- [x] Run focused production-load screen
      (`gguf_production_load_gate.py --workloads
      static_c1,static_c8,ragged_burst,continuous_fixed --skip-tuning
      --fixed-rate-per-second 20`): all gates passed (workloads, ownership,
      memory recovery), 0 rejects, occupancy-adaptive packing to width 8,
      TTFT p95 4.47s at full load.
- [x] Run incumbent complete production server packet
      (`gguf_production_load_gate.py` canonical workloads + tuning + 60s
      soak): **soak blocker reproduced** — `workloads_passed=False`, 48
      rejected requests (16 overload + 32 soak, all engine_busy), bucket
      saturated 0.875-1.0 through the soak, ITL p99 up to 1.84s (SLO 0.5s)
      and TTFT p95 up to 12.2s (SLO 10s) in some runs; policy sweep selected
      `fair_128` which passes its single run at 27.75 tok/s goodput (TTFT p95
      1.98s, ITL p99 0.42s) but the full workload still fails.
- [x] Run strict complete server packet (strict SLO-goodput/default
      denominator) — **done 2026-08-17**: `HIPENGINE_EXECUTION_PROFILE=strict`
      complete packet also fails (soak blocker present in both profiles): 49
      rejected (16 overload + 33 soak), selected `fair_128` @ 27.27 tok/s
      goodput (TTFT p95 1.88s, ITL p99 0.42s); bound into the baseline
      artifact as the strict denominator.
- [x] Reconcile >=90% of complete wall or record measured residual
      — measured residual recorded: rocprofv3 kernel durations unavailable on
      this gfx1151/ROCm combo (15,072 zero-duration dispatches), so timing
      authority is `same_stream_device_wall_clock_stages` (ROCTX stage
      markers); exclusive-kernel reconciliation not computable from this
      trace; bound into the baseline artifact.
- [x] Record soak occupancy, physical-width exposure, rejection timing, and the
      throughput ceiling required to clear offered load
      — soak occupancy 0.875-1.0 (bucket saturated); physical width exposure
      1->2->4->8 occupancy-adaptive packing; rejection timing production
      16 overload + 32 soak = 48, strict 16 + 33 = 49; measured goodput
      ceiling ~27.7 tok/s (strict ~27.3) vs ~48-96 tok/s offered
      (2 req/s x max_tokens 24-48) => ~1.7-3.5x throughput increase needed to
      clear the load; bound into the baseline artifact.
- [x] Publish compact PN2 baseline artifact and raw hashes
      (c1 core slice: `benchmarks/results/2026-08-17-zbook-qwen36-c1-pn2-baseline.json`;
      extended to bind the remaining gates once they land).

### PN3 — Select candidate 1

- [x] Confirm fresh stage ranking; do not rely on the prior 6.022/5.949 ms rows
      without reproduction. Reproduced from the PN2 rocprofv3 marker trace
      (24 decode tokens, gfx1151) with correct nested-exclusive GPU-visible
      walls: `scripts/pn3_stage_ranking_from_trace.py`. The PN2 host-wall
      authority over-attributes `decode_linear_attn_qkv_gate` (7.74 ms/token
      host wall vs 2.29 ms/token GPU-exclusive). Fresh GPU-exclusive ranking:
      `moe_router_combine` 10.66 (dominated by the retained cooperative/persistent
      router), `gdn_attention_core` 5.19 (conv+SSM recurrence), `gdn_decay` 3.85,
      shared experts ~3.0 each, selected experts ~2.7-2.9 each, full-attn core
      2.39, `gdn_input_projections` (linear-attn QKV/gate) 2.29. Leaf timing
      (`scripts/pn3_q4_t16_laq_gemv_leaf.py`) shows the qkv/gate stage is ~70%
      kernel-bound (leaf pair 1.52 ms/token) and is the most kernel-bound clean
      family in the top-10; the GDN core leaf (conv+recurrence, 0.67 ms/token
      burst / 1.2 ms/token single-launch, `scripts/pn3_gdn_core_leaf.py`) is far
      below its 5.19 ms marker wall, i.e. much of the profile is host-dispatch
      idle, so a T0 leaf mechanism on a kernel-bound family is the soundest
      first increment.

> **QUANT CORRECTION (PN4, supersedes the leaf attribution above):** the c1
> linear-attention `attn_qkv`/`attn_gate` weights in Qwen3.6-35B UD-Q4_K_M are
> **GGUF_Q8_0_T16, not Q4_K T16** (verified from materialized weight specs and
> the pair-dispatch cache: `q8_t16_dual_split`, 210 fused / 0 unfused during a
> 6-step decode; Q4_K T16 is only the MoE experts). The Q4_K local32 leaf above
> (1.52 ms/token) is therefore **not in the qkv_gate path**. The actual leaf is
> the **Q8_0 t16 dual GEMV, already fused** at burst50 0.0677 ms/layer = 2.03
> ms/token (30 layers) vs the 2.29 ms/token stage wall (incl. 6.5 us/layer
> norm) => the stage is ~90% **kernel-bound** on the Q8 dual GEMV, not
> dispatch-bound. The corrected candidate (P3-LAQ1-B below) targets that
> kernel.
- [x] Check `REFACTOR.md` and lineage to avoid rejected/superseded paths. The
      selected leaf (`dense_single_local32_bf16_bf16_out`, the exact Q4_K T16
      c1 dense GEMV) is the retained local32 owner; no REFACTOR entry or
      do-not-repeat ledger item covers a software-pipelined T0 variant of it
      (Q8_1x2, split-weight, rowtile, DP4A, and pack8 alternatives are distinct
      and remain rejected/retained as recorded).
- [x] Select one operation family only: c1 linear-attention QKV/gate projection
      (`gdn_input_projections` / `decode_linear_attn_qkv_gate`).
- [x] Fill every field in the candidate declaration template (below).
- [x] Predeclare scope, numerical class, controls, task/BF16 rows, performance
      ceiling, strict fallback, and retention/removal rules (below).
- [x] Add and observe RED before implementation: `tests/test_pn3_laq_gemv_red.py`
      (4 tests). Bit-exact guard (local32 T16 == pack8 control) GREEN; three
      leaf perf ceilings RED on the current kernel: attn_qkv 0.0328 > 0.026 ms,
      attn_gate 0.018 > 0.0145 ms, pair 0.0472 > 0.042 ms/layer (gfx1151).

#### P3-LAQ1 candidate declaration

- **Candidate ID / arithmetic class:** P3-LAQ1, T0 (bit-exact; per-lane K
  ownership, FMA order, wave tree, and BF16 rounding unchanged).
- **Scope / shape / stateful surfaces:** c1 (rows=1) linear-attention
  `attn_qkv` (2048->8192) and `attn_gate` (2048->4096), both Q4_K T16, all 30
  layers, leaf `gguf_q4_k_t16_dense_single_local32_bf16_bf16_out`. Stateless
  projections: no KV/state/mask surfaces affected; only the input hidden row
  and the two output scratch buffers.
- **Source / lineage / why it helps gfx1151:** the exact local32 Q4_K dense
  GEMV owner (`gguf_t16_selected_gemv.hip`, pack8-local32 bit-exact sibling).
  Single-wave32/block with a serial 8-super-block loop and dependent
  byte-scattered dequant loads -> latency-bound at rows=1; the leaf runs at
  ~296 GB/s effective (attn_qkv 32.8 us for 9.7 MiB tiles) on gfx1151's 40 CU /
  80 SIMD machine, i.e. a real cycle/occupancy ceiling far below achievable
  BW. Software-pipelining the super-block loads keeps the math bit-exact.
- **Operation-complete boundary / expected kernel or launch change / measured
  complete-wall ceiling:** producer->consumer is the c1 linear-attention
  norm->qkv->gate chain; the variant keeps the same kernel ABI/launch shape and
  only reschedules the inner loop, so the operation-complete gate is unchanged.
  Measured ceiling: leaf pair 1.52 ms/token of the 2.29 ms/token stage; a ~30%
  leaf reduction is worth ~0.5 ms/token (~1% complete-request, above the
  in-tree exact threshold; the 7.74 ms PN2 host-wall figure is not the
  denominator).
- **Strict fallback / rollback-removal trigger:** the unmodified local32 owner
  stays registered and default; the pipelined variant is a separately
  registered T0 variant. Any bit mismatch vs the pack8 control, any same-suite
  complete-wall regression, or any KL/top-1 binding failure removes the variant.
- **Downstream expert/top-1 change:** none (T0 bit-exact).
- **Binding rows:** leaf bit-exact (pack8 control); c1 450-row KL=0/top-1=1.0;
  18-prompt greedy-output equality; c1/c2/c4/c8, dynamic/sparse, graph/eager,
  and long-context rows that touch linear attention; task/BF16 rules in 4.3.
- **Performance ceiling rows:** c1 fixed p512/d128 and 18-prompt natural c1 A/B
  vs same-commit strict and incumbent production.

#### P3-LAQ1 REJECTED (PN4 binding failure, with quant correction)

**Material correction:** the Qwen3.6-35B UD-Q4_K_M c1 linear-attention
`attn_qkv` (2048->8192) and `attn_gate` (2048->4096) are **Q8_0 T16, not
Q4_K T16** (verified from materialized weight specs and the pair-dispatch
resolution cache: `q8_t16_dual_split`, 210 fused / 0 unfused calls during a
6-step decode). Q4_K T16 is used only by the MoE expert tensors. The P3-LAQ1
RED test measured a **Q4_K local32 leaf that is not in the qkv_gate path**;
the rejection outcome is still correct (the Q4 leaf could never help the
stage), but for the wrong reason. The actual stage is dominated by the
**Q8_0 t16 dual GEMV (already fused) at 0.0677 ms/layer burst50 = 2.03
ms/token** vs the 2.29 ms/token stage wall (incl. 6.5 us/layer norm) => the
stage is ~90% **kernel-bound**, not dispatch-bound.

For completeness, the Q4 leaf mechanism itself was also measured and found
without headroom: two bit-exact T0 variants of the local32 owner
- `vecq` (word-load / uint32 qword reads): ~10% slower on both shapes.
- `tile16` (one block computes all 16 columns of a tile): +5% on attn_qkv,
  -7% on attn_gate.

Neither flips the Q4 leaf ceilings; the leaf is at ~350 GB/s effective for the
byte-scattered Q4_T16 layout. The Q4 LAQ1 RED test stays as an xfailed
failure fixture (`tests/test_pn3_laq_gemv_red.py`: bit-exact guard live, three
leaf ceilings xfail with the rejection reason).

**Rejection disposition (plan 4.4):** candidate implementation removed
(`vecq`/`tile16` kernels, launchers, exports, wrappers, and registrations
reverted to the committed PN3 state). Evidence artifact:
`benchmarks/results/2026-08-17-zbook-qwen36-pn3-laq1-rejected.json` (contains
the material correction). Worklog:
`worklog/entries/20260817T060720.847694Z-lhl-pn3-laq1-rejected-fe2eae.md` and
the PN4 quant-correction entry. The corrected mechanism (P3-LAQ1-B below)
replaces the withdrawn P3-LA2 launch-fusion candidate.

#### P3-LAQ1-B candidate declaration (corrected mechanism; supersedes P3-LA2)

> P3-LA2 (launch-count reduction of the c1 linear-attention block) is
> **withdrawn**: the qkv+gate pair is already fused into one launch via
> `q8_t16_dual_split`, the norm is a separate 6.5 us/layer launch, and the
> stage wall (72 us/layer) is at/below the kernel sum (6.5 + 67.7 us) -> no
> dispatch headroom remains. A Q4 local32 pair kernel was prototyped and
> reverted (wrong quant, not routed).

- **Candidate ID / arithmetic class:** P3-LAQ1-B, T0 (bit-exact; per-lane
  K ownership, FMA order, wave tree, and BF16 rounding unchanged from the
  Q8_0 t16 dual GEMV owner `q8_0_t16_dual_split_gemv_kernel`).
- **Scope / shape / stateful surfaces:** c1 (rows=1) linear-attention
  `attn_qkv` (2048->8192) + `attn_gate` (2048->4096), both Q8_0 T16, all 30
  layers, leaf `hipengine_gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out`
  (already fused). Stateless projections: only the input hidden row and the
  two output scratch buffers.
- **Source / lineage / why it helps gfx1151:** the exact Q8_0 t16 dual_split
  GEMV owner (`gguf_q8_0_t16_gemv.hip`). 128-thread/4-wave blocks each handle
  one 16-col tile with a strided 2048-wide K loop and byte-scattered int8 q
  loads -> the leaf runs at ~372 GB/s effective (25.2 MiB tiles in 67.7 us)
  on gfx1151's 40 CU / 80 SIMD machine, i.e. below achievable L2 bandwidth;
  load scheduling / word-loads / tile geometry are the T0 knobs.
- **Operation-complete boundary / expected kernel or launch change / measured
  complete-wall ceiling:** producer->consumer unchanged (norm->qkv_gate); the
  variant keeps the same kernel ABI/launch shape and only reschedules the
  inner loop. Measured ceiling: leaf 2.03 ms/token of the 2.29 ms/token
  stage; a 20-30% leaf reduction is worth ~0.4-0.6 ms/token (~0.9-1.3%
  complete-request), above the 1% in-tree exact threshold.
- **Strict fallback / rollback-removal trigger:** the unmodified Q8 dual
  owner stays registered and default; the variant is a separately registered
  T0 variant. Any bit mismatch vs the Q8 dual owner, any same-suite
  complete-wall regression, or any KL/top-1 binding failure removes it.
- **Downstream expert/top-1 change:** none (T0 bit-exact).
- **Binding rows:** leaf bit-exact (variant vs dual owner); c1 450-row
  KL=0/top-1=1.0; 18-prompt greedy-output equality; c1/c2/c4/c8,
  dynamic/sparse, graph/eager, and long-context rows that touch linear
  attention; task/BF16 rules in 4.3.
- **Performance ceiling rows:** c1 fixed p512/d128 and 18-prompt natural c1 A/B
  vs same-commit strict and incumbent production.

### PN4 checkpoint — host-side dispatch analysis (decides P3-LAQ1-B or a host mechanism)

Measured on the live ZBook session (Qwen3.6-35B UD-Q4_K_M, gfx1151) with
per-call host timing during eager decode:

- Host dispatch per call (median): `launch_gguf_linear` 48.1 us,
  `launch_gguf_linear_pair` 55.4 us, `launch_gguf_linear_triple` 70.5 us,
  `qwen35_gdn_recurrent_rmsnorm_gate_lowp` 38.7 us, conv 34.6 us,
  attn_norm 20.7 us. Pure HIP enqueue (pre-resolved, pre-bound) is ~6 us, so
  ~40-45 us/call is Python dispatch machinery (cache-key construction,
  `_LAUNCH_ABI` resolution, allocation lookups).
- The `_DISPATCH_RESOLVE_CACHE` actually hits 97% (9 misses / 282 gets per
  step); earlier "80% miss" readings conflated `resolve_gguf_linear_dispatch`
  calls made by the pair resolver. So re-resolution is NOT the cost.
- Linear-attention block host time ~288 us/layer (~207 us of instrumented
  launches + ~81 us of surrounding runner logic); host step ~31 ms/token
  (device sync after enqueue ~0.04 ms). Production p512/d128 wall is
  ~46 ms/token (21.79 tok/s).
- rocm-smi reports 0% GPU use on this APU (metric unavailable), and stage
  markers are host-side timestamps whose GPU-exclusive walls overlap in GPU
  time (the "150 us/layer gdn_attention_core idle" is not necessarily GPU
  idle -- other stages' kernels can execute inside that window).
- **Verdict: host-vs-GPU bound is UNRESOLVED.** The qkv_gate stage itself is
  kernel-bound (Q8 dual 67.7 us vs 72 us wall). The decisive experiment is a
  complete-wall A/B with a host-speedup (launch-plan memoization) or a
  GPU-reduction: if the wall drops, the model is host-bound and the memoization
  is the biggest win; if flat, it is GPU-bound and P3-LAQ1-B (the Q8 dual
  kernel) is the target. Do NOT invest in a large gguf_linear.py refactor
  before this A/B.

> **A/B RESOLUTION (PN4):** the complete-wall A/B resolved the question. A
> memoized fast path on `launch_gguf_linear` (skips key construction / env
> reads / session gets / cache lookup, jumps to the launch tail; 4400 of 5781
> calls) moved the sync'd eager wall 30.59 -> 29.73 ms/token (only -857 us,
> -2.8%). The model is ~94% GPU-bound; a gguf_linear launch-memoization refactor
> is not worth it. P3-LAQ1-B (GPU-side Q8 dual) was the correct family.

> **P3-LAQ1-B REJECTED (PN4, binding failure):** three bit-exact T0 variants of
> the Q8_0 t16 dual GEMV were implemented and measured (wordload 2x uint64 +
> 4x uint32; wordload+occupancy launch_bounds 128,8; wordload+ILP+occupancy).
> All are bit-exact vs the owner, but none flips the leaf RED: owner 62 us,
> wordload 62 us, wordload+occ8 58 us, wordload+ILP+occ8 60 us/layer
> (interleaved A/B) vs the 42-48 us (20-30%) target. The kernel is
> latency/occupancy-bound (one block per 16-col tile, 768 blocks; per-block
> wave reduce + xchg + syncthreads) at ~510-540 GB/s vs a ~650 GB/s marginal
> L2 ceiling, and T0 variants cannot close the latency gap (threads=64 is both
> slower and not bit-exact). This matches the Q4_K T16 precedent (P3-LAQ1:
> vecq -10%, tile16 +5/-7%). The qkv_gate stage stays kernel-bound at
> ~2.29 ms/token. Evidence:
> `benchmarks/results/2026-08-17-zbook-qwen36-pn4-laq1b-rejected.json`;
> diagnostic `scripts/pn4_host_bound_ab_probe.py`. Candidate kernels reverted;
> owner remains default.

> **c1 CAMPAIGN NO-WIN CONCLUSION (PN4, closes PN3/PN4 for the c1 path):** every
> c1 mechanism attempted has a durable verdict and none is retainable:
> P3-LAQ1 (Q4 GEMV, wrong quant) rejected; P3-LA2 (launch fusion, already
> fused) rejected; P3-LAQ1-B (Q8 dual GEMV T0) rejected; host memoization
> A/B (`scripts/pn4_host_bound_ab_probe.py`) recovers only ~857 us/token from
> fast-pathing the biggest launcher; recurrence-path host fast-path A/B
> (`scripts/pn4_recurrence_fastpath_ab.py`) recovers ~0 (fully hidden). A
> recurrence conv+recurrence no-op probe
> (`scripts/pn4_recurrence_ab_probe.py`) drops the wall 30.86 -> 28.41 ms/token
> (-2.44 ms), i.e. the gdn_attention_core conv+recurrence kernels are a real
> ~2.4 ms/token GPU-bound slice (not the 5.19 ms marker wall, which overlaps
> other stages' GPU work, and not the 0.67-1.2 ms burst/single-launch leaf,
> which understates in-context time). The model is ~94% GPU-bound (eager
> sync'd wall ~30.5 ms/token). Next dominant owner: the MoE expert / router
> GPU path (the largest GPU slice per the PN3 ranking) -> the cN/A4 campaign
> (PN3-PN5 extension), not a c1 GEMV or host mechanism.

### PN4 — Candidate correctness

> **Superseded by rejection (P3-LAQ1-B, PN4):** the selected candidate was
> rejected at the leaf RED before any PN4 correctness gate was run. The
> bit-exact guard for all retained T0 variants (wordload/occ/ILP) PASSED (all
> equal the owner and the CPU reference), but no T0 variant flips the leaf
> timing ceiling, so the PN4 correctness rows below are N/A for a rejected
> candidate. Fixture: `tests/test_pn4_laq1b_red.py` (bit-exact guard GREEN,
> leaf timing xfail with rejection reason).

- [x] N/A (rejected at RED) — leaf oracle/edge/sentinel.
- [x] N/A (rejected at RED) — operation-complete producer/consumer.
- [x] N/A (rejected at RED) — cached `rocprofv3` kernel smoke.
- [x] N/A (rejected at RED) — 18-prompt full-logit gate.
- [x] N/A (rejected at RED) — exact control/ownership isolation.
- [x] N/A (rejected at RED) — c1/c2/c4/c8 scenarios.
- [x] N/A (rejected at RED) — task/BF16 rules.
- [x] Rejection committed immediately at the leaf-RED binding failure
      (P3-LAQ1-B, PN4) with the failure fixture preserved.

### PN5 — Candidate performance/SLO

> **N/A — the selected candidate was rejected at PN4 RED.** No PN5 counter-
> balanced performance/SLO rows apply to a rejected candidate. The host A/B
> diagnostics (`scripts/pn4_host_bound_ab_probe.py`) and recurrence A/Bs
> (`scripts/pn4_recurrence_*.py`) establish that no c1 host mechanism is
> worth a refactor.

- [x] N/A (rejected at RED) — operation-complete leaf measurements.
- [x] N/A (rejected at RED) — seven-pair p512/d128 A/B.
- [x] N/A (rejected at RED) — 18-prompt c1 A/B.
- [x] N/A (rejected at RED) — c2/c4/c8 dynamic timing.
- [x] N/A (rejected at RED) — 4K/32K/64K transfer.
- [x] N/A (rejected at RED) — profile reconcile.
- [x] N/A (rejected at RED) — server screen / packet.
- [x] N/A (rejected at RED) — SLO-goodput comparison.
- [x] N/A (rejected at RED) — verdict record (rejected, P3-LAQ1-B).

### PN6 — Evidence/cleanup

> Completed 2026-08-17 for the rejected candidates (P3-LAQ1, P3-LA2, P3-LAQ1-B)
> and the host/recurrence A/B diagnostics.

- [x] Write compact artifact with model/hardware/command/sample/control hashes
      (`2026-08-17-zbook-qwen36-pn3-laq1-rejected.json`,
      `2026-08-17-zbook-qwen36-pn4-laq1b-rejected.json`).
- [x] N/A — no retained performance to roll up (no-win); benchmark README/date
      and changelog are intentionally unchanged for this campaign.
- [x] Update worklog, this punchlist, kernel catalog/lineage, and refactor
      ledger (worklog entries + punchlist updated; no kernel ownership change
      and no new flags, so catalog/lineage/refactor are unchanged).
- [x] Remove rejected candidate code/flags or retain only an explicitly
      justified diagnostic (all candidate kernels reverted; host/recurrence/
      next-owner A/B probes retained as explicit diagnostics).
- [x] Run focused tests and applicable milestone suite
      (`tests/test_pn4_laq1b_red.py`: bit-exact guard PASS, leaf timing xfail;
      broader suite not rerun for a docs-only closure per the focused-repair
      rule).
- [x] Stage explicit files, inspect staged diff, and commit immediately.

### PN7 — Optional exact MTP follow-up

> **Deferred to a separate MTP unit (out of scope for this AR campaign).**
> MTP is an independent speculative/economics lane with its own run tag and
> artifact family; its numbers never form an AR candidate denominator. The AR
> verdict (P3-LAQ1-B rejected, c1 no-win) is recorded, so a future MTP unit
> may start from this state, but it is not part of this production-numerics
> AR punchlist.

- [x] N/A (separate MTP unit, deferred) — start after the AR verdict (recorded).
- [x] N/A (separate MTP unit, deferred) — clean same-host true AR denominator.
- [x] N/A (separate MTP unit, deferred) — full train/heldout/category suite.
- [x] N/A (separate MTP unit, deferred) — repeat adaptive B2/B3 brackets.
- [x] N/A (separate MTP unit, deferred) — keep MTP/default and AR/default
      decisions separate.

### PN8 — Closure

> **c1 mechanism search closed (PN4, no-win).** Every candidate has a durable
> verdict and artifact: P3-LAQ1 rejected (`2026-08-17-zbook-qwen36-pn3-laq1-
> rejected.json`), P3-LA2 rejected (withdrawn, no dispatch headroom), P3-LAQ1-B
> rejected (`2026-08-17-zbook-qwen36-pn4-laq1b-rejected.json`), host and
> recurrence A/Bs neutral (`scripts/pn4_host_bound_ab_probe.py`,
> `scripts/pn4_recurrence_*.py`). No transient runtime path is left untracked
> (all candidate kernels reverted; owner remains default).
>
> **Next dominant owner (measured):** the `launch_gguf_linear` GEMV path is
> the largest complete-wall slice at ~8.9 ms/token of the ~30.7 ms/token
> sync'd eager wall (`scripts/pn4_moe_gpu_cost_probe.py`: no-op drops 30.65 ->
> 21.76 ms/token; +pair -> 20.69; +triple -> 20.07). The census shows it is
> dominated by Q8_0 T16 attention projections (attn_q 2048->4096 x40/step,
> attn_k/v 2048->512 pair_concat + singleton) plus ~60 tiny dense 2048->32
> routing-logit GEMVs and the Q6 lm_head. These are the same T16 GEMV family
> proven at its practical T0 limit by P3-LAQ1-B, so a T0 bit-exact mechanism
> on this slice is not expected to flip a meaningful RED; a future non-T0
> (production-envelope) mechanism or a launch restructuring (fuse/batch the
> per-layer attention projections and routing logits) is the only path with
> real headroom.

- [x] Every candidate has a retained/rejected artifact and immutable worklog
      entry (P3-LAQ1, P3-LA2, P3-LAQ1-B, host/recurrence A/Bs).
- [x] No unexplained dirty state, temporary flag, or stale cache remains
      (all candidate kernel code reverted; diagnostics are explicit scripts).
- [x] Current default disposition (owner stays default) and next dominant
      owner (the launch_gguf_linear T16 GEMV path, non-T0/restructure only)
      are explicit.
- [x] `PLAN.md`, campaign navigation, worklog, and this punchlist are current.
