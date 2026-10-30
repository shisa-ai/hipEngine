---
status: current
owns: Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven.
---
# hipEngine Refactor / Dead-Path Ledger







## Prefill-Only Workspace Ownership

Current cleanup status (2026-10-25, R0020): complete. The production
prepared-workspace selector is removed; the same-owner diagnostic owns its
selection instrumentation. `scripts/qwen4exp_prefill_restore_ab.py` now uses a
prefill-only owner for timing comparisons. Its `--workspace-check-only` mode
keeps an independent full decoder because that mode executes the decoder as
its numerical reference, not merely as a scratch donor.

Extra-workspace ownership, cleanup, reserve checks, and original staging
capacity remain covered. The replacement passed real-model logits/state parity
and allocation-lifecycle checks. No new timing comparison is claimed; historical
benchmark commands and results remain source-pinned. See
`worklog/entries/20261025T205203.077962Z-cleanup-refactor-r0020-donor-b59da2.md`.

The following is historical context, not open cleanup: statements about the
private selector or the timing path's full donor no longer describe current code.

- `Qwen4ExpPrefillWorkspace` owns prefill scratch and input buffers without
  another decoder/KV allocation. The ordinary runner still allocates one
  workspace in the original order; dispatch and default chunk are unchanged.
- `_allocate_extra_prefill_workspace` prepares an inactive, runner-owned
  workspace. Callers must admit its extra memory before preparing it.
  Automatic selection is not enabled by this ownership refactor.
- The GPU owned-workspace gate passes with exact frozen-reference state
  and1,633,867,320 additional bytes at chunk1024/capacity4352.
  Evidence: `benchmarks/results/2026-09-15-prefill-workspace-owner/`.
  Replacing the benchmark's full donor and shape-dependent dispatch remain
  separate qualification steps. Keep historical donor commands
  source-pinned; remove duplicated donor plumbing once a qualified replacement
  covers state, lifetime, memory and matched performance.
- **`20260929T`: the shape policy this selector exists to provide is measured to be
  unnecessary, so the entry below is a removal candidate rather than a pending
  qualification.** On one host, one runner, two prepared owners, the same short
  prompt served by the small 1024 owner is **1.06% slower** than served by the large
  4096 owner, at **0.017x the within-arm spread** - no measurable benefit - with
  **identical** `logits_sha256`. The premise "preserving the smaller owner's short
  costs" fails because the workspace is scratch: its rows are
  `min(chunk_size, max_sequence_length)` and kernel work scales with the actual token
  count, so a 512-token prefill costs the same whichever owner supplies the buffers.
  The extra owner also costs 1.555 GiB. A single-owner configuration is therefore at
  least as good. Evidence:
  `benchmarks/results/2026-09-29-qwen4exp-rt1-same-owner-selection/`.
- `_configure_prefill_workspace_selection` is a private, default-disabled
  selector over already prepared owners. **As of `20260929T` the two largest
  reasons it was parked are cleared by measurement**: the chunk-4096 canonical
  numerical gate passes bit-identically to chunk 1024 (`X-6`), and chunk 1024 vs
  4096 publishes identical `logits_sha256` on 12 of 12 canonical cases, so the
  change is arithmetic-neutral. The p512 short-request loss that motivated the
  1024 default does not reproduce (**+0.46%, smaller than the within-arm
  spread**). The remaining requirements are unchanged and still binding:
  measure short/long requests **on the same prepared owners** (every measurement
  so far used one process per arm), qualify full state/lifetime and c2 on the
  selection path, account for all owners and host staging in public
  construction, and add allocation/lifetime gates. See
  `benchmarks/results/2026-09-29-qwen4exp-rt1-blocker-status/`. It allocates nothing, rejects owners
  beyond original staging capacity, and restores borrowed fields on failure.
  Its CPU tests pass; automatic selection has no GPU or performance
  qualification yet. The reserve check is not public memory admission.
  Before enabling it, qualify full state/lifetime and c2 behavior, account
  for all owners and host staging in public construction, preserve strict
  setup, and measure short/long requests on the same prepared owners.
  Remove this experimental setup route after a qualified public policy
  replaces it, or remove the selector if that policy is rejected.
- **Parked 2026-09-15 (deliberate, not abandoned).** Automatic selection stays
  default-off and unqualified. Its remaining value is bounded by the explicit
  chunk tradeoffs below: the best case is the 4K prefill gain already measured
  for the explicit options (2.32-3.94%), against short-request regressions of
  0.18-0.40%. Open work is a ~1.6x arithmetic recovery and a ~2.5x structural
  gap to the same-host external comparator, both of which dominate it by an
  order of magnitude. Resume only after the gap-attribution work ranks it back
  into the top of the seconds-recoverable list, or if a default chunk change
  makes the ownership split load-bearing. The private route and its tests stay
  in place; nothing else depends on it.
  - **Resume condition met; premises partly falsified (2026-09-29).** One of the
    two named resume conditions - "a default chunk change makes the ownership
    split load-bearing" - is now live: the serving path's chunk default is
    **256** (`EngineLoopConfig.max_prefill_chunk_tokens`), not the 1024 these
    tradeoffs assume, and that default is the thing under review.
    Both parked premises were re-measured on `code-p4096` and across the 12-case
    fixture (`benchmarks/results/2026-09-29-qwen4exp-rt1-serving-suite/`,
    `.../2026-09-29-qwen4exp-rt1-chunk-remeasure/`):
    - **The short-request regression does not reproduce.** A 512-token request is
      byte-identical in wall time *and* peak allocation at chunk 1024 vs 4096
      (+0.20%, 0.08x the within-arm spread, arms overlapping), because the
      workspace is already shape-aware: `rows = min(chunk_size,
      max_sequence_length)` and the harness sets `max_sequence_length =
      max(768, prompt_tokens + 1)`, so both chunk sizes resolve to 768 rows. The
      recorded 0.18-0.40% is not reproduced here.
    - **The 2.32-3.94% magnitude is correct for 1024 -> 4096** (we measure
      -4.94%, same order) - the park was not wrong about that comparison. What
      was never measured is the **256** baseline the serving path actually uses:
      256 -> 4096 is **-28.2% to -37.6%** across all 12 cases, monotone in prompt
      size and flat across category, 12/12 completely separated.
    So the lever is larger than the park assumed and its cost is confined to peak
    memory (+0.77/+1.24/+6.20 GB at p512/p1024/p4096), not short-request latency.
    **Still not enabled.** The qualification prerequisites below are unchanged and
    unmet: full state/lifetime and c2 behavior, accounting for all owners and host
    staging in public construction, preserving strict setup, and short/long
    requests on the same prepared owners. The reserve check is still not public
    memory admission. The 256 -> 1024 step is the one to measure next, in a single
    run, because `REFACTOR.md` forbids claiming a direct comparison from separate
    runs.
  - **The chunk decision landed, and it makes the selector unnecessary rather
    than load-bearing (2026-09-30).** The 256 -> 1024 step named above was
    measured in a single run (12 cases, 3 reps per arm, one process per arm:
    `benchmarks/results/2026-09-29-qwen4exp-rt1-256-vs-1024/`, -29.0% to -35.7%,
    +1.24 GB at 4K) and **`DEFAULT_MAX_PREFILL_CHUNK_TOKENS` is now 1024** -
    see `worklog/entries/20260930T011826.957349Z-lhl-qwen4exp-rt1-serving-chunk-default-3c5235.md`.
    So the premise this park rested on ("the serving path's chunk default is
    **256**, not the 1024 these tradeoffs assume") is no longer true: the serving
    path now uses the 1024 the tradeoffs assume.
    **That resolves the resume condition by removing its subject.** The
    condition was "a default chunk change makes the ownership split
    load-bearing"; the default changed, but the ownership split still is not
    load-bearing, because a two-owner policy measured **1.06% slower on the small
    owner at 0.017x the within-arm spread** with identical `logits_sha256`
    (`benchmarks/results/2026-09-29-qwen4exp-rt1-same-owner-selection/`) - the
    workspace is scratch and kernel work scales with the actual token count, not
    the size of the scratch it is handed. A single larger owner is therefore at
    least as good as the two-owner policy, without the extra 1.555 GiB, the
    borrow/restore path, or a selector needing qualification.
    **Revised disposition: remove the selector, do not enable it.** The four
    binding prerequisites below are no longer worth satisfying - they qualify a
    mechanism whose only measured benefit is a 1.06% *loss*. The one genuinely
    useful survivor is the capacity question (what peak can the host admit for a
    single larger owner), which is admission accounting, not this selector. Any
    removal must keep the chunk-1024 default's own tests
    (`tests/test_unit_engine_loop_prefill_chunk.py`) intact.
  - **The capacity question is not the 4096 lever, and the tick bound is (2026-10-03).** Owner
    admission cannot reach the measured `chunk 1024 -> 4096` step (`-4.94%`),
    because the owner does not set the chunk: the cap does. The workspace is
    scratch and kernel work scales with the actual token count - the same
    scratch-not-size mechanism this section already records - so a 4096-row owner
    with a 1024 cap still executes `4 x 1024`, the measured-slow configuration.
    What gates the step is the **per-tick prefill bound**: under the default
    `protect_decode` policy one round is exactly one chunk of
    `max_prefill_chunk_tokens` (`hipengine/generation/engine_loop.py:2245`), so
    raising the cap to 4096 quadruples per-tick prefill work and the worst-case
    delay before a decoding peer gets its next step. (Under
    `prefill_decode_policy == "token_budget"` the chunk is
    `min(prefill_chunk_size, round_prefill_token_budget)` at `:2266`, so the cap
    alone is inert there and the round budget must rise with it - lengthening the
    tick for the same reason.) That is a real fairness trade, unlike the landed
    `256 -> 1024` promotion whose per-tick work was argued unchanged. The missing
    measurement is decode interleaving under concurrency at a 4096 chunk, not a
    capacity probe. See
    `worklog/entries/20261003T152924.135144Z-lhl-correction-the-round-budget-does-not-bound-the-d-67c4d6.md`, which corrects the mechanism in `worklog/entries/20261003T152724.639601Z-lhl-rt-1-chunk-4096-is-gated-on-the-round-prefill-bu-3e7462.md`.

## GR Staged-F32 Selector (HC-1)

Cleanup status (2026-10-25, R0021): the whole-route gate rejected this candidate
with `measurement_valid: true`. The GR selector, runtime dispatch branch,
`production_gr_up_staged` candidate, and recovery sensitivity arm are removed.
The shared wrapper and its direct kernel tests remain because the separate
dense staged-F32 registration uses that implementation. The strict GR fallback
is unchanged. Historical gate commands require their source-pinned revision.

The following records the investigation before removal, not open selector work.

- `HIPENGINE_QWEN4_EXP_GR_STAGED` selects
  `coltile2_branch4_rowbatch4_staged_f32` for the GR up projection
  (`linear+gr_gated_mean`). The tile keeps the `dense_wide256` tiling and LDS
  staging but multiplies in f32 FMA on f32 operands, so its only arithmetic
  difference from the fused exact parent is the summation order. Screened at
  **2.388x** with the arithmetic intact
  (`benchmarks/results/2026-09-29-gr-staged-f32-screen/`), and its numerical
  contract test pins it within the f32 summation-order floor (relative L2 <=
  1e-6) of the exact fused kernel at five shapes.
- **Default-off, and it must stay off until the production gate passes.** The
  variant is not bit-exact with the parent, so under
  `docs/EXECUTION-PROFILES.md` it is a T1/T2 arithmetic change needing the
  calibrated mean/tail/max KL, top-1 by category, determinism, isolation and
  task gates. The fused exact parent remains the registered fallback.
- **Concrete blocker, not "needs more evidence":** the gate cannot be run for
  this route yet. Every route in
  `scripts/execution_profile_q8_wmma_prefill_layers_gate.py` is *layer-scoped* -
  its `Route.env_for` formats `{layers}` into the selector - but
  `run_qwen4_exp_gr_read` takes **no layer index**, so a layer-scoped GR
  selector cannot be expressed without threading a layer argument through the GR
  call chain. Without that, a gate run would engage the staged tile on **every**
  layer rather than the named scope, which is not the comparison the gate's
  scope checks are built to certify.
- Related but distinct: the *dense linear* staged tile
  (`HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS`,
  `staged_f32_bm128_bn64_bk32`, `KernelKey(..., "linear", ...)`) is a different
  kernel on a different layer. Its gate has already run and reaches
  `status=complete, canonical=True` on layers 16-47 and 14-47
  (`benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/`).
  That result does not transfer to this route: different layer, different
  epilogue, and the GR dispatch records its own prefill-boundary KL cliff.
- **Disposition:** keep the selector and its unit test while the layer-scoped
  gate is built; remove the flag and the dispatch branch if that gate rejects
  the route, or promote it (and delete the flag) if it passes.
- **`20260929T`: the blocker above is specific to the LAYER-SCOPED gate and does
  not block the whole-route gate, which already qualifies GR candidates.**
  `scripts/qwen4exp_layer2_profile_gate.py` compares a whole-profile candidate
  against its strict parent over 18 prompts x prefill-last plus 24 c1 rows (450
  full-vocabulary rows, three candidate repeats, state fingerprints, free
  outputs) and already carries `production_gr_up_restore` and
  `production_gr_up_wide` for this exact route - both bind GR selectors with no
  layer scope, because the layer2 gate's scope is the route, not a layer range.
  The `run_qwen4_exp_gr_read`-takes-no-layer-index objection therefore applies
  to `execution_profile_q8_wmma_prefill_layers_gate.py` only. The staged-f32
  route is now registered as `production_gr_up_staged` in the layer2 gate and
  its production gate is runnable; the dense sibling's layer-scoped result still
  does not transfer to this route.
- **Registration trap, recorded because it silently turns a gate into a control:**  the GR staged variant is registered only by the module-level
  `register_gr_staged_f32_kernel()` call in
  `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_gr_staged_f32.py`, and
  `register_gfx1151_kernels` reaches that module through `import_module`, which
  is a no-op once the module is imported. `hip_gfx1100/__init__.py` exposes no
  `register_backend_kernels`, so no backend-package entry point can re-create
  the key either. Consequence: after `tests/conftest.py` restores the registry
  to its collection-time baseline, neither `register_gfx1151_kernels` nor
  `load_backend_kernel_package` can restore the gfx1151 alias. Production is
  unaffected (the registry is never wiped) and the gate is fail-closed
  (`count_candidate_dispatch` raises `ValueError` when the key is unregistered),
  but it makes registration order-dependent. This is the same root cause as the
  pre-existing cross-file failure
  `tests/test_unit_qwen4exp_gr_staged_dispatch.py` +
  `tests/test_unit_qwen4exp_layer2_profile_gate.py`
  (`test_gfx1151_registry_contains_layer2_candidate_and_strict_fallback` fails
  when both files run together, and passes when the layer2 file runs alone -
  reproduced at `2812d8789` with no other changes). Cleanup: give the staged GR
  module a registration hook the gfx1100 and gfx1151 packages call explicitly,
  as `gguf_k_gemv` already does via `register_gguf_k_gemv_kernels`.
- **`20260929T`: the layer2 gate's DEFAULT prompt suites cannot engage any GR
  route, and this is the second half of the blocker above.** All four GR
  selectors (`_qwen4_exp_gr_up_sigmoid_mean_enabled`, `_qwen4_exp_gr_iu8_enabled`,
  `_qwen4_exp_gr_wide_enabled`, `_qwen4_exp_gr_staged_enabled`) require
  `rows > 256`, but `qwen4exp_layer2_profile_gate.py`'s default suites
  (`mtpbench-code-general-ja.jsonl` plus `gdn-prefill-category-heldouts.jsonl`)
  produce 18 prompts of **39-71 tokens**, so the prefill is a single 39-71 row
  call and every GR route declines. Measured: the first `production_gr_up_staged`
  run returned `candidate_dispatch_calls: 0` with `measurement_valid: false` and
  `status: failed` while every individual quality gate read `true` and all 18
  trajectories were bit-identical - a green-looking artifact describing a
  comparison of one configuration against itself. **The gate is fail-closed and
  caught it**: `measurement_valid` requires `counter["calls"] > 0` whenever the
  candidate declares a dispatch target, so the false pass was refused rather
  than certified. The fix is to pass the long-form suite as well:
  `--prompts benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl`
  (438/508/424/512 chat-formatted tokens covering all four required categories,
  written for exactly this row floor). Consequence for the record: the two
  pre-existing GR candidates (`production_gr_up_restore`,
  `production_gr_up_wide`) have **no gate artifact in `benchmarks/results/`**, and
  a default-suite run of either would be a control for the same reason - so
  neither route has measured numerical qualification from this gate.

## Qualified Chunk Tradeoffs

Cleanup status (2026-10-25, R0022): no open cleanup. The tradeoff and selection
instructions this section used to carry were superseded on 2026-10-04 by the
prefill capacity guard, so production no longer defaults to a fixed 1024 chunk.
`_bind_default_chunk` in `hipengine/generation/qwen4_exp_profiles.py` binds the
guard's `admitted_prefill_chunk_size`, and the guard in
`hipengine/generation/qwen4_exp_gguf.py` takes the largest rung of
`_AUTO_PREFILL_CHUNK_LADDER = (4096, 2048, 1024)` whose admission still yields
the requested context. Guard ownership, its revert lever and its removal
condition are in "Prefill chunk capacity guard"; they are not repeated here.

- An explicit `prefill_chunk_size` is passed through and bound verbatim, and the
  guard runs only when the chunk is unset, so 2048 and 4096 remain reachable per
  construction. The 2048 path passed canonical numerics, active4K tasks,
  full-payload boundary/reuse and the detailed/deferred-inspection c2 gates
  (`benchmarks/results/2026-09-14-journey-chunk-admission/`); 4096 passed the
  780-row, full-payload boundary, active-task and c2 gates after the
  chunk/context-derived scratch accounting repair
  (`benchmarks/results/2026-09-15-chunk4096-accounting/`).
- The 2048 and 4096 rates in those packets come from separate runs against
  separate baselines, so they are not a controlled 2048-versus-4096 comparison
  and no difference between the two can be claimed from them. The production
  promotion measurement and its bit-identical token ids are in
  `benchmarks/results/2026-10-04-qwen4exp-prefill-chunk-4096/`.
- Shape-dependent chunk selection, as distinct from the device-dependent
  admission that shipped, remains a product decision rather than recorded
  cleanup. No runtime flag exists for it; the explicit factory setting is the
  available lever.
- Historical: the routed-MoE half of the chunk lever, including the correction
  of its projection, is recorded in `benchmarks/README.md` and
  `benchmarks/results/2026-09-22-moe-main-kernel-cost/`.

## Withdrawn Routes That Stay Bound In The Production Plan

Cleanup status (2026-10-25, R0023): membership and binding precedence are
corrected below. No runtime cleanup remains in this section; individual routes
keep their recorded repair or removal conditions. `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`
in `hipengine/generation/qwen4_exp_profiles.py` holds 12 flags, and the sweep
that zeroes them runs for the `gguf_ud_q4_k_xl` quant, whose production and fast
plans share this binder. A binding plus a sweep entry means "the plan says on,
the sweep says off".

Three paths re-open a swept flag, or keep a flag out of the sweep entirely:

- `PRODUCTION_Q8_QSA_RESTORED_FLAGS` is applied immediately after the sweep and
  restores `Q8_0_SELECTED_WMMA_DOWN`, `QSA_H256_WAVE_PREFILL`,
  `QSA_ORDERED_DECODE`. Prefill head grouping and ordered-v2 decode selection
  are published separately as runner-local state by the resolved profile binder.
- `_F16_ROUTE_SELECTORS` is applied after both passes, so the F16 composition
  restores `GR_WIDE` on the same plan.
- `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL` is bound `"1" if production` and
  is deliberately absent from the sweep, because the 2026-10-08 tiled16
  admission passed the layer2 envelope on the UD lane. Its remaining cleanup
  is process transport: publish this gate with the runner's resolved profile
  state, migrate serial-owner diagnostics to scoped controls, and preserve the
  columnwarp/tiled and strict serial branches. Clear it with profile/forwarding
  tests and public generation that confirms both the default and diagnostic
  parent routes actually run.

The following route choices stay held off there, with their recorded withdrawal
reason and reopen condition:

| Selector | Withdrawn for | Reopen condition |
| --- | --- | --- |
| `PRODUCTION_MOE_PREFILL`, `Q4_IU8_PREFILL` | perf in the disabled-family sweep, then rejected on the task gate (15/18) | a mechanism that fixes the three divergent prompts; a faster tiling does not clear it |
| `GR_IU8` | accuracy (mean KL 1.40e-3) | a correction whose prefill-boundary KL is below ~5e-9 |
| `GR_IU8_DOWN` | accuracy (mean KL 1.31e-3) | a further ~2x prefill-KL reduction |
| `Q8_MMQ_PREFILL` | failed canonical depth | operand localization showing which role carries it |
| `runner.gdn_peer_prefill_layers` | perf, inside run-to-run spread | a complete-owner mechanism |
| `Q4_DP4A64` | isolated long-context decode discrepancy | both decode gates named in "2026-08-29 Qwen4Exp late-layer production prefill" |
| `QSA_FLASH_PREFILL` | part of the failed recovery composition | none recorded |

`tests/test_unit_qwen4exp_arithmetic_recovery.py` pins the sweep membership and
the post-sweep flag values and runner-owned peer scope, and
`tests/test_unit_qwen4_exp_profiles.py::test_production_binder_publishes_the_f16_routes_and_strict_retracts_them`
pins the F16 re-admission. These tests cover binder behavior, not the prose here.

- **These bindings are records of intent, not debt.** Each held-off route was
  withdrawn *pending a named reopen condition*, so keeping the binding records
  the plan the recovery overrode and makes re-admission a sweep edit plus a gate
  run. Removing the bindings would erase that record.
- **The failure mode to guard, learned from `Q8_IU8_WMM` (`20260930T`):** when a
  route is **rejected** rather than withdrawn-with-a-condition, the binding stops
  being a record of intent and becomes a contradiction. That route was rejected
  by role isolation (14 of 17 roles claim and all 14 fail the canonical envelope),
  yet still read `"1" if production`, so narrowing the sweep would have silently
  re-enabled arithmetic the project had already ruled out. The route and its flag
  are removed; the evidence is
  `benchmarks/results/2026-09-30-q8-iu8-role-isolation/` and worklog
  `20260930T092825`.
- **Removal condition:** remove a production binding when the recorded
  disposition rejects the mechanism with no remaining repair condition, or
  when its named reopen condition has been exhausted and the disposition calls
  for removal. A failed candidate evaluation alone is not that decision. The
  MoE packet below explicitly replaces its performance-based reopen condition
  with a task-quality repair condition; its binding stays suppressed pending
  that repair and the applicable production gate. Do not restore it merely by
  narrowing the recovery sweep.
- **No re-admission is claimed here.** A passing numerical envelope is not
  admission: the `production_moe_restore` candidate, which binds
  `PRODUCTION_MOE_PREFILL` together with `Q4_IU8_PREFILL`, passes every category
  and still fails the task gate, so the recorded disposition corrects its reopen
  condition rather than re-admitting it
  (`benchmarks/results/2026-09-30-qwen4exp-moe-restore-rejection/`). The
  2026-10-04 disabled-selector ranking likewise concluded that none of its arms
  can be re-admitted on its evidence
  (`benchmarks/results/2026-10-04-qwen4exp-disabled-selector-ranking/`).

## Isolated GR-Up IU8

Cleanup status (2026-10-25, R0026): no runtime cleanup. `GR_IU8` still binds
`"1" if production` in `hipengine/generation/qwen4_exp_profiles.py`, but
`PRODUCTION_ARITHMETIC_RECOVERY_FLAGS` zeroes it for `gguf_ud_q4_k_xl`, and
`gguf_q4_k_m` production falls back to strict, so the flag is off in every
production binding. The binding stays as the record of intent described in
"Withdrawn Routes That Stay Bound In The Production Plan".

- The unchanged `GR_IU8` restoration fails the 780-row production numerical
  gate independently: mean KL 1.402e-3, p95 7.402e-3, top-1 772/780. All 6,912
  direct calls are K320/N10240. Evidence:
  `benchmarks/results/2026-09-15-journey-gr-up/`.
- Localization is done (2026-09-22): the drift enters only at the Q8_0
  projection boundary (1-3 fp32 ulp, 1.12e-7 to 3.69e-7 relative L2), and the
  strict projection plus standalone sigmoid and gated-mean kernels reproduce
  the fused parent bit-exactly on actual inputs, so the epilogue is not a
  source. This route's prefill-boundary KL is 1.48e-8 against a ~5e-9 cliff,
  and all 8 of its top-1 flips are decode rows.
  Evidence: `benchmarks/results/2026-09-22-gr-drift-localization/`.
- Repair condition: a correction whose prefill-boundary KL is below ~5e-9,
  screened with one prefill per arm using
  `scripts/qwen4exp_gr_iu8_logits_probe.py` before spending a 780-row run. An
  unchanged flag retry does not qualify. The exact fusions and GR-down are
  separate decisions.
- The selector docstring now records the standalone depth-gate failure and
  repair requirement rather than claiming production qualification.

## Isolated GR-Down IU8

Cleanup status (2026-10-25, R0027): no runtime cleanup. `GR_IU8_DOWN` binds
`"1" if production` and is zeroed for `gguf_ud_q4_k_xl` by the same recovery
sweep; `gguf_q4_k_m` production falls back to strict, so it is off in every
production binding. The binding stays as the record of intent.

- `GR_IU8_DOWN` independently fails the 780-row gate with GR-up, generic dense
  and MMQ off: mean KL 1.313e-3, p95 6.835e-3, top-1 769/780. All 6,912 direct
  calls are K10240/N320. Evidence:
  `benchmarks/results/2026-09-15-journey-gr-down/`.
- This is an independent arithmetic failure, not a general prohibition on GR
  optimization; the up leg and the exact fusions are separate decisions.
- Localization (2026-09-22): prefill-boundary KL 4.32e-8 against the ~5e-9
  cliff. Its compensated correction reached 8.48e-9 - the closest any measured
  GR candidate has come, 1.9x above the passing bracket - while leaving the
  gate metric unchanged, which is the measurement that identifies the cliff.
  That correction is removed; its evidence stays in
  `benchmarks/results/2026-09-15-journey-gr-corrections/`.
- Repair condition: a further ~2x prefill-KL reduction (below ~5e-9), screened
  with one prefill per arm before a 780-row run.
- The selector docstring and binder comment now record the depth-gate failure
  and separate repair condition.

## GR-Up Wide Tile (`HIPENGINE_QWEN4_EXP_GR_WIDE`)

Cleanup status (2026-10-25, R0028): the route is not default-off. The
`gguf_ud_q4_k_xl` production plan binds `GR_WIDE=1` through
`_F16_ROUTE_SELECTORS`, which runs after the arithmetic-recovery sweep that
zeroes it; the fast profile shares that binder, and `gguf_q4_k_m` production
falls back to strict, so `GR_WIDE` is on for the `gguf_ud_q4_k_xl` production
and fast bindings, not strict.
The route replaces the GR-up projection with the promoted dense
`dense_wide256_f32_f32_out` tile plus the retained exact sigmoid and gated-mean
epilogue, and engages at `rows > 256` with `low_rank % 64 == 0`. The owner
decision is `e73e6bed1`; admission rests on the complete composition gate, not
on this route alone.

- Measured 4.38x on the projection at the production chunk shape (1.4925 ms
  against 6.5379 ms at rows=1024, K=320, N=10240). The fused kernel is
  latency-bound by its per-output structure, not by weight traffic; the
  2026-10-04 width screen refutes the traffic reading.
- Admission evidence: the 2026-10-14 composition gate passes every global limit
  and every top-1 floor (mean/p95/p99/max KL
  6.493e-4/3.452e-3/7.889e-3/1.186e-2, top-1 99.56%, 15/18 strict-exact, three
  bit-identical repeats). Two scope means (`prefill_last` 1.21e-3,
  `mixed_ja_en` 1.11e-3) are advisories under
  `docs/EXECUTION-PROFILES.md` section 6.1, which binds a scope on its top-1
  floor only. That gate is 450 rows over 18 prompts at chunk 4096, a different
  protocol from the canonical 780-row gate below. Evidence:
  `benchmarks/results/2026-10-14-qwen4exp-fast-composition-gate-2/`.
- The standalone rejection stays on record and is not the current disposition:
  the 780-row gate rejected this arithmetic on its own (mean KL 1.554e-3, p95
  7.081e-3, max 7.349e-2, top-1 768/780, 6,912 dispatch calls, all 12
  mismatches flip-eligible, determinism and lifecycle pass). Keep it; do not
  read it as removal of the composition's route, and do not read the
  composition admission as clearance for a different GR arithmetic change.
  Evidence: `benchmarks/results/2026-09-27-gr-up-wide-tile-screen/`.
- Retired tuning ideas, with their reopen conditions. `ROW_BATCH` 8 at
  `HIDDEN_TILE` 2 measures 0.833-0.844x and `ROW_BATCH` 16 0.430-0.437x, both
  bit-identical; the cause is occupancy (VGPR 72/130/256, 28/15/8 waves per
  CU), so reopen only on a machine class where 130 VGPR still buys 32 waves/CU,
  or with a body change that lowers the accumulator cost. Dropping the dead
  `gate` store measured 0.9797x (2% slower) and is closed. `HIDDEN_TILE` 4 is
  unbuilt and is not a traffic lever. The 2026-09-06 `RB8/HT1` retile is
  confounded - it changed `ROW_BATCH` and `HIDDEN_TILE` together - and is
  superseded by the clean width screen. Evidence:
  `benchmarks/results/2026-10-04-qwen4exp-gr-row-batch-width/` and
  `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md` section 4D HC-1/HC-3.
- **The prefill-boundary KL screen is not a predictor.** This route screened at
  5.17e-11 - 285x *better* than the rejected iu8 route's 1.48e-8 - while
  carrying 300x more projection drift (4.7e-5 against 1.1e-7 relative L2), and
  the standalone gate failed at the same plateau. The publication row's
  ~22-logit argmax gap makes the softmax one-hot, so KL is blind to logits that
  genuinely moved (median absolute drift 0.095, max 1.20). Screen a GR
  candidate with the canonical-fixture depth gate, not this KL.
- The gate's default prompt suite is 39-71 tokens, below these routes'
  `rows > 256` admission, so a default-suite run never executes the candidate
  and reads as a perfect pass; only the harness's independent dispatch count
  exposes it. Use `qwen4exp_q8_repair_depth_gate.py`, or the layer2 gate with
  `--prompts benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl`, for any
  GR route.
- No runtime cleanup remains. The measured-dead `rowbatch8`, `rowbatch16` and
  no-gate wrapper arms stay in `gguf_k_gemv.py` as screen arms (the no-gate arm
  is referenced by `scripts/qwen4exp_gr_projection_screen.py`). The selector
  docstring records the composition's selection rather than the obsolete
  prefill-boundary screen prerequisite.

## Router ordered DPP production owner

Cleanup status (2026-10-26, R0029): the ownership claim this section used to
make is superseded, and no runtime cleanup follows from it. The heading records
the route's DPP history; the current `gguf_ud_q4_k_xl` production owner of
`router_logits/qwen4exp_multirow_f32_router` is `f32_hidden_f16_wmma_gemm`, the
small-matrix F16 WMMA GEMM. `f32_hidden_token_tile4_dpp_exact` is the
arithmetic-recovery plan's default producer and the named rollback; strict and
`gguf_q4_k_m` production keep the tile4 parent. The selector comment, the
selector docstring and the runner's launch-time docstring now say so. The
reader stays, because the gate harnesses drive their arms through it.

Precedence at HEAD, in the order that decides which producer runs:

- The plan is built when the profiles are registered, and the selector is read
  at that point. `_f16_route_selections` in
  `hipengine/generation/qwen4_exp_profiles.py` starts from
  `_production_selections(recovery=True)`, whose router row is
  `_ROUTER_UD_LOGITS_SELECTED_VARIANT` - ordered DPP when nothing is named -
  and replaces DPP with the F16 GEMM unless the caller named a producer before
  the module was imported. A named producer replaces the F16 GEMM in the
  production plan only; the strict plan keeps tile4 either way.
- The production binder then publishes the resolved manifest row into
  `HIPENGINE_QWEN4_EXP_ROUTER_VARIANT`, so the runner's launch-time reader
  starts from the same name the manifest declares. That is the same
  publication channel that keeps strict from inheriting a production override.
- At launch, `_qwen4_exp_router_producer_variant` in
  `hipengine/runtime/qwen4_exp_runner.py` takes the linear GGUF path below two
  rows, the published producer from two rows, and ordered DPP instead of the
  F16 GEMM below 64 rows, because the GEMM stages 128-row tiles. That width
  fallback is the only place the running producer differs from the manifest
  row. An empty variable falls back to tile4, which a bound run reaches only by
  clearing the variable after the bind.

- The DPP producer's own record stands and is not a production claim: complete
  operation screen improves 1.135–1.172x on layer0/22/47 weights; canonical
  model prefill improves +0.553%/+0.486%/+0.589% at p512/1K/4K. D128
  full-state/KV, 576 natural-router calls and four GPU row-isolation/ownership
  tests pass. Its raw-pointer/selection ABI and per-thread FMA/reduction
  association are unchanged: one LDS publication, register inter-wave folds,
  DPP finish. Exact tile4 remains registered as strict fallback.
- `HIPENGINE_QWEN4_EXP_ROUTER_VARIANT` is the named rollback and bisection
  lever; the rollback set is the three non-production names
  `f32_hidden_token_tile4_dense_exact`, `f32_hidden_token_tile4_dpp_exact` and
  `f32_hidden_warp_reduce_dense_exact`. The recorded "remove the launch-time
  reader after one production checkpoint" condition was overtaken by the F16
  composition: the reader is now both the named rollback and the binder's
  publication channel.
- Removal condition: a plan-level producer-selection API the gate harnesses
  can drive (`scripts/qwen4exp_layer2_profile_gate.py`,
  `scripts/qwen4exp_profile_gap.py`, `scripts/qwen4exp_row4_state_gate.py` and
  `scripts/qwen4exp_halo_box_campaign_ab.py` set the variable directly today),
  then delete the launch-time reader and the binder's publication of it.
  Replacing the reader without that API would leave those arms measuring the
  production producer twice, so it is a design decision rather than cleanup.
- The historical warp-per-token sibling remains registered for diagnostics,
  unselected by default; its failed arithmetic gate does not qualify it.
  The DPP producer replaces neither its reduction association nor selection
  policy. Original model-wall evidence preserves a p4096 decode outlier and
  does not claim decode-speed or all-case request-wall non-regression.
- Evidence: `benchmarks/results/2026-10-05-qwen4exp-router-dpp/README.md`.

## Router Producer Variant (`HIPENGINE_QWEN4_EXP_ROUTER_VARIANT`)

Cleanup status (2026-10-26, R0030): no runtime cleanup. The closing ownership
sentence is corrected; the rejection record and its campaign-level reopening
condition stand. The reader-removal decision belongs to R0029 above, not here.

- Historical warp screening: `f32_hidden_warp_reduce_dense_exact` uses one
  warp per token, a 32-lane shuffle reduction and one expert row staged per
  block in LDS. Its comparison parent was `f32_hidden_token_tile4_dense_exact`.
  The current UD production producer is `f32_hidden_f16_wmma_gemm`; ordered DPP
  is the arithmetic-recovery default and the named rollback, as documented in
  the preceding section. The runner reads the selector at launch, and the
  profile binder publishes the resolved selection.
- Screened at `+1.65%` prefill on `code-p4096` (247.739 -> 251.823 tok/s,
  `-0.272 s`), with the router kernel itself `2.158x` faster
  (619.45 -> 287.11 ms). Same-session two-arm A/B, disjoint per-run spreads.
  Evidence: `benchmarks/results/2026-09-27-router-warp-reduce/`.
- **Not bit-exact**: the reduction order changes from a 256-thread LDS tree to a
  32-lane shuffle, so the two variants are distinct production arithmetics. The
  promotion bar is the production envelope in `docs/EXECUTION-PROFILES.md`, not
  parent parity. The tile4 parent stays registered as the strict fallback.
- **The risk mechanism is discrete, not drift.** The router logits feed
  `qwen35_router_select`, so a near-tie expert can change which of the 512
  experts is selected. A 4.9e-7 logit difference is irrelevant as drift and
  decisive as a selection flip. Judge this route on expert-selection stability
  and the model-logit envelope together; a KL-only reading can miss it. This is
  the "confirmed router-flip mechanism" the GR section below asks for, applied
  to the router itself rather than to a GR projection.
- **The 450-row production gate rejected the route.** `measurement_valid: true`,
  `candidate_dispatch_calls: 4320` (so it ran; this is not the `rows > 256`
  silent no-op). Numerical FAILED, task FAILED, state/determinism and lifecycle
  passed. Mean KL 5.714e-4, p95 2.322e-3, p99 6.228e-3, max 3.344e-2 - all four
  KL gates pass, at 57% of the mean budget - while **top-1 is 98.67% against a
  99% floor**. Scope failures: `category/mixed_ja_en` (mean KL 1.083e-3),
  `shape/prefill_last` and `transition/prefill_to_c1` (mean KL 3.515e-3).
- **The cause is the router's discrete selection, not drift.**
  `flip_eligible_rows` 48, `top1_mismatches_flip_eligible` **6**,
  `top1_mismatches_outside_flip_eligible` **0**, `max_abs_logit_delta` 1.673.
  Every top-1 mismatch sits on a row whose strict margin is small enough for the
  observed drift to explain it. The router logits feed `qwen35_router_select`, so
  a 4.9e-7 difference flips which of 512 experts is selected and becomes an O(1)
  model-logit change. A passing KL envelope alongside a failing top-1 is the
  signature of a discrete amplification node.
- **Tuning the router kernel cannot clear this gate.** Any reduction-order change
  moves the logits by ~1 ulp and flips the same near-ties. This also explains why
  the GR family below "is closed to arithmetic change": a router flip changes
  which experts run, so every downstream GR route inherits the same saturation.
  Treat the router selection as the amplification node for the whole model's
  arithmetic-change design space.
- **Reopening path, campaign-level.** Make the *selection* invariant instead of
  the arithmetic: rank experts on a quantized logit with a deterministic
  tie-break, so a 1-ulp difference cannot reorder them. That must apply to the
  reference too, so it re-baselines strict and needs a campaign decision, not a
  kernel change. Reopen this route only alongside that decision; do not spend
  another kernel attempt on it before then.
- Historical screening note: the selector also accepts the ordered DPP
  producer described above. UD production selects the F16-operand GEMM
  producer, DPP is the recovery plan's default and the named rollback, and the
  failed warp route remains opt-in and diagnostic only. Remove the reader under
  the plan-level selection API condition recorded in the preceding section, not
  by treating all named siblings as one failed candidate.

## MoE Decode warp256 GEMV Pair - REMOVED 2026-09-30

> 2026-10-14: the two kernels this section retained are selected again for
> the `gguf_ud_q4_k_xl` production plan after passing the layer2 gate; see
> "Qwen4Exp decode routes of the UD production plan". The re-admitted
> selectors were `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP` and `_DOWN`.
> Those controls are now runner-owned; gate/up still requires the separately
> enabled DP4A path, which the current production recovery plan disables.
> The bare `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP` selector stays deleted.
> The text below is the 2026-09-30 record.

- **Both runtime routes and both flags are DELETED**
  (`HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP` gate/up and `_DOWN`). The removal
  condition recorded here on `20260930T` was "localize the flip and test a
  narrower scope; promote and delete the flag if a scope clears the envelope,
  delete the flag and the route if none does." **No scope clears the envelope at
  an acceptable cost**, so the route is deleted and the rejection is recorded.
- **The scope table that decided it** (full 18-case suite, both arms vs strict,
  1494 rows, one harness; the production default is the 43-layer scope):

  | scope | envelope | net cost |
  | --- | --- | --- |
  | 43L (production) | fails max KL **0.261188**, 5.2x over | baseline |
  | 48L (all) | fails mean+max (max 0.128120) | - |
  | 35L | fails mean+p99+max (max 0.104060) | - |
  | **30L** | **passes every bar, 1.84x-3.10x margin** | **+1.426 ms/step, +3.10%** |
  | 17L | fails max (0.115957) | - |

- **The 30-layer scope is the only passing one, and it is not adopted.** It
  costs a **measured +1.426 ms/step = 3.10% of decode kernel time** (46.061 ->
  47.486, per-kernel join), while the production default already passes the
  **admission** gate with 4-8x margin (mean 2.44e-4 vs 1e-3, max 6.14e-3 vs
  5e-2, top-1 0.9911, determinism 3/3, clean isolation). The 18-case suite that
  the production default fails is a stress fixture already recorded as unable to
  rank arms.
- **The warp is not what costs time.** At a fixed 30-layer scope the warp
  *saves* **0.493 ms/step** against the same scope without it; the +1.426 is the
  price of the scope reduction. So there is no configuration where the warp pays
  for the numerics it needs.
- **The gate that had retained this flag never covered the failing case.** Its
  docstring cited "996 shared-chain teacher-forced rows (all bars pass with
  2.6-4.4x margin)", but that fixture does not contain `heldout_mixed_review`,
  which is exactly where the warp fails. A gate that omits the failing case
  cannot clear it. The ledger's own action cell had predicted this gap.
- **CLOSED `20260930T` - the unread-env-var obligation is discharged.** All four
  scripts named below are fixed, plus a fifth the obligation did not list.
  `scripts/qwen4exp_moe_decode_warp_logits_probe.py` and
  `scripts/qwen4exp_moe_decode_warp_state_gate.py` are **deleted**; the generic
  helpers the probe happened to host (`hip_available`, `_forced_logits`, `_kl`)
  moved to `scripts/qwen4exp_canonical_ar_bench.py`, which both surviving
  importers already used. `scripts/qwen4exp_row4_state_gate.py` and
  `scripts/qwen4exp_halo_box_campaign_ab.py` lost the `moe-decode-warp` and
  `moe-decode-warp-pair` route packages from their `choices`, their flag
  lookups, their `_DOWN` environment writes and their chunk-size sets - both had
  **no engagement guard**, so those arms would have silently measured the
  incumbent. **The fifth:** `scripts/qwen4exp_r2e_route_ablation.py` imported
  `FLAG`/`FLAG_DOWN` from the deleted probe and set both to `"0"` in its
  incumbent baseline; those writes are gone (they were no-ops once the flags
  stopped being read). Those stale writes are removed; the surviving
  `_GATE_UP`/`_DOWN` selectors in `scripts/qwen4exp_layer2_profile_gate.py`
  arm the re-admitted pair and are not stale writes. The kernel
  definitions stayed registered through the removal (both are registered for
  `hip_gfx1151` as `selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out`
  and `selected_weighted_sum_warp256_fast_bf16_bf16_out`) and the UD production
  binder selects them again.
- **One harness the obligation did not list was retired rather than fixed.**
  `scripts/qwen4exp_de1_layer_bisect.sh` drove the deleted logits probe as a
  subprocess, so it became a dangling reference. Its subject - the warp256
  gate/up max-KL flip - went away with the route, and even with the probe
  restored its arms would select the gate/up route through the removed
  `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP` variable and
  **silently measure the incumbent, reporting a clean bisect.** It therefore
  exits 2 with the reason in its header, and the rest of the file is retained as
  documentation of the retired experiment. The measurement it produced is
  immutable in worklog `20260930T023729`. Revisiting the warp kernels needs a
  **new** harness that selects them directly; restoring that file unedited would
  restore the silent mismeasurement.
- **The stale-owner obligation is also discharged `20260930T`.**
  `scripts/qwen4exp_de1_moe_per_kernel.py` hardcoded the gate/up owner as
  `gguf_q4_k_selected_dual_silu_logical128_t64`, which stopped matching once the
  dp4a variant became the production default; the marker now accepts either era
  (`GATE_UP_MARKERS`, dp4a first) and `find` reports which owner it charged.
  Verified against the existing pre-promotion capture, reproducing the recorded
  14.013 ms/step, 63.43 GB/s, 28.7%-of-roof and 10.091 ms/step headroom exactly.
  The failure was loud (`SystemExit`), not silent, but it made the tool unusable
  against any post-promotion capture. **The general lesson:** a measurement tool
  that matches kernels by name must tolerate the names a promotion changes, or
  it silently becomes single-era.
- Evidence: `benchmarks/results/2026-09-30-qwen4exp-moe-decode-warp-gateup-blocked/`,
  `benchmarks/results/2026-09-30-qwen4exp-de1-dual-vs-strict/`, and worklog
  `20260930T031044` (the measured scope table).

## Unconsumed GR Gate Publication

- `Qwen4ExpGRReadDeviceResult` has four fields and only `normalized`, `mixed`
  and `inject_logits` are consumed: no reader of `.gate` exists in the runner,
  the scripts, or the tests. The fused GR up kernel nevertheless stores the full
  `rows x branches x hidden` f32 gate plane - 167.8 MB per read at 4096 rows,
  roughly 43% of that kernel's per-read traffic - as a pure publication.
- The store is `gate[index] = sigmoid_gate` in
  `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32_kernel`
  (`hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip`). The strict unfused
  path still needs the plane: its sigmoid and gated-mean epilogue consumes it.
- A runtime null check is not viable - HIP rejects a null kernel pointer
  argument with `hipErrorInvalidValue` (measured), and the launcher already
  validates `gate != nullptr`. The removal path is a template instantiation
  (`template <bool WAVE_SCALE, bool WRITE_GATE>`) with a new `Q8_GR_SCALE_ENTRY`
  symbol plus a wrapper; the status below records what actually landed.
- Remove the publication once such an instantiation lands and no diagnostic
  still reads the plane; size the win with a kernel microbenchmark first, since
  the kernel achieves only ~15 GB/s effective on this shape and is therefore not
  bandwidth-bound. Context:
  `benchmarks/results/2026-09-22-gr-drift-localization/`.
- **Status 2026-10-04: the instantiation landed and the microbenchmark was run; the
  store stays.** `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32_kernel`
  now takes `WRITE_GATE` as a template parameter, and
  `Q8_GR_SCALE_ENTRY(..., false, false, 4)` exports the single
  `WRITE_GATE=false` instantiation - `ROW_BATCH` 4, no wave scale - as
  `hipengine_gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_no_gate_f32`,
  with a Python wrapper and a GPU test
  (`tests/test_gpu_qwen4exp_gr_no_gate.py`). **No registry key
  exists for it**: the variant is a direct-call export rather than a dispatch
  variant. Its callers
  are that test and `scripts/qwen4exp_gr_projection_screen.py`; the fused GR up
  the runner dispatches resolves through the registered
  `coltile2_branch4_rowbatch4_f32_exact` key, which is the gate-writing parent.
  Measured: `mixed` is bit-identical at four shapes, and the variant is
  **6.6551 ms against 6.5198 ms, 0.9797x, a 2% regression** (ledger `HC-3`). So the
  42 MB/layer publication is retained deliberately, and the no-gate export stays
  available as the strict comparison rather than as the default. The "not
  bandwidth-bound" framing above is now measured rather than inferred: widening
  this kernel's row reuse to halve its weight re-read is 16% *slower* (ledger §7D).
- Removal condition: none. Keep both instantiations - the gate-writing one as the
  production default and the no-gate one as the strict comparison - and do not
  re-open the store removal without a differently-scheduled body.

## GR-Up Widened Row Reuse Exports (2026-10-04)

- `gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch8_f32`,
  `..._rowbatch8_wave_scale_f32` and `..._rowbatch16_wave_scale_f32` are three
  retained exports of `ROW_BATCH` 8/16 instantiations of the same kernel body. They
  are bit-identical to the `ROW_BATCH` 4 parent at every shape and **measured slower
  at every shape** (0.833-0.844x and 0.430-0.437x against the production wave-scale
  parent, occupancy-bound: 130 and 256 VGPR against the parent's 72, i.e. 15 and 8
  waves per CU against 28). Evidence:
  `benchmarks/results/2026-10-04-qwen4exp-gr-row-batch-width/`.
- They are deliberately **not registered**: a registry key would make a
  measured-dead path reachable from dispatch, and
  `tests/test_gpu_qwen4exp_gr_row_batch8.py` scans the whole registry to assert that
  no key resolves to one of them.
- Removal condition: drop the three exports, their wrappers, the `_row_batch`
  parameter and this section together, once the ledger's §7D reopen condition is
  either met or formally retired - i.e. when a machine class with enough registers
  per wave has been measured, or when the GR up cost model has been replaced and the
  retile no longer needs a checked starting point.

## Q8 Block-Scale Restoration

- UD-Q4_K_XL now selects guarded block-scale WMMA through the profile-owned
  `HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT`. Raw integer codes avoid FP16-rounded
  weights; an empirical screen queues sparse strict-order corrections.
  Full short/depth gates and the shared-graph model comparison qualify the
  restoration. QSA-prefill grouping and ordered decode are restored too.
- The unguarded entry is not diagnostic-only. `selected_grouped_wmma_prefill_bf16_bf16_out`
  is the registered generic owner, the profile binder's
  `HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT` default and the launcher's fallback when
  that variable is unset. UD-Q4_K_XL's restored flags override it with the
  guarded variant, so no plan binds the unguarded one for UD production today;
  `gguf_q4_k_m` production is recorded as numerically unqualified and resolves
  to strict, which keeps `Q8_0_SELECTED_WMMA_DOWN` off and runs the per-expert
  selected GEMV. Remove the environment selector when profile plans supply
  variant keys directly; retain the registered strict fallback and other-quant
  legacy path.
- Dense MMQ and dense/GR iu8 restoration attempts fail canonical numerical
  gates and remain disabled: UD-Q4_K_XL's arithmetic-recovery sweep zeroes
  `Q8_MMQ_PREFILL`, `GR_IU8` and `GR_IU8_DOWN` and the restored flags do not
  re-enable them. Do not infer their admission from the corrected
  selected-down kernel or from short-prompt gates.

## UD-Q4_K_XL Arithmetic Recovery (2026-09-14)

Cleanup status (2026-10-26, R0036): the actionable request in this section is
done, and what remains is a closed historical record.
`docs/EXECUTION-PROFILES.md` §2.13 now describes the `gguf_ud_q4_k_xl` F16 route
composition, the 2026-10-14 composition gate and the quarantined `gguf_q4_k_m`
plan, so its product-policy framing is current at HEAD. Statements below that
were false at HEAD are corrected in place: selectors have been re-admitted
since the recovery, tiled GDN is no longer disabled, the hook-removal
obligation is discharged, and the recovery-cost record moved out of §2.13. The
binder is unchanged; the held-off set and each reopen condition live in
"Withdrawn Routes That Stay Bound In The Production Plan" above.

- The September14 recovery selected conservative arithmetic while
  retaining the exact grouped/iu8-repair, register-state GDN, wave-reduction
  and mapping-only PLE owners. The prior composition fails the current
  numerical envelope; disabling only Q8-down/MMQ passes automatic numerical
  limits but introduces a factual task regression in the paired review.
- Conservative recovery matches strict on594 natural-prompt rows and780
  canonical512/1K/4K rows, with repeatability, state and teardown gates.
  This is a measured recovery, not proof that every disabled family is faulty.
- The subsequent Q8/QSA restoration above supersedes those portions of the
  rollback. The families still off at HEAD are approximate MoE prefill, Q4 iu8
  prefill, Q8 MMQ prefill, the GR iu8 up/down pair, GDN peer prefill, Q4 DP4A64
  and QSA flash prefill. Two further families were reopened after the recovery
  and are not rollback debt: tiled GDN through the 2026-10-08 column-warp
  admission, which the 2026-10-14 composition binds to the row-split
  recurrence, and `GR_WIDE` through the composition itself. Re-enable a
  still-off family only with a complete same-suite numerical/task/depth gate
  and accurate manifest. No production override is added to bypass the
  recovery binder.
- The existing GDN/DPP suffix was retested over corrected Q8/QSA production:
  short/depth numerics pass, but a complete Japanese explanation introduces
  a material verification-parallelism error under the predeclared task rule.
  See `benchmarks/results/2026-09-15-journey-gdn-restoration/`. Keep that
  composition off; this is not proof that all GDN variants are faulty.
- DPP itself is exact relative to its tiled parent and is not removed.
  Its suffix is inactive: the recovery pass replaces the tiled GDN owner and
  the 2026-10-14 composition binds the row-split recurrence instead.
  Requalify that complete composition before restoring it; do not infer DPP
  caused the failure.
- Diagnostic layer/shape hooks live only in scripts. The three that remain
  clear both graph caches between arms - `scripts/qwen4exp_journey_localize.py`
  (`clear_arm_graphs`), `scripts/qwen4exp_conservative_cost.py` and
  `scripts/qwen4exp_prefill_restore_ab.py` - and the last two also clear the
  host dispatch cache. No production hook exists. These are the harnesses that
  reproduce the recovery evidence, so they stay; a new diagnostic must clear
  both cache layers or it measures a warm incumbent.
- **Per-selector cost is now measured, and it does not match the recovery's
  headline cost (2026-10-04).**
  `benchmarks/results/2026-10-04-qwen4exp-disabled-selector-ranking/` ran the
  nine disabled selectors individually and all eight prefill ones together on
  `code-p4096` (source `aa49c0850`, 2 measured reps, `accelerator-performance`):
  baseline median prefill **15.544 s**, `GR_WIDE` **+1.841 s (1.1344x)**,
  `GR_IU8`+`GR_IU8_DOWN` **+1.717 s (1.1241x)**, `GR_IU8` +1.590 s,
  `GR_IU8_DOWN` +0.146 s, and every other selector a loss or inside +/-0.4%
  (`Q8_MMQ_PREFILL` -0.539 s, `PRODUCTION_MOE_PREFILL` -0.550 s,
  `Q4_IU8_PREFILL` -2.822 s). **Enabling all of them at once is worth +1.7%.**
  The recorded recovery cost is `-40.94/-41.89/-54.53%` prefill
  (`benchmarks/results/2026-09-14-q8-prefill-numerics/README.md`, its "Recovery
  Cost" table; §2.13 no longer carries it). The two are not the same
  measurement (the cost packet is weighted p512/p1024/p4096 against the
  unrecovered UD composition in one residency; this is one case under
  post-binder overrides on the recovered UD plan), so the discrepancy is
  **unresolved**, but the recovery baseline itself improved 1.52x since the
  2026-09-16 sweep (23.648 s -> 15.544 s) and the recovery profile's own
  optimization appears to have closed most of the gap. §2.13 has since been
  re-derived at HEAD. Of the three material wins, `GR_WIDE` is now admitted by
  the 2026-10-14 composition and the GR iu8 pair stays off on its own repair
  condition; the other six arms were losses or inside run-to-run noise, and the
  GDN column-warp family was later admitted on the 2026-10-08 gate rather than
  on this ranking, while QSA flash prefill remains held off. The wide tile's
  bit-exact retile is still the lever that does not depend on re-admitting
  arithmetic.

## Disabled-Selector Sweep Harness Duplication (2026-10-04)

Cleanup status (2026-10-26, R0037): the duplication is real and both harnesses
are still present. The removal condition below is unmet, and the choice it
describes is a record-retention decision rather than an engineering blocker:
both 2026-09-16 artifacts still name
`scripts/qwen4exp_disabled_family_sweep.py` as their driver, so it remains their
reproduction path and stays. The arm sets drifted since this section was
written: the family sweep's `Q8_IU8_WMM` arm went with the rejected generic
dense Q8 iu8-WMMA route (`e108d110f`), and the arms the ranking harness lacks
are `Q8_DENSE_WIDE_16_47`, the three `Q8_WMMA_LAYERS_*` scopes, `Q4_DP4A64` and
a no-op override control. Nothing here changes a production route.

- **Two harnesses answer the same question and only one was ever run.**
  `scripts/qwen4exp_disabled_family_sweep.py` produced the 2026-09-16
  recoverable-time artifacts; `scripts/qwen4exp_disabled_selector_ranking.py`
  (written 2026-09-15, with unit guards in
  `tests/test_unit_qwen4exp_disabled_selector_ranking.py`) had never been run
  until 2026-10-04. They carry different arm sets and different report schemas
  (`qwen4exp_profile_gap_window` children vs a `ranking.json` roll-up).
- The ranking harness is the one now used, because it records the per-arm
  `effective_route` and `effective_route_missing_keys` and writes a roll-up
  report; the family sweep writes per-arm JSON only. Retain the family sweep
  while the two 2026-09-16 packets depend on it. Remove it only after those
  packets are no longer cited or a replacement harness covers its extra arms
  and reproduction contract. Do not edit historical driver rows: they record
  what ran.
- **The 2026-09-16 ranking must not be read as an opportunity list.** Its
  fallback was 23.648 s against 15.544 s today, and its ordering inverted:
  `Q8_MMQ_PREFILL` 1.203x -> **0.9665x**, `Q4_IU8_PREFILL` 0.997x ->
  **0.8463x**, the GDN pair 1.010x/1.005x -> 0.9988x/0.9960x. Its top entry
  `Q8_IU8_WMM` (1.388x) is closed independently. The artifact's 2026-09-30
  correction already says the savings are "the difference between the recovery
  profile and the production profile", which is the same warning.

## Journey Multi-Column GDN

Cleanup status (2026-10-26, R0039): the multi-column route is a registered
fallback, not the production default, and no removal follows from that. The
`gguf_ud_q4_k_xl` production binder names `qwen4exp_gdn_rowsplit_prefill` in
`HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT` (`_F16_ROUTE_BINDER_ENV`, applied
after the recovery sweep), and the runner resolves that variant inside the open
column-warp gate. The non-recovery selection table still declares
`qwen4exp_gdn_tiled16_multi_prefill` for the 27-47 window, but the recovery pass
replaces the GDN rows with the wave-norm narrow/wide owners, and `gguf_q4_k_m`
production is quarantined to strict, which closes the gate: multi does not
dispatch in a running production profile. Multi and DPP both stay registered in
`hipengine/kernels/hip_gfx1151/__init__.py`. Do not remove the multi-column
path: no numerical or performance gate rejected it.

- `qwen4_exp_gdn_multi` / `HIPENGINE_QWEN4_GDN_MULTI_COLUMN` and registry
  `qwen4exp_gdn_tiled16_multi_prefill` became the production default on
  `20260930T`: measured 1.34x-1.68x faster than the DPP route it replaced
  across 512/1024/2048/4096 tokens at Hk16/Hv48/Dk=Dv=128 on gfx1151, at NMSE
  ~1.8e-14 (fp32 round-off), with the production numerical gates passing. It
  superseded the DPP route as the production GDN prefill selection; DPP stayed
  registered for bisection and gate arms, and the serial strict owner remains
  the strict fallback. The 2026-10-14 composition superseded this selection
  without rejecting the route. Evidence:
  `benchmarks/results/2026-09-30-qwen4exp-gdn-multicolumn-ab/`.
- The earlier "gate rejected it" premise was a misattribution, which is why the
  removal never fired. The rejection was read from the ledger's section-7 row,
  which combined `R01` (tiled/DPP) and `R02` (multi-column) under `R01`'s
  evidence entry.
  `benchmarks/results/2026-09-15-journey-gdn-multi-restoration/README.md` shows
  `R02` passing the full 594-row short gate (593/594 top-1) and the canonical
  depth gate at every declared limit (mean KL 8.92215e-05, max 4.4548289e-03,
  top-1 777/780), and states that it does not reproduce the single-column
  Japanese error that blocked `R01`. Its own finding is a different prompt
  (`heldout_general_en_quant`) that the artifact scopes as "not proof of worse
  expected quality or a GDN kernel defect".
- Historical old-stack 594-row scope failures remain evidence, but are not the
  current blocker. Over corrected Q8/QSA production, short 594-row and canonical
  780-row numerical gates pass. The complete quantization task introduces a
  materially misleading native-INT8 support explanation; the current
  per-prompt task rule does not admit this composition. See
  `benchmarks/results/2026-09-15-journey-gdn-multi-restoration/task-review.json`.
  One Japanese-plan pair remains truncated; no extension or timing is spent
  after the separate task finding. This is not a numerical rejection or proof
  of worse expected quality. Do not rerun the unchanged composition or promote
  it merely because its numerical envelope passes.

## Journey GDN DPP Experiment

Cleanup status (2026-10-26, R0040): the DPP key is registered and unselected,
and no runtime cleanup remains. The active `gguf_ud_q4_k_xl` production GDN
owner is `qwen4exp_gdn_rowsplit_prefill`;
`qwen4exp_gdn_tiled16_dpp_prefill` is in the recovery pass's inactive set and
stays registered (`hipengine/kernels/hip_gfx1151/__init__.py`) as the layer2
gate's `gdn_isolated` arm and a bisection incumbent. The only interception left
is the diagnostic `scripts/qwen4exp_gdn_multicolumn_route_probe.py`, which
wraps the selection predicate to count dispatches and reproduces the
multi-column engagement evidence, so it stays.
`HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT` is profile-owned; diagnostics use
`gdn_tile16_prefill(runner, False)` to compare the columnwarp parent.

- The `qwen4_exp_gdn_dpp` build family and gfx1151
  `qwen4exp_gdn_tiled16_dpp_prefill` key passed the previous
  UD-Q4_K_XL suffix full 72-sample exact A/B. The suffix is inactive without
  rejecting DPP itself. Ordinary tile16 and serial strict keys remain
  registered; keep them for strict/bisection.

## Qwen4Exp PLE Mapping Advice (2026-09-14 JST)

- gfx1151 production now selects instance-owned mapping-only random advice.
  Strict selects normal; `HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS=normal`
  is the diagnostic rollback. Remove the environment override once remap,
  full-cache warming and long-unique cold behavior have broader qualification;
  preserve the explicit table API and normal fallback.
- Existing `configure_random_access` combines per-gather prefetch/range work
  and advice. It remains a separate explicit diagnostic; do not stack it on
  mapping-only evidence or attribute its cost to the new default.
- Exploratory sorted-dedup/pread/worker algorithms live only in the experiment
  harness. None is an inference fallback chain or adopted default.

## gfx1151 Q4 dual-SiLU prefill 16-column arms: removed (2026-09-13)

Cleanup status (2026-10-26, R0042): resolved and verified at HEAD. No
`col16_row256`/`col16_row512` export, wrapper, registry entry, test, or
`out_tiles_per_block` template parameter remains in `hipengine/` or `tests/`;
the remaining occurrences are this record, its worklog entries, the rejected-arm
artifact, the 2026-09-13 prefill-kernel profile artifact, the benchmark
changelog and the audit sweep inventory. The arms are reproducible only from
the pre-removal source named in the rejected-arm artifact. The open directions
below are unstarted hypotheses, not cleanup debt.

- `dense_dual_wmma_prefill_col16_row256_bf16_bf16_out`,
  `dense_dual_wmma_prefill_col16_row512_bf16_bf16_out`, and the
  `out_tiles_per_block` template parameter they were built on were added,
  measured, and removed inside one day. Same-host A/B on real Qwen3.8-27B
  gate/up tiles at rows 256/512/1024/2048/4096, three harness invocations per
  shape: the parent takes 3.31/6.62/13.12/26.03/52.15 ms, `col16_row256`
  **0.71/0.67/0.63/0.61/0.61x** and `col16_row512` **0.41/0.71/0.69/0.68/0.67x**,
  bit-identical output in all 30 cells. Do not repeat these unchanged variants;
  a different decode mapping or epilogue is a new hypothesis.
- Review correction: neither the original instruction-count ranking nor the
  later doubled-decode-time explanation isolates the loss. In particular,
  col16_row512 has the same block count as the parent at rows512+, not double.
  Source-level counts and effective bandwidth are not issue/traffic counters.
- The screen's occupancy arm (16 columns x 256 rows with
  `__launch_bounds__(128, 4)`) was built, measured, and dropped inside the same
  day: gfx1151 allocates 219 VGPRs with min-blocks 1 and with min-blocks 4,
  literal and template-parameter spellings both. This establishes unchanged
  compiled register counts, not actual residency. The previous 512-VGPR
  occupancy model was wrong: gfx1151 wave32 uses 1536 physical VGPRs and a
  24-register granule. Recompute mode/LDS limits before any register-cap work.
- Open directions that the removal leaves untouched, with their prerequisites:
  a wider column tile (48 columns) needs the decode to reach 192 threads or a
  lane remap, since 128 threads cannot cover 48 decode pairs; 32 columns x 512
  rows per block needs the row-tile limit extended and makes the epilogue
  union 64 KiB. Measure residency or test a register-local-SiLU epilogue.

## gfx1100 has no kernel-registration refresh hook (found 2026-10-12)

Cleanup status (2026-10-26, R0044): resolved by the test baseline, not by a
hook. `tests/conftest.py` imports every `hip_gfx1100` module before snapshotting
the collection-time registry (`pytest_collection_finish`, lines 80-101), so the
teardown restore now carries the lazily imported Gemma 4 owners. The recorded
GPU-visible `--suite all` run on the gfx1100 host passes in one invocation
(26,811 tests, 0 failures;
`worklog/entries/20261023T015203.261216Z-lhl-gpu-suite-health-eb4ac4.md`), and
that run includes `tests/test_gpu_gemma4_decode_graph_capture.py`. The hook is
still absent from `hipengine/kernels/hip_gfx1100/__init__.py`; it is not needed
because no route clears the registry outside tests and the baseline is the
refresh. Do not add a replay hook only to satisfy the withdrawn condition.

`load_backend_kernel_package` refills registrations a test's registry isolation
removed by calling the backend package's `register_backend_kernels(replace=False)`.
`hipengine/kernels/hip_gfx1151/__init__.py` exports that hook;
`hipengine/kernels/hip_gfx1100/__init__.py` does not, because its kernels
register when their modules are first imported. Before the baseline fix, a
second model load in one pytest process failed with
`MissingRuntimeProfilePlanError` once an earlier test's teardown had restored
the collection-time registry; `tests/test_gpu_gemma4_decode_graph_capture.py`
was the file that showed it. Product loads were unaffected; nothing clears the
registry outside tests.

Removal condition (superseded 2026-10-26): the observed failure is gone under
the baseline capture above, and no caller needs a gfx1100 replay hook. If a
future test needs to clear the registry inside its body and reload a model, add
a real replay hook then, with that test as its gate.

## Q8_0 T16 f16w tiles are found through a pointer table (added 2026-10-10)

Cleanup status (2026-10-26, R0045): the pointer table and its registration APIs
are removed. `Gemma4DeviceWeights.ensure_dense_f16w` attaches the converted
operand to its weight as an owned `f16w` sidecar; resident-byte accounting and
weight teardown include it. The memory reserve and conversion remain unchanged.

The registered `t16_wmma_prefill_f16w_bf16_bf16_out` variant uses the
`t16_f16w` launch ABI, passing tiles and the half-weight operand explicitly.
Dispatch selects it by matching owned-operand shape and registered capability;
its cache distinguishes operand presence, pointer and shape. Existing `t16`
callers and weights without a sidecar keep the decoding path. gfx1100 and
gfx1151 retain their respective two-wave and four-wave wrapper policies.

## Gemma 4 expert down WMMA rollback lever (added 2026-10-10)

Cleanup status (2026-10-26, R0046): removed. `HIPENGINE_GEMMA4_MOE_DOWN_WMMA`
and its reader are gone, and the unpinned Gemma 4 MoE prefill takes the grouped
WMMA down owner unconditionally -- `down_wmma = gate_up_wmma or use_mmq or not
pinned` in `gemma4_experts_forward_bf16`
(`hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py`).

The owner is the grouped WMMA down projection, in its Q5_1T16 tile form where
the tensor carries the tiles. It reads BF16 activations and reassociates
against the exact grouped owner, so it is a production-profile route gated under
the section 2.11 envelope; it is not the int8 MMQ down, whose lever and failure
record are separate. Every owner the `0` arm reached is still reachable
unchanged: a pinned `HIPENGINE_GEMMA4_MOE_PREFILL=grouped` (or `selected`) mode
runs the exact grouped owner, and a call whose WMMA tile plan is empty -- fewer
than 16 lanes per expert, or a down weight the WMMA owner declines -- falls
through the same chain to `gemma4_project_experts_rows`.

Both halves of the removal condition are on file. The numbers are in
`benchmarks/results/2026-10-10-gemma4-gfx1100-expert-down-wmma-promotion.json`
(section 2.11 packet, 18 cases / 1,134 rows, all bars passing; public prefill
1253.2 -> 3164.7 tok/s at 512 with generated ids equal at every length), and a
llama.cpp-paired W7900 production-profile ladder that includes the route is
`benchmarks/results/2026-10-15-gemma4-gfx1100-llamacpp-latest-comparison.json`
(back-to-back per length, prompt ids sha-equal). The route shipped in `v0.8.0`
and survived `v0.8.1`, `v0.8.2`, and `v0.8.3` with no recorded rollback.

Removing the lever also removed the only *isolated* strict arm for this
candidate: `HIPENGINE_GEMMA4_MOE_PREFILL=grouped` gives the exact down owner but
also changes gate/up routing, so it measures a different candidate. The packet
above is recorded; re-running its strict arm needs the source revision it was
measured at, not the current tree. That is the cost of the deletion, not a
reason to keep a lever whose condition was met.

## Staged PV singleton unroll rollback lever (added 2026-10-09)

Cleanup status (2026-10-26, R0047): removed. `HIPENGINE_GEMMA4_PV_UNROLL` and
its `getenv`/`atoi` reader are gone, and
`launch_gemma4_attention_staged_pv_singleton_rows`
(`hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention_staged.hip`) launches
the staged singleton P*V decode walk at a fixed 32-key in-flight V window. The
`kPvUnrollV` template parameter stays -- the kernel body's arithmetic is written
in terms of it -- so what the removal took out is the 8- and 16-wide
instantiations and the per-launch environment read, not the width contract.

The arithmetic is unchanged, so no execution-profile gate applies. For each
(row, dimension) the accumulator is one FP32 FMA per live key in ascending key
order at every width, with zero weights skipped; the width only changes how many
V loads a thread keeps in flight. The staged decomposition's strict fallback is
untouched: a shape whose staged plan declines -- a grid or workspace limit, or a
head geometry the score tree does not implement -- still runs
`gemma4_attention_prefill_bf16`, and the staged kernel stays bit-identical to it
on the shapes it serves.

Both halves of the removal condition are on file. The restored-state 262015
decode record is
`benchmarks/results/2026-10-09-gemma4-gfx1151-256ki-paired.json`: its
`execution_profile` names "PV singleton width 32", decode is 10.209 tok/s
against llama.cpp's 7.211 (1.416x) at the same depth, and its restored greedy
ids equal the mint-time decode 128/128. The paired width ladder is the table in
the immutable entry
`worklog/entries/20261009T054652.959669Z-lhl-gemma4-pv-unroll32-77ada1.md`
(interleaved 8/16/32, both head geometries, arms bit-identical at every point;
width 32 measured 0.74x the width-8 walk time on the sliding geometry and 0.66x
on the global one at 262144 keys). That table is the record: the leaf A/B JSON
it came from lived under `/tmp` and is no longer present, and the ladder was
never filed under `benchmarks/results/`. The lever is present in `v0.8.0`
(2026-10-20) through `v0.8.3` (2026-10-24) -- four tagged releases -- with no
recorded rollback: no test, script, profile, or worklog sets it.

Nothing isolated is lost with the lever. The arms were bit-identical, so it had
no strict-arm role to preserve, and reconstructing a width-8 or width-16 arm
needs the revision that introduced the widths (`6e1c5cd2e`, 2026-10-09), not a
lever in the current tree.

## Router full-block SGEMM rollback lever (added 2026-10-08)

Cleanup status (2026-10-26, R0048): removed. `HIPENGINE_GEMMA4_ROUTER_SGEMM`
and its `getenv` are gone; `gemma4_router_topk_bf16`
(`hipengine/kernels/hip_gfx1100/gemma4/gemma4_router.py:443`) selects on
`tokens >= _ROUTER_SGEMM_MIN_TOKENS` (`:155`, 1024) alone. Full blocks take the
F32 rocBLAS SGEMM tier ahead of any declared mode (1.68x the tile on gfx1151 at
the in-situ geometry); narrower blocks keep the pre-tier schedule -- the
backend's declared tiled mode where it has one (`token_tile_8` on gfx1151,
`:471-488`), otherwise the shared `token_tile_16` fallback above 32 rows
(`:489-499`), which is what gfx1100's `token_tile_4` mode reached at 1024 rows
before the tier landed.

Both halves of the removal condition were on file: the section 2.11 packet
`benchmarks/results/2026-10-09-gemma4-router-sgemm-section211-packet.json`
(18 cases / 1,134 rows; global median 1.21e-07, p90 3.68e-03, flip-damage
0.79%, top-1 98.68%, all cases passed) and four tagged releases (`v0.8.0`
through `v0.8.3`) with no recorded rollback. What the condition could not
satisfy as written was its own last clause -- the tile schedule stays "the
strict-arm owner for gate captures" -- because the env read was the only route
to it at full-block geometry.

The comparison control moved into the harness rather than being lost.
`scripts/gemma4_wmma_promotion_packet.py --candidate-kind router` pins the
strict arm in-process by raising `gemma4_router._ROUTER_SGEMM_MIN_TOKENS` above
any block width, and restores the shipped threshold for the candidate arm and
from the run's `finally` -- the mechanism
`scratch/p8_baseline_route_wrapper.py` and
`scratch/p8_gate_baseline_wrapper.py` already use for the teacher-forced gate.
Both arms still share one runner, so they differ in the router projection
alone. Re-running the packet needs the same artifact and command; the recorded
packet was produced by the env lever and remains the tier's evidence.
`tests/test_unit_gemma4_router_sgemm_control.py` pins the strict threshold
against the production block width, the candidate restore, the restore after a
raising capture, and the absence of an environment read in both files.

## Q5_1T16 down-prefill rollback lever (added 2026-10-08)

Cleanup status (2026-10-26, R0049): removed. The environment read is gone from
`_gemma4_project_experts_down_wmma_t16`
(`hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py`), with the
function-local `import os` it needed and its `docs/ENVS.md` row. The leaf now
decides on the weight's own facts alone: the `gguf_q5_1` quant key, the
`in_features % 32` / `out_features % 16` geometry, and whether the weight
carries the `t16_down` tile allocation. Each decline falls through to
`gemma4_project_experts_wmma`, the raw-block grouped WMMA owner, which stays
for weights that carry no tile allocation; the strict compensated variant
resolves through that owner by the call site's `not compensated` guard, so no
isolated path was deleted with the lever.

The lever existed only to A/B the two owners inside one process on the
shared-memory iGPU, where mid-band session variance (~10 percent) is larger
than the expected step-level delta. Both halves of its removal condition were
on file. The paired ladder is the 2026-10-08 entry
`worklog/entries/20261008T211640.994270Z-lhl-gemma4-q51-t16-down-5bc7e4.md`:
the interleaved leaf A/B at the in-situ geometry (2.817 -> 1.714 ms, 1.6435x,
40 alternating iterations) and two paired end-to-end rounds at 8192 with fresh
processes and the arm order reversed (prefill 1163.9/1166.0 -> 1244.3/1245.3
tok/s, 1.069x and 1.068x, both arms `status: ok` and `public_path_parity:
True`). The lever shipped in `v0.8.0` (2026-10-20) and survived `v0.8.1`
through `v0.8.3`; it is absent at `v0.7.1`, and no rollback is recorded. A
numerical gate is not owed: the tile value byte is the integer the raw decoder
derives and the arithmetic is the raw kernel's, so the two leaves are
bit-identical, pinned by
`tests/test_gpu_gguf_q5_1_t16_grouped_prefill.py` at uniform and ragged expert
loads.

The comparison control moved into the harness rather than being lost.
`scripts/gemma4_p3_down_ab.py --down-t16 {on,off}` pins the raw-block owner
in-process by pointing `_gemma4_down_t16_tiles_ready` at a constant `False` --
the guard the leaf itself reads, so there is no second selector -- and
restores it, with the run's HIP-event timing wrappers, from `main`'s `finally`.
Both down owners are timed on every call because the tile leaf launches outside
`gemma4_project_experts_wmma`; the run prints each owner's launch count and the
`gemma4_moe_expert_route_counts()` delta, and exits nonzero when the selected
arm's own owner never launched. Two fresh processes with the arm order reversed
are the paired ladder.
`tests/test_unit_gemma4_down_t16_control.py` pins the arm, its restore after a
raising run, the leaf's selection and fall-through, and the absence of an
environment read.

## Gemma 4 WMMA attention candidates were promoted to the production prefill default (updated 2026-10-15)

Production routes multi-row prefill to `gemma4_wmma_flash` (sliding) and
`gemma4_wmma_flash_full` (head_dim 512) on both gfx1151 and gfx1100. Each
backend's plan passed the section 2.11 router-routing-scoped envelope
(EXECUTION-PROFILES, 2026-10-07 lead decision) on its own hardware: on the
8060S median row KL 3.8e-6, p90 6.0e-3, flip-damage 1.3% against a 3% budget,
top-1 98.4%; on the W7900 median 2.5e-6, p90 5.8e-3, flip-damage 0.62%, top-1
98.94%; repeats and isolation green on both, and the gfx1151 packet also
records task non-inferiority on all 18 cases (the gfx1100 packet's case rows
record repeats and isolation only).

Single-token decode differs by plan. The gfx1151 plan appends `gemma4_staged`
to the prefill request, so a one-token block resolves through that entry: to
the strict decode kernel below the measured strict/flash crossover (4096 keys,
`_STRICT_DECODE_WINS_UNTIL_KEYS`), and to the strict-parity staged singleton
above it. The gfx1100 plan declares a decode scope, and the runner forwards no
prefill request on a decode step when one exists: the layer's decode selector
serves it, with `gemma4_flash_long` (`gemma4_flash` with more slices past 8192
keys) for the untrimmed global step past the resident-logit bound, and for a
step on either packing from 256 keys up to the 1024 where the native launch
asks for flash itself. `gemma4_staged` stays registered behind that route, but
it is not the scope's rollback lever: no plan selects it for decode any more,
the plan's rollback key is `gemma4_plain`, and on gfx1100 the flash wrapper's
own fallback is the only route to it (see the flash-decode section below).
Strict requests `gemma4_plain` on gfx1151; the gfx1100 strict plan declares it
as each scope's rollback key and forwards no prefill request, so strict prefill
there stays on the layer's own route chain. Unifying the two backends' decode
routing and binding plain for gfx1100 strict are the open cleanups here. The
profile remains the rollback lever: `HIPENGINE_EXECUTION_PROFILE=strict`
selects plain attention.

Before the section 2.11 envelope the WMMA candidates were held out of the
default: the multicategory teacher-chain packet found KL failures against the
earlier mean/tail bars, including chains on which the older full-attention
parent also fails, and precision controls that preserve only strict scores or
replace only the probability/value arithmetic also failed those bars. The
staged repair reproduces the two saved failing chains bitwise. That packet and
its public-route evidence belong to
`worklog/entries/20261003T040334.761582Z-lhl-gemma4-staged-exact-repair-c2e87f.md`.

The 2026-09-29 single-chain result in
`benchmarks/results/2026-09-29-gemma4-teacher-forced-gate-prefill-attention.json`
is historical evidence, not coverage of the failing categories or a performance
claim for staged attention. Its original kernel-speed artifacts likewise measure
WMMA, not the staged chain they were compared against.

**Diagnostic timing (2026-10-04; interpretation corrected 2026-10-05).**
The same-process A/B in
`benchmarks/results/2026-10-04-gemma4-gfx1151-prefill-attention-variant-ab.json`
measures staged/WMMA prefill at 93.3/477.8 tok/s at 131072 tokens (5.121x).
The WMMA candidate was held out of the default when this was measured; it has
been the production prefill default since 2026-10-07.
The stage trace at 32768 attributes 59.5% of attention time to scores, 37.8%
to P*V and 2.7% to softmax.

The trace's `LDS_Block_Size=0` does not account for dynamically allocated
shared memory. The staged source already loads Q into shared memory and stages
weights for grouped-head V reuse. Scores use one warp per key, not one thread
per score. The hypothetical 101 TB unstaged read calculation is not measured
traffic. WMMA also changes arithmetic, so its speedup cannot be attributed to
staging alone. Arithmetic-preserving reuse changes still require strict
parent-parity validation; only the reassociated-arithmetic promotion packet
can be avoided. See the correction and implementation in
`worklog/entries/20261005T152628.367992Z-lhl-gemma4-score-reuse-e96b93.md`.

Removal condition: none. The candidates are the shipped prefill default; the
staged decode entry and the strict plain fallback stay registered, and the
execution-profile selection is a product API, not a candidate-only flag. A
revert is a profile decision, not a file deletion.

## Gemma flash decode keeps a fallback the current scope cannot reach (found 2026-10-08; kept 2026-10-26)

Cleanup status (2026-10-26, R0052): keep, not debt. The wrapper's admission is
the route decision for two callers, only one of which goes through a selector,
so the recorded removal condition could not be executed as written.

`gemma4_attention_decode_flash_bf16`
(`hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention.py:1346`) and its
long-row sibling (`:1395`) share `_decode_flash_bf16` (`:1241`), which asks the
launcher's own exported admission -- `flash_request_slices` (`:541`) and
`flash_serves` (`:510`) -- before it requests the flash route (`:1266`, `:1274`)
and delegates to the staged singleton when that admission rejects the shape.
The delegate is the chain the decode scope used before flash, and the
launcher's own fallback -- the owned-scratch class kernel -- is far slower at
long context: measured 2026-10-08 at a 128K prefix, a global row that left the
flash route took **225.3 ms** per decode step against the staged singleton's
**73.4 ms** on the same teacher token and KV prefix
(`/tmp/gemma4-pv/flashvar-decode-attrib.json`, `baseline-decode-attrib.json`,
both at `308c4c38c`).

Two callers reach the wrapper. The layer's decode selector
(`_select_exact_decode_attention`,
`hipengine/kernels/hip_gfx1100/gemma4/gemma4_layer.py:880`) resolves it for the
production decode scope, and nothing that selector admits reaches the fallback:
for the one-token global scope the fitted slice count always exists and is
always admitted, which the fitting unit tests pin
(`tests/test_unit_gemma4_decode_attention_selection.py:536`, `:556`). The MTP
assistant head calls the long-row wrapper directly, with no selector in front
of it -- `hipengine/runtime/gemma4_assistant.py:730` and `:1277` launch
`gemma4_attention_decode_flash_long_bf16` for every draft row -- and its
geometry comes from the assistant artifact. A packing the flash kernel does not
instantiate (`flash_admits`, `gemma4_attention.py:579`: head_dim 256 with GQA
ratio 2, or head_dim 512 with ratio 8) therefore reaches the wrapper with no
route decision but this one, and would land on the class kernel if the check
were dropped. The unit battery pins the same contract for a shape no selector
can admit: `head_dim=128` falls to the staged chain
(`tests/test_unit_gemma4_decode_attention_selection.py:580`).

The recorded removal condition was also mis-costed. Moving the admission into
the selector is not a new device-library call on the route-decision path: the
launcher already makes the same two ctypes calls on the same launch
(`gemma4_attention.py:1036`), so the move is cost-neutral per launch. The
reason to keep the wrapper's check is that the selector is one caller among
two, not that the call is expensive.

Keep the wrapper's check. If the assistant head should share the layer's route
decision, give that path its own selector call; that is a draft-routing change
with its own measurement, not a cleanup of this wrapper.

## Plain Gemma attention needs raw-coordinate boundary validation (found 2026-10-03)

Cleanup status (2026-10-26, R0053): fixed. The plain prefill block kernel now
computes `first_kept` and `walk_end` with saturating int64 arithmetic clamped to
`[0, keys]` (`gemma4_attention.hip:163`, `:190`, `:209`, called at `:275` and
`:294`), the same form the staged family uses. The launcher (`:449`) validates
the `tokens * num_heads` grid product before forming it and bounds `head_dim`,
`keys`, and their sum before the LDS comparison, so an extreme shape returns
`hipErrorInvalidValue` before a launch instead of wrapping the unsigned grid or
the int64 sum. No legal shape is refused: the LDS ceiling and the device grid
limit are the ones the kernel already had, and the coordinate clamp only changes
inputs outside `[0, keys]`.

The arithmetic is unchanged for legal inputs: for `window > 0` and
`0 <= row_offset + token < keys`, `first_kept` is still
`max(0, row_offset + token - window + 1)` and `walk_end` is still
`min(row_offset + token + 1, keys)`; `window == 0` still keeps the full walk.
The clamp replaces a wrapped 32-bit cast only at the edges. This remains a
raw-launch boundary issue, not a measured regression on model-generated
coordinates; the pre-repair staged reproducer that faulted the GPU queue is the
RED evidence and is not re-run.

Regression coverage: `tests/test_gpu_gemma4_attention_prefill_bounds.py` drives
the raw `hipengine_gemma4_attention_prefill_f32` symbol (the one-token prefill
path the Python wrapper routes to decode) with the staged family's huge positive
and negative coordinates, asserts NaN for an empty clamped walk and the CPU
oracle for a clamped non-empty one, and asserts `hipErrorInvalidValue` for an
unrepresentable grid. The existing
`tests/test_gpu_gemma4_attention_geometry.py` narrowed-walk test and
`tests/test_gpu_gemma4_attention_staged.py` parity test remain the arithmetic
checks; the parent runs the GPU half.

## Gemma prefill attention tiled variant is a dead route (found 2026-09-28)

Cleanup status (2026-10-26, R0054): removed. `gemma4_attention_prefill_tiled_kernel`,
`launch_gemma4_attention_prefill_tiled`, and the two
`hipengine_gemma4_attention_prefill_tiled_{bf16,f32}` exports are gone from
`hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention.hip`, and
`scripts/gemma4_attention_prefill_tiled_ab.py` is deleted. The cross-reference
in `gemma4_attention_prefill_wmma.hip` now names the removed route rather than a
symbol that no longer exists.

The live tiled prefill route is a different kernel and was left untouched:
`gemma4_attention_tiled.hip` / `gemma4_attention_tiled.py`
(`hipengine_gemma4_attention_tiled_prefill_bf16`,
`gemma4_attention_prefill_tiled` at `:175`), registered as `gemma4_tiled`
(`gemma4_attention.py:1600`), selected by `_select_prefill_route`
(`hipengine/kernels/hip_gfx1100/gemma4/gemma4_layer.py:837`) and launched at
`:1569` for head_dim 512. It stays pinned by
`tests/test_gpu_gemma4_attention_tiled_parity.py` and
`tests/test_unit_gemma4_attention_prefill_routing.py:112`.

The removed kernel was written to make the shared-memory footprint independent
of context -- the strict kernel materialises one logit per key, so its shared
array is `(head_dim + keys + 256) * 4` bytes and occupancy falls from 16 blocks
per CU at 512 keys to 3 at 4096 -- on the theory that this is what limits the
strict kernel. It is not what limits it.

Measured with the since-deleted `scripts/gemma4_attention_prefill_tiled_ab.py`,
both kernels driven through ctypes on the same device buffers, 5 iters and 2
warmups:

| tokens | head_dim | strict | tiled | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 256 | 150.5 ms | 150.65 ms | 1.00 |
| 2048 | 256 | 46.72 ms | 67.21 ms | **1.44 (slower)** |
| 2048 | 512 | 88.83 ms | 95.52 ms | **1.075 (slower)** |

and the outputs are **bit-identical** at every one of those sizes -- 0
mismatched elements out of 8,388,608 and 16,777,216, max abs diff 0, max rel
diff 0. The two kernels are not two arithmetic contracts; they are one contract
with two schedules, and the schedule that costs a 1.44x slowdown is the one
whose only justification was the occupancy theory the measurement refutes.

Verification: the CPU routing and route-probe tests pass
(`tests/test_unit_gemma4_attention_prefill_routing.py`,
`tests/test_unit_gemma4_public_route_probe.py`). The parent's device build is
the compile check that the two exports are gone, and its GPU
`tests/test_gpu_gemma4_attention_tiled_parity.py` plus one public
`LLM.generate()` at head_dim 512 confirm the live route still selects
`gemma4_tiled`. The strict kernel stays: it is both faster than the removed
schedule and the one the decode-parity contract is written against.

## Dense Q8 MMQ prefill branch is unreachable behind the WMMA rewrite (found 2026-09-26)

Current status (2026-10-26): the historical heading is incorrect. The branch is
shadowed when WMMA claims the shape, but remains reachable otherwise.

Cleanup status (2026-10-26, R0055): keep, not debt. The dense branch of
`_q8_mmq_prefill_dispatch` (`hipengine/runtime/gguf_linear.py:10209`) is
reachable whenever the WMMA prefill opt-in is off, and it is the calibrated
route for exactly that configuration; deleting it would remove a live fallback.

`_wmma_prefill_dispatch` (`:10892`) does run first in the chain (`:5024` before
`:5105`) and rewrites a raw Q8_0 dispatch to `abi="wmma_raw"` /
`variant="wmma_prefill_bf16_bf16_out"`, so under the default opt-in the MMQ
branch returns the dispatch unchanged. But the opt-in is not unconditional:
`_gemma4_block_wmma_session` (`hipengine/runtime/gemma4.py:1022`) leaves
`use_wmma` false for a verification block and for any partial block when
`HIPENGINE_GGUF_WMMA_PREFILL` is unset (`gemma4.py:2237`, `:2062`, `:2124`), and
the env var is the documented rollback lever. In that configuration the rewrite
is a no-op and the MMQ chain takes the shape. With `DEFAULT_PREFILL_BLOCK` at
1024 (`gemma4.py:184`, `:1185`) and the policy's 512-row floor (`gemma4.py:1066`),
a 512-row tail is the shipped case that reaches it.

The Gemma policy was calibrated against that exact owner, not the WMMA one
(`gemma4.py:1034-1046`, "what a shape runs when WMMA prefill is off"), and
`tests/test_unit_gguf_q8_mmq_prefill_ordering.py` pins both directions: `:93`
holds a WMMA-claimed shape away from the chain, and `:118` holds the chain
taking the shape when `use_wmma` is false. The 2026-09-26 instrumentation -- 680
admitted dispatches rejected on the key mismatch, none on policy -- was a
WMMA-on measurement, so it describes the shadow, not unreachability.

**Do not reorder.** Making the branch fire ahead of the rewrite was measured 29%
slower (1054 against 1367 tok/s) although correct, and `kl_max` improved to
0.000893 from 0.001341; at 512 rows the chain is 1.44x slower than the WMMA
owner (258.6 -> 373.2 ms, campaign iteration 154). The ordering keeps the
slower route off the default path, and the guard test makes a reorder fail
loudly.

Removal condition: the branch goes only when its configuration does -- the exact
owner is retired, or WMMA prefill becomes unconditional for every block (no
partial/verification opt-out and no rollback lever). Then delete the dense
branch, the three unwrapped variant names it matches on
(`prefill_bf16_bf16_out`, `gemv_f32_f32_out`, `prefill_f32_f32_out`), and the
Gemma `Q8MMQPrefillPolicy` table together, and update the ordering guard. Until
then, keep it: it is the route the policy's numbers were measured against, and
a faster dense Q8 MMQ leaf is a new candidate with its own measurement, not a
restoration of this one.

## Gemma attention legacy slice-count ABI (2026-09-26)

Cleanup status (2026-10-26, R0056): **removed.** The signed count is gone from
the decode ABI. The route is named, and the flash key-slice count travels in its
own argument.

The two-pass decode attention in
`hipengine/kernels/hip_gfx1100/gemma4/gemma4_attention.hip` partitions output
dimensions and preserves the single-kernel key accumulation order. The legacy
two-pass key-slice and combine kernels were already gone; what remained was the
count that selected between the surviving routes. The decode entry points now
take `(workspace, route, flash_slices, live_extent)`:

- `kGemma4DecodeRouteStrict = 0` — the single-kernel oracle, including the
  resident-bound owned-scratch class kernel and the block fallback.
- `kGemma4DecodeRouteSplit = 1` — the two-pass dimension partition.
- `kGemma4DecodeRouteFlash = 2` — the key-sliced flash route with its combine.

The host mirrors those values as `DECODE_ROUTE_{STRICT,SPLIT,FLASH}`
(`gemma4_attention.py`). `flash_slices` is the flash request's key-slice count
and is zero on every other route; `flash_request_slices` still derives it, and
those flash key-slice and combine kernels
(`gemma4_attention_decode_flash_kernel`, `..._flash_combine_kernel`) are live.

Behavior is unchanged. The dimension partition is still entered at 1024 keys
(`DECODE_SPLIT_MIN_KEYS`, `decode_split_engages(keys=...)`), and its grid is
still the head width's (`(head_dim + 31) / 32` blocks per head), which is why
the count that used to be 2 or 4 never sized it — including the 2026-09-26
narrow-head 4-slice rule, which was already inert at the ABI. Flash still
engages at the partition threshold, or from `FLASH_DECODE_PREFERRED_MIN_KEYS`
(256) under a plan that prefers it. A rejected flash launch still falls through
to the partition on the same workspace, which the host sizes as
`max(split, flash)`; a request with no servable count falls through to the
route the shape would otherwise take. The whole-step graph capture reserves the
same expression before it starts recording (`gemma4_decode_graph.py`).
`gemma4_decode_split_workspace_bytes` / `split_workspace_bytes` lost the
`head_dim` and `slices` parameters that never affected the result; it is
`(tokens, num_heads, keys)`.

The partition is therefore a capability fallback rather than the shipped decode
route: both Gemma 4 geometries are flash packings, so at 1024 keys and above a
one-token decode takes flash (as it did before this change), and the partition
serves a geometry outside the ratio-2/ratio-8 packings — GQA ratio 4 at
`head_dim` 512, for instance. The multi-token prefill launcher is unaffected: it
enters the decode launcher with the strict route and a null workspace, which is
the bit-identical class kernel. No public `LLM.generate()` on the 26B-A4B
artifact selects the partition; its arithmetic guard is the GPU parity
battery's explicit route request.

Callers migrated with the ABI: the launch site itself
(`gemma4_attention.py`), the capture reservation
(`hipengine/runtime/gemma4_decode_graph.py`),
`scripts/gemma4_teacher_forced_gate.py` (`--slices` still selects the arm:
`1` = strict oracle, above one = partition), the decode bench and public route
probe, `tests/test_gpu_gemma4_attention_decode_parity.py`,
`tests/test_gpu_gemma4_flash_decode.py`, and the Gemma unit battery. The two
tracked diagnostic probes `scratch/flash_scale_probe.py` and
`scratch/gemma4_flash_slice_sweep.py` are migrated with them:
`route=DECODE_ROUTE_STRICT` on the single-kernel reference and
`route=DECODE_ROUTE_FLASH, flash_slices=slices` on the flash arm.

Guard: `tests/test_unit_gemma4_decode_route_abi.py` pins the argument tail, the
route values, the count's separation, and every workspace reservation; it also
compares the `.hip` export parameter lists against the host argtypes, and fails
if a tracked raw caller still passes the removed argument names;
`tests/test_unit_gemma4_attention_routing.py` pins the partition threshold and
asserts `decode_slices` does not come back; the Gemma unit battery and the two
GPU parity batteries keep their original assertions, including the partition's
bit-equality cases and the non-admitted-geometry fall-through. One public
`LLM.generate()` request past 1024 keys is the parent's device check that a real
request takes the flash route.

## gfx1151 compact attention wave producer — NUMERICAL CONTROL FAILURE

The gfx1151 BF16 compact attention build selects the existing generic grouped
producer. The wave-group6 producer failed the unchanged four-category G0
no-evict control; replacing only attention with dense arithmetic eliminated the
mixed-category mismatch. Evidence and exact commands are in
`worklog/entries/20260926T140616.347941Z-main-dms-producer-control-183661.md`.
No feature is disabled and no model identity is used for admission. gfx1100
selection is unchanged; this finding was measured on gfx1151 only.

Removal condition: repair the wave producer and pass the same G0 command from
that worklog, the matched-input attention GPU test, public DMS lifecycle tests,
and applicable execution-profile numerical gates before restoring gfx1151 wave
selection. Primitive tolerances alone did not catch this full-model failure.


## Dormant (Q3_K gate, IQ4_XS up), (IQ4_XS gate, Q3_K up), (IQ3_S gate, IQ4_XS up), and (Q5_K gate, Q6_K planar up) fused pair registrations (2026-09-25) — AWAITING A CLEARING SCREEN

E6b-5 built the fused pair+SiLU owner for the 3-layer (Q3_K, IQ4_XS)
family: `B_KIND=2` in `gguf_iq4_q4_pair.hip` (side B = the strict
per-row Q3_K GEMV's exact 128-thread tile emulated across the pair's
waves), the `hipengine_gguf_q3_iq4_pair_silu` entry point, its Python
wrapper, and the registered key `gguf_q3_k+gguf_iq4_xs`. It is
bit-exact against `q3 strict single + iq4 local32 single + silu_mul`
at K=5120 (GPU test) and at (5120, 17408) rows=1 (screen), but the
production screen measured **0.92-0.99x across five runs - never the
required >= 1.00** - so the route block ships dormant: no dispatch
path selects this key, and production behavior is unchanged by
construction.

**E6b-7 added the reverse direction the same day (iteration 18):**
(IQ4_XS gate, Q3_K up), 1 layer (blk.0), under a second instantiation
`<W, GATE_IS_Q4=false, B_KIND=2, A_IS_CHAIN=false>` and its own
extern-C `hipengine_gguf_iq4_q3_pair_silu` + wrapper
`gguf_iq4_q3_pair_silu_bf16_bf16_out` + registration key
`gguf_iq4_xs+gguf_q3_k` (route order == C geometry, no reorder; the
epilogue takes the gate from side A's IQ4 chain). Bit-exact at K=5120
(GPU suite) and at (5120, 17408) rows=1, but five production screens
measured **0.95 / 0.97 / 1.03 / 0.96 / 0.95 - mean 0.972, four of five
below the >= 1.00 gate** (the lone 1.03 inside the historical screen
spread), so its route block was also removed and production stayed
byte-identical by construction (LLM probe: pair7 = 0 dispatches, all
five shipped families intact). The structural cause is the same one
recorded for the first direction: sides keep identical arithmetic, so
the direction swap cannot change the wall picture.

**E6 closeout added two more dormant directions the same day
(iteration 21):** (IQ3_S gate, IQ4_XS up), 1 layer (layer 11), under
instantiation `<W, GATE_IS_Q4=true, B_KIND=3, A_KIND=0>` + extern-C
`hipengine_gguf_iq3s_iq4_pair_silu` + wrapper
`gguf_iq3s_iq4_pair_silu_bf16_bf16_out` + key
`gguf_iq3_s+gguf_iq4_xs` (B_KIND=3 is the IQ3_S local32 decode
owner's Q==2 split-K path verbatim - both sides of layer 11 take the
session's local32 owner, no ffn_gate pin; the wrapper reorders the
gate-first route args to C geometry). Bit-exact at K=5120 (GPU test)
and at (5120, 17408) rows=1 (screen), but four production screens
measured **0.99 / 0.99 / 0.98 / 0.97 - never the required >= 1.00**.
And (Q5_K gate, Q6_K planar up), 1 layer (layer 63), under
`<W, GATE_IS_Q4=true, B_KIND=1, A_KIND=3>` + extern-C
`hipengine_gguf_q5_q6_pair_silu` + wrapper
`gguf_q5_q6_pair_silu_bf16_bf16_out` + key
`gguf_q5_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1` (A_KIND=3 emulates
the planar single's exact 4-wave chain across the block's waves; the
wrapper reorders gate-first args so side A runs the planar chain).
Bit-exact everywhere, but four production screens measured
**0.60-0.61x** - the chain emulation across block waves costs far more
than the launch it saves. Both route blocks were removed with the unit;
production behavior is byte-identical by construction. The third
closeout direction, (IQ4_NL gate, Q5_K up) at 1.03-1.04x, passed its
gate and shipped (it is not part of this entry).

**E6 same-quant closeout recorded a capability gap (iteration 22):**
(Q3_K gate, Q3_K up), 1 layer (layer 14), has no expressible entry -
side A has no strict-Q3 instantiation (only `B_KIND=2` runs the strict
Q3 tile, on side B), and every strict-emulation screen to date lands
0.60-0.99x (E6b-5, E6b-7, E6 closeout), so a predicted-failing body
was not built. Nothing ships and production is unchanged by
construction; any artifact carrying this ordered pair hits the same
gap. The same unit adjudicated (Q4_K, Q4_K) as **already fused** by
the pre-existing dense-dual route (5 layers, 4.84 launches/token - no
entry needed, admission test only) and shipped (IQ4_NL gate, IQ4_NL
up) at 1.04x with `B_KIND=4`; neither belongs to this entry.

Clearing command (Q3_K/Q3_K gap): port the strict per-row Q3_K tile
into the pair's side A (mirror of `B_KIND==2`, e.g. `A_KIND=4`),
instantiate `<W, GATE_IS_Q4=false, B_KIND=2, A_KIND=4>` +
`hipengine_gguf_q3_q3_pair_silu` + wrapper + registration key
`gguf_q3_k+gguf_q3_k`, prove bit-exact vs `q3 strict + q3 strict +
silu_mul` at K=5120 and at (5120, 17408) rows=1, then screen
`~/ud-e1-census/e6_closeout_screen.py` (or successor): only at
>= 1.00x add the route block in `hipengine/runtime/gguf_linear.py`
keyed on both sides' raw strict `gemv_bf16_bf16_out` owners at
rows==1.

Clearing command: re-run `~/ud-e1-census/e6b5_screen.py`,
`~/ud-e1-census/e6b7_screen.py`, and
`~/ud-e1-census/e6_closeout_screen.py` (or successors) after a
structural change to the pair (e.g. overlapping side A and side B
phases, a lower-overhead strict emulation, or a wave-scheduled planar
chain that runs each single-wave on its own hardware wave); if any
direction measures >= 1.00x bit-exact, add that direction's route
block back in `hipengine/runtime/gguf_linear.py` keyed on its ordered
pair (`gguf_q3_k+gguf_iq4_xs` or `gguf_iq4_xs+gguf_q3_k` gate-first,
`gguf_iq3_s+gguf_iq4_xs`, `gguf_q5_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1`)
with the matching predicates (raw-IQ side = session-qualified decode
owner with the spec's slot_path - Q3/IQ3_S parents keep
`gemv_bf16_bf16_out` without a policy entry or under a pin, IQ4 sides
the local32 owner; Q5 side = E4a tile8 c1 owner; Q6 side = direct
planar decode key; both sides' ABIs as the shipped wrappers read them)
and delete this entry.

Removal condition: if the fused pair family is redesigned or those
layers change quant, delete `B_KIND==2`, `B_KIND==3`, `A_KIND==3`, all
four dormant extern-C wrappers, their Python wrappers/registrations
(all four ordered keys), and their GPU/route tests together - dead
code with no route does not accumulate a second life.
Evidence: `docs/campaigns/UD-GFX1151-OPTIMIZE2.md` E6b-5 row;
`worklog/entries/20260925T011636.401265Z-lhl-ud-gfx1151-optimize2-e6b5-q3-iq4-negative-c7f21a.md` (this unit's entry).

## Dense27B prefix oracle disagreement after prefix/MTP integration (2026-09-20) — RESOLVED

The merge qualification at p1024+s200 on Qwen3.8-27B Q4_K_M / gfx1151
compares active and completed prefix reuse against both private batched
recomputation and the gate's one-shot-prefix/serial-suffix oracle. All four
category probes have byte-identical prefix and post-suffix state versus
private batched recomputation, exact clone metadata, and correct lifecycle.
The serial oracle is not numerically equivalent: English and mixed-category
mean KL exceed the production envelope, and completed-source Japanese fails
even the outer numerical smoke. Do not call this full production numerical
qualification or hide it behind the passing clone/ownership checks.
The identical Japanese probe on untouched upstream `1de470e21` reproduces
the complete numerical and state-comparison records exactly; this discrepancy
predates the merge.

Next: localize the one-shot-prefix versus batched-prefix divergence using
matched teacher rows and real category/heldout prompts; then run the calibrated
production envelope and task gate. Preserve the batched HIT/MISS state check
separately from that investigation. This ledger item does not justify
prompt-conditioned routing or a blanket strict bit-equality requirement.
Evidence: `benchmarks/results/2026-09-20-gfx1151-prefix-mtp-merge-qualification.json`.

**Resolution.** The prescribed localization and envelope run are done, and the
finding above describes the comparison rather than the candidate. The gate's
reference is itself built by a prefix prefill, and the packed wave and the
single-session path persist different final-state arithmetic, so a reference
built through the single-session path made every numerical term a cross-route
comparison. `--reference-prefill-mode` now defaults to `packed`
(`prefill_batch_native`), sharing the candidate's and the served engine's route
family; `native` and the plugin's certified bulk scheduler remain as
diagnostics. That the identical Japanese probe reproduced exactly on untouched
upstream is what a comparison artifact looks like, not a merge regression.

On the same-route reference the gate passes all eight suites -- four natural
categories and four heldout suites, 1,024 teacher-forced rows, 2 top-1 flips,
0 state mismatches -- with the worst scope (`heldout-general_en`) at least 2.5x
inside every binding term of the calibrated envelope. `general_ja` moves from
`kl_mean` 0.6811212740489249 / `kl_max` 1.7583460667891433 / top-1 0.75 over 4
teacher-forced rows to 0.0001940 / 0.0050300 / 1.00 over 128. The batched
HIT/MISS state check was preserved separately as instructed.
Evidence: `benchmarks/results/2026-09-20-gfx1151-prefix-gate-packed-reference-1024rows.json`;
source: `worklog/entries/20260920T054514.573459Z-lhl-prefix-gate-packed-reference-1024rows-40a28f.md`.

**W7900 BF16 resolution (2026-09-20).** Gapped prefix-hit pages selected native
attention because their required gather inherited the optional contiguous
head-major-layout default. Resolving the registered gather independently removes
all 120 scheduler state mismatches. Contiguous defaults do not change.
The remaining completed-source KL failure also occurs for private batched
recomputation compared with token-serial suffix replay. The gate now compares
independent private batched-prefix/batched-suffix recomputation, applies the
unchanged canonical mean/p95/p99/max KL and top-1 envelope, and retains the
serial-suffix comparison against both candidate and private recomputation.
All four categories and four heldouts, both source lifecycles, pass with zero
KL and exact state over 2,048 teacher-forced rows. This is cache-hit fidelity,
not certification that distinct prefill schedules have identical arithmetic.
Evidence: `benchmarks/results/2026-09-20-w7900-prefix-bf16-layout-repair.json`.

HTTP and in-process prefix caching default to radix. `hipengine serve
--prefix-cache off` or `HIPENGINE_PREFIX_CACHE=off` is the rollback.
Eligibility fallbacks record `prefix_fallback_reason`; they do not detect
numerical divergence. The separate no-mirror INT8 suffix ownership repair is
also complete: resumed prefill reads retained INT8 payload and scales rather
than a transient BF16 oracle whose prefix was never written. The current
`tests/test_unit_gguf_prefix_int8_attention_source.py` pins the source choice,
gapped pages, mirrored fallback, and unsupported-layout rejection before writes.

The RX 7900 XTX packet records suffix/reclaim/pressure checks and public prefix
hits, including actual INT8/FP32 storage with no mirrors:
`benchmarks/results/2026-09-20-rx7900xtx-prefix-int8-ownership-repair.json` and
`worklog/entries/20260920T214917.189511Z-lhl-close-int8-radix-prefix-ownership-validation-0f1633.md`.
Its claim is cache-hit fidelity against independent same-schedule recomputation,
not cross-schedule identity with a cold BF16-oracle prefill. The earlier open
paragraph was superseded by that validation but left unchanged during
shared-file coordination.

## Three qwen36-dense serving rows carry no execution identity (found 2026-09-20)

The three `qwen36-dense-q4km-gfx1100-*` rows in
`hipengine/models/qwen35.py` describe Qwen3.6-27B-Q4_K_M
(`/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`, sha256 `a7cbd3ec…`, 17,106,773,120
bytes) and record its measured gfx1100 scope. Their
`artifact_execution_fingerprint` fields are still empty, so the rows cannot
supply matching evidence for a resident file. They do not control runnability:
automatic and explicit requests both resolve through the implementation
capability declaration when no evidence row matches.

**Clearing command.** `python3 scripts/gguf_execution_identity.py <artifact.gguf>`
prints an execution fingerprint if the historical artifact is available. Record
it only after verifying that it belongs to the measured artifact; do not invent
an identity or substitute a newer file. The regression
`tests/test_unit_speculative_mtp_serving_capability.py::test_qwen36_dense_rows_without_identity_still_run_on_capability`
pins both request modes running with `admission_basis="implementation"`.
Recording a verified identity can improve evidence attribution without changing
which implementation executes.

Both request modes resolve through the dense BF16 implementation declaration,
which does not require evidence rows. The
`qwen36-dense` and `qwen38` rows cannot be conflated by a byte-digest match
because the artifact axis is the execution identity, not the SHA-256.

## MTP admission no longer reads the artifact's byte digest (2026-09-20)

`SpeculativeMTPServingKey.artifact_sha256` and `artifact_size_bytes` are
provenance fields. Evidence matching uses `artifact_execution_fingerprint`
(`hipengine/loading/gguf.py`), while execution admission uses implementation
capabilities and does not require a matching evidence identity. The two Qwen3.8
plain builds differ by 1,024 metadata bytes and share one identity, and the
resident file's SHA-256 is no longer computed on the serving path because no
row reads it.

Consequence to be aware of: a `KVCapabilityKey`'s `capability_id` covers its key
payload, so it changed for every int8-KV contract. The two 2026-09-11 W7900
int8-KV artifacts under `benchmarks/results/` record the pre-change id
(`4b0e936f…` for the gfx1100 direct-c4 contract); they are frozen measurements
and are not rewritten.

## Wide MTP groups run without a prompt provider, and the over-width demotion is inert (found 2026-09-19) — RESOLVED

Resolved in three steps. (1) The `(8, 3)` cell left
`GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS["production"]` in 9c8ff5444; the
explicit capacity-8 route now falls to the registered strict fallback
`gguf_target_ar`, and the production-admission rerun recorded in
`benchmarks/results/2026-09-19-gfx1151-qwen38-mtp-width-census-and-c8-k3-withdrawal.json`
confirms the withdrawal. (2) Speculation routes by priming source instead of
sink refusal (3edac59c9): a row whose prompt sink was refused declines
capability with `provider_state_absent` before any cycle can open a provider
(regression:
`tests/test_unit_qwen35_gguf_mtp2_seam.py::test_capability_refuses_a_row_whose_prompt_sink_was_refused`).
(3) A zero partition bound is one whole-batch AR step in the engine instead of
"do not partition" (regression:
`tests/test_unit_specdec2_engine_loop.py::test_zero_partition_bound_stops_a_wide_capable_runner_from_cycling`).
The observations below record the original finding; the first bullet's claim
that the production width table lists `(8, 3)` predates the withdrawal.

- `hipengine/kernels/hip_gfx1151/__init__.py` lists `(8, 3)` in
  `GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS["production"]`, but
  `GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES` for the same
  geometry/file-type/profile admits widths `(1, 2, 3, 4)`, so
  `begin_prompt_streaming` refuses a pending set of 8 rows and stamps every row
  `mtp2_prompt_fallback_reason = "physical_streaming_category_rejected"`. The
  refusal does not fail the route closed: the group still reports
  `effective_route=speculative_mtp`, `used=true`, and executes 7-9 draft cycles
  per row against a provider with no prompt state. Measured on all 80 rows of
  the 10-cell width-8 protocol, with production admission as well as the
  diagnostic resolver: the MTP arm agrees with AR for 3-5 tokens and then emits
  the same `<|im_end|>`/`<|im_start|>` loop on all 10 prompts, so its output does
  not depend on the prompt, and the cell is 0.936x of its own AR baseline where
  the 2026-09-05 qualification recorded 1.0015x with 40/40 AR-equal cells.
  Evidence:
  `benchmarks/results/2026-09-19-gfx1151-qwen38-mtp-width-census-and-c8-k3-withdrawal.json`.
- `hipengine/generation/qwen35_gguf_mtp2.py` `partition_max_requests` returns 0
  for a due group wider than
  `GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS["production"]` (4), but
  `hipengine/generation/engine_loop.py`
  `_maybe_run_partitioned_speculative_decode` reads a non-positive bound as
  "do not partition" and returns `None`, after which `_run_decode` falls through
  to `_maybe_run_speculative_cycle` on the whole wide work item. The M5
  whole-batch-AR defense is therefore inert for an over-width all-speculative due
  item; `docs/REFACTOR.md` RF-M5 recorded that precondition as unreached before
  the 2026-09-05 width policy admitted `(8, 3)`.
- Remove the wide cell by either failing the route closed when the prompt sink is
  refused (refusal is not a partial state the cycle can run on), or by adding
  widths 5-8 to the prompt-streaming policy and re-qualifying the cell against a
  same-run true-AR baseline. Move the over-width demotion to the admission owner
  or make the engine treat a zero partition bound as a whole-batch AR step; then
  the partitioner-level defense can be deleted. Do not keep `(8, 3)` listed while
  the prompt sink refuses width 8.

## Per-token reused-prefix suffix prefill (found 2026-09-18)

Resolved during the upstream prefix-cache campaign and integrated into the
local MTP branch during the merge. The historical observation below explains the
original work; it no longer describes the default suffix route. Upstream's
`HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX` switch owns the serial fallback, and
single-token suffixes intentionally stay serial. The overlapping task #20
`HIPENGINE_GGUF_PREFIX_SUFFIX_PACKED` helper and switch are removed during
integration; the preservation commit `0f3bd43dc` keeps their history.

- `hipengine/generation/qwen35_gguf.py` `_prefill_native_chunk` and
  `_prefill_processed_argmax_chunk` both take an `if row.prefix_reused_tokens:`
  branch that prefills the suffix after the matched prefix with one
  `session.step(token_id)` per token, where the miss path calls the batched
  prefill. A step costs a decode step, so a hit prefills at **82.1 ms per
  executed token against 3.3 ms** on the miss rows (24.9x) and 28.9x the TTFT,
  which made a realistic multi-turn load decode 3.47 tok/s with
  `--prefix-cache radix` against 46.52 with it off. The route stays default-off
  and is not promotable until this branch is replaced by the batched entry; the
  retained 11.8x prefix win is a p256+s1 packet, where the loop costs one step.
  Evidence:
  `benchmarks/results/2026-09-18-gfx1151-qwen38-prefix-hit-multiturn-rejected.json`.
- The branch also has to go before MTP can be admitted on a hit: it is the same
  per-token processing that a draft-provider checkpoint would need to skip, and
  measuring the checkpoint against a decode-speed suffix would attribute the
  suffix's cost to the checkpoint.

## `prefix_reuse_k0` prompt-sink refusal (found 2026-09-18) — RESOLVED

Capture and restore now supply provider state for usable prefix hits. The
observations below describe the original refusal; the remaining label identifies
a missing or unusable checkpoint, not an unconditional refusal of prefix hits.

- `hipengine/generation/qwen35_gguf_mtp2.py` `_open_prompt_streaming_sinks`
  returns `None` as soon as any row in the group has `prefix_reused_tokens > 0`
  (`mtp2_prompt_fallback_reason = "prefix_reuse_k0"`), so the provider never
  receives the prompt's hidden rows and every decode cycle reports
  `no_provider`: measured 127 events and 0 draft cycles on hit rows against 126
  of 128 tokens from speculation on the matching miss rows. It is a correct
  refusal for the current target route (a reused prefix never ran the target in
  this request, so there are no hidden rows to stream) and a dead end for MTP.
- Remove it by supplying the provider's missing state instead of refusing: batch
  the suffix (entry above), then capture the provider's recurrent state at each
  256-aligned boundary during the catch-up it already performs and restore it on
  a matching hit. Rebuilding the target's hidden rows for a reused prefix is the
  rejected alternative, because it re-runs the target over the whole reused
  prefix.
- Capture (2026-09-19) and the restore (2026-09-20) are both in place, so the
  `restored_checkpoint_suffix_unavailable_k0` reason and the runner-side
  `prefill_activation_ready` / `prefill_needs_activation` shortcut that treated
  every reused row as never priming a provider are gone. The `prefix_reuse_k0`
  label now names only a hit row whose checkpoint is absent, stale, lacks its
  boundary row, or cannot be restored.
## Prefix snapshot eviction destroys the retained entry it just served (found 2026-09-18)

- `_capture_prefix_snapshot` trims with `while len(self._prefix_state_snapshots) >
  self._prefix_snapshot_limit: self._evict_prefix_snapshot(next(iter(...)))`,
  which takes the oldest entry whether or not it is retained. The resident loop's
  limit is 1 (`snapshot_limit: 1` in the 2026-09-18 gfx1151 serving run), and one
  request captures twice: once at its decode boundary and once at its completion
  prompt boundary (`_refresh_prefix_cache_at_prompt_boundary`). The second
  capture therefore always evicts the entry that served the current request,
  releasing its pool pin and its `cache.release_entry`. The next request finds an
  empty trie and reports the engine reason `miss`.
- Measured consequence on gfx1151 (41 served turns per arm, one arm per process):
  33 captures, 32 evictions, 6 hits, `unusable_hits: 0`. Every lane with reusable
  content hits **at most once**, on the turn after its first capture. Every
  missed eligible turn had at least 2048 tokens of verbatim resent prefix
  (`prompt_lcp_reusable_tokens`), so the tokens were matchable; the failure is
  retention, not structure. Artifact:
  `benchmarks/results/2026-09-18-gfx1151-prefix-cache-miss-path-diagnostic.json`.
- Fix direction: make the trim prefer unretained entries and never evict a
  retained entry while an unretained one exists, and size the retained working
  set for several conversations rather than for the loop capacity. RED test
  target: with `_prefix_snapshot_limit == 1`, capture for request A, promote it,
  capture for request B, then assert A's entry is still resolvable.
- Fixed 2026-09-18: transient and retained entries are now trimmed by separate
  budgets (`_prefix_snapshot_limit` for the current request's captures,
  `_prefix_retained_limit` plus a 1 GiB state-byte ceiling for retained ones).
  `_prefix_retained_evictions_by_reason` and `retained_snapshot_state_bytes`
  expose both budgets through `/ready`.
- The retained count defaults to the pre-fix effective budget
  `max(1, capacity)`: the wider working set is opt-in via
  `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS` (16 covers the 14-lane serving
  protocol, where a conversation's next turn arrives after up to 13 other lanes
  have captured and retained their own boundaries) and
  `HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES` (default 1 GiB). The old
  opt-in reason — the +11%/+308% serial-suffix regression — is gone: the
  batched suffix prefill and the gapped-page gather route landed on
  2026-09-19/20, and retained 16 has been the strongest configuration measured
  on every protocol since (14-lane campaign -18.4% wall vs -17.8% default on
  merged main; real-session replay lane -64.2% vs -44.1%, 30/32 hits, worklog
  `20260919T202013.823662Z-lhl-agentic-session-replay-e8a2c9`). The narrow count
  limits retained state and pinned KV residency. The state-byte ceiling may
  reduce the effective entry count, and it does not include retained KV pages.
  Missing concurrent-load measurements do not gate execution of the wider set.
  Changing the default memory/reuse trade-off is a product decision, not an
  unfinished eviction repair; the remaining decision is listed below under
  deferred work.

## Prefix eligibility floor excludes short multi-turn traffic (found 2026-09-18) — NOT A LOOKUP DEFECT

The resident cache stores complete 256-token pages and requires an unmatched
suffix token. A prompt of at most 256 tokens therefore cannot produce a usable
hit. The current check does not exclude a 400-token prompt with a 256-token
prefix: that prompt reaches the trie lookup. Boundary coverage in
`tests/test_unit_gguf_prefix_cache_runner.py::test_prefix_fallback_reasons_are_counted_per_reason`
checks short prompts, both sides of the page boundary, and a 400-token prompt.
Supporting sub-page reuse would require a different cache/state ownership
contract, not removal of this early check. The historical proposal below
misidentified that capability limit as a lookup-cost policy.

- `_prefix_source_for` returns `prompt_too_short` for `len(row.prompt_ids) <= 256`
  before any lookup. Every ShareGPT conversation in the serving suite is under
  that size (30-685 prompt tokens, four turns above 256), so 19 of 41 turns never
  consult the cache and 0 of 8 lookups hit. Multi-turn chat traffic is exactly
  the shape this cache exists for.
- Fix direction: separate the floor from the block size. A lookup can match zero
  full blocks cheaply and still be worth attempting for a 400-token prompt with a
  256-token prefix; the current constant conflates "too short to have any block"
  with "not worth a trie probe". Decide the floor from measured probe cost, not
  from `block_size_tokens`, and re-measure the ShareGPT lanes.

## Prefix snapshot capture allocates and frees its buffers per capture (found 2026-09-18) — RESOLVED

Snapshot capture now carves state views from one pooled arena, and snapshot
close returns that arena to the session pool. The capture/close/capture reuse
contract is covered by
`tests/test_unit_qwen35_gguf_prefix_state.py::test_capture_reuses_the_pooled_arena_after_close`.
The original repair is recorded in
`worklog/entries/20260918T182703.072006Z-lhl-prefix-snapshot-arena-pool-f8942e.md`.
Capture still synchronizes before publishing the snapshot: the serving caller
does not supply an independent stream, so replacing that fence is not part of
the allocation repair. The observations and proposal below are historical.

- `capture_prefix_state_snapshot` allocates two device buffers per linear-state
  layer, copies the Conv/GDN state into them with `memcpy_async`, and calls
  `device_synchronize`. `_evict_prefix_snapshot` frees them again. Measured on
  gfx1151 with rocprofv3 over 4 served requests: **+161 `hipMalloc` and +161
  `hipFree` calls** (~54 per capture) at **~3.2 ms per malloc**, against state
  copies of **~25 us each**. The engine's own phase counter puts a capture at
  **0.18-0.38 s** on the greedy route and up to **1.4 s** on the sampled route,
  and the prompt-boundary refresh pays it on **every eligible turn, hit or miss**.
- Fix direction: hold the snapshot buffers in a per-session pool sized to the
  largest capture (the state size is fixed by the model) instead of malloc/free
  per capture, and drop the `device_synchronize` to a stream sync when the
  capture is already on the request's stream. RED test target: capture, evict,
  capture again and assert the second capture performs no allocation.
- Evidence: `benchmarks/results/2026-09-18-gfx1151-prefix-cache-cost-attribution.json`
  (`rocprof.hip_api.radix.hipMalloc`, `ranked_cost[1]`).

## Reused-suffix prefill has two routes and only one is slow (found 2026-09-18)

- The greedy cumulative lanes prefill a reused row's suffix through
  `_prefill_native_chunk`'s serial loop, one `session.step()` per token: measured
  **33.8-35.3 ms per suffix token** (6.2-6.9 s on a hit turn) with **~1,679 extra
  kernel dispatches per token** against **~0.4 ms/token** for the batched prefill
  kernels.
- **Corrected 2026-09-18 (same day): there is no cheap second route.**
  `_prefill_processed_argmax_chunk`'s reused branch also calls `session.step()`
  once per token; it simply never recorded the `suffix_prefill` phase (it bumped
  `processed_argmax_prefix_c1_suffix_*` route counters instead), so the fixture
  lanes looked like they skipped the serial loop. The phase counter is now
  recorded on both routes. The sampled lanes' 3-5 s lower walls are therefore not
  explained by the missing phase and still need per-phase re-attribution against
  the same lane prompts.
- **Resolved 2026-09-18 (same day, later): the batched route exists and was
  gated by policy, not capability.** The packed AR prefill accepts non-zero
  start positions (`_GGUFPackedVerifySlotBlock.start_position`), seeds per-slot
  GDN state from the restored session (`_sync_packed_decode_initial_state`),
  imports prefix KV through page-aligned D2D segments, and commits the suffix
  back through the session block table. The only blocker was
  `_validate_packed_ar_prefill_context` refusing `max_live_count >= 1024` on
  the paged route (a 2026-07-20 policy gate from 84fd737a3, never numerically
  qualified past 1024 because the slot-local AOTriton route took over there).
  The gate is removed; the reused branch of both `_prefill_native_chunk` and
  `_prefill_processed_argmax_chunk` now prefills the suffix batched, with the
  serial loop kept as the opt-in fallback
  (`HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX=0`). Fixture evidence (Qwen3.5-0.8B,
  18 GDN + 6 full-attention layers): shared-prefix batched extend matches the
  serial oracle token-for-token with KL <= 0.05 and identical 8-step decode
  continuations at total contexts 712/1224/2248/4296, with and without
  `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN`.
- Historical note: the mid-day correction ("there is no cheap second route")
  was right about the sampled route's missing phase counter and wrong about
  the batched route's reach; the audit entry
  `worklog/entries/20260918T142514.675878Z-pi-prefix-cache-batched-suffix-audit-73980d.md`
  records the mechanism.


## Strict prefix gate cannot bind a batched candidate (found 2026-09-19, fixed 2026-09-20)

- **Resolved.** The original `scripts/gguf_prefix_reuse_gate.py` reference
  consumed the suffix through serial `session.step` calls and compared GDN
  state bytes. That comparison conflated cache fidelity with the arithmetic
  difference between serial and batched schedules.
- The gate now binds the comparison that reproduces the production shape
  (`scheduler_chunk_diagnostic`: `prefill_batch_native` over the prefix and then
  the suffix) through `gate_passed` / `_PRODUCTION_GATE_TERMS`. The serial,
  one-shot-prefix, and one-shot-bulk comparisons stay in the payload as
  diagnostics named in `_DIAGNOSTIC_COMPARISONS`, and `metrics_passed` keeps the
  numerical envelope binding against the teacher-forced reference. The
  localization and its evidence are in
  `worklog/entries/20260920T033547.660457Z-lhl-dense27b-oracle-suffix-shape-localized-5e052a.md`.
- Batched-versus-serial differences also occur without cache reuse. The current
  gate compares independent private batched-prefix/batched-suffix recomputation
  with the reused route; serial-suffix results remain diagnostic. A serial
  fallback is not guaranteed to pass the current same-schedule state gate.
- The gate now accepts `--prompt-file` / `--prompt-category` so its numerical
  checks can run on real tokenized suite text; its default repeated-token-id
  prompt produces degenerate logits that make KL unrepresentative.
- Removal scope: complete for the gate. Cross-schedule accuracy remains the
  separate open question in the split-prefill entry above. Historical Japanese
  serial-reference KL is not a failure of the current same-schedule gate.
  `tests/test_unit_gguf_prefix_reuse_gate.py` pins binding terms separately from
  diagnostic comparisons; no numerical threshold is relaxed by this correction.

## Grown KV pool could not serve an allocation spanning two chunks (found and fixed 2026-09-19)

- The KV-pool growth path had never been exercised until a retained-snapshot
  prefix-cache run pinned enough pages to force it. Three defects surfaced in a
  row; the first two are fixed:
  * the `before_grow` / `on_pressure` callbacks dereferenced
    `self._resident_batch_owner` bare, while every other reader in the file
    guards with `getattr`. A session that owns its own pool never has the
    attribute, so the first grow/pressure event raised `AttributeError`. Fixed,
    with `test_pool_pressure_callbacks_tolerate_a_session_without_a_batch_owner`.
  * `grow_storage` rebound `pointer_tables` without declaring it `nonlocal`,
    so `old_tables = pointer_tables` raised `UnboundLocalError` the first time
    the pool actually grew. Fixed.
  * bind-time validation checked `block_id - start_block_id < backing.pages`
    against the first chunk, so any allocation whose pages landed past it was
    rejected with `GGUF KV allocation is outside its backing chunk`. The
    backing genuinely describes one chunk (its metadata and buffer sizes are
    validated against `pages`), so the bound moved instead: allocations now
    carry `pool_page_capacity` and a pointer-table pool validates against it.
    Fixed, with `test_bound_blocks_validate_against_pool_capacity_after_growth`.
  * the GPU memory fault in
    `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel` was **not**
    stale pointer tables, as first recorded here. `bind_device_kv_allocation`
    gives the session the backing chunk's cache buffers while the block table
    carries global page ids, and every KV kernel addresses a chunk as
    `base + page * stride`. A grown pool handed out allocations whose pages sat
    in the appended chunk, so the write indexed past the end of the first one.
    Fixed by confining an allocation to one chunk: `_select_free` takes a page
    range, `GlobalDeviceKVPool` tracks `(start, pages, backing)` per chunk and
    reports the chunk that owns the allocation. `grow_storage` returns its new
    chunk's backing as a third element (two-element returns still work).
  * a shared-prefix admission must place its suffix in the chunk that holds the
    prefix, and growth appends a separate chunk, so growth can never satisfy
    that path - it now asks eviction instead and declines with `MemoryError`
    when the prefix's chunk is full. Admission turns that into the existing
    private-allocation fallback, so the request is still served.
- Exposure: eviction runs before growth - `_ensure_free_pages` calls
  `_on_pressure` (which evicts prefix-cache entries) first and only grows when
  that cannot free enough - and at `max_pages` growth is refused with
  `MemoryError` rather than attempted. So the fault needs sustained pressure
  plus headroom. `kv_pool_high_water_pages` defaults to `None`, which is
  unbounded headroom; setting it bounds the pool and keeps the pool on the
  eviction path. At the time of this finding, `GlobalDeviceKVPool.shrink_idle`
  returned 0 and held grown capacity for the pool's lifetime; the resolution
  below replaces that stub.
- Reproduction: `scripts/prefix_cache_multiturn_bench.py` with
  `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS=16` on Qwen3.6-35B-A3B. That arm
  completes, and after the placement cost model below it measures **-11.0%
  wall** where it first measured +9.5%. The default retention is still better
  (-18.6%), so the wide set stays opt-in, but it is no longer harmful.
- **Resolved in the refactor sweep:** the global device pool now retracts free
  trailing growth units after the configured idle grace period, preserving the
  load-time capacity. Chunk and VMM providers publish retained pointer tables
  before releasing superseded storage. Live leases, cache references, pins and
  in-flight graph epochs block reclamation. Unrecoverable VMM restoration faults
  stop further admission instead of advertising unmapped storage. Regression
  coverage includes repeated grow/shrink cycles and provider failure paths in
  `tests/test_unit_kvcache_global_device_pool.py`,
  `tests/test_unit_gguf_global_pool_growth.py` and
  `tests/test_unit_gguf_vmm_kv_pool.py`.


## Speculative candidate-budget default vs qualified depth (found 2026-09-17)

- Resolved 2026-09-17: an omitted `--speculative-candidate-budget`
  (`HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET`) no longer defaults to 4. The server
  resolves the depth from the loaded model's retained serving evidence (deepest
  qualified depth for the resident physical cell), or from the generic
  provider's declared shape (Laguna keeps B4), and reports requested/resolved/
  source at startup and in `/v1/hipengine/capabilities`. A pinned depth is used
  unchanged and an unservable one is warned about instead of silently
  rewritten. See `docs/ENVS.md` and
  `worklog/entries/20260917T113608.088508Z-lhl-mtp-candidate-budget-capability-f876c3.md`.
- Fixed 2026-10-26: `LLM` publishes the resolved depth through service wrappers
  to the resident generator, including when the depth was resolved before the
  owner was constructed. Dense C1 generation and workspace sizing consume that
  published value rather than rereading the fallback environment variable.
  The shared resolver now reads `HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET` for
  both serving and direct generators. Explicit values precede automatic
  defaults; malformed, conflicting or unsupported values fail loudly.
- `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` remains a compatibility alias because
  external direct-generator callers used the documented setting. It is not a
  second fallback: both names must agree when set together. Live repository
  scripts use the canonical name. Remove the alias only with an explicit API
  deprecation; deleting it silently would change existing callers' requests.
- Resolved 2026-10-26: provider registrations declare their candidate budget.
  Server pre-load reporting and `LLM` use the same registry accessor. Explicit
  and engine-resolved depths take precedence. A declaration that varies by
  registry axes reports pending before load; construction uses its known axes
  to choose the depth. An unregistered provider still fails by name.
  The generic config default remains for direct provider construction.

## MTP output attribution: two AR-in-plan paths and a duplicated telemetry write (found 2026-09-17)

- Resident dense and MoE MTP adapters attribute AR-in-plan output through one
  counter and one AR-step function, including depth-zero cycle commits.
  `ar_output_tokens_in_cycles` keeps its public meaning; no zero-token call is
  needed to avoid double attribution.
- Resident streaming and blocking outputs publish draft counts once in
  `diagnostics.specdec2_mtp2`. Server usage, metrics and response summaries read
  this per-request record independently of batch timing ownership. The public
  `output_accounting` block still separates MTP and AR output.
- Legacy providers still publish provider diagnostics or scalar `mtp_*` timing.
  These remain compatibility readers, not duplicate resident publication.
  Remove each fallback when its provider publishes the common per-request
  diagnostics contract and its streaming and blocking consumers are migrated.

## Speculative-MTP serving evidence rows (shape-axis removal, 2026-09-17)

- The three shadowed physical-cell rows were removed after checking resolver
  selection and default depths:
  `qwen38-q4km-gfx1151-production-bf16-c1-b3-context128`,
  `qwen38-q4km-gfx1151-production-bf16-cap4-c1-intent-k3-d24`, and
  `qwen36-dense-q4km-gfx1100-strict-bf16-c2-k2-d24`. Their artifact links remain
  on surviving evidence rows. Sampled c1–c4 declarations now derive from the
  automatic cap4-realized row; their physical axes and default depth are
  unchanged. The explicit-versus-automatic resolver test uses a synthetic
  duplicate instead of requiring a redundant production declaration.
- `scripts/gguf_mtp_c1c8_server_bench.py::_diagnostic_plan` and
  `scripts/qwen38_packet5_k4_watchdog_probe.py::_inject_k4_evidence_row` keep
  their own diagnostic scopes. The watchdog probe injects a row for the width
  it screens; if a future probe needs a *distinct* authorization on a cell that
  a registered row already covers, it must say which row wins rather than rely
  on declaration order.
- Campaign documents written before 2026-09-17 (`docs/CONCURRENCY2-*.md`,
  `docs/campaigns/QWEN36-27B-GGUF-CAMPAIGN.md`, `docs/MTP-*.md`) describe the admission
  scope in shape/horizon/manifest terms. They are historical records and were
  left intact; only `docs/API.md`, `docs/EXECUTION-PROFILES.md`, and
  `docs/archive/COMPARISON-vs-atlas.md` state the current contract.





This file tracks cleanup work that should happen after the fast/correct path is
proven. During optimization, temporary flags and fallback paths are useful for
bisection; after the optimal path stabilizes, they become dispatch confusion and
should be removed or collapsed.

## Policy

- Exact, same-suite non-regressive performance wins should become defaults.
- Production-profile T1/T2 wins may become profile defaults only after exact
  control/ownership plus the complete strict-teacher tail-KL/top-1,
  determinism/isolation, BF16-relative, and task gates pass.
- Keep opt-out flags only while they are useful for rollback, bisection, or a
  named validation gap.
- When a flag is left in place, record the removal trigger here.
- Do not remove registered strict unfused fallbacks required by `AGENTS.md` and
  `EXECUTION-PROFILES.md`; remove dead runtime dispatch branches and stale
  experiment toggles first.
















## 2026-08-28 Qwen3.8 production verifier-Q4 selector — open

- `HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE` currently carries the resolved
  Qwen3.8 production-profile decision from the cold-path binder into resident
  sessions. The session freezes it once; hot verifier methods consume only the
  typed boolean and never reread the environment.
- Remove the environment handoff when `ResolvedRuntimePlan` (RF-1) carries
  profile scope selections directly into `Qwen35GGUFResidentSession`. Preserve
  the profile manifest selections, strict WMMA fallbacks, qualified
  R6/R8/R9/R12 actual-shape bounds, and benchmark rollback evidence.
- R0184 prerequisite audit: `qwen35_dense_gfx1151_profiles._binder` still writes
  the environment handoff, and `Qwen35GGUFResidentSession` freezes it after
  resolving its backend. `_configure_session` is not a universal construction
  hook: direct generation and NextN also construct resident sessions. Replace
  the handoff with one typed construction input propagated through shared,
  direct, batch and NextN paths, plus explicit strict/candidate inputs in the
  numerical harness. Test two interleaved profile owners and nested diagnostic
  restoration before retiring the variable; do not substitute a process-global
  boolean or infer this arithmetic choice from the backend alone.

## SPECDEC2 MoE request-local C2 prototype

- Added 2026-08-27 as the first functional physical-C2 seam in
  `Qwen35GGUFMoEMTP2Adapter`. Multi-slot model-plugin construction selects
  the architecture-shaped `Qwen35GGUFMTP2Adapter` with a batched MoE NextN
  executor, so the prototype C2 proposal/accept implementation in the old MoE
  adapter is unreachable through the factory; capacity 1 still uses its
  complete-cycle implementation. Factory coverage includes capacities 2, 4 and 8
  and fails if any of them constructs the old adapter.
- The legacy adapter no longer advertises staged C2 capability or claim fit.
  Its obsolete C2 proposal body is removed; the required lifecycle method now
  reports that this adapter owns complete C1 cycles. Multi-slot production
  proposals continue through the generalized adapter.
- The obsolete staged target body, provider repair, batch-accept workspace and
  upload helpers are removed. Required staged interface methods explicitly
  refuse direct calls; the complete C1 cycle, shared slab accounting, failure
  recovery and release remain. Upload contiguity/capacity coverage exercises the
  generalized adapter now used by multi-slot MoE.
- Remaining validation: exercise the batched provider through public C2 serving
  with cancellation, rejection and resource cleanup before closing this entry.

## SPECDEC2 MoE physical-C2 numerical scope

- Added 2026-08-27 for the gfx1100 35B MoE physical-C2 campaign. The request-
  local `moe_physical_c2_numerics_session(...)` disables the C1 production
  F32-residual diagnostic only while the packed C2 target executes; the packed
  owner cannot represent that diagnostic and fails closed without the scope.
  C1 production and strict retain their existing arithmetic. The gfx1100
  production BF16 C2/K2/D24 row is already automatic-eligible in the model
  policy, backed by
  `benchmarks/results/2026-08-28-w7900-35b-moe-mtp2-c2-automatic-promotion.json`.
- Remaining cleanup: replace the temporary ContextVar with a profile-resolved
  physical-C2 target manifest, preserving the C1 production and strict
  fallbacks plus the focused scope test. Qualification is not the remaining
  prerequisite for that transport cleanup. Preserve nested disabled scopes,
  exception restoration and independent execution-context isolation when moving
  the selection to a profile-owned plan; these are existing ownership contracts.

## SPECDEC2 MoE physical-C2 exact-linear fallback

- Added 2026-08-27 after the packed rows6 linear-attention composite failed the
  binding MoE production gate. The request-local
  `moe_physical_c2_exact_linear_session(...)` selects the row-exact target path.
  The initial scalar-heavy fallback was subsequently optimized: the promotion
  artifact records R6 norm/projection, two R3 Conv segments and segmented GDN
  journals, followed by row-batched MoE FFN. See
  `benchmarks/results/2026-08-28-w7900-35b-moe-mtp2-c2-automatic-promotion.json`.
  The old claim that this path is inherently too slow for automatic promotion
  does not describe that implementation.
- Remaining cleanup: carry the exact-linear selection in the profile-resolved
  target manifest while preserving selected-state journals, C1 arithmetic and
  nested/context-local restoration. Any replacement kernels must independently
  pass the complete numerical and lifecycle gates and the original >=1.10x AR
  performance target; do not remove the correctness
  fallback merely because its original scalar implementation was slower.

## gfx1100 dense SPECDEC2 C8 physical-width rollback

The maximum-request environment override is removed from server admission and
adapter construction together. Backend width/depth policy, actual owner capacity,
derived buffer bounds and strict-profile fallback remain unchanged.

**Remaining validation:** exercise `hipengine serve` at C5/C8 using production
D1/D24 workloads, confirming physical grouping, deterministic/isolation behavior,
allocation-failure cleanup and cancellation. Preserve the original complete
category and control/ownership gates before closing this entry.

## Server default-AR ready-cohort rollback

- Added 2026-08-31 with task #29. Default-route HTTP items already inside the
  configured generation batch window now enter one
  `EngineService.submit_request_batches` command, so every child is admitted
  before the next model poll while retaining its own collector, cancellation,
  completion, backpressure, and reclaim lifecycle. Dynamically arriving items
  use free capacity in a later cohort. The environment bypass is removed;
  generators without the ready-cohort API retain independent submission.
  Non-independent generators, MTP and streaming routes are unchanged.
- Remaining validation: repeated C1-C3 public server runs must confirm full
  native-prefill groups, exact outputs, independent early completion/failure/
  cancellation and no C1 regression. The atomic EngineService cohort seam and
  per-child handles remain; integration lifecycle tests alone do not establish
  public native-prefill engagement.

## gfx1100 fused Q4-T16 prefill row-retile bisection

- Added 2026-08-31 with task #37. The separately registered standard-Q4
  gate/up+SiLU row48, row64 and row128 siblings preserve the 256-row parent's
  K16 WMMA, FP32 accumulation, BF16 projection boundary, and SiLU association.
  Backend policy selects row48 at rows33-48, row64 at rows49-64 and row128
  at rows65-128; rows129+ and every quant/shape/backend miss retain the
  parent. The environment lookup/cache is removed; the diagnostic prefill
  comparison uses a scoped selector override restored in `finally`.
- Remaining validation: defaults-only repeated C1-C3 public serving gate,
  preserving the bounded shape/range policy and permanent 256-row parent.
  Route tests and direct numerical checks do not establish public engagement.

## gfx1100 Q4-T16 row64 down-projection bisection

- Added 2026-08-31 with task #11. The exact `(17408,5120)` rows33-192 owner is
  selected by backend shape/range policy; the separately registered 256-row
  shared-B parent remains the strict fallback.
  The redundant environment cap and cache are removed; admission still follows
  the backend's shape and row-range declarations.
- Remaining validation: a defaults-only same-protocol public repeat, preserving
  shape/range policy and explicit registered parent fallback. Direct row64/parent
  GPU parity and dispatch tests do not close public serving validation.

## gfx1100 planar-Q6 exact prefill retile bisection

- Backend policy selects planar-Q6 row64 at `(17408,5120)` rows4-128 and
  shared256 at rows129-511; `(5120,1024)` uses row64 at rows4-128.
  The environment lookup/cache is removed. The registered one-wave parent
  handles shape/range misses; decode and peer backend owners are unchanged.
- Remaining validation: a defaults-only same-protocol C1-C3 public repeat with
  all categories, heldouts, exact trajectories, native groups, trace identity
  and drain checks. Preserve backend policy and registered one-wave fallback;
  direct numerical parity does not establish public engagement.

## SPECDEC2 dense NextN host/device proposal duplication

- Host-materialized and device-top1 batch entry points now call
  `_enqueue_batch_hidden_transition` for embedding, normalization, fusion and
  backbone execution. Each adapter retains token staging, synchronization/scoring
  options, cursor publication and result materialization. The host oracle remains
  callable without a duplicate model-launch body.
- Remaining validation: public device accept/commit, graph/eager agreement and
  cancellation/cleanup checks. CPU transition-order and provider tests preserve
  the refactor contract but do not establish public GPU execution.

## SPECDEC2 P4 device-chain qualification oracle

- Added 2026-08-25 with SPECDEC2-PERF P4. The promoted C1/C2/C4 cycle keeps
  proposal and target IDs device-resident through GPU accept and selected
  commit, then reads only bounded committed output/status. Explicit diagnostic
  construction with `device_chain_qualification_oracle=True` adds post-commit
  candidate/target-ID materialization and CPU acceptance comparison. The
  environment read is removed; default adapter instances remain unaffected.
  This option enables checks, not model admission or certification.
- Removal trigger: after P4 reject/every-partial/full, following-AR,
  cancellation/failure, profile, and full category gates are retained with
  GPU/CPU accept agreement and cache-only traces prove no pre-accept D2H in the
  default route, move the oracle body to a test/qualification adapter. Keep the
  explicit diagnostic constructor route until that replacement is available,
  the pure CPU acceptance implementation, and strict eager fallback.

## SPECDEC2 request-local ngram-mod experiment

- Added 2026-08-28 for the coding/repetition composition spike.
  Explicit `ngram_enabled=True` and `ngram_config=NgramModConfig(...)` give
  the request-local continuation cache first refusal before dense GGUF MTP2.
  Environment parsing is removed; typed defaults retain a 24-token match/minimum
  and 64-token probe bound. A hit verifies only
  the already-qualified K<=3 prefix, then catches target-attached MTP state up
  from target-owned hidden rows. A miss, mixed-source physical group, disabled
  option, unsupported shape, or failure keeps the registered ordinary MTP/AR
  paths. Exact request-local keys intentionally replace llama.cpp's shared
  modulo pool to avoid collisions, cross-tenant history leakage, and
  benchmark-order/batch-composition contamination.
- Current verdict: the full canonical production D24 suite is exact with 142
  lookups / zero hits and 0.99915x MTP-only throughput. The independent strict
  C2/K3 D80 repetition-heavy code suite is exact and improves MTP-only
  **20.434 -> 20.930 tok/s (+2.425%)**, with train/heldout both +2.42%, but
  remains 0.9875x true AR. Strict D96 also fails one SQL terminal-suffix cell,
  and the prior production D120 numerical tail remains rejected. Keep explicit
  and default-off; there is no automatic product cell.
- Remaining promotion trigger: a K>3 frontier (or another materially different
  lowering) must pass canonical + code train/heldout, long-horizon strict and production
  numerical/task/lifecycle gates, and beats true AR rather than MTP alone.
  Otherwise remove runtime composition at the end of the next dedicated wider-
  frontier campaign and keep the cache/harness as research references. Ordinary
  MTP plus AR remain permanent miss/failure fallbacks.

## Execution-profile migration seam

- Implemented for campaign P3: public `strict|production|batch_invariant`
  selection resolves to an immutable variant manifest through
  model/backend/quant plugin plans, with per-scope strict fallback and exact-key
  registry validation. `LLM.execution_profile_manifest_sha256` and server model
  discovery expose the resolution. `HIPENGINE_EXECUTION_PROFILE` is the sole
  environment adapter; explicit API/CLI values win.
- An omitted profile resolves to `production` when both strict and production
  plans are registered, or to `strict` when that production plan records a
  qualification failure. If either plan is missing, resolution returns `None`
  and preserves pre-profile package selection. This compatibility path is not
  a fourth named profile and does not prevent the implemented model from running.
  `tests/test_unit_execution_profile_defaults.py` and the runtime/quarantine
  tests cover these distinct cases; do not replace the fallback with blanket
  production selection for combinations without plans.
- Removal trigger for the remaining legacy path: after P4 re-certifies or
  replaces every current non-exact default so that every shipped
  model/backend/quant has a registered plan, and one release window confirms
  strict fallback and batch-invariant behavior. Then remove omitted-profile
  legacy selection and any duplicate env-to-profile adapter. Keep registered
  strict fallbacks.

## Generation-2 GGUF width-selection hardcoding

- Resident shared-slot AR widths come from the backend's
  `GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS` declaration; the process-wide width
  override is removed. The lifecycle diagnostic reads the same declaration.
  The standard non-resident implementation supports widths 1 through 8.
- The artifact-backed D2 cost resolver is an explicit planner experiment through
  `HIPENGINE_GGUF_AR_D2_COST_ARTIFACT`; ceiling remains the production default.
  Its earlier promotion sweep used eager resident sessions, not the server,
  and recorded a dirty composed-c13 lifecycle result. A new D2 promotion
  campaign is optional performance work, not a capability or integration gate.
- Primitive selection is spread across quant/layout capability maps and
  per-width `CASE(N)` launchers. Q5T16 and planar-qmicro Q6T16 were promoted to
  true rowtiles through 8 (2026-08-20,
  `GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT[gguf_q5_k_t16_v1]=8` and
  `[gguf_q6_k_t16_qmicro_planar_v1]=8`); the planar-Q6 export no longer falls
  to per-row for rows 5+. The separate FP32 LM-head rowtiles publish their own
  primitive bounds (standard Q6=6, planar-qmicro=8), which the chunk planner
  intersects with backend package policy (8 on both HIP backends). Remaining
  generic fallback ladders still demonstrate why one model-wide max-row value
  or variant name is not honest coverage.
- Removal trigger: compact D1/D2 schemas, strict group-cost map, and direct
  c3/c5/c6/c7 lifecycle are present. Still required: clean actual-server
  direct/masked/composed c1-c32 route telemetry; counterbalanced same-shape D2-
  vs-ceiling goodput and TTFT/ITL; refill/cancel, memory, and drain; then cold
  package resolution to ordinary registry/plugin evidence. Keep registered
  strict kernels and ceiling fallback until those gates pass; only afterward
  remove duplicate sidecar cases, generic row caps, and manual ladders.

## Generation-1 GGUF single-backing KV compatibility path

- Ordinary resident serving selects `create_global_device_kv_pool` when the
  session implements it. Its `GlobalDeviceKVPool` wraps `GlobalKVPoolSet` and
  stable per-plane page tables. The pool grows on demand and releases wholly
  idle grown tails; it is not a fixed allocation with no growth or shrink.
- With working HIP virtual memory, growth extends stable per-plane arenas and
  rebuilt backing objects retain their arena identity. Without that capability,
  the same global adapter appends ordinary device chunks. Base-pointer consumers
  then need each allocation within one backing chunk; pointer tables alone do
  not remove that execution constraint. Unsupported VMM symbols or a driver
  returning `hipErrorNotSupported` select this implemented fallback.
- `DeviceChunkedKVPool` also has an active DMS `dense_pool` consumer: the session
  allocates transient dense prefill storage before compact KV is built. DMS
  `layer_outer` does not use that pool. The generation runner retains a
  legacy-factory branch for session providers that implement only
  `create_device_kv_pool`.
- Removal work: migrate remaining base-pointer consumers to page-table storage
  views across appended chunks, replace the DMS transient-pool owner, and then
  remove unused legacy factory and identity branches. Preserve same-arena checks
  until snapshot restore can establish equivalent storage ownership. Missing
  benchmark coverage does not prevent these paths from running.
- Validate replacements with
  `uv run pytest tests/test_unit_kvcache_global_device_pool.py
  tests/test_unit_gguf_vmm_kv_pool.py tests/test_unit_gguf_global_pool_growth.py
  tests/test_unit_gguf_device_kv_binding.py
  tests/test_unit_gguf_prefix_cache_runner.py -q`, then real ordinary and DMS
  public requests covering growth, prefix reuse, cancellation and teardown.
  Keep registered strict/serial numerical fallbacks independently of allocator
  cleanup. Shared-slot width selection is tracked separately above.

## C2-6 production oracle process-partition seam

- Added 2026-08-17 after repeated W7900 page faults when a 13-key exact-oracle
  campaign handed the same HIP generation to measured serving. The production
  load harness can partition oracle keys into isolated workers, verify merged
  token digests, and start a separate measured owner. This is benchmark
  containment, not runtime serving architecture.
- The earlier W7900 oracle/handoff fault was resolved by invalidating decode
  graphs before packed prefill reused their workspace. The accepted canonical
  packet is recorded in
  `benchmarks/results/2026-08-18-concurrency2-c2-6-w7900-canonical-production-accepted.json`;
  the subsequent
  `benchmarks/results/2026-08-19-concurrency2-pipeline-verify-skip-tuning-accepted.json`
  also completed with same-owner serial oracles. These are historical results,
  not a current-source rerun or proof that every worker mode is equivalent.
- The harness defaults to `--oracle-mode same_owner`, disables decode graphs
  only during reference generation, and checks that requests drain before
  serving. Explicit `isolated` mode still partitions reference keys into workers
  and verifies merged key coverage. Neither mode is runtime serving architecture.
- Remaining cleanup: choose one independent-oracle protocol and remove the other
  implementation and CLI mode once no diagnostic caller needs it. Preserve exact
  token coverage, explicit worker failures and final ownership checks. Run
  `uv run pytest tests/test_integration_gguf_production_load_gate.py
  tests/test_unit_concurrency2_production_load.py -q`, then the selected real
  `scripts/gguf_production_load_gate.py` packet before deleting a protocol. The
  old device fault is not an outstanding runtime blocker.

## C2-6 guarded idle packed-workspace release API

- Added 2026-08-18. Row reclaim no longer frees the owner-shared packed
  workspace: the slab is union-geometry and shared by every resident view, so
  per-reclaim release forced a same-size hot-path reallocation on the next
  packed step (the accepted C2-6 canonical packet recorded 246 releases /
  242.39 GiB cumulative churn). The guarded
  `Qwen35GGUFResidentSession.release_idle_packed_workspace()` is now used by
  production KV-pool teardown during rebind, auto-context retry and close.
  It fails closed on unflushed state or live graphs and releases workspace
  ownership before the pool closes; it is not an unused API.
- Packed state scratch and private KV already reserve named byte charges in
  the global pool before allocation and release those charges on failure or
  teardown. The remaining ownership cleanup is to move the session-held
  workspace references and teardown orchestration into a dedicated resource
  owner, not to add accounting a second time.
- Keep the guarded session API until every teardown caller uses that replacement
  owner. Preserve scatter/graph quiescence, release scratch and its pool charges
  before pool close, and do not restore per-row reclaim releases. Validate with
  `uv run pytest tests/test_unit_gguf_packed_workspace_stability.py
  tests/test_unit_gguf_packed_verify_layout.py
  tests/test_unit_gguf_global_pool_growth.py -q` and public cancellation/reuse.

## Packed workspace page table and identity-table device kernels

- Added 2026-08-18 (task #5 stages S1b/S1c). `_GGUFPackedTargetState` carries
  slot-major `page_ids` + `copy_segments()`, and
  `_rebind_packed_verify_layout_pages` fills layout block tables from them.
  Production includes pool-leased, non-identity page IDs. Eager metadata
  preparation uploads those tables rather than rebuilding identity mappings.
  Packed decode graphs now preserve the staged authoritative table and redirect
  the identity-writing refresh output to graph-owned discard storage; positions
  and counts still update on device. Identity layouts keep the original path.
- Pool growth/shrink invalidates captured graphs for both self-owning sessions
  and borrowed views before pointer tables can move. Discard storage is released
  on failed capture and graph close.
- Remaining cleanup: delete private-chunk allocation from
  `_GGUFPackedTargetState` only once lease-backed allocation supports every
  current producer and fallback case. Arbitrary-page graph execution is no
  longer blocked on a future metadata-kernel extension.



## Qwen4Exp decode routes of the UD production plan

The `gguf_ud_q4_k_xl` production manifest declares GDN decode and two
conditional MoE warp choices; a bit-identical decode fusion is on by default.
The gate/up warp choice only applies when DP4A is enabled separately. The
current production recovery plan keeps DP4A off after the recorded decode
discrepancy, so ordinary production decode does not execute that warp kernel.
The historical measurements below used a Radeon 8060S on the canonical
`code-p512` / `code-p4096` cases (128 decode transitions).

- The profile's `decode_c1_d128` row selects `qwen4exp_gdn_rowsplit_decode`
  with `qwen4exp_sigmoid_strict` as its fallback. The binder publishes
  `gdn_decode_rowsplit` on the owning runner; the old environment key is
  ignored. Diagnostics override one runner through the scoped control.
  The kernel uses one 512-thread block per value head,
  four lanes per column, the state read and written once with both dot
  products in registers. Kernel 66 -> 27.5 us per layer; decode 20.87 ->
  21.47 tok/s at p512.
- The `decode_c1_compact_le12_q51_down` and
  `decode_c1_compact_le12_q4_dp4a_enabled_gate_up` manifest rows declare
  `selected_weighted_sum_warp256_fast_bf16_bf16_out` (Q5_1) and
  `selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out` (Q4_K).
  The binder publishes `moe_decode_warp` to scalar and batch runners; their
  graph cache keys include the selection. The old environment keys are ignored.
  Geometry and registration checks preserve the parent fallback. Diagnostics
  use a scoped runner override, without changing the DP4A prerequisite.
  These are the warp-per-expert routed-expert kernels. Down 124 -> 73 us, gate/up 128 ->
  114 us per layer; decode 21.47 -> 22.65 -> 23.00 tok/s at p512.
- The one-row hyper-connection read uses three launches instead of six and is
  bit-identical (`tests/test_gpu_qwen4exp_gr_read_decode.py`). Its environment
  opt-out is retired; unsupported geometry or weight layout still uses the
  strict chain. The live MTP test selects that chain with an explicit test
  override rather than a process-wide control.

The layer2 production gate on the three reduction-order routes together
(commit `542787507`): mean/p95/p99/max KL 6.36e-4/3.64e-3/8.07e-3/9.88e-3,
top-1 446/450, three bit-identical repeats, 13/18 strict-exact tasks
(`benchmarks/results/2026-10-14-qwen4exp-decode-batch/gate-a.json`).

The two warp kernels are the ones the "MoE Decode warp256 GEMV Pair"
section below records as removed on 2026-09-30. That removal judged the
gate/up kernel on an 18-case, 1,494-row stress suite whose 0.05 max-KL bar
the then-default plan also failed, and judged the down kernel's bit-exact
sibling (which is slower) rather than this vectorized one. Under the
binding layer2 gate the pair passes, so the kernels are registered for
`hip_gfx1151` and selected by the binder; the scripts that section deleted
stay deleted.

Not kept: the DP4A gate/up owner at all 48 layers instead of 43 (+2.2%
decode, but top-1 445/450 against the 0.99 floor, `gate-b.json`; reverted),
and a two-launch read that rebuilt the norm in every block (bit-identical,
1.4% slower).

Resolved: the reduction-order choices have explicit manifest scopes and
runner-owned controls. The GR fusion opt-out is retired. No process selector
remains for this unit; strict fallbacks and the separate DP4A prerequisite remain.

## Qwen4Exp direct QSA prefill selection (`HIPENGINE_QWEN4_EXP_QSA_DIRECT_SELECT`)

Resolved: the process rollback is retired. H4/D128 batched prefill always uses
causal scoring and, with WMMA attention, direct mask publication. Other
geometries retain the generic scorer, and per-row selection retains its mask
conversion. Only the final sparse row expands positions on the direct-mask
route to publish decode state. The generic kernels remain strict comparison
oracles in `tests/test_gpu_qwen4exp_qsa_select_mask.py`.

## Qwen4Exp QSA prefill gate read in place (`HIPENGINE_QWEN4_EXP_QSA_STRIDED_GATE`)

Resolved: the process rollback and copied-gate prefill branch are removed.
Prefill reads the projected gate in place and publishes an F16 mirror when the
existing workspace capacity permits. The separate gate kernel remains used by
scalar decode and as the strict oracle in
`tests/test_gpu_qwen4exp_qsa_select_mask.py`.

## Qwen4Exp residual write fused into the next norm (`HIPENGINE_QWEN4_EXP_GR_WRITE_NORM`)

Resolved: the process rollback is removed. The existing 1024-row floor,
pending-write handoff and separate-write fallbacks are unchanged. The token
mixer's residual write can fuse into the FFN norm, and the MoE write into the
next layer's attention norm. PLE boundaries, the final layer and incompatible
consumers still flush the write separately. Exact fused/separate parity is
covered by `tests/test_gpu_qwen4exp_gr_write_norm.py`.

## Qwen4Exp decode PLE staging overlapped with layer 0 (`HIPENGINE_QWEN4_EXP_PLE_OVERLAP`)

Resolved: the process rollback is removed. Pinned staging overlaps preceding
layers when the first PLE consumer is after layer 0. Unpinned staging and a
layer-0 consumer keep the blocking copy. Host gathering and same-stream copy
ordering are unchanged; device-side gathering is a separate implementation
change, not a prerequisite for retiring the switch.

## Qwen4Exp MTP batched verify (`HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY`)

Resolved: the environment rollback is removed. Runners implementing
`verify_target_block_state_rows` use batched verification and commit accepted
state rows without replay. Serial verification remains functional for a
single-token tail and external runners without that interface; it is not dead
code and must not be removed with the switch.

## Eight-column pack8 kernel behind the row-tile launcher (`HIPENGINE_GGUF_K_PACK8_ROWS`)

Resolved: the process rollback is removed. The row-tile launcher selects
pack8 when the output width is a multiple of eight. The one-column kernel
remains the functional fallback for all other positive output widths; removing
it would narrow the supported shapes. Both kernel bodies are unchanged.

## Batch manifest legacy substring binding

Resolved: removed the substring helper, static identity table and legacy
fake-scope tests. Adapter evidence reports no path identities until the
structured manifest is built. The layer-2 gate continues to validate live
identities, registry and dispatch keys, step widths, and profile/source binding
through `hipengine.runtime.qwen4_exp_batch_manifest`. Its c1-only rejection
and identity-tampering tests remain.

## Deferred decisions from the refactor sweep

### R0067 — Dense prefix retention budget

The eviction repair is complete. Dense prefix retention defaults to
`max(1, capacity)` snapshots, bounded separately by the state-byte ceiling;
retained KV pages also consume memory outside that ceiling. Wider retention
is implemented and explicitly selectable. DMS uses its own wider count and
host-state budget.

**Decision needed:** whether the dense default should favor a wider reuse
working set at the cost of additional pinned memory. This is not blocked on
qualification. **Clearing action:** the lead selects the default memory/reuse
trade-off; update the count default, its tests and user-facing configuration
reference together, then validate interleaved conversations and memory pressure.
No change to the default is made by this cleanup.

### R0085 — Layer-outer prefill rollback control

`HIPENGINE_GGUF_PACKED_LAYER_OUTER` defaults on. Its off value still selects
chunk-outer prefill and is an explicit control in
`scripts/gguf_p6e_cancel_refill_proof.py`. Chunk-outer execution also handles
real declines, including prefix boundaries and unavailable resumable owners;
removing the rollback control must not remove those paths.

**Decision needed:** retire the production rollback control, or keep it for
operational bisection. **Clearing action:** if retired, remove the environment
reader and its dispatch condition, migrate the service proof's explicit
control to its test harness, update CLI/configuration documentation, and run
`tests/test_unit_gguf_packed_layer_outer_prefill.py`,
`tests/test_unit_gguf_resumable_layer_outer_prefill.py` and
`tests/test_unit_gguf_p6e_cancel_refill_proof.py`, preserving decline coverage.
The existing promotion artifacts are
`benchmarks/results/2026-09-11-w7900-p3-promotion-ab-{chunkouter,layerouter}-{1024,2048,4096,8192}.json`
and `benchmarks/results/2026-09-11-w7900-p3-promotion-server-c1-{off,on}.json`.
The harness catalogue's stale default-off and failing-proof statements are
corrected; this cleanup does not change runtime selection.

### R0135/R0136 — Unselected dense integer-MMQ alternative

W4A16 is the dense raw-IQ prefill default in `GGUF_IQ_DENSE_PREFILL_POLICY`
on both HIP backends, including the cooperative variants on gfx1100. The
former W4A16 promotion and four-quant integer-MMQ policy entries are obsolete;
the strict-referenced comparison and default change are recorded in
`worklog/entries/20260908T191409.775774Z-lhl-ud-w4a16-default-6c43c5.md`.

The integer alternative
`dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out` remains registered
but unselected. Its workspace is allocated only when a policy selects it.
**Decision needed:** retain this explicit comparison alternative or remove it.
The original default-change record deliberately kept it for comparisons;
this sweep does not silently reverse that decision.
**Clearing action:** if retired, remove its registrations and route-specific
workspace/build plumbing, migrate the route-swap tests, then run
`tests/test_gpu_gguf_iq_dense_mmq_route.py` and
`tests/test_unit_gfx1151_iq_dense_policy_parity.py`. Preserve W4A16 and strict
fallbacks. If retained, record its diagnostic owner and close the cleanup.
The recorded integer-MMQ numerical failure is in
`benchmarks/results/2026-09-09-zbook-ud-production-gate-reference.json`.

### 2026-09-02 Qwen4Exp P9 load drop-behind

R0143 disposition: deferred product decision.

`HIPENGINE_QWEN4_EXP_LOAD_DROP_BEHIND=1` is an explicit default-off one-shot
serving control. It applies file `DONTNEED` only after successful device
ownership of non-PLE tensors; lazy PLE and validation readers are excluded.
The implementation and exclusion test pass; the ENVS description is corrected.

**Decision needed:** keep one-shot low-RSS deployment as a supported mode or
retire it. **Clearing action:** if kept, retain the explicit control (or replace
it with a named loader policy) and close this cleanup. If retired, remove the
flag and advice branch, update ENVS, and run
`tests/test_live_qwen4_exp_materialize.py` with its copy-order/PLE-exclusion
coverage. Do not turn it into a reload-heavy default: the original controls
forced 13.75-15.61 GB of request reads and about 6.1K major faults after load.
No loader behavior changes in this disposition.

### R0249/R0250 — Laguna public cache-owner validation

The obsolete resident-cache and activation-reuse opt-out arguments are removed.
Backend capabilities still select the owners; resource, unsupported-backend and
unshared-cache fallbacks remain. Focused tests, the CPU bundle and resident-cache
GPU checks pass. Historical results remain in the sweep inventory and worklogs.

**Dependency:** a local Laguna checkpoint and a gfx1100 runtime for public owner
engagement. No checkpoint is present at the repository's documented local paths.
**Clearing action:** run `python scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1100 --max-tokens 8 --output /tmp/laguna-public.json`
in the ROCm environment. Also exercise a sufficiently long public prefill to
engage the ordered-F32 owner and confirm resident-cache and activation-reuse
selection, blocking/stream/direct parity, and cleanup. Until then this is a
validation deferral, not a claim that the public path passed. It does not disable
or restrict the implemented paths.

### R0273 — Laguna public packed-query validation

The redundant packed-query constructor selector and route-rebuild setter are
removed. Backend capability metadata selects the layout; the whole-BLAS
rollback, unpacked primitive and unsafe-shape `KVLiveSpans` fallbacks remain.

**Dependency:** the same unavailable Laguna checkpoint noted above, on gfx1151
for this packed-query route. **Clearing action:** run
`python scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1151 --max-tokens 8 --output /tmp/laguna-packed-public.json`
in the ROCm environment, with a prompt reaching the dense-initial BLAS path.
Confirm packed-query selection, blocking/stream/direct parity and resource
cleanup. Do not infer public engagement from leaf numerical tests.

### R0275 — Laguna public softmax-owner validation

The redundant softmax constructor override, rebuild setter and one-purpose
comparison harness are removed. Backend selection and the block256 numerical
fallback primitive remain unchanged.

**Dependency:** the unavailable Laguna checkpoint noted for R0273.
**Clearing action:** run the R0273 gfx1151 public command with a dense-initial
BLAS prompt, additionally confirming that the backend-selected wave-per-row
softmax executes. Check blocking/stream/direct parity and teardown. Existing
packed/wave/block leaf comparisons do not establish this public engagement.

### R0274 — Laguna public packed-query producer validation

The producer constructor override and its dedicated comparison arm are removed.
R0336 repairs the separate output-gate diagnostic: both arms now hold the packed
producer on through a scoped package override restored on construction success
or failure. Its former generic-producer setting made both output-gate arms
use generic output. Backend selection and generic-row fallbacks are unchanged.

**Dependency:** the unavailable Laguna checkpoint noted for R0273.
**Clearing action:** run the R0273 gfx1151 public gate with a dense-initial BLAS
prompt and confirm direct packed-query production for the eligible tiles,
generic production for other rows, output parity and clean teardown.

### R0248 — Laguna public cache regression after H7Y retirement

The rejected H7Y map, mirror owner, writer and attention kernels are removed.
Promoted H6Z/H6W and their generic fallbacks remain; historical H7Y rejection
artifacts are unchanged.

**Dependency:** the unavailable Laguna checkpoint noted for R0249/R0250.
**Clearing action:** run their gfx1100 public generation gate with a prefill
long enough to exercise H6Z/H6W, confirm the selected variants and natural-cache
allocation lifecycle, and compare blocking/stream/direct outputs. The removed
H7Y route must not be selected or allocate mirrors.

### R0276 — Public Q4/Q6 tile-policy validation

The Q4/Q6 environment overrides and dense-Q4 runtime setter are removed.
The category benchmark owns its validated session-local comparison control;
backend defaults and primitive fallback paths remain unchanged.

**Dependency:** the unavailable Laguna checkpoint.
**Clearing action:** run the R0273 gfx1151 public gate with
`HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE=64x16` set and trace Q6 WMMA launches.
Confirm the backend's default tile still runs, outputs match the unset control,
and teardown is clean. Repeat with `HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE=64x16`
and confirm shape-selected Q4 tiles are unchanged. Explicit primitive tile
arguments remain supported.

### R0271 — Scaled F16 comparison retention and public validation

The runtime setter and norm-direct mode are removed. Production range-direct
construction validates dense-F16 layout and finite norm/gated-value bounds.
Historical publication mappings remain readers of old artifacts, not execution
selectors.

**Decision needed:** keep or retire the `prefill_350_candidate` comparison in
`scripts/laguna_grouped_down_category_bench.py`. Its scaled F16 route executes
different arithmetic from range-direct; silently substituting the default would
invalidate that comparison. **Clearing action:** if retained, name the diagnostic
owner and close the route-retirement request as intentional reference code. If
retired, remove that comparison arm and its executable scaled-mode plumbing,
migrate category/publication tests without rewriting historical artifacts, and
preserve general scaled-cast primitives needed outside the normalized boundary.

**Validation dependency:** a local Laguna checkpoint. Run the R0273 gfx1151
public command and confirm range-direct selection, output parity and teardown.
Existing constructor tests verify missing/unsafe metadata rejection and owned
weight cleanup but do not replace a real public request.

### R0001 — PARO verifier graph regression

Ordinary PARO MTP requests use eager verification; explicit graph plans remain
executable. The recorded gfx1151 graph comparison and confirmation were slower
than eager, so this preference has an observed cause rather than missing evidence.
See [protocol and results](../benchmarks/results/2026-10-24-gfx1151-paro-graphs/NOTES.md).

**Dependency:** a graph implementation that removes that observed regression.
**Clearing action:** use the linked reproduction block to run
`reproduce_graph_candidate.py` with three repetitions and two confirmation
repetitions, then `validate_graph.py` and `tests/test_live_paro_mtp_graph.py`.
Require faster paired execution without output/acceptance drift and successful
public state/recovery checks before removing the eager preference. Preserve the
eager comparison implementation; add no environment flag or MTP admission gate.

### R0165 — Public physical GDN wave-reduction validation

The environment opt-out is removed; physical exact-row scope, FP16-state
selection and geometry fallbacks remain. Routing and direct GPU parent-parity
checks pass. A gfx1151 public MTP request completed but did not call this selector.

**Dependency:** a public request on the gfx1100 physical exact-row route.
**Clearing action:** run `hipengine serve` with the dense GGUF MTP profile and
an eligible physical target width, trace the wave-reduced state-row launch,
and compare output/acceptance plus teardown against the registered strict parent.
The existing numerical gate is
`tests/test_gpu_qwen35_gguf_gdn_prefill_correctness.py::test_gdn_prefill_segments_state_rows_wave_reduce_is_parent_bit_exact`.
Do not substitute successful c1 or unrelated-backend generation for engagement.

### R0166 — Public physical Q4 rowtile validation

The row48 and two-wave opt-outs are removed. Scoped dispatch and production-shape
GPU parity pass; backend geometry and registered strict parents remain.

**Dependency:** gfx1100 public physical target execution reaching R6 and R36.
**Clearing action:** run `hipengine serve` with the physical dense-GGUF MTP
profile, record the R6 two-wave rowtile and R36 row48 fused gate/up+SiLU launches,
and compare outputs, acceptance and teardown against their registered parents. Use
`tests/test_gpu_gguf_q4_k_t16_dense.py::test_q4_t16_dense_dual_wmma_silu_matches_unfused_chain_at_production_shape`
for the independent numerical gate; it does not replace public engagement.

### R0168 — Public grouped Q4 active-slot validation

The grouped-R8 environment/cache opt-out is removed; backend policy still limits
selection to the admitted physical rows and live-slot counts. Grouped-R6 remains
registered and selected for scope misses.

**Dependency:** gfx1100 public physical execution with five and six active target
slots. **Clearing action:** run `hipengine serve` with the physical MTP profile,
trace grouped-R8 at R24 for C5/C6 and grouped-R6 on the corresponding C7/C8 scope
misses, and compare outputs/acceptance/teardown with the strict parent. Preserve
`tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_q4_t16_dense_rowtile16_w2_grouped_rows8_matches_grouped_rows6_bits`
as the separate numerical gate.

### R0169 — Public exact C7 target-row validation

The dedicated C7 environment override is removed. Backend policy still selects
R28 only inside the existing production physical-row scope; padded composition,
C8 policy and strict profile behavior remain available.

**Dependency:** gfx1100 public C7/K3 physical MTP execution.
**Clearing action:** run `hipengine serve` with seven live requests and the
physical MTP profile, confirm R28 target work rather than padded R30, and compare
full state/output/acceptance and teardown against the padded composition.
Mocked prefix/tail routing tests do not replace this public gate.

### R0170 — Public source-Q5 MMQ validation

The source-Q5 environment opt-out is removed; raw-MMQ capability, request-count
and shape admission remain. The verifier numerical tool retains a scoped parent
control and records it separately from environment provenance.

**Dependency:** gfx1100 public C8 physical MTP execution reaching the R32 source-Q5
owner. **Clearing action:** run `hipengine serve` with that configuration, trace
source-Q5 selection at R32 and raw-MMQ fallback on R24/shape misses, then compare
output/acceptance and teardown with the retained parent. Use
`tests/test_gpu_gguf_q5_k_mmq_wmma_prefill.py::test_c8_q5_i64_j32_source_mmq_passes_production_outer_floor`
as the separate numerical gate, not a substitute for public engagement.

### R0171 — Public raw-Q5 sidecar and route validation

The raw-Q5 environment switch is removed from both loading and runtime.
Capability-owned planning and runtime request-count/sidecar/workspace fallbacks
remain. The verifier diagnostic controls the parent route in its own scope.

**Dependency:** gfx1100 public C8 physical MTP execution with raw-Q5 sidecars.
**Clearing action:** run `hipengine serve` at eight active requests, confirm
sidecar ownership and MMQ launches at admitted R24/R32 shapes, then compare
output/acceptance and allocation cleanup with grouped T16. Include fewer-request
and missing-workspace cases to verify fallback, without weakening admission.

### R0172 — Public grouped Q5 C8 validation

The grouped-Q5 R8 environment opt-out is removed; physical row admission and
registered grouped-R6/strict-tail parents remain.

**Dependency:** gfx1100 public C8/R32 physical target execution.
**Clearing action:** run `hipengine serve` with eight live MTP requests, trace
grouped-Q5 R8 launches, and compare output/acceptance and teardown with the
registered padded-prefix/strict-tail parent composition. The direct numerical
gate is `tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_q5_t16_grouped_rows8_matches_repeated_rows8_bits`;
it does not establish public engagement.

### R0174 — Public exact C8 target-row validation

The C8-specific environment opt-out is removed from backend policy. Production
profile and global exact-target-row scope still control admission; padded R36
and the registered row48 parent remain available.

**Dependency:** public gfx1100 C8/K3 physical target execution.
**Clearing action:** run `hipengine serve` with eight live MTP requests, confirm
R32 target packets and the selected fused row32 owner, and compare output,
acceptance and cleanup with global exact-target rows disabled. Include a
non-C8 request count to confirm unchanged padding outside this policy.

### R0176 / R0394 — Public direct resident verifier validation

The direct-resident verifier environment opt-out is removed. Backend policy
owns the original multi-session route, whose singleton guard remains after
the recorded gfx1100 failure. The independent captured/deferred initial-state
route now binds a one-slot scratch view for a singleton: unsegmented Conv/GDN
leaves consume pointer bases directly and cannot select a nonzero slab slot
from `layout.state_indices`. Multi-session segmented leaves keep indexed slabs;
incompatible owners keep packed imports.

**Dependency:** public gfx1100 physical target execution. The borrowed-slot
root error is reproduced and repaired on gfx1151 against the copied-state
oracle, including read-only resident state. Public gfx1151 MTP validates
multi-session indexing, repeatability and drain. That does not establish the
cause of the historical gfx1100 slot-zero failure or clear its older guard.

**Clearing action:** trace layout indices, bound Conv/GDN pointers and captured
rows through public gfx1100 MTP. Compare direct and packed-import output and
accepted-state ownership across slot zero, nonzero borrowed singletons,
multi-session groups, cancellation and teardown. Reopen the older singleton
route only after reproducing its failure and passing the width-one explicit-K3
category suite plus C1-C2 validation. Keep the working captured/deferred
one-slot mapping enabled while that separate hardware follow-up runs.

### R0177 — Public planar-Q6 chunk-policy validation

The LM-head chunk environment override is removed. Backend policy and primitive
capacity own chunk size; `_small_b_rowtile_chunks` and the 2–8-row kernels remain.
Existing promotion evidence is in
`benchmarks/results/2026-09-02-w7900-q4km-k3-c5c8-q6-root-f32-rows8-retained.json`.

**Dependency:** defaults-only gfx1100 and gfx1151 direct/server refreshes.
**Clearing action:** exercise `LLM.generate()` and `hipengine serve` on both
backends, trace Q6 root row chunks at eight requests and a non-divisible packed
width, and compare logits/output and cleanup with the registered R4 composition.

### R0178 — Public grouped Q6 mixed-row validation

The grouped-Q6 environment opt-out and its unreachable repeated-launch dispatch
branch are removed. Physical mixed-row scope and shape policy remain; repeated
row primitives and out-of-scope R6 composition are unchanged.

**Dependency:** public gfx1100 physical MTP mixed-row execution.
**Clearing action:** run `hipengine serve` across the admitted mixed widths,
trace grouped prefix/tail launches, and compare outputs and acceptance with the
registered repeated primitives. Include a shape outside the mixed policy and
verify cancellation and cleanup. Direct grouped-versus-repeated GPU bit parity
is separate from this public-route requirement.

### R0179 — Public grouped Q5 R6 validation

The grouped-Q5 R6 environment override is removed. Physical-row and exact-wide
scope remain; grouped R8 for C8, strict tails and repeated primitives are unchanged.

**Dependency:** public gfx1100 physical target execution reaching grouped R6.
**Clearing action:** run `hipengine serve` across grouped R6 and exact-prefix/tail
widths, confirm selected launches, and compare output and acceptance with the
registered repeated R6 primitive. Include ordinary decode outside physical scope
and cancellation/cleanup. Direct GPU bit parity does not establish public routing.

### R0180 — Public grouped Q4 R6 pair validation

The grouped-pair environment override and its cached read are removed.
Backend defaults, shape/row checks and registered-kernel checks remain.

**Dependency:** public gfx1100 physical target execution reaching grouped pairs.
**Clearing action:** run `hipengine serve` across admitted pair widths, trace
both grouped output launches, and compare output/acceptance with the registered
repeated-R6 composition. Include a shape outside pair policy and cancellation
cleanup. Direct grouped-kernel bit parity is not public pair-route evidence.

### R0181 — Public grouped Q4 R6 single-owner validation

The single-owner grouped-Q4 R6 environment override and cache are removed.
Physical scope and native rowtile selection remain; registered repeated-R6
primitives are unchanged.

**Dependency:** public gfx1100 physical target execution reaching this wrapper.
**Clearing action:** run `hipengine serve` at admitted grouped widths, confirm
single-owner grouped launches, and compare output/acceptance with repeated R6.
Include non-grouped rows, a shape outside policy, and cancellation/cleanup.
Direct grouped-kernel bit parity does not establish public wrapper engagement.

### R0182 — Public fused packed-state transfer validation

The packed-state environment override and unused environment-policy helper are
removed. Existing plain backend capabilities own both transfer selectors;
strict per-layer and single-slot fallback copies remain unchanged.

**Dependency:** public gfx1100 physical MTP execution using packed state imports.
**Clearing action:** run `hipengine serve`, trace the fused pointer-table gather
where direct resident indexing is inapplicable, and compare imported Conv/GDN
state with the per-layer parent. Include single-slot fallback, rejection,
cancellation and resource cleanup. CPU ownership tests do not prove public GPU
state-transfer engagement.

### R0211/R0212 — Scheduling policy and generic-default decisions

`EngineLoopConfig` and its environment resolver default to `protect_decode`.
Qwen hybrid GGUF Q4_K_M factories select `fair` with chunk 256 on both HIP
backends: gfx1100 uses burst one and gfx1151 burst two. Gemma4 selects `fair`;
Qwen4Exp selects `token_budget` with a decode-active prefill bound. Explicit
`HIPENGINE_PREFILL_DECODE_POLICY` values override generator defaults.
`protect_ttft` is also used by the shared submit/poll construction path. These
are active scheduling behaviors, not dead compatibility names.

**Decision needed:** keep policy selection as a supported production control,
or standardize production on `token_budget`; separately, choose whether
unconfigured generators should inherit `fair` instead of `protect_decode`.
The old entries proposed both generic `fair` and eventual `token_budget`, so
neither is an unambiguous cleanup replacement. Different backend burst defaults
must not be erased as a side effect. No default or admission behavior changes
in this disposition; missing benchmark rows do not restrict any policy.

**Clearing action:** if policy selection is retained, close R0211 as supported
configuration. If standardized, migrate every generator default, submit/poll
constructor and CLI/env caller to explicit round budgets before removing the
old policy branches. For R0212, either keep the scoped defaults as supported
registry configuration or change the generic default and remove only redundant
package policy declarations. Update configuration docs and run
`tests/test_unit_llm_generate.py`, `tests/test_unit_concurrency2_token_budget.py`
and `tests/test_integration_generation_engine_loop_burst.py`; exercise public
mixed-prefill/decode, staggered admission, cancellation and cleanup on both
backends and confirm the selected policy. TTFT/ITL comparisons evaluate the
chosen behavior; they are not permission to run it. Historical fair-promotion
evidence stays in the sweep inventory and original worklogs.

### R0213 — Mirrored INT8 fallback migration and validation

The serial-only description is obsolete. Native row-batched direct INT8
attention is registered and consumed by packed decode; the gfx1100 physical-C4
promotion is recorded in
`worklog/entries/20260911T221928.116483Z-lhl-ikv-c2-completion-audit-e5b0e8.md`.
Packed direct prefill and decode enforce their session-declared width bounds
as distinct work classes. Shared-oracle prefill and uniform-INT8 graph support
also exist. None of those implementations is blocked on the old IKV-C2 work.
The earlier context-capacity rows are historical, not current admission limits.

BF16 mirrors still serve short INT8 layouts when no mirror-free capability is
selected. `_qwen35_gguf_kv_chunk_layout` omits mirrors for a selected mirror-free
contract independently of the short-context threshold; `bf16_mirror` and
`int8_direct` remain distinct attention sources. Deleting mirror allocation,
copy or decode handling would break that supported storage route. Serial
single-row kernels remain numerical and capability fallbacks.

**Dependency:** finish the remaining replacement validation and migrate actual
mirror consumers, not implement row-batched attention again. The historical
campaign's c2/c4 memory-positive prefill and later lifecycle/quality/economics
items are separate from its completed decode milestone; its closed front-matter
and old active body are not a current runtime contract.
**Clearing action:** exercise public direct and mirrored requests on both HIP
backends with short, long, shifted and ragged allocations, cancellation, refill,
resource rejection and cleanup; record selected storage/attention sources and
actual persistent/transient ownership. Use the current planner to price these
cases rather than reusing old capacity ceilings. Migrate or explicitly retire
remaining mirrored callers before removing its allocation/copy/dispatch code.
Run `tests/test_unit_qwen38_int8_kv_capability.py`,
`tests/test_unit_qwen35_gguf_int8_kv_policy.py`,
`tests/test_unit_gguf_packed_verify_layout.py` and
`tests/test_unit_gguf_int8_mtp.py` alongside that public validation. Missing
measurements do not disable the implemented direct path, and this disposition
makes no new GPU, capacity or performance claim.

### R0214/R0215 / R0378 — Cumulative-semantic rollback-control migration

`HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL`,
`HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL` and
`HIPENGINE_GGUF_DENSE_WMMA_BULK` default on. Their off values still select
real control arms in `scripts/qwen35_08b_cumulative_semantic.py`: strict and
X8 rollback use separate Q8T16 projections; X6 rollback uses projection plus
standalone BF16 add; strict/pre-X2 use scalar dense-BF16 prefill. The threshold
sweep also uses the dense bulk opt-out. That opt-out suppresses both ordinary
and residual-fused dense WMMA dispatch. Removing the environment reads without
migrating those arms would silently compare the candidate with itself.
Registered singleton projections and addition also handle shape/backend/registry
misses and must remain regardless of selector retirement.

**Dependency:** move these diagnostic controls out of production environment
selection and validate the actual Qwen3.5-0.8B Q4_K_M and Q8_0 callers. Neither
checkpoint is present in the local `/models/gguf` catalogue used for this sweep;
other available Qwen models do not establish engagement of the narrow
rows512/K1024/N16+N16 pair.
**Clearing action:** replace the harness's bulk/X6/X8 environment controls with
scoped diagnostic overrides that restore on success and exception, migrate the
threshold children, remove the production readers and Q8 pair-cache environment
dimension, and update ENVS. Run
`tests/test_unit_qwen35_08b_cumulative_semantic.py`,
`tests/test_unit_qwen35_08b_threshold_environment.py` and
`tests/test_unit_gguf_linear_dispatch.py`; with those checkpoints available,
exercise public p512 generation and eager/graph decode, trace pair/residual
selection, compare against the registered scalar/singleton/add controls, and
verify cleanup. Check bulk dispatch at 511/512/513 and unrelated lengths. Keep
backend/shape policy and primitive fallbacks. These are concrete harness and
input dependencies, not a release-window or missing-qualification restriction;
the fused defaults remain enabled. R0378's standalone release-age note is
consolidated here rather than creating a separate retirement decision.

### R0217 — PM4 diagnostic controls and transport rollback

The redundant `pm4_stateful_local` benchmark alias is removed; canonical `pm4`
already selects stateful registers and local-cache dependencies with timestamps
off. `pm4_stateful` still selects global dependencies, and `pm4_timestamps`
enables timestamps, so those labels are not duplicate encoders. The production
transport factory selects the canonical encoder without the retired experiment
environment controls. Backend policy selects automatic transport by geometry,
file type, physical width and replay horizon; HIP graph remains the portable
fallback. Explicit AQL and PM4 are supported diagnostic choices.

**Decision needed:** retain the distinct global-dependency/timestamp modes and
`HIPENGINE_SUBMISSION_TRANSPORT` rollback as supported diagnostics, or retire
their comparison interfaces. **Dependency:** timestamp/quarantine retirement
also needs a disposition of the recorded ROCm/ROCm#6529 lifecycle investigation.
No gfx1100 runtime is available in this local sweep to exercise that fault or
public PM4 selection; gfx1151 HIP execution does not substitute for it.
**Clearing action:** if retained, name the diagnostic owner and close the
remaining cleanup; if retired, migrate the lifecycle/graph benchmark callers
and remove only unused encoder options. Run
`tests/test_unit_pm4_graph_bench.py`, `tests/test_unit_pm4_transport.py` and
`tests/test_unit_pm4_lifecycle_repro.py`, then on gfx1100 exercise public
automatic and explicitly selected transport, rebuild/cancel/shutdown and
resource cleanup. Preserve native HIP graph and distinct AQL profiling, and
keep full provenance serialization out of replay. Original promotion and
attribution results remain in the sweep inventory and worklogs; this cleanup
changes no replay arithmetic, transport default or performance claim.

### R0219 — Source-F16 packed-prefill integration and rollback decision

The shared session type/context, private runner method/handle/library and
constructor option now use quant-neutral T16-F16 names. All known Q6 alias
callers were migrated and the aliases removed. The bounded scratch planes keep
their existing `q6_f16_*` plan keys; R0220 covers their ownership. Exact T16 and
registry misses remain required fallbacks. The constructor opt-out is now
`use_t16_f16_rocblas_prefill=False`; automatic selection is unchanged.

**Dependency:** ordinary resident `prefill()` opens the source-F16 context, but
packed `prefill_batch_native()` does not. Default public serving uses packed
prefill, so private-context coverage is not evidence of source-F16 execution
through that surface. **Clearing action:** integrate a request-row-scoped owner
with packed scratch lifetimes and ragged requests, then exercise
`LLM.generate_detailed()` with non-anchor, 512/1K/4K and mixed-length requests on
each applicable backend. Confirm selected registry variants and exact
request/slot ownership, cleanup, and the changed-arithmetic execution-profile
gates. Keep the ordered pair-only operand policy and Q4 row ceiling, which
preserve known fallback cases. **Decision needed:** retain the quant-neutral
constructor opt-out as a supported diagnostic or remove it once its callers
are migrated; a release window alone is not a dependency. No policy, arithmetic
or scratch layout changed in the naming cleanup.

### R0220 — Dense gfx1151 source-F16 scratch reuse

Removed the redundant empty Q4/Q6 source-F16 declarations: the resolver reads
each capability from the requested backend with `{}` as its default, so
omission preserves exact T16 selection and cannot inherit gfx1100 policy. Q5's
bounded source-F16 selection and the exact Q5 WMMA fallback remain unchanged.
The fallback is a supported numerical route, not release-window debt.

**Dependency:** dense gfx1151 has no
`GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES` entry, so its activation, weight
tile and output planes are independently allocated. The gfx1151 MoE owner-slot
policy and gfx1100 dense arena do not establish safe reuse for this geometry.
The 2026-08-16 memory-parity investigation recorded final-logit changes under
both dense owner-slot and single-arena aliases, plus trajectory divergence in
other reuse attempts; dedicated storage preserves that observed boundary.
See `worklog/entries/20260816T152449.196381Z-lhl-qwen38-gfx1151-memory-parity-d82d61.md`.
Q5 already casts dead recurrent-output input in place; no persistent F16
weight copy is needed. **Clearing action:** reproduce and isolate those alias
failures while establishing a dense gfx1151 lifetime plan covering all active
producers, then alias stage-disjoint planes and
remove only proven-redundant allocation owners. Exercise non-anchor and
512/1K/4K lengths, pairwise live-range overlap, full-attention/recurrent/FFN
handoffs and teardown. Run
`tests/test_unit_qwen35_gguf_prefill_scratch_liveness.py` and
`tests/test_unit_qwen35_gguf_prefill_plan.py`, followed by public selected-route
and cleanup validation with R0219's packed-prefill integration. Preserve sole
T16 residency and exact fallbacks. No workspace saving is claimed by this
policy-declaration cleanup.

### R0223 — Q6 prefill selector normalization

Role-specific Q6 layout is supported policy, not rollback debt: recurrent QKV
keeps its sole standard-T16 resident after an observed planar-c1 regression;
other declared roles use sole planar storage. The excluded planar residual
rowtile also has an observed regression, so gfx1151 uses projection plus BF16
add for that case. Keep these policies, peer primitives and strict fallbacks.
No duplicate payload or environment selector needs removal.

**Decision needed:** keep separate standard/planar package wrappers or express
their ordered row bands through a shared policy representation. The original
two rows>=512 selectors have expanded: standard shared4 starts at 96 with
earlier and overlapping sibling bands; planar shared4 starts at 256 with
shape-specific, exact-row and bounded-interval overrides. A single min-row
field cannot preserve that precedence. **Clearing action:** if normalization
is chosen, map each current branch to an ordered quant/shape/row rule in the
backend package, then compare selected function identities before/after over
every interval boundary, exact-row override, unrelated row and shape miss.
Keep engine/model dispatch free of backend/quant branches. Run
`tests/test_unit_gfx1151_backend.py`,
`tests/test_unit_loading_qwen35_gguf_policy.py` and
`tests/test_unit_gguf_linear_dispatch.py`, then confirm public selected owners
and cleanup on gfx1151. Do not couple this structural cleanup to retuning the
QKV layout or re-admitting the rejected residual composite. Historical tuning
results remain in the sweep inventory and worklogs; no routing change or new
performance claim accompanies this disposition.

### R0231 — Maple CUDA resident fixed-batch owner

CUDA c1 generation uses `MapleCudaRunner.prefill_native()` and CUDA decode.
Inherited serial `prefill()` is an independent correctness oracle, not an
unselected production route to delete. The original native-prefill result is
`benchmarks/results/2026-08-08-cuda-sm120a-maple-native-prefill-retained.json`.

**Dependency:** a CUDA resident-batch owner is not implemented. The CUDA
package already provides per-request batched QK/KV-write and attention leaves,
but `MapleBatchRunner` constructs HIP buffers and libraries, and the CUDA
factory disables that resident owner. Do not flip the factory boolean or
route CUDA through HIP builders to bypass the missing implementation.
**Clearing action:** implement a CUDA owner with independent per-request
span/KV arenas and CUDA libraries, wire existing batched leaves and row-reuse
head execution, and enable it through the public factory. Validate sparse
c2/c4/c8 admission, cancellation/reclaim, staggered refill, singleton-in-c8,
full natural/heldout trajectories and cleanup. On CUDA hardware run
`HIPENGINE_RUN_CUDA_MAPLE=1 python -m pytest tests/test_gpu_cuda_sm120a_maple.py -q`,
then `python scripts/maple_public_batch_bench.py --backend cuda_sm120a --out /tmp/maple-cuda-public-batch.json`.
Extend those gates to cover the new owner rather than counting serialized c1
calls as batching. This is implementation work, not a missing-evidence gate;
working c1 execution and its oracle remain unchanged.

### R0239 — Host-token embedding placement control

`HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` is a live placement override, not an
unused off switch. `scripts/qwen35_08b_cumulative_semantic.py`,
`scripts/qwen35_08b_threshold_sweep.py` and `scripts/qwen35_08b_three_way.py`
force device placement; `scripts/qwen35_08b_graph_rerank.py` reports the value.
The shared runner now maps supported Q4 tables automatically, including on
gfx1151 through an anonymous copy. The old statement that every shared runner
keeps device ownership is obsolete. External ownership and unsupported types
still preserve their device fallback. Private sessions retain their own
width/shared-runner checks and Q8 host-copy route.

Loader-time exclusion is implemented: host placement passes
`deferred_device_slots=("root.token_embedding",)` to materialization, rather
than allocating then freeing the table. Device-pointer consumers reuse a
mapped table or call `ensure_device_token_embedding()` for transactional
on-demand materialization. An allocation refusal preserves the host route;
external resident maps are not freed or relocated. The duplicate c1 embedding
row's allocate-then-free task and blanket shared-device requirement are stale,
not another implementation blocker.

**Decision needed:** retain this host-memory/VRAM placement override as a
supported resource policy, or replace it with an explicit public placement
option and migrate diagnostic callers before retiring the environment read.
**Clearing action:** if retained, document it as supported policy and close
this cleanup; if retired, migrate the named callers without changing their
device control arms, then run
`tests/test_gpu_qwen35_gguf_host_token_embedding.py`,
`tests/test_unit_qwen35_gguf_serving_width_sidecar.py` and
`tests/test_unit_qwen35_08b_cumulative_semantic.py`. Compare real public auto
and device placement, eager/graph outputs, selected storage and teardown on
each applicable backend. Preserve gfx1151's anonymous-copy requirement:
direct file-backed registration has an observed trajectory failure. Dedicated
dense-prefill fields and context-dependent chunk policy are supported safety
policy, not duplicate allocations to remove here; R0220 owns the unresolved
alias failures and R0219 the packed source-F16 integration. No placement,
chunk-size or allocation behavior changes in this disposition.

### R0258 — Laguna grouped-IQ diagnostic chain and policy normalization

The bounded backend maps select working registered kernels, not evidence
allowlists. gfx1100 uses H6T for IQ3 selected down, H5J for IQ4 down, H6L for
IQ2 pair16 gate/up and H6C for the layer-47 IQ3 role. These names identify the
variants in `hipengine/kernels/hip_gfx1100/__init__.py`. Generic grouped and
route-major chains handle shape, registration and backend misses. gfx1151
keeps its independent MMQ policy. The old H6I/H6P source-default descriptions
are historical; no default change is needed.

**Decision needed:** retain the older exact diagnostic variants and their
launch-interface mappings, or retire them with the comparison callers. Tests
for H5Q/H5Z/H6D/H6F/H6I/H6P/H6Q/H6R still exercise that chain. Folding the
maps into a source selector must preserve the distinction between active-
expert and generic grouped interfaces; deleting interface entries alone can
select a registered function with the wrong operands. The role and quant
maps do not allocate another weight payload or workspace.
**Clearing action:** inventory variant-specific test/profile callers, decide
which diagnostics remain supported, and remove only unused registrations,
bodies, mappings and fixtures together. Alternatively, normalize stable maps
without changing selected function/interface identities over all shape,
role, missing-registration and backend cases. Run
`tests/test_unit_laguna_h6t_source_default.py`,
`tests/test_unit_laguna_h6l_source_default.py`,
`tests/test_unit_laguna_h6c_source_default.py` and affected variant tests, then
`python scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1100 --max-tokens 8 --output /tmp/laguna-grouped-public.json`
with a bulk prompt that engages grouped prefill. The local checkpoint
dependency is recorded at R0249/R0250. Preserve Qwen GGUF's independent
consumers of shared IQ primitives, exact fallbacks and current defaults.
Original tuning results remain in benchmark artifacts and worklogs; this
reconciliation changes no runtime selection or arithmetic.

### R0259 — Laguna exact SWA and preappend policy normalization

Exact gfx1100 SWA policy keeps its qrow4/wave32 crossover, source-qualified
role and safe append-before-attention ordering. Dense-initial source policy
has advanced beyond H6A to H6Z global score/weight replay plus H6W SWA score
replay; H6A and H6W comparison maps remain. The old H5N and H5U runtime-policy
extensions are absent; their standalone numerical leaves remain registered.
Online SWA has independent gfx1151 consumers and an observed gfx1100 quality
failure, so do not delete or promote it as a side effect of structural cleanup.

**Decision needed:** keep bounded backend role maps as the supported policy
representation, or normalize them into a unified selector. They select cache
ordering as well as a function; absence, empty maps, independent global/SWA
overrides and explicit variants carry different fallback semantics.
**Clearing action:** if normalization is chosen, compare exact selected
function, cache ownership and append ordering before/after over starts
0/128/256 and unrelated positions, no-wrap/wrap, complete/partial chunks,
verifier/continuation, shape/backend and registration misses. Run
`tests/test_unit_laguna_h5r_runtime_policy.py`,
`tests/test_unit_laguna_h6a_runtime_policy.py`,
`tests/test_unit_laguna_h6a_source_default.py`,
`tests/test_unit_laguna_h6w_source_default.py` and
`tests/test_unit_laguna_h6z_source_default.py`, then public bulk-prefill
selection, output/state and teardown on the applicable backend using
R0249/R0250's command and checkpoint dependency. Retain span-aware primitive
fallbacks and the separately owned online arithmetic. A release window alone
is not a reason to remove safe ordering or working policy. This disposition
changes no runtime selection, storage or arithmetic.

### R0264 — Laguna full-score long-attention diagnostic retirement

The bounded 4K-key-block owner is the gfx1151 long-attention default. The
capacity-sized full-score owner is also reachable: the profile harness
compares `prefill_block_attention_hipblaslt=False/True`, and its dedicated
ceiling harness sweeps matrix-library algorithms. Both owners use
`LagunaAttentionHipblasLt`; short attention and rolling SWA also use that
class or its subclass. The algorithm override is not a full-score-only API.

**Dependency:** separate full-score diagnostics from long-attention
execution policy before removal. `prefill_long_attention_hipblaslt` and
`LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT` currently enable the bounded route
too; deleting them as though they only enabled the old owner would disable
production long attention. **Decision needed:** retain full-score as a
supported diagnostic, or retire its comparison/ceiling callers.
**Clearing action:** if retired, migrate the profile control to bounded versus
generic span-aware execution, preserve bounded width/context policy, and
remove only full-score lazy allocation/accounting/close plus unused controls.
Audit shared algorithm callers in both long-context and SWA ceiling harnesses
before narrowing that API. Run `tests/test_unit_laguna_long_context_profile.py`
and `tests/test_gpu_laguna_kv_attention.py`; on gfx1151 compare public long
prefill output/state and cleanup with prompts both inside and beyond the
block boundary, including non-aligned tails, continuation and generic misses.
Use R0249/R0250's public command/checkpoint dependency. Keep rolling SWA,
short attention, generic `KVLiveSpans` fallback and bounded scratch. No
capability, arithmetic or allocation behavior changes in this disposition.

### R0265/R0266/R0267 — Laguna attention geometry and layout controls

Rolling SWA, global query width and dense-contiguous cache access remain
independent policy axes. Rolling SWA uses its own bounded gather/attention
owner and a real profile comparison. `prefill_global_attention_chunk_size`
selects supported widths within matrix capacity; wider complete global
chunks do not absorb SWA or partial-tail handling. Dense-contiguous cache
selection applies only to eligible full-attention views; span-aware execution
remains necessary for non-dense semantics. Keep the implemented routes and
backend defaults; the old LC-stage/release-window notes are not blockers.

**Decision needed:** retain explicit constructor/profile controls as supported
diagnostics and geometry policy, or move comparisons into scoped test owners
and retire redundant mutation interfaces. The profile harness still calls
`set_prefill_swa_attention_hipblaslt()` and
`set_prefill_dense_contiguous_cache()` and supplies `--attention-rows`.
**Clearing action:** if retiring controls, migrate those arms first, preserving
actual generic-versus-selected execution and restoration after failures.
Keep backend width defaults, matrix-capacity validation, bounded allocation
and full `KVLiveSpans` fallbacks. Run
`tests/test_unit_laguna_gguf_runner.py`,
`tests/test_unit_laguna_long_context_profile.py` and
`tests/test_gpu_laguna_kv_attention.py`; then use R0249/R0250's public gate
with dense and rolling/wrapped prompts, non-aligned tails, continuation,
verifier and shape misses on each applicable backend. Confirm selected width,
owner/layout, output/state and cleanup. Global query geometry is a resource
choice, not an evidence allowlist; do not remove a supported width merely
because a benchmark did not cover it. No route, arithmetic or resource policy
changes in this disposition.

### R0269 — Laguna MoE scheduling diagnostic controls

MoE concurrency is already the gfx1151 default when the process has at least
two hardware queues. Automatic one-queue execution stays sequential; c1
concurrency also requires the shared stream/event owner. After-router and
low-priority scheduling are backend defaults, not default-off candidates.
Keep these implementations and general HIP priority helpers. The rejected
before-down selector is already absent. Historical promotion is recorded in
`benchmarks/results/2026-07-26-gfx1151-laguna-moe-branch-concurrency-production.json`
and the matching shared-after-router/priority artifacts.

**Decision needed:** retain constructor/profile scheduling overrides as
supported diagnostics, or move comparisons to scoped harness controls and
retire redundant mutation interfaces. `scripts/laguna_long_context_profile.py`
still exposes concurrency, after-router and low-priority choices.
**Clearing action:** if retired, migrate those callers while preserving
sequential and priority-0 controls; do not remove the live stream/events or
queue-capacity fallback. Run `tests/test_unit_laguna_gguf_runner.py`,
`tests/test_unit_laguna_long_context_profile.py` and
`tests/test_live_laguna_moe_gpu.py`; compare public single-/multi-row eager
and graph execution, queue limits, cancellation and shutdown with the local
checkpoint dependency at R0249/R0250. Confirm dependencies and resource
cleanup, not just token equality. No scheduling/default change accompanies
this reconciliation.

### R0270 — Laguna selected-MMQ diagnostic modes

The unused one-plane `mmq64x32_d4_f32` gate/up selector is retired following
its recorded quality failure. The same name still selects a live **down**
projection in the category harness and GPU composition tests; that independent
route and its shared kernels stay. gfx1151 defaults to the D8
wave-column/direct/double-buffer/raw-prefetch gate/up family and D4 Q6/Q4
down family. Direct/exact and quant/shape fallbacks are unchanged.

**Decision needed:** retain older D8, D4x2/D4x3 and down modes as supported
diagnostics, or migrate their comparison callers and remove redundant mode
names. `scripts/laguna_grouped_down_category_bench.py`,
`scripts/laguna_prefill_production_publication.py` and
`tests/test_live_laguna_moe_gpu.py` still use these interfaces. The old
one-release rollback note is not a runtime precondition. R0281's grouped-down
row uses the same `set_selected_down_mode()` owner: direct,
`grouped_smallm*` and adaptive/fused modes still have category callers.
Its older grouped-only default description does not replace the current MMQ
package default; include these mode names in the same retention/migration
decision rather than tracking a second selector debt.
**Clearing action:** migrate those callers before removing a mode; preserve
actual arithmetic comparisons, strict fallbacks and down primitives shared
with production. Run `tests/test_unit_laguna_selected_mmq_modes.py`,
`tests/test_unit_laguna_grouped_down_category_bench.py` and
`tests/test_live_laguna_moe_gpu.py`, then R0249/R0250's public model gate on
the applicable backend. Include non-aligned small/large rows, quant/shape
misses, continuation and cleanup. No arithmetic or default changes accompany
the gate/up-selector removal.

### R0277 — GGUF capture-inclusive graph policy and diagnostic interfaces

The old priority note mixes superseded stack results with an unresolved timing
boundary. Later gfx1151 graph/eager rows in
`benchmarks/results/2026-07-15-gfx1151-gguf-decode-closure-profile.json`
are positive but exclude capture; they do not resolve the earlier
capture-inclusive regression. Keep the current graph defaults and
`HIPENGINE_GGUF_DECODE_GRAPH=0` eager override while the policy is decided.
The graph key, cursor and cumulative transition-budget checks are required
ownership, not removable legacy plumbing.

Automatic scalar floors are backend- and storage-aware: base floors are
24 on gfx1100 and 128 on gfx1151; gfx1100 uniform per-token-head INT8 uses
128. R0088 already retired the INT8 opt-in. Direct capture below an automatic
floor remains an explicit diagnostic, and packed capture has separate
capabilities. Do not restore the old blanket statement that INT8 stays eager.
The later graph replay row shares this remaining policy/interface decision;
the duplicate rollback/benches row has been removed. Keep packed BF16 graph
selection and reuse separate from scalar INT8 admission. The eager override
still serves production-load and sampling-overhead comparison callers.

**Decision needed:** keep the eager override and comparison interfaces as
supported controls, or retire them after a current same-host,
capture-inclusive public-route comparison. The bench defaults to graph
requests; MTP's default target-graph flag still has consumers in applicable
verification modes, while the resident preset selects bulk verification.
Neither MTP mode is dead solely because it differs from scalar AR.
**Clearing action:** run `scripts/gguf_decode_graph_g5.py --help`, then its
state-parity and capture-inclusive comparison with an explicit local model,
backend and compiler-version file. Compare actual `LLM.generate()` eager
and automatic routes at below-/above-floor and unrelated horizons, including
reset, cancellation and cleanup. Keep queue count and KV policy matched;
separate capture, first replay and steady replay. Inspect mode consumers in
`scripts/gguf_mtp_bench.py` before retiring its flags and migrate their
comparison tests. Run `tests/test_unit_qwen35_gguf_decode_graph.py`,
`tests/test_unit_qwen35_gguf_decode_graph_kv_floor.py`,
`tests/test_unit_qwen35_gguf_bench_metadata.py` and
`tests/test_unit_gguf_mtp_bench_metrics.py`. This unit changes no default,
replay floor or implementation and makes no current performance claim.

### R0278 — Offline DMS capture API ownership

`prefill(..., dms_capture=...)` serves two distinct consumers. The offline
`DMSCaptureWriter` synchronously exports normalized hidden rows and Q/K to
checksummed shards through `scripts/qwen38_dms_capture.py`. Serving creates
`_ExternalDMSDevicePrefillCollector` internally; its `capture_device_chunk`
method bypasses that host export. A caller cannot substitute an offline sink
while the serving collector owns capture. Training and capture manifests are
already recorded in `docs/reference/DMS.md`; waiting for initial training is
not the remaining blocker.

**Decision needed:** expose offline export under a model-plugin-owned capture
API, or retire that offline API with its CLI and training/label consumers.
Keep the serving device collector and its callback path. Do not remove the
shared keyword/helper merely because the offline sink synchronizes.
**Clearing action:** if migrating, move `scripts/qwen38_dms_capture.py` and
`tests/test_live_dms_capture_gguf_hip.py` together, preserving input-stage,
layer/shape validation, teacher logits, checksums and sink ownership. Run
`tests/test_unit_dms_capture.py`,
`tests/test_unit_qwen38_dms_capture_script.py` and the live capture test with
a local compatible model; verify an actual capture manifest and its shards.
Then exercise the device collector through public DMS generation using
`tests/test_live_dms_public_serving.py`, with an explicitly labeled synthetic
sidecar where trained artifacts are absent. Verify no new synchronous offline
export in serving and correct close/cancellation behavior. No runtime or API
change accompanies this disposition.

### R0279 — General compact-DMS undo beyond append-only cycles

The serial verifier already uses one host-owned append-tail undo image when
existing/new eviction candidates cannot expire over the complete horizon and
every head has capacity. Row checkpoints share the image; restore rewrites
only the rejected suffix, and finalization keeps initial undo until the enclosing
commit completes. Other cycles retain full request-owned extent snapshots,
including payload, scales, metadata and allocator state. This is implemented
rollback, not a persistent dense shadow or an unimplemented compact owner.
See `worklog/entries/20261021T053327.726182Z-lhl-dms-bounded-append-undo-0217fc.md`.

**Remaining work:** general eviction/compaction still copies owned extents to
host. Choose a bounded reversible mutation log or a resource-claimed device-local
journal; do not replace it with an unpriced prompt-proportional VRAM snapshot.
Preserve the bounded append path and exact full fallback until the successor
covers expiry, movement and changing extent ownership.
**Clearing action:** implement the chosen owner in
`hipengine/kvcache/dms.py`, `hipengine/kvcache/dms_device.py` and
`_DMSStoreJournal` in `hipengine/runtime/qwen35_gguf_mtp.py`. Restore K/V,
INT8 scales, positions, eviction/live counts, extent/ledger state and non-KV
resident state at the same accepted-prefix boundary. Run the append-journal,
append-tail, append-integration, cycle-transaction, journal-finalize and
verifier-journal unit files plus `tests/test_gpu_dms_external_runtime.py` and
`tests/test_gpu_dms_int8_device_payloads.py`. Exercise public MTP rejection,
partial/full commit, following cycles, copy-failure retry, cancellation and
shutdown. Use a trace to verify copy/allocation ownership, not just token IDs.
No journal algorithm or runtime selection changes in this reconciliation.

### R0280 — DMS device ranking and decode control ownership

One-pass exact-budget prefill already reserves score-independent capacity and
packs each completed layer in its capture trunk pass. It does not remove host
control: `finalize_layer()` synchronizes, copies that layer's logits to host,
builds an exact-budget mask and uploads it. Ranking evicts highest scores first,
with oldest-position tie-breaking and the protected-window/count rule in
`build_dms_exact_budget_eviction()`. Decode uses device projection/append/attention,
then synchronizes after the configured full-attention layers, reads their
decisions and finalizes host metadata. Layer/head geometry is model-derived,
not a fixed 16-layer/64-byte contract.

**Remaining work:** implement bounded deterministic device ranking and
live-count/position/eviction finalization with one owner for host observability,
compact extents, prefix reuse and speculative rollback. Preserve the one-pass
ordering and layer staging/release lifetimes; the current host route is the
working reference, not an admission failure. Device packing alone is not this
successor.
**Clearing action:** add CPU-oracle RED tests for exact masks including ties,
non-finite rejection, window edges, non-aligned lengths and per-head counts;
then register device primitives and integrate collector/decode owners. Run
`tests/test_unit_kvcache_dms.py`,
`tests/test_unit_dms_one_pass_prefill.py`,
`tests/test_unit_dms_layer_collector.py`,
`tests/test_unit_dms_layer_outer_lifetime.py`,
`tests/test_unit_dms_prefill_cache.py` and the R0279 journal bundle, plus
`tests/test_gpu_dms_external_runtime.py` and public DMS lifecycle/MTP requests.
Confirm masks, payload/scales/metadata, accepted-prefix rollback, continuation
and cleanup on each implemented composition. Profile named operations and
transfers with resources priced before retiring the host fallback. No control
algorithm, API or runtime default changes in this reconciliation.

### R0281-ATTN — Laguna prefill/decode variant interfaces

The five token4/wave32/qrow2 rollback rows describe live variant interfaces,
not five unpromoted routes. gfx1100 defaults SWA decode to token4 exact and
SWA prefill to context-qualified qrow4 exact. gfx1151 defaults global/SWA
prefill to qrow4-M128 online, with later cache/BLAS policies layered over those
variants. Older qrow2 online is still selectable, not the package default.
Wave32 and exact qrow2 remain registered fallback/comparison implementations;
baseline decode remains the generic fallback. Preserve these primitives and
their span/short/partial semantics.

**Decision needed:** keep `global_prefill_variant`, `swa_prefill_variant` and
`swa_decode_variant` as supported diagnostic controls, or migrate comparison
callers and retire redundant public positive selection. The grouped-down
category, SWA leaf, target-AR, DFlash and publication harnesses still use these
interfaces. Existing runtime-policy/GPU tests exercise explicit variants too.
**Clearing action:** migrate those callers before removing a mode, preserving
independent comparison arithmetic and backend defaults. Run
`tests/test_unit_laguna_gguf_runner.py`,
`tests/test_unit_laguna_grouped_down_category_bench.py`,
`tests/test_unit_laguna_target_ar_bench.py`,
`tests/test_unit_laguna_dflash_category_bench.py` and
`tests/test_gpu_laguna_kv_attention.py`. Complete R0249/R0250's public gate on
each applicable backend with wrapped SWA, dense/global and non-dense spans,
empty/short/partial rows, continuation, verifier rows and cleanup. Do not
collapse strict exact fallbacks into online arithmetic during interface cleanup.
No variant, allocation, arithmetic or default changes in this consolidation.

### R0281-PARO-WIDTH — PARO batch policy and diagnostic controls

Width-group execution is shared by the greedy compatibility generator, persistent
resident loop and `scripts/qwen35_batch_retained_bench.py` through
`hipengine/runtime/qwen35_paro_decode_groups.py`. The caller still owns width
planning, request ordering, deadlines, sampling and telemetry. The executor
preserves sparse slots and group order and retries only a native
`NotImplementedError` through the existing serial implementation.

**Remaining work:** the two environment controls are not synonyms.
`HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS` also selects per-layer linear-output,
attention and projection defaults; `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE`
controls native decode. Both still have benchmark/test callers. Choose supported
separate policy controls or migrate those callers to explicit diagnostic scopes
before retiring their production environment readers; a rename alone does not
merge their behavior. The sampled-batch diagnostic is independent and must not
be substituted for the public per-request sampler.

The native-width cost profile loader also checks model identity and recorded
measurement ranges. Those checks describe current code, not kernel capability;
missing or mismatched evidence is not a reason to disable implemented native
execution. Replace admission checks with backend/quant/KV/shape capability and
recorded concrete failures, retaining measured costs only for automatic choice
among working variants. Do not add new artifact allowlists or preserve the old
instruction to require identity checks permanently.

**Clearing action:** separate these policies, migrate actual callers, then run
`tests/test_unit_qwen35_paro_decode_groups.py`,
`tests/test_unit_qwen35_batch_partition_bench.py`,
`tests/test_unit_generation_qwen35_paro.py`,
`tests/test_unit_qwen35_resident_batch_layout.py` and batch-width loader tests.
Exercise public native and serial c1/c2/c4/c8, sparse slots, staggered admission,
cancellation and cleanup, with the chosen path reported. The available local
checkpoint exercised the shared executor through public serial fallback; it did
not exercise native-width selection. gfx1100 is a separate workstation check.
No kernel, arithmetic, default, flag or profile-admission change in the executor
extraction.

### R0281-D12 — Q5 constructor retirement public validation

Unused `use_q5_wave32x2_output` and `use_q5_wave32x2_query_gate`
constructor arguments are removed. Backend capability now supplies those two
choices directly through the existing resolver. The resolver still exposes
explicit role controls for primitive diagnostics; fixed-metadata constructor
and benchmark comparisons remain separate. Selected-variant attributes,
registered coefficient-publication siblings, pack8 and shape/backend fallbacks
are unchanged.

**Remaining validation:** use the local checkpoint dependency at R0249/R0250
to exercise `LLM.generate()` on gfx1100 and gfx1151. Confirm selected Q5
attention-output/query-gate variants, one-token and multi-row fallback behavior,
continuation and allocation cleanup. Retirement/binding, resolver, runner,
benchmark and H8 ownership tests pass; no new public model run or performance
claim accompanies this change. Do not revive constructor controls or gate
implemented execution because the checkpoint is absent.

### R0281-Q5META — Q5 metadata comparison interface

`laguna_target_ar_bench.py` still passes and reports the independent
`use_q5_fixed_meta_output`, `use_q5_fixed_meta_query_gate` and
`use_q5_shared_fixed_meta` controls. The first two restore coefficient-publication
siblings; the third restores the shared-expert pack8 pair. Query/gate controls
apply to the pair route after mixed-quad disable or miss, not to a successful
mixed quad. These controls are separate from R0281-D12's removed constructor
arguments.

**Decision needed:** keep these constructor/CLI comparisons as supported
diagnostics, or migrate them to a scoped diagnostic owner before removing
the public arguments. **Clearing action:** migrate both target-AR construction
sites and result fields with resolver/binding tests. Exercise the
query/gate comparison with the mixed quad disabled or unavailable and the
shared pair independently; preserve registered coefficient-publication and
pack8 routes, multi-row/shape/key fallbacks and backend defaults. Run
`tests/test_unit_laguna_q5_control_retirement.py`,
`tests/test_unit_laguna_gguf_runner.py` and
`tests/test_unit_laguna_target_ar_bench.py`, the corresponding primitive GPU
comparisons and R0249/R0250's public generation/lifecycle gate. No selector,
implementation or default changes in this reconciliation.

### R0281-MIXED — Mixed-projection comparison levels

`laguna_target_ar_bench.py` still passes and reports
`use_mixed_q5_q6_attention`, `use_mixed_q6_fixed_meta_attention` and
`use_mixed_local32_fixed_meta_attention`. The launcher first tries local32
when enabled, then the fixed-Q6 or generic mixed quad selected by the second
control. Disabling fixed-Q6 alone does not replace a successful local32
launch. Disabling mixed projections skips both quad candidates. Registry,
row and layout misses reach the registered pair/singleton chain; layer-47
Q6/Q8 selection is keyed by the actual quant tuple, not model identity.

**Decision needed:** keep the three comparison levels as a supported
constructor/CLI diagnostic, or migrate callers to a scoped projection-policy
owner and remove public controls. **Clearing action:** migrate both target-AR
construction sites and result fields together. Preserve local32 miss ordering,
fixed-Q6 versus generic comparison (disable local32 to exercise it), full
quad disable, layer-47 tuple handling, multi-row and key-miss fallbacks.
Run `tests/test_unit_laguna_gguf_runner.py`,
`tests/test_unit_laguna_target_ar_bench.py`,
`tests/test_unit_laguna_mixed_pair_reuse_runtime.py` and
`tests/test_unit_laguna_q5_swar_pair_runtime.py`, corresponding primitive
GPU comparisons and R0249/R0250's public generation/lifecycle gate.
No projection implementation, control or backend default changes here.

### R0281-Q4HEAD — Scalar Q4 head comparison control

`use_q4_lm_head_local32_fixed_meta` and target-AR's
`--disable-q4-lm-head-local32-fixed-meta` still select and report the scalar
Q4 head comparison. `_project_and_sample()` passes the resolved variant;
`_project_rows_last()` and `_project_rows_all()` do not. The raw-ABI
registered-sibling selector preserves the ordinary dispatch for missing keys,
non-raw layouts and multiple rows. The separate Q6 producer-top1 path does
not use this Q4 control.

**Decision needed:** keep the constructor/CLI opt-out as a supported diagnostic,
or move the comparison to a scoped projection owner before retiring it.
**Clearing action:** migrate both target-AR construction sites and result fields,
then run `tests/test_unit_laguna_gguf_runner.py` and
`tests/test_unit_laguna_target_ar_bench.py`. Preserve scalar-versus-bulk/verifier
selection, exact fallback bodies and backend capability behavior. Exercise
actual Q4 full logits and a missing-key fallback, then R0249/R0250's public
continuation/lifecycle gate; do not treat a Q6 producer-top1 run as exercising
this Q4 path. No control, kernel or default changes in this reconciliation.

### R0281-IQ3 — c1 IQ3 schedule comparison interface

The gfx1100 package defaults `wave10_fused`; a missing wave10 key restores
its `wave4_reduce` schedule, while packages without either capability default
to `serial_weighted`. `iq3_c1_down_schedule` and target-AR's
`--iq3-c1-down-schedule` still expose all three modes and report the resolved
schedule. Wave4 uses a separate producer plus weighted reducer; wave10 and
serial select different weighted composites. Multi-row execution bypasses
this c1 composite owner. R0258 owns the separate grouped-prefill policy,
not these c1 schedules; verifier output-tile selection is separate too.

**Decision needed:** retain the constructor/CLI modes as supported diagnostics,
or migrate schedule comparisons to a scoped owner before retiring the public
selector. **Clearing action:** migrate both target-AR construction sites and
result fields, preserving the wave10-miss fallback, wave4 producer/reducer and
serial chain. Run the three `test_laguna_iq3_*` schedule/removed-signbit nodes
in `tests/test_live_laguna_moe_gpu.py`, runner and target-AR units; then exact
primitive comparisons and R0249/R0250's public c1/multi-row continuation and
cleanup gate. Keep unsupported/backend fallbacks, grouped-prefill consumers
and independent verifier tiles. No mode, kernel or default changes here.

### R0281-SPLIT — Split-attention and tile16 comparison controls

R0339's historical gfx1151 transfer entry is consolidated here; its promotion
is already reflected in package defaults. The profiler's explicit bundle
still sets global/SWA/tile16 thresholds together, while unset selection
preserves backend resolution.

Both gfx11 packages now define global/SWA split and SWA tile16 thresholds.
The session and KV allocator expose `use_split_attention`,
`use_swa_split_tile16` and explicit crossover values. Target-AR and the
long-context profile still pass these settings and report resolved values.
They affect workspace allocation as well as launch selection; contradictory
disable-plus-threshold requests are errors, not silently ignored controls.
Capacity filtering and below-threshold paths preserve ordinary readers.
Gate fusion and wave-local reduction are separate comparisons.

**Decision needed:** retain explicit crossover/disable controls as supported
resource and profiling policy, or migrate callers to a scoped KV policy owner
before retiring constructor/allocator/CLI arguments. **Clearing action:**
migrate both harnesses and their result fields with the session/allocator,
preserving workspace lifetime, invalid-combination errors, package defaults
and registered score/token4 fallbacks. Run the fake-runtime bounded-workspace
node in `tests/test_gpu_laguna_kv_attention.py`, runner, target-AR and
long-context-profile units. Exercise actual global/SWA lengths below, above
and unrelated to each crossover, wrap/continuation and multi-row fallback,
then R0249/R0250's public generation and allocation-cleanup gate. No control,
threshold, allocation or attention implementation changes here.

### R0281-FUSION — Attention fusion and reducer comparison controls

R0338's historical gfx1151 head/KV transfer entry is consolidated here; the
transfer is already enabled. Host CLI-to-constructor tests preserve unset,
true and false requests independently of the split-attention bundle.

Both gfx11 packages enable head/KV fusion, split-gate fusion and wave-local
SWA reduction. Target-AR still exposes the separate disable controls; the
long-context profile also passes and reports head and split-bundle settings.
`use_head_kv_fusion=False` selects the registered head-plus-writer chain;
missing fused keys preserve that chain. This is independent of attention
reducer selection.

The KV allocator requires wave-local reduction to have both split attention
and split-gate fusion. A gate-fusion comparison must also disable wave-local;
otherwise construction raises a named dependency error. Wave-local false
keeps shared-statistics gated reduction and disables dependent SWA overlays.
Later fused variants do not make these controls aliases.

**Decision needed:** keep each constructor/CLI comparison as a supported
diagnostic, or migrate its callers to scoped head/KV and attention-policy
owners before retiring it. **Clearing action:** migrate target-AR construction
sites/result fields and long-context profile controls together, preserving
fusion dependency errors, resolved-variant reporting and unfused chains.
Run the fake-runtime bounded-workspace node in
`tests/test_gpu_laguna_kv_attention.py`,
`tests/test_unit_laguna_global_head_wave0_tree_runtime.py`, runner, target-AR
and long-context-profile units. Compare the head/KV and split-gate primitive
chains and wave-local/shared-statistics reducers on GPU, including missing
keys, below-threshold lengths, wrap and multi-row paths; then run
R0249/R0250's public continuation/lifecycle gate. No fusion, control, kernel
or default changes in this reconciliation.

### R0281-IQ2 — c1 IQ2 grid comparison control

`use_iq2_grid64` and target-AR's `--disable-iq2-grid64` still select and
report the c1 expanded-grid versus compact-grid gate/up comparison. The
session resolves gfx1100's package default; the lower-level plan's false
default does not describe public session selection. Multi-row dispatch keeps
its compact-grid route; R0258's grouped-prefill policy and the removed local64
reduction selector are separate concerns.

**Decision needed:** retain the constructor/CLI grid comparison as a supported
diagnostic, or migrate callers to a scoped c1 policy owner before retiring it.
**Clearing action:** migrate both target-AR construction sites/result fields,
then run the host-only grid-route node in `tests/test_live_laguna_moe_gpu.py`,
`tests/test_unit_laguna_iq2_local64_reduction_runtime.py`, runner and target-AR
units. Preserve independent c1/multi-row selection and actual registered
compact-grid fallbacks. Compare exact primitive outputs and exercise
R0249/R0250's public continuation/lifecycle gate with the c1 route confirmed.
Do not reintroduce rejected local64 runtime controls or alter grouped-prefill
consumers. No grid implementation, control or backend default changes here.

### R0281-DFLASH — Verifier tiles and shared true-AR mismatch

`iq3_selected_down_tile` and `laguna_dflash_category_bench.py` still expose
tile1/tile4. The gfx1100 tile4 key is registered; gfx1151 lacks that key and
an explicit request fails with a required-kernel error. Public session opening
uses tile1 without a tile override. This verifier choice is separate from
R0281-IQ3's c1 schedule.

**Unresolved failure:** the historical gfx1151 packet
`benchmarks/results/2026-08-20-gfx1151-laguna-dflash-iq3-tile4-rejected.json`
records identical true-AR mismatches under tile1 and tile4, not a tile4
trajectory regression. It does not establish a current-stack verdict.
**Clearing action:** with the local target checkpoint noted at R0249/R0250
and its DFlash drafter available, run
`python scripts/laguna_dflash_category_bench.py "$LAGUNA_MODEL" "$LAGUNA_DRAFTER" --backend hip_gfx1151 --direct-gguf --iq3-selected-down-tile 1 --output /tmp/laguna-dflash-current.json`.
Preserve the fixed-horizon equality and request-reset checks; if the mismatch
reproduces, trace the first divergent verifier/prefill/commit state rather
than attributing it to the output tile. Then run the public blocking/streaming
DFlash gate and verify continuation, cancellation and allocation cleanup.

Keep the working gfx1100 tile route and exact tile1 fallback. Decide whether
the explicit tile control remains a supported diagnostic or moves with its
category caller to a scoped verifier policy; do not add identity-based
admission or disable working execution because a measurement is absent.
Host plan, category/public-gate, cycle and fake-provider integration tests
pass; no current full-model, GPU arithmetic or performance claim is made.

### R0281-PARO-ARGRAPH — gfx1151 greedy graph mismatch

The resident-session policy selects eager greedy c1 on gfx1151 and graph replay
on gfx1100. Explicit `capture_decode_graph()` remains callable by diagnostics.
The gfx1151 p512/d8 mismatch recorded in July reproduces on the October 27
stack: eager/graph/eager runs have the same prefill seed; both eager runs have
identical IDs and final logits, but graph first diverges at decode index 2.
All three runs produce finite logits and release tracked allocations. See
`worklog/entries/20261027T224932.969200Z-cleanup-refactor-r0281-paro-argraph-1d12bf.md`
for the explicit backend, checkpoint, fixture and reproduction. This is separate
from R0001's MTP verifier graph regression and makes no current timing claim.

**Remaining work:** trace and repair the first graph/eager state or arithmetic
difference before removing the gfx1151 eager policy. Keep the working eager
route, gfx1100 graph route and explicit diagnostic capture; absence of a
measurement on another architecture is not itself a capability failure.
**Clearing action:** repeat the linked eager/graph/eager control, then compare
hidden, Conv/GDN and live KV at each transition with identical inputs. Extend
the repair gate to long replay, reset/reuse, cancellation and shutdown. The
fixture CLI in `scripts/qwen35_decode_graph_fixture_gate.py` compares generated
IDs and final logits, not all persistent state; it also labels its backend
statically, so verify the runner's resolved backend rather than trusting that
field. Run the graph-policy nodes in
`tests/test_unit_generation_qwen35_paro.py` and
`tests/test_unit_qwen35_resident_batch_layout.py`, then exercise public
`LLM.generate()` with the repaired route confirmed and allocations drained.
Evaluate capture-inclusive wall separately before an automatic performance
choice. No runtime policy, kernel or graph implementation changes here.

### R0281-PARO-RAGGED — Ragged packed prefill and diagnostic ownership

Unequal prompt segments automatically use per-segment linear and full-attention
execution, reported as `per_segment_ragged_exact`. Equal-length slabs retain
`packed_segments` and `packed_varlen`. The July SOL-P2 gate recorded a packed
ragged state/token mismatch and validated the per-segment fallback; it did not
show that ragged requests cannot run. Keep this working fallback and its
slot-local Conv/GDN/live-KV ownership. An October 27 gfx1151 c3 check at
prompt lengths 8/9/11 preserves independent-c1 tokens, all linear state and
live KV through shrinking widths and retired-slot immutability, then frees
tracked allocations. That check exercises the fallback with serial decode,
not the rejected packed candidate or public native-width selection. See
`worklog/entries/20261027T225807.809651Z-cleanup-refactor-r0281-paro-ragged-4633ca.md`.

`HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR` and
`HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN` still have
hidden-bisect callers; the retained benchmark also controls the linear path
and isolates both flags for its c1 reference. These flags force the diagnostic
per-segment route. Setting them to zero does **not** bypass the automatic
ragged fallback, so the existing CLI cannot re-test packed ragged kernels by
simply requesting `packed_segments`/`packed_varlen`.

**Remaining work:** repair the original packed ragged difference under an
explicit diagnostic owner, or keep per-segment execution as the supported
ragged algorithm. Decide separately whether the live comparison controls stay
supported or move to an owner-scoped interface. Do not delete their callers,
exact fallback, equal-length paths or primitive implementations as stale
release-window cleanup.
**Clearing action:** first run
`scripts/qwen35_batch_shrinking_correctness.py --model "$PARO_MODEL" --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --batch-size 3 --prompt-lengths 8,9,11 --steps-per-width 1 --max-layers 40 --max-sequence-length 1024 --survivor-slot 1 --decode-execution serial --json /tmp/paro-ragged-current.json`
on the intended backend and verify resolved device and prefill metadata. This
checks the working fallback, not the packed candidate. Add a test-only packed
ragged bypass before comparing identical input trajectories and locating the
first unequal hidden, Conv/GDN or live-KV state. Cover c8, sparse slots,
continuation, EOS/cancellation and cleanup on each applicable backend. Run
`tests/test_unit_qwen35_resident_batch_layout.py`,
`tests/test_unit_qwen35_paro_serial_bridge.py` and
`tests/test_unit_qwen35_batch_partition_bench.py`, then public ragged
`LLM.generate()` requests with the selected path confirmed. No prefill policy,
flag, arithmetic or runtime implementation changes in this disposition.

### R0281-GGUF-REPACK — Decode layout control and comparison callers

`resolve_gguf_decode_repack()` is the shared default-on resolver for admission
and materialization; an explicit API value takes precedence over the environment.
`HIPENGINE_GGUF_DECODE_REPACK=0` selects raw materialization for diagnostics and
memory/layout comparisons. The GDN trajectory, semantic/profile, true-AR category
and MTP harnesses still request that control. Repacking is per-tensor: raw IQ
leaves whose optimized layout is unimplemented remain raw even with the model
request on. Do not conflate this resource/layout policy with public
`use_wmma_prefill` / `use_gemv_decode` launch selectors or remove raw quant support.
The resident benchmark resolves launch selectors from the public session defaults;
`--public-ar-profile` is now only a compatibility spelling for older commands.

**Decision needed:** keep environment/API layout selection as supported resource
and diagnostic policy, or migrate comparison callers to an explicit load owner
before retiring the environment reader. A passed release window alone does not
migrate those callers. **Clearing action:** preserve one resolver meaning across
admission and materialization, explicit-value precedence, per-tensor capability,
byte planning, cleanup and the existing observed raw-fastpath containment. Do
not turn layout selection into an artifact/measurement allowlist or change the
activation math as a side effect of repacking. Run
`tests/test_unit_qwen35_gguf_decode_repack_semantics.py`,
`tests/test_unit_qwen35_gguf_decode_repack_dispatch.py`,
`tests/test_unit_qwen35_gguf_fastpath_safety.py`,
`tests/test_unit_gguf_gdn_semantic_gate.py` and
`tests/test_unit_qwen35_gguf_bench_metadata.py`; then exercise actual public
`LLM.generate()` default/raw requests and migrated harness comparisons with
compatible local quants on each applicable backend. Confirm actual layouts,
effective selectors, no duplicate raw/T16 residency, bounded allocation and
cleanup. No layout, default, API or environment behavior changes here.

### R0281-SILU-FLOOR — Bulk-prefill and verifier fused-FFN policy

The generic dense Q4T16 fused dual+SiLU owner still starts at row 33 for
K5120/N17408. Later physical-target policy already overrides that generic floor:
gfx1100 selects registered row32/row48 siblings for physical rows 32/36, and
its optional production row-28 schedule preserves periodic strict layers.
The earlier slower verifier comparison therefore does not describe every
current physical owner. Preserve these scoped policies, the generic parent,
retile fallback and exact unfused chain.

**Remaining work:** replace the generic row-count proxy with explicit
bulk-prefill versus verifier policy, shared by all production prefill entries.
Do not lower the prefill floor as a cleanup side effect or treat this item as
permission to remove current physical-target siblings. The fused dispatch is
resolved directly by `launch_gguf_linear_pair_silu()`; its final choice is not
stored in `_DISPATCH_RESOLVE_CACHE` or `_PAIR_DISPATCH_RESOLVE_CACHE`. Any new
scope must participate in whichever cached decisions actually consume it,
rather than adding unrelated cache-key fields from the old prescription.

**Clearing action:** add nested/error/thread isolation and populated-cache
control/candidate/control tests before introducing an execution-local prefill
policy. Bind ordinary, packed and resumable prompt entries to the same owner;
keep verification outside that scope. Exercise below/above/unrelated row counts,
physical row28/32/36 schedules, quant/shape misses and registered-key fallback.
Run `tests/test_unit_gguf_linear_dispatch.py`,
`tests/test_unit_gguf_linear_dispatch_cache.py`,
`tests/test_unit_q4_row48_control_retirement.py` and the host-only physical/floor
nodes in `tests/test_gpu_gguf_q4_k_t16_dense.py`. Before changing a threshold,
run its fused/unfused primitive oracle and same-host public prefill/verification
comparison with selected variants, output/state, capture-inclusive wall and
cleanup recorded. No floor, dispatch, arithmetic or default changes here.

### R0281-AR-LOOP — Direct GGUF compatibility loop and shared flush helpers

Public `LLM` construction wraps the GGUF generator in `SubmitPollTextGenerator`,
whose resident factory creates `Qwen35GGUFResidentModelRunner`. That owner does
not call `_generate_ar_serving_slots()`, including for its scalar capability
fallbacks. The old entry is still called by direct
`Qwen35GGUFBringupGenerator.generate_detailed()` for prepared multi-prompt greedy
requests; direct-generator tests exercise packed, streamed and scalar arms.
It is a live compatibility path, not an independent public model owner.

A second direct prompt-list route, `_generate_greedy_batch()`, remains reachable
when `native_batch_decode` is enabled and the prepared packed/stream route is
not selected. It creates a call-local scheduler, compacts physical slots and
prefills admitted rows individually. Direct tests cover this eager/graph path
and readmission. The resident factory uses the same construction field to
select session execution routes, so deleting that field with the old loop would
also change resident preparation.

The old prompt-list ledger's persistent-serving task is complete:
`SubmitPollTextGenerator` retains its runner and loop across calls, server
submission/cancellation uses that owner, and the GGUF runner implements prefix
reuse and ragged packed full-prompt prefill. `GlobalDeviceKVPool` and its device
pools implement growth, shared-prefix ownership and idle shrink. Sampled rows
also use the resident model tick, with native or host selection as supported;
this is not a claim that every sampler mode uses a native batched sampler.
Native sampler coverage belongs to the separate sampler item. Chunked and
prefix-suffix prefill keep their capability fallbacks; do not infer universal
packed prefill from the full-prompt path.

The adjacent `_flush_ar_packed_decode_owners_if_chunk_changed()` and
`_flush_ar_packed_decode_owners()` helpers are also used by the resident runner
for physical-group transitions, scalar fallback and reclaim. `_GGUFARServingSlot`
is shared by both paths. Do not delete the whole legacy region or its state
record as an oracle extraction. The resident runner's `_step_native_serial()`
keeps its own unsupported-shape and sampled contracts; the legacy loop is not
its fallback.

**Decision needed:** retire both direct-generator batched compatibility loops,
or move them to an explicit diagnostic adapter with their callers. Then extract
only the loop-specific methods; first give the shared flush operations an owner
that both routes can consume. No release-window note proves these dependencies
have migrated. **Clearing action:** preserve the resident bypass sentinel in
`tests/test_live_generation_qwen35_gguf_sampling.py::test_gguf_submit_poll_runner_owns_and_reuses_resident_sessions`,
which rejects both direct loops with `native_batch_decode=True`, then migrate
the direct tests and check uniform/ragged prompts, widths below
and above eight, shrinking/EOS, unsupported packed prefill/decode, dirty-owner
flush and error cleanup. Preserve the older path's physical compaction,
readmission and eager/graph checks or migrate them to equivalent diagnostics.
Keep resident session-route selection independent of compatibility-loop removal.
Exercise public `LLM.generate_detailed()` plus
submit/poll cancellation and streaming after extraction. This disposition pins
existing ownership without changing runtime behavior.

### R0281-CONV — Convolution schedule selector and isolation profiles

`HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE` accepts `auto`, `baseline` and
`tile32x128`. Auto selects tiled convolution on gfx1151 and baseline on gfx1100.
The gfx1100 split is deliberate: the architecture-local tiled diagnostic
preserved primitive output/state but its full-model screen did not support
promotion. See
`benchmarks/results/2026-07-14-gfx1100-gguf-prefill-post-transfer-profile.json`;
this is a completed evaluation, not a pending transfer. Both registered
implementations remain executable explicitly.

`scripts/qwen35_readme_sweep.py` still pins baseline convolution in its
conservative and derived kernel-isolation profiles. The runtime caches its
resolved callable on the full-stack runner, so changing the environment after
first use does not reconfigure that runner. Segmented, state-capture and
low-precision convolution paths are separate and do not all use this selector.

**Decision needed:** retain the selector as a supported diagnostic or replace
its live isolation-profile callers with an explicit construction-time schedule
before removing the environment reader. Preserve backend auto choices, exact
baseline/segmented/state-capture fallbacks and the tiled implementation used by
gfx1151. **Clearing action:** run
`tests/test_unit_qwen35_gguf_conv_prefill_policy.py`,
`tests/test_unit_qwen35_linear_attn_conv_plan.py` and
`tests/test_unit_qwen35_prefill_kernel_profile.py`; migrate the sweep profiles
without changing their other selectors. Then run the guarded primitive test
`tests/test_gpu_qwen35_linear_attn_conv_tiled_prefill.py` and public generation
with route engagement and cleanup recorded on each intended backend. Do not
turn this cleanup into automatic gfx1100 promotion or delete kernels based on
old schedule names.

### R0281-SELECTED-PAIR — Selected-MoE pair-reuse control retirement

The three controls remain live: `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE`,
`HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE` and
`HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE`. Unset follows the packed
owner's backend scopes; explicit `0` restores the per-lane owner and `1`
requests pair reuse within the supported geometry. Q4 uses eight input rows
and 64 selected lanes; Q5/Q6 down use 64 input rows and 64 selected lanes.
The SPECDEC2 request-local Q4 small-row scope is separate and stays under
its existing ledger entry.

The old pending-transfer condition is stale. Both backends enable Q4 C8
pair reuse. gfx1151 also enables Q5/Q6 down, while gfx1100 keeps their package
floors at zero after completed performance evaluations, not missing evidence.
The transfer decisions are in
`benchmarks/results/2026-09-05-gfx1100-gguf-q4-selected-pairreuse-c8-retained.json`,
`benchmarks/results/2026-09-05-gfx1100-gguf-q5-selected-down-pairreuse-rejected.json`
and
`benchmarks/results/2026-09-05-gfx1100-gguf-q6-selected-down-pairreuse-rejected.json`.
The explicit paths remain runnable; do not force a default transfer as cleanup.

**Decision needed:** retain the controls as diagnostics or replace both on/off
requests with an explicit scoped policy before removing their readers. The
batch-route gate clears and restores these variables to exercise package policy;
its capture metadata must migrate too. Preserve per-lane fallback kernels,
backend defaults and the separate request-local small-row scope.
**Clearing action:** run
`tests/test_unit_qwen35_gguf_compact_moe_gemv_routing.py` and
`tests/test_unit_execution_profile_gguf_batch_route_gate.py`; migrate the gate
and `docs/ENVS.md` with any reader removal. A runtime migration also needs the
primitive pair-reuse parity tests and public C8 MoE generation on both intended
backends, with selected callables, neighboring/non-C8 widths and cleanup
recorded. A release-window deadline alone cannot replace these contracts.

### R0281-ROUTER — Cooperative router profile and diagnostic controls

`HIPENGINE_GGUF_ROUTER_F32W_COOP` and
`HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER` both default on in the runtime.
The first selects cooperative BF16-hidden/F32-weight c1 routing when a matching
registry implementation exists; otherwise the separate logits/select chain
runs. F32-hidden post-norm routing uses its separate chain. The second selects
the persistent cooperative variant with the scratch-owned self-resetting counter;
its opt-out uses the cooperative variant that clears an alias of the selected-ID
buffer before launch. These are different call interfaces, not dead synonyms.

`hipengine/generation/qwen35_moe_gfx1151_profiles.py` already binds strict off
and production on through these variables. Its router manifest records the base
`router_logits` family for both and relies on the binder to select the fused
`router_topk_split_shared` route. The gfx1100 MoE profile uses a different
binder; it must not acquire gfx1151 policy as an incidental cleanup.
The c1 and batch route gates, control smoke and arbitrary-c lifecycle harness
also retain these controls. Historical validation stays in
`benchmarks/results/2026-08-16-zbook-qwen36-c1-router-retained.json` and the
linked profile evidence, not as a new benchmark claim here.

**Remaining work:** publish the actual selected router variant and counter
ownership in a construction-time plan consumed by the runner, while preserving
strict/production semantics, direct defaults, diagnostic on/off requests and
the exact unfused fallback. Migrate profile binders and capture metadata before
removing readers. This is part of the existing RF-1 resolved-plan direction;
a profile registration alone does not remove its environment dependency.
**Clearing action:** run
`tests/test_unit_qwen35_moe_gfx1151_profiles.py`,
`tests/test_unit_qwen35_gguf_compact_moe_gemv_routing.py`,
`tests/test_unit_execution_profile_gguf_c1_route_gate.py` and
`tests/test_unit_execution_profile_gguf_batch_route_gate.py`. A runtime migration
also needs guarded cooperative/persistent primitive parity and public MoE c1
generation with selected variant/counter ownership, repeated graph/eager calls,
fallback engagement and cleanup recorded. Keep both cooperative variants and
unfused primitives until their callers and ownership contracts have migrated.

### R0281-MTP-CAP — Packed MTP batch policy and caller separation

`_MTP_SERVING_TARGET_BATCH_MAX_SLOTS = 4` in
`hipengine/generation/qwen35_gguf.py` still serves three purposes: target-verifier
chunk width, packed MTP prompt-prefill eligibility and startup warmup grouping.
Verifier partitioning avoids a singleton tail (five slots become three plus two);
multiple chunks can use independent owner streams, with serial chunk execution
when stream capabilities are absent. A first-chunk refusal may fall back to
per-slot verification; a refusal after an earlier chunk advanced state raises
instead of silently replaying that state.

The MoE `generate_speculative_mtp_detailed()` adapter still reaches this policy;
the dense transactional route is separate. The limit is a batch-selection policy,
not a kernel capability declaration or a maximum number of requests the server
may admit. The rejected single-eight-slot experiment and chunk-stream results
are historical evidence in
`worklog/entries/20260705T071433.000000Z-legacy-2026-07-05-gguf-server-mtp-c8-stream-verify-chun-17986a.md`.
They do not establish the best width for every current artifact or backend.

**Remaining decision:** retain bounded chunks, or tune a wider verifier and give
prefill/warmup their own policies. Do not delete the shared constant as dead code
or change prompt-prefill admission as a side effect of verifier tuning.
**Clearing action:** run
`tests/test_unit_gguf_mtp_serving_chunk_policy.py` and the batch-verifier,
packed-prefill and MTP warmup nodes in
`tests/test_live_generation_qwen35_gguf_sampling.py`. A runtime policy change also
needs public explicit-MTP requests at neighboring, unrelated and larger widths,
selected chunk/owner/stream traces, exact commit/rollback and refusal-after-progress
checks, repeated requests and cleanup. Compare complete-wall and latency on the
intended backend against the bounded policy before choosing a tuning result.
Keep per-slot unsupported-shape fallback and request-local state isolation.

### R0281-VERIFY-TIMING — Packed verifier profiling interface and teardown

`HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS` is a default-off diagnostic
read by `Qwen35GGUFResidentSession.verify_target_blocks_batch()`. It creates
same-stream HIP events only for host results; `device_result=True` bypasses the
recorder. Markers propagate through shared layer owners, and the legacy MTP
adapter copies the resulting buckets into target timing metadata. No checked-in
script sets the variable; external diagnostic commands can still request it.
The event implementation already lives in `qwen35_gguf_timing.py`, so another
recorder extraction would duplicate completed ownership work.

**Concrete cleanup gap:** the packed caller starts the recorder before embedding
and resolves it only on success. An exception before resolution bypasses event
cleanup; the recorder's own resolve-time `finally` cannot cover that interval.
Other prefill/decode timing callers already have explicit finalizers.
**Remaining work:** close the packed recorder across every post-creation exit,
then decide whether its opt-in belongs in an explicit profiling argument/helper
instead of process environment. Preserve default-off behavior, device-result
bypass, stream association, bucket aliases and negative-time accounting.
Do not enable diagnostic events on the default path to retire the flag.

**Clearing action:** add fault injection at event start/embedding/layer/head/
scatter and prove all created handles are released, including retryable destroy
failures. Run `tests/test_unit_qwen35_gguf_timing.py` and
`tests/test_unit_gguf_packed_verify_layout.py`; a runtime wiring change also needs
a guarded HIP event smoke and repeated public MTP requests with identical output,
selected instrumentation mode and cleanup recorded. Instrumented diagnostic
results are not uninstrumented performance evidence.

### R0281-LAUNCH-BOUNDS — Selected-WMMA compiler tuning surface

`HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` is a build option, not a runtime
feature switch. Raw Q4, T16 Q4, raw Q5/Q6, T16 Q5/Q6 and Q5 iu8 selected-prefill
owners accept 1/2/4/8, append the launch-bounds macro to compiler flags and
produce distinct cache keys. Unset keeps each source's default. Plan and build
consume the same option; changing it after a library is loaded does not patch
that library's code. Existing no-GPU build-plan tests exercise the option.

**Remaining decision:** keep the explicit compiler experiment as a supported
build control, or migrate it to shared build configuration and update every
plan/build caller. It is not a rejected runtime route to delete. The old
concurrency probe does not choose one compiler occupancy setting for all
families, architectures and current workloads. Do not promote a launch bound
or remove an explicit build request as incidental ledger cleanup.

**Clearing action:** run
`tests/test_unit_gguf_selected_wmma_launch_bounds.py` plus the registry/build-plan
nodes in the selected-WMMA GPU test modules. A build-interface migration must
preserve invalid-value errors, plan/build parity, distinct cache keys and
unchanged unset output. A compiler/default change additionally needs each
intended family's guarded numerical/fallback tests and public generation with
actual selected kernels, complete-wall/latency and cleanup recorded across
neighboring and unrelated concurrency/row regimes on the intended backend.

### R0281-AR-DECODE / R0308 — Packed decode controls and scalar baseline callers

`HIPENGINE_GGUF_AR_PACKED_DECODE` is default on and still read by both the
resident model runner and the direct compatibility AR adapter.
`scripts/gguf_concurrency_baseline.py` sets it off with packed prefill and stream
decode for its explicit serial control. `scripts/server_f1_concurrency_bench.py`
and `scripts/gguf_live_server_bench.py` also construct packed-versus-serial
server environments. Deleting the reader without migrating these callers
silently changes their comparison basis. The resident route must keep its
independent capability, row-width and sampler compatibility fallbacks.

**Remaining work:** migrate the diagnostic callers to an explicit resolved
execution control, or designate a separate serial oracle before retiring the
process-wide env reader. Update test fixtures and effective-route metadata in
the same unit. Preserve default packed execution, scalar unsupported-shape
behavior, physical grouping and slot/state cleanup. This is distinct from
R0281-AR-LOOP's direct compatibility-loop extraction; the resident scheduler
also consumes the flag.

**Clearing action:** run `tests/test_unit_gguf_ar_decode_control.py`,
`tests/test_unit_gguf_concurrency_baseline.py`,
`tests/test_unit_server_f1_concurrency_bench.py` and
`tests/test_unit_gguf_live_server_bench.py`, plus the packed-decode/fallback and
shared-slot grouping nodes in `tests/test_live_generation_qwen35_gguf_sampling.py`.
Reader retirement also needs real public packed and explicit-serial requests at
neighboring/unrelated logical widths, ragged and shrinking groups, repeated
output, selected path/fallback reasons and zero tracked cleanup. Historical
F1/E3 packets establish prior results, not migration of these live writers.
R0308's duplicate release-window item is consolidated here; elapsed time alone
does not retire an explicit serial comparison. Its original result history
remains in the sweep's pinned source and immutable worklogs.

### R0281-AR-PREFILL / R0309 — Packed prefill diagnostic control migration

`HIPENGINE_GGUF_AR_PACKED_PREFILL` defaults on. Startup warmup, the compatibility
adapter and the resident scheduler read it; `scripts/gguf_concurrency_baseline.py`
still disables it for the explicit serial comparison. Reader retirement must
migrate that control alongside R0281-AR-DECODE, not silently turn the serial
baseline into packed prefill.

Intermediate-tail suppression is implemented. Both gfx1100 and gfx1151 enable
`GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK`; bounded chunk execution passes only
slots finishing their prompt to output-norm/LM-head sampling. The layer-outer
route samples after its final layer pass. Hidden-row capture and MTP sinks
have their own full-row requirements. Keep the unmasked package fallback and
all required hidden seeds; do not remove them as duplicate output work.
The old intermediate-tail task and default-off instruction are stale.
Historical mask evidence remains in
`benchmarks/results/2026-09-06-gfx1100-gguf-packed-prefill-final-output-mask-retained.json`;
prior numerical-contract notes remain in their immutable worklogs, not as
permission to disable the implemented default.

**Remaining work:** give serial diagnostics an explicit prefill control and
migrate warmup/compatibility/resident consumers coherently before retiring the
env reader. Preserve packed arithmetic, unsupported-shape scalar fallback,
final-slot masks, target hidden sinks and prompt-length accounting.
**Clearing action:** run `tests/test_unit_gguf_packed_verify_layout.py`,
`tests/test_unit_gguf_concurrency_baseline.py` and packed-prefill/warmup nodes in
`tests/test_live_generation_qwen35_gguf_sampling.py`. A reader migration also
needs public packed and explicit-serial requests with short/long/ragged prompts,
selected route, final-output/hidden-seed accounting, repeated output and cleanup
recorded. Do not re-run historical arithmetic promotion as a prerequisite for
running the existing default; apply the numerical gates only to changed math.
R0309's duplicate release-window item is consolidated here. Historical
verifier-as-prefill, state-audit and F1/E3 results remain in the sweep's pinned
source and immutable worklogs, not as fresh claims about this unchanged route.

### R0281-FAIR-BURST — Scheduler control retirement decision

`fair_prefill_burst_chunks` is a positive integer in `EngineLoopConfig`, with
`HIPENGINE_FAIR_PREFILL_BURST_CHUNKS` and `--fair-prefill-burst-chunks` inputs.
It is not just an old rollback: the production-load gate declares and
reconfigures it per candidate, loop snapshots report it, and server metrics
export it. Registry-selected Q4_K_M defaults are one chunk on gfx1100 and two
on gfx1151; explicit env values override those defaults.

The fair scheduler uses the bound to finish short cold cohorts before decode
and to limit consecutive chunks when multiple prompts remain. A lone
staggered prompt keeps one-chunk alternation. Other scheduling policies do
not select their decode order from this burst bound.

**Decision needed:** retain this as an operator scheduling control, or replace
it with a named workload-derived policy. A release window and a defaults-only
rerun do not migrate the live CLI/env callers, tuning protocol, reconfiguration
or telemetry. Keep those interfaces and defaults unchanged until that decision;
do not add backend branches to the scheduler.

**Clearing action:** if retained, document the supported control and remove
this retirement item. If replaced, migrate `hipengine/llm.py`, engine-loop
configuration, `scripts/gguf_production_load_gate.py`, protocol fixtures and
metrics together. Run `tests/test_integration_generation_engine_loop_burst.py`,
engine-loop CLI/env nodes in
`tests/test_integration_generation_batch_scheduler.py`, generator-default nodes
in `tests/test_unit_llm_generate.py`, load-gate tuning nodes and resident-loop
metrics nodes in `tests/test_integration_server_api.py`. A behavior change also
needs mixed-arrival public serving validation of bounded decode interruption,
completion, cancellation and cleanup; cold-prefill throughput alone does not
validate fairness.

### R0281-NATIVE-CYCLE / R0384 — Native diagnostic interfaces and graph composition

The four `scripts/gguf_mtp_bench.py` switches are live diagnostic interfaces:
`--native-spec-target-cycle` selects the target submission (N1),
`--native-spec-device-accept-commit` adds target acceptance/commit (N2),
`--native-spec-complete-cycle` joins proposal, target and provider repair behind
one adapter call (N3), and `--native-spec-proposal-graph` adds the separate
proposal submission (N3P). `scripts/gguf_ar_mtp_suite.py` constructs all four
routes; `scripts/gguf_mtp_verifier_rocprof.py` consumes the target switch.
Deleting these inputs would remove distinct comparison boundaries, not dead
runtime branches.

GGUF target launchers are registered for both gfx1100 and gfx1151 under
`w4_gguf/native_v1_b2_target_graph`. The separate
`w4_gguf/native_v1_b2_proposal_graph` launcher is registered only for gfx1100.
That proposal launcher is not the shared NextN proposal-stage cache, which
already has a gfx1151 consumer. Keep their backend scopes distinct. The
complete-cycle adapter accepts B1/B2 even though target-only graphs support
other buckets. Unsupported capture/configuration falls back before mutation;
launch/readback failures must not replay an already-mutated transaction.

R0384's blanket long-graph non-routability note is superseded by current
selection. The transactional verifier considers native target graphs when its
caller permits them; gfx1151 declares a long-context graph bound, while other
backend/storage/state policies keep their own limits. The eight-entry target
cache keys candidate budget, accept/commit mode, immutable context bucket and
sampled acceptance; compatibility also checks execution options and bindings.
Eviction closes the oldest owner. A cycle cannot span short/split-K schedules,
split-workspace buckets, kernel-family transitions or resident capacity.
These are resource and graph-topology contracts, not missing-benchmark gates.
Do not remove the live cache because the historical August timing was neutral;
that result does not establish current capture-inclusive economics.

The provider-neutral PARO/DFlash target boundary is implemented and has its own
ledger item. A complete cross-provider proposal/target lifecycle and a combined
proposal-plus-target submission remain separate work, not prerequisites for
running existing GGUF graphs. Historical comparisons remain in the July 19
`llama-compat-native-cycle-n2`, `-n3` and `-n3p` result artifacts; no new timing
claim is made here.

**Decision needed:** retain these diagnostic stages as supported interfaces,
or replace them with one explicit stage selector while preserving route names,
effective options, result metadata and profiler comparisons. Treat combined
graph composition as an independent experiment; do not remove working controls
or change arithmetic/defaults merely to reduce the flag count.

**Clearing action:** run `tests/test_unit_gguf_mtp_bench_metrics.py`,
`tests/test_unit_gguf_ar_mtp_suite.py`,
`tests/test_unit_scripts_gguf_mtp_verifier_rocprof.py`,
`tests/test_unit_native_spec_cycle.py` and
`tests/test_unit_mtp_resident_draft_device_commit.py`, plus native launcher
registration, fallback, complete-cycle and bucket/cache nodes in
`tests/test_gpu_native_spec_cycle_graph.py` and
`tests/test_live_gguf_native_spec_cycle.py`. An interface migration also needs
real diagnostic and public requests confirming selected submission stages,
accepted-row/cursor ownership, B1/B2 and unsupported-shape behavior, repeat
output and teardown. A combined-graph or arithmetic experiment additionally
needs its applicable numerical and complete-wall gates, separately from this
interface cleanup.

### R0281-PARO-NATIVE — Provider graph and selected-commit control migration

`HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH` selects the native submission wrapper
for cached single-request chain verification. The lazy registry currently adds
only `hip_gfx1100/w4_paro/native_v1_target_graph`; the direct HIP graph remains
the fallback when the control/factory cannot be constructed. The wrapper owns
`VERIFY|ACCEPT`. For all-active B1/B2/B3/B4/B5/B8 PARO buckets with zero-width
hidden capture, it can also own selected linear-state `COMMIT|UPDATE_CURSORS`.
`HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT=0` selects the provider-side commit
comparison. Nonzero BF16 DFlash hidden capture keeps hidden/KV repair outside
the selected-commit boundary. Graph entries own their fixed source/destination
tables, and commit ownership is part of the capture key and binding.

**Fixed:** native launcher exceptions no longer fall through to a second direct
graph launch. Submission may have mutated state before result conversion or
validation raises; those exceptions must reach the request transaction. Only
pre-submission control/factory rejection may return a direct fallback. Terminal
native result statuses already raise. Preserve that distinction during any
control migration.

**Remaining decision/work:** replace the two process-wide selectors with
profile-owned submission/commit variants and an explicit diagnostic comparison,
or retain supported controls with concrete ownership. Preserve provider-specific
repair, result metadata and capture bindings. The old release-window trigger
and cross-provider speed prerequisite do not specify this migration. A missing
backend registration is an implementation gap, not evidence that its hardware
cannot execute the wrapper; evaluate/register supported peers independently,
without identity allowlists or missing-benchmark admission gates. Complete
proposal ownership is separate work and does not block existing target graphs.
Historical selected-commit comparisons remain in
`benchmarks/results/2026-07-20-w7900-paro-mtp-n4plus-selected-commit.json`.

**Clearing action:** exercise both submission/commit variants in
`tests/test_unit_qwen35_resident_batch_layout.py`,
`tests/test_unit_qwen35_paro_verify_graph_cache.py` and provider-launcher nodes in
`tests/test_gpu_native_spec_cycle_graph.py`. A migration needs real PARO and
DFlash requests confirming selected stages, active/inactive and unsupported
shapes, hidden-capture/accepted-row/cursor ownership, independent graph buckets,
repeat output, fault rollback and teardown. Keep the public post-submission
failure nodes in `tests/test_live_paro_mtp_graph.py`; they launch real HIP graphs
with injected result failure and do not claim native-C++ registration on an
unregistered peer. Apply numerical gates only if arithmetic changes.

### R0281-AR-STREAM — Stream decode control and serial diagnostic migration

`HIPENGINE_GGUF_AR_STREAM_DECODE` defaults on in the direct compatibility AR
adapter. It enables per-slot asynchronous decode when packed decode is off,
and independent packed-chunk streams when logical width exceeds the native
eight-slot cap. The old c4+c4 description is stale: the chunk planner uses up
to eight rows and avoids a trailing singleton when splitting a wider group.
The resident model runner uses its own packed/scalar grouping; it does not
select per-slot streams from this reader.

The explicit serial environments in `scripts/gguf_concurrency_baseline.py`,
`scripts/gguf_live_server_bench.py`, `scripts/gguf_production_load_gate.py` and
`scripts/gguf_sampled_api_gate.py` still set the stream flag off. Removing it
alone silently converts those compatibility comparisons to asynchronous decode.
Migrate this control with R0281-AR-DECODE/AR-PREFILL, not as a defaults-only
rerun after a release window. R0281-AR-LOOP separately covers compatibility-loop
extraction; it does not replace the serial comparison.

**Remaining work:** provide an explicit resolved serial/stream/packed
comparison control or designated serial oracle, then migrate writers, route
metadata, cleanup tests and the env reader together. Preserve singleton and
missing-hook fallback, packed-owner flush before per-slot work, launch-before-
read ordering and stream destruction. Once any chunk is launched, do not
retry the group as a serial fallback on a launch/result exception.

**Clearing action:** run `tests/test_unit_gguf_ar_decode_control.py`,
`tests/test_unit_gguf_concurrency_baseline.py`,
`tests/test_unit_gguf_live_server_bench.py`,
`tests/test_unit_server_f1_concurrency_bench.py`,
`tests/test_integration_gguf_production_load_gate.py` and
`tests/test_unit_gguf_sampled_api_gate.py`, plus stream/packed/scalar and
above-eight chunk nodes in `tests/test_live_generation_qwen35_gguf_sampling.py`.
Reader migration needs real default, explicit serial and stream requests at
widths below/above eight and an unrelated width, ragged/shrinking groups,
selected route, repeat output and teardown. Historical F1 throughput stays in
its original artifacts; no new timing claim is made by this triage.

### R0281-MTP-DRAFT-EXECUTOR — Compatibility draft/verifier worker lifetime

The compatibility `_try_draft_mtp_serving_slots_streams` and
`_try_verify_mtp_serving_cycle_chunks_streams` paths create a
`ThreadPoolExecutor` per phase. Their per-slot HIP streams already survive
between cycles and are released by `_close_mtp_serving_slots`. The resident
MTP2 adapter owns separate staged proposal/target paths; it does not read the
old `HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT` or
`HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY` switches. Both rollback readers are
removed: supported multi-slot drafts and multi-chunk verification use streams
by default. Draft singletons, one-token tails, missing seeds and missing
runtime hooks retain pre-launch serial fallback. Verification retains the
four-slot chunk cap and missing-runtime/kernel-hook pre-launch fallback; any
chunk failure after launch raises rather than resubmitting mutated work.
These controls selected device scheduling, not client token/result delivery.

**Remaining work:** decide whether to retire compatibility execution through
the existing loop extraction or give its worker pool a bounded request/owner
lifetime. Do not add another global switch. A persistent pool must retain
per-slot seed/KV ownership, ordered results, deadline/cancellation handling,
complete future draining, failure cleanup and shutdown before its device
resources are released. Profile this path first if claiming pool reuse improves
complete wall; old server throughput comparisons do not establish current
worker overhead or a resident MTP2 benefit.

**Clearing action:** run stream-draft, chunked-verifier and MTP2 nodes in
`tests/test_live_generation_qwen35_gguf_sampling.py`,
`tests/test_unit_gguf_mtp_serving_chunk_policy.py` and the GGUF MTP2 seam tests.
Worker-lifetime changes need direct compatibility requests and public
`LLM.generate()`/serving validation separately, recording which route actually
runs, repeat output, shrinking groups, deadline/cancellation/failure, drained
futures and zero leaked streams/allocations. Do not label a compatibility-only
worker change as a resident scheduler improvement.

### R0281-TARGET-ORACLE — Scalar row oracle versus native verification

`Qwen35GGUFResidentSession.step_rows()` is a scalar layer/full-logit diagnostic
over shared row storage. Its `verify_rows()` wrapper supports all-active chains
only, without parent-state snapshots or a transaction/accepted-row commit. Live
row-oracle tests compare native decode boundaries against it. It is not the
resident MTP2 verifier: that adapter calls native packed target-block/accept/
commit execution, with serial-exact variants where requested. The old task to
create a dependency-aware target forward is not an accurate description of
this oracle's callers or of today's production target path.

**Fixed:** `verify_rows()` validates the whole parent topology before calling
`step_rows()`. A late branch refusal previously left earlier rows advanced.
Tree/inactive refusal and supported chain ordering remain unchanged. Keep this
pre-mutation refusal separate from numerical equality and from transaction
rollback after a real launch.

**Remaining decision:** designate the scalar oracle as supported diagnostic
infrastructure and remove this retirement item, or migrate its full-logit and
layer-boundary consumers before deleting it. Strict unfused and independent
scalar comparison paths are not duplicate production execution. Do not replace
them with a faster candidate that is also the object being checked. Branched
oracle execution would require actual parent-state snapshots; it is not enabled
by relabelling `verify_chain` metadata.

**Clearing action:** run `tests/test_unit_gguf_target_rows_oracle.py`, target-row
nodes in `tests/test_live_qwen35_gguf_target_rows.py`, and native/serial accepted-
row/cursor tests in `tests/test_unit_qwen35_gguf_mtp2_seam.py` and the MoE MTP2
bundle. An oracle migration needs real scalar/native boundary and full-logit
comparisons plus unsupported topology refusal with no mutation. Production
request parity/cleanup validates wiring separately; it does not substitute for
the independent arithmetic oracle.

### R0281-TAIL-RMS — Chained norms and MoE graph diagnostic migration

`HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS` defaults on in full-stack and resident scalar
decode. The runner forwards the next layer's norm destination through the MoE
tail; selected-aggregate layers can fuse it, while slot-weighted layers keep the
feature-parallel combine followed by separate RMSNorm. Strict unfused primitives
remain required regardless of this control's eventual retirement.

The only non-unit writer is
`tests/test_live_qwen35_gguf_moe_graph_parity.py::_decode`, which disables chaining
for both eager and per-layer-MoE-graph arms. Chained norm execution does not enter
`_moe_graph_for_decode()`: its cross-layer destination is outside that graph's
stateless FFN capture boundary. Removing the tail control without migrating the
test would make its graph-engagement check fail, not qualify the graph route.
The deferred R0281-MOE-GRAPH item owns that runtime graph decision.

**Decision needed:** keep the tail control as a supported diagnostic, give the
graph comparison an explicit unchained construction policy, or retire that
per-layer graph route and its caller coherently. A defaults-only milestone does
not migrate this dependency. Do not disable production chaining or change the
slot-weighted arithmetic to make a diagnostic graph engage.

**Clearing action:** run `tests/test_unit_qwen35_gguf_moe_tail_next_rms_routing.py`
and `tests/test_unit_qwen35_gguf_iq_dispatch.py`, plus BF16/FP16 primitive fused/
unfused parity gates. The routing test pins that graph acquisition occurs only
for unchained layers. If keeping the graph comparison, run its live parity test
with real capture/replay counters after migrating its control. A reader change
also needs public decode with short/long/ragged requests, selected path,
repeat output and cleanup; independent state/logit comparison still owns any
arithmetic claim. No production default or numerical contract changes here.

### R0281-INT8-POLICY — Legacy context and layout diagnostic controls

The old INT8 diagnostics row predates capability-declared mirror-free storage,
uniform-INT8 graph capture and INT8 MTP. A declared uniform contract admits an
unmeasured artifact without the long-context opt-in, and its allocation skips
short-context mirrors. The public generator binds the declared scale dtype;
a recorded quality rejection or a named kernel-contract miss raises rather than
silently substituting BF16. Missing evidence is not a refusal. R0213 owns the
remaining mirrored-consumer migration; R0277 owns graph policy and interfaces.
The explicit `tail4_hadamard_group32` layout is also declared: FP16 group32
scales on the tail layers, BF16 prefix, oracle prefill and single-row decode.
R0088's public capacity/graph repair is recorded in
`worklog/entries/20261026T174511.632228Z-cleanup-refactor-r0088-admission-039ee3.md`.
This does not declare hybrid packed/MTP leaves or the no-oracle Hadamard path;
those are different contracts. The direct-prefill entry owns its observed
non-finite/consumer-dtype issue, and R0281-PARO-INT8-PREFILL owns PARO selection.
Do not reimplement those paths or reinstate blanket INT8 graph/MTP restrictions.

`HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG` still has two roles: explicit
capability diagnostic override and legacy long-context prefix/validation policy.
The BF16-prefix/full-layer, key-only and block16 controls still select distinct
storage layouts. Capability-admitted mirror-free sessions bypass the legacy
prefix/guard; other low-level or diagnostic sessions still consume it. Live
callers include the allocation probe, context soak, mixed-KV suite and INT8
correctness/MTP/pool-ownership tests. Removing a reader without migrating these
callers changes the storage/quality comparison they request.

**Decision needed:** retain explicit diagnostic controls, or migrate them to a
scoped policy that distinguishes a recorded-failure override from storage layout
selection. Keep actual rejection records and capability/resource errors. Do not
replace the current kernel declaration with model-name, file-hash or measured-
artifact admission lists. Historical quality/capacity numbers remain in the
sweep inventory and original artifacts; they are not current capacity limits.

**Clearing action:** enumerate both generator and resident-runner readers, migrate
`scripts/gguf_server_allocation_probe.py`, `scripts/qwen38_int8_server_context_soak.py`
and the mixed-KV/correctness callers, then remove the obsolete readers and update
ENVS. Run the KV-capability, INT8-policy, packed-verifier-layout, INT8-MTP and
per-layer-oracle unit files, including unseen artifacts and contexts below,
above and unrelated to the mirror threshold. Exercise public direct/mirrored
requests, explicit diagnostic overrides and genuine capability refusals on both
HIP backends, tracing storage/scales, graph/MTP engagement, cancellation,
refill and cleanup. Kernel-arithmetic changes require their independent quality
and state gates; control migration alone must not change arithmetic or defaults.

### R0281-Q8-EXPERIMENTS — Standalone raw-Q4/Q8_1 comparison retention

The scalar Q8_1/DS4 dot, one-/two-/four-wave WMMA, preview, packed-LDS and
expanded-LDS variants remain standalone modes of
`scripts/gguf_q4_k_t16_selected_prefill_microbench.py`. They are not public
model-runtime defaults. Their registry/build-plan, wrapper-contract,
preview-reconstruction and device CPU-reference comparisons still exist in
`tests/test_gpu_gguf_q4_k_q8_1_selected_prefill.py`. The removed T16 DS4 runtime
selector is separate from these direct kernel comparisons.

The containing library is not dead: Laguna's library owner builds it for
selected MMQ execution, and Qwen's resident runner uses its Q6 dense integer-MMQ
session. Do not delete the common library, packers, MMQ wrappers or unrelated
registrations as a shortcut to retiring the older experimental variants.
Historical negative performance rows remain in the sweep inventory and artifacts;
they do not establish the current status of every sibling implementation.

**Decision needed:** keep the old modes as explicit numerical/design controls,
or archive their performance interfaces while retaining the independent parity
references that still serve later work. An old release-window or promotion
condition does not decide which controls are useful now.

**Clearing action:** enumerate each mode's wrapper, symbol, registry key, tests
and non-test consumers before removing it. Migrate archived command replay if
needed; keep CPU-reference oracles and shared helpers with live consumers. Run
the microbench `--help`, its registry/build-plan, common-wrapper and preview
reconstruction nodes, then the affected device parity nodes if kernels change.
A wrapper/library change also needs public Laguna/Qwen MMQ route engagement and
cleanup. This disposition changes no kernel, CLI, runtime route or default and
makes no new performance claim.

### R0281-PARO-INT8-PREFILL — Oracle/streaming resource-policy ownership

`Qwen35ParoResidentSession` still implements direct retained-INT8 attention and
temporary BF16-oracle attention. Uniform `auto` requires the configured minimum
prompt length and either a total-memory limit or a free-memory comparison against
a page-rounded oracle pair plus reserve. Equality counts as pressure. Failed
meminfo reads keep the oracle route; explicit `streaming` or `oracle` overrides
the automatic choice. Tail-four `auto` retains its mixed BF16-oracle bridge,
while explicit streaming remains a separate diagnostic. BF16 storage and its
AOTriton threshold are unchanged.

The four `HIPENGINE_QWEN35_INT8_PREFILL_*` controls are therefore resource/path
policy, not a blanket INT8 feature enable. Prefill results also publish their
resolved values/path. Removing the slow direct route would remove a memory-
pressure alternative, not merely an unused performance candidate. The old speed
and capacity rows remain historical evidence in the sweep inventory.

**Decision needed:** retain operator resource overrides, or move them into the
resolved prefill/resource plan and update result metadata. A faster memory-safe
alternative is separate kernel work, not a prerequisite to exercising either
implemented route. Preserve explicit invalid-value errors, mixed-layout policy,
shared scratch/oracle lifetimes and capability fallbacks.

**Clearing action:** migrate both ordinary and packed prefill metadata/readers,
workspace-policy tests and any external diagnostic callers together; update
ENVS. Run `tests/test_unit_qwen35_prefill_workspace_policy.py`,
`tests/test_unit_qwen35_resident_batch_layout.py` and the native mixed-KV suite
contracts. Exercise public uniform and mixed requests with oracle/direct
selection, short/long/ragged prompts, resource rejection, reset/cancellation
and cleanup. Verify actual attention sources and transient oracle owners. Any
new attention arithmetic needs independent quality/state gates and matched
memory/performance measurements; this disposition changes none of them.

### R0281-SAMPLER / R0508 — Native placement rollback and algorithm controls

Native sampling defaults on for supported PARO singleton/serial-slot requests
and GGUF singleton/compatible packed requests. The old GGUF opt-in deletion
instruction predates that default and is removed. Forced queues, dynamic
constraints, unsupported combinations and unavailable native sessions still
select the host path; kernel speed does not implement missing request semantics.

`HIPENGINE_QWEN35_NATIVE_SAMPLER=0` is a live host-placement comparison in
`scripts/gguf_sampling_overhead_profile.py` and generator/API tests.
The native workspace's independent `sorted`/`strict` choice has a different
purpose: the default sorted full-vocabulary algorithm uses FP64 accumulation
of FP32 weights, not the original serial FP32 accumulation, so exact RNG/ID
parity with the strict algorithm is not implied. Model arithmetic and sampler
algorithm are independent axes. Keep registered strict primitives as numerical
oracles, the selection/provenance manifest and refusal to change an active
workspace's algorithm.

**Decision needed:** keep host-placement and strict-algorithm controls as
supported diagnostics, or migrate the environment comparison to scoped sampler
policy. Do not delete working native workspace wiring because an older tool
scenario failed, and do not broaden forced/dynamic request support implicitly.
The original A3 and hardware-transfer records remain in the sweep inventory;
this reconciliation makes no new task-quality or backend-performance claim.

**Clearing action:** migrate actual profiler, generator and capabilities-endpoint
consumers together if retiring the placement flag, preserving host/native
comparison arms and explicit fallback reasons. Run the GGUF default/prefill-owner,
PARO generator, sampler-contract and capabilities tests, plus real sampled
`LLM.generate()`/server requests with fixed-seed repeats, packed/staggered rows,
forced/dynamic fallback, cancellation/refill and cleanup. Algorithm changes also
need independent distribution and numerical-oracle gates; host/native repeat
stability does not mean their RNG streams must match. No runtime/default or
sampler arithmetic changes accompany this disposition.

### R0281-MOE-GRAPH — Scalar per-layer graph diagnostic ownership

`HIPENGINE_GGUF_MOE_GRAPH` defaults off and controls a session-bound scalar
stateless FFN cache; stateful attention remains eager. Chained next-layer norms
bypass its acquisition, so the live parity test explicitly disables chaining for
both arms and requires real captures/replays with zero rejects. R0281-TAIL-RMS
owns that diagnostic construction dependency. This is not the full-step AR,
packed decode, NativeSpecCycle or verifier graph implementation.

Whole-step capture explicitly refuses nesting this route, and automatic
whole-step selection excludes it. G5/PM4 harnesses set the flag off; they do not
compose both graphs. The generic `MoeGraphCache` also serves Qwen4Exp and has
standalone lifetime/parity/upload tests, so removing GGUF wiring does not imply
removing that reusable owner. Historical launch/wall results stay in the sweep
inventory and original artifact, not as a current bottleneck claim.

**Decision needed:** preserve this GGUF route as an explicit unchained FFN
comparison, migrate it to scoped diagnostic construction, or retire its session
wiring and live comparison together. Keep standalone graph-cache consumers and
strict eager FFN primitives regardless. Do not silently disable production norm
chaining to make the optional graph engage.

**Clearing action:** coordinate the tail-RMS comparison migration, G5/PM4 off
writers, benchmark metadata, session close and whole-step incompatibility tests.
Run tail-RMS routing, Qwen4Exp graph-control and graph-cache upload contracts;
if keeping the GGUF route, run the live MoE parity gate with actual capture,
replay/reject counters. A production reader/wiring change also needs public
short/long decode, graph/eager selection, repeat output, cancellation and cleanup.
No runtime, graph default or numerical contract changes here.

### R0281-DENSE-Q8 — Verifier sidecar placement and compatibility controls

The default-off dense-Q8 pair/all controls still have CLI writers and named
`llama-compat` suite consumers. They quantize BF16 activations once per call
and launch raw-Q8 dp4a rowtile pair, singleton or Q/K/V-triple kernels for
rows>1, falling back when layout, raw resident or workspace is unavailable.
Pair-only selection does not enable singleton/triple helpers. Historical speed
and acceptance trade-offs remain in the original sweep row and artifacts, not
as evidence for changing current production arithmetic.

`HIPENGINE_GGUF_Q8_0_RAW_SIDECAR` is a load-time residency choice. Alone it
retains raw bytes with T16 tiles only for linear-attention QKV/gate pairs;
`HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` broadens that scope to every dense Q8 T16
layer tensor only when sidecar placement is also enabled. Setting execution
flags after loading cannot create missing raw residents. The bench publishes
both placement and route controls before loading. The F32 comparison below
shares this residency. The rejected shared-expert selector is removed; normal
shared-expert projection owners remain. Shared raw-Q8 helpers and registered
kernels also support other numerical diagnostics and must not be deleted with
one BF16 selector.

The merged F32 row's `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` controls F32-input
singleton/triple helpers, including direct-state `ssm_out`, not all F32 calls.
F32 pair helpers use the base pair/all selector; their F32-output form also
requires `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS`. Source precision,
projection output precision and model arithmetic profile are distinct choices.
`--verify-dense-q8-dp4a-f32` publishes raw placement, ALL and F32 before load,
and named `f32ssm` suite routes still use it. Missing raw/workspace and rows1
keep their existing fallbacks. Preserve these comparisons unless migrating
or retiring their CLI/suite callers as part of the same unit.

**Decision needed:** keep named compatibility comparisons with explicit
load-time placement, or migrate their loose environment controls to one scoped
route policy. Retire pair-only/all variants only with their suite/CLI consumers;
do not turn a named comparison into a silent default-path rerun.

**Clearing action:** migrate loader plan/materialization, pair/single/triple
callers, bench setup/provenance and suite routes together. Keep normal shared
projections and resident-draft consumers separate. Run CPU sidecar-scope/row-domain contracts,
dense-Q8 routing, suite and bench-parser tests; a changed runtime route also
needs actual public short/long decode and target verification, logits/state
oracles, reset/cancellation and cleanup. Any changed arithmetic needs its
execution-profile numerical/task gates; removing a flag is not that gate.

### R0281-DRAFT-DENSE-Q8 / R0303 — Resident draft stage-policy migration

The resident draft runner's F32-to-q8_1/raw-Q8 dp4a route is separate from the
verifier dense-Q8 sidecar controls. Bench and profiler CLIs still set its main
toggle and stage list; named `draftdenseq8`/`draftonly` suite routes compare all
stages against proposal-only stages. Preserve those comparisons while their
callers remain. The original sweep row preserves the historical artifacts;
its unsafe direct-state timing does not establish current production parity.

A nonempty `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES` overrides the
main toggle. `draft` covers `project,qkv,attn_out,shared_gate_up,shared_down`;
`init` covers `init_project,init_kv`; explicit off disables both. The runner
builds its dp4a library and allocates q8_1 workspace from the resolved stage set.
Stage rejection, invalid geometry, absent library or insufficient workspace
keeps the original raw-Q8 F32 path. This is the older compatibility runner,
not the Generation-2 NextN provider used by normal public MTP requests.

R0303 describes this same implementation, including initialization/catch-up
projections, not a second rejected kernel family. Its historical full-suite
comparison is preserved in the pinned sweep source and unchanged July artifacts.
Keep the explicit all-stage and proposal-only comparison routes until the
stage-policy decision below migrates their CLI/result consumers together.
Single/dual/triple helper tests also bind no-launch refusal for disabled stages,
missing library/workspace, invalid row/input geometry and undersized scratch;
these are capability preconditions, not evidence-based admission rules.

**Decision needed:** keep explicit numerical comparison modes or migrate the
ambient controls to a constructor-owned stage policy, with bench/profiler and
named suite callers updated together. Do not silently turn a stage-specific
comparison into the default float path, or combine draft and verifier controls.

**Clearing action:** migrate parsing, JIT/workspace construction, initialization
and proposal callers, cache identity and result provenance together. Run parser
precedence/domain, single/dual/triple routing, upload ownership, CLI and suite
contracts; exercise changed stages through the live compatibility caller with
initial catch-up, proposal, reject/commit, reset/cancellation and cleanup. Keep
a normal public MTP request as a separate provider-preservation check. Any
changed-arithmetic default still needs its execution-profile numerical/task
gates; historical throughput is not that evidence.

### R0281-DRAFT-X8 — Resident Q6 top-1 arithmetic, layout and sidecar ownership

The compatibility generator still selects Q6 top-1 dp4a with draft shape
`x8`; bench/profiler and named `x8top1` suite routes also consume it. The
shape selector defaults to `pack8` outside those callers. It is not the
loader's `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR` control: the resident draft
uploads its own capped output-head layout through `WeightUploadCache`, keyed
by owner, runtime and vocabulary cap. `x8_dscale` additionally uploads scale
products. Raw head storage remains available for non-dp4a execution.

Shape selection currently builds these sidecars even when the independent
Q6 dp4a execution toggle is off. With execution enabled and a device top-1
output, synchronized stage timing uses stage1 plus stage2/gather; the normal
path uses the combined top-1/gather wrapper. Missing requested sidecars raise
rather than silently switching layouts. The shared wrapper thread control
also serves non-draft callers and cannot be removed with this shape choice.
The benchmark/profiler thread options and the wrapper's explicit 64/128-thread
support remain available for direct experiments and historical replay. The
rejected `x8q6-t64` and `x8q6-t64-allsync` suite aliases and fixed budgets are
removed; their recorded comparisons remain in the sweep inventory and artifacts.

`HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A` selects changed arithmetic, not a
rollback of the exact top-1/gather path. The standalone constructor keeps its
raw-Q6 default, while compatibility generation already requests q8_1/dp4a.
Migrate that activation choice and shared scheduler provenance with the layout
owner below; do not make a live compatibility head default-off based on the
standalone constructor, or remove exact/Q8_1 comparison oracles.

**Decision needed:** preserve explicit draft layout comparisons or migrate
shape, dp4a activation, scheduler selection and sidecar construction into one
owner-local plan.
Keep target materialization and the Generation-2 NextN provider separate.
This item does not authorize deleting common kernels or numerical comparisons.

R0294's rejected `x8q6-row` and `x8q6-row-allsync` suite presets are removed.
The explicit benchmark/profiler `row` shape, registered wrappers and numerical
oracles remain for geometry comparisons. With q8_1/dp4a and a top-1 output,
this shape writes one candidate per vocabulary row; synchronized timing reduces
`vocab` entries, not `vocab/8`. The combined wrapper uses that same domain.
Without dp4a, device top-1 still uses the raw-Q6 pack8 path; without a top-1
output the runner computes full BF16-input logits regardless of shape. Row
selection creates no X8 sidecar. Its historical timing result remains in the
original sweep source and unchanged July result artifacts. Include the row
comparison and its vocabulary-sized scratch in this ownership decision, rather
than removing shared reduction code or silently substituting X8.

R0295–R0298's rejected scalehoist, pack8_llama, pack16 and x8_dscale suite
presets are also removed. Explicit benchmark/profiler shapes and registered
numerical comparisons remain. Scalehoist and pack8_llama read the raw head and
reduce `vocab/8` candidates; pack16 reduces `vocab/16` and requires vocabulary
divisible by 16 even when dp4a is disabled. X8 dscale uses the capped X8 head
plus its separately owned FP32 scale-product sidecar. All shapes keep the same
raw BF16 top-1/full-logit behavior when their dp4a/device-top-1 preconditions
are absent. Preserve staged and combined wrappers, sidecar failure behavior,
shared scheduler choice and result provenance in this plan migration. The
historical shape comparisons remain in the pinned sweep source and unchanged
July artifacts; their negative timing rows do not authorize removing the
explicit comparison capability.

**Clearing action:** migrate draft construction, upload-cache names, top-1
stage/combined callers, bench/profiler result fields and suite aliases together.
Run capped-layout round-trip and upload ownership tests, exact q8_1/Q6 top-1
oracles, staged/combined gather routing, raw-head fallback and parser/suite
contracts. Any changed route needs real compatibility proposal/commit/reset
and teardown checks plus separate public NextN preservation; changed arithmetic
needs the execution-profile numerical/task gate, not the old timing row.

### R0313 / R0314 / R0315 / R0316 / R0318 / R0320 — Laguna legacy MMQ selector migration

`mmq128x32_d8_f32_rowvec`, `mmq64x32_d4_f32_rowvec`,
`mmq128x32_d8_f32_wavecols`, `mmq64x32_d4_f32_wavecols_q4` and
`mmq64x32_d4_f32_wavecols_direct_q4` plus gate/up
`mmq128x32_d8_f32_wavecols_direct_doublebuf` are explicit comparison modes,
not defaults. Current package defaults select later
wave-column modes; the low-level `rowvec` staging primitives also serve those
later modes and Q6 comparisons, so they are not removable with the old names.
The pair-decode and direct peers share row-vector/wave-column staging and one
residual pass; their direct-consume flag differs. The real host caller preserves
lane count, stream/runtime and fused shared-combine output ownership for both.
The Q6 row32/row64 comparison preserves those same caller contracts. Its GPU
fixture separately packs interleaved qmicro bytes for row32; the current
planar-qmicro resident is not interchangeable with that reference.
The double-buffer gate/up peer changes only the double-buffer flag relative
to direct consumption in the tested caller; common staging, residual pass,
lane/stream/runtime and combine destination remain unchanged.

`scripts/laguna_gate_doublebuf_ab.py` is no longer a one-purpose harness.
Its `--mode LABEL=RUNTIME_MODE` arms also select down modes, Q6 WMMA prefetch
and Q4/Q6 precomputed sums. Preserve those comparisons, mode ordering,
complete-state digest capture and child cleanup when migrating its default
pair. Do not delete the generalized harness or shared double-buffer primitive
with the old session name.

`tests/test_live_laguna_moe_gpu.py` still uses the gate/up mode across the full
Q4/Q6 comparison chain and directly compares row-vector/pair-decode down
against their peers. `scripts/laguna_prefill_production_publication.py`
recognizes the old gate/up and down names when normalizing historical quality
metadata. Preserve that historical reader even after runtime retirement.
Original July timing/quality
results remain in the pinned sweep source and immutable worklogs; they are not
new measurements from this pass.

**Blocker:** selector retirement must migrate those numerical comparisons to
explicit primitive ownership and validate the user-facing route. The configured
`~/models/gguf/laguna-s-2.1-Q4_K_M.gguf` artifact and the live fixture's
`/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf` are unavailable on this host. This is
a local validation-resource blocker, not an admission restriction or a reason
to disable the existing default.

**Clearing action:** provide an implemented Laguna model artifact, remove the
legacy names from resolver/fusion/config tables, migrate the generalized
comparison harness's default pair, and preserve the low-level row-vector and
double-buffer references in the Q4/Q6 GPU fixture. Run selected-MMQ, runner,
publication and grouped-prefill contracts plus the full MoE quant-oracle nodes;
then exercise `LLM.generate()` at short, boundary-neighbor and unrelated prompt
sizes, recording selected defaults, output parity and cleanup. Keep historical
publication normalization and the separate Q6 row-vector implementation.

### R0319 — Laguna serial compaction comparison ownership

Both gfx11 packages default to parallel stable MoE compaction. Explicit
`serial` remains a live comparison in
`scripts/laguna_group_compact_parallel_ab.py` and the Q4/Q6 GPU fixture;
registered serial primitives also remain the reference path. This is not an
unmeasured-backend default. The host gate/up caller selects serial or parallel
source-row compaction while preserving metadata pointers, lane dimensions,
stream/runtime and subsequent tile-map/quant/gate/SiLU ordering. The A/B harness
closes its child session after success or a prefill exception.

**Remaining work:** migrate session-level comparisons to explicit primitive
ownership before removing the `serial` override. Keep registered reference
primitives and historical result readers. Original July promotion evidence is
in the pinned sweep source and immutable worklogs, not a new result here.
The local Laguna model-resource blocker and public-route validation described
in R0313–R0318 also apply to this migration.

**Clearing action:** migrate the A/B harness, numerical fixtures and session
setter together; run compaction comparison, source-default, runner and full
MoE numerical contracts, then public `LLM.generate()` with default selection,
metadata/output equality and cleanup checks. Do not silently route a requested
serial comparison through parallel after retiring the selector.

### R0375/R0376 — gfx1151 pack8 comparison-control ownership

The cumulative semantic packet still selects three distinct chains: tile8x8
pack8, singleton pack8 WMMA plus separate SiLU, and operation-complete dual
WMMA/SiLU. `HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK` and
`HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL` remain enabled by default;
their opt-outs drive those explicit comparisons. Registered primitive
fallbacks and backend shape policies are unchanged. Historical timing and
numerical results remain in the original sweep source and August 15 artifacts.

The threshold sweep now explicitly disables the newer fused pair in older
control roles and removes inherited overrides for `current`. Disabling only
singleton WMMA did not prevent the later fused owner from running. Each
execution records its effective role environment, and synthetic child tests
also evaluate the real fused-dispatch predicate.

**Decision needed:** migrate active cumulative/threshold/probe consumers to
explicit diagnostic policy before deleting either environment control. Keep
tile8x8, singleton WMMA and unfused SiLU as low-level numerical references.
Do not change backend admission or arithmetic as a side effect of retiring
comparison controls; broader shape selection is a separate capability task.

**Clearing action:** run `uv run pytest
 tests/test_unit_qwen35_08b_threshold_environment.py
 tests/test_unit_qwen35_08b_cumulative_semantic.py
 tests/test_unit_qwen35_08b_threshold_artifact.py
 tests/test_unit_gguf_linear_dispatch.py`, then exercise the migrated policies
through the cumulative packet and public `LLM.generate()` on HIP. Check
current/older-role engagement, single/paired/fused dispatch, 511/512/513 and
unrelated lengths, full logits/state, repeated role order and cleanup before
removing the env readers.

### R0373/R0374 — Maple fused decode efficiency and selector retirement

`HIPENGINE_MAPLE_FUSE_MOE=1` selects fused dual gate/up plus clamped SiLU;
`HIPENGINE_MAPLE_FUSE_QKATTN=1` selects fused Q/K normalization, rotary/KV
write and attention. Both default to `0` because the original M2 comparisons
observed kernel/whole-step efficiency regressions, not because of missing
qualification. The strict unfused chains and their numerical tests remain.
Historical measurements and the rejected serial-expert down/residual design
are recorded in `docs/campaigns/MAPLE-PERF.md` and the immutable sweep source;
this cleanup makes no new performance claim.

Both flags are sampled once per `MapleRunner.step()`. Graph identity now
includes those sampled choices: a changed tuple releases the old executable
and graph, clears rejection state and captures the requested chain. Unchanged
choices replay the existing graph. Previously the cache ignored its key and
silently replayed the old chain after a between-call selector change.

**Decision needed:** remove the experimental runtime fusion routes while
preserving primitive references, or profile and repair the efficiency gap
before promoting a route and retiring its selector. Treat the two fusions
independently. Do not delete the mandatory unfused primitive chains, and do
not reintroduce the serial-expert down/residual design unchanged.

**Clearing action:** run `uv run pytest
 tests/test_unit_maple_graph_selection.py tests/test_gpu_maple_runtime.py
 tests/test_gpu_maple_ternary_kernels.py
 tests/test_gpu_maple_attention_kernels.py`, then use the Maple profiling and
c1 benchmark tools for same-host eager/graph comparisons. Include both
individual selector changes and their combination, actual public
`LLM.generate()` engagement, full-logit/state parity for strict variants,
reset/reuse and graph/workspace teardown. A replacement arithmetic variant
uses its execution-profile gate rather than inheriting strict bit parity.

### R0371 — Moonshine top-1 and unfused LM-head route ownership

`MoonshineResidentRuntime` defaults to `wave8_top1`, which writes the full
FP16 logit plane and bounded partial-value/index scratch while selecting
stable top-1. `wave8_argmax` remains an explicit full-logit projection plus
argmax path and is required for selective W8A16 LM-head execution. The
constructor rejects W8A16 LM heads with `wave8_top1` by naming its FP16
requirement; removing the fallback selector would remove that runnable
quantized path, not merely a bisection option.

The decoder smoke and route gate still compare both names, including eager
and graph execution. Host/runtime tests cover default and explicit routes,
actual token-DAG selection, scratch, full logits, stable tie contract,
invalid-route errors, W8A16 compatibility and cleanup. No runtime or kernel
change is needed for this disposition.

**Decision needed:** keep the explicit route selector or replace it with
caller-owned capability selection plus an explicit numerical reference
interface. Preserve the unfused projection/argmax implementations and the
W8A16 route; do not silently downgrade an explicit `wave8_top1` request.
Historical leaf/checkpoint/audio evidence remains in the original sweep
source and `benchmarks/results/2026-08-08-gfx1151-moonshine-lm-head-*`
artifacts; it is not a fresh measurement.

**Clearing action:** migrate constructor, decoder smoke, leaf benchmark and
route gate together. Run `uv run pytest
 tests/test_unit_moonshine_lm_head_reference.py
 tests/test_unit_moonshine_lm_head_route_gate.py
 tests/test_unit_moonshine_lm_head_bench.py
 tests/test_gpu_moonshine_runtime.py
 tests/test_gpu_moonshine_hip_projection.py`, then the real-audio route gate
and checkpoint eager/graph comparisons on HIP. Include FP16/default,
explicit unfused and selective W8A16 requests, full logits/top-1, graph
buckets, no timed allocation, repeat/reset/reuse and teardown before removing
or replacing the selector.

### R0370 — Moonshine exact two-region scheduler reference ownership

`MoonshineCudaExactContinuousBatchRuntime` remains an explicit t32/t256
arithmetic reference. `scripts/benchmark_moonshine_cuda_continuous.py
 --topology exact-two-region` selects it; omission selects uniform-t256.
The two-region scheduler owns separate early/mature graph caches and
transfers a request once at their boundary. Dynamic-arrival tests exercise
this behavior. Static lockstep is a separate comparison, not an automatic
replacement for the continuous reference.

The benchmark now owns each decoder immediately after construction and
closes scheduler graphs before decoder workspaces. This also releases a
first decoder if the second constructor, kernel preparation or scheduler
construction fails. Lockstep preparation is inside its cleanup scope.
Synthetic full-command tests cover both topologies, setup/body/close
failures and successful reports without CUDA. Scheduler arithmetic, uniform
defaults and runtime ownership APIs are unchanged.

**Decision needed:** keep the explicitly requested continuous exact reference,
or retire it after confirming no caller needs its hidden-boundary arithmetic
and migrating its benchmark/GPU-test consumers. Do not silently substitute
uniform-t256 for `--topology exact-two-region`. Historical corpus and timing
results remain in the original sweep source and Moonshine artifacts; no fresh
quality or performance claim is made.

**Clearing action:** audit the class, benchmark selector and dynamic-arrival
GPU tests together. Run `uv run pytest
 tests/test_unit_moonshine_continuous_setup_cleanup.py
 tests/test_unit_benchmark_moonshine_cuda_continuous.py
 tests/test_gpu_moonshine_cuda_sm120a_continuous_batch.py`, then both explicit
benchmark topologies on a CUDA host with the same fixtures and static lockstep
comparison. Check admission, one-time transfer, cancellation, compaction,
bounded graph cache, EOS, reset/reuse and teardown before deleting the route.
Public Moonshine admission is a separate architecture decision, not part of
this benchmark cleanup.

### R0368 — LC-D3 global geometry diagnostics and unused repair separation

The global leaf still compares D64/V64, D64/V32, local256 D32/V64,
direct-qhead D32 and score-tile4 against their named parents. D32/D64
4,096-token context-split wrappers and registrations are also live; the
production cache owner composes D64, compensation, score and prefetch policy
from backend capabilities. Preserve the generic complete-`KVLiveSpans`
route, exact normalized/deferred routes and compensated/uncompensated
production paths. The original entry's blanket prototype list is not a
safe deletion list.

The leaf permits diagnostic drift for D32 context splitting versus the
chronological exact parent. Its D64 comparison uses the D32 context-split
parent, not that exact parent, and keeps their equality check. Host tests
bind these distinct comparison contracts, wrapper symbols and all span
pointers across several live lengths. The shared leaf now frees partial
allocations and timing events, including when cache closure raises.

`laguna_global_attention_split_gqa6_ctx4096_pv_repair_kernel` has no launch
caller. The shared merge kernel still computes and writes its mask, and the
leaf downloads mask diagnostics for D32 context splitting. Remove that
unused repair machinery as one scoped kernel change rather than leaving
stale result fields or removing live production merges. The old 8,192-token
experiment has no current named wrapper or leaf selector. Historical rejected
experiments and their results remain in the sweep's pinned source and the
August LC-D3 artifacts; they do not describe current dispatch.

**Decision needed:** retain the explicit geometry comparisons or move them
to a diagnostic build before deleting their symbols. Separate dead repair
code and mask reporting from the shared production merge without changing
partial-output storage or arithmetic. This host lacks `libamdhip64.so`, so
this unit does not edit shared GPU code or claim numerical verification.

**Clearing action:** inventory wrapper, registration, leaf and GPU-test
consumers before deleting geometry symbols. For repair removal, test both
compensation modes and normalized/deferred merges, then run
`uv run pytest tests/test_unit_laguna_global_geometry_references.py
 tests/test_unit_laguna_global_issue_comparison.py
 tests/test_gpu_laguna_kv_attention.py`. Run the global leaf on both sides
of the split/policy boundaries and unrelated live lengths, then public
Laguna generation with route attribution, teacher logits, KV metadata,
reset/reuse and teardown. Preserve broader production numerical contracts;
strict-parent equality is not a replacement for the profile gate. No default,
kernel or arithmetic changes are made by this disposition.

### R0367 — Selected gate/up halfdot and exact-teacher ownership

`use_selected_halfdot_decode` remains a live constructor rollback, not the
retired session setter or profiler comparison. The gfx1151 package defaults
it on; gfx1100 defaults it off. Paired expert residents use halfdot only for
one input row. The exact interleaved route handles wider inputs and the
explicit false override. Ordinary expert residents keep their own routes;
the halfdot switch does not change layout or allocations.

`scripts/laguna_gguf_correctness.py --selected-halfdot` creates a halfdot
session, a repeat session and a false-override exact teacher sharing its
resident weights. Removing that override would erase the teacher comparison.
The natural-decode leaf also compares exact interleaved and halfdot symbols
on identical paired bytes. Host tests cover backend policy, explicit values,
single/multirow route choice, operand sharing and distinct leaf controls.
Partial allocation, upload and timing-event failures now release their owned
resources. Production routing and arithmetic are unchanged.

**Decision needed:** keep the constructor rollback and diagnostic selectors,
or migrate their exact-teacher and fallback ownership into an explicit
kernel plan. Preserve both registered implementations. This is not a request
to disable the default halfdot path or re-promote its arithmetic.

**Clearing action:** migrate the correctness script's candidate/repeat/exact
sessions and leaf selectors together, then run
`uv run pytest tests/test_unit_laguna_halfdot_reference.py
 tests/test_unit_laguna_scalar_q_retirement.py
 tests/test_unit_laguna_tile8_comparison.py
 tests/test_unit_laguna_weighted_hidden_split_runtime.py
 tests/test_unit_laguna_gguf_runner.py
 tests/test_gpu_gguf_t16_selected_gemv_decode.py`. On a HIP host with the
model/oracle, run `uv run python scripts/laguna_gguf_correctness.py
 --backend hip_gfx1151 --selected-halfdot` and compare a public generation
request under default and explicit exact policy, recording actual routes,
aligned teacher logits, reset/reuse and teardown. Do not remove the wider-row
exact fallback. Historical performance/quality evidence remains in the sweep
source and the July 31 result artifacts; no fresh numerical claim is made.

### R0354 — Paired expert cache and ordinary-layout migration

The default expert materializer already replaces two Q4 T16 gate/up allocations
with one `tiles_dual` owner on the gate weight and an empty up-weight allocation
map. It does not retain ordinary expert allocations beside the pair. Host tests
bind exact uploaded bytes, one upload, combined residency and teardown across
raw, fully cached and either mixed-source combination, including allocation,
upload and either profile-callback failure.

The profile previously labelled a mixed cached/raw pair entirely
`gguf_dual_interleaved`. It now reports
`repacked_cache+gguf_dual_interleaved`; fully cached/raw labels stay unchanged.
This reporting fix changes neither resident layout nor numerical execution.

**Remaining work:** the `laguna-repacked-v1` writer still persists independent
`tiles` payloads. `_materialize_q4_t16_dual_pair` must read and interleave those
at load time. Add a versioned paired cache payload that replaces both stored
matrices, with no duplicate pair, while retaining raw/partial cache loading.
Migrate the leaf scripts that directly read manifest `entries` and ordinary
`tiles` before retiring legacy cache acceptance or the public
`use_q4_expert_t16_dual_interleaved=False` / `--ordinary-q4-expert-t16` controls.
The natural-tile8 comparison also intentionally requests ordinary residents;
its replacement must preserve that distinct reference route (R0341).

**Clearing action:** bind writer/open/reload and direct-manifest consumers to
paired ownership, validate both cache versions and raw/partial-source loads,
then run paired/ordinary primitive oracles and public `LLM.generate()` decode
and prefill with byte/state/resource checks. Remove the ordinary seam only
after its callers migrate. Local ROCm/Laguna resources for the real numerical
and public runs remain unavailable as recorded by R0340/R0273. Historical
July measurements remain in the sweep's pinned source and artifacts, not
fresh validation of the cache migration.

### R0353 — SWA value-tail reference ownership

The standalone mixed40 value-tail primitive remains a live leaf comparison.
Its local512 and V-stage128 siblings are controls for later compound
comparisons, and the shared kernel template still implements producer-wave
staged-value tail copies. Deleting the wrapper/registry/leaf choices would
break these references; the July conditional removal instruction is stale.
Default production uses the later output-sharded/dense-ring route, not the
standalone reference. No production default or kernel changes here.

Host tests bind the four named leaf pairs, distinct wrapper symbols, spans and
shape rejection. They also exposed leaf resource leaks: buffers were recorded
only after every allocation succeeded, and timing created its second event
before entering cleanup. The leaf now records each allocation immediately,
releases the first event if second-event creation fails, and attempts outer
cleanup after cache or second-event destruction errors. Synthetic equality and
timing outputs in these tests are control-flow fixtures, not numerical or
performance evidence.

**Remaining work:** separate diagnostic-only instantiations/wrappers from the
production build without removing the shared template or live predecessor
oracles. Keep the generalized leaf harness and its parser/result readers until
replacement comparisons select the same symbols. Historical standalone
rejection remains in the pinned sweep source/artifacts, not a reason to disable
compound implementations.

**Clearing action:** run value-tail reference, SWA issue and KV attention host
contracts. On ROCm, execute standalone/local512/V-stage128/dual-tail leaf pairs
and primitive oracles, including allocation/event failure cleanup, then public
`LLM.generate()` confirming the default compound route and full state/resource
parity. Local ROCm/Laguna resources remain unavailable as recorded by
R0340/R0273; these real numerical/public checks remain open.

### R0349–R0351 — Global exp/mixed32 diagnostic migration

The three profiler comparisons cover direct versus assumed-domain GQA2,
serial versus wave-wide exponent issue, and GQA2 versus mixed32 ownership.
All three session setter pairs selected the newer mixed40/local1024 symbol
in both arms with package defaults; real KV route tests reproduced the defect.

`scripts/laguna_global_issue_diagnostic.py` now scopes each requested pair:
predecessors are enabled, producer/probability/local512/mixed40 successors
are disabled, and every touched field is restored before session close,
including nested scopes and failed bodies. The profiler records
`global_issue_comparison_basis`. Missing gated split attention fails before
mutation. Production selection, package defaults, kernels and arithmetic are
unchanged.

Host tests bind registered pairs at representative live counts and both sides
of the split and live4000 boundaries, plus allocator teardown and each CLI
option's scope/exception cleanup. July numerical and speed evidence stays in
the sweep's pinned source and artifacts; it does not validate the repaired
comparison. The rejected ordinary-grid and split-launch experiments were
already removed and require no implementation change here.

**Remaining work:** move the public setters/cache policy ladder and profiler
mode/result readers into the diagnostic owner, preserving generic and
peer-backend routes. Remove the scoped field policy when reference variant
selection no longer mutates production flags. Real numerical/public validation
shares R0340/R0273's missing local ROCm runtime and Laguna artifact.

**Clearing action:** run global issue, profiler, runner and KV attention
contracts; on ROCm, exercise each pair against its registered primitive oracle
on both sides of split/live4000 and capacity-4096 boundaries. Then run public
`LLM.generate()` with the default successor selected, checking complete state,
selected symbols and resource teardown before retiring controls or moving
reference primitives out of production builds.

### R0343–R0348 — SWA issue-schedule diagnostic migration

The six profiler comparisons cover direct versus assumed-exp GQA3, GQA3
versus mixed32, and the mixed32 serial/exp4/exp8/exp16/exp32 issue schedules.
Their session setters alone no longer select those pairs: current mixed40
successors override them. Real KV route tests reproduced the same mixed40
symbol in both arms of every comparison.

`scripts/laguna_swa_issue_diagnostic.py` now scopes each comparison's policy:
required predecessors are enabled, later issue/producer/mixed40 successors
are disabled, and every touched field is restored before session close,
including nested scopes and failed bodies. The profiler enters this scope
only for the six named comparisons and records `swa_issue_comparison_basis`.
An incompatible split/tile16/gate/wave-local policy raises a named error.
No production runtime, package defaults, kernels or arithmetic change.

Host checks execute actual allocation, serial position publication, setters
and KV route selection with an inert runtime, covering unsaturated, saturated
and wrapped rings plus teardown. Historical July measurements remain in the
sweep's pinned source and artifacts, not evidence for the repaired comparison.

**Remaining work:** consolidate the public setters/cache ladder and profiler
mode/result readers into the diagnostic owner, preserving generic shape and
peer-backend routes. Remove the temporary scoped field policy once reference
variant selection no longer depends on mutating production cache flags.
Actual-weight/GPU/public validation shares R0340/R0273's missing local ROCm
runtime and Laguna artifact.

**Clearing action:** run SWA issue-scope, KV attention, runner and profiler
contracts. On ROCm, execute each repaired pair against its registered primitive
oracle, including unsaturated/saturated/wrapped rings and resource cleanup;
then public `LLM.generate()` with the default successor route. Confirm symbols,
full-state parity and teardown before retiring the six public controls or
moving reference-only primitives out of production builds.

### R0341 — Laguna natural tile8 diagnostic ownership

The session's natural tile8 setter remains a live profiler control, but the
newer dual-interleaved resident gate/up route takes precedence over it. On
ordinary T16 bytes, parallel and fused-SiLU successors also override the
original tile8 body. The old profiler comparison therefore did not isolate
natural tile16 versus tile8 under package defaults.

`--compare-selected-natural-tile8-decode` now requests ordinary expert T16
residency, enables natural decoding, and holds parallel/fused-SiLU overlays
off in both arms. The existing setter then reaches distinct natural16/tile8
routes. JSON declares
`ordinary_t16_natural16_vs_tile8_without_parallel_silu` as the comparison
basis. Unset profiling and production loading retain package defaults;
multi-row ordinary dispatch keeps its registered route. Historical July
measurements remain in the sweep's pinned source and artifacts, not evidence
for the repaired comparison.

Host CLI-to-constructor and real route-selection checks cover both arms,
unset defaults and multi-row fallback. No runtime, kernel, arithmetic or
production residency changes are made.

**Remaining work:** migrate constructor/setter and profiler comparison to a
scoped diagnostic owner before retiring the public tile8 control. Keep
ordinary natural16/tile8 reference bodies and dual-layout dispatch distinct.
Actual-weight/public validation is blocked locally by the unavailable ROCm
runtime and Laguna artifact recorded in R0340/R0273.

**Clearing action:** run tile8-comparison, runner, profiler and MoE numerical
contracts; execute the repaired paired comparison on a ROCm host with an
available Laguna artifact. Confirm distinct selected symbols, full-state
parity and cleanup, then public `LLM.generate()` on default dual residency
and ordinary diagnostic residency. Migrate result readers with the diagnostic
owner; do not silently compare the same interleaved route twice.

### R0340 — Source-F16 decode comparison migration and shape validation

Automatic single/triple decode previously selected fixed-K from backend policy
without checking K or launch width. Real dispatch/wrapper host tests reproduced
rejection of supported generic-K inputs and 64-thread launches. Automatic
selection now limits fixed-K to K3072/K6144/K9216 with 256 threads, uses the
backend's generic-K one-barrier route for other K widths, and keeps ordinary
GEMV for other launch widths. Explicit `fixedk`/`onebarrier` requests preserve
named primitive incompatibility errors. Prefill, F32 activation, peer backend
policy and all kernels/arithmetic are unchanged.

`HIPENGINE_LAGUNA_F16_DECODE` still owns live long-context profiler comparisons
between GEMV, one-barrier and fixed-K. Its complete values and automatic shape
policy are documented in `docs/ENVS.md`. Historical July gates remain in the
sweep's pinned source and artifacts; their release-window conditions do not
replace migration of those callers.

**Remaining work:** move process-global comparison selection into a scoped
source-F16 diagnostic owner and migrate profiler environment restoration and
result readers before retiring the selector. Real GPU/public validation of the
shape repair is still outstanding: this host cannot load `libamdhip64.so`,
and the local Laguna artifact is unavailable as recorded in R0273.

**Clearing action:** run shape-policy, F16 projection, backend, runner and
profiler tests. On a ROCm host, execute generic-K widths below/above/unrelated
to the fixed set, single/triple and alternate launch widths against the GEMV
oracle. Run public `LLM.generate()` with an available artifact and verify the
intended variant, output/continuation and clean teardown. Preserve generic
registered owners and explicit capability failures during selector migration.

### R0336 — Laguna packed-output comparison migration

The enabled gfx1151 output gate consumes contiguous head-major M128 tiles and
handles generic prefix/suffix rows. Generic output/unpack kernels remain
required when no packed tiles run. BF16 and FP16-via-BF16 gates are distinct
output boundaries; neither is retired by removing a comparison control.

The dedicated `scripts/laguna_attention_packed_output_gate_ab.py` still owns
the resident constructor override. Its former generic-producer setting made
the candidate a no-op: the runtime only leaves output packed when its query
is packed. Real `_run_layer_rows()` tests reproduced this for both gate
precisions and mixed/complete tiles. Both arms now hold the packed-query
producer on and vary only the output gate. JSON records the changed basis as
`packed_query_producer_on_output_gate_only`; old generic-producer measurements
are not evidence for this repaired comparison. R0274's description is corrected.

Host checks cover generic-only rows, packed intervals with generic prefix and
ragged suffix, pointers/streams, distinct unpack/gate routes, full-state capture,
child cleanup and package restoration after constructor failure. Production
runtime, defaults and kernels are unchanged. Historical July publication/trace
rows remain in the sweep's pinned source and benchmark artifacts.

**Remaining work:** migrate the dedicated comparison to an explicit diagnostic
owner before retiring the constructor override and script. Preserve generic
unpack and gate primitives. The local Laguna artifact resource blocker in
R0273/R0274 prevents actual-weight/public validation here, not execution by a
user with an available artifact.

**Clearing action:** run packed-output, producer-retirement, runner and long
context profile host checks; execute the repaired paired comparison with an
available Laguna artifact and confirm distinct packed/generic gate routes,
full-state equality and cleanup. Exercise public `LLM.generate()` with varied
lengths including partial tiles. Then migrate the diagnostic owner and retire
its redundant public selector without changing backend defaults.

### R0333–R0335 — Laguna Q4 P8 and sum comparison ownership

The gfx1151 defaults select Q4 P8 gate/up and down at producer chunks of at
least 512 rows. Smaller chunks keep non-prefetch bodies. The enabled Q4 sum
policy writes a bounded K16 `int16` sidecar only for P8 gate/up; its consumer
receives that same pointer. Down uses no Q4 sum sidecar. R0313–R0320 own the
shared gate/down mode migration.

The P8/non-P8 primitives, mode strings and sum setter have live owners:
`scripts/laguna_q4_k_mmq_leaf_bench.py`, the generalized runtime comparison,
and `tests/test_live_laguna_moe_gpu.py`. These are not one-purpose harnesses.
Host tests in `tests/test_unit_laguna_q4_p8_policy.py` bind dispatch below,
at and above the threshold plus unrelated row counts, real caller/wrapper
sidecar pointers and symbols, fused/unfused down packing, and sum setter
isolation from Q6. Production defaults, scratch, arithmetic and kernels are
unchanged. The July leaf, publication and trace evidence remains recoverable
from the sweep's pinned source and benchmark artifacts; its campaign-name
conditions do not describe the remaining migration.

**Remaining work:** separate production shape policy from explicit primitive
references across resident constructor/setters, generalized comparison,
leaf tooling and live numerical fixtures. Retain non-P8 bodies for sub-512
chunks, and keep sidecar allocation/producer/consumer ownership together.
Actual-weight and public validation share the local Laguna artifact resource
blocker recorded in R0313–R0320; no runtime admission rule is added.

**Clearing action:** run Q4 P8 policy, selected-MMQ-mode, generalized-comparison,
runner and full MoE numerical contracts after the coordinated migration. Run
the generalized P8/non-P8 and sums-off/on comparisons with an available Laguna
artifact, then public `LLM.generate()` across varied producer chunk sizes.
Confirm intended shape/default selection, exact full state and cleanup before
retiring positive mode strings or the public sum setter. Do not delete the
shared leaf harness or silently map an explicit reference request to P8.

### R0331 — Laguna Q6 prefetch/sum comparison ownership

`q6_wmma_prefetch_weight`, `q6_wmma_prefetch_activation` and
`q6_precomputed_activation_sums` remain live constructor/setter controls in the
generalized runtime A/B, qmicro leaf and layout/staging comparisons. gfx1151
enables all three on its planar route. Later features require earlier ones;
precomputed sums must agree between activation producer and down consumer.
R0330 owns the shared integer-WMMA/hoist primitive-reference migration.

The generalized A/B's `q6_wmma_current` and `q6_wmma_prefetch` arms previously
left sums enabled when disabling activation prefetch. Real setter-state tests
reproduce those inconsistent states; both arms now disable sums first. The
four Q6 arms select distinct consistent weight/activation/sum states from
both enabled and disabled starting policies. Real caller/wrapper host tests
bind half-sum production and matching consumer symbol, pointers and streams.
Runtime setters, backend defaults, arithmetic and kernels are unchanged.
July publication/trace rows remain historical evidence in the sweep source and
artifacts; the old campaign-name condition no longer describes the blocker.

**Remaining work:** replace public controls with explicit production/reference
ownership across constructor, setters, generalized A/B/leaf and numerical
fixtures. Preserve layout/staging harness prerequisites and result readers.
Actual-weight/public validation shares the local Laguna model resource blocker
recorded in R0313–R0320, not a missing-measurement admission rule.

**Clearing action:** run runtime-mode, WMMA-reference, activation/layout,
runner and full MoE numerical contracts; then execute the generalized comparison
and public `LLM.generate()` with an available Laguna artifact. Confirm matching
sum producer/consumer, route/default selection, exact full state and cleanup
before retiring the three public controls. Keep non-prefetch reference bodies
needed by other declared shapes and comparisons.

### R0330 — Laguna Q6 integer-WMMA reference ownership

`integer_wmma` and `wmma_hoist_activation` are primitive wrapper controls, not
session flags. Unset values select integer WMMA for planar qmicro and then
activation hoist; explicit false values still select distinct packed-dot and
unhoisted references. Later prefetch/sum specializations depend on this chain.
The wrapper rejects incompatible combinations before compiler/runtime access.

`scripts/laguna_q6_qmicro_leaf.py` is a generalized comparison, not a one-purpose
integer-WMMA harness: it owns legacy/interleaved/planar residents, ordinary/
compact/half-row/padded staging, permute, WMMA, hoist, both prefetch stages and
precomputed sums, with separate outputs and mismatch checks. Do not delete it
with these two selector names. Host tests bind current default/reference symbols,
raw pointer/stream ownership across 63/64/65/193 rows and prerequisite failures.
Historical publication/trace evidence remains in the pinned sweep source and
original artifacts, not an outstanding checkpoint or a new measurement here.

**Remaining work:** separate production specialization from explicit reference
ownership across the shared wrapper and generalized leaf before collapsing
controls. Preserve packed-dot/non-planar bodies and the numerical comparisons.
Actual-weight/full-model validation shares the local Laguna artifact resource
blocker recorded in R0313–R0320.

**Clearing action:** migrate reference entry points and leaf callers together;
run WMMA reference-control, activation/layout comparison and full Q6 primitive
oracles, then the actual-weight generalized leaf and public `LLM.generate()`.
Confirm default symbol, byte-layout agreement, exact outputs and cleanup.
Changed arithmetic requires its execution-profile gate; preserve historical
result readers when retiring mode labels.

### R0329 — Laguna source-F16 boundary comparison ownership

`set_f16_boundary_fusion(...)` still owns the live separate/fused arms of
`scripts/laguna_f16_boundary_ab.py`. gfx1151 enables the capability. The runner
uses the fused RMSNorm/gate producers only in `hipblaslt_range_direct`; scaled
projection modes keep their scale/cast chain regardless of this setter. The
fused producer preserves the BF16 rounding boundary before FP16 consumption.
Registered unfused primitives remain required reference/fallback paths.

Host tests bind real projection/output cast selection, source/destination
pointers, stream/runtime ownership and the live comparison's state capture and
cleanup after setter or prefill failure. They exercise short and non-tile
row counts, not only the historical M512 point. July numerical/trace results
remain in the pinned sweep source and immutable artifacts; the old later-
checkpoint wording is not a current removal condition.

**Blocker:** removing the setter also requires explicit primitive/composite
ownership for the comparison. The local Laguna artifact resource blocker in
R0313–R0320 prevents public/full-model validation; it does not disable the path.

**Clearing action:** migrate the comparison without collapsing its arms, run
F16 boundary-comparison, runner and hipBLASLt contracts plus primitive
BF16-boundary equality/full-model oracles, then public `LLM.generate()` across
compatible and fallback projection routes. Confirm selected producers/casts,
state equality and teardown. Keep the enabled capability and strict unfused
chain when retiring the session setter.

### R0327 / R0328 — Laguna permute/planar comparison migration

`q6_qmicro_permute` remains an interleaved-record consumer, not the gfx1151
default: gfx1151 enables planar qmicro and leaves permute off. Both selectors
still serve the live planar A/B and long-context profiler, and the numerical
fixture explicitly compares separately packed interleaved/planar residents.
Preserve matching resident bytes and consumers across prefill/direct decode/
grouped-small-M; R0322 owns the common layout/cache boundary.

The planar A/B now holds later WMMA-prefetch and precomputed sums off for both
arms and records `comparison_policy`. Previously its interleaved arm failed
construction under inherited planar-only prefetch settings. The profiler now
maps an explicit permute request to interleaved layout when layout is unset,
and disables planar-only prefetch for explicit interleaved requests. Unset
requests still use package defaults; explicitly conflicting planar+permute
requests still fail rather than silently downgrade. Profile results record the
layout overrides. Historical mode labels and July evidence remain unchanged.

**Remaining work:** retire the session/profile selectors only with these live
comparison callers and their explicit primitive/resident ownership. The local
Laguna model-resource blocker in R0313–R0320 prevents actual full-model/public
validation; missing measurements are not a reason to disable existing paths.

**Clearing action:** run planar-comparison, activation-comparison,
resident-layout, profiler and runner contracts plus full Q6 numerical oracles;
then run both owner orders of the A/B and a public `LLM.generate()` check with
an available Laguna artifact. Record layout/consumer/default selection, exact
state and cleanup. Keep legacy/interleaved consumers and cache conversion where
required, and preserve historical result readers when removing old labels.

### R0324 / R0325 / R0326 — Laguna Q6 activation comparison migration

`q6_compact_activation`, `q6_half_row_activation` and
`q6_skip_padded_activation` still own the three live staging comparisons in
`scripts/laguna_q6_*activation_ab.py`. The current planar consumer requires
all three; it cannot run the earlier ordinary/full-row/padded peers against
planar resident bytes. Their scripts also inherited later activation-prefetch
defaults that rejected an interleaved resident before kernel planning.

The three harnesses now explicitly load and borrow interleaved qmicro weights,
disable permute/WMMA-prefetch/precomputed sums, and record `comparison_policy`
in results. Compact-cache comparison holds half-row and padding-elision off;
half-row comparison holds padding-elision off. Owner and child use matching
layout policy. Host tests reproduce the former construction failures, bind
both arms and result provenance, and run the real Q6 primitive wrapper with
stubbed launch symbols across tile-neighbor and unrelated lane counts.
Production defaults, arithmetic and kernels are unchanged. Original July
numbers remain in the pinned sweep source and immutable artifacts, not a
new performance claim for these repaired comparisons.

**Remaining work:** migrate constructor-level controls and numerical/A/B
callers to explicit primitive ownership together. Do not delete these distinct
comparison arms or registered fallback instantiations with the selectors.
Public validation is blocked by the local Laguna model resource described in
R0313–R0320; it is not an admission restriction.

**Clearing action:** run Q6 activation-comparison, resident-layout, runner,
selected-MMQ and full Q4/Q6 MoE oracles, then exercise each repaired harness
and public `LLM.generate()` with an available Laguna artifact. Confirm
matching resident/consumer layouts, selected defaults, exact outputs and
cleanup before retiring the public overrides. Preserve the enabled backend
capabilities and ordinary cache consumers needed by other quant/shape routes.

### R0323 — Laguna fused selected-SiLU pack comparison ownership

`set_fused_selected_silu_pack(...)` still selects the two live arms of
`scripts/laguna_fused_silu_pack_ab.py`; the MoE numerical fixture also requests
the composite directly. gfx1151 enables it by default. The caller admits only
compatible one-plane gate/up and down routes with enough selected-down scratch;
a failed compact gate launch restores the strict route-major chain.

Host contracts cover all declared mode pairs, scratch sizes below/at/away from
the capacity boundary, fused deferral, fallback reduction/output ownership,
stream/runtime forwarding and comparison child cleanup/state capture. Preserve
the standalone SiLU BF16 boundary and registered activation-quant fallback.
July numerical/promotion results remain in the pinned sweep source and original
evidence; the old one-later-checkpoint wording is not a current removal gate.

**Blocker:** retire the session setter with the live fused/unfused comparison,
not by silently making both arms fused. The local Laguna artifact resource
blocker described in R0313–R0320 also prevents public validation here.

**Clearing action:** migrate the A/B caller to explicit composite/primitive
ownership and retain its complete-state comparisons. Run fused-SiLU comparison,
selected-MMQ and runner contracts plus primitive and full Q4/Q6 MoE oracles;
then exercise public `LLM.generate()` with compatible and fallback shapes,
confirming selected route, outputs and teardown. Keep the enabled capability
and registered strict unfused chain.

### R0322 — Laguna Q6 resident-layout comparison migration

`q6_qmicro` controls both expert-down resident bytes and their consumers;
`q6_qmicro_planar` further selects the record ordering. gfx1151 defaults to
planar qmicro, while gfx1100 defaults to legacy bytes. Borrowed sessions adopt
the resident's layout before package defaults, and validation rejects a
resident/kernel-plan mismatch. Do not remove either side independently.

The materializer converts expert-down tiles after reading either GGUF source
or the versioned legacy cache. Root lm-head tiles stay legacy. Host tests bind
source/cache byte equality, unchanged cache payload/fingerprint, allocation
bytes, profile metadata, failure cleanup and session layout precedence.
The live MoE fixture still requests legacy and interleaved/planar comparisons;
`scripts/laguna_q6_planar_ab.py` also owns an explicit layout pair. Legacy
sessions on gfx1151 must coordinate later activation-prefetch and precomputed
sum controls, rather than inheriting incompatible default prerequisites.
Original July results remain in the pinned sweep source and immutable evidence;
this pass changes neither production defaults nor kernels.

**Blocker:** retirement must migrate layout comparison ownership, materializer
and session construction together. The local Laguna model-resource blocker in
R0313–R0320 prevents the required public-route validation, not execution.

**Clearing action:** migrate live MoE and planar A/B callers to explicit
resident/consumer layout ownership, retaining the legacy cache adapter and
registered non-qmicro consumers. Run Q6 resident-layout, materialization,
runner, linear-dispatch and full MoE numerical tests; then exercise public
`LLM.generate()` across short, boundary-neighbor and unrelated prompt lengths,
recording resident/consumer agreement, outputs and teardown. Preserve historical
result readers and never reinterpret legacy bytes as planar qmicro.

### R0321 — Laguna cached-attention comparison and policy migration

`prefill_kv_preappend`, `prefill_cached_meta`, `prefill_global_qrow6` and
`prefill_dense_initial` remain constructor/cache controls. gfx1151 enables all
four by package default. gfx1100 also enables preappend, with role-scoped
attention selection; it is no longer uniformly attend-then-append. Package
role variants take precedence over the legacy cache flags after their own
shape, range and metadata checks.

Unscoped preappend permits complete M128 tiles and global 256/512/1024/2048-row
tiles within capacity. Partial tiles and wrapped SWA preserve the source path;
the runner also excludes staged verifier transactions. Dense-initial selection
requires consecutive pending positions and unmodified metadata. Legacy global
qrow6 starts at position 128, while SWA stays qrow4. Preserve those automatic
checks and registered source fallbacks when retiring public overrides.

The cached-meta, global-qrow6 and dense-initial A/B scripts still construct
children with explicit controls and distinct prerequisite settings. The host
cache tests cover all legacy flag combinations, positions on both sides of
128 plus unrelated positions, role precedence, partial/wrap/wide admission and
metadata invalidation. Original July promotion evidence remains in the pinned
sweep source and immutable worklogs; it is not a new result from this pass.

**Blocker:** retire the public overrides with their live comparison callers,
without conflating legacy flags and package-role policy. The same local Laguna
artifact resource described in R0313–R0320 prevents public-route validation;
this does not restrict execution or disable any default.

**Clearing action:** migrate session/cache construction and all three attention
A/B callers together, preserving primitive comparisons and historical readers.
Run cached-attention, runner, role-policy and dense-metadata contracts; then
GPU global/SWA parity across partial, eviction, wrap and verifier cases plus
public `LLM.generate()` at short, boundary-neighbor and unrelated lengths.
Confirm selected variants, append ordering, state/output equality and cleanup.

### R0312 — Head-major K/V resource-budget integration

The redundant `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV` boolean reader is removed.
Contiguous head-major layout follows backend package policy: enabled on gfx1151,
disabled on gfx1100. Gapped BF16 gather independently resolves its registered
copy kernel and keeps `HIPENGINE_GGUF_GAPPED_GATHER` as its own control.
Oversized, allocation-denied and unsupported-copy cases keep their existing
strided/native fallbacks; kernel bodies and default arithmetic are unchanged.

`HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS` and `*_MAX_BYTES` still bound
the contiguous pair and demand-sized gapped gather. Planning and allocation use
the same pair-byte calculation. Historical promotion evidence remains in
`benchmarks/results/2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json`;
its old rollback command describes that historical implementation only.

**Remaining work:** replace both resource controls with the common scratch
budget without dropping requested-context sizing, allocation rounding or
partial-allocation cleanup. This is resource accounting, not a reason to
reintroduce the retired boolean or gate execution on artifact identity.

**Clearing action:** migrate the planner, contiguous allocator and gapped
allocator/admission together. Run head-major policy, capacity-route terms,
gapped-slot and request-ownership tests plus head-major GPU copy fixtures.
A resource-policy replacement also needs public short/long/gapped requests,
allocation-denial fallback and zero tracked cleanup. Do not use historical
copy-inclusive speed rows as evidence for a changed capacity policy.

### R0311 — Family-specific AOTriton stream policy

`HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM` is shared by PARO and GGUF,
but their defaults differ. PARO enables isolation at 512 query rows and above;
GGUF uses same-stream dispatch on both gfx11 packages after the recorded GPF-4
stability rejection. Explicit `0`/`1` selects either policy, still subject to
the row threshold. The PARO exactness script actively compares both settings
and restores inherited/nested values after errors.

Both resident families lazily allocate one stream and two events, reuse the
bridge and release it with the session. Partial event-allocation failure
releases newly created handles before retry; partial published bridge state
is an error, not permission to use mismatched handles. Existing decode-state
tests check caller/isolation stream event ordering.

Historical PARO evidence remains in the July 12
`gfx1151-paro-aotriton-stream-isolation` and
`w7900-gfx1100-paro-gfx1151-transfer` result artifacts. GGUF's observed slow/stuck
long-prompt attempts and same-stream control are recorded in
`benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-final-rejected.json`;
its candidate-focus and clean-promotion artifacts remain historical evidence.
This is an observed stability blocker, not missing qualification. No unchanged
failing workload needs a rerun just to retire a ledger paragraph.

**Decision needed:** keep the explicit family-aware A/B control or replace it
with resolved execution policy. Do not delete PARO's default isolation or make
GGUF inherit it merely because both readers use the same environment name.

**Clearing action:** migrate both readers, comparison script, bridge ownership
and execution metadata together. Run stream policy/lifecycle, PARO exactness
and decode-state event-order tests. Any policy replacement needs real public
prefill requests below/above/unrelated to the row threshold, state/KV/output
comparison, repeated cleanup and long-prompt stability for affected families.
A changed GGUF isolation design must finish the recorded long-prompt protocol
and same-stream controls before replacing its default; timing claims need
copy/queue-inclusive end-to-end evidence, not a clean short focus leg.

### R0310 — PARO rowchunk-boundary diagnostic ownership

`scripts/qwen35_batch_hidden_bisect.py --compare-full-attn-rowchunk-boundary`
compares two native batched runs per layer limit: explicit rowchunk size zero,
then the requested rowchunk size/layers. Its `c1` summary field labels the
second batched peer as `native_full_attention_rowchunk_repair`, not an
independent singleton oracle. The mode requires native batch attention and
`0 < chunk size < batch size`, and rejects simultaneous context-rowchunk
comparison. It records `performance_claim: false`.

The command works at other widths too; host-only orchestration tests exercise
3/6/7 rows and distinct chunk sizes, using the real summary path. PARO still
has automatic c6 rowchunk policy with selected layers. The rejected suffix
rowchunk route is removed, but that does not remove this full-layer comparison.
The historical investigation remains in
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-rowchunk-boundary-combined-summary.json`
and the pinned sweep source; no new numerical or timing result is asserted.

**Decision needed:** retain this named comparison diagnostic or replace it
with another trace that preserves the two-peer distinction. Solving the c6
rowchunk cost is separate kernel/performance work; do not claim it closed from
host fixtures or delete reachable c6 execution to avoid the question.

**Clearing action:** migrate parser, dry-run and executed comparison metadata
with the replacement, and run
`tests/test_unit_qwen35_rowchunk_comparison.py`, hidden-bisect command tests and
PARO rowchunk policy/forwarding tests. A c6 implementation change also needs
independent hidden/state/KV numerical checks, public requests at neighboring
and unrelated widths, selected paths and cleanup; any speed claim needs the
full benchmark protocol rather than this diagnostic's wall time.

### R0307 — Target score and hidden trace ownership

`--record-target-topk-scores` requests full-logit host capture from block
verification, then emits compact `target_lm_head_score_rows` and aligned
`target_hidden_seed_rows`. It requires block verification; native target-cycle
and enqueue-only graph verification refuse the host output. Capture does not
force a logits-producing head: a singleton direct top-1 head still returns no
logits. Only materialized row-head logits produce score rows, with one FP32
vocabulary-sized copy per verifier row. This diagnostic is not a timing route.

Compact rows preserve available input/target alignment, ordered unique draft,
target and extra candidates, FP32 near-tie scores and deltas. Missing logits
produce no score rows; invalid candidate IDs have null scores. Hidden summaries
mark unavailable rows rather than substituting another seed. The original
capture and hidden-score evidence remains in
`benchmarks/results/2026-07-03-mtp-target-score-capture-smoke.json` and
`benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-hidden-scores-live.json`.
These are historical diagnostics, not a new parity or performance result.

**Decision needed:** keep the compact cross-engine score diagnostic or replace
it with a tensor trace before retiring the CLI and result fields. Keep this
trace ownership separate from R0301's singleton top-1 arithmetic selector.

**Clearing action:** migrate verifier capture, benchmark callers/serialization
and attribution consumers together. Run benchmark metrics, verifier LM-head,
native-state attribution and forced-target probe tests. A replacement must
preserve exact row/position/candidate alignment, near-tie precision, explicit
missing-row behavior and graph readback refusal, then exercise real
verify/commit/reset plus a separate public NextN preservation request.

### R0306 — Legacy draft stage-trace replacement

`--record-draft-stage-stats` still collects FP32 sub-stage summaries into
`draft_hidden_state_trace`. `--record-draft-cache-rows` adds requested K/V
history; `--record-draft-attention-debug` reads dense attention inputs/output
and recomputes scores/weights on the host. The benchmark requires stage stats
for both extensions and device K/V for attention debug. Stage stats bypasses
the device chain; native graph proposal explicitly refuses these readbacks
before capture. This is a legacy diagnostic, not a timing or public NextN path.

Cache summaries include current and previous rows, then unique valid requested
history in request order. They use FP32 K/V strides and preserve depth, input
token and position metadata. Existing attention-debug tests compare a synthetic
host recomputation to independently prepared device-output rows; they do not
establish a new model-wide parity result. The pinned original row preserves
the historical RoPE-dimension fix and attention investigation.

**Decision needed:** keep this explicit host trace or replace it with a
capture-safe graph tensor trace before retiring its CLI/result fields.
No implemented graph trace currently substitutes for these summaries.

**Clearing action:** migrate benchmark argument validation, both proposal
callers, stage/cache/attention summaries and `draft_hidden_state_trace` consumers
together. Run `tests/test_unit_mtp_resident_draft_device_commit.py`, benchmark
parser and profiler contracts, then compare trace metadata/values for initial
catch-up, multiple proposal depths and commit/reset. Any replacement needs
independent numerical rows plus a public NextN preservation request; host
trace time must not be reported as a normal device-chain timing result.

### R0305 — Diagnostic F32 token-embedding seed ownership

`HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING` selects host dequantization of
requested `token_embd.weight` rows only within enabled F32 residual block
verification. The BF16 device embedding still runs. A cached GGUF reader feeds
ordered FP32 rows into `_verify_hidden_f32_a`; duplicate token IDs remain
duplicate rows, without an intermediate BF16 rounding. This is separate from
R0239's host/device placement control. Native-state attribution explicitly
refuses the combined F32-residual/F32-host-embedding diagnostic; it is not a
GPU timing route.

The pinned sweep source and July layer-0 trace artifacts preserve the original
parity finding. The later
`benchmarks/results/2026-09-07-zbook-ud-embedding-precision.json` records another
precision-isolation use. Neither establishes that all current diagnostic
callers can be deleted or that host lookup belongs in production timing.

**Decision needed:** retain this explicit host precision oracle or replace it
with a device-side FP32 embedding implementation before retiring the selector.
Keep reader/seed ownership distinct from the broader R0287 arithmetic plan.

**Clearing action:** migrate seed construction, F32-residual caller, trace and
native-attribution contracts together. Run
`tests/test_unit_qwen35_gguf_verify_f32_token_embedding.py`, token-embedding
parity and attribution tests. Any replacement must compare requested/duplicate
rows at FP32 precision, preserve the BF16 mirror and actual block inputs, then
exercise real verify/commit/reset and cleanup. A changed-arithmetic production
default needs its execution-profile gates; a host-only trace is not that gate.

### R0304 — Legacy selected-SiLU/down comparison ownership

The rejected `siludown` and `siludown-allsync` suite presets are removed.
Explicit benchmark `--resident-mtp-draft-selected-silu-down-fused` and profiler
`--selected-silu-down-fused` comparisons remain, as do the registered fused
primitive and its independent fused/unfused numerical test in
`tests/test_gpu_gguf_k_selected_pack8_dp4a_gemv.py`.
`HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` is the captured
construction control for these explicit comparisons. The original sweep source
and unchanged July artifacts preserve the rejection history.

`apply_moe_down_combine()` selects either separate BF16 SiLU then Q5 selected
down, or the fused primitive that rounds the intermediate to BF16 internally.
Both feed the same device-owned selected-order down output into routing-weighted
shared-gate/residual combine and the final BF16-to-F32 cast. The legacy draft
captures the selector at construction; it is separate from Generation-2 NextN.
Omitted and negative benchmark options preserve inherited settings, as does
omitting the profiler switch. No kernel, arithmetic or default changes here.

**Decision needed:** keep the explicit diagnostic or migrate it into an
owner-local legacy composite plan before removing its environment/CLI readers.
Do not delete shared registered primitives or silently run an unfused control
under a fused comparison label.

**Clearing action:** migrate constructor capture, helper argument, benchmark/
profiler cache/provenance and explicit CLI callers together. Run selected-down
branch/marker/stream ownership and parser contracts, then the fused/unfused
numerical comparison. A changed runtime route needs live legacy proposal,
commit/reset and cleanup plus separate public NextN preservation. A new
changed-arithmetic default needs its execution-profile gates.

### R0302 — Resident draft F32 router and shared-gate comparison ownership

The legacy resident draft captures
`HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL` at construction. It selects
`qwen35_router_logits_f32_f32w` for both the 256-expert router and one-output
shared-gate projection, each from F32 post-norm rows with 256 threads. The
standalone default uses `mtp_linear_f32` for both. Compatibility generation
already binds the switch on; benchmark/profiler CLIs and named `routerrow`
suite routes still compare it. Omitted/negative benchmark options and omitted
profiler options preserve inherited settings. Normal Generation-2 NextN MTP
uses a separate provider; do not infer its policy from this legacy toggle.

**Decision needed:** preserve the explicit two-projection comparison or move
its selection into an owner-local legacy draft plan. The original sweep source
and unchanged July artifacts preserve the historical timing comparison; it
is not a fresh claim about either provider. Do not remove generic MTP linear,
a shared primitive, merely because this caller selects another projection.

**Clearing action:** migrate constructor capture, both projection branches,
compatibility binding, benchmark/profiler cache/provenance and suite callers
together. Run `tests/test_unit_mtp_resident_draft_device_moe.py`,
`tests/test_unit_mtp_weight_upload_ownership.py`, benchmark/parser/profiler and
suite contracts, then the independent F32 router/shared-gate numerical
comparisons. A runtime change needs real legacy proposal/commit/reset and
teardown plus a separate public NextN preservation request. A changed-arithmetic
default still needs its execution-profile gates; this reconciliation changes
neither arithmetic nor defaults.

### R0301 — Verifier Q6 top-1 sidecar comparison ownership

`--verify-lm-head-q6-top1-dp4a` still publishes both the load-time
`HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR` and runtime
`HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A` switches. Omission and the
negative CLI option leave inherited settings unchanged. Target materialization
owns this X8 allocation; the draft's capped X8 upload is independent (R0281-DRAFT-X8).
Named `vlmheadtop1` suite routes still request the comparison.

The direct helper is singleton-only after the recorded multirow invalid-sentinel
failure in
`worklog/entries/20260817T000059.169563Z-lhl-concurrency2-c2-6-global-native-46b0f0.md`.
The block caller now matches that width boundary: multirow eager and enqueue-only
verification passes the BF16 mirror to the existing full-logit sampler rather
than passing FP32 to a fallback that rejects it. Singleton direct sampling still
uses FP32 hidden seeds, q8_1 quantization and the X8 head; a missing sidecar
raises. Shared kernels, ordinary logits sampling and default arithmetic are
unchanged. Historical B2 suite labels do not establish current direct-head
execution; inspect the actual head/sampler path.

**Decision needed:** retain the explicit singleton comparison or migrate the
load-time sidecar and runtime choice into one owner-local head policy. Retire or
rename multirow comparison aliases only with their CLI/result consumers, rather
than silently presenting fallback measurements as direct-top-1 measurements.
Do not delete the shared Q6 kernels or couple this allocation to draft layout.

**Clearing action:** migrate materialization, singleton/packed head callers,
benchmark setup/provenance and suite routes together. Run
`tests/test_unit_qwen35_gguf_verify_lm_head_top1.py`,
`tests/test_unit_gguf_mtp_bench_metrics.py`, suite and weight-upload contracts,
then public MTP requests confirming singleton versus multirow head/sampler paths,
continuation and cleanup. A proposed multirow direct path must first resolve the
recorded sentinel failure and pass independent numerical/ownership checks;
changed-arithmetic promotion still requires its execution-profile gates.

### R0281-SELECTED-X8 — Selected-expert replacement-layout policy

`HIPENGINE_GGUF_SELECTED_X8_REPACK` is a load-time primary-layout selector,
not a raw sidecar or runtime rollback. With decode repack enabled it chooses
Q5/Q6 rank-3 selected-down X8 tiles; gate/up experts and other quant families
keep their own layouts. Explicit selected-down raw placement takes precedence.
Changing the variable after loading cannot change the resident layout.

The live compatibility generator selects `q6`. Bench/profiler and named suite
routes still publish the choice before loading; Q5 X8 also has executable
BF16/F32 activation adapter and output contracts. Historical rejection of
Q5/both for one timing lane does not remove those implemented capabilities.
X8 selected GEMV requires q8_1 workspace and raises when absent; there is no
implicit raw/T16 fallback for already-X8 weights. Keep this owner separate from
draft Q6 top-1 and target lm-head X8 sidecars.

**Decision needed:** move primary-layout selection into the loader's explicit
plan with coordinated compatibility/bench/profiler callers, or preserve the
existing diagnostic selector. Do not remove Q5/both solely from the old timing
row or let a named X8 route silently load a different layout.

**Clearing action:** migrate quant-family parser/precedence, materialization,
prerequisite/admission byte plans, selected adapters/workspace, cache identity,
CLI/suite provenance and forced-trace replay together. Run both quant families
with repack on/off, raw precedence and non-down slots; retain converter,
selected-call/admission and BF16/F32 output tests. A runtime/layout change also
needs actual public and compatibility decode/verification, reset/cancellation
and teardown plus the applicable arithmetic profile gates.

The separate `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` loader selector chooses
rank-3 Q4_K gate/up primary tiles when decode repack is enabled; explicit
gate/up raw placement wins. Its old bench/profiler switch and `x8gateup`
suite aliases are removed. Keep the loader and selected-dual kernels for
parity workbench candidates `q4-x8`/`q4-x8-q6`, direct converter/admission
and adapter tests, and explicit historical trace replay. Bench/profiler
artifacts record the ambient selector through the loader parser so those
consumers do not silently report or replay T16 instead of X8. Include these
owners in the same explicit loader-plan migration; deleting rejected aliases
is not permission to remove an implemented layout or its numerical oracles.

Include `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW` in that loader-plan decision.
It selects Q4_K gate/up raw primary storage before X8/T16 selection, not an
extra raw sidecar, and does not itself enable q8_1/dp4a arithmetic. Its rejected
bench switch and `rawgateup` suite aliases are removed; the loader, raw
selected-dual consumers and historical trace replay remain. Bench artifacts
record the ambient value through the loader parser and retain raw environment
provenance. Migrate that selection and replay metadata together with X8;
removing comparison aliases does not authorize disabling explicit raw loads.

### R0281-PARO-PROVIDER — Manual provider and private-scorer migration

Public Generation-2 PARO MTP always captures final-normalized target hidden,
reseeds from the selected target row and borrows the target's full-vocabulary
W8A16 head. `HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT` and the private prefix cap
do not select that route. The profile module binds the variable for the older
`mtp_chain_e2e_smoke.py` persistent-device harness; both strict and production
use the corrected proposer, with distinct verifier choices.

An explicit chain mode without a profile still preserves the ambient selector.
The private F16 scorer also serves standalone proposer construction in the
layer/state drift audits and the lifecycle gate's private-versus-borrowed memory
comparison. Its `vocab_topk` diagnostic supports branching; the borrowed scorer
currently exposes top-1 only. Removing the private scorer is therefore not the
same operation as retiring the manual selector, and must not silently change
those comparisons or eliminate branching capability.

**Decision needed:** preserve these named historical diagnostics or migrate
manual provider choice to an explicit harness-owned policy while separating
private scoring from the public provider. Do not widen the registered B1
profile scope or treat its scope check as the public Generation-2 depth limit.

**Clearing action:** migrate profile binding, explicit-chain harness selection,
proposal/scoring metadata, drift/lifecycle callers and private-head allocation
ownership together. Retain selected-row reseeding, full-vocabulary borrowed
scoring, strict verifier fallback and stale-owner rejection. Run profile/manual
selector and public adapter tests, borrowed/private scoring and lifecycle tests,
plus actual strict/production public generation with selector unset/0/1 and
reset/close coverage. Any removal of private branching needs an explicit
replacement or retirement decision, not just a release-window expiry.

R0282's older chain-verifier row belongs to this ownership decision, not a
separate B1 default change. The historical profile harness registers B1
fixed-chain/graph-off routes: strict uses `c1_loop`, production uses
`decode_batched`. Public Generation-2 chooses those attention modes from its
execution profile directly; its automatic candidate depth is two, with explicit
depths one through three. Keep capability, tail-room planning, prefix repair
and session sizing separate from the historical profile scope. The completed
[depth unit](../worklog/entries/20261024T174430.936309Z-paro-mtp-depth-paro-mtp-depth-chain-96ac00.md)
owns that public change and its validation; no new timing claim is made here.

`HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP` and `HIPENGINE_VERIFY_GPU_ACCEPT`
already default on. Their off paths retain unfused/CPU-reference comparisons;
GPU acceptance also supports `validate`. The chain t-loop requires singleton,
linear-parent, all-active topology, not B1 depth. Native selected-commit and
DFlash verification also consume this machinery. Include these shared controls
and graph-cache keys in any execution-owner-local verifier-plan migration;
do not remove their reference paths or change provider-specific capture and
commit semantics as a side effect of retiring manual profile selectors.
Cover topology/depth, on/off/validate, graph-key isolation and accept payload
parity alongside the profile/public-depth tests above, then exercise real
public strict/production requests plus any affected DFlash/native callers.

R0282's tree/top-k row also joins this referral. PARO branching comparison is
already explicit in `scripts/mtp_chain_e2e_smoke.py` through
`--tree-mode branching_topk`, top-k width and confidence controls; its default
is chain. The persistent branch route uses private-head top-k and requires a
positive confidence threshold and at least two active candidate rows. It
compiles sibling topology, verifies through `verify_tree_bulk_and_commit` and
repairs the selected provider branch. Reload comparison has its own top-k
inputs and policy. These are not a default-off public tree implementation:
public Generation-2 declares chain, while shared tree verification and metadata
also serve DFlash comparisons.

Choose whether to preserve these explicit harness comparisons or retire them
with their private-scorer consumers. An old negative speed result does not by
itself justify deleting working tree primitives or changing public admission.
A migration must cover reload/persistent mode selection, confidence and active
budget boundaries, sibling acceptance/selected hidden and KV repair, graph
bindings and tail padding. Keep the shared compiler/mask/acceptance oracles and
confirm public chain routing stays unchanged. The corresponding GGUF
root/sibling acceptance policy is separately owned by R0281-TOPK below.

### R0281-TOPK — GGUF benchmark proposal-policy ownership

`scripts/gguf_mtp_bench.py` defaults to B1 with root top-40, sibling top-1
and branch redraft disabled. This is not a default-off tree. The AR/MTP
suite's default route explicitly chooses root/sibling top-1; other named
routes inherit the standalone root choice. The rejected adaptive strict-probe
hybrid and its root-top-40 fallback are removed. The parity workbench also
forwards a root choice. These are live acceptance-policy callers, not unused
CLI aliases.

Top-k acceptance uses target-selected IDs. At a non-argmax match it either
emits the next corrective target token or, with branch redraft enabled,
rebuilds the remaining proposal from the accepted target branch. Removing
it changes proposal accounting, verifier/hidden-row consumption and adaptive
policy, even when visible IDs happen to match. Keep the public NextN
provider and its separate acceptance owner outside this benchmark migration.

`--adaptive-full-vocab-after-cap-miss` is also a live proposal-policy control.
It is off in the standalone and suite-default routes, but named suite routes
and the parity workbench's `resident-cap32k-recover` comparison enable it.
When the configured cap is below the output width, it owns a second full-head
runner. A capped nonempty low-accept miss selects that runner for the next
eligible cycle and suppresses AR fallback for that miss; full acceptance on a
recovery cycle returns to the capped runner. A failed recovery may still
activate AR fallback. The configured accepted-count threshold, not just zero
acceptance, defines a low-accept miss. Keep runner selection, cache/session
reuse, full-runner teardown and current/next-cycle artifact fields together.
No public NextN provider switch is implied by this benchmark control.

**Decision needed:** keep explicit linear/tree and vocabulary-recovery
comparison policies or move those policies into a dedicated experiment harness
with coordinated suite and parity-workbench callers. Do not change the standalone
root default as a side effect of deleting controls or use a historical speed-negative tree
row as a reason to disable an implemented request.

**Clearing action:** migrate root/sibling limits, branch redraft and tail
gates, vocabulary-recovery runner lifetime and AR-fallback
suppression, route aliases, category forwarding and artifact candidate/acceptance
fields together. Retain linear/mismatch/out-of-set and
corrective-token boundary tests, branch reseed/commit and candidate counting,
plus parser/suite/workbench contracts. Cover empty/full/partial drafts, capped
miss then recovery success/failure, accepted-count threshold boundaries and
full-width/no-recovery-runner controls. A runtime migration needs real
linear/root/sibling/redraft and vocabulary-recovery cycles, hidden/KV/cursor,
reset and close checks; confirm public generation is unchanged. Any new performance
claim needs its own matched protocol, not the retired ledger row's numbers.

### R0281-CONTEXT — GGUF benchmark prompt priming and draft KV policy

`--mtp-context-replay` captures every prompt hidden row through bulk target
prefill and constructs the shifted llama.cpp-style prompt inputs. With
`--mtp-device-kv-cache`, resident `write_kv_rows` or the legacy KV-only draft
call primes device KV before measured cycles. Without device KV, the flag
captures host prompt rows only; it does not replay a growing host history.
The unused per-cycle host token/hidden-history append is removed.

Device KV also works without prompt catch-up. It retains the cycle-start row,
replaces accepted speculative rows with verifier-derived hidden rows and drops
rejected rows by resetting the cache length. Native complete-cycle execution
owns its own repair/cursor result instead. These flags are forced by
llama-compat and remain explicit in suite/workbench context comparisons;
complete-cycle diagnostics require device KV. They are not limited to B1.

The device-seed route is already attached to `Qwen35GGUFMTPContext`: initial
`from_target_seed()` captures the target's FP32 pending-seed pointer, and
proposal reads `context.pending_seed.hidden_ptr`. Non-native cycles update it
from accepted verifier rows or recapture the target seed; native complete-cycle
execution owns its update. Keep ready-single-row validation, pointer/token/
position identity and failed-capture preservation. This avoids a required
per-cycle host seed readback, but does not imply that all prompt, diagnostic or
KV-repair work is host-free. The live `--resident-mtp-device-seed` comparison
and its suite aliases remain part of this context-policy migration.

**Decision needed:** preserve these independent prompt/cache and host/device-seed
comparisons or replace the switches with an explicit benchmark context policy
across their callers. Prompt priming, cache placement and public NextN prompt streaming are
different owners; do not replace one with another or change the standalone
single-seed default merely to remove a switch. Historical timing results do
not decide whether an implemented cache configuration may execute.

**Clearing action:** migrate all-row prompt capture/shift, resident and legacy
KV writers, device-seed staging, native proposal cache ownership, suite/workbench
arguments and context/cache provenance together. Retain no-cache, device-only
and prompt-primed cases, B1 and deeper drafts, reject/partial/full repair,
capacity exhaustion, cached-session reset and close checks. Run real host/device
seed cycles and verify output, acceptance, hidden/KV/cursor continuity; confirm
the public NextN provider is unchanged. Do not remove the compatibility cache
until its replacement implements those same lifetime and commit semantics.

### R0281-TARGET-POLICY / R0284 / R0299 / R0300 — GGUF verifier controls and native naming

The standalone benchmark leaves block verification and direct state commit
off, but the AR/MTP suite's default route enables both and llama-compat forces
both. Suite aliases and the parity workbench still compare bulk, native and
serial-exact schedules, replay and captured-row commit. These are active
transaction-policy interfaces, not unused scaffolding. Use
`--target-block-verify --target-block-verify-mode MODE`; the verifier mode is
not an optional value on the boolean switch.

R0299's `--fused-b1-block-probe` is a separate adaptive benchmark schedule,
not a public feature awaiting enablement. Its named suite route starts with
one draft token and permits `[prev, draft0]` block verification before full
acceptance activates the configured larger draft window. It requires block
verification, adaptive full-accept selection, probe width one, minimum rows
two and strict root/sibling top-1 without branch redraft. AR fallback and an
already-active block schedule suppress the probe. The suite's default omits
the probe flag; llama-compat explicitly disables adaptive probing.

Full-suite staged and unstaged evaluations already exist in
`benchmarks/results/2026-06-30-ar-mtp-fused-b1-block-direct-cap32k-minrows2-pmin05-b5-full.json`
and the adjacent `-full-nostage.json` artifact. Their existence removes the
old pending-evaluation rationale, not the distinction between schedules.
Preserve the explicit comparison or retire it with this verifier-policy
migration; include adaptive activation, AR fallback, output-tail rows,
acceptance/commit accounting and cycle-policy provenance. Do not delete the
shared block verifier or change the suite default from a historical speed row.

R0284's `--llama-compat` benchmark composition forces B2/top1, full draft
vocabulary, shifted prompt replay, device draft KV, bulk verification and
captured-row commit, with adaptive probes/fallback disabled. Applying it is
idempotent; it preserves explicit arithmetic, seed/chain, native-cycle and
observation choices. The suite derives fixed `[2]` labels from every route
containing that flag, so new aliases cannot inherit misleading scope budgets.
The benchmark flag does not itself enable DP4A, and its direct partial-commit
policy is not a promise of exact target state or cross-engine token equality.

The separate `Qwen35GGUFBringupGenerator.generate_speculative_mtp_detailed()`
compatibility caller also uses the shared assets and adapters. Public resident
NextN scheduling and that direct compatibility loop are different owners;
removing a benchmark alias must not remove either implementation implicitly.
The historical rates and current parity rerun recipes remain in
`docs/archive/MTP-LLAMACPP-PARITY-HISTORY.md` and
`docs/campaigns/MTP-LLAMACPP-PARITY.md`; the original ledger text is preserved
in the sweep inventory rather than restated as current performance.

Direct commit selects a captured Conv/GDN/hidden row; replay policies restore
a snapshot and advance the consumed prefix instead. The benchmark's bulk
policy has its own end-position and partial-commit rules; serial-exact/native
and llama-style bulk direct partial commit are distinct contracts. Keep graph
invalidation after block execution, absolute positions, hidden-row selection,
KV/cursor accounting and snapshot cleanup together.

R0300's partial-replay selector remains in both the benchmark and forced
lifecycle probe. `serial-exact` is the parser default; llama-compat rewrites
that value to `direct-commit`, even when explicitly supplied. Other explicit
choices survive. Serial-state-only replay advances through the scalar target
without sampling; bulk/native state-only modes use their named block schedules.
Those state-only token IDs are echoed inputs, not new scores. Keep original
block-selected visible IDs separate from replayed state and metadata.

The forced lifecycle probe also uses the full-block exact-commit predicate:
for bulk, `start_position + rows < 1024`. At or beyond that boundary, even a
full accept or `direct-commit` choice restores a snapshot and replays. The
benchmark's partial-block direct-commit branch uses its separate partial
support predicate. Preserve this diagnostic distinction rather than claiming
the two callers always take the same commit path. Cover reject/partial/full
accepts on both sides of the boundary and release snapshots on forward,
restore, replay and missing-capture failures.

The original R0300 row and unchanged July lifecycle/full-suite artifacts keep
the observed state divergences and historical comparisons. They do not make
captured-row commit exact or authorize deleting the independent serial-state
control. Preserve these explicit diagnostics or retire their callers as part
of this same transaction-policy decision.

The dedicated `scripts/gguf_mtp_forced_target_probe.py
--state-lifecycle-compare` remains an independent diagnostic after removal
of the bench replay-on-every-block switch. Its replay arm keeps full-block
state on full accept and restores/serial-replays only an accepted partial
prefix. Its direct arm captures rows and uses explicit partial-commit or
replay policy. Each arm owns a fresh session and closes it on failure.
Comparison covers position, current token, FP32 hidden seed and per-layer
Conv/GDN bytes. This mode does not request KV fingerprints and stops an arm
when visible tokens diverge from the trace; a clean compared prefix is not
proof of full-KV equality or completion of the entire trace. Preserve the
component mismatch records and policy provenance until an independent,
equally scoped replacement is available. Migrate this diagnostic with the
verifier policy rather than deleting one of the compared state contracts.

`Qwen35GGUFResidentSession.verify_target_block_serial_exact()` is also a live
compatibility accepted-prefix fallback and independent scalar-state oracle,
not only a benchmark mode. Scored execution uses `step()` per token;
`advance_state_only=True` advances the same hidden/state path without LM-head
sampling and echoes input IDs, which must not be interpreted as scores. Both
stage FP32 hidden rows and may capture per-row Conv/GDN state. Keep its
absolute-position ordering, default-stream restriction and pre-execution
input/capacity checks. Benchmark selector consolidation must not remove this
fallback or replace its independent wrong-branch/state-only controls with the
bulk/native candidate being checked. A throughput win does not retire an
oracle; a replacement must first preserve its state and scoring contracts.

The native target implementation accepts root-prefixed B1-B7 chain shapes,
subject to backend, state/storage, context and capture constraints; that is a
code capability, not a new measurement claim. Its function/class/registry key
still includes `b2`. N1 target-only and N2 device accept/commit are independent
cache variants. Exact public transactional verification also calls this owner;
it is not restricted to the benchmark's compatibility route. N3 complete-cycle
and proposal-graph consumers must migrate with it, not replace it by deletion.

**Decision needed:** preserve named comparison controls or consolidate them
into an explicit benchmark verifier policy, and decide whether to rename the
legacy native API/registry surface in the same coordinated migration. R0284's
remaining alias-retention decision belongs to that migration alongside
R0281-CONTEXT, R0281-DRAFT-CHAIN and R0281-SELECTED-DP4A; it is not a separate
speed-only gate. Do not
remove a correct target-only or fallback capability because a historical
throughput comparison preferred a different complete-cycle route.

**Clearing action:** migrate bench parsing, suite/workbench aliases, verifier
schedule/commit choices, native helpers/class/registry references, graph cache
identity and requested/effective/fallback artifact fields together. Keep exact
public and compatibility owners separate. Run parser/suite, snapshot/replay,
selected-state commit, dynamic context/position and graph failure/close tests;
exercise real reject/partial/full and output-tail cycles, reset/cached-session
reuse and public generation. Any arithmetic change needs its profile gate;
any speed claim needs a matched protocol rather than the removed historical
ledger rows.

### R0281-CAPTURE — GGUF production row-state and diagnostic policy

`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN` selects production Conv/GDN
row-state capture, not just offline instrumentation. Public generation and
profile binders scope it on; packed verification requires it. Preserve its
segmented Conv and decode-order no-copy GDN writers, FP32/FP16 state typing,
selected-row commit and scored-boundary capture. The copying GDN row-state
wrapper is also an independent GPU correctness-test control, not a substitute
for the production no-copy writer.

The five other capture selectors change arithmetic or score/commit policy:

- `HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV`: F32 chain Conv.
- `HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN`: regular rather than
  c1-exact chain GDN.
- `HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT`: BF16 output rounding.
- `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV`: chain Conv plus
  prefill GDN.
- `HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL`: temporary prefill scoring
  with chain commit rows.

The forced-target and native-state attribution probes record their
values in diagnostic provenance. `docs/ENVS.md` describes their dependencies;
notably score-prefill does not capture attention scores. Removing these
readers while keeping their provenance names would silently change replay.

**Decision needed:** retain these named comparisons or move them into an
explicit debug capture policy, separately from the production row-state
mechanism. The earlier negative experiments are preserved in the sweep
manifest and benchmark artifacts; they do not make the live production
writers disposable.

**Clearing action:** migrate production/profile/gate binding and diagnostic
policy, scratch ownership, FP16 incompatibility errors, scored-boundary
readers and probe metadata together. Test segmented/single capture, deferred
and final state commit, reject/partial/full acceptance and exception cleanup.
Any route removal must reject unsupported historical requests explicitly or
leave an equivalent replay route. Run real state-row and prefix-consistency
checks and public generation before changing execution; arithmetic changes
also need their execution-profile gate. No execution or arithmetic changes
are made by this disposition.

### R0281-METADATA — Contiguous prefill metadata policy and host oracle

`HIPENGINE_GGUF_PREFILL_DEVICE_METADATA` selects the stream-ordered device
writer or six synchronous host-prepared copies in
`_GGUFFullAttentionPrefillScratch.for_chunk()`. Both gfx11 package defaults
select device metadata through 4,096 prompt tokens and host metadata above
that; explicit values override the default. Preserve all six fields:
`cu_q`, `cu_k`, atomic counter, GDN `cu_seqlens`, positions and contexts.

The host path is not an unused rollback: the conservative, `gdn_exact` and
`q4_shared_x` isolation profiles pin it, while the verifier profiler explicitly
selects device metadata. The host path also supplies an independent metadata
oracle. Native target-graph context limits no longer depend on this bulk-prefill
selector.

The original 128K device-metadata attempt stalled, but later isolation
reproduced the stall with device metadata off and implicated the Q4 shared-X
route. See [the stall investigation](reference/DEBUG-GFX1151-STALL.md),
including its selector maps and bounded reproduction protocol. That evidence
does not establish a metadata defect or a need to retain host metadata
indefinitely. This disposition changes neither backend default nor execution.

**Decision needed:** retain named host/device comparisons or migrate them into
an explicit prefill diagnostic policy; decide automatic long-context selection
separately from interface retirement. A release window alone does not migrate
the live isolation callers, and missing long-context measurements do not make
the explicit device path incapable of running.

**Clearing action:** migrate the selector, backend policy, isolation profiles,
profiler and provenance together, preserving an independent host oracle. Run:

```bash
uv run pytest -o addopts='' -q \
  tests/test_unit_gguf_packed_verify_layout.py \
  tests/test_unit_qwen35_prefill_kernel_profile.py \
  tests/test_unit_gguf_full_attn_split_demand.py \
  tests/test_gpu_runtime_state_unpack_metadata.py
```

Then exercise both writers through `LLM.generate()` below, above and away from
the old ceiling. Verify
all six metadata arrays, stream ordering, partial/final chunks, state/KV,
reset/reuse and teardown. For a long-context policy change, use an explicit
`HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=1` final child of the bounded 128K
protocol linked above, with current defaults otherwise intact; keep
same-stack host and device controls and record the selected route. Do not
attribute a stall to metadata from a blocked host stack alone.

### R0281-Q8-WAVES — Dense T16 prefill schedule policy

The LCP-3 four-wave and GPF-5A two-wave rows describe one live policy in
`hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_t16_prefill.py`, not two
independently disposable duplicates. Its automatic wrappers and direct registry
variants serve Qwen and Gemma 4, including the preconverted `f16w` operand ABI.
The four-wave shape policy includes Gemma 4's 1,408-wide expert gate/up;
other narrow shapes retain the decoding fallback.

`HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE` has explicit precedence for the four-wave
choice. Without it, an explicit `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE=1`
suppresses automatic four-wave and requests two-wave; `2WAVE=0` selects the
decoding fallback. With neither set, the request context and wrapper defaults
choose. Qwen scopes the context through 4,096 prompt tokens on gfx1100 and
65,536 on gfx1151. Direct variants and explicit environment overrides remain
available outside automatic selection. The registry, common `_wide_wave_launch`
and session context must migrate together; replacing only the two-wave body
would change a named comparison request.

**Decision needed:** retain both named schedule comparisons or move them into
an explicit prefill policy while preserving direct numerical oracles. Decide
any backend ceiling or kernel replacement separately. A release window and the
historical two-wave long-context regression do not determine the usefulness or
current behavior of the four-wave, decoding or `f16w` siblings. Original results
remain in the sweep inventory and benchmark artifacts.

**Clearing action:** migrate both selectors and their precedence, wrapper and
registry bindings, request context restoration, Qwen/Gemma consumers, direct
microbench callers, `f16w` ownership/fallback and result provenance together.
Run the no-GPU policy/build/wrapper nodes in
`tests/test_gpu_gguf_q8_0_t16_wmma_prefill.py`,
`tests/test_unit_gguf_q8_t16_f16w_dispatch.py`,
`tests/test_unit_qwen35_gguf_aotriton_stream.py` and backend alias tests. If
execution changes, run the CPU-reference, two-/four-wave and `f16w` device parity
nodes plus `tests/test_gpu_gemma4_dense_wmma_range.py`; exercise Qwen and Gemma
through `LLM.generate()`, tracing actual schedules, sidecars, partial tiles,
request reset/reuse and cleanup. Cover lengths below, above and away from each
changed ceiling. Arithmetic changes need their execution-profile gate; any
performance claim needs a same-host matched protocol. This disposition changes
no execution, arithmetic, default or selector.

### R0281-ROUTER-TILES — Laguna/Gemma router comparison ownership

`LAGUNA_ROUTER_LOGITS_MODE` is backend-package metadata, not an environment
variable. Laguna resolves `token_tile_4` on gfx1100 and `token_tile_8` on
gfx1151; its session setter explicitly selects 4/8/16 and rejects unknown
modes. `scripts/laguna_router_token_tile_ab.py` uses the setter for separate
child-session comparisons, while the token-tile microbench exposes all three
registered primitives. The modes share a library, not a single launch policy.

Gemma 4 also calls the resolver for its multirow tiled dispatch, separately
from its fused single-row and SGEMM routes. Its untiled comparison explicitly
uses 512 threads; Laguna's canonical tile-4 binding uses 256. Do not infer
cross-model or cross-width arithmetic equality from the older Laguna-only
result. The wider-token parity tests and float64/top-k attribution probes
supply independent schedule controls.

**Decision needed:** keep these named comparison modes or migrate the setter
and benchmark interfaces into a scoped diagnostic plan. A later default-only
refresh cannot decide whether to erase direct tile-16 or Gemma comparison
requests. Default changes, arithmetic changes and interface retirement are
separate decisions; the completed F32 router wrapper cleanup did not remove
any tile mode.

**Clearing action:** migrate the session's resolved mode, setter, child-session
A/B harness, microbench, plan function table, backend metadata and Gemma caller
with result provenance. Preserve explicit invalid-mode errors, default c1/
multirow selection and independent 256/512-thread numerical references. Run
`tests/test_live_laguna_moe_gpu.py::test_laguna_router_logits_mode_is_exact_and_explicit`,
backend alias and canonical-router tests, both script `--help` routes and the
Gemma SGEMM comparison tests. Before changing execution, run the router tile
and top-k GPU parity/attribution nodes and public Laguna/Gemma generation with
short, partial-tile and longer prompts, reset/reuse and allocation cleanup.
Declare profile gates for any changed arithmetic. This disposition changes no
route, default, kernel or arithmetic; historical measurements remain in the
sweep inventory and benchmark artifacts.

### R0281-PREFILL-DEBUG — Isolation profiles, progress recording and drains

These live diagnostic controls serve different contracts:

- `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS` overrides bulk router
  selection width; both gfx11 defaults are 128. The conservative profile pins
  512. Explicit 64/128/256/512 choices and invalid-width errors remain; an old
  full-model state divergence at 64 is not a reason to erase explicit diagnostic
  access. Decode width is independent.
- `qwen35_readme_sweep.py --prefill-kernel-profile` applies and records
  fail-closed selector maps. `conservative`, `gdn_exact`, `q4_shared_x` and
  `q4_pair32` pin the host metadata/router/convolution controls and vary the
  named family. `q4_baseline` and `q4_pair32_production` change only Q4 mode
  under otherwise-current defaults. `default` changes no environment values.
- `prefill_flight_recorder_path` and `--prefill-flight-recorder` create a
  mapped-host file ring with host submission and same-stream device retirement
  cursors. Chunk/layer granularity changes marker placement, not model-layer
  synchronization. The independent decoder reads persisted evidence without
  loading HIP; mapped-memory cleanup, sequence identity and ring wrap matter.
- `prefill_queue_drain` / `--prefill-queue-drain none|chunk|layer` add real
  host stream waits only at the selected boundaries; the default is `none`.
  Layer drain is an explicit containment path, not evidence that a named kernel
  caused or fixed the original stall.

The [stall reference](reference/DEBUG-GFX1151-STALL.md) owns the bounded
reproducer and historical attribution. R0281-METADATA owns host/device metadata
interface migration. Keep profiles and pinned selectors coordinated; deleting
just the router override silently changes an isolation experiment. The later
pair32 routes also make a release-window tied to the old baseline fallback an
insufficient description of current callers.

**Decision needed:** maintain these controls as explicit diagnostic policies,
or retire selected interfaces while preserving reproducible selector maps,
progress decoding and needed containment. Decide observation, execution
selection and synchronization separately; none is an unused performance flag.

**Clearing action:** migrate session/CLI forwarding, complete selector maps,
conflict errors, provenance and shutdown together. Run the router-select,
prefill-profile, queue-drain and flight-recorder test files, plus the bulk
prefill hidden-seed/capacity tests and both sweep/decoder `--help` routes.
If execution or recorder ownership changes, exercise real recorded prefill,
chunk/layer waits, reset/reuse, exception cleanup and persisted decoding; verify
marker and synchronization counts independently. Use the reference's bounded
current-default long-context protocol for any claimed containment removal,
without treating a recorded host stack or one completed pass as root-cause
proof. This disposition changes no runtime default, kernel or arithmetic;
original outcomes remain in the sweep inventory and benchmark artifacts.

### R0281-GDN / R0377 — Prefill selectors and session-owned scratch

`HIPENGINE_GGUF_GDN_PREFILL_MODE` combines automatic/explicit exact selection,
production recurrence routes and live comparison controls. The proposed
release-age collapse would remove modes still used by the prefill comparison,
semantic and trajectory harnesses; their prepare/recurrent/segmented variants
also differ in arithmetic and scratch requirements.

Current selection is resolved through backend metadata, not model identity:

- gfx1100 defaults to `chain_compact_peer_wave32`.
- gfx1151 defaults to `chain_lds32_direct_nonvolatile`; quant/geometry metadata
  selects `chain_peer_cluster8` for Q4_K_M and Q8_0 with `(16,16,128,128)`
  K-head/V-head/K-dimension/V-dimension geometry, and compact-peer for Q4_K_M
  and Q4_K_S with `(16,48,128,128)` geometry. Compact-peer recurrence uses
  the backend's chunk-row capability.
- `exact` resolves the backend's declared exact route. Explicit selectors run
  their registered route or raise a named missing-route error; only automatic
  selection may fall back to a complete fused or split chain.

`_gguf_gdn_prefill_session_mode()` resolves the mode before scratch allocation.
Production scratch freezes it for Q/K capacity, liveness and dispatch; changing
an environment value afterward must not redirect a live session. Compact Q/K
uses K-head capacity while the noncompact peer route uses V-head capacity.
Verify-gate diagnostics opt into superset scratch and mode alternation.
FP16 recurrent-state writers have a separate typed-route contract; removing a
selector must not silently reinterpret resident state or replace an exact
control with reassociated arithmetic.

**Decision needed:** choose which explicit comparison interfaces to maintain
or move to a scoped diagnostic plan, separately from production defaults and
arithmetic promotion. `chain_k2`, register-tree, tiled/LDS and noncompact peer
routes have comparison callers; they are not unused merely because another
route is automatic. Preserve the strict unfused fallback and named capability
errors. R0377's standalone Q8 cluster8 release-age note is consolidated here:
its automatic route and explicit exact control already share this ownership
decision. The FP16-state entry R0380 separately owns storage and speculative
writer migration; consolidating R0377 does not resolve it.

**Clearing action:** migrate selector parsing, backend metadata, registered
prepare/recurrent/segmented siblings, session scratch planning, typed writers
and all comparison CLI choices/provenance together. Run the session-mode,
prefill-routing, GDN geometry and four comparison-harness test files and help
routes. Before changing execution, run primitive exact/parent-parity and typed
state checks plus real `LLM.generate()` at short, segmented and longer prompts,
with selected-route attribution, reset/reuse and allocation cleanup. Any changed
arithmetic requires its declared execution-profile gate; a historical negative
result alone does not revoke an explicit implemented comparison. This decision
changes no runtime, kernel, arithmetic or default; original results remain in
the sweep inventory and immutable worklogs/artifacts.

### R0281-SELECTED-PREFILL — Baseline, shared-X and pair32 ownership

The two selected-expert prefill rows describe one live policy:
`HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=auto|baseline|shared_x|pair32`.
gfx1100 automatic selection uses shared-X; gfx1151 uses pair32, not the old
baseline automatic route. Baseline remains the strict comparison and rollback.
The October pair32 promotion did not repair or re-admit the separate shared-X
body implicated in the gfx1151 repeated-long-context no-progress investigation.
Explicit shared-X remains a diagnostic request, including the replay script's
`--q4-t16-shared-x` interface; it is not an automatic gfx1151 route.

`_COMPACT_MOE_Q4_DUAL_MODE_KEYS` selects gate/up. Pair32 additionally selects
matching Q4/Q5/Q5-qmicro/Q6 T16 down variants through
`_COMPACT_MOE_DOWN_MODE_KEYS`. Missing automatic paired kernels retry the
complete baseline gate/up-plus-down plan; a missing explicit selected mode
raises after registration retry rather than silently changing it. Raw-layout
weights bypass this T16 selector. The prefill-isolation profiles in
R0281-PREFILL-DEBUG depend on these named choices and record them independently.

**Decision needed:** retain the explicit comparison/rollback interfaces or
migrate them into one scoped diagnostic policy while preserving the registered
strict fallback. Pair32's promotion is completed work, not a reason to delete
its production bodies. Retiring shared-X globally would also remove gfx1100's
automatic route; changing that default is distinct from retiring diagnostics.

**Clearing action:** coordinate gate/up and down maps, backend defaults,
registration retries, raw/T16 allocation names, replay and isolation-profile
callers. Run compact-MoE resolver/routing, selected-prefill A/B and profile tests
and script help. Before changing execution, run raw-bit primitive parity for
both activation dtypes and all supported down quants, plus public generation
with short, partial-tile and longer prompts, reset/reuse and teardown. Any
proposal to replace the gfx1151 automatic route with shared-X must address its
recorded failure using the stall reference's bounded repeated-context protocol;
passing pair32 does not clear another body's failure. This disposition changes
no runtime, kernel, arithmetic or default. Original measurements remain in the
sweep inventory and immutable promotion/stall worklogs and artifacts.

### R0281-ROWTILE-CONTROL — Scoped comparison ownership

The ambient `HIPENGINE_GGUF_Q4K_ROWTILE` switch is retired. Exact small-batch
rowtiles remain default-on, with the same raw K-quant and native pack8 dispatch
bounds, explicit WMMA behavior and strict per-row fallbacks.
`q4k_rowtile_session(False)` and the setter retain explicit comparison access
used by dispatch-isolation and GPU oracle tests. They restore the previous
value on context exit; the resolver's explicit argument takes precedence.

**Decision needed:** keep this scoped comparison interface or migrate its
module-global state to execution-owner-local configuration. Removing a dormant
environment switch does not make the existing global setter thread-local.
Coordinate this with other GGUF session control ownership rather than deleting
strict reference kernels or silently altering concurrent calls.

**Clearing action:** migrate resolver consumers and comparison tests together;
exercise nested/exception restoration and concurrent independent owners, plus
singleton, pair and fused pair+SiLU dispatch at values below, within and above
their row bounds. Preserve WMMA precedence and registered strict fallbacks.
Run the rowtile-control and GEMV dispatch unit tests, raw-K and pack8 GPU parity
files and public generation with route attribution, reset/reuse and cleanup.
Original historical results remain in the sweep inventory and artifacts.

### R0281-SELECTED-DP4A / R0283 / R0288 — Explicit activation-adapter ownership

`HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A`,
`HIPENGINE_GGUF_T16_SELECTED_DP4A` and
`HIPENGINE_GGUF_RAW_SELECTED_DP4A` select q8_1 activation adapters for raw/T16
selected-expert projections. They are off in the unscoped runtime, but the
named compatibility generator enables all three in `_LLAMA_COMPAT_MTP_ENV`.
The parity workbench also uses them for independent T16, Q4 and raw candidates.
They are not unused rejected aliases. Q4 T16 c1 fused-SiLU stays on its direct
BF16 path. Split dual calls and supported fallback Q5/raw down calls use
caller-owned q8_1 workspace; registered exact singleton variants take
precedence over the optional Q5 adapter. Mandatory X8 adapters have separate
layout contracts.

R0283's `--verify-dp4a` benchmark flag enables these same three selectors before
loading. The suite's default route plus `-dp4a` adds only that flag; it neither
selects compatibility semantics nor changes the suite's budget set. Omission
and `--no-verify-dp4a` do not clear inherited adapter values. This is an explicit
arithmetic comparison, not an unused rollback or an automatic product default.
Historical Japanese-quality failures and timing comparisons are preserved in
`docs/archive/MTP-LLAMACPP-PARITY-HISTORY.md` and the sweep inventory; they are
not current measurements of every composition using the shared primitives.

**Decision needed:** preserve these diagnostics or replace ambient selectors
with an explicit caller-owned adapter plan across compatibility generation,
bench/profiler and workbench consumers. The selected-call binder currently
refuses certificates for optional raw/T16 adapters it cannot represent; it
does not certify their conversion by borrowing a BF16 or mandatory-X8 record.
That certificate limitation is not permission to remove implemented runtime
capabilities. Coordinate the binder's open invocation-ownership work rather
than independently deleting flags consumed by the named composition.

**Clearing action:** migrate workspace allocation, ordered dual/single
operands, BF16/F32 inputs, actual output types, adapter intents and artifact
provenance together. Run selected-call binding/admission, compact-MoE routing,
parity-workbench and suite tests; cover raw/T16/X8 layouts and the fused-c1
exception. An execution change also needs primitive numerical/profile gates
and real public plus compatibility proposal/verification/commit/reset and
teardown checks. Historical timing and quality results remain in the sweep
inventory and referenced artifacts; this disposition changes no execution or
default.

The redundant Q5-only `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS` override
is retired. Q4 and Q5 DP4A launches share
`HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS`, keeping the 64-thread default and
explicit 32/64/128-thread diagnostics. The same selector serves selected Q4
q8_1x2 SiLU and its dense singleton counterpart; it is not exclusive to the
verifier's Q4/Q5 pair/down calls. Host-only tests compile the actual selectors
for all four launchers and check shared fallback/retired-Q5 precedence without
loading HIP. Include that shared scheduler in the caller-owned adapter-plan
migration; keep both microbench scripts' raw environment provenance and strict
floating-point leaf oracles. Historical scheduler comparisons remain in the
sweep inventory and their result artifacts; two old benchmark repeats are not
a substitute for migrating current callers.

### R0281-DRAFT-CHAIN — Explicit chain policy and host fallback

The ambient `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN` selector is retired.
`device_chain_enabled` and benchmark `--resident-mtp-device-chain` retain
explicit control; compatibility generation and the draft profiler request the
device chain and prewarm directly. The unspecified low-level constructor keeps
its host-chain behavior. Native proposal/complete-cycle execution requires the
device chain; there is no silent downgrade of an explicit native-cycle request.

The host chain is not the retired per-expert MoE-down loop. It uses the same
device `_run_one` computation but owns depth-by-depth control for probability
cutoffs, confidence/hidden/stage diagnostics, depths above the device-chain
capacity and top-k widths beyond its accumulator. Preserve those capabilities
and the live exact host/device-chain oracle.

**Decision needed:** keep explicit host/device comparison control or move
chain selection into one caller-owned proposal plan with its observation and
workspace requirements. Do not delete the host path merely because normal
compatibility generation already chooses device-chain, or label the live
compatibility path default-off because the standalone constructor is different.

**Clearing action:** coordinate constructor, benchmark cache/provenance,
profiler, compatibility owner and native proposal/complete-cycle validation.
Cover explicit true/false/unspecified, prewarm ownership, probability cutoffs,
confidence/hidden/stage capture, depth/capacity and top-k boundaries, plus
proposal/commit/reset/reuse and teardown through public and compatibility
requests. Keep embedding gathers, stream ownership and primitive oracles.
Historical performance claims remain in the sweep inventory and artifacts;
retiring ambient selection changes no chain body or arithmetic.

### R0282-PREFILL-POLICY — Resident prefill diagnostic ownership

The small-row control gap is implemented, not an open feature task.
`_resident_session_wmma_prefill_default()` resolves
`HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL` at session acquisition: unset
keeps WMMA, explicit false disables that preference and empty or invalid values
raise. Resident AR/MTP pools, direct streaming, fallback AR and the compact-DMS
owner use it; the dense speculative MTP route retains its separate literal
selection. The benchmark consumes the same default and keeps explicit
`--[no-]use-wmma-prefill` comparisons.

Session pool keys include the resolved preference. Opposite choices cannot
reuse each other's pooled session; changing the environment does not retune an
already-live session. This selector does not promise that every projection
uses a different kernel: layout, geometry and capability dispatch still apply.
The historical all-GEMV comparison is closed in
`worklog/entries/20260830T164432.870072Z-lhl-gemv-prefill-worse-ab-776d22.md`.
Its timing result is not a reason to remove the supported reference route.
Original grouped-prefill and decode-preference findings remain in the sweep
inventory, immutable worklogs and result artifacts; they are not open
feature-admission or duplicate-selector defects.

**Decision needed:** keep the diagnostic selector or replace it with an
explicit session-owned prefill plan. Preserve working comparison routes and
the production WMMA default; an old negative timing result alone is not a
reason to delete supported GEMV kernels or change another provider's policy.

**Clearing action:** migrate generator/DMS acquisition, pooled and unpooled
sessions, warmup, benchmark precedence and effective-route provenance together.
Cover unset/on/off/invalid values, opposite-choice pool isolation, matching
reuse, AR/MTP/DMS owners, and the dense speculative exception. A runtime change
also needs public requests with route attribution, reset/reuse and teardown;
changing arithmetic or automatic selection requires its applicable profile
checks. The existing parser, pool and benchmark characterization tests bind
this referral without changing runtime behavior.

### R0282-DFLASH-DRAFTER — Dense and post-attention plan ownership

`HIPENGINE_DFLASH_DRAFTER_DENSE` already defaults to WMMA for BF16 and
F32-output dense projections. Non-tile-aligned input width falls back to the
naive implementation; explicit `naive` also keeps a direct numerical reference.
The wrappers serve the older benchmark drafter and DFlash2 native projections.
DFlash2 kernel tests explicitly select the naive reference. Do not infer that
this shared selector is unused from the older benchmark's product status.

`HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM` is different: its resolver is consumed
by the older `NativeDFlashChainDrafter` benchmark layer loop. That loop defaults
to the unfused add/norm chain and permits an explicit fused comparison. The
DFlash2 forward does not consume that resolver. The fused primitive and its
two-output parity oracle remain available independently; its declared hidden
width bound must not be lost by making a geometry-blind selection mandatory.
The retired target fused-LM-head selector is separate from both drafter choices.

**Decision needed:** preserve explicit comparisons or move dense and
post-attention selection into caller-owned drafter plans. Separate a shared
runtime dense policy from a benchmark-only fusion experiment; do not make a
new model-specific timing row a prerequisite for either implemented route to
run, or remove the naive primitive merely because WMMA is the default.

**Clearing action:** migrate the two output-type wrappers, actual DFlash2 and
benchmark consumers, compiler/profile provenance and direct GPU-oracle callers
together. Cover aligned/unaligned input width, output tails, explicit naive,
invalid values and fused hidden-width bounds; retain the strict unfused chain.
Run `tests/test_unit_dflash_drafter_controls.py`, drafter/context tests and
`tests/test_gpu_dflash_dense_wmma.py`, `tests/test_gpu_dflash_add_rmsnorm.py`
plus affected DFlash2 kernel/forward tests. A runtime selection change needs
real provider requests with variant attribution, reset/reuse and teardown;
arithmetic changes need their execution-profile checks. Historical comparisons
remain in the sweep inventory and original artifacts.

### R0285 / R0286 — Draft and target synchronized profiling ownership

`--resident-mtp-draft-sync-stage-timings` is an explicit diagnostic passed to
`Qwen35GGUFResidentMTPDraftRunner.sync_stage_timings`, not a product rollback.
The benchmark requires a resident draft and cycle timing collection. The
standalone `gguf_mtp_draft_rocprof.py` also forwards `--sync-stage-timings`;
its `--record-stage-timings` controls whether the sink is collected.

Extra synchronization requires both the flag and a timing sink. Draft layer
sub-stages and device top-k/gather drain the supplied stream, or the device
when the stream is zero. The final chain drain/readback is required even with
instrumentation off. Synchronized Q6 top-1 uses separate stage1 and
stage2/gather launches for attribution; normal execution keeps its composite
wrapper. These instrumented buckets are not uninstrumented throughput.

`--target-block-sync-stage-timings` passes the target verifier's separate
`sync_stage_timings` control. It requires block verification and cycle timing
collection in `gguf_mtp_bench.py`; `gguf_mtp_verifier_rocprof.py` forwards
`--sync-stage-timings` and `--record-stage-timings` independently. Target
boundaries drain the **whole device**, even when launches use a nonzero stream.
Selected-MoE timing additionally requires a nonempty stage prefix and splits
activation quantization from GEMV. The native target graph rejects synchronized
host timing rather than inserting these drains into capture/replay.

HIP-event stage timing is a separate diagnostic. It preserves queued work and
resolves events after the final drain; rocprof adds kernel/host evidence with
another scope. Their existence does not establish an equivalent replacement
for synchronized per-operation buckets or justify deleting strict primitives.

**Decision needed:** retain the explicit synchronized comparisons, or replace
them with one profiling policy across draft/target runners, benchmark cache
key/provenance, profiler parent/child and named `draftsync`/`allsync` suite
routes. Preserve the distinction between device-wide target boundaries and
stream-aware draft boundaries rather than changing either mode silently.
Keep instrumentation off by default; do not infer retirement from a campaign's
closed status or promote diagnostic synchronization to remove a switch.

**Clearing action:** demonstrate matching bucket/phase coverage and output
under the proposed replacement, including stream zero/nonzero, absent/present
sink, multiple depths, Q6 raw/X8 top-1 staging and event failure/close paths.
Cover target linear/full-attention and exact/DP4A selected-MoE buckets, empty
prefixes, direct state commit and the native graph's host-timing rejection.
Run resident-draft, target timing/selected-MoE, metrics/suite and both profiler
unit tests plus real proposal/verify/commit/reset/close comparisons. Runtime
wiring changes need public and compatibility requests with output and selected
instrumentation recorded; any timing claim compares complete windows under a
matched protocol.

### R0287 — Verifier FP32 arithmetic policy and probe provenance

The `HIPENGINE_GGUF_VERIFY_F32_*` stack has live numerical-probe and
execution-profile consumers:

- `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL` and
  `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM` select residual and attention input
  norm boundaries.
- `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS`,
  `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA` and
  `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT` select projection and attention-output
  boundaries.
- `HIPENGINE_GGUF_VERIFY_F32_POST_NORM` enables post-attention norm FP32;
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_ROUTER`,
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SELECTED_Q8` and
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8` default to true within that
  independently enabled path.
- `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE`,
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN`,
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE` and
  `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN` select weighted-sum accumulation,
  selected output, intermediate and shared output boundaries.

The FP32 combine switch controls weighted-sum accumulation independently.
Selected-down FP32 additionally requires FP32 residuals, its output switch, an
X8 Q5/Q6 adapter and scratch; intermediate/shared-down add their own controls.
These paths preserve BF16 mirrors where consumers require them.

`HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT` is an integer prefix length,
not a layer-list parser. Blank/unset covers every layer; zero rejoins BF16 at
the first layer. The forced-target probe records its raw value and the
`HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS` selector with the other
arithmetic controls. Historical layer-local observations remain in the
original sweep source and `benchmarks/results/2026-07-02-mtp-target-f32-*.json`
artifacts; they do not establish model-wide promotion or rejection.

The gfx1100 MoE production profile's binder enables residual and post-norm
FP32; the strict binder disables both. The dense production binder uses its
own strict-based composition. Physical MoE C2 has a request-scoped numerical
context that suppresses residual FP32 without rewriting the ambient value.
These are separate owners, not obsolete standalone probe flags. Preserve the
registered strict/production bindings and explicit probe comparisons during
cleanup. R0281-DENSE-Q8 owns raw sidecar placement and projection adapter
selection; this row owns numerical composition, not duplicate residency.

**Decision needed:** preserve the explicit per-boundary comparisons, or move
them into a runner-owned arithmetic plan with registered-profile bindings and
probe serialization. Do not delete the broad stack merely because one
historical composition failed, or enable every FP32 boundary to remove flags.
Candidate promotion and diagnostic retention are separate decisions.

**Clearing action:** migrate buffer materialization, BF16 mirrors, prefix
rejoin, layer/FFN consumers, profile binders, physical-C2 context and probe
metadata together. Run `tests/test_unit_qwen35_gguf_verify_f32_*.py`, dense-Q8
routing, profile/environment isolation, invocation-contract and forced-target
probe tests. Runtime changes require real strict/production requests, teacher
logits at matched contexts, prefix/intermediate/state checks and
verify/commit/reset/cancel/close comparisons. Changed-arithmetic defaults need
the execution-profile numerical/task gates; an old token-margin slice is not
that gate. Token-embedding precision remains separately inventoried.

### R0291 / R0293 — Q8T16 rowtile selection and comparison ownership

`HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE` and
`HIPENGINE_GGUF_Q8_T16_PAIR_COL8` have live rollback and diagnostic consumers.
Both gfx1100 and gfx1151 packed decode set
`GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS=8`; the original gfx1100 transfer
blocker is no longer current. Below that floor, gfx1151's separate
all-projection policy can select the 16-column rowtile body. The pair helper
is limited to BF16 Q8T16 `in=2048, out=8192+4096`, rows greater than one and
no explicit thread override. Other pairs retain the exact per-row owner.

Explicit pair `0` disables rowtiling even when the broad policy is enabled.
Explicit col8 `0` keeps rowtiling but selects the 16-column body; col8 alone
cannot enable a pair. Unset column selection follows the pair floor, not the
broad all-projection policy. The verifier profiler's
`--q8-t16-pair-rowtile` sets only the pair selector; omission preserves
inherited values. The batch-route gate clears and restores both selectors
and the broad override to exercise package policy. The floor and broad-policy
context managers restore prior values, but their backing state is
module-global, not isolated between concurrent execution owners.

R0293's `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` controls BF16 `in=2048`
singleton and triple rowtiling and the pair fallback below its own floor.
It does not disable the independent `(5120,1024)` singleton shape policy.
Packed decode sets the broad session value from each backend's width floor:
four on gfx1151, disabled on gfx1100. The gfx1151 MoE strict profile writes
`0`; production removes the environment override. That binder owns only the
broad selector, not the pair or column selectors. Keep that distinction when
interpreting a profile or changing its policy.

The verifier profiler and named `q8rowtileall` suite comparison can force the
broad selector on. The batch-route gate separately supports an explicit broad
candidate and the package-default bundled candidate, restoring caller values
on exit. These are live numerical/comparison consumers, not evidence that the
production path is default-off. Historical strict-teacher, repeat and timing
results remain in
`benchmarks/results/2026-08-16-gfx1151-q8t16-batch-route-retained.json` and
the sweep source; no fresh speed or task-quality claim is made here.

**Decision needed:** preserve these explicit comparisons or migrate the
ambient selectors and session globals into a caller-owned kernel plan across
packed decode, execution-profile binders, verifier profiling, named suite
routes and the batch-route gate. Keep the independent singleton shape policy
separate. Deleting the 16-column or per-row bodies would remove live fallbacks,
not just switches. Keep the already-enabled package routes while deciding
configuration ownership.

**Clearing action:** migrate the floor, broad-policy interaction, column
selection, explicit thread/shape fallback, profile binding and profiler
parent/child provenance together. Run single/pair/triple launch,
nested/exception restoration, strict/production binder, suite, package-policy
and profiler CLI tests, including widths below, at and above both floors and
unrelated widths. A runtime migration also needs both backend GPU bit-parity
files and public packed generation/serving with route attribution, width transitions,
reset/reuse and teardown. Historical implementation/server comparisons remain
in the original sweep source and
`benchmarks/results/2026-07-20-gfx1151-gguf-q8t16-pair-col8-c8-retained.json`;
they are not fresh measurements of this disposition. No execution, arithmetic
or default changes here.

### R0380 — Recurrent-state storage controls and speculative writers

FP16-state promotion is completed work: gfx1151's plain Q4_K_S default uses
FP16 storage with FP32 accumulation, while `HIPENGINE_GGUF_FP16_RECURRENT_STATE=0`
selects strict FP32 storage. The numerical and serving artifacts remain in the
original sweep source and August 20/27 results; their timings are not fresh
measurements of this disposition. The FP32/FP16 comparison harness still uses
the override, so deleting it would compare the default against itself.

The runner freezes storage dtype before allocation. Decode, compact-peer
single/segmented prefill, decode-order row writers, fused output/cast and
consumer-owned rollback snapshots must use that dtype. An environment mutation
cannot redirect an allocated session. Prefill modes without an FP16 recurrence,
prefill-score capture and regular fused prefill without a typed writer still
raise capability errors. Keep these checks and registered FP32 references;
removing a selector does not make unlike state buffers interchangeable.

SPECDEC2 has an eager FP16 target route whose current binder requires a complete
non-fallback production manifest selecting the typed chain-row writer. This is
not the legacy tree/chain path, which rejects FP16 state. Native target graphs
remain excluded after the recorded accepted-row divergence; the native device
proposal path still raises `FP16 recurrent state device proposal requires eager
selected commit`. Producer-folded output/snapshot aliases are separate from
consumer-owned typed rollback. Historical capacity-one economics do not prove
realized-group ownership for a capacity-greater-than-one service owner.

**Decision needed:** retain the storage override as a supported comparison or
migrate it into a caller-owned execution plan. Keep frozen allocation dtype;
retiring an environment name is not a reason to retire that invariant. Separate
this interface cleanup from implementing typed speculative selected commit,
repairing the observed native-graph divergence and reviewing profile/admission
ownership against capability-only product rules. Missing benchmark rows are
not runtime capability failures and must not become new restrictions.

**Clearing action:** migrate storage resolution, allocation pricing, all writer
selection, profile binding, snapshots and comparison provenance together. Run
`tests/test_unit_qwen35_gguf_gdn_prefill_routing.py`, session-mode,
scratch-plan-boundary, FP16-state batch-gate and MTP2-seam tests, including nested
failure restoration and explicit dtype after environment changes. On HIP run
`tests/test_gpu_qwen35_gguf_gdn_fp16_state.py` and public generation/serving for
FP32/FP16, short/segmented/long prompts, sparse and changing widths, reset/reuse,
failure rollback and teardown. Any graph repair must compare accepted-row
state against eager, including the recorded failing layer region. Any arithmetic
promotion needs its execution-profile gate; route availability and explicit
requests follow capability contracts independently. This disposition changes
no runtime, default, arithmetic or speculative admission policy.

### R0381 — Short-C1 attention thread comparison ownership

`GGUF_SHORT_C1_BATCH_ATTN_THREADS` selects gfx1151's 1024-thread short-C1
context-batch leaf. The exact 256-thread leaf is still a separate registered
reference. `HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS` is a live control in
`scripts/execution_profile_gguf_c1_threads_gate.py`, which compares the two
reduction trees. Other model/context routes, including the separate short
split-K policy, do not make this runtime-width entry point unused.

The runner caches its first resolved leaf. The gate explicitly resets that
cache at each strict/candidate boundary; changing the environment alone does
not redirect a live runner. Preserve that distinction when moving diagnostic
configuration, and keep the exact reference rather than treating 1024 threads
as byte-identical arithmetic. Historical promotion results stay in the original
sweep source and artifacts; this disposition measures no new numerical or
performance result.

**Decision needed:** retain the thread-width comparison interface or migrate it
to an explicit caller-owned diagnostic plan before retiring the environment
reader/runtime-width entry point. R0427 records the other live fixed256
consumers. Do not collapse the body
to a single constexpr while the numerical gate still selects both widths.

**Clearing action:** migrate resolver, leaf-cache boundaries, gate provenance,
strict registration and callers together. Run
`tests/test_unit_gguf_c1_threads_control.py`, decode-repack-dispatch and
layer-execution tests plus the gate's `--help`. Runtime changes need the full
thread-gate suite and public short-C1 requests on gfx1151, including both sides
of the context route boundary and unrelated lengths, repeated strict/candidate
order, output/state comparison and teardown. Keep the package default and
strict fallback during interface migration; any changed reduction default
requires its execution-profile numerical/task gate.

### R0382 — Long-context staged verifier and scalar-reference ownership

Dense multi-row linear-attention verification uses the staged chain above the
split boundary when `_gguf_staged_linear_rows_long_enabled()` selects it. The
package default covers plain gfx1151 Q4_K_M;
`HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG` selects staged or scalar rows in either
direction. Other current defaults use
the scalar-AR reference. Short-context staging, single-row decode and staged
full-attention rows have independent selection and must not be conflated with
this linear-attention policy.

The scalar branch runs attention and FFN once per row. A long-context numerical
gate that takes that branch compares scalar arithmetic with itself; zero KL
there does not certify staged multi-row arithmetic or speed. Historical
boundary differences and the later staged-route repair are preserved in the
original sweep source and immutable results. The old blanket-rowwise comment
is corrected, not the execution policy.

**Decision needed:** retain the explicit comparison or migrate the selector and
row-wise orchestration into one caller-owned strict/staged plan. Keep the
registered strict primitives and Conv/GDN/KV transaction ownership. Retiring the
branch must not make strict requests silently use staged arithmetic. Review the
existing artifact-derived default policy separately against capability-only
product rules; do not perpetuate the old instruction that every runnable
artifact needs its own benchmark packet. Arithmetic-default promotion and
runtime capability remain separate contracts.

**Clearing action:** migrate policy binding, strict registry chain, row-scratch
views, initial snapshots, selected-state commit and gate route attribution
together. Run `tests/test_live_gguf_native_spec_cycle.py` staged/strict routing
nodes and `scripts/gguf_mtp_long_context_gate.py --help`. Before changing
execution, exercise the gate with target graphs disabled and public generation
at split-boundary straddles and unrelated contexts; assert staged/scalar call
counts, full outputs/state, every acceptance prefix, reset/reuse, failure
rollback and teardown. Native graph/submission economics remain separate work.
This disposition changes no route, default, arithmetic or admission behavior.

### R0385 — Observed target/proposal sentinel crossing repair

The original dense Qwen3.6 Q4_K_M crossing test remains unconditionally skipped:
`test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback` in
`tests/test_live_qwen35_gguf_mtp_e2e.py`. The August merge reproduced invalid
INT32_MAX tokens and corruption of the following GPU case on pristine upstream.
That is an observed failure, not absent benchmark evidence. Keep
`target_graph_proposal_handoff_boundary_miss`: the readiness check prevents a
device-only proposal at the final cached bucket row before provider mutation.
Host-materialized proposals and otherwise supported target routes still run.
The old assertion that all automatic long-context MTP is disabled is not the
current routing contract; R0384 describes current target-graph ownership.

September's private FP32 Conv handoff, separate serial journal and by-value
proposal metadata upload repaired independently exercised Qwen3.8 failures.
Those results do not independently clear the original Qwen3.6 crossing or
prove that its cause was exclusively cross-stream publication. Do not delete
the guard or lift the skip by inference from those fixes.

**Remaining repair:** reproduce the original crossing in an isolated process
on a compatible, idle device; trace proposal payload publication, stream order,
Conv/GDN scratch and target/journal ownership. Because the recorded failure can
corrupt later GPU work, do not unskip it in a shared broad-suite process.
A failing reproduction must retire before any following health check or reset;
coordinate recovery if the device remains unhealthy.

**Clearing action:** first migrate the skipped node into a bounded subprocess
regression that can explicitly execute the original body with its HIP/model
preconditions intact. Run that isolated regression against the original local
model and both sides of 1024, plus unrelated contexts and B1–B7, before removing
the skip or handoff guard. Compare host/device proposals, every acceptance
prefix, payload validity, Conv/GDN/KV/cursors, next-cycle continuity, reset,
cancellation and teardown; follow it with a fresh-process health request.
Then run the complete MTP E2E, long-context, lifecycle/API/load packet and an
actual `LLM.generate()` request confirming the intended path. The long-context
harness's host-proposal mode alone cannot certify device handoff. Keep
`tests/test_unit_qwen35_gguf_mtp_session.py` boundary/submission checks,
`tests/test_live_gguf_native_spec_cycle.py` launch/binding checks and the delayed
metadata-upload test in `tests/test_live_qwen35_gguf_nextn_provider.py`.
This sweep adds host guard coverage only; it does not claim the crossing fixed.

### R0387 — Dense gfx1100 packed-step profiling follow-up

The August dense Qwen3.8-27B packed-versus-scalar comparison is historical
evidence, not an active token-correctness repair. Its source artifact remains
`benchmarks/results/2026-08-19-concurrency2-qwen38-27b-oldproto-c1-c4-old-vs-new-diagnostic.json`;
the classification correction is preserved in
`worklog/entries/20260819T092204.420429Z-lhl-qwen38-27b-packed-drift-correction-493ec4.md`.
The current F1 harness treats cross-width generated-ID equality as diagnostic
under `production` and keeps control/length failures binding. T2 identifies
width-dependent arithmetic; it does not waive numerical/task or state-ownership
gates, and fluent text alone cannot certify them. Preserve strict and
batch-invariant contracts separately.

**Remaining work:** profile the dense packed decode step on gfx1100 before
choosing a kernel, state-transfer or submission change. The old comparison did
not show aggregate scaling, but it is not a measurement of today's route.
Keep default packed execution and its scalar comparison; do not infer a
current speed result or disable the route from the old timing.

**Clearing action:** inspect `scripts/server_f1_concurrency_bench.py --help`,
then collect matched current c1/c2/c4 and neighboring/unrelated logical-width
requests on one gfx1100 host with model, quant, KV policy, context, sampler,
queue count and execution profile recorded. Confirm actual public packed
submission and slot/state ownership. Profile warmed cache-only execution with
`rocprofv3 --kernel-trace`; separate projection/attention work, state/KV copies,
host submission and synchronization, and separate capture from replay. Check
teacher-forced numerics/tasks for changed arithmetic and exact ownership,
ragged/shrinking groups, cancellation, repeats and cleanup. Preserve complete
request-wall and per-request latency alongside aggregate throughput.
Run `tests/test_unit_server_f1_concurrency_bench.py`,
`tests/test_unit_gguf_ar_decode_control.py` and execution-profile contract tests
before changing the comparison interface. This disposition adds no timing,
numerical result, production promotion or runtime change.

### R0388 — Dense budget diagnostic interface ownership

The sequence schedule and timing-adaptive controller now live in
`scripts/mtp_budget_diagnostics.py`, not the runtime speculative package. The
dense suite still exposes `--budget-sequence` and `--adaptive-budget`; its
fresh request-owned policy instances preserve existing comparison metadata.
`hipengine/speculative/mtp_budget.py` contains only the observation record and
policy protocol consumed by the optional dense-session injection seam.
Production does not import the concrete diagnostic controllers. Fixed-budget
serving, independent graph buckets and strict transaction fallbacks are
unchanged. The historical adaptive result remains evidence of that experiment,
not a reason to remove transition coverage or a claim about a new selector.

**Decision needed:** retain the dense-session policy seam and CLI diagnostics as
supported comparison interfaces, or replace them with a test/benchmark-owned
cycle driver. The implementations have moved, but the call sites are live;
do not remove their flags while silently dropping the requested experiment.
Public candidate-depth selection is a separate shared-planner contract.

**Clearing action:** migrate callers, request-start/choice/post-commit
observations, output-tail clipping, cancellation ordering and summary metadata
together if replacing the seam. Run `tests/test_unit_mtp_budget_policy.py`,
`tests/test_unit_qwen36_dense_gguf_suite.py` and session cancellation/bucket
checks, then exercise the suite with a complete B1/B2/B3 transition schedule,
every acceptance prefix, reset/failure/cleanup and a public fixed-budget
request. A new timing-adaptive default needs its own declared policy and
comparison against fixed budgets; this extraction does not change or promote
one. Keep the named diagnostic controls until their owner is decided.

### R0389 — Standalone draft-hidden diagnostic ownership

`Qwen35GGUFDraftHiddenPolicy` and the dense suite's `--draft-hidden-variant`
remain live. The standalone decoder defaults to pre-output-norm and can
explicitly normalize both prompt chunks and proposal seeds once into owned
buffers. Its manifest keeps target-commit hidden pre-output-norm and chained
NextN hidden post-output-norm. These are different ownership points, not a
single global model convention. The public NextN hidden-seed contract has its
own precision/provenance rules; do not infer its behavior from the standalone
policy's default.

**Decision needed:** retain this supported diagnostic constructor/CLI interface
or migrate the post-norm transform into a benchmark-owned adapter. No second
model campaign is required to run an explicit implemented convention. Do not
remove the manifest or let an explicit post-norm request silently use pre-norm.
A public model/provider convention must bind its actual hidden contract,
not select from prompt text or observed acceptance.

**Clearing action:** migrate constructor, manifest/result metadata, prompt
chunk transform, proposal normalization, stream retirement and owned-buffer
cleanup together. Keep the target journal's commit/rollback tap unchanged.
Run `tests/test_unit_qwen35_gguf_mtp_session.py`,
`tests/test_unit_qwen35_gguf_hidden_seed_contract.py`,
`tests/test_unit_qwen35_gguf_mtp2_seam.py` and
`tests/test_unit_qwen36_dense_gguf_suite.py`; exercise both explicit taps in the
dense suite, prompt chunks on both sides of the chunk bound and unrelated
lengths, first/following proposals, repeat/reset, failed priming, cancellation
and teardown. Verify the public fixed contract separately. A changed hidden
convention is a numerical/policy change and needs its applicable task and
execution-profile gate; this disposition changes neither convention nor API.

### R0392 — Optional marker-free profiler interface

The grouped-prefill dead-route premise is obsolete: gfx1100/gfx1151 declare
packed prefill, mixed-length compatible rows group, and serial execution
consumes the other selected chunks. Failed unexecuted peers are explicitly
rewound; completed buffered chunks are not. The worklog slug/title defects and
verifier INT32_MAX padding are also closed. The server benchmark now emits
independent per-arm route-counter snapshots and deltas. Closure evidence lives
in the R0392 sweep worklogs; these are not reasons to keep this entry open.

**Decision needed:** add a wrapper-level marker-free profiling mode when the
ROCTX library is absent, or keep the explicit child-mode procedure in
`RDNA3-TUNING-GUIDE.md`. Kernel tracing itself needs no ROCTX; marker-scoped
attribution does. Do not silently replace a requested marker-scoped result with
an unscoped trace. The wrapper's explicit missing-library error is not a
runtime model capability restriction.

**Clearing action:** either adopt the documented child-mode procedure as the
interface, or implement an explicit no-marker mode that records its scope and
uses the matching rollup. Test missing/present libraries, explicit override,
child forwarding and trace-marker matching. Coordinate shared discovery with
R0393; preserve each wrapper's dependency-path and measurement contracts.

### R0393-Q6 — Grouped-Q6 rollback interface and gfx1100 public engagement

The runner no longer matches model names or aggregate GGUF file-type labels.
`GGUF_Q6_DP4A_GROUPED_TARGET_REQUEST_COUNTS` declares automatic physical-C8
selection in the gfx1100 package. The planar-Q6 adapter checks actual rows,
input width and bounded workspace. Strict/production choices are owner-local;
explicit workspace scopes do not re-read the process environment. Keep those
resource contracts and the registered strict BF16 primitive.

**Decision needed / removal condition:** keep
`HIPENGINE_C8_Q6_DP4A_GROUPED` as a supported production rollback and standalone
diagnostic, or replace it with an explicit session/profile interface. Remove
the environment reader and reference only after the replacement preserves
those controls; do not remove the workspace scope as temporary policy debt.

**Hardware follow-up:** run a real public packed-C8 request on gfx1100 and
record selected Q6 variant, grouped-kernel engagement, repeatability and
allocation drain. gfx1151 normal public requests keep their existing BF16
selection. Explicit peer diagnostic scopes bound successfully, but selected
verifier variants bypassed the grouped decode wrapper; those runs do not prove
gfx1100 engagement. This is a validation follow-up, not a runnability gate.
Historical promotion evidence remains in
`../benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json`.

### R0393-Q5 — Optional planar-Q5 sidecar lifecycle

The planar-Q5 route is implemented and executable. The public dispatch and
allocation-formula defects are repaired; an explicit
`q5_raw_mmq_target_session(planar_dp4a=True, ...)` owns its resolved choice,
library and scratch. Dispatch selects it for compatible R24 recurrent-output
weights with a resident `qmicro_planar` sidecar; R32 keeps the source-layout
owner. Keep the existing strict/raw primitives and resource checks.

**Decision needed / removal condition:** retain
`HIPENGINE_C8_Q5_PLANAR_DP4A` as a supported optional resource choice, replace
the eager persistent sidecar with a request-owned allocation policy, or retire
the route. Normal loading leaves that additional resident allocation off;
this is a persistent-memory tradeoff, not missing numerical evidence. Do not
remove the kernel, loader sidecar, registry entry, session, tests or diagnostic
merely because another path is automatic. Operator non-use has not been
established, so the old campaign-cycle deletion condition is not satisfied.

**Clearing action:** choose the supported interface and allocation lifetime.
If retained, document it at its policy definition and remove the temporary
lifecycle entry. If replacing allocation, validate exact planned/converter
bytes, insufficient resources, session isolation, R24/R32 selection, repeated
public use and complete teardown. If retiring, remove the whole route in one
unit and preserve the historical artifacts. A future public gfx1100 C8 request
must record actual selected-kernel engagement rather than infer it from the
sidecar's presence.

Historical numerical and complete-model results remain in
`../benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p2-q5-planar-dp4a-l4-numerics.json`
and
`../benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p2-q5-planar-dp4a-retention-e2e.json`.
The original leaf harness calls the planar kernel directly; the early public
dispatch deadlock did not make that leaf owner-versus-owner. See the correction
and source evidence in
`../worklog/entries/20261029T114236.423489Z-cleanup-refactor-r0393-q5-lifecycle-f07019.md`.

### R0402 — Singleton and packed target-verifier ownership

`Qwen35GGUFMTP2Adapter` still has two target owners. The native singleton
verifier uses `_ensure_active_singleton_target_verifier` and graph-bucket
preparation; gfx1151 singleton requests still use it. Backend-declared physical
C1 requests and wider groups use `_execute_target_frontier_batch` with
`verify_target_blocks_batch`. That packed target pass is eager, but its device
proposal path enqueues acceptance and selects state on device before reading a
bounded summary. Host-materialized proposals still have a host-result path.
Do not describe every packed request as host-accepted or delete either owner
as dead code.

**Decision needed:** keep the singleton verifier as a supported specialization,
or converge on the packed target owner. If keeping it, document the ownership
boundary and close this duplication entry. If converging, implement packed
target graph capture and compare both routes through the same public C1
request before removing singleton dispatch and graph-bucket preparation.
Preserve backend capabilities, sampled acceptance, DMS handling, cancellation,
selected-row state commits and singleton survivors in nonzero resident slots.

**Clearing validation:** run
`uv run pytest -o addopts='' -q tests/test_unit_qwen35_gguf_mtp2_c1_physical.py tests/test_unit_qwen35_gguf_mtp2_seam.py tests/test_unit_gguf_mtp_native_state_attribution.py`,
then repeated public `LLM.generate()` or HTTP requests confirming the intended
owner, graph replay, matching outputs and complete allocation drain. Convergence
also needs a matched C1 route-comparison harness; the existing server benchmark
does not expose that comparison. The old timing artifact is historical, not a
measurement of the device-accept path:
`../benchmarks/results/2026-09-07-w7900-packed-route-c1-stage-timing-diagnostic.json`.

### R0403 — DMS and INT8 hidden-plane alias rollback interfaces

Both route-scoped alias controls default on. `HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS`
controls DMS `layer_outer`; `HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS` controls
non-DMS `layer_outer_shared_oracle`. They are independent controls, not duplicate
readers. `_gguf_bulk_prefill_hidden_plane_count` owns the allocator and capacity
estimator's shared answer. Workspace growth preserves the resolved plane count;
the geometry-wide one-plane policy still takes precedence over either rollback.

**Decision needed:** retain both rollback interfaces, replace them with an
explicit workspace policy, or retire them together. The INT8 control is also
the two-plane comparison arm in `scripts/gguf_resumable_prefill_gpu_proof.py`
and `scripts/qwen38_int8_alias_realprompt_ab.py`; preserve an explicit comparison
interface if removing the environment readers. Missing operator-use records do
not establish that either interface is unused.

**Clearing action:** record the supported interface. If replacing or retiring,
update allocator/estimator agreement, capacity and growth tests, DMS/non-DMS
isolation, proof configuration and `docs/ENVS.md` together. Run
`uv run pytest -o addopts='' -q tests/test_unit_int8_layer_outer_hidden_alias.py tests/test_unit_dms_layer_outer_prefill.py tests/test_unit_qwen35_gguf_capacity_route_terms.py tests/test_unit_gguf_bulk_prefill_workspace_release.py tests/test_unit_gguf_bulk_prefill_deferred_hidden_capacity.py tests/test_unit_gguf_resumable_prefill_gpu_proof.py`,
then public generation through both affected routes confirming actual plane
ownership, output agreement and complete teardown. This disposition changes no
allocation policy and does not reopen completed alias promotion.

### R0404 — Direct INT8 prefill evaluation after dtype repair

`HIPENGINE_GGUF_INT8_PREFILL_DIRECT=1` selects oracle-free retained-INT8
prefill where no BF16 mirror is present; the default keeps the oracle route.
The direct lifetime plan uses chunk-sized hidden storage. Per-token/head
attention selects `flash`, `wmma` or `sequential`; automatic flash selection
requires its supported geometry and FP32 scales. Hadamard-group32 uses its
separate direct leaf.

The GGUF gate producer writes BF16, while all these direct leaves consume an
FP16 gate. The runner now converts that buffer in place before attention and
converts the Hadamard leaf's FP16 output back to BF16 before the projection.
CPU numerical boundary tests, GPU CPU-reference comparisons and repeated public
gfx1151 sequential/flash/tail-four requests pass. The Hadamard branch is not dead
code. No kernel arithmetic or default is changed by the adapter repair.

**Remaining evaluation:** the historical direct-prefill rejection was measured
before these adapters. It records an observed failure, not proof that the
entire discrepancy is inherent INT8 quantization error. Preserve the artifact,
but rerun the calibrated same-prefix evaluator before making a new quality
claim or changing the arithmetic default. Start with
`python scripts/execution_profile_gguf_int8_direct_prefill_gate.py --model "$GGUF_MODEL" --backend hip_gfx1100 --kernel flash --min-prompt-rows 4096 --decode-steps 24 --repeat-runs 3 --use-wmma-prefill --use-gemv-decode --json /tmp/int8-direct-prefill-repaired.json`
in the ROCm environment. Complete the execution-profile category/heldout,
shape, repeat/isolation and task gates, including separate tail-four coverage;
this per-token evaluator does not establish Hadamard quality. Preserve explicit
direct execution and the oracle path while resolving that numerical choice.

Historical evidence:
`../benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-gate.json`
and
`../benchmarks/results/2026-08-15-qwen38-27b-int8-kv-quality-frontier-runtime-blocked.json`.
Repair and current validation:
`../worklog/entries/20261029T143522.416005Z-cleanup-refactor-r0404-direct-int8-dtypes-9a1059.md`.

### R0406 — RX 7900 XTX server capacity refresh

The eager full-context packed reservation is removed. The remaining eager
workspace lease uses only the packed context floor per slot; C1 with MTP and
prefix cache off omits that lease entirely. Longer or wider private KV is
charged against the shared pool budget. Keep lease, rebind and page-map
regressions; no additional reservation change is required by this entry.

**Deferred hardware validation:** refresh the RX 7900 XTX single-request server
context bound. This session has gfx1151 hardware, not the requested XTX;
substituting its result would not establish that bound. On the XTX host, start
with a short, multi-slab allocation probe, for example:

```bash
python scripts/gguf_server_allocation_probe.py --model "$GGUF_MODEL" --kv-storage int8_per_token_head --kv-scale-dtype fp32 --kv-scale-granularity per_token_head --speculative-mtp-serving off --prefix-cache off --max-active-requests 1 --max-context-tokens 54272 --prompt-rows 2048 --decode-tokens 8 --json /tmp/xtx-server-capacity-54272.json
```

Select the XTX with the host's device selector and run in its ROCm environment.
Record resolved server policy, raw readiness/capability/response documents,
allocation owners and teardown. Bracket with short probes only, then confirm
the selected context with the requested full-depth workload. The historical
54,272-token pass is not a current maximum. Preserve the original evidence in
`../benchmarks/results/2026-09-07-rx7900xtx-int8-repair-capacity-audit.json` and
`../benchmarks/results/2026-09-09-rx7900xtx-server-int8-route-census.json`.
No fresh capacity result is claimed here.

### R0407 — Slot-local AOTriton rollback interface

`HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON` defaults to `1`; `0`
selects native attention only for slot-local transient-BF16-oracle layers.
Backend capability, row crossover and wrapper-failure behavior remain separate.
The route's admission history is complete; do not reopen promotion to decide
whether this rollback interface is still useful.

The full-logit evaluator and route A/B example use the native arm. Their old
unset controls stopped selecting that arm after the runtime default changed.
Both now bind explicit off/on values, and the evaluator records its actual
strict and candidate environments. Its synthetic contract tests verify distinct
arm policy through the real capture driver; they do not establish GPU kernel
identity or a new quality verdict. Historical artifacts are unchanged.

**Deferred interface decision:** keep this layer-scoped rollback, or replace its
native-comparison capability before deleting the environment reader and helper.
A replacement must update both evaluator manifests/capture and the route A/B
example, preserve strict native fallback and backend/row constraints, and run
real public generation confirming selection, repeatability and teardown.
Changing the flag alone would erase a working comparison arm, not finish cleanup.
Implementation and focused evidence:
`../worklog/entries/20261029T151518.411951Z-cleanup-refactor-r0407-aotriton-evaluator-4f9845.md`.

### R0410 — Fused grouped-prefill combine adjudication

`HIPENGINE_QWEN4_EXP_FUSED_COMBINE=1` selects the implemented grouped-prefill
routed-sum/shared-gate fusion in `qwen4_exp_runner.py`. The default keeps
`weighted_lanes_sum_out_bf16_f32w` followed by
`shared_gate_combine_f32_shared_out_bf16`; the latter incorporates the shared
output's BF16 conversion. The candidate instead converts shared output first,
then launches `weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w`.
Preserve both supported routes and their registered strict primitives.

**Deferred decision:** retain the candidate and its explicit interface, promote
it after a current comparison, or retire the complete route. The corrected
historical artifact records four Python processes and four model residencies,
not one residency. It does not establish a binding loss; do not delete the
candidate or describe the old timing as a promotion/rejection gate.

**Clearing action:** if evaluating promotion, add an engagement-verified
fused-combine package to `scripts/qwen4exp_halo_box_campaign_ab.py` or an
equivalent same-process harness. The existing harness does not expose this
comparison. Measure both settings in one model residency across the complete
category/shape suite and heldouts, charging the separate cast and comparing
against today's cast-folded default. Verify exact outputs/state/KV,
repeatability, graph recapture where applicable and public `LLM.generate()`
selection/teardown. Run
`uv run pytest -o addopts='' -q tests/test_gpu_qwen4_exp_pf4_lever2_fused_combine.py`
in the ROCm environment before the model comparison. If retiring by product
decision, remove the runtime branch, wrapper, registry entry, selector and
candidate-only tests together, preserving strict primitives and history.

Historical diagnostic:
`../benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf4-lever2-fused-combine-whole-model-ab.json`.
This triage changes no execution behavior and claims no fresh performance or
numerical result.

### R0412 — H256 generic-wave diagnostic retirement

The H256 scalar sparse-prefill choice is now runner-owned:
`qsa_h256_prefill_mode` selects `parent`, `wave`, or `page256`. Production
publishes `page256`; strict publishes `parent`. The retired
`HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL` has no effect. Diagnostic scopes
restore the runner's selection on exit, including failures. Shape/registry
checks, WMMA precedence, head grouping, the score-product register boundary
and the separate ordered decode parent are unchanged.

**Deferred decision:** retire generic-wave bisection support when it is no
longer needed, or keep it as an explicit diagnostic. The lead's accepted
prefill/decode tradeoff is not reopened by this control migration. Decode
performance is separate optimization work, not a reason to undo that choice.

**Clearing action:** decide whether the generic-wave comparison remains useful.
If retiring it, remove the `wave` scope value and its campaign/state-gate
packages together; preserve page256 and the strict `strict_rows_spans` fallback.
Run the H256 scope/runtime tests, existing GPU H256 parity tests and public
`LLM.generate()` with the scalar fallback engaged. Any decode optimization
must first profile the current path and follow `OPTIMIZATION.md`.

Implementation and validation:
`../worklog/entries/20261029T163722.751625Z-cleanup-refactor-r0412-h256-prefill-2bd929.md`.

### R0415 — Register-state GDN decode follow-up

Register-state serial prefill is now runner-owned through `gdn_register_prefill`:
production on, strict off. The retired environment selector is ignored. The
rows>=2, Hk16/Hv48/D128 and registry guards, registered strict serial fallback,
suffix arithmetic and decode routes are unchanged. Diagnostic scopes restore
the owning runner on exit and isolate the serial comparison from tiled, peer
and wave-norm owners.

**Deferred work:** optimize decode without reopening the lead's accepted
prefill-first decision. The original entry recorded a p4096 decode loss amid
drift alongside the accepted whole-request improvement; this migration makes
no new timing claim and does not attribute that result to temperature. The
accepted resource review and rejection of per-column/every8 scheduling barriers
still stand; their original measurements remain in Git history.

**Clearing action:** profile today's decode path under `OPTIMIZATION.md` before
choosing an optimization. Run the full profile gate and matched whole-model
prefill/decode comparison for any arithmetic or performance candidate, keeping
the serial strict fallback. Remove this follow-up only after the decode work
is completed or the lead explicitly retires it.

Implementation and validation:
`../worklog/entries/20261029T172026.382460Z-cleanup-refactor-r0415-gdn-register-f765dd.md`.

### R0421 — Separate PARO and GGUF attention-key semantics

`hipengine/runtime/qwen35_paro.py` still binds its four paged-KV/attention
resolutions to `_PAGED_KV_REGISTRY_BACKEND = "hip_gfx1100"`. The gfx1151 package
maps `(paged_attn_decode, w4_paro, bf16_context_batch_c1_exact_spans)` to the
fixed256 generic reduction, while PARO uses the c1-exact body. These consumers
share a key but have different whole-model batch-versus-c1 behavior.

**Deferred work:** separate the consumer contracts before replacing the pin.
The [recorded same-host comparison](../worklog/entries/20260912T044256.896835Z-lhl-paro-paged-kv-backend-pin-load-bearing-e64405.md)
found deterministic PARO divergence when selecting the gfx1151 override;
current-pin and per-row controls matched independent c1. This is an observed
failure, not missing benchmark coverage. Preserve the pin and its regression
test; do not substitute the session backend unconditionally.

**Clearing action:** define distinct PARO/GGUF variant semantics, then compare
the replacement against current-pin, repeated replacement and `per_row` arms
using `scripts/qwen35_batch_retained_bench.py --model <PARO-model> --fixture
<derived-1024-token-fixture> --prompt-length 512 --batch-size 2 --decode-tokens
128 --warmup-decode-tokens 8 --batch-decode-full-attn-path
{per_row,native_batch}` on the same gfx1151 host. The linked decision specifies
the derived fixture and environment. Inspect
`correctness.generated_token_equality.passed`, not the harness's unrelated
promotion-plumbing status. Preserve the declared profile and ownership
contracts, check both consumers, and only then remove the pin. The direct-leaf
`qwen35_batch_correctness.py` harness does not exercise this registry choice.

Current-source review and focused regression evidence:
`../worklog/entries/20261029T182650.701375Z-cleanup-refactor-r0421-paro-pin-20d2af.md`.

### R0422 — Automatic context-sizing control policy

`hipengine/generation/qwen35_gguf.py` prices route-specific prefill workspace,
transient storage and speculative storage from allocator plans. The old
full-context flat transient accuracy item is no longer the default model.
The flat model remains the fallback when route terms cannot be derived; the
reserve covers memory not priced by those plans. R0406 owns the separate XTX
server capacity refresh, not this control-policy decision.

**Deferred decision:** retain the five environment controls as explicit
calibration/rollback interfaces, or replace them with an owned sizing policy.
`HIPENGINE_GGUF_AUTO_CONTEXT` controls both selection and allocation backoff;
`HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS` bounds retries.
`HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB`,
`HIPENGINE_GGUF_KV_TRANSIENT_KIB_PER_TOKEN` and
`HIPENGINE_GGUF_KV_TRANSIENT_FIXED_MIB` override the reserve and transient
terms. Route-derived pricing does not eliminate the flat-model fallback or
the unpriced reserve. A benchmark cycle or unspecified production mileage
alone does not settle the interface decision.

The generator's `_auto_resolved_max_sequence_lengths` cache selects contexts
by serving capacity/allocation mode. `_auto_resolved_max_sequence_length`
feeds resident-context reporting, DMS output limits and prepare-time retry.
`_auto_context_estimate` carries the pre-allocation estimate into
`session.kv_capacity_estimate`. These are live ownership roles, not redundant
server-reporting copies; reading `/ready` from the session is not sufficient
to remove them.

**Clearing action:** choose the control interface, migrate all readers and
allocation probes together, and preserve default route pricing, unavailable-
geometry fallback, explicit context handling, retry bounds and rollback
semantics. Run `tests/test_unit_qwen35_gguf_capacity.py`,
`tests/test_unit_qwen35_gguf_capacity_route_terms.py` and
`tests/test_unit_gguf_server_allocation_probe.py`; exercise the changed policy
through a real server allocation probe, checking reported versus allocated
context and teardown. Recalibrate reserve and transient terms together if
changing their values, following the capacity protocol in
`benchmarks/HARNESSES.md`. Do not delete the generator cache without assigning
its selection, DMS and retry responsibilities to another owner.

Source review and focused regression evidence:
`../worklog/entries/20261029T183255.657751Z-cleanup-refactor-r0422-auto-context-9c4ccd.md`.

### R0425 — Qwen4Exp pre-binder conflict guard

`hipengine/generation/qwen4_exp_profiles.py` still publishes route settings
through `_apply_profile_env`. It records differing caller values in
`last_prebinder_conflicts()` and warns, or raises when
`HIPENGINE_QWEN4_EXP_PREBINDER_CONFLICT=error`. Binder-owned previous writes
are not caller conflicts. Runner-owned selector migrations have reduced this
transport but have not eliminated it; keep the guard while writes remain.
`scripts/execution_profile_q8_wmma_prefill_layers_gate.py` checks the conflict
map after construction and applies its experimental route overrides afterward.

**Deferred dependency:** finish the explicit owner-field migration described
in the 2026-08-29 Qwen4Exp late-layer production prefill entry. Do not replace
it with implicit per-method profile context or a transaction framework.
**Clearing action:** migrate the remaining binder writes and their dispatch,
resource and diagnostic readers together, provide an explicit selected-route
getter, and update the layer gate to use it rather than environment ordering.
Only then remove the conflict map, binder-write tracking, warning/error control
and their documentation. Preserve caller-override observability and strict/
production isolation. Run `tests/test_unit_qwen4_exp_profiles.py`, the affected
runner ownership tests and a real generation request confirming selected routes
and isolation before closing the migration. The focused guard regression is:

```bash
uv run pytest tests/test_unit_qwen4_exp_profiles.py -k 'pre_binder or previous_bind'
```

This review changes no runtime behavior. The historical variable count and
script census are not a current inventory; their original text remains in Git
history. Source review and guard checks:
`../worklog/entries/20261029T184203.938095Z-cleanup-refactor-r0425-binder-conflicts-b4ac19.md`.

### R0426 — Wide-row prefill controls and diagnostic ownership

The route is implemented and enabled by the composed UD production profile.
`F16_ROUTE_Q8_DENSE_WIDE_PREFILL_LAYERS` covers all block layers; the
non-composed window is `PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`. The old
16–47 description and claim that every short-row request uses the exact
parent are obsolete. `_q8_dense_wide_dispatch` can admit short rows through
its separate short-row control, and it accepts the WMMA-rewritten parent
within the selected layer scope. This review changes neither arithmetic nor
defaults. Historical promotion results and the pack histogram remain in Git
history and their original benchmark artifacts.

**Deferred control-interface decision:** retain diagnostic selectors or
replace them with explicit profile/runner controls. The binder disables
`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` in the composed production path, but
`scripts/qwen4exp_dense_route_ab.py` and
`scripts/qwen4exp_disabled_family_sweep.py` still exercise WMMA and wide-route
alternatives. Do not delete shared kernels because one profile no longer
selects them. Migrate those diagnostic callers and the layer-scope gate with
R0425's owner-field migration before removing
`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE`, its layer control or the WMMA control.
Keep the registered exact fallback. Any extension of changed-arithmetic
production defaults follows `docs/EXECUTION-PROFILES.md`; missing historical
measurements are not a reason to refuse an explicit evaluation.

**Manifest dependency:** R0460, “Qwen4Exp production routes the manifest does
not enumerate”, owns the broader dense-wide/GDN/GR scope migration. The
K10240/N320 tail selection does not describe the whole dense-wide layer window.
Close that dependency with declared strict/production scopes, binder agreement
and manifest-identity tests; do not revive the obsolete 16–47-only proposal.
Until then, an artifact's manifest hash alone does not identify these routes.

**Library-lookup dependency:** ordinary linear launch paths use
`_variant_scoped_library`; pair and fused paths still contain quant-only
lookups. Before giving one of those families a separate shared object, route
its lookup through the helper and test variant-key precedence, quant-key
fallback and absent overrides through the affected launcher. A common quant
key does not establish that a library exports every variant's symbol. This
review does not claim those remaining readers have been migrated.

**Deferred census-interface decision:** keep explicitly bounded per-module
coverage or define a registry-wide launch record. The three dense families
listed in `hipengine/kernels/launch_census.py` record launches; routed kernels
and graph replay are outside that contract. A registry-boundary migration must
first define role/shape metadata and whether a record counts capture or replay,
then prevent double counting when removing module-level calls. Preserve
`tests/test_unit_gguf_launch_census_coverage.py` and add coverage for the newly
claimed families. An empty row is not evidence that an uncovered family did
not run.

**Deferred diagnostic-retention decision:** keep or retire the bitmap repair
and j32 tile experiments together with their callers. Bitmap repair is called
by `scripts/qwen4exp_repair_cost_structure.py` and
`tests/test_gpu_qwen4exp_q4_iu8_row_bitmap_repair.py`; the Q4_K/Q5_1 j32
wrappers are called by `scripts/qwen4exp_moe_main_kernel_cost.py` and
`tests/test_gpu_qwen4exp_q4_iu8_j32_tile.py`. These are diagnostic consumers,
not production selection. If retiring them, remove exports, registrations,
wrappers, kernel instantiations and caller arms together, preserving the
historical artifacts. If replacing them, first implement and test the
replacement against the existing exact-repair contract; the proposed LDS
weight-staging design is not proof of an implemented, exact replacement.
The historical proposals and measurements remain under
`benchmarks/results/2026-09-17-qwen4exp-iu8-repair-cost-structure/`,
`benchmarks/results/2026-09-22-moe-j32-tile/` and
`benchmarks/results/2026-09-27-moe-down-lds-weight-staging/`.

Source review, individual subitem dispositions and focused checks:
`../worklog/entries/20261029T184908.042620Z-cleanup-refactor-r0426-wide-prefill-fcf10e.md`.

### R0427 — Fixed256 reference and legacy comparison control

The fixed256 implementations are not globally superseded.
`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans` backs the
gfx1151 `w4_paro/bf16_context_batch_c1_exact_spans` registration, the
Qwen4Exp legacy comparison and numerical reference tests. Its
`..._fixed256_threads_spans` sibling serves the GGUF short-context
thread-width path, including the gfx1151 package default, as well as diagnostic
sweeps. Retire the old instruction to delete these shared symbols after a
Qwen4Exp category run; that run does not replace their other contracts.
R0381 owns the thread-width comparison interface.

**Deferred decision:** retain or retire
`HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY`, which selects the older body
inside the Qwen4Exp dense attention path. If retiring that comparison, remove
its reader and diagnostic overrides together, check the promoted route through
public generation and its exact-reference tests, and remove only the now-unused
Qwen4Exp import. Do not remove the shared body, exports or registry entries
while their independent consumers remain. Review
`scripts/qwen4exp_fullattn_prefill_microbench.py` separately from this switch:
it calls the reference directly and is not made obsolete by removing the switch.

Current-source consumer review and CPU route checks:
`../worklog/entries/20261029T185354.280363Z-cleanup-refactor-r0427-fixed256-f9d7e7.md`.

### R0428 — Full-attention unroll comparison interface

`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_experiment_spans`
is a diagnostic wrapper, not a production dispatch route. Its caller is
`scripts/qwen4exp_fullattn_prefill_microbench.py --replay <capture.npz>
--qk-variants 0,1,2`. The selector compares two-token/int32,
four-token/int32 and four-token/int64 address-table variants. The script
rejects `--qk-variants` without a replay rather than silently ignoring it;
help and argument validation do not require a HIP runtime.

**Deferred decision:** retain this comparison interface or retire its replay
arms. If retiring it, remove the experimental HIP export, Python wrapper,
package export and `--qk-variants` handling together. Preserve the promoted
launcher and its shared kernel template, and the fixed256 consumers recorded
in R0427. Run the CLI regressions and promoted-route GPU parity tests after
removal; historical captures and measurements remain evidence, not callers.

R0429's rejected per-block table is already removed from the implementation.
Its rejection and rationale belong to
`../worklog/entries/20260928T155352.726337Z-lhl-qwen4exp-fullattn-qk4-promoted-6575b7.md`
and the source comment at the per-token offset precompute loop, not an open
cleanup item. Current review and command-contract checks:
`../worklog/entries/20261029T185840.489411Z-cleanup-refactor-r0428-r0429-7e8f0a.md`.

### R0432 — Dense staged-F32 scope and comparison ownership

`staged_f32_bm128_bn64_bk32` remains registered for gfx1151 dense Q8_0.
`_raw_k_staged_f32_dispatch` can replace an exact raw parent within the
explicit layer scope; the production binder does not select this variant.
This is distinct from the removed GR selector in R0021. The shared module
also has direct diagnostic and GPU-test consumers, so rejecting one GR or
dense scope does not justify deleting the whole module.

**Deferred decision:** retain this evaluation interface, choose a new
production composition, or retire its runtime selector. The route has already
been evaluated: the September scope ladder includes failed lower-layer scopes
and passing numerical screens elsewhere. The composed result also records
outlier-review and qualification blockers. Those are historical results for
named compositions, not a fresh default decision or a claim that evaluation
cannot run. Preserve explicit evaluation and the exact fallback.

The gate's `staged_f32_over_wmma` teacher is fixed WMMA at layers 16–47 with
dense-wide cleared; it is not the current production profile. Its marginal
drift cannot substitute for `staged_f32_composed_vs_exact`, whose teacher is
the exact chain, or for a current-profile comparison. Before promoting a new
composition, declare its scopes and teacher, verify route engagement after
cache warmup and run its execution-profile gate and public generation check.
Before retiring the selector, migrate diagnostic callers and remove only
unused registrations/wrappers after auditing the shared module's consumers.

Keep the existing interface regressions: empty/unset staged scope means off,
`0` means layer zero, and scope plus owning layer participates in the dispatch
cache key. Runtime and backend variant constants agree without an eager backend
import. The gate expands `--layers` ranges to comma-separated indices; direct
environment values use that comma form. `WMMA_CERTIFIED_SCOPE` supplies the
historical fixed window. These are tested contracts, not pending parser fixes.

Historical evidence, including the invalid self-comparison and its correction,
remains in `benchmarks/results/2026-09-29-q8-staged-f32-dense-prefill-layers-gate/`.
Current-source review and focused checks:
`../worklog/entries/20261029T191254.312246Z-cleanup-refactor-r0432-fa6eb2.md`.

### R0434 — Pack8 GEMV width controls and exact-thread ownership

`HIPENGINE_GGUF_PACK8_GEMV_WIDE` remains default-on. For eligible single-row
`pack8_gemv_*` launches with output width at most 640, the selector requests
256 threads; explicit caller geometry, multi-row launches and the separate
`pack8_gemv_decode_*` family retain their own policy. Selection occurs after
the dispatch-cache lookup. The strict/candidate gate still changes this
control per arm; a release cycle does not make that caller disappear.

**Deferred interface decision:** keep the comparison control or migrate
`scripts/qwen4exp_layer2_profile_gate.py` and
`scripts/qwen4exp_de3_route_probe.py` to an explicit launch policy before
removing its environment reader. Preserve per-launch selection, caller
geometry precedence, narrow/wide boundaries and default behavior. Validate
with `tests/test_unit_gguf_pack8_gemv_wide_threads.py`, the migrated gate and
public generation confirming the selected widths and strict/candidate isolation.

Gemma 4's block and packed paths enter `pack8_gemv_exact_threads_session(True)`
to preserve their repeated-one-row equality contract. The wider block changes
the reduction tree; changing only the singleton side cannot preserve that
contract. Keep this scope until both sides have an explicit compatible owner,
then run `tests/test_gpu_gemma4_packed_batch_forward.py` and
`tests/test_gpu_gemma4_runner_verify_rows.py` plus public generation/verification
before removing the scope. Arithmetic-default changes require their profile
gate; removing a comparison control must not silently change arithmetic.

Historical performance and Q8_0 coverage remain in
`benchmarks/results/2026-09-30-qwen4exp-de3-threads-lever/`; they are not new
measurements or an admission list for other implemented quantizations.
Source review and focused checks:
`../worklog/entries/20261029T192349.450639Z-cleanup-refactor-r0434-4bc573.md`.

### R0435 — Bounded KV reset comparison ownership

Bounded reset remains default-on. `written_cache_bytes()` rounds the host
position to whole blocks and clamps the result to cache capacity; reset clears
that prefix before setting position to zero. Its contract requires allocation
and prior clears to leave the remaining bytes zero. The full-clear alternative
is still selected by `HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET=0`.

**Deferred decision:** keep the full/bounded comparison interface or migrate
`scripts/qwen4exp_rt4_kv_compare.py` and the layer2 candidate to explicit reset
policy before removing the environment reader. The RT-4 and DE-5 diagnostics
now report the default-on setting when the environment is unset. They preserve
explicit spellings; changing report defaults does not change runtime reset.
A release cycle alone does not remove these comparison consumers.

Keep `tests/test_gpu_qwen4exp_kv_reset_bounded.py` as the range and reuse
contract. Before retiring the full-clear control, exercise repeated requests,
long-to-short reuse, rollback, cancellation and changing request widths through
public generation/serving, comparing cleared buffers and post-reset outputs.
Do not assume the low-level range tests establish every lifecycle case.
The RT-4 reader synchronizes before device-to-host copying; keep that ordering
so unfinished device writes cannot create a false arm difference.

Historical parity and timing evidence stays in
`benchmarks/results/2026-09-30-qwen4exp-rt4-bounded-kv-reset/`. Compare bounded
and full-clear arms under the same arithmetic profile; strict-versus-production
logit differences do not isolate reset behavior. Reporting regression evidence:
`../worklog/entries/20261029T192838.841154Z-cleanup-refactor-r0435-cd24b0.md`.

### R0437 — Historical dense-wide engagement artifact refresh

The dense-wide selector recorder and route-specific summary are implemented.
Staged-F32 gates now record the final selected dispatch, including cache hits,
and all three staged gate names map to that record. The shared final-dispatch
path is the coverage boundary; this does not claim graph-replay instrumentation.

**Deferred evidence decision:** the canonical dense-wide 16–47 artifact from
`20260930T113116` predates engagement recording. Preserve it as historical
evidence with that limitation; rerun the same composition with independent
engagement records if a new claim relies on it. Do not treat its old scope as
the current production profile or retroactively add evidence to its artifact.
The original limitation is recorded in
`benchmarks/results/2026-09-30-qwen4exp-rt1-chunk4096-gate/DENSE-WIDE-REGATE-AT-HEAD.md`.
Implementation and validation:
`../worklog/entries/20261029T194103.398080Z-cleanup-refactor-r0437-a865f5.md`.

### R0438 — Chunked WY diagnostic retention

The unregistered `qwen4_exp_gdn_prefill_wy_f32` wrapper and HIP body remain
reachable through the `wy_chunked` arm of
`scripts/qwen4exp_gd3_recurrence_screen.py`, not through production dispatch.
The historical rejection stays in
`benchmarks/results/2026-10-01-qwen4exp-gd4-wy-kernel-built/`.

**Deferred decision:** keep that comparison reproducible from the working tree,
or retire its executable arm while preserving the historical artifact and Git
source. Waiting for historical references to disappear is not a useful removal
condition. If retiring it, remove the screen entry, host wrapper, HIP export,
WY kernel and its private constants together; preserve the other recurrence
implementations in the shared source. Run screen CLI/consumer checks, rebuild
the affected GDN library and run its surviving recurrence tests. Production
profile selection is independent and must not change as a side effect.

Source review and CLI validation:
`../worklog/entries/20261029T195224.685777Z-cleanup-refactor-r0438-1c85f6.md`.

### R0439 — Architecture chunk fallback ownership

Automatic gfx1151 prefill still has a `(256, 256)` architecture fallback.
GGUF passes its backend geometry policy into the common resolver; the PARO
resolver call supplies architecture but no geometry tuple. A non-zero manual
chunk field preserves the entire caller config, and disabled auto-tuning also
bypasses these policies. Otherwise geometry overrides architecture, which
overrides the shape/memory baseline. Reporting comes from the same resolver.

**Deferred policy decision:** make the architecture fallback geometry-aware or
move its supported shapes into backend-owned policies shared by the applicable
runners. Do not copy GGUF's quant-specific setting into PARO based only on hidden
size, or remove the fallback without defining unknown-geometry behavior.
Declare the proposed geometry/quant contract, compare PARO prefill on the same
host with explicit chunk overrides, then migrate both consumers without
changing manual-config precedence. Run chunk-precedence, GGUF prefill-plan and
row-cap tests, plus public generation on the affected GGUF and PARO paths.
Check short, chunk-boundary and long inputs and reported versus executed sizes.
Numerical or performance claims require the corresponding profile/benchmark
checks; missing measurements do not prevent explicit evaluation.

Current-source review and documentation correction:
`../worklog/entries/20261029T195712.777984Z-cleanup-refactor-r0439-258d7d.md`.

### R0442 — DPP family comparison control

The rejected Q8 row-first grid parameter, export, wrapper, registry entry,
screen arm and dedicated tests are removed. The column-first pipelined parent,
DPP reduction and checked-full-row DPP specialization remain. R0012 already
retired the checked-full-row opt-out; that is separate from the family control.

**Deferred interface decision:** retain `HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION`
or migrate its state-gate, checked-row and campaign consumers to explicit
variant selection before deleting it. The flag remains default-on and part of
dispatch-cache identity. The shuffle-tree parent is a live numerical reference;
do not make every template instantiation DPP merely to remove the runtime flag.
Check `scripts/qwen4exp_row4_state_gate.py`, the checked-full-row diagnostics,
`scripts/qwen4exp_layer2_profile_gate.py` and
`scripts/qwen4exp_halo_box_campaign_ab.py` together. Preserve fallback semantics,
run DPP/checked-row parity and cache tests, and confirm public dispatch before
retiring the selector. Historical grid-order results remain in their original
October4 artifacts; no new performance claim follows from code removal.

Removal validation:
`../worklog/entries/20261029T201635.969016Z-cleanup-refactor-r0442-e923ef.md`.

### R0443 — GR DPP comparison ownership

The profile still binds `HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION`; the selector
reads it independently of the runner-owned `gr_wave_scale` field. It applies
only above 256 rows with four branches. Missing DPP registration selects the
wave-scale parent when available, even if the independent wave setting is off,
then the original fused parent. The unfused chain remains available.

**Deferred interface decision:** migrate DPP selection to its explicit profile
owner together with the halo-box campaign, row4 state gate and K320 diagnostics,
or retain the comparison control. R0419 migrated wave-scale ownership, not
DPP; R0013 retired the separate K320 override, not this family selector. Do not
close those controls independently and accidentally compare a preempting wide
or IU8 route in both arms. Preserve declared scopes, fallback order, selected
variant reporting and failure restoration; verify the migrated state/campaign
comparisons and public generation with actual DPP engagement. Run
`tests/test_unit_qwen4exp_gr_dpp_reduction.py`, GR wave-scope and profile tests,
plus the affected GPU parent-parity/state tests before retiring the reader.

Historical numerical and timing evidence remains in
`benchmarks/results/2026-10-05-qwen4exp-gr-dpp-reduction/`.
Current ownership review:
`../worklog/entries/20261029T202553.939702Z-cleanup-refactor-r0443-95c64b.md`.

### R0444 — Risk multiplier ownership and repair-grid follow-up

The profile binder publishes `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT="1.0"`;
the unbound helper default remains `4.0`. The former ledger claim that the
profile deliberately stayed at `4.0` is superseded. The corrected historical
comparison applied overrides after binding and recorded differing repair queues;
the earlier apparent null comparison did not establish multiplier independence.
See `../worklog/entries/20261004T001845.395977Z-lhl-qwen4exp-risk-mult-promotion-e7992b.md`.
Do not reuse a historical candidate label as proof of its teacher's current value.

**Deferred ownership decision:** retain an explicit diagnostic multiplier or
replace environment transport with a profile-owned value. Preserve the bound
and unbound semantics, effective-value reporting, strict/reference contract and
post-binding diagnostic isolation. A repair-grid experiment is not permission
to change the risk threshold or the repair coverage contract.

**Deferred optimization follow-up:** the sparse repair wrapper still defaults
to 1024 grid blocks, and the runner does not override that count. The device
loop is bounded by the queued count and capacity, so fixed launch geometry is
not fixed repair work. Use `scripts/qwen4exp_repair_grid_probe.py` to compare
explicit grids and profile the actual repair calls before proposing a launch
policy. Record effective multiplier, queue counts, overflow handling and cache/
stream ownership; run numerical repair parity and public same-profile checks
for any adopted change. Earlier fixed-cost estimates are historical hypotheses,
not a fresh promise of recoverable wall time. Kernel traces follow the cached-
build protocol in `docs/OPTIMIZATION.md`.

Current-source correction and checks:
`../worklog/entries/20261029T203123.488199Z-cleanup-refactor-r0444-d80e45.md`.

### R0450 — Ordered QSA prerequisite ownership

`HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE` remains the profile-bound prerequisite
for the scalar ordered attention family. The fused-v2 experiment is removed,
but that does not retire ordered v1 or the runner-owned three-pass v2 choice.
Removal condition: migrate the prerequisite to explicit profile/runner ownership
with the campaign and drift callers, preserve non-ordered and v1 fallbacks,
and verify scalar and batched public routes independently before deleting it.

### R0446 — Launch-library lifetime and environment observation

Wrappers accept explicit library handles and otherwise resolve through the
build caches. `build_hip` checks fast-path, resolution and loaded-library caches;
the keys observe build-affecting arguments and environment. Explicit invalidation
now clears all three handle caches, including the formerly omitted fast path.

**Deferred lifetime decision:** pin resolved handles to a runner/execution owner
or continue observing build-environment changes on implicit launches. Define
owner lifetime, invalidation and mixed-backend/variant behavior before removing
wrapper fallbacks. A single process-wide handle cannot stand in for distinct
build identities; quant-only maps also cannot identify separate variant objects
(see R0426). Preserve explicit-handle overrides and direct diagnostic callers.
Test warm hits, environment changes, explicit invalidation, missing cached
artifacts and variant precedence, then exercise public generation with the
intended handles and teardown. Do not infer an engine speedup from fewer Python
lookups; historical identity experiments remain under
`benchmarks/results/2026-10-08-qwen4exp-build-identity-fast-read/`.

Invalidation repair and validation:
`../worklog/entries/20261029T204623.968297Z-cleanup-refactor-r0446-9e2320.md`.

### R0459 — Row-split GDN recurrence removal decision

The UD production composition binder selects row-split from its tiled GDN
manifest scope. The dense-wide and GR-up projections also have declared scopes.
The Q4_K_M production plan
still names multi-column but resolves to strict due to its recorded numerical
failure; it is not evidence of a live default multi-column execution.

**Deferred removal decision:** after R0460, decide whether UD still needs an
explicit multi-column comparison route and whether the Q4_K_M recovery plan
still needs the shared implementation. Verify both quant bindings and
user-facing execution before removing any UD-specific route. The earlier closure in
`../worklog/entries/20261030T002539.121609Z-cleanup-cleanup-refactor-r0459-e3bbfd.md`
missed the final composition override and is corrected by this entry.

### R0462 — QSA gather storage and variable-span follow-up

Selector ownership and scratch accounting are fixed: the manifest publishes
`runner.qsa_wmma_enabled`, diagnostic arms use `qsa_wmma_attention`, and context
admission reserves the shared lazy F16 K/V pair and mask once per runner.
Direct selection already writes masks from scores; the final-row expansion
publishes positions/counts for decode and is not redundant mask construction.
The direct-selector rollback remains a separate cleanup item.

**Deferred storage decision:** retain the shared full-context gather or spend
per-layer storage to cache F16 K/V across chunks. Public multi-chunk generation
confirms that distinct layers reuse the same output scratch within each chunk.
Changing only `first_position` to the chunk start would reuse another layer's
K/V. **Clearing action:** profile a representative long-context workload, choose
the memory/reuse trade-off, then implement per-layer ownership and admission
accounting together if caching is chosen. Validate reset, prefix adoption,
snapshot restore, speculative rollback, interrupted prefill and teardown,
including an intervening decode append before more prefill. Do not infer a
throughput benefit from the short instrumented ownership diagnostic.

**Deferred capability extension:** the gather wrapper accepts only uniform
BF16 `KVLiveSpans`. Before offering this route to a non-dense policy, implement
a gather consuming `live_counts`, `token_positions` and `evict_mask`, plus
matching sparse-mask/attention semantics. Verify nonidentity physical pages,
evicted positions and policy transitions against the strict reference; preserve
the existing dense route. This is an implementation prerequisite, not an
artifact-identity or benchmark-admission gate.

Disposition and public ownership diagnostic:
`../worklog/entries/20261030T012436.140169Z-cleanup-cleanup-r0462-disposition-92a3ce.md`.

### R0471 — Optional selected DP4A adapter intent

`bind_selected_call` still refuses optional raw/T16 selected DP4A adapters:
its intent describes BF16/F32 inputs but not the optional Q8_1 workspace and
quantization operation executed by those routes. Mandatory X8 adapter intents
already exist. A BF16 primitive record does not describe these additional
operands or their lifetime.

**Deferred contract extension:** encode optional quantization/workspace
ownership in the shared selected-call intent, bind the actual caller operands,
and only then remove that specific refusal. Preserve row limits, ordered dual
partners, workspace lifetime and the strict fallback. The clearing checks are
`tests/test_integration_gguf_invocation_contract.py`,
`tests/test_integration_gguf_selected_call_intents.py`,
`tests/test_integration_gguf_execution_authorization.py`, and public
native-row generation with the optional adapter engaged. Any changed arithmetic
default additionally follows the execution-profile gate. This is a missing
invocation representation, not missing benchmark evidence.

The compatibility dispatch table and misleading Q8T16 BF16 selector are
removed. Native entry consumption already binds resident dependencies and the
actual recurrent-state dtype; cold declarations remain required for standalone
preflight, not runtime setters. F5 caller-binding regression tests cover the
existing policy callers. Baseline GDN FP16/F32 state separation remains a
correctness contract, not cleanup debt.

### R0475 — Eager AR diagnostic retirement

`scripts/gguf_ar_eager_graph_equivalence.py` still invokes the eager branch of
`scripts/qwen36_dense_gguf_suite.py` as an independent reference for graph token
sequences. Unit tests cover comparison failures and runner wiring, not actual
device equivalence. Removing the flag alone breaks that live checker.

**Deferred decision:** keep this operator diagnostic or move its eager runner
into a routinely exercised device gate before retiring the suite option.
If migrating, remove `--ar-decode-mode`, its eager `_run_ar` branch and
`autoregressive_step` stage only after updating the checker and its callers.
Run the three unit modules `test_unit_gguf_ar_eager_graph_equivalence.py`,
`test_unit_qwen36_dense_gguf_suite.py` and `test_unit_ud_mtp_paired.py`, then
exercise the replacement device comparison with matching settings on both
paths. Keep graph mode as the paired denominator; the paired validator must
continue rejecting eager payloads. No product admission depends on this choice.

### R0479 — Q5 four-wave parity reference

The Q5 singleton rollback flag is removed. gfx1100 package policy selects
`t16_gemv_rowtile_single_wave_bf16_bf16_out`; the four-wave
`t16_gemv_rowtile_bf16_bf16_out` remains separately registered. Its wrapper in
`gguf_t16_selected_gemv.py` declares the ordered wave-0..3 reduction shared by
the grouped rows6/rows8 owners and the one-expert direct producer. Removing a
selector does not remove that reference contract.

**Deferred contract migration:** keep the distinct reference, or migrate all
of those consumers' parity contracts before deleting or rebinding it. The
clearing checks must compare the grouped and direct producers against their
replacement reference on device, followed by the applicable execution-profile
gates for any changed arithmetic default. Dispatch tests alone cannot establish
that equality. The capability table continues to distinguish the production
owner from the reference; it is not an admission gate.

### R0483 — EVIE/Surya runtime library ownership

`hipengine/runtime/evie.py` and `hipengine/runtime/surya.py` import builders
and callable wrappers from `hipengine.kernels.hip_gfx1100`. Their constructors
build the EVIE, conv and GDN libraries; Surya additionally builds its own
library. Both runners already consume shared linear-attention kernels, so
waiting for a second consumer is not a useful removal condition.

**Deferred architecture decision:** preserve explicit library ownership in
these specialized runners, or migrate both to backend-provided runtime
operations. The latter needs a declared boundary between the four-axis kernel
registry and whole-library construction; a second library registry must not
silently become a fifth kernel axis. Resolve builders and wrappers together,
keep injected test libraries and their lifetime semantics, and preserve the
existing arithmetic and supported HIP targets. Merely changing imports would
leave symbol dispatch and ownership duplicated.

If migrating, document the boundary in `docs/PLAN.md`, test backend selection
without loading HIP, and run EVIE/Surya live runtime and public API checks,
including `tests/test_live_evie_gpu_runtime.py`,
`tests/test_live_surya_gpu.py`, `tests/test_live_surya_public_api.py`, and
`tests/test_integration_surya_server_multimodal.py`. If explicit ownership is
chosen, document that exception and remove this entry. No runtime gate or
fallback is added while the decision is open.

### R0484 — Surya decode fallback and sparse-chunk work

The unused pre-spans scatter is removed; the debug script now hooks
`surya_scatter_kv_f32_spans`. The rocBLAS decode chain is different: it is
still the strict fallback for head dimensions other than 256 or GQA repeats
other than four, and the parent oracle for `tests/test_gpu_surya_kv_spans.py`.
Retiring a debug hook cannot justify removing that capability.

**Deferred contract migration:** provide a span-aware unfused chain for those
shapes before retiring the dense-offset fallback. Preserve its oracle role
until replacement comparisons cover nonidentity pages, evictions and partial
contexts, then exercise public Surya generation. Sharing a kernel with
Qwen3.5 is not itself a clearing condition.

**Deferred optimization follow-up:** profile sparse live spans before changing
the fused producer's full-chunk dot products. Bounding work by visibility
must preserve token positions and masking; verify with the spans GPU suite
and the applicable numerical gate. No performance improvement is claimed by
the scatter cleanup.

### R0488 — Surya same-process comparison failure

The recorded comparison in
`benchmarks/results/2026-09-13-gfx1151-surya-suite-recut.json` uses one process
per lane and `scripts/surya_perf_compare.py --merge`. The original combined
run produced degenerate hipEngine output for `full`, `columns` and `list`,
while the isolated hipEngine run passed. Shared-process device state is a
suspect, not an established root cause. The harness still supports both
routes; keep `--merge` and the artifact's `lane_isolation` provenance.

**Deferred reproduction:** the local cache lacks `datalab-to/surya-ocr-2`;
the R0484 public request failed at loading, before inference. Make the
checkpoint available to `resolve_surya_path`, then run
`python3 scripts/surya_perf_compare.py --split all --runs 3 --lanes hipengine_gpu,torch_cuda`
in the HIP environment. Compare with isolated runs, find and repair the
first divergent state, and require every combined row to pass before
publishing a replacement comparison. Preserve historical artifact provenance;
remove any still-applicable scoreboard isolation caveat only when the new
comparison no longer needs it. No runtime change or numerical fix is claimed.

### R0490 — Explain the Surya vision tile-shape spikes

The historical 4096-patch curve is non-monotone; the cause is unresolved.
Keep the existing planner rather than infer another shape rule. The recorded
sweeps remain in `benchmarks/results/2026-09-13-gfx1151-surya-vision-tiling-cost*.json`.
Wavefront rounding is a measured tie-break, not a guarantee that every multiple
is fast. The probe already lifts all three `SHAPE_TILE_*` constants for explicit
blocks and asserts that the selected block matches the request; preserve this
protection against accidentally measuring the production cap repeatedly.

**Deferred profiling:** make the Surya checkpoint available and provide
`rocprofv3` in the HIP environment (it is absent from the current shell's PATH).
Compare kernel traces for the 128-row and 512-row shapes at 4096 patches, using
`rocprofv3 --kernel-trace -- python3 scripts/surya_vision_tiling_cost.py --pages page_ja.png --blocks 128,512 --reps 3 --out /tmp/surya-4096-shapes.json`.
Attribute any shape-dependent latency to the actual kernels before changing
the policy; a rocBLAS selection change is still only a hypothesis. A policy
change requires the applicable numerical gate and public Surya validation.

### R0491 — Surya mid-size and A4 vision tile investigations

The CPU time-ceiling inversion is fixed in `2afb609e3`; it is no longer debt.
Keep the vision envelope and text-prefill budget policy unchanged while the
remaining shape questions are investigated. Historical sweeps are preserved
in `benchmarks/results/2026-09-13-gfx1151-surya-vision-tiling-cost*.json`,
`2026-09-13-gfx1151-surya-vision-narrow-budget-check.json` and
`2026-09-13-gfx1151-surya-vision-tiling-a4-ceiling.json` in that directory.

**Deferred experiments:** the local Surya checkpoint is missing. Once it is
available, run `python3 scripts/surya_vision_tiling_cost.py --pages page_dense.png,page_ja.png,page_long.png --blocks 64,96,128,192 --reps 3 --out /tmp/surya-midsize-shapes.json`
and extend the probe to nearby mid-size grids before choosing a new envelope.
Do not special-case the lone historical mid-size anomaly.

For A4, separate a larger scratch budget from an envelope change: the byte
budget is the limiting policy. Start with
`python3 scripts/surya_vision_tiling_cost.py --pages page_a4.png --blocks 2048,4096,8192 --reps 3 --out /tmp/surya-a4-shapes.json`.
The dense A4 comparison remains unrun and needs a separately resource-checked
large allocation; shape-to-shape results do not establish dense parity. The
historical launch-configuration failure did not reproduce at the tested wide
tiles, so do not add a memory refusal on that basis. Any policy change needs
the applicable numerical gate and public Surya validation.

### R0492 — Bound Surya causal-prefill keys per tile

`SuryaGpuRunner._attention_packed` still uses `m=tokens` for each score GEMM
and `k=tokens` for each value GEMM. Tiling bounds scratch memory but computes
future keys that the causal mask later excludes. A candidate can bound keys
to `start + bq`, updating the score strides, mask, softmax and value product
together. This changes matrix shapes and may change reduction arithmetic;
it is not just a scratch-allocation cleanup.

**Deferred optimization:** the local Surya checkpoint is missing and
`rocprofv3` is absent from PATH. After restoring both, profile with
`rocprofv3 --kernel-trace -- python3 scripts/surya_gpu_rocprof_driver.py --phase prefill --repeat 3`.
Use the driver phase markers to exclude vision/load work. If score computation
is worth optimizing, implement the bounded-key path and compare against the
current chain on dense, partial-tail and multi-tile prompts, then run the
applicable numerical gate and public Surya generation. Keep the current
working implementation until that replacement is ready.

### R0493 — Capture the Surya A4 fp32 reference

A4 remains in `QUALITY_ONLY_CASES` in
`tests/test_live_surya_transcription.py`; the ground-truth and reading-order
checks cover it, but `tests/fixtures/surya/oracle_fullpage_protocol.json` has
no A4 implementation-parity reference. Keep that distinction explicit.

**Deferred capture:** restore the missing `datalab-to/surya-ocr-2` checkpoint.
Add `a4` with an EOS-reaching budget to `PROTOCOL_MAX_TOKENS` in
`scripts/surya_oracle_greedy.py`, then run
`python3 scripts/surya_oracle_greedy.py --case protocol --only a4 --device cpu --page-dir tests/fixtures/surya --out-dir /tmp/surya-a4-oracle`.
The case is not accepted by the current script until that table is extended.
Inspect natural completion and merge the captured row into the existing
fixture; do not overwrite the other cases with the single-case output.
Move A4 into the implementation-parity test cases only after examining the
capture and running the comparison. Preserve quality coverage throughout;
use coordinate-tolerant comparison only if observed bbox-digit variation
justifies it, not merely to make a failure pass.

### R0494 — Choose one prompt-count output contract

Prompt counts are no longer Surya-only: Gemma image generation sets
`GenerationOutput.prompt_tokens`, while Qwen multimodal generation publishes
`telemetry.usage.prompt_tokens`. The multimodal server route now prefers the
explicit field, then telemetry, then one tokenizer fallback; zero is a valid
reported count. It no longer overwrites a telemetry count with a text estimate.

**Deferred interface decision:** choose whether the output field or telemetry
owns prompt usage before removing either representation or the compatibility
fallback. `GenerationOutput.prompt_tokens` remains optional, and text usage
also counts prepared prompt inputs through `_usage`. A complete migration must
cover ordinary, empty, batch, speculative and multimodal outputs, plus legacy
plugins that return no count. Keep the fallback until those producers have an
explicit replacement contract; do not infer image-token counts from raw text.
Validate migration with the Qwen/Gemma/Surya server multimodal modules,
`tests/test_unit_server_prefix_cache_usage.py` and public requests.

### R0495 — Decide CPU-reference vision admission

Configured GPU vision byte/time budget rejections now use
`SuryaVisionCapacityError`, translated to `SuryaRequestError` by the GPU
multimodal generator. Free-device-memory failures remain runtime failures;
they are not attributed to malformed requests.

**Deferred policy decision:** the CPU-reference generator has no vision
resource ceiling. Choose a configurable patch/memory budget or calibrate a
CPU timing estimate before adding admission there. GPU timing constants do
not describe CPU execution. Preserve existing runnable inputs until that
policy is chosen; no arbitrary default ceiling is added by this cleanup.
Validate the selected policy on both sides of its limit and an unrelated
size through `LLM.generate()`. Live Surya validation also needs a locally
available checkpoint; the current local cache lacks `datalab-to/surya-ocr-2`.

### R0499 — Decide whether to pursue exact dual GEMV scheduling

The VibeVoice LM decode caller does not override the dense dual GEMV wrapper's
`threads=256` default. The rejected thread-count experiment is not installed.
Corrected measurements and the audio-quality failure remain in
`worklog/entries/20260915T195500.000000Z-lhl-vibevoice-tts-gemv-bandwidth-corrected-9c4d18.md`;
the reverted exact staging attempt remains in
`worklog/entries/20260915T194341.808338Z-lhl-vibevoice-tts-staging-negative-3ca9f5.md`.

**Deferred research decision:** decide whether to fund a different workgroup
schedule or output partition that preserves the per-output reduction tree.
Neither restoring the rejected thread count nor repeating shared-memory staging
resolves this item. If pursued, profile the current session first, compare exact
outputs, synchronize cache flushes before isolated timing, and verify through
`scripts/vibevoice_tts_session_bench.py` and the generated-audio quality suite
(`scripts/vibevoice_tts_quality_suite.py`), not correctness tests alone.
Alternatively close this research direction explicitly. No runtime change is
needed merely to keep the rejected variants absent.

### R0500 — Choose cross-branch VibeVoice batching ownership

`VibevoiceTTSSession` constructs separate positive and negative LM runtimes;
each uploads its own layer weights and allocates its own KV caches. The loop
selects the next token before invoking `_negative_condition` for a diffusion
step. Existing attention calls take one cache pair, not per-row cache pointers.

**Deferred architecture decision:** choose shared immutable weight ownership
and a per-row KV interface before implementing speculative cross-branch batching.
Specify negative-position rollback on non-diffusion tokens, speech-boundary
resets, cancellation and errors; preserve RNG consumption and feedback ordering.
Ordinary multi-row prefill is not a replacement for this transaction contract.
Implement registered kernel variants only after that contract is chosen, then
verify branch conditions, token chains, rollback and request reuse through the
public synthesis surface and generated-audio quality suite. If arithmetic
changes, run the applicable execution-profile numerical gate as well.

Historical timing corrections remain in
`worklog/entries/20260915T200702.494130Z-lhl-vibevoice-tts-cfg-batch-scoping-51a916.md`.
Profile the complete session before claiming a benefit; per-call device syncs
can attribute work from other stages to this branch. Keep the existing working
sequential path until a replacement is implemented and validated.

### R0503 — Choose a bounded streaming handoff contract

`_stream_engine_text` (detailed and legacy paths) and `_stream_engine_many`
still call `_next_stream_item` in the threadpool for every chunk.
`EngineChildHandle.iter_chunks` and `EngineService.stream_many_detailed` drain
one collector event at a time. The collector supports larger nonblocking drains,
but the public synchronous iterators do not expose whether another chunk is ready.

**Deferred interface decision:** choose either a ready-only batch iterator
contract propagated through the LLM/generator boundary, or a bounded producer
bridge with explicit worker ownership and concurrency limits. Calling `next()`
repeatedly can block the first chunk behind future production; reserving a shared
threadpool worker for every live stream can starve other requests. Increasing
collector drain sizes alone does not remove the server's per-item handoff.

The replacement must deliver the first ready chunk without waiting for a full
batch, preserve per-choice ordering and fairness, bound buffering, and propagate
terminal errors after buffered output. Cancellation must signal the backend,
unblock a backpressured producer and close the iterator only after its active
`next()` finishes; the current batcher already documents that close race.
Test empty streams, slow producers/consumers, multi-choice fairness, saturation,
disconnects and errors through HTTP streaming. Then compare a streaming MTP
request before/after with chunk telemetry enabled; do not reuse the old overhead
estimate as a measurement of this implementation. Keep the existing bounded
mailboxes until the replacement contract is implemented.

### R0504 — Evaluate overlapping chunked prompt activations

The scheduler still serializes new prompt activations through
`prefill_activation_ready`/`prefill_needs_activation`. In-flight refusal preserves
speculative intent; priming-source availability decides subsequent speculation.
Partial success or abort now retains the activation claim until every remaining
sink has finished, rather than releasing it after the first row.

**Deferred scheduler work:** implement co-activation of chunked rows while
preserving provider-slot ownership and failure isolation. Exercise unequal
prompt lengths, different finish orders, partial failures and cancellation.
Compare generated tokens and acceptance against independent single-request runs
through the public generation surface; check that waiting requests keep their
provider and that all claims and sink buffers are released. The lifecycle fix
alone does not enable overlap or establish its numerical/acceptance behavior.

### R0505 — Separate prefill failure state from speculative intent

Restored-checkpoint priming is implemented. Prefix reuse, target-profile and
physical-streaming-category refusals no longer erase the candidate budget.
The remaining zero-budget writes in `qwen35_gguf.py` and `qwen35_gguf_mtp2.py`
cover exhausted target context and failed packed/incremental prefill streams.
They also prevent reopening an aborted stream; simply deleting them is not a
complete lifecycle refactor.

**Deferred interface decision:** give failed-stream state an explicit owner and
make eligibility consult it alongside `_priming_source`, or retain zero budget
as the terminal request-local representation. If separating intent from failure,
cover packed-to-serial fallback, incremental abort, cancellation, valid and
missing restored checkpoints, and context exhaustion. Confirm through public
requests that failed streams never reopen and eligible peers still speculate.

**Deferred experiment decision:** retire `HIPENGINE_SPEC_MTP_SPLIT_REFUSED_GROUPS`
and `speculative_split_refused_groups`, or keep the default-off reproduction arm
for the rejected singleton-split route. The alternative performs one provider
cycle per capable row rather than one autoregressive batch for a refused group.
Remove it when the cross-host/protocol comparison is complete or a replacement
wide-group route makes it unnecessary; preserve historical evidence in the
worklog, not as an admission rule. `docs/ENVS.md` documents the switch.

`HIPENGINE_DEBUG_SAMPLED_ROUTE` is a documented diagnostic in `docs/ENVS.md`,
not a pending admission prerequisite. Finish-rule resolution remains recorded
in `worklog/entries/20260922T140534.086274Z-lhl-mtp-finish-rule-c10d7c.md` and its
linked artifacts. Missing measurement rows do not block implemented MTP
capabilities; the old evidence-only admission requirement is retired.

### R0506 — Decide whether to retire the gapped-gather rollback switch

`HIPENGINE_GGUF_GAPPED_GATHER` is enabled by default and documented in
`docs/ENVS.md`; `0` selects the native paged fallback. The route uses registered
copy capability, real chunk-local page tables, demand-sized head-major buffers,
and token/byte allocation limits. INT8 and unsupported layouts retain their
own paths. These are runtime requirements, not temporary cleanup targets.

**Deferred decision:** retain the diagnostic rollback or remove only its env
lookup and switch-specific tests/documentation. The original retirement plan
calls for `uv run pytest --suite all -v` and served multi-turn A/B on gfx1151 and
W7900. This triage does not establish that cross-host result or change defaults.
If retiring the switch, keep the gapped-layout branch, demand-sized buffers,
forced AOTriton handling, missing-kernel/allocation fallbacks, and suffix guard
when the fast route is unavailable. Removing those mechanisms requires a
working replacement, not just successful switch retirement.

### R0507 — Consolidate the raw greedy MTP compatibility entry point

Public `LLM.generate_speculative_mtp_detailed` uses the generator constructed by
`LLM._get_text_generator`: the submit/poll wrapper and, for controlled streaming,
`EngineService`. `submit_speculative_detailed` selects staged speculative
submission when supported. The old public-versus-server sampling split is not
an accurate description of that shared ownership path.

The raw `Qwen35GGUFBringupGenerator.generate_speculative_mtp_detailed` method still
rejects non-greedy sampling. Its compatibility verifier uses `raw_target_top1`
and `compatible_sampling_modes=("greedy_fast",)`; those limits are correct for
that algorithm. The wrapper still calls the raw method for non-staged drivers.

**Deferred compatibility decision:** retire the raw GGUF entry point or route
its callers through a shared service owner without recursive delegation or
independent state owners. Preserve non-staged compatibility where required.
Do not broaden the raw verifier's sampling constants: sampled acceptance belongs
to the staged adapter. Validate blocking and streaming public requests, sampled
and greedy requests, cancellation, and non-staged fallback before removing the
raw implementation or its guards. No new sampling or GPU result is claimed here.

### R0509 — Choose shared target/provider checkpoint memory accounting

`HIPENGINE_MTP2_PREFIX_CHECKPOINT_ENTRIES` defaults to four bounded provider
snapshots with prefix caching enabled and none with caching off; zero is the
memory/bisection rollback. `PrefixCheckpointStore` owns release callbacks and
entry-count LRU eviction. Target `_evict_prefix_snapshot` already calls
`discard_prefix_checkpoint`, so target snapshot eviction does invalidate its
provider checkpoint. Token-prefix and block-id checks protect restore identity.

**Deferred budget decision:** define how provider recurrent-state/KV snapshot
bytes count against the target cache budget, including shared allocations,
replacement, failed capture and pressure-driven eviction. Then replace the
independent entry-count control with bounded shared accounting. Do not remove
the bound before that owner exists, or remove capture without restore.

Keep the rollback until shared accounting and coordinated eviction are tested.
Run `scripts/int8_mtp_prefix_gate.py` against a radix-enabled server: require real
prefix hits, compact INT8, speculative cycles and parity with the same
prefix-reusing AR route. A cache miss or AR-only response is not a pass. Include
budget pressure, overwrite, target eviction and shutdown release tests.

### R0510 — Define bounded-context startup warmup ownership and coverage

`_startup_probe_width_reduction` still prices concurrent resident sessions and
narrows startup warmup when their combined allocation exceeds usable memory.
The GGUF session acquisition path already accepts `max_sequence_length` and
includes it in its pool key; a separate-context constructor is not missing.
Startup does not yet use that facility to warm a wide packed route independently
of full-context resident allocations.

**Deferred warmup design:** connect a bounded probe context to explicit temporary
session ownership, retaining shared weights and releasing probe-only allocations.
Specify which packed kernels, workspace sizes and graph/context buckets it warms;
a short-context success must not certify full-context capacity or graph coverage.
Preserve actual request context limits and capacity estimates. Replace the width
reduction only after a wide startup probe and subsequent public concurrent
requests succeed under constrained memory, and failure/cancellation frees all
probe resources. Keep the current guard until that replacement is implemented.

### R0511 — Audit whole-session rollback after startup allocation failure

The recorded wide-startup failure was followed by a successful narrow probe and
then a chat-prefill out-of-memory error. Its original memory owner is not
established. Width preflight avoids that particular over-allocation but does not
prove recovery from a genuine constructor failure.

`_construct_shared_session` retries smaller contexts. The scratch allocator has
local rollback, but `Qwen35GGUFResidentSession.__post_init__` also allocates token,
hidden and other session buffers after scratch construction. There is no
constructor-wide cleanup wrapper around that sequence; retry tests alone do not
establish complete release. Normal `close()` is not proof that every partially
constructed state can safely use it.

**Deferred ownership repair:** inventory each constructor acquisition and its
release owner, add failure injection at each allocation boundary, and implement
transactional cleanup for resources not already covered. Preserve borrowed shared
weights and pre-existing pool entries. Test repeated failure/retry, late failure,
and cleanup errors without masking the original allocation error.

Then force an oversized startup ladder on the affected host and verify that a
full-context session and subsequent public chat request still allocate in the
same process, with allocation/free accounting around each attempt. Keep this
item open until that recovery check passes; do not weaken the probe or treat a
successful narrow retry as equivalent evidence.

### R0512 — Revisit Q5 decode-dual dispatch separately from working prefill

The Q5 prefill pair branch is reachable: `dense_pair_quant` explicitly includes
`gguf_q5_k_t16_v1` alongside the Q4 formats. Dispatch tests exercise its registered
owner at multiple prefill row counts. Do not remove it as dead code.

The separate rows==1 Q5 dual block still requires direct-GEMV singles, not tile8,
and inherits a policy variant that must be registered under Q5. A Q4 variant
name does not satisfy that registration. The source comment now states these
conditions rather than implying that the inherited variant always selects Q5.

**Deferred tuning decision:** re-screen the Q5 dual against the currently selected
two-single-kernel plus SiLU chain on gfx1151 before changing decode selection.
The historical screen rejected that dual; preserve its evidence in the original
ledger history at `ad2c28ff3`. If pursuing it, select a Q5-specific registered key,
verify the intended decode owner and run the applicable numerical and public
route gates. Do not change the working prefill route as part of this decision.

### R0513 — Define captured-accept logprob outputs for every published token

`_device_sampled_accept_plan` still declines logprobs because the captured graph
returns no host logits rows or equivalent token metadata. The eager path supplies
that metadata. Keep this capability fallback until the graph has an output
contract that can satisfy the request.

**Deferred graph interface:** emit the selected token's processed logprob and
requested top-logprobs for accepted draft tokens, rejection corrections and the
bonus token. The selected token need not be top-1. Bind each result to the target
row that predicted it, and discard metadata beyond stop/EOS or cancellation.
Define buffer ownership, graph replay updates and streaming publication together.

Then remove the decline and run the existing logprobs parity gate with zero delta
against the autoregressive route, confirming the captured accept actually ran.
Exercise partial acceptance, full acceptance, rejection, stop truncation and
blocking/streaming output. Existing eager-path tests are not evidence that the
graph metadata interface is implemented.

### R0514 — Define device forced-token inputs and queue commit semantics

Pending forced tokens still decline in `supports_native_gpu_sampling` and
`_device_sampled_accept_plan`. `_sampled_accept_summary` raises if a pending
override reaches the unsupported native route. The host accept path applies the
forced token as a point mass; keep that working fallback until device overrides
are implemented.

**Deferred device interface:** carry per-row forced-token overrides into sampling
and acceptance, with request-local queue position and graph replay updates.
Advance the queue only for committed published tokens, not rejected speculative
rows. Preserve queue state across rollback, partial acceptance, stop truncation
and cancellation; validate token bounds and interactions with constraints.

Remove the refusal only after the forced-token gate compares output tokens and
queue positions against the host path while telemetry confirms captured-graph
accept. Exercise forced sequences spanning cycle boundaries and ordinary rows
sharing a batch. Host-path test success does not establish device support.

### R0515 — Decide retirement of the Gemma 4 gate/up MMQ rollback

The fused Q4_K gate/up MMQ route remains enabled by the lead's decision.
`HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0` selects the fp32 grouped reference; unset
selects MMQ. Keep the implementation, registry path and correctness tests.

**Deferred rollback decision:** record the teacher-forced verdict against the
campaign's frozen evaluator before retiring the switch. Capture the reference
with `HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0`, then confirm distinct selected routes
for reference and candidate. Include short and long prefills, natural inputs and
low-margin cases; a single long-prefill comparison does not cover the recorded
length sensitivity. Preserve real expert geometry and multi-tile coverage.
This is switch retirement, not reconsideration of the default-on decision.

Historical attribution and short-prefill results remain in
`benchmarks/results/2026-09-27-gemma4-mmq-gate-up-three-arm-attribution.json` and
`benchmarks/results/2026-09-27-gemma4-mmq-gate-up-short-prefill-divergence.json`.
The original narrative is preserved in ledger history at `4b08b80fd`; do not
substitute that history or unit routing tests for the frozen-evaluator verdict.

### R0516 — Repair Gemma 4 down-MMQ continuation before retiring its lever

Automatic selection does not choose down MMQ. Explicit
`HIPENGINE_GEMMA4_MOE_DOWN_MMQ=1` and pinned `mmq` remain runnable. The recorded
continuation failure, not missing benchmark coverage, is the reason to preserve
this distinction. Keep the current default and registered implementation.

**Deferred numerical repair:** diagnose and repair the MMQ owner's generated-history
failure, then run the unchanged continuation screen and full applicable
execution-profile gate, including selected-route, repeatability and isolation
checks. Use the same model artifact for both arms:

```bash
HIPENGINE_GEMMA4_MOE_DOWN_MMQ=1 .venv/bin/python scripts/gemma4_generated_history_gate.py \
  --model <artifact>.gguf --output-dir /tmp/gemma4-down-mmq-repair
```

Retire the lever only after that repair passes; an earlier teacher-forced result
or the superseded default-on decision does not close the continuation failure.
See `worklog/entries/20261003T123946.224463Z-lhl-repair-tiled-expert-default-dispatch-after-gemma-e11ee2.md`.
Historical route attribution remains in
`benchmarks/results/2026-09-28-gemma4-prefill-expert-down-route-attribution.json`
and ledger history at `43477a865`.

### R0518 — Choose production-geometry evaluation or removal of wide-K MMQ

The `mmq32x128` Q4_K selected-dual export and Python wrapper still exist, but no
production Python route calls them. The earlier fixture overflow is repaired:
`_make_q4_k_block` now uses int32 arithmetic before casting packed fields to
uint8. Do not treat that old blocker as unfinished work.

**Deferred candidate decision:** run the narrow/wide comparison at production
geometry, or remove the unused wide export, wrapper and wide-only test cases.
The current benchmark accepts the intended geometry:

```bash
.venv/bin/python scripts/gemma4_mmq_k_width_leaf_bench.py \
  --in-features 2816 --out-features-a 1408 --out-features-b 1408
```

If selecting the wide route, preserve its input-width capability restriction,
verify numerical parity and demonstrate the selected route through public prefill
with the applicable gates. A leaf result alone does not establish end-to-end gain.
If deleting it, preserve the shared `KSUB` machinery and narrow implementation.
Historical screens remain in ledger history at `d82800038`; they are not new
production-geometry results.

### R0519 — Decide the compensated Gemma WMMA candidate independently of selectors

`HIPENGINE_GEMMA4_MOE_PREFILL` supports `auto`, `wmma`, `wmma_plain`, `mmq`,
`grouped` and `selected`. Compensated WMMA is an explicit candidate; plain WMMA
isolates weight-rounding effects. The exact modes remain useful comparison
routes. Automatic routing has evolved beyond the old exact-only description,
so the historical proposal to delete the selector wholesale is not appropriate.

**Deferred candidate decision:** evaluate the compensated route under the current
applicable execution-profile contract, then promote, repair or retire its Gemma
wiring. The old absolute-tail discussion is historical evidence, not permission
to weaken the current gate or to change unrelated defaults. Preserve exact
comparison modes, independently useful kernels and current automatic dispatch.
If retiring candidate modes, inventory their harness callers and remove only
candidate-specific wiring/tests after preserving the required comparisons.

Iteration 59 in `docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md` and ledger history
at `0bc3556f2` preserve the original numerical investigation. No new numerical
verdict is established by this triage.

### R0521 — Revisit backend-specific decode graph engagement separately from telemetry

`gemma4_decode_graph.py` still uses `_CAPTURE_MAX_KEYS` and the backend override
`GEMMA4_DECODE_GRAPH_CAPTURE_MAX_KEYS`; gfx1100 declares zero. This is an
engagement policy based on the historical capture/replay tradeoff, not a claim
that longer-context replay is unimplemented. The graph rollback and
`launched_fallbacks` reporting remain separate controls.

**Deferred optimization decision:** reduce capture/replay overhead or remeasure
the crossover on each affected physical host before changing its policy. Compare
public decode with graph selection confirmed, including capture amortization,
short and long requests, and contexts on both sides of the threshold. Preserve
replay parity and request/resource lifetime checks.

Broader engagement may retire a threshold; it does not automatically justify
removing the rollback or fallback telemetry. Retire those separately only when
no supported fallback or diagnostic use remains. Preserve backend independence
rather than applying one host's result to another. Historical measurements and
probe narrative remain in ledger history at `a8d686248` and the graph module's
design notes; this triage adds no performance or replay-correctness result.

### R0003 — Choose a DMS layout that preserves interior-expiry semantics

The append kernels still compact surviving rows in token order. A ring head
alone cannot replace this: `build_dms_live_mask` retains non-evictable history
regardless of age. For positions `[0, 1, 2, 3]`, eviction flags
`[false, true, true, true]`, and window 2, advancing the current position from
3 to 4 removes position 1 but retains position 0. Expiry is not necessarily a
prefix of the live extent. The HIP keep predicate implements the same rule.

**Deferred design decision:** choose between a prefix-only head-advance fast
path with interior compaction, or a layout that separately represents retained
history and the expiring window. The latter requires coordinated attention-reader,
allocator, snapshot, prefix-cache and speculative rollback changes; neither may
silently discard non-evictable history. Account for reserved extent bounds and
wrap/rebase behavior before changing `base_offsets`.

Clearing checks: cover interior expiry, prefix expiry, no expiry, mixed heads,
capacity boundaries and repeated wrap/rebase against the host oracle. Run
`tests/test_gpu_dms_append_decode_hip.py`, INT8 device-payload, compact-attention,
prefix and transaction tests, then public DMS generation including continuation
and rejected speculative suffixes. Profile the selected route before claiming
an append-cost improvement. Keep the existing race-safe staging until its data
movement is replaced safely. The original race repair and timings are preserved
in `worklog/entries/20261018T120817.976900Z-lhl-dms-append-compaction-race-6fa3d4.md`;
this triage does not change kernels or claim a speedup.

### R0011 — Define bounded generic-provider child ownership

`EngineService` still admits the generic spelling through speculative child
submission only when the driver also declares model-attached MTP support.
Laguna's generic provider attachment requires whole-request `generate_detailed`
and `stream_detailed` methods, not bounded child work. Its service restriction
therefore remains an unimplemented lifecycle capability, not missing benchmark
coverage. Direct provider implementation is not equivalent to service ownership.

**Deferred interface decision:** define generic child admission, bounded work
advancement, cancellation and retirement separately from the model-attached MTP
contract. Migrate Laguna DFlash and the submit/poll adapter to that interface,
then replace the compatibility gate with the actual generic-child capability.
Do not execute model work on frontend threads or hide a whole generation in an
administrative command; preserve the service's sole-driver ownership.

Clearing checks: extend `tests/test_integration_engine_service_speculative_ownership.py`
with an actual generic-only child adapter and test blocking/stream agreement,
slow work beside another request, cancellation, deadlines, provider errors,
shutdown and subsequent recovery. Then exercise generic-only generation through
`LLM.generate()` or the server and confirm the intended provider and driver
ownership. Existing tests of rejection and MTP compatibility are containment
coverage, not proof that the missing generic lifecycle is implemented.
Original repair evidence remains in
`worklog/entries/20261017T190456.513027Z-lhl-speculative-service-ownership-3d661c.md`.

### R0071 — Resolve split-prefill arithmetic before retiring its divergence pin

`tests/test_live_qwen35_gguf_chunked_prefill.py` still pins divergence for a
one-token suffix and a total context above 2048, alongside an equal-output
control at 2048. The pin covers direct prefill, not only prefix-cache reuse.
The separate batched-suffix equality test excludes the above-2048 shapes.
The completed one-token prefix-cache routing repair therefore does not close
this item. The R0077 handoff also explicitly leaves this arithmetic work open:
`worklog/entries/20261026T090655.553954Z-cleanup-refactor-r0077-8e7b36.md`.

**Deferred numerical-contract decision and repair:** establish the intended
cross-schedule contract using the execution-profile oracle, then repair the
one-row routing and long-context full-attention differences against that
contract. The old serial-reference comparison and layer-localization evidence
are preserved in `docs/REFACTOR.md` at commit `304c3324d`, under
“Split prefill diverges from a single call”. They are historical observations,
not a fresh quality verdict for current kernels. Do not cap query windows just
to force equality: the recorded experiment rejected that tradeoff.

Clearing checks: reproduce both families and the equal-output control, test
values on both sides of the boundary plus unrelated lengths and split points,
and compare independent sessions against the declared oracle. Run
`tests/test_live_qwen35_gguf_chunked_prefill.py` and the applicable numerical
profile gates; exercise public prefix reuse and fresh generation with
continuation. Replace `test_split_prefill_divergence_boundaries_are_unchanged`
with positive contract coverage only after the repair, and extend
`test_batched_suffix_extend_is_bit_exact_against_recompute` where that exactness
contract applies. Do not treat passing a test that requires inequality as proof
of numerical correctness or remove it merely to clear the ledger.

### R0074 — Decide when to remove the slow-route gapped-prefix cost guard

`HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX` remains a live admission cost policy
in `hipengine/generation/qwen35_gguf.py`, defaulting to 512. It declines a
long-suffix gapped cache hit into private full prefill only when
`_gguf_prefix_gapped_fast_route_available` is false. Contiguous hits and gapped
hits with the fast gather route bypass this threshold. This is selection
between executable routes, not refusal to generate or an artifact allowlist.

The fast-route predicate checks gather enablement, a registered BF16
`head_major_spans` copy kernel, and context/byte allocation bounds. R0312 owns
scratch-budget integration; R0506 owns the gather rollback-switch decision.
Completing either does not by itself eliminate all slow-route populations.
Historical placement and gather measurements remain in `docs/REFACTOR.md` at
commit `884e51fa4`, under “Placement decides the prefill route”. They do not
establish a current crossover on every backend.

**Deferred policy decision:** retain the cost guard for reachable slow-route
configurations, replace it with a measured cost policy, or remove it after
improving those routes. Before an optimization-driven removal, profile the
actual fallback populations against private recomputation with fragmented
placement, short and long suffixes, and allocation-boundary cases. Preserve
resource checks and cache ownership regardless of the policy selected.

Clearing checks: run `tests/test_unit_gguf_prefix_cache_runner.py`, covering
slow-route decline, short-suffix reuse and long-suffix fast-gather reuse;
exercise public generation with forced fragmentation and continuation, and
verify route telemetry and correctness against independent recomputation.
Retire the environment knob and decline branch together only when the selected
policy no longer needs them. This triage adds no timing or quality result.

### R0127 — Choose whether to replace host grammar selection

The llguidance matcher in `hipengine/generation/grammar.py` masks a host F32
copy and uses NumPy argmax for its greedy selection. Qwen4Exp controlled batch
rows request full logits; text MTP uses request-owned grammar through the
attached provider's `supports_grammar` contract. Multimodal grammar remains
rejected. The earlier MTP correction is recorded in
`worklog/entries/20261026T214808.638787Z-cleanup-refactor-r0127-r0128-status-c04382.md`.

**Deferred implementation choice:** retain host selection or replace its mask
application and selection with a registered device implementation. Profile the
controlled route before choosing an optimization. A replacement must preserve
request-local matcher state, selected-token handoff, token controls and sampling
semantics, EOS/content boundaries, cancellation and speculative commit/rollback.
Do not remove the working host route before its supported inputs have a
replacement; do not claim a speedup from reduced readback alone.

Model-owned embedded-template/XML parsing is a semantic requirement, not dead
code. Keep other model parsers independent. Do not revive generic JSON forced
prefixes or close-repair tokens on the XML route: required/named tool choice is
enforced by grammar. Unsupported XML string/schema forms must fail explicitly;
additional supported forms need lexer/parser coverage.

Clearing checks: run `tests/test_unit_grammar_masks.py` and
`tests/test_unit_qwen4exp_mtp_structured.py`, add host/device selection parity
and mixed-request isolation tests for any replacement, then exercise public
required/named tool generation with streaming, MTP and cancellation. Confirm
the intended selection route actually ran before removing host-only machinery.

### R0128 — Compose optional resource accounting and implement sparse device-position selection

`resolve_qwen4_exp_context` bounds native context by artifact metadata and
rejects explicit requests that do not fit. With a supplied prefill chunk, it
reserves the larger of the per-runner scratch floor and mandatory scratch plus
wide-F16 staging and QSA WMMA scratch. It also rounds physical KV storage to
page boundaries. These are implemented checks, not remaining cleanup.
The prior unused index estimate and obsolete full-reset note were already
removed; see
`worklog/entries/20261026T214808.638787Z-cleanup-refactor-r0127-r0128-status-c04382.md`.

**Deferred resource-design work:** compose optional MMQ sidecars, graph and
verification allocations with admission and their actual owner lifetimes.
Identify which allocations are already covered before adding reservations;
avoid double counting as well as over-admission. Preserve explicit-request
errors and keep native context, QSA dense-equivalent limits and selected-position
capacity independent. The scratch floor is not proof that every optional
combination fits.

**Separate missing capability:** device-position-owned QSA still rejects a
position beyond the dense-equivalent limit in the decode path. Implement native
sparse selection and its graph ownership before removing that guard; increasing
the limit does not supply sparse selection. This does not restrict the separate
host-position sparse route.

Clearing checks: run the scratch-plan, QSA-scratch-reserve and chunk-memory-ladder
unit tests, then exercise public admission with optional resources alone and in
combination, including failure cleanup and requests around allocation limits.
For device-position sparse support, compare selection, logits and state against
the declared oracle across the dense/sparse boundary and unrelated lengths,
then exercise public continuation and graph replay. Neither follow-up is claimed
implemented by this triage.

### R0129 — Decide when to retire Q8 MMQ prepack rollback

Runner-owned `Q8MMQWeightSidecars` already uses the registered
`weight_pack/gguf_q8_0/mmq_kmajor76` packer, publishes buffers after preparation,
and releases them on close or preparation failure. The runner prepares sidecars
only with prepack enabled and a prefill MMQ library present, and exposes them
through its bounded prefill session. A prepack setting alone does not allocate
sidecars. Keep CPU reference packing in tests, not runtime.

**Deferred rollback decision:** retain `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK`
until the enabled prepacked route passes assessment after an owner refresh and
release window, then remove redundant experiment selection.
Preserve the strict raw-Q8 route and raw weights needed by decode and risk
repair. Removing the rollback setting is distinct from changing the prefill
MMQ arithmetic profile; do not infer either from the other. R0128 owns composed
optional-resource admission, including the full additional sidecar allocation.

Historical enabled-path correctness, timings, mixed-request regression and
additional memory are preserved in `docs/REFACTOR.md` at commit `25e43b7ff`,
under “Qwen4Exp prepacked Q8 MMQ admission”. They describe that enabled
configuration, not a fresh measurement or a cost paid while no sidecars exist.
GR expansion remains separate and needs its own complete-chain/model gate.

Clearing checks: run sidecar lifecycle and execution-profile unit tests, GPU
packer byte parity and full logits/state/KV comparisons for the selected route.
Exercise public prefill with sidecars enabled and disabled, verify owner memory
and cleanup, and check mixed prefill/decode requests before a performance-driven
retirement. Preserve explicit resource and arithmetic failures rather than
silently falling back. No runtime selection changes in this triage.

### R0131 — Coordinate live-derived Surya split planning with launch bounds

`SuryaGpuRunner` still allocates partial buffers from its construction-time
capacity plan. `surya_full_attn_decode_f32_spans` derives both chunk and split
count from `spans.max_live_count`, which is the runner's capacity bound for this
immutable view. Historical context-sensitivity measurements remain in
`benchmarks/results/2026-09-13-gfx1151-surya-context-default.json` and the ledger
at commit `f2b9c188e`.

The old proposed chunk-only override is insufficient: the executed planner
returns `(96, 53)` for capacity 5000 and `(64, 64)` for live count 4096, but
capacity 5000 with chunk 64 returns `(64, 79)`. Allocating only
`SURYA_TARGET_SPLITS` partials while retaining that wrapper calculation would
under-allocate. Do not apply the old one-line proposal.

**Deferred arithmetic/launch design:** pass a validated live launch bound and
consistent split count through the wrapper, or provide storage for the actual
capacity-derived count. Keep the physical page/span bounds separate from the
live work bound, and size all three partial buffers for the launch they serve.
Account for growing live counts, graph reuse and empty producer ranges.

Clearing checks: add host launch/allocation regressions for unequal capacity and
live counts, including the example above and alignment boundaries. Run
`tests/test_gpu_surya_kv_spans.py`, the applicable production numerical gate and
strict fallback checks, then public Surya continuation at fixed live lengths
across different configured capacities. Profile before claiming convergence or
speedup. Retire capacity-derived chunking only after the coordinated replacement
passes; this triage changes neither launch behavior nor arithmetic.

### R0132 — Consolidate native context bounds without restoring evidence admission

The gfx1100 dense H5120 Q4_K_M cache-capacity policy already bypasses the old
p95 cap. `_backend_native_context_limit` accepts BF16 and INT8-per-token-head
KV with FP32 recurrent state for that policy; the old BF16-only description
was stale. Other configurations still encounter the backend fallback limit.
Graph admission separately checks capacity, rows, kernel transitions and split
workspace boundaries. Preserve those executable constraints and the independent
graph metadata producer.

**Deferred consolidation:** replace the generic cap/cache-policy split with
capability-based bounds after diagnosing the recorded MoE full-logit discrepancy.
The historical p8 observation is preserved in `docs/REFACTOR.md` at commit
`40a912a08`; it is not a fresh failure result or justification for blocking every
unmeasured context. Exercise the affected route, identify its actual failure
scope, and retain eager/serial execution where graph capability is absent.
Do not restore a 256 cap merely because longer contexts lack measurements.

R0382 owns long-context staged/scalar orchestration. `strict_long_rows` is
conditional on linear attention and staged-policy selection, not a blanket
scalarization of all long-context verification. Keep its strict arithmetic
contract separate from context admission. Public requests must be admitted by
implementation and resource capability, not benchmark records or serving-key
identity; automatic variant policy and arithmetic promotion are separate.

Clearing checks: run native-context and native-cycle unit tests, then the
native-context state/logit gates across affected geometries, KV layouts and
boundary-straddling rows. Verify public generation, every accepted-prefix state,
continuation, rollback and graph/eager route attribution before consolidating
bounds. This triage changes documentation only, not the surviving cap.

### R0137 — Move profile settings to owners without duplicating admission checks

GGUF profile binders still write process-wide environment settings, including
`qwen35_dense_gfx1100_profiles.py`, `qwen35_dense_gfx1151_profiles.py` and
`qwen35_moe_gfx1151_profiles.py` under `hipengine/generation/`. Separate model
owners can therefore share mutable configuration rather than independent
resolved settings.

**Deferred ownership design:** move settings needed by coexisting owners into
explicit fields consumed by those owners. Inventory each actual consumer and
its binding lifetime; do not surround every method with an implicit environment
context. Preserve ordinary profile plugin composition and header identity.

`Qwen35GGUFAdmissionCertificate` is still consumed by the loader, which checks
that a supplied certificate covers the intended plan. Simplification must name
specific redundant work and preserve shape/dtype, operation, slot and resident
plan checks. Source identity binds a supplied result to its source; it must not
become an allowlist of measured artifacts. An unspecified replacement framework
is not a clearing action.

The removed generic qualifiers, profile prototype and recursive per-call operand
scans are history, preserved in the ledger at `e30addbd5`. Do not recreate them
to defend unsupported private pointer mutation. Keep closed-owner and request
bounds, and add invalidation only for actual supported reconfiguration.

Clearing checks: test two differently configured owners in both construction
orders and interleaved execution, confirm no settings leak, then exercise public
model generation and teardown. For certificate changes, run intended-plan,
partial-slot and supplied-certificate loader tests in
`tests/test_live_gguf_ud_admission.py`. Keep construction checks separate from
per-token execution; this triage changes neither mechanism.

### R0138 — Complete native adapter ownership without blanket embedding refusal

Native target graph admission in `gguf_native_spec_cycle.py` requests
`_device_token_embedding_weight(reason="native_mtp_graph")` when host embedding
is enabled and that fallback exists. It refuses when no fallback is callable;
it does not categorically reject every host-embedding configuration. Expert
sidecars are separately refused because that graph requires resident replacement
expert layouts. Keep graph capture constraints distinct from eager execution.

**Deferred adapter work:** represent any additional embedding staging or expert
sidecar route in the actual shared call plan, including pre-allocation shape,
dtype, physical operands, lifetime and capture behavior. Remove only the
corresponding refusal once that implementation exists. Do not remove the working
device-resident embedding fallback or infer support from a nominal adapter name.

The legacy diagnostic was already removed with authoritative execution-report
coverage in
`worklog/entries/20261026T222549.230606Z-cleanup-refactor-r0138-native-diagnostic-12a5fc.md`.
The old invocation context, receipts and recursive inventories are completed
history, not mechanisms to resurrect. Preserve construction-time checks and
runtime owner/request bounds; supported reconfiguration needs narrow invalidation,
not per-layer full-model validation.

Clearing checks: extend `tests/test_integration_gguf_execution_authorization.py`
and `tests/test_integration_gguf_native_physical_operands.py` for each added
adapter, then test graph capture/replay, allocation failure and teardown. Exercise
public generation with the intended layout and confirm graph versus eager route
selection and numerical/state parity. This triage changes no runtime behavior.

### R0139 — Replace file-type policy predicates with actual operand contracts

`hipengine.loading.qwen35_gguf_policy` still reads
`GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES` and
`GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES`. Shared policy extraction removed
an audit/runtime mirror, not these stamp predicates. A header file-type label
does not describe every tensor's role, type or resident layout.

**Deferred policy migration:** inventory the stamp consumers in recurrent state,
qmicro, NextN/GDN, scratch/source-F16, Q6 and automatic MTP, then bind each to
actual role/shape/type/layout and execution-profile requirements. Separate
kernel capability from arithmetic-default selection: unfamiliar compatible
artifacts must not require a benchmark record to execute. Do not use a histogram
or successful load as proof of an arithmetic contract. Preserve existing
precision while separating global IQ repack eligibility from F32 contraction.

The route audit already uses canonical token-embedding paths, separate AR/NextN
scopes, partial-header diagnostics, sidecar/formula-byte accounting and the shared
policy API. Coverage is in `tests/test_unit_scripts_gguf_quant_route_audit.py`;
those completed report repairs are no longer open work. The historical UD K_S
planning failures and original campaign analysis remain in the ledger at
`9680870a2` and [UD-QUANTS.md](campaigns/UD-QUANTS.md), not fresh failure claims.

Clearing checks: test identical stamps with different maps, identical histograms
with types exchanged between sensitive roles, partial NextN plans and optional
sidecars. Run shared-policy and route-audit tests, then public loading/generation
for compatible unseen tensor maps and explicit incompatible requests. Any changed
arithmetic default must pass its profile gate. Preserve source-bound certificate
consistency without turning source identity into measured-artifact admission.
The source inventory's R0163 bullets at lines 3770–3817 belong to this item;
its unrelated speculative-default migration remains separate.

### R0141 — Decide whether to keep the explicit GDN w32 diagnostic

`qwen4exp_gdn_w32_prefill` remains explicitly registered to
`qwen4_exp_gdn_prefill_w32_f32`; its HIP launch symbol and dedicated parity test
still exist. R0140 completed tile16 scoped ownership, but that does not remove
the w32 registry entry or its direct diagnostic caller. Do not describe the
current default as universally columnwarps or infer w32 unreachability from
another variant's rollout.

**Deferred product decision:** retain the w32 geometry as an explicit occupancy
comparison or retire it because that diagnostic is no longer wanted. The prior
entry expressly reserved this decision; historical timing and parity evidence
remain in
`benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf5-gdn-w32-prefill-ab.json`.
This triage adds no performance or device-parity result.

If retired, remove the registry entry, wrapper, HIP instantiation and dedicated
test together after checking all direct consumers. Preserve the strict owner
and selected tile16/columnwarp routes. Run the applicable lineage, kernel,
registry and public-generation checks for that change. If retained, keep its
exact-parity diagnostic contract; historical RED-first prose is not evidence
that the implemented symbol is absent.

### R0147 — Consolidate the Q8 MMQ attention-gate candidate selector

`HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE` extends an existing Q8 MMQ prefill
policy with K2560/N6144 at a minimum of 64 rows. It does not create the policy
or allocate MMQ resources, and strict/production profile bindings still set it
to zero. Plane-count selection in the same helper is a separate control.

The recorded state-repeat failure was repaired; the subsequent comparison was
rejected for noisy performance, not state divergence. Preserve that distinction
and the original evidence:
`worklog/entries/20260901T220425.712695Z-lhl-qwen4exp-p3-q8-mmq-attn-gate-current-blocked-4fa414.md`
and `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-fivepair-rejected.json`.
These are historical results, not fresh measurements of today's composition.

**Deferred selector decision:** preserve the explicit comparison or replace it
with an operation-complete dense/GR policy owner. Remove the separate setting
once that owner represents the same shape and lifetime requirements. Preserve
exact raw-Q8 fallback and keep this scope independent of prepack/resource
selection. A performance-driven default change needs stabilized comparisons;
a changed-arithmetic default also needs its execution-profile gate.

Clearing checks: run attention-gate policy and profile unit tests, then exercise
the enabled route through public prefill and confirm actual dispatch. Check full
logits/state, repeatability and cleanup before retiring the comparison. Do not
silently claim engagement from setting this flag while the MMQ owner is absent.
This triage changes no runtime or default.

### R0149 — Replace remaining Qwen4Exp profile environment transport

**Deferred ownership migration:** replace `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL`,
`HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL`,
`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` and
`HIPENGINE_QWEN4_EXP_Q4_DP4A64{,_LAYERS}` with explicit resolved owner state
at their consumers. The binder still resets the Q8 WMMA layer list to empty;
provide a getter and explicit diagnostic override instead of requiring callers
to know binder order. Test independent owners in both construction orders and
interleaved requests before removing the environment transport.

Q5_1/Q4_K MMQ selector/layer-list transport and workspace branches were already
removed in
`worklog/entries/20261026T233425.920612Z-cleanup-refactor-r0149-mmq-0b7fec.md`.
Keep the registered diagnostic kernels and exact coltile-Q8/grouped Q4/Q5
fallbacks. Ownership migration must not widen MMQ shape support or change the
strict handling of hc-down and unsupported K alignment. R0128 owns optional
resource accounting; R0129 and R0147 own their distinct prepack/scope decisions.
Historical WMMA-MoE27 results remain in
`benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json`.

The UD-Q4_K_XL recovery disables Q4 DP4A because of the observed long-context
decode discrepancy, recorded in
`benchmarks/results/20261022-qwen38fn-q4-decode-repair.json`.
Removing global transport does not repair or promote that arithmetic. Preserve
the strict logical128/t64 owner and explicit diagnostic layer selection.
To reconsider DP4A, repair the operation and use the explicit candidate that
pins the logical128 consumer. Set `MODEL_ROOT` to the model directory and
`LONG_PROMPTS` to the long numerical JSONL fixture containing the 12,287-token
failure and controls:

```bash
python3 scripts/qwen4exp_layer2_profile_gate.py --model-root "$MODEL_ROOT" --candidate decode_dp4a_linear --decode-steps 32 --prefill-chunk-size 1024 --output /tmp/q4-decode-short.json
python3 scripts/qwen4exp_long_context_gate.py --model-root "$MODEL_ROOT" --candidate decode_dp4a_linear --numeric-prompts "$LONG_PROMPTS" --max-sequence-length 20000 --prefill-chunk-size 1024 --candidate-repeats 3 --output /tmp/q4-decode-long.json
```

Both numerical gates and complete task/serving gates must pass before restoring
that arithmetic default; a two-trajectory screen is insufficient. For transport
migration, run profile and owner-isolation tests and public generation confirming
selected routes, cleanup and unchanged outputs. This triage changes no runtime.

### R0151 — Separate prefill staging ownership from implemented AR batching

`Qwen4ExpResidentServingRunner._prepare_ar_batch` builds a shared
`Qwen4ExpARBatchOwner` for resident plain-AR requests, and `_step_ar_batch`
advances batched rows in one owner call. The old serial-only c2 description is
obsolete. External runners retain the c1 path because they do not own resident
batch allocations; MTP has separate ownership.

The runner pool still supplies staging runners, reusing the primary runner and
preparing additional runners up to `staging_capacity`. Multi-slot AR decode is
not sufficient reason to delete that pool or its request-transfer lifecycle.

**Deferred staging design:** determine whether prefill and remaining c1/MTP
consumers can use smaller dedicated workspaces or shared state storage without
losing independent request ownership. Inventory reserve/import/demotion and
reclaim consumers before replacing allocations. Preserve shared-once weights
and PLE, request-local KV/state, cancellation and failure cleanup. R0006 owns
the shared-decode follow-ups; do not reopen its completed batch implementation.

Clearing checks: run AR-batch admission/capacity/decode unit tests and public
mixed-prefill/decode requests, including transfer, cancellation and continuation.
Confirm actual batch engagement and staging allocation lifetimes. Any claimed
memory or speed improvement needs measured comparison; this triage changes no
runtime behavior and reports no new performance result.

### R0152 — Expand vision capacity and integrate typed media ownership separately

`Qwen4ExpVisionRunner` defaults `max_patch_tokens` to 256, validates a positive
multiple of four, and rejects inputs beyond its allocated capacity. The argument
is configurable; 256 is not a universal kernel limit. The HTTP capability report
still advertises bounded PNG data URLs and a 1024-token request limit.

**Deferred capacity work:** connect larger supported grids to explicit memory
admission and validate them against the vision reference before changing public
limits. Profile the existing attention implementation before selecting a faster
replacement; do not treat changing the limit as providing its storage or
numerical contract.

**Separate lifecycle integration:** replace the dedicated multimodal chat branch
only when the normal request owner carries typed media, cancellation, exact token
accounting and streaming state. Current HTTP checks reject streaming, n other
than one, tools, sessions, continuation, grammar and speculative generation.
Those missing semantics are not implemented by widening vision buffers. Remote
media handling needs an explicit fetching/resource policy; preserve bounded
inline decoding until that exists.

Clearing checks: run `tests/test_unit_qwen4_exp_multimodal.py` and
`tests/test_integration_qwen4_exp_server_multimodal.py`; add grid-boundary,
allocation-failure and mixed-request lifecycle coverage for the chosen extension.
Run vision reference/device parity and real public HTTP media requests, including
cancellation and malformed inputs. Remove only restrictions whose replacement
semantics actually run. This triage changes no runtime or public limit.

### R0153 — Retire raw-Q8 and expert-grid transport without deleting live ladders

`HIPENGINE_QWEN4_EXP_RAW_ROWBATCH` and `HIPENGINE_QWEN4_EXP_RAW_VARIANT`
remain prefill-time selectors, defaulting to 32 and `coltile8`. The runner can
substitute `coltile8_wave_scale` when its owned wave-scale decision is enabled;
the old fixed-owner description does not capture that composition.
`HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID` remains a shared selector read by the
separate exact Q4/Q5 grouped-owner ladders. Family attribution and fallback
selection still have callers and tests.

**Deferred ownership/retirement decision:** resolve these settings into explicit
owner state or retain a scoped diagnostic comparison. Remove environment reads
only after migrating callers and diagnostic overrides. Audit each geometry's
actual registry and dispatch consumers before removing wrappers; the old phrase
“losing geometry ladder” is not evidence that every non-default rung is dead.
Keep registered scalar raw-Q8 and direct selected strict primitives.

Clearing checks: run raw wave-scope, Q4 out4 and Q5 out8 scope tests, covering
family overrides and parent selection. For an implementation change, verify
full-output/state parity, independent owners, public prefill, cancellation and
graph replay with the selected route observed. Any kernel deletion also needs
lineage/registry checks. No arithmetic or runtime change is made by this triage.

### R0163 — Decide the scope of the FP16 recurrent-storage experiment

Both strict and production binders in
`hipengine/generation/qwen35_dense_gfx1151_profiles.py` select FP32 recurrent
state. The recorded long-horizon C1/packed failure is not cleared by short
verifier evidence. FP16 kernels and legacy Q4_K_S selection remain separate;
R0139 owns the file-type policy migration and R0137 owns process-global binding.
The misleading speculative-MTP migration heading is retired.

**Deferred storage decision:** retire the unbounded FP16 experiment or replace
it with a geometry/profile/horizon-owned storage plan. A replacement must pass
full long-horizon numerical and serving gates before changing the arithmetic
default. Do not re-enable it by copying the historical D24 verifier manifest.
Keep registered FP32 execution and distinguish other configurations from the
recorded failing scope rather than treating all FP16 routes as failed.

Clearing checks: run dense gfx1151 profile unit tests, then compare complete
recurrent state, logits, continuation and accepted-prefix rollback across short
and long histories for any replacement. Exercise public AR and packed MTP with
storage attribution and lifecycle checks. The original inventory's file-type
bullets were reconciled under R0139; its planar-Q6 environment item was resolved
under R0398. Neither is reopened here. Runtime and defaults are unchanged.

### R0164 — Separate packed-target provenance from graph-cache consolidation

The historical singleton substitution means its timings cannot be relabelled as
packed-target measurements. New claims must identify the actual target and
exercise numerical/task, lifecycle, slot-transition and repeated timing gates.
That evidence obligation does not make an implemented route diagnostic-only:
R0077 removed screening admission, and explicit requests follow declared
capability. Do not restore an evidence-key allowlist.

`MTP2_MAX_CANDIDATE_DEPTH` already centralizes the structural bound in
`hipengine/generation/qwen35_gguf_mtp2.py`. Its cleanup is recorded in
`worklog/entries/20261026T233233.859722Z-cleanup-refactor-r0164-depth-bound-ca1a20.md`.
The obsolete screening and duplicated-depth work are closed. Runtime injection
probes remain diagnostics, not proof of public dispatch.

**Deferred graph-owner cleanup:** identify any remaining duplicated depth-specific
cache fields and consolidate only equivalent ownership/lifetime behavior. A
single bound does not imply every graph has identical shape or capture operands.
Preserve uncached eager execution, singleton versus packed target identity,
refill transitions, cancellation and teardown. Do not merge graph ownership
solely because depths fit the same numeric range.

Clearing checks: run MTP2 seam tests for declared depth, uncached eager execution
and singleton/refill behavior; test every affected cache lifecycle before removing
fields. Exercise public generation with target attribution and all acceptance
prefixes. Historical claims remain in the ledger at `b2f30e52c`; this triage
changes no runtime and adds no packed-target measurement.

### R0173 — Retire the R28 rollback without removing shared rowtile machinery

`HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28` still controls the periodic
fused R28 policy. The dense gfx1100 profile binder writes zero for strict and
preserves an explicit production rollback. The backend schedule keeps layers
0/8/…/56 exact while allowing the other applicable layers to use the fused owner.
Historical evidence is
`benchmarks/results/2026-09-02-w7900-q4km-k3-c7-fused-r28-periodic-strict-retained.json`.

**Deferred rollback decision:** retain the explicit comparison or replace its
process-global transport with owner-bound policy after the release review.
Preserve strict-profile selection, omitted-profile behavior, periodic exact
layers and registered strict fallback. R0137 owns broader profile migration.
The `_rowtile_variant_policy_env_cache` in `gguf_linear.py` serves multiple
policies; do not delete it or unrelated tests as though it belonged only to R28.
Remove only obsolete switch-specific entries and coverage after callers migrate.

Clearing checks: run gfx1100 profile tests, exercise strict/production/rollback
in both owner construction orders, then verify exact-layer schedule and public
R28 dispatch with full-output/state and lifecycle checks. Any changed arithmetic
selection requires its profile gate. This triage changes no runtime or default
and does not renew the historical performance claim.

### RF-1 / R0187 — Finish owner-bound policy resolution using existing profile infrastructure

`ResolvedRuntimeProfile` in `hipengine/execution_profiles.py` already carries
an immutable profile manifest and plugin construction/binding hooks. It does not
resolve all launch decisions: `gguf_linear.py` still reads environment settings
and backend policies, and profile binders can write process-global state.
Do not introduce a parallel profile framework merely to satisfy the old proposed
`ResolvedRuntimePlan` name.

**Deferred ownership design:** extend the existing plugin-owned construction
boundary with typed resolved decisions and explicit resource/shape-dependent
plans. Identify which values belong to a model, session or request and which
must vary with live shape. Parse user environment input at adapters, not in hot
methods; preserve explicit scoped diagnostics without ambient-state leakage.
R0137 and R0149 own concrete GGUF/Qwen4Exp transport migrations under this goal.

Registry misses may use only a declared fallback that supports the requested
semantics. A missing capability must remain an explicit error, not silently
select unrelated strict behavior. Keep four-axis registry ownership and strict
arithmetic contracts; graph capture must bind the same decisions as eager work.

Clearing checks: test independently configured owners in both construction
orders, interleaved requests, strict/production selection and unsupported keys.
Verify migrated hot methods no longer reread their moved settings and exercise
public generation with selected-route attribution. Existing profile unit tests
cover the current abstraction, not completion of this migration. This triage
changes no runtime or architecture.

### RF-3 / R0187 — Decide source-script retention and replacement ownership

RF-2 removed research scripts from the wheel. Source-workspace commands and
sdist scripts remain available. The historical unreferenced-script count at
`c5ded0003` is not a current, named deletion inventory; absence of textual
references does not prove a command or numerical oracle is superseded.

**Deferred decision:** choose the supported source-command families and their
replacement entry points before consolidating or retiring historical drivers.
For each proposed deletion, record the path, current callers and published
protocol references, replacement command (or why none is needed), and the
immutable result hash and decision it reproduces. Preserve active oracles and
commands required to reproduce evidence. Parameter-only drivers may consolidate
only after their argument, output-schema and failure contracts are compared.

Clearing checks: a path-by-path disposition inventory, replacement command help
and focused driver tests, published-command drift checks, and verification that
each retained evidence reference is still reproducible or explicitly archived.
Do not delete an unnamed batch based on the old count.

The other RF-3 deletion scopes are reconciled: rejected compatibility selectors
and aliases were handled under R0281; live layout/verifier decisions remain at
R0281-DRAFT-X8, R0281-SELECTED-X8 and R0281-TARGET-POLICY. Native-XL loading is
capability-based and its provenance constants have active verifier/test users.
This triage does not complete the source-script consolidation.

### RF-4 / R0187 — Migrate the ledger after coverage reconciliation

**Deferred dependency:** complete the current sweep's original-item and subitem
coverage audit before replacing the ledger structure. The manifest under
`audit/refactor-sweep/` preserves original text and dispositions; it is not yet
proof that every current heading and remaining decision has an owner.

The migration must produce a compact active index with stable IDs, affected
paths, owner/scope, state (`ready`, `blocked`, `keep-fallback`), and an exact
clearing condition. Preserve still-binding numerical and lifecycle conditions
in linked current contracts rather than archive them as history. Move only
closed narrative to immutable worklogs or a separately indexed history page;
keep existing evidence links resolvable. Do not generate IDs from mutable line
numbers or silently merge distinct subitems.

Add a semantic checker for duplicate active IDs, closed entries in the active
table, required fields and dangling detail links; wire it into
`scripts/docs/check_all.py` and test malformed as well as valid ledgers. The
existing documentation gates validate indexes, environment coverage, benchmark
exports and published commands, not this ledger schema. No such checker or
format migration is claimed by this triage.

### RF-5 / R0187 — Choose compact decision schema and durable capture storage

**Deferred storage/schema decision:** choose a retrievable destination for large
captures and define which fields belong in compact decisions versus referenced
payloads. The proposed default decision ceiling is 256 KiB, not an existing
validation rule. Large accepted JSON results cannot be treated as disposable
logs merely because they exceed it. The provenance and published-command gates
do not impose that size limit.

Before migration, inventory paths, consumers, references, hashes and verdicts;
separate raw tensors/traces from summaries that existing validators consume.
Keep current and linked decisions available. External or LFS payloads need a
SHA-256, retrieval metadata and an exercised retrieval command. Ignored local
storage alone is not a durable replacement for published evidence.

Clearing checks: version the compact schema, test size enforcement and malformed
references, replay representative accepted/rejected decision validators after
retrieval, and verify all existing evidence links or explicit archival redirects.
Only then remove duplicated captures from tracked storage. Preserve historical
bytes and decisions; do not rewrite measured results to fit a size target.
This triage removes the stale directory-size claim, not artifacts, and does not
claim the schema or storage migration is implemented.

### RF-6 / R0187 — Finish runner boundaries without duplicating extracted owners

GGUF already delegates prefill planning, canonical/provisional state storage,
prefix snapshots, scalar attention/shared feed-forward execution, and diagnostic
capture/timing to `qwen35_gguf_*` modules. Their integrated validation is recorded
in `worklog/entries/20261025T121408.652145Z-cleanup-qwen-refactor-validation-6f9090.md`;
`docs/reference/QWEN-GGUF-RUNTIME.md` describes the implemented boundaries.
Do not recreate these owners inside the historical proposed package tree.

**Deferred boundary design:** split remaining bulk attention orchestration,
full-stack verification and graph lifetime only with explicit inputs and one
session/transaction owner. Decide which GGUF boundaries are reusable by PARO
before decomposing `qwen35_paro.py`; a whole-runner proxy or implicit-session
mixin is not an ownership split. Preserve import compatibility through thin
facades during migration and give each compatibility export a removal plan.

Clearing checks: separate move-only units from numerical/dispatch changes;
compare selected operations and output/state against the pre-move control,
exercise failure cleanup, graph lifetime, prefix restore and cancellation/reuse,
and verify public ordinary and speculative requests. Update PLAN only for
implemented boundaries. The prior bounded GGUF extraction is complete; this
remaining GGUF/PARO decomposition is not.

### RF-7 / R0187 — Split HTTP composition around the existing generation service

`hipengine/server/multimodal.py` and `hipengine/server/tool_stream.py` already own separate helpers,
but `hipengine/server/api.py` still combines protocol schemas, validation, stream
accounting, metrics, model lifecycle, transport queueing and route bodies.
`hipengine/generation/engine_service.py` already owns the generation service; the server
must not introduce a replacement scheduler during decomposition.

**Deferred boundary design:** define explicit application-state dependencies for
routers, model lifecycle, metrics and stream translation, then extract cohesive
units so `create_app()` composes them. Preserve transport backpressure and
queue admission while keeping device scheduling, request state and commit
ownership in the generation engine. Separate OpenAI schemas, tool/structured
validation and SSE accounting without changing public errors or response shapes.

Clearing checks: preserve public/import compatibility during move-only units;
exercise HTTP ordinary/streamed requests, tool and structured responses,
disconnect/cancel, queue saturation, startup failure, shutdown and model reuse.
Verify readiness and service-health attribution still report the same engine.
A real `hipengine serve` request must confirm the intended service path, not
only mocked route tests. The split is not implemented by this triage; the old
closure-size estimate is removed rather than presented as a current measurement.

### RF-8 / R0187 — Separate backend policy manifests from registration

`hipengine/kernels/policy.py` already provides frozen geometry identities shared
by backend policies. Backend initializers, including
`hipengine/kernels/hip_gfx1151/__init__.py`, still combine policy declarations,
implementation imports and alias registration. Geometry records alone do not
complete the proposed manifest split.

**Deferred boundary design:** divide immutable capability/policy declarations
by subsystem and quantization, retaining one backend package aggregator and
explicit validation. Reuse existing geometry types; do not invent a second
registry or move backend/quant conditionals into engine dispatch. Model names
and artifact stamps remain provenance, not admission keys.

Clearing checks: compare the exact `(backend, layer, quant, variant)` key set,
implementations, declared fallbacks and capability values before and after each
move. Cover cold import, repeated registration, replacement semantics and peer
backend isolation. Keep source-lineage mapping and alias ownership readable;
generate repetitive registrations only when those checks remain explicit.
Verify public dispatch for affected families before declaring a migration done.
No policy values, registrations or defaults change in this triage.

### RF-9 / R0187 — Partition Laguna attention with explicit build ownership

`hipengine/kernels/hip_gfx1100/attention/laguna_kv.py` still builds the shared
`laguna_kv_attention` library from one source and exposes KV-write, global and
sliding-window decode/prefill families. Earlier rejected-route removals do not
prove that remaining wrappers or bodies are unused. Historical file-size counts
are not deletion criteria.

**Deferred build-boundary decision:** define separate family ownership for KV
write, global attention, sliding-window attention and test-only experiments.
Inventory each exported symbol, registration, runtime caller and oracle before
moving or deleting it. A body without a public Python wrapper may still be a
native helper or exported test control. Preserve registered strict fallback
chains and active negative controls; archive only genuinely unconsumed code.

Clearing checks: preserve production names, raw-pointer and KVLiveSpans ABIs,
source lineage and exact registry mappings. Explicitly account for build-cache
keys, cached-only operation, library lifetime and graph-held symbols after any
source split. Run affected family parity/lifecycle tests and public generation
with selected-symbol attribution; keep mechanical movement separate from
arithmetic or default changes. No source split or kernel deletion is implemented
by this triage.

### RF-10 / R0187 — Separate launch planning from direct-control migration

`hipengine/runtime/gguf_linear.py` still resolves pair and fused projection
choices through launch-time branches. **Deferred plan design:** resolve typed
launch records at the owning shape/resource boundary, reusing RF-1 profile
infrastructure rather than caching decisions across incompatible owners. Keep
unsupported-shape fallbacks and explicit capability errors; do not bake device
pointers or stale scratch generations into reusable policy records.

`SubmitPollTextGenerator.submit_speculative_many_detailed()` already chooses
staged submission when supported. Its whole-request compatibility branch still
serves providers without staged support. The existing “SPECDEC2 whole-request
compatibility seam” item owns that provider migration; direct control code is
not dead merely because it is callable from a production module. Move independent
oracles only after identifying all live callers and preserving their contracts.

Clearing checks: compare exact selected operations, arguments and fallback
reasons across scalar/pair/triple/residual shapes, both sides of thresholds and
unrelated shapes. Exercise owner isolation, graph reuse and scratch invalidation.
For direct-control retirement verify public staged request selection, exact
commit/rollback state, cancellation, late admission and provider fallback;
each request must keep one scheduler and commit owner. Neither migration is
implemented by this triage.

### RF-11 / R0187 — Replace remaining broad live-source pins with scoped contracts

`tests/_laguna_policy_pin.py` already checks Laguna policy assignments instead
of hashing the shared gfx1100 initializer. The H8B source-default test preserves
historical ledger provenance inside its immutable artifact but no longer hashes
the live ledger or the peer initializer. It still hashes whole live runner and
MoE files through `_POST_MERGE_SOURCE_SHA256`; unrelated edits can still force
pin updates. Do not describe the earlier ledger/backend fix as the whole task.

**Deferred contract decision:** identify the semantic owner or frozen block each
remaining broad live-source pin protects, then replace it with assertions over
capabilities, exact registry keys, call boundaries and relevant source blocks.
Keep hashes of immutable evidence unchanged. Do not delete a hash merely to get
a passing test, and do not refresh it as evidence that behavior was preserved.

Clearing checks: inventory each migrated pin and its replacement contract; prove
that an unrelated source edit is accepted while mutations of the protected
policy, dispatch or ownership behavior fail. Run the associated focused runtime
and evidence-validator tests. Preserve numerical gates independently of source
provenance. Remaining runner/MoE pins are not migrated by this triage.

### RF-12 / R0187 — Split integration fixtures and refresh measured budgets

The scheduler and server integration modules remain broad behavior collections.
Other integration tests import `FakeLLM`, `SpeculativeMTPFakeLLM` and stream
helpers from `tests/test_integration_server_api.py`; moving test bodies without
extracting these shared fixtures would preserve or break that coupling.

**Deferred test-boundary design:** extract shared fixtures into explicit test
support, then split tests by behavior and ownership. Preserve parametrizations,
marks, HIP guards, setup/teardown and import compatibility during migration.
Follow the tier naming/discovery rules in `docs/TESTING.md`; do not silently move
expensive integration cases into default unit collection or duplicate cases
through compatibility re-exports.

Clearing checks: compare collected test identities and parameter coverage before
and after the split, exercise affected bundles in both ordinary and audit-first
collection orders, and verify no fixture/global-state leakage. Refresh PLAN's
current/target budgets with an executed, reproducible counting command and a
clear distinction between measured code and historical roadmap estimates. Add
growth checks only after meaningful module boundaries exist, not arbitrary
limits that reward moving code without changing ownership. Neither test splitting
nor budget refresh is implemented by this triage.

### R0187 — Simplification umbrella: preservation rules and deferred scope

RF-2's wheel ownership repair is implemented. The remaining RF decisions above
are deferred, not completed migrations. The original inventory and illustrative
directory tree remain in `docs/REFACTOR.md` at `c5ded0003`; current architecture
belongs to PLAN and the subsystem contracts, not those proposed filenames.

Preserve registered strict fused/unfused chains, complete-KVLiveSpans generic
attention/KV-write and unsupported-shape/backend fallbacks, scalar/segmented/
transactional oracles still used for comparisons, CPU-reference kernels, strict
fixtures, lifecycle/isolation gates and compact negative evidence. Preserve
explicit fallbacks required by an open numerical or hardware gap. Counts alone
are not grounds to delete any of these.

Remove ready rejected production selectors before mechanical moves, retaining
primitives only for named fallbacks, active oracles or current research. Search
runtime, tests and published protocols before deletion and name its focused gate.
Keep deletion, policy extraction and movement separate from tuning, arithmetic
and default changes. Preserve imports during migration, then retire temporary
re-exports after their compatibility window. Update PLAN when boundaries are
implemented. Keep primitive wrappers near their families; do not introduce an
untyped universal pointer launcher. Prove GGUF boundaries avoid import cycles
before applying them to PARO.

Implementation completion still requires owner-bound policy, removal of rejected
production reachability, a bounded indexed ledger and one scheduler/state/KV
commit owner per request. RF-2 alone satisfies only wheel exclusion. A completed
triage manifest does not establish those implementation outcomes; the user's
current sweep allows these named decisions to remain deferred for later review.

### R0189 — Packed-PARO transaction storage after journal integration

The shared checkpoint journal now owns PARO target/provider rollback.
`_checkpoint_target` in `hipengine/generation/qwen35_paro_mtp2.py` copies mutable
recurrent/hidden state to the spare slot with `kv_rows=0`, records its cursor,
and restores through the same journal. It does not copy the live KV prefix.
The implementation and separate gfx1151 validation are recorded in
`worklog/entries/20261024T114318.665675Z-mtp-speculative-state-journal-c81b36.md`;
that evidence is not a fresh gfx1100 result or an MTP-versus-AR comparison.

**Deferred storage design:** decide whether remaining target result, acceptance,
selected-commit and provider-update metadata and mutable backups should share
stable claim-backed slabs. Preserve one journal and explicit buffer lifetimes;
do not replace the working rollback with a second coordinator. Verify exact
state/cursors/live KV on rejected, partial and full acceptance, capture/restore
failure, cancellation and request reuse before retiring the spare-slot path.

Any multi-request expansion must implement physical proposal and target
ownership rather than broaden the singleton adapter's admission. Compare full
request wall time with matched AR/direct controls when claiming a performance
benefit. The old automatic-K0/default-off statement was a historical bring-up
result, not a current admission rule; current model/provider capability and
selection policy remain authoritative. No runtime defaults change here.

### R0195 — Consolidate physical-target policy without changing admission

The GGUF MTP adapter still reads `HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS`
and `HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES` during construction.
Exact row counts, padding and Q6 partitions also depend on backend capabilities
and execution profile. These controls are not retired by earlier C7/C8 policy
cleanup. Historical physical-C2 measurements and row partitions remain in this
ledger at `df87e047f`; they are not current universal admission bounds.

**Deferred ownership decision:** consolidate exact rows, mixed-Q6 partitions,
prompt streaming and padding into a profile-resolved physical-target manifest
keyed by concurrency, target rows and shape. Reuse RF-1/RF-10 ownership rather
than add a parallel selector. Retire the two rollback flags only with explicit
coverage of their replacement policies and diagnostic controls.

Preserve padded rows6 decomposition, repeated-R6 and serial-row strict owners,
ordinary-prefill behavior, scope-miss fallbacks and unfused pair ownership.
Keep synchronized callers synchronized; enqueue-only proposal and bounded
blocking accept readback have separate lifetime contracts. Preserve independent
physical-target and C1 policy fingerprints and historical evidence bounds as
provenance, not identity-based admission rules.

Clearing checks: exact output/state and offset/selector tests across C1, C2/K3,
C3/C4 and wider supported groups, including shape misses and strict profiles;
public generation must attribute the selected rows and fallback reason. Compare
matched full-request category/heldout performance when claiming a benefit.
No flag, arithmetic selection or model admission changes in this triage.

### R0197 — Give hot-vocabulary selection an explicit owner-bound interface

The dense GGUF NextN head still accepts `HIPENGINE_GGUF_MTP_HOT_VOCAB` choices,
including an explicit map path. `Qwen35GGUFGenerator._mtp_hot_vocab_path()`
returns the generator's capacity-resolved choice when available. Automatic
capacity sizing can release the optional compact head to retain requested
context; explicit choices must not be silently downgraded. This path is distinct
from Flash-Next's request-local subset head.

**Deferred interface decision:** replace ambient path selection with an explicit
model/generator option before retiring the environment transport. Preserve
packaged automatic selection, off/full-head control and explicit validated maps;
reducing the interface to on/off alone would remove a live capability. Keep map
integrity and tokenizer/token-ID compatibility checks separate from model
admission. The target remains authoritative and full-head fallback remains.

Clearing checks: automatic/off/explicit choices, invalid map failures, per-owner
isolation, capacity-driven automatic release without changing explicit intent,
full-token-ID remapping for singleton and physical batches, and public MTP
selection/teardown. Preserve strict-profile arithmetic behavior and distinguish
proposal changes from target correctness. No selector or runtime behavior changes
in this triage; the interface migration remains unimplemented.

### R0006 — Remaining Flash-Next batch capacity and scheduling decisions

The AR batch rollback selector, per-step metadata consolidation and WMMA tile
count readback cleanup are complete. Mixed AR/MTP target execution is owned by
R0010. Historical width-one timing is recorded in
`worklog/entries/20261019T132824.038407Z-lhl-qwen4exp-batch-decode-host-overhead-bce862.md`;
further tuning must profile current PLE/hash and launch work, not repeat the
completed upload changes.

**Deferred capacity design:** `QWEN4_EXP_BATCH_SLOT_POSITIONS` still caps batch
slot extent at 65,536 positions. Page per-position index storage through the
shared block table before removing that cap; preserve the c1 longer-request
path until the replacement storage supports it. Check below/at/above the cap,
resource refusal, reuse and public route selection.

**Deferred scheduling policy:** choose a measured layer-time/chunk policy that
bounds active-decode prefill interference without operator tuning. The live
`HIPENGINE_DECODE_PREFILL_TOKEN_BUDGET` control remains distinct from idle
prefill and the ordinary `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET` override.
Before retiring it, run `scripts/qwen38fn_prefill_interference.py` through
`hipengine serve` over short, medium and long prompts with zero, one and multiple
decoding peers; satisfy the campaign's frozen latency and prompt-cost criteria.
Historical interference measurements are not fresh results for current code.

**Deferred arithmetic selector ownership:** replace
`HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN` with profile-owned registry
selection preserving production WMMA and strict grouped/GEMV comparisons.
Keep map rebuild and capacity checks, run
`tests/test_unit_qwen4exp_q8_wmma_down_tile_map.py`, and verify public selected
routes. No capacity limit, scheduler policy or arithmetic choice changes here.

### R0007 — Batch import-copy and admission ownership after reset repair

Attention, index, GDN and PLE reset through the owner-local `reset_slot`
protocol; range preflight and index-generation publication were repaired under
`worklog/entries/20261027T095542.477175Z-cleanup-refactor-r0007-reset-622379.md`.
Do not reopen that completed reset extraction.

**Deferred copy design:** replace `import_prefilled` host per-block copies with
an explicit source/destination block-table copy operation. Preserve every
per-block prefix bound: copying an unwritten source tail into a new request is
incorrect. Preserve QSA key/value/index, GDN state and PLE history ownership.
Import happens once per admission, not per decode tick; profile that boundary
before claiming a benefit from a new kernel.

**Deferred admission design:** unify the runner's `_owners`/`_positions` mirrors,
owner `prepare` row order and `Qwen4ExpARBatchOwner._leases` under one admission
object. Preserve physical-slot identity, cancellation/reuse, preflight failure
and poisoned-step cleanup; publishing partial admission is not acceptable.

Clearing gates remain `tests/test_unit_qwen4exp_batch_runner.py`,
`tests/test_unit_qwen4exp_ar_batch_decode.py`,
`tests/test_unit_qwen4exp_ar_batch_capacity.py` and
`tests/test_gpu_qwen4exp_ar_batch_decode.py`. Verify dispatch shape, copy prefix
layout, reset, capacity/isolation, MTP exclusion, two imports/two-row steps,
per-row forced-token identity and c1-chain agreement. Exercise public serving
and teardown, and keep repeatability distinct from batch-composition invariance.
Neither remaining design is implemented by this triage.

### R0010 — Flash-Next physical MTP ownership and shared-target regression

Committed-token streaming and request-owned sampling, stop, grammar, tool and
thinking control tests are implemented. Keep stream/terminal token/text parity,
slow-consumer cancellation/recovery and the generic buffered-provider contract.
Evidence remains in `tests/test_integration_server_controlled_speculation.py`,
`tests/test_unit_qwen4exp_mtp_submission.py` and the October 17 staged-serving
and HTTP-control worklog entries; this is not a fresh performance result.

**Deferred ownership design:** replace single-cycle exclusive target/draft
ownership only with a snapshotable multi-request frontier that can restore
surviving peers independently. Logical concurrency and candidate rows are not
physical request width. Resident-session reuse separately requires explicit
session ownership and rollback; explicit unsupported requests must fail before
work, not silently switch to K0. Verify cancellation, isolation and recovery
through public requests before changing either contract.

**Observed regression, deferred optimization:**
`HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET` remains opt-in because the measured
gfx1151 eager shared pass was slower than separate execution. Python layer
submission, per-row/depth GDN copying and loss of AR graph replay are the
recorded costs, not proof that shared execution is unimplemented. Historical
measurements remain in this ledger at `06d6b33dd`. Clearing command:
`scripts/qwen38fn_mixed_economics.py --widths 1,2,4,8 --arms ar,mixed` against
`hipengine serve` with the variable set and unset. Retire the rollback and enable
the shared default once it meets or beats separate execution at every width,
with control/ownership gates intact. No runtime default changes in this triage.

### R0083 — VibeVoice prefill orchestration and device-tail ownership

The uncalled raw attention entry was removed and prefill scratch already uses
reusable storage. Those repairs remain closed; span attention and its shared
kernel remain live.

**Deferred orchestration design:** unify `q4_prefill` in
`hipengine/kernels/hip_gfx1100/vibevoice/q4.py` and `_prefill_batched` in
`hipengine/runtime/vibevoice_qwen2.py` through explicit GEMM, activation-staging
and scratch dependencies. Preserve norm, RoPE, KV-write, attention, SiLU and
residual order and dtype conversions; do not combine a mechanical extraction
with a changed-arithmetic promotion.

**Deferred capability work:** the Q4 batched route wires Q4_K/Q6_K and rejects
Q5_K before decoding it with the wrong layout. Implement and wire compatible
Q5_K prefill before claiming that route supports it; preserve explicit errors
until then. This is an implementation gap, not a model-identity restriction.

**Deferred profile ownership:** retain strict chains while moving
`prefill_variant=hipblaslt` and `frontend_variant=wmma` selection into profile
construction. Promotion requires the model-specific numerical/task gate;
standalone evaluation must remain runnable before promotion.

**Deferred device-tail storage:** replace host snapshots in
`hipengine/runtime/vibevoice_encoder.py` only while preserving whole/chunk,
60-second transition, recording-reset and post-join-noise behavior. Verify
buffer ownership, failed-call cleanup and public transcription with the selected
route before retiring host storage. No kernels, defaults or storage change in
this triage; historical raw-block speed estimates are not renewed.

### R0183 — Decide wide-Q6 shared-weight shape policy

The process-wide `HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4` fallback was removed;
`target_verifier_wide_q6_shared4_session()` now uses nested, restorable diagnostic
scope. That repair is complete, not a promotion. The scope exposes Q4
shared-b2w2 and standard/planar-Q6 shared-weight candidates at physical
R20/R24/R32; chunk/direct strict owners remain available.

**Deferred candidate decision:** complete the C6/C8 strict-teacher,
determinism/isolation/task, full-suite performance and profiler gates before
promoting or retiring the shape policy. Preserve the recorded no-promotion
results in
`benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q6-shared4-candidate.json`
and `benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q4-existing-candidate.json`;
the separately regressing Q5 addition is not included by this reconciliation.
Candidate scope remains explicit; current automatic admission belongs to model
capability declarations, not the old W1 automatic-K0 description.

Preserve the distinct two-wave/two-tile gfx1151 `shared4` and
four-wave/three-tile gfx1100 `shared4_gfx1100` symbols and ABIs. A future promotion
must leave one immutable profile/capability selection and registered strict
fallbacks; a rejection must remove only unconsumed candidate selection. Verify
nested disabled scopes and exception restoration as well as public selected
routes. No candidate, backend geometry or runtime default changes here.

### R0188 — SPECDEC2 whole-request compatibility seam

`SubmitPollTextGenerator.submit_speculative_many_detailed()` selects staged
submission for staged-capable providers. Other providers still use synchronous
model-owned generation under the submission/model-loop lock and synthetic
`VERIFY_CHAIN` metadata. Staged-first routing is verified; compatibility is not
an unused route.

**Deferred provider migration:** inventory and migrate the remaining non-staged
providers before removing synchronous selection, post-hoc work metadata and
duplicate request/result ownership. RF-10's direct-control migration refers to
this requirement, not a separate scheduler implementation.

Preserve the [`SPECDEC2.md`](reference/SPECDEC2.md) S3 direct-c1 ID/cycle and exact
hidden/Conv/GDN/KV/cursor/following-AR gates, and S4 physical-c2/c4,
late-admission/refill, streaming, cancellation and ownership gates. Keep named
direct controls and registered strict/eager/serial kernel fallbacks. Verify
staged providers never enter whole-request compatibility and unsupported modes
fail explicitly. Exercise migrated providers through public generation before
retiring their adapter; no provider support or runtime behavior changes here.

### R0190 — Retire prompt replay after provider lifecycle migration

The `HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT` forced-K0 override is removed.
Streaming-capable requests feed target chunks into NextN; adapter capability
and prefix-restoration checks remain. The adapter still owns
`_prompt_hidden_rows` and catch-up replay for non-streaming paths. Do not delete
these buffers merely because the normal fresh-request path streams.

**Deferred provider/lifecycle migration:** verify streamed prefix restore and
copy-on-write, every supported incremental-prefill/context transition, exact
provider fingerprints, cancellation/failure cleanup and complete same-suite
non-regression. Then remove `_prompt_hidden_rows` and
`_catch_up_provider{,_batch}` from production selection, preserving a focused
oracle where useful. Inventory all replay consumers before removing storage.

The clearing gate must exercise `hipengine serve` with restored prefixes,
incremental prefill, cancellation and provider cleanup and confirm streaming
actually ran. Unit adapter routing alone does not close it. Keep explicit
capability failures and strict replay semantics until consumers migrate. No
runtime path or fallback is removed by this triage.

### R0191 — Move live MoE pair-reuse selection into profile ownership

`moe_physical_c2_pairreuse_session(...)` is enabled by the multi-slot MoE
factory and used by the physical adapter. It selects duplicate-expert Q4 gate/up
for x_rows 2-6, selected rows `x_rows*8`, K2048/N512; C1, C8, non-MoE and Q5
down retain their existing owners. It is not an unused diagnostic switch.

**Deferred profile design:** fold this selection into the MoE physical-target
manifest without losing nested disabled-scope restoration or geometry checks.
Preserve row-bound/interior, selected-row mismatch and incompatible-geometry
tests. Verify public exact state, acceptance, cancellation/reuse and selected
kernel attribution before retiring the context. The original final three-repeat
performance gate still governs a claimed promotion or rejection; no such result
is supplied here. A losing candidate may be removed only after checking live
consumers and preserving strict fallbacks. No arithmetic or default changes.
