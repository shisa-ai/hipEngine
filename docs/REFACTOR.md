---
status: current
owns: Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven.
---
# hipEngine Refactor / Dead-Path Ledger

## PARO one-candidate verifier graph preference

The shared planner currently selects eager verification for ordinary PARO MTP
requests. The adapter honors that choice while accepting explicit graph plans.
The concrete cause is a repeated slowdown on Radeon 8060S (`gfx1151`), packed
Qwen3.6-35B-A3B ParoQuant W4, BF16 KV, one request/candidate and 128 outputs:
graph decode is 3.23% slower than eager over the full natural suite, confirmed
at 3.06%. [Protocol and results](../benchmarks/results/2026-10-24-gfx1151-paro-graphs/NOTES.md)
include all controls, route checks and separate synchronized attribution.

Remove this eager preference when a graph implementation passes the public
state/recovery oracle and the full paired comparison plus confirmation shows
it is faster without output or acceptance drift. The exact clearing commands
are the reproduction block in that result note: `reproduce_graph_candidate.py`
with three and two repetitions, `validate_graph.py`, and
`tests/test_live_paro_mtp_graph.py`. Keep the eager runtime implementation for
strict comparison; do not add an environment flag or refuse MTP requests.

## Router select/cast rollback

`HIPENGINE_QWEN4_EXP_ROUTER_SELECT_CAST` defaults to `1`. The fused kernel keeps
selection, routing and BF16 cast bits equal to the unfused pair and removes one
launch per expert layer. `0` keeps the old chain available for fault isolation.

Remove this environment branch when the unfused pair is selectable through the
same registered composite interface as the fused operation, so strict comparison
does not require a process-global switch. Preserve the unfused primitive chain.
Validate that replacement with
`uv run pytest tests/test_unit_qwen4exp_router_select_cast.py tests/test_gpu_qwen4exp_router_select_cast_fusion.py`
and a public MTP request that confirms the selected route. The current direct
wrapper dispatch predates this default change; this entry tracks its replacement.

## DMS extent compaction: replace the memmove with a ring head

The DMS append/decode kernel compacts the live extent on every decode step:
each expired row moves every later slot left by one. At an 8K window that is a
3.4 ms append per layer, which is what bounds the route's decode rate, and it is
why the 2026-10-18 compaction-race fix had to stage 16-slot tiles through shared
memory to stay correct.

Expiry always removes the oldest row, which sits at the lowest slot, so the
extent can advance `base_offsets` instead of moving data and only compact when
the window reaches the end of the reserved range. That removes the per-step move
entirely and the staging with it. The change touches the append kernel, the
attention readers that walk `[base, base+live)`, and the store's snapshot
bookkeeping, so it needs its own correctness gate against the host oracle in
`tests/test_gpu_dms_append_decode_hip.py`.

## Cached-build tests key off a mutable /tmp compiler-version file

`tests/test_gpu_qwen38_int8_batch_attention_gpu.py` and the other
`require_cached=True` tests take the build-cache key from
`/tmp/hipengine-hipcc-version.txt`, and skip only when that file is *absent*.
Any change to its contents silently rekeys the cache: truncating it to
`hipcc --version | head -1` instead of the documented full output produced four
`FileNotFoundError: cached build artifact missing` failures that look like
kernel defects but are pure environment drift. Either record the expected
contents alongside the artifact and fail loudly on a mismatch, or have these
tests build when the entry is missing instead of requiring a prebuild.

## Flash-Next shared decode batch: rollback lever and remaining gaps

The full-model AR batch is the default serving path for Flash-Next
(`Qwen4ExpARBatchOwner` in `hipengine/generation/qwen4_exp_ar_batch.py`). Plain
AR requests are prefilled on one of two c1 staging runners and imported into a
batch slot; the slots share one paged K/V and index-key block pool sized from
free device memory. In-process step time on gfx1151 after the graph-segment and
upload work: width 1 48.9 ms (c1 runner 45.6 ms), width 2 62.6 ms, width 4
71.4 ms, width 8 88.5 ms
(`worklog/entries/20261019T132824.038407Z-lhl-qwen4exp-batch-decode-host-overhead-bce862.md`).

- **`HIPENGINE_QWEN4_EXP_AR_BATCH=0` is a rollback lever.** It restores one c1
  runner per request. Remove it, and the c1 per-request decode branch of
  `decode_batch` for plain rows, once a release has served the batch path
  without a rollback; the c1 step stays for MTP rows and over-long requests.
- **A lone request pays about 7% against the c1 runner** (48.9 vs 45.6 ms). The
  remaining host work is the PLE n-gram hash and staging (about 2.4 ms before
  the first layer) and roughly 38 small per-step uploads (positions, contexts,
  index query positions for each QSA layer). Pack the per-step metadata into
  one upload, or graph-capture the head and PLE stages, then re-time width 1.
- **A batch slot holds at most 65,536 positions**
  (`QWEN4_EXP_BATCH_SLOT_POSITIONS`). Each slot keeps per-position index slabs
  (pooled keys, member map, scores) outside the shared pool, about 100 MiB per
  slot at that cap and 400 MiB at 262K, so a longer request decodes on its c1
  staging runner instead. Page the pooled index keys through the same block
  table to remove the cap.
- **A speculative row cannot share a batch step.** MTP rows verify a candidate
  chain on a request-owned c1 runner. Automatic MTP therefore applies only to a
  request running alone: the plugin declares greedy and processed-argmax MTP
  automatic-eligible at one row, an automatic request that arrives while others
  are registered decodes plainly, and an automatic MTP row is demoted to the
  batch when another request registers. Let MTP verify rows and AR decode rows
  share one physical step to remove the demotion. Sampled requests are not
  selected automatically: their acceptance and economics have not been
  measured.
- **Decoding requests slow down while another request prefills a long
  prompt.** The token-budget round runs one full prefill chunk, then one decode
  step, so each decode token waits behind a chunk: two short requests beside a
  122,076-token prompt finished their 18-22 tokens in 31-38 s (they waited
  about 170 s for the whole prefill under `fair`). Bound the prefill share of a
  round by time instead of by one chunk, or split the chunk while rows are
  decoding, then re-measure inter-token latency beside a long prefill.
- **Decode-active prefill budget override.**
  `HIPENGINE_DECODE_PREFILL_TOKEN_BUDGET` lets operators change the model's
  active-decode token bound independently of its idle prefill budget. It defaults
  to the production policy rather than disabling time sharing. Remove the extra
  control once the model's measured layer-time policy chooses a token bound
  without operator tuning; preserve `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET` as
  the ordinary explicit override. Clearing check: run
  `scripts/qwen38fn_prefill_interference.py` through `hipengine serve` over
  short, medium and long prompts with zero, one and multiple decoding peers,
  and meet the campaign's frozen latency and prompt-cost criteria without this
  override.
- **The Q8_0 expert-down layers read a tile count back to the host at two or
  more rows** (`HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN`), so that MoE
  segment is not graph-capturable and runs eagerly in every batch step. Derive
  the count on device to make it capturable.

## Flash-Next batch decode runner: hand-off copies and owner reset duplication

- `import_prefilled` copies a prefilled c1 request into a slot with a host loop
  of three `hipMemcpy` calls per paged block per QSA layer (key, value, index
  keys), plus one copy per GDN layer and one for the PLE conv history. Collapse
  it into one device-side copy once a paged-copy kernel takes a source and a
  destination block table, so the per-block prefix arithmetic lives in one
  kernel instead of Python. Do not remove the per-block prefix limit: a
  whole-block copy would move the source's unwritten tail into the new request.
  The serving path imports once per request at the end of prefill, so the cost
  is per admission, not per tick.
- `_zero_slot` hand-rolls a memset per owner (KV blocks, index blocks, GDN
  conv/recurrent, PLE history) and then calls the index owner's `reset_slot`.
  It should become one `reset_slot` protocol on the shared owner base that the
  four owners (attention, index, GDN, PLE) implement, matching the owner-base
  consolidation already recorded for the mixer lineage. The index owner's
  generation bump is part of that protocol, not an extra step.
- The batch runner keeps its own `_owners`/`_positions` host mirrors next to the
  owners' `prepare` active sets. Now that the batch runner is reachable from
  serving, one admission object should own the request-to-slot map and the
  position mirror for every stage, so a step cannot prepare two owners in
  disagreeing row order. The serving-side map
  (`Qwen4ExpARBatchOwner._leases`) is the first half of that object; it should
  absorb the runner's mirrors rather than sit beside them.
- Regression gates: `tests/test_unit_qwen4exp_batch_runner.py` (dispatch shape,
  admission, import preflight, per-block copy layout, owner reset),
  `tests/test_unit_qwen4exp_ar_batch_decode.py` (generator wiring: one N-row
  call per tick, physical-slot pinning, compaction no-op, cancel/reuse,
  admission rollback, poisoned-step cleanup, MTP exclusion),
  `tests/test_unit_qwen4exp_ar_batch_capacity.py` (default capacity, kernel
  row bound, staging runners, rollback lever, per-instance isolation,
  unchanged MTP bound) and
  `tests/test_gpu_qwen4exp_ar_batch_decode.py` (synthetic-model serving run:
  two imports, two 2-row steps, per-row forced-token identity, c1 chain
  agreement).

## Flash-Next indexed decode admission duplication

- The new GDN, F32 Conv and PLE primitives each validate bounded, unique state-slot
  indices and repeat a device-side bitmap guard. They are not yet selected by
  serving; the registered single-row owners remain fallbacks.
- Consolidate the host slot-vector admission helper when wiring the common
  batch owner, preserving diagnosable errors, bounds and no partial mutations.
  Consider a shared device helper only if it preserves each kernel's barrier
  placement and measured dispatch behavior. Do not remove ownership guards.
- Regression gates: `tests/test_unit_qwen4exp_{gdn,conv,ple}_indexed_decode.py`
  and `tests/test_gpu_qwen4exp_{gdn,conv,ple}_indexed_decode.py`.

## Flash-Next staged MTP physical width and controls

- `Qwen4ExpServingMTP` uses exclusive target/draft pool owners and reversible
  device snapshots for one scheduler-selected cycle at a time. The Python
  iterator cannot restore a surviving peer's internal state, so physical
  speculative width is one; the existing scheduler partitions logical C2 work.
  Replace this limit only with a snapshotable multi-request frontier and
  cancellation/isolation evidence. Do not label candidate rows as request width.
- Sampling, stops, grammar, tool and thinking processors now have request-owned
  MTP commit/rollback tests and provider-scoped HTTP admission. Resident session
  reuse is still rejected before work; remove that guard only with a session
  ownership/rollback gate, not by silently using K0.
- Public streaming is declared `buffered_public`. EngineService withholds partial
  scheduler text for this declaration and publishes the authoritative terminal
  output. Replace the Flash-Next declaration only after every committed MTP token
  has a stop-safe, Unicode-safe delta and cancellation/backpressure tests pass.
  Keep the generic buffered-provider service contract for providers that need it.
- Control HTTP evidence: `tests/test_integration_server_controlled_speculation.py`
  and `worklog/entries/20261017T220002.030487Z-lhl-mtp-http-controls-e305ba.md`.
  Live qualification remains PS6; incremental streaming remains PS7.
- Evidence: `tests/test_unit_qwen4exp_mtp_submission.py` and
  `worklog/entries/20261017T192055.400855Z-lhl-qwen4exp-staged-mtp-serving-0ed770.md`.
- **Shared MTP/AR target pass is opt-in**
  (`HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET`, default `0`). Cause: on gfx1151
  the eager shared pass is slower than separate execution. Measured
  same-session HTTP economics, 64 tokens per request: one explicit MTP request
  30.3 -> 21.0 tok/s; one MTP request beside seven AR peers 48.9 -> 44.0 tok/s.
  The shared pass dispatches every layer from Python, copies GDN state per row
  per depth wave, and moves AR peers off the graph-replayed batch step.
  Clearing command: `scripts/qwen38fn_mixed_economics.py --widths 1,2,4,8
  --arms ar,mixed` against `hipengine serve` with the variable set and unset;
  flip the default (and delete the variable) once the shared arm meets or beats
  the separate arm at every width.

## Generic speculative service lifecycle

- `EngineService` admits the generic spelling through existing speculative
  child submission only when the driver declares both generic and model-attached
  MTP support. Generic-only whole-request providers (Laguna DFlash) fail closed;
  they no longer execute on a frontend thread. This is a serving restriction,
  not completion of generic-provider lifecycle support.
- Replace the MTP compatibility gate with explicit generic child admission once
  generic-only providers expose bounded request-owned work. Do not restore
  frontend execution or put whole generation inside an administrative command.
  Keep generic and model-attached provider semantics separate in that SPI.
- Evidence: `tests/test_integration_engine_service_speculative_ownership.py` and
  `worklog/entries/20261017T190456.513027Z-lhl-speculative-service-ownership-3d661c.md`.

## Dense Q8_0 checked-int32/full-four-row DPP coltile screen

- `coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out`
  is a separately registered, default-off diagnostic sibling of the dense Q8_0
  DPP prefill owner. Its wrapper admits only complete four-row groups whose
  weight, activation, output and grid products fit signed int32; tails,
  overflows and unsupported shapes keep the generic DPP parent. No production
  selector points here.
- The sibling narrows the pipelined leaf's weight/activation/output offsets from
  int64 to int32 and removes the per-row validity control for complete groups. It
  preserves the Q8 scale decode, ascending per-lane K FMA order, DPP reduction,
  four-wave publication, F32 output, tile shape and grid order, so every admitted
  shape is bit-identical to the DPP parent.
- Remove the variant, wrapper and extern symbol once a different mechanism or a
  production decision supersedes the screen; the generic DPP parent needs no
  cleanup.
- Evidence:
  `benchmarks/results/2026-10-07-qwen4exp-dense-dpp-checked-full-rows/`.


## K320/full-row GR-up DPP screen

- `coltile2_branch4_rowbatch4_wave_scale_dpp_k320_full_rows_f32_exact` is a
  separately registered, default-off sibling of the promoted ordered-DPP GR-up
  owner. Its wrapper accepts only K320/H2560/four branches and complete
  rowbatch-4 groups; runtime selection requires the promoted DPP key first and
  otherwise retains that key. The generic DPP owner remains the fallback for
  tails, other K/H shapes and absent registration.
- The candidate preserves ascending FMA order, DPP reduction, four-wave
  publication, `expf`, branch mean, output layout and both materialized outputs.
  It removes only compile-time-provable generic row-validity and variable-K
  control. **Promoted 2026-10-08** after the full qualification chain
  (natural-input owner, full model-state gate, complete-engine A/B — see
  `benchmarks/results/2026-10-08-qwen4exp-gr-k320-natural-owner/`): the
  production binder binds it on for the qualified lane (`gguf_ud_q4_k_xl`,
  production) and `HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS=0` is the dedicated
  bisection opt-out; the generic DPP owner remains the registered strict
  fallback. Removal condition for the opt-out flag: delete the flag (and its
  profile binding + the selector's env read) once the K320 sibling has held the
  production default through a full campaign cycle without a rollback, or when
  the generic DPP leaf it narrows is retired.
- Unit/profile tests pass33 and the HIP GPU bundle passes10/10 on gfx1151,
  including exact parent parity at rows512/1024/4096. The balanced complete
  operation screen covers 12 real-weight cases and20 pairs/case: 1.084–1.103x
  operation speedup, geometric mean1.092x, exact outputs. Evidence:
  `benchmarks/results/2026-10-06-qwen4exp-gr-k320-full-rows-screen.md`.


## Qwen4Exp MT-8 rollback/replay accounting

- The CPU-only `scripts/qwen4exp_mtp_rollback_accounting.py` screen covers a
  four-token serial Qwen4Exp MTP proposal at every rejection depth, all-reject
  and full-accept. Target forwards are `1/2/3/4/4`; draft forwards are `4` in
  every case. Verification stops at the first mismatch and draft rollback is a
  cursor trim; restored bytes, restore allocations and per-cycle allocations are
  zero. Shared resident weights are counted once, not once per case.
- This is diagnostic-only. No duplicate target replay or dirty-destination
  full-state restore exists in the current route, so no production change is
  justified. Reopen only with a transaction redesign that introduces distinct
  state/KV/recurrent ownership or a real dirty-destination restore path.
  Evidence: `benchmarks/results/2026-10-07-qwen4exp-mt8-rollback-replay-accounting/`.


## Bounded HC grouped-norm F16 publication and rollback

- `f32_f16_publish` and `f32_f16_publish_unfused` are separately registered
  grouped-norm leaves. The first publishes existing F32 norm and identical
  staging F16 bytes; the second is parent norm plus the existing cast.
  `scripts/qwen4exp_norm_f16_publish_owner.py` replays both operations in
  private scratch; it is not a native ownership gate.
- `HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH` is enabled by the named gfx1151
  UD-Q4_K_XL production binder and disabled by strict and Q4_K_M. Explicit
  pre-registration `=0` rollback is recorded in the manifest. Backend
  registry-axis policy admits only the immediate enabled Q8_0
  dense-wide HC-down consumer at exactly 4096 rows, K10240/N320 and layers
  13–47. It borrows the existing session-owned workspace, adds no device
  allocation, and preserves parent norm plus ordinary cast on admission misses.
  The single-use loan binds source/count/stream/runtime/workspace and consumer
  key; nested scopes, mismatches and intervening staging invalidate it.
- Native model engagement, state/KV/logits and counterbalanced engine A/B are
  complete: 14 native canonical/heldout cases are exact with 70 publisher/
  consumer calls per candidate 4096 prefill, zero decode calls and zero
  teardown allocations. The tracked-clean 12-case A/B records a 1.0015x
  weighted 4096 prefill ratio, while aggregate request wall is 0.9970x and
  general-English p4096 is 2.77% slower. This is therefore a bounded prefill
  owner transfer, not an all-request or decode-speed claim. Shapes 512/1024
  remain on parent norm+cast. No persistent activation cache is planned.
- Remove the integration flag after one qualified release window, or remove
  the runtime handoff if later native pricing rejects it. Keep the registered
  parent and strict unfused chain. Evidence:
  `benchmarks/results/2026-10-06-qwen4exp-norm-f16-publish/`.

## Checked Q4 gate/up index diagnostic

- `selected_dual_wmma_iu8_risk_narrow_prefill_bf16_bf16_out` is a
  registered diagnostic sibling, not a production binding. It preserves the
  three-plane16-row arithmetic and int64 host/map ABI while checking complete
  activation/output/weight/map/risk byte ranges before narrowing device
  address/control intermediates. The wide parent and strict grouped chain
  remain registered.
- `scripts/qwen4exp_q4_narrow_owner.py` uses temporary registry substitution
  and private replay scratch to compare the complete natural packing,
  tile-map, risk, repair and SiLU operation. It does not add a runtime flag.
  The sibling remains a diagnostic bounds/resource leaf after this screen;
  remove it when a replacement bounds/reference path makes that coverage
  independently unnecessary. No production selector points to it.

## Padded-fragment dense-wide production tail and rollback

- The F16-input `dense_wide256_padded_f16in_f32_f32_out` leaf is selected
  only by gfx1151 UD-Q4_K_XL production for K10240/N320, rows512–4096,
  within the existing dense-wide layers13–47 scope. Backend registry-axis
  metadata bounds the launch-time rewrite after successful F16 staging;
  missing registration retains the F16 parent. Strict retains raw-F32 coltile.
  F32-input padded siblings and other shapes remain diagnostic.
- `HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT` is a temporary named-plan
  rollback: select `dense_wide256_f16in_f32_f32_out` before plan registration
  to create a distinct parent manifest. Binder output always agrees with the
  manifest; a diagnostic override must be post-binder. Remove this env reader
  after the release/bisection window and encode the qualified choice directly,
  keeping parent/strict registry leaves and shape/ownership tests.
- The full20-case state/KV/repeat gate and tracked-clean12-case three-pair
  engine A/B pass. Natural complete-GEMM gain1.312–1.361x, scratch92→0B,
  spills58→0; prefill+1.036%/+0.594%/+0.445%, total request wall−0.162%.
  Two request cases lose0.214%/0.279%, so no all-case request/decode-speed
  claim. Diagnostic substitution plumbing remains useful for explicit
  parent/candidate gates; remove it only after a generic named-route gate
  covers the same bounds and engagement assertions.
  Evidence: `benchmarks/results/2026-10-06-qwen4exp-dense-wide-padded-gate/`.

## Dense split-operand WMMA runtime cleanup — complete

- The failed `HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS` runtime selector,
  cache-key fields, profile-binder entry and full-model gate modes are removed.
  The obsolete environment variable has no runtime reader. Admitted raw-F32,
  late dense-wide and staged-F32 routes are unchanged.
- The independent `dense_split64x128_f32_f32_out` diagnostic leaf remains in
  `quant/gguf_q8_0_dense_wide.{hip,py}`, with its GPU numerical RED fixtures,
  CPU/F64 operation screen and private-buffer natural-input collector. It has
  no generator/runtime binding. The low-level registry key is for diagnostics,
  not production admission.
- Compensated F32 high-high accumulation, positive/negative WMMA averaging and
  cross-plane summation pass projection fixtures but fail the complete
  canonical production envelope: mean KL 1.4386e-3, p95 7.5336e-3, max
  0.059744, top-1 1526/1548. Projection error rankings do not prove model
  causation or justify another unchanged qualification run. Reopen only with
  a new arithmetic or ownership mechanism and a predeclared screen.
- Out-of-range operands still trigger the diagnostic leaf's in-kernel strict
  F32 fallback. Preserve immutable results at
  `benchmarks/results/2026-10-05-qwen4exp-dense-split-wmma/` and
  `benchmarks/results/2026-10-05-qwen4exp-dense-split-model/`. Remove the
  diagnostic leaf only after its independently useful oracle coverage moves
  to a replacement; never remove committed failure evidence.


## Prefill chunk capacity guard

- `_select_prefill_chunk` / `_AUTO_PREFILL_CHUNK_LADDER` in
  `hipengine/generation/qwen4_exp_gguf.py` select the production prefill chunk
  by admission rather than by a constant. Added 2026-10-04.
- **Revert lever:** the ladder is a single tuple. Dropping it to `(1024,)` or
  `(2048,)` restores a fixed chunk with one edit and no other change; 2048 is
  the free rung (it sits under the 4 GiB per-runner scratch floor and costs no
  device memory), so `(2048,)` is the conservative fallback if the 4096
  reservation's +6.80 GB ever binds.
- **Remove when:** the guard has a dedicated GPU test asserting the ladder under
  constrained `available_device_bytes`, and the `RT-3` gaps it does not cover
  (2051/2052 boundary, ragged final chunks, separate repair-queue / F16-workspace
  accounting) are either measured or formally dropped. Until then the ladder
  values are the only place the policy is expressed and must stay in one place.
- Evidence: `benchmarks/results/2026-10-04-qwen4exp-prefill-chunk-4096/`,
  worklog `20261004T145428.690726Z`.

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
> selectors are `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP` and `_DOWN`; the
> bare `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP` gate/up selector stays deleted.
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

## Split prefill diverges from a single call (found 2026-09-19)

- Prefilling a prompt in two `prefill_batch_native` calls does not always equal
  prefilling it in one. Two shapes diverge, measured on the Qwen3.5-0.8B
  fixture and reproduced identically on Qwen3.6-35B-A3B across all four
  mtp-bench categories:
  * `rows == 1` - a one-token chunk selects decode-shaped kernels
    (`dense_down_decode_fused`, the rows==1 projection routes), whose
    arithmetic differs from the bulk prefill path. **Closed for the prefix
    cache on 2026-09-19**: a one-token reused suffix now takes the serial step
    (`_gguf_prefix_batched_suffix_chunk_eligible`), so a hit never emits a
    one-row prefill, and prefill callers pass `force_bulk_rows` to
    `_run_post_attention_ffn_rows` to keep a one-row prefill off the dense GEMV
    decode variants. MoE cannot be redirected the same way -
    `_run_post_attention_moe_rows` refuses `rows <= 1` - so the raw
    `prefill_batch_native` API still exposes the shape and the pin covers it.
  * total context above 2048 - splitting anywhere diverges from one call.
    **Root cause located 2026-09-19**: per-layer capture shows the first
    divergent layer is the first `full_attention` layer (layer 3 on
    Qwen3.5-0.8B, max|d| 1.953e-03, BF16-ULP scale) while every preceding
    `linear_attention` layer is bit-identical, so the effect is in full
    attention, not GDN. Splits at different boundaries agree with each other
    and only the single call differs, which places it in query-window handling
    once one call carries more than 2048 query rows. Capping the query window
    would make misses match hits but measured *worse* against a serial
    reference (kl_mean 2.37e-2 single call versus 1.59e-1 split), so that trade
    was rejected. The
    threshold is absolute, not relative to `max_sequence_length` (verified with
    the session capacity held at 4096), and is independent of the split point:
    at total 2148 the split result is identical for boundaries 1024, 1792,
    1920 and 2048, and differs from the single call in all four.
- **This is not a prefix-cache defect.** The probe uses one session and two
  direct prefill calls - no pool sharing, no snapshot restore, no cache. Both
  families reproduce on clean HEAD (d01ec1b51) to three significant figures.
  The batched reused-suffix extend inherits them because a cache hit takes the
  split path; it does not cause them.
- Which side is right: against a serial autoregressive reference at total 2052,
  the single call is closer (kl_mean 2.37e-2, top-1 100%) than the split
  (kl_mean 1.59e-1, top-1 91.7%). So the >2048 split path is a real accuracy
  gap, not a neutral re-association.
- Pinned by `test_split_prefill_divergence_boundaries_are_unchanged` in
  `tests/test_live_qwen35_gguf_chunked_prefill.py`, which also asserts that
  total 2048 is still bit-exact. Fixing either family should break that test;
  replace it with coverage in
  `test_batched_suffix_extend_is_bit_exact_against_recompute`.
- Removal scope: the pin, once both families are fixed and every shape is
  covered by the bit-exact contract test.

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

## Placement decides the prefill route, and the routes are far apart (found 2026-09-19)

- A shared-prefix allocation whose pages are contiguous keeps the slot-local
  prefill route. A gapped one makes `_gguf_device_kv_contiguous_base_row`
  return None, `slot_local_full_prefill` false, and the suffix prefills through
  the packed paged route. Measured on Qwen3.5-0.8B with the source deliberately
  held so placement is gapped, an 8234-token prompt reusing 2048: the paged
  suffix costs **6.5 ms/token against 0.39 ms/token** for the full prefill it
  replaces, one call or twenty-five - chunking is not the driver, and stubbing
  the whole-history KV import changes it by 0.1%, so the cost is in the paged
  prefill compute itself.
- That single ratio explained the served regression. At
  `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS=16`, pinned retained pages fragment
  the arena, 8 of 18 admissions were gapped, and one lane spent **114 s**
  prefilling a 6,186-token suffix that a plain miss does in ~15 s. Phase
  attribution: `suffix_prefill` 5.8 s at the default retention against
  **125-135 s** at retained 16.
- Two changes, both measured: admission asks for a contiguous placement first
  and accepts a gapped one rather than refusing (15 of 16 hits are contiguous
  at the default retention, so most hits already took the fast route by the
  allocator's adjacency preference - this makes it explicit); and a gapped
  placement is only accepted when the suffix is at most
  `HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX` tokens, default 512. Past that the
  hit is declined into the full prefill it would have replaced. Crossover
  measured at 2048 and 4096 token boundaries: a gapped hit costs 0.63x and
  0.87x the miss at a 512-token suffix and 1.07x and 1.22x at 768.
- Result: retained 16 goes from +9.5% to **-11.0%** wall with `suffix_prefill`
  down from 134.9 s to 6.8 s; the default retention is unchanged at -18.6%.
- **Update 2026-09-19 (gapped gather route):** the cost model's premise is now
  gone where the gather route serves. A gapped BF16 slot swaps its identity
  spans for its real chunk-local block table and runs the same AOTriton
  attention over gathered head-major buffers the contiguous route uses, so a
  gapped hit pays contiguous cost at any suffix length (the 35B lane's forced
  gapped suffix prefill went 134,511 ms -> 8,306 ms against 8,226 ms
  contiguous). The suffix budget now binds only when the fast gapped route is
  unavailable - `HIPENGINE_GGUF_GAPPED_GATHER=0`, a backend without
  head-major KV, or a context beyond the validated head-major allocation
  class - where the paged-route costs above still hold. Removal scope below
  is unchanged for those populations.
- Removal scope: the threshold and the decline path, once the paged prefill
  route is no longer several times slower per token than the slot-local one
  on every backend. Gapped gfx1100 BF16 slots now resolve the registered copy
  kernel independently of the optional contiguous head-major-layout default;
  explicit rollback and allocation limits still retain the guard.
  Fixing that is the real prize - it would make every hit pay and let the
  retained working set grow.

## Shared-prefix suffix prefill runs one token at a time (found 2026-09-18, resolved 2026-09-18)

- `_prefill_native_chunk` has two routes for a chunk: a batched call through
  `prefill_batch_native`, and a serial loop that calls `session.step()` once per
  token. A row with `prefix_reused_tokens` takes the serial loop and never
  reaches the batched route. Measured cost on `zbook`/`gfx1151`: **34.0 ms per
  suffix token** (31.5-34.5 across all 16 hit turns of the prefix-cache A/B),
  against **1.8 ms/token** for the batched full prefill it replaces. That single
  route is why the measured A/B is net-negative (+6.4% wall overall, ShareGPT
  +22.7%) even though reuse resolves 16 of 27 lookups.
- **Resolved 2026-09-18:** the reused branch of both reused-suffix routes now
  prefills batched via `prefill_batch_native` (the <1024 paged-prefill policy
  gate was the only blocker; see the resolution note in "Reused-suffix prefill
  has two routes and only one is slow" above). Shared-prefix admission no longer
  asks for a contiguous suffix run: the paged route walks the block table, so
  `shared_admission_noncontiguous` cannot fire. The serial loop remains as the
  decline path for `_disable_incremental_prefill` (which fails closed for reused
  rows by design) and as the `HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX=0` rollback.
- Removal scope: the serial loop in `_prefill_native_chunk` and
  `_prefill_processed_argmax_chunk`, the `prefix_c1_suffix_prefill_chunks` /
  `prefix_c1_suffix_prefill_tokens` / `processed_argmax_prefix_c1_suffix_*`
  counters that only it feeds, and the `HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX`
  flag, once the served multi-turn A/B re-measurement with the wide retained
  set (`HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS=16`) shows the batched route
  holding exactness and the win in production shapes, and the
  `scripts/gguf_prefix_reuse_gate.py` gate passes at >=1024 total context on
  gfx1151 and gfx1100.
- The invariant stands while the loop lives: a reused row's prefix pages are
  shared with another request, so no fallback may write them.

## Long-context verifier rows are coupled to the global split-attention threshold (found 2026-09-17, resolved 2026-09-18)

- Resolved: `_run_full_attention_attn_chain_rows_exact` no longer raises
  `"staged full-attention chain currently requires non-split decode"` when the
  span-wide attention limit crosses the split threshold. Every verifier row now
  resolves its own c1 attention leaf (`_staged_full_attention_row_leaf`) — the
  split-K gate leaf with the scalar owner's own chunk size and split tiling at or
  above the threshold, and the short-context batch leaf or the plain leaf below
  it, whichever the scalar owner would pick from `position + 1` — so the batched
  projections, head norm/rotary, KV writes, and output projection stay batched
  across the boundary while ordinary decoding keeps split-K attention. Eager
  verification builds the per-row views the graph owner already gets from device
  metadata (`_eager_native_row_views`), and `_staged_full_attention_rows_ready`
  keeps the row-wise strict route for any row whose leaf does not resolve.
  Measured on the eager long-context packet: cycle wall 8.129 s -> 7.893 s median
  (-2.9%, 7 of 10 cases faster), 10/10 direct cases exact against the
  serial-exact teacher, 16/32 split-K calls per case unchanged
  (`worklog/entries/20260917T195925.422701Z-lhl-gguf-long-context-eager-staged-attention-2c4ca3.md`,
  `benchmarks/results/2026-09-18-gfx1151-qwen38-long-context-eager-verifier-staged-chain.json`).
- Resolved for the measured artifact (2026-09-19): the staged *linear*-attention
  chain is byte-exact above the split threshold on the current dense Q4_K_M
  path, so the blanket row-wise rule no longer has a correctness reason to cover
  it and the backend default admits the staged chain for that file type (see the
  row-wise strict dispatch entry above). The row-wise c1 route stays registered
  as the strict fallback for every other artifact identity and behind
  `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG=0`; it is not dead code. The recorded
  0.90x/0.68x serving ratios were measured with the row-wise rule in force
  (`benchmarks/results/2026-09-17-gfx1151-qwen38-long-context-mtp-screen.json`).
  The adapter's 1,023-token provider window that also refused serving MTP above a
  1,022-token prompt was a separate blocker from this route and has since been
  removed (see the entry above).
- Historical note: the only measured way to reach the batched long-row route
  before this was to raise the split threshold past the request's context (or
  disable it with `0`), which cost normal decoding about 9% at 3,530 prompt
  tokens. That workaround is no longer needed for full attention, and the
  linear rows now have their own default-off admission instead.

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

## VibeVoice ASR prefill and legacy dense attention

- The Q4_K_M batched prefill in `hipengine/kernels/hip_gfx1100/vibevoice/q4.py`
  (`q4_prefill`) mirrors the orchestration in
  `hipengine/runtime/vibevoice_qwen2.py::_prefill_batched` — the same norm / rope /
  KV-write / attention / silu / residual sequence with different GEMMs and
  activation dtypes. Unify them behind overridable GEMM, activation-staging
  and scratch hooks once the Q4 prefill has a production-profile numerical
  gate; the two copies currently differ only in those three places.
- The batched prefill route wires only Q4_K and Q6_K; Q5_K has no WMMA
  prefill kernel (only the naive raw-block `gguf_q5_k_prefill_*`, ~100x
  slower) and raises rather than mis-decoding. Add a Q5_K WMMA prefill before
  any Q5_K_M GGUF is put on this route.
- `prefill_variant=hipblaslt` and `frontend_variant=wmma` expose the arithmetic
  candidates to the production evaluator. Keep their registered strict chains
  as debugging oracles. Bind the public production profile only after the full
  model-specific numerical/task gate; remove standalone candidate-selection
  plumbing when profile construction owns that selection.
- Convolution tails currently use owned host snapshots at chunk boundaries.
  Replace these with device-resident tails only after the same whole/chunk,
  60-second transition, recording-reset and post-join-noise tests pass.

## Qwen4Exp tool grammar host path

- Model-owned embedded-template/XML parsing replaces the generic JSON tool
  instruction format only for Qwen4Exp. Keep other model parsers independent.
- llguidance masks currently use full F32 D2H plus NumPy argmax. A later device
  bitmask/argmax kernel may remove this cost, but must retain request isolation,
  selected-token handoff, EOS/content boundaries and the same grammar language.
- Do not revive generic JSON forced prefixes or close-repair tokens on the XML
  route. Required/named tool choice is enforced by its grammar.
- XML schema coverage is explicit; unsupported string/schema combinations fail
  rather than being silently ignored. Broader XML parameter schema forms require
  additional lexer/parser tests. Text MTP applies request-owned masks through
  the attached provider's `supports_grammar` contract; multimodal grammar
  remains rejected. See `tests/test_unit_qwen4exp_mtp_structured.py` for
  committed-token, cancellation and grammar-completion coverage.

## Qwen4Exp context capacity policy

- `native_context_length` is now consumed by public memory admission, bounded
  by artifact context metadata. Keep `qsa_dense_equivalent_max_tokens` and
  selected-position capacity independent and unchanged.
- Public admission reserves the larger of the4GiB per-runner floor and mandatory
  runner scratch plus cast staging for the requested context/chunk, with a
  separate4GiB reserve and vision payload. Optional MMQ/graph/verification
  growth still needs composed accounting before it exceeds that envelope;
  never silently over-admit.
- Device-position-owned QSA remains dense-regime probe-only; promotion requires
  native sparse selection support, not raising or bypassing the2051 guard.

## Qwen4Exp prepacked Q8 MMQ admission

- Separate prepacked registry candidate retains an exact raw-Q8 fallback.
  Production binds `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK=1`, but sidecar
  preparation also requires the prefill MMQ library. Arithmetic recovery keeps
  `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL=0`, so the shipped recovery path allocates
  no prepacked sidecars. When prefill MMQ is enabled, runner-owned sidecars for
  GDN QKV/SSM shapes are exposed only through the bounded prefill MMQ session.
  The historical enabled-path five full logits/state/KV cases and lifecycle
  pass; full12-case72-trajectory timing improves every prefill case. Strict
  binds0. Retain rollback through the next owner refresh and one release
  window; then remove redundant positive experiment selection.
- CPU reference packer lives only in the test/screen. The registered GPU
  `weight_pack/gguf_q8_0/mmq_kmajor76` API is now byte-exact and available
  for admission and used by runtime; do not import tests from runtime. Raw weights remain necessary
  for decode and risk repair, so account for the full additional sidecar.
- Keep the measured mixed512 request-wall regression0.14% and1.67GiB additional
  memory explicit for the enabled prepacked path under the prefill-first
  retention decision; that sidecar cost is not paid while prefill MMQ is off.
  GR remains excluded
  until an independent positive complete-chain/model gate.

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

## 2026-09-13 Surya decode split chunk keyed to `max_seq` — open

- `plan_surya_decode_splits` derives the decode split chunk from the runner's
  `max_seq` (`chunk = round_up(ceil(max_seq / 64), 32)`, `num_splits =
  ceil(max_seq / chunk)`) and `SuryaGpuRunner._alloc_span_state` builds one
  immutable `KVLiveSpans` view per runner, so a runner configured for a long
  context runs short requests with fewer, longer producer blocks. At a *fixed*
  live context that costs decode time, measured on gfx1151: 1024 live tokens
  decode in 18.31 / 18.36 / 18.88 / 18.96 / 19.55 ms per token at `max_seq`
  8192 / 12288 / 16384 / 20480 / 32768 (chunk 128 / 192 / 256 / 320 / 512), and
  8580 live tokens in 19.07 / 19.22 / 19.38 / 20.00 ms from 12288 up — 3% from
  8192 to 16384 and 4% more to 32768, against a per-row spread of 1.1-2.4%
  (`benchmarks/results/2026-09-13-gfx1151-surya-context-default.json`).
- Fix: derive the chunk from the live count instead of `max_seq`.
  `surya_full_attn_decode_f32_spans` already resolves the plan inside the
  wrapper from `spans.max_live_count` and already accepts a `chunk_size`
  override, so the change is a host-only
  `plan_surya_decode_splits(live_count)` call in `_attention_decode_spans` — no
  kernel change. The one allocation that must move with it is the split-K
  partial triple, sized today from the construction plan's `num_splits`: a
  live-derived plan can exceed it (at `max_seq` 5000 the construction plan is 53
  splits while a 4096-token request derives 64), so size those buffers for
  `SURYA_TARGET_SPLITS` — 512 KiB for the output partials at Surya's 8 q heads
  and head_dim 256.
- This changes decode arithmetic (different split boundaries reassociate the
  online-softmax partials), so promotion needs the production-profile numerical
  gate plus the registered rocBLAS strict fallback. Note the default path's
  decode arithmetic already depends on the configured `max_seq` for the same
  reason, so this does not introduce a new variability class — it removes one.
- Removal trigger: when the live-derived plan lands and the decode rows above
  converge across `max_seq`, delete the `max_seq`-derived chunk and this entry.

## 2026-09-08 Native C1 Context Admission

- Remove the obsolete gfx1100 dense H5120 Q4_K_M p95 native-context workaround
  after the scratch lifetime and scalarized initial-state snapshot repairs. A provisional
  geometry-scoped 256 cap was tested but is not retained: untested context
  lengths are not presumed broken.
- The backend cache-capacity policy is geometry/quant keyed and applies to BF16
  KV with FP32 recurrent state. Other routes keep their existing dispatch:
  the MoE control already fails strict full-logit equality at p8 under the old
  policy. Resolve that model-route discrepancy separately before consolidating
  the generic cap and cache-capacity policy.
- Native graph admission owns allocated capacity, supported rows, kernel-family
  transitions and split-workspace boundaries. Its private metadata producer is
  independent of the bulk-prefill device-metadata threshold.
- `strict_long_rows` in the native verifier currently scalarizes attention and
  dense FFN beyond the split-K crossover to preserve exact scalar arithmetic.
  This is an explicit execution path, not a disabled-context policy. Replace it
  only when a faster row-batched path passes its declared exact/profile gates.
- Public serving evidence still independently owns request/context/horizon
  admission. Do not equate removal of a backend workaround with new automatic
  serving keys. Keep eager/serial verification for legitimate unsupported
  graph configurations and as the correctness oracle.

## 2026-09-07 DMS codec evaluation bootstrap — open

- `create_dms_int8_evaluation_backend` permits offline model measurement without
  inventing a qualification record. Snapshots explicitly report
  `codec_evaluation_only`; default resident/serving construction stays BF16.
- `Qwen35GGUFResidentSession.dms_backend_factory` injects a codec factory without
  codec-specific dispatch branches. The quality tools expose
  `--codec int8_evaluation`, never a serving promotion switch.
- Keep the evaluation factory only while it is needed to qualify new artifacts
  or codecs. Consolidate it into a shared evaluator capability when one exists;
  never replace its explicit unqualified status with fabricated passing scores.
  Serving INT8 construction does not require a qualification record; if supplied,
  the record's artifact fingerprint must match. Evaluation-only status is observability,
  not a prerequisite for executing an explicitly requested codec.

## 2026-09-07 UD cleanup — reduce authorization surface

- Removed generic profile qualifiers, custom-factory rollback and the cancelled
  profile-application prototype. Preserve ordinary profile plugin composition
  and the shared header-identity helper; do not recreate a transaction or
  file-replacement defense around model loading. Model files are immutable
  during a session.
- Existing GGUF binders still write process-wide settings. Initial UD bring-up
  uses a fresh process without a named profile. If multiple model/profile owners
  must coexist, replace the needed settings with explicit owner fields at their
  actual consumers; do not wrap every method in implicit profile context.
- Removed per-call native physical-operand authorization and recursive scans.
  Native routes qualify at construction and use session-owned fixed allocations.
  Private pointer/geometry mutation is not a supported reconfiguration API.
  Closed-owner, row, token and context bounds remain runtime checks.
- Next: simplify reusable admission certificates. Keep loader shape/dtype checks
  and shared dispatch contracts; do not replace them with a second framework.

## 2026-09-07 GGUF native execution authorization — bounded cleanup

- The experimental `NativeInvocationContext`, index-publication receipts and
  recursive physical inventories are removed. They defended against arbitrary
  private-field mutation rather than a supported caller. Construction-time
  resident checks and shared kernel ABI descriptions stay. Do not restore
  per-layer full-model validation; add narrow invalidation at an actual public
  reconfiguration API if one is introduced. No GPU timing claim accompanies
  this host-only cleanup.
- Host/deferred embedding and optional expert-sidecar native adapters have no
  complete pre-allocation contract. Keep those route refusals until the actual
  registered/executed adapter and every resident/physical owner are represented
  in the shared call plan. Eager/c1 placement behavior is separate and unchanged.

## 2026-09-04 Qwen4Exp PF-5 GDN w32 prefill variant — retained rejected lever

- `qwen4exp_gdn_w32_prefill` (`hipengine_qwen4_exp_gdn_prefill_w32_f32`,
  a `WARPS_PER_BLOCK=32` instantiation of the templated columnwarps kernel in
  `hipengine/kernels/hip_gfx1100/linear_attn/qwen4_exp_gdn.hip`) is a measured
  loss vs the production columnwarps `<4, 4>` owner at every shape >= 64 rows
  (+1.6% to +6.8%, bit-exact parity; rows=16 within noise) and stays a
  registered non-default strict variant with its exact-parity RED tests
  (`tests/test_gpu_qwen4_exp_gdn_w32_prefill.py`). It is never selected by
  dispatch; the default remains columnwarps `<4, 4>`.
- Remove the variant registration, the `<32, 4>` instantiation, the host
  wrapper, and the test file if a later campaign wants the registry back —
  trigger: a named product decision that the w32 geometry will not be
  revisited on gfx11 (for example, at campaign closure), or if the still-open
  tile-16 lever lands in a form that makes the 32-warp geometry unreachable. Do not remove before that trigger: it is the only
  device-side reproduction of the 1024-thread occupancy loss measured in
  `benchmarks/results/2026-09-04-gfx1151-qwen38-flash-next-pf5-gdn-w32-prefill-ab.json`.

## 2026-08-31 Qwen4Exp P3 Q8 MMQ attention-gate scope

- `HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE=1` adds the omitted
  K2560/N6144 GDN attention-gate shape to the existing guarded F32 Q8 MMQ
  policy. The original state-repeat failure was fixed; the later complete
  correctness packet passed. The subsequent five-pair comparison was rejected
  for noisy performance, not state divergence. Both strict and named production
  bind this scope off. Revisit the selector only with stabilized five-pair
  evidence, or consolidate it into an operation-complete dense/GR owner; remove
  the separate selector when that owner replaces it. Exact raw-Q8
  coltile8/rowbatch4 remains fallback. See
  `worklog/entries/20260901T220425.712695Z-lhl-qwen4exp-p3-q8-mmq-attn-gate-current-blocked-4fa414.md`
  and `benchmarks/results/2026-09-02-gfx1151-qwen38-flash-next-p3-q8-mmq-attn-gate-fivepair-rejected.json`.

## 2026-08-29 Qwen4Exp late-layer production prefill

- The remaining `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL` process bridge controls
  Q8 MMQ resources; UD arithmetic recovery keeps it off. Remove its environment
  transport when resolved profile state is request-local. Superseded Q5_1/Q4_K
  MMQ selector/layer-list transport and workspace branches are removed; their
  registered kernels remain available for explicit diagnostics. Preserve exact
  coltile-Q8 and grouped Q4/Q5 fallbacks; the Q8 (320,10240) hc-down and
  K%256!=0 shapes remain strict.
- `HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL` and
  `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` were superseded for the named production
  profile by the faster admitted MMQ stack, but `PRODUCTION_MOE_PREFILL` is
  re-admitted 2026-08-29 as the certified transport for the WMMA-MoE27
  suffix and now owned by the named production profile manifest
  (`3b7a0644…`, see
  `benchmarks/results/2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json`).
  The binder keeps the env transport as the request binding surface; fold it
  into request-local profile state together with the other Qwen4Exp bridges
  and retain the registered strict-owner fallbacks. Note the binder writes
  `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` as `""` during its own pass, so a value
  set before generator construction is silently inert; gates that need the route
  must set it after the binder runs
  (`scripts/execution_profile_q8_wmma_prefill_layers_gate.py`). Give the folded
  request-local surface an explicit getter so callers stop depending on binder
  ordering.
- `HIPENGINE_QWEN4_EXP_Q4_DP4A64{,_LAYERS}` is the internal Q4 decode
  bridge. UD-Q4_K_XL production and fast bind `Q4_DP4A64=0`: the one-plane
  route caused the isolated long-context decode discrepancy, and the manifest
  already declared the strict logical128/t64 owner. Other production routes
  stay enabled. The historical layer list remains available for diagnostics.
  Evidence: `benchmarks/results/20261022-qwen38fn-q4-decode-repair.json`.
  Remove the process-global transport when profile state becomes request-local;
  keep the registered strict fallback.

  To reconsider DP4A, repair the operation and run the existing explicit
  candidate, which pins the logical128 consumer rather than allowing the warp
  override. Set `MODEL_ROOT` to the model directory and `LONG_PROMPTS` to the
  long numerical JSONL fixture including the 12,287-token failure and controls:

  ```bash
  python3 scripts/qwen4exp_layer2_profile_gate.py --model-root "$MODEL_ROOT" --candidate decode_dp4a_linear --decode-steps 32 --prefill-chunk-size 1024 --output /tmp/q4-decode-short.json
  python3 scripts/qwen4exp_long_context_gate.py --model-root "$MODEL_ROOT" --candidate decode_dp4a_linear --numeric-prompts "$LONG_PROMPTS" --max-sequence-length 20000 --prefill-chunk-size 1024 --candidate-repeats 3 --output /tmp/q4-decode-long.json
  ```

  Both numerical gates and the campaign's complete task/serving gates must pass
  before restoring this arithmetic default. A two-trajectory screen does not
  clear the failure.

## 2026-08-28 Qwen4Exp native c2 runner pool

- `Qwen4ExpResidentServingRunner` provides request-owned c2 residency and
  scheduler-native stream events while invoking one request transition at a
  time inside each resident-loop tick. Collapse the runner pool into packed
  c-aware kernels only after a complete exact c2 performance packet proves a
  gain; do not relabel the current functionality result as native batch math.
- The primary generator runner is reused as one pool row and a second runner is
  allocated lazily/eagerly by resident prepare. Remove the dedicated pool once
  Qwen4Exp owns a multi-slot state/KV slab; preserve shared-once weights/PLE.

## 2026-08-28 Qwen4Exp bounded general multimodal

- `Qwen4ExpVisionRunner` preallocates at most 256 patch tokens per temporal pair
  and public multimodal generation remains capped at 1K. Replace the
  correctness-first one-thread vision attention and remove this cap only after
  general-grid Transformers parity plus memory/admission gates pass at larger
  shapes.
- HTTP multimodal currently accepts inline bounded PNG data URLs only and is
  non-streaming `n=1` without tools/session/continuation. Remove the dedicated
  chat branch after the normal request/batcher/stream owner carries typed media,
  request cancellation, exact token accounting, and safe remote-media policy.

## 2026-08-28 Qwen4Exp exact raw-Q8 prefill tiling

- `HIPENGINE_QWEN4_EXP_RAW_ROWBATCH` defaults to `32` and
  `HIPENGINE_QWEN4_EXP_RAW_VARIANT` defaults to `coltile8`, selecting exact
  coltile8/rowbatch4 for raw Q8_0 FP32 projections. `coltile` restores the
  preceding coltile4/rowbatch8 owner. Remove both flags and the losing geometry
  ladder after short-context serving/graph soak; keep the registered scalar
  strict fallback.
- `HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID` defaults to `64`, selecting exact
  fixed-worker Q4_K/Q5_1 grouped kernels. `0`, `q4`, and `q5` isolate rollback
  and family attribution. Remove the flag plus full-512-expert launch wrappers
  after serving/cancellation/graph soak; retain direct selected primitives as
  the strict unfused fallback.

## 2026-08-27 Qwen4Exp scalable device top-512 selection

- Production c1 and chunked QSA use device radix top-k selection and expansion,
  preserving lower-start tie order. The old host-selection default description
  is obsolete; CPU selection remains an independent reference.
- Cleanup removes the unused device-state host-selection method and mirrors
  with no remaining consumer. Preserve metadata still used by physical-page
  mapping, retrieval diagnostics and stateful graph probes.
- Remaining validation: complete-model index parity at 262K. Existing 262K
  allocation evidence does not establish inference parity. Synthetic selector
  parity at that length is a separate contract, not a substitute. Run the
  retrieval/index-reference gate at the full length and record cancellation,
  rollback and isolation results; do not gate execution on this missing row.

## 2026-08-26 speculative-MTP default capability migration

- September 12 gfx1151 dense Q4_K_M correction: the named production binder
  now forces FP32 recurrent state after C1/packed D128 max-KL failures.
  FP16 kernels and legacy Q4_K_S policy remain separate. Remove or replace
  the unbounded FP16 experiment only after a geometry/profile/horizon-owned
  storage plan passes the full long-horizon numerical and serving gates.
  Do not re-enable it by copying the historical D24 verifier manifest.
- Planar-Q6 integer-MMQ now requires explicit target-verifier workspace
  admission. The historical `HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL` name is
  misleading: it no longer authorizes ordinary AR prefill. Rename it with
  compatibility handling when the verifier configuration is consolidated;
  widen AR admission only after independent long-prefill/stateful gates.

## 2026-09-06 GGUF file-type stamp switches — open

- The initial audit found two gfx1151 switches that choose behavior from the header
  (`general.file_type`) instead of the file's actual per-tensor format mix:
  `GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES` and
  `GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES`. Any file stamped `MOSTLY_Q4_K_S`
  passes those identity predicates. Current UD K_S still fails planning on 41
  tensors; its file-global IQ veto prevents actual Q4 qmicro repacking, so this
  is an admission hazard, not observed recurrent/kernel execution.
- The verified audit also finds K_M/K_S policy identities in NextN, GDN,
  scratch/source-F16, Q6 arithmetic and automatic-MTP admission. Audit all
  callers rather than fixing only the original two constants.
- Removal trigger: bind policy to the actual role/shape/type/resident-layout
  manifest and artifact/profile evidence. A histogram or successful load alone
  is insufficient. Test identical stamps with different maps and identical
  histograms with types exchanged between sensitive roles. Preserve certified
  plain-file behavior; new UD load support must not inherit its numerical or
  speculative certificates.
- Related audit debt: fix `root.token_embd` versus `root.token_embedding`,
  AR/NextN scope, partial-header validation, omitted sidecars/capabilities and
  allocation accounting in `scripts/gguf_quant_route_audit.py` and its tests.
  Decouple global IQ repack admission from `contract_f32_linear` without changing
  existing MoE precision. ~~Remove the diagnostic policy mirror after a pure
  cold-path shared planner/capability interface supplies the report.~~ Done
  2026-09-07 (UD-U0): capability/flag resolution now goes through the shared
  pure policy API `hipengine.loading.qwen35_gguf_policy` (source-reading AST
  reader in the audit, `backend_package_capability` at runtime); the audit's
  regex/quoted-substring capability mirror and the local raw-IQ predicate copy
  are removed. The stamp-switch hazard itself stays open until U1 binds policy
  to the actual role/shape/type/layout manifest.
- Analysis and the wider campaign this sits inside:
  [`UD-QUANTS.md`](campaigns/UD-QUANTS.md), sections 2, 7.3 and U0/U1.

## 2026-09-07 W7900 packed C1 target qualification — open

- The physical-provider C1 route previously installed a legacy singleton
  target during preparation. Review prevents that substitution; its old C1
  rates are not packed-target evidence. Public C1 rows are withdrawn.
- Keep the repaired route diagnostic-only until packed C1 K1-K7, native N=1,
  N=2/8 slots/transitions, numerical/task, lifecycle and repeated-economics
  gates pass. Then register exact evidence keys; do not restore old rows by
  copying their legacy-target measurements.
- The screening environment variable was removed in the R0077 cleanup above.
  Runtime row-injection probes remain diagnostic tools, not public capability
  declarations or evidence of the packed target's behavior.
- The MTP2 structural depth bound now has one public constant,
  `MTP2_MAX_CANDIDATE_DEPTH`; its value remains seven and backend/profile cell
  admission is unchanged. Consolidate any remaining duplicated B1-B7 graph-cache
  fields after lifecycle coverage proves the cleanup. Shared capacity is not
  permission to widen Qwen3.6/gfx1151 public evidence.

## 2026-09-01 gfx1100 segmented GDN wave reduction — closed

- The complete counterbalanced C5-C8 category+heldout gate retained the exact
  request-local physical K128/V128 wave-reduced row-state owner as the gfx1100
  production default. Strict, FP16-state, peer, non-physical, and shape-miss
  routes retain their prior owners.
- `HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE=0` remains a same-build rollback.
  Remove the environment opt-out after the next stable milestone audit; keep
  the registered strict parent.

## 2026-09-01 gfx1100 Q4 row6 two-wave screen — closed

- The first unrouted claim remains retracted as invalid history. The corrected
  physical-wrapper route is independently retained after marker-scoped
  candidate ownership plus a complete positive C5-C8 category/heldout gate.
- `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2=0` remains same-build rollback. Remove
  the environment/cache policy after the next stable milestone audit; keep the
  registered WG32 strict parent. A separate physical-registry-key extension was
  removed after marker-scoped C8 evidence showed no ownership change.
- `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48=0` remains same-build rollback for
  the retained physical-R36 three-active-wave fused gate/up+SiLU owner. Remove
  the environment/cache policy after the next stable milestone audit; keep the
  row64 registered strict parent.

## 2026-09-02 NextN accepted-tail K/V-only integration — closed

- The existing exact K/V-only full-attention branch is now the default singleton,
  device-token, and physical-batch accepted-tail state-transition owner. The
  complete C5-C8 category+heldout gate improves every width and every binding
  slice with all 520 generated-ID/acceptance rows exact and all packets drained.
  `HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY=0` retains the full NextN block as
  same-build rollback.
- Remove the environment branch after the next stable milestone audit; keep the
  complete state transition callable as a diagnostic correctness parent.

## 2026-09-02 gfx1100 grouped Q4 R8 at active C5-C6 — closed

- Promoted the exact grouped-R8 primitive through gfx1100 capability metadata
  only for physical R24 packets with five or six live target slots after the
  current-source marker and complete category+heldout gates passed. C7/C8 and
  all scope/shape/row misses retain grouped R6.
- `HIPENGINE_GGUF_Q4_T16_GROUPED_ROWS8_C5C6=0` remains same-build grouped-R6
  rollback. Remove the environment/cache override after the next stable
  milestone audit; keep the registered grouped-R6 fallback.

## 2026-09-02 gfx1100 exact physical C7 target rows — closed

- Promoted exact physical C7/K3 R28 through gfx1100 capability metadata after
  both process orders and every category/heldout slice passed. It removes two
  inactive padding rows and composes exact Q4/Q5/Q6 prefixes and tails. C8
  remains padded R36 because its unpadded R32 screen loses the fused gate/up
  owner and regresses complete wall.
- `HIPENGINE_GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS=0` retains padded R30 for
  same-build rollback. Remove the environment override after the next stable
  milestone audit; keep the exact padded composition available.

## 2026-09-02 gfx1100 C8 adaptive source-Q5 MMQ — closed

- `HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ` now defaults on for the R32-only K-major
  D4S4-FP32 I64/J32 integer-WMMA consumer inside the existing gfx1100
  eight-request raw-Q5 owner. R24, explicit zero, and shape misses retain the
  qualified D4S4 raw MMQ; disabling `HIPENGINE_GGUF_C8_Q5_RAW_MMQ` retains
  grouped T16.
- The original source-faithful DS4 R24/R32 route passed strict-teacher numerics
  and improved pooled task throughput by 0.99%, but changed one acceptance
  decision and regressed `general_en` by 1.38%/1.90% across the two orders, so
  it is superseded. The replacement retains FP32 activation and weight
  metadata; its fast-math R32 operation screen wins all 48 weights by 2.77%
  with max KL below 3e-9 versus raw MMQ, while the bit-exact baseline build
  loses all 48 R32 weights by 4.34%.
- The replacement passes its strict-teacher gate over 240 rows at 0.000158 mean
  KL, 0.005678 max KL, and 99.583% top-1. Its staged counterbalanced task gate
  improves pooled throughput 88.932→90.082 tok/s (+1.29%), and the
  post-promotion same-build default-versus-explicit-zero gate confirms
  88.953→90.139 tok/s (+1.33%). Both gates win every prompt cell and every
  category/heldout slice in both orders with identical generated IDs, identical
  acceptance ledgers, and clean teardown. Marker attribution for the retained
  route is Q5 operation wall 12.794→10.940 ms/cycle (-14.49%) at identical
  launch counts; the earlier 11.215→10.760 ms/cycle "complete wall" quote came
  from proposal-phase kernel-family values and is superseded. Retain explicit
  zero for same-build rollback and remove the environment read after the next
  stable milestone audit.

## 2026-09-02 gfx1100 C8 raw-Q5 MMQ — closed

- The gfx1100-only eight-request physical R24/R32 `ssm_out` route is now the
  production default. It retains raw sidecars beside exact T16 residents,
  quantizes each layer input into bounded scratch, and launches the registered
  operation-complete raw Q5 MMQ. The all-48 actual-weight screen, marker gate,
  counterbalanced ten-prompt task gate, and strict-teacher R24/R32 numerics all
  passed; the promoted-default task gate improves 87.186→89.377 tok/s (+2.51%).
- `HIPENGINE_GGUF_C8_Q5_RAW_MMQ=0` remains same-build rollback. Request counts
  other than eight, peer backends, shape misses, missing sidecars, and workspace
  misses retain exact grouped T16 ownership. Remove the environment opt-out after the next stable milestone
  audit; preserve the registered strict fallback.

## 2026-09-02 gfx1100 grouped Q5 R8 C8 rollback — open

- Exact grouped-R8 now owns physical C8/R32 by default; explicit
  `HIPENGINE_GGUF_Q5_T16_GROUPED_ROWS8_C8=0` retains grouped R30 plus strict R2.
- Remove the environment override and its selector test after the next stable
  milestone; keep the grouped-R8 policy/kernel and registered strict fallback.

## 2026-09-02 gfx1100 periodic-strict fused Q4 R28 rollback — open

- Explicit `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28=0` restores exact
  R28 on every layer; explicit production otherwise fuses layers whose ID is
  not divisible by 8 while layers 0/8/…/56 remain exact. Omitted-profile
  behavior stays unchanged.
- The fixed schedule is correctness- and performance-qualified. Remove the
  environment override and its cache/test after the next stable milestone;
  keep the backend policy, production manifest selection, and registered exact
  fallback.

## 2026-09-02 gfx1100 fused Q4 row32 / exact C8 target rows — closed

- Promoted exact physical C8/K3 R32 and its two-active-wave fused gate/up+SiLU
  owner together after both process orders and every category/heldout slice
  passed. This removes four inactive target rows without losing the fused FFN
  transition.
- `HIPENGINE_GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS=0` retains padded R36 plus the
  fused row48 owner as atomic same-build rollback. Remove the environment
  override after the next stable milestone audit; keep the registered row48
  parent and padded composition.

## 2026-09-02 gfx1100 direct resident verifier linear state — closed

- Promoted through gfx1100 capability metadata after exact same-build C8 tracing
  and a positive complete C5-C8 category/heldout gate. Packed target verification
  indexes the resident batch owner's Conv/GDN slabs directly, while captured
  state rows remain read-only and the existing GPU-selected row kernel remains
  the sole exact post-accept commit owner.
- `HIPENGINE_GGUF_VERIFY_DIRECT_RESIDENT_LINEAR_STATE=0` restores the fused
  resident-to-packed import for bisection. Remove the environment override after
  the next stable milestone audit; keep the copy-in route as strict fallback.

## 2026-09-02 gfx1100 planar-Q6 root R8 rollback — closed

- The planar-Q6 F32 R8 root owner is promoted on gfx1100 and gfx1151.
  Complete counterbalanced C5-C8 qualification is recorded in
  `benchmarks/results/2026-09-02-w7900-q4km-k3-c5c8-q6-root-f32-rows8-retained.json`;
  the older statement that this promotion was pending is superseded.
- Both packages advertise maximum chunk 8.
  `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK=4` retains the prior R4 composition for
  bisection; values outside [2,8] are invalid. Remove the override after one
  release window plus defaults-only gfx1151/gfx1100 direct and server refreshes.
  Keep `_small_b_rowtile_chunks` and all 2-8 row kernels as exact fallbacks for
  arbitrary packed widths. The artifact's leaf screen preserves the original
  actual-weight comparison; no new measurement is claimed here.

## 2026-09-01 gfx1100 grouped Q6 mixed-rowtile rollback — closed

- Promoted after exact tracked-clean marker ownership and complete C5-C8
  qualification. `HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES=0` retains the
  prior mixed launch sequence for bisection; remove the flag after the grouped
  owners are stable across the next production campaign.

## 2026-09-01 gfx1100 grouped Q5 rows6 rollback — closed

- Promoted after exact tracked-clean marker ownership and complete C5-C8
  qualification. `HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6=0` retains the
  repeated-R6 path for bisection; remove the flag after the grouped owner is
  stable across the next production campaign.

## 2026-09-01 gfx1100 grouped Q4 rows6 pair-seam rollback — closed

- Promoted after exact marker ownership and complete C5-C8 qualification.
  `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_PAIR_ROWS6=0` retains the
  interleaved repeated-R6 rollback for bisecting; remove the flag after the
  grouped pair owner is stable across the next production campaign.

## 2026-09-01 gfx1100 grouped Q4 rows6 rollback — closed

- The grouped-grid Q4 owner is retained by default for consecutive physical R6
  chunks. `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_ROWS6=0` restores repeated
  WG64/two-wave R6 launches. Remove the environment/cache policy after the next
  stable milestone audit; keep the registered repeated-R6 strict fallback.

## 2026-09-01 gfx1100 fused packed-state transfer rollback — closed

- The existing strict pointer-table pair-copy leaf is the retained gfx1100
  default for resident-to-packed Conv/GDN imports, including physical C5-C8
  packed-verifier initial-state gather. `HIPENGINE_GGUF_FUSED_PACKED_STATE_TRANSFER=0`
  restores the per-layer HIP D2D chain. Remove the environment/cache policy
  after the next stable milestone audit; keep the strict per-layer fallback.

## 2026-09-01 W1 wide-Q6 shared-weight candidate context — open

- `target_verifier_wide_q6_shared4_session()` and the temporary
  `HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4` cross-thread experiment flag expose
  exact Q4 shared-b2w2 and standard/planar-Q6 shared-weight owners at physical
  R20/R24/R32 for W1. The screened Q5 WMMA addition is excluded after its
  complete target stage regressed 0.25%. The context defaults off; current
  chunk/direct owners remain strict fallbacks, and automatic C6/C8 stays K0.
- Synchronization resolved the historical symbol collision: gfx1151 retains
  its measured two-wave/two-tile `shared4` ABI while gfx1100 uses a distinct
  four-wave/three-tile `shared4_gfx1100` symbol. Do not collapse these
  backend-qualified geometries into one global symbol.
- Remove the candidate context after complete C6/C8 strict-teacher,
  determinism/isolation/task, full-suite performance, and profiler gates either
  promote or reject the shape policy. A promotion should leave one immutable
  profile/capability selection plus registered strict fallbacks.

## 2026-08-28 Qwen3.8 production verifier-Q4 selector — open

- `HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE` currently carries the resolved
  Qwen3.8 production-profile decision from the cold-path binder into resident
  sessions. The session freezes it once; hot verifier methods consume only the
  typed boolean and never reread the environment.
- Remove the environment handoff when `ResolvedRuntimePlan` (RF-1) carries
  profile scope selections directly into `Qwen35GGUFResidentSession`. Preserve
  the profile manifest selections, strict WMMA fallbacks, qualified
  R6/R8/R9/R12 actual-shape bounds, and benchmark rollback evidence.

## 2026-08-21 codebase simplification review

This review covers the tracked tree at `a9419949d`. The main conclusion is that
strict fallbacks are not the primary complexity problem. Complexity is coming
from experiment ownership living beside production ownership, policy being
re-resolved throughout runtime code, importable benchmark programs being shipped
as a product package, and historical evidence remaining in active source and
this active ledger.

### Measured inventory

- `hipengine/` contains **304,814 Python lines**. The original PLAN statement
  that the host is roughly 700 lines is historical and must not be used as a
  current architecture budget.
- `qwen35_gguf_runner.py` is **29,324 lines**; its two main classes are about
  **7,739** and **10,109** lines. `qwen35_paro_runner.py` is **14,260 lines**
  with a **12,304-line** resident-session class, and `qwen35_paro.py` is
  **11,892 lines** with a **10,516-line** decode-state class.
- `server/api.py` is **16,733 lines** and `create_app()` alone spans about
  **4,410 lines**. Protocol translation, generation orchestration, tools,
  structured-output validation, streaming, metrics, and route construction are
  currently coupled in one module.
- Product Python contains **225 direct environment-read expressions across 38
  modules**, **308** `backend_package_capability(...)` call sites, and **144**
  registry `resolve(...)` call sites. `hip_gfx1151/__init__.py` is **2,411
  lines** with 256 top-level capability/policy assignments. This is selection
  work that should largely resolve once into an immutable session plan.
- `scripts/` contains **577 tracked Python programs / 289,396 lines** and is
  included in the wheel by `pyproject.toml`. A conservative textual scan found
  **405** with no reference from live `hipengine`, tests, current docs, README,
  or package metadata. That does not prove they are unused command-line tools,
  but it proves they are not an acceptable implicit product API. One product
  module also imports `scripts.quant_quality.metrics`, preventing a clean
  package boundary.
- `benchmarks/results/` contains **5,158 tracked files / 225.7 MiB**, including
  multi-megabyte tensor/capture JSON. New policy requires compact artifacts, but
  the tree still mixes compact decisions with raw historical captures.
- Before this review, this file was **3,951 lines / 398,544 bytes**, with 176
  top-level entries. Its cleanup table contained 154 rows under only 120 area
  labels, including stale duplicate rows, while closed and already-removed
  history remained mixed with actionable work. This review removes three stale
  duplicate rows; RF-4 owns the larger active/history split.

These counts are triage signals, not line-count targets to game. A pure move that
creates more files but one clear owner is a win; a generic abstraction that
hides kernel ABI, numerical boundaries, or fallback ownership is not.

### Non-negotiable keep set

Do not delete these merely to reduce counts:

1. registered strict fused/unfused fallback chains;
2. complete-`KVLiveSpans` generic attention/KV-write routes and unsupported-
   shape/backend fallbacks;
3. scalar c1, segmented arbitrary-length, and transactional verifier oracles
   while optimized routes are compared against them;
4. `kernels/cpu_reference/`, strict fixtures, lifecycle/isolation gates, and
   compact negative-decision evidence;
5. explicit fallbacks still named by an open production-profile or hardware
   qualification gap.

The simplification rule is: remove **selection and production reachability**
for a rejected experiment first; retain a low-level primitive only when a named
fallback, active test oracle, or current research campaign consumes it.

### Priority architecture work

| ID | Priority | Problem | Target design / completion gate |
| --- | --- | --- | --- |
| RF-1 | P0 | Environment variables, backend capabilities, and registry variants are repeatedly resolved inside runtime methods. | Add a typed immutable `ResolvedRuntimePlan` assembled by model/backend/quant/profile plugins at model or session construction. Environment parsing stays in CLI/API/benchmark adapters. Hot runner methods consume the plan and do not reread environment or package maps. Keep registry-key misses mapped to strict fallback. |
| RF-2 | P0 | `scripts` is both a 289K-line research workspace and a shipped/importable package. | Move reusable metrics, schemas, fixtures, and artifact validators to `hipengine.benchmark` or `tests/support`; remove the `hipengine -> scripts` import; package only `hipengine` plus explicitly supported tool entry points. Keep source-only experiments outside the wheel. Do not delete a script referenced by an active benchmark protocol until its replacement command exists. |
| RF-3 | P0 | Rejected/default-off runtime experiments remain intertwined with production methods. | Execute the verified deletion queue below as small behavior-preserving units. A removed runtime selector may leave a focused kernel parity test; a production module must not retain an unrouted experiment solely because historical evidence exists. |
| RF-4 | P0 | This active ledger is itself an unbounded historical journal. | Keep `REFACTOR.md` as a compact active index with stable IDs, owner/scope, state (`ready`, `blocked`, `keep-fallback`), exact removal gate, and affected paths. Move closed narrative to `REFACTOR-HISTORY.md` or immutable worklogs. Add a check that rejects duplicate active IDs and closed entries in the active table. |
| RF-5 | P0 | Compact benchmark decisions and raw captures share one 225.7-MiB directory. | Define a size-bounded decision-artifact schema (default ceiling 256 KiB); keep only current/linked compact decisions in `benchmarks/results/`. Raw tensor traces/logs go to ignored local storage or external/LFS storage with SHA-256 plus retrieval metadata. Inventory references before deleting any existing evidence. |
| RF-6 | P1 | GGUF/PARO runners combine policy, allocation, all layer families, verification, graph capture, and session lifecycle in giant classes. | Extract pure policy/shape planning first, then scratch/state ownership, full attention, linear attention, MoE, verification, and session façade modules. Preserve public imports through thin compatibility façades during migration. No numerical or dispatch change may share a move-only commit. |
| RF-7 | P1 | `server/api.py` combines unrelated protocol and execution concerns; `create_app()` is a 4K-line closure tree. | Split OpenAI request/response schemas, tool/structured validation, SSE/stream accounting, metrics, model lifecycle, and route handlers. `create_app()` should compose routers/services rather than define the implementation. The server must call one typed generation service; it must not become a second scheduler. |
| RF-8 | P1 | Backend package `__init__.py` files are capability databases, alias registrars, and implementation code. | Split backend policy by model/quant/subsystem into immutable manifest modules, with one package aggregator and validation. Keep the four-axis registry; do not replace it with backend conditionals. Generate or declaratively register repetitive aliases only when source-lineage and exact-key tests remain readable. |
| RF-9 | P1 | `laguna_kv_attention.hip` (18,799 lines), its Python wrapper (11,275), and 768-line registration function mix production and rejected geometry ladders. | Partition KV write, global decode/prefill, SWA decode/prefill, and experiment-only families. Delete unwrapped rejected bodies; move retained negative controls to explicit test-only builds where practical. Keep production kernel names and lineage stable during mechanical splits. |
| RF-10 | P1 | GGUF linear launchers and generation owners encode a combinatorial decision tree in large functions; direct controls remain callable from production modules. | Replace repeated pair/triple/residual branching with typed launch-plan records resolved before launch. Move direct AR/MTP control loops to explicit oracle helpers once `SubmitPollTextGenerator` owns the same gates. Keep unsupported-shape fallbacks, but give each request exactly one scheduler/commit owner. |
| RF-11 | P1 | Whole-file SHA tests pin unrelated evolving files, including this ledger and backend package initializers. | Replace whole-file hashes with semantic assertions over the capability/registry owner and hashes of immutable artifacts or localized frozen blocks. A documentation edit must not require refreshing an unrelated kernel-source qualification test. |
| RF-12 | P2 | Test and architecture documentation mirror the same monoliths (`test_integration_generation_batch_scheduler.py` is 28K lines; `test_integration_server_api.py` exceeds 20K), and PLAN budgets are stale. | Split tests by behavior while retaining shared fixtures, then refresh PLAN's measured current/target budgets. Add lightweight growth checks only after module boundaries exist; do not enforce arbitrary line limits on the present monoliths. |

### Verified deletion queue

These are code removals to execute separately, not deletions performed by this
documentation review:

| Scope | Why it is ready | What to remove / what to keep |
| --- | --- | --- |
| Rejected MTP/llama-compat flag zoo | Multiple Q6 top-1 shapes, shared-Q8, selected X8/raw gate-up, fused rotate, accept-position, and overlap routes are explicitly rejected but still runtime-selectable. | After the final named llama-compat transaction policy is chosen, retain one named compat composition plus the strict control; remove loose env/CLI combinations and demote useful leaf oracles to tests. Do not remove an option still consumed by that named composition before the decision. |
| Superseded one-off scripts | 405 scripts have no textual owner in live code/tests/current docs/package metadata, and many are per-layer or one-candidate audit copies. | First remove them from the wheel. Then consolidate parameter-only families into one driver and delete/archive superseded copies after preserving command, result hash, and decision in an immutable worklog/artifact. “No textual owner” is a review queue, not sufficient deletion proof. |
| Qwen3.8 native-XL v2 loader admission | `OI-7` retained the artifact planner/oracle evidence but rejected deployment: native B1/B2/B3 diverged from target AR on `general_ja_explain`, while serial-exact was correct but slower than AR and full-suite native rates regressed 43–61%. | Keep `qwen38_native_xl_v2` research-only. Remove its artifact-specific NextN loader recognition/tests after the local replay window, unless a replacement artifact first passes full native batch-composition exactness and same-host AR/MTP throughput gates. The generic planner, verifier, and independent BF16 oracle are reusable and remain. |

### Target organization

The intended direction is a collection of stable façades over cohesive modules,
not a new framework:

```text
hipengine/runtime/qwen35_gguf/
  policy.py        # immutable model/backend/quant/profile plan
  scratch.py       # byte plans, owners, views, lifecycle
  full_attention.py
  linear_attention.py
  moe.py
  verify.py        # journal/commit/rollback; no server policy
  session.py       # thin public façade
hipengine/server/
  routes.py        # HTTP translation only
  streaming.py     # SSE and token accounting
  structured.py    # JSON schema/guided output
  tools.py
  metrics.py
hipengine/kernels/hip_gfx1151/policy/
  qwen35.py
  qwen38.py
  laguna.py
```

Apply the same split to PARO only after the GGUF extraction pattern proves it
avoids import cycles. Keep primitive launch wrappers near their kernel family;
do not centralize raw pointers into an untyped universal launcher.

### Refactor execution rules

1. Remove ready rejected branches before moving production code; otherwise a
   mechanical split preserves the very complexity we want to eliminate.
2. One logical unit is either deletion, policy extraction, or file movement —
   never combine it with tuning, arithmetic, or default changes.
3. Before deleting a path, prove current default and strict fallback ownership,
   search runtime/tests/docs/benchmark protocols, and name the focused gate.
4. Preserve import compatibility during module splits, then remove re-exports
   after one compatibility window rather than creating permanent duplicate APIs.
5. Update PLAN only when a proposed target boundary becomes implemented
   architecture. This review is the cleanup queue, not permission for a big-bang
   rewrite.

### Completion signals

The refactor campaign is materially complete when production runtime modules no
longer read experiment env flags directly, backend/model/profile policy resolves
once, `scripts` is not part of the default wheel, rejected experiments are not
runtime-reachable, active refactor debt is a bounded indexed document, and the
server/model runners have one scheduler plus one state/KV commit owner. Strict
fallback count is not a success metric.

## SPECDEC2 whole-request compatibility seam

- Added 2026-08-24 for the approved [`SPECDEC2.md`](reference/SPECDEC2.md) campaign.
  During S1-S4 the new Generation-2 one-cycle provider/target path coexists with
  the synchronous `SubmitPollTextGenerator.submit_speculative_many_detailed()`
  route, which currently finishes complete model-owned generation under the
  submission/model-loop lock and then publishes synthetic `VERIFY_CHAIN`
  metadata. Existing whole-request provider protocols and direct loops remain
  only for rollback, parity, and exact-oracle use while the staged provider SPI
  is qualified.
- Removal trigger: S3 must match direct c1 IDs/cycle semantics and exact
  hidden/Conv/GDN/KV/cursor/following-AR state; S4 must pass physical c2/c4,
  late-admission/refill, streaming, cancellation, and ownership gates. Then
  remove production selection of the synchronous route, synthetic post-hoc work
  metadata, and duplicate request/result ownership. Keep separately named
  direct controls and registered strict/eager/serial kernel fallbacks required
  by the execution-profile contract.

## SPECDEC2 gfx1100 packed-PARO C1 bring-up debt

- Added 2026-08-25 for S7 P1. The staged provider streams target prompt hidden
  rows directly into NextN, pools the heavy proposer, and P4 now carries the K1
  candidate as a stable borrowed-W8A16 device scalar through target accept.
  Bounded candidate readback follows target synchronization. Target backup still
  copies canonical slot 0 into spare slot 1 each cycle, and complete production
  wall remains below AR/direct, so explicit/default-off and automatic K0 bind.
- Removal/promotion trigger: move target result/accept/selected-commit/provider-
  update metadata and hot backup to stable claim-backed slabs, prove exact
  rollback/lifecycle engagement, and pass complete same-suite wall against the
  qualified direct fast B1 control. C2/C4
  additionally requires physical multi-request PARO proposal/target ownership;
  do not broaden the singleton adapter.

## SPECDEC2 dense prompt-activation fallback

- Added 2026-08-25 with SPECDEC2-PERF P2. Fresh strict C1/C2/C4 requests stream
  pre-output-norm target chunks directly into NextN and retain only one 10 KiB
  carried BF16 row per request. `HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT=0`
  temporarily selects pre-mutation K0/AR for clean A/B and rollback; it does not
  pay the obsolete prompt-sized replay. The older prompt-sized FP32 hidden
  capture plus post-prefill F32→BF16 replay remains only as an internal strict
  oracle for providers without the streaming prompt ABI while P2 is qualified.
- Removal trigger: qualify streamed prefix restore/COW and every supported
  incremental-prefill/context transition with exact provider fingerprints,
  cancellation/failure cleanup, and complete same-suite non-regression. Then
  remove `HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT`, `_prompt_hidden_rows`, and
  `_catch_up_provider{,_batch}` from production selection while retaining a
  focused test oracle if still useful.

## SPECDEC2 MoE physical-C2 Q4 pair-reuse scope

- Added 2026-08-27 after actual R5/R6 microbench and full-suite evidence. The
  request-local `moe_physical_c2_pairreuse_session(...)` selects the exact
  duplicate-expert Q4 gate/up kernel only for x_rows 2-6, selected rows
  `x_rows*8`, K2048/N512. C1, C8, non-MoE, and Q5 down keep existing owners.
- Removal trigger: fold the route into a profile-resolved MoE physical-target
  manifest after C2 promotion or remove it if the final three-repeat gate loses.

## SPECDEC2 MoE request-local C2 prototype

- Added 2026-08-27 as the first functional physical-C2 seam in
  `Qwen35GGUFMoEMTP2Adapter`. Capacity-2 model-plugin construction now selects
  the architecture-shaped `Qwen35GGUFMTP2Adapter` with a batched MoE NextN
  executor, so the prototype C2 proposal/accept implementation in the old MoE
  adapter is unreachable; capacity 1 still uses its qualified complete-cycle
  implementation.
- Removal trigger: after the batched provider passes complete C2 production and
  lifecycle gates, delete the unreachable capacity-2 methods/branches and keep
  only the retained C1 adapter plus the capacity-aware factory.

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
  prerequisite for that transport cleanup.

## SPECDEC2 MoE physical-C2 exact-linear fallback

- Added 2026-08-27 after the packed rows6 linear-attention composite failed the
  binding MoE production gate. The request-local
  `moe_physical_c2_exact_linear_session(...)` keeps one layer-major physical
  target owner but executes two transaction-local R3 state segments and scalar
  attention rows, then one row-batched MoE FFN. This is the strict-arithmetic
  correctness fallback; it is intentionally too slow for automatic promotion.
- Removal trigger: replace scalar Q8/Conv/GDN launches with independently
  exact, row-amortized R6 kernels that preserve every selected state journal and
  pass the complete quality gate at >=1.10x AR. Then delete this scope and keep
  the row-exact implementation only as the registered fallback.

## SPECDEC2 physical production scope

- Added 2026-08-27 for the gfx1100 physical-C2 production campaign. The backend-
  and profile-scoped prompt-streaming capability enables normalized target rows
  only for the qualified production owner; strict retains host replay. The request-
  local `q4_t16_physical_extra_rowtiles_session(...)` selects exact C1-equivalent
  rows6 rowtiles for K5120/N17408 gate/up and K6144/N5120 attention output only
  when the production adapter and backend capability agree; strict keeps the
  registered single-wave/shared-B fallbacks. The same request-local policy
  selects the bit-exact Q5 rows6 four-column and planar-Q6 rows6 col8 rowtiles
  while strict keeps their serial-row owners. Since 2026-08-29, physical groups
  pad to rows6 multiples through R24 and split Q4/Q5/Q6 plus the unfused
  gate/up fallback into consecutive qualified rows6 launches. Scope misses,
  strict arithmetic, and ordinary prefill retain the prior owners.
  The scope avoids changing the existing C1 production manifest while physical
  production owns an independent policy fingerprint and quality/economics
  packet. P12 adds one exact automatic evidence row for standard Q4_K_M,
  gfx1100 production/BF16/cap2/C2/K2/context4-95/D24; all scope misses remain
  K0. The device-resident proposal's enqueue-only call retains synchronized
  default behavior for every other caller, and physical accept relies on its
  bounded blocking D2H while the legacy diagnostic path remains synchronized.
  The independently exact R8 owner is now the production default after its
  counterbalanced full C2/K3 category+heldout gate passed: physical C2/K3 omits
  four inactive padding rows and selects existing Q4/Q5/Q6 rows8 rowtiles,
  while R6 and every other physical width retain the established rows6-multiple
  route. `HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS=0` restores padded R12 on
  the same build.
- Removal trigger: after one release checkpoint, remove the exact-R8 env flag
  and fold R8 into the profile-resolved physical-target variant manifest while
  retaining the padded rows6 decomposition as the registered strict fallback.
  The planar-Q6 mixed-R8 composition is now the production default: R24 uses
  `8+8+8`, R30 uses `8+8+8+6`, and R36 uses `8+8+8+6+6` at measured actual
  recurrent-QKV, full-attention-V, and FFN-down shapes. R18 and shape misses
  retain repeated R6. Actual-weight leaves are exact in 12/12 cells and improve
  admitted rows by 1.079-1.234x; the complete C5-C8 gate improves every
  width/category/heldout scope. `HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES=0`
  restores repeated R6 on the same build.
- Removal trigger: after one release checkpoint, remove the mixed-Q6 env flag
  and fold the retained partitions into the physical-target variant manifest;
  preserve repeated R6 as the registered strict fallback.
- Removal trigger: after one release window with the exact automatic C2 key and
  rows6-multiple explicit C3/C4 route, collapse prompt streaming, padding, and
  these boolean scopes into a general profile-resolved physical-target variant
  manifest keyed by C/R/shape. Preserve the exact evidence-row bounds, strict
  fallbacks, synchronized caller fallback, unfused pair ownership, and focused
  offset/selector tests.

## gfx1100 dense SPECDEC2 C8 physical-width rollback

- Added 2026-08-31 with task #30. The gfx1100 production package qualifies one
  dense MTP2 physical group through eight requests; capability/frontier,
  provider, cycle-hidden, padded target, and accept owners derive their bounds
  from that package declaration. Strict remains on its existing C4/exact
  fallback route. `HIPENGINE_GGUF_SPECDEC2_MTP2_MAX_REQUESTS=4` restores both
  the explicit server route admission cap and the prior `[4,1]`/`[4,4]`
  production decomposition on the same build; it cannot widen beyond the
  package limit.
- Removal trigger: after one release window with repeated full-category C5/C8
  production D1/D24 evidence, exact control/ownership, deterministic/isolation
  and lifecycle gates, remove the environment read while preserving the package
  capability, derived owners, allocation-failure cleanup, and strict fallback.

## Qwen3.8 NextN selected-vocabulary proposal head

- Added 2026-08-31 as a screen sourced from exllamav3 PR #303 and promoted
  2026-09-01 for Qwen3.8-27B `Q4_K_M` production. The packaged model/tokenizer-bound
  131,072-row CJK-aware map repacks selected Q6 rows into a compact planar-T16
  head; singleton and physical-batch reducers map winners to full token IDs on
  device. The full target remains authoritative. Strict and
  `HIPENGINE_GGUF_MTP_HOT_VOCAB=0` retain the exact full-vocabulary proposal
  owner; an explicit path selects another validated map.
- Removal trigger: after one release checkpoint, remove the environment path
  override and retain a simple on/off rollback while preserving the packaged
  model-plugin artifact, exact identity validation, and full-head registered
  fallback.

## Server default-AR ready-cohort rollback

- Added 2026-08-31 with task #29. Default-route HTTP items already inside the
  configured generation batch window now enter one
  `EngineService.submit_request_batches` command, so every child is admitted
  before the next model poll while retaining its own collector, cancellation,
  completion, backpressure, and reclaim lifecycle. Dynamically arriving items
  use free capacity in a later cohort. `HIPENGINE_SERVER_DEFAULT_AR_READY_COHORT=0`
  restores the prior one-command-per-item race on the same build; non-independent
  generators and streaming routes are unchanged.
- Removal trigger: after one release window with clean repeated C1-C3 server
  evidence showing full native-prefill groups, exact outputs, independent early
  completion/failure/cancellation, and no C1 regression, remove only the env
  lookup and old independent launch branch. Preserve the atomic EngineService
  cohort seam and per-child handles.

## gfx1100 fused Q4-T16 prefill row-retile bisection

- Added 2026-08-31 with task #37. The separately registered standard-Q4
  gate/up+SiLU row48, row64 and row128 siblings preserve the 256-row parent's
  K16 WMMA, FP32 accumulation, BF16 projection boundary, and SiLU association.
  Backend policy selects row48 at rows33-48, row64 at rows49-64 and row128
  at rows65-128; rows129+ and every quant/shape/backend miss retain the
  parent. `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE=0` restores that parent on
  the same build; the exact retiles are now the default after the complete
  counterbalanced C1-C3 category+heldout gate passed. The resolved value is
  cached on the launch path and cannot broaden beyond the measured bands.
- Removal trigger: after one release window with a defaults-only repeated
  C1-C3 gate, remove the environment lookup/cache while preserving the bounded
  shape/range policy and permanent 256-row registered fallback.

## gfx1100 Q4-T16 row64 down-projection bisection

- Added 2026-08-31 with task #11. The exact `(17408,5120)` rows33-192 owner is
  selected by backend shape/range policy; the separately registered 256-row
  shared-B parent remains the strict fallback.
  `HIPENGINE_GGUF_Q4_T16_SHARED_B_ROW64_MAX_ROWS=0` restores the prior route:
  the 256-row shared-B parent for rows33-128 and the 64-column wide-N owner for
  rows129-192. The resolved value is cached because
  selection runs on every projection launch and the knob cannot widen beyond
  the qualified 33..192 range.
- Removal trigger: after one release window with the row64 default and a clean
  same-protocol repeat, remove the environment lookup/cache while preserving
  the shape/range policy and explicit registered parent fallback.

## gfx1100 planar-Q6 exact prefill retile bisection

- Added 2026-08-31 during the P13 C1-prefill attribution follow-up. Backend
  shape/range policy selects exact cooperative siblings only for planar-Q6
  `(17408,5120)` prompt-prefill rows33-511 (row64 through 128, shared256 from
  129); the one-wave parent is separately registered and
  `HIPENGINE_GGUF_Q6_PLANAR_EXACT_PREFILL=0` restores it on the same build. The
  resolved boolean is cached because this wrapper is reached by every
  qualifying projection launch. Decode, non-WMMA physical verification, peer
  backends, unqualified shapes, and out-of-range rows keep their prior owners.
- The binding two-pair full-suite C1-C3 gate is retained with all categories,
  heldouts, exact trajectories, native groups, trace identity, and drain green.
- Removal trigger: after one later release-checkpoint same-protocol repeat
  retains the policy, remove the environment lookup/cache while preserving the
  backend policy and registered one-wave fallback.

## SPECDEC2 dense NextN host/device proposal duplication

- Added 2026-08-25 with the S4 C2/C4 device-candidate handoff. The exact
  physically batched NextN model transition currently has host-materialized and
  device-top1 entry points in `qwen35_gguf_nextn.py`; they deliberately share
  weights/state/session ownership but duplicate the embedding/norm/fusion/
  backbone launch sequence while S4 keeps the old host path as an oracle.
- Removal trigger: after S4 device accept/commit and graph/eager agreement pass,
  extract one enqueue-only physical transition primitive with host/device result
  adapters. Keep the strict host materialization oracle callable from tests, but
  remove duplicate model-launch bodies from production selection.

## SPECDEC2 P4 device-chain qualification oracle

- Added 2026-08-25 with SPECDEC2-PERF P4. The promoted C1/C2/C4 cycle keeps
  proposal and target IDs device-resident through GPU accept and selected
  commit, then reads only bounded committed output/status. Setting
  `HIPENGINE_SPECDEC2_DEVICE_CHAIN_ORACLE=1` temporarily adds post-commit
  candidate/target-ID materialization and the CPU acceptance oracle for strict
  qualification; it is never part of default production timing.
- Removal trigger: after P4 reject/every-partial/full, following-AR,
  cancellation/failure, profile, and full category gates are retained with
  GPU/CPU accept agreement and cache-only traces prove no pre-accept D2H in the
  default route, move the oracle to a test/qualification adapter and remove the
  runtime environment read. Keep the pure CPU acceptance implementation and
  strict eager fallback.

## SPECDEC2 request-local ngram-mod experiment

- Added 2026-08-28 for the coding/repetition composition spike.
  `HIPENGINE_GGUF_SPECDEC2_NGRAM_MOD=1` gives an exact request-local 24-token
  continuation cache first refusal before dense GGUF MTP2; match/min/probe
  bounds are temporarily adjustable through
  `HIPENGINE_GGUF_SPECDEC2_NGRAM_{MATCH,MIN,PROBE_MAX}`. A hit verifies only
  the already-qualified K<=3 prefix, then catches target-attached MTP state up
  from target-owned hidden rows. A miss, mixed-source physical group, disabled
  flag, unsupported shape, or failure keeps the registered ordinary MTP/AR
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
- Removal/promotion trigger: replace environment parsing with a typed cold
  policy only after a K>3 frontier (or another materially different lowering)
  passes canonical + code train/heldout, long-horizon strict and production
  numerical/task/lifecycle gates, and beats true AR rather than MTP alone.
  Otherwise remove runtime composition at the end of the next dedicated wider-
  frontier campaign and keep the cache/harness as research references. Ordinary
  MTP plus AR remain permanent miss/failure fallbacks.

## Execution-profile migration seam

- Implemented for campaign P3: public `strict|production|batch_invariant`
  selection resolves to an immutable variant manifest through
  model/backend/quant plugin plans, with per-scope strict fallback and exact-key
  registry validation. `LLM.execution_profile_manifest_sha256` and server model
  discovery expose the resolution. To avoid an unannounced behavior change
  during calibration, an omitted profile temporarily bypasses plan resolution
  and preserves the pre-profile package selection. That compatibility behavior
  is not a fourth named profile. `HIPENGINE_EXECUTION_PROFILE` is the sole
  environment adapter; explicit API/CLI values win.
- 2026-09-12 (v0.5.0): an omitted profile now resolves to `production` for
  every combination with a registered strict+production plan (seven at this
  writing; the list is in `EXECUTION-PROFILES.md` §2.1). The named default, the
  strict fallback, and the batch-invariant fallback are all reachable, and the
  API/server docs name the default.
- Removal trigger for the remaining legacy path: after P4 re-certifies or
  replaces every current non-exact default so that every shipped
  model/backend/quant has a registered plan, and one release window confirms
  strict fallback and batch-invariant behavior. Then remove omitted-profile
  legacy selection and any duplicate env-to-profile adapter. Keep registered
  strict fallbacks.

## Generation-2 GGUF width-selection hardcoding

- Audited 2026-08-20. `GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS` was promoted to
  `(1,2,3,4,5,6,7,8)` after direct c3/c5/c6/c7 lifecycle certification. The
  artifact-backed D2 resolver remains explicit-config only through
  `HIPENGINE_GGUF_AR_D2_COST_ARTIFACT`; ceiling is the production default. The
  attempted default-promotion sweep was eager resident-session, not actual-
  server evidence, and its composed-c13 lifecycle observation was dirty. The
  2026-08-22 core-merge decision closes #29 on the safe ceiling default; a
  future D2 promotion/removal campaign is optional performance work, not a
  gfx1100 correctness or integration blocker.
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

- Added 2026-08-17 with C2-3. Generation-2 dense backends use one load-time
  `GlobalKVPoolSet`, stable per-plane arbitrary-page pointer tables,
  `KVStorageView`, typed leases, and KV-aware execution. C2-6 migrates the
  gfx1100 Qwen3.6 GGUF BF16 package through `GlobalDeviceKVPool`: one stable
  arena and per-layer/per-plane pointer tables back arbitrary free-page leases;
  no request-local chunk, contiguous-run admission, growth, or shrink remains
  on that selected path.
- `DeviceChunkedKVPool` and base-pointer identity binding remain as explicit
  numerical/peer-package fallbacks for unported GGUF layouts and backends. They
  must not leak same-chunk constraints into the Generation-2 scheduler, ledger,
  or new backend contracts.
- Shared-slot physical c4/c8 were first held unregistered after their old
  direct-top1 route emitted sentinel `2147483647`. The exact promoted package
  scoped Q8_1 direct-top1 to c1, registered physical `(1, 2)`, flushed canonical
  state between c2 groups, and decomposed logical c1-c32 honestly. On 2026-08-17
  the packed lm-head/state paths were re-proven byte-exact at c4/c8 (steady,
  masked-lane shrink-sparse, fixed-width graph, and p512 state-oracle gates), so
  the registered shared-slot widths are now `(1, 2, 3, 4, 5, 6, 7, 8)`:
  logical c4/c8 lower to one physical c4/c8 bucket. Matched W7900 p128/d8 c8 improved exact serial-c1
  **27.634 -> 44.031 tok/s (+59.27%)** (was +29.68% at c2), and p512/d128 c8
  reaches ~0.973x the old design. Serial remains the required numerical fallback.
- Removal trigger: after BF16 long-context and artifact-qualified no-mirror INT8
  global adapters execute the C2-6 graph-pointer, lifecycle, cancellation,
  pressure, SLO, and production-load gates on both gfx1100 and gfx1151, migrate
  every remaining registered GGUF package and remove the chunk allocator,
  same-backing admission/identity branches, and compatibility-only fixtures.
  Keep a separately registered unfused/serial numerical fallback as required.

## C2-6 production oracle process-partition seam

- Added 2026-08-17 after repeated W7900 page faults when a 13-key exact-oracle
  campaign handed the same HIP generation to measured serving. The production
  load harness can partition oracle keys into isolated workers, verify merged
  token digests, and start a separate measured owner. This is benchmark
  containment, not runtime serving architecture.
- The current GPU generation remained unstable after 4/4/4/1 partitioning and
  after replacing workers with fully drained same-owner serial c1 oracles.
  Disabling decode graphs during all 13 same-owner oracle rows still faults
  before the first tuning result. The canonical packet is blocked; worker and
  same-owner modes remain only for fresh-generation rerun/bisection.
- Removal trigger: after the canonical clean-source production packet completes
  twice without an oracle teardown/handoff fault, collapse to the simplest
  stable independent-oracle protocol and remove abandoned worker variants.

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
- Remaining cleanup: move packed-workspace ownership into the load-time resource
  ledger while preserving teardown order and lifecycle tests. Remove the session
  method only after every teardown caller uses the replacement owner; do not
  restore per-row reclaim releases.

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

## Generation-1 prefill/decode policy compatibility choices

- Added 2026-08-17 with C2-5. `token_budget` is the Generation-2 scheduling
  architecture: each round rotates bounded prefill quanta by stable slot and
  then advances every due decode row once. `protect_decode`, `protect_ttft`,
  and `fair` remain available because current GGUF packages and their retained
  evidence select them explicitly; they are compatibility/rollback policy
  names, not the long-term production planner interface.
- Removal trigger: after C2-6 derives per-model/backend round budgets from
  measured TTFT/ITL SLO rows and passes c1-c32 production soaks on gfx1100 and
  gfx1151, promote those registered token budgets, remove generic policy
  selection from public production configuration, and retain only a diagnostic
  compatibility seam while old GGUF runners remain supported.

## gfx1100 GGUF Q4_K_M fair launch-policy default (scoped registry default)

- Added 2026-08-17. The `("hip_gfx1100", "gguf_q4_k_m")` generator factory now
  declares `engine_loop_config_defaults = fair:256/burst-1` via the
  `GGUF_Q4_K_M_*` gfx1100 package capabilities, mirroring the F4-retained
  gfx1151 scoped default. The generic engine-loop default and the
  `HIPENGINE_PREFILL_DECODE_POLICY` env resolution still default to
  `protect_decode`; the env remains the explicit pin for configurations that
  must stay on `protect_decode` (the A4 frozen W7900 UD-Q4_K_M gates — a
  different quant, unaffected by this default).
- Measured justification: W7900 Qwen3.8-27B Q4_K_M/BF16-KV 16K server —
  `protect_decode` produced zero packed decode steps at c4/c8 bursts (requests
  serialized: one request's decode finished before the next prefill);
  `fair` produced width-4 packed AR decode (40 packed steps, zero serial
  fallback, zero fallback steps).
- Removal trigger: when `fair` is proven as the generic default (A4 rerun under
  the new default for any affected `gguf_q4_k_m` configuration, plus the other
  gfx1100 GGUF quants/models pass their load/SLO gates), collapse the scoped
  factory default into the generic `EngineLoopConfig`/env default and delete
  the per-package `GGUF_Q4_K_M_PREFILL_DECODE_POLICY` capability. Until then
  the env pin and the scoped default are the rollback/bisection seam.

## Qwen3.8 c>1 mirrored-INT8 serving seam

- Added 2026-08-15 after the dedicated-XTX context soak and corrected after the
  actual-limit and controlled-SSE audits. Long c1 uses genuine no-mirror
  FP32-scale INT8 at `8,519,680` bytes/page. The XTX one-request physical
  ceiling is now 126K while four natural 112K requests remain the recommended
  repeated row; W7900 reaches the model-native 256K limit.
- Current c>1 packed AR admits per-token/head INT8 only at short contexts where
  it also retains BF16 K/V, raising each page to `25,296,896` bytes. W7900
  c2/c4/c8 pass 8K/request and all fail closed at 8.25K because direct packed
  INT8 attention is not admitted. XTX actual execution passes reach 6K/request
  at c2 (8K is 0/2 HIP OOM), 1.5K at physical c4 (no failure tested), and 2K at
  offered c8 (no failure tested). Do not advertise any mirrored row as an INT8
  memory saving.
- The short mirrored scheduler lifecycle itself is no longer a blocker:
  gfx1100 staggered c1->c4 SSE is 4/4 exact with occupancy
  `0->1->4->3->2->1->0`, admitted/reclaimed `4/4`, and zero final ownership.
  The seam is storage/attention/prefill/admission, not continuous membership.
- `IKV-C1` completed the explicit serial no-mirror anchor on 2026-08-16 for
  the exact gfx1100 artifact through logical c4: shifted logits/state/KV are
  exact, active persistent BF16 bytes are zero, and cancellation/survivor
  ownership drains cleanly. The route deliberately reports physical width 1
  and makes no throughput claim.
- Removal trigger remains the approved
  [`IKV-C2`-`IKV-C7` campaign](campaigns/QWEN38-INT8-KV-CONTINUOUS.md): replace the serial
  route with row-batched direct INT8 attention, complete prefill and admission
  ownership, and rerun the artifact-scoped c1/c2/c4/c8 matrices. Remove the
  mirrored short route when the replacement is exact, memory-positive,
  non-regressive, and no supported caller needs it. Remove or demote the serial
  path after the native batch route passes; retain a separately registered
  numerical fallback as required by `AGENTS.md`.
## gfx1151 Q8T16 dual-WMMA alpha/beta rollback seam

- Added 2026-08-15 for D08-X8 exact rows512/K1024/N16+N16 alpha/beta
  promotion. `HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL=0` restores the two
  singleton registered Q8T16 WMMA projections. Shape/backend/registry misses
  also fail closed to those primitives; no resident or scratch allocation
  depends on the pair owner.
- Removal trigger: after one release window with Q4/Q8 p512, decode, cumulative
  semantic, and lifecycle gates stable, remove the selector/cache dimension.
  Keep the singleton kernels as required numerical and peer-backend fallbacks.

## gfx1151 dense-BF16 WMMA residual rollback seam

- Added 2026-08-15 for D08-X6 exact rows512 FFN-down plus residual promotion.
  `HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL=0` preserves the already-retained WMMA
  projection plus standalone `gguf_bf16_add` chain for paired bisection. Registry
  and shape misses also fail closed to that primitive chain; no resident or
  scratch bytes depend on the fused owner.
- Removal trigger: after one release window with Q4/Q8 p512, eager/graph decode,
  cumulative semantic, and lifecycle gates stable, remove the environment
  selector and dead default-off branch. Keep the registered primitive chain as
  the required numerical and peer-backend fallback.

## gfx1151 Qwen3.8 c1 down-residual rollback seam

- Added 2026-08-16 for the exact Q4T16/Q6-qmicro K17,408/N5,120
  projection-plus-rounded-residual graph contraction. Setting
  `HIPENGINE_GGUF_DENSE_DOWN_RESIDUAL_DECODE=0` restores the registered direct
  projections plus standalone BF16 add. Registry, model/file/shape, rows>1,
  and peer-backend misses also keep that primitive chain; no resident or
  scratch bytes depend on the composite.
- Removal trigger: after one release window with 512/1K/4K graph decode,
  natural AR/B1, and lifecycle gates stable, remove the environment selector
  and cached negative branch. Keep the primitive projection-plus-add chain as
  the required numerical and peer-backend fallback.

## gfx1100 in-tree retained-PM4 transport comparison seams

- Added 2026-08-07 for explicit `hipgraph|aql|pm4` isolation and promotion.
  `HIPENGINE_SUBMISSION_TRANSPORT`, direct AQL, timestamp/quarantine lifecycle
  controls, and `scripts/pm4_gguf_decode_gate.py` remain diagnostic/default-off;
  native HIP graph remains the portable default.
- Completed 2026-08-08: tracked-clean setup/performance plus the 19/19 natural,
  heldout, 4K, rebuild, cancellation, shutdown, and memory matrix selects
  stateful-register elision with local-cache dependency acquire as the sole
  explicit-PM4 production encoder. The two experiment environment selectors are
  removed from transport construction.
- Completed 2026-08-08: packed c2/c4/c8 submission wiring is retained for
  scoped gfx1100 production promotion. Full attribution found a 2.9–3.8 ms
  per-token full-provenance JSON cost, not a PM4 dependency defect; deferring
  proof generation moves clean replay from **+12.551%/+9.196%/+5.687%** to
  **-6.626%/-4.126%/-2.466%**, with 5/5 wins and exact natural/lifecycle gates.
  Never restore full native executable provenance serialization to the replay
  hot path; refresh it explicitly at diagnostic or artifact boundaries.
- The backend capability `GGUF_DECODE_GRAPH_SUBMISSION_POLICIES` is an exact
  model/file-type/physical-width/replay-window map, not a blanket gfx1100 PM4
  default. `HIPENGINE_SUBMISSION_TRANSPORT=hipgraph` is the whole-route rollback;
  explicit `pm4|aql` remains diagnostic. Re-profile capture break-even before
  lowering **160/64/96/80** at c1/c2/c4/c8 or admitting another model/quant,
  graph family, multi-step tape, width, or backend. Remove the environment
  rollback after one release window only if the scoped default remains exact and
  non-regressive; retain the package policy boundary and HIP fallback.
- Remove comparison-only `pm4_stateful*`/`pm4_timestamps` benchmark aliases and
  low-level conservative/global encoder options once they no longer provide
  useful rollback/bisection evidence. Retain direct AQL while it differentiates
  ROCr dispatch from architecture-specific PM4 or is needed for inner-kernel
  profiling that nested PM4 IBs cannot expose to `rocprofv3`.
- Never delete the native HIP graph implementation: it remains the required
  peer-architecture fallback and correctness oracle. Timestamp/quarantine
  controls may be removed only after ROCm/ROCm#6529 is reproduced/closed or they
  have conclusively failed to distinguish the lifecycle fault.

## Dense-H5120 exact unequal-Q4 pair context seam

- Added 2026-08-13 for the exact Q4T16 linear-attention QKV/gate bulk-prefill
  owner. The architecture/geometry-scoped runner context prevents the
  shape-general registry leaf from claiming unqualified models; rows below 512
  and all misses retain the two exact singleton projections. There is no
  environment selector, workspace, sidecar, or alternate payload.
- Removal trigger: after one release window with complete category, 512/1K/4K
  cross-board, and natural MTP safeguards stable, collapse the private boolean
  context into a quant-neutral model-policy object if more exact pair owners are
  admitted. Keep the separately registered singleton chain permanently as the
  required numerical/peer-backend fallback.

## Dense-H5120 bounded Q4/Q5/Q6 source-F16 rollback seam

- Added Q6-only constructor/context names with the 2026-08-13 bounded planar-Q6
  prefill owner; retained Q4 and Q5 extensions now share that owner and keep
  compatibility aliases plus `use_q6_f16_rocblas_prefill=False` as a whole-route
  exact rollback. The name is stale but the rollback remains useful while
  changed-arithmetic Q4/Q5/Q6 accumulates release evidence. Q5 is limited to
  the qualified dense-H5120 recurrent-output geometry at measured anchors and
  casts its dead K6,144 input in place. Exact Q4 pair-owned and Q5
  natural-octet-owned source-F16 production are selected through a generic
  quant/shape/inclusive-row
  variant policy; misses retain each scalar producer. An ordered pair-only
  policy can admit a second projection only when its already-qualified first
  peer owns the common activation, without exposing that shape to singleton or
  unrelated pair dispatch. A separate generic per-quant/shape row ceiling
  prevents nearest-anchor extrapolation through the
  known-losing Q4 full-attention-Q M2048/4K boundary while admitting M512/1K.
  Decode, verifier, peer backends, and all other misses retain exact T16.
- Removal trigger: after one release window with the complete category gate and
  512/1K/4K XTX/W7900 safeguards stable, rename the private runner fields/method
  to quant-neutral T16-F16 terminology and remove the Q6-named compatibility
  aliases if no external caller depends on them. Keep registry-key misses and
  exact T16 kernels as required numerical/peer-backend fallbacks.

## gfx1151 Qwen3.8 Q5 source-F16 rollback and workspace seam

- Added 2026-08-15 for the 48 sole-Q5T16 K6,144/N5,120 recurrent outputs.
  Keep exact Q5T16 WMMA registered and selectable through one release window;
  source-F16 changes arithmetic even though the octet producer is byte-exact,
  the CPU quality gate passes, and every retained natural token/acceptance
  trajectory is identical.
- The source-F16 context adds bounded activation/tile/output scratch, raising
  tracked peak 24.375 MiB at 512 and 65 MiB at 1K/4K. P8 must re-run owner-slot
  liveness and attempt safe dead-input/in-place or arena reuse. Remove any
  redundant standalone field after a proven alias, but do not trade the sole
  Q5T16 payload for persistent F16 weights.
- Q4/Q6 source-F16 maps are explicit empty denials after complete pp512 losses.
  Remove those declarations only when package capability defaults are made
  fail-closed per quant; do not interpret absence as inheriting gfx1100 policy.

## gfx1151 Qwen3.8 Q4_K_M residual-Q8_1x2 diagnostic

- Demoted for Q4_K_M on 2026-08-16 after production-numerics requalification.
  The changed route passes the 450-row strict-teacher envelope at max KL
  `0.0008333`, `99.778%` top-1, and exact three-run repeatability, but seven
  current same-session p512/d128 pairs measure only `0.998071x` the exact
  local32 owner with one candidate win. Q4_K_M therefore defaults this serial
  c1 shape to exact local32; no rollback flag or duplicate payload was added.
  Q4_K_S independently retains split-weight c1 and row-independent native
  rows2-4 because those owners pass its exact trajectory and performance gates.
- Removal trigger: after the Q4_K_M requalification artifact has served one
  release window, remove only its comparison harness if unused. Keep the shared
  Q8_1x2 producer and split-weight/rowtile registry keys while Q4_K_S owns them,
  keep the non-split native Q4_K_M consumer while its policy owns it, and keep
  the exact local32 fused/unfused chain permanently.

## gfx1151 Qwen3.8 compact-peer GDN rollback seam

- Added 2026-08-15 for Q4_K_M `(16K,48V,128,128)`. The compact-peer route
  right-sizes normalized Q/K to K-head cardinality and is bit-identical to the
  peer-wave oracle, but differs slightly from scalar-exact direct LDS32
  (maximum BF16 output 0.001953125, FP32 state 2.24e-8). Keep
  `chain_lds32_direct_nonvolatile` as the explicit exact selector through one
  release window and retain both primitive chains permanently as numerical
  fallbacks.
- The gfx1151 1,024-row recurrence cap is evidence, not a universal tuning
  constant: unchunked compact wins at 512/1K but loses 8.26% at 4K; state-carry
  chunking restores a 1.422x complete-chain win. Revisit 1K only during P4's
  full 128/256/512/1K chunk sweep, with peer-bit/state exactness, scalar-oracle
  bounds, 512/1K/4K full-engine gates, and the natural category suite. Do not
  remove or widen the cap from a single-prompt timing.

## gfx1151 Qwen3.8 role-qualified Q6 rollback seams

- Added 2026-08-15 after all-planar Q6 improved 512/1K/4K prefill by
  22.11-23.65% and removed 49,479,680 bytes, but regressed complete AR by
  0.14-0.19%. Actual-weight counterbalancing localized K=5,120/N=10,240 QKV c1
  at +8.72% wall with 0/11 planar wins. The backend capability therefore keeps
  one standard-T16 owner for those 24 tensors while down, narrow V, and root
  remain sole planar. There is no duplicate payload or environment selector.
- The exact planar down+residual alias is separately excluded on gfx1151. Its
  same-planar projection plus primitive BF16 add fallback is 1.1735x/1.1144x/
  1.1115x faster at rows2/3/4 with 15/15 wins and zero mismatches. Keep the
  fused registry primitive for gfx1100 and diagnostics; it is not a gfx1151
  production fallback chain.
- P4's exact 24-KiB-LDS/248-VGPR four-wave shared-weight prefill owners are
  limited to rows>=512 standard K5120/N10240 QKV and planar K17408/N5120
  FFN-down. The actual leaves improve 2.961-3.548x and 1.421-1.502x; complete
  incremental prefill improves 9.935-10.611% and 4.325-5.403%, respectively.
  The universal planar route remains rejected because narrow V misses its 1K
  leaf gate. There is no environment selector, sidecar, or workspace. Keep the
  16x16 standard and one-wave planar primitives registered for short rows,
  narrow V, root, peer backends, and all policy misses.
- Refactor trigger: after one release window, consolidate the two parallel
  `GGUF_Q6_*_PREFILL_SHARED4_{MIN_ROWS,SHAPES}` selectors/wrappers into one
  generic quant/shape/min-row policy if another Q6 shape or backend is admitted.
  Do not move the shape branch into engine/model dispatch; keep both exact
  primitive fallbacks permanently.
- Removal trigger: delete the QKV slot exclusion only after a byte-neutral
  planar c1 consumer wins an actual >64-MiB rotating-weight screen and complete
  512/1K/4K plus full natural AR/B1-B3 gates on gfx1151. Re-admit the fused
  residual alias only after it beats the unfused same-planar chain at every
  rows2-4 shape and remains non-regressive in the full category suite. Keep
  standard T16 and projection+add registered as required numerical fallbacks.

## gfx1151 dense-27B grouped-GQA split-attention rollback seam

- Added 2026-08-16 for dense H5120/L64/24Q/4KV/D256 BF16 decode from context
  4096. `HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX=0` restores the generic
  query-head split producer plus the same gated reducer. The package capability
  is geometry/context scoped; shorter contexts, peer backends, and policy misses
  retain the generic producer without extra resident or workspace bytes.
- Removal trigger: after one release window with exact 4K model, long-context,
  lifecycle, and peer-backend gates stable, remove the environment rollback for
  package-qualified grouped producers. Keep the generic producer/reducer chain
  permanently as the required numerical and unsupported-shape fallback.

## Dense-H5120 B4 exact eager-MTP seam

- Added 2026-08-15 so the public dense GGUF ladder can execute a real four-step
  NextN chain without claiming unproven graph ownership. The first root+four
  native-row fallback failed scalar-AR exactness on two Japanese prompts and is
  superseded. B4 is opt-in and now uses eager proposal plus serial-exact target
  rows; only tail cycles whose effective budget falls to B1-B3 use native
  graphs. The retained device graph ladder remains B1-B3 / rows 2-4. The clean
  full suite is exact but rejects B4 for performance: Qwen3.6 B3/B4 is
  **22.207/5.095 tok/s**, and Qwen3.8 is **19.808/4.586 tok/s**. B3 remains
  default; exact B4 is diagnostic-only.
- Removal trigger: admit B4 to both proposal and target device graphs only after
  rows=5 target state/KV/hidden commits pass reject/partial/full, category,
  heldout, long-context, GPU/CPU acceptance, and teardown gates. Promote B4 by
  model policy only if that same suite is non-regressive versus B3; otherwise
  retain exact eager B4 as an explicit diagnostic and do not widen the default.

## Dense Qwen3.6 bulk-prefill MTP native-verifier gap

- Added 2026-08-12 after public dense MTP was aligned with public AR prompt
  admission. Bulk target prefill plus shifted NextN catch-up is exact with the
  `serial_exact` transactional verifier for the natural control prompt, but the
  rows=3 dense native target verifier emits the `INT32_MAX` argmax sentinel for
  every row from the same valid bulk-prefilled state. Both its reusable N2 graph
  and eager native-row path fail; explicit initial-state copies and full-logit
  scoring do not repair it. Public dense MTP therefore uses bulk target prefill,
  shifted NextN catch-up, and `serial_exact` target verification. Keep the
  existing native verifier for its separately gated serial-prefill protocol.
- Removal trigger: make dense native rows exact after bulk prefill across the
  full natural/category/heldout suite and reject/partial/full transaction gate,
  then remeasure before changing the public route back to `native`.

## Qwen3.6 planar-Q6 Q8_1 scalar-dot diagnostic — closed

- The registered planar-Q6/Q8_1 projection/residual primitive remains as
  correctness and hardware evidence: its W7900 profile cuts the qualified
  rows1-4 family **46.900 -> 40.992 ms (-12.60%)**, and the full category gate
  improves B1/B2/B3 **1.875%/1.106%/2.315%**. It is not a production owner
  because true AR regresses **23.271 -> 21.821 tok/s (-6.235%)**.
- A cache-representative X8 c1 rescue wins its 56-weight component loop
  **6.269 -> 5.803 ms (1.080x)** on W7900, but adds **3,371,827,200 bytes / 3.140
  GiB**. The binding same-commit one-prompt gate improves B3 **65.896 -> 67.592
  tok/s (+2.573%)** while regressing AR **24.194 -> 22.533 tok/s (-6.867%)**.
- Runtime ownership and `HIPENGINE_GGUF_Q6_PLANAR_Q8_1` were removed after that
  rejection. The X8 allocation, helper, registry key, adapter candidate, tests,
  and `HIPENGINE_GGUF_Q6_X8_C1_SIDECAR` were removed together. Retain only the
  planar registered leaf, generic q8-input adapter, and focused diagnostics;
  reopen production only for a byte-neutral/sole-resident c1 representation
  that preserves transaction-consistent rows2-4 arithmetic and wins AR.
  Artifact: `benchmarks/results/2026-08-06-qwen36-27b-planar-q6-q8-1-runtime-rejected.json`.

## Qwen3.6 shared-cache verifier KV batch primitive

- Added 2026-08-05 after the exact rows2/4 primitive passed reversed-page CPU
  and scalar cache oracles, GPU1 component timing, cache-only tracing, and the
  W7900 B1-B3 transaction. Its production route is already removed: although
  writers improve **65.05%** and 336 dispatches disappear, target host and
  complete marked wall regress **1.405%/2.786%**.
- Retain only the gfx1100 registry wrapper/body and focused regression fixture
  while investigating a materially different graph/scheduling representation.
  Reopen runtime ownership only if a tracked-clean W7900 trace improves both
  target host and complete marked wall. If no such consumer is admitted by the
  next Qwen3.6 kernel-catalog cleanup, delete the registry key, wrapper/export,
  body, gfx1151 exclusion, and dedicated test together.

## Qwen3.6 alpha/beta plus snapshot-Conv composite primitive

- Added 2026-08-05 after the exact rows1-4 primitive passed scalar/CPU oracles,
  all-48-layer GPU1 component timing, cache-only tracing, and the W7900 B1-B3
  transaction. Its production route is removed: although the three-leaf family
  improves **41.09%** and 672 dispatches disappear, target host and complete
  marked wall regress **0.546%/1.629%**.
- Retain only the gfx1100 registry wrapper/body and direct regression fixture
  while investigating a materially different graph/scheduling representation.
  Reopen runtime ownership only if a tracked-clean W7900 trace improves both
  target host and complete marked wall. If no such consumer is admitted by the
  next Qwen3.6 kernel-catalog cleanup, delete the registry key, wrapper/export,
  body, gfx1151 exclusion, and dedicated test together.

## Qwen3.6 dense Q4T16 FFN dual residency — closed

- Added 2026-08-05 for the former exact dense-verifier FFN route, where 64 gate
  and 64 up tensors kept pack8 plus **6,595,543,040 bytes / 6.143 GiB** of T16
  decode sidecars.
- **Closed 2026-08-12:** gfx1100 now materializes one canonical
  `gguf_q4_k_t16_v1/tiles` payload for all 288 rank-2 Q4 tensors. Registered T16
  owners cover c1, rows 2-4, fused gate/up+SiLU, M512/M1024/M4096/tails, and the
  same-layout unfused chain. Live target+NextN census reports zero alternate or
  duplicate payload bytes; exact 512/1K/4K state, natural B1-B3 transactions,
  W7900 same-commit non-regression, and teardown gates pass. No runtime selector
  or package path retains the old Q4 pack8 shadow.
- Keep raw/pack8 kernels only as separately registered numerical/diagnostic
  fallbacks for other model/layout owners; do not reattach them as sidecars to
  this dense sole-T16 package without a new operation-complete residency gate.

## Maple CUDA sm_120a native-prefill follow-ups

- **Serial compatibility path closed 2026-08-08.** `MapleCudaRunner` now owns
  independent row-batched QKV/O, attention-prefill, router, stable int32 expert
  compaction, grouped selected-expert, and weighted-residual CUDA families. The
  public c1 generator calls the complete grouped native path; inherited
  `prefill()` remains only the explicit correctness/baseline oracle. The full
  18-prompt/90-position gate is KL 0 and state exact, 520/770 physical state is
  exact, and cache-only Nsight names every expected family. Evidence:
  `benchmarks/results/2026-08-08-cuda-sm120a-maple-native-prefill-retained.json`.
- The remaining compatibility owner is c1-only submit/poll. Replace it with a
  CUDA resident fixed-batch owner only after independent per-request span/KV
  arena, batched QK/KV+attention, sparse c2/c4/c8 admission/reclaim, row-reuse
  head, lifecycle, and full natural+heldout trajectory gates pass. Do not route
  CUDA through HIP builders or enable gfx11 batch metadata as a shortcut.

## Maple P4 public fixed-slot admission

- Added and retained 2026-08-08. `MapleResidentModelRunner` is now the sole
  public submit/poll owner: it shares one c1 checkpoint, prefills request-local
  fixed K/V slots, uses D0 for one active row, and uses D1 for c2/c4/c8. The
  complete public gate is **123.131/165.697/202.038/214.788 aggregate tok/s**
  with 15/15 repeated 18-prompt trajectory sets, sparse/staggered reclaim,
  physical-c8 singleton preservation, and lifecycle exact. Evidence:
  `benchmarks/results/2026-08-08-gfx1151-maple-p4-long-prefill-public-batch-retained.json`.
- **Closed at the final roadmap audit:** removed the duplicate
  `MapleContinuousBatcher` orchestration class and its private admission tests.
  `scripts/maple_batch_decode_bench.py` now drives `MapleBatchRunner.batch_step`
  directly to isolate D1 helper throughput; public sparse/staggered admission and
  reclaim remain owned and tested only by `MapleResidentModelRunner`. Fixed
  sparse K/V slots and registered numerical fallbacks remain intentional.

## SH16 private-c1 selective immutable-weight arena

- Added 2026-08-06 and promoted to the gfx1151 private-c1 default after exact
  four-transition plus 18-prompt gates and four-depth whole-GTT savings of
  **190/212/210/210 MiB**. One owner holds the exact **571 allocations <=16
  MiB** while all **161 larger owners / 95.965% of weight bytes**, all
  state/scratch, c>N requests, unsupported layouts, and denied routes keep
  dedicated behavior.
- The gfx1100 dense-H5120 geometry policy was added on 2026-08-12 and
  subsequently widened to the first complete 80-MiB inventory crossover. It
  packs 849 allocations into one owner, leaves only the 1,042,944,000-byte
  untied head
  dedicated, reduces physical weight owners **370 -> 2**, and cuts the XTX
  512/128 process peak **16.171 -> 16.095 GiB** with exact output. The complete
  W7900 three-shape plus natural-MTP safeguard now passes: prefill/decode and
  true-AR/B1-B3 are all faster than the same-commit dual-layout rollback, while
  peak delta falls **45.50-47.03%**.
- Task 24 adds an independent gfx1151 dense-H5120/Q4_K_S 80-MiB policy. With
  anonymous host embedding it packs 849 logical allocations into one arena,
  leaves only the 1,042,944,000-byte untied root dedicated, and exposes two
  physical weight owners. The retained combined 512/1K/4K rows are within the
  1% wall guard and below the lower llama process-GTT row at every shape.
- Keep `HIPENGINE_GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA=0` temporarily as a
  disable-only rollback/bisection seam. Remove the shared environment seam
  after task 26's clean gfx1100/gfx1151 cumulative closure confirms both dense
  H5120 policies and no supported private-c1 consumer requires rollback. The
  gfx1151 half of that confirmation is complete on 2026-08-17 at `20e5106da`
  (see the
  [`G6 closure artifact`](../benchmarks/results/2026-08-17-gfx1151-qwen38-27b-q4ks-g6-closure.json));
  the gfx1100 half remains pending on W7900 hardware, so the seam stays.
- Do not remove `DeviceMemoryArena`, the selective planner, ownership fields, or
  telemetry while this production owner is active. Do not revive the rejected
  SH15 whole-session state arena or compact-Q4 stack; task 24's dedicated 1K
  outer-chunk policy is the separately admitted Q4_K_S 4K owner.

## gfx1151 Qwen3.8 Q4 host embedding and bounded 4K scratch

- Added 2026-08-17 for private-c1 dense-H5120/Q4_K_S memory parity. Automatic
  Q4 placement uses a byte-for-byte anonymous copy of the 715,161,600-byte
  compressed embedding and HIP-registers that copy. Direct registration of the
  file-backed GGUF mmap changed complete-model trajectories and is forbidden on
  gfx1151. Q8_0 retains its existing CPU-copy route; shared/wider/device-forced
  sessions retain device ownership.
- `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING=device` is the temporary common rollback.
  Adjacent exact natural-suite measurement puts mapped host versus device at
  **+0.012% true AR / -0.027% B3** while removing exactly 715,161,600 tracked
  bytes. Remove the rollback only after task 26's clean shape/B3 refresh and one
  release window; keep explicit device fallback for shared and wider sessions.
- The Q4_K_S prefill policy keeps all bulk-prefill fields dedicated and is
  capacity-conditional (4K-class grows to the natural 4,096-row full-attention
  plateau to admit the Q5 source-F16 route, +2.95%; 8K and larger keep
  1,024-row chunks because 4,096-row chunks measured -2.3% slower at 8K). Do
  not replace the dedicated fields with owner-slot, single-arena, priority,
  hidden-buffer, or short-session aliases: stronger gates found final-logit or
  trajectory corruption. Revisit the 4,096-row plateau only if a later exact
  owner is non-regressive at 8K+ and keeps memory flat.

## SH15 gfx1151 private-c1 persistent allocation arena — closed

- Added 2026-08-06 as default-off
  `HIPENGINE_GGUF_PRIVATE_C1_SESSION_ARENA=1` after SH15-M1 measured a
  2-MiB HIP commit mechanism and exact metadata-derived plans reproduced all
  **732 weight + 189 state/scratch** requests at 512/4K/32K/64K.
- The screen was restricted by the gfx1151 package capability to an owned
  private-c1 runner. Shared runners, c>N, unsupported resident layouts, and
  either owner-allocation denial retained dedicated allocations.
- **Closed 2026-08-06:** exact state/lifecycle and measured **236/292-MiB**
  whole-GTT savings at 512/4K did not pass the wall contract. Prefill regressed
  **1.608%/1.809%**, beyond the <=1% loss limit. Removed the environment switch,
  package capability, generic arena primitive, weight/state planners, allocator
  routing, ownership telemetry, and candidate tests. SH14 production files are
  byte-identical to their pre-screen parent. Do not revive unchanged packing or
  stack compact Q4 onto this rejected owner.

## Laguna gfx1151 source-F16 non-temporal decode comparison seam — closed

- Added 2026-07-31 as a gfx1151 default plus
  `--compare-f16-nontemporal-decode` same-resident profiling seam after all
  four natural leaves improved **3.015-3.509%** and all seven exact
  p512/d128 pairs improved **22.792512 -> 22.855773 tok/s (+0.27755%)**.
- Keep the comparison switch through tracked-clean selector-unset publication
  and the next source-F16/kernel-span census, then remove the CLI switch.
  Retain constructor `false` and the cached-load registered composites as the
  exact peer-backend/rollback path.
- The standalone single/triple/quad non-temporal wrappers are leaf diagnostics,
  not production dispatch. After the census freezes the family attribution,
  collapse any wrappers no longer used by a retained benchmark or exact
  regression test while preserving the production composite variants.
- **Clean publication passed 2026-07-31:** selector-unset production advances
  **22.780604 -> 22.856155 tok/s (+0.33165%)**. Keep the comparison seam only
  through the pending source-F16/kernel-span census, then close it.
- **Closed 2026-07-31:** the post-promotion two-queue census confirms the
  expected source-F16 production symbols and lowers their inclusive family
  time **24.538908 -> 24.362893 ms/token (-0.717%)**. Removed the completed
  CLI/protocol comparison seam. Constructor `false`, cached-load registered
  composites, and exact leaf diagnostics remain the rollback and regression
  coverage.

## Laguna gfx1151 exact global local1024 selector

- Added 2026-07-31 as a default-off `LAGUNA_GLOBAL_LOCAL1024` capability and
  `--compare-global-local1024` profile seam after exact 513/576/639
  correctness, an all-positive 21x100 leaf, and cached resource tracing.
- If any same-resident p512/d128 pair regresses or changes state, remove the
  capability field, runtime branch, and comparison switch while retaining the
  exact registered primitive/leaf route. If every pair improves, promote the
  gfx1151 capability, publish tracked-clean selector-unset production, and
  remove the comparison switch after the next attention census. Keep local512
  as the exact non-dense/eviction/peer rollback.
- **Promotion gate passed 2026-07-31:** all seven exact resident pairs improve
  **22.358675 -> 22.383414 tok/s (+0.11065%)** with a
  **0.059368-ms/token** paired-median saving. The gfx1151 capability is
  promoted. Keep the comparison switch through tracked-clean publication and
  the post-promotion attention census, then remove it.
- **Closed 2026-07-31:** tracked-clean selector-unset production reaches
  **22.378602 tok/s**, and the 127-transition census cuts global
  **0.453932 -> 0.402996 ms/token (-11.221%)**. Remove the dedicated
  comparison switch; retain local512 as the exact non-dense,
  explicit-eviction, and peer-backend fallback.

## Laguna gfx1151 exact SWA local1024 selector

- Added 2026-07-31 as a default-off runtime capability and
  `--compare-swa-local1024` profile seam after the exact dense-ring candidate
  passed RED/GREEN, the real wrap oracle, a positive 21x100 leaf, and cached
  resource tracing.
- If any of seven same-resident p512/d128 pairs regresses or changes state,
  remove the capability field, profile switch, registered local1024 route,
  wrapper, export, and body specialization. If all seven improve, promote the
  gfx1151 capability, publish a tracked-clean selector-unset result, then
  remove the comparison switch after the next attention census. Keep
  local512 as the exact rollback.
- **Promotion gate passed 2026-07-31:** all seven exact resident pairs improve
  **22.273482 -> 22.356330 tok/s (+0.37195%)**. The gfx1151 capability is
  promoted.
- **Closed 2026-07-31:** tracked-clean selector-unset production reaches
  **22.335681 tok/s**, and the 127-transition census cuts SWA
  **0.893032 -> 0.721795 ms/token (-19.175%)**. Remove the dedicated
  comparison switch; retain local512 as the exact fallback for non-dense,
  explicit-eviction, and peer-backend cases.

## Laguna exact SWA producer-gate selector

- Added 2026-07-29 as a default-off gfx1151 capability plus session/profile
  comparison seam after the registered primitive passed byte-exact
  wrap/eviction, leaf, and cached resource gates.
- If matched resident p512/d128 decode is negative or indistinguishable, remove
  the runtime capability, cache field, session setter, and
  `--compare-swa-producer-gate` switch while retaining the exact registered
  primitive for diagnostics. If positive, promote the architecture capability,
  publish a clean selector-unset result, then remove the dedicated comparison
  switch after the next attention census. Keep producer-max without
  producer-gate as the exact rollback.
- **Promotion gate passed 2026-07-29:** all seven exact resident pairs improve
  **19.992650 -> 20.012052 tok/s (+0.097%)**. The gfx1151 architecture
  capability is promoted. Keep the comparison switch through the pending clean
  publication and post-promotion attention census, then remove it.
- **Clean publication passed:** selector-unset production is
  **20.003064 tok/s**, **+0.0835%** over the prior packet, with exact repeated
  state/lifecycle. The comparison switch now remains only for the pending
  post-promotion attention census and should be removed in that logical unit.
- **Closed 2026-07-29:** the census confirms SWA **-0.252%**, kernel sum
  **-0.043%**, unchanged resources, and the intended producer-gate symbol. The
  profile comparison flag, session setter, and setter-only test are removed;
  retain the architecture capability, fail-closed cache field, registered
  primitive, and producer-max rollback.

## Laguna exact SWA producer-maximum selector

- Added 2026-07-29 as a default-off gfx1151 capability plus session/profile
  comparison seam after the separately registered primitive passed byte-exact
  wrap/eviction, leaf, and cached resource gates.
- If matched resident p512/d128 decode is negative, remove the runtime
  capability, cache field, session setter, and profile comparison switch while
  retaining the exact registered primitive for diagnostics. If positive,
  promote the architecture capability, publish a clean selector-unset result,
  then remove the dedicated comparison switch after one later attention
  re-profile. Keep the prior mixed32/exp32 kernel as the exact rollback.
- **Promotion gate passed 2026-07-29:** all seven resident pairs improve
  **19.684442 -> 19.996117 tok/s (+1.583%)** with exact state/lifecycle.
  Keep the comparison switch through the pending clean publication and
  post-promotion attention census, then remove it.
- **Clean publication passed:** selector-unset production is
  **19.983610 tok/s** with exact state/lifecycle. The comparison switch now
  remains only for the pending post-promotion attention census and should be
  removed in that logical unit.
- **Closed 2026-07-29:** the census confirms SWA **-21.36%** and kernel sum
  **-1.46%**. The profile comparison flag and session setter are removed in
  the census unit; retain only the architecture capability, current owner,
  and registered mixed32/exp32 rollback.
## Laguna H7U parallel MoE compactor

- Cleanup completed 2026-08-02. Exact bounded fixed, source-default
  512/1K/4K, five-request wall, and selected-region topology gates promoted
  `LAGUNA_MOE_GROUP_COMPACT_MODE="parallel"` on gfx1100. The temporary
  `LAGUNA_MOE_GROUP_COMPACT_H7U_MODE` capability was deleted atomically; there
  is one live package owner and no duplicate runtime selector or allocation.
- Keep the explicit registered serial scheduler and
  `set_group_compact_mode("serial")` only as the required exact rollback and
  correctness oracle. Remove no fallback while it remains useful for backend
  qualification or regression bisection; any future collapse must preserve an
  unfused/numerically equivalent path under the project fallback policy.

## Laguna H7E bounded default-off IQ3 residual-D4 owner

- Cleanup completed 2026-08-02. The temporary exact-M512 owner passed narrow
  state/trace gates but failed the binding 18-prompt/576-step lane at max KL
  **5.630805** and general-Japanese top-1 **89.844%**. Promotion timing was
  skipped.
- Removed the residual-route resolver/plan fields, tile128 export, optional
  eager-library fields, qualified/live maps, and runtime test. H6T/IQ4 remain
  source and required fallbacks. Retain only the separately registered H7E leaf
  and rejection evidence; reopen only with a materially different repair or
  representation and a fresh RED/complete-quality contract.

## Laguna H7Y bounded default-off lane-major SWA cache owner

- Added 2026-08-03 for writer-inclusive qualification of the admitted H7Y
  standalone leaf. The temporary
  `LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS` map owns exact
  **72 MiB / 72 allocations** of SWA K/V mirrors and selects one fused
  natural+lane-major rows writer without adding dispatches.
- Bounded qualification is complete: full state, exact **2,286-dispatch**
  topology, writer-inclusive fixed C4096/M512, and clean 512/1K/4K medians pass.
  Source promotion is separately rejected: exact selector-unset fixed M512
  regresses **436.403→436.275 tok/s (-0.0294%)**. H6Z/H6W remains active and the
  72-MiB owner stays default-off as an explicit diagnostic/rollback candidate.
  Remove this candidate capability, mirror allocation route, fused writer, and
  focused runtime test when no further materially different source mechanism
  needs the layout; never allocate mirrors on the active H6Z/H6W path.

## Laguna H8A resident global-Q5 F32 cache production cleanup

- Added 2026-08-03 for exact owner-inclusive qualification. The gfx1100 source
  `LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE` capability owns one immutable
  **24-allocation / 1,811,939,328-byte** map for all 12 global `attn_q` and
  `attn_output` raw-Q5 tensors. Explicit child sharing reuses the same owner;
  every disabled, unsupported, unshared, allocation-failed, or non-target route
  keeps the registered transient producer + H7G fallback.
- Bounded and source qualification pass: all 24 complete planes, natural-M512
  state, exact **24 setup / 0 request producer / 2,262-dispatch** topology,
  fixed C4096/M512, and clean 512/1K/4K medians are positive and exact. Clean
  committed production reaches **440.353 tok/s** with a **1,151.215-ms**
  representative kernel sum, one queue, and zero compiler activity.
- The clean checkpoint cleanup is complete: the duplicate `_SUPPORTED`
  capability and explicit positive selector are removed. Source ownership is
  the sole positive route; `use_q5_f32_resident_global_cache=False` remains the
  explicit transient-H7G rollback. Remove that false opt-out after one later Q5
  family checkpoint if no rollback is needed.

## Laguna H8B scoped activation-pack reuse production cleanup

- Added 2026-08-03 for complete all-class qualification. The gfx1100 source
  `LAGUNA_ACTIVATION_PACK_REUSE` capability owns one generic one-entry
  context-local cache valid only inside attention Q/K/V, dense gate/up, or
  shared-expert gate/up scopes. It publishes the exact
  pointer/geometry/stream key only after a successful pack.
- Bounded qualification passes: complete state and exact **330→223 pack /
  2,262→2,155 dispatch** topology are preserved, fixed C4096/M512 improves
  **+0.116%**, and clean 512/1K/4K improves **+0.148%/+0.175%/+0.152%**.
  Source promotion and clean committed production pass at **440.893 tok/s /
  2,155 dispatches**.
- The clean checkpoint cleanup is complete: the duplicate `_SUPPORTED`
  capability and explicit positive selector are removed. Source ownership is
  the sole positive route; `use_activation_pack_reuse=False` remains the
  explicit complete-pack H8A rollback. Remove that false opt-out after one
  later Q5/IQ-down checkpoint if no rollback is needed.

## Laguna exact-ordered F32 Q5/Q6 production selector

- Added 2026-07-30 for WPF-H5D and advanced through H5I. The gfx1100 package
  owns all eight exact-M512 Q5 roles plus four exact-Q6 roles through one
  **150,994,944-byte** projection-local F32 weight plane; row/key/backend/shape/
  capacity misses retain raw coltile. H5I adds no allocation or sidecar. Complete
  package-default M512 state is byte-exact at KL 0, and clean selector-unset
  512/1K/4K publishes **191.713/178.080/134.411 tok/s**, improving H5G
  **+1.762%/+1.736%/+1.256%**. This is retained production, not a rejected
  experiment.
- Cleanup completed immediately after the H5E checkpoint: remove the public
  `LagunaGGUFResidentSession(use_q5_f32_ordered_prefill=...)` boolean, rename the
  resolver as a private package-only helper, and remove it from `__all__`.
  Retain the quant-keyed package role maps, context-local bounded owner,
  registered unfused producer/consumer chains, and raw-coltile miss fallbacks
  required by policy. Remove the legacy Q5-named internal session/scratch fields
  after one clean checkpoint confirms no downstream benchmark harness imports
  them; this is naming cleanup only and must not create a second owner.
  Benchmark harnesses use scoped package-policy mutation for exact A/B control;
  production has no per-session selector surface.
- WPF-H5W's eager-alias cleanup trigger is closed. Complete state is KL0,
  cached integration records **142 H5W + one H5I + three raw** consumers at
  unchanged topology, default-off 512/1K/4K is
  **+1.830%/+1.492%/+1.061%**, and selector-unset publication confirms
  **+1.785%/+1.532%/+1.100%** with 3/3 wins each. Retain exactly the three Q6
  eager aliases and promoted role entries; they add no selector or allocation.
  Keep H5I F32-N72 plus raw exact fallbacks. Do not add a public/session
  selector or duplicate owner.
- WPF-H5X's eager-alias cleanup trigger is closed. Complete state is KL0, four
  counter-rotated cached requests record **151 H5X + 37 H5L + 47 H5G**
  consumers at unchanged topology, default-off 512/1K/4K is
  **+0.439%/+0.468%/+0.518%**, and selector-unset publication confirms
  **+0.531%/+0.310%/+0.327%**, 3/3 wins each. Retain exactly the four eager
  aliases and H5X/H5L rollback primitives beneath H5Y. N48/N72 remain H5G and
  all Q6 routes remain unchanged. Do not add a public/session selector or
  duplicate owner.
- WPF-H5Y's eager/workspace cleanup trigger is closed. Complete state is KL0;
  cached integration records **188 packs + 235 weight producers + 188 H5Y +
  47 H5G**, default-off 512/1K/4K is **+10.939%/+9.051%/+5.920%**, and
  selector-unset publication confirms **+10.862%/+8.969%/+5.829%**, 3/3 wins
  each. Retain exactly six eager aliases and one combined **161,120,256-byte**
  ordered scratch allocation containing the F32 weight and BF16 activation
  planes. Keep H5X/H5L/H5G registered fallbacks and all Q6 routes unchanged;
  do not add a public/session selector, persistent sidecar, or duplicate owner.
- `GGUF_Q6_F32_ORDERED_PREFILL_POLICY` selects the H6U DPP-wave reduction
  variants. `GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY` selects distinct non-DPP
  variants and remains an exact rollback, not a duplicate of the source map.
  Do not replace its tests with production-map tests: that would lose rollback
  coverage. Keep H5W and H5I registered exact rollbacks and preserve the combined
  workspace/scratch owner; do not add a public/session selector, allocation,
  sidecar, or second ABI.

## Laguna raw-Q5/Q6 prefill rowbatch selector

- Added 2026-07-28 as the default-off WPF-1 runtime admission seam, promoted
  rowbatch8 after its exact gates, then promoted rowbatch32 after WPF-1W clean
  paired 512/1K gains of **7.783%/7.249%** and selector-unset publication at
  **85.481/79.555 tok/s**. `LagunaGGUFResidentSession(...,
  raw_k_prefill_rowbatch=0|4|8|16|32)` scopes fixed-grid-Y Q5/Q6 row reuse to one
  bulk-prefill execution owner; zero is scalar rollback. gfx1151 excludes all
  sixteen rowbatch plus eight output-column W7900 keys and remains zero.
- WPF-C1 confirms RB32 remains the full-model winner at retained M256 and traces
  its Y8 grid. WPF-2 then selects M512 without changing the RB32 owner; clean
  publication reaches **99.230/91.559 tok/s**. Keep RB8 as the preceding
  production control and zero as scalar rollback. The publication trigger is
  now satisfied: remove rowbatch4/16 and their explicit CLI/registry surface in
  a separate cleanup unit because no retained end-to-end policy selects either.
  Preserve scalar registered fallback for unsupported shapes/backends.
- WPF-1T first promoted exact `(4,8)` through the gfx1100 package for full
  RB32 slabs with output width divisible by four. The no-override M512 state
  gate is KL0 across all 48 boundaries/KV spans, clean 512/1K publishes
  **169.253/159.229 tok/s**, and restored 4K passes at **123.084 tok/s**.
  The bounded `(2,16)` decision is now complete: exactly four
  `(quant, output, K, N)` keys save **36.773 ms (2.011%)** from the 381-call
  Q5/Q6 family. The frozen seven-pair gate passes **+0.545%/+0.459%** at
  512/1K; a package-path repeat remains exact and positive at
  **+0.382%/+0.242%** but misses the repeated 1K `>0.3%` magnitude threshold.
  Keep `(2,16)` only for those four measured keys and `(4,8)` everywhere else;
  pooled 14-pair diagnostics are **+0.441%/+0.366%** with **14/14** and
  **12/14** wins. The benchmark-only Laguna variant constructor/setter surface
  is removed; explicit rowbatch selection still restores RB32. The internal
  context-local `coltile` owner remains necessary to scope production dispatch.

## Laguna exact grouped-IQ prefill selector

- Added 2026-07-29. WPF-2 first exported paired `grouped_exact`, preserving
  route-major pair16/local64 gate/up before registered expert-major IQ3/IQ4
  down and exact weighted restore. WPF-2b now exports
  `LAGUNA_SELECTED_GATE_UP_MODE="grouped_pair16"` with
  `LAGUNA_SELECTED_DOWN_MODE="grouped_exact"`. Explicit `grouped_exact` and
  paired `direct` remain rollbacks; unsupported keys fail closed to the exact
  route-major/grouped chain. gfx1151 keeps its independently measured MMQ
  modes.
- WPF-2b promotes the distinct local64/pair16 rowbatch8 primitive after exact
  all-46-layer, complete-state KL0, clean 512/1K, and cached-trace gates. Clean
  publication improves **99.230/91.559 -> 118.705/107.804 tok/s** and tracing
  cuts gate/up **62.549%/62.850%** at unchanged dispatch count. Bulk gfx1100
  now defaults gate/up to `grouped_pair16` plus `grouped_exact` down; c=1,
  unsupported keys, explicit `grouped_exact`, and paired `direct` remain exact
  fallbacks.
- WPF-H6L's pair16 same-ABI capability cleanup trigger is closed. Complete
  M512 is KL0/byte-exact; four cached requests substitute exact **46 rowbatch8
  -> 46 H6L** at unchanged **2,192** dispatches and cut IQ2/request/span
  **18.064%/5.153%/5.532%**. Selector-unset 512/1K/4K confirms
  **+5.666%/+4.468%/+3.246%**, 3/3 exact wins each, while fixed natural
  C4096/M512 wins **5/5** at **360.451 -> 381.893 tok/s (+5.949%)**. Keep H6L
  source plus rowbatch8 same-ABI rollback through one release checkpoint; then
  collapse the capability indirection to the source map if no downstream
  bisection uses it. Add no adapter, allocation, workspace, sidecar, or public
  selector. See the
  [H6L production artifact](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq2-pair16-rowbatch16-production.json).
- The cleanup trigger is closed. Cleanup removed the unowned Laguna rowbatch8
  and fused-SiLU local256/group8 gate/up variants, IQ3 down rowbatch4, and
  losing pair16 rowbatch4 wrappers, keys, HIP instantiations, and focused tests.
  It preserves pair16 rowbatch8, grouped IQ3 single/down rowbatch8, IQ4 auto,
  shared compaction metadata, c=1 route-major arithmetic, and direct exact
  fallback. The initially broader deletion premise was unsafe: IQ2/IQ3
  base/rowbatch4/adaptive/auto and IQ4 dual independently own Qwen3.5 GGUF's
  exact default-on grouped-prefill route and remain covered. P6 and WPF-1R
  remain closed.
- WPF-H5J's package-only `LAGUNA_GROUPED_IQ_DOWN_VARIANTS` cleanup trigger is
  closed. Complete M512 state is KL0/byte-exact, integrated tracing selects all
  **45+2** expected calls, and clean selector-unset 512/1K/4K publishes
  **196.103/181.859/137.169 tok/s (+2.290%/+2.122%/+2.052%)** over H5I. Keep
  the two-entry gfx1100 map and its bounded resolver: it replaces each
  registered IQ3/IQ4 key independently only for exact K1024/N3072, while map,
  shape, registration, and backend misses retain rowbatch8/IQ4-auto. gfx1151
  remains empty/fail-closed. There is no public/session selector, new
  allocation, workspace, or sidecar to remove.
- WPF-H5Q's `LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS` cleanup trigger is closed.
  Complete M512 state is KL0, tracing selects all **45** P64 IQ3 calls, and
  default-off 512/1K/4K is **+0.702%/+0.278%/+0.370%**. Selector-unset
  publication confirms **+0.663%/+0.355%/+0.267%**, 3/3 paired wins at every
  length. Keep H5Q's gfx1100 variant-to-`grouped_raw_iq_active_experts` entry
  and ABI-table adapter: they pass the existing device active-expert list/count
  with no allocation, producer, sidecar, or public selector. Map, ABI, shape,
  registration, and backend misses retain the generic exact route; gfx1151
  stays empty/fail-closed.
- WPF-H5Z's second variant-to-active-expert ABI entry cleanup trigger is closed.
  Complete M512 state is KL0/byte-exact; four paired traces select exact **45
  H5Q or 45 H5Z** calls at unchanged **2,050** dispatches and cut IQ3/request/
  span **2.342%/1.312%/1.539%**. Default-off 512/1K/4K is
  **+1.801%/+1.334%/+0.900%** and selector-unset confirms
  **+1.819%/+1.452%/+0.872%**, 3/3 wins each. Keep both ABI entries and H5Q as
  registered rollback beneath H5Z production. Do not add another adapter,
  allocation, workspace, sidecar, or public selector.
- WPF-H6D's third active-expert ABI capability cleanup trigger is closed.
  Complete M512 is KL0/byte-exact; four paired traces substitute exact **45 H5Z
  -> 45 H6D** calls at unchanged **2,050** dispatches and cut IQ3/request/span
  **2.564%/0.251%/0.861%**. Selector-unset 512/1K/4K confirms
  **+1.207%/+0.657%/+0.492%**, 3/3 wins each, while fixed C4096/M512 wins **5/5**
  at **329.327 -> 332.308 tok/s (+0.905%)**. Keep H5Z and H5Q as registered
  rollback beneath H6D production and keep all three ABI entries through one
  release checkpoint. Do not add another adapter, allocation, workspace,
  sidecar, or public selector. See the
  [H6D production artifact](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-row-interleaved-vopd-production.json).
- WPF-H6F's fourth active-expert ABI capability cleanup trigger is closed and
  superseded by H6I source production. Its original complete-state, trace, and
  selector-unset gates remain valid rollback evidence. Keep H6F as the immediate
  registered rollback beneath H6I through one release checkpoint; do not remove
  H6D/H5Z/H5Q entries before that checkpoint because they remain the deeper
  exact bisection chain. Add no adapter, allocation, workspace, sidecar, or
  public selector. See the
  [H6F production artifact](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-paired-output-reduction-production.json).
- WPF-H6I's fifth active-expert ABI capability cleanup trigger is closed.
  Complete M512 is KL0/byte-exact; four cached requests substitute exact **45
  H6F -> 45 H6I** at unchanged **2,192** dispatches and cut IQ3/request/span
  **9.559%/1.906%/2.200%**. Selector-unset 512/1K/4K confirms
  **+2.304%/+1.650%/+0.719%**, 3/3 exact wins each, while fixed C4096/M512 wins
  **5/5** at **352.966 -> 360.154 tok/s (+2.036%)**. Keep H6F as immediate
  rollback through one release checkpoint, then collapse superseded H5Q/H5Z/
  H6D capabilities only if no downstream bisection uses them. Retain the
  registered H6I source plus H6F rollback; add no adapter, allocation,
  workspace, sidecar, or public selector. See the
  [H6I production artifact](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-triple-output-reduction-production.json).
- WPF-H6P adds a sixth same-ABI active-expert capability as a bounded default-
  off owner. Complete M512 and exact four-request **45 H6I -> 45 H6P** topology
  pass; 512/1K/4K gains **+0.764%/+0.416%/+0.242%** and fixed C4096/M512 gains
  **+0.141%**. Remove the H6P ABI entry if the separate source-default gate
  fails. If H6P promotes, retain H6I as immediate rollback through one release
  checkpoint, then collapse older H5Q/H5Z/H6D/H6F entries only when no active
  bisection uses them. Add no adapter, allocation, workspace, sidecar, env flag,
  or public selector. See the
  [H6P candidate/runtime artifact](../benchmarks/results/2026-08-01-gfx1100-laguna-q2-xl-iq3-staged-wave-publication-candidate.json).
- WPF-H6C's grouped gate/up role-map cleanup trigger is closed. Complete M512
  state is KL0/byte-exact, paired tracing records exact **46 IQ2 + one H6C** at
  unchanged **2,050** dispatches, default-off 512/1K/4K is
  **+1.148%/+0.796%/+0.560%**, and selector-unset publication confirms
  **+1.326%/+0.897%/+0.490%**, 3/3 wins each. Fixed natural-M512/C4096 also wins
  **5/5** at **325.211 -> 328.863 tok/s (+1.123%)**. Keep the one-entry
  `LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS` map and qualified
  `grouped_raw_iq_dual_silu` ABI through one release checkpoint; then consider
  folding the stable role into the grouped gate/up plan while preserving the
  empty-map route-major rollback, bounded fail-closed checks, and no
  allocation/workspace/sidecar/public selector. See the
  [H6C production artifact](../benchmarks/results/2026-07-31-gfx1100-laguna-q2-xl-iq3-gate-up-expert-major-production.json).

## Laguna exact qrow4 SWA prefill selector

- Added 2026-07-29. WPF-3 registers the direct exact qrow4 body plus
  `swa_context_rows_qrow4_m128_c256_exact_spans`, which keeps wave32 below its
  measured rows/context crossover. The gfx1100 package now exports the
  qualified policy after no-override deep-state KL0 and paired 512/1K gains of
  **11.230%/16.774%**. Explicit local128, wave32, qrow2, qrow4, and online
  variants remain available for rollback and the separate arithmetic screen.
- Clean publication is complete at **131.919/125.960 tok/s** with cached SWA
  reductions of **55.411%/59.449%**. No benchmark-only comparison plumbing is
  committed. The separate complete online-qrow4 decision improves suite prefill
  **0.995%** but is rejected at maximum KL **0.394600** despite **564/576**
  top-1, deterministic repeats, and passing Poolside/lifecycle checks. Preserve
  local128, qualified wave32, and exact qrow4-C256 as gfx1100 fallback/default;
  do not remove online qrow2/qrow4 registrations because gfx1151 independently
  owns them. Do not reopen changed-association gfx1100 SWA without a materially
  different arithmetic premise and a new complete quality gate.
- WPF-H5M's package-only `LAGUNA_SWA_PREFILL_ROLE_VARIANTS` cleanup trigger is
  closed. Complete M512 state is KL0/byte-exact, integrated tracing selects all
  **72** expected calls, and clean package-default 512/1K/4K publishes
  **238.565/218.182/158.138 tok/s (+0.256%/+0.135%/+0.490% over H5L)**. Keep the
  one-entry gfx1100 map and bounded resolver: it replaces only the retained
  package-default auto route at M128, window 512, 72 SWA heads, and start >=256;
  explicit rollback plus shape, registration, and backend misses retain WPF-3.
  gfx1151 remains empty/fail-closed. There is no public/session selector, new
  allocation, workspace, or sidecar to remove. After one release checkpoint,
  consider folding the stable role into the existing auto wrapper and deleting
  only the map/cache-field indirection while preserving exact fallbacks.
- WPF-H5N's temporary default-off role-policy extension is removed. Complete
  M512 state and integrated tracing are exact/positive, but the clean gate is
  negative at 4K and a seven-repeat adjudication confirms **7/7** H5N samples
  below H5M (**158.152 -> 157.832 tok/s, -0.202%**). Retain the separately
  registered standalone H5N leaf and its M512 evidence; do not re-add it to the
  role-candidate policy or package map without a distinct long-context premise
  and a fresh 512/1K/4K gate. The older H5M map/cache-field cleanup remains
  independently eligible after its release checkpoint.
- WPF-H5R's `LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS` cleanup trigger is closed.
  Complete M512 state is KL0, corrected one-queue tracing records all **144**
  write->H5R pairs, and selector-unset 512/1K/4K is
  **+11.340%/+4.848%/+0.746%** with 3/3 wins each. Keep the one gfx1100 role
  entry, preappend capability, and generic fail-closed cache gate through one
  release checkpoint, then consider folding the stable role into package auto
  policy while preserving H5M/wave32/global/partial/wrapped/verifier/explicit/
  registration/backend misses. No allocation, workspace, sidecar, or public
  selector was added. The earlier uncapped speed packet is superseded.
- WPF-H5U's temporary runtime-policy seam is removed. Complete M512 state is
  KL0/byte-exact, integrated tracing records **48 H5U + 144 H5R** physical
  preappend pairs at unchanged topology, and active C4096 routes improve.
  Source-default publication nevertheless misses the binding role-ineligible
  1K non-regression gate: the balanced eight-pair adjudication is
  **230.181231 -> 230.175317 tok/s (-0.00257%, 2/8 wins)**. Remove the global
  package map, role candidate/resolver branch, allocator argument, runner
  bookkeeping, and runtime-policy test; retain only the separately admitted
  leaf. Reopen ownership only with a materially different package-scoping
  premise and a fresh complete 512/1K/4K gate. No allocation, workspace,
  sidecar, or public selector remains
  ([runtime rejection](../benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-global-preappend-cached-source-runtime-rejected.json)).
- WPF-H6A's dense-initial role-map cleanup trigger is closed. Complete M512
  state is KL0/byte-exact, paired tracing records exact **48 H6A global + 144
  H6A SWA** calls at unchanged **2,050** dispatches, default-off 512/1K/4K is
  **+1.713%/+0.677%/+0.326%**, and selector-unset publication confirms
  **+1.831%/+0.550%/+0.359%**, 3/3 wins each. Keep the two-entry map, dense role
  parser, independent allocator/runner selector bookkeeping, H5R/H5U registered
  rollbacks, and fail-closed policy through one release checkpoint; then
  consider folding only the stable owner into package auto policy. No
  allocation, workspace, sidecar, or public selector was added.

## Laguna P6 IQ2 MMQ32 diagnostic

- The pre-existing explicit four-axis primitive expands raw IQ2 to signed-byte
  fragments and consumes a caller-owned D4-Q8_1 pack. Actual M512 repricing
  improves the 46-layer gate/up leaf sum **1297.436 -> 388.901 ms (3.336x)**
  and a temporary session reaches **122.135/110.761 tok/s** at 512/1K.
- Runtime admission is closed: complete quality reaches max KL **0.683239** at
  **565/576** top-1, while P6-specific repair density reaches **85.946%**
  uncertain coordinates and **99.496%** touched active output rows. No runtime
  owner, selector, or repair queue was added. Keep the primitive only as an
  explicit arithmetic ceiling/negative control; do not add another P2/P6
  changed-arithmetic sibling. This result does not close WPF-1R's separate raw
  Q5/Q6 density screen.

## Laguna long-context F32 hipBLASLt rollback routes

- Added 2026-07-27. The first capacity-sized full-score owner improved
  mandatory 128K **22.088%**, but widened the complete BF16 K/V prefix and
  materialized an F32 `[48,128,C]` score tile, costing **4.298 GB** scratch.
- The retained successor uses exact online softmax state across 4K key blocks.
  It improves the full-score owner another **12.521%** at mandatory 128K and
  cuts scratch **96.655%** to **143,753,216 bytes**. gfx1151 now selects
  `LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT`; the generic `KVLiveSpans` chain
  remains the required fallback.
- Remove the capacity-sized full-score owner, its
  `LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT` capability/session setter, and
  ceiling-only algorithm-override API after one later long-context publication
  confirms the block route is sufficient for rollback. Keep the bounded owner,
  its explicit rollback while SWA/query-chunk work is active, and the generic
  numerical fallback. Retire the dedicated ceiling harness after the
  block-size/algorithm policy no longer needs bisection.

## Laguna rolling-SWA hipBLASLt production route

- Added and promoted 2026-07-28 after exact wrap-oracle plus
  4K/16K/64K/128K gates. The fixed owner gathers a 639-key union and owns
  **33,554,432 bytes** scratch; decode, verifier, partial, evicted, and
  nonconsecutive paths retain the registered generic fallback.
- `LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT` and the session setter remain useful
  while LC-3 widens attention query chunks. Collapse the explicit selector
  after the wider-query route either absorbs this owner or proves the M128
  route remains independently optimal. Do not remove the span-aware fallback.

## Laguna global M2048 attention-query route

- Added and promoted 2026-07-28 after exhaustive M128..M2048 global/SWA
  screens and positive 4K/16K/64K/128K complete-model gates. The production
  capability `LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS=2048` applies only to
  complete global matrix chunks; SWA and partial tails remain M128.
- Keep the explicit constructor/profile override through LC-4/LC-5 so dense
  addressing and larger matrix chunks can be bisected independently. Collapse
  it into the final chunk policy after those stages choose their retained
  geometry. Never remove the generic span-aware fallback.

## Laguna dense-contiguous global-cache widen

- Added and promoted on gfx1151 in LC-4 after the exact 4K block sub-window
  improved **0.250249 -> 0.234780 ms (-6.181%)** and the complete-model gates
  remained neutral with exact state through 128K.
- Keep `prefill_dense_contiguous_cache`, the profile comparison switch, and
  the generic span-aware block kernel through LC-5/LC-6 for rollback and
  non-dense semantics. After one release plus the final six-shape sweep,
  collapse the selector if no regression appears. Never remove or weaken the
  complete `KVLiveSpans` fallback for continuation, verifier, eviction, SWA,
  decode, or unmeasured backends.

## Laguna MoE shared/routed branch-concurrency candidate

- Added 2026-07-26 as an exact, default-off session and profile-harness
  candidate. For rows greater than one, the always-on shared expert runs on a
  nonblocking secondary stream while router selection and routed experts run
  on the caller stream. Two timing-disabled events preserve the producer and
  final-combine dependencies. Decode and the established single-stream route
  remain unchanged.
- The production-style Q4 GPU composition fixture is BF16-byte identical to
  the sequential route. The performance gate must use
  `GPU_MAX_HW_QUEUES=2` for both arms so queue count is not a confound, then
  prove an actual overlap in a cached trace rather than inferring it from wall
  time.
- If the clean queue-matched A/B is negative, remove the constructor option,
  profile flag, secondary-stream lifecycle, event plumbing, helper split, and
  concurrency fixture immediately. If it is positive, promote through an
  architecture capability only after complete-state/quality/lifecycle and
  traced-overlap gates; then remove the explicit positive selector after clean
  publication while retaining the required sequential fallback.
- The candidate trigger passes: seven queue-matched complete-state pairs move
  **560.837 -> 567.577 tok/s (+1.202%, 7/7 wins)**, clean default-off
  512/1K/4K reaches **565.457/525.733/443.027 tok/s**, and tracing proves
  **100.390/101.241 ms (99.16%)** of secondary kernel time overlaps caller
  kernels while kernel span falls **12.727 ms**. Promote through the gfx1151
  capability and two-queue process policy; retain the explicit selector only
  through clean selector-unset publication and one later MoE checkpoint.
- Promotion is complete in code: gfx1151 defaults the capability when at least
  two process queues are available, the process default is now two, an
  explicit one-queue override automatically selects the sequential fallback,
  and constructor/CLI rollback remains. Clean selector-unset publication and
  the follow-up MoE checkpoints are now complete; defer selector cleanup only
  through the lower-priority secondary-stream screen.
- Initial eager-concurrency selector-unset publication passed at
  **565.447/526.711/443.444 tok/s**. The production trace observes two
  queues/two streams, overlaps **76.883/77.763 ms (98.87%)** of secondary
  kernel time, and cuts the prior kernel span **11.265 ms**. Keep the explicit
  constructor/CLI and one-queue rollbacks through the scheduling cleanup while
  retaining the automatic sequential safety fallback.
- The delayed-launch checkpoint is closed. `before_down` preserved complete
  state but regressed **566.394 -> 565.011 tok/s (-0.244%, 2/7 wins)** and
  introduced a **535.465 tok/s** low tail. Its row-launcher, session, CLI, and
  test selectors are removed; no temporary phase-selection debt remains.
- The follow-up after-router checkpoint temporarily adds
  `moe_shared_after_router` at the row launcher, session, and long-profile
  CLI. It records the secondary dependency after router selection while
  retaining gate/up-plus-down overlap. The gate passes at **+0.073%, 5/7
  wins**, and tracing verifies a **0.310-ms** span reduction. Promote through
  a gfx1151 capability, then collapse redundant positive selection after clean
  selector-unset publication.
- Promotion now resolves `LAGUNA_MOE_SHARED_AFTER_ROUTER` from the gfx1151
  package; explicit constructor/CLI false remains rollback. Remove redundant
  public positive selection after clean selector-unset 512/1K/4K publication,
  while retaining the false rollback through the next MoE scheduling audit.
- Clean selector-unset publication passes at
  **566.839/527.381/444.447 tok/s**. Collapse redundant positive selection
  during the next MoE scheduling cleanup; retain the explicit false rollback
  while lower-priority secondary-stream scheduling is screened.
- The lower-priority checkpoint adds general HIP priority-range/query and
  priority-stream creation helpers plus a temporary
  `moe_shared_low_priority` session/CLI selector. gfx1151 reports
  `(least=+1, greatest=-1)`. Remove the Laguna selector if the complete-state
  gate fails; the general HIP helpers may remain because they are exact,
  tested runtime primitives.
- The checkpoint passes **+0.494%, 6/7 wins** and tracing cuts kernel span
  **7.255 ms**. Promote the least-priority selection through a gfx1151
  capability. After clean selector-unset publication, remove redundant
  positive CLI/session selection while retaining explicit priority-0 rollback
  through one later scheduling checkpoint.
- Promotion now resolves `LAGUNA_MOE_SHARED_LOW_PRIORITY` from the gfx1151
  package. `--no-moe-shared-low-priority` and an explicit constructor false
  remain exact priority-0 rollback. Clean selector-unset 512/1K/4K publication
  is the remaining promotion gate.
- Clean selector-unset publication passes at
  **568.849/527.113/444.508 tok/s**; 1K/4K are flat within
  **-0.051%/+0.014%**. The eager-priority1 checkpoint is now closed at
  **-0.198%, 1/7 wins**. Collapse redundant positive selection during the next
  MoE scheduling cleanup while retaining explicit priority-0 and sequential
  rollbacks.

## Laguna selected one-plane FP32-scale MMQ modes

- Added 2026-07-25 as explicit session-local LAP-3/LAP-4 candidates. The
  original `mmq64x32_d4_f32` route used one FP32 scale per 32 activations for
  selected Q4 gate/up and Q4/Q6 down. It crossed 350 tok/s but failed the
  complete quality gate at max KL 0.0767056. The repaired
  `mmq128x32_d8_f32` gate/up route stores one FP32 scale per 16 activations in
  the same 160-byte block and pairs with `mmq64x32_d4_f32` down. It retains
  direct/exact selected and grouped-down paths as required fallbacks;
  D4x2/D4x3 remain diagnostics.
- Promoted on gfx1151 after the clean category gate passed at max KL
  0.040724836, 317/320 top-1, 2.615x aggregate prefill, flat decode, and exact
  lifecycle recovery. Clean selector-unset publication now passes at
  **354.820 tok/s** median with all three pp512 samples above 350, and the final
  cached trace names the D8 128-column consumer. Retain the explicit
  exact/D4x2/D4x3 rollback modes for one release; the rollback window is now
  the sole removal condition for redundant experiment selectors. Remove the
  rejected D4 gate default candidate at that boundary. Pre-admission pp512
  samples were 353.951/356.082/356.473 tok/s.

## Laguna `f16_prefill_mode=hipblaslt_range_direct`

- Added 2026-07-25 as an explicit session-local LAP-6 candidate. Rows greater
  than one cast each BF16 producer row once into existing FP16-sized scratch,
  use an exact power-of-two row scale, run cached zero-workspace hipBLASLt
  FP16-input/F16-weight contractions with FP32 output, and restore the scale
  before the next boundary. Decode and the retained custom projection route
  remain unchanged.
- Promoted as the gfx1151 package capability by the clean compounded category
  gate. Clean selector-unset pp512/milestone publication and the final
  hipBLASLt trace now pass. Retain exact tiled as rollback for one release;
  remove the session selector after that rollback window.
- The 2026-07-26 exact range-qualified follow-up makes
  `hipblaslt_norm_direct` the gfx1151 default. Actual norm metadata proves a
  **16.34623** FP16 input bound; full logits, final/pre-final hidden, KV,
  cursor, token, and token-logit hashes match `hipblaslt_scaled`. Seven
  explicit pairs improve **502.348 -> 505.887 tok/s (+0.704%)**. Keep
  `hipblaslt_scaled` only as a one-release rollback for the range-qualified
  producer boundary, then collapse it while retaining the general scaled cast
  for the unnormalized attention-output projection.
- The second 2026-07-26 range proof computes per-layer value/gate maximum row
  L2 norms from the resident F16 source and applies Cauchy-Schwarz, FP32 dot,
  BF16-rounding, and 2x online-attention safety factors. The worst gated
  attention-output bound is **7,957.539**, still **4.116x** inside FP16 after
  the runtime's separate 2x admission reserve. `hipblaslt_range_direct` now
  removes the remaining output row reduction/scale and is the gfx1151 default;
  keep `hipblaslt_norm_direct` and `hipblaslt_scaled` only for the same
  one-release rollback window, then collapse both selectors.

## Laguna packed-query F32 hipBLASLt attention selector

- Added 2026-07-27 for the quality-gated two-call dense-initial attention
  route. `prefill_attention_hipblaslt_packed_queries=false` restores the
  admitted eight-QK/eight-PV route; unsafe attention shapes retain their
  established `KVLiveSpans` fallbacks independently.
- The route passes the tuned leaf, seven-pair pp512, CPU-reference,
  pp512-all-exact KL, cached trace, and clean selector-unset gates. Production
  improves **623.050/563.399/462.430 -> 629.101/566.858/463.903 tok/s**;
  tracing cuts pp512 attention **82.763 -> 73.330 ms** and dispatches
  **4,145 -> 2,417**.
- Retain explicit false through the next selected-projection checkpoint, then
  collapse the redundant positive selector/session rebuild path while keeping
  the whole-BLAS-route rollback and every unsafe-shape `KVLiveSpans` fallback.

## Laguna direct packed-query producer rollback

- Added 2026-07-27 as
  `prefill_attention_hipblaslt_packed_query_producer=false`. The gfx1151
  capability writes only the three qualified dense-initial M128 query tiles
  head-major from fused RMSNorm/RoPE; false restores the exact standalone
  generic-to-head-major query transpose. Generic row zero and every unsafe
  attention route are independent fallbacks, not part of this rollback.
- Eleven complete pp512 pairs are state-exact and improve
  **647.210 -> 650.651 tok/s (+0.532%, 7/11 wins)** with **1.557 ms** paired
  median wall saved. Clean tracing removes all **144 / 4.907-ms** query
  transposes, cuts dispatches **2,273 -> 2,129**, and improves the exact
  producer-plus-pack boundary **20.530 -> 16.666 ms (-18.82%)**.
  Selector-unset publication reaches **654.249/579.699/468.608 tok/s**.
  Keep explicit false through the next optimization checkpoint, then remove
  this one-purpose session selector and A/B comparison while retaining the
  generic producer for unqualified rows.

## Laguna wave-per-row causal-softmax rollback

- Added 2026-07-27 as
  `prefill_attention_hipblaslt_wave_rows_softmax=false` plus the session
  setter. The gfx1151 capability selects a local32 wave-per-row reduction;
  false restores the local256 block/LDS reduction. Unsafe shapes never enter
  either BLAS route and retain their established `KVLiveSpans` kernels.
- The candidate improves the qualified 48-layer attention model
  **72.738 -> 62.755 ms (-13.73%)** and seven-pair pp512
  **614.668 -> 620.032 tok/s (+0.873%, 6/7 wins)**. All-exact KL improves
  **0.002097 -> 0.001796**, top-1 stays 2930, and tracing reports
  local32/VGPR24/SGPR128/LDS0/scratch0.
- Clean selector-unset publication now passes at
  **632.618/568.845/464.606 tok/s**, and the required later corrected
  attention-family trace measures **69.983 ms** at unchanged **2,417**
  dispatches. The removal preconditions are satisfied. Remove the session
  setter and one-purpose A/B harness in the next rollback-cleanup unit while
  retaining the block256 kernel as the numerical fallback until the
  packed-query rollback window closes.

## Laguna `dense_q4_prefill_mode=wmma_pack8`

- Added 2026-07-25 as an explicit session-local LAP-5 candidate. Rows from 16
  use wave32 WMMA consumers over existing rank-2 Q4 pack8 bytes/effective FP32
  scale/min planes and raw Q6_K bytes. Q4 remains 64x16; Q6 now defaults to
  the exact scratch-free 16x32 tile after the 64x16 body exposed
  VGPR256/236 B-thread spilling. They add no sidecar, leave decode unchanged,
  and retain exact pack8/raw FMA paths as fallbacks.
- Promoted as the gfx1151 package capability by the clean compounded category
  gate. Clean selector-unset pp512/milestone publication and the final Q4/Q6
  WMMA trace now pass. Keep the exact fallback for one release; remove the
  session selector after that rollback window.
- `HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE=64x16` is the temporary Q6 tile
  rollback added 2026-07-26. Remove the environment branch after the clean
  16x32 publication and one release window remain non-regressive; explicit
  `tile_m`/`tile_n` arguments stay for tests and microbenchmarks.
- `HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE=64x16` is the temporary Q4 pack8
  shape-policy rollback added 2026-07-26. Remove it after clean publication,
  a confirming family trace, and one non-regressive release window; explicit
  tile arguments remain part of the primitive test/microbenchmark surface.

## Priority Cleanup (do first)

**Revalidate the gfx1151 GGUF graph default on the current stack and retire
unrelated legacy graph blocks.** SOL-G5 reintroduced a state-bound runtime graph
with a complete transition key. gfx1151 advertises its measured 128-step
admission; gfx1100 now advertises a separately measured 24-step admission after
passing all 24 hidden/GDN/KV/token transitions on W7900. Non-streaming c1 greedy
generation uses the graph only at each backend's admitted horizon, with
`HIPENGINE_GGUF_DECODE_GRAPH=0` retained for rollback. The explicit
`qwen35_gguf_bench.py --graph-replay-decode` surface is required for the current
default decision, while `scripts/gguf_mtp_bench.py --target-graph-verify` /
`--target-graph-batched-verify` remains separate stale diagnostic plumbing.

SOL-G4 provides the correct comparison floor: clean p512/d128 eager is
`20.290 ms/token`, while a 24-step marker profile contains `18.402 ms/token` of
GPU kernels (`88.62%` of profiled host wall). SOL-G5's clean production route
at `7f611fe3` passed 128 launches and measured a capture-inclusive
`20.334 -> 20.311 ms/token` (+0.112% throughput) edge. The 2026-07-12 TheRock
HIP 7.15 refresh is still 128/128 exact but rejects the graph wall on both the
scalar parent (`20.5230 -> 20.5736 ms/token`) and wave/block candidate
(`20.4723 -> 20.5324`). Do not remove the rollback flag or broaden graph
admission. A separate scoped decision must either reproduce a current graph win
or restore eager as the gfx1151 production selector; the wave/block kernel
itself helps both routes and is not the cause of this graph-policy result.

The gfx1100 decision is independently strong: clean W7900 p512/d24 SOL-G5 at
`833921ce` passed 24/24 byte-exact transitions and measured capture-inclusive
`30.5364 -> 12.5139 ms/token` (**2.4402x**, five runs). Per-token recapture was
only `22.22 tok/s`; the retained route is one state-bound capture followed by 24
validated relaunches, not recapture. Keep the gfx1100 admission at 24 until a
shorter-horizon audit establishes a lower break-even.

## Qwen GGUF external-DMS capture seam

- The default-off `Qwen35GGUFResidentSession.prefill(..., dms_capture=...)`
  diagnostic synchronously exports normalized pre-Q hidden rows and dense Q/K
  intermediates to an offline checksummed sink. It is never selected by
  `LLM.generate()` and intentionally stalls the prefill stream for bounded
  capture correctness.
- Removal trigger: after the Qwen3.8 external sidecar is trained and its
  reproducible capture manifest is retained, either promote this as an explicit
  model-plugin-owned offline capture API or remove the session keyword and
  runtime callback. Do not leave a generic synchronous D2H tap in the serving
  façade, and never make it an environment-selected production path.

## External-DMS compact device transaction journal

- The correctness-first external decision bridge snapshots only the active
  request's compact K/V/position/eviction extents to bounded host transaction
  scratch and restores them byte-for-byte on rollback. It is not a persistent
  dense shadow, but synchronous D2H/H2D journaling is not a production-speed
  endpoint.
- Removal trigger: once the integrated gfx1100 compact owner is wired, replace
  this with a resource-claimed device-local copy/journal or an exactly reversible
  in-place mutation protocol, then retain the host journal only as the registered
  strict fallback. Do not remove rollback coverage or restore metadata without
  restoring payload bytes.

## External-DMS serving control readbacks

- Correctness-first `exact_budget` prefill reads all external-linear logits to
  host, deterministically ranks them, and uploads the final device mask. Decode
  also synchronizes after all 16 full-attention layers and reads 64 decision
  bytes before host metadata finalization. Neither path retains dense KV, but
  both are production-speed debt and prevent graph-owned control.
- Removal trigger: implement bounded deterministic per-layer/head device ranking
  plus device-resident decode live-count/position/eviction finalization. Preserve
  the current host paths only as strict diagnostic fallbacks until exact mask,
  rollback, c1-c32 lifecycle, frozen-profile quality, and profiler gates pass.
  Remove their positive runtime selection once the device owner survives one
  release window.

## Cleanup Ledger

| Area | Debt | Current status | Removal trigger |
| --- | --- | --- | --- |
| Laguna source-F16 prefill selector | `HIPENGINE_LAGUNA_F16_PREFILL=auto|gemv|tiled|wmma_comp_swa` exposes the promoted scoped AR-O2 route plus exact LPF-1 rollback. | gfx1151 `auto` now selects compensated WMMA only for QKV/gate/O on the 36 SWA layers from M16. The clean gate moves weighted prefill 53.388->69.037 tok/s (+29.313%) at max KL 0.043888 and 318/320 top-1; every category, Poolside oracle, determinism, and lifecycle gate passes. All 12 full-attention layers, M2-15, rows=1, and unmeasured backends remain exact. The rejected direct WMMA route stays removed after max KL 0.097062. | After one release window and a defaults-only gfx1151 refresh, remove redundant positive `wmma_comp_swa` selector semantics and keep at most explicit `tiled`/`gemv` rollback. Never revive direct `wmma`, broaden the compensated scope without an independent full quality gate, or remove exact rows=1/unsupported fallbacks. |
| Laguna SWA token4-exact decode rollback | `LagunaGGUFResidentSession(..., swa_decode_variant=...)` and `allocate_laguna_kv_cache(..., swa_decode_variant=...)` retain baseline `swa_context_spans` beside gfx1100's backend-qualified `swa_context_token4_exact_spans` default. | Promoted on gfx1100. The exact 4-wave/4-slot schedule passes all focused wrap/eviction/KV/runner gates; clean SWA improves 49.60% short and 52.80-53.03% at 512/1K/near-4K. The full category gate moves h32 decode 38.840->43.081 tok/s and E2E 11.448->11.760 with prefill within -0.223%; gfx1151 and unmeasured backends remain baseline. | After one release window and a defaults-only gfx1100 refresh, remove public positive candidate selection and keep at most explicit baseline rollback. Never remove the registered baseline or change unmeasured backend defaults without independent evidence. |
| Laguna SWA wave32-exact prefill rollback | `LagunaGGUFResidentSession(..., swa_prefill_variant=...)` and `allocate_laguna_kv_cache(..., swa_prefill_variant=...)` retain explicit wave32 selection beneath gfx1151's online-qrow2 default. | LPF-5 first promoted wave32 on gfx1151: it reconstructs the original stride-64/32/16..1 FP32 tree, passes the 508..515 fixture byte-exactly, improves the leaf 20.434->9.229 ms (2.214x), and moves exact full-model 512/1K/4K prefill +8.31%/+12.85%/+14.06%. Exact context-qualified qrow2 later overlaid only M128/start>=128 slices; online qrow2 now owns measured gfx1151 SWA prefill after its full quality gate. Wave32 remains explicit rollback and the exact short/partial fallback inside context-qualified qrow2. Unmeasured backends retain prior defaults. | After one release window plus the post-prefill DFlash refresh, remove redundant public positive wave32 selection if the exact-qrow2 rollback no longer needs it; keep the registered wave32 fallback for unsupported/short rows. Never remove unmeasured-backend fallbacks without independent evidence. |
| Laguna SWA exact-qrow2 prefill rollback | Explicit `swa_context_rows_qrow2_exact_spans`, context-qualified `swa_context_rows_qrow2_m128_c128_exact_spans`, and retained wave32 variants coexist beneath the promoted online-qrow2 default. | The exact context-qualified route applies qrow2 only to complete M128 attention slices at absolute start>=128. Its final exact three-repeat gate improves 512/1K/4K prefill 0.893%/1.212%/1.040%; its complete category gate is exact and non-regressive at 0.999652x prefill and 0.999917/0.999999x h16/h32 E2E. It now serves as the primary exact rollback; empty-context, short, partial, and verifier rows delegate to wave32. gfx1100 and unmeasured backends are unchanged. | After one release window and a defaults-only gfx1151 512/1K/4K refresh, remove redundant positive direct-qrow2 selection if no bisection needs it; keep wave32 as the unsupported/short-row fallback and context-qualified exact qrow2 as the primary numerical rollback. Never broaden exact qrow2 below M128 or before start 128 without independent crossover and full gates. |
| Laguna global qrow2 online-prefill rollback | Explicit `global_context_rows_spans` remains beside gfx1151's promoted `global_context_rows_qrow2_online_spans` default. | The online route streams one BF16 K/V row across two adjacent queries without whole-context score LDS. It improves repeated 512/1K/4K **2.472%/5.444%/21.854%**; the complete category gate improves weighted prefill **0.315%** and h16/h32 E2E **0.184%/0.125%** with max KL `0.030836`, top-1 317/320, every category positive, and Poolside/repeats/lifecycle passing. gfx1100/unmeasured backends stay exact. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-global-qrow2-online-retained.json`. | After one release window and a defaults-only 512/1K/4K plus category refresh, remove redundant positive explicit-online selection if no bisection needs it; keep `global_context_rows_spans` as the required exact fallback and explicit rollback. Never enable on another backend without independent quality/performance evidence. |
| Laguna SWA qrow2 online-prefill rollback | Explicit context-qualified exact qrow2 and wave32 routes coexist beneath gfx1151's promoted `swa_context_rows_qrow2_online_spans` default. | One wave replaces exact qrow2's two ring scans with online max/denominator/output state. M128/full-window and start508 wrap improve **3.093x/2.904x**; repeated 512/1K/4K improves **6.828%/9.364%/10.766%**. The complete category gate improves weighted prefill **1.086%** and h16/h32 E2E **0.616%/0.420%** with max KL `0.042924`, top-1 316/320, every category positive, and Poolside/repeats/lifecycle passing. gfx1100/unmeasured backends retain prior defaults. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-swa-qrow2-online-retained.json`. | After one release window and a defaults-only 512/1K/4K plus category refresh, remove redundant positive explicit-online selection if no bisection needs it; keep exact context-qualified qrow2 and wave32 as rollback/fallback. Never enable on another backend without independent quality/performance evidence. |
| Laguna prompt prefill fallback | `LagunaGGUFResidentSession.prefill(..., use_bulk=False)` keeps the original token-serial prompt path beside default chunked rows. | Multi-length state/target-AR gates are complete. LPF-4 established exact bounded 128-row bulk chunks; AR-O3 defaults gfx1151 to its independently retained capacity. WPF-2 now defaults gfx1100 to matrix512/attention128/grouped-exact after all 48 boundaries/KV/spans/routing/lifecycle pass and clean publication reaches **99.230/91.559 tok/s**. Explicit overrides retain M128/M256 and paired direct routing. Token-serial remains only as the independent state oracle and rollback through the post-prefill DFlash refresh; `forward_token()` decode is unchanged. Evidence: `benchmarks/results/2026-07-23-gfx1151-laguna-matrix-chunk-retained.json` and `benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-grouped-iq-matrix512-retained.json`. | Move token-serial prefill to a correctness-only helper (or remove the public selector) after bulk passes the full prompt-length/context matrix, retained target-AR performance is non-regressive, and verifier accept/rollback gates no longer need prefill bisection. |
| Laguna source-F16 tiled prefill selector | `HIPENGINE_LAGUNA_F16_PREFILL=auto|gemv|tiled` exposes explicit selection around the separately registered LPF-1 row/column tile. | Promoted gfx1151 default from two rows. Clean same-session rows 2..128 are exact and all faster (2.0538x weighted); the two-repeat category gate moves prefill 23.333->48.560 tok/s, TTFT 3.481->1.692 s, and h32 E2E 5.719->8.717 with neutral decode and all correctness/lifecycle gates. The reassociated WMMA control remains removed after changing three trajectories. `gemv` is the release rollback; rows=1/unsupported backends always retain registry-driven GEMV. | After one release window and a defaults-only gfx1151 refresh, remove the positive `tiled`/`auto` experiment semantics and keep at most one clear `gemv` rollback until release confidence permits removing the env selector entirely. Never remove the registered rows=1/unsupported-backend GEMV fallback. |
| Laguna grouped-down diagnostic selector | `LagunaGGUFResidentSession.set_selected_down_mode(direct|grouped_smallm|adaptive_grouped_smallm|grouped_smallm_fused|adaptive_grouped_smallm_fused)` retains explicit A/B selection beside the backend-qualified exact grouped-combine Q4/Q6 down default. | Exact grouped combine is promoted on gfx1151 from 32 rows after a bit-exact 1.249-1.313x sub-window screen, 0.99972x five-repeat complete-model screen, and exact/non-regressive category gate. The quality-rejected M16/M32 runtime route, scratch, selectors, and harnesses are removed; its registered leaf oracle remains. gfx1100 and rows below 32 retain direct. | Keep `adaptive_grouped_smallm` as the explicit unfused grouped rollback through one release window. Then remove redundant positive `grouped_smallm*` experiment modes and keep at most the adaptive unfused rollback plus clear `direct` for unsupported backends/short rows. Never remove direct fallback without independent evidence. |
| Laguna D9 MoE-tail next-RMS rollback | `LagunaGGUFResidentSession(..., use_moe_tail_next_rmsnorm=False)` restores the exact registered BF16 add/add/F32-weight-RMS chain around the gfx1100/gfx1151 c=1 composite; rows>1 and unsupported backends keep that chain. | Promoted on gfx1100 c=1. Synthetic and all-47-boundary actual Q2 XL state are exact; clean short/512/1K/near-4K kernel sum, span, and profiled child rows improve, and the complete category gate moves h32 decode 46.409->47.132 tok/s with every decode/E2E row positive. Independently promoted on gfx1151: native hidden17/3072 is byte-exact, cached resources are local256/VGPR16/LDS1024/scratch0, and counterbalanced p512/d128 moves rollback 14.529573/14.525706 to 14.555265 tok/s while removing 94 launches/token. | Keep for one release/bisection window and the current DFlash/MTP refresh, then remove the constructor selector while retaining the architecturally required unfused chain. |
| Laguna D12 raw-Q5 wave32x2 selectors | `LagunaGGUFResidentSession(..., use_q5_wave32x2_output=..., use_q5_wave32x2_query_gate=...)` independently rolls the gfx1100 defaults back to the exact pack8 siblings for c=1 attention output and unequal query/gate. | Promoted together on gfx1100. Formal 50-warmup/15x200 actual-weight leaves improve 13.63-24.80% in HIP events; clean short/512/1K/near-4K kernel sum/span/child all improve; and the counterbalanced canonical gate moves h32 decode 47.046->48.987 tok/s with every category positive and prefill neutral. Synthetic/production leaves and full shared-weight state are exact; cached trace is local32/VGPR96/LDS0/scratch0. gfx1151, rows>1, shape/registry misses, and explicit disable retain pack8. | After one release/bisection window and the current DFlash/MTP refresh, remove positive constructor-selection semantics and keep at most a clear pack8 rollback. Never remove pack8 as the required fallback. |
| Laguna exact Q5 fixed-metadata rollback | `LagunaGGUFResidentSession(..., use_q5_fixed_meta_output=False, use_q5_fixed_meta_query_gate=False)` and `laguna_target_ar_bench.py --disable-q5-fixed-meta-{output,query-gate}` independently restore the retained D12 coefficient-publication siblings around the gfx1100 fixed-metadata default. | Promoted. Two wave-uniform 128-bit metadata loads remove 32 coefficient exchanges and reduce logical VGPR **89 -> 72** without a sidecar. First/last actual output/query-gate rows improve event **19.80-25.19%** / wall **17.59-24.07%** at exact bits; full state/lifecycle and cached c=1 trace pass. Both clean process orders improve Q5 **22.68-23.12%**, kernel sum **2.35-6.34%**, span **2.17-5.58%**, and child throughput **2.26-4.41%** at every context. Both complete 18-prompt orders move h32 **54.476 -> 57.711 tok/s (+5.938%)** with every train/heldout category decode and E2E row positive. | Retain explicit role-scoped `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered coefficient-publication siblings and pack8 routes for rollback, rows>1, shape/registry misses, and unsupported backends. |
| Laguna shared-Q5 fixed-metadata rollback | `LagunaGGUFResidentSession(..., use_q5_shared_fixed_meta=False)` and `laguna_target_ar_bench.py --disable-q5-shared-fixed-meta` restore the registered local128 pack8 BF16 pair around gfx1100's local32 fixed-metadata default for sparse layers 1-46. | Promoted. First/last actual pairs are BF16-bit exact and improve event **27.45-27.61%** / wall **26.88-27.48%**. Full state/lifecycle and default-vs-rollback pass; cached tracing records exactly 46 calls/token at local32/VGPR80/LDS0/scratch0 and unchanged 723 dispatches/token. Both clean orders improve shared-pair/kernel/span/child **45.99-47.13% / 1.32-3.02% / 1.43-2.62% / 0.89-3.33%**. Both complete category orders move h32 **59.500 -> 60.942 tok/s (+2.425%)** with every train/heldout category decode positive. Layer 47 Q6, rows>1, key misses, and unsupported backends retain existing routes. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered local128 pack8 pair for rollback, rows>1, shape/registry misses, and unsupported backends. |
| Laguna mixed attention-projection quad rollback | `LagunaGGUFResidentSession(..., use_mixed_q6_fixed_meta_attention=False)` / `laguna_target_ar_bench.py --disable-mixed-q6-fixed-meta-attention` restores generic Q6 blocks inside the retained mixed quad; `use_mixed_q5_q6_attention=False` / `--disable-mixed-q5-q6-attention` restores the exact Q5/Q6 pair plus layer-47 Q8 singleton chain. | Promoted. Layers 0-46 use Q5 fixed-metadata wave32 owners plus generic Q6 local128 blocks; layer 47 uses exact generic Q6/Q8 local128 blocks. Actual first/last inclusive screens improve event **4.52-16.57%** and wall **3.65-14.23%**; full logits, 48 hidden/47 routed boundaries, active K/V/`KVLiveSpans`, reset, and lifecycle are exact. Cached trace records 48 calls and **772 -> 723 dispatches/token**. Both clean process orders improve projection work **2.02-3.35%**, kernel sum **0.09-0.35%**, span **0.69-1.56%**, and child throughput **1.06-2.92%**. Both complete category orders move h32 **57.833 -> 58.425 tok/s (+1.024%)** with every train/heldout category decode positive and all guards passing. The subsequent fixed-Q6 sibling preserves launch structure and exact state while improving actual projection wall **8.50-38.85%**, clean projection/kernel/span **8.08-10.10% / 0.73-1.26% / 0.57-1.49%**, and two-order h32 decode **58.466 -> 59.211 tok/s (+1.275%)**; gfx1100 now defaults that sibling. | Retain both explicit `False` rollback levels for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the generic-Q6 mixed quad plus registered exact pair/singleton chain for rollback, rows>1, shape/registry misses, and unsupported backend defaults. |
| Laguna all-local32 mixed-projection rollback | `LagunaGGUFResidentSession(..., use_mixed_local32_fixed_meta_attention=False)` and `laguna_target_ar_bench.py --disable-mixed-local32-fixed-meta-attention` restore the registered local128 fixed-Q6 Q5/Q6 quad around gfx1100's local32 default; registry miss also restores that route, including layer 47's Q6/Q8 tuple. | Promoted. Global/SWA production outputs, full state, and default-vs-rollback are bit-exact; first/last actual layers improve event **11.39-14.77%** and wall **11.24-15.72%**. Both clean orders improve projection/kernel/span/child **7.00-8.12% / 0.49-2.12% / 0.45-2.77% / 0.20-1.29%**. Both category orders move h32 **60.900 -> 61.732 tok/s (+1.367%)** with every train/heldout category positive at unchanged 723 model kernels/token. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered local128 fixed-Q6 mixed quad and pair/singleton chain for layer 47, rows>1, key misses, and unsupported backends. |
| Laguna Q4 LM-head local32 fixed-metadata rollback | `LagunaGGUFResidentSession(..., use_q4_lm_head_local32_fixed_meta=False)` and `laguna_target_ar_bench.py --disable-q4-lm-head-local32-fixed-meta` restore the registered local128 c=1 BF16/F32 LM head around gfx1100's local32 default; bulk-prefill/verifier projection, gfx1151, rows>1, and registry miss also retain local128. | Promoted. The full 100,352-logit actual head and default-vs-rollback model state are bit-exact; actual event/wall improve **30.04%/21.44%**. Both clean orders improve LM-head/kernel-sum time **29.07-30.79% / 0.34-1.10%** at unchanged 723 kernels/token. Both complete category orders move paired h32 **61.675 -> 61.992 tok/s (+0.512%)** with every train/heldout category positive. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered local128 body for bulk/verifier, rows>1, key misses, and unsupported backends. |
| Laguna P0 IQ3 serial fallback | `LagunaGGUFResidentSession(..., iq3_c1_down_schedule="serial_weighted")` remains the second-level c=1 rollback behind the retained wave10 default and explicit wave4 rollback; unsupported backends continue to default serial. | Historical P0 promotion moved h32 **48.780 -> 50.254 tok/s (+3.022%)** with exact full state and every context/category positive. Wave4 is now the immediate fallback for the later retained wave10 composite. The slower `row4_reduce` runtime mode is removed; the exact sign-bit sibling remains diagnostic after failing clean span/child guards. The measured tile4 leaf remains explicit for DFlash verifier rows. | Fold this entry into the wave10 rollback after one release/defaults-only refresh. Never remove the registered serial producer/composite or unmeasured-backend fallback without independent evidence. |
| Laguna IQ3 wave10-fused rollback | `LAGUNA_IQ3_WAVE10_FUSED=True` defaults gfx1100 c=1 decode to the exact local320 producer/reducer composite; `LagunaGGUFResidentSession(..., iq3_c1_down_schedule="wave4_reduce")` and `scripts/laguna_target_ar_bench.py --iq3-c1-down-schedule wave4_reduce` restore the registered producer-plus-reducer chain. Exact-key miss, rows/prefill, gfx1151, and unsupported backends retain prior registered schedules. | Promoted. Shared-weight full state/default-vs-wave4 is exact through 16 transitions. Cache-only tracing records **45 candidate + two unchanged reducers / 678 model kernels/token**, local320/VGPR88/LDS512/scratch0. Every short/512/1K/near-4K clean order improves inclusive IQ3 and kernel sum with span/child guards passing. Both complete 18-prompt orders move h32 **62.318 -> 63.270 tok/s (+1.528%)**, every train/heldout category improves at both horizons, and E2E/prefill/TTFT guards pass. | Retain explicit wave4 rollback for one release/bisection window and a defaults-only matched-completion refresh, then remove positive `wave10_fused` experiment semantics while permanently keeping the registered wave4+weighted and serial fallbacks for key miss, rows/prefill, and unsupported backends. |
| Laguna P2 exact split attention rollback | `LagunaGGUFResidentSession(..., use_split_attention=False)`, `allocate_laguna_kv_cache(..., use_split_attention=False)`, and `laguna_target_ar_bench.py --disable-split-attention` restore the registered global/SWA token4 readers; explicit threshold arguments retain focused A/B control. | Promoted on gfx1100 at global `>=127` and SWA `>=65`. Actual layer event/wall gates and full state are exact; clean short/512/1K/near-4K attention improves **15.66-23.28%**, kernel sum **2.67-16.11%**, span **4.65-14.63%**, and child throughput **1.19-17.58%**. The complete two-order 18-prompt gate moves h32 **50.093 -> 51.436 tok/s (+2.681%)** and E2E **+0.496%**, with every train/heldout category positive. Two reusable buffers add 1,572,864 bytes. gfx1151 and unmeasured backends retain fallback. | After one release/bisection window and P2.2 adjudication, remove positive threshold experiment semantics and keep at most the clear disable rollback. Always retain the registered exact global/token4 readers for below-threshold and unsupported routes. |
| Laguna P2 exact SWA tile16 rollback | `use_swa_split_tile16=False` / `--disable-swa-split-tile16` restores P2.1's one-slot score producer above gfx1100's retained live-257 tile16 default; explicit threshold remains for focused A/B. | Promoted. Two process orders improve pooled 512/1K/near-4K SWA attention **0.571%/0.344%/0.208%** and total attention **0.461%/0.272%/0.056%** with unchanged dispatches/memory. The complete fallback gate and a 150-transition default-vs-rollback gate are exact/non-regressive. | Fold into the P2 rollback row after one release/defaults-only long-context refresh; retain the registered P2.1 producer permanently for below-threshold and unsupported routes. |
| Laguna P4.1 split-reducer gate-fusion rollback | `LagunaGGUFResidentSession(..., use_split_gate_fusion=False)`, `allocate_laguna_kv_cache(..., use_split_gate_fusion=False)`, and `laguna_target_ar_bench.py --disable-split-gate-fusion` restore the exact split reducer plus standalone softplus BF16 gate. | Promoted on gfx1100 split paths. First/last actual layers at live 128/257 are bit-exact and improve inclusive event **3.00-10.05%** and wall **2.89-9.60%**; full state/lifecycle are exact. The two-order category gate moves h32 **51.497 -> 51.825 tok/s (+0.637%)** with every train/heldout category decode row positive and no new allocation. Below-threshold, registry miss, gfx1151, and unsupported backends always use the registered unfused chain. | After one release/bisection window and a defaults-only matched-completion refresh, remove the positive constructor/CLI selection while retaining the architecturally required unfused registered fallback. |
| Laguna current-P4 head RMSNorm+RoPE+KV rollback | `LagunaGGUFResidentSession(..., use_head_kv_fusion=False)` and `scripts/laguna_target_ar_bench.py --disable-head-kv-fusion` restore the registered two-launch head-plus-writer chain around the gfx1100 default. | Promoted on gfx1100 c=1. Global/SWA boundary fixtures, 48 hidden/47 routed boundaries, active K/V and every span field, reset, and lifecycle are exact; first/last actual layers improve inclusive event **33.05-39.36%** and wall **33.41-39.13%**. Pooled short and all long clean kernel/span/child rows improve; the two-order 18-prompt gate moves h32 **51.872 -> 52.391 tok/s (+1.001%)** with every train/heldout category decode positive and all guards passing. Rows/prefill, P2/P4.1 bodies, gfx1151, and unsupported backends remain unchanged. | Retain `False`/CLI rollback for one release and a defaults-only matched-completion refresh, then remove positive constructor selection while permanently keeping the registered unfused chain for rows/prefill and unsupported backends. |
| Laguna SWA split wave-local reducer rollback | `LagunaGGUFResidentSession(..., use_swa_split_wave_local=False)` and `scripts/laguna_target_ar_bench.py --disable-swa-split-wave-local` restore the exact shared-statistics gated reducers around the gfx1100 wave-local default. | Promoted. Full logits, 48 hidden/47 routed boundaries, active K/V and every span byte, reset, and lifecycle are exact. First/last actual layers improve **4.84-18.96%**; two clean process orders improve reducer/SWA **4.63-5.22% / 4.24-4.55%**, kernel sum **0.94-1.98%**, and span **0.61-1.69%** at every context. The complete two-order gate moves h32 **52.211 -> 52.514 tok/s (+0.580%)** with every category decode positive and all guards passing. | Retain explicit false for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered shared-statistics reducer for unsupported backends and rollback. |
| Laguna exact IQ2 expanded-magnitude rollback | `LagunaGGUFResidentSession(..., use_iq2_grid64=False)` / `scripts/laguna_target_ar_bench.py --disable-iq2-grid64` restores the compact-grid c=1 IQ2 gate/up+SiLU tile2 route around gfx1100's expanded-magnitude default; rows>1 always retain compact grid. | Promoted. A 4 KiB canonical magnitude table replaces the retained 1 KiB packed-code reconstruction without duplicating persistent weights. First/last actual layers improve **30.00-33.73%**; full state is exact. Two clean process orders improve IQ2/kernel/span/child at every context, and both complete 18-prompt orders move h32 **52.650 -> 54.540 tok/s (+3.590%)** with every train/heldout category decode and E2E row positive. gfx1151 and unsupported backends remain compact-grid. | Retain explicit `False` for one release/bisection window and a defaults-only matched-completion refresh, then remove positive constructor semantics while permanently keeping the registered compact-grid route for rows>1, unsupported backends, and exact fallback. |
| Laguna DFlash IQ3 selected-down tile4 | `LagunaGGUFResidentSession(..., iq3_selected_down_tile=1|4)` and `scripts/laguna_dflash_category_bench.py --iq3-selected-down-tile` retain tile1 plus the explicit gfx1100 tile4 verifier path. gfx1151 excludes tile4 and its public provider uses tile1. | Retained only in the measured gfx1100 scope: tile4 is exact there, improves the profiled 45-call family 33.66%, and moves counterbalanced h32/h128 DFlash decode +4.725%/+4.536% with every category/heldout decode and E2E positive. The independent gfx1151 transfer is rejected: clean tile1 and tile4 each pass only **24/36** true-AR equality rows with the same deterministic six-prompt failures; tile4 is trajectory-identical to tile1 but complete decode is noise-flat **15.73505 -> 15.73726 tok/s (+0.014%)**, E2E regresses **5.29982 -> 5.29797 (-0.035%)**, and category rows regress. The public packet was ineligible after the binding correctness failure. | Keep tile4 only for the retained gfx1100 explicit verifier route. Keep gfx1151/public DFlash on tile1; reopen only after the shared true-AR mismatch is repaired and a materially different gfx1151 mechanism has a plausible complete-route gain. |
| Qwen tokenizer EOS discovery | The PARO generator falls back to looking up `<|im_end|>` and `<|endoftext|>` because `tokenizers.Tokenizer.from_file()` does not expose `generation_config.json` EOS metadata. | The fallback recognizes both Qwen EOS ids, preserves explicit scalar/sequence metadata first, and deliberately avoids unrelated model-family markers. | Remove the string lookup once the Qwen model plugin owns and supplies normalized BOS/EOS/PAD metadata from `generation_config.json` / `tokenizer_config.json` to generation and sampling. |
| PARO width-plan execution and rollback flags | The greedy generator and `scripts/qwen35_batch_retained_bench.py` each execute `BatchWidthPartitionPlan` groups and collect similar telemetry; `_generate_batch_sampled()` remains diagnostic. `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS` and the legacy-named `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` remain explicit rollback opt-outs. | G3 retains an identity-matched schema-2 gfx1151 c2/c4/c8 profile. G4/G5 attach it to the shared stable-slot owner; blocking F1, native/serial SSE, live admission, c8 repeatability, and a no-flag OpenAI c4 confirmation are exact. gfx1151 backend capabilities now enable the hash/sync-checked packaged profile by default even outside a repository CWD; explicit `=0` still selects the serial fallback. gfx1100 remains package-default off pending owner c4/c8 symmetry. | After one release window and a defaults-only gfx1151 refresh, remove the two positive opt-in semantics and rename/collapse them to one clear rollback selector; consolidate duplicate generator/benchmark plan execution after gfx1100 owner c4/c8 is independently retained. Keep profile identity checks, unsupported-shape partitions, and exact serial fallback permanently. |
| PARO selected-MoE compatibility alias | `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE` and CLI `--batch-decode-moe-path selected_c1` remain compatibility aliases even though true per-row paths still use `selected_c1_per_row_*` labels. | Runtime metadata emits canonical `moe_decode_path=selected_batch` plus `moe_selected_batch_layers`; retained validators accept either exact grouped-compact or selected-batch layers while still requiring zero per-row fallback layers. The canonical env/CLI spelling is `SELECTED_BATCH_MOE` / `selected_batch`, and the old alias has lower precedence. Clean current-revision gfx1100 c2 is now retained at **121.923 tok/s** selected-batch versus c1 **116.022** and serial c2 **100.925**; all-layer/lifecycle/category/profiler gates pass and `auto` resolves selected-batch. Evidence: `benchmarks/results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json`. | Remove the legacy env/CLI alias after one compatibility window from the retained c2 promotion. Keep explicit per-row fallback names and counters permanently. |
| PARO short-context c>N attention | The optimized 1,024-thread `bf16_context_batch_spans` kernel remains registered beside the default 256-thread dense-order `bf16_context_batch_c1_exact_spans` route. | The optimized route is faster but introduces FP32 reduction drift that changes the full-model c2 trajectory; the exact route preserves physical batch block rows and matches dense c1 bit-for-bit. There is no runtime env flag and dispatch selects the exact variant through the registry. | Remove the duplicate optimized route or replace it only after a c-aware implementation is bit-exact to dense c1 at the 513-token model shape, passes full L40 c2 token/hidden/state/KV gates, and is non-regressive in repeated p512/d128 timing. |
| PARO gfx1151 greedy decode graph | `capture_decode_graph()` remains available, but public greedy c1 consults a resident-session architecture policy before capture. | The 2026-07-13 gfx1151 p512/d8 graph/eager fixture gate rejected generated-token equality, and graph wall was slightly slower (`0.133842` vs `0.133048` s). gfx1151 therefore uses canonical eager resident steps; gfx1100 retains its previously validated graph path. | Repair and re-admit gfx1151 only after a current-stack graph/eager gate matches generated tokens plus hidden/Conv/GDN/KV state across long replay and improves capture-inclusive wall. If no repair is planned, move the rejected graph probe to a diagnostic-only harness and keep eager as the permanent gfx1151 policy. |
| PARO ragged packed prefill | Ragged compact slabs automatically use per-segment linear/full-attention prefill; `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR` and `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN` retain explicit bisection routes. | SOL-P2 found the packed segmented/varlen route changed the first ragged row's persistent state and, across all 40 layers, its generated tokens. `per_segment_ragged_exact` is the production-safe fallback; equal-length slabs retain `packed_segments`/`packed_varlen`. | Repair the ragged packed kernels, then remove the automatic fallback only after c8 ragged generated-token plus all-linear-state/all-full-KV identity passes independent c1 on gfx1151 and gfx1100 and an end-to-end prefill comparison is non-regressive. Remove the force flags after the repaired route survives one release window. |
| PARO short-prompt prefill | The public generator uses token-serial c1 steps when prompt tokens are fewer than `linear_conv_kernel_dim`; low-level `prefill_native()` remains strict. | Release fallback. A no-env gfx1151 wheel-path smoke generated from the one-token prompt `Hello`; normal prompts still use native prefill. | Replace the serial fallback only after a dedicated native short-prompt kernel matches c1 generated tokens plus recurrent/KV state for lengths 1-3. |
| GGUF public AR profile | `HIPENGINE_GGUF_DECODE_REPACK=0` remains a rollback opt-out; low-level WMMA/GEMV selectors remain available for benchmark bisection. | Release default: T16 decode-repack is on, and public generate/stream sessions pass the resolved backend plus `use_wmma_prefill=True` and `use_gemv_decode=True`. A no-env gfx1151 Q4_K_M smoke generated one token through `LLM(model)`. | Remove the decode-repack opt-out after one release window and a defaults-only gfx1100 refresh. Keep raw layouts only where a quant/kernel lacks a T16 fallback or a retained diagnostic requires them. |
| GGUF fused dual+SiLU gate/up prefill floor | `hipengine/runtime/gguf_linear.py::_Q4_T16_DUAL_WMMA_SILU_MIN_ROWS = 33` (was 512) for the dense H5120 K5120/N17408 `linear_pair_silu` owner. | Bit-identical to the unfused chain at that shape for rows 45/96/192/511/512 (fixture from rows 2), and W7900 Qwen3.8-27B-Q4_K_M measured **+4.2%/+4.2%/+4.9%** prefill at 45/96/192 rows. The same owner is 8.2% slower per cycle inside captured target-verify groups (16/32 rows, C4-C8 K3, replications 3/3), which is why the floor sits at 33 rather than at the small-B boundary: row count is a proxy for prefill-vs-verify context. | Replace the proxy with an explicit prefill-context selector: a session-scoped floor whose value participates in `_DISPATCH_RESOLVE_CACHE` / `_PAIR_DISPATCH_RESOLVE_CACHE` keys (a scope that does not would leak decisions across scopes). Then lower the bulk-prefill floor to the rows<=8 small-B boundary, qualify target verification separately at its own group widths, and delete the row-count proxy. |
| GGUF prefill route-session stack | `Qwen35GGUFResidentSession.prefill` opens the wmma/q8/pack8/unequal-pair/F16 route sessions as one stack, while the packed admission entry `prefill_batch_native` now opens only `q4_t16_unequal_pair_prefill_session` (the route whose row floor was qualified). | Sessions are added per qualified route because each needed its own W7900 bit-parity + A/B evidence; the shipping AR route admits through `prefill_batch_native` (bound in `hipengine/generation/qwen35_gguf.py` 5 times), so a route enabled only in `prefill` is invisible to it - that is how the rows16 QKV dual landed with a flat topline until the packed entry was wired. | Extract one `_prefill_route_sessions(...)` context stack used by both entries (and by `_prefill_batch_native_impl` if a caller ever reaches it directly), then delete the per-entry session lists. Route-eligibility stays in the capability policies, not in call sites. |
| GGUF bench shipping-route alias | `scripts/qwen35_gguf_bench.py --public-ar-profile` (plus the recorded `public_ar_profile` / `shipping_ar_route_mismatch` protocol fields and the stderr warning) selects `use_wmma_prefill=True` + `use_gemv_decode=True` instead of the low-level env-default-off selectors. | W7900 Qwen3.8-27B-Q4_K_M measured 45-token prefill 0.4577 s unset vs 0.2818 s shipping (+62.4%), 512 tokens +1.7%, identical generated id, final logit 22.1395 -> 21.9644. Without the alias the harness default route silently measures a route the product never runs; see worklog `20260829T215206.066388Z-lhl-w7900-qwen38-submodule-parity-1921c9`. | Collapse once the bench resolves its default through the same session-profile helper the product uses (so no separate alias is needed), or once every retained short-prompt prefill artifact has been re-measured on the shipping route and the low-level selectors are gone from the bench surface. |
| GGUF duplicate AR loop ownership | `Qwen35GGUFResidentModelRunner` owns public blocking and OpenAI submit/poll execution, but `_generate_ar_serving_slots()` remains as a direct control/oracle and explicit compatibility fallback. | D1–E3/F1 now prove one shared model-owning loop through exact arbitrary-C burst/live admission, cancellation, SSE, shutdown, real KV ownership, observability, and retained server scaling on both gfx11 targets. The direct loop still supplies the independent c1/native-width oracle used by the retained E1/E2/E3 packets. | Both gfx11 triggers are met. During F2, move the oracle into an explicit test/benchmark helper and remove production call sites after one release window; keep registry-resolved unsupported-shape fallbacks. |
| GGUF RadixCache early admission history | Radix is now the HTTP and in-process default, with batched suffix prefill and registered gapped-page gather. The earlier admission used a latest-boundary policy and serial suffix steps; its results below are historical, not the current default policy. Stochastic reuse remains unsupported; deterministic processed-argmax requests preserve their host processors. | Early p256+s1 ownership gates passed exact state, refcounts/COW, source reclaim, snapshot eviction, and final drain on gfx1151 and gfx1100. The historical W7900 A2 packet closed lifecycle/pressure but rejected that agentic configuration: radix hit 0/12, 3/24, and 3/18 C1 turns, regressed active-SSE goodput 64.19%/65.63%/26.64%, and worsened tool-ready latency 181.90%/196.09%/38.81%. The original numerical and timing artifacts remain in the benchmark history. The current BF16 layout-repair packet independently passes all 16 category/lifecycle scopes with exact state and zero KL; it makes no new performance claim. | Keep lifecycle tests and the explicit `off` rollback. Remove obsolete admission cost guards only after their concrete slow-path conditions no longer apply. Any new performance claim needs a fresh same-host workload; the historical A2 result neither measures the repaired path nor overrides the owner-directed radix default. |
| GGUF LCP-1 convolution prefill | `HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE=baseline|tile32x128` selects between the production global-read convolution and the registered exact shared-token route. | gfx1151 selects `tile32x128` automatically. The clean 512/4K 82-part and wall gates pass, the 4K body falls `954.134 -> 49.790 ms`, and all six right-sized prefill rows improve `+1.10%..+24.04%` with unchanged memory. gfx1100 remains on `baseline` pending hardware transfer. The production implementation is the required unfused fallback and explicit rollback. | Remove the explicit mode selector after one release window if the gfx1100 transfer remains stable. Never remove the exact production fallback. |
| GGUF packed-AR singleton-indexed GDN | Backend capability `GGUF_GDN_INDEXED_SINGLETON_DECODE` selects a one-token-per-row indexed sibling while retaining the arbitrary-length segmented recurrence. | gfx1151 defaults to the singleton sibling after independent-c1 byte equality and exact p512/d64 trajectories; gfx1100 remains on segmented GDN pending hardware transfer. The runtime manifest records `indexed_singleton` versus `segments` explicitly. | Remove the gfx1100 capability split only after an independent W7900 c2/c4/c8 correctness/performance gate. Keep the segmented implementation permanently as the arbitrary-length fallback. |
| GGUF selected-MoE duplicate-expert reuse | `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE=0`, `HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE=0`, and `HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE=0` roll physical-C8 Q4T16 gate/up plus Q5/Q6T16 down back to per-selected-lane kernels; backend capabilities keep gfx1100 unchanged. | gfx1151 pairs consecutive dynamic expert-ID occurrences inside 128-thread blocks while preserving each row's reduction order. Q4 gate/up and Q5/Q6 down share weights across independent per-row accumulators. Lower widths and unpaired IDs retain exact fallback behavior. | Remove the three env opt-outs after one release window plus defaults-only gfx1151 direct/server refreshes and an independent gfx1100 transfer. Keep per-lane kernels for unsupported widths and as required fallbacks. |
| GGUF F32-weight cooperative c1 router | `HIPENGINE_GGUF_ROUTER_F32W_COOP=0` retains the separate expert-logits/shared-logit/top-k chain as an explicit rollback around the incumbent cooperative route. `HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER=0` restores the selected-ID counter alias plus per-layer host reset instead of the dedicated self-resetting four-byte counter. | gfx1100 clean commit `4c743994` first improved 4K graph decode **97.234 -> 98.273 tok/s (+1.07%)**; its persistent follow-up removed 40 reset nodes/token and improved the graph gate another **+1.65%**. Independent ZBook/gfx1151 requalification now keeps the same package route: **450/450** full-logit rows, three repeats, 18/18 live-state fingerprints, and 18/18 natural d128 trajectories are byte-exact; natural eager AR improves **30.438 -> 33.219 tok/s (+9.136%)** versus the strict chain, and fixed p512/d128 improves **+8.748%**. The unfused chain remains the required numerical fallback. | Keep both opt-outs until real Qwen3.6 strict/production profile plans bind the selected/fallback variants and exact control capture passes. Then remove the env experiment surface after defaults-only refreshes on both gfx11 targets, while keeping the registered unfused fallback permanently. |
| GGUF long-context split-K reduction | `HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE=0` and the minimum-context override retain the serial split reduction for rollback/A/B around the prepare-plus-coalesced-output route. | Promoted gfx1100 default from 32K after LCP-D2 (**+1.23%/+3.95%/+7.80%** at 32K/64K/128K, max KL **1.904e-6**). SH7-A1 independently promotes gfx1151 from 32K: one-queue 32K/64K decode improves **+1.560%/+2.394% (-0.333/-0.593 ms/token)**, all **1,296/1,296** semantic logits are byte-exact, and tracked/GTT/lifecycle are unchanged. | Keep the serial implementation permanently for <32K, unsupported routes, and numerical fallback. Remove the env/min-context experiment overrides after one release window plus defaults-only long-context refreshes on both gfx11 backends. |
| GGUF MTP server packed verifier | `_MTP_SERVING_TARGET_BATCH_MAX_SLOTS = 4` chunks c>N server target verification instead of sending all active slots to one packed target forward. | Default serving policy after the first packed verifier landing and the stream-draft/stream-verify follow-ups. c=2/c=4 packed target verify wins, but one 8-slot packed batch is a measured rejected regime (`11.58 tok/s`, `target_verify_batch_ms=63733.783`). The current c=8 stream path still chunks verify at 4 slots and reaches **52.18 tok/s**, with verifier still dominant (`slots_verify_phase_ms=12345.442`). | Remove or raise the cap only after rows>=16 packed verifier and resident-draft row-count/cold-slot behavior are tuned and a c=8 natural24 rerun beats the chunked stream path without correctness or latency regressions. |
| GGUF packed verifier GPU-event instrumentation | `HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS` records HIP events through `Qwen35GGUFResidentSession.verify_target_blocks_batch()` and compact-MoE leaves. | Default-off diagnostic. It exposed c=8 server verifier GPU leaves on 2026-07-05 but adds event overhead (`47.17 tok/s` in the compact-WMMA event run), so it is not a retained speed path. | Keep only while c>N MTP verifier tuning is active; remove or move behind a dedicated profiling helper once the packed verifier bottleneck is closed. |
| GGUF compact-WMMA tight no-read scope | `HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS=0` restores the scalar `wmma_total` D2H read around the gfx1100/gfx1151 defaults capped at 4,096 selected rows. | The old rejected c=8 probe overlaunched `selected_rows * 16`. LCP-2B instead proves the exact worst-case tile count `A + floor((S-A)/16)` and clears unused tile ids to `-1`. SH9-D1 independently transfers it to gfx1151 pp512: exact full state, neutral prefill **1366.040 -> 1366.013 tok/s**, and cached ROCTX **44 -> 4 `hipMemcpy` / 1,409 -> 1,369 dispatches / -1.736 ms span** with unchanged 80 selected-WMMA launches. This is multi-row prefill only; c1 decode bypasses the helper. | Keep the scalar fallback permanently for selected rows above 4,096, insufficient capacity, and exact-count diagnostics. Remove the env opt-out after one defaults-only prefill refresh on both gfx11 packages; do not remove the fallback. |
| GGUF selected-WMMA launch-bounds tuning | `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` remains an R&D build flag for selected-WMMA kernels. | Default unchanged after the 2026-07-05 c>N server probe. `=2` was flat at c=8 (**52.55/52.23 tok/s**); `=4` helped c=8 (**53.22/53.44**) but regressed c=4 (**49.20/49.04** vs retained **49.65**), so no default promotion. | Keep as kernel R&D only; do not promote without a c=2/c=4/c=8 same-protocol rerun that is non-regressive at every concurrency. |
| GGUF AR server packed decode | `HIPENGINE_GGUF_AR_PACKED_DECODE` is a default-on rollback opt-out around decode-shaped packed resident target passes for c>N GGUF greedy AR serving. | E1/E2 retain eager/graph c8 p512/d128, ragged, sparse, cancellation, and **748 packed-native / 0 row-local / 0 copies** on both targets. E3/F1 adds exact C13 c8+sparse-c8, middle-hole/new admission, and real SSE logical c1/c8/c9/c13/serial-c13 at **25.583/136.122/88.592/111.380/31.708 tok/s** on gfx1100 and **15.701/86.338/57.127/72.522/42.764** on gfx1151, with zero packed-route fallback and **189/189** exact requests per packet. | Both gfx11 E3/F1 triggers are met. Keep the opt-out for one release window, then remove the env switch during F2 while retaining registry-resolved scalar fallback for unsupported shapes. |
| GGUF AR server packed prefill | `HIPENGINE_GGUF_AR_PACKED_PREFILL` is a default-on rollback opt-out around packed final-row prompt prefill for c>N GGUF greedy AR serving. Both gfx11 packages now enable `GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK`; gfx1100 qualified it independently on the physical W7900 on 2026-09-06 (audit packet F1: a p512 C8 prefill drops the full-vocab LM head from 7 to 2 launches and 70 dispatches, two counterbalanced state-oracle shapes and the cancel/admission lifecycle differ only on the mask's own bookkeeping fields, and the mask-engaged width-4/8 production screen passes every hard gate; `benchmarks/results/2026-09-06-gfx1100-gguf-packed-prefill-final-output-mask-retained.json`). The unmasked path remains only for packages that do not set the capability. Row-bounded multi-round prefill still computes and samples each intermediate chunk tail even though only the final prompt result is returned. | Packed linear/MoE stays multi-row and full attention preserves slot-local arithmetic across bounded rounds. On gfx1151, a clean C8/512 six-round gate preserves tokens, positions, hidden seeds, and final outputs while skipping 40 intermediate chunk-tail samples; matched tracing changes the Q6 LM head from six launches/34.599 ms to one/5.756 ms and removes 70 dispatches. A/B/B/A pooled complete prefill wall improves 0.149%/0.219%/0.217%/0.033% at C5-C8, so this is retained as an exact launch/sub-window reduction rather than a standalone throughput claim. E1/E2/E3 cover c8 and C13 eager/graph, ragged, sparse masks, cancellation/admission, and exact token/hidden/Conv/GDN/live-KV state. Each gfx11 F1 packet sends 512 exact prompt IDs/request through real prefill work and preserves all **189** prompt rows, usage counts, and outputs. | Keep the unmasked path for packages that do not set the capability; the W7900 transfer is now qualified and no longer a reason to retain it. Remove intermediate chunk-tail output-norm/LM-head sampling after a final-slot mask preserves hidden-seed/MTP contracts and is profiler-non-regressive. Keep the opt-out for one release window, then remove it during F2 on both gfx11 targets while retaining scalar fallback only for unsupported shapes. **2026-08-30 gate note:** the byte-exact `scripts/gguf_packed_ar_state_oracle.py --prefill-mode packed` result (`initial_state_exact=false` at rows 2..8, `tokens_exact=true`) cannot qualify or displace this route - the candidate persists `segmented_in_place_final_state` while its teacher persists per-token-exact state, and `scripts/gguf_state_slot_hash_match.py` shows 0 permuted of 126 groups, i.e. no slot-index defect. The missing evidence is a production-profile KL/top-1 run whose candidate admits prompts through packed prefill. `scripts/execution_profile_gguf_batch_route_gate.py --packed-prefill-candidate` (added 2026-08-30, `worklog/entries/20260830T010758.358171Z-lhl-w7900-qwen38-packed-prefill-profile-gate-b6c739.md`) now provides that admission mode, and the canonical 18-prompt suite at widths 3/5/6/7 passes the numeric production gate (`status: complete`, 1950 rows, per-category KL_mean 3.6e-5..1.2e-4, top-1 0.9906..1.0, the four top-1 rows being one near-tie position replayed across the width scenarios; `benchmarks/results/2026-08-30-w7900-qwen38-q4km-packed-prefill-profile-gate-c3567.json`). C4/C8 also pass under the q8t16_candidate protocol (attribution caveated: it forces `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` on the candidate; `benchmarks/results/2026-08-30-w7900-qwen38-q4km-packed-prefill-profile-gate-c4c8.json`), leaving width 2 as the only packed serving width with no capture, because neither width protocol admits it. Remove the flag only once those widths are covered AND the control-plane blockers (runtime-profile-resolved route manifest, complete exact control-plane record, fresh task and BF16-relative verdicts) are closed; until then keep it explicit and default-off, because the mode intentionally skips the byte-exact prefill calibration. See `worklog/entries/20260830T011300.377155Z-lhl-w7900-qwen38-packed-prefill-profile-gate-full-74863b.md`. See `worklog/entries/20260830T005330.724870Z-lhl-w7900-qwen38-packed-prefill-state-gate-3df834.md`. |
| GGUF fair prefill burst rollback | `HIPENGINE_FAIR_PREFILL_BURST_CHUNKS` and `--fair-prefill-burst-chunks` retain strict one-chunk alternation as an explicit rollback around bounded fair-prefill bursting. | gfx1151 Q4_K_M defaults to at most two consecutive 256-token chunks only while at least two prompts still need prefill; lone staggered arrivals keep one-chunk alternation. The static p512/C8 host contract removes six duplicate partial-width ticks, accepted real-Uvicorn C8 improves over the retained exact SSE/live rows, and `continuous_fixed` moves ITL p99 **0.5068 -> 0.2949 s** while all 12 rows and SLO checks pass. Other backend/quant packages remain at one chunk. | Remove the env/CLI rollback after one release window and a defaults-only full F4 production matrix plus C1/C2/C4/C8 packet remain exact, SLO-clean, and non-regressive. Keep workload-derived pressure/fairness logic rather than backend branches in the scheduler. |
| GGUF MTP server packed prefill | `HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL` is a default-on opt-out around packed prompt prefill for eligible c=2/c=4 GGUF MTP serving batches. | Default-on after the 2026-07-06 steady-state natural24 rerun. The path reuses packed prompt rows and returns FP32 prompt hidden rows for MTP catch-up, moving server MTP **46.75/49.65/52.18 -> 59.94/66.60/54.88 tok/s** at c=2/c=4/c=8. It keeps the four-slot safety cap: c=8's first wave still uses serial prompt open and only the trailing c=2 wave uses packed prefill. Startup now warms hidden-seed packed prefill at widths 2/4 when MTP serving is enabled, moving fresh c=2 to **56.59 tok/s** and warm c=2/c=4 to **59.71/65.57 tok/s**. | Keep the opt-out until one more c=2/c=4/c=8 rerun confirms the default. Do not remove the four-slot cap until c=8 full packed prefill is non-regressive; pool-filling eight startup slots was rejected (**35.25 tok/s** c=8 rerun, **76.5 GiB** used). |
| GGUF MTP server startup warmup | `HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP` is an internal server-scoped env marker set only during startup scratch probing when `--speculative-mtp-serving` is not `off`. | Added after the 2026-07-06 cold-start audit. It lets the GGUF backend warm MTP hidden-seed packed prefill plus one tiny packed verifier at supported widths 2/4 without changing the generic `prepare_request_scratch(...)` hook signature. It removes the worst c=2 first-request MTP cliff but deliberately does not attempt unsupported width-8 packed prefill. | Replace this env handoff with an explicit backend scratch-preparer option if the startup hook grows typed capabilities. Keep it while MTP serving is opt-in/auto and c=2/c=4 cold-start evidence remains positive; remove or narrow it if startup memory/time becomes a production blocker. |
| GGUF MTP server deferred verifier scatter | `HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER` is a default-on opt-out around delaying packed target verifier state scatter until after the accept decision. | Default-on after the 2026-07-06 resetfix rerun. It keeps owner-side packed verifier state live and commits only accepted hidden/full-attention KV/linear-state rows, moving retained no-env natural24 MTP **70.06/77.29/76.46 -> 70.53/78.76/79.61 tok/s** at c=2/c=4/c=8 with unchanged economy (`draft=165`, `accepted=141`, accept rate **0.8545**, **250** target rows). Reset now invalidates packed verifier/decode session metadata so startup verifier warmup cannot leave stale packed KV write-position bookkeeping for the first real request. | Remove the env opt-out after one more no-env c=2/c=4/c=8 server rerun plus API tests show no regression. Keep the eager-scatter branch only while it is useful for bisecting packed verifier state lifecycle bugs. |
| GGUF llama-compat native MTP cycle | Benchmark flags `--native-spec-target-cycle`, `--native-spec-device-accept-commit`, `--native-spec-complete-cycle`, and `--native-spec-proposal-graph` retain N1 target-only, N2 device-policy, N3 complete-adapter, and N3P proposal-submission boundaries over separate reusable B1/B2 graph buckets. | Explicit accuracy-traded gfx1100 scope. N1 remains the 122.667 tok/s topline over llama.cpp's 115.444 floor. N2 owns strict acceptance, selected hidden/Conv/GDN state, target cursor, and one bounded payload. N3 makes proposal, N2, MTP-KV repair, reseed, and cursor/result accounting one public single-request adapter call; its clean committed gate is exact for 240 IDs / 96 cycles and aggregate-neutral at 118.592 vs clean N2 117.557 tok/s (+0.88%). N3P stages dynamic proposal inputs into fixed runner buffers and replays one B1/B2 NextN graph before the unchanged target graph. A detached clean publication is exact at 118.183 tok/s / 8.610 ms-output; a same-source pair is 117.589 N3P vs 116.793 N3 tok/s, and cached eight-cycle tracing replaces 542 `hipLaunchKernel` plus 80 synchronous `hipMemcpy` calls with eight proposal `hipGraphLaunch` calls. N3P still has two native graph submissions per public cycle, not one combined submission, and gfx1151 remains unregistered. | Migrate the provider-neutral adapter to PARO/DFlash (N4), independently gate gfx1151, and decide whether combining proposal+target behind one native boundary yields a measurable gain. Remove benchmark-only N1/N2/N3 splits after cross-provider full-suite/profiler gates. Keep independent B1/B2 buckets and exact unsupported-shape fallback. |
| PARO MTP / DFlash native target graph | `HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH=1` explicitly routes eligible shared single-request B1/B2/B3/B4/B5/B8 target graph replays through the registered gfx1100 `w4_paro/native_v1_target_graph` NativeSpecCycle boundary. `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT=0` temporarily rolls capture-width-zero FP16 PARO MTP back from selected linear-state commit/device cursor ownership to the neutral N4+ provider commit. | N4 remains globally default-off. Inside explicit N4, PARO selected commit is default-on after three exact 240-ID arms, every split/category, accepted-row/B2 state gates, and consistent capture-adjusted wins. It uses graph-owned combined state-pointer tables and declares `VERIFY|ACCEPT|COMMIT|UPDATE_CURSORS`; BF16 DFlash hidden/KV repair, proposal, and scheduler result construction remain unchanged. Cached profiles mechanically change **80.6875 -> 75.6875 APIs**, **2 -> 1 sync**, **36.1875 -> 34.1875 host launches**, and **1248.5 -> 1247.5 kernels**; mean wall is neutral. Pointer/shape/stream/inactive drift retains validation/direct fallback, and gfx1151 remains unregistered. No AR speed row/global default is retained. | Remove `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT` after one release window and a defaults-only explicit-N4 refresh remains exact and mechanically non-regressive. Keep the global N4 switch until complete PARO proposal ownership and independent full category+heldout PARO/DFlash gates have a measured complete-wall advantage over direct graph control; gate gfx1151 separately. Remove the bound-control experimental split once that provider boundary is the only admitted implementation. Do not remove or downgrade the current model artifact. |
| GGUF AR server stream decode | `HIPENGINE_GGUF_AR_STREAM_DECODE` is a default-on fallback/opt-out around per-slot HIP stream decode when packed AR is disabled or unavailable, and also gates the historical parallel c4+c4 chunk path above four rows. | Direct physical-c8 beats c4+c4 on both targets. F1 measures the same-loop packed-off serial C13 bridge at **31.708/42.764 tok/s** versus grouped packed C13 **111.380/72.522 tok/s (3.513x/1.696x)** on gfx1100/gfx1151; every static/live packed route records zero serial/resident fallback, including real c8→c13 admission. | Both gfx11 triggers are met. Keep the bridge as an explicit unsupported-shape/test oracle and rollback for one release window, then remove the public env opt-out during F2 while retaining backend capability fallback. |
| GGUF MTP server stream draft | `HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT` is a default-on opt-out around per-slot HIP stream draft proposal for c>N GGUF MTP serving. | Default-on after the 2026-07-05 natural24 server run. It moves packed-verifier c=2/c=4/c=8 MTP **45.57/47.48/47.18 -> 46.75/49.65/48.72 tok/s** and beat the then-current stream-AR rows by **1.058x/1.063x/1.021x**; the later packed-AR route supersedes that same-server AR comparison, so current MTP is below AR. It currently creates a `ThreadPoolExecutor` per draft phase and the c=8 wall remains verifier-heavy. | Remove the env opt-out and/or replace the per-cycle executor with a persistent scheduler after c>N server reruns and profiler evidence show the stream path is non-regressive. Keep the opt-out while tuning verifier wall and thread-pool overhead. |
| GGUF MTP server stream verify | `HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY` is a default-on opt-out around running independent chunk-4 packed target verifier chunks on separate owner-session streams at c>N. | Default-on after the 2026-07-05 natural24 c=8 rerun. It keeps the four-slot verifier cap but overlaps the two c=8 chunks, moving warm c=8 MTP **48.72 -> 52.18 tok/s** and `slots_verify_phase_ms` **15354.902 -> 12345.442**. c=4 remains on the single-chunk path. | Remove the env opt-out after c=2/c=4/c=8 reruns and profiler evidence show the stream verify path is non-regressive; replace it if a tuned rows>=16 verifier or true batched scheduler supersedes chunk-stream overlap. |
| GGUF row-shaped target executor | `Qwen35GGUFResidentSession.step_rows()` still loops physical slots through the retained c=1 layer path; `verify_rows()` likewise advances each linear-chain row serially. They are explicit correctness/speculative bridges rather than the default independent-decode route. | UD-Q3_K_M C=2/4/8 uses native indexed Conv/GDN, batched `KVLiveSpans` attention, selected-row MoE, row lm-head/argmax, and shape-specific HIP graphs; synchronous greedy prompt lists select the scheduler-owned native path. The serial executor remains the independent layer/full-logit oracle. Task #31's transactional MTP verifier deliberately reuses the exact c=1 row arithmetic and journals each row rather than weakening correctness. | Keep `step_rows()` as an explicit unsupported-shape/bisection oracle. Replace the eager serial MTP chain only when a native dependency-aware root+candidate forward preserves the B=1/2/3 scalar-logit and state/KV commit gates and is same-suite non-regressive; do not remove unfused numerical fallbacks. |
| GGUF MTP eager verifier | `Qwen35GGUFTransactionalVerifier` retains the `serial_exact` target chain as the correctness oracle; its stable full-shape buffers are keyed by scheduler graph shape and `captured=false` remains explicit. Dense public MTP now bulk-prefills the target once, retains every trunk-hidden row on device, and catches NextN up with the shifted `(zero, h0, ..., h[n-2])` sequence. | Correctness route and current dense public fallback. Candidate-only/shared ABI, verify-chain spans, GPU accept, rollback, reject/partial/full commit, exact greedy output, and the 601-second 408-request all-category public soak pass. Native dense rows after bulk prefill remain blocked as documented above; the older serial-prefill native protocol is separately gated. | Keep `serial_exact` permanently as the unfused transaction oracle. Promote native verification for public bulk-admitted dense MTP only after full-suite output/state and reject/partial/full gates pass from the same prefix and complete wall is non-regressive. |
| GGUF prompt-list scheduling | The promoted scheduler is synchronous and greedy-only: it does not persist across `generate()` calls, serve HTTP arrivals, cancel/disconnect rows, share prefixes, grow/shrink an elastic KV pool, or batch non-greedy sampling. Prefill into a reclaimed slot is still per-request rather than a packed multi-prompt slab. | Stable request ids, 2–8 physical slots, EOS/length reclaim, downward recurrent-state/live-KV compaction, mixed readmission/prefill between decode steps, C/context/mask graph buckets, row argmax, and admission/completion timestamps are retained and exact. Fixed C=2/4/8 scaling is accepted; one in-call 4-request/capacity-2 schedule exercises three admission waves, three compactions, and two readmissions with `serial_decode_fallback=false`. | Move this ownership into the persistent server engine loop, add cancellation/disconnect and elastic/prefix-shared KV, then native per-row non-greedy sampling and packed mixed-prompt prefill. Keep the synchronous path until the persistent route passes the same exactness/provenance gates. |
| UD-Q3_K_M grouped raw-IQ prefill | `HIPENGINE_GGUF_IQ_GROUPED_PREFILL=0` is the temporary rollback around the default-on expert-major scalar IQ3/IQ4 route; direct selected kernels remain the required fallback. | Promoted after mixed-prompt native-row-bulk parity (`KL=0`, top-1 `1.0`), exact 512/4K trajectories, zero scratch/copies, raw-IQ time `994.668 -> 613.995 ms` (-38.27%), and total kernel sum `4396.145 -> 4078.667 ms` (-7.22%). The formal 512/128 headline is correctly labeled flat within spread (`16.648 -> 16.685 tok/s`, +0.22%); promotion retains the verified sub-window win, not a noisy topline claim. WMMA and scalar RT2 remain rejected/test-only. | Keep the opt-out for one rollback/bisection window through task #20/#15 D0. Remove the env branch after the next defaults-only 512/4K milestone rerun confirms the grouped symbols and direct fallback tests remain healthy. |
| GGUF MoE-tail plus next-input RMSNorm | `HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS=0` is the temporary rollback around the default-on decode chain. The retained route fuses only the already-rounded selected-aggregate ABI; slot-weighted layers preserve the feature-parallel combine followed by the exact unfused RMSNorm because the one-block weighted composite regressed cold production-layer time. | Q3 GPU1 counterbalanced graph decode improves `100.195 -> 101.216 tok/s` (+1.02%) at 512/128 and `107.366 -> 108.383 tok/s` (+0.95%) at 4K/128, with all five paired deltas positive, exact layer-limit 1/4/40 buffers, exact eager/graph trajectory, unchanged final IDs/logits/memory, and no bulk-prefill route change. Q4/PARO slot-weighted paths keep their existing two-kernel math. | Keep the opt-out for one Q3/Q4 defaults-only milestone and task-#16 overlap bisection window. Then remove the env branch while retaining the registered unfused combine→RMSNorm fallback and kernel-level BF16/FP16 tests. |
| GGUF c1 host token embedding | `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` offloads the Q8_0 token embedding from device residency and performs exact host Q8_0→BF16 embedding copies. | Retained exact c1 opt-in, global default off. The old gfx1100 `128K/128` capacity proof fit at `23.400/23.913 GiB` tracked/sampled but ran only `11.141 tok/s`. Current gfx1151 SH2-M1 instead passes complete byte-state and stays within 1% of graph/eager at 512/4K while removing exactly `0.503235 GiB` live ownership and `0.443409/0.503933 GiB` whole-GTT. The short load still materializes then frees the device table, so tracked high water saves only `0.418797 GiB`; shared c>N, packed AR, and MTP require device-token-pointer embedding. | Replace allocate-then-free with loader-time ownership only after an explicit shared-runner-safe c1 policy automatically preserves the device table for c>N/packed/MTP/unsupported sessions. Then rerun exact c1/c>N/MTP lifecycle and performance gates before considering a gfx1151 default. Keep the env route until that policy lands or another exact memory owner supersedes it. |
| GGUF INT8 KV diagnostics | GGUF accepts explicit `--kv-storage int8_per_token_head` for resident full-attention KV, reusing the PARO per-token/head INT8 write/decode kernels. A geometry/policy byte planner now selects full-prompt layer-outer execution with one reusable BF16 prefill-oracle pair when that is smaller than chunk-outer per-layer oracles; a fail-closed guard forbids shared ownership under chunk-outer execution. Short contexts (`<=8192` rounded max positions) retain an additional BF16 mirror. Long contexts default to the 35B-safe `HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS=8`; lower prefixes, pure INT8, key-only, block16, and custom masks still require `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1`. | Default remains BF16 unless explicit INT8 is requested. Qwen3.6 pure FP32-scale INT8 still fails complete 512/8 at `77.78%` minimum-prompt top-1. On gfx1151, Qwen3.6-35B-A3B `UD-Q4_K_M` with zero BF16 mirror passes 512/8 and 1024/8 (fp16 scales) and fails 4096/16 on the `mixed_v1` control alone at mean KL `0.1597`, while all ten natural category prompts pass at 100% top-1 and mean KL <= `1.09e-3`; its runtime gain is `+1.44%/+1.50%/+1.59%` against the BF16 graph route at depths 0/4096/16384, so an INT8 KV copy is not the depth-sensitive lever and the gfx1151 artifact stays BF16. Exact-file Qwen3.8 pure FP32-scale INT8 passes complete 512/8 and 4K/16 plus bounded `mixed_v1` 64K/16. Its bounded owner lowers 32K-capacity tracked peak `18.943 -> 17.330 GiB` versus the old INT8 route and `17.920 -> 17.330 GiB` versus BF16; matched prefill is `-0.050%` with overlapping sample ranges and eager AR decode is `+6.499%` versus BF16 graph. Real 24-GiB single-request server rows pass at 64K/96K/112K with `3.420/1.570/0.662 GiB` headroom; 120K fails the 1-GiB startup guard and 128K OOMs. Pure-INT8 graph capture remains fail-closed, and exact natural B3 is only `0.6423x` true AR. | Keep BF16 supported/default and retain pure Qwen3.8 INT8 as an explicit AR capacity route. Remove the unverified-long gate only after a model-plugin capability distinguishes Qwen3.8 from quality-rejected Qwen3.6 without filename/backend branches and representative quality passes at the intended ceiling. Graph-admit pure INT8 only after exact eager/graph state and throughput gates; do not enable INT8 MTP until full natural categories are non-regressive. Remove the short mirror, custom-mask/key-only/block16 branches when superseded by one supported calibrated layout. Do not add two-arena temporal storage. |
| GGUF selected-prefill diagnostics | `HIPENGINE_GGUF_T16_DS4_PREFILL` guarded runtime route for resident `gguf_q4_k_t16_v1` DS4/Q8_1 selected-prefill. | Default-off diagnostic. Full-model Q4_K_S GPU1 gate showed useful prefill speed (`1833.185 -> 1989.578 tok/s` at `512/128`, `2159.561 -> 2372.228 tok/s` at `4K/128`) but changed final token IDs versus default (`220/570 -> 3241/1510`) and added `+0.070 GiB` opt-in activation scratch. The scratch is allocated only when the flag is enabled, so default memory/IDs remain unchanged. | Remove or demote to a microbench/test-only path unless a later exact-enough Q8_1/DS4 calibration path preserves default final IDs/logits on `512/128`, `4K/128`, and the `128K/128` promotion gate while keeping memory bounded. |
| GGUF selected-prefill diagnostics | Microbench-only raw-Q4_K/Q8_1 selected-prefill variants in `gguf_q4_k_q8_1_selected_prefill` (`q8-1-dot`, `q8-1-ds4-dot`, `q8-1-ds4-wmma`, `q8-1-ds4-wmma32`, `q8-1-ds4-wmma64`, `q8-1-ds4-preview-wmma32`, `q8-1-ds4-wmma32-ldspack`, and rejected `q8-1-ds4-wmma32-lds`). | Diagnostic-only, not model runtime defaults. The 2026-06-16 DS4 WMMA path is useful as a fragment/math reference. Expanded-Q4 LDS staging regressed `8.210 -> 18.257 ms/call`; packed-Q4 LDS staging recovered to `11.438 ms/call` but still lost to raw WMMA32; the pre-unpacked preview path measured `12.020 ms/call` with higher fixture memory; four-wave WMMA64 was only flat/sub-1% better than WMMA32. These same-shape staging/pre-unpack/independent-wave probes should not become dispatch paths without a later shared-tile reuse win. | After a real GGUF MMQ/T16 prefill path is promoted or the llama.cpp parity detour closes, demote negative variants to tests or remove them from the microbench to avoid a permanent zoo of rejected kernels. |
| Qwen3.5/PARO INT8 KV prefill | `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION`, `HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS`, `HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB`, and `HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB` gate direct streaming INT8 prefill after the 2026-06-15 GPU1 sweep found a severe 128K prefill regression. | Default `auto`: use the fast temporary BF16-oracle/AOTriton bridge unless prompts are at least `224Ki` rows **and** device/meminfo pressure says the oracle is unsafe (`total <= 26 GiB` or free memory cannot cover oracle bytes plus a 1 GiB reserve). This keeps W7900/Strix-style larger-memory runs on the fast path while preserving the 24GB 262K scratch win. Direct streaming is not throughput-promoted (`1020.723 -> 23.425 tok/s` at 128K/128). | Remove the gate after a memory-safe fast INT8 prefill path matches the BF16-oracle/AOTriton speed envelope at `512/128`, `4K/128`, and `128K/128` while retaining the 262K no-oracle memory gate; otherwise demote direct streaming to a dedicated diagnostic/scratch-probe path. |
| Qwen3.5 GGUF/PARO tail-four mixed KV | Explicit `tail4_hadamard_group32` policy plus the BF16-attention/packed-retention prefill bridge; PARO keeps `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION=streaming` as the direct-INT8 diagnostic. | Explicit-only and **not supported/default**. Clean therock-7.15 35B GGUF gates pass all 11 prompts at 512/8 and 4K/16 plus bounded `mixed_v1` at 128K/16, save exactly 18.75% persistent K/V, and retain zero persistent BF16 shadow, but reject on prefill/decode and allocator high water. Exact-file Qwen3.8-27B now also passes complete 512/8 and 4K/16 plus bounded 32K/16. Its graph decode improves `2.58%` and live ownership falls `0.236 GiB`, but prefill falls `0.342%` and four 32K oracle pairs raise tracked peak `0.268 GiB`; binding the retained cache directly to the no-oracle Hadamard prefill path produces non-finite logits. Native PARO separately fails its quality/capacity gates. | Keep as a reproducible diagnostic. The geometry/policy planner now selects a bounded shared-oracle owner for memory-positive GGUF tail-four layouts, but this qualification reran only pure INT8; rerun complete tail-four quality, server high-water, performance, graph, and natural MTP before changing its status. PARO still requires a quality-safe layout and physical capacity pass. Remove rejected direct/oracle plumbing only after a promoted design supersedes it. |
| Qwen3.5 GGUF/PARO native sampler | `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` is the PARO rollback opt-out; `HIPENGINE_QWEN35_NATIVE_SAMPLER=1` enables the separate GGUF correctness candidate. | PARO remains default-on for supported c1 and scheduler-owned c>N serial-per-slot rows. GGUF is explicit-only: supported c1 and dense compatible c>N rows use native row/rows selection with zero full-vocabulary D2H; forced/dynamic/unsupported combinations remain host-backed. The W7900 c4 correctness artifact passes fixed-seed/state/KV/finish/ownership gates, but A3 stops before timing: native-eligible auto tools fail the frozen turn-1 strict-envelope oracle, while 4/4 exact forced-tool turns report `host_logits_sample` / `native_gpu_unsupported_request`. No active-SSE/tool-ready claim exists. | The A3 non-promotion trigger has fired. In a dedicated cleanup unit, remove the GGUF env branch and resident workspace wiring while retaining the standalone sampler/kernel correctness gates, unless a model-general native forced/dynamic-queue or constrained auto-tool design is approved and separately correctness-gated first. Separately remove the PARO opt-out after its release window. |
| GGUF MoE decode graph | `HIPENGINE_GGUF_MOE_GRAPH` opt-in around per-layer rows==1 MoE FFN capture/replay (`hipengine/runtime/moe_graph.py` + `_run_decode_layer_graphed`). | Default-off. Proven bit-exact on 35B-A3B Q4_K_M (KL=0, 40 captures / 3800 replays / 0 rejects). Cuts the FFN ~64% in launch count (~440->40/token) but multi-trial (3x32 steps) shows a **consistent ~0.84% wall regression** (eager 18.12 vs graph 18.27 ms/step, non-overlapping p10-p90 bands) — NOT noise. The host was already ahead of the GPU (decode is bandwidth/compute bound), so removing launches reclaims no wall and the graph launch overlaps marginally worse with the surrounding eager attention. Fails the non-regressive gate, so NOT promoted. Kept as the validated A/B lever proving launch-count is not the decode bottleneck. Artifact `benchmarks/results/2026-06-28-moe-graph-rows1-ab.json`, WORKLOG 2026-06-28. | Remove the flag + `MoeGraphCache` wiring once the bandwidth-bound conclusion is accepted and no rows=4 verifier-graph follow-up is planned. Re-A/B (and only then reconsider default-on) if a future per-GEMV bandwidth cut shifts the bottleneck back to host-dispatch and makes the launch saving net-positive on wall. Keep `moe_graph.py` + its unit gate as a reusable tool even if the runner wiring is removed. |
| GGUF row-compact MoE GEMV | `HIPENGINE_GGUF_ROW_COMPACT_GEMV` opt-in around `_try_run_post_attention_moe_rows_compact_gemv` for rows>1 selected-MoE verifier blocks. | Default-off. Rechecked on 2026-07-01 against the current llama-compat B2 dp4a all-sync smoke after direct-state cleanup. It regressed badly: B2 **36.05 tok/s**, cycle **27.765 ms/output**, `target_block_verify_total` **24.277 ms/output**; the new `target_block_linear_attn_ffn_moe_compact_gemv` bucket alone cost **8.977 ms/output**. Current split selected-MoE GEMVs are faster for this verifier shape. Artifact `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-rowcompact-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the runtime env gate or demote the compact row-GEMV path to tests/microbench-only unless a new compact scheduler/kernel beats the split selected-MoE path on a full-suite `llama-compat-device-chain-dp4a` B2 run with unchanged acceptance. |
| GGUF dense Q8 dp4a sidecar | `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR` materialization sidecar plus `HIPENGINE_GGUF_DENSE_Q8_DP4A` / `--verify-dense-q8-dp4a` and `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` / `--verify-dense-q8-dp4a-all`, routed by `_try_launch_dense_q8_pair_dp4a`, `_try_launch_dense_q8_single_dp4a`, and `_try_launch_dense_q8_triple_dp4a` for rows>1 verifier blocks. | Default-off. Added for the llama.cpp replication lane. The original route paid a q8_1 quantize launch plus two singleton dense Q8 GEMV launches and lost on B2 smoke; the rowtile-pair retry improved smoke/all-sync verifier timing but full-suite regressed **60.36 -> 59.42 tok/s**, cycle **16.587 -> 16.852 ms/output**, acceptance **0.583 -> 0.559**, target rows/output **1.250 -> 1.322**, and verifier drain **13.023 -> 13.093 ms/output**. The broader all-sidecar route adds raw singleton and Q/K/V triple wrappers and cuts the block profile dense-Q8 bucket **11.420 -> 8.902 ms/block** / kernel **26.053 -> 23.427 ms/block**; full-suite improves speed **60.36 -> 60.89 tok/s** and verifier drain **13.023 -> 12.742 ms/output**, but acceptance regresses **0.583 -> 0.567** and draft acceptance **0.700 -> 0.655**. Later retained lanes add Q8 shared dual, X8 draft lm-head, and F32 `ssm_out` on top of this all-sidecar base. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-{smoke,full}.json`, and earlier denseq8 rowtile-pair artifacts, WORKLOG 2026-07-01. | Keep only as part of the named accuracy-traded llama-compat route while parity work is active. Remove loose env/bench/suite variants after the current llama-compat audit unless a true llama-style Q8 layout/scheduler beats the active compat lane on the full suite, or unless the compat acceptance contract is explicitly changed. |
| GGUF verifier F32 dense-Q8 dp4a diagnostic | `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` / `--verify-dense-q8-dp4a-f32` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm{,-allsync}` route rows>1 direct-state F32 `ssm_out` through F32 q8_1 quantization plus the raw-Q8 dp4a singleton body. | Default-off globally; retained only for the accuracy-traded llama-compat lane. Isolated block profile moved host **32.470 -> 30.936 ms/block** and kernel **23.893 -> 22.881 ms/block**; same-session smoke moved **70.74 -> 71.43 tok/s** with identical acceptance; full-suite B2 moved **61.31 -> 63.63 tok/s**, cycle **16.331 -> 15.735 ms/output**, verifier drain **12.662 -> 12.158 ms/output**, acc/output **0.567 -> 0.578**, and target rows/output **1.299 -> 1.266**. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-f32ssm.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-{control-smoke,smoke,full}.json`, WORKLOG 2026-07-01. | Keep only as part of the named compat route while the safe verifier transaction gap is audited. Do not promote to exact default unless an exact/non-regressive replacement exists. Collapse this flag behind the final named compat route or remove it during post-compat cleanup if a later verifier rewrite supersedes direct F32 q8_1/raw-Q8 dp4a. |
| GGUF verifier shared-Q8 dp4a diagnostic | `HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED` / `--verify-dense-q8-dp4a-shared` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-sharedq8{,-allsync}` route verifier shared-expert `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp` through the raw-Q8 q8_1/dp4a helpers. | Default-off and rejected on the then-active llama-compat B2 lane. Isolated block profile moved kernel time **23.893 -> 23.648 ms/block** and smoke improved **70.64 -> 71.66 tok/s**, cycle **14.181 -> 13.978 ms/output**, verifier drain **11.377 -> 11.183 ms/output** with identical smoke acceptance. Full-suite rejected it: then-active `denseq8all-x8top1` **61.31 tok/s**, cycle **16.331 ms/output**, acc/output **0.567**, target rows/output **1.299**, verifier **12.662 ms/output**; sharedq8 **59.63 tok/s**, cycle **16.793 ms/output**, acc/output **0.556**, target rows/output **1.333**, verifier **13.038 ms/output**. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-{refresh,sharedq8}.json` and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-sharedq8-{control-smoke,smoke,full}.json`. | Remove the env/bench/suite route during post-compat flag cleanup unless a later fused shared-expert body or launch-collapsed shared route beats the active compat lane on the full suite with unchanged acceptance/economy. Do not promote this per-projection q8_1/dp4a shared path. |
| GGUF resident MTP draft Q8 shared dual | `HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL` opt-out around the default-on raw-Q8 dual F32/F32 GEMV for resident draft shared gate/up projections. | Default-on. Added 2026-07-01 for the llama-compat lane and exact resident draft path. It is bit-exact vs two single `gguf_q8_0_gemv_f32_f32_out` launches (`tests/test_gpu_gguf_k_gemv.py::test_q8_0_dual_f32_matches_two_single_gemvs`). Draft rocprof A/B reduced `gguf_k_prefill_out` from 16 -> 12 calls/cycle and added `gguf_k_dual_prefill_out` 2 calls/cycle; same-session smoke improved **69.44 -> 70.20 tok/s** with identical acceptance, and full-suite llama-compat improved **60.96 -> 61.19 tok/s** with unchanged acceptance/economy. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-{control,dual}.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-full.json`, WORKLOG 2026-07-01. | Remove the opt-out branch and make the dual call unconditional after the next full-suite default exact and semantic-safe llama-compat parity reruns stay non-regressive, unless a later draft rewrite supersedes the shared-expert path. |
| GGUF resident MTP draft dense-Q8 dp4a stage selector | `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` plus `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES` / `--resident-mtp-draft-dense-q8-dp4a-stages` route resident draft F32 dense projections through F32->q8_1 plus raw-Q8 dp4a float-output wrappers by stage. | Default-off globally and retained only in the accuracy-traded llama-compat lane with `stages=draft`. The legacy all-stage route, including initial KV seeding stages, regressed full-suite B2 **64.41 -> 64.14 tok/s** with worse acc/output and target rows/output. The draft-only selector preserved row economy and moved the then-active unsafe direct-state compat row **74.39 -> 75.15 tok/s**, cycle **13.463 -> 13.325 ms/output**, and `draft_initial` **2.204 -> 2.066 ms/output** with unchanged acc/output **0.621**, draft acceptance **0.820**, and target rows/output **1.136**. That performance row is now superseded as an exact-state claim. The current llama-style directcommit replication row is **60.56 tok/s**, cycle **16.534 ms/output**, verifier drain **14.071 ms/output**, replay/commit **0.043 ms/output**, target rows/output **1.172**, and zero replay rows; the serial-state exact control remains **51.85 tok/s** / **19.308 ms/output**. Artifacts `benchmarks/results/2026-07-02-ar-mtp-llama-compat-draftdenseq8-draftonly-full.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`, `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-draftdenseq8-draftonly-gpuevents.json`, WORKLOG 2026-07-02. | Keep only as part of the named compat route while the llama-replication lane is under parity audit. Do not treat the unsafe 75.15 row as a cleanup/promote trigger. Collapse/remove the selector after the final compat route is settled or a verifier/draft rewrite supersedes this route. |
| GGUF resident MTP draft Q6 top-1 X8 sidecar | `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8` / `--resident-mtp-draft-q6-top1-stage1-shape x8` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1{,-allsync}` for an X8-packed Q6_K draft lm-head top-1 sidecar. | Default-off globally; retained only for the accuracy-traded llama-compat lane. It materializes `output.weight[:vocab]` into contiguous groups of eight GGUF Q6_K rows and routes the q8_1/dp4a top-1 stage1 through `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1`. Correctness passes against the q8_1/Q6_K oracle. Same-session smoke improved **71.53 -> 71.76 tok/s** with identical acceptance; draft rocprof moved stage1 **3.603 -> 3.558 ms/cycle**; full-suite compat moved **61.19 -> 61.31 tok/s**, cycle **16.364 -> 16.331 ms/output**, and `draft_initial` **3.378 -> 3.352 ms/output** with unchanged acceptance/economy. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-{control-smoke,smoke,full}.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1.json`, WORKLOG 2026-07-01. | Keep only as part of the named llama-replication route while the safe verifier transaction gap remains under analysis. Remove/demote the X8 sidecar and route variants if a later fused draft lm-head/sampler or different Q6_K body/layout supersedes it, or if parity closure decides the accuracy-traded llama-compat lane should not retain separate draft lm-head sidecars. Do not promote to exact default without exactness/full-suite correctness evidence. |
| GGUF selected-down X8 repack | `HIPENGINE_GGUF_SELECTED_X8_REPACK` materialization gate plus bench flag `--selected-down-x8-repack {off,q5,q6,both}` for Q5_K/Q6_K selected-down X8 q8_1/dp4a replacement layouts. | Default-off globally. Retained only for the accuracy-traded llama-compat B2 lane with `q6`; first-class suite route is `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6`. Full suite **59.63 -> 60.36 tok/s**, `cycle_wall_ms_per_output` **16.793 -> 16.587**, and `target_block_verify_total` **13.178 -> 13.023 ms/output**. q5/both remains rejected for that route (`64.81 tok/s` smoke vs q6-only `69.03 tok/s`), so Q5_K selected-down stays on T16. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-full.json`, `...x8q6-allsync-smoke.json`, route smoke `...x8q6-route-smoke.json`, and `benchmarks/results/2026-07-01-llama-compat-b2-x8-selected-down-dp4a-current-micro.json`. | Remove/demote q5/both materialization from performance paths unless a future full-suite route beats q6-only with unchanged acceptance. Do not promote to exact default without exactness/full-suite correctness evidence. Once the compat lane is final, consider collapsing the env gate behind the named route and leaving raw env use to tests/microbenches. |
| GGUF selected gate/up X8 repack | `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` materialization gate plus bench flag `--selected-gate-up-x8` and suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup{,-allsync}` for Q4_K selected gate/up X8 q8_1/dp4a replacement layouts. | Default-off and rejected on the current retained llama-compat B2 lane. Same-session smoke regressed **67.62 -> 59.08 tok/s**, cycle **14.810 -> 16.948 ms/output**, and target verifier drain **12.005 -> 14.117 ms/output** with identical smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). All-sync attribution shows the loss is the selected gate/up GEMV body: linear-attn gate/up GEMV **1.408 -> 3.050 ms/output** and full-attn gate/up GEMV **0.462 -> 1.015 ms/output**, while q8_1 quantize is unchanged/slightly lower. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup{,-control}-smoke.json` and `...x8gateup{,-control}-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the bench/suite route during the post-compat flag cleanup unless a different Q4 X8 scheduler/body beats retained T16 dp4a on the same async/full-suite route with unchanged acceptance. Future selected-MoE work should compare against llama.cpp `mul_mat_vec_q_moe` rather than broadening this X8 gate/up path. |
| GGUF selected gate/up raw materialization | `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW` materialization gate plus bench flag `--selected-gate-up-raw` and suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup{,-allsync}` for keeping Q4_K selected gate/up experts in raw GGUF layout under decode-repack. With `--verify-dp4a`, the runtime uses the raw selected-dual q8_1/dp4a body instead of the retained T16 replacement-layout body. | Default-off and rejected on the current retained llama-compat B2 lane. Same-session smoke regressed **68.55 -> 62.04 tok/s**, cycle **14.612 -> 16.142 ms/output**, and target verifier drain **11.792 -> 13.328 ms/output** with identical smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). All-sync attribution shows the loss is the selected gate/up GEMV body: linear-attn gate/up GEMV **1.422 -> 2.153 ms/output** and full-attn gate/up GEMV **0.461 -> 0.729 ms/output**, while q8_1 quantize changes by only ~0.01 ms/output. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup{,-control}-smoke.json` and `...rawgateup{,-control}-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the bench/suite route during the post-compat flag cleanup unless a new raw-GGUF scheduler/body beats retained T16 dp4a on the same async/full-suite route with unchanged acceptance. Do not retry a mechanical raw `mul_mat_vec_q_moe` body copy for selected gate/up; the measured path is slower than retained T16. |
| GGUF decode-graph rollback and benches | `HIPENGINE_GGUF_DECODE_GRAPH=0` keeps eager as a production rollback; `scripts/qwen35_gguf_bench.py --graph-replay-decode` remains an explicit measurement surface; `scripts/gguf_mtp_bench.py` still has separate `--target-graph-verify` / `--target-graph-batched-verify` modes. | gfx1100 defaults to the state-bound graph for non-streaming c1 greedy windows with at least 24 remaining transitions after clean W7900 p512/d24 SOL-G5 passed 24/24 state/token checks and moved capture-inclusive wall **30.5364 -> 12.5139 ms/token (2.4402x)**. The current gfx1151 one-queue recheck also retains graph: exact 128-token trajectories improve over eager **+1.00%/+0.86%** at 512/4K across 1+3 and **+0.36%** in a bounded 128K confirmation. The final production refresh publishes exact graph rows through 64K; its missing 128K topline is a prefill lifecycle blocker, not graph rejection. The gfx1151 resident BF16 owner now captures state-bound packed C2/C4/C8 graphs per complete wave and replays them for greedy blocking/SSE. Physical C8 improves blocking **75.702 -> 83.771 tok/s (+10.66%)**; low-width capture costs **2.24%/1.58%** blocking at C2/C4 but improves the primary SSE surface, so graph remains default-on with eager rollback. Sampled, short, INT8-KV, host-embedding, and per-layer-MoE-graph routes remain eager. MTP graph modes are not part of this result. | Keep the opt-out through one release window and the next matched gfx1100/gfx1151 refresh. Then make admitted graph routes unconditional if no release regression appears. Before removing the rollback, either reuse state-safe captures across low-width waves or otherwise eliminate their fixed capture tax without regressing SSE. Remove or isolate stale MTP graph modes separately. |
| GGUF AR-baseline timing contract | `gguf_true_ar_category_bench.py` and `scripts/gguf_ar_mtp_suite.py` request state-bound graph admission and record the effective per-prompt route. | Repaired for backend-qualified production timing: gfx1100 horizons >=24 report `graph_replay` with capture/instantiate/close included; unsupported/short horizons honestly report `eager_step`. This replaces the stale unconditional eager artifact while preserving an explicit no-graph diagnostic. The older attachment validator still has a fixed graph-only requirement and does not yet model backend/horizon admission. | Teach the attachment validator to consume the recorded backend capability/effective route instead of one global graph-only constant, then remove duplicate suite-side protocol logic. Preserve anti-gaming rejection of raw/non-production timing. |
| MTP P1 verifier | `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL` opt-out around the promoted split-output dual W4 shared-gate/up route. | Default-on after 2026-06-11 D32 9-prompt exact A/B: same acceptance, verify `22.98 -> 22.37 ms/cycle`. | After the next retained MTP gate with defaults-on passes at the target sprint shape, remove the opt-out or demote it to a test-only override. |
| MTP P1 verifier | `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED` opt-out around promoted `f32_to_fp16 + paro_rotate1` fusion. | Default-on after raw-bit RED test and 2026-06-11 D32 9-prompt exact A/B; removes 30 launches/pass and contributes to the stacked `-0.60 ms/cycle` suite delta. | After the next retained MTP gate with defaults-on passes, collapse the old runtime dispatch branch if no other path still needs it. |
| MTP P1 verifier | `HIPENGINE_SELECTED_MOE_DOWN_STAGED` opt-in around the superseded staged selected SiLU/down-rotate + down GEMV path. | Flipped default-off on 2026-06-11 after current graph-auto D32 9-prompt exact A/B: identical acceptance, cycle `27.648 -> 27.408 ms/cycle`, verify `22.377 -> 22.131 ms/cycle`. The 2026-06-12 graph-off current-best compound retest also stayed exact but regressed ratio `0.8252x -> 0.8204x`, cycle `21.661 -> 21.763 ms/cycle`, and verify `16.511 -> 16.628 ms/cycle`. The staged path remains available with `=1` for bisection and historical comparison. | After the next retained MTP gate with defaults-on passes, remove the staged runtime branch or demote it to a kernel test-only path unless a new barrier-free implementation beats the fallback. |
| PARO MTP provider target contract | `HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT` remains a manual-route compatibility selector around the qualified final-normalized/selected-reseed/borrowed-W8A16 provider. Registered production now binds it automatically; strict uses the same corrected provider with c1-loop verification. | Production/default after complete parity, 10-prompt/three-repeat numerical/task/state/lifecycle gates and fast D24 economics (`1.0446x` true AR, +10.33% vs strict MTP). | Remove direct env selection and the private F16/cap scorer after one release window with production/strict profile smokes. Keep corrected provider unconditional for registered B1 routes; preserve strict verifier fallback, not the old provider. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS` opt-out around skipped discarded expert-topk host reads, update-only lm-head/argmax results, and final draft snapshot saves. | Default-on after 2026-06-11 D32 9-prompt exact gates: same acceptance/visible tokens, read/result skip moved actual speed `0.664x -> 0.670x`, cycle wall `27.94 -> 27.68 ms`, proposal/update `2.145 -> 2.052 ms`; final-snapshot skip then stayed exact `9/9`, skipped `142` D2D snapshot saves, and trimmed proposal/update `2.052 -> 2.045 ms` with flat actual ratio within noise. | After the next retained MTP gate with defaults-on passes, remove the opt-out or demote it to a test-only override. Keep the functional code path; it is the desired proposer behavior. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION` opt-out around the packed token+position metadata H2D copy. | Default-on after 2026-06-11 same-tree D32 9-prompt exact A/B: exact `9/9`, identical acceptance, wall `26.922 -> 26.869 ms/cycle`, proposal/update `1.9766 -> 1.9758 ms/cycle`; ratio is noisy/down because the AR control changed. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep the packed one-copy metadata path. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT` opt-out around route-0 FP32 MoE accumulator initialization. | Default-on after 2026-06-11 D32 9-prompt exact A/B: exact `9/9`, identical acceptance, standalone `moe_accum` memset removed by route 0 overwrite, cycle wall `27.081246 -> 27.079143 ms/cycle`, proposal/update `1.96299 -> 1.95303 ms/cycle`; ratio is noisy/down because AR changed. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep route-0 accumulator initialization as the only proposer MoE accumulation path. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE` opt-out around direct sidecar K/V cache writes. | Default-on after 2026-06-11 D32 9-prompt exact A/B: exact `9/9`, identical acceptance, K rotary and V projection producers write directly into cache slots instead of temp buffers followed by two D2D copies per advance, proposal/update `1.9955 -> 1.9801 ms/cycle`; total wall was flat/noisy-negative because verify moved independently. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep direct cache writes as the only proposer K/V materialization path. |
| MTP/DFlash verifier accept | `HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD` opt-out around the packed accept-summary D2H payload. | Default-on after 2026-06-11 MTP D32 9-prompt exact same-tree A/B: exact `9/9`, identical accepted lengths and active budgets, cycle wall `27.279 -> 27.122 ms/cycle`, verify `22.162 -> 21.997 ms/cycle`; packs seven tiny D2H reads into one int32 payload read while keeping the legacy output buffers for commit/compatibility. | After one follow-up defaults-only MTP gate and one DFlash chain smoke pass, remove the opt-out or demote it to a test-only override. Keep the packed payload path as the default verifier accept API. |
| MTP/DFlash verifier metadata | `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA` opt-out around the packed token/position/context metadata H2D path. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths and active budgets, actual ratio `0.68417x -> 0.68898x`, cycle wall `27.02196 -> 26.99252 ms/cycle`, verify `21.87984 -> 21.85918 ms/cycle`; replaces five tiny per-cycle H2D submissions with one packed int64 copy plus `unpack_verify_chain_dynamic_metadata_i64_kernel`. Rocprof confirms the kernel; a 27B dense DFlash D16 one-prompt shared-path smoke passed. | After one follow-up defaults-only MTP gate and the next full 27B DFlash hardening/defaults-only gate pass, remove the opt-out and keep the packed metadata path. |
| MTP/DFlash verifier commit | `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` opt-out around the chunked linear-state commit copy grid. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths and active budgets, verify `21.8518 -> 21.8308 ms/cycle`; rocprof moved `linear_state_pair_commit` `0.250 -> 0.203 ms/pass`, total verifier kernel `14.395 -> 14.341 ms/pass`, and host marker `18.301 -> 18.263 ms/pass`. Whole-cycle wall was neutral/noisy, so this is retained as a verifier sub-window micro-slice. A 27B dense DFlash D16 shared-path smoke passed. | After one follow-up defaults-only MTP gate and one DFlash chain smoke/defaults-only gate pass, remove the opt-out and keep the chunked 64 KiB commit grid. |
| MTP verifier host cache | `HIPENGINE_VERIFY_SCRATCH_CACHE` opt-out around fixed-shape verifier scratch object caching. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates, wall `27.0958 -> 26.7015 ms/cycle`, verify `21.9328 -> 21.5511 ms/cycle`, and actual ratio `0.6860x -> 0.6987x`. Graph-auto profile showed only a small steady replay host change (`18.290 -> 18.275 ms/pass`), while graph-off control showed the raw Python rebuild win (`33.469 -> 32.988 ms/pass`). | After the next retained MTP defaults-only gate passes with scratch, tensor lookup, and resident view caches enabled, remove the opt-out and keep the workspace-validated scratch cache as the only verifier scratch reservation path. |
| MTP verifier host cache | `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE` opt-out around immutable model tensor lookup memoization on each decode state. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates, wall `26.6621 -> 26.6433 ms/cycle`, verify `21.5290 -> 21.4984 ms/cycle`, and actual ratio `0.69160x -> 0.69200x`. Graph-auto profile was neutral/noisy (`18.218 -> 18.236 ms/pass`), while graph-off isolated the raw Python lookup win (`34.757 -> 32.288 ms/pass`). | After the next retained MTP defaults-only gate passes, remove the opt-out or demote it to a test-only override; keep raw-name tensor lookup memoization as the default host path. |
| MTP verifier host cache | `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE` opt-out around resident non-owning Tensor view caching. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates and identical per-prompt accepted lengths, wall `26.6424 -> 26.4259 ms/cycle`, verify `21.5059 -> 21.2785 ms/cycle`, and actual ratio `0.69239x -> 0.69857x`. Graph-auto profile was neutral/noisy (`18.235 -> 18.244 ms/pass`), while graph-off isolated raw host improvement (`32.52 -> 31.70 ms/pass`). | After the next retained MTP defaults-only gate passes, remove the opt-out or demote it to a test-only override; keep cached `_slot_linear_state`, `_slot_full_cache`, and `_full_cache_all_slots` views as the default host path. |
| MTP verifier scratch policy | `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED` opt-out around c1/grouped verifier MLP scratch selection. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates plus identical accepted lengths/active budgets, wall `26.3089 -> 25.6898 ms/cycle`, verify `21.1757 -> 20.5228 ms/cycle`, and actual ratio `0.7003x -> 0.7172x`. Graph-auto profile kept `932` calls/pass and moved host `18.314 -> 18.246 ms/pass`; graph-off host moved `32.445 -> 32.273 ms/pass`. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep verifier MLP scratch reservation aligned with `_verify_moe_grouped_min_tokens()` as the only path. |
| MTP verifier host cache | `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP` opt-out around generation-stamped verifier scratch cache hits. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates plus identical accepted lengths/active budgets, wall `25.7085 -> 25.5955 ms/cycle`, verify `20.5460 -> 20.4342 ms/cycle`, and actual ratio `0.7145x -> 0.7252x`. Graph-auto profile kept `932` calls/pass and moved host `18.322 -> 18.298 ms/pass`; graph-off host moved `32.659 -> 31.971 ms/pass`. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep generation-stamped cache-hit validation as the only verifier scratch cache path. |
| MTP graph-off verifier scratch | `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY` opt-out around keeping verifier-shaped scratch live between MTP verify cycles. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths/active budgets, graph-off batched wall `37.207 -> 24.076 ms/cycle`, verify `32.069 -> 18.933 ms/cycle`, and actual ratio `0.4969x -> 0.7730x`. Rocprof showed this is host-only cleanup: calls unchanged `932/pass`, kernel `14.332 -> 14.330 ms/pass`, host `32.505 -> 18.272 ms/pass`. The follow-on `decode_batched + graph_off + skip` row is current best at `0.8252x`, `21.661 ms/cycle`, `16.511 ms` verify. | After the next retained MTP defaults-only gate passes and the c1/AR handoff path has explicit coverage for `canonicalize_after=True`, remove the env opt-out or demote it to a test-only override; keep the `canonicalize_after` API only where handoff semantics need it. |
| MTP verifier rejected gate | `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` opt-in for fused full-attention Q/Gate split plus FP16-to-FP32 key cast. | Default-off; bit-exact GPU parity vs `qwen35_split_qgate_fp16 + fp16_to_f32`, exact quicksort, and profile-positive for launch count (`932 -> 922` calls/pass, host `18.269 -> 18.234 ms/pass`), but two exact 9-prompt D32 A/B pairs with identical acceptance regressed average wall/verify (`26.925 -> 27.010 ms/cycle`, `21.754 -> 21.828 ms/cycle`). | Remove the runtime gate or demote it to a kernel test-only path after the break-even sprint unless a broader full-layer composite reuses the kernel and beats the prompt-suite gate. |
| MTP/DFlash verifier rejected gate | `HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION` opt-in for writing resident base-slot position/context from the packed accept kernel. | Default-off; exact quicksort and D32 `9/9`, and the locked profile removed one scalar `set_decode_position_i64` launch/pass (`932 -> 931` calls/pass), but profile host window worsened (`0.2004 -> 0.2010 s` over 11 passes) and the prompt suite regressed wall/verify (`27.038 -> 27.135 ms/cycle`, `21.884 -> 21.969 ms/cycle`). | Remove the runtime gate or fold it into a broader accept/commit composite only if that composite beats the exact prompt-suite gate. |
| MTP/DFlash verifier LM-head | `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD` opt-in for fused W8A16 LM-head + argmax rows. | Default-off; 2026-06-11/12 MTP retests regressed, and clean gfx1151 SOL-S7 now reconfirms exact-but-slower DFlash (`9.676 -> 9.177 tok/s`, -5.16%). The fused body outweighs the saved argmax launch; synchronized accept-readback is only 0.042% of wall. | Removal trigger is satisfied. Demote the runtime branch to a kernel-test-only parity path in a scoped cleanup; do not expose it as a tuning option unless a new schedule beats unfused on gfx1100 and gfx1151. |
| MTP proposer/verifier overlap | `HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER` opt-in for running proposer update on a side stream while verifier commit drains. | Default-off; 2026-06-12 D32 9-prompt exact A/B kept identical acceptance and hid some verify/commit time (`16.166 -> 16.028 ms/cycle`) but regressed proposer/update more (`1.243 -> 1.438 ms/cycle`), moving wall `19.441 -> 19.506 ms/cycle` and ratio `0.9216x -> 0.9184x`. | Remove the harness flag after the current break-even sprint unless a broader proposer graph/update redesign makes the side stream positive on the exact prompt suite. Keep only generic stream plumbing if another retained path uses it. |
| MTP D64 state-drift diagnostics | `scripts/mtp_chain_e2e_smoke.py --ar-fallback-after-mtp-cycles` diagnostic override. | Default-off; added to bracket the D64 `translation` token-34 resident-state drift by forcing target AR after a fixed number of MTP verifier cycles. It is diagnostic evidence plumbing, not an acceptance policy. | Remove or move to a dedicated debug harness after the D64 target-state audit is fixed and artifacted, or if a per-layer state comparator supersedes forced-cycle bisection. |
| MTP GDN state-drift diagnostics | `HIPENGINE_GDN_CHAIN_TLOOP_VTILE` temporary env selector for chain GDN t-loop VTILE 1 vs retained VTILE 4. | Default path remains VTILE=4. `VTILE=1` helped localize the first accepted-row D64 handoff (`force_after=2/3` exact) but did not fix `force_after=4`, so it is not a promoted speed or correctness path. | Remove after the D64 chain GDN/materialized-state bug is fixed or after a narrower per-layer comparator identifies a different root cause. Do not leave this as a permanent user-facing tuning flag. |
| MTP verifier rejected gate | `HIPENGINE_MOE_FUSED_ROTATE` opt-in for M13.B.1 selected-dual rotate+GEMV. | Default-off; 2026-05-23 W7900 gate stayed exact and removed 40 rotate launches/pass, but regressed total kernel time `17.32 -> 29.76 ms/pass` because the fused kernel repeated the rotation per `(out_pack,row)` block. | Remove or demote to kernel test-only after break-even path stabilizes; only keep runtime access if a new non-redundant staged design replaces it. |
| MTP verifier rejected gate | `HIPENGINE_SELECTED_MOE_STAGED_ROTATE` opt-in for M13.B.3 staged selected gate/up rotate+GEMV. | Default-off; staged/keyed gate-up path is exact but later W7900 verifier-window measurement regressed kernel time (`15.344 -> 15.611 ms/pass`) despite launch-count reduction. | Remove or demote to kernel test-only unless a no-spin/no-barrier-reset design beats the unfused chain on the current D32 prompt suite. |
| MTP verifier rejected gate | `HIPENGINE_SHARED_EXPERT_FUSED_ROTATE` opt-in for M13.B.2 shared-expert rotate+dual GEMV. | Default-off; exact, but the saved rotate launch was replaced by a barrier reset in the original path and the keyed-barrier follow-up was neutral (`15.350 -> 15.365 ms/pass`). | Remove or demote to kernel test-only unless a future full-layer C-dispatch path can reuse it without adding per-launch synchronization overhead. |
| MTP verifier rejected gate | `HIPENGINE_FUSED_RMSNORM_ROTATE` opt-in for M15.4 fused input RMSNorm + PARO rotate2. | Default-off; current-stack retest on 2026-06-11 stayed exact but regressed verifier kernel `13.41 -> 14.09 ms/pass` and host window `18.45 -> 19.05 ms/pass`. | After the MTP break-even path is stable, remove the runtime gate or demote it to a kernel test-only path unless a new implementation avoids the one-block RMSNorm occupancy trap. |
| MTP verifier docs | Older "default-off diagnostic" notes for P1 gates can become stale as promoted defaults land. | `docs/reference/MTP.md`, `benchmarks/README.md`, and `WORKLOG.md` carry historical rows plus current status. | During each MTP sprint commit, update current-status language and leave old measurements only as dated history. |
| GGUF decode graph replay | Session-bound graphs retain a full key and cumulative transition budget; callers can still explicitly capture diagnostic windows outside the production selector. | SOL-G5 resolves the old uncapped debt: the context cap is inferred from `position + max_replay_steps` or validated when explicit; the key covers backend/model/weights/buffers/KV/layers/route/sampler/recording/state generation; replay rejects cursor drift or budget overflow. Current gfx1151 HIP passes 128/128 byte-exact checkpoints. | Keep the strict cap/key checks permanently. Remove only redundant legacy graph arguments after the MTP diagnostic modes are retired and all callers use the state-bound API. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --no-target-graph-verify` eager target-verification opt-out. | Default path now uses capped resident decode-graph target verification with fp32 hidden-seed capture; the eager opt-out is useful for bisection and correctness/perf comparison but is >2x slower on the B1 full suite. | Remove or move to a dedicated debug harness once the capped target graph has survived the next MTP break-even sprint and a multi-token GGUF graph correctness gate is part of the regular validation bundle. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --no-mtp-draft-warmup` cold-start diagnostic opt-out. | Default path now runs one stateless untimed draft warmup so MTP timing matches the true-AR warmup protocol; the cold-start opt-out is useful only for measuring wrapper/library/weight-cache first-use cost. | Remove or move to a dedicated cold-start harness once the MTP draft runtime has persistent device buffers and per-process cold-start cost is documented separately from steady-state tok/s. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --root-topk-accept >1`, `--sibling-topk-accept >1`, and `--topk-branch-redraft` exact-verified coverage tree. | Default-off after the 2026-06-23 speed-first reset to B1 linear greedy: K4096 B5 raised accepted/output but regressed draft efficiency (`0.000227`) and tok/s versus B1 linear. Keep it only for coverage diagnostics and llama.cpp parity investigations. | Remove or move to a dedicated experiment harness if the real GGUF MTP verifier path/adaptive policy supersedes top-k tree diagnostics, or if repeated full-suite speed-first gates show the tree remains speed-negative. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-context-replay` slow default-off prompt-catchup replay path. | Useful to bracket llama.cpp-style MTP prompt catch-up while the resident bulk target path exposes only the final hidden row. Current smoke is acceptance-negative and should not be the default benchmark path. | Remove or move to a dedicated debug harness after bulk prefill exposes all-row fp32 hidden seeds and the real MTP KV/RoPE path is implemented, or if that path supersedes replay diagnostics. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-device-kv-cache` default-off device-resident MTP dense KV context. | Implemented for B1 llama.cpp-parity investigation: MTP attention writes post-RoPE K/V to persistent device buffers, draft steps attend over the cache, and accepted target rows use a KV-write-only commit path. Smoke is much faster than host replay/prefix diagnostics but still below the retained no-cache default, so it is opt-in only. | Promote to default only if a same-protocol full-suite row improves raw tok/s, accepted/output, and draft_acceptance; otherwise move to a dedicated debug harness or replace with a paged/KVLiveSpans implementation once bulk hidden-row capture lands. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-graph-batched-verify` full-accept-required verifier graph replay experiment. | Default-off; records generated target IDs and FP32 hidden seeds for a whole strict verifier block. The 2026-06-25 merge-sort B3 smoke stayed exact but was speed-neutral/slower because target kernels still execute sequentially inside the graph and hidden-row recording adds overhead. | Remove unless a true rollback-safe block verifier starts using the recorded hidden rows, or promote only after full-suite evidence shows a speed win over one-step graph replay without requiring prompt-specific full acceptance. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-verify[=bulk/native]` rollback-safe target continuation block verifier. | Default-off; snapshots linear-attention conv/recurrent state with persistent buffers, runs row-bulk target continuation over `[prev]+drafts`, records target IDs + FP32 hidden seeds, and restores/replays the consumed prefix on partial accept. Tiny B3/B5 verifier blocks now default to `--no-target-block-wmma-prefill` because selected/WMMA prefill regressed the B3+32k smoke (`37.8 tok/s`); the GEMV verifier fallback lifts the same smoke to `48.1 tok/s` with `15/15` accepts, but B5 partial rollback remains slow. | Keep as a correctness scaffold and small-B scheduler harness. Promote only if verifier `ar_decode_ms` beats one-step graph replay on full-suite/heldout without reducing acceptance; otherwise replace the linear-attention/rollback pieces with a dedicated small-B continuation kernel and do not re-enable selected-prefill for tiny verifier blocks without evidence. |
| GGUF NativeSpecCycle target route | `scripts/gguf_mtp_bench.py --native-spec-target-cycle` and the legacy-named `native_v1_b2_target_graph` provider route. N1 VERIFY admits independent B1/B2/B3 shape buckets; N2 remains B1/B2. | Retained for explicit accuracy-traded `llama-compat` on both admitted gfx11 backends. One fixed-address graph per admitted bucket uses live device token/position/context/cursor metadata. gfx1100 clean repeats reach **123.332/122.667 tok/s**, preserve every prior 240-ID/96-cycle trajectory and 80.45% acceptance, and clear the 115.444 tok/s llama.cpp floor. The independent gfx1151 transfer reaches **80.132 tok/s** for N1 and **80.099** for public N3 versus **70.020** direct commit (+14.39% N3); all 240 IDs/97 cycle semantics, the real N1/N2 state/KV oracle, and the cached profiler gate pass. Exact dense Qwen3.6 B3 has changing-position plus reject/partial/full transaction coverage and is retained after clean natural25 moved B1/B2/B3 20.846/22.102/21.840 -> 23.225/24.820/25.193 tok/s with every prompt/category/heldout row exact and non-regressive. Other exact/default MTP routes remain unchanged, and the gfx1151 N3P proposal graph remains unregistered. | Keep target-only N1 as the proven performance/rollback boundary while gfx1100 N3 remains below it. Remove the user-facing target-only flag only when complete-cycle ownership is non-regressive on both backends. Rename the legacy `b2` helper/class/registry surface only in a coordinated ABI migration rather than forking another route. Do not route accuracy-traded policy into exact/default MTP; the dense extension itself remains exact. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-direct-state-commit` and suite routes `resident-strict-block-direct-commit` / `resident-hybrid-strict-block-direct-cap32k` / `resident-hybrid-strict-block-direct-native-cap32k` / `resident-b1-branch-safe-direct-cap32k-device-seed`. | Default-off diagnostic. It materializes per-row GGUF linear-attention Conv/GDN state during target block verification and commits captured verifier rows directly when the verifier mode is serial-exact/native, or default `bulk` with a short verifier block (`end < 1024`). Native mode is scalar-byte-exact through row 1 after the row-serial full-attention fallback was fixed to use absolute continuation positions and capture row states. Optimized bulk mode now follows the admitted peer-aligned reassociated GDN contract rather than scalar byte identity. Under the production no-copy capture mode (`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`), W7900 gates prove B5 row 0 is byte-identical to an independent B1 bulk prefix and a deferred B2 row-1 commit is byte-identical to normal B2 final state; serial-exact/native remain the scalar-exact controls. 2026-06-29 smoke is still not positive: pure strict B3 **37.20 tok/s = 0.678x AR**, hybrid direct B3 **49.01 tok/s = 0.893x AR**, native hybrid direct B3 **48.17 tok/s = 0.875x AR**, and B1 branch-safe direct **26.66 tok/s = 0.4849x AR**. | Keep only as rollback-slot verifier scaffolding. Remove or move to a dedicated experiment harness unless a future full-suite/heldout row uses exact direct commit inside a verifier-amortization path that beats true AR. Do not promote from row-level exactness or smoke-level diagnostics alone. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-replay-state-commit`. | Default-off diagnostic. It scores strict target blocks with the selected block verifier without direct linear-state capture, then restores and replays the accepted prefix through `verify_target_block_serial_exact()` for resident state. The corrected 2026-07-02 13-cycle F32 selected-intermediate run proves the transaction wiring (`target_verify_replay_rows=38`, `target_verify_direct_commit_rows=0`) but rejects it as a replication path: it diverges early at cycle 2 (`[40798, 1590]`) and falls to **31.14 tok/s** because every accepted prefix pays serial replay. | Remove after the direct-state capture path is made semantically identical to the intended block scoring path, or move this to a dedicated debug harness if it remains useful as a state-lifecycle comparator. Do not promote; it is intentionally slower and negative semantically. |
| GGUF MTP capture-path diagnostics | Default-off env probes `HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV`, `HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN`, `HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT`, `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN`, `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV`, and `HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL`, plus diagnostic Conv/GDN row-state wrappers `qwen35_linear_attn_conv_prefill_f32_state_rows` and `qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows`. | Added 2026-07-02 to split the forced pair-12 direct-state mismatch. The diagnostic artifact `benchmarks/results/2026-07-02-mtp-capture-path-diagnostics.json` rejects the simple token-output fixes: BF16 GDN output, prefill-shaped Conv/GDN state rows, and score-prefill/chain-commit all still sample `[15495, 539, 1151]` with row-1 margin **+0.29526**. The later lifecycle comparator shows `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1` is required for byte-identical full-accept captured state, so `--llama-compat` now forces that env while the other capture flags remain diagnostic-only. The 2026-07-18 W7900 gate confirmed the old default-off chain-capture test route can produce sentinel/NaN output; the production prefill-GDN route is finite and makes B5 row 0 exactly prefix-equivalent to independent B1 across token, 2,048 hidden values, and 16,711,680 Conv/GDN state floats. Added 2026-07-03 `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV` to test the raw-state finding that selected Conv state has much larger default-vs-prefill-GDN pairwise drift than recurrent state. | Remove or move the unused capture probes to a dedicated debug harness after the verifier FP32 hidden/KV history contract is aligned against llama.cpp. Keep the prefill-GDN capture mechanism as part of llama-compat until a narrower always-prefix-equivalent row-state capture supersedes it. Do not expose the other flags as user tuning; they are negative diagnostics unless the hybrid chain-Conv/prefill-GDN route wins the forced-pair and suite gates. |
| GGUF MTP state-lifecycle diagnostics | `scripts/gguf_mtp_forced_target_probe.py --state-lifecycle-compare` hashes post-cycle FP32 hidden seed plus per-layer Conv/GDN resident state for replay-state vs direct-state verifier policies. | Default-off diagnostic. Added 2026-07-02 for the active llama-compat trace. Base direct capture first mismatches replay at cycle 0 despite identical visible tokens `[12305, 198, 727]`. With `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`, cycles 0-2 become byte-identical and the first mismatch moves to cycle 3, a partial/reject cycle where both policies emit `[65342]` but direct commit diverges from `serial_exact_accepted_prefix` in hidden seed and all linear states. The partial-commit policy fix keeps direct commit for full accepts but serial-replays the accepted prefix on rejected bulk blocks; `benchmarks/results/2026-07-02-mtp-state-lifecycle-prefillgdn-partialfix-compare.json` is clean through cycle 12 (`first_mismatch: null`). The retained serial state-only replay comparator is also clean and improves the exact control **50.96 -> 51.85 tok/s**. The new directcommit comparator `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json` intentionally diverges from serial replay at cycle 3 while emitting the same visible token `[65342]`; that is expected for the llama-replication lane, whose full-suite row is **60.56 tok/s** / **16.534 ms/output** with zero replay rows. | Keep while the split contract remains useful: serial-state for exact-state control, directcommit for llama-replication timing. Remove or move to a dedicated debug harness after parity closure picks the final compat transaction policy and a narrower per-state/KV-tail comparator supersedes this broad hash check. |
| GGUF MTP diagnostics | `HIPENGINE_FUSED_LINEAR_STATE_COMMIT` opt-out around the fused captured Conv/GDN row commit in `Qwen35GGUFResidentSession._commit_verify_linear_state_row`; `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` selects the existing chunked commit copy grid. | Default-on for direct-commit diagnostic paths only. It reuses the DFlash `linear_state_pair_commit_*` kernels to replace per-layer D2D copies when all live GGUF linear layers have captured verifier rows; it falls back to the legacy per-layer copies for non-uniform state sizes and rejects partial captured-row state through the legacy all-or-error check. Focused unit coverage and the GGUF verifier state exactness gate passed on 2026-06-29. No e2e speed claim is retained for this row; it is rollback-slot scaffolding for a future verifier-amortization path. | Remove the opt-out after a full-suite verifier-amortization path that uses direct commit beats true AR, or delete the GGUF direct-commit experiment if it never contributes to such a path. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-verify-mode serial-exact` and `Qwen35GGUFResidentSession.verify_target_block_serial_exact()`. | Default-off diagnostic. It consumes verifier block rows with the token-serial decode scheduler, stages FP32 hidden rows, and optionally records per-row Conv/GDN state. The focused wrong-branch gate proves direct row-0 commit is bit-exact after `[prev, wrong_child]`, unlike the current row-bulk capture path, but this deliberately does not amortize target weight loads. | Keep only as the rollback-slot correctness oracle while developing the row-bulk/amortized verifier. Remove or move to a dedicated debug harness after an exact transactional verifier beats true AR on the full suite, or if a better per-row oracle supersedes it. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-b1-branch-safe-block-verify` and suite route `resident-b1-branch-safe-block-cap32k-device-seed`. | Default-off rejected diagnostic. It probes a B1/root-topK block verifier that batches `[prev, draft0]`, uses row 1 only for strict draft top-1 accepts, and restores/replays row 0 for root-topK branch accepts/rejects. Row-0 direct commit is now exact in the direct route, but both branch-safe variants are smoke-negative: restore/replay B1 AR **54.93 tok/s**, MTP **31.11 tok/s = 0.566x AR**; direct row-0 B1 AR **54.97 tok/s**, MTP **26.66 tok/s = 0.4849x AR**; accepted/output **0.400**. | Remove or move to a dedicated experiment harness unless a future verifier-row lifecycle plus full-suite row beats true AR. Do not re-run as a goal path while branch-safe B1 remains slower than the retained serial/cap32k routes. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --draft-p-min` on the resident draft host-logits path, `--record-draft-confidence` artifact-only top-1 probability capture, and suite route `resident-strict-context-block-pmin08`. | Default-off rejected diagnostic. The resident path supports probability-threshold diagnostics by reading full logits only when `draft_p_min > 0`, computing top-1 softmax probability, and stopping before appending a weak draft. `--record-draft-confidence` records the same top-1 probability in raw cycle artifacts without changing acceptance; the 2026-06-29 full 10-prompt B1 diagnostic (`/tmp/hipengine-draft-confidence-b1.json`) showed strict top-1 `p>=0.999` was clean (`13/13`) but only **21.7%** recall of strict hits, while `p>=0.98` had **28/29** strict hits. The strict-context/block p=0.8 route is not competitive: smoke AR **55.00 tok/s**, B3 **38.44 tok/s = 0.6991x AR**, accepted/output **0.571**. | Keep the covered probability/confidence plumbing only for diagnostics. Remove the suite route and `--record-draft-confidence`, or move them to a dedicated experiment harness, unless a future full-suite/heldout row proves confidence gating helps a structural verifier-amortization path beat true AR. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-draft-vocab-cap N` hot-token draft LM-head cap. | Default full-vocab (`0`). A 32k cap improved the corrected merge-sort B3 smoke from `42.29` to `44.51 tok/s` with unchanged `15/15` strict accepts, but acceptance/quality are prompt-sensitive and the cap is not yet suite-validated. | Either promote a cap after full-suite train/heldout/category validation shows non-regressive acceptance and better true-AR speed ratio, or keep it as an explicit experiment knob and document that retained default remains full vocab. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --adaptive-full-vocab-after-cap-miss` and the suite/workbench `resident-cap32k-recover` route. | Default-off diagnostic. It instantiates a second full-vocab resident draft runner and switches to it after a generic capped-vocab zero-accept miss, suppressing permanent AR fallback for that miss. 2026-06-29 partial suite recovered the known cap32k B1 collapse (`accepted/output 17/37 -> 19/39`) but still measured only **52.45 tok/s = 0.958x true AR**; full suite measured **51.71 tok/s = 0.9478x AR**, `mtp_beats_ar=false`; smaller caps were not goal-closing. | Remove or move to a dedicated experiment harness once the real resident MTP lifecycle/verifier-amortization path lands, unless a future full-suite + heldout row proves capped recovery beats true AR without prompt-sensitive cap tuning. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --resident-mtp-device-seed` and suite route `resident-cap32k-device-seed`. | Default-off structural diagnostic. It seeds the resident draft from the target session's device-resident fp32 hidden seed pointer, mirroring llama.cpp's resident `pending_h` lifecycle direction and avoiding the pending-seed host round trip. 2026-06-29 full suite: AR **54.59 tok/s**, best MTP B1 **52.08 tok/s = 0.9540x AR**, accepted/output unchanged vs cap32k recovery at **78/178 = 0.438**, `mtp_beats_ar=false`. | Promote into the final resident lifecycle only if a future full-suite + heldout row beats true AR, or if the real `GGUFMTPDraftContext` absorbs the device-seed path. Otherwise remove or move it to an experiment harness after the verifier-amortization path lands. |
| GGUF MTP diagnostics | Suite route `resident-cap32k-device-seed-kv` plus `scripts/gguf_mtp_bench.py` path combining `--resident-mtp-device-seed` and `--mtp-device-kv-cache`. | Default-off rejected diagnostic. The route uses new device verifier-row staging and device-base accepted-row KV commit plumbing, but without llama.cpp prompt/context catch-up it collapses draft acceptance: B3 smoke **38.94 tok/s = 0.7124x AR**, draft_acceptance **0.032**; B1 smoke **39.73 tok/s = 0.7235x AR**, draft_acceptance **0.017**. | Keep `Qwen35GGUFResidentSession.stage_current_hidden_seed_as_verify_row()` and `Qwen35GGUFResidentMTPDraftRunner.write_kv_rows_from_device_seed_base()` as lifecycle primitives. Remove or move the no-context-replay suite route unless a future prompt-catch-up resident lifecycle uses it and beats true AR on full suite. |
| GGUF MTP diagnostics | Suite route `resident-context-cap32k-device-seed` plus `scripts/gguf_mtp_bench.py` compatibility path combining `--resident-mtp-device-seed`, `--mtp-context-replay`, and `--mtp-device-kv-cache`. | Default-off rejected structural diagnostic. It wires llama.cpp shifted prompt catch-up, resident device `pending_h`, staged verifier rows, and device MTP KV, but the target verifier remains serial: B1 smoke **50.84 tok/s = 0.9257x AR**, accepted/output **0.400**; B3 smoke **46.97 tok/s = 0.856x AR**, accepted/output **0.571**. | Keep only as a compatibility scaffold while building real target-pass amortization. Remove or move to an experiment harness once a block/graph verifier path owns the llama.cpp lifecycle and beats true AR on the full suite. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --adaptive-strict-block-probe` and suite route `resident-hybrid-strict-block-cap32k`. | Default-off rejected diagnostic. It starts with strict top-1 block-promotion probing and falls back generically to root-topK B1 + cap32k recovery when the strict probe under-accepts. 2026-06-29 full suite: AR **54.58 tok/s**, best MTP B3 **50.91 tok/s = 0.9328x AR**, accepted/output **94/194 = 0.485**, `mtp_beats_ar=false`; worse than cap32k recovery B1 **51.71 tok/s = 0.9478x AR**. | Remove or move to a dedicated experiment harness unless a future full-suite + heldout row proves a strict-block hybrid beats true AR. Do not promote based on smoke/partial closeness. |
| GGUF contiguous prefill metadata | `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA=0|1` selects six synchronous per-chunk H2D copies or the stream-ordered `prepare_prefill_chunk_metadata` kernel; gfx1100 and gfx1151 package policies select the device path through 4,096 prompt tokens. | Scoped default-on on both gfx11 backends. gfx1151 clean 512/1K/4K prefill improves **+1.56%/+0.90%/+0.53%** with 83/83 exact state. W7900 control tracing confirms **167.213 us** of six-copy HIP API wall at 512; exact device metadata improves balanced 512/4K aggregate prefill **+0.41%/+2.43%** (paired **+0.26%/+2.26%**), with non-regressive decode and unchanged memory/IDs. gfx1151 explicit 128K still enters its low-power no-progress state, and gfx1100 long-context behavior is unmeasured, so both retain synchronous metadata above 4K. | Keep `=0` and the synchronous branch for one release and indefinitely for >4K until completed architecture-local long-context gates prove otherwise. After one release, remove the short-context host branch/selector only if both backends remain non-regressive; do not widen the ceiling from short-context evidence. |
| GGUF Q8T16 prefill LCP-3 four-wave | `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE=0|1`, the `wmma_prefill_4wave_bf16_bf16_out` registry variant, and gfx1151 automatic wrapper retain a four-wave duplicate plus the GPF-5A two-wave rollback. | Default-on on gfx1151 through 64K. Clean 512/4K full-model state is 83/83 exact and five-pair prefill improves **1214.510 -> 1220.993 (+0.53%)** and **1269.030 -> 1288.986 tok/s (+1.57%)**; dominant 4K projection micros improve **7.50%-14.08%**. gfx1100 remains production. | After one rollback release and gfx1100 transfer evidence, replace the two-wave body/automatic wrapper and remove its selector if no regression appears. Keep the production long-context path while the predecessor's measured 128K regression remains. |
| GGUF F32 router LCP-4A | Callable `qwen35_router_logits_bf16_f32w_auto_256` duplicates only the Python launch default for the existing exact token-tiled kernel; both gfx1100 and gfx1151 registry bindings now select it. | Default-on on both gfx11 backends. gfx1151 clean 512/4K prefill improves **+2.76%/+3.28%** with 83/83 exact state; W7900 improves **2689.171 -> 2795.242 (+3.94%)** and **2955.867 -> 3070.905 tok/s (+3.89%)**, with a bit-exact 4K primitive, flat graph decode, and unchanged memory. | Keep the 512-thread base callable for one rollback release, then collapse the named wrapper into the normal F32-router binding if no release regression appears. Both architecture-local gates are complete. |
| Laguna gfx1151 router token tile | `LAGUNA_ROUTER_LOGITS_MODE` and session setter retain exact token-tile-4/16 diagnostics around the promoted token-tile-8 route. | Clean seven-pair pp512 improves **497.625 -> 503.349 tok/s (+1.150%)**, wins every pair, keeps every tile-8 sample above 500, and closes the production 500 gate. Production-shape F32 logits, selected IDs, routing weights, complete MoE BF16 output, and token 2930 are exact; cached tracing cuts router **30.658 -> 23.315 ms**. | After one later defaults-only clean refresh remains non-regressive, remove the session setter and token-tile-16 route. Keep token-tile-4 as the low-row/unmeasured-backend fallback. |
| GGUF prefill router select LCP-4B | `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS=64|128|256|512` overrides both backends' promoted 128-thread package capability for bulk prefill only; decode retains its independent launch. | Default-on at 128 on both gfx11 backends. gfx1151 clean 512/4K prefill improves **+0.34%/+0.36%** and the named family falls **12.539 -> 3.741 ms (-70.17%)**. W7900 aggregate prefill improves **+0.32%/+0.81%** (paired medians **+0.30%/+0.12%**) with bit-exact selected IDs/routing weights, flat graph decode, and unchanged memory. The faster 64-thread gfx1151 primitive remains ineligible due to state divergence. | Keep explicit 512 rollback for one release, then remove the env resolver and retain unconditional 128-thread package capabilities if no release regression appears. Never expose/promote 64 without a genuinely different exact implementation. |
| gfx1151 persistent prefill flight recorder | `Qwen35GGUFResidentSession.prefill_flight_recorder_path`, the `qwen35_readme_sweep.py --prefill-flight-recorder{,-granularity}` CLI, file-backed mapped-host ring, and `prefill_flight_recorder_mark_i64_kernel`. | Default-off diagnostic for the unresolved repeated-128K silent no-progress state. `chunk` records every host layer submission but emits one retirement marker per reset/outer chunk; `layer` is explicit high-perturbation refinement. A 512/1 gfx1151 smoke is exact and rocprof confirms the marker, but recorder runs cannot support performance claims or lifecycle incidence estimates. | Keep through the fixed-kernel/firmware three-policy investigation and upstream triage. After the 128K gate is lifecycle-safe, either generalize it into a maintained runtime tracing facility with a schema contract or remove the session/CLI wiring and marker kernel; do not leave an undocumented diagnostic hot-path branch indefinitely. |
| gfx1151 conservative prefill kernel profiles | `qwen35_readme_sweep.py --prefill-kernel-profile conservative|gdn_exact|q4_shared_x|q4_baseline` applies and records fail-closed selector sets. Conservative uses unfused exact GDN `chain`, baseline Q4 selected prefill, baseline linear-attention convolution, 512-thread router selection, and host metadata; `gdn_exact` and `q4_shared_x` each restore one family, while `q4_baseline` changes only the implicated Q4 route beneath otherwise-current package defaults. | Default-off diagnostics for the repeated-128K stall without host drains. Conservative, exact-GDN, and production-shaped Q4 baseline each pass three independent warmup+3 processes (**12/12** prefills). Q4 shared-X reproduces the stall on measured prefill 3 of its first process, isolating `gguf_q4_k_t16_selected_dual_wmma_prefill_compact32_shared_x_kernel<uint16_t>`. These routes are diagnostic, not independent performance paths. | Remove the conservative/GDN split profiles after one release with the gfx1151 Q4 baseline fallback. Retain the baseline-only and shared-X selectors until a repaired shared-X body passes the repeated-128K gate; then remove the isolation profiles and keep only the normal rollback needed for release safety. |
| gfx1151 prefill queue-drain containment | `Qwen35GGUFResidentSession.prefill_queue_drain` and `qwen35_readme_sweep.py --prefill-queue-drain none|chunk|layer` add real host stream synchronization at outer-chunk or model-layer boundaries. | Default `none` is unchanged. `layer` bounds ROCm/ROCm#6437's observed two-layer submission lead and passes the predeclared gate: matched exact 512/4K/64K prefill changes **-1.791%/-0.242%/-0.864%**, while three independent 128K warmup+3 processes complete all **12/12** prefills with finite identical logits, token 9707, normal telemetry, measured graph replay, and full cleanup. It is a qualified slower opt-in containment path, not a default-path fix or proof of root cause. Artifact `benchmarks/results/2026-08-04-gfx1151-q4km-prefill-layer-drain-containment.json`. | Keep explicit `layer` while the upstream issue remains open; never silently default it on. Do not restore the removed `enable_lr_compute_wa` workaround: the captured kernel already has the replacement gfx1151 VGPR-size correction active. Remove after a fixed kernel/firmware/runtime passes the original default-path three-process gate. If maintained beyond that investigation, graduate it from diagnostic CLI/session plumbing into a documented safe-mode policy with explicit admission and cost semantics. |
| gfx1151 HIP hardware-queue default | `configure_hip_process_environment()` sets `GPU_MAX_HW_QUEUES=2` before loading `libamdhip64` when all recognized visible HIP arches map to gfx1151. Existing user values win; gfx1100 and mixed recognized arches are unchanged. | Retained after the complete 2026-08-24 Qwen3.8 FP16-state campaign: explicit `1,2,4,8` and ROCm runtime-default all pass fixed-width, arrival/soak, supported context, pressure/changed-page graph, and prefix lifecycle mechanics. Queue2 has the best normalized 13-width throughput; alternatives are within -0.75% to -0.42%, load/context is neutral, no alternative improves the product SLO, and 64K c2 remains blocked for all policy claims. Queue2 and unset direct cache-only rocprof children each use Queue_Id 1 on their measured single stream. | Keep queue2 as backend metadata and preserve direct `GPU_MAX_HW_QUEUES` user override. The temporary runtime-default suppression, process report, and matrix driver are removed. Revisit only with a new ROCm/firmware stack and a complete same-host protocol; do not infer safety or a default from an intermittent 128K pass. |
| GGUF GDN prefill | `HIPENGINE_GGUF_GDN_PREFILL_MODE=auto|exact|fused|chain|chain_k2|chain_peer_wave32|chain_compact_peer_wave32|chain_peer_cluster8|chain_tile64|chain_tile32|chain_wave32|chain_wave32_tree|chain_lds64|chain_lds32|chain_lds32_direct|chain_lds32_direct_nonvolatile` is the fail-closed rollback/bisection selector. Explicit `chain` is the GGUF-only exact split; `chain_k2` exposes the registered PARO normalized-Q/K two-wave recurrence; `chain_peer_wave32` exposes GPF-9C's llama.cpp-HIP-shaped normalized-Q/K XOR reduction; `chain_peer_cluster8` exposes GPF-9D's llama.cpp-Vulkan-shaped eight-lane clustered reduction; tile64/tile32 and the older wave32 routes are historical controls. The semantic harness sets `HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE=1` internally before session allocation so materialized candidates receive dedicated Q/K/V scratch rather than production direct-route null views. gfx1100 `auto` now selects the bit-equivalent `chain_compact_peer_wave32`; its normalized Q/K scratch is sized at K-head capacity, while explicit `chain_peer_wave32` sessions retain V-head capacity and widening the mode requires session reconstruction. the prior `chain_peer_wave32` remains its rollback. gfx1151 selects byte-exact compiler-cacheable `chain_lds32_direct_nonvolatile` except for the Qwen3.5-0.8B Q4 `(16K,16V,128,128)` plugin key, which selects `chain_peer_cluster8`; Q8 and all other shapes stay exact. | On gfx1151, LCP-2A preserves six-case state plus 250/250 natural transitions exactly and improves balanced 512/1K/4K prefill +34.76%/+36.63%/+36.58%. Under the prospective 18-prompt 0.05/0.90 contract, clean W7900 rejects K2 at KL `0.059031` and register-tree at `0.068757`; both pass top-1 and decode. GPF-9C passes at KL `0.041737`, top-1 `445/450`, and non-regressive decode, but originally missed the 512 floor. LCP-5A removes HIP 7.2 spills from the exact selected-Q5/Q6 T16 prefill leaves, moving clean-target pp512 peer kernels/span to **184.513/194.886 ms**, faster than llama.cpp HIP's **203.301/212.236 ms**. The clean selector-unset 512/4K screen reaches **2588.231/2757.752 tok/s**, clears both floors, keeps IDs stable, and preserves the liveness arena at **21.670 GiB** tracked peak. The final W7900 strict-exact convergence screen selects nonvolatile direct-LDS32: it halves VGPR **64 -> 32**, cuts the 512 trace-family median **74.39%**, and improves volatile-direct full-model 512/4K prefill **+73.01%/+82.46%** with byte-exact primitive state, flat decode, and unchanged compact-scratch memory. The new architecture-scoped `exact` alias exposes that rollback without changing gfx1100 peer-wave production. GPF-9D remains rejected outside its new 0.8B Q4 scope. Inside that key, cluster8 improves paired pp512 +4.33%, passes 448/450 top-1 at max KL 0.003455, and is graph-decode non-regressive; the Q8 candidate remains rejected after missing strict decode by 0.0108%. | Retain explicit volatile `chain_lds32_direct` for one release beneath the promoted `exact` nonvolatile route, then remove it if no exact-route regression appears. Remove `chain_k2` and `chain_wave32_tree` after the final gfx1151 peer transfer decision; retain cluster8 while the 0.8B Q4/Q8_0 plugin keys are shipped (X2-K2 added Q8_0 after a fresh gate), plus fused, `chain`, `exact`, and peer-wave as production/correctness/bisection routes. After one release of compact-peer coverage, remove the per-V-head `chain_peer_wave32` producer/consumer and collapse `HIPENGINE_GGUF_GDN_PREFILL_MODE` to `auto|exact|fused|chain|chain_compact_peer_wave32`. |
| GGUF Q4T16 selected-prefill GPF-3A | `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=auto|baseline|shared_x`, explicit baseline/shared-X registry variants, and replay switch `scripts/qwen35_gguf_moe_replay.py --q4-t16-shared-x` retain duplicate Q4T16 compact32 bodies. gfx1100 `auto` selects `shared_x`; gfx1151 `auto` is rolled back to `baseline`. | BF16/FP16 fixture bytes are exact; tiny trace is `44.725 -> 33.343 us` (-25.45%), real Q4 gate/up replay is `114.633 -> 97.082 ms` (-15.31%), and clean gfx1151 full-model 512/1K/4K prefill improves +3.11%/+2.42%/+1.94%. However, the gfx1151 repeated-128K split changes only this route and reproduces the no-progress state on prefill 4; baseline-only passes **12/12** at **575.180-580.626 tok/s**. The W7900 transfer remains independently admitted. | Keep both bodies and the explicit selector while repairing/bisecting shared-X on gfx1151. Do not restore gfx1151 automatic shared-X until primitive correctness, kernel tracing, and three independent 128K warmup+3 processes pass. gfx1100 remains unchanged. |
| R12 selected-expert prefill pair32 mode | `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=baseline|pair32` selects the selected-expert prefill bodies, and `_COMPACT_MOE_DOWN_MODE_KEYS` adds the matching mode map for the selected-down leg. The paired body (`gguf_k_t16_selected_wmma_prefill_pair32_nwave_kernel`, `NUM_WAVES=4`, two 16-row tiles per wave, 1 KiB LDS-staged activation tile) is a registered sibling of `selected_wmma_prefill_compact_bf16_bf16_out` for Q4/Q5/Q5-qmicro/Q6 T16; the gate/up leg reuses the existing `selected_dual_wmma_prefill_pair32_nwave_bf16_bf16_out` entry (`NUM_WAVES=2`). | **Promoted to the gfx1151 automatic route** (`GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "pair32"`; gfx1100 unchanged). Bit-exact with the compact baseline on all four quants x BF16/FP16 fixture shapes (raw device bits, `test_r12_pair32_nwave_is_bit_exact_with_compact_baseline`). The gate that blocked `shared_x` was cleared: three independent conservative-set 128K warmup+3 processes passed 12/12 (319.25/319.21/319.30 tok/s, stdev <= 0.11, finite logits, IDs [9707,9707,9707]) and three independent production-shaped processes passed 12/12 at 620.61/619.99/619.57 tok/s, above the 575.180-580.626 baseline-only figure on the same workload. Selector A/B 1450.8 -> 1603.7 (+10.54%) and 1454.0 -> 1639.3 (+12.75%); default-path A/B against the explicit rollback 1450.78 -> 1633.33 (+12.58%). Isolated gate+up 1.309x, down Q4_K 1.305x. | Keep the compact `baseline` body and the `_COMPACT_MOE_DOWN_MODE_KEYS` map as the release-safety rollback; `shared_x` stays blocked. Remove the paired bodies, the mode entry, and the mode map together only if a future gfx1151 long-context regression implicates them, and re-admit `shared_x` only if it independently passes the same three-process gate. |
| GGUF Q8T16 prefill GPF-5A | `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE=0|1` selects production/two-wave on gfx1100 and remains the first gfx1151 rollback beneath the promoted four-wave route; request-scoped package ceilings restore production above 65,536 gfx1151 prompt tokens and above 4,096 gfx1100 prompt tokens. | Exact 32-column waves share 1 KiB LDS; 80 VGPR/zero scratch. Published gfx1151 512-64K prefill is **889.904/919.598/762.940/648.948/546.296 tok/s (+1.01% to +8.57%)**; same-commit 128K rejects two-wave **382.041 vs 392.219 tok/s (-2.59%)**. Independent W7900 focus improves 512/4K **645.901/676.444 -> 654.872/683.164 tok/s (+1.389%/+0.993%)** with exact primitive bytes, so gfx1100 is intentionally capped at the measured 4K scope. | Keep env rollback and both architecture ceilings for one release. Expand the gfx1100 ceiling only after a clean hardware-local long-context A/B; collapse the duplicate only if a replacement schedule passes 128K too. |
| GGUF small-B linear dispatch | `HIPENGINE_GGUF_Q4K_ROWTILE` / `q4k_rowtile_session(False)` opt out of raw K-quant rowtiles (rows 2..8, WMMA off) and native resident-pack8 Q4 rowtiles (rows 2..4). | Raw rowtiles remain default-on and exact; their historical MoE full-path result was flat because selected experts dominated. Dense Qwen3.6 now routes native pack8 singles and gate/up pairs through exact local32 row reuse. Rows 2/3/4 improve the real pair 1.715/2.353/3.114x on GPU1; rowtile-4 is VGPR96/scratch0. The routed W7900 transaction oracle observes `{2,3,4}`, and clean natural25 improves B1/B2/B3 **+10.04%/+16.00%/+19.38%** with every full/train/heldout/category row positive, exact IDs/state/acceptance, unchanged memory, and B2 at **1.0678x** own AR. | Keep the opt-out and dual/single rollback for one release after the `4181b85fb` promotion, then make qualified rowtiles unconditional. Preserve per-row kernels for rows==1, rows above each admitted ceiling, registry/backend misses, and explicit WMMA. |
| GGUF selected-MoE dp4a diagnostic | `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` opt-in around q8_1 activation quantization plus sudot4 for raw Q4_K selected-dual gate/up and the T16 rows>1 split gate/up path; `HIPENGINE_GGUF_T16_SELECTED_DP4A` opt-in for the broader T16 selected diagnostic, currently Q4 split plus Q5 selected-down; `HIPENGINE_GGUF_RAW_SELECTED_DP4A` opt-in for the raw no-decode-repack Q4/Q5/Q6 selected-MoE diagnostic bundle. | Default-off. The raw Q4 fallback launches correctly with caller-owned q8_1 workspace but production B3 decode-repack does not hit it; isolated raw Q4 POC measured `0.946 ms -> 0.357 ms`. Raw Q5/Q6 selected-down is also positive in isolation (`0.0916 -> 0.0395 ms` Q5, `0.0419 -> 0.0259 ms` Q6 including q8_1 quantize) and improves no-decode-repack B3 `31.63 -> 39.61 tok/s`, but still trails default decode-repack B3 `51.31 tok/s`. The active T16 split path cuts that row-bulk kernel (`~172 us -> ~142 us` in the two-cycle trace), but B3 remains flat (`49.31 tok/s`, warm `50.60`). Q5T16 selected-down also launches and is `1.10x` faster in isolation (`0.0335 ms -> 0.0306 ms` including quant), but the c1-shaped synthetic top-1 is `0.875` and B3 regresses to `47.62 tok/s` (warm `48.44`). The callable T16 fused-SiLU dp4a variant and Q6T16 selected-down dp4a are intentionally not routed. X8 selected-down is tracked in the dedicated row above because q6-only now has llama-compat evidence while q5/both remains rejected. **2026-06-28 GPU-bound re-test (post lib-cache): `HIPENGINE_GGUF_T16_SELECTED_DP4A` clean interleaved A/B (3 runs x 12 cycles, warm) on the full resident-draft B3 bench is flat-negative `48.60 -> 48.42 tok/s` (-0.4%), acceptance identical — dp4a wins at the kernel level (-35% MoE GEMV) and +5% on the verify-isolated harness but does NOT move the full-bench wall, i.e. the full B3 verifier is host-dispatch-bound (~875 launches), not GPU-kernel-bound. dp4a stays default-off; the lever is launch-count reduction (#9 + fusion). Artifact `benchmarks/results/2026-06-28-verifier-dp4a-fullbench-b3-ab.json`.** | Remove these raw/T16 flags unless a later GGML-style q8_1/x4 layout clears the quality gate and improves the same B3/full-suite protocol. Promote only the production-compatible route; keep raw no-decode-repack diagnostics separate from production T16 routing. |
| GGUF Q5 T16 selected-down one-wave diagnostic | `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS` q5-only override for the T16 selected-down q8_1/dp4a direct kernel. Unset inherits `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` and therefore the retained 64-thread selected-MoE scheduler; valid diagnostic values are `32`, `64`, and `128`. | Default-off. The llama.cpp-shaped one-wave Q5 check improved the isolated selected-down microbench at rows=16 on gfx1151: prequantized dot **0.03608 -> 0.03305 ms**, quantize+dot **0.04031 -> 0.03685 ms**, KL mean **0.00398**, KL max **0.03093**, top-1 **0.9375**. The Q4 control stayed on the 64-thread path (`t16_dp4a_dot_prequantized` **0.04007 ms**), so the override only affected Q5. The real compat smoke rejected it: `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` with q5t32 measured **68.14 tok/s / 14.776 ms/output**, worse than same-route pack8/q6 smoke around **69.06 tok/s / 14.501 ms/output**, with identical smoke acceptance. Artifacts `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-t64-rerun-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-q5t32-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q4-t16-selected-dual-dp4a-q5t32-control-micro.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q5t32-smoke.json`. | Remove this q5-specific override after the selected-MoE scheduler/layout is replaced, or sooner if the next verifier body/layout optimization leaves q5t32 rejected on the same async/full-suite protocol. Do not promote it into the active `llama-compat` route. |
| Env flag surface | Benchmark and diagnostic flags still cover rejected or superseded experiments. | The 2026-07-10 release audit removed env requirements from public backend/quant/GGUF fast-path selection. Accuracy-traded MTP, unsafe kernels, profiler synchronization, and rejected paths remain explicit diagnostics. | Move rejected runtime flags into benchmark-only configuration as each associated MTP/PARO investigation closes; retain correctness fallbacks for one release window after a default promotion. |
| GGUF MTP resident draft | `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE` opt-out around the device-resident selected-MoE down + combine in `mtp_resident_draft.py` (`apply_moe_down_combine`: `silu_mul_separate_out_bf16` + `gguf_q5_k_selected_gemv_bf16_bf16_out` + `weighted_sum_shared_gate_combine_residual_out_bf16_f32w`). The `=0` legacy path keeps the host-readback per-expert Python down loop for bisection. | Default-on after 2026-06-28 B3/c5 A/B: exact-acceptance (drafts byte-identical, accepted_per_output identical in all 4 categories code/general_en/general_ja/mixed_ja_en) and tok/s consistently up per-category (~+0.7-1.4%); removes 2 blocking D->H and ~24 launches/depth from the MoE-down section. Matches verifier bf16 precision. Artifact `benchmarks/results/2026-06-28-resident-mtp-draft-device-moe-down-ab.json`. | Drop the opt-out and delete the legacy host-loop branch (and the then-unused `gate_f32`/`up_f32`/`inter_f32`/`down_out`/`scaled`/`gated_shared` buffers) after sub-win B (device argmax + embedding gather) lands and a same-protocol full-suite MTP row is retained. **Sub-win B landed 2026-06-29 (device-chain row below); the legacy-host-loop deletion + buffer cleanup is now unblocked once a retained full-suite MTP row exists.** |
| GGUF MTP resident draft device-chain | `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN` opt-in (default off) for the device-chained draft in `_propose_chain_device`; explicit bench flag `--resident-mtp-device-chain` prewarms the path for llama.cpp replication routes. Each depth's top-1 is device-gathered from a resident FP32 embedding table (`gather_f32_rows_by_i32id`, exact copy of `token_embd_f32[:vocab]`, 268MB cached upload), top-k is accumulated on device, one drain + readback happens at chain end, and rope/pos/ctx are precomputed once. | Default-OFF. BIT-EXACT vs the legacy host loop (0/5 top-1 + topk-row divergence unit gate; e2e B3 drafts + total_accepted 37/1008 identical). Original B3 evidence was flat (`39.81 -> 39.79 tok/s`) because the draft is GPU-compute-bound, not host/sync-bound. Llama-replication evidence on 2026-06-30: prewarm removes the short-run 268MB upload artifact (`draft_device_chain_ensure_embed_table` **11.888 -> 0.000 ms/output**) but full-suite compat dp4a only moves **52.48 -> 52.79 tok/s**. Split timing shows `draft_topk_readback` is almost all GPU drain (`draft_device_chain_drain` **3.830 ms/output**) and not D2H (`draft_topk_d2h` **0.008 ms/output**). 2026-07-01 sync-stage attribution moves that drain into section buckets: `draft_run_lm_head` **1.882 ms/output**, `draft_run_attention` **0.718**, `draft_run_ffn_up_shared` **0.557**, `draft_device_topk_gather` **0.357**. Artifacts `benchmarks/results/2026-06-29-resident-mtp-draft-device-chain.json`, `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full.json`, `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json`. | Keep as an explicit llama.cpp replication diagnostic while the compat draft/verifier lifecycle is being matched. Do not promote just to remove host copies; promote/collapse only if a future resident draft fusion or verifier-side change cuts the **GPU drain** and improves the same full-suite protocol. Otherwise delete the flag/route during the post-MTP flag cleanup; the `_run_one` device-pointer refactor + gather kernel stay regardless. |
| GGUF MTP resident draft Q6 top-1/gather | `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER` opt-out around the exact Q6_K lm-head top-1 specialization in `mtp_resident_draft.py` for resident device-chain `top_k == 1`. The new `pack8_gemv_decode_bf16_top1_gather_f32` kernel writes the selected id/value and optionally gathers the next FP32 embedding row, replacing full-logits materialization + separate top-k + gather in the llama-compat draft path. | Default-on after 2026-07-01 same-tree full-suite A/B on `llama-compat-device-chain-dp4a`: **52.60 -> 53.34 tok/s** (+1.4%), `cycle_wall_ms_per_output` **19.033 -> 18.772**, `draft_initial` **4.033 -> 3.712**, acceptance unchanged (`acc/output 0.561`, draft acceptance `0.640`). Unit gate proves identical selected id/value/embedding row vs the old logits -> top-k -> gather chain. Sync-stage attribution confirms `draft_device_topk_gather` **0.357 -> 0.001 ms/output** while verifier remains ~14.66 ms/output. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-full.json`, disabled control `...q6top1-control-full.json`, and sync-stage `...q6top1-draftsync-full.json`. | Keep the opt-out for short-term A/B while the llama-compat replication lane is still active. Make the top-1/gather path unconditional for resident device-chain `top_k == 1` after the next compat verifier-layer optimization validates against the same full-suite route, then remove the env flag and old top-k/gather branch for that case. |
| GGUF MTP resident draft Q6 top-1 q8_1/dp4a | `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A` / `--resident-mtp-draft-q6-top1-dp4a` opt-in around the llama-compat resident draft Q6_K lm-head top-1/gather path. It q8_1-quantizes the BF16 head input and calls `pack8_gemv_decode_q8_1_dp4a_top1_gather_f32`, matching llama.cpp's quantized matvec economy more closely than the exact raw-Q6 path. The diagnostic `HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS` / `--resident-mtp-draft-q6-top1-stage1-threads {64,128}` exists only to A/B the stage1 scheduler; suite routes `...x8q6-t64` and `...x8q6-t64-allsync` force 64 threads. | Default-off, accuracy-traded. 2026-07-01 full-suite `llama-compat-device-chain-dp4a-q6top1dp4a` B2: **58.83 -> 59.63 tok/s** (+1.36%), `cycle_wall_ms_per_output` **17.019 -> 16.793**, `draft_initial` **3.564 -> 3.293 ms/output**, acceptance unchanged (`acc/output 0.578`, draft acceptance 0.685). All-sync smoke confirmed `draft_run_lm_head` **1.471 -> 1.253 ms/output**; the q6-X8 stage split now attributes the retained 128-thread aggregate to Q6 top-1 stage1 **1.218 ms/output** plus stage2/gather **0.041 ms/output**, while norm+cast+q8_1 quantize is **0.030 ms/output**. The 64-thread scheduler check is rejected on the real route: all-sync stage1 **1.218 vs 1.246 ms/output** and same-session async **69.06 tok/s / 14.501 ms** vs t64 **68.79 tok/s / 14.557 ms**, identical acceptance. Unit gate matches a CPU q8_1/Q6_K oracle; rocprofv3 confirms both 128-thread and 64-thread stage1 kernels ran. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-full.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-top1split128-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-top1split-allsync-smoke.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-smoke.json`. | Keep the q8_1/dp4a draft head as an explicit llama-compat diagnostic until full-suite heldouts decide whether the accuracy-traded draft head should be folded into the named compat route. Do not promote to the exact/default resident draft path. Remove the t64 thread flag/routes after the next draft lm-head optimization because the scheduler-width copy is rejected; standalone activation-quantization/cast and final reduce/gather tweaks are already ruled out, so future work should target Q6_K stage1 compute/layout or a broader llama-style MMVQ layout. |

### Active mechanical rocprof cache workflow cleanup

- `HIPENGINE_BUILD_CACHE_ROOT` and `HIPENGINE_REQUIRE_CACHED_BUILD` provide a
  process-wide cache contract for lazy JIT builders that do not expose per-call
  plumbing. `scripts/gguf_continuous_owner_rocprof.py` uses them with a new,
  isolated cache, unprofiled build/warm children, a compiler guard, descendant
  monitor, and immutable pre/post cache hashes before wrapping only the final
  owner child in rocprof.
- Keep the two build-layer variables as supported profiler/tooling controls while
  Python/ctypes JIT remains. After profiling entry points share one maintained
  SDK helper, fold the owner-specific orchestrator and duplicate guard/snapshot
  code into that helper; do not retain multiple near-identical rocprof parents.
  Remove neither fail-closed cache-only behavior nor compiler-version pinning.

## Post-Optimal-Path Cleanup Targets

These are not optimization tasks for the current sprint. They are the cleanup
pass to run once a path is fast and correct enough that the benchmark defaults
should be boring.

| Path | Cleanup target | Keep | Remove / collapse trigger |
| --- | --- | --- | --- |
| All-GEMV small-row prefill A/B (`hipengine/generation/qwen35_gguf.py:1107`, `:1180`) | Both shipping call sites pass `use_wmma_prefill=True` literally into `_acquire_shared_session`, and `HIPENGINE_GGUF_WMMA_PREFILL` is opt-in only (`""`/`0` resolve False but cannot override an explicit True), so the route cannot be turned off from a bench or a diagnostic override - `setattr` misses it too because the constant is bound as a default argument. Thread one diagnostic knob from `LLM(...)` to those two sites if the Q6 tile sweep does not close the 45-row gap. | The hardcoded True as the production default; the env var's opt-in-only precedence. | Delete the knob once a measured dispatch rule picks the route by row count, or drop the A/B entirely if small tiles close the gap. **Landed 2026-08-30 as `HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL`** (`_resident_session_wmma_prefill_default`, 16 tests): read at session-acquire time at the `ar_batch`/`mtp_target` sites, unset means the shipped True, an unrecognised value raises instead of silently keeping the route. Deliberately a call-time resolver rather than an `LLM(...)` kwarg - the blocker was precedence, not plumbing; see the Q4K_ROWTILE null result for what a pinned flag does to an env A/B. Other `use_wmma_prefill=True` literals in this module (non-resident routes) are untouched on purpose. |
| Bench tools whose invocation a committed artifact records (`gguf_prefill_row_rollup.py`, `gguf_engine_submodule_decomposition.py`) | Cover the CLI with a synthetic-packet test before the next refactor and keep every artifact's `source_command` executable. A rewrite of the prefill rollup deleted `--prior-config-changed` and replaced max-per-width spread with an aggregate, breaking a published repro recipe and weakening its band. | Fail-closed refusals; effect size kept distinct from drift. | Shared packet reader once a second tool needs the guards. | Partially resolved 2026-08-30: `scripts/check_published_command_drift.py` now gates the set that matters - artifacts cited by `benchmarks/README.md` - by AST-extracting each cited tool's declared flags and refusing commands that use flags the tool no longer declares, commands pointing outside the repo, and commands citing tools that are not in the tree (51 cited artifacts, 0 violations, 4 pre-existing other-lane problems recorded with dates rather than rewritten). It found a second instance the same day: the headline grouped-prefill promotion artifact recorded `--require-mtp`, which argparse no longer accepts, and also omitted `--model` after the tool default moved to Qwen3.6-27B, so a literal rerun would have measured a different model. Still open: per-tool CLI tests for the rollup and the decomposition tool, and 125 historical drifts across all 1579 recorded `scripts/*.py` commands, which are not a fixable backlog.
| 35B MTP chain verifier | Collapse the sprint-era stack of env flags into the default dispatch path and document the current optimal B=1 chain route, while keeping B=2/B=3 available for adaptive-density policy experiments. | Numerical fallbacks, exactness tests, and rollback toggles that are still needed for one release window. | Retained `>1.0x` same-suite row plus one follow-up defaults-only rerun after the adaptive-policy decision. |
| 35B MTP tree/top-k | Keep tree code default-off until it beats chain on the same wall and prompt suite; do not let tree-specific dispatch obscure the chain hot path. | Tree correctness tests and graph replay scaffolding. | If tree remains negative after the verifier wall cut, demote branch/top-k runtime flags to explicit experiment scripts. |
| 27B dense DFlash | Separate deployable online routing from profile-history diagnostics. The current positive production row is the online whole-cycle confidence gate; older prompt-history route/terminal-tail rows are retained evidence, not the default API shape. | Online gate config, oracle/calibration tooling, exact AR comparisons. | After the DFlash hardening rerun and decode API update, trim profile-history routing from the main hot path or move it behind an explicit research harness. |
| DFlash drafter/verifier flags | Audit `HIPENGINE_DFLASH_DRAFTER_DENSE`, `HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM`, and `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD`. | Default-on exact dense WMMA if the fresh 27B gate confirms it; tests for rejected fused kernels. | Fresh 27B DFlash rerun decides: promote exact positive flags to defaults, remove negative runtime branches, or demote them to test-only overrides. |
| Benchmark commands | Stop requiring long flag piles once defaults represent the optimal path. | Flags that select workload shape, model, quant, and explicit experiments. | After MTP/DFlash defaults-only rows are retained, update benchmark docs to show default commands first and move historical A/B flags into dated notes. |

## `--verify-dp4a` / `*-pmin05-dp4a` route (default OFF, opt-in accuracy-traded)
- Added 2026-06-30. Bench flag `--verify-dp4a` (gguf_mtp_bench.py) + suite route
  `resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a` enable llama.cpp-style
  dp4a (q8_1) selected-expert verify GEMVs. **Default off; accuracy-degrading.**
- Purpose: let users who accept llama's precision loss get max accuracy-traded MTP
  perf (~61.6 tok/s / 1.13x B5). FAILS the ja correctness gate (greedy top-1 0.700 <
  0.90). Does NOT match llama HIP MTP (67.3) — dp4a is necessary but not sufficient.
- Remove when: either a Vulkan backend supersedes the perf goal, or the project
  decides to drop dp4a experimentation entirely. Until then it is the documented
  opt-in for the dp4a/accuracy tradeoff. See docs/campaigns/MTP-LLAMACPP-PARITY.md "COFFIN NAIL".

## `--llama-compat` / `llama-compat*` routes (default OFF, semantic diagnostic)
- Added 2026-06-30. Bench flag `--llama-compat` forces the closest hipEngine
  replica of llama.cpp MTP semantics: B2, `draft_p_min=0`, full draft vocab,
  shifted MTP context replay, device MTP KV, no adaptive B1 probe/fallback, and
  one target block verifier per cycle. Suite routes `llama-compat` and
  `llama-compat-dp4a` are fixed to B2 so the artifact label matches the forced
  child `draft_n_max`. Follow-up replication routes
  `llama-compat-device-chain{,-dp4a}` and
  `llama-compat-device-seed-chain{,-dp4a}` add prewarmed resident device-chain
  drafting and optional resident target `pending_h` starts without changing the
  shipped default path.
- Purpose: isolate whether the remaining llama HIP MTP gap is semantic-policy
  mismatch versus implementation/backend cost. The exact compat route is
  precision-preserving; `llama-compat-dp4a` adds the already-known
  accuracy-traded q8_1/dp4a regime. Full-suite B2 evidence landed the same day:
  exact compat **51.16 tok/s = 0.934x AR**, dp4a compat **52.48 tok/s = 0.958x
  AR**, prewarmed device-chain dp4a **52.79 tok/s = 0.965x AR**, and
  device-seed-chain dp4a **52.53 tok/s = 0.960x AR**. All remove the serial B1
  probe and keep acc/output ~0.56, but lose to AR because the compat
  draft/context + block verifier lifecycle is too costly. Split instrumentation
  shows device-chain `draft_topk_readback` is almost all GPU drain
  (`draft_device_chain_drain` **3.830 ms/output**) and not D2H
  (`draft_topk_d2h` **0.008 ms/output**). Follow-up Q6 top-1/gather plus
  direct-state verifier cleanup lifts the best compat dp4a B2 diagnostic row to
  **55.41 tok/s = 1.014x AR**, with unchanged acceptance. The new
  `llama-compat-device-chain-dp4a-allsync` route adds
  `--resident-mtp-draft-sync-stage-timings` and
  `--target-block-sync-stage-timings` for attribution only; its buckets show the
  remaining verifier cost is target linear-attention/MoE operation time, not
  snapshot/commit bookkeeping. The current active replication route is
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit`:
  **60.56 tok/s**, **16.534 ms/output**, **1.1055x AR**, acc/output **0.609**,
  draft acceptance **0.780**, target rows/output **1.172**, verifier drain
  **14.071 ms/output**, replay/commit **0.043 ms/output**, and zero replay rows.
  The semantic-safe serial-state row remains the exact control at **51.85 tok/s**
  / **19.308 ms/output**.
- Remove / promote when: after full-suite stage-bucket evidence decides the
  question. Current evidence says these are replication diagnostics, not default
  promotion candidates; keep only the smallest route set needed for future parity
  audits, or delete the routes during the next MTP flag cleanup unless another
  llama.cpp semantic delta is identified.

## `--resident-mtp-draft-sync-stage-timings` (default OFF, attribution-only)
- Added 2026-07-01. Bench flag inserts `hipDeviceSynchronize()` boundaries inside
  the resident MTP draft `_run_one()` path when `--record-cycle-stage-timings` is
  enabled. Suite route `llama-compat-device-chain-dp4a-draftsync` wires it into the
  llama.cpp replication lane.
- Purpose: split the previous `draft_device_chain_drain` bucket into
  `draft_run_project`, `draft_run_qkv_kvwrite`, `draft_run_attention`,
  `draft_run_ffn_up_shared`, `draft_run_moe_down_combine`, `draft_run_lm_head`, and
  `draft_device_topk_gather`. The flag changes timing by adding synchronization and
  is not a performance path.
- Remove when: the resident draft LM-head/top-k or verifier layer-time follow-up has
  its own lower-overhead profiler/rocprof attribution, or after the llama.cpp
  replication lane is closed. Until then keep it only as a named diagnostic route.

## `--target-block-sync-stage-timings` (default OFF, attribution-only)
- Added 2026-07-01. Bench flag inserts `hipDeviceSynchronize()` boundaries inside
  the target block verifier when `--record-cycle-stage-timings` is enabled. Suite
  route `llama-compat-device-chain-dp4a-allsync` combines it with resident draft
  sync timings for one-pass draft+verifier attribution.
- Purpose: split `target_block_linear_attn_layers` and
  `target_block_full_attn_layers` into operation buckets (`norm_qkv_gate`,
  `chain_gdn`, selected-MoE expert gate/up/down, shared expert, combine, and
  full-attn KV/attention/output sections). The flag changes timing by adding
  synchronization and is not a performance path.
- Remove when: a lower-overhead verifier profiler/rocprof harness can produce the
  same operation split, or after the llama.cpp replication lane is closed.

## `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL` / `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT` / `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM` / `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS` / `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA` / `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT` / `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE` / `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN` / `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE` / `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN` / `HIPENGINE_GGUF_VERIFY_F32_POST_NORM` (default OFF, semantic diagnostic)
- Added 2026-07-02. Env flag keeps target-block verifier residual outputs in
  FP32 for an opt-in llama.cpp parity probe while preserving BF16 mirrors for
  existing projection kernels. It adds FP32 add/RMSNorm and MoE combine helpers.
  The follow-up diagnostic also feeds layer-entry attention RMSNorm from FP32
  residual rows when available, but intentionally does not claim full llama.cpp
	  graph parity. `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` materializes
	  layer-entry attention RMSNorm into FP32 scratch, casts a BF16 mirror for
	  unsupported consumers, and routes dense-Q8 dp4a QKV / QKV+gate consumers from
	  the FP32 tensor when the F32 dense-Q8 diagnostic is already active.
	  `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS=1` additionally routes
	  compatible row-bulk linear-attention Q8 `attn_qkv`/`attn_gate` projections
	  into FP32 scratch through the raw-Q8 dp4a F32-output dual wrapper, casts BF16
	  mirrors for existing downstream kernels, and emits explicit BF16 mirror
	  capture keys. It also routes dense-F32 `ssm_alpha`/`ssm_beta` through the
	  registry-dispatched F32-input/F32-output dense GEMV route when available,
	  while preserving BF16 mirrors for existing downstream consumers.
	  `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA=1` additionally routes row-bulk
	  linear-attention `ssm_alpha`/`ssm_beta` from that FP32 attention-norm tensor to
  mirror llama.cpp's `build_layer_attn_linear` source shape.
  `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` keeps the row-bulk linear-attention
  `ssm_out` projection output in FP32 through the post-attention residual/RMSNorm
  add while preserving the BF16 mirror for existing captures and downstream
  kernels. It also keeps row-bulk full-attention `attn_output` in FP32 through
  the same residual/RMSNorm helper when a raw Q8 sidecar BF16-input/F32-output
  path is available.
  `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE=1` keeps the selected-expert weighted
  sum in FP32 inside the F32-residual MoE combine instead of BF16-rounding that
  selected sum before adding residual and sigmoid-gated shared output.
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN=1` requires the F32 MoE combine
  diagnostic and routes compatible X8 Q5/Q6 selected-down GEMV outputs into an
  FP32 scratch buffer before combining selected rows with BF16 shared output.
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE=1` requires the F32 MoE
  combine + selected-down stack, computes selected `silu(gate) * up` into FP32
  scratch, preserves the BF16 mirror, and feeds the FP32 activation into
  selected-down.
  `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN=1` requires the F32 MoE combine and
  selected-down stack, routes shared-expert down output into FP32 scratch,
  preserves the BF16 mirror, and combines FP32 selected rows with FP32 shared
  rows.
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` extends the probe by materializing
  post-attention RMSNorm into FP32 scratch and independently gating router /
  selected-q8 / shared-q8 consumers with
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_{ROUTER,SELECTED_Q8,SHARED_Q8}`.
  `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT=N` bounds F32 residual
  propagation to layers `[0, N)` and rejoins the ordinary BF16 verifier at the
  next layer. It defaults to the complete layer count and is active only when
  F32 residual is enabled. Added 2026-08-27 to screen a model-wide fixed
  production mechanism after layer-local strict-teacher evidence localized the
  dominant K2 amplification at a full-attention boundary; remove the flag after
  this qualification campaign chooses a fixed manifest variant or rejects all
  prefix lengths.
- Purpose: test the current semantic hypothesis that accumulated BF16 verifier
  layer-boundary drift is enough to flip near-tie target decisions versus
  llama.cpp's F32 target `l_out` graph tensors. The diagnostic artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-diagnostic.json`
  confirms the lever is active: the old cycle-12 trace cannot replay unchanged
  because cycle 2 flips from exact `[40798, 25, 1103]` / accepted 2 to
  FP32-residual `[40798, 1590, 1103]` / accepted 1. The follow-up artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-attnnorm-diagnostic.json`
  reaches the old cycle-12 branch but still accepts `539`, with the wrong
  `539 - 26126` margin increasing from **+0.11822** to **+0.14309** versus
  llama.cpp **-0.00896**. The attention-norm-output dense-Q8 split
  (`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-output-denseq8-diagnostic.json`,
  control
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-bulk-control-diagnostic.json`)
  moves the bulk pair-12 `539 - 26126` margin from **+0.31369** to
  **+0.18198**, still opposite llama.cpp. The linear-attention output-to-residual
  split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-denseq8-diagnostic.json`
  moves that margin only **+0.18198 -> +0.17663**, so the `ssm_out` BF16 round is
  not the main missing semantic lever. The alpha/beta F32 input split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-denseq8-diagnostic.json`
  is byte-identical to the attention-output slice and leaves the row-1 margin at
  **+0.17663**, ruling out `ssm_alpha`/`ssm_beta` projection input precision for
  the active branch. The full-attention output-to-residual split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-fullattnout-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]`, accepts 2, and worsens row-1
  `539 - 26126` to **+0.27480**, so full-attention `attn_output` BF16 output
  rounding is ruled out as well. The MoE selected-sum accumulator split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]` and accepts 2, but narrows row-1
  `539 - 26126` from **+0.27480** to **+0.03385**. That makes the combine
  selected-sum BF16 boundary semantically active, but still not sufficient to
  match llama.cpp's **-0.00896** margin. The selected-down output split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-denseq8-diagnostic.json`
  keeps compatible X8 selected-down rows in FP32 and narrows the same margin to
  **+0.00536** (`26.06115 - 26.05580`), still on the wrong side of llama.cpp by
  about **0.0143 logits**. The selected-intermediate split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-selectedintermediate-denseq8-diagnostic.json`
  is the first pair-12 side-matching slice: sampled tokens become
  `[15495, 26126, 1151]`, accepted drafts fall to 1, and row-1 `539 - 26126`
  moves to **-0.00303** (`26.04795 - 26.05098`), close to llama.cpp's
  **-0.00896**. This confirms the selected SwigLU/intermediate BF16 boundary
  is a parity contract to fold into a cohesive llama-compat verifier mode if
  longer-trace/full-suite acceptance validates it. The shared-down output split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-shareddown-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]` and accepts 2, and widens the same margin
  to **+0.03043** (`26.12703 - 26.09660`), ruling out isolated shared-down
  output precision as the missing parity fix. Combining MoE combine with
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` failed prior-cycle replay at cycle 2
  (`[40798, 1590, 1103]` vs trace `[40798, 25, 1103]`), so the combination is
  not a pair-12 parity result. The post-norm split artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-postnorm-split-diagnostic.json`
  shows the combined router+selected-q8 consumer path breaks the old trace at
  cycle 7; selected-q8 alone flips row 1 (`413 - 4071` **+0.13053 -> -0.14458**),
  while router-only reaches pair 12 but worsens `539 - 26126` to **+0.33520**.
  This makes these flags instrumentation only, not a broad candidate promotion.
  The 2026-08-27 35B Generation-2 qualification reuses only the bounded
  `F32_RESIDUAL + F32_POST_NORM` K2 composition as an explicit production
  candidate while forcing K1 back to the incumbent BF16 verifier. That
  shape-specific use remains pending its final full quality/performance/serving
  packet and does not certify any other flag combination or route.
- Remove when: fold a passing K2 composition into a typed registered variant
  (and delete its env plumbing), or remove the candidate if final qualification
  fails. The historical broad diagnostic modes remain non-production.

## `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` (diagnostic rollback)
- Added 2026-07-01. Host-side launch switch for the selected T16 q8_1/dp4a
  verifier kernels used by `--verify-dp4a` / `llama-compat-device-chain-dp4a`.
  Default is now `64`; setting the env var to `128` restores the old launch
  shape.
- Purpose: keep a rollback/A-B hook for the first selected-MoE scheduler change
  that survived async/full-suite validation. Full-suite `llama-compat` B2 moved
  **55.45 -> 58.83 tok/s** and `target_block_verify_total`
  **14.025 -> 13.134 ms/output** on gfx1151.
- Remove when: either the selected-MoE scheduler is replaced by a llama-style
  `mul_mat_vec_q_moe` port or two later full-suite compat runs confirm 64 is
  stable enough that the 128-thread rollback path is no longer useful.

## `HIPENGINE_GGUF_Q8_T16_THREADS` (diagnostic rejected)
- Added 2026-07-01. Host-side launch switch for Q8_0 T16 single/pair/triple
  GEMV wrappers. Default/unset keeps the existing 128-thread launch; setting
  the env var to `64` exercises a smaller workgroup for verifier projections.
- Purpose: test whether the llama-compat verifier hot leaf
  `attn_qkv+attn_gate` is losing time because Q8T16 pair projection uses the
  wrong launch width. The focused qwen35 pair microbench rejected 64 threads:
  rows 2/3/4 measured **197.77/224.80/251.96 us** at 64 threads versus
  **179.26/207.05/237.02 us** at 128 threads. `rocprofv3` confirmed the 64-thread
  override launched with `Workgroup_Size_X=64`.
- Remove when: the Q8T16 verifier pair work moves to a different llama-style
  kernel body/schedule, or when the parity sprint no longer needs this A/B hook.
  It is not a performance path and should not be promoted.

## `gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out` (diagnostic rejected)
- Added 2026-07-01. Callable T16 Q8_0 dual-split pair kernel that consumes
  GGML q8_1 activation blocks and uses `sudot4`, intended to test whether the
  llama.cpp Q8_0×Q8_1 arithmetic recipe transfers to the existing Q8T16
  `attn_qkv+attn_gate` verifier pair layout.
- Purpose: isolate the kernel-body question after the 64-thread launch-width
  check failed. Correctness passed against a q8_1 CPU oracle plus KL/top-1 gate,
  and `rocprofv3` confirmed `q8_0_t16_dual_split_q8_1_dp4a_kernel<unsigned short>`
  launched with `Workgroup_Size_X=128`. Performance rejected the route: for the
  qwen35 pair shape, rows 2/3/4 exact 128-thread pair is
  **181.50/207.98/236.26 us**, while quantize+dp4a is
  **304.78/448.32/558.14 us** and prequantized dp4a is
  **303.05/452.51/566.29 us**.
- Remove when: the parity sprint moves Q8 verifier work to a true llama-style
  mmvq/T16 replacement layout or row-amortized verifier kernel. This callable is
  evidence, not a performance path; do not route it into `llama-compat` runtime.

## `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE` / `HIPENGINE_GGUF_Q8_T16_PAIR_COL8` (gfx1151 C8 rollback)
- The 2026-07-01 rowtile diagnostic was repaired to 128 threads so it preserves
  the production reduction partition, then scoped by backend metadata to
  gfx1151 physical C8. C2/C4 and gfx1100 retain per-row Q8T16.
- F3I keeps rowtile4's two row groups but computes eight output columns per
  block. This halves live accumulators and moves static resources from 136 to
  72 VGPR and 1,024 to 512 B LDS, with zero scratch. Exact direct C8 improves
  **151.015 -> 152.164 tok/s (+0.76%)** in implementation evidence; matched
  blocking/SSE/delayed server rows all improve **+0.38%/+0.54%/+0.46%**.
- `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` rolls the physical-C8 owner back to
  the per-row kernel. `HIPENGINE_GGUF_Q8_T16_PAIR_COL8=0` keeps row
  amortization but restores the prior 16-column body. Explicit `PAIR_ROWTILE=1`
  outside the backend-certified session remains a diagnostic and does not
  automatically select col8.
- Remove the two rollbacks after one release window plus defaults-only gfx1151
  direct/server refreshes and an independent gfx1100 transfer. Keep the old
  per-row and 16-column rowtile bodies as unsupported-width fallbacks.

## `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` (gfx1151 c4/c8 rollback; c2 diagnostic)
- Added 2026-07-01 as a default-off runtime hook for broad Q8T16 verifier
  row-amortization. Setting `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` routes qwen35
  `rows>1, in=2048` singleton, pair, and triple Q8T16 projections through
  rowtile4 wrappers where available. It also enables pair rowtiling unless
  `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` is explicit.
- The original verifier and F3 broad-promotion decisions remain useful history,
  but both bound free-running generated-ID equality as if it were request/state
  correctness. The calibrated production contract instead binds strict-teacher
  logits and keeps cross-route free-running IDs diagnostic.
- Fresh clean ZBook-local evidence applies that distinction without weakening
  determinism; its rates are independent of the other gfx1151 host:
  18 prompt+heldout cases produce **1,050/1,050 bit-identical strict-teacher
  full-logit rows** across static c4/c8, c8->c4->c2->c1 retirement, sparse
  physical c8, every category/shape/transition scope, and three exact repeats.
  Same-route p512/d128 trajectories are repeat-deterministic; strict/candidate
  trajectories differ, as allowed for production.
- Seven current-package counterbalanced pairs retain only physical c4/c8:
  c4 **78.624 -> 78.974 tok/s (+0.445%, 7/7)** and c8 **96.482 -> 97.125
  (+0.666%, 6/7)**. C2 loses **59.412 -> 58.345 (-1.795%, 0/7)**.
  Therefore gfx1151 sets `GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS=4` while
  `GGUF_Q8_T16_DECODE_ROWTILE_ALL` stays false; gfx1100 keeps floor `0`.
- Unset uses the backend width floor. Explicit `=0` restores direct singleton/
  triple and the pre-existing pair policy at every width; explicit `=1` still
  forces the broad c2 diagnostic. Remove the broad override after one release
  window and defaults-only dynamic/server refresh, while retaining direct
  kernels as unsupported-width and rollback fallbacks.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=row` / row Q6 top-1 routes (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape row` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row` /
  `...-row-allsync` exercise a llama.cpp-shaped Q6_K draft lm-head top-1
  stage: one output row per block, two wave32 warps, and a signed
  `__vsubss4`/dot4 Q6_K MMVQ body. Default stays `pack8`.
- Purpose: test whether the remaining draft-side Q6_K top-1 gap is caused by
  hipEngine's pack8 output-row geometry rather than the vector-dot body itself.
  Correctness passes against the q8_1/Q6_K oracle, but performance rejects the
  route: all-sync row stage1 is only slightly faster than pack8
  (**1.202 vs 1.218 ms/output**) while row stage2/gather grows
  **0.041 -> 0.252 ms/output** because it reduces over `vocab` instead of
  `vocab/8`; async smoke regresses **69.06 tok/s / 14.501 ms** to
  **66.95 tok/s / 14.958 ms** with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused row-stage/top-1 reduce makes this row-shape diagnostic
  obsolete. It is evidence, not a performance route.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack8_scalehoist` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack8_scalehoist` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist` /
  `...-scalehoist-allsync` exercise a Q6_K draft lm-head top-1 stage that keeps
  the retained pack8 `vocab/8` final reduce but hoists each Q6_K block's
  `d*scale[16]` values into shared memory.
- Purpose: test whether the remaining q8_1/dp4a Q6_K draft stage1 cost is from
  repeated Q6 scale loads rather than the dot body or output geometry. Correctness
  passes against the q8_1/Q6_K oracle, and `rocprofv3` confirms
  `gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_scalehoist_stage1_kernel` launches.
  Same-session smoke rejected it: retained `x8q6` rerun **68.65 tok/s**,
  cycle **14.589 ms/output**, `draft_initial` **2.482 ms/output** vs
  scalehoist **68.54 tok/s**, cycle **14.610 ms/output**, `draft_initial`
  **2.485 ms/output**, with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack8_llama` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack8_llama` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama` /
  `...-pack8llama-allsync` exercise a Q6_K draft lm-head top-1 stage that keeps
  the retained pack8 `vocab/8` final reduce but uses the llama.cpp Q6_K MMVQ
  vecdot decomposition inside stage1.
- Purpose: test whether the remaining q8_1/dp4a Q6_K draft stage1 cost is the
  pack8 dot body rather than final-reduce geometry. Correctness passes against
  the q8_1/Q6_K oracle for fused and split stage1+stage2 paths. Same-session
  all-sync moved the intended leaf **1.220 -> 1.205 ms/output**, but async B2
  smoke rejected the route: retained control **68.88 tok/s**, cycle
  **14.541 ms/output**, `draft_initial` **2.487 ms/output** vs pack8_llama
  **67.92 tok/s**, cycle **14.747 ms/output**, `draft_initial`
  **2.493 ms/output**, with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack16` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack16` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack16`,
  `...-denseq8all-pack16`, and `...-denseq8all-pack16-allsync` exercise a
  Q6_K draft lm-head top-1 stage that keeps the retained pack reduction but
  doubles the output group from 8 to 16 vocab rows per block.
- Purpose: test whether the current draft Q6_K stage1 cost is dominated by q8_1
  activation reloads and final-reduce entries rather than register pressure in
  the per-output Q6 body. Correctness passes against the q8_1/Q6_K oracle for
  fused and split stage1+stage2 paths. Same-session denseq8all smoke rejected it:
  retained control **71.74 tok/s**, cycle **13.961 ms/output**,
  `draft_initial` **2.479 ms/output** vs pack16 **71.72 tok/s**, cycle
  **13.963 ms/output**, `draft_initial` **2.487 ms/output**, with identical
  acceptance. Draft rocprof confirms the kernel-family loss:
  `gguf_q6_k_pack16_gemv_q8_1_dp4a_top1_stage1` is **3.684 ms/cycle** vs the
  retained pack8 stage1 **3.603 ms/cycle**.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8_dscale` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape x8_dscale` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8dscale-f32ssm`
  / `...-x8dscale-f32ssm-allsync` exercise the retained X8-packed Q6_K draft
  lm-head top-1 layout with an extra X8-aligned FP32 `d*scale` sidecar.
- Purpose: test whether the remaining retained X8 Q6_K top-1 cost is dominated
  by repeatedly unpacking/multiplying Q6 block scales inside the dot body.
  Correctness passes against the q8_1/Q6_K oracle for fused and split
  stage1+stage2 paths, but draft-chain rocprof rejects the route:
  `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8dscale.json`
  reports host wall **6.805 -> 8.023 ms/cycle**, kernel time
  **6.427 -> 7.615 ms/cycle**, and `draft_lm_head_q6_top1`
  **3.648 -> 4.859 ms/cycle** versus the retained X8 artifact. Extra FP32
  sidecar memory traffic/register pressure is worse than recomputing scales.
- Remove when: a later draft Q6_K top-1 body/layout or fused top-1/sampler route
  supersedes the current X8 evidence set. This route is evidence only; do not
  promote or rerun full-suite unless a separate change materially alters the
  dscale memory path.

## `--fused-b1-block-probe` / `resident-fused-b1-block-direct-cap32k-minrows2-pmin05`
- Added 2026-06-30. Bench flag `--fused-b1-block-probe` keeps the retained
  adaptive B1-probe policy, but lets B1 probe cycles verify `[prev, draft0]` with
  one strict two-row target block instead of the serial target step loop. The suite
  route mirrors `resident-b1-probe-block-direct-cap32k-minrows2-pmin05` plus this
  flag. **Default off** until a full-suite row proves it improves wall time.
- Purpose: test the first queued llama.cpp-parity fix from
  `docs/campaigns/MTP-LLAMACPP-PARITY.md`: remove or shrink `target_serial_verify_step`
  without merely shifting the same cost into `target_block_verify_total`.
- Remove / promote when: promote into the retained route only if exact full-suite
  B5 beats the current default and stage buckets show serial verifier cost falls
  below ~2 ms/output with no acceptance regression. Otherwise delete the flag/route
  after the parity A/B is recorded.

## `--target-block-direct-partial-replay-mode`
- Added 2026-07-02. Bench and forced-target-probe flag with choices
  `serial-exact` (default), `serial-state-only`, `direct-commit`,
  `bulk-state-only`, and `native-state-only`. It only affects direct-state block verification when a
  bulk verifier block is rejected or partially accepted. The retained
  `serial-state-only` mode restores the snapshot, advances the accepted prefix
  through `verify_target_block_serial_exact(..., advance_state_only=True)`, and
  skips replay LM-head sampling; target tokens still come from the original
  full-block scoring pass. The active llama-replication `direct-commit` mode
  commits the captured verifier row on rejected/partial blocks, matching
  llama.cpp's normal MTP accept lifecycle rather than serial-prefix replay. The
  rejected bulk/native modes replay the accepted
  prefix with `verify_target_block(..., advance_state_only=True)`, with
  `native-state-only` using the native row-serial-attention verifier only for
  that state replay.
- Purpose: reduce the semantic-safe `llama-compat` replay/commit bucket while
  preserving state lifecycle, and provide a separate llama-style replication
  lane. `--llama-compat` promotes unspecified `serial-exact` replay to
  `direct-commit`; explicit serial-state, bulk, and native diagnostic modes remain
  opt-in.
- Result: `direct-commit` is retained for the llama-compat replication lane. The
  full-suite row
  `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`
  moves **51.85 -> 60.56 tok/s**, cycle **19.308 -> 16.534 ms/output**,
  verifier drain **16.891 -> 14.071 ms/output**, replay/commit
  **2.489 -> 0.043 ms/output**, and replay rows **38 -> 0** versus the
  serial-state control. The lifecycle diagnostic
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json`
  intentionally diverges from serial replay at cycle 3 with matching visible
  token `[65342]`; that is expected for llama-replication, not an exact-state
  claim.
- Exact-control result: `serial-state-only` is retained as the semantic-safe
  control.
  The lifecycle comparator artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-serial-state-only-partial-replay-compare.json`
  reports `first_mismatch: null`, and the full-suite row
  `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`
  moves **50.96 -> 51.85 tok/s**, cycle **19.645 -> 19.308 ms/output**,
  verifier drain **17.222 -> 16.891 ms/output**, and replay/commit
  **2.775 -> 2.489 ms/output** with unchanged acceptance/economy.
- Rejected diagnostics: artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-bulk-state-only-partial-replay-compare.json`
  reports `first_mismatch` at cycle 3. The visible token still matches
  `[65342]`, but `bulk_state_only_replay` diverges from
  `serial_exact_accepted_prefix` in hidden seed plus Conv/GDN state across 61
  fingerprints. The active-shape native replay artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-native-state-only-partial-replay-active-compare.json`
  also fails at cycle 3 with matching visible token `[65342]` but 59
  hidden/linear-state mismatches.
- Remove when: parity closure picks the final compat transaction policy. If
  `direct-commit` remains the llama-replication path, collapse the route/flag
  surface so only the named compat mode and the exact serial-state control remain.
  If exact-state semantics become the compat target, delete directcommit as a
  perf diagnostic. Do not promote bulk/native state-only replay into any retained
  route.

## `--verify-lm-head-q6-top1-dp4a` / verifier lm-head X8 sidecar
- Added 2026-07-01. Bench flag `--verify-lm-head-q6-top1-dp4a` sets
  `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR=1` before materialization and
  `HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A=1` at runtime. Suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-vlmheadtop1`
  and `...-allsync` exercise it on top of the active llama-compat lane.
- Purpose: test whether the verifier-side `target_block_lm_head_sample`
  bucket can copy the draft-side q8_1/dp4a Q6_K top-1 economy by skipping full
  verifier logits plus argmax. This is accuracy-traded and default-off; exact
  verifier lm-head sampling remains the shipped behavior.
- Remove / promote when: promote only inside the llama-compat replication lane
  if a full-suite B2 row moves total wall and `target_block_lm_head_sample`
  toward the llama.cpp verifier target without unacceptable row-economy loss.
  Delete the route if smoke/full-suite shows the extra X8 sidecar/top-1 path does
  not move `target_block_verify_total` or if a later fused verifier sampler
  supersedes it.

## `HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL`
- Added 2026-07-02. Bench flag `--resident-mtp-draft-router-row-parallel`,
  draft-profile flag `--router-row-parallel`, and suite route
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow`
  route the resident MTP draft F32 router projection through the row-parallel
  `qwen35_router_logits_f32_f32w` kernel instead of the generic one-block
  `hipengine_mtp_linear_f32` path.
- Purpose: retained llama.cpp-replication optimization. Full-suite B2 moved
  **63.63 -> 64.41 tok/s** and cycle **15.735 -> 15.547 ms/output** with
  unchanged acceptance/economy. Draft-chain sync attribution moved
  `draft_run_ffn_router_linear` **0.508 -> 0.048 ms/cycle**.
- Remove / promote when: make this the unconditional resident draft router
  projection once it is either promoted beyond the llama-compat lane or no
  longer needs A/B isolation. Delete the env/CLI flag and old generic-router
  fallback route after the next parity checkpoint no longer needs the direct
  control.

## `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` (diagnostic rejected)
- Added 2026-07-02. Bench flag `--resident-mtp-draft-dense-q8-dp4a`,
  draft-profile flag `--dense-q8-dp4a`, and suite route
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8`
  route resident draft dense Q8_0 F32 projections through F32->q8_1 plus
  raw-Q8 dp4a float-output wrappers.
- Purpose: test whether copying the verifier/llama.cpp q8_1/raw-Q8 dp4a economy
  into the draft dense projections (`eh_proj`, Q/K/V, attention output, shared
  gate/up, shared down) closes the non-Q6 draft drain gap.
- Result: draft-chain rocprof moved the intended kernel bucket, but full-suite
  B2 rejected the route: active router-row **64.41 tok/s / 15.547 ms/output**
  vs draftdenseq8 **64.14 tok/s / 15.612 ms/output**, with worse acceptance and
  target rows/output.
- Remove when: the next parity checkpoint no longer needs this negative
  evidence, or if a future fused dense-draft design replaces the standalone
  quantize+dp4a calls. Do not promote the flag as-is.

## `HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` (diagnostic rejected)
- Added 2026-07-02. Bench flag
  `--resident-mtp-draft-selected-silu-down-fused`, draft-profile flag
  `--selected-silu-down-fused`, and suite routes ending in `-siludown` route
  selected MoE `silu(gate)*up` directly into a Q5_K selected-down GEMV.
- Purpose: test a llama.cpp-shaped fused GLU/down idea without changing row
  economy or draft precision. The fused kernel is bit-exact versus the existing
  BF16 chain (`silu_mul_separate_out_bf16` + `gguf_q5_k_selected_gemv_bf16_bf16_out`).
- Result: rejected by draft-chain profile before full-suite. It removes one
  launch, but the fused Q5 body is slower: active router-row control
  **5.973 ms/cycle kernel / 7.044 ms/cycle host** vs fused
  **6.054 ms/cycle kernel / 7.206 ms/cycle host**; selected-down family
  **0.325 -> 0.391 ms/cycle**.
- Remove when: the next parity checkpoint no longer needs this negative evidence,
  or if a different fused Q5 selected-down body replaces it and wins the draft
  parent profile. Do not promote the flag as-is.

## `HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING`
- Added 2026-07-03. Default-off verifier diagnostic that seeds the target
  verifier F32 residual buffer from host-dequantized `token_embd.weight` rows
  when `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` is enabled. The BF16 token
  embedding launch still runs and still populates the BF16 mirror path.
- Purpose: isolate llama.cpp MTP parity for layer-0 target input construction.
  The first task-9/cycle-3/row-2 run closed `hidden_in` and
  `attn_norm_f32_scratch` to exact llama.cpp parity, but it did not flip the
  bonus token; the remaining split moved to F32-input projection/dequant and
  later residual/LM-head amplification.
- Remove / promote when: remove once the projection/dequant split is resolved
  and this host-side diagnostic is no longer needed. Promote only by replacing
  it with a real device-side F32 embedding path if full-suite llama-compat
  evidence shows it is required and non-regressive; do not keep host dequant/H2D
  in a timing route.

## `--record-draft-stage-stats`
- Added 2026-07-02. Bench flag that records compact FP32 summaries for resident
  MTP draft sub-stage tensors and dense MTP K/V cache rows in
  `draft_hidden_state_trace`. Default-off extensions
  `--record-draft-cache-rows` and `--record-draft-attention-debug` add selected
  history rows plus host-recomputed dense-attention score/weight diagnostics.
  It forces host-chain resident drafting when enabled so intermediate buffers
  can be read back, and is not a timing route.
- Purpose: diagnose the remaining llama.cpp parity miss after hidden-state
  tracing narrowed the first divergence to the depth-0 MTP block. The first use
  found and fixed the resident MTP RoPE dimension mismatch (`qk_head_dim=256`
  vs model `rope.dimension_count=64`); the attention-debug extension ruled out
  hipEngine's dense-attention kernel math for the seq-position-49 divergence.
- Remove when: llama.cpp tensor/KV parity is either achieved or superseded by a
  more complete graph-tensor trace facility. Keep it default-off until then.

## `--record-target-topk-scores`
- Added 2026-07-03. Bench flag that asks `verify_target_block()` to copy the
  already-materialized full target lm-head logits back to host for block-verifier
  rows and serialize compact `target_lm_head_score_rows` with top-k plus
  candidate-token scores. `--target-score-candidate-tokens` adds explicit
  llama.cpp near-tie tokens to the candidate list. When score rows are present,
  the same diagnostic also emits compact `target_hidden_seed_rows` summaries so
  the scored verifier hidden row can be lined up with llama.cpp `verify_h`
  traces without dumping full hidden vectors in the normal artifact.
- Purpose: diagnose the active llama.cpp parity miss on target verifier
  near-ties without relying on forced-target replay. The first smoke artifact
  `benchmarks/results/2026-07-03-mtp-target-score-capture-smoke.json` populated
  three live target verifier rows on the active `llama-compat` direct-commit
  shape. The hidden-seed follow-up artifact
  `benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-hidden-scores-live.json`
  captures the live task-9/cycle-3/row-2 hidden summary for the `8940` vs `668`
  rank flip.
- Remove when: target hidden-to-logit parity is closed or replaced by a broader
  cross-engine tensor trace. Keep it default-off; the extra full-logit D2H copy
  makes it invalid for retained timing claims.

## `HIPENGINE_GGUF_AR_PACKED_DECODE` (default-on packed decode)
- Added 2026-07-05 as a packed-verifier AR diagnostic, then replaced by the
  retained decode-shaped packed AR path. The current default-on route calls
  `Qwen35GGUFResidentSession.step_batch_native(..., scatter_state=False)` for
  prepared multi-prompt GGUF greedy AR, keeps packed multi-slot state canonical
  across decode cycles, and scatters back only before stream/scalar fallback or
  a changed chunk layout.
- Purpose: provide the first useful GGUF AR c>N server backend after fixing
  default-route request coalescing while preserving each request's canonical
  state. The exact route uses c1 per-slot linear-attention state slices and
  keeps full attention plus MoE/FFN row-batched. Deferred flush copies the full
  live KV prefix instead of only the last dirty row.
- Result: E1/E2 retain direct physical c8 at **127.902/246.872 aggregate
  tok/s** on gfx1151/gfx1100 with complete token/hidden/state/KV and profiler
  gates. E3/F1 then retain C13 as c8+sparse-c8, middle-hole/new admission, and
  real p512/128-output SSE logical c1/c8/c9/c13/serial-c13 at
  **15.701/86.338/57.127/72.522/42.764 tok/s** on gfx1151 and
  **25.583/136.122/88.592/111.380/31.708** on gfx1100. Each packet preserves
  all **189** server prompt/output rows, and every packed static/live route
  records zero serial/resident fallback.
- Remove when: the one-release rollback window ends on both gfx11 targets; both
  correctness/profiler/repetition/live-server triggers are met. Keep only
  registry-resolved unsupported-shape fallback.

## `HIPENGINE_GGUF_AR_PACKED_PREFILL` (default-on packed prompt prefill)
- Added 2026-07-05 after the packed-decode AR route exposed prompt prefill as
  the remaining c>N server AR limiter. The current default-on route calls
  `Qwen35GGUFResidentSession.prefill_batch_native(...)` for multi-prompt GGUF
  greedy AR, packs prompt rows slot-major, scatters the resulting KV/recurrent
  state back to each resident session, and samples only each slot's final
  prompt row before entering packed decode.
- Purpose: remove the serial per-slot prompt-prefill wall from coalesced
  OpenAI-server AR batches without reusing the MTP verifier result contract.
  The rejected verifier-as-prefill probe sampled/copied all prompt rows and
  measured c=8 **50.56 tok/s**, below the retained packed-decode baseline.
- Result: the 2026-07-13 state audit found the old packed full-attention decode
  reduction first changed BF16 layer output at layer 31, then Conv/GDN state at
  layer 32 and live KV at layer 35. Packed prefill now uses the span-aware paged
  prefill reduction below 512 rows; if any slot crosses the AOTriton threshold,
  full attention runs slot-locally with c1 math while linear/MoE remains packed.
  Steady c4 and ragged `[512,64,64,64]` are token/Conv/GDN/live-KV exact. The old
  **65.91/82.41/63.17 tok/s** row predates these gates and exact generated-ID
  accounting; it is historical rather than a current retained baseline.
- Result addendum: E3/F1 covers exact C13 prompt/state/KV plus **189/189** real
  server prompt-ID/usage/output rows per gfx11 target under repeated p512 bursts
  and live c8→c13 admission.
- Remove when: the one-release rollback window ends on both gfx11 targets. Keep
  fallback for total prompt slabs beyond the current packed hidden-row guard.

## `scripts/qwen35_batch_hidden_bisect.py --compare-full-attn-rowchunk-boundary`
- Added 2026-07-09 as a default-off PARO c>N diagnostic mode. It compares two
  native rows=6 batch variants directly: no-rowchunk full attention versus the
  selected full-layer rowchunk repair, using the existing hidden-bisect summary
  machinery but labelling the rowchunk repair as the comparison peer instead of
  an independent c1 oracle.
- Purpose: isolate whether the remaining c6 full-attention rowchunk tax comes
  from KV append/page placement, context-only work, or from a whole-layer
  numerical boundary introduced by rowchunking.
- Result: the L8 trace showed layer 3 full-layer rowchunk output drift still
  under tolerance (`0.000122 max_abs`), layer 4 `attn_input` first over
  tolerance (`0.001953`), and layer 7 `attn_input_pre_qkv` at `0.0078125`.
  Lower-level hidden/KV tracing remains the active diagnostic; rejected suffix
  rowchunk composition has been removed.
- Remove when: the c6 full-layer rowchunk tax is either fixed or the scheduler
  avoids live c6 groups in retained/default operation. Do not use this
  comparison mode for throughput claims.

## `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM`

- Added 2026-07-12 as a default-on PARO prefill rollback/bisection control.
  The retained path creates one lazy nonblocking stream plus two reusable HIP
  events for AOTriton query rows above the proven-safe 256-row bucket, and
  keeps all pre/post work on the caller stream. Setting the flag to `0`
  restores the old same-stream dispatch without changing model math.
- Purpose: isolate AOTriton's high-scratch dispatch from the queue used by
  later low-scratch linear-attention convolution kernels. On gfx1151 the same
  captured convolution changes from about `1.83 ms` after same-stream
  4096-row AOTriton to `119 us` when AOTriton uses the isolated queue. The
  clean 256-row AOTriton trace uses much less scratch (`992/1008` versus
  `2560` bytes) and does not trigger the cliff. The first
  final clean matched prefill A/Bs improve 4K `885.141 -> 1089.031` (+23.03%),
  32K `765.316 -> 906.145` (+18.40%), 64K `621.691 -> 716.775` (+15.29%),
  and 128K `418.838 -> 474.641 tok/s` (+13.32%). Decode stays within
  `-0.16%..+0.12%`, tracked peak is unchanged, and every shape matches sampled
  seed, final hidden, all 30 Conv/GDN state families, and all 10 live K/V
  families. The 1K 256-query negative control never enters isolation and is
  not promoted. Retained evidence:
  `benchmarks/results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json`.
- A clean W7900/gfx1100 transfer matrix measured the earlier global-isolation
  policy in balanced 15-sample legs. Isolated prefill changes by
  `+1.638%/+0.495%/+0.192%` and total measured wall falls by
  `1.653%/0.127%/0.562%` at 512/1K/4K, with byte-exact hidden/state/KV at every
  shape. The merged threshold intentionally leaves the 256-query 512/1K path
  same-stream; its 4K/4096-query result directly validates the scoped gfx1100
  default. Retained evidence:
  `benchmarks/results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json`.
- GPF-4 extends the same selector to GGUF but is **default-off on both
  backends** after the final stability gate rejected its provisional gfx1151
  capability. Clean focus is byte-exact and often fast, but automatic-route
  32K collapses once to 294.254 tok/s, its fresh 1+5 replacement fails to
  finish the warmup after 481 s, and 128K measured run 2 stays GPU-active past
  1200 s. A same-stream 32K control immediately completes normally. Keep `=1`
  only as a diagnostic; do not remove the same-stream GGUF route. Evidence:
  `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-candidate-focus.json`,
  `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-clean-promotion.json`,
  and
  `benchmarks/results/2026-07-14-gfx1151-gguf-prefill-gpf4-final-rejected.json`.
- Remove when: either upstream removes the queue-scratch cliff so isolation is
  unnecessary, or a replacement isolation design passes the full stability
  gate and survives one release cycle. Then delete the rejected duplicate
  route and selector; keep only the proven scheduling policy.

## `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV*`

- Added 2026-08-04 for the Nathan-review P0 gfx1151 promotion. The backend
  capability now defaults one cross-layer BF16 K/V scratch pair on before
  AOTriton when the resident rounded capacity is at most 65,792 tokens and the
  pair is at most 512 MiB. `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV=0` restores
  strided AOTriton; `*_MAX_TOKENS` and `*_MAX_BYTES` can tighten admission.
  Allocation or registry failure also falls back exactly, and gfx1100 remains
  default-off.
- The retained gate is byte-exact for permuted page-copy fixtures and complete
  p512 hidden/GDN/KV/logit state. Copy-inclusive 512/4K/32K/64K full prefill is
  **-0.028%/+0.616%/+3.383%/+7.001%**; the 64K pair adds 134,742,016 tracked
  bytes. Evidence:
  `benchmarks/results/2026-08-04-gfx1151-q4km-aotriton-head-major-prefill.json`.
- Remove the boolean A/B environment branch after one non-regressive release
  checkpoint no longer needs direct bisection. Keep strided AOTriton as the
  mandatory oversized/allocation-denied/unsupported-backend fallback. Keep the
  two capacity controls until a general scratch-budget admission API replaces
  them; do not raise the default beyond 65,792 without a separate 128K-class
  correctness, memory, stability, and copy-inclusive performance gate.

## Laguna D8 row-vector selected gate/up rollback

- Added 2026-07-26 after the exact wave-column MMQ128x32 consumer became the
  gfx1151 production default. The older `mmq128x32_d8_f32_rowvec` mode remains
  an explicit session-level rollback. Clean selector-unset pp512 improves
  **385.602 -> 432.355 tok/s**, cached tracing measures **434.994 tok/s**, and
  the canonical absolute-quality gate passes unchanged at maximum KL
  **0.049542582** and **316/320** top-1.
- Removal trigger is now satisfied by the retained Q4-down checkpoint at
  **448.203 tok/s**. Remove the explicit gfx1151 session rollback during the
  immediate post-publication cleanup; keep the underlying scalar-staged D8
  body only if it still serves a tested unmeasured-backend fallback.

## Laguna Q4 down row-vector rollback

- Added 2026-07-26 after the exact Q4-only wave-column MMQ64x32 consumer
  became the gfx1151 production default. The older
  `mmq64x32_d4_f32_rowvec` mode remains an explicit session-level rollback.
  The implementation-worktree four-mode screen improves Q4-wave/Q6-row
  **433.791 -> 448.945 tok/s (+3.493%)**; Q6 wave columns regress and remain
  rejected. Clean publication confirms **433.081 -> 448.203 tok/s
  (+3.492%)**, max KL **0.049542582**, and the complete trace/lifecycle gate.
- Removal trigger is satisfied. Remove the explicit all-row-vector selector
  during immediate cleanup while retaining the row-vector kernel internally
  for production Q6 unless a different direct-consume mapping wins its own
  quant-isolated gate.

## Laguna Q4 gate/up pair-decode wave-column rollback

- Added 2026-07-26 when direct per-column T16 decode became the next gfx1151
  candidate default. The older `mmq128x32_d8_f32_wavecols` mode remains an
  explicit session-level rollback beside
  `mmq128x32_d8_f32_wavecols_direct`. The candidate preserves resident bytes
  and arithmetic, is BF16-byte identical, improves the actual layer-1 leaf
  **8.107 -> 6.916 ms**, and improves the implementation-worktree pp512 screen
  **447.582 -> 472.533 tok/s**.
- Removal trigger is satisfied: clean selector-unset pp512 improves
  **449.020 -> 474.363 tok/s**, direct all-exact quality/lifecycle pass, and
  cached all-family tracing names the intended direct template. Remove the
  explicit pair-decode session route during immediate cleanup; keep no
  duplicate positive selector solely for historical benchmarking.

## Laguna Q4-down pair-decode wave-column rollback

- Added 2026-07-26 when direct per-column Q4-down decode became the next
  gfx1151 candidate default. The older
  `mmq64x32_d4_f32_wavecols_q4` mode remains explicit rollback beside
  `mmq64x32_d4_f32_wavecols_direct_q4`. The candidate is BF16-byte identical
  and improves the implementation-worktree pp512 screen
  **473.774 -> 483.409 tok/s** while Q6 stays on row-vector production.
- Removal trigger is satisfied: clean selector-unset pp512 improves
  **473.963 -> 480.629 tok/s**, direct all-exact quality/lifecycle pass, and
  cached tracing cuts the Q4 consumer **90.280 -> 71.378 ms**. Remove the
  explicit Q4 pair-decode selector during immediate cleanup. Do not couple
  this cleanup to the independently retained Q6 row-vector kernel.

## Laguna Q6 selected-down 32-row rollback

- Added 2026-07-26 when the exact 64-row/local128 Q6 selected-down body became
  the gfx1151 candidate default. The older
  `mmq64x32_d4_f32_wavecols_direct_q4` session mode retains Q4 direct
  wave-column decode and Q6 32-row row-vector consumption as an explicit
  rollback. The actual runtime-bound Q6 leaf improves **5.260 -> 5.161 ms**,
  dirty pp512 improves **490.105 -> 491.335 tok/s**, and outputs are BF16-bit
  exact.
- Clean publication is complete at **489.110 -> 492.640 tok/s (+0.722%)**
  with all seven paired wins and cached attribution at **493.509 tok/s**.
  Remove the explicit 32-row Q6 session selector after one subsequent retained
  selected-expert checkpoint. Keep the underlying 32-row template if it
  remains the fallback for non-gfx1151 backends or shapes below the 64-row
  admission threshold.

## Laguna serial stable MoE compaction rollback

- Added 2026-07-26 when gfx1151 changed from one-workgroup serial
  expert-major compaction to the byte-identical per-expert parallel
  count/prefix/ballot-scatter path.
- The session-level `serial` selector and registered one-pass kernels remain
  for rollback and unmeasured backends. The production-shape leaf improves
  **0.348880 -> 0.058969 ms (-83.10%)**, and clean publication improves
  **490.824 -> 497.408 tok/s (+1.341%)** with all seven paired wins. Cached
  attribution independently reaches **500.325 tok/s**.
- Remove the explicit gfx1151 serial session selector after the next retained
  exact MoE metadata checkpoint. Keep the serial registry primitive if another
  backend still requires it as an unfused/reference fallback.

## Laguna Q4 gate/up activation-double-buffer candidate selector

- Added 2026-07-26 for the exact
  `mmq128x32_d8_f32_wavecols_direct_doublebuf` session candidate. It
  ping-pongs the 1.5 KB activation tile, removes one of two barriers per K32,
  preserves resident bytes and arithmetic order, and is BF16 byte-exact.
  The actual natural-M512 inclusive leaf improves **6.995 -> 6.907 ms
  (-1.258%)**.
- If clean complete-state A/B is positive, make the double-buffer body the
  gfx1151 package default and remove the redundant positive selector after its
  clean publication; retain the current direct body only as the explicit
  synchronization rollback through one later expert-family checkpoint. If the
  complete-state A/B is negative, remove the double-buffer export, selector,
  harness mode, and template branch immediately.
- The first trigger is satisfied: clean pp512 improves **505.970 -> 507.405
  tok/s (+0.284%)** with **5/7** pair wins and complete-state exactness, and
  the package default is promoted. Remove the redundant positive selector and
  A/B harness after clean selector-unset publication; keep the prior direct
  body as rollback through one subsequent expert-family checkpoint.
- The publication trigger is now satisfied: selector-unset production is
  **505.084 tok/s** median / **504.984 tok/s** minimum, matched A/B remains
  **+0.284%**, and cached tracing cuts gate/up **318.559 -> 314.378 ms**.
  Remove the redundant positive selector and one-purpose A/B harness in the
  next cleanup unit; keep the direct synchronization rollback until the next
  retained expert-family checkpoint.

## Laguna cached-only attention scheduling rollback

- Added 2026-07-26 with the gfx1151
  `LAGUNA_PREFILL_KV_PREAPPEND` capability and the optional
  `prefill_kv_preappend` session override. Complete M128 global tiles and
  pre-wrap SWA tiles append before cached-only qrow4 attention; partial,
  wrapped, verifier, gfx1100, and unmeasured paths preserve the established
  attend-then-append sequence.
- The implementation-tree gate is exact and positive at pp512
  (**507.391 -> 528.771 tok/s, +4.214%**) and at 1K/4K
  (**1.103x/1.047x**). Keep the explicit rollback through clean
  selector-unset publication and one later attention-family checkpoint.
- Clean selector-unset publication is now satisfied at **526.451 tok/s**
  median / **526.288 tok/s** minimum; 1K/4K improve **3.218%/5.701%**, and
  cached tracing cuts attention **219.709 -> 176.580 ms (-19.63%)**.
  The one-later-attention-checkpoint trigger remains open.
- After both triggers pass, remove the public constructor override while
  retaining the backend capability/default and the automatic safety
  fallbacks. Do not remove the source-qualified kernels: wrapped SWA and
  staged verifier transactions still require them.
- The next exact checkpoint adds `LAGUNA_PREFILL_CACHED_META` and the optional
  `prefill_cached_meta` session/cache rollback. It selects metadata-only qrow4
  for every safe SWA tile and global tiles from position 128, retaining the
  established cached body at global position 0. Matched seven-pair pp512
  improves **533.507 -> 542.785 tok/s (+1.739%, 7/7 wins)** and every compared
  output/state digest is exact. Clean selector-unset publication is now
  satisfied at **542.088 tok/s** median / **542.022 tok/s** minimum, with
  1K/4K at **478.856/387.725 tok/s** and traced attention
  **175.802 -> 160.123 ms (-8.92%)**. Remove the public positive selector after
  one later attention-family checkpoint; retain source-qualified registered
  fallbacks for global start 0, partial tiles, wrapped SWA, verifier
  transactions, and unmeasured backends.
- The next exact global-only checkpoint adds
  `LAGUNA_PREFILL_GLOBAL_QROW6` and the optional
  `prefill_global_qrow6` session/cache rollback. It applies only to complete
  preappended global M128 tiles from position 128; global position 0 and every
  SWA tile retain qrow4. Seven complete-state pairs improve
  **546.056 -> 548.774 tok/s (+0.498%, 7/7 wins)** with identical
  logits/hidden/KV/token/cursor. Remove the public positive selector after
  clean selector-unset publication plus one later attention checkpoint. The
  first trigger is satisfied at **547.064/513.180/428.628 tok/s**, with traced
  attention **158.702 -> 152.406 ms (-3.97%)**. Keep automatic qrow4
  fallbacks and do not reintroduce the rejected SWA qrow6 surface.
- The next exact checkpoint adds `LAGUNA_PREFILL_DENSE_INITIAL` and optional
  `prefill_dense_initial` session/cache rollback. It selects only consecutive
  complete M128 initial tiles with no wrap or explicit eviction, preserving
  cached-metadata/current-source fallbacks everywhere else. Seven matched
  complete-state pairs improve **552.144 -> 559.539 tok/s (+1.339%, 5/7
  wins)** with identical logits/hidden/KV/token/cursor. Remove the public
  positive selector after clean selector-unset publication plus one later
  attention checkpoint; keep the automatic safety fallbacks permanently.
  The publication trigger is satisfied at **559.290/523.090/439.044 tok/s**,
  with traced attention **153.226 -> 141.846 ms (-7.43%)**. Keep the rollback
  through one later attention checkpoint.

## Laguna Q6 qmicro resident-layout rollback

- Added 2026-07-26 with the gfx1151 `LAGUNA_Q6_QMICRO` capability and optional
  `q6_qmicro=False` resident-session rollback. Sparse `ffn_down_exps` Q6
  payloads convert from the existing legacy cache once before upload; the
  cache fingerprint, quant key, byte count, root lm-head, and other backends
  stay unchanged.
- The actual-weight leaf is BF16-byte exact and improves natural-M512
  selected prefill **1.65%** plus top-10 exact decode **6.99%**. Clean
  selector-unset publication is now satisfied at **530.447 tok/s** median /
  **525.864 tok/s** minimum, with 1K/4K at **473.118/381.375 tok/s** and
  traced Q6 **126.594 -> 123.473 ms (-2.465%)**. Keep the explicit rollback
  through one later selected-down checkpoint.
- After both triggers pass, remove the public constructor override while
  retaining the backend capability, legacy consumer instantiations required
  by other backends, and the host legacy-to-qmicro cache adapter. If clean
  publication regresses, disable the capability and remove the qmicro runtime
  default before starting another selected-down premise.

## Laguna fused selected-SiLU pack rollback

- Added 2026-07-26 with the gfx1151
  `LAGUNA_FUSED_SELECTED_SILU_PACK` capability and session-local
  `set_fused_selected_silu_pack(...)` rollback. The exact composite preserves
  the standalone SiLU BF16 boundary, reuses selected-down scratch, and leaves
  the required standalone SiLU plus activation-quant primitive fallback
  registered.
- Primitive and production Q4_K/Q6_K MoE equality pass. Seven complete-state
  pp512 pairs are exact and win **7/7**, paired geometric throughput improves
  **0.651%**, and tracing removes 47 launches while cutting the target window
  **10.301 -> 6.377 ms (-38.09%)**. Clean selector-unset publication then
  improves **543.807 -> 546.100 tok/s (+0.422%)** and records the intended 47
  fused/zero standalone calls. Keep the session rollback through one later
  selected-down checkpoint.
- After that trigger passes, remove the public setter while retaining the
  backend capability/default and registered unfused numerical fallback. If
  a later checkpoint regresses, disable the capability before another
  selected-down experiment.

## Laguna Q6 compact-activation rollback

- Added 2026-07-26 with gfx1151 `LAGUNA_Q6_COMPACT_ACTIVATION` and the optional
  `q6_compact_activation=False` resident-session rollback. The Q6-specific
  activation cache omits unused Q8 sum metadata and stores its bounded K16
  quant sums as int16, reducing LDS **5,632 -> 5,120 B** without changing
  arithmetic or resident bytes.
- The actual leaf improves **3.36%**, fifteen complete-state pp512 pairs
  improve **550.584 -> 552.807 tok/s (+0.404%, 15/15 wins)**, and tracing
  cuts Q6 **125.380 -> 119.566 ms (-4.64%)**. Keep the explicit rollback
  through clean selector-unset publication and one later selected-down
  checkpoint.
- Clean selector-unset publication is satisfied at
  **550.625/517.017/431.789 tok/s**, and the production trace records
  **119.384 ms** Q6 plus **71.641 ms** Q4. The one-later-selected-down
  checkpoint trigger remains open.
- After both triggers pass, remove the public constructor override and the
  one-purpose A/B harness. Retain the ordinary 48-byte activation cache only
  where other Q4/Q6 template instantiations or unmeasured backends require it.

## Laguna Q6 half-row activation rollback

- Added 2026-07-26 with gfx1151 `LAGUNA_Q6_HALF_ROW_ACTIVATION` and optional
  `q6_half_row_activation=False` session rollback. It is constrained to the
  qmicro/compact-cache/64-row/local128 Q6 body and changes only activation
  staging ownership.
- The actual 23-layer screen improves **21/23** layers and
  **111.798 -> 111.490 ms (-0.276%)** with exact BF16 output. Complete pp512
  A/B is exact and positive at **552.562 -> 553.018 tok/s (+0.083%)**.
- Keep the rollback through clean selector-unset publication and one later
  selected-down checkpoint. Then remove the public constructor override and
  the one-purpose A/B harness while retaining the backend capability/default.
- Clean selector-unset publication is satisfied at
  **549.150/514.956/430.300 tok/s**; it is headline-neutral, while the clean
  trace preserves the Q6 sub-window win at **119.384 -> 118.568 ms (-0.684%)**.
  The one-later-selected-down checkpoint remains open.

## Laguna Q6 padded-activation rollback

- Added 2026-07-26 with gfx1151
  `LAGUNA_Q6_SKIP_PADDED_ACTIVATION` and optional
  `q6_skip_padded_activation=False` session rollback. It is constrained to
  the exact qmicro/compact/half-row/64-row Q6 body.
- The candidate skips only LDS stores and K16 sums for padded slots that the
  guarded dot and output loops never consume. The actual 23-layer screen is
  exact and improves **112.008 -> 111.806 ms (-0.180%, 19/23 layers)**;
  complete pp512 is exact at **552.983 -> 553.559 tok/s (+0.104%)**.
- Keep the rollback through clean selector-unset publication and one later
  selected-down checkpoint. Then remove the constructor override and
  one-purpose A/B harness while retaining the backend default.
- Clean selector-unset publication is satisfied at
  **551.459/517.307/432.099 tok/s**. The repeated 23-layer and complete-state
  A/B evidence remains positive; the one-later-selected-down checkpoint is
  still open.

## Laguna Q6 qmicro-permute rollback

- Added 2026-07-26 with gfx1151 `LAGUNA_Q6_QMICRO_PERMUTE` and optional
  `q6_qmicro_permute=False` session/profile rollback. It is constrained to
  the exact qmicro/compact/half-row/skip-padded/64-row Q6 body and changes
  neither resident bytes nor arithmetic order.
- The actual layer-1 leaf is BF16-byte exact and improves
  **4.872 -> 4.741 ms (-2.67%)**. Seven complete-state pp512 pairs improve
  **567.998 -> 569.563 tok/s (+0.276%, 5/7 wins)**. Cached tracing observes
  all **115** intended calls with scratch 0 and reduces their total
  **1,138.893 -> 1,124.852 ms (-1.23%)** across 512/1K/4K.
- Keep the rollback through clean selector-unset publication and one later
  selected-down checkpoint. Then remove the constructor/profile override
  while retaining the backend capability and scalar-unpack fallback for
  unmeasured backends and non-production template shapes.

## Laguna Q6 planar-qmicro candidate rollback

- Added 2026-07-26 with gfx1151
  `LAGUNA_Q6_QMICRO_PLANAR` plus the temporary
  `q6_qmicro_planar=True` session/profile selector. It changes only the
  byte order inside each existing 12-byte qmicro record and selects matching
  prefill, direct-decode, and grouped-small-M consumers.
- The exact actual-weight leaf is positive. Two opposite owner-order
  complete-state blocks are aggregate-neutral (+0.013% combined mean,
  +0.139% combined median) with exact logits/hidden/KV/token/cursor, so the
  repo's verified-subwindow rule retains and enables the capability. Keep
  `q6_qmicro_planar=False` only through clean selector-unset publication and
  one later selected-down checkpoint, then remove the constructor/profile
  override and obsolete production permute selector. Preserve legacy and
  interleaved qmicro fallbacks for unmeasured backends and rollback caches.
- Clean selector-unset publication is satisfied at
  **573.354/530.351/446.189 tok/s**, improving every preceding production
  length. The one-later-selected-down checkpoint remains open.

## Laguna source-F16 boundary-fusion rollback

- Added 2026-07-26 with session-local
  `set_f16_boundary_fusion(...)`. The exact range-direct specialization folds
  each separate BF16-to-FP16 cast into its RMSNorm or softplus-gate producer
  while preserving the BF16 rounding boundary bit-for-bit. The registered
  unfused primitives remain the required fallback.
- Primitive composition is exact and improves the two production M512
  boundaries by **1.877x/1.425x**. Seven matched pp512 pairs improve
  **554.909 -> 559.320 tok/s (+0.795%, 6/7 wins)** with identical token and
  logit. The gfx1151 package capability now enables the exact path by default;
  clean selector-unset publication passes at
  **559.554/523.912/440.809 tok/s**. A second seven-pair gate preserves
  logits, hidden states, complete KV, token/logit, and cursor exactly.
  Cached tracing removes all **96** standalone pp512 casts and records
  **1,696** dispatches.
- After promotion and one later source-F16 checkpoint, remove the public
  setter and one-purpose A/B harness while retaining the backend capability,
  fused registered kernels, and registered unfused numerical fallback.

## Laguna Q6 selected-down integer-WMMA leaf selector

- Added 2026-07-26 as the optional `integer_wmma` wrapper argument and matching
  actual-weight leaf A/B mode. An unspecified value selects integer WMMA only
  for the already-constrained planar-qmicro/compact/half-row/skip-padded
  row64 path; `False` preserves the packed-dot comparator.
- The CPU-reference gate and actual layer-1 leaf are BF16-byte exact. Twenty-one
  counter-rotated natural-M512 pairs improve **4.7654 -> 4.5655 ms (-4.20%,
  21/21 wins)** with zero resident-byte change.
- Keep the explicit comparator through clean selector-unset publication and
  one refreshed selected-down family trace. Then remove the one-purpose leaf
  A/B mode and make the admitted production specialization unconditional
  inside the planar-row64 wrapper while retaining packed-dot bodies for all
  other template shapes and unmeasured backends.
- Clean selector-unset publication is satisfied at
  **576.137/543.213/459.054 tok/s**, improving every preceding production
  length with deterministic tokens, exact positions, and complete allocation
  return. The refreshed trace cuts selected down
  **189.049 -> 181.583 ms (-3.95%)** and the 115-call Q6 window
  **1,124.852 -> 792.625 ms (-29.54%)**.
- The immediately queued activation-fragment hoist is also admitted at the
  leaf gate: **4.5645 -> 4.5126 ms (-1.136%, 20/21 wins)**, zero BF16
  mismatches, and unchanged local128/VGPR96/LDS5120B/scratch0 resources.
  Clean selector-unset publication is satisfied at
  **577.396/545.366/459.716 tok/s**, improving every required length. Keep
  `wmma_hoist_activation=False` only through one refreshed family trace. Then
  remove both one-purpose
  wrapper selectors and collapse the admitted hoisted specialization into the
  unconditional planar-row64 gfx1151 production route.
- The refreshed family trace is now satisfied: the 115-call Q6 window improves
  **792.625 -> 779.709 ms (-1.63%)** with unchanged resources. Selector
  collapse is unblocked and should accompany the next accepted Q6 body or the
  post-campaign cleanup, whichever comes first.

## Laguna Q6 WMMA weight-prefetch rollback

- Added 2026-07-27 as gfx1151
  `LAGUNA_Q6_WMMA_PREFETCH_WEIGHT`, the optional
  `q6_wmma_prefetch_weight` resident-session field/setter, and two
  one-purpose benchmark modes. It applies only to the admitted
  planar-qmicro/compact/half-row/skip-padded/integer-WMMA/activation-hoist
  row64 specialization.
- The candidate overlaps the next K32 planar-qmicro record plus `d`/scale
  global loads with current fragment compute. It adds no resident bytes,
  second LDS plane, arithmetic change, or scratch. The exact layer leaf
  improves **4.518 -> 4.104 ms (-9.156%)**; seven complete-state pairs
  improve **618.294 -> 623.900 tok/s (+0.907%)**. The resource cost is
  VGPR **96 -> 104** with LDS fixed at 5,120 bytes.
- Clean selector-unset publication is satisfied at **636.073 tok/s** pp512,
  and the subsequent all-family trace cuts the exact 23-call Q6 body
  **112.746 -> 101.963 ms (-9.564%)**. Collapse the prefetch and
  already-satisfied activation-hoist selectors into the unconditional
  gfx1151 production specialization with the next retained Q6 body or
  post-campaign cleanup; retain the non-prefetch template only for unmeasured
  backends and non-production geometries.
- The retained successor adds `LAGUNA_Q6_WMMA_PREFETCH_ACTIVATION`, the
  `q6_wmma_prefetch_activation` session field/setter, and a candidate wrapper
  solely for A/B rollback. It improves clean pp512
  **636.073 -> 639.114 tok/s** and the 23-call Q6 body
  **101.963 -> 100.367 ms**, with VGPR104 -> 112 and unchanged
  LDS/scratch/resident bytes. Collapse weight prefetch, activation prefetch,
  and activation hoist into one unconditional gfx1151 production symbol after
  the current 700 campaign no longer needs the three independent bisection
  points; preserve one non-prefetch fallback for unmeasured geometries.
- The published exact successor adds
  `LAGUNA_Q6_PRECOMPUTED_ACTIVATION_SUMS`, a resident-session rollback, and
  distinct producer/consumer exports. It reuses Q6's unused D4 raw-sum field
  for two `int16` K16 quant sums without changing the 160-byte ABI. The
  actual inclusive leaf improves **4.1501 -> 4.1162 ms (-0.818%)** and 11
  complete-state pp512 pairs save **1.407 ms** at the paired median with
  **6/11 wins**. Clean selector-unset 512/1K/4K subsequently improves
  **645.803/575.942/468.311 -> 647.207/576.799/468.431 tok/s**, while the
  traced 23-call Q6 body falls **100.367 -> 99.459 ms (-0.905%)**. Both
  promotion gates pass. Keep the rollback through the current 700 campaign,
  then collapse this and the three older Q6 WMMA selectors into one
  unconditional gfx1151 production symbol.

## Laguna Q4 raw-nibble P8 prefetch rollback

- Added 2026-07-27 as gfx1151 mode
  `mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512` and P8
  wrapper export. It carries only the next K32 interval's eight raw T16
  nibble words, preserves the existing D8 activation double buffer, and
  selects P8 only at producer chunks of at least 512 rows.
- The exact actual M512 leaf improves **6.8727 -> 6.7389 ms (-1.948%)**;
  seven complete-state pp512 pairs improve
  **636.367 -> 640.003 tok/s (+0.571%, 7/7 wins)**. M256 regresses
  **0.211%**, so the non-prefetch specialization remains required below the
  threshold rather than serving only as historical rollback.
- After clean selector-unset publication and one refreshed trace, collapse
  the user-visible positive mode into the gfx1151 shape capability. Keep one
  tested non-prefetch specialization for sub-512/unmeasured geometries, and
  remove the one-purpose leaf/complete-state comparison modes when they no
  longer serve campaign bisection.
- The clean publication trigger is satisfied at
  **643.554/573.066/466.290 tok/s**, improving all required lengths with
  exact lifecycle. The refreshed trace trigger is also satisfied: selected
  Q4 gate/up falls **337.395 -> 333.701 ms** and the P8 resource tuple is
  confirmed. Keep the explicit old/new symbols only through the immediately
  following compact-metadata-prefetch A/B; then collapse the winning >=512
  shape policy as described above.
- Compact metadata prefetch is rejected and fully removed at
  **6.7265 -> 7.0330 ms (+4.556%)**, so that bisection dependency is closed.
  Collapse the retained P8/non-P8 shape capability in the next cleanup unit;
  preserve the non-P8 specialization below 512 because it is measured faster.

## Laguna Q4 selected-down raw-nibble P8 rollback

- Added 2026-07-27 as the gfx1151 shape mode
  `mmq64x64_d4_f32_q6_wavecols_direct_rawprefetch_q4_ge512`. It reuses the
  exact gate/up P8 mechanism in the single-output 64x32/local64 Q4-down body,
  adds no resident bytes/LDS/scratch, and preserves the previous body below
  512 producer rows.
- The M512 trace cuts 72 Q4-down launches
  **217.416 -> 212.090 ms (-2.450%)**. Seven complete-state pp512 pairs
  improve **639.574 -> 643.166 tok/s (+0.562%, 7/7 wins)** with every output
  and state digest exact. Clean selector-unset publication is
  **643.141/573.717/466.913 tok/s**, with pp512 flat within run variance.
- The clean publication trigger is satisfied. After a refreshed all-family
  trace, collapse the explicit positive selector into the gfx1151 shape
  capability. Preserve the
  non-prefetch symbol for sub-512 rows and rollback/bisection because the
  admitted route is deliberately shape-qualified.

## Laguna Q4 precomputed activation-sum rollback

- Added 2026-07-27 as an exact gfx1151 M512+ default behind
  `LAGUNA_Q4_PRECOMPUTED_ACTIVATION_SUMS`. The D8 producer writes exact K16
  integer sums to a bounded activation-only `int16` sidecar, and the retained
  P8 consumer loads them instead of rebuilding them in 16 output-column
  workgroups.
- The actual leaf improves **0.434%/0.933%** at M256/M512. Eleven
  complete-state pp512 pairs save **2.491 ms** at the paired median with
  **9/11 wins** and exact state. The maximum scratch plan grows by
  **786,432 bytes**; resident weights do not grow.
- Keep the capability, setter, old consumer, and two-symbol pack/consumer
  split through the active 700 campaign for rollback and attribution. After
  clean cached trace plus selector-unset 512/1K/4K publication, collapse the
  positive path into the gfx1151 M512 shape policy. Remove the sidecar and
  candidate symbols if clean publication regresses any required length or the
  traced gate/up family does not improve.
- The clean trace/publication trigger is satisfied: selected gate/up falls
  **334.229 -> 330.720 ms (-1.050%)**, pp512 reaches **649.791 tok/s**, and
  the aggregate-flat 1K result is positive in same-process paired wall by
  **4.428 ms (7/11 wins)**. Keep rollback through the 700 campaign, then
  collapse the positive path as described above.

## Laguna packed attention-output rollback

- Added 2026-07-27 as the gfx1151 default candidate behind
  `LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE`. Qualified
  dense-initial M128 library PV tiles write directly to head-major context
  slices; one exact packed-aware softplus gate restores generic BF16 or
  FP16-via-BF16 output. The old output transpose remains available through
  the session constructor for clean A/B and rollback.
- Eleven complete-state pp512 pairs are exact and improve independent medians
  **645.735 -> 647.920 tok/s (+0.338%)**, but paired-median wall is only
  **-0.057 ms** with **6/11 wins**. Keep the capability, optional
  `unpack_output`, generic gate, packed gate, and dedicated A/B harness only
  through the immediate clean trace. Remove the candidate if the trace does
  not eliminate all **144** output-transpose launches or if the replacement
  gate repays the saved sub-window. On admission, preserve the generic route
  for fallback attention and unmeasured backends, but collapse the dedicated
  A/B harness after the active 700 campaign.
- The repaired clean trace satisfies admission: all **144** output transposes
  are absent, dispatches fall **2,417 -> 2,273**, and transpose plus gate
  improves **11.240 -> 10.318 ms (-8.20%)**. Keep the generic route because
  fallback attention and unmeasured backends require generic output; remove
  only the dedicated A/B harness and constructor override after the active
  700 campaign no longer needs this bisection point.

## Laguna gfx1151 head/KV transfer selector

- Added 2026-07-28 as a default-off gfx1151 screen for the retained gfx1100
  current-P4 head-RMSNorm + partial-RoPE + BF16 KV-write composites. The
  registry aliases and `--head-kv-fusion` profiler selector exist only to run
  the architecture-local p512/d128 and trace gates.
- If the candidate is exact and improves the complete decode median, promote
  `LAGUNA_HEAD_KV_FUSION` and keep explicit False as the unfused rollback
  through the next decode campaign. If it is neutral or negative, restore the
  gfx1151 alias exclusions and remove the profiler selector and candidate
  tests instead of carrying a dead backend path.
- The promotion trigger is satisfied: clean p512/d128 improves
  **11.466687 -> 11.483587 tok/s (+0.147%)**, and all three candidate samples
  beat all three controls with identical trajectory/state. Keep explicit
  False through the next split-attention transfer screen, then collapse the
  positive CLI selector if it is no longer needed for attribution.

## Laguna gfx1151 split-attention transfer selector

- Added 2026-07-28 as a default-off gfx1151 screen for the retained gfx1100
  exact global/SWA split-attention bundle. `--decode-split-attention` requests
  global/SWA/tile16 thresholds **127/65/257**, gated reducers, and the
  wave-local SWA reducer together; package defaults remain unsplit.
- If the clean p512/d128 candidate is exact and faster, promote the complete
  threshold/capability bundle and retain `--no-decode-split-attention` through
  one follow-up context crossover check. If the full-model gate is neutral or
  negative, restore all ten gfx1151 alias exclusions and remove the profiler
  selector/test rather than carrying partial split registrations.
- The promotion trigger is satisfied: clean p512/d128 improves
  **11.485885 -> 14.533955 tok/s (+26.538%)**, saves **2.318892 seconds** over
  127 calls, and preserves the complete trajectory/lifecycle. Keep the
  explicit serial rollback until a short context-crossover check confirms the
  inherited **127/65/257** thresholds on gfx1151.

## Laguna source-F16 decode fixed-K selector

- Added 2026-07-28 as
  `HIPENGINE_LAGUNA_F16_DECODE=auto|gemv|onebarrier|fixedk`.
  gfx1151 `auto` now selects exact K3072/K6144/K9216 fixed-K single/triple
  siblings for rows==1. Explicit `onebarrier` retains the prior exact generic-K
  default, `gemv` retains the two-barrier root, and peer backends retain GEMV.
- The primitive preserves every output byte and improves all six natural
  roles **0.57-1.71%**, with weighted leaf family
  **31.316 -> 31.097 ms/token (-0.698%)**. If full-state, cached trace, and
  clean p512/d128 gates transfer positively, keep `gemv` only through the
  active decode campaign and then collapse the gfx1151 positive selection
  into the default dispatch. Remove the candidate and selector if any
  production gate regresses.
- The promotion trigger is satisfied: seven same-session p512/d128 pairs move
  **14.758912 -> 14.800191 tok/s (+0.280%)**, every candidate beats every
  control, all generated hashes match, and cached tracing records exactly
  **18,288 = 144 x 127** candidate calls with zero retained decode GEMVs.
  Keep explicit `gemv` through LD-2, then collapse the positive selector as
  described above.
- The fixed-K successor also satisfies promotion: the six-role family moves
  **30.952 -> 24.482 ms/token (-20.90%)**, seven exact production pairs move
  **14.786076 -> 16.391201 tok/s (+10.856%)**, and cache-only tracing records
  **18,288/18,288** fixed-K calls with no fallback. Keep `onebarrier` for one
  release/bisection window; then remove positive `fixedk` selector semantics
  and make it unconditional for qualified gfx1151 K widths. Permanently retain
  the generic registered path for unsupported K/rows/backends.

## Laguna selected natural tile8 decode selector

- Added 2026-07-28 as
  `LagunaGGUFResidentSession(..., use_selected_natural_tile8_decode=False)`
  plus `set_selected_natural_tile8_decode(...)` and
  `--compare-selected-natural-tile8-decode`. False restores the exact
  16-column natural gate/up owner; non-natural shapes and peer backends retain
  the registered generic/natural fallbacks.
- The promotion trigger is satisfied. Actual-weight tile8 improves
  **5.35-7.13%** with zero BF16 mismatches. Seven p512/d128 pairs improve
  **16.991621 -> 17.007001 tok/s (+0.091%)** with 7/7 wins and exact state.
  Cache-only tracing records all **5,969** intended calls, zero fallback, and
  local128/VGPR96/LDS512/scratch0.
- Keep the false rollback and comparison flag through one decode campaign
  checkpoint. Then remove positive selector semantics and make tile8
  unconditional for the qualified gfx1151 shape. Permanently retain the
  natural tile16 and generic registered owners for unsupported
  shapes/backends.

## Laguna gfx1151 exact SWA exp-domain selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512` field,
  `LagunaGGUFResidentSession.set_decode_swa_assume_exp(...)`, and
  `--compare-swa-assume-exp`. It leaves compiler `expf` intact and only asserts
  the exact score-minus-maximum domain at saturated
  72Q/8KV/D128/SWA512. The retained direct-store body is the rollback.
- Promote only if seven tracked-clean p512/d128 pairs improve and every
  generated ID/state field remains exact. On promotion, make the gfx1151
  capability select the assumed-domain sibling, retain explicit false rollback
  through the decode campaign, then collapse the positive selector after one
  checkpoint. On rejection, remove the field, setter, comparison lane, and
  registered candidate primitive.
- Promotion gate satisfied 2026-07-29: **19.140826 -> 19.245912 tok/s
  (+0.549%, -0.285 ms/token)** with 7/7 wins, complete sample separation, and
  exact IDs/state. gfx1151 now defaults the capability true. Keep explicit
  false rollback through this decode campaign; then remove positive selector
  semantics while retaining the generic registered rollback.

## Laguna gfx1151 mixed32 SWA decode selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32(...)`, and
  `--compare-swa-mixed32`. False restores the exact 24-block GQA3
  assumed-domain owner; shorter/non-natural shapes and peer backends retain
  their existing registered routes.
- Promotion gate satisfied: the exact 32-block 2+2+2+3 owner improves the
  leaf **5.41%** and all seven resident p512/d128 pairs
  **19.268862 -> 19.371717 tok/s (+0.534%)** with identical trajectories and
  allocation lifecycle.
- Keep the explicit false rollback through clean production publication and
  the next attention campaign checkpoint. Then remove positive selector
  semantics and make mixed32 unconditional for the qualified gfx1151 shape,
  while permanently retaining the registered 24-block GQA3 and generic
  fallbacks.

## Laguna gfx1151 mixed32 exact exp4 selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp4(...)`, and
  `--compare-swa-mixed32-exp4`. False restores the retained serial-exp
  mixed32 sibling; it never selects a different geometry, arithmetic order,
  or non-natural route.
- Promotion gate satisfied: the exact four-lane issue schedule improves the
  leaf **0.091487 -> 0.089135 ms (-2.57%)** and all seven resident p512/d128
  pairs **19.368030 -> 19.432503 tok/s (+0.333%, -0.171 ms/token)** with
  complete sample separation and identical trajectories/state/lifecycle.
- Keep the explicit false rollback through clean production publication and
  the next attention checkpoint. Then collapse the positive selector into the
  qualified gfx1151 mixed32 dispatch while permanently retaining the
  registered serial-exp mixed32 sibling for compiler/codegen bisection.

## Laguna gfx1151 mixed32 exact exp8 selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp8(...)`, and
  `--compare-swa-mixed32-exp8`. False restores retained exp4; it changes no
  geometry, arithmetic order, or non-natural route.
- Promotion gate satisfied: the exact eight-lane issue schedule improves the
  leaf **0.089191 -> 0.083755 ms (-6.09%)**, stable cached trace
  **83.557 -> 78.667 us (-5.85%)**, and all seven resident p512/d128 pairs
  **19.427449 -> 19.510986 tok/s (+0.430%, -0.220 ms/token)** with complete
  sample separation and identical trajectories/state/lifecycle.
- Clean selector-unset production is published at **19.515697 tok/s**
  (**+0.470% / -0.241 ms/token** over clean exp4), with exact repeated state.
  Keep the explicit false rollback through the next attention checkpoint.
  Then collapse the positive selector into the qualified gfx1151 mixed32
  dispatch while retaining registered exp4 for compiler/codegen bisection.

## Laguna gfx1151 mixed32 exact exp16 selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp16(...)`, and
  `--compare-swa-mixed32-exp16`. False restores retained exp8; it changes no
  geometry, arithmetic order, or non-natural route.
- Promotion gate satisfied: exact sixteen-lane issue improves the leaf
  **0.083740 -> 0.082224 ms (-1.81%)**, stable cached trace
  **78.814 -> 77.265 us (-1.97%)**, and all seven resident p512/d128 pairs
  **19.506557 -> 19.523370 tok/s (+0.0862%, -0.0441 ms/token)** with complete
  sample separation and identical trajectories/state/lifecycle.
- Clean selector-unset production is published at **19.530105 tok/s**
  (**+0.0738% / -0.0378 ms/token** over clean exp8), with exact repeated
  state. Keep the explicit false rollback through the next attention
  checkpoint. Then collapse the positive selector into the qualified gfx1151
  mixed32 dispatch while retaining registered exp8 for compiler/codegen
  bisection.

## Laguna gfx1151 mixed32 exact exp32 selector

- Added 2026-07-29 as the session-scoped
  `LagunaKVCache.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512`
  field, `LagunaGGUFResidentSession.set_decode_swa_mixed32_exp32(...)`, and
  `--compare-swa-mixed32-exp32`. False restores retained exp16.
- Promotion gate satisfied: exact wave32 issue improves the leaf
  **0.082313 -> 0.081551 ms (-0.93%)**, stable trace
  **77.185 -> 76.838 us (-0.45%)**, and all seven resident pairs
  **19.524103 -> 19.538164 tok/s (+0.0720%, -0.0369 ms/token)** with complete
  separation and exact state.
- Clean selector-unset production is aggregate-flat at **19.530839 tok/s**
  (**+0.0038%** versus exp16), so retain on the stronger separated A/B and
  leaf/trace evidence. Keep false rollback through the next wall census.
  Then collapse positive selector semantics while retaining exp16 as the
  compiler/codegen rollback. Wave32 closes the natural issue-width screen.

## Laguna gfx1151 exact global exp-domain selector

- Added 2026-07-29 as the default-off
  `LagunaKVCache.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_assume_exp(...)`, and
  `--compare-global-assume-exp`. It changes no arithmetic and applies only to
  natural gated capacity-4096 global decode through live4000; the retained
  direct-store body is the rollback.
- Promote only if the tracked-clean counterbalanced p512/d128 screen is
  exact and positive. On promotion, default the gfx1151 capability true and
  retain explicit false rollback through the decode campaign. On rejection,
  remove the field, setter, comparison lane, capability, and registered
  primitive rather than carrying a dead global fork.
- Promotion gate satisfied 2026-07-29: all seven pairs improve
  **19.235596 -> 19.243968 tok/s (+0.0435%, -0.0226 ms/token)** with exact
  IDs/state/lifecycle. gfx1151 now defaults the capability true. Keep explicit
  false rollback through this campaign, then collapse positive selector
  semantics while retaining the generic-domain registered rollback.

## Laguna gfx1151 exact global exp32 selector

- Added 2026-07-29 as the session-scoped
  `LagunaKVCache.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_exp32(...)`, and
  `--compare-global-exp32`. False restores the retained serial-issue
  assume-exp sibling.
- The primitive is byte-exact at live 513/576/639 and improves the leaf
  **2.25%/3.22%/3.79%**. Promote only if cached tracing is spill-free and the
  tracked-clean counterbalanced p512/d128 screen is exact and positive.
- Promotion gate satisfied 2026-07-29: cached tracing improves
  **88.486 -> 85.601 us (-3.26%)** at VGPR56/scratch0, and all seven resident
  pairs improve **19.547209 -> 19.556569 tok/s (+0.0479%)** with exact state.
- After the decode campaign, collapse positive selector semantics while
  retaining the serial-issue primitive as the compiler/codegen rollback.

## Laguna gfx1151 exact global mixed32 selector

- Added 2026-07-29 as the session-scoped
  `LagunaKVCache.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape`
  field, `LagunaGGUFResidentSession.set_decode_global_mixed32(...)`, and
  `--compare-global-mixed32`. False restores the retained 24-block GQA2
  exp32 owner.
- Promotion gate satisfied: live513/576/639 leaves improve
  **5.19%/8.39%/8.39%**, the cached trace names the intended
  grid8192/local256/VGPR56/scratch0 specialization, and all seven resident
  pairs improve **19.641357 -> 19.668893 tok/s (+0.1402%, -0.0713
  ms/token)** with byte-exact attention and exact generated state.
- Clean selector-unset production is **19.667705 tok/s**, **+0.1917% /
  -0.0975 ms/token** over the preceding packet, with exact repeated state.
- The 40-block ordinary-grid point is rejected at live576/live639 and removed.
  The pair-local256 plus singleton-local128 split-launch screen is also removed
  after exact leaves regress **142.26-154.66%**. Keep the explicit false
  rollback through the next clean attention census and the resulting
  single-launch screen. Then collapse positive selector semantics while
  retaining the 24-block GQA2 exp32 primitive as the exact
  compiler/occupancy rollback.

## Laguna gfx1151 SWA producer-value-tail diagnostic

- Added 2026-07-30 as a separately registered exact mixed40 attention
  primitive. Finished tail probability-producer waves copy the last 64
  staged-V vectors on pair owners and the last 32 on singleton owners,
  balancing the copy loop without changing attention arithmetic.
- The 21x100 leaf improves **3.270%** at unchanged kernel resources, but seven
  resident p512/d128 pairs reject standalone ownership at
  **20.509962 -> 20.507264 tok/s (-0.01316%)**, only **4/7** wins.
- Keep the symbol only through the current attention campaign for a compounded
  producer/denominator/PV scheduling screen. Remove it, its wrapper, registry
  key, leaf selector, and focused test call if no compound candidate uses it;
  never make it the default independently.

## Laguna gfx1151 Q4T16 dual-interleaved expert production rollback

- Originated 2026-07-30 as an unregistered exact leaf primitive and
  actual-weight harness mode. It is now registered and production-owned. It is
  byte-neutral only when it replaces both resident gate/up T16 matrices;
  keeping it as a sidecar would add about **43.76 GB** across 47 routed layers.
- The exact byte-neutral D8 MMQ128x32 prefill consumer now wins M128/M256/M512
  **2.460%/2.034%/1.187%**, so keep the dual-layout host helpers, decode and
  prefill symbols/wrappers, leaf modes, and focused fixtures through resident
  integration. Remove the two ordinary expert gate/up allocations when the
  paired allocation becomes canonical; never add a decode-only production
  sidecar. Remove the temporary comparison selectors and legacy cache
  acceptance after tracked-clean decode plus the prefill sweep publish.
- Resident integration is now default and byte-neutral. The temporary rollback
  is `scripts/laguna_long_context_profile.py --ordinary-q4-expert-t16` plus
  `LagunaGGUFResidentSession(...,
  use_q4_expert_t16_dual_interleaved=False)`. Same-revision production improves
  **22.130173 -> 22.260802 tok/s** with exact state and unchanged
  **79,022,522,196-byte** residency. Tracked-clean publication reaches
  **22.262504 tok/s** decode and **656.990 tok/s** pp512.
- The existing repacked cache still stores ordinary gate and up T16 payloads,
  so default load interleaves them on the host and rises
  **92.084 -> 142.902 seconds**. Retain the rollback until a versioned paired
  cache payload can be written without duplicating the roughly **43.76 GB**
  resident expert pair. After that cache migration and one clean decode/prefill
  recertification, remove the ordinary CLI/session seam and ordinary paired
  materializer fallback.

## Laguna gfx1151 dense/shared dual-interleaved T16 selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_q4_decode_t16_dual_interleaved(...)` and
  `--compare-q4-decode-t16-dual-interleaved`. The gfx1151 capability defaults
  on; false restores the exact resident-pack8 fused gate/up path.
- The temporary same-resident diagnostic duplicated the old separate sidecars
  only for seven counterbalanced pairs. All seven candidates win
  **21.898558 -> 21.954474 tok/s (+0.25534%)** with exact state.
  Production removes the duplicates: 48 paired payloads replace 96 separate
  payloads at the same **214,597,632-byte** auxiliary residency.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset publication and the first post-retention wall census. Retain
  the architecture capability, paired materializer contract, registered tile2
  owner, and pack8 exact fallback.

## Laguna gfx1151 selected-Q4 scalar-Q rollback

- Added 2026-07-30 after exact adjacent-column T16 Q-payload reuse improved the
  actual gate/up leaf **1.945%** and all seven p512/d128 model pairs.
  Production keeps the existing
  `selected_dual_t16_natural_tile8_parallel_silu...` variant name; the
  separately registered `...parallel_silu_scalarq...` sibling restores the
  pre-change one-nibble-per-load body exactly.
- Keep scalar-Q through clean selector-unset publication and one later decode
  wall census. Remove its launcher, wrapper, registry key, and comparison-only
  exactness call if pair-Q remains positive and spill-free after that census.
- **Clean publication passed:** selector-unset production improves
  **20.800509 -> 20.830515 tok/s (+0.14426%)** with exact repeated state.
  Keep scalar-Q only through the next decode wall census.

## Laguna gfx1151 dense/shared Q4 T16 sidecar selector

- Added 2026-07-30 as
  `LagunaGGUFResidentSession.set_q4_decode_t16_sidecar(...)` and
  `--compare-q4-decode-t16-sidecar`. The gfx1151 capability defaults on;
  false restores the resident-pack8 fused gate/up plus SiLU path.
- The additive 96-weight sidecar passes exact actual-weight leaves and all
  seven same-resident p512/d128 pairs, improving
  **21.311596 -> 21.852204 tok/s (+2.53669%)**.
- Remove the comparison CLI and session setter after tracked-clean
  selector-unset publication and its first wall census. Retain the
  architecture capability, materialized T16 sidecars, and registered pack8
  fallback.
- Tracked-clean publication passes at **21.851538 tok/s**, with all three
  trajectories exact and the capability active. Remove the comparison seam
  after the pending post-sidecar wall census.
- The post-sidecar census passes at **43.972461 ms/token** kernel sum and
  **46.007636 ms/token** span. The comparison seam is now eligible for
  removal; retain only the capability, T16 materialization, and pack8 fallback.

## Laguna gfx1151 selected gate/up halfdot rollback

- Added 2026-07-31 as constructor override
  `use_selected_halfdot_decode`. The gfx1151 capability defaults on; `false`
  restores the exact interleaved pair-coefficient F32 owner without changing
  resident bytes or the surrounding fused SiLU boundary.
- The quality-gated candidate improves the actual layer-1 leaf
  **0.110406 -> 0.090545 ms (-17.989%)** and all seven same-resident p512/d128
  pairs **22.999793 -> 23.084044 tok/s (+0.36631%)**. Recurrent
  candidate-vs-exact quality is max KL **0.008202** and top-1 **93.75%**.
- The temporary session setter and comparison CLI are removed after
  default-on production measures **23.089693 tok/s**. Retain the constructor
  rollback, gfx1151 capability, both four-axis registrations, exact fallback,
  and diagnostic leaf/quality selectors until a later exact path matches or
  exceeds the candidate.

## Laguna gfx1151 LC-D3 long-global geometry and context-split prototypes

- Added 2026-08-02 while screening the second through fifth LC-D3 milestones.
  The HIP source and leaf harness retain internal exact D64/V64, D64/V32, local256
  D32/V64, direct-qhead D32, GQA6 score-tile4, and context-split/merge
  variants. Separately registered D32 and D64 4,096-token context owners are
  not part of that dead set: gfx1151 production selects D64 at admitted layer
  28 and layers 32/36/40/44, while D32 stays the peer-backend/rollback owner.
  Exact deferred-normalization D32/V64 is the earlier-layer fallback.
- Every remaining internal exact geometry alternative is slower at a
  mandatory depth. The
  all-layer context-split route is faster but fails the same-state 127-step
  quality gate at maximum KL **0.687034** versus the **0.05** ceiling; an
  8,192-token sibling fails at **0.776134**, and exact sparse BF16-boundary
  repair is **7.25x** slower than the retained GQA6 owner. Kahan compensation
  still fails all-layer quality at **0.450784**. A compensated layer-28-plus
  route passes teacher forcing and is faster through 128K, but changes the
  mandatory recurrent 128K final token/hash and is removed. The narrower
  compensated-layer-28 symbol is retained: it leaves uncompensated layers
  32/36/40/44 unchanged, passes at maximum KL **0.007761**, and preserves the
  128K trajectory while improving it **1.667%**. Both admitted arithmetic
  variants now use byte-identical D64 PV geometry, improving production
  16K/64K/128K another **1.705%/2.558%/2.348%**. These five measured layers
  are admitted; layers 0..24 remain exact. A subsequent isolated compensated
  layer-24 screen is also removed after teacher70 reaches **0.235600 max KL**
  despite **69/70 top-1**; no temporary capability or runtime branch remains.
  A D128/local1024 sibling is likewise fully removed: its isolated active
  leaves improve **0.93-1.85%**, but d16K is flat and the mandatory 128K
  recurrent gate aborts on an out-of-vocabulary token rather than preserving
  the retained trajectory.
  An exact D64/local1024 retry is removed as well: increasing loader capacity
  does not repair D64's lost workgroup parallelism and regresses D32 by
  **12.28-13.12%** across live4K-d128K.
  Exact D32 V-stage128 is also removed after improving 16K/64K but regressing
  the required 4K/128K depths; V64 remains the only exact registered owner.
  Non-temporal exact-D32 V loads are removed after regressing three of four
  depths; cached cross-workgroup V behavior remains beneficial.
  Cached 32-byte ext-vector V loads are removed for the same mixed-depth
  result; native uint4 remains the exact D32 load contract.
  Exact D32/local1024 is removed after losing to local512 at every depth;
  further PV local-size permutations are closed.
  Exact score wave8/local256 and wave4/local128 packing is removed after both
  regress every 4K-128K leaf; the production score owner remains one local32
  wave per KV-head/token.
  Exact PV query-pair ILP is removed after halving output waves regresses every
  depth by about **21-22%**; one query/output accumulator per wave remains the
  required D32 ownership.
  A plain four-token PV source unroll and its volatile repair are also removed.
  The plain unroll let the compiler form four partial accumulators and failed
  the mandatory recurrent 128K hash; the volatile scalar restored ordering but
  made the leaf 3-4x slower. Production retains only four-token operand
  prefetch followed by explicit ordered `v_fma_f32`, which improves exact
  leaves **23.13-34.03%** and complete 16K/64K/128K
  **4.313%/9.400%/12.851%** with the established trajectory.
  Ordered prefetch8 is removed after improving the exact leaf at all depths and
  complete 16K/64K **0.188%/0.177%**, but regressing mandatory 128K
  **0.140%**. Four tokens is the measured prefetch-distance knee.
  Direct output-wave probability loads are removed after becoming
  **2.88-3.02x slower** at every leaf depth; loader-wave LDS probability
  broadcast remains required latency hiding.
  Tied-accumulator `v_fmac_f32_e32` is removed after mixed leaf direction;
  production retains the three-source ordered `v_fma_f32` spelling.
  Score-sentinel visibility is removed after its leaf win reverses to complete
  4K/16K/64K regressions; the denominator keeps `physical_scratch` checks.
- Remove rejected geometries, the unused repair kernel/mask calculation, the
  8,192-token experiment, and non-registered symbols after one clean release
  plus the next exact/precision LC-D3 iteration. Preserve the registered
  uncompensated 4,096-token production owner, the isolated compensated-layer
  owner, their gfx1151 capabilities, compact evidence, and same-state quality
  harness. The exact normalized/deferred GQA6 siblings and generic
  complete-`KVLiveSpans` split route remain required rollback/fallback paths.

## Moonshine exact two-region continuous scheduler — opt-in strict fallback

- The apparent uniform-t256 quality regression was a docs-side three-frame mask
  undercount, not scheduler arithmetic. With the corrected installed mask,
  uniform t256 and exact two-region both match exact custom CUDA on all 266
  Japanese FLEURS transcripts. Uniform is the production default because it is
  faster at every matched c=2-512 capacity and uses half the workspace.
- `MoonshineCudaExactContinuousBatchRuntime` remains an explicit off-by-default
  t32/t256 arithmetic fallback and bisection route. Reassess removal after one
  clean release if no deployment requires strict hidden-boundary arithmetic;
  preserve the static lockstep runtime as the exact comparison route.

## Moonshine gfx1151 exact wave8 top-1 — temporary rollback selector

- `wave8_top1` is now the G1 FP16 runtime default. It materializes the same
  complete FP16 logit plane, owns bounded FP16-value/int64-index scratch, and
  replaces only the full-logit argmax scan with stable block partials plus a
  final reduction. `MoonshineResidentRuntime(lm_head_route="wave8_argmax")`
  remains the explicit unfused rollback and is required for selective W8A16
  LM-head diagnostics.
- The clean production leaf passes at **-12.003% event / -12.005% wall**; the
  actual-checkpoint 194-step graph gate is byte-exact; and the regenerated real-
  audio gate passes all **24/24** six-file fallback/candidate eager/graph rows
  through EOS with pairwise-exact outcomes, zero timed allocation, four graphs,
  194 replays, and clean teardown. Keep the constructor/script selector through
  one clean release for rollback and bisection. Then remove the public selector
  unless a deployment still requires it; retain the separately registered
  wave8 projection and stable argmax as the mandatory composite fallback.

## Maple P1 expert-major prefill rollback — temporary opt-out

- `HIPENGINE_MAPLE_PREFILL_GROUPED_MOE` defaults to `1`; `=0` restores the
  exact row/route-gather gate/up/down chain. The grouped path reuses registered
  stable count/prefix/scatter metadata and writes outputs in original lane
  order, so no arithmetic boundary changes.
- Keep the opt-out through clean 128/320/512 recertification and one downstream
  optimization milestone because P1 misses its 2.826x expert-family target even
  though the current exact candidate is non-regressive. Remove the flag and
  runtime gather branch once the grouped path plus its eventual matrix/SIMD
  ternary consumer has passed that recertification; retain the gather kernels
  only as low-level correctness oracles if tests still need them.

## Maple M2 dual+swiglu fusion — opt-in, efficiency blocker

- `HIPENGINE_MAPLE_FUSE_MOE` default `0` (unfused is the fast path). `=1` uses
  `maple_moe_dual_swiglu_bf16` (fuses dual gate/up gemv + clamped SiLU), which is
  bit-exact and cuts decode launches 295→271 but is ~9% slower per MoE layer
  (interleaved micro-benchmark 229.7 vs 210.7 us). The launch savings offset the
  kernel regression in eager mode (step wall neutral), but in the optional M1
  c1 hipGraph path launch overhead is amortized so the fusion would regress
  ~1.4%. The M6 batch helper does not use the c1 graph.
- Blocker to promote: understand/eliminate the fused kernel's ~9% efficiency gap
  vs the unfused pair (likely swiglu `expf` tail in tid 0 and/or register/
  occupancy change). Once the fused kernel is <= the unfused pair, flip the
  default to 1 and remove the flag + unfused dual+swiglu branch.
- **Rejected path (do not re-promote as-is):** fused down GEMV + weighted
  residual in one kernel. It is bit-exact but regresses decode ~3.5% because
  grid=`out_features` serializes the eight selected experts per output row,
  losing the parallel-expert down grid and stride-8 weight reads. If revisited,
  use a parallel-expert grid with a two-pass/atomic weighted residual instead
  of serializing experts.

## Maple M2 qknorm+attention fusion — opt-in, ~1% regression

- `HIPENGINE_MAPLE_FUSE_QKATTN` default `0` (unfused is the fast path). `=1` uses
  `maple_attention_fused_qknorm_decode_bf16` (fuses per-layer
  `qknorm_rope_kv_write` + `attention_decode`). Bit-exact (attention output + K/V
  cache match bit-for-bit, GPU test) but ~1% slower per decode step (interleaved
  eager 6442 vs 6377 us): the in-group redundant K/V write (each of the 4 q-head
  blocks writes the same K/V) plus the folded 88 us qknorm work offset the single
  launch saved. Remove the flag + fused branch once the fused kernel's efficiency
  gap is eliminated (e.g. leader-only K/V write + cooperative grid sync).
- D0 now snapshots both fusion flags once at the `MapleRunner.step()` boundary,
  rather than rereading them in all 24 layers. The retained fresh-process A/B
  is **200.279 -> 202.580 tok/s (+1.15%)** with no kernel/launch change.
  Environment changes remain supported between calls, not during an in-flight
  token. Remove the two snapshots with the corresponding branches when these
  flags are retired.

## gfx1151 Q4 pack8 bulk-prefill WMMA rollback env

- `HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK=0` (default `1`) rolls the gfx1151
  bulk-prefill pack8 route back to the exact tile8x8 leaf without touching
  Laguna's `wmma_pack8` mode. Added with the D08-X parity campaign route
  (2026-08-15): pack8 WMMA measures 1.97x/2.33x tile8x8 on the Qwen3.5-0.8B
  dense-FFN pp512 shapes. The backend default now fails closed outside the five
  p512 pack8 shapes exercised by the complete-model campaign A/B. Remove the
  env branch once the parity campaign closes and no A/B bisection still needs
  the tile8x8 control; expand the shape policy only with matched evidence.

## gfx1151 Q4 operation-complete pack8 prefill rollback env

- `HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL=0` (default `1`) rolls the
  exact Qwen3.5-0.8B Q4_K_M p512 dense gate/up owner back to two registered
  singleton pack8 WMMAs plus standalone BF16 SiLU. Added with D08-X3
  (2026-08-15): the fused leaf is byte-exact and 2.089x faster; five paired
  complete-model blocks improve core/public prefill 13.81%/13.85% with neutral
  decode and Q8 guards. Remove the env branch after one non-regressive release
  window when no A/B bisection needs it. Do not broaden the model/quant/rows/
  K/N policy without separate complete-model evidence.

## gfx1151 Q8_0 GDN cluster8 route

- `GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE` now routes
  `("MOSTLY_Q8_0", 16, 16, 128, 128)` to `chain_peer_cluster8` (D08-X2-K2,
  2026-08-15). Rollback: `HIPENGINE_GGUF_GDN_PREFILL_MODE=
  chain_lds32_direct_nonvolatile`. Remove the rollback note after the parity
  campaign closes and no A/B still needs the exact route.

## gfx1151 dense-BF16 WMMA bulk prefill

- `HIPENGINE_GGUF_DENSE_WMMA_BULK=0` (default `1`) rolls gfx1151 dense-BF16
  bulk prefill back to the naive 32x8 scalar tile (D08-X2-K5, 2026-08-15).
  The backend default is limited to the two Qwen3.5-0.8B p512 shapes exercised
  by the complete-model campaign A/B; other shapes fail closed. Remove the env
  branch when the parity campaign closes and no A/B still needs the naive
  control; expand the shape policy only with matched evidence.

## Qwen3.6 shared-prefill SiLU/rotate rollback env

- `HIPENGINE_SHARED_PREFILL_SILU_ROTATE_FUSED=0` rolls packed-PARO shared
  expert prefill back to registered `silu_mul_separate_out_fp16` followed by
  `paro_rotate1_fp16`. The default exact pair-rotate route preserves the FP16
  activation rounding point, removes one launch and the intermediate store/load
  per layer, and measures **29.643 -> 21.617 us (1.371x)** at the actual
  512x512/group128/krot8 leaf on gfx1151. Bracketed c4/c8 complete runs remain
  within noise while prefill throughput is non-regressive. Remove this env
  branch after one non-regressive release window when no campaign bisection
  needs the two-launch control; keep the registered primitive fallback.

## fp16 recurrent state (HIPENGINE_GGUF_FP16_RECURRENT_STATE)

- Added 2026-08-19 for the gfx1151 Qwen3.8 Q4_K_S R2 topline: GDN recurrent
  state stored fp16 (fp32 accumulate, RNE half round-trip) via templated
  `state_t` kernels. The flag routes decode gate/segments, the compact
  peer-wave32 prefill writer, the chain prefill, and the fused
  `gdn_recurrent_rmsnorm_gate+cast` owner (the Qwen3.8 `ssm_out` quantizes as
  `gguf_q5_k_t16_v1`) to fp16-state instantiations; strict FP32 remains the
  registered fallback.
- Fail-closed guards raise when the flag is combined with an FP32-state writer
  that has no fp16 variant yet. As of 2026-08-20 the decode_order state-rows
  writers (verify-capture and segmented) are fp16-capable (see the
  20260820T065950 worklog entry and `kernel: fp16-state decode_order state-rows
  prefill writers`), so their fail-closed guard in
  `_run_linear_attention_prefill_layer_rows` was removed and the dispatch
  selects the fp16-state variant under the flag. The in-place segmented
  decode_order writer remains fp16-capable as a generic fallback, but the
  Q4_K_S production packed-AR path now uses the registered compact-peer
  segmented sibling: each indexed slot stays FP32-register-resident across its
  packed token segment and converts only at a chunk boundary. This repairs the
  failed c4/c8 prefill top-1 scopes and removes the per-token global-state
  fallback from the measured route; the repaired 1,170-row hard gate passes.
  The packed AR prefill/decode guards that required
  `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN` remain removed. Remaining
  fail-closed guards:
  producer-folded chain-journal output/snapshot fusion under FP16 state, any
  MTP2 request without the complete non-fallback production manifest, and any
  `HIPENGINE_GGUF_GDN_PREFILL_MODE` other than the auto
  `chain_compact_peer_wave32` Q4_K_S route. P8 keeps consumer-owned dtype-sized
  rollback snapshots and the strict unfused cast instead of enabling the
  independently excluded gfx1151 producer-folded aliases.
- Removal triggers: (1) after the c8 fp16-vs-fp32 benchmark and the
  strict-teacher gate validate the production route, decide whether the fp16
  state becomes the default and the flag collapses to a
  remove/rollback seam; (2) complete the SPECDEC2 P8 numerical/task/lifecycle/
  economics packet for the registered FP16 chain-row manifest, then decide
  whether the remaining legacy/env MTP guards can collapse; (3) drop
  `gdn_effective_mode`/frozen-mode assumptions if
  the flag is later allowed beyond the compact-peer-wave32 route.
- Checkpoint 2026-08-20 (audited): the 450-row c1 numerical capture passes at
  global mean/p95/p99/max KL `5.28e-5/2.80e-4/5.81e-4/1.43e-3` and 99.78%
  top-1. The repaired packed c4/c8 gate passes every hard category/shape/
  transition threshold over 1,170 rows at `1.02e-4/3.19e-4/1.05e-3/3.2866e-2`
  and 99.829% top-1, with exact repeats, sparse/neighbor/permutation isolation,
  and indexed decode ownership. One long-c8 row passes manual review (same
  wide-margin top-1 and improved teacher NLL). See
  `benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-production-gate.json`
  and `benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-packed-production-gate.json`.
- The clean FP32-A/FP16/FP32-B p512 bracket retains FP16 as non-regressive:
  aggregate decode improves c1/c4/c8 `12.9005/42.5856/57.2905 ->
  13.0204/43.8535/59.0106 tok/s` (`+0.93%/+2.98%/+3.00%`); aggregate prefill
  improves `+0.47%/+1.65%/+3.31%`; tracked peak saves 1.15 GiB. See
  `benchmarks/results/2026-08-20-gfx1151-qwen38-27b-r2-fp16-state-repaired-production.json`.
- Decision correction: the fixed 3% materiality threshold is removed. FP16 is
  now the backend/file-type scoped default for gfx1151 `mostly_q4_k_s`;
  `HIPENGINE_GGUF_FP16_RECURRENT_STATE=0` is the strict-storage rollback. The
  clean predeclared serving screen is non-regressive: c1 exact goodput improves
  `8.2121 -> 8.2434 tok/s`, exact c8 server throughput improves
  `13.0999 -> 13.2738 tok/s` (**+1.33%**), and c8 ITL-p99 improves
  `0.8532 -> 0.8287 s`. Both modes share the absolute 0.5-second SLO failure;
  that serving-path issue does not erase the candidate win. See `benchmarks/results/2026-08-20-gfx1151-qwen38-27b-fp16-state-serving-screen-rejected.json`.
  SPECDEC2 P8 retains the runtime-resolved production manifest, exact FP32
  fallback, and complete C1/C2/C4 numerical/lifecycle/economics packet. Native
  target graph FP16 state diverges after layer 51 for accepted rows; exact eager
  target verify/selected commit is therefore mandatory until a separately
  qualified graph fix lands. P9 promotes no automatic cell because the positive
  capacity-1 C1 premise does not engage on the normal capacity-4 server owner;
  automatic stays K0 and producer-folded chain snapshot/fusion remains excluded.
- 2026-08-27 Q4_K_M follow-up confirms the blocker under the production bar:
  capacity-matched c1/B3 is 14.287 vs 9.730 tok/s (+46.84%) with the 1,170-row
  KL/top-1/determinism/isolation gate passed, but c1 on capacity>1 remains K0.
  HTTP plan `realized_group_rows` is prompt multiplicity inside one request,
  not the eventual group of concurrent children; resident physical streaming,
  refill, and survivor ownership remain unqualified. An oracle-only FP16 device
  proposal reached the eager target then failed precommit with `FP16 recurrent
  state device proposal requires eager selected commit`, recovered 10/10 to AR,
  and regressed 6.77%; its diagnostic source was removed. Removal trigger:
  qualify resident realized-group c1-only admission on capacity>1 plus dynamic
  refill/survivor/load/SLO, and either implement+production-qualify FP16 eager
  selected commit or repair the FP16 native graph. Until then automatic K0 and
  eager host proposal/target are required. Evidence:
  `benchmarks/results/2026-08-27-gfx1151-qwen38-concurrency2-t04-production-suite.json`,
  `...-t11-t13-ownership.json`, and `...-t23-t24-policy-scheduler.json`.

## gfx1151 c1 short-batch attention thread-geometry (GGUF_SHORT_C1_BATCH_ATTN_THREADS / env)

- Added 2026-08-20. The fixed256 context-batch body gained a runtime block-width
  entrypoint (`...fixed256_threads_spans`); gfx1151 promotes `threads=1024`
  (T2 non-exact, execution-profile gate-passed) via the capability, with the
  exact 256-thread leaf kept as the registered strict fallback.
  `HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS` forces any width for A/B/bisection.
- Removal trigger: after split-K3 / split-policy or WMMA-tile variants on the c1
  exact spine replace this geometry (or a fresh gfx1151 route/profile check
  overturns the 1024 win), collapse the runtime-threads entrypoint back to a
  fixed constexpr and drop the capability/env unless a named validation gap
  still needs the override.

## Dense long split-K multi-row eager verifier row-wise strict dispatch

- Retained 2026-08-21 as the RF1 strict eager/oracle fallback in
  `Qwen35GGUFFullStackRunner._run_native_attention_bulk_ffn_layer_rows`
  (`strict_long_rows`). When a dense (non-MoE) multi-row verifier batch is in
  long split-K territory (`_use_gguf_full_attention_split_decode(start_position +
  rows)`), each attention and FFN row routes through the registered c1/strict
  dispatch while keeping the native split-K attention leaf, so selected
  Conv/GDN state and BF16 K/V stay byte-exact. Retained short-context batched
  staging is unchanged. Rationale: a real B3 full accept ending at 1024 crossed
  a BF16 rounding boundary in the staged multi-row linear-attention/FFN output
  versus scalar AR (layer 46 row 3, max abs 0.015625). The retained 13/13
  transition packet and real crossing are in
  `worklog/entries/20260821T052947.067917Z-gfx1151-mtp-rf1-boundary-4k-46c738.md`.
- Measured 2026-09-12: the flip is exactly at `start_position + rows == 1024`.
  `scripts/ud_mtp_ar_verify_numerics_gate.py` reports `max_abs_diff` 0.073-0.280
  at 1021/1022/1023 and exactly `0.0` at 1024, 1536, 2048 and 4096, because the
  per-row branch runs the same per-token kernels as `session.step`. **A
  long-context run of that gate therefore reports zero KL for a reason that has
  nothing to do with the multi-row arithmetic**: at 4096 it pooled 162 rows with
  mean = p95 = p99 = max = `0.000e+00` and top-1 1.0000. Do not read a
  long-context zero-KL result from this gate as a multi-row verifier pass, and do
  not read it as a speed claim either — the retained per-row route is 44.7 s per
  8 generated tokens and the long-graph alternative measured 0.9989x (see the RF2
  entry above).
- This is deliberately not the fast long-context route (direct cycle cost
  0.4–1.6 s, 44.7 s per 8 generated tokens) and does not raise the 1023 graph
  context cap.
- The staged route is now the default for the measured artifact (2026-09-19):
  the backend package capability
  `GGUF_STAGED_LINEAR_ROWS_LONG_DEFAULT_FILE_TYPES` (`mostly_q4_k_m` on
  gfx1151) admits dense linear-attention rows above the split threshold, and
  `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG` overrides it in either direction as
  the rollback seam. The row-wise route in this entry stays registered as the
  strict fallback for a preset-bound or unknown-manifest artifact, an unlisted
  file type, gfx1100, and `=0`. The 2026-09-12 BF16 boundary that motivated the
  blanket rule (layer 46, max abs 0.015625) does not reproduce on the current
  dense Q4_K_M path: on the eager route
  (`scripts/gguf_mtp_long_context_gate.py --disable-target-graph`, gfx1151,
  Qwen3.8-27B Q4_K_M) every straddle case at 1,020-1,028, 3,528-3,532 and 8,192
  is exact against the serial-exact teacher with `staged_chain_calls` 48 and
  `row_wise_attn_calls` 0, the matched row-wise control is exact as well, and
  the committed long-context task packet binds task, AR-id and eager-ownership
  equality at a 4,096-token context.
- Removal trigger: the gates have now qualified the staged linear route on the
  admitted (backend, model, quant) cell, the capability default is the promoted
  path, and `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG` is a documented rollback
  exactly like `GGUF_AOTRITON_HEAD_MAJOR_KV`. Retire the env override and the
  row-wise long-context branch together when the row-wise c1 route stops being
  the registered strict fallback for an unlisted artifact - that is, when every
  admitted artifact family has its own packet and the strict chain is reachable
  through the registry rather than through this branch. A gate failure on a
  newly admitted family means removing that family from the capability, not
  restoring a default-off flag.
- Removal trigger: after RF2 context-bucketed split-K N1/N2 graphs are qualified
  and the eager long path becomes a non-authoritative fallback, re-evaluate
  whether the row-wise serialization can be consolidated into the registered
  strict chain. The strict eager/unfused fallback itself must remain per
  `EXECUTION-PROFILES.md` invariant #8; only the duplicated per-row staging in
  the eager verifier may be collapsed once RF2 changes ownership semantics.

## gfx1151 native full-attention prefill dynamic LDS overflows at large context

- Found 2026-08-22 during RF1 long-context qualification:
  `launch_qwen35_paged_full_attn_prefill_gqa_gate_bf16` sizes dynamic LDS as
  `(max_context_len + threads + head_dim) * 4`, which exceeds the 64 KiB
  workgroup limit once `max_context_len ~= 16K` and the launch fails with
  `HIP error 1 (invalid argument)`. It is reachable only when the AOTriton
  prefill default is disabled (`HIPENGINE_GGUF_AOTRITON_PREFILL_ENABLE=0`) at
  large context; gfx1151 production uses AOTriton prefill by default and native
  is only for sub-512, so no RF1 matrix row hit it.
- Fix direction: bound or bucket the dynamic LDS by an actual workgroup-budgeted
  tile (or fall back to the AOTriton/registered strict prefill at long context)
  instead of sizing by whole context.
- Removal trigger: once the native long prefill route is either capped to a
  budget-safe tile or superseded, keep the guard that selects AOTriton/strict at
  large context and drop any unbounded dynamic-LDS path.

## RF2 long-context target graphs are exact but not auto-routable

- Added 2026-08-22: `gguf_native_spec_cycle.py` retains a bounded eight-entry
  cache of exact target graphs keyed by candidate budget, N1/N2 ownership, and
  attention schedule/context limit. It is useful for explicit graph diagnostics
  and future submission optimization; transition and kernel-family boundaries
  fail closed to eager.
- Current blocker: same-protocol 4079-token/eight-output wall is graph 97.206 s
  versus eager 97.101 s (0.9989x), while true AR is 69.635 s (0.7164x graph
  MTP/AR). No long graph bucket is admitted to automatic routing.
- Removal/promotion trigger: remove the long graph cache if it remains without a
  measured exact sub-window or complete-wall benefit after the next targeted
  submission campaign. Promote a bucket only after a predeclared same-protocol
  experiment improves complete wall and passes its full context/lifecycle/SLO
  packet; never use exactness alone as a speed claim.

## RF1 1024 target/proposal sentinel blocker

- Reconfirmed during the 2026-08-23 Concurrency2 merge against pristine
  `origin/main@f5cb6ffe6`: the real dense Q4_K_M MTP crossing test can publish
  the int32 sentinel `2147483647` at the 1024 target/proposal boundary. This is
  pre-existing in upstream RF1, not introduced by C2. Running the node also
  poisons the following physical GPU test, so the crossing node is skipped with
  automatic long-context MTP remaining fail-closed.
- The short-context device-proposal readiness path now rejects the final graph-
  bucket handoff row before launching proposal work
  (`target_graph_proposal_handoff_boundary_miss`). Ordinary host-proposal target
  graphs, eager fallback, and qualified short cycles remain available.
- Removal trigger: repair and independently qualify cross-stream proposal-result
  publication at the 1024 boundary, rerun the skipped crossing test in a fresh
  process, then the complete MTP E2E/long-context/lifecycle/API/load bundle.

## Compact DMS device payloads opt-in (C2-7 U6)

- `DMSCompactBackend(..., device_payloads=True)` or
  `HIPENGINE_DMS_DEVICE_PAYLOADS=1` moves the BF16 K/V payload into the
  device slot buffers owned by `DMSDevicePayloadStore`
  (`hipengine/kvcache/dms_device.py`); the host keeps only the
  O(rows x heads x capacity) extent metadata. Default is off: the host
  parent remains the production path until the C2-7 product gate passes
  (trained `dms_metadata.json`, rocprof kernel-identity rows on a stable
  GPU, c1-c32 lifecycle rerun). Remove the flag (device mode becomes the
  only BF16 path) after that gate, keeping the host parent as the
  registered `cpu_reference` fallback for no-HIP hosts.
- `HIPENGINE_DMS_DEVICE_TRIPWIRE=1` (default off) reads back the
  append-kernel overflow status after every device append. The host
  pre-check already raises `MemoryError` before any device mutation, so
  the tripwire is a state-drift canary for soak/lifecycle runs, not a
  correctness gate. Revisit the per-layer-per-step D2H readback in the
  optimization pass (e.g. fold the status into the next step's metadata
  staging) before enabling it by default.

## Qwen3.8-27B gfx1100 packed route: T2 production drift, not a blocker (corrected 2026-08-19)

- The current C2 packed decode route is NOT byte-exact vs the c1 oracle for
  Qwen3.8-27B Q4_K_M on gfx1100: prompt row 2 (last token 9708) diverges at
  decode token 73 (observed 6866 vs oracle 3154), deterministic at c2 and c4
  (the two packed widths are byte-identical to each other). The pre-C2 tree
  (b08ed12d6) was byte-exact on the same host/protocol.
- **Corrected classification (supersedes the earlier "regression/blocker"
  label):** this is a **T2 production drift** (width-specific packed
  arithmetic vs c1), which `docs/EXECUTION-PROFILES.md` explicitly allows for
  the `production` profile ("width- and shape-specific arithmetic is
  allowed"; "cross-width generated-ID equality is diagnostic, not a
  promotion requirement"). The produced text is fluent and valid ("giving
  back to society" vs "making a positive impact on society"), the packed
  trajectory is the oracle shifted by exactly 2 tokens after the single
  step-73 flip, and both end in the same repeat loop. It is **not** a
  correctness bug and **not** a production gate failure; the F1 harness
  "mismatch" is a composition-invariance detector, not a correctness oracle.
- It becomes a requirement only if this route is ever declared `strict`
  (bit-stable on retained fixtures) or `batch_invariant` (result preserved
  across widths/slots/admission order). If that is needed, restore
  cross-width parity (bisect af5d00098 / 29a786afc) or keep the route
  `production`. Optional: measure the step-73 logit margin (3154 vs 6866)
  in a high-precision reference to confirm the narrow-margin flip. Evidence:
  `benchmarks/results/2026-08-19-concurrency2-qwen38-27b-oldproto-c1-c4-old-vs-new-diagnostic.json`,
  worklog `qwen38-27b-old-vs-new-ab` and
  `qwen38-27b-packed-drift-correction`.
- Separate pre-existing performance fact (not C2-specific, the item to
  profile): the dense 27B packed c2/c4 step costs ~5x the c1 step (ITL 36ms
  -> 184/198ms) in BOTH trees, so neither design scales this model's
  aggregate throughput; rocprof kernel-family audit of the packed step is
  the follow-up.

## RF-OI2 — default-off adaptive MTP budget diagnostic (2026-08-25)

- **State:** rejected for promotion; retained only as explicit diagnostic and
  transition-contract infrastructure.
- **Paths:** `hipengine/speculative/mtp_budget.py`, optional `budget_policy` in
  `runtime/qwen35_gguf_mtp.py`, and `--budget-sequence` / `--adaptive-budget`
  in `scripts/qwen36_dense_gguf_suite.py`.
- **Why retained:** the request-owned sequence policy is the first real gfx1151
  proof of all nine B1/B2/B3 transitions over independent proposal/target graph
  buckets, including reject/partial/full outcomes and clean teardown. The EMA
  controller supplies reproducible negative evidence without prompt/token
  branches, but loses to fixed B3 on aggregate/train/category economics.
- **Removal trigger:** before exposing adaptive depth in public generation,
  either (a) replace the rejected EMA controller with a deterministic policy
  that beats the best fixed budget on full train and every heldout/category
  gate, then remove the old controller/config, or (b) if no successor campaign
  consumes variable-budget transitions, move the sequence owner into test
  support and remove the runtime policy hook and benchmark CLI flags. Never
  remove the fixed B1/B2/B3 graph buckets or strict transaction fallbacks.

## RF-OI4 — default-off post-norm draft-hidden diagnostic (2026-08-25)

- **State:** immutable policy infrastructure retained; `post_output_norm`
  rejected for Qwen3.8 promotion, `pre_output_norm` remains strict/default.
- **Paths:** `Qwen35GGUFDraftHiddenPolicy` and the explicit
  `draft_hidden_variant` decoder argument in `runtime/qwen35_gguf_mtp.py`, plus
  `--draft-hidden-variant` in `scripts/qwen36_dense_gguf_suite.py`.
- **Why retained:** it is the request-owned single source of truth for prompt
  priming and steady target proposals, preserves strict target commit/rollback
  ownership, prevents accidental double normalization, and enables future model
  artifacts to test their declared convention without ad-hoc branches.
- **Removal trigger:** if no second model/conversion campaign consumes the
  manifest, move post-norm transformation into benchmark/test support and remove
  the runtime candidate path. If a future model promotes post-norm, bind the
  variant to that model/provider manifest and rerun full category, heldout,
  long/lifecycle, and same-host economics gates; never select by prompt or live
  acceptance.

## PARO MTP verifier numerical evaluator migration (2026-08-24)

- `scripts/mtp_paro_verifier_numerics.py` temporarily retains the original
  simultaneous two-session `run()` beside the sequential `run_sequential()`
  selected by the CLI. The simultaneous owner needs about 85% W7900 VRAM with
  the proposer and timed out during setup; it remains only for bisection while
  the sequential strict-capture/fast-replay evaluator is qualified.
- **Ready:** `run_sequential()` now passes the complete 10-prompt/three-repeat
  D64 gate plus focused strict schedule/commit and state-owner regressions.
  Remove the old simultaneous `run()` in the next cleanup unit; keep only the
  sequential evaluator.

## SPECDEC2 post-reject cooldown experiment (2026-08-29)

- `HIPENGINE_SPECDEC2_POST_REJECT_COOLDOWN` (default off) forces a
  K0-transitional cycle after any request whose last physical cycle accepted
  zero candidates, routing the provider repair through the target-hidden
  catchup instead of immediate re-speculation. Wiring:
  `plan_speculative_requests(suppress_speculation=...)`, engine-loop probe,
  runner hook `speculative_post_reject_cooldown`, adapter telemetry read.
- **Settled 2026-08-29:** with the packed-decode hidden fix (`6b580ad25`) the
  policy nets 1.1700x vs the 1.1890x default at D24 (code benefits, en/mixed
  pay more for the lost speculation cycles). Not retained as default.
  **Ready:** remove the flag, hook, and planner parameter whenever the
  campaign stops considering longer-D cooldown economics; the measured
  evidence lives in worklog entries 20260829T052637/074350.

## gfx1100 grouped multi-request prefill is unreachable by construction (2026-08-30)

`ResidentBatchScheduler.next_prefill_batch_work` ("emit one stable multi-request
prefill quantum") exists and `engine_loop` will select it only when
`runner.packed_prefill_max_rows > 1`. On gfx1100 that attribute comes from
`backend_package_capability(backend, "GGUF_C2_PACKED_PREFILL_MAX_ROWS", 1)` and
**no backend package declares it**, so the branch is dead today.

Declaring the capability would still not batch a realistic wave, because
`_try_prefill_native_work_batch` (`hipengine/generation/qwen35_gguf.py:6401`)
additionally requires: every selected chunk to be the request's **entire prompt**
(`chunk == row.prompt_ids`, first chunk only, no prefix reuse, no decode slot) and
**all chunk lengths equal** (`len({len(chunk)}) == 1`). It therefore implements
only the same-length "common boundary" case, and the standard suite's prompt
lengths (35, 36, 36, 39, 39, 43, 46, 48, 60, 67 tokens) contain no 8-wide
equal-length group. Lowering `HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` cannot help:
a partial chunk fails `chunk == row.prompt_ids`.

Measurable cost of the dead path: hipEngine admission is flat at ~305 ms/lane and
grows 304 ms per added lane, while llama.cpp's falls 223 -> 106 ms/lane
(1.41x behind at C1, 2.88x at C8). Shipping-route prefill measures
167.6 tok/s at 45 rows versus 877.0 tok/s at 512 rows, so a grouped 360-row wave
is worth roughly 2441 -> ~855 ms of C8 admission (predicted 69.97 tok/s versus
the current 58.74).

Removal/enabling trigger: either (a) extend the grouped native prefill to accept
unequal-length rows via padding plus attention/KV write masking and partial
chunks, qualify it against the strict unfused per-request chain, and then declare
`GGUF_C2_PACKED_PREFILL_MAX_ROWS` for gfx1100; or (b) if padded grouping is not
pursued, delete `next_prefill_batch_work` plus its `engine_loop` branch so the
dead route stops advertising a capability the backend does not have. Measured
evidence: `benchmarks/results/2026-08-30-w7900-q4km-c1c8-hipengine-refresh-post-promotions.json`.

Correction while the section above is still fresh: the equal-length rule was a
policy of the **resident AR route only**, not a runner limit. The serving route
reaches the very same `prefill_batch_native` entry point with arbitrary prompt
lengths today (`hipengine/generation/qwen35_gguf.py:3025`, chunked only by
`_MTP_SERVING_TARGET_BATCH_MAX_SLOTS`), and the AR call site already forwarded
`full_prompt_lengths` per row. So no padded/masked kernel work is implied.

**Landed:** that equal-chunk-length early-return has been removed from
`_try_prefill_native_work_batch`, so the AR route groups mixed-length waves like
the serving route, and a wave no longer loses grouping because one lane is over the
chunk cap (the compatible subset groups; the odd lane stays on the chunked serial
path). Two tests pin the contract
(`test_resident_ar_packed_prefill_groups_mixed_prompt_lengths` for the AR route via
`__new__`-stubbed rows, and
`test_gguf_ar_packed_prefill_batches_unequal_prompt_lengths` for the serving route).

**Promoted:** `GGUF_C2_PACKED_PREFILL_MAX_ROWS = 8` is now declared for gfx1100, so
`engine_loop` reaches the batch route by default. The old tripwire
(`test_gguf_gfx1100_packed_prefill_capability_is_undeclared_by_default`) was inverted
into `test_gguf_gfx1100_packed_prefill_capability_is_declared`, which keeps the
protective half (the lookup still defaults to 1, so an unqualified package stays
lane-by-lane). Qualifying evidence: canonical-suite packet with byte-identical
protocol, 432 cross-packet per-row id comparisons and 0 mismatches, 80/80
correctness cells, width 1 unchanged, C8 AR 45.68 -> 78.67 tok/s, and acceptance
flattened to 0.789 at every width. Remove the declaration (back to 1) only if a
long-prompt or multi-session workload shows a regression the canonical suite cannot
see - the remaining known limit is `chunk == row.prompt_ids`, so prompts longer than
the chunk cap still prefill serially and gain nothing.

`worklog.py new` can mint an entry that `worklog.py check` then rejects. `new`
derives the filename from `slugify(topic)`, but `slugify` truncates at ~48
characters and can leave a trailing `-`, while `check` (a) requires
`slugify(topic) == topic` and (b) requires the filename to start with the
verbatim `topic`. Note also that `new` writes the **title** text into the `## Topic`
body, so a title containing `>=` or `;` fails `check` even when `--topic` is a clean
slug - keep both flat. A long or bracketed title therefore produces a filename whose
embedded slug can never validate - `20260830T025142...guard--0609f8.md` needed a
manual topic shortening plus `git mv` inside an amend of an unpushed commit, and
`check` blocks every later commit until that is done. Fix in `new`: after
slugifying, strip trailing/Repeated `-`, hard-truncate to the round-trip length,
and assert `slugify(topic) == topic` and the filename prefix before writing the
file. Remove this entry when `new` refuses to emit an invalid pair.

Wave-grouped AR prefill also needs **compatibility-aware selection**, not just the
equal-length relaxation landed today. `next_prefill_batch_work`
(`hipengine/generation/batch_scheduler.py:1249`) advances every selected prompt
cursor *before* publication and states that incompatible rows "must fall back before
this method is selected" - but nothing enforces that upstream: `_row`-level
compatibility is only tested afterwards inside
`_try_prefill_native_work_batch`, whose per-row loop returns `False` for the **whole**
work item. Two consequences to remove when the capability is declared:

* With `DEFAULT_MAX_PREFILL_CHUNK_TOKENS = 256`, any prompt longer than 256 tokens
  yields `chunk != row.prompt_ids`, so a single long lane refuses every other lane in
  that item and the wave prefills serially anyway. The standard 10-prompt suite
  (35-67 tokens) is unaffected, so a measurement on it cannot see this.
* The refusal happens *after* the scheduler has already consumed the prefill cursors,
  so the fallback runs on post-advance state instead of never having committed.

**Landed (inert by default):** `_try_prefill_native_work_batch` now partitions
instead of refusing - it groups the largest compatible subset and returns the handled
request ids, and `prefill_batch` skips those ids and serially prefills the rest, so an
over-long lane goes to the chunked path without taxing its neighbours. The requested
fixture exists and is green
(`test_resident_ar_packed_prefill_survives_one_over_long_lane`: one 300-token lane plus
seven short lanes asserts seven grouped rows, one call, and `prefill_tokens_seen == 0`
for the long lane).

**Still open:** the partition happens *after* `next_prefill_batch_work` has already
advanced every selected cursor, so the scheduler still over-commits on rows it knows
cannot group - move the compatibility test to selection time, and declare
`GGUF_C2_PACKED_PREFILL_MAX_ROWS` once a mixed wave with a >256-token lane is measured
end to end (the standard 35-67-token suite cannot exercise it).

`resident_observability.routes.counts` in `gguf_mtp_c1c8_server_bench.py` packets are
**process-global cumulative snapshots**, not per-cell deltas - AR and K3 run in the same
process, so a cell's counts include the other arm's work. They *are* attributable, but
only with the right method: the packet's cell **list order** is execution order (every
delta comes out non-negative), and inside a cell the `order` field says which arm's
snapshot was taken first, so differencing cell-to-cell and arm-to-arm splits the work
exactly. Do not use `order` as a cell sequence - that is what made an earlier attempt
collapse every delta onto one arm, and it produced a now-corrected claim that MTP never
reaches the packed route. Preferred fix still stands: have the bench snapshot counters
before and after each cell and store the delta next to the cumulative value, so
attribution does not depend on reconstructing execution order.

`gguf_mtp_verifier_rocprof.py --mode block-verify` cannot sweep `--block-rows`: the child
raises `ValueError: token_id 2147483647 outside [0, 248320)` from
`qwen35_gguf_runner.py:19000`, i.e. `verify()` rejects the INT32_MAX padding the harness
submits for rows past the prompt tail. That sweep - rows 4 -> 32, since K3 is 4 rows per lane
and C8 is 32 - is the only cheap way to separate launch overhead from real kernel work in the
MTP verify path, and it is currently impossible, which is why the per-row verify cost stayed
open on 2026-08-30. Fix: have the harness submit a real prompt long enough to fill the block
(`--prompt-ids` sized to `block_rows + prompt`), or give `verify()` an explicit padding token
it accepts. Also consider a `--roctx-sdk auto` that degrades to plain kernel tracing when no
pip ROCm SDK exists - see the profiling traps section in `docs/RDNA3-TUNING-GUIDE.md`.

## gfx1100 prefill route booleans are duplicated and inert-by-pooling (2026-08-30)

### Resolved 2026-08-30: the branch is live, this section's premise is superseded

The claim above ("no backend package declares it, so the branch is dead today") is no longer true.
`hipengine/kernels/hip_gfx1100/__init__.py:593` now declares `GGUF_C2_PACKED_PREFILL_MAX_ROWS = 8`
and exports it at :959, shipped by `72a29e9ad` ("perf: ship grouped prefill on gfx1100, C8 AR
45.68 -> 78.67"). The equal-length requirement described above is what that work replaced: the
route now groups the compatible subset instead of refusing the whole wave, and the effect is
observable, not inferred - `native_full_prefill_groups` is exactly 0 at C1 and 39-56 per prompt at
C4 / 151-176 at C8 (min-max over 10 prompts), in
`benchmarks/results/2026-08-30-w7900-q4km-c1c8-grouped-prefill-promotion.json`. What still stands
from this section: the mixed-length case is handled by subsetting, so a wave of all-different
lengths still yields many one-row groups - 6-9% of decode steps at each width run below the logical
width - and that residue is what still has no owner.

`hipengine/generation/qwen35_gguf.py:214-215` defines `_GGUF_PUBLIC_USE_WMMA_PREFILL = True` and
`_GGUF_PUBLIC_USE_GEMV_DECODE = True` as defaults for `_resident_session_scope`, but 11 call sites
pass `use_wmma_prefill=True` literally, including the warmup pools that create the sessions the AR
route later reuses. Measured consequence: flipping the public default changes nothing observable -
two probe arms gave 0.2950 s vs 0.2956 s at 42 tokens and 0.7115 s vs 0.7121 s at 285 - because the
measured session is the pooled one, constructed from a literal `True`. Two follow-ups: delete the
booleans and keep one route owner, or route every call site through the single default. Until then,
any experiment that toggles prefill route by the public default is silently vacuous, which is how
this nearly got reported as "GEMV prefill is neutral".

Measured 2026-08-30, the GEMV decode half is worse than pooling-inert: it is inert **at the
consumer**. `qwen35_gguf_runner.py:13292` resolves the preference into
`Qwen35GGUFFastPathSafety(requested_gemv_decode=...)` and its only two consumers, `disabled_wmma` and
`disabled_gemv`, are gated on `is_qwen35moe`, so on the dense GGUF path the value moves one field of a
safety record. Flipping it at the resolver - the only place a flip can land, since precedence is
explicit kwarg > session > env - changed C1/C2 by 0.4%/0.7% (22.170 vs 22.250, 31.087 vs 31.290
tok/s), with the resolver logging 3 calls per run, all incoming=True resolved=False. Delete the 11
literal `use_gemv_decode=True` sites on the dense path, or make the MoE safety check query a real
capability instead of a caller-supplied preference. Instrument notes for the next person: this flip is
not reachable by environment variable, and patching `hipengine/runtime/gguf_linear.py` alone does
nothing because `qwen35_gguf_runner.py` imports `gguf_gemv_decode_enabled` into its own namespace.


### Pending 2026-08-30: the packed prefill row is an A/B question, not a missing capability

The 2026-08-29 grouped-prefill promotion left the Q4_K "prefill" column as `—` with "this row needs a
separate full-engine measurement". `scripts/gguf_packed_grouping_probe.py --kind prefill --widths 1..8`
now shows the grouped path is reachable at every width (e.g. 8.00x at rows=2, 1.97x at rows=97), so the
gate is "does a win in two of six projections move the engine", not "can grouped prefill run".

> **2026-08-30 correction: do not treat the probe as evidence about the engine at all.**
> `resolve_gguf_linear_dispatch(weight, rows)` ignores every route preference - env and session session
> flags alike - and picks purely by quant layout and row count: dense T16 uses `gemv_rowtile_n16` at
> rows==1 and `t16_wmma_prefill_bf16_bf16_out` for every rows>1 (proven by
> `tests/test_unit_gguf_gemv_decode_dispatch.py::test_gguf_row_decode_preferences_do_not_override_the_t16_rowtile`,
> which passes for all three env combinations). The Q4_K_T16 1536x1536 rows=1 probe arm therefore ran
> `gemv_rowtile` on both sides: `--route packed-grouped` only swaps qkv/gate_up, and `gemv-prefill` only
> differs at rows==1. A "full-GEMV prefill" arm is unreachable through the resolver; it needs the session
> route below.

The A/B that is worth the expensive run: same process, same model, same schedule, `--route exact` /
`--route packed-grouped` / `--route gemv-prefill`, at each engine width 1/2/4/8, plus the default-packed
engine control, and the correctness gate on every arm. Widths matter independently - 23% at rows=2 may
not carry an e2e win while 72% at rows=129 probably does - so record per width, and do not promote the
row without a measured engine delta. The probe supports both missing arms (`--route gemv-prefill` now
sweats rows>1 explicitly; `--route packed-grouped` covers rows>1 at every width).

> **2026-08-30: the engine-side A/B was then run, and it did not move.** Packed-grouped QKV+gate_up at
> the resident AR route costs 63ms (17.70ms -> 18.07ms median, +0.37ms, over 58 cells / 194 tokens and
> 264 paired launches) - see
> `benchmarks/results/w7900-gguf-q4ks-ar-packed-grouping-e2e-default-route-2026-08-30.json`. The row stays
> `—`: 3 of 6 projections regressed (down_proj 0.62x at rows=2) and prefill is a small share of AR wall at
> these shapes.

Remove when: the packed row is published, or the A/B says the branch is not needed.

### Pending 2026-08-30: the all-GEMV prefill arm - knob landed, width-1 arm answered

The 2026-08-20 "gemv-prefill-blocked" row rejected a *global* `use_wmma_prefill=False` at -8.4%: it
changed decode too. A prefill-only arm needs no new kernel - Q4_K GEMV is already registered for
rows>1 (`gguf_gemm.py:429 full_gemv`), and the rowtile is already a flag-free route.

The gap was a control-surface gap, and it was worse than "a knob is missing":
`launch_gguf_linear:2447` resolves `use_wmma_prefill` through `_resolve_use_wmma_prefill`, which prefers
an **active session flag over the environment** (`gguf_linear.py:1469`), and every resident session
construction passed `use_wmma_prefill=True` unconditionally. A bench could not reach the all-GEMV arm by
any env variable.

> **Landed at 802a7e32b + 356be6858, and the first attempt was inert.** `HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL`
> is validated (`0`/`1` only, empty rejected - an empty value would otherwise look like "env unset" and
> fall back to the session's True). The first wiring only touched the two sites REFACTOR named, and those
> are inside `prepare_request_scratch`: the resident AR route pins True in `_open_ar_serving_slots` (pool
> `ar_batch`) and `_reserve_sessions` (pool `continuous_ar_dynamic_kv`), plus four `mtp_target`
> acquire/construct sites. Ten sites now resolve through one helper; the single remaining literal is
> `_generate_dense_speculative_mtp_detailed` (dense speculative MTP, deliberately out of scope, asserted
> by name). Lesson for any future "make a route reachable" claim: name the function the server reaches,
> and prove it with a 2-minute launch-count check before scheduling a 12-minute sweep.
>
> **Width-1 result (measured on host `epyc`, AMD Radeon Pro W7900 (gfx1100) GPU 0, C=1, 64
> AR-only packets, packet-window trace):**
> all-GEMV prefill moves exactly 2464 launches (`wmma_prefill` 7392 -> 4928, `local32_gemv`
> 136224 -> 138688), halves prefill-class kernel wall 9279.8ms -> 4465.1ms and drops `other` wall 29% as
> the Q4_K repacks disappear - and AR median gets **7.0% worse** (17.68ms -> 18.91ms, mean
> 17.799 -> 18.933). Independent reproduction of the 2026-08-20 -8.4%, with sub-window and end-to-end now
> agreeing. Artifact:
> `benchmarks/results/w7900-gguf-q4ks-ar-wmma-prefill-session-knob-width1-2026-08-30.json`.

Remove when: the all-GEMV prefill arm is measured at widths >= 2, or the arm is dropped. If it is
dropped, delete the "prefill" column from the packed-grouping probe rather than leaving a column nobody
can fill.

### Pending 2026-08-30: `packed e2e` stays default when the win is concentrated in two projections

Same discipline as the rowtile 50%-share rule below: a 3.9x sub-window win that lands on qkv+gate_up
while down_proj regresses (0.62x at rows=2) is not a promotion. No env flag was added for the grouped
AR route, so there is nothing to remove - this line exists so the next agent does not re-add a
`HIPENGINE_GGUF_*PACKED*` default-on flag from the 2026-08-29 grouped-prefill evidence without the
engine-side A/B.

### `_default_roctx_sdk` is duplicated across the profiler wrappers (2026-08-30)

`gguf_packed_ar_rocprof.py`, `gguf_mtp_verifier_rocprof.py`, `gguf_decode_rocprof.py`,
`gguf_mtp_draft_rocprof.py`, `gguf_continuous_owner_rocprof.py`, `gguf_sh_c0_profile.py`,
`qwen35_rocprof_audit.py` and `mtp_verifier_rocprof.py` each carry their own copy of
`_default_roctx_sdk` + `_prepare_roctx_override`. Today's fix (probe the prefix of
`which("rocprofv3")`, then fall back to legacy `libroctx64`) landed in `gguf_packed_ar_rocprof.py`
only, because that is the one with a measured end-to-end verification and a test. The others still
fail with "rocprofiler SDK ROCTX library not found" under `.venv` unless `--roctx-sdk` is passed.
Adopt it by extracting `scripts/roctx_discovery.py` and importing it from all eight the next time any
of them is touched for another reason - do not sweep them blind, since wrapper behaviour changes
invalidate profile comparisons other lanes may be mid-run on.

Status update (later 2026-08-30). Four of the eight now resolve: `gguf_mtp_verifier_rocprof.py` via
8c59be6d8, and `gguf_continuous_owner_rocprof.py` / `gguf_decode_rocprof.py` / `gguf_packed_ar_rocprof.py`
by this unit, each carrying an inline candidate list over `{sys.prefix, sys.base_prefix} x
{_rocm_sdk_core, _rocm_sdk_devel}` plus `/opt/rocm`. `tests/test_unit_scripts_roctx_sdk_discovery.py` pins
all four against the installed SDK, so a regression to a prefix-only list fails a test rather than a
profiling arm. This touched only discovery, not what any wrapper measures, and lanes that pass
`--roctx-sdk` explicitly are unaffected - so no profile comparison is invalidated. One correction to
the paragraph above: `gguf_packed_ar_rocprof.py` did carry the `which("rocprofv3")` helper (still at
line 988), but `_default_roctx_sdk` in that file was still `sys.prefix`-only, so its `--roctx-sdk`
default was broken like the others; "the one with the fix" was true of the helper, not of the default.
Extraction into a shared module is still open, and the remaining four (`gguf_mtp_draft_rocprof.py`,
`gguf_sh_c0_profile.py`, `qwen35_rocprof_audit.py`, `mtp_verifier_rocprof.py`) discover the SDK by
other means and were not touched.

### GGUF MTP accept pinned staging (kept, measured neutral)

- `hipengine/generation/qwen35_gguf_mtp2.py::_accept_staging_backing` / `_upload_accept_array_staged` is
  default-on with the blocking `copy_host_to_device` path as automatic fallback. Measured perf-neutral
  (worklog `20260830T160316…`): uploads 85.4-101.6 ms -> 0.1 ms per lane-cycle with no rate change.
- Audit defect (2026-08-30): the fallback is not actually complete. `runtime.host_register(...)` exceptions
  escape instead of selecting the pageable path, and `Qwen35GGUFMTP2Adapter.close()` neither calls
  `host_unregister()` nor clears the registered NumPy arena. Before keeping this path as load-bearing code,
  track the registered pointer/runtime, unregister it on close and partial initialization, catch registration
  failure, and test both paths. If the accept cycle redesign removes staging, delete it instead.
- **Repaired 2026-08-31.** A failed `host_register` now selects the pageable path and latches the adapter
  off (`_accept_staging_backing` -> `_reset_accept_staging`), the registered pointer and `host_unregister`
  callable are tracked, and `close()` releases the arena exactly once as its final teardown action. A failed
  unregister therefore still propagates after ngram, batch-accept workspace, and target scratch cleanup;
  a runtime without `host_unregister` still drops every arena reference. The page-locked 4 MiB arena no
  longer leaks per adapter. CPU lifecycle coverage lives in
  `tests/test_unit_qwen35_gguf_mtp2_accept_staging.py` (registration failure, sticky-off, single registration,
  unregister-on-close, failure-safe remaining cleanup, missing-unregister release, no re-registration). Both runner call
  sites are teardown-only, so the latch cannot disable staging for a live session. Still neutral: this is
  lifecycle hygiene, not a rate win.
- Remove the fallback only if pipelining work makes the page-locked path load-bearing, or delete the staging
  path too if the accept cycle is redesigned (route (b), `20260830T155541…`). Do not re-justify it as a speedup.

### All-GEMV small-row prefill A/B - measured verdict: GEMV is worse

- Measured with `HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL=0` against the shipping WMMA route, same protocol, widths
  1-3 (worklog `20260830T164432…`): AR -15.0% / -21.8% / -29.9% and explicit K3 -20.4% / -28.6% / -36.1%.
- Verdict: the WMMA route is the better one and the knob stays diagnostic-only. Remove the knob once the
  small-row WMMA kernel itself is fixed; do not promote the GEMV route as a small-row fallback.

### Grouped q8_1 DP4A planar-Q6 variant - retained C8 default with temporary rollback

- Kernel `q6_k_t16_qmicro_planar_q8_1_dp4a_gemv_grouped_kernel<8>` (rows
  8-64, integer dp4a decode over planar-qmicro tiles, u32 record loads) is
  registered as `linear_q8_1 / t16_q8_1_dp4a_gemv_grouped_bf16_bf16_out`.
  The L4 gate and fresh final-stack composition gate passed; production
  Qwen3.8 `Q4_K_M` physical C8/K3 now defaults on at `c1bb38d49`. Artifact:
  `benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json`.
- The runner still carries the temporary `q6_dp4a_grouped_target_session`
  ContextVar and `HIPENGINE_C8_Q6_DP4A_GROUPED` env. The production profile
  binds one; strict and explicit zero bind zero. Runtime admission is fail
  closed to gfx1100 + request count 8 + `Qwen3.8` model metadata +
  `MOSTLY_Q4_K_M`, so C1-C7, Qwen3.6, Q4_K_S, and peer backends keep grouped
  BF16.
- **Removal condition:** after one release/bake interval without a production
  regression, replace the env/ContextVar experiment plumbing with a packaged
  model/profile-qualified dispatch policy and remove the rollback variable.
  Keep the registered DP4A kernel and strict BF16 fallback. If the default is
  rolled back during that interval, revert the production binding and retain
  this entry with the concrete failure artifact.

### C8-P2 Q5 planar-dp4a grouped route (default-off; added 2026-09-04, iter40)

- **What exists:** grouped integer-dp4a planar-Q5 sibling
  (`kernels/hip_gfx1100/quant/gguf_q5_k_qmicro_planar_gemv.*`), a
  session-gated R24 dispatch route in `gguf_linear.py`
  (`t16_q5_planar_dp4a`; rows 24 only, R32 keeps the retained source-layout
  owner), a resident `qmicro_planar` weight sidecar (loader flag
  `dense_q5_qmicro_planar_ssm_out`, env `HIPENGINE_C8_Q5_PLANAR_DP4A=1`
  + `GGUF_C8_Q5_RAW_MMQ_SSM_OUT` capability), runner session wiring, and
  the leaf script `scripts/qwen38_c8_grouped_q5_planar_dp4a_leaf.py`.
  The zero-valued env rolls the route back at both layers.
- **Why:** leaf screen (2026-09-04, actual Qwen3.8-27B-Q4_K_M Q5_K ssm_out
  weights, shared quantize included): candidate 0.77-0.84x of the retained
  R24 raw MMQ owner, wins 9/9 across rows 8/24/32; artifact
  `benchmarks/results/2026-09-04-w7900-qwen38-q4km-c8-q5-planar-dp4a-grouped-leaf.json`.
- **Default-off by construction:** nothing dispatches it without the env;
  the retained raw (R24) / source-layout (R32) MMQ owners remain the route
  defaults and registered strict fallbacks. Promotion requires the full L4
  production-profile campaign (arms differ by bf16-reassociation only at
  the decode, sharing the d4s4 activation producer; owner-parity 1e-4 and
  outer KL/top-1 contracts already registered) plus the tracked-clean
  ten-prompt e2e plus heldouts with candidate/control ordering.
- **Engagement deadlock fix (iter42, 7e03b150c):** the iter40 dispatch gate
  never fired (session.source_layout conflation + registration deadlock);
  the leaf/e2e "bit-exact" evidence before this fix was vacuous. Post-fix
  engagement is evidenced by launch counter (144 launches / 3 verify
  passes), and the leaf artifact above must be re-read as
  owner-vs-owner (route never fired).
- **L4 + retention outcome (engaged, iter42):** full L4 gate PASSED (18
  prompts, kl_mean 1.02e-4, kl_max 2.6e-3, top-1 0.9977, determinism 3/3;
  artifact
  `benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p2-q5-planar-dp4a-l4-numerics.json`);
  k2/R24 retention e2e +0.44% MTP aggregate, positive both orders, IDs
  identical, 40/40 exact
  (`benchmarks/results/2026-09-04-w7900-q4km-k3-c8-p2-q5-planar-dp4a-retention-e2e.json`).
  The route STAYS default-off: +1.23 GB persistent sidecar for a target-stage
  delta inside control spread does not justify promotion; the env pair
  remains the operator opt-in for k2-shaped runs.
- **Allocation-formula gap closed (2026-09-07, UD-U1 repair regression
  round):** `planned_qwen35_gguf_weight_allocation_nbytes` initially had no
  `qmicro_planar` formula (the route audit reported the resident
  formula-unavailable rather than guessing), and the new mandatory admission
  materializability check turned that gap into a hard refusal of this
  previously loadable resident. The formula is now exact and derived from the
  actual converter/allocation ABI: the INT8 `planar.tiles` payload of
  `convert_gguf_q5_k_qmicro_tile16_to_planar(repack_gguf_q5_k_qmicro_tile16(raw[None, ...]))`
  = `experts * (out/16) * (bytes_per_row/176) * GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES`
  (3328 B/block; measured 5120x6144 ssm_out sidecar = 25,559,040 bytes),
  gated to Q5_K T16 residents exactly like the materializer's own sidecar
  gate. Byte-exactness is pinned against the real CPU converter in
  `tests/test_live_gguf_ud_admission.py`; the audit reports the sidecar
  formula-sized. No kernel, dispatch, or default-change is involved.
- Remove the kernel, wrapper, registry entry, session/runner/env wiring,
  loader sidecar, leaf script, and tests if the P2 residual closes by
  another axis AND the flag pair has had no operator use for a full
  campaign cycle, or if a later owner-parity regression voids the axis.

## C8: direct-resident verify route scope (added 2026-09-04, iter60)

The 2026-09-02 direct-resident promotion (7daa03723) enabled
`GGUF_DIRECT_RESIDENT_VERIFY_LINEAR_STATE_POLICY` for all packed target
verification, but its gate evidence covered only C5-C8 multi-lane shapes. The
iter60 P8 matrix recapture found the single-session (width-1) shape reads a
wrong/stale initial Conv/GDN state through the direct mapping: ar_mtp_equal
lost on all 10 prompts, acceptance 0.789→0.166, w1 K3 34.7→21.5 tok/s, while
widths 2-8 stayed exact. Fix 29b06611f scopes the route to multi-session
batches (len(sessions) >= 2) with a RED-first contract
(`test_gguf_direct_resident_verify_route_requires_multi_session`).

- Root cause is scoped, not root-caused: why the single-slot mapping diverges
  (slot-index vs layout interplay at one packed slot) is unknown. If anyone
  re-widens the route, first reproduce the width-1 divergence with a
  device-level trace of `layout.state_indices` vs the captured-row reads and
  fix the mapping, then re-run the width-1 explicit-K3 e2e probe
  (ar_exact 10/10 required) plus the C1-C2 two-order gate.
- Remove this entry only when the route is either root-caused and re-qualified
  for the single-session shape, or the policy is deleted outright.

## RF-OI5 — resolved: profile-owned MTP width/depth policy (2026-09-05)

- The scalar physical-width ceiling is now a package-owned set of exact
  `(physical width, draft depth)` cells resolved by the shared GGUF MTP
  adapter. gfx1151 production preserves C1-C4/K1-K3 and adds C8/K3; C5-C7
  fail closed to AR. Its strict profile preserves the certified C1-C4 cells.
  The independently evidenced gfx1100 policy is narrower: production owns
  `(1,2)`, `(1,3)`, `(2,2)`, and `(8,3)`, while strict owns `(2,2)`.
  Model evidence still rejects every unmatched model/quant/profile/context/
  horizon/sampling key to K0.
- gfx1151 C8/K3 measures 52.103 tok/s versus 52.025 true AR (1.0015x), with
  exact cancellation/refill lifecycle and full drain. gfx1100 automatic C8/K3
  measures 98.643 tok/s versus 88.250 true AR (1.1178x), with 10/10 exact
  task cells. Evidence:
  `benchmarks/results/2026-09-05-gfx1151-qwen38-c8-k3-width-policy-retained.json`
  and
  `benchmarks/results/2026-09-05-w7900-q4km-k3-c8-automatic-promotion.json`.
- **The gfx1151 half of that row is withdrawn (2026-09-19).** Re-measured on the
  same instrument the cell is 46.88 tok/s against 50.10 true AR (0.936x) with
  0 of 10 cells reproducing the AR output, because the prompt-streaming policy
  refuses a pending set of 8 rows while the route runs anyway; the gfx1100 cell
  is untouched by that measurement. See the 2026-09-19 entry at the top of this
  ledger and
  `benchmarks/results/2026-09-19-gfx1151-qwen38-mtp-width-census-and-c8-k3-withdrawal.json`.
- Roll back by removing only the failing exact cell from that backend/profile
  policy. Do not restore a scalar ceiling or broaden intervening widths. Keep
  the registered strict fallback.

## RF-M5 — production whole-batch AR route for over-width MTP due items (2026-08-31) — RESOLVED

Resolved: `_maybe_run_partitioned_speculative_decode` now treats a zero
partition bound as one whole-batch AR step instead of "do not partition"
(`tests/test_unit_specdec2_engine_loop.py::test_zero_partition_bound_stops_a_wide_capable_runner_from_cycling`
pins the engine decision against a runner whose capability would grant the
wide cycle), and the admission owner caps gfx1151 production MTP widths at
four (9c8ff5444), so C5-C8 never enter MTP and the census artifact's true-AR
column stays their operative baseline -- no remeasure row is claimed here.
Decision: the partitioner-level threshold is retained deliberately as the
zero-bound signal the engine maps; it is not a redundant defense. The
observation below records the measured inert state from 2026-09-19.

`GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS = {"production": 4}` plus a
zero `partition_max_requests` result make an over-bound due batch take one
full-width AR step rather than chained MTP subgroups. **Measured 2026-09-19: it
does not.** `_maybe_run_partitioned_speculative_decode` treats a zero bound as
"do not partition" and returns `None`, so `_run_decode` falls through to
`_maybe_run_speculative_cycle` on the whole wide work item, and the 8-row
width-8 protocol runs 8-row speculative cycles this way. The defense was inert
on the server benchmark only while admission capped explicit groups at four
before the partitioner saw them; admission broadened when the 2026-09-05 width
policy listed `(8, 3)`, so this entry's precondition is met. Evidence:
`benchmarks/results/2026-09-19-gfx1151-qwen38-mtp-width-census-and-c8-k3-withdrawal.json`.
Move demotion to the admission owner (or make a zero partition bound mean one
whole-batch AR step), remeasure C5-C8 against current AR, and remove the
partitioner-level defense once redundant.

## RF-B1A — MTP serving target WMMA-prefill transfer (2026-09-02)

`HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL` remains the explicit override
around the production-profile transfer from July's per-row GEMV verifier
owners to retained exact prefill-band owners. The current gfx1151 one-group
C5-C8 gate improves 56.3-72.7%, preserves all 40 IDs, and passes the sec-6
teacher gate at 100% top-1/max KL 6.5e-4. Strict/profile fallback keeps GEMV.
Retire the override only when a future verifier family supersedes this owner;
keep registered strict fallbacks.

## RF-B5A — selective planar-Q6 integer-MMQ route (2026-09-03)

`HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL` is the rollback override for gfx1151
B5's production rows17-48 planar-Q6 route. The dispatcher selects the
registered Q8_1/`mmq64x64` composite only under its exact quant/shape/row
policy and an active bounded resident workspace. The complete gate improves
C5-C8 one-group MTP 1.75-2.08%, passes 216 rows at 100% top-1/max KL 0.002231,
and preserves all task IDs and acceptance. Strict, peer, standard-Q6, Q4/Q5,
and misses keep exact A owners. Retire the env override after its rollback
window; retain policy and strict fallback.

## RF-B2A — prefill F16-staging route switch (2026-09-02)

`HIPENGINE_GGUF_PREFILL_F16_STAGING` remains the rollback/diagnostic override
for gfx1151 Qwen3.8 production Q4/Q5-T16 prefill rows17-1024. The session casts
BF16 activation input into a bounded resident-owned F16 workspace and launches
the registered F16-input/BF16-output siblings; missing or undersized ownership
fails closed to BF16. Complete C2/C8 wall falls 18.13%/17.35% and profile logits
are bit-identical. Strict, fallback, no-profile, and env zero keep BF16. Retire
the override after the normal rollback window; keep registered BF16 fallbacks.

## 2026-09-06 Qwen3.5-family XML tool-call surface — open follow-ups

The server now prompts for and lifts the Qwen3.5-family XML function envelope
(`<tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n</function>\n</tool_call>`);
the legacy JSON body remains accepted by the parser. Three pieces were
deliberately kept on the legacy JSON path and should move once a real-model
run justifies each:

- `_tool_call_name_prefix_text` still forces the atomic JSON prefix
  (`<tool_call>{"name":...,"arguments":`) for `tool_choice=required/function`.
  Forced tokens keep those requests machine-formatted and parseable, but the
  prefix fights the trained XML format. Replace with
  `<tool_call>\n<function=NAME>\n` after a required-mode XML run on the PARO
  or Qwen3.8-Flash-Next model, and update the capability descriptor's
  `specific_tool_name_prefix_forcing_scope`.
- `_tool_call_close_repair_token_sequences` has no
  `</function>\n</tool_call>` completion candidate, so a length-truncated XML
  envelope fails closed as `invalid_tool_call` instead of being auto-closed
  the way truncated JSON envelopes are. Add the tokenized XML close sequence
  once truncated XML calls are observed in practice.
- `render_chat_prompt` wraps each `tool` message in its own
  `<|im_start|>user` block; the checkpoint template merges consecutive tool
  messages into one user block. Merge when multi-tool transcripts show
  degraded behavior.

## 2026-09-06 screening override — removed 2026-10-26

The switch, runtime screening branch and response markers were removed by
R0077. See "Explicit MTP capability admission replaces screening" above.
The following describes the superseded diagnostic contract:

`HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS=1` lets `Qwen35GGUFMTP2Adapter`
admit width/depth cells outside the backend's listed
`GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS` policy so the Qwen3.8 gfx1100
better-MTP campaign (Packet 0) can measure explicitly unqualified cells.
It is scoped to groups whose every request carries static eligibility with
`automatic_eligible=false`, so automatic policy cannot widen and qualified
Qwen3.6/gfx1151 routes are untouched; admission is recorded in
`physical_width_contract()["last_screening_cell"]`. Remove this env, the
`_physical_width_depth_admitted_for_group` screening branch, and its tests
when the campaign's final C1-C8/K1-K7 matrix lands and every measured cell
either has a qualified serving-evidence row or stays explicitly rejected.

**Open interaction (2026-09-21).** The guard now also blocks screening every
BF16 cell the implementation declarations cover, because those declarations are
automatic-eligible: `automatic_eligible` no longer implies "this cell is in the
listed policy", which is the invariant the guard was written against. A cell
covered by an explicit-only evidence row (`automatic_eligible=false`) is still
screenable; a cell with no row of its own is not, and reaches the adapter as
ordinary automatic intent instead. Measuring a BF16 cell outside the listed
policy therefore needs the guard keyed on the listed cell table rather than on
`automatic_eligible` -- decide that with the campaign owner before extending the
matrix, since removing the guard lets screening widen automatic intent.

## 2026-09-07 Dual C1 target-route dispatch — singleton vs packed frontier

`Qwen35GGUFMTP2Adapter` dispatches one-request C1 cycles to the legacy
singleton target verifier (graph-captured, device-side acceptance,
`_ensure_active_singleton_target_verifier`) and everything else (C2-C8 and
packed-C1) to `_execute_target_frontier_batch` (eager, host accept sync).
The 2026-09-07 stage-timing diagnostic
(`benchmarks/results/2026-09-07-w7900-packed-route-c1-stage-timing-diagnostic.json`)
showed the packed C1 cycle costs ~88 ms versus the legacy graph replay's
~48 ms for the identical 4-row frontier, entirely from eager launch overhead
plus the ~57 ms/cycle host accept readback. The packed route can at best
reach parity at C1 (same verify work; no batching advantage with one
request), so the legacy singleton route is the correct C1 default and stays.
When (and only when) the packed frontier path is graph-captured with
device-side acceptance and re-measured at parity-or-better on the C1
balanced-pair protocol, delete the singleton verifier dispatch branch and
its graph-bucket preparation so one target route serves C1-C8. Until then
this is a documented duplicate dispatch route, not dead code.

## 2026-09-09 INT8 layer_outer hidden-plane alias — route-scoped gate pending promotion

`HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS` aliases the two full-capacity
BF16 prefill hidden planes into one physical plane on the non-DMS
pure-INT8 route whose lifetime plan resolved to
`layer_outer_shared_oracle` (the direct-resident capacity route from the
2026-09-08 engine comparison; two planes are 2.69 GB at 128K there).
Promoted to default-ON 2026-09-09 with the route's own GPU evidence
(synthetic 73,728 A/B: one-plane peak reduction with identical decode
rate and tokens; real-prompt 8,192-token same-session A/B: byte-identical
greedy IDs; capacity ladder 139,264/147,456/155,648 pass, 163,840 OOM).
Env "0" is the explicit rollback. Fold this gate into the DMS
`HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS` decision (or remove whichever env
becomes redundant) once both routes have carried their adopted evidence
through a same-suite release cycle without a rollback invocation.


## 2026-09-09 GGUF direct-INT8 prefill attention — speed fixed, numerics rejected, opt-in capacity lever

`HIPENGINE_GGUF_INT8_PREFILL_DIRECT` ports the PARO route's
`streaming_direct` structure to the GGUF resident INT8 layers: prefill
attention reads the retained INT8 store directly (BF16-output kernel
family), the BF16 oracle pair is never allocated, and the lifetime plan
resolves to `chunk_outer_direct_int8` with chunk-sized hidden planes.

Follow-ups to the original screen (which was speed-blocked at 164.2 tok/s
and mis-baselined: the 761.6 oracle control requires the
`--public-ar-profile` WMMA projection path):

- **Speed**: the sequential kernel is superseded by the GQA-grouped flash
  kernel `qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans`
  (one block per (kv head, query row), one warp per q head of the
  six-head group, 32-token K/V tiles cooperatively staged into LDS with
  16-byte vector loads, online softmax in registers; no score workspace
  or split-K partials at any context length; 24/4/256 geometry).
  `HIPENGINE_GGUF_INT8_PREFILL_KERNEL` selects flash (default) or
  sequential. Harness A/B at 8,192 tokens: oracle 767.4 / direct+flash
  552.5 tok/s (72%); the remaining 1.39x needs a WMMA score GEMM
  (bf16 fragments are exact for int8 K and the already-BF16-rounded Q).
- **Numerics**: the campaign REJECTED the route for default-on. Against
  the BF16 reference at 4,096 tokens (no-mirror configuration): the
  oracle route measures 0.003 mean row-KL; the direct route measures
  0.215 (max 6.4, top-1 to 0.71) — 70x the smoke ceiling — identically
  for the flash and sequential kernels, so the drift is the INT8-read
  prefill arithmetic itself, not the kernel. The BF16 oracle pair is
  what preserves prefill quality on the pure-INT8 route. The gate stays
  default OFF permanently unless the read precision changes.
- **Capacity**: the route remains an opt-in capacity lever with the
  quality cost measured above; the flash kernel makes the ladder
  practical (163,840 tokens completes; see the gate artifact).

Related latent bug found and documented: the pre-existing
`direct_hadamard_int8` GGUF branch is dead code whose fp16 output dtype
also mismatches the BF16 `full_gated` consumer - it shares the fixed out
type but remains unexercised; do not enable it without its own e2e gate.
A second latent route bug fixed alongside: dense H5120 Q4_K_M auto
full-attention chunks (4,096 rows below the 52K threshold) overflowed
the 1,024-row scratch cap (positions-buffer copy error at 8K/16K
prompts); both chunk resolutions now clamp to the row cap.

## Slot-local INT8 oracle prefill AOTriton admission flag

- Added 2026-09-09. `HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON` (default
  OFF) admits AOTriton on slot-local full-attention prefill layers that own a
  transient BF16 oracle. Before the flag, the `int8_direct` route hard-disabled
  AOTriton for those layers (`allow_aotriton=not transient_direct_oracle`, added
  by the compact-serial-c4 qualification `c2ca58171` with no recorded rationale),
  leaving every server INT8 request above the 8,192-position mirror threshold on
  the native split-K paged kernel at 16 query rows per batch. Measured on the
  W7900 at 8,192 tokens: 44.31 -> 121.44 tok/s (2.74x), peak -24 MiB, generated
  IDs unchanged. Default OFF is bit-identical to the pre-flag tree.
- Removal trigger: run the production-profile numerics gate with strict = flag
  off and candidate = flag on over the full mtp-bench category suite. On a pass,
  make admission unconditional, delete the flag and the
  `_gguf_slot_local_prefill_allow_aotriton` helper, and fold its tests into the
  route's parity coverage. On a fail, keep the flag OFF and record the binding
  threshold here.

## Packed execution workspace doubles the declared-context page budget

- Recorded 2026-09-09. `configure_engine_loop`
  (`hipengine/generation/qwen35_gguf.py:5560-5592`) sizes the eager packed
  workspace lease as `capacity * max(max_pages_per_request, 1024//256)`, so the
  global KV arena is allocated at exactly **2x** the declared context: request
  pages plus an equal pinned lease. Measured on the retained XTX audit
  (`2026-09-07-rx7900xtx-int8-repair-capacity-audit.json`): pool pages are
  126/256/384/424 for 63/128/192/212 request pages at 16,128/32,768/49,152/54,272
  declared context, at 33,280 B/token, i.e. 32.5 KiB/token of duplicate budget on
  INT8 and 64 KiB/token on BF16. The 4-point server peak fit is 99.4 KiB/token
  against the direct engine's 33.6.
- The lease is **not** simply over-reserved: although the packed verifier is
  guarded to context < 1024 (`runtime/qwen35_gguf_runner.py:19907`) and slot-local
  prefill never writes the packed KV planes (`copy_kv=not slot_local_full_prefill`),
  `_sync_packed_decode_initial_state` (`runtime/qwen35_gguf_runner.py:25702-25713`)
  copies the entire KV history into those planes on the prefill->decode
  transition. Shrinking the lease alone converts a silent over-reservation into a
  `RuntimeError` at long context.
- Removal trigger: the CONCURRENCY2 task-5 follow-up already recorded in
  `worklog/entries/20260818T121922.772662Z-lhl-concurrency2-task5-layout-rebind-c9eadd.md`
  ("packed slots borrow the row's own lease pages; flush scatters only
  conv/recurrent state"). Land that, then size the lease to
  `capacity * (_PACKED_VERIFY_MIN_MAX_SEQUENCE // 256)` and re-measure the server
  single-request context ceiling, which is stale at 54,272 independently of this.

## `HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON` — promoted, remove after soak

- Promoted to default ON 2026-09-10 (see the capacity campaign doc and
  `benchmarks/results/2026-09-10-w7900-int8-slot-local-aotriton-gate-corrected.json`):
  the slot-local transient-oracle AOTriton admission now matches the engine-wide
  AOTriton default that the direct arm already runs. `=0` is the explicit
  rollback to the native split-K paged kernel.
- Removal trigger: once the parity roadmap P1/P2 follow-ups (server-faithful
  allocation probe, R0-R4 matched walls) confirm the route with the default on
  and no rollback demand is recorded, delete the flag and the
  `transient_direct_oracle` gate term so slot-local admission is unconditional,
  keeping the native kernel only as the sub-512-row crossover and the
  AOTriton-wrap failure fallback.

- [2026-09-09] TimesFM `_Buffers.scores` is no longer read by the FP16 attention path (WMMA flash kernel replaced the scores GEMM + softmax chain; 2026-09-09 flash-attn unit). Remove the allocation and its `f16()` sizing once the FP32 strict path also stops using it or a strict-path replacement is registered.
## PF-1d MMQ tile template (2026-09-02)

- `gguf_q8_0_mmq_prefill.hip` keeps the tile-parametrized
  `q8_0_raw_mmq_tile_q8_1_d4_kernel<MMQ_M_T, MMQ_N_T, ...>` plus
  `launch_mmq_tile<>`; only the `<128,128>` instantiation is registered.
  Remove the template parameters (fold back to constants) if no second tile
  variant is admitted by the end of PF-1e. Measured losers m64x64 (0.89x) and
  m32n128 (0.35x) were removed after rejection; the cross-variant bit-parity
  gate in `tests/test_gpu_qwen4exp_pf1_dense_parity.py` is the readmission path.

## Q8 MMQ `policy.planes` is variant-scoped (2026-09-03)

- `Q8MMQPrefillPolicy.planes` (`gguf_q8_0_mmq_prefill.py:63`) is consulted only
  for the `gemv_f32_f32_out` / `prefill_f32_f32_out` variants
  (`hipengine/runtime/gguf_linear.py:5883`); `prefill_bf16_bf16_out` is
  hard-wired to `d4x3` at line 5871. A gate that flips `planes` therefore
  measures nothing on shapes that dispatch the BF16 variant, which is how the
  18-prompt reconciliation run produced trajectories bit-identical between
  `planes=2` and `planes=3` on the two prompts that *did* engage the route
  (`2026-09-03-...-pf1-admission-suite-reconciliation.json`).
- **Not yet traced.** The mechanism above is inferred from dispatch source, not
  confirmed by `rocprofv3 --kernel-trace`. Confirm or refute it before any
  further plane-family work.
- **Removal trigger:** the d4x3 chain is final for this family (PF-1d closed
  rejected). If no plane candidate is readmitted, delete the `planes` field and
  the `d4x2` dispatch branch together with the `mmq128_prefill_q8_1_d4x2_*`
  registrations, so no knob remains that silently applies to a subset of
  variants. If a candidate is readmitted, make `planes` bind on every Q8 MMQ
  variant or rename it to state its scope.



## PF-4 lever 2 fused combine is opt-in pending valid A/B (2026-09-04)

- `hipengine/runtime/qwen4_exp_runner.py` keeps the fused
  `weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w` grouped-prefill
  path reachable only via `HIPENGINE_QWEN4_EXP_FUSED_COMBINE=1`; the default
  is the unfused `weighted_lanes_sum` + `shared_gate_combine` chain. The flag
  was originally the inverse (`HIPENGINE_QWEN4_EXP_UNFUSED_COMBINE=1` gating a
  fused default) before a process-separated diagnostic measured pp −1.69%
  (all 12 cases negative). That run cannot close the lever under the required
  one-residency protocol.
- The fused kernel and its registry entry
  (`KernelKey("hip_gfx1100", "weighted_lanes_sum+shared_gate_combine", <quant>, "out")`)
  stay bit-exact and pinned by `tests/test_gpu_qwen4_exp_pf4_lever2_fused_combine.py`.
- **Removal trigger:** repeat the whole-model comparison in one process and
  one model residency. Promote only if that run is exact and non-regressive;
  otherwise delete the fused kernel, its wrapper/registry entry, the
  `HIPENGINE_QWEN4_EXP_FUSED_COMBINE` gate, and the lever-2 tests once PF-4 is
  formally closed and no follow-up candidate reuses the fused chain. Keep the bit-exact-parity test file if the unfused chain ever changes
  arithmetic.

## Qwen4Exp Q5_K grouped row4 candidate (2026-09-05)

`selected_grouped_row4_gemv_bf16_bf16_out` is now selected by the production
binder through `HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL`; strict binds zero.
The clean Framework all-category gate retained the path. Keep the post-binder
opt-out for rollback and the reproducible A/B harness; replace it with direct
manifest-driven selection when runtime profile dispatch owns this boundary.
Always retain selected-GEMV strict fallback.

## Qwen4Exp H256 wave sparse rows (2026-09-05)

Production selects page256 wave prefill through
`HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL=page256`; strict binds zero.
The owner accepted the measured hot-decode tradeoff on 2026-09-05.
Keep the score-product register boundary: removing it changes arithmetic
under the current compiler. Preserve `strict_rows_spans` and the separate
ordered decode parent. Retain the post-binder opt-out for reproducing the
tradeoff and rollback; remove the generic-wave experimental selector when
no longer needed for bisection. Decode performance remains open work.

## Qwen4Exp Q4 bundled publication (2026-09-05)

Production selects bundled Q4 publication through
`HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL=1`; strict binds zero. Its complete
12-case timing and real-model state gates pass correctness, and the owner
accepted the measured decode tradeoff. Keep the old exact grouped owner as
strict fallback. Replace the environment selector with direct profile-owned
selection when that runtime boundary is migrated; keep rollback capability.

## Qwen4Exp Q5_1 output-pair reuse (2026-09-05)

Production selects pair2 through `HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL`,
guarded by rows>=64 and registry capability; strict binds zero. The full
12-case A/B retains a3.73-4.17% prefill win with exact outputs. Short rows
and the post-binder opt-out keep M1; strict keeps its original exact parent.
Remove the environment selector when this runtime dispatch moves directly
under the manifest; retain rollback and registered strict fallbacks.

## Qwen4Exp register-state serial GDN rollback

- Production binds `HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL=1`; strict binds0.
  The registry/shape-guarded serial branch uses Hk16/Hv48/D128 and rows>=2.
  Suffix arithmetic and decode routes are unchanged.
- Full72-trajectory admission passes exactly; prefill improves9.50-10.55% and
  every request wall improves3.28-6.79%. Decode p4096 loses3.63% amid drift.
  Retain the win under the owner's prefill-first direction and optimize decode
  next; do not hide the adverse phase rows or claim thermal causality.
- Keep the registered strict serial fallback. Remove the environment selector
  when dispatch is owned directly by the manifest and rollback needs expire.
- Resource review accepts256 VGPR/24-byte scratch with the measured whole-model
  win. Avoid the rejected per-column/every8 scheduling barriers.

## Qwen4Exp Q4 shared-activation output-pair admission

- `selected_dual_grouped_pair2_bf16_bf16_out` is retained after exact
  primitive/actual-weight gates and1.345-1.440x tokens512 gate/up+SiLU wins.
  Full state/KV/logit and72-trajectory A/B admission now pass.
- Production selects pair2 for supported rows>=64; retain the registered
  bundled/serial strict fallback. Do not reopen the rejected independent-team
  or wider serial-output sweeps: this path shares each activation load.
- Selector `HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL` is production1/strict0,
  guarded by rows>=64, supported K and registry availability.
  Remove the environment selector when the admitted profile owns dispatch
  directly and rollback needs expire; retain smaller-row/bundled fallbacks.

## Qwen4Exp Q4 larger row-batch screen rejected

- RB16/32 template instantiations and temporary wrappers/registry selectors
  are removed after all12 masked uniform/skewed x tokens512/1024/2048
  gate/up+SiLU cells lose to production RB8.
- Keep the harness's inferred weight-pass counts as diagnostic metadata;
  never label them measured DRAM/MALL traffic. Register counts grow88->120/176,
  with zero scratch in observed traces.
- Reopen only for a different resource/reuse mechanism, not the same template
  enlargement. No persistent candidate flag or default change remains.

## Qwen4Exp Q8 wave-scale rollback

- Separate registered `coltile8_rowbatch4_wave_scale_f32_f32_out` is retained
  after exact F32/CPU gates and actual attention-gate/shared-down wins.
  These wins are conditioned on symmetric256MiB device-fill preconditioning.
  Unconditioned attention-gate timings reverse by order; that adverse evidence
  remains. Normal-model full72-trajectory admission now passes exactly, with
  all12 prefill and request-wall results positive.
- Preserve original coltile as strict fallback. Remove candidate-only
  dispatch experimentation once admitted or rejected; do not claim SGPR
  instruction counts without inspecting generated code.
- Model-admission flag `HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE=1` selects a
  scoped prefill geometry with registry-checked fallback. Production1/strict0;
  MMQ/WMMA and decode stay unchanged. Remove the flag after admission/rollback
  needs end, or remove the candidate if the actual-model gate loses.

## Qwen4Exp GR wave-scale production rollback

- Separate registered GR up wave-scale composite is promoted after exact
  gate/mixed-output, full logits/state/KV and all72 normal-model trajectories.
  All12 cases improve prefill and request wall (weighted prefill+0.34-0.44%).
- Scope model admission to existing rows>256 fused GR use; rows64 microbench
  has small order reversals and is not a small-row promotion basis.
- Keep parent composite and unfused chain as strict fallbacks.
- Production-bound `HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE=1` selects the registered
  candidate only inside existing rows>256/four-branch fused GR use. Parent
  and unfused fallback selection remain intact; strict binds0. Remove the
  positive selector after one release window and a defaults-only refresh,
  retaining explicit rollback only while bisection needs it.

## 2026-09-10 TimesFM kernel families lack four-axis registry entries — open

Both TimesFM kernel families (`kernels/hip_gfx1100/timesfm/` for 2.5 and
`kernels/hip_gfx1100/timesfm3/` for 3.0) are direct-launched through their
Python wrapper modules with no `(backend, layer, quant, variant)` registry
registrations. The fused composites carry their strict fallbacks at the
runtime level instead (2.5: fp32 unfused path; 3.0: fp32 routes through the
separate var-q/k normalization chain). Register both families under the
four-axis registry with their strict fallback chains when a second consumer
of the kernels appears; until then the runtime-level fallback is the
contract of record (see docs/model-cards/MODEL-TIMESFM3.md).

## 2026-09-12 PARO paged-attention backend pin - removal rejected

`hipengine/runtime/qwen35_paro.py` sets
`_PAGED_KV_REGISTRY_BACKEND = "hip_gfx1100"` and passes it to
`resolve_paged_kv_write`, `resolve_paged_attn_decode`, and
`resolve_paged_attn_prefill`. The class docstring on
`Qwen35ParoDecodeState` states the assumption explicitly ("Kernel selection
still flows through the registry/wrappers added in the gfx1100 backend tree"),
and PARO is a live gfx1151 lane.

Consequence on gfx1151: PARO gets the gfx1100 body for every key it resolves.
For `paged_attn_decode / w4_paro / bf16_context_batch_c1_exact_spans` that is
visible, because `hipengine/kernels/hip_gfx1151/__init__.py` deliberately
overrides that key to `..._fixed256_spans` ("Keep gfx1151 on the generic
reduction, but pin its geometry to the c4/c8-proven 256-thread shape"). PARO
therefore runs the c1-exact kernel on gfx1151 while every other gfx1151
paged-attention caller runs the pinned generic reduction. The later same-host
PARO comparison showed that this difference is load-bearing: resolving against
gfx1151 diverged from independent c1 at decode token 25 of 137, twice, while
the current pin and the per-row control both matched all 137 tokens. The
generic override's GGUF qualification does not establish its correctness for
this PARO route.

Do not remove the pin as cleanup. Any replacement requires a separately
qualified PARO route preserving its declared profile and ownership contracts.
The open follow-up is to understand and separate the PARO and GGUF semantics
sharing this registry key, not to substitute the session backend unconditionally.
See the [upstream decision and reproduction protocol](../worklog/entries/20260912T044256.896835Z-lhl-paro-paged-kv-backend-pin-load-bearing-e64405.md).

## 2026-09-12 GGUF automatic context sizing knobs

`hipengine/generation/qwen35_gguf.py` added `HIPENGINE_GGUF_AUTO_CONTEXT`
(default on), `HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB`,
`HIPENGINE_GGUF_KV_TRANSIENT_KIB_PER_TOKEN`,
`HIPENGINE_GGUF_KV_TRANSIENT_FIXED_MIB`, and
`HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS` for the automatic resident context
selection.

`HIPENGINE_GGUF_AUTO_CONTEXT` is the rollback for the behavior change and is
kept until the automatic selection has a full benchmark cycle on the supported
target models. Removal condition is that no bisection case remains where the
fixed 256-token context is wanted. It also disables the allocation backoff, so
the rollback restores the historical behavior completely rather than half of it.

The other four are calibration overrides. `HIPENGINE_GGUF_KV_TRANSIENT_KIB_PER_TOKEN`
and `HIPENGINE_GGUF_KV_TRANSIENT_FIXED_MIB` front measured values (the packed
slot-local prefill route's live oracle and hidden owners, and its
context-independent execution workspace, measured 2026-09-11 and documented at
`_GGUF_TRANSIENT_BYTES_PER_TOKEN_DEFAULT`).
`HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB` fronts a measured value too: the
whole-card minus hipEngine-tracked difference on the W7900, 2.66 GiB, which is
HIP context, JIT kernel modules, AOTriton, and pool pointer tables — all
allocated after the free-memory reading the model prices against. Removal
condition for these three: once the reserve and transient terms are expressed as
named model terms rather than an override, delete the overrides and keep the
calibrated constants. `HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS` stays until the
backoff has production mileage.

An open accuracy item recorded in the `gguf-auto-context-sizing` worklog entries:
the transient term is priced at a full-context worst case (4.22 GiB at the 27B's
auto-selected context) against 0.30 GiB measured with a 43,011-token prompt, so
the model is roughly 3.9 GiB conservative there. That conservatism is currently
load-bearing — it is what kept the pre-fix 512 MiB reserve from turning into an
OOM — so any attempt to tighten the transient term must re-derive the reserve at
the same time.

The `_auto_resolved_max_sequence_length` / `_auto_resolved_max_sequence_lengths` /
`_auto_context_estimate` fields on `Qwen35GGUFBringupGenerator` exist only to feed
the server's KVCache summary and `/ready` payload. If the reporting path ever
reads the estimate from the session alone, drop the generator-side copies.

## 2026-09-12 GGUF serving defaults: INT8 KV, fp32 scales, one slot

The dense GGUF serving defaults changed so that a zero-flag `hipengine serve`
reaches the context the INT8 KV work makes available. Three coupled defaults:
`--kv-storage` is `int8_per_token_head` (was `auto`, which resolved to BF16),
`--kv-scale-dtype` is `fp32` (was `fp16`), and
`_GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY` is `1` (was `4`).

`HIPENGINE_KV_STORAGE`, `HIPENGINE_KV_SCALE_DTYPE`, and `--max-active-requests`
are the rollbacks and are documented in `docs/API.md`; they are the supported
tuning surface, not temporary flags, so no removal condition applies to them.

The `scale_dtype` coupling is the one that needs watching. Qualification is keyed
on the exact artifact SHA-256, size, backend, target, weight quant, layout, and
scale dtype, so `fp16` scales build a key that matches no retained contract and
INT8 fails closed to BF16 **without an error**: the server reports the requested
storage while the estimator and the pool allocate BF16. That failure mode is
silent by construction, because a fallback is a normal outcome. Two guards now
exist: `test_server_defaults_match_the_qualified_int8_contract` asserts the CLI
defaults key a registered `qualified` contract, and the KVCache summary line
reports the effective storage. If a future artifact's evidence is keyed on a
different scale dtype, that test fails rather than the context silently shrinking.

`admission_gated_int8` in `hipengine/kvcache/policy.py` remains unused. It was
the intended mechanism for "`auto` means INT8 when qualified", but the INT8
capability gate lives only in `qwen35_gguf.py`, so routing `auto` through it would
give the PARO route unqualified INT8 — the PARO call sites have no gate. The
server therefore reaches INT8 through an explicit storage request instead. Either
remove `admission_gated_int8` or give PARO a gate and use it; leaving it defined
and unwired invites someone to wire it and silently break PARO.

The KV policy comparison in `hipengine/server/api.py` now resolves the
*effective* storage from the engine's capability provenance rather than
re-resolving `config.kv_storage`. Before that, an artifact that failed closed to
BF16 would reject an explicit BF16 request, because the comparison still believed
the server was on INT8.

## 2026-09-16 Artifact provenance: execution-affecting dirty classification

Artifact provenance is schema 3. `collect_repo_state` now records
`dirty_paths` / `untracked_paths` (capped at 64 entries each, with
`dirty_path_count` and the existing `untracked_count` carrying the true totals)
plus an `execution_affecting_dirty` flag derived from
`is_execution_affecting()`. The classification is a denylist — `docs/`,
`worklog/`, `benchmarks/results/`, `.claude/`, `.github/`, and any `*.md` are
inert; everything else, including fixtures, `pyproject.toml`, and unrecognised
paths, counts. The conservative answer is therefore the default.

Two motivations. A bare `untracked_dirty` boolean meant any concurrent agent's
scratch file invalidated every measurement in the shared worktree, which is what
happened to both arms of
`benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/`. And a bare
boolean does not say *which* file was dirty, so a measurement cannot be judged
after the fact — the passing 32-47 arm of that gate records
`unstaged_dirty: true` with no way to recover what was modified.

`scripts/execution_profile_q8_wmma_prefill_layers_gate.py` and
`scripts/execution_profile_q8_mmq_plane_gate.py` key `measurement_valid` and
their qualification blocker on the new flag. Seven other call sites still use
the coarse `not provenance.get("dirty")` and raise rather than record a blocker:
`scripts/qwen36_moe_mtp2_production_gate.py`,
`scripts/qwen36_dense_mtp2_c2_production_gate.py`,
`scripts/specdec2_perf_bridge.py`, `scripts/qwen4exp_halo_box_campaign_ab.py`,
`scripts/q8t16_batch_route_perf.py`, `scripts/run_gfx1151_readme_refresh.sh`,
and `scripts/run_w7900_readme_refresh.sh`. Removal condition: migrate those to
the same helper once the publication protocols have been re-read, then delete
the `provenance.get("dirty")` fallback argument from the two migrated gates.
Retained-row publication scripts may deliberately want the stricter rule; decide
per script rather than sweeping.

## 2026-09-16 Qwen4Exp binder: pre-binder env conflicts are reported

The Qwen4Exp profile binder owns 66 env variables and overwrites all of them
during its pass, so a value set before generator construction is silently
discarded: the caller's route never runs and the experiment reports a clean null
that reads like a real negative result. That hazard is now surfaced rather than
documented. `_apply_profile_env` records every caller-set value the bind
discards, `last_prebinder_conflicts()` exposes them, and
`_report_prebinder_conflicts` warns — or raises when
`HIPENGINE_QWEN4_EXP_PREBINDER_CONFLICT=error`. The module tracks its own writes
separately so strict-then-production binding in one process is not a conflict.
`scripts/execution_profile_q8_wmma_prefill_layers_gate.py` raises on a non-empty
conflict map rather than measuring the wrong route.

An audit of all 31 scripts that mention a binder-owned variable found **no**
script that sets one before constructing a generator, so no prior null result is
attributable to this hazard. The established convention is
`qwen4exp_profile_gap.py`'s `_apply_post_binder_overrides`, which is correct.
The layer-scope gate's bespoke setup was the only site at risk and it already
flips post-binder.

Removal condition: when Qwen4Exp profile state becomes request-local (tracked in
the 2026-08-29 entry above), the env transport disappears and this conflict
reporting goes with it. Until then the warning is the guard; do not downgrade it
to a comment.

## 2026-09-17 Q8_0 wide-row dense prefill: a route for an unreachable kernel

`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` routes Q8_0 F32/F32 prefill linears with
`rows > 256` and `in_features % 64 == 0` to `dense_wide256_f32_f32_out`. Before
this the kernel was registered but nothing in `hipengine/runtime/` or
`hipengine/dispatch/` referenced it, so no dispatch path could select it: it
could not be gated end-to-end, benchmarked in the engine, or promoted. That is
the state to avoid repeating — a registered kernel with no selector is invisible
to every gate we own.

The route is the shipped default for `gguf_ud_q4_k_xl` on `hip_gfx1151` at
layers 16-47: a caller that names no profile resolves to `production`, whose
binder sets this flag, so no user action and no env var is involved. The
selector's own default stays off, so a process that binds nothing keeps the
exact chain — the unbound direction is always the exact route, never the f16
one. Both halves are pinned by
`test_the_shipped_default_routes_the_certified_window_to_the_wide_kernel`, which
joins the default resolution, the binder, and the selector's decision, because
pinning them in three separate tests let a break in the middle go unnoticed.

The reason the default cannot simply live in the selector is that the selector
cannot see the lane: `gguf_q8_0` is the tensor quant key in every pack that
stores Q8_0 dense tensors, so a lane-agnostic default-on would claim lanes whose
envelopes were measured against the exact chain. See "The default is per lane"
below for the measured set.

The exact coltile parents remain the registered strict fallback and the sole
owner of the sub-256-row path, so the selector declining is always a fall back
to an exact route.

`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS` narrows it to a layer scope, added
2026-09-17 after review found the first cut was a plain boolean: with no scope
the route covers all 48 layers, which is the 0-47 scope the sibling f16 route
already fails, so the flag could only express a configuration known to fail. A
weight with no layer index (`token_embd`, `output`, both Q8_0 in this model but
outside the blocks) is outside any scope and the route declines it.

Removal condition: run the calibrated production envelope in
`docs/EXECUTION-PROFILES.md` against this route at the deepest admissible layer
scope. **Met and executed 2026-09-17 at 16-47**: the route passes the 12-case
envelope with strict/candidate logits digests byte-identical to the certified WMMA
route's, so the production binder now binds it at that scope for
`gguf_ud_q4_k_xl`. What is left of this entry is the flag itself: once nothing
needs to bisect against the wide route, delete `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE`
and let the layer scope alone decide, keeping the coltile fallback registered. If
the promotion is ever reverted, delete the selector and the flag rather than
leaving a dead default-off route - the kernel's measured 6.74x over the
production dispatch is only worth keeping if it is reachable under a gate.

`_variant_scoped_library` was extracted at the same time. Two launch paths had
independently resolved caller-supplied libraries, one keyed by quant alone. That
is a live trap for any variant with its own shared object: `dense_wide256` is
exported from `gguf_q8_0_dense_wide.so` while the coltile family lives in the
q8_0 gemv library, so a quant-only lookup hands the wide kernel a library that
does not export its symbol. Both paths now share the helper. Fold the remaining
`libraries.get(<key>.quant)` call sites for pair and fused launches into it when
one of those families next gains a variant with its own object.

### Precedence over the f16 WMMA family is a decision, not a line number

`_wmma_prefill_dispatch` runs earlier in the launch chain than
`_q8_dense_wide_dispatch` and rewrites a claimable parent to
`wmma_<parent>`/`wmma_raw`. Until 2026-09-17 the wide selector required the
exact coltile parent, so on the promoted production default - which binds
`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` to layers 16-47 for `gguf_ud_q4_k_xl` - it
declined in its first condition, before its own gates, its layer scope or its
registration were consulted. Its certified window and the WMMA route's window
were the same window, so no configuration could run it: 134 eligible dispatches
in layers 0-15 passed the gates and fell to the scope check, and layers 16-47
never reached the selector's first print. The route now claims the rewritten
parent in its own scope, and the layer scope is the authority on where it runs.
Verified end to end on `code-p512`: layers 16-47 run
`hipengine_gguf_q8_0_dense_wide256_f32_f32_out` (264 roles) where the named
default runs `hipengine_gguf_q8_0_wmma_prefill_f32_f32_out` (same 264 roles),
layers 0-15 keep the exact coltile chain in both, and the two arms produce the
same `logits_sha256` (`e15dce79…`) - the wide kernel is a retiling of the
certified f16 arithmetic, not a new arithmetic class. Its end-to-end value at
that scope is measured too: on the `code` category at 4K it saves 5.43 s against
the exact chain where the WMMA default saves 4.44 s, and 5.4% against the default
at both 1K and 4K, with the two routes indistinguishable at 512 (three
interleaved arms, one process, two processes agreeing within 0.1 pp -
`benchmarks/results/2026-09-17-q8-dense-route-ab/`).

Removal trigger: once the calibrated envelope passes at the promoted scope,
make the wide route the default there and retire the f16 WMMA dense route for
this quant (keeping the registered exact coltile fallback). Do not keep both
families selectable for the same layers once one is the default: that is the
contention this fix had to resolve. **Executed 2026-09-17**: the envelope passed
with a bit-identical candidate digest to this route's
(`benchmarks/results/2026-09-17-q8-dense-wide-16-47-gate/`), the production
binder binds the wide route at 16-47 and no longer binds
`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS`, and the promoted default was verified to
launch `dense_wide256` on the 264 layer-16-47 roles with zero `wmma_prefill`
launches. What is left is deleting this route's selector and env var once
nothing needs to bisect against it.

### The default is per lane, and the route cannot see the lane

Promotion means the fast path is the default for the lane it was certified on,
and for `gguf_ud_q4_k_xl` on `hip_gfx1151` that is now true without any caller
action. It does not mean a lane-agnostic selector default, and the measured
reason is worth keeping written down. `_q8_dense_wide_dispatch` sees the tensor
quant key, and every pack that stores Q8_0 dense tensors reports `gguf_q8_0`, so
flipping its default on claims all of them. A GGUF header scan of the packs on
this box (2026-09-17, tensor-type histogram, no model load) gives the set:

| Pack | Shape-eligible Q8_0 dense | Block layers | Inside 16-47 |
| --- | ---: | --- | ---: |
| Qwen3.6-35B-A3B-UD-Q4_K_M | 259 | 0-40 | **25** |
| Qwen3.8-27B-UD-Q4_K_M | 106 | 0-64 | 26 (10 with >128 outputs) |
| Qwen3.8-27B-UD-Q4_K_S | 99 | 0-64 | 24 (3 with >128 outputs) |
| mtp-Qwen3.8-Flash-Next-Q8_0 | 18 | 48 only | 0 |

The Qwen3.6 pack is the case that decides it: its Q8_0 dense tensors are
`attn_qkv` (2048x8192), `attn_gate` (2048x4096) and `ffn_{gate,up,down}_shexp`,
which is real prefill work on 25 of its 41 layers, and that lane's production
envelope was certified against the exact coltile parents. The Qwen3.8-27B packs
are mostly `ssm_alpha`/`ssm_beta` (5120x48) with a handful of `attn_k`/`attn_v`
(5120x1024), so the exposure there is smaller but not zero. The MTP pack's single
block is layer 48, outside the window, so the drafter is unaffected either way.

Keep the unbound default on the exact chain. The asymmetry is the argument: a
lane that binds nothing today gets exact arithmetic, so a missing binding is a
slow measurement, while a lane-agnostic default-on would make a missing binding a
silently different arithmetic composition in lanes that believe they are exact -
the same misattribution class as the census coverage gap below.

Extension trigger: a lane joins the default by running the calibrated envelope
(`benchmarks/results/2026-09-17-q8-dense-wide-16-47-gate/` is the protocol and
the template) and then binding the route in that lane's production plan. The next
candidate is this model's own `gguf_q4_k_m` plan, which still binds `0`; its pack
is not on this box, so the envelope has not been run and the binding stays off.
A lane whose envelope fails keeps the exact chain at that scope rather than a
narrower window nobody measured.

### The variant manifest does not name env-scoped post-binder routes

The Qwen4Exp production manifest's `prefill_exact_q8_f32_coltile` selection still
names `coltile8_rowbatch4_wave_scale_f32_f32_out` for the Q8 dense prefill role,
while the binder has routed layers 16-47 to `wmma_prefill_f32_f32_out` since
2026-09-17 and to `dense_wide256_f32_f32_out` after this promotion. Both gate
artifacts therefore record the same `profile_manifest_sha256`
(`35e57360e648bcc8…`) for three different arithmetic compositions, and only their
`route.surface` distinguishes them. Removal trigger: mirror the layer-windowed
precedent this package already has (`prefill_rows_ge16_hk16_hv32_48_d128_layers27_47`
exists in both the strict and production plans) by adding a
`prefill_rows_gt256_layers16_47_q8_f32_dense_wide` scope to both plans, so the
manifest hash identifies the composition again. Do it in a unit that re-runs the
16-47 envelope under the new manifest, since a manifest change after a gate means
the gate did not cover the new identity. Until then, read the composition from
`route.surface` in the artifact, never from the manifest hash alone.

### The launch census has a per-module coverage boundary

`launch_census.record_launch` was called only from `gguf_k_gemv`. The f16 WMMA
prefill family (`gguf_q8_0_prefill`) and the wide kernel
(`gguf_q8_0_dense_wide`) did not record, so the census reported "2144 coltile
launches and zero `wmma_prefill` launches" on the promoted default and the
2026-09-17 promotion entry read that as evidence the Q8 dense linear route was
not where its +4.542 s came from. The instrument was blind to the family under
question: the 134 roles it did see were exactly the layers the WMMA route had
not claimed. Both families now record. Removal trigger: move the record call to
the registry launch boundary so coverage is a property of the registry rather
than of which module remembered to call it, and keep the coverage list in the
census docstring until then. Never read an empty census row as "did not run"
while coverage is per-module.

### The row-bucketed iu8 repair pair is measurement surface, not a route

`gguf_q4_k_selected_dual_risk_bitmap_kernel` and
`gguf_q4_k_selected_dual_row_bitmap_repair_bf16_kernel` (with
`risk_bitmap_words` / `risk_bitmap_bytes` and their registry entries) are
bit-identical to the shipped per-slot repair and **22% slower** on the admitted
gate/up shape: 3.5054 ms against 2.8685 ms at the measured 93,342-risk queue,
despite moving 186 MB instead of 612 MB. The repair is bound by memory-level
parallelism, so traffic reduction does not buy time
(`benchmarks/results/2026-09-17-qwen4exp-iu8-repair-cost-structure/`). They stay
because the packet that records the dead end uses them, and because the bitmap
pass is the natural input to a pipelined repair. Removal trigger: no packet or
test refers to them, or a pipelined/async-copy repair lands that supersedes the
whole family.

### The 32-row MoE M-tile variants are registered but unused

`selected_wmma_iu8_risk_j32_prefill_bf16_bf16_out` for both `gguf_q4_k`
(gate/up) and `gguf_q5_1` (down) instantiate the routed-MoE main kernels at a
32-row M tile: an 8-warp block covering 32 rows x 128 columns as two row groups
x four column groups, so each lane still owns 16 rows and the register footprint
matches the 16-row owner (VGPR 208, scratch 0). They are bit-identical to the
16-row owners and cut the *modeled* expert-weight reads 1.67-1.82x, but they are
**1.3-2.2x slower** because the tile's shared read misses cache: measured DRAM
read traffic falls 6.3% instead of the modeled 45%
(`benchmarks/results/2026-09-22-moe-j32-tile/`). Nothing dispatches to them; only
the cost packet and the RED test call them.

They stay because they are the substrate for the design that would actually bank
the traffic: stage the K chunk's quantized weights in LDS once per block and
have all 8 warps read them from shared memory, instead of each lane gathering
its own column's blocks from global. That removes the redundant read instead of
hoping for cache reuse, and it is the same change that would let a taller tile
pay at all.

Size the staging from the quant, not from a round number. A 128-column x 256-K
chunk is **24 KB** for Q5_1 down (256/32 x 24 B x 128) and **18 KB** for Q4_K
gate/up (one 144-byte superblock per column x 128), not 32 KB. With the
activation staging the kernel already holds (32,384 B at `TILE_ROWS=32`:
`lds_q8a` 26,112 + `lds_da` 3,072 + `lds_sumq8a` 3,072 + `lds_row_atrisk` 128),
a 256-K Q5_1 chunk totals 56,960 B of the 65,536 B gfx11 workgroup budget - it
fits with 8.4 KB spare. A 128-K chunk halves the weight tile to 12,288 B for
20.4 KB of headroom, enough to double-buffer the staging against the MMA loop.

This path is **bit-exact** and does not need an arithmetic change: the staged
bytes are the same Q5_1 codes read in the same order into the same iu8 WMMA and
the same Kahan/at-risk repair, and only the address the lane reads from changes.
The GR family's closure to arithmetic change therefore does not apply to it.
Do not port MMB's BF16 operands along with it: `LLAMA_MMB_BF16W=1` is the part
of the comparator's win we deliberately do not gate. See
`benchmarks/results/2026-09-27-moe-down-lds-weight-staging/`.

Removal trigger: the LDS weight-staging variant lands and passes the same
bit-identity gate, or no packet or test refers to the 32-row entry points.

## Full-attention context leaf: superseded fixed256 route

`qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans` and its
`..._fixed256_threads_spans` sibling predate the promoted route
(`..._fixed256_promoted_spans`) that `qwen4_exp_runner` now calls for the dense
full-attention chunk. The promoted route differs by two byte-identical levers -
an int32 per-token offset table when the KV capacity proves the range fits, and a
two-token QK unroll - and by taking the KV capacity as an explicit argument.

The superseded symbols stay until nothing needs them: the microbench uses
`..._fixed256_spans` as the bit-exactness reference for the promoted route, and
`..._fixed256_threads_spans` still carries the runtime thread-width sweep. The
generic `..._batch_spans` route remains the default for the non-fixed256 case.

Removal trigger: the promoted route has held as the default across a full
category suite and the runtime thread-width sweep is retired, at which point the
fixed256 pair, its `__init__` exports, and the microbench reference path go away.

## Full-attention context leaf: QK unroll depth selector

`hipengine_qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_experiment_spans`
(`hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip`) selects a QK
unroll depth by an integer `variant` argument, so the microbench can compare
depths in one session without adding one exported symbol per depth:

| variant | QK tokens | addressing |
| --- | ---: | --- |
| 0 | 2 | per-token int32 offset table |
| 1 | 4 | per-token int32 offset table |
| 2 | 4 | per-token int64 offset table |

Nothing in `hipengine/` dispatches to it; only
`scripts/qwen4exp_fullattn_prefill_microbench.py` and the worklog entries for this
campaign call it.

Removal trigger: the unroll depth stops being a live tuning axis - for example
once the leaf's per-token offset table itself is replaced - at which point this
symbol, its Python wrapper, its `__init__` export, and the microbench
`--qk-variants` path all go away.

## Full-attention context leaf: rejected per-block base table

Do not re-attempt compressing the per-token offset table. Replacing it with one
int64 element offset per logical block, rebuilt in-loop from an incrementally
tracked (block, offset-in-block) pair, cuts the shared-memory request from 18,471
to 10,423 bytes and doubles residency from 3 to 6 blocks per CU, and measured
**1.41x slower** (40.38 ms against 28.59 ms) on the p4096 prefill launch.

The reason is that the table is not overhead but a precomputed address stream:
reading the next address from shared memory lets the hardware issue independent
loads, whereas the incremental tracker puts the address on a serial recurrence
every load depends on. The same effect explains the int64 variant's 1.38x
regression at only one fewer block per CU.

An earlier revision of this ledger listed this as the next lever to implement; it
was implemented, measured, and rejected. The kernel now carries a comment at the
precompute loop recording why the table is per-token. Full measurements are in
the campaign worklog entry and `benchmarks/results/`.

## GR up-projection: unregistered LDS-staged f32 prototype

`hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_gr_staged_f32.{hip,py}` is a
screen prototype, not a production variant. It is not registered, not reachable
from the engine, and its only consumer is the `staged_f32` arm in
`scripts/qwen4exp_gr_projection_screen.py` plus
`tests/test_gpu_qwen4exp_gr_staged_f32.py`.

It exists because the ledger recorded the `dense_wide256` tile's 4.59x speed and
its numerical-gate failure as one fact, which made the weight-traffic mechanism
look unusable at production precision. The prototype separates them: 2.388x on
the up projection at a relative-L2 deviation of 7.442e-08 from the exact kernel,
against the wide tile's 4.709e-05.

Removal or promotion trigger, whichever comes first:

- **Promote** by registering it as the hyper_connection up-projection variant
  and clearing the `docs/EXECUTION-PROFILES.md` production gate, which requires
  a registered strict fallback because the prototype is not bit-exact with the
  parent. At that point the screen arm becomes a gate test and this entry's
  "unregistered" claim stops being true.
- **Remove** if the production gate fails, or if the remaining 1.92x is
  collected by a different mechanism. Delete the `.hip`/`.py` pair, the
  `staged_f32` arms in the screen, and the GPU test; archive the evidence README
  as the rejection record.

Do not leave it in the tree as a third permanent GR route. Two things would make
it actively misleading if left alone: it loses on the down projection (0.859x,
an occupancy defect at 40 blocks for 40 CUs, not a mechanism failure), and its
`BN=128` configuration is measurably worse (3.038 ms against 2.737 ms) because
the 33 KB tile drops residency to one block per CU. Both are recorded in the
kernel header so a future reader does not re-run them.

## GR no-gate variant: retained only as a bit-identity oracle

`hipengine_gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_no_gate_f32`
(the `WRITE_GATE=false` instantiation of
`q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32_kernel`) is **not** a
candidate and must not be adopted as the GR route. It was built to price the dead
gate write named in ledger HC-3, and it measured **2% slower** than the
gate-writing production kernel (6.6551 ms against 6.5198 ms at rows=1024, K=320,
N=10240, non-overlapping samples) while producing a bit-identical `mixed`.

It is kept for one reason: it is the instrument that proves the gate store is
purely additive, which is what makes that timing comparison a store-cost
comparison rather than a comparison of two different computations.
`tests/test_gpu_qwen4exp_gr_no_gate.py` asserts the uint32-view equality of
`mixed` between the two variants at four shapes.

Removal trigger: delete the `WRITE_GATE` template parameter, the export, the
Python wrapper, the screen's `fused_no_gate` arm, and the GPU test once either
(a) something else establishes that the gate store is additive, or (b) the fused
GR kernel itself is replaced — for example by the staged-f32 path in the entry
above, which does not write a gate tensor at all. Do not leave a slower variant
registered as an option; nothing selects it today, and nothing should.

## Staged-f32 dense Q8_0 route: registered, off by default, owes a gate

`staged_f32_bm128_bn64_bk32` is registered for
`(hip_gfx1151, linear, gguf_q8_0)` and selected only when
`HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS` names a layer scope. It is a
production-gate candidate, not a default.

Measured 1.23× to 2.22× faster than the exact coltile leaf across all six Q8_0
dense projections of a layer step (`attn_qkv`, `attn_gate`, `ssm_out`,
`ffn_{gate,up,down}_shexp`), best variant in 30/30 (layer, tensor) cases, with
relative L2 4.6e-07 to 9.1e-07 — 4–8 f32 ulp, against the f16 WMMA sibling
route's 4.7e-05. Evidence:
`benchmarks/results/2026-09-29-dp8-staged-f32-dense-family/`.

It is **not bit-identical** to the leaf it replaces, so it is a T1/T2 arithmetic
change under `docs/EXECUTION-PROFILES.md` and cannot be promoted on strict
parity. Two exits:

- **Promote** by passing the calibrated production gate (mean/p95/p99/max KL,
  top-1 by category, determinism, isolation, task quality) at the declared layer
  scope, then making that scope the default. The exact coltile leaf stays
  registered as the strict fallback either way.
- **Remove** if the gate fails: delete
  `hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_gr_staged_f32.{hip,py}`, the
  registration block and `STAGED_F32_VARIANT` in
  `hipengine/kernels/hip_gfx1151/__init__.py`, `_Q8_STAGED_F32_LAYERS_ENV`,
  `_q8_staged_f32_layers`, `_Q8_STAGED_F32_VARIANT`,
  `_raw_k_staged_f32_dispatch` and its call site in
  `hipengine/runtime/gguf_linear.py`, and
  `tests/test_unit_qwen4exp_staged_f32_route.py`. Keep the evidence README as
  the rejection record.

Two things to know before touching this:

- **Unset means off, and `=0` means layer 0, not off.** Layer 0 is a real index
  and a one-layer scope is legitimate, so there is no in-band way to spell both.
  To disable, unset the variable or set it to the empty string. Pinned by
  `test_layer_zero_is_layer_zero_not_off`.
- **The variant name exists in two places.** `STAGED_F32_VARIANT` in the backend
  package and `_Q8_STAGED_F32_VARIANT` in `gguf_linear.py`, kept separate so
  `hipengine.runtime` does not import a backend package eagerly. If they drift,
  the dispatch silently no-ops and the route looks enabled while doing nothing.
  `test_variant_name_agrees_between_runtime_and_backend_package` is the check.

Do not leave this route on by default without the gate. The whole reason it is
registered-but-off is that its arithmetic differs from the leaf it replaces.

### Gate gotcha: `--layers` takes ranges, the env selectors do not

`--layers 16-47` works because the gate runs it through `_parse_layers`, which
expands ranges into a comma-joined list before substituting it into a route's
`{layers}` placeholder. The **selectors themselves** split on commas and call
`int`, so a route that hardcodes a scope in its own env mapping must write the
comma form. Writing `WMMA_LAYERS_ENV: "16-47"` fails at the first prefill with
`ValueError: invalid literal for int() with base 10: '16-47'`, after the model
has loaded.

`WMMA_CERTIFIED_SCOPE` exists so the `staged_f32_over_wmma` route does not
repeat that mistake. If you add a route with a fixed scope, use the same form.

### Why the composed route exists at all

`staged_f32` clears WMMA in **both** arms, so its numbers certify "staged on the
named scope with no WMMA anywhere". That is not production: WMMA owns 16-47, so
a staged scope of 13-15 would only ever replace the exact chain on 13-15.
`staged_f32_over_wmma` uses **current production** (WMMA@16-47) as its teacher,
so its measured drift is the marginal cost of the staged tile rather than a
composition that does not exist. Do not reuse the plain route's numbers for the
composed question - adding the two drifts is a summed-KL budget, and this
packet's own ladders show scope contributions are neither additive nor
monotonic (0-7's mean exceeds 0-47's; 15-47's max exceeds both 14-47's and
16-47's).


## Dispatch memoization key: three routes, one defect class, two occurrences

`launch_gguf_linear` only reaches the dispatch chain inside `if cached is None:`.
Any route whose **decision inputs** are absent from that key can be replayed
across configurations that should differ. Two occurrences have been measured, and
both failed *silently* - the symptom is a KL of exactly zero and a green status,
not a wrong answer:

- **staged-f32** (`_staged_f32_key`, fixed). A gate's two arms are the same
  shapes on the same weight pointers and differ only in the selector, so the
  teacher arm replayed the candidate's exact-chain dispatch.
- **f16 WMMA** (`_wmma_scope_key`, fixed). `use_wmma` is in the key as a boolean
  but the layer scope is applied through a context-local weight filter, which the
  key cannot see. The certified 16-47 configuration run at HEAD returned
  bit-identical candidate/teacher hashes with `measurement_valid: true`.

**Check for this whenever you add or touch a layer-scoped route.** The key must
carry the enable flag, the scope, **and** the owning layer index - the last
because otherwise the first resolution for a shape is replayed for every layer
and the scope becomes an artifact of cache-fill order.

Two consequences worth knowing:

- A test that observes cache *misses* cannot detect the omission. The registry
  `generation()` is the first key element and lazy kernel registration inside the
  call bumps it, so every call misses for an unrelated reason. Record the key
  tuple instead; both `tests/test_unit_qwen4exp_staged_f32_route.py` and
  `tests/test_unit_qwen4exp_wmma_route_cache_key.py` do this.
- **A gate's green status does not prove its route ran.** `measurement_valid`
  and `qualification_coverage()` verify the protocol, not engagement. Check that
  the candidate and teacher logits hashes differ before reading any KL; an
  identical pair is a failed run regardless of the summary statistics.

## `HIPENGINE_GGUF_PACK8_GEMV_WIDE` (DE-3 narrow-projection block size)

- Added 2026-09-30; **default ON since 2026-09-30** after the production numerical
  envelope passed. The flag is now an opt-out (`=0`) retained for rollback and
  bisection, not an opt-in. The decode-shaped pack8 GEMV launches
  `out_features / 8` workgroups, so a narrow projection cannot fill the device: measured on gfx1100, out=320
  runs 40 workgroups at 92.4 GB/s (42% of the 221 GB/s roof) and out=640 runs 80
  at 121.5 GB/s (55%), while every shape at 2560 or more output features is
  already at 90.8-102.9%. Raising the block from 128 to 256 at the starved shapes
  is worth -32.4% and -19.4% of kernel time respectively (+1.490 ms/step over the
  three narrow shapes) and is neutral within 2%, mixed in sign, above the knee.
  Artifact: `benchmarks/results/2026-09-30-qwen4exp-de3-threads-lever/`.
- **This changes arithmetic.** 256 threads per 8-output workgroup means 32
  threads per output rather than 16, so the K-reduction tree differs; measured
  divergence is max relative 2.6e-7.
- **Envelope PASSED on 2026-09-30** via
  `scripts/qwen4exp_layer2_profile_gate.py --candidate pack8_gemv_wide_256` on
  Qwen3.8-Flash-Next-UD-Q4_K_XL: `kl_mean` 0.000287 (bound 0.001), `kl_p99`
  0.004831 (0.02), `kl_max` 0.018131 (0.05), top-1 0.99778 (0.99), all four
  categories above their 0.97 per-scope floor, three repeat runs bit-identical,
  clean lifecycle. The `prefill_last` scope scores `kl_mean = 6.9e-14`, so
  prefill is untouched and the lever is confined to decode. Artifact:
  `benchmarks/results/2026-09-30-qwen4exp-de3-threads-lever/kl-envelope-gate-qwen4exp.json`.
- Removal condition: once the wide block has survived a release cycle without a
  rollback, delete the env flag, `_pack8_gemv_wide_threads_enabled`, and the
  falsy-spelling tests, and keep only the shape-boundary cases in
  `tests/test_unit_gguf_pack8_gemv_wide_threads.py`.
- Only `gguf_q8_0` was measured end to end. The rule is deliberately gated on
  grid size rather than on the quantization scheme because the reason is
  occupancy, so the other quants in the decode-shaped pack8 family inherit it;
  that inheritance is still reasoned rather than measured.
- **Strict-exact opt-out scope.** The lever applies to one-row launches only, so
  a caller whose contract is that an N-row block equals N one-row forwards bit
  for bit cannot inherit it. Gemma 4's runner enters
  `pack8_gemv_exact_threads_session(True)` around its block and packed-prefill
  forwards (`hipengine/runtime/gemma4.py`); `_pack8_gemv_decode_threads`
  returns the parent block size inside that scope. Without it,
  `tests/test_gpu_gemma4_packed_batch_forward.py` and
  `tests/test_gpu_gemma4_runner_verify_rows.py` fail. Removal condition: a
  Gemma 4 profile gate that admits the wide block on both sides of the
  comparison (or a registered wide variant the strict route can name), after
  which the scope and `_pack8_gemv_exact_threads` go away.

## `HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET` (RT-4) — promoted to default

- Added 2026-09-30, **promoted to default the same day**. When enabled (now the
  default), `Qwen4ExpGGUFResidentModelRunner.reset()` clears only
  `[0, written_cache_bytes())` of each K/V cache instead of the full allocated
  capacity. Measured at 128K capacity: reset 15.689/15.519 ms -> 1.133/1.741 ms,
  **14.17 ms saved per request**, decode identical in both regimes. Artifact:
  `benchmarks/results/2026-09-30-qwen4exp-rt4-bounded-kv-reset/`.
- **Exactness proven by parent parity, not asserted.** `scripts/qwen4exp_layer2_profile_gate.py`
  run twice, `production_baseline` (no flag) and `bounded_kv_reset` (flag on),
  both under the production profile: **identical `candidate_logits_sha256`**
  (`260979d6...`). Both runs also produce an **identical `strict_logits_sha256`**,
  which is the control showing the harness is reproducible, so the match is
  evidence rather than coincidence. Independently, the raw K/V buffers are
  **byte-identical at all 12 attention states** across three arms (flag off, off,
  on), and the invariant `last_nonzero < bound` holds at every state
  (1,064,959 vs 1,310,720).
- **Note on the gate comparison:** an earlier single run compared the candidate
  against the **strict** arm and showed differing logits hashes. That is the
  gate's intended strict-vs-production difference, not a property of this flag -
  which is why the parent-parity pair above, not the strict comparison, is the
  binding evidence for a T1 candidate.
- Removal condition: once the bounded path has survived a release cycle without a
  rollback, delete the env flag and `_qwen4_exp_bounded_kv_reset_enabled`, keep
  `tests/test_gpu_qwen4exp_kv_reset_bounded.py` as the permanent contract for the
  bound, and drop the flag-contract test.
- Not exercised by any test: rollback, cancellation and c2. The bound is
  re-derived from `position` on each prefill, which is an argument rather than a
  measurement.
- **Diagnostic script gotcha:** `scripts/qwen4exp_rt4_kv_compare.py` initially
  read buffers with `copy_device_to_host` before `device_synchronize()`, which
  captured partially-written state and produced a phantom arm difference. Any
  probe that reads device buffers must sync **before** the copy.

## `scripts/execution_profile_gguf_c1_pack8_wide_gate.py` (superseded harness)

- Added 2026-09-30 (`e5f43ce21`) to measure this lever's KL envelope, and
  superseded the same day by `scripts/qwen4exp_layer2_profile_gate.py`.
- The adapted gate drove the Qwen3.6-35B-A3B model, whose decode never dispatches
  the `pack8_gemv_*` family at all. Its run therefore produced bit-identical
  arms (`strict_and_candidate_bytes_equal` on all 18 prompts) and a vacuously
  green quality status. See `scripts/qwen4exp_de3_route_probe.py` for the
  measurement that established this.
- Removal condition: delete the file once no other worker is using it as a
  generic c1 threads gate; the Qwen4Exp gate covers this lever's requirement.

## Route-engagement recorder coverage (`wmma_prefill_dispatch_recorder`)

- Found 2026-09-30 while re-gating the live default at 16-47
  (`benchmarks/results/2026-09-30-qwen4exp-rt1-chunk4096-gate/DENSE-WIDE-REGATE-AT-HEAD.md`).
- `hipengine/runtime/gguf_linear.py`'s `_record_wmma_prefill_dispatch` records
  only the `wmma_prefill` selector, and `_record_selector` records the collateral
  selectors `use_wmma` moves (`q4k_rowtile`, `native_batch_decode_session`, …).
  **The `dense_wide` route has no recorder.**
- Consequence: the engagement check added in `056e6fbd8` - the mechanism that
  caught the WMMA-route false pass - **cannot see the route that is the live
  production default at 16-47** (`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`). A
  `--route dense_wide` gate reports `candidate_route_engaged: False` regardless
  of whether the route ran, because the selector never appears in
  `selector_states`. The gate has a row floor for it
  (`ROUTE_ROW_FLOORS["dense_wide"]`) but no dispatch recorder, so the floor is
  never fed.
- The detector covers the *retired* route and is blind to the live one. Any
  future `dense_wide` certificate is unverifiable on route engagement for the
  same reason the pre-`056e6fbd8` certificates are.
- **CLOSED 2026-09-30 (`20260930T113806`).** Both defects are fixed:
  `_record_route_engagement` now records the route at all six decision points of
  `_q8_dense_wide_dispatch`, and `_summarise_engagement` takes a `route_selector`
  from `ROUTE_ENGAGEMENT_SELECTORS` instead of hardcoding `wmma_prefill`. Only
  the first was in the original entry; fixing it alone would have left the flags
  wrong, because the gate read the wrong key.
- Verified end-to-end on the reduced screen: `--route dense_wide --layers 16-47`
  now reports `candidate_route_engaged: True`, `route: dense_wide`,
  `selectors_differing: ['dense_wide']`, with `dense_wide` at t=0.0000 c=0.2640
  differs=True. The `wmma` route is unchanged.
- **Residual, not closed:** the canonical `dense_wide` 16-47 re-gate of
  `20260930T113116` predates this fix, so that artifact still cannot say whether
  the route ran. Repeat it to give the live default engagement evidence.
- **Residual, not closed:** a route with no entry in `ROUTE_ENGAGEMENT_SELECTORS`
  (`staged_f32`, `staged_f32_composed_vs_exact`, `staged_f32_over_wmma`) now
  reports `route_engagement_recorded: false` rather than a wrong answer, but it
  still has no recorder. Add one before relying on any of those gates for
  engagement.

## Qwen4Exp chunked WY GDN prefill screen arm (GD-4)

- `qwen4_exp_gdn_prefill_wy_f32_kernel` / `hipengine_qwen4_exp_gdn_prefill_wy_f32`
  and the screen's `wy_chunked` arm are the FP32 chunked triangular /
  Woodbury-Young candidate GD-4 proposed. The row is closed: the build is
  parent-parity correct but 1.15-1.26x slower than the `wave_norm` route the
  UD-Q4_K_XL lane dispatches at 1K/2K, and 1.75-2.26x slower than the registered
  `tiled16_multi` route (85 VGPR) that already escapes the 256-VGPR register
  bound the row cited. Evidence:
  `benchmarks/results/2026-10-01-qwen4exp-gd4-wy-kernel-built/`.
- The kernel is **not registered** and no profile selects it; it is retained
  only so `scripts/qwen4exp_gd3_recurrence_screen.py` can reproduce the
  comparison.
- Removal condition: delete the kernel, its `extern "C"` wrapper, the host
  wrapper and the `wy_chunked` screen arm once no ledger row or artifact
  references the screen. The open follow-up from this row - admitting
  `tiled16_multi` on the UD lane - is a profile/gate task and does not need
  this arm.

## The gfx1151 arch chunk profile is geometry-blind (2026-10-03)

- The H2048-MoE chunk *value* is settled: the geometry policy is now
  `(1024, 1024)`, bit-exact against the 512 it replaced and +1.4% prefill at
  4K. See
  `worklog/entries/20261003T100200.274738Z-lhl-gfx1151-h2048moe-prefill-chunk-1024-promoted-d001da`.
- What remains is the arch layer. `_ARCH_CHUNK_PROFILES["gfx1151"]` is
  `(256, 256)` and is applied without regard to model geometry:
  - H5120 dense: right, 398.6 vs 387.7 tok/s at 1024;
  - H2048-MoE: wrong, 1363.7 tok/s, the worst of five values measured.
- The H2048-MoE geometry entry only shields the GGUF lane from that profile. The
  PARO 35B-A3B lane has the same geometry and is likely running 256.
- Removal condition: key the arch profile by geometry, or fold the measured
  per-geometry values into the geometry layer and drop the arch layer for
  gfx1151, **after** measuring the PARO H2048-MoE path. Then delete this entry.

## Direct `Qwen35GGUFResidentSession` construction is not the shipping AR route (2026-10-03)

- `use_wmma_prefill` and `use_gemv_decode` default to `False` at the session
  (`_env_wmma_prefill_enabled()` / `_env_gemv_decode_enabled()`), so a direct
  construction gets a 1.62x slower decode than the shipping route (4K/128:
  28.831 vs 17.806 ms/step, GPU busy 72.8% vs 99%). The generation path passes
  `use_gemv_decode=True` at both shipping call sites, so `hipengine.LLM.generate()`
  is unaffected; the harness warns via `shipping_ar_route_mismatch`.
- The conservative default is deliberate (`resolve_qwen35moe_fastpath_safety`
  blocks the P9.E2-rejected raw-GGUF WMMA+GEMV combination), so this is a
  documentation and diagnosability item, not a defect: any direct-session
  profiler, benchmark, or downstream caller must pass the two kwargs explicitly
  or it will measure and ship a non-shipping route.
- Removal condition: keep the conservative resolution, but make the effective
  route visible on the session (it already is, via `fastpath_safety`) and state
  the required kwargs in the resident-session docstring.

## Selected Q4_K T16 gate+up tile8 owner (2026-10-03)

- Added `HIPENGINE_GGUF_SELECTED_Q4_GATE_UP_TILE8` (default on) in
  `qwen35_gguf_runner._launch_selected_raw_gguf_moe_pair_silu`. It routes the
  c1 selected Q4_K gate+up pair to
  `gguf_q4_k_t16_selected_dual_silu_tile8_gemv_bf16_bf16_out` instead of the
  16-column `gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out`. Setting the
  variable to `0` restores the 16-column owner.
- The two owners are byte-identical: both launch 128 threads, so thread K
  ownership (`k = tid`, `k = tid + 128`), the per-column
  `d * scale * q - dmin * minv` expression, the 32-lane butterfly wave
  reduction, the 4-sub-group cross-wave sum order, the BF16 round-trip, and the
  fused SiLU are unchanged. The tile8 owner only covers 8 columns per
  workgroup instead of 16. rocprofv3 at the production 2048x512x8 shape records
  VGPR 200 -> 88, LDS 1024 -> 512, workgroups 32 -> 64.
- The selector is read through `_env_flag`, which is not memoized, so the flag
  costs one `os.environ.get` plus a strip and a set lookup per layer per step
  (40/step on this model). The measured A/B already includes that cost and is
  net positive, and the existing repo convention is per-call reads, so it is not
  hoisted here.
- Removal condition: keep the `=0` rollback through one decode campaign
  checkpoint for bisection, then drop the selector, hoist the (now constant)
  owner choice out of the per-layer path, and remove the 16-column wrapper once
  the tile8 owner is the only qualified gfx1151 owner.

## Q8 dense prefill: the DPP wave reduction and the row-first grid order

- `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip` carries two boolean
  template parameters on the pipelined Q8 prefill kernel, `DPP_REDUCTION` and
  `ROW_FIRST`, both defaulting to `false` so that every pre-existing
  instantiation and exported symbol is unchanged. `DPP_REDUCTION` is the retained
  win (bit-identical, mean -1.64% across five real layers); `ROW_FIRST` is the
  measured negative (free and bit-identical, 2.1-2.9% slower on all five layers).
- **`ROW_FIRST` should be deleted rather than kept as an experiment.** It exists
  only to produce the one measurement that refutes the traffic reading of this
  kernel family, and that measurement is now recorded in three places
  (`benchmarks/results/2026-10-04-qwen4exp-q8-prefill-dpp-reduction/README.md`,
  `benchmarks/README.md`, `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md` §7A). Its arm in
  `scripts/qwen4exp_dense_projection_ab.py` and its two test cases in
  `tests/test_gpu_qwen4exp_q8_dpp_reduction.py` exist for the same single purpose.
  Do not re-run it expecting a different answer; a locality change cannot win here.
- `HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION` (default on, `=0` restores the pipelined
  leaf) follows the `HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE` precedent: the flag is in
  the dispatch memoization key so a flip inside one process cannot replay the
  other arm's leaf, and the rewrite is `is_registered`-guarded so the chain
  degrades DPP -> pipelined -> wave-scale parent -> unscaled exact parent.
- Removal condition: keep the `=0` rollback through one prefill campaign
  checkpoint for bisection, then make `DPP_REDUCTION` unconditional in the
  pipelined leaf, delete the `ROW_FIRST` parameter and its arm/test cases, and
  drop the flag once the DPP leaf is the only qualified gfx1151 owner for this
  tile shape.

## GR-up ordered DPP candidate

- `HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION` is enabled for the measured
  `gguf_ud_q4_k_xl` production profile after the tracked-clean canonical
  twelve-case A/B. It replaces only the existing rows>256, four-branch
  wave-scale GR-up reduction. Missing registration falls back to the
  wave-scale parent, then the unscaled fused parent; the unfused primitive
  chain stays available.
- The real-weight operation screen is bit-identical and 1.54-1.57x faster
  on gfx1151 at rows 512/1024/4096. The tracked-clean engine A/B then measured
  weighted prefill gains of 6.57%, 7.15%, and 6.98% at 512/1024/4096 tokens,
  with every canonical category winning and all 72 output trajectories exact.
  Decode was effectively flat and is not claimed. The UD production manifest
  names the DPP variant and retains the exact fused parent as its fallback.
- Removal condition: retain `=0` for one campaign checkpoint for bisection,
  then remove the selector once the DPP leaf is the only qualified gfx1151
  owner for this tile shape.
- Evidence: `benchmarks/results/2026-10-05-qwen4exp-gr-dpp-reduction/`.

## Exact-iu8 risk multiplier: pinned at 4.0 in production, and why

- `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` is now **explicitly bound to `"4.0"`** in
  `hipengine/generation/qwen4_exp_profiles.py`, which is the value the code default
  already had. The binding is a pin, not a change: it stops a future edit to the
  default in `_qwen4_exp_q4_iu8_risk_multiplier()` from silently moving the
  production profile, and it puts the measured alternative at the binding site.
- **The alternative was measured, gated and deliberately not taken.** 1.0 cuts the
  repaired share of rows 3.40x and passes the production numerical gate with
  3.3-6.4x margin, but its wall benefit does not survive the canonical protocol
  (isolating harness -2.11%, canonical interleaved A/B/A flat with the 4.0 arm's
  clean repetitions slightly ahead). Evidence and both halves of the reading:
  `benchmarks/results/2026-10-04-qwen4exp-q4-iu8-risk-mult-1/README.md`.
- **The gate candidate `q4_iu8_risk_mult_1` stays in
  `scripts/qwen4exp_layer2_profile_gate.py`**, but note that its recorded artifact
  was produced while production was still 4.0, so its teacher arm is the pre-change
  default. Re-gating 1.0 or 0.5 later needs production bound back to 4.0 or an
  explicitly different teacher.
- **The real lever this exposed is the repair kernel's launch geometry.** The wall
  falls 2.11% where a proportional model over the recorded 1370 ms share predicts
  5.7%, because the pass launches over `risk_capacity = compact * 2 * ffn`
  regardless of `risk_count`. Making it proportional to the queued count is worth
  more than the multiplier and is orthogonal to the numerics question. This is the
  entry to act on; the multiplier is parked behind it.
- **The first measurement the launch-geometry unit should make, before building
  anything.** `gguf_q4_k_selected_dual_sparse_exact_repair_bf16` takes
  `grid_blocks` defaulting to **1024** (`gguf_q4_k_selected_prefill.py:600,1161`)
  and the runner never passes it, so every call launches 1024 blocks whatever the
  risk count is. The kernel's loop is `for (slot = blockIdx.x; slot < bounded; slot
  += gridDim.x)` with `bounded = min(risk_count, max_risks)`, so the *work* is
  already proportional - 46.6k slots per call at 4.0 against 13.7k at 1.0, ~46
  against ~13 iterations per block. A 3.4x cut in inner work showing up as 2.11% of
  wall is the anomaly to explain, and the cheapest explanations to distinguish are:
  (a) **the 1370 ms share is inflated** - it is from a 2026-09-16 capture whose own
  README says its per-kernel times are diagnostic, and if the true share is nearer
  500 ms then 359 ms saved is most of it and there is no anomaly; (b) the 4.0-only
  extra rows are not uniformly spread across calls, so the aggregate ratio
  overstates the wall effect; (c) empty-block cost, which should be nearly free on a
  GPU but is a one-line experiment. **Measure (a) first** by profiling the repair
  kernel with `--kernel-trace` at both multipliers rather than inferring from the old
  capture - it is the explanation that would make this whole lever smaller than the
  1370 ms figure promises, so it decides whether the unit is worth building.
- Removal condition for the explicit pin: once the repair pass is proportional to
  `risk_count`, re-measure the multiplier on the canonical protocol in the same
  session as the change, and either bind the winner or delete the env var and make
  the chosen bound a named constant (keeping the strict path's 4.0 as a constant
  rather than an environment string).

## The host/device loop split in `qwen4exp_decode_host_bound_probe.py`

`scripts/qwen4exp_decode_host_bound_probe.py` reports `host_bound: true` and a
`device_drain_ms` of `0.0` from splitting the decode loop at its trailing
synchronize. That inference is invalid: `runner.step()` ends with a blocking
device-to-host copy of the argmax result (`_finish_lm_head` ->
`copy_device_to_host`), so the "host enqueue" window already contains the device
wait and a zero drain is guaranteed. `qwen4exp_canonical_ar_bench.py` carries
the same split as `decode_host_enqueue_ms` / `decode_device_drain_ms`, but its
code comment now states the limitation and its artifact protocol records the
capture mode, so a reader can tell a `compact` row (no explicit per-step
synchronize) from a `full` one.

Removal condition: give the probe a mode that drops the readback before it
claims a split — that makes the drain meaningful and it is then non-zero (0.343
ms/step at p512, 0.575 at p4096). Drop the `host_bound` field, or rename both
fields so they cannot be read as an attribution. The counter-rotated same-state
A/B that replaces the probe's conclusion lives in
`benchmarks/results/2026-10-04-qwen4exp-decode-readback-split/`.

## The per-launch `build_X(load=True)` call

Every kernel-launch helper resolves its library per launch - `library = library or
build_gguf_k_gemv(load=True)` and its siblings - which is **957 calls in one
`code-p4096` decode step**, measured by the canonical bench's `--host-profile`.
Each call re-derives a resolution identity from **eight** environment variables
(2.281 us micro-benchmarked), so the term is ~2.49 ms/step even after the memo in
`d0f...` removed the key derivation.

**Identity component, attempted 2026-10-08, reverted:** reading the identity from
the live raw environment mapping measured 173 ns/call against 2,292 ns for the
str path (13x cheaper in isolation, ~1.95 ms/step projected at the 980
resolutions/step decode rate) with live-change observation preserved, but the
same-residency decode A/B showed only +0.12% at p512 and an unexplained -2.7%
at p4096 on a heat-soaked host (tight, order-balanced samples; zero memo
misses, identical libraries and launch counts - no mechanism identified).
Reverted pending a cooled paired re-measurement; evidence in
`benchmarks/results/2026-10-08-qwen4exp-build-identity-fast-read/`. The
resolution-key construction and the library-handle path remain; the library
threading below is still open. **Cooled re-measurement (2026-10-08, later):** a
fresh-process single-case p4096 A/B on a fully cooled host tied the arms
(+0.019%) - the hot-host inversion was thermal non-stationarity, and the 13x
isolated saving moves nothing at engine level. The per-launch Python enqueue
cost is NOT the decode-depth remainder; the micro-cut is closed as an
engine-level lever and the remainder work must target launch-queue gaps,
per-step transfers, or step-boundary synchronization instead.

**Do not expect wall movement from removing it.** Two ablations of ~2 ms/step of
host work at p4096 measured +2.53% and -0.45% against base spreads of 0.73% and
0.15%, and the host CPU is flat with context (70.816 -> 71.478 us/step for a
fixed workload while the wall grows 37.5%). The wall's growth is device-side.

Removal condition: thread the resolved library through the launch sites (the
`library` / `libraries` parameters already exist) so the launch path resolves
once per process rather than once per launch, then delete the per-launch
`build_X(load=True)` fallback. The build-affecting environment contract lives in
`build_hip` and is unchanged by that; decide explicitly whether the *launch*
path must also observe a mid-process environment change, since that is the only
reason the per-launch call exists.

## The checked full-rows DPP opt-out flag

`HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS` (default on) is the dedicated
rollback/bisection opt-out for the checked-int32/full-four-row dense Q8 DPP
sibling promoted to the default leaf for its qualified shape window (the six
declared dense Q8_0 geometries, 512-4096 complete rows) in
`_raw_k_coltile_pipelined_dispatch`. The generic
`HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION=0` already removes the whole DPP family
from the chain; the dedicated flag exists only to bisect the checked sibling
against the plain DPP leaf without touching anything else.

Removal condition: delete the flag (and its `_q8_dpp_checked_full_rows_raw()`
cache-key term) once the checked sibling has held the default through a full
campaign cycle without a rollback, or when the plain DPP leaf it narrows is
itself retired; keep the `HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION` family flag
until then.

## The T16 down route flag

`HIPENGINE_QWEN4_EXP_T16_DOWN` (default off) is the env gate for the T16
Q5_1 expert-major compensated down route — the down-leg candidate built and
gated on 2026-10-09 (ledger row T16-DOWN; engine A/B +9.46% aggregate
prefill, 12/12 positive; layer2 gate rejected only on the prefill_last
mean-KL floor at 1.53x with top-1 18/18). The dispatch is tile-cache-keyed
with a structural fallback to the production down chain on a miss, so the
default path is untouched.

Removal condition: this flag is the X-7 decision's direct subject. If the
owner admits the prefill_last scope for the compensated-T16 class, promote
the route to the production default and delete the flag (the strict
fallback stays registered). If the owner rejects it, delete the flag, the
runner branch, and the sidecar together once the rejection is final —
keeping the kernel, layout, and their round-trip/quality-gate tests as
reference lineage.

## The T16 residency flags

`HIPENGINE_QWEN4_EXP_T16_RESIDENT` and
`HIPENGINE_QWEN4_EXP_T16_RESIDENT_MAX_LAYERS` (both default off/empty) drive
the binding-owned tile-residency populate at generator construction (the
docs/QWEN4EXP-T16-RESIDENCY.md mechanism: read -> repack -> upload -> attach
as the weight's `t16` allocation stage, raw kept). They exist so the
X-7-admitted full-window phase has its mechanism ready and so the
transition off the module sidecar caches is testable without a model load
of the full tile set.

Removal condition: fold the populate into the load path unconditionally (no
flag) once the X-7 decision lands and either (a) the T16 routes are
promoted — then the binding stage is the production residency and the
sidecar caches themselves move to this same removal list — or (b) the
routes are rejected — then delete the populate, the `t16` stage helpers'
generator call site, and keep the loading-layer primitives only if another
consumer names them. The `MAX_LAYERS` bound exists solely because the
transient dual residency (tiles + raw) may not fit; it dies with the
free_raw phase that makes residency net-zero.

## The QSA ordered-decode fusion flag

`HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED` (default off) selects the
fused QSA coefficients+values v2 sibling — bit-identical to the unfused
pair (2/2 cross-mode exact) and measured wall-neutral on the canonical
decode protocol (0.9997x, inside the same-session AR variance), committed
at `0ea6d7ae2`. The fusion line's verdict is now complete: both rungs
(router select+cast at 96 launches/step, QSA at 12) are bit-identical and
below the decode wall's resolution, so the flag has no measured reason to
turn on.

Removal condition: delete the flag and the fused sibling at the next
cleanup pass unless a whole-step graph (the committed bigger-seam build)
demonstrates the launch-count savings compound at graph scope — in which
case the fused kernel is the graph's building block and the flag becomes
its temporary selector until the graph lands.

## The Qwen4Exp T16 routed-expert routes and tile residency

The `production` plan for `qwen4_exp_gguf` / `hip_gfx1151` /
`gguf_ud_q4_k_xl` runs its routed experts on the wide-tile routed F16 GEMMs
(`quant/qwen4_exp_routed_f16`), which read the raw expert weights. No profile
selects the T16 compensated routes any more: `HIPENGINE_QWEN4_EXP_T16_GATE_UP`,
`HIPENGINE_QWEN4_EXP_T16_DOWN`, `HIPENGINE_QWEN4_EXP_T16_RESIDENT`,
`HIPENGINE_QWEN4_EXP_T16_RESIDENT_MAX_LAYERS`,
`HIPENGINE_QWEN4_EXP_T16_RESIDENT_SLOTS`, the runner's T16 branches and
sidecar caches, `Qwen4ExpGGUFTextGenerator.populate_t16_resident_stages`, and
the `t16` allocation-stage helpers remain only as default-off experiments.
The T16 routes need a second copy of the expert weights (41.3 GiB gate/up,
25.2 GiB down) that does not fit beside the raw weights the decode owners
read, and measured about +8.5% prefill for the down leg against about +48%
for the raw-layout routed GEMMs.

Removal condition: met on 2026-10-13, when the routed F16 GEMMs passed the
production gate and became the production route. Delete all of the above,
keeping the T16 kernels and their round-trip tests only if another model's
plan names them.
`Qwen4ExpGGUFResidentModelRunner.rebind_resident_weights` stays: it is the
general fix for weights republished after construction.

## The Qwen4Exp routed F16 selector

`HIPENGINE_QWEN4_EXP_ROUTED_F16` (default off) is published by the
`gguf_ud_q4_k_xl` production binder and read by `run_qwen4_exp_moe`. It
exists because routed dispatch in this runner is env-selected throughout.

Removal condition: replace it with manifest-driven dispatch when the runner's
routed-expert selection moves off environment variables. The exact grouped
chains stay registered as the strict fallbacks and as the production routes
of `gguf_q4_k_m`; their repair-pass siblings that only the old
`gguf_ud_q4_k_xl` production plan selected can go once no manifest names
them.

## Qwen4Exp unused `moe_f32` layer scratch

Since 2026-10-14 the residual write after the routed experts reads the BF16
MoE output directly (`qwen4_exp_gr_write_bf16_bf16`, bit-identical to
widening it first), so the per-layer-scratch `moe_f32` buffers
(`rows * hidden * 4` bytes in both layer scratch classes of
`hipengine/runtime/qwen4_exp_runner.py`) are allocated but never read.
Removal condition: delete the field, its allocation and its capacity-plan
entry in the next scratch-planner change.

## Qwen4Exp decode routes of the UD production plan

The `gguf_ud_q4_k_xl` production binder selects three decode routes that
change reduction order, and one bit-identical decode fusion is on by
default. All were measured on a Radeon 8060S on the canonical `code-p512` /
`code-p4096` cases (128 decode transitions).

- `HIPENGINE_QWEN4_EXP_GDN_DECODE_ROWSPLIT=1`
  (`qwen4exp_gdn_rowsplit_decode`): one 512-thread block per value head,
  four lanes per column, the state read and written once with both dot
  products in registers. Kernel 66 -> 27.5 us per layer; decode 20.87 ->
  21.47 tok/s at p512.
- `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN=1`
  (`selected_weighted_sum_warp256_fast_bf16_bf16_out`, Q5_1) and
  `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP=1`
  (`selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out`, Q4_K): the
  warp-per-expert routed-expert kernels. Down 124 -> 73 us, gate/up 128 ->
  114 us per layer; decode 21.47 -> 22.65 -> 23.00 tok/s at p512.
- `HIPENGINE_QWEN4_EXP_GR_DECODE_FUSED` (default on, `0` restores the chain):
  the one-row hyper-connection read as three launches instead of six,
  bit-identical (`tests/test_gpu_qwen4exp_gr_read_decode.py`); decode 22.99
  -> 23.43 at p512 and 22.58 -> 23.00 at p4096.

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

Removal conditions: declare the three selected variants in the plan with
strict scopes; delete `HIPENGINE_QWEN4_EXP_GR_DECODE_FUSED` and the
six-launch one-row branch once a release has shipped on the fused read.

## Qwen4Exp direct F16 hyper-connection launches (`HIPENGINE_QWEN4_EXP_GR_F16_DIRECT`)

The `gguf_ud_q4_k_xl` production binder sets
`HIPENGINE_QWEN4_EXP_GR_F16_DIRECT=1`. With it, a hyper-connection read of
more than 256 rows launches its two Q8_0 projections directly instead of
through the generic linear dispatch: the grouped norm publishes its F16 plane
at every layer and row count, the K10240/N320 down projection runs on the
128x128 F16 tile, and the K320/N10240 up projection reads an F16 activation
converted once up front. Both projections produce the same bits as the
dispatch they replace (`tests/test_gpu_qwen4exp_gr_f16_direct.py`), and the
down projection takes the direct launch only where the dispatch would already
select the F16 wide tile. Measured on `code-p4096`, Radeon 8060S, same
process: 1176.4 -> 1224.4 prefill tok/s, identical output tokens.

On the same path the read no longer writes the FP32 normalized plane
(`HIPENGINE_QWEN4_EXP_GR_SCALE_ONLY`, default on, `0` restores the plane):
the grouped norm stores the F16 plane and `inv_rms[rows, branches]`, and the
fused up epilogue and a four-logit inject kernel recompute
`bf16(residual) * inv_rms * weight`, the value the norm stores. `inv_rms`
lives in `scratch.gate`, which that path does not otherwise use, and
`scratch.normalized` is left unwritten. Bit-identical
(`tests/test_gpu_qwen4exp_gr_read_residual.py`); `code-p4096` 1320.5 ->
1382.1 and `code-p512` 850.7 -> 911.4 prefill tok/s, same process, identical
output tokens. The flag goes with `HIPENGINE_QWEN4_EXP_GR_F16_DIRECT`; at that
point `Qwen4ExpGRScratch.normalized` and `.gate` can shrink to the decode row
bound and `inv_rms` gets its own buffer in the scratch planner.

The scale-only read also writes `half(mixed)` into the otherwise unwritten
`scratch.normalized` and declares it through
`hipengine/kernels/hip_gfx1100/linear/f16_mirror.py`
(`HIPENGINE_QWEN4_EXP_GR_F16_MIRROR`, default on, `0` keeps the
per-projection conversions). The F16 wide dispatch and the small-matrix F16
GEMM look the image up by exact pointer and extent and read it instead of
converting `mixed` again; every read and every prefill chunk invalidates the
declaration first. Bit-identical; `code-p4096` 1390.8 -> 1440.1 and
`code-p512` 912.2 -> 951.4 prefill tok/s, same process, identical output
tokens. The image makes the pair loan (`HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR`)
redundant for projections that read `mixed`. Removal with the two flags
above: give the image its own buffer in the scratch planner and delete the
flag.

The scale-only norm also computes the four inject logits in the same launch
when the read has an inject projection
(`grouped_rmsnorm_bf16_f16_scale_inject_kernel`, one block per row,
`HIPENGINE_QWEN4_EXP_GR_NORM_INJECT=0` keeps the separate inject launch).
Bit-identical (`tests/test_gpu_qwen4exp_gr_read_residual.py`); `code-p4096`
1560.4 -> 1587.1 and `code-p512` 1018.4 -> 1024.5 prefill tok/s, same
process, identical output tokens. The flag goes with the others in this
section.

The direct launch makes three older mechanisms unreachable for this quant at
prefill row counts: the publication loan and its policy table
(`GGUF_DENSE_WIDE_F16_PUBLICATION_POLICIES`, layers 13-47, rows 4096 only),
the padded tail leaf (`GGUF_DENSE_WIDE_F16_TAIL_POLICIES`,
`dense_wide256_padded_f16in_f32_f32_out`) and its manifest scope.

Removal condition: once a `gguf_ud_q4_k_xl` release has shipped with the
direct launch, delete the flag, make the direct launch the only path for the
F16 wide down projection, and delete the loan, both policy tables and the
padded tail variant with their tests.

## Qwen4Exp routed gate/up without a grouped activation copy (`HIPENGINE_QWEN4_EXP_ROUTED_F16_INDEXED`)

When the routed F16 expert kernels are selected, the paired gate/up kernel
reads token-major BF16 activations through the grouped lane map
(`row_lanes[r] // top_k`) and the runner launches the metadata-only
`qwen35_moe_group_scatter` instead of `qwen35_moe_group_scatter_gather_lowp`,
so the `rows * top_k * hidden` grouped BF16 copy is not written. Output bits
are those of the gathered copy
(`tests/test_gpu_qwen4exp_routed_f16.py::test_gate_up_reads_token_rows_through_the_lane_map`).
Measured on `code-p4096`, Radeon 8060S, same process: 1309.7 -> 1338.1
prefill tok/s; `code-p512` 841.6 -> 855.5; identical output tokens.

`HIPENGINE_QWEN4_EXP_ROUTED_F16_INDEXED=0` restores the gathered copy for
bisection. The flag defaults on and has no manifest row; it only takes
effect under the routed F16 selection.

Removal condition: once a `gguf_ud_q4_k_xl` release has shipped with the
indexed read, delete the flag and the gathered branch of the routed path.

## Long-K F16 wide tile without CU mode (`HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K`)

A Q8_0 F16 wide launch with at least 1024 rows, K >= 4096 and N >= 1024
(the GDN output projection, K=6144, N=2560) runs the 128x128 tile from a
build of the same source without `-mcumode`
(`dense_wide128x128_wgp_f16in_f32_f32_out`) instead of the CU-mode 128x256
tile. Bit-identical (`tests/test_gpu_qwen4exp_gr_f16_direct.py`). Radeon
8060S microbench at K=6144, N=2560: 4096 rows 4.14 -> 3.47 ms, 1024 rows
1.19 -> 1.03 ms; at 512 rows the parent is faster (0.39 against 0.60 ms), at
K=2560 the builds tie, and at K=10240, N=320 CU mode wins. `code-p4096`
1520.7 -> 1535.1 prefill tok/s, same process, identical output tokens.

The rule is a default-on shape test in `hipengine/runtime/gguf_linear.py`
with `HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K=0` as the opt-out. Removal
condition: fold the shape test into the wide route's tile selection once a
release has shipped with it, and delete the flag.

## Dense wide build without CU mode for short chunks (`HIPENGINE_GGUF_Q8_DENSE_WIDE_WGP_MAX_ROWS`)

Q8_0 dense wide launches of 257 to 1024 rows load a build of the same source
without `-mcumode`; larger launches keep the CU-mode build. Bit-identical.
Same-process A/B on a Radeon 8060S (Qwen3.8-Flash-Next UD-Q4_K_XL prefill):
`code-p512` 981.2 -> 1020.5 and `code-p1024` 1256.0 -> 1270.1 tok/s,
`code-p4096` unchanged; with the bound raised to 4096 rows `code-p4096`
drops (1542.6 -> 1504.2). The same switch measured no effect for the
`qwen4_exp_gr` and `qwen35_linear_attn_conv` families and a loss for the
prefill-profile families with CU mode added (`HIPENGINE_PREFILL_MCUMODE=1`:
`code-p4096` 1535 -> 1509), so only this family and row window changed.
`HIPENGINE_NO_MCUMODE_FAMILIES` in `hipengine/core/build.py` is the
diagnostic knob those measurements used; it selects nothing by default.

Removal condition: the env override goes once a release has shipped with
the row rule; the diagnostic knob can stay with the other build ablations or
go with them.

## Qwen4Exp row-split GDN prefill recurrence (`qwen4exp_gdn_rowsplit_prefill`)

The `gguf_ud_q4_k_xl` production binder names
`qwen4exp_gdn_rowsplit_prefill` in
`HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT`; `gguf_q4_k_m` keeps
`qwen4exp_gdn_tiled16_multi_prefill`. The row-split kernel is a port of
gufo's `GdnRowSplitKernel` (commit 9cad139, MIT) with the per-token norms,
decay and beta computed in the block. It is the same recurrence with the key
and query scales applied to the two dot products and a different FP32
summation order: relative L2 difference from the column-warp owner 1.5e-7 on
the gated output and 8e-8 on the final state over 4096 tokens
(`tests/test_gpu_qwen4exp_gdn_rowsplit_prefill.py`).

Measured on a Radeon 8060S at 4096 tokens and 48 value heads, recurrence plus
norm/gate epilogue: 5.58 ms against 8.00 ms for the multi-column tile-16
body. Same process, `code-p4096` 1478.5 -> 1529.8 and `code-p512` 963.7 ->
982.2 prefill tok/s. The gain is the workgroup size as much as the
arithmetic: the same row-split kernel in 256-thread blocks measures 7.93 ms.
Two bit-exact relayouts of the tile-16 body (four lanes per column and eight
lanes per two columns, both 256-thread blocks) measured 8.5 and 9.6 ms and
were not kept, and shrinking the tile-16 body's own blocks to 128 threads
made it slower (10.4 ms) because each block restages the token tile with
scalar global loads.

The row-split owner also writes `half(core)` from its norm/gate epilogue
into the wide F16 stage and the runner declares it as the image of `core`,
so the output projection skips its conversion (same
`HIPENGINE_QWEN4_EXP_GR_F16_MIRROR` switch as the hyper-connection image;
bit-identical; `code-p4096` 1535.1 -> 1554.6 prefill tok/s). The runner
retires the declaration right after the output projection because the stage
is shared with later conversions.

Like the routes in the next section this selection has no manifest row.
Rollback: set `HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT` after the bind, as the
A/B harnesses do. Removal condition: declare the variant in the plan with a
strict scope, then delete the tile-16 multi-column body for this quant once a
release has shipped on the row-split recurrence.

## Qwen4Exp production routes the manifest does not enumerate

The `gguf_ud_q4_k_xl` production binder sets three routes through
binder-owned environment values that have no manifest row: the F16 dense wide
route's layer window (`HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS`, 0-47), the
tiled GDN recurrence's layer window
(`HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS`, every GDN layer; the manifest's
GDN rows do not name a tiled owner for this quant), and the hyper-connection
up projection on the wide route (`HIPENGINE_QWEN4_EXP_GR_WIDE`). Only the
K10240/N320 tail leaf of the dense wide route is a declared scope. The
manifest hash therefore changes with the routed-expert, attention and router
selections but not with these three, and the names
`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS` (13-47) and
`PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS` (27-47) now describe only a quant
without the composition.

Removal condition: add strict scopes for the dense wide layer window, the
GDN recurrence and the hyper-connection up projection, declare them in the
plan, and have the binder publish the window from the manifest row the way it
already does for the router producer.

## The small-matrix F16 GEMM selector

`HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM` (default off) is published by the
`gguf_ud_q4_k_xl` production binder and read by `_dense_f16_gemm_dispatch` in
`runtime/gguf_linear.py`; it is part of the dispatch-resolve cache key. The
96-column floor in that function is a measured setting for this model's
shapes, not a general policy.

Removal condition: replace the selector with a manifest scope for the
indexer projections when the linear launcher's dense-weight dispatch becomes
manifest-driven; revisit the column floor if a kernel variant with a
narrower column tile is added for the 48-column GDN gate projections and the
4-column hyper-connection inject.

## The WMMA QSA attention selector and its F16 K/V gather

`HIPENGINE_QWEN4_EXP_QSA_WMMA` (default off) is published by the
`gguf_ud_q4_k_xl` production binder and read by
`run_qwen4_exp_qsa_prefill_token_mixer`. The route
gathers the whole context's paged BF16 K/V into contiguous F16 scratch rows
for every QSA layer on every prefill chunk (`wmma_k_f16`, `wmma_v_f16` on the
QSA scratch), and builds a per-query block bitmask (`wmma_mask`) from the
selected-position lists the existing top-k expand kernel writes.

The gather wrapper takes a `KVLiveSpans` and rejects anything but uniform
BF16 spans; the kernel reads the page table those spans carry. The three
scratch buffers are allocated on first use and are not in the capacity
planner (`loading/qwen4_exp_scratch.py`): about 2 KiB per context token for
the two F16 buffers.

Removal conditions: (0) a variable-span gather (`live_counts`,
`token_positions`, `evict_mask`) before the route is offered to any
non-dense KV policy, and the scratch counted by the capacity planner;
(1) gather only the positions a chunk appended and keep
the F16 rows per layer once multi-chunk prefill at long context makes the
full regather measurable; (2) have the block selection write the bitmask
directly, which also removes the expand-to-positions kernel from this route;
(3) replace the selector with manifest-driven dispatch.


## Qwen4Exp direct QSA prefill selection (`HIPENGINE_QWEN4_EXP_QSA_DIRECT_SELECT`)

Default on. Prefill rows past the dense-equivalent limit score only the
complete blocks they can see (`qsa_score_h4_d128_rows_f32_kernel`, one
staged query tile per workgroup), and with the WMMA attention the top-k
writes the selection mask directly (`qsa_topk_mask_rows_f32_kernel`); only
the last row still expands positions, for decode. Selection and output are
bit-identical (`tests/test_gpu_qwen4exp_qsa_select_mask.py`; same-process
`code-p4096` 1612.6 -> 1629.4 tok/s, identical tokens). `=0` restores the
tiled all-block scorer, the full expand and the separate mask kernel.

Remove the flag, the tiled scorer's prefill call and the mask kernel's
prefill call once the next Qwen4Exp rate rollup passes with it on.

## Qwen4Exp QSA prefill gate read in place (`HIPENGINE_QWEN4_EXP_QSA_STRIDED_GATE`)

Default on. The prefill gate multiply reads the gate half of each query head
from the q projection instead of a copy the split norm/RoPE kernel wrote,
and writes the gated context's F16 image into the wide-route workspace
(declared as the F16 mirror) so the output projection skips its conversion.
Bit-identical (`tests/test_gpu_qwen4exp_qsa_select_mask.py`); same-process
`code-p4096` 1631.4 -> 1634.5 prefill tok/s. `=0` restores the copy and the
separate multiply. Remove with the direct-selection flag above.

## Qwen4Exp residual write fused into the next norm (`HIPENGINE_QWEN4_EXP_GR_WRITE_NORM`)

Default on for prefill chunks of at least 1024 rows (at 512 rows the
separate per-branch write fills the GPU better). The residual write after
the token mixer is deferred into the FFN hyper-connection read, and the
write after the MoE into the next layer's attention read (flushed before a
PLE layer and after the last layer): where that read runs the
direct norm+inject launch, one kernel (`gr_write_norm_inject_kernel`) writes
the residual and normalizes the row from LDS; every other read path
launches the write on its own first. Bit-identical
(`tests/test_gpu_qwen4exp_gr_write_norm.py`); same-process `code-p4096`
1630.1 -> 1639.3 prefill tok/s, identical tokens. `=0` restores the separate
write. With both writes deferred: same-process `code-p4096` 1634.4 ->
1646.3, `code-p1024` 1300.0 -> 1304.5. Remove the flag after the next rate
rollup.

## Qwen4Exp decode PLE staging overlapped with layer 0 (`HIPENGINE_QWEN4_EXP_PLE_OVERLAP`)

Default on when the PLE staging ring is pinned and the first PLE layer is
not layer 0. A decode step's host PLE hash, row gather and dequantization
run after the layers before the first PLE layer are enqueued, and the copy
is a stream-ordered `hipMemcpyAsync` from the pinned ring instead of a
blocking copy, so about 0.5 ms of host work per step overlaps GPU work.
Bit-identical; same-process decode after `code-p512` 23.52 -> 23.67 tok/s
(warm PLE page cache). `=0` restores the inline blocking copy. Remove the
flag once a device-side PLE gather (planned) replaces the host staging.

## Qwen4Exp MTP batched verify (`HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY`)

Default on in the Qwen4Exp MTP provider (itself opt-in). A cycle verifies
`[root, c1 .. c(n-1)]` in one prefill-shaped pass that keeps each row's GDN
and conv state (`verify_target_block_state_rows`), then
`commit_target_block_rows` keeps the accepted rows without a replay. `=0`
restores serial target steps. Remove the serial verify path once the
batched path is the gated production MTP route.

## Eight-column pack8 kernel behind the row-tile launcher (`HIPENGINE_GGUF_K_PACK8_ROWS`)

Default on. `launch_gguf_k_gemv_rowtile_out` runs
`gguf_k_pack8_rows_out_kernel` (eight output columns per block) when the
output width is a multiple of eight. Bit-identical to the one-column
row-tile kernel. `=0` (read at each launch) keeps the row-tile kernel.
Remove the flag and the one-column body once no caller needs widths that
are not multiples of eight.

## Batch manifest legacy substring binding

`scripts/qwen4exp_ar_batch_candidate.py:batch_manifest_binding` and
`BATCH_PATH_IDENTITIES` remain only for the legacy admission unit tests. The
layer-2 gate now validates the structured
`hipengine.runtime.qwen4_exp_batch_manifest` artifact: live direct identities,
registry keys, projection dispatch keys, step widths, and the profile/source
binding. Delete the substring helper, its identity table, and the fake-scope
tests in `tests/test_unit_qwen4exp_ar_batch_admission.py` once the structured
manifest is committed and no older evidence needs the legacy binding.

## 2026-09-07 UD-U1 F3 — raw rank-3 Q4_K selected-expert consumer is not registry-mediated

The F3 concrete-consumer qualification (UD-QUANTS U1 round 6) binds every
certified admission coverage record to a concrete four-axis registry key,
except one documented escape hatch: the raw rank-3 Q4_K selected-expert
consumer. The production runtime (`_launch_selected_raw_gguf_moe_linear` in
`runtime/qwen35_gguf_runner.py`) calls the direct module wrapper
`hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv:gguf_q4_k_selected_gemv_bf16_bf16_out`,
which is not registered under its own registry key (only the
`selected_dual_gemv_bf16_bf16_out`/`selected_pack8_gemv_bf16_bf16_out`
siblings are). The admission record therefore names the module symbol
(`consumer_module`/`consumer_symbol`) and the parity test proves the symbol
exists on both HIP backends.

Removal trigger: register the single selected wrapper under
`(backend, "linear", "gguf_q4_k", "selected_gemv_bf16_bf16_out")` in
`register_gguf_q4_k_gemv_kernels` (the Q5_K/Q6_K/Q8_0 raw selected singles are
already registered this way by `register_gguf_k_gemv_kernels`), switch the
runtime call site to registry resolution, and drop the
`consumer_module`/`consumer_symbol` fields plus their parity branch from
`hipengine/loading/qwen35_gguf_admission.py`.

## UD-U1 invocation ownership follow-up

- The linear-table mirror and admission-side selected variant-substring ABI
  inference are removed. `qwen35_gguf_consumer_surface` owns production rows,
  named auxiliary contracts and marshalling; `gguf_selected_contract` owns
  selected call kinds, ordered partner bindings and default caller topology.
  Keep the generated `_DISPATCH_TABLE` compatibility view only while tests or
  callers inspect it; remove that view once those users query the resolver.
- The old Q8T16 `(bf16 selector, fp16 output)` key actually launches an FP16
  input kernel. Its key/runtime behavior is preserved, but the descriptor
  exposes the real pointer dtype and admission does not certify BF16 input
  against it. Remove the selector alias after its FP16 callers migrate to an
  explicit typed input contract; do not insert an unqualified conversion.
- `f32_input_operations`, `gdn_force_bf16`, and `recurrent_state_dtype` are
  cold caller declarations, not setters for runtime numerical behavior.
  Replace redundant manual declarations when F1 wires actual entry intents
  into certificate consumption. Native head input must remain the actual
  BF16 scratch owner unless a real caller adapter is implemented separately.
- Optional raw/T16 selected DP4A adapter flags remain runtime diagnostics,
  but admission refuses the affected unrepresented adapter contracts before
  allocation. Mandatory X8 adapters are explicit and qualified as interfaces.
  Remove this refusal only with a shared explicit adapter intent, actual
  caller/operand tests and the applicable existing numerical/profile gate;
  never treat a nearby BF16 record as that proof. Likewise baseline GDN
  prefill cannot certify FP16 state by borrowing its F32-state contract.
- Direct caller boundaries and registered primitive availability are distinct.
  The ordered dual helper owns its two-single fallback; raw Q4 singleton
  registry mediation remains the item above. F1 entry/dependency enforcement
  and F5 profile-binder authorization remain open, not implicit benefits of
  the new immutable metadata.

## 2026-09-11 MTP verify-journal allocation cost on gfx1100

- `Qwen35GGUFTransactionalVerifier` now allocates a row-capable rollback
  journal whenever any reachable decode position selects the serial row
  route. On gfx1100 the native target context limit is 95, so every session
  with `scratch.max_positions + budget + 1 > 95` (all of them, since the
  resident scratch rounds to a 256-position floor) keeps
  `max_rows + 1 = 5` rollback rows per linear-attention layer instead of one:
  measured **+606 MiB** for candidate budget 3 on Qwen3.8-27B
  (5 x 151.50 MiB vs 1 x 151.50 MiB). gfx1151 is unaffected because its
  65,544-token limit admits the initial-state-only journal.
- Recover that memory only with a design that keeps the route contract sound:
  reuse the session-owned multi-row `_verify_linear_*_state_rows` buffers for
  serial rollback, or re-allocate the journal at the top of `prepare()` before
  `capture_initial` when the route first becomes serial. Do not re-enable the
  initial-state-only journal for a session whose position range can reach the
  serial route, and do not widen the capability value without the
  corresponding native-target qualification.

## 2026-09-11 U6 — UD MTP scope pin is withheld

- `_UD_MTP_PRESET_FINGERPRINTS` (`hipengine/loading/qwen35_gguf_admission.py`)
  is the separate MTP certification table. It is **derived** from
  `_UD_MTP_CERTIFICATIONS` and admits only a complete certification unit, so it
  is empty on purpose and can no longer be hand-granted. The UD draft dtype pin,
  the UD accept-chain registrations, the slot-scoped blk.64 draft operation set,
  and the draft residency/`eh_proj`/head ownership contract are landed and
  gated; draft/verifier state disjointness, exact AR/MTP control (blocked by the
  K_S `general_ja_plan` AR divergence), the declared context/width envelope, and
  the category heldouts are not.
- Population trigger: flip the affected `Qwen35GGUFUDMTPCertificationItem`
  entries to `qualified=True` with their passing evidence **and** clear the
  remaining `docs/campaigns/UD-QUANTS.md` U6 checklist items. A record's `blocker` field
  names exactly what is missing; `is_complete()` is what grants the pin.
- Do not populate the pin from a single-prompt or single-width result, and do
  not widen `_UD_PRESET_FINGERPRINTS` (the AR table) to grant MTP scope: the
  two tables exist precisely so an AR certificate cannot imply MTP admission.
- The UD draft dtype manifest (`_UD_NEXTN_DRAFT_QTYPES`) is inert until the pin
  lands. It is not dead code to delete: it is the identity-bound replacement
  for the caller-claimed native-XL variant exception, and its coverage
  contract is enforced and tested.

## 2026-09-11 Committed profiler traces and oversized captures under `benchmarks/results/`

**State: resolved for raw profiler output.** The one-time history cleanup removes
all eight September 11 profiler CSVs. Six existing compact censuses remain; the
two q5-local32 study censuses preserve every kernel's dispatch count and total
duration, with source SHA-256 hashes. Their values were checked against every
source CSV row before removal.

The earlier recommendation to delete oversized window captures was too broad.
Nine named replay inputs remain because scripts or published comparisons consume
them. Twenty-nine other captures retain their metadata and tensor summaries but
omit raw boundary arrays. New raw-vector comparisons require recapture or the
offline recovery archive.

The exact exclusions, removal list and migration checks are in
`docs/campaigns/CLEANUP-HISTORY.md` and its companion manifest. Generated audit
caches are removed from history, but durable triage decisions remain unchanged.

## 2026-09-12 `--ar-decode-mode eager` retained as a diagnostic control

**State.** `scripts/qwen36_dense_gguf_suite.py` gained `--ar-decode-mode
{graph,eager}`, defaulting to `graph`, so the true-AR performance denominator
runs the production decode-graph replay path instead of synchronous scalar
`step()` submission. `eager` is retained as a synchronous diagnostic control.

**Why it is retained.** It is the only way to re-derive the eager/graph
comparison that localized the original denominator defect (same device work,
28.77 vs 28.88 ms/transition at about 860 launches; wall 65-69 ms eager against
31 ms graph). `scripts/gguf_ar_eager_graph_equivalence.py` uses it.

**Removal trigger.** Remove `--ar-decode-mode`, the `eager` branch in `_run_ar`,
and the `autoregressive_step` stage key once no open question needs the eager
path — concretely, after the eager/graph equivalence check is either folded into
a routinely run gate or the eager route is retired. Until then the flag must not
be used to produce a paired or campaign denominator: `scripts/ud_mtp_paired.py`
pins `graph` in `PAIRED_PROTOCOL` and rejects a payload whose
`workload.ar_decode_mode` disagrees.

**Do not promote `eager` to a default or a second supported denominator.** It is
a diagnostic control; every published rate uses `graph`.

## 2026-09-12 `--mtp-scope-grant` on the c1-c8 server bench

**State.** `scripts/gguf_mtp_c1c8_server_bench.py` gained `--mtp-scope-grant`,
off by default, which grants the named preset's MTP scope in process via
`ud_mtp_paired.mtp_scope_granted` so a width cell can be measured while the U6
pin is still unminted. It is documented as requiring
`--generation2-diagnostic`, whose rows are labelled `diagnostic_physical_gguf_mtp`.

**Why it is retained.** The U6 pin mints MTP scope, and the pin needs the width
evidence, so without a candidate-mode grant the widths cannot be measured at
all. The same pattern already exists for the paired protocol.

**Removal trigger.** Remove the flag once the U6 certification records are
complete and the pin is derived, since a granted run is then no longer needed to
reach the cell. A granted run must never be used as a rate claim: it is a
diagnostic whose only purpose is to decide whether a real evidence row is
warranted.

## 2026-09-13 Q5_K gate/up WMMA prefill pair — now unreachable in production

**State.** `_q5_t16_dense_pair_silu_variant` returns `None` below one full row
tile (`_Q5_T16_DENSE_DUAL_SILU_MIN_TILE_ROWS = 32`), so the fused Q5_K gate/up
pair no longer fires anywhere in the native verifier envelope (2-8 rows). Every
pair call in the profiled workload had `gridY == 1`, i.e. a 32-row tile, so the
gate removed the owner's only caller.

**Why it is retained.** The five registered variants and the pair path stay
registered so a rows >= 32 caller can still take them, and so the change is a
policy gate rather than a deletion: the section-6.1 gate evidence is tied to
this dispatch shape.

**Removal trigger.** If no production path is measured taking a Q5_K gate/up
pair at 32 or more rows, delete the pair path together with the five
`dense_dual_wmma_prefill*` variants, the `_Q5_T16_DENSE_DUAL_SILU_ROW_VARIANTS`
table, and the `q5_k_t16_dense_dual_wmma_prefill_*` kernels. Confirm first that
the prefill owner for Q5_K gate/up is the single WMMA prefill
(`gguf_q5_t16_dense_wmma_prefill_bf16`) and not a chunked call into this pair.

## 2026-09-13 Q8T16 shape-explicit rowtile admission — env kill switch

**State.** `_Q8_T16_ALL_ROWTILE_SHAPES` (`{(5120, 1024)}`) admits the Q8_0
`attn_k`/`attn_v` projections at rows 2-8 to the 128-thread
`q8_0_t16_rowtile_gemv` owner instead of the 32-thread
`gguf_q8_0_t16_prefill_wmma` owner. `HIPENGINE_GGUF_Q8_T16_ROWTILE_SHAPES`
overrides the set on/off for bisection.

**Why it is retained.** The set is measured, not broad: the broad
`HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` boolean stays off because audit packet C1
rejected that route on gfx1100 for the decode widths, and because the broad
boolean also reaches the `ssm_alpha`/`ssm_beta` projections and changes their
arithmetic.

**Removal trigger.** Once the `(5120, 1024)` Q8_0 shape is covered by a
registered variant policy rather than an admission helper — or once no UD
artifact carries Q8_0 attention K/V projections — fold the set into the
registry and delete both the set and the env override, leaving
`_Q8_T16_QWEN35_ATTN_IN` as the only in-feature admission.

## 2026-09-13 Q5T16 single-wave verifier rowtile — env kill switch

**State.** `GGUF_T16_NATIVE_ROWTILE_SINGLE_WAVE_BY_QUANT` in
`hipengine/kernels/hip_gfx1100/__init__.py` names the production owner for
`gguf_q5_k_t16_v1` rows 2-8. `HIPENGINE_GGUF_Q5_T16_ROWTILE_SINGLE_WAVE`
overrides it on/off for bisection; clearing it to `0` restores the four-wave
WG128 parent-parity owner `t16_gemv_rowtile_bf16_bf16_out`.

**Why it is retained.** The four-wave entry point cannot be rebound: the
grouped rows6/rows8 owners declare bit-identity to it applied to six-row
chunks, so the two geometries must stay separately registered for as long as
that parity contract is asserted.

**Removal trigger.** If the grouped rows6/rows8 owners are retired, or their
parity contract is restated against the single-wave owner, delete the four-wave
variant, the capability table, and the env switch, and bind
`t16_gemv_rowtile_bf16_bf16_out` to the single-wave body.

## 2026-09-13 strict dense IQ row slab covers the prompt rows — env kill switch

**State.** `_row_batch()` in `hipengine/kernels/hip_gfx1100/quant/gguf_iq_dense.py`
returns the smallest slab in `(1, 2, 4, 8)` at or above the row count, so
`grid.y` is one and each weight slice is read once. `HIPENGINE_GGUF_IQ_DENSE_ROW_BATCH_DOWN`
overrides it for bisection; setting it to `1` restores the largest slab at or
below the row count.

**Why it is retained.** The two rules differ only for `rows` in 3, 5, 6 and 7,
where the old rule left `grid.y > 1`. The kernel declares every `R`
bit-identical, so the switch exists to bisect a launch-geometry suspicion, not
to preserve an arithmetic variant.

**Removal trigger.** Once no artifact or test needs the round-down comparison,
delete `_row_batch_round_down()`, `_ROW_BATCH_DOWN_ENV` and the switch test, and
keep the covering rule alone.

## 2026-09-13 rounded add+rmsnorm fixed-5120 leaf — env kill switch

**State.** `GGUF_ROUNDED_NORM_RESIDUAL_DECODE_POLICIES` in
`hipengine/kernels/hip_gfx1100/__init__.py` routes the `add+rmsnorm` layer to
`rounded_bf16_out_fixed5120_wave256` for `(rows, 5120)` with `2 <= rows <= 8`.
`HIPENGINE_GGUF_ROUNDED_NORM_FIXED5120=0` restores the generic
`rounded_bf16_out` owner.

**Why it is retained.** The two owners are bit-identical - the new leaf
reproduces the generic 256-thread tree level for level and only caches the 20
per-thread rounded values - so the switch exists to bisect a launch-geometry or
routing suspicion, not to preserve an arithmetic variant. It is also what makes
the interleaved census A/B possible, because the switch is read from the
environment inside the profiled child process.

**Removal trigger.** Once no artifact or test needs the generic-owner
comparison, drop the `enabled_env` / `enabled_default` keys from the table, the
`_backend_environment_policy_enabled` branch in
`_gguf_norm_residual_decode_kernel`, and the switch test; keep the fixed-5120
policy alone. The generic `rounded_bf16_out` registration must stay: it is the
declared fallback for every shape the policy does not cover.

## IQ4_XS T16 replacement layout (`GGUF_IQ4_XS_T16_*`)

**What.** `hipengine/quant/gguf_t16.py` carries the IQ4_XS T16 layout
constants, `GGUFIQ4XSTile16`, `repack_gguf_iq4_xs_tile16` and
`unpack_gguf_iq4_xs_tile16`, plus `IQ4_XS_T16_SHAPE` in
`hipengine/quant/gguf_repack.py` and `tests/test_live_gguf_iq4_xs_t16_layout.py`.

**Why it is unused.** It was built as the first half of the "IQ4_XS repack
into a tile layout" item in `docs/campaigns/UD-GFX1151-OPTIMIZE.md`, on the assumption
that IQ4_XS's lack of a tile layout explained the 1.53x gap to the t16 rowtile
owners. The T16-layout owner was then built and measured, and it is
bit-identical to the shipped local32 owner but only 0.79-1.12x on six real
tensors, with the single-wave geometry worse still. The item is rejected, so
nothing consumes the layout.

**Removal trigger.** If no IQ4_XS T16 owner lands, delete the layout
constants, the dataclass, both functions, their `__all__` entries,
`IQ4_XS_T16_SHAPE`, and `tests/test_live_gguf_iq4_xs_t16_layout.py`. The layout is
byte-neutral and correct, so it is cheap to keep as a tested primitive; the
reason to remove it is that dead code in the quant layer reads as a live
option. If it is removed and the item is ever re-opened, the test file is the
specification to restore it from.

## 2026-09-11 Surya HIP runtime: direct backend imports — open

`hipengine/runtime/surya.py` imports its JIT libraries directly from
`hipengine.kernels.hip_gfx1100` (`evie_ops`, `linear_attn.conv`,
`linear_attn.gdn`, `surya.surya_ops`) and resolves kernels by symbol name from
the built `.so`. This is the same pattern the committed EVIE runtime
(`hipengine/runtime/evie.py`) already uses, so the deviation is repo-wide, not
Surya-specific: the four-axis registry currently governs engine/dispatch
kernels, while hand-written runtime runners own their JIT libraries. Migrate
EVIE and Surya together onto a backend-keyed kernel-library resolver when the
runtime runners become a second consumer of shared linear-attention kernels;
until then the direct import is the contract of record.

## 2026-09-11 Surya KV write and decode: dense offset scheme — closed 2026-09-13

`surya_scatter_kv_f32(k, cache, tokens, base_pos, nk, hd, max_seq, row)` wrote
into a contiguous `(nk, max_seq, hd)` plane from a host-supplied
`(tokens, token_offset)` pair, and `_attention_decode` built per-head cache
pointer arrays from `max_seq * hd` strides. Both read no span metadata, which
the architectural invariant forbids.

Closed by two spans-aware kernels in `surya/surya_ops.{hip,py}` plus the runtime
wiring: `surya_scatter_kv_f32_spans` resolves each logical token index through
`base_offsets` and skips tokens outside `live_counts` with a negative
`token_positions` entry or an `evict_mask` mark, and
`surya_full_attn_decode_f32_spans` is a fused fp32 head_dim-256 GQA-4 split-K
producer plus `..._split_k_reduce_f32`. The producer runs one block per
`(kv_head, context chunk)` so all four query heads of a KV head share one K/V
read, and the split count comes from `plan_surya_decode_splits`, which targets a
constant split count rather than a constant chunk because the launch has to stay
wide enough to fill the machine at every context length. The dense policy fills
all four span fields uniformly (identity page table, `arange` positions, empty
eviction mask), so the default path exercises the ABI.

Retained as explicit fallbacks/bisection oracles, not dead code:
`attention_decode_rocblas_f32` (the four-dispatch parent) is the strict
fallback for head_dim != 256 or GQA repeat != 4 and is what
`tests/test_gpu_surya_kv_spans.py` measures parent parity against;
`surya_scatter_kv_f32` (the pre-spans scatter) is used by
`scripts/surya_gpu_debug_attn.py`. Remove both when the fp32 spans decode kernel
is shared with the Qwen3.5 decode path or when the debug script no longer needs
the parent, whichever comes first.

The fused producer still passes `chunk_size` as the score GEMM's `m`, so its
QK^T work is `q_per_kv * head_dim * chunk_size` regardless of how many tokens in
the chunk are invisible; a chunk whose tokens are mostly evicted or empty does
the same dot products as a full one. Bounding each chunk's keys to its last
visible token would cut that, but the dense policy has no invisible tokens, so
there is nothing to measure yet.

## 2026-09-11 Surya vision attention: quadratic score scratch — closed

`hipengine/runtime/surya.py` `_vis_scores` materialized the full vision
attention score matrix, `vision_num_heads * n^2 * 4` bytes for `n` patches.
`vision_num_heads` is 12 and patches are 16 px (`SURYA_RESIZE_FACTOR` 32 is
patch size times spatial merge), and attention runs *before* the merger, so `n`
is the full patch grid `h * w` — four times the merged vision token count.
Scratch was therefore quadratic in page area: 805 MB for the 1024x1024 fixture
(64x64, 4096), 12.88 GB for 2048x2048 (128x128, 16384), **56.5 GB for a
300-DPI A4 page (220x156, 34320 patches)** and **206 GB at the checkpoint's
`SURYA_MAX_PIXELS` ceiling** (256x256, 65536).

Closed by query-row tiling: `plan_vision_attention` derives a query block from
`max_vision_scratch_bytes` (512 MiB default) and `_vision_attention_packed`
walks the score matrix one tile at a time. Each tile still holds the full key
range, so the softmax rows are exactly the rows the dense path computed and
the result is unchanged; only `heads * n * block * 4` bytes are live. The A4
grid now needs 535 MB (106x smaller) and the ceiling 535 MB (386x smaller), and
admission no longer rejects a routine document. Evidence:
`benchmarks/results/2026-09-12-gfx1151-surya-attention-memory.json`,
`tests/test_live_surya_gpu.py::test_gpu_tiled_vision_matches_dense_and_oracle`.

The "result is unchanged" claim is now measured, not inferred: under a 12 MiB
budget (4-10 tiles per page, partial tiles included) the tiled and dense paths
are bit-identical over 4562 teacher-forced full-vocabulary rows — mean/p95/p99/
max KL `0.000e+00`, top-1 `100%`. Evidence:
`benchmarks/results/2026-09-12-gfx1151-surya-numerical-gate.json`.

## 2026-09-12 Surya bench-suite pages clip their own text — closed 2026-09-13

Four of the seven pages in `scripts/surya_bench_pages.py` drew text past the
canvas, so the text the suite claimed to measure was not the text in the image:
`ja` body lines 576-663 px wide on a 512 px page (480 px usable), `mixed` Latin
lines 503/506 px against 480, `scan` ~531 px against 476, and `page_long` six
224 px blocks from y=168 so blocks 5 and 6 fell off a 1024 px canvas. Scoring
against the intended text read correct model output as error, which is how the
earlier `line.` -> `literature` "hallucination" arose: the model completed a
clipped word.

The four pages are re-cut onto canvases the fit assertion accepts (1024x1024
for `ja`/`mixed`/`scan`, 1024x1600 for `long`) and every text page now draws
through `_draw_text_fit`, which raises rather than clipping. The re-cut bytes
are exactly the former `page_<name>_fit.png` bytes, which is why the captured
full-page-protocol oracle only needed its `page` field renamed: the images did
not change. `FIT_PAGES`, `write_fit_pages`, `acceptance_page` and the `--fit`
flag are gone; `page_filename` returns `page_<name>.png` for every page, and
`tests/test_unit_surya_transcription_fixtures.py` asserts each generator still
rejects the old clipping geometry so the pages cannot silently clip again.

`oracle_bench.json` was re-captured (torch fp32, 7 pages) and the lane
comparison re-run, both recorded in
`benchmarks/results/2026-09-13-gfx1151-surya-suite-recut.json`. The vision grids
grew with the canvas (`ja`/`mixed`/`scan` 32x32 -> 64x64, `long` 64x64 ->
100x64), so the absolute rates are not comparable with the superseded
2026-09-11 baseline.

## 2026-09-12 Surya text prefill: quadratic causal score scratch — closed

`SuryaGpuRunner._attention_packed` materialized the full causal score matrix
for the text prefill, `num_attention_heads * tokens^2 * 4` bytes
(`num_attention_heads` 8). Measured on gfx1151 with
`scripts/surya_attention_memory.py`: at `max_seq` 16384 the score matrix was
2.36 GB of a 7.03 GB peak for the 8580 image tokens of a 300-DPI A4 page, and
8.59 GB of a 14.71 GB peak at a full 16384-token prompt; the remaining scratch
was linear at 180.8 KiB/token.

Closed by the same query-row tiling the vision tower uses: `plan_score_tiles`
(one implementation, shared by both paths; `plan_vision_attention` is now the
vision-shaped name for it) derives a query block from
`max_prefill_scratch_bytes` (512 MiB default) and `_attention_packed` walks the
score matrix one tile at a time. Each tile still holds the full key range, so
the softmax rows are exactly the rows the dense path computed, and the mask
stays absolute through a new `query_offset` argument on
`surya_causal_mask_scale_f32` (the causal condition is relative to the tile's
first query, not the tile-local row index). The measured prefill peak falls
7.03 -> 5.21 GB for the page and 14.71 -> 6.66 GB at a full prompt for 0.8%
wall-clock at 16384 tokens, and `block >= tokens` reproduces the dense calls
exactly, so the dense plan remains the strict fallback. Evidence:
`benchmarks/results/2026-09-12-gfx1151-surya-text-prefill-tiling.json`,
`tests/test_live_surya_gpu.py::test_gpu_tiled_prefill_matches_dense_and_cpu_reference`,
`tests/test_live_surya_gpu.py::test_gpu_causal_mask_honors_the_query_offset`.

## 2026-09-13 Surya lane comparison is measured one process per lane — open

`benchmarks/results/2026-09-13-gfx1151-surya-suite-recut.json` is built from
one `surya_perf_compare.py` run per lane merged with `--merge`, because both
lanes in one process return the last three hipEngine rows (`full`, `columns`,
`list`) degenerate from the first generated token with every stage time
unchanged. Measuring hipEngine alone over the same 12 cases passes every gate,
so the corruption rides on shared-process device state left by the torch lane
rather than on the pages or the harness. This is task #80's bug.

When #80 is fixed, re-run the suite as a single combined invocation
(`--split all --runs 3 --lanes hipengine_gpu,torch_cuda`), confirm all 24 rows
pass, and drop the `lane_isolation` protocol note and the per-lane caveat from
`benchmarks/README.md`. `--merge` itself stays: merging per-lane artifacts is a
legitimate way to keep a comparison honest when one lane perturbs the other.

## 2026-09-13 Surya tile shape: grid divisibility was tested and rejected

Grid divisibility was the other candidate shape rule and it was tested: the
sweep records `even` (full last tile) per row, and even division wins at 256 and
6400 patches and loses at 1024 and 4096, so it does not predict the curve. The
rule uses the wavefront multiple instead, which does separate on the A4 page
(where no measured block divides the grid, so the separation is not a tail
effect). If a future rule wants to use divisibility, the data to re-check is
already in the artifacts.

## 2026-09-13 Surya tile shape: 4096-patch spikes have no mechanism — open

The 4096-patch shape curve is reproducibly non-monotone, confirmed by four
independent measurements (two full sweeps, a one-runner probe with a mutated
budget, and a fresh-runner probe; all on a quiet GPU, all with the block
asserted):

| rows | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 256 | 341 | 512 | 1024 | 2048 | 2730 | 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 1575 | 1089 | 939 | 894 | 881 | 909 | 1102 | 1168 | 1175 | 1667 | 1327 | 1137 | 1118 | 1040 |

192/256/341/512/1024 rows are 24-89% slower than 128, and 512 -- a multiple of
32 -- is the worst shape on the grid, 1.6x the dense tile. Nothing measured
explains where the spikes fall; rocBLAS kernel selection at particular
`(M, N, K)` triples is the leading candidate, and a `rocprofv3 --kernel-trace`
over one fast shape (128) and one slow shape (512) at 4096 patches would settle
whether the score-scale or softmax kernel changes or a GEMM switches tiles.
Worth doing before anyone adds another shape rule: the envelope's floor of 128
happens to land on a spike-free value here, which is luck, not design.

Two related consequences, both already applied:

- Rounding down to a multiple of 32 is a memory-neutral tie-break justified by
the A4 page's clean separation, not a guarantee. 512 rows at 4096 patches is a
multiple of 32 and the slowest shape measured on that grid.
- A shape probe that varies the block from outside
  `scripts/surya_vision_tiling_cost.py` must lift `SHAPE_TILE_ROWS` /
  `SHAPE_TILE_DIVISOR` / `SHAPE_TILE_MULTIPLE`, or the production cap overrides
  the requested block and the probe reports one block for every shape. That
  happened here and produced a flat, plausible curve that looked like evidence
  the sweep was wrong. The vision envelope applies whenever it is narrower than
  the tile the byte budget admits, so on any grid where the budget's own tile is
  wider a probe must lift all three constants; lifting unconditionally stays the
  rule.

## 2026-09-13 Surya attention tile shape: the vision envelope's mid-size anomaly — open

`plan_score_tiles` takes a per-path shape policy. The vision tower consults the
envelope `max(128, ceil(rows / 32))` rows, rounded down to a multiple of 32
(`SHAPE_TILE_ROWS`, `SHAPE_TILE_DIVISOR`, `SHAPE_TILE_MULTIPLE` in
`hipengine/runtime/surya.py`) whenever it is narrower than the tile the byte
budget admits; the text prefill passes `shape_envelope=False` and keeps the
budget's width, because its own sweeps show that is its optimum at every
measured length (`2026-09-13-gfx1151-surya-text-prefill-shape-sweep.json`,
`2026-09-13-gfx1151-surya-text-prefill-small-prompts.json`). A conditional
envelope — applying it only while the budget alone left one or two tiles —
fixed the text prefill but regressed narrow vision budgets: at 4096 patches a
64 MiB budget picked 320 rows (1081.07 ms) and a 128 MiB budget 672 rows
(1378.13 ms), against 882.39 ms at the envelope's 128 rows, because the
4096-patch curve has a spike band between roughly 256 and 1344 rows
(`2026-09-13-gfx1151-surya-vision-narrow-budget-check.json`). What stays open:

- **The 6400-patch anomaly.** The envelope picks 192 rows (2130.11 ms across 34
tiles) where 96 rows is 1938.40 ms, 9% faster, and 6400 remains the only
measured grid that wants *more* tiles than `rows / 32` while the surrounding
points do not separate on the wavefront multiple the way the A4 page does
(64 rows is 2057.20 ms, 96 is 1938.40, 128 is 2112.64). Reaching 96 rows needs
the divisor near 64 *and* the floor below 128, which costs 2.7% at 1024 patches
(96 rows 139.65 ms against 136.00 ms at 128). Revisit when a sweep over more
mid-size grids shows what makes this one different; special-casing one grid
would be overfitting.
- **Open: the reported time ceiling.** The estimate has small non-monotone steps
at the 32-row wavefront boundaries: 14 descending steps in 1..20000 patches,
the largest -1.8% at 5089 (1435.97 -> 1409.67 ms), as the envelope's
`ceil(rows / 32)` jumps a row and the block rounds back down to the multiple.
`vision_patch_ceiling` bisects that estimate, so in principle a time budget
inside a step can report a count short of the true maximum; it matches a
brute-force scan at every budget tried from 0.5 s to 200 s, including the 120 s
default (56856 patches). It is sound — the reported count's own estimate fits —
and admission uses `check_vision_capacity` on the requested grid rather than the
ceiling, so nothing is admitted unsafely. A monotone bound would remove the
caveat if the reported ceiling ever needs to be exact.
- **Open: the A4 page.** Its time optimum is a plateau at 2048-8192 rows (40076-40618 ms,
3217.5-12870.0 MiB) and the 320-row default is 7.4% above it, but the byte
budget is what stops there (512 MiB admits 325 rows), not the envelope — the cap
admits 1073 rows at 34320 patches. Raise `max_vision_scratch_bytes` rather than
loosen the cap. 4096 rows (6435.0 MiB, 40076.23 ms) is the measured minimum and
8192 rows is 0.93% slower in the same run, so the return from fewer tiles
saturates around 9 tiles; `rows / 8` would be 4290 rows here and is not a
candidate rule (see the envelope's other grids).

Also found while sweeping, and now refuted: a user-raised
`max_vision_scratch_bytes` above ~6 GB on an A4 grid was recorded as making the
vision attention fail with
`hipErrorInvalidConfiguration` (error 9) from the `vision score scale` check
instead of erroring cleanly. **That does not reproduce**: 4096 rows (6435.0 MiB)
and 8192 rows (12870.0 MiB) both ran cleanly, twice each, in four separate
processes, at 40076-40449 ms
(`2026-09-13-gfx1151-surya-vision-tiling-a4-ceiling.json`). The original sweep's
logs are not retained, so its cause is unknown and no memory ceiling should be
taken from it. The 512 MiB default derives 320 rows and never approaches that
scratch. What is still untested is the dense tile itself — 56.5 GB of score
matrix at 34320 patches, which is inside the host's 124 GiB GTT aperture but is
a ~10-15 minute forward and a 56.5 GB allocation on a shared machine, so the A4
evidence remains shape-to-shape rather than shape-to-dense. If a launch-config
guard on `_vision_attention_packed` is still wanted, the first question is
whether the dense tile runs at all.

## 2026-09-12 Surya prefill tiles still compute the masked-away keys — open

The text-prefill tiling bounds the *memory* but not the QK^T arithmetic: every
tile passes `m=tokens` (the full key range) to the score GEMM and then masks the
keys beyond its last query, so the tiled prefill does the same ~`nq * tokens^2`
multiply-adds the dense path did, of which only the lower triangle survives.
Bounding each tile's keys to `start + bq` would halve the QK^T FLOPs
(`sum_t bq * (start + bq) ~ tokens^2 / 2`), at the cost of a per-tile `keys`
argument in `surya_causal_mask_scale_f32` and a per-tile `head_stride` in the
tile layout (`evie_softmax_rows_f32` already takes `cols` and `tokens`
separately, so it needs no new argument). Do it when the prefill's score-GEMM
share is measured to matter; it is a separate change from the memory fix, which
neither added nor removed QK^T work.

## 2026-09-12 Surya A4 page has no torch fp32 reference — open

The 300-DPI A4 page is in the transcription ground-truth and reading-order gates
(`QUALITY_CASES` in `tests/test_live_surya_transcription.py`) but not in the
implementation-parity gates, because those compare against a captured torch fp32
greedy chain and no such capture exists for this page. So the page's *quality* is
gated and its *parity with the reference implementation* is not.

Capture it when a page-scale parity claim is wanted: add the case to
`scripts/surya_oracle_greedy.py --case protocol` and to
`tests/fixtures/surya/oracle_fullpage_protocol.json`, then move `a4` from
`QUALITY_ONLY_CASES` into `CASES`/`EXACT_ID_CASES` (or `COORDINATE_CASES` if its
bbox digits turn out to be as fragile as the degraded scan's). The capture is a
torch fp32 CPU run of 2108 tokens; nothing else about the page needs to change.

## 2026-09-12 Only Surya reports `GenerationOutput.prompt_tokens` — open

The server prefers a generator-reported `prompt_tokens` and falls back to
`engine.count_tokens(prompt)` when none is reported. Only the Surya generators
set it, because only they build a prompt whose length a tokenizer cannot
recover (text plus image tokens). Every text-only generator still takes the
fallback path.

Set it in the text-only generators when the fallback is wanted gone: it removes
a `count_tokens` call per request and one branch in `chat_completions`. There is
no correctness reason to hurry, since for a text-only prompt the tokenizer count
is already exact.

## 2026-09-13 Surya vision admission is GPU-only and reports runner errors — open

`SuryaGpuRunner.check_vision_capacity` enforces three budgets before any device
work: the score-tile bytes, the vision wall clock, and free device memory. Two
edges of that boundary are unowned.

- The CPU-reference generator (`SuryaOCRGenerator`) has no capacity check at
  all, so a page-scale grid runs a pure-numpy vision tower for as long as it
  takes. Give it the same estimate (`vision_forward_seconds` is fitted to
  device rates, so a CPU lane needs its own constants) or declare a patch
  ceiling; neither generator is wired into a server yet, so nothing reaches
  this today.
- An over-budget page reaches a generator's caller as the runner's
  `SuryaGpuRuntimeError`, which is a request-level fault: the server maps
  `SuryaRequestError` to a 400 and anything else to a 500. Convert at the
  generator boundary when the Surya generators are wired into a server, and
  decide the free-memory check separately — that one is genuinely a
  server-state fault rather than a bad request.

## 2026-09-13 Generation-registry clears cannot be replayed — open

`clear_generation_registry_for_tests()` empties `_FACTORIES`, and nothing can
repopulate it afterwards: the built-in registrations are module-scope
`register_text_generator(...)` calls, so once those modules are in `sys.modules`
a later `register_builtin_generators()` does not re-execute them, and
`_BUILTINS_REGISTERED` stays `True` so it does not even try. A test that called
the clear would empty the registry for the rest of the process. Nothing calls it
today and its docstring now says this.

Give each generation module a callable `register_*()` and call it from
`register_builtin_generators()` — the pattern the runtime-profile registries in
that same function already use. The clear then becomes safe, and the generation
registry can join the collection-time baseline restore in `tests/conftest.py`
that already makes the kernel registry order-independent under
`clear_registry_for_tests()`.

## Compact scoreboard: 2-line headroom at 498/500

- `benchmarks/README.md` was returned to its documented shape on 2026-09-13
  (1962 -> 494 lines). The Qwen4Exp/Framework optimization journal, the
  Qwen3.8-Flash-Next implementation-first status, the Surya planner/kernel
  internals, the harness catalog that duplicates `benchmarks/HARNESSES.md`, and
  the per-campaign delta prose moved verbatim to `benchmarks/HISTORY.md`; no
  line was dropped and every current row stayed.
- What remains is current by construction (Surya topline, root-README export
  block, the `## Current ...` sections, evidence status, reading, maintenance).
  Two lines of headroom absorbs nothing; the next unit must move a section.
- When the next substantial lane lands, either raise the line budget in
  `tests/test_benchmark_readme_sync.py` under the 2026-09-06 rationale
  ("guardrails against worklog-style prose, not a content budget"), or move the
  next-least-current section to `HISTORY.md`. The closest calls this time were
  the harness catalog (already duplicated in `HARNESSES.md`) and the
  Maple-Preview retained backend comparison (Maple still has concurrency rows
  in the scoreboard).
- Do not let the file re-grow: the line/byte/link gate is the ratchet, and the
  failure mode was silent drift, not a single large addition.

## VibeVoice-TTS semantic encoder `rows=1` GEMV dispatch (rejected on quality)

`kernels/hip_gfx1100/vibevoice/registered.py`'s `frontend_gemm` routes every row
count to the tiled WMMA prefill GEMM. Routing `rows < 2` to `dense_gemv_out_bf16`
is **3.00x faster on the semantic stage** (per-chunk `encode_chunk_streaming`
55.22 -> 18.42 ms, isolated A/B in one process) and takes pooled RTF from 1.337 to
a median of 1.146 over five runs. **It must not land: it fails the generated-audio
quality gate.**

- Measured over ten seeds on freshly synthesized audio: **54/60 request-runs pass
  with the dispatch on against 58/60 with it off**, and the on arm reproduces a
  catastrophic failure -- `two-2turn` seed 20260917 at WER 0.6471 with 5 repeated
  4-grams, 11.73 s of output (duration ratio 1.725) and a broken speaker
  attribution. Two more seeds regress, one losing its transcript entirely.
- The numerical difference at the boundary is small (max 0.0625 absolute, 3.7e-3
  relative, about one bf16 ulp at the signal's 16.875 scale), which is why the
  correctness suite and the exact-token-chain gate both pass while the audio
  quality gate fails. Do not treat "small perturbation" as "safe perturbation":
  the two-speaker trajectory is chaotically sensitive to it.
- **A false negative was recorded here once.** A ten-seed A/B was believed to show
  bit-identical outcomes on and off. That comparison was invalid: the harness
  labels its arms `wmma` for threshold 0 and `gemv` for threshold 2, so `wmma`
  means the dispatch is off, and both arms under comparison were off. Any future
  A/B on this path must assert the arm's threshold in the artifact rather than
  trusting the label.

To remove this entry: it would need a mechanism that removes the trajectory
sensitivity, not a re-measurement. The isolated stage win is real and stays on
record in this file so it is not re-attempted from scratch.

## VibeVoice-TTS LM decode gate+up at `threads=64` (rejected on quality, ~1.2x)

`vibevoice_qwen2.py`'s decode gate+up projection calls `dense_dual_gemv_out_bf16`
at the `threads=256` default. **This entry has been wrong twice; the numbers below
are the corrected ones.**

What is solid:

- Under DRAM streaming the kernel reaches **169.5 GB/s at `threads=64` against
  138.1 GB/s at `threads=256`** (55.1 MB of weights per call, 256 MB flush drained
  and synchronized before timing, median of 12). That is a real **~1.23x**, not the
  3.8x an earlier draft claimed and not zero as a later draft claimed.
- End-to-end, applying it took the LM stage from **1.450 s to 1.266 s** and pooled
  RTF from 1.201 to 1.174. That is consistent with ~1.2x on this kernel, which is
  the useful cross-check on any isolated figure here.
- **It is rejected anyway.** It moves 5 of 17920 outputs by one ulp, keeps
  `chain_exact` true, keeps the correctness suite green at 126 passed, and still
  takes the ten-seed generated-audio suite from **58/60 to 57/60** by adding a
  `single-numbers` failure at WER 0.1667.

Two measurement traps were hit while establishing this, both worth avoiding:

1. **Reusing one input buffer measures L2, not the workload.** A benchmark that
   ran 30 iterations against one 55 MB weight buffer reported 74.8 us at
   `threads=64` (an impossible 736 GB/s) because the weights stayed cached.
2. **An unsynchronized flush steals the bandwidth you are measuring.** Writing a
   256 MB buffer between calls without draining it first dropped the apparent rate
   to ~37 GB/s for every thread count, which is how an earlier draft concluded the
   thread count was irrelevant. The memset and the kernel were competing.

**A bit-exact attempt was made and failed to capture the win.** The 256-thread
path at K=1536 falls entirely into the scalar stride-256 tail loop, so a variant
that stages the weight row through shared memory with 8-wide loads should have
recovered the load efficiency while keeping each thread's element set and order
identical. It was **bit-exact on all four shapes tested** (K=1536/N=8960,
K=1536/N=1536, K=1024/N=4096, K=2047/N=512) and measured **0.99x** -- no
improvement. The kernel is latency-bound on DRAM rather than limited by load
width, so the `threads=64` advantage comes from having more workgroups resident
per CU, which cannot be reproduced without changing the thread count and therefore
the reduction width. The variant was reverted rather than kept as neutral code.

To remove this entry: the ~1.2x is worth having but must arrive without moving the
summation order, and the two obvious routes are now closed -- a bare thread-count
change moves 5 of 17920 outputs by one ulp, and shared-memory staging is neutral.
What is left is a change to how workgroups are scheduled or how many outputs each
one owns, and any such change has to keep the 256-wide tree pairing per output.

**Method note:** for a streaming kernel, flush between calls AND synchronize the
flush before timing, then cross-check the isolated figure against the end-to-end
stage delta.

## VibeVoice-TTS: batch the CFG positive and negative LM branches (open, kernel-ABI change)

The session builds two LM runtimes over the same weights, `positive` and `negative`.
`_negative_condition` runs a full LM forward per diffusion frame, streaming all 28
layers for one row. The negative branch is bounded by `stages.lm_total_seconds`
(1.563 s of 4.230 s warm synthesis), which it shares with the positive branch, so its
plausible share is roughly half of `lm_total` -- **~0.7-0.8 s, ~18% of warm synthesis**.

> **Superseded magnitude.** Earlier revisions of this section reported a 65.5 GB /
> ~47% traffic share and a "bit-exact, session-orchestration refactor" framing. Both
> were retracted by `worklog/entries/20260915T200702.494130Z`; the original figure came
> from per-call-sync instrumentation that absorbed outstanding kernels from other
> stages and summed to more than the session can contain. Do not reuse the 49.8% or
> 47% figures.

This is **not** a small refactor. The dependency chain inside one generation step is
strictly sequential:

```
pos_forward(k-1) -> next_token -> neg_forward(k) -> diffusion(k) -> pos_forward(k)
```

`neg_forward(k)` cannot be hoisted, because whether iteration `k` diffuses is decided
by the logits `pos_forward(k-1)` produces. A speculative batch is semantically sound
(`neg_forward(k)` consumes the same `_prev_feedback` row `pos_forward(k-1)` consumed)
but needs a rollback of the negative KV position when speculation misses.

The existing multi-row machinery does not reach across runtimes:

- each `VibevoiceQwen2Runtime` owns its own `self.layers` weight buffers, so one GEMV
  cannot read both branches' weights from a single pointer;
- `vv_kv_write_spans(...)` and `vv_attention_spans(...)` take a **single**
  `k_cache` / `v_cache` pointer, and the two branches hold different KV caches.

So it requires new kernel variants taking per-row cache pointers, plus speculative
execution and KV rollback in the session loop. **That is a kernel-ABI change, not a
tuning change, and it reassociates arithmetic** -- so it needs the production-profile
numerical gate in `docs/EXECUTION-PROFILES.md`, not the bit-exact parent-parity gate.

Expected if it lands: LM 1.450 s toward ~0.86 s. That is a bound from the corrected
~18% share, not a promise, and it is well below the 2-10x framing.

## VibeVoice-TTS: the model is bandwidth bound, and that bounds the remaining work

Not a refactor item so much as the frame for every other one. The diffusion head
holds **246.5 MB** of weights and is invoked once per generated frame with a
20-step solver, so it streams **123.3 GB per request** -- a **~0.59 s floor** at
the ~209 GB/s this host sustains.

> **Corrected by direct device measurement.** `worklog/entries/20260915T205121.492340Z`
> traced the diffusion region under `rocprofv3` and measured **28.18 ms/frame of GEMV
> device time** (89.6% of the head's 31.44 ms of device time), i.e. **~175 GB/s**, or
> **~84% of sustained bandwidth** -- not the 45% this section previously claimed. The
> remaining head time is 2.78 ms/frame of `vv_diff_rmsnorm_modulate` and ~0.5 ms of
> everything else. The head is also invoked with `_GEMV_THREADS = 64` already
> (`vibevoice_tts_diffusion.py` line 36).

Frames cannot be batched to fix this: each frame's diffusion condition is the LM
hidden state for that frame, which depends on the previous frame's semantic
feedback, so the frames are sequentially dependent. The weights cannot be cached
either -- 246.5 MB against a much smaller L2, re-read on all 20 steps of every
frame. The CFG branches are already batched into rows=2, so the weights are read
once for both.

The practical consequence: the diffusion GEMV is already at ~84% of sustained
bandwidth, so it has no large win left. Closing the residual means changing the
summation order, which this model's generated audio rejects. The accepted semantic
win worked because it changed the column tile and not the accumulation order.

Anyone looking for more speed should start from the bandwidth number rather than
from a kernel profile, and should expect the answer to be a work-partitioning
change that preserves the reduction tree, or a quality candidate (fewer solver
steps, lower precision) that the generated-audio suite has to justify.

## VibeVoice-TTS: per-step diffusion D2H readback (resolved 2026-09-16)

The default diffusion loop keeps CFG and DPM-Solver++ state on device. The fused
step preserves the reference rounding points, and timestep/condition invariants
are cached. The host path remains the diagnostic oracle used by parity tests.

The device solver measured a 14.9% diffusion-stage reduction and approximately
3.5% warm-request reduction in a same-session A/B under host load. The invariant
cache separately measured a 12.2% diffusion-stage reduction and 3.4% warm-request
reduction on an idle host. These percentages are separate measurements, not an
established combined speedup. Fresh ten-seed quality evaluation reproduces 58/60;
broader TTS quality qualification remains pending.

Evidence: `worklog/entries/20260916T033657.511650Z-lhl-vibevoice-tts-device-solver-loop-bd1697.md`
and `worklog/entries/20260916T042339.511052Z-lhl-vibevoice-tts-diffusion-cache-measurement-90ed57.md`.
Desktop gfx1151 measurements must establish a new physical-host baseline before
claiming further gains over torch or the previous implementation.

## Streaming consumer pays one threadpool round trip per chunk (open 2026-09-17)
`hipengine/server/api.py` pulls each streamed chunk with a separate
`await run_in_threadpool(_next_stream_item, iterator)` call, and
`StreamingOutputCollector.drain()` is invoked with `max_chunks=1`. The engine
producer therefore only has to be modestly faster than that round trip for the
per-child mailbox to fill, which is what the mailbox-bound fix in
`worklog/entries/20260917T150847.490398Z-lhl-mtp-stream-mailbox-budget-f63132.md`
turned from a cancellation into buffering. Buffering is the correct immediate
behavior, but it hides the real cost: a consumer that is slower than the producer
now delays the request instead of losing it.

Remove the debt by draining a batch per round trip (yield the first ready chunk
immediately, then hand over whatever is already buffered without blocking) and by
raising `max_chunks` in the `drain()` call sites. Re-measure decode throughput on
a streaming MTP request with per-chunk telemetry enabled before and after; the
current per-chunk telemetry doubles the consumer's per-chunk cost, which is what
made the margin visible.

## Concurrent prompt activation is refused instead of overlapped (open 2026-09-17)

`Qwen35GGUFMTP2Adapter.begin_prompt_streaming` allows one prompt activation at a
time: the activation primes shared provider state (prompt rows, carried hidden
rows, provider request slots), and the claim that guards it is released only at
the request's final prefill chunk. A prompt longer than
`max_prefill_chunk_tokens` (256 by default) therefore holds the claim across
ticks, so a second request starting its prefill is refused with
`prompt_activation_in_flight` and decodes without a draft provider. Before
2026-09-17 that condition raised inside the driver thread and closed the whole
engine service, which is what made the concurrency pass at c>=4 unmeasurable.

The provider side looks ready for overlap: every priming structure is keyed by
request (`_request_slots`, `_prompt_prime_rows`, `_prompt_priming_staging`) and
hidden rows land in a per-slot region, and
`GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES` already admits prompt-streaming
widths 1-4 on gfx1151. The refusal is therefore conservative, not a hardware
limit.

**Half replaced 2026-09-18 by scheduling, not overlap.** The scheduler now
consults `prefill_activation_ready` and `prefill_needs_activation`, so a fresh
speculative row's first chunk is deferred until the claim is free and at most one
claim-opening chunked row is admitted per prefill item. A concurrent arrival
keeps its provider instead of being refused; activations still serialize, which
costs the waiting request prefill-start latency. What remains owed is the
overlap path itself: co-activating several chunked rows in one activation, which
needs the exactness gate below (generated-token equality plus non-collapsed
acceptance against independent c=1 runs) and a claim that stays held until every
co-activated sink finishes (`finish_prompt_streaming` currently clears
`_active_prompt_claims` as soon as any one row finishes).

**Intent no longer dies with the refusal (2026-09-19).** The refusal now records
the row in `_streaming_refused_requests` and leaves `mtp2_candidate_budget`
alone, because whether a refused row can still speculate is decided by its
priming source (`_priming_source`: live provider state, a complete and current
prompt buffer, or neither) rather than by the refusal. Zeroing the budget there
was the blanket disable this entry was really paying for.

## Priming-source refusals that still zero the budget (open 2026-09-19)

The unsupported priming classes -- `prefix_reuse_k0`, `target_profile_k0`,
`target_context_k0`, `physical_streaming_category_rejected`,
`operator_disabled_streaming_prompt_k0`, `packed_prefill_unsupported_k0`,
`incremental_streaming_unsupported_k0` -- still set
`row.mtp2_candidate_budget = 0` at their refusal sites, spread across
`hipengine/generation/qwen35_gguf.py` and `qwen35_gguf_mtp2.py`. That is correct
today: each names a property of the row on this route, so the row cannot prime a
provider at all. It should not stay spread out. `_priming_source` is the one
place that answers "can this row speculate", so these sites should ask it
instead of writing the budget, and `prefix_reuse_k0` should stop being a refusal
entirely once #21 lands a checkpoint priming source: a reused-prefix row would
then carry the fourth class (a restored checkpoint) rather than zeroing its
budget. Removing the zeroing without that branch would let a reused-prefix row
reach the gate with no source at all, which is why the two move together.

Related, and separately owed: the serving evidence row for this cell is
`cap4-realized-c1-b3`, which matches only `realized_group_rows == 1`, so a
request admitted while it is alone is later batched into a wider decode group and
silently realizes `effective_route="default"`. The wider realized widths need
their own measured evidence rows before MTP can be admitted at c>1.

**`HIPENGINE_SPEC_MTP_SPLIT_REFUSED_GROUPS` (default off, added 2026-09-18).**
A due group the provider refuses as a whole now decodes autoregressively in one
batch, which is the route the adapter's partition contract specifies. The flag
re-opens the alternative -- one one-row provider cycle per capable row, with the
refused rows autoregressive in the same tick -- so the serving ladder can
re-measure it. The 2026-09-18 measurement rejected it: at c=2 and c=4 with a row
that crosses the provider's 1023-token window beside a healthy co-resident, the
split route raised MTP output share from 1.8%/2.1% to 69.3%/73.9% and lowered
decode throughput by 19%/26% per token, including a row that accepted depth 3 on
every one of its 162 cycles. Remove the flag and
`speculative_split_refused_groups` once the coupling has a retained row on a
second protocol or host, or once a wider-group route actually beats the batch AR
decode; until then the flag is the only way to reproduce the rejected arm.

**Sampled-route finish-rule blockers — resolved (2026-09-22).** The route's
servable blocker set is the sampling law and nothing else
(`temperature`, `logit_bias`, penalties, `suppress_token_ids`), plus the
finish-rule relaxations the cycle commit honors: `ignore_eos`, `min_tokens`,
`eos_token_id`, `stop_token_ids`, and `stop_token_sequences`. The commit applies
the autoregressive finish rule to the whole verified chain through
`hipengine/speculative/streaming.py` `limit_chain_accept_finish`, selecting the
terminal prefix and reporting `eos` or `stop`, so a stop token or EOS that lands
mid-cycle publishes nothing after it.
Evidence: `benchmarks/results/2026-09-22-gfx1151-qwen38-int8-mtp-finish-rule.json`
and `benchmarks/results/2026-09-22-gfx1151-qwen38-bf16-mtp-finish-rule.json`
(`scripts/mtp_finish_rule_gate.py`, with `tests/test_unit_mtp_finish_rule_gate.py`
driving each check to fail against a server that violates it).
Source: `worklog/entries/20260922T140534.086274Z-lhl-mtp-finish-rule-c10d7c.md`.

**Sampled-route debug traces (env-gated, added 2026-09-18; retained 2026-09-19).**
`HIPENGINE_DEBUG_SAMPLED_ROUTE` gates stderr traces that name every gate on this
route: `[route-debug]` and `[cap-debug]` and `[cap-silent-none]` and
`[prefill-debug]` in `hipengine/generation/qwen35_gguf_mtp2.py`,
`[cycle-failure]` in the same file's `recover_cycle_failure`, `[plan-debug]` in
`hipengine/generation/engine_loop.py`, and `[sampled-plan]` in
`_device_sampled_accept_plan` (six refusal sites: row/sampler shape, host state,
planner mode, logits processors, tool-call and json constraints, forced tokens,
temperature, and the missing captured graph with its warmup error). They cost one
`os.environ` lookup per call when the variable is unset and print nothing.

The device-side sampled accept landed on 2026-09-19 and the route now serves by
default at the cap4 c1 key, so the original removal trigger is met; they are kept
because they were decisive twice in that unit (a contained cycle exception
planned `no_provider` for every row, and a warmup skipped at capacity > 1 left
every sampled cycle without its graph) and the route's c>=2 admission cell is
still open. Remove them, or replace them with a documented diagnostic, once the
sampled route's admission cell is widened and measured.
## `HIPENGINE_GGUF_GAPPED_GATHER` kill-switch for the gapped slot-local gather route (added 2026-09-19)

Default **on**. A gapped device-KV placement used to drop the whole packed
prefill slab onto the native three-pass paged prefill kernel (~34x per-launch
vs AOTriton on gfx1151; the 35B regression lane measured 134,511 ms vs 8,226 ms
for the same 6,144-token suffix). With the flag on, a gapped BF16 slot keeps the
slot-local executor: `_gguf_gapped_slot_local_prefill_scratch` swaps its identity
spans for spans carrying the session's real chunk-local block table, the paged
KV write / native fallback / head-major gather all walk it, and AOTriton reads
the same gathered dense head-major buffers the contiguous route uses.

The gather buffers are demand-sized per session
(`_gguf_gapped_slot_head_major_scratch`): resident sessions built at a model's
full context class (e.g. 262,144 positions) exceed the validated 64K head-major
allocation class, so `_ensure_bulk_prefill_workspace`'s capacity-sized pair is
never allocated there. The first gapped slot-local prefill instead grows one
bucketed pair (4,096-token buckets) sized to the context the session actually
prefills, capped by the same validated token/byte class limits, and a context
past the class keeps the native spans fallback.

The forced AOTriton slots survive the gather route: the engine loop slices long
prompts into 256-row prefill chunks, below the 512-token AOTriton threshold,
and contiguous slots cross it via `aotriton_min_tokens=1`. Clearing the force
for gapped slabs (the old packed-route behavior) stranded every gapped chunk
on the native kernel and reproduced the original 104 s lane inside the served
bench; keeping it makes gapped chunks pay contiguous cost.

Removal scope: the flag, the `gapped_slot_local_gather` branch in
`_prefill_batch_native_single_slab`, the demand-sized gather buffers, and the
fast-route condition that keeps `HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX`
binding only when the gather is unavailable - once the gather route holds
through a full `--suite all` milestone plus the served multi-turn A/B on both
the gfx1151 zbook and a W7900 host. Gapped gfx1100 BF16 slots use the registered
gather by default; the optional contiguous-layout capability is not a
prerequisite. Explicit rollback, missing copy kernels, and allocation limits
still select the native spans fallback and retain the suffix guard.

## Two MTP entry points disagree on the sampling contract (open 2026-09-19)

The served GGUF MTP route (the MTP2 adapter, `qwen35_gguf_mtp2.py`) is
capability-gated per sampling mode: a temperature request is served by the
sampled route when an evidence row advertises it, and that route is a measured
1.90x at c=1. The direct-generator MTP entry still declares the pre-gate
contract in two places:

- `hipengine/generation/qwen35_gguf.py::_generate_speculative_mtp_detailed`
  (reached from the entry above it) raises
  `NotImplementedError("GGUF speculative MTP currently supports only greedy-fast
  sampling")` for any non-`GREEDY_FAST` sampler plan.
- The transactional verify work that path drives
  (`hipengine/generation/batch_scheduler.py::next_speculative_verify_work`)
  builds every work item with `target_sampling_policy="raw_target_top1"` and
  `compatible_sampling_modes=("greedy_fast",)`, and raises
  `ValueError("speculative verification requires greedy-fast sampling; ...
  incompatible fields: temperature")` for a sampled request.

Both statements are *accurate for that implementation* -- its verifier commits a
raw target top-1 and cannot serve a sampled accept -- so the repair is not to
widen the constants but to stop having two sampling contracts. When the
direct-generator path is next touched, a sampled request should be routed to the
MTP2 adapter (which already owns the sampled accept) rather than growing a
second sampled accept in the transactional verifier, and the two sites above
should then be deleted rather than relaxed. Until then the refusal is honest and
the gap is confined to direct-generator callers: the server's sampled requests
reach the adapter route and are measured there.

A test-level artifact of the same confusion is gone: the greedy route's
predicate test is now named
`test_greedy_speculative_route_predicate_allows_only_greedy_fast_requests` and
says in its docstring that it pins one half of the policy.

## Full-vocabulary native sampler fallback and GGUF placement (2026-09-21)

`NativeSamplerWorkspace(full_vocab_algorithm="sorted")` selects the full-vocabulary
sort/merge/scan implementation; `"strict"` retains the original native algorithms
for debugging and numerical comparisons. The original registered temperature and
top-p primitives remain strict fallbacks. Sorted FP64 accumulation of FP32 weights
is not bit-identical to the old serial FP32 accumulation, and temperature-only
sampling changes draw ordering while preserving the categorical law.

Keep the constructor choice as an explicit debugging opt-out. GGUF generators
and sessions accept `native_sampler_algorithm="strict"`; their default is
`"sorted"`. The model and sampler arithmetic axes are independent: a strict model
can use the production sampler without claiming strict sampler parity.
`native_sampler_provenance` reports a stable selection manifest, named fallbacks,
and the last successful workspace route; packed execution manifests include it.
A live workspace refuses conflicting selection changes. Preserve the registered
strict primitives as the debugging oracle.
The new implementation has hardware evidence on gfx1151 only. gfx1100 transfer
and the broader model/quant matrix are unverified.

GGUF placement defaults on for supported, available native requests;
`HIPENGINE_QWEN35_NATIVE_SAMPLER=0` retains rollback/bisection value. Keep that
opt-out while independent hardware transfer is evaluated; do not use it as an
admission gate for supported requests. Dynamic constraints and forced queues
still require the host path. The full-vocabulary kernel improvement is
not evidence that those request shapes became supported.

## Provider Prefix Checkpoint Rollback

`HIPENGINE_MTP2_PREFIX_CHECKPOINT_ENTRIES` defaults to four bounded provider
snapshots when radix caching is enabled, and allocates none with prefix caching
off. Zero is an explicit memory/bisection rollback. A target prefix hit alone
cannot reconstruct the draft provider's recurrent state and KV; capture and
restore must travel together.

Keep the rollback until provider snapshots share the target cache's byte budget
and eviction lifecycle. Then remove the independent entry-count control while
retaining the bounded cache policy. Validate the composed user path with
`scripts/int8_mtp_prefix_gate.py` against a radix-enabled server: it requires
real prefix hits, compact INT8, speculative cycles and parity with the same
prefix-reusing AR route. A cache miss or an AR-only response is not a pass.

## Startup scratch probe warms the packed route only when N sessions fit (open 2026-09-21)

The startup scratch probe acquires one resident session per probe slot, and each of
those sessions is constructed at the auto-sized resident context with its own
preallocated KV pages and packed workspace lease. A width-N probe therefore costs
about N full-context sessions. On a box whose free memory holds fewer than N of them
the probe is reduced to one slot, so the packed prefill and verifier warmups the
wider probe exists to run are skipped and the first concurrent request pays for them.

Remove the reduction once the probe can warm the packed route without holding N
full-context sessions -- for example by constructing probe sessions at a short
context, since the warm prompts are at most 128 tokens. That needs a session-pool
key or construction path carrying a context distinct from the request context. Until
it exists, `_startup_probe_width_reduction` in `hipengine/server/api.py` is what
keeps a wide probe from turning a warmup into HIP out-of-memory attempts.

## A failed wide startup probe leaves the process unable to allocate the next stage (open 2026-09-21)

On the 120 GiB box, a startup at `--max-active-requests 4` sent the scratch probe's
resident-session ladder down four HIP out-of-memory attempts. The probe then retried at
width 1 and passed, and the very next stage -- the 9-token chat smoke -- failed with
`HipError: HIP error 2: out of memory` during prefill. The ladder documents that a
failed attempt is rolled back by the session constructor, and the width-1 probe right
after it did allocate a full-context session, so what held the memory afterwards is
unestablished.

`_startup_probe_width_reduction` in `hipengine/server/api.py` now prices the width
before allocating it, so the attempt that failed here does not happen. A genuine
failure still reaches the ladder, and the retry then runs on a machine that may be
tighter than it looks.

Remove this once a failed `_construct_shared_session` in
`hipengine/generation/qwen35_gguf.py` is shown to roll back completely: force the
ladder with a width the box cannot hold, then assert a full-context session plus the
chat smoke still allocate. If the failed attempt does not roll back completely, fix
the rollback instead of weakening the probe.

## Dormant Q5_T16 rows==1 dual block with poisoned variant inheritance (open 2026-09-25)

`launch_gguf_linear_pair_silu`'s Q5 block (`hipengine/runtime/gguf_linear.py`, the
`q5_t16_pair_variant = registered_decode_variant or "q5_dense_dual_silu..."` path) is
dormant on `hip_gfx1151` for three stacked reasons found in E6a (UD-GFX1151-OPTIMIZE2
iteration 11): the identity policy row hands back the **Q4** variant name
`dense_dual_local32_bf16_bf16_out` (the row mirrors plain by design; the IQ4 branch
selects its own key and ignores the value, the Q5 block inherits it blindly, and the
resulting key is unregistered under `gguf_q5_k_t16_v1`); post-E4a both sides dispatch
`t16_gemv_decode_tile8_bf16_bf16_out` while the rule demands the direct key; and the
earlier `_q5_t16_dense_pair_silu_variant(rows)` branch at the `dense_pair_quant ==
"gguf_q5_k_t16_v1"` test is dead code (`dense_pair_quant` only ever holds quants in
`_Q4_T16_DENSE_QUANTS`, which excludes Q5).

E6a's gate says leave it dormant: at the production shape (5120, 17408) rows=1 the
dual is bit-exact against the 2× tile8 + `silu_mul_separate_out_bf16` chain but
**loses** it — 705.6 vs 653.6 µs/layer (0.93×, allocate-once timing, 200 launches,
`~/ud-e1-census/e6a_screen.py`) — because the dual kernel mirrors the direct
owner's schedule while the singles now run on the faster tile8 owner.

Clearing command: re-screen `dual vs 2×tile8 chain` at rows=1 before any change
here; only if the dual wins at the current dispatched singles does it make sense to
fix the variant selection (Q5 branch selects its own key, mirroring the IQ4
precedent), extend the dispatch predicate to the tile8 key, and update the block's
stale comment (it still describes the direct-only rows==1 condition). Until that
screen passes, the dormancy is load-bearing and the dead branch plus the stale
comment should be cleaned up instead.
## Logprobs requests skip the captured-graph sampled accept (open 2026-09-23)

`_device_sampled_accept_plan` in `hipengine/generation/qwen35_gguf_mtp2.py`
declines a request that asked for logprobs. The captured graph samples and
accepts on the device and returns no logits row to the host, so the route has
nothing to report a published token's logprob from; the request falls to the
eager host accept, which scores each published token against the verified row
that predicted it. Serving the metadata is what the logprobs path requires, and
the fallback is exact -- the gate compares it against the autoregressive route
token for token at zero delta -- so the decline costs the graph's speed on those
requests, not their correctness.

Remove this once the graph accept can return the per-row values it selected
from -- the top-1 candidate and its processed logprob, plus the
retained top-k the request asked for -- for the rows it accepted. The evidence
that would justify it is the same gate at zero delta with the decline removed
and the execution path reporting the captured-graph accept.

## Forced-token requests skip the captured-graph sampled accept (open 2026-09-23)

A request with a pending forced token cannot use the captured-graph sampled
accept, for the same reason logprobs requests cannot: the graph samples on the
device and the route needs the row's law on the host to substitute the forced
token for it. `supports_native_gpu_sampling` in
`hipengine/generation/sampling.py` refuses those requests, so they take the
eager host accept, where the override is a point mass on the forced token and
the accept walk corrects to it. `_sampled_accept_summary` in
`hipengine/generation/qwen35_gguf_mtp2.py` raises rather than silently sampling
if a forced token ever reaches the native shape. The fallback is exact -- the
gate compares both arms token for token, including the queue's position in the
output -- so the decline costs those requests the graph's speed, not their
correctness.

Remove this once the graph accept can consume a per-row override, which needs
the device sampler to take a row's forced token as an input the way the host
path does. The evidence that would justify it is the same gate with the refusal
removed and the execution path reporting the captured-graph accept.

## Gemma 4 fused MMQ gate/up prefill is the default; the flag is a rollback lever (open 2026-09-27)

`HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ` selects a fused int8-dp4a MMQ32 route for
Gemma 4's `Q4_K` expert gate/up prefill. It replaces two fp32 grouped launches
plus a fused-stride read with one pack, one tile map, and one MMQ launch that
reads Gemma 4's per-expert `gate | up` row block directly, and it measures
**4.23 s -> 3.33 s on the 512/128 prefill (1.27x)** with path parity intact.

**The route is now the default path.** The lead made that call on 2026-09-27
after the teacher-forced check passed on the gfx1100 branch at worst-case KL
0.0013 and 0.00077 against the 0.05 bar with no top-1 changes, and this branch
reaches the same conclusion by measurement: on realistic prose (2004 ids) the
route matches the fp32 arm for ten consecutive tokens with the first divergence
being llama.cpp alone, and on a truncated Python snippet - the exact class this
flag cited as its cause - it matches for all 23. The variable is now the rollback
lever: `HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0` restores the fp32 grouped route, and
unset means on. `tests/test_unit_gemma4_expert_route.py` pins both halves, because
the campaign's harnesses select their arms through the same function and an
inversion would have had them compare the route with itself while reporting
agreement. The reason this flag was originally raised was a recorded observed failure, and
that failure did not reproduce. The flag's text cited a Python snippet that
agrees for two tokens and then collapses into a fourteen-token repetition;
against llama.cpp on a comparable prompt, cut mid-comparison so the continuation
is genuinely ambiguous, all three arms agree on all 23 first tokens with no
repetition. What does reproduce is a divergence at low-margin positions, and it
is a property of out-of-distribution input rather than of the route. The route's
raw-logit perturbation is three times smaller in distribution than out (3.46 max
absolute difference on prose against 9.76 on random ids), which is the mechanism
that was previously recorded as unidentified: the int8 activation-quantisation
step is small enough not to move a decision on input the model was trained for,
and random ids leave the logits flat enough that it is. The route replaces only
the prefill's expert gate/up projection, so one forward of N ids under each arm
reproduces what a longer teacher-forced chain shows, and
`scripts/gemma4_mmq_prefill_length_probe.py` sweeps that. Measured on the frozen
campaign chain, deterministically: the route is over the binding 0.05 `kl_max`
bar at **7 of the 13 prefill lengths from 16 to 1024 ids** where it runs - `kl`
0.305 at 16, 1.007 at 24, 0.109 at 32, 0.135 at 48, **2.036 at 64** - with greedy
decision flips at 24 and 64 ids. At 1024 ids it is 7.3e-07.

The earlier recorded reason for this flag was wrong in a way that argued for
promotion. It read the breach as a near-certain-row tail effect: `kl_max` 0.0651
on 2 of 1022 rows, top-1 100% with zero flips, and the already-shipped attention
split breaching the same bar at the same rows. That measurement was taken at a
1024-id prefill, which is this route's **best case** rather than a representative
one, and the current default path is bit-identical to the strict arm on both
chains (KL exactly 0.0 over 1023 rows, including all 77 close-margin rows, with
30690 split launches engaged), so the split does not breach that bar today. The
applicability question that text raised was not the operative one.

The mechanism is input-distribution-dependent rather than a per-row
activation-quantization bound: that bound is per row and does not vary by six
orders of magnitude with the row count, and what varies is how often the model's
margin is smaller than the bound. The leaf's only unit coverage was once 4
experts over 10 rows in `tests/test_unit_gemma4_expert_route.py`, which does not
reach the real geometry of 128 experts;
`tests/test_gpu_gemma4_expert_route_true_geometry.py` now covers the real geometry
including an expert spanning more than one MMQ tile.

Remove it once a teacher-forced gate against the campaign's frozen evaluator
has been recorded for the new default and its verdict is on file. Note
that `capture` must run with `HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0` to freeze the
incumbent fp32 path, since capture otherwise freezes the new default and the gate
would compare the route with itself. The implementation, its registered launch
path, and its correctness tests stay in place. Evidence:
`benchmarks/results/2026-09-27-gemma4-mmq-gate-up-three-arm-attribution.json`,
`benchmarks/results/2026-09-27-gemma4-mmq-gate-up-short-prefill-divergence.json`.

## Gemma 4 expert down MMQ: exact automatic default after continuation failure (open 2026-10-03)

Automatic selection does not use the MMQ down; since 2026-10-10 it runs the
grouped WMMA owner (see the WMMA rollback lever entry at the top of this
ledger), and before that the registered exact down owner. The existing one-plane
MMQ route failed nine of ten generated-history screens; three-plane feedback
failed seven. The exact-down control matched all 610 paired strict logits.
Explicit `HIPENGINE_GEMMA4_MOE_DOWN_MMQ=1` and pinned `mmq` selection remain
runnable; neither the representation nor the evaluator thresholds changed.

Clearing command (use the same artifact for both arms):

```bash
HIPENGINE_GEMMA4_MOE_DOWN_MMQ=1 .venv/bin/python scripts/gemma4_generated_history_gate.py \
  --model <artifact>.gguf --output-dir /tmp/gemma4-down-mmq-repair
```

Remove the lever after a repaired MMQ owner passes this unchanged continuation
screen and the full applicable execution-profile gate, with selected-route,
repeatability and isolation evidence. See
`worklog/entries/20261003T123946.224463Z-lhl-repair-tiled-expert-default-dispatch-after-gemma-e11ee2.md`.

The following measurements describe the earlier 2026-09-28 decision, not the
current default.

`HIPENGINE_GEMMA4_MOE_DOWN_MMQ` selects the Q5_1 DS4 DP4A MMQ route for Gemma 4's
expert down projection. Before it, Q5_1 had no MMQ path at all --
`gemma4_project_experts_gate_up_mmq` accepts only `gguf_q4_k` and `gguf_q5_k` -- so
the down projection ran the fp32 grouped family at **8.6 GB/s** where the Q4_K
gate/up MMQ on the same layer ran at **67 GB/s**, 11.8x per FLOP. It measures
**7.191 s -> 5.213 s on the 2048-token prefill (+38.0%)**, and the expert share of
prefill falls from 43.4% to 11.0%. On the loop's 512/128 shape the prefill goes
from 1.52 s to 1.01 s.

**Historical default (2026-09-28).** The route defaulted on, and
`HIPENGINE_GEMMA4_MOE_DOWN_MMQ=0` restored the FP32 grouped route.
`tests/test_unit_gemma4_expert_route.py` and
`tests/test_gpu_gguf_q5_1_mmq_selected_prefill.py` test the implementation.

**Why the width constraint had to be lifted, not worked around.** The Q5_1 MMQ
kernel tiled `in_features` in 128-wide DS4 blocks and computed
`ds4_blocks = in_features / 128` with integer division, so a width that is not a
multiple of 128 silently dropped its tail; its `in_features % 128` guard existed to
prevent exactly that. Gemma 4 26B-A4B's expert down width is **704**, which is
`5 x 128 + 64`. The route now rounds `ds4_blocks` up and skips the Q5_1 sub-blocks
a trailing group does not have, and the DS4 activation pack rounds `blocks_per_row`
up and reads `0.0` past `hidden`. **This is exact, not an approximation:** the
elements past `hidden` are zero, and a zero cannot raise a `max_abs` or move a sum,
so the scale the real elements get is the one they would have got without the
padding.

**Numerical envelope, measured.** At the model's own geometry -- 4096 compact rows,
128 experts, 704 -> 2816, one plane -- against the fp32 grouped owner and against an
exact dequant oracle: grouped vs oracle max 0.00789 / mean 0.000712; MMQ vs oracle
max 0.01297 / mean 0.001336; MMQ vs grouped max 0.01218 / mean 0.001110. That is the
same envelope the Q4_K gate/up route above already runs at, and it uses the same
plane count (`_MMQ_ACTIVATION_PASSES = 1`) for the same reason.

**It is over the 0.05 `kl_max` bar on the 2048/1024 teacher-forced gate, and that
over-bar is not this route's.** Measured 2026-09-28 against
`/tmp/gemma4-gate-strict-gfx1151.npz`: all routes on gives `kl_max` 0.137246 with
`kl_mean` 0.000317, `kl_p99` 0.000179, `top1_rate` 1.0 and 0 flips; with this route
off it is **0.170633**; with the WMMA dense prefill off it is 0.167806; with both off
it is 0.156917. Every configuration fails the bar and turning this route off makes it
*worse*, so the over-bar is the pre-existing property of the Q4_K gate/up MMQ route
recorded in the section above -- over the bar at 7 of 13 frozen-campaign prefill
lengths -- and not something this route introduced. All-on also reproduces to six
decimal places across runs, so the result is deterministic rather than a tail draw.

Remove it once a teacher-forced gate against the campaign's frozen evaluator
has been recorded for the new default and its verdict is on file. Note that
`capture` must run with `HIPENGINE_GEMMA4_MOE_DOWN_MMQ=0` to freeze the incumbent
fp32 path, for the same reason the gate/up entry gives. The implementation, its
registered launch path, and its correctness tests stay in place. Evidence:
`benchmarks/results/2026-09-28-gemma4-prefill-expert-down-route-attribution.json`.

## Gemma 4's pack8 expert layout is deleted (settled and removed 2026-09-28)

**RESOLVED 2026-09-28: the layout was measured never ahead, and the flag and its
unreached code have been deleted.** Nothing here is outstanding.

The clearing condition this entry set was measured at 512 prefill rows against 128
experts, the one regime the ladder does not give to `grouped_prefill`: prefill
343.2 against 340.0 tok/s (a 0.9 percent difference, inside run-to-run variation)
and decode 21.24 against 7.78 tok/s, reproducing the documented 2.73x regression on
a fresh run. Evidence:
`benchmarks/results/2026-09-28-gemma4-gfx1151-expert-pack8-layout-settlement.json`.

### What was removed

The flag `HIPENGINE_GEMMA4_EXPERT_PACK8_LAYOUT` and the unreached code behind it,
as this entry's second bullet specified: `LAYOUT_Q4_K_PACK8`, `pack8_arrays`,
`_materialize_pack8`, `_pack8_shapes`, `_pack8_nbytes`, `pack8_layout_enabled`, and
`gemma4_project_experts_pack8`, along with `_PACK8_ALLOCATION_NAMES`, the
`pack8_selected` rung of the expert route ladder, the planning branch in
`hipengine/loading/gemma4_gguf_device.py`, the pack8 branch in `resident_bytes`,
the `gemma4_project_experts_pack8` leaf in `scripts/gemma4_prefill_kernel_census.py`,
the env-var row in `docs/ENVS.md`, and the six tests in
`tests/test_unit_gemma4_gguf_device.py` that existed to exercise or pin the layout.
`_product` and `import os` became unused in the loading module and went with them.

The removal takes a flag, a 19.2 GB raw/packed duplication path, and 116.6 s of
load-time repacking off the path of anyone who had set the variable.

### What deliberately remains

**`_SELECTED_PACK8_VARIANT` and the `pack8_selected` GEMV kernels are a different
thing and were left in place.** That family is the 8-output-columns-per-block form
of the per-lane selected GEMV over the *raw* blocks; it is preferred where it is
registered by `gemma4_project_experts_selected`, a live route, and it is pinned
bit-exact against its single-output owner by
`tests/test_unit_gemma4_expert_route.py`. It shares the name and nothing else. The
dense `require_pack8` GEMV path in `hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.py`
is likewise unrelated.

### Why the default-off was legitimate while it lasted

It was one of the real default-off cases: a **measured cause**, not a missing
qualification. Decode measured 21.03 tok/s without the layout and 7.64 tok/s with
it, with `pack8_selected` confirmed as the route that ran. The layout was
arithmetically correct, so every correctness test passed either way -- which is why
the default was pinned by its own test, because a silent flip back would have
reintroduced a 2.75x decode regression with nothing else noticing. That pinning
test is gone with the flag, which is the correct end state rather than a coverage
loss: there is no longer a default that can flip.

The duplication the flag enabled was 19.2 GB on the 26B artifact: 60 rank-3 expert
tensors at 14.4 GB raw expanding by 1.33x. It was affordable against the measured
124 GiB device ceiling, which was never the reason the flag was off. Evidence:
`benchmarks/results/2026-09-27-gemma4-gfx1151-pack8-expert-route-measured.json`.

## The 128-wide MMQ32 K pass is implemented, correct and unwired (open 2026-09-28)

`gguf_q4_k_selected_dual_q8_1_ds4_mmq32x128_prefill_compact32_kernel` stages four Q4_K
subblocks per pass instead of one, taking the gate/up projection from 88 staging iterations
and 176 `__syncthreads()` to 22 and 44 at identical arithmetic. It is exported as
`hipengine_gguf_q4_k_selected_dual_q8_1_ds4_mmq32x128_prefill_compact32_bf16_bf16_out` with a
Python wrapper, and `tests/test_gpu_gguf_q4_k_q8_1_selected_prefill.py` checks it **bitwise**
against the 32-wide route on three geometries plus the CPU reference. Nothing routes to it.

Removal condition: delete
`gguf_q4_k_selected_dual_q8_1_ds4_mmq32x128_prefill_compact32_kernel`, its `extern "C"`
launcher and its Python symbol unless the route is either re-measured at the production gate/up
geometry and wins, or wired behind an `in_features % 128 == 0` dispatch condition and shown to
improve end-to-end prefill. Clearing command: rerun
`scripts/gemma4_mmq_k_width_leaf_bench.py` at `out_features` 1408 once the fixture can reach it,
or promote the route and take an end-to-end number.

**It is unwired because it was measured and did not earn the place.** On the leaf bench
(`scripts/gemma4_mmq_k_width_leaf_bench.py`) the wide pass was 0.986x at `in_features` 512,
0.897x at 1024, and 0.914x at 2816 -- and the 2816 medians moved between runs (narrow 138.2
then 159.5 us) while the minima sat within 4% of each other, so at the production K depth it
is a wash. The barrier count was worth a factor of 1.36 in the earlier estimate; it is worth
about 10% at best and nothing at 2816.

Two things would clear it, and either is enough:

- **Re-measure at the production geometry.** The bench could not reach `out_features` 1408
  because the synthetic weight generator overflows `uint8` past 128, so every point above runs
  at 128 output features and a quarter of the intended arithmetic intensity.
- **Wire it and take an end-to-end number.** That requires an `in_features % 128 == 0`
  dispatch condition, which the narrow route does not need.

If neither happens, delete the kernel, its `extern "C"` launcher, its Python symbol and its
three test parameters. The `KSUB` template machinery underneath stays either way: it is a
verified no-op at `KSUB = 1` and it is what made the experiment cheap.
## `HIPENGINE_GEMMA4_MOE_PREFILL` selects a prefill route that fails an absolute bar (open 2026-09-26)

The Gemma 4 prefill MoE has two WMMA owner variants per projection that are 2.7x
faster than the exact routes: a compensated form (fp16 high part plus fp16 residual
per weight, two WMMA ops per k-tile) and the plain uncompensated form kept only as
the diagnostic that isolates the fp16 weight-rounding term. `auto` runs neither.
The compensated form passes every aggregate logits bar with 10-200x margin and
leaves top-1 unchanged on all 1023 scored rows, but breaches the campaign's absolute
`kl_max` bar at 0.060867 against 0.05 on 1 of 1023 rows. The breach is reduction
association, not a defect; a two-way accumulator split moved it to 0.081599 rather
than reducing it. See the iteration-59 diagnostic in
`docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md`.

Remove this once the lead rules on whether an absolute `kl_max` applies to a
reordering-class change. If it does not apply, delete `_PREFILL_MODE_ENV`,
`_prefill_mode`, `_prefill_route_flags` and the exact-route default, make the
compensated variants the prefill route unconditionally, and delete the plain
variants and the diagnostic modes with them. If it does apply, delete the Gemma
wiring for both WMMA owner variants and their `compensated` plumbing instead, and
keep the kernels registered for callers that only need their own parity contract.

## Gemma 4 gate/up split allocation (resolved 2026-09-27)

`materialize_gemma4_gguf_device_weight` used to upload the fused
`ffn_gate_up_exps` stack **and** two per-expert `gate`/`up` copies, because the
Q4_K int8 MMQ32 leaf derived each expert's weight stride from its own output
width and so could not address a fused stack. `Gemma4GGUFWeightSpec.split_gate_up`,
`_mmq_split_requested`, `_is_fused_expert_gate_up` and the
`_MMQ_DUAL_WEIGHTS_ARE_SPLIT` refusal in `gemma4_experts_forward_bf16` carried
the condition. The duplicate cost 8.04 GiB plus allocator overhead: peak device
use at `--prompt 1024` was 32.35 GB, and Gemma 4 stopped loading on the 24 GB
RX 7900 XTX with `HIP error 2: out of memory`.

Removed rather than completed. The leaf takes an explicit `expert_stride_rows`
(0 keeps the old per-half behavior for every other caller), Gemma 4 passes the
fused width, and the fused allocation is the only copy again. Peak device use
on the W7900 is 22.06 GiB (49.0% of 45.0 GiB) and the XTX loads and runs. None
of the symbols above exist any more, so there is nothing left in this ledger's
sense to delete; this row records the removal and closes it.

Evidence: `worklog/entries/20260927T081858.830497Z-lhl-gemma4-mmq-fused-stride-242422.md`;
iteration 152 in `docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md`;
`tests/test_gpu_gguf_q4_k_q8_1_selected_prefill.py`
(`test_q4_k_q8_1_ds4_mmq32_fused_stride_matches_split_tensors` -- fused output
bit-identical to split, with a negative control proving the comparison cannot
pass on an ignored stride); and
`benchmarks/results/2026-09-29-gemma4-moe-fused-stride-gate-w7900.json`
(the campaign gate at `--prompt 2048 --prefill 1024`: `passed: true`,
`kl_max` 7.7e-04 against a 0.05 bar, 0 of 1023 top-1 flips).

## `HIPENGINE_GEMMA4_DECODE_GRAPH` whole-step graph replay is prize-bounded, not correctness-bounded (open 2026-10-10)

The Gemma 4 generation loop replays decode steps from a whole-step HIP graph
capture (`hipengine/runtime/gemma4_decode_graph.py`). Since 2026-10-10 the
flash phase kernel re-derives its slice partitioning per block from a staged
device live-extent slot (`[delta, live]`, staged per replay per geometry),
so every route a decode step takes below 15328 keys — flash, the two-phase
split, the class kernel — replays bit-identically to the launched path; the
capture window is bounded only by the measured launch-bound prize
(`_CAPTURE_MAX_KEYS` = 2176 keys: +15.4 percent decode at 1024 prompt
tokens, +2.2 percent at 2048, flat-to-negative from 2560 where the step is
device-bound). The env `HIPENGINE_GEMMA4_DECODE_GRAPH` (default on) is the
rollback seam for the whole route, and the in-session `launched_fallbacks`
counter makes the fallback visible per request.

History: the 2026-10-09 restriction blamed the split's host-computed slice
geometry and blocked capture from 1024 keys; the 2026-10-10 layer probe
(worklog 20261010T010705) attributed the replay divergence to flash's
live-key-count partitioning (the split and class routes always replayd
bit-exact), and a route-crossover attempt to re-route the launched path to
the split was refuted on natural tokens the same day (worklog
20261010T024452). The device-resolved slot keeps the launched path
byte-identical (a null slot reproduces the scalar partitioning exactly) and
needs no execution-profile gate.

The window is per backend: a backend package may declare
`GEMMA4_DECODE_GRAPH_CAPTURE_MAX_KEYS`, and `_capture_max_keys` reads it. The
2176-key default and the gains above are Radeon 8060S measurements.
`hip_gfx1100` declares `0`: on a W7900 (2026-10-12, graph on against off at the
default window) decode was 0.989x at 256 prompt tokens, 1.000x at 512, 0.994x
at 1024, 0.934x at 1536 and 0.927x at 2048 with equal generated ids, so its
decode steps stay launched. Replay exactness on that device is still pinned by
`tests/test_gpu_gemma4_decode_graph_capture.py`, which captures inside the
default window whatever the backend declares.

Removal condition: correctness no longer bounds the window -- the
class-global kernel (resident row above the 64 KiB LDS budget, keys > 15328
for head_dim 512) and the staged singleton deep variant were probed at
16384 and 32768 prompt depths on 2026-10-10 and replay bit-identically,
like every other route, so the guard is purely the measured engagement
threshold. It goes when the capture cost falls below the device-bound
crossover (today flat-to-negative from 2560 keys: the per-bucket capture
cost plus the frozen superset walk outweighs the launch-gap savings on
device-bound steps), or if the deep steps become launch-bound again. Then
the bucket guard, the fallback counter, and the env all go, and the graph
covers the full decode range. The intermediate capture design notes
(retarget cost, capture-amortized buckets) live in the module docstring
and the 2026-10-09 worklog entry.

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
