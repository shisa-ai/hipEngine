---
status: current
owns: Cleanup ledger for dead flags, duplicate dispatch paths, and fallback code to remove after optimal paths are proven.
---
# Refactor backlog

Only unfinished work belongs here. Remove an entry when its action or decision
is complete; keep implementation history and measurements in worklogs and Git.
An entry is not necessarily blocked: local design and implementation can start
before model-backed or hardware-specific completion checks. The
[per-entry prerequisite audit](../audit/refactor-sweep/2026-10-31-classification.md)
separates the next work from those completion dependencies.
Every entry names its owning repository path, concrete scope and completion
condition. `python3 scripts/docs/check_refactor.py` checks these fields, active
IDs, closed statuses and local links; it also runs in the documentation gate.
Preserve supported fallback paths and apply the verification rules in
[AGENTS.md](../AGENTS.md) and [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md).

## Dense Llama attention only tiles head_dim 128

Recorded 2026-10-27, rewritten 2026-10-28 when the flash route landed.
`hipengine/kernels/hip_gfx1100/attention/dense_flash_prefill.py` is the tiled
causal attention for dense Llama-family prompts and decode steps, and
`dense_causal_prefill_bf16` routes to it by default with the exact block kernel
as the strict fallback. Its K/V tile is fixed, so shared memory does not mention
the context and the path has no LDS-derived context ceiling; a null
`keep_mask` means "the causal mask", so a prompt does not materialize a
`tokens x keys` byte array either.

What remains: the tiled kernel's register tile is specialized for `head_dim`
128. Any other width falls back to the exact block attention, which is
LDS-bounded and needs a real mask -- `max_dense_prefill_tokens` derives its
prompt bound from that launcher's own admission rule, and `_DecodeScratch`
allocates an all-ones mask row only for that geometry. A dense checkpoint with a
different head width therefore serves the exact kernel's context bound, not the
checkpoint's.

Removal condition: tiled attention supports all admitted head widths. Keep the
exact fallback and its mask for explicitly requested strict arithmetic; remove
only geometry-driven mask allocation and bounds once they are unnecessary.
Validate with
`uv run pytest tests/test_gpu_llama_dense_prefill.py tests/test_gpu_llama_dense_decode.py`.


## Outstanding work

### R0023 — Preserve recovery precedence and decide held-off family repairs

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `PRODUCTION_MOE_PREFILL`; `Q4_IU8_PREFILL`; `HIPENGINE_QWEN4_EXP_Q4_DP4A64`.

**Deferred family decisions:** retain recorded repair conditions independently:

- `PRODUCTION_MOE_PREFILL`/`Q4_IU8_PREFILL`: fix the divergent task responses;
  faster tiling or numerical-envelope passes alone do not clear the task gate.
- `GR_IU8` and `GR_IU8_DOWN`: R0026/R0027 own their independent correction gates.
- `Q8_MMQ_PREFILL`: localize the operand role responsible for canonical-depth failure.
- `runner.gdn_peer_prefill_layers`: require a complete-owner mechanism, not noise-level timing.
- `Q4_DP4A64`: pass both decode gates from the late-layer production-prefill
  entry before restoring the arithmetic default. All callers carry explicit
  `q4_dp4a_decode_enabled`/`q4_dp4a_decode_layers`; the process fallback is removed.
- `QSA_FLASH_PREFILL`: choose a concrete repair or retirement disposition for
  the failed recovery composition.

Done when: Remove an intent binding only after final rejection without a repair path, or an exhausted reopen
condition followed by a removal decision. Do not silently re-enable withdrawn arithmetic by
narrowing the sweep.

### R0030 — Decide router-selection semantics before reopening warp arithmetic

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `f32_hidden_warp_reduce_dense_exact`.

Done when: Deferred campaign decision: the recorded reopening proposal quantizes ranking logits with
deterministic tie-breaking, including the reference. That changes strict selection semantics and
needs an explicit campaign decision and new reference validation, not a kernel-only retry or
threshold relaxation. Do not rerun the unchanged rejected route before that decision.

Preserve tile4 strict fallback and the diagnostic warp implementation. F16 production and
ordered-DPP recovery choices are separate siblings, not rejected by this candidate's result.
The diagnostic `router_producer_variant(runner, mode)` scope selects these siblings;
do not delete shared selection infrastructure to remove one candidate.

### R0028 — GR-wide profile handoff and engagement checks

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `HIPENGINE_QWEN4_EXP_GR_WIDE`.

Owner-bound `gr_wide` selection is implemented across profile construction,
runtime dispatch and candidate-driven diagnostics. Preserve profile precedence,
strict fallback, and the `rows > 256` and `low_rank % 64 == 0` capability checks.

Done when: Localize and repair the composition's failed canonical-depth and
natural-long numerical comparisons, then finish task review with complete code
outputs. The reports in
[`audit/refactor-sweep/2026-11-01-r0028-gates.json`](../audit/refactor-sweep/2026-11-01-r0028-gates.json)
record actual F16-direct engagement, passing state/repeats/teardown and failed
numerical limits. The four-arm
[isolation report](../audit/refactor-sweep/2026-11-01-r0028-isolation.json)
shows wide-on changes logits within strict and production, while production
wide-off independently misses the envelope against strict wide-off. The
[sampled primitive report](../audit/refactor-sweep/2026-11-01-r0028-primitive.json)
tracks the F16-rounded oracle closely; operand rounding dominates its error
against original F32 inputs on those samples. Screen a precision-preserving
projection against both primitive oracles, then run full depth/state/task checks.
Isolate the remaining production changes separately.
Public generation at 73/257/389 tokens confirms threshold engagement. Keep
independent counts for both F32 and F16-direct leaves. Use the canonical depth
gate and layer2 gate with `--prompts
benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl` after a concrete repair.
A one-hot prefill-boundary KL is not a substitute for full-depth numerical/task
validation.

### R0027 — Repair isolated GR-down IU8 drift independently

Owner: `hipengine/generation/qwen4_exp_profiles.py`.

Scope: `hipengine/generation/qwen4_exp_profiles.py` — Repair isolated GR-down IU8 drift independently.

Done when: Repair GR_IU8_DOWN independently of GR-up and exact fusions. Screen one prefill per arm below the
approximately 5e-9 prefill-boundary KL limit before the complete 780-row gate. Do not restore the
rejected compensated correction solely because it improved one boundary metric. Keep R0023 recovery
precedence.

### R0026 — Repair isolated GR-up IU8 projection drift

Owner: `hipengine/generation/qwen4_exp_profiles.py`.

Scope: `scripts/qwen4exp_gr_iu8_logits_probe.py`.

Done when: Repair GR_IU8 projection drift without changing the exact sigmoid/gated-mean epilogue. Screen one
prefill per arm with `scripts/qwen4exp_gr_iu8_logits_probe.py` below approximately 5e-9
prefill-boundary KL, then run the complete 780-row depth gate. Keep exact-fusion and GR-down
decisions independent; do not retry unchanged flag settings.

### R0039 — Separate multi-column GDN capability from composition task review

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `qwen4exp_gdn_tiled16_multi_prefill`.

**Deferred composition decision:** preserve the distinct R02 task finding in
`benchmarks/results/2026-09-15-journey-gdn-multi-restoration/task-review.json`. Its numerical passes
do not clear the misleading native-INT8 explanation under the recorded per-prompt rule. Do not
conflate it with R01's Japanese failure or claim proof of worse expected quality/a kernel defect.
The truncated Japanese pair is not completed evidence. Do not rerun the unchanged composition or
promote it solely from its numerical envelope; any revised composition needs complete numerical,
task and depth gates with accurate route attribution.

Done when: Resolve the composition-task decision with complete outputs and
accurate route attribution. If a revised composition is selected, run its full
numerical, state, depth and task gates. Preserve the registered multi-column,
DPP and serial strict comparisons; passing owner-transport checks does not
resolve the R02 task finding.

### R0060 — Bind historical dense serving evidence when its artifact is available

Owner: `hipengine/models/qwen35.py`.

Scope: `hipengine/models/qwen35.py`, `artifact_execution_fingerprint`.

Done when: Deferred artifact dependency: obtain and verify the historical measured artifact, then run
`python3 scripts/gguf_execution_identity.py <artifact.gguf>` and record its fingerprint for evidence
attribution only. Preserve
`tests/test_unit_speculative_mtp_serving_capability.py::test_qwen36_dense_rows_without_identity_still_run_on_capability`
and `admission_basis="implementation"` in both modes. Never invent a fingerprint or use a
byte-digest match to conflate qwen36-dense and qwen38 execution identity.

### R0058 — Decide dormant ordered-pair redesign and Q3/Q3 capability

Owner: `hipengine/kernels/hip_gfx1100/fused/gguf_iq4_q4_pair.py`.

Scope: `gguf_q3_k+gguf_iq4_xs`, `gguf_iq4_xs+gguf_q3_k`, `gguf_iq3_s+gguf_iq4_xs`,
`gguf_q5_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1`.

**Deferred redesign/removal decision:** change scheduling or strict-emulation structure before
repeating `~/ud-e1-census/e6b5_screen.py`, `~/ud-e1-census/e6b7_screen.py` and
`~/ud-e1-census/e6_closeout_screen.py` (or successors). Require bit-exact parent+SiLU output at
K5120 and (K5120,N17408), rows1, and the original >=1.00x screen before selecting a candidate.
Preserve ordered gate/up keys, slot_path and both wrapper ABIs: Q3/IQ3_S strict owners without
policy or under a pin, IQ4 local32, Q5 tile8 c1, and Q6 direct-planar parents. Do not enable a
different arithmetic composition.

If the family is redesigned or the relevant quant layouts retire, remove the four dormant
wrappers/keys and their exclusive B_KIND2/B_KIND3/A_KIND3 bodies and tests together after checking
consumers. Preserve shipped IQ4_NL/Q5, IQ4_NL/IQ4_NL, existing dense Q4/Q4 and strict unfused
chains.

Done when: Implement the missing Q3_K/Q3_K side-A strict-Q3 owner, wrapper and registry key before screening it
against two strict Q3 leaves plus SiLU. If retiring the dormant family instead, remove its exclusive
wrappers, registrations and tests together after checking shared consumers; preserve shipped
IQ4_NL/Q5, IQ4_NL/IQ4_NL and dense Q4/Q4 pairs.

### R0057 — Repair gfx1151 compact attention wave producer

Owner: `hipengine/kernels/hip_gfx1100/attention/dms_compact.hip`.

Scope: `worklog/entries/20260926T140616.347941Z-main-dms-producer-control-183661.md`.

Done when: Repair the gfx1151 wave-group6 compact attention producer. Before restoring its selection, pass the
unchanged four-category G0 no-evict command in
`worklog/entries/20260926T140616.347941Z-main-dms-producer-control-183661.md`, matched-input GPU
attention tests, public DMS lifecycle tests and the applicable numerical gates. Preserve the working
generic producer and independent gfx1100 selection.

### R0051 — Review backend decode and strict attention scope ownership

Owner: `hipengine/generation/gemma4_profiles.py`.

Scope: `gemma4_plain`.

Done when: Decide whether backend decode routing and strict attention binding should share one scope contract.
Inventory prefill, one-row prefill tails, incremental decode and verification selectors; explicit
`gemma4_plain` verification does not establish strict behavior for every phase. Preserve
staged/plain and shape/backend fallbacks. Test profile isolation, scope precedence, graph capture
and public selected routes on both backends before unifying them.

### R0080 — Candidate-budget compatibility alias deprecation

Owner: `hipengine/speculative/candidate_budget.py`.

Scope: `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET`; `HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET`.

Done when: Deferred alias retirement: ship the implemented `FutureWarning` transition before removing
`HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET`. Active in-repo scripts already write the canonical
`HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET`; compatibility tests and historical evidence still name
the alias. Before removal, recheck active callers and user configuration examples, migrate any
remaining writers, and update resolver/reporting tests and configuration documentation together.
See `worklog/entries/20261031T202425.449059Z-cleanup-refactor-r0080-deprecation-ab53a8.md`.
Until retirement, preserve exact requested depth, agreement checks when both names are present,
and malformed/conflicting/unsupported-value errors. Keep the generic direct-provider default;
never reinstate independent fallback reads or silently rewrite requested depth.

### R0196 — gfx1100 dense SPECDEC2 C8 physical-width rollback

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `hipengine/generation/qwen35_gguf.py` — gfx1100 dense SPECDEC2 C8 physical-width rollback.

Done when: Remaining validation: exercise `hipengine serve` at C5/C8 using production D1/D24 workloads,
confirming physical grouping, deterministic/isolation behavior, allocation-failure cleanup and
cancellation. Preserve the original complete category and control/ownership gates before closing
this entry.

### R0199 — gfx1100 fused Q4-T16 prefill row-retile bisection

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — gfx1100 fused Q4-T16 prefill row-retile bisection.

Done when: Validate defaults-only C1-C3 public serving for fused Q4-T16 gate/up+SiLU retile selection. Cover
row48 at rows33-48, row64 at rows49-64 and row128 at rows65-128; preserve the 256-row parent for
larger rows and quant/shape/backend misses. Require actual route engagement and
full-output/state/lifecycle checks.

### R0200 — gfx1100 Q4-T16 row64 down-projection bisection

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — gfx1100 Q4-T16 row64 down-projection bisection.

Done when: Run a defaults-only, same-protocol public repeat for the Q4-T16 down owner at `(17408,5120)`,
rows33-192. Confirm selected launches, output/state parity and teardown; preserve the registered
256-row shared-B parent outside that scope.

### R0067 — Dense prefix retention budget

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS`; `HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES`.

Done when: Decide whether the dense prefix-retention default should allow a wider reuse working set at the cost
of pinned memory. Coordinate `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS` and
`HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES`; retained KV pages consume memory outside the
state-byte ceiling. Update defaults, configuration docs and tests together, then validate
interleaved conversations and memory pressure. Preserve explicit operator controls until equivalent
owner-bound policy replaces them.

### R0249/R0250 — Laguna public cache-owner validation

Owner: `scripts/laguna_public_correctness.py`.

Scope: `scripts/laguna_public_correctness.py` — Laguna public cache-owner validation.

Done when: Dependency: a local Laguna checkpoint and a gfx1100 runtime for public owner engagement. No
checkpoint is present at the repository's documented local paths. **Clearing action:** run `python
scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1100 --max-tokens 8 --output
/tmp/laguna-public.json` in the ROCm environment. Also exercise a sufficiently long public prefill
to engage the ordered-F32 owner and confirm resident-cache and activation-reuse selection,
blocking/stream/direct parity, and cleanup. Until then this is a validation deferral, not a claim
that the public path passed. It does not disable or restrict the implemented paths.

### R0273 — Laguna public packed-query validation

Owner: `scripts/laguna_public_correctness.py`.

Scope: `scripts/laguna_public_correctness.py` — Laguna public packed-query validation.

Done when: Dependency: the same unavailable Laguna checkpoint noted above, on gfx1151 for this packed-query
route. **Clearing action:** run `python scripts/laguna_public_correctness.py "$LAGUNA_MODEL"
--backend hip_gfx1151 --max-tokens 8 --output /tmp/laguna-packed-public.json` in the ROCm
environment, with a prompt reaching the dense-initial BLAS path. Confirm packed-query selection,
blocking/stream/direct parity and resource cleanup. Do not infer public engagement from leaf
numerical tests.

### R0275 — Laguna public softmax-owner validation

Owner: `hipengine/kernels/hip_gfx1151/`.

Scope: `hipengine/kernels/hip_gfx1151/` — Laguna public softmax-owner validation.

Done when: Dependency: the unavailable Laguna checkpoint noted for R0273. **Clearing action:** run the
R0273 gfx1151 public command with a dense-initial BLAS prompt, additionally confirming that the
backend-selected wave-per-row softmax executes. Check blocking/stream/direct parity and teardown.
Existing packed/wave/block leaf comparisons do not establish this public engagement.

### R0274 — Laguna public packed-query producer validation

Owner: `scripts/laguna_public_correctness.py`.

Scope: `scripts/laguna_public_correctness.py` — Laguna public packed-query producer validation.

Done when: Dependency: the unavailable Laguna checkpoint noted for R0273. **Clearing action:** run the
R0273 gfx1151 public gate with a dense-initial BLAS prompt and confirm direct packed-query
production for the eligible tiles, generic production for other rows, output parity and clean
teardown.

### R0248 — Laguna public cache regression after H7Y retirement

Owner: `scripts/laguna_public_correctness.py`.

Scope: `scripts/laguna_public_correctness.py` — Laguna public cache regression after H7Y retirement.

Done when: Dependency: the unavailable Laguna checkpoint noted for R0249/R0250. **Clearing action:** run
their gfx1100 public generation gate with a prefill long enough to exercise H6Z/H6W, confirm the
selected variants and natural-cache allocation lifecycle, and compare blocking/stream/direct
outputs. The removed H7Y route must not be selected or allocate mirrors.

### R0276 — Public Q4/Q6 tile-policy validation

Owner: `tests/test_unit_q6_wmma_tile_retirement.py`.

Scope: `tests/test_unit_q6_wmma_tile_retirement.py` — Public Q4/Q6 tile-policy validation.

Done when: Dependency: the unavailable Laguna checkpoint. **Clearing action:** run the R0273 gfx1151 public
gate with the retired `HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE=64x16` input deliberately set; it must
be inert. Trace Q6 WMMA launches and confirm the backend's default tile still runs, outputs match
the unset control, and teardown is clean. Repeat with the retired
`HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE=64x16` input and confirm shape-selected Q4 tiles are unchanged.
Explicit primitive tile arguments remain supported and are not the subject of this negative control.

### R0271 — Scaled F16 comparison retention and public validation

Owner: `scripts/laguna_grouped_down_category_bench.py`.

Scope: `prefill_350_candidate`; `scripts/laguna_grouped_down_category_bench.py`.

**Reference diagnostic:** retain `prefill_350_candidate` under
`scripts/laguna_grouped_down_category_bench.py` as the named scaled-F16 comparison owner;
its control and candidate remain distinct from range-direct. The route-retirement decision is
closed as intentional reference code. See
`worklog/entries/20261031T202627.067172Z-subagent-r0271-r0271-scaled-f16-diagnostic-0ff940.md`.

Done when: Validation dependency: a local Laguna checkpoint. Run the R0273 gfx1151 public command and
confirm range-direct selection, output parity and teardown.

### R0001 — PARO verifier graph regression

Owner: `tests/test_live_paro_mtp_graph.py`.

Scope: `benchmarks/results/2026-10-24-gfx1151-paro-graphs/NOTES.md`; `reproduce_graph_candidate.py`; `validate_graph.py`.

Done when: Dependency: a graph implementation that removes that observed regression. **Clearing action:**
use the reproduction block in `benchmarks/results/2026-10-24-gfx1151-paro-graphs/NOTES.md` to run
`reproduce_graph_candidate.py` with three repetitions and two confirmation repetitions, then
`validate_graph.py` and `tests/test_live_paro_mtp_graph.py`. Require faster paired execution without
output/acceptance drift and successful public state/recovery checks before removing the eager
preference. Preserve the eager comparison implementation; add no environment flag or MTP admission
gate.

### R0165 — Public physical GDN wave-reduction validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `hipengine/runtime/qwen35_gguf_runner.py` — Public physical GDN wave-reduction validation.

Done when: Dependency: a public request on the gfx1100 physical exact-row route. **Clearing action:** run
`hipengine serve` with the dense GGUF MTP profile and an eligible physical target width, trace the
wave-reduced state-row launch, and compare output/acceptance plus teardown against the registered
strict parent. The existing numerical gate is
`tests/test_gpu_qwen35_gguf_gdn_prefill_correctness.py::test_gdn_prefill_segments_state_rows_wave_reduce_is_parent_bit_exact`.
Do not substitute successful c1 or unrelated-backend generation for engagement.

### R0166 — Public physical Q4 rowtile validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public physical Q4 rowtile validation.

Done when: Dependency: gfx1100 public physical target execution reaching R6 and R36. **Clearing action:**
run `hipengine serve` with the physical dense-GGUF MTP profile, record the R6 two-wave rowtile and
R36 row48 fused gate/up+SiLU launches, and compare outputs, acceptance and teardown against their
registered parents. Use
`tests/test_gpu_gguf_q4_k_t16_dense.py::test_q4_t16_dense_dual_wmma_silu_matches_unfused_chain_at_production_shape`
for the independent numerical gate; it does not replace public engagement.

### R0168 — Public grouped Q4 active-slot validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public grouped Q4 active-slot validation.

Done when: Dependency: gfx1100 public physical execution with five and six active target slots. **Clearing
action:** run `hipengine serve` with the physical MTP profile, trace grouped-R8 at R24 for C5/C6 and
grouped-R6 on the corresponding C7/C8 scope misses, and compare outputs/acceptance/teardown with the
strict parent. Preserve
`tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_q4_t16_dense_rowtile16_w2_grouped_rows8_matches_grouped_rows6_bits`
as the separate numerical gate.

### R0169 — Public exact C7 target-row validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `hipengine/runtime/qwen35_gguf_runner.py` — Public exact C7 target-row validation.

Done when: Dependency: gfx1100 public C7/K3 physical MTP execution. **Clearing action:** run `hipengine
serve` with seven live requests and the physical MTP profile, confirm R28 target work rather than
padded R30, and compare full state/output/acceptance and teardown against the padded composition.
Mocked prefix/tail routing tests do not replace this public gate.

### R0170 — Public source-Q5 MMQ validation

Owner: `tests/test_gpu_gguf_q5_k_mmq_wmma_prefill.py`.

Scope: `tests/test_gpu_gguf_q5_k_mmq_wmma_prefill.py` — Public source-Q5 MMQ validation.

Done when: Dependency: gfx1100 public C8 physical MTP execution reaching the R32 source-Q5 owner.
**Clearing action:** run `hipengine serve` with that configuration, trace source-Q5 selection at R32
and raw-MMQ fallback on R24/shape misses, then compare output/acceptance and teardown with the
retained parent. Use
`tests/test_gpu_gguf_q5_k_mmq_wmma_prefill.py::test_c8_q5_i64_j32_source_mmq_passes_production_outer_floor`
as the separate numerical gate, not a substitute for public engagement.

### R0171 — Public raw-Q5 sidecar and route validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `hipengine/runtime/qwen35_gguf_runner.py` — Public raw-Q5 sidecar and route validation.

Done when: Dependency: gfx1100 public C8 physical MTP execution with raw-Q5 sidecars. **Clearing action:**
run `hipengine serve` at eight active requests, confirm sidecar ownership and MMQ launches at
admitted R24/R32 shapes, then compare output/acceptance and allocation cleanup with grouped T16.
Include fewer-request and missing-workspace cases to verify fallback, without weakening admission.

### R0172 — Public grouped Q5 C8 validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public grouped Q5 C8 validation.

Done when: Dependency: gfx1100 public C8/R32 physical target execution. **Clearing action:** run `hipengine
serve` with eight live MTP requests, trace grouped-Q5 R8 launches, and compare output/acceptance and
teardown with the registered padded-prefix/strict-tail parent composition. The direct numerical gate
is
`tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_q5_t16_grouped_rows8_matches_repeated_rows8_bits`;
it does not establish public engagement.

### R0174 — Public exact C8 target-row validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `hipengine/runtime/qwen35_gguf_runner.py` — Public exact C8 target-row validation.

Done when: Dependency: public gfx1100 C8/K3 physical target execution. **Clearing action:** run `hipengine
serve` with eight live MTP requests, confirm R32 target packets and the selected fused row32 owner,
and compare output, acceptance and cleanup with global exact-target rows disabled. Include a non-C8
request count to confirm unchanged padding outside this policy.

### R0176 / R0394 — Public direct resident verifier validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `layout.state_indices`.

**Dependency:** public gfx1100 physical target execution. Reproduce and resolve the gfx1100 slot-zero failure before removing its guard.

Done when: trace layout indices, bound Conv/GDN pointers and captured rows through public
gfx1100 MTP. Compare direct and packed-import output and accepted-state ownership across slot zero,
nonzero borrowed singletons, multi-session groups, cancellation and teardown. Reopen the older
singleton route only after reproducing its failure and passing the width-one explicit-K3 category
suite plus C1-C2 validation. Keep the working captured/deferred one-slot mapping enabled while that
separate hardware follow-up runs.

### R0177 / R0292 — Public planar-Q6 chunk-policy validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `_small_b_rowtile_chunks`.

Done when: Dependency: defaults-only gfx1100 and gfx1151 direct/server refreshes. **Clearing action:**
exercise `LLM.generate()` and `hipengine serve` on both backends, trace Q6 root row chunks at eight
requests and a non-divisible packed width, and compare logits/output and cleanup with the registered
R4 composition.

### R0178 — Public grouped Q6 mixed-row validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public grouped Q6 mixed-row validation.

Done when: Dependency: public gfx1100 physical MTP mixed-row execution. **Clearing action:** run `hipengine
serve` across the admitted mixed widths, trace grouped prefix/tail launches, and compare outputs and
acceptance with the registered repeated primitives. Include a shape outside the mixed policy and
verify cancellation and cleanup. Direct grouped-versus-repeated GPU bit parity is separate from this
public-route requirement.

### R0179 — Public grouped Q5 R6 validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public grouped Q5 R6 validation.

Done when: Dependency: public gfx1100 physical target execution reaching grouped R6. **Clearing action:**
run `hipengine serve` across grouped R6 and exact-prefix/tail widths, confirm selected launches, and
compare output and acceptance with the registered repeated R6 primitive. Include ordinary decode
outside physical scope and cancellation/cleanup. Direct GPU bit parity does not establish public
routing.

### R0180 — Public grouped Q4 R6 pair validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public grouped Q4 R6 pair validation.

Done when: Dependency: public gfx1100 physical target execution reaching grouped pairs. **Clearing
action:** run `hipengine serve` across admitted pair widths, trace both grouped output launches, and
compare output/acceptance with the registered repeated-R6 composition. Include a shape outside pair
policy and cancellation cleanup. Direct grouped-kernel bit parity is not public pair-route evidence.

### R0181 — Public grouped Q4 R6 single-owner validation

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public grouped Q4 R6 single-owner validation.

Done when: Dependency: public gfx1100 physical target execution reaching this wrapper. **Clearing action:**
run `hipengine serve` at admitted grouped widths, confirm single-owner grouped launches, and compare
output/acceptance with repeated R6. Include non-grouped rows, a shape outside policy, and
cancellation/cleanup. Direct grouped-kernel bit parity does not establish public wrapper engagement.

### R0182 — Public fused packed-state transfer validation

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `hipengine/runtime/qwen35_gguf_runner.py` — Public fused packed-state transfer validation.

Done when: Dependency: public gfx1100 physical MTP execution using packed state imports. **Clearing
action:** run `hipengine serve`, trace the fused pointer-table gather where direct resident indexing
is inapplicable, and compare imported Conv/GDN state with the per-layer parent. Include single-slot
fallback, rejection, cancellation and resource cleanup. CPU ownership tests do not prove public GPU
state-transfer engagement.

### R0211/R0212 — Scheduling policy and generic-default decisions

Owner: `hipengine/kernels/hip_gfx1100/__init__.py`.

Scope: `protect_ttft`; `HIPENGINE_PREFILL_DECODE_POLICY`.

**Decision needed:** keep policy selection as a supported production control, or standardize
production on `token_budget`; separately, choose whether unconfigured generators should inherit
`fair` instead of `protect_decode`. The old entries proposed both generic `fair` and eventual
`token_budget`, so neither is an unambiguous cleanup replacement. Different backend burst defaults
must not be erased as a side effect.

Done when: if policy selection is retained, close R0211 as supported configuration. If
standardized, migrate every generator default, submit/poll constructor and CLI/env caller to
explicit round budgets before removing the old policy branches. For R0212, either keep the scoped
defaults as supported registry configuration or change the generic default and remove only redundant
package policy declarations. Update configuration docs and run `tests/test_unit_llm_generate.py`,
`tests/test_unit_concurrency2_token_budget.py` and
`tests/test_integration_generation_engine_loop_burst.py`; exercise public mixed-prefill/decode,
staggered admission, cancellation and cleanup on both backends and confirm the selected policy.
TTFT/ITL comparisons evaluate the chosen behavior; they are not permission to run it.

### R0213 — Mirrored INT8 fallback migration and validation

Owner: `hipengine/kvcache/dense.py`.

Scope: `tests/test_unit_qwen38_int8_kv_capability.py`; `tests/test_unit_qwen35_gguf_int8_kv_policy.py`; `tests/test_unit_gguf_packed_verify_layout.py`.

Done when: Dependency: finish the remaining replacement validation and migrate actual mirror consumers, not
implement row-batched attention again. **Clearing action:** exercise public direct and mirrored
requests on both HIP backends with short, long, shifted and ragged allocations, cancellation,
refill, resource rejection and cleanup; record selected storage/attention sources and actual
persistent/transient ownership. Use the current planner to price these cases rather than reusing old
capacity ceilings. Migrate or explicitly retire remaining mirrored callers before removing its
allocation/copy/dispatch code. Run `tests/test_unit_qwen38_int8_kv_capability.py`,
`tests/test_unit_qwen35_gguf_int8_kv_policy.py`, `tests/test_unit_gguf_packed_verify_layout.py` and
`tests/test_unit_gguf_int8_mtp.py` alongside that public validation.

### R0214/R0215 / R0378 — Cumulative-semantic rollback-control migration

Owner: `hipengine/kernels/hip_gfx1100/__init__.py`.

Scope: `scripts/qwen35_08b_cumulative_semantic.py`; `HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL`,
`HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL`; `HIPENGINE_GGUF_DENSE_WMMA_BULK`.

Done when: validate the actual Qwen3.5-0.8B Q4_K_M and Q8_0 callers. Neither
checkpoint is present in the local `/models/gguf` catalogue; other available Qwen
models do not establish engagement of the narrow rows512/K1024/N16+N16 pair.
The diagnostic-control migration is implemented: the cumulative harness uses
`cumulative_prefill_diagnostic`, threshold children carry explicit scoped arguments,
and production no longer reads these three environment controls. Dispatch tests
cover bulk policy at 511/512/513 and unrelated lengths; defaults and primitive
fallbacks are unchanged. **Clearing action:** with those checkpoints available,
exercise public p512 generation and eager/graph decode, trace pair/residual
selection, compare against registered scalar/singleton/add controls, and verify
cleanup. See the cumulative-prefill worklog entries for migration evidence.
This remaining checkpoint dependency does not restrict runtime admission.

### R0217 — PM4 diagnostic controls and transport rollback

Owner: `scripts/pm4_graph_bench.py`.

Scope: `pm4_stateful`, `pm4_timestamps`, `HIPENGINE_SUBMISSION_TRANSPORT`.

The comparison interfaces have live owners: `scripts/pm4_graph_bench.py` owns
explicit global-dependency (`pm4_stateful`) and timestamp (`pm4_timestamps`)
comparisons in separate submission contexts; `scripts/pm4_lifecycle_repro.py`
owns timestamp/quarantine stress controls. `pm4_stateful_local` is already
rejected: canonical `pm4` uses stateful registers with local dependencies.
The transport selector remains the supported explicit rollback interface in
`hipengine/core/pm4/transport.py`; do not delete these distinct comparisons as
unused aliases.

Done when: resolve the recorded ROCm/ROCm#6529 lifecycle investigation on
gfx1100 and validate public automatic/explicit transport, rebuild, cancellation,
shutdown and resource cleanup. No gfx1100 runtime is available on this gfx1151
host. **Clearing action:** on gfx1100 run the safe public transport lifecycle
checks and follow `docs/reference/PM4.md`'s issue-grade runbook for investigation;
reset-risk queue-recreation stress needs its separate explicit approval.
The local graph-bench/transport/lifecycle unit bundle passes; it does not prove
gfx1100 device behavior. Preserve HIP graph and distinct AQL profiling, and keep
full provenance serialization out of replay.

### R0219 — Source-F16 packed-prefill integration and rollback decision

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `q6_f16_*`, `use_t16_f16_rocblas_prefill=False`.

Done when: complete the remaining gfx1100 public selected-route and numerical checks, plus the
applicable named-strict profile and task gates on each affected backend. The request-row-scoped
packed owner and geometry-based admission are implemented. On gfx1151, the repaired bounded K16
consumer passes the canonical/category-heldout incremental parent comparison and public
`LLM.generate_detailed()` at non-anchor, 512/1K/4K and mixed lengths with selected-route and cleanup
evidence; see `worklog/entries/20261031T190151.817522Z-lhl-r0219-k16-accumulation-eecd1d.md`.
That same-production source-off comparison is not a named-strict profile certification. Run the
same public shape matrix on gfx1100 with its Q4 row ceiling and ordered pair-only operand policy,
then evaluate full-logit strict-teacher/category/repeat/isolation and task behavior before closing
this item. Preserve the supported quant-neutral constructor opt-out for numerical diagnostics;
automatic source-F16 remains enabled. Do not repeat the completed FP16/FP32/publication-order
diagnostics or gate compatible artifacts on file identity.

### R0220 — Dense gfx1151 source-F16 scratch reuse

Owner: `hipengine/kernels/hip_gfx1100/__init__.py`.

Scope: `GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES`; `worklog/entries/20260816T152449.196381Z-lhl-qwen38-gfx1151-memory-parity-d82d61.md`; `tests/test_unit_qwen35_gguf_prefill_scratch_liveness.py`.

Done when: Dependency: dense gfx1151 has no `GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES` entry, so its
activation and weight tiles are independently allocated. The gfx1151 MoE owner-slot
policy and gfx1100 dense arena do not establish safe reuse for this geometry. The 2026-08-16
memory-parity investigation recorded final-logit changes under both dense owner-slot and
single-arena aliases, plus trajectory divergence in other reuse attempts; dedicated storage
preserves that observed boundary. See
`worklog/entries/20260816T152449.196381Z-lhl-qwen38-gfx1151-memory-parity-d82d61.md`. Q5 already
casts dead recurrent-output input in place; no persistent F16 weight copy is needed. **Clearing
action:** reproduce and isolate those alias failures while establishing a dense gfx1151 lifetime
plan covering all active producers, then alias stage-disjoint planes and remove only
proven-redundant allocation owners. Exercise non-anchor and 512/1K/4K lengths, pairwise live-range
overlap, full-attention/recurrent/FFN handoffs and teardown. Run
`tests/test_unit_qwen35_gguf_prefill_scratch_liveness.py` and
`tests/test_unit_qwen35_gguf_prefill_plan.py`, followed by public selected-route and cleanup
validation with R0219's packed-prefill integration. Preserve sole T16 residency and exact fallbacks.

### R0231 — Maple CUDA resident fixed-batch owner

Owner: `hipengine/generation/maple.py`.

Scope: `MapleCudaRunner.prefill_native()`.

Done when: Dependency: a CUDA resident-batch owner is not implemented. The CUDA package already provides
per-request batched QK/KV-write and attention leaves, but `MapleBatchRunner` constructs HIP buffers
and libraries, and the CUDA factory disables that resident owner. Do not flip the factory boolean or
route CUDA through HIP builders to bypass the missing implementation. **Clearing action:** implement
a CUDA owner with independent per-request span/KV arenas and CUDA libraries, wire existing batched
leaves and row-reuse head execution, and enable it through the public factory. Validate sparse
c2/c4/c8 admission, cancellation/reclaim, staggered refill, singleton-in-c8, full natural/heldout
trajectories and cleanup. On CUDA hardware run `HIPENGINE_RUN_CUDA_MAPLE=1 python -m pytest
tests/test_gpu_cuda_sm120a_maple.py -q`, then `python scripts/maple_public_batch_bench.py --backend
cuda_sm120a --out /tmp/maple-cuda-public-batch.json`. Extend those gates to cover the new owner
rather than counting serialized c1 calls as batching. This is implementation work, not a
missing-evidence gate; working c1 execution and its oracle remain unchanged.

### R0258 — Laguna grouped-IQ diagnostic chain and policy normalization

Owner: `hipengine/kernels/hip_gfx1100/__init__.py`.

Scope: `hipengine/kernels/hip_gfx1100/__init__.py`.

Done when: Decision needed: retain the older exact diagnostic variants and their launch-interface mappings,
or retire them with the comparison callers. Tests for H5Q/H5Z/H6D/H6F/H6I/H6P/H6Q/H6R still exercise
that chain. Folding the maps into a source selector must preserve the distinction between active-
expert and generic grouped interfaces; deleting interface entries alone can select a registered
function with the wrong operands. The role and quant maps do not allocate another weight payload or
workspace. **Clearing action:** inventory variant-specific test/profile callers, decide which
diagnostics remain supported, and remove only unused registrations, bodies, mappings and fixtures
together. Alternatively, normalize stable maps without changing selected function/interface
identities over all shape, role, missing-registration and backend cases. Run
`tests/test_unit_laguna_h6t_source_default.py`, `tests/test_unit_laguna_h6l_source_default.py`,
`tests/test_unit_laguna_h6c_source_default.py` and affected variant tests, then `python
scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1100 --max-tokens 8 --output
/tmp/laguna-grouped-public.json` with a bulk prompt that engages grouped prefill. The local
checkpoint dependency is recorded at R0249/R0250. Preserve Qwen GGUF's independent consumers of
shared IQ primitives, exact fallbacks and current defaults.

### R0259 — Laguna exact SWA and preappend policy normalization

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `tests/test_unit_laguna_h5r_runtime_policy.py`; `tests/test_unit_laguna_h6a_runtime_policy.py`; `tests/test_unit_laguna_h6a_source_default.py`.

Done when: Bounded backend role maps remain the supported representation, documented in
`reference/KVCACHE.md` "Laguna prefill append ordering"; no selector normalization is planned.
Host resolution, tile bounds and launch-interface checks pass, but the runner wiring is only
source-inspected. Remaining: public bulk-prefill selected function, output/state and teardown on
the applicable backend using R0249/R0250's command and missing checkpoint dependency. Include
starts 0/128/256 and unrelated positions, wrap/partial chunks, verifier/continuation and explicit
variants. Preserve absent-versus-empty and independent global/SWA override semantics, span-aware
primitive fallbacks and separately owned online arithmetic. The scratch requirement accessor now
returns the Boolean the runner consumes; device selection and state remain part of the public gate.

### R0264 — Laguna full-score long-attention diagnostic retirement

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `prefill_block_attention_hipblaslt=False/True`.

**Implemented separation:** immutable `LagunaLongAttentionPolicy` resolves construction choices;
the long-context profiler passes it explicitly. Route selection names bounded versus full-score
owners without changing legacy defaults, geometry, short-route precedence or generic fallback.
Host policy/runner/profile checks pass; lazy-owner tests cover independent allocation, same-width
reuse and incompatible-width rejection. See
`worklog/entries/20261106T143423.875861Z-lhl-r0264-lazy-owners-fa51c1.md`.

**Caller inventory:** the long-context profiler still supports both comparisons. The long-context
ceiling owns an explicit optional `block_context`; it and the SWA ceiling both call
`algorithm_counts`. Those APIs are live comparison dependencies, not unused full-score-only code.

**Remaining:** choose whether to retain full-score as a supported diagnostic or retire its
comparison/ceiling callers. Until that decision, preserve the working oracle and shared algorithm
APIs. Public long-prefill output/state/cleanup validation needs the checkpoint in R0249/R0250.

Done when: separate full-score diagnostics from long-attention execution policy before removal.
`prefill_long_attention_hipblaslt` and `LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT` currently enable
the bounded route too; deleting them as though they only enabled the old owner would disable
production long attention. **Decision needed:** retain full-score as a supported diagnostic, or
retire its comparison/ceiling callers. **Clearing action:** if retired, migrate the profile control
to bounded versus generic span-aware execution, preserve bounded width/context policy, and remove
only full-score lazy allocation/accounting/close plus unused controls. Audit shared algorithm
callers in both long-context and SWA ceiling harnesses before narrowing that API. Run
`tests/test_unit_laguna_long_context_profile.py` and `tests/test_gpu_laguna_kv_attention.py`; on
gfx1151 compare public long prefill output/state and cleanup with prompts both inside and beyond the
block boundary, including non-aligned tails, continuation and generic misses. Use R0249/R0250's
public command/checkpoint dependency. Keep rolling SWA, short attention, generic `KVLiveSpans`
fallback and bounded scratch.

### R0265/R0266/R0267 — Laguna attention geometry and layout controls

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `prefill_global_attention_chunk_size`.

Done when: complete the actual-weight public lifecycle matrix below. Retain explicit
constructor/profile controls as geometry policy and supported diagnostics. The profile
harness owns generic-versus-selected SWA and dense-contiguous comparisons through
`prefill_attention_diagnostic()` in its `ExitStack`; success and failure restore the
owner before close. Global attention rows are constructor-time resource sizing, not
an arithmetic comparison override. Keep backend width defaults, matrix-capacity
validation, bounded allocation and full `KVLiveSpans` fallbacks.

**Clearing action:** with R0249/R0250's checkpoint, run public dense and rolling/wrapped
prompts, non-aligned tails, continuation, verifier and shape misses on each applicable
backend. Confirm selected width, owner/layout, output/state and cleanup. Runner,
profile and diagnostic-scope tests cover nested failure, partial setup failure and
independent owners; they do not replace this actual-weight run. Global query geometry
is a resource choice, not an evidence allowlist; do not remove a supported width
merely because a benchmark did not cover it.

### R0269 — Laguna MoE scheduling diagnostic controls

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `scripts/laguna_long_context_profile.py`; `tests/test_unit_laguna_gguf_runner.py`; `tests/test_unit_laguna_long_context_profile.py`.

Done when: validate the supported constructor/profile diagnostics through the
public lifecycle matrix below. `scripts/laguna_long_context_profile.py` owns
explicit sequential/concurrent, after-router and low-priority comparisons.
`LagunaGGUFResidentSession` owns the resulting streams/events from construction
to close, including constructor-failure cleanup. Concurrency and priority are
resource-construction policy, not redundant live mutation interfaces; retain
the sequential and priority-0 controls and automatic queue-capacity fallback.
The after-router setter only changes launch ordering, not resource ownership.

**Clearing action:** with the checkpoint at R0249/R0250, compare public single-
and multi-row eager/graph execution, queue limits, cancellation and shutdown.
Confirm stream/event dependencies and cleanup, not just token equality. The
runner/profile unit bundle covers queue admission, priority and resource owners;
it does not replace the missing actual-weight public run.

### R0270 — Laguna selected-MMQ diagnostic modes

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `mmq64x32_d4_f32`.

Done when: validate the supported comparison modes through the public model gate
below. Retain D8, D4x2/D4x3 and independent down modes: publication isolation arms
select plain D8/D4 against wave-column defaults, the category benchmark owns direct,
adaptive/fused grouped-smallm and cumulative MMQ lanes, and primitive tests own the
residual-pass comparisons. These are distinct arithmetic/staging comparisons, not
redundant aliases. The rejected one-plane D4 gate/up selector remains removed;
its independent D4 down primitive is still supported.

R0281's grouped-down row uses the same `set_selected_down_mode()` owner. Its old
grouped-only default description does not replace current package policy. Preserve
strict fallbacks and down primitives shared with production. Category-session
configuration closes its newly constructed owner if selected-mode or cumulative-lane
setup fails; successful setup transfers ownership to the caller.

**Clearing action:** with R0249/R0250's checkpoint, run the applicable-backend public
model gate with non-aligned small/large rows, quant/shape misses, continuation and
cleanup. Selected-mode, category/publication and host MoE tests cover the diagnostic
interfaces; they do not establish actual-weight model parity. No reader/mode removal
is required while these comparisons remain maintained.

### R0277 — GGUF capture-inclusive graph policy and diagnostic interfaces

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `HIPENGINE_GGUF_DECODE_GRAPH`.

Done when: Decision needed: keep the eager override and comparison interfaces as supported controls, or
retire them after a current same-host, capture-inclusive public-route comparison. The bench defaults
to graph requests; MTP's default target-graph flag still has consumers in applicable verification
modes, while the resident preset selects bulk verification. Neither MTP mode is dead solely because
it differs from scalar AR. **Clearing action:** run `scripts/gguf_decode_graph_g5.py --help`, then
its state-parity and capture-inclusive comparison with an explicit local model, backend and
compiler-version file. Compare actual `LLM.generate()` eager and automatic routes at
below-/above-floor and unrelated horizons, including reset, cancellation and cleanup. Keep queue
count and KV policy matched; separate capture, first replay and steady replay. Inspect mode
consumers in `scripts/gguf_mtp_bench.py` before retiring its flags and migrate their comparison
tests. Run `tests/test_unit_qwen35_gguf_decode_graph.py`,
`tests/test_unit_qwen35_gguf_decode_graph_kv_floor.py`,
`tests/test_unit_qwen35_gguf_bench_metadata.py` and `tests/test_unit_gguf_mtp_bench_metrics.py`.

### R0279 — General compact-DMS undo beyond append-only cycles

Owner: `hipengine/runtime/qwen35_gguf_mtp.py`.

Scope: `hipengine/kvcache/dms.py`; `hipengine/kvcache/dms_device.py`; `_DMSStoreJournal`.

Done when: Remaining work: general eviction/compaction still copies owned extents to host. Choose a bounded
reversible mutation log or a resource-claimed device-local journal; do not replace it with an
unpriced prompt-proportional VRAM snapshot. Preserve the bounded append path and exact full fallback
until the successor covers expiry, movement and changing extent ownership. **Clearing action:**
implement the chosen owner in `hipengine/kvcache/dms.py`, `hipengine/kvcache/dms_device.py` and
`_DMSStoreJournal` in `hipengine/runtime/qwen35_gguf_mtp.py`. Restore K/V, INT8 scales, positions,
eviction/live counts, extent/ledger state and non-KV resident state at the same accepted-prefix
boundary. Run the append-journal, append-tail, append-integration, cycle-transaction,
journal-finalize and verifier-journal unit files plus `tests/test_gpu_dms_external_runtime.py` and
`tests/test_gpu_dms_int8_device_payloads.py`. Exercise public MTP rejection, partial/full commit,
following cycles, copy-failure retry, cancellation and shutdown. Use a trace to verify
copy/allocation ownership, not just token IDs.

### R0280 — DMS device ranking and decode control ownership

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `finalize_layer()`, `build_dms_exact_budget_eviction()`.

Done when: Remaining work: implement bounded deterministic device ranking and live-count/position/eviction
finalization with one owner for host observability, compact extents, prefix reuse and speculative
rollback. Preserve the one-pass ordering and layer staging/release lifetimes; the current host route
is the working reference, not an admission failure. Device packing alone is not this successor.
**Clearing action:** add CPU-oracle RED tests for exact masks including ties, non-finite rejection,
window edges, non-aligned lengths and per-head counts; then register device primitives and integrate
collector/decode owners. Run `tests/test_unit_kvcache_dms.py`,
`tests/test_unit_dms_one_pass_prefill.py`, `tests/test_unit_dms_layer_collector.py`,
`tests/test_unit_dms_layer_outer_lifetime.py`, `tests/test_unit_dms_prefill_cache.py` and the R0279
journal bundle, plus `tests/test_gpu_dms_external_runtime.py` and public DMS lifecycle/MTP requests.
Confirm masks, payload/scales/metadata, accepted-prefix rollback, continuation and cleanup on each
implemented composition. Profile named operations and transfers with resources priced before
retiring the host fallback.

### R0281-ATTN — Laguna prefill/decode variant interfaces

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `global_prefill_variant`; `swa_prefill_variant`; `swa_decode_variant`.

Done when: complete R0249/R0250's actual-weight public gate below. Retain these
explicit constructor diagnostics: grouped-down categories compare exact spans,
online qrow2 and wider qrow4 routes; publication arms isolate global/SWA prefill;
SWA leaf and ceiling harnesses select their own reference/candidate; target-AR
forwards requests and DFlash reports resolved target variants. Resolvers validate
implemented names, while omitted requests use independent backend package defaults.
No shared mutable selector or redundant alias requires migration here.

**Clearing action:** with the Laguna checkpoint, run public wrapped SWA,
dense/global and non-dense spans, empty/short/partial rows, continuation, verifier
rows and cleanup on each applicable backend. Runner, category, target-AR and DFlash
unit tests pass for the interfaces; primitive tests are separate from actual-weight
validation. Preserve independent exact fallbacks and do not collapse them into
online arithmetic during interface cleanup.

### R0281-PARO-WIDTH — PARO batch policy and diagnostic controls

Owner: `hipengine/generation/qwen35_paro.py`.

Scope: `scripts/qwen35_batch_retained_bench.py`, `hipengine/runtime/qwen35_paro_decode_groups.py`.

**Remaining work:** the two environment controls are not synonyms.
`HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS` also selects per-layer linear-output, attention and
projection defaults; `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` controls native decode.
Both still have benchmark/test callers. Choose supported separate policy controls or migrate those
callers to explicit diagnostic scopes before retiring their production environment readers; a rename
alone does not merge their behavior. The sampled-batch diagnostic is independent and must not be
substituted for the public per-request sampler.

Done when: separate these policies, migrate actual callers, then run
`tests/test_unit_qwen35_paro_decode_groups.py`, `tests/test_unit_qwen35_batch_partition_bench.py`,
`tests/test_unit_generation_qwen35_paro.py`, `tests/test_unit_qwen35_resident_batch_layout.py` and
batch-width loader tests. Exercise public native and serial c1/c2/c4/c8, sparse slots, staggered
admission, cancellation and cleanup, with the chosen path reported.

### R0281-D12 — Q5 constructor retirement public validation

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `use_q5_wave32x2_output`, `use_q5_wave32x2_query_gate`.

Done when: Remaining validation: use the local checkpoint dependency at R0249/R0250 to exercise
`LLM.generate()` on gfx1100 and gfx1151. Confirm selected Q5 attention-output/query-gate variants,
one-token and multi-row fallback behavior, continuation and allocation cleanup. Do not revive
constructor controls or gate implemented execution because the checkpoint is absent.

### R0281-Q5META — Q5 metadata comparison interface

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `laguna_target_ar_bench.py`, `use_q5_fixed_meta_output`, `use_q5_fixed_meta_query_gate`,
`use_q5_shared_fixed_meta`.

Done when: complete primitive and actual-weight binding checks below. Retain these
constructor/CLI diagnostics: output, query/gate and shared-pair metadata comparisons
select independent roles, not aliases. Both target-AR construction sites now use
one forwarding helper; effective result fields remain separate. Resolver and harness
tests cover each independent opt-out and backend-default preservation.

**Clearing action:** exercise query/gate with mixed quad disabled or unavailable
and the shared pair independently at runtime; preserve coefficient-publication and
pack8 routes, multi-row/shape/key fallbacks and backend defaults. Run the corresponding
primitive GPU comparisons and R0249/R0250's public generation/lifecycle gate.
Host Q5 control, runner, target-AR, SWAR-pair and mixed-pair unit tests pass.
The gfx1151 fixed-metadata singleton and BF16/F32 shared-pair primitive comparisons
also pass against independent references. Actual-weight route engagement still
requires the Laguna checkpoint; gfx1100 checks require that hardware.

### R0281-MIXED — Mixed-projection comparison levels

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `laguna_target_ar_bench.py`, `use_mixed_q5_q6_attention`,
`use_mixed_q6_fixed_meta_attention`, `use_mixed_local32_fixed_meta_attention`.

Done when: complete actual-weight public binding/lifecycle checks with the
R0249/R0250 checkpoint, then the gfx1100 device checks on that hardware. Retain
these constructor/CLI diagnostics: local32, fixed-Q6 and generic comparison levels
have distinct fallback semantics. Both target-AR construction sites forward the
three choices independently and report their effective values separately.

Host tests cover local32 hit/miss ordering with fixed-Q6 either enabled or disabled,
and direct fixed-Q6/generic selection with local32 off. The runner, target-AR,
mixed-pair and SWAR-pair unit bundle passes; gfx1151 mixed primitive comparisons
pass at role shapes, including the Q6/Q8 tuple and local32/fixed-metadata parity.
Preserve full quad disable, layer-47 tuple handling, multi-row and missing-key
fallbacks. Primitive evidence does not replace public route engagement.

### R0281-Q4HEAD — Scalar Q4 head comparison control

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `use_q4_lm_head_local32_fixed_meta`, `_project_and_sample()`, `_project_rows_last()`,
`_project_rows_all()`.

Done when: validate the supported scalar-Q4 diagnostic through the checkpoint-dependent
public continuation/lifecycle gate at R0249/R0250. The constructor owns the resolved
variant; `scripts/laguna_target_ar_bench.py` passes its CLI opt-out at both construction
sites and records effective selection. Keep that comparison: scalar full-logit decode
uses the registered variant, while bulk/verifier projection keeps its independent
selection. Runner, target-AR and dispatcher unit tests cover selection and missing-key
fallback; those host tests are not actual-weight Q4 output validation. Preserve exact
fallback bodies and backend capability behavior. Exercise actual Q4 full logits; do
not treat a Q6 producer-top1 run as exercising this Q4 path.

### R0281-IQ3 — c1 IQ3 schedule comparison interface

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `wave10_fused`, `wave4_reduce`, `serial_weighted`, `iq3_c1_down_schedule`.

Done when: complete exact primitive and public c1/multi-row continuation/cleanup
checks with the R0249/R0250 checkpoint. The constructor and kernel plan own the
resolved schedule; target-AR passes its explicit CLI comparison at both construction
sites and records the effective mode. Keep `wave10_fused`, `wave4_reduce` and
`serial_weighted` as supported independent diagnostics, including the wave10-miss
serial fallback and wave4 producer/reducer. Runner/target-AR host tests cover
selection and invalid/removed modes, not actual-weight arithmetic. Run the guarded
`test_laguna_iq3_*` schedule nodes in `tests/test_live_laguna_moe_gpu.py` in the ROCm
environment. Preserve unsupported/backend fallbacks, grouped-prefill consumers and
independent verifier tiles.

### R0281-SPLIT / R0339 — Split-attention and tile16 comparison controls

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `tests/test_gpu_laguna_kv_attention.py`.

**Interface decision:** retain explicit crossover/disable constructor and CLI controls as
supported resource and profiling policy. They are already owner-bound inputs shared with the
session allocator, not ambient environment transport. Preserve both harnesses/result fields,
workspace lifetime, invalid-combination errors, package defaults and registered score/token4
fallbacks. See `worklog/entries/20261105T221358.471073Z-lhl-laguna-attention-interface-7e6394.md`.

Done when: supply the Laguna checkpoint for R0249/R0250 public generation,
wrap/continuation, route attribution and allocation cleanup. Local host/fake-runtime,
split/tile16 CPU-oracle crossover and multi-row wrap checks pass on gfx1151; see
`worklog/entries/20261105T221706.111110Z-lhl-laguna-attention-validation-21ff11.md`.
The checkpoint is absent from the local model store; no runtime admission gate is added.

### R0281-FUSION / R0338 — Attention fusion and reducer comparison controls

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `tests/test_gpu_laguna_kv_attention.py`; `tests/test_unit_laguna_global_head_wave0_tree_runtime.py`.

**Interface decision:** retain each explicit constructor/CLI comparison as a supported
diagnostic. Preserve target-AR construction sites/result fields and long-context profile controls,
fusion dependency errors, resolved-variant reporting and unfused chains. These inputs are already
owner-bound, so no environment migration is needed. See
`worklog/entries/20261105T221358.471073Z-lhl-laguna-attention-interface-7e6394.md`.

**Validation remaining:** supply the absent Laguna checkpoint and run R0249/R0250
public continuation/lifecycle with selected-route attribution. Local host/fake-runtime,
head/KV and split-gate fused/unfused, wave-local/shared-statistics and wrap/multi-row
primitive checks pass on gfx1151; see
`worklog/entries/20261105T221706.111110Z-lhl-laguna-attention-validation-21ff11.md`.

Done when: Wave-local reduction requires split attention and split-gate fusion. Gate-fusion comparisons must
disable wave-local too; preserve named dependency errors and the shared-statistics fallback.

### R0281-IQ2 — c1 IQ2 grid comparison control

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `use_iq2_grid64`.

Done when: complete exact primitive comparisons and the R0249/R0250 public
continuation/lifecycle gate with c1 route attribution. Retain the constructor/CLI
compact-grid comparison as a supported diagnostic: the constructor resolves it into
`LagunaMoEKernelPlan.c1_selected_gate_up_routes`; target-AR passes it at both
construction sites and reports the effective value. Host route tests verify that
on/off choices leave grouped-prefill and multi-row keys unchanged. Preserve raw-IQ
allocation ABI and registered fallbacks. Do not reintroduce rejected local64 runtime
controls. Actual-weight validation requires the Laguna checkpoint, not additional
selector migration.

### R0281-DFLASH — Verifier tiles and shared true-AR mismatch

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `iq3_selected_down_tile`, `laguna_dflash_category_bench.py`.

Done when: Unresolved failure: the historical gfx1151 packet
`benchmarks/results/2026-08-20-gfx1151-laguna-dflash-iq3-tile4-rejected.json` records identical
true-AR mismatches under tile1 and tile4, not a tile4 trajectory regression. It does not establish a
current-stack verdict. **Clearing action:** with the local target checkpoint noted at R0249/R0250
and its DFlash drafter available, run `python scripts/laguna_dflash_category_bench.py
"$LAGUNA_MODEL" "$LAGUNA_DRAFTER" --backend hip_gfx1151 --direct-gguf --iq3-selected-down-tile 1
--output /tmp/laguna-dflash-current.json`. Preserve the fixed-horizon equality and request-reset
checks; if the mismatch reproduces, trace the first divergent verifier/prefill/commit state rather
than attributing it to the output tile. Then run the public blocking/streaming DFlash gate and
verify continuation, cancellation and allocation cleanup.

Keep the working gfx1100 tile route and exact tile1 fallback. Decide whether the explicit tile
control remains a supported diagnostic or moves with its category caller to a scoped verifier
policy; do not add identity-based admission or disable working execution because a measurement is
absent.

### R0281-PARO-ARGRAPH — gfx1151 greedy graph mismatch

Owner: `hipengine/runtime/qwen35_paro_runner.py`.

Scope: `capture_decode_graph()`.

Done when: Remaining work: trace and repair the first graph/eager state or arithmetic difference before
removing the gfx1151 eager policy. Keep the working eager route, gfx1100 graph route and explicit
diagnostic capture; absence of a measurement on another architecture is not itself a capability
failure. **Clearing action:** run an eager/graph/eager control with identical inputs, then compare hidden,
Conv/GDN and live KV at each transition with identical inputs. Extend the repair gate to long
replay, reset/reuse, cancellation and shutdown. The fixture CLI in
`scripts/qwen35_decode_graph_fixture_gate.py` compares generated IDs and final logits, not all
persistent state; it also labels its backend statically, so verify the runner's resolved backend
rather than trusting that field. Run the graph-policy nodes in
`tests/test_unit_generation_qwen35_paro.py` and `tests/test_unit_qwen35_resident_batch_layout.py`,
then exercise public `LLM.generate()` with the repaired route confirmed and allocations drained.
Evaluate capture-inclusive wall separately before an automatic performance choice.

### R0281-PARO-RAGGED — Ragged packed prefill and diagnostic ownership

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `per_segment_ragged_exact`, `packed_segments`, `packed_varlen`;
`HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR`,;
`HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN`.

Done when: Remaining work: repair the original packed ragged difference under an explicit diagnostic owner,
or keep per-segment execution as the supported ragged algorithm. Decide separately whether the live
comparison controls stay supported or move to an owner-scoped interface. Do not delete their
callers, exact fallback, equal-length paths or primitive implementations as stale release-window
cleanup. **Clearing action:** first run `scripts/qwen35_batch_shrinking_correctness.py --model
"$PARO_MODEL" --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --batch-size 3
--prompt-lengths 8,9,11 --steps-per-width 1 --max-layers 40 --max-sequence-length 1024
--survivor-slot 1 --decode-execution serial --json /tmp/paro-ragged-current.json` on the intended
backend and verify resolved device and prefill metadata. This checks the working fallback, not the
packed candidate. Add a test-only packed ragged bypass before comparing identical input trajectories
and locating the first unequal hidden, Conv/GDN or live-KV state. Cover c8, sparse slots,
continuation, EOS/cancellation and cleanup on each applicable backend. Run
`tests/test_unit_qwen35_resident_batch_layout.py`, `tests/test_unit_qwen35_paro_serial_bridge.py`
and `tests/test_unit_qwen35_batch_partition_bench.py`, then public ragged `LLM.generate()` requests
with the selected path confirmed.

### R0281-GGUF-REPACK — Decode layout control and comparison callers

Owner: `hipengine/loading/qwen35_gguf_policy.py`.

Scope: `resolve_gguf_decode_repack()`, `use_wmma_prefill`, `use_gemv_decode`;
`HIPENGINE_GGUF_DECODE_REPACK`.

Done when: complete public layout/lifecycle validation on gfx1100. Retain explicit
API/environment layout selection as load-time resource policy and diagnostics:
`resolve_gguf_decode_repack()` already gives admission and materialization one
meaning with explicit-value precedence. Semantic and true-AR comparison harnesses
still exercise both values. No reader retirement is needed for these maintained
comparisons. Preserve per-tensor capability, byte planning, cleanup and observed
raw-fastpath containment; never use artifact/measurement admission allowlists or
change activation math as a repacking side effect.

**Clearing action:** exercise actual public default/raw requests with compatible
quants on gfx1100 and confirm layouts, selectors, no duplicate raw/T16 residency,
bounded allocation and cleanup. The decode-repack semantics/dispatch/fastpath,
GDN semantic-gate and benchmark-metadata unit bundle passes. Local gfx1151 public
Q4_K_S default/raw requests produce distinct planned/resident layouts, no raw/T16
allocation pair on one weight, matching short greedy output and clean tracked
allocation teardown. This is local lifecycle evidence, not a full model-quality
or cross-backend parity claim.

### R0281-CONV — Convolution schedule selector and isolation profiles

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE`.

Done when: finish gfx1100 primitive/public generation validation with route engagement
and cleanup. Keep the schedule selector as a supported diagnostic: the full-stack
runner caches its resolved registered callable, and the maintained sweep's baseline
isolation profile sets it before use. Host policy/registry/profile tests and gfx1151
primitive output/final-state checks pass. The public ordinary-prefill test
`tests/test_live_gguf_conv_prefill_owner.py` confirms both baseline/tiled callable
selection, repeated output and allocation teardown on gfx1151. Packed NextN prompts
use a different convolution entry and are not this selector's engagement gate.
Preserve backend auto choices, exact baseline/segmented/state-capture fallbacks and
the tiled gfx1151 implementation. No environment-reader retirement is required while
this comparison remains supported. Do not turn cleanup into gfx1100 promotion or
remove kernels based on old schedule names.

### R0281-SELECTED-PAIR — Selected-MoE pair-reuse control retirement

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE`, `HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE`,
`HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE`.

Done when: migrate all on/off diagnostic callers to explicit scoped policy before removing
the compatibility environment readers. `selected_pair_diagnostic(pair=..., down=...,
q6_down=...)` overrides selection within one context; an unspecified field uses its
request-local package floor. The batch-route gate now uses that scope without modifying
these environment values and reports scoped policy separately from ambient metadata.
Preserve per-lane fallback kernels, backend defaults and the separate request-local
small-row scope.
The three packed width scopes now use context-local token/reset storage rather than
process globals; overlapping-thread and nested-failure tests cover isolation. Local
public C8 MoE generation confirms all three package-floor scopes engage and restore
with clean tracked allocation teardown. This does not complete reader migration or
the full callable/neighbor-width matrix below. **Clearing
action:** run `tests/test_unit_qwen35_gguf_compact_moe_gemv_routing.py` and
`tests/test_unit_execution_profile_gguf_batch_route_gate.py`; migrate the gate and `docs/ENVS.md`
with any reader removal. A runtime migration also needs the primitive pair-reuse parity tests and
public C8 MoE generation on both intended backends, with selected callables, neighboring/non-C8
widths and cleanup recorded. A release-window deadline alone cannot replace these contracts.

### R0281-MTP-CAP — Packed MTP batch policy and caller separation

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `_MTP_SERVING_PREFILL_MAX_SLOTS`, `_MTP_SERVING_VERIFY_MAX_SLOTS`,
`_MTP_WARMUP_BATCH_MAX_SLOTS` in `hipengine/generation/qwen35_gguf.py`.

Done when: validate the independently named bounded prefill, verifier and warmup
policies through the public route. All three limits remain four; the shared
constant has been removed without widening any path. Unit coverage pins chunk
order, singleton-tail avoidance and verifier independence from the other limits.
Do not change prompt-prefill admission as a side effect of verifier tuning. **Clearing action:** run
`tests/test_unit_gguf_mtp_serving_chunk_policy.py` and the batch-verifier, packed-prefill and MTP
warmup nodes in `tests/test_live_generation_qwen35_gguf_sampling.py`. A runtime policy change also
needs public explicit-MTP requests at neighboring, unrelated and larger widths, selected
chunk/owner/stream traces, exact commit/rollback and refusal-after-progress checks, repeated
requests and cleanup. Compare complete-wall and latency on the intended backend against the bounded
policy before choosing a tuning result. Keep per-slot unsupported-shape fallback and request-local
state isolation.

### R0281-PARO-NATIVE — Provider graph and selected-commit control migration

Owner: `hipengine/runtime/qwen35_paro_runner.py`.

Scope: `hip_gfx1100/w4_paro/native_v1_target_graph`, `COMMIT|UPDATE_CURSORS`;
`HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH`, `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT`.

**Remaining decision/work:** replace the two process-wide selectors with profile-owned
submission/commit variants and an explicit diagnostic comparison, or retain supported controls with
concrete ownership. Preserve provider-specific repair, result metadata and capture bindings. The old
release-window trigger and cross-provider speed prerequisite do not specify this migration. A
missing backend registration is an implementation gap, not evidence that its hardware cannot execute
the wrapper; evaluate/register supported peers independently, without identity allowlists or
missing-benchmark admission gates. Complete proposal ownership is separate work and does not block
existing target graphs.

Done when: exercise both submission/commit variants in
`tests/test_unit_qwen35_resident_batch_layout.py`,
`tests/test_unit_qwen35_paro_verify_graph_cache.py` and provider-launcher nodes in
`tests/test_gpu_native_spec_cycle_graph.py`. A migration needs real PARO and DFlash requests
confirming selected stages, active/inactive and unsupported shapes,
hidden-capture/accepted-row/cursor ownership, independent graph buckets, repeat output, fault
rollback and teardown. Keep the public post-submission failure nodes in
`tests/test_live_paro_mtp_graph.py`; they launch real HIP graphs with injected result failure and do
not claim native-C++ registration on an unregistered peer. Apply numerical gates only if arithmetic
changes.

### R0281-AR-STREAM — Stream decode control and serial diagnostic migration

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `HIPENGINE_GGUF_AR_STREAM_DECODE`.

**Remaining work:** provide an explicit resolved serial/stream/packed comparison control or
designated serial oracle, then migrate writers, route metadata, cleanup tests and the env reader
together. Preserve singleton and missing-hook fallback, packed-owner flush before per-slot work,
launch-before- read ordering and stream destruction. Once any chunk is launched, do not retry the
group as a serial fallback on a launch/result exception.

Done when: run `tests/test_unit_gguf_ar_decode_control.py`,
`tests/test_unit_gguf_concurrency_baseline.py`, `tests/test_unit_gguf_live_server_bench.py`,
`tests/test_unit_server_f1_concurrency_bench.py`,
`tests/test_integration_gguf_production_load_gate.py` and
`tests/test_unit_gguf_sampled_api_gate.py`, plus stream/packed/scalar and above-eight chunk nodes in
`tests/test_live_generation_qwen35_gguf_sampling.py`. Reader migration needs real default, explicit
serial and stream requests at widths below/above eight and an unrelated width, ragged/shrinking
groups, selected route, repeat output and teardown.

### R0281-MTP-DRAFT-EXECUTOR — Compatibility draft/verifier worker lifetime

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `_try_draft_mtp_serving_slots_streams`, `_try_verify_mtp_serving_cycle_chunks_streams`;
`HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT`, `HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY`;
`_close_mtp_serving_slots`.

**Remaining work:** decide whether to retire compatibility execution through the existing loop
extraction or give its worker pool a bounded request/owner lifetime. Do not add another global
switch. A persistent pool must retain per-slot seed/KV ownership, ordered results,
deadline/cancellation handling, complete future draining, failure cleanup and shutdown before its
device resources are released. The current per-call draft pool now collects every
submitted future on failure and frees successful sibling snapshots before raising;
submission failure also drains already launched work. The chunked verifier now
collects every submitted future after worker or submission failure, preserving the
first error and attaching later worker failures as notes. Pool shutdown already
waited for execution, but did not previously retrieve every result. Neither pool
reuses workers.
Profile this path first if claiming pool reuse improves complete
wall; old server throughput comparisons do not establish current worker overhead or a resident MTP2
benefit.

Done when: run stream-draft, chunked-verifier and MTP2 nodes in
`tests/test_live_generation_qwen35_gguf_sampling.py`,
`tests/test_unit_gguf_mtp_serving_chunk_policy.py` and the GGUF MTP2 seam tests. Worker-lifetime
changes need direct compatibility requests and public `LLM.generate()`/serving validation
separately, recording which route actually runs, repeat output, shrinking groups,
deadline/cancellation/failure, drained futures and zero leaked streams/allocations. Do not label a
compatibility-only worker change as a resident scheduler improvement.

### R0281-INT8-POLICY — Legacy context and layout diagnostic controls

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `tail4_hadamard_group32`; `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG`.

**Decision needed:** retain explicit diagnostic controls, or migrate them to a scoped policy that
distinguishes a recorded-failure override from storage layout selection. Keep actual rejection
records and capability/resource errors. Do not replace the current kernel declaration with
model-name, file-hash or measured- artifact admission lists.

Done when: enumerate both generator and resident-runner readers, migrate
`scripts/gguf_server_allocation_probe.py`, `scripts/qwen38_int8_server_context_soak.py` and the
mixed-KV/correctness callers, then remove the obsolete readers and update ENVS. Run the
KV-capability, INT8-policy, packed-verifier-layout, INT8-MTP and per-layer-oracle unit files,
including unseen artifacts and contexts below, above and unrelated to the mirror threshold. Exercise
public direct/mirrored requests, explicit diagnostic overrides and genuine capability refusals on
both HIP backends, tracing storage/scales, graph/MTP engagement, cancellation, refill and cleanup.
Kernel-arithmetic changes require their independent quality and state gates; control migration alone
must not change arithmetic or defaults.

### R0281-PARO-INT8-PREFILL — Oracle/streaming resource-policy ownership

Owner: `hipengine/runtime/qwen35_paro_runner.py`.

Scope: `HIPENGINE_QWEN35_INT8_PREFILL_`.

**Decision needed:** retain operator resource overrides, or move them into the resolved
prefill/resource plan and update result metadata. A faster memory-safe alternative is separate
kernel work, not a prerequisite to exercising either implemented route. Preserve explicit
invalid-value errors, mixed-layout policy, shared scratch/oracle lifetimes and capability fallbacks.

Done when: migrate both ordinary and packed prefill metadata/readers, workspace-policy
tests and any external diagnostic callers together; update ENVS. Run
`tests/test_unit_qwen35_prefill_workspace_policy.py`,
`tests/test_unit_qwen35_resident_batch_layout.py` and the native mixed-KV suite contracts. Exercise
public uniform and mixed requests with oracle/direct selection, short/long/ragged prompts, resource
rejection, reset/cancellation and cleanup. Verify actual attention sources and transient oracle
owners.

### R0313 / R0314 / R0315 / R0316 / R0318 / R0320 — Laguna legacy MMQ selector migration

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `mmq128x32_d8_f32_rowvec`, `mmq64x32_d4_f32_rowvec`, `mmq128x32_d8_f32_wavecols`,
`mmq64x32_d4_f32_wavecols_q4`, `mmq64x32_d4_f32_wavecols_direct_q4`,
`mmq128x32_d8_f32_wavecols_direct_doublebuf`.

**Blocker:** selector retirement must migrate those numerical comparisons to explicit primitive
ownership and validate the user-facing route. The configured
`~/models/gguf/laguna-s-2.1-Q4_K_M.gguf` artifact and the live fixture's
`/models/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf` are unavailable on this host. This is a local
validation-resource blocker, not an admission restriction or a reason to disable the existing
default.

Done when: provide an implemented Laguna model artifact, remove the legacy names from
resolver/fusion/config tables, migrate the generalized comparison harness's default pair, and
preserve the low-level row-vector and double-buffer references in the Q4/Q6 GPU fixture. Run
selected-MMQ, runner, publication and grouped-prefill contracts plus the full MoE quant-oracle
nodes; then exercise `LLM.generate()` at short, boundary-neighbor and unrelated prompt sizes,
recording selected defaults, output parity and cleanup. Keep historical publication normalization
and the separate Q6 row-vector implementation.

### R0319 — Laguna serial compaction comparison ownership

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `scripts/laguna_group_compact_parallel_ab.py`.

**Host migration complete:** the A/B harness uses `group_compact_diagnostic` for warmup
and measured calls; the persistent session setter is removed. Serial/parallel primitive
arguments remain explicit numerical oracles, and historical result labels are unchanged.
Compaction/source-default/runner host checks pass (119 tests); synthetic full-MoE Q4/Q6
numerical contracts pass on gfx1151. See
`worklog/entries/20261106T122018.828940Z-lhl-r0319-compaction-setter-681e08.md`.

**Remaining dependency:** public default-selection metadata/output/cleanup validation
needs an accessible Laguna checkpoint; configured local benchmark/live artifacts are absent.
Do not remove serial primitive references or repeat completed session migration.

Done when: migrate the A/B harness, numerical fixtures and session setter together; run
compaction comparison, source-default, runner and full MoE numerical contracts, then public
`LLM.generate()` with default selection, metadata/output equality and cleanup checks. Do not
silently route a requested serial comparison through parallel after retiring the selector.

### R0375/R0376 — gfx1151 pack8 comparison-control ownership

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK`, `HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL`.

**Decision needed:** migrate active cumulative/threshold/probe consumers to explicit diagnostic
policy before deleting either environment control. Keep tile8x8, singleton WMMA and unfused SiLU as
low-level numerical references. Do not change backend admission or arithmetic as a side effect of
retiring comparison controls; broader shape selection is a separate capability task.

Done when: run `uv run pytest tests/test_unit_qwen35_08b_threshold_environment.py
tests/test_unit_qwen35_08b_cumulative_semantic.py tests/test_unit_qwen35_08b_threshold_artifact.py
tests/test_unit_gguf_linear_dispatch.py`, then exercise the migrated policies through the cumulative
packet and public `LLM.generate()` on HIP. Check current/older-role engagement, single/paired/fused
dispatch, 511/512/513 and unrelated lengths, full logits/state, repeated role order and cleanup
before removing the env readers.

### R0373/R0374 — Maple fused decode efficiency and selector retirement

Owner: `hipengine/runtime/maple.py`.

Scope: `HIPENGINE_MAPLE_FUSE_MOE`, `HIPENGINE_MAPLE_FUSE_QKATTN`.

**Decision needed:** remove the experimental runtime fusion routes while preserving primitive
references, or profile and repair the efficiency gap before promoting a route and retiring its
selector. Treat the two fusions independently. Do not delete the mandatory unfused primitive chains,
and do not reintroduce the serial-expert down/residual design unchanged.

Done when: run `uv run pytest tests/test_unit_maple_graph_selection.py
tests/test_gpu_maple_runtime.py tests/test_gpu_maple_ternary_kernels.py
tests/test_gpu_maple_attention_kernels.py`, then use the Maple profiling and c1 benchmark tools for
same-host eager/graph comparisons. Include both individual selector changes and their combination,
actual public `LLM.generate()` engagement, full-logit/state parity for strict variants, reset/reuse
and graph/workspace teardown. A replacement arithmetic variant uses its execution-profile gate
rather than inheriting strict bit parity.

### R0368 — LC-D3 global geometry diagnostics and unused repair separation

Owner: `scripts/laguna_global_issue_diagnostic.py`.

Scope: `scripts/laguna_global_issue_diagnostic.py` — LC-D3 global geometry diagnostics and unused repair separation.

**Local cleanup complete:** retain explicit geometry comparisons. The unused repair kernel,
reverse-association probe, aliased repair-mask writes and misleading leaf repair counters are
removed. Partial-output layout and forward accumulation remain unchanged. Both compensation
modes and normalized/deferred merges have matching pre/post context and gated-output hashes
at live lengths 3997, 4095, 4096, 4097 and 6131 on gfx1151. Host and device suites pass;
see `worklog/entries/20261105T224402.566213Z-lhl-r0368-repair-2b6a03.md`.

Done when: supply the absent Laguna checkpoint and run public generation with route attribution,
teacher logits, KV metadata, reset/reuse and teardown. Preserve broader production numerical
contracts; unchanged primitive hashes do not establish a new profile or full-model result.

### R0367 — Selected gate/up halfdot and exact-teacher ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `use_selected_halfdot_decode`.

**Interface decision:** retain the constructor rollback and maintained leaf/correctness
selectors as supported diagnostics. The constructor resolves backend defaults once per session;
paired singleton routing consumes that choice, while wider rows keep the exact fallback.
Candidate/repeat sessions and the explicit exact-teacher session remain independent. Preserve
both registered implementations; no arithmetic or default changes.

Done when: supply the absent Laguna checkpoint/oracle and run
`uv run python scripts/laguna_gguf_correctness.py --backend hip_gfx1151 --selected-halfdot`,
then public default/exact generation with aligned teacher logits, selected routes, reset/reuse
and teardown. Local 117-test host bundle and natural/interleaved GPU primitive oracle pass
on gfx1151; see `worklog/entries/20261105T223602.134441Z-lhl-r0367-owner-94a7e0.md`.

### R0354 — Paired expert cache and ordinary-layout migration

Owner: `hipengine/loading/laguna_gguf_materialize.py`.

Scope: `tiles_dual`.

**Remaining work:** the `laguna-repacked-v1` writer still persists independent `tiles` payloads.
`_materialize_q4_t16_dual_pair` must read and interleave those at load time. Add a versioned paired
cache payload that replaces both stored matrices, with no duplicate pair, while retaining
raw/partial cache loading. Migrate the leaf scripts that directly read manifest `entries` and
ordinary `tiles` before retiring legacy cache acceptance or the public
`use_q4_expert_t16_dual_interleaved=False` / `--ordinary-q4-expert-t16` controls. The natural-tile8
comparison also intentionally requests ordinary residents; its replacement must preserve that
distinct reference route (R0341).

Done when: bind writer/open/reload and direct-manifest consumers to paired ownership,
validate both cache versions and raw/partial-source loads, then run paired/ordinary primitive
oracles and public `LLM.generate()` decode and prefill with byte/state/resource checks. Remove the
ordinary seam only after its callers migrate. Local ROCm/Laguna resources for the real numerical and
public runs remain unavailable as recorded by R0340/R0273.

### R0353 — SWA value-tail reference ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `LLM.generate()`.

**Remaining work:** separate diagnostic-only instantiations/wrappers from the production build
without removing the shared template or live predecessor oracles. Keep the generalized leaf harness
and its parser/result readers until replacement comparisons select the same symbols.

Done when: run value-tail reference, SWA issue and KV attention host contracts. On ROCm,
execute standalone/local512/V-stage128/dual-tail leaf pairs and primitive oracles, including
allocation/event failure cleanup, then public `LLM.generate()` confirming the default compound route
and full state/resource parity. Use the configured ROCm environment; the Laguna checkpoint
remains a dependency for the model-backed checks in R0340/R0273.

### R0349–R0351 — Global exp/mixed32 diagnostic migration

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `LLM.generate()`.

**Host migration complete:** profiler arms select explicit registered references through
`global_issue_arm`; the three persistent setters are removed. Diagnostic scopes restore one
cache-owned reference slot without changing production prerequisite or successor flags. Generic
capacity and peer-backend routes remain available.

**Validation:** global issue/profiler/runner contracts pass, as does the maintained primitive
CPU-oracle case on gfx1151. Device owner-versus-registered-primitive checks cover both arms of
each pair across live126/127/576/3999/4000/4001 and capacity4095/4096/4097. These dispatch checks
are distinct from independent arithmetic oracles. See
`worklog/entries/20261106T133146.186131Z-lhl-r0349-boundaries-ce03f0.md`.

**Remaining dependency:** public `LLM.generate()` selected-symbol, full-state and teardown
validation needs an accessible Laguna checkpoint; configured local artifacts are absent.
ROCm gfx1151 is available. Reference primitives remain in production builds pending that check.

Done when: run global issue, profiler, runner and KV attention contracts; on ROCm,
exercise each pair against its registered primitive oracle on both sides of split/live4000 and
capacity-4096 boundaries. Then run public `LLM.generate()` with the default successor selected,
checking complete state, selected symbols and resource teardown before retiring controls or moving
reference primitives out of production builds.

### R0343–R0348 — SWA issue-schedule diagnostic migration

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `LLM.generate()`.

**Host migration complete:** profiler comparisons use `swa_issue_arm` to select explicit
registered references. Six persistent setters and temporary production-flag mutation are removed;
nested and failed scopes restore one cache-owned reference slot. Unsaturated, generic-shape and
peer-backend routes retain their production policy.

**Validation:** affected host contracts pass. The maintained primitive CPU oracle passes on
gfx1151, and owner-versus-registered-primitive checks cover all six pairs and both arms at
positions382/510/511/777 (unsaturated, saturated and wrapped). These dispatch checks are distinct
from independent arithmetic oracles. See
`worklog/entries/20261106T134323.488026Z-lhl-r0343-device-cf6afa.md`.

**Remaining dependency:** public `LLM.generate()` default-successor selected-symbol, full-state
and teardown checks need an accessible Laguna checkpoint; configured local artifacts are absent.
ROCm gfx1151 is available. Reference primitives remain in production builds pending that check.

Done when: run SWA issue-scope, KV attention, runner and profiler contracts. On ROCm,
execute each repaired pair against its registered primitive oracle, including
unsaturated/saturated/wrapped rings and resource cleanup; then public `LLM.generate()` with the
default successor route. Confirm symbols, full-state parity and teardown before retiring the six
public controls or moving reference-only primitives out of production builds.

### R0341 — Laguna natural tile8 diagnostic ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `LLM.generate()`.

**Host migration complete:** the profiler selects each comparison arm with
`selected_natural_tile8_diagnostic`; persistent setter and constructor override are removed.
Backend package defaults and ordinary natural16/tile8 reference bodies are unchanged.
Result metadata names the scoped owner and preserves the historical comparison basis.
Host tile8-comparison, runner and profiler contracts pass (127 tests); see
`worklog/entries/20261106T121454.078780Z-lhl-r0341-tile8-constructor-055760.md`.

**Remaining dependency:** actual-weight/full-state and public ordinary/dual validation
needs an accessible Laguna checkpoint. The configured benchmark and live-fixture paths
are absent locally; ROCm is available on gfx1151. Do not repeat completed host migration.

Done when: run tile8-comparison, runner, profiler and MoE numerical contracts; execute the
repaired paired comparison on a ROCm host with an available Laguna artifact. Confirm distinct
selected symbols, full-state parity and cleanup, then public `LLM.generate()` on default dual
residency and ordinary diagnostic residency. Migrate result readers with the diagnostic owner; do
not silently compare the same interleaved route twice.

### R0340 — Source-F16 decode comparison migration and shape validation

Owner: `hipengine/runtime/f16_weight_linear.py`.

Scope: `HIPENGINE_LAGUNA_F16_DECODE`.

**Comparison migration implemented:** `f16_decode_strategy(...)` supplies context-local
selection, and the long-context profiler no longer mutates the process environment for
baseline/candidate comparisons. The environment control remains a compatibility input.
See `worklog/entries/20261106T112718.560230Z-lhl-r0340-decode-scope-b2f8f1.md`.
Real public model validation still needs the Laguna artifact (R0273); GPU leaf shape
validation can run independently of model weights.

Done when: run shape-policy, F16 projection, backend, runner and profiler tests. On a ROCm
host, execute generic-K widths below/above/unrelated to the fixed set, single/triple and alternate
launch widths against the GEMV oracle. Run public `LLM.generate()` with an available artifact and
verify the intended variant, output/continuation and clean teardown. Preserve generic registered
owners and explicit capability failures during selector migration.

### R0336 — Laguna packed-output comparison migration

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `LLM.generate()`.

**Comparison migration implemented:** the dedicated harness uses owner-local
`packed_attention_diagnostic(...)` for warmup and measured runs, without package-global
mutation or a constructor override. Backend defaults and generic unpack/gate primitives
are unchanged. Host route, restoration and cleanup checks pass; device/full-state and
public validation still require the absent Laguna artifact (R0273/R0274).

Done when: run packed-output, producer-retirement, runner and long context profile host
checks; execute the repaired paired comparison with an available Laguna artifact and confirm
distinct packed/generic gate routes, full-state equality and cleanup. Exercise public
`LLM.generate()` with varied lengths including partial tiles. Keep paired result readers on the
explicit owner scope without changing backend defaults.

### R0333–R0335 — Laguna Q4 P8 and sum comparison ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `LLM.generate()`.

**Host migration complete:** Q4 sum comparisons use `q4_activation_sums_diagnostic`;
selected gate/down references use `selected_moe_diagnostic`. Persistent setters and the
unused Q4 sum constructor override are removed. Generalized, rows64, grouped-prefill,
category/cumulative-oracle, matrix and profiler callers use restoring scopes. Production
P8 thresholds, non-P8 sub-512 bodies, leaf sidecar producers/consumers and result labels
are unchanged. See
`worklog/entries/20261106T131437.754459Z-lhl-r0333-selected-setters-3e855a.md`.

**Remaining dependency:** generalized actual-weight P8/non-P8 and sums comparisons,
plus public varied-chunk selection/full-state/cleanup checks, need an accessible Laguna
checkpoint; configured local artifacts are absent. Host affected/leaf checks and synthetic
full Q4/Q6 MoE numerical cases pass on gfx1151.

Done when: run Q4 P8 policy, selected-MMQ-mode, generalized-comparison, runner and full
MoE numerical contracts after the coordinated migration. Run the generalized P8/non-P8 and
sums-off/on comparisons with an available Laguna artifact, then public `LLM.generate()` across
varied producer chunk sizes. Confirm intended shape/default selection, exact full state and cleanup
before retiring positive mode strings or the public sum setter. Do not delete the shared leaf
harness or silently map an explicit reference request to P8.

### R0331 — Laguna Q6 prefetch/sum comparison ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `q6_wmma_prefetch_weight`, `q6_wmma_prefetch_activation`, `q6_precomputed_activation_sums`.

**Host migration complete:** Q6 prefetch/sums use the validated restoring
`q6_prefetch_diagnostic` bundle. The three setters and constructor overrides are removed;
production settings derive from resident layout and backend capability. Generalized and
planar A/B arms preserve producer/consumer selection; non-planar staging harnesses use
layout-derived defaults. Direct primitive references and historical mode labels remain.
See `worklog/entries/20261106T123709.064814Z-lhl-r0331-dictionary-callers-cee3c2.md`.

**Remaining dependency:** generalized actual-weight comparison and public selection/state/
cleanup validation need an accessible Laguna checkpoint; configured local artifacts are absent.
Host runtime/WMMA/layout/profiler checks and 13 activation/full-MoE GPU cases passed on gfx1151.

Done when: run runtime-mode, WMMA-reference, activation/layout, runner and full MoE
numerical contracts; then execute the generalized comparison and public `LLM.generate()` with an
available Laguna artifact. Confirm matching sum producer/consumer, route/default selection, exact
full state and cleanup before retiring the three public controls. Keep non-prefetch reference bodies
needed by other declared shapes and comparisons.

### R0330 — Laguna Q6 integer-WMMA reference ownership

Owner: `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_q8_1_selected_prefill.py`.

Scope: `integer_wmma`, `wmma_hoist_activation`.

**Remaining work:** separate production specialization from explicit reference ownership across the
shared wrapper and generalized leaf before collapsing controls. Preserve packed-dot/non-planar
bodies and the numerical comparisons. Actual-weight/full-model validation shares the local Laguna
artifact resource blocker recorded in R0313–R0320.

Done when: migrate reference entry points and leaf callers together; run WMMA
reference-control, activation/layout comparison and full Q6 primitive oracles, then the
actual-weight generalized leaf and public `LLM.generate()`. Confirm default symbol, byte-layout
agreement, exact outputs and cleanup. Changed arithmetic requires its execution-profile gate;
preserve historical result readers when retiring mode labels.

### R0329 — Laguna source-F16 boundary comparison ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `f16_boundary_diagnostic(...)`, `scripts/laguna_f16_boundary_ab.py`,
`hipblaslt_range_direct`.

**Comparison migration implemented:** warmup and measured runs use an owner-local
restoring diagnostic scope; the persistent setter is removed. Fused/separate-cast
arms and production defaults are unchanged. The local Laguna artifact resource blocker
in R0313–R0320 still prevents public/full-model validation; it does not disable the path.

Done when: migrate the comparison without collapsing its arms, run F16
boundary-comparison, runner and hipBLASLt contracts plus primitive BF16-boundary equality/full-model
oracles, then public `LLM.generate()` across compatible and fallback projection routes. Confirm
selected producers/casts, state equality and teardown. Keep the enabled capability and strict
unfused chain with the scoped diagnostic owner.

### R0327 / R0328 — Laguna permute/planar comparison migration

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `q6_qmicro_permute`.

**Remaining work:** retire the session/profile selectors only with these live comparison callers and
their explicit primitive/resident ownership. The Laguna model-resource dependency in R0313–R0320
prevents actual full-model/public validation; missing measurements are not a reason to disable
existing paths.

Done when: run planar-comparison, activation-comparison, resident-layout, profiler and
runner contracts plus full Q6 numerical oracles; then run both owner orders of the A/B and a public
`LLM.generate()` check with an available Laguna artifact. Record layout/consumer/default selection,
exact state and cleanup. Keep legacy/interleaved consumers and cache conversion where required, and
preserve historical result readers when removing old labels.

### R0324 / R0325 / R0326 — Laguna Q6 activation comparison migration

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `q6_compact_activation`, `q6_half_row_activation`, `q6_skip_padded_activation`,
`scripts/laguna_q6_*activation_ab.py`.

**Remaining work:** migrate constructor-level controls and numerical/A/B callers to explicit
primitive ownership together. Do not delete these distinct comparison arms or registered fallback
instantiations with the selectors. Public validation is blocked by the local Laguna model resource
described in R0313–R0320; it is not an admission restriction.

Done when: run Q6 activation-comparison, resident-layout, runner, selected-MMQ and full
Q4/Q6 MoE oracles, then exercise each repaired harness and public `LLM.generate()` with an available
Laguna artifact. Confirm matching resident/consumer layouts, selected defaults, exact outputs and
cleanup before retiring the public overrides. Preserve the enabled backend capabilities and ordinary
cache consumers needed by other quant/shape routes.

### R0323 — Laguna fused selected-SiLU pack comparison ownership

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `set_fused_selected_silu_pack(...)`, `scripts/laguna_fused_silu_pack_ab.py`.

**Host migration complete:** `fused_selected_silu_pack_diagnostic` scopes the A/B
harness's fused/unfused selection through complete-state capture and restores it before
close, including failure. The persistent setter is removed; enabled capability and strict
unfused primitives are unchanged. Validation: 198 focused host tests and both synthetic
full Q4/Q6 MoE oracle cases pass on gfx1151. See
`worklog/entries/20261106T122512.446789Z-lhl-r0323-silu-scope-d39c09.md`.

**Remaining dependency:** public compatible/fallback-shape selection, output and teardown
checks need an accessible Laguna checkpoint; configured local artifacts are absent.

Done when: migrate the A/B caller to explicit composite/primitive ownership and retain its
complete-state comparisons. Run fused-SiLU comparison, selected-MMQ and runner contracts plus
primitive and full Q4/Q6 MoE oracles; then exercise public `LLM.generate()` with compatible and
fallback shapes, confirming selected route, outputs and teardown. Keep the enabled capability and
registered strict unfused chain.

### R0322 — Laguna Q6 resident-layout comparison migration

Owner: `hipengine/runtime/laguna_moe.py`.

Scope: `q6_qmicro`, `q6_qmicro_planar`.

**Blocker:** retirement must migrate layout comparison ownership, materializer and session
construction together. The Laguna model-resource dependency in R0313–R0320 prevents the required
public-route validation, not execution.

Done when: migrate live MoE and planar A/B callers to explicit resident/consumer layout
ownership, retaining the legacy cache adapter and registered non-qmicro consumers. Run Q6
resident-layout, materialization, runner, linear-dispatch and full MoE numerical tests; then
exercise public `LLM.generate()` across short, boundary-neighbor and unrelated prompt lengths,
recording resident/consumer agreement, outputs and teardown. Preserve historical result readers and
never reinterpret legacy bytes as planar qmicro.

### R0321 — Laguna cached-attention comparison and policy migration

Owner: `hipengine/runtime/laguna_gguf_runner.py`.

Scope: `prefill_kv_preappend`, `prefill_cached_meta`, `prefill_global_qrow6`,
`prefill_dense_initial`.

**Construction and caller migration:** `LagunaCachedPrefillPolicy` resolves independent legacy
choices once; resident construction accepts that immutable policy. All three cached-meta/qrow6/
dense-initial A/B child constructors pass it explicitly, preserving unspecified backend defaults,
fixed overrides and historical result readers. Package-role substitutions remain separate.
Compatibility constructor overrides are still supported; conflicting policy plus overrides fail.

**Validation:** 235 focused host contracts pass, including explicit-policy resident construction,
KV forwarding and close. Maintained cached-preappend/source and bulk global/SWA wrap device
comparisons pass on gfx1151. See
`worklog/entries/20261106T140435.988093Z-lhl-r0321-device-b5271d.md`.

**Cache migration:** resident construction forwards policy identity into KV allocation, and cached
attention selection consumes the policy directly. Direct-cache legacy defaults remain false;
ambiguous policy plus overrides fails before runtime acquisition. Explicit-policy eviction
fallback and selector-to-device initial global/SWA output equality pass. See
`worklog/entries/20261106T141438.269623Z-lhl-r0321-device-route-6f4ad7.md`.

**Additional validation:** internal KV construction no longer duplicates resolved policy fields.
Device-selected global/SWA cached-meta fallback after eviction matches explicit primitives
(`8d32e380c`). Bulk-versus-serial partial tiles at 2, 7, 8 and 12 rows pass around SWA wrap
(`19b08b9d8`); these are primitive comparisons, not cached-policy selection for partial tiles.
Host layer routing covers 17, 127, 128, 129, 257 and 384 rows, including staged-verifier
bypass (`1eef3f733`). Device verifier prefix/cancel publication matches canonical control
K/V and span metadata for 0, 1, 3 and 7 accepted rows (`ac2bc887f`).

**Remaining work:** public selected-variant/output/state/cleanup matrix, including partial
prefill and verifier logits, before retiring compatibility inputs. Device publication checks
alone do not establish model-level verifier output parity.
Public model checks need the accessible Laguna artifact described in R0313–R0320. This dependency
does not restrict execution or disable defaults.

Done when: migrate session/cache construction and all three attention A/B callers
together, preserving primitive comparisons and historical readers. Run cached-attention, runner,
role-policy and dense-metadata contracts; then GPU global/SWA parity across partial, eviction, wrap
and verifier cases plus public `LLM.generate()` at short, boundary-neighbor and unrelated lengths.
Confirm selected variants, append ordering, state/output equality and cleanup.

### R0281-PARO-PROVIDER — Manual provider and private-scorer migration

Owner: `hipengine/speculative/paro_mtp_profiles.py`.

Scope: `mtp_chain_e2e_smoke.py`; `HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT`,
`HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP`,; `HIPENGINE_VERIFY_GPU_ACCEPT`.

**Decision needed:** preserve these named historical diagnostics or migrate manual provider choice
to an explicit harness-owned policy while separating private scoring from the public provider. Do
not widen the registered B1 profile scope or treat its scope check as the public Generation-2 depth
limit.

Done when: migrate profile binding, explicit-chain harness selection, proposal/scoring
metadata, drift/lifecycle callers and private-head allocation ownership together. Retain
selected-row reseeding, full-vocabulary borrowed scoring, strict verifier fallback and stale-owner
rejection. Run profile/manual selector and public adapter tests, borrowed/private scoring and
lifecycle tests, plus actual strict/production public generation with selector unset/0/1 and
reset/close coverage. Any removal of private branching needs an explicit replacement or retirement
decision, not just a release-window expiry.

Choose whether to preserve these explicit harness comparisons or retire them with their
private-scorer consumers. An old negative speed result does not by itself justify deleting working
tree primitives or changing public admission. A migration must cover reload/persistent mode
selection, confidence and active budget boundaries, sibling acceptance/selected hidden and KV
repair, graph bindings and tail padding. Keep the shared compiler/mask/acceptance oracles and
confirm public chain routing stays unchanged. The corresponding GGUF root/sibling acceptance policy
is separately owned by `scripts/gguf_mtp_bench.py`; its supported-comparison disposition is recorded
in `worklog/entries/20261105T232832.251011Z-lhl-r0281-topk-74268b.md`.

### R0380 — Recurrent-state storage controls and speculative writers

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `HIPENGINE_GGUF_FP16_RECURRENT_STATE`.

**Decision needed:** retain the storage override as a supported comparison or migrate it into a
caller-owned execution plan. Keep frozen allocation dtype; retiring an environment name is not a
reason to retire that invariant. Separate this interface cleanup from implementing typed speculative
selected commit, repairing the observed native-graph divergence and reviewing profile/admission
ownership against capability-only product rules. Missing benchmark rows are not runtime capability
failures and must not become new restrictions.

Done when: migrate storage resolution, allocation pricing, all writer selection, profile
binding, snapshots and comparison provenance together. Run
`tests/test_unit_qwen35_gguf_gdn_prefill_routing.py`, session-mode, scratch-plan-boundary,
FP16-state batch-gate and MTP2-seam tests, including nested failure restoration and explicit dtype
after environment changes. On HIP run `tests/test_gpu_qwen35_gguf_gdn_fp16_state.py` and public
generation/serving for FP32/FP16, short/segmented/long prompts, sparse and changing widths,
reset/reuse, failure rollback and teardown. Any graph repair must compare accepted-row state against
eager, including the recorded failing layer region. Any arithmetic promotion needs its
execution-profile gate; route availability and explicit requests follow capability contracts
independently.

Preserve dtype frozen before allocation across decode, prefill, snapshots and selected-state commit;
do not reinterpret storage after changing policy.

### R0382 — Long-context staged verifier and scalar-reference ownership

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `_gguf_staged_linear_rows_long_enabled()`; `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG`.

**Decision needed:** retain the explicit comparison or migrate the selector and row-wise
orchestration into one caller-owned strict/staged plan. Keep the registered strict primitives and
Conv/GDN/KV transaction ownership. Retiring the branch must not make strict requests silently use
staged arithmetic. Review the existing artifact-derived default policy separately against
capability-only product rules; do not perpetuate the old instruction that every runnable artifact
needs its own benchmark packet. Arithmetic-default promotion and runtime capability remain separate
contracts.

Done when: migrate policy binding, strict registry chain, row-scratch views, initial
snapshots, selected-state commit and gate route attribution together. Run
`tests/test_live_gguf_native_spec_cycle.py` staged/strict routing nodes and
`scripts/gguf_mtp_long_context_gate.py --help`. Before changing execution, exercise the gate with
target graphs disabled and public generation at split-boundary straddles and unrelated contexts;
assert staged/scalar call counts, full outputs/state, every acceptance prefix, reset/reuse, failure
rollback and teardown. Native graph/submission economics remain separate work.

### R0385 — Observed target/proposal sentinel crossing repair

Owner: `hipengine/runtime/qwen35_gguf_mtp.py`.

Scope: `test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback`,
`tests/test_live_qwen35_gguf_mtp_e2e.py`, `target_graph_proposal_handoff_boundary_miss`.

**Remaining repair:** reproduce the original crossing in an isolated process on a compatible, idle
device; trace proposal payload publication, stream order, Conv/GDN scratch and target/journal
ownership. Because the recorded failure can corrupt later GPU work, do not unskip it in a shared
broad-suite process. A failing reproduction must retire before any following health check or reset;
coordinate recovery if the device remains unhealthy.

Done when: run the isolated regression at
`tests/test_live_qwen35_gguf_mtp_e2e.py::test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback`
against the original local model and both sides of 1024, plus unrelated contexts and
B1–B7, before removing the handoff guard. The parent process runs the original guarded body in a
bounded child and starts a fresh-process health request only after that child exits successfully;
failure or timeout exits the test session before any following GPU case. Compare host/device proposals, every acceptance
prefix, payload validity, Conv/GDN/KV/cursors, next-cycle continuity, reset, cancellation and
teardown; follow it with a fresh-process health request. Then run the complete MTP E2E,
long-context, lifecycle/API/load packet and an actual `LLM.generate()` request confirming the
intended path. The long-context harness's host-proposal mode alone cannot certify device handoff.
Keep `tests/test_unit_qwen35_gguf_mtp_session.py` boundary/submission checks,
`tests/test_live_gguf_native_spec_cycle.py` launch/binding checks and the delayed metadata-upload
test in `tests/test_live_qwen35_gguf_nextn_provider.py`.

### R0387 — Dense gfx1100 packed-step profiling follow-up

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `tests/test_unit_server_f1_concurrency_bench.py`; `tests/test_unit_gguf_ar_decode_control.py`.

**Remaining work:** profile the dense packed decode step on gfx1100 before choosing a kernel,
state-transfer or submission change. The old comparison did not show aggregate scaling, but it is
not a measurement of today's route. Keep default packed execution and its scalar comparison; do not
infer a current speed result or disable the route from the old timing.

Done when: inspect `scripts/server_f1_concurrency_bench.py --help`, then collect matched
current c1/c2/c4 and neighboring/unrelated logical-width requests on one gfx1100 host with model,
quant, KV policy, context, sampler, queue count and execution profile recorded. Confirm actual
public packed submission and slot/state ownership. Profile warmed cache-only execution with
`rocprofv3 --kernel-trace`; separate projection/attention work, state/KV copies, host submission and
synchronization, and separate capture from replay. Check teacher-forced numerics/tasks for changed
arithmetic and exact ownership, ragged/shrinking groups, cancellation, repeats and cleanup. Preserve
complete request-wall and per-request latency alongside aggregate throughput. Run
`tests/test_unit_server_f1_concurrency_bench.py`, `tests/test_unit_gguf_ar_decode_control.py` and
execution-profile contract tests before changing the comparison interface.

### R0388 — Dense budget diagnostic interface ownership

Owner: `hipengine/speculative/mtp_budget.py`.

Scope: `scripts/mtp_budget_diagnostics.py`, `hipengine/speculative/mtp_budget.py`.

**Ownership decision:** retain the typed request-owned policy seam and CLI
controllers as supported diagnostics. The runtime exports only observations and
the protocol; sequence/adaptive controllers live in `scripts/mtp_budget_diagnostics.py`.
The maintained dense suite creates a fresh policy per request and reports choices
and post-commit observations. Public candidate-depth selection remains a separate
shared-planner contract. No extraction is needed merely to move this protocol.

**Validation:** host policy/suite/session tests pass, including cancellation before
budget choice. A local gfx1151 dense suite run exercises all nine B1/B2/B3 ordered
transitions, matches true-AR tokens for fixed and sequence arms, agrees with CPU
acceptance and drains allocations. See
`worklog/entries/20261105T205050.722023Z-lhl-r0388-diagnostic-owner-95d970.md`.

Done when: complete live forced-prefix state-repair and device fault-cleanup
checks before closure. Forced acceptance-prefix and failure-order observations have
CPU control-flow coverage in `tests/test_unit_mtp_budget_session_lifecycle.py`,
including request reset/reuse, tail clipping and cancellation after draft repair.
The scheduler and resource boundaries are doubled: these tests do not prove
forced-prefix GPU state repair or actual-device fault cleanup. Complete those
live checks before closure. See
`worklog/entries/20261105T205846.018145Z-lhl-r0388-cycle-order-10ca0c.md`.
A new timing-adaptive default needs its own declared policy and comparison against
fixed budgets; this ownership decision does not change or promote one.

### R0393-Q6 — Grouped-Q6 rollback interface and gfx1100 public engagement

Owner: `hipengine/generation/qwen35_dense_gfx1100_profiles.py`.

Scope: `GGUF_Q6_DP4A_GROUPED_TARGET_REQUEST_COUNTS`.

**Interface decision:** retain `HIPENGINE_C8_Q6_DP4A_GROUPED` as a supported production
rollback and standalone diagnostic. Profiles snapshot the selection on the generator;
explicit workspace scopes do not reread the environment. Preserve the workspace scope
and bounded q8_1 resource checks. See
`worklog/entries/20261105T215959.241558Z-lhl-r0393-q6-e91c33.md`.

**Concrete dependency:** this host exposes only gfx1151; no gfx1100 device is available
for the remaining public engagement check.

Done when: Hardware follow-up: run a real public packed-C8 request on gfx1100 and record selected Q6
variant, grouped-kernel engagement, repeatability and allocation drain. gfx1151 normal public
requests keep their existing BF16 selection. This is a validation follow-up, not a runnability gate.

### R0404 — Direct INT8 prefill evaluation after dtype repair

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `HIPENGINE_GGUF_INT8_PREFILL_DIRECT`.

Done when: Remaining evaluation: the historical direct-prefill rejection was measured before the
dtype-adapter repair. It records an observed failure, not proof that the entire discrepancy is inherent INT8
quantization error. Preserve the artifact, but rerun the calibrated same-prefix evaluator before
making a new quality claim or changing the arithmetic default. Start with `python
scripts/execution_profile_gguf_int8_direct_prefill_gate.py --model "$GGUF_MODEL" --backend
hip_gfx1100 --kernel flash --min-prompt-rows 4096 --decode-steps 24 --repeat-runs 3
--use-wmma-prefill --use-gemv-decode --json /tmp/int8-direct-prefill-repaired.json` in the ROCm
environment. Complete the execution-profile category/heldout, shape, repeat/isolation and task
gates, including separate tail-four coverage; this per-token evaluator does not establish Hadamard
quality. Preserve explicit direct execution and the oracle path while resolving that numerical
choice.

### R0406 — RX 7900 XTX server capacity refresh

Owner: `scripts/gguf_server_allocation_probe.py`.

Scope: `scripts/gguf_server_allocation_probe.py` — RX 7900 XTX server capacity refresh.

**Deferred hardware validation:** refresh the RX 7900 XTX single-request server context bound. This
session has gfx1151 hardware, not the requested XTX; substituting its result would not establish
that bound. On the XTX host, start with a short, multi-slab allocation probe, for example:

```bash
python scripts/gguf_server_allocation_probe.py --model "$GGUF_MODEL" --kv-storage int8_per_token_head --kv-scale-dtype fp32 --kv-scale-granularity per_token_head --speculative-mtp-serving off --prefix-cache off --max-active-requests 1 --max-context-tokens 54272 --prompt-rows 2048 --decode-tokens 8 --json /tmp/xtx-server-capacity-54272.json
```

Done when: Select the XTX explicitly in its ROCm environment. Record resolved server policy,
readiness/capability/response documents, allocation ownership and teardown. Bracket capacity with
short probes, then confirm the chosen context with the full-depth workload; the example context is
not a maximum.

### R0410 — Fused grouped-prefill combine adjudication

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `qwen4_exp_runner.py`, `weighted_lanes_sum_out_bf16_f32w`;
`fused_combine_enabled(owner, enabled)` and the explicit layer/runner argument;
`shared_gate_combine_f32_shared_out_bf16`,
`weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w`.

**Deferred decision:** retain the candidate and its explicit interface, promote it after a current
comparison, or retire the complete route. The corrected historical artifact records four Python
processes and four model residencies, not one residency. It does not establish a binding loss; do
not delete the candidate or describe the old timing as a promotion/rejection gate.

Done when: if evaluating promotion, add an engagement-verified fused-combine package to
`scripts/qwen4exp_halo_box_campaign_ab.py` or an equivalent same-process harness. The existing
harness does not expose this comparison. Measure both settings in one model residency across the
complete category/shape suite and heldouts, charging the separate cast and comparing against today's
cast-folded default. Verify exact outputs/state/KV, repeatability, graph recapture where applicable
and public `LLM.generate()` selection/teardown. Run `uv run pytest -o addopts='' -q
tests/test_gpu_qwen4_exp_pf4_lever2_fused_combine.py` in the ROCm environment before the model
comparison. If retiring by product decision, remove the runtime branch, wrapper, registry entry,
selector and candidate-only tests together, preserving strict primitives and history.

### R0415 — Register-state GDN decode follow-up

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `gdn_register_prefill`.

**Deferred work:** optimize decode without reopening the lead's accepted prefill-first decision. The
accepted resource review and rejection of per-column/every8 scheduling barriers still stand.

Done when: profile today's decode path under `OPTIMIZATION.md` before choosing an
optimization. Run the full profile gate and matched whole-model prefill/decode comparison for any
arithmetic or performance candidate, keeping the serial strict fallback. Remove this follow-up only
after the decode work is completed or the lead explicitly retires it.

### R0444 — Risk multiplier ownership and repair-grid follow-up

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT`.

**Deferred ownership decision:** retain an explicit diagnostic multiplier or replace environment
transport with a profile-owned value. Preserve the bound and unbound semantics, effective-value
reporting, strict/reference contract and post-binding diagnostic isolation. A repair-grid experiment
is not permission to change the risk threshold or the repair coverage contract.

Done when: Deferred optimization follow-up: the sparse repair wrapper still defaults to 1024 grid blocks,
and the runner does not override that count. The device loop is bounded by the queued count and
capacity, so fixed launch geometry is not fixed repair work. Use
`scripts/qwen4exp_repair_grid_probe.py` to compare explicit grids and profile the actual repair
calls before proposing a launch policy. Record effective multiplier, queue counts, overflow handling
and cache/ stream ownership; run numerical repair parity and public same-profile checks for any
adopted change. Earlier fixed-cost estimates are historical hypotheses, not a fresh promise of
recoverable wall time. Kernel traces follow the cached- build protocol in `docs/OPTIMIZATION.md`.

### R0459 — Row-split GDN recurrence removal decision

Owner: `hipengine/generation/qwen4_exp_profiles.py`.

Scope: `hipengine/generation/qwen4_exp_profiles.py` — Row-split GDN recurrence removal decision.

Done when: Deferred removal decision: decide whether UD still needs an explicit multi-column
comparison route and whether the Q4_K_M recovery plan still needs the shared implementation. Verify
both quant bindings and user-facing execution before removing any UD-specific route.

The maintained capability decision is to keep the registered multi-column,
row-split, DPP and strict references; R0039 owns the independent composition
quality decision. The local Q4_K_M files are `qwen35`/`qwen35moe`, not
`qwen4_exp`, so they cannot exercise this profile's public fallback. The
remaining check needs a Qwen4Exp Q4_K_M checkpoint: load it through
`hipengine.LLM(..., execution_profile="production")`, verify the resolved strict
fallback and actual GDN leaf, then generate and close. Do not substitute a
similarly named model or count the CPU binder test as that public execution.

### R0462 — QSA gather storage and variable-span follow-up

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `runner.qsa_wmma_enabled`, `qsa_wmma_attention`.

**Deferred storage decision:** retain the shared full-context gather or spend per-layer storage to
cache F16 K/V across chunks. Distinct layers share the output scratch within each chunk. Changing
only `first_position` to the chunk start would reuse another layer's K/V. **Clearing action:**
profile a representative long-context workload, choose the memory/reuse trade-off, then implement
per-layer ownership and admission accounting together if caching is chosen. Validate reset, prefix
adoption, snapshot restore, speculative rollback, interrupted prefill and teardown, including an
intervening decode append before more prefill. Do not infer a throughput benefit from the short
instrumented ownership diagnostic.

Done when: Deferred capability extension: the gather wrapper accepts only uniform BF16 `KVLiveSpans`.
Before offering this route to a non-dense policy, implement a gather consuming `live_counts`,
`token_positions` and `evict_mask`, plus matching sparse-mask/attention semantics. Verify
nonidentity physical pages, evicted positions and policy transitions against the strict reference;
preserve the existing dense route. This is an implementation prerequisite, not an artifact-identity
or benchmark-admission gate.

### R0471 — Optional selected DP4A adapter intent

Owner: `hipengine/loading/gguf_selected_contract.py`.

Scope: `bind_selected_call`, `SelectedCallIntent`, `SELECTED_OPTIONAL_ADAPTERS`.

The shared selected-call intent now represents the optional DP4A routes
(`HIPENGINE_GGUF_RAW_SELECTED_DP4A`, `HIPENGINE_GGUF_T16_SELECTED_DP4A`,
`HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A`) instead of refusing them: the intent
declares the route and the caller-owned Q8_1 workspace, and a bound call records
the conversion symbol, the dp4a kernel, the Q8_1 operand and the plan-owned
workspace (`moe_q8_1`, scratch-plan lifetime, per-token rows and the resident's
K). Engagement still comes from the live selectors, so one plan binds both
arithmetic contracts and a certificate never covers the other. Row limits,
ordered dual partners, the strict two-single fallback and F32 capabilities are
unchanged.

The host bundle and public gfx1151 optional-adapter on/off checks pass, including
C2 AR after the BF16-QKV/FP32-Conv scratch repair. They execute `ar_decode_rows`
and `ar_prefill`, not `ar_decode_native_rows`. The remaining blocker is public
scheduler integration: `SubmitPollTextGenerator` uses packed `step_batch_native`
or serial session steps, never the separate `_generate_greedy_batch` native-row
owner. Disabling packed/stream controls or adding a certificate operation does
not connect those state owners. See
[`worklog/entries/20261031T230425.271053Z-lhl-r0471-native-public-gap-869c31.md`](../worklog/entries/20261031T230425.271053Z-lhl-r0471-native-public-gap-869c31.md).

Done when: integrate compact native-row execution with resident scheduler session,
position, active-slot, state/KV and reclaim ownership, preserving packed/serial
routes; then run public native-row generation with an optional adapter engaged
and report its actual launch plus the bound kernel, converter and workspace
owner from the resident certificate. An unengaged run of the same artifact must
report the unconverted record. The clearing host checks are
`tests/test_integration_gguf_invocation_contract.py`,
`tests/test_integration_gguf_selected_call_intents.py` and
`tests/test_integration_gguf_execution_authorization.py`, all passing today for
the host contract. Any changed arithmetic default additionally follows the
execution-profile gate. The optional contract and compatibility native-row route
are implemented; their public scheduler integration is not.

### R0484 — Surya decode fallback and sparse-chunk work

Owner: `hipengine/runtime/surya.py`.

Scope: `surya_scatter_kv_f32_spans`, `tests/test_gpu_surya_kv_spans.py`.

Current: the runtime fallback is now a span-aware, separate QK/softmax/AV chain
for arbitrary positive head widths and divisible GQA ratios. The dense rocBLAS
chain remains only a diagnostic oracle. Nonidentity pages, eviction, causal and
partial contexts, empty visible sets and masked NaN payloads pass on gfx1151;
see `worklog/entries/20261106T010248.235022Z-lhl-r0484-span-fallback-cfe822.md`.
Public continuation still needs the locally absent `datalab-to/surya-ocr-2`
checkpoint; do not treat device fixtures as public generation evidence.

**Remaining contract validation:** exercise public Surya generation and continuation
with the span-aware fallback before retiring the dense diagnostic oracle.

Done when: Deferred optimization follow-up: profile sparse live spans before changing the fused producer's
full-chunk dot products. Bounding work by visibility must preserve token positions and masking;
verify with the spans GPU suite and the applicable numerical gate.

### R0488 — Surya same-process comparison failure

Owner: `scripts/surya_perf_compare.py`.

Scope: `scripts/surya_perf_compare.py --merge`, `lane_isolation`.

Done when: Deferred reproduction: the local cache lacks `datalab-to/surya-ocr-2`; the R0484 public request
failed at loading, before inference. Make the checkpoint available to `resolve_surya_path`, then run
`python3 scripts/surya_perf_compare.py --split all --runs 3 --lanes hipengine_gpu,torch_cuda` in the
HIP environment. Compare with isolated runs, find and repair the first divergent state, and require
every combined row to pass before publishing a replacement comparison. Preserve historical artifact
provenance; remove any still-applicable scoreboard isolation caveat only when the new comparison no
longer needs it.

### R0490 — Explain the Surya vision tile-shape spikes

Owner: `scripts/surya_vision_tiling_cost.py`.

Scope: `SHAPE_TILE_*`.

Done when: Deferred profiling: make the Surya checkpoint available and use `rocprofv3` from the configured
ROCm environment. Compare kernel traces for the 128-row and
512-row shapes at 4096 patches, using `rocprofv3 --kernel-trace -- python3
scripts/surya_vision_tiling_cost.py --pages page_ja.png --blocks 128,512 --reps 3 --out
/tmp/surya-4096-shapes.json`. Attribute any shape-dependent latency to the actual kernels before
changing the policy; a rocBLAS selection change is still only a hypothesis. A policy change requires
the applicable numerical gate and public Surya validation.

### R0491 — Surya mid-size and A4 vision tile investigations

Owner: `scripts/surya_vision_tiling_cost.py`.

Scope: `scripts/surya_vision_tiling_cost.py` — Surya mid-size and A4 vision tile investigations.

**Deferred experiments:** the local Surya checkpoint is missing. Once it is available, run `python3
scripts/surya_vision_tiling_cost.py --pages page_dense.png,page_ja.png,page_long.png --blocks
64,96,128,192 --reps 3 --out /tmp/surya-midsize-shapes.json` and extend the probe to nearby mid-size
grids before choosing a new envelope. Do not special-case the lone historical mid-size anomaly.

Done when: For A4, separate a larger scratch budget from an envelope change: the byte budget is the limiting
policy. Start with `python3 scripts/surya_vision_tiling_cost.py --pages page_a4.png --blocks
2048,4096,8192 --reps 3 --out /tmp/surya-a4-shapes.json`. The dense A4 comparison remains unrun and
needs a separately resource-checked large allocation; shape-to-shape results do not establish dense
parity. Any policy change needs the applicable numerical gate and public Surya validation.

### R0492 — Bound Surya causal-prefill keys per tile

Owner: `hipengine/runtime/surya.py`.

Scope: `SuryaGpuRunner._attention_packed`.

Done when: Deferred optimization: make the Surya checkpoint available, then use the configured ROCm
environment to profile with `rocprofv3 --kernel-trace -- python3
scripts/surya_gpu_rocprof_driver.py --phase prefill --repeat 3`. Use the driver phase markers to
exclude vision/load work. If score computation is worth optimizing, implement the bounded-key path
and compare against the current chain on dense, partial-tail and multi-tile prompts, then run the
applicable numerical gate and public Surya generation. Keep the current working implementation until
that replacement is ready.

### R0493 — Capture the Surya A4 fp32 reference

Owner: `scripts/surya_oracle_greedy.py`.

Scope: `QUALITY_ONLY_CASES`, `tests/test_live_surya_transcription.py`,
`tests/fixtures/surya/oracle_fullpage_protocol.json`.

Done when: Deferred capture: restore the missing `datalab-to/surya-ocr-2` checkpoint and run
`python3 scripts/surya_oracle_greedy.py --case protocol --only a4 --device cpu --page-dir
tests/fixtures/surya --out-dir /tmp/surya-a4-oracle`. The A4 case and 4000-token provisional
budget are implemented; inspect actual natural EOS rather than assuming that budget completes
capture. See `worklog/entries/20261031T201834.327598Z-subagent-refactor-surya-a4-oracle-cases-15f60a.md`.
Merge the captured row into the existing fixture only after review; preserve all other cases.
Move A4 into the implementation-parity test cases only after examining the capture and running the comparison.
Preserve quality coverage throughout; use coordinate-tolerant comparison only if observed bbox-digit
variation justifies it, not merely to make a failure pass.

### R0494 — Report prompt usage from the generation output, not a second count

Owner: `hipengine/generation/registry.py`.

Scope: `GenerationOutput.prompt_tokens`; `_usage` in `hipengine/server/api.py`.

Done when: complete remaining real-checkpoint prompt-count validation for Gemma 4 image on
gfx1100 and Surya OCR on supported backends. The local Gemma model and matching
`mmproj-F16.gguf` ran through gfx1151 HTTP image chat: usage, generation output and actual
expanded prefill all report 284 prompt tokens, with clean teardown. See
`worklog/entries/20261105T222851.099261Z-lhl-r0494-gemma-bbe09d.md`.
Surya checkpoint is absent locally; gfx1100 hardware is unavailable on this host.
Qwen4Exp gfx1151 HTTP image validation passes; see
`worklog/entries/20261031T201027.069818Z-lhl-r0494-qwen4exp-real-http-usage-c3cbd6.md`.
Preserve the implemented contract: `GenerationOutput.prompt_tokens` owns the count, `None`
falls back to counting the prepared prompt (legacy plugins), a stated `0` stays `0`, and
`DecodeState.prompt_tokens` / `telemetry.usage["prompt_tokens"]` are diagnostic mirrors that never
authorize a number. Keep `tests/test_unit_server_prompt_token_contract.py` and
`tests/test_integration_server_prompt_token_contract.py`.

### R0495 — Decide CPU-reference vision admission

Owner: `hipengine/generation/surya.py`.

Scope: `LLM.generate()`; `datalab-to/surya-ocr-2`.

Done when: Deferred policy decision: the CPU-reference generator has no vision resource ceiling. Choose a
configurable patch/memory budget or calibrate a CPU timing estimate before adding admission there.
GPU timing constants do not describe CPU execution. Preserve existing runnable inputs until that
policy is chosen; no arbitrary default ceiling is added by this cleanup. Validate the selected
policy on both sides of its limit and an unrelated size through `LLM.generate()`. Live Surya
validation also needs a locally available checkpoint; the current local cache lacks
`datalab-to/surya-ocr-2`.

### R0499 — Decide whether to pursue exact dual GEMV scheduling

Owner: `scripts/vibevoice_tts_session_bench.py`.

Scope: `scripts/vibevoice_tts_session_bench.py`; `scripts/vibevoice_tts_quality_suite.py`.

Done when: Deferred research decision: decide whether to fund a different workgroup schedule or output
partition that preserves the per-output reduction tree. Neither restoring the rejected thread count
nor repeating shared-memory staging resolves this item. If pursued, profile the current session
first, compare exact outputs, synchronize cache flushes before isolated timing, and verify through
`scripts/vibevoice_tts_session_bench.py` and the generated-audio quality suite
(`scripts/vibevoice_tts_quality_suite.py`), not correctness tests alone. Alternatively close this
research direction explicitly.

### R0500 — Choose cross-branch VibeVoice batching ownership

Owner: `hipengine/runtime/vibevoice_tts_session.py`.

Scope: `_negative_condition`.

Done when: Deferred architecture decision: choose shared immutable weight ownership and a per-row KV
interface before implementing speculative cross-branch batching. Specify negative-position rollback
on non-diffusion tokens, speech-boundary resets, cancellation and errors; preserve RNG consumption
and feedback ordering. Ordinary multi-row prefill is not a replacement for this transaction
contract. Implement registered kernel variants only after that contract is chosen, then verify
branch conditions, token chains, rollback and request reuse through the public synthesis surface and
generated-audio quality suite. If arithmetic changes, run the applicable execution-profile numerical
gate as well.

### R0504 — Evaluate overlapping chunked prompt activations

Owner: `hipengine/generation/engine_loop.py`.

Scope: `prefill_activation_ready`, `prefill_needs_activation`.

Prompt activation now owns request-scoped claims and may overlap another
request's chunked prefill lifetime. Scheduler items admit multiple first chunks;
provider slots and sinks retain independent completion/abort ownership. CPU
coverage checks separate openings, peer setup failure, cancellation and claim
retirement. Public GPU token/acceptance comparison remains outstanding.

Done when: Deferred scheduler work: implement co-activation of chunked rows while preserving provider-slot
ownership and failure isolation. Exercise unequal prompt lengths, different finish orders, partial
failures and cancellation. Compare generated tokens and acceptance against independent
single-request runs through the public generation surface; check that waiting requests keep their
provider and that all claims and sink buffers are released. The lifecycle fix alone does not enable
overlap or establish its numerical/acceptance behavior.

### R0509 — Choose shared target/provider checkpoint memory accounting

Owner: `scripts/int8_mtp_prefix_gate.py`.

Scope: `_evict_prefix_snapshot`, `discard_prefix_checkpoint`;
`HIPENGINE_MTP2_PREFIX_CHECKPOINT_ENTRIES`.

**Current implementation:** trimming counts target and all provider device owners,
including unpaired provider records and target arena padding. Idle arena pools consume
remaining allowance; quarantine is charged but never reclaimed before retirement.
Provider host KV has a separate allowance of the same configured size; DMS target cache
snapshots consume that host allowance too. Default provider capture now uses owner
byte budgets instead of an independent entry cap. Explicit positive caps and zero-disable
remain rollback controls until same-route server parity is repaired. Implementation:
`worklog/entries/20261106T022713.106769Z-lhl-r0509-owner-cap-ae5d66.md`.
Capture/overwrite and public pressure evidence:
`worklog/entries/20261106T021942.247074Z-lhl-r0509-dms-pressure-1825e3.md`.

Public dense and DMS pressure checks exercise real 512-token reuse, speculative cycles
and zero tracked allocations after close. Fresh/resumed INT8 token differences disappear
when both paths select direct-INT8 prefill; default fresh prefill uses a transient BF16
oracle instead. This diagnostic does not change the arithmetic default or establish its
numerical quality. The separate Japanese AR/speculative mismatch reproduces with caching
off, strict mode, and BF16 KV; it remains an unresolved server parity-gate failure.

Done when: Keep the rollback until shared accounting and coordinated eviction are tested. Run
`scripts/int8_mtp_prefix_gate.py` against a radix-enabled server: require real prefix hits, compact
INT8, speculative cycles and parity with the same prefix-reusing AR route. A cache miss or AR-only
response is not a pass. Include budget pressure, overwrite, target eviction and shutdown release
tests.

### R0510 — Define bounded-context startup warmup ownership and coverage

Owner: `hipengine/server/api.py`.

Scope: `_startup_probe_width_reduction`, `max_sequence_length`.

**Bounded session implementation complete:** startup probes use temporary bounded-context
sessions sharing immutable weights and release probe allocations on success, failure and
cancellation. Configured 4 GiB KV-budget server checks passed at width four for AR and
MTP startup, with ordinary concurrent HTTP requests succeeding. See
[MTP server checkpoint](../worklog/entries/20261106T024942.449109Z-lhl-r0510-mtp-server-f8fd1d.md).
These runs do not establish physical low-free-memory behavior or full-context/graph coverage.

Done when: wide startup probes and subsequent public concurrent requests succeed under
physical constrained memory, with failure/cancellation releasing all probe resources.
Preserve actual request context limits and capacity estimates; short probes must not certify
full-context capacity or graph buckets. Keep the width-reduction guard until that validation.

### R0512 — Revisit Q5 decode-dual dispatch separately from working prefill

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `dense_pair_quant`, `gguf_q5_k_t16_v1`.

Done when: Deferred tuning decision: re-screen the Q5 dual against the currently selected two-single-kernel
plus SiLU chain on gfx1151 before changing decode selection. If pursuing it, select a Q5-specific
registered key, verify the intended decode owner and run the applicable numerical and public route
gates. Do not change the working prefill route as part of this decision.

### R0513 — Define captured-accept logprob outputs for every published token

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `_device_sampled_accept_plan`.

**Deferred graph interface:** emit the selected token's processed logprob and requested top-logprobs
for accepted draft tokens, rejection corrections and the bonus token. The selected token need not be
top-1. Bind each result to the target row that predicted it, and discard metadata beyond stop/EOS or
cancellation. Define buffer ownership, graph replay updates and streaming publication together.

Done when: After removing the decline, run the existing logprobs parity gate with zero delta against AR and
confirm captured accept actually ran.

### R0514 — Define device forced-token inputs and queue commit semantics

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `supports_native_gpu_sampling`, `_device_sampled_accept_plan`, `_sampled_accept_summary`.

**Deferred device interface:** carry per-row forced-token overrides into sampling and acceptance,
with request-local queue position and graph replay updates. Advance the queue only for committed
published tokens, not rejected speculative rows. Preserve queue state across rollback, partial
acceptance, stop truncation and cancellation; validate token bounds and interactions with
constraints.

Done when: Remove the refusal only after the forced-token gate compares output tokens and queue positions
against the host path while telemetry confirms captured-graph accept. Exercise forced sequences
spanning cycle boundaries and ordinary rows sharing a batch. Host-path test success does not
establish device support.

### R0515 — Decide retirement of the Gemma 4 gate/up MMQ rollback

Owner: `hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py`.

Scope: `hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py` — Decide retirement of the Gemma 4 gate/up MMQ rollback.

Done when: Deferred rollback decision: record the teacher-forced verdict against the campaign's frozen
evaluator before retiring the switch. Capture the reference with
`HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ=0`, then confirm distinct selected routes for reference and
candidate. Include short and long prefills, natural inputs and low-margin cases; a single
long-prefill comparison does not cover the recorded length sensitivity. Preserve real expert
geometry and multi-tile coverage. This is switch retirement, not reconsideration of the default-on
decision.

### R0516 — Repair Gemma 4 down-MMQ continuation before retiring its lever

Owner: `hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py`.

Scope: `hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py` — Repair Gemma 4 down-MMQ continuation before retiring its lever.

Done when: Deferred numerical repair: diagnose and repair the MMQ owner's generated-history failure, then
run the unchanged continuation screen and full applicable execution-profile gate, including
selected-route, repeatability and isolation checks. Use the same model artifact for both arms:

```bash
HIPENGINE_GEMMA4_MOE_DOWN_MMQ=1 .venv/bin/python scripts/gemma4_generated_history_gate.py \
  --model <artifact>.gguf --output-dir /tmp/gemma4-down-mmq-repair
```

Keep the down-MMQ rollback until continuation and full-profile repair gates pass; a teacher-forced
screen alone does not clear generated-history failure.

### R0519 — Decide the compensated Gemma WMMA candidate independently of selectors

Owner: `hipengine/kernels/hip_gfx1100/gemma4/gemma4_experts.py`.

Scope: `wmma_plain`; `HIPENGINE_GEMMA4_MOE_PREFILL`.

Done when: Deferred candidate decision: evaluate the compensated route under the current applicable
execution-profile contract, then promote, repair or retire its Gemma wiring. The old absolute-tail
discussion is historical evidence, not permission to weaken the current gate or to change unrelated
defaults. Preserve exact comparison modes, independently useful kernels and current automatic
dispatch. If retiring candidate modes, inventory their harness callers and remove only
candidate-specific wiring/tests after preserving the required comparisons.

### R0521 — Revisit backend-specific decode graph engagement separately from telemetry

Owner: `hipengine/runtime/gemma4_decode_graph.py`.

Scope: `gemma4_decode_graph.py`, `_CAPTURE_MAX_KEYS`, `GEMMA4_DECODE_GRAPH_CAPTURE_MAX_KEYS`,
`launched_fallbacks`.

Done when: Deferred optimization decision: reduce capture/replay overhead or remeasure the crossover on
each affected physical host before changing its policy. Compare public decode with graph selection
confirmed, including capture amortization, short and long requests, and contexts on both sides of
the threshold. Preserve replay parity and request/resource lifetime checks.

Retire graph rollback and fallback telemetry separately only when no supported fallback or
diagnostic caller remains. Preserve backend independence; broader engagement on one host does not
establish another host’s crossover.

### R0003 — Choose a DMS layout that preserves interior-expiry semantics

Owner: `hipengine/kvcache/dms.py`.

Scope: `build_dms_live_mask`.

**Deferred design decision:** choose between a prefix-only head-advance fast path with interior
compaction, or a layout that separately represents retained history and the expiring window. The
latter requires coordinated attention-reader, allocator, snapshot, prefix-cache and speculative
rollback changes; neither may silently discard non-evictable history. Account for reserved extent
bounds and wrap/rebase behavior before changing `base_offsets`.

Done when: cover interior expiry, prefix expiry, no expiry, mixed heads, capacity boundaries
and repeated wrap/rebase against the host oracle. Run `tests/test_gpu_dms_append_decode_hip.py`,
INT8 device-payload, compact-attention, prefix and transaction tests, then public DMS generation
including continuation and rejected speculative suffixes. Profile the selected route before claiming
an append-cost improvement. Keep the existing race-safe staging until its data movement is replaced
safely.

### R0011 — Define bounded generic-provider child ownership

Owner: `hipengine/generation/engine_service.py`.

Scope: `generate_detailed`, `stream_detailed`.

**Implementation in place:** generic child admission, bounded advancement and retirement now
have an explicit contract independent of model-attached MTP. The submit/poll adapter advances
children on the service driver and contains per-child failures. Laguna DFlash uses private target
sessions, drafters and cycles, advancing one prompt token or one speculative cycle per tick.
Cancellation, deadlines, shutdown, partial setup failures and zero-token requests have focused
coverage. See `worklog/entries/20261106T075454.368851Z-lhl-r0011-laguna-children-d77b10.md`
and `worklog/entries/20261106T075838.544970Z-lhl-r0011-real-deadline-fdc5ae.md`.

**Remaining validation:** public LLM method tests inject the service with synthetic model work;
they do not exercise loader construction or actual inference. The local gfx1151 host lacks the
Laguna GGUF/repacked target and DFlash drafter required for actual-model validation. Keep this
entry open until that public route and provider lifecycle packet run with those artifacts.
Do not execute model work on frontend threads or hide a whole generation in an administrative
command; preserve the service's sole-driver ownership.

Done when: extend `tests/test_integration_engine_service_speculative_ownership.py` with an
actual generic-only child adapter and test blocking/stream agreement, slow work beside another
request, cancellation, deadlines, provider errors, shutdown and subsequent recovery. Then exercise
generic-only generation through `LLM.generate()` or the server and confirm the intended provider and
driver ownership. Existing tests of rejection and MTP compatibility are containment coverage, not
proof that the missing generic lifecycle is implemented.

### R0071 — Resolve split-prefill arithmetic before retiring its divergence pin

Owner: `hipengine/runtime/qwen35_gguf_runner.py`.

Scope: `docs/REFACTOR.md`; `tests/test_live_qwen35_gguf_chunked_prefill.py`; `test_split_prefill_divergence_boundaries_are_unchanged`.

**Deferred numerical-contract decision and repair:** establish the intended cross-schedule contract
using the execution-profile oracle, then repair the one-row routing and long-context full-attention
differences against that contract. The old serial-reference comparison and layer-localization
evidence are preserved in `docs/REFACTOR.md` at commit `304c3324d`, under “Split prefill diverges
from a single call”. They are historical observations, not a fresh quality verdict for current
kernels. Do not cap query windows just to force equality: the recorded experiment rejected that
tradeoff.

Done when: reproduce both families and the equal-output control, test values on both sides of
the boundary plus unrelated lengths and split points, and compare independent sessions against the
declared oracle. Run `tests/test_live_qwen35_gguf_chunked_prefill.py` and the applicable numerical
profile gates; exercise public prefix reuse and fresh generation with continuation. Replace
`test_split_prefill_divergence_boundaries_are_unchanged` with positive contract coverage only after
the repair, and extend `test_batched_suffix_extend_is_bit_exact_against_recompute` where that
exactness contract applies. Do not treat passing a test that requires inequality as proof of
numerical correctness or remove it merely to clear the ledger.

### R0127 — Choose whether to replace host grammar selection

Owner: `hipengine/generation/grammar.py`.

Scope: `hipengine/generation/grammar.py`, `supports_grammar`.

**Deferred implementation choice:** retain host selection or replace its mask application and
selection with a registered device implementation. Profile the controlled route before choosing an
optimization. A replacement must preserve request-local matcher state, selected-token handoff, token
controls and sampling semantics, EOS/content boundaries, cancellation and speculative
commit/rollback. Do not remove the working host route before its supported inputs have a
replacement; do not claim a speedup from reduced readback alone.

Done when: run `tests/test_unit_grammar_masks.py` and
`tests/test_unit_qwen4exp_mtp_structured.py`, add host/device selection parity and mixed-request
isolation tests for any replacement, then exercise public required/named tool generation with
streaming, MTP and cancellation. Confirm the intended selection route actually ran before removing
host-only machinery.

Keep model-owned embedded-template/XML parsing and independent model parsers; replacing host
selection must not restore generic JSON forced-prefill behavior.

### R0128 — Compose optional resource accounting and implement sparse device-position selection

Owner: `hipengine/loading/qwen4_exp_context.py`.

Scope: `resolve_qwen4_exp_context`.

**Deferred resource-design work:** compose optional MMQ sidecars, graph and verification allocations
with admission and their actual owner lifetimes. Identify which allocations are already covered
before adding reservations; avoid double counting as well as over-admission. Preserve
explicit-request errors and keep native context, QSA dense-equivalent limits and selected-position
capacity independent. The scratch floor is not proof that every optional combination fits.

**Separate missing capability:** device-position-owned QSA still rejects a position beyond the
dense-equivalent limit in the decode path. Implement native sparse selection and its graph ownership
before removing that guard; increasing the limit does not supply sparse selection. This does not
restrict the separate host-position sparse route.

Done when: run the scratch-plan, QSA-scratch-reserve and chunk-memory-ladder unit tests, then
exercise public admission with optional resources alone and in combination, including failure
cleanup and requests around allocation limits. For device-position sparse support, compare
selection, logits and state against the declared oracle across the dense/sparse boundary and
unrelated lengths, then exercise public continuation and graph replay.

### R0129 — Decide when to retire Q8 MMQ prepack rollback

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `weight_pack/gguf_q8_0/mmq_kmajor76`.

**Deferred rollback decision:** retain the explicit `q8_mmq_prepack_enabled` owner scope until the enabled
prepacked route passes assessment after an owner refresh and release window, then remove redundant
experiment selection. The ambient `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK` fallback is removed;
production still publishes true, strict false, and diagnostics bind the owner directly.
Preserve the strict raw-Q8 route and raw weights needed by decode and risk
repair. Removing the rollback setting is distinct from changing the prefill MMQ arithmetic profile;
do not infer either from the other. R0128 owns composed optional-resource admission, including the
full additional sidecar allocation.

Done when: run sidecar lifecycle and execution-profile unit tests, GPU packer byte parity and
full logits/state/KV comparisons for the selected route. Exercise public prefill with sidecars
enabled and disabled, verify owner memory and cleanup, and check mixed prefill/decode requests
before a performance-driven retirement. Preserve explicit resource and arithmetic failures rather
than silently falling back.

### R0131 — Coordinate live-derived Surya split planning with launch bounds

Owner: `hipengine/runtime/surya.py`.

Scope: `surya_full_attn_decode_f32_spans`, `spans.max_live_count`.

**Deferred arithmetic/launch design:** pass a validated live launch bound and consistent split count
through the wrapper, or provide storage for the actual capacity-derived count. The runtime now
passes its allocation plan's chunk and split capacity explicitly; the wrapper rejects undersized
partial buffers before submission. Host regressions cover allocation/launch agreement (see
`worklog/entries/20261105T212021.191426Z-lhl-r0131-workspace-943878.md`). Keep the physical
page/span bounds separate from the live work bound, and size all three partial buffers for the
launch they serve. Account for growing live counts, graph reuse and empty producer ranges.

Done when: add host launch/allocation regressions for unequal capacity and live counts,
including capacity/live-count mismatches and alignment boundaries. Run `tests/test_gpu_surya_kv_spans.py`, the
applicable production numerical gate and strict fallback checks, then public Surya continuation at
fixed live lengths across different configured capacities. Profile before claiming convergence or
speedup.

Do not apply a chunk-only override: capacity 5000 with chunk64 requires 79 partials, not the
live4096 plan’s 64. Cover that mismatch in regression tests. Retire capacity-derived chunking only
after wrapper counts and all partial-buffer allocations agree.

### R0132 — Consolidate native context bounds without restoring evidence admission

Owner: `hipengine/runtime/gguf_native_spec_cycle.py`.

Scope: `native_target_context_limit`.

The resolver now replaces the dense geometry/quant allowlist with the backend's structural
feed-forward capability and the allocated session capacity. Conflicting capacity sources use the
smallest positive bound; a present nonpositive slot raises a named resource error instead of using
a stale layout. gfx1151 public dense/MoE checks at prompt lengths 73, 96 and 137 pass with native
route attribution and clean teardown; they do not establish gfx1100 behavior.

**Remaining consolidation:** diagnose the recorded gfx1100 MoE full-logit discrepancy before
removing its numeric fallback. Exercise the affected route, identify its actual failure scope, and
retain eager/serial execution where graph capability is absent. Do not restore a 256 cap merely
because longer contexts lack measurements.

Done when: run native-context and native-cycle unit tests, then the native-context state/logit
gates across affected geometries, KV layouts and boundary-straddling rows. Verify public generation,
every accepted-prefix state, continuation, rollback and graph/eager route attribution before
consolidating bounds.

### R0138 — Complete native adapter ownership without blanket embedding refusal

Owner: `hipengine/runtime/gguf_native_spec_cycle.py`.

Scope: `gguf_native_spec_cycle.py`, `_device_token_embedding_weight(reason="native_mtp_graph")`.

**Deferred adapter work:** represent any additional embedding staging or expert sidecar route in the
actual shared call plan, including pre-allocation shape, dtype, physical operands, lifetime and
capture behavior. Remove only the corresponding refusal once that implementation exists. Do not
remove the working device-resident embedding fallback or infer support from a nominal adapter name.

Done when: extend `tests/test_integration_gguf_execution_authorization.py` and
`tests/test_integration_gguf_native_physical_operands.py` for each added adapter, then test graph
capture/replay, allocation failure and teardown. Exercise public generation with the intended layout
and confirm graph versus eager route selection and numerical/state parity.

### R0139 — Replace file-type policy predicates with actual operand contracts

Owner: `hipengine/loading/qwen35_gguf_policy.py`.

Scope: `hipengine.loading.qwen35_gguf_policy`, `GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES`,
`GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES`.

**Deferred policy migration:** inventory the stamp consumers in recurrent state, qmicro, NextN/GDN,
scratch/source-F16, Q6 and automatic MTP, then bind each to actual role/shape/type/layout and
execution-profile requirements. Separate kernel capability from arithmetic-default selection:
unfamiliar compatible artifacts must not require a benchmark record to execute. Do not use a
histogram or successful load as proof of an arithmetic contract. Preserve existing precision while
separating global IQ repack eligibility from F32 contraction.

Done when: test identical stamps with different maps, identical histograms with types
exchanged between sensitive roles, partial NextN plans and optional sidecars. Run shared-policy and
route-audit tests, then public loading/generation for compatible unseen tensor maps and explicit
incompatible requests. Any changed arithmetic default must pass its profile gate. Preserve
source-bound certificate consistency without turning source identity into measured-artifact
admission.

### R0147 — Consolidate the Q8 MMQ attention-gate candidate selector

Owner: `hipengine/runtime/qwen4_exp_runner.py`.

Scope: `HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE`.

**Deferred selector decision:** preserve the explicit comparison or replace it with an
operation-complete dense/GR policy owner. Remove the separate setting once that owner represents the
same shape and lifetime requirements. Preserve exact raw-Q8 fallback and keep this scope independent
of prepack/resource selection. A performance-driven default change needs stabilized comparisons; a
changed-arithmetic default also needs its execution-profile gate.

Done when: run attention-gate policy and profile unit tests, then exercise the enabled route
through public prefill and confirm actual dispatch. Check full logits/state, repeatability and
cleanup before retiring the comparison. Do not silently claim engagement from setting this flag
while the MMQ owner is absent.

### R0151 — Separate prefill staging ownership from implemented AR batching

Owner: `hipengine/generation/qwen4_exp_gguf.py`.

Scope: `Qwen4ExpResidentServingRunner._prepare_ar_batch`, `_step_ar_batch`.

**Deferred staging design:** determine whether prefill and remaining c1/MTP consumers can use
smaller dedicated workspaces or shared state storage without losing independent request ownership.
Inventory reserve/import/demotion and reclaim consumers before replacing allocations. Preserve
shared-once weights and PLE, request-local KV/state, cancellation and failure cleanup. R0006 owns
the shared-decode follow-ups; do not reopen its completed batch implementation.

Done when: run AR-batch admission/capacity/decode unit tests and public mixed-prefill/decode
requests, including transfer, cancellation and continuation. Confirm actual batch engagement and
staging allocation lifetimes.

### R0152 — Expand vision capacity and integrate typed media ownership separately

Owner: `hipengine/runtime/qwen4_exp_vision.py`.

Scope: `max_patch_tokens`.

**Deferred capacity work:** connect larger supported grids to explicit memory admission and validate
them against the vision reference before changing public limits. Profile the existing attention
implementation before selecting a faster replacement; do not treat changing the limit as providing
its storage or numerical contract.

**Separate lifecycle integration:** replace the dedicated multimodal chat branch only when the
normal request owner carries typed media, cancellation, exact token accounting and streaming state.
Current HTTP checks reject streaming, n other than one, tools, sessions, continuation, grammar and
speculative generation. Those missing semantics are not implemented by widening vision buffers.
Remote media handling needs an explicit fetching/resource policy; preserve bounded inline decoding
until that exists.

Done when: run `tests/test_unit_qwen4_exp_multimodal.py` and
`tests/test_integration_qwen4_exp_server_multimodal.py`; add grid-boundary, allocation-failure and
mixed-request lifecycle coverage for the chosen extension. Run vision reference/device parity and
real public HTTP media requests, including cancellation and malformed inputs. Remove only
restrictions whose replacement semantics actually run.

### R0163 — Decide the scope of the FP16 recurrent-storage experiment

Owner: `hipengine/generation/qwen35_dense_gfx1151_profiles.py`.

Scope: `hipengine/generation/qwen35_dense_gfx1151_profiles.py`.

**Deferred storage decision:** retire the unbounded FP16 experiment or replace it with a
geometry/profile/horizon-owned storage plan. A replacement must pass full long-horizon numerical and
serving gates before changing the arithmetic default. Do not re-enable it by copying the historical
D24 verifier manifest. Keep registered FP32 execution and distinguish other configurations from the
recorded failing scope rather than treating all FP16 routes as failed.

Done when: run dense gfx1151 profile unit tests, then compare complete recurrent state,
logits, continuation and accepted-prefix rollback across short and long histories for any
replacement. Exercise public AR and packed MTP with storage attribution and lifecycle checks.

### R0173 — Verify owner-bound R28 dispatch on gfx1100

Owner: `hipengine/generation/qwen35_dense_gfx1100_profiles.py`.

Scope: `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28`.

**Remaining hardware check:** exercise strict, production and explicit rollback through public
generation on gfx1100 in both owner construction orders. Confirm the selected R28 implementation,
periodic exact-layer schedule, full outputs/state and graph/session teardown. Common transport
execution on gfx1151 does not exercise the gfx1100 R28 kernel.

Done when: the gfx1100 public comparisons preserve the declared arithmetic selection, strict
fallback and lifecycle behavior. Do not remove the shared `_rowtile_variant_policy_env_cache`:
unbound/direct diagnostics and unrelated rowtile policies still use it.

### RF-1 / R0187 — Finish owner-bound policy resolution using existing profile infrastructure

Owner: `hipengine/execution_profiles.py`.

Scope: `hipengine/execution_profiles.py`, `gguf_linear.py`.

**Deferred ownership design:** extend the existing plugin-owned construction boundary with typed
resolved decisions and explicit resource/shape-dependent plans. Identify which values belong to a
model, session or request and which must vary with live shape. Parse user environment input at
adapters, not in hot methods; preserve explicit scoped diagnostics without ambient-state leakage.
R0149 owns the remaining concrete Qwen4Exp transport migration under this goal.

Done when: test independently configured owners in both construction orders, interleaved
requests, strict/production selection and unsupported keys. Verify migrated hot methods no longer
reread their moved settings and exercise public generation with selected-route attribution. Existing
profile unit tests cover the current abstraction, not completion of this migration.

Registry misses may use only a declared fallback supporting the requested semantics; missing
capabilities must fail explicitly. Preserve four-axis ownership and identical eager/capture
decisions.

### RF-3 / R0187 — Decide source-script retention and replacement ownership

Owner: `scripts/`.

Scope: `scripts/` — Decide source-script retention and replacement ownership.

**Deferred decision:** choose the supported source-command families and their replacement entry
points before consolidating or retiring historical drivers. For each proposed deletion, record the
path, current callers and published protocol references, replacement command (or why none is
needed), and the immutable result hash and decision it reproduces. Preserve active oracles and
commands required to reproduce evidence. Parameter-only drivers may consolidate only after their
argument, output-schema and failure contracts are compared.

Done when: a path-by-path disposition inventory, replacement command help and focused driver
tests, published-command drift checks, and verification that each retained evidence reference is
still reproducible or explicitly archived. Do not delete an unnamed batch based on the old count.

### RF-5 / R0187 — Choose compact decision schema and durable capture storage

Owner: `benchmarks/schemas/`.

Scope: `benchmarks/schemas/` — Choose compact decision schema and durable capture storage.

**Deferred storage/schema decision:** choose a retrievable destination for large captures and define
which fields belong in compact decisions versus referenced payloads. The proposed default decision
ceiling is 256 KiB, not an existing validation rule. Large accepted JSON results cannot be treated
as disposable logs merely because they exceed it. The provenance and published-command gates do not
impose that size limit.

Before migration, inventory paths, consumers, references, hashes and verdicts; separate raw
tensors/traces from summaries that existing validators consume. Keep current and linked decisions
available. External or LFS payloads need a SHA-256, retrieval metadata and an exercised retrieval
command. Ignored local storage alone is not a durable replacement for published evidence.

Done when: version the compact schema, test size enforcement and malformed references, replay
representative accepted/rejected decision validators after retrieval, and verify all existing
evidence links or explicit archival redirects. Only then remove duplicated captures from tracked
storage. Preserve historical bytes and decisions; do not rewrite measured results to fit a size
target.

### RF-8 / R0187 — Verify Laguna dispatch after policy-owner migration

Owner: `hipengine/kernels/policy.py`.

Scope: `hipengine/kernels/hip_gfx1151/policy_laguna.py`,
`hipengine/runtime/laguna_gguf_runner.py`.

**Remaining model-backed check:** obtain a local Laguna checkpoint (absent; see R0273)
and compare public prefill/decode/MoE routes, outputs and teardown against the
alias-bound control. Policy declarations, package consumers, source readers and backend
wrapper reads now use the declared ordered owners; package policy aliases are removed.
Explicit package overrides, including `None`/`False`, still take precedence.

Done when: exercise Laguna through its public generation surface and confirm the named
attention and MoE implementations ran with unchanged outputs and resource ownership.
Static registry identity and the completed GGUF AR/MTP checks do not replace this
model-backed check. Do not add an admission gate while the checkpoint is unavailable.

### RF-9 / R0187 — Partition Laguna attention with explicit build ownership

Owner: `hipengine/kernels/hip_gfx1100/attention/laguna_kv.py`.

Scope: `hipengine/kernels/hip_gfx1100/attention/laguna_kv.py`, `laguna_kv_attention`.

**Deferred build-boundary decision:** define separate family ownership for KV write, global
attention, sliding-window attention and test-only experiments. Inventory each exported symbol,
registration, runtime caller and oracle before moving or deleting it. A body without a public Python
wrapper may still be a native helper or exported test control. Preserve registered strict fallback
chains and active negative controls; archive only genuinely unconsumed code.

Done when: preserve production names, raw-pointer and KVLiveSpans ABIs, source lineage and
exact registry mappings. Explicitly account for build-cache keys, cached-only operation, library
lifetime and graph-held symbols after any source split. Run affected family parity/lifecycle tests
and public generation with selected-symbol attribution; keep mechanical movement separate from
arithmetic or default changes.

### R0189 — Packed-PARO transaction storage after journal integration

Owner: `hipengine/generation/qwen35_paro_mtp2.py`.

Scope: `_checkpoint_target`, `hipengine/generation/qwen35_paro_mtp2.py`, `kv_rows=0`.

**Deferred storage design:** decide whether remaining target result, acceptance, selected-commit and
provider-update metadata and mutable backups should share stable claim-backed slabs. Preserve one
journal and explicit buffer lifetimes; do not replace the working rollback with a second
coordinator. Verify exact state/cursors/live KV on rejected, partial and full acceptance,
capture/restore failure, cancellation and request reuse before retiring the spare-slot path.

Done when: Any multi-request expansion must implement physical proposal and target ownership rather than
broaden the singleton adapter's admission. Compare full request wall time with matched AR/direct
controls when claiming a performance benefit.

### R0195 — Consolidate physical-target policy without changing admission

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS`,
`HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES`.

**Partial ownership migration:** exact rows, padding and Q4/Q5/Q6 rowtile choices
are immutable `PhysicalTargetRowsPlan` values bound by dense gfx1100/gfx1151
and MoE gfx1151 profile construction. Width/depth cells are also frozen on the
profile owner; adapter capacity clamping remains independent. Direct callers use
the same resolvers. See
[`row-owner checkpoint`](../worklog/entries/20261106T094204.248065Z-lhl-r0195-row-owner-2c7013.md)
and [`width-owner checkpoint`](../worklog/entries/20261106T095506.365433Z-lhl-r0195-width-owner-457c24.md).
Loaded-geometry prompt policy tables are frozen on those owners; adapter
construction selects widths with the existing file-type fallback. The adapter's
`physical_width_contract()` exposes JSON-compatible resolved row/arithmetic,
width/depth and prompt-width policy plus the last padding decision and its reason.
See [`prompt-owner checkpoint`](../worklog/entries/20261106T095747.652410Z-lhl-r0195-prompt-owner-934411.md),
[`policy contract`](../worklog/entries/20261106T095934.807046Z-lhl-r0195-policy-contract-cb3fdb.md)
and [`padding attribution`](../worklog/entries/20261106T100045.077441Z-lhl-r0195-shape-contract-6f075c.md).
These reports describe policy and planning, not actual kernel dispatch.
Profile construction now combines those snapshots in one immutable
`PhysicalTargetPlan`; adapters consume it while legacy fields remain compatibility
views. See [`unified plan`](../worklog/entries/20261106T101816.788220Z-lhl-r0195-unified-plan-30cfe8.md).
Public gfx1151 production and strict width 1–4 probes confirm same-width token
repeatability, packed shape accounting and clean teardown, not exact state parity.
The general registry manifest schema is unchanged. Exact device state/offset/
selector coverage, shape misses and wider gfx1100 device validation remain
outstanding.
Reuse RF-1/RF-10 ownership rather than add a parallel selector. Retire the two rollback flags
only with explicit coverage of their replacement policies and diagnostic controls.

Preserve padded rows6 decomposition, repeated-R6 and serial-row strict owners, ordinary-prefill
behavior, scope-miss fallbacks and unfused pair ownership. Keep synchronized callers synchronized;
enqueue-only proposal and bounded blocking accept readback have separate lifetime contracts.
Preserve independent physical-target and C1 policy fingerprints and historical evidence bounds as
provenance, not identity-based admission rules.

Done when: exact output/state and offset/selector tests across C1, C2/K3, C3/C4 and wider
supported groups, including shape misses and strict profiles; public generation must attribute the
selected rows and fallback reason. Compare matched full-request category/heldout performance when
claiming a benefit.

### R0006 — Remaining Flash-Next batch capacity and scheduling decisions

Owner: `hipengine/generation/qwen4_exp_ar_batch.py`.

Scope: `QWEN4_EXP_BATCH_SLOT_POSITIONS`; `HIPENGINE_DECODE_PREFILL_TOKEN_BUDGET`; `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET`.

**Deferred capacity design:** `QWEN4_EXP_BATCH_SLOT_POSITIONS` still caps batch slot extent at
65,536 positions. Page per-position index storage through the shared block table before removing
that cap; preserve the c1 longer-request path until the replacement storage supports it. Check
below/at/above the cap, resource refusal, reuse and public route selection.

**Deferred scheduling policy:** choose a measured layer-time/chunk policy that bounds active-decode
prefill interference without operator tuning. The live `HIPENGINE_DECODE_PREFILL_TOKEN_BUDGET`
control remains distinct from idle prefill and the ordinary `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET`
override. Before retiring it, run `scripts/qwen38fn_prefill_interference.py` through `hipengine
serve` over short, medium and long prompts with zero, one and multiple decoding peers; satisfy the
campaign's frozen latency and prompt-cost criteria.

**Selected-down ownership implemented:** profile construction publishes the effective
Q8 selected-down choice on generator and runner; new lanes inherit it, and dispatch reads
owner state rather than `HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN`.
Production/recovery/restoration decisions and strict grouped/GEMV comparisons are preserved.
Profile isolation, batch forwarding and tile-map checks pass; see
[selected-owner audit](../worklog/entries/20261106T103209.831222Z-lhl-r0006-selected-owner-audit-6aa8a3.md).
Public production generation reached the selected grouped blockscale guarded owner and
closed with zero tracked allocations; see
[public route check](../worklog/entries/20261106T103511.593935Z-lhl-r0006-public-selected-75483f.md).
Keep map rebuild and capacity checks.

Done when: paged index storage clears the cap with boundary/resource/reuse checks;
interference policy meets the frozen campaign criteria.

### R0007 — Batch import-copy and admission ownership after reset repair

Owner: `hipengine/generation/qwen4_exp_ar_batch.py`.

Scope: `import_prefilled`; `_owners`; `_positions`.

**Copy-plan extraction implemented:** `import_prefilled` builds explicit source/destination
block-table copy plans for QSA keys, values and index keys before the first device write. Plans
preserve each block's valid prefix; GDN and PLE copies retain their existing ownership. Execution
still uses host-issued device-to-device copies. A batched device operation remains separate work;
profile the import boundary before claiming a benefit from a new kernel.

**Remaining admission design:** the runner now stores owner token and cursor in one per-slot
admission record; release detaches it and reuse creates a new record. Request leases reference
that same record rather than copying physical slot and owner fields; mixed-target consumers use
checked runner methods. Owner row planning preserves the caller's request order through these
records. Device validation passed the batch runner and serving fixtures (one pooled-capacity
fixture repaired); the named CPU bundle passed. Public gfx1151 production reuse with
`protect_ttft`, two requests and a fixed horizon confirmed two imports and seven width-two
steps per call, matching repeated token IDs and zero tracked allocations after close.
Default scheduling produced different shape sequences and tokens; this is not evidence of
same-shape failure or batch-composition invariance. See
[public validation](../worklog/entries/20261106T102915.238841Z-lhl-r0007-public-admission-0fa288.md).
Preserve physical-slot identity, cancellation/reuse, preflight failure and poisoned-step cleanup;
publishing partial admission is not acceptable.

Done when: Clearing gates remain `tests/test_unit_qwen4exp_batch_runner.py`,
`tests/test_unit_qwen4exp_ar_batch_decode.py`, `tests/test_unit_qwen4exp_ar_batch_capacity.py` and
`tests/test_gpu_qwen4exp_ar_batch_decode.py`. Verify dispatch shape, copy prefix layout, reset,
capacity/isolation, MTP exclusion, two imports/two-row steps, per-row forced-token identity and
c1-chain agreement. Exercise public serving and teardown, and keep repeatability distinct from
batch-composition invariance.

### R0010 — Flash-Next physical MTP ownership and shared-target regression

Owner: `hipengine/generation/qwen4_exp_gguf.py`.

Scope: `tests/test_integration_server_controlled_speculation.py`,
`tests/test_unit_qwen4exp_mtp_submission.py`.

Done when: Deferred ownership design: replace single-cycle exclusive target/draft ownership only with a
snapshotable multi-request frontier that can restore surviving peers independently. Logical
concurrency and candidate rows are not physical request width. Resident-session reuse separately
requires explicit session ownership and rollback; explicit unsupported requests must fail before
work, not silently switch to K0. Verify cancellation, isolation and recovery through public requests
before changing either contract.

**Observed regression, deferred optimization:** `HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET` remains
opt-in because the measured gfx1151 eager shared pass was slower than separate execution. Python
layer submission, per-row/depth GDN copying and loss of AR graph replay are the recorded costs, not
proof that shared execution is unimplemented. Clearing command: `scripts/qwen38fn_mixed_economics.py
--widths 1,2,4,8 --arms ar,mixed` against `hipengine serve` with the variable set and unset. Retire
the rollback and enable the shared default once it meets or beats separate execution at every width,
with control/ownership gates intact.

### R0083 — VibeVoice prefill orchestration and device-tail ownership

Owner: `hipengine/kernels/hip_gfx1100/vibevoice/q4.py`.

Scope: `q4_prefill`; `hipengine/kernels/hip_gfx1100/vibevoice/q4.py`; `_prefill_batched`.

**Deferred orchestration design:** unify `q4_prefill` in
`hipengine/kernels/hip_gfx1100/vibevoice/q4.py` and `_prefill_batched` in
`hipengine/runtime/vibevoice_qwen2.py` through explicit GEMM, activation-staging and scratch
dependencies. Preserve norm, RoPE, KV-write, attention, SiLU and residual order and dtype
conversions; do not combine a mechanical extraction with a changed-arithmetic promotion.

**Deferred capability work:** the Q4 batched route wires Q4_K/Q6_K and rejects Q5_K before decoding
it with the wrong layout. Implement and wire compatible Q5_K prefill before claiming that route
supports it; preserve explicit errors until then. This is an implementation gap, not a
model-identity restriction.

**Deferred profile ownership:** retain strict chains while moving `prefill_variant=hipblaslt` and
`frontend_variant=wmma` selection into profile construction. Promotion requires the model-specific
numerical/task gate; standalone evaluation must remain runnable before promotion.

Done when: Deferred device-tail storage: replace host snapshots in `hipengine/runtime/vibevoice_encoder.py`
only while preserving whole/chunk, 60-second transition, recording-reset and post-join-noise
behavior. Verify buffer ownership, failed-call cleanup and public transcription with the selected
route before retiring host storage.

### R0183 — Decide wide-Q6 shared-weight shape policy

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `target_verifier_wide_q6_shared4_session()`.

**Deferred candidate decision:** complete the C6/C8 strict-teacher, determinism/isolation/task,
full-suite performance and profiler gates before promoting or retiring the shape policy. Preserve
the recorded no-promotion results in
`benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q6-shared4-candidate.json` and
`benchmarks/results/2026-09-01-gfx1151-qwen38-w1-q4-existing-candidate.json`.

Done when: Preserve the distinct two-wave/two-tile gfx1151 `shared4` and four-wave/three-tile gfx1100
`shared4_gfx1100` symbols and ABIs. A future promotion must leave one immutable profile/capability
selection and registered strict fallbacks; a rejection must remove only unconsumed candidate
selection. Verify nested disabled scopes and exception restoration as well as public selected
routes.

### R0188 — SPECDEC2 whole-request compatibility seam

Owner: `hipengine/generation/engine_loop.py`.

Scope: `SubmitPollTextGenerator.submit_speculative_many_detailed()`, `VERIFY_CHAIN`.

**Provider inventory:** Qwen GGUF exposes staged adapter hooks through its resident runner;
Flash-Next exposes capability and cycle hooks through `Qwen4ExpServingMTP`; Gemma's built-in
provider exposes `generation_steps` through `Gemma4ResidentModelRunner` buffered decode.
Laguna DFlash uses generic children (R0011), not MTP compatibility. Providers attached without
these bounded interfaces can still reach whole-request compatibility. Inventory their attachment
and caller contracts before removing synchronous selection, post-hoc work metadata and duplicate
request/result ownership. Mixed-parameter compatibility admission now rolls back earlier children
on a later admission failure; see
`worklog/entries/20261106T090212.240760Z-lhl-r0188-admission-rollback-869685.md`.
The former RF-10 planning entry delegates direct-control retirement to this provider migration;
it does not require a separate scheduler implementation.

Done when: Preserve the [`SPECDEC2.md`](reference/SPECDEC2.md) S3 direct-c1 ID/cycle and exact
hidden/Conv/GDN/KV/cursor/following-AR gates, and S4 physical-c2/c4, late-admission/refill,
streaming, cancellation and ownership gates. Keep named direct controls and registered
strict/eager/serial kernel fallbacks. Verify staged providers never enter whole-request
compatibility and unsupported modes fail explicitly. Exercise migrated providers through public
generation before retiring their adapter; no provider support or runtime behavior changes here.

### R0190 — Retire prompt replay after provider lifecycle migration

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `_prompt_hidden_rows`.

**Deferred provider/lifecycle migration:** verify streamed prefix restore and copy-on-write, every
supported incremental-prefill/context transition, exact provider fingerprints, cancellation/failure
cleanup and complete same-suite non-regression. Then remove `_prompt_hidden_rows` and
`_catch_up_provider{,_batch}` from production selection, preserving a focused oracle where useful.
Inventory all replay consumers before removing storage.

Done when: The clearing gate must exercise `hipengine serve` with restored prefixes, incremental prefill,
cancellation and provider cleanup and confirm streaming actually ran. Unit adapter routing alone
does not close it. Keep explicit capability failures and strict replay semantics until consumers
migrate.

### R0191 — Move live MoE pair-reuse selection into profile ownership

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `moe_physical_c2_pairreuse_session(...)`, `x_rows*8`.

Pair-reuse selection is now a field of the immutable profile-owned arithmetic plan,
carried by `PhysicalTargetPlan` and bound around physical verification. Nested diagnostic
scopes remain supported. The MoE factory prefers that plan over stale compatibility fields;
see [factory precedence](../worklog/entries/20261106T104009.319250Z-lhl-r0193-moe-plan-precedence-5fb3a8.md).

Done when: preserve nested disabled-scope restoration and geometry checks through public
validation before retiring the context. Preserve row-bound/interior,
selected-row mismatch and incompatible-geometry tests. Verify public exact state, acceptance,
cancellation/reuse and selected kernel attribution before retiring the context. The original final
three-repeat performance gate still governs a claimed promotion or rejection. A losing candidate may
be removed only after checking live consumers and preserving strict fallbacks.

### R0193 — Replace MoE numerical-scope transport without changing arithmetic

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `moe_physical_c2_numerics_session(...)`.

Construction now binds an immutable arithmetic plan on dense gfx1100/gfx1151
and MoE gfx1151 profile owners. Physical verification carries it through one
request-local plan binding; supported diagnostic scopes replace one field and
restore the previous plan. See
[`transport checkpoint`](../worklog/entries/20261106T093416.260440Z-lhl-r0193-plan-transport-c46791.md)
and [`profile-owner checkpoint`](../worklog/entries/20261106T093803.590732Z-lhl-r0193-profile-owner-89e0f8.md).

Done when: finish public validation of the integrated physical-target plan while
preserving C1/strict fallbacks, nested disabled scopes, exception restoration and independent
execution-context isolation. Reuse RF-1's owner boundary and keep this migration separate from
numerical tuning. Run `tests/test_unit_moe_target_arithmetic_scopes.py` and affected adapter tests;
verify public C1/C2 selected arithmetic and rollback state before retiring the scope.

### R0194 — Preserve exact-linear ownership during profile transport migration

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `moe_physical_c2_exact_linear_session(...)`.

Exact-linear selection now travels in the immutable profile-owned arithmetic plan and
unified physical-target plan; construction and diagnostic scope isolation have host coverage.

Done when: validate that transport publicly while preserving selected-state journals, C1
arithmetic, nested scopes, exception restoration and
execution-context isolation. Verify public selected arithmetic, exact rollback state and lifecycle
before retiring the context. Keep this transport change separate from replacement-kernel tuning. Any
replacement must independently pass the complete numerical/lifecycle gates and the original >=1.10x
AR target; the current correctness fallback remains until then.

### R0201 — Public repeat after planar-Q6 selector cleanup

Owner: `hipengine/runtime/gguf_linear.py`.

Scope: `hipengine/runtime/gguf_linear.py` — Public repeat after planar-Q6 selector cleanup.

Done when: Deferred validation: perform the later-release defaults-only, same-protocol two-pair full-suite
C1-C3 public repeat with all categories and heldouts, exact trajectories, native groups, trace
identity and drain checks. Preserve backend policy and the registered one-wave fallback.

### R0203 — Extract the explicit device-chain diagnostic oracle

Owner: `hipengine/generation/qwen35_gguf_mtp2.py`.

Scope: `device_chain_qualification_oracle=True`.

**Extraction implemented:** the comparison body lives in
`hipengine/benchmark/device_chain_oracle.py`, lazily reached through the existing explicit
`device_chain_qualification_oracle=True` adapter route. The default route does not import it.
Focused CPU tests cover reject, partial and full acceptance plus mismatched summaries. Public
physical-C2 generation has exercised the extracted oracle; the complete device/category matrix
and default no-pre-accept-D2H trace below remain to be verified.

Done when: Preserve pure CPU acceptance and strict eager
fallback. Retain P4 reject/every-partial/ full, following-AR, cancellation/failure, profile and
full-category checks with GPU/CPU accept agreement. Cache-only traces must show no pre-accept D2H in
the default route. Public acceptance and lifecycle verification remain required; constructor
isolation tests alone do not establish them.

### R0204 — Decide the typed request-local ngram composition candidate

Owner: `hipengine/speculative/ngram_mod.py`.

Scope: `ngram_enabled=True`, `ngram_config=NgramModConfig(...)`.

**Deferred candidate decision:** keep the explicit default-off candidate's recorded failures
visible. The strict D96 SQL terminal-suffix failure and production D120 numerical-tail rejection are
not repaired by typed settings.

Done when: A wider K>3 frontier or materially different lowering must pass canonical and code train/heldout,
long-horizon strict/production numerical, task and lifecycle gates and beat true AR. Otherwise
retire runtime composition after the dedicated wider-frontier campaign, preserving cache/harness
research references and permanent ordinary MTP/AR miss/failure fallbacks. Do not conflate this
candidate with other request-local prompt lookup implementations.

### R0205 — Finish profile-plan coverage before retiring compatibility selection

Owner: `hipengine/execution_profiles.py`.

Scope: `strict|production|batch_invariant`, `LLM.execution_profile_manifest_sha256`;
`HIPENGINE_EXECUTION_PROFILE`.

Done when: Deferred plan-coverage migration: register plans for every shipped model/backend/quant,
re-certifying or replacing non-exact defaults under their arithmetic gates. After a release window
verifies strict fallback and batch-invariant behavior, retire omitted-profile legacy selection and
duplicate environment adapters, retaining registered strict fallbacks. Verify public resolution and
fallback attribution across registered, failed and missing-plan cases before removing compatibility.

### R0206 — Per-primitive width planning and optional D2 promotion

Owner: `hipengine/generation/qwen35_gguf.py`.

Scope: `GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS`; `HIPENGINE_GGUF_AR_D2_COST_ARTIFACT`.

**Deferred planner design:** consolidate capability maps, sidecar cases and manual ladders through
cold registry/plugin resolution without flattening per-primitive limits into one model-wide bound.
Q5T16 and planar-qmicro Q6T16 rowtiles cover through eight rows; FP32 LM-head standard-Q6 and
planar-qmicro bounds differ and must still intersect backend policy. Keep shape/layout misses and
registered strict kernels explicit.

Done when: Before retiring sidecars, row caps or ladders, require actual-server direct/masked/composed c1-c32
telemetry, matched counterbalanced D2-versus-ceiling goodput and TTFT/ITL, refill/cancel, memory and
drain checks.

### R0207 — Migrate remaining chunk-local KV consumers before allocator retirement

Owner: `hipengine/kvcache/`.

Scope: `dense_pool`, `layer_outer`.

**Deferred storage migration:** replace base-pointer execution with page-table views across chunks
and replace DMS transient ownership before removing unused legacy factories, chunk allocation or
identity branches. Keep same-arena checks until snapshot restore proves equivalent ownership;
missing benchmark coverage is not a refusal reason.

Done when: `uv run pytest tests/test_unit_kvcache_global_device_pool.py
tests/test_unit_gguf_vmm_kv_pool.py tests/test_unit_gguf_global_pool_growth.py
tests/test_unit_gguf_device_kv_binding.py tests/test_unit_gguf_prefix_cache_runner.py -q`. Then
exercise real ordinary/DMS growth, prefix reuse, cancellation and teardown. Preserve the original
BF16 long-context and no-mirror INT8 global-adapter gates on both gfx1100 and gfx1151:
graph-pointer, lifecycle, cancellation, pressure, SLO and production load. Only after remaining
registered packages migrate may compatibility fixtures retire. Registered strict/unfused/serial
numerical fallbacks remain independent of allocator cleanup.
