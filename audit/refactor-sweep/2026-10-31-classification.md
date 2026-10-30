# Refactor referral classification — 2026-10-31

All current `docs/REFACTOR.md` headings were reviewed against their scoped worklogs and source evidence. The JSON [row inventory](2026-10-31-classification.json) records each next action, completion gate, evidence and named inputs. Its source snapshot is `c7b7e695c`; the ledger hash identifies the corrected text. This is an audit snapshot, not a second live backlog.

## How the categories work

Classify **what happens next**, separately from **what resources completion needs**. A GPU check after a code migration does not make the migration hardware-blocked. A missing checkpoint can prevent a public run without preventing fixture, API or ownership work. An engineering choice is not automatically a request for lead approval.

For mixed entries, mandatory implementation takes precedence over an independent optional disposition. If keeping a supported comparison would close the item, classify its interface as a diagnostic disposition rather than assume retirement. Profiling and reproducing an unknown failure are not validation-only work. Categories are reviewed judgments with an explicit row inventory, not keyword counts.

| Next work | Headings | Share |
| --- | ---: | ---: |
| Implementation / harness / repair | 114 | 40.3% |
| Diagnostic or rollback disposition | 91 | 32.2% |
| Product / architecture / resource choice | 27 | 9.5% |
| Profiling / reproduction / candidate evaluation | 21 | 7.4% |
| Validation of implemented work | 29 | 10.2% |
| Historical evidence binding | 1 | 0.4% |

Total: **283 headings**. No whole heading was verified complete. This does not mean all work is immediately executable or all completion gates have passed.

## Corrections to the proposed classification

- **R0428 is not closed.** The CLI repairs are complete, but the unroll replay interface still has a retention/removal decision and live export/wrapper/parser consumers.
- **R0435 is not closed.** Reporting the default correctly did not retire or migrate the full/bounded-reset comparison and its lifecycle obligations.
- **R0446 is not closed.** Cache invalidation clears all caches; owner-pinned handles versus live environment observation remains a distinct lifetime decision.
- **R0460 is complete.** Removed its obsolete manifest prerequisite from R0426 and R0459. The other subitems remain open. RF-4 no longer waits for coverage reconciliation; its semantic checker is still unimplemented.
- **The five “not an evidence gate” entries remain real work.** R0214/R0215/R0378 is control migration with later small-model checks; R0231 needs CUDA resident orchestration; R0462 combines a storage choice with sparse-gather implementation; R0471 needs an invocation representation; R0479 owns a distinct parity-reference contract. Re-filing these does not complete them.
- **The twelve “plain bug repairs” are mixed.** R0067 is retention-budget policy after eviction repair; R0404 is evaluation after dtype repair; R0444 combines multiplier ownership with grid profiling; R0210 is allocation migration after page-map repair; R0231 is missing CUDA orchestration; R0001 is a verifier-graph performance investigation. R0026/R0027 concern disabled numerical candidates, not evidence that the current default is broken. R0511 still needs acquisition-level fault injection, not an assumed root cause. R0385 can start with subprocess containment before its model-backed reproducer.
- **Diagnostic retention is not dead-code status.** Many rows name numerical oracles, rollback interfaces or live harness callers. Their owners can inventory and decide them without individual product meetings, but a blanket retire-by-default policy would discard supported comparisons.

## Resource checks and acquisition leverage

The bare shell cannot load `libamdhip64.so`, but the existing therock environment loads HIP and enumerates a gfx1151 Radeon 8060S. It also provides `rocprofv3`. Corrected the ledger’s claims that the local ROCm runtime/profiler was unavailable. These checks establish environment availability, not kernel correctness or model capacity.

Surya resolution still raises `FileNotFoundError`. Laguna weights are absent from the checked standard model directories; the tokenizer alone is not a checkpoint. No filesystem-wide or remote-host inventory was performed. gfx1100/XTX and CUDA checks cannot be replaced by a gfx1151 result.

Of the proposed fourteen Laguna entries, **seven are validation-only**, **four need diagnostic dispositions**, and **three need implementation**:

| Next work | Proposed Laguna entries |
| --- | --- |
| Validation-only | R0249/R0250, R0273, R0274, R0275, R0248, R0276, R0281-D12 |
| Diagnostic disposition | R0258, R0259, R0264, R0269 |
| Implementation | R0353, R0354, R0341 |

A suitable checkpoint enables model-backed checks; acquiring it does not close those entries. Some also require gfx1100, particular quant/layout coverage or a separate DFlash drafter. Other Laguna entries outside the proposed fourteen also have model-backed completion checks.

The proposed missing-input list is not exhaustive: R0214/R0215/R0378 and R0239 need the small Qwen Q4_K_M/Q8_0 comparison inputs, R0385 names an original model fixture, and R0060 needs identification of the actual historical measured artifact. These are different dependencies; they are not satisfied by any arbitrary GGUF. Surya R0493 can begin by extending the A4 oracle case table, although capture still needs the checkpoint.

## Validation batching

The validation-only rows are listed below. Batch setup by backend, model/quant, public entry point and compatible state/trace protocol, not by whether an entry mentions `LLM.generate()` or `hipengine serve`. Most dense physical-target checks can share model setup, but each still needs its requested route, shape boundaries, output/state and teardown evidence. Keep XTX capacity work separate. Keep R0385’s potentially corrupting reproducer in an isolated process; it is not a routine shared-session validation.

R0276 deliberately sets retired tile environment variables to prove they are inert. Replacing that test with active primitive tile arguments would test a different contract.

## Per-entry index

The JSON inventory carries the evidence and completion details. IDs below preserve combined headings; they are not counts of individual original sweep IDs.

### Implementation / harness / repair

| Entry | Next action |
| --- | --- |
| R0023 | Move colwarp transport into runner ownership; separately address each family repair or candidate disposition under its existing gate. |
| R0029 | Add a plan-level producer option and migrate the four gate harnesses together; today every arm selects its producer by writing HIPENGINE_QWEN4_EXP_ROUTER_VARIANT into the child environment and the runner reads it at launch. |
| R0028 | Carry GR_WIDE as an owner-bound variant input across profile construction, dispatch and diagnostics instead of the launch-time read at qwen4_exp_runner.py:5070; keep the rows > 256 and low_rank % 64 == 0 checks. |
| R0027 | Localize and repair the GR_IU8_DOWN projection drift, then screen one prefill per arm with scripts/qwen4exp_gr_iu8_logits_probe.py below the ~5e-9 prefill-boundary KL limit before the 780-row depth gate. The route is off in every production binding, so this is candidate repair, not a live defect. |
| R0026 | Localize and repair the GR_IU8 projection drift without touching the exact sigmoid/gated-mean epilogue; screen one prefill per arm with scripts/qwen4exp_gr_iu8_logits_probe.py below ~5e-9 prefill-boundary KL, then run the complete 780-row depth gate. |
| R0035 | Replace the launch-time read of HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT (qwen4_exp_runner.py:7044) with direct profile-owned variant keys across construction and launch callers. |
| R0051 | Inventory the prefill, one-row prefill tail, incremental decode and verification attention selectors on both backends, then decide and implement whether backend decode routing and strict attention binding share one scope contract. |
| R0080 | Run the explicit compatibility transition: migrate direct callers, configuration documentation and resolver/reporting tests to HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET, keeping both names agreeing when present, before removing the legacy alias. |
| R0081 | Publish the common per-request diagnostics contract from each provider, migrate both blocking and streaming consumers, then remove the legacy diagnostics.specdec2_mtp2 readers and scalar mtp_* timing fallbacks one at a time. |
| R0184 | Carry the verifier-Q4 rowtile selection through the existing profile infrastructure into shared, direct, batch and NextN construction instead of the global environment value read at gguf_linear.py:341. |
| R0213 | Inventory the remaining mirrored INT8 callers and migrate them to the replacement path (or explicitly retire them) before removing the mirror allocation/copy/dispatch code; the direct batch leaf already exists. |
| R0214/R0215 / R0378 | Replace the harness's bulk/X6/X8 environment controls with scoped diagnostic overrides that restore on success and exception, migrate the threshold children, remove the production readers and the Q8 pair-cache environment dimension, and update ENVS. This can start now; the checkpoints are only needed for the public step. |
| R0219 | Integrate a request-row-scoped source-F16 owner with packed scratch lifetimes and ragged requests into prefill_batch_native(); ordinary prefill() opens the source-F16 context but the packed entry does not, and public serving uses packed prefill. |
| R0231 | Implement a CUDA resident-batch owner with independent per-request span/KV arenas and CUDA libraries, wire the existing batched QK/KV-write and attention leaves plus row-reuse head execution, and enable it through the public factory. Do not flip the factory boolean or route CUDA through HIP builders. |
| R0278 | Expose offline DMS export under a model-plugin-owned capture API (moving scripts/qwen38_dms_capture.py and tests/test_live_dms_capture_gguf_hip.py together) or retire that offline API with its CLI and training/label consumers, keeping the serving device collector and its callback path. |
| R0279 | Choose and implement a bounded reversible mutation log or a resource-claimed device-local journal in hipengine/kvcache/dms.py, hipengine/kvcache/dms_device.py and _DMSStoreJournal, so general eviction/compaction no longer copies owned extents to host. |
| R0280 | Add CPU-oracle RED tests for the exact masks (ties, non-finite rejection, window edges, non-aligned lengths, per-head counts), then register device primitives and integrate the collector/decode owners so ranking and live-count/position/eviction finalization have one owner. |
| R0281-PARO-WIDTH | Mixed. (a) Separate HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS (also selects per-layer linear-output, attention and projection defaults) from HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE (native decode), migrate their benchmark/test callers, then retire the production readers; a rename alone does not merge them. (b) Replace the model-identity/measured-range admission in the native width loader with backend/quant/KV/shape capabilities plus recorded concrete failures. |
| R0281-SILU-FLOOR | Add nested/error/thread isolation and populated-cache control/candidate/control tests, then introduce an execution-local prefill-vs-verifier policy that binds ordinary, packed and resumable prompt entries to one owner and keeps verification outside it; do not lower the prefill floor as a side effect. |
| R0281-AR-LOOP | First give the shared flush operations an owner both routes can consume, then migrate or retire both direct-generator batched compatibility loops (Qwen35GGUFBringupGenerator.generate_detailed() and _generate_ar_serving_slots()) to an explicit diagnostic adapter with their callers, and only then extract loop-specific methods. |
| R0281-ROUTER | Publish the actual selected router variant and counter ownership in a construction-time plan consumed by the runner (part of the RF-1 resolved-plan direction), then migrate profile binders and capture metadata before removing HIPENGINE_GGUF_ROUTER_F32W_COOP / _PERSISTENT_COUNTER readers. |
| R0281-VERIFY-TIMING | Close the packed recorder across every post-creation exit (wrap creation/start through resolve in try/finally, as prefill and decode already do), then decide whether the opt-in belongs in an explicit profiling argument/helper instead of process environment. |
| R0281-AR-STREAM | Provide an explicit resolved serial/stream/packed comparison control or a designated serial oracle, then migrate the env reader, writers, route metadata and cleanup tests together. |
| R0281-MTP-DRAFT-EXECUTOR | Decide whether compatibility draft/verifier execution is retired through the existing loop extraction or given a bounded request/owner worker-pool lifetime; do not add another global switch. |
| R0281-TAIL-RMS | Give the per-layer MoE graph comparison an explicit unchained construction policy (or retire that route and its caller coherently) without disabling production norm chaining. |
| R0313 / R0314 / R0315 / R0316 / R0318 / R0320 | Migrate the six legacy Laguna MMQ selectors to explicit primitive ownership and validate the user-facing route; provide the Laguna artifact only for the later public validation. |
| R0319 | Migrate the Laguna serial compaction A/B harness and session setter to explicit primitive ownership before removing the serial override. |
| R0375/R0376 | Migrate the active cumulative/threshold/probe consumers of the two gfx1151 pack8 controls to explicit diagnostic policy before deleting either environment reader. |
| R0368 | Retain the explicit global-geometry comparisons or move them to a diagnostic build, and separate dead repair code/mask reporting from the shared production merge without changing partial-output storage or arithmetic. |
| R0354 | Implement the versioned paired Laguna expert cache payload: the writer must persist one interleaved pair instead of two independent tiles payloads, and _materialize_q4_t16_dual_pair must read it, while retaining raw/partial loading. |
| R0353 | Separate diagnostic-only value-tail instantiations/wrappers from the production build without removing the shared template or live predecessor oracles. |
| R0349–R0351 | Move the public global exp/mixed32 setters, cache-policy ladder and profiler mode/result readers into the diagnostic owner, preserving generic and peer-backend routes. |
| R0343–R0348 | Consolidate the six public SWA issue-schedule setters, cache ladder and profiler readers into the diagnostic owner, preserving generic shape and peer-backend routes. |
| R0341 | Migrate the Laguna natural tile8 constructor/setter and profiler comparison to a scoped diagnostic owner before retiring the public tile8 control. |
| R0340 | Move the process-global source-F16 decode comparison selection into a scoped diagnostic owner and migrate profiler environment restoration/result readers; the shape repair itself is already implemented. |
| R0336 | Migrate the dedicated Laguna packed-output comparison to an explicit diagnostic owner before retiring the constructor override and script. |
| R0333–R0335 | Separate Laguna Q4 P8 shape policy from explicit primitive references across resident constructors/setters, generalized comparison, leaf tooling and numerical fixtures, retaining non-P8 bodies for sub-512 chunks. |
| R0331 | Replace the three public Q6 prefetch/sum controls with explicit production/reference ownership across constructor, setters, generalized A/B/leaf and numerical fixtures. |
| R0330 | Separate production specialization from explicit reference ownership for integer_wmma/wmma_hoist_activation across the shared wrapper and generalized leaf before collapsing controls. |
| R0329 | Migrate the source-F16 boundary comparison without collapsing its arms, then remove the session setter only after primitive/composite ownership is explicit. |
| R0327 / R0328 | Retire the q6_qmicro_permute session/profile selectors only with their live comparison callers and explicit primitive/resident ownership. |
| R0324 / R0325 / R0326 | Migrate the three Q6 activation comparison controls (compact, half-row, skip-padded) and their numerical/A-B callers to explicit primitive ownership together. |
| R0323 | Migrate the fused selected-SiLU pack A/B caller to explicit composite/primitive ownership and retain its complete-state comparisons before removing the session setter. |
| R0322 | Migrate layout comparison ownership, materializer and session construction together for q6_qmicro/q6_qmicro_planar, retaining the legacy cache adapter and registered non-qmicro consumers. |
| R0321 | Migrate session/cache construction and all three Laguna attention A/B callers together, preserving primitive comparisons and historical readers. |
| R0312 | Replace the remaining head-major K/V and gapped-gather resource controls with the common scratch budget without dropping requested-context sizing, allocation rounding or partial-allocation cleanup. |
| R0311 | Migrate both family readers, the comparison script, bridge ownership and execution metadata to a resolved family-aware stream policy, finishing the recorded GGUF long-prompt stability protocol. |
| R0281-PARO-PROVIDER | Migrate manual provider/profile binding, explicit-chain harness selection, proposal/scoring metadata and private-head allocation ownership to a harness-owned policy, retaining selected-row reseeding, borrowed scoring, strict fallback and stale-owner rejection. |
| R0281-CONTEXT | Replace the prompt-priming/cache switches with an explicit benchmark context policy across their callers, preserving independent prompt/cache and host/device-seed comparisons. |
| R0281-ROWTILE-CONTROL | Audit and implement execution-owner isolation for module-global comparison state; preserve the supported scoped comparison interface. |
| R0281-SELECTED-DP4A / R0283 / R0288 | Replace the ambient selected activation-adapter selectors with an explicit caller-owned adapter plan across compatibility generation, bench/profiler and workbench consumers; preserve the optional raw/T16 capabilities. |
| R0380 | Implement typed speculative commit and diagnose native-graph divergence separately from deciding storage-override retention. |
| R0385 | Migrate the skipped 1024-crossing node into a bounded subprocess regression that executes the original body with its HIP/model preconditions, then reproduce and repair the sentinel crossing before removing the skip or handoff guard. |
| R0421 | Define distinct PARO and GGUF attention-key variant semantics, then compare a replacement against the current pin, repeated replacement and per_row arms before removing the gfx1100 registry pin. |
| R0425 | Finish the explicit owner-field migration for Qwen4Exp profile binding, provide an explicit selected-route getter, and update the layer gate to use it instead of environment ordering; only then remove the conflict map, binder-write tracking and warning/error control. |
| R0426 | Review variant-aware library lookup and complete any missing caller migration; separately decide wide-row selectors, census coverage and bitmap/j32 diagnostic ownership. |
| R0439 | Define geometry/quant fallback behavior and migrate both chunk-policy consumers; compare explicit overrides before tuning. |
| R0444 | Move the risk multiplier to an explicit owner without changing its semantics; profile repair-grid sizes separately before tuning. |
| R0450 | Move ordered QSA prerequisite selection into runner/profile state and migrate diagnostic callers. |
| R0471 | Represent optional DP4A quantization, actual operands and workspace lifetime in the selected-call intent. |
| R0484 | Build a span-aware unfused fallback; profile sparse-chunk optimization separately. |
| R0493 | Extend the oracle case table for A4; checkpoint availability blocks capture, not the table implementation. |
| R0494 | Unify prompt-count production and consumption while preserving explicit zero and legacy-plugin fallback. |
| R0500 | Design shared immutable weights and transactional per-row KV ownership for cross-branch batching. |
| R0503 | Implement a bounded ready-output handoff preserving first-chunk delivery, fairness and cancellation. |
| R0504 | Implement co-activation while preserving independent claims and sink lifetimes. |
| R0505 | Separate failed-stream state from speculative intent; independently decide retention of the split-refused experiment. |
| R0507 | Inventory raw compatibility callers and route them through one service owner before retiring the raw entry. |
| R0509 | Specify and implement shared byte accounting and coordinated eviction for provider/target snapshots. |
| R0510 | Implement bounded temporary warmup session ownership and explicit coverage accounting. |
| R0511 | Audit constructor acquisitions and add failure injection before repairing uncovered rollback gaps. |
| R0513 | Implement captured-accept metadata for every published token with replay/stream ownership. |
| R0514 | Implement forced-token device inputs and committed queue semantics. |
| R0516 | Diagnose and repair generated-history down-MMQ failure; GPU reproducer is part of repair, not a reason to postpone starting. |
| R0003 | Design interior-expiry-safe storage with coordinated reader, allocator and rollback semantics. |
| R0011 | Implement bounded generic-provider child lifecycle and migrate the Laguna/submit-poll adapters. |
| R0071 | Apply the execution-profile oracle to split-prefill repair and replace inequality pins only after positive coverage. |
| R0128 | Inventory and compose optional resource accounting; separately implement native sparse device-position selection. |
| R0131 | Coordinate live launch bounds, split count and all partial-buffer allocations. |
| R0132 | Diagnose the affected MoE discrepancy and consolidate actual native-context capabilities. |
| R0137 | Inventory profile consumers and migrate each moved setting to its model/session/request owner. |
| R0138 | Represent missing native adapters' actual operands and capture/resource lifetimes before lifting specific refusals. |
| R0139 | Replace stamp-based predicates with role/shape/type/layout contracts, preserving arithmetic-default requirements. |
| R0149 | Migrate remaining Qwen4Exp environment transport; repair Q4 DP4A arithmetic separately. |
| R0151 | Inventory live staging consumers and consolidate only compatible resource lifetimes. |
| R0152 | Implement larger-grid admission and typed-media lifecycle separately; each has its own public validation. |
| R0164 | Identify equivalent graph-cache lifetimes and consolidate only proven duplicates. |
| RF-1 / R0187 | Extend existing profile construction with typed owner-bound and live-shape plans. |
| RF-4 / R0187 | Implement semantic ledger validation; coverage reconciliation is no longer a prerequisite. |
| RF-6 / R0187 | Extract remaining bulk-attention, verification, graph and PARO ownership boundaries without duplicating owners. |
| RF-7 / R0187 | Extract cohesive HTTP composition units with explicit application-state dependencies. |
| RF-8 / R0187 | Separate immutable subsystem/quant policy declarations from the backend registration aggregator. |
| RF-9 / R0187 | Inventory Laguna symbols and split build/family ownership while preserving cache keys and graph-held library lifetime. |
| RF-10 / R0187 | Resolve typed shape/resource-bound launch plans; non-staged provider retirement stays under R0188. |
| RF-11 / R0187 | Replace broad mutable-source pins with scoped behavior contracts and mutation tests. |
| RF-12 / R0187 | Extract shared fixtures and split integration tests while preserving collection identities and ownership. |
| R0189 | Plan claim-backed transaction slabs without replacing the existing journal or breaking rollback. |
| R0195 | Consolidate physical-target decisions into an owner-bound manifest and migrate diagnostic controls. |
| R0197 | Replace ambient hot-vocabulary path selection with explicit automatic/off/validated-map inputs. |
| R0006 | Page per-position index storage and migrate arithmetic selectors; interference policy requires separate measurement. |
| R0007 | Implement prefix-bounded block-table import copies and unify admission ownership. |
| R0010 | Implement independent snapshotable physical MTP frontiers; separately diagnose shared-target performance. |
| R0083 | Consolidate VibeVoice prefill ownership, implement Q5 support and replace host-tail storage under separate contracts. |
| R0188 | Inventory and migrate remaining non-staged providers before removing whole-request compatibility. |
| R0190 | Migrate replay consumers and validate prefix/incremental lifecycle before removing prompt replay storage. |
| R0191 | Move live MoE pair-reuse selection into the physical-target manifest without changing arithmetic. |
| R0193 | Replace MoE numerical ContextVar transport with resolved physical-target ownership. |
| R0194 | Move exact-linear transport into the target manifest while preserving journal and fallback contracts. |
| R0203 | Extract the explicit device-chain oracle only after providing a replacement diagnostic route. |
| R0205 | Complete shipped profile-plan coverage before retiring legacy compatibility selection. |
| R0206 | Consolidate primitive-specific width planning; optional D2 promotion remains separately measured. |
| R0207 | Migrate remaining chunk-local and DMS transient consumers to page-table/lease ownership. |
| R0209 | Extract packed workspace references and teardown into a single lifetime owner. |
| R0210 | Migrate remaining private packed allocations to lease-backed ownership; page-map repair is already done. |

### Diagnostic or rollback disposition

| Entry | Next action |
| --- | --- |
| R0039 | Separate the two questions and dispose of the composition: keep the multi-column GDN capability (implemented, registered) distinct from the R02 per-prompt task finding in benchmarks/results/2026-09-15-journey-gdn-multi-restoration/task-review.json, and record whether that composition is rejected or repaired. Then retire HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT only once an owner-bound construction input preserves every profile and diagnostic caller. |
| R0037 | Inventory per-arm consumers of scripts/qwen4exp_disabled_family_sweep.py against the already-existing scripts/qwen4exp_disabled_selector_ranking.py, decide retain-or-replace, and only then delete the old driver after replaying the migrated interfaces. |
| R0034 | Decide whether the unregistered GR rowbatch8, rowbatch8-wave-scale and rowbatch16-wave-scale comparisons are worth keeping for a different register-resource class or cost model; if retired, remove exports, wrappers and exclusive _row_batch plumbing together after checking shared instantiations. |
| R0041 | Verify remap, full-cache warming and long-unique cold behavior, then decide whether HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS=normal remains the documented diagnostic rollback or is removed. |
| R0085 | Decide whether HIPENGINE_GGUF_PACKED_LAYER_OUTER stays as an operational bisection control or is retired; if retired, remove the reader and dispatch condition and move the service proof's explicit control into its test harness. |
| R0135/R0136 | Decide whether dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out is retained as an explicit comparison alternative with a named diagnostic owner, or removed with its registrations and route-specific workspace/build plumbing. |
| R0271 | Decide whether the prefill_350_candidate comparison in scripts/laguna_grouped_down_category_bench.py is retained as intentional reference code with a named diagnostic owner or removed with its executable scaled-mode plumbing; the public validation is a later, checkpoint-dependent step. |
| R0217 | Decide whether the distinct global-dependency/timestamp modes and HIPENGINE_SUBMISSION_TRANSPORT rollback remain supported diagnostics or their comparison interfaces retire; timestamp/quarantine retirement additionally needs a disposition of the ROCm#6529 lifecycle investigation. |
| R0223 | Decide whether standard and planar Q6 package wrappers keep separate ordered row bands or are expressed through a shared ordered quant/shape/row policy representation; a single min-row field cannot carry the current precedence. |
| R0258 | Inventory the variant-specific test and profile callers of the H5Q/H5Z/H6D/H6F/H6I/H6P/H6Q/H6R grouped-IQ chain, then decide which diagnostics remain supported and either remove unused registrations/bodies/mappings/fixtures together or normalize the maps without changing selected function/interface identities. |
| R0259 | Decide whether the bounded backend role maps stay the supported policy representation or are normalized into a unified selector; the maps select cache ordering as well as a function, and absence/empty/global/SWA overrides carry different fallback semantics. |
| R0264 | Separate full-score diagnostics from long-attention execution policy before removing anything: the same prefill_long_attention_hipblaslt control and LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT constant also enable the bounded production route. |
| R0265/R0266/R0267 | Decide whether the explicit constructor/profile controls (prefill_global_attention_chunk_size, set_prefill_swa_attention_hipblaslt, set_prefill_dense_contiguous_cache, --attention-rows) remain supported geometry policy or move into scoped test owners; migrate the profile harness arms first if retiring them. |
| R0269 | Decide whether the constructor/profile MoE scheduling overrides (concurrency, after-router, low priority) remain supported diagnostics or move to scoped harness controls; migrate scripts/laguna_long_context_profile.py arms first while preserving sequential and priority-0 controls. |
| R0270 | Decide whether the older D8, D4x2/D4x3 and down modes stay supported diagnostics or their comparison callers migrate and redundant mode names are removed; R0281's grouped-down modes share the same set_selected_down_mode owner and belong in the same decision. |
| R0277 | Run the current same-host, capture-inclusive comparison first (scripts/gguf_decode_graph_g5.py --help, then its state-parity comparison with an explicit local model, backend and compiler-version file), then decide whether the eager override and comparison interfaces stay supported controls or retire. |
| R0281-ATTN | Decide whether global_prefill_variant, swa_prefill_variant and swa_decode_variant stay supported diagnostic controls or their comparison callers migrate and redundant public positive selection is retired; migrate the grouped-down category, SWA leaf, target-AR, DFlash and publication harnesses first. |
| R0281-Q5META | Decide whether use_q5_fixed_meta_output, use_q5_fixed_meta_query_gate and use_q5_shared_fixed_meta stay supported constructor/CLI diagnostics or migrate to a scoped diagnostic owner; migrate both target-AR construction sites and result fields together. |
| R0281-MIXED | Decide whether use_mixed_q5_q6_attention, use_mixed_q6_fixed_meta_attention and use_mixed_local32_fixed_meta_attention stay as a supported comparison diagnostic or migrate to a scoped projection-policy owner; migrate both target-AR construction sites and result fields together. |
| R0281-Q4HEAD | Decide whether use_q4_lm_head_local32_fixed_meta stays a supported constructor/CLI diagnostic or moves to a scoped projection owner; migrate both target-AR construction sites and result fields, then exercise actual Q4 full logits and a missing-key fallback. |
| R0281-IQ3 | Decide whether the wave10_fused, wave4_reduce and serial_weighted modes and iq3_c1_down_schedule stay supported constructor/CLI diagnostics or migrate to a scoped owner; migrate both target-AR construction sites and result fields. |
| R0281-SPLIT / R0339 | Decide whether the explicit split-attention crossover/disable and SWA tile16 controls stay as supported resource/profiling policy or migrate to a scoped KV policy owner; migrate both harnesses and their result fields with the session/allocator. |
| R0281-FUSION / R0338 | Decide per control whether each fusion/reducer comparison stays a supported diagnostic or migrates to scoped head/KV and attention-policy owners; migrate target-AR construction sites/result fields and long-context profile controls together. |
| R0281-IQ2 | Decide whether use_iq2_grid64 stays a supported constructor/CLI grid comparison or migrates to a scoped c1 policy owner; migrate both target-AR construction sites/result fields. |
| R0281-CONV | Decide whether HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE stays a supported diagnostic or its live isolation-profile callers are replaced with an explicit construction-time schedule before the environment reader is removed. |
| R0281-SELECTED-PAIR | Decide whether HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE, _SELECTED_DOWN_PAIRREUSE and _SELECTED_Q6_DOWN_PAIRREUSE stay diagnostics or both on/off requests are replaced by an explicit scoped policy before the readers are removed; migrate the batch-route gate's capture metadata with them. |
| R0281-AR-DECODE / R0308 | Decide whether the diagnostic callers (scripts/gguf_concurrency_baseline.py, server_f1_concurrency_bench.py, gguf_live_server_bench.py) migrate to an explicit resolved execution control or get a designated serial oracle before the process-wide HIPENGINE_GGUF_AR_PACKED_DECODE reader retires; update test fixtures and effective-route metadata in the same unit. |
| R0281-AR-PREFILL / R0309 | Give serial diagnostics an explicit prefill control and migrate the warmup, compatibility and resident consumers coherently before retiring the HIPENGINE_GGUF_AR_PACKED_PREFILL reader. |
| R0281-NATIVE-CYCLE / R0384 | Decide whether the four native diagnostic stages (target cycle, proposal cycle, proposal graph, complete cycle) remain supported interfaces or are replaced by one explicit stage selector; if replacing, preserve route names, effective options, result metadata and profiler comparisons. |
| R0281-PARO-NATIVE | Replace the two process-wide PARO native target selectors with profile-owned submission/commit variants (or retain them with explicit ownership), preserving provider-specific repair, result metadata and capture bindings. |
| R0281-TARGET-ORACLE | Designate Qwen35GGUFResidentSession.step_rows()/verify_rows() as supported diagnostic infrastructure and remove the retirement item, or migrate its full-logit and layer-boundary consumers before deleting it. |
| R0281-Q8-EXPERIMENTS | Keep the standalone raw-Q4/Q8_1 modes as explicit numerical/design controls or archive their performance interfaces while retaining parity references that still serve later work. |
| R0281-SAMPLER / R0508 | Migrate the host-placement and strict-algorithm sampler controls into a scoped sampler policy (or keep them as supported diagnostics), without broadening forced/dynamic request support. |
| R0281-MOE-GRAPH | Migrate HIPENGINE_GGUF_MOE_GRAPH to scoped diagnostic construction (or retire its session wiring and live comparison together), coordinating the tail-RMS migration and keeping standalone graph-cache consumers. |
| R0281-DENSE-Q8 | Migrate the loose dense-Q8 sidecar/placement controls into one scoped route policy (or keep the named compatibility comparisons), preserving load-time placement and pair/single/triple callers. |
| R0281-DRAFT-DENSE-Q8 / R0303 | Migrate the resident draft dense-Q8 stage selector to a constructor-owned stage policy (or keep explicit numerical comparison modes), updating bench/profiler and named suite callers together. |
| R0281-DRAFT-X8 / R0294–R0298 | Migrate the Q6 top-1 draft shape/dp4a/sidecar controls into one owner-local draft-head plan (or preserve explicit layout comparisons), keeping target materialization and the Generation-2 NextN provider separate. |
| R0371 | Replace the explicit Moonshine LM-head route selector with caller-owned capability selection plus an explicit numerical reference interface (or keep it as a supported selector), preserving the unfused projection/argmax and W8A16 route. |
| R0370 | Keep the explicitly requested continuous exact two-region reference or retire it after confirming no caller needs its hidden-boundary arithmetic and migrating its benchmark/GPU-test consumers. |
| R0367 | Migrate the selected gate/up halfdot rollback and exact-teacher selectors into an explicit kernel plan (or keep the constructor rollback and diagnostic selectors), preserving both registered implementations. |
| R0310 | Keep the named PARO rowchunk-boundary comparison (or replace it with another trace preserving the two-peer distinction); solving the c6 rowchunk cost is separate kernel work. |
| R0307 | Keep the compact cross-engine target score/hidden trace or replace it with a tensor trace before retiring the CLI and result fields. |
| R0306 | Keep the explicit host draft stage trace or replace it with a capture-safe graph tensor trace before retiring its CLI/result fields; no implemented graph trace currently substitutes. |
| R0305 | Retain the host F32 token-embedding precision oracle or replace it with a device-side FP32 embedding implementation before retiring the selector. |
| R0304 | Migrate the legacy selected-SiLU/down comparison into an owner-local legacy composite plan (or keep it as an explicit diagnostic) before removing its env/CLI readers. |
| R0302 | Preserve the explicit two-projection comparison or move its selection into an owner-local legacy draft plan, without removing generic MTP linear or the shared primitive. |
| R0301 | Retain the explicit singleton Q6 top-1 comparison or migrate the load-time X8 sidecar and runtime choice into one owner-local head policy; retire/rename multirow aliases only with their consumers. |
| R0281-SELECTED-X8 | Move primary selected-expert layout selection (repack, gate-up X8, gate-up raw precedence) into the loader's explicit plan with coordinated compatibility/bench/profiler callers, or preserve the existing diagnostic selector. |
| R0281-TOPK | Move the linear/tree and vocabulary-recovery proposal policies into a dedicated experiment harness (or keep them explicit), with coordinated suite and parity-workbench callers. |
| R0281-TARGET-POLICY / R0284 / R0299 / R0300 | Consolidate the named verifier comparison controls into an explicit benchmark verifier policy (or keep them), migrating bench parsing, suite/workbench aliases, schedule/commit choices and provenance together; treat the native API/registry rename as a separate naming decision. |
| R0281-CAPTURE | Move the six GGUF verifier capture controls into an explicit debug capture policy, separate from the production row-state mechanism (or retain them as named comparisons). |
| R0281-METADATA | Migrate the host/device contiguous-prefill metadata selector into an explicit prefill diagnostic policy while preserving an independent host oracle; decide automatic long-context selection separately. |
| R0281-Q8-WAVES | Move the 2-wave/4-wave dense T16 prefill selectors into an explicit prefill policy while preserving direct numerical oracles; decide any backend ceiling or kernel replacement separately. |
| R0281-ROUTER-TILES | Migrate the Laguna/Gemma router token-tile modes and setter/benchmark interfaces into a scoped diagnostic plan, preserving invalid-mode errors and independent 256/512-thread references. |
| R0281-PREFILL-DEBUG | Maintain the prefill router-select/isolation/progress controls as explicit diagnostic policies (or retire selected interfaces), deciding observation, execution selection and synchronization separately. |
| R0281-GDN / R0377 | Choose which GGUF GDN prefill comparison interfaces remain and migrate them to a scoped diagnostic plan, preserving the strict unfused fallback and named capability errors; R0377's cluster8 note is consolidated here. |
| R0281-SELECTED-PREFILL | Retain the explicit selected-prefill comparison/rollback interfaces or migrate them into one scoped diagnostic policy while preserving the registered strict fallback and distinct gfx1100/gfx1151 defaults. |
| R0281-DRAFT-CHAIN | Move device-chain selection into one caller-owned proposal plan with its observation/workspace requirements (or keep the explicit host/device comparison), without deleting the host path. |
| R0282-PREFILL-POLICY | Keep the resident WMMA prefill diagnostic selector or replace it with an explicit session-owned prefill plan, preserving the production WMMA default and working comparison routes. |
| R0282-DFLASH-DRAFTER | Preserve the dense and post-attention drafter comparisons or move their selection into caller-owned drafter plans, separating a shared runtime dense policy from a benchmark-only fusion experiment. |
| R0285 / R0286 | Retain the explicit synchronized draft/target profiling or replace it with one profiling policy across runners, benchmark cache/provenance, profiler parent/child and draftsync/allsync suite routes. |
| R0287 | Preserve the explicit per-boundary FP32 comparisons or move them into a runner-owned arithmetic plan with registered-profile bindings and probe serialization; do not enable every boundary to remove flags. |
| R0291 / R0293 | Preserve the Q8T16 pair/column/rowtile-all comparisons or migrate the ambient selectors and session globals into a caller-owned kernel plan across packed decode, profile binders, verifier profiling and the batch-route gate. |
| R0381 | Retain the short-C1 attention thread-width comparison or migrate it to an explicit caller-owned diagnostic plan before retiring the env reader/runtime-width entry point. |
| R0382 | Retain the long-context staged verifier comparison or migrate the selector and row-wise orchestration into one caller-owned strict/staged plan, keeping registered strict primitives and Conv/GDN/KV transaction ownership. |
| R0388 | Retain the dense-session MTP budget policy seam and CLI diagnostics as supported comparison interfaces, or replace them with a test/benchmark-owned cycle driver. |
| R0389 | Retain the standalone draft-hidden post-norm diagnostic constructor/CLI interface or migrate the transform into a benchmark-owned adapter; do not let an explicit post-norm request silently use pre-norm. |
| R0392 | Either adopt the documented child-mode procedure as the profiler interface or implement an explicit no-marker mode that records its scope and uses the matching rollup; do not silently replace a marker-scoped result with an unscoped trace. |
| R0393-Q6 | Keep HIPENGINE_C8_Q6_DP4A_GROUPED as a supported production rollback and standalone diagnostic, or replace it with an explicit session/profile interface; remove the env reader only after the replacement preserves those controls. |
| R0403 | Record the supported DMS/INT8 hidden-plane alias interface, or replace both rollback interfaces with an explicit workspace policy (or retire them together); the INT8 control is also a two-plane comparison arm. |
| R0407 | Keep the layer-scoped slot-local AOTriton rollback, or replace its native-comparison capability (evaluator manifests/capture and route A/B) before deleting the env reader and helper. |
| R0412 | Decide whether the H256 generic-wave bisection comparison remains useful; if retiring it, remove the wave scope value and its campaign/state-gate packages together while preserving page256 and strict_rows_spans. |
| R0422 | Choose the automatic context-sizing control interface and migrate all five readers and allocation probes together, preserving route pricing, unavailable-geometry fallback, explicit context handling, retry bounds and rollback semantics. |
| R0427 | Retain or retire HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY, which selects the older Qwen4Exp dense-attention body; if retiring, remove its reader and diagnostic overrides together and remove only the now-unused Qwen4Exp import. |
| R0428 | Retain the full-attention QK-unroll replay comparison or retire its replay arms, removing the experimental HIP export, Python wrapper, package export and --qk-variants handling together. |
| R0434 | Keep the pack8 GEMV wide/narrow comparison control or migrate scripts/qwen4exp_layer2_profile_gate.py and scripts/qwen4exp_de3_route_probe.py to an explicit launch policy before removing the env reader. |
| R0435 | Keep the full/bounded KV-reset comparison interface or migrate scripts/qwen4exp_rt4_kv_compare.py and the layer2 candidate to an explicit reset policy before removing the env reader. |
| R0438 | Keep the chunked WY comparison reproducible from the working tree, or retire its executable arm while preserving the historical artifact and Git source. |
| R0442 | Keep the DPP comparison control or migrate its state-gate and campaign callers to explicit variants. |
| R0443 | Retain the GR DPP comparison or migrate its distinct caller scopes to profile ownership. |
| R0459 | Decide whether the supported quant bindings still need the multi-column comparison before deleting row-split siblings. |
| R0475 | Retain eager AR diagnostics or migrate the checker and suite callers to a replacement device gate. |
| R0479 | Keep the distinct four-wave parity reference or migrate each producer's comparison contract before deletion. |
| R0506 | Decide rollback-switch retention without deleting working gapped-layout mechanisms. |
| R0518 | Choose production-geometry evaluation or removal of the unused wide-only MMQ export and wrapper. |
| R0129 | Assess rollback need and migrate prepack control without conflating it with arithmetic promotion. |
| R0141 | Decide retention of the explicit w32 occupancy comparison; it is not missing production functionality. |
| R0147 | Keep the attention-gate comparison or provide an operation-complete owner before removing its selector. |
| R0153 | Choose explicit owner migration or retained scoped raw/expert-grid diagnostics after caller inventory. |
| R0173 | Retain the R28 rollback or migrate its transport while preserving shared rowtile cache consumers. |
| R0208 | Choose retention or migration of the two production-load oracle modes after checking their callers. |

### Product / architecture / resource choice

| Entry | Next action |
| --- | --- |
| R0030 | Decide at campaign level whether strict router selection may be re-baselined so selection is invariant under a ~1 ulp logit change (including the reference); no kernel retry, threshold relaxation, or rerun of the rejected warp-reduce arm is valid before that call. |
| R0059 | Decide whether HIPENGINE_PREFIX_CACHE / --prefix-cache off stays a supported environment/CLI adapter or is replaced by an explicit cache policy; keep both surfaces working until that call is made. |
| R0058 | Choose redesign and Q3/Q3 capability work versus retirement of the dormant family; implementation follows that scope decision. |
| R0067 | Decide whether the dense prefix-retention default should allow a wider reuse working set at the cost of pinned memory, coordinating HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS and HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES; the eviction defect itself is already repaired. |
| R0143 | Decide whether one-shot low-RSS deployment stays a supported loading mode (retaining the explicit control or replacing it with a named loader policy) or is retired with the flag and its advice branch. |
| R0211/R0212 | Decide whether policy selection (protect_ttft / protect_decode / fair / token_budget) stays a supported production control or production standardizes on token_budget, and separately whether unconfigured generators inherit fair instead of protect_decode; keep the per-backend burst defaults until then. |
| R0239 | Decide whether HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING stays a supported host-memory/VRAM placement policy or is replaced by an explicit public placement option, migrating the four named 0.8B harness callers without changing their device control arms. |
| R0281-GGUF-REPACK | Decide whether HIPENGINE_GGUF_DECODE_REPACK and the API layout selection stay a supported resource/diagnostic policy or comparison callers migrate to an explicit load owner before the environment reader retires; a passed release window does not migrate them. |
| R0281-MTP-CAP | Decide whether the bounded four-slot verifier chunks stay the policy or a wider verifier is tuned with separate prefill/warmup policies; do not delete the shared constant as dead code or change prompt-prefill admission as a side effect. |
| R0281-LAUNCH-BOUNDS | Decide whether HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS stays a supported build control or migrates to shared build configuration with every plan/build caller updated; it is not a rejected runtime route to delete. |
| R0281-FAIR-BURST | Decide whether fair_prefill_burst_chunks / HIPENGINE_FAIR_PREFILL_BURST_CHUNKS stays an operator scheduling control or is replaced by a named workload-derived policy; keep the live CLI/env callers, tuning protocol, reconfiguration and telemetry unchanged until then. |
| R0281-INT8-POLICY | Decide the scoped INT8-KV policy that separates a recorded-failure override (HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG) from storage-layout selection (tail4_hadamard_group32), then migrate both generator and resident-runner readers together. |
| R0281-PARO-INT8-PREFILL | Move the operator INT8 prefill resource overrides into the resolved prefill/resource plan (or retain them as explicit overrides) and update result metadata, preserving invalid-value errors and oracle/scratch lifetimes. |
| R0393-Q5 | Choose the supported planar-Q5 sidecar interface and allocation lifetime: retain the optional persistent sidecar, replace it with a request-owned allocation policy, or retire the route. |
| R0402 | Keep the singleton target verifier as a supported specialization and document its ownership boundary, or converge on the packed target owner with packed graph capture and a matched C1 route comparison. |
| R0410 | Adjudicate the fused grouped-prefill combine candidate: retain the explicit interface, promote it after a current same-process comparison, or retire the complete route. |
| R0432 | Choose the dense staged-F32 production composition (or retain the evaluation interface / retire its runtime selector), preserving the empty/unset-off, layer-zero and dispatch-cache-key contracts. |
| R0446 | Choose owner-pinned library lifetime versus live build-environment observation; invalidation repair does not decide this contract. |
| R0462 | Choose shared gather scratch versus per-layer cached K/V storage; sparse-span gather is a separate implementable extension. |
| R0483 | Choose explicit specialized-runner libraries versus a documented backend runtime-operations boundary. |
| R0495 | Choose CPU vision memory/patch admission policy; no arbitrary ceiling or GPU timing substitution. |
| R0499 | Choose whether to pursue a new exact dual-GEMV schedule or explicitly close the research direction. |
| R0074 | Choose retained cost guard, measured replacement policy, or optimization-driven removal for slow routes. |
| R0163 | Decide whether to retire the FP16 experiment or pursue a bounded storage replacement with long-horizon gates. |
| RF-3 / R0187 | Choose supported source-command families using a per-path caller and replacement inventory; do not delete unnamed batches. |
| RF-5 / R0187 | Choose durable capture storage and compact decision schema before moving accepted evidence payloads. |
| R0204 | Choose materially different ngram composition work or retirement after the dedicated wider-frontier campaign. |

### Profiling / reproduction / candidate evaluation

| Entry | Next action |
| --- | --- |
| R0057 | Localize the wave-group6 producer's numerical failure first. The generic producer is already the shipping gfx1151 control, dense-injection gives zero KL, and the shared-memory race hypothesis is not established, so the next step is a targeted trace/ablation of the wave arithmetic before any kernel edit; only then implement the repair and restore selection. |
| R0001 | Re-measure the eager-vs-graph verifier comparison on the current stack with the committed reproduction block (three repetitions plus two confirmation repetitions), and separate capture, first replay and steady replay to locate the cost before any graph implementation change; only a measured cause justifies changing the graph path. |
| R0176 / R0394 | Reproduce the older singleton route's gfx1100 slot-zero failure and localize it (layout indices, bound Conv/GDN pointers, captured rows) before changing code or removing the guard; keep the working captured/deferred one-slot mapping enabled while that investigation runs. |
| R0220 | Reproduce and isolate the recorded alias failures first (which alias produces the final-logit change, on which geometry), then establish the dense gfx1151 lifetime plan; only alias stage-disjoint planes and remove allocation owners once the failure boundary is measured. |
| R0281-DFLASH | Run the current-stack DFlash category bench with the Laguna target checkpoint and its drafter (tile 1) to establish whether the true-AR mismatch reproduces; if it does, trace the first divergent verifier/prefill/commit state rather than attributing it to the output tile. |
| R0281-PARO-ARGRAPH | Trace the first graph/eager state or arithmetic difference with identical inputs (hidden, Conv/GDN and live KV at each transition) before changing the graph path; the mismatch is reproduced, but its cause is not localized, so any repair would be unjustified until that trace exists. |
| R0281-PARO-RAGGED | Measure first: run scripts/qwen35_batch_shrinking_correctness.py with the committed fixture to confirm the working per-segment fallback and resolved device/prefill metadata, then add a test-only packed ragged bypass and compare identical input trajectories to locate the first unequal hidden, Conv/GDN or live-KV state. Decide separately whether the FORCE_PER_SEGMENT controls stay supported or move to an owner-scoped interface. |
| R0373/R0374 | Profile and repair the Maple fusion efficiency gap (or remove the experimental runtime routes while preserving primitive references); treat the MoE and QK-attention fusions independently. |
| R0387 | Profile today's dense packed decode step on a gfx1100 host before choosing a kernel, state-transfer or submission change; this host is gfx1151 and cannot supply the requested measurement. |
| R0404 | Rerun the calibrated same-prefix direct-INT8 evaluator after the dtype repair, then complete the execution-profile category/heldout, shape, repeat/isolation and task gates including separate tail-four coverage. |
| R0415 | Profile today's register-state GDN decode path under OPTIMIZATION.md, then choose and validate a decode optimization without reopening the accepted prefill-first decision. |
| R0488 | Supply the Surya checkpoint and reproduce combined lanes before diagnosing divergence. |
| R0490 | Supply the Surya checkpoint, then trace tile shapes in the existing profiler-equipped ROCm environment. |
| R0491 | Supply the Surya checkpoint, then probe neighboring shapes and resource-check A4 dense comparison. |
| R0492 | Supply the Surya checkpoint and profile score work before implementing bounded causal keys. |
| R0512 | Re-screen Q5 decode-dual against current singles on gfx1151 before selecting a candidate. |
| R0515 | Capture the rollback reference and frozen-evaluator verdict with actual route attribution before switch retirement. |
| R0519 | Evaluate the compensated WMMA candidate against the current numerical contract before disposition. |
| R0521 | Measure capture/replay crossover on each affected host before engagement-policy tuning. |
| R0127 | Profile the working host grammar selection before choosing a device replacement; no measured bottleneck is established. |
| R0183 | Complete wide-Q6 shape-policy numerical, task and end-to-end evaluations before candidate disposition. |

### Validation of implemented work

| Entry | Next action |
| --- | --- |
| R0196 | Run hipengine serve at C5/C8 with the production D1/D24 workloads on gfx1100 and record physical grouping, determinism/isolation, allocation-failure cleanup and cancellation. The environment override itself was already removed (worklog 20261027T155317). |
| R0198 | Extend the existing public driver to repeated C1-C3 widths plus injected early completion and injected failure; the C1/C2/C3 width, output and cancellation cells already passed on gfx1151. |
| R0199 | Run defaults-only C1-C3 public serving on gfx1100 and record actual route engagement for row48 at rows33-48, row64 at rows49-64 and row128 at rows65-128. The retile environment/cache was already removed (worklog 20261027T160304). |
| R0200 | Run the defaults-only, same-protocol public repeat for the Q4-T16 down owner at (17408,5120), rows33-192 on gfx1100, recording selected launches, output/state parity and teardown. The row64 environment cap/cache was already removed (worklog 20261027T160757). |
| R0202 | Validate the shared _enqueue_batch_hidden_transition through public device accept/commit, eager/graph agreement and cancellation/cleanup; the duplicate launch bodies were already extracted (worklog 20261027T161809). |
| R0249/R0250 | Run the public gate with a Laguna checkpoint: python scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1100 --max-tokens 8 --output /tmp/laguna-public.json, with a prefill long enough to engage the ordered-F32 owner. Implementation is complete; only this evidence is missing. |
| R0273 | Run the gfx1151 public gate with the Laguna checkpoint and a prompt reaching the dense-initial BLAS path: scripts/laguna_public_correctness.py "$LAGUNA_MODEL" --backend hip_gfx1151 --max-tokens 8 --output /tmp/laguna-packed-public.json. |
| R0275 | Run the R0273 gfx1151 public command with a dense-initial BLAS prompt and additionally confirm the backend-selected wave-per-row softmax executes. |
| R0274 | Run the R0273 gfx1151 public gate with a dense-initial BLAS prompt and confirm direct packed-query production for the eligible tiles, generic production for other rows, output parity and clean teardown. |
| R0248 | Run the R0249/R0250 gfx1100 public generation gate with a prefill long enough to exercise H6Z/H6W, and confirm the removed H7Y route is neither selected nor allocates mirrors. |
| R0276 | Run the public Q4/Q6 route check with retired tile environment values deliberately set; confirm they are inert and backend/shape policy remains authoritative. |
| R0165 | Run hipengine serve with the dense GGUF MTP profile and an eligible physical target width on gfx1100, trace the wave-reduced state-row launch, and compare output/acceptance and teardown against the strict parent. |
| R0166 | Run hipengine serve with the physical dense-GGUF MTP profile on gfx1100, record the R6 two-wave rowtile and R36 row48 fused gate/up+SiLU launches, and compare outputs, acceptance and teardown against their registered parents. |
| R0168 | Run hipengine serve with the physical MTP profile on gfx1100, trace grouped-R8 at R24 for C5/C6 and grouped-R6 on the corresponding C7/C8 scope misses, and compare with the strict parent. |
| R0169 | Run hipengine serve with seven live requests and the physical MTP profile on gfx1100, confirm R28 target work rather than padded R30, and compare full state/output/acceptance and teardown against the padded composition. |
| R0170 | Run hipengine serve with the C8 physical MTP configuration on gfx1100, trace source-Q5 selection at R32 and raw-MMQ fallback at R24/shape misses, then compare output/acceptance and teardown with the retained parent. |
| R0171 | Run hipengine serve at eight active requests on gfx1100, confirm sidecar ownership and MMQ launches at admitted R24/R32 shapes, and include fewer-request and missing-workspace cases to verify fallback. |
| R0172 | Run hipengine serve with eight live MTP requests on gfx1100, trace grouped-Q5 R8 launches, and compare output/acceptance and teardown with the registered padded-prefix/strict-tail parent composition. |
| R0174 | Run hipengine serve with eight live MTP requests on gfx1100, confirm R32 target packets and the selected fused row32 owner, and include a non-C8 request count to confirm unchanged padding. |
| R0177 / R0292 | Exercise LLM.generate() and hipengine serve on gfx1100 and gfx1151 with defaults, tracing Q6 root row chunks at eight requests and at a non-divisible packed width. |
| R0178 | Run hipengine serve across the admitted mixed widths on gfx1100, trace grouped prefix/tail launches, and include a shape outside the mixed policy plus cancellation/cleanup. |
| R0179 | Run hipengine serve across grouped R6 and exact-prefix/tail widths on gfx1100, confirm selected launches, and include ordinary decode outside physical scope plus cancellation/cleanup. |
| R0180 | Run hipengine serve across admitted pair widths on gfx1100, trace both grouped output launches, and include a shape outside pair policy plus cancellation cleanup. |
| R0181 | Run hipengine serve at admitted grouped widths on gfx1100, confirm single-owner grouped launches, and include non-grouped rows, an out-of-policy shape and cancellation/cleanup. |
| R0182 | Run hipengine serve with packed state imports on gfx1100, trace the fused pointer-table gather where direct resident indexing is inapplicable, and compare imported Conv/GDN state with the per-layer parent. |
| R0281-D12 | Run LLM.generate() on gfx1100 and gfx1151 with the Laguna checkpoint and confirm the selected Q5 attention-output/query-gate variants, one-token and multi-row fallback, continuation and allocation cleanup. The constructor controls are already removed; do not revive them. |
| R0406 | Refresh the RX 7900 XTX single-request server context bound with a short multi-slab allocation probe, bracketing capacity and then confirming with the full-depth workload. |
| R0192 | Run public physical-C2 MoE NextN serving with rejection, cancellation, peer progress and teardown checks. |
| R0201 | Run the defaults-only same-protocol C1-C3 public repeat with category and lifecycle coverage. |

### Historical evidence binding

| Entry | Next action |
| --- | --- |
| R0060 | Identify the actual artifact used for the three historical dense-serving rows, verify access to it, then compute and attach its execution fingerprint. This is a provenance dependency, not a runtime admission condition. |

## Verification limits

Every heading has exactly one row; totals are computed from those rows. Referenced evidence paths were checked. Source inspection verifies live interfaces and known completed repairs, not new numerical parity. No full-model, GPU correctness, performance or capacity gate was rerun. The partial ordinal audit in `2026-10-31-R0096-R0190.json` is retained evidence; this merged audit supersedes its provisional categorization and stale R0460/environment statements.

Shard findings were reconciled against source. The Q4 projection methods exist, the router-ranking proposal is recorded in its rejection worklog, and R0276’s retired environment values are intentional negative controls.
