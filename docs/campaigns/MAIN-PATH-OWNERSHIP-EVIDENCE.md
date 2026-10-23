---
status: closed
owns: Completed main-path cleanup ownership map, prerequisite decisions, and validation evidence for construction, admission refusal, and MTP upload lifetime.
---
# Main-path ownership and safety evidence

This is the execution record for
[MAIN-PATH-ARCHITECTURE-CLEANUP.md](MAIN-PATH-ARCHITECTURE-CLEANUP.md).
Source baseline: `3d246c549`; concurrent documentation-only work advanced HEAD
to `7ada31812` without changing the Python implementation. The selected campaign
closed on October 23 after construction ownership, request refusal containment,
and MTP upload lifetime repairs. The initial findings below describe the baseline;
the final sections record implementation and acceptance evidence.

## Selected path

The real-model baseline uses `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`,
`hip_gfx1151`, `gguf_q4_k_m`, the default execution profile, BF16 KV,
`max_active_requests=1`, `max_sequence_length=1024`, prefix caching off, and
four greedy output tokens. The artifact is available on the current machine;
it is an example configuration, not an allowlist. Initial CPU checks use
`uv run pytest`; GPU execution uses the single TheRock installation in
`~/miniforge3/envs/therock`, not a second ROCm runtime in the same process.

`LLM` is lazy. Its first generation, preparation or capability query loads
metadata and constructs the selected generator. For this configuration the
public baseline observed:

```text
LLM.generate()
  -> EngineService
  -> SubmitPollTextGenerator
  -> Qwen35GGUFResidentModelRunner
  -> Qwen35GGUFBringupGenerator shared runner/session
```

After the first ordinary request, pool counters reported
`native_full_prefill_rows=1`, `native_c1_decode_steps=3` and no fallback reasons.
Closing a partially consumed stream and generating again preserved the output,
left no resident rows, and reported healthy service status. These are ownership
checks, not throughput or numerical-quality measurements.

The oversized-request recovery check exposed a separate failure: a request with
1,025 exact prompt IDs and four output tokens needs 1,028 KV positions. Its
`reserve_admission` refusal raised `ValueError` and killed the entire service;
a subsequent valid request raised `EngineServiceClosed`. Final close returned
tracked device allocations to zero. This baseline failure was repaired by
`a8cea4c6f`; final speculative and HTTP results are recorded below.

## Baseline ownership map

Anchors are symbols rather than line numbers so ordinary edits do not invalidate
the table. Paths below are under `hipengine/`.

| Decision or transition | Current owner and callers | State/resource and lifetime | Failure/cancellation behavior; proposed change |
| --- | --- | --- | --- |
| Metadata, backend, quant and execution profile | `llm.py:LLM._get_text_generator`, called by public generation/preparation/capability methods | Resolves registry factory and optional profile before constructing weights | Preserve registry selection, defaults and errors; no backend branches added |
| Factory arguments and serving declarations | `llm.py:_factory_capacity_kwargs`, `_declare_serving_capacity`, `_declare_serving_context` | Factory parameters are opt-in; Qwen GGUF receives width/context via post-construction hooks; Gemma4/Qwen4Exp/Surya also use constructor parameters | Configuration refusal becomes `ResidentPreparationError`; raw generator currently leaks. Add construction ownership, preserve order and exception type |
| DMS, candidate budget, speculative provider | `LLM._get_text_generator`, `_publish_candidate_budget`, provider registry, generator `attach_speculative_provider` | Sidecar configuration precedes context declaration; compact DMS and candidate budget follow it; target owns provider after successful attachment | Missing attach closes only provider today. New temporary owner closes provider on attachment failure and generator on any later setup failure |
| Logical resident capacity | `LLM._get_text_generator`, generator `automatic_serving_capacity`, driver constructor | Explicit width wins; otherwise native generator sizes residency. Non-native compatibility generators clamp to registered physical width | Preserve arithmetic and precedence. This unit does not unify logical residency with HTTP batching width |
| HTTP queue and group width | `server/api.py:_GenerationBatcher._route_request_cap`, `_group_has_capacity`, submit/stream tasks | Transport queues, cancellation tokens, per-route coalescing and active accounting | These are not the scheduler's device slots. Preserve grouping/independent-request distinction rather than removing one queue |
| Driver and resident runner preparation | `generation/engine_loop.py:SubmitPollTextGenerator.__init__`; Qwen generator `create_resident_model_runner` | Driver acquires runner, builds loop, configures runner and prepares pools before service startup | Runner preparation can fail before driver locks exist. Add a constructor abort path for the acquired runner; do not call normal close on an incomplete object |
| Service startup and publication | `generation/engine_service.py:EngineService.__init__`; `LLM._get_text_generator` | Service owns scheduler-driving thread; `LLM._text_generator` is published only on success | Capability probing may fail before service locks exist. Keep the fully constructed driver owned until service handoff succeeds; do not publish partial objects |
| Request IDs, waiting and device admission | `SubmitPollTextGenerator.submit_detailed`; `ResidentEngineLoop.submit`; `batch_scheduler.py:BatchScheduler.submit/admit_pending` | Logical pending rows become active only after reservation succeeds | Predictable oversized-input refusal currently escapes a poll and poisons service. Repair separately without containing fatal device errors as ordinary request errors |
| KV reservations and row/session leases | Qwen resident runner `reserve_admission`, `discard`, `_acquire_lease`; runtime `Qwen35GGUFResidentSession` | Row claims device pages and a prepared session; commit/cancel/discard release ownership | Preserve atomic reserve, per-row discard and pool lease accounting; no kernel/KV ABI changes |
| Speculative commit and rollback | Qwen generator `_commit_mtp_serving_cycle`; runtime snapshot/commit helpers; service speculative children | Draft/verification state is provisional until accepted; published tail belongs to request | Existing CPU tests protect partial-row selection and cancellation. Real public speculative route must be observed before closure |
| Completion, cancellation and teardown | Service per-child collectors; driver `abort_submission/close`; Qwen runner and generator `close` | Service serializes driver operations, driver closes runner then generator; generator frees pooled children before shared weights | Preserve child isolation and close order. Family close may transitively close generator, so distinguish idempotent resource release from raw method-call counts |

## First implementation decision

The selected improvement is **construction-lifetime ownership**, not a broad
protocol extraction. Four probes reproduced unpublished-resource leaks:
configuration-hook failure, runner preparation failure, service capability-probe
failure, and a missing provider-attachment hook. Retrying constructs new objects
without releasing the previous ones. `LLM.close()` cannot reach any of them.

Make the temporary owner explicit and transfer it at successful generator,
driver and service boundaries. Remove the one-off provider-only unwind. Keep
optional capabilities structural; no giant protocol, flags or new registry axis
is needed. Constructors must unwind only resources they acquired; an outer owner
must never call normal `close()` on a partially initialized driver/service.
A cleanup error must not replace the original setup failure.

Error compatibility is deliberate: factory/capacity/sidecar/context failures
remain wrapped in `ResidentPreparationError`; later setup failures preserve their
existing types. Candidate-budget publication stays on `LLM`, in its existing
post-generator/pre-driver position. Its three-source selection ladder is not
part of this lifecycle change.

## Baseline behavioral safety matrix

All paths below are under `tests/`. Initial results were independently run on
the unchanged implementation. Mocked tests prove host contracts, not GPU math.

| Contract | Runnable target(s) and oracle | Initial result and limitation |
| --- | --- | --- |
| Registry construction/defaults/provider attach | `test_unit_llm_generate.py`, `test_unit_llm_generate_mtp_flag.py`, `test_unit_llm_mtp_eligibility.py`, `test_unit_qwen35_gguf_serving_width_sidecar.py`, `test_unit_dms_serving.py`; selected factory, arguments, request shape, precedence | 69 passed; synthetic registry generators and bare-instance hooks |
| Logical capacity versus physical group width | `test_unit_qwen35_gguf_capacity_route_terms.py`, `test_integration_generation_engine_service.py`; exact capacity and sole-driver assertions | 77 passed; synthetic memory and fake drivers |
| Allocation arithmetic | `test_unit_qwen35_gguf_capacity.py`; allocator-derived byte counts | 52 passed; CPU only |
| Isolation, row ownership and partial speculative commit | `test_live_generation_qwen35_gguf_sampling.py`; terminal-slot discard, atomic admission, captured-row commit, cancellation release | 126 passed, 1 skipped; mostly bare instances/fake sessions despite live-tier name |
| Close order, rollback and speculative finish | `test_unit_engine_loop_driver_close.py`, `test_unit_qwen35_gguf_linear_state_snapshot.py`, `test_unit_qwen35_gguf_linear_state_commit.py`, `test_integration_specdec2_engine_service.py`, `test_integration_engine_service_speculative_ownership.py`, `test_integration_engine_service_speculative_finish.py` | 49 passed; real service with fake execution, patched kernel operations |
| HTTP admission/cancellation/error surface | `test_integration_server_api.py`: route-group limits, live-neighbor survival, default coalescing, cooperative stream close, SSE cancellation and structured preparation refusal | Eight selected nodes passed; FakeLLM, not real HTTP-to-model execution |
| Cross-family close/setup contracts | `test_unit_gemma4_decode_graph_llm_owner.py`, `test_unit_gemma4_decode_attribution.py`, `test_unit_generation_qwen4_exp_gguf.py`, `test_unit_yue2_song_generator.py`, `test_unit_vibevoice_protocol.py`, `test_unit_gguf_packed_workspace_stability.py`, `test_unit_qwen38_production_ar_gate.py`, `test_unit_server_error_retryability.py` | 168 passed; CPU-only family/harness ownership |
| Gemma context forwarding | `test_unit_gemma4_generate.py` | Four passed, three HIP-dependent skips; real public GPU branch not covered by this run |
| Failed construction and clean retry | New `test_unit_generation_construction_lifetime.py`; each acquired owner released, unpublished cache, original exception preserved | Required RED coverage in progress |
| Context declaration | New public construction guard; exact context delivered before driver preparation | Existing 154-test selection did not catch dropping the hook; new guard required |
| Real public route/cancellation/recovery | Qwen configuration above via `LLM.generate` and `stream_detailed` | Ordinary and cancellation/refill passed; oversized-request recovery failed, repair required |

Commands use `uv run pytest <targets> -q`. Final validation must include the
remaining queued/pre-admission cancellation, disconnect/error frame, stop/EOS,
request isolation and shutdown nodes in the integration files, not just the
initial eight server nodes. The campaign milestone command is
`uv run pytest --suite all -v`, with the repository focused-repair rule.

Fault-injection evidence from the independent review:

- Replacing `_declare_serving_capacity` with a no-op fails the three pinned-width
  guards (`None`, 2, 8) in `test_unit_qwen35_gguf_serving_width_sidecar.py`.
- Ignoring the context map in `_server_plain_ar_capacity` fails
  `test_llm_selects_registered_short_context_plain_ar_width` (4 instead of 13).
- Replacing `_declare_serving_context` with a no-op leaves the existing selected
  tests green but fails the prototype explicit context guard. Promote that guard
  into the construction regression file.

## Finite prerequisite and ledger decisions

No unrelated code deletion is required to implement this ownership change.
The finite prerequisites are the construction-failure RED tests, the context
handoff guard, and the explicit ownership/error-boundary decisions above.
Constructor abort handling is the implementation itself, not an additional
cleanup campaign.

| Existing ledger subject | Disposition and clearing action |
| --- | --- |
| Speculative candidate-budget default versus depth | Preserve current `LLM._publish_candidate_budget` placement and precedence; ladder cleanup is a separate follow-up, not a prerequisite |
| Direct `Qwen35GGUFResidentSession` construction is not shipping AR | Constraint: public tests use `LLM`, preserving generator-selected GEMV route; never bypass generator to simplify the test |
| Native C1 context admission | Separate responsibility; preserve admission contracts. Oversized-input service failure is a concrete adjacent repair, not grounds to redesign all admission |
| Generic speculative service lifecycle | Preserve provider admission; only construction cleanup changes in first unit |
| Serving-capacity workspace leases / bounded packed reservation | Already resolved entries; no new work |
| Flash-Next indexed admission, HTTP text-stop safety, prefill chunk guard | Unrelated to construction ownership; leave in existing queue |
| Remaining overwritten definitions | Independent audit cleanup, no demonstrated dependency on this unit |

The initial audit refresh archived three absent `ungoverned-flag-branch` rows
without claiming their debt resolved. Existing engine-loop environment flags and
their precedence are unchanged; no new flag or debt-budget increase is proposed.

## Implemented ownership and removed paths

| Boundary | Final owner and implementation | Removed ambiguity; unchanged behavior |
| --- | --- | --- |
| Unpublished generator/provider | `generation/construction.py:ConstructionOwner`, used by `LLM._get_text_generator` | One temporary owner replaces the provider-only unwind. A successful attachment transfers provider responsibility to the target; failure closes all acquired resources. Registry selection, hook order, candidate budget and public exception types stay unchanged. |
| Runner preparation | `SubmitPollTextGenerator.__init__` temporarily owns the returned runner | Failed prepare/configure closes the runner without calling normal close on a partially initialized driver. Successful driver construction transfers ownership outward. |
| Service startup/publication | `EngineService.__init__` and the outer construction owner | Startup timeout joins the unpublished service; idempotent driver close prevents duplicate release. `LLM._text_generator` is published only on success. |
| Impossible request context | `Qwen35GGUFResidentModelRunner.register_batch` | Validate the whole cohort against prepared session capacity before publishing rows or advancing batch IDs. Available and busy leases provide the same bound; DMS effective output limits and zero-output behavior are preserved. Unexpected execution/device faults are not swallowed. |
| Resident MTP weights/embedding uploads | `_GGUFMTPServingAssets.upload_cache`; standalone drafts own a private `WeightUploadCache` | Resident runners no longer use the kernel wrapper's process-global name cache. Models have independent caches; allocating runtime identity separates entries inside one owner. Pooled drafts borrow uploads, and the generator closes uploads after drafts but before shared target weights. |

Construction code and tests landed in `0e56e4a9d` together with a concurrent
worker's PARO unit because the shared index was committed by that worker. The
exact construction paths were verified afterward; history was preserved rather
than rewriting the other worker's commit. `8c6ba95bf` normalized the regression
file ending. Admission is `a8cea4c6f`; upload lifetime is `3fea5d9f9`.

No kernel body, arithmetic, precision, selected variant, default or registry axis
changed in these ownership units. Numerical promotion and performance gates are
therefore not applicable. Short greedy-output agreement below is a lifecycle
regression check, not a quality or batch-invariance certification.

## Final behavioral acceptance matrix

Run CPU targets with `uv run pytest <file-or-node> -q`. All named CPU nodes below
passed in the milestone run unless a focused result is explicitly named. The
real-model probes used the selected configuration above with the TheRock Python
runtime. Their scripts and logs are local evidence under
`/tmp/hipengine-main-path-campaign/`, not repository dependencies.

| Contract | Exact target or public operation | Oracle and result | Dependency and coverage limit |
| --- | --- | --- | --- |
| Construction, failed setup and retry | `tests/test_unit_generation_construction_lifetime.py` | 14 cases pass: cleanup, clean retry, unpublished cache, error identity, provider attachment, width/context forwarding and startup timeout. Original RED run had ten cleanup failures and three controls passing. | Real public construction with synthetic registry generators; no GPU math. |
| Capacity boundary and atomic refusal | `tests/test_unit_gguf_admission_context.py` | Prompt lengths 17, 1020, 1021, 1022 and 1400 against 1024 positions and four output tokens; mixed cohort rejection preserves a busy neighbor. DMS limits and zero output covered. Admission bundle: 77 passed. | Fake session capacity; real refusal/recovery also exercised below. |
| Isolation and slot reuse | `tests/test_integration_generation_engine_service.py::test_engine_service_is_sole_driver_and_refills_before_long_neighbor_finishes`, `::test_engine_service_slow_consumer_and_cancel_are_child_scoped`; `tests/test_live_generation_qwen35_gguf_sampling.py::test_gguf_resident_runner_discards_only_terminal_packed_state` | Independent children preserve output and reclaim only their own state; all pass. | Real service with fake driver, and bare resident owner with fake sessions. No multi-request GPU numerical-invariance claim. |
| Queued/pre-admission cancellation | `tests/test_integration_server_api.py::test_generation_batcher_skips_cancelled_queued_submit`, `::test_generation_batcher_releases_prestart_cancelled_controlled_reservation`, `::test_generation_batcher_drains_pending_controlled_cancellation_on_exit` | Queue/reservation accounting passes, including cancellation before backend work starts. | CPU transport tests with FakeLLM. |
| Running cancellation and recovery | `tests/test_integration_generation_engine_service.py::test_engine_service_timeout_is_child_scoped_and_reclaimed`; real `LLM.stream_detailed` partial close then `generate` | Child-scoped timeout passes; real refill matches ordinary output and service remains healthy. | One real GPU request at a time; neighbor isolation covered by CPU tests. |
| Speculative provisional state | `tests/test_unit_qwen35_gguf_linear_state_commit.py`, `tests/test_unit_qwen35_gguf_linear_state_snapshot.py`; `tests/test_live_generation_qwen35_gguf_sampling.py::test_gguf_speculative_mtp_partial_commit_uses_captured_row`, `::test_gguf_speculative_mtp_cancellation_releases_request_buffers_and_poisoned_draft` | Correct captured-row commit, pointer/journal rollback and cancelled draft release pass. | Kernel calls are patched in state ownership tests; arithmetic unchanged. |
| Stop, finish, usage and terminal error | `tests/test_integration_generation_engine_service.py::test_engine_service_stop_holdback_is_owned_per_child_collector`; `tests/test_integration_engine_service_speculative_finish.py`; `tests/test_integration_server_controlled_speculation.py::test_text_stop_not_matching_standalone_tokenization_never_leaks`; `tests/test_integration_server_api.py::test_streaming_completion_backend_cancelled_exception_emits_error_and_done` | Stop text does not leak, finish precedence and accounting pass, terminal SSE error is followed by done. | CPU service/HTTP fixtures; real short generations ended by length, not a forced GPU EOS case. |
| Shutdown and upload lifetime | `tests/test_unit_mtp_weight_upload_ownership.py`; `tests/test_live_generation_qwen35_gguf_sampling.py::test_gguf_generator_close_releases_pooled_children_before_shared_weights` | 18 owner tests pass; real constructor metadata/upload paths use fake JIT and allocation boundaries. Independent caches/runtimes, failed copy/free, constructor unwind and draft-before-upload close verified. | No simulated GPU free failure is claimed to recover a poisoned device; tests check host ownership and error preservation. |
| Affected non-Qwen consumers | Cross-family files in the baseline matrix; `tests/test_unit_engine_loop_driver_close.py` | Shared construction/close contracts pass for Gemma, Qwen4Exp, Surya-related infrastructure, YuE2 and VibeVoice fixtures. | CPU coverage, not new model certification. Existing GPU prerequisites skip where absent. |
| Actual HTTP text/chat and disconnect | `http-public.py`: real `create_app(ServerConfig(...))`, Uvicorn loopback; `/v1/completions`, `/v1/chat/completions`, close SSE after first data frame, then refill | HTTP success, oversized request returns 400, later text equals baseline, service healthy, final tracked allocations zero. | Real model and HTTP stack, not FakeLLM; server launched programmatically with the same app used by the CLI. |
| Actual speculative route and close | `public-probe.py`: `LLM.generate_speculative_mtp_detailed(...)`; explicit HTTP `speculative_mtp: true` | Both report two MTP cycles; HTTP selected `gguf_specdec2_mtp2`, no fallback, IDs `[271, 248068, 198, 8160]` match ordinary generation. Global diagnostic cache stays empty in LLM probe. Both close with zero tracked allocations/bytes. | Four greedy output tokens on the selected artifact/configuration; no general accuracy or throughput claim. |

Real admission probe `admission-public.py` separately rejected prompts of 1025
and 1400 repeated tokens, preserving service health and ordinary output after
each refusal. Its close also returned tracked allocations to zero. This repairs
the baseline service poisoning rather than reclassifying it as expected behavior.

The final context fault probe replaced `_declare_serving_context` with a no-op
without editing source. The committed
`test_llm_forwards_its_serving_context_and_width_to_the_generator` failed at the
missing declaration assertion. Together with the capacity and context-map fault
results above, this establishes that the relevant guards detect wrong behavior.

### Milestone result and focused repair

`uv run pytest --suite all -v` first stopped during collection because
`test_integration_test_discovery.py` imported moved hooks from `tests/conftest.py`.
`e7f5da918` reads the repository-root hooks instead; 21 discovery checks passed.
No tests executed in that first milestone attempt, so the full command was rerun.

The completed run reported **19,516 passed, 6,489 skipped and one failure** in
254.76 seconds. The sole failure was the live sampling file's fake serving-assets
object lacking the new upload cache. Its fixture now includes an upload owner and
asserts that it closes after draft runners and before shared target weights.
The complete file plus five affected files then passed **249 tests with one
skip**. The owner file subsequently passed **18 tests**, including two added
constructor-failure cases. No production code changed after the completed broad
run; this uses the repository's focused-repair rule, not an all-green rerun claim.

The skips include unavailable HIP/model/backend prerequisites in the uv Python
environment. They are not passing GPU tests. The selected Qwen ownership path was
separately exercised with real GPU execution through LLM and HTTP, so missing
unrelated models do not block this bounded campaign. No claim is made that every
backend or all 6,489 skipped cases were validated.

## Result and next bounded unit

The safety work found three concrete problems and protected their repairs:
unpublished construction resources leaked, predictable oversized-input rejection
poisoned the service, and resident MTP uploads outlived their model owner. Each
now has explicit ownership, regression coverage and public-path evidence. The
old provider-only unwind and resident use of the global upload cache are removed;
there is no new fallback route or flag.

The next candidate is the speculative candidate-budget selection ladder on
`LLM`, already recorded in the scoped ledger table. First compare its three
sources and define one owner without changing precedence. Transport backpressure
and device scheduling remain distinct responsibilities; this campaign found no
basis to collapse them. Broad protocol extraction, family unification, kernel
selection cleanup and repository-wide debt remain separate work, not unfinished
items in this bounded campaign.
