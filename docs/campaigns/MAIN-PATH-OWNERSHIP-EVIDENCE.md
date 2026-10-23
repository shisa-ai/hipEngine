---
status: current
owns: Main-path cleanup ownership map, finite prerequisite decisions, and behavioral validation evidence for the construction-lifetime refactor.
---
# Main-path ownership and safety evidence

This is the execution record for
[MAIN-PATH-ARCHITECTURE-CLEANUP.md](MAIN-PATH-ARCHITECTURE-CLEANUP.md).
Source baseline: `3d246c549`; concurrent documentation-only work advanced HEAD
to `7ada31812` without changing the Python implementation. Results below are
baseline evidence, not a claim that the campaign is complete.

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
tracked device allocations to zero. This is an open repair, not a passing
failure-recovery check. Speculative and HTTP checks remain to be completed.

## Ownership map

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

## Behavioral safety matrix

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

## Remaining acceptance evidence

The implementation, real speculative/HTTP checks, recovery repair, full suite,
documentation/audit/worklog gates and campaign completion audit are still open.
Record results here as units finish, then mark the campaign checklist from
concrete evidence rather than from the size of this investigation.
