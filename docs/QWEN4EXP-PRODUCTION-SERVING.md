# Flash-Next production-serving campaign

Status: **in progress; no production-serving completion claim**.
Approved objective: 2026-10-17. Starting source: `897ff13b4` on
`qwen3.8-flash-next`.

This campaign repairs and qualifies the serving gaps in the
[HTTP coverage grid](QWEN4EXP-HTTP-COVERAGE.md). The grid is the measured
starting point, not a list of capabilities already satisfied. The user has
approved implementation, necessary expensive validation, and fixing additional
bugs discovered along the way.

## Scope and evidence

The first physical qualification lane is zbook, machine
`87c566d30a5645cf8d12ed7ef6b6e1e8`, Ryzen AI Max+ Pro 395, Radeon 8060S/gfx1151,
128 GiB unified memory. The initial configuration is Unsloth Flash-Next
`UD-Q4_K_XL`, its Q8_0 MTP sidecar, production profile and BF16 KV. The
[baseline packet](../benchmarks/results/2026-10-17-zbook-qwen4exp-http-grid/README.md)
contains model paths, the server command, 50 HTTP requests and six reporting
rechecks. Framework rates do not serve as zbook denominators.

The implementation covers ordinary completions/chat, AR and explicit MTP,
streaming and blocking, request controls, concurrent users and long contexts.
Existing image/video support is separately tested with its required mmproj.
Unsupported hardware, artifacts or capacity must be reported or rejected
truthfully; evidence cannot transfer by architecture name. A safe rejection is
an interim containment measure, not completion of a requested missing feature.
Native 262K allocation is not native 262K inference qualification.

Architecture follows [PLAN](PLAN.md), [CONCURRENCY2](CONCURRENCY2.md),
[SPECDEC2](SPECDEC2.md), and [EXECUTION-PROFILES](EXECUTION-PROFILES.md).
Use the existing sole-driver service, request lifecycle, scheduler, collectors,
resource ledger and speculative transaction contracts. Do not create a second
model scheduler or make a frontend thread a GPU owner. Runtime remains
torch-free; dispatch remains plugin-based with registered strict fallbacks.

## Implementation and qualification checklist

Task numbers refer to the session task list; the stable identifiers below are
used in subsequent worklogs and evidence. Checked items require linked evidence,
not just a completed implementation. Split an item when a distinct defect is
found, and preserve unresolved requirements.

| ID | Task | Work and required acceptance evidence | State |
| --- | ---: | --- | --- |
| PS0 | 14 | Freeze scope, baseline, dependencies, acceptance matrix and safe execution protocol. | Plan established; implementation open |
| PS1 | 15 | Remove generation's dependence on administrative command timeout. Prove request-owned deadlines, cancellation, queued cancellation, driver failure, truthful readiness, shutdown and next-request recovery with slow fake drivers and long HTTP AR/MTP. | **Complete for the tested production configuration:** focused fake/real service failure and shutdown tests, 29,912-token AR/MTP, explicit deadlines, client disconnect, truthful readiness, and post-event recovery pass. |
| PS2 | 29 | Audit and repair omitted/auto quant and profile selection against artifact identity. Section 2.9 of EXECUTION-PROFILES documents an unqualified Q4_K_M default distinct from UD-Q4_K_XL; prove actual resolution, do not silently select that composition for a different artifact. Preserve explicit selections with honest qualification/admission. | Open |
| PS3 | 16 | Enforce hard thinking budgets and answer reservation; preserve deliberate hint semantics. Test delimiters, zero/small budgets, blocking/SSE, AR/MTP and token accounting. | Complete for text AR and named hard-policy MTP-to-AR fallback: [29 live cases](../benchmarks/results/2026-10-17-zbook-qwen4exp-thinking-http.json), 620 focused tests. Explicit hint still executes raw MTP; controlled speculation is separate PS6 work. |
| PS4 | 17 | Implement advertised sampling and logprobs, finite/probability-correct top-logprobs, seeded repeatability and per-request RNG ownership. Sampled speculation must preserve the intended distribution. Reject unsupported combinations before work while implementation proceeds. | Complete for tested C1 text routes: [26 live AR/fallback checks and four admission-only rejections](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampling-bytes-http.json), [12 whitespace/thinking rechecks](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampling-whitespace-http.json), and [98 sampled-MTP/AR request checks plus 60 comparisons](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampled-mtp-http.json) combining the original run with focused EOS/SSE repairs. All 18 category/heldout prompts pass actual MTP, repeats, isolation and buffered SSE. CPU distribution/transaction tests cover target draws and RNG rollback. Controlled MTP, incremental SSE, wider numerical/task qualification and automatic selection are separate gates. |
| PS5 | 18 | Resolve generic speculative compatibility from effective KV format. BF16 must not depend on irrelevant scale metadata; INT8 guards and fixed server KV policy stay intact. Both request spellings must execute the intended route with honest diagnostics. | **Complete for the tested BF16 configuration:** focused 617-test bundle passes; live explicit and generic spellings both execute request-owned MTP with matching counters/IDs under BF16 KV plus FP32 scale metadata. INT8, sampled, automatic, concurrency and streaming scopes remain open. |
| PS6 | 19 | Complete MTP EOS/ignore-EOS, multi-token stops, JSON Schema and tool processors. Prove correct transaction rollback, processor state, committed IDs and stop filtering; successful AR fallback is not MTP coverage. | Complete for C1 text controls: [170 actual-MTP request checks and 82 blocking/SSE comparisons](../benchmarks/results/2026-10-17-zbook-qwen4exp-controlled-mtp-http.json), combining completed audits and source-pinned focused repairs, not a fresh full final-source run. Covers both spellings, greedy/sampled controls, English/Japanese stops, alternate EOS, minimum tokens, schema, required/automatic tools, hard thinking and deliberate hint override. [Request-owned grammar/processor transactions](../worklog/entries/20261017T214256.026824Z-lhl-mtp-structured-transactions-0424f0.md) cover cancellation/rollback; exact RNG and semantic finish boundaries have focused CPU gates. Stop-string SSE is buffered for safety, including AR. Numerical/task quality and incremental streaming are not qualified here. |
| PS7 | 20 | Publish incremental committed MTP SSE tokens, valid Unicode/reasoning/tool deltas, terminal counters and usage. Verify first content before completion, disconnect/backpressure cancellation and subsequent recovery. | Complete for unstructured C1 text, including stops, logprobs, direct-provider output, disconnect and live TCP backpressure/recovery; see the [requirement-to-evidence audit](QWEN4EXP-HTTP-COVERAGE.md#ps7-streaming-qualification). Schema-validated and tool-enabled SSE retains complete-result buffering required by the API failure contract; those shapes are correct but not incrementally streamed. No permission to publish unvalidated arguments is implied. |
| PS8 | 21 | Implement actual multi-request batching and fair live admission, including C1/C2/C4, ragged/mixed AR/MTP, late arrivals, retire/refill and cancellation. Prove native request widths, isolation and bounded resource claims; candidate rows and serial loops are not concurrent model execution. | **Complete for the opt-in AR batch path at widths 1-4 and at most 2048 positions:** [width-4](../benchmarks/results/2026-10-18-zbook-qwen4exp-arbatch-w4-normative-task-review.json), [width-1](../benchmarks/results/2026-10-18-zbook-qwen4exp-arbatch-w1-normative.json) and [width-2](../benchmarks/results/2026-10-18-zbook-qwen4exp-arbatch-w2-normative.json) gates each pass 450 teacher-forced rows (mean KL 0.000655-0.000667, max 0.0161, top-1 447-448/450), three-repeat determinism, state/ownership and lifecycle. [72 live HTTP checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-arbatch-http.json) cover C1/C2 physical widths, mixed prompt lengths, C4 arrivals over C2 capacity with observed queueing and admission reuse, cancellation with a surviving peer, recovery, and a concurrent AR+MTP request whose MTP row stays on its own c1 runner; [67 more](../benchmarks/results/2026-10-18-zbook-qwen4exp-arbatch-c4-http.json) cover width-4 blocking, streaming, cancellation and recovery. Server telemetry supplies the width histogram, and dispatch-manifest binding proves the observed rows. Free-output task review found no semantic failure: the code and Japanese migration answers complete and pass, while the English memory-bandwidth and Japanese speculative answers cover every required point but exhaust the 1024-token review budget before EOS, so their completion is unverified rather than failed. Scope restrictions: the switch is default-off, `HIPENGINE_QWEN4_EXP_AR_BATCH_SLOTS` rejects a width above 4, the raised capacity ceiling applies only at `max_sequence_length <= 2048`, MTP rows never join the batch, and long-context batch behaviour is PS10 work. |
| PS9 | 22 | Make capabilities, readiness, route/fallback reasons, counters, verifier rows versus request widths, finish reasons and SSE usage match execution. Exercise positive and negative paths without wasted generation or contradictory metadata. | **Complete for the measured surfaces.** The capabilities manifest publishes the decode route and the opt-in AR batch envelope from the same probe the decode telemetry endpoint uses, so the two surfaces cannot disagree: `available`, `path`, `active`, the qualified width/position bound and an unusable `HIPENGINE_QWEN4_EXP_AR_BATCH_SLOTS` value are reported instead of inferred. Live evidence covers the switch off ([9 checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-decode-capability-off.json)), width 4 on with four physical rows per step ([9 checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-decode-capability-w4.json)), and an unqualified width refused at startup with readiness naming the refused setting instead of a KV remedy ([5 checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-decode-capability-invalid-slots.json)). A completed response also reports its own executed width under `choices[].hipengine.diagnostics.decode_batch`, verified live with four concurrent requests ([5 checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-per-request-decode-width.json)); `generation_shape` stays queue-scoped. Generated/draft/accepted/rejected counters are audited live against the per-cycle records, the rejected remainder, the acceptance rate, `usage.completion_tokens_details` and provider attribution, with non-speculative requests claiming nothing ([39 checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-speculative-counters.json)). Route/fallback reason and finish-reason agreement are now measured live: [20 checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-finish-route-audit.json) cover length, EOS, request stop, `ignore_eos` and sampled MTP, including that a server stop reports `matched_stop` instead of keeping the backend `eos_token_id` beside a rewritten reason, and that served text carries no chat-template special tokens. A sampled speculative request reports `used: true` with a named provider and counts that equal its reported cycles, so the route claim matches the work. An operator-level resident refusal returns a structured 503 `resident_prepare_failed` naming the refused setting and both remedies; readiness keeps its redacted form. Streaming usage with the documented `stream_options.include_usage` opt-in equals blocking usage and text for the same request (26/51/77 tokens), so the SSE accounting matches the buffered path. |
| PS10 | 23 | Qualify context capacity, retrieval/continuations and lifecycle at supported bounds, including approximately 30K AR/MTP, higher admitted contexts, prefix reuse/admission, capacity errors, repeat requests and long cancellation. Allocation probes precede full-prompt points. | **Capacity and lifecycle qualified on the served route:** [capacity ladder](../benchmarks/results/2026-10-18-zbook-qwen4exp-capacity-ladder.json) — the admission planner admits 4 resident slots at the native 262,144 context (116.89 GiB of 121.6 GiB available), while the engine enforces a narrower *qualified* width of 2 slots at that context and refuses 4 with a structured `resident_prepare_failed` 503 that names both remedies. Tier-2 retrieval over HTTP passes at 8,165 / 16,337 / 65,517 / 200,666 prompt tokens with 4/4 answers each at 10/40/70/95% depth in English and Japanese ([artifact](../benchmarks/results/2026-10-18-zbook-qwen4exp-http-long-context.json)). [Lifecycle](../benchmarks/results/2026-10-18-zbook-qwen4exp-http-lifecycle.json) passes 14 checks: 4 concurrent live slots observed then released to 0, oversubscribed requests answered, a disconnected stream releases its slot and the server keeps serving, and a stateful session reports `append_visible_only`, lists, and deletes. Prefix reuse stays `off` and is not claimed. AR-batch slots are qualified only at `max_sequence_length <= 2048`, so long-context batching and long-context cancellation under batch remain PS10-open. |
| PS11 | 24 | Verify image/video serving with mmproj, bounded contexts, sidecar errors and accurate capabilities. Fix observed regressions; keep text and multimodal qualification distinct. | **Qualified for one bounded image route.** [17 live checks](../benchmarks/results/2026-10-18-zbook-qwen4exp-http-multimodal.json) with the BF16 mmproj sidecar: a 64x64 solid red PNG is read (content `Red`, empty think block in `reasoning_content`), text keeps serving, and JPEG, remote URL, `stream`, `n>1`, tools, and an oversized payload are each refused with a named reason. A server started without `--vision-model` refuses an image part with `unsupported_content_type` instead of ignoring it. Fixed in this pass: multimodal answers now carry `finish_details`, apply request `stop` strings, separate reasoning from `content`, report `generated_token_ids`, and count prompt tokens from the rendered prompt; `/v1/hipengine/capabilities` publishes `features.multimodal` with the transport, formats, bounds, unsupported list, and reported fields. Scope: base64 PNG data URLs only, one user message, 8 MiB / 1024 px bounds, 1K request token limit, blocking only, no tools/continuation/session/grammar/logprobs. Video frames are accepted by the parser but were not requalified in this pass. |
| PS12 | 25 | Close full production numerical, finite-state, three-repeat, isolation, dynamic-width and semantic task gates. Resolve the 13/18 strict-exact task review without treating token differences alone as defects. | Open |
| PS13 | 26 | Measure same-host full-suite true-AR versus actual-MTP economics and serving latency. Qualify safe automatic selection by supported shape; retain improvements, never force a slower exact verifier merely for ID agreement. | Open |
| PS14 | 27 | Run the complete HTTP grid, mixed-request recovery soak, full pytest milestone gate and independent review. Repair every discovered defect and verify the final candidate. | Open |
| PS15 | 28 | Publish supported configuration, operating instructions, exact qualification evidence, remaining scope restrictions and final requirement-to-evidence audit. | Open |

PS1 and PS2 precede feature promotion. PS3–PS11 may be developed independently
where file ownership and contracts permit. PS12 gates PS13; both gate the final
PS14/PS15 audit. Every runtime unit needs its own narrow tests before commit;
the final gate does not defer correctness until the end.

## Acceptance rules

- HTTP status alone is insufficient. Keep exact generated IDs, actual route,
  proposal/accept/cycle counts, physical request widths, errors and stream event
  timestamps. Inspect evidence rather than converting every response into a pass.
- Controls and ownership are exact in every profile: request IDs, slots, token
  positions, KV/state destinations, cancellation, commit/rollback and sampler
  accounting. A terminal response must not leave an unowned operation running.
- Production arithmetic follows the calibrated numerical envelope, category
  floors, long-context round-off gate and task non-inferiority in
  EXECUTION-PROFILES. Do not weaken thresholds or replace them with generated-ID
  equality. Changing the sampling distribution is not numerical relaxation.
- Use the complete `benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite and
  category heldouts for speculative quality/economics. Code, English, Japanese
  and mixed-language failures cannot be hidden in an overall average.
- Automatic MTP remains off for an unqualified or regressive scope. Functional
  explicit MTP and profitable automatic MTP are separate verdicts. Disabling
  MTP or changing its label cannot close missing explicit-MTP behavior.
- Preserve same-host baseline/candidate shape, profile, artifact, KV and command
  identity. Record latency, throughput, memory, actual occupancy, failures and
  task gates; choose the serving SLO and offered load before the final soak,
  not after seeing its results.
- Configuration/startup errors, overload, queue limits and slow/disconnected
  clients must fail predictably without corrupting another request. Audit
  prompt-limit enforcement, tool/schema errors, resource cleanup and GPU health.
- A missing model, device, oracle or evidence is an open qualification item,
  not a skipped success. Final completion is scoped to demonstrably supported
  configurations and must not claim universal correctness.

## Execution protocol

1. Reproduce each bug with a narrow RED test before changing behavior. Use fake
   drivers for lifecycle/control boundaries; use numerical fixtures for math.
2. Implement through the existing architecture, run targeted and affected tests,
   and record results in a new immutable worklog entry. Commit each validated
   logical unit immediately, with explicit staging. Leave unrelated files alone.
3. Run GPU jobs serially on this machine; check for other owners first. Announce
   why a run exceeding five minutes is needed and its expected duration, then
   use a background task. Run test servers only on loopback/private audit ports.
   Stop owned servers after testing; do not reset the GPU or kill other owners.
4. Use `scripts/gguf_capacity_probe.py` to bracket capacity. Run full-prompt
   points only to confirm a bound or qualify inference at depth. Prebuild kernels
   outside rocprof and require cached builds for profiling.
5. Expand `scripts/qwen4exp_http_grid.py` with assertions and independent controls
   as features land. Keep failure/error payloads and malformed-response evidence.
   Add streaming, cancellation and concurrency tests rather than only happy paths.
6. Run one full `uv run pytest --suite all -v` at milestone closure. After isolated
   failures, preserve that result and rerun repaired nodes/files plus affected
   bundles; repeat the full run only when the documented escalation rules apply.
7. Record performance qualifications in compact artifacts, `benchmarks/README.md`
   and `benchmarks/CHANGELOG.md`. Track temporary flags and fallback cleanup in
   `docs/REFACTOR.md`; do not put an implementation diary in the public README.
8. Before declaring completion, inspect every acceptance item and its final-source
   evidence. Passing tests, a large commit count, or substantial effort do not
   establish a production-ready server. Open defects keep the campaign open.
