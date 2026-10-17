# Qwen3.8 Flash-Next HTTP coverage

**Updated: 2026-10-17. The grid is not complete.** Short greedy AR and
explicit MTP work. The focused lifetime follow-up also passes approximately
30K AR/MTP, cancellation and recovery. Hard thinking controls pass on text AR
and named MTP-to-AR fallback. Text AR sampling/logprobs pass focused live
checks, including combined hard thinking/logprobs. C1 sampled MTP and its
buffered-SSE logprobs also pass the full category/heldout functional audit.
C1 controlled MTP passes 170 functional request checks and 82 blocking/SSE
comparisons: stops, alternate EOS, minimum tokens, ignore-EOS, JSON Schema,
required/automatic tools and hard-thinking controls. C1 unstructured MTP SSE,
including logprobs and text stops, now publishes committed output incrementally.
Schema-validated and tool-enabled SSE intentionally buffers output until complete
validation; it is not token-live. True multi-request batching remains open.
A passing unit suite does not establish those product capabilities.

Implementation and final qualification are tracked by the approved
[production-serving campaign](QWEN4EXP-PRODUCTION-SERVING.md). The observations
below remain the October 17 starting audit; implementation progress does not
retroactively turn its failing requests into passes.

This audit used branch `qwen3.8-flash-next` at `e7eca661c` on physical host
`zbook` (`87c566d30a5645cf8d12ed7ef6b6e1e8`), Ryzen AI Max+ Pro 395,
Radeon 8060S / gfx1151, 128 GiB unified memory. The model was the four-shard
Unsloth Flash-Next `UD-Q4_K_XL` with the Q8_0 MTP sidecar, production profile,
BF16 KV, 32,768-token capacity, two active slots, candidate budget 3, and the
server's default zero batch-coalescing window. This is not the Framework host
used for most October optimization results.

The [evidence packet](../benchmarks/results/2026-10-17-zbook-qwen4exp-http-grid/grid.json)
contains 50 baseline HTTP requests, six focused checks after a reporting repair,
exact returned token IDs, route decisions, backend group widths, errors, stream
event timing, source and physical-host identity, and reproduction commands.
The ten-prompt suite includes all four categories and their heldouts. Requests
allow up to 128 output tokens and respect EOS: eight emitted 128, two emitted
58 and 31. These are single-pass functional observations, **not a new speedup,
quality, or automatic-admission qualification**.

## PS1 lifetime follow-up

The baseline long-MTP failure and immediate recovery failure are superseded by
an isolated lifecycle qualification at commit
`f1ee044e5136bddd6ba477c6efacb6356fc7b080`. On the same physical zbook
host, a 29,912-token archive prompt completed in both AR and explicit MTP. AR
returned `VIOLET-7391` in 30.724 s; MTP returned the same answer in 91.979 s
through `qwen4exp_mtp_batched_state_rows_verify` with 6 proposed and 6 accepted
draft tokens across 2 cycles. The MTP wall time is recorded for lifecycle
evidence only; this is not a performance claim.

The request-owned lifetime checks also returned HTTP 408 `deadline_exceeded`
for both AR and MTP with `timeout_ms=5000`, logged an aborted MTP request as
HTTP 499 `cancelled`, and completed six post-event recovery requests with HTTP
200. Final readiness reported a live accepting service, zero active resident
requests, 17 admissions, and 17 reclaims. The compact evidence is
[`2026-10-17-zbook-qwen4exp-http-lifetime.json`](../benchmarks/results/2026-10-17-zbook-qwen4exp-http-lifetime.json);
raw local captures are under `/tmp/he-qwen38fn-lifetime-f1ee044e5`.
This closes PS1/task 15 for the tested configuration. It does not qualify
automatic MTP, hard controls, sampled/logprob paths, physical concurrency,
incremental MTP SSE, multimodal serving, or production numerical/task gates.

## PS5 BF16 generic-provider follow-up

The effective-KV admission repair is also live-qualified at commit
`f1ee044e5`. With BF16 KV and the server's FP32 scale metadata, both
`speculative_mtp=true` and generic `speculative=true` return HTTP 200 and run
`qwen4exp_mtp_batched_state_rows_verify`. Each short 32-token probe reports 27
proposed, 21 accepted, 6 rejected draft tokens and 9 cycles; the generic row
also reports `selected_route: speculative` and
`decision_reason: resident_dynamic_mtp`. Both speculative outputs match the AR
reference token IDs. The compact evidence is
[`2026-10-17-zbook-qwen4exp-bf16-generic-live.json`](../benchmarks/results/2026-10-17-zbook-qwen4exp-bf16-generic-live.json).
This closes PS5/task 18 for the tested BF16 configuration only. It does not
qualify INT8, sampled speculation, automatic MTP selection, concurrency, or
streaming.

## PS3 thinking-control follow-up

Text AR now enforces tokenized hard-thinking controls. A 29-request check on
server source `d3bc73256`, production `UD-Q4_K_XL`, BF16 KV, 8K context and
one active slot verifies zero/small caps, max-think-only and reserve-only
requests, custom close delimiters and JSON Schema. Each case runs blocking
and SSE, with AR explicitly selected and with MTP requested under `hard`
policy. The latter takes the named AR fallback; this is not controlled MTP.

All seven AR/fallback token-ID pairs agree. All 14 blocking/SSE pairs agree
on reasoning text, answer text, finish details and usage. Closing tokens are
consumed by the model; the reasoning counter includes them. Answer reservation
leaves space after closing but does not force a minimum answer length before
EOS. A separate `hint` override executes raw MTP and reports
`prompt_hint_only`. Impossible close/reserve budgets reject before generation
in the focused API tests. The final service is ready and accepting requests.

[Commands, request shapes, IDs and results](../benchmarks/results/2026-10-17-zbook-qwen4exp-thinking-http.json)
and [`scripts/qwen4exp_thinking_http_gate.py`](../scripts/qwen4exp_thinking_http_gate.py)
provide the functional evidence. This closes PS3/task 16 for those text routes;
controlled speculation, multimodal controls and full task-quality qualification
remain separate work.

## PS4 sampling/logprob follow-up

Text AR now supports request-owned seeded sampling and selected/top logprobs.
The [byte-aware HTTP packet](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampling-bytes-http.json)
at server source `e7d52091e` records 26 passing generation checks and four
separate early-rejection checks. It covers English/Japanese completions and
chat, greedy/sampled distributions, repeated and changed seeds, blocking/SSE,
Japanese stop trimming and explicit-MTP-requested AR fallback. Japanese chat
includes both UTF-8 fragments and the final punctuation's probability; all 12
blocking cases matched against the earlier audit keep their generated IDs.
See [token-byte metadata](TOKEN-LOGPROBS.md) for reconstruction and legacy
completion candidate identity.

The extra hard-thinking/logprob check found a blocking/SSE whitespace mismatch:
blocking content removed the final generated newline while SSE and logprobs
kept it. The repair passes 633 focused tests and a
[12-case live recheck](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampling-whitespace-http.json)
on server source `67fb29d55`. Thinking text, generated IDs, probabilities and
finish accounting are unchanged; blocking content now includes the generated
newline. Greedy, Japanese and stop cases also pass blocking/SSE checks. The
initial probe used an internal thinking field name; the corrected request uses
public `hard_think_cap`. Both earlier attempts are preserved.

`speculative_mtp=true` with sampling took named AR fallback in those live
captures; generic `speculative=true` rejected the controls before execution.
Those are historical admission observations, **not sampled MTP coverage**.

The implementation now admits the Flash-Next provider's declared temperature,
top-k/top-p/min-p, seed, penalties, bias, suppression, ignore-EOS and logprob
controls through either spelling. Its deterministic proposals are accepted
only when they match a draw from the processed target distribution. Capability
metadata reports `processed_target_verification=true`,
`sampling_algorithm=target_draw_match_deterministic_draft`, and the enumerated
`supported_sampling_fields`; this does not relax other providers' guards.
Logprob requests execute the selected provider instead of a direct-AR shortcut.
CPU HTTP tests use the real resident MTP owner with analytic target logits and
verify actual speculative work for completions/chat, blocking/SSE, with and
without logprobs. The PS6 follow-up below qualifies structured controls, stops,
alternate EOS and hard thinking through both MTP spellings. Resident sessions
remain outside that qualification. Those captures used buffered SSE; PS7 below
covers incremental text delivery. Automatic MTP remains off for this provider.

The first [98-request sampled-MTP audit](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampled-mtp-http-diagnostic.json)
passes 75 request checks. Twenty SSE checks lacked the opt-in execution metadata
because the probe omitted `stream_options.include_hipengine`. Three repeated
requests expose a service bug when model-default EOS is exactly the last allowed
token: finish normalization replaces EOS with length and drops sampler metadata.
The CPU repair preserves the provider's resolved EOS while honoring explicit
EOS/ignore-EOS controls. The probe now requests telemetry and can replay only
failed rows. The 23-case live replay on `6d158ad7b` confirms all three EOS
repairs with unchanged text, IDs and probabilities. Its 20 SSE requests expose
a second defect: buffered logprob responses omit the documented compact MTP
summary and route shape despite carrying actual execution telemetry. The
buffered response paths now forward that summary, with CPU coverage for both
endpoints/spellings and opt-in/opt-out metadata. Both earlier attempts are
preserved as failed packets.

The [functional qualification packet](../benchmarks/results/2026-10-17-zbook-qwen4exp-sampled-mtp-http.json)
combines the original 75 passing requests, three repaired EOS requests, and
20 passing SSE requests on server source `e2fd3bf87`: **98/98 request checks
and 60/60 comparisons pass**. This is focused-repair evidence, not a fresh
98-request run on one final source. All ten category-suite prompts and eight
heldouts have true AR controls, both sampled-MTP spellings, buffered SSE and
late same-seed repeats. Two completion cases also test changed seeds and
logprob-disabled requests. Eighty requests execute actual MTP; 18 execute AR.
The final SSE replay preserves text, IDs, probabilities and usage while adding
the missing summary. Every merged body and comparison was revalidated.

This closes PS4/task 17 for C1 text sampling/logprobs on the tested zbook
production `UD-Q4_K_XL`, BF16 KV, Q8_0 MTP B3, 8K-capacity configuration with
48-token outputs. CPU distribution and transaction tests complement the live
functional checks. It does not qualify task quality, production numerical
envelopes, constrained MTP, incremental SSE, concurrency, economics or
automatic MTP. Final readiness passed and the private audit server was stopped.

## PS6 controlled-MTP follow-up

The [controlled-MTP evidence packet](../benchmarks/results/2026-10-17-zbook-qwen4exp-controlled-mtp-http.json)
qualifies C1 text controls on zbook with production `UD-Q4_K_XL`, BF16 KV,
Q8_0 MTP B3 and 8K capacity. All **170 request checks and 82 blocking/SSE
comparisons pass**. Every request proves actual MTP through the execution path,
target-draw algorithm, nonzero draft/cycle counters and canonical token usage;
AR fallback is not counted.

The checks comprise 80 schema/tool/thinking requests, 84 completion boundary
requests, and six additional chat/hint checks. Both `speculative=true` and
`speculative_mtp=true` run at temperatures 0 and 0.8, with blocking and buffered
SSE. Structured cases run with and without logprobs. English/Japanese multi-token
stops cover both forms; alternate EOS, `min_tokens` and `ignore_eos` cover
termination and masking. Tool cases include required calls, automatic calls
and automatic direct answers. Tool logprobs describe raw model tool syntax,
not probabilities of parsed or reserialized arguments.

This is completed-audit plus focused-repair evidence, **not a fresh full run on
one final source**. The original structured audit exposed missing pure-tool SSE
logprobs; stronger review also found that explicit-MTP queue submission relaxed
hard thinking despite the server's hard policy. Sixteen tool and sixteen thinking
requests were replayed after their respective repairs. Hard-thinking verification
checks the four-token cap, tokenizer-derived closing ID, phase and answer counters,
not merely the presence of reasoning text. Intentional request-level `hint`
override still reports `prompt_hint_only` without forced hard closure.

The boundary audit found four no-logprob SSE failures: an in-context token sequence
need not match a stop string's standalone BPE encoding. The live path published
text past the stop before applying its final text filter, then lost canonical
accounting. Server source `f69838726` routes nonempty stop strings through the
buffered response path in both endpoints. Eight completion SSE replays match their
original blocking results, and two additional chat blocking/SSE pairs pass.
When backend token matching misses such a stop, generation may reach its token
limit before text trimming; usage correctly counts all generated tokens, not just
visible text. PS7 below qualifies incremental text filtering; tokenization-mismatched
stops still drain generation rather than terminating it early.

All merged saved bodies and comparisons were revalidated. Lower-layer tests cover
request-owned grammar and processor state, root/mid-verify/terminal cancellation,
exact RNG checkpoints, target/draft rollback and withholding uncommitted IDs and
logprobs. The final stop repair passes 782 affected CPU HTTP tests. Final live
readiness passes, and the private audit server is stopped. This closes PS6/task 19
for the stated functional scope, not task quality, numerical-envelope qualification,
incremental streaming, physical concurrency, multimodal controls or automatic MTP.

## PS7 streaming qualification

C1 text MTP streaming is functionally qualified on zbook gfx1151 with production
`UD-Q4_K_XL`, BF16 KV, Q8_0 MTP B3 and 8K capacity. This is completed-audit plus
focused-repair evidence, not a fresh run of the full campaign on one source.

| Requirement | Evidence |
| --- | --- |
| Incremental resident completion/chat output, logprobs and terminal accounting | [Eight blocking/SSE pairs](../benchmarks/results/2026-10-17-zbook-qwen4exp-chat-cycle-stream-diagnostic.json) match text, IDs, logprobs, usage and semantic finish; streams publish before completion. |
| UTF-8 and stop boundaries | [Eight English/Japanese completion/chat pairs](../benchmarks/results/2026-10-17-zbook-qwen4exp-stop-stream-diagnostic.json) suppress stops and preserve complete-token probabilities. Partial-token suffixes have no invented logprob. |
| Direct Python provider | [Four hardware cases](../benchmarks/results/2026-10-17-zbook-qwen4exp-direct-stream-functional.json) yield before completion, match blocking output and recover after early iterator closure. |
| Disconnect and recovery | The chat and stop packets close streams early, then verify HTTP 200 recovery and healthy readiness. |
| Slow-client backpressure | [Live TCP audit](../benchmarks/results/2026-10-17-zbook-qwen4exp-mtp-live-backpressure.json) establishes speculative scheduler work, stalls the receive window, receives `client_backpressure` and `[DONE]` after resuming reads, and verifies two subsequent MTP requests. |
| Validated schema/tool/thinking responses | [Ten structured SSE cases](../benchmarks/results/2026-10-17-zbook-qwen4exp-structured-sse-timing.json) pass correctness but deliver only at completion. These are buffered responses, not incremental qualification. |

The scope excludes removing buffers required by the [API validation contract](API.md).
Tool arguments and schema answers are withheld until complete validation; reasoning
in those request shapes is also buffered. Invalid calls must never appear as successful
`tool_calls`. The task's earlier blanket instruction to eliminate full-response
buffering is narrowed to unstructured C1 text, not permission to weaken this contract.
PS7/task 20 is complete for this explicit scope. Incremental validated reasoning/tool
output is not advertised or qualified. Final capability review must preserve that
restriction. When a text-level stop does not match the backend's tokenized stop,
the server filters visible output but drains generation for canonical accounting.

These functional checks do not qualify concurrent widths, a latency SLO, production
numerical/task quality, automatic MTP selection or other hardware. The final campaign
regression and soak remain required.

## Baseline feature grid (before PS1/PS3/PS5 fixes)

| Check | AR | Explicit MTP | Verdict |
| --- | --- | --- | --- |
| Default request | Runs AR; reports `automatic_mtp_scope_not_promoted` | Not selected automatically, even with the sidecar attached | Deliberate default, not a working automatic MTP feature |
| Ordinary short HTTP generation | All ten category-suite requests succeed | All ten really execute `qwen4exp_mtp_batched_state_rows_verify` | Working in this tested scope |
| 256-token request | Returns 256 IDs | Returns 256 IDs with actual MTP execution | Working; not just HTTP 200 |
| Same output across AR and MTP | Comparison reference | 6/10 complete sequences identical; four first differ after 70, 22, 107 and 51 matching tokens | Production diagnostic, not a strict-equivalence claim |
| Thinking enabled | Reasoning text is separated from answer text | Works under the server's `hint` policy | Prompt-level thinking works; hard enforcement is separate |
| Explicit thinking budget | Accepts an 8-token budget but emits 64 reasoning tokens and no answer | `hint` deliberately relaxes enforcement; explicit `hard` falls back to AR and inherits its missing enforcement | **Not working as a hard budget** |
| Logprobs | 501 `unsupported_feature` after generation | Falls back to AR, then the same 501 | **Missing; rejection is too late** |
| Multi-token stop | Returns `alpha`, excludes ` beta gamma`, reports stop | Succeeds only by running AR with named `stop_token_sequences` blocker | AR works; MTP support is not established |
| JSON Schema | Returns schema-valid `{"answer": 42}` | Succeeds only by running AR with named processor blocker | AR works; MTP-constrained decoding is missing |
| Required tool call | Fresh AR request emits the declared `weather` call | Not qualified in this audit | AR native-tool path works; no new semantic tool-quality claim |
| SSE streaming | 64 content events; first content arrives before completion | One content event at completion; actual MTP executes | AR live streaming works; **MTP is buffered** |
| `speculative: true` | Not applicable | 400 because the generic provider requires FP16 KV-scale metadata while the server defaults to FP32, despite BF16 KV | **Broken with these normal server defaults**; this is a separate provider route, not an unconditional alias |
| `ignore_eos: true` with MTP requested | Runs successfully | Runs AR with `unsupported_sampling_k0` and the exact blocker | Named fallback, not MTP coverage |
| Four simultaneous requests | All succeed, but backend groups are single-row and completion times accumulate | All succeed with actual MTP, but also single-row and sequential | **C4 physical batching is absent** |
| Approximately 30K prompt | 29,912-token request retrieves `VIOLET-7391` | 500 at 30 seconds: `engine service command timed out` | AR retrieval smoke passes; **long HTTP MTP fails** |
| Request after long MTP failure | Immediate short AR recovery request returns HTTP 500 | `/ready` still reports ready | **Recovery/readiness defect**; fresh process handles AR again |
| Non-greedy sampling | Temperature 0.7 rejected with 400 | Falls back to AR, then rejected | Greedy-only, not sampled generation support |

No `used: true` / `fallback_reason: no_provider` contradiction appeared in
these Flash-Next responses. Instead, the baseline had a different reporting
bug: detailed diagnostics showed real proposals/accepts/cycles while the
response summary reported all three as zero. That bug is repaired in this
unit by publishing the existing counts through the standard telemetry keys.
A fresh HTTP request confirms matching nonzero summary/diagnostic counts and
the same 128 generated IDs as before; buffered SSE carries the same counters.
The repair changes reporting only, not inference or automatic selection.
`generation_shape.verifier_rows` still does not describe Flash-Next's actual
verification work; candidate rows must not be called concurrent requests.

## What needs fixing first

1. **Completed: separate generation lifetime from administrative command
   timeout.** `f1ee044e5` gates generation on request-owned lifetime rather than
   the legacy 30-second command wait and publishes truthful service readiness.
   The slow fake/real service tests and the tracked live follow-up cover driver
   failure, shutdown, 29,912-token AR/MTP, explicit deadlines, client
   cancellation, and subsequent recovery. See the PS1 lifetime follow-up above.
2. **Completed for the tested text scope: thinking controls and sampling/logprobs.**
   PS3 verifies hard caps and answer reservation in direct/resident AR,
   blocking and SSE, with honest hard-policy MTP fallback. PS4 verifies AR and
   actual sampled-MTP logprobs, byte-accurate Japanese metadata, seeded repeats,
   request isolation, and combined AR thinking/logprob output. PS6 verifies
   controlled MTP; full production task/numerical qualification remains open.
3. **Completed for BF16:** generic speculative admission now resolves the
   effective KV storage instead of rejecting irrelevant BF16 scale metadata.
   Both request spellings execute the intended service-owned MTP route with
   matching counters and token IDs. The effective non-BF16/INT8 guards and
   fixed server KV policy remain covered by the focused tests. See the PS5
   follow-up above.
4. **Keep automatic MTP off until its served scope is qualified.** Short
   MTP execution is real, but this audit does not qualify task quality,
   determinism/isolation, longer-context economics, or controls. The prior
   [7.7K-context diagnostic](../benchmarks/results/2026-10-15-qwen4exp-mtp-long-context-economics/README.md)
   also measured a loss, not a default-promotion result. Do not select a slower
   exact verifier merely to remove isolated production token differences.
5. **Implement and verify actual concurrency.** The baseline generator caps
   its resident runner pool at two, and its AR batch methods loop over
   request-owned single-row runners. Four HTTP clients do not establish
   width-four kernels. Require measured backend widths, overlapping progress,
   cancellation and isolation before changing concurrency grid cells to pass.
   C1 committed-token streaming is covered by PS7, with validation-buffered
   schema/tool responses explicitly excluded from incremental claims.
6. **Finish product-quality and observability gates.** The
   [October 15 combined production packet](../benchmarks/results/2026-10-15-qwen4exp-x10-final-quality/README.md)
   passes numerical/state-repeat/lifecycle checks but leaves task/scope review
   open (13/18 strict-exact task continuations). Review task validity under the
   production contract rather than using generated-ID equality as the sole
   criterion. Correct verifier-row and capability telemetry alongside that
   work.

## Additional scope limits

- Native 262K allocation admission is not 262K inference qualification. This
  audit establishes one approximately 30K AR retrieval case, not broad
  long-context task quality or repeatability.
- Image/video paths require the mmproj sidecar and retain a 1K request limit.
  They were not requalified in this text-serving audit.
- gfx1100, other quantizations, prefix-cache reuse, and dynamic cancellation
  under every shape remain outside this packet. Do not transfer gfx1151
  evidence or rates to another physical machine.
- The dated bring-up/campaign documents contain historical 1K/2,051-token
  limits and obsolete rates. Use this grid for served behavior; use the
  source-pinned benchmark packet for each performance claim.
