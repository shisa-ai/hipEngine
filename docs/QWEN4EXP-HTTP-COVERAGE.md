# Qwen3.8 Flash-Next HTTP coverage

**Audit date: 2026-10-17. The grid is not complete.** Short greedy AR and
explicit MTP work, but long-context MTP, hard thinking budgets, logprobs,
true multi-request batching, and live MTP streaming are not ready to check off.
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
`f1ee044e5136bddd6ba477c6efacb6356fc7b080`. On the same zbook class of physical
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
`bd0775b2d`. With BF16 KV and the server's FP32 scale metadata, both
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

## Baseline feature grid (before PS1/PS5 fixes)

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
2. **Honor or reject hard controls before decoding.** The Qwen4Exp controlled
   token path applies grammar/suppression/logit bias, not the requested
   thinking state machine or logprob collection. An explicit budget must not
   receive a successful unbounded answer. Unsupported logprobs must fail
   before spending a decode. Capability reporting must distinguish prompt
   hints from enforced budgets. Test both blocking and streaming requests.
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
5. **Implement and verify actual concurrency and incremental MTP output.**
   The generator caps its resident runner pool at two. Its AR batch methods
   loop over request-owned single-row runners, and the provider completes
   prompts sequentially. Four HTTP clients do not establish width-four
   kernels. MTP streaming calls complete generation before yielding one
   chunk. Require measured backend widths, overlapping progress, cancellation,
   and committed-token timing before changing these grid cells to pass.
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
