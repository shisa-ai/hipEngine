# Agentic Inference Roadmap

Last updated: 2026-06-14

This document tracks product and implementation work for the **basic working
inference engine** and its OpenAI-compatible server. It is intentionally focused
on features that make hipEngine useful to local agent harnesses (pi, coding
agents, evaluation runners) while preserving the torch-free runtime, plugin
registry boundaries, and greedy graph fast path.

For detailed sampler mechanics, keep `docs/SAMPLING.md` as the source of truth.
For public request/response behavior, keep `docs/API.md` current. Kernel,
benchmark, and speculative-decode backlogs belong in their dedicated docs unless
they directly affect the serving/harness contract below.

## Current working baseline

Already available or recently added:

- OpenAI-style `/v1/completions` and `/v1/chat/completions` server endpoints.
- Qwen-style chat rendering with `<think>` splitting into `reasoning_content` in
  non-streaming responses and `delta.reasoning_content` in streaming responses.
- Chat `max_tokens=None` can use remaining context automatically.
- Resident session/context preallocation hooks for server use.
- Eager model warmup before server readiness.
- Host-backed functional sampling for PARO/GGUF c=1 and serialized multi-row
  requests; greedy-equivalent requests stay on the graph/argmax fast path.
- Sampling parameters are plumbed through public/server/runtime layers:
  temperature, top-p, top-k, min-p, penalties, logit bias, stop token ids,
  `seed`, and per-row seeds.
- Explicit single-token stop ids and exactly-one-token OpenAI `stop` strings can
  terminate host-sampled PARO/GGUF rows early; multi-token stop suffix matching
  remains open.
- OpenAI-style chat `tools` / `tool_choice` prompt injection and output parsing
  for Qwen-style `<tool_call>{...}</tool_call>` blocks.
- Qwen no-think / thinking-effort compatibility via `enable_thinking`,
  `reasoning_effort`, `chat_template_kwargs`, and nested `thinking`/`reasoning`
  request objects.
- Unknown top-level request parameters are rejected instead of silently ignored.

Known baseline limitations:

- Tool calling is prompt-and-parse, not constrained decoding; malformed
  `<tool_call>` JSON is treated as assistant text.
- Thinking controls are prompt/template controls only; there is no token-level
  thinking budget, logit processor, or forced close sequence yet.
- Server-side reasoning/tool parsing lives above generation; the generation loop
  does not yet expose a canonical token-level decode state.
- Public finish metadata is still coarse (`stop`, `length`, `tool_calls`) and
  does not explain budget pressure, forced tokens, cancellation, cache behavior,
  or per-phase token counts.
- Streaming is content-first; metadata deltas for reasoning/answer/tool token
  counts, timing, and cache state are not first-class yet.
- Resident session reuse needs an explicit commit policy before hidden reasoning
  or failed tool-call attempts can safely be retained across turns.

## Guiding principles

1. **Build primitives, not one-off hacks.** Thinking budgets, stop sequences,
   JSON/tool constraints, min-answer reserves, logit bias, and suppress-EOS
   behavior should share decode-state and logit-processor primitives.
2. **Be honest in metadata.** If the engine forces a token, appends synthetic
   text, truncates mid-structure, cancels a request, or drops session state, the
   response must say so.
3. **Keep harnesses in control.** Agents need token counts, continuation handles,
   cancellation, deadlines, cache handles, and predictable tool/JSON behavior.
4. **Do not poison resident context.** Hidden reasoning, malformed tool-call
   attempts, and truncated outputs should not be silently committed into a
   long-lived session.
5. **Preserve the fast path.** Greedy-equivalent requests remain on the current
   graph/argmax path unless a replacement is proven exact and non-regressive.
6. **Keep boundaries clean.** No torch on the hot path; no backend/quant/model
   `if` ladders in engine or dispatch code.

## Priority roadmap

### P0 — Decode observability and robust finish semantics

These are the foundation for every agent-friendly feature below.

- [ ] **Canonical `DecodeState` / `GenerationTelemetry`.** Track, per row:
  request id, row index, step index, prompt tokens, generated tokens,
  reasoning/answer/tool-call phase, stop-suffix state, forced-token queue state,
  EOS/stop/length status, and sampler mode. Keep it generation-layer owned, not
  server-only.
  - Exit gate: fake-session tests prove streaming and non-streaming paths report
    the same phase/token counts; greedy path output remains byte-identical.
- [ ] **Structured finish details.** Extend results with machine-readable detail
  beyond the OpenAI `finish_reason`: `reason`, `eos_token_id`, `stop_sequence`,
  `length_limit`, `deadline_exceeded`, `cancelled`, `forced_close`,
  `synthetic_tokens`, `reasoning_tokens`, `answer_tokens`, `tool_call_tokens`,
  and `sampler_mode`.
  - Exit gate: server tests cover stop, EOS, length, tool-call, and malformed
    tool-call finishes without changing existing minimal OpenAI fields.
- [ ] **Streaming metadata deltas.** Add optional stream events or `usage_delta`
  payloads for token counts and phase transitions: thinking -> answer ->
  tool-call -> done. Include TTFT, prefill ms, decode tok/s, cache hit/miss, and
  KV bytes when available.
  - Exit gate: `stream_options.include_usage` remains compatible; opt-in
    metadata is ignored safely by plain OpenAI clients.
- [ ] **Token diagnostics endpoints.** Add `/tokenize`, `/detokenize`,
  `/count_tokens`, and `/fit_context` helpers for harness preflight.
  - Exit gate: requests can learn prompt token count, remaining output budget,
    model/server max context, and truncation decisions before generation.
- [ ] **Request cancellation and deadlines.** Make cancellation/deadline checks
  visible to the scheduler and decode loop, not only the HTTP layer.
  - Exit gate: cancel/deadline tests leave no active row/session leak and return
    explicit finish details.

### P1 — Controlled decoding and thinking budgets

Build this on top of P0 telemetry rather than as a Qwen-only special case.

- [ ] **General logit-processor framework.** Define a documented processor order
  shared by host and future GPU paths: static logit bias, suppress tokens,
  penalties, min-token/EOS suppression, forced-token queue, stop DFA, dynamic
  budget processors, and later grammar constraints.
  - Exit gate: processed-argmax fixtures prove deterministic tie-breaking and
    no regression for greedy-equivalent requests.
- [ ] **Forced-token queue.** Allow policies to force a known token sequence
  through the normal decode path so KV state stays consistent.
  - Exit gate: forced multi-token delimiters are emitted exactly once and count
    as forced in finish metadata.
- [ ] **Multi-token stop suffix matching.** Lower stop strings to token-id
  sequences, track suffix/DFA state per row, and finish as soon as a full stop
  sequence completes while preserving response trimming.
  - Exit gate: one-token stop behavior stays green; overlapping multi-token stop
    fixtures pass for PARO and GGUF host-sampled paths.
- [ ] **Thinking budget policy.** Add request-level controls such as
  `max_think_tokens`, `min_answer_tokens`, `hard_think_cap`, and
  `thinking_soft_close_tokens`. When budget pressure begins, bias the first
  token(s) of accepted close sequences such as `</think>`, then force the rest
  of the delimiter once started.
  - Exit gate: a model still in reasoning near the reserve budget transitions to
    answer mode or returns `thinking_budget_exhausted` with `forced_close=true`.
- [ ] **Graceful length exhaustion.** If length is hit mid-reasoning,
  mid-tool-call, or mid-JSON object, return honest details and a continuation
  option instead of silently producing an empty or malformed final answer.
  - Exit gate: no synthetic text is appended without `synthetic_tokens` metadata;
    forced-through-model tokens remain eligible for session commit.
- [ ] **Continuation handles.** For `finish_reason="length"` or controlled tail
  stops, return a resumable generation handle so harnesses can ask for more
  output without reprefilling the entire prompt.
  - Exit gate: continuation works after normal text and after a partial tool/JSON
    structure, or explicitly reports why it cannot resume.

### P2 — Tool-call and structured-output reliability

The current tool-call support is enough for local smoke tests; harness-grade tool
use needs decoding constraints and better protocol coverage.

- [ ] **Strict tool-call mode.** Support `tool_choice="required"`, specific tool
  choice, and no-tool mode at decode time rather than prompt-only. Emit exactly
  one valid call when required, or a structured refusal/error finish.
  - Exit gate: server fixtures cover `none`, `auto`, `required`, and specific
    function choice with deterministic fake logits.
- [ ] **Tool JSON schema validation.** Validate generated tool arguments against
  the provided JSON schema and surface validation errors separately from normal
  assistant text. Decide whether invalid calls are returned, repaired, retried,
  or downgraded to text.
  - Exit gate: malformed JSON, unknown tool names, missing required args, and
    wrong types each have stable finish details.
- [ ] **Constrained JSON / schema decoding.** Add JSON-object and JSON-schema
  constrained decoding usable for tool arguments, `response_format`, and harness
  control messages.
  - Exit gate: valid-close-brace before EOS is enforced on fixture prompts; plain
    sampling behavior is unchanged when constraints are absent.
- [ ] **Patch/diff constrained mode.** Add an optional unified-diff or patch
  grammar for coding agents that need valid edit blocks instead of free-form
  prose.
  - Exit gate: generated patches parse under the selected grammar and report
    grammar finish details.
- [ ] **Tool streaming polish.** Stream tool-call name/arguments chunks in a way
  OpenAI-compatible clients can consume incrementally, including stable call ids
  and index handling for multiple calls.
  - Exit gate: streaming and non-streaming tool responses round-trip to the same
    parsed tool-call list.

### P3 — Session, cache, and context control

Agents repeatedly reuse long system prompts, repo summaries, and tool traces.
Expose explicit controls instead of relying on implicit resident-session behavior.

- [ ] **Selective session commit policy.** Add per-request/session commit modes:
  `append_all`, `append_visible_only`, `append_none`, and possibly
  `append_prompt_only`. Default server behavior should avoid committing hidden
  reasoning unless explicitly requested.
  - Exit gate: hidden `<think>` tokens and malformed/truncated tool-call attempts
    are not retained under `append_visible_only`; state accounting remains exact.
- [ ] **Visible-only re-prefill path.** If the generated raw tokens differ from
  the visible committed transcript, re-prefill the visible assistant answer/tool
  call so resident KV matches what future turns will see.
  - Exit gate: follow-up turn logits match a stateless prompt built from the
    visible transcript.
- [ ] **Forkable prefix/session cache.** Add cache handles for pinned prefixes and
  forkable conversations: `cache_key`, `fork_from`, `rollback_to`, `delete`, and
  cache usage metadata.
  - Exit gate: two branches can fork from one prefix without cross-contaminating
    generated turns; eviction reports are deterministic.
- [ ] **Context fitting policy.** Make truncation/auto-clear behavior explicit:
  fail, truncate oldest visible turns, keep pinned system prefix, or start a new
  session. Return what was kept/dropped.
  - Exit gate: `/fit_context` and generation use the same token accounting.
- [ ] **Session snapshot save/restore.** Persist prefix/session state once the
  cache layout is stable.
  - Exit gate: restored sessions pass deterministic continuation fixtures.

### P4 — Scheduler, batching, and native sampling polish

These items overlap with `docs/SAMPLING.md`; keep the detailed sampler plan
there and use this section to track serving impact.

- [ ] **Native c>N stochastic execution.** Use scheduler-owned `RowSamplingState`
  for true batched sampled decode instead of serial c=1 row routing.
  - Exit gate: c=2/4 fixed-seed fixtures are deterministic and match independent
    c=1 semantics where expected.
- [ ] **GPU sampler kernels.** Promote top-k/temperature/logit-processor kernels
  after CPU-reference fixtures and GPU1 determinism gates pass.
  - Exit gate: selected requests avoid full-vocab D2H logits copies and provide
    rocprof evidence before any performance claim.
- [ ] **Exact GPU top-p.** Implement or explicitly defer full-vocab nucleus
  sampling without weakening retain-one/top-p semantics.
  - Exit gate: boundary fixtures match host retained sets.
- [ ] **Public logprobs.** Return selected logprob and optional top-logprobs
  through library and server responses.
  - Exit gate: response schema tests pass for greedy, processed-argmax, and
    sampled requests.
- [ ] **Admission/backpressure policy.** Expose queue depth, max active requests,
  reject/429 behavior, and Retry-After hints for harness retry loops.
  - Exit gate: overload tests do not deadlock active resident sessions.

### P5 — Harness integration and operations

- [ ] **Capabilities manifest.** Expose model/server capabilities through
  `/v1/models` metadata or `/v1/hipengine/capabilities`: context sizes,
  tokenizer name, chat template family, tool support, reasoning controls,
  sampling modes, logprobs support, continuation support, and cache support.
- [ ] **Pi/local-agent config snippets.** Keep a tested minimal pi config for
  hipEngine: base URL, tool calling, Qwen thinking format, reasoning toggle,
  timeout/deadline recommendations, and known unsupported fields.
- [ ] **Golden harness traces.** Maintain fixtures for a full agent loop:
  assistant -> tool call -> tool result -> assistant final answer, with both
  streaming and non-streaming variants.
- [ ] **Error taxonomy.** Standardize `unsupported_parameter`,
  `invalid_tool_call`, `schema_violation`, `context_overflow`,
  `deadline_exceeded`, `cancelled`, and `engine_busy` responses.
- [ ] **Health/readiness diagnostics.** Extend readiness to include model loaded,
  warmup complete, allocator/KV capacity, graph cache status, and selected GPU.
- [ ] **Deterministic replay bundle.** Allow a failed harness request to emit a
  compact replay artifact: request JSON, model id, sampler params, seed, token
  counts, finish details, and optional redacted prompt hashes.

## Near-term implementation slices

Good next logical units, in order:

1. **DecodeState MVP:** add generation-layer phase/token accounting and finish
   details, then thread it through server responses without changing output text.
2. **Token diagnostics endpoints:** expose tokenizer/count/fit helpers; useful to
   every harness and low risk to runtime performance.
3. **Stop DFA + forced-token queue:** finish the stop-sequence gap from
   `docs/SAMPLING.md` and create the primitive needed for thinking close and
   structured decoding.
4. **Thinking budget MVP:** close `</think>` through forced tokens, reserve answer
   tokens, and report budget-exhaustion metadata.
5. **Strict tool-call fixtures:** lock down prompt/render/parse behavior and add
   schema-validation errors before attempting grammar-constrained decoding.
6. **Session commit policy:** prevent hidden reasoning or malformed partial tool
   calls from being silently retained in resident sessions.

## Validation expectations

For each roadmap item that changes runtime behavior:

- Add a focused unit/fake-session test first where practical.
- Prove greedy-equivalent generation still uses the fast path and remains exact.
- Keep server OpenAI compatibility tests for old minimal responses.
- Add or update `docs/API.md` for public behavior changes.
- Add a `WORKLOG.md` entry with commands and results.
- For any performance claim, follow `docs/BENCHMARK.md`; this roadmap alone is
  not benchmark evidence.
