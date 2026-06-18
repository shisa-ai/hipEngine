# Agentic Inference Roadmap

Last updated: 2026-06-18

`AGENTIC.md` is the implementation handoff for making hipEngine useful as a
local **agent runtime**. The scope is not broad project management; it is the
serving/library behavior that coding agents, harnesses, and evaluation runners
need from an inference engine: controllable reasoning, useful output under tight
budgets, reliable tool and structured-output decoding, resumable generations,
explicit session/cache control, cancellation/deadline handling, diagnostics, and
later model routing / multi-model / multi-GPU serving.

The roadmap assumes these invariants remain fixed:

- the runtime hot path stays torch-free;
- dispatch remains registry-driven, not backend/quant/model `if` ladders;
- greedy-equivalent requests stay on the retained graph/argmax fast path unless
  a replacement is proven exact and non-regressive;
- public behavior changes are documented in `docs/API.md`;
- sampler mechanics stay aligned with `docs/SAMPLING.md`.

Kernel, benchmark, speculative-decode, and low-level performance backlogs belong
in their dedicated docs unless they directly affect the agent/harness contract
below.

## Current working baseline

Already available or recently added:

- OpenAI-style `/v1/completions` and `/v1/chat/completions` server endpoints.
- Qwen-style chat rendering with `<think>` splitting into `reasoning_content` in
  non-streaming responses and `delta.reasoning_content` in streaming responses.
- Chat requests that omit `max_tokens` use the server's bounded
  `--chat-default-max-tokens` default (`4096`) clamped to remaining admitted
  context; `--chat-default-max-tokens auto` restores full remaining-context
  behavior.
- Resident session/context preallocation hooks for server use.
- Eager model warmup before server readiness.
- Host-backed functional sampling for PARO/GGUF c=1 and serialized multi-row
  requests; when the vocabulary/logits width is known, backend decode-state
  telemetry marks the full-vocab host readback with `full_vocab_logits_d2h=true`
  and per-token `logits_d2h_bytes`. Greedy-equivalent requests stay on the
  graph/argmax fast path.
- A default-on PARO native GPU sampler route exists for supported sampled
  requests; `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` disables it for rollback. It
  covers c=1 and
  scheduler-owned c>N serial per-slot decode when every row is GPU-sampler
  eligible. It supports logit bias/history-penalty processors, full-vocab
  temperature sampling, bounded `top_k <= 64`, full-vocab and bounded top-k
  `top_p`/`min_p`, suppress-token ids, min-token/EOS policy, selected-token
  logprobs, full-vocab `top_logprobs` for `top_k=0`, bounded
  `top_logprobs <= top_k <= 64`, and post-accept token stops. It does not cover
  true batched c>N sampling, GGUF, or `top_logprobs > top_k` for bounded top-k
  requests.
- Sampling parameters are plumbed through public/server/runtime layers:
  temperature, top-p, top-k, min-p, penalties, logit bias, suppress token ids,
  min-token/EOS policy, stop token ids, stop token sequences, `seed`, and
  per-row seeds.
- Detailed logprob metadata is available through the host-logits metadata path:
  completions accept `logprobs: N`; chat accepts `logprobs: true` plus optional
  `top_logprobs: N`; completion `echo+logprobs` returns the echoed prompt as a
  prefix entry with `null` prompt logprob and omission reason
  `prompt_logprob_unavailable`; streaming completion/chat logprobs use a
  buffered detailed-generation path by default, or live token/chunk
  streams when the engine explicitly advertises `supports_stream_logprobs` and
  yields per-chunk token metadata. PARO/GGUF c=1 host-sampled streams emit that
  live token metadata for logprob requests. Chat logprobs attach only to visible
  assistant content after `<think>` splitting; hidden reasoning deltas do not
  receive OpenAI `logprobs.content` entries. If generated-token metadata exists
  but the selected score is omitted, the OpenAI-compatible score field remains
  `null` and `choices[].logprobs.hipengine.omitted_token_logprobs[]` reports
  token index/id/text plus reason `backend_omitted_logprob`.
- Tokenizable OpenAI `stop` strings lower to `stop_token_ids` or
  `stop_token_sequences`; PARO/GGUF host-sampled rows terminate on suffix match
  while responses still use post-trimming for consistency. PARO c=1 and serial
  per-slot c>N native sampling check the same stop metadata after token
  selection; GGUF GPU sampling still needs parity.
- Request deadlines are exposed as per-request `timeout_ms` and server default
  `--request-timeout-ms` / `HIPENGINE_REQUEST_TIMEOUT_MS`. Buffered requests
  return HTTP 408 with structured deadline finish details; live streams emit an
  SSE error chunk with the same detail and then `[DONE]`. The server lowers the
  deadline and cancellation token into `SamplingParams` / `GenerationRequest`,
  and PARO/GGUF generation checks them cooperatively around tokenization,
  prefill, decode, host-sampled steps, and graph replay boundaries.
- OpenAI-style chat `tools` / `tool_choice` prompt injection and output parsing
  for Qwen-style `<tool_call>{...}</tool_call>` blocks.
- Parsed tool calls are always checked against the request's declared tool
  names, and multiple parsed calls are rejected unless the request explicitly
  opts in with `parallel_tool_calls=true`.
- Strict tool result validation for `tool_choice="none"`, `"required"`,
  specific function choices, functions with `"strict": true`, and explicit
  `parallel_tool_calls`; failures return normal chat responses with no
  successful `tool_calls` and stable `finish_details.reason`.
- Tokenizer-backed tool-call start controls: no-tool mode suppresses the first
  `<tool_call>` token; required/specific tool modes force the tokenized
  `<tool_call>` marker immediately, or after tokenized `</think>` close when a
  thinking budget is active. Specific function choices, plus `required` mode
  with exactly one function tool, also force the tokenized
  `<tool_call>{"name":"...","arguments":` prefix when tokenizer composition
  proves it is aligned with the forced start marker. Required/specific tool
  modes repair tokenized `</tool_call>` close markers by forcing the remaining
  suffix once the marker starts.
- Qwen no-think / thinking-effort compatibility via `enable_thinking`,
  `reasoning_effort`, `chat_template_kwargs`, and nested `thinking`/`reasoning`
  request objects. Numeric budget aliases and explicit budget fields are
  accepted and rendered as prompt hints: `thinking_token_budget`,
  `chat_template_kwargs.thinking_budget`, `thinking.budget_tokens`,
  `thinking.max_tokens`, `max_think_tokens`, `min_answer_tokens`,
  `hard_think_cap`, `soft_close_window`, `hard_close_message`, and
  `hard_close_sequence`. When the served engine exposes tokenization, chat
  hard caps lower the close marker/string into token ids and host sampling
  forces that close sequence at the cap.
- Unknown top-level request parameters are rejected instead of silently ignored.

Known baseline limitations:

- Tool calling is prompt-and-parse plus limited marker repair, not full
  constrained decoding. A duplicated `<tool_call>` start marker wrapping valid
  inner tool JSON is recovered in compatibility parsing. Tool-enabled requests
  now fail closed on unparseable `<tool_call>` markup, including malformed or
  unclosed JSON blocks, by returning a normal chat response with
  `finish_details.reason="invalid_tool_call"` and no assistant content. The
  parser selects valid tool-call spans by successful JSON tool-call parsing, so
  literal marker text before a later valid call is preserved as assistant
  content and marker-looking text inside JSON arguments does not prematurely
  terminate the tool-call block.
- Thinking control is still not constrained decoding. Tokenized thinking caps
  are enforced only on host-sampled PARO/GGUF rows: the soft window applies a
  sparse close-token bias ramp, EOS is suppressed until answer phase when an
  EOS token id is available, and the hard cap forces the close sequence. Native
  GPU/MTP parity remains future work.
- Server-side reasoning/tool parsing lives above generation. PARO/GGUF
  generation loops emit final decode-state telemetry snapshots. PARO and GGUF
  c=1 true streaming also emit live `GenerationStreamChunk` snapshots for
  greedy and sampled answer tokens, including sampler mode and
  fallback/blocker metadata. Scheduler-owned submit/poll token events now carry
  telemetry-bearing `GenerationStreamChunk` snapshots synthesized from
  `RowSamplingState`, including sampler mode, active processors, native fallback,
  token stop suffix state, forced-token queues, thinking-budget pressure, and
  scheduler execution flags.
  PARO scheduler-owned c>N batch diagnostics and GGUF serial c>N final
  diagnostics retain JSON-ready `scheduler_token_chunks` with per-token
  `GenerationStreamChunk` payloads, and final telemetry reports scheduler
  execution path plus native-prefill/native-decode/serial-fallback state where
  the backend exposes those flags. Buffered completion
  streams, plain buffered chat answer/reasoning streams, plain chat content
  streams with logprobs, reasoning streams with hipEngine-private logprob
  metadata, plus validated structured chat
  content streams and validated tool-call argument spans, for a single HTTP
  request can forward those scheduler chunks as per-token public SSE deltas or
  `delta.tool_calls` argument fragments when the chunk text exactly reconstructs
  the public choice text; if separate HTTP requests were coalesced into one
  backend batch, the server withholds the backend row chunks instead of exposing
  ambiguous row ids. Invalid tool calls, tool outputs whose argument spans
  cannot be mapped safely, structured-output validation failures, and logprob
  chunks that cannot be mapped to emitted content/reasoning deltas keep the
  conservative buffered parser paths. Unmappable scheduler tool/logprob chunks
  add sanitized final-choice diagnostics with counts, hashes, and execution
  paths, not generated text.
  Other buffered streaming paths preserve final backend
  telemetry on choice `done` chunks and, when tokenizer/counting hooks are
  available, emit server-derived per-delta token/decode-state snapshots for
  parsed answer/reasoning/tool/structured chunks while inheriting stable backend
  sampler/execution metadata such as processor blockers, sampler fallback,
  logits-readback state, and scheduler execution flags. Canonical live
  reasoning/tool-call/structured phases still need lower-loop signals.
- Public finish metadata now carries basic PARO/GGUF backend reasons for EOS,
  token stop, stop sequence, length, sampler mode, server post-parse tool-call
  phase/counts, and host-sampled thinking-budget forced close. Cooperative
  backend deadline/cancellation checks are wired through server HTTP/SSE error
  contracts and PARO/GGUF generation-loop boundaries, including scheduler-owned
  PARO c>N reclaim. True mid-kernel or mid-graph preemption, cache behavior,
  canonical live per-phase token counts, and broader budget-pressure/cache/KV
  runtime signals still need lower-loop coverage.
- Streaming logprobs are buffered detailed responses unless the engine
  advertises live stream logprob support and each emitted chunk carries token
  metadata. PARO/GGUF c=1 host-sampled streams now satisfy that contract for
  live logprob requests. Completion `echo+logprobs` does not compute real
  prompt-token logprobs yet; the echoed prompt is represented as a prefix entry
  with `null` logprob and stable omission reason
  `prompt_logprob_unavailable`.
- Streaming supports opt-in `stream_options.include_hipengine` metadata with
  server-measured elapsed time, TTFT, decode-rate timing, final finish details,
  and best-effort token accounting. When a backend yields
  `GenerationStreamChunk.telemetry`, the server preserves that decode-state
  snapshot and raw backend `timing` payload in the choice payload, adds only
  server-derived token counters, and mirrors backend timing into top-level SSE
  `hipengine.timing` with `backend_` prefixes such as `backend_prefill_ms`.
  When the final live chunk also carries `GenerationStreamChunk.finish_details`,
  the server uses those backend-authored details on the final choice unless
  server post-processing, such as stop-string trimming or tool/structured
  validation, overrides the result.
  Final done/usage chunks also include sanitized server-observed KV pool stats
  when the served engine exposes them. Exact backend-authored per-phase counts,
  cache hit/miss state, per-request KV-byte deltas, and budget pressure still
  need broader runtime signals.
- Public agent/runtime capability discovery is exposed through
  `/v1/hipengine/capabilities`. Limited deterministic buffered continuation
  handles exist, but they re-prefill stored rendered prompt plus generated text;
  resident KV continuation reuse is still future work. App-local buffered chat
  transcript sessions exist for explicit `session.id` requests, but they
  re-render the visible transcript rather than reusing resident KV state.
- Server startup supports one loaded model at a time. Multiple resident models,
  capability-aware routing, and model-family fallback are not implemented.
  Tensor parallelism has a disabled capability manifest and a design gate in
  `docs/TENSOR_PARALLEL.md`, but no runtime path yet.

## Implementation principles

1. **Build primitives, not one-off hacks.** Thinking budgets, stop sequences,
   JSON/tool constraints, min-answer reserves, logit bias, and suppress-EOS
   behavior should share decode-state, logit-processor, DFA, and forced-token
   primitives.
2. **Keep behavior observable.** If the engine forces a token, appends synthetic
   text, truncates mid-structure, cancels a request, clears context, falls back
   to host sampling, or drops session state, the response must say so.
3. **Keep harnesses in control.** Agents need token counts, continuation handles,
   cancellation, deadlines, cache handles, model capabilities, and predictable
   tool/JSON behavior.
4. **Do not poison resident context.** Hidden reasoning, malformed tool-call
   attempts, and truncated outputs should not be silently committed into a
   long-lived session.
5. **Preserve the fast path.** Greedy-equivalent requests remain on the current
   graph/argmax path unless a replacement is proven exact and non-regressive.
6. **Keep model-specific syntax at model/template boundaries.** Qwen `<think>`
   and `<tool_call>` markup is a current served-template behavior, not a reason
   to hardcode Qwen checks in engine/dispatch internals.

## Server contract tests

Agentic behavior needs explicit server-level tests. Prefer deterministic fake-LLM
tests over live-agent loops for the default suite; they pin the contract without
depending on model sampling luck.

Minimum matrix:

- chat rendering for developer/user/assistant/tool messages, tool schemas,
  `tool_choice`, and thinking/no-think controls;
- non-streaming and streaming response shapes for reasoning spans, parsed tool
  calls, parallel tool-call indexes, structured-output validation, and final
  `finish_details`;
- strict failure cases for missing required tools, wrong tool names, malformed
  tool-call blocks, schema violations, `tool_choice="none"`, response-format
  violations, length truncation, deadline/cancel errors, and unsupported fields;
- local-agent and pi config validation against `/v1/hipengine/capabilities`, so
  clients do not advertise thinking/tool features the server cannot honor;
- opt-in replay artifacts for failed HTTP requests and normal strict
  result-validation failures, with prompt/tool-result redaction verified;
- session commit modes, transcript prepending, visible-only reasoning stripping,
  tool-call transcript replay, and unsafe-finish downgrade behavior;
- sampler/MTP compatibility guards for every processor field that changes token
  selection or post-accept behavior.

Current code reality:

- `tests/test_server_api.py`, `tests/test_agentic_server_conformance.py`,
  `tests/test_agentic_harness_traces.py`, and `tests/test_local_agent_config.py`
  cover the current matrix with fake engines and checked-in golden traces.
- `tests/test_sampling.py` exhaustively maps every advertised
  `sampling.speculative_mtp.incompatible_fields` entry to a concrete blocker
  case, and verifies every actual blocker is advertised. This preserves the
  current policy that `logit_bias`, penalties, suppressions, forced-token
  queues, thinking budgets, and logprob requests are normal AR-sampling
  features but raw-argmax MTP blockers.
- These tests prove the server contract and diagnostics; they do not prove a
  particular live model will reliably choose the right tool without future
  decode-time grammar/schema constraints.

## Core primitives to add

These primitives are referenced throughout the punchlist. Build them once and
reuse them.

### Decode state and telemetry

A generation-layer state object should exist per row/request and be available to
server streaming and non-streaming paths. Suggested home: a small module such as
`hipengine/generation/decode_state.py` or `hipengine/generation/control.py`.

Minimum shape:

```python
class DecodePhase(Enum):
    PREFILL = "prefill"
    THINK = "think"
    CLOSING_THINK = "closing_think"
    ANSWER = "answer"
    TOOL_CALL = "tool_call"
    STRUCTURED = "structured"
    DONE = "done"

@dataclass
class DecodeState:
    request_id: str
    row_index: int
    step_index: int
    prompt_tokens: int
    generated_tokens: int
    phase: DecodePhase
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    structured_tokens: int = 0
    stop_suffix_state: object | None = None
    forced_tokens_pending: tuple[int, ...] = ()
    forced_token_id: int | None = None
    forced_token_reason: str | None = None
    forced_tokens_remaining: int | None = None
    active_processors: tuple[str, ...] = ()
    sampler_fast_path_blockers: tuple[str, ...] = ()
    sampler_fallback_reason: str | None = None
    budget_pressure: str | None = None
    sampler_mode: str | None = None
    full_vocab_logits_d2h: bool | None = None
    logits_d2h_bytes: int | None = None
    execution_path: str | None = None
    native_compact_prefill: bool | None = None
    native_caware_decode: bool | None = None
    serial_decode_fallback: bool | None = None
    native_sampler_rows: bool | None = None
    continuation_eligible: bool = False
```

Implementation notes:

- Keep the state independent from model plugins; model plugins may provide
  marker token sequences and chat-template metadata.
- Update it during generation, not only after server text parsing.
- Keep the existing server `_ReasoningSplitter` as a compatibility/output layer
  until decode-state streaming fully replaces it.
- Thread state into `GenerationOutput` / streaming chunks without changing
  existing minimal OpenAI fields for clients that ignore extras.

### Finish details

Add a structured detail object next to coarse OpenAI `finish_reason` values.
Suggested shape:

```python
@dataclass(frozen=True)
class FinishDetails:
    reason: str                         # eos, stop, length, cancelled, ...
    eos_token_id: int | None = None
    stop_sequence: tuple[int, ...] = ()
    length_limit: int | None = None
    deadline_exceeded: bool = False
    cancelled: bool = False
    forced_close: bool = False
    synthetic_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    structured_tokens: int = 0
    budget_pressure: str | None = None
    cache_action: str | None = None
    sampler_mode: str | None = None
    phase: str | None = None
    continuation_eligible: bool | None = None
```

Server responses can expose this under an extension field such as
`finish_details` while preserving `finish_reason` compatibility.

Current code reality:

- `FinishDetails` exists on `GenerationOutput` and the OpenAI server exposes a
  compact `choices[].finish_details` extension on non-streaming responses and
  final SSE choice chunks.
- The server maps detailed backend reasons to coarse OpenAI `finish_reason`
  values without changing legacy fallback behavior: `eos` remains public
  `stop`, `length` maps to public `length`, and parsed tool calls report
  `tool_calls`.
- Chat length stops get best-effort post-parse `finish_details.phase` metadata
  (`reasoning`, `closing_think`, `tool_call`, `structured`, or `answer`) plus
  deterministic buffered continuation handles for normal answer and partial
  structured-output phases. Ineligible length stops, including reasoning,
  closing-think, tool-call, streaming, logprob, sampled, tool, and active
  thinking-budget paths, carry `continuation_eligible=false`.
- Parsed chat tool-call finishes get best-effort server post-parse
  `finish_details.phase="tool_call"` plus `reasoning_tokens`, `answer_tokens`,
  and `tool_call_tokens` when the served engine exposes token counting. This is
  final-response count metadata, not canonical live decode-state telemetry yet.
- PARO/GGUF detailed generation now emits backend finish details for EOS, token
  stop, stop sequence, length, sampler mode, and host-sampled thinking-budget
  forced close. When a `ThinkingBudgetState` forced the close sequence, finish
  details include `forced_close=true`, reasoning/answer token counts,
  `budget_pressure="hard_close"`, and the budget phase. Normal backends still
  need native cancellation/deadline, cache, sampler fallback reason, and richer
  per-phase metadata; absent backend detail, the server falls back to
  `{"reason": finish_reason}`. Server-side deadline errors already emit
  `{"reason": "deadline_exceeded", "deadline_exceeded": true}`.

### Logit processor stack

Add one ordered stack for all token-selection policy. It should work on host
logits first and later be mirrored by GPU kernels for promoted paths.

Current code reality:

- `hipengine.generation.sampling.select_token()` already applies finite-logit
  cleanup, OpenAI-style token-id `logit_bias`, repetition/presence/frequency
  penalties, suppress-token ids, min-token/EOS suppression, top-k/top-p/min-p
  filtering, deterministic row seeds, and logprob summaries on the host path.
- PARO/GGUF stop tokens and stop token sequences are checked after token
  selection in model-specific loops, then server post-trimming keeps OpenAI
  `stop` string responses consistent.
- `PerRowSamplingParams` / `SamplerParamsBlock` already carry per-row
  `logit_bias`, penalties, suppress tokens, min-token/EOS policy, stops, seeds,
  and temperature fields for scheduler integration.
- Standalone GPU sampler-family kernels exist for logit bias, penalties,
  suppress-token ids, min-token/EOS policy, full-vocab temperature sampling,
  bounded `top_k <= 64`, exact full-vocab `top_p`/`min_p`, and bounded top-k
  `top_p`/`min_p` filters. Supported PARO c=1 sampled requests use this route by
  default, and supported PARO c>N sampled batches can route each physical slot
  through the same native sampler state; `HIPENGINE_QWEN35_NATIVE_SAMPLER=0`
  disables the native route for rollback. True batched c>N sampling, GGUF,
  `top_logprobs > top_k`, and dynamic processor parity remain future work for
  the native route.

Required pre-selection processors, in order:

1. normalize non-finite logits to the documented host-sampler semantics;
2. apply static OpenAI-style `logit_bias`;
3. apply repetition/presence/frequency penalties;
4. apply suppress-token ids and min-token / suppress-EOS policy;
5. apply dynamic budget processors, including thinking soft-close bias;
6. apply grammar/JSON/tool-call masks or sparse biases;
7. apply the forced-token queue as the final override when a delimiter/grammar
   sequence is already in progress;
8. run argmax or sampling.

Required post-accept state updates:

1. observe the selected token in the row RNG/history state;
2. update stop DFA suffix state;
3. update grammar/tool/reasoning phase state and token counters;
4. populate telemetry for active processors, suppressed/forced tokens, budget
   pressure, and why a request left the fast path.

Implementation notes:

- Keep a pure, testable planner that decides whether processors require
  `GREEDY_FAST`, `PROCESSED_ARGMAX`, `HOST_LOGITS_SAMPLE`, or a guarded
  `GPU_SAMPLE` path.
- Greedy-equivalent requests with no active processors must still take the graph
  fast path.
- Stop tokens are emitted by the model and then terminate/trim the response; do
  not implement ordinary stop sequences by suppressing the stop token unless the
  requested mode is an explicit grammar/constraint mode.
- Processors should emit metadata: names active, tokens suppressed/forced,
  budget pressure, and why a request left the fast path.

### Speculative/MTP sampler compatibility

Speculative decode must use the same effective token-selection policy as the AR
path before it can be advertised for agent requests with custom sampling policy.

Current code reality:

- Native MTP draft proposals use raw/capped draft-vocab argmax. Exactness is
  preserved only because accepted draft tokens are committed after target-model
  verification over the full vocab.
- The scheduler/speculative accept path accepts candidates by comparing draft
  candidate tokens to target `top1`/accept-summary output; it does not currently
  apply `logit_bias`, penalties, dynamic masks, forced tokens, grammar
  constraints, or stochastic RNG state to target verification.
- The public `LLM.generate()` / OpenAI server path does not expose an MTP
  speculative route today. Existing MTP code is available through speculative
  primitives, resident-runner verify helpers, and benchmark/profiling scripts.
  When that route is promoted into serving, it must gate on the same
  `plan_sampler()` decision used by PARO/GGUF AR generation.
- Current static processors are compatible with normal AR sampling:
  `logit_bias`, suppress-token ids, and min-token/EOS policy are normalized in
  `SamplingParams` / `GenerationRequest`, participate in the sampler plan, apply
  on host logits, and flow through scheduler per-row sampler blocks. `logit_bias`,
  penalty processors, suppress-token ids, and min-token/EOS policy are also
  covered by standalone native sampler tests. Token-level thinking-budget hard
  close, token-sequence completion repair, and JSON object close-suffix forcing
  now follow the same planning contract: they bind to host row state, block
  native GPU sampling, and
  are reported as speculative/MTP blockers. Explicit `eos_token_id` and
  `ignore_eos=true` are also MTP-only blockers because the current speculative
  accept path records accepted target tokens as unfinished until the decode
  budget is exhausted and does not apply the full AR EOS finish policy. None of
  these processors are compatible with MTP verification yet because verify top-1
  is raw argmax and commit does not apply AR finish policy.
- `hipengine.generation.sampling.supports_speculative_mtp_sampling()` and
  `speculative_mtp_sampling_blockers()` encode that policy. The resident
  scheduler rejects speculative verify work for rows with active blocker fields
  before materializing target verification metadata, including row-local
  logprob / top-logprob requirements that are not token processors but still
  make raw target top-1 insufficient. Successful `SpeculativeVerifyWork` and
  `SpeculativeVerifyPlan` objects carry `target_sampling_policy="raw_target_top1"`,
  `processed_target_verification=false`, and
  `compatible_sampling_modes=("greedy_fast",)` so downstream verifier runners
  cannot mistake the current raw verifier for processed AR sampling parity. The
  capabilities manifest advertises this guard, the blocker fields, and
  condition strings such as
  `temperature > 0`, `eos_token_id set`, `ignore_eos=true`, and
  `logprobs requested` instead of relying only on prose.
- Post-generation validation controls (`response_format`, `guided_json`,
  `guided_regex`, `guided_choice`, `guided_patch`, and `guided_diff`) are not
  MTP blockers as request fields; they add prompt hints and validate visible
  output after generation. Object-root JSON requests that enable host
  close-suffix forcing are represented separately as the
  `json_object_close_forcing` processor blocker. If any remaining validation
  control is promoted to decode-time grammar masks or forced-token repair, that
  promoted path must be represented as an explicit processor blocker until
  target verification applies the same constraint.

Default rule:

- Speculative/MTP routes are allowed only for `GREEDY_FAST` requests with no
  active pre-selection processors and no logprob/metadata requirement that would
  force processed logits.
- If `logit_bias`, penalties, suppressions, forced tokens, sequence-completion
  repair, JSON object close forcing, grammar constraints, thinking budget
  processors, `temperature > 0`, explicit EOS finish policy, or requested
  logprobs are active, or `ignore_eos=true` changes EOS handling, route to AR
  `PROCESSED_ARGMAX` / `HOST_LOGITS_SAMPLE` until the verifier can produce the
  same processed target selection and finish policy per verify row.

Future exact speculative support:

- Target verification must run the same processor stack over each root/candidate
  row before producing target top-1 or sampled-token decisions.
- Logit bias does not need to affect draft proposals for correctness, because
  target verification can reject biased-away candidates, but applying compatible
  hard masks/biases to the proposer is important for acceptance density.
- Hard constraints and forced-token queues must either constrain the proposer or
  reject illegal draft tokens before any KV/session commit.
- Stochastic speculative sampling requires explicit RNG-state ownership and
  probability-ratio semantics; do not treat draft/target argmax equality as
  compatible with sampled requests.

### DFA / constraint engine

Use one tokenizer-aware DFA primitive for stop sequences, forced delimiters,
JSON/tool constraints, and patch grammars.

Required operations:

- lower strings/schemas/grammar states into token-id constraints;
- update state with each accepted token;
- report allowed next token set or sparse token bias;
- detect complete stop/grammar/failure states;
- provide forced-token suffixes when a delimiter/object must be closed.

Start with exact token-sequence matching and JSON-object close-brace fixtures;
do not block P1 on full JSON Schema support.

Current code reality:

- `hipengine.generation.constraints.TokenSequenceDFAState` provides exact
  token-sequence matching and partial-suffix reporting for stop sequences and
  forced delimiter repair.
- `ForcedTokenQueue` provides the decode-through-model FIFO used for thinking
  hard close, tool-call marker/name prefix forcing, and delimiter suffix repair.
- `JsonObjectConstraintState` now provides a tokenizer-agnostic structural state
  for JSON-object outputs: it tracks a root object, string/escape state,
  object/array nesting, trailing content, invalid close delimiters, and the
  deterministic close suffix needed to finish the current object when closing is
  safe. It is covered by fixtures for complete objects, nested object/array
  suffixes, escaped delimiters inside strings, and invalid root/trailing/mismatch
  cases.
- Full JSON/tool/patch grammar constraints are still absent from the public
  server path. The JSON object state is wired into length-finished root-object
  JSON continuation eligibility for JSON-object and JSON Schema result-
  validation requests, so structurally invalid prefixes report
  `schema_violation` and remain non-continuable. It is also wired into host
  decode-time close-suffix forcing for JSON-object requests and object-root
  JSON Schema / guided-JSON requests: when the remaining decode budget exactly
  fits the tokenizer-lowered close suffix, the suffix is queued through
  `ForcedTokenQueue` so it still goes through normal model decode/KV updates.
  Full token masks, schema grammar constraints, and native/MTP parity remain
  future work.

### Session/cache control

Resident-session APIs need explicit commit semantics before long-lived agent use.
The request API should use a `session` object:

```json
{
  "session": {
    "id": "sess_...",
    "commit": "append_visible_only"
  }
}
```

The generation result should tell the session layer what to retain:

```text
append_all          # raw generated tokens, including hidden reasoning
append_visible_only # final visible assistant answer/tool calls only
append_none         # stateless response; keep only reusable prompt/prefix cache
append_prompt_only  # retain admitted prompt/prefix, drop generated tail
```

Visible-only commit may require re-prefilling the visible transcript after
generation. That cost is acceptable compared with retaining hidden reasoning or
malformed partial tool calls in KV.

Default policy:

- stateless requests without `session.id`: `append_none`;
- stateful chat/tool sessions: `append_visible_only`;
- raw transcript/debug sessions must explicitly request `append_all`;
- `length`, `cancelled`, `deadline_exceeded`, malformed tool-call, invalid JSON,
  and synthetic-token finishes downgrade `append_visible_only` to
  `append_prompt_only` unless the request explicitly chose `append_all`;
- synthetic text is never committed unless a future explicit debug option says
  otherwise.

## Reasoning budget / soft-close design

This is the highest-leverage agentic feature: avoid the failure mode where a
thinking model spends the whole budget in `<think>` and returns no useful
answer/tool call.

### Problem

`max_tokens` is currently one undifferentiated pool. For Qwen-style visible
thinking, a response may look like:

```text
<think>
...long chain of thought...
</think>
final answer or <tool_call>{...}</tool_call>
```

If the model spends nearly all generated tokens inside `<think>`, the harness may
receive an empty answer, a partial delimiter, or a malformed tool/JSON block.
Prompt-only `reasoning_effort` hints reduce this risk but cannot guarantee a
visible answer under budget pressure.

### State needed before control

Soft-close depends on token-level state inside generation, not only the server's
post-hoc text splitter:

- current phase: `THINK`, `CLOSING_THINK`, `ANSWER`, `TOOL_CALL`, `DONE`;
- generated token count and remaining hard budget;
- reasoning token count;
- guaranteed visible-answer reserve (`min_answer_tokens`);
- whether the current suffix is a partial close delimiter or stop sequence;
- whether EOS should be suppressed until a visible answer/tool call starts;
- whether the response already contains a valid tool call or structured object.

The existing `_ReasoningSplitter` logic can inform the state machine, but budget
control must run before token selection.

### Policy surface

Candidate request/session controls:

```python
@dataclass(frozen=True)
class ThinkingBudget:
    max_think_tokens: int | None = None          # soft cap before close pressure
    hard_think_cap: int | None = None             # force close at/after this cap
    min_answer_tokens: int | None = None          # reserve visible output budget
    soft_close_window: int = 128                  # ramp over final N think tokens
    close_sequences: tuple[tuple[int, ...], ...] = ()  # tokenized </think> variants
    suppress_eos_until_answer: bool = True
```

Public API options:

- keep current `enable_thinking` / `reasoning_effort` prompt controls;
- add an optional structured `thinking` object for hard controls, e.g.
  `{ "max_tokens": 2048, "min_answer_tokens": 512, "soft_close_window": 128 }`;
- accept explicit budget aliases used by other runtimes where practical:
  `thinking_token_budget`, `thinking.budget_tokens`, and
  `chat_template_kwargs.thinking_budget`;
- expose server defaults so agent harnesses can rely on bounded behavior without
  sending every field manually.

The generation layer should lower model-template strings (`</think>`,
`</think>\n`, etc.) to token sequences using the served tokenizer. Model plugins
or chat-template metadata should provide candidate close strings.

### Reference behavior

vLLM and llama.cpp both treat reasoning-budget enforcement as token-level decode
control, not just prompt text:

- vLLM documents `thinking_token_budget` as a per-request sampling parameter. It
  starts counting at `reasoning_start_str` and, at the budget, forces
  `reasoning_end_str`; that end string may include a transition phrase before
  `</think>` for a more natural close
  ([docs](https://github.com/vllm-project/vllm/blob/470229c37efaf69c86e8bc97482b0b1ff7551c65/docs/features/reasoning_outputs.md#L248-L276)).
- vLLM's `ReasoningConfig` stores `reasoning_start_str` / `reasoning_end_str`
  and tokenizes those strings to IDs using the model tokenizer
  ([source](https://github.com/vllm-project/vllm/blob/470229c37efaf69c86e8bc97482b0b1ff7551c65/vllm/config/reasoning.py#L13-L27),
  [tokenization](https://github.com/vllm-project/vllm/blob/470229c37efaf69c86e8bc97482b0b1ff7551c65/vllm/config/reasoning.py#L71-L107)).
- vLLM's budget state holder tracks rows with `thinking_token_budget`, switches
  to an end-forcing state when the budget is exceeded, and raises the forced end
  token logits to a very large value
  ([state](https://github.com/vllm-project/vllm/blob/470229c37efaf69c86e8bc97482b0b1ff7551c65/vllm/v1/sample/thinking_budget_state.py#L34-L58),
  [budget transition](https://github.com/vllm-project/vllm/blob/470229c37efaf69c86e8bc97482b0b1ff7551c65/vllm/v1/sample/thinking_budget_state.py#L383-L410),
  [forcing](https://github.com/vllm-project/vllm/blob/470229c37efaf69c86e8bc97482b0b1ff7551c65/vllm/v1/sample/thinking_budget_state.py#L470-L526)).
- llama.cpp exposes `--reasoning-budget N` and
  `--reasoning-budget-message MESSAGE`, where the message is injected before the
  end-of-thinking tag when the budget is exhausted
  ([CLI](https://github.com/ggerganov/llama.cpp/blob/961e9a3e46ca4cf7e6e86cfceb5b5e32084bf5f0/common/arg.cpp#L3184-L3198),
  [server README](https://github.com/ggerganov/llama.cpp/blob/961e9a3e46ca4cf7e6e86cfceb5b5e32084bf5f0/tools/server/README.md#L223-L226)).
- llama.cpp's sampler is an explicit state machine:
  `IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE`; in `FORCING` it masks
  all logits except the next forced token. It also exposes a manual
  `common_sampler_reasoning_budget_force(...)` path for mid-generation control
  ([header](https://github.com/ggerganov/llama.cpp/blob/961e9a3e46ca4cf7e6e86cfceb5b5e32084bf5f0/common/reasoning-budget.h#L8-L30),
  [implementation](https://github.com/ggerganov/llama.cpp/blob/961e9a3e46ca4cf7e6e86cfceb5b5e32084bf5f0/common/reasoning-budget.cpp#L39-L49),
  [forcing logits](https://github.com/ggerganov/llama.cpp/blob/961e9a3e46ca4cf7e6e86cfceb5b5e32084bf5f0/common/reasoning-budget.cpp#L143-L160),
  [manual force](https://github.com/ggerganov/llama.cpp/blob/961e9a3e46ca4cf7e6e86cfceb5b5e32084bf5f0/tools/server/server-context.cpp#L2158-L2166)).

hipEngine should follow the same core pattern: tokenize configurable start/end
strings, count generated reasoning tokens, and hard-force a configurable close
sequence through the normal decode path when the budget is reached. The vLLM and
llama.cpp references both support transition text before `</think>`; hipEngine
should support that as an override, but use a conservative tag-only default for
agent harnesses unless the operator opts in.

### Default effort mapping

`reasoning_effort` should map to hard defaults even when the client does not
send explicit budget fields. These are target P1 defaults; the current server
only renders prompt hints.

| Effort | Hard think cap | Soft-close window | Min visible answer reserve | Intended use |
| --- | ---: | ---: | ---: | --- |
| `none` / `off` / `disabled` | 0 | 0 | all generated tokens | Pre-close thinking and answer directly. |
| `minimal` | 256 | 64 | 256 | Tiny arithmetic / one-step tool decisions. |
| `low` | 512 | 128 | 512 | Coding-agent default when quick tool calls are expected. |
| `medium` | 4,096 | 512 | 1,024 | Non-trivial debugging or synthesis. |
| `high` | 16,384 | 1,024 | 2,048 | Deep planning / complex code changes. |
| `xhigh` / `max` | 32,768 | 2,048 | 4,096 | Rare long-horizon reasoning; still bounded. |

Clamp these defaults to the actual generation budget. If the final hard think cap
is `0`, skip the general formula and reserve the full generation budget for
visible answer/tool-call output.

```text
effective_generation_budget =
  min(request.max_tokens or server_chat_default, remaining_context)

effective_min_answer =
  min(table_min_answer, max(0, effective_generation_budget // 2))

effective_think_cap =
  min(table_hard_think_cap, max(0, effective_generation_budget - effective_min_answer))

effective_soft_close_window =
  min(table_soft_close_window, effective_think_cap)

soft_close_starts_at =
  max(0, effective_think_cap - effective_soft_close_window)
```

Manual request fields override the table before clamping. `xhigh` / `max` is not
unbounded; users who want unbounded thinking must explicitly disable the hard
budget or set a larger `thinking.max_tokens` with enough `max_tokens` / context.

Precedence and disabling rules:

1. Server defaults are lowest precedence.
2. `chat_template_kwargs` aliases are compatibility hints.
3. Top-level convenience fields (`enable_thinking`, `reasoning_effort`,
   `thinking_token_budget`) override template aliases.
4. Nested `thinking` / `reasoning` objects override top-level convenience fields.
5. In the final merged control object, `enabled=false`, `type=none/off/disabled`,
   or an effort value of `none/off/disabled` wins over a non-disabling effort.
6. `allow_unbounded=false` by default. A request may disable the hard think cap
   only when it explicitly sets `thinking.allow_unbounded=true`; even then,
   `min_answer_tokens` remains active unless the request explicitly sets it to
   `0`.

Current server behavior is weaker than the final token-level target but stronger
than a plain tone hint: it accepts these control surfaces to render
prompt/template hints, pre-close Qwen thinking, and host-sampler hard-close
state when tokenization is available. `reasoning_effort`
`minimal`/`low`/`medium`/`high`/`xhigh`/`max` maps to the default hard-cap,
answer-reserve, and soft-close-window table above, clamped to request
`max_tokens`, the configured chat default when bounded, and remaining admitted
context when tokenizer/context metadata is available. Numeric
`thinking_token_budget`, `chat_template_kwargs.thinking_budget`,
`thinking.budget_tokens`, `thinking.max_tokens`, `reasoning.budget_tokens`, and
`reasoning.max_tokens` aliases normalize to the effective hard-cap hint; string
`thinking_budget`/`budget_tokens` values still act as effort aliases for
compatibility. Explicit `max_think_tokens`, `min_answer_tokens`,
`hard_think_cap`, `soft_close_window`, `hard_close_message`, and
`hard_close_sequence` fields are accepted at top level and under nested
`thinking` / `reasoning` objects; nested fields override top-level convenience
fields. Numeric hard/min/soft hints are clamped to the same bounded generation
budget, and `hard_close_sequence` is rejected unless it contains `</think>`. For chat
requests with an effective `hard_think_cap`, the server tokenizes
`hard_close_sequence` or the default `</think>` marker and passes
`thinking_close_token_ids`, `thinking_hard_token_cap`, and
`thinking_soft_close_window` through `SamplingParams`, `GenerationRequest`, and
`PerRowSamplingParams`. Host sampling binds those fields to a fresh
`ThinkingBudgetState` per row and forces the close sequence when the hard cap is
reached. If tokenization is unavailable, normal chat generation stays
prompt-hint-only instead of failing.
Nested `thinking.allow_unbounded=true` or `reasoning.allow_unbounded=true`
disables effort-derived default hard-cap injection when no explicit hard cap is
present; explicit budget/cap fields still set the hard cap and lower into
sampler params as usual. Disabling signals (`enabled=false`,
`type=none/off/disabled`, or disabling effort aliases) win over non-disabling
signals.

### Hard-close sequence and overrides

The default hard close should be deterministic and minimal:

```text
</think>
```

This minimizes extra hidden tokens and avoids adding model-visible prose that
could leak into unusual templates. Operators may opt into a transition phrase for
models that close more cleanly with natural language, matching the vLLM and
llama.cpp pattern:

```text
I have reached the reasoning budget and will answer now.</think>\n
```

Expose these knobs:

```json
{
  "thinking": {
    "enabled": true,
    "effort": "medium",
    "max_tokens": 4096,
    "min_answer_tokens": 1024,
    "soft_close_window": 512,
    "hard_close": "tag_only",
    "hard_close_message": null,
    "hard_close_sequence": null,
    "allow_unbounded": false
  }
}
```

Rules:

- `hard_close="tag_only"` forces the tokenizer-lowered close tag sequence only.
- `hard_close="message_then_tag"` forces `hard_close_message + close_tag`.
- `hard_close_sequence` is an expert override for the exact string to tokenize
  and force; it must contain the parser-recognized close marker.
- If a parser has an intrinsic close marker, validate that the configured hard
  close sequence contains it; otherwise the server must reject the request.
- If forced text is emitted outside model decoding for any fallback path, mark it
  as `synthetic_tokens` and do not commit it silently to resident session state.

### Close target

Prefer closing the reasoning delimiter, not forcing answer-start prose:

1. Bias the first token(s) of accepted close sequences such as `</think>`.
2. Once the model begins a close sequence, force the remaining tokens through the
   normal decode path.
3. Switch phase accounting from reasoning to answer/tool-call after the delimiter
   completes.

A rejected alternative is boosting `response`/answer-start tokens directly, but
that is template-specific and risks weird partial transitions. Closing `</think>`
is more robust and keeps the model in control of the final answer.

### Logit-processor mechanics

Implement thinking control as one policy in the general processor stack. The
pre-selection half is:

1. Start from FP32 logits.
2. Apply static OpenAI-style `logit_bias`.
3. Apply penalties and suppress-token constraints.
4. Apply min-token / suppress-EOS policy when no visible answer exists yet.
5. Apply dynamic budget processors, including thinking soft-close.
6. Apply grammar/tool masks or sparse biases.
7. Apply forced-token queue if a delimiter/grammar sequence is already in
   progress.
8. Run argmax or sampling.

After accepting the token, update reasoning phase, answer reserve accounting,
history penalties, stop/grammar DFA state, and telemetry.

For soft-close, compute a ramp from remaining think/answer budget. As the soft
window is consumed, add a sparse positive bias to viable first tokens of
`close_sequences`; at the hard cap, force the close sequence. If a sequence is
partially matched, the forced-token queue emits the rest so KV state stays
consistent.

### Graph-capture implications

Dynamic budget bias changes per decode step, so it does not fit a single static
multi-token graph replay without extra machinery. Practical options:

- **A: graph bulk + host-stepped tail.** Use the greedy graph path until the
  soft-close window, then switch to a host-stepped logits/processor path for the
  final ~64-256 tokens. This is the simplest first implementation and the tail
  cost is acceptable for harness reliability.
- **B: graph variants.** Capture separate graphs with and without a constant
  close bias. Lower overhead, less flexible.
- **C: device-side budget counter.** Add a device-visible step/budget counter and
  processor kernel that updates close-token bias. Most elegant, most kernel work.

P1 should start with option A and document the performance tradeoff; only promote
native/GPU processors after correctness and benchmark gates pass.

### Graceful exhaustion contract

If the request still exhausts its hard limit:

- Best case: force `</think>` through the normal decode path, mark
  `forced_close=true`, and continue for the reserved answer budget.
- If no answer budget remains, return `finish_reason="length"` plus
  `finish_details.reason="thinking_budget_exhausted"` and a continuation handle
  when possible.
- If text is appended outside model decoding, mark it as synthetic and do not
  silently commit it to resident session state.

Example finish metadata:

```json
{
  "finish_reason": "length",
  "finish_details": {
    "reason": "thinking_budget_exhausted",
    "forced_close": true,
    "synthetic_tokens": 0,
    "reasoning_tokens": 2048,
    "answer_tokens": 128,
    "budget_pressure": "hard_cap"
  },
  "continuation_id": "gen_abc123"
}
```

## Implementation punchlist

Each item below should be implemented as a self-contained logical unit with
focused tests, docs/API updates when public behavior changes, and a WORKLOG entry.
The `Implementation notes` are intentionally concrete enough for another agent to
start coding without re-deriving the design.

### P0 — Decode observability and robust finish semantics

These are the foundation for every agent-friendly feature below.

#### P0.1 Canonical `DecodeState` / `GenerationTelemetry`

Implement:

- add generation-owned per-row phase/token accounting;
- expose a snapshot on `GenerationOutput` and streaming chunks;
- keep a compatibility bridge from server `_ReasoningSplitter` until generation
  phase accounting is authoritative;
- record `sampler_mode`, fast-path vs host-logits fallback, stop suffix state,
  forced-token queue state, and continuation eligibility.

Likely touchpoints:

- `hipengine/generation/registry.py` for result dataclasses;
- `hipengine/generation/sampling.py` for sampler mode metadata;
- `hipengine/generation/qwen35_paro.py` / `qwen35_gguf.py` for output plumbing;
- `hipengine/server/api.py` for response/stream serialization.

Current code reality:

- `hipengine.generation.registry` now defines torch-free `DecodePhase`,
  `DecodeState`, `GenerationTelemetry`, and `GenerationStreamChunk` primitives;
  `GenerationOutput` and detailed stream chunks can carry optional telemetry
  snapshots.
- Opt-in server stream metadata uses `DecodeState` for token-bearing
  `choices[].hipengine.decode_state` payloads derived from the current
  `_ReasoningSplitter` / token-counting compatibility layer. When a backend
  stream yields `GenerationStreamChunk.telemetry`, the SSE choice-level
  `hipengine.decode_state` preserves that backend-authored snapshot and only
  layers server-derived stream token counters beside it.
  Live `GenerationStreamChunk.finish_details` is also preserved on final choice
  chunks when no server-side stop/tool/structured override changes the result.
- Token-emitting PARO/GGUF generation loops now author final
  `GenerationTelemetry` snapshots with prompt/generated token counts, row index,
  sampler mode, stop suffix match/partial-suffix state where applicable, and
  sampled thinking-budget phase, reasoning/answer counts, budget pressure,
  pending forced-token state when row state is available, and sampler fallback
  reasons for processed-argmax / host-logits paths. PARO/GGUF host-logits
  sampled paths also mark `full_vocab_logits_d2h=true` with per-token vector
  byte counts when the vocabulary/logits width is known. PARO c=1 native
  sampled paths mark `full_vocab_logits_d2h=false` and `logits_d2h_bytes=0` in
  the decode-state snapshot when the native GPU sampler route actually runs.
  PARO c>N scheduler-owned final snapshots also carry `execution_path`,
  `native_compact_prefill`, `native_caware_decode`, and
  `serial_decode_fallback`, plus `native_sampler_rows` for sampled batches, so
  harnesses can tell whether the row used native packed prefill, c-aware native
  decode, the serial decode bridge, or serial per-slot native GPU sampling.
  PARO/GGUF host-sampled forced-token selections also expose the selected
  `forced_token_id`, `forced_token_reason`, and post-selection
  `forced_tokens_remaining` alongside pending generic forced-token queues,
  post-thinking forced-token queues, and sequence-completion repair policy when
  those controller states are active.
- Non-streaming OpenAI-compatible completion/chat choices now expose backend
  `GenerationTelemetry` under `choices[].hipengine` when it is present,
  mirroring the final `finish_details` alongside the backend-authored
  `decode_state` plus optional backend-authored `timing` and `usage` payloads.
  If the server creates a deterministic buffered continuation handle, the final
  choice telemetry mirrors `finish_details.continuation_eligible=true` into the
  exposed `decode_state` so clients do not see contradictory eligibility
  metadata.
- PARO and GGUF c=1 true streaming emit live per-token
  `GenerationStreamChunk` telemetry for greedy and sampled answer tokens,
  including host-sampled budget-pressure metadata when row state is available.
  The final live chunk also carries backend-authored `finish_details` for
  EOS/token-stop/sequence-stop/length plus sampler mode, so SSE final choices
  can preserve backend finish reasons when server post-processing does not
  override them.
  PARO c=1 native-GPU sampled streams also mark `sampler_mode="gpu_sample"`
  with `full_vocab_logits_d2h=false` / `logits_d2h_bytes=0`, matching the
  non-streaming native route diagnostics.
  The scheduler submit/poll wrapper preserves an inner generator's
  `stream_detailed()` chunks instead of downgrading them to plain text, so
  wrapped `LLM.stream_detailed()` / server streaming paths keep backend-authored
  telemetry when the underlying generator provides it. If a wrapped backend only
  exposes `generate_detailed()`, the adapter now bridges those detailed final
  outputs into `GenerationStreamChunk` values before falling back to plain text,
  preserving finish details and final decode-state telemetry through buffered
  stream paths.
  `ResidentBatchScheduler.record_generated_events()` records generated token ids
  while returning scheduler-authored token events with `GenerationStreamChunk`
  telemetry, and `ResidentEngineLoop.poll()` attaches those stream chunks to
  token events. The chunks are telemetry carriers only until a native runner can
  detokenize or otherwise provide the text surface.
  Buffered streaming preserves backend final telemetry on choice `done` chunks
  and now emits server-derived token/decode-state snapshots for opt-in buffered
  answer/reasoning/tool/structured deltas when tokenizer counting is available.
  Those buffered deltas inherit stable backend sampler/execution metadata from
  the final telemetry, but keep token-specific fields such as forced-token state,
  stop suffixes, budget pressure, timing, and usage on final/backend-authored
  snapshots, scheduler token-event chunks, or PARO/GGUF c>N engine/wrapped-
  generator `last_batch_generation.scheduler_token_chunks` diagnostics.
  Buffered `/v1/completions` streams, plain answer/reasoning buffered
  `/v1/chat/completions` streams, plain chat content logprob streams, live and
  buffered reasoning streams with hipEngine-private logprob metadata, validated
  structured chat content streams, and validated tool-call argument spans now
  use PARO c>N and GGUF serial c>N scheduler token chunks as
  per-token public deltas for a single HTTP request when the chunks exactly
  reconstruct the final choice text and logprob chunks can be mapped to emitted
  content/reasoning deltas. Live parser-final splitter leftovers also reuse
  retained source-chunk token metadata when the held public delta is mappable.
  Engines that explicitly advertise `supports_stream_many` and implement
  `stream_many_detailed` can also feed runtime-native live c>N chat chunks
  directly to public SSE deltas for the plain no-tools/no-structured/no-logprob
  request subset, provided every chunk carries
  `GenerationTelemetry.decode_state.row_index`. The server keeps the existing
  buffered parser/validation paths for tool, structured-output, logprob, stop,
  continuation, and unsupported-engine cases.
  Coalesced
  multi-request batches, invalid or
  unmappable tool outputs, structured-output validation failures, canonical
  tool/structured phases, unmappable parser-final logprob spans, and real
  continuation eligibility remain future lower-loop work.

Exit gates:

- fake-session tests prove streaming and non-streaming paths report the same
  phase/token counts;
- greedy-equivalent output text and token ids remain unchanged;
- no torch import is introduced on the hot path.

#### P0.2 Structured finish details

Implement:

- add `FinishDetails` or equivalent to generation outputs;
- map internal reasons to OpenAI-compatible `finish_reason` while preserving
  detailed extension metadata;
- include EOS, stop sequence, length, cancellation, deadline, forced close,
  synthetic tokens, budget pressure, cache action, and sampler mode.

Current code reality:

- `hipengine.generation.FinishDetails` is the canonical torch-free structured
  finish object, and `GenerationOutput` normalizes either a `FinishDetails`
  instance or a mapping into that shape.
- `FinishDetails.to_json_dict()` emits compact JSON only for active fields while
  preserving the full AGENTIC surface: EOS token id, stop token sequence, length
  limit, deadline/cancel flags, forced close, synthetic token count,
  reasoning/answer/tool/structured token counts, budget pressure, cache action,
  sampler mode, phase, and continuation eligibility.
- The OpenAI server keeps coarse compatibility (`eos` -> public `stop`,
  `length` -> public `length`, parsed tool calls -> public `tool_calls`) while
  preserving backend details under `choices[].finish_details`,
  final SSE choice chunks, and structured error payloads.
- Unit and server tests cover structured finish-detail normalization,
  completion/chat coarse mapping, EOS, token and sequence stops, length phases,
  tool-call/malformed-tool/schema failures, thinking-budget exhaustion,
  deadline/cancellation errors, sampler mode, budget pressure, and session cache
  action reporting.

Exit gates:

- server tests cover EOS, token stop, stop sequence, length, tool call,
  malformed tool call, budget exhaustion, cancellation, and deadline;
- old clients still see the same coarse `finish_reason` strings;
- errors are stable enough for harness retry logic.

#### P0.3 Streaming metadata deltas

Implement:

- add opt-in streaming metadata via
  `stream_options: {"include_hipengine": true}`;
- put request-level timing/cache metadata in a top-level `hipengine` object on
  SSE payloads, and per-choice phase/token metadata in `choices[].hipengine`;
- stream phase transitions (`think`, `closing_think`, `answer`, `tool_call`,
  `structured`, `done`);
- include TTFT, prefill ms, decode tok/s, cache hit/miss, KV bytes, stop reason,
  and budget-pressure state when available.

Current code reality:

- `stream_options: {"include_hipengine": true}` is accepted for completion and
  chat streams without changing default OpenAI-compatible SSE payloads.
- Opt-in SSE payloads include top-level `hipengine.metadata_version`,
  `hipengine.event`, and `hipengine.timing.elapsed_ms`. Token-bearing chunks
  add server-measured `ttft_ms` once the first generated chunk is emitted; final
  done/usage chunks also include server-measured `decode_elapsed_ms` and
  `decode_tokens_per_second` when generated-token counts are available. Choice
  chunks include `choices[].hipengine.phase` for answer/reasoning/tool/done
  chunks, and buffered structured-output result-validation streams report a
  final `structured` choice phase. Top-level opt-in SSE metadata also includes
  `hipengine.routing` for the current single-model exact route.
- When tokenizer/counting hooks are available, live and buffered
  completion/chat deltas also include `choices[].hipengine.tokens` with
  per-chunk `delta_tokens`, cumulative `streamed_tokens`, and best-effort
  server-side phase counters. Buffered tool/reasoning/structured deltas count
  the parsed chunks the server emits, not lower-loop decode-time grammar state.
  Final choice chunks include usage-derived prompt/completion/total token counts
  plus those streamed phase counters. Token-bearing chunks also include a
  canonical `choices[].hipengine.decode_state` snapshot unless backend
  telemetry supplies an authoritative decode-state snapshot. Buffered deltas that
  have final backend telemetry inherit safe backend sampler/execution fields in
  that decode-state snapshot while retaining server-derived phase and token
  counts.
- Final choice chunks mirror `finish_details` under `choices[].hipengine`, and
  usage chunks mirror `usage` under top-level `hipengine.usage`. Live backend
  `GenerationStreamChunk.finish_details` can seed those final choice details
  when server-side post-processing does not override them.
- Backend-authored `GenerationTelemetry.timing` remains available under
  `choices[].hipengine.timing` and is mirrored into top-level
  `hipengine.timing` with `backend_` prefixes when a live stream chunk or final
  buffered response provides it.
- Final done/usage chunks include top-level `hipengine.kv_pool` with sanitized
  server-observed KV pool stats when the served engine exposes them.
- `/v1/hipengine/capabilities` advertises the stream metadata scopes separately:
  token-accounting/decode-state scopes cover `live_delta`, `buffered_delta`, and
  `final_choice` when tokenizer counting is available, while backend telemetry
  scopes cover `live_chunk`, `buffered_delta_safe_decode_state`, and
  `buffered_done` when generation telemetry is emitted. It also reports
  `features.stream_metadata.buffered_scheduler_chunks` so clients can see which
  buffered c>N surfaces may replay engine or wrapped-generator
  `last_batch_generation.scheduler_token_chunks` and which conditions force
  conservative buffering. Invalid or unmappable buffered tool-call scheduler
  chunks and unmappable scheduler logprob chunks remain withheld from public
  deltas, but final done choices include sanitized
  `choices[].hipengine.withheld_scheduler_tool_chunks` or
  `choices[].hipengine.withheld_scheduler_logprob_chunks` diagnostics when
  clients opt in with `stream_options.include_hipengine=true`.
- Streaming error chunks also honor `include_hipengine`: they use top-level
  `hipengine.event="error"` and mirror structured finish details under
  `choices[].hipengine.finish_details` when those details are available.
- Cache hit/miss, per-request KV-byte deltas, budget-pressure, and
  authoritative per-phase token counts are still omitted until
  generation/runtime code emits those signals.

Exit gates:

- `stream_options.include_usage` remains OpenAI-compatible;
- plain clients can ignore metadata without breaking content streaming;
- tests verify final accumulated deltas match non-streaming usage/telemetry.

#### P0.4 Token diagnostics endpoints

Implement:

- `POST /v1/hipengine/tokenize` and `POST /v1/hipengine/detokenize` using the
  served tokenizer;
- `POST /v1/hipengine/count_tokens` for raw text and chat messages after server
  template rendering;
- `POST /v1/hipengine/fit_context` that returns prompt tokens, admitted context,
  effective `max_tokens`, chat default, truncation/clear policy, and what would
  be dropped.

Current code reality:

- `POST /v1/hipengine/tokenize`, `/detokenize`, `/count_tokens`, and
  `/fit_context` are implemented and authenticated like the other `/v1/*`
  endpoints.
- Raw text diagnostics use the served tokenizer/counting hooks. Chat diagnostics
  render the same Qwen-style prompt as generation, including tool markup and
  thinking controls, before counting. When a chat diagnostic request includes
  `session.id`, `count_tokens` prepends the stored app-local transcript exactly
  as stored. `fit_context` shares generation's overflow-policy decision: the
  default is `reject`, `session.context_overflow_policy="auto_clear_transient"`
  reports that there are currently no transient stored segments to clear,
  `new_session` drops the stored prefix only when prefix+request overflows and
  the current request alone fits, and `truncate_oldest_visible` drops the
  shortest oldest prefix whose retained suffix remains a valid rendered
  transcript and fits. The responses report session prefix/request/rendered
  message counts plus
  `resident_state_reuse=false`. Chat count/fit diagnostics also expose lowered
  thinking-budget close tokens, the initialized
  `ThinkingBudgetState` payload, and `allow_unbounded=true` when that merged
  control is active and tokenization is available; capability metadata
  advertises that diagnostic lowering/state support separately from live
  token-budget enforcement.
- `/fit_context` uses the same context arithmetic and chat default max-token
  policy as generation admission, reports the current clear policy, and includes
  sanitized kept/dropped/reset segment metadata for the selected policy.
- Endpoints return explicit unsupported-feature errors when the served model does
  not expose tokenizer/counting/decoding hooks.

Exit gates:

- `/fit_context` and actual generation use the same token accounting;
- tests cover raw prompts, chat messages, tool messages, and thinking controls;
- capability metadata advertises these endpoints.

#### P0.5 Request cancellation and deadlines

Implement:

- request-level cancellation token checked by scheduler/decode loops;
- `timeout_ms` as the public relative-deadline field plus a server default
  `--request-timeout-ms`; internally lower to a monotonic deadline timestamp;
- cleanup that releases active rows/session reservations on cancel/deadline;
- finish details with `cancelled=true` or `deadline_exceeded=true`.

Current code reality:

- `CompletionRequest` and `ChatCompletionRequest` accept `timeout_ms`.
- `ServerConfig.request_timeout_ms`, `hipengine serve --request-timeout-ms`,
  and `HIPENGINE_REQUEST_TIMEOUT_MS` provide a server-wide default. A request
  field overrides the server default.
- Server code lowers the relative timeout to a `time.perf_counter()` deadline
  and applies it to preparation, queued/buffered generation, token-stream
  iteration, and buffered `n>1` streaming.
- The lowered deadline is carried through `SamplingParams.deadline_at` and
  `GenerationRequest.deadline_at`. `GenerationDeadlineExceeded` carries
  `FinishDetails(reason="deadline_exceeded", deadline_exceeded=true)`, and the
  OpenAI server maps either that exception or a backend-authored deadline finish
  detail to the same HTTP 408 / SSE error contract.
- Buffered deadline expiry returns HTTP 408 with OpenAI-style
  `error.type="timeout_error"`, `error.code="deadline_exceeded"`, and
  `error.finish_details.reason="deadline_exceeded"`.
- Live SSE streams cannot change the already-sent HTTP status; timeout expiry
  emits a final SSE error payload whose choice and error object both include
  deadline finish details, then emits `data: [DONE]`.
- A request-control object checks the FastAPI request disconnect signal at the
  same await/iteration boundaries. Detected disconnects map to structured
  `cancelled` finish details and HTTP-style status 499 when the transport can
  still surface an error payload.
- `GenerationCancellationToken` is carried through `SamplingParams` and
  `GenerationRequest`. Server-side deadline/disconnect detection cancels the
  token, `GenerationCancelled` carries
  `FinishDetails(reason="cancelled", cancelled=true)`, and the OpenAI server
  maps backend-observed cancellations to the same 499 / SSE error contract.
- `_GenerationBatcher` marks abandoned stream items cancelled and skips queued
  futures that were cancelled before dispatch. Requests with different absolute
  deadlines or cancellation tokens do not coalesce into the same batcher engine
  call.
- `ResidentBatchScheduler` has a unified pending/active reclaim path for
  `cancel`, `disconnect`, and `timeout` exits. Active rows are removed from the
  batch, scheduler-owned sampler state is dropped, and the reclaim callback runs
  so KV reservations can be released while surviving rows continue decoding.
  Completed scheduler rows and per-request observability now carry structured
  `FinishDetails`: `cancel`/`disconnect` map to
  `{"reason":"cancelled","cancelled":true}`, `timeout` maps to
  `{"reason":"deadline_exceeded","deadline_exceeded":true}`, and length exits
  include the scheduler row's decode limit.
- PARO and GGUF resident generation paths check deadlines before and after
  tokenization/prefill/decode calls and check cancellation tokens at the same
  cooperative boundaries, including host-sampled and scheduler-owned PARO c>N
  loops. Captured graph replay and individual GPU kernels are not preempted
  mid-call; the check happens immediately before and after the replay or
  kernel-backed step.

Remaining implementation:

- true mid-kernel or mid-graph preemption remains future work; cancellation and
  deadline checks still happen at cooperative boundaries.

Exit gates:

- deadline tests cover HTTP 408 errors, streaming error+done, and follow-up
  server reuse;
- cancellation tests cover queued work cleanup and request-control disconnect
  mapping;
- active row/session leak tests cover lower decode loops and KV reservation
  release;
- streaming cancellation emits a final error/done event instead of hanging.

### P1 — Controlled decoding and thinking budgets

Build this on top of P0 telemetry rather than as a Qwen-only special case.

#### P1.1 General logit-processor framework

Current code reality:

- `hipengine.generation.sampling.plan_sampler()` reports
  `active_processors`, `fast_path_blockers`, and `fallback_reason`, so callers
  can distinguish token-selection processors from fields such as `temperature`
  / requested logprobs that also leave the graph argmax fast path.
- Scheduler-owned `SamplerParamsBlock` rows expose the same planner decision
  through `sampler_plan_for()`, `sampler_plans()`, and JSON-ready
  `sampler_plan_metadata()`, so c>N/native scheduler callers can inspect
  per-row processor blockers, logprob metadata requirements, and
  native-fallback reasons without duplicating sampling policy.
- Host `select_token()` returns the same processor/blocker metadata on
  `SampleResult`; row-local forced-token queues are included even when they were
  not part of request-level params.
- `DecodeState` serializes `active_processors` and
  `sampler_fast_path_blockers` plus `sampler_fallback_reason`, and PARO/GGUF
  final telemetry snapshots attach those fields for sampled / processed
  requests. PARO scheduler-owned c>N final snapshots also expose scheduler
  execution path and native/serial fallback state. PARO scheduler-owned c>N
  token chunks now carry per-token planner metadata, fallback reason,
  host-logits D2H accounting, and execution flags for buffered server replay;
  broader runtime-native live c>N parity for logprobs/tool/structured surfaces,
  GGUF/native GPU sampler parity, and phase/logprob semantics for
  reasoning/tool/structured chunks still need lower-loop coverage before server
  streams can become fully
  backend-authoritative.
- Suppress-token ids and min-token/EOS policy are implemented as pre-selection
  processors after static bias/history penalties and before the forced-token
  override. They are exposed through public/server request fields, scheduler
  per-row blocks, planner metadata, fast-path blockers, capabilities, MTP
  blocker lists, and the scoped PARO native GPU processor path. PARO and GGUF
  c=1 sampled streaming now emit the planner's processor/blocker fields and
  fallback reason on live chunks. True
  runtime-native c>N/scheduler streams, GGUF/native GPU sampler paths, dynamic
  thinking-budget processors beyond the host sampled stream path, and grammar
  masks still need the same lower-loop stream metadata before server streams can
  become fully backend-authoritative.

Implement:

- a pure processor plan that decides `GREEDY_FAST`, `PROCESSED_ARGMAX`,
  `HOST_LOGITS_SAMPLE`, or future `GPU_SAMPLE`;
- ordered processor execution for static bias, penalties, suppressions,
  min-token/EOS policy, stop DFA, forced tokens, dynamic budgets, and grammars;
- metadata explaining which processors were active and why fast path was left.

Exit gates:

- processed-argmax fixtures prove deterministic tie-breaking;
- greedy-equivalent requests still use graph/argmax fast path;
- host sampler tests cover processor ordering;
- speculative/MTP routes are disabled or clearly rejected for active processors
  until target verification applies the same processed token-selection policy.

#### P1.2 Forced-token queue

Current code reality:

- `hipengine.generation.constraints.ForcedTokenQueue` is a torch-free FIFO queue
  primitive for model-decoded forced tokens.
- Host `select_token()` consumes one pending forced token before argmax or
  sampling, records forced metadata on `SampleResult`, and updates row history
  so the token still goes through the normal decode/KV path.
- PARO/GGUF host-sampled telemetry carries selected forced-token metadata in
  `DecodeState` as `forced_token_id`, `forced_token_reason`, and
  `forced_tokens_remaining`; `forced_tokens_pending`,
  `post_thinking_forced_tokens_pending`, and
  `force_sequence_completion_token_sequences` remain pending/controller policy
  snapshots.
- `RowSamplingState` can now bind a `ThinkingBudgetState`; before each host
  token selection, hard budget pressure queues the tokenizer-lowered close
  sequence as forced tokens, and every selected token updates the
  reasoning/answer phase counters.
- Pending forced tokens participate in sampler planning as
  `forced_tokens_pending`, are rejected from the current native GPU sampler
  route, dynamically fall back to host token selection if they appear while a
  row is configured for native sampling, and block raw-argmax MTP verification.
- `RowSamplingState` can also bind tokenizer-aware sequence-completion repair
  rules. Once a configured delimiter prefix is selected, the remaining delimiter
  suffix is queued as forced tokens. The server uses this today for selected
  tool-name prefix completion and required/specific tool-call `</tool_call>`
  close repair, plus JSON-object structural close-suffix forcing for object-root
  structured-output requests when the remaining budget exactly fits the lowered
  suffix. Full grammar processors remain future server/controller work.

Implement:

- per-row forced-token queue integrated before argmax/sampling;
- queue population from close delimiters, stop/grammar repair, and future JSON
  close-brace enforcement;
- telemetry for forced token count and reason.

Exit gates:

- forced multi-token delimiters are emitted exactly once;
- forced tokens go through normal decode so KV state stays consistent;
- finish details distinguish forced-through-model tokens from synthetic text.

#### P1.3 Stop DFA promotion

Current host PARO/GGUF suffix matching exists; make it a first-class decode
primitive.

Current code reality:

- `hipengine.generation.constraints.TokenSequenceDFAState` is a torch-free
  tokenizer-agnostic DFA primitive for exact token-sequence matches and partial
  suffix reporting.
- PARO/GGUF stop checks and final decode telemetry now use this shared helper
  for multi-token stop match state.
- Scheduler-owned `RowSamplingState` now maintains the same ordinary stop
  sequence DFA and emits partial/matched suffix state on per-token
  `GenerationStreamChunk` telemetry.
- PARO scheduler-owned c>N processed rows are covered by a regression that
  proves per-row stop handling across packed prefill and decode: one row can
  stop immediately on a stop token id while another continues until an
  overlapping multi-token stop sequence completes, with stop details preserved
  in final `FinishDetails` and decode telemetry.
- PARO serial per-slot native c>N sampling consumes the same post-selection stop
  metadata as c=1; true batched native c>N sampling, GGUF GPU sampling, and
  grammar reuse remain future work. Host row state already reuses this DFA for
  forced delimiter suffix repair.

Implement:

- keep server post-trimming for response consistency;
- keep true batched native c>N and GGUF GPU sampler paths on the same stop
  metadata contract before they claim token-level stop parity.

Exit gates:

- one-token and overlapping multi-token stop fixtures pass for PARO and GGUF
  host paths;
- true batched native c>N/GGUF GPU paths either pass the same fixtures or report
  unsupported path clearly;
- stop details include the matched token sequence.

#### P1.4 Thinking budget policy

Implement:

- `reasoning_effort` defaults from the table above, with budget/context clamping;
- explicit override fields for `max_think_tokens`, `min_answer_tokens`,
  `hard_think_cap`, `soft_close_window`, `hard_close_message`, and
  `hard_close_sequence`;
- compatibility aliases: `thinking_token_budget`, `thinking.budget_tokens`, and
  `chat_template_kwargs.thinking_budget`;
- tokenizer lowering for accepted close strings and transition-message strings;
- parser validation that any configured hard close contains the parser-recognized
  end marker;
- soft logit-bias ramp near the budget boundary;
- hard forced close at cap;
- manual hard-stop/force hook for cancellation, external controller requests, or
  stream-time budget pressure;
- EOS suppression until answer/tool-call starts when configured.

Current code reality:

- accepted and tested: explicit budget fields, compatibility aliases, numeric
  alias normalization, effort-to-default budget hints clamped to request
  `max_tokens`, the bounded chat default, and remaining admitted context,
  prompt-hint rendering, and parser-marker validation for
  `hard_close_sequence`;
- `hipengine.generation.constraints.ThinkingBudgetState` is a torch-free
  token-id primitive that tracks reasoning/answer counts, reports soft/hard
  budget pressure, enqueues a tokenizer-lowered close sequence through
  `ForcedTokenQueue`, and transitions to answer phase when the close DFA
  matches;
- host `RowSamplingState` / `select_token()` can enforce a supplied
  `ThinkingBudgetState` hard cap by forcing the close sequence before ordinary
  argmax/sampling, with forced-token metadata, normal row-history updates, and
  final `FinishDetails` fields for forced close, token counts, budget pressure,
  and phase;
- final PARO/GGUF `GenerationTelemetry.decode_state` snapshots now inherit
  sampled thinking-budget phase, reasoning/answer token counts, budget pressure,
  pending forced-token state from `RowSamplingState` where available, and
  selected forced-token metadata when a host sampler result consumed one;
- PARO/GGUF c=1 host-sampled `stream_detailed()` chunks also expose live
  thinking-budget phase, reasoning-token, budget-pressure, processor, fallback,
  selected forced-token, and logits-readback metadata on the chunk
  `GenerationTelemetry`;
- `SamplingParams`, `GenerationRequest`, and `PerRowSamplingParams` carry the
  lowered `thinking_close_token_ids`, `thinking_hard_token_cap`, and
  `thinking_soft_close_window` fields; PARO/GGUF host-sampled rows and
  resident scheduler rows bind those fields into per-row
  `ThinkingBudgetState` instances;
- chat completions lower effective hard thinking caps into sampler fields when
  the close string can be tokenized; if tokenization is unsupported, generation
  remains prompt-hint-only rather than failing the request;
- host sampling applies a ramped sparse soft-close bias to the first close token
  inside the configured soft window. If that token is accepted, the remaining
  close suffix is queued as forced tokens so the close delimiter is emitted
  through the normal decode/KV path;
- host sampling suppresses EOS while a tokenized thinking budget is still in
  the reasoning or close-delimiter phase. Qwen PARO/GGUF host-sampled paths
  resolve tokenizer EOS into the sampler when the request did not provide
  `eos_token_id`;
- `ThinkingBudgetState.force_close(reason=...)` is the controller-facing
  primitive for manual hard-stop/force requests. It queues the full lowered
  close sequence with caller-supplied reason metadata, emits the close tokens
  through the same normal decode/KV path as hard-cap and soft-close forcing,
  and refuses to queue after answer/done phase or while another forced sequence
  is pending;
- if host-sampled hard-close enforcement consumes the entire generation budget
  before any visible answer token is emitted, PARO/GGUF finish metadata reports
  `finish_details.reason="thinking_budget_exhausted"` with
  `forced_close=true`; the OpenAI-compatible server still reports coarse
  `finish_reason="length"` and keeps active thinking-budget continuations
  ineligible;
- chat token diagnostics can lower the configured hard close sequence (or the
  default `</think>` marker) into token ids and return an initialized
  `ThinkingBudgetState` payload for harness/debug verification;
- sampler planning treats an active thinking budget as a fast-path blocker:
  host sampling is used, native GPU sampling falls back, and raw-argmax
  speculative/MTP verification is rejected until those paths implement the same
  soft-close, EOS-suppression, and hard-close semantics;
- not implemented: native GPU thinking-budget enforcement, speculative/MTP
  processed-target verification, public HTTP external-controller close requests,
  and live backend-authored per-token phase metadata.

Exit gates:

- default effort fixtures verify low/medium/high/xhigh map to the documented caps
  after clamping to `max_tokens` and remaining context;
- deterministic fake-logit fixtures show soft-close bias changes token choice;
- hard-cap fixtures force the full close sequence, including optional transition
  message + `</think>` when configured;
- requests whose override close sequence omits the parser close marker are
  rejected clearly;
- a reasoning-heavy fixture returns either visible answer/tool-call tokens or
  `thinking_budget_exhausted` with `forced_close=true`;
- graph fast path is preserved outside controlled tail windows.

#### P1.5 Graceful length exhaustion

Implement:

- detect length exhaustion by phase: in reasoning, closing delimiter, answer,
  tool-call, JSON/grammar object, or plain text;
- return honest finish details and continuation eligibility;
- avoid appending synthetic text unless explicitly marked and excluded from
  resident-session commit.

Current code reality:

- chat length stops are classified post-generation into `reasoning`,
  `closing_think`, `tool_call`, `structured`, or `answer`;
- deterministic buffered answer/structured length stops return single-use
  15-minute continuation handles; ineligible length stops explicitly report
  `continuation_eligible=false`;
- length-finished root-object JSON outputs use the shared
  `JsonObjectConstraintState` to reject structurally invalid partial prefixes
  from continuation for JSON-object mode and JSON Schema/guided-JSON schema
  requests that have begun with `{`: the partial text is preserved with coarse
  `finish_reason="length"`, but `finish_details.reason="schema_violation"` and
  `continuation_eligible=false`;
- no synthetic text is appended; decode-loop phase accounting remains
  DecodeState work.

Exit gates:

- tests cover length in `<think>`, partial `</think>`, partial `<tool_call>`,
  partial JSON, and normal answer;
- no case silently returns empty assistant content without an explanatory detail;
- continuation handle presence/absence is deterministic.

#### P1.6 Continuation handles

Implement:

- persistent generation handles for resumable length stops, returned as
  `continuation_id` and resumed by sending the same `continuation_id` on a
  follow-up request;
- handle metadata: model id, session/cache reference, decode state, tokenizer,
  sampler params, seed/RNG state, and allowed continuation budget;
- single-use handles by default, with an explicit `reuse=true` debug option only
  after forkable cache semantics exist;
- expiration and cancellation cleanup. Default TTL: 15 minutes or server
  shutdown, whichever comes first. Handles are scoped to model id, tokenizer id,
  auth principal/API key, and session id.

Current code reality:

- app-local continuation handles are implemented for deterministic buffered
  `/v1/completions` and `/v1/chat/completions` length stops;
- handles are scoped to the served model, endpoint, tokenizer compatibility
  metadata, authenticated bearer principal, and session id, single-use, expire
  after 15 minutes, and return stable `invalid_continuation` /
  `continuation_expired` errors. Stateless handles store a null session id;
  session-backed chat handles are scoped to the app-local transcript session
  and require that session to still exist on resume;
- resume requests use the stored prompt/rendered chat plus prior generated text,
  reject fresh completion `prompt` or chat `messages` payloads before
  generation, and inherit stored `response_format` when the follow-up omits it;
- the implementation re-prefills the stored rendered prompt plus prior generated
  text and reports `resident_state_reuse=false`; it does not yet preserve
  decode state, full tokenizer state, RNG state, or resident KV;
- final choice telemetry mirrors stored-handle eligibility into
  `choices[].hipengine.decode_state.continuation_eligible` when backend
  telemetry is present. Explicit backend `FinishDetails.continuation_eligible=false`
  suppresses handle creation even when the server-side request shape would
  otherwise be deterministic, while backend `true` is still downgraded when
  server policy makes the request ineligible, such as stop/`ignore_eos` length
  finishes. Live lower-loop decode states still do not create or scope
  continuation handles themselves;
- streaming, fresh prompt/messages payloads, logprobs, completion `echo`,
  `n != 1`, non-deterministic sampling/logit processors, `ignore_eos=true`,
  OpenAI `stop` controls, chat tools, explicit `response_format` overrides, and
  thinking-budget controls (`reasoning_effort`, top-level budget fields,
  `chat_template_kwargs`, nested `thinking`, and nested `reasoning`) are
  rejected on resume and are not eligible for new handles. The capabilities
  manifest exposes these
  creation/resume blockers under
  `sessions.continuations.ineligible_when` and
  `sessions.continuations.unsupported_resume_fields`.

Exit gates:

- continuation resumes after normal text and after partial structured output;
- invalid/expired handles return stable errors;
- resuming does not reprefill the whole prompt when resident state is available.

### P2 — Tool-call and structured-output reliability

The current tool-call support is enough for local smoke tests; harness-grade tool
use needs decoding constraints and better protocol coverage.

Current state:

- Chat requests accept OpenAI-style `tools`, `tool_choice`, and
  `parallel_tool_calls`.
- Tool output remains prompt-and-parse: Qwen-style `<tool_call>{...}</tool_call>`
  blocks are parsed after generation and converted to OpenAI `tool_calls`.
  Compatibility parsing recovers the common
  `<tool_call><tool_call>{...}</tool_call>` duplicated-start wrapper when the
  inner JSON is valid, including under strict tool validation before schema
  checks. Endpoint regressions pin the repaired non-streaming message and
  streaming `delta.tool_calls[]` shape, including `id`, `type="function"`,
  stable indexes, and string-valued JSON `function.arguments`.
- Once a tool block parses, the server always rejects undeclared tool names and
  multiple parsed calls without `parallel_tool_calls=true`, including in
  compatibility auto-tool mode.
- Strict result validation now runs when `tool_choice` is `none`, `required`, or
  a specific function, when any tool function declares `"strict": true`, or when
  `parallel_tool_calls` is explicitly supplied. It validates selected tool
  names, malformed tool-call blocks, and a minimal function `parameters` JSON
  schema subset: scalar types, `enum`, `const`, local `$ref` into `$defs` or
  `definitions`, conditionals with `if` / `then` / `else`, object
  `properties` / `patternProperties` / `propertyNames` / `required` /
  `dependentRequired` / `dependentSchemas` / `additionalProperties=false` or a
  supported subschema / `minProperties` / `maxProperties`, array `items` /
  `contains` / `minItems` / `maxItems` / `minContains` / `maxContains` /
  `uniqueItems`,
  string `minLength` / `maxLength` / `pattern`, and numeric min/max /
  `multipleOf` bounds. `enum`, `const`, and
  array `uniqueItems` use JSON-typed value equality after generation, so
  booleans are distinct from numbers while numeric `1` and `1.0` compare equal.
  String `pattern` and object
  `patternProperties` use Python regular-expression search semantics after
  generation, and invalid regexes are rejected before generation. Numeric
  `multipleOf` uses decimal divisibility semantics after generation, and invalid
  divisors are rejected before generation. Unsupported validation keywords are
  rejected before generation when strict validation would use the schema;
  annotation keys such as `title`, `description`, and `default`, plus schema
  `format`, are accepted but ignored by validation.
- Tool-policy and strict-validation failures return normal chat responses with
  no successful `tool_calls`, stable `finish_details.reason` values
  (`invalid_tool_call`, `tool_required_not_satisfied`, or `schema_violation`),
  and coarse `finish_reason="stop"` except when the backend ended by generation
  length. In length-exhausted strict tool failures, `finish_reason` remains
  `"length"` and `finish_details` keeps the length limit, classified phase, and
  `continuation_eligible=false`. These normal-response failure reasons are
  advertised under
  `features.tools.result_validation_failure_reasons`. The default
  `invalid_tool_call` surface remains this normal response, but chat requests
  may opt into `invalid_tool_call_error_mode="hard_error"` to receive an HTTP
  error in non-streaming responses or an SSE `error` event in streaming
  responses after generation-time validation fails.
- Inconsistent `tool_choice` requests are rejected before generation:
  `required` or specific-function choices require a non-empty `tools` list, and
  specific-function choices must use a valid object shape and name a declared
  tool.
- For `tool_choice="none"`, chat sampling suppresses the first token of the
  Qwen `<tool_call>` start marker when tokenization is available. This keeps
  no-tool requests on the existing processor path while preserving result
  validation as the final guard.
- For `tool_choice="required"` and specific function choices, chat sampling
  forces the tokenized Qwen `<tool_call>` start marker when tokenization is
  available. If a tokenized thinking budget is active, the marker is queued
  until the `</think>` close sequence moves the row into answer phase. This uses
  the same host forced-token queue family as thinking hard-close, so it routes
  through processed AR sampling and blocks current raw-argmax MTP verification.
- Required and specific function modes also tokenize the Qwen `</tool_call>`
  close marker when possible. If generation begins that close marker, host row
  state forces the remaining suffix through model decoding so the closing tag is
  not left half-emitted.
- Specific function choices, plus `required` mode with exactly one function
  tool, best-effort tokenize the Qwen
  `<tool_call>{"name":"...","arguments":` prefix. If that full prefix starts
  with the separately-tokenized `<tool_call>` marker, host row state forces the
  selected name and `arguments` field prefix through model decoding after the
  start marker. If tokenization is unavailable or non-composable, the server
  falls back to start-marker forcing plus result validation. Multi-tool
  `required` mode intentionally does not force a name.
- `response_format={"type":"json_object"}` and
  `response_format={"type":"json_schema","json_schema":{"schema": ...}}` are
  accepted for completion and chat requests as post-generation result
  validation. Valid visible JSON is returned normally; invalid stop-finished
  outputs return `finish_details.reason="schema_violation"`, advertised under
  `features.structured_outputs.result_validation_failure_reasons`. Length
  finishes preserve visible partial text; structurally repairable root-object
  JSON prefixes may create deterministic continuation handles, while
  structurally invalid root-object prefixes report `schema_violation` and
  `continuation_eligible=false`. JSON-object and object-root JSON Schema
  requests also enable host decode-time close-suffix forcing when the remaining
  budget exactly fits the lowered suffix.
- `guided_json` is accepted for completion and chat requests as a
  post-generation JSON guidance field with the same object-root close-forcing
  behavior as `response_format`. `true` uses JSON-object validation; schema
  objects, `{"schema": ...}` wrappers, and JSON strings containing schema
  objects use the same fail-closed JSON Schema subset as `response_format` and
  strict tool schemas. Invalid stop-finished outputs are suppressed with
  `finish_details.reason="schema_violation"`. Length-finished structurally
  repairable root-object JSON prefixes keep their text and may use
  deterministic buffered continuation handles; structurally invalid root-object
  prefixes keep their text but report `schema_violation` and
  `continuation_eligible=false`.
- `guided_regex` is accepted for completion and chat requests as
  post-generation result validation. It accepts a non-empty Python regular
  expression string, adds a chat prompt hint, and accepts stop-finished visible
  output when `re.fullmatch()` succeeds after surrounding whitespace is
  stripped. Invalid regex syntax is rejected before generation; invalid
  stop-finished outputs are suppressed with
  `finish_details.reason="schema_violation"`. Length-finished partial regex
  outputs keep their text and may use deterministic buffered continuation
  handles.
- `guided_choice` is accepted for completion and chat requests as
  post-generation result validation. It accepts a non-empty string-choice list,
  adds a chat prompt hint, and accepts stop-finished visible output when it
  matches one listed choice after surrounding whitespace is stripped. Invalid
  stop-finished outputs are suppressed with
  `finish_details.reason="schema_violation"`. Length-finished partial choices
  keep their text and may use deterministic buffered continuation handles.
- `guided_patch` and `guided_diff` are accepted as post-generation unified-diff
  result validation. Valid raw unified diffs or a single fenced `diff`/`patch`
  block are returned normally; invalid stop-finished outputs are suppressed with
  `finish_details.reason="schema_violation"`. Length-finished partial patches
  keep their text and may use deterministic buffered continuation handles. The
  capabilities manifest advertises the supported format, fence labels, allowed
  `fenced` policies, and default policy under `features.structured_outputs`.
- Auto-mode constraints, full argument/schema grammar forcing, and
  grammar-constrained JSON/tool generation remain future work. Public
  structured-output controls still use post-generation result validation for
  schema correctness, but JSON-object and object-root JSON Schema / guided-JSON
  requests now also use `JsonObjectConstraintState` for host decode-time
  structural close-suffix forcing when the remaining budget exactly fits the
  suffix.

#### P2.1 Strict tool-call mode

Implement:

- decode-time enforcement for `tool_choice="auto"` and stronger
  required/specific argument constraints;
- structured refusal/error when a required call cannot be produced under budget.

Exit gates:

- server result-validation and decode-time fixtures cover `none`, `required`,
  and specific function choice;
- no-tool mode suppresses `<tool_call>` starts;
- required/specific-tool modes force `<tool_call>` starts when tokenization is
  available, including after tokenized thinking close;
- specific-tool and single-tool required modes force the selected function-name
  prefix when tokenizer composition is safe;
- required/specific-tool modes repair partial `</tool_call>` close markers when
  tokenization is available, and still do not return ordinary prose as success.

#### P2.2 Tool JSON schema validation

Implement:

- extend the current post-generation validation into decode-time constraints for
  tool names and JSON schema when grammar support exists;
- broaden the current minimal schema subset only when tests and compatibility
  fixtures require it;
- retry/repair is a later explicit policy, not the default.

Current code reality:

- post-generation strict validation covers selected tool names, malformed tool
  blocks, required/extra arguments, scalar types, `enum`, `const`, nested
  objects, object property-count bounds, object-valued additional-property
  schemas, pattern-property schemas, property-name schemas, dependent-required
  properties, dependent schemas, local `$ref` through `$defs` / `definitions`,
  arrays, array length/contains bounds, string length bounds/patterns,
  JSON-typed `enum`/`const`/array uniqueness, numeric bounds/multiples, schema
  composition with `allOf` / `anyOf` / `oneOf` / `not`, and conditionals with
  `if` / `then` / `else`;
- strict schemas reject unsupported validation keywords before generation rather
  than silently ignoring constraints;
- decode-time schema constraints and retry/repair remain future grammar work.

Exit gates:

- malformed JSON, unknown tool name, missing required arg, wrong type, extra
  disallowed arg, and multi-call-without-opt-in each have stable finish details;
- streaming and non-streaming paths agree on parsed calls/errors;
- prior assistant tool calls and `role: "tool"` results still replay correctly.

#### P2.3 Constrained JSON / schema decoding

Implement:

- JSON-object mode for `response_format` and tool arguments;
- minimal JSON Schema subset: object, required properties, scalar types, enum,
  arrays with bounded depth, and additionalProperties policy;
- close-brace/quote enforcement before EOS when a JSON object is incomplete.

Exit gates:

- generated JSON parses on fixture prompts;
- invalid schema requests return `unsupported_parameter` or `schema_violation`;
- unconstrained sampling behavior is unchanged when no constraint is active.

Current code reality:

- `response_format={"type":"json_object"}` and
  `response_format={"type":"json_schema","json_schema":{"schema": ...}}` are
  implemented as prompt-hint plus post-generation result validation for
  completion and chat outputs. Streaming requests use buffered response paths so
  invalid JSON is not emitted as successful deltas;
- JSON-schema requests use the same fail-closed subset validation as strict tool
  schemas: supported validation keywords are enforced, unsupported validation
  keywords are rejected before generation, and annotation keys, including
  `format`, are ignored by validation;
- `response_format={"type":"text"}` is a no-op;
- `guided_json` is implemented as an alternate request field for JSON-object
  and JSON-schema result validation plus object-root close-suffix forcing, with
  identical schema subset, streaming buffering, continuation inheritance, and
  failure semantics;
- `guided_regex` is implemented as prompt-hint plus Python `re.fullmatch()`
  post-generation result validation against a non-empty regex string. Streaming
  requests use buffered response paths so invalid regex results are not emitted
  as successful deltas;
- `guided_choice` is implemented as prompt-hint plus exact post-generation
  result validation against a non-empty string-choice list. Streaming requests
  use buffered response paths so invalid choices are not emitted as successful
  deltas, and deterministic buffered continuation handles inherit the original
  choice list across partial length stops;
- host decode-time structural close-suffix forcing is implemented for
  JSON-object mode and object-root JSON Schema / guided-JSON requests. It can
  force a suffix only when the current prefix plus that suffix parses as a valid
  JSON root object. This covers brace/bracket closes and value-string quote
  repair at the exact remaining-budget boundary; unfinished keys, missing
  values, escape-state strings, token masks, and schema-constrained generation
  remain future grammar work.

#### P2.4 Guidance / grammar plugin interface

Implement:

- a tokenizer-aware grammar interface over the DFA primitive;
- registry or plugin mechanism for JSON, JSON schema, tool-call-only, and patch
  grammars;
- capability reporting for supported grammars.

Exit gates:

- JSON object and tool-call grammars share the same processor/forced-token path;
- unsupported grammar requests fail clearly;
- adding a new grammar does not require server/model dispatch branches.

Current code reality:

- `/v1/hipengine/capabilities` reports `features.grammars.enabled=false`,
  `strict_decoding=false`, an empty supported grammar list, and known
  unsupported grammar/guidance fields (`grammar`, `guided_grammar`, and
  `guided_decoding_backend`).
  JSON, regex, choice, and patch/diff guidance are reported separately under
  `features.grammars.result_validation_only` because they do not install
  grammar masks; the narrow parse-validated object-root close-suffix path is
  advertised separately under
  `features.structured_outputs.decode_time_close_forcing`.
- Requests that send those grammar/guidance fields are rejected before
  generation through the normal unsupported-parameter path with `error.param`
  set to the rejected field. JSON-object / JSON-schema / guided JSON support
  remains result-validation plus the narrow host close-suffix path described
  above, not grammar decoding. Regex / choice / patch-diff support remains
  result-validation-only. Tests cover every advertised unsupported
  grammar/guidance field.

#### P2.5 Patch/diff constrained mode

Implement:

- optional unified-diff grammar for coding agents;
- configurable policy for file headers, hunks, and fenced vs unfenced patches;
- finish details for grammar success/failure/truncation.

Exit gates:

- generated patch fixtures parse under the selected grammar;
- partial patch at length stop returns continuation handle or structured error;
- plain text mode remains unaffected.

Current code reality:

- Patch/diff constrained decoding is not implemented, but `guided_patch` and
  `guided_diff` now provide fail-closed result validation for coding agents.
  The server accepts `true`, `"unified_diff"`, or an object with
  `type`/`format="unified_diff"` and `fenced` policy. `fenced:"optional"` is
  the default and accepts raw unified diffs or a single fenced `diff`/`patch`
  block; `fenced:true` / `"required"` requires the fenced block, and
  `fenced:false` / `"forbidden"` requires raw unified diff text. Chat rendering
  adds a unified-diff prompt hint. Stop-finished outputs that do not satisfy the
  selected policy return empty content with
  `finish_details.reason="schema_violation"`.
- `/v1/hipengine/capabilities` reports the unified-diff format, `diff`/`patch`
  fence labels, allowed `fenced` policies, and default `fenced` policy under
  `features.structured_outputs` so agent harnesses can discover the contract.
- Length-finished partial guided patches keep their partial text, are classified
  as structured length finishes, and inherit patch validation across
  deterministic buffered continuation handles.
- Plain text generation remains unaffected when `guided_patch` / `guided_diff`
  are absent; true token-level patch grammar enforcement remains future work.

#### P2.6 Tool streaming polish

Implement:

- incremental `delta.tool_calls` chunks with stable ids and indexes;
- argument streaming that clients can concatenate to the non-streaming payload;
- final finish details for parsed/validated tool calls.

Current code reality:

- streaming chat prompt-and-parse paths buffer the generated stream text until it
  can be parsed and validated. Parsed tool-call argument strings are then
  streamed as one or more `delta.tool_calls` chunks with a stable generated id
  and preserved `index`; clients concatenate `function.arguments` chunks to
  recover the same JSON argument string as non-streaming responses. The first
  chunk carries the function name, and later chunks carry argument fragments.
- Runtime-native `n>1` live streaming is intentionally disabled for chat tools,
  logprobs, structured-output validation, OpenAI stop strings, and continuation
  resumes. Those surfaces use the buffered path until live multi-row chunk/final
  metadata can preserve the same validation, logprob, stop, and tool-call
  semantics. Endpoint regressions pin this fail-closed gate even when an engine
  advertises `stream_many_detailed`, and continuation-resume regressions pin
  `stream=true` / `n>1` rejection without consuming the saved handle.
- final streaming chunks report `finish_reason="tool_calls"` and
  `finish_details.reason="tool_calls"` for parsed tool calls, or the same
  strict-result failure details as non-streaming when validation rejects the
  parsed call;
- covered fixtures include single-call streaming, malformed JSON strict failure,
  strict schema failure, duplicated-start compatibility recovery, multi-call
  streaming with preserved indexes, long argument chunk concatenation, and the
  `n>1` live-many fail-closed gate for tools/logprobs/structured/stop. True
  token-live argument chunking before full parse/validation remains future
  decode/streaming work.

Exit gates:

- streaming and non-streaming tool responses round-trip to the same parsed list;
- multi-call fixtures preserve indexes;
- malformed streaming tool JSON is reported consistently.

### P3 — Session, cache, and context control

Agents repeatedly reuse long system prompts, repo summaries, and tool traces.
Expose explicit controls instead of relying on implicit resident-session behavior.

#### P3.1 Selective session commit policy

Implement:

- request/session commit modes under `session.commit`: `append_all`,
  `append_visible_only`, `append_none`, `append_prompt_only`;
- default policy that does not retain hidden reasoning unless explicitly asked;
- automatic downgrade from `append_visible_only` to `append_prompt_only` for
  length/cancel/deadline/invalid-structure/synthetic-output finishes;
- finish metadata recording the selected cache action.

Exit gates:

- hidden `<think>` tokens and malformed/truncated tool-call attempts are not
  retained under `append_visible_only`;
- `append_all` remains available for users who want raw transcript continuity;
- stateless requests without `session.id` do not accidentally retain generated
  tails;
- state accounting remains exact across turns.

Current code reality:

- Stateless requests without a `session` object default to no generated-tail
  retention, and `session.commit="append_none"` is accepted as an explicit
  stateless no-retain policy. Final choice metadata includes
  `finish_details.cache_action="append_none"` so client harnesses can record the
  selected cache behavior.
- Buffered `/v1/chat/completions` accepts `session.id` for `n=1` requests and
  prepends the stored app-local transcript before rendering the next prompt.
  Stateful sessions do not support streaming, completions, or `n>1` in this
  slice.
- With `session.id`, the default commit is `append_visible_only`.
  `append_prompt_only`, `append_none`, and explicit debug `append_all` are also
  accepted. Visible-only commits store the incoming user/tool messages plus the
  final visible assistant answer/tool calls and strip server-parsed
  `reasoning_content`; `append_all` stores raw generated assistant text.
- `length`, `cancelled`, `deadline_exceeded`, invalid/missing tool-call,
  schema-violation, and synthetic-token finishes downgrade
  `append_visible_only` to `append_prompt_only`; `finish_details.cache_action`
  reports the effective action on normal responses and structured error
  payloads.
- Deterministic buffered session-backed chat length stops may mint
  continuation handles. The first length-stopped turn downgrades
  `append_visible_only` to `append_prompt_only`; the resume request must send
  the same existing `session.id` with no fresh `messages`, then commits the
  completed visible assistant answer according to the session policy.
- Authenticated `GET /v1/hipengine/sessions` lists metadata for app-local
  transcript sessions and active continuation-handle counts without transcript,
  prompt, generated, or tool-result text. Authenticated
  `DELETE /v1/hipengine/sessions/{session_id}` removes one app-local transcript
  session.
- There is no resident KV commit or visible-only KV re-prefill yet; transcript
  sessions re-render/re-prefill through the normal prompt path. The capabilities
  manifest reports this under
  `sessions.commit_policy.resident_kv_commit=false`,
  `sessions.commit_policy.visible_only_reprefill=false`, and
  `sessions.commit_policy.visible_only_replay="rerender_app_local_transcript"`.

#### P3.2 Visible-only re-prefill path

Implement:

- detect when raw generated tokens differ from the visible committed transcript;
- re-prefill visible assistant answer/tool calls into resident KV;
- fall back to `append_none` with metadata if visible re-prefill cannot fit.

Exit gates:

- follow-up turn logits match a stateless prompt built from the visible
  transcript;
- re-prefill cost is logged separately from decode;
- no hidden reasoning leaks into visible-only sessions.

#### P3.3 Forkable prefix/session cache

Implement:

- cache handles for pinned prefixes and conversations: `cache_key`, `fork_from`,
  `rollback_to`, `delete`, and `commit`;
- prefix vs turn-history eviction policy;
- cache usage metadata in responses.

Exit gates:

- two branches can fork from one prefix without cross-contaminating generated
  turns;
- rollback restores deterministic continuation state;
- eviction decisions are observable and deterministic under fixed inputs.

Current code reality:

- App-local chat transcript sessions can be listed and deleted through
  authenticated metadata-only endpoints, and `/ready` reports active session,
  stored-message, and continuation-handle counts without exposing transcript
  text.
- App-local transcript sessions can be forked with
  `POST /v1/hipengine/sessions/{session_id}/fork` into a new session id. Forks
  clone the visible transcript messages at request time, respect the configured
  chat-session cap, return matched `engine_busy` route metadata on cap
  overflow, report `resident_state_reuse=false`, and then diverge independently
  on later commits. Forks deep-copy JSON-like transcript messages, including
  nested assistant `tool_calls`, and advertise that behavior under
  `sessions.metadata.fork_deep_copies_transcript`. This gives agent harnesses a
  branch primitive without exposing transcript text through metadata endpoints.
- App-local transcript sessions can be rolled back with
  `POST /v1/hipengine/sessions/{session_id}/rollback` and a target
  `message_count`. Rollback trims visible transcript messages only, reports
  previous/retained counts without exposing transcript text, and keeps
  `resident_state_reuse=false`. Retained messages are deep-copied before the new
  record is installed, and the manifest reports
  `sessions.metadata.rollback_deep_copies_retained_transcript`.
- Forkable pinned prefixes, resident KV cache handles, resident KV
  fork/rollback, and prefix-vs-turn-history eviction policy remain future work.

#### P3.4 Context fitting and auto-clear policy

Implement:

- explicit overflow policies: `fail`, `auto_clear_transient`,
  `truncate_oldest_visible`, `keep_pinned_prefix`, `compact_summary`, and
  `new_session` (`reject` is the implemented default name, while `fail` is an
  accepted alias);
- `/fit_context` preflight with the same decision logic as generation;
- response metadata listing kept/dropped/reset segments.

Current code reality:

- The default overflow policy is explicit `reject`: generation does not
  truncate, auto-clear, or drop request content.
- Stateful buffered chat requests may set
  `session.context_overflow_policy="auto_clear_transient"`. The current
  app-local transcript store has no transient stored segments, so this policy is
  deterministic and conservative: it reports `would_clear_transient=false` and
  `transient_message_count=0`, preserves committed visible turns, never drops
  current request messages, and rejects overflows with the normal
  `fit_context` payload.
- Stateful buffered chat requests may set
  `session.context_overflow_policy="new_session"`. Generation and
  `/fit_context` first render the stored app-local transcript prefix plus the
  current request; if that overflows but the current request alone fits, the
  request is rendered without the stored prefix and the app-local transcript is
  replaced on successful commit. If the current request alone still overflows,
  the request is rejected and the transcript is left intact.
- Stateful buffered chat requests may also set
  `session.context_overflow_policy="truncate_oldest_visible"`. Generation and
  `/fit_context` search for the shortest dropped oldest stored prefix whose
  retained suffix plus current request renders through the normal chat
  tool-transcript validation and fits the context budget. On successful commit,
  the app-local transcript is replaced with that kept suffix plus the new turn.
  If no valid suffix fits, the request is rejected and the transcript is left
  intact.
- `/fit_context` reports prompt tokens, effective max tokens, required context,
  max allowed/recommended `max_tokens`, overflow tokens, `fits`,
  `clear_policy`, `would_truncate`, `would_reset_session` for
  `new_session`, sanitized `would_drop`, and `kept_segments` using the same
  helper as generation admission. Chat preflight with `session.id` includes the
  app-local stored transcript prefix before computing prompt tokens unless
  `new_session` can safely reset to the current request or
  `truncate_oldest_visible` can safely keep a fitting suffix; it reports the
  same session-prefix metadata as generation.
- Generation `context_overflow` errors include `error.fit_context` with the same
  actionable shape, so clients can retry with a smaller `max_tokens` or run the
  preflight endpoint without reverse-engineering the error message. For
  session-backed chat requests, generation overflow errors also include the
  same `session` prefix-count and overflow-policy metadata as `/fit_context`.
- Transient-segment clearing beyond the current no-op policy,
  `keep_pinned_prefix`, and summary compaction are still future work;
  `auto_clear_transient`, `new_session`, and `truncate_oldest_visible` never
  drop current request messages.

Exit gates:

- auto-clear never silently drops pinned prefixes or committed visible turns;
- context overflow errors include actionable fit data;
- generation and `/fit_context` agree on token counts and decisions.

#### P3.5 Session snapshot save/restore

Implement after cache layout stabilizes:

- snapshot format for prefix tokens, visible transcript, KV payload references,
  decode/sampling state, tokenizer/model id, and cache policy;
- atomic write/read and versioning;
- clear incompatibility errors when model/quant/backend/tokenizer differs.

Exit gates:

- restored sessions pass deterministic continuation fixtures;
- corrupted/incompatible snapshots fail safely;
- snapshot files never include secrets or unredacted debug payloads by default.

Current code reality:

- App-local transcript sessions support authenticated snapshot export and
  restore at `/v1/hipengine/sessions/{session_id}/snapshot`.
  Snapshots are versioned as `hipengine.chat_session_snapshot.v1`, include the
  visible transcript messages, served model id, backend, quant, tokenizer
  compatibility metadata, storage, and timestamps, and explicitly report
  `resident_state_reuse=false`.
- Restore validates the snapshot object/schema envelope, top-level
  resident-state flag, same session id, model id, backend, quant, storage,
  transcript-inclusion flag, session resident-state flag, tokenizer metadata
  when the model is loaded, timestamps, message shape, supported roles
  (`system`, `developer`, `user`, `assistant`, `tool`), text content parts,
  message string metadata, role-specific `tool_calls` / `tool_call_id`
  placement, and nested assistant `tool_calls` shape, including valid JSON
  `function.arguments` strings, before creating or replacing the app-local
  transcript session. Incompatible snapshots fail with stable
  `invalid_request` errors and do not create a session.
- Restoring a new snapshot session respects the configured chat-session cap and
  fails with matched `engine_busy` route metadata without creating partial
  session state when full.
- Checked-in server-conformance and golden-trace fixtures cover restoring an
  app-local session after a hidden-reasoning assistant tool call, then
  continuing from a tool response. They verify the snapshot and restored prompt
  retain only visible tool-call state and do not replay hidden reasoning.
  Snapshot export deep-copies transcript payloads and advertises that under
  `sessions.metadata.snapshot_export_deep_copies_transcript`.
- Resident KV payload references, prefix token blobs, full tokenizer state, and
  decode/sampling state are not snapshotted yet; restored sessions re-render the
  transcript through the normal prompt path.

### P4 — Scheduler, batching, and native sampling polish

These items overlap with `docs/SAMPLING.md`; keep the detailed sampler plan
there and use this section to track serving impact.

#### P4.1 Native c>N stochastic execution

Implement:

- use scheduler-owned `RowSamplingState` for true batched sampled decode;
- project logits for active rows in batch where possible;
- keep per-row seeds/history/stop DFA aligned with request ids and row indexes.

Exit gates:

- c=2/4 fixed-seed fixtures are deterministic;
- semantics match independent c=1 where expected;
- unsupported processor combinations fall back or reject explicitly.

Current code reality:

- The scheduler owns per-request `RowSamplingState` and per-row
  `PerRowSamplingParams` for c>N batches. Its `SamplerParamsBlock` preserves
  row-aligned sampler knobs, deterministic per-row seeds, forced-token queues,
  post-thinking forced-token queues, sequence-completion repair metadata, and
  thinking-budget fields.
- `SamplerParamsBlock` now exposes the shared `plan_sampler()` outcome for each
  request row, plus JSON-ready per-row metadata containing sampler mode,
  active processors, fast-path blockers, host-logits use, native availability,
  and fallback reason. This gives native/scheduler decode callers a single
  policy source for controlled-decoding fallback/rejection decisions.
- PARO c>N sampled generation records that row-aligned metadata under
  `last_batch_generation.sampler_plan_metadata` and builds final per-choice
  telemetry from the scheduler-owned row plans, so native-sampler request
  fallback and processor blockers are observable from the actual batch path.
  Greedy and sampled PARO c>N batch paths and GGUF serial c>N final paths also
  retain JSON-ready `last_batch_generation.scheduler_token_chunks` with
  per-token text, finish-details, decode-state telemetry, stop-suffix state,
  execution-path flags, and sampled logprob payloads when requested.
- True batched sampled decode is still not native-promoted: env-enabled PARO
  c>N sampled batches can use native packed prefill plus serial per-slot native
  GPU sampling when every row is covered by the native sampler contract, but the
  decode loop remains the serial layer bridge. Unsupported PARO c>N rows and
  GGUF sampled requests stay on host sampling with explicit fallback metadata.

#### P4.2 GPU sampler kernels

Current state:

- standalone GPU sampler kernels cover row-wise processor application,
  full-vocab temperature sampling, bounded `top_k <= 64`, exact full-vocab
  `top_p`/`min_p`, bounded top-k probability filters, counter-based row/step
  RNG, selected-token logprob output, suppress-token/min-token masking, and
  small-vocab GPU1 CPU-reference fixtures;
- supported PARO c=1 sampled requests route through those kernels by default;
  `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` disables the route for rollback. PARO c=1
  sampled telemetry reports
  `full_vocab_logits_d2h=false` and `logits_d2h_bytes=0` when that native
  route actually runs. Default-enabled PARO c>N sampled batches use the same
  no-full-logits-readback metadata when every row is routed through serial
  per-slot native sampler state. PARO/GGUF host-logits fallback paths report
  `full_vocab_logits_d2h=true` plus known per-token vector bytes, so server
  metadata can distinguish native sampling from host readback selection. PARO
  c>N unsupported sampled rows and GGUF sampled requests still run through host
  sampling, but when native sampling is enabled or explicitly requested for a
  route shape, their decode-state telemetry reports
  `sampler_fallback_reason="native_gpu_unsupported_request"` instead of the
  generic host fallback reason. The capabilities manifest now advertises the
  same current native-sampler blockers enforced by `supports_native_gpu_sampling`,
  including forced-token queues, post-thinking forced-token queues,
  sequence-completion repairs, and JSON object close forcing. It also separates
  native GPU pre-selection
  `processors` from `post_selection_controls` for stop token ids and
  multi-token stop sequences, which PARO c=1 and serial per-slot c>N native
  sampling check after each selected token. The manifest scope is
  `paro_c1_and_serial_per_slot_c_gt_1`, while true batched c>N remains
  explicitly unsupported;
- true batched c>N/GGUF integration and broader
  retained profiler/shape coverage remain unimplemented.

Exit gates:

- CPU-reference fixtures pass on small vocab for any newly promoted shape;
- GPU1 deterministic smoke passes for any newly promoted shape;
- rocprof evidence is recorded before any performance claim.

#### P4.3 Exact GPU top-p

Current state: a standalone correctness-first exact full-vocab GPU
`top_p`/`min_p` sampler exists and the default PARO native route can dispatch
supported `top_k=0` top-p requests. A faster nucleus selector remains future
performance work.

Remaining implementation:

- performance-oriented full-vocab nucleus selection without weakening
  retain-one and tie-break semantics;
- true batched c>N/GGUF native routing for unsupported native shapes.

Exit gates:

- GPU top-p retained sets match host on fixtures;
- fallback behavior is explicit when exact GPU top-p is unsupported;
- no approximate top-p path is promoted as exact.

#### P4.4 Logprobs parity

Current state: host-logits logprobs are implemented for non-streaming
completions/chat, buffered streaming completions/chat, live streaming
completion/chat chunks from engines that advertise `supports_stream_logprobs`,
and completion `echo+logprobs` with the echoed prompt represented as one prefix
entry with `null` prompt logprob plus `prompt_logprob_unavailable` omission
metadata. PARO/GGUF c=1 host-sampled `stream_detailed()`
chunks emit per-token selected logprob and top-logprob metadata for logprob
requests, so the OpenAI server can expose live logprob deltas without buffering
on those paths. Backend paths that cannot provide token metadata for a logprobs
request return stable HTTP 501 `unsupported_feature` with
`error.param="logprobs"`, and the capabilities manifest advertises that
fallback under `features.logprobs.missing_backend_metadata_error`; it also
advertises opt-in live stream support under
`features.logprobs.live_chunk_metadata`. Chat logprobs are mapped to visible
assistant content spans after reasoning markup is split, so hidden
`reasoning_content` spans do not steal or receive public `logprobs.content`
entries. Server tests cover representative completion/chat logprob success,
buffered streaming, live chunk streaming, visible-content mapping after
reasoning, and the missing-backend-metadata fallback. When token metadata is
present but a generated-token selected score is omitted, successful
completion/chat responses keep the OpenAI-compatible score as `null` and add
`choices[].logprobs.hipengine.omitted_token_logprobs[]` with reason
`backend_omitted_logprob`; echoed prompt prefixes use reason
`prompt_logprob_unavailable`. `/v1/hipengine/capabilities` advertises the
stable reason vocabulary under `features.logprobs.omission_reasons`. PARO c>N
host-sampled and serial native-sampled final outputs preserve per-token selected
logprob metadata from scheduler step results; host-sampled rows also preserve
top-logprob metadata. Remaining work is true prompt-token logprobs, true
batched native c>N/GGUF GPU-path coverage, and broader performance promotion.

Implement:

- true runtime-native c>N/GGUF GPU live streaming `logprobs` / `top_logprobs`
  chunks once those routes can provide token metadata incrementally;
- completion `echo+logprobs` prompt-token metadata with real prompt-token
  logprobs when the model/session can score prompt tokens;
- native/GPU sampler logprob output when those paths are promoted.

Exit gates:

- response schema tests pass for greedy, processed-argmax, sampled, streaming,
  and echo cases where advertised;
- logprob values are finite or explicitly omitted with reason;
- unsupported combinations fail with stable OpenAI-compatible errors;
- OpenAI compatibility is documented.

#### P4.5 Admission/backpressure policy

Current state:

- The OpenAI server batcher has an opt-in queue cap via
  `--max-queued-requests` / `HIPENGINE_MAX_QUEUED_REQUESTS`. When the queued
  request count is already at the configured cap, the request is rejected before
  enqueue with HTTP 429, canonical `engine_busy`, `Retry-After: 1`, and matched
  `error.hipengine.routing` metadata with
  `overload_source="generation_queue_cap"` for HTTP generation callers.
- App-local chat transcript sessions have an opt-in cap via
  `--max-chat-sessions` / `HIPENGINE_MAX_CHAT_SESSIONS`. New `session.id`
  requests that would allocate a transcript are rejected before prompt
  preparation/generation with the same HTTP 429 `engine_busy` and
  `Retry-After: 1` plus `overload_source="chat_session_cap"` route metadata.
  Session fork and snapshot-restore requests that would create a new transcript
  use the same cap and route metadata. Existing sessions may continue, and
  deleting a session frees capacity.
- The OpenAI server batcher also has an opt-in active backend request cap via
  `--max-active-requests` / `HIPENGINE_MAX_ACTIVE_REQUESTS`. It limits how many
  HTTP requests can be coalesced into one active backend generation batch;
  overflow stays queued and is still bounded by the queue cap when configured.
- `/ready` reports queue depth, configured max depth, active backend request
  count, and configured active-request cap. Prometheus metrics include
  `hipengine_request_rejected_total`, queue depth, configured queue cap,
  active/max backend request gauges, active/pending/max chat-session gauges, and
  worker-active gauges in addition to completed/failed/cancelled request
  counters. Metrics also expose
  `hipengine_generation_scheduler_fairness_policy_info` with the current
  `fifo_compatible_sampling_key` policy.
- Default behavior remains unlimited server queueing until a cap is configured.
  Default active backend request grouping remains uncapped until a cap is
  configured. Default chat-session behavior remains unlimited until a cap is
  configured. Runtime fairness beyond FIFO-compatible batching remains future
  scheduler work.

Remaining implementation:

- continuous-decode scheduler fairness beyond the current FIFO-compatible
  batcher policy.

Exit gates:

- overload tests do not deadlock active resident sessions;
- rejected requests do not allocate KV/session state;
- metrics expose active, queued, completed, failed, cancelled, and rejected counts.

### P5 — Harness integration and operations

#### P5.1 Capabilities manifest

Implement:

- `GET /v1/hipengine/capabilities` as the canonical manifest; keep `/v1/models`
  OpenAI-minimal, with only optional links/summary fields that do not require
  clients to parse model-list internals;
- context sizes, effective chat default, tokenizer name, chat template family,
  tool support, reasoning controls, sampling modes, logprobs support,
  continuation support, cache/session support, loaded-model count, routing
  support, and known unsupported fields.

Current code reality:

- `GET /v1/hipengine/capabilities` returns an authenticated manifest without
  forcing lazy model load.
- The manifest reports served model/config, configured/effective context tokens,
  bounded vs auto chat default, tokenizer/count-token callable availability,
  Qwen chat-template family, tools/reasoning/logprobs/streaming support,
  logprobs backend-metadata fallback errors,
  the stream metadata extension version/event/timing fields, no-tool
  start-marker suppression, required/specific tool start-marker forcing plus
  its initial-or-post-thinking scope, Qwen tool-call compatibility parser
  repairs, declared-tool-name validation, parallel-tool opt-in enforcement,
  tool-enabled malformed-JSON fail-closed policy, sampling parameters and execution
  modes, strict tool result-validation support,
  transcript-level tool-call/tool-result validation under
  `features.tools.transcript_validation`,
  JSON-object, JSON-schema, guided-JSON, guided-regex, and guided-choice
  structured-output result validation, the
  root-object JSON length-finish structural guard under
  `features.structured_outputs.length_finish_structural_validation`,
  reasoning-control field list with
  `budget_policy="prompt_hint_plus_tokenized_soft_and_hard_close"`,
  tokenizer-dependent `token_budget_enforced`, explicit
  `hard_close_token_forcing`, tokenizer-dependent `soft_close_bias`, and
  tokenizer-dependent `eos_suppression`, the default-on scoped PARO native GPU
  sampler route, speculative/MTP sampling compatibility,
  tool/structured-output result-validation failure reason sets,
  request-timeout/client-disconnect support, backend-authored choice telemetry,
  the optional `decode_state` field vocabulary, grammar-support status,
  queue/active-request/chat-session admission caps, scheduler fairness policy,
  cache/session settings, loaded-model count, and unsupported fields.
- Continuations are advertised as supported with `stateful=false`,
  `resident_state_reuse=false`, `single_use=true`, tokenizer and
  authenticated-principal scope, a 15-minute TTL, deterministic-buffered
  sampling scope, length-only finishes, and no streaming support. Session commit
  policy is advertised as stateful app-local transcript
  storage with `resident_state_reuse=false`, buffered chat-only scope,
  `append_visible_only` as the stateful default, no resident-KV commit, no
  visible-only resident re-prefill, transcript replay through normal rendering,
  and downgrade reasons for unsafe visible-only finishes. Session metadata advertises
  `transcript_message_copy="json_deep_copy"` plus deep-copy guarantees for
  forks, rollbacks, and snapshot export. Multi-model routing and strict tool
  decoding remain advertised as unsupported until their runtime paths exist. Tensor
  parallelism is advertised as disabled with single-process topology and
  explicit unsupported multi-GPU/sharding features. Request timeouts and
  client-disconnect cancellation are advertised as supported with cooperative
  backend deadline/cancel checks and `preemptive_decode_cancel=false`; token
  diagnostics are advertised from current tokenizer/counting callables.
- Known unsupported agent fields are rejected explicitly before generation work:
  `session.id` outside buffered chat, unsupported streaming/`n`/continuation
  combinations, unsupported `session.commit` modes, known unsupported
  grammar/guidance fields, and other `session` payloads return
  `unsupported_parameter` with `error.param` set to the rejected field.
  Server tests now derive the advertised `unsupported_fields` list from
  `/v1/hipengine/capabilities` and prove each listed field is rejected on both
  completions and chat before generation is called.
  Unknown, consumed, wrong-endpoint, or wrong-model
  `continuation_id` values return `invalid_continuation`; expired handles return
  `continuation_expired`.
  Backend-authored choice telemetry capabilities also advertise passthrough of
  optional backend `timing` and `usage` payloads when generation code supplies
  them.

Exit gates:

- pi/local harness can configure itself from the manifest without guessing;
- values match server startup config and current model capabilities;
- tests cover both default and custom `--chat-default-max-tokens`.

#### P5.2 Pi/local-agent config snippets

Implement:

- checked-in minimal config examples for pi or OpenAI-compatible local agents;
- recommended settings for Qwen thinking format, tool calling, timeout/deadline,
  max-token behavior, and unsupported fields;
- a smoke command that validates the config against a running server.

Current code reality:

- `docs/examples/local-agent/openai-compatible.json` is a checked-in
  adapter-neutral config for OpenAI-compatible local agents. It points at
  `/v1`, reads the served model id from `/v1/hipengine/capabilities`, defaults
  to deterministic Qwen-friendly chat (`temperature=0`, `reasoning_effort=none`),
  bounds ordinary generations to `max_tokens=4096`, enables SSE usage and
  hipEngine extension metadata, sets a request `timeout_ms`, explicitly sends
  `session.commit="append_none"` for stateless no-retain behavior, sends tools
  per request, and keeps stateful `session.id` out of the default streaming
  payload plus intentionally unused tool-policy/logprob fields
  (`parallel_tool_calls`, `top_logprobs`) and unsupported grammar/guidance
  fields in `do_not_send`. `guided_json`, `guided_regex`, `guided_choice`,
  `guided_patch`, and `guided_diff` are no longer in `do_not_send` because they
  are supported as result-validation controls.
- `docs/examples/pi-agent/models.json` is a checked-in pi config example for the
  Qwen 3.6 PARO endpoint. It enables pi's thinking UI with `reasoning=true` and
  `compat.thinkingFormat="qwen"` while leaving `supportsReasoningEffort=false`
  so pi sends Qwen `enable_thinking` rather than OpenAI `reasoning_effort`.
- `scripts/validate_local_agent_config.py` validates the snippet against a
  running server capability manifest and can optionally POST a small
  chat/tools smoke request:

  ```bash
  python3 scripts/validate_local_agent_config.py \
    --config docs/examples/local-agent/openai-compatible.json \
    --base-url http://127.0.0.1:8000/v1 \
    --chat-smoke
  ```

- Unit tests keep the snippet synchronized with the advertised unsupported-field
  list and prove the generated smoke payload strips every `do_not_send` field.
  When tools are enabled, the local-agent `--chat-smoke` request forces the
  `record_result` function and requires a parsed tool call with valid JSON
  arguments that set `result` to `"ok"` plus the exact OpenAI function-call
  envelope (`id`, `type="function"`, and `function.name` / JSON-string
  `function.arguments` with no unexpected keys); non-assistant messages,
  ordinary assistant content alongside tool calls, raw `<tool_call>` assistant
  text, leakage in assistant `content` or `reasoning_content`, malformed
  `tool_calls` objects, or wrong tool arguments are rejected as a config/server
  mismatch.
  When tools are disabled, it only validates a normal chat response.
- `scripts/validate_pi_agent_models.py` validates the checked-in pi
  `models.json` shape offline and fails on the common `reasoning=false` or
  missing `compat.thinkingFormat="qwen"` misconfiguration that disables pi's
  thinking UI for a Qwen endpoint. The offline check aggregates these common
  provider/model mistakes, so a config can report missing `thinkingFormat`,
  disabled `supportsUsageInStreaming`, and model-level `reasoning=false` in one
  run instead of failing at the first mismatch. It also rejects disabled
  `supportsUsageInStreaming` because the server advertises usage-bearing SSE
  responses. With `--base-url`, it checks the config model id, context window,
  streaming usage, Qwen thinking control, and tool support against
  `/v1/hipengine/capabilities`; `--chat-smoke` additionally POSTs a small Qwen
  tool-call request and requires a parsed `record_result` tool call with JSON
  arguments that set `result` to `"ok"` and the exact OpenAI function-call
  envelope. Raw `<tool_call>` assistant text, including a doubled start-marker
  form or assistant `content` / `reasoning_content` leakage alongside parsed
  `tool_calls`, non-assistant messages, ordinary assistant content beside a
  tool call, and malformed `tool_calls` objects are rejected as tool-calling
  mismatches. `--streaming-smoke` sends the same tool request as `stream=true`
  with `stream_options.include_usage=true`, requires each first
  `delta.tool_calls[]` fragment to carry a valid OpenAI `id` / `type` /
  `function.name` envelope, reconstructs streamed argument fragments, rejects
  raw `<tool_call>` leakage, and requires both a usage SSE payload and final
  `data: [DONE]`; the helper that performs the live streaming smoke POST is
  regression-tested directly for the `/chat/completions` SSE request shape.
  `--reasoning-smoke` POSTs a small `enable_thinking=true` request, requires
  parsed non-empty `message.reasoning_content`, and rejects raw `<think>` tags
  in assistant text fields as a Qwen thinking/parser mismatch.
- `tests/test_local_agent_config.py` posts that exact pi smoke payload through a
  FastAPI `create_app()` test server with fake Qwen `<tool_call>` output and
  asserts the response is parsed OpenAI `message.tool_calls`, not raw markup. It
  also posts the pi streaming-smoke and reasoning-smoke payloads through the fake
  server, asserting streamed tool-call deltas plus usage metadata round-trip and
  Qwen `<think>` output is split into OpenAI-compatible `message.reasoning_content`.
- Existing server fake-session tests and the P5.3 golden trace harness cover
  parsed Qwen tool calls in non-streaming and streaming responses, including
  multi-turn tool loops, raw-markup rejection, and identical nested assistant
  `tool_calls` validation for live prior messages and restored snapshots before
  prompt rendering. Live prior messages and restored snapshots also validate
  transcript-level tool-result matching: assistant tool-call ids must be unique,
  and every `role: "tool"` result must reference a prior unconsumed assistant
  tool-call id. Once an assistant tool call is pending, only tool-result
  messages may follow until the pending ids are consumed, though transcripts may
  still end with pending calls for the next session-backed request. App-local
  session forks, rollbacks, prefix replay, visible-message commits, and snapshot
  exports deep-copy JSON-like transcript messages so nested assistant
  `tool_calls` / content parts are not shared between session records,
  branches, or exported payloads.

Exit gates:

- snippets stay synchronized with `docs/API.md`;
- a golden tool-call loop works with the documented config;
- unsupported fields are not silently sent by the recommended config.

#### P5.3 Golden harness traces

Implement:

- fixtures for assistant -> tool call -> tool result -> final answer;
- streaming and non-streaming variants;
- reasoning/no-thinking variants;
- length/cancellation/error variants.

Current code reality:

- `tests/fixtures/agentic_traces/golden_traces.json` defines normalized
  deterministic traces for a two-turn assistant -> tool call -> tool result ->
  final-answer loop, streaming tool-call deltas, reasoning extraction next to
  tool calls, strict malformed/missing/wrong/no-tool/parallel-call tool
  rejection, duplicated-start compatibility recovery, JSON-schema
  result-validation failure, stateless
  `session.commit="append_none"` finish metadata, no-thinking prompt rendering,
  session-backed reasoning/tool-call retention that strips hidden reasoning
  from the next prompt, completion length finish metadata, chat reasoning/
  closing-think/answer/structured/tool-call length phase metadata,
  continuation-eligible answer and structured length stops, a multi-request
  continuation-resume sequence, deterministic session-backed continuation
  resume, transcript session rollback that trims later turns from the next
  prompt, guided patch/diff validation success and
  fail-closed rejection paths, context-overflow fit data, wrong-model routing
  errors, unsupported parameter/feature errors, request schema-validation
  errors, streaming context-overflow SSE error chunks, chat-session-cap
  `engine_busy`, invalid and expired continuation handles, backend deadline
  HTTP/SSE error metadata, backend cancellation HTTP/SSE errors,
  request-control cancellation,
  and logprob omission-reason payloads.
- `tests/test_agentic_harness_traces.py` runs those traces against the
  OpenAI-compatible server with deterministic fake generation and strips only
  dynamic IDs/timestamps from assertions. The runner asserts visible transcript,
  parsed tool calls, reasoning deltas, finish details, and expected absence of
  raw tool markup where strict validation handles malformed output. A dedicated
  coverage guard keeps the required server-side agentic pattern buckets
  explicit: multi-turn tool loops, reasoning/no-thinking controls, tool
  validation and compatibility, structured agent outputs, session/snapshot/
  rollback/continuation behavior, finish-phase/sampling contracts, and server
  error paths.
- `tests/test_agentic_server_conformance.py` adds a compact client-pattern
  matrix for the FastAPI `/v1/chat/completions` surface: strict
  reasoning-plus-tool responses, prior assistant tool-call/tool-result replay
  rendering exactly once, live and snapshot prior-assistant tool-call shape
  rejection plus orphan/duplicate/skipped tool-result rejection before
  generation, reasoning-only final-answer responses, reasoning plus structured
  JSON responses in buffered and streaming modes, `enable_thinking=false`
  pre-close rendering,
  duplicated-start tool-call recovery, malformed tool JSON fail-closed behavior,
  streamed malformed tool JSON fail-closed behavior,
  `session.commit="append_none"` finish metadata, app-local `session.id`
  visible-only transcript retention, snapshot export/restore of a hidden-
  reasoning tool-call loop, streaming tool-call parity without raw
  `<tool_call>` leakage, exact OpenAI-compatible streamed
  `delta.tool_calls` envelopes without stray content/reasoning fields,
  deterministic continuation creation/resume/single-use behavior, and stateless
  streamed parallel tool-call continuation from replayed `assistant.tool_calls`
  plus multiple `role="tool"` results. The
  trace suite also includes representative
  completion/chat logprob success paths, explicit selected-score omission
  reasons when backend token metadata is partial, and the stable
  `unsupported_feature` fallback when a backend cannot return token metadata.
- `tests/test_server_api.py` covers server streaming edge paths that need
  direct SSE-shape assertions outside the golden trace harness, including
  backend scheduler token chunks forwarded as buffered completion and plain
  chat answer/reasoning deltas, plus plain chat content logprob and validated
  structured content deltas and validated tool-call argument fragments, with
  per-choice decode-state metadata. Capability and replay-artifact assertions
  compare speculative/MTP incompatible fields and condition strings against the
  sampler module constants so public metadata cannot drift from the runtime
  guard source of truth.

Exit gates:

- traces are deterministic under fixed fake logits or fixed seeds;
- parsed tool calls and final visible transcript match expected JSON;
- traces can be used for regression by future agents.

#### P5.4 Error taxonomy

Implement stable errors for:

- `unsupported_parameter`;
- `unsupported_feature`;
- `invalid_tool_call`;
- `schema_violation`;
- `invalid_continuation`;
- `continuation_expired`;
- `context_overflow`;
- `deadline_exceeded`;
- `cancelled`;
- `engine_busy`;
- `model_unavailable`;
- `routing_failed`.

Current code reality:

- HTTP and SSE error payloads preserve existing OpenAI-style `error.code` values
  for compatibility and add `error.hipengine` with the canonical AGENTIC
  taxonomy code, HTTP status, retryability, and `legacy_code` when the public
  code still uses an older name.
- Request-body validation errors populate OpenAI-style `error.param` with the
  first FastAPI/Pydantic field path when available. FastAPI
  `validation_error` payloads and server-side `invalid_request` payloads both
  map to canonical `error.hipengine.code="schema_violation"`.
- Non-text chat content parts preserve public
  `error.code="unsupported_content_type"` for compatibility while mapping to
  canonical `error.hipengine.code="unsupported_parameter"`.
- `/v1/hipengine/capabilities` advertises
  `errors.schema="hipengine.error_taxonomy.v1"`, canonical code metadata, and
  legacy aliases. Currently emitted canonical codes include
  `unsupported_parameter`, `unsupported_feature`, `invalid_tool_call`,
  `schema_violation`, `invalid_continuation`, `continuation_expired`,
  `context_overflow`, `deadline_exceeded`, `cancelled`, `engine_busy`, and
  `model_unavailable`.
- `docs/API.md` lists the same client-handled taxonomy table, and a server test
  compares that public table's code/status/retry columns against the live
  `/v1/hipengine/capabilities` manifest.
- `invalid_tool_call` is emitted by default as a normal chat
  `finish_details.reason` for parsed tool-policy failures and strict tool
  result-validation failures. With
  `invalid_tool_call_error_mode="hard_error"`, the same validation failure is
  emitted as an HTTP error for non-streaming chat or an SSE `error` event for
  streaming chat. `schema_violation` can likewise be a request-body error or a
  normal `finish_details.reason` for invalid `response_format` / strict tool
  schema results.
- `engine_busy` currently means the opt-in OpenAI server queue cap or app-local
  chat-session cap rejected a request before generation with HTTP 429 and
  `Retry-After: 1`. The active backend request cap limits coalesced backend
  batch width but does not itself reject unless queued overflow also hits the
  configured queue cap.
- `routing_failed` is reserved in the manifest and marked `emitted=false` until
  multi-model routing exists. Decode-time invalid-tool-call hard errors remain
  future grammar work; the current hard-error mode is still post-generation
  result validation.

Exit gates:

- each error has status code, machine code, parameter/path when applicable, and
  human-readable message;
- streaming errors use a consistent SSE error chunk;
- docs/API lists the errors clients should handle.

#### P5.5 Health/readiness diagnostics

Implement:

- readiness fields for model loaded, warmup complete, allocator/KV capacity,
  graph cache status, selected GPU, queue depth, active sessions, loaded models,
  and last startup/load timings;
- distinction between liveness (`/health`) and readiness/capability.

Current code reality:

- `GET /health` is a liveness-only probe that returns `status=ok` and the served
  model id without implying readiness.
- `GET /ready` returns HTTP 200 when ready and HTTP 503 while startup is not
  ready. The payload reports model loaded state, eager-load/warmup completion,
  last startup timings, configured/effective context, KV policy and capacity
  estimate, KV pool metrics, graph cache metrics, backend/device environment,
  parsed visible GPU ids and selected visible device from ROCm visibility env
  vars, generation queue depth/worker state, active backend request
  count/configured cap, active session count, stored-message count, pending
  session creations, configured session cap, continuation-handle count, and
  loaded-model count.
- Readiness is `false` for eager-load servers until startup preparation and
  warmup complete. Lazy-load servers report ready after startup with
  `model.loaded=false` until the first lazy model load.
- Eager startup failures keep the process live but unready: `/ready` returns
  HTTP 503 with `status="error"`, a stage-specific `startup.error`, and
  operator guidance in `diagnostics`.
- Tests assert the readiness payload does not expose warmup prompt text or
  generated warmup output in both success and failure readiness payloads, and
  session observability tests assert prompt and generated text stay out of
  readiness/session metadata payloads.

Exit gates:

- readiness turns true only after eager load/warmup when enabled;
- failures include actionable diagnostics;
- no sensitive prompt/generated text appears in health payloads.

#### P5.6 Deterministic replay bundle

Implement:

- opt-in compact artifact for failed harness requests: request JSON, model id,
  sampler params, seeds/RNG state, token counts, finish details, capability
  snapshot, and redacted prompt hashes;
- clear redaction controls for prompts/tool results.

Current code reality:

- Failed-request replay artifacts are opt-in through `--replay-dir` or
  `HIPENGINE_REPLAY_DIR`; no artifact is emitted by default. This includes
  normal HTTP error responses and streaming SSE error events such as backend
  deadline/cancellation errors.
- Artifacts use `schema="hipengine.replay.v1"` and include method/path,
  redacted request JSON, prompt/tool-result hashes, served model id, requested
  sampler and agentic control fields, seed fields, error or agentic
  result-validation metadata, finish details when present, completion/chat
  prompt token counts when an already-loaded engine can count them safely,
  explicit unavailable reasons otherwise, and a compact capability snapshot with
  current sampler/MTP compatibility, tokenizer-dependent tool/reasoning
  controls, and cache/session support.
- Strict tool, structured-output, and validation-only guided-output
  result-validation failures (`guided_json`, `guided_regex`, `guided_choice`,
  `guided_patch`, and `guided_diff`) that return normal HTTP 200 responses also
  write replay artifacts when replay is enabled; the artifact stores the failure
  `finish_details` and affected choice indexes, not generated assistant text.
- `--replay-redaction hash` / `HIPENGINE_REPLAY_REDACTION=hash` is the default
  and replaces request strings plus compact sampler/agentic-control strings
  with SHA-256/length metadata. The explicit `none` mode stores raw strings for
  local debugging only.
- Tests cover default-off behavior for failed HTTP requests and normal HTTP 200
  agentic result-validation failures, failed HTTP request redaction, streaming
  backend deadline/cancel errors, strict tool validation failures, streaming
  strict tool failures, completion structured failures, validation-only guided
  JSON/regex/choice/patch/diff failures, and streaming chat structured
  failures. Replay artifact tests load each emitted artifact and re-serialize it
  with `allow_nan=false`, so non-standard JSON values fail the suite.

Exit gates:

- replay artifacts are finite JSON;
- tests verify redaction and stable schema;
- artifacts are not emitted by default for sensitive deployments.

### P6 — Multi-model, routing, and tensor parallel serving

These are not needed for the first single-model agent harness, but they are part
of the longer-term agent-runtime surface and should stay visible.

#### P6.1 Multiple resident models

Implement:

- server config for multiple model/quant/backend entries;
- per-model weight residency and KV pool accounting;
- explicit VRAM admission at startup or dynamic load time;
- model unload/eviction policy, if dynamic residency is supported.

Exit gates:

- `/v1/models` advertises loaded/resident status and capability per model;
- requests can target two loaded models without cross-contaminating sessions;
- overload/admission failure is explicit and leaves existing models usable.

Current code reality:

- `/v1/models` remains single-model, but each OpenAI-compatible model entry now
  includes a `hipengine` extension with backend/quant/path, loaded state,
  resident-context support, context defaults, KV policy/capacity estimate when
  available, a compact capability summary, a capabilities URL for the detailed
  manifest, and routing count metadata.
- Successful non-streaming `/v1/completions` and `/v1/chat/completions`
  responses include `hipengine.routing` metadata with the requested model,
  served model, single-model exact policy, loaded model count,
  `multiple_models=false`, and `fallback_used=false`.
- Streaming responses include the same routing metadata in top-level
  `hipengine.routing` when `stream_options.include_hipengine=true`.
- Wrong-model requests fail before generation with `model_unavailable` and
  include `error.hipengine.routing` metadata for the failed single-model match:
  requested model, configured model, no served model, exact-match policy,
  loaded-model count, `matched=false`, and `reason="model_unavailable"`.
- Context-overflow requests that match the served model include matched
  `error.hipengine.routing` metadata with `reason="context_overflow"` alongside
  the existing `error.fit_context` diagnostics; live SSE context-overflow errors
  preserve the same nested error diagnostics.
- Admission-overload requests that match the served model include matched
  `error.hipengine.routing` metadata with `reason="engine_busy"` and an
  `overload_source` such as `generation_queue_cap` or `chat_session_cap`.
- Multiple resident models, per-model VRAM admission, unload/eviction, and
  cross-model request targeting remain future routing work.

#### P6.2 Capability-aware routing

Implement:

- route by requested model id, context length, grammar/tool capability, quant,
  backend, current load, and optional policy labels;
- no implicit fallback unless the request opts in;
- stable routing metadata in responses.

Exit gates:

- routing fixtures cover missing model, unsupported grammar, context overflow,
  overloaded target, and explicit fallback;
- clients can see which model actually served the request;
- routing does not add backend/quant branches in generation code.

Current code reality:

- Missing-model routing is covered for the current single-model route:
  `/v1/completions` and `/v1/chat/completions` reject mismatched model ids
  before generation with `model_unavailable`, and the error extension includes
  the failed exact-match route metadata.
- Context-overflow routing is covered after the current single-model route
  matches: buffered and live SSE errors keep `error.fit_context` plus matched
  route metadata.
- Overload routing is covered for current admission caps: generation queue cap
  and app-local chat-session cap errors carry matched route metadata plus an
  overload source. Unsupported grammar/guidance fields fail before generation
  with stable unsupported-parameter errors plus matched route metadata
  (`reason="unsupported_grammar"`, the rejected field, and
  `unsupported_capability="grammar"`). They are not capability-routed across
  alternate models yet because multi-model routing does not exist.

#### P6.3 Tensor parallelism / multi-GPU plan

Design before implementation:

- define TP boundaries for weight shards, KV cache, collectives, sampler output,
  graph capture, and session snapshots;
- identify required collective kernels/libraries and failure handling;
- decide how TP interacts with routing and multiple resident models.

Exit gates:

- design doc identifies required kernels/collectives and smallest measurable
  smoke;
- no TP code lands on the default path without hardware validation;
- capability manifest reports TP topology and unsupported features.

Current code reality:

- `docs/TENSOR_PARALLEL.md` is the current P6.3 design gate. It defines rank-0
  ownership for routing, scheduling, sampling, sessions, and response assembly;
  weight-shard, replicated-KV, loop-visible collective, sampler-output,
  graph-capture, session/snapshot, routing, and failure-handling boundaries;
  the smallest required multi-GPU smoke; and the default rule that no TP code
  lands on the single-GPU default path without multi-GPU hardware validation.
- `/v1/hipengine/capabilities` reports
  `parallelism.tensor_parallel.enabled=false` with a single-process topology
  (`world_size=1`, rank/local-rank `0`), no collective backend, and explicit
  unsupported features for multi-GPU weight/KV sharding, collectives, graph
  capture, and cross-rank session snapshots.
- No tensor-parallel runtime, sharded weight loader, distributed KV cache,
  collectives, or multi-GPU graph capture path exists yet.

#### P6.4 Model-family fallback policy

Implement only after routing exists:

- opt-in request field for fallback model families;
- deterministic preference order and capability checks;
- response metadata showing requested vs served model.

Exit gates:

- no implicit model substitution;
- fallback failures are explicit;
- session/cache handles remain scoped to the served model/tokenizer.

## Implementation dependency map

Use this as the handoff for an implementation agent:

1. **Do P0 before deep P1/P2 work.** Token-level phase/accounting and finish
   metadata are prerequisites for reasoning budgets, honest truncation, and
   structured-output errors.
2. **Build the processor primitives once.** Static logit bias, suppress tokens,
   stop DFA, forced-token queue, budget processors, and grammar processors should
   share one ordered stack instead of separate per-feature hooks.
3. **Reasoning soft-close is the first policy on that stack.** It exercises
   dynamic bias, forced close sequences, answer-token reserve, graph-tail
   fallback, finish details, and session commit rules.
4. **Constrained tool/JSON decoding should reuse the same DFA/forced-token path.**
   Do not build a separate tool-call parser that cannot later support JSON schema
   or patch grammars.
5. **Session commit policy must land before long-lived agent sessions are enabled
   by default.** Visible-only commit may require re-prefilling visible output; do
   that explicitly rather than retaining hidden reasoning in KV.
6. **P6 routing/TP/multi-model work should not precede the single-model contract.**
   Expose capabilities first so clients can detect what the server can actually
   do.

## Remaining priority buckets

The P0-P6 headings above are subsystem roadmap areas, not the order to finish
the current agent runtime. Use the buckets below for remaining implementation
priority:

- **P1: core agent contract.** Incorrect behavior here can break a coding agent,
  leak hidden state, poison a session, or make a client believe an unsafe path
  is supported. P1 work should be implemented or covered by a deterministic
  endpoint regression before moving to broad runtime parity.
- **P2: core capability that may fail closed.** These are important features,
  but the server may safely reject, buffer, withhold, or fall back while still
  being a usable local agent runtime, as long as capabilities and finish/error
  metadata are explicit.
- **P3: later scale/performance/convenience.** Useful for production serving,
  performance, or larger deployments, but not required for a correct
  single-model local coding-agent loop.

### P1 — Core agent contract

The single-model server contract, capabilities manifest, token diagnostics,
session transcript commit/context fitting policy, deterministic continuations,
host processor stack, thinking-budget hard/soft close, strict tool result
validation, replay artifacts, local/pi config validation, and golden harness
traces are implemented. Remaining P1 work is therefore mostly audit and
edge-case hardening:

1. **Contract regression map.** Keep the fake-engine endpoint matrix exhaustive
   for reasoning spans, parsed tool calls, structured-output validation,
   continuation/session behavior, replay artifacts, capability fields, and
   unsupported/error taxonomy. Any reported client loop or malformed response
   shape gets a direct non-streaming and, when applicable, streaming regression.
2. **Session-state safety.** Preserve the invariant that hidden reasoning,
   malformed tool calls, schema-violating structured output, cancelled/deadline
   output, and synthetic/repair metadata are never silently committed to
   visible-only transcript state. Session fork/rollback/snapshot/restore,
   continuation resume, and context-overflow policies must continue to validate
   tool transcripts and deep-copy nested `tool_calls`.
3. **Supported reasoning controls.** Keep host/server Qwen thinking controls
   correct where they are advertised: budget aliases/defaults/clamping, token
   diagnostics, parser-close validation, soft-close bias, hard close forcing,
   EOS suppression, forced-token telemetry, and
   `thinking_budget_exhausted` finish details. Native GPU and MTP paths stay
   blocked/fallback until they implement the same policy.
4. **Tool and structured fail-safe behavior.** Keep prompt-and-parse tool calls
   and post-generation structured-output validation fail-closed: no raw
   `<tool_call>` or `<think>` leakage, no successful `tool_calls` on invalid
   outputs, no successful visible content on suppressed schema violations,
   literal marker text must not steal parser state from a later valid tool
   call, and streaming/non-streaming envelope parity.
5. **Sampler/MTP guard completeness.** Raw-argmax speculative/MTP verification
   remains limited to greedy-fast rows. Every field that changes token
   selection or post-accept finish behavior (`logit_bias`, penalties,
   suppressions, min-token/EOS policy, stop controls, forced tokens, thinking
   budgets, logprobs) must keep an advertised blocker and a test.

### P2 — Core but allowed to fail closed

These items should improve the runtime when feasible, but current behavior is
acceptable when it rejects explicitly, falls back to host/server buffering, or
withholds ambiguous backend chunks with diagnostics.

Status: the P2 fail-closed gates are implemented and covered by deterministic
endpoint or scheduler tests. Treat new work here as P2 only when it changes an
advertised reject/fallback/withheld surface or causes the server to claim support
it cannot honor. Full parity implementations belong in the subsystem sections
above and in P3/P4 until the server advertises those capabilities.

1. **Runtime-native live c>N telemetry parity.** Broader c>N
   tool/structured/logprob streams still need emitted chunk/final metadata,
   public handling for invalid/unmappable tool-call chunks, live unmappable
   parser-final logprob spans, and lower-loop continuation creation/scoping.
   Until then, buffering or withheld diagnostics is correct. Current endpoint
   tests pin that tools, logprobs, structured-output validation, and stop
   strings stay on the buffered path even when the engine advertises live
   multi-row streaming; invalid/unmappable scheduler tool chunks and unmappable
   logprob spans are withheld with sanitized `choices[].hipengine` diagnostics;
   and streaming or `n>1` continuation resumes reject without consuming the
   handle.
2. **Native/scheduler controlled-decoding parity.** GGUF/native GPU sampler
   paths and live c>N surfaces should eventually match host AR sampling
   metadata and logprob semantics. Unsupported processor combinations may keep
   falling back or rejecting explicitly. Current code uses the shared sampler
   planner for scheduler rows, advertises native GPU sampler unsupported
   capability fields from the same source as `supports_native_gpu_sampling()`,
   reports `native_gpu_unsupported_request` or processed-logits fallback
   metadata when requested native sampling cannot run, and keeps unsupported
   processor shapes plus bounded `top_logprobs > top_k` off the native route.
3. **Speculative/MTP processed-target verification.** Processed-target MTP
   should apply the same EOS, logit-bias, penalty, suppression, forced-token,
   thinking-budget, stop, and logprob policy as AR sampling. Until that exists,
   the public server advertises no speculative serving route, capabilities list
   every raw-argmax blocker from
   `SPECULATIVE_MTP_INCOMPATIBLE_FIELDS`, scheduler verify work rejects rows
   with those blockers before materializing target verification, and successful
   speculative verify plans preserve `raw_target_top1`,
   `processed_target_verification=false`, and
   `compatible_sampling_modes=("greedy_fast",)` metadata.
4. **Decode-time grammar constraints.** Tokenizer-aware JSON/tool/patch grammars
   should reuse the shared DFA/forced-token path. Current result-validation plus
   narrow JSON close-suffix forcing is acceptable because strict decoding is
   advertised as unsupported, grammar/guidance fields that would imply token
   masks (`grammar`, `guided_grammar`, `guided_decoding_backend`) reject before
   generation on completion and chat endpoints, and supported JSON/regex/choice/
   patch guidance is advertised as result-validation-only.
5. **Additional context policies.** `auto_clear_transient`, `new_session`, and
   `truncate_oldest_visible` cover the currently safe transcript-prefix
   policies. The current `auto_clear_transient` path is a deterministic no-op
   because app-local sessions have no transient stored segments; pinned-prefix
   preservation and summary compaction are P2 only when they can report
   deterministic kept/dropped/reset segments and never drop current request
   content.
6. **HTTP/SSE hard-error variants for invalid tool calls.** The default normal
   chat response with `finish_details.reason="invalid_tool_call"` remains
   supported, and `invalid_tool_call_error_mode="hard_error"` now provides
   opt-in HTTP/SSE hard-error variants for generation-time invalid tool-call
   validation. Decode-time grammar failures remain future work.

### P3 — Later scale, performance, and convenience

These items should not displace P1 hardening or P2 fail-closed correctness work:

Status: the app-local single-model agent surface is implemented enough for
local harness use. P3 now means resident-state reuse, performance promotion, or
larger serving topology, not correctness of the current prompt/replay contract.

Already done at app-local transcript level:

1. **Session safety and explicit commit policy.** Stateless no-retain behavior,
   `append_visible_only`, `append_prompt_only`, `append_none`, debug
   `append_all`, unsafe-finish downgrade to prompt-only, final
   `cache_action`, metadata-only session list/delete, and visible-only
   reasoning stripping are implemented for buffered chat `n=1` sessions.
2. **Branching and recovery without resident KV reuse.** App-local session fork,
   rollback by message count, snapshot export/restore, transcript deep-copy
   guarantees, cap handling, route metadata, and snapshot compatibility checks
   are implemented. These paths all report `resident_state_reuse=false` and
   re-render through the normal prompt path.
3. **Context fitting policies.** Explicit `reject`/`fail`,
   `auto_clear_transient`, `new_session`, and `truncate_oldest_visible`
   policies are implemented for stateful buffered chat, with `/fit_context`
   parity and actionable generation-time overflow payloads. No current policy
   drops request content.
4. **Harness/ops support.** Capabilities, pi/local-agent config snippets,
   golden harness traces, error taxonomy, health/readiness diagnostics, replay
   artifacts, queue/session caps, and basic routing metadata are implemented for
   the current single-model server.

Remaining P3 work:

1. **Resident KV session/cache work.** Visible-only KV re-prefill, forkable
   resident prefix/cache handles, resident rollback/delete semantics,
   resident-state continuation reuse, full tokenizer/decode/sampling state
   snapshots, and prefix-vs-turn-history eviction policy are not implemented.
   Do this only when avoiding re-prefill is a measured bottleneck for real
   agent sessions.
2. **Native sampler full polish.** The scoped PARO native sampler route is
   default-on for supported c=1 and serial per-slot c>N shapes, with
   `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` as rollback. True batched c>N/GGUF native
   sampling and broader profiler/shape coverage remain unimplemented while host
   fallback is explicit.
3. **Multi-model routing, model-family fallback, and TP runtime.** Current
   routing metadata is single-model exact-match only. Multiple resident models,
   capability-aware fallback, model-family substitution, and tensor parallel
   runtime remain deferred; `docs/TENSOR_PARALLEL.md` is still the TP design
   gate.
4. **Patch/tool/JSON grammar completeness and retry/repair policies.** Full
   token-level patch grammar enforcement, broad JSON Schema decoding, and
   automatic retry/repair are later policies unless a concrete agent harness
   cannot operate with current fail-closed validation and narrow JSON
   close-suffix forcing.
5. **Production-serving polish.** True mid-kernel or mid-graph preemption,
   richer cache/KV byte accounting, advanced fair-share routing beyond the
   current FIFO-compatible batcher policy, and other production multi-tenant
   server features remain later work.

## Validation expectations

For each roadmap item that changes runtime behavior:

- Add a focused unit/fake-session test first where practical.
- Prove greedy-equivalent generation still uses the fast path and remains exact.
- Keep server OpenAI compatibility tests for old minimal responses.
- Add or update `docs/API.md` for public behavior changes.
- Add a `WORKLOG.md` entry with commands and results.
- For any performance claim, follow `docs/BENCHMARK.md`; this roadmap alone is
  not benchmark evidence.
