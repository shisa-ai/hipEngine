# OpenAI-Compatible Server API

Last updated: 2026-06-14

hipEngine ships a thin FastAPI layer that adapts OpenAI-style requests to the
torch-free `hipengine.LLM.generate()` library API. Server dependencies are
installed by default. HTTP requests route through the in-process generation
batcher; compatible queued prompts can coalesce into one engine call, while the
remaining async lock is limited to short model/session preparation mutations.

## Install

```bash
pip install hipengine
```

## Run

```bash
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed \
  --quant w4_paro \
  --served-model-name qwen-paro \
  --host 127.0.0.1 \
  --port 8000
```

`--model` accepts a local filesystem path or a Hugging Face model ID that is
already present in the local HF cache. hipEngine resolves IDs with local cache
lookups only; it does not download weights during server startup.

The module entry point is equivalent for environments that prefer `python -m`:

```bash
python -m hipengine serve --model /path/to/model --served-model-name qwen-paro
```

The server defaults to `--backend auto`, which maps exact `gfx1100`/`gfx1151`
ROCm detections to `hip_gfx1100`/`hip_gfx1151`. Unknown HIP targets warn and
select `cpu_reference` where a CPU implementation exists; nearby targets such as
`gfx1101`/`gfx1102` can force a backend with `--backend hip_gfx1100` or
`HIPENGINE_BACKEND=hip_gfx1100` after local validation.

By default the server eagerly loads the model, loads resident weights, estimates
remaining HIP memory for KV cache plus persistent context metadata, then
preallocates `min(model max context, estimated allocatable context)`. Pass
`--max-context-tokens` (or `HIPENGINE_MAX_CONTEXT_TOKENS`) to force a lower cap.
Startup fails with a clear error if the requested cap cannot be allocated; lower
`--max-context-tokens` or use `--kv-storage int8_per_token_head`. Disable eager
startup with `--no-eager-load` or `HIPENGINE_EAGER_LOAD=0`. The warmup prompt and
token count are configurable via `--eager-load-prompt` and
`--eager-load-max-tokens`. Eager startup logs `LOAD_TIMING` rows for resident
preparation, warmup generation, and total startup so weight/session load cost is
visible in ordinary server logs.

The resident KV policy is server-wide: set `--kv-storage` (`auto`, `bf16`, or
`int8_per_token_head`), `--kv-scale-dtype`, and `--kv-scale-granularity` at
startup. Requests that ask for a different KV policy are rejected instead of
rebuilding the resident model. Startup logs include a compact KVCache summary
from current HIP free memory and warn when even INT8 KV is below the model's
advertised max context. Chat requests that omit `max_tokens` use
`--chat-default-max-tokens` (default `4096`) clamped to the remaining admitted
context. Pass `--chat-default-max-tokens auto` to restore the previous behavior
of using the full remaining context (`max_context_tokens - prompt_tokens - 1`).

Per-request deadlines are opt-in via request `timeout_ms`. Set
`--request-timeout-ms` or `HIPENGINE_REQUEST_TIMEOUT_MS` to apply a default
deadline to requests that omit the field. A request-level `timeout_ms` overrides
the server default.

Set `HIPENGINE_API_KEY` or pass `--api-key` to require OpenAI-style bearer
authentication:

```bash
export HIPENGINE_API_KEY=local-secret
curl -H 'Authorization: Bearer local-secret' http://127.0.0.1:8000/v1/models
```

## Endpoints

| Endpoint | Status | Notes |
| --- | --- | --- |
| `GET /health` | Built in | Unauthenticated liveness probe; does not imply model readiness. |
| `GET /ready` | Built in | Unauthenticated readiness/capacity probe. Returns HTTP 200 when ready and HTTP 503 while startup is not ready. |
| `GET /v1/models` | Built in | Returns the single served model id plus `hipengine` status metadata: backend/quant, loaded state, compact capability summary, context defaults, KV policy/estimate, routing count, and capabilities URL. |
| `GET /v1/hipengine/capabilities` | Built in | Authenticated hipEngine manifest for served model/config, context defaults, tokenizer availability, streaming/logprobs/tool/reasoning support, sampling execution/native/MTP status, request-timeout support, cache/session status, routing count, tensor-parallel topology/status, and unsupported fields. |
| `GET /v1/hipengine/sessions` | Built in | Authenticated metadata-only listing for app-local chat transcript sessions plus continuation-handle counts. Does not include prompt, generated, or tool-result text. |
| `DELETE /v1/hipengine/sessions/{session_id}` | Built in | Authenticated deletion of one app-local chat transcript session. Returns whether a session was removed. |
| `POST /v1/hipengine/tokenize` | Built in | Tokenizes raw text with the served tokenizer when available. |
| `POST /v1/hipengine/detokenize` | Built in | Decodes token ids with the served tokenizer when available. |
| `POST /v1/hipengine/count_tokens` | Built in | Counts raw text or rendered chat messages after applying the server chat template, tool markup, thinking controls, and optional app-local `session.id` transcript prefix. Chat diagnostics include lowered thinking-budget close-token metadata when tokenizer support is available. |
| `POST /v1/hipengine/fit_context` | Built in | Reports prompt tokens, effective max tokens, max allowed/recommended `max_tokens`, required/overflow context, and reject/truncation policy using the same admission arithmetic as generation, including optional app-local `session.id` transcript prefixes for chat. Chat diagnostics include the same thinking-budget close-token metadata as `count_tokens`. |
| `POST /v1/completions` | Built in | Text prompt(s) to `LLM.generate()`. For a single prompt with `n=1` and `echo=false`, `stream=true` uses token/chunk SSE from `LLM.stream()` when available; multi-prompt, `n>1`, and echo streaming fall back to buffered SSE. |
| `POST /v1/chat/completions` | Built in | Renders text-only messages to a Qwen-style prompt and calls `LLM.generate()` / `LLM.stream()`. Supports token-level `stream=true` SSE for `n=1`; `n>1` streaming returns buffered per-choice chunks. `<think>` spans are separated into `reasoning_content` (non-streaming) or `delta.reasoning_content` chunks (streaming). Accepts OpenAI `tools` / `tool_choice` and returns `tool_calls` from Qwen-style `<tool_call>{...}</tool_call>` output. |

## Examples

### Text completion

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-paro",
    "prompt": "Hello, hipEngine.",
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

### Chat completion

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-paro",
    "messages": [
      {"role": "system", "content": "Be concise."},
      {"role": "user", "content": "What is hipEngine?"}
    ],
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

### Logprobs

Non-streaming completions accept OpenAI-style `logprobs: N` and return
`choices[].logprobs` with `tokens`, `token_logprobs`, `top_logprobs`, and
`text_offset`. Non-streaming chat accepts `logprobs: true` plus optional
`top_logprobs: N` and returns `choices[].logprobs.content` entries. Requests for
logprobs are routed through the host-logits metadata path so the selected token
logprob/top candidates are based on the same processed logits used for sampling.
For completion `echo+logprobs`, the echoed prompt is represented as a prefix
entry with `null` logprob and generated-token offsets are shifted accordingly.
Streaming requests with logprobs use a buffered detailed-generation path so SSE
chunks can carry logprob metadata; ordinary streams without logprobs remain
live token/chunk streams.

### Routing metadata

Successful non-streaming `/v1/completions` and `/v1/chat/completions`
responses include a top-level `hipengine.routing` extension. In the current
single-model server it reports the requested model, served model,
`fallback_used: false`, `policy: "single_model_exact"`, loaded model count, and
`multiple_models: false`.

### Streaming usage and hipEngine metadata

Both completion endpoints accept OpenAI-compatible `stream_options`. Set
`"stream_options": {"include_usage": true}` with `"stream": true` to request a
final SSE payload with `choices: []` and `usage` before `data: [DONE]`.

Set `"stream_options": {"include_hipengine": true}` to request hipEngine
extension metadata on SSE payloads. Each payload gets a top-level `hipengine`
object with `metadata_version`, `event`, and `timing.elapsed_ms`. After the
first generated chunk, timing also includes server-measured `ttft_ms`; final
done/usage payloads include `decode_elapsed_ms` and `decode_tokens_per_second`
when generated-token counts are available. Choice chunks also get
`choices[].hipengine.phase` (`think`, `answer`, `tool_call`, or `done`) when a
phase is known. When the served engine exposes `count_tokens`, live stream
deltas also include `choices[].hipengine.tokens` with
`delta_tokens`, cumulative `streamed_tokens`, and phase counters such as
`reasoning_tokens` / `answer_tokens`; final choice chunks include usage-derived
`prompt_tokens`, `completion_tokens`, and `total_tokens` in the same object.
Those token-bearing chunks also include a canonical
`choices[].hipengine.decode_state` snapshot with row index, step index, phase,
prompt/generated token counts, and continuation eligibility. Final choice chunks
include the same `finish_details` under
`choices[].hipengine.finish_details`, and usage chunks mirror usage under
`hipengine.usage`. Streaming error chunks use top-level
`hipengine.event: "error"` and include `choices[].hipengine.finish_details` when
structured finish details are available.

The `/v1/hipengine/capabilities` manifest reports the same extension under
`features.stream_metadata`, including metadata version, event names, timing
field names, and whether the server can attach token-accounting-backed
`decode_state` payloads for the loaded engine.

Cache, backend prefill timing, budget-pressure, KV-byte, and backend-authored
per-phase token metadata are omitted until the runtime exposes those signals.

### Choice telemetry

Non-streaming completion and chat choices include `choices[].hipengine` when the
backend returns `GenerationTelemetry`. This extension currently mirrors the
backend-authored `decode_state` snapshot and the final `finish_details`, giving
agent harnesses access to row index, prompt/generated token counts, sampler
mode, active processors, fast-path blockers, stop suffix state, forced-token
queue state, and budget pressure when those fields were authored by the
generation loop. The field is omitted when the backend or fake engine does not
provide telemetry.

### Finish details

Completion and chat choices include a hipEngine extension field,
`finish_details`, next to the OpenAI-compatible `finish_reason`. The extension
always contains `reason` and may include `eos_token_id`, `stop_sequence`,
`length_limit`, `deadline_exceeded`, `cancelled`, `forced_close`,
`synthetic_tokens`, `reasoning_tokens`, `answer_tokens`, `tool_call_tokens`,
`structured_tokens`, `budget_pressure`, `cache_action`, `sampler_mode`, `phase`,
`continuation_eligible`, and `continuation_id`.

`finish_reason` remains the coarse OpenAI value for compatibility. For example,
backend `reason: "eos"` is exposed as `finish_reason: "stop"` with
`finish_details.reason: "eos"`, while backend `reason: "length"` maps to
`finish_reason: "length"`. Tool-call parsing reports
`finish_reason: "tool_calls"` and `finish_details.reason: "tool_calls"`.
Streaming responses include `finish_details` on the final choice chunk;
ordinary delta chunks are unchanged. When chat tool calls are parsed and the
served engine exposes token counting, tool-call finish details also include
best-effort server post-parse `phase: "tool_call"` plus `reasoning_tokens`,
`answer_tokens`, and `tool_call_tokens` for the final parsed response.

For chat length stops, the server adds best-effort post-parse metadata:
`phase` is one of `reasoning`, `closing_think`, `tool_call`, `structured`, or
`answer`. Deterministic buffered length stops in normal answer text or partial
structured output can return a single-use `continuation_id` with
`continuation_eligible: true`; reasoning, closing-think, tool-call, streaming,
logprob, sampled, and active tool/thinking-budget paths report
`continuation_eligible: false`.

PARO/GGUF detailed generation reports backend finish details for EOS, token
stops, stop sequences, length limits, sampler mode, and host-sampled thinking
hard-close enforcement. When the close sequence was forced, details include
`forced_close: true`, `budget_pressure: "hard_close"`, reasoning/answer token
counts, and the current budget phase.

When a backend does not yet provide structured finish metadata, the server emits
the conservative fallback `{"reason": finish_reason}`.

### Continuation handles

For deterministic buffered `/v1/completions` and `/v1/chat/completions`
requests that end by generation length, the server may return a top-level
`choices[].continuation_id` and mirror it in
`choices[].finish_details.continuation_id`. Handles are app-local, single-use,
scoped to the served model and endpoint, expire after 15 minutes, and are
cleared on server restart.

Resume by sending the returned `continuation_id` to the same endpoint. The
original `prompt` or `messages` can be omitted on resume; resumes also inherit
the stored `response_format` when the follow-up request omits it. This
first implementation re-prefills the stored rendered prompt plus prior generated
text instead of reusing resident KV state, and the capabilities manifest reports
`sessions.continuations.resident_state_reuse: false`.

Unsupported resume combinations fail before generation: `stream=true`,
`n != 1`, logprobs, completion `echo=true`, non-deterministic sampling/logit
processors, OpenAI `stop` controls, chat tools, `parallel_tool_calls`,
explicit `response_format` overrides, and thinking-budget controls such as
`reasoning_effort`,
`chat_template_kwargs`, nested `thinking`, and nested `reasoning`. The
capabilities manifest exposes the same contract under
`sessions.continuations.ineligible_when` and
`sessions.continuations.unsupported_resume_fields`.
Unknown or already consumed handles return HTTP 400
`error.code="invalid_continuation"`; expired handles return HTTP 410
`error.code="continuation_expired"`.

### Request deadlines and cancellation

`POST /v1/completions` and `POST /v1/chat/completions` accept `timeout_ms` as a
positive relative deadline in milliseconds. Buffered requests that exceed the
deadline return HTTP 408 with:

```json
{
  "error": {
    "type": "timeout_error",
    "code": "deadline_exceeded",
    "param": "timeout_ms",
    "finish_details": {
      "reason": "deadline_exceeded",
      "deadline_exceeded": true
    }
  }
}
```

Streaming requests send HTTP `200 OK` when the SSE stream starts. If a deadline
expires after that, the stream emits a final error SSE payload with
`finish_reason: "error"` and the same `finish_details`, then emits
`data: [DONE]`. The server also passes the absolute deadline into
`SamplingParams.deadline_at`; PARO/GGUF generation checks it cooperatively at
tokenization, prefill, decode, host-sampling, and graph-replay boundaries.

Client disconnects are checked at the same server await/stream iteration
boundaries. Detected disconnects cancel queued work, mark the generation
`cancellation_token` passed through `SamplingParams`, and use structured
`{"reason": "cancelled", "cancelled": true}` finish details when cancellation
can still be surfaced as an error payload. PARO/GGUF generation checks that
token at the same cooperative boundaries as request deadlines.

Set `HIPENGINE_MAX_QUEUED_REQUESTS` or `--max-queued-requests` to enable an
OpenAI-server generation queue cap. Set `HIPENGINE_MAX_ACTIVE_REQUESTS` or
`--max-active-requests` to limit how many HTTP requests can be coalesced into
one active backend generation batch; overflow remains queued and is still
bounded by the queue cap when configured. Set `HIPENGINE_MAX_CHAT_SESSIONS` or
`--max-chat-sessions` to cap app-local chat transcript sessions. When a
rejecting admission cap is full, new work fails before enqueue/generation with
HTTP 429 `engine_busy` and `Retry-After: 1`; rejected requests do not allocate
KV/session state. Existing chat sessions may continue when the session cap is
full, and deleting a session frees capacity.

### Tool calling

`POST /v1/chat/completions` accepts OpenAI-style `tools`, `tool_choice`, and
`parallel_tool_calls` for local-agent clients such as pi. hipEngine injects a
Qwen-style tool block into the rendered chat prompt and expects the model to
emit tool calls as:

```text
<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>
```

The server converts those blocks to OpenAI-compatible `message.tool_calls` in
non-streaming responses or `delta.tool_calls` chunks in streaming responses, with
`finish_reason: "tool_calls"`. Long streaming `function.arguments` strings are
split into concatenable fragments after the full tool-call block has been parsed
and validated; the first chunk carries the function name, and all chunks carry
the same tool-call id and index. Prior assistant `tool_calls` and `role: "tool"`
messages are also replayed into the prompt as `<tool_call>` and
`<tool_response>` blocks so multi-turn tool loops can continue.
Inconsistent request shapes fail before generation: `tool_choice="required"`
or a specific function choice requires at least one `tools` entry, and a
specific function choice must use a valid object shape and name a declared
tool.

Tool decoding is still prompt-and-parse, not grammar-constrained. The server now
does strict result validation when `tool_choice` is `none`, `required`, or a
specific function, when any tool function declares `"strict": true`, or when
`parallel_tool_calls` is explicitly supplied. Strict validation checks selected
tool names, one-call-vs-parallel policy, malformed tool-call blocks, and the
declared function `parameters` JSON schema subset. Strict failures return a
normal chat response with no successful `tool_calls`, and
`finish_details.reason` set to `invalid_tool_call`,
`tool_required_not_satisfied`, or `schema_violation`. The coarse
`finish_reason` is usually `"stop"`, but remains `"length"` when the backend
ended because the generation budget was exhausted; in that case
`finish_details` also includes length-limit phase metadata.
`/v1/hipengine/capabilities` reports these normal-response failure reasons under
`features.tools.result_validation_failure_reasons`. When
`tool_choice="none"` and tokenization is available, the sampler also suppresses
the first token of the Qwen `<tool_call>` start marker; this is a no-tool
guard, not full grammar-constrained tool decoding. When
`tool_choice="required"` or a specific function is requested and tokenization is
available, the sampler forces the tokenized `<tool_call>` start marker before
ordinary token selection. If a tokenized thinking budget is active, the same
marker is queued until the `</think>` close sequence has moved the row into
answer phase. This prevents ordinary prose from being selected as the first
visible answer/tool token.
Specific function choices, plus `required` mode with exactly one function tool,
also force the selected `<tool_call>{"name":"...","arguments":` prefix when
tokenizer composition shows that prefix starts with the same tokenized
`<tool_call>` marker. Multi-tool `required` mode leaves tool-name selection to
the model, and argument JSON is still result-validated rather than
grammar-constrained. Required and specific function modes also tokenize
`</tool_call>` when possible; once that close marker begins, the host sampler
forces the remaining suffix through model decoding so the closing tag is not
left partially emitted.

The current post-generation schema subset covers `type`, `enum`, `const`,
object `properties` / `required` / `additionalProperties: false`, array `items`
with `minItems` / `maxItems`, string `minLength` / `maxLength`, and numeric
`minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`. Unsupported
validation keywords such as `oneOf`, `anyOf`, `$ref`, `pattern`, and `format`
are rejected before generation when strict tool validation would use the schema;
annotation keys such as `title`, `description`, and `default` are accepted but
ignored by validation. This is result validation only; decode-time JSON/schema
constraints remain unsupported.

### Structured outputs

Completion and chat requests accept `response_format: {"type": "json_object"}`,
`{"type": "json_schema", "json_schema": {"schema": ...}}`, or
`{"type": "text"}`. JSON-object and JSON-schema modes add prompt hints and
validate the completed visible output after generation. Valid JSON is returned
normally; invalid stop-finished outputs return a normal response with empty
successful content and `finish_details.reason: "schema_violation"`. Length
finishes keep their length finish metadata because the partial JSON may be
resumable once continuation handles exist. The capabilities manifest reports
this normal-response failure reason under
`features.structured_outputs.result_validation_failure_reasons`.

JSON-schema result validation uses the same supported subset as strict tool
argument validation: `type`, `enum`, `const`, object `properties` / `required` /
`additionalProperties: false`, array `items` / `minItems` / `maxItems`, string
`minLength` / `maxLength`, and numeric min/max bounds. Unsupported validation
keywords are rejected before generation instead of being silently ignored;
annotation keys are accepted but ignored by validation. This is result
validation, not grammar-constrained decoding.

Grammar/guidance request fields are not currently supported. The capabilities
manifest reports `features.grammars.enabled=false` and lists known unsupported
fields such as `grammar`, `guided_json`, `guided_regex`, `guided_choice`,
`guided_grammar`, `guided_decoding_backend`, `guided_patch`, and
`guided_diff`; sending them returns HTTP 400 with
`error.code: "unsupported_parameter"` and `error.param` set to the field.

### Thinking / no-think controls

Chat requests accept common OpenAI/Qwen thinking controls:

- `reasoning_effort`: `none`/`off`/`disabled` pre-closes Qwen thinking;
  `minimal`, `low`, `medium`, `high`, `xhigh`, and `max` add bounded prompt
  hints for hidden-reasoning hard cap, visible-answer reserve, and soft-close
  window. Defaults are clamped to request `max_tokens`, the configured bounded
  chat default, and remaining admitted context when tokenizer/context metadata
  is available.
- `enable_thinking`: `false` pre-fills `<think>\n\n</think>\n\n` after the
  assistant header, matching Qwen no-think chat-template behavior.
- `chat_template_kwargs.enable_thinking`: accepted for Qwen-compatible clients;
  `chat_template_kwargs.reasoning_effort` is mapped to the same soft effort
  hints.
- Numeric budget aliases are accepted and normalized:
  `thinking_token_budget`, `chat_template_kwargs.thinking_budget`, and
  `thinking.budget_tokens` / `thinking.max_tokens` map to the effective hard
  thinking cap.
- Explicit hint fields are accepted on chat requests:
  `max_think_tokens`, `min_answer_tokens`, `hard_think_cap`,
  `soft_close_window`, `hard_close_message`, and `hard_close_sequence`.
  Numeric hard/min/soft hints are clamped to the same effective generation
  budget.
- Nested `thinking` or `reasoning` objects with `type`, `enabled`, or `effort`
  are accepted for OpenAI-compatible proxy variants; nested `thinking` and
  `reasoning` also accept the budget fields above plus `allow_unbounded`.
- `thinking.allow_unbounded=true` or `reasoning.allow_unbounded=true` disables
  the effort-derived default hard thinking cap when the request does not also
  set an explicit hard cap. Explicit `thinking.max_tokens`,
  `reasoning.max_tokens`, `*.budget_tokens`, `*.hard_think_cap`, top-level
  `thinking_token_budget`, or top-level `hard_think_cap` still set a hard cap
  and are enforced as usual.

Budget fields are prompt hints plus host-sampler thinking-budget policy today. The
server validates that any `hard_close_sequence` contains the parser-recognized
`</think>` marker. When the served engine exposes tokenization and an effective
`hard_think_cap` is present, chat generation lowers `hard_close_sequence` or the
default `</think>` marker into token ids, applies a host-sampler sparse
soft-close bias ramp to the first close token inside the soft window, suppresses
EOS while the row is still in reasoning/closing phase when an EOS token id is
available, and forces the full close sequence at the hard cap on host-sampled
PARO/GGUF paths. Qwen PARO/GGUF host-sampled paths resolve tokenizer EOS into
the sampler when the request does not supply `eos_token_id`. If a soft-biased
first close token is selected, the remaining close suffix is forced through
normal token selection so KV state remains consistent. If tokenization is
unavailable, generation remains prompt-hint-only rather than failing. Native GPU
sampler parity and speculative/MTP parity are not implemented yet. Generic
sampler `min_tokens` / `eos_token_id` still suppresses EOS for ordinary
generation independent of thinking-budget phase policy. Chat `count_tokens` and
`fit_context` diagnostics also honor app-local `session.id` transcript prefixes,
lower the configured close sequence into token ids, and return an initial
thinking-budget state plus `allow_unbounded=true` when that merged control is
active for harness/debug verification when tokenization is available. The
capabilities manifest exposes enforcement under
`features.reasoning_controls.token_budget_enforced`,
`hard_close_token_forcing`, `soft_close_bias`, `eos_suppression`,
`diagnostic_close_token_lowering`, and `diagnostic_initial_state`.

For pi, prefer `compat.thinkingFormat: "qwen"` with `reasoning: true` if you want
pi's thinking toggle to send `enable_thinking`; keep `supportsReasoningEffort`
set to `false` if you only want the Qwen flag and not OpenAI
`reasoning_effort`. A minimal pi `models.json` example is checked in at
`docs/examples/pi-agent/models.json`. Keep its `contextWindow` aligned with the
server's effective `/v1/hipengine/capabilities` context; the checked-in value is
a conservative W7900 example, not a model-family guarantee.

Validate the pi snippet, including the fields that keep pi's thinking UI
enabled, with:

```bash
python3 scripts/validate_pi_agent_models.py \
  --config docs/examples/pi-agent/models.json
```

Validate the same snippet against a running server capability manifest, and
optionally POST a small Qwen tool-call smoke request, with:

```bash
python3 scripts/validate_pi_agent_models.py \
  --config docs/examples/pi-agent/models.json \
  --base-url http://127.0.0.1:8000/v1 \
  --chat-smoke
```

The `--chat-smoke` check requires the response to finish with a parsed
`record_result` tool call whose JSON arguments set `result` to `"ok"`;
ordinary assistant text, raw `<tool_call>` markup, a missing `tool_calls`
payload, or the wrong tool argument fails validation.

### Local-agent config validation

A minimal OpenAI-compatible local-agent config is checked in at
`docs/examples/local-agent/openai-compatible.json`. It discovers the served
model id from `/v1/hipengine/capabilities`, uses deterministic Qwen-friendly
defaults (`temperature=0`, `reasoning_effort=none`), enables SSE usage and
hipEngine extension metadata, sets `timeout_ms`, sends tool schemas per request,
explicitly sends `session.commit="append_none"` as the current stateless
no-retain policy, and keeps stateful `session.id` out of the default streaming
payload plus intentionally unused tool-policy and grammar/guidance fields in
`do_not_send`.

When `--chat-smoke` is used and the config enables tools, the validator sends a
specific `record_result` tool choice and requires the server response to contain
that parsed tool call with JSON arguments that set `result` to `"ok"`. If tools
are disabled, the smoke only requires a valid chat completion response.

Known agent fields that are advertised as unsupported are rejected before
generation work starts. Stateless requests without a `session` object default
to no generated-tail retention, and `session.commit="append_none"` is accepted
as an explicit no-retain marker. Buffered non-streaming chat requests may set
`session.id`. With a session id, the default commit is `append_visible_only`;
`append_none`, `append_prompt_only`, and explicit debug `append_all` are also
accepted. The server stores an app-local visible transcript, strips parsed
`reasoning_content` from visible-only assistant commits, and reports the
effective `finish_details.cache_action`. Session requests do not mint
continuation handles. This is not resident KV reuse.
`session.id` on completions, streaming chat, `n>1`, continuation resumes,
unsupported `session.commit` modes, and other `session` payloads return HTTP 400
with `error.code: "unsupported_parameter"` and `error.param` set to the rejected
field. `continuation_id` is intentionally kept in the example config's
`do_not_send` list so local agents do not invent handles; a handle returned by
the server can be sent back on the supported resume path described above.

Authenticated `GET /v1/hipengine/sessions` returns metadata only: active session
count, storage type, `resident_state_reuse=false`, per-session id/message-count
timestamps, configured `max_active` cap, pending creation count, and active
continuation-handle count. It does not return transcript, prompt, generated, or
tool-result text. `DELETE /v1/hipengine/sessions/{session_id}` removes one
app-local transcript session and returns `deleted: true` or `false`.

Authenticated `GET /v1/hipengine/sessions/{session_id}/snapshot` exports a
versioned `hipengine.chat_session_snapshot.v1` snapshot for that app-local
transcript session. Unlike the metadata list, this response intentionally
includes the visible transcript messages so a client can save it. The snapshot
records served model id, backend, quant, storage, timestamps, and
`resident_state_reuse=false`; it does not include resident KV, tokenizer state,
or decode/sampling state. Authenticated
`POST /v1/hipengine/sessions/{session_id}/snapshot` restores the snapshot into
the same session id after validating schema, model id, backend, quant, storage,
message shape, text content parts, message string metadata, nested assistant
`tool_calls` objects, and valid JSON `function.arguments` strings. Incompatible
or corrupted snapshots fail before creating the session. Restoring a new session
is subject to the configured chat-session cap; when the cap is full, the server
returns `engine_busy` without creating partial session state.

Validate the config against a running server with:

```bash
python3 scripts/validate_local_agent_config.py \
  --config docs/examples/local-agent/openai-compatible.json \
  --base-url http://127.0.0.1:8000/v1
```

If `HIPENGINE_API_KEY` is set, the validator uses it automatically. Add
`--chat-smoke` to POST a small non-streaming chat request with the documented
tool schema shape and verify that the generated request does not include fields
listed in `do_not_send`.

### Error taxonomy

All JSON error responses use OpenAI-style `{"error": ...}` payloads with
`message`, `type`, `code`, and `param`. hipEngine also adds
`error.hipengine` when the error has a stable machine code. That extension
contains the canonical AGENTIC code, HTTP status, retryability, and
`legacy_code` when the OpenAI-facing `error.code` is kept for compatibility.
Streaming failures use the same error object inside the final SSE error chunk.
Request-body validation failures set `error.param` to the first field path
reported by FastAPI/Pydantic when available, for example `prompt` or
`messages.0.content`.

Clients should handle these canonical codes from `error.hipengine.code` on
error payloads. The same manifest also advertises `invalid_tool_call` because
strict tool result-validation emits it as a normal chat
`finish_details.reason`; it is not currently returned as an HTTP error payload.

| Code | Status | Retry | Current emission |
| --- | ---: | --- | --- |
| `unsupported_parameter` | 400 | no | Unsupported request field/value. Legacy `error.code` can be `unsupported_content_type` for non-text chat content parts. |
| `unsupported_feature` | 501 | no | Requested optional runtime feature is unavailable for the served model, for example tokenizer/counting diagnostics without tokenizer hooks. |
| `invalid_tool_call` | 400 | no | Normal chat `finish_details.reason` for strict tool result-validation failures; malformed tool JSON remains assistant text when strict validation is inactive. |
| `schema_violation` | 422 | no | Request body or server-side request validation errors; also normal `finish_details.reason` for invalid `response_format` or strict tool schema results. Legacy `error.code` is `validation_error` or `invalid_request`. |
| `invalid_continuation` | 400 | no | Unknown, consumed, wrong-endpoint, wrong-model, or otherwise incompatible `continuation_id`. |
| `continuation_expired` | 410 | no | Known `continuation_id` that expired before resume. |
| `context_overflow` | 400 | no | Prompt plus `max_tokens` exceeds admitted context; legacy `error.code` is `context_length_exceeded`; payload includes `error.fit_context` with max allowed/recommended `max_tokens` and overflow tokens. |
| `deadline_exceeded` | 408 | yes | `timeout_ms` or server default deadline expired. |
| `cancelled` | 499 | yes | Client disconnect/cancel observed at server await or stream boundaries. |
| `engine_busy` | 429 | yes | Generation queue or chat-session cap rejected the request before generation. |
| `model_unavailable` | 404 | no | Requested model is not served; legacy `error.code` is `model_not_found`. |
| `routing_failed` | 502 | yes | Reserved for future multi-model or multi-worker routing failures. |

The same table is advertised programmatically under
`/v1/hipengine/capabilities` as `errors`.

### Health and readiness

`GET /health` is a liveness probe. It returns only `status=ok` plus the served
model id and should not be used to infer that eager warmup has completed.

`GET /ready` is the readiness probe for local harnesses and process managers. It
returns HTTP 200 with `ready=true` after startup is ready, or HTTP 503 with
`ready=false` while startup is not ready. The payload includes non-sensitive
diagnostics for model loaded state, eager warmup completion, last startup timing,
configured/effective context, KV policy/capacity estimate, KV pool counters,
graph cache counters, selected backend/device environment, generation queue
depth/max-depth, active worker state, active backend request count/configured
cap, app-local session counts, stored message counts, configured chat-session
cap, pending session creations, and continuation-handle counts. It
intentionally omits prompts, generated text, tool results, and raw
request/response payloads.

## Diagnostics

Unsupported/unknown request fields, validation failures, and generation failures
log `REQUEST_FAILED` at warning or error level with status, code, parameter, and
message. To log full HTTP request and response payloads for local debugging, pass
`--debug` or set `HIPENGINE_DEBUG=1`:

```bash
HIPENGINE_DEBUG=1 hipengine serve --model /path/to/model
# or: hipengine serve --model /path/to/model --debug
```

Debug payload logs include prompts and generated text; do not enable them for
shared or sensitive deployments.

Replay artifacts are separately opt-in. Pass `--replay-dir /path/to/replays` or
set `HIPENGINE_REPLAY_DIR` to write finite JSON artifacts for failed HTTP
requests and normal strict result-validation failures such as
`invalid_tool_call`, `tool_required_not_satisfied`, or `schema_violation`.
Artifacts use `schema: "hipengine.replay.v1"` and include the request
method/path, redacted request JSON, prompt/tool-result hashes, served model id,
requested sampler and agentic control fields, seed fields, error or
result-validation metadata, finish details when available, completion/chat
prompt token counts when the engine is already loaded and supports counting,
explicit unavailable reasons otherwise, and a compact capability snapshot
including sampler/MTP compatibility, tokenizer-dependent tool/reasoning
controls, and cache/session support. Normal result-validation artifacts record
affected choice indexes and finish metadata, not generated assistant text.

The default `--replay-redaction hash` replaces every string value in the request
JSON and compact sampler/agentic-control payload with SHA-256 and length
metadata. `--replay-redaction none` stores raw strings and should only be used
in local, non-sensitive debugging sessions.

## Current limitations

- Streaming responses necessarily send HTTP `200 OK` once the SSE stream starts;
  runtime failures after that point are reported as SSE error chunks and
  `REQUEST_FAILED` logs, not a different HTTP status.
- Request deadlines and detected client disconnects are enforced at server
  await/iteration boundaries and at PARO/GGUF cooperative decode boundaries.
  Already-running GPU kernels or a captured graph replay are not preempted
  mid-call.
- HTTP generation requests route through the in-process generation batcher.
  Compatible queued prompts can coalesce into one prompt-list engine call, but
  true continuous decode, concurrent backend execution, and scheduler fairness
  remain later runtime work. Backend batch width can be capped with
  `--max-active-requests`; app-local chat transcript sessions can be capped with
  `--max-chat-sessions`. These are admission/batching limits, not resident KV
  fairness schedulers.
  Prometheus mode exposes `hipengine_generation_queue_depth`,
  `hipengine_generation_queue_max_depth`, and
  `hipengine_generation_worker_active` gauges plus
  `hipengine_generation_requests_active` and
  `hipengine_generation_requests_max_active` for backpressure monitors.
  `hipengine_generation_scheduler_fairness_policy_info` advertises the current
  FIFO-compatible batching policy. Request counters include completed, failed,
  rejected, and cancelled totals.
- PARO and GGUF sampling support `temperature`, `top_p`, `top_k`, `min_p`,
  `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
  `suppress_token_ids`, forced-token queues, `min_tokens` / `eos_token_id`,
  `seed`, and `n` through the host-logits compatibility path.
  Greedy-equivalent requests stay on each engine's graph/argmax fast path. PARO
  c=1 also has a default-off native GPU sampler route for supported sampled
  requests behind
  `HIPENGINE_QWEN35_NATIVE_SAMPLER=1`; c>N, GGUF, `top_logprobs`,
  suppress-token ids, min-token/EOS policy, and unsupported native filter
  combinations fall back to the host path.
- The capabilities manifest reports `sampling.speculative_mtp` with
  `compatibility_guard: "supports_speculative_mtp_sampling"`. Current MTP
  serving compatibility is greedy-fast only; `logit_bias`, penalties, token
  suppressions, min-token/EOS policy, token stops, pending forced-token queues,
  post-thinking forced-token queues, token-sequence completion repair,
  temperature sampling, and requested logprobs require autoregressive fallback.
  The manifest also includes `incompatible_conditions`, for example
  `temperature > 0`, so inert greedy `top_p` / `top_k` / `min_p` settings are
  not mistaken for MTP blockers.
- The capabilities manifest reports `parallelism.tensor_parallel` as disabled
  with `world_size=1`, `mode="single_process"`, no collective backend, and a
  stable unsupported-feature list for multi-GPU weight/KV sharding, collectives,
  graph capture, and cross-rank session snapshots.
- Non-text chat content parts are rejected.
- OpenAI `stop` strings are always post-trimmed; when tokenizer access is
  available, one-token stops lower to runtime `stop_token_ids` and multi-token
  stops lower to suffix-matched `stop_token_sequences` for early runtime
  termination. PARO c=1 native sampling checks the same metadata after token
  selection; native c>N and GGUF GPU paths still need parity.
- Tool calling uses Qwen-style prompt markup and output parsing; malformed
  `<tool_call>` JSON is treated as ordinary assistant text.
- Unknown top-level request parameters are rejected instead of silently ignored.
- Token `usage` and diagnostics are exact only when the served engine exposes
  tokenizer/counting hooks; unsupported models return explicit diagnostics
  errors or zero-count usage placeholders.
- Model-specific tokenizer chat templates are not public yet. Chat messages are
  rendered with a Qwen-style `<|im_start|>...<|im_end|>` text template.

See [`PLAN.md`](PLAN.md) for the server-optional architecture invariant and
[`TESTING.md`](TESTING.md) for public API/server validation rules.
