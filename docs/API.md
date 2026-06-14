# OpenAI-Compatible Server API

Last updated: 2026-06-14

hipEngine ships a thin FastAPI layer that adapts OpenAI-style requests to the
torch-free `hipengine.LLM.generate()` library API. Server dependencies are
installed by default, and execution is intentionally serialized today because
the current runnable Qwen/PARO path is still single-request / `c=1`.

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
| `GET /health` | Built in | Unauthenticated health/model probe. |
| `GET /v1/models` | Built in | Returns the single served model id. |
| `GET /v1/hipengine/capabilities` | Built in | Authenticated hipEngine manifest for served model/config, context defaults, tokenizer availability, streaming/logprobs/tool/reasoning support, sampling execution/native/MTP status, request-timeout support, cache/session status, and unsupported fields. |
| `POST /v1/hipengine/tokenize` | Built in | Tokenizes raw text with the served tokenizer when available. |
| `POST /v1/hipengine/detokenize` | Built in | Decodes token ids with the served tokenizer when available. |
| `POST /v1/hipengine/count_tokens` | Built in | Counts raw text or rendered chat messages after applying the server chat template, tool markup, and thinking controls. |
| `POST /v1/hipengine/fit_context` | Built in | Reports prompt tokens, effective max tokens, required context, and reject/truncation policy using the same admission arithmetic as generation. |
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

### Streaming usage and hipEngine metadata

Both completion endpoints accept OpenAI-compatible `stream_options`. Set
`"stream_options": {"include_usage": true}` with `"stream": true` to request a
final SSE payload with `choices: []` and `usage` before `data: [DONE]`.

Set `"stream_options": {"include_hipengine": true}` to request hipEngine
extension metadata on SSE payloads. Each payload gets a top-level `hipengine`
object with `metadata_version`, `event`, and `timing.elapsed_ms`. Choice chunks
also get `choices[].hipengine.phase` (`think`, `answer`, `tool_call`, or
`done`) when a phase is known. Final choice chunks include the same
`finish_details` under `choices[].hipengine.finish_details`, and usage chunks
mirror usage under `hipengine.usage`.

Cache, prefill/TTFT, decode-rate, budget-pressure, and exact per-phase token
metadata are omitted until the runtime exposes those signals.

### Finish details

Completion and chat choices include a hipEngine extension field,
`finish_details`, next to the OpenAI-compatible `finish_reason`. The extension
always contains `reason` and may include `eos_token_id`, `stop_sequence`,
`length_limit`, `deadline_exceeded`, `cancelled`, `forced_close`,
`synthetic_tokens`, `reasoning_tokens`, `answer_tokens`, `tool_call_tokens`,
`structured_tokens`, `budget_pressure`, `cache_action`, and `sampler_mode`.

`finish_reason` remains the coarse OpenAI value for compatibility. For example,
backend `reason: "eos"` is exposed as `finish_reason: "stop"` with
`finish_details.reason: "eos"`, while backend `reason: "length"` maps to
`finish_reason: "length"`. Tool-call parsing reports
`finish_reason: "tool_calls"` and `finish_details.reason: "tool_calls"`.
Streaming responses include `finish_details` on the final choice chunk;
ordinary delta chunks are unchanged.

PARO/GGUF detailed generation reports basic backend finish details for EOS,
token stops, stop sequences, length limits, and sampler mode when those signals
are available from the generation loop.

When a backend does not yet provide structured finish metadata, the server emits
the conservative fallback `{"reason": finish_reason}`.

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
`data: [DONE]`.

Client disconnects are checked at the same server await/stream iteration
boundaries. Detected disconnects cancel queued work and use structured
`{"reason": "cancelled", "cancelled": true}` finish details when cancellation
can still be surfaced as an error payload.

### Tool calling

`POST /v1/chat/completions` accepts OpenAI-style `tools` and `tool_choice` for
local-agent clients such as pi. hipEngine injects a Qwen-style tool block into
the rendered chat prompt and expects the model to emit tool calls as:

```text
<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>
```

The server converts those blocks to OpenAI-compatible `message.tool_calls` in
non-streaming responses or `delta.tool_calls` chunks in streaming responses, with
`finish_reason: "tool_calls"`. Prior assistant `tool_calls` and `role: "tool"`
messages are also replayed into the prompt as `<tool_call>` and
`<tool_response>` blocks so multi-turn tool loops can continue.

### Thinking / no-think controls

Chat requests accept common OpenAI/Qwen thinking controls:

- `reasoning_effort`: `none`/`off`/`disabled` pre-closes Qwen thinking; `low`,
  `medium`, and `high` add soft instructions to keep `<think>` content bounded
  and close `</think>` before the final answer or a tool call.
- `enable_thinking`: `false` pre-fills `<think>\n\n</think>\n\n` after the
  assistant header, matching Qwen no-think chat-template behavior.
- `chat_template_kwargs.enable_thinking`: accepted for Qwen-compatible clients;
  `chat_template_kwargs.reasoning_effort` / `thinking_budget` are mapped to the
  same soft effort hints.
- nested `thinking` or `reasoning` objects with `type`, `enabled`, or `effort`
  are accepted for OpenAI-compatible proxy variants.

For pi, prefer `compat.thinkingFormat: "qwen"` with `reasoning: true` if you want
pi's thinking toggle to send `enable_thinking`; keep `supportsReasoningEffort`
set to `false` if you only want the Qwen flag and not OpenAI
`reasoning_effort`.

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

## Current limitations

- Streaming responses necessarily send HTTP `200 OK` once the SSE stream starts;
  runtime failures after that point are reported as SSE error chunks and
  `REQUEST_FAILED` logs, not a different HTTP status.
- Request deadlines and detected client disconnects are enforced at server
  await/iteration boundaries. They fail the HTTP/SSE request promptly, but
  already-running backend calls or GPU kernels are not preempted until those
  calls return.
- Request execution is serialized with an in-process lock. Continuous batching,
  concurrent decode, and scheduling fairness are later runtime work.
- PARO and GGUF sampling support `temperature`, `top_p`, `top_k`, `min_p`,
  `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
  `seed`, and `n` through the host-logits compatibility path. Greedy-equivalent
  requests stay on each engine's graph/argmax fast path. PARO c=1 also has a
  default-off native GPU sampler route for supported sampled requests behind
  `HIPENGINE_QWEN35_NATIVE_SAMPLER=1`; c>N, GGUF, `top_logprobs`, and
  unsupported native filter combinations fall back to the host path.
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
