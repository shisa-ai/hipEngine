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
`max_tokens=auto`, meaning the remaining admitted context
(`max_context_tokens - prompt_tokens - 1`).

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
| `POST /v1/completions` | Built in | Text prompt(s) to `LLM.generate()`. For a single prompt with `n=1` and `echo=false`, `stream=true` uses token/chunk SSE from `LLM.stream()` when available; multi-prompt, `n>1`, and echo streaming fall back to buffered SSE. |
| `POST /v1/chat/completions` | Built in | Renders text-only messages to a Qwen-style prompt and calls `LLM.generate()` / `LLM.stream()`. Supports token-level `stream=true` SSE for `n=1`; `n>1` streaming returns buffered per-choice chunks. `<think>` spans are separated into `reasoning_content` (non-streaming) or `delta.reasoning_content` chunks (streaming). |

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

### Streaming usage chunks

Both completion endpoints accept OpenAI-compatible `stream_options`. Set
`"stream_options": {"include_usage": true}` with `"stream": true` to request a
final SSE payload with `choices: []` and `usage` before `data: [DONE]`.

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

- Request execution is serialized with an in-process lock. Continuous batching,
  concurrent decode, and scheduling fairness are later runtime work.
- PARO and GGUF sampling support `temperature`, `top_p`, `top_k`, `min_p`,
  `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias`,
  `seed`, and `n` through the host-logits compatibility path. Greedy-equivalent
  requests stay on each engine's graph/argmax fast path.
- `logprobs` and non-text chat content parts are rejected.
- Unknown top-level request parameters are rejected instead of silently ignored.
- Token `usage` is exact only if the injected engine exposes `count_tokens`; the
  default public `LLM` path currently reports zero-count placeholders until
  tokenizer accounting is exposed.
- Model-specific tokenizer chat templates are not public yet. Chat messages are
  rendered with a Qwen-style `<|im_start|>...<|im_end|>` text template.

See [`PLAN.md`](PLAN.md) for the server-optional architecture invariant and
[`TESTING.md`](TESTING.md) for public API/server validation rules.
