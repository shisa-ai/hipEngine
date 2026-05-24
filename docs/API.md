# OpenAI-Compatible Server API

Last updated: 2026-05-18

hipEngine ships a thin optional FastAPI layer that adapts OpenAI-style requests
to the torch-free `hipengine.LLM.generate()` library API. It is installed only
with the `server` extra and is intentionally serialized today because the
current runnable Qwen/PARO path is still single-request / `c=1`.

## Install

```bash
pip install -e ".[server]"
```

## Run

```bash
python -m hipengine.server \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed \
  --quant w4_paro \
  --served-model-name qwen-paro \
  --host 127.0.0.1 \
  --port 8000
```

`--model` accepts a local filesystem path or a Hugging Face model ID that is
already present in the local HF cache. hipEngine resolves IDs with local cache
lookups only; it does not download weights during server startup.

After installation, the console script is equivalent:

```bash
hipengine-server --model /path/to/model --served-model-name qwen-paro
```

The server defaults to `--backend auto`, which maps exact `gfx1100`/`gfx1151`
ROCm detections to `hip_gfx1100`/`hip_gfx1151`. Unknown HIP targets warn and
select `cpu_reference` where a CPU implementation exists; nearby targets such as
`gfx1101`/`gfx1102` can force a backend with `--backend hip_gfx1100` or
`HIPENGINE_BACKEND=hip_gfx1100` after local validation.

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
| `POST /v1/completions` | Built in | Text prompt(s) to `LLM.generate()`. Supports `stream=true` as one server-sent event chunk plus `[DONE]`. |
| `POST /v1/chat/completions` | Built in | Renders text-only messages to a Qwen-style prompt and calls `LLM.generate()`. Supports `stream=true` as one server-sent event chunk plus `[DONE]`. |

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

## Current limitations

- Request execution is serialized with an in-process lock. Continuous batching,
  concurrent decode, and scheduling fairness are later runtime work.
- `n > 1`, `logprobs`, and non-text chat content parts are rejected.
- Token `usage` is exact only if the injected engine exposes `count_tokens`; the
  default public `LLM` path currently reports zero-count placeholders until
  tokenizer accounting is exposed.
- Model-specific tokenizer chat templates are not public yet. Chat messages are
  rendered with a Qwen-style `<|im_start|>...<|im_end|>` text template.

See [`PLAN.md`](PLAN.md) for the server-optional architecture invariant and
[`TESTING.md`](TESTING.md) for public API/server validation rules.
