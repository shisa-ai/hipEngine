---
status: current
owns: Gemma 4 GGUF text inference, server usage, exercised coverage, and implementation limits.
---
# Gemma 4 text inference

Gemma 4 GGUF text inference runs through `hipengine.LLM` and `hipengine serve`
on the `hip_gfx1100` backend. The generator uses greedy decoding and BF16 KV
storage. Server chat uses the GGUF artifact's embedded Gemma template and
separates the thought channel into `reasoning_content`.

## Start a server

Set `MODEL` to your local Gemma 4 GGUF file. The exercised artifact is
`gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`, on an RX 7900 XTX. Artifact identity is
not an admission condition: loading depends on implemented architecture,
tensor geometry, quantization, and available resources.

```bash
MODEL=/path/to/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
hipengine serve --model "$MODEL" --backend hip_gfx1100 \
  --max-context-tokens 8192 --host 127.0.0.1 --port 8000 \
  --served-model-name gemma4
```

Request greedy chat completion:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of France?"}],"temperature":0,"max_tokens":96,"chat_template_kwargs":{"enable_thinking":true}}'
```

`enable_thinking` selects the embedded template's thinking mode. With thinking
enabled, the completion budget includes reasoning tokens as well as the answer;
a short budget can end before an answer appears. The response places the answer
in `content` and the thought text in `reasoning_content`, without Gemma's channel
or end-of-turn markers. Set `stream: true` for the server's SSE response format.
The adapter does not implement native incremental token streaming; SSE support
does not imply token-by-token delivery during generation.

## Implementation limits

- Sampling is greedy: use `temperature: 0`. Nontrivial sampling settings,
  penalties, logit bias, multi-token stop sequences, structured constraints,
  thinking budgets, and log probabilities are not implemented and fail with a
  named error instead of silently changing the request.
- KV storage is BF16. Other KV storage policies are not implemented for this
  generator.
- Tool-call response parsing is not implemented. A server request containing
  tools returns HTTP 400 `unsupported_parameter`.
- The attention implementation stores live-key scores in GPU shared memory.
  For the 26B artifact's 512-wide attention heads, the kernel's capacity ceiling
  is 15616 tokens; the default context is 8192. This is an implementation limit,
  not the model's advertised context length. Larger capacities require a tiled
  attention implementation and fail with a named shared-memory error. Memory
  availability can impose a lower practical capacity.
- This guide covers text inference, not image/audio input or speculative decode.

## Exercised coverage

The review exercised direct generation, a 793-token prompt crossing the default
512-token prefill block, server chat with thinking enabled and disabled, SSE
answer/reasoning separation, and an unsupported-tool request. The final live
server returned a clean Paris answer with end-of-turn stopping and HTTP 400 for
the unsupported tool request. Targeted Gemma regression tests cover request
controls, allocation cleanup, staging reuse, FFN and output-head geometry, and
attention reduction/shared-memory boundaries.

These are functional checks, not a model-quality evaluation or a published
throughput comparison. The [review closure](../../worklog/entries/20260923T091939.935292Z-lhl-gemma4-review-closure-474f1e.md)
records the commands, commits, and validation limitations.

The [optimization campaign](../campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md) defines
the planned same-GPU baseline, phase timing, profiling, and correctness gates.
No campaign performance results are published yet.
