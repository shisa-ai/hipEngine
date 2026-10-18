---
status: current
owns: Public single-request compact DMS configuration and lifecycle.
---
# Compact DMS serving

Dynamic Memory Sparsification (DMS) uses an external sidecar to decide which
attention history to retain. The Qwen GGUF generator exposes its compact BF16
store through `LLM` and the HTTP server. Model weights remain shared; each
request owns its compact retention state and releases it on completion,
cancellation, or failure.

## Python

```python
from hipengine import DMSConfig, LLM, SamplingParams

llm = LLM(
    "model.gguf",
    max_sequence_length=8192,
    dms=DMSConfig("sidecar/dms_metadata.json"),
)
try:
    output = llm.generate(["Explain gravity briefly."], SamplingParams(max_tokens=64))
finally:
    llm.close()
```

`DMSConfig` accepts `metadata_path`, `prefill_mode` (`dense_pool` or
`layer_outer`), and `decision_mode` (`sidecar` or the `no_evict` control).
`dense_pool` is the default prefill implementation. `layer_outer` uses the
existing layer-at-a-time compact packing implementation. The sidecar loader
validates tensor geometry and metadata before execution.

For Python calls, omitted concurrency, KV storage, prefix-cache, and speculation
settings resolve to one active request, BF16, prefix-cache off, and speculation
off. Explicit incompatible settings raise an error rather than silently running
dense retention. Other model generators without a DMS adapter fail explicitly.

## HTTP server

```bash
hipengine serve --model model.gguf --dms-metadata sidecar/dms_metadata.json \
  --max-active-requests 1 --max-context-tokens 8192 --kv-storage bf16 \
  --prefix-cache off --speculative-mtp-serving off
```

Use `--dms-prefill-mode layer_outer` to select layer-at-a-time prefill. HTTP
request shapes are unchanged. The server requires the incompatible prefix and
speculation settings to be disabled explicitly at startup.

## Execution and limits

- The existing resident scheduler owns admission, deadlines, token streaming,
  and cancellation. DMS does not introduce a second request queue.
- The adapter collects scheduler prefill chunks and invokes the compact
  session's full-prompt prefill once. Cancellation is checked before and after
  that device call, not between its internal layers.
- Decode is eager. The dense fixed-address graph implementation does not cover
  changing compact extents and retention decisions.
- `retention` in runner observability reports `policy: dms`, `storage: bf16`,
  prefill mode, and the active compact backend snapshot. Per-request execution
  metadata also identifies retention and decision mode.
- One active request, BF16 storage, no radix prefix reuse, and no speculative
  serving are the supported composition. Queued requests reuse shared weights
  but get a fresh request session. Compact INT8 and speculative experimental
  primitives are not exposed by this public adapter.

The lifecycle tests verify routing and ownership; they do not establish quality
for a sidecar on untested prompts. Use `no_evict` to separate compact-attention
numerics from learned retention when investigating output differences.
