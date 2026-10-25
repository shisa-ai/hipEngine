---
status: current
owns: Public compact-DMS entry points, supported compositions, configuration and request lifecycle.
---
# Compact DMS serving

Dynamic Memory Sparsification (DMS) uses an external sidecar to decide which
attention history to retain. Qwen hybrid GGUF serving exposes two DMS selections
with different supported compositions. They share model weights, session setup
and request lifecycle; each request owns its compact retention state.

For implementation ownership, see [KV/DMS integration](KV-DMS-INTEGRATION.md).
For other subsystem boundaries, start at [Architecture](../ARCHITECTURE.md).

## Entry points and supported compositions

| Selection | Python / CLI | Supported composition |
| --- | --- | --- |
| Typed configuration | `dms=DMSConfig(...)` / `--dms-metadata` | One active request, BF16 compact storage, prefix cache off, speculation off; explicit prefill and decision modes |
| Metadata-path integration | `dms_metadata_path=...` / `--dms-metadata-path` | One active request, BF16 or implemented compact INT8 codec; DMS-owned prefix snapshots and native NextN/MTP integration where the target/provider supports them; layer-outer prefill |

Do not supply both selectors. LLM and server configuration reject that ambiguity
before model construction. They do not merge sidecar paths or choose one silently.

The metadata-path prefix setting uses DMS-owned compact snapshots, not dense
radix-page sharing. The presence of a sidecar does not make every model, codec or
draft provider compatible; explicit unsupported requests fail by capability.
The typed entry point's restrictions remain explicit and are not removed by
sharing the implementation.

## Typed Python configuration

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
`dense_pool` is its default prefill implementation. The loader validates sidecar
geometry and metadata before execution. A matching geometry permits execution;
trained quality evidence is separate from runtime capability.

For typed Python calls, omitted concurrency, KV storage, prefix-cache and
speculation settings resolve to one active request, BF16, prefix-cache off and
speculation off. Explicit incompatible settings raise an error rather than
silently using dense retention. Other generators without a DMS adapter fail
explicitly. Session creation uses the generator's shared context selection and
configuration path; omitting a context no longer bypasses automatic sizing.

## Metadata-path Python configuration

```python
from hipengine import LLM, SamplingParams

llm = LLM(
    "model.gguf",
    max_sequence_length=8192,
    dms_metadata_path="sidecar/dms_metadata.json",
    kv_storage="bf16",
    prefix_cache="off",
    speculative_mtp_serving="off",
)
try:
    output = llm.generate(["Explain gravity briefly."], SamplingParams(max_tokens=64))
finally:
    llm.close()
```

This selects the existing metadata-path composition. Prefix reuse and speculative
serving may be requested through their normal options where implemented; neither
is implied by the example. Compact output capacity accounts for sidecar retention
policy, and a store-bounded output limit is reported as a length finish.

## HTTP server

Typed configuration:

```bash
hipengine serve --model model.gguf --dms-metadata sidecar/dms_metadata.json \
  --max-active-requests 1 --max-context-tokens 8192 --kv-storage bf16 \
  --prefix-cache off --speculative-mtp-serving off
```

Use `--dms-prefill-mode layer_outer` for layer-at-a-time typed prefill. The typed
server requires incompatible prefix and speculation options to be disabled
explicitly at startup. The metadata-path selection uses `--dms-metadata-path`
instead; do not combine the flags. HTTP request shapes remain unchanged.

## Execution and limits

- The existing resident scheduler owns admission, deadlines, token streaming and
  cancellation. DMS introduces no second request queue.
- Context and short-prompt capability refusals occur before rows are published,
  so subsequent valid requests can use the same healthy service. DMS capture
  requires bulk prefill with at least the model's convolution width in tokens.
  Shorter prompts are rejected by name; this is an implementation precondition,
  not a benchmark or sidecar allowlist.
- The adapter collects scheduler chunks and invokes the compact session's
  full-prompt prefill once. Cancellation is checked around that call, not between
  its internal layers.
- The admitted request owns its session lease. Cancelling a queued request does
  not close an idle or neighboring session. Completion/cancellation closes the
  compact session rather than resetting and reusing its retention history.
- Decode uses the compact backend. The dense fixed-address graph implementation
  does not cover changing compact extents and retention decisions.
- Typed runner observability reports `retention.policy: dms`, storage, prefill
  mode and the active compact backend snapshot; request metadata also reports
  decision mode. Metadata-path serving additionally reports its DMS prefix and
  speculative activity through the existing route diagnostics.

Lifecycle tests establish routing and ownership, not sidecar quality on untested
prompts. `no_evict` separates compact-attention behavior from learned retention
when diagnosing output differences. A synthetic zero-weight sidecar can test
loading and cleanup, but must not be represented as a trained retention model.
