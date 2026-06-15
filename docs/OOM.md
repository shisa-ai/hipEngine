# OOM / 24GB Full-Context Memory Analysis

Last updated: 2026-06-15

This is the living ledger for full-context OOM work on the 24GB-class GPU path
(AMD Radeon Pro W7900 GPU1 in this workstation, exposed as `HIP_VISIBLE_DEVICES=1`).
The goal is to make `hipengine serve` either start with a context length that can
actually serve production requests, or fail at startup with an actionable reason.

## TL;DR

- Legacy startup proved **resident KV allocation** plus a one-token raw prompt
  warmup. It did **not** prove every production chat request shape or long-prompt
  prefill scratch could run.
- Current startup now has a P0 fail-fast path: bounded raw warmup, max-prompt
  scratch probe (`prepare_request_scratch(max_prompt_tokens=context-1)`), bounded
  chat-shaped smoke through the generation batcher, GPU memory snapshots in
  `/ready`, and optional `--startup-min-free-mib` headroom enforcement.
- On the current tree, the exact command below can allocate model-max context,
  run the max-prompt scratch probe, and serve a one-row `hello` request with the
  server default token budget on the 24GB GPU:

  ```bash
  HIP_VISIBLE_DEVICES=1 hipengine serve \
    --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
    --kv-storage int8_per_token_head \
    --host 0.0.0.0
  ```

- The same run leaves about **2.0 GiB free** on GPU1 after startup. After the
  #88 streaming INT8 prefill-attention path removed the BF16 oracle workspace,
  the direct 262k scratch probe leaves **1.14 GiB free** at the live peak. This
  is still tight, but no longer spends the prior 0.5 GiB BF16 oracle transient.
- Streaming does **not** appear to add meaningful VRAM. It changes response and
  cancellation behavior, not retained device memory. The streamed hidden
  reasoning is emitted as separate `reasoning_content` chunks.
- We recently found and fixed two clear scratch regressions: verifier trunk
  buffers were accidentally sized to full prompt capacity, costing about
  **0.98 GiB at 128k**, and INT8 prefill attention kept a temporary BF16 oracle
  cache, costing **0.5 GiB at 262k**. More scratch/regression accounting remains
  open.
- The current auto-context estimator accounts for KV and persistent context
  metadata, but most lazy scratch is represented only by a flat 512 MiB reserve.
  That reserve is probably too optimistic for a hard fail-fast server policy.

## Full summary: how startup was tightened and 24GB full context was recovered

We fixed this in layers, moving from "can allocate KV" to "can serve this
context with the production path and measured transient scratch peak."

1. **Defined the right guarantee.** Startup readiness now means more than weight
   load and retained KV allocation. It must also prove a max admitted c=1 prompt
   can allocate prefill scratch without decoding to the output limit, prove a
   bounded chat-shaped request path, and expose/guard remaining GPU headroom.

2. **Added a bounded startup gate.** Startup now keeps the legacy raw warmup but
   also runs:
   - `prepare_request_scratch(max_prompt_tokens=context-1, max_new_tokens=0)`;
   - a bounded `hello` chat smoke through the generation batcher;
   - optional `--startup-min-free-mib` / `HIPENGINE_STARTUP_MIN_FREE_MIB` guard;
   - `/ready` startup checks and memory samples.

3. **Made memory logging actionable.** Per-stage memory samples are retained in
   `/ready` and debug logs, but normal startup emits one info-level summary:
   final free/used, peak used, min-free stage, total memory, and sample count.
   The scratch probe samples memory while transient scratch is still live, so the
   peak is not hidden by post-probe cleanup.

4. **Fixed probe correctness.** Two subtle issues made early probes misleading:
   - A tiny raw warmup prompt had re-resolved PARO prefill config to the
     small-prompt unchunked policy. The scratch probe now re-resolves prefill
     config for the probed max prompt length.
   - The first live-peak probe kept linear and full-attention workspaces live at
     the same time. Real long-context prefill releases workspace when the layer
     type changes, so the probe now mirrors that phase lifetime. This changed the
     128k peak from a pessimistic `0.29 GiB` free to `1.35 GiB` free.

5. **Found the true 24GB blocker.** With a phase-accurate probe, 262k still
   failed under the existing 1024/4096 chunk profile in the linear-attention
   phase at `linear_attn.tree_recurrent_state`. Manual probes showed:
   - 512-token chunks could pass but with effectively zero free memory;
   - 256-token chunks passed with about `0.61 GiB` free at peak;
   - the peak was `linear_prefill_scratch_live`, not full-attention scratch;
   - after linear scratch was compacted, the remaining INT8 BF16 prefill oracle
     became the next large transient until #88 replaced it with streaming INT8
     prefill attention.

6. **Recovered model-max startup on the 24GB card.** The auto prefill policy now
   selects a conservative 256-token chunk profile for 24GB-class devices
   (`<=26 GiB`) at model-max-ish contexts (`>=196608` tokens), while retaining
   the faster 1024/4096 profile for mid-context and larger-memory GPUs. With
   this policy, `--max-context-tokens 262144 --kv-storage int8_per_token_head`
   reaches ready on GPU1.

Current measured 262k/GPU1 state after #88:

```text
Direct scratch gate, context=262144, int8 KV:
linear_chunk_rows=256 full_chunk_rows=256 tree_rows=1 tree_saved=0.533 GiB int8_oracle=0 GiB
peak_stage=linear_prefill_scratch_live peak_used=22.846 GiB min_free=1.139 GiB scratch_probe_s=0.096
artifact=benchmarks/results/2026-06-15-gpu1-int8-prefill-streaming-scratch-262k.json
```

So the full 24GB use case is recovered, but it is still tight. This was not a
blanket speed-policy change: mid-context prompts and larger-memory GPUs retain
the existing 1024/4096 chunk profile. The 256-token profile is selected only for
24GB-class, model-max-ish prompts where the faster profile cannot fit. That
means broad speed regression is avoided, but near-max long-prompt prefill may be
slower; we have not yet run a full long-prompt throughput benchmark to quantify
that tradeoff. The latest direct scratch probe is still startup-class fast
(`scratch_probe_s=0.096s` after the one-token warmup), but there is still no
successful 24GB/262k long-prompt throughput baseline to compare against.

The next optimization targets are to reduce the remaining linear prefill scratch
high-water phase and compact persistent prefill metadata so 262k has a safer
live-peak margin and can potentially return to larger/faster chunks.

### Optimization gate protocol

For scratch optimizations, keep three gates visible before promoting the change:

1. **Short/mid speed smoke:** run `scripts/qwen35_paro_bench.py` at `512` and
   `4096` prompt tokens with the packed PARO model, `w4_paro`, graph replay,
   and `int8_per_token_head` KV. Treat these as regression smokes rather than
   rollup benchmarks unless repeated-run/correctness requirements are met.
2. **Full-context memory gate:** rerun the 24GB GPU1 startup or direct scratch
   probe at `262144` context and track `STARTUP_MEMORY` min-free/peak plus the
   scratch payload (`linear_prefill_chunk_rows`, `full_prefill_chunk_rows`,
   `int8_oracle_bytes`).
3. **Allocator accounting:** keep `prepare_request_scratch()` output with the
   artifact/log. It now also reports `linear_prefill_tree_state_rows`,
   `linear_prefill_tree_state_bytes`, `linear_prefill_tree_state_full_bytes`,
   and `linear_prefill_tree_state_saved_bytes` so the linear scratch reduction is
   explicit.

First linear scratch reduction implemented for measurement: ordinary c=1 prompt
prefill now reserves only one sentinel row for verifier/tree state buffers
(`tree_conv_state`, `tree_recurrent_state`, `tree_gdn_acc`). Those buffers are
needed by verifier tree/t-loop paths, but not by normal long-prompt prefill. The
verifier tree path still requires full tree rows and raises if handed compact
prefill scratch.

Expected model-config saving from that change:

- tree-state row size: `2.140625 MiB`;
- 24GB low-memory full-context chunks (`256` rows): `~0.533 GiB` saved at
  `linear_prefill_scratch_live`;
- larger-memory/manual-long chunks (`1024` rows): `~2.139 GiB` saved.

Post-change GPU0 smoke results (W7900 48GB, not the final 24GB gate; one run,
`decode_tokens=8`, artifact/logs under `/tmp/hipengine-prefill-gates-20260614-190158/`):

| Gate | Prefill tok/s | Decode tok/s | Tracked peak | HIP peak | Chunks |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/8 | 2345.653 | 105.565 | 18.138 GiB | 18.151 GiB | unchunked |
| 4096/8 | 2858.327 | 105.026 | 19.128 GiB | 18.226 GiB | `linear=moe=1024`, `full_query=4096`, `full_post=full_rope=1024` |
| 262k scratch smoke | n/a | n/a | n/a | live used 24.186 GiB on GPU0 | 1024/4096 chunks, `tree_rows=1`, `tree_saved=2.139 GiB`, `int8_oracle=0.5 GiB` |

Before #88, the 24GB/GPU1 exact direct scratch gate confirmed compact tree
scratch was active (`tree_rows=1`, `tree_saved=0.533 GiB`) and moved the
observed live peak from the linear scratch phase to the full-attention
scratch/oracle phase:

```text
GPU1 direct scratch gate, context=262144, int8 KV:
linear_chunk_rows=256 full_chunk_rows=256 tree_rows=1 tree_saved=0.533 GiB int8_oracle=0.5 GiB
peak_stage=full_prefill_scratch_live peak_used=23.320 GiB min_free=0.664 GiB scratch_probe_s=0.115
artifact=/tmp/hipengine-gpu1-24gb-gate-20260614-200239/scratch-262k-gpu1.json
```

That was only a modest net min-free improvement vs the prior `0.61 GiB` because
full-attention scratch plus the BF16 int8 prefill oracle became the high-water
phase.

#88 removes that oracle with a streaming INT8 prefill-attention kernel. An
allocation-only shrink was not safe for exact prefill: in chunked full-attention
prefill, chunk `[start, end)` must attend over K/V rows `[0, end)`, so the final
chunk of each full-attention layer needs the entire prompt's K/V image. The old
BF16 oracle was already one reused per-layer workspace slot, rounded only to the
KV block size. The existing `int8_per_token_head` HIP attention route was
decode-shaped (single query row), while a row-batched split-K route would require
`[rows, heads, splits, head_dim]` partials; at 256 rows and 262k context that
partial buffer would exceed the oracle by far. The new path instead reads
retained INT8 K/V plus per-token/head scales directly and performs an online
softmax reduction without BF16 K/V materialization.

Current #88 GPU1 direct scratch gate:

```text
GPU1 direct scratch gate, context=262144, int8 KV:
linear_chunk_rows=256 full_chunk_rows=256 tree_rows=1 tree_saved=0.533 GiB int8_oracle=0 GiB
peak_stage=linear_prefill_scratch_live peak_used=22.846 GiB min_free=1.139 GiB scratch_probe_s=0.096
artifact=benchmarks/results/2026-06-15-gpu1-int8-prefill-streaming-scratch-262k.json
```

## Current measurements

Measurements below are from GPU1 with a clean GPU before launch unless noted.
The current launch-knob/headroom refresh was run on 2026-06-15 under TheRock
ROCm 7.13 (`HIP version: 7.13.26162-1140233ffe`) on kernel
`Linux 7.0.10-1-cachyos`; artifact:
[`benchmarks/results/2026-06-15-gpu1-24gb-launch-knobs-headroom.json`](../benchmarks/results/2026-06-15-gpu1-24gb-launch-knobs-headroom.json).

### Startup max-prompt scratch probe (new P0 gate)

Direct probe command shape used for the measurements below:

```python
llm.prepare(max_sequence_length=N, sampling_params=SamplingParams(max_tokens=1, kv_storage="int8_per_token_head"))
llm.generate(("one two three four",), sampling)
llm.prepare_request_scratch(max_prompt_tokens=N - 1, max_new_tokens=0, sampling_params=sampling)
```

All runs used `HIP_VISIBLE_DEVICES=1`, model
`shisa-ai/Qwen3.6-35B-A3B-PARO-packed`, backend `hip_gfx1100`, quant `w4_paro`,
`kv_storage=int8_per_token_head`, and a clean GPU1 before launch.

| Context | Scratch probe | Peak live stage | Peak used | Min free | Evidence / chunk policy |
| ---: | --- | --- | ---: | ---: | --- |
| 65,536 | pass in 0.050s (2026-06-14) | — | — | — | `prefill_hidden_bytes=268,431,360`, `linear_prefill_chunk_rows=1024`, `full_prefill_chunk_rows=4096`, `int8_oracle_bytes=134,217,728` |
| 131,072 | pass in 0.171s | `full_prefill_scratch_live` | 21.395 GiB | 2.590 GiB | faster chunks `linear=1024`, `full=4096`; `tree_rows=1`, `tree_saved=2.139 GiB`, `int8_oracle=0.250 GiB` |
| 163,840 | pass in 0.200s | `full_prefill_scratch_live` | 22.057 GiB | 1.928 GiB | faster chunks `linear=1024`, `full=4096`; `tree_rows=1`, `tree_saved=2.139 GiB`, `int8_oracle=0.312 GiB` |
| 196,608 | pass in 0.220s | `full_prefill_scratch_live` | 22.746 GiB | 1.238 GiB | faster chunks `linear=1024`, `full=4096`; `tree_rows=1`, `tree_saved=2.139 GiB`, `int8_oracle=0.375 GiB` |
| 262,144 | pass in 0.096s (#88, 2026-06-15) | `linear_prefill_scratch_live` | 22.846 GiB | 1.139 GiB | low-memory/full-context auto chunks `linear=moe=full=256`; `tree_rows=1`, `tree_saved=0.533 GiB`, `int8_oracle=0 GiB`; artifact [`2026-06-15-gpu1-int8-prefill-streaming-scratch-262k.json`](../benchmarks/results/2026-06-15-gpu1-int8-prefill-streaming-scratch-262k.json) |

Important implementation detail: the first raw warmup prompt is tiny, and the
PARO session resolves prefill chunking based on the active prompt length. The
scratch probe must therefore re-resolve `prefill_config` for `max_prompt_tokens`
before allocating; otherwise it incorrectly reuses the tiny-prompt unchunked
policy and overstates OOM risk. The current probe does this and reports
`prefill_chunk_tuning` in `/ready`.

Phase-accurate probing matters: the first live-peak summary was too pessimistic
because the probe kept linear and full-attention prefill workspaces live together.
After mirroring the real long-context phase lifetime, `--max-context-tokens
128000` reports:

```text
STARTUP_MEMORY: final_stage=guard final_free=4.17 GiB final_used=19.81 GiB peak_stage=scratch_probe:linear_prefill_scratch_live peak_used=22.64 GiB min_free_stage=scratch_probe:linear_prefill_scratch_live min_free=1.35 GiB total=23.98 GiB samples=7
```

For full 262k on GPU1, the default 1024/4096 chunk profile still fails at
`linear_attn.tree_recurrent_state`. A memory-constrained auto profile for
24GB-class, model-max-ish contexts now selects 256-token chunks across
linear/MoE/full-attention prefill. With that profile, real startup reaches ready. The latest direct scratch gate
for the same admitted context reports:

```text
Direct scratch gate: peak_stage=linear_prefill_scratch_live peak_used=22.846 GiB min_free=1.139 GiB total=23.98 GiB
```

Conclusion / launch guidance:

- **Full 256Ki (`262144`) context is usable** on the 24GB GPU with
  `--kv-storage int8_per_token_head`, but it remains a tight full-context mode:
  the current direct probe leaves `1.139 GiB` free at the live
  `linear_prefill_scratch_live` peak.
- For short chat and 4K-prompt use cases that still want maximum context
  availability, the default/full-context launch is acceptable when startup passes
  the max-prompt scratch probe.
- For a safer 24GB service profile, cap `--max-context-tokens` to **163840**
  (about `1.93 GiB` live-probe free) or **131072** (about `2.59 GiB` live-probe
  free). Both retain the faster `1024/4096` chunk policy.
- **196608** is a middle ground but still fairly tight (`1.24 GiB` live-probe
  free) and should be treated as a diagnostic/advanced profile until more
  request-shape probes are recorded.
- #88 removed the temporary BF16 INT8 prefill oracle. The next memory target is
  again the linear prefill scratch high-water phase and any remaining persistent
  prefill metadata that affects the 262k margin.

### Legacy exact full-context server startup

Command:

```bash
HIP_VISIBLE_DEVICES=1 hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --kv-storage int8_per_token_head \
  --host 0.0.0.0
```

Startup log excerpt:

```text
LOAD_TIMING: phase=startup resident_prepare_s=23.940 max_context_tokens=262144
Config: model=shisa-ai/Qwen3.6-35B-A3B-PARO-packed served_model=shisa-ai/Qwen3.6-35B-A3B-PARO-packed max_context_tokens=262144 chat_default_max_tokens=4096 kv_storage=int8_per_token_head kv_scale_dtype=fp16 kv_scale_granularity=per_token_head eager_load=True
KVCache: storage=int8_per_token_head scale=fp16 max_context_tokens=262144 model_max_context_tokens=262144 allocatable_context_tokens=366592 requested_kv=2.52 GiB metadata=1.01 GiB total=3.53 GiB bytes_per_token=10320 usable=5.49 GiB reserve=0.50 GiB
WARMUP: prompt_tokens<=262144 max_tokens=1
LOAD_TIMING: model=shisa-ai/Qwen3.6-35B-A3B-PARO-packed engine_create_s=0.000 resident_prepare_s=23.940 warmup_s=0.365 startup_total_s=24.611
hipEngine is ready.
```

What this proves:

- weights load;
- a `262144`-token c=1 resident session can be allocated;
- the current one-token direct `engine.generate()` warmup can execute.

What this does **not** prove:

- the real `/v1/chat/completions` prompt renderer path;
- the server default `chat_default_max_tokens=4096` shape;
- streaming response behavior;
- `n>1` / request batching / concurrency;
- response-format, tool, logprob, sampler-processor, or speculative paths;
- long prompt prefill scratch at 128k/256k tokens.

### `hello`, non-streaming, explicit/default max token budget

Request shape:

```json
{
  "model": "shisa-ai/Qwen3.6-35B-A3B-PARO-packed",
  "messages": [{"role": "user", "content": "hello"}],
  "temperature": 0,
  "max_tokens": 4096
}
```

Result:

- HTTP 200.
- `finish_details.reason=eos`, `eos_token_id=248044`, `sampler_mode=greedy_fast`.
- Usage: `prompt_tokens=9`, `completion_tokens=199`, `total_tokens=208`.
- Visible `message.content` prefix: `"\n\nHello! How can I help you today? ��"`.
- Hidden `message.reasoning_content` length: 705 characters.
- GPU1 after request: about **1.996 GiB free** by `hipMemGetInfo` from a
  `HIP_VISIBLE_DEVICES=1` process; `rocm-smi` showed about 23.69 GB used out of
  25.75 GB device-reported total.

Omitting `max_tokens` produced the same result because the server default is
`chat_default_max_tokens=4096`.

### Streaming `hello`

Request shape was the same except `"stream": true` and no explicit `max_tokens`.

Result:

- HTTP 200.
- SSE started with a role chunk, then many `reasoning_content` chunks, then
  visible `content` chunks, then a final done event.
- Response body was about 47.8 KiB for this run.
- Device free memory before and after matched within measurement noise.

Conclusion: streaming itself is not the VRAM problem. It does not allocate a
second retained session or a large device-side response buffer. The same backend
prefill/decode path runs; SSE framing and response chunk buffering are host-side.

Streaming does affect operations:

- it exposes reasoning separately in the OpenAI-compatible `reasoning_content`
  field, so seeing reasoning chunks in pi output is expected if the client shows
  that field;
- it gives earlier user-visible progress;
- cancellation behavior must still be audited. A large non-streaming request
  timed out client-side while the server worker remained active until the test
  server was stopped. We should confirm streaming cancellation interrupts backend
  decode promptly.

### Very large `max_tokens`

Request shape:

```json
{
  "model": "shisa-ai/Qwen3.6-35B-A3B-PARO-packed",
  "messages": [{"role": "user", "content": "hello"}],
  "temperature": 0,
  "max_tokens": 262000
}
```

Result:

- The request was admitted because `prompt + max_tokens + 1` still fit the
  retained `262144` context.
- `curl` timed out after 90 seconds with no bytes because non-streaming waits for
  the whole generation.
- The server logged HTTP 499 (`request cancelled`).
- Memory did not spike meaningfully during the observation window.

This is not an OOM reproduction, but it is an admission/operations problem: a
non-streaming request can ask for nearly the full retained output budget and tie
up the worker. Startup smoke should not use such a huge output cap, and request
control should make cancellation stop backend decode promptly.

### `n=2` / c>N edge case

Request shape:

```json
{
  "model": "shisa-ai/Qwen3.6-35B-A3B-PARO-packed",
  "messages": [{"role": "user", "content": "hello"}],
  "temperature": 0,
  "max_tokens": 1,
  "n": 2
}
```

Result:

- HTTP 400: `compact c>N native prefill is not wired for int8_per_token_head retained KV`.
- GPU1 free memory jumped from about **2.15 GiB** to **6.09 GiB** after the
  failed request, which indicates the request path tore down the resident c=1
  session before returning the unsupported-parameter error.

This is not the user's reported immediate OOM either, but it is a real bug class:
unsupported c>N/int8 request shapes should be rejected before closing/replacing a
working resident session. Request batching (`generation_batch_window_ms > 0`) or
`n>1` must be included in fail-fast policy because a c=2 full-context session
would roughly double retained KV/context metadata.

## Reproduction discrepancy to resolve

The user's reported failure is an immediate OOM on a `hello` request at full
context. The measured request shapes above did **not** reproduce that exact OOM
on the current tree:

- non-streaming, omitted `max_tokens` / default `4096`: HTTP 200;
- non-streaming, explicit `max_tokens=4096`: HTTP 200;
- streaming, omitted `max_tokens` / default `4096`: HTTP 200;
- very large `max_tokens=262000`: no immediate OOM, but operationally bad
  because the non-streaming request tied up the worker until cancellation;
- `n=2`: rejected with HTTP 400, but incorrectly tore down the resident session.

Likely remaining explanations are: a different client request shape, request
concurrency/batching, non-greedy/logprob/tool/JSON options, allocator state after
a previous failed request, or a server process running a different tree/env. Any
future OOM report should capture the replay JSON, `/ready`, startup log, and GPU
free memory immediately before the request.

## Why legacy startup was too weak

The old server startup path in `hipengine/server/api.py` did this:

1. construct `SamplingParams(max_tokens=config.eager_load_max_tokens, temperature=0.0, top_p=1.0, ignore_eos=True, ...)`;
2. call `ensure_resident_context(...)`, which calls `engine.prepare(max_sequence_length=configured_or_auto, sampling_params=sampling)`;
3. validate the context budget for `config.eager_load_prompt`;
4. run `engine.generate((config.eager_load_prompt,), sampling)`.

Default `eager_load_max_tokens` is `1` and default `eager_load_prompt` is the raw
completion prompt `"one two three four"`.

The normal chat path does more:

1. render OpenAI messages to Qwen chat text (`<|im_start|>user...` and assistant prefix);
2. derive request `max_tokens`; for chat with no explicit value this is
   `chat_default_max_tokens=4096` clamped by remaining context;
3. route through the generation batcher / request-control wrapper;
4. run non-streaming `engine.generate` or streaming `engine.stream`;
5. split `<think>...</think>` into `reasoning_content` vs visible `content`;
6. optionally run tool/JSON/logprob/sampler-processing paths.

Therefore legacy startup could say ready when only the cheap retained allocation
and a narrow direct generation path had succeeded. The current P0 startup keeps
that raw warmup for compatibility but adds the max-prompt scratch probe and a
bounded chat-shaped smoke before setting readiness.

## Current memory model

### Accounted retained memory

For Qwen3.6/PARO at full model context with `int8_per_token_head` KV:

- `qwen35_paro_kv_bytes_per_token(...)` reports **10,320 bytes/token** for the
  retained full-attention KV payload plus fp16 per-token/head scales.
- At `262144` tokens, retained KV payload+scale is about **2.52 GiB**.
- Persistent context metadata, dominated by the prefill block table
  (`prefill_rows * blocks * int32`), is about **1.01 GiB**.
- Total accounted retained context memory is about **3.53 GiB**.
- The capacity estimator subtracts a flat reserve from free memory before KV
  allocation. The current default is `HIPENGINE_KV_CAPACITY_RESERVE_MIB=512`.

Approximate accounted retained-context totals for c=1/int8/fp16-scale:

| Context tokens | KV payload+scale | context metadata | total accounted | delta vs 262k |
| ---: | ---: | ---: | ---: | ---: |
| 262,144 | 2.52 GiB | 1.01 GiB | 3.53 GiB | baseline |
| 196,608 | 1.89 GiB | 0.57 GiB | 2.46 GiB | ~1.07 GiB less |
| 131,072 | 1.26 GiB | 0.25 GiB | 1.51 GiB | ~2.02 GiB less |

This explains why 128k feels much safer: it frees roughly two GiB before any
other scratch reductions.

### Under-accounted or lazy memory

The 512 MiB reserve is only a coarse proxy. The following can allocate after or
outside the cheap capacity estimate:

- per-layer `RuntimeWorkspace` allocations in decode states;
- `prefill_workspace` and `_prefill_scratch_state` allocations during prompt
  prefill;
- `prefill_hidden_buffer`, sized by prompt rows and released after prefill;
- temporary BF16 prefill oracle key/value tensors used when retained KV is int8;
- MoE/router/grouped prefill scratch;
- sampler buffers, especially non-greedy/full-vocab/top-p/logprob paths;
- speculative/DFlash/MTP buffers and metadata;
- HIP/rocBLAS/JIT runtime allocations and allocator fragmentation;
- any c>N/batched request that requires `max_batch_size > 1`.

Some of these are intentionally released after prefill (`_restore_decode_scratch_after_prefill()`),
but they still need enough transient headroom at the moment they are allocated.

## Known regression / suspect ledger

### Fixed: verifier trunk full-prefill allocation

A recent DFlash/MTP verifier-trunk restoration accidentally allocated two
verifier hidden buffers at full prompt prefill capacity:

```text
2 * max_sequence_length * hidden_size * fp16
```

For `128000` context and hidden size 2048 that cost about **0.98 GiB** resident
memory on GPU1. The verifier entrypoints reject `rows > max_batch_size`, so the
trunk pair only needed verifier-row capacity. The fix right-sized the trunk pair
to `max_batch_size * hidden_nbytes` each.

Recorded validation after the fix:

- `llm.prepare(max_sequence_length=128000, kv_storage=int8_per_token_head)` left
  about **4.209 GiB free** instead of **3.232 GiB**.
- A 4096-token prompt smoke succeeded and ended at about **4.168 GiB free**.

### Still suspect: persistent prefill metadata size

At 262k, context metadata is about **1.01 GiB**, largely from the full
`prefill_block_table` materialization. If earlier 24GB runs did not keep this
full table resident, or used a more compact/chunked representation, this alone
could explain a large part of the regression.

Follow-up: audit when `prefill_block_table` became resident/full-sized and
whether it can be chunked, generated on device, or represented as a smaller
uniform mapping for dense full-context policies.

### Still suspect: speculative/MTP resident buffers

The verifier trunk was one MTP-related leak, but the speculative tree still has
other resident buffers and scratch caches. We need a per-buffer inventory for a
262k session and a comparison with the pre-MTP baseline.

Follow-up: add a resident memory dump that reports each persistent buffer name,
shape, dtype, bytes, and owner (`KV`, `linear_state`, `decode_scratch`,
`prefill_metadata`, `MTP/speculative`, `sampler`, etc.).

### Still suspect: sampler/logprob paths

Greedy-fast `hello` works. Non-greedy sampling, top-p, logprobs, top-logprobs,
forced-token/grammar processors, and full-vocab GPU sampler paths can allocate
additional buffers. Startup does not currently test them.

Follow-up: run a request matrix at 262k and 128k:

- greedy chat;
- `temperature>0` sampled chat;
- `top_p<1`;
- `logprobs/top_logprobs`;
- structured JSON/tool requests;
- streaming/non-streaming variants.

### Still suspect: unsupported c>N request teardown

The `n=2` int8 request returned 400 but also freed the resident c=1 session. This
should be rejected before any session close/reallocation. It is also a reminder
that full-context c>N is a different memory class.

## Streaming analysis

Streaming is new enough that it deserves separate tracking, but current evidence
does not point to it as a VRAM consumer.

What streaming shares with non-streaming:

- same retained resident session;
- same chat prompt rendering and `SamplingParams` derivation;
- same prefill and decode kernels for each generated token;
- same `<think>` splitting logic, just incremental via `_ReasoningSplitter`.

What streaming changes:

- response framing is SSE instead of one JSON body;
- chunks are emitted as soon as they are decoded;
- hidden reasoning is emitted under `delta.reasoning_content`, not mixed into
  `delta.content`;
- visible output can look odd in clients that concatenate role/reasoning/content
  fields without labeling them;
- cancellation and backpressure are more visible operationally.

Open questions for streaming:

1. Does client disconnect stop backend decode immediately for long generations?
2. Does the streaming path preserve stop-token/role-boundary behavior exactly
   like non-streaming for Qwen chat templates?
3. Do streaming tool-call buffered paths allocate extra host memory for long tool
   JSON outputs? This should not be VRAM, but it affects server memory.

## Desired fail-fast policy

For `hipengine serve`, readiness should mean:

1. resident session allocation succeeded;
2. max admitted c=1 prompt prefill scratch can be allocated without decoding to
   the output limit;
3. a bounded production-shaped chat request succeeded;
4. the server retained enough configured free-memory headroom after that smoke;
5. unsupported production shapes are rejected before they can destroy a working
   resident session;
6. the selected context is reported clearly in `/ready` and startup logs.

Current P0 startup now:

- keeps the raw warmup as a cheap decode-path check;
- probes `max_prompt_tokens=context-1`, `max_new_tokens=0`, `max_batch_size=1`
  through the backend `prepare_request_scratch` hook;
- runs a bounded `hello` chat smoke through the generation batcher using
  `max_tokens=eager_load_max_tokens` (default `1`) so startup never decodes until
  the context limit;
- records GPU memory snapshots before prepare, after prepare, after raw warmup,
  after scratch probe, after chat smoke, and at the optional guard point;
- fails startup when the scratch probe or `--startup-min-free-mib` guard fails.

Still open: auto-reduce context when `--max-context-tokens` is omitted and the
scratch probe rejects the KV-only auto-selected context. Today that case fails
fast with an actionable error instead of silently advertising an unsafe server.

A conservative initial threshold for the W7900 24GB target should be at least
**2 GiB** post-smoke free, and probably **3 GiB** while the scratch inventory is
incomplete. Current 262k/hello measured about 2.0 GiB free, so a 3 GiB policy
would select a lower context or fail instead of advertising a marginal 262k
server.

## Re-optimization plan for 24GB / 256k

### P0: make failures early and actionable

- Done: add startup max-prompt scratch probe (`--startup-scratch-probe`, default
  on) that allocates long-prompt prefill scratch without decoding to the limit.
- Done: add bounded startup chat smoke (`--startup-chat-smoke`, default on).
- Done: add `HIPENGINE_STARTUP_MIN_FREE_MIB` / `--startup-min-free-mib` for
  required post-smoke headroom.
- Done: include startup checks and memory snapshots in `/ready`.
- Open: reject unsupported c>N/int8 request shapes before closing/reallocating
  the current resident session.
- Open: ensure cancellation interrupts long backend decode promptly.
- Open: auto-reduce context when KV-only auto-selection passes but scratch probe
  fails.

### P1: build an exact memory ledger

- Add a debug endpoint or script that dumps all resident device buffers by owner,
  shape, dtype, and bytes.
- Add a per-request allocation trace mode around prefill/decode/sampler paths.
- Compare current 128k and 262k ledgers with the pre-MTP/pre-sampler baseline.
- Track high-water memory for:
  - startup prepare;
  - startup raw warmup;
  - startup chat smoke;
  - 4096-token prompt prefill;
  - streaming and non-streaming decode;
  - sampler/logprob variants.

### P2: claw back 256k headroom

- Compact or eliminate resident `prefill_block_table` for uniform dense policies.
- Keep speculative/MTP verifier buffers lazy or behind an explicit enabled flag
  unless speculative decode is active.
- Audit sampler scratch so greedy-fast and exact top-p only retain what they need.
- Release decode scratch before large prefill whenever overlap is unsafe, and
  restore it after prefill.
- Further reduce chunked linear-attention transient footprint so 262k on 24GB has
  more than the current `~0.61 GiB` live-peak margin.
- Tune default auto-context reserve by measured scratch high-water, not a fixed
  512 MiB guess.

## Useful reproduction commands

Direct max-prompt scratch probe without launching a web server:

```bash
HIP_VISIBLE_DEVICES=1 uv run python3 - <<'PY'
from hipengine import LLM, SamplingParams
model = "shisa-ai/Qwen3.6-35B-A3B-PARO-packed"
sampling = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0,
                          ignore_eos=True, kv_storage="int8_per_token_head")
llm = LLM(model, backend="hip_gfx1100", quant="w4_paro")
try:
    ctx = llm.prepare(max_sequence_length=262144, sampling_params=sampling)
    llm.generate(("one two three four",), sampling)
    print(llm.prepare_request_scratch(max_prompt_tokens=ctx - 1,
                                      max_new_tokens=0,
                                      sampling_params=sampling))
finally:
    gen = llm._get_text_generator()
    close = getattr(gen, "close", None)
    if callable(close):
        close()
PY
```

Full-context startup + default chat request:

```bash
HIP_VISIBLE_DEVICES=1 hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --kv-storage int8_per_token_head \
  --host 0.0.0.0

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"shisa-ai/Qwen3.6-35B-A3B-PARO-packed","messages":[{"role":"user","content":"hello"}],"temperature":0}' \
  | python -m json.tool
```

Streaming variant:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"shisa-ai/Qwen3.6-35B-A3B-PARO-packed","messages":[{"role":"user","content":"hello"}],"temperature":0,"stream":true}'
```

GPU1 free memory from a separate process:

```bash
HIP_VISIBLE_DEVICES=1 python - <<'PY'
from hipengine.core.hip import get_hip_runtime
rt = get_hip_runtime()
free, total = rt.mem_get_info()
print('free_GiB', round(free / 1024**3, 3))
print('used_GiB', round((total - free) / 1024**3, 3))
PY
```
