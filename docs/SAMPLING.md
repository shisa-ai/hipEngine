# Sampling Design

Last updated: 2026-06-18

This document defines how hipEngine should grow from the current greedy-only
Qwen3.5/PARO and GGUF generation paths to normal server/library sampling
without weakening the torch-free runtime, plugin-registry boundaries, or retained
greedy performance path.

## Current state

The public API and server now expose the functional host-sampling surface for
PARO and GGUF. PARO native GPU sampling is the default for the supported scoped
route, and unsupported native shapes fail closed to host logits sampling:

- `hipengine.llm.SamplingParams` carries the functional sampler fields needed
  for host sampling: `temperature`, `top_p`, `top_k`, `min_p`, penalties,
  `logit_bias`, token-level stops, KV policy knobs, `seed`, and `row_seeds`.
- `hipengine.generation.registry.GenerationRequest` mirrors those canonical
  fields, and `hipengine.generation.sampling` owns validation, sampler planning,
  row seed derivation, and CPU/NumPy token selection.
- `hipengine.server.api` accepts OpenAI-style `temperature`, `top_p`, `top_k`,
  `min_p`, penalties, `logit_bias`, `seed`, `stop`, `n`, non-streaming
  `logprobs` / `top_logprobs`, buffered streaming logprobs, and streaming
  `stream_options.include_usage`. Tokenizable `stop` strings are lowered to
  runtime single-token stops or multi-token stop sequences; all stop strings
  still use response post-trimming. Unknown top-level request extras are rejected
  instead of silently ignored, and rejected/failed requests log `REQUEST_FAILED`
  diagnostics for local server bring-up.
- `Qwen35ParoOneTokenGenerator` now keeps greedy-equivalent requests on the
  graph/argmax fast path and routes non-greedy or processed-argmax requests
  through a correctness-first host-logits sampler.
- `Qwen35GGUFBringupGenerator` keeps greedy-equivalent requests on its graph path
  and routes non-greedy or processed-argmax requests through the shared
  host-logits sampler using resident-session logits readback.
- `Qwen35ParoResidentSession._sample_device_from_hidden(...)` remains the
  device-resident greedy suffix. It has been split internally into logits
  projection plus argmax selection so `_sample_from_hidden(...)` can copy FP32
  logits to host for the functional sampler when configured.
- c>N PARO sampled requests use scheduler-owned row state and native packed
  prefill. Fully GPU-sampler-eligible rows use serial per-slot native GPU
  sampling by default when the resident session exposes the native row sampler;
  `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` disables that route for rollback or
  bisection. GGUF still samples prompt rows serially because its bring-up path
  has no c>N resident scheduler.
- `PerRowSamplingParams` / `SamplerParamsBlock` carry the canonical scalar
  sampler metadata and logit-bias rows for scheduler/native-sampler shape.
  `ResidentBatchScheduler` now owns `RowSamplingState` rows, exposes them in
  decode-work order, and updates generated-token history through
  `record_generated` / speculative accept paths. PARO sampled batches clone
  those states per physical slot for host token selection while keeping the
  scheduler as the persistent history owner.
- `lm_head.hip` has a row-wise top-k helper capped at `k <= 8`, useful for
  drafter/verifier diagnostics but not enough for normal user sampling.
- `hipengine/kernels/hip_gfx1100/sampling/sampler.hip` adds standalone
  GPU-smoked row-wise native sampler pieces over FP32 logits: finite-clamping
  logits processors for logit bias, repetition/presence/frequency penalties,
  suppress-token ids, and min-token/EOS suppression; full-vocab `top_k=0`
  temperature sampling; bounded `1 <= top_k <= 64` sampling; and
  correctness-first exact full-vocab `top_p`/`min_p` filtering.
  They support per-row temperature, per-row seed, counter-based RNG, selected
  token id/logprob, retained-count reporting, full-vocab `top_k=0`
  top-logprob metadata, and optional bounded-candidate logprobs. Supported PARO
  c=1 temperature requests route through these kernels
  by default with tiny selected-id/logprob/logit readbacks and decode-state
  telemetry of `full_vocab_logits_d2h=false` plus `logits_d2h_bytes=0`;
  supported PARO c>N sampled batches can also route each physical slot through
  the same native sampler state and report no full-vocabulary logits readback.
  A synthetic resident-session smoke covers full-vocab, full-vocab top-logprobs,
  top-k+processor, bounded top-k probability filters, and top-p route dispatch
  against CPU references. Unsupported PARO route shapes still use the host
  sampler and report
  `sampler_fallback_reason="native_gpu_unsupported_request"` while native
  sampling is enabled. GGUF remains host-sampled; it reports the native
  unsupported fallback reason only when `HIPENGINE_QWEN35_NATIVE_SAMPLER=1`
  explicitly requests native fallback metadata for that non-native route. Host
  decode-state telemetry marks
  `full_vocab_logits_d2h=true` and reports the per-token full-vocab logits
  vector byte count when the vocabulary/logits width is known.

The original user-visible failure for non-greedy Qwen3.5/PARO and GGUF requests
is fixed for the host-logits path. Remaining implementation work is true batched
native GPU sampler c>N, GGUF native integration, and native parity for remaining
dynamic processor/response shapes.

## Native sampler promotion scope

The 2026-06-16 promotion makes native GPU sampling the default only where the
guarded implementation already exists:

- PARO c=1 sampled requests with `temperature > 0` that satisfy
  `supports_native_gpu_sampling()`;
- PARO scheduler-owned c>N sampled requests when every active row satisfies
  `supports_native_gpu_sampling()` and the resident session exposes
  `configure_native_sampler_rows`;
- selected-token logprobs, logit bias, repetition/presence/frequency penalties,
  full-vocab temperature, bounded `1 <= top_k <= 64`, exact full-vocab
  `top_p`/`min_p` with `top_k=0`, bounded top-k `top_p`/`min_p` filters,
  full-vocab native `top_logprobs` for `top_k=0`, bounded native
  `top_logprobs` when `top_logprobs <= top_k <= 64`,
  suppress-token ids, min-token/EOS suppression, and post-selection stop token
  ids/sequences.

Promotion blockers closed in this pass:

- default route no longer requires `HIPENGINE_QWEN35_NATIVE_SAMPLER=1`;
- `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` remains an explicit rollback opt-out;
- capabilities now advertise `sampling.native_gpu.default_path=true` and the
  disable env;
- generator tests cover default c=1, default serial per-slot c>N, explicit
  opt-out host fallback, and unsupported native-shape fallback metadata.

Remaining native-sampler gaps are not blockers for the scoped default because
the planner falls back before native execution:

- true batched c>N token selection;
- GGUF native sampler integration;
- `top_logprobs > top_k` for bounded top-k requests;
- forced-token queues, sequence-completion repair, JSON object close forcing,
  thinking-budget dynamic processors, and `top_k > 64`;
- broader retained benchmarks/profiler coverage beyond the first W7900 promotion
  smoke.

### Native follow-up triage

The supported native route is promotion-ready because it avoids full-vocabulary
logits D2H and keeps unsupported shapes on the host fallback. The remaining
native-vs-greedy gap is expected but has a few concrete places to tighten:

- **Current closest-to-greedy mode:** bounded `top_k` sampling. It adds one
  sampler kernel after projection and avoids the host full-vocab copy. This is
  the mode to use for native/greedy performance comparisons.
- **Expected slower native modes:** `top_k=0` full-vocab temperature sampling
  still scans the vocabulary during selection, and exact full-vocab
  `top_p`/`min_p` uses a correctness-first retained-set construction. They are
  native correctness paths, not yet the performance shape to compare against
  greedy argmax.
- **Avoidable scalar traffic:** `_sample_from_hidden_native(...)` still reads
  selected token/logprob/logit to host and then writes the selected id/value back
  into the legacy `lm_out_index` / `lm_out_value` buffers. Those buffers are
  part of the resident-session contract today, but the writeback should move
  into the sampler kernels or become conditional once sampled AR no longer needs
  the legacy lm-head outputs for compatibility tests.
- **Avoidable per-step uploads:** temperature, top-p/min-p, seed, and compact
  processor metadata are uploaded through the Python/ctypes path each sampled
  step. Request-constant scalar buffers can be cached at sampler configuration
  time; processor history uploads are still expected for penalties until row
  history is resident on device.
- **Processor overhead is real:** logit bias and repetition/presence/frequency
  penalties add a full-vocab processor kernel before sampling. They are covered
  for native correctness, but they should not be treated as greedy-adjacent
  performance until the processor ABI and row-history uploads are tightened.
- **Graph replay remains greedy-only:** sampled requests need mutable RNG,
  row-history, and finish/constraint decisions. Do not block native promotion on
  graph replay until that state is device-resident enough to make replay useful.

Unsupported native cases, ranked by likely ease and payoff:

| Priority | Case | Why first / blocker |
| --- | --- | --- |
| Done | Native `top_logprobs` for full-vocab and bounded `top_k <= 64` | Planner/runtime/server/GPU tests now keep `top_k=0` full-vocab `top_logprobs` and bounded `top_logprobs <= top_k <= 64` on the native route. Bounded requests with `top_logprobs > top_k` still fall back. |
| Done | `suppress_token_ids` and `min_tokens`/EOS suppression | The native processor kernel now applies suppress offsets and row step-indexed EOS masking after bias/penalties. Planner/runtime/server/GPU tests keep these shapes on the native route. |
| Done | Combined `top_k` with `top_p`/`min_p` | The bounded top-k native sampler now applies host-order probability filters over the selected candidate list, renormalizes before sampling, and keeps bounded `top_logprobs` aligned with the retained set. |
| P2 | Forced-token queues when already pending | A per-step forced-token fast path can emit the queued token and metadata without host logits sampling. It needs careful interaction with sequence repair, JSON close, and thinking-budget queues. |
| Done | Request-constant native scalar buffer caching | Implemented for request-constant native sampler scalars and logit-bias buffers; the retained profiler artifact showed warmed native sampler H2D copies/token drop to zero. Dynamic history and step-index metadata remain per-step. |
| P3 | `top_k > 64` | Raises register/shared-memory pressure in the bounded sampler. Needs a measured reason before increasing the cap. |
| P3 | GGUF native sampling | Useful eventually, but GGUF lacks the PARO resident scheduler/runtime shape that made scoped native promotion easy. |
| P3 | True batched c>N native token selection | Requires row-aware projection/sampling into `batch_lm_out_index` plus generated-token equality and replay readiness. This is larger than sampler-only work. |
| P3 | Faster full-vocab top-p/min-p selector | Performance work after correctness parity; current exact path is acceptable for scoped native coverage but not the greedy-comparison target. |

## Hardware lane for this work

Standalone sampler development may use **GPU1**, the local AMD Radeon RX
7900 XTX (`gfx1100`), with explicit environment selection:

```bash
HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. <command>
```

Use GPU1 for sampler unit smoke tests, profiler experiments, and native-kernel
bring-up when the W7900 is occupied. Retained promotion benchmarks should use
the project default W7900/gfx1100 lane, for example `HIP_VISIBLE_DEVICES=0`
when GPU0 is the W7900. Any retained performance claim from GPU1 must record:

- hardware: AMD Radeon RX 7900 XTX, `gfx1100`;
- selected device: `HIP_VISIBLE_DEVICES=1`;
- model, quant, prompt/decode shape, KV policy, sampler mode, and exact command;
- correctness gate and whether the path used host logits readback or native GPU
  sampling.

The 7900 XTX has less VRAM than the W7900, so full-model smoke commands on GPU1 should
prefer short contexts and explicit KV policy. If a model/checkpoint fits only on
W7900 for a given shape, keep GPU1 validation at the sampler-unit or synthetic
logits level and record the memory blocker instead of weakening the test.

## Goals

1. Preserve the current greedy fast path as the default for greedy-equivalent
   requests.
2. Support normal text-generation parameters through the library API and
   OpenAI-compatible server without silently ignoring fields.
3. Keep the runtime torch-free; CPU-side fallback math may use NumPy because it
   is already an optional/light dependency in the project plan, but not torch.
4. Make fixed-seed sampling deterministic across runs for a fixed engine build,
   prompt, and sampling parameter set.
5. Let c=1 and c>N use the same request-level sampling model even when the first
   implementation samples rows serially.
6. Keep correctness gates explicit: greedy-equivalent requests must match the
   existing argmax path exactly; stochastic requests need deterministic fixture
   checks at fixed seeds plus distribution sanity tests where appropriate.
7. Make the server error policy explicit: supported sampler fields are plumbed;
   unsupported sampler fields return a clear `unsupported_parameter` response.

## Non-goals for the first functional milestone

- Grammar / JSON-schema constrained decoding.
- Beam search or `best_of` ranking.
- Prompt-token scoring for completion `echo+logprobs`, live per-token streaming
  logprobs without buffering, and broad native-GPU parity outside the scoped
  PARO sampler route.
- Speculative sampling / probability-ratio acceptance. That belongs with the
  relaxed/speculative documents because it changes the accept contract.
- Matching another engine's exact random stream. hipEngine should define its own
  deterministic stream and document it.
- Broad native GPU sampling parity outside the scoped PARO default. The current
  native route is promoted only for the covered request shapes; unsupported
  processors and response shapes still fall back to host logits sampling.

## Parameter contract

The canonical parameter set should live in `SamplingParams`, flow into
`GenerationRequest`, and be lowered to per-row sampler state. Server request
models should either populate these fields or reject unsupported aliases.

| Field | Current state | Target behavior | Initial complexity |
| --- | --- | --- | --- |
| `max_tokens` | Public/server/runtime | Supported. | Low |
| `ignore_eos` | Public/server/runtime | Supported for EOS; stop-token rows need deeper integration. | Low |
| `temperature` | Public/server/runtime | `<= 0` means deterministic argmax after processors; `> 0` enables host multinomial sampling for PARO. | Medium |
| `top_p` | Public/server/runtime | Nucleus filtering after processors; inert for plain `temperature <= 0` argmax. | Medium |
| `top_k` | Public/server/runtime + scheduler partial | Keep highest `k` tokens before sampling; `0` means disabled. | Medium |
| `min_p` | Public/server/runtime | Optional probability floor relative to max probability. | Medium |
| `repetition_penalty` | Public/server/runtime + scheduler partial | HF-style penalty using prompt + generated history. | Medium |
| `presence_penalty` | Public/server/runtime | Subtract once for tokens already present. | Medium |
| `frequency_penalty` | Public/server/runtime | Subtract proportional to token count. | Medium |
| `logit_bias` | Public/server/runtime | Token-id keyed bias map before filtering. | Medium |
| `suppress_token_ids` | Public/server/runtime + scheduler state | Set listed token logits to `-inf` after bias/penalties and before argmax/sampling. Supported by the scoped PARO native GPU route. | Medium |
| `min_tokens` / `eos_token_id` | Public/server/runtime + scheduler state | Suppress `eos_token_id` until `RowSamplingState.step_index >= min_tokens`; `min_tokens > 0` requires `eos_token_id`. Supported by the scoped PARO native GPU route. | Medium |
| `seed` / `row_seeds` | Public/server/runtime | Stable row RNG seed; `n > 1` rows diverge deterministically. | Low/Medium |
| `stop` strings | Server post-trim + token lowering | Keep post-trim; lower one-token stops to `stop_token_ids` and multi-token stops to suffix-matched `stop_token_sequences`. | Medium |
| `stop_token_ids` / `stop_token_sequences` | Public/runtime + scheduler state | Token stops finish PARO/GGUF host-sampled rows; PARO c=1 and serial per-slot c>N native sampling check the same stop metadata after selection. | Medium |
| `logprobs` / `top_logprobs` | Public/server/runtime for host-logits paths | Return selected logprob and optional top candidates; completion `echo+logprobs` shifts generated-token offsets after a null-logprob prompt prefix. Streaming logprobs use a buffered detailed-generation SSE path while ordinary non-logprob streams remain live token/chunk streams. | High |

Compatibility rule: if `temperature <= 0` and the request has no active logit
processors (`logit_bias`, penalties, suppressions, min-token/EOS policy,
forced-token queues, etc.), `top_p` and `top_k` do not change the selected token
because the top logit remains included. Those requests should use the greedy
fast path rather than failing merely because a client sent `top_p=0.95` with
`temperature=0`.

Speculative/MTP compatibility is stricter until target verification can run the
same processed-logit policy as autoregressive generation.
`supports_speculative_mtp_sampling()` returns true only for `GREEDY_FAST`
requests; `speculative_mtp_sampling_blockers()` reports the fields that require
AR fallback today, including `logit_bias`, penalties, suppress-token ids,
min-token/EOS policy, token stops, pending forced-token queues, post-thinking
forced-token queues, token-sequence completion repair, `temperature > 0`, and
requested logprobs. The resident scheduler applies this guard before emitting
speculative target-verification work, so rows that need processed logits cannot
silently enter the raw-argmax MTP path. Successful scheduler verify work and
plans carry `target_sampling_policy="raw_target_top1"`,
`processed_target_verification=false`, and
`compatible_sampling_modes=("greedy_fast",)` until a processed target verifier
exists. The public capabilities manifest exposes both the flat blocker field
list and `sampling.speculative_mtp.incompatible_conditions`, so clients can
distinguish conditional blockers such as `temperature > 0` from inert greedy
filters like `top_p`, `top_k`, and `min_p`.

### Server/API mapping

| External request field | Library field | Notes |
| --- | --- | --- |
| `max_tokens` | `SamplingParams.max_tokens` | Chat `None` can still mean remaining context. |
| `temperature` | `SamplingParams.temperature` | Validate finite and non-negative. `0` is greedy-equivalent unless processors are active. |
| `top_p` | `SamplingParams.top_p` | Validate `0 <= top_p <= 1`. `0` should retain one token or be rejected consistently; prefer retain-one semantics inside sampler. |
| `top_k` | `SamplingParams.top_k` | Included in request schemas and `_sampling_key`; `0` disables. |
| `min_p` | `SamplingParams.min_p` | Public hipEngine extension; `0` disables. |
| `repetition_penalty` | `SamplingParams.repetition_penalty` | Default `1.0`; positive only. |
| `presence_penalty` | `SamplingParams.presence_penalty` | Default `0.0`. |
| `frequency_penalty` | `SamplingParams.frequency_penalty` | Default `0.0`. |
| `logit_bias` | `SamplingParams.logit_bias` | Token-id keyed map initially; token-string aliases can be a later tokenizer feature. |
| `suppress_token_ids` | `SamplingParams.suppress_token_ids` | Token-id list; each listed logit is suppressed before argmax/sampling. |
| `min_tokens` / `eos_token_id` | `SamplingParams.min_tokens` / `.eos_token_id` | `min_tokens > 0` suppresses the configured EOS token until that many generated steps have been accepted; requires `eos_token_id`. |
| `seed` | `SamplingParams.seed` | Base seed for row derivation. |
| `n` | prompt expansion + `row_seeds` | Server expands rows and derives deterministic per-row seeds. |
| `stop` | server trim + token lowering | Tokenizable stops lower to token IDs/sequences for early host-path termination and remain post-trimmed for response consistency. |
| `logprobs` / `top_logprobs` | `SamplingParams.logprobs` / `.top_logprobs` | Completions use OpenAI `logprobs: N`; chat uses `logprobs: true` plus optional `top_logprobs: N`. Non-streaming and buffered streaming responses include selected token logprobs and optional top candidates. |
| unknown sampler extras | reject explicitly | Pydantic still preserves extras for OpenAI compatibility, but `_validate_generation_request()` rejects them with `unsupported_parameter` before generation work. |

## Runtime architecture

### 1. Canonical sampler state

Add a small sampler module, for example `hipengine.generation.sampling`, with:

- `SamplingParams` validation helpers shared by library and server.
- `SamplingMode` or `SamplerPlan`:
  - `GREEDY_FAST`: current graph/argmax path, no active processors.
  - `PROCESSED_ARGMAX`: deterministic argmax after penalties/biases.
  - `HOST_LOGITS_SAMPLE`: correctness-first CPU/NumPy sampler over device logits.
  - `GPU_SAMPLE`: retained native sampler once kernels exist.
- `RowSamplingState`:
  - request id and row index;
  - row seed;
  - generated step index;
  - prompt token history;
  - generated token history;
  - count table for penalties;
  - stop-token rows if available.
- `SampleResult` fields shared with existing autoregressive results:
  - `token_id`;
  - `token_text`;
  - selected `logit`;
  - optional `logprob` and `top_logprobs`;
  - sampler mode used for observability.

`RowSamplingState` belongs to generation/session code, not model plugins. Model
plugins provide tokenizer metadata and special tokens; sampler policy remains
runtime/generation infrastructure.

### 2. Split projection from token selection

The current resident sessions combine projection and argmax in helpers named
`_sample_*`. Refactor internally into two conceptual steps while preserving the
existing public behavior:

1. `project_logits_from_hidden(hidden, row)`:
   - final RMSNorm;
   - cast;
   - LM-head projection;
   - returns or fills an FP32 logits buffer.
2. `select_token(logits, row_state, params)`:
   - greedy argmax, processed argmax, host sampling, or GPU sampling.

For `GREEDY_FAST`, keep the current fused sequence and graph replay. For all
other modes, disable multi-token decode graph replay initially because token
selection needs host-side state and/or kernels that are not graph-safe yet.

For PARO c=1, the natural extraction point is inside
`Qwen35ParoResidentSession`:

```text
_sample_device_from_hidden(hidden)
  final_rmsnorm -> fp16_to_bf16 -> w8a16_lm_head -> argmax
```

Split this into:

```text
_project_logits_device_from_hidden(hidden)  # leaves `lm_logits` valid
_select_from_logits(...)                    # argmax, host sampler, or GPU sampler
```

The greedy graph path can keep calling the original device-resident sequence or a
thin wrapper that performs projection plus argmax. Host sampling should call the
projection helper, copy `lm_logits` to host, and then run the sampler.

### 3. Sampler plan selection

Sampler planning should be pure and testable. A request is `GREEDY_FAST` only
when all of these are true:

- `temperature <= 0`;
- no active `logit_bias`;
- `repetition_penalty == 1.0`;
- `presence_penalty == 0.0`;
- `frequency_penalty == 0.0`;
- no suppress-token ids, min-token/EOS policy, forced tokens, token stops, or
  other token-level constraints beyond plain EOS/ignore-EOS;
- no requested response logprobs.

A request is `PROCESSED_ARGMAX` when `temperature <= 0` but one or more logit
processors are active. It needs full-logits processing but no multinomial draw.

A request is `HOST_LOGITS_SAMPLE` when `temperature > 0` and native GPU sampling
is not explicitly selected and validated. This is the first functional sampled
path.

A request is `GPU_SAMPLE` only after the native sampler kernels pass the GPU
sampler gates below and the selected parameter combination is supported. If a
request uses a field not supported by the native sampler, fall back to
`HOST_LOGITS_SAMPLE` unless the user asked for native-only behavior.

### 4. Processor order

Use one documented order across CPU and GPU paths:

1. Start from FP32 logits.
2. Apply `logit_bias`.
3. Apply repetition, presence, and frequency penalties using prompt + generated
   token history.
4. Apply `suppress_token_ids` and `min_tokens` / `eos_token_id` suppression.
5. If a forced token is pending, emit it through the normal decode path and
   record forced metadata.
6. If `temperature <= 0`, choose argmax over processed logits.
7. If `temperature > 0`, divide logits by temperature.
8. Apply `top_k` filter.
9. Convert to probabilities with max-subtracted softmax.
10. Apply `top_p` / `min_p` filters, always retaining at least one token.
11. Renormalize and draw one token from the row RNG.
12. Append the token to row history and update counts.

For `top_p`, sorting is by descending probability with deterministic tie-break on
lower token id. For argmax, ties also pick the lower token id to match existing
argmax kernels.

### 5. RNG policy

The host implementation should use a stable, explicitly seeded generator owned
by hipEngine, not Python's process-random `hash()` or global `random` module.
A simple first implementation can use NumPy `Generator(PCG64(seed))` per row,
with the derived `row_seed` recorded in metadata/tests.

The future GPU sampler should use a counter-based stream keyed by
`(row_seed, step_index, row_index)` so graph capture and replay do not depend on
mutable host RNG state. Exact CPU/GPU random-stream equality is nice but not a
requirement for first GPU promotion; fixed-seed determinism and distribution
sanity are required.

### 6. EOS and stop handling

EOS remains a token-level finish condition:

- if `ignore_eos` is false and the selected token is an EOS token, finish the row;
- if `ignore_eos` is true, EOS is just another sampled token.

Stop strings are always server-side trimmed for response consistency. When
served tokenizer access is available, the server also lowers them to token stops:

- one-token stops become `stop_token_ids`;
- multi-token stops become `stop_token_sequences`;
- PARO/GGUF host-sampled rows finish as soon as a generated token suffix matches
  a lowered sequence;
- PARO c>N sampled batches and future GPU sampler paths consume the same
  scheduler stop metadata before claiming token-level stop parity.

## Host logits sampler path

The first user-facing implementation should be correctness-first and host-backed:

1. Run the existing model forward/prefill/decode to produce the final hidden row.
2. Run final RMSNorm, cast, and W8A16 LM-head projection into the resident FP32
   logits buffer.
3. Copy one logits row (`vocab_size * sizeof(float32)`) to host.
4. Apply processors and sample on CPU/NumPy.
5. Copy or set the chosen token id as the next decode input.
6. Update row history and finish flags.

This path deliberately gives up graph replay for sampled requests at first. The
important invariant is that `GREEDY_FAST` is untouched, while sampled requests no
longer fail at the guard.

Host sampler implementation notes:

- Use `float64` for softmax accumulation if it simplifies numerical stability;
  store source logits as FP32.
- Clamp non-finite logits to `-inf` except when all logits are non-finite, which
  should raise a clear error.
- Always retain at least one candidate after `top_k`, `top_p`, and `min_p`.
- Record the selected token's original processed logit, sampled logprob, and
  requested top-logprob summary in `SampleResult` for public response plumbing.
- Keep CPU sampler code independent from Qwen/PARO so GGUF can reuse it.

## Native GPU sampler path

Native GPU sampling is a performance track, not a blocker for functional support.
It should reuse the same `SamplerPlan` and processor order.

### GPU kernel decomposition

A practical first native path can be split into small kernels:

1. **Logits processors:** apply supported logit bias, penalties, suppressions,
   and min-token/EOS masks row-wise over the full vocab. The standalone S6
   processor kernel covers finite-clamping, logit bias, repetition penalty,
   presence penalty, frequency penalty, suppress offsets, and per-row
   step-indexed EOS suppression from compact per-row inputs; generation routing
   can optimize the compact-list ABI later.
2. **Top-k candidate selection:** select a bounded `k` candidate set. The
   legacy `lm_head` top-k helper remains capped at `k <= 8`; the S6 sampler
   bring-up adds a standalone `top_k <= 64` candidate path for user-sampling
   smoke tests. The standalone `top_k=0` path skips candidate truncation and
   samples over all finite vocab logits.
3. **Temperature + softmax over candidates:** compute probabilities for the
   retained candidate set. The standalone S6 kernels cover this for FP32 logits,
   per-row temperatures, and both full-vocab and bounded top-k modes.
4. **RNG + sample:** counter-based RNG produces one uniform draw per row/step;
   cumulative probabilities select the token. The standalone S6 kernel uses a
   deterministic SplitMix64-derived row/step stream.
5. **Output write:** write token id, logprob, and optional top-logprobs summary;
   update device token scalar/batch token vector for the next decode step.

This covers temperature + top-k efficiently. Exact top-p over the full vocab now
has a standalone correctness-first S7 kernel that sorts by repeated full-vocab
selection and matches retain-one semantics on boundary fixtures. It is routed
for supported scoped PARO native requests; any performance-oriented sort/select
replacement still needs separate validation.

### Native sampler validation commands

Use the project default W7900 lane for promotion validation when available:

```bash
HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 \
  python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"

HIP_VISIBLE_DEVICES=0 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. \
  python3 -m pytest tests/test_gpu_sampler_kernel.py -q
```

GPU1 / RX 7900 XTX is still acceptable for standalone sampler smoke and
profiler loops when the W7900 is occupied:

```bash
HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 \
  python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"

HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. \
  python3 -m pytest tests/test_gpu_sampler_kernel.py -q
```

The test above is the current standalone native sampler unit/integration
coverage plus synthetic resident-session route smoke for the default PARO native
sampler. For profiler evidence, prebuild JIT libraries before `rocprofv3` and
run only a narrow sampling smoke under the profiler. Do not wrap a parent
harness that spawns nested Python children.

## c>N and server batching

c>N support should reuse the same sampler state, but it does not have to be
fully vectorized at first:

- `ResidentBatchScheduler` owns `RowSamplingState` per request id and returns
  state tuples aligned with decode work so native/host row selection can see
  prompt history, generated history, stable row seeds, and step indices.
- `SamplerParamsBlock` represents all public sampler fields, not only
  temperature/top-k/top-p/repetition.
- The first c>N functional paths project rows through native packed prefill and
  sample each row serially: by the default native GPU sampler when all rows are
  covered by the native sampler contract, otherwise by host logits fallback.
- A true batched native c>N path should write selected token ids directly to
  `batch_lm_out_index` so graph replay can feed the next step without host
  token-list readback.
- `n > 1` should derive stable row seeds from the request seed and choice index;
  rows should diverge when logits and sampler settings allow it.

## Code touchpoints

| Area | Files/functions | Required change |
| --- | --- | --- |
| Public API | `hipengine/llm.py::SamplingParams`, `_generation_request` | Add canonical fields and validation. |
| Generation request | `hipengine/generation/registry.py::GenerationRequest` | Mirror canonical fields; keep dataclass torch-free. |
| Server schema | `hipengine/server/api.py` request models | Add fields, validate extras, reject unsupported parameters explicitly. |
| Server batching key | `hipengine/server/api.py::_sampling_key` | Include all sampler fields that affect output. |
| Row seeds | `hipengine/server/api.py::_row_seeds_for_request`, scheduler seed derivation | Keep deterministic and align with `RowSamplingState`. |
| PARO guards | `hipengine/generation/qwen35_paro.py` | Replace greedy-only rejection with sampler planning; keep greedy graph path. |
| GGUF guards | `hipengine/generation/qwen35_gguf.py` | Follow shared sampler extraction after PARO path is green. |
| PARO projection | `hipengine/runtime/qwen35_paro_runner.py` | Split logits projection from argmax selection. |
| Batch scheduler | `hipengine/generation/batch_scheduler.py` | Extend per-row sampler params/history and finish reasons. |
| Native kernels | `kernels/hip_gfx1100/linear/lm_head.hip` and new sampler kernels if needed | Add GPU processors/top-k/softmax/RNG/sample selection under registry keys. |
| Tests | `tests/test_sampling*.py`, server tests, Qwen smoke tests | Add pure CPU sampler tests, request plumbing tests, and GPU1 smoke gates. |

## Implementation tracks

| Track | Scope | Complexity | Approx. LoC | Dependencies | Exit gate |
| --- | --- | --- | --- | --- | --- |
| S0: API/schema cleanup | Extend `SamplingParams`, `GenerationRequest`, server request models, `_sampling_key`, and validation. Reject unsupported fields explicitly. | Low | ~100-200 Python/tests | None | **Done for public/server canonical fields.** |
| S1: greedy-compatible unblock | Allow `temperature <= 0` with inert `top_p`/`top_k`; preserve current graph replay and argmax behavior. | Low | ~50-100 Python/tests | S0 | **Done for PARO and GGUF greedy-equivalent requests.** |
| S2: host logits sampler | Add CPU/NumPy `select_token` over copied FP32 logits for temperature/top-k/top-p/min-p/seed. Disable graph replay for sampled requests. | Medium | ~400-700 Python/tests | S0 | **Done for PARO and GGUF c=1 plus serial multi-row host sampling.** |
| S3: token-history/static processors | Add prompt/generated history, repetition/presence/frequency penalties, logit bias, suppress-token ids, min-token/EOS policy, and deterministic processed-argmax. | Medium | ~250-500 Python/tests | S2 | **Done for host sampler:** synthetic-logit processor tests, suppress/min-token fixtures, and fixed-seed generator fixtures pass. |
| S4: token-level stop | Lower stop token IDs/sequences where possible and terminate rows early in generation, while retaining server stop-string trimming. | Medium | ~150-350 Python/tests | S2/S3 | **Done:** single-token IDs and multi-token server stop sequences finish PARO/GGUF host-sampled rows plus PARO c=1 and serial per-slot c>N native-sampled rows. |
| S5: c>N sampler state | Carry `RowSamplingState` through `ResidentBatchScheduler` and batch decode work; rows may still sample serially. | Medium/High | ~400-800 Python/tests | S2/S3 | **Done for PARO host/native row samplers:** sampled prompt batches use scheduler-owned state, native packed prefill, and serial host-sampled decode or default serial per-slot native sampling when all rows are covered; `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` disables native rows for rollback. GGUF remains serial by design until it gets a c>N resident scheduler. |
| S6: GPU top-k/temperature sampler | Native row-wise kernels for logits processing, top-k selection beyond the current `k <= 8` helper, softmax, RNG, and sample selection. | Medium/High | ~500-900 HIP/Python/tests | S2/S3 | **Promoted for scoped PARO default:** standalone FP32 logits processors plus full-vocab `top_k=0` and bounded `1 <= top_k <= 64` temperature samplers pass CPU-reference filtering/logprob parity and fixed-seed determinism; synthetic resident-session route smoke covers full-vocab, full-vocab top-logprobs, top-k+processor dispatch, suppress/min-token masks, bounded top-k probability filters, and bounded top-k `top_logprobs`. Supported c=1 PARO requests and fully covered PARO c>N sampled rows use native sampling by default; native decode-state telemetry reports no full-vocab logits D2H (`full_vocab_logits_d2h=false`, `logits_d2h_bytes=0`) while host fallbacks report `full_vocab_logits_d2h=true` with known per-token vector bytes. Unsupported PARO rows fall back with `native_gpu_unsupported_request`; GGUF remains host-sampled unless explicitly requested for native fallback metadata. |
| S7: exact GPU top-p | Full-vocab nucleus sampling without host logits readback. Requires efficient sort/select/cumulative probability strategy. | High | ~1000-2000 HIP/Python/tests | S6 | **Promoted for scoped PARO default:** standalone correctness-first GPU top-p/min-p sampler matches CPU retain counts, selected tokens, logprobs, tie order, and fixed-seed determinism on boundary fixtures; the synthetic resident-session route smoke covers top-p dispatch. Performance-oriented full-vocab nucleus selection remains future work. |
| S8: logprobs responses | Return selected logprob and optional top-logprobs through library/server schemas. | Medium/High | ~300-700 Python/HIP/tests | S2, optional S6/S7 | **Done for host-logits server/library paths:** completion/chat response tests pass for selected logprob/top-logprobs cases, completion `echo+logprobs`, and buffered streaming logprobs. |

The first useful user-facing milestone is S0+S1+S2. That gives correct normal
sampling with a known performance tradeoff and no change to greedy performance.
S3, S4, S5, and S8 are complete for the current host-sampler/PARO scheduler
scope. S6 and S7 are promoted for the scoped PARO native default, while true
batched c>N, GGUF native sampling, and broader
sampler processor parity remain future native GPU work and should not block
functional host support.

## Correctness and validation gates

### CPU/host sampler gates

- Pure sampler unit tests on synthetic logits:
  - greedy tie-break selects lower token id;
  - temperature sampling is deterministic at fixed seed;
  - `top_k` removes all but the top `k`;
  - `top_p` keeps the minimal nucleus and at least one token;
  - `min_p` keeps tokens above the relative threshold;
  - penalties and logit bias alter logits in the documented order.
- Generator tests with fake sessions proving:
  - greedy-equivalent requests still take the existing graph path;
  - sampled requests take the host sampler path and do not attempt graph replay;
  - `row_seeds` produce distinct `n > 1` outputs when logits allow it.
- Server tests proving fields are plumbed into `SamplingParams` and unknown or
  unsupported sampler fields are not silently ignored.
- Torch-free import/generate-path check: no `import torch` on the hot path.

### GPU1 smoke gates

- `HIP_VISIBLE_DEVICES=1` HIP load check passes.
- Small-vocab GPU sampler fixtures pass CPU-reference parity for processor order,
  filtering, ties, and retain-one behavior.
- Fixed-seed generated-token smoke is deterministic on RX 7900 XTX.
- Any full-model smoke records whether it used PARO or GGUF, KV policy, context
  length, and peak tracked allocation.

### GPU sampler promotion gates

- CPU-reference parity on small vocab fixtures for logits processing and
  filtering, including ties and boundary probabilities.
- Fixed-seed deterministic generated-token fixtures for c=1 and c>N rows.
- For stochastic distribution behavior, a bounded statistical smoke over a small
  vocabulary is acceptable; exact long-run distribution equality to the CPU path
  is not required unless the GPU path claims bit-for-bit sampler parity.
- `rocprofv3 --kernel-trace` evidence only when making a performance claim:
  record sampler kernel names, launch counts, and whether full-vocab D2H copies
  disappeared.

## Performance policy

- Greedy retained rows must stay on the current graph/argmax path unless a new
  path is proven exact and non-regressive under the normal benchmark policy.
- Host sampling is a compatibility path, not a retained performance claim. It may
  copy one `[vocab]` FP32 row per generated token.
- GPU sampling promotion requires both correctness gates and benchmark evidence.
  If exact GPU top-p is too costly, keep it as a separate high-complexity track
  instead of weakening the semantics of `top_p`.
- GPU1 / RX 7900 XTX measurements must not be merged into W7900 benchmark rows.
  They can be retained as explicitly labeled 7900 XTX artifacts if the normal
  evidence policy is satisfied.
- Any new default-off sampler experiment, fallback flag, or default-on opt-out
  must be added to `docs/REFACTOR.md` with a removal/promotion condition.

## Resolved decisions and open questions

Resolved for the current host-sampler milestone:

- `min_p` is public in `SamplingParams` and accepted by the server.
- `logit_bias` accepts raw token-id keys only; token-string aliases remain a
  tokenizer-lowering feature.
- GGUF uses the shared host sampler now instead of waiting for a later port.
- The public API exposes `stop_token_ids` / `stop_token_sequences`; OpenAI
  `stop` strings are post-trimmed and tokenizable stops are lowered for early
  host-path termination when tokenizer access is available.

Still open:

1. Should fixed-seed GPU sampling match the host RNG exactly, or is stable
   GPU-only determinism sufficient for retained native sampling?
