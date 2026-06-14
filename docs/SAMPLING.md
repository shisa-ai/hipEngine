# Sampling Design

Last updated: 2026-06-14

This document defines how hipEngine should grow from the current greedy-only
Qwen3.5/PARO and GGUF generation paths to normal server/library sampling
without weakening the torch-free runtime, plugin-registry boundaries, or retained
greedy performance path.

## Current state

The public API and server now expose the functional host-sampling surface for
PARO and GGUF while native GPU sampling remains incomplete:

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
- c>N PARO sampled requests use scheduler-owned row state, native packed prefill,
  and the serial host-sampled decode bridge. GGUF still samples prompt rows
  serially because its bring-up path has no c>N resident scheduler.
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
  logits processors for logit bias plus repetition/presence/frequency penalties,
  full-vocab `top_k=0` temperature sampling, bounded `1 <= top_k <= 64`
  sampling, and correctness-first exact full-vocab `top_p`/`min_p` filtering.
  They support per-row temperature, per-row seed, counter-based RNG, selected
  token id/logprob, retained-count reporting, and optional bounded-candidate
  logprobs. `HIPENGINE_QWEN35_NATIVE_SAMPLER=1` routes supported c=1 PARO
  temperature requests through these kernels with tiny selected-id/logprob/logit
  readbacks; a GPU1 synthetic resident-session smoke covers full-vocab,
  top-k+processor, and top-p route dispatch against CPU references. c>N PARO,
  GGUF, `top_logprobs`, and unsupported filter combinations still use the host
  sampler.

The original user-visible failure for non-greedy Qwen3.5/PARO and GGUF requests
is fixed for the host-logits path. Remaining implementation work is native GPU
sampler c>N/GGUF integration and performance promotion.

## Hardware lane for this work

Functional sampling development should use **GPU1**, the local AMD Radeon RX
7900 XTX (`gfx1100`), with explicit environment selection:

```bash
HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. <command>
```

Use GPU1 for sampler smoke tests, profiler experiments, and native-kernel
bring-up. The project default benchmark hardware remains the W7900 unless a row
is explicitly labeled RX 7900 XTX. Any retained performance claim from GPU1 must
record:

- hardware: AMD Radeon RX 7900 XTX, `gfx1100`;
- selected device: `HIP_VISIBLE_DEVICES=1`;
- model, quant, prompt/decode shape, KV policy, sampler mode, and exact command;
- correctness gate and whether the path used host logits readback or native GPU
  sampling.

The 7900 XTX has less VRAM than the W7900, so full-model smoke commands should
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
  logprobs without buffering, and native-GPU `top_logprobs` parity.
- Speculative sampling / probability-ratio acceptance. That belongs with the
  relaxed/speculative documents because it changes the accept contract.
- Matching another engine's exact random stream. hipEngine should define its own
  deterministic stream and document it.
- Promoting GPU sampling performance. Native GPU sampling is a later retained
  performance track after functional support is correct.

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
| `seed` / `row_seeds` | Public/server/runtime | Stable row RNG seed; `n > 1` rows diverge deterministically. | Low/Medium |
| `stop` strings | Server post-trim + token lowering | Keep post-trim; lower one-token stops to `stop_token_ids` and multi-token stops to suffix-matched `stop_token_sequences`. | Medium |
| `stop_token_ids` / `stop_token_sequences` | Public/runtime + scheduler state | Token stops finish PARO/GGUF host-sampled rows; PARO c>N sampled batches consume the same scheduler stop metadata; future native GPU sampler kernels must preserve parity. | Medium |
| `logprobs` / `top_logprobs` | Public/server/runtime for host-logits paths | Return selected logprob and optional top candidates; completion `echo+logprobs` shifts generated-token offsets after a null-logprob prompt prefix. Streaming logprobs use a buffered detailed-generation SSE path while ordinary non-logprob streams remain live token/chunk streams. | High |

Compatibility rule: if `temperature <= 0` and the request has no active logit
processors (`logit_bias`, penalties, bad-token constraints, etc.), `top_p` and
`top_k` do not change the selected token because the top logit remains included.
Those requests should use the greedy fast path rather than failing merely because
a client sent `top_p=0.95` with `temperature=0`.

Speculative/MTP compatibility is stricter until target verification can run the
same processed-logit policy as autoregressive generation.
`supports_speculative_mtp_sampling()` returns true only for `GREEDY_FAST`
requests; `speculative_mtp_sampling_blockers()` reports the fields that require
AR fallback today, including `logit_bias`, penalties, token stops,
`temperature > 0`, and requested logprobs. The resident scheduler applies this
guard before emitting speculative target-verification work, so rows that need
processed logits cannot silently enter the raw-argmax MTP path. The public
capabilities manifest exposes both the flat blocker field list and
`sampling.speculative_mtp.incompatible_conditions`, so clients can distinguish
conditional blockers such as `temperature > 0` from inert greedy filters like
`top_p`, `top_k`, and `min_p`.

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
- no token-level constraints beyond EOS/ignore-EOS;
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
4. If `temperature <= 0`, choose argmax over processed logits.
5. If `temperature > 0`, divide logits by temperature.
6. Apply `top_k` filter.
7. Convert to probabilities with max-subtracted softmax.
8. Apply `top_p` / `min_p` filters, always retaining at least one token.
9. Renormalize and draw one token from the row RNG.
10. Append the token to row history and update counts.

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

1. **Logits processors:** apply logit bias and penalties row-wise over the full
   vocab. The standalone S6 processor kernel covers finite-clamping, logit bias,
   repetition penalty, presence penalty, and frequency penalty from compact
   per-row bias and token/count lists; generation routing can optimize the
   compact-list ABI later.
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
selection and matches retain-one semantics on boundary fixtures. It is not the
promoted generation path yet; routing and any performance-oriented sort/select
replacement still need separate validation.

### GPU1 bring-up commands

Use GPU1 explicitly for native sampler smoke and profiler loops:

```bash
HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 \
  python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"

HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. \
  python3 -m pytest tests/test_gpu_sampler_kernel.py -q
```

The test above is the current standalone native sampler unit/integration
coverage plus a synthetic resident-session c=1 route smoke for the opt-in PARO
native sampler. For profiler evidence, prebuild JIT libraries before `rocprofv3` and
run only a narrow sampling smoke under the profiler. Do not wrap a parent harness
that spawns nested Python children.

## c>N and server batching

c>N support should reuse the same sampler state, but it does not have to be
fully vectorized at first:

- `ResidentBatchScheduler` owns `RowSamplingState` per request id and returns
  state tuples aligned with decode work so native/host row selection can see
  prompt history, generated history, stable row seeds, and step indices.
- `SamplerParamsBlock` represents all public sampler fields, not only
  temperature/top-k/top-p/repetition.
- The first c>N functional path may project rows in batch and sample each logits
  row serially on host.
- A native c>N path should write selected token ids directly to
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
| S3: token-history processors | Add prompt/generated history, repetition/presence/frequency penalties, logit bias, and deterministic processed-argmax. | Medium | ~250-500 Python/tests | S2 | **Done for host sampler:** synthetic-logit processor tests and fixed-seed generator fixtures pass. |
| S4: token-level stop | Lower stop token IDs/sequences where possible and terminate rows early in generation, while retaining server stop-string trimming. | Medium | ~150-350 Python/tests | S2/S3 | **Done for host sampler:** single-token IDs and multi-token server stop sequences finish PARO/GGUF host-sampled rows; native c>N/GPU execution still consumes this later. |
| S5: c>N sampler state | Carry `RowSamplingState` through `ResidentBatchScheduler` and batch decode work; rows may still sample serially. | Medium/High | ~400-800 Python/tests | S2/S3 | **Done for PARO host sampler:** sampled prompt batches use scheduler-owned state, native packed prefill, and serial host-sampled decode; GGUF remains serial by design until it gets a c>N resident scheduler. |
| S6: GPU top-k/temperature sampler | Native row-wise kernels for logits processing, top-k selection beyond the current `k <= 8` helper, softmax, RNG, and sample selection. | Medium/High | ~500-900 HIP/Python/tests | S2/S3 | **Partial:** standalone FP32 logits processors plus full-vocab `top_k=0` and bounded `1 <= top_k <= 64` temperature samplers pass GPU1 CPU-reference filtering/logprob parity and fixed-seed determinism; a synthetic resident-session c=1 route smoke covers full-vocab and top-k+processor dispatch. Supported c=1 PARO requests can opt in with `HIPENGINE_QWEN35_NATIVE_SAMPLER=1`; c>N/GGUF and `top_logprobs` still fall back. |
| S7: exact GPU top-p | Full-vocab nucleus sampling without host logits readback. Requires efficient sort/select/cumulative probability strategy. | High | ~1000-2000 HIP/Python/tests | S6 | **Partial:** standalone correctness-first GPU top-p/min-p sampler matches CPU retain counts, selected tokens, logprobs, tie order, and fixed-seed determinism on GPU1 boundary fixtures; the synthetic resident-session c=1 route smoke covers top-p dispatch. Supported c=1 PARO `top_k=0` requests can opt in; not performance-promoted. |
| S8: logprobs responses | Return selected logprob and optional top-logprobs through library/server schemas. | Medium/High | ~300-700 Python/HIP/tests | S2, optional S6/S7 | **Done for host-logits server/library paths:** completion/chat response tests pass for selected logprob/top-logprobs cases, completion `echo+logprobs`, and buffered streaming logprobs. |

The first useful user-facing milestone is S0+S1+S2. That gives correct normal
sampling with a known performance tradeoff and no change to greedy performance.
S3, S4, S5, and S8 are complete for the current host-sampler/PARO scheduler
scope. S6 and S7 are partially landed as standalone GPU logits processors plus
full-vocab, bounded top-k, and exact top-p/min-p temperature samplers. A
supported c=1 PARO route is available behind `HIPENGINE_QWEN35_NATIVE_SAMPLER`,
while c>N/GGUF routing and performance promotion remain native GPU work and
should not block functional host support.

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
- Any default-off sampler experiment or fallback flag must be added to
  `docs/REFACTOR.md` with a removal/promotion condition.

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
