# Sampling Design

Last updated: 2026-06-14

This document defines how hipEngine should grow from the current greedy-only
Qwen3.5/PARO and GGUF generation paths to normal server/library sampling
without weakening the torch-free runtime, plugin-registry boundaries, or retained
greedy performance path.

## Current state

The public API and server already expose a small sampling surface, but the
runnable Qwen3.5 paths only execute greedy argmax:

- `hipengine.llm.SamplingParams` carries `max_tokens`, `temperature`, `top_p`,
  `ignore_eos`, KV policy knobs, `seed`, and `row_seeds`.
- `hipengine.generation.registry.GenerationRequest` mirrors that narrow shape.
- `hipengine.server.api` accepts OpenAI-style `temperature`, `top_p`, `seed`,
  `stop`, `n`, and a few completion-only flags; most other common sampler fields
  are neither modeled nor deliberately rejected.
- `hipengine.generation.qwen35_paro.Qwen35ParoOneTokenGenerator` and
  `hipengine.generation.qwen35_gguf.Qwen35GGUFBringupGenerator` reject requests
  unless `temperature == 0.0` and `top_p == 1.0`.
- `Qwen35ParoResidentSession._sample_device_from_hidden(...)` is final RMSNorm
  -> BF16 cast -> W8A16 LM-head logits -> `argmax_f32`.
- c>N sampler dispatch only selects between serial and row-aware LM-head argmax;
  it is not a stochastic sampler.
- `lm_head.hip` has a row-wise top-k helper capped at `k <= 8`, useful for
  drafter/verifier diagnostics but not enough for normal user sampling.

The immediate user-visible failure is therefore correct for the current runtime:
non-greedy requests reach a guard before generation starts.

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

## Non-goals for the first implementation

- Grammar / JSON-schema constrained decoding.
- Beam search or `best_of` ranking.
- Full OpenAI `logprobs` / `top_logprobs` response support.
- Speculative sampling / probability-ratio acceptance. That belongs with the
  relaxed/speculative documents because it changes the accept contract.
- Matching another engine's exact random stream. hipEngine should define its own
  deterministic stream and document it.

## Parameter contract

The canonical parameter set should live in `SamplingParams`, flow into
`GenerationRequest`, and be lowered to per-row sampler state. Server request
models should either populate these fields or reject unsupported aliases.

| Field | Current state | Target behavior | Initial complexity |
| --- | --- | --- | --- |
| `max_tokens` | Public/server/runtime | Already supported. | Low |
| `ignore_eos` | Public/server/runtime | Already supported for EOS; stop-token rows need integration. | Low |
| `temperature` | Public/server, guarded in runtime | `<= 0` means deterministic argmax after processors; `> 0` enables multinomial sampling. | Medium |
| `top_p` | Public/server, guarded in runtime | Nucleus filtering after processors. Inert for plain `temperature <= 0` argmax. | Medium |
| `top_k` | Scheduler row type only | Keep highest `k` tokens before sampling; `0` means disabled. | Medium |
| `min_p` | Missing | Optional probability floor relative to max probability. | Medium |
| `repetition_penalty` | Scheduler row type only | HF-style penalty using prompt + generated history. | Medium |
| `presence_penalty` | Missing | Subtract once for tokens already present. | Medium |
| `frequency_penalty` | Missing | Subtract proportional to token count. | Medium |
| `logit_bias` | Missing | Add per-token bias map before filtering. | Medium |
| `seed` / `row_seeds` | Public/server, not consumed by sampler | Stable row RNG seed; `n > 1` rows must diverge deterministically. | Low/Medium |
| `stop` strings | Server post-trim only | Keep post-trim initially; add token-level early stop when stop-token lowering is available. | Medium |
| `logprobs` / `top_logprobs` | Rejected or absent | Later response feature; do not block basic sampling. | High |

Compatibility rule: if `temperature <= 0` and the request has no active logit
processors (`logit_bias`, penalties, bad-token constraints, etc.), `top_p` and
`top_k` do not change the selected token because the top logit remains included.
Those requests should use the greedy fast path rather than failing merely because
a client sent `top_p=0.95` with `temperature=0`.

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
  - optional `logprob` and `top_logprobs` later.

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

### 3. Processor order

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

### 4. RNG policy

The host implementation should use a stable, explicitly seeded generator owned
by hipEngine, not Python's process-random `hash()` or global `random` module.
A simple first implementation can use NumPy `Generator(PCG64(seed))` per row,
with the derived `row_seed` recorded in metadata/tests.

The future GPU sampler should use a counter-based stream keyed by
`(row_seed, step_index, row_index)` so graph capture and replay do not depend on
mutable host RNG state. Exact CPU/GPU random-stream equality is nice but not a
requirement for first GPU promotion; fixed-seed determinism and distribution
sanity are required.

## Implementation tracks

| Track | Scope | Complexity | Approx. LoC |
| --- | --- | --- | --- |
| S0: API/schema cleanup | Extend `SamplingParams`, `GenerationRequest`, server request models, `_sampling_key`, and validation. Reject unsupported fields explicitly. | Low | ~100-200 Python/tests |
| S1: greedy-compatible unblock | Allow `temperature <= 0` with inert `top_p`/`top_k`; preserve current graph replay and argmax behavior. | Low | ~50-100 Python/tests |
| S2: host logits sampler | Add CPU/NumPy `select_token` over copied FP32 logits for temperature/top-k/top-p/min-p/seed. Disable graph replay for sampled requests. | Medium | ~400-700 Python/tests |
| S3: token-history processors | Add prompt/generated history, repetition/presence/frequency penalties, logit bias, and deterministic processed-argmax. | Medium | ~250-500 Python/tests |
| S4: token-level stop | Lower stop token IDs where possible and terminate rows early in generation, while retaining server stop-string trimming. | Medium | ~150-350 Python/tests |
| S5: c>N sampler state | Carry `RowSamplingState` through `ResidentBatchScheduler` and batch decode work; rows may still sample serially. | Medium/High | ~400-800 Python/tests |
| S6: GPU top-k/temperature sampler | Native row-wise kernels for logits processing, top-k selection beyond the current `k <= 8` helper, softmax, RNG, and sample selection. | Medium/High | ~500-900 HIP/Python/tests |
| S7: exact GPU top-p | Full-vocab nucleus sampling without host logits readback. Requires efficient sort/select/cumulative probability strategy. | High | ~1000-2000 HIP/Python/tests |
| S8: logprobs responses | Return selected logprob and optional top-logprobs through library/server schemas. | Medium/High | ~300-700 Python/HIP/tests |

The first useful user-facing milestone is S0+S1+S2. That gives correct normal
sampling with a known performance tradeoff and no change to greedy performance.
S3 and S4 make the behavior match common client expectations. S6/S7 are
performance work and should not block functional support.

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

### GPU sampler gates

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
- Any default-off sampler experiment or fallback flag must be added to
  `docs/REFACTOR.md` with a removal/promotion condition.

## Open questions

1. Should `min_p` be public in the first API expansion or only server-compatible
   via extra fields?
2. Which logit-bias token-id namespace should the server accept for tokenizer
   variants: raw token id only, token string aliases, or both?
3. Should host sampling be available for GGUF immediately, or should PARO land
   first and GGUF follow after shared sampler extraction?
4. How much OpenAI `logprobs` compatibility is needed before marking sampled
   server responses production-ready?
5. Should fixed-seed GPU sampling match the host RNG exactly, or is stable
   GPU-only determinism sufficient for retained native sampling?
