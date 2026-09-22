# Changelog

All notable user-facing changes for hipEngine releases are documented here.

This changelog is for package/API releases. Performance rollup history remains in
[`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md), with detailed benchmark
evidence under [`benchmarks/results/`](benchmarks/results/).

## v0.7.0 - 2026-09-22

Fixes long-prompt prefix reuse, widens speculative decoding on INT8 KV, and adds
measurement to the terminal chat client. Prefix reuse now works on prompts long
enough that a reply crosses a cache boundary — the case where it previously
reused nothing and re-prefilled the entire prompt on the non-speculative route.
Speculative decoding on INT8 KV now covers sampled requests and multi-request
groups, and the autoregressive finish rule applies to a whole verified chain, so
`min_tokens`, EOS and stop ids, and stop sequences are servable on the
speculative routes. Hardware validation covers Radeon RDNA 3 (`gfx1100`: RX 7900
XTX, Pro W7900) and Strix Halo (`gfx1151`: Ryzen AI MAX+ 395 with Radeon 8060S);
support and results are specific to each model and backend. Numbers live in
[`benchmarks/README.md`](benchmarks/README.md) and the dated performance history
in [`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md).

### Added

- **`/bench` in the terminal chat client.** `hipengine chat` gains `/bench [in]
  [out]` (default 512 in, 32 out), which sends one cold and one repeat request
  per route of the running server and prints one column per metric:
  `mtp | cache | prefill tok/s | prompt tok/s | decode tok/s | ttft s | tpot ms`.
  TTFT and TPOT are separate columns because they are the two sides of the trade
  speculative decoding makes against decode throughput. Each route gets its own
  nonce-prefixed prompt so one route's cold run cannot hit the other's cache
  entry, and a route the server refuses is reported as a note instead of failing
  the bench. The plain, dependency-free client prints the same table.
- **Prefix-cache accounting in responses and chat.** Chat completions now report
  the vLLM-compatible `usage.prompt_tokens_details.cached_tokens`, summed from
  the per-request prefix-cache diagnostics. It is omitted entirely when no output
  carried prefix telemetry, so the server never reports a zero it did not
  measure, while a cache that was consulted and missed reports `0`. The chat
  client's per-reply line reports `N tokens · prefill X tok/s · decode X tok/s ·
  cached N · ttft · total`, adds `prompt X tok/s` when a cache hit made the
  prefilled portion smaller than the prompt, and `/usage` gains a cumulative
  `cached` row.
- **Startup warms the width the server admits and the speculative route.** The
  scratch probe was sized from `max_active_requests or 1`, and unset means "no
  server-wide cap", so on a default server the probe asked for one row and
  skipped every packed and MTP warmup it exists to run. The probe now takes the
  width the server can actually admit, and a new `mtp_smoke` startup stage sends
  one bounded request with `speculative_mtp: true` through the production
  batcher before readiness, recording both the resolved route and the route the
  batcher realized, so a silent fallback to normal decoding is visible. A failing
  MTP warmup is recorded as `failed` and does not fail startup.
- **Sampled and multi-request speculative decoding on INT8 KV.** The packed
  target verifier binds the retained INT8 payload planes and their per-token-head
  scale metadata, so a speculative request group of more than one row speculates
  instead of falling back to normal decoding, and multi-session packed INT8
  prefill no longer collapses to one row. Requests using `temperature`,
  penalties, `logit_bias`, or `suppress_token_ids` now speculate on INT8 KV as
  well; the declarations list `greedy_fast` and `sampled` for INT8, and the
  runtime gate that requires an evidence row matching this backend,
  architecture, weight quant, and artifact size is unchanged. A group width the
  artifact does not qualify for fails with a named capability error rather than
  downgrading silently.
- **Speculative routes honour the autoregressive finish rule.** The cycle commit
  implemented one finish rule — EOS on the last visible token of a greedy chain —
  so `min_tokens`, `eos_token_id`, `stop_token_ids`, and
  `stop_token_sequences` had to stay off the speculative routes. The rule now
  applies to the whole verified chain, selects the terminal prefix, commits
  before the terminal token, retains no model state past that prefix, and reports
  `eos` or `stop` with the same `stop_token_id` / stop-sequence detail the
  autoregressive route reports. The four fields are servable.
- **An engine-neutral agentic session replay harness.**
  `scripts/agentic_session_compare.py` replays a recorded agent session against
  any OpenAI-compatible endpoint over HTTP, preserving `tool_calls`,
  `tool_call_id`, and `name` verbatim instead of flattening them to role and
  content. It requests `stream_options.include_usage` and records per-turn
  prompt tokens, first-token time, decode rate, wall time, content deltas, and
  tool-call chunks, aggregating cold and warm passes separately. `--dry-run`
  checks request fidelity offline and `--no-tools` omits the tool array.

### Changed

- **The startup scratch probe prices its width before allocating it.** The probe
  acquires one resident session per slot, each preallocating its own KV pages and
  packed workspace lease, but the capacity estimator prices one session — so a
  widened probe asked for four slots that each fitted alone and the fourth
  allocation failed after four out-of-memory attempts at startup. The probe runs
  at the configured request width, reduces to one slot when the machine cannot
  hold that width, records the numbers in `startup.checks.scratch_probe` and the
  `/ready` diagnostics, and retries at width one when a wider probe fails to
  allocate. Only a width-one failure is fatal, and a fatal eager-startup stage
  now exits non-zero instead of leaving the process listening while unready.
- **A diagnostic override runs the direct INT8 route on its declared contract.**
  Route resolution required the runtime action to be `admit`, so a documented
  `diagnostic_override` — where the engine has already answered the selection
  question — never consulted the kernels. "May this session execute mirror-free
  INT8" is now a capability question that admits the override, while "is it
  admitted to allocate" stays strict and still governs the retained BF16 mirrors.
  The registry lookup remains the capability check, so a contract no kernel
  registers still refuses.

### Fixed

- **Prefix reuse no longer loses the boundary a following turn can reach.** The
  native decode step refreshes the prefix cache after every token, so any reply
  long enough to cross the next 256-token boundary captured a boundary *past*
  that request's own prompt end, and the supersede loop evicted the
  prompt-aligned capture with it. The survivor sat deeper than any token count a
  following turn can match, so that turn matched no live prefix at all and
  re-prefilled the whole prompt. Both boundaries are now captured and retained,
  with the prompt-aligned one ranked above the decode-time one, which a resend
  cannot reach; under retained-budget pressure the decode-time entry is trimmed
  first. Measured on `qwen3.8-27b` on `hip_gfx1151` with one server command and
  the `/bench` cold-then-cached pair, before and after on the same machine:
  with speculation off, a 2048-token prompt reused 0 tokens at 8.19 s to first
  token before the fix and 1792 tokens at 1.32 s after it; 8192 reused 0 at
  32.41 s and 7680 at 1.43 s. The speculative route already reused 1792 and 7680,
  and every route that already reused keeps its depth.
- **`LLM` exposes the resident capacity estimate.** The startup probe's width
  preflight looked for `resident_capacity_estimate` on the object the server
  holds, and `LLM` delegates generator hooks through explicit methods rather
  than attribute passthrough. The lookup found nothing, so every probe ran at its
  configured width and the preflight was dead code on a real server. Unit tests
  passed because their fakes carried the method.
- **`/help` no longer drops the `/bench` arguments.** The help table passed
  command strings straight to the renderer, which read `[in]` and `[out]` as
  style tags and removed them, so `/help` showed `/bench` with no arguments while
  the plain client printed them.
- **`docs/ENVS.md` describes which commands get the 503.** The
  `HIPENGINE_ENGINE_COMMAND_TIMEOUT_SECONDS` entry claimed an exhausted command
  budget is reported as HTTP 503 `engine_unavailable`. That holds for commands
  that serve a request, and not for the readiness and metrics diagnostics, which
  swallow the exception and lose their live snapshot fields instead.

### Known Limitations

- `/ready`'s worst-case latency is the engine command timeout, which defaults to
  300 seconds, because the readiness read enqueues a command onto the single
  driver thread and waits behind whatever engine work is already running. The
  event loop stays responsive and the verdict comes from the server's own startup
  state, so this costs diagnostics rather than correctness, but no load balancer
  treats a five-minute probe response as healthy. Lower
  `HIPENGINE_ENGINE_COMMAND_TIMEOUT_SECONDS` for a tight probe. `/health` returns
  a static payload and is unaffected.
- The finish-rule and metadata set (`min_tokens`, `eos_token_id`, stop ids, stop
  sequences, and `logprobs`) is still refused on both BF16 and INT8 KV
  speculative routes; only the sampled route applies the finish rule. This is not
  a storage question and the refusal is explicit.
- The prefix cache's retained budget is still `max(1, capacity)` entries while a
  long-decoding request can now hold two, so under a saturated budget
  conversation capacity trades against decode-time reuse depth. The decode-time
  entry is trimmed first, so the effective floor is the previous behavior.
- INT8 speculative decoding does not establish INT8-versus-BF16 output quality or
  a throughput gain. Packed INT8 verification now runs at group widths up to
  four, and broader memory-pressure qualification remains open.
- The direct INT8 route under a diagnostic override binds the declared leaf on
  live `gfx1151`: `/ready` reports `runtime_action: diagnostic_override` with
  `effective_kv_storage: int8_per_token_head`, and the server gate's per-request
  assertions see `kv_attention_source: int8_direct` with zero persistent BF16
  mirror bytes. Reaching that route needs a context above 8192 tokens, because at
  or below that length the INT8 route deliberately keeps the exact BF16 decode
  mirror as a correctness measure.
- On this host's `Qwen3.8-27B` `Q4_K_M` artifact, whose INT8 KV quality record is
  rejected, that override-gated INT8 route can produce a different token from its
  own autoregressive route; one case on the server gate's `code_lru_cache` prompt
  differs by a single space. It reproduces identically at v0.6.0, so this release
  does not introduce it. The BF16 default is exact: all 18 gate prompts match
  between autoregressive and speculative decoding on BF16 KV.
- A long-context INT8 KV server can return HTTP 500 on a request that follows a
  concurrent autoregressive/speculative pair. The default INT8 layout keeps a
  BF16 prefix of eight full-attention layers, and a packed decode with more than
  one row routes those layers and the INT8 layers through different attention
  paths, which the packed-decode consistency check rejects. The lifecycle gate
  fails at its multi-choice check on that layout and passes all seven checks when
  the layers are uniformly INT8, which is the diagnostic layout. The concurrent
  requests themselves succeed; the failure appears on the next request. Avoid
  this combination by keeping `--max-context-tokens` at or below 8192, which uses
  a uniform layout.
- APIs and supported combinations may change before 1.0.

## v0.6.0 - 2026-09-20

Alpha release adding OCR and speech runtimes, broader dynamic-GGUF support,
and improvements to speculative decoding, prefix reuse, and serving reliability.
Model, quantization, hardware, and workload qualifications remain specific to
the combinations documented below; this is not a blanket production certificate.

### Added

- **Surya OCR 2:** torch-free image preprocessing, vision and text inference,
  full-page OCR output handling, and OpenAI-compatible HTTP serving. CPU
  reference and gfx1151 HIP implementations include request isolation,
  cancellation, and bounded attention scratch. Install `hipengine[surya]`.
- **VibeVoice-ASR:** native `LLM.transcribe()` with stateful audio encoding,
  transcription finish reasons, and request isolation. A standalone Q4_K_M
  GGUF transcription CLI embeds tokenizer and configuration assets for offline
  inference. Broader production qualification is incomplete.
- **Experimental VibeVoice TTS:** `LLM.synthesize()`, reference-voice prompting,
  a HIP acoustic decoder, diffusion head, and device-side solver. Numerical
  and generated-speech quality qualification remains partial.
- **Dynamic GGUF quantization:** Qwen3.8-27B UD-Q4_K_M and UD-Q4_K_S execution
  with compressed IQ/Q3 kernels, compact Q5/Q6 residency, artifact-aware
  admission, and optimized prefill, decode, and speculative verification.
- **Sampled speculative decoding:** temperature-based MTP acceptance on
  Qwen3.8-27B Q4_K_M on gfx1151 with BF16 KV and one through four active
  requests, including native GPU sampling, request-local random state, and
  selected-state commit. Direct-generator sampled MTP is not yet supported.
- **INT8 KV speculative decoding:** dense GGUF single-request MTP, including
  inside a wider resident server, with eager/graph verification and prefix
  checkpoint restoration. Packed multi-request INT8 MTP and compact-DMS MTP
  are not implemented; INT8 KV quality admission remains independent.
- **Serving diagnostics:** per-request speculative/AR token attribution,
  provider readiness, refusal reasons, realized group widths, and
  proposal/verification/provider-update timings.

### Changed

- Prefix reuse now includes batched suffix prefill, pooled snapshots,
  placement-aware routing, and draft-provider checkpoint restore for eligible
  cache hits. Both the direct engine and HTTP server default to radix caching.
- The server CLI defaults to BF16 KV. Compressed INT8 KV remains an explicit
  option subject to artifact-specific quality admission.
- Supported dense and MoE GGUF implementations can use automatic MTP without
  a matching benchmark row, including supported UD artifacts. Backend,
  storage, sampling, physical group/depth, and memory limits still apply.
  Configurations measured slower than AR remain excluded from automatic policy.
- MTP context is bounded by the target's allocated capacity, not a fixed
  1,023-token admission window.
- Packed workspace and private KV memory accounting follow serving capacity.
- Long-context speculative verification uses staged linear-attention kernels
  on supported gfx1151 Q4_K_M routes.
- Source archives now omit repository-only benchmark data, surveys, worklogs,
  docs, and test fixtures while retaining all wheel build inputs.
- CI checks every distribution's size before any upload, preventing a partial
  PyPI release when an artifact exceeds the project limit.

### Fixed

- Contain request-local failures where ownership permits, route unavailable
  speculative/prefill work to supported fallbacks, and return a typed 503 when
  the engine is closed.
- Bound streaming queues by output budgets and defer conflicting MTP prompt
  activation instead of failing the service.
- Keep HTTP request handling responsive while engine work waits on the GPU.
- Stop MTP at EOS without publishing extra tokens or retaining model state
  beyond the terminal prefix; preserve neighboring request cursors.
- Report explicit-only MTP as default AR at startup, and label unmeasured
  admission using evidence metadata rather than reason-string prefixes.
- Repair KV-pool growth/binding, host-upload lifetimes, recycled state
  initialization, and NaN-safe state clearing.
- Fix softmax synchronization races in EVIE, paged attention, and TimesFM.
- Withdraw the invalid gfx1151 C8/K3 automatic MTP configuration.

### Known Limitations

- Dense-Qwen prefix reuse has same-route numerical checks on gfx1151 and
  gfx1100, including separate BF16 and INT8 ownership checks. This does not
  establish equality between different prefill arithmetic paths. Disable
  prefix caching with `prefix_cache="off"` or `--prefix-cache off` when needed.
- MTP availability depends on the implemented backend, storage, sampling, and
  realized batch shape. Automatic requests can fall back to AR; explicit
  requests for unimplemented combinations return a named capability error.
- INT8 MTP does not establish INT8-versus-BF16 quality or a throughput gain.
  Packed INT8 verification, compact-DMS transactions, and broader memory-pressure
  qualification remain open.
- New OCR and speech hardware evidence is primarily gfx1151. VibeVoice ASR
  and TTS remain early implementations, not fully qualified production ports.
- APIs and supported combinations may change before 1.0.

## v0.5.0 - 2026-09-12

hipEngine now runs more than the Qwen 3.6 35B mixture-of-experts models: dense
Qwen models, a multimodal retrieval encoder, a time-series forecaster, and
Qwen3.8 Flash-Next on Strix Halo. Several performance choices that used to need
manual settings now happen on their own, but only where they have been measured
as safe for the model and shape in use. Hardware validation covers Radeon
RDNA 3 (`gfx1100`: RX 7900 XTX, Pro W7900) and Strix Halo (`gfx1151`: Ryzen
AI MAX+ 395 with Radeon 8060S); support and results are specific to each model
and backend. Numbers live in
[`benchmarks/README.md`](benchmarks/README.md) and the dated performance history
in [`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md).

### Added

- **Terminal chat.** `hipengine chat` connects to a running local server and
  discovers its model automatically. Install `hipengine[chat]` for live
  Markdown, optional reasoning display, per-turn stats, input history, and
  command completion. `/status` shows server limits, `/usage` shows token
  counts, and `/retry` regenerates a reply. Sampling, system prompts, and
  reasoning settings can be changed without leaving the conversation.
  A dependency-free plain mode is also available.
- **Visible startup progress.** Model preparation shows an approximate loading
  bar based on the increase in VRAM usage relative to the model file size.
  Terminal output updates in place; redirected logs receive periodic snapshots.
  A startup summary groups context, KV format, pool budget, and device memory.
  The bar is a memory proxy, not an exact weight-loading percentage or ETA.
- **KV pool budget controls for dense GGUF.** The shared pool starts with a
  configurable page allocation and has a growth path and prefix-cache pressure
  handling. `--kv-pool-memory-budget-mib` sets a KV-pool budget independently
  of `--max-active-requests`. `/ready` and Prometheus expose the pool's budget,
  maximum pages, and current usage.
- **Automatic dense-GGUF context sizing.** Starting the server without
  `--max-context-tokens` selects a context using available device memory and
  preparation reserves. Qualified artifacts use INT8 KV with FP32 scales;
  unsupported GGUF artifact contracts fall back to BF16.
- **Dense Qwen 27B models.** Qwen3.6-27B and Qwen3.8-27B now load, generate, and
  serve from GGUF `Q4_K_M` on both AMD backends. Qwen3.8-27B also runs in
  `Q4_K_S` on Strix Halo. Both sizes can use speculative decoding driven by the
  model's own multi-token prediction (MTP) head. Qwen3.8-27B is measured from one
  request up to eight running at once, alongside two llama.cpp HIP builds
  measured the same way.
- **Initial support for more model families.** EVIE-4.5B and EVIE-8B run
  multimodal retrieval encoding ([model record](docs/model-cards/MODEL-EVIE.md)), TimesFM
  2.5 200M and TimesFM 3.0 500M run time-series forecasting
  ([2.5](docs/model-cards/MODEL-TIMESFM.md), [3.0](docs/model-cards/MODEL-TIMESFM3.md)), and Qwen3.8
  Flash-Next 125B-A6B runs on Strix Halo
  ([survey](docs/campaigns/QWEN3.8-FLASH-NEXT-STRIX-HALO-SURVEY.md)). These are early
  ports: each record names the measured performance and the current limits.
- **The server decides when to speculate.** The default for
  `--speculative-mtp-serving` (env `HIPENGINE_SPECULATIVE_MTP_SERVING`) moved
  from `off` to `auto`. In `auto`, a request uses speculative decoding only when
  the loaded model, quantization, GPU, kernel plan, cache type, batch size, and
  prompt length all match a combination hipEngine has measured; otherwise the
  request decodes normally and the `hipengine.speculative_mtp` block in the
  response reports why (for example `backend_k0_fallback`). Use
  `enabled` to take the speculative route for every compatible request, `opt_in`
  to require `"speculative_mtp": true` per request, or `off` to switch it off.
- **A kill switch and acceptance counts for speculation.**
  `POST /v1/hipengine/speculative_mtp/rollback` sends every new request to normal
  decoding until the server restarts, while requests already running finish the
  work they have in flight. Repeated backend failures trip a circuit breaker for the affected
  model, GPU, kernel plan, and context range, and stop speculation there until
  restart; a client disconnecting or exceeding a deadline does not trip it.
  Responses that speculated report `accepted_prediction_tokens` and
  `rejected_prediction_tokens` under `usage.completion_tokens_details`, and
  `/metrics` carries matching counters, so existing speculative-decode tooling
  works unchanged.
- **Reasoning requests can speculate too.** The speculative path is token-exact
  only for plain greedy decoding, and hipEngine's host-side reasoning-budget
  enforcement breaks that. So `--speculative-mtp-thinking` (env
  `HIPENGINE_SPECULATIVE_MTP_THINKING`, per-request
  `"speculative_mtp": {"thinking": ...}`) picks the trade: `hint`, the default,
  keeps the thinking markers in the prompt but stops forcing the budget from the
  host, so a `reasoning_effort` request can stay on the exact speculative route;
  `hard` keeps full enforcement and decodes such requests normally instead. The
  policy actually used is reported in the response.
- **Pick a kernel plan explicitly, and get the fast one by default.**
  `LLM(execution_profile=...)`, `--execution-profile`, and
  `HIPENGINE_EXECUTION_PROFILE` accept `strict`, `production`, or
  `batch_invariant`. hipEngine checks that the plan's kernels and their
  plain-decoding fallbacks are installed, refuses an unregistered combination
  instead of guessing one, and reports the plan's hash in server metadata.
  Leaving it unset now selects `production` for every model, backend, and
  quantization with a certified plan, so a default run gets the measured fast
  composition instead of the exact one; models with no certified plan keep
  their previous behaviour. Use `strict` for exact arithmetic.
- **Controls for the GGUF speculative path.**
  `HIPENGINE_GGUF_MTP_VERIFY_MODE` chooses the fast candidate checker (`native`,
  the default) or the one that replays normal decoding for each candidate and
  matches it token for token (`serial_exact`, which cannot be faster than normal
  decoding). `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` sets how many draft tokens to
  try per step, 1-4, default 3; 4 can be slower.
- **Tune how much a scheduler tick does.** Round prefill-token and decode-row
  budgets are now flags with matching environment variables, next to the
  existing `--prefill-decode-policy` and `--max-active-requests` options.

### Changed

- **Qwen3.8-27B no longer speculates by default on RDNA 3.** Measured on
  2026-09-06 at every batch size from two to eight and every draft depth from one
  to three, each against normal decoding of the same model in the same run: all
  twenty combinations were slower than normal decoding, the closest within 1%. The
  batch-size-2/depth-2 and batch-size-8/depth-3 speedups published for this model
  on this GPU are withdrawn, and the engine now decodes normally at every batch
  size. Asking for speculation
  explicitly still works. On Strix Halo, `Q4_K_M` keeps exactly one automatic
  setting: strict kernel plan, BF16 cache, one request at a time, three draft
  tokens, and prompts of 67 tokens or fewer.
- **Two device reserves are smaller.** On RDNA 3, hipEngine now sets the per-process
  scratch single-limit to 8 MiB rather than ROCm's 140 MiB, freeing 132 MiB per
  process per GPU that nothing was using; set
  `HSA_SCRATCH_SINGLE_LIMIT=146800640` to restore the old reservation. On Strix
  Halo, the default is now `GPU_MAX_HW_QUEUES=2` instead of `1`. The Laguna
  mixture-of-experts kernels were checked against two queues at short prompts.
  Neither value fixes the known long-context stall described under limits.
- **Qwen3.8-27B `Q4_K_S` on Strix Halo keeps its recurrent state in FP16** with
  FP32 accumulation. `HIPENGINE_GGUF_FP16_RECURRENT_STATE` is on for that model
  and GPU after engine and serving comparisons came out at least as fast with no
  extra memory. Set it to `0` for FP32 state. Speculative decoding and the chain
  journal still require FP32.
- **Fairer scheduling by default.** For `Q4_K_M` on both AMD backends, the
  scheduler now picks `fair` rather than `protect_decode` when
  `HIPENGINE_PREFILL_DECODE_POLICY` is unset, so a long prompt shares each loop
  tick with decoding instead of monopolising it.
- **Batched GGUF prefill and decode are on by default**
  (`HIPENGINE_GGUF_AR_PACKED_PREFILL`, `HIPENGINE_GGUF_AR_PACKED_DECODE`):
  requests that arrive together are prefilled and decoded in one pass instead of
  one slot at a time. Set either to `0` to force the one-at-a-time path when
  comparing the two. The rejected `HIPENGINE_GGUF_AR_STREAM_PREFILL` setting was
  removed.
- **Python 3.11 is the minimum supported version.** Python 3.10 is no longer
  packaged or tested, and the install requirements say so.
- **The INT8 KV route serves concurrent requests faster.** Row-batched direct
  INT8 decode now handles up to four requests in one pass. Against the same
  server serving the same requests one at a time, aggregate complete-request
  throughput rose to **1.25x at two requests** and **1.42x at four**; one
  request is unchanged (0.99x). This applies to the explicit
  `--kv-storage int8_per_token_head` route. Per-request latency still grows with
  the number of concurrent requests.
- **Long prompts on the INT8 KV route use less memory and no longer block other
  requests.** The packed prefill now runs layer-outer by default, sharing one
  BF16 oracle pair per request instead of one per retained layer: tracked peak
  memory fell by **0.438 GiB** at every multi-chunk prompt length, with identical
  generated tokens and no measured throughput regression. The same change makes
  the resumable, bounded-yield prefill the default for leased requests, so
  prefill work yields to in-flight decoding instead of running to completion
  first. Set `HIPENGINE_GGUF_PACKED_LAYER_OUTER=0` to restore the previous
  executor.
- **Published numbers were re-measured for this release,** not copied forward:
  the current rows come from runs dated 2026-08-03 through 2026-09-11, and each
  table names its model, protocol, and hardware. Rows from v0.4.0 and v0.5.0 use
  different models and protocols, so they are not an old-to-new speed comparison.
- **INT8 cache still saves no memory on dense 27B on Strix Halo.** The
  attention code INT8 cache needs is now written and checked against the CPU
  reference, but there INT8 with FP32 scales fails the short-prompt quality
  suite, and a mixed BF16/INT8 layer map that passes quality costs more memory,
  cannot use graph capture safely, and decodes about 10% slower than the BF16
  path. The default dense 27B cache stays BF16; the explicit W7900
  `int8_per_token_head` route is separately qualified.

### Fixed

- Streamed text from the speculative and GGUF paths is rebuilt one token at a
  time, GGUF special tokens no longer appear in streamed output, and a
  speculative stream stops cleanly at a special token.
- XML tool calls written by Qwen3.5-family chat templates are parsed correctly
  again.
- Fixed multi-request bugs where model state, cache ownership, graph reuse, or a
  change in request width could corrupt tokens later in a response or follow a
  request into the slot it reused.
- A request that does not qualify for speculative decoding now falls back to
  normal decoding before any GPU state changes, instead of failing or continuing
  on a route it does not qualify for. This covers long prompts,
  mixture-of-experts drafts, and speculative graphs beyond their supported length.
- Fixed teardown and cancellation in the speculative path so in-flight work
  finishes and its cache memory is released on client disconnect and server
  shutdown.

### Known limitations

- hipEngine uses one GPU. Multi-GPU inference and CPU model inference are not
  implemented.
- GGUF support covers the listed model families only; hipEngine does not run
  arbitrary GGUF architectures. See the model guides for per-model limits.
- NVIDIA Blackwell support is single-request Maple generation through the Python
  API. CUDA serving and multi-request execution are not ready.
- Maple uses greedy generation only.
- Repeated 128K-context runs on Strix Halo can still stall with low power draw
  and no progress, so no 128K number is published. A model's advertised context
  length is not a hipEngine support claim, so set a conservative server limit.
- What Qwen3.8-27B tolerates on a 24 GB card is unmeasured, and INT8 cache shows
  no saving there. The probe that would answer this needs its measurement gaps
  closed before any limit can be published
  ([capacity notes](docs/campaigns/QWEN38-27B-GFX1100-24GB-CAPACITY.md)).
- Many simultaneous requests work but are not inside latency targets. On Strix
  Halo, Qwen3.8-27B passes its one-to-eight physical and one-to-thirty-two
  logical request checks, yet at 32 requests it reaches 10.590 tok/s with an
  18.617 s 95th-percentile first-token time, and 0 of 3 target runs pass.
- Automatic speculation covers only narrow measured shapes: Qwen3.8-27B
  `Q4_K_M` on Strix Halo with one request at a time and prompts of 67 tokens or
  fewer, and two qualified Qwen3.6 batch sizes on the W7900. Everything else
  decodes normally unless you ask for speculation explicitly.
- Learned cache eviction (DMS) and reusable prompt-prefix state stay off by
  default wherever they are not qualified.
- Published wheels are Linux x86-64 and require glibc 2.39 or newer, such as
  Ubuntu 24.04. ROCm 7.x is the recommended AMD runtime for this release.
- APIs and supported combinations can still change before 1.0.

## v0.4.0 - 2026-08-10

This is a large alpha release focused on making hipEngine useful for more local
models and more than one request at a time. It adds Laguna S 2.1, Maple-Preview,
and native Moonshine ASR runtime work, broadens Qwen GGUF support, and introduces
experimental CUDA paths on NVIDIA Blackwell. AMD RDNA 3 and RDNA 3.5 remain the
primary platforms.

### Added

- Added public loading, text generation, streaming, chat, reasoning, and tool
  support for Laguna S 2.1 `Q4_K_M` on Ryzen AI MAX+ 395 / Radeon 8060S
  systems. The matching Laguna DFlash model is available as an explicit option;
  normal autoregressive generation remains the default.
- Added direct support for the official 2-bit Maple-Preview checkpoint on
  `gfx1100` and `gfx1151`. AMD generation can share one resident model across
  multiple active requests and reclaim finished or cancelled request slots.
- Added experimental `cuda_sm120a` support for single-request Maple generation
  on NVIDIA Blackwell. This path loads the same 2-bit checkpoint directly and
  uses native CUDA prompt processing and generation kernels.
- Added native Moonshine ASR runtime and kernel work for Radeon 8060S and NVIDIA
  Blackwell, including a tuned FP16 decoder and a torch-free CUDA encoder. A
  public audio-to-transcript API remains under development.
- Added Qwen3.5/Qwen3.6 GGUF support for additional common and importance-matrix
  formats, including `Q4_K_S`, `UD-Q3_K_M`, and `UD-Q4_K_M` where listed in the
  model support table.
- Added resident multi-request execution for supported Qwen GGUF and PARO
  routes. The engine chooses only request widths that passed the corresponding
  correctness checks.
- Added device-side GGUF sampling, reusable prompt-prefix state, stop-safe
  streaming, exact generated token IDs in streaming responses, and more
  detailed tokenizer and request timing.
- Added opt-in speculative providers and a complete native Qwen GGUF
  speculative cycle. These routes remain explicit when output differs from
  normal generation or when the speed benefit is not reliable.

### Changed

- Improved Qwen GGUF and ParoQuant prompt processing, generation, memory
  ownership, and multi-request throughput on both AMD backends. The current
  measured results and full test conditions are in
  [`benchmarks/README.md`](benchmarks/README.md).
- Improved Laguna loading, prompt processing, generation, and server latency
  through native kernels and resident session reuse.
- Changed GGUF text encoding to use the Hugging Face `tokenizers` library while
  keeping model execution torch-free.
- Rewrote the root README around practical installation, model/GPU
  compatibility, first server startup, and plain-language limitations.
- Clarified the Qwen format choice: optimized ParoQuant W4 remains the slightly
  faster and lower-memory option for Qwen3.6 35B-A3B in current AMD tests, while
  ongoing compatibility work now focuses on GGUF.

### Fixed

- Fixed several multi-request state, KV-cache ownership, graph-reuse, and
  request-width transition bugs that could affect later tokens or reused
  request slots.
- Fixed cancellation and disconnect cleanup so streaming requests release their
  reservations and background work reliably.
- Fixed sampled GGUF prefill, end-of-sequence handling, structured output, and
  Qwen tool-call cleanup across resident and streaming paths.
- Fixed long-prompt and sliding-window state handling for the newly supported
  Laguna and Maple paths.

### Known limitations

- hipEngine still uses one GPU. CPU model inference and multi-GPU inference are
  not implemented.
- GGUF support is model-specific; hipEngine does not yet run arbitrary GGUF
  architectures.
- CUDA text-generation support is limited to direct, single-request, greedy
  Maple generation. CUDA HTTP serving and multi-request execution are not
  included in v0.4.0.
- Moonshine currently exposes internal runtime and benchmark surfaces rather
  than a public audio-to-transcript API.
- Maple sampling is greedy-only. Model-advertised maximum context lengths are
  not blanket hipEngine support claims.
- Speculative modes can trade output equivalence for speed and remain opt-in
  where appropriate.
- Published wheels are Linux x86-64 and currently require glibc 2.39 or newer.
  ROCm 7.x is the recommended AMD runtime for this release.

## v0.3.0 - 2026-07-13

Minor release expanding hipEngine from the initial resident PARO/GGUF runtime
into a substantially broader Python and OpenAI-compatible serving surface, with
normal sampling, local-agent features, exact token accounting, and guarded
speculative decoding.

### Added

- Expanded the public Python API with `LLM.generate_detailed()`,
  `stream_detailed()`, `stream_many_detailed()`, tokenizer helpers, resolved
  backend/quant inspection, and a model-owned detailed MTP route. Detailed
  outputs can carry exact generated token ids, per-token logprobs, structured
  finish details, and backend execution telemetry.
- Added normal sampling for the PARO and GGUF generators: `top_k`, `min_p`,
  repetition/presence/frequency penalties, logit bias, token suppression,
  deterministic seeds, minimum-token/EOS policy, token and multi-token stops,
  logprobs/top-logprobs, and `n>1` choice lowering. Supported PARO request
  shapes use the native GPU sampler by default; other shapes fail over to the
  host sampler with explicit fallback metadata.
- Added exact token-id prompts to direct generation and non-streaming text
  completions. Responses expose exact prompt hashes/counts and generated-token
  accounting so usage and benchmark tooling do not need to re-tokenize decoded
  text.
- Added OpenAI-compatible tool calling, including `tools`, `tool_choice`,
  parallel-call policy, streaming argument fragments, tool transcript
  validation, strict JSON Schema result validation, and stable invalid-tool
  diagnostics.
- Added structured-output result validation for JSON object/schema, guided
  JSON, choice, regex, and unified-diff requests. Object-root JSON can use
  tokenizer-lowered close-suffix forcing when safe; this is not full
  grammar-constrained decoding.
- Added Qwen thinking/no-thinking controls, reasoning-effort and token-budget
  aliases, host-sampler soft/hard thinking closure, EOS suppression while
  reasoning, and separate reasoning-content/token telemetry.
- Added request deadlines, cooperative cancellation, deterministic buffered
  continuation handles, app-local transcript sessions, session
  fork/rollback/snapshot operations, and `new_session` /
  `truncate_oldest_visible` context-overflow policies.
- Added `/ready`, `/v1/hipengine/capabilities`, session-management, tokenizer,
  token-counting, and context-fit endpoints. Optional Prometheus output exposes
  generation queue, request, scheduler, and KV-pool counters.
- Added compatible-request coalescing, prompt-list batching, per-row request
  ids/seeds, queue and active-request admission caps, `n>1` lowering, detailed
  choice timing ownership, and generation-shape metadata.
- Added native Qwen3.6 GGUF NextN/MTP loading, proposal, verification,
  acceptance, commit, and public detailed-generation support. The server has a
  guarded, explicit, non-streaming greedy `llama-compat` MTP route for GGUF
  models with NextN tensors. Native DFlash loading, drafting, verification, and
  benchmark/runtime building blocks are also available in-tree.
- Added state-bound GGUF decode-graph admission on gfx1100 for supported greedy
  windows of at least 24 transitions. Shorter, sampled, streaming,
  unsupported-KV, and rollback-sensitive routes remain eager.
- Added a top-level `hipengine` console command. `hipengine serve` launches the
  OpenAI-compatible server, `hipengine bench` lists or launches packaged
  benchmark helpers, and `hipengine version` reports package metadata.

### Changed

- FastAPI/Uvicorn server dependencies now install by default because most users
  want the OpenAI-compatible API. The old `hipengine-server` console script has
  been replaced by `hipengine serve`.
- `LLM(..., quant=)` and the server now default to `quant="auto"`, allowing the
  selected model plugin to choose its registered PARO or GGUF quant route.
- Chat requests that omit `max_tokens` now use a bounded, configurable 4096
  token default, clamped to remaining admitted context. Set
  `--chat-default-max-tokens auto` to retain the v0.2.2 full-remaining-context
  behavior.
- Unknown top-level generation parameters are rejected instead of being
  silently ignored. Optional feature failures use a stable OpenAI-compatible
  error taxonomy and capability manifest.
- Server and benchmark output now distinguish queue width, backend call width,
  verifier rows, timing ownership, sampler execution, and exact token counts.
  Production PARO batch routing fails closed to exact width-1 sessions when a
  native width does not pass the independent single-request oracle.
- Refreshed the retained W7900/gfx1100 README toplines with a clean six-shape
  PARO/GGUF/llama.cpp matrix, a W7900-local GGUF state oracle, corrected
  whole-device VRAM scope, and current PARO context-capacity evidence. The
  accepted rollup and exact commands are preserved in
  `benchmarks/results/2026-07-12-w7900-v030-8116c453-summary.json`.
- Corrected gfx1100 speculative-decode economics against production graph AR.
  Exact/default and explicit `llama-compat` MTP remain functional but no longer
  beat the fastest same-protocol autoregressive route on W7900; older
  eager-denominator speedup rows are historical only.

### Fixed

- Missing Hugging Face repo IDs now report that the full model ID is absent from
  the local cache instead of falling through to a misleading partial-path
  `config.json` error.
- Qwen PARO generation now recognizes the tokenizer/model EOS set, including
  `<|im_end|>` as well as `<|endoftext|>`, instead of continuing chat output to
  the length limit.
- Fixed GGUF decode-graph replay and speculative block-commit lifecycle bugs,
  including stale graph reuse after resident-state mutation.
- Hardened PARO/GGUF sampling, stop handling, exact usage accounting, tool-call
  parsing, session transcript validation, context admission, and startup scratch
  probes across eager, streaming, sampled, and speculative paths.

### Known limitations

- Production PARO native `c>1` decode remains disabled because current native
  candidates do not pass the independent `c=1` token/state/KV oracle. The HTTP
  batcher can coalesce requests, but this release does not claim true continuous
  decode or native multi-request throughput.
- GGUF MTP serving is explicit, non-streaming, greedy-fast, and uses the
  accuracy-traded `llama-compat` contract. Exact/default MTP serving and
  streaming MTP remain future work; automatic requests use exact AR fallback.
- Tool calling and structured outputs are prompt-and-parse/result-validation
  features, with limited safe token forcing. Full grammar-constrained decoding
  is not implemented.
- App-local sessions and continuation handles re-render/re-prefill transcript
  text; they do not save or reuse resident KV state.
- PARO 256K INT8 KV physically allocates below the 24 GiB portability gate, but
  fails the required Qwen3.6 128K/128 rollout quality gate. It is an allocation
  capacity result, not a supported or usable inference route.
- Tensor parallelism and other multi-GPU execution remain unimplemented.

## v0.2.2 - 2026-05-26

Patch release improving server startup context preallocation, KV memory
admission, and request defaults.

### Added

- Server-wide resident context/KV preallocation controls:
  `--max-context-tokens`, `--kv-storage`, `--kv-scale-dtype`, and
  `--kv-scale-granularity`. Eager startup prepares the resident PARO session for
  the configured context, and requests beyond that context or with a different
  KV policy are rejected instead of resizing/reloading the model.
- Automatic server context sizing when `--max-context-tokens` is omitted: after
  resident weights load, the runtime estimates the selected KV dtype plus
  persistent context metadata and preallocates
  `min(model_max_context_tokens, allocatable_context_tokens)`.
- Fast PARO retained-KV capacity estimate during resident session build. The
  runtime uses current `hipMemGetInfo` after model weights load to report the
  estimated max context for the selected KV dtype and for INT8 KV, warning when
  INT8 still falls below the model's advertised max context.

### Changed

- Chat requests that omit `max_tokens` now use `max_tokens=auto`, meaning the
  remaining admitted context (`max_context_tokens - prompt_tokens - 1`).

### Fixed

- Clean up partially-built PARO resident sessions if capacity preflight or
  allocation fails, avoiding leaked resident buffers on startup/admission OOM.

## v0.2.1 - 2026-05-25

Patch release improving server session management, streaming, and
OpenAI-compatible reasoning output.

### Added

- Eager model warmup on server startup: the configured model and a short
  warmup generation run before uvicorn reports ready, so the first real
  request does not pay load/compile cost. Controlled by `--eager-load` /
  `--no-eager-load` (default: on), `--eager-load-prompt`, and
  `--eager-load-max-tokens`, with `HIPENGINE_EAGER_LOAD`,
  `HIPENGINE_EAGER_LOAD_PROMPT`, and `HIPENGINE_EAGER_LOAD_MAX_TOKENS`
  environment variable equivalents.
- `LLM.stream()` method for single-prompt token-by-token generation when
  the underlying text generator supports it.
- Reasoning-content splitting for chat completions: `<think>…</think>`
  spans (Qwen/DeepSeek-style) are now separated into
  `message.reasoning_content` (non-streaming) or `delta.reasoning_content`
  chunks (streaming), matching the OpenAI reasoning-content convention.

### Changed

- PARO text generators and their resident sessions are now cached on the
  `LLM` instance and reused across requests. Session capacity is bucketed
  (floor 4 Ki tokens, configurable via `HIPENGINE_SESSION_MIN_TOKENS` and
  `HIPENGINE_SESSION_BUCKET_TOKENS`) so normal chat-history growth does not
  force reallocation every turn.
- Chat `stream=true` now yields token-level SSE chunks from the resident
  decode loop instead of buffering the full response and wrapping it in a
  single SSE frame.
- Chat completions default `max_tokens` raised from 16 to 8192 so clients
  that omit the field get usable reply lengths, including verbose
  chain-of-thought reasoning.

### Fixed

- Fixed `LLM.generate()` re-resolving the generation factory on every call,
  which discarded generator-local caches and caused the PARO resident
  session (layer weights, KV buffers) to be allocated and freed per request.

## v0.2.0 - 2026-05-25

Minor release for the GGUF runtime path and W7900 benchmark refresh. GGUF is a
meaningful new model-loading surface rather than a patch-level fix, so this
supersedes the previously planned v0.1.2 patch.

### Added

- Added Qwen3.6 35B MoE GGUF support for `Q4_K_M` and `Q4_K_S` model files,
  including resident GGUF loading, bulk prefill, graph-replay decode,
  decode-repacked T16 layouts, and WMMA/GEMV fast-path controls used by the
  W7900 benchmark profile.
- Added `docs/ENVS.md` as the canonical environment-variable reference, including
  TheRock ROCm process setup, cached-build profiling guidance, and safe GGUF
  benchmark profiles.
- Added a persistent README sweep harness that loads each hipEngine model once
  and runs repeated in-session workload measurements, matching llama-bench-style
  repetition without multiplying model load/decode-repack time by every shape.

### Changed

- Refreshed W7900 README performance tables with 5-run persistent-session medians
  for packed PARO and GGUF Q4_K_S while keeping the existing llama.cpp HIP/Vulkan
  comparison rows unchanged.
- Documented the current GGUF tradeoffs: higher one-time load cost and resident
  memory from decode-repack, Q4_K_S preferred for tighter VRAM budgets, and
  performance still behind PARO on some shapes while already competitive in the
  broader W7900 comparison.

### Fixed

- Fixed the PARO resident prefill workspace-overlap regression that shipped in
  v0.1.1: short and mid prompts now keep prefill workspaces resident through
  32K tokens, restoring 512/128-class prefill throughput while retaining the
  long-context memory-saving path for prompts above 32K when active chunking
  splits the prompt.
- Fixed GGUF non-split full-attention decode in max-context persistent sessions
  by launching the context kernel with the active decode context instead of the
  session's maximum allocation length.

### Known limitations

- GGUF support remains alpha: production correctness and performance coverage is
  strongest for the documented Qwen3.6 35B MoE Q4_K_M/Q4_K_S paths on gfx1100,
  and other GGUF quants/models require local validation.
- GGUF model load is slower than packed PARO on the same host because current
  decode-repack happens on load and is not yet cached on disk.

## v0.1.1 - 2026-05-19

Patch release focused on long-context memory documentation and the INT8 KV cache
bring-up that landed after v0.1.0.

### Added

- INT8 KV cache policy controls and dispatch coverage for Qwen/PARO resident
  inference paths, including CPU/layer/E2E correctness gates and memory audits.
- Documented Qwen3.6 packed PARO memory rows for 128K BF16 KV, 128K INT8 KV, and
  256K INT8 KV on W7900/gfx1100, with retained-KV and loaded-weight VRAM notes.

### Changed

- Reduced the 256K INT8 KV tracked allocator high-water mark below the 24 GiB
  class target by releasing/reusing prefill scratch and AOTriton query buffers.
- Clarified that packed vs unstripped PARO checkpoint size does not translate to
  meaningfully different resident model-weight VRAM for the current text runtime.

### Known limitations

- INT8 KV correctness is gated by deterministic fixtures and layer probes; it is
  not yet a long-rollout perplexity or compounding-error study.
- Qwen3.6 packed throughput rows remain diagnostic pending a promoted public
  `LLM.generate()` correctness/repetition gate.

## v0.1.0 - 2026-05-18

Initial public alpha release.

### Added

- Torch-free Python runtime hot path for local ROCm inference bring-up.
- Plugin registries keyed by model/backend/quant/layer variants.
- HIP backends for `gfx1100` and `gfx1151`, plus `backend="auto"` detection with
  `HIPENGINE_BACKEND` force override guidance for nearby targets.
- Qwen3.5/Qwen3.6 PARO W4 runtime path, JIT HIP build/cache plumbing, AOTriton
  prefill runtime packaging, and OpenAI-compatible server entry point.
- CPU reference kernels and focused correctness/performance documentation.

### Packaging

- PyPI project name: `hipengine`.
- Python import package: `hipengine`.
- Canonical repository/wordmark: `hipEngine`.
- Release wheels are Linux x86-64 `manylinux_2_39` platform wheels because the
  package bundles a ROCm/AOTriton shared-library runtime; ROCm runtime libraries
  remain external system dependencies.

### Known limitations

- Alpha-quality API and model coverage; expect sharp edges outside the documented
  Qwen/PARO paths.
- Default supported GPU targets are `gfx1100` and `gfx1151`; other AMD targets
  require explicit backend forcing and local validation.
- Model weights are not distributed with the package.
