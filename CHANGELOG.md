# Changelog

All notable user-facing changes for hipEngine releases are documented here.

This changelog is for package/API releases. Performance rollup history remains in
[`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md), with detailed benchmark
evidence under [`benchmarks/results/`](benchmarks/results/).

## Unreleased

## v0.5.0 - 2026-09-06

hipEngine now runs dense Qwen models, not only the Qwen 3.6 35B
mixture-of-experts models. Several performance choices that used to need manual
settings now happen on their own, but only where they have been measured as safe
for the model and shape in use. Everything below was tested on Radeon RDNA 3
(`gfx1100`: RX 7900 XTX, Pro W7900) and on Strix Halo (`gfx1151`: Ryzen AI MAX+
395 with Radeon 8060S). Numbers live in
[`benchmarks/README.md`](benchmarks/README.md) and the dated performance history
in [`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md).

### Added

- **Dense Qwen 27B models.** Qwen3.6-27B and Qwen3.8-27B now load, generate, and
  serve from GGUF `Q4_K_M` on both AMD backends. Qwen3.8-27B also runs in
  `Q4_K_S` on Strix Halo. Both sizes can use speculative decoding driven by the
  model's own multi-token prediction (MTP) head. Qwen3.8-27B is measured from one
  request up to eight running at once, alongside two llama.cpp HIP builds
  measured the same way.
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
- **Pick a kernel plan explicitly.** `LLM(execution_profile=...)`,
  `--execution-profile`, and `HIPENGINE_EXECUTION_PROFILE` accept `strict`,
  `production`, or `batch_invariant`. hipEngine checks that the plan's kernels
  and their plain-decoding fallbacks are installed, refuses an unregistered
  combination instead of guessing one, and reports the plan's hash in server
  metadata. Leave it unset to keep the behaviour you had in v0.4.0.
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
- **Published numbers were re-measured for this release,** not copied forward:
  the current rows come from runs dated 2026-08-03 through 2026-09-06, and each
  table names its model, protocol, and hardware. Rows from v0.4.0 and v0.5.0 use
  different models and protocols, so they are not an old-to-new speed comparison.
- **INT8 cache still saves no memory on dense 27B.** The attention code INT8
  cache needs is now written and checked against the CPU reference, but INT8 with
  FP32 scales fails the short-prompt quality suite, and a mixed BF16/INT8 layer
  map that passes quality costs more memory, cannot use graph capture safely, and
  decodes about 10% slower than the BF16 path. Dense 27B cache stays BF16.

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
  ([capacity notes](docs/QWEN38-27B-GFX1100-24GB-CAPACITY.md)).
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
