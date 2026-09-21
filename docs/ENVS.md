---
status: current
owns: Complete environment-variable reference and the recommended profiles for normal use, ROCm setup, and profiling.
---
# Environment variables

Last updated: 2026-09-20

This is the complete env-var reference for hipEngine. Every environment
variable read by the runtime, the tests, or the bundled benchmark/development
harnesses is listed here, organized by subsystem. Most users should not need
any hipEngine-specific env vars for normal `LLM.generate()` use; prefer
Python/CLI arguments when available. Use env vars mainly for backend forcing,
ROCm/TheRock process setup, cached-build profiling, rollback/bisection of a
retained default, and explicitly documented benchmark or diagnostic profiles.

Boolean values generally accept `1/true/yes/on` as true and `0/false/no/off` as
false unless the variable says otherwise. Unless a row says otherwise, a
variable is read when the relevant module first resolves its route, so set it
before constructing the session or server.

Reading guide for the "Default" column: a default of `unset` means the
variable's absence selects a code default described in the row. A
"retained default" is a shipped, gated default you can roll back with an
explicit `0`; a "diagnostic" or "R&D" default is off or unset and exists for
bisection, screening, and campaign artifacts, not for production.

This file is kept complete by `scripts/check_envs_docs.py`, which scans the
tree for every environment-variable read and fails while any name is missing
from this document (or from its "Names that are not environment variables"
appendix). Run it after adding or renaming an env knob.

## Recommended profiles

### Normal local use

- No hipEngine env vars required when `backend="auto"` detects a native target.
- `LLM(model)` and `hipengine serve --model ...` resolve the model plugin's
  quantization. Supported GGUF models also select decode-repack and the public
  WMMA-prefill/GEMV-decode session profile.
- Server metadata reports the concrete backend, quantization, and the
  execution-profile manifest hash after the model loads. Leave
  `HIPENGINE_EXECUTION_PROFILE` unset for the default: a model, backend, and
  quantization with a certified plan runs `production`, and one with no
  registered plan keeps the migration path. A plan hipEngine cannot complete is
  an error, not a silent fallback.
- Set `HIPENGINE_BACKEND=hip_gfx1100` or `HIPENGINE_BACKEND=hip_gfx1151` only
  when auto-detection falls back or you are forcing a nearby target explicitly.
- Leave diagnostic fusion/tuning knobs unset.
- Leave `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` unset.

### TheRock ROCm process setup

For stable ROCm 10 installation/upgrade/rollback on gfx1151 and the separately
retained W7900 ROCm 7.13 stack, see [`THEROCK.md`](reference/THEROCK.md). This section only
shows the current gfx1151 clean-process wrapper.

TheRock installs ROCm pieces inside the Python environment. Build the process
environment around that prefix rather than mixing its libraries with
`/opt/rocm`. The canonical gfx1151 prefix is the dedicated `therock` environment
(Python 3.12, ROCm 10.0.0), not the Miniforge base prefix; the base interpreter
has no ROCm packages installed:

```bash
ENV_PREFIX=~/miniforge3/envs/therock
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$ENV_PREFIX/lib/python3.12/site-packages
ROCM_LIBS="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib"

env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
  SHELL="$SHELL" TERM="${TERM:-xterm}" \
  PATH="$ENV_PREFIX/bin:$ROOT/bin:$ROOT/lib/llvm/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$ROCM_LIBS" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_LIB_PATH="$ROOT/lib" \
  HIP_INCLUDE_PATH="$ROOT/include" \
  HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" \
  HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=. \
  "$PY" <command>
```

Use the W7900-specific wrapper in `THEROCK.md` for retained gfx1100 rows; its
legacy package directory is different. Do not set `HSA_OVERRIDE_GFX_VERSION` for
a real gfx1151 device. Use it on gfx1100 only as a measured local compatibility
workaround, never as a general hipEngine default.

### Multi-GPU ROCm device selection

Use one ROCm visibility filter per process when reserving a card for another
workload. For the current dual-gfx1100 lab host, GPU0 is the 48GB Radeon Pro
W7900 and GPU1 is the Radeon RX 7900 XTX; use GPU1/XTX for concurrency
re-baseline work so the W7900 stays free:

```bash
HIP_VISIBLE_DEVICES=1 python <command>
```

Before a long run, confirm the visible HIP device from the same shell:

```bash
HIP_VISIBLE_DEVICES=1 python3 - <<'PY'
import ctypes
hip = ctypes.CDLL('libamdhip64.so')
count = ctypes.c_int()
assert hip.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1
name = ctypes.create_string_buffer(256)
assert hip.hipDeviceGetName(name, ctypes.c_int(len(name)), ctypes.c_int(0)) == 0
print(name.value.decode(errors='replace'))
PY
```

Do not stack `HIP_VISIBLE_DEVICES=1` and `ROCR_VISIBLE_DEVICES=1` unless that
specific shell has been re-tested; on the current host that combination exposed
zero HIP devices, while either filter alone exposed the XTX.

### Benchmarking/profiling cached HIP builds

When using `rocprofv3` or repeated benchmark subprocesses, precompute the compiler
version and require cached builds so the measured/profiler process never spawns
`hipcc`:

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
  python scripts/qwen35_paro_bench.py ... \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
```

For reproducible GGUF Qwen3.6 MoE benchmark rows, keep the selected profile
explicit in the command even though the public generator selects it by default:

```bash
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
python scripts/qwen35_gguf_bench.py --persistent-session \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode ...
```

`HIPENGINE_GGUF_AOTRITON_PREFILL=v3` is no longer needed for the current default;
`v3` is already the default. Do not set
`HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS=1` for normal use; it is only
for reproducing old unsafe/R&D artifacts that deliberately bypassed the
qwen35moe fast-path safety gate.

## Core runtime variables

| Variable | Owner | Default | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_BACKEND` | Backend selection | unset / `auto` | Force a backend key such as `hip_gfx1100` or `hip_gfx1151`; otherwise auto-detects supported HIP arches and falls back to `cpu_reference` with a warning. |
| `HIPENGINE_EXECUTION_PROFILE` | Execution-profile plan selection | unset (shipped default) | `strict`, `production`, or `batch_invariant`; same as the Python `execution_profile=` argument and the server `--execution-profile` flag. Each value selects a kernel plan registered for that model, backend, and quantization, and hipEngine checks the plan's kernels and their fallbacks before running. An unregistered combination is an error, not a silent substitution. Leaving it unset selects `production` for a model, backend, and quantization combination with a certified production plan, and keeps the migration path otherwise; the migration path is not a fourth profile. |
| `HIPENGINE_HIP_ARCH` | HIP JIT build | unset | Force native HIP offload arch in build cache keys, e.g. `gfx1100` or `gfx1151`. The backend helper sets this temporarily when needed; benchmark harnesses set it explicitly to pin rows. |
| `HIPENGINE_HIP_OFFLOAD_ARCH` | HIP JIT build | unset | Alias-style fallback for `HIPENGINE_HIP_ARCH`. |
| `HIPENGINE_ROCM_DEVICE_LIB_PATH` | HIP JIT build | unset | Adds `--rocm-device-lib-path=<path>` to `hipcc`. Falls back to standard `HIP_DEVICE_LIB_PATH` if unset. Useful for TheRock. |
| `GPU_MAX_HW_QUEUES` | HIP runtime / gfx1151 branch concurrency | gfx1151: `2`; otherwise unset (ROCm default `4`) | Must be set before `libamdhip64` loads. hipEngine sets `2` only when every visible HIP architecture it recognizes is gfx1151 and you have not set a value; your own value always wins. Use `1` for the previous single-queue behaviour or `4` for ROCm's default when testing the scheduler. The Laguna shared and routed expert kernels were checked against two queues at short prompts, so neither `1` nor `2` says anything about surviving repeated 128K-context runs. |
| `HSA_SCRATCH_SINGLE_LIMIT` | HIP runtime / gfx1100 scratch reserve | gfx1100: `8388608` (8 MiB); otherwise ROCr default | Must be set before `libamdhip64` loads. ROCr 7.2.4 reserves 140 MiB per process per GPU up front and takes a slower allocate-once path above that limit. hipEngine lowers only the gfx1100 default to 8 MiB, which releases 132 MiB of reserve nothing was using and still covers the 300 MiB AOTriton allocate-once path. Your own value always wins; use `146800640` for the upstream 140 MiB. Mixed-architecture machines get no default. |
| `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` | HIP/ROCr runtime | unset (all devices) | Device visibility filters honored by the HIP runtime; hipEngine's auto-detection only sees what the filters leave visible. Do not stack both filters without re-testing (see above). |
| `HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256` | Execution profiles | set internally | Written by the execution-profile binder when a plan resolves; records the manifest identity of the running plan for metadata/provenance. Not a user input. |

## OpenAI-compatible server variables

| Variable | Default | CLI flag | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_API_KEY` | unset | `--api-key` | Optional bearer token used by `hipengine serve` when `--api-key` is omitted. |
| `HIPENGINE_KV_STORAGE` | `bf16` | `--kv-storage` | Server-wide KV storage policy: `auto`, `bf16`, or `int8_per_token_head`. BF16 is the higher-precision baseline and the only KV storage the speculative (MTP) route has retained evidence for. |
| `HIPENGINE_KV_SCALE_DTYPE` | `fp32` | `--kv-scale-dtype` | INT8 KV scale dtype: `fp16` or `fp32`. The retained INT8 KV qualification evidence for supported dense GGUF artifacts is keyed on fp32 scales; fp16 scales resolve to no qualified plan. |
| `HIPENGINE_KV_SCALE_GRANULARITY` | `per_token_head` | `--kv-scale-granularity` | INT8 KV scale granularity. |
| `HIPENGINE_GENERATION_BATCH_WINDOW_MS` | `0` | `--generation-batch-window-ms` | Opt-in cold-path coalescing delay for compatible HTTP requests. Default `0` adds no intentional delay; same-event-loop-turn requests may still share the batcher worker, while positive values are for explicit coalescer experiments. |
| `HIPENGINE_STREAM_QUEUE_MAX_CHUNKS` | `16` | `--stream-queue-max-chunks` | Bounded token-event queue per streaming HTTP request; must be at least 2. The resident loop independently keeps a bounded 64-event scheduling buffer per subscription to absorb transient cross-thread bursts. A client that remains slow enough to overflow it is cancelled with `budget_pressure=client_backpressure` instead of stalling unrelated rows. |
| `HIPENGINE_SHUTDOWN_GRACE_SECONDS` | `5.0` | `--shutdown-grace-seconds` | Grace period for queued/active generation to drain during server shutdown. At expiry, request cancellation tokens are tripped, producers are cancelled, and the long-lived model runner is closed. |
| `HIPENGINE_SPECULATIVE_MTP_SERVING` | `auto` | `--speculative-mtp-serving` | `off`, `opt_in`, `auto`, or `enabled`. `auto` uses speculative decoding only for the model, GPU, kernel plan, cache type, batch size, and prompt-length combination hipEngine has measured, and decodes normally elsewhere. `enabled` takes that speculative route for every compatible request on a dense Qwen model whose draft path matches normal decoding token for token, and falls back to normal decoding otherwise. `opt_in` exposes the `llama-compat` route only to a request that sends `speculative_mtp=true`; that route does not produce the same text as normal decoding. `off` disables speculation. |
| `HIPENGINE_SPECULATIVE_MTP_THINKING` | `hint` | `--speculative-mtp-thinking` | `hint` or `hard`. Speculative decoding stays token-exact only when nothing else decides the next token, and hipEngine's host-side thinking-budget enforcement (soft-close bias, end-of-sequence suppression, hard-close forcing) does exactly that. `hint`, the default, keeps the thinking markers in the rendered prompt but drops that enforcement, so a `reasoning_effort` request can use the speculative route. `hard` keeps the enforcement and runs thinking requests through normal decoding instead. A single request can override the server setting with `speculative_mtp: { "thinking": "hint" \| "hard" }`. |
| `HIPENGINE_SPECULATIVE_PROVIDER` | unset | `--speculative-provider` | Explicit provider registry key, such as `dflash`. Requires `HIPENGINE_DRAFT_MODEL`; ordinary requests remain AR and a request must send `speculative=true` (or the explicit object form). |
| `HIPENGINE_DRAFT_MODEL` | unset | `--draft-model` | Local pinned draft-model path paired with `HIPENGINE_SPECULATIVE_PROVIDER`. The Laguna DFlash owner validates its revision and content-addressed safetensors SHA before allocation. |
| `HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET` | unset | `--speculative-candidate-budget` | Optional candidate depth for the configured speculative owner. When unset, the server resolves the depth from the loaded model's retained serving evidence (the deepest qualified depth for the resident physical cell, reported as `candidate_budget.source: model_plugin_evidence`), or from the generic provider's declared shape (Laguna keeps B4, `provider_default`). Laguna admits only B4 and rejects mismatched request objects. A set value is used unchanged and reported as `explicit`; an unusable one (zero, negative, or deeper than the retained evidence qualifies) is reported at startup and in `/v1/hipengine/capabilities` rather than silently rewritten. |
| `HIPENGINE_MAX_QUEUED_REQUESTS` | unset | `--max-queued-requests` | Optional generation queue cap. When set and the server batcher queue is full, new generation requests fail with HTTP 429 `engine_busy` and `Retry-After: 1`. |
| `HIPENGINE_METRICS` | `off` | `--metrics` | `off` or `prometheus`. When `prometheus`, `/metrics` exposes additive HTTP counters plus resident pending/admitted/active occupancy, work-class totals and policy, bounded queue/TTFT/ITL/service/completion summaries, real KV byte/page/ref/pin/grow/shrink/failure stats, graph capture/replay/invalidation counts by bucket, and route/fallback counters when the loaded generator provides a live-loop snapshot. |
| `HIPENGINE_INFO` | `false` | `--info` | Per-request summary logging. When on, each completed generation request logs one `REQUEST_INFO` line with the endpoint, stream mode, served model, prompt/completion token counts, backend prefill time and rate, server-observed time to first token and stream window, wall time, the request's persistent KV allocation in the shared pool (the space it reserves for its own context, not the bytes its prompt touched), and the pool's own byte/page/pin/growth state. It logs no prompt or generated text. Stream timing is tracked even when the client did not request `stream_options.include_hipengine`; blocking requests have no first-token timestamp and report their decode rate from engine phase accounting instead, so the two bases are reported as separate fields (`stream_decode_ms`/`stream_tok_s` for the server-observed window, `decode_ms`/`decode_tok_s` for the engine's phases). |
| `HIPENGINE_LOG_COLOR` | `auto` | `--log-color` | `auto`, `always`, or `never`. `auto` colorizes a TTY only and honors `NO_COLOR` and `FORCE_COLOR`; `always` also colorizes redirected output such as a `| tee` pane and therefore writes ANSI codes into that file; `never` disables colorization. Styling adds ANSI codes only, so log text and `key=value` fields stay byte-identical for `grep` and log shippers. An unrecognized value falls back to `auto` rather than failing startup. |
| `HIPENGINE_PREFIX_CACHE` | `radix` | `--prefix-cache` | Defaults to `radix` in HTTP serving and the in-process engine. Eligible deterministic requests reuse page-aligned KV and hybrid state from active sources or completed-source snapshots, then prefill only the suffix. Ineligible requests prefill privately and record `prefix_fallback_reason`; this is an eligibility check, not a numerical detector. `/ready` reports cache ownership. The W7900 BF16 layout-repair packet covers four categories and four heldouts with both source lifecycles, exact state, and zero KL against private same-schedule recomputation. Serial-suffix arithmetic and INT8 reuse are separate contracts; see `docs/REFACTOR.md`. Set `off` to roll back. |
| `HIPENGINE_REPLAY_DIR` | unset | `--replay-dir` | Opt-in directory for finite JSON failed-request replay artifacts. Disabled by default for sensitive deployments. |
| `HIPENGINE_REPLAY_REDACTION` | `hash` | `--replay-redaction` | Replay artifact string redaction mode: `hash` replaces strings with SHA-256/length metadata, while `none` stores raw strings for explicit local debugging only. |
| `HIPENGINE_ENGINE_COMMAND_TIMEOUT_SECONDS` | `300` | `--engine-command-timeout-seconds` | Liveness budget for one command issued to the resident engine service, in seconds; must be positive. The service runs the engine on a single driver thread, so a command queues behind whatever that thread is already doing: a context prepare that grows the KV pool and captures graphs, or one prefill tick of a large prompt. Both legitimately exceed half a minute, so the default is generous and this value is not a bound on engine work. A command that waits longer than 10 seconds logs one `ENGINE_COMMAND_SLOW` line naming the blocking activity (`driver_command=`, `driver_command_s=`, `driver_tick_s=`, `queued_commands=`, `active_requests=`, `driver_alive=`); an exhausted budget raises an error carrying the same detail, which the server reports as HTTP 503 `engine_unavailable` (retryable), the same shape it uses for a closed engine service. A timed-out command is abandoned by its caller but still runs on the driver thread once that thread is free. |
| `HIPENGINE_EAGER_LOAD_PROMPT` | `one two three four` | `--eager-load-prompt` | Prompt text used for the server startup warmup. |
| `HIPENGINE_EAGER_LOAD` | `true` | `--eager-load` | Whether to warm the model/session during server startup. |
| `HIPENGINE_EAGER_LOAD_MAX_TOKENS` | `1` | `--eager-load-max-tokens` | Generated tokens used for the server startup warmup; must be positive. |
| `HIPENGINE_STARTUP_CHAT_SMOKE` | `true` | `--startup-chat-smoke` | Runs bounded production-shaped chat requests during eager startup, including one on the speculative MTP route when that route is enabled. |
| `HIPENGINE_STARTUP_SCRATCH_PROBE` | `true` | `--startup-scratch-probe` | Asks the backend to allocate max-context request scratch during eager startup without decoding to the output limit. The probe warms at the width the server can actually admit: the model's route width cap when no server-wide cap is set. |
| `HIPENGINE_STARTUP_MIN_FREE_MIB` | unset | `--startup-min-free-mib` | Optional minimum free GPU memory after startup warmup/probes; below this, startup fails. Default disabled. |
| `HIPENGINE_MAX_CONTEXT_TOKENS` | unset | `--max-context-tokens` | Resident session/KV context tokens preallocated at startup; unset takes the model/session default. |
| `HIPENGINE_CHAT_DEFAULT_MAX_TOKENS` | `4096` | `--chat-default-max-tokens` | Default `max_tokens` for chat requests that omit it; `N\|auto` where `auto` uses the remaining context. |
| `HIPENGINE_MAX_CHAT_SESSIONS` | unset | `--max-chat-sessions` | Optional app-local chat session cap before HTTP 429 `engine_busy`; unset is unlimited. |
| `HIPENGINE_REQUEST_TIMEOUT_MS` | unset | `--request-timeout-ms` | Default request deadline in milliseconds; omitted disables the default deadline. |
| `HIPENGINE_DEBUG` | `false` | `--debug` | Logs full HTTP request/response payloads and extra server diagnostics. |
| `HIPENGINE_SERVER_DEFAULT_AR_READY_COHORT` | unset (on) | none | When unset, the AR batcher groups ready rows into its default cohort; `0` disables the default-cohort grouping for tests/diagnostics. |
| `HIPENGINE_MTP2_MAX_CONTEXT_TOKENS` | unset | none | Resolves the MTP2 context window used by the dense speculative route; unset takes the model's retained window. Benchmark harnesses export it through `scripts/bench_env_preflight.py` to catch wrappers that silently drop it. |
| `HIPENGINE_MTP2_PREFIX_CHECKPOINT_ENTRIES` | `4` | none | Bounded provider KV/state checkpoints for MTP restoration after a target radix-cache hit. Capture is disabled when target prefix caching is off. Set `0` for the memory/bisection rollback; prefix hits without a provider checkpoint cannot restore MTP provider state. |
| `HIPENGINE_GGUF_SPECDEC2_MTP2_MAX_REQUESTS` | `4` | none | Server cap on concurrently active MTP2 (dense speculative) requests. |

## Vision (multimodal server input) variables

| Variable | Default | CLI flag | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_VISION_MODEL` | unset | `--vision-model` | Optional Qwen4Exp mmproj GGUF enabling bounded HTTP multimodal input. |
| `HIPENGINE_VISION_MAX_PIXELS` | unset | `--vision-max-pixels` | Decoded-pixel bound for HTTP vision input; unset takes the model plugin's bound. |
| `HIPENGINE_VISION_MAX_IMAGE_BYTES` | unset | `--vision-max-image-bytes` | Compressed base64 PNG payload bound for HTTP vision input. |

## HIP JIT build, compiler cache, and AOTriton discovery

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_COMPILER_VERSION_TEXT` | unset | Literal compiler-version text for cache keys; avoids probing `<compiler> --version`. |
| `HIPENGINE_COMPILER_VERSION_FILE` | unset | Reads compiler-version text from a file. Recommended for cached benchmarks/profiling. |
| `HIPENGINE_HIPCC_VERSION_TEXT` / `HIPENGINE_HIPCC_VERSION_FILE` | unset | Compiler-specific override for `hipcc`; takes precedence over the generic compiler-version vars. The same per-compiler pattern applies to other compiler basenames (e.g. `HIPENGINE_NVCC_VERSION_FILE` for `nvcc`). |
| `HIPENGINE_REQUIRE_CACHED_BUILD` | unset | When true, JIT builds must hit the build cache; a cache miss is an error instead of a `hipcc` spawn. Set by benchmark/test harnesses from `--require-cached-build` so a measured or profiled process never invokes the compiler. |
| `HIPENGINE_BUILD_CACHE_ROOT` | unset | Explicit build-cache root directory; used by the continuous-owner profiling harness alongside `HIPENGINE_REQUIRE_CACHED_BUILD`. |
| `HIPENGINE_AOTRITON_LIB` | unset | Explicit `libaotriton_v2.so` override. The matching `include/` and `aotriton.images/` trees must be in the standard release layout. |
| `HIPENGINE_AOTRITON_HOME` | unset | Explicit cache root containing `<version>/lib/libaotriton_v2.so`. Missing explicit roots fail loudly instead of falling back silently. |
| `HIPENGINE_CUDA_ARCH` | unset | CUDA-side target arch (e.g. `sm_120a`) for the `cuda_sm120a` backend tests and the Maple CUDA bench harness; the CUDA analogue of `HIPENGINE_HIP_ARCH`. |
| `HIPENGINE_CUDA_TARGET_ARCH` | unset | Alternative CUDA shared-library build-plan target spelling; `HIPENGINE_CUDA_ARCH` wins when both are set. |
| `HIPENGINE_CUTLASS_ATTENTION` | unset | Arms the AOT CUTLASS attention route for the CUDA sm120a Moonshine attention (default off, so the custom kernel stays the deployment path). Requires a source: `HIPENGINE_CUTLASS_ATTENTION_SO` (prebuilt `.so`, deployment) or `HIPENGINE_CUTLASS_DIR` (pinned CUTLASS checkout, development). |
| `HIPENGINE_CUTLASS_ATTENTION_SO` | unset | Prebuilt `.so` path for the armed CUTLASS attention route. |
| `HIPENGINE_CUTLASS_DIR` | unset | Pinned CUTLASS source root for the armed CUTLASS attention route (compiled through the hashed build cache); unset skips the CUDA CUTLASS gate test. |

Removed historical AOTriton knobs (`HIPENGINE_AOTRITON_SOURCE_ROOT` and
`HIPENGINE_AOTRITON_RUNTIME_ROOT`) are no longer read by the runtime.

## Continuous batching / engine-loop variables

These knobs are resolved by the public torch-free `LLM` adapter and passed to
one native resident scheduler/runner configuration. The gfx1100 GGUF BF16 path
uses them for its real request-sized device-KV pool; host/fake pools consume the
same contract in tests. The compatibility bridge preserves prompt-list ordering
with `protect_ttft`, and D4 still owns OpenAI streaming/backpressure lowering.
CLI flags with the same names (lowercase, dash-separated) override env values
when an adapter/parser calls `add_engine_loop_config_args(...)`.

| Variable | Default | CLI flag | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_PREFILL_DECODE_POLICY` | `protect_decode` | `--prefill-decode-policy` | One of `protect_decode`, `protect_ttft`, or `fair`. The independently gated gfx1100 and gfx1151 Q4_K_M package defaults select `fair` when the env is unset. |
| `HIPENGINE_MAX_ACTIVE_REQUESTS` | unset | `--max-active-requests` | Optional active resident request cap used as the engine-loop scheduler capacity when set; must be > 0. |
| `HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` | `256` | `--max-prefill-chunk-tokens` | Maximum prefill chunk tokens per loop tick; must be > 0. |
| `HIPENGINE_FAIR_PREFILL_BURST_CHUNKS` | `1` | `--fair-prefill-burst-chunks` | Maximum consecutive prefill chunks while `fair` scheduling also has decode-ready rows; must be > 0. The independently gated gfx1151 Q4_K_M package default may override this when the env is unset. |
| `HIPENGINE_KV_POOL_INITIAL_PAGES` | `128` | `--kv-pool-initial-pages` | Initial resident device-KV pages, clamped to the runner's maximum useful capacity; must be > 0. |
| `HIPENGINE_KV_POOL_LOW_WATER_PAGES` | `128` | `--kv-pool-low-water-pages` | Idle-shrink low-water pages, clamped with the initial allocation; must be > 0 and no greater than configured initial pages. |
| `HIPENGINE_KV_POOL_HIGH_WATER_PAGES` | unset | `--kv-pool-high-water-pages` | Optional atomic grow-on-admission page cap; unset means no configured pool cap. |
| `HIPENGINE_KV_POOL_CHUNK_PAGES` | `128` | `--kv-pool-chunk-pages` | Real device pages per grow/shrink tail chunk, clamped to useful runner capacity; must be > 0. |
| `HIPENGINE_KV_POOL_MEMORY_BUDGET_MIB` | automatic | `--kv-pool-memory-budget-mib` | Dense-GGUF KV payload ceiling in MiB, shared by the allocated arena (including pinned workspace pages) and private packed-workspace KV. Initial allocation, private fallback, and pool growth reject requests that exceed it. When unset, derived from live free HIP memory after reserve. Recurrent state, execution scratch, and pointer-table metadata are outside this KV payload ceiling and remain visible in allocator/workspace telemetry. |
| `HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS` | `30.0` | `--kv-pool-idle-grace-seconds` | Seconds before fully-free, graph-unpinned tail chunks are eligible to shrink; must be >= 0. |
| `HIPENGINE_MAX_PENDING_REQUESTS` | unset | `--max-pending-requests` | Optional pending request queue cap for the resident scheduler; must be > 0 when set. |
| `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET` | `1024` | `--round-prefill-token-budget` | Token-budget prefill work per engine-loop round; must be positive. |
| `HIPENGINE_ROUND_DECODE_ROW_BUDGET` | `32` | `--round-decode-row-budget` | Token-budget due decode rows per engine-loop round; must be positive. |
| `HIPENGINE_SUBMISSION_TRANSPORT` | `hipgraph` | none | Kernel-submission transport for graph replay: `hipgraph` (default), `aql`, or `pm4`. |
| `HIPENGINE_SESSION_MIN_TOKENS` | `4096` | none | PARO resident-session context floor when the caller does not require more. |
| `HIPENGINE_SESSION_BUCKET_TOKENS` | `1024` | none | PARO resident-session context rounding bucket; capacity rounds up to a multiple of this. |
| `HIPENGINE_KV_CAPACITY_RESERVE_MIB` | `512` | none | Safety reserve subtracted from free HIP memory when the PARO runner estimates KV capacity. |
| `HIPENGINE_SPEC_MTP_SPLIT_REFUSED_GROUPS` | `0` | none | Default-off experiment that lets the speculative planner split refused speculative groups. Reopened by the serving ladder via explicit opt-in; removal condition tracked in `docs/REFACTOR.md`. |

The table lists generic engine defaults. For each unset scheduler knob, the
registered gfx1151 Qwen GGUF `gguf_q4_k_m` generator refines the policy/chunk
pair to the F4-retained `fair:256`; explicit env values always win independently.
Other GGUF quants, gfx1100, and PARO retain their prior defaults until they pass
independent workload gates.

## GGUF variables

| Variable | Default | Classification | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_GGUF_DECODE_REPACK` | true | Retained release default with rollback opt-out | Materializes resident T16 decode layouts on load. The public Qwen3.6 GGUF path uses this accepted layout despite its load-time and resident-memory cost because raw decode is substantially slower. Set false only for diagnostics or memory comparisons. |
| `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` | backend-scoped (gfx1151 physical rows >=4; otherwise false) | Retained c4/c8 rollback and broad diagnostic | Unset keeps each backend's own rule: on gfx1151, one-, two-, and three-row Q8T16 decode uses the tiled kernels from four physical rows up, while two rows and all gfx1100 widths stay on the direct kernels. Set `0` to turn the tiled path off at every width (the separately measured eight-row pair policy is unaffected). Set `1` only to reproduce the wider-rows experiment, which costs 1.795% at two rows. |
| `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE` | false | Diagnostic | Enables the Q8T16 selected-pair rowtile helper; set by the MTP verifier rocprof harness for route attribution. |
| `HIPENGINE_GGUF_WMMA_PREFILL` | false | Low-level performance selector | Process-wide default for low-level GGUF sessions. The public generator passes `use_wmma_prefill=True`; benchmark CLI/session arguments remain explicit for artifact provenance. |
| `HIPENGINE_GGUF_GEMV_DECODE` | false | Low-level performance selector | Process-wide default for low-level GGUF sessions. The public generator passes `use_gemv_decode=True`. For qwen35moe, effective use is safety-gated unless decode-repack is active or the unsafe override is set. |
| `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` | false | Unsafe diagnostic | Bypasses qwen35moe GGUF fast-path safety. Do not set for normal use or promoted correctness claims. |
| `HIPENGINE_GGUF_AOTRITON_PREFILL` | `v3` | Attention implementation selector | `v3`, `v2`, or `auto`/`v2-if-safe`. `v2` is rejected for chunked suffix prefill because it has the wrong causal-mask semantics there. |
| `HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` | `1024` | Decode threshold | Context length where GGUF full-attention decode uses split/paged decode; `0` disables. Compatibility alias: `NANOVLLM_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT`. |
| `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG` | backend default for `MOSTLY_Q4_K_M` on gfx1151 | Rollback seam | Lets dense (non-MoE) linear-attention verifier rows stage their projections and FFN above the full-attention split threshold instead of taking the row-wise c1 route. The default reads the backend package capability `GGUF_STAGED_LINEAR_ROWS_LONG_DEFAULT_FILE_TYPES` and applies only to a loader-qualified plain artifact whose stamp is listed there (`mostly_q4_k_m` on gfx1151); a preset-bound (UD) or unknown-manifest artifact, an unlisted stamp, and gfx1100 all keep the row-wise c1 strict route. `0` forces the row-wise route and `1` forces the staged chain on any artifact; either value overrides the capability, and the variable never changes single-row decode. |
| `HIPENGINE_GGUF_GDN_PREFILL_MODE` | `auto` | Correctness/performance diagnostic | Selects `auto`, `exact`, `fused`, `chain`, or the named `chain_k2`, `chain_peer_wave32`, `chain_compact_peer_wave32`, `chain_peer_cluster8`, `chain_tile64`, `chain_tile32`, `chain_wave32`, `chain_wave32_tree`, `chain_lds64`, `chain_lds32`, `chain_lds32_direct`, and `chain_lds32_direct_nonvolatile` GGUF GDN prompt-prefill routes. `chain` is the raw-Q/K-plus-scale exact split fallback; `chain_lds32_direct_nonvolatile` is the promoted GPF-2E compact-scale/direct-`conv_out` route on gfx1151, while volatile direct and materialized `chain_lds32` remain rollback/bisection controls. The tile/wave/LDS64 routes are rejected diagnostics. `auto` uses backend-package policy: byte-exact direct LDS32 on gfx1151 and the bit-equivalent compact peer-wave route on gfx1100, with fused correctness fallback if the preferred route is unavailable. `chain_peer_wave32` retains the prior per-V-head materialization as rollback; `chain_compact_peer_wave32` stores normalized Q/K once per K head. Every explicit selection overrides backend policy and fails closed if its complete implementation is unavailable; invalid values are errors. |
| `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` | `1025` | Prefill threshold | Minimum rows for GGUF GDN recurrent-segments prefill routing; invalid values fall back to the default, values below 1 clamp to 1. |
| `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA` | backend-scoped (`1` through 4K on gfx1100/gfx1151; otherwise `0`) | LCP-M2 rollback/diagnostic | Selects the stream-ordered `prepare_prefill_chunk_metadata` kernel (`1`) or six synchronous host-prepared metadata copies (`0`). With the variable unset, gfx1100 and gfx1151 select the exact device path only through 4,096 prompt tokens; longer prompts retain synchronous metadata. Explicit values override the ceiling. Keep `0` for rollback and never force `1` at long context in production: the explicit gfx1151 128K one-queue lifecycle gate still entered the low-power no-progress state. |
| `HIPENGINE_GGUF_PREFILL_F16_STAGING` | profile-scoped | Rollback seam | Selects F16 activation staging for GGUF bulk prefill; the production profile enables it, strict does not (asserted in `tests/test_gpu_gguf_k_t16_dense_f16_activation_prefill.py`). |
| `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS` | gfx1100/gfx1151: `128`; otherwise `512` | LCP-4B rollback/diagnostic | Overrides the bulk-prefill `qwen35_router_select_kernel` workgroup with `64`, `128`, `256`, or `512`. Both gfx1100 and gfx1151 default to their independently full-model-exact 128-thread launch; decode retains its independent 256-thread geometry. Use `512` for rollback. Do not use `64` for production: it is fastest in a primitive screen but failed the gfx1151 4K full-model exact-state gate. |
| `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE` | `auto` | GPF-3A rollback/performance selector | `auto`, `baseline`, or `shared_x`. `auto` reads backend-package capability: gfx1100 and gfx1151 select their independently clean-gated byte-exact `shared_x` schedules. Explicit selections fail closed when their registry variant is unavailable; `shared_x` is mutually exclusive with the DS4 selected-prefill diagnostic. Leave unset for the production backend policy; use `baseline` only for rollback/A-B. |
| `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` | false | Capacity diagnostic | Offloads the raw Q8_0 token embedding from device residency and serves exact Q8_0→BF16 embedding rows from host. This can make Q4_K_M 128K fit on 24 GiB, but disables GGUF HIP decode graph replay and is not a promoted performance path. |
| `HIPENGINE_GGUF_DECODE_GRAPH` | true | Retained default with rollback opt-out | Enables GGUF resident decode HIP graph replay. Backend capability gates whether a graph is captured at all; `0` forces eager decode for bisection. |
| `HIPENGINE_GGUF_MOE_GRAPH` | false | Opt-in diagnostic | Graph-captures the selected-MoE decode segment in addition to the retained decode graph; used by decode-graph and PM4 profiling harnesses, off by default. |
| `HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS` | `8` | Correctness fallback | Number of leading GGUF full-attention layers kept as BF16 primary storage for long explicit `int8_per_token_head` sessions. Long contexts require at least 8 BF16-prefix layers unless `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` is set. The 2026-06-24 W7900 gate accepts prefix 8 at `128K/128` after the layer-local BF16 prefill-oracle fix (`KL mean=0.01448`, top-1 `0.96124`, no persistent BF16 mirror); prefix 7 still fails `128K/16`, and pure INT8 fails `4K/1`. Short contexts (`<=8192` rounded max context) still use the exact BF16 mirror instead. |
| `HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS` | unset | Unsafe diagnostic | Comma/range list of zero-based GGUF full-attention indices to keep as BF16 primary storage instead of using the leading-prefix rule, e.g. `0-5,7`. Requires `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` unless the list exactly matches the admitted default prefix. Added for non-contiguous sensitivity sweeps; 2026-06-24 W7900 `128K/16` masks with INT8 layers `{6,8,9}` and `{5,8,9}` still failed the BF16-vs-INT8 guard, so no custom mask is promoted. |
| `HIPENGINE_GGUF_INT8_KV_KEY_ONLY` | false | Unsafe diagnostic | For explicit GGUF `int8_per_token_head` sessions, store retained K as INT8 but V as BF16 for INT8-selected full-attention layers. This is a diagnostic key-only layout, not a promoted 24GB path: 2026-06-24 W7900 prefix `0` failed `4K/1`, prefix `6` failed `128K/16` top-1, and prefix `7` passed `128K/16` but saved less memory and had higher prefill peak than the admitted prefix-8 per-token/head path. |
| `HIPENGINE_GGUF_INT8_KV_BLOCK16` | false | Unsafe diagnostic | For explicit GGUF `int8_per_token_head` sessions, use the guarded block16 INT8 K/V scale granularity (`[blocks, block_size, kv_heads, 16]`) and route GGUF retained-KV write/decode through the block16 HIP kernels. This is a runtime diagnostic for the Q8-format follow-up, not a promoted path: 2026-06-24 W7900 forced-long `4K/1` BF16-vs-block16 gates fail top-1 even at prefix `8`. Do not combine with `HIPENGINE_GGUF_INT8_KV_KEY_ONLY`. |
| `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG` | false | Unsafe diagnostic | Allows long GGUF `int8_per_token_head` diagnostics below the verified 8-layer BF16 prefix (including pure INT8-only, key-only, block16, or non-contiguous BF16 layer masks when the custom mask env is set). Leave unset for normal use: pure INT8-only and lower memory-saving prefixes failed BF16-vs-INT8/GGUF hybrid logit gates and are capacity-diagnostic only. |
| `HIPENGINE_GGUF_COMPACT_MOE_C1` | false | Diagnostic fallback | Forces the older compact c=1 MoE decode scheduler; current retained default uses direct selected T16 kernels instead. |
| `HIPENGINE_GGUF_SIDECAR_CACHE` | `~/.cache/hipengine/gguf_sidecars` (or `XDG_CACHE_HOME`) | Sidecar cache | Cache directory for optional GGUF expert pack8 sidecars. |
| `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` | unset | Kernel R&D | Optional launch-bounds macro for selected WMMA prefill builds; unset uses the retained defaults. |
| `HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS` | false | Diagnostic instrumentation | Adds non-sync HIP-event intervals inside the GGUF packed target verifier and rolls them into `target_packed_verify_gpu_*` timing buckets. Leave unset for speed claims because event recording adds overhead. |
| `HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS` | `4096` on gfx1100/gfx1151 | Retained prefill rollback/scope override | Maximum multi-row compact selected-row count that skips the synchronous `wmma_total` scalar read. The retained path uses the tight routing-independent bound `16 * (A + floor((S-A)/16))`, `A=min(S,E)`, rather than the rejected old `selected_rows * 16` probe. SH9-D1 independently admits gfx1151: pp512 removes 40 `hipMemcpy`/copy dispatches per request with exact state and neutral unprofiled wall; set `0` to restore the scalar read. Rows==1 decode bypasses this helper. |
| `HIPENGINE_GGUF_AR_PACKED_PREFILL` | true | Retained default with rollback opt-out | Enables GGUF server greedy-AR packed final-row prompt prefill for c>N coalesced requests. Set `0`/`false`/`off` to force serial per-slot prefill fallback for bisection. |
| `HIPENGINE_GGUF_AR_PACKED_DECODE` | true | Retained default with rollback opt-out | Enables GGUF server greedy-AR packed resident decode for c>N coalesced requests. Set `0`/`false`/`off` to force stream/scalar fallback for bisection. |
| `HIPENGINE_GGUF_AR_STREAM_DECODE` | true | Retained fallback with rollback opt-out | Enables per-slot stream decode fallback and c>4 packed-decode chunk streams. Set false only for bisection. |
| `HIPENGINE_GGUF_FP16_RECURRENT_STATE` | backend/model scoped (true for gfx1151 `mostly_q4_k_s`; false otherwise) | Retained default with strict-storage rollback opt-out | Stores the Qwen3.8 `Q4_K_S` Gated DeltaNet recurrent state as FP16 while accumulating in FP32. On by default for that model on gfx1151 after engine and serving comparisons came out faster with no measured quality or isolation loss. Set `0`/`false`/`off` for FP32 state; set `1` to force it outside that scope for a comparison only. Speculative decoding and the chain journal keep FP32 state. |
| `HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL` | true | Retained default with rollback opt-out | Enables the GGUF MTP server packed prompt-prefill opener for eligible c=2/c=4 serving batches, returning FP32 prompt hidden rows for MTP catch-up. Set false for bisection. The c=8 first wave still uses the serial opener because the packed prefill path keeps the four-slot safety cap; the trailing c=2 wave uses packed prefill. |
| `HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT` | true | Retained default with rollback opt-out | Streams draft tokens from the GGUF MTP server route; `0` restores buffered draft emission for bisection. |
| `HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY` | true | Retained default with rollback opt-out | Streams verify results from the GGUF MTP server route; `0` restores buffered verify emission for bisection. |
| `HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER` | true | Retained default with rollback opt-out | Defers the verify scatter stage inside a streamed MTP server cycle; `0` restores the eager scatter order for bisection. |
| `HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP` | false (set by the server) | Internal coupling | Set to `1` by `hipengine serve` startup scratch-probe only when speculative MTP serving is enabled, so the warmup exercises the MTP route. Not a user knob. |
| `HIPENGINE_GGUF_MTP_HOT_VOCAB` | `auto` under `production`; otherwise unset | Performance selector | Hot-vocab LM-head cache scope for the GGUF MTP server. Unset resolves `auto` under the production profile and no hot-vocab cache otherwise; an explicit positive integer selects the head count and `0`/`off` disables the cache. |
| `HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT` | true | Retained default with rollback opt-out | Streams prompt-prefill progress on the specdec2 GGUF server route; `0` restores non-streaming prompt handling for bisection. |
| `HIPENGINE_GGUF_PACKED_KV_LEASE` | `0` | Diagnostic | Forces the packed-KV lease helper on for single-request (max_batch_size==1) sessions where it would otherwise stay off. |
| `HIPENGINE_GGUF_MTP_VERIFY_MODE` | `native` | GGUF dense MTP server | `native` or `serial_exact`. `native`, the default, checks drafted tokens with the fast llama.cpp-style row-attention and GPU-side acceptance path; on the dense Qwen suites that measures about 1.5-1.7x normal decoding, with occasional differences in the token chosen. `serial_exact` replays normal single-request decoding for each candidate, so it agrees with normal decoding token for token, but it cannot be faster. |
| `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` | `3` | GGUF dense MTP server | How many draft tokens the dense speculative path may propose per step, 1-4. Three is the measured default; a budget of four can be slower. This is the generator's own fallback: `hipengine serve` resolves an omitted `--speculative-candidate-budget` from the loaded model's retained serving evidence first, and falls back to this value only when no retained row describes the resident physical cell. |
| `HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS` | `off` | GGUF dense MTP server | Operator-only screening override for tuning campaigns. When on, a request that explicitly asks for speculation (`speculative_mtp: true`) may enter the dense MTP route for a physical cell no retained evidence row qualifies; the response then reports `qualification: explicit_screening_unqualified_cell`, `unqualified: true`, and the original rejection reason inside `speculative_mtp`. Automatic intent, and `auto`/`enabled` without an explicit request, stay fail-closed, and sampling-mode, artifact-identity, and memory-fit rejections stay fail-closed for every request. Screening measurements are diagnostics and are never retained evidence. |
| `HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M`, `HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_N` | `32` / `16` | Kernel R&D | Q4_K selected WMMA tile override (both must be set as an allowed pair; default pin is 32x16). Allowed tile pairs are validated by the build helper. |
| `HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_M`, `HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_N` | `16` / `16` | Kernel R&D | Q5_K selected WMMA tile override. |
| `HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_M`, `HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_N` | `16` / `16` | Kernel R&D | Q6_K selected WMMA tile override. |
| `HIPENGINE_GGUF_Q8_0_WMMA_TILE_M`, `HIPENGINE_GGUF_Q8_0_WMMA_TILE_N` | unset | Kernel R&D | Q8_0 dense WMMA prefill tile override; both must be set together and the pair is validated by the build helper. The production Qwen4Exp profile pins `64`/`32`. |
| `HIPENGINE_GGUF_DENSE_WMMA_BULK` | `1` | Retained default with rollback opt-out | Uses the dense BF16 WMMA bulk-prefill kernel for dense GGUF linear layers (backend capability `GGUF_DENSE_BF16_WMMA_BULK_PREFILL` permitting); `0` restores the non-WMMA bulk body. |
| `HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL` | `1` | Retained default with rollback opt-out | Uses the dense BF16 WMMA path for residual-class projections in bulk prefill; `0` restores the plain dispatch. |
| `HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK` | `1` | Retained default with rollback opt-out | Uses the pack8 Q4 WMMA bulk-prefill owner; `0` restores the non-WMMA pack8 body. |
| `HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL` | `1` | Retained default with rollback opt-out | Uses the pack8 Q4 dual WMMA SiLU prefill pair; `0` restores the separate owners (set to `0` by the Qwen3.5-0.8B cumulative role environment). |
| `HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL` | `1` | Retained default with rollback opt-out | Uses the Q8T16 dual WMMA prefill pair helper; `0` restores the separate per-projection owners. |
| `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR` | unset | Harness-set materialization | Retains raw GGUF bytes alongside T16 tiles for Q8_0 dense weights at materialization time. Required by the dp4a verifier/dense-Q8 diagnostic routes; set by the MTP bench harness before model load. |
| `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR` | unset | Harness-set materialization | Retains the X8 Q6_K sidecar for `lm_head` at materialization time; required by the verifier direct top-1 dp4a route. |
| `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW` | unset | Harness-set materialization | Keeps selected gate/up expert tensors in raw GGUF layout at materialization time; with the dp4a verify route the runtime then uses the raw q8_1/dp4a selected-dual body instead of the T16 replacement body. |
| `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` | unset | Harness-set materialization | Repacks selected gate/up expert tensors into the X8 replacement layout at materialization time. |
| `HIPENGINE_GGUF_SELECTED_X8_REPACK` | unset | Harness-set materialization | Repacks selected-down expert tensors into the X8 replacement layout; value is the repack mode string chosen by the harness. |
| `HIPENGINE_GGUF_RAW_SELECTED_DP4A` | unset | Diagnostic route | Enables the llama-style raw q8_1/dp4a selected-expert GEMV route (read live by the runner and the selected-expert caller contract). Default off; the accuracy-degrading variant fails the ja correctness gate, so only harness/gate runs set it. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A` | unset | Diagnostic route | Enables the dense Q8_0 dp4a pair helper; requires `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR=1` to be set before materialization. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` | unset | Diagnostic route | Extends the dense Q8 dp4a route across the full dense set; also enables the pair helper. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED` | unset | Diagnostic route | Uses the dp4a body for the shared gate/up/down projections (kept independently gated because the shared tensors are much smaller). |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` | unset | Diagnostic route | Uses the dp4a helper with F32 activations for direct-state `ssm_out`, which the BF16 denseq8all helper cannot cover. |
| `HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A` | unset | Diagnostic route | Verifier direct top-1 path reading the X8 Q6_K `lm_head` sidecar; production gates force it off. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN` | unset | Gate instrumentation | Makes the GDN prefill chain emit per-row captured state for offline full-logit/semantic gates; set by the execution-profile and MTP gate harnesses and by the production profile binder. |
| `HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE` | unset | Gate instrumentation | Reserves the GDN semantic-gate capture scratch and enables the semantic-gate comparison route; must be set before session construction so the scratch allocation precedes capture. |
| `HIPENGINE_GGUF_PACKED_VERIFY_CAPTURE_DIR` | unset | Diagnostic capture | Writes diagnostic BF16 output-norm rows for offline full-logit gates into this directory. |
| `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK` | unset | Kernel R&D | Overrides the maximum row chunk for the Q6_K `lm_head` decode path; unset uses the derived default. |
| `HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL` | profile-scoped | Rollback seam | Uses the integer MMQ prefill owner for Q6_K rows; enabled by the production profile binder, disabled under strict. |
| `HIPENGINE_GGUF_Q5_MMQ32_PIPE` | unset | Kernel R&D | Selects the pipelined MMQ32 consumer variant for Q5_K prefill dispatch; used by the MMQ test/bisect harness. |
| `HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS` / `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE` | harness defaults | Kernel R&D | Workgroup thread count and launch shape for the Q6 top-1 stage-1 draft kernel; set by the MTP draft bench/rocprof harnesses. |
| `HIPENGINE_GGUF_FUSED_MOE_FFN` | unset | Teacher-forced gate only | Set only inside `scripts/gguf_fused_moe_ffn_teacher_forced_kl.py` to A/B a fused MoE FFN candidate; not a runtime selector. |
| `HIPENGINE_GGUF_AR_D2_COST_ARTIFACT` | unset | Harness | Path to a measured D2 cost artifact; when set, `scripts/gguf_arbitrary_c_lifecycle.py` derives the D2 composition from the artifact instead of the ceiling heuristic. |
| `HIPENGINE_GGUF_AUTO_CONTEXT` | on | Retained default with rollback opt-out | When the caller does not pin a context, the resident GGUF session prices its footprint against free HIP memory after weights load and takes the largest block-aligned context that fits with a safety reserve. `0` keeps the historical fixed default. |
| `HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB` | measured (>= 2560) | Auto-context safety | Reserve subtracted from free memory while pricing the auto context; the default covers ~2.66 GiB of measured untracked device memory. |
| `HIPENGINE_GGUF_KV_TRANSIENT_KIB_PER_TOKEN` | measured | Auto-context pricing | Transient prefill workspace KiB per token used by the auto-context fit. |
| `HIPENGINE_GGUF_KV_TRANSIENT_FIXED_MIB` | measured | Auto-context pricing | Fixed transient workspace MiB used by the auto-context fit. |
| `HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS` | `4` | Auto-context retry | Maximum shrink-and-retry attempts after a failed context allocation; a failed allocation falls back to a smaller context instead of surfacing. |
| `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS` | narrow default | Prefix cache | Number of completed-source snapshots the radix prefix cache retains. The wider working set (16) is opt-in because it measured as a regression until the shared-admission contiguity path lands. |
| `HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES` | `1073741824` (1 GiB) | Prefix cache | Byte cap on retained prefix-snapshot state. |
| `HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX` | `512` | Prefix cache | Maximum gapped-suffix tokens a prefix-cache hit may reuse; measured against the full prefill a hit replaces. |
| `HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX` | true | Retained default with rollback opt-out | Batched reused-suffix ("extend") prefill; `0` restores the serial `session.step()` suffix loop for rollback/bisection. |
| `HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS` | `1..8` | Diagnostic | Comma/space-separated override of the shared-slot AR physical width set; the packaged production default is every width 1-8 after direct c3/c5/c6/c7 lifecycle certification. |
| `HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY` | true | Retained default with rollback opt-out | Next-N accepted-tail KV write-only path; `0` rolls back to the full write. |
| `HIPENGINE_GGUF_PACKED_LAYER_OUTER` | true | Retained default with rollback opt-out | Session-paired (one pair per session) packed layer-outer executor; `0` restores the per-layer-keyed chunk-outer rollback owner. |
| `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED` | unset | Unsafe diagnostic | Shorter spelling accepted alongside `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG` in the INT8-KV diagnostic override set. |
| `HIPENGINE_GGUF_INT8_KV_DECODE_GRAPH` | unset | Diagnostic | Admits INT8 KV storage into decode graph capture, and only for layouts the capture path accepts; BF16 is always admitted. |
| `HIPENGINE_GGUF_INT8_PREFILL_DIRECT` | unset | Correctness/perf gate | Selects the oracle-free direct INT8 prefill attention route (`twopass` candidate) instead of the oracle-bridge strict path (write-through BF16 oracle pair via AOTriton). |
| `HIPENGINE_GGUF_INT8_PREFILL_KERNEL` | `flash` | Route selector | Direct INT8 prefill attention kernel implementation: `flash`, `wmma`, or `sequential`; other values are errors. |
| `HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON` | unset | Diagnostic | Slot-local AOTriton variant of the INT8 prefill route (gate provenance key). |
| `HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS` | unset (adopted single plane) | Rollback seam | Declares the layer-outer hidden-plane alias mode for INT8 evaluation sessions; `0` rolls back to the multi-plane layout. |
| `HIPENGINE_GGUF_AOTRITON_PREFILL_ENABLE` | backend capability | Attention policy | Overrides the backend's AOTriton-prefill threshold policy (the measured 512-token crossover) on either side. |
| `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV` | unset | Attention R&D | Selects the optional contiguous head-major KV layout. Gapped BF16 prefix pages use the registered gather independently of that backend default; explicit `0` also disables the gapped gather. |
| `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES` | `536870912` (512 MiB) | Attention R&D | Byte cap for the head-major KV route. |
| `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS` | unset | Attention R&D | Token cap for the head-major KV route. |
| `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM` | unset | Attention R&D | Runs AOTriton prefill on an isolated stream for large query rows (>= 512). |
| `HIPENGINE_GGUF_GAPPED_GATHER` | unset (enabled) | Prefill R&D | Gather gapped BF16 prefix pages into demand-sized head-major buffers so suffix attention uses the contiguous route. Requires the registered copy kernel and obeys token/byte caps, including allocation rounding. `0` selects the native paged fallback. |
| `HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE` | `baseline` | Rollback seam | Linear-attention conv prefill mode: `baseline` or `tile32x128`. |
| `HIPENGINE_GGUF_DENSE_DOWN_RESIDUAL_DECODE` | true | Retained default with rollback opt-out | Exact model/backend/shape-qualified c1 down+residual owner; `0` restores the generic owner. |
| `HIPENGINE_GGUF_Q4K_ROWTILE` | true | Retained default with rollback opt-out | Small-B weight-amortized row-tile GEMV for raw K-quants and resident-pack8 Q4_K verifier continuation blocks; `0` disables for bisection. |
| `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE` | unset | Rollback seam | Rowtile variant policy for the Q4_T16 dual-SiLU prefill family. |
| `HIPENGINE_GGUF_Q4_T16_SINGLE_WAVE_MAX_ROWS` | unset | Kernel R&D | Row cap for the Q4_T16 single-wave rowtile variant. |
| `HIPENGINE_GGUF_Q4_T16_SHARED_B_ROW64_MAX_ROWS` | unset | Kernel R&D | Row cap for the Q4_T16 shared-B row64 variant. |
| `HIPENGINE_GGUF_Q8_T16_THREADS` | backend-scoped (`64` or `128`) | Rollback seam | Q8_0 T16 decode workgroup thread override; backend packages select independently retained defaults. |
| `HIPENGINE_GGUF_Q8_T16_PAIR_COL8` | unset | Diagnostic | Col8 pair-helper variant for Q8T16 decode (batch-route gate provenance key). |
| `HIPENGINE_GGUF_Q8_T16_ROWTILE_SHAPES` | `(5120,1024)` | Rollback seam | Shape-explicit Q8T16 rowtile admission set for the UD verifier. |
| `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE` / `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE` | unset | Diagnostic | Select the two-/four-wave Q8T16 wide WMMA prefill session variants (shape-scoped). |
| `HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE` | unset | Kernel R&D | Dense Q4_K WMMA prefill tile override (A/B harness writes `64x16`-style values). |
| `HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE` | `64x16` | Kernel R&D | Dense Q6_K WMMA prefill tile override. |
| `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` | unset | Diagnostic route | llama-compat selected-expert dp4a adapter; set by the llama-compat MTP environment map and read by the selected-caller contract. |
| `HIPENGINE_GGUF_T16_SELECTED_DP4A` | unset | Diagnostic route | llama-compat T16 selected dp4a adapter; set by the llama-compat MTP environment map and read by the selected-caller contract. |
| `HIPENGINE_GGUF_T16_DS4_PREFILL` | unset | Diagnostic route | DS4 T16 selected-prefill variant selector (compact-MoE WMMA routing tests). |
| `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE` | unset | Diagnostic route | T16 selected pair-reuse route selector (batch-route gate). |
| `HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE` | unset | Diagnostic route | T16 selected-down pair-reuse route selector. |
| `HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE` | unset | Diagnostic route | T16 selected Q6-down pair-reuse route selector. |
| `HIPENGINE_GGUF_IQ_DENSE_ROW_BATCH_DOWN` | unset | Rollback seam | IQ dense row-slab rule: unset picks the smallest slab at or above the row count (measured default); `1` restores the largest slab at or below it. |
| `HIPENGINE_GGUF_IQ_GROUPED_PREFILL` | backend-scoped | Rollback seam | Grouped compact prefill session for IQ-quant MoE pairs; `0` disables. |
| `HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS` | true | Retained default with rollback opt-out | Fuses the MoE tail with the next layer's input RMSNorm; `0` restores the separate chain (set to `0` in the MoE-graph parity test). |
| `HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE` | unset | Diagnostic | Wave-reduce variant of the GDN decode-order segments state-rows kernel. |
| `HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL` | unset | B1 transfer | Explicit override for the target-side WMMA prefill transfer: `1`/on forces the transfer, `0`/off restores the GEMV owners everywhere. |
| `HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE` | backend-scoped | Rollback seam | Long-context parallel-reduce decode route; `0` is the explicit opt-out kept by dispatch tests. |
| `HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT` | backend default | Decode threshold | Minimum context length admitting the parallel-reduce route. |
| `HIPENGINE_GGUF_ROW_COMPACT_GEMV` | unset | Diagnostic route | Parity-workbench candidate: compact grouped selected-MoE GEMV routing for the target block verifier. |
| `HIPENGINE_GGUF_VERIFY_ROW_LM_HEAD` | unset | Diagnostic route | Parity-workbench candidate: row-batched lm-head/argmax sampling in the target block verifier. |
| `HIPENGINE_GGUF_SELECTED_DOWN_RAW` | unset | Harness-set materialization | Keeps selected-down expert tensors in raw GGUF layout (workbench value `both`), pairing with the raw dp4a route. |
| `HIPENGINE_GGUF_C8_Q5_RAW_MMQ` | true (capability-gated) | Retained default | Gates the raw-MMQ Q5 sidecar on the `GGUF_C8_Q5_RAW_MMQ_SSM_OUT` backend capability. |
| `HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ` | unset | Diagnostic | Source-MMQ Q5 verifier-numerics route selector. |
| `HIPENGINE_C8_Q5_PLANAR_DP4A` | false | Diagnostic route | Gates the optional planar INT8 Q5 sidecar (uploaded at materialization) for the planar-dp4a route. |
| `HIPENGINE_C8_Q6_DP4A_GROUPED` | false | Opt-in diagnostic | Routes planar-decode rows 8-64 to the grouped integer-dp4a Q6 sibling (x quantized to q8_1); the BF16 grouped owner stays the exact default and strict fallback. |
| `HIPENGINE_GGUF_Q6_PLANAR_EXACT_PREFILL` | unset | Rollback seam | Selects the Q6 planar exact prefill owner over the default T16 route (resolved per session). |
| `HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES` | unset | Kernel R&D | Grouped target-rowtile admission override for the Q6_T16 decode wrapper. |
| `HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6` | unset | Rollback seam | Dense-rowtile grouped-rows6 target selector for the Q5_T16 gemv owner. |
| `HIPENGINE_UD_REPACK_ELIGIBILITY` | unset | Rollback seam | Overrides model-wide UD repack eligibility (e.g. `model-wide`) so a synthetic geometry keeps its historical pack8/raw residents. |
| `HIPENGINE_GGUF_SPECDEC2_NGRAM_MOD` | unset | Specdec2 route | N-gram modifier-window size for the specdec2 ngram mode. |
| `HIPENGINE_GGUF_SPECDEC2_NGRAM_MATCH` | unset | Specdec2 route | Required n-gram match count. |
| `HIPENGINE_GGUF_SPECDEC2_NGRAM_MIN` | unset | Specdec2 route | Minimum n-gram probe length. |
| `HIPENGINE_GGUF_SPECDEC2_NGRAM_PROBE_MAX` | unset | Specdec2 route | Maximum n-gram probe tokens. |
| `HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS` | unset | Specdec2 route | Exact target row-count override for the specdec2 verifier rows. |
| `HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES` | unset | Specdec2 route | Q6 mixed target rowtile admission override. |
| `HIPENGINE_SPECDEC2_DEVICE_CHAIN_ORACLE` | `0` | Diagnostic | Marks the adapter as device-chain-qualified by an external oracle instead of retained evidence; screening only. |
| `HIPENGINE_SPECDEC2_POST_REJECT_COOLDOWN` | `0` | Diagnostic | Default-off cooldown that keeps a request out of speculation for one cycle after a rejected speculative attempt. |
| `HIPENGINE_GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA` | backend-scoped | Rollback seam | Allocates private-c1 small weights from a session-owned arena; `0` restores the shared allocator path. |
| `HIPENGINE_GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA` | backend-scoped | Rollback seam | Allocates private-c1 decode scratch from a session-owned arena; `0` restores the shared path. |
| `HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS` | gfx1151: `1024` | Rollback seam | Overrides the short-prompt c1 batch-attention workgroup; the calibrated 1024-thread variant passed the execution-profile c1 threads gate and the exact 256-thread leaf stays the strict fallback. |
| `HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL` | unset | Diagnostic | Session-acquire-time diagnostic resolver that forces the WMMA-prefill route without touching the production session default. Unrecognised values raise rather than silently reading as route-unchanged. |

### GGUF target-verifier capture and F32-diagnostic family

These gate/probe flags drive the GGUF packed target verifier's offline
correctness gates and the F32 per-stage diagnostics. The capture flags make a
named intermediate observable for an offline gate; the F32 flags replay a
single verifier stage in F32 to localize divergence. All are gate/harness
inputs, not production knobs, and the production profile binder sets
`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN` and
`HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE` itself.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV` | unset | Captures prefill GDN chain conv state for offline gates. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN` | unset | Captures regular-chain GDN state. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT` | unset | Captures BF16 GDN output rows. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL` | unset | Captures prefill attention scores. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV` | unset | Captures F32 chain conv state. |
| `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL` | unset | Replays the residual stage in F32 (supports a subset of shapes). |
| `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT` | unset | Comma/range layer limit for the F32 residual replay; must be within `[0, layer_count]`. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM` | unset | Replays the post-attention norm in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_ROUTER` | unset | Replays the post-norm router read in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SELECTED_Q8` | unset | Replays the selected-Q8 post-norm read in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8` | unset | Replays the shared-Q8 post-norm read in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING` | unset | Replays token-embedding lookup in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM` | unset | Replays the attention input norm in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS` | unset | Replays linear-attention projections in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA` | unset | Replays the SSM alpha/beta gates in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT` | unset | Replays the attention output projection in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE` | unset | Replays the MoE combine in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN` | unset | Replays the selected-expert down projection in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE` | unset | Replays the selected-expert intermediate in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN` | unset | Replays the shared-expert down projection in F32. |
| `HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE` | set by the Qwen3.8 production profile binder | Production Q4 verifier rowtile admission. |
| `HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4` | default off | Wide (logical width >= 8) Q6 shared4 verifier candidate table. |
| `HIPENGINE_GGUF_ROUTER_F32W_COOP` | backend-scoped | F32-router cooperative scan route (cold-path policy key shared by runtime and batch gate). |
| `HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER` | backend-scoped | F32-router persistent-counter route. |

### GGUF backend-package capability seams (gfx1100)

These are retained defaults declared as backend-package capabilities; each has
an explicit env override for rollback/bisection. All are read by the adapter,
not by backend-branched runtime code.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GGUF_FUSED_PACKED_STATE_TRANSFER` | true | Exact resident-to-packed Conv/GDN state gather through the registered chunked pointer-table copy leaf. `0` restores the per-layer D2D transfer chain as explicit rollback. |
| `HIPENGINE_GGUF_VERIFY_DIRECT_RESIDENT_LINEAR_STATE` | true | Packed C5-C8 target verification indexes the resident multi-slot Conv/GDN slabs directly without mutation. `0` restores the fused resident-to-packed import. |
| `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48` | true | Specdec2 Q4 dual-SiLU rowtile owner at row48 for the qualified H5120 geometry (6144x5120 and 17408x5120 shapes). |
| `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28` | false | Default-off screen selecting the dense dual WMMA row32/row48 variants for specdec2 production R28 rows. |
| `HIPENGINE_GGUF_Q5_T16_GROUPED_ROWS8_C8` | true | Q5_T16 grouped rows-8 owner for physical C8 verifier packets. |
| `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2` | true | Q4_T16 native rowtile16 W2 variant for rows 32. |
| `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_ROWS6` | true | Grouped-rows6 variant of the rowtile16 W2 Q4_T16 prefill owner (kernel-side seam of the backend capability policy). |
| `HIPENGINE_GGUF_Q5_T16_ROWTILE_SINGLE_WAVE` | true | Q5_T16 single-wave rowtile owner; `0` restores the four-wave owner for rollback and bisection. |
| `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_PAIR_ROWS6` | true | Grouped-grid ownership for the rowtile16 W2 grouped pair at rows 6; `0` restores repeated R6 pair fallback. |
| `HIPENGINE_GGUF_Q4_T16_GROUPED_ROWS8_C5C6` | true | Grouped-R8 weight traversals for physical C5-C6 verifier R24 packets (C7-C8 stay on R6 because the same sibling regresses their gate). |
| `HIPENGINE_GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS` | true | Exact C7 R28 (grouped prefix plus strict tail) instead of padded R30. |
| `HIPENGINE_GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS` | true | Exact C8 R32 with the exact two-active-wave fused gate/up owner; `0` retains padded R36 plus row48. |
| `HIPENGINE_GGUF_ROUNDED_NORM_FIXED5120` | true | Fixed-5120 rounded-norm residual decode sibling (rows 2-8); `0` restores the generic rounded-norm tree. |

## Shared paged-attention decode variables

These affect both PARO and GGUF decode paths where applicable.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_PAGED_ATTN_MAX_SPLITS` | `4096` | Maximum split count used by PARO resident split-K decode config. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS`. |
| `HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX` | true | Enables grouped-GQA split decode for Qwen3.5/Qwen3.6 GQA shapes. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX`. |
| `HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_SPLITS` | `64` | Minimum split count that selects grouped-GQA split decode. |
| `HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_CONTEXT` | `4096` | Minimum context length that selects grouped-GQA split decode. |
| `HIPENGINE_PAGED_ATTN_WARP_SPLIT_CTX` | true | Enables warp-split GQA fallback where grouped-GQA is not selected. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_WARP_SPLIT_CTX`. |

## Fusion, dispatch, and retained-default rollback seams

Single-request and low-row dispatch defaults shared by the PARO and GGUF
runners. Every one of these is a shipped default with an explicit rollback
value; leave them unset in production and use the override only for bisection.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_MOE_C1_C_DISPATCH` | true | Master gate for the C-side MoE c=1 dispatch helper (default on after M14.dispatch.1 prewarm validation). `0` restores the Python router wrappers; diagnostics that need to intercept router calls set it themselves. |
| `HIPENGINE_FUSED_LINEAR_STATE_COMMIT` | true | Fused linear-attention state commit kernel; `0` restores the unfused commit chain in both PARO and GGUF runners. |
| `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` | true | Chunked (pointer-table) linear state commit; `0` restores the unchunked variant. |
| `HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED` | true | Fuses the linear-attention shared down/combine epilogue; `0` restores the parallel epilogue. |
| `HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED` | true | Fuses the linear-attention shared SiLU-rotate slice; `0` restores the unfused chain. |
| `HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED` | true | Fuses the full-attention shared down/combine epilogue; `0` restores the parallel epilogue. |
| `HIPENGINE_DISABLE_PACK8_OUTPUT_TILED` | unset | Set to any non-empty value to drop rows 2/4/8 from the pack8 output-tiled GEMV route (rollback for the byte-exact AWQ output-tiled path). |
| `HIPENGINE_W4_OUTPUT_TILED_PREFILL` | true | Output-tiled W4 prefill epilogue; `0` restores the plain epilogue. |
| `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL` | true | Stacked dual output-tiled split W4 prefill route (default on after the 2026-06-11 W7900 D32 9-prompt gate). |
| `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_SITES` | unset | Comma-separated site names restricting the dual output-tiled split route; unset uses the retained default sites. |
| `HIPENGINE_W4_MULTI_ROW_PACK8_SITES` | unset | Comma-separated site names enabling specific multi-row W4 pack8 sites, `all` for the full M12.6 set, or `none` to disable every multi-row W4 site while leaving the umbrella gate on. Unset uses the default safe sites. |
| `HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH` | unset | Selects the small-batch shared down-proj mode (`prefill`, `multi_row_decode`, or the named mode aliases); unset resolves `2` when the output-tiled prefill is on and `1` otherwise. |
| `HIPENGINE_W4_PREFILL_SMALLBATCH_TILE_M` | `16` | Output tile M for small-batch W4 prefill; one of `16`, `32`, `64`. The retained 16-row tile improves the small-B MTP prompt suite while preserving exact prefill numerics. |
| `HIPENGINE_MARLIN_K_MULTI_ROW_SITES` | unset | Comma-separated Marlin-K multi-row site names; unset uses the retained default sites. |
| `HIPENGINE_W8A16_LM_HEAD_MULTI_ROW` | true | Multi-row W8A16 LM-head decode path; `0` restores the per-row owner. |
| `HIPENGINE_SMALL_BATCH_DECODE_THRESHOLD` | `7` | Row-count threshold below which decode takes the small-batch multi-row route; `1` restores the pre-M7.C behavior exactly. |
| `HIPENGINE_W4_MULTI_ROW_PACK8` | true | M12.6 umbrella gate for multi-row pack8 W4 GEMV; `0` disables the family. |
| `HIPENGINE_W4_MULTI_ROW_PACK8_SINGLE` | follows the umbrella | Single-output multi-row pack8 override. |
| `HIPENGINE_W4_MULTI_ROW_PACK8_DUAL` | follows the umbrella | Dual-output multi-row pack8 override. |
| `HIPENGINE_W4_MULTI_ROW_SMALL_BATCH` | true | Weight-amortized multi-row W4 GEMV for small batches (win scales with row count); `0` disables. |
| `HIPENGINE_LINEAR_AB_DUAL_SEPARATE` | true | Uses the separate-output dual A/B GEMV for small-batch decode rows; `0` restores the single GEMV fallback. |
| `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED` | true | Fuses `f32_to_fp16` + `paro_rotate1_fp16` for verifier shapes (raw-FP16 bit-exact; removes 30 launches/pass at B=3). |
| `HIPENGINE_SHARED_PREFILL_SILU_ROTATE_FUSED` | true | Fuses shared-prefill SiLU-mul + rotate into one launch (default on after the 2026-08-16 gfx1151 byte-equality gate); `0` opts out. |
| `HIPENGINE_FUSED_RMSNORM_ROTATE` | false | Opt-in fused RMSNorm+rotate for linear layers (exact-AR preserved; pending verifier economics). |
| `HIPENGINE_MOE_FUSED_ROTATE` | false | Opt-in fused in-LDS rotate for the selected MoE chain; only worthwhile at very small `out_pack x top_k` totals. |
| `HIPENGINE_SHARED_EXPERT_FUSED_ROTATE` | false | Opt-in fused rotate for the shared expert; off until a fresh W7900 exact/rocprof row proves the net launch reduction. |
| `HIPENGINE_SELECTED_MOE_STAGED_ROTATE` | false | Opt-in staged-keyed selected-MoE rotate; staged-kernel duration and verifier economics regressed. |
| `HIPENGINE_SELECTED_MOE_DOWN_STAGED` | false | Opt-in staged selected-MoE down projection; kept for bisection. |
| `HIPENGINE_PARO_FFN_MEGAKERNEL` | false | Opt-in single-kernel selected FFN for the verify path (tokens > 1); AR decode keeps the unfused path so the AR baseline is untouched. |
| `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` | false | Opt-in fused full-QKV split-key launch; two exact A/B pairs regressed aggregate wall/verify, so it stays a diagnostic. |
| `HIPENGINE_LAGUNA_F16_PREFILL` | `auto` | gfx1151 Laguna F16-weight prefill route: `auto`, `gemv`, `tiled`, or `wmma_comp_swa`; invalid values are errors. |
| `HIPENGINE_LAGUNA_F16_DECODE` | unset | gfx1151 Laguna F16-weight decode route: `gemv` or `onebarrier`; used by the long-context profiling harness for A/B arms. |
| `HIPENGINE_MAPLE_PREFILL_GROUPED_MOE` | true | Maple exact expert-major grouped MoE prefill; `0` restores the original row/route-gather chain. |
| `HIPENGINE_MAPLE_FUSE_MOE` | false | Opt-in fused Maple MoE chain (kept off pending a kernel-efficiency fix; see `docs/REFACTOR.md`). |
| `HIPENGINE_MAPLE_FUSE_QKATTN` | false | Opt-in fused Maple qknorm_rope_kv_write + attention_decode kernel (see `docs/REFACTOR.md`). |
| `HIPENGINE_MAPLE_GRAPH` | false | Opt-in whole-step Maple c1 hipGraph replay; measured only +0.47% bit-exact, so eager stays the default. |
| `HIPENGINE_EVIE_GDN_RECURRENCE` | unset | Evie GDN prefill recurrence owner: unset uses the normalized cluster8 kernel (bit-repeatable, ~9%/page faster than k2); `k2` restores the split fallback for rollback/bisection. |
| `HIPENGINE_DFLASH_DRAFTER_DENSE` | `wmma` | Dense-project DFlash drafter body: `wmma` (default after R3.4 validated exact-AR plus a per-prompt perf gain) or `naive` to revert. |
| `HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM` | `off` | Fused add+RMSNorm drafter variant; `fused`/`on` selects it, with the unfused `dflash_add_bf16` + `dflash_rmsnorm_bf16` chain as the registered fallback. |
| `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD` | `off` | Fused LM-head kernel inside the DFlash verifier window; bit-exact vs the unfused path because the cooperative per-vocab-row dot product order is preserved. |
| `HIPENGINE_MTP_DRAFT_VOCAB_CAP` | internal default | Caps the MTP native drafter's hot vocabulary to at most this many entries (clamped to the model vocab); the packaged default drafter cap applies when unset. |
| `HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION` | true | Packs token/position updates into one proposer upload (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT` | true | Initializes route-0 accumulators inside the fused proposer kernel (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE` | true | Proposer writes draft KV directly (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE` | false | Opt-in indexed (slot-local) proposer KV write path. |
| `HIPENGINE_MTP_PROPOSER_ROUTER_TOPK_FUSED` | true | Fused router top-k inside the proposer kernel (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_ROUTE_BATCHED_EXPERT` | true | Batched expert route in the proposer (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_SHARED_GATE_UP_DUAL` | true | Fused shared gate/up dual in the proposer (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT` | false | Opt-in selected final target hidden plus target-owned W8 scorer contract. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE` | true | Runs the resident MTP draft MoE on device (rollback opt-out). |
| `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER` | true | Q6 top-1 gather kernel in the resident MTP drafter (rollback opt-out). |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN` | false | Opt-in device-side draft chain execution. |
| `HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS` | true | Skips MTP proposer host reads/results that the persistent chain discards. |
| `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY` | true | Keeps verifier-shaped scratch live after MTP verify cycles instead of canonicalizing immediately. |
| `HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER` | false | Runs the proposer update on a side stream while the verifier commit drains; default-off experiment. |

## PARO variables

Prompts shorter than `linear_conv_kernel_dim` use token-serial c1 prefill in
the public generator. Longer prompts use native prefill; neither route requires
an env variable.

### Core PARO selectors

| Variable | Default | Classification | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_PARO_MARLIN_K_REPLACE` | true | Retained default | Uses the retained PARO Marlin-K replacement path during loading. Set false only for bisection. |
| `HIPENGINE_QWEN35_LM_HEAD_THREADS` | `128` | Runtime tuning | Valid values: `128`, `256`, `512`. |
| `HIPENGINE_QWEN35_NATIVE_SAMPLER` | true | Retained default with rollback opt-out | Enables native GPU sampling for supported PARO c=1 and scheduler-owned serial per-slot c>N requests, and GGUF c=1 and compatible packed c>N requests. Unset selects native placement; `0`/`false`/`no`/`off` or an empty value forces host sampling. Supported filters include `top_k=0`, `1<=top_k<=64`, and `top_p`/`min_p` with `top_k=0`. Full-vocab `top_logprobs` with `top_k=0` and bounded `top_logprobs <= top_k <= 64` stay native. Forced queues, dynamic constraints, unsupported filter/processor combinations, and unavailable native sessions use the host fallback. |
| `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` | false | Experimental diagnostic | Enables the guarded Qwen/PARO `step_batch_native` c>N decode path. Leave unset for normal use; retained throughput claims require generated-token equality and currently keep this path ineligible. |
| `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS` | false | Experimental diagnostic | Selects the evidence-backed PARO c>N attention/MoE/projection/sampler repair routes. It does not activate native decode by itself. When native decode is active, unsupported live widths use the exact partition/serial planner below. |
| `HIPENGINE_QWEN35_AVOID_C6_GROUPS` | unset | Test/diagnostic | Used by the retained-batch scheduler tests to force the planner away from c6 subgroups; not a production knob. |
| `HIPENGINE_QWEN35_NATIVE_BATCH_WIDTH_PROFILE` | `benchmarks/results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json` | Correctness/performance gate | Relative regular JSON path under `benchmarks/results/`. The scheduler accepts native subgroup widths only when backend, target arch, model snapshot, quant, KV dtype, generated-token equality, primitive correctness, and decode-position range match. Missing, malformed, mismatched, or out-of-range evidence falls back to serial decode. The default artifact is diagnostic and does not create a retained throughput claim. |
| `HIPENGINE_QWEN35_SERVER_STARTUP_NATIVE_BATCH_WARMUP` | false | Experimental startup diagnostic | When `prepare_request_scratch(..., max_batch_size>1)` runs during server startup, also exercises tiny PARO packed c>N prefill widths 2/4/8 up to `max_batch_size` and records the warmed widths under `/ready` startup diagnostics. Native c>N decode warmup is attempted only when `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1` is also set. |
| `HIPENGINE_QWEN35_SERVER_STARTUP_NATIVE_BATCH_WARMUP_TOKENS` | `64` | Experimental startup diagnostic | Prompt-token count used by the opt-in PARO server native-batch warmup, clamped to the scratch probe's `max_prompt_tokens`. Lower this for fast shape smoke; raise it only when measuring cold packed-prefill setup for a specific server protocol. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE` | `serial_lm_head` | Correctness diagnostic | `serial_lm_head` samples each native c>N row through the c=1 LM-head path; `batched_lm_head` requests batched LM-head buffers but falls back to serial for c>N unless the equality-evidence vars below are set. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK` | false | Correctness gate | Required true before `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=batched_lm_head` is honored for c>N rows. Leave false until generated-token equality vs independent c=1 is green. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT` | unset | Correctness gate | Relative regular `.json` path under `benchmarks/results/` to the generated-token equality artifact supporting `HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK=true`; missing, non-JSON, symlinked, non-regular, failed, wrong-row, self-mismatched `artifact_path`/`source_artifact_path`, skipped, mismatching sequence, or non-empty-mismatch artifacts keep batched LM-head on the serial fallback. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ROWS` | unset | Correctness gate | Row count covered by the generated-token equality artifact; for c>N batched LM-head it must equal both the active row count and the artifact's row count or the sampler stays on the serial fallback. |
| `HIPENGINE_QWEN35_PROJECTION_DISPATCH_ARTIFACT` | unset | Correctness/performance gate | Relative regular JSON path under `benchmarks/results/` with `projection_dispatch_candidates`; missing or invalid artifacts keep runtime metadata on row-GEMV fallback and do not create a retained throughput claim. `scripts/qwen35_batch_retained_bench.py --projection-dispatch-artifact ...` sets this for retained runs and fails closed before the run if the artifact is symlinked/non-regular, cannot provide schema-checked candidates, or any candidate evidence artifact is missing, unsafe, rejected, self-mismatched, out of row bounds, or lacks matching >1 aggregate/per-request row-GEMV ratios. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_BATCH_MOE` | false | Native c-aware selector | Selects one rows=N selected-MoE batch transition per layer, reported as `moe_decode_path=selected_batch` with an explicit `moe_selected_batch_layers` count. Retained-bench spelling: `--batch-decode-moe-path selected_batch`. This is distinct from `selected_c1_per_row_*` fallbacks and is eligible for retained validation only with zero fallback layers and a complete correctness/scaling packet. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE` | false | Deprecated compatibility alias | Legacy alias for `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_BATCH_MOE`; the canonical variable wins when both are set. The CLI spelling `selected_c1` similarly normalizes to `selected_batch`. Remove both aliases after one compatibility window and a retained selected-batch packet. |
| `HIPENGINE_QWEN35_SHARED_EXPERT_PARO_W4_FORCE_GEMV` | false | Diagnostic fallback | For packed PARO W4 shared experts with `tokens<=8`, uses the row-aware GEMV path instead of the batched prefill W4 kernel. The retained-bench selected-batch route sets it for c=2..c=8; retained promotion still requires the complete route-level gate. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR` | false | Diagnostic fallback | Routes linear-attention decode through the per-row c=1 layer path. Hidden-bisect equivalent: `--batch-decode-linear-path per_row`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR_MOE` | false | Diagnostic fallback | Routes the MoE transition after linear attention through the per-row c=1 owner. Hidden-bisect equivalent: `--batch-decode-linear-moe-path per_row_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS` | false | Diagnostic fallback | Replays linear-attention QKV/Z/A/B projections with token-1 kernels per row, then copies planar rows back into batch scratch. Hidden-bisect equivalent: `--batch-decode-linear-projection-path selected_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ` | false | Correctness diagnostic | Replays only linear-attention QKV/Z projections with token-1 kernels per row while leaving A/B on the native batch path. Hidden-bisect equivalent: `--batch-decode-linear-projection-path selected_qkv_z`. Correctness-green for c<=8 in the retained bench, but still non-retained until native projection dispatch is accepted. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ_INPUT` | false | Diagnostic fallback | Same replay family, entered from the pre-input boundary. Hidden-bisect value `selected_qkv_z_input`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKV` | false | Diagnostic fallback | Replays only QKV through token-1 kernels per row. Hidden-bisect value `selected_qkv`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_Z` | false | Diagnostic fallback | Replays only the Z projection through token-1 kernels per row. Hidden-bisect value `selected_z`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB` | false | Diagnostic fallback | Replays only linear-attention A/B projections with token-1 kernels per row while leaving QKV/Z on the selected batch path. Hidden-bisect equivalents: `--batch-decode-linear-projection-path selected_ab` or `batch_gemv_selected_ab`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS` | false | Diagnostic fallback | Uses row-aware GEMV kernels for c>N linear-attention QKV/Z projections while keeping native A/B projection and segmented state. Hidden-bisect equivalent: `--batch-decode-linear-projection-path batch_gemv`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE` | false | Diagnostic fallback | Replays linear-attention conv/GDN/recurrent state updates with token-1 kernels over slot-local state. Hidden-bisect equivalent: `--batch-decode-linear-state-path selected_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT` | `auto` | Diagnostic fallback | Linear-attention output projection override: `auto`, `batch`, `batch_gemv`, or `selected_c1`. `auto` follows selected-c1 state replay; `batch_gemv` bypasses the row>1 AWQ prefill projection kernel while staying non-retained. Hidden-bisect equivalent: `--batch-decode-linear-output-path ...`. |
| `HIPENGINE_QWEN35_BATCH_DECODE_LINEAR_ROW_CHUNK_SIZE` | unset | Diagnostic staging | Row chunk size used when the native c>N linear-attention decode stages rows; set by the hidden-bisect/retained-bench harnesses. |
| `HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE` | true when experimental decode is enabled | Diagnostic selector | Set `0` to force the existing per-row full-attention fallback in hidden-bisect/native-batch probes. Non-retained fallback metadata records `full_attention_decode_path=per_row_*`. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE`, `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_LAYERS` | unset | Diagnostic staging | Row chunk size (and optional comma-separated layer list; empty means every layer) for staging the native c>N full-attention decode across row chunks. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_CONTEXT_ROW_CHUNK_SIZE`, `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_CONTEXT_ROW_CHUNK_LAYERS` | unset | Diagnostic staging | Same staging pair for the full-attention context readback. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT` | false | Diagnostic fallback | Forces only the full-attention input RMSNorm/QKV-prep boundary through token-1 row kernels. Hidden-bisect equivalent: `--batch-decode-attn-input-path per_row`. Non-retained. |
| `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR` | false | Diagnostic fallback | Forces packed prefill linear-attention segments through per-segment c=1-style linear prefill in hidden-bisect probes. Non-retained. |
| `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN` | false | Diagnostic fallback | Forces packed full-attention prefill through per-segment c=1-style full-attention prefill in hidden-bisect probes. Non-retained. |
| `HIPENGINE_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` | `1024` | Decode threshold | Context length where PARO full-attention decode uses split/paged decode; `0` disables. Compatibility alias: `NANOVLLM_PARO_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT`. |
| `HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS` | `2` | Retained default | Minimum rows for compact WMMA MoE prefill. Values clamp to at least 2. |
| `HIPENGINE_LINEAR_AB_PREFILL_ROCBLAS_MIN_TOKENS` | `0` | Rejected/diagnostic | `0` disables the rocBLAS AB prefill route. Leave unset. |
| `HIPENGINE_SHARED_GATE_UP_PREFILL_TOKEN_TILE` | `2` | Retained prefill tiling | Valid values: `0`, `2`, `4`; `0` disables. |
| `HIPENGINE_SHARED_GATE_UP_PREFILL_MIN_TOKENS` | `1024` | Retained prefill tiling | Minimum tokens for shared gate/up token tiling. |
| `HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_TOKEN_TILE` | `2` | Retained prefill tiling | Valid values: `0`, `2`, `4`; `0` disables. |
| `HIPENGINE_SHARED_DOWN_COMBINE_PREFILL_MIN_TOKENS` | `2` | Retained prefill tiling | Minimum tokens for shared down/combine token tiling. |
| `HIPENGINE_PARO_ROTATE_DUAL_PACK8_FUSED` | false | Rejected/diagnostic | Leave unset unless reproducing fusion probes. |
| `HIPENGINE_PARO_FULL_ATTN_KV_PACK8_FUSED` | false | Rejected/diagnostic | Leave unset unless reproducing fusion probes. |
| `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION` | `auto` (unset) | Capacity/perf policy | Streams the direct INT8 prefill attention path that reads the retained INT8 store directly instead of a temporary BF16 oracle; `auto` requires a very long prompt under low memory pressure. |
| `HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS` | `229376` (224K) | Capacity policy | Minimum prompt tokens before `auto` selects the streaming direct INT8 prefill. |
| `HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB` | `26.0` | Capacity policy | Total-memory GiB threshold below which `auto` considers the streaming INT8 prefill. |
| `HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB` | `1024` | Capacity policy | MiB reserved for the write-through BF16 oracle pair when the oracle-bridge prefill path runs. |
| `HIPENGINE_PARO_ROUTER_TOPK_COOP` | false | Rejected/diagnostic | Leave unset unless reproducing router-coop probes. |
| `HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED` | false | Rejected/diagnostic | Leave unset unless reproducing fusion probes. |
| `HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED` | false | Rejected/diagnostic | Leave unset unless reproducing fusion probes. |

PARO prefill workspace-overlap minimization is now a code default, not an env
var: workspaces stay resident through 32K tokens and the memory-saving overlap
minimization path is used only for prompts above 32K when resolved chunk sizes
actually split the prompt.

### PARO MTP verifier route profiles and bisection arms

The PARO MTP (w4_paro_mtp) profile binder and the verifier-numerics harness
resolve the verifier route through these flags. All the `ROW_*` / `EXACT_SUFFIX`
flags are default-off force arms used by `scripts/mtp_paro_verifier_numerics.py`
and the resident runner to bisect the batched full-attention verifier stage by
stage; `1` forces that boundary through the per-row (token-1) owner. Leave them
unset except for bisection.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_MTP_ROUTE_VARIANT` | unset | Route-variant key resolved by the PARO MTP profile binder. |
| `HIPENGINE_MTP_CHAIN_ATTN_MODE` | unset | Chain attention mode resolved by the PARO MTP profile binder. |
| `HIPENGINE_GDN_TLOOP_C1_EXACT` | unset | Selects the exact GDN t-loop c1 route in the verifier-numerics route flags. |
| `HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS` | false | Replays linear-attention output rows through the token-1 owner while preserving order. |
| `HIPENGINE_QWEN35_MOE_C1_FORCE_SMALL_BATCH_SHARED_EXPERT` | unset | Forces the small-batch shared-expert c1 MoE route (verifier-numerics route flag). |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_INPUT` | false | Forces the verifier full-attention input boundary per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_QKV` | false | Forces the QKV projection per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_QKV_TEMP` | false | Forces the QKV temp-buffer staging per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX` | false | Forces the exact-suffix verifier route (implies the per-row KV append). |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_LAYER` | false | Forces the per-layer staging per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_LAYER_BATCH` | false | Forces per-layer staging into batch buffers. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_CONTEXT` | false | Forces the context read per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_CONTEXT_ONLY` | false | Forces only the context read per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_DENSE_CONTEXT_ONLY` | false | Forces only the dense-context read per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_PAGED_CONTEXT_ONLY` | false | Forces only the paged-context read per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_GATE` | false | Forces the gate boundary per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_KV_APPEND` | false | Forces the KV append per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_APPEND_CONTEXT` | false | Forces the append-context order per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_SUFFIX` | false | Forces the suffix staging per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_OUTPUT` | false | Forces the output projection per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_BATCH_GEMV_OUTPUT` | false | Forces the row-aware GEMV output projection. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_POST` | false | Forces the post-attention boundary per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_ROW_MOE` | false | Forces the MoE transition per-row. |
| `HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_HELPER` | false | Forces the serial decode helper for the batched verifier stage. |

### PARO speculative-verify and graph diagnostics

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_VERIFY_GPU_ACCEPT` | true | GPU-resident accept path for the MTP verifier (default on: M12 verifier cycles need it). `0` restores the legacy CPU-oracle read; `validate` cross-checks the GPU payload against the direct path each cycle. |
| `HIPENGINE_VERIFY_CHAIN_LINEAR_TLOOP` | true | Fused linear-attention t-loop inside the verify chain; `0` restores the unfused chain. |
| `HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD` | true | Packed accept-payload readback for the verifier; `0` restores the unpacked readback. |
| `HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS` | `16` | Minimum tokens before the verifier MoE uses the grouped owner; clamps to at least 2. |
| `HIPENGINE_VERIFY_GRAPH_RECAPTURE` | unset | Debug-only (`#107`): drop the cached verifier graph each cycle so replay always executes a freshly captured graph. |
| `HIPENGINE_VERIFY_GRAPH_REVALIDATE` | unset | Debug-only (`#107`): re-run the direct pass before each graph replay and compare per-row top1 plus accept payload to localize replay drift. |
| `HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION` | false | Opt-in: the packed verify-accept path also advances request positions (needs the packed-payload path enabled). |
| `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA` | true | Packs dynamic per-row metadata for the verify-chain batch; `0` restores the static path. |
| `HIPENGINE_VERIFY_SCRATCH_CACHE` | true | Caches verifier scratch buffers across cycles; `0` restores per-cycle allocation. |
| `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP` | true | Stamps scratch generations so workspace lookups can skip stale entries; `0` restores the unstamped lookup. |
| `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED` | true | Aligns the verifier MLP scratch policy with the verifier grouped-MoE threshold; `0` restores the raw rows rule. |
| `HIPENGINE_VERIFY_DENSE_GEMV_WMMA` | false | Opt-in WMMA owner for dense verifier GEMV shapes (`1 < tokens <= 16`, features % 16 == 0). |
| `HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH` | false | Opt-in graph capture for the native-spec target-verify chain (single-request verify_chain only). |
| `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT` | true | Commits target state from the captured native-spec graph; requires the graph flag. |
| `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE` | true | Caches resident tensor views across steps; `0` restores per-step view construction. |
| `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE` | true | Caches weight-tensor lookups in the PARO runner; `0` restores direct lookup. |
| `HIPENGINE_QWEN35_DECODE_BATCHED_DIRECT_GATE` | true | Uses the direct-gate batched split-K decode kernel when `num_splits == 1`; `0` restores the indirect gate owner. |
| `HIPENGINE_DFLASH_VERIFY_SYNC_PHASES` | false | Profiling-only: adds device-synchronized verifier sub-buckets for layer-family and LM-head/top1 attribution at the cost of extra stream synchronizations. |

### PARO native c>N full-attention bisect family

These one-bit probe arms are set by
`scripts/qwen35_batch_hidden_bisect.py` and
`scripts/qwen35_batch_retained_bench.py` from their `--batch-decode-*` CLI
flags; the runtime reads them live. All are non-retained bisection controls for
the native c>N full-attention decode route: set `1` forces that boundary through
the per-row (token-1) owner, `0` keeps the native batch owner. Do not set them
for production serving.

| Variable | Boundary forced per-row |
| --- | --- |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_QKV` | QKV projection |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SCRATCH` | Whole attention scratch staging |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_BATCH_SCRATCH` | Per-row compute into batch scratch |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_MOE` | Attention output into batch MoE scratch |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_POST_MOE` | Attention output staged post-MoE |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_ATTN_BATCH_O_POST_MOE` | Attention `o` projection staged post-MoE |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_BATCH_CONTEXT_O_POST_MOE` | Pre-QKV append + batch context + `o` post-MoE chain |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_CONTEXT_BATCH_GATE_O_POST_MOE` | Same chain with per-layer gate kept batched |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PREQKV_APPEND_CONTEXT_GATE_BATCH_O_POST_MOE` | Same chain with per-layer `o` kept batched |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PERSISTENT_SCRATCH` | Persistent c1 scratch reuse (`persistent_c1` modes) |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SKIP_BATCH_SETUP` | Persistent c1 without batch setup |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_GATE` | Post-attention gate boundary |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT` | Full attention context read |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT_ONLY` | Context read only |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_ONLY` | Dense-context read only |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_LAYERS` | Comma-separated layer list for the dense-context-only arm |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_BATCH_GATE` | Dense-context read with batched gate |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_DENSE_CONTEXT_BATCH_GATE_LAYERS` | Comma-separated layer list for the dense-context batch-gate arm |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_PAGED_CONTEXT_ONLY` | Paged-context read only |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_BATCH_TEMP_FULL_ATTN_CONTEXT` | Batch temp-output context staging |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_BATCH_COMPACT_FULL_ATTN_CONTEXT` | Batch compact-cache context staging |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_KV_APPEND` | KV append write |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_APPEND_CONTEXT` | Interleaved append-context order |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_SUFFIX` | Interleaved suffix order |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_OUTPUT` | Attention output projection |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_FULL_ATTN_OUTPUT` | Row-aware GEMV output projection |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_NATIVE_FULL_ATTN_OUTPUT` | Native output projection |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_NATIVE_ROW_CHUNK_FULL_ATTN_OUTPUT` | Native row-chunked output projection |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_LAYER_COPY` | Per-layer weight/plan copy |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_MOE` | MoE transition |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN` | Post-attention norm/residual boundary |

### PARO batched-LM-head sampler family

Stage selectors and sync-fence bisection arms for the c>N batched LM-head
sampling route, read by the PARO runner and set by the retained-bench and
hidden-bisect harnesses. Leave everything except the documented defaults unset.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_NORM_PATH` | `batch` | Final-norm execution: `batch` or `per_row`. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_CAST_PATH` | `auto` | Final BF16 cast execution: `auto` (follows the norm path), `batch`, or `per_row`. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_MODE` | `batch` | Argmax execution: `batch` or `serial_per_row`. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_ARGMAX_AUDIT` | false | Emits an audit comparing argmax outcomes across the selected path. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_LM_HEAD_AUDIT` | false | Emits an LM-head logits audit for the batched route. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_LM_HEAD_KERNEL_FENCE` | false | Inserts a sync fence around the LM-head kernel for bisection. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_NORM_AUDIT` | false | Emits a final-norm audit for the batched route. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_NORM_KERNEL_FENCE` | false | Inserts a sync fence around the final-norm kernel. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_RMSNORM_KERNEL_FENCE` | false | Inserts a sync fence around the final RMSNorm kernel. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_RMSNORM_TEMP_FENCE` | false | Inserts a temp-buffer fence around the final RMSNorm. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_TEMP_FENCE` | false | Inserts a temp-buffer fence around the final cast. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_TINY_FENCE` | false | Inserts a minimal-scope fence around the final cast. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_FINAL_CAST_ELEMS_FENCE` | `0` | Casts at most this many elements between fences (0 disables). |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_STABILIZE_CAST_ELEMS` | `0` | Stabilizes the cast in element batches of this size (0 disables). |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_SYNC_FENCE` | false | Inserts a full sync fence inside the sampler for bisection. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_SUFFIX_FENCE` | false | Inserts a sync fence at the sampler suffix boundary. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_SUFFIX_KERNEL_FENCE` | false | Inserts a kernel-scope fence at the sampler suffix boundary. |

## Qwen4 experimental variables

These select kernel routes inside the Qwen4Exp GGUF runner. The registered
production profile binder sets most of them to their retained values when the
production profile resolves; their direct use is rollback (`0` restores the
registered parent/fallback route) or bisection of a single route. They fail
closed when the named variant is not registered. Leave them unset in normal
use.

### Qwen4Exp decode and state

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_QWEN4_EXP_BATCHED_POSITION` | `1` | One shared 192B H2D position upload instead of 24 per-layer 8B blocking `set_position` copies per decode step (bit-exact, ~1.3% median TG gain). `0` restores per-layer copies. |
| `HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION` | `1` | Uses the batched QSA selection owner for multi-row decode; `0` restores the per-row selection. |
| `HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL` | `0` (production binder sets `1`) | Master gate for the production grouped MoE prefill session (rows >= 16, supported quant shapes). |
| `HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1` | `0` (production-only binding) | Q5_1 down M1 expert-grid64 variant binding. |
| `HIPENGINE_QWEN4_EXP_Q5_1_WMMA` | `0` (production binder) | Q5_1 grouped WMMA down-projection admission under the production grouped MoE route. |
| `HIPENGINE_QWEN4_EXP_Q5_1_OUT8` | `1` | Uses the out8 Q5_1 grouped compact prefill variant. |
| `HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN` | `0` | Opt-in Q5_1-style grouped WMMA down for Q8_0 experts. |
| `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP` | `0` | Opt-in warp256 Q5_1 selected weighted-sum decode (free-running divergence routed to its packet per contract). |
| `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN` | `0` | Opt-in down-leg-only warp256 weighted sum; the incumbent bit-exact weighted-sum stays the production default. |
| `HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT` | `0` (production-only binding) | Weight-exact iu8-WMMA Q5_1 down with risk+repair; bit-identical to the pair2 row-publish parent in practice at multiplier 16. |
| `HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID` | `64` | Expert-grid mode for the exact grouped routes: `64` (admitted), plus `q4`/`q5` spellings at their call sites. |
| `HIPENGINE_QWEN4_EXP_EXACT_CONV_PREFILL` | `1` | Uses the exact bulk conv prefill owner; `0` restores the generic chain. |
| `HIPENGINE_QWEN4_EXP_LAYER_GRAPHS` | unset | Override for per-layer graph capture admission (backend capability gates the default). |
| `HIPENGINE_QWEN4_EXP_RAW_VARIANT` | `coltile8` | Raw-Q rowbatch prefill variant selector. |
| `HIPENGINE_QWEN4_EXP_RAW_ROWBATCH` | `32` | Row-batch size for raw Q4 prompt-prefill row owners; `0`/`false` disables, `1` selects 32. |
| `HIPENGINE_QWEN4_EXP_QSA_WAVE32` | `1` | Uses the wave32 QSA paged sparse-attention decode kernel for head_dim 128. |
| `HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE` | `0` | Opt-in ordered sparse-attention decode route for head_dim 256. |
| `HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2` | `0` | Second-generation ordered decode route (requires the v1 flag and `0 < selected_count <= 4096`). |
| `HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR` | `0` | Head-pairing for the strict H256 page256 wave prefill: `1`/`quad` pairs query heads (24q/2kv shapes). |
| `HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL` | `page256` under production | Selects the `strict_h256_page256_wave_rows_spans` prefill variant; otherwise `strict_h256_wave_rows_spans`. |
| `HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL` | production profile | Uses the flash sparse-attention prefill owner for dense rows with head_dim 256, gated by the `HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS` layer list. |
| `HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL` | production profile | Uses the sigmoid register GDN recurrence prefill kernel for rows>=2 and the (16, 48, 128) head geometry. |
| `HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM` | production profile | Remaps the register GDN prefill to the wave-norm sibling variant when registered. |
| `HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL` | `1` | Uses the tile16 GDN recurrence prefill variant; `0` opts out to the columnwarp parent (the serial strict route stays registered below both). |
| `HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL` | production profile | Uses the peer-wave GDN prefill variant for head_dim 128, gated by the `HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS` layer list. |
| `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL` | production profile | Uses the columnwarp GDN prefill variant, gated by the `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS` layer list. |
| `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS` | unset | Comma-separated layer list admitting the columnwarp GDN decode variant; empty under production. |
| `HIPENGINE_QWEN4_EXP_MOE_GRAPH` | backend capability | Enables the captured MoE decode graph on backends whose capability admits it; the strict profile forces `0`. |
| `HIPENGINE_QWEN4_EXP_PLE_WARM` | `0` | Warms the PLE mmap table page cache at load for long-lived serving; short-lived runners never amortize it. |
| `HIPENGINE_QWEN4_EXP_LOAD_DROP_BEHIND` | `0` | Drop-behind for the PLE mmap load path (release page cache after materialization). |

### Qwen4Exp MoE prefill routes

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL` | unset (production sets `0`) | Admits the grouped MoE prefill session for rows>=16 with supported quant shapes. |
| `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN` | `1` | Exact grouped down-projection for Q5_1 expert weights at rows>=2. |
| `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4` | `1` | Exact grouped Q4_K gate/up route at rows>=2. |
| `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL` | `1` | Extends the exact grouped Q4 route to all shapes (not only where the grouped-down route already applies). |
| `HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN` | `0` | ForkB grouped down-projection kernel serving sorted-lane rows. Production sets `1`. |
| `HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL` | production profile | Grouped row4 GEMV prefill variant. |
| `HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL` | production profile | Selected-dual grouped rowbatch8/out4 expertgrid64 bundle prefill for Q4_K. |
| `HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL` | production profile | Selected-dual grouped pair2 prefill for Q4_K (rows>=64, in_features<=4096). |
| `HIPENGINE_QWEN4_EXP_Q4_OUT4` | `1` | Uses the out4 (four-output) selected-dual grouped rowbatch8 Q4 variants. |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL` | production profile | Integer-uint8 risk+repair selected Q4 gate/up prefill (rows>=2, hidden%256==0, ffn%128==0). |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS` | production layer list | Comma-separated layer list admitting the Q4 iu8 prefill route. |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT` | production profile | Exact iu8-risk Q4 gate/up route (the layer-2 Q4 route). |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES` | `3` | Plane count for the iu8-risk Q4 gate/up variant: `3` (default) or `2` (the p2 variant). |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` | `4.0` | Risk multiplier margin for the iu8 risk+repair Q4 route; the default keeps a >=4x margin at the measured 1.50-1.82x operation-complete speedup. |
| `HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL` | `0` (production keeps off) | Builds the Q4_K ds4-MMQ selected prefill library and workspace. |
| `HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS` | unset | Comma-separated layer list admitting the Q4_K MMQ route. |
| `HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL` | `0` (production keeps off) | Builds the Q5_1 ds4-MMQ selected prefill library and workspace. |
| `HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS` | unset | Comma-separated layer list admitting the Q5_1 MMQ route. |
| `HIPENGINE_QWEN4_EXP_Q5_1_WAVE64` | unset | Uses the wave64 Q5_1 selected GEMV decode variant. |
| `HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT` | `0` | Default-off exact iu8-risk+repair selected Q5_K gate/up route (the layer-2 Q5_K layer). |
| `HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT` | `16.0` | Risk multiplier for the Q5_K iu8 route; 16.0 keeps the route bit-identical to the strict parent in practice. |
| `HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL` | production profile | Selected grouped pair2 prefill for Q5_1 (rows>=64). |
| `HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL` | production profile | Fold128 sibling of the Q5_1 pair2 prefill (rows>=64). |
| `HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL` | production profile | Fold-pair sibling of the Q5_1 fold128 prefill (rows>=512). |
| `HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE` | production profile | Register-cache sibling of the Q5_1 fold-pair prefill (rows>=512, in_features==640). |
| `HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH` | production profile | Row-publish sibling of the Q5_1 register-cache prefill. |
| `HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT` | `16.0` | Risk multiplier for the Q5_1 iu8 route. |
| `HIPENGINE_QWEN4_EXP_Q4_TILE_M` / `HIPENGINE_QWEN4_EXP_Q4_TILE_N` | `16` / `16` (production pins 64/16 at rows>=512) | WMMA tile override for the Q4_K iu8 prefill build. |
| `HIPENGINE_QWEN4_EXP_Q8_0_GROUPED` | unset | Grouped compact prefill for Q8_0 expert down weights. |
| `HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA` | unset | Grouped WMMA prefill owner for Q8_0 experts (reads `expert_start` on device via a fixed worker grid). |
| `HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL` | production profile | Row4 GEMV down-projection variant (rows>=512). |
| `HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL` | production profile | Bundle GEMV down-projection variant (rows>=512). |
| `HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN` | production profile | Admits the mapped-down key when the mmap table is ready (rows>=512). |
| `HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER` | production profile | Register GEMV down-projection variant (rows>=512, in_features==640). |
| `HIPENGINE_QWEN4_EXP_FUSED_COMBINE` | unset | Opt-in fused PF-4 lever-2 combine; default stays unfused after the whole-model A/B measured a loss. |
| `HIPENGINE_QWEN4_EXP_Q4_DP4A64` | production profile | DP4A64 selected dual q8_1 dp4a GEMV decode for matching gate/up quant at rows==1. |
| `HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS` | unset | Comma-separated layer list admitting the DP4A64 decode route. |
| `HIPENGINE_QWEN4_EXP_GR_IU8` | `0` | iu8 WMM prefill remap for the raw-Q rows>256 route. |
| `HIPENGINE_QWEN4_EXP_GR_IU8_DOWN` | `0` | iu8 WMM prefill remap for the down projection of the rows>256 route. |
| `HIPENGINE_QWEN4_EXP_Q8_IU8_WMM` | `0` | iu8 WMMA dense prefill dispatch for Q8_0 weights (rows>256, features%32==0). |
| `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` | unset | Comma-separated layer list admitting the Q8 WMMA dense prefill. |
| `HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE` / `HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE` | production profile | Wave-scaled variants of the Q8 coltile prefill and the raw-Q route. |
| `HIPENGINE_QWEN4_EXP_MMQ_TOKEN64` | production profile | Remaps the raw-vec4 MMQ prefill to the token64 variant (rows>=512, 2560x12288 shape). |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL` | production profile | Admits the MMQ prefill session for Q8_0 dense projections. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES` | unset (`2` or `3`) | Plane count override for the raw Q8 MMQ policy. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK` | production profile | Uses prepacked weight sidecars in the Q8 MMQ prefill session. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4` | production profile | Remaps the prepacked MMQ variant to its vec4 sibling. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR` | production profile | Remaps the guarded MMQ prefill to the raw-vec4 variant (rows>=64). |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE` | unset | Production keeps off; admits the MMQ route for the attention gate projection. |

## Build-ablation variables

These change JIT compiler flags and therefore change cache keys. They are for
kernel R&D only, not normal use.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_PREFILL_MCUMODE` | false | Adds `-mcumode` to remaining `prefill` profile builds that do not already request it. Prior ablations rejected making this broad default. |
| `HIPENGINE_DISABLE_UNROLL600` | false | Strips `-mllvm -amdgpu-unroll-threshold-local=600` from profile flags for ablation. Leave unset for retained builds. |

## Diagnostics and debugging

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_ENGINE_SERVICE_TRACEBACK` | unset | When the engine-service driver thread fails, prints the full Python traceback to stderr before recording the service unhealthy. |
| `HIPENGINE_DEBUG_SAMPLED_ROUTE` | unset | Prints one `[prefill-debug]`/route-debug line per request describing which sampler/speculation route resolved and why (engine loop and MTP2 adapter). |
| `HIPENGINE_DEBUG_SAMPLED_QUALIFY` | unset | Prints the MTP2 qualification inputs (backend, arch, quant, artifact size) as the adapter decides whether a physical cell qualifies. |
| `HIPENGINE_MTP2_TRACE_DECLINE` | unset | Prints one `[mtp2-resolve]`/`[mtp2-prefix-checkpoint]` line per declined MTP2 resolution, so an operator can tell the distinct decline reasons apart. |
| `HIPENGINE_PARO_FFN_MEGAKERNEL_DEBUG` | unset | Prints a one-time line when the PARO FFN megakernel first fires (counts activations). |
| `HIPENGINE_VK_OWNER_TRACE` | unset | Inside the generated Vulkan-owner build probe, prints `HE_NODES=` member lists during graph capture. |
| `HIPENGINE_LC_PROBE_FORCE_SCALAR` | unset | In the GGUF linear-context staged-chain probe, keeps the per-row views but forces the scalar row-wise attention owner, separating a row-view defect from a staged-chain defect. |
| `HIPENGINE_LC_PROBE_OUT` | unset | Also writes the probe report to this path; the report is always printed to stdout as one `PATH_PROBE <json>` line. |
| `HIPENGINE_MTP_BENCH_CACHE_SESSION` | unset | Benchmark-harness only: load the model once and reuse the resident-session cache across every (prompt, budget) arm instead of reloading per subprocess. |
| `HIPENGINE_MTP2_OUTPUT_SPANS` | unset | Diagnostic: records committed output spans (mode, reason, position, tokens) per speculative cycle; off by default because the span list grows with cycle count and is audit evidence, not production telemetry. |
| `HIPENGINE_VIBEVOICE_REQUIRE_CACHED` | unset | Set by the VibeVoice rocprof driver to require cached builds during profiling. |

## Test gates and fixtures

These are read by the test suite (and the live tests) to locate local
artifacts or explicitly opt into GPU-gated suites. They are not product knobs.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_RUN_CUDA_SM120A` | unset | Master opt-in for the CUDA sm120a (B300-class) GPU test suites. Must be `1`, paired with `HIPENGINE_CUDA_ARCH=sm_120a`. |
| `HIPENGINE_RUN_CUDA_CUTLASS_ATTENTION_GATE` | unset | Additional opt-in for the CUTLASS attention gate test; also requires `HIPENGINE_CUTLASS_DIR` and `libcudart.so.13`. |
| `HIPENGINE_RUN_CUDA_LT_FIXTURE_GATE` | unset | Opt-in for the LT fixture gate of the CUDA sm120a batch-encoder test. |
| `HIPENGINE_RUN_CUDA_CUDNN_ROUTE_GATE` | unset | Opt-in for the cuDNN route gate of the CUDA sm120a batch-encoder test. |
| `HIPENGINE_RUN_CUDA_MAPLE` | unset | Opt-in for the CUDA Maple backend tests. |
| `HIPENGINE_RUN_MOONSHINE_MODEL_GATE` | unset | Opt-in for the real-model Moonshine runtime gate (requires `HIPENGINE_HIP_ARCH=gfx1151` and the local snapshot). |
| `HIPENGINE_TEST_PARO_MODEL` | unset | PARO model directory override for the live PARO tokenizer/EOS tests; falls back to the standard `/models/hipengine/...` candidates. |
| `HIPENGINE_TEST_REQUIRE_CACHED_BUILD` | unset | When truthy, GPU kernel tests require cached JIT builds (the test-side analogue of `HIPENGINE_REQUIRE_CACHED_BUILD`). |
| `HIPENGINE_INT8_MTP_MODEL` | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` | Dense GGUF model path for the live INT8 MTP test. |
| `HIPENGINE_IQ4_XS_LAYOUT_GGUF` | `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | GGUF artifact for the live IQ4_XS T16 layout test. |
| `HIPENGINE_UD_ROLE_MODEL` | `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | Published UD K_M model path for the live UD Q5/Q6 role tests. |
| `HIPENGINE_DMS_DEVICE_PAYLOADS` | unset | DMS device-payload test fixture knob; asserted unset for the host-payload baseline. |
| `HIPENGINE_DMS_DEVICE_TRIPWIRE` | unset | Set by the DMS device-pack parity test to tripwire device-payload mode for bit-exact comparison. |
| `HIPENGINE_MOONSHINE_CHECKPOINT` | local default | Moonshine model checkpoint override for the Moonshine GPU tests. |
| `HIPENGINE_MOONSHINE_SNAPSHOT` | local default | Moonshine HuggingFace snapshot directory override for the Moonshine GPU tests. |
| `HIPENGINE_MOONSHINE_FIXTURE_DIR` / `HIPENGINE_MOONSHINE_FIXTURES_SIX` / `HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR` | local defaults | Moonshine audio fixture directory overrides for the single- and six-fixture suites. |
| `VIBEVOICE_STANDALONE_GGUF` / `VIBEVOICE_STANDALONE_REPORT` | local defaults | VibeVoice standalone-encoder GGUF and report fixture paths for the VibeVoice tests. |
| `VIBEVOICE_REFERENCE_HF` / `VIBEVOICE_REFERENCE_AUDIO` | local defaults | VibeVoice HuggingFace reference snapshot and reference audio fixture paths for the encoder comparison tests. |

## Benchmark and development harness variables

These are read only by scripts under `scripts/` and `benchmarks/`; they never
affect `hipengine.LLM.generate()` or `hipengine serve`. Defaults shown are the
harness's own fallbacks. They are grouped by harness; see the harness catalog
in [`benchmarks/HARNESSES.md`](../benchmarks/HARNESSES.md) for the protocols.

### MTP/GGUF bench harnesses (`gguf_mtp_bench.py`, `gguf_mtp_category_bench.py`, `gguf_mtp_draft_rocprof.py`, `gguf_mtp_verifier_rocprof.py`)

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GGUF_RESIDENT_MTP_DRAFT` | `0` | Enables the resident MTP draft runner in the bench (truthy parse). |
| `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A` | unset | Q6 top-1 dp4a route for the resident MTP draft runner; set before the runner is constructed. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` | unset | Dense Q8 dp4a route for the resident MTP draft runner. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES` | unset | Stage-selection string for the dense Q8 dp4a draft route. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL` | `1` | Shared-dual Q8 draft route; `0`/`off` disables. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL` | unset | Row-parallel draft router variant. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` | unset | Fused selected SiLU+down draft variant. |
| `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` | `64` | Workgroup threads for the selected dp4a microbenches; recorded into their artifacts. |
| `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS` | falls back to `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` | Q5-specific thread override for the Q5_K selected-down dp4a microbench. |

### DFlash chain bench (`dflash_chain_e2e_bench.py`)

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_DFLASH_CONF_ORACLE_OUT` | unset | Appends the drafter confidence-oracle JSON (per-cycle p1 trace plus summary) to this file. |
| `HIPENGINE_DFLASH_WHOLE_CYCLE_GATE` | unset | Backward-compat activation of the whole-cycle confidence gate when the CLI flag is absent; a float in `[0, 1]`. |
| `HIPFIRE_DFLASH_LOOP_BREAK` | unset | Diagnostic early-break for the hipFire DFlash loop driver. |

### IQ/Q3K gate and sweep scripts

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_LOCAL32_GATE_NATURAL` | unset | `1` uses the model's own greedy extension as the gate probe prompts instead of uniform-random token ids, which measure a prompt-sensitivity floor that masks route differences. |
| `HIPENGINE_Q3K_GATE_NATURAL` | unset | Same natural-prompt mode for the Q3_K gate. |
| `HIPENGINE_Q3K_GATE_STRICT_ARM` | unset | `1` adds the all-strict reference arm to the Q3_K gate, measuring the incumbent's own noise level. |
| `HIPENGINE_TILE_SWEEP` | `16x64,32x64,16x128` | Comma-separated `MxN` tile list for the IQ W4A16 tile sweep. |
| `HIPENGINE_IQ_WMMA_TILE_M` / `HIPENGINE_IQ_WMMA_TILE_N` | unset | Current tile pair for the sweep iteration (set by the sweep script itself). |

### Cross-harness labels and inputs

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_HW_LABEL` | derived from `HIPENGINE_HIP_ARCH` | Hardware label written into bench artifacts. |
| `HIPENGINE_LLAMACPP_HEALTH_TIMEOUT` | `600` | Seconds to wait for a llama.cpp server's health endpoint in the engine-matrix harness. |
| `HIPENGINE_AR_D2_COST_ARTIFACT` / `HIPENGINE_GGUF_AR_D2_COST_ARTIFACT` | unset | Measured cost artifact feeding D2 composition in the arbitrary-c lifecycle harness. |
| `HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS` | unset | Declares the hidden-plane alias mode for the INT8 resumable-prefill GPU proof before bulk-prefill workspace construction. |
| `HIPENGINE_LAGUNA_PREFILL_CHUNK_SIZE` | unset | Prefill chunk size echoed by the Laguna gate/bench harnesses into their provenance environment. |
| `HIPENGINE_MTP2_MAX_CONTEXT_TOKEN` | unset | Singular spelling that `scripts/bench_env_preflight.py` watches as a common lookalike of `HIPENGINE_MTP2_MAX_CONTEXT_TOKENS`; setting it changes nothing. |
| `HIPENGINE_QUANT_QUALITY_MIOPEN_DEPTHWISE` | `0` | In the torch-side quant-quality teacher, `1` keeps the MIOpen depthwise route on gfx1151 instead of the fallback. |
| `CROSSOVER_MODEL` / `CROSSOVER_QUANT` | unset | Model/quant pair for the crossover sweep scripts. |
| `SWEEP_MODEL` / `HEADROOM_MODEL` / `HEADROOM_QUANT` / `GGUF_Q4KM_MODEL` / `PARO_MODEL` / `MODEL` / `MODELS_DIR` | per-script defaults | Model-path inputs for the sweep, headroom, and matrix harnesses. |
| `EVIE_MATCHED_DIR` | unset | Matched-output directory for the Evie comparison harness. |
| `BENCH_EXTRA_JSON` | unset | Extra JSON merged into a bench artifact. |
| `BENCH_INCLUDE_HIPENGINE` | unset | Includes the hipEngine arm in the shared comparison bench. |
| `AUDIT_NO_VETO` | unset | Runs the audit script without its veto gate. |
| `PROBE_LIB` / `PROBE_ROWS` / `PROBE_BLOCKS` / `HE_PROBE_TARGET` | unset | Shape/target knobs for the kernel probe scripts. |
| `HE_DUAL_WMMA_SILU_MIN_ROWS` / `HE_UNEQUAL_DUAL_WMMA_MIN_ROWS` | unset | Minimum-row thresholds for the dual/unequal WMMA probe scripts. |
| `AQ2_SANDBOX_SECRET` | unset | Sandbox secret for the AQ2 sandbox scripts. |
| `WORKLOG_WORKER` | unset | Worker identity used by `scripts/worklog.py` tooling contexts. |
| `GGUF_CORRECTNESS_ARTIFACT` / `PARO_CORRECTNESS_ARTIFACT` / `CHAIN_JSON` | unset | Correctness artifact paths for the parity/report harnesses. |
| `ALLOW_UNTRACKED` | unset | Lets the release-audit scripts tolerate untracked files. |
| `CONCURRENCY_REQUIRE_CACHED` | unset | Requires cached builds for the concurrency re-baseline harness. |
| `SKIP_HIPENGINE_PREBUILD` | unset | `1` skips the hipEngine prebuild step in the TheRock wrapper scripts. |
| `FIXTURE` / `DTYPE` / `ARMS` / `REQS` / `HERE` / `RECREATE` / `SITE` / `PORT` / `HOST` / `PYTHON` / `PYTHON_BIN` / `BASE_PYTHON` / `VENV` | per-script defaults | Per-script parameter locals (fixture path, dtype, arm list, requirement set, output root, recreate flag, site/port/host, and interpreter paths) read by the shell harnesses; each script assigns its own default. |

### External-engine comparison harnesses (vLLM, llama.cpp, atlas)

These belong to the comparison harnesses, not to hipEngine. `VLLM_*` variables
configure the vLLM server/docker arm (`VLLM_MODEL`, `VLLM_SERVED_MODEL`,
`VLLM_SERVED_MODEL_NAME`, `VLLM_PORT`, `VLLM_URL`, `VLLM_PY`, `VLLM_BIN`,
`VLLM_GPU`, `VLLM_DTYPE`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_MODEL_LEN`,
`VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_MAX_NUM_SEQS`, `VLLM_TENSOR_PARALLEL_SIZE`,
`VLLM_KV_CACHE_DTYPE`, `VLLM_ENABLE_EXPERT_PARALLEL`, `VLLM_ENABLE_TOOL_CALLING`,
`VLLM_SPECULATIVE_CONFIG`, `VLLM_READY_TIMEOUT`, `VLLM_DOCKER_TTY`,
`VLLM_ROCM_IMAGE`, `VLLM_ROCM_AMD_IMAGE`, `VLLM_ROCM_PINNED_IMAGE`).
`LLAMACPP_HIP_BENCH`, `LLAMACPP_VULKAN_BENCH`, `LLAMACPP_VULKAN_REPO`,
`LLAMACPP_VULKAN_SERVER`, and `LLAMACPP_Q4KM_MODEL` configure the llama.cpp
comparison arm; `ATLAS_ROOT`, `ATLAS_MODEL`, `ATLAS_NAME`, `ATLAS_PORT`,
`ATLAS_ROCM_HOME`, and the run-plan knobs (`CONCURRENCY`, `TURNS`, `REPEATS`,
`NUM_DRAFTS`, `CANDIDATE_BUDGET`, `MAX_BATCH`, `MAX_CONTEXT`, `MAX_PREFILL_TOKENS`,
`MAX_SEQ_LEN`, `PROMPT_FILE`, `PROMPT_CATEGORY`, `PROMPT_LIMIT`, `OUTPUT_LEN`,
`ENGINES`, `SPECULATIVE_CONFIG`, `DEFAULT_SPECULATIVE_CONFIG`, `DEFAULT_LLAMACPP_Q4KM_MODEL`,
`SERVED_NAME`, `SERVED_MODEL_NAME`, `SNAPSHOT`, `SNAPSHOT_DIR`, `LOG_DIR`,
`LOGDIR`, `OUTDIR`, `ENVWRAP`, `HIP_NAME`, `HIP_PORT`, `MODEL_NAME`, `DOCKER_BIN`,
`IMAGE`, `PINNED_IMAGE`, `AMD_GFX110X_IMAGE`, `CXX`) configure the atlas-agent
matrix and its Docker/vLLM baselines. The TheRock wrapper scripts read
`THEROCK_PY`, `THEROCK_ROOT`, `THEROCK_ENV`, `THEROCK_SITE`, `THEROCK_CORE_LIB`,
`THEROCK_GFX_LIB`, and `HIPCC_VERSION_FILE`; the parity/economics harnesses read
`GGUF_Q4KM_MODEL`, `LLAMACPP_Q4KM_MODEL`, `MAX_CONTEXT`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `KV_CACHE_DTYPE`, `MAX_NUM_BATCHED_TOKENS`,
`MAX_NUM_SEQS`, `HF_CACHE`, `PAROQUANT_HIP_ARCH`, `ROCM_HOME`, `RUN_TAG`,
`DATE_PREFIX`, `TIMEOUT_SHORT`, and `TIMEOUT_LONG`.

Many shell harnesses also assign their own script-internal locals and read
them back with `${...}` (for example `SCRIPT_DIR`, `REPO_ROOT`, `REPO`,
`LOGDIR`,
`OUTDIR`, `TAG`, `L`, and the standard shell variables `PATH`, `HOME`, `USER`,
`SHELL`, `TERM`, `LOGNAME`, `BASH_SOURCE`, `PIPESTATUS`, `LD_LIBRARY_PATH`). An
exported value with one of these names is normally overwritten by the script's
own assignment before use; they are listed here only so every `${...}` read in
the tree has a documented home.

## Third-party environment variables

These are honored from the surrounding process environment; hipEngine or its
harnesses read them but do not own them.

| Variable | Read by | Values / notes |
| --- | --- | --- |
| `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` | HIP/ROCr runtime | Device visibility filtering (see the multi-GPU profile above). |
| `GPU_MAX_HW_QUEUES`, `HSA_SCRATCH_SINGLE_LIMIT` | HIP/ROCr runtime | Process setup; see the core runtime table. |
| `HIP_DEVICE_LIB_PATH`, `HIP_PATH`, `ROCM_PATH`, `HIP_TARGET_ARCHS`, `HIP_OFFLOAD_ARCH`, `HSA_OVERRIDE_GFX_VERSION` | hipcc/ROCm toolchain and harness wrappers | Standard ROCm toolchain inputs echoed by the TheRock wrappers and cache-key probes. |
| `XDG_CACHE_HOME` | GGUF sidecar cache | Overrides the default `~/.cache` root for `HIPENGINE_GGUF_SIDECAR_CACHE`. |
| `XDG_STATE_HOME` | server state | Overrides the default `~/.local/state` root for resident server state. |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `HF_TOKEN` | HuggingFace loaders and comparison harnesses | Standard HuggingFace cache/token inputs for hub downloads. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | CPU math libraries | Thread caps used by the CPU-reference and analysis harnesses. |
| `PYTORCH_ROCM_ARCH`, `PYTORCH_ENABLE_MPS_FALLBACK` | torch bridge / quant-quality scripts | Standard torch inputs used by the optional torch-side comparison scripts. |
| `CUDA_VISIBLE_DEVICES` | CUDA comparison harnesses | Standard CUDA visibility filter. |
| `VK_ICD_FILENAMES`, `RADV_PERFTEST` | Vulkan loaders (llama.cpp Vulkan arm) | Standard Vulkan ICD/performance inputs for the comparison harnesses. |
| `CC`, `CXX` | build harnesses | Compiler overrides consumed by benchmark build wrappers. |
| `PYTHONPATH`, `PYTHONUNBUFFERED`, `PYTORCH_...` (above) | harness wrappers | Standard Python process inputs. |
| `NO_COLOR`, `FORCE_COLOR` | server log styling | Honored by `HIPENGINE_LOG_COLOR=auto`. |
| `AMDGPU_CARD_NAME`, `AMDGPU_CARD_NAME_W7900`, `HIP_VISIBLE_DEVICES_W7900` | TheRock/lab wrapper scripts | Card-name and device-pin locals used by the W7900 wrapper scripts. |

## Names that are not environment variables

These names appear in the tree but must not be confused with env knobs:

- `-D` preprocessor macros inside kernel and micro-benchmark builds
  (`HIPENGINE_LOCAL_SIZE_X`, `HIPENGINE_ROW_TILE`,
  `HIPENGINE_FIXED_WORKGROUP_SIZE`, `HIPENGINE_BLOCK_SIZE`,
  `HIPENGINE_VOPD_MODE`, `HIPENGINE_VOPD_ACCUMS`, `HIPENGINE_MEM_MODE`,
  `HIPENGINE_MEM_PARAM`, `HIPENGINE_MEM_FIXED_BLOCK`, `HIPENGINE_DOT_MODE`,
  `HIPENGINE_DOT_GROUPS`, `HIPENGINE_DOT_FIXED_BLOCK`, `HIPENGINE_REDUCTION_VARIANT`,
  `HIPENGINE_ARGMAX_WG`, `HIPENGINE_ARGMAX_TOPK`, `HIPENGINE_ACCUM_COUNT`,
  `HIPENGINE_MICRO_TIMING_HEADER_HASH`, `HIPENGINE_DMS_ENABLE_WAVE_GROUP6`,
  `HIPENGINE_GDN_GROUPED_HEADS`, `HIPENGINE_IQ_GEMV_SOURCE_TAG`,
  `HIPENGINE_IQ_WMMA_TABLE_HASH`, `HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS`,
  `HIPENGINE_NATIVE_SPEC_CYCLE_ABI_VERSION`) are compile definitions, not
  process env vars.
- String-prefix fragments used to construct names dynamically:
  `HIPENGINE_GGUF_`, `HIPENGINE_GGUF_VERIFY_`, `HIPENGINE_QWEN35_BATCH_`,
  `HIPENGINE_QWEN35_BATCH_DECODE_`, `HIPENGINE_QWEN35_BATCH_SAMPLE_`,
  `HIPENGINE_NATIVE_SPEC_CYCLE_ABI_HEADER_SHA256_`.
- Module-level Python constant *identifiers* whose values are the real variable
  names, not env names themselves: `HIPENGINE_GGUF_DECODE_REPACK_ENV`,
  `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR_ENV`, `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR_ENV`,
  `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL_ENV`, `HIPENGINE_GGUF_SELECTED_X8_REPACK_ENV`,
  `HIPENGINE_GGUF_SELECTED_DOWN_RAW_ENV`, `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW_ENV`,
  `HIPENGINE_GGUF_SELECTED_GATE_UP_X8_ENV`, `HIPENGINE_GGUF_C8_Q5_RAW_MMQ_ENV`,
  `HIPENGINE_C8_Q5_PLANAR_DP4A_ENV`, `HIPENGINE_UD_REPACK_ELIGIBILITY_ENV`.
- Backend-package capability *keys* (dict fields consumed by the adapter, not
  env names): `GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS`,
  `GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES`.
- Test/fixture identifiers and printf formats, not env reads:
  `HIPENGINE_D32_TOKEN_FIXTURE` (a `Path` constant), `HIPENGINE_CONCURRENCY_JSON`
  (an env-file output format written by a harness), and the `DEFAULT_HIPENGINE_*`
  identifier fragments `HIPENGINE_ARRAYS`, `HIPENGINE_ARTIFACT`,
  `HIPENGINE_TOKENS`, `HIPENGINE_RAW_ROOT`.
- Placeholder names used in doc/test examples and generic metavars
  (`HIPENGINE_FOO`, `HIPENGINE_ZZZ`, `HIPENGINE_AAA`, `HIPENGINE_EXAMPLE`,
  `HIPENGINE_SOMETHING_ELSE`, `HIPENGINE_STALE_FLAG`, `HIPENGINE_ONE`,
  `HIPENGINE_TWO`, `HIPENGINE_PREBUILD`, `HIPENGINE_DUP`, `HIPENGINE_KEY`,
  `HIPENGINE_ROUTE`).
- Removed or never-shipped flags that tests assert are absent from the runtime
  source: `HIPENGINE_GGUF_AR_STREAM_PREFILL`,
  `HIPENGINE_GGUF_MTP_SERVER_ROLLING_SLOTS`,
  `HIPENGINE_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH`,
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK`,
  `HIPENGINE_MAPLE_PREFILL_GQA4`, `HIPENGINE_MAPLE_ROUTER_SINGLE_DISPATCH`,
  `HIPENGINE_MAPLE_AFFINE4_WAVE32_EXACT`,
  `HIPENGINE_MAPLE_BATCH_AFFINE4_ROWREUSE_EXACT`,
  `HIPENGINE_PM4_STATEFUL_REGISTERS`,
  `HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES`,
  `HIPENGINE_GPU_MAX_HW_QUEUES_POLICY` (scrubbed from provenance keys),
  `HIPENGINE_HIP_REQUIRE_CACHED_BUILD` (a lookalike that does nothing; the real
  knob is `HIPENGINE_REQUIRE_CACHED_BUILD`),
  `HIPENGINE_PROCESS_ENV_REPORT_PATH` (scrubbed from child environments), and
  the removed AOTriton knobs `HIPENGINE_AOTRITON_SOURCE_ROOT` /
  `HIPENGINE_AOTRITON_RUNTIME_ROOT`.
