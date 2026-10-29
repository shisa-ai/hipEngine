---
status: current
owns: Complete environment-variable reference and the recommended profiles for normal use, ROCm setup, and profiling.
---
# Environment variables

Last updated: 2026-10-21

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
(Python 3.12, ROCm 10.1.0) in the `mambaforge` prefix, not the base interpreter;
the base interpreter has no ROCm packages installed:

```bash
ENV_PREFIX=~/mambaforge/envs/therock
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
  "${PASSTHROUGH[@]}" \
  "$PY" <command>
```

`env -i` discards the caller's environment, so an experiment switch such as
`HIPENGINE_...=1 cmd` on the command line never reaches the child. The command
then runs its default path and reports a plausible number for work it did not
do. Forward the switches explicitly instead of relying on inheritance:

```bash
PASSTHROUGH=()
while IFS= read -r line; do
  case "$line" in
    HIPENGINE_*=*|QWEN4EXP_*=*) PASSTHROUGH+=("$line") ;;
  esac
done < <(env)
```

Add the array to the `env -i` argument list after `PYTHONPATH=.`. Every wrapper
in this repository that uses `env -i` needs the same passthrough; without it, an
A/B whose arms differ only by an environment switch measures one arm twice.

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
| `HIPENGINE_DRAFT_MODEL` | unset | `--draft-model` | Local draft-model or MTP-head path. Paired with `HIPENGINE_SPECULATIVE_PROVIDER` it pins that provider's draft (the Laguna DFlash owner validates its revision and content-addressed safetensors SHA before allocation). Alone, it loads the model plugin's declared MTP provider and fails at load if the plugin declares none. A model that ships its MTP head as a separate GGUF (Gemma 4, Qwen3.8 Flash-Next) finds it beside the target without this setting unless `HIPENGINE_SPECULATIVE_MTP_SERVING=off`. |
| `HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET` | unset | `--speculative-candidate-budget` | Optional candidate depth shared by serving, `LLM`, and direct generators. An explicit positive value is used unchanged; execution must satisfy the owner's implementation capabilities. The legacy `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` alias must agree when both names are set. When neither is set, the loaded model resolves its default from serving evidence (`model_plugin_evidence`) or its generator fallback. A configured generic provider declares its default in the provider registry (`provider_default`; Laguna uses B4). If that declaration varies by registry axes, pre-load reporting stays `pending` until model construction selects those axes. Startup and `/v1/hipengine/capabilities` report the requested value, resolved value and source. |
| `HIPENGINE_MAX_QUEUED_REQUESTS` | unset | `--max-queued-requests` | Optional generation queue cap. When set and the server batcher queue is full, new generation requests fail with HTTP 429 `engine_busy` and `Retry-After: 1`. |
| `HIPENGINE_METRICS` | `off` | `--metrics` | `off` or `prometheus`. When `prometheus`, `/metrics` exposes additive HTTP counters plus resident pending/admitted/active occupancy, work-class totals and policy, bounded queue/TTFT/ITL/service/completion summaries, real KV byte/page/ref/pin/grow/shrink/failure stats, graph capture/replay/invalidation counts by bucket, and route/fallback counters when the loaded generator provides a live-loop snapshot. |
| `HIPENGINE_INFO` | `false` | `--info` | Per-request summary logging. When on, each completed generation request logs one `REQUEST_INFO` line with the endpoint, stream mode, served model, prompt/completion token counts, backend prefill time and rate, server-observed time to first token and stream window, wall time, the request's persistent KV allocation in the shared pool (the space it reserves for its own context, not the bytes its prompt touched), and the pool's own byte/page/pin/growth state. It logs no prompt or generated text. Stream timing is tracked even when the client did not request `stream_options.include_hipengine`; blocking requests have no first-token timestamp and report their decode rate from engine phase accounting instead, so the two bases are reported as separate fields (`stream_decode_ms`/`stream_tok_s` for the server-observed window, `decode_ms`/`decode_tok_s` for the engine's phases). |
| `HIPENGINE_LOG_COLOR` | `auto` | `--log-color` | `auto`, `always`, or `never`. `auto` colorizes a TTY only and honors `NO_COLOR` and `FORCE_COLOR`; `always` also colorizes redirected output such as a `| tee` pane and therefore writes ANSI codes into that file; `never` disables colorization. Styling adds ANSI codes only, so log text and `key=value` fields stay byte-identical for `grep` and log shippers. An unrecognized value falls back to `auto` rather than failing startup. |
| `HIPENGINE_PREFIX_CACHE` | `radix` | `--prefix-cache` | Defaults to `radix` in HTTP serving and the in-process engine. Eligible deterministic requests reuse page-aligned KV and hybrid state from active sources or completed-source snapshots, then prefill only the suffix. Ineligible requests prefill privately and record `prefix_fallback_reason`; this is an eligibility check, not a numerical detector. `/ready` reports cache ownership. The W7900 BF16 layout-repair packet covers four categories and four heldouts with both source lifecycles, exact state, and zero KL against private same-schedule recomputation. Serial-suffix arithmetic and INT8 reuse are separate contracts; see `docs/REFACTOR.md`. Flash-Next keeps device checkpoints of its prompt state instead; see `docs/API.md` "Flash-Next prefix reuse". Set `off` to roll back. |
| `HIPENGINE_REPLAY_DIR` | unset | `--replay-dir` | Opt-in directory for finite JSON failed-request replay artifacts. Disabled by default for sensitive deployments. |
| `HIPENGINE_REPLAY_REDACTION` | `hash` | `--replay-redaction` | Replay artifact string redaction mode: `hash` replaces strings with SHA-256/length metadata, while `none` stores raw strings for explicit local debugging only. |
| `HIPENGINE_ENGINE_COMMAND_TIMEOUT_SECONDS` | `300` | `--engine-command-timeout-seconds` | Liveness budget for one command issued to the resident engine service, in seconds; must be positive. The service runs the engine on a single driver thread, so a command queues behind whatever that thread is already doing: a context prepare that grows the KV pool and captures graphs, or one prefill tick of a large prompt. Both legitimately exceed half a minute, so the default is generous and this value is not a bound on engine work. A command that waits longer than 10 seconds logs one `ENGINE_COMMAND_SLOW` line naming the blocking activity (`driver_command=`, `driver_command_s=`, `driver_tick_s=`, `queued_commands=`, `active_requests=`, `driver_alive=`); an exhausted budget raises an error carrying the same detail, which the server reports as HTTP 503 `engine_unavailable` (retryable), the same shape it uses for a closed engine service. A timed-out command is abandoned by its caller but still runs on the driver thread once that thread is free. The 503 applies to commands that serve a request; the readiness and metrics diagnostics read the engine through a helper that swallows a timeout and returns no snapshot, so `/ready` still answers from its own startup state and simply omits the live snapshot fields rather than failing. Those endpoints are therefore bounded by this budget without turning a busy driver thread into a 503, and `/health` never reads the engine at all. |
| `HIPENGINE_EAGER_LOAD_PROMPT` | `one two three four` | `--eager-load-prompt` | Prompt text used for the server startup warmup. |
| `HIPENGINE_EAGER_LOAD` | `true` | `--eager-load` | Whether to warm the model/session during server startup. |
| `HIPENGINE_EAGER_LOAD_MAX_TOKENS` | `1` | `--eager-load-max-tokens` | Generated tokens used for the server startup warmup; must be positive. |
| `HIPENGINE_STARTUP_CHAT_SMOKE` | `true` | `--startup-chat-smoke` | Runs bounded production-shaped chat requests during eager startup, including one on the speculative MTP route when that route is enabled. |
| `HIPENGINE_STARTUP_SCRATCH_PROBE` | `true` | `--startup-scratch-probe` | Asks the backend to allocate max-context request scratch during eager startup without decoding to the output limit. The probe runs at the configured request width and prices that width first: each probe slot is its own resident session with preallocated KV pages and workspace lease, so a width the machine cannot hold is reduced to one slot and reported in `startup.checks.scratch_probe` and the `/ready` diagnostics. A probe that cannot allocate even one slot ends startup. |
| `HIPENGINE_STARTUP_MIN_FREE_MIB` | unset | `--startup-min-free-mib` | Optional minimum free GPU memory after startup warmup/probes; below this, startup fails. Default disabled. |
| `HIPENGINE_MAX_CONTEXT_TOKENS` | unset | `--max-context-tokens` | Resident session/KV context tokens preallocated at startup; unset takes the model/session default. |
| `HIPENGINE_CHAT_DEFAULT_MAX_TOKENS` | `4096` | `--chat-default-max-tokens` | Default `max_tokens` for chat requests that omit it; `N\|auto` where `auto` uses the remaining context. |
| `HIPENGINE_MAX_CHAT_SESSIONS` | unset | `--max-chat-sessions` | Optional app-local chat session cap before HTTP 429 `engine_busy`; unset is unlimited. |
| `HIPENGINE_REQUEST_TIMEOUT_MS` | unset | `--request-timeout-ms` | Default request deadline in milliseconds; omitted disables the default deadline. |
| `HIPENGINE_DEBUG` | `false` | `--debug` | Logs full HTTP request/response payloads and extra server diagnostics. |
| `HIPENGINE_MTP2_MAX_CONTEXT_TOKENS` | unset | none | Resolves the MTP2 context window used by the dense speculative route; unset takes the model's retained window. Benchmark harnesses export it through `scripts/bench_env_preflight.py` to catch wrappers that silently drop it. |
| `HIPENGINE_MTP2_PREFIX_CHECKPOINT_ENTRIES` | `4` | none | Bounded provider KV/state checkpoints for MTP restoration after a target radix-cache hit. Capture is disabled when target prefix caching is off. Set `0` for the memory/bisection rollback; prefix hits without a provider checkpoint cannot restore MTP provider state. |
| `HIPENGINE_GEMMA4_ARTIFACT` | unset | none | Overrides the Gemma 4 campaign GGUF that `scripts/gemma4_campaign_bench.py` and the probes built on it default to. When unset, the first of the recorded locations that exists is used. Read by the scripts only, never by the runtime. |
| `HIPENGINE_GEMMA4_MOE_GATE_UP_MMQ` | unset (on) | none | Rollback lever for the fused int8-dp4a MMQ32 route on Gemma 4's `Q4_K` expert gate/up prefill, which is the default path and measures 1.27x on the 512/128 prefill. Set it to `0` to restore the fp32 grouped route. On realistic input the route matches its fp32 owner: ten consecutive tokens over 2004 ids of prose and all 23 over a truncated Python snippet, with the arms parting only at low-margin positions and, where they do, parting from llama.cpp together rather than from each other. Its raw-logit perturbation is three times smaller in distribution than out (3.46 against 9.76 max absolute difference), which is the mechanism behind that: the int8 activation-quantization step is small enough not to move a decision on input the model was trained for, and random ids leave the logits flat enough that it is. It is over the binding 0.05 `kl_max` bar at 7 of the 13 frozen-campaign prefill lengths with greedy decision flips at 24 and 64 ids, and at 1024 ids it is 7.3e-07; divergence counts are corpus-specific, because the trigger is the prompt's routing rather than its length, so the same length is bit-exact under one corpus and a decision flip under the other. Retained only as a rollback lever; removed once a teacher-forced gate verdict for the new default is on file. See `docs/REFACTOR.md`. |
| `HIPENGINE_GEMMA4_MOE_DOWN_MMQ` | unset (off) | none | Enables the optional DS4 DP4A MMQ expert down owner with `1`. Automatic selection does not use it because the one-plane MMQ implementation fails nine of ten generated-history screens and three-plane feedback still fails seven. The exact-down control matches all 610 strict rows. Explicit pinned `mmq` selection remains available. The clearing command and removal condition are in `docs/REFACTOR.md`; `scripts/gemma4_generated_history_gate.py` reproduces the paired continuation screen without changing numerical thresholds. |
| `HIPENGINE_GEMMA4_PREFILL_ATTENTION_LOG` | false | Diagnostic | Prints one stderr line per distinct `(prefill-attention variant, head_dim)` naming the variant that ran and why it was chosen. The execution profile selects a variant as a *request*; only the layer knows each layer's head geometry, so a capability miss falls back to the strict kernel there and this line is what makes that visible. Set it to reproduce the route attribution for a Gemma 4 prefill, or when a change to the profile wiring has to be shown to have taken effect rather than inferred from a timing change. |

## Vision (multimodal server input) variables

| Variable | Default | CLI flag | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_VISION_MODEL` | unset | `--vision-model` | Optional Qwen4Exp mmproj GGUF enabling bounded HTTP multimodal input. |
| `HIPENGINE_VISION_MAX_PIXELS` | unset | `--vision-max-pixels` | Decoded-pixel bound for HTTP vision input; unset takes the model plugin's bound. |
| `HIPENGINE_VISION_MAX_IMAGE_BYTES` | unset | `--vision-max-image-bytes` | Compressed base64 PNG payload bound for HTTP vision input. |

## Song generation (companion decoder) variables

| Variable | Default | CLI flag | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_YUE2_VAE_DIR` | HF cache snapshot of `m-a-p/YuE2-Vae` | `--vae-model` | Companion decoder checkpoint for a model that needs one. YuE2 cannot produce audio without it. An explicit path wins; otherwise the local Hugging Face cache is used, and a missing decoder is reported by name. |

## HIP JIT build, compiler cache, and AOTriton discovery

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_COMPILER_VERSION_TEXT` | unset | Literal compiler-version text for cache keys; avoids probing `<compiler> --version`. |
| `HIPENGINE_COMPILER_VERSION_FILE` | unset | Reads compiler-version text from a file. Recommended for cached benchmarks/profiling. |
| `UD_HIPCC_VERSION_FILE` | `/tmp/ud-hipcc-version.txt` | Script-local alias: the leaf-timing harness (`scripts/gguf_iq_local32_decode_leaf.py`) reads its pinned compiler-version text from this path before building the microbenchmark library. |
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
| `HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` | `1024`; model override | `--max-prefill-chunk-tokens` | Maximum tokens per prefill slice; must be > 0. When unset, Qwen4Exp uses the runner's admitted chunk size (up to 4096 with automatic capacity selection). Its decode-active round budget remains at most 1024 tokens. An explicit value overrides the model default. |
| `HIPENGINE_FAIR_PREFILL_BURST_CHUNKS` | `1` | `--fair-prefill-burst-chunks` | Maximum consecutive prefill chunks while `fair` scheduling also has decode-ready rows; must be > 0. The independently gated gfx1151 Q4_K_M package default may override this when the env is unset. |
| `HIPENGINE_KV_POOL_INITIAL_PAGES` | `128` | `--kv-pool-initial-pages` | Initial resident device-KV pages, clamped to the runner's maximum useful capacity; must be > 0. |
| `HIPENGINE_KV_POOL_LOW_WATER_PAGES` | `128` | `--kv-pool-low-water-pages` | Idle-shrink low-water pages, clamped with the initial allocation; must be > 0 and no greater than configured initial pages. |
| `HIPENGINE_KV_POOL_HIGH_WATER_PAGES` | unset | `--kv-pool-high-water-pages` | Optional atomic grow-on-admission page cap; unset means no configured pool cap. |
| `HIPENGINE_KV_POOL_CHUNK_PAGES` | `128` | `--kv-pool-chunk-pages` | Real device pages per grow/shrink tail chunk, clamped to useful runner capacity; must be > 0. |
| `HIPENGINE_KV_POOL_MEMORY_BUDGET_MIB` | automatic | `--kv-pool-memory-budget-mib` | Dense-GGUF KV payload ceiling in MiB, shared by the allocated arena (including pinned workspace pages) and private packed-workspace KV. Initial allocation, private fallback, and pool growth reject requests that exceed it. When unset, derived from live free HIP memory after reserve. Recurrent state, execution scratch, and pointer-table metadata are outside this KV payload ceiling and remain visible in allocator/workspace telemetry. |
| `HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS` | `30.0` | `--kv-pool-idle-grace-seconds` | Seconds before fully-free, graph-unpinned tail chunks are eligible to shrink; must be >= 0. |
| `HIPENGINE_MAX_PENDING_REQUESTS` | unset | `--max-pending-requests` | Optional pending request queue cap for the resident scheduler; must be > 0 when set. |
| `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET` | `1024` | `--round-prefill-token-budget` | Token-budget prefill work per engine-loop round; must be positive. |
| `HIPENGINE_DECODE_PREFILL_TOKEN_BUDGET` | model-selected (`1024` for Flash-Next, capped by its chunk size); otherwise unset | `--decode-prefill-token-budget` | Optional bound on the whole prefill share of a token-budget round while decode is due. Idle rounds keep the ordinary prefill budget. Flash-Next can suspend that work at completed-layer boundaries; a layer is indivisible, so this does not guarantee inter-token latency. An explicit `HIPENGINE_ROUND_PREFILL_TOKEN_BUDGET` suppresses the model-selected active bound unless this variable is also set; must be positive. |
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
| `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` | unset (packed-decode package floor) | Broad rowtile rollback/diagnostic | Selects BF16 Q8T16 rowtiling for eligible `in=2048` singleton, pair and triple projections with rows >1 and no explicit thread override. Packed decode enables the broad policy from four rows on gfx1151; gfx1100 leaves it off. `0` disables this broad policy; `1` forces it within those shape bounds. The independent pair floor and `HIPENGINE_GGUF_Q8_T16_ROWTILE_SHAPES` singleton policy are unaffected. The gfx1151 MoE strict profile sets `0`; production removes the override to use package policy. Named suite/profiler comparisons can force `1`. |
| `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE` | unset (session policy) | Pair-kernel rollback/diagnostic | Selects the BF16 Q8T16 `in=2048, out=8192+4096` pair for rows >1 without an explicit thread override. Packed decode sets a row floor of 8 on gfx1100/gfx1151; gfx1151's broader rowtile policy also enables the pair below that floor. `0` forces per-row execution; `1` forces rowtiling within these shape bounds. The verifier profiler's `--q8-t16-pair-rowtile` sets `1`; omission preserves inherited settings. Column selection is independent. |
| `HIPENGINE_GGUF_WMMA_PREFILL` | false | Low-level performance selector | Process-wide default for low-level GGUF sessions. The public generator passes `use_wmma_prefill=True`; benchmark CLI/session arguments remain explicit for artifact provenance. |
| `HIPENGINE_GGUF_GEMV_DECODE` | false | Low-level performance selector | Process-wide default for low-level GGUF sessions. The Qwen3.5 public generator passes `use_gemv_decode=True` at its call sites in `hipengine/generation/qwen35_gguf.py`; the Gemma 4 generator does not pass it, so Gemma 4 takes the process-wide default and this variable is the only thing that selects the newer family for it. For qwen35moe, effective use is safety-gated unless decode-repack is active or the unsafe override is set. **Measured neutral on Gemma 4, so it is not a Gemma 4 performance lever.** On 2026-09-28 the flag moved 205 of 410 dense `gguf_q8_0` decode resolutions from `pack8_gemv_bf16_bf16_out` to `pack8_gemv_decode_bf16_bf16_out` and left the 512/128 decode at 24.81 against 24.83 tok/s. Dense `gguf_q8_0` is 1866.7 MB of the 3219.6 MB a decode token moves, so this was the right place to look; the two decoders simply run at the same rate on these shapes, which means the 58 percent of decode traffic on the legacy decoder is not limited by the decoder's choice. |
| `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` | false | Unsafe diagnostic | Bypasses qwen35moe GGUF fast-path safety. Do not set for normal use or promoted correctness claims. |
| `HIPENGINE_GGUF_AOTRITON_PREFILL` | `v3` | Attention implementation selector | `v3`, `v2`, or `auto`/`v2-if-safe`. `v2` is rejected for chunked suffix prefill because it has the wrong causal-mask semantics there. |
| `HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` | `1024` | Decode threshold | Context length where GGUF full-attention decode uses split/paged decode; `0` disables. Compatibility alias: `NANOVLLM_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT`. |
| `HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG` | backend default for `MOSTLY_Q4_K_M` on gfx1151 | Rollback seam | Lets dense (non-MoE) linear-attention verifier rows stage their projections and FFN above the full-attention split threshold instead of taking the row-wise c1 route. The default reads the backend package capability `GGUF_STAGED_LINEAR_ROWS_LONG_DEFAULT_FILE_TYPES` and applies only to a loader-qualified plain artifact whose stamp is listed there (`mostly_q4_k_m` on gfx1151); a preset-bound (UD) or unknown-manifest artifact, an unlisted stamp, and gfx1100 all keep the row-wise c1 strict route. `0` forces the row-wise route and `1` forces the staged chain on any artifact; either value overrides the capability, and the variable never changes single-row decode. |
| `HIPENGINE_GGUF_GDN_PREFILL_MODE` | `auto` | Correctness/performance diagnostic | Selects `auto`, `exact`, `fused`, `chain`, or the named `chain_k2`, `chain_peer_wave32`, `chain_compact_peer_wave32`, `chain_peer_cluster8`, `chain_tile64`, `chain_tile32`, `chain_wave32`, `chain_wave32_tree`, `chain_lds64`, `chain_lds32`, `chain_lds32_direct`, and `chain_lds32_direct_nonvolatile` GGUF GDN prompt-prefill routes. `chain` is the raw-Q/K-plus-scale exact split fallback; `chain_lds32_direct_nonvolatile` is the promoted GPF-2E compact-scale/direct-`conv_out` route on gfx1151, while volatile direct and materialized `chain_lds32` remain rollback/bisection controls. The tile/wave/LDS64 routes are rejected diagnostics. `auto` uses backend-package policy: byte-exact direct LDS32 on gfx1151 and the bit-equivalent compact peer-wave route on gfx1100, with fused correctness fallback if the preferred route is unavailable. `chain_peer_wave32` retains the prior per-V-head materialization as rollback; `chain_compact_peer_wave32` stores normalized Q/K once per K head. Every explicit selection overrides backend policy and fails closed if its complete implementation is unavailable; invalid values are errors. |
| `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` | `256` | Prefill threshold | Minimum rows for GGUF GDN recurrent-segments prefill routing; invalid values fall back to the default, values below 1 clamp to 1. |
| `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA` | backend-scoped (`1` through 4K on gfx1100/gfx1151; otherwise `0`) | LCP-M2 rollback/diagnostic | Selects the stream-ordered `prepare_prefill_chunk_metadata` kernel (`1`) or six synchronous host-prepared metadata copies (`0`). With the variable unset, gfx1100 and gfx1151 select the exact device path only through 4,096 prompt tokens; longer prompts use host metadata. Explicit values override the ceiling. Conservative/split prefill isolation profiles pin `0`; the verifier profiler selects `1`. A historical explicit gfx1151 128K run stalled, but later isolation reproduced with metadata off and implicated Q4 shared-X; this is not an established metadata failure. See `docs/reference/DEBUG-GFX1151-STALL.md` for the bounded protocol and `docs/REFACTOR.md` R0281-METADATA for interface cleanup. |
| `HIPENGINE_GGUF_PREFILL_F16_STAGING` | profile-scoped | Rollback seam | Selects F16 activation staging for GGUF bulk prefill; the production profile enables it, strict does not (asserted in `tests/test_gpu_gguf_k_t16_dense_f16_activation_prefill.py`). |
| `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS` | gfx1100/gfx1151: `128`; otherwise `512` | LCP-4B rollback/diagnostic | Overrides the bulk-prefill `qwen35_router_select_kernel` workgroup with `64`, `128`, `256`, or `512`. Both gfx1100 and gfx1151 default to their independently full-model-exact 128-thread launch; decode retains its independent 256-thread geometry. Use `512` for rollback. Do not use `64` for production: it is fastest in a primitive screen but failed the gfx1151 4K full-model exact-state gate. |
| `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE` | `auto` | GPF-3A rollback/performance selector | `auto`, `baseline`, or `shared_x`. `auto` reads backend-package capability: gfx1100 and gfx1151 select their independently clean-gated byte-exact `shared_x` schedules. Explicit selections fail closed when their registry variant is unavailable; `shared_x` is mutually exclusive with the DS4 selected-prefill diagnostic. Leave unset for the production backend policy; use `baseline` only for rollback/A-B. |
| `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` | unset (auto) | Placement override | The shared Qwen GGUF runner (`hipengine serve`, `LLM`) maps raw Q4_K token tables from pinned host memory on gfx1100 and gfx1151; gfx1151 registers an anonymous byte-for-byte copy, never the file-backed mmap. Unsupported types and externally owned weights keep device ownership. `0`/`device` keeps the table on the device. For private c1 sessions, `1`/`host` also offloads a raw Q8_0 table and serves exact Q8_0→BF16 rows from host, which disables GGUF HIP decode graph replay. |
| `HIPENGINE_GGUF_DECODE_GRAPH` | true | Retained default with rollback opt-out | Enables GGUF resident decode HIP graph replay. Backend capability gates whether a graph is captured at all; `0` forces eager decode for bisection. |
| `HIPENGINE_GGUF_MOE_GRAPH` | false | Opt-in diagnostic | Captures scalar stateless per-layer FFN work while attention stays eager. The parity test disables next-RMS chaining to engage it. Incompatible with whole-step decode graphs; G5/PM4 harnesses explicitly disable it. |
| `HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS` | `8` | Correctness fallback | Number of leading GGUF full-attention layers kept as BF16 primary storage for long explicit `int8_per_token_head` sessions. Long contexts require at least 8 BF16-prefix layers unless `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` is set. The 2026-06-24 W7900 gate accepts prefix 8 at `128K/128` after the layer-local BF16 prefill-oracle fix (`KL mean=0.01448`, top-1 `0.96124`, no persistent BF16 mirror); prefix 7 still fails `128K/16`, and pure INT8 fails `4K/1`. Short contexts (`<=8192` rounded max context) still use the exact BF16 mirror instead. |
| `HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS` | unset | Unsafe diagnostic | Comma/range list of zero-based GGUF full-attention indices to keep as BF16 primary storage instead of using the leading-prefix rule, e.g. `0-5,7`. Requires `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` unless the list exactly matches the admitted default prefix. Added for non-contiguous sensitivity sweeps; 2026-06-24 W7900 `128K/16` masks with INT8 layers `{6,8,9}` and `{5,8,9}` still failed the BF16-vs-INT8 guard, so no custom mask is promoted. Set it to `none` (or `empty`/`-`) to clear the BF16 prefix and make the full-attention stack uniformly INT8, which restores the packed decode width to the declared `max_direct_rows`. `0` means layer 0, not none. |
| `HIPENGINE_GGUF_INT8_KV_KEY_ONLY` | false | Unsafe diagnostic | For explicit GGUF `int8_per_token_head` sessions, store retained K as INT8 but V as BF16 for INT8-selected full-attention layers. This is a diagnostic key-only layout, not a promoted 24GB path: 2026-06-24 W7900 prefix `0` failed `4K/1`, prefix `6` failed `128K/16` top-1, and prefix `7` passed `128K/16` but saved less memory and had higher prefill peak than the admitted prefix-8 per-token/head path. |
| `HIPENGINE_GGUF_INT8_KV_BLOCK16` | false | Unsafe diagnostic | For explicit GGUF `int8_per_token_head` sessions, use the guarded block16 INT8 K/V scale granularity (`[blocks, block_size, kv_heads, 16]`) and route GGUF retained-KV write/decode through the block16 HIP kernels. This is a runtime diagnostic for the Q8-format follow-up, not a promoted path: 2026-06-24 W7900 forced-long `4K/1` BF16-vs-block16 gates fail top-1 even at prefix `8`. Do not combine with `HIPENGINE_GGUF_INT8_KV_KEY_ONLY`. |
| `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG` | false | Unsafe diagnostic | Allows long GGUF `int8_per_token_head` diagnostics below the verified 8-layer BF16 prefix (including pure INT8-only, key-only, block16, or non-contiguous BF16 layer masks when the custom mask env is set). Leave unset for normal use: pure INT8-only and lower memory-saving prefixes failed BF16-vs-INT8/GGUF hybrid logit gates and are capacity-diagnostic only. |
| `HIPENGINE_GGUF_COMPACT_MOE_C1` | false | Diagnostic fallback | Forces the older compact c=1 MoE decode scheduler; current retained default uses direct selected T16 kernels instead. |
| `HIPENGINE_GGUF_SIDECAR_CACHE` | `~/.cache/hipengine/gguf_sidecars` (or `XDG_CACHE_HOME`) | Sidecar cache | Cache directory for optional GGUF expert pack8 sidecars. |
| `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` | unset | Kernel R&D | Optional launch-bounds macro for selected WMMA prefill builds; unset uses the retained defaults. |
| `HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS` | false | Diagnostic instrumentation | Adds non-sync HIP-event intervals inside the GGUF packed target verifier and rolls them into `target_packed_verify_gpu_*` timing buckets. Leave unset for speed claims because event recording adds overhead. |
| `HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS` | retired | No longer read | Backend policy keeps the multi-row compact no-read ceiling at 4096 selected rows on gfx1100/gfx1151. The tight routing-independent padded bound must also fit scratch capacity; larger or capacity-limited calls retain the scalar `wmma_total` read. c1 decode bypasses this helper. |
| `HIPENGINE_GGUF_AR_PACKED_PREFILL` | true | Retained default with rollback opt-out | Enables GGUF server greedy-AR packed final-row prompt prefill for c>N coalesced requests. Set `0`/`false`/`off` to force serial per-slot prefill fallback for bisection. |
| `HIPENGINE_GGUF_AR_PACKED_DECODE` | true | Retained default with rollback opt-out | Enables GGUF server greedy-AR packed resident decode for c>N coalesced requests. Set `0`/`false`/`off` to force stream/scalar fallback for bisection. |
| `HIPENGINE_GGUF_AR_STREAM_DECODE` | true | Retained fallback with rollback opt-out | Enables per-slot stream decode fallback and c>4 packed-decode chunk streams. Set false only for bisection. |
| `HIPENGINE_GGUF_FP16_RECURRENT_STATE` | backend/artifact scoped (true for plain gfx1151 `mostly_q4_k_s`; false otherwise) | Default with strict-storage comparison override | Resolves and freezes FP16 or FP32 recurrent-state storage before allocation; FP16 writers accumulate in FP32. Preset-bound or unknown-manifest artifacts keep FP32 unless explicitly overridden. Set `0`/`false`/`off` for FP32 or `1` for FP16; an unsupported writer/route raises. SPECDEC2 has a profile-bound eager FP16 target with typed snapshots; native target graphs and the legacy tree/chain route reject FP16 state. |
| `HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT` | retired | No longer read | Compatible multi-slot drafts use per-slot HIP streams. Singleton/tail, missing seed and runtime-hook fallback remain; client token delivery is unchanged. |
| `HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY` | retired | No longer read | Compatible multi-chunk target verification uses per-owner HIP streams. Missing runtime/kernel hooks keep pre-launch fallback; client result delivery is unchanged. |
| `HIPENGINE_GGUF_MTP_HOT_VOCAB` | `auto` under `production`; otherwise unset | Performance selector | Hot-vocab LM-head cache scope for the GGUF MTP server. Unset resolves `auto` under the production profile and no hot-vocab cache otherwise, and the server drops the cache (about 0.5 GiB on Qwen3.8-27B, about 2% slower MTP decode) when the context would otherwise fall short of the model maximum or the pinned context. An explicit value is always kept: `auto` keeps the packaged cache, a positive integer selects the head count, and `0`/`off` disables the cache. |
| `HIPENGINE_GGUF_PACKED_KV_LEASE` | `0` | Diagnostic | Forces the packed-KV lease helper on for single-request (max_batch_size==1) sessions where it would otherwise stay off. |
| `HIPENGINE_GGUF_MTP_VERIFY_MODE` | `native` | GGUF dense MTP server | `native` or `serial_exact`. `native`, the default, checks drafted tokens with the fast llama.cpp-style row-attention and GPU-side acceptance path; on the dense Qwen suites that measures about 1.5-1.7x normal decoding, with occasional differences in the token chosen. `serial_exact` replays normal single-request decoding for each candidate, so it agrees with normal decoding token for token, but it cannot be faster. |
| `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` | unset | Legacy candidate-depth alias | Compatibility alias for `HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET`, not an independent fallback. Both names must agree if set together; invalid or conflicting values fail. Direct generator callers and `LLM` preserve the explicit depth. A constructor argument takes precedence over environment settings. |
| `HIPENGINE_MTP2_SCREEN_UNQUALIFIED_CELLS` | retired | — | No runtime reader. Explicit MTP requests use implementation capability declarations; automatic selection keeps its performance policy. This old screening switch cannot override shape, storage, sampling or resource limits. |
| `HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M`, `HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_N` | `32` / `16` | Kernel R&D | Q4_K selected WMMA tile override (both must be set as an allowed pair; default pin is 32x16). Allowed tile pairs are validated by the build helper. |
| `HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_M`, `HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_N` | `16` / `16` | Kernel R&D | Q5_K selected WMMA tile override. |
| `HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_M`, `HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_N` | `16` / `16` | Kernel R&D | Q6_K selected WMMA tile override. |
| `HIPENGINE_GGUF_Q8_0_WMMA_TILE_M`, `HIPENGINE_GGUF_Q8_0_WMMA_TILE_N` | retired | No longer read | Q8_0 dense WMMA tiles use execution-local policy or backend heuristics; explicit kernel tile arguments take precedence. Qwen4Exp publishes `64`/`32` on its runner without changing other models' defaults. |
| `HIPENGINE_GGUF_DENSE_WMMA_BULK` | `1` | Retained default with rollback opt-out | Uses the dense BF16 WMMA bulk-prefill kernel for dense GGUF linear layers (backend capability `GGUF_DENSE_BF16_WMMA_BULK_PREFILL` permitting); `0` restores the non-WMMA bulk body. |
| `HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL` | `1` | Retained default with rollback opt-out | Uses the dense BF16 WMMA path for residual-class projections in bulk prefill; `0` restores the plain dispatch. |
| `HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK` | `1` | Retained default with rollback opt-out | Uses the pack8 Q4 WMMA bulk-prefill owner; `0` restores the non-WMMA pack8 body. |
| `HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL` | `1` | Retained default with rollback opt-out | Uses the pack8 Q4 dual WMMA SiLU prefill pair; `0` restores the separate owners (set to `0` by the Qwen3.5-0.8B cumulative role environment). |
| `HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL` | `1` | Retained default with rollback opt-out | Uses the Q8T16 dual WMMA prefill pair helper; `0` restores the separate per-projection owners. |
| `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR` | unset | Harness-set materialization | Retains raw GGUF bytes alongside T16 tiles for Q8_0 dense weights at materialization time. Required by the dp4a verifier/dense-Q8 diagnostic routes; set by the MTP bench harness before model load. |
| `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR` | unset | Harness-set materialization | Retains the X8 Q6_K sidecar for `lm_head` at materialization time; required by the verifier direct top-1 dp4a route. |
| `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW` | off | Load-time primary layout | With decode repack enabled, keeps rank-3 Q4_K gate/up experts in raw GGUF storage, taking precedence over gate/up X8. Set before loading. The runtime uses its raw selected-dual consumer; q8_1/dp4a requires the separate selected-kernel control. Historical trace replay preserves this choice; the old bench switch and `rawgateup` suite aliases have been removed. |
| `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` | off | Load-time replacement layout | With decode repack enabled, selects primary X8 tiles for rank-3 Q4_K gate/up experts; explicit gate/up raw placement takes precedence. Set before loading. Parity workbench candidates `q4-x8` and `q4-x8-q6` and historical trace replay use it; the old bench/profiler switch and `x8gateup` suite aliases have been removed. |
| `HIPENGINE_GGUF_SELECTED_X8_REPACK` | off | Load-time replacement layout | `q5`, `q6` or `both` selects X8 tiles for rank-3 selected-down experts when decode repack is enabled; true/`1` aliases `both`. Compatibility generation selects `q6`. Explicit selected-down raw placement takes precedence. This replaces primary storage, not a sidecar; execution requires q8_1 workspace. Set before loading, not after materialization. |
| `HIPENGINE_GGUF_RAW_SELECTED_DP4A` | unset | Explicit activation adapter | Enables raw q8_1/dp4a selected-expert GEMVs; read live by the runner and selected-caller contract. The named compatibility generator and benchmark `--verify-dp4a` enable it. It changes arithmetic; the historical Japanese-quality failure is recorded in `docs/archive/MTP-LLAMACPP-PARITY-HISTORY.md`. Omission or `--no-verify-dp4a` does not clear inherited settings. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A` | unset | Diagnostic route | Enables the dense Q8_0 dp4a pair helper; requires `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR=1` to be set before materialization. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` | unset | Diagnostic route | Enables BF16 singleton/QKV-triple helpers as well as the pair helper for rows>1. At load time, broadens raw sidecars to every dense Q8 T16 layer tensor only when `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR=1` is also set. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED` | retired | No runtime reader | The rejected per-projection shared-expert verifier route and its CLI/suite/profiler switches are removed. Normal shared-expert projections and independent resident-draft shared-dual execution remain. |
| `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` | unset | Diagnostic route | Enables rows>1 F32-input singleton/triple dp4a helpers with raw Q8 sidecars and workspace, including `ssm_out`. F32 pair helpers use the base pair/all control instead; F32 pair outputs also require `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS`. |
| `HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A` | unset | Diagnostic route | Singleton verifier direct top-1 path reading the X8 Q6_K `lm_head` sidecar; multirow blocks use BF16 hidden rows and the full-logit sampler. Production gates force it off. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN` | unset (bound on by public/profile callers) | Verifier state capture | Selects prefill Conv and decode-order no-copy GDN row-state writers. Packed verification requires this path; public generation and profile binders enable it, and llama-compat forces it on. This is execution/state policy, not only offline instrumentation. |
| `HIPENGINE_GGUF_VERIFY_GDN_SEMANTIC_GATE` | unset | Gate instrumentation | Reserves the GDN semantic-gate capture scratch and enables the semantic-gate comparison route; must be set before session construction so the scratch allocation precedes capture. |
| `HIPENGINE_GGUF_PACKED_VERIFY_CAPTURE_DIR` | unset | Diagnostic capture | Writes diagnostic BF16 output-norm rows for offline full-logit gates into this directory. |
| `HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL` | profile-scoped | Rollback seam | Uses the integer MMQ prefill owner for Q6_K rows; enabled by the production profile binder, disabled under strict. |
| `HIPENGINE_GGUF_Q5_MMQ32_PIPE` | unset | Kernel R&D | Selects the pipelined MMQ32 consumer variant for Q5_K prefill dispatch; used by the MMQ test/bisect harness. |
| `HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS` | `128` | Kernel R&D | Q6 top-1 wrapper workgroup size, `64` or `128`; explicit wrapper arguments override it. Shared by draft and other direct wrapper callers. |
| `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE` | `pack8` | Resident draft layout | Read at draft-runner construction: `pack8`, `pack16`, `pack8_llama`, `pack8_scalehoist`, `row`, `x8`, `x8_dscale`. Compatibility generation sets `x8`. X8 layouts upload a vocabulary-capped draft sidecar independently of `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR`; this shape alone does not enable q8_1/dp4a execution. `pack16` requires vocabulary divisible by 16; unknown shapes raise. |
| `HIPENGINE_GGUF_FUSED_MOE_FFN` | unset | Teacher-forced gate only | Set only inside `scripts/gguf_fused_moe_ffn_teacher_forced_kl.py` to A/B a fused MoE FFN candidate; not a runtime selector. |
| `HIPENGINE_GGUF_AR_D2_COST_ARTIFACT` | unset | Harness | Path to a measured D2 cost artifact; when set, `scripts/gguf_arbitrary_c_lifecycle.py` derives the D2 composition from the artifact instead of the ceiling heuristic. |
| `HIPENGINE_GGUF_AUTO_CONTEXT` | on | Retained default with rollback opt-out | When the caller does not pin a context, the resident GGUF session prices its footprint against free HIP memory after weights load and takes the largest block-aligned context that fits with a safety reserve. `0` keeps the historical fixed default. |
| `HIPENGINE_GGUF_KV_CAPACITY_RESERVE_MIB` | `768` | Auto-context safety | Reserve subtracted from free memory for device memory the capacity model does not price (HIP context growth, JIT modules, graph captures). |
| `HIPENGINE_GGUF_KV_TRANSIENT_KIB_PER_TOKEN` | unset (route-priced) | Auto-context pricing | When set, replaces the route-derived prefill transient with a flat KiB per token. |
| `HIPENGINE_GGUF_KV_TRANSIENT_FIXED_MIB` | unset (`0`) | Auto-context pricing | When set, adds a fixed transient MiB to the auto-context fit. |
| `HIPENGINE_GGUF_AUTO_CONTEXT_ATTEMPTS` | `4` | Auto-context retry | Maximum shrink-and-retry attempts after a failed context allocation; a failed allocation falls back to a smaller context instead of surfacing. |
| `HIPENGINE_GGUF_PREFIX_RETAINED_SNAPSHOTS` | narrow default | Prefix cache | Number of completed-source snapshots the radix prefix cache retains; defaults to `max(1, capacity)`. A wider count retains more state and pins more KV pages. The state-byte ceiling also limits retention; it does not include the pinned KV pages. |
| `HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES` | Dense: `1073741824` (1 GiB); DMS: `4294967296` (4 GiB); Flash-Next: half of free device memory at the first checkpoint, at most 32 GiB | Prefix cache | Byte cap on retained prefix snapshots, allocated on demand. The DMS cap includes host-owned cold prompt KV and hidden rows for prefix extension. An explicit value overrides either default; lower it on memory-constrained hosts. |
| `HIPENGINE_GGUF_PREFIX_GAPPED_SUFFIX_MAX` | `512` | Prefix cache | Maximum gapped-suffix tokens a prefix-cache hit may reuse; measured against the full prefill a hit replaces. |
| `HIPENGINE_GGUF_PACKED_LAYER_OUTER` | true | Retained default with rollback opt-out | Session-paired (one pair per session) packed layer-outer executor; `0` restores the per-layer-keyed chunk-outer rollback owner. |
| `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED` | unset | Unsafe diagnostic | Shorter spelling accepted alongside `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG` in the INT8-KV diagnostic override set. |
| `HIPENGINE_GGUF_INT8_PREFILL_DIRECT` | unset | Correctness/perf gate | Selects the oracle-free direct INT8 prefill attention route (`twopass` candidate) instead of the oracle-bridge strict path (write-through BF16 oracle pair via AOTriton). |
| `HIPENGINE_GGUF_INT8_PREFILL_KERNEL` | `flash` | Route selector | Direct INT8 prefill attention kernel implementation: `flash`, `wmma`, or `sequential`; other values are errors. |
| `HIPENGINE_GGUF_INT8_PREFILL_SLOT_LOCAL_AOTRITON` | unset | Diagnostic | Slot-local AOTriton variant of the INT8 prefill route (gate provenance key). |
| `HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS` | unset (adopted single plane) | Rollback seam | Declares the layer-outer hidden-plane alias mode for INT8 evaluation sessions; `0` rolls back to the multi-plane layout. |
| `HIPENGINE_GGUF_AOTRITON_PREFILL_ENABLE` | backend capability | Attention policy | Overrides the backend's AOTriton-prefill threshold policy (the measured 512-token crossover) on either side. |
| `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES` | `536870912` (512 MiB) | Attention R&D | Resource cap for the head-major K/V pair, including demand-sized gapped gather buffers. |
| `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS` | `65792` | Attention R&D | Resource cap for head-major K/V capacity. Contiguous layout follows backend package policy; gapped gather uses its own switch below. |
| `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM` | unset | Attention R&D | At >= 512 query rows, PARO defaults to an isolated stream; GGUF defaults to same-stream after observed stability failures. `0`/`1` selects the comparison policy for either family. |
| `HIPENGINE_GGUF_GAPPED_GATHER` | unset (enabled) | Prefill R&D | Gather gapped BF16 prefix pages into demand-sized head-major buffers so suffix attention uses the contiguous route. Requires the registered copy kernel and obeys token/byte caps, including allocation rounding. `0` selects the native paged fallback. |
| `HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE` | `auto` (gfx1100: `baseline`; gfx1151: `tile32x128`) | Schedule diagnostic | Accepts `auto`, `baseline` or `tile32x128`; used by kernel-isolation profiles. The full-stack runner caches the callable after first resolution; select before construction/use. |
| `HIPENGINE_GGUF_Q4_T16_SINGLE_WAVE_MAX_ROWS` | unset | Kernel R&D | Row cap for the Q4_T16 single-wave rowtile variant. |
| `HIPENGINE_GGUF_Q8_T16_PAIR_COL8` | unset (session pair floor) | Pair-kernel rollback/diagnostic | Chooses the eight-column body only after pair rowtiling is selected. Unset selects it at or above the packed-decode pair floor of 8 on gfx1100/gfx1151; `0` uses the 16-column body, and `1` forces eight columns for an eligible rowtiled pair. This variable alone does not enable rowtiling. |
| `HIPENGINE_GGUF_Q8_T16_ROWTILE_SHAPES` | `(5120,1024)` | Rollback seam | Shape-explicit Q8T16 rowtile admission set for the UD verifier. |
| `HIPENGINE_GGUF_Q8_T16_PREFILL_2WAVE` / `HIPENGINE_GGUF_Q8_T16_PREFILL_4WAVE` | unset | Diagnostic | Shape-scoped dense Q8T16 WMMA schedule overrides, shared by decoding-weight and owned-`f16w` launches. Explicit `4WAVE` controls the four-wave choice first; without it, any explicit `2WAVE` suppresses automatic four-wave, with `1` requesting two-wave and `0` the decoding fallback. With neither set, request context and wrapper defaults choose; Qwen's automatic wide-wave context ends at 4,096 prompt tokens on gfx1100 and 65,536 on gfx1151. Four-wave also covers Gemma 4's 1,408-wide expert gate/up; unsupported tile/width shapes keep the decoding fallback. See `docs/REFACTOR.md` R0281-Q8-WAVES for coordinated interface cleanup. |
| `HIPENGINE_GEMMA4_DECODE_GRAPH` | `1` | Rollback seam | Replays Gemma 4 decode steps from a whole-step HIP graph capture in the generation loop and the campaign bench's instrumented path. The flash phase kernel re-derives its slice partitioning per block from a staged device live-extent slot, so flash, split and class routes all replay bit-identically; the capture window is bounded by the measured launch-bound prize (buckets below 2176 keys capture — +15.4% decode at 1024, +2.2% at 2048; deeper device-bound steps take the launched fallback, which costs less than a capture). Those figures are from the Radeon 8060S; a backend package can declare its own window as `GEMMA4_DECODE_GRAPH_CAPTURE_MAX_KEYS`, and `hip_gfx1100` declares `0` because a W7900 measured no gain at any depth (0.93x at 1536 and 2048), so its decode stays launched. `0` restores fully launched decode. |
| `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` | unset | Diagnostic route | llama-compat selected-expert dp4a adapter; set by the llama-compat MTP environment map and read by the selected-caller contract. |
| `HIPENGINE_GGUF_T16_SELECTED_DP4A` | unset | Diagnostic route | llama-compat T16 selected dp4a adapter; set by the llama-compat MTP environment map and read by the selected-caller contract. |
| `HIPENGINE_GGUF_T16_DS4_PREFILL` | ignored | Retired diagnostic | No runtime reader. Regression tests pin stale values; standalone DS4 parity tests and microbench kernels remain available. |
| `HIPENGINE_GGUF_T16_SELECTED_PAIRREUSE` | unset | Diagnostic route | T16 selected pair-reuse route selector (batch-route gate). |
| `HIPENGINE_GGUF_T16_SELECTED_DOWN_PAIRREUSE` | unset | Diagnostic route | T16 selected-down pair-reuse route selector. |
| `HIPENGINE_GGUF_T16_SELECTED_Q6_DOWN_PAIRREUSE` | unset | Diagnostic route | T16 selected Q6-down pair-reuse route selector. |
| `HIPENGINE_GGUF_IQ_DENSE_ROW_BATCH_DOWN` | unset | Rollback seam | IQ dense row-slab rule: unset picks the smallest slab at or above the row count (measured default); `1` restores the largest slab at or below it. |
| `HIPENGINE_GGUF_IQ_GROUPED_PREFILL` | retired | No runtime reader | Grouped raw-IQ prefill is selected by assignment count and registered kernel capability. Stale values have no effect; direct selected fallback remains. |
| `HIPENGINE_GGUF_MOE_TAIL_NEXT_RMS` | true | Retained default with rollback opt-out | Fuses the MoE tail with the next layer's input RMSNorm; `0` restores the separate chain (set to `0` in the MoE-graph parity test). |
| `HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL` | unset | B1 transfer | Explicit override for the target-side WMMA prefill transfer: `1`/on forces the transfer, `0`/off restores the GEMV owners everywhere. |
| `HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE` / `HIPENGINE_GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT` | retired | No longer read | Backend package policy selects parallel reduction from 32768 context tokens on gfx1100/gfx1151. Serial reduction remains below that threshold and for unsupported routes. |
| `HIPENGINE_GGUF_ROW_COMPACT_GEMV` | retired | No runtime reader | The rejected row-compact verifier route and workbench candidate are removed. Bulk selected-GEMV fallback, compact WMMA and the independent singleton compact route remain. |
| `HIPENGINE_GGUF_VERIFY_ROW_LM_HEAD` | unset | Diagnostic route | Parity-workbench candidate: row-batched lm-head/argmax sampling in the target block verifier. |
| `HIPENGINE_GGUF_SELECTED_DOWN_RAW` | unset | Harness-set materialization | Keeps selected-down expert tensors in raw GGUF layout (workbench value `both`), pairing with the raw dp4a route. |
| `HIPENGINE_C8_Q5_PLANAR_DP4A` | false | Diagnostic route | Gates the optional planar INT8 Q5 sidecar (uploaded at materialization) for the planar-dp4a route. |
| `HIPENGINE_C8_Q6_DP4A_GROUPED` | false | Opt-in diagnostic | Routes planar-decode rows 8-64 to the grouped integer-dp4a Q6 sibling (x quantized to q8_1); the BF16 grouped owner stays the exact default and strict fallback. |
| `HIPENGINE_UD_REPACK_ELIGIBILITY` | unset | Rollback seam | Overrides model-wide UD repack eligibility (e.g. `model-wide`) so a synthetic geometry keeps its historical pack8/raw residents. |
| `HIPENGINE_GGUF_SPECDEC2_EXACT_TARGET_ROWS` | unset | Specdec2 route | Exact target row-count override for the specdec2 verifier rows. |
| `HIPENGINE_GGUF_SPECDEC2_Q6_MIXED_TARGET_ROWTILES` | unset | Specdec2 route | Q6 mixed target rowtile admission override. |
| `HIPENGINE_SPECDEC2_POST_REJECT_COOLDOWN` | retired | No runtime reader | The rejected one-cycle suppression experiment is removed. Normal K0 provider catchup remains; stale values do not suppress speculation. |
| `HIPENGINE_GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA` | retired | No longer read | Private-c1 weight allocation follows backend policy; programmatic opt-out, shared-runner and allocation fallbacks remain. |
| `HIPENGINE_GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA` | retired | No longer read | Private-c1 decode scratch follows the backend policy; programmatic opt-out, shared-runner and allocation fallbacks remain. |
| `HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS` | gfx1151: `1024` | Rollback seam | Overrides the short-prompt c1 batch-attention workgroup; the calibrated 1024-thread variant passed the execution-profile c1 threads gate and the exact 256-thread leaf stays the strict fallback. |
| `HIPENGINE_GGUF_DIAGNOSTIC_WMMA_PREFILL` | unset | Diagnostic | Session-acquire-time diagnostic resolver that forces the WMMA-prefill route without touching the production session default. Unrecognised values raise rather than silently reading as route-unchanged. |

### Resident MTP draft dense-Q8 diagnostic controls

These controls are read by `Qwen35GGUFResidentMTPDraftRunner`, used by the
compatibility generator and draft harnesses, not the Generation-2 NextN
provider. Set them before constructing that runner. They quantize F32
activations into q8_1 and use raw-Q8 dp4a projections; they do not select the
verifier's T16/raw sidecar placement.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` | unset | With no nonempty stage selector, truthy enables every draft and initialization stage; false leaves the route off. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES` | unset | A nonempty comma-separated list overrides the main toggle, even when it is unset or false. `draft` selects `project,qkv,attn_out,shared_gate_up,shared_down`; `init` selects `init_project,init_kv`; `all` selects both. Individual stage names, whitespace, hyphen aliases and duplicates are accepted. `0,false,off,no,none` disables the route; unknown stages raise. Empty/whitespace-only falls back to the main toggle. |

### GGUF target-verifier capture and F32-diagnostic family

The prefill-GDN control selects the production verifier's per-row Conv/GDN
state capture. Public generation and profile binders set
`HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN` and
`HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE` themselves. The five alternative
capture controls below change Conv/GDN execution or the scored output, not
just which intermediates are observable. They are diagnostic inputs recorded
by the forced-target and native-state probes, not user tuning controls. The
F32 flags select per-stage diagnostic arithmetic to localize divergence.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV` | off | With prefill-GDN capture on, substitutes F32 chain Conv for prefill Conv; keeps the prefill no-copy GDN row-state writer. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN` | off | Without prefill-GDN capture, substitutes regular chain GDN for the c1-exact chain writer. Rejects FP16 recurrent state because no typed variant exists. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT` | off | Without prefill-GDN capture, rounds chain GDN output to BF16 before `ssm_out`; captured state still comes from the selected chain writer. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL` | off | Without prefill-GDN capture, replays Conv/GDN into temporary state and scores its BF16 output while keeping chain-captured commit rows. Rejects FP16 recurrent state; this does not capture attention scores. |
| `HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV` | off | Without prefill-GDN capture, converts projected QKV from BF16 to F32 and uses F32 chain Conv. Prefill-GDN capture already selects F32 Conv input. |
| `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL` | unset | Replays the residual stage in F32 (supports a subset of shapes). |
| `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT` | all layers | Integer prefix length `N` in `[0, layer_count]`; F32 residual propagation covers `[0, N)` before rejoining BF16. Unset or blank means all layers; comma/range syntax is not supported. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM` | unset | Replays the post-attention norm in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_ROUTER` | unset | Replays the post-norm router read in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SELECTED_Q8` | unset | Replays the selected-Q8 post-norm read in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_SHARED_Q8` | unset | Replays the shared-Q8 post-norm read in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING` | unset | Host-dequantized F32 verifier embedding seed, only within enabled F32 residual verification. The BF16 device embedding still runs; not a GPU timing route or host-placement selector. |
| `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM` | unset | Replays the attention input norm in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS` | unset | Replays linear-attention projections in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA` | unset | Replays the SSM alpha/beta gates in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT` | unset | Replays the attention output projection in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE` | unset | Replays the MoE combine in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN` | unset | Replays the selected-expert down projection in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE` | unset | Replays the selected-expert intermediate in F32. |
| `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN` | unset | Replays the shared-expert down projection in F32. |
| `HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE` | set by the Qwen3.8 production profile binder | Production Q4 verifier rowtile admission. |
| `HIPENGINE_GGUF_ROUTER_F32W_COOP` | true; gfx1151 strict binder sets `0` | Cooperative BF16-hidden/F32-weight c1 router when registered; `0` uses separate logits/select. The production binder sets `1`. |
| `HIPENGINE_GGUF_ROUTER_F32W_PERSISTENT_COUNTER` | true; gfx1151 strict binder sets `0` | Cooperative router counter policy: `1` uses the dedicated self-resetting scratch counter; `0` uses the selected-ID alias and launch-time reset. Only used when cooperative routing runs. |

### GGUF backend-package capability seams (gfx1100)

These are retained defaults declared as backend-package capabilities; each has
an explicit env override for rollback/bisection. All are read by the adapter,
not by backend-branched runtime code.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_PRODUCTION_R28` | false | Default-off screen selecting the dense dual WMMA row32/row48 variants for specdec2 production R28 rows. |
| `HIPENGINE_GGUF_Q5_T16_ROWTILE_SINGLE_WAVE` | true | Q5_T16 single-wave rowtile owner; `0` restores the four-wave owner for rollback and bisection. |
| `HIPENGINE_GGUF_ROUNDED_NORM_FIXED5120` | true | Fixed-5120 rounded-norm residual decode sibling (rows 2-8); `0` restores the generic rounded-norm tree. |

### Gemma 4 MoE prefill routes

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GEMMA4_MOE_PREFILL` | `auto` | Pins the prefill owner for the Gemma 4 MoE expert projections. `auto` runs the exact routes only: the grouped rowbatch8 owner where a quant registers one and the selected GEMV otherwise; both measured bit-identical to the strict teacher-forced reference. `wmma` selects the compensated WMMA owners (each dequantised weight carried as an fp16 high part plus an fp16 residual, two WMMA ops per k-tile); they are 2.7x faster on a 1024-token prefill but breach the campaign's absolute `kl_max` bar on 1 of 1023 rows, so the arm stays off the default path pending the lead ruling recorded in `docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md`. `wmma_plain` is the uncompensated form, kept as the diagnostic that isolates the fp16 weight-rounding term. `grouped` and `selected` pin the exact arms. Unrecognised values fall back to `auto`. |

## Shared paged-attention decode variables

These affect both PARO and GGUF decode paths where applicable.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_PAGED_ATTN_MAX_SPLITS` | `4096` | Maximum split count used by PARO resident split-K decode config. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS`. |
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
| `HIPENGINE_FUSED_LINEAR_STATE_COMMIT` | retired | No runtime reader. GGUF selected-row/packed and PARO device-row commits use the chunked kernel when their pointer-table/layout contracts hold; per-layer copies remain for unsupported layouts, host rows, nonzero slots and missing tables. |
| `HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED` | true | Fuses the linear-attention shared down/combine epilogue; `0` restores the parallel epilogue. |
| `HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED` | true | Fuses the linear-attention shared SiLU-rotate slice; `0` restores the unfused chain. |
| `HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED` | true | Fuses the full-attention shared down/combine epilogue; `0` restores the parallel epilogue. |
| `HIPENGINE_DISABLE_PACK8_OUTPUT_TILED` | unset | Set to any non-empty value to drop rows 2/4/8 from the pack8 output-tiled GEMV route (rollback for the byte-exact AWQ output-tiled path). |
| `HIPENGINE_W4_OUTPUT_TILED_PREFILL` | true | Output-tiled W4 prefill epilogue; `0` restores the plain epilogue. |
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
| `HIPENGINE_PARO_FFN_MEGAKERNEL` | false | Opt-in single-kernel selected FFN for the verify path (tokens > 1); AR decode keeps the unfused path so the AR baseline is untouched. |
| `HIPENGINE_LAGUNA_F16_PREFILL` | retired | No longer read. Diagnostics use `f16_prefill_strategy(mode)` in `hipengine.runtime.f16_weight_linear`; target-AR exposes `--f16-prefill-strategy`. Automatic backend policy and shape fallbacks are unchanged. |
| `HIPENGINE_LAGUNA_F16_DECODE` | unset (`auto`) | Source-F16 decode strategy: `auto`, `gemv`, `onebarrier`, or `fixedk`. Automatic selection uses backend capabilities, fixed-K widths 3072/6144/9216 and 256-thread launch compatibility; other widths use the generic-K route, and other launch widths use GEMV. Explicit requests keep their primitive capability errors. The long-context profiler uses this control for paired comparisons. |
| `HIPENGINE_MAPLE_FUSE_MOE` | false | Opt-in fused Maple MoE chain (kept off pending a kernel-efficiency fix; see `docs/REFACTOR.md`). |
| `HIPENGINE_MAPLE_FUSE_QKATTN` | false | Opt-in fused Maple qknorm_rope_kv_write + attention_decode kernel (see `docs/REFACTOR.md`). |
| `HIPENGINE_MAPLE_GRAPH` | false | Opt-in whole-step Maple c1 hipGraph replay; measured only +0.47% bit-exact, so eager stays the default. |
| `HIPENGINE_EVIE_GDN_RECURRENCE` | unset | Evie GDN prefill recurrence owner: unset uses the normalized cluster8 kernel (bit-repeatable, ~9%/page faster than k2); `k2` restores the split fallback for rollback/bisection. |
| `HIPENGINE_DFLASH_DRAFTER_DENSE` | `wmma` | Dense-project DFlash drafter body: `wmma` (default after R3.4 validated exact-AR plus a per-prompt perf gain) or `naive` to revert. |
| `HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM` | `off` | Fused add+RMSNorm drafter variant; `fused`/`on` selects it, with the unfused `dflash_add_bf16` + `dflash_rmsnorm_bf16` chain as the registered fallback. |
| `HIPENGINE_MTP_DRAFT_VOCAB_CAP` | internal default | Prefix cap for the private F16 scorer used by manual native-proposer diagnostics; zero/empty selects full vocabulary and other values clamp to model vocabulary. A borrowed target W8A16 scorer, including public Generation-2 PARO MTP, always uses full vocabulary and ignores this cap. |
| `HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE` | false | Opt-in indexed (slot-local) proposer KV write path. |
| `HIPENGINE_MTP_PROPOSER_ROUTER_TOPK_FUSED` | true | Fused router top-k inside the proposer kernel (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_ROUTE_BATCHED_EXPERT` | true | Batched expert route in the proposer (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_SHARED_GATE_UP_DUAL` | true | Fused shared gate/up dual in the proposer (rollback opt-out). |
| `HIPENGINE_MTP_PROPOSER_TARGET_CONTRACT` | false before profile binding | Manual persistent-device harness selector for final target hidden, selected-row reseed and borrowed full-vocabulary W8A16 scoring. Registered strict/production harness profiles force it on; an explicit chain mode without a profile preserves the ambient selector. Public Generation-2 PARO MTP always uses the target contract independently of this variable. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE` | retired | Device-resident selected-MoE down/combine is unconditional. The host-readback rollback and its exclusive buffers are removed; stale values have no effect. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER` | retired | Device-chain `top_k == 1` always writes its ID and optional next embedding through the Q6 top-1 path. Wider top-k keeps general projection/reduction; stale values have no effect. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN` | retired | No runtime reader. Select the resident draft chain explicitly with constructor `device_chain_enabled` or benchmark `--resident-mtp-device-chain`; compatibility generation and the draft profiler request it directly. |

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
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_BATCH_MOE` | false | Native c-aware selector | Selects one rows=N selected-MoE batch transition per layer, reported as `moe_decode_path=selected_batch` with an explicit `moe_selected_batch_layers` count. Retained-bench spelling: `--batch-decode-moe-path selected_batch`; benchmark tools also accept `selected_c1` to replay archived commands, but write only the canonical environment variable. This is distinct from `selected_c1_per_row_*` fallbacks and is eligible for retained validation only with zero fallback layers and a complete correctness/scaling packet. |
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
| `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION` | `auto` (unset) | Capacity/perf policy | `streaming` reads retained INT8 directly; `oracle` uses a temporary BF16 pair. Uniform `auto` requires the minimum prompt length and either total memory at/below the configured limit or free memory at/below oracle bytes plus reserve. Tail-four `auto` keeps the mixed BF16-oracle bridge. |
| `HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS` | `229376` (224K) | Capacity policy | Minimum prompt tokens before `auto` selects the streaming direct INT8 prefill. |
| `HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB` | `26.0` | Capacity policy | Total-memory GiB threshold below which `auto` considers the streaming INT8 prefill. |
| `HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB` | `1024` | Capacity policy | Free-memory headroom added to the page-rounded BF16 oracle-pair estimate when `auto` tests pressure; not an allocation. Equality selects streaming once the prompt-length condition holds. |
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
| `HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS` | `16` | Minimum tokens before the verifier MoE uses the grouped owner; clamps to at least 2. |
| `HIPENGINE_VERIFY_GRAPH_REVALIDATE` | unset | Debug-only (`#107`): re-run the direct pass before each graph replay and compare per-row top1 plus accept payload to localize replay drift. |
| `HIPENGINE_VERIFY_DENSE_GEMV_WMMA` | false | Opt-in WMMA owner for dense verifier GEMV shapes (`1 < tokens <= 16`, features % 16 == 0). |
| `HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH` | false | Opt-in graph capture for the native-spec target-verify chain (single-request verify_chain only). |
| `HIPENGINE_PARO_NATIVE_SPEC_TARGET_COMMIT` | true | Commits target state from the captured native-spec graph; requires the graph flag. |
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

### Qwen3.8 Flash-Next serving, route, and rollback selectors

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY` | `1` | MTP verifies a candidate chain in one prefill-shaped target pass; `0` steps the target serially per candidate (rollback/bisection). |
| `HIPENGINE_QWEN4_EXP_MTP_COMPACT_PROPOSAL` | retired | No longer read. Text MTP proposals return token IDs through device-resident chaining, as mixed serving already does. The runner's explicit full-output interface remains available for diagnostics. |
| `HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET` | `1` | Runner reset clears only the K/V a request wrote; `0` restores the full-capacity clear. |
| `HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY` | `0` | `1` measures the pre-promotion fixed-256 full-attention context route against the promoted one (A/B only). |
| `HIPENGINE_QWEN4_EXP_PLE_OVERLAP` | `1` | Overlaps the pinned PLE staging upload with layer-0 work; `0` serializes it. |
| `HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS` | `random` (production UD), else `normal` | `madvise` access hint for the host-mapped PLE table; `normal` restores sequential access. |
| `HIPENGINE_QWEN4_EXP_QSA_DIRECT_SELECT` | `1` | QSA decode writes its selection directly; `0` is the rollback. |
| `HIPENGINE_QWEN4_EXP_QSA_STRIDED_GATE` | `1` | QSA gate reads the strided projection in place; `0` is the rollback. |
| `HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2_FUSED` | `0` | `1` selects the fused ordered-v2 QSA decode kernel (candidate). |
| `HIPENGINE_QWEN4_EXP_QSA_WMMA` | unset (fast/F16 route sets `1`) | QSA prefill attention on the WMMA kernel. |
| `HIPENGINE_QWEN4_EXP_ROUTED_F16` | unset (production/F16 route sets `1`) | Routed-expert gate/up and down on the wide F16 WMMA GEMMs from two rows (Q4_K) or 64 rows (Q5_K); multi-row decode batches use it too. |
| `HIPENGINE_QWEN4_EXP_ROUTED_F16_INDEXED` | `1` | The routed F16 gate/up reads token rows through the lane map; `0` builds the gathered activation copy. |
| `HIPENGINE_QWEN4_EXP_ROUTER_VARIANT` | profile-selected | Names the router-logit producer; UD production selects the F16 WMMA GEMM, with ordered DPP for small multirow widths and named rollback. Strict/Q4_K_M keep tile4. |
| `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP` | unset (F16 route sets `1`) | Warp-per-expert routed gate/up decode owner (reduction order differs). |
| `HIPENGINE_QWEN4_EXP_GDN_DECODE_ROWSPLIT` | unset (F16 route sets `1`) | Row-split GDN decode step. |
| `HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT` | retired | No longer read. Named profile registration selects the serial GDN variant; the binder publishes it to the owning runner. |
| `HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT` | profile-selected | Replaces the multi-column tile-16 GDN prefill body (for example with the row-split recurrence). |
| `HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM` | unset (F16 route sets `1`) | Multi-row FP32 indexer projections on the small-matrix F16 WMMA GEMM. |
| `HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR` | retired | No longer read. Whether the dense-wide F16 projections share one activation is a request-local decision the execution profile publishes to the runner that owns the prefill; regression tests assert this switch cannot enable it. |
| `HIPENGINE_QWEN4_EXP_GR_DECODE_FUSED` | `1` | Three-launch hyper-connection read at one row (bit-identical); `0` restores the six-launch chain. |
| `HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION` | `1` on production UD | DPP wave reduction in the hyper-connection read owners; `0` keeps the wave-scale owner. |
| `HIPENGINE_QWEN4_EXP_GR_F16_DIRECT` | unset (F16 route sets `1`) | Hyper-connection read projections launched directly on F16 activations (bit-identical). |
| `HIPENGINE_QWEN4_EXP_GR_F16_MIRROR` | `1` | Scale-only GR read also publishes half(mixed); `0` keeps per-projection conversions. |
| `HIPENGINE_QWEN4_EXP_GR_NORM_INJECT` | `1` | The scale-only norm launch also computes the inject logits (bit-identical); `0` keeps them separate. |
| `HIPENGINE_QWEN4_EXP_GR_SCALE_ONLY` | `1` | Scale-only GR read for Q8_0 up projections; `0` is the rollback. |
| `HIPENGINE_QWEN4_EXP_GR_WIDE` | unset (F16 route sets `1`) | Hyper-connection up projection on the F16 wide route. |
| `HIPENGINE_QWEN4_EXP_GR_WRITE_NORM` | `1` | Fused GR write plus next norm for chunks of at least 1,024 rows (bit-identical); `0` rolls back. |
| `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE` | profile-selected | Q8_0 dense-wide prefill route master switch. |
| `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS` | profile-selected | Comma-separated layers whose Q8_0 dense projections take the dense-wide prefill route. |
| `HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_SHORT_ROWS` | profile-selected | Admits rows 9-256 to the dense-wide owner under the short-prompt gate. |
| `HIPENGINE_QWEN4_EXP_Q8_DENSE_SPLIT_LAYERS` | profile-selected | Comma-separated layers on the split-operand dense WMMA route. |
| `HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT` | profile-selected | Names the Q8_0 expert-down prefill owner. |
| `HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION` | `1` | VALU DPP wave reduction in the Q8_0 coltile leaf; `0` restores the shuffle tree. |
| `HIPENGINE_QWEN4_EXP_Q8_DPU_REDUCTION` | unset | Recorded in host-speed probe provenance only. |
| `HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS` | unset | Comma-separated layers on the staged-F32 Q8_0 route (candidate). |
| `HIPENGINE_QWEN4_EXP_SHARED_MTP_TARGET` | `0` | `1` runs explicit MTP verification and AR peers in one shared batch target pass. Off by default because it measures slower than separate execution (`docs/REFACTOR.md`). |
| `HIPENGINE_QWEN4_EXP_T16_DOWN` | unset | T16 tile route for the Q5_1 expert down projection; a tile-cache miss falls through to the default chain. |
| `HIPENGINE_QWEN4_EXP_T16_GATE_UP` | unset | T16 tile route for the expert gate/up pair; a miss falls back to the production chain. |
| `HIPENGINE_QWEN4_EXP_T16_GATE_UP_MAX_LAYERS` | unset | Layer cap for `scripts/qwen4exp_t16_populate_tiles.py`. |
| `HIPENGINE_QWEN4_EXP_T16_RESIDENT` | unset | Materializes T16 expert tiles at load (`docs/QWEN4EXP-T16-RESIDENCY.md`). |
| `HIPENGINE_QWEN4_EXP_T16_RESIDENT_MAX_LAYERS` | unset | Layer cap for T16 residency. |
| `HIPENGINE_QWEN4_EXP_T16_RESIDENT_SLOTS` | all three roles | Comma-separated roles (`expert_gate`, `expert_up`, `expert_down`) made T16-resident. |
| `HIPENGINE_QWEN4_EXP_RISK_DIAGNOSTICS` | unset | Records iu8 risk/repair counts; synchronizes the stream, diagnostics only. |
| `HIPENGINE_QWEN4_EXP_PREBINDER_CONFLICT` | unset | Policy for a caller-set route variable that conflicts with the profile binder. |
| `HIPENGINE_QWEN4_EXP_GR_STAGED` | retired | No longer read. Regression tests ensure the rejected GR staged-F32 selector cannot be restored; the separate dense staged route remains registered. |
| `HIPENGINE_QWEN4_EXP_MOE_BACKEND_CACHE` | retired | No longer read; tests assert that setting it does not disable the backend registration cache. |
| `HIPENGINE_QWEN4_EXP_Q8_IU8_WMM` | retired | No longer read. Regression tests ensure it cannot restore the rejected generic dense Q8 IU8 WMMA route; the independent GR kernel remains registered. |

### GGUF and build selectors used by the Flash-Next routes

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_GGUF_PACK8_GEMV_WIDE` | unset | Wide-thread pack8 GEMV grid for eligible projections. |
| `HIPENGINE_GGUF_Q8_DENSE_WIDE_LONG_K` | profile-selected | Long-K dense-wide 128x128 WGP variant in place of the 256 parent. |
| `HIPENGINE_GGUF_Q8_DENSE_WIDE_WGP_MAX_ROWS` | `1024` | Upper row bound of the WGP-mode dense-wide build (`0` disables it). |
| `HIPENGINE_GGUF_SELECTED_Q4_GATE_UP_TILE8` | `1` | Tile8 ownership for the selected Q4_K T16 gate+up pair (byte-identical); `0` uses the 16-column owner. |
| `HIPENGINE_KERNEL_CENSUS` | unset | Records launch shapes and roles (never pointers or data) for kernel census reports. |
| `HIPENGINE_NO_MCUMODE_FAMILIES` | unset | Comma-separated kernel families built without CU mode; part of the build cache key. |

### Qwen4Exp decode and state

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_QWEN4_EXP_BATCHED_POSITION` | `1` | One shared 192B H2D position upload instead of 24 per-layer 8B blocking `set_position` copies per decode step (bit-exact, ~1.3% median TG gain). `0` restores per-layer copies. |
| `HIPENGINE_QWEN4_EXP_QSA_BATCHED_SELECTION` | retired | No longer read. Chunked QSA prefill uses batched score/top-k/expansion by default; `qsa_batched_selection(runner, False)` selects the per-row diagnostic fallback on that runner. |
| `HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL` | `0` (production binder sets `1`) | Master gate for the production grouped MoE prefill session (rows >= 16, supported quant shapes). |
| `HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1` | retired | No longer read. The profile publishes the Q5_1 down M1 choice on its runner; registered strict fallback kernels remain. |
| `HIPENGINE_QWEN4_EXP_Q5_1_WMMA` | retired | No longer read. Optional compact WMMA down selection is runner-owned; production grouped-MoE admission, tile-map preparation and strict fallback remain unchanged. |
| `HIPENGINE_QWEN4_EXP_Q5_1_OUT8` | retired | No longer read. The owning runner selects the out8 fallback below the expertgrid family; `q51_out8(runner, False)` retains the output1 comparison without changing higher-priority owners. |
| `HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN` | `0` | Opt-in Q5_1-style grouped WMMA down for Q8_0 experts. |
| `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN` | unset (F16 route sets `1`) | Warp-per-expert routed down decode owner (reduction order differs); the other half of the pair selected by `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP`. |
| `HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT` | `0` (production-only binding) | Weight-exact iu8-WMMA Q5_1 down with risk+repair; bit-identical to the pair2 row-publish parent in practice at multiplier 16. |
| `HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID` | `64` | Expert-grid mode for the exact grouped routes: `64` (admitted), plus `q4`/`q5` spellings at their call sites. |
| `HIPENGINE_QWEN4_EXP_EXACT_CONV_PREFILL` | retired | No longer read. Exact bulk convolution remains the prefill default; `gdn_exact_bulk_conv(runner, False)` selects the serial diagnostic fallback on the measured runner. |
| `HIPENGINE_QWEN4_EXP_LAYER_GRAPHS` | unset | Override for per-layer graph capture admission (backend capability gates the default). |
| `HIPENGINE_QWEN4_EXP_RAW_VARIANT` | `coltile8` | Raw-Q rowbatch prefill variant selector. |
| `HIPENGINE_QWEN4_EXP_RAW_ROWBATCH` | `32` | Row-batch size for raw Q4 prompt-prefill row owners; `0`/`false` disables, `1` selects 32. |
| `HIPENGINE_QWEN4_EXP_QSA_WAVE32` | retired | No longer read. H128 sparse decode and prefill fallback selection is runner-owned; `qsa_wave32(runner, False)` selects the wave64 diagnostic parent without changing H256 routes. |
| `HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE` | `0` | Opt-in ordered sparse-attention decode route for head_dim 256. |
| `HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2` | retired | No longer read. The owning target/draft runner carries ordered-v2 selection; the v1 prerequisite, geometry/count guards and separate fused diagnostic remain. |
| `HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR` | retired | No longer read. The resolved profile publishes prefill head grouping on the owning runner; scoped diagnostics may select pair or quad without changing independent batch attention selection. |
| `HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL` | `page256` under production | Selects the `strict_h256_page256_wave_rows_spans` prefill variant; otherwise `strict_h256_wave_rows_spans`. |
| `HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL` | production profile | Uses the flash sparse-attention prefill owner for dense rows with head_dim 256, gated by the `HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS` layer list. |
| `HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL` | production profile | Uses the sigmoid register GDN recurrence prefill kernel for rows>=2 and the (16, 48, 128) head geometry. |
| `HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM` | retired | No longer read. Serial GDN prefill consumes runner-local profile selection with registered wide/register fallback. |
| `HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL` | retired | No longer read. The current GDN tile16-family owner remains selected inside its capability guards; scoped `gdn_tile16_prefill(runner, False)` compares the columnwarp parent on that runner. Strict fallback kernels remain registered. |
| `HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL` / `HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS` | retired | No longer read. The profile publishes the peer layer scope on its runner; scoped diagnostics preserve explicit layer selection and the strict fallback. |
| `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL` | production profile | Uses the columnwarp GDN prefill variant, gated by the `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS` layer list. |
| `HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS` | unset | Comma-separated layer list admitting the columnwarp GDN decode variant; empty under production. |
| `HIPENGINE_QWEN4_EXP_MOE_GRAPH` | retired | No longer read. Backend capability selects the default c1 MoE graph path; explicit runner/generator `moe_graph_enabled` controls support eager/captured diagnostics. Automatic eager fallback remains. |
| `HIPENGINE_QWEN4_EXP_PLE_WARM` | `0` | Warms the PLE mmap table page cache at load for long-lived serving; short-lived runners never amortize it. |
| `HIPENGINE_QWEN4_EXP_LOAD_DROP_BEHIND` | `0` | Advise `DONTNEED` for non-PLE source ranges after their device copy succeeds. Lazy PLE and validation readers are excluded; intended for one-shot low-RSS loading, not repeated loads. |

### Qwen4Exp MoE prefill routes

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL` | retired | No longer read. Grouped MoE admission follows the explicit routed, exact-grouped and production routes; the superseded grouped Q5_K experiment selector is removed. |
| `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN` | retired | No longer read. Exact grouped Q5_1 down selection is runner-owned and remains enabled by default; scoped diagnostics retain the direct fallback. |
| `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4` / `HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL` | retired | No longer read. Exact grouped Q4 gate/up admission is runner-owned; scoped controls preserve the direct fallback and the restricted policy's exact-Q5-down prerequisite. |
| `HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN` | retired | No longer read. The profile publishes grouped Q8_0 down selection on its runner; scoped diagnostics preserve the strict comparison. |
| `HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL` | production profile | Grouped row4 GEMV prefill variant. |
| `HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL` | production profile | Selected-dual grouped rowbatch8/out4 expertgrid64 bundle prefill for Q4_K. |
| `HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL` | production profile | Selected-dual grouped pair2 prefill for Q4_K (rows>=64, in_features<=4096). |
| `HIPENGINE_QWEN4_EXP_Q4_OUT4` | retired | No longer read. The owning runner selects the out4 fallback below the exact expertgrid family; `q4_out4(runner, False)` retains the base comparison without changing higher-priority owners. |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL` | production profile | Integer-uint8 risk+repair selected Q4 gate/up prefill (rows>=2, hidden%256==0, ffn%128==0). |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS` | production layer list | Comma-separated layer list admitting the Q4 iu8 prefill route. |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT` | production profile | Exact iu8-risk Q4 gate/up route (the layer-2 Q4 route). |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_PLANES` | `3` | Plane count for the iu8-risk Q4 gate/up variant: `3` (default) or `2` (the p2 variant). |
| `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` | `4.0` | Risk multiplier margin for the iu8 risk+repair Q4 route; the default keeps a >=4x margin at the measured 1.50-1.82x operation-complete speedup. |
| `HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL` / `HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS` | retired | No longer read. Superseded Q4_K ds4-MMQ selection and workspace plumbing are removed; registered kernels remain available for explicit diagnostics. |
| `HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL` / `HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS` | retired | No longer read. Superseded Q5_1 ds4-MMQ selection and workspace plumbing are removed; active Q8 MMQ and strict fallback paths are unchanged. |
| `HIPENGINE_QWEN4_EXP_Q5_1_WAVE64` | retired | No longer read. Q5_1 selected GEMV uses the logical256 t64 owner; the wave64 variant remains registered for direct diagnostics. |
| `HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT` | `0` | Default-off exact iu8-risk+repair selected Q5_K gate/up route (the layer-2 Q5_K layer). |
| `HIPENGINE_QWEN4_EXP_Q5_K_IU8_RISK_MULT` | `16.0` | Risk multiplier for the Q5_K iu8 route; 16.0 keeps the route bit-identical to the strict parent in practice. |
| `HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL` | production profile | Selected grouped pair2 prefill for Q5_1 (rows>=64). |
| `HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE` | retired | No longer read. The owning runner carries the register-cache fallback decision, with rows>=512, K640 and folded-pair parent guards unchanged. |
| `HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH` | retired | No longer read. The owning runner carries the profile's row-publication decision; the register-cache parent and shape guards remain. |
| `HIPENGINE_QWEN4_EXP_Q51_IU8_RISK_MULT` | `16.0` | Risk multiplier for the Q5_1 iu8 route. |
| `HIPENGINE_QWEN4_EXP_Q4_TILE_M` / `HIPENGINE_QWEN4_EXP_Q4_TILE_N` | retired | No longer read. The binder publishes 16/16 on the runner for the compact F16 Q4 fallback; an unbound runner keeps M=64 at rows>=512, otherwise 16, and N=16. |
| `HIPENGINE_QWEN4_EXP_Q8_0_GROUPED` / `HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA` | retired | No longer read. Removed the legacy fixed-grid and host-loop expert-down dispatch branches; registered diagnostic kernels and the strict selected fallback remain. |
| `HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN` | retired | No longer read. The owning runner carries the mapped-down decision; admission still requires a map built by the current invocation and rows>=512. |
| `HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER` | retired | No longer read. The owning runner carries the register-down fallback decision; the rows>=512, K640 and bundle-parent guards remain. |
| `HIPENGINE_QWEN4_EXP_FUSED_COMBINE` | unset | Opt-in fused PF-4 lever-2 combine; default stays unfused after the whole-model A/B measured a loss. |
| `HIPENGINE_QWEN4_EXP_Q4_DP4A64` | `0` in UD production/fast | Profile-owned Q4 gate/up decode selector. The strict owner is selected after an observed long-context numerical failure. Setting `1` after profile binding selects the registered one-plane DP4A diagnostic; see the clearing gates in `REFACTOR.md`. |
| `HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS` | unset | Comma-separated layer list admitting the DP4A64 decode route. |
| `HIPENGINE_QWEN4_EXP_GR_IU8` | `0` | iu8 WMM prefill remap for the raw-Q rows>256 route. |
| `HIPENGINE_QWEN4_EXP_GR_IU8_DOWN` | `0` | iu8 WMM prefill remap for the down projection of the rows>256 route. |
| `HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS` | unset | Comma-separated layer list admitting the Q8 WMMA dense prefill. |
| `HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE` / `HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE` | production profile | Wave-scaled variants of the Q8 coltile prefill and the raw-Q route. |
| `HIPENGINE_QWEN4_EXP_MMQ_TOKEN64` | retired | No longer read. The profile publishes token64 selection to the owning runner; scoped diagnostics retain explicit A/B control. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL` | production profile | Admits the MMQ prefill session for Q8_0 dense projections. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES` | unset (`2` or `3`) | Plane count override for the raw Q8 MMQ policy. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK` | production profile | Uses prepacked weight sidecars in the Q8 MMQ prefill session. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4` | retired | No longer read. The owning MMQ session carries prepacked vector selection; explicit scoped diagnostics preserve the scalar comparison. |
| `HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR` | retired | No longer read. The owning MMQ session carries raw vector selection, preserving the rows>=64 parent-variant guard and token64 dependency. |
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
| `HIPENGINE_TEST_GEMMA4_MODEL_DIR` | `/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF` | Fixture directory for live Gemma MTP flag, INT8 KV generation and resident-session tests. A missing target skips these tests; set this to the installed GGUF directory on other hosts. |
| `HIPENGINE_GEMMA4_GGUF` | unset | Text GGUF path for `tests/test_live_gemma4_vision.py`; both Gemma image artifact variables must point to existing files. |
| `HIPENGINE_GEMMA4_MMPROJ` | unset | Gemma 4 vision-sidecar GGUF path for the live encoder-parity and public image-generation tests. |
| `HIPENGINE_TEST_PARO_MODEL` | unset | PARO model directory override for the live PARO tokenizer/EOS tests; falls back to the standard `/models/hipengine/...` candidates. |
| `HIPENGINE_TEST_PARO_MTP_MODEL` | `/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16` | Packed PARO model directory with BF16 MTP weights for `tests/test_live_paro_mtp_prompt_priming.py`. The test skips when the model or HIP runtime is unavailable. |
| `HIPENGINE_TEST_QWEN4EXP_GGUF` | `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL` | Model fixture for the live constrained-chunk admission test; skips when the model or HIP runtime is unavailable. |
| `HIPENGINE_TEST_REQUIRE_CACHED_BUILD` | unset | When truthy, GPU kernel tests require cached JIT builds (the test-side analogue of `HIPENGINE_REQUIRE_CACHED_BUILD`). |
| `HIPENGINE_INT8_MTP_MODEL` | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` | Dense GGUF model path for the live INT8 MTP test. |
| `HIPENGINE_DMS_MTP_MODEL` | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` | Dense GGUF model path for the live DMS+INT8 MTP parity gate. |
| `HIPENGINE_DMS_MTP_METADATA` | `~/dms-artifacts/qwen38-external-v1/sidecar/dms_metadata.json` | Trained external DMS sidecar metadata for the live DMS+INT8 MTP parity gate; produced by `scripts/qwen38_dms_train_sidecar.py`. |
| `HIPENGINE_IQ4_XS_LAYOUT_GGUF` | `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | GGUF artifact for the live IQ4_XS T16 layout test. |
| `HIPENGINE_UD_ROLE_MODEL` | `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | Published UD K_M model path for the live UD Q5/Q6 role tests. |
| `HIPENGINE_DMS_DEVICE_PAYLOADS` | retired | No reader. Backend construction selects host/device payloads explicitly; public DMS requests device storage. A missing requested device capability raises instead of silently choosing host payloads. |
| `HIPENGINE_DMS_DEVICE_TRIPWIRE` | retired | No reader. Host-staged append always checks its current overflow status. Direct-device serving append does not add status readback. |
| `HIPENGINE_MOONSHINE_CHECKPOINT` | local default | Moonshine model checkpoint override for the Moonshine GPU tests. |
| `HIPENGINE_MOONSHINE_SNAPSHOT` | local default | Moonshine HuggingFace snapshot directory override for the Moonshine GPU tests. |
| `HIPENGINE_MOONSHINE_FIXTURE_DIR` / `HIPENGINE_MOONSHINE_FIXTURES_SIX` / `HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR` | local defaults | Moonshine audio fixture directory overrides for the single- and six-fixture suites. |
| `HIPENGINE_TEST_VIBEVOICE_TTS_MODEL` | unset | Local checkpoint directory for the live VibeVoice-TTS tests. Overrides the cached `microsoft/VibeVoice-1.5B` snapshot. An explicitly named missing directory raises `FileNotFoundError`; an absent default snapshot skips checkpoint-dependent tests. |
| `VIBEVOICE_STANDALONE_GGUF` / `VIBEVOICE_STANDALONE_REPORT` | local defaults | VibeVoice standalone-encoder GGUF and report fixture paths for the VibeVoice tests. |
| `HIPENGINE_TORCH_ABSENCE_CHILD` | unset | Internal re-execution flag set by `tests/_torch_absence.py` to the pytest node id running in a fresh interpreter; the clean child detects it and runs the real test body instead of recursing. |
| `VIBEVOICE_REFERENCE_HF` / `VIBEVOICE_REFERENCE_AUDIO` | local defaults | VibeVoice HuggingFace reference snapshot and reference audio fixture paths for the encoder comparison tests. |
| `HIPENGINE_CAPACITY_MODEL` / `HIPENGINE_CAPACITY_DMS_METADATA` | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` / `/models/dms/qwen38-27b-q4km-dms-w8192-local/dms_metadata.json` | Model and DMS sidecar for the 24 GB serving-capacity gate (`tests/test_slow_qwen38_24gb_serving_capacity.py`); a missing file skips its cases. |

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
| `HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL` | retired | No runtime or profiler reader. Shared gate/up uses the dual raw-Q8 F32/F32 call unless the separate dense-Q8 dp4a stage route handles it. Stale values have no effect. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL` | unset | Captured at legacy draft construction; selects row-parallel F32 projections for both the 256-expert router and the one-output shared gate. Compatibility generation sets `1`; standalone construction defaults to generic MTP F32 linear. |
| `HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` | unset | Fused selected SiLU+down draft variant. |
| `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` | `64` | Shared Q4/Q5 selected dp4a and Q4 q8_1x2 selected/dense SiLU workgroup width: `32`, `64`, or `128`; other values fall back to `64`. Read at launch; microbench artifacts record the raw environment value. |
| `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS` | retired | No reader. Q5 uses the shared `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` scheduler; stale values have no effect. |

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
| `G4_LAYER` | `0` | Zero-based Gemma 4 layer selected by `scripts/gemma4_real_layer_dump.py` and `scripts/gemma4_real_layer_hf_check.py` for real-weight layer diagnostics; does not affect public inference. |
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

### YuE2 model, gate, and timing harnesses

| Variable | Default | Values / notes |
| --- | --- | --- |
| `YUE2_MODEL_DIR` | HF cache snapshot of `m-a-p/YuE2-3B` | Checkpoint directory for the YuE2 AR replay, session, product and timing harnesses. When unset each script resolves the pinned revision from the Hugging Face cache and fails with an explicit message if it is absent. |
| `YUE2_VAE_DIR` | HF cache snapshot of `m-a-p/YuE2-Vae` | Decoder checkpoint directory for the VAE gate, case-timing, and end-to-end harnesses. |
| `YUE2_SHOOTOUT` | `~/yue2-shootout` | Reference checkout and recorded-run root the oracle harness reads and writes. |

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


### Harness-local shell and script variables

Read only by the named harness, usually set inline when it is invoked:
`ALLOW_PIN_MISMATCH`, `FORK`, `FORK_PIN`, `UPSTREAM`, `UPSTREAM_PIN`, `WORK`,
`HIPENGINE_WORKTREE`
(Qwen3.8-27B ablation), `BLOCKING_ARM`, `ROCM_ROOT` (Qwen3.8-27B comparison),
`COOL_TARGET`, `SCRATCH`, `TEMP`, `DE11_COMPILER_VERSION_FILE`,
`DE11_COOL_TARGET`, `DE11_OUT`, `DE11_SCRATCH`, `DE11_TEMP` (DE11 paired
decode-wall harness), `ENV_PREFIX`, `ROCM_LIBS` (F16 integration GPU wrapper),
`KVARGS`, `NO_FALLBACK`, `PORT_BASE`, `PROFILE`, `RUNGS`, `WARMUPS` (long-context
ladder), `COMMON`, `PROBE` (DE1 layer bisect and gate probes), `ROCMLIB` (A/B
shell wrappers), `HIPCC` (router micro-benchmarks), `COMPILER_VERSION_FILE`
(Qwen3.5 0.8B rerank scripts), `MODEL_ROOT`, `FIXTURE_DIR`, `ROOT`, `REPS`,
`CHUNK` (script and test fixture locations and repeat counts), `DF2_FWD_BS`
(DFlash2 drafter test), `QWEN4EXP_DUMP_GEOM`, `QWEN4EXP_DUMP_PATH`
(full-attention launch probe), and `SECONDS` (the shell's own elapsed-time
counter).

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

- `HIPENGINE_MAPLE_PREFILL_GROUPED_MOE` is retired and ignored. Native
  prefill always uses grouped expert metadata and grouped gate/up/down kernels;
  the name remains only in regression tests.

- `HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER` is retired and ignored.
  MTP startup warmup and serving request deferred packed verifier scatter;
  serving commits only accepted rows. Direct verifier diagnostics can still pass
  `defer_state_scatter=False` in `verify_target_blocks_batch()` jobs.

- `HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP` is retired and ignored. Server
  startup and allocation diagnostics request MTP scratch warmup through
  `prepare_request_scratch(sampling_params=SamplingParams(speculative_mtp=True))`.
  A false or omitted sampling intent leaves MTP warmup disabled without
  changing process environment.

- `HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL` is retired and ignored. Eligible
  direct-adapter MTP batches use packed prefill with FP32 hidden rows; startup
  warmup uses bounded packed groups when requested. Singleton, over-policy-width
  and unavailable packed-hook cases keep their per-request fallback. The
  resident scheduler uses its own packed-prefill path.

- `HIPENGINE_GGUF_Q8_T16_THREADS` is retired and appears only in regression
  tests. It is ignored; backend policy and explicit launch arguments select
  thread width.

These names appear in the tree but must not be confused with env knobs:

- `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE` is retired and ignored.
  Runtime selection reads `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_BATCH_MOE`;
  the old name remains only in regression tests and historical-command validation.

- `HIPENGINE_GGUF_SPECDEC2_NGRAM_MOD`, `HIPENGINE_GGUF_SPECDEC2_NGRAM_MATCH`,
  `HIPENGINE_GGUF_SPECDEC2_NGRAM_MIN` and `HIPENGINE_GGUF_SPECDEC2_NGRAM_PROBE_MAX`
  are retired and ignored. Diagnostic adapters use explicit `ngram_enabled=True`
  and `ngram_config=NgramModConfig(...)`; default adapters keep this experiment off.

- `HIPENGINE_SPECDEC2_DEVICE_CHAIN_ORACLE` is retired and ignored. Diagnostics
  explicitly construct `Qwen35GGUFMTP2Adapter` with
  `device_chain_qualification_oracle=True` to materialize IDs and compare GPU
  acceptance with the CPU oracle. This does not certify or admit a model.

- `HIPENGINE_GGUF_Q6_PLANAR_EXACT_PREFILL` is retired and ignored. Backend
  shape/range policy selects planar-Q6 prefill retiles, with the registered
  one-wave parent handling policy misses.

- `HIPENGINE_GGUF_Q4_T16_SHARED_B_ROW64_MAX_ROWS` is retired and ignored.
  Backend shape/range policy owns row64 down-projection admission; other shapes
  and row counts retain their registered parent routes.

- `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE` is retired and ignored. Backend
  shape/range policy selects exact prefill retiles; diagnostic comparisons use
  a scoped parent selector and preserve registered parent kernels.

- `HIPENGINE_SERVER_DEFAULT_AR_READY_COHORT` is retired and ignored. Ready AR
  items use the cohort API when the generator provides it; generators without
  that API retain independent submission. Streaming and MTP routes are unchanged.

- `HIPENGINE_GGUF_SPECDEC2_MTP2_MAX_REQUESTS` is retired and ignored. Server
  physical MTP admission and adapter width selection follow backend policy;
  owner capacity and strict-profile fallback still constrain execution.

- `HIPENGINE_GGUF_SPECDEC2_STREAMING_PROMPT` is retired and ignored. Prompt
  streaming follows adapter capability and prefix-restoration checks; the
  legacy provider oracle remains available where the streaming ABI is absent.

- `HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4` is retired and ignored. The wide-Q6
  diagnostic candidate is controlled only by `target_verifier_wide_q6_shared4_session()`;
  a nested disabled scope cannot be overridden by process environment.

- `HIPENGINE_GGUF_FUSED_PACKED_STATE_TRANSFER` is retired and ignored.
  Backend capabilities select fused state transfer and packed initial-state
  gather; strict per-layer and single-slot fallback copies remain available.

- `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_ROWS6` is retired and ignored.
  Physical-row scope and native rowtile policy select grouped Q4 R6 launches;
  repeated two-wave primitives remain available.

- `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2_GROUPED_PAIR_ROWS6` is retired and ignored.
  Backend defaults, row/shape admission and registered-kernel availability
  select grouped Q4 R6 pairs; fallback primitives remain registered.

- `HIPENGINE_GGUF_Q5_T16_GROUPED_TARGET_ROWS6` is retired and ignored.
  Physical-row scope selects grouped Q5 R6 prefixes; exact wide rows preserve
  their strict tails, and ordinary decode remains outside that scope.

- `HIPENGINE_GGUF_Q6_T16_GROUPED_TARGET_ROWTILES` is retired and ignored.
  Grouped Q6 launches follow physical mixed-row scope and shape policy;
  out-of-scope inputs retain their existing R6 or ordinary decode routes.

- `HIPENGINE_GGUF_Q6_LM_HEAD_MAX_CHUNK` is retired and ignored. The backend
  chunk limit is capped by the registered Q6 LM-head primitive's row capacity;
  arbitrary widths continue to use the row-chunk planner.

- `HIPENGINE_GGUF_VERIFY_DIRECT_RESIDENT_LINEAR_STATE` is retired and ignored.
  Backend capability selects direct resident verifier indexing; single-session
  verification and incompatible resident ownership still use packed imports.

- `HIPENGINE_GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS` is retired and ignored.
  Backend policy supplies C8 exact target rows within the production profile
  and global exact-target-row scope; padded composition remains available.

- `HIPENGINE_GGUF_Q5_T16_GROUPED_ROWS8_C8` is retired and ignored. Backend
  policy and physical exact-row scope select grouped Q5 R8; other rows keep
  their existing grouped-prefix and strict-tail composition.

- `HIPENGINE_GGUF_C8_Q5_RAW_MMQ` is retired and ignored. Backend capability
  owns raw-Q5 sidecar planning; runtime request-count, sidecar and workspace
  prerequisites still determine whether the MMQ route can execute.

- `HIPENGINE_GGUF_C8_Q5_SOURCE_MMQ` is retired and ignored. Source-Q5 selection
  retains the C8 raw-MMQ capability and shape prerequisites; the numerical
  diagnostic uses a scoped selector control rather than an environment value.

- `HIPENGINE_GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS` is retired and ignored.
  Backend policy declares exact C7 target rows; separate C8 and whole-profile
  physical-row controls are unchanged.

- `HIPENGINE_GGUF_Q4_T16_GROUPED_ROWS8_C5C6` is retired and ignored. Backend
  policy still limits this grouped-R8 choice to its physical-row and active-slot
  scope, preserving grouped-R6 fallback outside that scope.

- `HIPENGINE_GGUF_NEXTN_ACCEPT_KV_WRITE_ONLY` is retired and ignored.
  Accepted-tail updates use the K/V-only transition; the complete transition
  remains callable directly as a diagnostic correctness parent.

- `HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2` is retired and ignored. Backend
  shape/row policy selects the two-wave physical owner; separate grouped-row
  controls and registered strict parent kernels are unchanged.

- `HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48` is retired and ignored. Backend
  policy and physical execution scope select the rowtile; registered parent
  kernels remain available for strict comparisons and unsupported shapes.

- `HIPENGINE_GGUF_GDN_STATE_ROWS_WAVE_REDUCE` is retired and ignored.
  Physical exact-row scope selects the wave-reduced FP32 owner; FP16 state,
  unsupported geometry and non-physical execution retain their existing owners.

- `HIPENGINE_GGUF_PREFIX_BATCHED_SUFFIX` is retired and ignored. Reused suffixes
  use the owner's batched method when eligible; one-token suffixes and owners
  without that method retain the scalar fallback.

- `HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS` is retired and ignored.
  Registered checked full-row selection follows the DPP family and its shape
  predicate; the generic DPP leaf remains the tail/unsupported-shape fallback.

- `HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH` is retired and ignored. The profile
  binds publication to the owning runner and its prefill execution scope;
  parent/candidate diagnostics change only their own runner's decision.

- `HIPENGINE_QWEN4_EXP_DENSE_WIDE_TAIL_VARIANT` is retired and ignored. The
  profile binds the tail variant to its generator and runners; prefill scopes
  expose it to launch selection. Diagnostic parent controls are runner-local.

- `HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS` is retired and ignored. The registered
  K320 specialization follows the selected DPP parent for supported full-row
  geometry; diagnostic parent comparisons use a scoped script-local override.

- `HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX` and
  `NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX` are retired and ignored in both
  GGUF and PARO. Backend shape policy and the context/split thresholds select
  grouped attention; generic and warp producers remain available for misses.

- `HIPENGINE_GGUF_DENSE_DOWN_RESIDUAL_DECODE` is retired and ignored.
  Backend/geometry/shape policy selects the c1 fused owner; registry and
  capability misses retain the primitive projection-plus-add chain.

- `HIPENGINE_GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS` is retired and ignored.
  Resident shared-slot AR uses the backend width declaration; the lifecycle
  diagnostic reads that same declaration. Ceiling planning and the explicit
  D2 cost experiment are unchanged.

- `HIPENGINE_QWEN4_EXP_AR_BATCH` is retired and ignored. Resident plain-AR
  requests use the shared batch; external runners and requests that do not fit
  its resource contract retain their c1 execution paths.

- `HIPENGINE_QWEN4_EXP_ROUTER_SELECT_CAST` is retired and ignored. The
  `router_select+cast/fp32` registry family exposes the fused default and
  `strict_unfused` comparison under the same call interface.

- `HIPENGINE_GGUF_Q4_K_DENSE_WMMA_TILE` is retired and ignored. Explicit
  primitive tile arguments remain supported; the shape-comparison harness uses
  a scoped process-local policy replacement, restored after each arm.

- `HIPENGINE_GGUF_Q6_K_DENSE_WMMA_TILE` is retired and ignored. Regression
  tests retain the name; backend wrapper defaults and explicit tile arguments
  select Q6 WMMA geometry.

- `LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER` is backend package
  metadata. The output-gate diagnostic temporarily overrides that attribute;
  it is not read from the process environment.

- `HIPENGINE_GGUF_Q8_DENSE_WIDE_F16_ACT` is retired. Profile and session
  selection own activation staging; the old environment value has no effect.
- `HIPENGINE_GGUF_INT8_KV_DECODE_GRAPH` is retired. Capture APIs admit
  implemented KV layouts; automatic capture uses backend-declared replay
  floors instead. Historical harness artifacts and regression tests may name it.
- `HIPENGINE_FUSED_RMSNORM_ROTATE` is retired and ignored.
  Verifier linear attention normalizes inputs then rotates them. The standalone
  fused RMSNorm/rotate primitive remains registered for direct comparisons.
- `HIPENGINE_SHARED_EXPERT_FUSED_ROTATE` is retired and ignored.
  Shared experts use separate input rotation and projection. The unused keyed
  barrier scratch, host counters and graph-capture toggle are removed; direct
  staged primitive APIs and independent attention fusion remain available.
- `HIPENGINE_SELECTED_MOE_STAGED_ROTATE` is retired and ignored.
  Selected gate/up execution uses separate rotation and projection in Python
  and native C dispatch. Selected-only barrier fields are removed; staged
  primitives stay registered.
- `HIPENGINE_MOE_FUSED_ROTATE` is retired and ignored.
  Selected MoE uses separate rotation and projection by default; this old
  value no longer blocks native C dispatch. The fused primitive remains
  registered for direct diagnostics.
- `HIPENGINE_GDN_CHAIN_TLOOP_VTILE` is retired and ignored.
  Chain GDN uses tile-4 for value widths divisible by four and the scalar-column
  schedule otherwise. The serial-c1 exact route is unchanged.
- `HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER` is retired and ignored.
  The persistent PARO diagnostic harness synchronizes verifier commit before
  proposer repair/update on the default stream. Runtime stream-capable APIs
  are unchanged.
- `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD` is retired and ignored.
  PARO target verification projects logits then runs row argmax. The fused
  primitive remains available to the native MTP proposer and direct parity
  diagnostics.
- `HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION` is retired and ignored.
  Ordinary PARO verifier commits set position/context separately. Native
  graph-owned commits still explicitly request packed cursor updates; their
  primitive and graph binding are unchanged.
- `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` is retired and ignored.
  PARO full-attention preparation always splits Q/Gate then casts the key;
  the fused primitive remains available for direct parity tests.
- `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY` is retired and ignored.
  MTP diagnostic scripts always keep verifier-shaped scratch between cycles.
  The resident `canonicalize_after` API is unchanged; ordinary decode still
  restores one-row scratch and invalidates graphs before replacing workspace.
- `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP` is retired and ignored.
  PARO verifier scratch hits always check the owner's generation, scratch
  type and row layout. Prefill restore and graph invalidation clear both
  caches and advance the generation before workspace reuse.
- `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED` is retired and ignored.
  PARO verifier MLP scratch always follows the live grouped-MoE threshold;
  dense models reserve dense scratch, and cache keys still include policy.
- `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE` is retired and ignored.
  PARO resident linear-state, per-slot KV and all-slot KV views are always
  cached. Slot checks still run on hits; reset, capacity changes and KV
  rebuilds still invalidate the non-owning views.
- `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE` is retired and ignored.
  Each PARO decode state always memoizes immutable weight tensor views by
  the caller's exact name. Missing names still raise the normalized key error;
  this cache does not own device allocations.
- `HIPENGINE_VERIFY_SCRATCH_CACHE` is retired and ignored.
  PARO verifier scratch objects are always cached by layer, row count and
  MLP policy. Hits still use generation stamps or explicit workspace checks;
  prefill restore and graph invalidation still clear the caches.
- `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` is retired and ignored.
  PARO and GGUF pointer-table linear-state commits always use the chunked
  copy primitive, including native selected-state graph commits. The
  unchunked registry primitive and required per-layer copy fallbacks remain.
- `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA` is retired and ignored.
  Verify-chain batches always pack the dynamic token, position and context
  updates into one upload followed by device-side unpacking. Tree metadata
  keeps its separate uploads and ancestor/cache-slot records.
- `HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD` is retired and ignored.
  The PARO verifier reads one packed int32 accept summary per batch; the
  packed kernels still populate separate device buffers used for commit.
- `HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE` is retired and ignored.
  Scalar advances and bulk prompt priming write K/V directly into cache slots.
  Indexed writes remain independently selected; temporary K/V buffers and
  their device-to-device copies have been removed.
- `HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT` is retired and ignored.
  The scalar expert loop always overwrites its FP32 accumulator on route 0,
  then adds subsequent routes. Batched experts initialize their own output.
- `HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION` is retired and ignored.
  Native proposer scalar advances always upload the shared 16-byte token,
  position and cache-slot record once on the caller's stream.
- `HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS` is retired and ignored. The
  persistent-device harness skips unused proposer metadata, update-only
  scoring and the final redundant snapshot; candidate token reads and
  rollback-prefix snapshots remain.
- `HIPENGINE_SELECTED_MOE_DOWN_STAGED` is retired and ignored. Python and
  C-dispatched selected-MoE down execution use the SiLU/rotate then projection
  chain. The staged primitive remains independently registered for diagnostics.
- `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED` is retired and ignored.
  Multi-row linear-attention output projection fuses the FP32-to-FP16 cast
  with rotation; single-row and explicit exact-row paths keep their
  cast-then-rotate chain. Regression tests retain the old name.

- `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL` and
  `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_SITES` are retired and ignored.
  Shared gate/up uses the promoted split-output route for supported row
  counts; other shapes keep their existing prefill fallback. Regression
  tests retain the names.

- `HIPENGINE_SHARED_PREFILL_SILU_ROTATE_FUSED` is retired and ignored.
  Shared-expert FP16 prefill uses the pair-rotate primitive; small batches keep
  their existing packed SiLU/rotate route. Registered separate SiLU and
  rotation primitives remain numerical references. Tests retain the old name.

- `HIPENGINE_YUE2_NAR_ATTENTION` is retired and appears only in regression
  tests. Attention selection uses kernel geometry; setting it has no effect.

- `-D` preprocessor macros inside kernel and micro-benchmark builds
  (`HIPENGINE_DENSE_WIDE_EXPORT`, `HIPENGINE_DENSE_WIDE_EXPORT_BF16`,
  `HIPENGINE_QWEN4_GDN_DPP`, `HIPENGINE_QWEN4_GDN_MULTI_COLUMN`, `HIPENGINE_LOCAL_SIZE_X`, `HIPENGINE_ROW_TILE`,
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
  `HIPENGINE`, `HIPENGINE_QWEN4_EXP_`, `HIPENGINE_QWEN4_EXP_GDN_`,
  `HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_`, `HIPENGINE_GGUF_PACK8_`, `HIPENGINE_GGUF_`, `HIPENGINE_GGUF_VERIFY_`, `HIPENGINE_QWEN35_BATCH_`,
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
- Argument *metavars* that name an artifact slot rather than a variable:
  `HIPENGINE_JSON` (`scripts/yue2_ar_matched_timing.py`,
  `scripts/yue2_stage_matched_timing.py`).
- Shell-local variables inside the `scripts/*_ab.sh` A/B harnesses, read as
  `$NAME` and assigned in the same script, not process env vars: `BENCH`,
  `FILE`, `FILES`.
- Placeholder names used in doc/test examples and generic metavars
  (`HIPENGINE_FOO`, `HIPENGINE_ZZZ`, `HIPENGINE_AAA`, `HIPENGINE_EXAMPLE`,
  `HIPENGINE_SOMETHING_ELSE`, `HIPENGINE_STALE_FLAG`, `HIPENGINE_ONE`,
  `HIPENGINE_TWO`, `HIPENGINE_PREBUILD`, `HIPENGINE_DUP`, `HIPENGINE_KEY`,
  `HIPENGINE_ROUTE`, `PROFILE_ISOLATION_TEST_UNRELATED`, `HIPENGINE_UNRELATED`,
  `HIPENGINE_TEST_RAW_ENV_MEMO`, `HIPENGINE_ACTIVE_TASK_CONTENT__` (a chat
  template marker), `HIPENGINE_ENV` (a conftest identifier fragment), and the
  diagnostic flag constants `HIPENGINE_DIAGNOSTIC_MMQ_SHAPE_OFF` /
  `HIPENGINE_DIAGNOSTIC_Q8_DOWN_LAYER_OFF` that the Q8 localize harness sets).
- Removed or never-shipped flags that tests assert are absent or ineffective:
  `HIPENGINE_GGUF_Q4K_ROWTILE` (rowtiles are default-on; scoped comparisons
  use `q4k_rowtile_session(False)`), `HIPENGINE_GGUF_AR_STREAM_PREFILL`,
  `HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV` (contiguous layout uses package
  policy; gapped gather has its own switch; token/byte resource caps remain),
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
