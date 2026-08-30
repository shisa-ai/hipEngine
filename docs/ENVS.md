# Environment variables

Last updated: 2026-08-30

This is the user-facing env-var reference for hipEngine. Most users should not
need any hipEngine-specific env vars for normal `LLM.generate()` use; prefer
Python/CLI arguments when available. Use env vars mainly for backend forcing,
ROCm/TheRock process setup, cached-build profiling, and explicitly documented
benchmark or diagnostic profiles.

Boolean values generally accept `1/true/yes/on` as true and `0/false/no/off` as
false unless the variable says otherwise.

## Recommended profiles

### Normal local use

- No hipEngine env vars required when `backend="auto"` detects a native target.
- `LLM(model)` and `hipengine serve --model ...` resolve the model plugin's
  quantization. Supported GGUF models also select decode-repack and the public
  WMMA-prefill/GEMV-decode session profile.
- Server metadata reports the concrete backend, quant, and explicit execution
  profile manifest hash after model load. Omit `HIPENGINE_EXECUTION_PROFILE`
  during migration to preserve incumbent package behavior; set it only when the
  selected model/backend/quant has a registered fail-closed plan.
- Set `HIPENGINE_BACKEND=hip_gfx1100` or `HIPENGINE_BACKEND=hip_gfx1151` only
  when auto-detection falls back or you are forcing a nearby target explicitly.
- Leave diagnostic fusion/tuning knobs unset.
- Leave `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` unset.

### TheRock ROCm process setup

For stable ROCm 10 installation/upgrade/rollback on gfx1151 and the separately
retained W7900 ROCm 7.13 stack, see [`THEROCK.md`](THEROCK.md). This section only
shows the current gfx1151 clean-process wrapper.

TheRock installs ROCm pieces inside the Python environment. Build the process
environment around that prefix rather than mixing its libraries with
`/opt/rocm`:

```bash
ENV_PREFIX=/home/lhl/miniforge3
PY=$ENV_PREFIX/bin/python
ROOT=$("$PY" -m rocm_sdk path --root)
SITE=$ENV_PREFIX/lib/python3.13/site-packages
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

## Core runtime and build variables

| Variable | Owner | Default | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_BACKEND` | Backend selection | unset / `auto` | Force a backend key such as `hip_gfx1100` or `hip_gfx1151`; otherwise auto-detects supported HIP arches and falls back to `cpu_reference` with a warning. |
| `HIPENGINE_EXECUTION_PROFILE` | Execution-profile plan selection | unset (migration default) | `strict`, `production`, or `batch_invariant`; equivalent to Python `execution_profile=` and server `--execution-profile`. Explicit selectors resolve only registered model/backend/quant plans, verify selected/fallback kernel keys, and fail closed. An omitted selector temporarily preserves incumbent package behavior and is not a fourth profile. |
| `HIPENGINE_HIP_ARCH` | HIP JIT build | unset | Force native HIP offload arch in build cache keys, e.g. `gfx1100` or `gfx1151`. The backend helper sets this temporarily when needed. |
| `GPU_MAX_HW_QUEUES` | HIP runtime / gfx1151 branch concurrency | gfx1151: `2`; otherwise unset (ROCm default `4`) | Must be applied before `libamdhip64` loads. hipEngine sets `2` only when all recognized visible HIP arches map to gfx1151 and the user has not provided a value. Explicit values always win; use `1` for the prior single-queue rollback or `4` for the ROCm-default scheduler diagnostic. The exact Laguna shared/routed MoE gate admits two queues at short contexts, but neither one nor two queues is a repeated-128K lifecycle guarantee. |
| `HSA_SCRATCH_SINGLE_LIMIT` | HIP runtime / gfx1100 scratch reserve | gfx1100: `8388608` (8 MiB); otherwise ROCr default | Must be applied before `libamdhip64` loads. On ROCr 7.2.4 the upstream default is 140 MiB, is reserved per process/GPU, and dispatches above it use a use-once scheme. hipEngine lowers only the homogeneous gfx1100 process default to 8 MiB; this retains full-engine behavior with the 300-MiB AOTriton use-once path while removing 132 MiB of unused reserve. Explicit user values always win; use `146800640` to reproduce the upstream default. Mixed recognized architectures receive no backend-local default. |
| `HIPENGINE_HIP_OFFLOAD_ARCH` | HIP JIT build | unset | Alias-style fallback for `HIPENGINE_HIP_ARCH`. |
| `HIPENGINE_ROCM_DEVICE_LIB_PATH` | HIP JIT build | unset | Adds `--rocm-device-lib-path=<path>` to `hipcc`. Falls back to standard `HIP_DEVICE_LIB_PATH` if unset. Useful for TheRock. |
| `HIPENGINE_COMPILER_VERSION_TEXT` | HIP JIT cache | unset | Literal compiler-version text for cache keys; avoids probing `<compiler> --version`. |
| `HIPENGINE_COMPILER_VERSION_FILE` | HIP JIT cache | unset | Reads compiler-version text from a file. Recommended for cached benchmarks/profiling. |
| `HIPENGINE_HIPCC_VERSION_TEXT` / `HIPENGINE_HIPCC_VERSION_FILE` | HIP JIT cache | unset | Compiler-specific override for `hipcc`; takes precedence over the generic compiler-version vars. The same pattern applies to other compiler basenames. |
| `HIPENGINE_AOTRITON_LIB` | AOTriton discovery | unset | Explicit `libaotriton_v2.so` override. The matching `include/` and `aotriton.images/` trees must be in the standard release layout. |
| `HIPENGINE_AOTRITON_HOME` | AOTriton discovery | unset | Explicit cache root containing `<version>/lib/libaotriton_v2.so`. Missing explicit roots fail loudly instead of falling back silently. |
| `HIPENGINE_API_KEY` | OpenAI-compatible server | unset | Optional bearer token used by `hipengine serve` when `--api-key` is omitted. |
| `HIPENGINE_GENERATION_BATCH_WINDOW_MS` | OpenAI-compatible server | `0` | Opt-in cold-path coalescing delay for compatible HTTP requests. Default `0` adds no intentional delay; same-event-loop-turn requests may still share the batcher worker, while positive values are for explicit coalescer experiments. |
| `HIPENGINE_STREAM_QUEUE_MAX_CHUNKS` | OpenAI-compatible server | `16` | Bounded token-event queue per streaming HTTP request; must be at least 2. The resident loop independently keeps a bounded 64-event scheduling buffer per subscription to absorb transient cross-thread bursts. A client that remains slow enough to overflow it is cancelled with `budget_pressure=client_backpressure` instead of stalling unrelated rows. Equivalent CLI flag is `--stream-queue-max-chunks`. |
| `HIPENGINE_SHUTDOWN_GRACE_SECONDS` | OpenAI-compatible server | `5.0` | Grace period for queued/active generation to drain during server shutdown. At expiry, request cancellation tokens are tripped, producers are cancelled, and the long-lived model runner is closed. Equivalent CLI flag is `--shutdown-grace-seconds`. |
| `HIPENGINE_SPECULATIVE_MTP_SERVING` | OpenAI-compatible server | `enabled` | `off`, `opt_in`, `auto`, or `enabled`. `opt_in` exposes the current `llama-compat` route only when a request sends `speculative_mtp=true`. The corrected category gate proves that route is not true-AR exact, so `auto` records the realized group/horizon and selects exact/default AR until a separate exact MTP hook is admitted. `enabled` (the default) routes compatible requests through exact dense MTP automatically for dense Qwen models whose NextN route is serial-exact against AR, and falls back to plain AR otherwise. |
| `HIPENGINE_SPECULATIVE_MTP_THINKING` | OpenAI-compatible server | `hint` | `hint` or `hard`. `hint` (default) keeps thinking hints in the rendered prompt but relaxes host-sampler thinking-budget enforcement (soft-close bias, EOS suppression, hard-close forcing) so `reasoning_effort` requests can use the exact raw-argmax MTP route. `hard` keeps full host-sampler enforcement and treats the thinking budget as a hard MTP blocker, falling back to plain AR for thinking requests. A request can override with `speculative_mtp: { "thinking": "hint" \| "hard" }`. |
| `HIPENGINE_GGUF_MTP_VERIFY_MODE` | GGUF dense MTP server | `native` | `native` or `serial_exact`. `native` (default) is the fast llama.cpp-style native row-attention / GPU-accept verify path validated on the dense MTP suites (roughly 1.5-1.7x decode speedup over AR with rare sub-token-level argmax differences). `serial_exact` re-runs exact c=1 AR per candidate row (token-exact against AR but cannot beat AR decode speed). |
| `HIPENGINE_GGUF_MTP_CANDIDATE_BUDGET` | GGUF dense MTP server | `3` | Candidate draft-token budget (1-4) for the dense MTP serving path. Default 3 matches the validated sweet spot; budget 4 can regress. |
| `HIPENGINE_SPECULATIVE_PROVIDER` | OpenAI-compatible server / generic draft-model provider | unset | Explicit provider registry key, such as `dflash`. Requires `HIPENGINE_DRAFT_MODEL`; ordinary requests remain AR and a request must send `speculative=true` (or the explicit object form). Equivalent CLI flag is `--speculative-provider`. |
| `HIPENGINE_DRAFT_MODEL` | OpenAI-compatible server / generic draft-model provider | unset | Local pinned draft-model path paired with `HIPENGINE_SPECULATIVE_PROVIDER`. The Laguna DFlash owner validates its revision and content-addressed safetensors SHA before allocation. Equivalent CLI flag is `--draft-model`. |
| `HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET` | OpenAI-compatible server / generic draft-model provider | `4` | Fixed positive candidate budget for the configured owner. Laguna currently admits only B4 and rejects mismatched request objects. Equivalent CLI flag is `--speculative-candidate-budget`. |
| `HIPENGINE_MAX_QUEUED_REQUESTS` | OpenAI-compatible server | unset | Optional generation queue cap. When set and the server batcher queue is full, new generation requests fail with HTTP 429 `engine_busy` and `Retry-After: 1`; equivalent CLI flag is `--max-queued-requests`. |
| `HIPENGINE_METRICS` | OpenAI-compatible server | `off` | Metrics endpoint mode used by `hipengine serve --metrics`: `off` or `prometheus`. When `prometheus`, `/metrics` exposes additive HTTP counters plus resident pending/admitted/active occupancy, work-class totals and policy, bounded queue/TTFT/ITL/service/completion summaries, real KV byte/page/ref/pin/grow/shrink/failure stats, graph capture/replay/invalidation counts by bucket, and route/fallback counters when the loaded generator provides a live-loop snapshot. |
| `HIPENGINE_PREFIX_CACHE` | OpenAI-compatible server / KV sharing | `off` | Prefix-cache mode used by `hipengine serve --prefix-cache`: `off` or `radix`. `radix` enables the scoped fail-closed GGUF path: a greedy request with a non-empty suffix may reuse an exact positive 256-token boundary from either an active-current source or a bounded cache-owned completed-source snapshot. The runner clones hybrid Conv/GDN state, shares page-aligned KV with a private COW suffix, preserves the latest aligned boundary across an unaligned tail, and otherwise reports an explicit private-prefill fallback. Sampling, short prompts, and exact-full-prompt boundaries do not reuse. `/ready` reports bounded snapshot/page/byte ownership. Default stays `off` until the multi-turn economics and lifecycle/pressure gates in `docs/CONCURRENCY.md` pass. |
| `HIPENGINE_REPLAY_DIR` | OpenAI-compatible server diagnostics | unset | Opt-in directory for finite JSON failed-request replay artifacts. Disabled by default for sensitive deployments; equivalent CLI flag is `--replay-dir`. |
| `HIPENGINE_REPLAY_REDACTION` | OpenAI-compatible server diagnostics | `hash` | Replay artifact string redaction mode: `hash` replaces strings with SHA-256/length metadata, while `none` stores raw strings for explicit local debugging only. Equivalent CLI flag is `--replay-redaction`. |

Removed historical AOTriton knobs (`HIPENGINE_AOTRITON_SOURCE_ROOT` and
`HIPENGINE_AOTRITON_RUNTIME_ROOT`) are no longer read by the runtime.

## GGUF variables

| Variable | Default | Classification | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_GGUF_DECODE_REPACK` | true | Retained release default with rollback opt-out | Materializes resident T16 decode layouts on load. The public Qwen3.6 GGUF path uses this accepted layout despite its load-time and resident-memory cost because raw decode is substantially slower. Set false only for diagnostics or memory comparisons. |
| `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` | backend-scoped (gfx1151 physical rows >=4; otherwise false) | Retained c4/c8 rollback and broad diagnostic | Unset uses the backend-qualified width floor: gfx1151 routes Q8T16 singleton/pair/triple decode through rowtile bodies from physical c4, while c2 and gfx1100 remain direct. Set `0` to disable all-projection rowtiling at every width (the separately qualified c8 pair policy still resolves independently); set `1` only to reproduce the broad rows>1 diagnostic, which loses `1.795%` at c2. |
| `HIPENGINE_GGUF_WMMA_PREFILL` | false | Low-level performance selector | Process-wide default for low-level GGUF sessions. The public generator passes `use_wmma_prefill=True`; benchmark CLI/session arguments remain explicit for artifact provenance. |
| `HIPENGINE_GGUF_GEMV_DECODE` | false | Low-level performance selector | Process-wide default for low-level GGUF sessions. The public generator passes `use_gemv_decode=True`. For qwen35moe, effective use is safety-gated unless decode-repack is active or the unsafe override is set. |
| `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` | false | Unsafe diagnostic | Bypasses qwen35moe GGUF fast-path safety. Do not set for normal use or promoted correctness claims. |
| `HIPENGINE_GGUF_AOTRITON_PREFILL` | `v3` | Attention implementation selector | `v3`, `v2`, or `auto`/`v2-if-safe`. `v2` is rejected for chunked suffix prefill because it has the wrong causal-mask semantics there. |
| `HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` | `1024` | Decode threshold | Context length where GGUF full-attention decode uses split/paged decode; `0` disables. Compatibility alias: `NANOVLLM_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT`. |
| `HIPENGINE_GGUF_GDN_PREFILL_MODE` | `auto` | Correctness/performance diagnostic | Selects `auto`, `exact`, `fused`, `chain`, or the named `chain_k2`, `chain_peer_wave32`, `chain_compact_peer_wave32`, `chain_peer_cluster8`, `chain_tile64`, `chain_tile32`, `chain_wave32`, `chain_wave32_tree`, `chain_lds64`, `chain_lds32`, `chain_lds32_direct`, and `chain_lds32_direct_nonvolatile` GGUF GDN prompt-prefill routes. `chain` is the raw-Q/K-plus-scale exact split fallback; `chain_lds32_direct_nonvolatile` is the promoted GPF-2E compact-scale/direct-`conv_out` route on gfx1151, while volatile direct and materialized `chain_lds32` remain rollback/bisection controls. The tile/wave/LDS64 routes are rejected diagnostics. `auto` uses backend-package policy: byte-exact direct LDS32 on gfx1151 and the bit-equivalent compact peer-wave route on gfx1100, with fused correctness fallback if the preferred route is unavailable. `chain_peer_wave32` retains the prior per-V-head materialization as rollback; `chain_compact_peer_wave32` stores normalized Q/K once per K head. Every explicit selection overrides backend policy and fails closed if its complete implementation is unavailable; invalid values are errors. |
| `HIPENGINE_GGUF_PREFILL_DEVICE_METADATA` | backend-scoped (`1` through 4K on gfx1100/gfx1151; otherwise `0`) | LCP-M2 rollback/diagnostic | Selects the stream-ordered `prepare_prefill_chunk_metadata` kernel (`1`) or six synchronous host-prepared metadata copies (`0`). With the variable unset, gfx1100 and gfx1151 select the exact device path only through 4,096 prompt tokens; longer prompts retain synchronous metadata. Explicit values override the ceiling. Keep `0` for rollback and never force `1` at long context in production: the explicit gfx1151 128K one-queue lifecycle gate still entered the low-power no-progress state. |
| `HIPENGINE_GGUF_PREFILL_ROUTER_SELECT_THREADS` | gfx1100/gfx1151: `128`; otherwise `512` | LCP-4B rollback/diagnostic | Overrides the bulk-prefill `qwen35_router_select_kernel` workgroup with `64`, `128`, `256`, or `512`. Both gfx1100 and gfx1151 default to their independently full-model-exact 128-thread launch; decode retains its independent 256-thread geometry. Use `512` for rollback. Do not use `64` for production: it is fastest in a primitive screen but failed the gfx1151 4K full-model exact-state gate. |
| `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE` | `auto` | GPF-3A rollback/performance selector | `auto`, `baseline`, or `shared_x`. `auto` reads backend-package capability: gfx1100 and gfx1151 select their independently clean-gated byte-exact `shared_x` schedules. Explicit selections fail closed when their registry variant is unavailable; `shared_x` is mutually exclusive with the DS4 selected-prefill diagnostic. Leave unset for the production backend policy; use `baseline` only for rollback/A-B. |
| `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` | `1025` | Prefill threshold | Minimum rows for GGUF GDN recurrent-segments prefill routing; invalid values fall back to the default, values below 1 clamp to 1. |
| `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` | false | Capacity diagnostic | Offloads the raw Q8_0 token embedding from device residency and serves exact Q8_0→BF16 embedding rows from host. This can make Q4_K_M 128K fit on 24 GiB, but disables GGUF HIP decode graph replay and is not a promoted performance path. |
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
| `HIPENGINE_GGUF_FP16_RECURRENT_STATE` | backend/model scoped (true for gfx1151 `mostly_q4_k_s`; false otherwise) | Retained default with strict-storage rollback opt-out | Stores Qwen3.8 Q4_K_S GDN recurrent state as FP16 with FP32 accumulation. Complete packed numerical/isolation and engine/serving A/B are non-regressive and faster. Set `0`/`false`/`off` to restore FP32 state; set `1` only to force the validated leaf outside the default scope for diagnostics. Chain-journal and MTP remain fail-closed. |
| `HIPENGINE_GGUF_AR_STREAM_DECODE` | true | Retained fallback with rollback opt-out | Enables per-slot stream decode fallback and c>4 packed-decode chunk streams. Set false only for bisection. |
| `HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL` | true | Retained default with rollback opt-out | Enables the GGUF MTP server packed prompt-prefill opener for eligible c=2/c=4 serving batches, returning FP32 prompt hidden rows for MTP catch-up. Set false for bisection. The c=8 first wave still uses the serial opener because the packed prefill path keeps the four-slot safety cap; the trailing c=2 wave uses packed prefill. |
| `HIPENGINE_GGUF_Q4_K_SELECTED_WMMA_TILE_M` / `_TILE_N` | `32` / `16` | Kernel R&D | Q4_K selected WMMA tile override. Allowed tile pairs are validated by the build helper. |
| `HIPENGINE_GGUF_Q5_K_SELECTED_WMMA_TILE_M` / `_TILE_N` | `16` / `16` | Kernel R&D | Q5_K selected WMMA tile override. |
| `HIPENGINE_GGUF_Q6_K_SELECTED_WMMA_TILE_M` / `_TILE_N` | `16` / `16` | Kernel R&D | Q6_K selected WMMA tile override. |

## Shared paged-attention decode variables

These affect both PARO and GGUF decode paths where applicable.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_PAGED_ATTN_MAX_SPLITS` | `4096` | Maximum split count used by PARO resident split-K decode config. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_MAX_SPLITS`. |
| `HIPENGINE_PAGED_ATTN_GQA_GROUPED_CTX` | true | Enables grouped-GQA split decode for Qwen3.5/Qwen3.6 GQA shapes. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_GQA_GROUPED_CTX`. |
| `HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_SPLITS` | `64` | Minimum split count that selects grouped-GQA split decode. |
| `HIPENGINE_PAGED_ATTN_GQA_GROUPED_MIN_CONTEXT` | `4096` | Minimum context length that selects grouped-GQA split decode. |
| `HIPENGINE_PAGED_ATTN_WARP_SPLIT_CTX` | true | Enables warp-split GQA fallback where grouped-GQA is not selected. Compatibility alias: `NANOVLLM_AMD_PAGED_ATTN_WARP_SPLIT_CTX`. |

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
| `HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS` | `30.0` | `--kv-pool-idle-grace-seconds` | Seconds before fully-free, graph-unpinned tail chunks are eligible to shrink; must be ≥ 0. |
| `HIPENGINE_MAX_PENDING_REQUESTS` | unset | `--max-pending-requests` | Optional pending request queue cap for the resident scheduler; must be > 0 when set. |

The table lists generic engine defaults. For each unset scheduler knob, the
registered gfx1151 Qwen GGUF `gguf_q4_k_m` generator refines the policy/chunk
pair to the F4-retained `fair:256`; explicit env values always win independently.
Other GGUF quants, gfx1100, and PARO retain their prior defaults until they pass
independent workload gates.

## PARO variables

Prompts shorter than `linear_conv_kernel_dim` use token-serial c1 prefill in
the public generator. Longer prompts use native prefill; neither route requires
an env variable.

| Variable | Default | Classification | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_PARO_MARLIN_K_REPLACE` | true | Retained default | Uses the retained PARO Marlin-K replacement path during loading. Set false only for bisection. |
| `HIPENGINE_QWEN35_LM_HEAD_THREADS` | `128` | Runtime tuning | Valid values: `128`, `256`, `512`. |
| `HIPENGINE_QWEN35_NATIVE_SAMPLER` | true | Retained default with rollback opt-out | Enables the scoped PARO native GPU sampler for supported c=1 and scheduler-owned serial per-slot c>N sampled requests (`top_k=0`, `1<=top_k<=64`, or exact `top_p`/`min_p` with `top_k=0`). Set `0`/`false`/`off` to force host sampling for rollback. Full-vocab `top_logprobs` with `top_k=0` and bounded `top_logprobs <= top_k <= 64` stay native; true batched c>N, GGUF, bounded `top_logprobs > top_k`, and unsupported processor/filter combinations fall back to host sampling. |
| `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` | false | Experimental diagnostic | Enables the guarded Qwen/PARO `step_batch_native` c>N decode path. Leave unset for normal use; retained throughput claims require generated-token equality and currently keep this path ineligible. |
| `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS` | false | Experimental diagnostic | Selects the evidence-backed PARO c>N attention/MoE/projection/sampler repair routes. It does not activate native decode by itself. When native decode is active, unsupported live widths use the exact partition/serial planner below. |
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
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS` | false | Diagnostic fallback | Replays linear-attention QKV/Z/A/B projections with token-1 kernels per row, then copies planar rows back into batch scratch. Hidden-bisect equivalent: `--batch-decode-linear-projection-path selected_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ` | false | Correctness diagnostic | Replays only linear-attention QKV/Z projections with token-1 kernels per row while leaving A/B on the native batch path. Hidden-bisect equivalent: `--batch-decode-linear-projection-path selected_qkv_z`. Correctness-green for c<=8 in the retained bench, but still non-retained until native projection dispatch is accepted. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB` | false | Diagnostic fallback | Replays only linear-attention A/B projections with token-1 kernels per row while leaving QKV/Z on the selected batch path. Hidden-bisect equivalents: `--batch-decode-linear-projection-path selected_ab` or `batch_gemv_selected_ab`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS` | false | Diagnostic fallback | Uses row-aware GEMV kernels for c>N linear-attention QKV/Z projections while keeping native A/B projection and segmented state. Hidden-bisect equivalent: `--batch-decode-linear-projection-path batch_gemv`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE` | false | Diagnostic fallback | Replays linear-attention conv/GDN/recurrent state updates with token-1 kernels over slot-local state. Hidden-bisect equivalent: `--batch-decode-linear-state-path selected_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT` | `auto` | Diagnostic fallback | Linear-attention output projection override: `auto`, `batch`, `batch_gemv`, or `selected_c1`. `auto` follows selected-c1 state replay; `batch_gemv` bypasses the row>1 AWQ prefill projection kernel while staying non-retained. Hidden-bisect equivalent: `--batch-decode-linear-output-path ...`. |
| `HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE` | true when experimental decode is enabled | Diagnostic selector | Set `0` to force the existing per-row full-attention fallback in hidden-bisect/native-batch probes. Non-retained fallback metadata records `full_attention_decode_path=per_row_*`. |
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
| `HIPENGINE_PARO_ROUTER_TOPK_COOP` | false | Rejected/diagnostic | Leave unset unless reproducing router-coop probes. |
| `HIPENGINE_LINEAR_GDN_PREFILL_ROTATE_FUSED` | false | Rejected/diagnostic | Leave unset unless reproducing fusion probes. |
| `HIPENGINE_PREFILL_ROUTER_SHARED_GATE_SIGMOID_FUSED` | false | Rejected/diagnostic | Leave unset unless reproducing fusion probes. |

PARO prefill workspace-overlap minimization is now a code default, not an env
var: workspaces stay resident through 32K tokens and the memory-saving overlap
minimization path is used only for prompts above 32K when resolved chunk sizes
actually split the prompt.

## Build-ablation variables

These change JIT compiler flags and therefore change cache keys. They are for
kernel R&D only, not normal use.

| Variable | Default | Values / notes |
| --- | --- | --- |
| `HIPENGINE_PREFILL_MCUMODE` | false | Adds `-mcumode` to remaining `prefill` profile builds that do not already request it. Prior ablations rejected making this broad default. |
| `HIPENGINE_DISABLE_UNROLL600` | false | Strips `-mllvm -amdgpu-unroll-threshold-local=600` from profile flags for ablation. Leave unset for retained builds. |
