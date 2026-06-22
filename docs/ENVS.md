# Environment variables

Last updated: 2026-06-14

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
- Set `HIPENGINE_BACKEND=hip_gfx1100` or `HIPENGINE_BACKEND=hip_gfx1151` only
  when auto-detection falls back or you are forcing a nearby target explicitly.
- Leave diagnostic fusion/tuning knobs unset.
- Leave `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` unset.

### TheRock ROCm process setup

For the retained W7900 benchmark stack, package install/repair commands, and
ROCm 7.14 regression notes, see [`THEROCK.md`](THEROCK.md). This section only
covers the process environment wrapper.

TheRock installs ROCm pieces inside the Python environment. Activate it by
building a clean process environment around the TheRock root rather than mixing
random ROCm libraries from `/opt/rocm`:

```bash
CONDA_PREFIX=/home/lhl/mambaforge/envs/therock
ROOT=$($CONDA_PREFIX/bin/python3.12 -m rocm_sdk path --root)
env -i HOME=$HOME USER=$USER LOGNAME=$LOGNAME SHELL=$SHELL TERM=${TERM:-xterm} \
  PATH="$ROOT/bin:$ROOT/lib/llvm/bin:$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$ROOT/lib:$ROOT/lib64:$ROOT/lib/llvm/lib" \
  HIP_PATH="$ROOT" ROCM_PATH="$ROOT" HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode" \
  PYTHONPATH=. \
  python <command>
```

Use `HSA_OVERRIDE_GFX_VERSION=11.0.0` only as a local compatibility workaround
when the ROCm stack requires it for the attached gfx11 card; it is not a general
hipEngine default.

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

For GGUF Qwen3.6 MoE performance rows that intentionally use the accepted
resident T16 decode-repack path, use explicit flags rather than making them
process-global defaults:

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
| `HIPENGINE_HIP_ARCH` | HIP JIT build | unset | Force native HIP offload arch in build cache keys, e.g. `gfx1100` or `gfx1151`. The backend helper sets this temporarily when needed. |
| `HIPENGINE_HIP_OFFLOAD_ARCH` | HIP JIT build | unset | Alias-style fallback for `HIPENGINE_HIP_ARCH`. |
| `HIPENGINE_ROCM_DEVICE_LIB_PATH` | HIP JIT build | unset | Adds `--rocm-device-lib-path=<path>` to `hipcc`. Falls back to standard `HIP_DEVICE_LIB_PATH` if unset. Useful for TheRock. |
| `HIPENGINE_COMPILER_VERSION_TEXT` | HIP JIT cache | unset | Literal compiler-version text for cache keys; avoids probing `<compiler> --version`. |
| `HIPENGINE_COMPILER_VERSION_FILE` | HIP JIT cache | unset | Reads compiler-version text from a file. Recommended for cached benchmarks/profiling. |
| `HIPENGINE_HIPCC_VERSION_TEXT` / `HIPENGINE_HIPCC_VERSION_FILE` | HIP JIT cache | unset | Compiler-specific override for `hipcc`; takes precedence over the generic compiler-version vars. The same pattern applies to other compiler basenames. |
| `HIPENGINE_AOTRITON_LIB` | AOTriton discovery | unset | Explicit `libaotriton_v2.so` override. The matching `include/` and `aotriton.images/` trees must be in the standard release layout. |
| `HIPENGINE_AOTRITON_HOME` | AOTriton discovery | unset | Explicit cache root containing `<version>/lib/libaotriton_v2.so`. Missing explicit roots fail loudly instead of falling back silently. |
| `HIPENGINE_API_KEY` | OpenAI-compatible server | unset | Optional bearer token used by `hipengine serve` when `--api-key` is omitted. |
| `HIPENGINE_GENERATION_BATCH_WINDOW_MS` | OpenAI-compatible server | `0` | Opt-in cold-path coalescing delay for compatible HTTP requests. Default `0` adds no intentional delay; same-event-loop-turn requests may still share the batcher worker, while positive values are for explicit coalescer experiments. |
| `HIPENGINE_MAX_QUEUED_REQUESTS` | OpenAI-compatible server | unset | Optional generation queue cap. When set and the server batcher queue is full, new generation requests fail with HTTP 429 `engine_busy` and `Retry-After: 1`; equivalent CLI flag is `--max-queued-requests`. |
| `HIPENGINE_METRICS` | OpenAI-compatible server | `off` | Metrics endpoint mode used by `hipengine serve --metrics`: `off` or `prometheus`. When `prometheus`, `/metrics` exposes additive request counters plus KV-pool and graph-bucket counters. |
| `HIPENGINE_PREFIX_CACHE` | OpenAI-compatible server / KV sharing | `off` | Prefix-cache mode used by `hipengine serve --prefix-cache`: `off` or `radix`. `radix` enables the token-id trie scaffold for block-aligned shared-prefix admission; default remains `off` until C5 acceptance is broader. |
| `HIPENGINE_REPLAY_DIR` | OpenAI-compatible server diagnostics | unset | Opt-in directory for finite JSON failed-request replay artifacts. Disabled by default for sensitive deployments; equivalent CLI flag is `--replay-dir`. |
| `HIPENGINE_REPLAY_REDACTION` | OpenAI-compatible server diagnostics | `hash` | Replay artifact string redaction mode: `hash` replaces strings with SHA-256/length metadata, while `none` stores raw strings for explicit local debugging only. Equivalent CLI flag is `--replay-redaction`. |

Removed historical AOTriton knobs (`HIPENGINE_AOTRITON_SOURCE_ROOT` and
`HIPENGINE_AOTRITON_RUNTIME_ROOT`) are no longer read by the runtime.

## GGUF variables

| Variable | Default | Classification | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_GGUF_DECODE_REPACK` | false | Performance / memory tradeoff | Materializes resident T16 decode layouts on load. Required for current accepted Qwen3.6 GGUF MoE decode performance rows, but costs load time and resident memory, so it remains explicit. |
| `HIPENGINE_GGUF_WMMA_PREFILL` | false | Performance opt-in | Process-wide opt-in for GGUF rows>1 WMMA prefill kernels. CLI/session `--use-wmma-prefill` overrides are preferred for benchmarks. |
| `HIPENGINE_GGUF_GEMV_DECODE` | false | Performance opt-in | Process-wide opt-in for GGUF rows=1 GEMV decode kernels. For qwen35moe, effective use is safety-gated unless decode-repack is active or the unsafe override is set. |
| `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS` | false | Unsafe diagnostic | Bypasses qwen35moe GGUF fast-path safety. Do not set for normal use or promoted correctness claims. |
| `HIPENGINE_GGUF_AOTRITON_PREFILL` | `v3` | Attention implementation selector | `v3`, `v2`, or `auto`/`v2-if-safe`. `v2` is rejected for chunked suffix prefill because it has the wrong causal-mask semantics there. |
| `HIPENGINE_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT` | `1024` | Decode threshold | Context length where GGUF full-attention decode uses split/paged decode; `0` disables. Compatibility alias: `NANOVLLM_GGUF_FULL_ATTN_DECODE_PAGED_MIN_CONTEXT`. |
| `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` | `1025` | Prefill threshold | Minimum rows for GGUF GDN recurrent-segments prefill routing; invalid values fall back to the default, values below 1 clamp to 1. |
| `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` | false | Capacity diagnostic | Offloads the raw Q8_0 token embedding from device residency and serves exact Q8_0→BF16 embedding rows from host. This can make Q4_K_M 128K fit on 24 GiB, but disables GGUF HIP decode graph replay and is not a promoted performance path. |
| `HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS` | `3` | Correctness guard | Number of leading GGUF full-attention layers kept as BF16 primary storage for long explicit `int8_per_token_head` sessions. The default 3-layer prefix plus effective FP32 scales passes the W7900 forced-long `4K` BF16-vs-hybrid logit gate; short contexts (`<=8192` rounded max context) still use the exact BF16 mirror instead. |
| `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG` | false | Unsafe diagnostic | Reproduces the rejected pure GGUF `int8_per_token_head` path above the short BF16-mirror limit (`8192` rounded max context) by disabling the default BF16-prefix hybrid guard. Leave unset for normal use: pure INT8-only failed the BF16-vs-INT8 GGUF logit gate and is capacity-diagnostic only. |
| `HIPENGINE_GGUF_COMPACT_MOE_C1` | false | Diagnostic fallback | Forces the older compact c=1 MoE decode scheduler; current retained default uses direct selected T16 kernels instead. |
| `HIPENGINE_GGUF_SIDECAR_CACHE` | `~/.cache/hipengine/gguf_sidecars` (or `XDG_CACHE_HOME`) | Sidecar cache | Cache directory for optional GGUF expert pack8 sidecars. |
| `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` | unset | Kernel R&D | Optional launch-bounds macro for selected WMMA prefill builds; unset uses the retained defaults. |
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

These C4 knobs are wired into the torch-free engine-loop option resolver and
fake KV-pool scaffolding. Runtime server lowering to the loop lands separately;
until then they are primarily for scheduler/pool tests and future adapters.
CLI flags with the same names (lowercase, dash-separated) override env values
when an adapter/parser calls `add_engine_loop_config_args(...)`.

| Variable | Default | CLI flag | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_PREFILL_DECODE_POLICY` | `protect_decode` | `--prefill-decode-policy` | One of `protect_decode`, `protect_ttft`, or `fair`. |
| `HIPENGINE_MAX_ACTIVE_REQUESTS` | unset | `--max-active-requests` | Optional active resident request cap used as the engine-loop scheduler capacity when set; must be > 0. |
| `HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` | `256` | `--max-prefill-chunk-tokens` | Maximum prefill chunk tokens per loop tick; must be > 0. |
| `HIPENGINE_KV_POOL_INITIAL_PAGES` | `128` | `--kv-pool-initial-pages` | Initial dynamic KV-pool pages; must be > 0. |
| `HIPENGINE_KV_POOL_LOW_WATER_PAGES` | `128` | `--kv-pool-low-water-pages` | Idle-shrink low-water pages; must be > 0 and no greater than initial pages. |
| `HIPENGINE_KV_POOL_HIGH_WATER_PAGES` | unset | `--kv-pool-high-water-pages` | Optional grow-on-admission page cap; unset means no scaffold cap. |
| `HIPENGINE_KV_POOL_CHUNK_PAGES` | `128` | `--kv-pool-chunk-pages` | Pages per grow/shrink chunk; must be > 0. |
| `HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS` | `30.0` | `--kv-pool-idle-grace-seconds` | Seconds before fully-free tail chunks are eligible to shrink; must be ≥ 0. |
| `HIPENGINE_MAX_PENDING_REQUESTS` | unset | `--max-pending-requests` | Optional pending request queue cap for the resident scheduler; must be > 0 when set. |

## PARO variables

| Variable | Default | Classification | Values / notes |
| --- | --- | --- | --- |
| `HIPENGINE_PARO_MARLIN_K_REPLACE` | true | Retained default | Uses the retained PARO Marlin-K replacement path during loading. Set false only for bisection. |
| `HIPENGINE_QWEN35_LM_HEAD_THREADS` | `128` | Runtime tuning | Valid values: `128`, `256`, `512`. |
| `HIPENGINE_QWEN35_NATIVE_SAMPLER` | true | Retained default with rollback opt-out | Enables the scoped PARO native GPU sampler for supported c=1 and scheduler-owned serial per-slot c>N sampled requests (`top_k=0`, `1<=top_k<=64`, or exact `top_p`/`min_p` with `top_k=0`). Set `0`/`false`/`off` to force host sampling for rollback. Full-vocab `top_logprobs` with `top_k=0` and bounded `top_logprobs <= top_k <= 64` stay native; true batched c>N, GGUF, bounded `top_logprobs > top_k`, and unsupported processor/filter combinations fall back to host sampling. |
| `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` | false | Experimental diagnostic | Enables the guarded Qwen/PARO `step_batch_native` c>N decode path. Leave unset for normal use; retained throughput claims require generated-token equality and currently keep this path ineligible. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE` | `serial_lm_head` | Correctness diagnostic | `serial_lm_head` samples each native c>N row through the c=1 LM-head path; `batched_lm_head` requests batched LM-head buffers but falls back to serial for c>N unless the equality-evidence vars below are set. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK` | false | Correctness gate | Required true before `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=batched_lm_head` is honored for c>N rows. Leave false until generated-token equality vs independent c=1 is green. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ARTIFACT` | unset | Correctness gate | Relative regular `.json` path under `benchmarks/results/` to the generated-token equality artifact supporting `HIPENGINE_QWEN35_BATCH_SAMPLE_C2_EQ_OK=true`; missing, non-JSON, symlinked, non-regular, failed, wrong-row, self-mismatched `artifact_path`/`source_artifact_path`, skipped, mismatching sequence, or non-empty-mismatch artifacts keep batched LM-head on the serial fallback. |
| `HIPENGINE_QWEN35_BATCH_SAMPLE_EQ_ROWS` | unset | Correctness gate | Row count covered by the generated-token equality artifact; for c>N batched LM-head it must equal both the active row count and the artifact's row count or the sampler stays on the serial fallback. |
| `HIPENGINE_QWEN35_PROJECTION_DISPATCH_ARTIFACT` | unset | Correctness/performance gate | Relative regular JSON path under `benchmarks/results/` with `projection_dispatch_candidates`; missing or invalid artifacts keep runtime metadata on row-GEMV fallback and do not create a retained throughput claim. `scripts/qwen35_batch_retained_bench.py --projection-dispatch-artifact ...` sets this for retained runs and fails closed before the run if the artifact is symlinked/non-regular, cannot provide schema-checked candidates, or any candidate evidence artifact is missing, unsafe, rejected, self-mismatched, out of row bounds, or lacks matching >1 aggregate/per-request row-GEMV ratios. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE` | false | Correctness diagnostic | Forces batched selected-c1 MoE for c>N decode rows. Retained-bench correctness diagnostics may set this through `--batch-decode-moe-path selected_c1`; c<=8 equality artifacts may use it as a correctness-first native path, but retained throughput promotion still requires grouped-compact MoE. |
| `HIPENGINE_QWEN35_SHARED_EXPERT_PARO_W4_FORCE_GEMV` | false | Diagnostic fallback | For packed PARO W4 shared experts with `tokens<=8`, uses the row-aware GEMV path instead of the batched prefill W4 kernel. The retained-bench selected-c1 MoE diagnostic sets it so c=2/c=4/c=8 MoE replay is batched but remains non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR` | false | Diagnostic fallback | Routes linear-attention decode through the per-row c=1 layer path. Hidden-bisect equivalent: `--batch-decode-linear-path per_row`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_PROJECTIONS` | false | Diagnostic fallback | Replays linear-attention QKV/Z/A/B projections with token-1 kernels per row, then copies planar rows back into batch scratch. Hidden-bisect equivalent: `--batch-decode-linear-projection-path selected_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ` | false | Correctness diagnostic | Replays only linear-attention QKV/Z projections with token-1 kernels per row while leaving A/B on the native batch path. Hidden-bisect equivalent: `--batch-decode-linear-projection-path selected_qkv_z`. Correctness-green for c<=8 in the retained bench, but still non-retained until native projection dispatch is accepted. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB` | false | Diagnostic fallback | Replays only linear-attention A/B projections with token-1 kernels per row while leaving QKV/Z on the selected batch path. Hidden-bisect equivalents: `--batch-decode-linear-projection-path selected_ab` or `batch_gemv_selected_ab`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS` | false | Diagnostic fallback | Uses row-aware GEMV kernels for c>N linear-attention QKV/Z projections while keeping native A/B projection and segmented state. Hidden-bisect equivalent: `--batch-decode-linear-projection-path batch_gemv`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_STATE` | false | Diagnostic fallback | Replays linear-attention conv/GDN/recurrent state updates with token-1 kernels over slot-local state. Hidden-bisect equivalent: `--batch-decode-linear-state-path selected_c1`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_OUT` | `auto` | Diagnostic fallback | Linear-attention output projection override: `auto`, `batch`, `batch_gemv`, or `selected_c1`. `auto` follows selected-c1 state replay; `batch_gemv` bypasses the row>1 AWQ prefill projection kernel while staying non-retained. Hidden-bisect equivalent: `--batch-decode-linear-output-path ...`. |
| `HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE` | true when experimental decode is enabled | Diagnostic selector | Set `0` to force the existing per-row full-attention fallback in hidden-bisect/native-batch probes. Non-retained fallback metadata records `full_attention_decode_path=per_row_*`. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT` | false | Diagnostic fallback | Forces only the full-attention input RMSNorm/QKV-prep boundary through token-1 row kernels. Hidden-bisect equivalent: `--batch-decode-attn-input-path per_row`. Non-retained. |
| `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN` | false | Diagnostic fallback | Forces only the post-attention add/RMSNorm boundary through token-1 row kernels. Hidden-bisect equivalent: `--batch-decode-post-attn-path per_row`. Non-retained. |
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
