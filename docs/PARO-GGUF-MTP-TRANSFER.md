# PARO Follow-Ups from GGUF/MTP Work

Last updated: 2026-07-09.

This is the active TODO list for applying recent GGUF/MTP server and verifier
lessons to PARO. The key split is:

- GGUF quant kernels mostly do not port directly to PARO (`Q*_K`, Q8/dp4a,
  Q6_K rowtile LM-head, GGUF T16 layouts).
- Server shape handling, startup warmup, timing attribution, route caps,
  scheduler telemetry, and verifier lifecycle ideas do apply and need PARO
  evidence before promotion.

## Current Evidence Split

| Scope | Current status | Evidence |
| --- | --- | --- |
| PARO direct retained c>N harness | Accepted on gfx1100/RX 7900 XTX for c=4 and c=8 512/128. This proves the native compact PARO runtime can scale outside the server. | `benchmarks/README.md` retained rows: c=4 `155.987 tok/s`, c=8 `212.093 tok/s`; artifacts `benchmarks/results/2026-06-02-hipengine-qwen35-native-c4-profiler-preflight/native-diagnostic-c4.json` and `benchmarks/results/2026-06-02-hipengine-qwen35-native-c8-exact-profile/profiled-retained-c8.json`. |
| PARO OpenAI server c>N | Not yet measured with the same retained direct protocol. Do not infer server throughput from the direct harness. | Server path must prove whether it reaches native packed prefill and native c-aware decode, or falls back through the serial slot bridge. |
| PARO on gfx1151/Radeon 8060S | Needs a fresh server diagnostic sweep. Existing gfx1151 README rows show weaker hipEngine concurrency than llama.cpp Vulkan on this host, so backend/scheduler shape effects must be measured rather than assumed. | Use the local shisa packed model cache and record host, model, quant, command, telemetry, and JSON artifacts for every row. |

## Missed Opportunities From The GGUF Audit

| Priority | Opportunity | Why it matters | First evidence to collect |
| --- | --- | --- | --- |
| P0 | Measure PARO OpenAI server c=1/2/4/8 vs the direct retained c>N harness. | The direct harness is fast, but support is advertised through the server. A server gap would hide the direct path's gains from users. | Run 512/128 exact-token OpenAI completions through `hipengine.server`; capture wall/e2e throughput, `scheduler_token_chunks`, `/ready`, and `/v1/hipengine/capabilities`. |
| P0 | Prove native server route selection. | `hipengine/generation/qwen35_paro.py` attempts `step_batch_native`, but `hipengine/runtime/qwen35_paro_runner.py` gates it behind `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1` and can fall back to serial decode. | Compare default, native-decode, and native-decode-plus-startup-warmup runs. Retain `execution_path`, `native_compact_prefill`, `native_caware_decode`, and `serial_decode_fallback` telemetry. |
| P0 | Separate prefill-inclusive server throughput from decode-only direct throughput. | Recent GGUF server work showed full-request throughput can hide decode wins behind per-request prefill and scheduling costs. | Record both full-request client aggregate and per-choice/server decode telemetry where available. Mark rows diagnostic unless the correctness and telemetry gates are complete. |
| P1 | Sweep route/group caps on gfx1151. | GGUF server work found that a smaller backend group can beat unconstrained c=8 on Strix Halo. PARO currently uses global `--max-active-requests`; it may need separate caps. | Sweep `--max-active-requests` 2/4/8 after the baseline path is stable; start with 8, then cap to 4/2 if queue or native-route telemetry degrades. |
| P1 | Add PARO server stage buckets. | GGUF only became actionable after per-stage buckets exposed where AR/MTP wall time moved. PARO server currently lacks equivalent timing attribution. | Minimum AR buckets: prompt prefill, decode layer wall, projection dispatch, MoE, attention, sampler/LM-head, graph replay, host readback, scheduler/queue wall. |
| P1 | Carry GGUF startup warmup discipline into PARO. | Startup scratch and graph-cache misses can make the first server requests unrepresentative and can select weaker shapes. | Test `HIPENGINE_QWEN35_SERVER_STARTUP_NATIVE_BATCH_WARMUP=1` with native decode enabled, and record `/ready` startup diagnostics. |
| P1 | Reconcile PARO concurrency docs after server evidence. | `benchmarks/README.md` has accepted direct retained c4/c8 rows, while `docs/CONCURRENCY.md` still has older cautionary wording. The right wording depends on server results. | After the server sweep, update `docs/CONCURRENCY.md` to distinguish direct retained c>N from server c>N. |
| P2 | Audit PARO MTP/DFlash commit/scatter against GGUF deferred verifier scatter. | GGUF verifier wins came from avoiding rejected-tail state work and only committing accepted rows. PARO target verify APIs are still metadata-heavy, and commit can execute copies. | Add verifier buckets before changing code: accepted-row scatter/copy, rejected-tail discard, hidden/KV/state copies, host sync/readback. |
| P2 | Compare PARO MTP/DFlash small-B verifier shapes against GGUF exact and llama-compat. | GGUF exact and llama-compat became faster than the older PARO speculative path partly by avoiding bad small-B shapes and tightening LM-head/sample. | Bucketed PARO speculative profile over draft/propose, target verify by layer family, LM-head/top1, accept summary, commit/scatter, graph replay, and scheduler wall. |
| P3 | Review GGUF llama-compat verifier against PARO MTP/DFlash only after AR server is understood. | The verifier review is useful, but AR server routing and shape selection are lower-level prerequisites. | One-to-one map: draft, target verify, LM-head/sample, state capture, commit/rollback, rejection-tail handling. |

## First Server Sweep

Hardware and model for the local first pass:

- Hardware: gfx1151 / Radeon 8060S.
- Model: `/home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`.
- Quant: `w4_paro`.
- Target workload: exact token-id prompts, prompt length 512, decode 128,
  concurrency c=1/2/4/8, greedy completions. If the server route rejects
  token-id prompts, run a natural-prompt diagnostic first and keep it separate
  from retained direct-harness comparisons.
- Fixture: recreate `/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json`
  from repo prompt fixtures if the retained benchmark fixture is absent.

Run three server configurations before any code changes:

| Config | Env | Purpose |
| --- | --- | --- |
| A: default server | No PARO native batch env flags. | Establish the actual advertised server path. |
| B: native decode | `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`. | Test whether server throughput closes toward the direct retained harness when native c-aware decode is allowed. |
| C: native decode + startup warmup | `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1 HIPENGINE_QWEN35_SERVER_STARTUP_NATIVE_BATCH_WARMUP=1`. | Test whether startup c>N shape warmup changes first-request and steady server throughput. |

Capture for every row:

- Exact server command, client command, git commit, dirty status, HIP arch,
  ROCm/HIP version, GPU name, model path, quant, prompt/decode/concurrency.
- `/ready` startup diagnostics and `/v1/hipengine/capabilities`.
- Client JSON from `scripts/vllm_openai_concurrency_sweep.py` for exact-token
  completions, or `scripts/mtp-bench.py` server mode for natural-prompt
  diagnostics.
- Per-choice hipEngine telemetry if present, especially `scheduler_token_chunks`,
  `execution_path`, `native_compact_prefill`, `native_caware_decode`,
  `serial_decode_fallback`, and native sampler metadata.
- Mark rows diagnostic until correctness, telemetry, and artifact gates match
  `docs/BENCHMARK.md`.

After A/B/C, sweep route caps:

| Sweep | Values | Gate |
| --- | --- | --- |
| `--max-active-requests` | 2, 4, 8 | Only compare rows with the same env config, same prompt fixture, same decode length, and matching native-route telemetry. |
| `--generation-batch-window-ms` | start at 5, then 1/10 if the baseline is unstable | Report full-request throughput separately from decode-only telemetry. |

## First Server Sweep Results

Measured 2026-07-09 on gfx1151 / Radeon 8060S, shisa
`Qwen3.6-35B-A3B-PARO-packed`, `w4_paro`, `scripts/mtp-bench.py` server mode,
8 natural chat prompts from `benchmarks/fixtures/llamacpp_mtp_bench_prompts.json`,
`max_tokens=128`, greedy, `ignore_eos=true`, batch window 5 ms. These rows are
diagnostic and **not** the direct retained 512-token exact-token protocol:
`/v1/completions` currently rejects token-id prompt arrays, so exact-token server
benchmarking is itself a follow-up.

Artifact summary:
`benchmarks/results/2026-07-09-hipengine-paro-server-ar-mtpbench-natural8-summary.json`.

| Server config | c=1 | c=2 | c=4 | c=8 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| A: default server | 35.50 | 35.39 | 35.50 | 35.61 | Flat; default server serializes backend decode for this workload. |
| B: native decode env | 35.50 | 37.48 | 40.33 | 42.36 | Modest c>N lift only. |
| C: native decode + startup warmup | 35.66 | 37.55 | 40.56 | 41.51 | Startup c>N warmup ran, but steady throughput is unchanged vs B. |

Route-cap sweep under config C, c=8:

| `--max-active-requests` | Backend generated tok/s | Result |
| ---: | ---: | --- |
| 8 | 41.51 | Best of the cap sweep. |
| 4 | 39.59 | Worse than cap 8. |
| 2 | 37.47 | Worse than cap 8. |

Telemetry notes:

- Default c=8 reports `execution_path=scheduler_native_packed_prefill_serial_decode`
  and `serial_decode_fallback=true`.
- Native-decode c=8 reports
  `execution_path=scheduler_native_packed_prefill_native_decode`, but
  `native_caware_decode=false` in the emitted decode state. That explains why the
  env flag does not approach the direct retained c>N harness.
- The direct retained c=8 harness on gfx1100 is `212.093 tok/s`; this gfx1151
  server diagnostic reaches only `42.36 tok/s`. Hardware and workload differ, so
  do not treat the ratio as a direct regression claim, but it proves the server
  route is still the first blocker before PARO speculative/MTP work.

Immediate next targets from this sweep:

1. Add or expose an exact-token server benchmark route so PARO server rows can
   run the same 512/128 fixture as the direct retained harness.
2. Inspect why `scheduler_native_packed_prefill_native_decode` still emits
   `native_caware_decode=false` and scales weakly.
3. Add server AR buckets for prefill, decode layer wall, projection, attention,
   MoE, sampler/LM-head, host sync/readback, and scheduler wall.
4. Only after the AR server path scales should PARO MTP/DFlash verifier
   lifecycle optimizations be ported from GGUF.

## Retained Defaults Bridge and Current Bottleneck

After the first sweep, the server path gained two pieces of observability and
an opt-in retained-evidence bridge from the direct retained harness:

- `GenerationTelemetry` now accepts a `diagnostics` payload, and the OpenAI
  response capability metadata advertises `choice_telemetry.diagnostics`.
- PARO batch generation now emits backend timing buckets:
  `batch_total_ms`, `batch_prefill_ms`, `batch_decode_ms`,
  `batch_decode_step_ms_avg`, `batch_decode_steps`, and
  `batch_native_decode_steps`.
- When `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS=1` is explicitly set and no
  explicit projection/sampler override is set, the runner can load retained
  direct-harness defaults where repo evidence exists:
  `benchmarks/results/2026-07-09-hipengine-qwen35-native-c2468-projection-dispatch-catalog/summary.json`
  for c2/c4/c6/c8 projection dispatch, and
  `benchmarks/results/2026-06-02-hipengine-qwen35-c{2,4,8}-native-batch-sampler-equality.json`
  for row-aware batched LM-head sampling.

Important status: this bridge is **not** auto-enabled by
`HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1`. A local gfx1151/shisa
recheck rejected the old retained defaults on generated-token equality, so they
remain diagnostic/opt-in until fresh local c>N equality evidence is green.

Measured 2026-07-09 on the same gfx1151/Radeon 8060S server setup, c=8,
batch window 20 ms, native decode plus startup warmup plus retained defaults:

Artifact:
`benchmarks/results/2026-07-09-hipengine-paro-server-ar-mtpbench-natural8-c8-bw20-native-decode-warmup-retained-defaults-timing.json`.

| Metric | Value |
| --- | ---: |
| Aggregate backend generated throughput | `42.02 tok/s` |
| Aggregate wall | `24.37 s` |
| Total backend generated tokens | `1024` |
| Backend prefill timing total | `2358.156 ms` |
| Backend decode timing total | `51821.026 ms` |
| Mean per-choice prefill timing | `294.769 ms` |
| Mean per-choice decode timing | `6477.628 ms` |
| Mean per-choice decode step timing | `51.005 ms` |

The run is decode-bound, not prefill-bound. Its detailed diagnostics split the
8 requests into one faster c2 group and one slow c6 group:

| Active rows | Requests | Decode step timing | Projection | Sampler | Current blocker |
| ---: | ---: | ---: | --- | --- | --- |
| 2 | 2 | `27.36 ms` | `gemv_awq_selected_dual_pack8_strided_c2` | evidenced batched LM-head | Fast in this timing run, but later local equality recheck rejected the retained bridge. |
| 6 | 6 | `58.89 ms` | row-GEMV fallback | serial LM-head | No c6 projection dispatch candidate and no c6 sampler equality artifact. |

That server timing row predates the local c6 projection repair below. The c6
projection fallback is now covered by a local generated-token equality probe,
but the full c6 path remains diagnostic because selected-c1 MoE plus rowchunked
full/linear attention are still fallbacks and the retained primitive/profiler/
baseline gates are not complete.

Follow-up local generated-token checks changed the immediate diagnosis:

| Probe | Rows | Result | First mismatch | Artifact |
| --- | --- | --- | --- | --- |
| Old retained c2/c4/c8 bridge on local gfx1151 shisa | c2, c4, c8 | Rejected correctness | token 2 for every row | `benchmarks/results/2026-07-09-hipengine-qwen35-c248-local-retained-defaults-check/summary.json` |
| Intermediate-row sampler seed matrix | c3, c5, c6, c7 | Rejected correctness | token 4 for c3/c5/c6; c7 mostly token 4 with one row at token 2 | `benchmarks/results/2026-07-09-hipengine-qwen35-c3567-serial-sampler-equality-seed/summary.json` |
| c6 all-full-attention rowchunk probe | c6 | Rejected correctness | generated-token equality failed | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-rowchunk-all-probe.json` |
| c6 per-row full-attention probe | c6 | Rejected correctness | generated-token equality failed | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-fullattn-perrow-probe.json` |

This changed the recovery target from "add row-shape coverage" to "recover
local generated-token equality first." The old retained c2/c4/c8 bridge was red,
and the server naturally admits intermediate live row counts like c3/c5/c6 and
c7. Later probes recovered c2/c4/c8 with different diagnostic shapes, documented
below, while c6 remains the server-relevant blocker. Until c6 is isolated or the
scheduler avoids c6 live-row groups, no PARO server c>N perf path should be
promoted by default.

## c2 Generated-Token Bisection

Measured 2026-07-09 on the same gfx1151/Radeon 8060S local shisa setup with a
short c2 diagnostic (`prompt=512`, `decode=8`, `warmup=0`) and final sampler
audit enabled. The sampler audit reruns serial per-row final RMSNorm/cast/LM-head
and argmax from the same hidden rows.

| Diagnostic path | Generated-token result | First mismatch | Sampler suffix audit | Artifact |
| --- | --- | --- | --- | --- |
| Native c2 retained bridge | Red | token 2: batch `220`, c1 `17` | Clean: 0 mismatches over 16 checked rows | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-finalnorm-audit.json` |
| Native c2 + selected-c1 linear projection | Red | token 2: batch `220`, c1 `17` | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-selected-c1-proj-finalnorm-audit.json` |
| Linear per-row fallback + native full-attention | Red | token 2: batch `220`, c1 `17` | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-linear-perrow-finalnorm-audit.json` |
| Full-attention per-row fallback | Red, but later | token 6: batch `220`, c1 `17` | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-fullattn-perrow-finalnorm-audit.json` |
| Full-attention per-row + selected-c1 linear state | Red | token 6: batch `220`, c1 `17` | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-fullattn-perrow-linearstate-c1-finalnorm-audit.json` |
| Full-attention per-row + selected-c1 linear output | Red | token 6: batch `220`, c1 `17` | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-fullattn-perrow-linearout-c1-finalnorm-audit.json` |
| Full-attention per-row + global selected-c1 MoE | Green for this short diagnostic | none | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-fullattn-perrow-moe-selected-c1-finalnorm-audit.json` |
| Full-attention per-row + all linear per-row | Green for this short diagnostic | none | Clean | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-short-linear-full-perrow-finalnorm-audit.json` |

Interpretation:

- The sampler suffix is not the c2 blocker; final norm/cast/LM-head/argmax
  matched serial per-row for the same hidden rows.
- Native full-attention is the first local c2 divergence. Forcing
  full-attention per-row moves the failure from token 2 to token 6.
- After full-attention is per-row, grouped compact MoE is the next visible
  divergence. Global selected-c1 MoE clears the short c2 diagnostic; selected-c1
  linear state and output do not.
- The narrow `--batch-decode-linear-moe-path per_row_c1` diagnostic currently
  crashes with `HIP error 1: invalid argument`, so use the broader
  `--batch-decode-moe-path selected_c1` fallback as the working MoE
  discriminator until that diagnostic branch is fixed.

Follow-up full 512/128 probes refined this repair order: c2 no longer needs a
per-row full-attention fallback when the global selected-c1 MoE discriminator is
enabled, c4/c8 need all full-attention producer layers split into rowchunk2
groups, and c6 needs selected early full-attention producer layers split into
rowchunk2 groups. c6 rowchunk3 is hidden-red, so the current safe c6 row group
cap is two rows.

## Local c>N Recovery Frontier

Measured 2026-07-09 on gfx1151 / Radeon 8060S with local shisa
`Qwen3.6-35B-A3B-PARO-packed`, `w4_paro`, fixture
`/tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json`,
prompt 512, decode 128, greedy. These are generated-token equality probes
against independent c1 resident runs. They are **not** retained throughput
claims yet because the primitive/profiler/baseline retained gates are still
missing, and selected-c1 MoE/rowchunk repairs are diagnostic fallbacks.

| Rows | Diagnostic shape | Generated-token equality | Decode aggregate | Artifact |
| ---: | --- | --- | ---: | --- |
| 2 | native full-attention, selected-c1 MoE, batched LM-head | Pass | `78.021 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c2-p512-d128-selected-c1-moe-local-equality.json` |
| 4 | rowchunk2 on every full-attention layer, selected-c1 MoE, batched LM-head | Pass | `99.046 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c4-p512-d128-rowchunk2-all-moe-selected-c1-local-equality.json` |
| 8 | rowchunk2 on every full-attention layer, selected-c1 MoE, batched LM-head | Pass | `115.066 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c8-p512-d128-rowchunk2-all-moe-selected-c1-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, selected-c1 MoE, serial LM-head | Red at token 2 | invalid `106.869 tok/s` diagnostic | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-all-moe-selected-c1-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on full-attention layers `3,7,11,15,19,23,27,31`, native linear attention, selected-c1 MoE with forced small-batch shared expert, serial LM-head, native/batch projection | Pass | `108.929 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-selected-full-rowchunks-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, native linear attention, selected-c1 MoE with forced small-batch shared expert, serial LM-head, native/batch projection | Pass | `107.891 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-smallbatch-shared-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, native linear attention, per-row c1 MoE on linear layers, serial LM-head, native/batch projection | Pass | `92.800 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-native-linear-linearmoe-perrow-full-rowchunk2-selected-c1-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, linear-attention rowchunk2, selected-c1 MoE, serial LM-head, native/batch projection | Pass | `87.612 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-full-linear-moe-selected-c1-nativeproj-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, linear-attention rowchunk2, selected-c1 MoE, serial LM-head, selected-c1 projection fallback | Pass | `84.709 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-full-linear-moe-selected-c1-selectedproj-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, linear-attention rowchunk2, selected-c1 MoE, batched LM-head | Pass | `87.519 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-full-linear-moe-selected-c1-batched-lmhead-local-equality.json` |
| 6 | per-row full-attention, per-row linear, selected-c1 MoE, serial LM-head | Pass | `69.802 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-linear-full-perrow-moe-selected-c1-serial-sampler-local-equality.json` |

Focused c6 hidden-bisect controls:

| Probe | Result | Artifact |
| --- | --- | --- |
| full-attention per-row, native linear batch segments, selected-c1 MoE | Hidden red at layer 2 / generated index 4 | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-linear-rowchunk-hidden-bisect-summary.json` |
| full-attention per-row, native linear batch segments, selected-c1 MoE, forced small-batch shared expert | Pass, no linear-stage drift | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-hidden-bisect-L1-L4-d4-fullattn-perrow-linear-native-selectedc1-smallbatch-shared.json` |
| full-attention per-row, native linear batch segments, per-row c1 MoE on linear layers | Pass, no linear stage drift | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-hidden-bisect-L1-L4-d4-fullattn-perrow-linear-native-linearmoe-perrow-after-prefixfix.json` |
| full-attention per-row, linear rowchunk2, selected-c1 MoE | Pass | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-linear-rowchunk-hidden-bisect-summary.json` |
| full-attention per-row, linear rowchunk3, selected-c1 MoE | Hidden red | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-linear-rowchunk-hidden-bisect-summary.json` |
| full-attention per-row, linear rowchunk2, grouped-compact MoE | Hidden red | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-linear-rowchunk-hidden-bisect-summary.json` |

Focused c6 projection control:

| Projection path | Generated-token equality | Decode aggregate | Median step | Artifact |
| --- | --- | ---: | ---: | --- |
| native/batch projection candidate | Pass, prefix `[129,129,129,129,129,129]` | `87.612 tok/s` | `68.282 ms` | `benchmarks/results/2026-07-09-hipengine-qwen35-native-c6-projection-dispatch/c6-projection-evidence.json` |
| selected-c1 row-GEMV fallback | Pass, prefix `[129,129,129,129,129,129]` | `84.709 tok/s` | `70.580 ms` | `benchmarks/results/2026-07-09-hipengine-qwen35-native-c6-projection-dispatch/c6-projection-evidence.json` |

This adds `gemv_awq_selected_dual_pack8_strided_c6` to the default projection
dispatch catalog with a `1.0343x` aggregate decode gain over the row-GEMV
fallback. It is projection evidence only, not a retained/default throughput
claim for the whole c6 path.

Focused c6 native-linear and shared-expert repair:

Compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-smallbatch-shared-repair-summary.json`.

| Shape | Generated-token equality | Decode aggregate | Median step | Status |
| --- | --- | ---: | ---: | --- |
| old c6 bridge: full rowchunk2 + linear rowchunk2 + selected-c1 MoE | Pass | `87.612 tok/s` | `68.282 ms` | diagnostic correctness bridge |
| new c6 repair: full rowchunk2 + native rows=6 linear + per-row c1 MoE on linear layers | Pass | `92.800 tok/s` | `64.335 ms` | diagnostic correctness bridge |
| current c6 repair: selected full rowchunk2 layers `3,7,11,15,19,23,27,31` + native rows=6 linear + selected-c1 MoE with forced small-batch shared expert | Pass | `108.929 tok/s` | `54.716 ms` | diagnostic correctness bridge |
| previous c6 repair: full rowchunk2 on every full-attention layer + native rows=6 linear + selected-c1 MoE with forced small-batch shared expert | Pass | `107.891 tok/s` | `55.293 ms` | diagnostic correctness bridge |

Delta: the current selected-full-rowchunk repair is `+24.33%` aggregate
decode and `-13.566 ms` median decode step versus the linear-rowchunk bridge,
and `+17.38%` / `-9.619 ms` versus the per-row-linear-MoE repair. It also
beats the previous all-full-layer rowchunk repair by `+0.96%` and
`-0.577 ms/step`. The
hidden-bisect says the native rows=6 linear layer is bit-clean through
`mlp_input`; the first drift in the old native-linear probe was the batched
selected-c1 MoE output, specifically the multi-row packed shared-expert path.
Forcing only the shared expert back to the small-batch/GEMV path repairs that
drift without replaying every linear-layer MoE row. Fixing
`reserve_moe_c1_scratch(prefix=...)` to prefix `shared_out`, `moe_out`, and the
shared-rotate barrier remains useful for the older row-local replay diagnostic.

The current retained-bench auto diagnostic path should therefore start from the
green local frontier:

- c2/c4/c6/c8: auto-select selected-c1 MoE.
- c2/c4/c6/c8: load the c-aware projection dispatch catalog; c6 now has a
  generated-token-green projection candidate.
- c4/c8: auto-select full-attention rowchunk2 with an empty layer list, meaning
  every full-attention layer is rowchunked.
- c6: auto-select full-attention rowchunk2 only on layers
  `3,7,11,15,19,23,27,31`; no-rowchunk is faster but rejected at token 7, while
  all-layer rowchunk is green but slower.
- c6: auto-select native rows=6 linear-attention segments, selected-c1 batch
  MoE, and forced small-batch shared expert. Explicit
  `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR_MOE=1` keeps the older
  per-row-linear-MoE bridge available for bisection, and explicit linear-rowchunk
  env overrides keep the older rowchunk2 bridge available.
- c3/c5: keep the older selected-layer rowchunk diagnostic scope until
  local full 512/128 equality evidence replaces it.

Server relevance: the natural c=8 server diagnostic previously split into a
fast c2 group and a slow c6 group. Recovering c2/c4/c8 direct diagnostics helps,
and the c6 direct diagnostic now has a correct fallback shape. It is still a
diagnostic fallback, not a retained/default throughput claim, because it relies
on selected-c1 MoE and rowchunked full attention and is missing the
primitive/profiler/baseline retained gates.

Runtime retained-default recheck:

| Probe | Result |
| --- | --- |
| Forced OpenAI `n=6` server request, code_python, max_tokens=128, batch window 200 ms, native decode + startup warmup + retained defaults, before the per-row-linear-MoE repair | rows=6 used `moe_decode_path=selected_c1_batch`, `linear_attention_decode_path=native_batch_row_chunks`, `linear_attention_row_chunk_size=2`, `full_attention_decode_path=native_batch_row_chunks`, `full_attention_row_chunk_size=2`, and `linear_attention_projection_path=native_batch`; artifact `benchmarks/results/2026-07-09-hipengine-paro-server-ar-mtpbench-code-python-n6-bw200-native-decode-warmup-retained-defaults-c6repair.json`. |
| Timing vs prior unrepaired c6 server probe | Decode step changed `56.873 ms -> 64.101 ms`; aggregate backend generated throughput changed `8.67 -> 8.14 tok/s`. This confirms the server-visible correctness shape, but not a speed win. |
| Runtime default after the small-batch shared-expert repair | `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS=1` now selects selected-c1 batch MoE plus `moe_c1_shared_expert_decode_path=small_batch_forced` for rows=6 when selected-c1 MoE is active and the shared-expert env override is blank. The first direct retained-bench generated-token equality bridge passed at `107.891 tok/s`, median `55.293 ms`; artifact `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-smallbatch-shared-local-equality.json`. |
| Forced OpenAI `n=6` server request after the small-batch shared-expert repair | rows=6 used `moe_decode_path=selected_c1_batch`, `moe_c1_shared_expert_decode_path=small_batch_forced`, no linear rowchunk, `full_attention_decode_path=native_batch_row_chunks`, and `full_attention_row_chunk_size=2`; backend generated throughput `9.16 tok/s`, decode step `51.053 ms`. Compact summary `benchmarks/results/2026-07-09-hipengine-paro-server-ar-c6-smallbatch-shared-summary.json`; raw artifact `benchmarks/results/2026-07-09-hipengine-paro-server-ar-mtpbench-code-python-n6-bw200-native-decode-warmup-retained-defaults-smallbatch-shared.json`. |
| Runtime default after the selected-full-rowchunk repair | rows=6 now uses full-attention rowchunk2 only on `3,7,11,15,19,23,27,31`, native rows=6 linear attention, selected-c1 batch MoE, and forced small-batch shared expert. Direct generated-token equality passed at `108.929 tok/s`, median `54.716 ms`; no-rowchunk is faster (`114.329 tok/s`, median `52.140 ms`) but rejected at token 7. Summary `benchmarks/results/2026-07-09-hipengine-qwen35-c6-selected-full-rowchunks-summary.json`; direct artifact `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-selected-full-rowchunks-local-equality.json`. |
| Forced OpenAI `n=6` server request after the selected-full-rowchunk repair | rows=6 used `moe_decode_path=selected_c1_batch`, `moe_c1_shared_expert_decode_path=small_batch_forced`, no linear rowchunk, `full_attention_decode_path=native_batch_row_chunks`, `full_attention_row_chunk_size=2`, and `full_attention_row_chunk_layers=[3,7,11,15,19,23,27,31]`; backend generated throughput `9.26 tok/s`, decode step `50.434 ms`. Raw artifact `benchmarks/results/2026-07-09-hipengine-paro-server-ar-mtpbench-code-python-n6-bw200-native-decode-warmup-retained-defaults-selected-full-rowchunks.json`. |

c6 server splitter:

Compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-paro-server-ar-c6-linear-moe-split-summary.json`.

| c6 server shape | MoE | Linear attention | Decode step | Backend generated | Status |
| --- | --- | --- | ---: | ---: | --- |
| grouped native-linear | grouped-compact | native segments | `56.873 ms` | `8.67 tok/s` | hidden-red timing diagnostic |
| selected rowchunk repair | selected-c1 | rowchunk2 | `64.101 ms` | `8.14 tok/s` | current server-visible correctness bridge |
| selected native-linear | selected-c1 batch | native segments | `51.619 ms` | `9.01 tok/s` | fastest prior timing, but hidden-red because rows=6 selected-c1 batch MoE drifts |
| selected native-linear + per-row linear MoE | per-row c1 on linear layers | native segments | pending server rerun | pending | direct retained-bench generated-token green at `92.800 tok/s` |
| selected native-linear + small-batch shared expert + selected full rowchunks | selected-c1 batch with small-batch shared expert | native segments | `50.434 ms` | `9.26 tok/s` | server-visible; full-attention rowchunk diagnostic remains |
| selected native-linear + small-batch shared expert + all full rowchunks | selected-c1 batch with small-batch shared expert | native segments | `51.053 ms` | `9.16 tok/s` | prior server-visible bridge |
| grouped rowchunk | grouped-compact | rowchunk2 | `71.877 ms` | `7.59 tok/s` | hidden-red and slower |

This split says the biggest c6 tax was linear rowchunk2, not projection. The
direct repair shows native rows=6 linear-attention segments are correctness-clean
when the selected-c1 shared expert uses the small-batch path. The remaining
server rerun confirms the selected-rowchunk default is visible and recovers the
c6 server timing to `50.434 ms/step`; the remaining work is to decide whether to
keep the shared-expert small-batch bypass, fold it into the bundled C dispatcher,
remove the remaining full-attention rowchunk blocker, or avoid live c6 groups in
scheduling.

Follow-up c6 C-dispatch folding result:

| Shape | Generated-token equality | Decode aggregate | Median step | Artifact |
| --- | --- | ---: | ---: | --- |
| Prior runtime bypass for forced small-batch shared expert | Pass | `108.929 tok/s` | `54.716 ms` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-selected-full-rowchunks-local-equality.json` |
| C dispatcher handles forced small-batch shared expert | Pass | `109.205 tok/s` | `54.610 ms` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-cdispatch-smallbatch-shared-selected-full-rowchunks-local-equality.json` |
| Same-session C dispatcher disabled | Pass | `109.123 tok/s` | `54.672 ms` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-retained-default-cdispatch-off-smallbatch-shared-selected-full-rowchunks-local-equality.json` |

This folds the small-batch shared-expert correctness route into the bundled
MoE C dispatcher, but the measured delta is noise-level (`+0.075%` vs the
same-session C-dispatch-off fallback). Treat it as structural cleanup, not a
c6 speed win. The active c6 performance blocker remains full-attention
rowchunking on selected layers or, alternatively, scheduling that avoids c6
live-row groups.

Follow-up c6 no-rowchunk full-attention rejects:

Compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-full-attn-no-rowchunk-substage-rejects-summary.json`.

| Probe | Generated-token equality | First mismatch | Decode aggregate | Median step | Conclusion |
| --- | --- | ---: | ---: | ---: | --- |
| Native rows=6 full attention, no rowchunk, forced full-attention O projection batch GEMV | Red | token 9 | `108.525 tok/s` | `52.264 ms` | O projection alone is not the native c6 divergence. |
| Same plus dense-context batch-gate override on `3,7,11,15,19,23,27,31` | Red | token 2 | `102.893 tok/s` | `55.530 ms` | Dense-context batch-gate override is slower and correctness-worse. |

These rejects keep the selected full-attention rowchunk2 bridge as the only
known generated-token-green c6 shape. The next useful isolation step is a
full-attention substage hidden-bisect around the selected producer layers,
especially row interaction or scratch/state aliasing in native rows=6 context
construction, rather than more O-projection-only probes.

Follow-up c6 hidden-bisect no-rowchunk diagnostics:

Compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-hidden-bisect-no-rowchunk-summary.json`.

The raw hidden traces were `8.3M` to `37M`, so they are kept locally and the
compact summary records only the diagnosis. The diagnostic script now has
`--skip-full-context-oracle` and `--skip-linear-state-summary` to avoid the
expensive NumPy full-context oracle and linear-state summaries when the active
question is hidden/token/KV source equality.

| Probe | Equality result | Key signal | Conclusion |
| --- | --- | --- | --- |
| Full-native no-rowchunk hidden-bisect, L8-L12, trace generated index 9 | hidden red, token green | first full KV sample mismatch at layer 7 key, row 0, sample positions `[0,255,256,519,520]`; positions match; current `batch_source_vs_c1_source` fails at `key_after_prepare` while cache-vs-source is clean | Cache placement/page boundaries are not the lead; layer-7 K/V source production differs before append. |
| Rowchunk layer 7 only hidden-bisect | hidden red, token green | same layer-7 class of mismatch remains in the shallow trace | Rowchunking layer 7 alone is not enough; the known green bridge still needs selected rowchunks `3,7,11,15,19,23,27,31`. |
| No-rowchunk generated-token probe with full-attention QKV forced per-row | generated-token red at token 9 | `107.578 tok/s`, median `53.022 ms`; batch token `12` vs c1 token `27` on all rows | Per-row QKV scratch alone is not the repair. |
| No-rowchunk generated-token probe with full-attention input forced per-row | generated-token red at token 9 | `108.465 tok/s`, median `52.851 ms`; batch token `12` vs c1 token `27` | Per-row input/RMSNorm does not repair the native c6 divergence. |
| No-rowchunk generated-token probe with full-attention scratch forced per-row | generated-token red at token 2 | `93.031 tok/s`, median `61.943 ms` | Whole-layer c1-like scratch fallback is correctness-worse and too slow. |
| No-rowchunk generated-token probe with context forced per-row only | generated-token red at token 2 | `100.940 tok/s`, median `56.668 ms` | Per-row context replay does not repair the native c6 divergence. |
| No-rowchunk generated-token probe with KV append forced per-row | generated-token red at token 9 | `107.954 tok/s`, median `52.424 ms`; same failure shape as baseline | Append mechanics alone are not the repair. |
| Full-attention rowchunk only on layer 7 | generated-token red at token 7 | `108.616 tok/s`, median `52.486 ms` | Layer 7 rowchunking alone is insufficient. |
| Full-attention rowchunk on layers `3,7,11,15`, short 16-token full model | generated-token green, diagnostic blocked by missing primitive/profiler gates | `107.413 tok/s`, median `53.301 ms` | Early rowchunks are enough for the short probe, but the 128-token retained gate still requires the eight-layer set `3,7,11,15,19,23,27,31`. |
| Selected-c1 staged diagnostic: per-row pre-QKV+append, batch context/O/post/MoE | generated-token red at token 9 | `108.209 tok/s`, median `52.714 ms` | Pre-QKV and append are not sufficient; keeping batch context preserves the original failure shape. |
| Selected-c1 staged diagnostic: per-row pre-QKV+append+context, batch gate/O/post/MoE | generated-token red at token 2 | `100.095 tok/s`, median `57.137 ms` | Moving context per-row changes the failure earlier and is slower. |
| Selected-c1 staged diagnostic: per-row pre-QKV+append+context+gate, batch O/post/MoE | generated-token red at token 2 | `100.707 tok/s`, median `56.905 ms` | Batch gate is not the token-2 shift; per-row context/gate remains worse than batch context. |
| Context-only native rowchunk2 on selected full-attention layers `3,7,11,15,19,23,27,31` | generated-token red at token 2 | `106.596 tok/s`, median `53.834 ms`; batch token `220` vs c1 token `17` | Splitting only the native context kernel is not a substitute for the green full-layer rowchunk bridge. |
| Context-only native rowchunk2 on all full-attention layers | generated-token red at token 2 | `106.543 tok/s`, median `53.757 ms`; same token `220` vs `17` failure | The context-only split is not just missing a selected layer; the branch is correctness-red in the same way when applied everywhere. |
| Suffix native rowchunk2 after batch context+gate on selected full-attention layers `3,7,11,15,19,23,27,31` | generated-token red at token 9 | `106.864 tok/s`, median `53.609 ms`; batch token `12` vs c1 token `27` | Chunking only O/post/MoE after batch context+gate preserves the no-rowchunk failure shape. |
| Suffix native rowchunk2 including gate on selected full-attention layers `3,7,11,15,19,23,27,31` | generated-token red at token 9 | `107.508 tok/s`, median `53.189 ms`; batch token `12` vs c1 token `27` | Chunking gate/O/post/MoE after batch context also does not repair c6. |

The staged full-attention substage diagnostics now work with the selected-c1 MoE
bridge used by the server-visible c6 path. They did not repair no-rowchunk c6:
pre-QKV+append per-row preserves the token-9 failure, while per-row context
moves the failure to token 2. The context-only native rowchunk diagnostic also
rejects at token 2 for both selected-layer and all-layer scopes; compact summary
artifact:
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-context-rowchunk-rejects-summary.json`.
The suffix rowchunk diagnostic rejects at token 9 whether the gate stays batched
or is included in each chunk; compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-suffix-rowchunk-rejects-summary.json`.
The remaining full-attention rowchunk tax is not explained by O projection,
input/RMSNorm, QKV scratch, KV append, batch gate, context-kernel row chunking,
or post-context suffix chunking in isolation. The next useful split is lower-level
hidden/KV source instrumentation across the full native rowchunk boundary:
identify which pre-context producer state changes when the entire layer is
executed as rowchunk2 versus a single rows=6 native pass.

Next repair order:

1. Treat c2/c4/c6/c8 selected-c1 MoE plus c4/c8 all-layer full-attention
   rowchunk2, and c6 selected full-attention rowchunks
   `3,7,11,15,19,23,27,31` plus native-linear and forced small-batch shared
   expert, as the local equality starting point for direct retained sweeps and
   the opt-in runtime retained-default bridge.
2. Re-run natural c=8 server traffic after the selected-rowchunk repair if the
   scheduler forms c6 groups; the latest natural c=8 probe formed two c4 groups,
   so it did not exercise c6. Forced `n=6` now measures `50.434 ms/step` and
   `9.26 backend generated tok/s`.
3. Remove the remaining selected full-attention rowchunk blocker, or add an
   explicit scheduler grouping policy that avoids live c6 groups when a faster
   c2/c4/c8 path is available.
4. Add primitive correctness/profiler/baseline gates for any recovered shape
   before promoting a retained/default throughput claim.
5. Re-run the c=8 server diagnostic with a server-visible c6 repair or an
   explicit scheduler grouping policy, because the current server path naturally
   admits c6 live-row groups.

## PARO MTP/DFlash Buckets To Add Next

Add these buckets before porting GGUF verifier mechanisms:

| Bucket | What it should include |
| --- | --- |
| `draft_propose` | MTP/DFlash draft layer wall, selected experts, draft token generation, draft KV writes. |
| `metadata_upload` | Candidate token upload, position metadata, row maps, accepted-prefix metadata. |
| `target_verify_attention` | Target verifier attention layers, KV append/read, attention output copy. |
| `target_verify_moe_projection` | Target verifier dense projections, MoE gate/up/down, selected/shared experts. |
| `lm_head_top1` | LM-head, top-1/top-k sampler, logits readback if any. |
| `accept_summary` | Accepted-prefix scan, per-row accept counts, reject location, fallback decisions. |
| `commit_scatter` | Accepted-row hidden/KV/state commit, rejected-tail discard, draft/target state updates. |
| `graph_replay` | Replay launch wall and replay cache hit/miss per shape. |
| `host_sync_readback` | Explicit host synchronizations, D2H reads, Python/ctypes boundary waits. |
| `scheduler_wall` | Queue wait, batching window, slot admission/reclaim, streaming response assembly. |

## Non-Portable GGUF Wins

These should stay as reference evidence, not direct PARO tasks:

- GGUF `Q*_K`/Q8/dp4a/T16 rowtile kernel bodies.
- GGUF Q6_K rowtile verifier LM-head.
- llama.cpp compatibility precision trades.
- GGUF-specific no-copy GDN verifier capture state layout.
- GGUF MTP direct partial commit semantics, except as a speculative lifecycle
  pattern to compare against PARO verifier commit/rollback.
