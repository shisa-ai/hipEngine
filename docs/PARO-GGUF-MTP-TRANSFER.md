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
groups, and c6 needs both all-layer full-attention rowchunk2 and linear-attention
rowchunk2. c6 rowchunk3 is hidden-red, so the current safe c6 row group cap is
two rows.

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
| 6 | rowchunk2 on every full-attention layer, native linear attention, per-row c1 MoE on linear layers, serial LM-head, native/batch projection | Pass | `92.800 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-native-linear-linearmoe-perrow-full-rowchunk2-selected-c1-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, linear-attention rowchunk2, selected-c1 MoE, serial LM-head, native/batch projection | Pass | `87.612 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-full-linear-moe-selected-c1-nativeproj-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, linear-attention rowchunk2, selected-c1 MoE, serial LM-head, selected-c1 projection fallback | Pass | `84.709 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-full-linear-moe-selected-c1-selectedproj-serial-sampler-local-equality.json` |
| 6 | rowchunk2 on every full-attention layer, linear-attention rowchunk2, selected-c1 MoE, batched LM-head | Pass | `87.519 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-rowchunk2-full-linear-moe-selected-c1-batched-lmhead-local-equality.json` |
| 6 | per-row full-attention, per-row linear, selected-c1 MoE, serial LM-head | Pass | `69.802 tok/s` | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-p512-d128-linear-full-perrow-moe-selected-c1-serial-sampler-local-equality.json` |

Focused c6 hidden-bisect controls:

| Probe | Result | Artifact |
| --- | --- | --- |
| full-attention per-row, native linear batch segments, selected-c1 MoE | Hidden red at layer 2 / generated index 4 | `benchmarks/results/2026-07-09-hipengine-qwen35-c6-linear-rowchunk-hidden-bisect-summary.json` |
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

Focused c6 native-linear repair:

Compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-qwen35-c6-native-linear-perrow-moe-repair-summary.json`.

| Shape | Generated-token equality | Decode aggregate | Median step | Status |
| --- | --- | ---: | ---: | --- |
| old c6 bridge: full rowchunk2 + linear rowchunk2 + selected-c1 MoE | Pass | `87.612 tok/s` | `68.282 ms` | diagnostic correctness bridge |
| new c6 repair: full rowchunk2 + native rows=6 linear + per-row c1 MoE on linear layers | Pass | `92.800 tok/s` | `64.335 ms` | diagnostic correctness bridge |

Delta: `+5.92%` aggregate decode and `-3.947 ms` median decode step
(`-5.78%`) versus the linear-rowchunk bridge. The hidden-bisect says the
native rows=6 linear layer is bit-clean through `mlp_input`; the first drift in
the old native-linear probe was the batched selected-c1 MoE output. Fixing
`reserve_moe_c1_scratch(prefix=...)` to prefix `shared_out`, `moe_out`, and
the shared-rotate barrier also repaired the row-local MoE replay diagnostic.

The current retained-bench auto diagnostic path should therefore start from the
green local frontier:

- c2/c4/c6/c8: auto-select selected-c1 MoE.
- c2/c4/c6/c8: load the c-aware projection dispatch catalog; c6 now has a
  generated-token-green projection candidate.
- c4/c6/c8: auto-select full-attention rowchunk2 with an empty layer list, meaning
  every full-attention layer is rowchunked.
- c6: auto-select per-row c1 MoE on linear-attention layers and keep native
  rows=6 linear-attention segments; explicit linear-rowchunk env overrides keep
  the older rowchunk2 bridge available for bisection.
- c3/c5: keep the older selected-layer rowchunk diagnostic scope until
  local full 512/128 equality evidence replaces it.

Server relevance: the natural c=8 server diagnostic previously split into a
fast c2 group and a slow c6 group. Recovering c2/c4/c8 direct diagnostics helps,
and the c6 direct diagnostic now has a correct fallback shape. It is still a
diagnostic fallback, not a retained/default throughput claim, because it relies
on selected-c1 MoE and rowchunked linear/full attention and is missing the
primitive/profiler/baseline retained gates.

Runtime retained-default recheck:

| Probe | Result |
| --- | --- |
| Forced OpenAI `n=6` server request, code_python, max_tokens=128, batch window 200 ms, native decode + startup warmup + retained defaults, before the per-row-linear-MoE repair | rows=6 used `moe_decode_path=selected_c1_batch`, `linear_attention_decode_path=native_batch_row_chunks`, `linear_attention_row_chunk_size=2`, `full_attention_decode_path=native_batch_row_chunks`, `full_attention_row_chunk_size=2`, and `linear_attention_projection_path=native_batch`; artifact `benchmarks/results/2026-07-09-hipengine-paro-server-ar-mtpbench-code-python-n6-bw200-native-decode-warmup-retained-defaults-c6repair.json`. |
| Timing vs prior unrepaired c6 server probe | Decode step changed `56.873 ms -> 64.101 ms`; aggregate backend generated throughput changed `8.67 -> 8.14 tok/s`. This confirms the server-visible correctness shape, but not a speed win. |
| Runtime default after the per-row-linear-MoE repair | `HIPENGINE_QWEN35_RETAINED_BATCH_DEFAULTS=1` now selects per-row linear MoE for rows=6 when selected-c1 MoE is active and linear-MoE/linear-rowchunk env overrides are blank. A server rerun is still pending; the direct retained-bench local equality row above is the current evidence. |

c6 server splitter:

Compact summary artifact:
`benchmarks/results/2026-07-09-hipengine-paro-server-ar-c6-linear-moe-split-summary.json`.

| c6 server shape | MoE | Linear attention | Decode step | Backend generated | Status |
| --- | --- | --- | ---: | ---: | --- |
| grouped native-linear | grouped-compact | native segments | `56.873 ms` | `8.67 tok/s` | hidden-red timing diagnostic |
| selected rowchunk repair | selected-c1 | rowchunk2 | `64.101 ms` | `8.14 tok/s` | current server-visible correctness bridge |
| selected native-linear | selected-c1 batch | native segments | `51.619 ms` | `9.01 tok/s` | fastest prior timing, but hidden-red because rows=6 selected-c1 batch MoE drifts |
| selected native-linear + per-row linear MoE | per-row c1 on linear layers | native segments | pending server rerun | pending | direct retained-bench generated-token green at `92.800 tok/s` |
| grouped rowchunk | grouped-compact | rowchunk2 | `71.877 ms` | `7.59 tok/s` | hidden-red and slower |

This split says the biggest c6 tax was linear rowchunk2, not projection. The
direct repair shows native rows=6 linear-attention segments are correctness-clean
when linear-layer MoE is replayed as true per-row c1. The remaining server work
is to rerun the forced `n=6` probe with the new retained default and then decide
whether to keep the per-row linear MoE bridge, repair rows=6 selected-c1 batch
MoE, or avoid live c6 groups in scheduling.

Next repair order:

1. Treat c2/c4/c6/c8 selected-c1 MoE plus c4/c6/c8 all-layer full-attention
   rowchunk2, and c6 native-linear plus per-row linear MoE, as the local
   equality starting point for direct retained sweeps and the opt-in runtime
   retained-default bridge.
2. Re-run the c6 server probe with the new retained default. Direct local
   equality recovered `+5.92%` over the linear-rowchunk bridge; the server needs
   the same shape before c=8 natural traffic can be re-evaluated.
3. Repair rows=6 selected-c1 batch MoE if the per-row linear MoE bridge remains
   too expensive; it is now isolated as the source of the native-linear hidden
   drift.
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
