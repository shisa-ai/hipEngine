# hipEngine Refactor / Dead-Path Ledger

This file tracks cleanup work that should happen after the fast/correct path is
proven. During optimization, temporary flags and fallback paths are useful for
bisection; after the optimal path stabilizes, they become dispatch confusion and
should be removed or collapsed.

## Policy

- Exact, same-suite non-regressive performance wins should become defaults.
- Keep opt-out flags only while they are useful for rollback, bisection, or a
  named validation gap.
- When a flag is left in place, record the removal trigger here.
- Do not remove unfused numerical fallbacks required by `AGENTS.md`; remove dead
  runtime dispatch branches and stale experiment toggles first.

## Priority Cleanup (do first)

**Revalidate the gfx1151 GGUF graph default on the current stack and retire
unrelated legacy graph blocks.** SOL-G5 reintroduced a state-bound runtime graph
with a complete transition key. gfx1151 advertises its measured 128-step
admission; gfx1100 now advertises a separately measured 24-step admission after
passing all 24 hidden/GDN/KV/token transitions on W7900. Non-streaming c1 greedy
generation uses the graph only at each backend's admitted horizon, with
`HIPENGINE_GGUF_DECODE_GRAPH=0` retained for rollback. The explicit
`qwen35_gguf_bench.py --graph-replay-decode` surface is required for the current
default decision, while `scripts/gguf_mtp_bench.py --target-graph-verify` /
`--target-graph-batched-verify` remains separate stale diagnostic plumbing.

SOL-G4 provides the correct comparison floor: clean p512/d128 eager is
`20.290 ms/token`, while a 24-step marker profile contains `18.402 ms/token` of
GPU kernels (`88.62%` of profiled host wall). SOL-G5's clean production route
at `7f611fe3` passed 128 launches and measured a capture-inclusive
`20.334 -> 20.311 ms/token` (+0.112% throughput) edge. The 2026-07-12 TheRock
HIP 7.15 refresh is still 128/128 exact but rejects the graph wall on both the
scalar parent (`20.5230 -> 20.5736 ms/token`) and wave/block candidate
(`20.4723 -> 20.5324`). Do not remove the rollback flag or broaden graph
admission. A separate scoped decision must either reproduce a current graph win
or restore eager as the gfx1151 production selector; the wave/block kernel
itself helps both routes and is not the cause of this graph-policy result.

The gfx1100 decision is independently strong: clean W7900 p512/d24 SOL-G5 at
`833921ce` passed 24/24 byte-exact transitions and measured capture-inclusive
`30.5364 -> 12.5139 ms/token` (**2.4402x**, five runs). Per-token recapture was
only `22.22 tok/s`; the retained route is one state-bound capture followed by 24
validated relaunches, not recapture. Keep the gfx1100 admission at 24 until a
shorter-horizon audit establishes a lower break-even.

## Cleanup Ledger

| Area | Debt | Current status | Removal trigger |
| --- | --- | --- | --- |
| Qwen tokenizer EOS discovery | The PARO generator falls back to looking up `<|im_end|>` and `<|endoftext|>` because `tokenizers.Tokenizer.from_file()` does not expose `generation_config.json` EOS metadata. | The fallback recognizes both Qwen EOS ids, preserves explicit scalar/sequence metadata first, and deliberately avoids unrelated model-family markers. | Remove the string lookup once the Qwen model plugin owns and supplies normalized BOS/EOS/PAD metadata from `generation_config.json` / `tokenizer_config.json` to generation and sampling. |
| PARO width-plan execution | The greedy generator and `scripts/qwen35_batch_retained_bench.py` each execute `BatchWidthPartitionPlan` groups and collect similar telemetry; `_generate_batch_sampled()` also remains as a diagnostic packed path. | SOL-P1's clean gfx1151 catalog rejects c2-c8 fully native execution. A later exact greedy-BF16 c2 hybrid bypasses the profile partition only for its gfx1151/context<1024 contract; all other shapes remain exact width-1 sessions. SOL-P2 proves those sessions preserve ragged sparse-slot state/KV/output identity through c8-to-c1 retirement. | Consolidate the generator/benchmark group loops and remove the diagnostic sampled packed route after the c2 hybrid is represented in a schema-v2 profile, broader native groups are accepted, and the shared scheduler telemetry ABI is stable. |
| PARO gfx1151 greedy decode graph | `capture_decode_graph()` remains available, but public greedy c1 consults a resident-session architecture policy before capture. | The 2026-07-13 gfx1151 p512/d8 graph/eager fixture gate rejected generated-token equality, and graph wall was slightly slower (`0.133842` vs `0.133048` s). gfx1151 therefore uses canonical eager resident steps; gfx1100 retains its previously validated graph path. | Repair and re-admit gfx1151 only after a current-stack graph/eager gate matches generated tokens plus hidden/Conv/GDN/KV state across long replay and improves capture-inclusive wall. If no repair is planned, move the rejected graph probe to a diagnostic-only harness and keep eager as the permanent gfx1151 policy. |
| PARO ragged packed prefill | Ragged compact slabs automatically use per-segment linear/full-attention prefill; `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR` and `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN` retain explicit bisection routes. | SOL-P2 found the packed segmented/varlen route changed the first ragged row's persistent state and, across all 40 layers, its generated tokens. `per_segment_ragged_exact` is the production-safe fallback; equal-length slabs retain `packed_segments`/`packed_varlen`. | Repair the ragged packed kernels, then remove the automatic fallback only after c8 ragged generated-token plus all-linear-state/all-full-KV identity passes independent c1 on gfx1151 and gfx1100 and an end-to-end prefill comparison is non-regressive. Remove the force flags after the repaired route survives one release window. |
| PARO short-prompt prefill | The public generator uses token-serial c1 steps when prompt tokens are fewer than `linear_conv_kernel_dim`; low-level `prefill_native()` remains strict. | Release fallback. A no-env gfx1151 wheel-path smoke generated from the one-token prompt `Hello`; normal prompts still use native prefill. | Replace the serial fallback only after a dedicated native short-prompt kernel matches c1 generated tokens plus recurrent/KV state for lengths 1-3. |
| GGUF public AR profile | `HIPENGINE_GGUF_DECODE_REPACK=0` remains a rollback opt-out; low-level WMMA/GEMV selectors remain available for benchmark bisection. | Release default: T16 decode-repack is on, and public generate/stream sessions pass the resolved backend plus `use_wmma_prefill=True` and `use_gemv_decode=True`. A no-env gfx1151 Q4_K_M smoke generated one token through `LLM(model)`. | Remove the decode-repack opt-out after one release window and a defaults-only gfx1100 refresh. Keep raw layouts only where a quant/kernel lacks a T16 fallback or a retained diagnostic requires them. |
| GGUF MTP server packed verifier | `_MTP_SERVING_TARGET_BATCH_MAX_SLOTS = 4` chunks c>N server target verification instead of sending all active slots to one packed target forward. | Default serving policy after the first packed verifier landing and the stream-draft/stream-verify follow-ups. c=2/c=4 packed target verify wins, but one 8-slot packed batch is a measured rejected regime (`11.58 tok/s`, `target_verify_batch_ms=63733.783`). The current c=8 stream path still chunks verify at 4 slots and reaches **52.18 tok/s**, with verifier still dominant (`slots_verify_phase_ms=12345.442`). | Remove or raise the cap only after rows>=16 packed verifier and resident-draft row-count/cold-slot behavior are tuned and a c=8 natural24 rerun beats the chunked stream path without correctness or latency regressions. |
| GGUF packed verifier GPU-event instrumentation | `HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS` records HIP events through `Qwen35GGUFResidentSession.verify_target_blocks_batch()` and compact-MoE leaves. | Default-off diagnostic. It exposed c=8 server verifier GPU leaves on 2026-07-05 but adds event overhead (`47.17 tok/s` in the compact-WMMA event run), so it is not a retained speed path. | Keep only while c>N MTP verifier tuning is active; remove or move behind a dedicated profiling helper once the packed verifier bottleneck is closed. |
| GGUF compact-WMMA no-read probe | `HIPENGINE_GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS` skips the `wmma_total` host scalar read for bounded small selected-row probes by launching compact WMMA with a conservative upper row count. | Default-off rejected diagnostic. Two c=8 natural24 reruns measured **52.05/51.96 tok/s** versus retained **52.18**, and timing attribution shifted into the later LM-head/sample drain rather than producing a wall win. | Remove unless a future compact-WMMA body can consume bounded row counts without extra padded work and beats the retained c=2/c=4/c=8 server rows. |
| GGUF selected-WMMA launch-bounds tuning | `HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS` remains an R&D build flag for selected-WMMA kernels. | Default unchanged after the 2026-07-05 c>N server probe. `=2` was flat at c=8 (**52.55/52.23 tok/s**); `=4` helped c=8 (**53.22/53.44**) but regressed c=4 (**49.20/49.04** vs retained **49.65**), so no default promotion. | Keep as kernel R&D only; do not promote without a c=2/c=4/c=8 same-protocol rerun that is non-regressive at every concurrency. |
| GGUF AR server packed decode | `HIPENGINE_GGUF_AR_PACKED_DECODE` is a default-on opt-out around decode-shaped packed resident target passes for c>N GGUF greedy AR serving. | The 2026-07-13 lifecycle repair keeps packed state canonical but runs linear attention through c1-exact per-slot state slices; full attention and MoE/FFN remain batched. Deferred flush now copies the full live KV prefix. Retained-flag steady c4 and sparse c4→c1 are token/Conv/GDN/live-KV byte-exact. The July 5 fused-state-commit throughput rows describe the prior token-only route and are historical, not an exact-state baseline. | Keep the opt-out until per-layer hidden, live cancellation/admission, profiler, and repeated exact-accounting c=1/2/4/8 gates pass. Then keep scalar/stream fallback only for unsupported shapes. |
| GGUF AR server packed prefill | `HIPENGINE_GGUF_AR_PACKED_PREFILL` is a default-on opt-out around packed final-row prompt prefill for c>N GGUF greedy AR serving. | Packed linear/MoE stays multi-row. Full attention now uses row-span paged prefill below the AOTriton threshold and slot-local c1 math when a long prompt crosses it; the latter forces exact cache reimport before packed decode. Steady c4 and ragged `[512,64,64,64]` are token/Conv/GDN/live-KV exact. The old **65.91/82.41/63.17 tok/s** rows predate state/KV and exact-ID accounting and remain historical diagnostics. | Keep the opt-out until broader API/cancellation coverage and repeated exact-accounting c=1/2/4/8 plus profiler evidence are green; retain fallback for slabs beyond the packed hidden-row guard. |
| GGUF MTP server packed prefill | `HIPENGINE_GGUF_MTP_SERVER_PACKED_PREFILL` is a default-on opt-out around packed prompt prefill for eligible c=2/c=4 GGUF MTP serving batches. | Default-on after the 2026-07-06 steady-state natural24 rerun. The path reuses packed prompt rows and returns FP32 prompt hidden rows for MTP catch-up, moving server MTP **46.75/49.65/52.18 -> 59.94/66.60/54.88 tok/s** at c=2/c=4/c=8. It keeps the four-slot safety cap: c=8's first wave still uses serial prompt open and only the trailing c=2 wave uses packed prefill. Startup now warms hidden-seed packed prefill at widths 2/4 when MTP serving is enabled, moving fresh c=2 to **56.59 tok/s** and warm c=2/c=4 to **59.71/65.57 tok/s**. | Keep the opt-out until one more c=2/c=4/c=8 rerun confirms the default. Do not remove the four-slot cap until c=8 full packed prefill is non-regressive; pool-filling eight startup slots was rejected (**35.25 tok/s** c=8 rerun, **76.5 GiB** used). |
| GGUF MTP server startup warmup | `HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP` is an internal server-scoped env marker set only during startup scratch probing when `--speculative-mtp-serving` is not `off`. | Added after the 2026-07-06 cold-start audit. It lets the GGUF backend warm MTP hidden-seed packed prefill plus one tiny packed verifier at supported widths 2/4 without changing the generic `prepare_request_scratch(...)` hook signature. It removes the worst c=2 first-request MTP cliff but deliberately does not attempt unsupported width-8 packed prefill. | Replace this env handoff with an explicit backend scratch-preparer option if the startup hook grows typed capabilities. Keep it while MTP serving is opt-in/auto and c=2/c=4 cold-start evidence remains positive; remove or narrow it if startup memory/time becomes a production blocker. |
| GGUF MTP server deferred verifier scatter | `HIPENGINE_GGUF_MTP_SERVER_DEFER_VERIFY_SCATTER` is a default-on opt-out around delaying packed target verifier state scatter until after the accept decision. | Default-on after the 2026-07-06 resetfix rerun. It keeps owner-side packed verifier state live and commits only accepted hidden/full-attention KV/linear-state rows, moving retained no-env natural24 MTP **70.06/77.29/76.46 -> 70.53/78.76/79.61 tok/s** at c=2/c=4/c=8 with unchanged economy (`draft=165`, `accepted=141`, accept rate **0.8545**, **250** target rows). Reset now invalidates packed verifier/decode session metadata so startup verifier warmup cannot leave stale packed KV write-position bookkeeping for the first real request. | Remove the env opt-out after one more no-env c=2/c=4/c=8 server rerun plus API tests show no regression. Keep the eager-scatter branch only while it is useful for bisecting packed verifier state lifecycle bugs. |
| GGUF AR server stream decode | `HIPENGINE_GGUF_AR_STREAM_DECODE` is a default-on fallback/opt-out around per-slot HIP stream decode for c>N GGUF greedy AR serving when packed AR decode is disabled or unavailable, and now also gates parallel stream execution of packed decode chunks when c>4. | Superseded as the scalar headline AR route by packed decode, but retained for the packed c=8 chunk-stream path. The older scalar-stream route moved c=2/c=4/c=8 AR **41.17/41.45/41.42 -> 44.20/46.69/47.70 tok/s**; packed chunk streams then moved retained c=8 **56.35 -> 59.17 tok/s** while c=2/c=4 stayed flat. | Remove the env opt-out after packed AR decode is stable, or keep it only as a test/bisection fallback for packed-shape failures and c>4 chunk-stream bisection. |
| GGUF AR server stream prefill | `HIPENGINE_GGUF_AR_STREAM_PREFILL` is a default-off diagnostic around launching AR prompt prefill and top-1 sampling on each slot's decode stream before the stream-decode loop. | Rejected on the 2026-07-05 natural24 c=8 rerun: AR stream prefill measured **47.44 tok/s**, below the retained stream-decode baseline **47.70 tok/s**. It also raised prompt-prefill wall (`prefill_stream_batch_ms=15076.522`, `prefill_ms=8665.634`) on the short natural24 workload, so concurrent prefill appears to contend more than it helps. | Remove the env and async prefill plumbing unless a longer-prompt c>N sweep or a true batched prefill implementation proves it non-regressive versus the default stream-decode path. Do not promote this path without a c=2/c=4/c=8 rerun that beats retained AR. |
| GGUF MTP server stream draft | `HIPENGINE_GGUF_MTP_SERVER_STREAM_DRAFT` is a default-on opt-out around per-slot HIP stream draft proposal for c>N GGUF MTP serving. | Default-on after the 2026-07-05 natural24 server run. It moves packed-verifier c=2/c=4/c=8 MTP **45.57/47.48/47.18 -> 46.75/49.65/48.72 tok/s** and beat the then-current stream-AR rows by **1.058x/1.063x/1.021x**; the later packed-AR route supersedes that same-server AR comparison, so current MTP is below AR. It currently creates a `ThreadPoolExecutor` per draft phase and the c=8 wall remains verifier-heavy. | Remove the env opt-out and/or replace the per-cycle executor with a persistent scheduler after c>N server reruns and profiler evidence show the stream path is non-regressive. Keep the opt-out while tuning verifier wall and thread-pool overhead. |
| GGUF MTP server stream verify | `HIPENGINE_GGUF_MTP_SERVER_STREAM_VERIFY` is a default-on opt-out around running independent chunk-4 packed target verifier chunks on separate owner-session streams at c>N. | Default-on after the 2026-07-05 natural24 c=8 rerun. It keeps the four-slot verifier cap but overlaps the two c=8 chunks, moving warm c=8 MTP **48.72 -> 52.18 tok/s** and `slots_verify_phase_ms` **15354.902 -> 12345.442**. c=4 remains on the single-chunk path. | Remove the env opt-out after c=2/c=4/c=8 reruns and profiler evidence show the stream verify path is non-regressive; replace it if a tuned rows>=16 verifier or true batched scheduler supersedes chunk-stream overlap. |
| GGUF 24GB capacity diagnostics | `HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING` offloads the Q8_0 token embedding from device residency and performs exact host Q8_0→BF16 embedding copies. | Default-off diagnostic. It proves Q4_K_M `128K/128` can fit on GPU1 (`23.400 GiB` tracked / `23.913 GiB` sampled) but disables GGUF HIP decode graph replay, so decode falls to `11.141 tok/s`; not a promoted path. | Remove or demote to a one-off harness after a retained 24GB `128K/128` path keeps device-side graph-class decode, likely via GGUF INT8/full-attention KV or another device-side embedding/cache strategy. |
| GGUF INT8 KV diagnostics | GGUF accepts explicit `--kv-storage int8_per_token_head` for resident full-attention KV, reusing the PARO per-token/head INT8 write/decode kernels plus layer-local temporary BF16 prefill-oracle caches. Short contexts (`<=8192` rounded max positions) retain an additional BF16 mirror cache so primary short gates use exact BF16 decode while still exercising INT8 writes. Long contexts now default to `HIPENGINE_GGUF_INT8_KV_BF16_PREFIX_FULL_LAYERS=8` as a correctness fallback; lower prefixes, pure INT8, key-only (`HIPENGINE_GGUF_INT8_KV_KEY_ONLY=1`), block16 scale granularity (`HIPENGINE_GGUF_INT8_KV_BLOCK16=1`), and custom non-contiguous BF16 masks via `HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS` require `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` for diagnostics. | Default remains BF16 unless explicit INT8 KV is requested. GPU1 Q4_K_M pure INT8-only diagnostic fit and ran `128K/128` with graph-class decode (`760.724` prefill / `64.923` decode tok/s, `22.911 GiB` tracked / `23.472 GiB` sampled), but W7900 BF16-vs-INT8 no-mirror correctness rejects pure INT8 at `4K/1` (`KL=0.275781`, top-1 agreement `0.5`). The 2026-06-24 layer-local prefill-oracle fix shows the old prefix `3..8` prefill failures were partly a shared-oracle chunk-outer bug; current prefix `8` passes full `128K/128` (`KL mean=0.01448`, top-1 `0.96124`, no persistent BF16 mirror), while prefix `7` still fails `128K/16` top-1. Non-contiguous 3-INT8-layer masks that skip the known-sensitive full-attention layer 7 (`{6,8,9}` and `{5,8,9}` INT8) also failed `128K/16`, so no custom mask is promoted. The real HIP key-only diagnostic is primitive-correct, but prefix `0` fails `4K/1`, prefix `6` fails `128K/16`, and prefix `7` saves less memory than admitted prefix-8 per-token/head while raising prefill peak; no key-only path is promoted. The real HIP block16 diagnostic is primitive-correct too, but forced-long W7900 `4K/1` BF16-vs-block16 gates fail top-1 at prefix `0`, `6`, `7`, and `8`; no block16 path is promoted. Prefix 8 per-token/head is correctness-admitted but not a retained 24GB throughput row. | Remove the short BF16 mirror, BF16-prefix/custom-mask/key-only/block16 envs, and unverified-long env only after an all-INT8 or more compact calibrated KV format preserves GGUF BF16 logits at `4K` and `128K/128` long-context gates and completes a retained 24GB `128K/128` throughput benchmark. |
| GGUF selected-prefill diagnostics | `HIPENGINE_GGUF_T16_DS4_PREFILL` guarded runtime route for resident `gguf_q4_k_t16_v1` DS4/Q8_1 selected-prefill. | Default-off diagnostic. Full-model Q4_K_S GPU1 gate showed useful prefill speed (`1833.185 -> 1989.578 tok/s` at `512/128`, `2159.561 -> 2372.228 tok/s` at `4K/128`) but changed final token IDs versus default (`220/570 -> 3241/1510`) and added `+0.070 GiB` opt-in activation scratch. The scratch is allocated only when the flag is enabled, so default memory/IDs remain unchanged. | Remove or demote to a microbench/test-only path unless a later exact-enough Q8_1/DS4 calibration path preserves default final IDs/logits on `512/128`, `4K/128`, and the `128K/128` promotion gate while keeping memory bounded. |
| GGUF selected-prefill diagnostics | Microbench-only raw-Q4_K/Q8_1 selected-prefill variants in `gguf_q4_k_q8_1_selected_prefill` (`q8-1-dot`, `q8-1-ds4-dot`, `q8-1-ds4-wmma`, `q8-1-ds4-wmma32`, `q8-1-ds4-wmma64`, `q8-1-ds4-preview-wmma32`, `q8-1-ds4-wmma32-ldspack`, and rejected `q8-1-ds4-wmma32-lds`). | Diagnostic-only, not model runtime defaults. The 2026-06-16 DS4 WMMA path is useful as a fragment/math reference. Expanded-Q4 LDS staging regressed `8.210 -> 18.257 ms/call`; packed-Q4 LDS staging recovered to `11.438 ms/call` but still lost to raw WMMA32; the pre-unpacked preview path measured `12.020 ms/call` with higher fixture memory; four-wave WMMA64 was only flat/sub-1% better than WMMA32. These same-shape staging/pre-unpack/independent-wave probes should not become dispatch paths without a later shared-tile reuse win. | After a real GGUF MMQ/T16 prefill path is promoted or the llama.cpp parity detour closes, demote negative variants to tests or remove them from the microbench to avoid a permanent zoo of rejected kernels. |
| Qwen3.5/PARO INT8 KV prefill | `HIPENGINE_QWEN35_INT8_PREFILL_ATTENTION`, `HIPENGINE_QWEN35_INT8_PREFILL_STREAMING_MIN_TOKENS`, `HIPENGINE_QWEN35_INT8_PREFILL_LOW_MEMORY_TOTAL_GIB`, and `HIPENGINE_QWEN35_INT8_PREFILL_ORACLE_RESERVE_MIB` gate direct streaming INT8 prefill after the 2026-06-15 GPU1 sweep found a severe 128K prefill regression. | Default `auto`: use the fast temporary BF16-oracle/AOTriton bridge unless prompts are at least `224Ki` rows **and** device/meminfo pressure says the oracle is unsafe (`total <= 26 GiB` or free memory cannot cover oracle bytes plus a 1 GiB reserve). This keeps W7900/Strix-style larger-memory runs on the fast path while preserving the 24GB 262K scratch win. Direct streaming is not throughput-promoted (`1020.723 -> 23.425 tok/s` at 128K/128). | Remove the gate after a memory-safe fast INT8 prefill path matches the BF16-oracle/AOTriton speed envelope at `512/128`, `4K/128`, and `128K/128` while retaining the 262K no-oracle memory gate; otherwise demote direct streaming to a dedicated diagnostic/scratch-probe path. |
| Qwen3.5/PARO native sampler | `HIPENGINE_QWEN35_NATIVE_SAMPLER=0` opt-out around the promoted scoped PARO native sampler route. | Default-on for supported c=1 PARO sampled requests and scheduler-owned c>N serial per-slot rows when every row is covered by `supports_native_gpu_sampling`; host fallback remains explicit for GGUF, true batched c>N, `top_logprobs`, and unsupported processors/filter combinations. | After the next retained multi-shape W7900 sampler gate plus profiler smoke passes with defaults-on, remove the env opt-out or demote it to a test-only rollback hook. |
| GGUF MoE decode graph | `HIPENGINE_GGUF_MOE_GRAPH` opt-in around per-layer rows==1 MoE FFN capture/replay (`hipengine/runtime/moe_graph.py` + `_run_decode_layer_graphed`). | Default-off. Proven bit-exact on 35B-A3B Q4_K_M (KL=0, 40 captures / 3800 replays / 0 rejects). Cuts the FFN ~64% in launch count (~440->40/token) but multi-trial (3x32 steps) shows a **consistent ~0.84% wall regression** (eager 18.12 vs graph 18.27 ms/step, non-overlapping p10-p90 bands) — NOT noise. The host was already ahead of the GPU (decode is bandwidth/compute bound), so removing launches reclaims no wall and the graph launch overlaps marginally worse with the surrounding eager attention. Fails the non-regressive gate, so NOT promoted. Kept as the validated A/B lever proving launch-count is not the decode bottleneck. Artifact `benchmarks/results/2026-06-28-moe-graph-rows1-ab.json`, WORKLOG 2026-06-28. | Remove the flag + `MoeGraphCache` wiring once the bandwidth-bound conclusion is accepted and no rows=4 verifier-graph follow-up is planned. Re-A/B (and only then reconsider default-on) if a future per-GEMV bandwidth cut shifts the bottleneck back to host-dispatch and makes the launch saving net-positive on wall. Keep `moe_graph.py` + its unit gate as a reusable tool even if the runner wiring is removed. |
| GGUF row-compact MoE GEMV | `HIPENGINE_GGUF_ROW_COMPACT_GEMV` opt-in around `_try_run_post_attention_moe_rows_compact_gemv` for rows>1 selected-MoE verifier blocks. | Default-off. Rechecked on 2026-07-01 against the current llama-compat B2 dp4a all-sync smoke after direct-state cleanup. It regressed badly: B2 **36.05 tok/s**, cycle **27.765 ms/output**, `target_block_verify_total` **24.277 ms/output**; the new `target_block_linear_attn_ffn_moe_compact_gemv` bucket alone cost **8.977 ms/output**. Current split selected-MoE GEMVs are faster for this verifier shape. Artifact `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-rowcompact-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the runtime env gate or demote the compact row-GEMV path to tests/microbench-only unless a new compact scheduler/kernel beats the split selected-MoE path on a full-suite `llama-compat-device-chain-dp4a` B2 run with unchanged acceptance. |
| GGUF dense Q8 dp4a sidecar | `HIPENGINE_GGUF_Q8_0_RAW_SIDECAR` materialization sidecar plus `HIPENGINE_GGUF_DENSE_Q8_DP4A` / `--verify-dense-q8-dp4a` and `HIPENGINE_GGUF_DENSE_Q8_DP4A_ALL` / `--verify-dense-q8-dp4a-all`, routed by `_try_launch_dense_q8_pair_dp4a`, `_try_launch_dense_q8_single_dp4a`, and `_try_launch_dense_q8_triple_dp4a` for rows>1 verifier blocks. | Default-off. Added for the llama.cpp replication lane. The original route paid a q8_1 quantize launch plus two singleton dense Q8 GEMV launches and lost on B2 smoke; the rowtile-pair retry improved smoke/all-sync verifier timing but full-suite regressed **60.36 -> 59.42 tok/s**, cycle **16.587 -> 16.852 ms/output**, acceptance **0.583 -> 0.559**, target rows/output **1.250 -> 1.322**, and verifier drain **13.023 -> 13.093 ms/output**. The broader all-sidecar route adds raw singleton and Q/K/V triple wrappers and cuts the block profile dense-Q8 bucket **11.420 -> 8.902 ms/block** / kernel **26.053 -> 23.427 ms/block**; full-suite improves speed **60.36 -> 60.89 tok/s** and verifier drain **13.023 -> 12.742 ms/output**, but acceptance regresses **0.583 -> 0.567** and draft acceptance **0.700 -> 0.655**. Later retained lanes add Q8 shared dual, X8 draft lm-head, and F32 `ssm_out` on top of this all-sidecar base. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-{smoke,full}.json`, and earlier denseq8 rowtile-pair artifacts, WORKLOG 2026-07-01. | Keep only as part of the named accuracy-traded llama-compat route while parity work is active. Remove loose env/bench/suite variants after the current llama-compat audit unless a true llama-style Q8 layout/scheduler beats the active compat lane on the full suite, or unless the compat acceptance contract is explicitly changed. |
| GGUF verifier F32 dense-Q8 dp4a diagnostic | `HIPENGINE_GGUF_DENSE_Q8_DP4A_F32` / `--verify-dense-q8-dp4a-f32` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm{,-allsync}` route rows>1 direct-state F32 `ssm_out` through F32 q8_1 quantization plus the raw-Q8 dp4a singleton body. | Default-off globally; retained only for the accuracy-traded llama-compat lane. Isolated block profile moved host **32.470 -> 30.936 ms/block** and kernel **23.893 -> 22.881 ms/block**; same-session smoke moved **70.74 -> 71.43 tok/s** with identical acceptance; full-suite B2 moved **61.31 -> 63.63 tok/s**, cycle **16.331 -> 15.735 ms/output**, verifier drain **12.662 -> 12.158 ms/output**, acc/output **0.567 -> 0.578**, and target rows/output **1.299 -> 1.266**. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-f32ssm.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-f32ssm-{control-smoke,smoke,full}.json`, WORKLOG 2026-07-01. | Keep only as part of the named compat route while the safe verifier transaction gap is audited. Do not promote to exact default unless an exact/non-regressive replacement exists. Collapse this flag behind the final named compat route or remove it during post-compat cleanup if a later verifier rewrite supersedes direct F32 q8_1/raw-Q8 dp4a. |
| GGUF verifier shared-Q8 dp4a diagnostic | `HIPENGINE_GGUF_DENSE_Q8_DP4A_SHARED` / `--verify-dense-q8-dp4a-shared` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-sharedq8{,-allsync}` route verifier shared-expert `ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp` through the raw-Q8 q8_1/dp4a helpers. | Default-off and rejected on the then-active llama-compat B2 lane. Isolated block profile moved kernel time **23.893 -> 23.648 ms/block** and smoke improved **70.64 -> 71.66 tok/s**, cycle **14.181 -> 13.978 ms/output**, verifier drain **11.377 -> 11.183 ms/output** with identical smoke acceptance. Full-suite rejected it: then-active `denseq8all-x8top1` **61.31 tok/s**, cycle **16.331 ms/output**, acc/output **0.567**, target rows/output **1.299**, verifier **12.662 ms/output**; sharedq8 **59.63 tok/s**, cycle **16.793 ms/output**, acc/output **0.556**, target rows/output **1.333**, verifier **13.038 ms/output**. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-verifier-rocprof-llama-compat-block-b2-denseq8all-x8top1-{refresh,sharedq8}.json` and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-sharedq8-{control-smoke,smoke,full}.json`. | Remove the env/bench/suite route during post-compat flag cleanup unless a later fused shared-expert body or launch-collapsed shared route beats the active compat lane on the full suite with unchanged acceptance/economy. Do not promote this per-projection q8_1/dp4a shared path. |
| GGUF resident MTP draft Q8 shared dual | `HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL` opt-out around the default-on raw-Q8 dual F32/F32 GEMV for resident draft shared gate/up projections. | Default-on. Added 2026-07-01 for the llama-compat lane and exact resident draft path. It is bit-exact vs two single `gguf_q8_0_gemv_f32_f32_out` launches (`tests/test_gguf_k_gemv.py::test_q8_0_dual_f32_matches_two_single_gemvs`). Draft rocprof A/B reduced `gguf_k_prefill_out` from 16 -> 12 calls/cycle and added `gguf_k_dual_prefill_out` 2 calls/cycle; same-session smoke improved **69.44 -> 70.20 tok/s** with identical acceptance, and full-suite llama-compat improved **60.96 -> 61.19 tok/s** with unchanged acceptance/economy. Artifacts `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q8shared-{control,dual}.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-q8shareddual-full.json`, WORKLOG 2026-07-01. | Remove the opt-out branch and make the dual call unconditional after the next full-suite default exact and semantic-safe llama-compat parity reruns stay non-regressive, unless a later draft rewrite supersedes the shared-expert path. |
| GGUF resident MTP draft dense-Q8 dp4a stage selector | `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` plus `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A_STAGES` / `--resident-mtp-draft-dense-q8-dp4a-stages` route resident draft F32 dense projections through F32->q8_1 plus raw-Q8 dp4a float-output wrappers by stage. | Default-off globally and retained only in the accuracy-traded llama-compat lane with `stages=draft`. The legacy all-stage route, including initial KV seeding stages, regressed full-suite B2 **64.41 -> 64.14 tok/s** with worse acc/output and target rows/output. The draft-only selector preserved row economy and moved the then-active unsafe direct-state compat row **74.39 -> 75.15 tok/s**, cycle **13.463 -> 13.325 ms/output**, and `draft_initial` **2.204 -> 2.066 ms/output** with unchanged acc/output **0.621**, draft acceptance **0.820**, and target rows/output **1.136**. That performance row is now superseded as an exact-state claim. The current llama-style directcommit replication row is **60.56 tok/s**, cycle **16.534 ms/output**, verifier drain **14.071 ms/output**, replay/commit **0.043 ms/output**, target rows/output **1.172**, and zero replay rows; the serial-state exact control remains **51.85 tok/s** / **19.308 ms/output**. Artifacts `benchmarks/results/2026-07-02-ar-mtp-llama-compat-draftdenseq8-draftonly-full.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`, `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`, `benchmarks/results/2026-07-02-gguf-mtp-draft-rocprof-llama-compat-b2-draftdenseq8-draftonly-gpuevents.json`, WORKLOG 2026-07-02. | Keep only as part of the named compat route while the llama-replication lane is under parity audit. Do not treat the unsafe 75.15 row as a cleanup/promote trigger. Collapse/remove the selector after the final compat route is settled or a verifier/draft rewrite supersedes this route. |
| GGUF resident MTP draft Q6 top-1 X8 sidecar | `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8` / `--resident-mtp-draft-q6-top1-stage1-shape x8` plus suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1{,-allsync}` for an X8-packed Q6_K draft lm-head top-1 sidecar. | Default-off globally; retained only for the accuracy-traded llama-compat lane. It materializes `output.weight[:vocab]` into contiguous groups of eight GGUF Q6_K rows and routes the q8_1/dp4a top-1 stage1 through `gguf_q6_k_x8_gemv_q8_1_dp4a_top1_stage1`. Correctness passes against the q8_1/Q6_K oracle. Same-session smoke improved **71.53 -> 71.76 tok/s** with identical acceptance; draft rocprof moved stage1 **3.603 -> 3.558 ms/cycle**; full-suite compat moved **61.19 -> 61.31 tok/s**, cycle **16.364 -> 16.331 ms/output**, and `draft_initial` **3.378 -> 3.352 ms/output** with unchanged acceptance/economy. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-denseq8all-x8top1-{control-smoke,smoke,full}.json`, `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8top1.json`, WORKLOG 2026-07-01. | Keep only as part of the named llama-replication route while the safe verifier transaction gap remains under analysis. Remove/demote the X8 sidecar and route variants if a later fused draft lm-head/sampler or different Q6_K body/layout supersedes it, or if parity closure decides the accuracy-traded llama-compat lane should not retain separate draft lm-head sidecars. Do not promote to exact default without exactness/full-suite correctness evidence. |
| GGUF selected-down X8 repack | `HIPENGINE_GGUF_SELECTED_X8_REPACK` materialization gate plus bench flag `--selected-down-x8-repack {off,q5,q6,both}` for Q5_K/Q6_K selected-down X8 q8_1/dp4a replacement layouts. | Default-off globally. Retained only for the accuracy-traded llama-compat B2 lane with `q6`; first-class suite route is `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6`. Full suite **59.63 -> 60.36 tok/s**, `cycle_wall_ms_per_output` **16.793 -> 16.587**, and `target_block_verify_total` **13.178 -> 13.023 ms/output**. q5/both remains rejected for that route (`64.81 tok/s` smoke vs q6-only `69.03 tok/s`), so Q5_K selected-down stays on T16. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-full.json`, `...x8q6-allsync-smoke.json`, route smoke `...x8q6-route-smoke.json`, and `benchmarks/results/2026-07-01-llama-compat-b2-x8-selected-down-dp4a-current-micro.json`. | Remove/demote q5/both materialization from performance paths unless a future full-suite route beats q6-only with unchanged acceptance. Do not promote to exact default without exactness/full-suite correctness evidence. Once the compat lane is final, consider collapsing the env gate behind the named route and leaving raw env use to tests/microbenches. |
| GGUF selected gate/up X8 repack | `HIPENGINE_GGUF_SELECTED_GATE_UP_X8` materialization gate plus bench flag `--selected-gate-up-x8` and suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup{,-allsync}` for Q4_K selected gate/up X8 q8_1/dp4a replacement layouts. | Default-off and rejected on the current retained llama-compat B2 lane. Same-session smoke regressed **67.62 -> 59.08 tok/s**, cycle **14.810 -> 16.948 ms/output**, and target verifier drain **12.005 -> 14.117 ms/output** with identical smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). All-sync attribution shows the loss is the selected gate/up GEMV body: linear-attn gate/up GEMV **1.408 -> 3.050 ms/output** and full-attn gate/up GEMV **0.462 -> 1.015 ms/output**, while q8_1 quantize is unchanged/slightly lower. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-x8gateup{,-control}-smoke.json` and `...x8gateup{,-control}-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the bench/suite route during the post-compat flag cleanup unless a different Q4 X8 scheduler/body beats retained T16 dp4a on the same async/full-suite route with unchanged acceptance. Future selected-MoE work should compare against llama.cpp `mul_mat_vec_q_moe` rather than broadening this X8 gate/up path. |
| GGUF selected gate/up raw materialization | `HIPENGINE_GGUF_SELECTED_GATE_UP_RAW` materialization gate plus bench flag `--selected-gate-up-raw` and suite routes `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup{,-allsync}` for keeping Q4_K selected gate/up experts in raw GGUF layout under decode-repack. With `--verify-dp4a`, the runtime uses the raw selected-dual q8_1/dp4a body instead of the retained T16 replacement-layout body. | Default-off and rejected on the current retained llama-compat B2 lane. Same-session smoke regressed **68.55 -> 62.04 tok/s**, cycle **14.612 -> 16.142 ms/output**, and target verifier drain **11.792 -> 13.328 ms/output** with identical smoke acceptance (`acc/output=0.667`, draft acceptance `1.000`). All-sync attribution shows the loss is the selected gate/up GEMV body: linear-attn gate/up GEMV **1.422 -> 2.153 ms/output** and full-attn gate/up GEMV **0.461 -> 0.729 ms/output**, while q8_1 quantize changes by only ~0.01 ms/output. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-rawgateup{,-control}-smoke.json` and `...rawgateup{,-control}-allsync-smoke.json`, WORKLOG 2026-07-01. | Remove the bench/suite route during the post-compat flag cleanup unless a new raw-GGUF scheduler/body beats retained T16 dp4a on the same async/full-suite route with unchanged acceptance. Do not retry a mechanical raw `mul_mat_vec_q_moe` body copy for selected gate/up; the measured path is slower than retained T16. |
| GGUF decode-graph rollback and benches | `HIPENGINE_GGUF_DECODE_GRAPH=0` keeps eager as a production rollback; `scripts/qwen35_gguf_bench.py --graph-replay-decode` remains an explicit measurement surface; `scripts/gguf_mtp_bench.py` still has separate `--target-graph-verify` / `--target-graph-batched-verify` modes. | gfx1100 defaults to the state-bound graph for non-streaming c1 greedy windows with at least 24 remaining transitions after clean W7900 p512/d24 SOL-G5 passed 24/24 state/token checks and moved capture-inclusive wall **30.5364 -> 12.5139 ms/token (2.4402x)**. gfx1151 retains its separate 128-step admission pending the current-stack policy recheck. Sampled/streaming/c>N/short/INT8-KV/host-embedding/per-layer-MoE-graph routes remain eager. MTP graph modes are not part of this result. | Keep the opt-out through one release window and the full natural-prompt gfx1100 refresh. Recheck gfx1151 separately; restore eager there if another balanced run remains negative. Remove or isolate stale MTP graph modes separately. |
| GGUF AR-baseline timing contract | `gguf_true_ar_category_bench.py` and `scripts/gguf_ar_mtp_suite.py` request state-bound graph admission and record the effective per-prompt route. | Repaired for backend-qualified production timing: gfx1100 horizons >=24 report `graph_replay` with capture/instantiate/close included; unsupported/short horizons honestly report `eager_step`. This replaces the stale unconditional eager artifact while preserving an explicit no-graph diagnostic. The older attachment validator still has a fixed graph-only requirement and does not yet model backend/horizon admission. | Teach the attachment validator to consume the recorded backend capability/effective route instead of one global graph-only constant, then remove duplicate suite-side protocol logic. Preserve anti-gaming rejection of raw/non-production timing. |
| MTP P1 verifier | `HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL` opt-out around the promoted split-output dual W4 shared-gate/up route. | Default-on after 2026-06-11 D32 9-prompt exact A/B: same acceptance, verify `22.98 -> 22.37 ms/cycle`. | After the next retained MTP gate with defaults-on passes at the target sprint shape, remove the opt-out or demote it to a test-only override. |
| MTP P1 verifier | `HIPENGINE_LINEAR_OUT_CAST_ROTATE_FUSED` opt-out around promoted `f32_to_fp16 + paro_rotate1` fusion. | Default-on after raw-bit RED test and 2026-06-11 D32 9-prompt exact A/B; removes 30 launches/pass and contributes to the stacked `-0.60 ms/cycle` suite delta. | After the next retained MTP gate with defaults-on passes, collapse the old runtime dispatch branch if no other path still needs it. |
| MTP P1 verifier | `HIPENGINE_SELECTED_MOE_DOWN_STAGED` opt-in around the superseded staged selected SiLU/down-rotate + down GEMV path. | Flipped default-off on 2026-06-11 after current graph-auto D32 9-prompt exact A/B: identical acceptance, cycle `27.648 -> 27.408 ms/cycle`, verify `22.377 -> 22.131 ms/cycle`. The 2026-06-12 graph-off current-best compound retest also stayed exact but regressed ratio `0.8252x -> 0.8204x`, cycle `21.661 -> 21.763 ms/cycle`, and verify `16.511 -> 16.628 ms/cycle`. The staged path remains available with `=1` for bisection and historical comparison. | After the next retained MTP gate with defaults-on passes, remove the staged runtime branch or demote it to a kernel test-only path unless a new barrier-free implementation beats the fallback. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_SKIP_UNUSED_READS` opt-out around skipped discarded expert-topk host reads, update-only lm-head/argmax results, and final draft snapshot saves. | Default-on after 2026-06-11 D32 9-prompt exact gates: same acceptance/visible tokens, read/result skip moved actual speed `0.664x -> 0.670x`, cycle wall `27.94 -> 27.68 ms`, proposal/update `2.145 -> 2.052 ms`; final-snapshot skip then stayed exact `9/9`, skipped `142` D2D snapshot saves, and trimmed proposal/update `2.052 -> 2.045 ms` with flat actual ratio within noise. | After the next retained MTP gate with defaults-on passes, remove the opt-out or demote it to a test-only override. Keep the functional code path; it is the desired proposer behavior. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION` opt-out around the packed token+position metadata H2D copy. | Default-on after 2026-06-11 same-tree D32 9-prompt exact A/B: exact `9/9`, identical acceptance, wall `26.922 -> 26.869 ms/cycle`, proposal/update `1.9766 -> 1.9758 ms/cycle`; ratio is noisy/down because the AR control changed. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep the packed one-copy metadata path. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_ROUTE0_ACCUM_INIT` opt-out around route-0 FP32 MoE accumulator initialization. | Default-on after 2026-06-11 D32 9-prompt exact A/B: exact `9/9`, identical acceptance, standalone `moe_accum` memset removed by route 0 overwrite, cycle wall `27.081246 -> 27.079143 ms/cycle`, proposal/update `1.96299 -> 1.95303 ms/cycle`; ratio is noisy/down because AR changed. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep route-0 accumulator initialization as the only proposer MoE accumulation path. |
| MTP P2 proposer | `HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE` opt-out around direct sidecar K/V cache writes. | Default-on after 2026-06-11 D32 9-prompt exact A/B: exact `9/9`, identical acceptance, K rotary and V projection producers write directly into cache slots instead of temp buffers followed by two D2D copies per advance, proposal/update `1.9955 -> 1.9801 ms/cycle`; total wall was flat/noisy-negative because verify moved independently. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep direct cache writes as the only proposer K/V materialization path. |
| MTP/DFlash verifier accept | `HIPENGINE_VERIFY_ACCEPT_PACKED_PAYLOAD` opt-out around the packed accept-summary D2H payload. | Default-on after 2026-06-11 MTP D32 9-prompt exact same-tree A/B: exact `9/9`, identical accepted lengths and active budgets, cycle wall `27.279 -> 27.122 ms/cycle`, verify `22.162 -> 21.997 ms/cycle`; packs seven tiny D2H reads into one int32 payload read while keeping the legacy output buffers for commit/compatibility. | After one follow-up defaults-only MTP gate and one DFlash chain smoke pass, remove the opt-out or demote it to a test-only override. Keep the packed payload path as the default verifier accept API. |
| MTP/DFlash verifier metadata | `HIPENGINE_VERIFY_PACK_DYNAMIC_METADATA` opt-out around the packed token/position/context metadata H2D path. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths and active budgets, actual ratio `0.68417x -> 0.68898x`, cycle wall `27.02196 -> 26.99252 ms/cycle`, verify `21.87984 -> 21.85918 ms/cycle`; replaces five tiny per-cycle H2D submissions with one packed int64 copy plus `unpack_verify_chain_dynamic_metadata_i64_kernel`. Rocprof confirms the kernel; a 27B dense DFlash D16 one-prompt shared-path smoke passed. | After one follow-up defaults-only MTP gate and the next full 27B DFlash hardening/defaults-only gate pass, remove the opt-out and keep the packed metadata path. |
| MTP/DFlash verifier commit | `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` opt-out around the chunked linear-state commit copy grid. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths and active budgets, verify `21.8518 -> 21.8308 ms/cycle`; rocprof moved `linear_state_pair_commit` `0.250 -> 0.203 ms/pass`, total verifier kernel `14.395 -> 14.341 ms/pass`, and host marker `18.301 -> 18.263 ms/pass`. Whole-cycle wall was neutral/noisy, so this is retained as a verifier sub-window micro-slice. A 27B dense DFlash D16 shared-path smoke passed. | After one follow-up defaults-only MTP gate and one DFlash chain smoke/defaults-only gate pass, remove the opt-out and keep the chunked 64 KiB commit grid. |
| MTP verifier host cache | `HIPENGINE_VERIFY_SCRATCH_CACHE` opt-out around fixed-shape verifier scratch object caching. | Default-on after 2026-06-11 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates, wall `27.0958 -> 26.7015 ms/cycle`, verify `21.9328 -> 21.5511 ms/cycle`, and actual ratio `0.6860x -> 0.6987x`. Graph-auto profile showed only a small steady replay host change (`18.290 -> 18.275 ms/pass`), while graph-off control showed the raw Python rebuild win (`33.469 -> 32.988 ms/pass`). | After the next retained MTP defaults-only gate passes with scratch, tensor lookup, and resident view caches enabled, remove the opt-out and keep the workspace-validated scratch cache as the only verifier scratch reservation path. |
| MTP verifier host cache | `HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE` opt-out around immutable model tensor lookup memoization on each decode state. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates, wall `26.6621 -> 26.6433 ms/cycle`, verify `21.5290 -> 21.4984 ms/cycle`, and actual ratio `0.69160x -> 0.69200x`. Graph-auto profile was neutral/noisy (`18.218 -> 18.236 ms/pass`), while graph-off isolated the raw Python lookup win (`34.757 -> 32.288 ms/pass`). | After the next retained MTP defaults-only gate passes, remove the opt-out or demote it to a test-only override; keep raw-name tensor lookup memoization as the default host path. |
| MTP verifier host cache | `HIPENGINE_RESIDENT_TENSOR_VIEW_CACHE` opt-out around resident non-owning Tensor view caching. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates and identical per-prompt accepted lengths, wall `26.6424 -> 26.4259 ms/cycle`, verify `21.5059 -> 21.2785 ms/cycle`, and actual ratio `0.69239x -> 0.69857x`. Graph-auto profile was neutral/noisy (`18.235 -> 18.244 ms/pass`), while graph-off isolated raw host improvement (`32.52 -> 31.70 ms/pass`). | After the next retained MTP defaults-only gate passes, remove the opt-out or demote it to a test-only override; keep cached `_slot_linear_state`, `_slot_full_cache`, and `_full_cache_all_slots` views as the default host path. |
| MTP verifier scratch policy | `HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED` opt-out around c1/grouped verifier MLP scratch selection. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates plus identical accepted lengths/active budgets, wall `26.3089 -> 25.6898 ms/cycle`, verify `21.1757 -> 20.5228 ms/cycle`, and actual ratio `0.7003x -> 0.7172x`. Graph-auto profile kept `932` calls/pass and moved host `18.314 -> 18.246 ms/pass`; graph-off host moved `32.445 -> 32.273 ms/pass`. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep verifier MLP scratch reservation aligned with `_verify_moe_grouped_min_tokens()` as the only path. |
| MTP verifier host cache | `HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP` opt-out around generation-stamped verifier scratch cache hits. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical visible/accepted cycle aggregates plus identical accepted lengths/active budgets, wall `25.7085 -> 25.5955 ms/cycle`, verify `20.5460 -> 20.4342 ms/cycle`, and actual ratio `0.7145x -> 0.7252x`. Graph-auto profile kept `932` calls/pass and moved host `18.322 -> 18.298 ms/pass`; graph-off host moved `32.659 -> 31.971 ms/pass`. | After the next retained MTP defaults-only gate passes, remove the opt-out and keep generation-stamped cache-hit validation as the only verifier scratch cache path. |
| MTP graph-off verifier scratch | `HIPENGINE_MTP_SKIP_CANONICALIZE_AFTER_VERIFY` opt-out around keeping verifier-shaped scratch live between MTP verify cycles. | Default-on after 2026-06-12 MTP D32 9-prompt exact A/B: exact `9/9`, identical accepted lengths/active budgets, graph-off batched wall `37.207 -> 24.076 ms/cycle`, verify `32.069 -> 18.933 ms/cycle`, and actual ratio `0.4969x -> 0.7730x`. Rocprof showed this is host-only cleanup: calls unchanged `932/pass`, kernel `14.332 -> 14.330 ms/pass`, host `32.505 -> 18.272 ms/pass`. The follow-on `decode_batched + graph_off + skip` row is current best at `0.8252x`, `21.661 ms/cycle`, `16.511 ms` verify. | After the next retained MTP defaults-only gate passes and the c1/AR handoff path has explicit coverage for `canonicalize_after=True`, remove the env opt-out or demote it to a test-only override; keep the `canonicalize_after` API only where handoff semantics need it. |
| MTP verifier rejected gate | `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` opt-in for fused full-attention Q/Gate split plus FP16-to-FP32 key cast. | Default-off; bit-exact GPU parity vs `qwen35_split_qgate_fp16 + fp16_to_f32`, exact quicksort, and profile-positive for launch count (`932 -> 922` calls/pass, host `18.269 -> 18.234 ms/pass`), but two exact 9-prompt D32 A/B pairs with identical acceptance regressed average wall/verify (`26.925 -> 27.010 ms/cycle`, `21.754 -> 21.828 ms/cycle`). | Remove the runtime gate or demote it to a kernel test-only path after the break-even sprint unless a broader full-layer composite reuses the kernel and beats the prompt-suite gate. |
| MTP/DFlash verifier rejected gate | `HIPENGINE_VERIFY_ACCEPT_UPDATES_POSITION` opt-in for writing resident base-slot position/context from the packed accept kernel. | Default-off; exact quicksort and D32 `9/9`, and the locked profile removed one scalar `set_decode_position_i64` launch/pass (`932 -> 931` calls/pass), but profile host window worsened (`0.2004 -> 0.2010 s` over 11 passes) and the prompt suite regressed wall/verify (`27.038 -> 27.135 ms/cycle`, `21.884 -> 21.969 ms/cycle`). | Remove the runtime gate or fold it into a broader accept/commit composite only if that composite beats the exact prompt-suite gate. |
| MTP/DFlash verifier LM-head | `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD` opt-in for fused W8A16 LM-head + argmax rows. | Default-off; 2026-06-11/12 MTP retests regressed, and clean gfx1151 SOL-S7 now reconfirms exact-but-slower DFlash (`9.676 -> 9.177 tok/s`, -5.16%). The fused body outweighs the saved argmax launch; synchronized accept-readback is only 0.042% of wall. | Removal trigger is satisfied. Demote the runtime branch to a kernel-test-only parity path in a scoped cleanup; do not expose it as a tuning option unless a new schedule beats unfused on gfx1100 and gfx1151. |
| MTP proposer/verifier overlap | `HIPENGINE_MTP_OVERLAP_VERIFY_COMMIT_PROPOSER` opt-in for running proposer update on a side stream while verifier commit drains. | Default-off; 2026-06-12 D32 9-prompt exact A/B kept identical acceptance and hid some verify/commit time (`16.166 -> 16.028 ms/cycle`) but regressed proposer/update more (`1.243 -> 1.438 ms/cycle`), moving wall `19.441 -> 19.506 ms/cycle` and ratio `0.9216x -> 0.9184x`. | Remove the harness flag after the current break-even sprint unless a broader proposer graph/update redesign makes the side stream positive on the exact prompt suite. Keep only generic stream plumbing if another retained path uses it. |
| MTP D64 state-drift diagnostics | `scripts/mtp_chain_e2e_smoke.py --ar-fallback-after-mtp-cycles` diagnostic override. | Default-off; added to bracket the D64 `translation` token-34 resident-state drift by forcing target AR after a fixed number of MTP verifier cycles. It is diagnostic evidence plumbing, not an acceptance policy. | Remove or move to a dedicated debug harness after the D64 target-state audit is fixed and artifacted, or if a per-layer state comparator supersedes forced-cycle bisection. |
| MTP GDN state-drift diagnostics | `HIPENGINE_GDN_CHAIN_TLOOP_VTILE` temporary env selector for chain GDN t-loop VTILE 1 vs retained VTILE 4. | Default path remains VTILE=4. `VTILE=1` helped localize the first accepted-row D64 handoff (`force_after=2/3` exact) but did not fix `force_after=4`, so it is not a promoted speed or correctness path. | Remove after the D64 chain GDN/materialized-state bug is fixed or after a narrower per-layer comparator identifies a different root cause. Do not leave this as a permanent user-facing tuning flag. |
| MTP verifier rejected gate | `HIPENGINE_MOE_FUSED_ROTATE` opt-in for M13.B.1 selected-dual rotate+GEMV. | Default-off; 2026-05-23 W7900 gate stayed exact and removed 40 rotate launches/pass, but regressed total kernel time `17.32 -> 29.76 ms/pass` because the fused kernel repeated the rotation per `(out_pack,row)` block. | Remove or demote to kernel test-only after break-even path stabilizes; only keep runtime access if a new non-redundant staged design replaces it. |
| MTP verifier rejected gate | `HIPENGINE_SELECTED_MOE_STAGED_ROTATE` opt-in for M13.B.3 staged selected gate/up rotate+GEMV. | Default-off; staged/keyed gate-up path is exact but later W7900 verifier-window measurement regressed kernel time (`15.344 -> 15.611 ms/pass`) despite launch-count reduction. | Remove or demote to kernel test-only unless a no-spin/no-barrier-reset design beats the unfused chain on the current D32 prompt suite. |
| MTP verifier rejected gate | `HIPENGINE_SHARED_EXPERT_FUSED_ROTATE` opt-in for M13.B.2 shared-expert rotate+dual GEMV. | Default-off; exact, but the saved rotate launch was replaced by a barrier reset in the original path and the keyed-barrier follow-up was neutral (`15.350 -> 15.365 ms/pass`). | Remove or demote to kernel test-only unless a future full-layer C-dispatch path can reuse it without adding per-launch synchronization overhead. |
| MTP verifier rejected gate | `HIPENGINE_FUSED_RMSNORM_ROTATE` opt-in for M15.4 fused input RMSNorm + PARO rotate2. | Default-off; current-stack retest on 2026-06-11 stayed exact but regressed verifier kernel `13.41 -> 14.09 ms/pass` and host window `18.45 -> 19.05 ms/pass`. | After the MTP break-even path is stable, remove the runtime gate or demote it to a kernel test-only path unless a new implementation avoids the one-block RMSNorm occupancy trap. |
| MTP verifier docs | Older "default-off diagnostic" notes for P1 gates can become stale as promoted defaults land. | `docs/MTP.md`, `benchmarks/README.md`, and `WORKLOG.md` carry historical rows plus current status. | During each MTP sprint commit, update current-status language and leave old measurements only as dated history. |
| GGUF decode graph replay | Session-bound graphs retain a full key and cumulative transition budget; callers can still explicitly capture diagnostic windows outside the production selector. | SOL-G5 resolves the old uncapped debt: the context cap is inferred from `position + max_replay_steps` or validated when explicit; the key covers backend/model/weights/buffers/KV/layers/route/sampler/recording/state generation; replay rejects cursor drift or budget overflow. Current gfx1151 HIP passes 128/128 byte-exact checkpoints. | Keep the strict cap/key checks permanently. Remove only redundant legacy graph arguments after the MTP diagnostic modes are retired and all callers use the state-bound API. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --no-target-graph-verify` eager target-verification opt-out. | Default path now uses capped resident decode-graph target verification with fp32 hidden-seed capture; the eager opt-out is useful for bisection and correctness/perf comparison but is >2x slower on the B1 full suite. | Remove or move to a dedicated debug harness once the capped target graph has survived the next MTP break-even sprint and a multi-token GGUF graph correctness gate is part of the regular validation bundle. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --no-mtp-draft-warmup` cold-start diagnostic opt-out. | Default path now runs one stateless untimed draft warmup so MTP timing matches the true-AR warmup protocol; the cold-start opt-out is useful only for measuring wrapper/library/weight-cache first-use cost. | Remove or move to a dedicated cold-start harness once the MTP draft runtime has persistent device buffers and per-process cold-start cost is documented separately from steady-state tok/s. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --root-topk-accept >1`, `--sibling-topk-accept >1`, and `--topk-branch-redraft` exact-verified coverage tree. | Default-off after the 2026-06-23 speed-first reset to B1 linear greedy: K4096 B5 raised accepted/output but regressed draft efficiency (`0.000227`) and tok/s versus B1 linear. Keep it only for coverage diagnostics and llama.cpp parity investigations. | Remove or move to a dedicated experiment harness if the real GGUF MTP verifier path/adaptive policy supersedes top-k tree diagnostics, or if repeated full-suite speed-first gates show the tree remains speed-negative. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-context-replay` slow default-off prompt-catchup replay path. | Useful to bracket llama.cpp-style MTP prompt catch-up while the resident bulk target path exposes only the final hidden row. Current smoke is acceptance-negative and should not be the default benchmark path. | Remove or move to a dedicated debug harness after bulk prefill exposes all-row fp32 hidden seeds and the real MTP KV/RoPE path is implemented, or if that path supersedes replay diagnostics. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-device-kv-cache` default-off device-resident MTP dense KV context. | Implemented for B1 llama.cpp-parity investigation: MTP attention writes post-RoPE K/V to persistent device buffers, draft steps attend over the cache, and accepted target rows use a KV-write-only commit path. Smoke is much faster than host replay/prefix diagnostics but still below the retained no-cache default, so it is opt-in only. | Promote to default only if a same-protocol full-suite row improves raw tok/s, accepted/output, and draft_acceptance; otherwise move to a dedicated debug harness or replace with a paged/KVLiveSpans implementation once bulk hidden-row capture lands. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-graph-batched-verify` full-accept-required verifier graph replay experiment. | Default-off; records generated target IDs and FP32 hidden seeds for a whole strict verifier block. The 2026-06-25 merge-sort B3 smoke stayed exact but was speed-neutral/slower because target kernels still execute sequentially inside the graph and hidden-row recording adds overhead. | Remove unless a true rollback-safe block verifier starts using the recorded hidden rows, or promote only after full-suite evidence shows a speed win over one-step graph replay without requiring prompt-specific full acceptance. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-verify[=bulk/native]` rollback-safe target continuation block verifier. | Default-off; snapshots linear-attention conv/recurrent state with persistent buffers, runs row-bulk target continuation over `[prev]+drafts`, records target IDs + FP32 hidden seeds, and restores/replays the consumed prefix on partial accept. Tiny B3/B5 verifier blocks now default to `--no-target-block-wmma-prefill` because selected/WMMA prefill regressed the B3+32k smoke (`37.8 tok/s`); the GEMV verifier fallback lifts the same smoke to `48.1 tok/s` with `15/15` accepts, but B5 partial rollback remains slow. | Keep as a correctness scaffold and small-B scheduler harness. Promote only if verifier `ar_decode_ms` beats one-step graph replay on full-suite/heldout without reducing acceptance; otherwise replace the linear-attention/rollback pieces with a dedicated small-B continuation kernel and do not re-enable selected-prefill for tiny verifier blocks without evidence. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-direct-state-commit` and suite routes `resident-strict-block-direct-commit` / `resident-hybrid-strict-block-direct-cap32k` / `resident-hybrid-strict-block-direct-native-cap32k` / `resident-b1-branch-safe-direct-cap32k-device-seed`. | Default-off diagnostic. It materializes per-row GGUF linear-attention Conv/GDN state during target block verification and commits captured verifier rows directly when the verifier mode is serial-exact/native, or default `bulk` with a short verifier block (`end < 1024`). Native mode is exact through row 1 after the row-serial full-attention fallback was fixed to use absolute continuation positions and capture row states. Bulk mode is exact through row 1 for short verifier blocks after the suffix full-attention path switched to a c1-exact row-batch decode context and the batch context kernel was fixed to honor shared physical block IDs. 2026-06-29 smoke is still not positive: pure strict B3 **37.20 tok/s = 0.678x AR**, hybrid direct B3 **49.01 tok/s = 0.893x AR**, native hybrid direct B3 **48.17 tok/s = 0.875x AR**, and B1 branch-safe direct **26.66 tok/s = 0.4849x AR**. | Keep only as rollback-slot verifier scaffolding. Remove or move to a dedicated experiment harness unless a future full-suite/heldout row uses exact direct commit inside a verifier-amortization path that beats true AR. Do not promote from row-level exactness or smoke-level diagnostics alone. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-replay-state-commit`. | Default-off diagnostic. It scores strict target blocks with the selected block verifier without direct linear-state capture, then restores and replays the accepted prefix through `verify_target_block_serial_exact()` for resident state. The corrected 2026-07-02 13-cycle F32 selected-intermediate run proves the transaction wiring (`target_verify_replay_rows=38`, `target_verify_direct_commit_rows=0`) but rejects it as a replication path: it diverges early at cycle 2 (`[40798, 1590]`) and falls to **31.14 tok/s** because every accepted prefix pays serial replay. | Remove after the direct-state capture path is made semantically identical to the intended block scoring path, or move this to a dedicated debug harness if it remains useful as a state-lifecycle comparator. Do not promote; it is intentionally slower and negative semantically. |
| GGUF MTP capture-path diagnostics | Default-off env probes `HIPENGINE_GGUF_VERIFY_CAPTURE_F32_CHAIN_CONV`, `HIPENGINE_GGUF_VERIFY_CAPTURE_REGULAR_CHAIN_GDN`, `HIPENGINE_GGUF_VERIFY_CAPTURE_BF16_GDN_OUT`, `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN`, `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV`, and `HIPENGINE_GGUF_VERIFY_CAPTURE_SCORE_PREFILL`, plus diagnostic Conv/GDN row-state wrappers `qwen35_linear_attn_conv_prefill_f32_state_rows` and `qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_state_rows`. | Added 2026-07-02 to split the forced pair-12 direct-state mismatch. The diagnostic artifact `benchmarks/results/2026-07-02-mtp-capture-path-diagnostics.json` rejects the simple token-output fixes: BF16 GDN output, prefill-shaped Conv/GDN state rows, and score-prefill/chain-commit all still sample `[15495, 539, 1151]` with row-1 margin **+0.29526**. The later lifecycle comparator shows `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1` is nevertheless required for byte-identical full-accept captured state, so `--llama-compat` now forces that env while the other capture flags remain diagnostic-only. Added 2026-07-03 `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN_CHAIN_CONV` to test the raw-state finding that selected Conv state has much larger default-vs-prefill-GDN pairwise drift than recurrent state. | Remove or move the unused capture probes to a dedicated debug harness after the verifier FP32 hidden/KV history contract is aligned against llama.cpp. Keep the prefill-GDN capture mechanism as part of llama-compat until a narrower always-prefix-equivalent row-state capture supersedes it. Do not expose the other flags as user tuning; they are negative diagnostics unless the hybrid chain-Conv/prefill-GDN route wins the forced-pair and suite gates. |
| GGUF MTP state-lifecycle diagnostics | `scripts/gguf_mtp_forced_target_probe.py --state-lifecycle-compare` hashes post-cycle FP32 hidden seed plus per-layer Conv/GDN resident state for replay-state vs direct-state verifier policies. | Default-off diagnostic. Added 2026-07-02 for the active llama-compat trace. Base direct capture first mismatches replay at cycle 0 despite identical visible tokens `[12305, 198, 727]`. With `HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`, cycles 0-2 become byte-identical and the first mismatch moves to cycle 3, a partial/reject cycle where both policies emit `[65342]` but direct commit diverges from `serial_exact_accepted_prefix` in hidden seed and all linear states. The partial-commit policy fix keeps direct commit for full accepts but serial-replays the accepted prefix on rejected bulk blocks; `benchmarks/results/2026-07-02-mtp-state-lifecycle-prefillgdn-partialfix-compare.json` is clean through cycle 12 (`first_mismatch: null`). The retained serial state-only replay comparator is also clean and improves the exact control **50.96 -> 51.85 tok/s**. The new directcommit comparator `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json` intentionally diverges from serial replay at cycle 3 while emitting the same visible token `[65342]`; that is expected for the llama-replication lane, whose full-suite row is **60.56 tok/s** / **16.534 ms/output** with zero replay rows. | Keep while the split contract remains useful: serial-state for exact-state control, directcommit for llama-replication timing. Remove or move to a dedicated debug harness after parity closure picks the final compat transaction policy and a narrower per-state/KV-tail comparator supersedes this broad hash check. |
| GGUF MTP diagnostics | `HIPENGINE_FUSED_LINEAR_STATE_COMMIT` opt-out around the fused captured Conv/GDN row commit in `Qwen35GGUFResidentSession._commit_verify_linear_state_row`; `HIPENGINE_LINEAR_STATE_COMMIT_CHUNKED` selects the existing chunked commit copy grid. | Default-on for direct-commit diagnostic paths only. It reuses the DFlash `linear_state_pair_commit_*` kernels to replace per-layer D2D copies when all live GGUF linear layers have captured verifier rows; it falls back to the legacy per-layer copies for non-uniform state sizes and rejects partial captured-row state through the legacy all-or-error check. Focused unit coverage and the GGUF verifier state exactness gate passed on 2026-06-29. No e2e speed claim is retained for this row; it is rollback-slot scaffolding for a future verifier-amortization path. | Remove the opt-out after a full-suite verifier-amortization path that uses direct commit beats true AR, or delete the GGUF direct-commit experiment if it never contributes to such a path. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-block-verify-mode serial-exact` and `Qwen35GGUFResidentSession.verify_target_block_serial_exact()`. | Default-off diagnostic. It consumes verifier block rows with the token-serial decode scheduler, stages FP32 hidden rows, and optionally records per-row Conv/GDN state. The focused wrong-branch gate proves direct row-0 commit is bit-exact after `[prev, wrong_child]`, unlike the current row-bulk capture path, but this deliberately does not amortize target weight loads. | Keep only as the rollback-slot correctness oracle while developing the row-bulk/amortized verifier. Remove or move to a dedicated debug harness after an exact transactional verifier beats true AR on the full suite, or if a better per-row oracle supersedes it. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --target-b1-branch-safe-block-verify` and suite route `resident-b1-branch-safe-block-cap32k-device-seed`. | Default-off rejected diagnostic. It probes a B1/root-topK block verifier that batches `[prev, draft0]`, uses row 1 only for strict draft top-1 accepts, and restores/replays row 0 for root-topK branch accepts/rejects. Row-0 direct commit is now exact in the direct route, but both branch-safe variants are smoke-negative: restore/replay B1 AR **54.93 tok/s**, MTP **31.11 tok/s = 0.566x AR**; direct row-0 B1 AR **54.97 tok/s**, MTP **26.66 tok/s = 0.4849x AR**; accepted/output **0.400**. | Remove or move to a dedicated experiment harness unless a future verifier-row lifecycle plus full-suite row beats true AR. Do not re-run as a goal path while branch-safe B1 remains slower than the retained serial/cap32k routes. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --draft-p-min` on the resident draft host-logits path, `--record-draft-confidence` artifact-only top-1 probability capture, and suite route `resident-strict-context-block-pmin08`. | Default-off rejected diagnostic. The resident path supports probability-threshold diagnostics by reading full logits only when `draft_p_min > 0`, computing top-1 softmax probability, and stopping before appending a weak draft. `--record-draft-confidence` records the same top-1 probability in raw cycle artifacts without changing acceptance; the 2026-06-29 full 10-prompt B1 diagnostic (`/tmp/hipengine-draft-confidence-b1.json`) showed strict top-1 `p>=0.999` was clean (`13/13`) but only **21.7%** recall of strict hits, while `p>=0.98` had **28/29** strict hits. The strict-context/block p=0.8 route is not competitive: smoke AR **55.00 tok/s**, B3 **38.44 tok/s = 0.6991x AR**, accepted/output **0.571**. | Keep the covered probability/confidence plumbing only for diagnostics. Remove the suite route and `--record-draft-confidence`, or move them to a dedicated experiment harness, unless a future full-suite/heldout row proves confidence gating helps a structural verifier-amortization path beat true AR. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --mtp-draft-vocab-cap N` hot-token draft LM-head cap. | Default full-vocab (`0`). A 32k cap improved the corrected merge-sort B3 smoke from `42.29` to `44.51 tok/s` with unchanged `15/15` strict accepts, but acceptance/quality are prompt-sensitive and the cap is not yet suite-validated. | Either promote a cap after full-suite train/heldout/category validation shows non-regressive acceptance and better true-AR speed ratio, or keep it as an explicit experiment knob and document that retained default remains full vocab. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --adaptive-full-vocab-after-cap-miss` and the suite/workbench `resident-cap32k-recover` route. | Default-off diagnostic. It instantiates a second full-vocab resident draft runner and switches to it after a generic capped-vocab zero-accept miss, suppressing permanent AR fallback for that miss. 2026-06-29 partial suite recovered the known cap32k B1 collapse (`accepted/output 17/37 -> 19/39`) but still measured only **52.45 tok/s = 0.958x true AR**; full suite measured **51.71 tok/s = 0.9478x AR**, `mtp_beats_ar=false`; smaller caps were not goal-closing. | Remove or move to a dedicated experiment harness once the real resident MTP lifecycle/verifier-amortization path lands, unless a future full-suite + heldout row proves capped recovery beats true AR without prompt-sensitive cap tuning. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --resident-mtp-device-seed` and suite route `resident-cap32k-device-seed`. | Default-off structural diagnostic. It seeds the resident draft from the target session's device-resident fp32 hidden seed pointer, mirroring llama.cpp's resident `pending_h` lifecycle direction and avoiding the pending-seed host round trip. 2026-06-29 full suite: AR **54.59 tok/s**, best MTP B1 **52.08 tok/s = 0.9540x AR**, accepted/output unchanged vs cap32k recovery at **78/178 = 0.438**, `mtp_beats_ar=false`. | Promote into the final resident lifecycle only if a future full-suite + heldout row beats true AR, or if the real `GGUFMTPDraftContext` absorbs the device-seed path. Otherwise remove or move it to an experiment harness after the verifier-amortization path lands. |
| GGUF MTP diagnostics | Suite route `resident-cap32k-device-seed-kv` plus `scripts/gguf_mtp_bench.py` path combining `--resident-mtp-device-seed` and `--mtp-device-kv-cache`. | Default-off rejected diagnostic. The route uses new device verifier-row staging and device-base accepted-row KV commit plumbing, but without llama.cpp prompt/context catch-up it collapses draft acceptance: B3 smoke **38.94 tok/s = 0.7124x AR**, draft_acceptance **0.032**; B1 smoke **39.73 tok/s = 0.7235x AR**, draft_acceptance **0.017**. | Keep `Qwen35GGUFResidentSession.stage_current_hidden_seed_as_verify_row()` and `Qwen35GGUFResidentMTPDraftRunner.write_kv_rows_from_device_seed_base()` as lifecycle primitives. Remove or move the no-context-replay suite route unless a future prompt-catch-up resident lifecycle uses it and beats true AR on full suite. |
| GGUF MTP diagnostics | Suite route `resident-context-cap32k-device-seed` plus `scripts/gguf_mtp_bench.py` compatibility path combining `--resident-mtp-device-seed`, `--mtp-context-replay`, and `--mtp-device-kv-cache`. | Default-off rejected structural diagnostic. It wires llama.cpp shifted prompt catch-up, resident device `pending_h`, staged verifier rows, and device MTP KV, but the target verifier remains serial: B1 smoke **50.84 tok/s = 0.9257x AR**, accepted/output **0.400**; B3 smoke **46.97 tok/s = 0.856x AR**, accepted/output **0.571**. | Keep only as a compatibility scaffold while building real target-pass amortization. Remove or move to an experiment harness once a block/graph verifier path owns the llama.cpp lifecycle and beats true AR on the full suite. |
| GGUF MTP diagnostics | `scripts/gguf_mtp_bench.py --adaptive-strict-block-probe` and suite route `resident-hybrid-strict-block-cap32k`. | Default-off rejected diagnostic. It starts with strict top-1 block-promotion probing and falls back generically to root-topK B1 + cap32k recovery when the strict probe under-accepts. 2026-06-29 full suite: AR **54.58 tok/s**, best MTP B3 **50.91 tok/s = 0.9328x AR**, accepted/output **94/194 = 0.485**, `mtp_beats_ar=false`; worse than cap32k recovery B1 **51.71 tok/s = 0.9478x AR**. | Remove or move to a dedicated experiment harness unless a future full-suite + heldout row proves a strict-block hybrid beats true AR. Do not promote based on smoke/partial closeness. |
| GGUF GDN prefill | `HIPENGINE_GGUF_GDN_PREFILL_MODE=auto|fused|chain|chain_tile64|chain_tile32|chain_wave32|chain_wave32_tree|chain_lds64|chain_lds32|chain_lds32_direct` is the fail-closed rollback/bisection selector. Explicit `chain` is the GGUF-only exact split; tile64/tile32 and both wave32 routes are rejected controls. `chain_lds32_direct` is the promoted byte-exact GPF-2E route on gfx1151; materialized `chain_lds32` is its GPF-2D rollback control and `chain_lds64` its slower geometry control. GPF-2E temporarily duplicates compact-scale prepare and direct-conv LDS32 plain/segment bodies so the materialized baseline's codegen stays fixed during A/B. The old normalized-Q/K k2 chain remains registered for PARO, compatibility fakes, and diagnostic comparison. | Clean current-default/direct 512/1K/4K prefill improves `776.428/825.319/700.824 -> 823.093/889.209/744.577 tok/s`; the six-case full-model matrix and all 250 natural transitions are exact, with aggregate decode +0.075%. The final 1+3 right-sized row is `819.641/893.266/752.308/640.096/540.850/387.334 tok/s`. gfx1151 `auto` is direct-conv; gfx1100 stays fused. | Final publication is complete. Retain materialized LDS32 for one release rollback window, then factor duplicated recurrence/prepare arithmetic and remove obsolete rejected tile/wave/LDS64 selectors. Keep gfx1100 fused until transfer evidence. |
| GGUF Q4T16 selected-prefill GPF-3A | `HIPENGINE_GGUF_Q4_T16_SELECTED_PREFILL_MODE=auto|baseline|shared_x`, explicit baseline/shared-X registry variants, and replay switch `scripts/qwen35_gguf_moe_replay.py --q4-t16-shared-x` retain duplicate Q4T16 compact32 bodies. gfx1151 `auto` is now `shared_x`; gfx1100 stays baseline. | BF16/FP16 fixture bytes are exact; tiny trace is `44.725 -> 33.343 us` (-25.45%), real Q4 gate/up replay is `114.633 -> 97.082 ms` (-15.31%), and clean full-model 512/1K/4K prefill improves +3.11%/+2.42%/+1.94%. Full-model logits and trajectories are exact; aggregate decode is -0.0031%; the final right-sized publication sweep includes the route. | Retain explicit baseline rollback for one release window, then collapse the gfx1151 losing body/alias and remove the env/replay switches; keep gfx1100 baseline until its independent transfer gate. |
| GGUF small-B linear dispatch | `HIPENGINE_GGUF_Q4K_ROWTILE` / `q4k_rowtile_session(False)` opt-out for the weight-amortized raw row-tile GEMV (`rowtile_*` variants for Q4_K/Q5_K/Q6_K/Q8_0, rows 2..8, WMMA off). | Default-on. Bit-exact vs the per-row prefill alias and ~3x faster on the dense projection shape at B=4 (microbench); fires ~250x in the B3 verifier. End-to-end verifier is flat within noise because dense projections are only ~11-17% of the verifier (the MoE selected-expert GEMV is the ~54% bottleneck). The opt-out exists for bisection only. | Make the rowtile path unconditional (drop the env/session opt-out) after the current GGUF-MTP verifier path in `docs/MTP-LLAMACPP-PARITY.md` has a same-protocol full-suite non-regression row; keep the per-row kernel only as the rows==1 / rows>8 path. |
| GGUF selected-MoE dp4a diagnostic | `HIPENGINE_GGUF_Q4K_SELECTED_DUAL_DP4A` opt-in around q8_1 activation quantization plus sudot4 for raw Q4_K selected-dual gate/up and the T16 rows>1 split gate/up path; `HIPENGINE_GGUF_T16_SELECTED_DP4A` opt-in for the broader T16 selected diagnostic, currently Q4 split plus Q5 selected-down; `HIPENGINE_GGUF_RAW_SELECTED_DP4A` opt-in for the raw no-decode-repack Q4/Q5/Q6 selected-MoE diagnostic bundle. | Default-off. The raw Q4 fallback launches correctly with caller-owned q8_1 workspace but production B3 decode-repack does not hit it; isolated raw Q4 POC measured `0.946 ms -> 0.357 ms`. Raw Q5/Q6 selected-down is also positive in isolation (`0.0916 -> 0.0395 ms` Q5, `0.0419 -> 0.0259 ms` Q6 including q8_1 quantize) and improves no-decode-repack B3 `31.63 -> 39.61 tok/s`, but still trails default decode-repack B3 `51.31 tok/s`. The active T16 split path cuts that row-bulk kernel (`~172 us -> ~142 us` in the two-cycle trace), but B3 remains flat (`49.31 tok/s`, warm `50.60`). Q5T16 selected-down also launches and is `1.10x` faster in isolation (`0.0335 ms -> 0.0306 ms` including quant), but the c1-shaped synthetic top-1 is `0.875` and B3 regresses to `47.62 tok/s` (warm `48.44`). The callable T16 fused-SiLU dp4a variant and Q6T16 selected-down dp4a are intentionally not routed. X8 selected-down is tracked in the dedicated row above because q6-only now has llama-compat evidence while q5/both remains rejected. **2026-06-28 GPU-bound re-test (post lib-cache): `HIPENGINE_GGUF_T16_SELECTED_DP4A` clean interleaved A/B (3 runs x 12 cycles, warm) on the full resident-draft B3 bench is flat-negative `48.60 -> 48.42 tok/s` (-0.4%), acceptance identical — dp4a wins at the kernel level (-35% MoE GEMV) and +5% on the verify-isolated harness but does NOT move the full-bench wall, i.e. the full B3 verifier is host-dispatch-bound (~875 launches), not GPU-kernel-bound. dp4a stays default-off; the lever is launch-count reduction (#9 + fusion). Artifact `benchmarks/results/2026-06-28-verifier-dp4a-fullbench-b3-ab.json`.** | Remove these raw/T16 flags unless a later GGML-style q8_1/x4 layout clears the quality gate and improves the same B3/full-suite protocol. Promote only the production-compatible route; keep raw no-decode-repack diagnostics separate from production T16 routing. |
| GGUF Q5 T16 selected-down one-wave diagnostic | `HIPENGINE_GGUF_T16_SELECTED_Q5_DP4A_THREADS` q5-only override for the T16 selected-down q8_1/dp4a direct kernel. Unset inherits `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` and therefore the retained 64-thread selected-MoE scheduler; valid diagnostic values are `32`, `64`, and `128`. | Default-off. The llama.cpp-shaped one-wave Q5 check improved the isolated selected-down microbench at rows=16 on gfx1151: prequantized dot **0.03608 -> 0.03305 ms**, quantize+dot **0.04031 -> 0.03685 ms**, KL mean **0.00398**, KL max **0.03093**, top-1 **0.9375**. The Q4 control stayed on the 64-thread path (`t16_dp4a_dot_prequantized` **0.04007 ms**), so the override only affected Q5. The real compat smoke rejected it: `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6` with q5t32 measured **68.14 tok/s / 14.776 ms/output**, worse than same-route pack8/q6 smoke around **69.06 tok/s / 14.501 ms/output**, with identical smoke acceptance. Artifacts `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-t64-rerun-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q5-t16-selected-down-dp4a-q5t32-micro.json`, `benchmarks/results/2026-07-01-llama-compat-b2-q4-t16-selected-dual-dp4a-q5t32-control-micro.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q5t32-smoke.json`. | Remove this q5-specific override after the selected-MoE scheduler/layout is replaced, or sooner if the next verifier body/layout optimization leaves q5t32 rejected on the same async/full-suite protocol. Do not promote it into the active `llama-compat` route. |
| Env flag surface | Benchmark and diagnostic flags still cover rejected or superseded experiments. | The 2026-07-10 release audit removed env requirements from public backend/quant/GGUF fast-path selection. Accuracy-traded MTP, unsafe kernels, profiler synchronization, and rejected paths remain explicit diagnostics. | Move rejected runtime flags into benchmark-only configuration as each associated MTP/PARO investigation closes; retain correctness fallbacks for one release window after a default promotion. |
| GGUF MTP resident draft | `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE` opt-out around the device-resident selected-MoE down + combine in `mtp_resident_draft.py` (`apply_moe_down_combine`: `silu_mul_separate_out_bf16` + `gguf_q5_k_selected_gemv_bf16_bf16_out` + `weighted_sum_shared_gate_combine_residual_out_bf16_f32w`). The `=0` legacy path keeps the host-readback per-expert Python down loop for bisection. | Default-on after 2026-06-28 B3/c5 A/B: exact-acceptance (drafts byte-identical, accepted_per_output identical in all 4 categories code/general_en/general_ja/mixed_ja_en) and tok/s consistently up per-category (~+0.7-1.4%); removes 2 blocking D->H and ~24 launches/depth from the MoE-down section. Matches verifier bf16 precision. Artifact `benchmarks/results/2026-06-28-resident-mtp-draft-device-moe-down-ab.json`. | Drop the opt-out and delete the legacy host-loop branch (and the then-unused `gate_f32`/`up_f32`/`inter_f32`/`down_out`/`scaled`/`gated_shared` buffers) after sub-win B (device argmax + embedding gather) lands and a same-protocol full-suite MTP row is retained. **Sub-win B landed 2026-06-29 (device-chain row below); the legacy-host-loop deletion + buffer cleanup is now unblocked once a retained full-suite MTP row exists.** |
| GGUF MTP resident draft device-chain | `HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_CHAIN` opt-in (default off) for the device-chained draft in `_propose_chain_device`; explicit bench flag `--resident-mtp-device-chain` prewarms the path for llama.cpp replication routes. Each depth's top-1 is device-gathered from a resident FP32 embedding table (`gather_f32_rows_by_i32id`, exact copy of `token_embd_f32[:vocab]`, 268MB cached upload), top-k is accumulated on device, one drain + readback happens at chain end, and rope/pos/ctx are precomputed once. | Default-OFF. BIT-EXACT vs the legacy host loop (0/5 top-1 + topk-row divergence unit gate; e2e B3 drafts + total_accepted 37/1008 identical). Original B3 evidence was flat (`39.81 -> 39.79 tok/s`) because the draft is GPU-compute-bound, not host/sync-bound. Llama-replication evidence on 2026-06-30: prewarm removes the short-run 268MB upload artifact (`draft_device_chain_ensure_embed_table` **11.888 -> 0.000 ms/output**) but full-suite compat dp4a only moves **52.48 -> 52.79 tok/s**. Split timing shows `draft_topk_readback` is almost all GPU drain (`draft_device_chain_drain` **3.830 ms/output**) and not D2H (`draft_topk_d2h` **0.008 ms/output**). 2026-07-01 sync-stage attribution moves that drain into section buckets: `draft_run_lm_head` **1.882 ms/output**, `draft_run_attention` **0.718**, `draft_run_ffn_up_shared` **0.557**, `draft_device_topk_gather` **0.357**. Artifacts `benchmarks/results/2026-06-29-resident-mtp-draft-device-chain.json`, `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full.json`, `benchmarks/results/2026-06-30-ar-mtp-llama-compat-device-chain-dp4a-b2-full-split.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-draftsync-full.json`. | Keep as an explicit llama.cpp replication diagnostic while the compat draft/verifier lifecycle is being matched. Do not promote just to remove host copies; promote/collapse only if a future resident draft fusion or verifier-side change cuts the **GPU drain** and improves the same full-suite protocol. Otherwise delete the flag/route during the post-MTP flag cleanup; the `_run_one` device-pointer refactor + gather kernel stay regardless. |
| GGUF MTP resident draft Q6 top-1/gather | `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_GATHER` opt-out around the exact Q6_K lm-head top-1 specialization in `mtp_resident_draft.py` for resident device-chain `top_k == 1`. The new `pack8_gemv_decode_bf16_top1_gather_f32` kernel writes the selected id/value and optionally gathers the next FP32 embedding row, replacing full-logits materialization + separate top-k + gather in the llama-compat draft path. | Default-on after 2026-07-01 same-tree full-suite A/B on `llama-compat-device-chain-dp4a`: **52.60 -> 53.34 tok/s** (+1.4%), `cycle_wall_ms_per_output` **19.033 -> 18.772**, `draft_initial` **4.033 -> 3.712**, acceptance unchanged (`acc/output 0.561`, draft acceptance `0.640`). Unit gate proves identical selected id/value/embedding row vs the old logits -> top-k -> gather chain. Sync-stage attribution confirms `draft_device_topk_gather` **0.357 -> 0.001 ms/output** while verifier remains ~14.66 ms/output. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-b2-q6top1-full.json`, disabled control `...q6top1-control-full.json`, and sync-stage `...q6top1-draftsync-full.json`. | Keep the opt-out for short-term A/B while the llama-compat replication lane is still active. Make the top-1/gather path unconditional for resident device-chain `top_k == 1` after the next compat verifier-layer optimization validates against the same full-suite route, then remove the env flag and old top-k/gather branch for that case. |
| GGUF MTP resident draft Q6 top-1 q8_1/dp4a | `HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A` / `--resident-mtp-draft-q6-top1-dp4a` opt-in around the llama-compat resident draft Q6_K lm-head top-1/gather path. It q8_1-quantizes the BF16 head input and calls `pack8_gemv_decode_q8_1_dp4a_top1_gather_f32`, matching llama.cpp's quantized matvec economy more closely than the exact raw-Q6 path. The diagnostic `HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS` / `--resident-mtp-draft-q6-top1-stage1-threads {64,128}` exists only to A/B the stage1 scheduler; suite routes `...x8q6-t64` and `...x8q6-t64-allsync` force 64 threads. | Default-off, accuracy-traded. 2026-07-01 full-suite `llama-compat-device-chain-dp4a-q6top1dp4a` B2: **58.83 -> 59.63 tok/s** (+1.36%), `cycle_wall_ms_per_output` **17.019 -> 16.793**, `draft_initial` **3.564 -> 3.293 ms/output**, acceptance unchanged (`acc/output 0.578`, draft acceptance 0.685). All-sync smoke confirmed `draft_run_lm_head` **1.471 -> 1.253 ms/output**; the q6-X8 stage split now attributes the retained 128-thread aggregate to Q6 top-1 stage1 **1.218 ms/output** plus stage2/gather **0.041 ms/output**, while norm+cast+q8_1 quantize is **0.030 ms/output**. The 64-thread scheduler check is rejected on the real route: all-sync stage1 **1.218 vs 1.246 ms/output** and same-session async **69.06 tok/s / 14.501 ms** vs t64 **68.79 tok/s / 14.557 ms**, identical acceptance. Unit gate matches a CPU q8_1/Q6_K oracle; rocprofv3 confirms both 128-thread and 64-thread stage1 kernels ran. Artifacts `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-full.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-top1split128-allsync-smoke.json`, `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-top1split-allsync-smoke.json`, and `benchmarks/results/2026-07-01-ar-mtp-llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-t64-smoke.json`. | Keep the q8_1/dp4a draft head as an explicit llama-compat diagnostic until full-suite heldouts decide whether the accuracy-traded draft head should be folded into the named compat route. Do not promote to the exact/default resident draft path. Remove the t64 thread flag/routes after the next draft lm-head optimization because the scheduler-width copy is rejected; standalone activation-quantization/cast and final reduce/gather tweaks are already ruled out, so future work should target Q6_K stage1 compute/layout or a broader llama-style MMVQ layout. |

## Post-Optimal-Path Cleanup Targets

These are not optimization tasks for the current sprint. They are the cleanup
pass to run once a path is fast and correct enough that the benchmark defaults
should be boring.

| Path | Cleanup target | Keep | Remove / collapse trigger |
| --- | --- | --- | --- |
| 35B MTP chain verifier | Collapse the sprint-era stack of env flags into the default dispatch path and document the current optimal B=1 chain route, while keeping B=2/B=3 available for adaptive-density policy experiments. | Numerical fallbacks, exactness tests, and rollback toggles that are still needed for one release window. | Retained `>1.0x` same-suite row plus one follow-up defaults-only rerun after the adaptive-policy decision. |
| 35B MTP tree/top-k | Keep tree code default-off until it beats chain on the same wall and prompt suite; do not let tree-specific dispatch obscure the chain hot path. | Tree correctness tests and graph replay scaffolding. | If tree remains negative after the verifier wall cut, demote branch/top-k runtime flags to explicit experiment scripts. |
| 27B dense DFlash | Separate deployable online routing from profile-history diagnostics. The current positive production row is the online whole-cycle confidence gate; older prompt-history route/terminal-tail rows are retained evidence, not the default API shape. | Online gate config, oracle/calibration tooling, exact AR comparisons. | After the DFlash hardening rerun and decode API update, trim profile-history routing from the main hot path or move it behind an explicit research harness. |
| DFlash drafter/verifier flags | Audit `HIPENGINE_DFLASH_DRAFTER_DENSE`, `HIPENGINE_DFLASH_DRAFTER_ADD_RMSNORM`, and `HIPENGINE_DFLASH_VERIFY_FUSED_LM_HEAD`. | Default-on exact dense WMMA if the fresh 27B gate confirms it; tests for rejected fused kernels. | Fresh 27B DFlash rerun decides: promote exact positive flags to defaults, remove negative runtime branches, or demote them to test-only overrides. |
| Benchmark commands | Stop requiring long flag piles once defaults represent the optimal path. | Flags that select workload shape, model, quant, and explicit experiments. | After MTP/DFlash defaults-only rows are retained, update benchmark docs to show default commands first and move historical A/B flags into dated notes. |

## `--verify-dp4a` / `*-pmin05-dp4a` route (default OFF, opt-in accuracy-traded)
- Added 2026-06-30. Bench flag `--verify-dp4a` (gguf_mtp_bench.py) + suite route
  `resident-b1-probe-block-direct-cap32k-minrows2-pmin05-dp4a` enable llama.cpp-style
  dp4a (q8_1) selected-expert verify GEMVs. **Default off; accuracy-degrading.**
- Purpose: let users who accept llama's precision loss get max accuracy-traded MTP
  perf (~61.6 tok/s / 1.13x B5). FAILS the ja correctness gate (greedy top-1 0.700 <
  0.90). Does NOT match llama HIP MTP (67.3) — dp4a is necessary but not sufficient.
- Remove when: either a Vulkan backend supersedes the perf goal, or the project
  decides to drop dp4a experimentation entirely. Until then it is the documented
  opt-in for the dp4a/accuracy tradeoff. See docs/MTP-LLAMACPP-PARITY.md "COFFIN NAIL".

## `--llama-compat` / `llama-compat*` routes (default OFF, semantic diagnostic)
- Added 2026-06-30. Bench flag `--llama-compat` forces the closest hipEngine
  replica of llama.cpp MTP semantics: B2, `draft_p_min=0`, full draft vocab,
  shifted MTP context replay, device MTP KV, no adaptive B1 probe/fallback, and
  one target block verifier per cycle. Suite routes `llama-compat` and
  `llama-compat-dp4a` are fixed to B2 so the artifact label matches the forced
  child `draft_n_max`. Follow-up replication routes
  `llama-compat-device-chain{,-dp4a}` and
  `llama-compat-device-seed-chain{,-dp4a}` add prewarmed resident device-chain
  drafting and optional resident target `pending_h` starts without changing the
  shipped default path.
- Purpose: isolate whether the remaining llama HIP MTP gap is semantic-policy
  mismatch versus implementation/backend cost. The exact compat route is
  precision-preserving; `llama-compat-dp4a` adds the already-known
  accuracy-traded q8_1/dp4a regime. Full-suite B2 evidence landed the same day:
  exact compat **51.16 tok/s = 0.934x AR**, dp4a compat **52.48 tok/s = 0.958x
  AR**, prewarmed device-chain dp4a **52.79 tok/s = 0.965x AR**, and
  device-seed-chain dp4a **52.53 tok/s = 0.960x AR**. All remove the serial B1
  probe and keep acc/output ~0.56, but lose to AR because the compat
  draft/context + block verifier lifecycle is too costly. Split instrumentation
  shows device-chain `draft_topk_readback` is almost all GPU drain
  (`draft_device_chain_drain` **3.830 ms/output**) and not D2H
  (`draft_topk_d2h` **0.008 ms/output**). Follow-up Q6 top-1/gather plus
  direct-state verifier cleanup lifts the best compat dp4a B2 diagnostic row to
  **55.41 tok/s = 1.014x AR**, with unchanged acceptance. The new
  `llama-compat-device-chain-dp4a-allsync` route adds
  `--resident-mtp-draft-sync-stage-timings` and
  `--target-block-sync-stage-timings` for attribution only; its buckets show the
  remaining verifier cost is target linear-attention/MoE operation time, not
  snapshot/commit bookkeeping. The current active replication route is
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit`:
  **60.56 tok/s**, **16.534 ms/output**, **1.1055x AR**, acc/output **0.609**,
  draft acceptance **0.780**, target rows/output **1.172**, verifier drain
  **14.071 ms/output**, replay/commit **0.043 ms/output**, and zero replay rows.
  The semantic-safe serial-state row remains the exact control at **51.85 tok/s**
  / **19.308 ms/output**.
- Remove / promote when: after full-suite stage-bucket evidence decides the
  question. Current evidence says these are replication diagnostics, not default
  promotion candidates; keep only the smallest route set needed for future parity
  audits, or delete the routes during the next MTP flag cleanup unless another
  llama.cpp semantic delta is identified.

## `--resident-mtp-draft-sync-stage-timings` (default OFF, attribution-only)
- Added 2026-07-01. Bench flag inserts `hipDeviceSynchronize()` boundaries inside
  the resident MTP draft `_run_one()` path when `--record-cycle-stage-timings` is
  enabled. Suite route `llama-compat-device-chain-dp4a-draftsync` wires it into the
  llama.cpp replication lane.
- Purpose: split the previous `draft_device_chain_drain` bucket into
  `draft_run_project`, `draft_run_qkv_kvwrite`, `draft_run_attention`,
  `draft_run_ffn_up_shared`, `draft_run_moe_down_combine`, `draft_run_lm_head`, and
  `draft_device_topk_gather`. The flag changes timing by adding synchronization and
  is not a performance path.
- Remove when: the resident draft LM-head/top-k or verifier layer-time follow-up has
  its own lower-overhead profiler/rocprof attribution, or after the llama.cpp
  replication lane is closed. Until then keep it only as a named diagnostic route.

## `--target-block-sync-stage-timings` (default OFF, attribution-only)
- Added 2026-07-01. Bench flag inserts `hipDeviceSynchronize()` boundaries inside
  the target block verifier when `--record-cycle-stage-timings` is enabled. Suite
  route `llama-compat-device-chain-dp4a-allsync` combines it with resident draft
  sync timings for one-pass draft+verifier attribution.
- Purpose: split `target_block_linear_attn_layers` and
  `target_block_full_attn_layers` into operation buckets (`norm_qkv_gate`,
  `chain_gdn`, selected-MoE expert gate/up/down, shared expert, combine, and
  full-attn KV/attention/output sections). The flag changes timing by adding
  synchronization and is not a performance path.
- Remove when: a lower-overhead verifier profiler/rocprof harness can produce the
  same operation split, or after the llama.cpp replication lane is closed.

## `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL` / `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM` / `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS` / `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA` / `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT` / `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE` / `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN` / `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE` / `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN` / `HIPENGINE_GGUF_VERIFY_F32_POST_NORM` (default OFF, semantic diagnostic)
- Added 2026-07-02. Env flag keeps target-block verifier residual outputs in
  FP32 for an opt-in llama.cpp parity probe while preserving BF16 mirrors for
  existing projection kernels. It adds FP32 add/RMSNorm and MoE combine helpers.
  The follow-up diagnostic also feeds layer-entry attention RMSNorm from FP32
  residual rows when available, but intentionally does not claim full llama.cpp
	  graph parity. `HIPENGINE_GGUF_VERIFY_F32_ATTENTION_NORM=1` materializes
	  layer-entry attention RMSNorm into FP32 scratch, casts a BF16 mirror for
	  unsupported consumers, and routes dense-Q8 dp4a QKV / QKV+gate consumers from
	  the FP32 tensor when the F32 dense-Q8 diagnostic is already active.
	  `HIPENGINE_GGUF_VERIFY_F32_LINEAR_PROJECTIONS=1` additionally routes
	  compatible row-bulk linear-attention Q8 `attn_qkv`/`attn_gate` projections
	  into FP32 scratch through the raw-Q8 dp4a F32-output dual wrapper, casts BF16
	  mirrors for existing downstream kernels, and emits explicit BF16 mirror
	  capture keys. It also routes dense-F32 `ssm_alpha`/`ssm_beta` through the
	  registry-dispatched F32-input/F32-output dense GEMV route when available,
	  while preserving BF16 mirrors for existing downstream consumers.
	  `HIPENGINE_GGUF_VERIFY_F32_ALPHA_BETA=1` additionally routes row-bulk
	  linear-attention `ssm_alpha`/`ssm_beta` from that FP32 attention-norm tensor to
  mirror llama.cpp's `build_layer_attn_linear` source shape.
  `HIPENGINE_GGUF_VERIFY_F32_ATTN_OUT=1` keeps the row-bulk linear-attention
  `ssm_out` projection output in FP32 through the post-attention residual/RMSNorm
  add while preserving the BF16 mirror for existing captures and downstream
  kernels. It also keeps row-bulk full-attention `attn_output` in FP32 through
  the same residual/RMSNorm helper when a raw Q8 sidecar BF16-input/F32-output
  path is available.
  `HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE=1` keeps the selected-expert weighted
  sum in FP32 inside the F32-residual MoE combine instead of BF16-rounding that
  selected sum before adding residual and sigmoid-gated shared output.
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN=1` requires the F32 MoE combine
  diagnostic and routes compatible X8 Q5/Q6 selected-down GEMV outputs into an
  FP32 scratch buffer before combining selected rows with BF16 shared output.
  `HIPENGINE_GGUF_VERIFY_F32_SELECTED_INTERMEDIATE=1` requires the F32 MoE
  combine + selected-down stack, computes selected `silu(gate) * up` into FP32
  scratch, preserves the BF16 mirror, and feeds the FP32 activation into
  selected-down.
  `HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN=1` requires the F32 MoE combine and
  selected-down stack, routes shared-expert down output into FP32 scratch,
  preserves the BF16 mirror, and combines FP32 selected rows with FP32 shared
  rows.
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` extends the probe by materializing
  post-attention RMSNorm into FP32 scratch and independently gating router /
  selected-q8 / shared-q8 consumers with
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM_{ROUTER,SELECTED_Q8,SHARED_Q8}`.
- Purpose: test the current semantic hypothesis that accumulated BF16 verifier
  layer-boundary drift is enough to flip near-tie target decisions versus
  llama.cpp's F32 target `l_out` graph tensors. The diagnostic artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-diagnostic.json`
  confirms the lever is active: the old cycle-12 trace cannot replay unchanged
  because cycle 2 flips from exact `[40798, 25, 1103]` / accepted 2 to
  FP32-residual `[40798, 1590, 1103]` / accepted 1. The follow-up artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-attnnorm-diagnostic.json`
  reaches the old cycle-12 branch but still accepts `539`, with the wrong
  `539 - 26126` margin increasing from **+0.11822** to **+0.14309** versus
  llama.cpp **-0.00896**. The attention-norm-output dense-Q8 split
  (`benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-output-denseq8-diagnostic.json`,
  control
  `benchmarks/results/2026-07-02-mtp-target-f32-residual-bulk-control-diagnostic.json`)
  moves the bulk pair-12 `539 - 26126` margin from **+0.31369** to
  **+0.18198**, still opposite llama.cpp. The linear-attention output-to-residual
  split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-denseq8-diagnostic.json`
  moves that margin only **+0.18198 -> +0.17663**, so the `ssm_out` BF16 round is
  not the main missing semantic lever. The alpha/beta F32 input split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-denseq8-diagnostic.json`
  is byte-identical to the attention-output slice and leaves the row-1 margin at
  **+0.17663**, ruling out `ssm_alpha`/`ssm_beta` projection input precision for
  the active branch. The full-attention output-to-residual split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-fullattnout-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]`, accepts 2, and worsens row-1
  `539 - 26126` to **+0.27480**, so full-attention `attn_output` BF16 output
  rounding is ruled out as well. The MoE selected-sum accumulator split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]` and accepts 2, but narrows row-1
  `539 - 26126` from **+0.27480** to **+0.03385**. That makes the combine
  selected-sum BF16 boundary semantically active, but still not sufficient to
  match llama.cpp's **-0.00896** margin. The selected-down output split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-denseq8-diagnostic.json`
  keeps compatible X8 selected-down rows in FP32 and narrows the same margin to
  **+0.00536** (`26.06115 - 26.05580`), still on the wrong side of llama.cpp by
  about **0.0143 logits**. The selected-intermediate split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-selectedintermediate-denseq8-diagnostic.json`
  is the first pair-12 side-matching slice: sampled tokens become
  `[15495, 26126, 1151]`, accepted drafts fall to 1, and row-1 `539 - 26126`
  moves to **-0.00303** (`26.04795 - 26.05098`), close to llama.cpp's
  **-0.00896**. This confirms the selected SwigLU/intermediate BF16 boundary
  is a parity contract to fold into a cohesive llama-compat verifier mode if
  longer-trace/full-suite acceptance validates it. The shared-down output split
  `benchmarks/results/2026-07-02-mtp-target-f32-attnnorm-attnout-alphabeta-moecombine-selecteddown-shareddown-denseq8-diagnostic.json`
  still samples `[15495, 539, 1151]` and accepts 2, and widens the same margin
  to **+0.03043** (`26.12703 - 26.09660`), ruling out isolated shared-down
  output precision as the missing parity fix. Combining MoE combine with
  `HIPENGINE_GGUF_VERIFY_F32_POST_NORM=1` failed prior-cycle replay at cycle 2
  (`[40798, 1590, 1103]` vs trace `[40798, 25, 1103]`), so the combination is
  not a pair-12 parity result. The post-norm split artifact
  `benchmarks/results/2026-07-02-mtp-target-f32-postnorm-split-diagnostic.json`
  shows the combined router+selected-q8 consumer path breaks the old trace at
  cycle 7; selected-q8 alone flips row 1 (`413 - 4071` **+0.13053 -> -0.14458**),
  while router-only reaches pair 12 but worsens `539 - 26126` to **+0.33520**.
  This makes these flags instrumentation only, not a candidate promotion.
- Remove when: either a fuller F32 verifier graph path supersedes this partial
  residual-boundary slice, or parity work decides llama.cpp's F32 graph
  semantics are not the target. Do not promote this flag as a speed path.

## `HIPENGINE_GGUF_T16_SELECTED_DP4A_THREADS` (diagnostic rollback)
- Added 2026-07-01. Host-side launch switch for the selected T16 q8_1/dp4a
  verifier kernels used by `--verify-dp4a` / `llama-compat-device-chain-dp4a`.
  Default is now `64`; setting the env var to `128` restores the old launch
  shape.
- Purpose: keep a rollback/A-B hook for the first selected-MoE scheduler change
  that survived async/full-suite validation. Full-suite `llama-compat` B2 moved
  **55.45 -> 58.83 tok/s** and `target_block_verify_total`
  **14.025 -> 13.134 ms/output** on gfx1151.
- Remove when: either the selected-MoE scheduler is replaced by a llama-style
  `mul_mat_vec_q_moe` port or two later full-suite compat runs confirm 64 is
  stable enough that the 128-thread rollback path is no longer useful.

## `HIPENGINE_GGUF_Q8_T16_THREADS` (diagnostic rejected)
- Added 2026-07-01. Host-side launch switch for Q8_0 T16 single/pair/triple
  GEMV wrappers. Default/unset keeps the existing 128-thread launch; setting
  the env var to `64` exercises a smaller workgroup for verifier projections.
- Purpose: test whether the llama-compat verifier hot leaf
  `attn_qkv+attn_gate` is losing time because Q8T16 pair projection uses the
  wrong launch width. The focused qwen35 pair microbench rejected 64 threads:
  rows 2/3/4 measured **197.77/224.80/251.96 us** at 64 threads versus
  **179.26/207.05/237.02 us** at 128 threads. `rocprofv3` confirmed the 64-thread
  override launched with `Workgroup_Size_X=64`.
- Remove when: the Q8T16 verifier pair work moves to a different llama-style
  kernel body/schedule, or when the parity sprint no longer needs this A/B hook.
  It is not a performance path and should not be promoted.

## `gguf_q8_0_t16_dual_gemv_decode_q8_1_dp4a_bf16_bf16_out` (diagnostic rejected)
- Added 2026-07-01. Callable T16 Q8_0 dual-split pair kernel that consumes
  GGML q8_1 activation blocks and uses `sudot4`, intended to test whether the
  llama.cpp Q8_0×Q8_1 arithmetic recipe transfers to the existing Q8T16
  `attn_qkv+attn_gate` verifier pair layout.
- Purpose: isolate the kernel-body question after the 64-thread launch-width
  check failed. Correctness passed against a q8_1 CPU oracle plus KL/top-1 gate,
  and `rocprofv3` confirmed `q8_0_t16_dual_split_q8_1_dp4a_kernel<unsigned short>`
  launched with `Workgroup_Size_X=128`. Performance rejected the route: for the
  qwen35 pair shape, rows 2/3/4 exact 128-thread pair is
  **181.50/207.98/236.26 us**, while quantize+dp4a is
  **304.78/448.32/558.14 us** and prequantized dp4a is
  **303.05/452.51/566.29 us**.
- Remove when: the parity sprint moves Q8 verifier work to a true llama-style
  mmvq/T16 replacement layout or row-amortized verifier kernel. This callable is
  evidence, not a performance path; do not route it into `llama-compat` runtime.

## `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE` (diagnostic rejected)
- Added 2026-07-01. Default-off runtime hook for the exact Q8T16
  `attn_qkv+attn_gate` pair rowtile diagnostic. Setting
  `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=1` routes the qwen35
  `rows>1, in=2048, out=(8192,4096)` pair through
  `gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out` at 64 threads.
- Purpose: test whether llama.cpp-style row amortization closes the verifier
  pair gap while preserving exact arithmetic. Correctness is bit-identical to
  the existing exact pair, including the large qwen35 pair fixture. The isolated
  microbench was positive (`rows=2/3/4/5/6` exact 128:
  **179.75/207.70/236.41/265.87/298.97 us** vs rowtile4-64:
  **154.05/170.55/191.16/254.19/271.06 us**) and same-code smoke improved
  **66.00 -> 67.14 tok/s**, but the full-suite llama-compat row rejected it:
  **59.63 -> 57.25 tok/s**, `target_block_verify_total`
  **13.178 -> 13.697 ms/output**.
- Remove when: the parity sprint moves Q8 verifier work to a true llama-style
  layout/scheduler port, or after another full-suite row confirms this exact
  rowtile route remains non-retainable. It is an evidence hook only; default and
  llama-compat runtime paths stay on the existing exact pair wrapper.

## `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL` (diagnostic rejected)
- Added 2026-07-01. Default-off runtime hook for broad exact Q8T16 verifier
  row-amortization. Setting `HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL=1` routes qwen35
  `rows>1, in=2048` singleton, pair, and triple Q8T16 projections through
  rowtile4 wrappers where available. It also enables the pair rowtile diagnostic
  unless `HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE=0` is set explicitly. Suite route:
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-q8rowtileall`.
- Purpose: test whether the isolated exact pair-rowtile win can be extended over
  the full retained llama-compat verifier shape. Correctness passes against the
  existing exact singleton/pair/triple wrappers. The B2 block profile moved the
  dense-Q8 bucket **11.420 -> 10.811 ms/block** and total kernel time
  **26.053 -> 25.276 ms/block**, mostly by cutting the Q8 pair body
  **6.025 -> 5.316 ms/block**. The async smoke rejected promotion:
  same-session retained `x8q6` reached **68.78 tok/s / 14.561 ms/output** while
  q8rowtileall reached **68.54 tok/s / 14.614 ms/output** with identical
  acceptance.
- Remove when: the parity sprint replaces the current T16 Q8 verifier layout
  with a true llama.cpp-style Q8_0 x Q8_1 MMVQ layout/scheduler, or after the
  dense-Q8 verifier target is resolved another way. This is evidence only; it
  should not become default or update the retained llama-compat lane without a
  future full-suite win.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=row` / row Q6 top-1 routes (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape row` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-row` /
  `...-row-allsync` exercise a llama.cpp-shaped Q6_K draft lm-head top-1
  stage: one output row per block, two wave32 warps, and a signed
  `__vsubss4`/dot4 Q6_K MMVQ body. Default stays `pack8`.
- Purpose: test whether the remaining draft-side Q6_K top-1 gap is caused by
  hipEngine's pack8 output-row geometry rather than the vector-dot body itself.
  Correctness passes against the q8_1/Q6_K oracle, but performance rejects the
  route: all-sync row stage1 is only slightly faster than pack8
  (**1.202 vs 1.218 ms/output**) while row stage2/gather grows
  **0.041 -> 0.252 ms/output** because it reduces over `vocab` instead of
  `vocab/8`; async smoke regresses **69.06 tok/s / 14.501 ms** to
  **66.95 tok/s / 14.958 ms** with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused row-stage/top-1 reduce makes this row-shape diagnostic
  obsolete. It is evidence, not a performance route.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack8_scalehoist` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack8_scalehoist` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-scalehoist` /
  `...-scalehoist-allsync` exercise a Q6_K draft lm-head top-1 stage that keeps
  the retained pack8 `vocab/8` final reduce but hoists each Q6_K block's
  `d*scale[16]` values into shared memory.
- Purpose: test whether the remaining q8_1/dp4a Q6_K draft stage1 cost is from
  repeated Q6 scale loads rather than the dot body or output geometry. Correctness
  passes against the q8_1/Q6_K oracle, and `rocprofv3` confirms
  `gguf_q6_k_pack8_gemv_q8_1_dp4a_top1_scalehoist_stage1_kernel` launches.
  Same-session smoke rejected it: retained `x8q6` rerun **68.65 tok/s**,
  cycle **14.589 ms/output**, `draft_initial` **2.482 ms/output** vs
  scalehoist **68.54 tok/s**, cycle **14.610 ms/output**, `draft_initial`
  **2.485 ms/output**, with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack8_llama` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack8_llama` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack8llama` /
  `...-pack8llama-allsync` exercise a Q6_K draft lm-head top-1 stage that keeps
  the retained pack8 `vocab/8` final reduce but uses the llama.cpp Q6_K MMVQ
  vecdot decomposition inside stage1.
- Purpose: test whether the remaining q8_1/dp4a Q6_K draft stage1 cost is the
  pack8 dot body rather than final-reduce geometry. Correctness passes against
  the q8_1/Q6_K oracle for fused and split stage1+stage2 paths. Same-session
  all-sync moved the intended leaf **1.220 -> 1.205 ms/output**, but async B2
  smoke rejected the route: retained control **68.88 tok/s**, cycle
  **14.541 ms/output**, `draft_initial` **2.487 ms/output** vs pack8_llama
  **67.92 tok/s**, cycle **14.747 ms/output**, `draft_initial`
  **2.493 ms/output**, with identical acceptance.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=pack16` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape pack16` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-pack16`,
  `...-denseq8all-pack16`, and `...-denseq8all-pack16-allsync` exercise a
  Q6_K draft lm-head top-1 stage that keeps the retained pack reduction but
  doubles the output group from 8 to 16 vocab rows per block.
- Purpose: test whether the current draft Q6_K stage1 cost is dominated by q8_1
  activation reloads and final-reduce entries rather than register pressure in
  the per-output Q6 body. Correctness passes against the q8_1/Q6_K oracle for
  fused and split stage1+stage2 paths. Same-session denseq8all smoke rejected it:
  retained control **71.74 tok/s**, cycle **13.961 ms/output**,
  `draft_initial` **2.479 ms/output** vs pack16 **71.72 tok/s**, cycle
  **13.963 ms/output**, `draft_initial` **2.487 ms/output**, with identical
  acceptance. Draft rocprof confirms the kernel-family loss:
  `gguf_q6_k_pack16_gemv_q8_1_dp4a_top1_stage1` is **3.684 ms/cycle** vs the
  retained pack8 stage1 **3.603 ms/cycle**.
- Remove when: the draft Q6_K top-1 path moves to a different retained
  body/layout or a fused top-1/sampler path supersedes this evidence hook. It is
  not a performance route and should not update the active llama-compat headline.

## `HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE=x8_dscale` (diagnostic rejected)
- Added 2026-07-01. Bench flag
  `--resident-mtp-draft-q6-top1-stage1-shape x8_dscale` and suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8dscale-f32ssm`
  / `...-x8dscale-f32ssm-allsync` exercise the retained X8-packed Q6_K draft
  lm-head top-1 layout with an extra X8-aligned FP32 `d*scale` sidecar.
- Purpose: test whether the remaining retained X8 Q6_K top-1 cost is dominated
  by repeatedly unpacking/multiplying Q6 block scales inside the dot body.
  Correctness passes against the q8_1/Q6_K oracle for fused and split
  stage1+stage2 paths, but draft-chain rocprof rejects the route:
  `benchmarks/results/2026-07-01-gguf-mtp-draft-rocprof-llama-compat-b2-q6-x8dscale.json`
  reports host wall **6.805 -> 8.023 ms/cycle**, kernel time
  **6.427 -> 7.615 ms/cycle**, and `draft_lm_head_q6_top1`
  **3.648 -> 4.859 ms/cycle** versus the retained X8 artifact. Extra FP32
  sidecar memory traffic/register pressure is worse than recomputing scales.
- Remove when: a later draft Q6_K top-1 body/layout or fused top-1/sampler route
  supersedes the current X8 evidence set. This route is evidence only; do not
  promote or rerun full-suite unless a separate change materially alters the
  dscale memory path.

## `--fused-b1-block-probe` / `resident-fused-b1-block-direct-cap32k-minrows2-pmin05`
- Added 2026-06-30. Bench flag `--fused-b1-block-probe` keeps the retained
  adaptive B1-probe policy, but lets B1 probe cycles verify `[prev, draft0]` with
  one strict two-row target block instead of the serial target step loop. The suite
  route mirrors `resident-b1-probe-block-direct-cap32k-minrows2-pmin05` plus this
  flag. **Default off** until a full-suite row proves it improves wall time.
- Purpose: test the first queued llama.cpp-parity fix from
  `docs/MTP-LLAMACPP-PARITY.md`: remove or shrink `target_serial_verify_step`
  without merely shifting the same cost into `target_block_verify_total`.
- Remove / promote when: promote into the retained route only if exact full-suite
  B5 beats the current default and stage buckets show serial verifier cost falls
  below ~2 ms/output with no acceptance regression. Otherwise delete the flag/route
  after the parity A/B is recorded.

## `--target-block-direct-partial-replay-mode`
- Added 2026-07-02. Bench and forced-target-probe flag with choices
  `serial-exact` (default), `serial-state-only`, `direct-commit`,
  `bulk-state-only`, and `native-state-only`. It only affects direct-state block verification when a
  bulk verifier block is rejected or partially accepted. The retained
  `serial-state-only` mode restores the snapshot, advances the accepted prefix
  through `verify_target_block_serial_exact(..., advance_state_only=True)`, and
  skips replay LM-head sampling; target tokens still come from the original
  full-block scoring pass. The active llama-replication `direct-commit` mode
  commits the captured verifier row on rejected/partial blocks, matching
  llama.cpp's normal MTP accept lifecycle rather than serial-prefix replay. The
  rejected bulk/native modes replay the accepted
  prefix with `verify_target_block(..., advance_state_only=True)`, with
  `native-state-only` using the native row-serial-attention verifier only for
  that state replay.
- Purpose: reduce the semantic-safe `llama-compat` replay/commit bucket while
  preserving state lifecycle, and provide a separate llama-style replication
  lane. `--llama-compat` promotes unspecified `serial-exact` replay to
  `direct-commit`; explicit serial-state, bulk, and native diagnostic modes remain
  opt-in.
- Result: `direct-commit` is retained for the llama-compat replication lane. The
  full-suite row
  `benchmarks/results/2026-07-02-ar-mtp-llama-compat-directcommit-partial-full.json`
  moves **51.85 -> 60.56 tok/s**, cycle **19.308 -> 16.534 ms/output**,
  verifier drain **16.891 -> 14.071 ms/output**, replay/commit
  **2.489 -> 0.043 ms/output**, and replay rows **38 -> 0** versus the
  serial-state control. The lifecycle diagnostic
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-directcommit-partial-compare.json`
  intentionally diverges from serial replay at cycle 3 with matching visible
  token `[65342]`; that is expected for llama-replication, not an exact-state
  claim.
- Exact-control result: `serial-state-only` is retained as the semantic-safe
  control.
  The lifecycle comparator artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-serial-state-only-partial-replay-compare.json`
  reports `first_mismatch: null`, and the full-suite row
  `benchmarks/results/2026-07-02-ar-mtp-llama-compat-serial-state-only-partial-replay-full.json`
  moves **50.96 -> 51.85 tok/s**, cycle **19.645 -> 19.308 ms/output**,
  verifier drain **17.222 -> 16.891 ms/output**, and replay/commit
  **2.775 -> 2.489 ms/output** with unchanged acceptance/economy.
- Rejected diagnostics: artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-bulk-state-only-partial-replay-compare.json`
  reports `first_mismatch` at cycle 3. The visible token still matches
  `[65342]`, but `bulk_state_only_replay` diverges from
  `serial_exact_accepted_prefix` in hidden seed plus Conv/GDN state across 61
  fingerprints. The active-shape native replay artifact
  `benchmarks/results/2026-07-02-mtp-state-lifecycle-native-state-only-partial-replay-active-compare.json`
  also fails at cycle 3 with matching visible token `[65342]` but 59
  hidden/linear-state mismatches.
- Remove when: parity closure picks the final compat transaction policy. If
  `direct-commit` remains the llama-replication path, collapse the route/flag
  surface so only the named compat mode and the exact serial-state control remain.
  If exact-state semantics become the compat target, delete directcommit as a
  perf diagnostic. Do not promote bulk/native state-only replay into any retained
  route.

## `--verify-lm-head-q6-top1-dp4a` / verifier lm-head X8 sidecar
- Added 2026-07-01. Bench flag `--verify-lm-head-q6-top1-dp4a` sets
  `HIPENGINE_GGUF_LM_HEAD_Q6_X8_SIDECAR=1` before materialization and
  `HIPENGINE_GGUF_VERIFY_LM_HEAD_Q6_TOP1_DP4A=1` at runtime. Suite routes
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-vlmheadtop1`
  and `...-allsync` exercise it on top of the active llama-compat lane.
- Purpose: test whether the verifier-side `target_block_lm_head_sample`
  bucket can copy the draft-side q8_1/dp4a Q6_K top-1 economy by skipping full
  verifier logits plus argmax. This is accuracy-traded and default-off; exact
  verifier lm-head sampling remains the shipped behavior.
- Remove / promote when: promote only inside the llama-compat replication lane
  if a full-suite B2 row moves total wall and `target_block_lm_head_sample`
  toward the llama.cpp verifier target without unacceptable row-economy loss.
  Delete the route if smoke/full-suite shows the extra X8 sidecar/top-1 path does
  not move `target_block_verify_total` or if a later fused verifier sampler
  supersedes it.

## `HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL`
- Added 2026-07-02. Bench flag `--resident-mtp-draft-router-row-parallel`,
  draft-profile flag `--router-row-parallel`, and suite route
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow`
  route the resident MTP draft F32 router projection through the row-parallel
  `qwen35_router_logits_f32_f32w` kernel instead of the generic one-block
  `hipengine_mtp_linear_f32` path.
- Purpose: retained llama.cpp-replication optimization. Full-suite B2 moved
  **63.63 -> 64.41 tok/s** and cycle **15.735 -> 15.547 ms/output** with
  unchanged acceptance/economy. Draft-chain sync attribution moved
  `draft_run_ffn_router_linear` **0.508 -> 0.048 ms/cycle**.
- Remove / promote when: make this the unconditional resident draft router
  projection once it is either promoted beyond the llama-compat lane or no
  longer needs A/B isolation. Delete the env/CLI flag and old generic-router
  fallback route after the next parity checkpoint no longer needs the direct
  control.

## `HIPENGINE_RESIDENT_MTP_DRAFT_DENSE_Q8_DP4A` (diagnostic rejected)
- Added 2026-07-02. Bench flag `--resident-mtp-draft-dense-q8-dp4a`,
  draft-profile flag `--dense-q8-dp4a`, and suite route
  `llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8`
  route resident draft dense Q8_0 F32 projections through F32->q8_1 plus
  raw-Q8 dp4a float-output wrappers.
- Purpose: test whether copying the verifier/llama.cpp q8_1/raw-Q8 dp4a economy
  into the draft dense projections (`eh_proj`, Q/K/V, attention output, shared
  gate/up, shared down) closes the non-Q6 draft drain gap.
- Result: draft-chain rocprof moved the intended kernel bucket, but full-suite
  B2 rejected the route: active router-row **64.41 tok/s / 15.547 ms/output**
  vs draftdenseq8 **64.14 tok/s / 15.612 ms/output**, with worse acceptance and
  target rows/output.
- Remove when: the next parity checkpoint no longer needs this negative
  evidence, or if a future fused dense-draft design replaces the standalone
  quantize+dp4a calls. Do not promote the flag as-is.

## `HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED` (diagnostic rejected)
- Added 2026-07-02. Bench flag
  `--resident-mtp-draft-selected-silu-down-fused`, draft-profile flag
  `--selected-silu-down-fused`, and suite routes ending in `-siludown` route
  selected MoE `silu(gate)*up` directly into a Q5_K selected-down GEMV.
- Purpose: test a llama.cpp-shaped fused GLU/down idea without changing row
  economy or draft precision. The fused kernel is bit-exact versus the existing
  BF16 chain (`silu_mul_separate_out_bf16` + `gguf_q5_k_selected_gemv_bf16_bf16_out`).
- Result: rejected by draft-chain profile before full-suite. It removes one
  launch, but the fused Q5 body is slower: active router-row control
  **5.973 ms/cycle kernel / 7.044 ms/cycle host** vs fused
  **6.054 ms/cycle kernel / 7.206 ms/cycle host**; selected-down family
  **0.325 -> 0.391 ms/cycle**.
- Remove when: the next parity checkpoint no longer needs this negative evidence,
  or if a different fused Q5 selected-down body replaces it and wins the draft
  parent profile. Do not promote the flag as-is.

## `HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING`
- Added 2026-07-03. Default-off verifier diagnostic that seeds the target
  verifier F32 residual buffer from host-dequantized `token_embd.weight` rows
  when `HIPENGINE_GGUF_VERIFY_F32_RESIDUAL=1` is enabled. The BF16 token
  embedding launch still runs and still populates the BF16 mirror path.
- Purpose: isolate llama.cpp MTP parity for layer-0 target input construction.
  The first task-9/cycle-3/row-2 run closed `hidden_in` and
  `attn_norm_f32_scratch` to exact llama.cpp parity, but it did not flip the
  bonus token; the remaining split moved to F32-input projection/dequant and
  later residual/LM-head amplification.
- Remove / promote when: remove once the projection/dequant split is resolved
  and this host-side diagnostic is no longer needed. Promote only by replacing
  it with a real device-side F32 embedding path if full-suite llama-compat
  evidence shows it is required and non-regressive; do not keep host dequant/H2D
  in a timing route.

## `--record-draft-stage-stats`
- Added 2026-07-02. Bench flag that records compact FP32 summaries for resident
  MTP draft sub-stage tensors and dense MTP K/V cache rows in
  `draft_hidden_state_trace`. Default-off extensions
  `--record-draft-cache-rows` and `--record-draft-attention-debug` add selected
  history rows plus host-recomputed dense-attention score/weight diagnostics.
  It forces host-chain resident drafting when enabled so intermediate buffers
  can be read back, and is not a timing route.
- Purpose: diagnose the remaining llama.cpp parity miss after hidden-state
  tracing narrowed the first divergence to the depth-0 MTP block. The first use
  found and fixed the resident MTP RoPE dimension mismatch (`qk_head_dim=256`
  vs model `rope.dimension_count=64`); the attention-debug extension ruled out
  hipEngine's dense-attention kernel math for the seq-position-49 divergence.
- Remove when: llama.cpp tensor/KV parity is either achieved or superseded by a
  more complete graph-tensor trace facility. Keep it default-off until then.

## `--record-target-topk-scores`
- Added 2026-07-03. Bench flag that asks `verify_target_block()` to copy the
  already-materialized full target lm-head logits back to host for block-verifier
  rows and serialize compact `target_lm_head_score_rows` with top-k plus
  candidate-token scores. `--target-score-candidate-tokens` adds explicit
  llama.cpp near-tie tokens to the candidate list. When score rows are present,
  the same diagnostic also emits compact `target_hidden_seed_rows` summaries so
  the scored verifier hidden row can be lined up with llama.cpp `verify_h`
  traces without dumping full hidden vectors in the normal artifact.
- Purpose: diagnose the active llama.cpp parity miss on target verifier
  near-ties without relying on forced-target replay. The first smoke artifact
  `benchmarks/results/2026-07-03-mtp-target-score-capture-smoke.json` populated
  three live target verifier rows on the active `llama-compat` direct-commit
  shape. The hidden-seed follow-up artifact
  `benchmarks/results/2026-07-03-mtp-mixed-ja-en-translate-target-hidden-scores-live.json`
  captures the live task-9/cycle-3/row-2 hidden summary for the `8940` vs `668`
  rank flip.
- Remove when: target hidden-to-logit parity is closed or replaced by a broader
  cross-engine tensor trace. Keep it default-off; the extra full-logit D2H copy
  makes it invalid for retained timing claims.

## `HIPENGINE_GGUF_AR_PACKED_DECODE` (default-on packed decode)
- Added 2026-07-05 as a packed-verifier AR diagnostic, then replaced by the
  retained decode-shaped packed AR path. The current default-on route calls
  `Qwen35GGUFResidentSession.step_batch_native(..., scatter_state=False)` for
  prepared multi-prompt GGUF greedy AR, keeps packed multi-slot state canonical
  across decode cycles, and scatters back only before stream/scalar fallback or
  a changed chunk layout.
- Purpose: provide the first useful GGUF AR c>N server backend after fixing
  default-route request coalescing while preserving each request's canonical
  state. The exact route uses c1 per-slot linear-attention state slices and
  keeps full attention plus MoE/FFN row-batched. Deferred flush copies the full
  live KV prefix instead of only the last dirty row.
- Result: retained-flag steady c4 and c4→c3→c2→c1 middle-hole shrink are
  token/Conv/GDN/live-KV byte-exact against independent c1. The July 5
  **50.89/56.79/59.17 tok/s** packed-decode rows remain useful history for the
  prior token-only algorithm, but cannot baseline the repaired path. A single
  exact-accounting diagnostic measured c1/c2/c4/c8
  **35.34/51.07/59.23/59.19 generated tok/s**; `performance_claim=false`.
- Remove when: per-layer hidden, live cancellation/admission, profiler, and
  repeated exact-accounting c=1/2/4/8 gates pass. Then collapse the opt-out and
  keep only unsupported-shape fallback paths.

## `HIPENGINE_GGUF_AR_PACKED_PREFILL` (default-on packed prompt prefill)
- Added 2026-07-05 after the packed-decode AR route exposed prompt prefill as
  the remaining c>N server AR limiter. The current default-on route calls
  `Qwen35GGUFResidentSession.prefill_batch_native(...)` for multi-prompt GGUF
  greedy AR, packs prompt rows slot-major, scatters the resulting KV/recurrent
  state back to each resident session, and samples only each slot's final
  prompt row before entering packed decode.
- Purpose: remove the serial per-slot prompt-prefill wall from coalesced
  OpenAI-server AR batches without reusing the MTP verifier result contract.
  The rejected verifier-as-prefill probe sampled/copied all prompt rows and
  measured c=8 **50.56 tok/s**, below the retained packed-decode baseline.
- Result: the 2026-07-13 state audit found the old packed full-attention decode
  reduction first changed BF16 layer output at layer 31, then Conv/GDN state at
  layer 32 and live KV at layer 35. Packed prefill now uses the span-aware paged
  prefill reduction below 512 rows; if any slot crosses the AOTriton threshold,
  full attention runs slot-locally with c1 math while linear/MoE remains packed.
  Steady c4 and ragged `[512,64,64,64]` are token/Conv/GDN/live-KV exact. The old
  **65.91/82.41/63.17 tok/s** row predates these gates and exact generated-ID
  accounting; it is historical rather than a current retained baseline.
- Remove when: broader server/API cancellation coverage and repeated
  exact-accounting c=1/2/4/8 plus profiler evidence are green. Keep fallback for
  total prompt slabs beyond the current packed hidden-row guard.

## `HIPENGINE_GGUF_MTP_SERVER_VERIFY_FINAL_STATE_FASTPATH`
- Added 2026-07-06 as a default-off MTP serving diagnostic after the first
  no-capture packed-verifier probe changed MTP economy. The corrected version
  keeps packed slot segments through Conv/GDN prefill, mutates per-slot packed
  final linear state directly, and falls back to accepted-prefix replay for
  partial/reject cycles.
- Purpose: test whether skipping per-row linear-state capture can beat the
  retained captured-row verifier once the no-capture path is semantically
  equivalent for packed c>N serving.
- Result: rejected on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
  Qwen3.6-35B-A3B `UD-Q4_K_M`, natural24 `max_tokens=24`, 5 ms server batch
  window. c=4 measured **66.75 tok/s** in
  `benchmarks/results/2026-07-06-hipengine-server-mtp-natural24-c4-bw5-finalstate-fastpath2.json`
  versus retained **76.83 tok/s** for
  `benchmarks/results/2026-07-06-hipengine-server-mtp-natural24-c4-bw5-rowtilechunk-verify.json`.
  Acceptance stayed identical (**0.8545**, draft **165**, accepted **141**), but
  `target_state_commit_ms` rose **10.443 -> 405.559 ms** because
  partial/reject cycles must replay the consumed prefix without captured rows.
- Remove when: a compact selected-row capture path exists, or if no follow-up
  uses the segment-aware no-capture kernel. The flag must stay default-off and
  must not be used for retained timing claims.

## `HIPENGINE_GGUF_MTP_SERVER_ROLLING_SLOTS`
- Added 2026-07-06 as a default-off MTP serving diagnostic while trying to lift
  the four-request MTP route cap without using a true width-8 verifier. The
  route keeps at most four live resident slots, opens replacements in warmed
  widths when possible, and can hold a stable packed-verifier owner session so
  replacement slots do not allocate owner workspaces mid-batch.
- Purpose: test whether c=8 can avoid the fixed two-backend-group barrier while
  preserving the retained four-slot packed verifier shape.
- Result: rejected on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
  Qwen3.6-35B-A3B `UD-Q4_K_M`, natural24 `max_tokens=24`, 5 ms server batch
  window. Naive rolling measured **11.22 tok/s** at c=8; the stable-owner /
  warmed-width variant improved to **61.23 tok/s**, still below retained
  **79.61 tok/s**. Economy stayed normal (`draft=165`, `accepted=141`, accept
  rate **0.8545**), but replacement slot opening/prefill exposed
  **14.613 s** aggregate `slots_open_ms`. The default MTP route cap remains
  four; guarded default c=8 rerun measured **78.91 tok/s**.
- Remove when: a true cap>4 MTP scheduler can pre-open/reuse replacement slots
  without exposing slot-open/prefill wall, or after the next c>N MTP scheduler
  direction supersedes it. The flag must stay default-off and must not be used
  for retained timing claims.

## `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_*`
- Added 2026-07-09 as a default-off PARO c>N diagnostic:
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_SIZE`,
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_LAYERS`, and
  `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_SUFFIX_ROW_CHUNK_INCLUDE_GATE`.
  The path keeps batch full-attention QKV/append/context, then chunks either
  O/post/MoE or gate/O/post/MoE over row sub-batches. It is exposed through
  `scripts/qwen35_batch_retained_bench.py` and
  `scripts/qwen35_batch_hidden_bisect.py`, records suffix-rowchunk metadata in
  `last_batch_decode_execution`, and blocks native-caware claims.
- Purpose: isolate the remaining gfx1151 c6 full-attention rowchunk tax after
  context-only rowchunking rejected. It tests whether the green selected
  full-layer rowchunk bridge is paying for post-context suffix work.
- Result: rejected on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with
  `Qwen3.6-35B-A3B-PARO-packed`, `w4_paro`, rows=6, prompt=512,
  decode=16, selected-c1 MoE, forced small-batch shared expert, and suffix
  rowchunk2 on layers `3,7,11,15,19,23,27,31`. After batch context+gate:
  **106.864 tok/s**, median **53.609 ms**, generated-token red at token 9
  (`12` vs c1 `27`). Including gate in the suffix chunks: **107.508 tok/s**,
  median **53.189 ms**, same token-9 failure. Compact summary:
  `benchmarks/results/2026-07-09-hipengine-qwen35-c6-suffix-rowchunk-rejects-summary.json`.
- Remove when: c6 full-attention rowchunk isolation moves to lower-level
  hidden/KV source tracing or a retained green non-rowchunk c6 path exists.
  Keep the flags default-off and do not use them for retained timing claims.

## `scripts/qwen35_batch_hidden_bisect.py --compare-full-attn-rowchunk-boundary`
- Added 2026-07-09 as a default-off PARO c>N diagnostic mode. It compares two
  native rows=6 batch variants directly: no-rowchunk full attention versus the
  selected full-layer rowchunk repair, using the existing hidden-bisect summary
  machinery but labelling the rowchunk repair as the comparison peer instead of
  an independent c1 oracle.
- Purpose: isolate whether the remaining c6 full-attention rowchunk tax comes
  from KV append/page placement, context-only work, suffix work, or from a
  whole-layer numerical boundary introduced by rowchunking.
- Result: the L8 trace showed layer 3 full-layer rowchunk output drift still
  under tolerance (`0.000122 max_abs`), layer 4 `attn_input` first over
  tolerance (`0.001953`), and layer 7 `attn_input_pre_qkv` at `0.0078125`.
  The corrected combined context+suffix rowchunk probe records both
  `native_context_row_chunks` and `native_suffix_row_chunks_include_gate` on
  layers `3,7,11,15,19,23,27,31`, but remains generated-token red at token 2
  (`220` vs c1 `17`) and slower than the current green selected full-layer
  rowchunk bridge (`103.998 tok/s`, median `54.865 ms`). Compact summary:
  `benchmarks/results/2026-07-09-hipengine-qwen35-c6-rowchunk-boundary-combined-summary.json`.
- Remove when: the c6 full-layer rowchunk tax is either fixed or the scheduler
  avoids live c6 groups in retained/default operation. Do not use this
  comparison mode for throughput claims.

## `HIPENGINE_QWEN35_AOTRITON_ISOLATED_PREFILL_STREAM`

- Added 2026-07-12 as a default-on PARO prefill rollback/bisection control.
  The retained path creates one lazy nonblocking stream plus two reusable HIP
  events for AOTriton query rows above the proven-safe 256-row bucket, and
  keeps all pre/post work on the caller stream. Setting the flag to `0`
  restores the old same-stream dispatch without changing model math.
- Purpose: isolate AOTriton's high-scratch dispatch from the queue used by
  later low-scratch linear-attention convolution kernels. On gfx1151 the same
  captured convolution changes from about `1.83 ms` after same-stream
  4096-row AOTriton to `119 us` when AOTriton uses the isolated queue. The
  clean 256-row AOTriton trace uses much less scratch (`992/1008` versus
  `2560` bytes) and does not trigger the cliff. The first
  final clean matched prefill A/Bs improve 4K `885.141 -> 1089.031` (+23.03%),
  32K `765.316 -> 906.145` (+18.40%), 64K `621.691 -> 716.775` (+15.29%),
  and 128K `418.838 -> 474.641 tok/s` (+13.32%). Decode stays within
  `-0.16%..+0.12%`, tracked peak is unchanged, and every shape matches sampled
  seed, final hidden, all 30 Conv/GDN state families, and all 10 live K/V
  families. The 1K 256-query negative control never enters isolation and is
  not promoted. Retained evidence:
  `benchmarks/results/2026-07-12-gfx1151-paro-aotriton-stream-isolation.json`.
- A clean W7900/gfx1100 transfer matrix measured the earlier global-isolation
  policy in balanced 15-sample legs. Isolated prefill changes by
  `+1.638%/+0.495%/+0.192%` and total measured wall falls by
  `1.653%/0.127%/0.562%` at 512/1K/4K, with byte-exact hidden/state/KV at every
  shape. The merged threshold intentionally leaves the 256-query 512/1K path
  same-stream; its 4K/4096-query result directly validates the scoped gfx1100
  default. Retained evidence:
  `benchmarks/results/2026-07-12-w7900-gfx1100-paro-gfx1151-transfer.json`.
- Remove when: the ROCr/AOTriton queue-scratch issue is fixed upstream or the
  rollback has survived one release cycle. Then remove the opt-out and its
  duplicate same-stream route; keep one proven scheduling policy.
