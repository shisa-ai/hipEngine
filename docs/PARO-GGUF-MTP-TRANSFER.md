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
- Workload: exact token-id prompts, prompt length 512, decode 128,
  concurrency c=1/2/4/8, greedy completions.
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
- Client JSON from `scripts/vllm_openai_concurrency_sweep.py`.
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
