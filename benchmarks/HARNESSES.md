# Benchmark Harness Catalog

There is no single "run everything" benchmark. Different questions are answered
by different harnesses, each with a specific timing scope, numerical contract,
and shape. The table below maps each harness to the axes it owns so results are
only compared like for like. A check mark means the harness measures that axis;
a blank does not mean zero. Run hipEngine rows through the hermetic thecrock
wrapper for the target architecture; see [`docs/BENCHMARK.md`](../docs/BENCHMARK.md).

## Capacity testing: ALWAYS probe first (two-tier protocol)

**Never run a full-prompt ladder point to answer "does context N fit".**
The resident scratch's per-layer KV caches, scales, and metadata tables
are sized by `max_positions` at session initialization - not by the
prompt - and the bulk prefill workspace acquires before any attention
work. The allocation decision (and the tracked peak) is therefore fully
determined within the first ~3 minutes, while a full synthetic prompt
at 160K+ tokens costs ~90 minutes per point on the 27B. This cost was
actually paid on 2026-09-09 (three full ladder points bracketing a bound
that two 3-minute probes would have bracketed); do not repeat it.

- **Tier 1 - allocation probe** (`scripts/gguf_capacity_probe.py`,
  ~3 min/point): session at the target `--max-sequence-length` with the
  route's env vars, one short bulk prefill, a few decode steps, finite
  logits, tracked peak reported. Answers every *does-it-fit* question:
  bracketing bounds, OOM checks, memory-regression checks across
  changes. Binary-search the bound here.
- **Tier 2 - full-prompt harness point** (`qwen35_gguf_bench.py`,
  ~1 min per 32K tokens): only when the question requires the prefill
  path to actually RUN at that depth - kernel stability across the full
  context, finite logits after deep attention, or a tok/s-at-depth
  claim. Use it to CONFIRM the bound Tier 1 found, with at most a
  pass/fail point or two; never to search for one.

A "capacity ladder" artifact should state which tier produced each point.
A Tier-1 point is a memory-envelope claim; a Tier-2 point additionally
carries a completion/stability claim at depth.

**Legend:** AR = true no-MTP autoregressive decode; MTP = speculative
multi-token-prediction decode with a true-AR denominator where a ratio is
reported; Prefill = prompt-processing tok/s; Decode = generation tok/s; Mem =
graphics-memory usage; Conc = a concurrency sweep.

| Harness (`scripts/`) | What it answers | AR | MTP | Prefill | Decode | Mem | Conc | Canonical entrypoint |
| --- | --- | :-: | :-: | :-: | :-: | :-: | :-: | --- |
| `qwen35_readme_sweep.py` | Single-request prefill/decode/memory per shape (llama-bench-style), one resident session, per-shape reset | ✓ | | ✓ | ✓ | ✓ | | `--engine gguf --model <model> --backend hip_gfx1151 --workloads 512/128 1K/128 ...` |
| `qwen35_gguf_bench.py` | GGUF c=1 AR prefill/decode, fresh resident session per run, HIP-graph decode | ✓ | | ✓ | ✓ | ✓ | | `--model <model> --prompt-length 512 --decode-tokens 128` |
| `gguf_prefill_route_ab.py` | **Prefill entry-point A/B and multi-chunk oracle anchor**: scalar bulk parent vs the packed slot-local route the server is forced onto, one process, deterministic varied prompt - tok/s, tracked peak, greedy IDs, logit agreement. `--assert-entry-agreement` exits non-zero when the entries disagree; above the 1,024-row chunk cap that is the multi-chunk oracle-history contract (fixed 2026-09-10, per-layer keying). Always pass the shipping selectors (defaults on) or absolute rates are ~6x low | ✓ | | ✓ | | ✓ | | `--prompt-length 2048 --max-sequence-length 16384 --assert-entry-agreement` (route envs set by caller) |
| `gguf_packed_kv_import_profile.py` | **Packed slot-local KV-import census**: counts the per-slab whole-history KV imports (calls/rows/`memcpy` dispatches/bytes), Conv/GDN import preservation, per-slab sync counts, wall time, a sync-neutralized CPU-submission pass, and final-logits sha256 + greedy IDs so an import change can be gated on exactness. Measures copy work directly - never infer import cost from throughput alone (2026-09-10: removing the import changed no wall time) | | | ✓ | | ✓ | | `--rows 1024,2048,4096,8192 --json out.json` (route envs set by caller) |
| `gguf_server_cwidth_probe.py` | **C-width decode baseline (IKV-C2 reference)**: N concurrent fixed-fixture completions on the real int8 server per width, plus a measured same-server serial control, reporting complete-request throughput, per-lane walls, per-lane `usage` token counts, and measured decode/prefill model-step and route/fallback counter deltas. No decode-only rate: the withdrawn subtraction estimate is not replaced by another inference. `--server-log-dir` captures each width's server stdout/stderr (previously discarded, which made a failed width undiagnosable) and `--min-vram-free-gib` skips a width whose pre-launch free VRAM is too low, so a device still held by another process cannot be published as an out-of-memory rate | | | | | | | | `--widths 1,2,4 --server-log-dir /tmp/cw --json out.json` |
| `qwen38_int8_packed_transition_gate.py` | **IKV-C2 packed transition gate**: prefills six sessions through the exact scalar path, then replays a 4 -> 2 -> 4 lane schedule (two lanes retire, two newcomers take the freed lanes) as packed steps, diffing every active row's token, logits, captured hidden state, and Conv/GDN + INT8 KV payload/scale state against its own c1 trajectory, and asserting excluded retired lanes stay byte-frozen. Correctness/ownership, never a benchmark | ✓ | | | ✓ | | | `--diagnostic-direct-rows 4 --json out.json` |
| `qwen38_int8_batch_decode_gate.py` | **IKV-C2 steady-state model gate**: four packed-AR sessions at a fixed physical width versus independent c1 - token exactness, full-vocabulary logit KL, captured hidden hashes, Conv/GDN and INT8 KV payload/scale state, and the per-step route/physical-width manifest. `--diagnostic-direct-rows N` runs a pre-promotion width the artifact has not admitted yet | ✓ | | | ✓ | | | `--diagnostic-direct-rows 4 --decode-steps 4 --json out.json` |
| `qwen38_int8_batch_decode_ownership_trace.py` | **IKV-C2 packed-ownership trace**: runs `rocprofv3 --kernel-trace` over the `c4-ragged` primitive-gate case and reduces the CSV to the launch-count and kernel-time evidence that the packed INT8 producer and its strided reducer each launch once for all rows while the c1 leaf launches once per row. Warms the JIT cache outside the profiler and sets `HIPENGINE_REQUIRE_CACHED_BUILD=1` so an accidental compile inside the profiler fails instead of corrupting the trace; refuses a per-row batch path rather than certifying it; resolves the capability snapshot live so the artifact cannot record an admitted width the registry has moved past. `--trace-dir` re-parses an existing trace without a GPU run | | | | ✓ | | | `--case c4-ragged --json out.json` |
| `gguf_p6e_cancel_refill_proof.py` | **P6e service cancel/refill proof**: drives the real FastAPI app under `uvicorn` on a loopback socket, streams a reference short completion, then sends a multi-round long prompt over a **raw TCP connection** and closes it 600 ms after admission to cancel a prefill that is genuinely in flight (`bytes_before_close` is 0, so no token had been produced yet). Then refills the short prompt and verifies **eight** gates: the cancellation counter moved, the refill reproduces the reference text exactly (sha256), the short survivor stays exact when the concurrent long request is **blocking (non-streaming)** and when the concurrent reader is a **slow consumer** (throttled per SSE line), acknowledgement and inter-token gaps stay inside limits declared before measuring, the scheduler drains with prefill owner bytes back to baseline, and — the load-bearing one — **`resumable_path_engaged`**, which requires the layer-outer route's shared per-layer oracle owner to be observed non-zero during a long prefill. That last gate exists because the resumable prefill lives on the layer-outer route (`HIPENGINE_GGUF_PACKED_LAYER_OUTER`, default OFF, and additionally requiring `kv_attention_source == "int8_direct"`); without it the other gates pass on the default chunk-outer path and read as P6 evidence. **It currently fails**, so the artifact is a diagnostic, not a P6 certification. A raw socket is required because during prefill the SSE stream carries no bytes, so a blocked reader cannot observe an abort and a `TestClient` close only takes effect once the reader returns. Exits non-zero when a gate fails; `performance_claim` false | | | | ✓ | | | `--long-prompt-rows 3072 --short-prompt-rows 512 --cancel-delay-ms 600 --slow-consumer-line-delay-ms 40 --packed-layer-outer` |
| `gguf_resumable_prefill_gpu_proof.py` | **P6e resumable-prefill GPU proof**: proves on hardware what P6c only proved with monkeypatched CPU tests. Creates one resident INT8-direct batch with three sessions (one-shot target, resumable target, interleaved decoder), reads the real bulk-prefill row capacity, prefills a multi-round prompt through the one-shot packed entry (Arm A) and through `prefill_batch_native_layer_outer_resumable` in bounded layer segments (Arm B), and runs a decode step on the third session between every pair of segments. Four gates, all declared before measuring: exact continuation (equal sampled token), real yield (more than one segment, each above a work floor), bounded decode gap (interleaved p95/max within declared factors of the standalone decode step), and cleanup (the suspended scratch does not outlive the checkpoint). Exits non-zero when a gate fails. Eager, greedy, prefix-off, MTP-off; `performance_claim` false | | | | ✓ | | | `--prompt-rounds 3 --layer-budget 4 --decode-tokens 24` |
| `gguf_c1_decode_attribution.py` | **C1 decode attribution (P5)**: runs the raw resident session (the ladder's R0 arm) with a ROCTx range around every measured decode step, traces kernels, markers, HIP runtime calls, and memory copies with `rocprofv3`, and slices everything to the decode-step windows so model load, prefill, and warmup are excluded. Reports kernel families, the device union, HIP API time and its top functions, copy counts and time by direction, and an additive decomposition `window = device union + HIP API union + residual` that asserts it balances; the residual is reported rather than attributed to a guessed cause. States graph capture cost and amortization explicitly (this route is eager, so no capture cost is inside the windows). Copy byte counts are not available from these trace types and are not invented. Warms the JIT cache outside the profiler with `HIPENGINE_REQUIRE_CACHED_BUILD=1`; `--trace-dir` re-parses without a GPU run | | | | ✓ | | | `--prompt-rows 2048 --decode-tokens 63 --json out.json` |
| `gguf_decode_boundary_ladder.py` | **R0-R4 decode-boundary ladder (P5)**: same model/quant/KV policy/prompt token IDs/sampler/eager mode at each stack layer - R0 raw session prefill+step per-token walls, R1 the packed slot-local server entry, R3 the public LLM token-ID path; per-token median/p95 and stage deltas. Run with PYTHONPATH pinned to the repo (site-packages may point at another worktree) | | | | | | | | `--prompt-rows 2048 --decode-tokens 64 --json out.json` |
| `gguf_server_allocation_probe.py` | **Server-faithful allocation probe (tier 1, server route)**: one request through the real server process (actual pool binding, slot views, packed slot-local prefill entry, AOTriton default), scraping the owner-deduplicated observability gauges at high cadence plus the while-live oracle peak gauges - per-domain envelope (pool/lease/workspace/oracle/hidden) without full-prompt searching | | | | | ✓ | | `--max-context-tokens 32768 --prompt-rows 2048 --json out.json` |
| `gguf_capacity_probe.py` | **Capacity-tier-1 probe**: does a context size FIT (allocation validity, tracked peak) - minutes per point, no full prefill | | | | | ✓ | | `--max-sequence-length 229376` (route envs set by caller) |
| `qwen38_prefill_sweep_trace.py` + `qwen38_prefill_sweep_analyze.py` | Fixed-row prefill wall/HIP-event capture plus dispatch-matched quant-family sweep attribution | | | ✓ | | | | `--rows 16,35,48,72,96,256,288,536,1024` |
| `gguf_true_ar_category_bench.py` | True no-MTP AR baseline over the mtp-bench category suite (the legitimate MTP speed denominator) | ✓ | | ✓ | ✓ | | | `--model <model> --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl` |
| `gguf_mtp_category_bench.py` | MTP category matrix over budgets 1..8 with guarded objective extraction; attach a true-AR baseline for ratios | | ✓ | | ✓ | | | `--budgets 1,3,5 --objective-budget b5` |
| `gguf_mtp_long_context_gate.py` | Eager-native MTP correctness vs serial-exact teacher across context/page/budget/acceptance boundaries; optional real host-proposal AR-ID gate (no speed claim) | ✓ | ✓ | | | | | `--cycle-ends 1016-1032,4K --candidate-budgets 1,2,3 --fail-on-fail` |
| `gguf_ar_mtp_suite.py` | One-command AR-vs-MTP decode ratio over the category suite under one enforced decode config | ✓ | ✓ | | ✓ | | | `--scope partial --output <json>` |
| `specdec2_perf_bridge.py` | Current-source Generation-2 true AR vs staged SPECDEC2 plus C1 direct control; complete/decode timing, ownership stages, physical C/K, exact IDs, and ROCTX leaf mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | `--backend hip_gfx1151 --concurrency 1 --budgets 1,2,3 ...` then separate `--concurrency 2,4 --budgets 2 ...` |
| `qwen35_batch_retained_bench.py` | PARO-path compact c>N batch decode; aggregate + per-request tok/s, equality vs c1, optional MTP draft depth | ✓ | ✓ | | ✓ | ✓ | ✓ | `--batch-size 8 --decode-tokens 128` |
| `qwen35_batch_gguf_diagnostic.py` | GGUF c>N generated-token correctness equality vs independent c1 (no throughput claim) | ✓ | | | | | ✓ | `--rows 8 --execute` |
| `server_f1_concurrency_bench.py` | Matched gfx1151 F1 HTTP concurrency through c32; profile-aware throughput, SLOs, routes, control, and memory | ✓ | | | ✓ | ✓ | ✓ | `--engine hipengine --model <model> --concurrencies 1,2,4,8,17,32` |
| `gguf_concurrency_baseline.py` | GGUF c1 + explicit serial c2/c4 timing controls (Phase-A route baseline) | ✓ | | ✓ | ✓ | | ✓ | `--model <model> --concurrencies 1,2,4` |
| `mtp-bench.py` | llama.cpp-compatible MTP prompt-suite benchmark (server economics); can wrap hipEngine verifier economics | ✓ | ✓ | | ✓ | | | `--mode hipengine-current` |
| `exact_token_generation.py` | Direct/HTTP generated-token identity gate (correctness, not throughput) | ✓ | ✓ | | | | | `direct --model-path ...` then `http --oracle ...` |
| `benchmark_matrix.py` | Join exact-token direct/server rows into a validated matrix report | ✓ | ✓ | | | | | `build --manifest ...` |

The concurrency scoreboards primarily come from
`qwen35_batch_retained_bench.py` (direct engine) and
`server_f1_concurrency_bench.py` (OpenAI server). Single-request tables come
from `qwen35_readme_sweep.py` and `qwen35_gguf_bench.py`. Speculative-decode
tables use `gguf_ar_mtp_suite.py` or `gguf_mtp_category_bench.py` with a
`gguf_true_ar_category_bench.py` true-AR denominator.

## Profiling these harnesses under rocprofv3 (read before launching)

A profiled Python/ctypes process must never discover the compiler: the
profiler preloads into children, and a `hipcc --version` probe inside
`rocprofv3` can block indefinitely (two stalled attempts on 2026-09-10, each
stopped after ~15-30 min with no trace written). The recipe, and the fuller
trap catalog, live in [`docs/KERNELS.md`](../docs/KERNELS.md) and
[`RDNA3-TUNING-GUIDE.md`](../docs/RDNA3-TUNING-GUIDE.md) section 4.9:

1. **Prewarm the JIT cache outside the profiler first** - run the exact
   harness once uninstrumented so every kernel is built and cached.
2. Export `HIPENGINE_COMPILER_VERSION_FILE=<file>` containing the `hipcc
   --version` output (captured on the host, not under the profiler) so the
   cache key resolves without probing the compiler.
3. Export `HIPENGINE_REQUIRE_CACHED_BUILD=1` so any cache miss fails closed
   instead of spawning `hipcc` under the profiler.
4. Prefer a short configuration (a few seconds of device work); per-launch
   interception makes launch-dense host-bound workloads many times slower
   than their uninstrumented wall time.

Update this catalog in the same logical unit whenever a harness gains or loses
an axis.

## Parity measurement manifest (R0-R4)

Server/direct parity comparisons never assemble a ratio from unrelated runs.
The matched case manifest with exact token-ID fixtures, boundary definitions
(R0 raw session → R4 HTTP), stage timing list, and control freeze list is
[`parity/r0r4-c1-fixture.json`](parity/r0r4-c1-fixture.json); the boundary
semantics live in
[`docs/SERVER-DIRECT-PARITY-ROADMAP.md`](../docs/SERVER-DIRECT-PARITY-ROADMAP.md)
section 3. Every R0-R4 timing harness must name the manifest it consumed and
record per-stage times and independent peaks.
