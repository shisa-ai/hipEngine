---
status: current
owns: Source comparison, implementation priorities, test recipes, and handoff criteria for the Inkling 2 Mini packed-GGUF optimization campaign.
---
# Inkling 2 Mini: GGUF optimization implementation plan

**First assignment: compare the same hidden state on both paths.** A CPU check shows that the full-block prototype and the host trace expose *different tensors* at two loop restarts and at model exit. The reported differences there do not localize an attention bug. Once the comparison is sound, the highest-priority changes are to keep activations and routing on the GPU across visits, and to speed up expert projections at the group sizes real prompts produce. These target different costs: host waits during generation and GPU arithmetic during prompt processing.

This is the implementation companion to [Match Loom on Strix Halo](INKLING2-MINI-LOOM-OPTIMIZATION.md), which owns the target, checkpoint locks, workload matrix and definition of a match. Code audited: `inkling2-mini` at `10d814df5`, including the pinned-upload repair; diagnostics through [the November 24 boundary sweep](../../worklog/entries/20261124T131703.985686Z-lhl-inkling2-mini-34e569.md). Before taking a task, check `git status -sb`, the latest Inkling worklog entries and the actual callers. If another worker owns the resident-block repair or the GPU, use their result; work on the CPU oracle or comparison harness instead of running a competing benchmark.

## Starting point: what actually runs

| Surface | Implemented | Still open |
| --- | --- | --- |
| Packed text | [Loader](../../hipengine/loading/inkling2_gguf.py) and [generator](../../hipengine/generation/inkling2_gguf.py) run both `inkling2` (legacy root GGUF) and `inkling2_loop` (`Mainline_Support`). Physical weights and scales are shared; each visit owns its KV cache and convolution history. [Public bring-up](../../worklog/entries/20261122T005515.753787Z-lhl-inkling2-mini-6d925f.md) includes same-ID mainline llama.cpp smokes. | Do not rebuild the loader. `fast` is an explicit GGUF profile; unnamed, `strict` and `fast` are not interchangeable. These are not original-checkpoint rates. Preserve original text, vision and DSpark; GGUF vision is not a deliverable here. |
| Ordinary prefill | `Inkling2ResidentRunner.prefill_batch` collects canonical blocks even when scheduler slices are smaller. The model's block limit, default scheduler chunk and round budget are all 64 tokens. In `fast`, [`_prepare_layer`](../../hipengine/runtime/inkling2_mini_reference.py) can keep controllers, input projections, clip/normalization, attention and output convolution on device, then download the branch. The relative-position table is already read on device. | Residual mixing, routing and other visit work still return to the host. [`_resident_block`](../../hipengine/runtime/inkling2_gguf.py) and its scratch arena exist; binding `model.resident_block` is **not** a call from `_forward`. It is not a public speed result. |
| Ordinary decode | One row still goes through `session.step`. With multiple rows, [`decode_rows`](../../hipengine/runtime/inkling2_gguf.py) groups controller, attention and expert work; [`project_experts_mixed_rows`](../../hipengine/loading/inkling2_gguf_device.py) already shares gate/up and down work for eligible NVFP4 rows. The head is also grouped. | `_select_routes` returns to the CPU; residual/controller math and exchanges split the token. The [resident runner](../../hipengine/generation/inkling2_gguf_resident.py) selects the token from host logits. The GGUF plan specifies FP32 KV and eager launches. Do not describe it as CPU inference or start a second scheduler. |
| Depth, serving and drafting | [8,195-token public completion on Strix](../../worklog/entries/20261123T035026.568703Z-lhl-inkling2-mini-507bfa.md), [32,771 on W7900](../../worklog/entries/20261122T163739.159006Z-lhl-inkling2-mini-842a42.md), and [cancel/refill on both](../../worklog/entries/20261123T035001.063034Z-lhl-inkling2-mini-23aa7f.md). Explicit GGUF DSpark proposes, then verifies with the ordinary scalar target; [the serial verifier restored parity](../../worklog/entries/20261123T075822.415282Z-lhl-inkling2-mini-e049fc.md). | A fit or completion is not a throughput result. No clean same-host 32K or four-stream Loom match. Serial verification does not skip ordinary target steps; do not count proposals as a speedup. |

**Original checkpoint:** the [W7900 generator](../../hipengine/generation/inkling2_mini_gfx1100.py) is a separate path. Production text uses prompt blocks up to 256 rows; strict and prefill with image embeddings retain their scalar schedules. Its verifier can process a draft block, unlike GGUF's serial verifier. If a change touches shared traversal, run the [generation](../../tests/test_unit_inkling2_generation.py), [vision](../../tests/test_unit_inkling2_vision.py) and [DSpark](../../tests/test_unit_inkling2_dspark.py) regressions and an original-checkpoint public request. The [original W7900 width comparison](../../worklog/entries/20261121T164326.753047Z-lhl-inkling2-mini-1fc0c1.md) is its control, not a GGUF baseline.

### Timing references, not a cross-GPU leaderboard

All rows here use one stream and no drafter. The linked runs record the commands, model, profile, sampling, correctness checks and measurement boundaries. These are historical observations, not timings of the full-block prototype.

| Engine, physical host and weights | Prompt processing | Decode |
| --- | --- | --- |
| hipEngine `fast`, `epyc` W7900, **mainline `UD-Q4_K_M`** | [575 tokens through first sampled token in 2.974 s](../../worklog/entries/20261124T093241.447865Z-lhl-inkling2-mini-4b3cf8.md): ~193 input tok/s including request overhead; four interleaved pairs, same generated IDs as control. | [~18.0 tok/s after 575](../../worklog/entries/20261123T094058.230583Z-lhl-inkling2-mini-5cb5ed.md), from 16 intervals over 0.888–0.889 s. Separate Nov 23 run, not the Nov 24 prefill run. |
| hipEngine `fast`, `strixhalo` Radeon 8060S, **root `UD-Q4_K_XL`** | [Nov 22 paired run][pairs]: 0.952–0.971 s to first streamed text at 47 input; 10.145–10.595 s at 575. | 13.9–14.1 tok/s after 47; 10.7–10.8 after 575. |
| Loom 0.2.0, **same local host and file** | [Same run][pairs]: 0.215–0.230 s to first streamed text at 47 input; 0.529–0.540 s at 575. | 41.2–43.4 tok/s after 47; 39.1–41.0 after 575. |

The Strix ranges are the two run-order medians, five warm requests per arm per order. Both engines produced 128 output tokens per request. The client decode proxy uses 127 intervals from first to last streamed text, not total request time. Loom used FP16 KV and chat; hipEngine used FP32 KV and explicit IDs. They reproduced their own text, **not each other's**. A [later timing pass was discarded](../../worklog/entries/20261124T094336.032013Z-lhl-inkling2-mini-4da570.md) under host contention. Do not call the old Strix numbers today's speeds or use the W7900 improvement to update them. The [published Loom card][loom-card] uses depth-one drafting for its single-stream rows, not the local ordinary protocol.

## Source comparison: what to borrow and what not to copy

| Engine / pinned source | What the source actually does | Difference that matters here |
| --- | --- | --- |
| **[AMD Loom](https://github.com/ROCm/hrx-system)** · [generated attention program][loom-attn], [expert program][loom-moe] | For a prompt it stages new K/V, attends over old ring plus staged rows, then commits. Its single-row branch combines convolution, KV preparation and store. For grouped experts it launches routing, scale, bucketing, sorted projections and recombination on device. | Keep the ordering our block path already implements. Borrow *device-resident route records* and distinct prompt/token schedules, not another block-convolution port. Use the pinned generated program, not just the published card. |
| **[Gufo Inkling HIP extension][gufo-repo]** · [decode kernels][gufo-decode], [grouped matrix kernels][gufo-wmma], [weight loader][gufo-engine] | `k_multi_gemv` submits multiple matrix-vector jobs together; `k_post1` clips/normalizes and stores KV. `k_route_lists1` creates IDs, weights, counts and lists on device. Grouped gate/up reads an expert's pair list, unpacks weights in shared memory, accumulates in FP32 and writes the capped activation. `Engine::dt` repacks Q8_0/NVFP4 codes and scales into separate planes at load time. | Closest HIP implementation. Our grouping of Python-side jobs is not necessarily one kernel launch. Test launch fusion and tile occupancy separately. Gufo writes winners in selection order; our [reference router][router] accumulates in **expert-ID order**. Do not borrow its fixed dimensions, cache format or reduction order as a contract. Its [~650 prefill / 29–30 plain decode tok/s][gufo-results] are author results on Strix Halo, mainline `UD-Q4_K_M`. |
| **[Mainline llama.cpp](https://github.com/ggml-org/llama.cpp)** · [loop graph][llama-loop], [layer and head graph][llama-layer], [routed matmul dispatch][llama-dispatch] | `post_arch_tensors` shares physical tensors after scales are populated; `build_layer(il, ip, …)` keeps cache slots distinct and passes expert scales to `build_moe_ffn`. The switched matrix-vector, few-row and matrix-matrix paths consume expert IDs directly where supported; the generic fallback synchronizes. The HIP build [uses those same ggml sources][llama-hip]. | Correctness oracle for `Mainline_Support`, not for the legacy file. Compare a captured visit's `Qcur`, `Kcur`, `ffn_inp`, `l_out` and unpadded logits. Match type/shape before attributing a fast run to a particular kernel. The [W7900 timing run][llama-local] used `e50c254a` (23.33 tok/s after 512 tokens, FP32 KV, mainline `UD-Q4_K_M`); it too [shares layers after scale loading](https://github.com/ggml-org/llama.cpp/blob/e50c254a22c590c9d2a60db86690d48d42497e2b/src/models/inkling2-loop.cpp#L65). Keep that timing separate from the oracle run at `690b87d8`. |
| **[vengeful180's llama.cpp Vulkan fork][vk-notes]** · [depth mix][vk-depth], [NVFP4 matvec][vk-mv], [relative mask][vk-mask] | One workgroup mixes a token's residual history. The matvec decodes packed nibbles and can multiply the reduced output by `scale[expert_id]` before storing. `REL_MASK` writes the mask plus bias minus sink directly as FP16; it still writes a mask. | Compare our residual reads, scale gathers and scalar bias traffic, not Vulkan submission knobs. We already have packed projection and a table-based `fast` **block** path. The author's 2,075 prefill (512 tokens) / 108.9 plain decode (64 outputs) used RX 7900 XT, **root** `UD-Q4_K_M`, not our mainline file. |
| **[Maya `maya-ik2`][maya-guide]** · [`fast_token`][maya-token], [`experts_mma_kernel`][maya-mma], [device grouping][maya-group] | `fast_token` captures a resident token's launches, supplies changing position/inputs, then reads back an argmax, candidates or full logits depending on output mode. Packed NVFP4 weights are decoded for matrix instructions; gate/up uses two FP16 parts of an FP32 input with FP32 accumulation. Multi-token work groups routes by expert. | Borrow the order: resident eager first, graph second, narrow readback only when sampler semantics permit it. Test two-part inputs as a separate arithmetic candidate. CUDA `mma.sync` is not an AMD kernel. The author's RTX 3090 `UD-Q4_K_XL` row is 2,742 prefill / 124.2 plain decode **at 4,096 input**, not the short-prompt decode rate. |

These are design references, not requirements to reproduce any other engine's internal representation. Check licenses and record upstream file and commit before deriving code. In hipEngine use the `(backend, layer, quant, variant)` registry, raw device pointers, strict unfused primitives and the existing request owners. Never dispatch by model hash, file name, benchmark prompt or known answer.

## Work queue, ordered by expected impact

Priorities are hypotheses tied to measured work, not promised speedups. **Start with step 1's CPU comparison.** Step 0 can run in parallel without the GPU; finish it before claiming a new public timing. Step 3 can use captured routes independently of whole-block integration. Steps 4 and 7 depend on a stable ordinary token path. Take one independently testable change per commit. Re-rank the remaining work if a new public trace changes which phase dominates.

### 0. Make the baseline and comparison reproducible

- **Deliverable:** check in a small ordinary-pairs driver and CPU tests for its timing/aggregation. The [paired HTTP worklog][pairs] describes an off-tree driver; a fresh checkout must not depend on it or its logs. Reuse [the decode marker script][trace] for attribution and [the known-answer check][tasks] for output sanity. Record the effective profile/manifest, actual variants and call counts, not just requested settings. The packed public route reports `execution_path=inkling2_gguf_packed_hip`; that distinguishes packed execution but does **not** identify a full resident block.
- For each request record input-token digest, generated IDs **when exposed**, first selected ID/time if available, first/last nonempty text (including reasoning), stream close, actual output count, finish reason, prefix-cache hits, draft state and request ID. A first text delta may represent more than one token. Report prompt-only throughput when a real prompt timer exists; otherwise label `input tokens / first-text latency` a proxy. Keep unprofiled wall measurements separate from marked kernel, copy and host timings.
- Freeze matched inputs, engine versions, quant, tokenizer, capacity, KV types, clock/power and sampling. Run one engine at a time. Check host memory, swap/I/O pressure **and** GPU ownership; no other GPU process is not sufficient. Preserve timeouts and missing reverse runs as inconclusive. If the host is occupied, finish fixtures and schema tests off GPU. Do not silence a timeout, change the output cap, or switch hosts to complete a pair.

### 1. Correct the block oracle, then finish device-resident prefill

**Why first:** the [resident attention run][attention-run] removed a measured part of request time; completing the traversal is the shortest path to testing the remaining host visits. The [pinned upload race][upload-fix] was real and fixed. The [previous public block attempt][block-reject] failed repeat/quality checks **before** that fix. Neither establishes the result after it.

1. **Make a CPU RED test for observation points.** The scratch sweep used `prefill_block(..., capture=...)` and subtracted `trace[k]` from `_resident_block(..., limit=k)[0]`. In [`_traverse`/`_forward`][traversal], `capture(k, h)` is *entry* to visit `k`, **after** a prelude restore. At 18 and 30 that is not the preceding visit's exit. At 48 the callback gets `_output_hidden(h, blocks)`, not raw `h`. The [reported spikes][sweep] land on precisely these three boundaries. The [CPU-only reproduction][capture-repro] uses `TinyWeights` from [the existing fixture][tiny]. Turn its assertions into a unit test in `tests/test_unit_inkling2_reference.py` before editing a kernel. It invalidates this subtraction, **not** the earlier failed public test or the possibility of a real device error.
2. Name and copy the same point on both arms: `(request, position, row, visit index, weight layer, state slot, phase)`. Use raw post-MLP hidden vs raw post-MLP hidden **before** the next restore; compare restored entry separately. At exit compare raw to raw, or apply the same final depth mix, norm, unpadded head and softcap to both. Snapshot before a mutable list grows or an arena is reused. Prove host-vs-host zero on the same schedule and inject a known wrong row to prove the checker detects it. Do not add fences to the timed path.
3. Re-run the isolated and **public** candidate on both layouts, same inputs in repeat and fresh sessions. Assert the *full-block* call count is positive for eligible blocks (not the sum of attention-half and full-block calls). Compare 32/64 rows, 63/64/65 tokens, 151, 257, 511, 575 and 1,031; include a multi-block prompt with final scalar tail, ring wrap, final logits and subsequent forced decode steps. Check the upload pool with overlapping payloads, late event completion, no-pinning fallback, growth and close. Preserve exact positions and independent visit histories.
4. If aligned outputs disagree, stop at the **first same-stage mismatch**. Capture inputs and output together: residual read → normalized Q/K/V or message blend → projected dynamic/static convolution → attention branch → residual add → expert inputs, outputs and post-convolution. Cross-check independently chosen routes *and* fixed-route outputs before attributing an error to routing. If only the old checker failed, retire it; do not change kernel math to reproduce it.
5. Only after the applicable gate, use the full traversal for eligible committed blocks. Keep scalar/tail, capture, provisional verifier and strict routes correct. `project_experts_mixed_rows(device=...)` currently requires a usable shared tile set: test zero reuse/small tails before handing it arbitrary blocks, provide the existing supported route **before any state mutation**, never replay a half-committed visit. Prove `LLM.generate()` and serving selected the intended path. Measure time through first token, per-visit downloads, bytes, waits and peak scratch versus attention-half `fast`.

### 2. Remove scalar decode round trips: router, experts, then the next visit

**Why:** the [W7900 suffix trace][waits] counted 2,088 stream waits across eight steps at both short and wrapped depth; the [host profile][host-profile] attributed more self time to synchronization than to any other single call. These are not 2,088 removable kernel durations. Start with `project_from_device` → `_select_routes` → `_plan_expert_chain`; the full-block prototype still downloads logits and constructs route metadata on the host. Even its `device=` expert arm builds `packed`, uploads a staging copy with the hidden region zeroed, then copies resident hidden into it. “Device output” does not mean “no transfer.”

- First separate immutable per-physical-layer gate/up and down scale tables from per-row IDs, mixture weights and tile records. Keep all per-request inputs in device buffers; have the consumer take pointers rather than upload an empty activation payload. Reuse the [fixed-route scale check][scale-replay] and the [guarded router experiment][router-trial], not a new standalone top-k demo. Count bytes and synchronizing reads per visit in addition to launches.
- Implement and test the [router contract][router]: sigmoid scores; optional correction bias **for ranking only**; lower ID breaks ties; sort winners by ID; normalize the **unbiased** scores with the model scale, then mix in order. Cover near-ties, identical logits, ignored padding, gate/up and down scales, with/without scales, and repeated physical layers. The [legacy layout check][coverage] covers 44 NVFP4 visits plus four Q8_0 visits; mainline also has Q5_K. Counts describe these artifacts, not admission rules.
- If a guarded path uses host resolution near a cutoff, a rejected arm must not publish speculative output or advance state owned by its fallback. Commit the selected expert convolution **once**; do not replay already committed attention or KV. Force accept, reject, alternating and mixed rows. Test device routing independently of forced-ID arithmetic; compare logits through subsequent steps, not just winner sets. The previous accepted arm changed logit bits; use the numerical gate, not a bitwise rejection or a claim of exact arithmetic.
- Extend residency **one boundary at a time** on the scalar route: residual/controller → grouped Q/K/V and gated attention → experts/post-convolution → next visit. Keep eager launches first. Test a token-only greedy readback after softcap and masking, including tie order/unpadded vocabulary; retain full logits wherever penalties, sampling, logprobs or verifier scoring require them. Gate on exact sampler, stop and streamed output contracts. Advance when counters fall and an unprofiled suffix or its isolated public phase improves in both orders.

### 3. Speed up grouped expert *prefill*, then widen prompt blocks

**Why:** [the pinned Strix trace][strix-trace] attributes 2,821 of 5,158 ms of summed kernel time to grouped NVFP4 at 575 tokens (not wall time). On W7900, [wider blocks alone][width] reduced re-reads but did **not** reduce NVFP4 kernel time. Do not start by copying Maya's chunk size.

1. Capture per-expert `(count, token row, rank)` histograms separately for gate/up and down on real short, long, code, Japanese and mixed prompts; include sparse, skewed and uneven tails. Reuse [`build_expert_route_tiles`](../../hipengine/loading/inkling2_device_experts.py) and the existing [NVFP4 kernels](../../hipengine/kernels/hip_gfx1100/quant/gguf_nvfp4_gemv.hip) as controls. Keep the current small-group path. Bucket populated pairs on device; compare *useful* rows per tile, not nominal matrix throughput.
2. Compare one change at a time: raw packed weights vs Gufo-style aligned code/scale planes; selected four-route FP32 vs a dense-enough AMD wave matrix multiply-accumulate (WMMA) group; gate/up vs down; one FP16 activation vs high plus residual FP16 parts with FP32 accumulation. Check input conversion, true outer scales, capped gate/up activation and route reduction order against CPU and strict. Include conversion, group construction, partially filled tiles and recombination in the timed interval; record cold repack time, peak and resident bytes. No permanently expanded full expert bank without a memory and latency comparison.
3. Two negative controls matter: [16-route FP16 tiles lost on Strix][wmma-loss] at the tested densities, and [grouped Q8 prefill exceeded the KL ceiling][q8-loss] with the same top token. Require a changed occupancy, layout or arithmetic hypothesis and replay the failing cell. Screen on both GPUs; re-profile the secondary Q5_K/Q8_0 banks and dense projections only if they become limiting.
4. After a projection win, vary **both** model block width and scheduler budget independently and together. Log actual dispatched widths. Test 64/128/256, tails and unrelated lengths; try larger blocks only if measured useful work warrants them. Measure first token, mixed-load decode starvation, memory and split-schedule quality. Do not generalize a session-only result to `serve`.

### 4. Fuse or replay the resident token, not a six-kernel island

After step 2, trace a warmed eager token. Fuse the largest still-dependent small operations the trace identifies (for example projection + convolution, or clip/norm + KV store). Keep the eager version for the control. Then, **as a separate A/B**, capture the long stable visit chain or token through the [existing HIP graph API](../../hipengine/core/hip.py). A [six-node expert graph][graph-loss] already lost on both GPUs; repeating it is not this experiment.

Keep inputs, request/slot mapping, positions, route records and sampler state mutable **data** at stable addresses; prewarm weights, kernels and scratch. No allocation, host route decisions, dangling staging buffers or borrowed output in capture. Key graphs by executable shape, storage and output requirements, not prompt identity. On a bucket miss execute eager for that step; test wrap, shape transitions, cancellation, slot reuse and teardown while another request survives. Report capture cost, memory, hits/misses, replays, host dispatch time and warm *wall* per token. Keep a fusion only if its own comparison passes; do not ship replay on node count alone.

### 5. Attack prepared-depth attention and KV traffic separately

The earlier [prepared Strix trace][depth-trace] showed the value pass dominating at 2,057; `fast` **already has** chunk-parallel split values and table-based resident block attention. Re-trace before selecting a candidate. Instrument [`Inkling2DeviceAttention`](../../hipengine/runtime/inkling2_device_attention.py): local vs global slots, bias bytes, K/V bytes, scratch, split score/value durations and host work. Compare a scalar table lookup/streaming softmax or grouped-query K/V reuse against today's split path, holding FP32 cache fixed. Exercise sink-only and masked scores, lag-table extent, multiple heads, wrap and partial tiles; stage new K/V before the old ring is overwritten and preserve `KVLiveSpans`.

Treat physical FP16 KV as a **separate storage experiment** against FP32, not a prerequisite for running or a fix for all output drift: [rounding K/V alone did not explain the Loom divergence][kv-trial]. Measure quality, capacity, bytes and prepared decode before considering FP8 or a different key/value quant. Run the allocation probe below first on the intended host, then one full unaligned prompt per required depth with finite logits, actual local/global live counts and cleanup. A W7900 allocation or 32K completion is not evidence for Strix.

### 6. Serve four ordinary streams

Extend the [existing resident cohort implementation][cohort], not a parallel server. Use the trace to choose between reducing row-serial prompt ingestion and filling grouped expert/attention/head work during decode. Preserve per-request KV, convolution, sampler and stop state. Test two, three and four rows with zero/partial/full expert overlap, unequal prompts, mixed finish times, refill, disconnect, deadline and slot reuse. Verify each row against its declared serial/control contract, then use a barrier to submit four actual HTTP streams. Report both steady aggregate during the common decode window and total output/cohort wall time including prefill, plus per-request first text and inter-token latency. [Prior pressured cohorts][cohort-pair] are not a parity result.

### 7. Test whether DSpark saves target work

Start with true draft-off versus explicit depth one on the **same** target. Keep `serial_target` as control. Before enabling packed verification, pin the known [mixed-language divergence][verify-fail] and compare successive verifier windows against ordinary logits on the *same history*, including the first token with taps. Test rejection at every offset, full rejection, partial acceptance, wrap, close and rollback of KV, convolution and draft features. Count proposed/accepted tokens, distinct experts, bytes and time in draft, verify and restore. Use [all four prompt categories](../../benchmarks/prompts/mtpbench-code-general-ja.jsonl) and disjoint category held-outs, then wider drafts only if verified tokens per unit wall time improve. Correct serial verification or a high acceptance fraction alone is not acceleration.

## Commands and decision rules

Run from the repository root. `MODEL` is the **local GGUF file**, `BACKEND` is `hip_gfx1151` on `strixhalo` or `hip_gfx1100` on `epyc`, and `OUT` is a new scratch directory for this run. For Loom use the [locked target, tokenizer, binary and fixed-backing command][loom-local]. For the mainline oracle use the [verified mainline file][llama-local] and llama.cpp `690b87d88151dbd775bb6eff5581663688785928`. Verify hashes for comparisons; do not use them to gate serving.

```bash
# CPU first; no model or GPU required.
.venv/bin/python -m pytest -q tests/test_unit_inkling2_reference.py tests/test_unit_inkling2_activation_transfer.py tests/test_unit_inkling2_gguf_runtime.py tests/test_unit_inkling2_expert_routes.py tests/test_unit_inkling2_gguf_resident.py
```

Only on the assigned, available GPU:

```bash
export PYTHONPATH=. OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
python3 scripts/check_lineage.py --kind kernel --diff stat
.venv/bin/python -m pytest -q tests/test_gpu_inkling2_residual.py tests/test_gpu_inkling2_resident_attention.py tests/test_gpu_inkling2_target_attention.py tests/test_gpu_inkling2_grouped_experts.py
.venv/bin/python scripts/inkling2_gguf_task_check.py "$MODEL" --backend "$BACKEND"
.venv/bin/python scripts/inkling2_gguf_task_check.py "$MODEL" --backend "$BACKEND" --reverse

# Profile decode only. Warm outside the profiler, including both shapes; never time JIT compilation.
hipcc --version > "$OUT/compiler.txt"
export HIPENGINE_COMPILER_VERSION_FILE="$OUT/compiler.txt" HIPENGINE_EXECUTION_PROFILE=fast
.venv/bin/python scripts/inkling2_gguf_decode_trace.py "$MODEL" --backend "$BACKEND" --short 47 --prepared 2057
HIPENGINE_REQUIRE_CACHED_BUILD=1 rocprofv3 --kernel-trace --hip-runtime-trace --memory-copy-trace --marker-trace --output-format csv --output-directory "$OUT/trace" -- .venv/bin/python scripts/inkling2_gguf_decode_trace.py "$MODEL" --backend "$BACKEND" --short 47 --prepared 2057 --trace

# Allocation plus a short prompt, NOT 32K completion or a prefill benchmark.
HIPENGINE_BACKEND="$BACKEND" .venv/bin/python scripts/gguf_capacity_probe.py --model "$MODEL" --max-sequence-length 33024 --prompt-length 65 --decode-tokens 4 --prefill-chunk-size 64 --json "$OUT/capacity.json"
```

Exercise the shipping surface separately. Start one server at a time; this smoke request tests reachability, not a benchmark. Then run the step 0 driver with frozen rendered token IDs, 128 outputs, cache reuse and drafting off. Repeat a candidate through the intended **default** surface before calling it shipped; an explicit `fast` result stays labeled `fast`.

```bash
.venv/bin/hipengine serve --model "$MODEL" --served-model-name inkling --backend "$BACKEND" --execution-profile fast --max-context-tokens 4096 --max-active-requests 4 --prefix-cache off --speculative-mtp-serving off --host 127.0.0.1 --port 8000
# In another terminal:
curl -fsS http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"inkling","prompt":"The capital of France is","max_tokens":8,"temperature":0,"stream":true}'
```

**Acceptance and handoff for each unit:**

1. Record base/candidate revisions, physical host and device, software stack, artifact/tokenizer hashes, effective profile and manifest, KV/sampler, prompt IDs, warmup, run order, commands, raw sample location and a correctness result. Record selected variant, invocation count, actual widths, fallback reason, allocations and releases. Keep cold load/repack/capture separate. Save compact results per [the evidence rules](../OPTIMIZATION.md); no model files, raw traces or temporary drivers as the only reproducible harness.
2. Exact state, request ownership and repeatability are mandatory. For a changed-arithmetic production candidate use [Execution Profiles §6](../EXECUTION-PROFILES.md#6-production-numerical-gate): full-vocabulary strict-teacher mean/p95/p99/max KL divergence, overall and scoped top-1, repeat/isolation, applicable higher-precision and task checks. For promotion, capture 500–1,000 positions on a shared strict token history; do not multiply the row count by repeats. Add an Inkling capture adapter for the [existing evaluator and packet format](../TESTING.md#run-an-evaluation-packet) instead of inventing new thresholds. Predeclare category/shape/transition scopes. Force expert IDs to isolate projections **and** independently re-route to test ranking. A short top-token smoke, the scratch block screen, or a good average hiding a failed row is not a production gate. Run [the deterministic bundle](../TESTING.md#2-cpu-deterministic-bundle) and the affected GPU fixtures, with availability guards for new GPU tests.
3. Time matched ordinary 47/128 and 575/128 first; then fresh prefill near 1K/2K/8K/32K, nonaligned wrap, prepared decode near 2K/32K, and four ordinary streams on the pinned Strix host. Use at least five warmed pairs **per cell in each engine order**. Match requires ≥97% of Loom throughput or ≤103% of its latency in both orders; a beat requires >3% improvement in both. If outputs, timing boundaries, KV, or host conditions differ, record which comparison still holds: different KV can support a complete-stack timing, not a matched-KV claim; different text does not establish output parity. Verify shared prompt IDs and find the first divergent token on a **shared history** before assigning blame. Keep a measured phase win distinct from a faster request; noise is inconclusive, not a reason to pick the best run.
4. End each unit with one decision: integrate, keep as a bounded experiment, or stop with the failing cell and a command that reopens it. Name the next owner/path, strict control, clearing test and next unblocked task in an immutable worklog entry. Do not publish estimated speedups, relax a test to pass a candidate, or add permanent experiment switches. Campaign closure needs actual public-route and same-host results, not all the code boxes checked.

[pairs]: ../../worklog/entries/20261122T053851.330818Z-lhl-inkling2-mini-e5e1c9.md
[attention-run]: ../../worklog/entries/20261124T093241.447865Z-lhl-inkling2-mini-4b3cf8.md
[upload-fix]: ../../worklog/entries/20261124T130650.833733Z-lhl-inkling2-mini-d00dbc.md
[block-reject]: ../../worklog/entries/20261124T111251.934665Z-lhl-inkling2-mini-275b7d.md
[sweep]: ../../worklog/entries/20261124T131703.985686Z-lhl-inkling2-mini-34e569.md
[traversal]: ../../hipengine/runtime/inkling2_mini_reference.py
[capture-repro]: ../../worklog/entries/20261124T132110.057455Z-lhl-inkling2-gguf-execution-plan-d1bc56.md#validation
[tiny]: ../../tests/test_unit_inkling2_reference.py
[router]: ../../hipengine/kernels/cpu_reference/inkling2_mini.py
[waits]: ../../worklog/entries/20261123T085415.962871Z-lhl-inkling2-mini-5fbf5b.md
[host-profile]: ../../worklog/entries/20261123T084317.753898Z-lhl-inkling2-mini-dc148d.md
[scale-replay]: ../../worklog/entries/20261123T110708.925852Z-lhl-inkling2-mini-877dd6.md
[router-trial]: ../../worklog/entries/20261123T112025.233711Z-lhl-inkling2-mini-48c45b.md
[coverage]: ../../worklog/entries/20261123T112532.970342Z-lhl-inkling2-mini-7d40f0.md
[strix-trace]: ../../worklog/entries/20261123T064320.809989Z-lhl-inkling2-mini-769df2.md
[width]: ../../worklog/entries/20261123T234541.751229Z-lhl-inkling2-mini-2f4ab7.md
[wmma-loss]: ../../worklog/entries/20261122T054650.102910Z-lhl-inkling2-mini-0d2355.md
[q8-loss]: ../../worklog/entries/20261122T055342.013646Z-lhl-inkling2-mini-54a375.md
[graph-loss]: ../../worklog/entries/20261123T024117.929489Z-lhl-inkling2-mini-d3a41e.md
[depth-trace]: ../../worklog/entries/20261122T160210.567291Z-lhl-inkling2-mini-46a155.md
[kv-trial]: ../../worklog/entries/20261122T040436.953829Z-lhl-inkling2-mini-15f810.md
[cohort]: ../../worklog/entries/20261123T063723.966009Z-lhl-inkling2-mini-2b2961.md
[cohort-pair]: ../../worklog/entries/20261123T040520.805091Z-lhl-inkling2-mini-01e931.md
[verify-fail]: ../../worklog/entries/20261123T075641.882243Z-lhl-inkling2-mini-39db2a.md
[trace]: ../../scripts/inkling2_gguf_decode_trace.py
[tasks]: ../../scripts/inkling2_gguf_task_check.py
[loom-local]: ../../worklog/entries/20261121T170419.285290Z-lhl-inkling2-mini-780776.md
[llama-local]: ../../worklog/entries/20261121T153001.919971Z-lhl-inkling2-mini-4c8219.md
[loom-card]: https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/blob/277f099b5003b725e099d48caa1a2c0db1f9b650/REPORT.md
[loom-attn]: https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/blob/277f099b5003b725e099d48caa1a2c0db1f9b650/program/llm.loom#L45896
[loom-moe]: https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/blob/277f099b5003b725e099d48caa1a2c0db1f9b650/program/llm.loom#L45983
[gufo-repo]: https://github.com/sebt3/inkling-2-mini-gufo-extension
[gufo-decode]: https://github.com/sebt3/inkling-2-mini-gufo-extension/blob/7d816730df7537f9b32c5e1d13d91bd13a7e697c/engine/decode.hpp
[gufo-wmma]: https://github.com/sebt3/inkling-2-mini-gufo-extension/blob/7d816730df7537f9b32c5e1d13d91bd13a7e697c/engine/wmma.hpp
[gufo-engine]: https://github.com/sebt3/inkling-2-mini-gufo-extension/blob/7d816730df7537f9b32c5e1d13d91bd13a7e697c/engine/engine.hpp
[gufo-results]: https://github.com/sebt3/inkling-2-mini-gufo-extension/blob/7d816730df7537f9b32c5e1d13d91bd13a7e697c/PROGRESS.md
[llama-loop]: https://github.com/ggml-org/llama.cpp/blob/690b87d88151dbd775bb6eff5581663688785928/src/models/inkling2-loop.cpp
[llama-layer]: https://github.com/ggml-org/llama.cpp/blob/690b87d88151dbd775bb6eff5581663688785928/src/models/inkling2.cpp#L273
[llama-dispatch]: https://github.com/ggml-org/llama.cpp/blob/690b87d88151dbd775bb6eff5581663688785928/ggml/src/ggml-cuda/ggml-cuda.cu#L2638
[llama-hip]: https://github.com/ggml-org/llama.cpp/blob/690b87d88151dbd775bb6eff5581663688785928/ggml/src/ggml-hip/CMakeLists.txt#L60
[vk-notes]: https://github.com/vengeful180-dot/llama.cpp/blob/99da9b14be5093199f0d19d4813bb3948fab8332/INKLING.md
[vk-depth]: https://github.com/vengeful180-dot/llama.cpp/blob/99da9b14be5093199f0d19d4813bb3948fab8332/ggml/src/ggml-vulkan/vulkan-shaders/depth_mix.comp
[vk-mv]: https://github.com/vengeful180-dot/llama.cpp/blob/99da9b14be5093199f0d19d4813bb3948fab8332/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_nvfp4.comp
[vk-mask]: https://github.com/vengeful180-dot/llama.cpp/blob/99da9b14be5093199f0d19d4813bb3948fab8332/ggml/src/ggml-vulkan/vulkan-shaders/rel_mask.comp
[maya-guide]: https://github.com/alesha-pro/maya-inkling-mini/blob/fcd523faca0c825131df45100b24eacb47dd2cf1/docs/INKLING2-MINI.md#numbers
[maya-token]: https://github.com/alesha-pro/maya-inkling-mini/blob/fcd523faca0c825131df45100b24eacb47dd2cf1/src/ik2/ik2_main.cu#L2604
[maya-mma]: https://github.com/alesha-pro/maya-inkling-mini/blob/fcd523faca0c825131df45100b24eacb47dd2cf1/src/ik2/ik2_kernels.cu#L2827
[maya-group]: https://github.com/alesha-pro/maya-inkling-mini/blob/fcd523faca0c825131df45100b24eacb47dd2cf1/src/ik2/ik2_kernels.cu#L5106
