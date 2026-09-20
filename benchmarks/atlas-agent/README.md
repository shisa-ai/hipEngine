# hipEngine vs atlas — 1:1 served comparison

A head-to-head measurement of hipEngine and [atlas](../../../atlas) on one host,
over the OpenAI-compatible HTTP surface, with both engines' own validated
configuration. Everything needed to reproduce it is in this directory.

This directory exists because the comparison is easy to get wrong. Two engines on
one box invite three specific mistakes — measuring with one engine's own harness,
comparing rates that were not taken on the same host, and presenting a
cross-quantification difference as an engine difference. The harness, the launch
scripts and the caveat list below are all built to prevent those.

## Status on this host (2026-09-20)

Recorded for posterity, because getting atlas to serve at all took several
non-obvious steps, and because its MTP path does not currently work here.

### What was needed to get atlas running

1. **A ROCm build.** Both `target/release/spark` and
   `/tmp/atlas-e2e/target/release/spark` were CUDA builds linking `libcuda.so`
   and could not start. `ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL='*'
   ./build-amd.sh` built the HIP backend in **2m02s** with a warm kernel cache
   (122 `.cu` files, four `strix-hip` targets).
2. **A ROCm SDK that is not in `/opt/rocm`.** This host's ROCm is a conda SDK at
   `_rocm_sdk_devel` (HIP 7.15.26333, clang 23.0.0git). atlas's `build-amd.sh`
   carries a local hipcc-discovery patch for exactly this case, and
   `serve-amd.sh` documents the same lookup order. Set `ATLAS_ROCM_HOME`.
3. **The HIP-to-CUDA shim on the loader path.** The ROCm build still references
   the CUDA symbol names, supplied by three HIP shims that `build.rs` writes to
   `target/release/build/atlas-kernels-*/out`. `serve-amd.sh` finds that directory
   and prepends it to `LD_LIBRARY_PATH`. **Invoking the binary directly cannot
   work**; the script must be used.
4. **`--max-batch-size` no larger than the KV pool fits.** atlas accepts
   `--max-seq-len 262144` and reports `max_model_len: 262144`, but it sizes the KV
   pool from `--gpu-memory-utilization` (105.6 GB of this host's 120 GB GTT at the
   default 88%) and then warns how many sequences fit at full length. At 256K that
   is 3. `serve-atlas.sh` therefore defaults `MAX_BATCH=3`.

### What does not work

**atlas's MTP path fails on this host, at every context length tried.** With
`NUM_DRAFTS=4` the server starts, logs `Speculative decoding: ENABLED (4
drafts/step)` and `MTP gate: throughput-arbitrated (K=4)`, answers `/v1/models`
correctly, and then fails every real request:

```
ERROR verify_dflash_step: decode_verify_dflash: SSM MTP intermediate buffers not allocated (need K-1 ...)
ERROR sequence: free_sequence: gpu.synchronize after zero_slot(0): cuStreamSynchronize ...
ERROR phase_start_prefills: Prefill start error: cuMemsetD32Async failed: status 901
```

The visible symptom is a request that returns one short burst of content and then
every subsequent request completing in ~1 ms with **zero content tokens** -- which,
if it were averaged naively, would report a meaningless ~260 tok/s. It reproduces
at `MAX_SEQ_LEN` 262144 and 32768 and at `MAX_BATCH` 8 and 3, so it is neither a
context-length nor a batch-size limit.

**With `NUM_DRAFTS=0` the same build is healthy**: four requests at 32K produced
29/28/29/28 content deltas, ~700 ms TTFT, ~2.2 s decode, zero errors, and a median
of 14.53 tok/s. So atlas is usable here as an **autoregressive** yardstick, and it
is its speculative path specifically that is broken.

A likely cause is the ROCm version: atlas's published Strix Halo numbers were
taken on **ROCm 7.13** and this host has **7.15**. Chasing that is atlas's work,
not hipEngine's -- this directory exists to measure hipEngine, and a yardstick
that needs its own debugging is not worth more of that budget than the note here.

### Consequence for the comparison

A best-vs-best **MTP** comparison cannot be made against atlas on this host,
because one side's MTP does not run. The honest options, in order of value:

1. **AR vs AR at 256K** (both engines, `NUM_DRAFTS=0`). Clean and fully matched on
   every axis except quant, and directly relevant to what a long-context agent
   workload gets today.
2. **hipEngine MTP vs atlas AR**, clearly labelled as such. Not apples-to-apples
   and not a speculation comparison.
3. **A different MTP-capable yardstick** -- `llama.cpp` or a `strix-llama.cpp`
   build, both of which have speculative decoding. `scripts/mtp-bench.py` is
   already described as a llama.cpp-compatible MTP prompt-suite benchmark, so the
   protocol side is largely built.

Do not report the ~260 tok/s figure above. It is an artifact of dividing by the
near-zero elapsed time of empty responses.

## What is compared

| axis | hipEngine | atlas | matched |
| --- | --- | --- | --- |
| Model | Qwen3.8-27B | Qwen3.8-27B | yes |
| Host / GPU | Strix Halo, `gfx1151`, Radeon 8060S, Ryzen AI Max+ 395 | same host | yes |
| Protocol | OpenAI `/v1/chat/completions`, streaming, greedy | same | yes |
| Client | `http_1to1_bench.py` | `http_1to1_bench.py` | yes |
| KV dtype | BF16 | BF16 | yes |
| Context | `--max-context-tokens 262144` | `--max-seq-len 262144` | yes |
| Prefix caching | radix, on | radix, on | yes |
| **Quant** | **Q4_K_M GGUF** | **NVFP4 (W4A8 DP4A)** | **no** |
| **Speculation depth** | **MTP K=3** | **MTP K=4** | **no** |

The two `no` rows are the reason every number this directory produces must carry
its configuration with it.

**Why the quant cannot be matched.** Atlas's kernels are organised as
`kernels/<platform>/<model>/<quant>/` and every entry is `nvfp4`; a search for
k-quant kernel directories across its tree returns nothing, so atlas cannot load a
Q4_K_M GGUF. hipEngine's GGUF reader understands the NVFP4 *layout*
(`hipengine/quant/gguf.py`) but has no NVFP4 execution kernels. There is therefore
no quant both engines can run. Each side runs its own shipped, validated quant.

**Why the depth differs.** K=4 is atlas's headline configuration. K=3 is the
deepest candidate budget hipEngine's serving evidence qualifies
(`docs/EXECUTION-PROFILES.md`; deeper requests are refused with the typed
`candidate_budget_not_qualified`), so K=3 is hipEngine's best.

## Long context and what hipEngine actually does there

The dense MTP adapter is bounded only by the target's own `max_sequence_length`,
which is allocated capacity rather than an evidence window. A row whose prompt
leaves no room for even one generated token is set to `candidate_budget = 0`
with `target_context_k0` and decodes autoregressively; prompt length is
otherwise not an admission axis.

Measured on this host (`gfx1151`, Qwen3.8-27B `Q4_K_M`, BF16 KV, candidate
budget 3), greedy requests are served through `speculative_mtp` at 128, 512,
600, 1,024, 1,025, 2,048, 4,096 and 8,192 prompt tokens via `/v1/completions`,
and at 917, 1,738, 5,674 and 11,291 prompt tokens via `/v1/chat/completions`.
At agentic context lengths the hipEngine arm of this comparison is therefore
speculative.

The axis that does still fall back to autoregressive decoding is sampling, not
context: a `temperature > 0` request is refused with
`automatic_mtp_scope_not_promoted`, and an `ignore_eos` request with
`sampling_mode_not_qualified`. This comparison sends greedy requests, so neither
applies to it.

## Prerequisites

1. **ROCm.** On this host it is a conda ROCm SDK, not `/opt/rocm`:
   `~/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel`
   (HIP 7.15). Override with `ATLAS_ROCM_HOME` / the hipEngine env wrapper.
2. **hipEngine env wrapper.** `ENVWRAP` (default `/tmp/t17-env.sh`) pins
   `LD_LIBRARY_PATH`, `HIP_PATH`, `HIPENGINE_HIP_ARCH=gfx1151` and the Python
   interpreter. Recreate it from that file if it is missing.
3. **atlas checkout and a ROCm build.** `ATLAS_ROOT` (default `~/atlas`).
   The build is required: a CUDA build links `libcuda.so` and cannot run here.
   ```bash
   cd ~/atlas
   ATLAS_ROCM_HOME=<rocm root> ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL='*' ./build-amd.sh
   ```
   This compiles atlas's `.cu` files to HSACO with `hipcc` (122 files across four
   `strix-hip` targets) and took 2m02s here with a warm kernel cache. atlas is a
   read-only peer: this repository never edits it, and the build writes only to
   its own `target/`.
4. **Model weights.**
   - hipEngine: `~/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`
   - atlas: `models--nvidia--Qwen3.8-27B-NVFP4` snapshot (~21 GB). `serve-atlas.sh`
     passes the local snapshot path, so a run never depends on the network.
5. **Memory.** 256K context needs about 16 GB of BF16 KV: 16 full-attention layers
   (of 64; the other 48 are linear/GDN with constant-size state) × 4 KV heads ×
   256 head_dim × 2 (K and V) × 2 bytes = 64 KB/token. With ~17–21 GB of weights
   both engines fit comfortably in this host's 120 GB GTT.
   `max_position_embeddings` is 262144, so 256K needs no rope scaling.

## Reproducing

```bash
# Both engines, all three arms, default 256K context.
benchmarks/atlas-agent/run.sh

# One engine only, one arm, explicit knobs.
ENGINES=hipengine ARMS=single OUTPUT_LEN=128 PROMPT_CATEGORY=code \
  benchmarks/atlas-agent/run.sh
```

`run.sh` writes a timestamped run directory (default
`/tmp/atlas-agent/run-<UTC>`) containing, per engine, the server log, the
capability probe, one JSON per arm, and the assembled `artifact.json`; plus
`environment.txt` (host, repo head, atlas head, GPU, all knobs) and
`orchestration.log`.

### Knobs

| variable | default | meaning |
| --- | --- | --- |
| `OUTPUT_LEN` | 128 | completion tokens per request, fixed |
| `CONCURRENCY` | 4 | top of the concurrency ladder |
| `TURNS` | 3 | multi-turn arm turn count |
| `REPEATS` | 2 | repeats per prompt in the single arm |
| `PROMPT_FILE` / `PROMPT_CATEGORY` / `PROMPT_LIMIT` | `mtpbench-code-general-ja.jsonl` / `code` / 4 | prompt selection |
| `MAX_CONTEXT` | 262144 | context length, both engines |
| `ENGINES` / `ARMS` | `hipengine atlas` / `single multi conc` | what to run |

## Arms and metrics

All requests are greedy (`temperature=0`) and streamed. Timing is measured from
the SSE stream alone — no engine-specific diagnostics field is read — so the same
script runs unchanged against both servers.

| arm | what it exercises | headline metric |
| --- | --- | --- |
| `single` | one request at a time, no prefix reuse | TTFT, ITL, decode tok/s per request |
| `multi` | each turn re-sends the transcript, so turns 2+ share a long prefix | later-turn TTFT and decode tok/s against turn 0 |
| `conc` | concurrency ladder 1..N | aggregate tok/s, per-request decode tok/s |

**The completion length is fixed and early stopping is not expected**, so both
engines emit exactly `OUTPUT_LEN` tokens and the decode rate is
`(N-1) / (end_to_end - ttft)`. This removes tokenizer and SSE-chunking
differences from the rate: an engine that batches several tokens into one stream
chunk would otherwise have its rate overstated by a delta count. The harness
records the delta count *and* any `usage` block, and reports how many requests
actually reached the target, so a run that stopped early is visible rather than
silently averaged.

## Reading the result

- Report the median, and the per-request decode rate alongside any aggregate.
- Always state the quant, the speculation depth, the context length, and the fact
  that both sides were measured on this host in this run.
- The `multi` arm is the interesting one for prefix caching, but a turn-2 gain is
  only a prefix-cache gain if the reused prefix was actually long enough to be
  reusable — check the transcript length the harness reports.
- Nothing here is a same-quant comparison and nothing here may be placed beside
  either project's published numbers as though it were.

## Files

| file | role |
| --- | --- |
| `http_1to1_bench.py` | the engine-agnostic OpenAI streaming client; the only measurement path |
| `serve-hipengine.sh` | hipEngine server launch (qualified production cell) |
| `serve-atlas.sh` | atlas server launch, including the HIP-shim and ROCm environment |
| `run.sh` | orchestration: one engine at a time, all arms, then analysis |
| `analyze.py` | assembles the comparison artifact and its caveat list |
