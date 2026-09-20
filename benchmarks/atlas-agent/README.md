# hipEngine vs atlas — 1:1 served comparison

A head-to-head measurement of hipEngine and [atlas](../../../atlas) on one host,
over the OpenAI-compatible HTTP surface, with both engines' own validated
configuration. Everything needed to reproduce it is in this directory.

This directory exists because the comparison is easy to get wrong. Two engines on
one box invite three specific mistakes — measuring with one engine's own harness,
comparing rates that were not taken on the same host, and presenting a
cross-quantification difference as an engine difference. The harness, the launch
scripts and the caveat list below are all built to prevent those.

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

hipEngine's dense MTP adapter admits only inside a **1,023-token window**
(`hipengine/generation/qwen35_gguf_mtp2.py`, `_MTP2_QUALIFIED_CONTEXT_WINDOW`).
Above it a row is set to `candidate_budget = 0` with
`target_context_k0` and decodes autoregressively.

At agentic context lengths this means the hipEngine arm of this comparison is
**autoregressive**, and that is the qualified behaviour, not a failure. Raising
the window with `HIPENGINE_MTP2_MAX_CONTEXT_TOKENS` is explicitly *not* a
promotion: raising it alone measures **0.57×**, because the target and draft
graphs decline into their eager paths per cycle (`docs/REFACTOR.md`,
"Long-context MTP window override"). `FORCE_LONG_MTP=1` runs that unqualified
diagnostic arm and labels it in the output.

## Prerequisites

1. **ROCm.** On this host it is a conda ROCm SDK, not `/opt/rocm`:
   `/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel`
   (HIP 7.15). Override with `ATLAS_ROCM_HOME` / the hipEngine env wrapper.
2. **hipEngine env wrapper.** `ENVWRAP` (default `/tmp/t17-env.sh`) pins
   `LD_LIBRARY_PATH`, `HIP_PATH`, `HIPENGINE_HIP_ARCH=gfx1151` and the Python
   interpreter. Recreate it from that file if it is missing.
3. **atlas checkout and a ROCm build.** `ATLAS_ROOT` (default `/home/lhl/atlas`).
   The build is required: a CUDA build links `libcuda.so` and cannot run here.
   ```bash
   cd /home/lhl/atlas
   ATLAS_ROCM_HOME=<rocm root> ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL='*' ./build-amd.sh
   ```
   This compiles atlas's `.cu` files to HSACO with `hipcc` (122 files across four
   `strix-hip` targets) and took 2m02s here with a warm kernel cache. atlas is a
   read-only peer: this repository never edits it, and the build writes only to
   its own `target/`.
4. **Model weights.**
   - hipEngine: `/home/lhl/models/gguf/Qwen3.8-27B-Q4_K_M.gguf`
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

# The unqualified long-context-MTP diagnostic arm.
FORCE_LONG_MTP=1 ENGINES=hipengine ARMS=single benchmarks/atlas-agent/run.sh
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
| `FORCE_LONG_MTP` | 0 | raise hipEngine's MTP window (unqualified) |

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
