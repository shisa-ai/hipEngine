# hipEngine Topline Benchmarks

Last reviewed: **2026-07-11**

Latest measured hipEngine revision in this scoreboard:
`0c1845170955af48fd52413228d1699dcf72364c`

This file is the source of truth for repository-level performance tables. It
records which snapshots are eligible for use, the exact protocol behind each
table, the measured source revision and build environment, and the command used
to refresh it. [`README.md`](../README.md) contains copies of the marked export
blocks below; update them with:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
```

Machine-readable evidence is under [`benchmarks/results/`](results/). Promotion
requirements are defined in [`docs/BENCHMARK.md`](../docs/BENCHMARK.md).
Reverse-chronological changes are in [`benchmarks/CHANGELOG.md`](CHANGELOG.md).
The previous experiment notebook is preserved in
[`benchmarks/HISTORY.md`](HISTORY.md).

## Status Rules

| Status | Meaning | May appear as a repository topline? |
| --- | --- | --- |
| **Retained** | The artifact passes the protocol's correctness, provenance, and performance gates. | Yes, for the named protocol only. |
| **Diagnostic** | The run is useful but has a known comparability, correctness, repetition, or provenance limitation. | Only when the limitation is printed next to the table. Do not describe it as retained or fastest. |
| **Stale** | A measured path, dependency, or required evidence contract changed after the run. | No. It may remain as the last dated snapshot while a refresh is pending. |
| **Blocked** | No row satisfies the protocol. | No numeric topline. Record the blocker and the next command. |

`Latest` means the newest artifact for one exact protocol tuple. A newer
diagnostic does not replace a retained row. A row is identified by:

```text
platform + GPU + model fingerprint + quant + KV type + backend +
workload + concurrency + sampling/speculative policy + timing scope
```

Documentation-only commits do not make a row stale. Changes to a measured
runtime path, model, quant, KV policy, compiler/runtime, benchmark timing scope,
correctness gate, or comparison engine do.

New server, retained PARO, GGUF, and micro artifacts must embed a valid
`hipengine_artifact_provenance` v1 block. The canonical schema is
[`schemas/artifact-provenance.schema.json`](schemas/artifact-provenance.schema.json).
For retained model-performance rows, the resolved backend must be concrete,
the selected target/device must be recorded, the model fingerprint must refer
to existing content, and staged/unstaged/untracked dirtiness must all be false.
Legacy provenance fields remain useful diagnostics but do not satisfy this
contract for a new row.

New non-streaming hipEngine server rows also require a complete
`hipengine.generation_shape` v1 rollup. Route caps retain their
`queue_requests` scope; queue request/prompt counts, actual backend calls and
widths, and verifier rows remain separate and are deduplicated by queue-group
ID. Client concurrency is never substituted for backend or verifier width.

## Platform Index

| Platform | Benchmark family | Run date | Measured revision / build | Evidence status | Root README | Refresh condition |
| --- | --- | --- | --- | --- | --- | --- |
| Radeon Pro W7900, gfx1100 | PARO BF16/INT8 KV context capacity | 2026-05-19 | hipEngine `ae229513`; compiler version not retained; artifact tree changed only `README.md` | **Stale diagnostic**: full commands and memory scopes are retained, but the build environment is incomplete and the quality gate uses a Qwen3.5 fixture for a Qwen3.6 capacity run | Dated diagnostic | Rerun BF16 and INT8 capacity plus a Qwen3.6 long-rollout quality gate at one clean revision |
| Radeon Pro W7900, gfx1100 | Qwen3.6 35B model sweep | 2026-07-07 | hipEngine `b4edca09`; TheRock HIP `7.13.26162-1140233ffe`; llama.cpp `263cc04a5` build 9600 | **Stale diagnostic**: top-level artifact has `performance_claim=false`; GGUF emits repeated token `9707` and is not correctness-certified | Dated diagnostic | Rerun after GGUF target/state correctness is restored, then pass the normal repeated-run gate on a clean revision |
| Radeon Pro W7900, gfx1100 | PARO/llama.cpp/vLLM concurrency | 2026-07-07 | hipEngine `b4edca09`; same TheRock stack; vLLM `0.22.1rc1.dev499+g470229c37.d20260613` | **Stale diagnostic**: cross-quant and mixed timing scopes; source artifacts set `performance_claim=false`; measured PARO code predates the July concurrency changes | Dated diagnostic | Rerun one timing scope with exact generated-token accounting across all engines |
| Radeon Pro W7900, gfx1100 | Dense 27B DFlash | 2026-06-11 | hipEngine `9faa731c`; ROCm 7.2; artifact records a dirty tree | **Retained under the recorded DFlash gate**, with legacy dirty-source provenance | Yes, qualified | Refresh on a clean tree before changing the public claim |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | Qwen3.6 35B model sweep | 2026-06-15 | hipEngine `64b86b9a`; TheRock HIP `7.13.60980-c76140fa27`; llama.cpp `6e9007ae6` build 9641 | **Stale diagnostic**: one measured run, no measured warmup, and commit/environment live only in WORKLOG rather than the summary artifact | Dated diagnostic | Add a committed gfx1151 refresh runner, emit full provenance in the artifact, and rerun the repeated protocol |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO direct c1-c8 shape matrix | 2026-07-10 | timing rows: tracked files matched hipEngine `4175dabf` and `02aec604`, with unrelated untracked files present; true-c1 shrink gate: `0c184517`, `hipengine_dirty=false`; TheRock HIP `7.13.60980-c76140fa27`; detected and target arch gfx1151 | **Diagnostic**: c2-c8 timing rows used a batch-shaped width-1 oracle and cannot select routing. At `0c184517`, serial c8-to-c1 passes all rows against independent c1; native c8 fails every row at generated token index 2. Production uses width-1 sessions. | Dated diagnostic | Localize the native c8 divergence, then rerun c1-c8, sparse, ragged, and shrinking gates against independent single-request prefill/decode before collecting retained timings |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | PARO/llama.cpp concurrency | 2026-06-15 | measured hipEngine revision not recorded in summary; gfx1151 forced through `HIPENGINE_HIP_ARCH` | **Stale diagnostic**: `performance_claim=false`, mixed quant, and incomplete backend provenance | Dated diagnostic | Rerun c=1..8 plus shrinking batches at one clean revision with detected arch and all-choice token counts |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF MTP exact, fixed 10-cycle suite | 2026-07-02 | hipEngine `44c4d3d4`; GGUF Q4_K_M | **Retained** for fixed-cycle exact/default semantics | Yes | Rerun when the exact MTP route or verifier math changes |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | GGUF MTP `llama-compat`, natural24 direct | 2026-07-03 | hipEngine `ca571bf6`; GGUF Q4_K_M | **Retained for the compatibility contract**: direct-commit/dp4a semantics are not serial-prefix-equivalent | Yes, qualified | Rerun when the compatibility route, budget, or output horizon changes |
| Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 | HIP versus Vulkan timing-contract v2 micro matrix | 2026-07-10 | clean detached hipEngine `ca241dae`; TheRock ROCm `7.13.0a20260411`; RADV/Mesa `26.1.2` | **Retained**, 22/22 comparisons pass provenance, correctness, and exact-matrix gates | Linked, not copied here | Rerun the bounded matrix after a timed kernel/harness change |
| Radeon Pro W7900, gfx1100 | HIP versus Vulkan timing-contract v2 micro matrix | Not run | None | **Blocked** | No | Run the same bounded v2 matrix used on gfx1151 |

## Public Snapshot Tables

These marked blocks are the public performance and capacity tables managed by
this scoreboard. The sync script compares their contents byte-for-byte. Status
and protocol text is maintained in both files because relative links and
section context differ.

### W7900 PARO context capacity, 2026-05-19

**Status: stale diagnostic.** The artifact records hipEngine `ae229513`, exact
commands, immutable model snapshots, tracked allocator memory, sampled HIP VRAM,
retained KV bytes, and no BF16 shadow. It does not record the compiler version,
and its correctness gate uses a deterministic Qwen3.5 fixture rather than a
Qwen3.6 long-rollout evaluation.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Model | Context | KV cache | Sampled HIP peak | Allocator peak | Retained KV | Prefill | Decode |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B PARO | 128K | BF16 | 21.04 GiB | 21.88 GiB | 2.69 GB | 1091.9 tok/s | 62.2 tok/s |
| Qwen3.6 35B-A3B PARO | 128K | INT8 per token/head | 19.80 GiB | 20.89 GiB | 1.36 GB | 1076.5 tok/s | 60.0 tok/s |
| Qwen3.6 35B-A3B PARO | 256K | INT8 per token/head | 21.96 GiB | 23.71 GiB | 2.71 GB | 670.2 tok/s | 40.3 tok/s |
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

Run record:

| Field | Value |
| --- | --- |
| GPU/backend | AMD Radeon Pro W7900, gfx1100, `hip_gfx1100` |
| Source | `ae22951377865f0db65b57c4641dc82bdf4db3f9`; artifact write had only `README.md` modified |
| Model | Qwen3.6 packed PARO snapshot `501ef8635e5cfb5a7497d232358ca8d1afc0c66e`; W4 PARO; 40 layers |
| Prompt/decode | Repeated token `9707`; 128 or 256K prompt; 128 decode tokens; 4 warmup decode tokens |
| Prefill | AOTriton threshold 512; chunks `1024/1024/3072/1024/1024` for linear/MoE/full-attention query/post/RoPE |
| KV | BF16 or INT8 per token/head with FP16 scales; fixed paged KV; no persistent BF16 shadow |
| Memory | Sampled HIP whole-device peak and hipEngine tracked allocator peak are different scopes; retained KV is decimal GB as reported by the artifact |
| Artifact | [`2026-05-19...memory-diagnostic.json`](results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json) |

The artifact embeds all three original commands. The 256K INT8 command was:

```bash
python3 scripts/qwen35_paro_bench.py \
  --model /models/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-full4096-e5-packed/snapshots/501ef8635e5cfb5a7497d232358ca8d1afc0c66e \
  --prompt-length 262144 --token-id 9707 \
  --decode-tokens 128 --warmup-decode-tokens 4 --max-layers 40 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --attn-aotriton-min-tokens 512 \
  --prefill-linear-chunk-size 1024 --prefill-moe-chunk-size 1024 \
  --prefill-full-attn-query-chunk-size 3072 \
  --prefill-full-attn-post-chunk-size 1024 \
  --prefill-full-attn-rope-chunk-size 1024 \
  --kv-storage int8_per_token_head \
  --json benchmarks/results/<date>-w7900-paro-256k-int8-kv.json
```

No artifact in the repository supports the former llama.cpp Q8_0 memory values
printed in the root README. Those numeric tables were removed. A future
llama.cpp memory row needs the model fingerprint, llama.cpp commit/build, GPU,
exact command, whole-card sampling method, and compact artifact.

### W7900 model sweep, 2026-07-07

**Status: stale diagnostic.** These values show the last complete same-host
sweep. They are not retained performance claims. hipEngine PARO is W4 PARO with
BF16 KV; the other three columns use Q4_K_M GGUF with BF16/f16 KV. The PARO
column is therefore not a same-quant comparison. The GGUF column uses the
correctness-first eager route whose generated output repeatedly selected token
`9707`; do not use it as a performance baseline until target/state correctness
passes.

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 2796.853 | 653.979 | 2502.690 | 2731.086 |
| 1K/128 | 2917.115 | 664.564 | 2423.728 | 2642.684 |
| 4K/128 | 2904.920 | 668.125 | 2294.828 | 2539.920 |
| 32K/128 | 2103.724 | 635.321 | 1680.677 | 1950.575 |
| 64K/128 | 1575.284 | 578.702 | 1319.054 | 1417.008 |
| 128K/128 | 1063.951 | 490.289 | 913.108 | 1075.764 |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 112.207 | 35.838 | 79.603 | 107.216 |
| 1K/128 | 102.458 | 35.610 | 79.498 | 106.851 |
| 4K/128 | 102.918 | 34.836 | 78.627 | 102.677 |
| 32K/128 | 91.745 | 35.162 | 72.228 | 91.480 |
| 64K/128 | 77.213 | 35.592 | 66.437 | 83.106 |
| 128K/128 | 59.999 | 35.426 | 57.712 | 70.479 |

#### Peak GiB

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 21.029 | 25.492 | 21.598 | 21.260 |
| 1K/128 | 21.241 | 25.492 | 21.610 | 21.220 |
| 4K/128 | 21.973 | 25.492 | 21.666 | 21.278 |
| 32K/128 | 22.082 | 25.492 | 22.208 | 21.855 |
| 64K/128 | 22.082 | 25.492 | 22.887 | 22.512 |
| 128K/128 | 22.124 | 25.492 | 24.080 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

Run record:

| Field | Value |
| --- | --- |
| GPU | AMD Radeon Pro W7900, gfx1100, `HIP_VISIBLE_DEVICES=0`, amdgpu `card1`, 44.984 GiB |
| hipEngine source | Clean detached worktree at `b4edca09f9553e5b3de755d2d0ada9e30f8c7d1e` (2026-07-07) |
| hipEngine build | TheRock HIP `7.13.26162-1140233ffe`; cached JIT required during measurement |
| hipEngine model/session | Qwen3.6-35B-A3B PARO snapshot `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`; Qwen3.6-35B-A3B MTP-bearing `UD-Q4_K_M.gguf`; one resident session allocated for `128K/128` |
| hipEngine repetitions | Repeated-token prompts; 2 discarded warmup runs and 5 measured runs per shape; medians reported; PARO uses 4 warmup decode tokens and graph replay; GGUF uses 1 warmup decode token and eager decode |
| llama.cpp | Commit `263cc04a5`, build 9600; `-ngl 99 -fa 1 -ctk f16 -ctv f16`; `ROCm0` or `Vulkan0`; one `llama-bench` repetition per split prefill/decode phase |
| Memory scope | hipEngine tracked allocator peak; llama.cpp whole-card amdgpu peak sampled every 10 ms. Compare memory columns only with this scope difference in view. |
| Refresh runner | [`scripts/run_w7900_readme_refresh.sh`](../scripts/run_w7900_readme_refresh.sh) |
| Summary | [`2026-07-07...summary.json`](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json) |

Component artifacts: [hipEngine PARO](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-paro-packed-5run.json),
[hipEngine GGUF](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-hip-q4km-f16kv.json), and
[llama.cpp Vulkan](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-vulkan-q4km-f16kv.json).

### gfx1151 model sweep, 2026-06-15

**Status: stale diagnostic.** This was one measured run per shape with no
measured warmup. The summary artifact omits the source revision and build
environment; the values below use the run record recovered from `WORKLOG.md`.
The next refresh must put those fields in the artifact.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 956.666 | 833.366 | 1016.696 | 1043.209 |
| 1K/128 | 1067.175 | 854.308 | 1069.681 | 1055.050 |
| 4K/128 | 1062.248 | 729.117 | 1021.186 | 1027.069 |
| 32K/128 | 822.255 | 619.570 | 742.869 | 809.619 |
| 64K/128 | 622.752 | 522.872 | 569.611 | 658.399 |
| 128K/128 | 425.727 | 384.011 | 384.959 | 473.651 |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 66.967 | 56.581 | 51.640 | 62.434 |
| 1K/128 | 61.768 | 52.832 | 51.446 | 61.572 |
| 4K/128 | 62.910 | 53.638 | 49.581 | 60.012 |
| 32K/128 | 50.368 | 44.383 | 43.628 | 50.911 |
| 64K/128 | 41.966 | 37.741 | 38.604 | 44.010 |
| 128K/128 | 30.286 | 28.043 | 31.598 | 34.714 |

#### hipEngine tracked allocator peak GiB

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M |
| --- | ---: | ---: |
| 512/128 | 20.924 | 26.264 |
| 1K/128 | 20.926 | 26.264 |
| 4K/128 | 20.937 | 26.264 |
| 32K/128 | 21.047 | 26.264 |
| 64K/128 | 21.047 | 26.264 |
| 128K/128 | 21.248 | 26.264 |
<!-- END TOPLINE:GFX1151_SWEEP -->

Run record:

| Field | Value |
| --- | --- |
| GPU | AMD Ryzen AI MAX+ 395 / Radeon 8060S, gfx1151 |
| hipEngine source | Clean detached worktree at `64b86b9a` (recorded in `WORKLOG.md`, absent from summary JSON) |
| hipEngine build | TheRock HIP `7.13.60980-c76140fa27`; Python 3.13.13; `HIPENGINE_HIP_ARCH=gfx1151` |
| llama.cpp | Commit `6e9007ae6`, build 9641; same MTP-bearing `UD-Q4_K_M.gguf` for HIP and Vulkan |
| Protocol | Shapes `512/128` through `128K/128`; one measured run, zero measured warmup runs |
| Memory scope | Only hipEngine tracked allocator peaks are reported. The APU sysfs interface exposed a 512 MiB aperture, so llama.cpp whole-card memory values are unusable. |
| Old runner | `/tmp/run_gfx1151_readme_udq4km.sh`; this temporary path is not reproducible and must not be used for another promotion |
| Summary | [`2026-06-15...summary.json`](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json) |

Component artifacts: [hipEngine PARO](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-paro-packed-1run.json),
[hipEngine GGUF](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-gguf-ud-q4km-1run.json),
[llama.cpp HIP](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-hip-ud-q4km-f16kv.json), and
[llama.cpp Vulkan](results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-vulkan-ud-q4km-f16kv.json).

### Speculative decode

The public table includes only contracts with a true same-protocol AR control.
The MTP exact and `llama-compat` rows have different state semantics and output
horizons; they must not be compared as two implementations of one contract.

<!-- BEGIN TOPLINE:SPECULATIVE -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
| GGUF MTP exact B5 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B Q4_K_M; 10-prompt category suite; fixed 10 cycles; exact/default state semantics | 61.98 vs 54.79 AR tok/s, **1.1312x** | Retained for this fixed-cycle contract |
| GGUF MTP `llama-compat` B2 | Radeon 8060S/gfx1151; same GGUF and prompt suite; natural24/cyclecap24; direct-commit/dp4a compatibility semantics | 71.52 vs 54.79 AR tok/s, **1.3055x** | Retained for this compatibility contract; accuracy-traded and not serial-prefix-equivalent |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [DFlash](results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
[exact MTP](results/2026-07-02-ar-mtp-default-parallelattn-full.json), and
[`llama-compat` MTP](results/2026-07-03-ar-mtp-llama-compat-directcommit-nocopy-natural24-cyclecap24-f32head-full.json).

### W7900 concurrency, 2026-07-07

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV, llama.cpp uses Vulkan
Q4_K_M/f16 KV, and vLLM uses GPTQ Int4. hipEngine and llama.cpp report backend
decode timing; vLLM reports OpenAI client wall throughput. The numbers expose
scaling behavior within each column, not an apples-to-apples engine ranking.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Concurrency | hipEngine PARO decode aggregate | hipEngine per sequence | llama.cpp Vulkan decode aggregate | llama.cpp per sequence | vLLM OpenAI wall aggregate | vLLM per sequence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 114.98 | 114.98 | 105.63 | 105.63 | 21.32 | 21.32 |
| 2 | 113.34 | 56.67 | 156.06 | 78.03 | 40.61 | 20.31 |
| 4 | 158.25 | 39.56 | 76.52 | 19.13 | 78.41 | 19.60 |
| 8 | 189.59 | 23.70 | 26.47 | 3.31 | 116.44 | 14.55 |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. hipEngine
`c=1` uses the single-sequence graph-replay benchmark and `c>1` uses the native
batch benchmark. llama.cpp restarts `llama-server` for each concurrency and
repetition with `-np c -c 1024*c`. vLLM uses the OpenAI completions endpoint.

Artifacts: [hipEngine](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-hipengine-concurrency-w7900/summary.json),
[llama.cpp Vulkan](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-llamacpp-vulkan-concurrency-w7900/summary.json),
[vLLM](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-vllm-localbuild-gptq-int4-concurrency-c1-c8-w7900.json), and
[combined summary](results/2026-07-07-w7900-gpu0-readme-refresh-20260707-104756-summary.json).

### gfx1151 PARO legacy shape-timing diagnostic, 2026-07-10

**Status: diagnostic, not routing-eligible.** The timing rows ran on the Radeon
8060S at hipEngine `4175dabf` for c1/c2/c4/c6/c8 and `02aec604` for c3/c5/c7.
Tracked files matched those commits; unrelated untracked files were present.
Their generated-token comparison used
`prefill_native_packed()` plus `step_batch_native(rows=1)` as the reference.
That is a batch-shaped path, not the independent single-request contract.

At `0c184517` with `hipengine_dirty=false`, the gate compares packed c8
execution with `prefill_native()+step()` over holey c8-to-c1 cancellation. The
serial c1 decode bridge passes all eight rows. Native decode fails all eight
rows at generated token index 2 while still at c8 (`17` instead of c1 token
`220`).
The schema-1 timing profile is rejected by production; greedy and sampled
multi-request generation use exact width-1 sessions until a schema-2 profile
passes packed-prefill, sparse-slot, and shrinking gates.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Width | Aggregate decode tok/s | Per sequence tok/s | Median step ms | Legacy diagnostic gate | Measured route |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 66.806 | 66.806 | 14.969 | Three-run reference; different prompt | Single-sequence graph replay; repeated token 9707 |
| 2 | 78.578 | 39.289 | 25.465 | Primitive pass; batch-shaped IDs 3/3 | Native full attention; selected-c1 MoE; batched LM-head |
| 3 | 87.488 | 29.163 | 34.310 | Primitive pass; batch-shaped IDs 3/3 | Selected-c1 MoE; small-batch shared expert; all-layer rowchunk2; serial LM-head |
| 4 | 99.616 | 24.904 | 40.158 | Primitive pass; batch-shaped IDs 3/3 | Selected-c1 MoE; all-layer rowchunk2; batched LM-head |
| 5 | 102.137 | 20.427 | 48.927 | Primitive pass; batch-shaped IDs 3/3 | Selected-c1 MoE; small-batch shared expert; all-layer rowchunk2; serial LM-head |
| 6 | 109.909 | 18.318 | 54.568 | Primitive pass; batch-shaped IDs 3/3 | Selected-c1 MoE; selected-layer rowchunk2; serial LM-head |
| 7 | 109.596 | 15.657 | 63.905 | Primitive pass; batch-shaped IDs 3/3 | Selected-c1 MoE; small-batch shared expert; all-layer rowchunk2; serial LM-head |
| 8 | 115.515 | 14.439 | 69.254 | Primitive pass; batch-shaped IDs 3/3; true-c1 red | Selected-c1 MoE; all-layer rowchunk2; batched LM-head |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Protocol: Qwen3.6-35B-A3B PARO snapshot
`437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`, W4 PARO, BF16 KV, 40 layers,
8 warmup decode steps, 128 measured decode steps, and greedy sampling. The c1
reference repeats token `9707` for 512 prompt positions and uses the
single-sequence graph-replay path. Widths c2-c8 use fixed prompt slices and ran
the seed-1234 KV/attention primitive gate. Their three-run generated-token
field compares against the legacy batch-shaped width-1 route. Every displayed
width has three measured repetitions; displayed values are medians and are not
production performance claims.

Run record:

| Field | Value |
| --- | --- |
| GPU/backend | AMD Ryzen AI MAX+ 395 / Radeon 8060S, detected gfx1151, target gfx1151 |
| Source/build | hipEngine `4175dabf145d2054ff751c56bf019febd03ced65` for c1/c2/c4/c6/c8 and `02aec6043c73171df5747034f01f2f46a22152b6` for c3/c5/c7; tracked files matched each commit; unrelated untracked files were present; Python 3.12.13; TheRock HIP `7.13.60980-c76140fa27` |
| Timing scope | Direct retained-batch backend decode wall; aggregate generated tokens divided by measured decode wall |
| Correctness | Primitive c2-c8 checks pass and are within `5.961e-8` of NumPy. The old 137-token c2-c8 comparison is invalid as a true-c1 gate. At `0c184517` with `hipengine_dirty=false`, shrinking serial passes 8/8 rows; native passes 0/8, first mismatch at c8 token index 2. |
| Partition profile | The schema-1 c2-c8 profile has `performance_claim=false`, uses the invalid batch-shaped oracle, and is rejected. Production route: `scheduler_true_c1_fallback`. Schema 2 requires independent-c1 packed-prefill, sparse-slot, and shrinking evidence. |
| Missing gates | First native hidden/state divergence, independent-c1 direct c2-c8 matrix, ragged contexts, rocprof trace, and retained scaling controls |
| Artifact | [`2026-07-10...current-diagnostic-summary.json`](results/2026-07-10-gfx1151-paro-cn-current-diagnostic-summary.json) |
| True-c1 gate | [`2026-07-10...true-c1-shrinking-gates.json`](results/2026-07-10-gfx1151-paro-true-c1-shrinking-gates.json) |

Reproduce one width by replacing `<rows>`, `<rep>`, and `<outdir>` in the
artifact's `commands` templates. Run `scripts/qwen35_batch_correctness.py`,
`scripts/qwen35_batch_retained_bench.py`, and then
`scripts/qwen35_batch_shrinking_correctness.py` in both serial and native
modes. Do not retain a timing unless the independent-c1 direct, sparse, ragged,
and shrinking gates pass.

### gfx1151 historical cross-engine concurrency, 2026-06-15

**Status: stale diagnostic.** hipEngine uses PARO W4/BF16 KV; llama.cpp uses
Vulkan Q4_K_S/f16 KV. vLLM did not produce a healthy server. The summary lacks
the measured hipEngine commit, and the then-used per-run device properties could
report gfx1100 even though the run forced `HIPENGINE_HIP_ARCH=gfx1151`.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
| Concurrency | hipEngine PARO decode aggregate | hipEngine per sequence | llama.cpp Vulkan decode aggregate | llama.cpp per sequence | vLLM OpenAI |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 66.62 | 66.62 | 62.16 | 62.16 | Blocked: server unhealthy |
| 2 | 69.54 | 34.77 | 94.12 | 47.06 | Blocked |
| 4 | 88.39 | 22.10 | 119.51 | 29.88 | Blocked |
| 8 | 100.68 | 12.59 | 119.94 | 14.99 | Blocked |
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Protocol: prompt 512, decode 128, 8 warmup decode tokens, median of 3. Primitive
c>1 attention/KV checks passed. The generated-token field used the older
batch-shaped reference and is not independent-c1 evidence. Profiler, scaling,
and provenance gates also did not pass.

Artifacts: [combined summary](results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[hipEngine](results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[llama.cpp Vulkan](results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[vLLM blocker](results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

## README Sweep Test Procedure

### W7900 model and concurrency refresh

Use a clean detached worktree. The wrapper fixes the GPU mapping, TheRock
environment, model paths, llama.cpp binaries, JIT cache policy, and output
layout.

```bash
RUN_TAG=$(date -u +%Y%m%d-%H%M%S)
WORKTREE="/tmp/hipengine-readme-w7900-${RUN_TAG}"
git worktree add --detach "$WORKTREE" HEAD

OUTDIR="$PWD/benchmarks/results" \
RUN_TAG="$RUN_TAG" \
REPO_ROOT="$WORKTREE" \
  "$WORKTREE/scripts/run_w7900_readme_refresh.sh" all
```

Subset commands:

```bash
scripts/run_w7900_readme_refresh.sh hipengine
scripts/run_w7900_readme_refresh.sh llamacpp
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

Required W7900 settings:

| Surface | Settings |
| --- | --- |
| Device mapping | `HIP_VISIBLE_DEVICES=0`; W7900 is amdgpu `card1`; llama.cpp uses `ROCm0` and `Vulkan0` after masking |
| hipEngine environment | `/home/lhl/mambaforge/envs/therock/bin/python3.12`; hermetic TheRock root from `python -m rocm_sdk path --root`; `HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| Model sweep | `512/128 1K/128 4K/128 32K/128 64K/128 128K/128`; 2 warmups; 5 measured; resident max-context session |
| PARO | snapshot `437eba06df05aad71a4dacdcaf3fff70ae1ee8a1`; `hip_gfx1100`; `packed_paro_w4`; BF16 KV; AOTriton threshold 512; graph replay decode |
| hipEngine GGUF | MTP-bearing Qwen3.6-35B-A3B `UD-Q4_K_M`; decode repack; WMMA bulk prefill; GEMV eager decode; BF16 KV |
| llama.cpp | Same GGUF; `-ngl 99 -fa 1 -ctk f16 -ctv f16`; split prefill/decode; one repetition per phase |
| Concurrency | prompt 512; decode 128; warmup 8; c=1,2,4,8; 3 repetitions; fixed token-id fixture |

Do not replace the table if the combined summary has
`performance_claim=false` unless the table remains labeled diagnostic. A
retained refresh also needs the correctness and repetition gates from
[`docs/BENCHMARK.md`](../docs/BENCHMARK.md).

### gfx1151 model and concurrency refresh

No committed gfx1151 equivalent of
[`run_w7900_readme_refresh.sh`](../scripts/run_w7900_readme_refresh.sh) exists.
The 2026-06-15 model sweep used `/tmp/run_gfx1151_readme_udq4km.sh`, so it cannot
serve as the next refresh command. Before updating the gfx1151 tables:

1. Add a committed wrapper that emits the same top-level source, environment,
   model, command, correctness, and artifact-index fields as the W7900 runner.
2. Detect and record `gfx1151` from the runtime/build output; do not fill the
   artifact from a CLI label alone.
3. Run 2 discarded warmups and 5 measured repetitions for the six model-sweep
   shapes.
4. Run PARO concurrency for c=1 through c=8, including odd widths and dynamic
   c=8 to c=1 shrinking, with exact all-choice generated-token counts.
5. Keep comparison engines in separate columns when quant or timing scope
   differs. Do not bold a cross-quant winner.

The 2026-07-10 timing diagnostic at `4175dabf`/`02aec604` satisfies
detected/target architecture and primitive c2-c8 checks. Its generated-token
oracle was batch-shaped. At `0c184517` with `hipengine_dirty=false`, the
shrinking gate rejects native c8 and accepts the serial bridge through c1.
Neither artifact is a full refresh:
the independent-c1 direct c2-c8 matrix, ragged contexts, profiler, and scaling
gates are missing. Use the compact artifacts for the lower-level commands, not
as a substitute for the five requirements above.

The lower-level hipEngine sweep command is:

```bash
PYTHONPATH=. \
HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/qwen35_readme_sweep.py \
  --engine paro \
  --model /home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1 \
  --backend hip_gfx1151 \
  --shared-expert-format packed_paro_w4 \
  --token-id 9707 \
  --workloads 512/128 1K/128 4K/128 32K/128 64K/128 128K/128 \
  --warmup-runs 2 --measured-runs 5 --warmup-decode-tokens 4 \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-gfx1151.txt \
  --require-cached-build \
  --json benchmarks/results/<date>-gfx1151-hipengine-paro-readme-sweep.json
```

This lower-level command is not a complete refresh: it does not run GGUF,
llama.cpp, vLLM, environment capture, or summary assembly.

### Speculative decode refresh

Exact/default GGUF MTP, fixed 10-cycle suite:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route resident-b1-probe-block-direct-cap32k-minrows2-pmin05 \
  --record-cycle-stage-timings \
  --require-cached-build \
  --output benchmarks/results/<date>-ar-mtp-exact-full.json
```

`llama-compat` natural24 direct contract:

```bash
PYTHONPATH=. HIPENGINE_HIP_ARCH=gfx1151 \
python3 scripts/gguf_ar_mtp_suite.py \
  --scope full \
  --mtp-route llama-compat-device-chain-dp4a-q6top1dp4a-x8q6-denseq8all-x8top1-f32ssm-routerrow-draftdenseq8-draftonly-directcommit \
  --budgets 2 --cycles 24 --max-output-tokens 24 \
  --record-cycle-stage-timings \
  --output benchmarks/results/<date>-ar-mtp-llama-compat-natural24.json
```

Dense DFlash B=4:

```bash
python3 scripts/dflash_chain_e2e_bench.py \
  --target-model /home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-PARO/snapshots/84f86409151d4f2ec86dc0b6a096d5f6daa7f207 \
  --drafter-model /home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106 \
  --backend hip_gfx1100 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --max-prompts 9 --decode-tokens 64 --draft-budgets 4 \
  --draft-top-k 2 --whole-cycle-gate 0.90 \
  --verifier-mode native_bulk_bplus1 --verifier-graph auto \
  --full-attn-chain-mode batched --canonical-commit-mode branch_copy \
  --adaptive-budget off --hardware-gpu "AMD Radeon Pro W7900" \
  --json benchmarks/results/<date>-dflash-27b-b4.json
```

Use equivalent immutable local snapshots only when the recorded fingerprints
match these target and drafter revisions.

### HIP versus Vulkan microbenchmarks

Microbenchmark claims do not belong in the model-throughput tables. The v2
timing contract and exact bounded rerun commands are in
[`docs/HIP-vs-VULKAN.md`](../docs/HIP-vs-VULKAN.md) and
[`benchmarks/micro/README.md`](micro/README.md). Retained gfx1151 evidence is
[`2026-07-10-hip-vulkan-timing-v2-bounded.json`](micro/results/gfx1151/strix-halo/2026-07-10-hip-vulkan-timing-v2-bounded.json).

## Update Checklist

1. Choose one protocol tuple and record the old artifact before running.
2. Create a clean detached worktree at the revision being measured.
3. Capture the canonical provenance block: GPU identity, configured/resolved
   backend, target arch, VBIOS, power/clock state, kernel, Python, ROCm/HIP
   compiler, Vulkan driver, comparison-engine commit, existing model
   fingerprint, exact argv/environment, and separate staged, unstaged, and
   untracked source state.
4. Run the named warmup, repetition, correctness, and memory protocol. Store raw
   logs outside git and a compact artifact under `benchmarks/results/`.
5. Reject artifacts with missing provenance or failed correctness. A diagnostic
   may be recorded, but it cannot replace a retained row.
6. Update the platform index, table, run record, artifact links, run date, and
   measured revision in this file.
7. Add the required entry to [`benchmarks/CHANGELOG.md`](CHANGELOG.md) and append
   the commands and decision to `WORKLOG.md`.
8. Run the root README sync and validation commands:

```bash
python3 scripts/sync_benchmark_readme.py --write
python3 scripts/sync_benchmark_readme.py --check
python3 -m json.tool benchmarks/results/<new-artifact>.json >/dev/null
git diff --check
```

Run `json.tool` once for each new or changed compact artifact. Do not scan
untracked experiment files as part of the rollup gate.

<a id="natural24-mtp-vs-ar-concurrency-diagnostic"></a>
<a id="blocked--diagnostic-benchmark-attempts"></a>

## Blocked and Diagnostic Benchmark Attempts

- **W7900 GGUF Q4_K_M:** the 2026-07-07 eager run is the last measured path but
  has repeated token `9707` output and `performance_claim=false`. Restore target
  and recurrent-state correctness before using its throughput as a baseline.
- **OpenAI MTP server c=1/2/4/8:** the 2026-07-06 artifacts use decoded-text
  re-tokenization for completion counts and repeat one batch-scoped timing
  payload per choice. The current harness now counts exact IDs across every
  choice, deduplicates owned batch timing, emits canonical provenance, and
  validates route-cap/queue/backend/verifier shape independently, but those
  historical rows predate all four contracts. They remain ineligible until the
  same protocol is rerun.
- **gfx1151 PARO native batching:** the 2026-07-10 primitive gate passes c2-c8,
  but the direct timing matrix used a batch-shaped width-1 oracle. At
  `0c184517` with `hipengine_dirty=false`, the c8-to-c1 gate rejects native c8
  on all rows at generated token index 2 and accepts the serial bridge through
  every width. No native width is routing-eligible until the direct, sparse,
  ragged, and shrinking matrix passes against independent single-request
  prefill/decode.
- **gfx1151 model sweep:** the committed summary omits source/build provenance
  and contains one measured repetition. Its values remain a dated diagnostic.
- **llama.cpp 24 GiB Q8_0 memory:** the former root README tables had no compact
  artifact, model fingerprint, llama.cpp revision, or run date. The numbers were
  removed; rerun before publishing another capacity table.
- **gfx1100 HIP versus Vulkan v2:** no timing-contract v2 matrix has been run on
  W7900. Do not transfer the gfx1151 ratios to gfx1100.

Rejected and superseded rows remain in JSON artifacts, `WORKLOG.md`,
[`benchmarks/CHANGELOG.md`](CHANGELOG.md), and
[`benchmarks/HISTORY.md`](HISTORY.md). Source-lineage targets and external
baselines in the archive are reference values, not hipEngine toplines.

## Table Conventions

- Workload format is `prompt_tokens/decode_tokens`.
- `tok/s` is reported separately for prefill, backend decode, and full request
  wall. Never compare those scopes without labeling them.
- Aggregate concurrency throughput is total generated tokens divided by the
  concurrent group wall. Per-sequence throughput is aggregate divided by live
  rows only when every row generates the same number of tokens.
- `Peak GiB` names the allocator or whole-card scope in the run record.
- Bold ratios in retained speculative rows identify speedup against the true
  same-protocol AR control. Plain maxima in diagnostic cross-engine tables are
  not promoted as wins.
