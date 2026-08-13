# Qwen3.6-27B Q4_K_M on RX 7900 XTX: Single-Layout Campaign

Status: **reopened on 2026-08-13 for the prefill gate after retained materially new sole-T16 dataflows; all other cross-engine blockers remain.** All requested 512/128, 1K/128, and 4K/128 AR shapes plus natural B1-B3 fit; live target+NextN residency has zero duplicate/alternate payload bytes; deep eager/PM4 state and complete `KVLiveSpans`, transactions, cancellation, public torch-free lifecycle, and the 601-second mixed soak pass; PM4 wins all 15/15 paired HIP-graph samples; and the shared gfx1100 route passes the complete W7900 safeguard. The original “beat both llama.cpp backends everywhere” objective is still **not met**: the 512 HIP+1% prefill margin, every memory row, 4K AR decode, Vulkan B4 MTP speed, and Vulkan MTP memory fail. The latest ordered pair-only Q6-QKV/Q4-gate source-F16 route improves binding M512/M1024 full prefill **0.80%/0.57%** on XTX and **0.45%/0.74%** on W7900, moving selector-unset XTX to **965.209/1003.206/983.082 tok/s** at 512/1K/4K. This reaches raw-HIP parity at 512 with a **0.06% lead**, puts 1K/4K **2.26%/3.84% above** raw HIP, and clears the frozen HIP+1% gates at 1K/4K by **1.25%/2.81%**; 512 is now only **0.93%** short of that margin.

Primary hardware: AMD Radeon RX 7900 XTX / `gfx1100` / 24 GiB, currently
HIP GPU1, Vulkan device `Vulkan1`, PCI `0000:10:00.0`, sysfs `card0`, unique ID
`0xcc4d02090dc9c3ff`. Re-resolve and record these identities before every
retained run; enumeration order is not an identity.

Model:
`/models/gguf/Qwen3.6-27B-Q4_K_M.gguf`, 17,106,773,120 bytes
(15.9319 GiB), SHA-256
`a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f`.

This is the 24 GiB continuation of
[`QWEN36-27B-GGUF-CAMPAIGN.md`](QWEN36-27B-GGUF-CAMPAIGN.md). That campaign
optimized a 48 GiB W7900 route and intentionally accepted broad Q4 alternate
layouts. This campaign changes the optimization constraint: **one logical
weight gets one physical device payload.** It must fit the XTX without losing
the retained W7900 performance and then beat clean, same-commit llama.cpp HIP
and Vulkan on the XTX.

Related authorities:

- [`PLAN.md`](PLAN.md) — architecture and four-axis registry invariants.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence, anti-gaming, true-AR, and MTP
  timing rules.
- [`TESTING.md`](TESTING.md) — RED/GREEN, CPU-reference, KL/top-1, and profiler
  gates.
- [`KERNELS.md`](KERNELS.md) — in-tree kernel/source-lineage workflow.
- [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) — speculative-cycle ownership.

---

## 1. Objective and non-negotiable closure gates

The goal is not merely to make the model fit. The final package-default
`hip_gfx1100` Q4_K_M path must satisfy all of these conditions:

1. **Zero alternate-layout weight duplication.** AR and MTP each keep exactly
   one owned device payload per logical immutable model tensor. Aliases are
   allowed only when they point into the same physical allocation. Pack8+T16,
   raw+T16, T16+X8, raw+X8, and target/drafter copies of the same immutable
   tensor are forbidden in the promoted path.
2. **Fit with operating headroom.** Every 512/128, 1024/128, 4096/128, and
   natural-MTP run completes on the 23.984 GiB device. Peak whole-device VRAM
   must leave at least 1.0 GiB free; this safety bound does not replace the
   stricter cross-engine memory gate below.
3. **No W7900 performance regression.** The same single-layout route becomes
   the gfx1100 default on both cards. It must retain the current W7900 512/128
   and 4096/128 prefill/decode controls and natural-MTP result within a
   predeclared paired noise gate. A capacity-specific slow XTX fallback is not
   closure.
4. **Beat both clean llama.cpp backends on the XTX.** At every AR shape,
   hipEngine prefill and context-matched AR decode must beat both llama.cpp HIP
   and llama.cpp Vulkan. Natural compatible MTP must beat each backend's best
   valid budget over the complete category suite, using the common 24-transition
   timing boundary.
5. **Match or reduce memory.** For every matched AR workload and the selected
   natural-MTP workload, hipEngine's whole-device peak delta must be less than
   or equal to the lower of the clean llama.cpp HIP and Vulkan peak deltas.
   Internal tracked memory is reported too, but cannot substitute for the
   cross-engine sysfs measurement.
6. **Correct and stable.** Layout/kernel correctness, full model state,
   transactional MTP, graph/PM4 lifecycle, repeated reset, cold load/unload,
   and teardown gates all pass. No performance or memory win can weaken the
   KL <= 0.05 / top-1 >= 90% product gate.
7. **No hidden fallback allocation.** Missing single-layout primitives fail
   during planning/admission or use a registered same-layout unfused chain.
   They never lazily allocate a second weight representation during prefill,
   graph capture, decode, or MTP.

“Beat” means a positive same-card median delta with every required sample and
provenance field retained. A result inside normal timing noise is not declared
a win. Before the final campaign run, freeze the repetition count and an
explicit practical margin (initially 1%) from clean baseline variance; do not
choose the margin after seeing the candidate.

### 1.1 Scope

In scope:

- c=1 engine prefill and AR decode at 512/128, 1024/128, and 4096/128;
- dense trailing-NextN MTP under llama.cpp-compatible sampling/output timing;
- resident model, NextN assets, state/KV, graph, scratch, and lifecycle memory;
- clean llama.cpp HIP and Vulkan baselines on the same physical XTX;
- W7900 non-regression after changing the shared gfx1100 default.

Out of scope for closure:

- c>1 serving, TP/offload, other quants, and other models;
- host-RAM parity as a substitute for VRAM parity;
- repeated-token MTP as a natural acceptance/speed claim;
- a W7900-only dual-layout route or an XTX-only slow route;
- prompt-, token-, category-, layer-subset-, or candidate-ID-specific tuning.

---

## 2. Known starting point

### 2.1 Why the 27B currently fails on 24 GiB

The GGUF itself is 15.9319 GiB. The current W7900-optimized route is larger
because every one of 288 rank-2 Q4 projections keeps both expanded pack8 and
compact T16 device layouts:

| Q4 family | Tensors | Source/raw GGUF | Current pack8 | Current T16 |
| --- | ---: | ---: | ---: | ---: |
| `attn_gate` | 48 | 0.7910 GiB | 1.0547 GiB | 0.8129 GiB |
| `attn_k` | 16 | 0.0439 GiB | 0.0586 GiB | 0.0452 GiB |
| `attn_output` | 16 | 0.2637 GiB | 0.3516 GiB | 0.2710 GiB |
| `attn_q` | 16 | 0.5273 GiB | 0.7031 GiB | 0.5420 GiB |
| `attn_qkv` | 24 | 0.6592 GiB | 0.8789 GiB | 0.6775 GiB |
| `attn_v` | 8 | 0.0220 GiB | 0.0293 GiB | 0.0226 GiB |
| `ffn_down` | 32 | 1.4941 GiB | 1.9922 GiB | 1.5356 GiB |
| `ffn_gate` | 64 | 2.9883 GiB | 3.9844 GiB | 3.0713 GiB |
| `ffn_up` | 64 | 2.9883 GiB | 3.9844 GiB | 3.0713 GiB |
| **Total** | **288** | **9.7778 GiB** | **13.0371 GiB** | **10.0494 GiB** |

The 23.0865 GiB combined pack8+T16 row is only these 288 Q4 tensors; it does not
include Q5/Q6/Q8/F32/BF16 weights, embeddings/head, KV/state, scratch, graphs,
or allocator/runtime overhead.

Current retained W7900 tracked peaks are:

| Workload | Current peak | Published performance |
| --- | ---: | --- |
| 512/128 AR | 29.7864 GiB | 235.434 prefill / 23.296 decode tok/s |
| 4096/128 AR | 32.6107 GiB | 216.784 prefill / 21.897 decode tok/s |
| natural25 MTP | 30.4774 GiB | true AR 22.926; B3 61.147 tok/s (2.6671x) |

Sources: the final
[`2026-08-07 Qwen3.6-27B audit`](../benchmarks/results/2026-08-07-qwen36-27b-latest-vulkan-parity-exhaustion-audit.json),
[`direct proposal/target handoff`](../benchmarks/results/2026-08-06-qwen36-27b-direct-proposal-target-handoff-retained.json),
and the two Q4 sidecar artifacts linked below.

A current-production XTX probe fails during model admission before prefill. P0
must preserve that blocked result as a compact artifact; there is no valid XTX
hipEngine performance baseline until the resident layout fits.

### 2.2 Capacity projections, not measurements

Simple subtraction gives the first implementation target:

| Candidate residency change | 512/128 projection | 4096/128 projection | natural-MTP projection |
| --- | ---: | ---: | ---: |
| Remove T16, keep pack8 | 19.7370 GiB | 22.5613 GiB | 20.4279 GiB |
| Remove pack8, keep T16 | **16.7493 GiB** | **19.5736 GiB** | **17.4403 GiB** |
| Replace both with sole raw GGUF | 16.4777 GiB | 19.3020 GiB | 17.1687 GiB |

These are arithmetic projections from existing W7900 tracked peaks. They do
not account for layout-specific scratch changes or undiscovered NextN/Q8/root
copies and are not XTX memory claims.

Removing T16 is an acceptable short-lived fit diagnostic, but it is not the
campaign goal: it discards the fast decode/verify layout, leaves the 33%-larger
pack8 representation, and leaves only about 1.42 GiB projected headroom at 4K.
The first production candidate is therefore **sole-resident T16**, with direct
raw/byte-neutral execution as the next rung if T16 cannot meet the external
memory floor.

### 2.3 Existing evidence that must not be lost

- Q4 T16 FFN sidecars improved complete B3 wall and are BF16-bit-exact:
  [`2026-08-05-qwen36-27b-q4t16-ffn-sidecars-retained.json`](../benchmarks/results/2026-08-05-qwen36-27b-q4t16-ffn-sidecars-retained.json).
- Extending T16 to all 288 rank-2 Q4 projections improved every natural
  prompt-budget row but disclosed 10.049 GiB of added residency:
  [`2026-08-05-qwen36-27b-q4t16-row-selective-sidecars-retained.json`](../benchmarks/results/2026-08-05-qwen36-27b-q4t16-row-selective-sidecars-retained.json).
- Wide Q6, Q5 `ssm_out`, root head, and full-attention V already use
  sole-resident compressed layouts. Preserve those wins; audit them rather than
  reverting to dense BF16.
- Current llama-compatible serving temporarily enables raw Q8 and X8-related
  modes. Their physical ownership must be audited; “default-off elsewhere” is
  not proof that MTP is duplicate-free.

---

## 3. Single-layout architecture contract

### 3.1 Physical ownership invariant

For each logical source tensor, the resident manifest records:

```text
(source file hash, tensor name, source GGML type, source byte range)
-> (one physical device range, canonical layout, quant registry key, owners/views)
```

The audit treats overlapping aliases into one allocation as one payload and
non-overlapping allocations as separate payloads. The promoted AR and MTP
manifests must report:

```text
alternate_layout_weight_bytes = 0
duplicate_logical_weight_bytes = 0
logical_weights_with_multiple_physical_payloads = 0
steady_state_weight_allocations = 0
```

Allowed and separately classified:

- non-owning tensor views/aliases into the same byte range;
- KV/Conv/GDN state, graph parameter slabs, and bounded reusable activation
  workspace;
- code objects and backend allocator bookkeeping;
- host mmap/raw arrays used to build the sole device representation.

Not allowed:

- a persistent fallback shadow, even if a fast path usually wins;
- a second target copy owned by the MTP provider;
- T16 plus raw/X8 under llama-compatible mode;
- per-context, per-budget, or per-row alternate weight payloads;
- load-time upload of one layout followed by a second without freeing the
  first before admission is declared complete.

### 3.2 Operation-complete canonical formats

A format can become canonical only when all operations that consume the tensor
have registered implementations over that same format:

- c=1 GEMV/dual-GEMV/fused SiLU;
- rows 2-6 verifier rowtiles as required by retained/selected MTP budgets;
- M=512/1024/4096 bulk prefill, including tails;
- fused projections and a numerically equivalent unfused fallback chain;
- graph capture/replay and explicit PM4 transport where admitted;
- CPU-reference/dequant oracle and source-layout round trip.

If one operation is missing, implement it against the canonical bytes. Do not
solve the gap by retaining a second layout.

### 3.3 Candidate order

1. **Sole T16 (preferred first).** Keep the current exact fast decode/verify
   bytes, remove pack8 allocation, and add/route dense bulk-prefill plus every
   fallback against T16. This preserves the measured Q4 decode wins and removes
   13.0371 GiB from the current target residency.
2. **Sole raw GGUF.** If sole T16 misses the lower llama.cpp memory peak or loses
   important prefill performance, use source Q4_K blocks directly for both
   MMVQ/GEMV and MMQ/prefill. This is the llama.cpp model and saves a further
   projected 0.2716 GiB across the 288 matrices.
3. **Sole byte-neutral repack.** Consider only if it has one payload, no larger
   whole-device peak than both comparators, and a measured complete-wall reason.
   A new sidecar beside raw/pack8/T16 is forbidden.

The old pack8 implementation may remain temporarily as a separately selected
**sole-layout** bisection/control session. No process may materialize it beside
T16. Add its removal condition to [`REFACTOR.md`](REFACTOR.md) if it survives
promotion.

### 3.4 Registry and backend rules

- Add layout/operation variants through `(backend, layer, quant, variant)`
  registration. Do not add model/engine branches on backend, quant, GPU name,
  or VRAM size.
- Both XTX and W7900 are `gfx1100`; the production representation is shared.
- Admission reads the planned manifest and workspace/KV budget before upload.
  It does not discover capacity by catching OOM after partial materialization.
- Existing exact fused paths keep registered same-format unfused fallbacks.
- Any temporary selector is default-off until the named gate passes and gets a
  concrete deletion condition in `REFACTOR.md`.

---

## 4. What to learn from llama.cpp

This is source guidance, not permission to copy without lineage and tests.
At the current clean comparator commit `c8e03ce8122b7af76f836d53efde6df1ce5ec437`:

- `src/llama-model.cpp` and `src/llama-model-loader.cpp` allocate backend model
  buffers and upload/map each GGUF tensor once. Host mmap is a loading strategy;
  on a discrete XTX it must not be described as zero-copy device inference.
- HIP uses the CUDA/HIP backend's direct quant paths:
  `ggml/src/ggml-cuda/mmvq.cu`, `mmq.cu`, `mmq.cuh`, and `vecdotq.cuh` consume
  Q4_K blocks while quantizing activations into transient Q8_1 tiles.
- Vulkan uses
  `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q4_k.comp` and the Q4_K
  matmul pipelines in `ggml-vulkan.cpp`; quant decode happens in the shader,
  not through a model-sized expanded shadow.

Port only an operation/layout mechanism that a current profile justifies.
Before kernel work, follow `KERNELS.md`, run the lineage check, cite exact
source file+commit, write the RED fixture first, and retain a named
`rocprofv3 --kernel-trace` execution.

The lesson is structural: dequantize/transform weights in registers or bounded
LDS/workspace while reading one compressed payload. It is not “copy all of
llama.cpp,” and it does not override hipEngine's `KVLiveSpans`, plugin registry,
or torch-free runtime contracts.

---

## 5. Clean llama.cpp HIP and Vulkan baseline protocol

### 5.1 Freeze one source revision for both backends

The existing `/home/lhl/llama.cpp/llama.cpp-hip` tree is unsuitable for the
headline: it is ahead/behind upstream, has tracked MTP instrumentation changes,
and its current `build/` targets `gfx1151`. The Vulkan tree is clean at
`c8e03ce81` except an untracked `.pi/tasks/` directory.

At campaign start, fetch once, select one tracked-clean commit, and create a
clean detached comparison worktree. Build HIP and Vulkan from that **same host
source revision**:

```bash
LLAMA_SOURCE=/home/lhl/llama.cpp/llama.cpp-vulkan
LLAMA_COMMIT=$(git -C "$LLAMA_SOURCE" rev-parse HEAD)
LLAMA_XTX=/tmp/llama-qwen36-27b-xtx-$LLAMA_COMMIT

git -C "$LLAMA_SOURCE" worktree add --detach "$LLAMA_XTX" "$LLAMA_COMMIT"

cmake -S "$LLAMA_XTX" -B "$LLAMA_XTX/build-hip" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_VULKAN=OFF \
  -DAMDGPU_TARGETS=gfx1100
cmake --build "$LLAMA_XTX/build-hip" -j 16 --target llama-bench llama-server

cmake -S "$LLAMA_XTX" -B "$LLAMA_XTX/build-vulkan" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF -DGGML_VULKAN=ON
cmake --build "$LLAMA_XTX/build-vulkan" -j 16 --target llama-bench llama-server
```

Configure and run the HIP build under the same hermetic TheRock ROCm
component environment used for the hipEngine comparison; do not compile with
one HIP stack and load another at runtime. Record source commit/describe/status,
complete CMake cache, compiler, ROCm, Mesa/RADV, binary/shared-library hashes,
and whether upstream changed from `c8e03ce81`. If compatible MTP needs a local
instrumentation patch, use a separate profile build and diff hash; it cannot
replace the clean speed/memory baseline.

### 5.2 Device preflight

The current mapping is HIP GPU1 / Vulkan1 / sysfs card0, but every run must
prove it:

```bash
rocm-smi --showbus --showproductname --showuniqueid --showmeminfo vram
vulkaninfo --summary
cat /sys/class/drm/card0/device/{unique_id,mem_info_vram_total}
grep PCI_SLOT_NAME /sys/class/drm/card0/device/uevent
```

Expected physical identity is XTX, `gfx1100`, PCI `0000:10:00.0`, total
25,753,026,560 bytes. Stop if another process is using material VRAM. Run HIP
and Vulkan serially; no W7900 benchmark may overlap CPU/I/O with a retained XTX
run.

### 5.3 Standardized AR and peak-memory floor

Use the existing split-phase wrapper for five repetitions at all requested
shapes. It samples the same amdgpu VRAM domain for both APIs:

```bash
ROOT=/tmp/llama-qwen36-27b-xtx-$LLAMA_COMMIT
MODEL=/models/gguf/Qwen3.6-27B-Q4_K_M.gguf
OUT=/tmp/qwen36-27b-xtx-baselines
mkdir -p "$OUT"

HIP_VISIBLE_DEVICES=1 python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench "$ROOT/build-hip/bin/llama-bench" \
  --model "$MODEL" --quant Q4_K_M --backend hip \
  --workloads 512/128 1K/128 4K/128 --repetitions 5 \
  --ngl 99 --flash-attn 1 --cache-type-k f16 --cache-type-v f16 \
  --extra-args="-dev ROCm0" --poll 5 --memory-domain vram \
  --card-name card0 --output "$OUT/llamacpp-hip-ar-peak.json"

python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench "$ROOT/build-vulkan/bin/llama-bench" \
  --model "$MODEL" --quant Q4_K_M --backend vulkan \
  --workloads 512/128 1K/128 4K/128 --repetitions 5 \
  --ngl 99 --flash-attn 1 --cache-type-k f16 --cache-type-v f16 \
  --extra-args="-dev Vulkan1" --poll 5 --memory-domain vram \
  --card-name card0 --output "$OUT/llamacpp-vulkan-ar-peak.json"
```

This gives the standardized `llama-bench` floor. It does not replace the
context-matched server decode row below.

### 5.4 Context-matched AR

Use the repeated-token server protocol only for AR shape control. The harness
interprets each shape's decode count as timed transitions, requests one extra
visible output (129 for a 128-transition row), and records native plus
transition-normalized timing, client wall, exact IDs/hash, and phase peak VRAM
for each shape. Token-repeat mode is a shape control, not a natural MTP claim.

```bash
# HIP: HIP_VISIBLE_DEVICES=1, device becomes ROCm0.
HIP_VISIBLE_DEVICES=1 python3 scripts/llamacpp_mtp_bench.py \
  --server-bin "$ROOT/build-hip/bin/llama-server" --model "$MODEL" \
  --ctx-size 8192 --gpu-layers 99 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 --mode base \
  --protocol token-repeat --token-id 9707 \
  --shapes 512/128 1024/128 4096/128 \
  --seed 12345 --temperature 0 --top-k 1 --top-p 1 --min-p 0 \
  --sample-memory --poll 5 --memory-domain vram --card-name card0 \
  --server-extra-arg=-dev --server-extra-arg=ROCm0 \
  --server-extra-arg=--reasoning --server-extra-arg=off \
  --server-extra-arg=--perf --output "$OUT/llamacpp-hip-context-ar.json"

# Vulkan: preserve global enumeration and select the XTX explicitly.
python3 scripts/llamacpp_mtp_bench.py \
  --server-bin "$ROOT/build-vulkan/bin/llama-server" --model "$MODEL" \
  --ctx-size 8192 --gpu-layers 99 --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 --mode base \
  --protocol token-repeat --token-id 9707 \
  --shapes 512/128 1024/128 4096/128 \
  --seed 12345 --temperature 0 --top-k 1 --top-p 1 --min-p 0 \
  --sample-memory --poll 5 --memory-domain vram --card-name card0 \
  --server-extra-arg=-dev --server-extra-arg=Vulkan1 \
  --server-extra-arg=--reasoning --server-extra-arg=off \
  --server-extra-arg=--perf --output "$OUT/llamacpp-vulkan-context-ar.json"
```

### 5.5 Natural compatible MTP

Run clean HIP and Vulkan separately for B1-B5 over the committed ten-prompt
suite. Each budget gets its own candidate-local warmup. Request 25 visible
outputs and compare 24 timed transitions per prompt. Use a true no-MTP run from
the same backend/harness/protocol; `off`/B0 verifier telemetry is not a speed
denominator.

Common contract:

- `benchmarks/prompts/mtpbench-code-general-ja.jsonl` and its recorded SHA;
- all `code`, `general_en`, `general_ja`, `mixed_ja_en`, six train, and four
  heldout prompts;
- reasoning off, temperature 0, top-k 1, top-p 1, min-p 0, seed 12345;
- `-ngl 99 -fa on -ctk f16 -ctv f16 -c 8192 -np 1`;
- no prompt cache; server `--perf`; device and speculative-draft device pinned
  to the XTX;
- complete engine transition wall, client wall, proposal/acceptance ledger,
  output hashes, and peak VRAM.

Command skeleton (run once per backend and budget after P0 adds server-memory
sampling):

```bash
for B in 1 2 3 4 5; do
  python3 scripts/llamacpp_mtp_bench.py \
    --server-bin "$SERVER" --model "$MODEL" --ctx-size 8192 \
    --gpu-layers 99 --flash-attn on --cache-type-k f16 --cache-type-v f16 \
    --draft-max "$B" --mode both --protocol natural \
    --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
    --max-tokens 25 --seed 12345 --temperature 0 --top-k 1 --top-p 1 --min-p 0 \
    --sample-memory --poll 5 --memory-domain vram --card-name card0 \
    --server-extra-arg=-dev --server-extra-arg="$DEVICE" \
    --server-extra-arg=--spec-draft-device --server-extra-arg="$DEVICE" \
    --server-extra-arg=--reasoning --server-extra-arg=off \
    --server-extra-arg=--perf --output "$OUT/$BACKEND-natural-b$B.json"
done
```

For HIP, invoke the loop under `HIP_VISIBLE_DEVICES=1` with `DEVICE=ROCm0`.
For Vulkan use `DEVICE=Vulkan1`. Select each backend's winning budget by
full-suite transition tok/s subject to complete protocol, finite outputs, and
all category/heldout disclosures. Do not inherit the W7900 Vulkan B4 winner.

---

## 6. hipEngine benchmark and evidence protocol

### 6.1 AR matrix after first fit

Use right-sized resident sessions, one discarded warmup, five measured resets
for final XTX comparison, bulk prefill, production c=1 graph/transport, and 128
measured decode transitions:

```bash
MODEL=/models/gguf/Qwen3.6-27B-Q4_K_M.gguf
OUT=/tmp/qwen36-27b-xtx-hipengine
HIPCC_VERSION="$OUT/hipcc-version.txt"
mkdir -p "$OUT"
hipcc --version > "$HIPCC_VERSION"

for P in 512 1024 4096; do
  HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 \
  HIPENGINE_COMPILER_VERSION_FILE="$HIPCC_VERSION" \
  HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
  python3 scripts/qwen35_gguf_bench.py \
    --model "$MODEL" --quant gguf_q4_k_m --token-id 9707 \
    --prompt-length "$P" --decode-tokens 128 --warmup-decode-tokens 1 \
    --warmup-runs 1 --measured-runs 5 --persistent-session \
    --force-bulk-prefill --bulk-prefill-attention-mode bulk \
    --use-wmma-prefill --use-gemv-decode \
    --graph-replay-decode --graph-steps-per-replay 1 \
    --compiler-version-file "$HIPCC_VERSION" --require-cached-build \
    --json "$OUT/hipengine-${P}x128.json"
done
```

Final artifacts attach 5 ms sysfs `card0` sampling around the complete process
and phase markers. Internal tracked peak, current-before-close,
after-close, owned-weight breakdown, KV/state, graph, and workspace remain
separate fields.

### 6.2 Dense natural AR/MTP

The canonical hipEngine suite is `scripts/qwen36_dense_gguf_suite.py`. Start
with B1-B3; admit B4/B5 only through complete transaction and preliminary
complete-wall gates, never because llama.cpp selected that budget.

```bash
HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 \
HIPENGINE_COMPILER_VERSION_FILE="$HIPCC_VERSION" \
HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
python3 scripts/qwen36_dense_gguf_suite.py \
  --model "$MODEL" --quant gguf_q4_k_m \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-new-tokens 25 --candidate-budgets 1,2,3 \
  --target-verify-mode native --runs 1 \
  --compiler-version-file "$HIPCC_VERSION" --require-cached-build \
  --output "$OUT/hipengine-natural25-b1-b3.json"
```

The retained comparison requires true AR, all ten prompts, full/train/heldout
and every-category rates, 250 visible outputs/240 transitions, acceptance and
state ledgers, complete proposal/verify/commit/residual wall, output identity,
and both internal and whole-device memory. Run enough independent full suites
to establish the frozen variance rule; ask before the >5-minute final rerun as
required by `AGENTS.md`.

### 6.3 Cross-engine columns

Every final row contains:

- exact model/source/binary/hardware identities;
- prompt and output counts plus token/hash identity;
- prefill ms and tok/s;
- transition-normalized AR or MTP complete wall and tok/s;
- native self-reported timing separately;
- warmups, all samples, median, spread, and selection policy;
- whole-device baseline/peak/delta and sampling interval;
- hipEngine tracked peak/current/teardown plus manifest class totals;
- cache/KV dtype, context allocation, graph/PM4 mode, and startup exclusions.

Cross-engine closure is computed per shape:

```text
hipEngine prefill > max(llama HIP prefill, llama Vulkan prefill)
hipEngine AR decode > max(llama HIP AR decode, llama Vulkan AR decode)
hipEngine peak delta <= min(llama HIP peak delta, llama Vulkan peak delta)
```

For MTP, apply the same speed rule to each backend's independently selected
valid budget and require no hidden train/heldout/category loss. Also report
hipEngine MTP/true-AR and each llama backend's MTP/true-AR; ratios cannot replace
absolute tok/s.

---

## 7. Implementation plan and punchlist

Statuses begin as `[ ]`; check an item only after its acceptance evidence is
committed. Each retained logical unit gets its own immutable worklog entry and
atomic commit.

### P0 — Freeze comparator and blocker evidence

- [x] **P0.1 Record immutable XTX identity.** Capture HIP/Vulkan/sysfs mapping,
  total/idle VRAM, clocks/power policy, driver, ROCm, Mesa/RADV, CPU, and host.
  Stop on identity mismatch or material background VRAM use. Evidence:
  [`2026-08-12 pre-single-layout blocker`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-pre-single-layout-blocked.json).
- [x] **P0.2 Preserve the current hipEngine blocker.** Re-run the shortest
  current-production AR admission and dense MTP admission once, record exact
  OOM stage/current peak/active allocations, and publish a blocked artifact.
  Do not benchmark a partial model. Evidence: the same artifact records both
  entries failing target materialization at `blk.49.ffn_gate.weight.pack8.qweight`
  after 24,462,066,048 tracked bytes, with zero tracked bytes after cleanup.
- [x] **P0.3 Build clean same-commit llama.cpp HIP and Vulkan.** Use the process
  in section 5.1; never use the dirty/stale HIP tree for the headline. Both
  clean `c8e03ce8122b` builds and their cache/binary/library identities are
  frozen in
  [`2026-08-12 clean llama.cpp builds`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-clean-llamacpp-builds.json).
- [x] **P0.4 Extend server memory sampling.** Reuse
  `hipengine.util.amdgpu_vram.VramSampler` in `llamacpp_mtp_bench.py` (or a
  small shared wrapper) so base and each MTP server process record baseline,
  phase peak, process peak, delta, sample count, interval, card identity, and
  startup/teardown scope. Fake-sysfs/subprocess coverage lives in
  `tests/test_llamacpp_mtp_bench_metrics.py`; real-run evidence is attached by
  P0.5.
- [x] **P0.5 Run the complete llama.cpp XTX baseline.** Standardized five-sample
  AR, context-matched 128-transition AR, and natural B1-B5 completed for both
  clean same-commit backends. Each natural budget covers ten prompts, 250
  visible outputs, 240 transitions, all four categories, and the fixed
  train/heldout split with process-scope 5 ms VRAM sampling. Evidence:
  [`2026-08-12 clean llama.cpp floors`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-clean-llamacpp-floors.json).
- [x] **P0.6 Freeze the target table.** HIP wins every standardized-prefill and
  context-AR speed column; Vulkan sets every lower AR memory ceiling and its B4
  wins natural MTP. The scorecard and artifact freeze the exact rows plus a 1%
  practical speed margin before runtime changes. Baseline refresh after a
  source/driver change creates a new table; it does not overwrite this one.

Exit: two clean external backends have reproducible perf+memory rows and the
current hipEngine no-number/OOM state is explicit.

### P1 — Make weight duplication mechanically visible

- [x] **P1.1 Add a plan-time byte census.** For every weight spec, report source
  bytes, planned allocations, canonical layout, aliases, and alternate-layout
  bytes without requiring a GPU. Implemented by
  `census_qwen35_gguf_weight_specs()` with a device-free actual-model fixture.
- [x] **P1.2 Add a runtime physical-range census.** The retained census
  deduplicates actual `(device, ptr, nbytes)` ranges with owner/view and memory
  class labels. Live target+NextN evidence covers target, `root_shared`, and
  NextN weights; benchmark ownership snapshots separately report state/KV,
  graph/runtime, and workspace classes.
- [x] **P1.3 Add a duplicate invariant checker.** Fail if one logical source
  tensor owns multiple physical payloads or if a payload appears under an
  undeclared logical owner. `Qwen35GGUFRuntimeResidencyCensus.assert_single_layout()`
  rejects duplicate allocation roles, alternate layouts, and cross-source
  physical aliases.
- [x] **P1.4 Cover aliases.** Unit coverage exercises tied/untied roots,
  declared shared target assets, incompatible aliases, arena views, duplicate
  copies, and alternate layouts. The actual 27B live census additionally proves
  mapped `token_embd.weight` and untied `output.weight` are each shared once
  while the distinct NextN head norm remains owned.
- [x] **P1.5 Freeze current and target manifests.** The current census
  reproduces all 288 Q4 pack8/T16 owners and 10,790,502,400 alternate-layout
  bytes. The candidate predicts the same 10,790,502,400 canonical T16 bytes,
  zero alternate-layout bytes, and passes `assert_single_layout()`. Evidence:
  [`sole-T16 first fit`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-sole-t16-first-fit.json).

Exit: memory ownership is a testable contract, not inferred from peak deltas.

### P2 — RED: define the sole-Q4 representation

- [x] **P2.1 Flip the mapping expectation.** Candidate mapping coverage now
  requires all 288 rank-2 Q4 specs to own only `gguf_q4_k_t16_v1/tiles`, while
  separately freezing the old dual-layout control census.
- [x] **P2.2 Add actual-weight layout oracles.** The nine-role actual model map,
  representative cache-cold actual-weight screens, rows 16/33/512/1024/4096 and
  tail GPU/CPU gates, plus deep full-model state prove the retained T16 bytes and
  finite outputs. The raw-source rung's changed-association rows were screened
  separately and rejected before model integration.
- [x] **P2.3 Add operation coverage REDs.** One T16 quant/layout key covers c1,
  production verifier rows 2-4, M512/M1024/M4096/tails, fused dual+SiLU, single
  projection, and the same-layout unfused chain. Rows 5/6 stay outside package
  admission after the B4/B5 break-even rejection.
- [x] **P2.4 Add no-lazy-shadow REDs.** Dispatch tests assert canonical `tiles`
  pointers for c1/verifier/prefill/fused routes. Runtime target+NextN census,
  five-reset AR, natural MTP, 100 warm resets/400 PM4 submits, and the 601-second
  public soak retain zero alternate/duplicate payload bytes and constant warm
  ownership, with zero tracked allocations after close.
- [x] **P2.5 Predeclare candidate order and keep rule.** Sole T16 ran first. Its
  failed external columns opened sole raw; cache-cold verifier, fused FFN,
  operation-completeness, correctness, and maximum-memory-ceiling screens reject
  raw without hybrid role salvage.

Exit: complete; the retained sole-T16 route passes these representation and
operation-coverage gates, while the predeclared raw alternative is rejected.

### P3 — Sole-T16 materialization and same-layout fallbacks

- [x] **P3.1 Change rank-2 Q4 planning.** The gfx1100 package capability now
  materializes only `gguf_q4_k_t16_v1/tiles` for all 288 rank-2 Q4 tensors;
  pack8 is never uploaded first.
- [x] **P3.2 Keep host loading bounded.** Materialization repacks one tensor at a
  time from the GGUF mmap, uploads the sole final payload, and drops the temporary
  before continuing. A separate 106.6-second monitor on the first post-warm
  public-soak attempt records 0.886-GiB maximum live RSS (5.086-GiB process
  `VmHWM`, which includes earlier loading transients) and 0.718-GiB final RSS;
  it is host-memory evidence, not coverage of the final 601-second soak.
  Right-sized 512/1K/4K sessions and the selected natural route all complete
  with 6.74-7.83 GiB of
  measured device headroom; no OOM-driven fallback or partial second upload is
  observed.
- [x] **P3.3 Route c1 decode and fused FFN.** Exact T16 single/dual/fused c1
  owners consume canonical `tiles`; no pack8 allocation remains.
- [x] **P3.4 Route verifier rows.** Existing exact rows-2-4 single and fused
  owners now consume canonical `tiles`, covering the admitted B1-B3 shapes.
  Rows 5/6 remain outside admission.
- [x] **P3.5 Implement dense T16 bulk prefill.** Q4/Q5/Q6 now have registered
  dense T16 WMMA owners. Actual-weight 512/1024/4096 and tail screens are exact
  to their prior T16 associations; the retained Q4 producer shares one decoded
  K256 slab across four waves without another weight payload.
- [x] **P3.6 Implement same-layout unfused fallbacks.** Unsupported composites
  use the registered T16 primitive chain; no pack8 shadow is available to lazy
  fallback.
- [x] **P3.7 Reconcile registry keys.** Dense/small-row/c1 owners resolve through
  four-axis keys and package capabilities; no model/backend capacity branch was
  added.

Exit: **complete.**
Plan census reports zero rank-2 Q4 alternate-layout bytes; exact c1/rows-2-4
coverage and M512/M1024/M4096/tail prefill pass, and the complete model first
fits the XTX. A sole device-visible mapped GGUF mmap subsequently removes the
715,161,600-byte root token-table VRAM shadow: tracked residency falls 16.749
to 16.083 GiB and same-workload sampled peak delta falls 17.347 to 16.679 GiB
(the standard 512/128 PM4 row is 16.712 GiB). A model-scoped dense scratch
arena cuts physical bulk scratch 0.589 to 0.111 GiB and a model-scoped
small-weight arena reduces physical weight owners 850 to 370. An output-major
Q4 shared-B slab then cuts the traced 288-call Q4 family 433.351 to 421.447 ms
and raises exact 512/128 prefill 719.232 to 730.589 tok/s without changing
decode or residency. Widening the model-scoped arena from 16 MiB to the first
complete 80-MiB inventory crossover then packs 849 immutable allocations into
one owner, leaves only the 1,042,944,000-byte untied head dedicated, reduces
physical weight owners 370 to 2, and lowers standard sampled peak delta 16.171
to 16.095 GiB (-77.840 MiB). Three rearmed runs remain neutral/exact at
731.185 prefill / 33.525 PM4 decode tok/s; tracked peak is 15.605 GiB. Decode
passes its frozen gate, while prefill and memory do not. Evidence:
[`sole-T16 first fit`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-sole-t16-first-fit.json),
[`mapped-host/PM4 partial pass`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-mapped-host-embedding.json),
[`dense scratch liveness`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-dense-prefill-scratch-liveness.json),
[`small-weight arena`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-small-weight-arena.json),
[`Q4T16 output-major LDS`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-q4-t16-output-major-lds.json),
and [`wide weight arena`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-wide-weight-arena.json).

### P4 — Remove the remaining AR/MTP duplicate assets

- [x] **P4.1 Audit Q8 llama-compatible overrides.** The production target and
  draft plans contain no raw/sidecar Q8 override and both report exactly zero
  alternate-layout bytes.
- [x] **P4.2 Audit Q6 root/NextN X8.** Root and draft consume their sole
  qualified planar/T16 owners; neither plan nor the live physical census finds
  a parallel X8/raw payload.
- [x] **P4.3 Share target-owned immutable roots.** The live target+provider
  census proves `token_embd.weight` and untied `output.weight` are the same
  mapped/device objects in both owners. The distinct
  `blk.64.nextn.shared_head_norm.weight` correctly remains NextN-owned.
- [x] **P4.4 Keep only NextN-specific weights additional.** Live MTP residency
  adds 15 distinct NextN weights / 293,869,568 physical bytes; the shared roots
  account for 1,758,105,600 bytes without a second payload.
- [x] **P4.5 Audit graph and cached uploads.** Across 870 target/draft references
  the live census reports 866 exact physical ranges, zero duplicate roles,
  zero duplicate payload bytes, zero alternate-layout bytes, and no issues.
  The complete natural suite and lifecycle gates return tracked ownership to
  zero after proposal/target graph teardown.
- [x] **P4.6 Make the invariant package-default.** Sole dense T16/planar
  materialization, mapped Q4 embedding, and root borrowing are package defaults;
  rollback deletion criteria remain tracked in `REFACTOR.md`.

Exit evidence:
[`correctness and runtime residency`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-correctness-residency.json).
Complete AR and MTP manifests both report zero duplicate/alternate-layout
weight bytes.

### P5 — Correctness and transaction gate

- [x] **P5.1 CPU/layout bundle.** Mapping/materialization, quant layout,
  registry/fusion, model/import, host-mapping, NextN borrowing, residency, and
  allocator-audit coverage is green. The focused shared-gfx1100 run also found
  and repaired unsupported Q6_K roots being auto-deferred by the mapped-Q4
  capability (`719bd743f`); final selected-node accounting is 64 passed / 5
  expected hardware-or-capacity skips.
- [x] **P5.2 Kernel correctness.** Q4 shared-B is BF16-bit exact to the prior
  T16 producer at rows 16/33/512/1024/4096 and passes its independent CPU gate;
  Q5 is exact to selected T16; planar Q6 is exact to legacy T16 and passes the
  CPU quality gate. Cache-only rocprof names all three retained dense bodies
  with scratch0; mapped-host Q4 embedding is bit-exact to a device owner. The
  output-major Q4 shared-B slab keeps those exact outputs while replacing 132
  scalar LDS instructions with 28 b128 instructions at unchanged resources.
- [x] **P5.3 Full eager/graph AR state.** At 512, 1024, and 4096, clean
  same-file llama.cpp and production bulk eager emit token 9707; each of four
  eager transitions is byte-exact to a freshly recomputed serial-prefix
  reference for FP32 hidden, all 48 Conv/GDN state pairs, and every live byte in
  all 16 full-attention BF16 K/V pairs. From the same resident checkpoints, PM4
  matches eager at all 12 transitions for those fields plus full FP32 logits.
  Complete `KVLiveSpans` (`base_offsets`, device live counts, row positions,
  host `max_live_count`, token positions, and evict mask) is exact when compared
  at the common pre-execution boundary: PM4 stages the next append during replay,
  while scalar eager stages it on entry to `step()`. Each PM4 generation has four
  launches, zero native fallback/unretired submissions, and a retired executable
  after close. The retained 128-transition PM4-vs-HIP speed claim remains the
  separate +1.601% focused control; this packet is correctness-only. Evidence:
  [`correctness and runtime residency`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-correctness-residency.json).
- [x] **P5.4 Dense MTP transaction.** The real XTX B1-B3 gate passes scalar
  logits, reject/partial/full accept, correction, forced and post-commit
  rollback/reseed, dynamic positions, proposal/target graph reuse, provider
  generation, state/KV identity, and zero removed-layout pack8 calls in 93.88 s.
  Cancellation injected during proposal is now observed before target
  verify/commit; request KV buffers are freed and poisoned target/draft owners
  close rather than re-enter their pools.
- [x] **P5.5 Natural semantic gate.** All ten prompts, four categories, six
  train cases, and four fixed heldouts pass for true AR and B1-B3. Every MTP
  token ledger is exact to true AR and every GPU accept result matches CPU.
  Layout-only changes preserve the exact current hipEngine output.
- [x] **P5.6 Torch-free and lifecycle.** A real public XTX gate runs AR, two
  deterministic dense layer-64 MTP requests with target/provider reuse, and the
  opt-in HTTP MTP route. AR/MTP IDs and HTTP text match exactly, torch is absent
  before and after `LLM.close()`, tracked current bytes/allocations return to
  zero, and final whole-card VRAM is 0.8 MiB above baseline. The maximum public
  AR+pooled-MTP/server footprint is disclosed separately at 20.413 GiB delta;
  it is lifecycle evidence, not the selected natural-MTP memory topline.

Exit: correctness is green before any keep is based on performance.

### P6 — XTX fit and performance optimization

- [x] **P6.1 First complete XTX fit.** The final one-warmup/five-measurement
  512/128, 1024/128, and 4096/128 matrix includes first-token and full
  128-transition execution, records tracked/sysfs peaks, and closes cleanly.
- [x] **P6.2 Compare sole T16 to frozen floors.** The initial sole-T16 package
  fit every shape and passed 512/1K decode, but failed every prefill/memory gate
  and 4K decode, triggering the predeclared raw-GGUF evaluation in P6.4. Later
  byte-neutral arithmetic work now also passes 4K prefill; 512/1K prefill,
  every memory row, and 4K decode remain open.
- [x] **P6.3 Profile only the failing column.** The latest cache-only 512 trace
  reconciles 700.435 ms kernel sum against 739.837 ms profiled wall: Q4 is
  421.447 ms (60.17%), Q6 131.298 ms (18.75%), and Q5 63.647 ms (9.09%). The
  output-major LDS keep reduced Q4 from the prior 433.351 ms (-2.747%). Decode
  was separately compared at complete-wall scope before PM4 promotion.
- [x] **P6.4 Raw-GGUF rung if needed.** The source audit covered llama.cpp
  `c8e03ce8122b` CUDA/HIP MMVQ and Vulkan integer-dot MMVQ/MMQ arithmetic.
  Actual-weight raw-Q4/Q8_1 col1/2/4/8, fused gate/up+SiLU, and raw WMMA
  prefill were screened before runtime integration. Favorable repeated
  one-weight results were cache artifacts: distinct same-role pools larger
  than the XTX cache regressed every tested rows-4 family by **12.53-65.73%**,
  and raw fused FFN regressed **29.03-34.56%**. Exact T16 tile4 also regressed
  **9.60%** versus tile8. No source body or route was retained.
- [x] **P6.5 No hybrid salvage.** Raw won only a cache-resident `attn_k`
  prefill leaf while losing that role's required rows-4 verifier operation.
  Every production-representative verifier pool preferred T16. Moreover,
  converting all 288 target Q4 tensors would save at most **0.2716 GiB**,
  below every remaining AR/MTP memory gap; any role subset saves less. No
  mixed canonical policy or dual payload was admitted.
- [x] **P6.6 Re-run complete shapes after each structural keep.** The final
  engine matrix followed the retained shared-B/arena/PM4 changes. The raw and
  tile4 candidates made no runtime change, so no additional broad rerun is
  warranted.

Current exit status: all shapes fit and 4K prefill now passes, but closure still
fails 512/1K prefill, all memory rows, and 4K decode; both operation-complete Q4
representation rungs are adjudicated and compact T16 remains production. Evidence:
[`complete engine AR matrix`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-engine-ar-matrix.json),
[`rejected raw-Q4 rung`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-raw-q4-rung-rejected.json).

### P7 — Compatible natural MTP closure

- [x] **P7.1 Run hipEngine B1-B3 full suite.** One clean independent suite
  covers true AR plus B1-B3 across all ten prompts, four categories, six train
  prompts, four heldouts, and 240 timed transitions/mode. B3 wins at **72.887
  tok/s / 3.5071x AR / 77.17% acceptance**; every MTP token ledger and GPU
  acceptance result matches true AR exactly. A second independent performance
  suite is unnecessary for closure because the binding Vulkan speed and memory
  gaps already fail by 11.06% and 0.509 GiB; no amount of variance qualification
  can promote this row without a new implementation.
- [x] **P7.2 Screen B4/B5 only if justified.** The current B3 result is already
  below Vulkan B4 and target verify owns 94.92% of B3 decode wall; prior complete
  transaction and one-prompt B4/B5 admission evidence rejected the larger
  budgets. Do not reopen them without a materially new verifier schedule.
- [x] **P7.3 Compare against both external winners.** Comparison is complete and
  blocked: B3 beats HIP B2 **72.887 vs 46.863 tok/s (+55.53%)** and every HIP
  category floor, but trails Vulkan B4 **81.952 tok/s (-11.06%)**. It beats
  Vulkan only for `general_ja` (**70.898 vs 67.384**); code, general-English,
  mixed, and heldout scopes do not pass.
- [x] **P7.4 Adjudicate the MTP memory floor.** The gate is measured and fails:
  tracked peak is **16.684 GiB** and whole-device peak delta is **17.183 GiB**,
  **0.509 GiB** above Vulkan's 16.673-GiB floor. Ownership returns to zero and
  the mapped token table is borrowed without a raw shadow. Even converting all
  288 Q4 tensors to raw could save at most 0.2716 GiB, so that rejected rung
  cannot close the measured gap.
- [x] **P7.5 Profile if still behind.** The complete wall attributes **3.126 /
  3.293 s (94.92%)** to target verify, **0.123 s (3.74%)** to proposal, and
  **0.044 s (1.34%)** to commit/host residual. A warmed direct B3 child then
  captured seven exact target cycles: 267.172 / 293.496 ms (91.03%) of the
  target ROCTX wall is kernel time. Dense Q4 T16 owns 113.777 ms (38.77% of
  wall), Q6 qmicro-planar 80.002 ms (27.26%), and Q5 T16 19.961 ms (6.80%).
  The resulting actual-weight Q4 rung found that repeated one-weight wins were
  cache artifacts: cache-cold raw pools lost **12.53-65.73%**, fused raw lost
  **29.03-34.56%**, and exact T16 tile4 lost **9.60%**. No Q4 verifier candidate
  was integrated; profiling remains non-topline and the remaining Vulkan gap is
  an explicit blocker rather than an unmeasured Q4 proposal.

Current exit status: compatible B3 is exact and faster than llama.cpp HIP, but
Vulkan speed and memory gates fail. The final repeated-suite variance rerun is
intentionally skipped under the stop rule because it cannot reverse either
binding failure. Evidence:
[`llama-compatible natural MTP matrix`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-llama-compatible-mtp.json),
[`warmed B3 target-verify profile`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-b3-target-verify-profile.json),
[`rejected raw-Q4 rung`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-raw-q4-rung-rejected.json).

### P8 — W7900 non-regression and default promotion

- [x] **P8.1 Re-run current W7900 controls.** One discarded warmup plus three
  measured PM4 resets at 512/128, 1024/128, and 4096/128 compare package-default
  sole T16 against a same-commit diagnostic rollback of only
  `GGUF_DENSE_Q4_T16`. Every row is exact with zero native fallback and clean
  teardown.
- [x] **P8.2 Re-run W7900 natural MTP.** Both routes cover the complete ten-prompt
  suite, all four categories, six train/four heldout prompts, true AR and B1-B3,
  240 transitions/mode, exact output/acceptance ledgers, complete-wall timing,
  and whole-card memory.
- [x] **P8.3 Apply the frozen paired gate.** Single-layout improves same-commit
  AR prefill by **152.61-184.82%**, AR decode by **17.58-18.74%**, true AR by
  **2.61%**, and B1/B2/B3 by **2.47%/0.16%/0.24%**. It cuts whole-device peak
  delta by **45.50-47.03%**. The lower fresh true-AR absolute row versus the
  historical 22.926 tok/s is protocol/code drift, not a layout regression: the
  same-commit dual-layout control is slower at 19.993 versus 20.516 tok/s.
- [x] **P8.4 Promote one gfx1100 default.** XTX and W7900 both use the same
  package-default single-layout registry route; the old dual layout is retained
  only as an out-of-tree diagnostic wrapper for this evidence, not a runtime
  selector or default.

Exit evidence:
[`same-commit W7900 non-regression`](../benchmarks/results/2026-08-12-qwen36-27b-w7900-single-layout-non-regression.json).
Capacity is not purchased with a regression on the original target card.

### P9 — Stability, fragmentation, and transport lifecycle

- [x] **P9.1 Cold lifecycle.** Three independent AR processes (512/1K/4K) and
  three production-equivalent natural AR+B1-B3 MTP processes all return tracked
  bytes/allocations to zero. Every final sysfs sample is within 1.7 MiB of its
  pre-run baseline (64-MiB gate), with exact tokens and GPU/CPU acceptance.
- [x] **P9.2 Warm lifecycle.** One resident process completes 100 deterministic
  mixed reset/rearms (90x512, 5x1K, 5x4K) and 400 PM4 submissions. Tokens are
  9707 throughout, each shape has one final-logit value, post-warm tracked
  ownership is constant, and fallback/unretired counts stay zero. Session close
  returns tracked ownership to zero; process exit returns whole-card VRAM
  exactly to baseline.
- [x] **P9.3 Graph/PM4 recreate.** Three position-specific PM4 generations plus
  the earlier HIP-graph control cover capture, replay, policy fallback, and
  teardown. After session close every graph is closed, every executable is
  retired, context/native child counts are zero, callback status is zero, and
  there are no unretired submissions. Approximately 0.8-0.9 GiB of HIP/JIT
  module residency remains until interpreter exit, then sysfs returns exactly
  to baseline; tracked weights are already zero. The destructive
  submit+queue/resource-recreate ROCm#6529 arm remains separately guarded and
  requires explicit reset-risk approval. Evidence:
  [`cold/warm PM4 lifecycle`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-lifecycle.json).
- [x] **P9.4 MTP rollback stress.** The dense B1-B3 transaction gate covers
  reject/partial/full acceptance, correction, forced and post-commit rollback,
  reseed, changing admitted budget/graph shapes, dynamic positions, repeated
  graph/provider operation, and exact state/KV journals. Proposal-time
  cancellation stops before target mutation and closes poisoned owners; a real
  public lifecycle then proves two normal requests reuse the same target/provider
  and final teardown returns tracked ownership to zero.
- [x] **P9.5 Soak.** A fixed 601.083-second public-route interval completes 204
  cycles / 408 alternating AR+MTP requests over all ten prompts and four
  categories. Every pair is exact with one output hash per prompt, one reused
  target/provider, zero live-byte spread, no torch, and tracked close zero.
  Peak VRAM delta is 18.209 GiB; maximum edge/junction/memory temperatures are
  62/83/94 C, maximum sampled power is 363 W, and final VRAM is +827,392 bytes.
  Evidence: [`public AR/MTP soak`](../benchmarks/results/2026-08-12-qwen36-27b-xtx-public-ar-mtp-soak.json).

Exit: fit is repeatable, not a lucky fresh-process allocation.

### P10 — Publication and cleanup

- [x] **P10.1 Publish compact artifacts.** The evidence index includes the old
  admission blocker, clean llama HIP/Vulkan floors, sole-T16 candidate,
  residency/correctness, final AR/MTP matrices, rejected raw rung, profiles,
  PM4 all-context comparison, lifecycle/soak, and W7900 safeguard.
- [x] **P10.2 Update rollups for measured claims.** The canonical benchmark
  README, compact root export, and dated changelog publish the current W7900
  rows and the XTX partial/blocked outcome without promoting failed MTP.
- [x] **P10.3 Update architecture/process docs.** This punchlist, the superseded
  W7900 campaign status, kernel path map, and refactor triggers are synchronized.
  `PLAN.md` is unchanged because no architectural invariant moved.
- [x] **P10.4 Delete rejected residue.** Raw-Q4 and tile4 screens made no runtime
  integration and left no body/key/selector/test residue. The old dual-layout
  route is absent from production; its one out-of-tree benchmark wrapper is not
  a package surface. Required registered numerical fallbacks remain.
- [x] **P10.5 Final clean-tree validation and atomic commit.** Documentation,
  JSON, worklog, README-sync, focused test, and staged-diff checks complete in
  the closing logical unit.

Exit: the public topline accurately states same-card XTX speed and memory versus
both clean llama.cpp backends, with no hidden duplicate representation.

---

## 8. Acceptance scorecard

Populate only from committed artifacts:

| Metric | llama.cpp HIP XTX | llama.cpp Vulkan XTX | hipEngine XTX | Gate |
| --- | ---: | ---: | ---: | --- |
| 512 prefill tok/s | 964.606 | 870.872 | **965.209 current** | >=974.252 (1% margin) — fail |
| 512 AR transition tok/s | 33.025 | 13.391 | **33.569 current** | >=33.356 (1% margin) — **pass** |
| 512 peak VRAM delta GiB | 16.348 | **15.690** | **16.095** | <=15.690 — fail |
| 1024 prefill tok/s | 981.040 | 836.898 | **1003.206 current** | >=990.850 (1% margin) — **pass** |
| 1024 AR transition tok/s | 32.924 | 13.379 | **34.506 current** | >=33.254 (1% margin) — **pass** |
| 1024 peak VRAM delta GiB | 16.373 | **15.700** | **16.320** | <=15.700 — fail |
| 4096 prefill tok/s | 946.733 | 835.765 | **983.082 current** | >=956.201 (1% margin) — **pass** |
| 4096 AR transition tok/s | 32.560 | 13.309 | **31.366 current** | >=32.886 (1% margin) — fail |
| 4096 peak VRAM delta GiB | 16.562 | **15.912** | **17.119** | <=15.912 — fail |
| Natural true AR tok/s | 31.576 | 13.386 | **20.782** | disclosed same protocol |
| Selected MTP budget | B2 | B4 | **B3** | independently selected |
| Natural MTP transition tok/s | 46.863 | **81.952** | **72.887** | >=82.771 (1% margin) — fail |
| Natural MTP / true AR | 1.4841x | 6.1223x | **3.5071x** | >1.0; absolute gate still binds |
| Natural MTP peak VRAM delta GiB | 16.940 | **16.673** | **17.183** | <=16.673 — fail |
| Alternate-layout weight bytes | not audited | not audited | **0 plan + live physical bytes** | exactly 0 |
| Duplicate logical weight bytes | not audited | not audited | **0 live duplicate-payload bytes** | exactly 0 |
| Minimum free VRAM at measured 512 peak | 7.636 GiB | 8.294 GiB | **7.829 GiB** | hipEngine >=1.0 GiB |
| Tracked bytes after close | n/a | n/a | **0** | exactly 0 |
| Cold/warm/transport lifecycle | server teardown clean | server teardown clean | **3 AR + 3 MTP cold passes; 100 mixed resets / 400 PM4 submits exact; 3 generations retire cleanly; dense rollback/cancel/public reuse pass; 601-s / 408-request soak exact** | **pass** |

The current prefill cells are the strictly serial selector-unset
one-warmup/three-measurement production matrix after promoting the ordered
pair-only Q6-QKV/Q4-gate route on top of the exact Q4/Q5/Q6 producer and
consumer package. Binding M512/M1024 full prefill improves XTX
**+0.801%/+0.571%** and W7900 **+0.447%/+0.738%**, with **25/28** admitted
pairs winning, exact trajectories, and byte-identical tracked peaks. Only the
24 mixed Q6-QKV/Q4-gate layers change; standalone Q4 gates and all 24 Q4/Q4
unequal pairs remain exact. M2048/4K retains the exact T16 owner. Production
tracing observes **432 pair / 0 scalar** Q4 producer launches at M512, exactly
72 more than the prior package. Current XTX is **0.063%/2.259%/3.839% above**
llama.cpp HIP at 512/1K/4K; relative to the frozen HIP+1% gates, 512 misses by
only **0.928%**, while 1K/4K pass by **1.247%/2.811%**. Decode, natural
verifier/MTP, peer backends, and all shape/model misses retain exact prior
owners
([artifact](../benchmarks/results/2026-08-13-qwen36-27b-q6-qkv-q4-gate-pair-only-engine-retained.json)).

The earlier zero-workspace hipBLASLt route remains rejected: best-of-32
FFN-gate source-F16 is **0.691x/1.051x/1.183x** retained exact T16 at
M512/1K/4K, M512 loses **0/7**, and the live handle consumes **172 MiB** outside
the tracked arena
([artifact](../benchmarks/results/2026-08-13-qwen36-27b-q4-f16-hipblaslt-prefill-rejected.json)).

W7900 safeguard:

| Metric | Fresh duplicated control | Single-layout candidate | Gate |
| --- | ---: | ---: | --- |
| 512 prefill/decode | 265.323 / 23.955 | **670.227 / 28.444** | **pass: +152.61% / +18.74%** |
| 1024 prefill/decode | 258.497 / 24.491 | **714.771 / 28.988** | **pass: +176.51% / +18.36%** |
| 4096 prefill/decode | 244.983 / 22.442 | **697.749 / 26.388** | **pass: +184.82% / +17.58%** |
| Natural true AR / selected B3 | 19.993 / 60.732 | **20.516 / 60.875** | **pass: +2.61% / +0.24%** |
| Natural peak delta | 31.680 GiB | **17.183 GiB** | **pass: -45.76%** |
| Alternate-layout weight bytes | 13.037-GiB pack8 payload plus T16 sidecars | **0** | **pass: candidate exactly 0** |

---

## 9. Stop/reopen rules

Do not declare closure merely because the model fits. Continue until all scorecard
gates pass or publish an explicit blocker with the failed column and measured
ceiling.

Stop a candidate immediately when it:

- allocates a second persistent representation;
- fails layout/CPU/model/MTP correctness;
- wins a microbenchmark but loses its complete requested shape beyond the
  frozen noise rule;
- exceeds either comparator's lower memory floor;
- depends on a prompt, token, category, candidate ID, or arbitrary favorable
  layer subset;
- requires a model/engine backend-or-quant branch rather than a registered
  variant.

Reopen a rejected layout only for a materially new mechanism: a new
operation-complete sole representation, producer-fused transient transform,
new compiler/driver/source behavior, or a reconciled profile showing enough
complete-wall ceiling. “It fits if we keep both on W7900” is not a reopen
condition.
