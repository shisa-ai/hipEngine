# Qwen3.6-27B Q4_K_M on RX 7900 XTX: Single-Layout Campaign

Status: **in progress; P0 comparator and blocker evidence complete.**

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

- [ ] **P1.1 Add a plan-time byte census.** For every weight spec, report source
  bytes, planned allocations, canonical layout, aliases, and alternate-layout
  bytes without requiring a GPU.
- [ ] **P1.2 Add a runtime physical-range census.** Deduplicate by actual
  `(device, ptr, nbytes)` range and attach owner/view names. Report target,
  NextN, root-shared, state/KV, graph, code/backend, and workspace classes.
- [ ] **P1.3 Add a duplicate invariant checker.** Fail if one logical source
  tensor owns multiple physical payloads or if a payload appears under an
  undeclared logical owner.
- [ ] **P1.4 Cover aliases.** Test tied and untied embedding/head, shared target
  assets, model-map aliases, arena slices, and intentional views so the checker
  neither double-counts aliases nor hides disjoint allocations.
- [ ] **P1.5 Freeze current and target manifests.** Current 288-Q4 pack8/T16
  duplication must be reproduced exactly; the candidate target manifest must
  predict zero alternate-layout bytes before GPU implementation begins.

Exit: memory ownership is a testable contract, not inferred from peak deltas.

### P2 — RED: define the sole-Q4 representation

- [ ] **P2.1 Flip the mapping expectation.** Replace the current test that
  requires 288 Q4 sidecars with RED assertions that each Q4 spec owns exactly
  one allocation family.
- [ ] **P2.2 Add actual-weight layout oracles.** Cover all nine Q4 roles and
  representative/tail shapes for source->T16/raw->dequant equivalence, BF16
  output bits where the current path is exact, and finite F32 accumulation.
- [ ] **P2.3 Add operation coverage REDs.** Require c1, production verifier
  rows 2-4, conditional row-5/6 gates before B4/B5 admission, M512/M1024/M4096,
  fused dual+SiLU, single projection, and unfused fallback dispatch from one
  quant/layout key.
- [ ] **P2.4 Add no-lazy-shadow REDs.** Capture allocator history through load,
  warmup, prefill, graph capture, 128 decode steps, and MTP; fail on any second
  weight payload or steady-state weight allocation.
- [ ] **P2.5 Predeclare candidate order and keep rule.** Sole T16 is first. Raw
  GGUF is opened only if T16 misses the external memory/perf gate or a profile
  identifies a direct-source advantage. No favorable role subset may salvage a
  dual-layout package.

Exit: the current implementation fails the new tests for the intended reason.

### P3 — Sole-T16 materialization and same-layout fallbacks

- [ ] **P3.1 Change rank-2 Q4 planning.** Under the candidate selector,
  materialize only `gguf_q4_k_t16_v1`; do not build/upload pack8 first.
- [ ] **P3.2 Keep host loading bounded.** Repack from mmap/host source directly
  to one upload buffer, release temporary arrays promptly, and record host RSS
  separately. Device admission is based on the final manifest plus worst-case
  workspace/KV/graph reserve.
- [ ] **P3.3 Route c1 decode and fused FFN.** Preserve current exact T16
  single/dual/fused owners and their row policy without pack8 fallback.
- [ ] **P3.4 Route verifier rows.** Cover every production B1-B3 row shape; add
  rows 5/6 only if B4/B5 is separately admitted.
- [ ] **P3.5 Implement dense T16 bulk prefill.** Reuse or adapt registered T16
  WMMA/MMQ primitives for M512/M1024/M4096 and tails. Activation quantization is
  bounded reusable scratch; weights remain one payload.
- [ ] **P3.6 Implement same-layout unfused fallbacks.** Unsupported fusion or
  graph capture uses T16 primitive chains, never pack8. Fail planning if no
  semantically equivalent T16 chain exists.
- [ ] **P3.7 Reconcile registry keys.** No direct backend/quant branch enters
  engine/model dispatch; architecture role mapping selects registered variants.

Exit: plan/runtime census reports zero rank-2 Q4 alternate-layout bytes and an
isolated layer stack passes decode, verifier, and three-shape prefill.

### P4 — Remove the remaining AR/MTP duplicate assets

- [ ] **P4.1 Audit Q8 llama-compatible overrides.** Eliminate T16+raw
  coexistence. Implement missing Q8 operations on one canonical format or keep
  raw as the sole format for that session.
- [ ] **P4.2 Audit Q6 root/NextN X8.** Eliminate T16/qmicro+X8 coexistence in
  root top-1 and proposal paths. Prefer the retained sole planar/T16 payload and
  consume it directly.
- [ ] **P4.3 Share target-owned immutable roots.** NextN/provider must alias the
  target's embedding/output/norm assets where semantically identical. The
  model's untied `output.weight` remains distinct from `token_embd.weight`, but
  neither may be copied merely because both AR and draft use it.
- [ ] **P4.4 Keep only NextN-specific weights additional.** Enumerate the
  trailing `blk.64` tensors and prove MTP adds only truly distinct NextN bytes,
  its state/KV, and bounded cycle scratch.
- [ ] **P4.5 Audit graph and cached uploads.** `_cached_upload` keys, shared
  runners, warmups, graph slabs, and PM4 manifests must not retain stale model
  payloads or duplicate a target after ownership transfer.
- [ ] **P4.6 Make the invariant package-default.** Candidate selectors may
  remain for rollback only until final promotion; add removal criteria to
  `REFACTOR.md`.

Exit: complete AR and MTP manifests both report zero duplicate/alternate-layout
weight bytes.

### P5 — Correctness and transaction gate

- [ ] **P5.1 CPU/layout bundle.** Run mapping/materialization, quant round-trip,
  registry/fusion, and allocator-audit tests.
- [ ] **P5.2 Kernel correctness.** Every new/ported body passes CPU-reference,
  representative/tail shapes, launch smoke, and a named cached rocprof trace.
- [ ] **P5.3 Full eager/graph AR state.** At 512, 1024, and 4096, compare eager
  and production graph/transport logits, hidden, Conv/GDN, KV+`KVLiveSpans`,
  positions, final IDs, and teardown.
- [ ] **P5.4 Dense MTP transaction.** Cover reject, partial accept, full accept,
  correction, rollback/reseed, dynamic positions, proposal/target graph reuse,
  and provider-vs-scalar behavior for every admitted budget.
- [ ] **P5.5 Natural semantic gate.** Run all ten prompts and fixed heldouts.
  Layout-only changes are expected to preserve exact current hipEngine output;
  any changed-association/quality-gated route is a separate explicitly approved
  lane and still requires KL <= 0.05 / top-1 >= 90%.
- [ ] **P5.6 Torch-free and lifecycle.** Public `LLM.generate()` and server paths
  remain torch-free, finite, deterministic under greedy settings, and leak-free.

Exit: correctness is green before any keep is based on performance.

### P6 — XTX fit and performance optimization

- [ ] **P6.1 First complete XTX fit.** Run 512/1, 1024/1, 4096/1, then the
  512/128, 1024/128, 4096/128 matrix. Record tracked and whole-device peaks.
- [ ] **P6.2 Compare sole T16 to frozen floors.** If it beats both backends and
  meets lower memory peak, skip raw-layout work.
- [ ] **P6.3 Profile only the failing column.** Use cached direct children and
  reconcile kernel sum plus host/submission residual to wall. Rank complete-wall
  impact; do not optimize merely because a llama.cpp intrinsic exists.
- [ ] **P6.4 Raw-GGUF rung if needed.** Port/audit MMVQ for c1 and MMQ for
  prefill/verifier one family at a time, with exact source commit, RED fixture,
  registry key, same-layout fallback, and actual-weight component evidence.
- [ ] **P6.5 No hybrid salvage.** A raw win for one role and T16 win for another
  may define different canonical formats by tensor role only if each logical
  tensor still has one payload and the rule is architecture/shape based. Never
  retain both formats for one tensor or tune a favorable arbitrary layer set.
- [ ] **P6.6 Re-run complete shapes after each structural keep.** Exact small
  cycle-wall wins are retained under project policy, but closure still requires
  every requested shape and both external backends.

Exit: XTX AR beats HIP and Vulkan at all three shapes with memory no higher
than the lower comparator.

### P7 — Compatible natural MTP closure

- [ ] **P7.1 Run hipEngine B1-B3 full suite.** Select by complete transition
  tok/s vs true AR, with all category/heldout ledgers.
- [ ] **P7.2 Screen B4/B5 only if justified.** Complete transaction first, then
  a predeclared one-prompt wall ceiling, then one full suite if admitted. Do not
  use fixed prompts to search for a favorable selection.
- [ ] **P7.3 Compare against both external winners.** Beat the selected clean
  HIP and Vulkan absolute transition tok/s, true-AR ratio, and every disclosed
  category speed floor; preserve acceptance and exact state semantics.
- [ ] **P7.4 Meet the MTP memory floor.** Whole-device peak delta is no greater
  than the lower external winner, zero weight duplication is proven, and MTP
  incremental bytes reconcile to NextN-specific weights/state/KV/scratch.
- [ ] **P7.5 Profile if still behind.** Slice proposal/verify/commit markers in a
  direct child under rocprofv3. Profile external backends only with separate
  non-topline instrumentation. Optimize the current complete-wall leader.

Exit: compatible natural MTP is faster than both clean llama.cpp backends and
uses no more memory than the lower-memory comparison row.

### P8 — W7900 non-regression and default promotion

- [ ] **P8.1 Re-run current W7900 controls.** 512/128 and 4096/128 use one
  discarded warmup plus at least three measured resets; add 1024/128 as the new
  campaign midpoint.
- [ ] **P8.2 Re-run W7900 natural MTP.** Preserve full category/heldout output,
  acceptance, state, and complete-wall semantics.
- [ ] **P8.3 Apply the frozen paired gate.** Initial published controls are
  235.434/23.296 at 512 and 216.784/21.897 at 4096, with natural true AR 22.926
  and production B3 61.147 tok/s. Compare against freshly measured same-commit
  controls, not only these historical medians.
- [ ] **P8.4 Promote one gfx1100 default.** XTX and W7900 use the same
  single-layout registry route. Keep an opt-out only for concrete rollback
  value and give it a deletion milestone.

Exit: capacity is not purchased with a regression on the original target card.

### P9 — Stability, fragmentation, and transport lifecycle

- [ ] **P9.1 Cold lifecycle.** At least three complete process load/run/close
  cycles for AR and MTP return tracked bytes to zero and sysfs VRAM to within
  64 MiB of pre-run baseline after a bounded settle interval.
- [ ] **P9.2 Warm lifecycle.** One resident process performs at least 100 mixed
  reset/rearm runs across 512/1024/4096 without rising current/peak ownership,
  stale pointer use, or output drift.
- [ ] **P9.3 Graph/PM4 recreate.** Exercise package-default transport plus HIP
  graph control across capture, replay, resource recreation, fallback, and
  teardown. Memory pressure must not reopen the guarded recreate/lifecycle
  issue or retain old weight ranges.
- [ ] **P9.4 MTP rollback stress.** Cycle accepted/rejected patterns, budget
  changes among admitted graphs, cancellation, and repeated provider reuse;
  check all state/KV journals and allocation ownership.
- [ ] **P9.5 Soak.** Run a predeclared c=1 AR/MTP mixed workload long enough to
  observe thermals and allocator stability. Record clocks, temperatures,
  errors, max VRAM, and throughput drift. Get explicit approval before this
  >5-minute run.

Exit: fit is repeatable, not a lucky fresh-process allocation.

### P10 — Publication and cleanup

- [ ] **P10.1 Publish compact artifacts.** Include blocked old layout, clean
  llama HIP/Vulkan baselines, residency manifests, retained candidate, final
  XTX matrix, MTP suite, W7900 non-regression, profiles, and lifecycle evidence.
- [ ] **P10.2 Update rollups for measured claims.** Refresh
  `benchmarks/README.md` `Last updated` and rows, add a dated
  `benchmarks/CHANGELOG.md` old->new line with deltas/reason/artifact, and sync
  root README exports.
- [ ] **P10.3 Update architecture/process docs.** Mark this punchlist with exact
  evidence links, update the W7900 campaign cross-link/status, `KERNELS.md` path
  map for new kernels, `PLAN.md` only if architecture moved, and `REFACTOR.md`
  for every temporary route.
- [ ] **P10.4 Delete rejected residue.** Remove losing bodies, wrappers,
  registry keys, selectors, tests that only support rejected code, and generated
  caches. Do not leave a second default or fallback shadow.
- [ ] **P10.5 Final clean-tree validation and atomic commit.** Run the applicable
  focused/full gates under `TESTING.md`, worklog validation, JSON validation,
  README sync check, staged diff inspection, and commit each logical unit
  immediately after it passes.

Exit: the public topline accurately states same-card XTX speed and memory versus
both clean llama.cpp backends, with no hidden duplicate representation.

---

## 8. Acceptance scorecard

Populate only from committed artifacts:

| Metric | llama.cpp HIP XTX | llama.cpp Vulkan XTX | hipEngine XTX | Gate |
| --- | ---: | ---: | ---: | --- |
| 512 prefill tok/s | 964.606 | 870.872 | TBD | >=974.252 (1% margin) |
| 512 AR transition tok/s | 33.025 | 13.391 | TBD | >=33.356 (1% margin) |
| 512 peak VRAM delta GiB | 16.348 | **15.690** | TBD | <=15.690 |
| 1024 prefill tok/s | 981.040 | 836.898 | TBD | >=990.850 (1% margin) |
| 1024 AR transition tok/s | 32.924 | 13.379 | TBD | >=33.254 (1% margin) |
| 1024 peak VRAM delta GiB | 16.373 | **15.700** | TBD | <=15.700 |
| 4096 prefill tok/s | 946.733 | 835.765 | TBD | >=956.201 (1% margin) |
| 4096 AR transition tok/s | 32.560 | 13.309 | TBD | >=32.886 (1% margin) |
| 4096 peak VRAM delta GiB | 16.562 | **15.912** | TBD | <=15.912 |
| Natural true AR tok/s | 31.576 | 13.386 | TBD | disclosed same protocol |
| Selected MTP budget | B2 | B4 | TBD | independently selected |
| Natural MTP transition tok/s | 46.863 | **81.952** | TBD | >=82.771 (1% margin) |
| Natural MTP / true AR | 1.4841x | 6.1223x | TBD | >1.0; absolute gate still binds |
| Natural MTP peak VRAM delta GiB | 16.940 | **16.673** | TBD | <=16.673 |
| Alternate-layout weight bytes | not audited | not audited | TBD | exactly 0 |
| Duplicate logical weight bytes | not audited | not audited | TBD | exactly 0 |
| Minimum free VRAM at selected MTP peak | 6.957 GiB | 7.251 GiB | TBD | hipEngine >=1.0 GiB |
| Tracked bytes after close | n/a | n/a | TBD | exactly 0 |
| Cold/warm/transport lifecycle | server teardown clean | server teardown clean | TBD | all pass |

W7900 safeguard:

| Metric | Fresh duplicated control | Single-layout candidate | Gate |
| --- | ---: | ---: | --- |
| 512 prefill/decode | TBD | TBD | frozen paired non-regression |
| 1024 prefill/decode | TBD | TBD | frozen paired non-regression |
| 4096 prefill/decode | TBD | TBD | frozen paired non-regression |
| Natural true AR / selected MTP | TBD | TBD | correctness + paired non-regression |
| Alternate-layout weight bytes | current >0 | TBD | candidate exactly 0 |

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
