# gfx1151 / Strix Halo Tuning Plan

Last updated: 2026-06-15

This is the tuning and validation playbook for native Strix Halo / `gfx1151`
runs. It exists because gfx1151 is not just a smaller W7900. Strix Halo is an
APU with unified memory, different bandwidth/latency behavior, different launch
and cache tradeoffs, and a different ROCm/TheRock package lane. W7900/gfx1100
kernel evidence is useful lineage, but every promoted gfx1151 default needs a
same-device correctness and performance gate.

Scope:

- hipEngine PARO packed decode/prefill on `hip_gfx1151`;
- hipEngine GGUF `UD-Q4_K_M` / `UD-Q4_K_S` decode/prefill on `hip_gfx1151`;
- MTP on PARO+MTP-BF16 versus llama.cpp HIP/Vulkan draft-MTP;
- DFlash / DDTree verifier, accept, commit, and drafter paths;
- gfx1151-specific HIP JIT/profile settings, memory/GTT assumptions, and artifact
  policy.

Non-goals:

- Do not replace W7900/gfx1100 tuning docs. Cross-link them, but keep gfx1151
  evidence separate.
- Do not promote a W7900 flag or kernel shape on gfx1151 without local evidence.
- Do not use llama.cpp memory counters on this APU as model-footprint evidence;
  the sysfs VRAM aperture is raw-only/noisy. Use hipEngine tracked allocator
  peaks for hipEngine memory.

## Evidence To Start From

### Lesson 0 — The Original gfx1151 Win Was Shape/Chunking

The first gfx1151 pass found a large, relatively free win by profiling before
changing math. The 2026-05-17 prefill gap diagnosis compared hipEngine shisa
PARO packed 4K/128 (`630.585` prefill tok/s, `63.364` decode tok/s) against
upstream llama.cpp HIP GGUF (`1004.220` prefill tok/s, `49.379` decode tok/s).
A 4K/1 `rocprofv3` run showed the gap was real GPU work (`6.439 s` kernel time)
and **not** full-attention-bound:

| Kernel group | Time | Calls | Share |
| --- | ---: | ---: | ---: |
| linear GDN recurrent K2 | 1.568 s | 30 | 24.3% |
| linear conv prefill lowp | 0.945 s | 30 | 14.7% |
| PARO rotate1 | 0.939 s | 190 | 14.6% |
| grouped MoE selected dual/down WMMA | 0.821 s | 80 | 12.8% |
| AWQ prefill dual/single | 0.834 s | 170 | 12.9% |
| linear prepare | 0.319 s | 30 | 5.0% |
| AOTriton full attention | 0.157 s | 10 | 2.4% |

Manual 256-row chunks on every prefill surface then moved 4K/1 from `635.884`
to `1029.808` prefill tok/s. The profile explained why: linear conv dropped
`0.945 -> 0.091 s`, rotate dropped `0.939 -> 0.181 s`, and GDN recurrent dropped
`1.568 -> 0.931 s`; MoE/AWQ/attention time rose from extra launches, but the
linear-attention/rotate savings dominated.

The retained all256 diagnostic sweep stayed simple and robust:

| Workload | all256 prefill tok/s | all256 decode tok/s | Note |
| --- | ---: | ---: | --- |
| 512/128 | 983.206 | 62.060 | still below upstream HIP prefill, above HIP decode |
| 4K/128 | 1029.402 | 63.605 | fixed the original 4K prefill gap |
| 32K/128 | 792.296 | 50.629 | above upstream HIP prefill/decode |
| 128K/128 | 413.489 | 30.245 | above upstream HIP prefill, slightly below HIP decode |
| 4K/4K | 1001.266 | 62.438 | retained decode win |

Follow-up chunk probes are just as important as the win: global chunks below 256
were worse (`all128=935.871`, `all64=716.446` at 4K/1), and global larger chunks
were not robust. `linear512` was a possible +~1% mid/long-context prefill tweak,
but it lost at 512 and was not promoted. The correct lesson is **not** “always
make chunks smaller”; it is: gfx1151 needs same-device row-shape profiling. On
this Strix Halo APU, memory bandwidth/cache behavior can dominate even when the
GPU has comparatively generous compute, so fewer/larger launches are not
automatically better than cache-fitting row chunks.

Implications for this MTP/DFlash pass:

- Start from profile buckets, not W7900 assumptions.
- Check GDN/linear-attention, rotate, selected-MoE, sampler/logit, and host gaps
  before attention micro-tuning.
- Re-test row/budget shapes on gfx1151. W7900’s B=1 MTP operating point may be
  too launch/verify-heavy for Strix Halo if B=2/B=3 amortize scarce memory traffic
  better.
- Keep all256 as the prefill baseline unless a repeated same-suite sweep proves a
  better gfx1151-specific setting.

### W7900 MTP Lessons To Reuse Carefully

The W7900/gfx1100 MTP sprint did produce a real retained D32 win, but the winning
shape was the result of many exact/no-hold gates. Current `docs/MTP.md` records
the retained D32 best as `1.023x` AR with `14.134 ms/cycle`, using B=1,
`chain_attn_mode=decode_batched`, graph mode off, draft vocab cap `65536`, and
verify canonicalization skip.

Retained levers worth retesting on gfx1151:

- fixed B=1 operating point after B=3 became too expensive;
- graph-off verifier host cleanup and post-verify canonicalize skip;
- `decode_batched` full-attention verifier mode;
- specialized proposer router top-k/softmax and route-batched proposer experts;
- verifier scratch/tensor/view caches and scratch generation stamps;
- safe small-B W4 verifier GEMV sites and reduced-DAG micro-fusions;
- draft vocab cap `65536` as a density/cost compromise;
- D64 exact fallback flags as correctness infrastructure, not speed defaults.

No-hold traps that should be treated skeptically on gfx1151 unless a fresh profile
says otherwise:

- `c1_loop` at the current B=1 operating point was exact but slower than
  `decode_batched` on W7900.
- Full-vocab draft LM-head recovered density but lost wall time.
- B=5/global larger budgets improved density but lost economics.
- Active-budget max-shape caps and online confidence gates were exact or partly
  exact but not speed rows.
- HIP graph capture for the current verifier/proposer shapes did not pay off.
- LM-head 256/512-thread retunes and several larger fused W4/rotate/router
  kernels regressed from spills, barriers, or larger kernel bodies.

The gfx1151 tuning rule is therefore: retest retained W7900 defaults first, but
also retest the **operating point**. Strix Halo’s compute:memory balance may make
B=2/B=3 verifier rows more attractive than they were on W7900, especially if
larger rows amortize host/readback/sampler overhead without exploding HBM traffic.

### llama.cpp Audit — What It Is Doing Differently

Source basis: local read-only `/home/lhl/llama.cpp/llama.cpp-hip` at commit
`6e9007ae61f4e994c27484759caac6ef2aa32b30` (`b9637-4-g6e9007ae6`). Use these as
implementation references, not as code to edit in-place:

- Qwen3.5/3.6 MoE loads explicit NextN tensors (`eh_proj`, `enorm`, `hnorm`,
  optional `embed_tokens` and shared head) and appends MTP layers after the trunk:
  [`qwen35moe.cpp#L135-L149`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/models/qwen35moe.cpp#L135-L149).
- The Qwen35MoE MTP graph is a separate `LLM_GRAPH_TYPE_DECODER_MTP` draft head:
  it consumes the target hidden row plus the next token embedding, runs the MTP
  block, emits `h_nextn`, then uses either the NextN shared head or the target LM
  head for logits:
  [`qwen35moe.cpp#L550-L738`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/models/qwen35moe.cpp#L550-L738).
- Server startup creates an `LLAMA_CONTEXT_TYPE_MTP` context against the target
  model. For MTP it measures/reserves only context+compute bytes, not another
  full model copy, because the draft context lives on the target model:
  [`server-context.cpp#L909-L1071`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/tools/server/server-context.cpp#L909-L1071).
- For hybrid Qwen35/Qwen35MoE, the MTP context uses a plain dense-attention KV
  cache and filters to MTP layers only:
  [`llama-model.cpp#L2047-L2051`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/llama-model.cpp#L2047-L2051),
  [`llama-model.cpp#L2144-L2146`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/llama-model.cpp#L2144-L2146).
- The target decode path can extract `h_nextn` rows from the target pass for the
  MTP drafter:
  [`llama-context.cpp#L1962-L1978`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/src/llama-context.cpp#L1962-L1978).
- The MTP speculative implementation keeps per-sequence pending hidden rows,
  supports backend-side top-k draft sampling, iterates draft generation up to
  `n_max`, and on accept copies the hidden row corresponding to the accepted
  length into the next pending seed:
  [`speculative.cpp#L816-L930`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L816-L930),
  [`speculative.cpp#L1048-L1184`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/speculative.cpp#L1048-L1184).
- Verification/acceptance is centralized: server verifies `draft + 1` rows on the
  target, `common_sampler_sample_and_accept_n()` accepts until the first mismatch,
  and the server records `draft_n` / `draft_n_accepted` metrics:
  [`server-context.cpp#L3520-L3605`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/tools/server/server-context.cpp#L3520-L3605),
  [`sampling.cpp#L621-L648`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/common/sampling.cpp#L621-L648),
  [`server-task.cpp#L643-L646`](https://github.com/ggerganov/llama.cpp/blob/6e9007ae61f4e994c27484759caac6ef2aa32b30/tools/server/server-task.cpp#L643-L646).

Observed result delta on the MTP-bearing `UD-Q4_K_M` file:

| Engine / mode | Decode tok/s | Speedup | Accepted/output |
| --- | ---: | ---: | ---: |
| hipEngine PARO+MTP B1 exact fallback | 59.56 | 0.912x vs AR | 0.360 |
| llama.cpp HIP B4 | 91.11 | 1.790x vs llama HIP base | 0.743 |
| llama.cpp Vulkan B4 | 108.96 | 1.733x vs llama Vulkan base | 0.747 |

Do **not** conclude from this table alone that llama.cpp has better HIP kernels.
It is a mixed algorithm/model/runtime delta: llama.cpp uses the MTP-bearing GGUF
with integrated NextN tensors; hipEngine currently uses PARO packed trunk plus a
BF16 MTP sidecar and exact fallback flags. Prompt rendering, draft budgets,
sampling, quantization, and model identity must be aligned before assigning cause.

Concrete audit questions for the gfx1151 pass:

1. **Acceptance density:** reproduce per-prompt `draft_n`/`draft_n_accepted` for
   llama.cpp B1-B4 and compare to hipEngine accepted/output on the same rendered
   prompts and token budget. If llama wins mainly by density, prioritize model
   identity, draft vocab, prompt rendering, and B=2/B=3/B=5 policy before kernel
   tuning.
2. **Verifier economics:** profile llama.cpp HIP B4 and hipEngine B1/B2/B3 exact
   rows into proposal/draft, target verify, sampler/readback, accept/rollback,
   and host buckets. If llama verifies B4 rows cheaply, test whether hipEngine’s
   W7900 B=1 default is the wrong gfx1151 operating point.
3. **Sampler/logit movement:** llama.cpp can attach backend sampler chains to the
   draft context and copy sampled/candidate outputs instead of always using full
   host-side logits. Measure hipEngine LM-head/logit/readback cost on gfx1151,
   especially with cap `65536` versus full vocab.
4. **MTP context shape:** llama.cpp’s MTP context is filtered to the NextN layer
   and plain dense-attention KV for Qwen35 hybrid models. Compare this to our
   sidecar/proposer/update path: are we paying extra trunk, recurrent-state, or
   exact-suffix costs that llama avoids?
5. **HIP vs Vulkan as a driver datapoint:** llama.cpp Vulkan is faster than its
   HIP backend on this APU in both base and MTP rows. Treat Vulkan as an external
   roofline/driver clue, not as a direct implementation target.

## Current Local Stack

Reference setup lives in [`THEROCK.md`](THEROCK.md). Current local gfx1151 stack:

- Hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`.
- ROCm/TheRock: HIP `7.13.60980-c76140fa27` from
  `/home/lhl/miniforge3/envs/therock`.
- TheRock package lane: `https://rocm.nightlies.amd.com/v2/gfx1151/`.
- PyTorch/ROCm install is pinned to one ROCm nightly tag; do not use floating
  `rocm[libraries,devel]` for this host.
- For hipEngine JIT/profiling, set:

```bash
PYSDK=/home/lhl/miniforge3/envs/therock/bin/python
ROOT=$($PYSDK -m rocm_sdk path --root)
SITE=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages

export PATH="/home/lhl/miniforge3/envs/therock/bin:$ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ROOT/lib:$ROOT/lib64:$ROOT/lib/llvm/lib:$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_libraries_gfx1151/lib:${LD_LIBRARY_PATH:-}"
export HIP_PATH="$ROOT"
export ROCM_PATH="$ROOT"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export HIPENGINE_HIP_ARCH=gfx1151
```

Before any retained profile, write a compiler-version file and use cached builds
inside profiled processes:

```bash
hipcc --version > /tmp/hipengine-gfx1151-hipcc-version.txt
```

When using a detached worktree, run `git lfs install --local && git lfs pull` in
that worktree before importing hipEngine; otherwise vendored AOTriton images may
be checked out as LFS pointer files.

## Current Baselines To Reproduce First

These are diagnostic retained rows, not final performance claims. They are the
starting point for a gfx1151 tuning pass.

### README Rationalization Sweep

Artifact:
[`2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json`](../benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json)

One measured run over `512/128`, `1K/128`, `4K/128`, `32K/128`, `64K/128`, and
`128K/128`.

| Workload | hipEngine PARO | hipEngine GGUF UD-Q4_K_M | llama.cpp HIP UD-Q4_K_M | llama.cpp Vulkan UD-Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 956.67 / 66.97 | 833.37 / 56.58 | 1016.70 / 51.64 | 1043.21 / 62.43 |
| 1K/128 | 1067.18 / 61.77 | 854.31 / 52.83 | 1069.68 / 51.45 | 1055.05 / 61.57 |
| 4K/128 | 1062.25 / 62.91 | 729.12 / 53.64 | 1021.19 / 49.58 | 1027.07 / 60.01 |
| 32K/128 | 822.25 / 50.37 | 619.57 / 44.38 | 742.87 / 43.63 | 809.62 / 50.91 |
| 64K/128 | 622.75 / 41.97 | 522.87 / 37.74 | 569.61 / 38.60 | 658.40 / 44.01 |
| 128K/128 | 425.73 / 30.29 | 384.01 / 28.04 | 384.96 / 31.60 | 473.65 / 34.71 |

hipEngine tracked peaks: PARO `21.248 GiB`; GGUF `26.264 GiB`.

### MTP D32 Suite

Artifact:
[`2026-06-15-gfx1151-mtp-compare-20260615-060801-summary.json`](../benchmarks/results/2026-06-15-gfx1151-mtp-compare-20260615-060801-summary.json)

The llama.cpp comparison uses the MTP-bearing Unsloth GGUF:
[`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf).
This file has `753` tensors and `20` `blk.40` / `nextn` MTP tensors. The older
non-MTP Q4_K_M GGUF had the same basename but only `733` tensors and cannot run
llama.cpp draft-MTP.

| Engine / mode | Mean decode tok/s | Speedup | Accepted/output | Notes |
| --- | ---: | ---: | ---: | --- |
| hipEngine PARO AR | 65.37 | 1.000x | — | same-session AR baseline |
| hipEngine PARO+MTP B1 exact fallback | 59.56 | 0.912x | 0.360 | exact 9/9; fallback flags required |
| llama.cpp HIP base | 50.90 | 1.000x | 0.000 | UD-Q4_K_M MTP GGUF, f16 KV |
| llama.cpp HIP B4 | 91.11 | 1.790x | 0.743 | best HIP row in B1-B4 sweep |
| llama.cpp Vulkan base | 62.87 | 1.000x | 0.000 | UD-Q4_K_M MTP GGUF, f16 KV |
| llama.cpp Vulkan B4 | 108.96 | 1.733x | 0.747 | best Vulkan row in B1-B4 sweep |

Important: W7900/gfx1100 retained MTP is above AR on D32, but this gfx1151 row is
below AR even after exact fallbacks. That makes MTP the first gfx1151-specific
tuning lane.

### DFlash / DDTree Status

DFlash has gfx1151 correctness and rocprof smoke coverage in
[`KERNELS.md`](KERNELS.md) and implementation context in [`DFLASH.md`](DFLASH.md):

- `dflash_accept_chain_i32` and `dflash_commit_chain_i32` smoke on gfx1151;
- drafter root/query kernels, tiny decoder block, native top-k, and GQA smoke on
  gfx1151;
- context K/V materializer smoke on gfx1151;
- full-model DFlash/DDTree rows are still diagnostic and slower than AR.

DFlash should get its own gfx1151 tuning pass because it stresses exactly the
same verifier/commit infrastructure as MTP, but with different row topology and
drafter cost.

## Baseline Reproduction Protocol

Before changing kernels or flags, reproduce the current baseline in a clean
worktree and save artifacts.

1. Check repo and hardware state:

```bash
git status -sb
rocminfo | grep -E 'Name:|gfx'
hipcc --version > /tmp/hipengine-gfx1151-hipcc-version.txt
```

2. Run the shortest health smoke:

```bash
PYTHONPATH=. python3 - <<'PY'
import ctypes
ctypes.CDLL('libamdhip64.so')
print('hip OK')
PY
```

3. Confirm model identity:

```bash
python3 scripts/inspect_gguf.py /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf | head -40
```

For MTP, require `753` tensors and `20` MTP-like tensors. If not present, stop
and replace/download the MTP-bearing GGUF before comparing llama.cpp draft-MTP.

4. Reproduce one-run diagnostics first. Only move to 3-run retained rows after a
hypothesis improves or fixes something.

## Tuning Lanes

### Lane A — MTP Break-Even On gfx1151

Current blocker: exact B1 fallback is slower than AR (`0.912x`) on gfx1151 even
though the W7900 retained path is above break-even. The exact fallback stack is:

```bash
HIPENGINE_GDN_TLOOP_C1_EXACT=1
HIPENGINE_LINEAR_OUT_C1_EXACT_ROWS=1
HIPENGINE_MTP_DECODE_BATCHED_FULL_ATTN_EXACT_SUFFIX=1
```

First questions:

1. Is the fast path wrong because of gfx1151-only numeric/order behavior, or is
   the current public PARO-packed + copied BF16 sidecar intrinsically different
   from the older exact W7900 artifact?
2. How much wall time is recoverable from the exact fallback components on
   gfx1151?
3. Is the D32 verifier bottleneck the same as W7900 (`verify_ms/cycle`) or more
   launch/occupancy sensitive?
4. Is llama.cpp ahead mainly because of acceptance density (`~0.74` accepted per
   output at B4), because its MTP context/verifier is cheaper, or because the
   model/prompt/runtime is not yet apples-to-apples?
5. Given Strix Halo's compute:memory balance, does B=2/B=3 amortize scarce memory
   movement and host/logit overhead better than the W7900-retained B=1 row?

Initial experiments:

- Re-run D32 prompt suite with each exact fallback disabled one at a time and
  record the first failing prompt/token plus per-cycle wall deltas.
- Run `scripts/mtp_verifier_rocprof.py` rather than wrapping the parent prompt
  suite in `rocprofv3`.
- Bucket cycle wall into proposal/update, verifier, accept/commit, sampler/logit
  movement, and host overhead for B=1, B=2, and B=3. Test B=5 only if the smaller
  rows show exactness and a density/cost trend.
- Capture per-prompt acceptance traces for hipEngine B=1/B=2/B=3 and llama.cpp
  HIP/Vulkan B=1-B4 on the same D32 prompt set. Normalize by output token count,
  not just by generated draft count.
- Compare `chain_attn_mode=decode_batched` vs `c1_loop` with exact fallbacks on
  gfx1151. Keep exactness mandatory.
- Re-test draft vocab cap (`32768`, `65536`, no cap) because gfx1151 bandwidth
  and cache behavior may shift the W7900 optimum.
- Measure LM-head/logit/readback cost directly; llama.cpp's MTP path can use
  backend draft sampling and may avoid the same host-side full-logit movement we
  pay in hipEngine.
- Verify whether graph replay is still negative/neutral on gfx1151 after cached
  builds and fixed shapes, but do not lead with graph capture unless a profile
  shows a host gap large enough to matter.

Acceptance criteria:

- Correctness: exact AR match for all 9 D32 prompts. Longer-horizon D64 remains
  a separate correctness lane; do not promote a D32 speed row if it regresses a
  required D64 fallback.
- Performance: keep any exact same-suite improvement. Promote only after a 3-run
  confirmation beats the current gfx1151 B1 exact-fallback baseline by a real
  margin or crosses `>=1.0x` vs same-session AR.
- Evidence: JSON artifact, exact command, environment, model identity, fallback
  flags, and profiler bucket if claiming root cause.

### Lane B — DFlash / DDTree gfx1151 Pass

DFlash should not inherit W7900 decisions blindly. The APU memory system may make
state reuse, warm scratch, and smaller verifier rows relatively more valuable.

Initial experiments:

- Re-run the current DFlash chain and tree smokes on gfx1151 with cached builds:

```bash
HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_accept_chain_smoke.py \
  --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt \
  --require-cached-build --debug-top1-readback

HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_commit_chain_smoke.py \
  --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt \
  --require-cached-build

HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/dflash_drafter_root_query_smoke.py \
  --compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt \
  --require-cached-build
```

- Reproduce full-model DFlash chain baseline with same-session AR and fixed
  prompt/decode length. Keep this as a diagnostic until exactness and speed are
  both non-regressive.
- Profile verifier row kernels separately from drafter kernels; avoid profiling
  parent harnesses that launch nested Python children.
- Revisit warm scratch and no-barrier variants on gfx1151: W7900 had partial
  verifier wins that still did not clear AR; gfx1151 may have a different
  balance.
- Test `tree_mode=chain_as_tree` overhead on gfx1151 before real DDTree policy
  work. The tree kernel must be near-chain cost before branching policy matters.

Acceptance criteria:

- GPU accept summaries match CPU oracle.
- Commit path does not leak rejected suffix rows into state/KV/output rings.
- Same-session AR baseline included in every throughput artifact.
- DFlash/DDTree row remains diagnostic unless exact, finite, and faster than AR
  or explicitly retained as infrastructure with a blocker note.

### Lane C — Prefill/Decode Shape Retuning

The initial gfx1151 pass found simple chunk-size wins. Re-run these after MTP and
GGUF changes because dispatch mix and model files changed.

Candidate knobs:

- PARO prefill chunk sizes: linear, MoE, full-attention query/post/rope chunks.
- AOTriton threshold for full-attention prefill.
- Decode graph replay on/off and graph capture shape keys.
- PARO/GGUF native sampler route on/off for c=1.
- GGUF WMMA/tile knobs that were W7900-specific, especially selected-MoE and Q8
  shape heuristics.

Acceptance criteria:

- Start with one-run diagnostics, then keep only same-suite improvements with no
  final-token instability.
- If a win is exact and non-regressive, make it default for `hip_gfx1151` via
  registry/config separation, not an `if gfx1151` branch in engine code.
- Document rejected knobs in [`REFACTOR.md`](REFACTOR.md) or this file.

### Lane D — Memory/GTT And Long Context

Strix Halo shares system memory. The default GTT/TTM cap may be lower than
installed RAM, and raw amdgpu aperture counters do not reflect model footprint.

Required for long-context claims:

- host RAM total and GTT/TTM configuration;
- hipEngine tracked allocator peak;
- model/quant and KV dtype;
- workload shape and exact command;
- whether the run used `UD-Q4_K_M` MTP GGUF, non-MTP GGUF, or PARO packed.

## Profiling Rules

- Prebuild JIT `.so` files outside profiled windows.
- Use `--compiler-version-file /tmp/hipengine-gfx1151-hipcc-version.txt` and
  `--require-cached-build` in profiled processes.
- Use `scripts/mtp_verifier_rocprof.py` or a final child process for MTP; do not
  wrap `scripts/mtp-bench.py` / economics parent harnesses in `rocprofv3`.
- Record kernel names, counts, and duration buckets. If a kernel is not the one
  expected, check dispatch/registry/fusion before editing device code.
- For every kernel-level claim, include `DurationNs` or `End_Timestamp -
  Start_Timestamp`, scratch/LDS/VGPR where available, and the command used.

## Artifact Policy

Every retained or rejected gfx1151 tuning attempt needs a compact artifact under
`benchmarks/results/` with:

- `status`: `diagnostic_retained`, `accepted`, `rejected`, or `blocked`;
- `performance_claim`: `false` until it has enough repetitions/correctness;
- hardware + ROCm + hipEngine commit + dirty state;
- model source and local path;
- exact command and environment flags;
- correctness gate result;
- baseline and new numbers with delta;
- reason for keep/reject/block.

If a row is promoted into README-facing tables, also update:

- [`benchmarks/README.md`](../benchmarks/README.md)
- [`benchmarks/CHANGELOG.md`](../benchmarks/CHANGELOG.md)
- [`WORKLOG.md`](../WORKLOG.md)

## Proposed First Pass

1. Reproduce the current MTP-bearing Q4_K_M D32 artifact from a clean worktree and
   confirm the GGUF identity (`753` tensors, `20` NextN tensors).
2. Build a llama.cpp comparison ledger, not just a throughput table:
   - run HIP and Vulkan B1-B4 on the same D32 prompts;
   - capture `draft_n`, `draft_n_accepted`, per-prompt speed, and prompt text /
     rendered-token hashes where possible;
   - keep source revision, command line, server flags, KV dtype, and model path in
     the artifact.
3. Build a hipEngine budget/fallback ledger:
   - B=1/B=2/B=3 with all exact fallbacks on;
   - each exact fallback disabled one at a time with first-failure logging;
   - `c1_loop` exact fallback;
   - `decode_batched` fast path with first-failure logging;
   - draft vocab caps `32768`, `65536`, and full vocab only after the exact rows
     establish a cost baseline.
4. Profile the best exact hipEngine rows with `scripts/mtp_verifier_rocprof.py`
   and split verifier/proposal/update/sampler/readback/host time. Profile one
   llama.cpp HIP B4 prompt or short prompt set separately to understand whether
   its B4 verifier economics are genuinely cheaper on gfx1151.
5. Re-run DFlash accept/commit/drafter smokes with cached builds and capture a
   small rocprof kernel table for gfx1151.
6. Reproduce one full-model DFlash chain diagnostic with same-session AR.
7. Only after that, start kernel tuning. The first likely targets are verifier
   exact fallback overhead, proposer/update launch count, sampler/logit movement,
   B=2/B=3 row economics, and DFlash warm-scratch reuse.

## Open Questions

- Can the public PARO packed trunk + copied BF16 MTP sidecar be made fast-path
  exact on gfx1151, or do we need the older exact MTP artifact for speed work?
- Is gfx1151 MTP below AR because of exact fallback overhead, lower memory
  bandwidth, launch latency, or a different acceptance/cycle-cost balance?
- Does llama.cpp's high accepted/output on the MTP-bearing Q4_K_M file reflect a
  better draft model, prompt rendering differences, or quant/model differences?
- Is DFlash more promising than MTP on gfx1151 because the APU may benefit more
  from state reuse and smaller row materialization?
- Which gfx1151 defaults should eventually live as backend-specific registry
  variants rather than environment flags?
