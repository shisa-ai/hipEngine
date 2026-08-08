# Moonshine — gfx1151 transfer campaign

Last updated: **2026-08-08** (`main`)

## Summary

hipEngine's Moonshine work targets the pinned
[`shisa-ai/shisa-realtime-asr-0.92b`](https://huggingface.co/shisa-ai/shisa-realtime-asr-0.92b)
checkpoint at revision `cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d`.
It is a compact encoder-decoder speech-recognition model with a convolutional
front end, eight encoder layers, eight autoregressive decoder layers, and
self/cross attention over hidden size 416. Source weights are F32; the resident
runtime converts or directly loads a packed FP16 deployment artifact.

The Radeon 8060S / `hip_gfx1151` path already owns a tuned FP16 decoder,
fixed-address caches, four HIP graph buckets, and complete hybrid routes using a
compiled-PyTorch or MIGraphX encoder. The newly merged `cuda_sm120a` campaign is
therefore **not** a kernel source replacement. Much of CUDA C1 was ported from
these HIP primitives and independently retuned for Blackwell. This campaign
transfers only new architecture and scheduling ideas that can pass gfx1151's
own exactness, lifecycle, profiler, and performance gates.

Moonshine is not yet listed in the root README's supported-model matrix. The
current CUDA and HIP paths are internal runtime/benchmark surfaces rather than a
fully admitted `LLM.generate()` audio API. Public admission is the final phase,
not an assumption of this campaign.

## Pinned model contract

| Field | Value |
| --- | --- |
| Model plugin | `moonshine_asr` / `MoonshineForConditionalGeneration` |
| Runtime quant | FP16 weights and activations, FP32 reductions/statistics |
| Encoder / decoder layers | 8 / 8 |
| Hidden / MLP width | 416 / 1,664 |
| Attention | 8 query heads / 8 KV heads, logical head dimension 52 |
| Partial RoPE | factor 0.62, rotary dimension 32, theta 10,000 |
| Vocabulary | 36,864, tied decoder embedding / output head |
| Decode capacity | 194 positions |
| Certified audio/frame buckets | 16,000/40, 80,000/207, 480,000/1,248 |
| Stored / runtime dtype | F32 checkpoint / FP16 resident payload |

The decoder keeps one fixed FP16 self-cache and one head-major encoder
cross-cache per layer. Encoder masks are int32. Greedy selection is stable:
lowest vocabulary ID wins a visible-logit tie.

## Current gfx1151 path

`hipengine/runtime/moonshine.py` composes the retained HIP primitives:

- tuned local32/64 projection and MLP boundaries;
- exact residual+LayerNorm and RoPE+self-cache composites;
- parallel masked cross attention;
- four exact self-attention schedules and graph regions: position `0`, position
  `1`, positions `2-3`, and positions `4-193`;
- wave8 tied FP16 LM-head projection followed by a separate stable argmax;
- explicit eager and primitive fallbacks;
- no tracked allocation inside a token step and complete teardown.

Historical clean gfx1151 evidence recorded in `WORKLOG.md`:

| Scope | Current retained result | Qualification |
| --- | ---: | --- |
| Cached position-1 graph token | **0.868 ms event** | one graph launch / 103 kernels, exact required tokens and state gates |
| Six-file decoder-only median | **5.446 ms** | encoder excluded, exact generated IDs |
| Compiled-PyTorch encoder + graph decoder | **9.220 ms** | 0.03-0.05-ms D2D handoff, exact six-file IDs |
| Selected MIGraphX encoder + graph decoder | **8.943 ms** | fastest measured complete HIP route, exact six-file IDs |

These rows predate the repository's current compact-artifact/rollup policy and
point to the historical external Moonshine ledger. They are transfer baselines,
not fresh public scoreboard rows. Any new retained number must be reproduced
from a clean main revision and committed under `benchmarks/results/`.

Selective W8A16 remains research-only. Its only quality-admissible component,
`mlp_fc1`, produced unstable sub-1% timing and regressed the broader Japanese
proxy-CER comparison. Production stays FP16.

## What the CUDA campaign added

The merged `cuda_sm120a` work extends beyond the original HIP decoder:

1. a torch-free fixed-address encoder plus independent CPU encoder oracle;
2. packed-FP16 deployment loading and content verification;
3. fused bounded LM-head/top-1 variants;
4. async encoder handoff, device-owned token/position state, and device result
   publication;
5. exact static-B encoder and decoder runtimes;
6. a decoder-side FIFO continuous scheduler with compaction and graph LRU;
7. long-bucket cuBLASLt, cuDNN, and AOT CUTLASS attention candidates;
8. benchmark/report helpers with raw samples, dependency bytes, and source
   manifests.

Blackwell measurements are directional evidence only. They do not establish a
Radeon speedup, launch geometry, numerical contract, or default.

## Transfer rules

### Directly reusable now

- `hipengine/core/runtime.py` and the backend-neutral device-memory owner;
- `hipengine/kernels/cpu_reference/moonshine_encoder.py` as an independent HIP
  encoder oracle;
- packed-FP16 loader validation and `scripts/pack_moonshine_fp16.py`;
- benchmark schema/provenance ideas, after Python 3.10 compatibility and compact
  artifact publication are repaired.

### Reimplement and remeasure on gfx1151

- fused wave8 LM-head plus stable top-1;
- async handoff and device-owned graph-tail state;
- static-B glue, attention, cache, and head kernels;
- continuous scheduling over gfx1151's four arithmetic regions;
- native encoder primitives;
- hipBLASLt long-row projections and AOTriton encoder attention.

### Do not copy

- CUDA thread/block choices or the two CUDA graph buckets;
- the uniform-t256 continuous topology without an independent gfx1151 gate;
- NVCC, cuBLASLt, cuDNN, or CUTLASS wrappers/source;
- Blackwell latency/speedup ratios as Radeon expectations;
- CUDA's tolerance-only encoder state as proof of byte-exact HIP state.

## Campaign phases

| Phase | Scope | Status | Promotion gate |
| --- | --- | --- | --- |
| **G0** | Main-promotion hygiene | Active | Python 3.10 import, registry reset, no-EOS result, docs/evidence defects repaired with focused tests |
| **G1** | Exact wave8 LM-head + stable top-1 | Next | full FP16 logits, selected token, hidden/KV state and lifecycle equal to current wave8+argmax; matched gfx1151 wall non-regressive; named cached trace |
| **G2** | Async handoff + device-owned decode | Planned | same six-file stream/state, zero per-step token/position H2D in selected graph route, zero ownership after close |
| **G3** | Static c2/c4/c8 decoder | Planned | each row bit-exact to independent c1, exact lockstep graph topology, mixed lengths/reclaim, no timed allocation |
| **G4** | Continuous decoder scheduling | Planned | full mixed-arrival lifecycle plus exact state; preserve four regions or independently qualify reassociation on the full Japanese corpus |
| **G5** | Native torch-free HIP encoder | Planned | CPU/HF oracle, six real files, broader Japanese corpus, fixed addresses, no timed allocation; beat or provide a concrete deployment advantage over selected MIGraphX |
| **G6** | Long-bucket encoder acceleration | Planned | hipBLASLt/AOTriton candidates only after G5; complete-token/state and per-bucket timing; MIOpen last because CUDA's analogous cuDNN route changed one synthetic token |
| **G7** | Public model admission | Planned | audio-to-transcript public API, packed deployment artifact, complete route benchmarks/artifacts/rollups, lifecycle and failure paths |

Phases are ordered by dependency, not by CUDA commit number. Exact same-suite
non-regressive wins become defaults. Rejected candidates are removed or retained
only as explicit fallback/bisection routes with a `docs/REFACTOR.md` trigger.

## G0 audited defects

The main-promotion audit found these blockers before public CUDA admission:

- four new Moonshine scripts import `datetime.UTC`, which is unavailable under
  the declared Python `>=3.10` floor;
- `load_backend_kernel_package("cuda_sm120a")` does not restore the 15 encoder
  registrations or AOT-attention registrations after registry isolation;
- CUDA `read_result_tokens()` returns the complete unwritten capacity when EOS
  is absent instead of the generated `self_cache_length` prefix;
- the continuous-runtime module header still calls uniform t256 rejected even
  though the corrected full-mask corpus gate qualified it;
- [`PLAN.md`](PLAN.md) and [`KERNELS.md`](KERNELS.md) stop at early CUDA
  bring-up and do not catalog C2-C8;
- no compact Moonshine result or benchmark-rollup row was promoted into this
  repository;
- continuous submission accepts precomputed host cross caches, so it is a
  decoder scheduler rather than complete continuous audio serving.

G0 fixes mechanical defects. It does not add Moonshine to the root support
matrix or manufacture missing CUDA performance evidence.

## G1 exact fused-head contract

The first gfx1151 implementation candidate adapts the CUDA idea to hipEngine's
stronger state contract:

- preserve the existing wave8 FP32 dot-product order and FP16 logit rounding;
- continue to materialize the complete `[1, 36_864]` FP16 logit plane so the
  existing full-logit KL/top-1 and byte-state gates remain meaningful;
- derive one stable block partial from the same just-rounded logits, then reduce
  those partials with lowest-ID tie breaking;
- use fixed caller-owned value/index scratch reserved before timed execution;
- retain separately registered `tied_wave8_fp32_accum` plus `lowest_id` argmax
  as the unfused fallback;
- select the candidate only after primitive, full-runtime, lifecycle, profiler,
  and repeated wall gates pass.

Current gfx1151 traces place separate argmax at **31.219 us** and historical
LM-head+argmax near 20% of cached position-1 kernel time. That is an Amdahl
admission estimate, not a promised speedup. The candidate keeps two device
kernels (wave8 producer/partial plus final partial reduction); its intended gain
is removal of the standalone global-logit scan, not launch-count gaming.

## Correctness and lifecycle gates

Every math/runtime phase must satisfy the applicable rules in
[`TESTING.md`](TESTING.md):

1. RED test before implementation, or an explicit rationale in `WORKLOG.md`.
2. Independent NumPy/CPU oracle at production geometry and edge shapes.
3. For G1, byte-exact full FP16 logits and exact stable token versus the current
   wave8+argmax chain, including ties and a non-multiple-of-eight vocabulary.
4. Complete runtime hidden/self-cache/cross-cache equivalence at retained
   positions; generated IDs exact through first EOS.
5. KL <= 0.05 and top-1 >= 90%; the campaign's G1 target is stricter: KL 0 and
   100% top-1 because the candidate preserves every FP16 logit bit.
6. Zero allocation in timed regions and tracked ownership returned to baseline
   after close, reset, partial failure, and repeated reuse.
7. `rocprofv3 --kernel-trace` from a prebuilt cached library naming the expected
   kernel with plausible duration and no compiler child.

For continuous or encoder arithmetic reassociation, six retained files are only
bring-up. Promotion also requires the corrected full 266-case Japanese FLEURS
corpus (or a committed successor with at least equivalent coverage), exact
normalized transcripts, and category/heldout reporting. No fixture-conditioned
branch is admissible.

## Performance and evidence protocol

Follow [`BENCHMARK.md`](BENCHMARK.md). A retained Moonshine artifact records:

- pinned model revision and checkpoint/packed-artifact hashes;
- Radeon 8060S / gfx1151, ROCm/driver/`hipcc`, queue policy, exact command, and
  clean source revision;
- timing scope: leaf, decoder-only, or complete encoder+autoregressive ASR;
- warmups, raw samples, median/P95/min/max, and baseline/candidate call order;
- full correctness and lifecycle results;
- expected kernel identity and compact profiler summary;
- acceptance/rejection decision and exact baseline.

Leaf timing cannot replace complete-route timing. Decoder-only timing cannot be
presented as full ASR. CUDA and HIP results remain separate hardware rows.
Every retained result updates `benchmarks/README.md`,
`benchmarks/CHANGELOG.md`, and a compact JSON under `benchmarks/results/`.

## Reproduction entry points

GPU health and focused current-path validation:

```bash
python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
rocminfo | grep -E 'Name:|gfx'
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
  uv run --python 3.12 --extra dev pytest -q \
  tests/test_cpu_reference_moonshine.py \
  tests/test_moonshine_hip_attention.py \
  tests/test_moonshine_hip_glue.py \
  tests/test_moonshine_hip_layernorm.py \
  tests/test_moonshine_hip_mlp.py \
  tests/test_moonshine_hip_projection.py \
  tests/test_moonshine_runtime.py
```

Prebuild and fixture gate pattern:

```bash
hipcc --version > /tmp/hipengine-moonshine-hipcc-version.txt
GPU_MAX_HW_QUEUES=1 HIPENGINE_HIP_ARCH=gfx1151 \
  uv run --python 3.12 python scripts/moonshine_decoder_smoke.py \
  --compiler-version-file /tmp/hipengine-moonshine-hipcc-version.txt \
  --require-cached-build --model-path /path/to/pinned/snapshot \
  --fixture /path/to/moonshine-fixture.npz --pad-to-certified-bucket \
  --token-route graph --json /tmp/moonshine-gfx1151-gate.json
```

The model-derived fixture bundle is not committed because it is large. A local
or CI gate must record its content hashes and producer identity; compact result
summaries belong in the repository, raw tensors and profiler CSVs do not.

## Current next action

Close G0, then implement G1 as a default-off exact route. Do not start static or
continuous batching until the fused-head decision is committed and its fallback
contract is stable.
