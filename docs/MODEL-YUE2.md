# MODEL-YUE2.md — YuE2 music generation on hipEngine

Status: **M0-M3 complete (oracle fixtures and closure, protocol/loader contracts,
HIP AR runtime, native generation session); no NAR, VAE, or product-path claim
yet.**
Reviewed 2026-09-16 on branch `yue2`.

Implement `m-a-p/YuE2-3B` plus `m-a-p/YuE2-Vae` as a torch-free HIP pipeline:
lyrics/style → optional symbolic composition → semantic tokens → acoustic flow
matching → 48 kHz stereo PCM. Use the DeepSeek shootout's final strict path as
an implementation reference, with the official implementation as the behavioral
oracle. Reimplement its computation in hipEngine; a torch/Triton wrapper does
not satisfy this plan.

Initial measurement target: desktop `gfx1151`, with `gfx1100` compatibility by
design and separate hardware qualification. Neither backend is qualified yet.
The first objective is complete, correct generation; the next is a measured
same-host improvement over the torch baseline. No speedup magnitude or realtime
claim is promised before profiling the complete pipeline.

## Sources and frozen baseline

Public references: [model card][model], [model configuration][config],
[VAE card][vae], [official GitHub][github], [generation guide][generation],
and [cover guide][covers]. The public package has moved to 0.1.6; the shootout
used 0.1.5. Do not silently combine their implementations or fixture identities.
The reviewed GitHub HEAD was `4d53bd5fc7e96a53cb907d3eb407a65df67a8b79`;
the guide links below pin that revision. Milestone 0 must inspect the changes
from the wheel before deciding whether to establish a newer oracle.

| Input | Baseline identity |
| --- | --- |
| YuE2-3B checkpoint | `29b3558dd46954a0cd9021dc76d5c91864a0f1c7` |
| YuE2-Vae checkpoint | `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a` |
| `yue2_infer-0.1.5-py3-none-any.whl` SHA-256 | `8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641` |
| Local shootout export | `~/yue2-shootout/` at `87e1271fe8a658471e58c3e3d8869765eb82e895` |
| DeepSeek final original candidate | `08887599f17b3a2ccc5909c1ec498b934bb1bf5d` |
| GLM final original candidate, optional ideas only | `149981d1216747950cf9de738ccb7b05f2b0c4b8` |

The export preserves source and evidence but omits weights, environments, and
binary generation artifacts. Its JSON records cannot substitute for regenerating
fixtures. Keep this external repository read-only. Cite original source paths
and commits in ports; do not vendor contestant trees, traces, or private logs.
Record upstream code/model licenses and required notices with the loader/export
inventory; source-code licensing and checkpoint licensing are separate.

Read these local files in order:

1. `README.md`, `ANALYSIS.md`, `proctor/FINAL_REVIEW.md`: what was actually checked.
2. `shared/upstream/yue2/{protocol,sampling,modeling_yue2,nar,modeling_vae,pipeline}.py`:
   source behavior for the pinned wheel.
3. `contestants/deepseek/src/yue2/{tuned,triton_ops}.py`: strict AR implementation,
   compact projection, stable caches, and graph mechanics.
4. `proctor/independent_ar.py`, `proctor/audit_final_audio.py`: independent checks.
5. `OPTIMIZE.md` and the specifically named GLM functions: candidate experiments.

The shootout independently checked 22 AR cases / 6,694 comparisons. DeepSeek
matched the reference exactly on that matrix and its 12 regenerated production
cases matched saved official tokens, latents, and audio. Those cases are
**English/Mandarin × off/full/melody × seeds 1234/5678**, not a complete CFG
factorial. Independent timing was waived; contestant CUDA timings on RTX PRO
6000 are not hipEngine results or AMD speedup predictions.

## Model geometry and computation

These details come from the pinned local source, especially `modeling_yue2.py`,
`protocol.py`, `nar.py`, and `modeling_vae.py`. Verify them against loaded tensors
in milestone 0; the model name alone is not a full parameter/memory inventory.

| Surface | Required behavior |
| --- | --- |
| Backbone | 28 layers; hidden 2048; FFN 6144; vocabulary 184704 |
| Attention | 16 query / 8 KV heads × 128; per-head Q/K RMSNorm; RoPE theta 1,000,000 |
| Normalization | RMS epsilon `1e-6`; reference uses FP32 square/mean, then casts inverse RMS before multiplying BF16 activations/weights |
| Context | 24576 positions; reject over-budget requests, never silently truncate a prefix |
| Mixture of Transformers | Separate AR and NAR attention projections, MLPs, and layer norms in every layer; not routed MoE |
| AR phases | Optional ABC planning, then semantic codec-token generation; causal attention |
| NAR | Cached AR conditioning plus bidirectional NAR attention; 64-dimensional acoustic state |
| Flow solver | Midpoint integration, default 32 steps / 64 velocity evaluations per chunk |
| Latent conditioning | `vae2llm`, sinusoidal timestep MLP, fixed sinusoidal audio-position buffer; `llm2vae` output |
| VAE | Separate FP32 Oobleck/SnakeBeta decoder, stereo, stride product 1920 |
| Audio | 48000 Hz; nominal 25 latent frames/s; natural decoder length `1920*T - 64` samples |

This is not an ASR Qwen2 runtime with a replacement head. Existing GEMM, GEMV,
RoPE, norm, attention, arena, and quant primitives are candidates for reuse;
Q/K normalization, rounding order, masks, and the dual transformer paths require
explicit validation. The VibeVoice decoder and DPM solver are different networks
and algorithms, not substitutes for this VAE and midpoint solver.

### Exact prompt, token, and sampling protocol

| Token/domain | ID or range |
| --- | --- |
| EOD | 151643 |
| ABC start / end | 151847 / 151848 |
| Music start / end | 151851 / 151852 |
| Semantic codec vocabulary | 151853–184620, inclusive; raw codes 0–32767 |
| Latent start / end / pad | 184621 / 184622 / 184623 |
| Planning output support | Ordinary IDs `[0,151643)` plus ABC end |
| Semantic output support | Codec IDs plus music end |

Freeze native instruction strings, tokenizer assets, special-token treatment,
and assembled IDs for every mode. `off` constructs the empty ABC delimiters;
`melody` and `full` generate ABC or accept an external score. An unchanged saved
plan preserves its exact IDs; edited ABC is a new request. External transcription
such as SheetSage2 is outside this implementation: accepting its ABC is in scope.

ABC planning has no CFG. Semantic CFG defaults to 1.0 for symbolic modes and
1.01 for `off`; any non-unit scale needs a negative branch. Its prefix excludes
style/lyrics but includes the same positive ABC IDs in symbolic modes. Branch
prefix lengths and positions differ. Preserve reference BF16 subtraction,
multiplication, and addition in CFG before any FP32 conversion.

Sampling order is part of the interface: phase mask → minimum-length EOS mask →
frequency-based repetition penalty → zero-temperature argmax or temperature →
top-k threshold → top-p → sampling. Preserve sign-dependent penalties, cutoff
ties, real vocabulary IDs, and the different `off` arithmetic: legacy `off`
keeps BF16 scores and retains at least three sorted candidates for top-p;
symbolic modes use FP32 and retain at least one. EOS is not appended to content
history. Maximum-budget exit must report truncation distinctly from EOS.

Pin effective defaults, not just the seed: semantic `(temperature=1, top_p=.95,
top_k=100, penalty=1.2, window=50, min=200, max=9000)`; ABC
`(.7, .9, 30, 1.005, 100, 32, 4096)`. The pinned request validator permits
repetition windows 1–100; unsupported values must fail explicitly.

A seed does not guarantee identical CUDA/HIP/CPU multinomial or normal draws.
Use recorded random operands for cross-backend arithmetic isolation. Separately
specify a versioned torch-free, request-local RNG and test its reproducibility,
stream resets, and distributions. The reference resets the AR seed per phase;
NAR draws one full-song CPU FP32 noise array and slices it into chunks. Never
claim native seeded identity unless the actual RNG algorithms also match.

### NAR conditioning and solver

For a semantic sequence of `T` raw codec IDs, preserve upstream chunk boundaries:
`capacity = min((24576 - prefix_tokens - 3) // 2, 24576)`.
Each chunk's AR conditioning is its prefix, offset codec IDs, and music end.
NAR has `chunk_frames + 2` positions, including zero-state boundary rows. Confirm
`AR_length + NAR_length <= 24576` before allocation. Freeze non-default
`nar_cond_end` visibility behavior too; it changes which AR keys NAR can see.

Prefill the chunk's AR conditioning once and retain each layer's K/V. Every
velocity evaluation computes NAR Q/K/V with the NAR weights and attends to the
visible cached AR keys plus **all** current NAR keys. NAR positions for RoPE
start at `AR_length`; audio-position embedding indices start locally at zero.
Do not apply a causal mask to NAR, misalign rectangular causal attention in AR,
or materialize a context-squared attention mask as the production solution.

Let `h=1/steps`, `t=1-i*h`. The reference evaluates
`u=velocity(x, clamp(logit(t), -20, 20))`, then
`x = x - h * velocity(x - h*u/2, clamp(logit(t-h/2), -20, 20))`.
The velocity function applies the configured sigmoid/time shift in model dtype.
Record the FP64 host schedule, BF16 state/intermediate rounding, time embeddings,
first/midpoint velocities, and final FP32 latents. Precompute invariant schedules
and conditioning only after proving they are invariant for that chunk/config.

### VAE and long audio

Implement the actual pinned decoder: weight-normalized Conv1d/ConvTranspose1d,
dilated residual blocks, and SnakeBeta with exponential alpha/beta and `1e-9`
denominator epsilon. Inspect weight-normalization axes and fold weights once
with validated FP32 arithmetic. Decoder ratios are `[2,2,4,4,5,6]`; encoder
weights are inventoried but are not needed for the generation-only path.

Validate fixed latent → unclipped FP32 stereo PCM before connecting AR/NAR.
Match padding, transposed-convolution output lengths, channel order, and natural
end length. Tiled decode uses core 1024 / halo 16 by default, but validate the
halo against the decoder's dependency interval. Preserve exact interior crops;
no crossfade, invented padding, or concatenation of uncropped overlap. Exercise
one-frame input, tile boundaries, a partial last tile, and multiple tiles.
Stream completed cores to host to bound device residency. Distinguish tiled VAE
output from incremental music generation: AR and NAR can still delay first audio.
Default and legacy VAE checkpoints must never share an unlabeled quality result.

## hipEngine design and ownership

Resolve all hardware/quant/profile variants through the existing four-axis
registry. Add a `yue2` model/generator plugin, with an immutable resolved variant
manifest at session construction. Both backend trees remain peers. Use
`KVLiveSpans` for AR writes/attention and a documented span-based extension or
composition for NAR's cached-AR plus live-NAR attention; do not hide a private
`(block_table, context_len)` ABI in the new runner.

Proposed file layout (these files are deliverables, not existing APIs):

| Proposed path | Responsibility |
| --- | --- |
| `hipengine/models/yue2.py` | Config, model plugin, request/phase contracts |
| `hipengine/loading/yue2.py` | Safetensors inventory, tensor mapping, assets, owned weight handles |
| `hipengine/generation/yue2.py` | Pure protocol, sampling state, staged result types |
| `hipengine/runtime/yue2_session.py` | Lifecycle, stage transitions, cancellation, artifact identities |
| `hipengine/runtime/yue2_ar.py` | AR prefill/decode, independent CFG branches, head projection |
| `hipengine/runtime/yue2_nar.py` | Conditioning cache, hybrid attention, device midpoint solver |
| `hipengine/runtime/yue2_vae.py` | FP32 decoder, bounded tiled output |
| `hipengine/kernels/cpu_reference/yue2*.py` | Small NumPy reference operators |
| `hipengine/kernels/{hip_gfx1100,hip_gfx1151}/yue2/` | Only missing/justified specialized HIP kernels |
| `scripts/yue2_{fixtures,bench,quality}.py` | Separate oracle generation, timing, and task evaluation |

Start with a model-owned `YuE2Session` exposing `plan`, `generate_semantic`,
`synthesize`, and `decode`, plus an end-to-end call. Register orchestration rather
than putting model switches in the engine. Select any public `LLM` convenience
method during API integration; do not overload the existing TTS script contract.
Results carry ABC IDs/text, raw semantic codes, latent/audio identities, sample
rate, per-phase truncation, effective settings, timings, revisions, and manifest.
Saved stages must validate hashes, configuration compatibility, shapes, and token
domains on reload; tokens and tensors are data, not executable pickle objects.

Weights own allocations; sessions borrow with explicit lifetimes. Session-local
arenas, KV, sampler state, RNG, and graph buffers must not leak across requests.
Close is idempotent; cancellation/exception paths restore a usable session and
release graph references before freeing storage. Initially serialize requests
per session explicitly; independent sessions must isolate state. Concurrent
batching is a later feature, not implicit behavior of shared scratch.

Memory accounting must include both transformer paths, embedding/head, VAE,
conditioning K/V, state, scratch, graph pools, host staging, and duplicate head
weights. As a sizing estimate, BF16 AR K/V costs
`28 * 2 * 8 * 128 * 2 = 114688` bytes/token/branch, or 2.625 GiB at 24576
positions, before other allocations. CFG has two independent branches. This is
an allocation estimate, not a measured model footprint. Inventory all tensors;
do not infer total weights from the “3B” label or assume 96 GiB residency.

Choose an explicit memory policy from measured budgets: phase residency with
transfers counted, or full residency where it fits. Cache capacity and generation
budget are separate quantities. Use allocation probes before long generation
([capacity protocol](../benchmarks/HARNESSES.md)); warm loops should not allocate
per layer/token/solver step. GGUF/standalone packaging and quantization follow a
working safetensors baseline rather than blocking the first correctness gate.

## Correctness and evidence gates

[EXECUTION-PROFILES.md](EXECUTION-PROFILES.md) governs promotion. Control,
ownership, masking, token mapping, and request isolation are exact in every
profile. Strict arithmetic is a diagnostic oracle/fallback; non-bit-identical
candidates are evaluated under calibrated production numerical **and task**
gates, not automatically discarded or silently excused.

Before tuning, freeze YuE2-specific numerical thresholds and task margins using
reference repeatability, BF16-relative calibration, and separate calibration
inputs. Record mean/p95/p99/max KL and top-1 by phase/category/shape/transition
for logits; choose appropriate norm/max/RMS and spectral/time-domain metrics for
velocity, latent, and PCM tensors. KL/top-1 alone cannot certify audio. Do not
invent a “passing” audio threshold after seeing a candidate fail. Until these
thresholds exist, production promotion is blocked, not granted by a smoke test.
The shootout's max-absolute .0625 / RMS .002 / 100% per-comparison greedy limits
are historical CUDA diagnostics, not an automatic HIP production contract.

| Gate | Required coverage and failure behavior |
| --- | --- |
| Protocol / loader | All modes, supplied ABC, exact IDs, defaults, invalid input, missing/extra tensors, shape/dtype errors, assets and revision/hash mismatch |
| Operator arithmetic | Norm rounding, Q/K norm + split-half RoPE, residual/SiLU, masks/GQA, projected heads, conv/deconv/SnakeBeta; CPU oracle and pinned torch fixtures |
| AR numerical replay | Real phase tokens; prefixes 128/512/2048; 128-step replay plus 1024-step semantic cases; seeds 1234/5678; semantic CFG off/on with unequal prefixes; planning without CFG |
| Sampling | Branch and vocabulary axes intact; full real-ID distributions, mask/order/sign, ties, EOS/min/max, zero temperature, window rollover/reset; native RNG determinism and distribution checks |
| NAR | Mixed AR/NAR attention vs uncached reference, visibility limits, both boundary rows, all solver steps, chunk boundaries, full-song noise slicing, finite final latents |
| VAE | Fixed latents, full vs tiled, halo sufficiency, first/last samples, stereo/channel layout, unclipped PCM and file conversion separately |
| Lifecycle | Repeated/interleaved sessions, borrowed weight handles, close twice, cancellation at each stage, stale graph output/aliasing, reset and no residual state |
| End-to-end | Twelve production cases above plus held-out longer lyrics, repetitions, punctuation, styles, and CFG settings; natural completion vs truncation reported |

Fixtures include prefill and every branch's decode logits, actual combined CFG
scores, distributions, intermediate NAR states/velocities, and fixed-latent PCM.
Large model CPU execution is not required for every case: use small NumPy
operator fixtures plus pinned torch GPU end-to-end outputs. Run a compatible
ROCm torch oracle for same-host comparisons; preserve CUDA reference provenance
when importing fixtures. Never label CUDA-vs-HIP timings same-host AMD evidence.

The fixture validator must fail closed on missing cases/files, unexpected case
counts, stale hashes, NaNs, or altered metadata. Audit `semantic.npy`,
`abc_tokens.npy`, `latent.npy`, `audio.flac`, `request.json`, and `config.json`;
negative tests must deliberately corrupt/remove evidence and fail. The submitted
DeepSeek production verifier skips absent reference files; do not port that bug.

Task evaluation needs paired reference/candidate requests and seeds, English and
Mandarin lyric adherence, truncation/failure rate, structure/duration, artifacts,
and blinded listening for vocal/music quality. ASR WER/CER is supporting evidence,
not a music-quality oracle. Predeclare a practical non-inferiority margin and
report paired uncertainty, including failed outputs. A confidence interval that
includes zero does not establish equivalence. Record the listening VAE identity;
upstream leaderboard replication with the legacy VAE is a separate experiment.

## Performance plan: reuse the good parts, measure the rest

DeepSeek strict is the starting design: bounded KV, separate branch positions,
phase-specific head rows and stable decode buffers. Its retained torch mean and
CUDA GEMM choices must become validated HIP implementations. Preserve a full-head
fallback: planning needs 151644 rows, semantic needs 32769. Map projected rows
back into the full real-ID score domain before the reference sampler. DeepSeek's
`weight @ x.T` parity result is specific to its CUDA shapes, not proof for hipBLAS.

| Priority | Candidate / source | Decision gate |
| --- | --- | --- |
| Baseline | Existing hipEngine GEMM/GEMV/attention, reusable arenas; DeepSeek strict control flow | Validate actual YuE2 shapes and profile the complete stages before specialization |
| First isolated experiment | GLM `silu_mul`, `contestants/glm/src/yue2tune/kernels2.py` | Preserve required rounding/strides; CPU + numerical/task gates; measure complete MLP/AR step |
| Second | GLM `add_pow` from the same file | Square the already-rounded BF16 residual; validate reduction and scratch lifetime; account for the extra buffer |
| Next | GLM `PenaltyState` and `prep_scores` in `sampling.py` | Correct window counts/reset, no fixed-table overflow, explicit temperature-zero path, full-domain tie tests |
| Profile driven | HIP graph AR replay, then NAR velocity/solver replay | Show launch gaps; include capture cost, stable pointers, branch/capacity/config keys, and output ownership |
| Profile driven | Compact semantic head, invariant NAR conditioning/schedules, weight residency | Measure memory and whole-stage latency; no unsupported cache sharing or changed conditioning |
| Later | AR-only weight INT8/Q4, then independently NAR weight precision | Separate calibration/quant identity; production numerical and unassisted audio gates; VAE remains FP32 initially |

`OPTIMIZE.md`'s “keep torch mean” means preserve its numerical reference, not
retain torch on the hot path. Likewise an official torch fallback belongs only
in the oracle harness; runtime fallbacks are registered HIP/CPU implementations.
Check [KERNELS.md](KERNELS.md) and source lineage before actual kernel ports.
Fused variants retain registered unfused fallbacks. Temporary flags and duplicate
routes require entries in [REFACTOR.md](REFACTOR.md) with removal criteria.

Do not import GLM's `top_k + 64` candidate shortcut, compact-domain sampling,
fixed 129-entry penalty table without bounds, divergent fused-projection preset,
unqualified FP8 path, or its coverage verifier. Deferred EOS checks must not
silently consume extra random draws or violate cancellation/next-request state.
GraphNAR showed no meaningful device-time win in the source experiments; it is
not a mandatory optimization. Fewer flow steps and activation/KV precision changes
are separate quality experiments. Neither INT4 nor a smaller head promises a
particular end-to-end speedup before measurement.

### Matched benchmark protocol

Compare official torch, DeepSeek strict torch where ROCm-compatible, and hipEngine
on the **same physical device**, pinned checkpoints, effective requests, dtype,
CFG, solver steps, VAE, residency policy, and timing boundaries. Unsupported
DeepSeek CUDA paths must be named, not silently counted as custom HIP execution.
Use two complementary lanes: fixed tokens/noise/latents for work-matched stage
performance, and unassisted generation for task quality and user-visible RTF.
Injected replay is not proof of unassisted generation quality.

Measure loading, cold initialization/capture, planning prefill/decode, semantic
prefill/decode, NAR conditioning/solver, VAE, transfers, first playable audio, and
end-to-end wall time. Main warm timing runs avoid artificial stage fences;
separate synchronized diagnostic runs validate attribution against the total.
RTF is total inference wall time / actual generated PCM duration. Report token
counts, chunk counts, truncation and failures so shorter/broken songs cannot
manufacture a speedup. File encoding and preprocessing exclusions are explicit.

Start with three warmups and at least ten measured stage samples, raw ordering,
median/p95, and interleaved A/B arms on an idle host; use a predeclared multi-request
suite for expensive end-to-end runs. Report peak device/host memory and first vs
reused session behavior. Archive machine identity, clocks/power where available,
software/commit/input hashes, commands, manifest, and correctness gate. Recheck
small wins against session noise rather than combining unrelated best runs.
Keep measured non-regressive improvements even when they improve a substage but
not the noisy headline; follow the repository's evidence/rollup policy.

## Implementation punchlist and closure

Every row is initially **open**. Do not mark implementation complete after AR
logits alone. Dependencies allow fixed-latent VAE work once M0 exists; long-audio
qualification and production gates are required for full scope, not optional
postscript work after declaring completion.

| Milestone | Deliverables / exit condition |
| --- | --- |
| M0 — Reproducible oracle and inventory | Pin wheel/source/checkpoints/tokenizer/VAE/env; diff 0.1.6 against baseline; inventory every tensor/buffer and bytes by component; regenerate all 12 cases; freeze intermediate fixtures and fail-closed validator; independently repeat representative outputs; record actual oracle hardware |
| M1 — Protocol, loader, CPU contracts | Pure request/prompt/token/sampling references; validated safetensors mapping; owned weights; tiny operator oracles; all protocol/error tests pass without GPU or torch runtime imports |
| M2 — HIP AR | Both prefill and decode, exact CFG branch accounting, full head first then validated reduced head, `KVLiveSpans`; numerical replay matrix and kernel trace gates pass on initial hardware |
| M3 — Native generation | Torch-free sampler/RNG, ABC + semantic loops, EOS/truncation, staged save/reload, session isolation/cancellation; distribution gates and unassisted short requests pass |
| M4 — HIP NAR | Cached AR conditioning, bidirectional NAR attention, timestep/position features, device midpoint loop; per-step/chunk fixtures pass, including a request crossing a real chunk boundary |
| M5 — HIP VAE | FP32 decoder and bounded tiles; fixed-latent PCM parity/production gates; correct crop/length/halo and multi-tile output |
| M6 — Complete product path | Registered session/API, local/offline asset loading, all modes including supplied ABC, full 12-case + held-out task suite, validated production thresholds, repeated requests and lifecycle gates; no torch import reachable from generation |
| M7 — Optimize and compare | Profile then isolated candidates above; matched same-host multi-request baseline and optimized measurements; quality gates, artifacts, rollups, lineage/catalog, refactor ledger updated |
| M8 — Hardware qualification | Independently run applicable correctness/capacity/task/performance gates on gfx1151 and gfx1100; clearly retain unverified status for unavailable hardware |

Milestone status:

- **M1 complete** (2026-09-16): protocol, tokenizer, loader, model plugin, CPU
  operator contracts, unit tests without torch or a GPU. See
  `worklog/entries/20260916T142012.217732Z-lhl-yue2-m1-protocol-loader-b6a2b8.md`.
- **M2 complete** (2026-09-16): torch-free AR runtime with both prefill routes,
  decode, per-branch caches and positions, `KVLiveSpans`, and the 18-case replay
  matrix plus a kernel-trace smoke. Pooled over 112 recorded full-vocabulary rows
  and 576 decode-step rows against the pinned oracle on the same host: mean KL
  1.43e-3 (strict) / 2.11e-3 (batched hipBLASLt), top-1 95.83% / 96.70%, and the
  batched route takes the matrix's prefill from 522.5 s to 91.3 s (5.7x). The reduced head
  is not implemented, so the validated path uses the full lm_head. See
  `worklog/entries/20260916T144759.713427Z-lhl-yue2-m2-ar-runtime-bb886e.md` and
  `benchmarks/README.md` "Radeon 8060S: YuE2 3B AR replay".
- **M3 complete** (2026-09-16): torch-free staged session
  (`hipengine/runtime/yue2_session.py`) over the M2 runtime - ABC and semantic
  loops, per-phase request-local streams, CFG branch assembly, phase masks,
  EOS/budget truncation, cancellation, session reset, and JSON round-trips of the
  plan/result types with domain and prefix-embedding checks on reload. 22 CPU
  unit tests drive the loop against a deterministic fake runtime; the existing
  reference-score sampling fixtures cover the distribution gate.
  `scripts/yue2_session_gate.py` replays free greedy reference trajectories
  (`tests/fixtures/yue2/greedy/`, temperature zero, AR stages only, generated on
  the same host) through the session: prefix identity on all three cases, ABC greedy agreement 100.0% / 99.1% / no ABC stage, semantic greedy agreement under teacher forcing 96.9% / 97.9% / 96.1% with every mismatch on a near-tie, and matching truncation classes (natural ABC exit, budget-truncated semantic). See
  `worklog/entries/20260916T161024.244009Z-lhl-yue2-m3-native-generation-7786f9.md`.
- **M0 complete** (2026-09-16): oracle pinned, tensors inventoried by component,
  fixtures frozen behind a fail-closed validator, oracle environment recorded, all
  twelve production cases regenerated and committed
  (`tests/fixtures/yue2/cases/`), the 0.1.6 release reviewed against the pinned
  0.1.5 source (no oracle module changed, so the oracle stays pinned), and a
  production case independently re-run bit-for-bit against its committed fixture.
  See `worklog/entries/20260916T155756.081223Z-lhl-yue2-m0-oracle-closure-dd8799.md`.

The oracle fixtures were generated on gfx1151 (Ryzen AI MAX+ PRO 395 / Radeon
8060S) with torch 2.13.0+rocm10.0.0; `tests/fixtures/yue2/oracle_env.json` is the
committed record. Any same-host comparison must state the architecture it ran on.

Before closing each unit: focused RED/GREEN where practical, applicable profile
gate, HIP availability guards in GPU tests, worklog, and atomic commit. Name new
tests `test_unit_yue2_*` / `test_gpu_yue2_*` under the repository's discovery
rules. Milestone closure uses the full suite required by [TESTING.md](TESTING.md)
and the milestone-specific gates; follow focused repair rules for isolated failures.

**First coder assignment:** complete M0, then M1. Read the pinned source files,
recover/download weights outside Git, inventory them before residency decisions,
and produce a fixture manifest plus validator that demonstrably fails on missing
or corrupted evidence. Use the separate torch oracle environment; do not install
the shootout's CUDA lock into hipEngine's ROCm runtime. Record unresolved upstream
version differences and measured oracle repeatability before kernel development.
No optimization benchmark or public speed claim is needed to close this unit.

[model]: https://huggingface.co/m-a-p/YuE2-3B/tree/29b3558dd46954a0cd9021dc76d5c91864a0f1c7
[config]: https://huggingface.co/m-a-p/YuE2-3B/blob/29b3558dd46954a0cd9021dc76d5c91864a0f1c7/config.json
[vae]: https://huggingface.co/m-a-p/YuE2-Vae/tree/9a94e1d0ea9f8087e98f77fa88df4a4068104d2a
[github]: https://github.com/multimodal-art-projection/YuE/tree/4d53bd5fc7e96a53cb907d3eb407a65df67a8b79
[generation]: https://github.com/multimodal-art-projection/YuE/blob/4d53bd5fc7e96a53cb907d3eb407a65df67a8b79/docs/generation.md
[covers]: https://github.com/multimodal-art-projection/YuE/blob/4d53bd5fc7e96a53cb907d3eb407a65df67a8b79/docs/covers.md
