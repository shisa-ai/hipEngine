# MODEL-VIBEVOICE-TTS.md — VibeVoice 1.5B TTS on hipEngine

Status: **plan tightened; milestone 1 (freeze the torch oracle) is the next action**
(2026-09-14). The architecture review is unchanged; the API, benchmark protocol
and closure criteria below are now specified. No weights were downloaded and no
model has been executed, so every upstream claim here is a source review, not a
measurement.

This is the largest new runtime surface in this model group: a Qwen2 language
model controls text/speech transitions, a diffusion head generates continuous
acoustic latents, an acoustic decoder creates PCM, and a semantic encoder
feeds the generated audio back into the next language-model step. It needs a
speech-generation session, not just a different LM output head.

## Model and source availability

Target: `microsoft/VibeVoice-1.5B`, the long-form multi-speaker TTS release,
not `VibeVoice-Realtime-0.5B`. The card advertises 64K context, up to four
speakers and approximately 90 minutes. The 1.5B label names the language-model
scale; tokenizer/decoder modules add weights. Inventory the full checkpoint
before estimating residency. [Card][card], [config][config]

Microsoft's current repository records removal of the original TTS code and
marks its inference entry disabled. Audio tokenizers, diffusion-head components
and newer realtime code remain, but they do not constitute the complete original
1.5B inference loop. The reviewed community fork retains that loop and a local
DPM scheduler. Establish a working pinned checkpoint/fork/environment oracle
before implementing; this review read sources but did not execute the fork.
[Microsoft status][official], [community inference][inference]

The card declares MIT and separately describes research-only intended use,
audible disclosure and an imperceptible watermark. Do not turn the TODO's
wording into a claim that checkpoint tensors automatically enforce either.
The reviewed inference loop does not establish a complete watermark/disclaimer
mechanism. Record actual output behavior and provenance of the selected oracle;
consult the pinned card when defining a distributable product. [Card][card]

### Geometry and sampling contract

| Component | Configuration |
| --- | --- |
| Text backbone | Qwen2.5-1.5B / Qwen2 architecture; width 1536, 28 layers, FFN 8960 |
| Attention | 12 query / 2 KV heads × 128; causal attention, no GDN |
| Text config | Vocab 151936, tied embeddings, RoPE theta 1,000,000, context 65536 |
| Audio | Mono 24 kHz, compression 3200, nominal 7.5 frames/s |
| Continuous latents | Acoustic 64, semantic 128 |
| Acoustic codec | Causal encoder plus waveform decoder; six rate changes, total ×3200 |
| Diffusion head | 4 conditioned feed-forward layers, width 1536, FFN ratio 3, 64-d output |
| Diffusion metadata | 1000 training steps, cosine schedule, velocity prediction, default 20 inference steps |
| Inference scheduler | Reviewed loop uses `DPMSolverMultistepScheduler`; metadata says `ddpm` |
| Reference audio processor | 24 kHz, loudness normalization target −25 dBFS |

[Config][config], [processor config][preprocessor], [head][head],
[scheduler construction][model]. The scheduler class, timestep spacing, solver
order/history, CFG scale and step count are part of the oracle contract.
A generic DDPM sampler is not interchangeable with the inspected DPM solver.

```text
speaker references → acoustic encoder → sampled/scaled latent → connector ┐
script + speaker controls → token embeddings                              ┴ Qwen2
  text/control token → normal text/control transition
  speech token → conditioned diffusion → acoustic latent
               → undo latent scale/bias → streaming waveform decoder → PCM
               → semantic encoder(PCM)
               → acoustic connector(latent) + semantic connector(features)
               → next decoder input embedding
```

Reference voices use the acoustic encoder and connector in the reviewed loop;
the semantic encoder processes generated PCM during feedback.
The loop maintains positive and negative conditioning branches for classifier-free
guidance. Guidance combines conditional and unconditional head predictions;
negative-language-model cache updates and resets are part of the computation.
It streams decoded audio through a separate semantic cache and resets codec
state at speech boundaries. Preserve these mechanics before batching or capture.
[Generation loop][inference]

## Reuse and correctness traps

### Reuse from the ASR lane

The ASR lane is now on `main`, so the starting point is concrete. This table
separates what is reusable as-is from what is only a candidate that has not
passed a production gate.

| Surface | Module | State |
| --- | --- | --- |
| Causal audio encoder + connectors | `hipengine/runtime/vibevoice_encoder.py` (`VibevoiceFrontendRuntime.encode`, `_ScratchPool`) | **Validated for ASR.** The encoder topology and connector shapes overlap with TTS, but the TTS weights, the reference-audio processor contract and the latent scaling below need independent validation. |
| Qwen2 backbone | `hipengine/runtime/vibevoice_qwen2.py` (`VibevoiceQwen2Runtime`: `push_token`, `forward_layers`, `logits_argmax`, `reset`, `prefill_rows`) | **Validated for ASR** text generation. Same architecture TTS needs; TTS adds a second (negative) branch and speech-token transitions. |
| Q4_K_M backbone | `hipengine/runtime/vibevoice_qwen2_q4.py` | **Production-unqualified candidate.** Byte-exact against its own strict parent and WER-parity with bf16 over 200 clips, but no production profile is registered and gfx1100 is unverified. Do not treat it as the reference lane. |
| Kernel primitives | `hipengine/kernels/vibevoice.py` `PRIMITIVES`, registered by `hipengine/kernels/hip_gfx1100/vibevoice/registered.py` | RMSNorm, SiLU, GEMM, RoPE, KV-write and attention-span primitives are **validated for ASR shapes**. The conditioned diffusion head additionally needs adaptive normalization, which is not among them. |
| Scratch / arena reuse | `hipengine/core/memory.py` `DeviceMemoryArena` (with `rewind`), `_ScratchPool`, `_PrefillScratchArena` | **Validated.** The reuse pattern and its regression tests carry over directly. |
| Fixture and layout helpers | `hipengine/loading/vibevoice_layout.py`, `hipengine/generation/vibevoice_protocol.py` | `f32_to_bf16_bits`, `conv_rows_out` and `transpose_conv_weight_t` are shared. `build_prompt`/`parse_transcript` are ASR-specific and do not apply. |
| Test and harness patterns | `tests/test_gpu_vibevoice_qwen2_runner.py`, `scripts/vibevoice_q4_e2e_stage_breakdown.py` | The matched-request protocol, the negative-control pattern and synchronize-bracketed stage timing carry over. |

Two surfaces are **not** reusable and must be built new: the acoustic waveform
decoder (upsampling, chunk flush, sample accounting) and the diffusion head with
its solver. Moonshine's text output and cross-attention loop cannot substitute
for this design.

- Load `speech_scaling_factor` and `speech_bias_factor` from the checkpoint.
  The loop decodes `latent / scale - bias`; the acoustic feedback connector
  receives the generated model-space latent. Missing/NaN scale values are a
  loader error, not a reason to recompute training statistics at inference.
- Each speech frame has an inner denoising loop. Do not increment KV position
  per solver step; advance the language model according to its control sequence.
- Capture initial noise and scheduler state for fixtures. A seed alone does not
  align torch and HIP random streams. Per-request RNG ownership must survive
  batching, cancellation and speaker transitions.
- Preserve causal transposed-convolution padding, overlap, final flush and sample
  counts. Correct chunks can still click or duplicate samples at joins.
- Pin speech start/end/diffusion IDs and speaker formatting through the actual
  tokenizer/processor. The original checkpoint lacks a complete tokenizer bundle.
- Keep the original model's text/speech interleaving distinct from realtime TTS's
  text-streaming interface. Neither a fast first chunk nor a short example proves
  hour-long continuity.

[Inference and scaling source][inference], [audio tokenizer source][tokenizers].

## Initial API and scope

The first implementation is deliberately narrow: one serialized request at a
time, up to four speakers, a short script. Concurrency, batching and long-form
qualification are later milestones, and the first API must not assume them.

Proposed surface, mirroring the ASR adapter's shape:

```python
engine = LLM("microsoft/VibeVoice-1.5B")
result = engine.synthesize(
    script,                    # text to speak, including speaker turns
    speaker_references,        # list of 24 kHz mono float PCM, one per speaker
    sample_rate=24000,
    max_new_tokens=None,
    seed=None,
)
```

| Contract | Decision |
| --- | --- |
| Script input | Plain text. Multi-speaker turns use the checkpoint's speaker formatting, resolved through the actual tokenizer/processor rather than a literal template string. |
| Speaker references | Finite mono 24 kHz float PCM, one per speaker, at most four. Other rates are resampled by the caller, as in ASR. Reference audio is normalized to the processor's target, and that normalization is part of the oracle contract. |
| Output | Mono 24 kHz float PCM, including the codec's final flush. Output length is derived from generated speech frames, not assumed from the script. |
| Chunk format | PCM chunks of a declared size, with the first chunk reported separately from the pooled total. A boundary must not duplicate or drop samples. |
| Completion status | `eos` when the end token is emitted, `length` when a token or frame budget cuts generation off, `cancelled` for a cancelled request, and `error` with a reason when a step fails. Truncated output is returned with `length`, never silently padded. |
| Cancellation | Cancellable between steps. The session releases LM, negative-LM, codec and semantic state, and returns the audio completed so far, marked `cancelled`. |
| Reset | `reset()` returns the session to its initial state: positive and negative KV caches, codec streaming state, semantic cache and RNG. A speaker transition inside one request resets codec state at the speech boundary without discarding the request. |
| Ownership | The engine owns weights, codec state and scratch. Per-request RNG is owned by the request and must survive batching, cancellation and speaker transitions. Requests on one initialized engine serialize access to scratch and KV, as ASR does. |

## Suggested implementation sequence

1. Freeze and smoke-test the community oracle against the Microsoft checkpoint;
   inventory all weights, speech tokens, scheduler settings and codec state.
   Add proposed `scripts/vibevoice_tts_oracle_torch.py` and a model contract.
2. Reuse the ASR encoder/core work, then independently implement the acoustic
   decoder. Validate latent→PCM with recorded latents, including chunk boundaries
   and final flush, before involving the language model.
3. Port the diffusion head and solver against saved conditions/noise; compare
   every solver step and final latent. Preserve CFG branch behavior.
4. Add a proposed `hipengine/runtime/vibevoice_tts.py` session with explicit
   LM/negative-LM/codec/semantic/RNG ownership. Start with a short single-speaker
   script and then test multi-turn/multi-speaker speech controls.
5. Qualify complete audio generation and serving semantics. Optimize measured
   diffusion GEMMs, codec convolutions or session dispatch only after profiling.
   Reducing diffusion steps or changing guidance is a separate quality candidate.

## Task and performance gates

Boundary gates cover reference features, LM logits, positive/negative conditions,
per-step diffusion predictions, final latents, waveform samples and semantic
feedback. Use common random inputs for numerical gates. Raw waveform equality
is a debugging oracle; production acceptance also needs perceptual and task
quality, not just a low spectrogram error or similarity to another approximate
implementation.

Evaluate intelligibility with independent ASR WER/CER, speaker similarity,
turn attribution, duration/alignment, clipping, silence, repetition, join
artifacts and blinded listening on held-out scripts/voices. Include English
and Chinese, punctuation/numbers, short and long turns, multiple speakers,
interrupted requests and mixed-length batches. Calibrate quality criteria before
precision/solver tuning and compare multiple fixed seeds.

Measure time to first audible chunk, total wall time, output-audio duration,
real-time factor (wall/output seconds), peak memory, per-frame LM time, diffusion
steps/time, acoustic decoding and semantic feedback. Keep voice prompts, text,
CFG, step count, noise policy and output duration matched. Report initialization
and preprocessing separately. A smaller diffusion-step budget cannot be claimed
as a kernel speedup, and upstream Apple/NVIDIA results do not predict ROCm rates.

### Executable protocol

A lane comparison is only meaningful if every lane consumes the same frozen
request. Before any timing is retained:

- Commit **one request manifest** per benchmark, hashed, holding the script,
  speaker-reference PCM hashes, resolved prompt token IDs, CFG scale, solver
  order, timestep spacing, step count, seed, token/frame budget and the
  synchronization points. Every lane reads that manifest; a lane that cannot
  reproduce its hashes fails rather than reporting a number.
- Record the **random tensors**, not just the seed: initial diffusion noise, the
  per-step schedule and the codec's stochastic state. A seed does not align torch
  and HIP random streams, so a shared seed is not a shared request.
- Commit the **harness** that produced the number. An artifact whose command lives
  outside the repository is not reproducible, as the ASR scratch artifacts
  demonstrated before their probes were committed.
- Bracket each stage with explicit **device synchronizes**, so a stage is
  execution wall time rather than enqueue time, and state the boundaries in the
  artifact.
- Report four numbers separately, because they answer different questions:
  **cold start** (first request after load, including initialization),
  **warm synthesis** (steady-state per-request time), **time to first audible
  chunk** (the latency a listener perceives) and **pooled RTF** (total wall time
  over total output audio, which is duration-weighted). One headline RTF hides
  all four.
- RTF here is wall time over **output** audio seconds, the inverse convention of
  ASR's wall time over input audio. Every reported number must say which it uses.

## Milestone closure and failure accounting

### Milestone 1 closure

The first milestone is complete when a frozen torch oracle runs end to end and
its outputs are captured as fixtures. Concretely, all of:

- the pinned checkpoint, fork revision and environment are recorded, and the
  oracle produces non-empty PCM for one single-speaker and one two-speaker script;
- a hashed request manifest and its recorded random tensors are committed, and a
  re-run of the oracle reproduces its own output hashes;
- fixtures exist for reference-audio features, LM logits, positive and negative
  conditions, per-step diffusion predictions, the final latent, waveform samples
  and semantic feedback, each with its shape and dtype;
- every checkpoint weight is inventoried and accounted for, including
  `speech_scaling_factor` and `speech_bias_factor`;
- the scheduler is pinned by class, order, timestep spacing and step count, and a
  generic DDPM sampler is documented as not interchangeable;
- the oracle is known to work on the *reviewed* fork revision rather than a newer
  one, and any difference is recorded if it does not.

Out of scope for milestone 1: batching, concurrency, hour-scale continuity,
kernel porting, quantization, and any quality or speed claim.

### Failure accounting

Failed and degraded generations stay visible in **both** the quality and the
timing results. They are not dropped from a mean, and they are never reported as
a fast lane:

| Failure | Accounting |
| --- | --- |
| Truncation | Returned with `length`, counted separately, and reported alongside the quality score rather than excluded from it. |
| Missing or unparsed turns | Reported per request with expected and observed speaker-turn counts. A turn-count mismatch is a failure, not a formatting detail. |
| Empty or near-silent audio | Detected by duration and level, counted, and excluded from speaker-similarity and listening aggregates only when listed explicitly. |
| Step failure or cancellation | Reported with the reason, the stage, and the audio produced so far. A cancelled request is never counted as a completed synthesis. |
| Join artifacts | Sample-count continuity plus a click/duplication check at every chunk boundary, because correct chunks can still click at joins. |

Timing tables report the failure counts next to the latencies, so a lane cannot
look faster by failing more requests.

Acceptance thresholds are calibrated against the frozen oracle **before**
optimization begins. Choosing a threshold after seeing a candidate's result is
the benchmark-gaming failure mode, and the `AGENTS.md` "Anti-gaming" rules apply
here as they do to the MTP paths.

## Repository contracts and evidence

This is a source review, not an implemented plugin or a measured performance
result. Proposed implementation paths are future work. Keep runtime imports
torch-free, resolve kernels through `(backend, layer, quant, variant)`, retain
registered strict fallbacks, and preserve `KVLiveSpans` for cached attention.
Before a kernel port, read [KERNELS.md](KERNELS.md) and run
`python3 scripts/check_lineage.py --kind kernel --diff stat`; develop in-tree
and record upstream file/commit provenance. Add HIP-availability guards to GPU
tests and verify new kernels with a prebuilt-cache `rocprofv3` trace.

[EXECUTION-PROFILES.md](EXECUTION-PROFILES.md) governs promotion: exact control
and ownership, calibrated mean/p95/p99/max KL and category top-1 where logits
apply, determinism, isolation, BF16-relative and task-quality gates. Continuous
features need their own calibrated error measures plus downstream task checks.
The KL ≤ 0.05 / top-1 ≥ 90% outer smoke floor alone is insufficient. Assess
non-bit-identical candidates under production gates before rejecting them.

Benchmark matched artifacts, inputs, settings and workloads on the same physical
host. Record model/quant, shape, host identity, hardware, exact commands, result
and correctness gate. W7900/gfx1100 is the default lane; gfx1151 measurements
are independent. Follow [BENCHMARK.md](BENCHMARK.md) and update the result
artifact, scoreboard and changelog for any accepted measurement. No speedup or
validated capacity is claimed here. Update `PLAN.md` if implementation changes
architecture and track temporary loaders/flags in `REFACTOR.md`.

## Source pins

Reviewed on 2026-09-11 against hipEngine
`0dacb0df29864c622a787f8e42f64b4e36be9894`. Links below pin the inspected
config/code revisions where available. No weights were downloaded and no model
was executed. Library sources describe those revisions, not a tested dependency
set; the first implementation milestone must freeze a working oracle environment.

The community [scheduler implementation][scheduler] is part of the oracle pin.

[card]: https://huggingface.co/microsoft/VibeVoice-1.5B/blob/c00898d257e6b46004e3e2866a47534085fb685a/README.md
[config]: https://huggingface.co/microsoft/VibeVoice-1.5B/blob/c00898d257e6b46004e3e2866a47534085fb685a/config.json
[preprocessor]: https://huggingface.co/microsoft/VibeVoice-1.5B/blob/c00898d257e6b46004e3e2866a47534085fb685a/preprocessor_config.json
[official]: https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/README.md
[inference]: https://github.com/vibevoice-community/VibeVoice/blob/952326ddb264062466a888cf32a5b2f4e803e16e/vibevoice/modular/modeling_vibevoice_inference.py
[model]: https://github.com/vibevoice-community/VibeVoice/blob/952326ddb264062466a888cf32a5b2f4e803e16e/vibevoice/modular/modeling_vibevoice.py
[head]: https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vibevoice/modular/modular_vibevoice_diffusion_head.py
[tokenizers]: https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vibevoice/modular/modular_vibevoice_tokenizer.py
[scheduler]: https://github.com/vibevoice-community/VibeVoice/blob/952326ddb264062466a888cf32a5b2f4e803e16e/vibevoice/schedule/dpm_solver.py
