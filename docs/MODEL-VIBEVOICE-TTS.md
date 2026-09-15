# MODEL-VIBEVOICE-TTS.md — VibeVoice 1.5B TTS on hipEngine

Status: **milestones 1 (frozen torch oracle), 2 (the acoustic decoder),
3 (the diffusion head + DPMSolver) and 4 (the torch-free generation session)
closed 2026-09-15.** The session runs the generation loop on HIP without torch
and reproduces the frozen oracle's 27-token constrained greedy chain exactly
(25 diffusion frames, 25 decoded chunks) on the pinned single-speaker request.
The API, benchmark protocol and closure criteria below are specified.

The open work is integration correctness on the unassisted request path. The
session benchmark replays recorded prompt embeddings and negative conditions, so
it exercises neither reference-audio preparation nor the negative-LM branch, and
the voice-prompt and negative-conditioning entry points are covered by
reconstructed paths rather than called directly. Until one unassisted
single-speaker request passes end to end, the timing numbers are diagnostic
rather than retained performance claims.

Every measurement in this document comes from an `AMD RYZEN AI MAX+ PRO 395
w/ Radeon 8060S` (`gfx1151`) host, not the repository-default
W7900/`gfx1100`. Upstream architecture claims that the fixtures do not cover
are source review.

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

### Trajectory sensitivity in the frozen fixtures

The solver's 20-step trajectory is not uniformly well conditioned, so a single
frozen call cannot carry a tight numerical gate. Replaying the two-speaker
request's first diffusion call from a **one bf16-ULP** change in its recorded
initial noise produces a different trajectory: the final latent moves by 3.3
relative and one eps step by 1.5 of its peak. The single-speaker call does not
amplify that change (0.044 on eps, 0.019 on the final latent). Both the CPU
reference and the device head behave this way, so the spread is a property of
the frozen trajectory and not of either implementation: on that call the CPU
reference lands 0.27 relative from the oracle and the device head 0.036, and
moving either one by a single ULP moves both to about 3.4.

The diffusion unit gate therefore measures each trajectory's own sensitivity
band, by replaying it from +/-1 bf16 ULP on the initial noise, and uses the band
as the limit wherever the band is wider than the nominal envelope. Every
nominal gate keeps its full force on the single-speaker call, and
`test_fixture_chaos_band_is_measured_not_assumed` asserts that the band is both
present for the two-speaker call and load-bearing there.

This also constrains anything that feeds diffusion output back into the LM: a
chaotic trajectory makes the feedback embedding reproducible only to the
trajectory's band, so agreement downstream of it cannot be tightened by making
the head more accurate.

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

What is **not** available to reuse: the diffusion head with its solver, and the
language-model generation loop that drives text/speech transitions. Those exist
only in the community fork and must be ported. transformers 5.15.0 has exactly
two VibeVoice families, `vibevoice_acoustic_tokenizer` and `vibevoice_asr`;
there is no `vibevoice` causal-LM or diffusion-head model. (The `Diffusion*`
symbols it does export are `DiffusionGemma`, unrelated.)

**The acoustic waveform decoder: native classes exist, but not for this
checkpoint.** An earlier revision of this document said the decoder must be built
new. That was too strong, but the opposite reading — that transformers ships it
ready to use — is wrong in a way that matters more.

transformers 5.15.0 does ship `VibeVoiceAcousticTokenizerModel` and
`VibeVoiceAcousticTokenizerDecoderModel`, with a streaming
`forward(hidden_states, padding_cache=None, use_cache=False)`. They implement the
**`vibevoice/VibeVoice-1.5B-hf`** conversion, not `microsoft/VibeVoice-1.5B`,
and the two are different audio-tokenizer topologies:

| | `microsoft/VibeVoice-1.5B` (this lane) | `vibevoice/VibeVoice-1.5B-hf` (native) |
| --- | --- | --- |
| Decoder key shape | `stages.N.M`, `upsample_layers.N` | `conv_layers.N.stage.M`, `conv_layers.N.convtr` |
| Decoder stage counts | `[8, 3, 3, 3, 3, 3, 3]` | `[3, 3, 3, 3, 3, 3, 8]` |
| Decoder parameters | 343,695,969 | 175,793,409 |
| Depth source | `reversed(encoder_depths)` when `decoder_depths` is null | `depths` stored directly |
| Scale/bias keys | `speech_scaling_factor`, `speech_bias_factor` | `latent_scaling_factor`, `latent_bias_factor` |

Building the native decoder from this checkpoint's `acoustic_tokenizer_config`
produces a 175,793,409-parameter network, not the checkpoint's 343,695,969. The
native module ignores `decoder_depths` entirely: forcing
`decoder_depths="8-3-3-3-3-3-3"` changes nothing. The checkpoint's decoder
weights therefore cannot be loaded into it.

The consequence for the port is that the decoder still has to be built against
the original's weights. The native implementation is worth reading for the
streaming and padding-cache mechanics and is a second reference for the module
structure, but it is not a substitute and it is not the oracle. Do not let a
matching `vae_dim` (64) or `decoder_ratios` (`[8, 5, 5, 4, 2, 2]`) suggest
the rest matches: those two values do match, and they are not sufficient.

Moonshine's text output and cross-attention loop cannot substitute for this
design.

- Load `speech_scaling_factor` and `speech_bias_factor` from the checkpoint.
  The loop decodes `latent / scale - bias`; the acoustic feedback connector
  receives the generated model-space latent. Missing/NaN scale values are a
  loader error, not a reason to recompute training statistics at inference.

  Measured detail, because these two scalars are easy to misplace: they are
  **weights**, not config values. `config.json` has no such keys, but the state
  dict carries `model.speech_scaling_factor` and `model.speech_bias_factor` as
  bf16 scalars in shard 1 (`0.1962890625` and `-0.04931640625` for
  `microsoft/VibeVoice-1.5B`). The module registers them as `NaN` buffers at
  construction time, which is the default for a from-scratch model; a weight
  load overwrites that default. The inference path only ever *applies* them
  (`(latent + bias) * scale` on encode, `latent / scale - bias` on decode); the
  `NaN`-triggered recomputation exists only in the training model
  (`modeling_vibevoice.py:307`). So the failure mode is real and one-directional:
  a model built from config without a weight load keeps `NaN` and poisons every
  speech embedding downstream. Reading the config to find them will not work;
  reading the state dict will.
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

## Oracle environment

Measured on 2026-09-14; this is the environment milestone 1 must freeze.

| Item | Value |
| --- | --- |
| Checkpoint | `microsoft/VibeVoice-1.5B` revision `c00898d257e6b46004e3e2866a47534085fb685a` |
| Checkpoint size | 5.41 GB, 1204 tensors in 3 safetensors shards |
| Parameters | **2704.0 M** in bf16. The "1.5B" label names the language-model scale; tokenizer, decoder and diffusion modules add the rest. |
| Fork | `vibevoice-community/VibeVoice` revision `952326ddb264062466a888cf32a5b2f4e803e16e` |
| `transformers` | **4.51.3, pinned.** See the collision note below. |
| `tokenizers` | 0.21.4 (what 4.51.3 resolves) |
| `huggingface-hub` | 0.36.2 (`<1.0`; 1.x is rejected by both 4.51.3 and `tokenizers` 0.21.4) |
| `torch` | 2.13.0+rocm10.0.0, reused from the host, HIP available |
| Lane | AMD Radeon 8060S (gfx1151). The W7900/gfx1100 lane is unverified for this model. |

**Why 4.51.3 is pinned.** transformers 5.15.0 already registers model type
`vibevoice_acoustic_tokenizer` (plus `_encoder`, `_decoder`, and `vibevoice_asr`),
and `AutoModel.register` keys on the `model_type` string rather than the class.
The fork declares the same `model_type`, so its registration at
`modular_vibevoice_tokenizer.py:1188` aborts the import. Measured under 5.15.0:

```
ValueError: '<class 'vibevoice.modular.configuration_vibevoice.VibeVoiceAcousticTokenizerConfig'>'
is already used by a Transformers model.
```

The same finding is what makes the acoustic decoder *look* reusable, and the
appearance is misleading: sharing a `model_type` string means only that both
implementations claim the name, not that they build the same network. 5.15.0's
native tokenizer targets the `-hf` conversion, whose decoder has a different
topology from this checkpoint's. See "The acoustic waveform decoder" above for
the measured comparison. What 5.15.0 does **not** have is any `vibevoice`
`AutoModelForCausalLM` entry, so the diffusion head and the generation loop exist
only in the fork.

Working venv: `/home/lhl/venvs/vibevoice-tts-oracle`. Recreate it with
`scripts/setup_vibevoice_tts_oracle_env.sh`. It is created with
`--system-site-packages` so the host's ROCm torch is reused instead of
reinstalled, then pinned. Resolved versions are recorded in
`scripts/vibevoice_tts_oracle_requirements.txt`.

The fork does not call `AutoConfig.register`, so `AutoConfig.from_pretrained`
fails with `model type 'vibevoice'`. Use the fork's own classes:
`VibeVoiceConfig.from_pretrained`, then
`VibeVoiceForConditionalGenerationInference.from_pretrained`.

Confirmed working: a single-speaker and a two-speaker script both produce
non-empty PCM at 24 kHz through `VibeVoiceProcessor` + `model.generate(...)`.

`scripts/vibevoice_tts_oracle_torch.py` is the oracle. It writes
`tests/fixtures/vibevoice_tts/` and must run under the oracle venv:

```
PYTHONPATH=/home/lhl/VibeVoice-community \
    /home/lhl/venvs/vibevoice-tts-oracle/bin/python \
    scripts/vibevoice_tts_oracle_torch.py
```

Provenance is enforced, not just recorded: the processor and model are loaded
from the exact local snapshot named by `--model-revision` (default: the pin
above), and a dirty fork tree is refused unless overridden. It captures by
wrapping the reference module or method that does each piece of work, so no
arithmetic is reimplemented and a fixture cannot disagree with the reference by
construction. Two runs produce byte-identical artifacts. Fixture schema 2 (one
connected execution trace per request) records, per request, `manifest.json`
and `<name>_reference`, `_lm`, `_diffusion`, `_feedback` and `_audio` `.npz`
files: the generation pass's speech encode with its random draws, the prefill
pass's own encode beside its logits, **every** diffusion call with its initial
noise, per-step eps and latent state, final latent, post-scale decoder input
and batch sample indices, the acoustic and semantic feedback embeddings and
their sum, the full scheduler config (class, `dpmsolver++`, order 2, midpoint,
cosine, v_prediction, exact timesteps), streaming-cache reset events, and all
decoder chunks with the concatenated waveform.

Two argument shapes are easy to get wrong and both fail quietly:

- `voice_samples=[[alice, carter]]` is one batch item with two speakers.
  `[[alice], [carter]]` is two batch items with one voice each: generation still
  runs and still produces audio, but the reference audio covers only the first
  voice and the prompt is roughly half its correct length.
- The shipped reference voices are **16 kHz**, not 24 kHz. Take `speech_tensors`
  from the processor, which owns resampling, rather than reading the WAV files.

### Multi-speaker reference padding

The oracle batches every voice of a request into one `forward_speech_features`
call: `speech_tensors` is right zero padded to the longest reference in the
batch before the encoder runs, and `speech_masks` selects each voice's real
frames afterwards. The acoustic encoder is **not translation invariant at its
tail**, so that pad is part of the arithmetic rather than a storage detail. On
the frozen two-speaker fixture the shorter speaker's final frame lands 0.23
relative away from the oracle when it is encoded alone and 0.04 when it is
encoded inside the batch, so a per-voice encode cannot be substituted.
`VibevoiceTtsSession.voice_prompt_rows_multi` therefore pads to the batch max
and returns each voice's sampled latents beside the concatenation of the
per-voice connected rows in mask order. For one voice the pad is empty and the
path is bit-identical to `voice_prompt_rows`.

Status on the frozen fixtures: the two-speaker prompt is built correctly — 70 +
208 = 278 connected rows spliced at the mask's two runs, final-position prefill
logits within 0.012 of the oracle with a matching argmax. The 59-token greedy
chain is **not yet exact**. The first 31 tokens match and the divergence is the
32nd, the first span's end, where the oracle emits `speech_end` and this session
emits another `speech_diffusion` (top-2 gap 9 logits, so not a near tie). It is
localized to the prompt rows rather than the LM or the diffusion head: the
head reproduces the oracle's own per-call latents to 0.036 and the LM reproduces
the oracle's prefill hidden to 0.05 given the oracle's own connected rows, while
the 208-frame voice's connected rows are 0.206 off because the encoder's bf16
drift on that voice is amplified by the connector. One open question remains:
the first diffusion call of this request is a chaotic trajectory (see
"Trajectory sensitivity in the frozen fixtures"), so the feedback embedding that
feeds the LM may itself be reproducible only to that band, which would make the
span end inherently unreproducible rather than fixable by accuracy work.
`test_two_speaker_prompt_rows_match_reference` and
`test_two_speaker_prefill_logits_match_reference` gate the parts that do hold;
`test_two_speaker_greedy_chain_matches_torch` is a strict xfail that fails
loudly once the chain is exact.

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
   Done 2026-09-15 (see "Milestone closure" below).
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

**Status: closed (2026-09-15), re-established after a second review.** The
first closure used fixtures that were reproducible but incomplete as boundary
fixtures: the recorder cleared its per-call state before the decoder hook ran,
so no decoder input was captured and only 8 of the diffusion calls were
retained; the reference, prefill and generation captures each held a different
speech-encode pass's random draws, so the saved reference embeddings were not
the ones that produced the saved logits; the feedback fixtures carried semantic
features only; the scheduler record omitted the solver configuration; and the
checkpoint pin was recorded from one snapshot resolution while the weights were
loaded through another. All of that is fixed in fixture schema 2 and the
fixtures were regenerated.

The frozen fixtures are `tests/fixtures/vibevoice_tts/` (10 `.npz` files plus
`manifest.json`, 1,690 arrays, 12 MB). A fresh run reproduces every one of
them byte-identically. Each request now holds one connected execution trace:
the generation pass's speech encode with its draws, the prefill pass's encode
beside its logits, every diffusion call with its initial noise, per-step eps
and latent state, final latent, post-scale decoder input and sample indices,
feedback embeddings with their sum, the full scheduler config, cache-reset
events, and all decoder chunks. Measured on the pinned run: single 148
generated ids → 80,000 samples (3.333 s, rms 0.055272) with all 25 diffusion
calls captured; two 411 ids → 176,000 samples (7.333 s, rms 0.091783) with all
55 calls captured. The weight inventory is
`tests/fixtures/vibevoice_tts/weight_inventory.json`, produced by
`scripts/vibevoice_tts_weight_inventory.py`: 1204 tensors, 2,704,021,987
parameters, 5.037 GiB, zero orphan keys. The one key absent from the
checkpoint, `lm_head.weight`, is tied to `embed_tokens.weight` with verified
shared storage. The two scaling factors are buffers rather than parameters,
which is why a `.parameters()` sum undercounts the checkpoint by exactly 2.
See `worklog/entries/20260915T032856.748048Z-lhl-vibevoice-tts-milestone1-closure-511389.md`
for the first closure and the schema-2 entry that supersedes its capture
claims.

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

Milestone 1 produced measured reference outputs — the frozen oracle fixtures
and weight inventory — but this is still not an implemented plugin and not a
measured performance result. The implementation path below is future work.
Keep runtime imports torch-free, resolve kernels through
`(backend, layer, quant, variant)`, retain
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
config/code revisions where available. At review time no weights had been
downloaded and no model had been executed; milestone 1 has since frozen the
working oracle environment (see "Oracle environment" above).

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
