# MODEL-VIBEVOICE-ASR.md — VibeVoice ASR on hipEngine

Status: **native implementation available; production qualification incomplete**
(2026-09-14). The gfx1151 lane has CPU/GPU fixture checks, stateful 60-second
encoder chunks and a torch-free public transcription smoke through content and
EOS. gfx1100 hardware execution is unverified. Earlier faster-than-torch numbers
were withdrawn because the input/timing protocol differed between lanes.

Use `LLM("microsoft/VibeVoice-ASR-HF").transcribe(pcm, sample_rate=24000,
context="hotwords", max_new_tokens=256, seed=20260914)`. Input is finite mono
24 kHz floating PCM; other rates must be resampled by the caller. Normalization
and the template follow the HF processor. Duration uses the original sample
count; the encoder pads the final partial frame. Each result exposes raw text,
validated segments (or `None`), token IDs and `eos`/`length` finish status.
Requests on one initialized generator serialize access to scratch and KV.
The default context capacity is 4096 tokens (configurable up to 16000).
The public adapter uses strict GEMM and incremental prefill fallbacks; optimized
arithmetic remains a low-level qualification candidate. No production profile
is registered for this model yet. Multi-category quality, full-logit production
envelopes, sustained concurrency and same-protocol speed remain qualification
work; the initial short greedy fixture proves only the JSON prefix.

The new work is a pair of causal audio encoders, their embedding connectors,
and audio-aware Qwen2 generation. There is no vision tower and no synthesis
diffusion loop on this ASR path. Moonshine provides useful runtime/fixture
patterns, but its encoder-decoder cross-attention is a different architecture.

## Model and reference implementations

Target: `microsoft/VibeVoice-ASR`, advertised as 9B parameters with long-form
transcription, speaker labels, timestamps and optional hotword context. The
upstream 60-minute/64K claim is a model capability claim, not a tested hipEngine
limit. The checkpoint card declares MIT. [Model card][card]

Two official artifacts need to be kept distinct:

| Artifact | Loader contract | Use |
| --- | --- | --- |
| `microsoft/VibeVoice-ASR` | `model_type=vibevoice`, `VibeVoiceForASRTraining` metadata; Microsoft custom inference classes | Original-weight oracle and provenance |
| `microsoft/VibeVoice-ASR-HF` | `model_type=vibevoice_asr`, `VibeVoiceAsrForConditionalGeneration`; ships tokenizer, processor and template | Preferred initial native-Transformers fixture source, after a smoke run |

The HF version changes namespaces and serializes encoder configuration
differently; do not mix weights, config, or token IDs across them. Despite
the original config's top-level `"dtype": "float32"` metadata, both
checkpoints' safetensors are bfloat16 (about 17-18 GB of weights); the
HF artifact remains the initial target because it ships the tokenizer,
processor and chat template. Current
Microsoft source offers an inference-specific ASR class despite the original
checkpoint's training-class metadata. Both implementations can supply oracle
fixtures; neither was run for this review. Microsoft's repository also links
vLLM ASR serving, useful for a later same-host baseline. Streaming ASR is a
separate checkpoint and protocol, not a flag that makes this model streaming.
[Original config][config], [HF config][hfconfig], [Microsoft source][asrsource]

### Geometry

| Component | Configuration |
| --- | --- |
| Audio | Mono 24 kHz; total stride 3200 samples, nominal 7.5 latent frames/s |
| Acoustic encoder | Causal convolution/depthwise-convolution blocks, 64-d latent |
| Semantic encoder | Related causal encoder, 128-d latent |
| Encoder stages | Depths `[3,3,3,3,3,3,8]`, base filters 32; downsampling product 3200 |
| Connectors | Each latent → linear → RMSNorm → linear → text width; sum both streams |
| Text backbone | Qwen2, width 3584, 28 layers, FFN 18944 |
| Attention | Full causal GQA; 28 query / 4 KV heads, head dim 128; no GDN |
| Text positions | Scalar RoPE, theta 1,000,000; config ceiling 131,072 |
| Vocabulary | 152,064; HF config uses untied output head |
| HF audio IDs | Begin 151646, end 151647, placeholder 151648 |
| HF encoder chunk | 1,440,000 samples = 60 seconds; carries convolution padding cache |

[Configs][hfconfig], [encoder/projector implementation][hfmodel]. Original
ratios are stored `[8,5,5,4,2,2]`; the HF encoder serializes execution order as
`[2,2,4,5,5,8]`. Confirm weight/stage ordering rather than reversing tensors
from the list alone. The 60-second tokenizer chunk is memory management;
it does not split the language model's long-form context into independent ASR jobs.

```text
waveform → resample/mono/loudness processing
         → acoustic encoder → sampled acoustic latent → connector ┐
         → semantic encoder → mean semantic latent → connector    ┴ sum
         → replace speech placeholders in prompt embeddings
         → Qwen2 prefill + cached text decode → structured transcription
```

## Important implementation decisions

**Inference has acoustic randomness.** The original encoder uses Gaussian
sampling with per-example scale `fix_std / 0.8` (0.5/0.8 = 0.625) multiplied
by per-element Gaussian noise; the HF path explicitly samples the same two
noise levels. Semantic features use the encoder mean. Greedy token sampling
therefore does not by itself make runs deterministic. Fixtures must record
noise tensors or sampled latents, not merely a seed shared between torch and
HIP RNGs. Sample after joining encoded chunks, as the reviewed implementations
do; drawing noise separately per chunk changes the protocol. A mean-only
candidate changes model behavior and needs a task gate. [Sampling source][tokenizers],
[HF audio feature path][hfmodel]

Preserve causal padding at every convolution stage, chunk carry, final partial
frames and `ceil(samples/3200)` masking. Test 3199/3200/3201-sample boundaries,
unequal padded batches and 60-second transitions. Carry encoder state while
chunking one recording and reset it between recordings. Do not confuse the
encoder padding cache with the decoder KV cache. [HF processor][hfprocessor]

Original weights do not ship text-tokenizer files; the original processor
resolves a base tokenizer and speech tokens. Prefer the HF artifact's complete
processor/template/tokenizer bundle for initial reproducibility. Pin waveform
normalization, channel folding, resampling, silence handling, hotword prompt,
padding side, EOS and JSON parsing. Do not substitute Whisper mel features or
Moonshine's input frontend. [Original processor][processor]

At 7.5 Hz, one hour is nominally 27,000 audio positions, plus text and output.
For this decoder, uncompressed BF16 KV alone is
`2 * 28 * 4 * 128 * 2 = 57,344 bytes/token`, about 1.44 GiB at 27,000 tokens.
This is a derived allocation estimate excluding output, weights, scratch and
fragmentation; budget the real padded workload before claiming hour-long support.

## Reuse and implementation sequence

| Piece | In-tree starting point | Work still required |
| --- | --- | --- |
| Causal depthwise conv + state carry | `hipengine/kernels/hip_gfx1100/linear_attn/conv.hip` decode/prefill/segments/state variants | Multi-stage strided topology: kernel 7, strides 2–8, pointwise channel mixing, RMSNorm/layer-scale/GELU blocks, 60-second chunk carry |
| Audio model and lifetime pattern | `hipengine/models/moonshine.py`, `hipengine/loading/moonshine.py`, `hipengine/runtime/moonshine.py` | New causal tokenizer topology and continuous feature contract |
| Dense decoder primitives | Existing linear/norm/rotary/attention kernels and GGUF runners | Qwen2 layer contract, QKV biases, RoPE, untied LM head; no automatic Qwen3.5 compatibility |
| Numerical fixtures | `hipengine/kernels/cpu_reference/moonshine_encoder.py` and model oracle scripts | Independent VibeVoice CPU reference and recorded encoder noise |
| Serving | Existing generator registry and request lifecycle | Audio input/result API and feature injection; no demonstrated ASR endpoint compatibility |

1. Add proposed `hipengine/models/vibevoice_asr.py` and a loader contract.
   Freeze one artifact plus tokenizer and tested oracle environment. Inventory
   tensor names, shapes, dtypes and heads before allocating the model.
2. Add `scripts/vibevoice_asr_oracle_torch.py` and fixtures for processed PCM,
   encoder stage outputs, sampled latents, connector sum, first logits and
   teacher-forced cached steps. Prove Qwen2 text decoding independently.
3. Port CPU references and HIP audio encoders/connectors with common noise
   inputs. Start single-recording short audio, then stateful chunk equivalence.
4. Connect audio embedding prefill to cached generation and structured output.
   Validate hotwords, timestamps and speaker parsing without silently repairing
   errors into plausible successful results.
5. Qualify longer recordings, concurrency and production numerics; only then
   tune convolution batching, prefill memory, attention and weight precision.

## Task and performance gates

### Q4_K_M prefill arithmetic

The Q4_K_M backbone's batched prefill reads the unmodified GGUF K-quant blocks
and is dispatched by the resolved GGML type, so the arithmetic depends on the
tensor's quant:

- **Q4_K** (`q/k/v`, `o`, `gate/up`, and `ffn_down` on half the layers) runs a
  native f32- or bf16-in / f32-out WMMA prefill kernel. `o_proj` keeps f32
  activations and uses the f32/f32 variant.
- **Q6_K** (`attn_v`, `ffn_down` on the other half) has no f32-output prefill
  entry point. Its result is produced in bf16 and widened to f32 with a
  `bf16_to_f32` pass. For a Q6_K `o_proj`, the f32 attention output is likewise
  narrowed to bf16 before the kernel.

That widening is a **real arithmetic change, not a reassociation**: the Q6_K
path rounds through bf16 where the Q4_K path accumulates in f32. Top-1
agreement alone is therefore insufficient evidence for the route, and the
production gate in `docs/EXECUTION-PROFILES.md` section 6.1 is applied
per prompt length as a scope (mean/p95/p99/max full-vocabulary row KL plus
top-1), measured against the sequential row-by-row route as the strict parent.

### Which profile each clause belongs to

The batched prefill and the sequential row-by-row route are two different
**schedules**. Under `docs/EXECUTION-PROFILES.md` section 4.2 and the
`production` profile row, cross-width generated-ID equality is diagnostic
rather than a promotion requirement. So the two clause families in that gate are
not the same kind of evidence:

- The four **KL** clauses are the binding production drift bound for this route.
  They pass, with mean KL ~7e-4 against an envelope mean of 1e-3.
- The **top-1 against the sequential route** clause is a cross-schedule
  diagnostic. It lands near 95% against a 0.99 envelope because the rows that
  flip carry a model decision margin of 0.006-0.039 nats against a 0.79 median,
  so a sub-1e-3 perturbation decides them. The dense bf16 lane, which this work
  does not touch, shows the same effect (87/89), so it is a property of the
  schedule comparison rather than of the Q4 arithmetic. It stays asserted as a
  non-strict xfail so the unmet clause remains visible.

A green run of the runner suite therefore means the KL drift bound, the strict
fallback and the ownership/reuse gates pass. It does not certify the model for
production: that additionally needs isolation, BF16-relative and task-quality
gates, which are tracked separately in this document.

The scratch-allocation work on this model (prefill scratch arena, front-end
arena rewind, decode `o_proj` scratch, Q6_K `ffn_down` direct bf16) is
**byte-exact** on its affected surface, so it meets the `strict` arithmetic
contract, which is stronger than `production` requires. It changes no values and
needs no numerical re-qualification.

### One measured trap

`prefill_rows` overwrites its input buffer with its post-layer-stack result.
Any determinism or repeat-comparison check must re-upload the input before
each run; reusing one buffer feeds the previous output back in as input, which
looks exactly like run-to-run non-determinism. The batched prefill is
deterministic in both lanes once the input is restored.

Use multi-speaker, overlapping speech, silence, noisy/far-field recordings,
English, Japanese and code-switching cases, hotwords and held-out recordings.
Measure WER/CER, speaker-attributed or concatenated-permutation WER, diarization
error and timestamp error. Preserve task definitions and speaker-label matching;
raw numeric speaker IDs need not match across equivalent label permutations.
Separate raw generation from parser success and report malformed/truncated output.

Measure input seconds, output length, preprocessing, both encoders, prefill,
decode, total latency and peak memory. Define real-time factor as processing
wall time / input audio duration (lower is better). Short clips and long meetings
are separate workloads. Use the same PCM, noise and task settings for numerical
comparison, then fixed-seed ensembles for stochastic task quality. No diffusion
head or acoustic waveform decoder is required merely because the original
config contains diffusion metadata.

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

[card]: https://huggingface.co/microsoft/VibeVoice-ASR/blob/d0c9efdb8d614685062c04425d91e01b6f37d944/README.md
[config]: https://huggingface.co/microsoft/VibeVoice-ASR/blob/d0c9efdb8d614685062c04425d91e01b6f37d944/config.json
[hfconfig]: https://huggingface.co/microsoft/VibeVoice-ASR-HF/blob/f22241c2062b3b25272bf117397e03d73381037a/config.json
[asrsource]: https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vibevoice/modular/modeling_vibevoice_asr.py
[hfmodel]: https://github.com/huggingface/transformers/blob/177e90dd2d51273fa235dd8bacee7c80f1eef067/src/transformers/models/vibevoice_asr/modeling_vibevoice_asr.py
[hfprocessor]: https://github.com/huggingface/transformers/blob/177e90dd2d51273fa235dd8bacee7c80f1eef067/src/transformers/models/vibevoice_asr/processing_vibevoice_asr.py
[tokenizers]: https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vibevoice/modular/modular_vibevoice_tokenizer.py
[processor]: https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vibevoice/processor/vibevoice_asr_processor.py


## Standalone Q4 GGUF

`python scripts/vibevoice_asr_transcribe.py --model model.gguf --audio speech.wav`
loads all frontend/backbone weights and tokenizer/configuration assets from one
GGUF. Input is mono 24 kHz PCM16 WAV; no HF checkpoint or network is needed at
inference. This CLI uses the existing evaluated Q4/WMMA candidates, without
changing the public strict BF16 profile or claiming broader production qualification.

The exporter embeds `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`processor_config.json`, `generation_config.json` and `chat_template.jinja` in
versioned, SHA-256-checked metadata. `--repackage old.gguf --model HF_SNAPSHOT
--out new.gguf` upgrades an existing file without requantizing tensors. The
frontend loader shares the HF path's tensor-name mapping and geometry checks.
See `scripts/vibevoice_asr_standalone_check.py` for weight/asset verification and
`tests/test_gpu_vibevoice_standalone.py` for offline HF-frontend parity.
