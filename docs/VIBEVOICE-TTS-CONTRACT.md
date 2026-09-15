# VibeVoice 1.5B TTS — implementation contract

Status: **oracle frozen and smoke-tested** (2026-09-15). This document records
the measured contract the hipEngine implementation must satisfy and the exact
oracle it is validated against. The review that motivated it is
[MODEL-VIBEVOICE-TTS.md](MODEL-VIBEVOICE-TTS.md); that file's traps and gates
remain binding. This file is the concrete, measured companion.

## Oracle pin

| Component | Value |
| --- | --- |
| Checkpoint | `microsoft/VibeVoice-1.5B` @ `c00898d257e6b46004e3e2866a47534085fb685a`, local copy at `/models/vibevoice/VibeVoice-1.5B` |
| Inference loop | community fork `vibevoice-community/VibeVoice` @ `952326ddb264062466a888cf32a5b2f4e803e16e`, clone at `/home/lhl/VibeVoice-community` (read-only peer) |
| Environment | `/home/lhl/venvs/vibevoice-oracle` (python 3.12, `--system-site-packages` over the therock env: torch 2.10.0+rocm, transformers pinned 4.51.3, diffusers 0.39.0, librosa; fork installed `pip install -e . --no-deps`) |
| Env quirk | the venv's `sitecustomize.py` patches transformers' flash_attn probe to report missing — the base env's CUDA flash_attn build raises on import inside this ROCm venv; the oracle runs SDPA |
| Driver | `scripts/vibevoice_tts_oracle_torch.py` (in-repo; run with the oracle venv's python, never the engine venv) |
| Smoke test | seed 20260915, speaker Alice, one-sentence script: 148 total tokens, 27 generated, 3.33 s audio, peak 0.348, RTF 3.75× under PyTorch-ROCm bf16 on the W7900 |

## Measured geometry and runtime constants

From the checkpoint and the loaded model (full data:
`benchmarks/results/2026-09-15-vibevoice-tts-oracle-inventory.json`):

- **Decoder**: Qwen2, width 1536, 28 layers, 12 query / 2 KV heads (head dim
  128), FFN 8960, vocab 151936, tied embeddings, RoPE theta 1e6, RMSNorm eps
  1e-6, context 65536, bf16.
- **Speech control tokens** (Qwen2 vision tokens): start `<|vision_start|>` =
  151652, end `<|vision_end|>` = 151653, diffusion pad `<|vision_pad|>` =
  151654, EOS 151643. The generated stream is constrained to
  {start, end, diffusion, EOS} (+BOS allowed).
- **Latent scaling**: `speech_scaling_factor` = 0.1962890625,
  `speech_bias_factor` = −0.04931640625. Decode rule
  `scaled = latent / scale − bias`; the acoustic feedback connector receives
  the unscaled model-space latent.
- **Diffusion head**: 4 conditioned layers, width 1536, FFN 4608, latent 64,
  RMSNorm eps 1e-5, adaLN modulation, v-prediction, cosine schedule, 1000
  train steps, 20 inference steps.
- **Scheduler** (`DPMSolverMultistepScheduler`, fork-local): timesteps 999,
  949, … (spacing 50, trailing step to sigma 0); first sigma ≈ 20291.3 —
  the exact sigma/timestep vectors are in the inventory JSON and are part of
  the per-step gate.
- **Acoustic tokenizer**: VAE dim 64, encoder depths `3-3-3-3-3-3-8`, ratios
  [8, 5, 5, 4, 2, 2] (total ×3200 to 24 kHz), encoder/decoder n_filters 32,
  causal conv, depthwise-conv mixer, RMSNorm eps 1e-5, layer-scale 1e-6,
  `fix_std` 0.5 with `std_dist_type=gaussian` for reference encoding
  (`value = fix_std / 0.8` per-sample draw in vae mode).
- **Semantic tokenizer**: same family, VAE dim 128, no decoder, no sampling
  (`.mean` of the encoder output feeds the semantic connector).

## Component ownership and reuse in hipEngine

| Component | Reuse | New surface |
| --- | --- | --- |
| Qwen2 decoder | RMSNorm, SiLU-gated GEMM, rope, attention primitives, KV cache | bf16 safetensors loader (not GGUF); Qwen2 model plugin keyed by geometry; 12Q/2KV MHA-with-GQA decode kernel path |
| Negative-LM branch | same weights, second KV cache | cache-shift/correct-count semantics from the reference loop (`refresh_negative=True` default: negative forward only at diffusion steps; non-diffusion samples shift the negative cache by `correct_cnt`) |
| Diffusion head | RMSNorm, SiLU, GEMMs | adaLN-modulated residual blocks; timestep sinusoidal embedder; the head is small enough to run per-frame |
| Scheduler | — | exact port of the fork's multistep update (math only; diffusers not required at runtime) |
| Acoustic/semantic tokenizers | SiLU, GEMMs, RMSNorm | causal streaming `SConv1d`/`SConvTranspose1d` with per-layer context caches (`context_size = (k−1)·dilation − (stride−1)`), depthwise convs, per-block layer-scale residuals, streaming decode/encode orchestration |
| Connectors | RMSNorm | `fc1 → RMSNorm(eps 1e-6) → fc2` SpeechConnector, acoustic(64→1536) + semantic(128→1536) additive embedding |
| Session | — | `hipengine/runtime/vibevoice_tts.py` owning LM caches (positive + negative), codec caches, per-request RNG, and the speech/text transition state machine |

Runtime stays torch-free; the oracle boundary is torch's job. Kernel ports
follow `docs/KERNELS.md` (lineage check + strict/production gates) and
`docs/EXECUTION-PROFILES.md` (control/ownership exactness; continuous-output
components need calibrated error measures plus downstream task checks — KL
floors do not apply to waveforms).

## Boundary gates (measured fixtures)

The oracle driver writes fixtures (`--fixtures`): generated token ids and the
final waveform per seed/script. Gates, per the review's task-gate section:

1. Reference features: acoustic connector output on the pinned voice prompt.
2. LM logits: argmax-equivalent constrained token stream (greedy; sampling is
   a separate gate) for the pinned script/seed.
3. Conditions: positive and negative last-hidden states at diffusion steps.
4. Diffusion: every solver step's prediction and the final latent (v-pred,
   cfg 1.3, 20 steps, fixed initial noise).
5. Waveform: decoder output per frame and concatenated, including join and
   final-flush sample counts.
6. Semantic feedback: encoder `.mean` features and the additive next-input
   embedding.

Raw waveform equality against PyTorch is a debugging oracle; production
acceptance additionally needs the intelligibility/quality battery from the
review doc before any precision or solver change.

## Provenance

- Oracle established 2026-09-15 on the W7900 host. Checkpoint files are not
  committed; the inventory JSON and fixtures references are.
- Fork sources are read at the pinned commit; any ported code records
  file + commit in the commit message per AGENTS.md.
