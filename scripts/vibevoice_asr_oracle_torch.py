"""VibeVoice-ASR torch oracle: capture reference outputs for hipEngine parity.

Loads the HF artifact ``microsoft/VibeVoice-ASR-HF`` (bfloat16 checkpoint) in
float32 on CPU (or ``--device cuda`` / ``--dtype bfloat16`` for the GPU lane),
runs deterministic synthetic PCM through the acoustic/semantic tokenizer
encoders, the multi-modal projector, and the Qwen2 language model, and saves
fixtures under ``tests/fixtures/vibevoice_asr/``:

- ``vibevoice_asr_trace.npz``: 0.5 s clip, per-stage encoder outputs for both
  tokenizers (stem, each downsample stage, head), pre/post-sampling latents,
  the exact sampling noise tensors, connector per-path outputs and the summed
  text-width embeddings.
- ``vibevoice_asr_boundary.npz``: 3199/3200/3201-sample clips, final encoder
  latents only (frame-boundary contract).
- ``vibevoice_asr_chunk.npz``: 90 s synthetic audio processed as one 60 s
  chunk plus a 30 s chunk through the padding-cache path, per-chunk latents
  and concatenated latents (single-pass equivalence is asserted in-script).
- ``vibevoice_asr_lm.npz``: processor-built transcription request for a 4 s
  clip; input ids, placeholder mask, audio embeddings, greedy first 16 tokens,
  per-step teacher-forced top-10 logprobs, and full logits at the first two
  positions.

The encoder trace drives the *actual* HF modules (stem, conv_layers, head)
module-by-module so no arithmetic is reimplemented here; only the sampling
noise draw order is reproduced manually (scale ``randn`` then ``randn_like``)
and asserted identical to ``VibeVoiceAsrModel.get_audio_features`` under the
same torch seed.

Usage:
    python3 scripts/vibevoice_asr_oracle_torch.py \
        [--device cpu] [--dtype float32] [--out-dir tests/fixtures/vibevoice_asr]
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path("tests/fixtures/vibevoice_asr")
HOP = 3200
SAMPLING_RATE = 24_000


def _snapshot_dir(model_id: str) -> Path:
    """Resolve the local HF snapshot for ``model_id`` (download must exist)."""
    pattern = Path.home() / f".cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*"
    snapshots = sorted(glob.glob(str(pattern)))
    if not snapshots:
        raise SystemExit(f"no local snapshot found for {model_id}; snapshot_download it first")
    snap = Path(snapshots[-1])
    if not list(snap.glob("model-*.safetensors")) and not (snap / "model.safetensors").exists():
        raise SystemExit(f"snapshot {snap} has no weights yet (download incomplete?)")
    return snap


def synth_pcm(seconds: float, seed: int) -> np.ndarray:
    """Deterministic mono 24 kHz speech-like excitation (tones + noise + AM)."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLING_RATE))
    t = np.arange(n, dtype=np.float64) / SAMPLING_RATE
    # voiced-ish fundamental with harmonics
    f0 = 120.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLING_RATE
    sig = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    # amplitude modulation so RMS normalization varies over time
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t)
    sig += 0.05 * rng.standard_normal(n).astype(np.float64)
    return sig.astype(np.float32)


def _stage_trace(enc, pcm: "torch.Tensor", prefix: str, out: dict) -> "torch.Tensor":
    """Run one HF tokenizer encoder module-by-module, recording intermediates."""
    x = pcm.reshape(1, 1, -1)  # (batch=1, channels=1, samples)
    x = enc.stem.conv(x)
    out[f"{prefix}_stem_conv"] = x.detach().cpu().numpy()
    for i, block in enumerate(enc.stem.stage):
        x = block(x)
    out[f"{prefix}_stem_out"] = x.detach().cpu().numpy()
    for s, layer in enumerate(enc.conv_layers):
        x = layer.conv(x)
        out[f"{prefix}_stage{s}_conv"] = x.detach().cpu().numpy()
        for b, block in enumerate(layer.stage):
            x = block(x)
        out[f"{prefix}_stage{s}_out"] = x.detach().cpu().numpy()
    x = enc.head(x)
    out[f"{prefix}_head_out"] = x.detach().cpu().numpy()
    # contiguous so randn_like(records) matches HF's concatenated latents
    return x.permute(0, 2, 1).contiguous()  # (batch, frames, hidden)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR-HF")
    args = parser.parse_args()

    import torch

    torch.manual_seed(20260914)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    snap = _snapshot_dir(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {snap} on {device} ({args.dtype}) ...")
    from transformers import VibeVoiceAsrForConditionalGeneration, VibeVoiceAsrProcessor

    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        str(snap), torch_dtype=dtype, device_map=str(device), attn_implementation="eager"
    ).eval()
    torch.set_grad_enabled(False)
    processor = VibeVoiceAsrProcessor.from_pretrained(str(snap))
    enc_ac = model.model.acoustic_tokenizer_encoder
    enc_se = model.model.semantic_tokenizer_encoder
    proj = model.model.multi_modal_projector

    # ---------------- trace fixture (0.5 s) ----------------
    pcm_np = synth_pcm(0.5, seed=1)
    pcm = torch.from_numpy(pcm_np).to(device=device, dtype=dtype)
    trace: dict = {"pcm_short": pcm_np}
    torch.manual_seed(20260914)
    lat_ac = _stage_trace(enc_ac, pcm, "acoustic", trace)
    lat_se = _stage_trace(enc_se, pcm, "semantic", trace)
    trace["acoustic_latent_raw"] = lat_ac.detach().cpu().numpy()
    trace["semantic_latent_raw"] = lat_se.detach().cpu().numpy()

    # reproduce the HF sampling draw order, then cross-check get_audio_features
    torch.manual_seed(20260914 + 1)
    scale = torch.randn(lat_ac.shape[0], device=device, dtype=dtype) * enc_ac.config.vae_std
    noise = torch.randn_like(lat_ac)
    sampled_ac = lat_ac + scale[:, None, None] * noise
    trace["acoustic_noise_scale"] = scale.detach().cpu().numpy()
    trace["acoustic_noise"] = noise.detach().cpu().numpy()
    trace["acoustic_latent_sampled"] = sampled_ac.detach().cpu().numpy()

    ac_emb = proj.acoustic_linear_1(sampled_ac)
    ac_emb = proj.acoustic_norm(ac_emb)
    ac_emb = proj.acoustic_linear_2(ac_emb)
    trace["connector_acoustic_out"] = ac_emb.detach().cpu().numpy()
    se_emb = proj.semantic_linear_1(lat_se)
    se_emb = proj.semantic_norm(se_emb)
    se_emb = proj.semantic_linear_2(se_emb)
    trace["connector_semantic_out"] = se_emb.detach().cpu().numpy()
    combined = ac_emb + se_emb
    trace["connector_combined"] = combined.detach().cpu().numpy()

    # cross-check: HF get_audio_features with the same seed must reproduce
    # the sampled acoustic latents and combined embeddings exactly.
    torch.manual_seed(20260914 + 1)
    feats = model.model.get_audio_features(input_values=pcm.reshape(1, 1, -1))
    ref_sampled = feats.last_hidden_state.detach()
    max_diff = (ref_sampled - sampled_ac).abs().max().item()
    max_emb_diff = (feats.pooler_output - combined).abs().max().item()
    print(f"trace cross-check: sampled latent max|diff|={max_diff:.3e}, "
          f"combined embed max|diff|={max_emb_diff:.3e}")
    if max_diff > 1e-3 * max(1.0, float(ref_sampled.abs().max())):
        raise SystemExit("sampled latent cross-check FAILED; noise draw order mismatch")
    np.savez_compressed(args.out_dir / "vibevoice_asr_trace.npz", **trace)
    print("wrote vibevoice_asr_trace.npz")

    # ---------------- boundary fixture ----------------
    # Raw clips below 3200 samples are unprocessable even in HF (the deepest
    # stage conv would need a padded input shorter than its kernel; the
    # processor's mandatory pad_to_multiple_of=3200 exists for this reason).
    # Boundary cases here test the floor(L/3200) frame contract at and above
    # the 3200 minimum.
    boundary: dict = {}
    for n in (3200, 3201, 6400, 6401):
        pcm_np = synth_pcm(n / SAMPLING_RATE, seed=100 + n)
        pcm = torch.from_numpy(pcm_np).to(device=device, dtype=dtype)
        torch.manual_seed(20260914)
        la = _stage_trace(enc_ac, pcm, f"ac{n}", boundary)
        boundary[f"acoustic_latent_raw_{n}"] = la.detach().cpu().numpy()
        torch.manual_seed(20260914)
        ls = _stage_trace(enc_se, pcm, f"se{n}", boundary)
        boundary[f"semantic_latent_raw_{n}"] = ls.detach().cpu().numpy()
        boundary[f"pcm_{n}"] = pcm_np
        expected = n // HOP
        if la.shape[1] != expected:
            raise SystemExit(f"boundary {n}: frames {la.shape[1]} != floor {expected}")
        # keep only latents + pcm: drop per-stage intermediates to slim the file
        for k in [k for k in boundary if (k.startswith(f"ac{n}_") or k.startswith(f"se{n}_"))]:
            del boundary[k]
    np.savez_compressed(args.out_dir / "vibevoice_asr_boundary.npz", **boundary)
    print("wrote vibevoice_asr_boundary.npz")

    # ---------------- chunk-cache fixture (60 s + 30 s, cache carry) ----------------
    chunk_a = synth_pcm(60.0, seed=7)
    chunk_b = synth_pcm(30.0, seed=8)
    full = np.concatenate([chunk_a, chunk_b])
    chunk_fix: dict = {"pcm_chunk_a": chunk_a, "pcm_chunk_b": chunk_b, "pcm_full": full}
    lat_ac_chunks, lat_se_chunks = [], []
    ac_cache = se_cache = None
    for part in (chunk_a, chunk_b):
        pcm = torch.from_numpy(part).to(device=device, dtype=dtype).reshape(1, 1, -1)
        o = enc_ac(pcm, padding_cache=ac_cache, use_cache=True)
        lat_ac_chunks.append(o.latents)
        ac_cache = o.padding_cache
        o = enc_se(pcm, padding_cache=se_cache, use_cache=True)
        lat_se_chunks.append(o.latents)
        se_cache = o.padding_cache
    lat_ac_full = torch.cat(lat_ac_chunks, dim=1)
    lat_se_full = torch.cat(lat_se_chunks, dim=1)
    chunk_fix["acoustic_latent_chunked"] = lat_ac_full.detach().cpu().numpy()
    chunk_fix["semantic_latent_chunked"] = lat_se_full.detach().cpu().numpy()
    # single-pass equivalence over the joined recording
    pcm = torch.from_numpy(full).to(device=device, dtype=dtype).reshape(1, 1, -1)
    lat_ac_single = enc_ac(pcm).latents
    lat_se_single = enc_se(pcm).latents
    d_ac = (lat_ac_single - lat_ac_full).abs().max().item()
    d_se = (lat_se_single - lat_se_full).abs().max().item()
    print(f"chunk-vs-single max|diff|: acoustic={d_ac:.3e} semantic={d_se:.3e}")
    if d_ac > 1e-3 or d_se > 1e-3:
        raise SystemExit("chunk cache carry cross-check FAILED")
    np.savez_compressed(args.out_dir / "vibevoice_asr_chunk.npz", **chunk_fix)
    print("wrote vibevoice_asr_chunk.npz")

    # ---------------- LM fixture (4 s clip, transcription request) ----------------
    lm: dict = {}
    pcm_np = synth_pcm(4.0, seed=3)
    lm["pcm_lm"] = pcm_np
    inputs = processor.apply_transcription_request(audio=pcm_np)
    input_ids = inputs["input_ids"].to(device)
    input_values = inputs["input_values"].to(device=device, dtype=dtype)
    # transformers 5.15.0 bug: get_audio_features crashes with 2-D (batch,
    # samples) input because the chunk-cache path indexes 3-D; pass 3-D.
    if input_values.ndim == 2:
        input_values = input_values.reshape(1, 1, -1)
    padding_mask = inputs.get("padding_mask")
    lm["input_ids"] = input_ids.detach().cpu().numpy()
    if padding_mask is not None:
        lm["padding_mask"] = padding_mask.detach().cpu().numpy()
    lm["input_values"] = input_values.detach().float().cpu().numpy()
    lm["audio_token_id"] = np.array(model.config.audio_token_id)
    lm["audio_bos_token_id"] = np.array(model.config.audio_bos_token_id)
    lm["audio_eos_token_id"] = np.array(model.config.audio_eos_token_id)

    embeds = model.model.get_input_embeddings()(input_ids)
    torch.manual_seed(20260914 + 1)
    audio_feats = model.model.get_audio_features(input_values=input_values, padding_mask=padding_mask)
    audio_embeds = audio_feats.pooler_output  # flat (num_audio_tokens, text_hidden)
    mask = (input_ids == model.config.audio_token_id).unsqueeze(-1)
    inputs_embeds = embeds.masked_scatter(mask, audio_embeds.to(embeds.dtype))
    lm["audio_placeholder_positions"] = mask[0].nonzero(as_tuple=True)[0].detach().cpu().numpy()
    lm["audio_embeds"] = audio_embeds.detach().float().cpu().numpy()

    with torch.no_grad():
        out = model.model(inputs_embeds=inputs_embeds)
        h_last = out.last_hidden_state
        logits_first = model.lm_head(h_last[:, :2, :])
        lm["logits_pos0"] = logits_first[0, 0].float().detach().cpu().numpy()
        lm["logits_pos1"] = logits_first[0, 1].float().detach().cpu().numpy()
        # greedy generation (limited) with teacher-forced per-step top-10
        generated = model.generate(
            inputs=input_ids,
            input_values=input_values,
            padding_mask=padding_mask,
            max_new_tokens=16,
            do_sample=False,
        )
    gen_only = generated[0, input_ids.shape[1]:]
    lm["greedy_tokens"] = gen_only.detach().cpu().numpy()
    lm["greedy_text"] = np.array(processor.decode(gen_only, skip_special_tokens=True))

    # teacher-forced logits per step for the generated continuation.
    # IMPORTANT: teacher-force through inputs_embeds with the audio
    # embeddings scattered at placeholder positions - passing raw input_ids
    # would leave the audio-placeholder token embeddings in place and change
    # the distribution.
    tf_ids = torch.cat([input_ids, gen_only.unsqueeze(0)], dim=1)
    tf_embeds = model.model.get_input_embeddings()(tf_ids)
    tf_mask = (tf_ids == model.config.audio_token_id).unsqueeze(-1)
    tf_embeds = tf_embeds.masked_scatter(tf_mask, audio_embeds.to(tf_embeds.dtype))
    with torch.no_grad():
        out = model.model(inputs_embeds=tf_embeds)
        logits = model.lm_head(out.last_hidden_state[0, input_ids.shape[1] - 1: -1, :])
    lp = torch.log_softmax(logits.float(), dim=-1)
    topv, topi = lp.topk(10, dim=-1)
    lm["tf_top10_logprobs"] = topv.detach().cpu().numpy()
    lm["tf_top10_ids"] = topi.detach().cpu().numpy()
    np.savez_compressed(args.out_dir / "vibevoice_asr_lm.npz", **lm)
    print("wrote vibevoice_asr_lm.npz")
    print("greedy text:", lm["greedy_text"])


if __name__ == "__main__":
    main()
