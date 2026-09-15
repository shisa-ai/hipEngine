#!/usr/bin/env python3
"""VibeVoice-1.5B TTS oracle driver (PyTorch, pinned environment).

Runs the pinned community-fork oracle (VibeVoice-community @ 952326dd, pinned
checkpoint revision c00898d) on the W7900 and dumps the artifacts the hipEngine
implementation is validated against:

- an inventory JSON: geometry, speech token ids, scheduler settings, scaling
  factors, and the per-module weight inventory;
- optional tensor fixtures (final latent, waveform, generated token ids) for
  the boundary gates in docs/MODEL-VIBEVOICE-TTS.md.

This script imports torch and the fork's package; it is an oracle driver, not
runtime code. Run it inside the pinned oracle environment (see the worklog
entry for the exact venv), never through the engine venv.

Example:
  ~/venvs/vibevoice-oracle/bin/python scripts/vibevoice_tts_oracle_tts.py \
      --model /models/vibevoice/VibeVoice-1.5B \
      --script /tmp/vv/smoke.txt --speaker Alice --seed 20260915 \
      --inventory benchmarks/results/2026-09-15-vibevoice-tts-oracle-inventory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FORK = "/home/lhl/VibeVoice-community"


def _load_fork(fork_path: str) -> None:
    fork = Path(fork_path)
    if not (fork / "vibevoice").is_dir():
        raise SystemExit(f"community fork not found at {fork}")
    sys.path.insert(0, str(fork))


def build_inventory(model, processor, args) -> dict:
    """Geometry, tokens, scheduler and weight inventory from a loaded model."""
    config = model.config
    acoustic_cfg = config.acoustic_tokenizer_config
    semantic_cfg = config.semantic_tokenizer_config
    head_cfg = config.diffusion_head_config
    decoder_cfg = config.decoder_config

    tokenizer = processor.tokenizer
    speech_ids = {
        "speech_start_token": "<|vision_start|>",
        "speech_start_id": tokenizer.speech_start_id,
        "speech_end_token": "<|vision_end|>",
        "speech_end_id": tokenizer.speech_end_id,
        "speech_diffusion_token": "<|vision_pad|>",
        "speech_diffusion_id": tokenizer.speech_diffusion_id,
        "eos_token_id": tokenizer.eos_id,
    }

    scheduler = model.model.noise_scheduler
    scheduler.set_timesteps(head_cfg.ddpm_num_inference_steps)
    scheduler_state = {
        "class": type(scheduler).__name__,
        "num_train_timesteps": scheduler.config.num_train_timesteps,
        "beta_schedule": scheduler.config.beta_schedule,
        "prediction_type": scheduler.config.prediction_type,
        "solver_order": scheduler.config.solver_order,
        "num_inference_steps": head_cfg.ddpm_num_inference_steps,
        "timesteps": [int(t) for t in scheduler.timesteps.tolist()],
        "sigmas": [float(s) for s in scheduler.sigmas.tolist()],
    }

    scaling = model.model.speech_scaling_factor
    bias = model.model.speech_bias_factor
    if scaling is None or bool(scaling != scaling):  # NaN check
        raise SystemExit(
            "speech_scaling_factor is unset/NaN in the checkpoint; the doc "
            "makes missing scale values a loader error"
        )

    weight_inventory: dict[str, dict] = {}
    for name, param in model.state_dict().items():
        entry = weight_inventory.setdefault(
            name.split(".")[0], {"tensors": 0, "elements": 0}
        )
        entry["tensors"] += 1
        entry["elements"] += param.numel()

    return {
        "checkpoint": {
            "repo": "microsoft/VibeVoice-1.5B",
            "revision": "c00898d257e6b46004e3e2866a47534085fb685a",
            "path": str(args.model),
            "dtype": str(next(model.parameters()).dtype),
        },
        "fork": {
            "repo": "vibevoice-community/VibeVoice",
            "commit": "952326ddb264062466a888cf32a5b2f4e803e16e",
            "path": str(args.fork),
        },
        "decoder": {
            "model_type": decoder_cfg.model_type,
            "hidden_size": decoder_cfg.hidden_size,
            "num_hidden_layers": decoder_cfg.num_hidden_layers,
            "num_attention_heads": decoder_cfg.num_attention_heads,
            "num_key_value_heads": decoder_cfg.num_key_value_heads,
            "head_dim": decoder_cfg.hidden_size // decoder_cfg.num_attention_heads,
            "intermediate_size": decoder_cfg.intermediate_size,
            "vocab_size": decoder_cfg.vocab_size,
            "rope_theta": decoder_cfg.rope_theta,
            "rms_norm_eps": decoder_cfg.rms_norm_eps,
            "tie_word_embeddings": decoder_cfg.tie_word_embeddings,
            "max_position_embeddings": decoder_cfg.max_position_embeddings,
        },
        "acoustic_tokenizer": {
            "vae_dim": acoustic_cfg.vae_dim,
            "encoder_depths": acoustic_cfg.encoder_depths,
            "encoder_ratios": acoustic_cfg.encoder_ratios,
            "encoder_n_filters": acoustic_cfg.encoder_n_filters,
            "decoder_ratios": acoustic_cfg.decoder_ratios,
            "decoder_n_filters": acoustic_cfg.decoder_n_filters,
            "total_upsample": int(_product(acoustic_cfg.encoder_ratios)),
            "fix_std": acoustic_cfg.fix_std,
            "std_dist_type": acoustic_cfg.std_dist_type,
            "pad_mode": acoustic_cfg.pad_mode,
            "causal": acoustic_cfg.causal,
        },
        "semantic_tokenizer": {
            "vae_dim": semantic_cfg.vae_dim,
            "encoder_depths": semantic_cfg.encoder_depths,
            "encoder_ratios": semantic_cfg.encoder_ratios,
            "total_upsample": int(_product(semantic_cfg.encoder_ratios)),
            "fix_std": semantic_cfg.fix_std,
            "std_dist_type": semantic_cfg.std_dist_type,
        },
        "diffusion_head": {
            "hidden_size": head_cfg.hidden_size,
            "head_layers": head_cfg.head_layers,
            "head_ffn_ratio": head_cfg.head_ffn_ratio,
            "ffn_dim": int(head_cfg.hidden_size * head_cfg.head_ffn_ratio),
            "latent_size": head_cfg.latent_size,
            "prediction_type": head_cfg.prediction_type,
            "ddpm_num_steps": head_cfg.ddpm_num_steps,
            "ddpm_num_inference_steps": head_cfg.ddpm_num_inference_steps,
            "ddpm_beta_schedule": head_cfg.ddpm_beta_schedule,
            "rms_norm_eps": head_cfg.rms_norm_eps,
        },
        "latent_scaling": {
            "speech_scaling_factor": float(scaling),
            "speech_bias_factor": float(bias),
            "decode_rule": "scaled_latent = latent / scale - bias",
        },
        "speech_tokens": speech_ids,
        "scheduler": scheduler_state,
        "weight_inventory": weight_inventory,
    }


def _product(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/models/vibevoice/VibeVoice-1.5B")
    parser.add_argument("--fork", type=str, default=DEFAULT_FORK)
    parser.add_argument("--script", type=str, default=None, help="txt script to generate")
    parser.add_argument("--speaker", type=str, default="Alice")
    parser.add_argument("--voice", type=str, default=None, help="wav path overriding the preset")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--inventory", type=str, default=None, help="output inventory JSON path")
    parser.add_argument("--fixtures", type=str, default=None, help="output fixture .npz path")
    args = parser.parse_args()

    import torch

    _load_fork(args.fork)
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    )
    model.eval()
    model.set_ddpm_inference_steps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    processor = VibeVoiceProcessor.from_pretrained(args.model)

    if args.inventory:
        inventory = build_inventory(model, processor, args)
        inv_path = Path(args.inventory)
        inv_path.parent.mkdir(parents=True, exist_ok=True)
        inv_path.write_text(json.dumps(inventory, indent=2) + "\n")
        print(f"wrote inventory: {inv_path}")

    if args.script:
        voice = args.voice
        if voice is None:
            voice = str(Path(args.fork) / "demo" / "voices" / f"en-{args.speaker}_woman.wav")
            if not Path(voice).exists():
                voices = sorted((Path(args.fork) / "demo" / "voices").glob("*.wav"))
                voice = str(voices[0])
        script_text = Path(args.script).read_text()
        inputs = processor(
            text=[script_text],
            voice_samples=[[voice]],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=args.cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={"do_sample": False},
                is_prefill=True,
            )
        audio = output.speech_outputs[0]
        sequences = output.sequences
        print(
            f"generated {len(sequences[0])} tokens, "
            f"audio {audio.shape[-1] / 24000.0:.2f}s" if audio is not None else "no audio"
        )

        if args.fixtures:
            import numpy as np

            fx = {
                "token_ids": sequences[0].to(torch.int64).cpu().numpy(),
            }
            if audio is not None:
                fx["waveform_f32"] = audio.to(torch.float32).cpu().numpy()
            fx_path = Path(args.fixtures)
            fx_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(fx_path, **fx)
            print(f"wrote fixtures: {fx_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
