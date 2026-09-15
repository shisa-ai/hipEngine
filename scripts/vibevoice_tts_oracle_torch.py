"""VibeVoice-TTS torch oracle: capture reference outputs for hipEngine parity.

Runs the *reference* implementation -- the community fork
``vibevoice-community/VibeVoice`` -- under the pinned environment described in
``docs/MODEL-VIBEVOICE-TTS.md`` "Oracle environment", and freezes the tensors
hipEngine's port must reproduce.

This script must run under the oracle venv, not the hipEngine venv:

    PYTHONPATH=/home/lhl/VibeVoice-community \\
        /home/lhl/venvs/vibevoice-tts-oracle/bin/python \\
        scripts/vibevoice_tts_oracle_torch.py

Rebuild that venv with ``scripts/setup_vibevoice_tts_oracle_env.sh``. The pins
are in ``scripts/vibevoice_tts_oracle_requirements.txt``; the one that matters
most is ``transformers==4.51.3``, because 5.15.0 already claims model type
``vibevoice_acoustic_tokenizer`` and blocks the fork's import.

Fixtures written under ``tests/fixtures/vibevoice_tts/``, one set per request:

- ``<name>_request.json``: the request manifest -- script, speakers, seed,
  dtype, cfg_scale, solver steps, checkpoint and fork revisions, and the
  reference WAV hashes. Every other artifact is derived from this.
- ``<name>_reference.npz``: reference-audio encoder output (pre- and
  post-scale/bias), the connected embeddings, and the two scaling factors.
- ``<name>_lm.npz``: prompt ids, prefill logits and last hidden state, and the
  full generated token sequence.
- ``<name>_diffusion.npz``: per-call positive/negative conditions, the initial
  noise draw, per-solver-step eps and latent state, and the final pre-/post-scale
  latents.
- ``<name>_feedback.npz``: semantic feedback -- semantic features, acoustic and
  semantic embeddings, and their sum.
- ``<name>_audio.npz``: per-chunk decoder output and the concatenated waveform.

Randomness is recorded, not re-derived. A seed alone is not enough here: the
acoustic encoder samples under ``std_dist_type='gaussian'`` (two draws per
encode), ``sample_speech_tokens`` draws ``torch.randn(2 * n_tokens, 64)`` at the
*doubled* CFG batch size and discards the second half, and the decoder carries
streaming caches across steps. The script seeds the global RNG so a run is
repeatable, and separately saves every draw it observes.

Every capture is done by wrapping the reference module or method that performs
the work. No arithmetic is reimplemented here, so a fixture cannot silently
disagree with the reference by construction.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_OUT_DIR = Path("tests/fixtures/vibevoice_tts")
DEFAULT_FORK = "/home/lhl/VibeVoice-community"
SAMPLING_RATE = 24_000
DEFAULT_SEED = 20260915


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


def _git_revision(repo: Path) -> str:
    """HEAD of a git working tree, or a marker if it is not one."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else "<not-a-git-tree>"
    except Exception:
        return "<unknown>"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _np(t) -> np.ndarray:
    """Detach to a float32 CPU numpy array (bf16 has no numpy dtype)."""
    return t.detach().float().cpu().numpy()


def _npi(t) -> np.ndarray:
    """Detach to CPU numpy keeping the native dtype (ids, masks, bools)."""
    return t.detach().cpu().numpy()


class _Recorder:
    """Capture the reference generation loop's internals without reimplementing them.

    Each patch wraps a real module or method, records what passed through, and
    restores the original on exit. The per-call detail is capped so a long
    generation does not produce an unusable fixture; the summary tensors (final
    latent, waveform, semantic feedback) are always recorded.
    """

    def __init__(self, model, max_calls: int = 8):
        self.model = model
        self.inner = model.model
        self.max_calls = max_calls
        self.calls: list[dict] = []
        self.semantic: list[dict] = []
        self.chunks: list[np.ndarray] = []
        self._saved: dict = {}
        self._cur: dict | None = None
        self._noise: list = []
        self._randn_like: list = []

    # -- patching ---------------------------------------------------------
    def __enter__(self) -> "_Recorder":
        inner, model = self.inner, self.model
        self._saved = {
            "sample": model.sample_speech_tokens,
            "head_fwd": inner.prediction_head.forward,
            "sched_step": inner.noise_scheduler.step,
            "decode": inner.acoustic_tokenizer.decode,
            "sem_encode": inner.semantic_tokenizer.encode,
            "randn": __import__("torch").randn,
            "randn_like": __import__("torch").randn_like,
        }
        torch = __import__("torch")

        def _randn(*a, **kw):
            out = self._saved["randn"](*a, **kw)
            self._noise.append(_np(out))
            return out

        def _randn_like(*a, **kw):
            out = self._saved["randn_like"](*a, **kw)
            self._randn_like.append(_np(out))
            return out

        torch.randn = _randn
        torch.randn_like = _randn_like

        def sample_speech_tokens(condition, neg_condition, cfg_scale=3.0):
            self._noise = []
            self._cur = {
                "condition": _np(condition),
                "neg_condition": _np(neg_condition),
                "cfg_scale": float(cfg_scale),
                "eps": [], "speech": [], "timesteps": [],
            }
            latent = None
            try:
                latent = self._saved["sample"](condition, neg_condition, cfg_scale=cfg_scale)
            finally:
                # The returned latent is the final diffusion output, before the
                # decode-side ``/ scale - bias``. Keep it: it is the value the
                # port must match, and it is not recoverable from the eps trace.
                self._cur["speech_latent"] = (
                    _np(latent) if latent is not None else np.zeros(0, dtype=np.float32)
                )
                self._cur["initial_noise"] = (
                    self._noise[0] if self._noise else np.zeros(0, dtype=np.float32)
                )
                # Pin the solver schedule: class, step count and exact timesteps.
                sched = self.inner.noise_scheduler
                self._cur["scheduler_class"] = type(sched).__name__
                ts = getattr(sched, "timesteps", None)
                self._cur["scheduler_timesteps"] = (
                    _np(ts) if ts is not None else np.zeros(0, dtype=np.float32)
                )
                if len(self.calls) < self.max_calls:
                    self.calls.append(self._cur)
                self._cur = None
            return latent

        def head_fwd(x, timestep, condition=None, **kw):
            out = self._saved["head_fwd"](x, timestep, condition=condition, **kw)
            if self._cur is not None and len(self._cur["eps"]) < 200:
                self._cur["eps"].append(_np(out))
                self._cur["timesteps"].append(_np(timestep.reshape(-1)[:1]))
            return out

        def sched_step(eps, t, sample, **kw):
            out = self._saved["sched_step"](eps, t, sample, **kw)
            if self._cur is not None and len(self._cur["speech"]) < 200:
                self._cur["speech"].append(_np(out.prev_sample))
            return out

        def decode(latent, **kw):
            out = self._saved["decode"](latent, **kw)
            if self._cur is not None:
                self._cur.setdefault("scaled_latent", _np(latent))
            self.chunks.append(_np(out))
            return out

        def sem_encode(audio, **kw):
            out = self._saved["sem_encode"](audio, **kw)
            self.semantic.append({"features": _np(out.mean)})
            return out

        model.sample_speech_tokens = sample_speech_tokens
        inner.prediction_head.forward = head_fwd
        inner.noise_scheduler.step = sched_step
        inner.acoustic_tokenizer.decode = decode
        inner.semantic_tokenizer.encode = sem_encode
        return self

    def __exit__(self, *exc) -> None:
        torch = __import__("torch")
        torch.randn = self._saved["randn"]
        torch.randn_like = self._saved["randn_like"]
        self.model.sample_speech_tokens = self._saved["sample"]
        self.inner.prediction_head.forward = self._saved["head_fwd"]
        self.inner.noise_scheduler.step = self._saved["sched_step"]
        self.inner.acoustic_tokenizer.decode = self._saved["decode"]
        self.inner.semantic_tokenizer.encode = self._saved["sem_encode"]

    # -- packing ----------------------------------------------------------
    def diffusion_fixture(self) -> dict:
        out: dict = {"num_calls_recorded": np.array(len(self.calls))}
        for i, c in enumerate(self.calls):
            out[f"call{i}_condition"] = c["condition"]
            out[f"call{i}_neg_condition"] = c["neg_condition"]
            out[f"call{i}_initial_noise"] = c["initial_noise"]
            out[f"call{i}_speech_latent"] = c["speech_latent"]
            out[f"call{i}_scheduler_timesteps"] = c["scheduler_timesteps"]
            out[f"call{i}_scheduler_class"] = np.array(c["scheduler_class"])
            out[f"call{i}_cfg_scale"] = np.array(c["cfg_scale"], dtype=np.float32)
            if c["eps"]:
                out[f"call{i}_eps"] = np.stack(c["eps"])
                out[f"call{i}_timesteps"] = np.concatenate(c["timesteps"])
            if c["speech"]:
                out[f"call{i}_speech"] = np.stack(c["speech"])
            if "scaled_latent" in c:
                out[f"call{i}_scaled_latent"] = c["scaled_latent"]
        return out

    def feedback_fixture(self) -> dict:
        out: dict = {"num_calls_recorded": np.array(len(self.semantic))}
        for i, s in enumerate(self.semantic):
            out[f"call{i}_semantic_features"] = s["features"]
        return out


def _reference_fixture(model, speech, speech_masks, device, dtype, seed: int) -> dict:
    """Encode the reference audio and capture the scale/bias the fork derives from it.

    ``speech`` / ``speech_masks`` come from the processor, which owns resampling
    to 24 kHz. Reading the voice WAV directly is wrong: the shipped reference
    voices are 16 kHz.
    """
    import torch

    out: dict = {}
    speech = speech.to(device=device, dtype=dtype)
    speech_masks = speech_masks.to(device=device)
    out["ref_pcm"] = _np(speech)
    out["ref_speech_masks"] = speech_masks.detach().cpu().numpy()

    # The factors are checkpoint weights, not derived values: the state dict
    # carries ``model.speech_scaling_factor`` / ``model.speech_bias_factor`` as
    # bf16 scalars, and ``from_pretrained`` loads them over the ``NaN`` default
    # that the module registers at construction time. Read them back; do not
    # recompute them and do not reset them to NaN, which would poison every
    # downstream embedding.
    with torch.no_grad():
        draws: list[np.ndarray] = []
        orig_randn, orig_randn_like = torch.randn, torch.randn_like

        def _randn(*a, **kw):
            o = orig_randn(*a, **kw)
            draws.append(_np(o))
            return o

        def _randn_like(*a, **kw):
            o = orig_randn_like(*a, **kw)
            draws.append(_np(o))
            return o

        torch.randn, torch.randn_like = _randn, _randn_like
        try:
            torch.manual_seed(seed)
            enc = model.model.acoustic_tokenizer.encode(speech.unsqueeze(1))
            latents = enc.sample(dist_type=model.model.acoustic_tokenizer.std_dist_type)[0]
        finally:
            torch.randn, torch.randn_like = orig_randn, orig_randn_like

        out["encoder_mean"] = _np(enc.mean)
        out["encoder_std"] = _np(enc.std)
        out["latents"] = _np(latents)
        for i, d in enumerate(draws):
            out[f"encode_draw{i}"] = d

        # Drive the fork's own path so the connected embeddings are produced
        # exactly as in generation, then read the factors back.
        model._process_speech_inputs(speech, speech_masks, speech_type="audio")
        out["scaling_factor"] = np.array(float(model.model.speech_scaling_factor), dtype=np.float32)
        out["bias_factor"] = np.array(float(model.model.speech_bias_factor), dtype=np.float32)
        out["features_scaled"] = _np(
            (latents + model.model.speech_bias_factor.to(latents.device))
            * model.model.speech_scaling_factor.to(latents.device)
        )
        out["connected"] = _np(model.model.acoustic_connector(
            torch.from_numpy(out["features_scaled"]).to(device=device, dtype=dtype)
        ))
    return out


def _run_request(
    name: str,
    script: str,
    voice_paths: list[Path],
    *,
    model,
    processor,
    device: str,
    dtype,
    seed: int,
    cfg_scale: float,
    ddpm_steps: int,
    max_calls: int,
    logit_positions: int,
    out_dir: Path,
) -> dict:
    """Generate one request, capture its internals, and write its fixtures."""
    import torch

    torch.manual_seed(seed)
    model.set_ddpm_inference_steps(num_steps=ddpm_steps)

    inputs = processor(
        text=[script],
        # One flat list of voice paths per batch item. Passing a list-of-lists
        # silently produces one batch item per speaker instead of one item with
        # several speakers, and the reference audio then reflects only the
        # first voice -- which is easy to miss because generation still runs.
        voice_samples=[[str(p) for p in voice_paths]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    # -- reference features (processor-resampled, one pass over the audio) --
    ref = _reference_fixture(
        model, moved["speech_tensors"], moved["speech_masks"], device, dtype, seed
    )
    np.savez_compressed(out_dir / f"{name}_reference.npz", **ref)
    print(f"  wrote {name}_reference.npz (scale {float(ref['scaling_factor']):.6f}, "
          f"bias {float(ref['bias_factor']):.6f})")

    # -- LM prefill (explicit, deterministic) -----------------------------
    # Token ids and masks keep their integer/bool dtype; only floats go through
    # _np. Saving ids as float32 would round-trip but loses the type contract.
    lm: dict = {"input_ids": _npi(moved["input_ids"])}
    if "attention_mask" in moved:
        lm["attention_mask"] = _npi(moved["attention_mask"])
    for key in ("speech_masks", "speech_input_mask"):
        if key in moved:
            lm[key] = _npi(moved[key])
    # The reference waveform is already stored as ``ref_pcm`` in the reference
    # fixture; ``speech_tensors`` is the same array. Keep one copy so the two
    # cannot drift.
    with torch.no_grad():
        pre = model(**moved, return_dict=True)

    # Full-vocab logits for every prompt position is ~73 MB of float32 for a
    # 121-token prompt, which is too large to commit. Keep a few positions: the
    # start (prompt encoding) and the end (the token that starts generation).
    seq_len = pre.logits.shape[1]
    half = max(1, logit_positions // 2)
    idx = sorted(set(list(range(min(half, seq_len)))
                     + list(range(max(0, seq_len - (logit_positions - half)), seq_len))))
    lm["prefill_logits"] = _np(pre.logits[0, idx, :])
    lm["prefill_logit_positions"] = np.array(idx, dtype=np.int64)
    lm["prefill_last_hidden"] = _np(pre.last_hidden_state)

    # -- generation with capture ------------------------------------------
    with _Recorder(model, max_calls=max_calls) as rec:
        out = model.generate(
            **moved,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            is_prefill=True,
            show_progress_bar=False,
        )

    lm["generated_ids"] = _npi(out.sequences)
    np.savez_compressed(out_dir / f"{name}_lm.npz", **lm)
    print(f"  wrote {name}_lm.npz (prompt {lm['input_ids'].shape[1]} -> generated {lm['generated_ids'].shape[1]} ids)")

    diff = rec.diffusion_fixture()
    np.savez_compressed(out_dir / f"{name}_diffusion.npz", **diff)
    print(f"  wrote {name}_diffusion.npz ({len(rec.calls)} calls recorded)")

    np.savez_compressed(out_dir / f"{name}_feedback.npz", **rec.feedback_fixture())
    print(f"  wrote {name}_feedback.npz ({len(rec.semantic)} semantic encodes)")

    audio: dict = {}
    if rec.chunks:
        audio["chunks"] = np.stack(rec.chunks)
    pcm = out.speech_outputs[0]
    pcm = _np(pcm).reshape(-1) if pcm is not None else np.zeros(0, dtype=np.float32)
    audio["pcm"] = pcm
    audio["sample_rate"] = np.array(SAMPLING_RATE, dtype=np.int64)
    np.savez_compressed(out_dir / f"{name}_audio.npz", **audio)
    print(f"  wrote {name}_audio.npz ({pcm.size} samples = {pcm.size / SAMPLING_RATE:.2f} s)")

    if pcm.size == 0:
        raise SystemExit(f"{name}: generation produced no audio; fixture is not usable")

    return {
        "name": name,
        "script": script,
        "voices": [str(p) for p in voice_paths],
        "voice_sha256": [_sha256(p) for p in voice_paths],
        "num_speakers": int(ref["ref_pcm"].shape[0]),
        "cfg_scale": cfg_scale,
        "ddpm_inference_steps": ddpm_steps,
        "seed": seed,
        "prompt_tokens": int(lm["input_ids"].shape[1]),
        "generated_tokens": int(lm["generated_ids"].shape[1]),
        "logit_positions": int(lm["prefill_logit_positions"].size),
        "diffusion_calls_recorded": len(rec.calls),
        "semantic_encodes": len(rec.semantic),
        "pcm_samples": int(pcm.size),
        "pcm_seconds": round(float(pcm.size) / SAMPLING_RATE, 3),
        "pcm_rms": round(float(np.sqrt((pcm.astype(np.float64) ** 2).mean())), 6),
        "pcm_absmax": round(float(np.abs(pcm).max()), 6),
        "scaling_factor": float(ref["scaling_factor"]),
        "bias_factor": float(ref["bias_factor"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--fork-dir", type=Path, default=Path(DEFAULT_FORK))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--ddpm-steps", type=int, default=20)
    parser.add_argument("--max-diffusion-calls", type=int, default=8,
                        help="per-call diffusion detail cap; summaries are always written")
    parser.add_argument("--logit-positions", type=int, default=4,
                        help="prompt positions whose full-vocab logits are stored (default 4)")
    parser.add_argument("--only", default=None, choices=["single", "two"],
                        help="run one request only (default: both)")
    args = parser.parse_args()

    if str(args.fork_dir) not in sys.path:
        sys.path.insert(0, str(args.fork_dir))

    import torch
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but HIP reports no device")

    snap = _snapshot_dir(args.model)
    voices = args.fork_dir / "demo" / "voices"
    alice, carter = voices / "en-Alice_woman.wav", voices / "en-Carter_man.wav"
    for wav in (alice, carter):
        if not wav.exists():
            raise SystemExit(f"reference voice missing: {wav}")

    print(f"device={args.device} dtype={args.dtype} seed={args.seed} cfg={args.cfg_scale} ddpm={args.ddpm_steps}")
    processor = VibeVoiceProcessor.from_pretrained(args.model)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device, attn_implementation="sdpa",
    )
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    requests = []
    if args.only != "two":
        requests.append((
            "single",
            "Speaker 1: The quick brown fox jumps over the lazy dog.",
            [alice],
        ))
    if args.only != "single":
        requests.append((
            "two",
            "Speaker 1: I heard there is big news in text to speech lately.\n"
            "Speaker 2: Yes, and it runs locally on this machine.",
            [alice, carter],
        ))

    manifest = {
        "model_id": args.model,
        "model_revision": snap.name,
        "fork_dir": str(args.fork_dir),
        "fork_revision": _git_revision(args.fork_dir),
        "device": args.device,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "sampling_rate": SAMPLING_RATE,
        "requests": [],
    }

    for name, script, voices in requests:
        print(f"[{name}] generating")
        manifest["requests"].append(_run_request(
            name, script, voices,
            model=model, processor=processor, device=args.device, dtype=dtype,
            seed=args.seed, cfg_scale=args.cfg_scale, ddpm_steps=args.ddpm_steps,
            max_calls=args.max_diffusion_calls, logit_positions=args.logit_positions,
            out_dir=args.out_dir,
        ))

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    for r in manifest["requests"]:
        print(f"  {r['name']}: {r['generated_tokens']} ids, {r['pcm_seconds']} s audio, "
              f"rms {r['pcm_rms']}, scale {r['scaling_factor']:.6f}, bias {r['bias_factor']:.6f}")


if __name__ == "__main__":
    main()
