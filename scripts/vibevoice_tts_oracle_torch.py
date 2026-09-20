"""VibeVoice-TTS torch oracle: capture reference outputs for hipEngine parity.

Runs the *reference* implementation -- the community fork
``vibevoice-community/VibeVoice`` -- under the pinned environment described in
``docs/model-cards/MODEL-VIBEVOICE-TTS.md`` "Oracle environment", and freezes the tensors
hipEngine's port must reproduce.

This script must run under the oracle venv, not the hipEngine venv:

    PYTHONPATH=/home/lhl/VibeVoice-community \\
        /home/lhl/venvs/vibevoice-tts-oracle/bin/python \\
        scripts/vibevoice_tts_oracle_torch.py

Rebuild that venv with ``scripts/setup_vibevoice_tts_oracle_env.sh``. The pins
are in ``scripts/vibevoice_tts_oracle_requirements.txt``; the one that matters
most is ``transformers==4.51.3``, because 5.15.0 already claims model type
``vibevoice_acoustic_tokenizer`` and blocks the fork's import.

Provenance is enforced, not just recorded. Both the processor and the model are
loaded from the exact local snapshot named by ``--model-revision`` (default: the
revision pinned in the doc), so the manifest cannot name weights other than the
ones actually loaded. The fork must be a clean git tree at one revision; a dirty
tree is refused unless ``--allow-dirty-fork`` is passed explicitly.

Fixtures written under ``tests/fixtures/vibevoice_tts/``, one set per request:

- ``manifest.json``: checkpoint snapshot (resolved == expected), fork revision
  and dirty state, environment versions, and per-request summary statistics.
- ``<name>_reference.npz``: the reference-audio encoding **of the generation
  pass** -- the encode whose connected embeddings actually entered the language
  model for the generated audio: encoder mean/std, sampled latents, every random
  draw that pass consumed, the scaled features and connected embeddings, and the
  two scaling factors.
- ``<name>_lm.npz``: prompt ids and masks, the prefill pass's own speech encode
  (draws, latents, features, connected embeddings), full-vocab logits at a few
  prompt positions, last hidden state, and the full generated token sequence.
- ``<name>_diffusion.npz``: **every** diffusion call (no cap): conditions, the
  initial noise draw and how many draws the call consumed, per-solver-step eps
  and latent state, the final pre-scale latent, the post-scale decoder input
  (``scaled_latent``) with its batch sample indices, whether the decoder ran,
  and the scheduler pin (class, config JSON, exact timesteps).
- ``<name>_feedback.npz``: per diffusion call, the semantic features, the
  acoustic and semantic connector embeddings, their recorded sum, and the batch
  sample indices -- the complete next-input embedding the loop fed back.
- ``<name>_audio.npz``: every decoder chunk in order and the concatenated
  waveform.

Cache resets are recorded in the diffusion fixture (``reset<i>_role`` /
``reset<i>_sample_indices``): the acoustic and semantic streaming caches are
zeroed at every speech-end token, and the recorder labels which cache by
correlating the object identity with the decode/semantic-encode calls.

Randomness is recorded, not re-derived. A seed alone is not enough here: the
acoustic encoder samples under ``std_dist_type='gaussian'``, the diffusion head
draws its initial noise, and the decoder carries streaming caches across steps.
The script seeds the global RNG once per request; every draw on that single
stream is logged in order, and each capture labels the slice of the log that
produced it. There are exactly two speech-encoding passes per request -- the
explicit prefill and generate's first step -- and each fixture states which one
it holds. A capture that cannot be attributed is an error, not a silent gap.

Every capture is done by wrapping the reference module or method that performs
the work. No model arithmetic is reimplemented here, so a fixture cannot
silently disagree with the reference by construction. (The one exception is the
feedback sum, which is the addition of two recorded tensors, stored alongside
its addends.)
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
# The checkpoint revision pinned in docs/model-cards/MODEL-VIBEVOICE-TTS.md.
DEFAULT_MODEL_REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"
# Bumped whenever fixture layout or capture semantics change. Schema 2 adds:
# enforced snapshot loading, complete per-call diffusion capture (scaled latent,
# sample indices, decoder flag), per-pass speech-encode attribution, feedback
# embeddings with their sum, scheduler config, and cache-reset events.
ORACLE_SCHEMA = 2


def _snapshot_dir(model_id: str, expect_revision: str | None) -> Path:
    """Resolve the local HF snapshot for ``model_id`` (download must exist).

    With ``expect_revision`` set, only that snapshot is acceptable: the returned
    path is the exact directory both loads use, so the revision recorded in the
    manifest is the revision whose weights were loaded.
    """
    base = Path.home() / f".cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots"
    if expect_revision:
        snap = base / expect_revision
        if not snap.is_dir():
            raise SystemExit(
                f"pinned snapshot {expect_revision} not found under {base}; "
                "download that revision (snapshot_download with revision=...) first"
            )
    else:
        snapshots = sorted(glob.glob(str(base / "*")))
        if not snapshots:
            raise SystemExit(f"no local snapshot found for {model_id}; snapshot_download it first")
        snap = Path(snapshots[-1])
    if not list(snap.glob("model-*.safetensors")) and not (snap / "model.safetensors").exists():
        raise SystemExit(f"snapshot {snap} has no weights yet (download incomplete?)")
    return snap


def _git_state(repo: Path) -> tuple[str, bool]:
    """(HEAD, dirty) of a git working tree; a non-tree counts as dirty."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
        )
        if head.returncode != 0 or status.returncode != 0:
            return "<not-a-git-tree>", True
        return head.stdout.strip(), bool(status.stdout.strip())
    except Exception:
        return "<unknown>", True


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


def _sched_config(sched) -> str:
    """The scheduler's full config as canonical JSON (solver order/algorithm etc.)."""
    cfg = getattr(sched, "config", None)
    out: dict = {}
    if cfg is not None:
        for k, v in dict(cfg).items():
            # ``_use_default_values`` is built from a set difference upstream, so
            # its list order varies per Python process; sort it or the fixture
            # is not reproducible.
            if k == "_use_default_values" and isinstance(v, (list, tuple, set)):
                v = sorted(str(x) for x in v)
            out[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return json.dumps(out, sort_keys=True)


class _Recorder:
    """Capture the reference generation loop's internals without reimplementing them.

    Each patch wraps a real module or method, records what passed through, and
    restores the original on exit. Every diffusion call and every decoder chunk
    is retained -- the fixtures must be able to replay the complete latent->PCM
    chain, so there is no per-call cap.
    """

    def __init__(self, model):
        self.model = model
        self.inner = model.model
        self.phase = "prefill"
        self.calls: list[dict] = []          # one per sample_speech_tokens
        self.passes: list[dict] = []         # one per _process_speech_inputs
        self.encodes: list[dict] = []        # acoustic_tokenizer.encode outputs
        self.enc_samples: list[dict] = []    # encoder-output .sample() results
        self.decodes: list[dict] = []        # one per acoustic decode
        self.feedback: list[dict] = []       # semantic encodes + connector embeds
        self.resets: list[dict] = []         # streaming-cache set_to_zero events
        self.chunks: list[np.ndarray] = []   # every decode output, in order
        self.draws: list[np.ndarray] = []    # global ordered torch.randn log
        self._saved: dict = {}
        self._cur: dict | None = None
        self._in_call = False
        self._call_idx = -1
        self._in_prompt = 0
        self._cache_roles: dict[int, str] = {}

    # -- patching ---------------------------------------------------------
    def __enter__(self) -> "_Recorder":
        import torch
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerEncoderOutput,
            VibeVoiceTokenizerStreamingCache,
        )

        inner, model = self.inner, self.model
        self._saved = {
            "randn": torch.randn,
            "randn_like": torch.randn_like,
            "sample": model.sample_speech_tokens,
            "speech_inputs": model._process_speech_inputs,
            "encode": inner.acoustic_tokenizer.encode,
            "enc_sample": VibeVoiceTokenizerEncoderOutput.sample,
            "head_fwd": inner.prediction_head.forward,
            "sched_step": inner.noise_scheduler.step,
            "decode": inner.acoustic_tokenizer.decode,
            "sem_encode": inner.semantic_tokenizer.encode,
            "ac_conn_fwd": inner.acoustic_connector.forward,
            "sem_conn_fwd": inner.semantic_connector.forward,
            "set_to_zero": VibeVoiceTokenizerStreamingCache.set_to_zero,
        }

        def _randn(*a, **kw):
            out = self._saved["randn"](*a, **kw)
            self.draws.append(_np(out))
            return out

        def _randn_like(*a, **kw):
            out = self._saved["randn_like"](*a, **kw)
            self.draws.append(_np(out))
            return out

        torch.randn = _randn
        torch.randn_like = _randn_like

        def sample_speech_tokens(condition, neg_condition, cfg_scale=3.0):
            call = {
                "condition": _np(condition),
                "neg_condition": _np(neg_condition),
                "cfg_scale": float(cfg_scale),
                "draw_start": len(self.draws),
                "eps": [], "timesteps": [], "speech": [],
                "decoder_called": False,
            }
            self.calls.append(call)
            self._call_idx = len(self.calls) - 1
            self._cur = call
            self._in_call = True
            latent = None
            try:
                latent = self._saved["sample"](condition, neg_condition, cfg_scale=cfg_scale)
            finally:
                self._in_call = False
                # The returned latent is the final diffusion output, before the
                # decode-side ``/ scale - bias``. Keep it: it is the value the
                # port must match, and it is not recoverable from the eps trace.
                call["speech_latent"] = (
                    _np(latent) if latent is not None else np.zeros(0, dtype=np.float32)
                )
                window = self.draws[call["draw_start"]:]
                call["initial_noise"] = (
                    window[0] if window else np.zeros(0, dtype=np.float32)
                )
                call["noise_draw_count"] = len(window)
                # Pin the solver: class, full config, and the exact timesteps.
                sched = self.inner.noise_scheduler
                call["scheduler_class"] = type(sched).__name__
                ts = getattr(sched, "timesteps", None)
                call["scheduler_timesteps"] = (
                    _np(ts) if ts is not None else np.zeros(0, dtype=np.float32)
                )
                call["scheduler_config"] = _sched_config(sched)
                # _cur deliberately stays set: the decoder hook runs after this
                # returns and must still find its call to attach the decoder
                # input. It is cleared by the decode hook.
                self._cur = call
            return latent

        def speech_inputs(speech_tensors, speech_masks, speech_type="audio"):
            enc_start, samp_start, draw_start = (
                len(self.encodes), len(self.enc_samples), len(self.draws),
            )
            self._in_prompt += 1
            try:
                features, connected = self._saved["speech_inputs"](
                    speech_tensors, speech_masks, speech_type,
                )
            finally:
                self._in_prompt -= 1
            self.passes.append({
                "phase": self.phase,
                "speech_type": speech_type,
                "features_scaled": _np(features),
                "connected": _np(connected),
                "draw_start": draw_start, "draw_end": len(self.draws),
                "enc_start": enc_start, "enc_end": len(self.encodes),
                "samp_start": samp_start, "samp_end": len(self.enc_samples),
            })
            return features, connected

        def encode(audio, **kw):
            out = self._saved["encode"](audio, **kw)
            std = out.std
            self.encodes.append({
                "mean": _np(out.mean),
                "std": _np(std) if hasattr(std, "detach") else np.float32(std),
            })
            return out

        def enc_sample(self_, dist_type="fix"):
            out = self._saved["enc_sample"](self_, dist_type)
            self.enc_samples.append({"dist_type": dist_type, "latents": _np(out[0])})
            return out

        def head_fwd(x, timestep, condition=None, **kw):
            out = self._saved["head_fwd"](x, timestep, condition=condition, **kw)
            if self._in_call and self._cur is not None and len(self._cur["eps"]) < 200:
                self._cur["eps"].append(_np(out))
                self._cur["timesteps"].append(_np(timestep.reshape(-1)[:1]))
            return out

        def sched_step(eps, t, sample, **kw):
            out = self._saved["sched_step"](eps, t, sample, **kw)
            if self._in_call and self._cur is not None and len(self._cur["speech"]) < 200:
                self._cur["speech"].append(_np(out.prev_sample))
            return out

        def decode(latent, **kw):
            out = self._saved["decode"](latent, **kw)
            cache = kw.get("cache")
            if cache is not None:
                self._cache_roles[id(cache)] = "acoustic"
            rec = {
                "scaled_latent": _np(latent),
                "chunk": _np(out),
                "sample_indices": (
                    _npi(kw["sample_indices"]) if kw.get("sample_indices") is not None
                    else np.zeros(0, dtype=np.int64)
                ),
                "cache_id": id(cache) if cache is not None else None,
            }
            self.decodes.append(rec)
            self.chunks.append(rec["chunk"])
            if self._cur is not None:
                self._cur["decoder_called"] = True
                self._cur["scaled_latent"] = rec["scaled_latent"]
                self._cur["decode_sample_indices"] = rec["sample_indices"]
                self._cur = None
            return out

        def sem_encode(audio, **kw):
            out = self._saved["sem_encode"](audio, **kw)
            cache = kw.get("cache")
            if cache is not None:
                self._cache_roles[id(cache)] = "semantic"
            self.feedback.append({
                "call_idx": max(self._call_idx, 0),
                "kind": "semantic_features",
                "features": _np(out.mean),
                "sample_indices": (
                    _npi(kw["sample_indices"]) if kw.get("sample_indices") is not None
                    else np.zeros(0, dtype=np.int64)
                ),
                "cache_id": id(cache) if cache is not None else None,
            })
            return out

        def ac_conn_fwd(x, **kw):
            out = self._saved["ac_conn_fwd"](x, **kw)
            if not self._in_prompt:
                self.feedback.append({
                    "call_idx": max(self._call_idx, 0),
                    "kind": "acoustic_embed", "embed": _np(out), "input": _np(x),
                })
            return out

        def sem_conn_fwd(x, **kw):
            out = self._saved["sem_conn_fwd"](x, **kw)
            if not self._in_prompt:
                self.feedback.append({
                    "call_idx": max(self._call_idx, 0),
                    "kind": "semantic_embed", "embed": _np(out), "input": _np(x),
                })
            return out

        def set_to_zero(self_, sample_indices):
            self._saved["set_to_zero"](self_, sample_indices)
            self.resets.append({
                "cache_id": id(self_),
                "sample_indices": _npi(sample_indices),
            })

        model.sample_speech_tokens = sample_speech_tokens
        model._process_speech_inputs = speech_inputs
        inner.acoustic_tokenizer.encode = encode
        VibeVoiceTokenizerEncoderOutput.sample = enc_sample
        inner.prediction_head.forward = head_fwd
        inner.noise_scheduler.step = sched_step
        inner.acoustic_tokenizer.decode = decode
        inner.semantic_tokenizer.encode = sem_encode
        inner.acoustic_connector.forward = ac_conn_fwd
        inner.semantic_connector.forward = sem_conn_fwd
        VibeVoiceTokenizerStreamingCache.set_to_zero = set_to_zero
        return self

    def __exit__(self, *exc) -> None:
        import torch
        from vibevoice.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerEncoderOutput,
            VibeVoiceTokenizerStreamingCache,
        )
        torch.randn = self._saved["randn"]
        torch.randn_like = self._saved["randn_like"]
        self.model.sample_speech_tokens = self._saved["sample"]
        self.model._process_speech_inputs = self._saved["speech_inputs"]
        self.inner.acoustic_tokenizer.encode = self._saved["encode"]
        VibeVoiceTokenizerEncoderOutput.sample = self._saved["enc_sample"]
        self.inner.prediction_head.forward = self._saved["head_fwd"]
        self.inner.noise_scheduler.step = self._saved["sched_step"]
        self.inner.acoustic_tokenizer.decode = self._saved["decode"]
        self.inner.semantic_tokenizer.encode = self._saved["sem_encode"]
        self.inner.acoustic_connector.forward = self._saved["ac_conn_fwd"]
        self.inner.semantic_connector.forward = self._saved["sem_conn_fwd"]
        VibeVoiceTokenizerStreamingCache.set_to_zero = self._saved["set_to_zero"]

    # -- consistency ------------------------------------------------------
    def check_complete(self) -> None:
        """Refuse to pack an incoherent capture."""
        n_calls, n_decodes = len(self.calls), len(self.decodes)
        n_sem = len([f for f in self.feedback if f["kind"] == "semantic_features"])
        if n_calls != n_decodes:
            raise SystemExit(
                f"capture incomplete: {n_calls} diffusion calls but {n_decodes} decoder calls"
            )
        if n_calls != n_sem:
            raise SystemExit(
                f"capture incomplete: {n_calls} diffusion calls but {n_sem} semantic encodes"
            )
        missing = [i for i, c in enumerate(self.calls) if not c["decoder_called"]]
        if missing:
            raise SystemExit(f"decoder input not captured for diffusion calls {missing}")

    # -- packing ----------------------------------------------------------
    def diffusion_fixture(self) -> dict:
        out: dict = {
            "num_calls_recorded": np.array(len(self.calls)),
            "num_cache_resets": np.array(len(self.resets)),
            "oracle_schema": np.array(ORACLE_SCHEMA, dtype=np.int64),
        }
        for i, c in enumerate(self.calls):
            p = f"call{i}_"
            out[p + "condition"] = c["condition"]
            out[p + "neg_condition"] = c["neg_condition"]
            out[p + "cfg_scale"] = np.array(c["cfg_scale"], dtype=np.float32)
            out[p + "initial_noise"] = c["initial_noise"]
            out[p + "noise_draw_count"] = np.array(c["noise_draw_count"], dtype=np.int64)
            out[p + "speech_latent"] = c["speech_latent"]
            out[p + "decoder_called"] = np.array(c["decoder_called"])
            if "scaled_latent" in c:
                out[p + "scaled_latent"] = c["scaled_latent"]
                out[p + "decode_sample_indices"] = c["decode_sample_indices"]
            if c["eps"]:
                out[p + "eps"] = np.stack(c["eps"])
                out[p + "timesteps"] = np.concatenate(c["timesteps"])
            if c["speech"]:
                out[p + "speech"] = np.stack(c["speech"])
            out[p + "scheduler_class"] = np.array(c["scheduler_class"])
            out[p + "scheduler_config"] = np.array(c["scheduler_config"])
            out[p + "scheduler_timesteps"] = c["scheduler_timesteps"]
        for i, r in enumerate(self.resets):
            role = self._cache_roles.get(r["cache_id"], "unknown")
            out[f"reset{i}_role"] = np.array(role)
            out[f"reset{i}_sample_indices"] = r["sample_indices"]
        return out

    def feedback_fixture(self) -> dict:
        by_call: dict[int, list[dict]] = {}
        for f in self.feedback:
            by_call.setdefault(f["call_idx"], []).append(f)
        out: dict = {
            "num_calls_recorded": np.array(
                len([f for f in self.feedback if f["kind"] == "semantic_features"])
            ),
            "oracle_schema": np.array(ORACLE_SCHEMA, dtype=np.int64),
        }
        for ci in sorted(by_call):
            events = by_call[ci]
            for kind in ("semantic_features", "acoustic_embed", "semantic_embed"):
                group = [e for e in events if e["kind"] == kind]
                if not group:
                    raise SystemExit(f"feedback capture incomplete: no {kind} for call {ci}")
                for j, e in enumerate(group):
                    suffix = "" if len(group) == 1 else str(j)
                    if kind == "semantic_features":
                        out[f"call{ci}_semantic_features{suffix}"] = e["features"]
                        out[f"call{ci}_sample_indices{suffix}"] = e["sample_indices"]
                    else:
                        out[f"call{ci}_{kind}{suffix}"] = e["embed"]
            a = out.get(f"call{ci}_acoustic_embed")
            s = out.get(f"call{ci}_semantic_embed")
            if a is not None and s is not None:
                # Packing arithmetic only: the sum the loop fed back, stored
                # alongside both addends so the port can re-derive it.
                out[f"call{ci}_feedback_sum"] = a + s
        return out


def _pass_keys(fixture: dict, prefix: str, rec: "_Recorder", p: dict) -> None:
    """Fill ``fixture`` with one speech-encode pass's actual inputs and draws."""
    fixture[f"{prefix}phase"] = np.array(p["phase"])
    for j, e in enumerate(rec.encodes[p["enc_start"]:p["enc_end"]]):
        fixture[f"{prefix}encode{j}_mean"] = e["mean"]
        fixture[f"{prefix}encode{j}_std"] = e["std"]
    for j, s in enumerate(rec.enc_samples[p["samp_start"]:p["samp_end"]]):
        fixture[f"{prefix}encode{j}_sample_dist"] = np.array(s["dist_type"])
        fixture[f"{prefix}encode{j}_latents"] = s["latents"]
    for j, d in enumerate(rec.draws[p["draw_start"]:p["draw_end"]]):
        fixture[f"{prefix}encode_draw{j}"] = d
    fixture[f"{prefix}features_scaled"] = p["features_scaled"]
    fixture[f"{prefix}connected"] = p["connected"]


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

    ref_pcm = _np(moved["speech_tensors"])
    ref_masks = _npi(moved["speech_masks"])

    # The factors are checkpoint weights, not derived values: the state dict
    # carries ``model.speech_scaling_factor`` / ``model.speech_bias_factor`` as
    # bf16 scalars, and ``from_pretrained`` loads them over the ``NaN`` default
    # that the module registers at construction time. Read them back and refuse
    # NaN: a model built from config without a weight load would otherwise
    # poison every downstream embedding silently.
    scale = model.model.speech_scaling_factor
    bias = model.model.speech_bias_factor
    if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(bias)):
        raise SystemExit(f"{name}: speech_scaling/bias_factor is not finite; weights did not load")

    # One connected execution trace: the recorder is live for the explicit
    # prefill and the whole generation, so every random draw and every speech
    # encode is attributed to the pass that consumed it.
    with _Recorder(model) as rec:
        rec.phase = "prefill"
        with torch.no_grad():
            pre = model(**moved, return_dict=True)

        rec.phase = "generate"
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

    rec.check_complete()
    passes = rec.passes
    phases = [p["phase"] for p in passes]
    if phases != ["prefill", "generate"]:
        raise SystemExit(
            f"{name}: expected exactly one prefill and one generation prompt pass, got {phases}"
        )
    pre_pass, gen_pass = passes

    # -- reference features = the generation pass's speech encode -------------
    # These are the connected embeddings that actually entered the language
    # model and produced the generated audio, with the draws that made them.
    ref: dict = {
        "ref_pcm": ref_pcm,
        "ref_speech_masks": ref_masks,
        "scaling_factor": np.array(float(scale), dtype=np.float32),
        "bias_factor": np.array(float(bias), dtype=np.float32),
        "oracle_schema": np.array(ORACLE_SCHEMA, dtype=np.int64),
    }
    _pass_keys(ref, "", rec, gen_pass)
    np.savez_compressed(out_dir / f"{name}_reference.npz", **ref)
    print(f"  wrote {name}_reference.npz (scale {float(ref['scaling_factor']):.6f}, "
          f"bias {float(ref['bias_factor']):.6f}, phase {gen_pass['phase']}, "
          f"draws {gen_pass['draw_end'] - gen_pass['draw_start']})")

    # -- LM prefill and generation -------------------------------------------
    # Token ids and masks keep their integer/bool dtype; only floats go through
    # _np. Saving ids as float32 would round-trip but loses the type contract.
    lm: dict = {"input_ids": _npi(moved["input_ids"])}
    if "attention_mask" in moved:
        lm["attention_mask"] = _npi(moved["attention_mask"])
    for key in ("speech_masks", "speech_input_mask"):
        if key in moved:
            lm[key] = _npi(moved[key])
    # The prefill logits were produced from the prefill pass's speech encode;
    # record that encode beside them so the fixture is self-contained.
    _pass_keys(lm, "prefill_", rec, pre_pass)
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
    lm["generated_ids"] = _npi(out.sequences)
    np.savez_compressed(out_dir / f"{name}_lm.npz", **lm)
    print(f"  wrote {name}_lm.npz (prompt {lm['input_ids'].shape[1]} -> generated {lm['generated_ids'].shape[1]} ids)")

    diff = rec.diffusion_fixture()
    np.savez_compressed(out_dir / f"{name}_diffusion.npz", **diff)
    print(f"  wrote {name}_diffusion.npz ({len(rec.calls)} calls, "
          f"{len(rec.decodes)} decoder inputs, {len(rec.resets)} cache resets)")

    np.savez_compressed(out_dir / f"{name}_feedback.npz", **rec.feedback_fixture())
    n_sem = len([f for f in rec.feedback if f["kind"] == "semantic_features"])
    print(f"  wrote {name}_feedback.npz ({n_sem} semantic encodes with embeddings)")

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
        "decoder_inputs_recorded": len(rec.decodes),
        "semantic_encodes": n_sem,
        "cache_resets": len(rec.resets),
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
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION,
                        help="checkpoint snapshot that must be loaded (default: the doc pin; "
                             "pass an empty string to allow whatever snapshot is cached)")
    parser.add_argument("--fork-dir", type=Path, default=Path(DEFAULT_FORK))
    parser.add_argument("--allow-dirty-fork", action="store_true",
                        help="permit a fork working tree with uncommitted changes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--ddpm-steps", type=int, default=20)
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

    snap = _snapshot_dir(args.model, args.model_revision or None)
    fork_head, fork_dirty = _git_state(args.fork_dir)
    if fork_dirty and not args.allow_dirty_fork:
        raise SystemExit(
            f"fork working tree is dirty (HEAD {fork_head}); commit or stash it, or pass "
            "--allow-dirty-fork to record the dirty state and continue"
        )

    voices = args.fork_dir / "demo" / "voices"
    alice, carter = voices / "en-Alice_woman.wav", voices / "en-Carter_man.wav"
    for wav in (alice, carter):
        if not wav.exists():
            raise SystemExit(f"reference voice missing: {wav}")

    print(f"device={args.device} dtype={args.dtype} seed={args.seed} cfg={args.cfg_scale} ddpm={args.ddpm_steps}")
    print(f"snapshot={snap.name} fork={fork_head} fork_dirty={fork_dirty}")
    # Load from the resolved snapshot directory itself: the revision recorded in
    # the manifest is then the revision whose weights were actually loaded,
    # independent of the hub cache's own resolution.
    processor = VibeVoiceProcessor.from_pretrained(str(snap))
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        str(snap), torch_dtype=dtype, device_map=args.device, attn_implementation="sdpa",
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
        "oracle_schema": ORACLE_SCHEMA,
        "model_id": args.model,
        "model_revision": snap.name,
        "model_revision_expected": args.model_revision or None,
        "model_revision_matches": (args.model_revision or None) in (None, snap.name),
        "fork_dir": str(args.fork_dir),
        "fork_revision": fork_head,
        "fork_dirty": fork_dirty,
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
            logit_positions=args.logit_positions, out_dir=args.out_dir,
        ))

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    for r in manifest["requests"]:
        print(f"  {r['name']}: {r['generated_tokens']} ids, {r['pcm_seconds']} s audio, "
              f"rms {r['pcm_rms']}, scale {r['scaling_factor']:.6f}, bias {r['bias_factor']:.6f}")


if __name__ == "__main__":
    main()
