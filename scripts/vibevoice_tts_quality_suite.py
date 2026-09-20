"""Unassisted multi-request quality suite for the VibeVoice-TTS session.

Numerical agreement with the oracle is not available as acceptance evidence for
every request: the two-speaker fixture's first diffusion solve is chaotic, so its
late trajectory is unreproducible by construction (see
``docs/model-cards/MODEL-VIBEVOICE-TTS.md``). This suite covers those requests the way the
review that prompted it asked for -- by qualifying the audio that comes out
rather than by widening a tolerance.

What it does per request, with nothing injected:

1. builds the prompt in-tree from the script (``build_tts_prompt``), so the
   request is one the fixtures never recorded;
2. encodes the speaker references and samples the VAE with the session's own
   generator, and runs every diffusion frame with the session's own noise and the
   session's own negative-LM branch -- no recorded operand is fed in;
3. caps generation from the request's declared ``max_audio_seconds`` rather than
   from the oracle's token count, and records ``finish_reason`` so a truncated
   utterance is reported as truncated;
4. transcribes the result with the in-tree VibeVoice-ASR lane and scores
   intelligibility (word error rate), turn attribution (the transcript's speaker
   labels against the script's turns), missing or repeated speech, duration, and
   level and silence.

The thresholds below are declared constants. They are not derived from a run:
the pinned single-speaker request transcribes word-perfectly, so the
intelligibility gate is set where clean synthetic speech should sit, not where
this suite happens to land.

Usage:
    uv run --with jiwer --with transformers python \\
        scripts/vibevoice_tts_quality_suite.py [--phase all|synthesize|score] [--seeds 1,2]

``--phase score`` re-reads the PCM a previous ``--phase synthesize`` wrote, so a
failure in scoring does not require re-synthesising. ``jiwer`` and
``transformers`` are needed only for the scoring phase; they are imported there.
``--seeds`` runs the whole suite at several base seeds and reports which checks
are stable across them, because a request that comes out with the right number of
voices half the time is not a passing request.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUESTS = ROOT / "benchmarks" / "prompts" / "vibevoice-tts-quality.json"
DEFAULT_FIXTURES = ROOT / "tests" / "fixtures" / "vibevoice_tts"
DEFAULT_OUT = ROOT / "benchmarks" / "results" / "2026-09-15-gfx1151-vibevoice-tts-quality-suite.json"
DEFAULT_WORK = ROOT / ".vibevoice-tts-quality"

TTS_MODEL_ID = "microsoft/VibeVoice-1.5B"
#: The TTS snapshot ships no tokenizer, so prompts are tokenized with the sibling
#: ASR checkpoint's tokenizer. tests/test_unit_vibevoice_tts_prompt.py pins that
#: substitution by reproducing both oracle fixtures token-for-token.
TOKENIZER_MODEL_ID = "microsoft/VibeVoice-ASR-HF"
ASR_MODEL_ID = "microsoft/VibeVoice-ASR-HF"

SAMPLE_RATE = 24000
FRAME_SAMPLES = 3200  # one diffusion frame of output audio
CFG_SCALE = 1.3
TRANSCRIBE_SEED = 20260914

#: ~150 words per minute, used only to place a plausibility band on the output
#: duration. Speaking rate varies by a factor of two between speakers, so the
#: band is deliberately wide and a violation means "the audio is nowhere near the
#: length of its text", not "the pacing is off".
WORDS_PER_SECOND = 2.5
DURATION_MIN_RATIO = 0.5
DURATION_MAX_RATIO = 2.5

#: Clean synthetic speech that the same model family transcribes should be
#: near-perfect; the ASR lane's own LibriSpeech gate is 1.4-2%. One word in ten
#: is the point at which a listener is reading the transcript rather than
#: hearing the sentence.
WER_GATE = 0.10

#: A gap this long inside an utterance means speech went missing, not a pause.
MAX_INTERNAL_SILENCE_SECONDS = 1.5
SILENCE_AMPLITUDE = 1e-3

#: Below this the output is too quiet to be speech; above 1.0 it clips.
MIN_RMS = 0.005
MAX_ABS = 1.0

#: A join artifact is a sample-to-sample step at a chunk boundary the waveform's own
#: dynamics do not produce, so the gate is a multiple of the signal's interior p99.9
#: step rather than an absolute level.
JOIN_CLICK_STEP_RATIO = 4.0

#: A repeated 4-gram is the classic sampling failure; once is already audible.
NGRAM_SIZE = 4
MAX_REPEATED_NGRAMS = 0

#: Voice attribution windows. The acoustic encoder emits one latent per 3200
#: samples, so a one-second window holds seven frames and the half-second hop
#: gives a readable assignment sequence over a three-second utterance.
ATTRIBUTION_WINDOW_SECONDS = 1.0
ATTRIBUTION_HOP_SECONDS = 0.5
#: A window counts towards attribution only when one reference leads the other by
#: this cosine margin. One window at an utterance's onset can be a near tie
#: (measured 0.305 against 0.345 on the 4-turn script) and a single ambiguous
#: window must not decide a turn. The verdict is the same at 0.05 and 0.15.
ATTRIBUTION_MARGIN = 0.10
#: Below this many decided windows the assignment is too sparse to conclude.
ATTRIBUTION_MIN_DECIDED_WINDOWS = 3


def _reference_embeddings(session, pcms):
    """Unit-normalized mean encoder latents for the reference voices.

    ``encode_reference`` is the deterministic pass (no VAE sampling), so this is
    reproducible and not noise-limited.
    """
    out = []
    for pcm in pcms:
        latent = session.frontend.encode_reference(np.asarray(pcm, dtype=np.float32))
        mean = latent.mean(axis=0)
        out.append(mean / max(float(np.linalg.norm(mean)), 1e-9))
    return out


def _voice_assignment(session, audio, references):
    """Nearest reference voice per window, by encoder-latent cosine similarity.

    This is the attribution instrument the suite gates on, and it is deliberately
    not the ASR's speaker labels. Measured on these requests, the ASR reports a
    single speaker for two-speaker scripts that demonstrably contain both voices
    -- for the 2-turn script on all three seeds, and for the long 2-turn script
    on two of three -- while the encoder assignment makes a clean single
    transition between the two references. The encoder is also the mechanism the
    model itself uses to condition on a voice, so "this window is closer to the
    reference it was conditioned on" is a statement about the model's own voice
    conditioning rather than about a second model's diarization.
    """
    sr = SAMPLE_RATE
    window = int(ATTRIBUTION_WINDOW_SECONDS * sr)
    hop = int(ATTRIBUTION_HOP_SECONDS * sr)
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
    assignment = []
    similarities = []
    start = 0
    while start + window <= pcm.size:
        latent = session.frontend.encode_reference(pcm[start : start + window])
        mean = latent.mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-9)
        cosines = [float(np.dot(mean, reference)) for reference in references]
        similarities.append([round(value, 4) for value in cosines])
        assignment.append(int(np.argmax(cosines)))
        start += hop
    return assignment, similarities


def _runs(decided: list[int]) -> list[int]:
    """Collapse a per-window assignment into its sequence of speaker runs.

    A gate that compares only the first and last confidently classified window
    cannot see a lost speaker change in the middle: a four-turn script that came
    back as one voice, then the other, then the first, then the second would still
    open and close on the right voices. Comparing run sequences does see it.
    """
    out: list[int] = []
    for speaker in decided:
        if not out or out[-1] != speaker:
            out.append(speaker)
    return out


def _frame_token_cap(max_audio_seconds: float) -> int:
    """Token budget from a declared audio budget, not from the oracle.

    Two tokens of overhead (the speech-start and speech-end markers) plus one
    diffusion token per 3200 samples of output audio.
    """
    frames = math.ceil(float(max_audio_seconds) * SAMPLE_RATE / FRAME_SAMPLES)
    return 2 + frames


def _true_waveform(padded: np.ndarray, expected_frames: int, name: str) -> np.ndarray:
    """Trim a batch-padded reference back to its own samples.

    ``two_reference.npz`` stores both speakers right zero padded to the longest
    one, because the fork encodes every voice of a request in a single batched
    call. Feeding the padded rows back in would re-pad every speaker to the
    longest *padded* row and double the connected-row count. The frame count
    implied by the trimmed length is checked against the recorded mask, so a
    trim that lands in the wrong place fails here instead of silently building a
    different request.
    """
    from hipengine.runtime.vibevoice_encoder import reference_frame_count

    flat = np.asarray(padded, dtype=np.float32).reshape(-1)
    nonzero = np.nonzero(flat)[0]
    if not nonzero.size:
        raise ValueError(f"{name}: reference waveform is all zeros")
    trimmed = flat[: int(nonzero.max()) + 1]
    frames = reference_frame_count(trimmed.size)
    if frames != expected_frames:
        raise ValueError(
            f"{name}: trimmed reference implies {frames} frames, the fixture records "
            f"{expected_frames}; the padding assumption is wrong"
        )
    return trimmed


def _reference_pcms(fixtures: Path, reference_set: dict, names: list[str]):
    """Per-speaker true waveforms and their voice-prompt token counts."""
    path = fixtures / reference_set["fixture"]
    if not path.is_file():
        raise SystemExit(f"fixture not present: {path}")
    with np.load(path) as data:
        padded = np.asarray(data["ref_pcm"], dtype=np.float32)
        masks = np.asarray(data["ref_speech_masks"], dtype=bool)
    counts = [int(mask.sum()) for mask in masks]
    if len(counts) != len(reference_set["voices"]):
        raise SystemExit(
            f"{path.name} holds {len(counts)} references but the request set lists "
            f"{len(reference_set['voices'])} voices"
        )
    pcms = [
        _true_waveform(padded[index], counts[index], names[index])
        for index in range(len(counts))
    ]
    return pcms, counts


def _request_seed(base_seed: int, index: int) -> int:
    """A distinct seed per request, so a request is independent of its neighbours."""
    return int(base_seed) + index


def synthesize(
    requests, suite, fixtures: Path, model_id: str, work: Path, seeds: list[int]
) -> list[dict]:
    """Run every request unassisted and write its PCM plus synthesis facts.

    The session is reseeded before each request. Its generator is shared by the
    voice-prompt draw and every diffusion frame, so without that a request's
    audio depends on how many requests ran before it -- which is how the same
    two-speaker script produced one voice in a full run and two when run alone.
    """
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.vibevoice_tts_prompt import (
        build_tts_prompt,
        load_tts_tokenizer,
    )
    from hipengine.loading.vibevoice_tts_session import load_vibevoice_tts_session
    from hipengine.runtime.vibevoice_tts_session import VibevoiceTtsSession

    tokenizer = load_tts_tokenizer(resolve_model_path(TOKENIZER_MODEL_ID))
    weights = load_vibevoice_tts_session(resolve_model_path(model_id))
    session = VibevoiceTtsSession(weights, max_context=1024)
    records: list[dict] = []
    reference_cache: dict[str, tuple[list[np.ndarray], list[int], list[np.ndarray]]] = {}
    # The two-voice pair, used to classify every request -- including the
    # single-speaker ones -- so the instrument has a non-vacuous control.
    pair_embeddings: list[np.ndarray] | None = None
    try:
        for base_seed in seeds:
            seed_dir = work / f"seed{base_seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            for index, request in enumerate(requests):
                key = request["reference_set"]
                if key not in reference_cache:
                    reference_set = suite["reference_sets"][key]
                    pcms, counts = _reference_pcms(
                        fixtures, reference_set, reference_set["voices"]
                    )
                    # The reference embeddings do not depend on the seed.
                    reference_cache[key] = (pcms, counts, _reference_embeddings(session, pcms))
                pcms, counts, reference_embeddings = reference_cache[key]
                if pair_embeddings is None:
                    pair_pcms, _ = _reference_pcms(
                        fixtures, suite["reference_sets"]["two"], suite["reference_sets"]["two"]["voices"]
                    )
                    pair_embeddings = _reference_embeddings(session, pair_pcms)
                prompt = build_tts_prompt(request["script"], counts, tokenizer)
                request_seed = _request_seed(base_seed, index)
                session.reseed(request_seed)

                start = time.perf_counter()
                # No noise/neg injection: the session's own generator and its own
                # negative-LM branch are the path under test.
                _, connected = session.voice_prompt_rows_multi(pcms)
                if connected.shape[0] != sum(prompt.speech_input_mask):
                    raise SystemExit(
                        f"{request['name']}: {connected.shape[0]} connected rows for "
                        f"{sum(prompt.speech_input_mask)} speech positions"
                    )
                rows = session.build_prompt_rows(
                    prompt.input_ids,
                    np.asarray(prompt.speech_input_mask, dtype=bool),
                    connected,
                )
                cap = _frame_token_cap(request["max_audio_seconds"])
                result = session.generate(rows, cfg_scale=CFG_SCALE, max_new_tokens=cap)
                elapsed = time.perf_counter() - start

                audio = (
                    np.concatenate(
                        [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in result.chunks]
                    )
                    if result.chunks
                    else np.zeros(0, dtype=np.float32)
                )
                pcm_path = seed_dir / f"{request['name']}.npy"
                np.save(pcm_path, audio)
                assignment, similarities = _voice_assignment(
                    session, audio, reference_embeddings
                )
                pair_assignment, pair_similarities = _voice_assignment(
                    session, audio, pair_embeddings
                )
                turn_speakers = [
                    int(line.split(":", 1)[0].split()[-1])
                    for line in request["script"].strip().splitlines()
                    if line.strip()
                ]
                turn_speakers = [
                    speaker - 1 if min(turn_speakers) > 0 else speaker
                    for speaker in turn_speakers
                ]

                record = {
                    "name": request["name"],
                    "base_seed": int(base_seed),
                    "request_seed": request_seed,
                    "pcm": str(pcm_path.relative_to(ROOT) if pcm_path.is_relative_to(ROOT)
                               else pcm_path.resolve()),
                    "reference_set": request["reference_set"],
                    "script": request["script"],
                    "turns": int(request["turns"]),
                    "turn_speakers": turn_speakers,
                    "voice_prompt_token_counts": counts,
                    "prompt_tokens": len(prompt),
                    "max_audio_seconds": float(request["max_audio_seconds"]),
                    "token_cap": cap,
                    "generated_tokens": len(result.ids),
                    "diffusion_frames": len(result.chunks),
                    "chunk_samples": [int(chunk.size) for chunk in result.chunks],
                    "finish_reason": result.finish_reason,
                    "synthesis_seconds": round(elapsed, 3),
                    "output_audio_seconds": round(float(audio.size) / SAMPLE_RATE, 4),
                    "voice_assignment": assignment,
                    "voice_assignment_cosines": similarities,
                    "voice_assignment_pair": pair_assignment,
                    "voice_assignment_pair_cosines": pair_similarities,
                }
                records.append(record)
                print(
                    f"synthesized seed {base_seed} {request['name']}: "
                    f"{len(result.ids)} tokens, {len(result.chunks)} frames, "
                    f"{record['output_audio_seconds']:.3f} s, {result.finish_reason}",
                    flush=True,
                )
    finally:
        session.close()
    return records


def _level_checks(audio: np.ndarray) -> dict:
    if not audio.size:
        return {"rms": 0.0, "absmax": 0.0, "longest_silence_seconds": 0.0}
    silent = np.abs(audio) < SILENCE_AMPLITUDE
    longest = 0
    run = 0
    for is_silent in silent:
        run = run + 1 if is_silent else 0
        longest = max(longest, run)
    return {
        "rms": round(float(np.sqrt(np.mean(audio**2))), 6),
        "absmax": round(float(np.abs(audio).max()), 6),
        "longest_silence_seconds": round(longest / SAMPLE_RATE, 3),
    }


def _normalized_words(text: str) -> list[str]:
    """Lowercase alphanumeric words, for repetition detection only."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _repeated_ngrams(words: list[str], size: int = NGRAM_SIZE) -> list[str]:
    seen: dict[tuple[str, ...], int] = {}
    for index in range(len(words) - size + 1):
        gram = tuple(words[index : index + size])
        seen[gram] = seen.get(gram, 0) + 1
    return [" ".join(gram) for gram, count in seen.items() if count > 1]


def _expected_words(script: str) -> list[str]:
    text = " ".join(
        line.split(":", 1)[1] if ":" in line else line for line in script.strip().splitlines()
    )
    return _normalized_words(text)


def _evaluator_floor(engine, fixtures: Path, manifest: dict) -> dict:
    """Measure the ASR lane on audio that is known good.

    ``single_audio.npz`` and ``two_audio.npz`` hold the *reference implementation's*
    own PCM for the pinned requests, so transcribing them isolates the evaluator:
    every word is there, and any WER is the ASR lane's own. Without this, a request's
    WER cannot be attributed -- a small non-zero value could be the TTS lane or the
    transcript. The measured floor is recorded in the artifact and is the level below
    which a request's WER is not evidence of a synthesis defect.
    """
    from hipengine.generation.vibevoice_protocol import parse_transcript

    wer = _load_wer_module()
    entries = []
    for request in manifest["requests"]:
        path = fixtures / f"{request['name']}_audio.npz"
        if not path.is_file():
            continue
        audio = np.load(path)["pcm"].astype(np.float32)
        text = engine.transcribe(audio, max_new_tokens=256, seed=TRANSCRIBE_SEED).text
        segments = parse_transcript(text)
        content = " ".join(str(s["Content"]) for s in segments) if segments else text
        expected = " ".join(_expected_words(request["script"]))
        entries.append(
            {
                "reference": request["name"],
                "seconds": round(audio.size / 24000, 3),
                "transcript": content,
                "wer": (
                    round(float(wer._wer_content([expected], [content])), 4)
                    if segments is not None
                    else None
                ),
            }
        )
        print(
            f"evaluator floor {request['name']}: wer={entries[-1]['wer']} {content!r}",
            flush=True,
        )
    wers = [e["wer"] for e in entries if e["wer"] is not None]
    return {
        "method": (
            "the ASR lane transcribes the reference implementation's own PCM for the "
            "pinned requests (single_audio.npz / two_audio.npz), which contains every "
            "word; the WER it reports there is the evaluator's, not the TTS lane's"
        ),
        "requests": entries,
        "max_wer": max(wers) if wers else None,
        "interpretation": (
            "a request whose WER is at or below max_wer is not evidence of a synthesis "
            "defect. A floor of 0.0 means the evaluator reads the reference "
            "implementation's audio exactly; it does NOT mean every non-zero WER on "
            "this lane's audio is a synthesis defect, because the evaluator can still "
            "mishear audio whose phonetics differ from the reference. Establishing "
            "that case needs an independent ASR on the same waveform"
        ),
    }


def score(records, asr_model_id: str, fixtures: Path, manifest: dict):
    """Transcribe each synthesized request and apply the quality checks."""
    from hipengine import LLM
    from hipengine.generation.vibevoice_protocol import parse_transcript

    wer = _load_wer_module()
    engine = LLM(asr_model_id, max_sequence_length=4096)
    scored: list[dict] = []
    try:
        floor = _evaluator_floor(engine, fixtures, manifest)
        for record in records:
            path = ROOT / record["pcm"]
            audio = np.load(path) if path.is_file() else np.zeros(0, dtype=np.float32)
            start = time.perf_counter()
            text = (
                engine.transcribe(audio, max_new_tokens=256, seed=TRANSCRIBE_SEED).text
                if audio.size
                else ""
            )
            transcribe_seconds = time.perf_counter() - start
            segments = parse_transcript(text)
            content = (
                " ".join(str(segment["Content"]) for segment in segments) if segments else ""
            )
            expected = _expected_words(record["script"])
            hypothesis_words = _normalized_words(content)
            reference = " ".join(expected)

            entry = dict(record)
            entry.update(
                {
                    "transcribe_seconds": round(transcribe_seconds, 2),
                    "transcript_status": "ok" if segments is not None else "malformed",
                    "transcript": content,
                    "segments": (
                        [
                            {
                                "start": float(segment["Start"]),
                                "end": float(segment["End"]),
                                "speaker": str(segment["Speaker"]),
                                "content": str(segment["Content"]),
                            }
                            for segment in segments
                        ]
                        if segments is not None
                        else []
                    ),
                    "transcript_speakers": (
                        sorted({str(segment["Speaker"]) for segment in segments})
                        if segments is not None
                        else []
                    ),
                    "expected_words": len(expected),
                    "transcript_words": len(hypothesis_words),
                    "repeated_ngrams": _repeated_ngrams(hypothesis_words),
                }
            )
            entry.update(_level_checks(audio))
            entry["join_checks"] = _join_checks(audio, entry.get("chunk_samples") or [])
            entry["wer"] = (
                round(float(wer._wer_content([reference], [content])), 4)
                if segments is not None and expected
                else None
            )
            entry["word_count_ratio"] = (
                round(len(hypothesis_words) / len(expected), 4) if expected else None
            )
            # Whether this WER is above what the evaluator itself contributes on audio
            # that is known good. A failure below the floor is not attributable to the
            # TTS lane.
            entry["wer_above_evaluator_floor"] = (
                None
                if entry["wer"] is None or floor.get("max_wer") is None
                else bool(entry["wer"] > floor["max_wer"])
            )
            entry["expected_seconds"] = round(len(expected) / WORDS_PER_SECOND, 3)
            entry["duration_ratio"] = (
                round(entry["output_audio_seconds"] / entry["expected_seconds"], 3)
                if entry["expected_seconds"]
                else None
            )
            entry["checks"] = _checks(entry)
            entry["attribution_decided_windows"] = (
                len(_decided_windows(entry["voice_assignment_cosines"]))
                if len(set(entry["turn_speakers"])) > 1
                else None
            )
            # The run sequences the gate compares, and the pair-classified runs the
            # single-voice control reads, kept in the artifact so a failure says
            # which speaker change was lost rather than only that one was.
            entry["attribution_runs"] = _runs(_decided_windows(entry["voice_assignment_cosines"]))
            entry["expected_runs"] = _runs(entry["turn_speakers"])
            entry["pair_runs"] = _runs(_decided_windows(entry.get("voice_assignment_pair_cosines") or []))
            pair_cosines = entry.get("voice_assignment_pair_cosines")
            if pair_cosines and len(set(entry["turn_speakers"])) == 1:
                pair = np.asarray(pair_cosines, dtype=np.float64)
                margins = pair.max(axis=1) - pair.min(axis=1)
                entry["attribution_identity_control"] = {
                    "pair_runs": entry["pair_runs"],
                    "mean_cosines": [round(float(v), 4) for v in pair.mean(axis=0)],
                    "mean_margin": round(float(margins.mean()), 4),
                    "decided_windows": len(_decided_windows(pair_cosines)),
                    "windows": len(pair_cosines),
                    "interpretation": "diagnostic only; see the note in _checks",
                }
            entry["passed"] = all(entry["checks"].values())
            scored.append(entry)
            print(
                f"scored seed {entry['base_seed']} {entry['name']}: wer={entry['wer']} "
                f"speakers={entry['transcript_speakers']} "
                f"duration_ratio={entry['duration_ratio']} passed={entry['passed']}",
                flush=True,
            )
    finally:
        engine.close()
    return scored, floor


def _decided_windows(cosines) -> list[int]:
    """Reference index per window that clearly leads, ignoring near ties."""
    return [
        int(np.argmax(pair)) for pair in cosines if max(pair) - min(pair) >= ATTRIBUTION_MARGIN
    ]


def _join_checks(audio: np.ndarray, chunk_samples: list[int]) -> dict:
    """Sample-count continuity plus a click check at every codec chunk boundary.

    Correct chunks can still click at their joins, and a duplicated or dropped sample
    at a boundary changes the count without changing the duration much. The click
    test compares the largest sample-to-sample step at a boundary against the
    distribution of steps inside the signal: a join artifact is a discontinuity the
    waveform's own dynamics do not produce, so the threshold is relative to that
    distribution rather than an absolute level. A duplicated sample shows up as the
    opposite, a boundary step far *below* the signal's typical step, and is reported
    as a diagnostic beside it.
    """
    total = int(sum(chunk_samples))
    result = {
        "chunk_count": len(chunk_samples),
        "sample_count_continuous": total == int(audio.size),
        "whole_frames": all(size % FRAME_SAMPLES == 0 for size in chunk_samples),
    }
    if len(chunk_samples) < 2 or audio.size < 2:
        return result
    boundaries = np.cumsum(np.asarray(chunk_samples, dtype=np.int64))[:-1]
    boundaries = boundaries[(boundaries > 0) & (boundaries < audio.size)]
    if boundaries.size == 0:
        return result
    step = np.abs(np.diff(np.asarray(audio, dtype=np.float64)))
    inside = np.delete(step, boundaries - 1)
    if inside.size == 0:
        return result
    at_boundary = step[boundaries - 1]
    typical = float(np.percentile(inside, 99.9))
    floor = float(np.median(inside))
    worst = float(at_boundary.max())
    smallest = float(at_boundary.min())
    result.update(
        {
            "boundary_max_step": round(worst, 6),
            "interior_p999_step": round(typical, 6),
            "boundary_max_step_ratio": round(worst / typical, 4) if typical > 0 else None,
            "boundary_min_step_ratio": round(smallest / floor, 4) if floor > 0 else None,
            "no_join_click": bool(typical <= 0.0 or worst <= JOIN_CLICK_STEP_RATIO * typical),
        }
    )
    return result


def _checks(entry: dict) -> dict:
    """Named pass/fail checks. Every key must hold for the request to pass."""
    checks = {
        "transcript_well_formed": entry["transcript_status"] == "ok",
        "not_truncated": entry["finish_reason"] == "stop",
        "has_audio": entry["output_audio_seconds"] > 0.0,
        "audible": entry["rms"] >= MIN_RMS,
        "not_clipping": entry["absmax"] <= MAX_ABS,
        "no_long_silence": entry["longest_silence_seconds"] <= MAX_INTERNAL_SILENCE_SECONDS,
        "duration_plausible": (
            entry["duration_ratio"] is not None
            and DURATION_MIN_RATIO <= entry["duration_ratio"] <= DURATION_MAX_RATIO
        ),
        "intelligible": entry["wer"] is not None and entry["wer"] <= WER_GATE,
        "word_count_plausible": (
            entry["word_count_ratio"] is not None
            and DURATION_MIN_RATIO <= entry["word_count_ratio"] <= DURATION_MAX_RATIO
        ),
        "no_repeated_speech": len(entry["repeated_ngrams"]) <= MAX_REPEATED_NGRAMS,
        "no_join_click": entry.get("join_checks", {}).get("no_join_click", True),
        "sample_count_continuous": entry.get("join_checks", {}).get(
            "sample_count_continuous", True
        ),
        "turn_count": len(entry["segments"]) >= 1,
    }
    speakers = entry["turn_speakers"]
    if len(set(speakers)) > 1:
        # A multi-speaker script must reproduce its whole speaker sequence, not
        # just its endpoints, measured on the encoder's own window assignment
        # rather than on the ASR's speaker labels -- see _voice_assignment for why
        # those are not usable.
        decided = _decided_windows(entry["voice_assignment_cosines"])
        checks["voice_attribution"] = (
            len(decided) >= ATTRIBUTION_MIN_DECIDED_WINDOWS
            and _runs(decided) == _runs(speakers)
        )
    # There is deliberately no gate on a single-speaker request's *identity*.
    # Classifying single-voice audio against both reference voices was measured
    # and it does not support one: the per-window margin between the two
    # references sits at the noise floor (for `single-numbers` at seed 20260916
    # the mean cosines are 0.4290 against 0.4227, a margin of 0.006), it fails
    # on one request in one of three seeds, and on `single-medium` at seed
    # 20260917 every window is assigned to the voice the request was NOT
    # conditioned on. The instrument resolves speaker *changes* inside an
    # utterance that contains both voices; it does not resolve absolute voice
    # identity at these durations. `attribution_identity_control` records the
    # measurement per request so the limitation stays visible.
    return checks


def _load_wer_module():
    """The ASR lane's own WER, so the number is comparable to its gate."""
    import importlib.util

    path = Path(__file__).resolve().parent / "vibevoice_asr_wer.py"
    spec = importlib.util.spec_from_file_location("vibevoice_asr_wer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vibevoice_asr_wer"] = module
    spec.loader.exec_module(module)
    return module


def _gpu_name() -> str:
    import subprocess

    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=30).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Marketing Name:") and "Radeon" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _stability(runs: list[dict]) -> dict:
    """Per-request, per-check pass counts across seeds.

    A check that passes on every seed is stable; one that passes on some is
    reported as such rather than averaged away, because a request that comes out
    with the right number of voices half the time is not a passing request.
    """
    by_name: dict[str, dict] = {}
    for run in runs:
        for entry in run["requests"]:
            record = by_name.setdefault(
                entry["name"], {"seeds": 0, "passed": 0, "checks": {}}
            )
            record["seeds"] += 1
            record["passed"] += 1 if entry["passed"] else 0
            for name, ok in entry["checks"].items():
                record["checks"].setdefault(name, {"passed": 0})
                record["checks"][name]["passed"] += 1 if ok else 0
    for name in by_name:
        by_name[name]["stable"] = by_name[name]["passed"] == by_name[name]["seeds"]
    return {"seeds": len(runs), "requests": by_name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default=str(DEFAULT_REQUESTS))
    parser.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    parser.add_argument("--work", default=str(DEFAULT_WORK))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tts-model", default=TTS_MODEL_ID)
    parser.add_argument("--asr-model", default=ASR_MODEL_ID)
    parser.add_argument("--phase", choices=("all", "synthesize", "score"), default="all")
    parser.add_argument("--only", default=None, help="comma-separated request names")
    parser.add_argument(
        "--seeds",
        default="20260915",
        help="comma-separated base seeds; each request gets base_seed + its index",
    )
    args = parser.parse_args()

    suite = json.loads(Path(args.requests).read_text())
    requests = suite["requests"]
    if args.only:
        wanted = {name.strip() for name in args.only.split(",")}
        requests = [r for r in requests if r["name"] in wanted]
        if not requests:
            raise SystemExit(f"no request matches --only {args.only}")
    seeds = [int(part) for part in str(args.seeds).split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--seeds must name at least one seed")

    fixtures = Path(args.fixtures)
    work = Path(args.work)
    records_path = work / "synthesis.json"

    if args.phase in ("all", "synthesize"):
        records = synthesize(requests, suite, fixtures, args.tts_model, work, seeds)
        work.mkdir(parents=True, exist_ok=True)
        records_path.write_text(json.dumps(records, indent=2) + "\n")
    else:
        if not records_path.is_file():
            raise SystemExit(f"no synthesis records at {records_path}; run --phase synthesize")
        records = json.loads(records_path.read_text())
        wanted = {r["name"] for r in requests}
        records = [r for r in records if r["name"] in wanted and r["base_seed"] in set(seeds)]
        if not records:
            raise SystemExit("no synthesis records match --only/--seeds")

    if args.phase in ("all", "score"):
        scored, evaluator_floor = score(
            records,
            args.asr_model,
            fixtures,
            json.loads((fixtures / "manifest.json").read_text()),
        )
    else:
        scored, evaluator_floor = [], None

    if not scored:
        # A synthesize-only run has no verdict to publish; writing an artifact
        # here would overwrite the last scored one with an empty result.
        print("\nsynthesis only; nothing scored, no artifact written")
        return

    runs = []
    for seed in seeds:
        entries = [entry for entry in scored if entry["base_seed"] == seed]
        if not entries:
            continue
        failed = [entry["name"] for entry in entries if not entry["passed"]]
        runs.append({"seed": seed, "requests": entries, "failed_requests": failed})
    failed_any = sorted({name for run in runs for name in run["failed_requests"]})

    result = {
        "suite": suite["suite"],
        "suite_version": suite["version"],
        "model": args.tts_model,
        "quant": "bf16",
        "measurement_basis": "unassisted-generated-audio-quality",
        "asr_lane": args.asr_model,
        "tokenizer": TOKENIZER_MODEL_ID,
        "host": {"name": Path("/etc/hostname").read_text().strip(), "gpu": _gpu_name()},
        "command": (
            "uv run --with jiwer --with transformers python "
            "scripts/vibevoice_tts_quality_suite.py"
        ),
        "injected": "nothing (own VAE draw, own diffusion noise, own negative LM)",
        "seeding": (
            "each request is reseeded to base_seed + its index, so a request does not "
            "depend on how many requests ran before it"
        ),
        "gates": {
            "wer_max": WER_GATE,
            "duration_ratio_range": [DURATION_MIN_RATIO, DURATION_MAX_RATIO],
            "max_internal_silence_seconds": MAX_INTERNAL_SILENCE_SECONDS,
            "min_rms": MIN_RMS,
            "max_abs": MAX_ABS,
            "max_repeated_4grams": MAX_REPEATED_NGRAMS,
            "turn_count_min": 1,
            "voice_attribution": (
                "for a script with more than one speaker, the encoder's window assignment "
                "must reproduce the script's whole speaker sequence, not only its endpoints"
            ),
        },
        "attribution_instrument": {
            "method": (
                "hipengine.runtime.vibevoice_encoder.encode_reference (deterministic, no "
                "VAE sampling); each window assigned to the reference voice with the "
                "higher cosine similarity to its mean latent"
            ),
            "window_seconds": ATTRIBUTION_WINDOW_SECONDS,
            "hop_seconds": ATTRIBUTION_HOP_SECONDS,
            "decision_margin": ATTRIBUTION_MARGIN,
            "min_decided_windows": ATTRIBUTION_MIN_DECIDED_WINDOWS,
            "resolves": (
                "speaker changes inside an utterance that contains both voices: the "
                "4-turn script reproduces [0, 1, 0, 1] on all three seeds"
            ),
            "does_not_resolve": (
                "absolute voice identity for single-voice audio. Classifying a "
                "single-speaker request against both references sits at the noise "
                "floor (mean cosines 0.4236 against 0.4125 for one request), assigns "
                "every window to the wrong reference for another, and is therefore "
                "recorded per request as attribution_identity_control rather than "
                "gated"
            ),
            "asr_speaker_labels": (
                "reported per request as a diagnostic, not gated: on these requests the "
                "ASR reports one speaker for two-speaker scripts that contain both voices"
            ),
        },
        "limitations": suite["limitations"],
        "evaluator_floor": evaluator_floor,
        "stability": _stability(runs),
        "runs": runs,
        "failed_requests": failed_any,
        "passed": not failed_any,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    total = sum(len(run["requests"]) for run in runs)
    failed_count = sum(len(run["failed_requests"]) for run in runs)
    print(f"\n{total - failed_count}/{total} request-runs passed over {len(runs)} seed(s)")
    if failed_any:
        print("failed: " + ", ".join(failed_any))
    print(f"wrote {out}")
    if failed_any:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
