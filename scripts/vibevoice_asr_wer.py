#!/usr/bin/env python3
"""VibeVoice-ASR WER gate on the Open ASR Leaderboard protocol.

Scores the same LibriSpeech test-clean clips through multiple systems
(torch-GPU bf16 reference, hipEngine) and reports WER with the
leaderboard's normalization (Whisper EnglishTextNormalizer + jiwer),
comparable in spirit to the model card's published row
(LibriSpeech test-clean 2.20% WER).

Protocol notes (recorded honestly, not a leaderboard claim):
- a fixed-clips subset, not the full 2620-utterance test-clean, so the
  absolute WER differs from the published row by subset variance;
- greedy decoding in every lane, identical 24 kHz PCM per clip;
- 16 kHz LibriSpeech audio is resampled to 24 kHz with scipy
  polyphase filtering (torch-free) before any lane sees it.

Usage:
    python3 scripts/vibevoice_asr_wer.py --num-clips 50 --systems torch hip
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

BENCH_SCRIPT = Path(__file__).resolve().parent / "vibevoice_asr_bench.py"


def _import_bench():
    spec = importlib.util.spec_from_file_location("vibevoice_asr_bench", BENCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["vibevoice_asr_bench"] = module
    spec.loader.exec_module(module)
    return module


def _normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    # an empty spelling map behaves like the plain normalizer; the
    # leaderboard passes the Whisper tokenizer's UK/US spelling map
    return EnglishTextNormalizer({})


def parse_transcript(text: str) -> tuple[str, str]:
    """Split a VibeVoice transcript into (spoken text, status).

    The model emits a JSON array of segments. Status is ``"ok"`` for a
    well-formed array, ``"no_json"`` when no array is present, and
    ``"bad_json"`` when an array is present but does not parse. Callers
    must treat anything other than ``"ok"`` as a malformed generation:
    scoring the raw body silently turns a schema failure into a word
    error and hides it from the quality gate.
    """
    import json as _json

    body = text.strip()
    start = body.find("[")
    end = body.rfind("]")
    if start < 0:
        return body, "no_json"
    if end <= start:
        return body, "bad_json"
    try:
        segments = _json.loads(body[start:end + 1])
    except Exception:
        return body, "bad_json"
    if not isinstance(segments, list):
        return body, "bad_json"
    return " ".join(str(s.get("Content", "")) for s in segments), "ok"


def _transcription_only(text: str, *, strict: bool = True) -> str:
    """Extract the spoken text from VibeVoice's structured output.

    ``strict`` (default) raises on malformed output so a schema failure
    can never be scored as an ordinary transcription error.
    """
    content, status = parse_transcript(text)
    if strict and status != "ok":
        raise ValueError(f"malformed VibeVoice transcript ({status}): {text[:200]!r}")
    return content


def _wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate as a **fraction** in [0, 1] (jiwer convention).

    Use :func:`_wer_pct` for the percentage form. Mixing the two is a
    100x error, so call sites that print or store a summary must use the
    ``_pct`` helper and label the field accordingly.
    """
    from jiwer import process_words

    hyps = [_transcription_only(h) for h in hyps]
    ref_t = [_normalizer()(r) for r in refs]
    hyp_t = [_normalizer()(h) for h in hyps]
    # drop pairs the normalizer emptied (pure-punctuation refs)
    pairs = [(r, h) for r, h in zip(ref_t, hyp_t) if r]
    score = process_words([r for r, _ in pairs], [h for _, h in pairs])
    return score.wer


def _wer_pct(refs: list[str], hyps: list[str]) -> float:
    """Word error rate in percent (``_wer`` scaled by 100)."""
    return 100.0 * _wer(refs, hyps)


def _load_clips(num_clips: int, cache_dir: Path,
                librispeech_root: str = "/tmp/LibriSpeech/test-clean"):
    """Read LibriSpeech test-clean FLACs, cache 24 kHz float PCM per clip.

    The OpenSLR distribution (librispeech_asr) is script-based and no
    longer loads with datasets>=3, so the harness reads the unpacked
    test-clean tree directly: <root>/<spk>/<chap>/<uttr>.flac plus
    <spk>-<chap>.trans.txt.
    """
    import soundfile as sf
    from scipy.signal import resample_poly

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.jsonl"
    clips = []
    done = set()
    if manifest_path.exists():
        with open(manifest_path) as fh:
            for line in fh:
                rec = json.loads(line)
                clips.append(rec)
                done.add(rec["clip_id"])
    root = Path(librispeech_root)
    if not root.is_dir():
        raise SystemExit(
            f"LibriSpeech test-clean not unpacked at {root}; "
            "wget https://www.openslr.org/resources/12/test-clean.tar.gz"
        )
    # <root>/<spk>/<chap>/<uttr>.flac, <root>/<spk>/<chap>/<spk>-<chap>.trans.txt
    if len(clips) < num_clips:
        # spread across chapters/speakers: one utterance per chapter per
        # round, so a gate subset is not a single-speaker read
        chapter_uttrs = {}
        for trans in sorted(root.glob("*/*/*.trans.txt")):
            entries = [ln.strip().split(" ", 1) for ln in open(trans)]
            chapter_uttrs[trans] = [e for e in entries if e[0] not in done]
        round_robin = []
        while any(chapter_uttrs.values()):
            for trans in sorted(chapter_uttrs):
                if chapter_uttrs[trans]:
                    round_robin.append((trans, chapter_uttrs[trans].pop(0)))
        for trans, (uttr_id, text) in [(t, u) for t, u in round_robin][: max(0, num_clips - len(clips))]:
            spk, chap = uttr_id.split("-")[:2]
            flac = root / spk / chap / f"{uttr_id}.flac"
            audio, rate = sf.read(flac, dtype="float32")
            if rate != 24000:
                audio = resample_poly(audio, 24000, rate).astype(np.float32)
            wav_path = cache_dir / f"{uttr_id}.f32.npy"
            np.save(wav_path, audio)
            entry = {
                "clip_id": uttr_id,
                "wav": str(wav_path),
                "text": text,
                "seconds": len(audio) / 24000.0,
            }
            clips.append(entry)
            done.add(uttr_id)
            with open(manifest_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        return clips[:num_clips]
    return clips[:num_clips]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=50)
    parser.add_argument("--systems", nargs="+", default=["torch", "hip"],
                        choices=["torch", "hip"],
                        help="lane to score; unknown names are rejected rather "
                             "than silently mapped onto a default lane")
    parser.add_argument("--cache-dir", default="/tmp/librispeech-clean-cache")
    parser.add_argument("--out", default=None, help="JSON artifact path")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    bench = _import_bench()
    bench_args = argparse.Namespace(
        pcm_file=None,
        seconds=11.0,
        repeats=1,
        max_new_tokens=args.max_new_tokens,
        skip_torch="torch" not in args.systems,
        model="microsoft/VibeVoice-ASR-HF",
        weights="microsoft/VibeVoice-ASR",
    )

    clips = _load_clips(args.num_clips, Path(args.cache_dir))
    print(f"loaded {len(clips)} clips, total "
          f"{sum(c['seconds'] for c in clips):.1f} s audio")

    results = {"systems": {}, "clips": len(clips),
               "clip_ids": [c["clip_id"] for c in clips]}
    refs = [c["text"] for c in clips]
    for system in args.systems:
        lane = bench.bench_torch_gpu if system == "torch" else bench.bench_hipengine
        hyps = []
        for i, clip in enumerate(clips):
            pcm = np.load(clip["wav"])
            out = lane(pcm, clip["seconds"], bench_args)
            hyps.append(_transcription_only(out["text"]))
            print(f"[{system} {i+1}/{len(clips)}] {hyps[-1][:60]!r}")
        malformed = [clip["clip_id"] for clip, hyp in zip(clips, hyps)
                     if parse_transcript(hyp)[1] != "ok"]
        wer = _wer(refs, hyps)
        per_clip = []
        for clip, hyp in zip(clips, hyps):
            content, status = parse_transcript(hyp)
            per_clip.append({"clip_id": clip["clip_id"],
                             "wer_fraction": _wer([clip["text"]], [content]),
                             "parse_status": status,
                             "hyp": hyp})
        results["systems"][system] = {
            "wer_fraction": wer,
            "wer_pct": 100.0 * wer,
            "malformed_transcripts": malformed,
            "ref_texts": refs,
            "hypotheses": hyps,
            "per_clip": per_clip,
        }
        print(f"{system} WER: {100.0 * wer:.2f}%"
              + (f"  [{len(malformed)} malformed transcript(s): {malformed}]"
                 if malformed else ""))

    results["protocol"] = {
        "dataset": "openslr/librispeech_asr clean/test",
        "num_clips": len(clips),
        "normalizer": "Whisper EnglishTextNormalizer via jiwer",
        "resampling": "scipy resample_poly 16k->24k",
    }
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
