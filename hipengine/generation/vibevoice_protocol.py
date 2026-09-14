"""Torch-free request template, audio preprocessing and transcript schema."""
import json
import math
import numpy as np

AUDIO_TOKEN = "<|box_start|>"
AUDIO_BOS = "<|object_ref_start|>"
AUDIO_EOS = "<|object_ref_end|>"
AUDIO_TOKEN_ID = 151648
IM_END_ID = 151645
SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format."
)


def build_prompt(duration: float, frames: int, *, context: str | None = None) -> str:
    """Hand-rolled chat template matching the processor's transcription request."""
    audio_block = f"{AUDIO_BOS}{AUDIO_TOKEN * frames}{AUDIO_EOS}\n"
    if context:
        info = (
            f"This is a {duration:.2f} seconds audio, with extra info: {context}\n\n"
            "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        )
    else:
        info = (
            f"This is a {duration:.2f} seconds audio, please transcribe it with "
            "these keys: Start time, End time, Speaker ID, Content"
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{audio_block}{info}<|im_end|>\n"
    )


def parse_transcript(text: str) -> list[dict] | None:
    stripped = text.strip()
    if stripped.startswith("assistant"):
        stripped = stripped[len("assistant"):].strip()
    if not stripped.startswith("["):
        return None
    try:
        segments = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(segments, list):
        return None
    for segment in segments:
        if not isinstance(segment, dict) or not {"Start", "End", "Speaker", "Content"} <= segment.keys():
            return None
        times = (segment["Start"], segment["End"])
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in times):
            return None
        if not 0 <= times[0] <= times[1] or not isinstance(segment["Content"], str):
            return None
        if isinstance(segment["Speaker"], bool) or not isinstance(segment["Speaker"], (int, str)):
            return None
    return segments



def preprocess_audio(audio, *, sample_rate=24000, normalize_audio=True, target_dB_FS=-25, eps=1e-6):
    """HF mono contract; reject other rates rather than silently resampling."""
    if sample_rate != 24000:
        raise ValueError("audio must be sampled at 24000 Hz; resample before transcription")
    pcm = np.array(audio, dtype=np.float32, copy=True)
    if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
        raise ValueError("audio must be a nonempty finite mono waveform")
    if normalize_audio:
        rms = np.sqrt(np.mean(pcm ** 2, dtype=np.float32))
        pcm *= np.float32(10 ** (target_dB_FS / 20)) / (rms + np.float32(eps))
        peak = np.max(np.abs(pcm))
        if peak > 1:
            pcm /= peak + np.float32(eps)
    from hipengine.loading.vibevoice_layout import f32_to_bf16_bits
    return (f32_to_bf16_bits(pcm).astype(np.uint32) << 16).view(np.float32)
