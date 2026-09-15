"""Torch-free VibeVoice-TTS prompt construction.

Reproduces the fork's ``VibeVoiceProcessor._process_single`` and
``_create_voice_prompt`` (VibeVoice community fork @952326ddb264) so a request
can be built from a script and a set of speaker references without the fork:

.. code-block:: text

    <system prompt>
    " Voice input:\\n"
      per speaker: " Speaker N:" <speech_start> <speech_diffusion> * vae_len
                   <speech_end> "\\n"
    " Text input:\\n"
      per turn:    " Speaker N:<text>\\n"
    " Speech output:\\n" <speech_start>

Two details are easy to get wrong, and both are pinned against the frozen oracle
fixtures by ``tests/test_unit_vibevoice_tts_prompt.py``:

- A speaker's voice-prompt token count comes from that speaker's **own**
  reference-audio length, ``ceil(samples / 3200)``, taken before the batch pads
  every speaker to the longest one. Deriving it from the padded array inflates
  the two-speaker prompt from 352 tokens to 490.
- Voice blocks are emitted for speaker ids ``0 .. n-1``, while the text turns
  follow the script's line order. The fork builds its speaker list with
  ``list(set(...))``, which happens to be ordered for small sequential ids; this
  module requires the script's ids to be exactly ``0 .. n-1`` instead of relying
  on that.

The tokenizer is not part of the ``microsoft/VibeVoice-1.5B`` snapshot, so
callers pass one explicitly. The pinned ``microsoft/VibeVoice-ASR-HF`` tokenizer
reproduces both fixtures' ``input_ids`` and ``speech_input_mask`` exactly, which
the test asserts rather than assumes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from hipengine.loading.vibevoice_tts_session import (
    SPEECH_DIFFUSION_ID,
    SPEECH_END_ID,
    SPEECH_START_ID,
)

#: The fork's system prompt, including its leading space and trailing newline.
SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)
VOICE_INPUT_HEADER = " Voice input:\n"
TEXT_INPUT_HEADER = " Text input:\n"
SPEECH_OUTPUT_HEADER = " Speech output:\n"

#: The fork's ``speech_tok_compress_ratio``: one voice-prompt latent per 3200
#: samples of 24 kHz reference audio.
SPEECH_TOK_COMPRESS_RATIO = 3200

_SPEAKER_LINE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


def load_tts_tokenizer(path: str | Path):
    """Load a tokenizer from a ``tokenizer.json`` or a directory holding one."""
    from tokenizers import Tokenizer

    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / "tokenizer.json"
    return Tokenizer.from_file(str(resolved))


def vae_token_count(samples: int) -> int:
    """Voice-prompt latents the fork derives from ``samples`` reference samples."""
    return math.ceil(int(samples) / SPEECH_TOK_COMPRESS_RATIO)


def parse_speaker_script(script: str) -> list[tuple[int, str]]:
    """``(speaker_id, text)`` per script line, ids renumbered from 0.

    Mirrors the fork's ``_parse_script``: lines that are not ``Speaker N: ...``
    are dropped, each text is prefixed with one space, and ids are shifted down
    by one when every id is positive. Raises rather than returning an empty
    prompt, because the fork raises there too.
    """
    parsed: list[tuple[int, str]] = []
    for line in script.strip().split("\n"):
        if not line.strip():
            continue
        match = _SPEAKER_LINE.match(line.strip())
        if match is not None:
            parsed.append((int(match.group(1)), " " + match.group(2).strip()))
    if not parsed:
        raise ValueError("no 'Speaker N:' lines found in the script")
    if min(speaker_id for speaker_id, _ in parsed) > 0:
        parsed = [(speaker_id - 1, text) for speaker_id, text in parsed]
    return parsed


@dataclass(frozen=True)
class TtsPrompt:
    """The session's prefill input: token ids plus which of them are speech."""

    input_ids: list[int]
    speech_input_mask: list[bool]

    def __len__(self) -> int:
        return len(self.input_ids)

    def speech_token_positions(self) -> list[int]:
        """Positions the voice-prompt encoder fills with reference latents."""
        return [index for index, is_speech in enumerate(self.speech_input_mask) if is_speech]


def build_tts_prompt(
    script: str,
    vae_token_counts: Sequence[int],
    tokenizer: Any,
) -> TtsPrompt:
    """Build the prefill ids and speech mask for ``script``.

    ``vae_token_counts[i]`` is speaker ``i``'s own voice-prompt token count
    (:func:`vae_token_count` of that speaker's reference samples). The script's
    speaker ids must be exactly ``0 .. len(vae_token_counts) - 1``.
    """
    def encode(text: str) -> list[int]:
        return [int(token) for token in tokenizer.encode(text, add_special_tokens=False).ids]

    turns = parse_speaker_script(script)
    speakers = sorted({speaker_id for speaker_id, _ in turns})
    counts = [int(count) for count in vae_token_counts]
    if speakers != list(range(len(counts))):
        raise ValueError(
            f"script speaker ids {speakers} do not match {len(counts)} reference voices"
        )
    if any(count <= 0 for count in counts):
        raise ValueError(f"voice-prompt token counts must be positive, got {counts}")

    ids = encode(SYSTEM_PROMPT)
    mask = [False] * len(ids)

    header = encode(VOICE_INPUT_HEADER)
    ids += header
    mask += [False] * len(header)
    for speaker_id, count in enumerate(counts):
        prefix = encode(f" Speaker {speaker_id}:")
        ids += prefix + [SPEECH_START_ID] + [SPEECH_DIFFUSION_ID] * count + [SPEECH_END_ID]
        mask += [False] * len(prefix) + [False] + [True] * count + [False]
        newline = encode("\n")
        ids += newline
        mask += [False] * len(newline)

    header = encode(TEXT_INPUT_HEADER)
    ids += header
    mask += [False] * len(header)
    for speaker_id, text in turns:
        turn = encode(f" Speaker {speaker_id}:{text}\n")
        ids += turn
        mask += [False] * len(turn)

    trailer = encode(SPEECH_OUTPUT_HEADER) + [SPEECH_START_ID]
    ids += trailer
    mask += [False] * len(trailer)
    return TtsPrompt(ids, mask)
