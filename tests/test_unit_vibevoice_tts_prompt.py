"""The in-tree TTS prompt builder must reproduce the oracle's prefill exactly.

The frozen fixtures record the ``input_ids`` and ``speech_input_mask`` the fork's
``VibeVoiceProcessor`` handed the model, so they are an independent check on
:mod:`hipengine.loading.vibevoice_tts_prompt`. A held-out request is only a valid
request if the prompt built for it is the prompt the fork would have built, and
this is where that is established for the two requests we can check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.vibevoice_tts_prompt import (
    SPEECH_TOK_COMPRESS_RATIO,
    TtsPrompt,
    build_tts_prompt,
    load_tts_tokenizer,
    parse_speaker_script,
    vae_token_count,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vibevoice_tts"
PINNED_MODEL_ID = "microsoft/VibeVoice-1.5B"
# The VibeVoice-1.5B snapshot ships no tokenizer.json, so the prompt builder is
# pointed at the sibling ASR checkpoint's tokenizer. The reproduction assertions
# below are what make that substitution legitimate rather than assumed.
TOKENIZER_MODEL_ID = "microsoft/VibeVoice-ASR-HF"

if not (FIXTURE_DIR / "manifest.json").is_file():
    pytest.skip("VibeVoice-TTS trace fixtures not present", allow_module_level=True)


@pytest.fixture(scope="module")
def manifest():
    return json.loads((FIXTURE_DIR / "manifest.json").read_text())


@pytest.fixture(scope="module")
def tokenizer():
    from hipengine.loading.hf_cache import resolve_model_path

    try:
        path = resolve_model_path(TOKENIZER_MODEL_ID)
    except (FileNotFoundError, ValueError):
        pytest.skip(f"{TOKENIZER_MODEL_ID} not in local HF cache", allow_module_level=True)
    if not (Path(path) / "tokenizer.json").is_file():
        pytest.skip(f"{TOKENIZER_MODEL_ID} tokenizer.json not in local HF cache")
    return load_tts_tokenizer(path)


def _oracle(name: str) -> tuple[list[int], list[bool]]:
    with np.load(FIXTURE_DIR / f"{name}_lm.npz") as data:
        ids = [int(t) for t in np.asarray(data["input_ids"])[0]]
        mask = [bool(v) for v in np.asarray(data["speech_input_mask"]).reshape(-1)]
    return ids, mask


@pytest.mark.parametrize("name", ["single", "two"])
def test_built_prompt_reproduces_the_oracle(manifest, tokenizer, name):
    """Both fixtures must match token-for-token, mask included."""
    request = next(r for r in manifest["requests"] if r["name"] == name)
    with np.load(FIXTURE_DIR / f"{name}_reference.npz") as reference:
        # Per-speaker counts come from the true (unpadded) lengths.
        counts = [int(mask.sum()) for mask in reference["ref_speech_masks"]]

    prompt = build_tts_prompt(request["script"], counts, tokenizer)
    oracle_ids, oracle_mask = _oracle(name)

    assert len(prompt) == len(oracle_ids), f"{name}: {len(prompt)} ids vs oracle {len(oracle_ids)}"
    assert prompt.input_ids == oracle_ids, f"{name}: input_ids differ from the oracle"
    assert prompt.speech_input_mask == oracle_mask, f"{name}: speech mask differs from the oracle"
    assert prompt.speech_token_positions() == [
        index for index, is_speech in enumerate(oracle_mask) if is_speech
    ]


@pytest.mark.parametrize("name", ["single", "two"])
def test_padded_reference_length_is_not_the_voice_prompt_length(manifest, tokenizer, name):
    """The count must come from each speaker's own audio, not the padded batch.

    The two-speaker fixture pads both references to 665600 samples. Taking the
    count from that shape gives 208 latents for each speaker instead of 70 and
    208, which is the 490-vs-352 token difference this guards.
    """
    request = next(r for r in manifest["requests"] if r["name"] == name)
    with np.load(FIXTURE_DIR / f"{name}_reference.npz") as reference:
        padded = int(np.asarray(reference["ref_pcm"]).shape[1])
        counts = [int(mask.sum()) for mask in reference["ref_speech_masks"]]

    wrong = [vae_token_count(padded)] * len(counts)
    if wrong == counts:
        pytest.skip(f"{name}: padding and true lengths coincide, nothing to distinguish")

    prompt = build_tts_prompt(request["script"], counts, tokenizer)
    inflated = build_tts_prompt(request["script"], wrong, tokenizer)
    oracle_ids, _ = _oracle(name)

    assert len(prompt) == len(oracle_ids)
    assert len(inflated) != len(oracle_ids), (
        f"{name}: padded-length counts {wrong} gave {len(inflated)} tokens, "
        f"which should not match the oracle's {len(oracle_ids)}"
    )


def test_vae_token_count_rounds_up():
    assert vae_token_count(0) == 0
    assert vae_token_count(1) == 1
    assert vae_token_count(SPEECH_TOK_COMPRESS_RATIO) == 1
    assert vae_token_count(SPEECH_TOK_COMPRESS_RATIO + 1) == 2


def test_parse_speaker_script_renumbers_positive_ids_and_keeps_zero():
    assert parse_speaker_script("Speaker 1: a\nSpeaker 2: b") == [(0, " a"), (1, " b")]
    assert parse_speaker_script("Speaker 0: a\nSpeaker 1: b") == [(0, " a"), (1, " b")]
    # Blank and unparseable lines are dropped, like the fork's parser.
    assert parse_speaker_script("\nSpeaker 1: a\nnonsense\n\nSpeaker 1: b\n") == [
        (0, " a"),
        (0, " b"),
    ]


def test_parse_speaker_script_rejects_a_script_with_no_turns():
    with pytest.raises(ValueError, match="no 'Speaker N:' lines"):
        parse_speaker_script("just prose, no speaker labels")


def test_build_rejects_a_voice_count_that_does_not_match_the_script(tokenizer):
    with pytest.raises(ValueError, match="do not match 2 reference voices"):
        build_tts_prompt("Speaker 1: a\nSpeaker 2: b\nSpeaker 3: c", [10, 10], tokenizer)


def test_build_rejects_a_nonpositive_voice_count(tokenizer):
    with pytest.raises(ValueError, match="must be positive"):
        build_tts_prompt("Speaker 1: a", [0], tokenizer)


def test_prompt_exposes_a_plain_dataclass_contract():
    prompt = TtsPrompt([1, 2, 3], [False, True, False])
    assert len(prompt) == 3
    assert prompt.speech_token_positions() == [1]
