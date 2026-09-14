"""VibeVoice-ASR Qwen2 GPU runner parity vs the torch LM fixture.

Gates: first-position logits, the full 16-token greedy chain vs the torch
oracle, and teacher-forced top-1 tokens. Skips without HIP or the local
HF artifact / fixture. The acoustic sampling uses the fixture's recorded
noise so the comparison is RNG-independent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

if not hip_runtime_available():
    pytest.skip("no usable HIP runtime for VibeVoice Qwen2 runner tests", allow_module_level=True)

from hipengine.runtime.vibevoice_qwen2 import VibevoiceQwen2Runtime, greedy_generate

LM_FIXTURE = Path(__file__).parent / "fixtures" / "vibevoice_asr" / "vibevoice_asr_lm.npz"
PINNED_HF_MODEL_ID = "microsoft/VibeVoice-ASR-HF"

if not LM_FIXTURE.is_file():
    pytest.skip("VibeVoice LM fixture not present", allow_module_level=True)


def _hf_snapshot_or_skip():
    from hipengine.loading.hf_cache import resolve_model_path

    try:
        return resolve_model_path(PINNED_HF_MODEL_ID)
    except Exception:
        pytest.skip(f"{PINNED_HF_MODEL_ID} not in local HF cache")


@pytest.fixture(scope="module")
def runtime():
    from hipengine.loading.vibevoice_asr import load_vibevoice_qwen2

    snapshot = _hf_snapshot_or_skip()
    weights = load_vibevoice_qwen2(str(snapshot))
    runner = VibevoiceQwen2Runtime(weights, max_context=512)
    yield runner
    runner.close()


@pytest.fixture(scope="module", params=['vibevoice_asr','vibevoice_asr_gpu'])
def lm(request) -> dict[str, np.ndarray]:
    path=LM_FIXTURE.parent.parent/request.param/LM_FIXTURE.name
    if not path.is_file():
        pytest.skip(f'{request.param} LM fixture unavailable')
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _prompt_rows(runner, lm) -> list[np.ndarray]:
    input_ids = np.asarray(lm["input_ids"])[0]
    positions = np.asarray(lm["audio_placeholder_positions"])
    audio = lm["audio_embeds"].astype(np.float32)
    rows = [runner.embed_row(int(t)) for t in input_ids]
    for p in positions:
        rows[p] = audio[p - positions[0]]
    return rows


def test_first_position_logits(runtime, lm) -> None:
    rows = _prompt_rows(runtime, lm)
    runtime.reset()
    runtime.push_token(rows[0], 0)
    runtime.forward_layers(0)
    logits, _ = runtime.logits_argmax()
    ref = lm["logits_pos0"]
    diff = np.abs(logits - ref).max()
    scale = max(np.abs(ref).max(), 1e-9)
    assert diff / scale <= 2e-2, f"logits_pos0 rel {diff / scale:.3e}"
    assert int(logits.argmax()) == int(ref.argmax())


@pytest.mark.parametrize('variant',['strict','hipblaslt'])
def test_greedy_chain_matches_torch(runtime, lm, variant) -> None:
    """Prefix regression only: these fixtures end before Content, not a task gate."""
    rows = _prompt_rows(runtime, lm)
    previous=runtime.prefill_variant
    runtime.prefill_variant=variant
    try:
        generated = greedy_generate(runtime, rows, max_new_tokens=16)
    finally:
        runtime.prefill_variant=previous
    fixture = [int(t) for t in np.asarray(lm["greedy_tokens"])]
    assert generated == fixture, f"{generated} != {fixture}"


def test_teacher_forced_top1(runtime, lm) -> None:
    """All 16 continuation steps keep the torch oracle's argmax."""
    rows = _prompt_rows(runtime, lm)
    fixture = [int(t) for t in np.asarray(lm["greedy_tokens"])]
    runtime.reset()
    for pos, row in enumerate(rows):
        runtime.push_token(row, pos)
        runtime.forward_layers(pos)
    _, token = runtime.logits_argmax()
    for step, expected in enumerate(fixture):
        assert token == expected, (step, token, expected)
        pos = len(rows) + step
        runtime.push_token(runtime.embed_row(token), pos)
        runtime.forward_layers(pos)
        _, token = runtime.logits_argmax()
