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


@pytest.mark.parametrize('variant',['strict','hipblaslt'])
def test_batched_prefill_matches_the_per_row_loop(runtime, lm, variant) -> None:
    """``prefill_host_rows`` must land where the per-row prefill lands.

    The batched path reassociates every projection through hipBLASLt, so it is
    a different arithmetic order from the one-GEMV-per-row loop. This pins the
    two together on the last prompt position's hidden state and logits, which
    is what generation conditions on. ``greedy_generate`` and the VibeVoice TTS
    session both use the batched path, so a divergence here would move every
    downstream chain.
    """
    rows = _prompt_rows(runtime, lm)
    previous = runtime.prefill_variant
    runtime.prefill_variant = variant
    try:
        runtime.prefill_host_rows(rows)
        batched_hidden = runtime.hidden_state()
        batched_logits, batched_token = runtime.logits_argmax()
    finally:
        runtime.prefill_variant = previous
    runtime.reset()
    for position, row in enumerate(rows):
        runtime.push_token(row, position)
        runtime.forward_layers(position)
    loop_hidden = runtime.hidden_state()
    loop_logits, loop_token = runtime.logits_argmax()
    hidden_peak = max(float(np.abs(loop_hidden).max()), 1e-9)
    logit_peak = max(float(np.abs(loop_logits).max()), 1e-9)
    # Measured envelope for the hipblaslt route: 0.028 on the hidden and 0.024
    # on the logits; the strict route is below both. These are max-over-channel
    # metrics on heavy-tailed post-norm activations, so they are far looser than
    # the fp32 reassociation error they are really looking for -- agreement with
    # torch is gated separately by ``test_greedy_chain_matches_torch`` and
    # ``test_teacher_forced_top1``, which both run the batched route. What this
    # catches is a schedule bug: wrong padding, a wrong last-row copy, or a
    # stale KV position. The envelope is also what the TTS session's two-speaker
    # trajectory gate is calibrated against, since it is the condition the
    # prefill feeds that moves.
    assert np.abs(batched_hidden - loop_hidden).max() / hidden_peak <= 0.05, (
        "batched prefill hidden drifted from the per-row loop"
    )
    assert np.abs(batched_logits - loop_logits).max() / logit_peak <= 0.05, (
        "batched prefill logits drifted from the per-row loop"
    )
    assert batched_token == loop_token, f"{batched_token} != {loop_token}"


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
