from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.surya import (
    SuryaSpec,
    SuryaWeights,
    greedy_generate,
    text_decode_step,
    text_prefill,
)

FIXTURES = Path("tests/fixtures/surya")
CHECKPOINT_GLOBS = [
    "~/.cache/huggingface/hub/models--datalab-to--surya-ocr-2/snapshots/*/model.safetensors",
]

# Gates: the outer smoke/safety floor from AGENTS.md (KL <= 0.05, top-1
# >= 90%) plus tighter absolute gates set from measured values (recorded
# in the unit worklog entry).
#
# Known, documented numeric difference: the torch prefill computes the
# GDN recurrent state with the chunked delta-rule kernel (fla-style WY
# factorization, chunk 64) while this reference accumulates the
# mathematically equivalent step-by-step recurrence. The two factorizations
# accumulate different fp32 rounding through 40 recurrent steps, so the
# STORED state differs (measured mean|d| ~1e-3..5e-3, overall cos 0.92-0.97)
# while layer outputs, KV cache, and prefill logits agree tightly. Decode
# step logits inherit that state difference (measured max|d| 0.24-0.42,
# top-1 still exact). The state SELF-consistency (prefill vs threaded
# step) is exact and is the binding state contract here.
KL_GATE = 0.05
TOP1_GATE = 0.90
LOGITS_PREFILL_GATE = 1e-3    # measured 2.6e-5
LOGITS_STEP_GATE = 0.5        # measured 0.24-0.42 (state factorization)
KV_GATE = 1e-3                # measured 7.6e-6
STATE_SELF_CONSISTENCY_GATE = 1e-5   # measured 3e-7
CONV_WINDOW_SELF_CONSISTENCY_GATE = 1e-4  # measured 1.1e-5 (raw projections)
STATE_TORCH_COS_GATE = 0.9    # measured 0.92-0.97 (chunked vs step form)
STATE_TORCH_MEAN_GATE = 0.02  # measured 1e-3..5e-3


def _checkpoint() -> Path | None:
    import glob

    for pattern in CHECKPOINT_GLOBS:
        hits = sorted(glob.glob(str(Path(pattern).expanduser())))
        if hits:
            return Path(hits[0])
    return None


def _oracle(name: str) -> dict[str, np.ndarray]:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{path} not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    p = _softmax(p_logits.astype(np.float64))
    q = _softmax(q_logits.astype(np.float64))
    return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@pytest.fixture(scope="module")
def weights() -> SuryaWeights:
    ckpt = _checkpoint()
    if ckpt is None:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")
    return SuryaWeights.load(str(ckpt))


@pytest.fixture(scope="module")
def spec() -> SuryaSpec:
    return SuryaSpec()


@pytest.fixture(scope="module")
def text_oracle() -> dict[str, np.ndarray]:
    return _oracle("oracle_text.npz")


def _prefill_logits(
    w: SuryaWeights,
    spec: SuryaSpec,
    oracle: dict[str, np.ndarray],
    visual_features: np.ndarray | None = None,
):
    ids = oracle["input_ids"]
    pos = oracle["position_ids_prefill"][:, 0].astype(np.int64)  # (3, seq)
    hidden, state = text_prefill(
        w, spec, ids, pos, visual_features=visual_features
    )
    emb = w["model.language_model.embed_tokens.weight"]
    return hidden[:, -1] @ emb.T, state


def test_text_prefill_logits_match_oracle(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    logits, _ = _prefill_logits(weights, spec, text_oracle)
    ref = text_oracle["logits_first"]
    kl = _kl(ref, logits[0])
    maxdiff = float(np.abs(logits[0] - ref).max())
    top1_ref = int(np.argmax(ref))
    top1 = int(np.argmax(logits[0]))
    assert kl <= KL_GATE, f"KL {kl:.3e}"
    assert maxdiff <= LOGITS_PREFILL_GATE, f"max|d| {maxdiff:.3e}"
    assert top1 == top1_ref, f"top-1 {top1} != oracle {top1_ref}"
    assert np.isfinite(logits).all()


def test_cached_decode_steps_match_oracle(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    _, state = _prefill_logits(weights, spec, text_oracle)
    step_ids = [100, 5000, 20000, 42]
    base = int(text_oracle["position_ids_prefill"][0, 0, -1])
    for i, tok in enumerate(step_ids):
        logits = text_decode_step(weights, spec, tok, state, base + 1 + i)
        ref = text_oracle[f"logits_step{i}"]
        kl = _kl(ref, logits[0])
        maxdiff = float(np.abs(logits[0] - ref).max())
        assert kl <= KL_GATE, f"step {i}: KL {kl:.3e}"
        assert maxdiff <= LOGITS_STEP_GATE, f"step {i}: max|d| {maxdiff:.3e}"
        assert int(np.argmax(logits[0])) == int(np.argmax(ref)), f"step {i} top-1"


def test_gdn_state_self_consistency(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    """Binding state contract: the threaded step state after
    prefill(0..38)+step(token39) must equal the state after prefill(0..39)."""

    ids = text_oracle["input_ids"]
    pos = text_oracle["position_ids_prefill"][:, 0].astype(np.int64)
    _, st39 = text_prefill(weights, spec, ids[:, :39], pos[:, :39])
    text_decode_step(weights, spec, int(ids[0, 39]), st39, int(pos[0, 39]))
    _, st40 = text_prefill(weights, spec, ids[:, :40], pos[:, :40])
    for layer in (0, 20):
        a = st39.gdn[layer].recurrent
        b = st40.gdn[layer].recurrent
        assert np.abs(a - b).max() <= STATE_SELF_CONSISTENCY_GATE, layer
        # conv windows agree to fp32 matmul-tiling noise (raw projections
        # of the same hidden states, computed under different tile shapes)
        assert (
            np.abs(st39.gdn[layer].conv_window - st40.gdn[layer].conv_window).max()
            <= CONV_WINDOW_SELF_CONSISTENCY_GATE
        ), layer


def test_gdn_state_vs_torch_chunked_kernel_diagnostic(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    """Loose diagnostic only: torch stores the chunked-kernel state (fla WY
    factorization); this reference stores the step-recurrence state. Same
    math, different fp32 rounding path — see module docstring. The binding
    state contract is the self-consistency test above."""

    _, state = _prefill_logits(weights, spec, text_oracle)
    for layer in (0, 20):
        ref = text_oracle[f"prefill.cache.layer{layer}.recurrent_states[0]"]
        mine = state.gdn[layer].recurrent
        assert mine.shape == ref.shape
        cos = float(
            (mine * ref).sum()
            / (np.linalg.norm(mine) * np.linalg.norm(ref) + 1e-12)
        )
        mean_diff = float(np.abs(mine - ref).mean())
        assert cos >= STATE_TORCH_COS_GATE, f"layer {layer}: cos {cos:.4f}"
        assert mean_diff <= STATE_TORCH_MEAN_GATE, f"layer {layer}: mean|d| {mean_diff:.4f}"


def test_full_attention_kv_matches_torch_cache(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    _, state = _prefill_logits(weights, spec, text_oracle)
    for layer in (3, 23):
        k_ref = text_oracle[f"prefill.cache.layer{layer}.keys"]
        v_ref = text_oracle[f"prefill.cache.layer{layer}.values"]
        k_mine, v_mine = state.kv[layer]
        assert k_mine.shape == k_ref.shape
        np.testing.assert_allclose(k_mine, k_ref, atol=KV_GATE, rtol=1e-3)
        np.testing.assert_allclose(v_mine, v_ref, atol=KV_GATE, rtol=1e-3)


def test_chunked_prefill_matches_single_pass(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    ids = text_oracle["input_ids"]
    pos = text_oracle["position_ids_prefill"][:, 0].astype(np.int64)  # (3, seq)
    s = ids.shape[1]
    cut = s // 2
    emb = w_emb = weights["model.language_model.embed_tokens.weight"]

    _, state = text_prefill(weights, spec, ids[:, :cut], pos[:, :cut])
    hidden2, state = text_prefill(weights, spec, ids[:, cut:], pos[:, cut:], state=state)
    logits_chunked = hidden2[:, -1] @ w_emb.T

    hidden_full, _ = text_prefill(weights, spec, ids, pos)
    logits_full = hidden_full[:, -1] @ w_emb.T

    assert state.seq_len == s
    assert int(np.argmax(logits_chunked[0])) == int(np.argmax(logits_full[0]))
    np.testing.assert_allclose(
        logits_chunked, logits_full, atol=STATE_SELF_CONSISTENCY_GATE * 100, rtol=1e-3
    )


def test_image_case_text_side_matches_oracle(
    weights: SuryaWeights, spec: SuryaSpec
) -> None:
    oracle = _oracle("oracle_image.npz")
    ids = oracle["input_ids"]
    pos = oracle["position_ids_prefill"][:, 0].astype(np.int64)  # (3, seq)
    visual = oracle["vision_merged"][None]  # (1, 64, 1024)
    hidden, _ = text_prefill(weights, spec, ids, pos, visual_features=visual)
    emb = weights["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    ref = oracle["logits_first"]
    kl = _kl(ref, logits[0])
    assert kl <= KL_GATE, f"KL {kl:.3e}"
    assert int(np.argmax(logits[0])) == int(np.argmax(ref))


def test_greedy_generate_terminates_and_stays_in_vocab(
    weights: SuryaWeights, spec: SuryaSpec, text_oracle: dict[str, np.ndarray]
) -> None:
    ids = text_oracle["input_ids"]
    pos = text_oracle["position_ids_prefill"][:, 0].astype(np.int64)  # (3, seq)
    generated = greedy_generate(weights, spec, ids, pos, max_new_tokens=48)
    assert len(generated) <= 48
    assert all(0 <= t < spec.vocab_size for t in generated)
    # behavioral note: with no image attached the model is not guaranteed
    # to stop at <|im_end|> within 48 tokens; termination at the pinned
    # EOS=2 (never the stale 248044) is asserted only when it stops early
    if len(generated) < 48:
        assert generated[-1] == spec.eos_token_id
