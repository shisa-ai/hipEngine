"""Correctness metrics used by benchmark and kernel gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogitCorrectness:
    kl_mean: float
    kl_max: float
    top1_agreement: float
    passed: bool


def evaluate_logits(
    reference_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    kl_threshold: float = 0.05,
    top1_threshold: float = 0.90,
) -> LogitCorrectness:
    """Return KL/top-1 correctness for two logit tensors."""

    reference = np.asarray(reference_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"logit shape mismatch: {reference.shape} vs {candidate.shape}")
    if reference.shape[-1] < 1:
        raise ValueError("logit tensors must have a non-empty vocabulary dimension")

    p = _softmax(reference, axis=-1)
    log_p = _log_softmax(reference, axis=-1)
    log_q = _log_softmax(candidate, axis=-1)
    kl = np.sum(p * (log_p - log_q), axis=-1)
    top1 = np.argmax(reference, axis=-1) == np.argmax(candidate, axis=-1)

    kl_mean = float(np.mean(kl))
    kl_max = float(np.max(kl))
    top1_agreement = float(np.mean(top1))
    return LogitCorrectness(
        kl_mean=kl_mean,
        kl_max=kl_max,
        top1_agreement=top1_agreement,
        passed=kl_mean <= kl_threshold and top1_agreement >= top1_threshold,
    )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
