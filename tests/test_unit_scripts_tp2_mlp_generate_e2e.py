"""Tests for scripts/tp2_mlp_generate_e2e.py (full-model TP2 checkpoint).

The host-checkable pieces are the gate arithmetic: the KL metric must
reproduce a hand-computed value, count top-1 flips exactly, and the
production envelope must bind on the right side of each threshold. The
end-to-end run needs two gfx1100 devices and the real Q4_K_M artifact and is
skipped elsewhere; its hardware evidence is the committed result artifact.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "tp2_mlp_generate_e2e.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("tp2_mlp_generate_e2e", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf").exists()


def test_softmax_shift_invariance(mod):
    # Integer values and shifts are exact in f32, so any difference the
    # comparison sees would be a real softmax bug, not representation loss.
    logits = np.array([[1024.0, 1025.0, 1024.0, 1023.0]], dtype=np.float32)
    shifted = logits + np.float32(1024.0)
    assert np.allclose(mod._softmax(logits), mod._softmax(shifted), rtol=0, atol=1e-12)


def test_kl_metrics_zero_for_identical_rows(mod):
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(5, 32)).astype(np.float32)
    metrics = mod._kl_metrics(logits, logits.copy())
    assert metrics["mean_kl"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["max_kl"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["flipped_rows"] == 0
    assert metrics["rows"] == 5


def test_kl_metrics_matches_hand_computed_value(mod):
    # P = softmax([ln 2, ln 2, 0]) = [0.4, 0.4, 0.2]
    # Q = softmax([ln 2, 0, 0])     = [0.5, 0.25, 0.25]
    # KL(P||Q) = 0.6 ln 0.8 + 0.4 ln 1.6
    teacher = np.log(np.array([2.0, 2.0, 1.0], dtype=np.float64)).reshape(1, 3)
    student = np.log(np.array([2.0, 1.0, 1.0], dtype=np.float64)).reshape(1, 3)
    metrics = mod._kl_metrics(teacher.astype(np.float32), student.astype(np.float32))
    expected = 0.6 * math.log(0.8) + 0.4 * math.log(1.6)
    assert metrics["mean_kl"] == pytest.approx(expected, rel=1e-6)
    assert metrics["p95_kl"] == pytest.approx(expected, rel=1e-6)
    assert metrics["max_kl"] == pytest.approx(expected, rel=1e-6)


def test_kl_metrics_counts_top1_flips_exactly(mod):
    teacher = np.zeros((4, 8), dtype=np.float32)
    teacher[0, 3] = 5.0
    teacher[2, 6] = 5.0
    student = teacher.copy()
    student[0, 4] = 5.5  # flips row 0's argmax
    metrics = mod._kl_metrics(teacher, student)
    assert metrics["flipped_rows"] == 1
    assert metrics["top1_agreement"] == pytest.approx(3.0 / 4.0)


def test_kl_metrics_rejects_shape_mismatch(mod):
    with pytest.raises(ValueError, match="shape mismatch"):
        mod._kl_metrics(np.zeros((2, 4), dtype=np.float32), np.zeros((3, 4), dtype=np.float32))


def test_gate_passes_inside_envelope(mod):
    metrics = {
        "mean_kl": mod.PRODUCTION_GATE["mean_kl"] * 0.5,
        "p95_kl": mod.PRODUCTION_GATE["p95_kl"] * 0.5,
        "p99_kl": mod.PRODUCTION_GATE["p99_kl"] * 0.5,
        "max_kl": mod.PRODUCTION_GATE["max_kl"] * 0.5,
        "top1_agreement": 1.0,
    }
    passed, failures = mod._gate_passes(metrics)
    assert passed
    assert failures == []


def test_gate_fails_on_each_side_of_the_envelope(mod):
    base = {
        "mean_kl": 0.0,
        "p95_kl": 0.0,
        "p99_kl": 0.0,
        "max_kl": 0.0,
        "top1_agreement": 1.0,
    }
    over = dict(base, mean_kl=mod.PRODUCTION_GATE["mean_kl"] * 1.0001)
    passed, failures = mod._gate_passes(over)
    assert not passed
    assert len(failures) == 1 and "mean_kl" in failures[0]

    under = dict(base, top1_agreement=0.98999)
    passed, failures = mod._gate_passes(under)
    assert not passed
    assert len(failures) == 1 and "top1_agreement" in failures[0]


def test_gate_boundary_values_bind_inclusively(mod):
    base = {
        "mean_kl": 0.0,
        "p95_kl": 0.0,
        "p99_kl": 0.0,
        "max_kl": 0.0,
        "top1_agreement": 1.0,
    }
    # KL thresholds refuse only strictly-greater values; top-1 refuses
    # strictly-less ones, so the limit value itself passes.
    at_limit = dict(base, mean_kl=mod.PRODUCTION_GATE["mean_kl"], top1_agreement=0.99)
    passed, _ = mod._gate_passes(at_limit)
    assert passed


def test_fixed_inputs_are_stable(mod):
    assert len(mod.PROMPTS) >= 4
    assert mod.TEACHER_FORCED_TOKENS == (
        9707, 198, 1115, 596, 13365, 311, 11202, 1226,
        25, 1879, 11, 662, 3290, 13, 5966, 2675,
    )


@pytest.mark.skipif(not _hip_available(), reason="requires HIP and the Q4_K_M artifact")
def test_teacher_forced_tokens_in_vocabulary():
    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    module = _load()
    info = scan_gguf("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    vocab = len(build_qwen35_gguf_tensor_map(info).metadata_vocab if hasattr(build_qwen35_gguf_tensor_map(info), "metadata_vocab") else tokenizer.tokens)
    assert vocab == len(tokenizer.tokens)
    assert all(0 <= t < vocab for t in module.TEACHER_FORCED_TOKENS)
    for prompt in module.PROMPTS:
        ids = tokenizer.encode(prompt)
        assert ids and all(0 <= t < vocab for t in ids)
