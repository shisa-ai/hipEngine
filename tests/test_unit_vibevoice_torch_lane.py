"""The torch lane's recorded-operand shim, and the fixture it must reproduce.

The HIP and torch lanes are only comparable if they are timed on one request, so
the torch lane serves the fixture's recorded random operands instead of drawing
its own. These tests pin the parts of that shim that a passing benchmark run
cannot distinguish: which draw each recorded operand answers, that an
unrecognised shape falls through to the normal RNG, and that running out of
recorded diffusion frames is reported rather than silently resampled.

No GPU work happens here; ``torch`` is imported because the shim patches its
``randn``/``randn_like``, and the guard turns an unusable ROCm/torch stack into a
skip rather than a collection error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests._rocm_guard import torch_or_skip

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vibevoice_tts"
REQUEST = "single"


def _fixtures() -> tuple[dict, dict]:
    diff = dict(np.load(FIXTURES / f"{REQUEST}_diffusion.npz"))
    ref = dict(np.load(FIXTURES / f"{REQUEST}_reference.npz"))
    return diff, ref


def test_recorded_operands_answer_their_own_draws():
    torch = torch_or_skip("torch lane recorded-operand shim")
    from scripts.vibevoice_tts_torch_bench import install_recorded_operands

    diff, ref = _fixtures()
    n_calls = int(diff["num_calls_recorded"])
    real_randn, real_randn_like = torch.randn, torch.randn_like

    uninstall, served = install_recorded_operands(diff, ref, "cpu", torch.float32)
    try:
        # Voice-prompt VAE: randn(batch_size) then randn_like(mean).
        scale = torch.randn(int(ref["encode_draw0"].shape[0]))
        latent = torch.randn_like(torch.zeros_like(torch.as_tensor(ref["encode_draw1"])))
        # Diffusion: one randn(2, vae_dim) per call, in call order.
        frames = [torch.randn(2, 64) for _ in range(n_calls)]
    finally:
        uninstall()

    assert np.array_equal(scale.numpy(), ref["encode_draw0"])
    assert np.array_equal(latent.numpy(), ref["encode_draw1"])
    for i, got in enumerate(frames):
        assert np.array_equal(got.numpy(), diff[f"call{i}_initial_noise"]), i
    assert served == {
        "encode_draw0": 1,
        "encode_draw1": 1,
        "frame_noise": n_calls,
        "unmatched_frames": 0,
    }
    # The hooks must come back off, or every later draw in the process is shimmed.
    assert torch.randn is real_randn and torch.randn_like is real_randn_like


def test_unrecognised_shapes_fall_through_and_exhaustion_is_reported():
    torch = torch_or_skip("torch lane recorded-operand shim")
    from scripts.vibevoice_tts_torch_bench import install_recorded_operands

    diff, ref = _fixtures()
    n_calls = int(diff["num_calls_recorded"])

    uninstall, served = install_recorded_operands(diff, ref, "cpu", torch.float32)
    try:
        # A shape the shim does not know must reach the real RNG, not be mistaken
        # for a recorded operand.
        stray = torch.randn(7, 3)
        assert tuple(stray.shape) == (7, 3)
        assert served["frame_noise"] == 0
        # More diffusion frames than the fixture recorded must be reported, not
        # wrapped around to call0.
        for _ in range(n_calls + 1):
            torch.randn(2, 64)
    finally:
        uninstall()

    assert served["frame_noise"] == n_calls
    assert served["unmatched_frames"] == 1


def test_the_shim_is_a_noop_when_the_operands_are_not_used():
    """A lane that never calls randn must leave ``served`` at zero."""

    torch = torch_or_skip("torch lane recorded-operand shim")
    from scripts.vibevoice_tts_torch_bench import install_recorded_operands

    diff, ref = _fixtures()
    uninstall, served = install_recorded_operands(diff, ref, "cpu", torch.float32)
    uninstall()
    assert served == {
        "encode_draw0": 0,
        "encode_draw1": 0,
        "frame_noise": 0,
        "unmatched_frames": 0,
    }


def test_fixture_records_the_three_draw_kinds_the_shim_serves():
    """Guard the fixture itself: the shim's shapes come from these keys."""

    diff, ref = _fixtures()
    assert ref["encode_draw0"].shape == (1,)
    assert ref["encode_draw1"].shape == ref["encode0_mean"].shape
    n_calls = int(diff["num_calls_recorded"])
    assert n_calls == 25
    for i in range(n_calls):
        assert diff[f"call{i}_initial_noise"].shape == (2, 64)
        assert int(diff[f"call{i}_noise_draw_count"]) == 1
