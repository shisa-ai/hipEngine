"""Qualified DP-5 publication scope and native engagement accounting."""
from hipengine.kernels.registry import KernelKey

ENV = "HIPENGINE_QWEN4_EXP_GR_NORM_F16_PUBLISH"
PRODUCER_KEY = KernelKey("hip_gfx1151", "gr_grouped_rmsnorm", "bf16_f32", "f32_f16_publish")


def expected_calls(tokens: int, chunk: int) -> int:
    if tokens < 1 or chunk < 1:
        raise ValueError("positive tokens and chunk required")
    full, tail = divmod(tokens, chunk)
    return 70 * (full * (chunk == 4096) + int(tail == 4096))


def validate_producer_call(args) -> None:
    if len(args) != 8 or tuple(args[4:7]) != (4096, 4, 2560):
        raise ValueError("publisher escaped qualified rows4096/branches4/H2560")


def heldout_tokens(tokens):
    """Uniform repeat/truncate to4096 for ownership coverage, not task scoring."""
    if not tokens:
        raise ValueError("empty heldout")
    return (list(tokens) * ((4095 // len(tokens)) + 1))[:4096]
