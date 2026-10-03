"""Gemma 4 gfx1100 kernel and rotary-table contracts.

The rotary-table tests need no GPU: they pin the layout that
``qwen35_partial_rotary_kernel`` indexes against HuggingFace's ``rotate_half``
form, which is the reason no new rotary kernel is needed. The kernel tests are
guarded so a no-ROCm runner skips them rather than failing release validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4RopeConfig,
    _apply_rope,
    gemma4_experts_forward,
    gemma4_gelu_tanh,
    gemma4_rope_tables,
    gemma4_rmsnorm,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rope import (
    GEMMA4_ROPE_DEFAULT_TYPE,
    GEMMA4_ROPE_PROPORTIONAL_TYPE,
    gemma4_rope_angles,
    gemma4_rope_cos_sin_tables,
    gemma4_rotate_split_half,
)
from tests._rocm_guard import hip_runtime_available

# ---------------------------------------------------------------------------
# Rotary tables: the claim that the existing rotary kernel fits Gemma 4
# ---------------------------------------------------------------------------


def _hf_rotate_half(value: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """HuggingFace's rotation, written the way ``modeling_gemma4`` does it."""

    width = value.shape[-1]
    first = value[..., : width // 2]
    second = value[..., width // 2 :]
    rotated = np.concatenate((-second, first), axis=-1)
    return (value * cos + rotated * sin).astype(np.float32)


@pytest.mark.parametrize(
    ("rope_type", "head_dim", "factor", "theta"),
    [
        (GEMMA4_ROPE_DEFAULT_TYPE, 256, 1.0, 10_000.0),
        (GEMMA4_ROPE_PROPORTIONAL_TYPE, 512, 0.25, 1_000_000.0),
    ],
)
def test_doubled_table_reproduces_huggingface_rotation(
    rope_type: str, head_dim: int, factor: float, theta: float
) -> None:
    """The kernel's formula with this table equals HF's ``rotate_half`` form."""

    rope = Gemma4RopeConfig(
        rope_theta=theta,
        head_dim=head_dim,
        rope_angles=gemma4_rope_angles(
            head_dim=head_dim, partial_rotary_factor=factor, rope_type=rope_type
        ),
        rope_type=rope_type,
    )
    positions = np.array([0, 1, 7, 33], dtype=np.int64)
    cos, sin = gemma4_rope_cos_sin_tables(rope, positions)
    assert cos.shape == (4, head_dim)
    assert sin.shape == (4, head_dim)

    rng = np.random.default_rng(0)
    value = rng.standard_normal((4, head_dim)).astype(np.float32)
    assert np.allclose(
        gemma4_rotate_split_half(value, cos, sin),
        _hf_rotate_half(value, cos, sin),
        atol=1e-6,
    )


def test_proportional_pairs_outside_the_span_are_untouched() -> None:
    """Rotated elements are two spans, not a contiguous prefix."""

    head_dim = 512
    rope_angles = gemma4_rope_angles(
        head_dim=head_dim,
        partial_rotary_factor=0.25,
        rope_type=GEMMA4_ROPE_PROPORTIONAL_TYPE,
    )
    assert rope_angles == 64

    rope = Gemma4RopeConfig(
        rope_theta=1_000_000.0,
        head_dim=head_dim,
        rope_angles=rope_angles,
        rope_type=GEMMA4_ROPE_PROPORTIONAL_TYPE,
    )
    cos, sin = gemma4_rope_cos_sin_tables(rope, np.array([5], dtype=np.int64))
    # The table is the doubled form, so the tail of each half is the unrotated
    # span and must carry a unit cosine and a zero sine.
    assert np.allclose(cos[0, rope_angles : head_dim // 2], 1.0)
    assert np.allclose(sin[0, rope_angles : head_dim // 2], 0.0)
    assert np.allclose(cos[0, head_dim // 2 + rope_angles :], 1.0)
    assert np.allclose(sin[0, head_dim // 2 + rope_angles :], 0.0)
    # And the rotated spans must actually rotate.
    assert not np.allclose(sin[0, :rope_angles], 0.0)
    assert not np.allclose(sin[0, head_dim // 2 : head_dim // 2 + rope_angles], 0.0)


def test_rotation_is_orthogonal_at_every_position() -> None:
    """A rotation preserves the pair norm, which a wrong pairing would break."""

    rope = Gemma4RopeConfig(
        rope_theta=1_000_000.0,
        head_dim=512,
        rope_angles=64,
        rope_type=GEMMA4_ROPE_PROPORTIONAL_TYPE,
    )
    positions = np.arange(0, 64, 7, dtype=np.int64)
    cos, sin = gemma4_rope_cos_sin_tables(rope, positions)
    rng = np.random.default_rng(3)
    value = rng.standard_normal((positions.shape[0], 512)).astype(np.float32)
    rotated = gemma4_rotate_split_half(value, cos, sin)
    assert np.allclose(np.linalg.norm(rotated, axis=-1), np.linalg.norm(value, axis=-1), atol=1e-4)


def test_default_layers_rotate_every_pair() -> None:
    assert (
        gemma4_rope_angles(
            head_dim=256,
            partial_rotary_factor=0.25,
            rope_type=GEMMA4_ROPE_DEFAULT_TYPE,
        )
        == 128
    )


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

_HIP_AVAILABLE = hip_runtime_available()
_needs_hip = pytest.mark.skipif(
    not _HIP_AVAILABLE, reason="HIP runtime unavailable; skipping gfx1100 kernel tests"
)

if _HIP_AVAILABLE:
    from hipengine.core.memory import (
        DeviceBuffer,
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        build_gemma4_norm,
        gemma4_add_rmsnorm_scale_bf16,
        gemma4_branch_add_bf16,
        gemma4_expert_weight_scale_f32,
        gemma4_head_rmsnorm_f32w_bf16,
        gemma4_rmsnorm_f32w_bf16,
        gemma4_rmsnorm_f32w_f32,
        gemma4_rmsnorm_weightless_bf16,
        gemma4_router_prescale_bf16,
        gemma4_scale_bf16,
    )


def _to_bf16_bits(array: np.ndarray) -> np.ndarray:
    """Pack float32 values into the bfloat16 bit patterns the kernels read.

    The kernels take ``const uint16_t*`` for activations, so the device buffer
    must hold two-byte bf16 values. Passing a float32 buffer would hand the
    kernel interleaved garbage.
    """

    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    """Widen bfloat16 bit patterns back to float32."""

    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


# bf16 carries about three decimal digits, so an element of magnitude M is only
# good to roughly M * 2**-9. The tolerances below are sized for that, and every
# test that could be fooled by a wrong formula also asserts the wrong formula is
# rejected.
_BF16_ATOL = 0.01
_BF16_RTOL = 0.02


class _Device:
    """Owns one device allocation per host array and frees them together."""

    def __init__(self) -> None:
        self._buffers: list[DeviceBuffer] = []

    def put(self, array: np.ndarray) -> int:
        host = np.ascontiguousarray(array)
        buffer = malloc(host.nbytes)
        copy_host_to_device(buffer, host_array_ptr(host), host.nbytes)
        self._buffers.append(buffer)
        return buffer.ptr

    def get(self, ptr: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        buffer = DeviceBuffer(ptr=ptr, nbytes=out.nbytes)
        copy_device_to_host(host_array_ptr(out), buffer, out.nbytes)
        return out

    def out(self, shape: tuple[int, ...], dtype: np.dtype) -> int:
        return self.put(np.zeros(shape, dtype=dtype))

    def close(self) -> None:
        for buffer in self._buffers:
            free(buffer)
        self._buffers.clear()


@pytest.fixture()
def device():
    owner = _Device()
    try:
        yield owner
    finally:
        owner.close()


@_needs_hip
def test_library_builds_and_exports_the_family(device) -> None:
    library = build_gemma4_norm(load=True)
    for symbol in (
        "hipengine_gemma4_rmsnorm_f32w_bf16",
        "hipengine_gemma4_rmsnorm_f32w_f32",
        "hipengine_gemma4_rmsnorm_weightless_bf16",
        "hipengine_gemma4_head_rmsnorm_f32w_bf16",
        "hipengine_gemma4_router_prescale_bf16",
        "hipengine_gemma4_router_topk_fused_bf16",
        "hipengine_gemma4_add_rmsnorm_scale_bf16",
        "hipengine_gemma4_expert_weight_scale_f32",
        "hipengine_gemma4_branch_add_bf16",
        "hipengine_gemma4_scale_bf16",
    ):
        assert hasattr(library, symbol), symbol


@_needs_hip
def test_plain_weight_rmsnorm_matches_the_reference_in_f32(device) -> None:
    """The weight is applied as-is, not as ``1 + weight``."""

    rng = np.random.default_rng(11)
    rows, hidden = 5, 256
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)
    # Weights near zero separate the plain form from the Gemma-2 form: adding
    # one would scale every output by roughly the same factor and be obvious.
    weight = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    eps = 1e-6

    src = device.put(hidden_states)
    w = device.put(weight)
    dst = device.out((rows, hidden), np.float32)
    gemma4_rmsnorm_f32w_f32(src, w, dst, rows, hidden, eps)
    got = device.get(dst, (rows, hidden), np.float32)

    expected = gemma4_rmsnorm(hidden_states, weight, eps)
    assert np.allclose(got, expected, atol=1e-5)

    # Guard the specific failure this kernel exists to avoid.
    gemma2_form = gemma4_rmsnorm(hidden_states, weight + 1.0, eps)
    assert not np.allclose(got, gemma2_form, atol=1e-3)


@_needs_hip
def test_plain_weight_rmsnorm_matches_the_reference_in_bf16(device) -> None:
    rng = np.random.default_rng(12)
    rows, hidden = 4, 512
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)
    weight = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
    eps = 1e-6

    src = device.put(_to_bf16_bits(hidden_states))
    w = device.put(weight)
    dst = device.out((rows, hidden), np.uint16)
    gemma4_rmsnorm_f32w_bf16(src, w, dst, rows, hidden, eps)
    got = _from_bf16_bits(device.get(dst, (rows, hidden), np.uint16))

    expected = gemma4_rmsnorm(hidden_states, weight, eps)
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)

    # Guard the specific failure this kernel exists to avoid. Weights near zero
    # make the Gemma-2 form diverge by a large factor rather than a constant.
    gemma2_form = gemma4_rmsnorm(hidden_states, weight + 1.0, eps)
    assert not np.allclose(got, gemma2_form, atol=1e-2, rtol=1e-2)


@_needs_hip
def test_weightless_rmsnorm_matches_the_reference(device) -> None:
    """The value norm and the router norm have no weight tensor at all."""

    rng = np.random.default_rng(13)
    rows, hidden = 3, 512
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)

    src = device.put(_to_bf16_bits(hidden_states))
    dst = device.out((rows, hidden), np.uint16)
    gemma4_rmsnorm_weightless_bf16(src, dst, rows, hidden, 1e-6)
    got = _from_bf16_bits(device.get(dst, (rows, hidden), np.uint16))

    expected = gemma4_rmsnorm(hidden_states, None, 1e-6)
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_multi_rmsnorm_matches_the_chain_bitwise_same_input(device) -> None:
    """D6 group G3: three outputs over ONE input must equal the chain bitwise.

    The three normalizations that read the same residual row (pre-FFN, pre-FFN-2,
    the weightless router norm) may be emitted by one launch instead of three.
    The contract is not approximate equality: the fused kernel must compute each
    output with the same reduction tree and expression as its standalone kernel,
    so every bit matches. A tolerance would hide exactly the drift this fusion
    must not introduce.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_multi_rmsnorm_bf16,
    )

    rng = np.random.default_rng(51)
    rows, hidden = 3, 2816  # the real decode geometry: one block, 11 per thread
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)
    weight_a = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    weight_b = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    eps = 1e-6

    src = device.put(_to_bf16_bits(hidden_states))
    wa = device.put(weight_a)
    wb = device.put(weight_b)

    chain = [device.out((rows, hidden), np.uint16) for _ in range(3)]
    gemma4_rmsnorm_f32w_bf16(src, wa, chain[0], rows, hidden, eps)
    gemma4_rmsnorm_f32w_bf16(src, wb, chain[1], rows, hidden, eps)
    gemma4_rmsnorm_weightless_bf16(src, chain[2], rows, hidden, eps)

    fused = [device.out((rows, hidden), np.uint16) for _ in range(3)]
    gemma4_multi_rmsnorm_bf16(
        src, src, src, wa, wb, 0, fused[0], fused[1], fused[2],
        3, rows, hidden, eps,
    )

    for index, (before, after) in enumerate(zip(chain, fused)):
        assert np.array_equal(
            device.get(before, (rows, hidden), np.uint16),
            device.get(after, (rows, hidden), np.uint16),
        ), f"output {index} differs from the chain"


@_needs_hip
def test_multi_rmsnorm_matches_the_chain_bitwise_distinct_inputs(device) -> None:
    """D6 group G4: two different inputs in one launch, each bitwise its chain.

    post_ffw_norm_1 and post_ffw_norm_2 normalize different rows (dense vs
    expert output). One block may compute both sequentially only if each
    output's reduction is exactly the standalone one, and a null weight in any
    slot must behave as the weightless kernel does.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_multi_rmsnorm_bf16,
    )

    rng = np.random.default_rng(52)
    rows, hidden = 2, 2816
    dense = rng.standard_normal((rows, hidden)).astype(np.float32)
    expert = rng.standard_normal((rows, hidden)).astype(np.float32)
    weight = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    eps = 1e-6

    src_a = device.put(_to_bf16_bits(dense))
    src_b = device.put(_to_bf16_bits(expert))
    w = device.put(weight)

    chain_a = device.out((rows, hidden), np.uint16)
    chain_b = device.out((rows, hidden), np.uint16)
    chain_w = device.out((rows, hidden), np.uint16)
    gemma4_rmsnorm_f32w_bf16(src_a, w, chain_a, rows, hidden, eps)
    # The chain reference for a null weight slot IS the weightless kernel:
    # gemma4_rmsnorm_kernel dereferences weight unconditionally.
    gemma4_rmsnorm_weightless_bf16(src_b, chain_b, rows, hidden, eps)
    gemma4_rmsnorm_weightless_bf16(src_a, chain_w, rows, hidden, eps)

    out_a = device.out((rows, hidden), np.uint16)
    out_b = device.out((rows, hidden), np.uint16)
    out_w = device.out((rows, hidden), np.uint16)
    # Three distinct references: weighted on A, null-weight on B, weightless on A.
    gemma4_multi_rmsnorm_bf16(
        src_a, src_b, src_a, w, 0, 0, out_a, out_b, out_w,
        3, rows, hidden, eps,
    )

    for name, before, after in (
        ("weighted", chain_a, out_a),
        ("null-weight", chain_b, out_b),
        ("weightless", chain_w, out_w),
    ):
        assert np.array_equal(
            device.get(before, (rows, hidden), np.uint16),
            device.get(after, (rows, hidden), np.uint16),
        ), f"{name} output differs from the chain"


@_needs_hip
def test_multi_rmsnorm_writes_exactly_its_rows(device) -> None:
    """grid=rows must hold: a fencepost overrun poisons the row after the last."""
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_multi_rmsnorm_bf16,
    )

    rng = np.random.default_rng(53)
    rows, hidden = 4, 256
    hidden_states = rng.standard_normal((rows + 1, hidden)).astype(np.float32)
    sentinel = _to_bf16_bits(np.full((1, hidden), np.nan, dtype=np.float32))
    payload = _to_bf16_bits(hidden_states[:-1])
    device_array = np.concatenate([payload, sentinel]).astype(np.uint16)
    weight = (rng.standard_normal(hidden) * 0.05).astype(np.float32)

    src = device.put(device_array)
    w = device.put(weight)
    out = device.put(device_array.copy())
    gemma4_multi_rmsnorm_bf16(
        src, src, src, w, w, w, out, out, out, 1, rows, hidden, 1e-6,
    )

    got = device.get(out, (rows + 1, hidden), np.uint16)
    assert np.array_equal(got[-1], sentinel[0]), "kernel wrote past its rows"


@_needs_hip
def test_dense_combine_rmsnorm_matches_the_chain_bitwise(device) -> None:
    """D6 tail fold: post_ffw_norm_1 -> branch_add -> add_rmsnorm_scale in one launch.

    The three kernels are strictly sequential on the main stream and each
    intermediate has exactly one consumer, so one block may keep them in
    registers -- provided every intermediate passes through the same bf16
    rounding the standalone kernels would apply, and every reduction keeps its
    own tree. The assertion is bitwise against the chain; the null layer-scalar
    form is exercised too because Gemma 4 layers may carry no scale.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_dense_combine_rmsnorm_scale_bf16,
    )

    rng = np.random.default_rng(54)
    rows, hidden = 3, 2816
    dense = rng.standard_normal((rows, hidden)).astype(np.float32)
    experts = rng.standard_normal((rows, hidden)).astype(np.float32)
    residual = rng.standard_normal((rows, hidden)).astype(np.float32)
    w1 = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    eps = 1e-6

    def run_chain(scalar) -> np.ndarray:
        d = device.put(_to_bf16_bits(dense))
        e = device.put(_to_bf16_bits(experts))
        h = device.put(_to_bf16_bits(residual))
        w1p, w2p = device.put(w1), device.put(w2)
        s = device.put(np.array([scalar], dtype=np.float32)) if scalar is not None else 0
        bs = device.out((rows, hidden), np.uint16)
        out = device.out((rows, hidden), np.uint16)
        gemma4_rmsnorm_f32w_bf16(d, w1p, d, rows, hidden, eps)
        gemma4_branch_add_bf16(d, e, bs, rows * hidden)
        gemma4_add_rmsnorm_scale_bf16(bs, h, w2p, s, out, rows, hidden, eps)
        return device.get(out, (rows, hidden), np.uint16)

    for scalar in (1.0, None):
        d = device.put(_to_bf16_bits(dense))
        e = device.put(_to_bf16_bits(experts))
        h = device.put(_to_bf16_bits(residual))
        w1p, w2p = device.put(w1), device.put(w2)
        s = device.put(np.array([scalar], dtype=np.float32)) if scalar is not None else 0
        out = device.out((rows, hidden), np.uint16)
        gemma4_dense_combine_rmsnorm_scale_bf16(d, e, h, w1p, w2p, s, out, rows, hidden, eps)
        got = device.get(out, (rows, hidden), np.uint16)
        expected = run_chain(scalar)
        assert np.array_equal(got, expected), f"scalar={scalar}: fold differs from the chain"


@_needs_hip
def test_dense_combine_rmsnorm_writes_exactly_its_rows(device) -> None:
    """The fold writes the residual row in place; a fencepost overrun poisons row rows."""
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_dense_combine_rmsnorm_scale_bf16,
    )

    rng = np.random.default_rng(55)
    rows, hidden = 4, 256
    sentinel = _to_bf16_bits(np.full((1, hidden), np.nan, dtype=np.float32))
    dense = _to_bf16_bits(rng.standard_normal((rows, hidden)).astype(np.float32))
    experts = _to_bf16_bits(rng.standard_normal((rows, hidden)).astype(np.float32))
    residual = np.concatenate(
        [_to_bf16_bits(rng.standard_normal((rows, hidden)).astype(np.float32)), sentinel]
    ).astype(np.uint16)
    w1 = (rng.standard_normal(hidden) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal(hidden) * 0.05).astype(np.float32)

    d = device.put(dense)
    e = device.put(experts)
    h = device.put(residual.copy())
    w1p, w2p = device.put(w1), device.put(w2)
    gemma4_dense_combine_rmsnorm_scale_bf16(d, e, h, w1p, w2p, 0, h, rows, hidden, 1e-6)

    got = device.get(h, (rows + 1, hidden), np.uint16)
    assert np.array_equal(got[-1], sentinel[0]), "kernel wrote past its rows"


@_needs_hip
def test_head_rmsnorm_matches_a_per_head_reference(device) -> None:
    rng = np.random.default_rng(14)
    heads, head_dim = 6, 512
    hidden_states = rng.standard_normal((heads, head_dim)).astype(np.float32)
    weight = (rng.standard_normal(head_dim) * 0.5).astype(np.float32)

    src = device.put(_to_bf16_bits(hidden_states))
    w = device.put(weight)
    dst = device.out((heads, head_dim), np.uint16)
    gemma4_head_rmsnorm_f32w_bf16(src, w, dst, heads, head_dim, 1e-6)
    got = _from_bf16_bits(device.get(dst, (heads, head_dim), np.uint16))

    expected = gemma4_rmsnorm(hidden_states, weight, 1e-6)
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_head_rmsnorm_is_weightless_when_the_weight_is_null(device) -> None:
    rng = np.random.default_rng(15)
    heads, head_dim = 4, 256
    hidden_states = rng.standard_normal((heads, head_dim)).astype(np.float32)

    src = device.put(_to_bf16_bits(hidden_states))
    dst = device.out((heads, head_dim), np.uint16)
    gemma4_head_rmsnorm_f32w_bf16(src, 0, dst, heads, head_dim, 1e-6)
    got = _from_bf16_bits(device.get(dst, (heads, head_dim), np.uint16))

    assert np.allclose(
        got, gemma4_rmsnorm(hidden_states, None, 1e-6), atol=_BF16_ATOL, rtol=_BF16_RTOL
    )


@_needs_hip
def test_router_prescale_matches_the_reference_router_input(device) -> None:
    rng = np.random.default_rng(16)
    rows, hidden = 3, 256
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)
    scale = (rng.standard_normal(hidden) * 0.1 + 0.5).astype(np.float32)
    root_size = float(hidden**-0.5)

    src = device.put(_to_bf16_bits(hidden_states))
    s = device.put(scale)
    dst = device.out((rows, hidden), np.uint16)
    gemma4_router_prescale_bf16(src, s, dst, rows, hidden, 1e-6, root_size=root_size)
    got = _from_bf16_bits(device.get(dst, (rows, hidden), np.uint16))

    expected = (gemma4_rmsnorm(hidden_states, None, 1e-6) * scale * np.float32(root_size)).astype(
        np.float32
    )
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)

    # Dropping either scalar must change the result.
    missing_scale = (gemma4_rmsnorm(hidden_states, None, 1e-6) * np.float32(root_size)).astype(
        np.float32
    )
    assert not np.allclose(got, missing_scale, atol=1e-2, rtol=1e-2)


@_needs_hip
def test_add_rmsnorm_scale_applies_the_layer_scalar(device) -> None:
    rng = np.random.default_rng(17)
    rows, hidden = 4, 256
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)
    residual = rng.standard_normal((rows, hidden)).astype(np.float32)
    weight = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
    scalar = np.array([0.0703125], dtype=np.float32)

    src = device.put(_to_bf16_bits(hidden_states))
    res = device.put(_to_bf16_bits(residual))
    w = device.put(weight)
    sc = device.put(scalar)
    dst = device.out((rows, hidden), np.uint16)
    gemma4_add_rmsnorm_scale_bf16(src, res, w, sc, dst, rows, hidden, 1e-6)
    got = _from_bf16_bits(device.get(dst, (rows, hidden), np.uint16))

    expected = ((residual + gemma4_rmsnorm(hidden_states, weight, 1e-6)) * scalar[0]).astype(
        np.float32
    )
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)

    # The scalar is real, not one: skipping it must change the result.
    unscaled = (residual + gemma4_rmsnorm(hidden_states, weight, 1e-6)).astype(np.float32)
    assert not np.allclose(got, unscaled, atol=1e-2, rtol=1e-2)


@_needs_hip
def test_add_rmsnorm_scale_skips_the_scalar_when_null(device) -> None:
    rng = np.random.default_rng(18)
    rows, hidden = 2, 256
    hidden_states = rng.standard_normal((rows, hidden)).astype(np.float32)
    residual = rng.standard_normal((rows, hidden)).astype(np.float32)
    weight = (rng.standard_normal(hidden) * 0.5).astype(np.float32)

    src = device.put(_to_bf16_bits(hidden_states))
    res = device.put(_to_bf16_bits(residual))
    w = device.put(weight)
    dst = device.out((rows, hidden), np.uint16)
    gemma4_add_rmsnorm_scale_bf16(src, res, w, 0, dst, rows, hidden, 1e-6)
    got = _from_bf16_bits(device.get(dst, (rows, hidden), np.uint16))

    expected = (residual + gemma4_rmsnorm(hidden_states, weight, 1e-6)).astype(np.float32)
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_expert_weight_scale_targets_the_selected_experts(device) -> None:
    """The scale lands on the selected slot's own expert, not on the slot."""

    weights = np.array([[0.5, 0.25, 0.25]], dtype=np.float32)
    selected = np.array([[7, 2, 9]], dtype=np.int64)
    per_expert_scale = np.arange(1, 17, dtype=np.float32)

    w = device.put(weights)
    sel = device.put(selected)
    pes = device.put(per_expert_scale)
    gemma4_expert_weight_scale_f32(w, sel, pes, 1, 3)
    got = device.get(w, (1, 3), np.float32)

    expected = weights * per_expert_scale[selected]
    assert np.allclose(got, expected)


@_needs_hip
def test_branch_add_sums_the_dense_and_expert_branches(device) -> None:
    rng = np.random.default_rng(19)
    total = 512
    a = rng.standard_normal(total).astype(np.float32)
    b = rng.standard_normal(total).astype(np.float32)

    ap = device.put(_to_bf16_bits(a))
    bp = device.put(_to_bf16_bits(b))
    out = device.out((total,), np.uint16)
    gemma4_branch_add_bf16(ap, bp, out, total)
    got = _from_bf16_bits(device.get(out, (total,), np.uint16))

    assert np.allclose(got, (a + b).astype(np.float32), atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
@pytest.mark.parametrize("scale", [1.0, 53.066, 0.5, -2.0, 0.0])
def test_scale_multiplies_a_row_by_a_scalar(device, scale: float) -> None:
    """The embedding scale, against a NumPy multiply.

    ``sqrt(hidden_size)`` for the real model is 53.066, so that is the case that
    matters; the others check that the multiply is a multiply rather than
    something that happens to be right at one factor. ``0.0`` is included
    because the reference can produce it and a kernel that special-cased zero
    would be wrong in a way a non-zero test would not see.
    """

    rng = np.random.default_rng(23)
    rows, hidden_size = 7, 96
    x = rng.standard_normal((rows, hidden_size)).astype(np.float32)

    xp = device.put(_to_bf16_bits(x))
    out = device.out((rows, hidden_size), np.uint16)
    gemma4_scale_bf16(xp, out, rows, hidden_size, scale)
    got = _from_bf16_bits(device.get(out, (rows, hidden_size), np.uint16))

    # The oracle multiplies the bf16-rounded input, not the original f32, so the
    # only error the test attributes to the kernel is the output rounding.
    expected = _from_bf16_bits(_to_bf16_bits(x)) * np.float32(scale)
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_scale_is_correct_in_place(device) -> None:
    """``out == x`` is allowed, so the runner can scale the embedding buffer."""

    rng = np.random.default_rng(29)
    rows, hidden_size = 4, 64
    x = rng.standard_normal((rows, hidden_size)).astype(np.float32)
    scale = float(np.sqrt(hidden_size))

    buf = device.put(_to_bf16_bits(x))
    gemma4_scale_bf16(buf, buf, rows, hidden_size, scale)
    got = _from_bf16_bits(device.get(buf, (rows, hidden_size), np.uint16))

    expected = _from_bf16_bits(_to_bf16_bits(x)) * np.float32(scale)
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_kernels_reject_degenerate_shapes(device) -> None:
    src = device.put(np.zeros(8, dtype=np.float32))
    dst = device.out((8,), np.float32)
    with pytest.raises(ValueError):
        gemma4_rmsnorm_f32w_f32(src, src, dst, 0, 8, 1e-6)
    with pytest.raises(ValueError):
        gemma4_branch_add_bf16(src, src, dst, 0)
    with pytest.raises(ValueError):
        gemma4_scale_bf16(src, dst, 0, 8, 2.0)
    with pytest.raises(ValueError):
        gemma4_scale_bf16(src, dst, 8, 0, 2.0)


def test_the_family_registers_against_the_four_axis_registry() -> None:
    """The Gemma 4 norms are reachable by registry key, not by a backend branch."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_add_rmsnorm_scale_bf16,
        gemma4_branch_add_bf16,
        gemma4_head_rmsnorm_f32w_bf16,
        gemma4_rmsnorm_f32w_bf16,
        gemma4_rmsnorm_weightless_bf16,
        gemma4_router_prescale_bf16,
        register_gemma4_norm_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_gemma4_norm_kernels(replace=True)
    expected = {
        "rmsnorm": gemma4_rmsnorm_f32w_bf16,
        "rmsnorm_weightless": gemma4_rmsnorm_weightless_bf16,
        "head_rmsnorm": gemma4_head_rmsnorm_f32w_bf16,
        "router_prescale": gemma4_router_prescale_bf16,
        "add_rmsnorm_scale": gemma4_add_rmsnorm_scale_bf16,
        "branch_add": gemma4_branch_add_bf16,
    }
    for layer, function in expected.items():
        kernel = resolve(
            backend="hip_gfx1100",
            layer=layer,
            quant="gguf_q4_k_m",
            variant="gemma4_plain",
        )
        assert kernel is function, f"{layer} resolved to {kernel!r}"


def test_a_bare_variant_does_not_reach_the_gemma4_norm() -> None:
    """Why the Gemma 4 runner must pin ``gemma4_plain`` on every norm call.

    ``resolve`` falls back from the exact key to the same key with an empty
    variant, then to the fp16 quant, then to the ``cpu_reference`` backend. Gemma
    4's norm applies its weight as-is while the Qwen3.5 family next door applies
    ``1 + weight``, so a fallback hit is a silent correctness bug rather than a
    missing kernel. This test records that the fallback does not land on Gemma 4.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
        gemma4_rmsnorm_f32w_bf16,
        register_gemma4_norm_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_gemma4_norm_kernels(replace=True)
    resolved = resolve(
        backend="hip_gfx1100",
        layer="rmsnorm",
        quant="gguf_q4_k_m",
        variant="",
        missing="none",
    )
    assert resolved is not gemma4_rmsnorm_f32w_bf16

    # An unregistered variant for the same layer and quant also misses, which is
    # the case a typo in the runner would produce.
    wrong = resolve(
        backend="hip_gfx1100",
        layer="rmsnorm",
        quant="gguf_q4_k_m",
        variant="gemma4_plian",
        missing="none",
    )
    assert wrong is not gemma4_rmsnorm_f32w_bf16


# ---------------------------------------------------------------------------
# Partial rotary and the K-to-V copy
# ---------------------------------------------------------------------------


def _rotary_rope(rope_type: str, head_dim: int, factor: float, theta: float):
    return Gemma4RopeConfig(
        rope_theta=theta,
        head_dim=head_dim,
        rope_angles=gemma4_rope_angles(
            head_dim=head_dim, partial_rotary_factor=factor, rope_type=rope_type
        ),
        rope_type=rope_type,
    )


@_needs_hip
@pytest.mark.parametrize(
    ("rope_type", "head_dim", "factor", "theta"),
    [
        (GEMMA4_ROPE_DEFAULT_TYPE, 256, 1.0, 10_000.0),
        (GEMMA4_ROPE_PROPORTIONAL_TYPE, 512, 0.25, 1_000_000.0),
    ],
)
def test_gpu_rotary_f32_matches_the_reference_rotation(
    device, rope_type: str, head_dim: int, factor: float, theta: float
) -> None:
    """The kernel reproduces the reference rotation on both layer geometries."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
        gemma4_partial_rotary_f32,
    )

    tokens, num_q_heads, num_kv_heads = 3, 4, 2
    positions = np.array([0, 5, 11], dtype=np.int64)
    rope = _rotary_rope(rope_type, head_dim, factor, theta)
    half_cos, half_sin = gemma4_rope_tables(rope, positions)
    full_cos, full_sin = gemma4_rope_cos_sin_tables(rope, positions)

    rng = np.random.default_rng(21)
    query = rng.standard_normal((tokens, num_q_heads, head_dim)).astype(np.float32)
    key = rng.standard_normal((tokens, num_kv_heads, head_dim)).astype(np.float32)

    qp = device.put(query)
    kp = device.put(key)
    cp = device.put(full_cos)
    sp = device.put(full_sin)
    qo = device.out(query.shape, np.float32)
    ko = device.out(key.shape, np.float32)
    gemma4_partial_rotary_f32(qp, kp, cp, sp, qo, ko, tokens, num_q_heads, num_kv_heads, head_dim)

    expected_q = _apply_rope(query, half_cos[:, None, :], half_sin[:, None, :], head_dim)
    expected_k = _apply_rope(key, half_cos[:, None, :], half_sin[:, None, :], head_dim)
    assert np.allclose(device.get(qo, query.shape, np.float32), expected_q, atol=1e-5)
    assert np.allclose(device.get(ko, key.shape, np.float32), expected_k, atol=1e-5)


@_needs_hip
def test_gpu_rotary_bf16_matches_the_reference_rotation(device) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
        gemma4_partial_rotary_bf16,
    )

    tokens, num_q_heads, num_kv_heads, head_dim = 4, 4, 2, 512
    positions = np.arange(tokens, dtype=np.int64)
    rope = _rotary_rope(GEMMA4_ROPE_PROPORTIONAL_TYPE, head_dim, 0.25, 1_000_000.0)
    half_cos, half_sin = gemma4_rope_tables(rope, positions)
    full_cos, full_sin = gemma4_rope_cos_sin_tables(rope, positions)

    rng = np.random.default_rng(22)
    query = rng.standard_normal((tokens, num_q_heads, head_dim)).astype(np.float32)
    key = rng.standard_normal((tokens, num_kv_heads, head_dim)).astype(np.float32)

    qp = device.put(_to_bf16_bits(query))
    kp = device.put(_to_bf16_bits(key))
    cp = device.put(full_cos)
    sp = device.put(full_sin)
    qo = device.out(query.shape, np.uint16)
    ko = device.out(key.shape, np.uint16)
    gemma4_partial_rotary_bf16(qp, kp, cp, sp, qo, ko, tokens, num_q_heads, num_kv_heads, head_dim)

    expected_q = _apply_rope(query, half_cos[:, None, :], half_sin[:, None, :], head_dim)
    expected_k = _apply_rope(key, half_cos[:, None, :], half_sin[:, None, :], head_dim)
    got_q = _from_bf16_bits(device.get(qo, query.shape, np.uint16))
    got_k = _from_bf16_bits(device.get(ko, key.shape, np.uint16))
    assert np.allclose(got_q, expected_q, atol=_BF16_ATOL, rtol=_BF16_RTOL)
    assert np.allclose(got_k, expected_k, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_gpu_rotary_leaves_the_unrotated_span_bit_identical(device) -> None:
    """The proportional tail must pass through, not merely land nearby.

    A contiguous-prefix rotary would rotate elements 64..255 of each half, which
    is the failure this asserts against.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
        gemma4_partial_rotary_f32,
    )

    head_dim = 512
    rope_angles = 64
    tokens, num_q_heads, num_kv_heads = 2, 2, 1
    positions = np.array([3, 9], dtype=np.int64)
    rope = _rotary_rope(GEMMA4_ROPE_PROPORTIONAL_TYPE, head_dim, 0.25, 1_000_000.0)
    full_cos, full_sin = gemma4_rope_cos_sin_tables(rope, positions)

    rng = np.random.default_rng(23)
    query = rng.standard_normal((tokens, num_q_heads, head_dim)).astype(np.float32)
    key = np.zeros((tokens, num_kv_heads, head_dim), dtype=np.float32)

    qp = device.put(query)
    kp = device.put(key)
    cp = device.put(full_cos)
    sp = device.put(full_sin)
    qo = device.out(query.shape, np.float32)
    ko = device.out(key.shape, np.float32)
    gemma4_partial_rotary_f32(qp, kp, cp, sp, qo, ko, tokens, num_q_heads, num_kv_heads, head_dim)
    got = device.get(qo, query.shape, np.float32)

    half = head_dim // 2
    tail = np.s_[..., rope_angles:half]
    tail_upper = np.s_[..., half + rope_angles :]
    assert np.array_equal(got[tail], query[tail]), "lower unrotated span moved"
    assert np.array_equal(got[tail_upper], query[tail_upper]), "upper unrotated span moved"

    # And the rotated spans must actually have moved.
    assert not np.allclose(got[..., :rope_angles], query[..., :rope_angles])
    assert not np.allclose(
        got[..., half : half + rope_angles], query[..., half : half + rope_angles]
    )


@_needs_hip
def test_gpu_rotary_refuses_a_contiguous_prefix_width(device) -> None:
    """A rotary_dim below head_dim would rotate the wrong pairs."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
        gemma4_partial_rotary_f32,
    )

    src = device.put(np.zeros((1, 1, 512), dtype=np.float32))
    dst = device.out((1, 1, 512), np.float32)
    with pytest.raises(ValueError, match="rotary_dim must be head_dim"):
        gemma4_partial_rotary_f32(src, 0, src, src, dst, dst, 1, 1, 0, 512, rotary_dim=128)


@_needs_hip
def test_gpu_rotary_accepts_a_null_key(device) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
        gemma4_partial_rotary_f32,
    )

    tokens, num_q_heads, head_dim = 2, 2, 256
    positions = np.arange(tokens, dtype=np.int64)
    rope = _rotary_rope(GEMMA4_ROPE_DEFAULT_TYPE, head_dim, 1.0, 10_000.0)
    half_cos, half_sin = gemma4_rope_tables(rope, positions)
    full_cos, full_sin = gemma4_rope_cos_sin_tables(rope, positions)

    rng = np.random.default_rng(24)
    query = rng.standard_normal((tokens, num_q_heads, head_dim)).astype(np.float32)
    qp = device.put(query)
    cp = device.put(full_cos)
    sp = device.put(full_sin)
    qo = device.out(query.shape, np.float32)
    gemma4_partial_rotary_f32(qp, 0, cp, sp, qo, 0, tokens, num_q_heads, 0, head_dim)
    expected = _apply_rope(query, half_cos[:, None, :], half_sin[:, None, :], head_dim)
    assert np.allclose(device.get(qo, query.shape, np.float32), expected, atol=1e-5)


@_needs_hip
def test_k_to_v_copies_the_raw_projection(device) -> None:
    """attention_k_eq_v makes V the raw K projection, before k_norm and rope."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import gemma4_k_to_v_bf16

    rng = np.random.default_rng(25)
    key = rng.standard_normal((3, 2, 512)).astype(np.float32)
    kp = device.put(_to_bf16_bits(key))
    vp = device.out(key.shape, np.uint16)
    gemma4_k_to_v_bf16(kp, vp, int(np.prod(key.shape)))
    got = _from_bf16_bits(device.get(vp, key.shape, np.uint16))
    assert np.array_equal(got, _from_bf16_bits(_to_bf16_bits(key)))


def test_rotary_family_registers_under_gemma4_plain() -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_rotary import (
        gemma4_k_to_v_bf16,
        gemma4_partial_rotary_bf16,
        register_gemma4_rotary_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_gemma4_rotary_kernels(replace=True)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="partial_rotary",
            quant="gguf_q4_k_m",
            variant="gemma4_plain",
        )
        is gemma4_partial_rotary_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="k_to_v",
            quant="gguf_q4_k_m",
            variant="gemma4_plain",
        )
        is gemma4_k_to_v_bf16
    )


# ---------------------------------------------------------------------------
# Routed-expert FFN primitives
# ---------------------------------------------------------------------------


@_needs_hip
def test_gelu_tanh_mul_matches_the_reference_activation(device) -> None:
    """GeGLU is gelu_tanh(gate) * up, not silu(gate) * up."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_gelu_tanh_mul_bf16,
    )

    rows, intermediate = 5, 64
    rng = np.random.default_rng(31)
    gate = rng.standard_normal((rows, intermediate)).astype(np.float32) * 3.0
    up = rng.standard_normal((rows, intermediate)).astype(np.float32)
    fused = np.concatenate([gate, up], axis=1)

    fused_ptr = device.put(_to_bf16_bits(fused))
    out = device.out((rows, intermediate), np.uint16)
    gemma4_gelu_tanh_mul_bf16(fused_ptr, out, rows, intermediate)
    got = _from_bf16_bits(device.get(out, (rows, intermediate), np.uint16))

    expected = gemma4_gelu_tanh(gate) * up
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)

    # And it must not be SwiGLU, which is the plausible wrong reuse.
    silu = gate / (1.0 + np.exp(-gate))
    assert not np.allclose(got, silu * up, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_gelu_tanh_mul_keeps_gate_and_up_halves_distinct(device) -> None:
    """The second half of the fused buffer must be read as `up`, not as gate."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_gelu_tanh_mul_bf16,
    )

    rows, intermediate = 2, 32
    gate = np.full((rows, intermediate), 0.75, dtype=np.float32)
    up = np.full((rows, intermediate), -2.5, dtype=np.float32)
    fused = np.concatenate([gate, up], axis=1)
    fused_ptr = device.put(_to_bf16_bits(fused))
    out = device.out((rows, intermediate), np.uint16)
    gemma4_gelu_tanh_mul_bf16(fused_ptr, out, rows, intermediate)
    got = _from_bf16_bits(device.get(out, (rows, intermediate), np.uint16))
    expected = gemma4_gelu_tanh(gate) * up
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)
    # Reading gate twice would give gelu(0.75) * 0.75, a positive number.
    assert np.all(got < 0)


@_needs_hip
def test_weighted_accumulate_matches_the_reference_with_shared_tokens(device) -> None:
    """Two lanes of one token must both land, with no lost update.

    This is the regression test for the block-per-row version of the kernel,
    which had different blocks read-modify-writing the same token row. With
    `top_k` lanes per token landing in different experts, that version dropped
    or double-counted contributions depending on block scheduling.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_moe_lane_to_row_i32,
        gemma4_moe_weighted_accumulate_bf16,
    )

    tokens, top_k, hidden = 4, 4, 128
    total_lanes = tokens * top_k
    rng = np.random.default_rng(32)

    # A compact order that deliberately splits each token's lanes across experts:
    # interleave so consecutive lanes of a token are far apart in row order.
    sorted_lanes = np.empty(total_lanes, dtype=np.int64)
    for token in range(tokens):
        for slot in range(top_k):
            sorted_lanes[slot * tokens + token] = token * top_k + slot
    weights = rng.random(total_lanes).astype(np.float32)
    expert_out = rng.standard_normal((total_lanes, hidden)).astype(np.float32)

    lanes_ptr = device.put(sorted_lanes)
    l2r_ptr = device.out((total_lanes,), np.int32)
    gemma4_moe_lane_to_row_i32(lanes_ptr, l2r_ptr, total_lanes)

    expert_ptr = device.put(_to_bf16_bits(expert_out))
    weights_ptr = device.put(weights)
    out_ptr = device.out((tokens, hidden), np.uint16)
    gemma4_moe_weighted_accumulate_bf16(
        expert_ptr, l2r_ptr, weights_ptr, out_ptr, tokens, hidden, top_k
    )
    got = _from_bf16_bits(device.get(out_ptr, (tokens, hidden), np.uint16))

    expected = np.zeros((tokens, hidden), dtype=np.float32)
    for row, lane in enumerate(sorted_lanes):
        expected[lane // top_k] += _from_bf16_bits(_to_bf16_bits(expert_out[row])) * weights[row]
    assert np.allclose(got, expected, atol=_BF16_ATOL, rtol=_BF16_RTOL)


@_needs_hip
def test_weighted_accumulate_is_bitwise_grid_independent(device) -> None:
    """D7: column tiling must not change one bit of the accumulate output.

    The decode launch is one block per token (tokens == 1) and measured
    17.8 us/launch against a ~2 us floor for neighbouring single-block
    kernels -- a latency-bound structure, not work. Splitting the column
    dimension across ``gridDim.y`` redistributes the SAME per-column math:
    each column's slot loop still runs in one thread in lane order, so every
    ``col_tiles >= 1`` form must be bit-identical to the one-block-per-token
    form. This test is the proof that makes the grid change safe -- it is
    stronger than the reference check above, which only pins the formula.

    ``col_tiles=0`` is the wrapper's auto choice (the shipped default), and
    the sentinel row catches a tile loop that overruns ``hidden``.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_moe_lane_to_row_i32,
        gemma4_moe_weighted_accumulate_bf16,
    )

    tokens, top_k, hidden = 3, 4, 300  # 300 is ragged against a 256-wide tile
    total_lanes = tokens * top_k
    rng = np.random.default_rng(77)
    sorted_lanes = rng.permutation(total_lanes).astype(np.int64)
    weights = rng.random(total_lanes).astype(np.float32)
    expert_out = rng.standard_normal((total_lanes, hidden)).astype(np.float32)

    lanes_ptr = device.put(sorted_lanes)
    l2r_ptr = device.out((total_lanes,), np.int32)
    gemma4_moe_lane_to_row_i32(lanes_ptr, l2r_ptr, total_lanes)
    expert_ptr = device.put(_to_bf16_bits(expert_out))
    weights_ptr = device.put(weights)

    # The numpy oracle replicates the kernel's f32 slot-order sum by hand so
    # the comparison can be bitwise, not allclose.
    l2r = device.get(l2r_ptr, (total_lanes,), np.int32)
    expert_out_bf16 = _from_bf16_bits(_to_bf16_bits(expert_out))
    expected = np.zeros((tokens, hidden), dtype=np.float32)
    for token in range(tokens):
        for col in range(hidden):
            acc = np.float32(0.0)
            for slot in range(top_k):
                lane = token * top_k + slot
                row = int(l2r[lane])
                if row < 0 or row >= total_lanes:
                    continue
                acc = np.float32(
                    acc
                    + expert_out_bf16[row, col] * np.float32(weights[row])
                )
            expected[token, col] = acc
    expected_bits = _to_bf16_bits(expected)

    results = {}
    for col_tiles in (1, 2, 5, 0):
        # One extra row stays untouched: a tile loop that runs past `hidden`
        # or a token guard that slips would poison it.
        payload = np.concatenate(
            [_to_bf16_bits(expert_out), np.zeros((1, hidden), dtype=np.uint16)], axis=0
        )
        pad_ptr = device.put(payload)
        out_ptr = device.put(np.full((tokens + 1, hidden), 0xBEEF, dtype=np.uint16))
        gemma4_moe_weighted_accumulate_bf16(
            pad_ptr,
            l2r_ptr,
            weights_ptr,
            out_ptr,
            tokens,
            hidden,
            top_k,
            col_tiles=col_tiles,
        )
        got = device.get(out_ptr, (tokens + 1, hidden), np.uint16)
        assert np.all(got[-1] == 0xBEEF), f"col_tiles={col_tiles}: wrote past tokens"
        results[col_tiles] = got[:-1]

    for col_tiles, got in results.items():
        assert np.array_equal(got, expected_bits), (
            f"col_tiles={col_tiles} differs bitwise from the slot-order reference"
        )


@_needs_hip
def test_lane_to_row_is_the_inverse_permutation(device) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_moe_lane_to_row_i32,
    )

    rng = np.random.default_rng(33)
    total = 37
    sorted_lanes = rng.permutation(total).astype(np.int64)
    lanes_ptr = device.put(sorted_lanes)
    l2r_ptr = device.out((total,), np.int32)
    gemma4_moe_lane_to_row_i32(lanes_ptr, l2r_ptr, total)
    got = device.get(l2r_ptr, (total,), np.int32)
    for row, lane in enumerate(sorted_lanes):
        assert got[lane] == row


@_needs_hip
def test_moe_zero_clears_the_buffer(device) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import gemma4_moe_zero_bf16

    buf = device.put(np.full((4, 32), 7, dtype=np.uint16))
    gemma4_moe_zero_bf16(buf, 4 * 32)
    assert np.all(device.get(buf, (4, 32), np.uint16) == 0)


@_needs_hip
def test_gelu_tanh_mul_rejects_empty_shapes(device) -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_gelu_tanh_mul_bf16,
    )

    ptr = device.put(np.zeros((1, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match="must be positive"):
        gemma4_gelu_tanh_mul_bf16(ptr, ptr, 0, 4)


def test_moe_family_registers_under_gemma4_plain() -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_moe import (
        gemma4_gelu_tanh_mul_bf16,
        gemma4_moe_weighted_accumulate_bf16,
        register_gemma4_moe_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_gemma4_moe_kernels(replace=True)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="expert_geglu",
            quant="gguf_q4_k_m",
            variant="gemma4_plain",
        )
        is gemma4_gelu_tanh_mul_bf16
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_weighted_accumulate",
            quant="gguf_q4_k_m",
            variant="gemma4_plain",
        )
        is gemma4_moe_weighted_accumulate_bf16
    )


@_needs_hip
def test_experts_forward_matches_the_reference(device) -> None:
    """The orchestrated expert block reproduces gemma4_experts_forward.

    Exercises the whole chain: group/compact, hidden gather, per-expert gate_up
    GEMV, GeGLU, per-expert down GEMV, and the weighted accumulate. Weights and
    activations are bf16 so the comparison carries bf16 rounding, but a
    structural error (wrong expert, missing lane, gate/up swapped, unweighted
    accumulate) is orders of magnitude larger than that.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        Gemma4ExpertScratch,
        gemma4_experts_forward_bf16,
    )

    tokens, top_k, hidden_size, intermediate, num_experts = 6, 4, 64, 48, 8
    rng = np.random.default_rng(41)

    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)
    # Distinct experts per lane, so every lane's contribution is identifiable.
    selected = np.stack([rng.permutation(num_experts)[:top_k] for _ in range(tokens)]).astype(
        np.int64
    )
    weights = rng.random((tokens, top_k)).astype(np.float32)
    gate_up = (rng.standard_normal((num_experts, 2 * intermediate, hidden_size)) * 0.3).astype(
        np.float32
    )
    down = (rng.standard_normal((num_experts, hidden_size, intermediate)) * 0.3).astype(np.float32)

    hidden_ptr = device.put(_to_bf16_bits(hidden))
    selected_ptr = device.put(selected)
    weights_ptr = device.put(weights)
    gate_up_ptr = device.put(_to_bf16_bits(gate_up))
    down_ptr = device.put(_to_bf16_bits(down))
    out_ptr = device.out((tokens, hidden_size), np.uint16)

    scratch = Gemma4ExpertScratch(
        tokens=tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    try:
        gemma4_experts_forward_bf16(
            hidden_ptr,
            selected_ptr,
            weights_ptr,
            gate_up_ptr,
            down_ptr,
            out_ptr,
            scratch=scratch,
        )
        got = _from_bf16_bits(device.get(out_ptr, (tokens, hidden_size), np.uint16))
    finally:
        scratch.free()

    expected = gemma4_experts_forward(
        hidden,
        selected,
        weights,
        gate_up_proj=gate_up,
        down_proj=down,
    )
    assert np.allclose(got, expected, atol=5e-2, rtol=5e-2), (
        f"max abs diff {np.abs(got - expected).max():.4g}"
    )


@_needs_hip
@pytest.mark.parametrize("scratch_tokens", [12, 40])
def test_experts_forward_runs_a_narrow_block_through_a_wide_scratch(
    device, scratch_tokens
) -> None:
    """A narrower block must produce the same result as a wider scratch.

    The runner sizes the expert scratch once for the widest block it will run
    and then pushes single-token decode steps through it. That makes
    ``scratch.tokens`` a capacity, not the block width, so every kernel must be
    told the *live* lane count rather than the buffer's.

    Passing the capacity instead lets the group-scatter kernels walk lanes that
    were never written, indexing with uninitialized expert ids and offsets. That
    reads outside every expert buffer and the device raises an SQ privilege
    fault it never recovers from: no error is reported and the next
    synchronizing call blocks forever, which is indistinguishable from a hang in
    whatever call happens to be next.

    Sizing the scratch to exactly the block width, as every other expert test
    does, cannot detect this.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        Gemma4ExpertScratch,
        gemma4_experts_forward_bf16,
    )

    tokens, top_k, hidden_size, intermediate, num_experts = 6, 4, 64, 48, 8
    rng = np.random.default_rng(41)

    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)
    selected = np.stack([rng.permutation(num_experts)[:top_k] for _ in range(tokens)]).astype(
        np.int64
    )
    weights = rng.random((tokens, top_k)).astype(np.float32)
    gate_up = (rng.standard_normal((num_experts, 2 * intermediate, hidden_size)) * 0.3).astype(
        np.float32
    )
    down = (rng.standard_normal((num_experts, hidden_size, intermediate)) * 0.3).astype(np.float32)

    hidden_ptr = device.put(_to_bf16_bits(hidden))
    selected_ptr = device.put(selected)
    weights_ptr = device.put(weights)
    gate_up_ptr = device.put(_to_bf16_bits(gate_up))
    down_ptr = device.put(_to_bf16_bits(down))
    out_ptr = device.out((tokens, hidden_size), np.uint16)

    scratch = Gemma4ExpertScratch(
        tokens=scratch_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    try:
        gemma4_experts_forward_bf16(
            hidden_ptr,
            selected_ptr,
            weights_ptr,
            gate_up_ptr,
            down_ptr,
            out_ptr,
            scratch=scratch,
            rows=tokens,
        )
        got = _from_bf16_bits(device.get(out_ptr, (tokens, hidden_size), np.uint16))
    finally:
        scratch.free()

    expected = gemma4_experts_forward(
        hidden,
        selected,
        weights,
        gate_up_proj=gate_up,
        down_proj=down,
    )
    assert np.allclose(got, expected, atol=5e-2, rtol=5e-2), (
        f"scratch_tokens={scratch_tokens} rows={tokens}: "
        f"max abs diff {np.abs(got - expected).max():.4g}"
    )


@_needs_hip
def test_experts_forward_weights_each_lane_by_its_route(device) -> None:
    """Zeroing one lane's route weight must remove exactly that lane's expert.

    This is what distinguishes a weighted accumulate from a plain sum, and what
    catches a compact-order/lane mix-up: the wrong expert would be removed.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        Gemma4ExpertScratch,
        gemma4_experts_forward_bf16,
    )

    tokens, top_k, hidden_size, intermediate, num_experts = 3, 2, 32, 24, 6
    rng = np.random.default_rng(42)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)
    selected = np.array([[1, 4], [0, 3], [2, 5]], dtype=np.int64)
    gate_up = (rng.standard_normal((num_experts, 2 * intermediate, hidden_size)) * 0.3).astype(
        np.float32
    )
    down = (rng.standard_normal((num_experts, hidden_size, intermediate)) * 0.3).astype(np.float32)
    full = np.ones((tokens, top_k), dtype=np.float32)
    masked = full.copy()
    masked[1, 0] = 0.0

    hidden_ptr = device.put(_to_bf16_bits(hidden))
    selected_ptr = device.put(selected)
    gate_up_ptr = device.put(_to_bf16_bits(gate_up))
    down_ptr = device.put(_to_bf16_bits(down))

    def run(weights: np.ndarray) -> np.ndarray:
        weights_ptr = device.put(weights)
        out_ptr = device.out((tokens, hidden_size), np.uint16)
        scratch = Gemma4ExpertScratch(
            tokens=tokens,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate=intermediate,
            num_experts=num_experts,
        )
        try:
            gemma4_experts_forward_bf16(
                hidden_ptr,
                selected_ptr,
                weights_ptr,
                gate_up_ptr,
                down_ptr,
                out_ptr,
                scratch=scratch,
            )
            return _from_bf16_bits(device.get(out_ptr, (tokens, hidden_size), np.uint16))
        finally:
            scratch.free()

    got_full = run(full)
    got_masked = run(masked)
    expected_masked = gemma4_experts_forward(
        hidden, selected, masked, gate_up_proj=gate_up, down_proj=down
    )
    assert np.allclose(got_masked, expected_masked, atol=5e-2, rtol=5e-2)
    # Token 1 lost expert 0 and must have changed; the other tokens must not.
    assert not np.allclose(got_full[1], got_masked[1], atol=1e-3)
    assert np.array_equal(got_full[0], got_masked[0])
    assert np.array_equal(got_full[2], got_masked[2])


@_needs_hip
def test_experts_forward_handles_an_unused_expert(device) -> None:
    """An expert with no lanes must be skipped, not launched with rows=0."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
        Gemma4ExpertScratch,
        gemma4_experts_forward_bf16,
    )

    tokens, top_k, hidden_size, intermediate, num_experts = 4, 2, 32, 16, 16
    rng = np.random.default_rng(43)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)
    # Only experts 0 and 1 are ever selected; 2..15 are empty.
    selected = np.array([[0, 1], [0, 1], [1, 0], [0, 0]], dtype=np.int64)
    weights = rng.random((tokens, top_k)).astype(np.float32)
    gate_up = (rng.standard_normal((num_experts, 2 * intermediate, hidden_size)) * 0.3).astype(
        np.float32
    )
    down = (rng.standard_normal((num_experts, hidden_size, intermediate)) * 0.3).astype(np.float32)

    hidden_ptr = device.put(_to_bf16_bits(hidden))
    selected_ptr = device.put(selected)
    weights_ptr = device.put(weights)
    gate_up_ptr = device.put(_to_bf16_bits(gate_up))
    down_ptr = device.put(_to_bf16_bits(down))
    out_ptr = device.out((tokens, hidden_size), np.uint16)

    scratch = Gemma4ExpertScratch(
        tokens=tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate=intermediate,
        num_experts=num_experts,
    )
    try:
        gemma4_experts_forward_bf16(
            hidden_ptr,
            selected_ptr,
            weights_ptr,
            gate_up_ptr,
            down_ptr,
            out_ptr,
            scratch=scratch,
        )
        got = _from_bf16_bits(device.get(out_ptr, (tokens, hidden_size), np.uint16))
    finally:
        scratch.free()

    expected = gemma4_experts_forward(
        hidden, selected, weights, gate_up_proj=gate_up, down_proj=down
    )
    assert np.allclose(got, expected, atol=5e-2, rtol=5e-2)


def test_expert_scratch_rejects_nonpositive_shapes() -> None:
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import Gemma4ExpertScratch

    with pytest.raises(ValueError, match="top_k must be positive"):
        Gemma4ExpertScratch(tokens=2, top_k=0, hidden_size=8, intermediate=8, num_experts=4)


# --------------------------------------------------------------------------
# Router: prescale -> logits -> top-k -> softmax -> per-expert scale
# --------------------------------------------------------------------------


@_needs_hip
def test_router_topk_matches_the_reference():
    """The assembled router reproduces gemma4_router_topk end to end."""

    from hipengine.kernels.cpu_reference.gemma4 import gemma4_rmsnorm, gemma4_router_topk
    from hipengine.kernels.cpu_reference.ops import linear
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_router import (
        Gemma4RouterScratch,
        gemma4_router_topk_bf16,
    )

    device = _Device()
    tokens, hidden_size, num_experts, top_k = 5, 64, 12, 4
    rng = np.random.default_rng(1701)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)
    scale = (rng.random(hidden_size).astype(np.float32) + 0.5) * 0.1
    proj = (rng.standard_normal((num_experts, hidden_size)) * 0.2).astype(np.float32)
    per_expert = (rng.random(num_experts).astype(np.float32) + 0.5) * 0.3

    hidden_ptr = device.put(_to_bf16_bits(hidden))
    scale_ptr = device.put(scale)
    proj_ptr = device.put(proj)
    per_expert_ptr = device.put(per_expert)
    selected_ptr = device.out((tokens, top_k), np.int64)
    weights_ptr = device.out((tokens, top_k), np.float32)

    scratch = Gemma4RouterScratch(
        tokens=tokens, hidden_size=hidden_size, num_experts=num_experts, top_k=top_k
    )
    try:
        gemma4_router_topk_bf16(
            hidden_ptr,
            scale_ptr,
            proj_ptr,
            per_expert_ptr,
            selected_ptr,
            weights_ptr,
            tokens=tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            scratch=scratch,
        )
        got_selected = device.get(selected_ptr, (tokens, top_k), np.int64)
        got_weights = device.get(weights_ptr, (tokens, top_k), np.float32)
    finally:
        scratch.free()

    _probs, want_weights, want_selected = gemma4_router_topk(
        hidden,
        norm_weight=None,
        scale=scale,
        proj_weight=proj,
        per_expert_scale=per_expert,
        top_k=top_k,
        scalar_root_size=hidden_size**-0.5,
        eps=1e-6,
    )

    # The prescale stage rounds the activation to bf16 before the projection, so
    # the kernel is not computing the f32 reference exactly. Measure that
    # perturbation in logit space, which is where the selection decision lives.
    ref_hidden = _from_bf16_bits(_to_bf16_bits(hidden))
    pre_f32 = (
        gemma4_rmsnorm(ref_hidden, None, 1e-6) * scale * np.float32(hidden_size**-0.5)
    ).astype(np.float32)
    pre_bf16 = _from_bf16_bits(_to_bf16_bits(pre_f32))
    prescale_rel = np.abs(pre_bf16 - pre_f32).max() / np.abs(pre_f32).max()
    assert prescale_rel < 0.01, f"bf16 prescale cost {prescale_rel:.4f} is unexpectedly large"
    logits_f32 = linear(pre_f32, proj)
    perturbation = np.abs(logits_f32 - linear(pre_bf16, proj)).max()

    # Exact selection agreement with an f32 reference is not the right contract
    # for a bf16-prescale path: the rounding can legitimately flip a decision
    # whose logit margin is smaller than the rounding. The contract is therefore
    # "agrees wherever the decision is not inside the perturbation, and every
    # disagreement is inside it".
    ordered_logits = np.sort(logits_f32, axis=-1)
    gap = ordered_logits[:, -top_k] - ordered_logits[:, -(top_k + 1)]
    decisive = gap > perturbation
    assert decisive.any(), "no decisive token; the assertion below would prove nothing"

    got = np.asarray(got_selected)
    want = np.asarray(want_selected).astype(np.int64)
    agrees = (got == want).all(axis=-1)
    np.testing.assert_array_equal(
        agrees[decisive],
        np.ones(int(decisive.sum()), dtype=bool),
        err_msg=(
            "selection differs on a token whose k-th/(k+1)-th logit gap exceeds "
            f"the bf16 prescale perturbation ({perturbation:.5f})"
        ),
    )
    for token in np.flatnonzero(~agrees):
        assert gap[token] <= perturbation, (
            f"token {token} selection differs with logit gap {gap[token]:.5f} "
            f"above the perturbation {perturbation:.5f}"
        )

    # Weights are compared against the f32 reference with a tolerance derived
    # from the measured perturbation rather than a hand-picked constant.
    np.testing.assert_allclose(got_weights, want_weights, atol=4 * perturbation, rtol=0)


@_needs_hip
def test_router_topk_selection_is_ordered_by_descending_logit():
    """Selected experts must be ordered best-first, not by expert index."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_router import (
        Gemma4RouterScratch,
        gemma4_router_topk_bf16,
    )

    device = _Device()
    tokens, hidden_size, num_experts, top_k = 3, 32, 8, 3
    rng = np.random.default_rng(99)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)
    scale = np.full(hidden_size, 0.1, dtype=np.float32)
    proj = (rng.standard_normal((num_experts, hidden_size)) * 0.4).astype(np.float32)
    per_expert = np.ones(num_experts, dtype=np.float32)

    hidden_ptr = device.put(_to_bf16_bits(hidden))
    scale_ptr = device.put(scale)
    proj_ptr = device.put(proj)
    per_expert_ptr = device.put(per_expert)
    selected_ptr = device.out((tokens, top_k), np.int64)
    weights_ptr = device.out((tokens, top_k), np.float32)

    scratch = Gemma4RouterScratch(
        tokens=tokens, hidden_size=hidden_size, num_experts=num_experts, top_k=top_k
    )
    try:
        gemma4_router_topk_bf16(
            hidden_ptr,
            scale_ptr,
            proj_ptr,
            per_expert_ptr,
            selected_ptr,
            weights_ptr,
            tokens=tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            scratch=scratch,
        )
        got_weights = device.get(weights_ptr, (tokens, top_k), np.float32)
    finally:
        scratch.free()

    # Renormalised over the selected set with unit per-expert scale, so the
    # weights are descending and sum to one for every token. A by-index sort
    # or an unnormalised softmax breaks both.
    assert np.all(np.diff(got_weights, axis=-1) <= 1e-6), got_weights
    np.testing.assert_allclose(got_weights.sum(axis=-1), np.ones(tokens), atol=1e-5)


def test_router_scratch_is_a_capacity_not_an_identity():
    """A scratch sized for the widest block serves narrower blocks too.

    Generation sizes one scratch for the prefill width and then routes
    single-token decode steps through it. Rejecting a narrower block would make
    that impossible, so `tokens` is a capacity bound rather than an identity.
    The shape parameters still describe the weights and must match exactly, and
    a block wider than the capacity is still an error rather than an overflow.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_router import (
        Gemma4RouterScratch,
        gemma4_router_topk_bf16,
    )

    scratch = Gemma4RouterScratch(tokens=8, hidden_size=32, num_experts=8, top_k=2)
    try:
        with pytest.raises(ValueError, match="exceeds scratch capacity"):
            gemma4_router_topk_bf16(
                0,
                0,
                0,
                0,
                0,
                0,
                tokens=9,
                hidden_size=32,
                num_experts=8,
                top_k=2,
                scratch=scratch,
            )
        # A shape disagreement is still rejected: these parameters describe the
        # weights, not the block, so they cannot be satisfied by spare capacity.
        with pytest.raises(ValueError, match="scratch was built for"):
            gemma4_router_topk_bf16(
                0,
                0,
                0,
                0,
                0,
                0,
                tokens=2,
                hidden_size=64,
                num_experts=8,
                top_k=2,
                scratch=scratch,
            )
    finally:
        scratch.free()


# --------------------------------------------------------------------------
# Prefill attention: ungated, caller-supplied keep mask, scale 1.0
# --------------------------------------------------------------------------


def _reference_attention(query, key, value, keep_mask, *, num_heads, num_kv_heads):
    """GQA attention in f32, mirroring the reference's _grouped_attention."""

    tokens, _, head_dim = query.shape
    groups = num_heads // num_kv_heads
    context = np.zeros_like(query, dtype=np.float32)
    for token in range(tokens):
        for head in range(num_heads):
            kv_head = head // groups
            logits = query[token, head] @ key[:, kv_head, :].T
            keep = keep_mask[token].astype(bool)
            logits = np.where(keep, logits, -np.inf)
            top = logits.max()
            weights = np.exp(logits - top)
            weights = weights / weights.sum()
            context[token, head] = weights @ value[:, kv_head, :]
    return context


def _attn_reduction_tree(values, width):
    """The prefill kernel's reduction: ``p[i] += p[i + stride]``, halving.

    Written as explicit slices so numpy cannot reassociate the adds: each step
    is one fp32 elementwise addition of the same pairs the kernel sums.
    """

    partial = np.asarray(values, dtype=np.float32)
    stride = width // 2
    while stride >= 1:
        partial = (partial[:stride] + partial[stride : 2 * stride]).astype(np.float32)
        stride //= 2
    return float(partial[0])


def _prefill_attention_association(query, key, value, keep_mask, *, num_heads, num_kv_heads,
                                   head_dim, threads, scale=1.0):
    """Masked GQA attention in the prefill kernel's own association order.

    The kernel is one block of ``threads`` threads per (token, head):

    * logit[j] is ``tree_sum_d(q[d] * k[j][d]) * scale``, a fixed pairwise tree
      over ``threads`` partials, where thread *t* holds the partial over
      ``d = t, t + threads, ...``;
    * the maximum is exact, so any order gives the same value;
    * the denominator is a ``threads``-wide tree over per-thread partials
      ``j = t, t + threads, ...`` of ``exp(logit[j] - max)``;
    * the numerator is a *serial* sum over keys in increasing order, one thread
      per output dimension.

    ``head_dim == threads`` is the case Gemma 4 uses, where each thread's
    partial is a single product.
    """

    tokens, _, _ = query.shape
    groups = num_heads // num_kv_heads
    context = np.zeros_like(query, dtype=np.float32)
    for token in range(tokens):
        for head in range(num_heads):
            kv_head = head // groups
            keep = keep_mask[token].astype(bool)
            logits = np.full((tokens,), -np.inf, dtype=np.float32)
            for j in range(tokens):
                if not keep[j]:
                    continue
                products = np.zeros((threads,), dtype=np.float32)
                for d in range(head_dim):
                    partial = np.float32(
                        np.float32(query[token, head, d]) * np.float32(key[j, kv_head, d])
                    )
                    products[d % threads] = np.float32(
                        products[d % threads] + partial
                    )
                logits[j] = np.float32(_attn_reduction_tree(products, threads) * scale)
            top = logits.max()
            weights = np.exp(logits - top).astype(np.float32)
            denominator = _attn_reduction_tree(
                np.array(
                    [
                        np.sum(weights[t::threads].astype(np.float32), dtype=np.float32)
                        for t in range(threads)
                    ],
                    dtype=np.float32,
                ),
                threads,
            )
            for d in range(head_dim):
                acc = np.float32(0.0)
                for j in range(tokens):
                    if weights[j] == 0.0:
                        continue
                    acc = np.float32(acc + np.float32(weights[j] * value[j, kv_head, d]))
                context[token, head, d] = np.float32(acc / np.float32(denominator))
    return context


_PREFILL_HEAD_DIM_256_CASE = dict(tokens=12, num_heads=4, num_kv_heads=2, head_dim=256)


def _prefill_head_dim_256_inputs():
    """The recorded case: causal, with one masked-out recent key per row."""

    case = _PREFILL_HEAD_DIM_256_CASE
    rng = np.random.default_rng(913_204)
    shape_q = (case["tokens"], case["num_heads"], case["head_dim"])
    shape_kv = (case["tokens"], case["num_kv_heads"], case["head_dim"])
    query = (rng.standard_normal(shape_q) * 0.5).astype(np.float32)
    key = (rng.standard_normal(shape_kv) * 0.5).astype(np.float32)
    value = (rng.standard_normal(shape_kv) * 0.5).astype(np.float32)
    keep = np.tril(np.ones((case["tokens"], case["tokens"]), dtype=np.uint8))
    for row in range(2, case["tokens"], 3):
        keep[row, row] = 0
    return query, key, value, keep


def _run_prefill_head_dim_256(dtype):
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_bf16,
        gemma4_attention_prefill_f32,
    )

    case = _PREFILL_HEAD_DIM_256_CASE
    query, key, value, keep = _prefill_head_dim_256_inputs()
    if dtype == "bf16":
        query, key, value = _to_bf16_bits(query), _to_bf16_bits(key), _to_bf16_bits(value)
        storage, launch = np.uint16, gemma4_attention_prefill_bf16
    else:
        storage, launch = np.float32, gemma4_attention_prefill_f32

    device = _Device()
    try:
        out_ptr = device.out(
            (case["tokens"], case["num_heads"], case["head_dim"]), storage
        )
        launch(
            device.put(query),
            device.put(key),
            device.put(value),
            device.put(keep),
            out_ptr,
            tokens=case["tokens"],
            num_heads=case["num_heads"],
            num_kv_heads=case["num_kv_heads"],
            head_dim=case["head_dim"],
            scale=1.0,
        )
        return device.get(
            out_ptr, (case["tokens"], case["num_heads"], case["head_dim"]), storage
        )
    finally:
        device.close()


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@_needs_hip
def test_attention_prefill_head_dim_256_is_bit_identical_to_recorded_outputs(dtype):
    """head_dim=256 prefill keeps every output bit it had before the rewrite.

    The reduction for this shape was rewritten to remove a block-wide barrier
    tree per key. That rewrite is only allowed to change *how* the tree is
    computed, never which numbers it sums or in what order, so the outputs are
    pinned bit-for-bit against tensors captured from the kernel that already
    implemented this association. A numpy reference cannot stand in for this:
    the device ``expf`` differs from numpy's ``exp`` in the last bit.
    """

    expected = np.load(
        Path(__file__).parent
        / "fixtures"
        / "gemma4"
        / f"attention_prefill_head_dim_256_{dtype}.npy"
    )
    np.testing.assert_array_equal(_run_prefill_head_dim_256(dtype), expected)


@pytest.mark.parametrize("dtype", ["f32", "bf16"])
@_needs_hip
def test_attention_prefill_head_dim_256_follows_its_reduction_association(dtype):
    """The head_dim=256 prefill path sums the pairs its association names.

    This is the readable form of the same contract: the logits are a fixed
    pairwise tree over one product per thread, the maximum is exact, the
    denominator is a tree over per-thread key partials, and the numerator is a
    serial sum over keys in increasing order. The tolerance is one fp32 ulp of
    the device's ``expf``, so this catches a wrong structure rather than a
    different last bit.
    """

    case = _PREFILL_HEAD_DIM_256_CASE
    query, key, value, keep = _prefill_head_dim_256_inputs()
    got = _run_prefill_head_dim_256(dtype)
    if dtype == "bf16":
        # The kernel reads bf16 inputs, so the reference has to see the same
        # rounded values, not the f32 originals.
        query, key, value = (
            _from_bf16_bits(_to_bf16_bits(query)),
            _from_bf16_bits(_to_bf16_bits(key)),
            _from_bf16_bits(_to_bf16_bits(value)),
        )
        got = _from_bf16_bits(got)
    want = _prefill_attention_association(
        query,
        key,
        value,
        keep,
        num_heads=case["num_heads"],
        num_kv_heads=case["num_kv_heads"],
        head_dim=case["head_dim"],
        threads=case["head_dim"],
    )
    if dtype == "bf16":
        # The kernel stores bf16, so the reference rounds the same way before
        # the comparison; the tolerance below is one fp32 ulp of expf.
        want = _from_bf16_bits(_to_bf16_bits(want))
    np.testing.assert_allclose(got, want, atol=5e-7, rtol=0.0)


@_needs_hip
def test_attention_prefill_f32_matches_the_reference():
    """The f32 entry point reproduces masked GQA attention exactly."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )

    device = _Device()
    tokens, num_heads, num_kv_heads, head_dim = 6, 4, 2, 16
    rng = np.random.default_rng(2024)
    query = (rng.standard_normal((tokens, num_heads, head_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    keep = np.tril(np.ones((tokens, tokens), dtype=np.uint8))

    q_ptr = device.put(query)
    k_ptr = device.put(key)
    v_ptr = device.put(value)
    m_ptr = device.put(keep)
    out_ptr = device.out((tokens, num_heads, head_dim), np.float32)

    gemma4_attention_prefill_f32(
        q_ptr,
        k_ptr,
        v_ptr,
        m_ptr,
        out_ptr,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
    )
    got = device.get(out_ptr, (tokens, num_heads, head_dim), np.float32)
    want = _reference_attention(
        query, key, value, keep, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@_needs_hip
def test_attention_prefill_applies_the_mask_it_is_given():
    """A masked-out key must contribute nothing, including when it is recent.

    The kernel must apply the caller's keep mask rather than re-deriving
    causality: a sliding-window layer keeps only the last `window` keys, which
    the kernel cannot know on its own.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )

    device = _Device()
    tokens, num_heads, num_kv_heads, head_dim = 5, 2, 2, 8
    rng = np.random.default_rng(77)
    query = (rng.standard_normal((tokens, num_heads, head_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)

    # Causal AND sliding window of 2: keeps only the previous key, not the whole
    # prefix. A causal-only kernel would attend over keys 0..token.
    window = 2
    positions = np.arange(tokens)
    keep = (
        (positions[None, :] <= positions[:, None])
        & (positions[None, :] > positions[:, None] - window)
    ).astype(np.uint8)
    assert keep.sum() == tokens + (tokens - 1), keep

    q_ptr = device.put(query)
    k_ptr = device.put(key)
    v_ptr = device.put(value)
    m_ptr = device.put(keep)
    out_ptr = device.out((tokens, num_heads, head_dim), np.float32)

    gemma4_attention_prefill_f32(
        q_ptr,
        k_ptr,
        v_ptr,
        m_ptr,
        out_ptr,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
    )
    got = device.get(out_ptr, (tokens, num_heads, head_dim), np.float32)
    want = _reference_attention(
        query, key, value, keep, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)

    # Token 0 attends only to itself, so its context must be exactly V[0].
    for head in range(num_heads):
        kv_head = head // (num_heads // num_kv_heads)
        np.testing.assert_allclose(got[0, head], value[0, kv_head], atol=1e-5)


@_needs_hip
@pytest.mark.parametrize("start,rows,window", [(0, 4, 2), (0, 6, 3), (10, 1, 4), (0, 5, None)])
def test_attention_prefill_matches_the_reference_through_the_production_mask(start, rows, window):
    """Production keep-mask -> HIP kernel -> naive reference, end to end.

    V2's hole was a composition gap. Two bindings existed independently: the
    kernel was checked against a naive reference using a mask the *test* built,
    and nothing checked the mask production builds (``_keep_mask``) against the
    HuggingFace-gated ``gemma4_attention_mask``. Each could pass while the real
    path was wrong, because neither exercised the pair the runtime actually
    issues -- this closes that seam by feeding the production mask straight
    through the kernel and comparing to numpy.

    ``start > 0`` matters: that is the decode/paged shape where the key range
    runs ahead of the query rows, and it is the geometry the causal-only tests
    never construct.
    """
    from hipengine.kernels.cpu_reference.gemma4 import (
        FULL_ATTENTION,
        SLIDING_ATTENTION,
        Gemma4AttentionGeometry,
        Gemma4RopeConfig,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )
    from hipengine.runtime.gemma4 import _keep_mask

    geometry = Gemma4AttentionGeometry(
        layer_type=SLIDING_ATTENTION if window is not None else FULL_ATTENTION,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        rope=Gemma4RopeConfig(rope_theta=10000.0, head_dim=16, rope_angles=8, rope_type=1),
        sliding_window=window,
        k_eq_v=False,
    )

    num_heads, num_kv_heads, head_dim = 4, 2, 16
    keys = start + rows
    # This raw-kernel composition uses a full cached range (the graph's
    # frozen-origin ABI), so request its origin explicitly.
    keep = _keep_mask(geometry, start, rows, key_begin=0)
    assert keep.shape == (rows, keys)

    rng = np.random.default_rng(20260929)
    query = (rng.standard_normal((rows, num_heads, head_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.5).astype(np.float32)

    device = _Device()
    q_ptr, k_ptr, v_ptr = device.put(query), device.put(key), device.put(value)
    m_ptr = device.put(keep)
    out_ptr = device.out((rows, num_heads, head_dim), np.float32)

    gemma4_attention_prefill_f32(
        q_ptr, k_ptr, v_ptr, m_ptr, out_ptr,
        tokens=rows,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
        keys=keys,
    )
    got = device.get(out_ptr, (rows, num_heads, head_dim), np.float32)
    want = _reference_attention(
        query, key, value, keep, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    np.testing.assert_allclose(
        got, want, atol=1e-5, rtol=1e-5,
        err_msg=f"production-mask path disagreed at start={start} rows={rows} window={window}",
    )


@_needs_hip
@pytest.mark.parametrize(
    "head_dim,num_kv_heads,window,layer_kind",
    [
        (256, 8, 1024, "sliding"),  # the 25 sliding layers: the window binds
        (512, 2, None, "global"),   # the 5 global layers: head_dim 512, causal only
    ],
)
def test_windowed_attention_matches_the_reference_at_the_gate_shape(
    head_dim, num_kv_heads, window, layer_kind
):
    """The windowed path is right at the shape V1 gates, not just in fixtures.

    V2's open half. Everything else about the window is already pinned: the
    mask builder is bound to the HuggingFace-gated reference, and
    ``test_unit_gemma4_attention_flash_admission.py`` proves the routing policy
    refuses the flash path the moment the window binds (``keys`` 1024 admits,
    1025 does not), so a sliding layer past its window always reaches the exact
    kernel that reads ``keep_mask`` -- while the flash kernel reads no mask at
    all. What no test did was run the pair the runtime actually issues at the
    gate's own geometry. The composition test above stops at head_dim 16 with
    six rows and a four-token window; a bug that only appears once the key walk
    spans thousands of tiles -- the pass-1 skip over fully-masked tiles, the
    window binding on every row past the first 1024 -- would sail past it.

    ``start=3584, rows=512`` is the last block of a 4096-token prefill: keys
    4096 against a 1024 window, so the window binds on the sliding layers. The
    global class runs the same block with ``sliding_window=None``, which is the
    check that a window is *not* applied where none is declared. Both are the
    configuration the gate measures and the one V3 found the kernel skipping
    tiles for.
    """
    from hipengine.kernels.cpu_reference.gemma4 import (
        FULL_ATTENTION,
        SLIDING_ATTENTION,
        Gemma4AttentionGeometry,
        Gemma4RopeConfig,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )
    from hipengine.runtime.gemma4 import _keep_mask

    num_heads = 16
    rows, start = 512, 3584
    keys = start + rows
    assert keys == 4096
    if window is not None:
        assert keys > window, "the window must bind or this shape proves nothing"

    geometry = Gemma4AttentionGeometry(
        layer_type=SLIDING_ATTENTION if window is not None else FULL_ATTENTION,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope=Gemma4RopeConfig(
            rope_theta=10000.0, head_dim=head_dim, rope_angles=head_dim // 2, rope_type=1
        ),
        sliding_window=window,
        k_eq_v=False,
    )

    # Exercise the frozen full-range mask; eager shortened ranges are covered
    # by the runner and independent mask-origin tests.
    keep = _keep_mask(geometry, start, rows, key_begin=0)
    assert keep.shape == (rows, keys)
    # Every query sees its own key (column start+i, not i -- the mask spans the
    # whole cached range), so no row is masked away entirely.
    assert keep[np.arange(rows), start + np.arange(rows)].all()
    assert keep.any(axis=1).all()
    if window is not None:
        # Row 0's query sits at ``start``=3584, so only keys in
        # (3584-1024, 3584] are visible -- 1024 of 4096 columns.
        assert keep[0, : start - window + 1].sum() == 0      # older than the window
        assert keep[0, start - window + 1: start + 1].all()  # the window itself
        assert keep[0, start + 1:].sum() == 0                # causal: no future keys
        assert keep[0].sum() == window
        # The window binds on every row, not just the first.
        assert keep.sum(axis=1).max() <= window
    else:
        # A global layer must be pure causality: no window may leak in.
        assert keep.sum(axis=1).max() == keys

    rng = np.random.default_rng(20260929)
    query = (rng.standard_normal((rows, num_heads, head_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((keys, num_kv_heads, head_dim)) * 0.5).astype(np.float32)

    device = _Device()
    try:
        q_ptr, k_ptr, v_ptr = device.put(query), device.put(key), device.put(value)
        m_ptr = device.put(keep)
        out_ptr = device.out((rows, num_heads, head_dim), np.float32)
        gemma4_attention_prefill_f32(
            q_ptr, k_ptr, v_ptr, m_ptr, out_ptr,
            tokens=rows,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=1.0,
            keys=keys,
        )
        got = device.get(out_ptr, (rows, num_heads, head_dim), np.float32)
    finally:
        device.close()

    want = _reference_attention(
        query, key, value, keep, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    np.testing.assert_allclose(
        got, want, atol=1e-5, rtol=1e-5,
        err_msg="windowed prefill disagreed with the reference at the gate shape "
                f"({layer_kind} head_dim={head_dim} start=3584 rows=512 keys=4096 "
                f"window={window})",
    )


def test_attention_prefill_scale_is_not_assumed():
    """The kernel multiplies by the scale it is given, not head_dim**-0.5.

    Gemma 4 folds the softmax scaling into the query norm weight and passes 1.0.
    A kernel that hard-coded the reciprocal square root would pass every test
    that also used 1.0 only by coincidence of geometry, so exercise a scale that
    is deliberately not head_dim**-0.5 and not 1.0.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )

    device = _Device()
    tokens, num_heads, num_kv_heads, head_dim = 4, 2, 2, 32
    rng = np.random.default_rng(5)
    query = (rng.standard_normal((tokens, num_heads, head_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    keep = np.tril(np.ones((tokens, tokens), dtype=np.uint8))

    # head_dim**-0.5 would be 0.17677669; use something unrelated to it.
    scale = 0.37
    assert abs(scale - head_dim**-0.5) > 0.1

    q_ptr = device.put(query)
    k_ptr = device.put(key)
    v_ptr = device.put(value)
    m_ptr = device.put(keep)
    out_ptr = device.out((tokens, num_heads, head_dim), np.float32)

    gemma4_attention_prefill_f32(
        q_ptr,
        k_ptr,
        v_ptr,
        m_ptr,
        out_ptr,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
    )
    got = device.get(out_ptr, (tokens, num_heads, head_dim), np.float32)

    scaled_query = query * scale
    want = _reference_attention(
        scaled_query, key, value, keep, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@_needs_hip
def test_attention_prefill_bf16_tracks_the_f32_path():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_bf16,
        gemma4_attention_prefill_f32,
    )

    device = _Device()
    tokens, num_heads, num_kv_heads, head_dim = 8, 4, 1, 16
    rng = np.random.default_rng(31337)
    query = (rng.standard_normal((tokens, num_heads, head_dim)) * 0.5).astype(np.float32)
    key = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value = (rng.standard_normal((tokens, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    keep = np.tril(np.ones((tokens, tokens), dtype=np.uint8))

    # bf16 round the inputs so both paths see identical values, then the only
    # difference is the accumulation and output precision.
    qb = _from_bf16_bits(_to_bf16_bits(query))
    kb = _from_bf16_bits(_to_bf16_bits(key))
    vb = _from_bf16_bits(_to_bf16_bits(value))
    m_ptr = device.put(keep)
    f32_out = device.out((tokens, num_heads, head_dim), np.float32)
    gemma4_attention_prefill_f32(
        device.put(qb),
        device.put(kb),
        device.put(vb),
        m_ptr,
        f32_out,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
    )
    want = device.get(f32_out, (tokens, num_heads, head_dim), np.float32)

    bf16_out = device.out((tokens, num_heads, head_dim), np.uint16)
    gemma4_attention_prefill_bf16(
        device.put(_to_bf16_bits(query)),
        device.put(_to_bf16_bits(key)),
        device.put(_to_bf16_bits(value)),
        m_ptr,
        bf16_out,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
    )
    got = _from_bf16_bits(device.get(bf16_out, (tokens, num_heads, head_dim), np.uint16))

    diff = np.abs(got - want)
    scale_ref = np.abs(want).max()
    assert diff.max() / scale_ref < 0.02, (
        f"bf16 attention differs from f32 by {diff.max():.5g} against scale {scale_ref:.4g}"
    )


# --------------------------------------------------------------------------
# Decoder layer forward against gemma4_decoder_layer_forward
# --------------------------------------------------------------------------


def _layer_fixture(
    seed,
    tokens,
    hidden_size,
    geometry,
    dense_intermediate,
    num_experts,
    top_k,
    expert_intermediate,
    k_eq_v,
):
    from hipengine.kernels.cpu_reference.gemma4 import (
        Gemma4AttentionGeometry,
        Gemma4LayerWeights,
        Gemma4RopeConfig,
        Gemma4TextConfig,
    )

    rng = np.random.default_rng(seed)
    rope = Gemma4RopeConfig(
        rope_theta=10000.0,
        head_dim=geometry[2],
        rope_angles=geometry[2] // 2,
        rope_type="default",
    )
    attn_geometry = Gemma4AttentionGeometry(
        layer_type="sliding_attention" if k_eq_v else "full_attention",
        num_heads=geometry[0],
        num_kv_heads=geometry[1],
        head_dim=geometry[2],
        rope=rope,
        sliding_window=None,
        k_eq_v=k_eq_v,
    )
    config = Gemma4TextConfig(
        hidden_size=hidden_size,
        intermediate_size=dense_intermediate,
        moe_intermediate_size=expert_intermediate,
        num_experts=num_experts,
        top_k_experts=top_k,
        rms_norm_eps=1e-6,
        attention=(attn_geometry,),
    )

    def norm():
        return (rng.random(hidden_size).astype(np.float32) + 0.5).astype(np.float32)

    weights = Gemma4LayerWeights(
        input_layernorm=norm(),
        post_attention_layernorm=norm(),
        pre_feedforward_layernorm=norm(),
        post_feedforward_layernorm=norm(),
        post_feedforward_layernorm_1=norm(),
        post_feedforward_layernorm_2=norm(),
        pre_feedforward_layernorm_2=norm(),
        q_proj=(rng.standard_normal((geometry[0] * geometry[2], hidden_size)) * 0.2).astype(
            np.float32
        ),
        k_proj=(rng.standard_normal((geometry[1] * geometry[2], hidden_size)) * 0.2).astype(
            np.float32
        ),
        o_proj=(rng.standard_normal((hidden_size, geometry[0] * geometry[2])) * 0.2).astype(
            np.float32
        ),
        q_norm=(rng.random(geometry[2]).astype(np.float32) + 0.5).astype(np.float32),
        k_norm=(rng.random(geometry[2]).astype(np.float32) + 0.5).astype(np.float32),
        mlp_gate_proj=(rng.standard_normal((dense_intermediate, hidden_size)) * 0.2).astype(
            np.float32
        ),
        mlp_up_proj=(rng.standard_normal((dense_intermediate, hidden_size)) * 0.2).astype(
            np.float32
        ),
        mlp_down_proj=(rng.standard_normal((hidden_size, dense_intermediate)) * 0.2).astype(
            np.float32
        ),
        router_scale=(rng.random(hidden_size).astype(np.float32) + 0.5) * 0.1,
        router_proj=(rng.standard_normal((num_experts, hidden_size)) * 0.2).astype(np.float32),
        router_per_expert_scale=(rng.random(num_experts).astype(np.float32) + 0.5) * 0.3,
        experts_gate_up_proj=(
            rng.standard_normal((num_experts, 2 * expert_intermediate, hidden_size)) * 0.2
        ).astype(np.float32),
        experts_down_proj=(
            rng.standard_normal((num_experts, hidden_size, expert_intermediate)) * 0.2
        ).astype(np.float32),
        layer_scalar=np.float32(0.0703125),
        v_proj=None
        if k_eq_v
        else (rng.standard_normal((geometry[1] * geometry[2], hidden_size)) * 0.2).astype(
            np.float32
        ),
    )
    return weights, config, attn_geometry


def _run_layer_on_gpu(weights, config, attn_geometry, hidden, tokens, *, k_eq_v, seed):
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        Gemma4LayerGeometry,
        Gemma4LayerPointers,
        Gemma4LayerScratch,
        gemma4_layer_forward_bf16,
    )

    device = _Device()
    geometry = Gemma4LayerGeometry(
        num_heads=attn_geometry.num_heads,
        num_kv_heads=attn_geometry.num_kv_heads,
        head_dim=attn_geometry.head_dim,
        scale=attn_geometry.scale,
        k_eq_v=k_eq_v,
    )
    positions = np.arange(tokens, dtype=np.int64)
    # The kernel indexes a DOUBLED (tokens, head_dim) table; the reference
    # builds its own half-width tables internally.
    cos, sin = gemma4_rope_cos_sin_tables(attn_geometry.rope, positions)
    keep = np.tril(np.ones((tokens, tokens), dtype=np.uint8))

    def put_bf16(arr):
        return device.put(_to_bf16_bits(np.asarray(arr, dtype=np.float32)))

    hidden_ptr = put_bf16(hidden)
    cos_ptr = device.put(cos.astype(np.float32))
    sin_ptr = device.put(sin.astype(np.float32))
    mask_ptr = device.put(keep)

    pointers = Gemma4LayerPointers(
        input_layernorm=device.put(weights.input_layernorm),
        q_proj=put_bf16(weights.q_proj),
        k_proj=put_bf16(weights.k_proj),
        o_proj=put_bf16(weights.o_proj),
        q_norm=device.put(weights.q_norm),
        k_norm=device.put(weights.k_norm),
        post_attention_layernorm=device.put(weights.post_attention_layernorm),
        pre_feedforward_layernorm=device.put(weights.pre_feedforward_layernorm),
        # The resident weight carries both halves in one buffer, laid out
        # (rows, 2 * intermediate) exactly as the fused loader emits it.
        mlp_gate_up_proj=put_bf16(
            np.concatenate([weights.mlp_gate_proj, weights.mlp_up_proj], axis=0)
        ),
        mlp_down_proj=put_bf16(weights.mlp_down_proj),
        post_feedforward_layernorm_1=device.put(weights.post_feedforward_layernorm_1),
        router_scale=device.put(weights.router_scale),
        router_proj=device.put(weights.router_proj),
        router_per_expert_scale=device.put(weights.router_per_expert_scale),
        pre_feedforward_layernorm_2=device.put(weights.pre_feedforward_layernorm_2),
        experts_gate_up_proj=put_bf16(weights.experts_gate_up_proj),
        experts_down_proj=put_bf16(weights.experts_down_proj),
        post_feedforward_layernorm_2=device.put(weights.post_feedforward_layernorm_2),
        post_feedforward_layernorm=device.put(weights.post_feedforward_layernorm),
        v_proj=0 if k_eq_v else put_bf16(weights.v_proj),
        layer_scalar=device.put(np.array([weights.layer_scalar], dtype=np.float32)),
    )
    scratch = Gemma4LayerScratch(
        tokens=tokens,
        hidden_size=config.hidden_size,
        dense_intermediate=config.intermediate_size,
        geometry=geometry,
        num_experts=config.num_experts,
        top_k=config.top_k_experts,
        expert_intermediate=config.moe_intermediate_size,
    )
    try:
        gemma4_layer_forward_bf16(
            hidden_ptr,
            cos_ptr,
            sin_ptr,
            mask_ptr,
            pointers,
            scratch=scratch,
            eps=config.rms_norm_eps,
        )
        got = _from_bf16_bits(device.get(hidden_ptr, (tokens, config.hidden_size), np.uint16))
    finally:
        scratch.free()
    return got


def _run_layer_reference(weights, config, attn_geometry, hidden, tokens):
    from hipengine.kernels.cpu_reference.gemma4 import gemma4_decoder_layer_forward

    return gemma4_decoder_layer_forward(
        hidden, weights, attn_geometry, config, positions=np.arange(tokens, dtype=np.int64)
    )


@pytest.mark.parametrize("k_eq_v", [False, True])
@_needs_hip
def test_layer_forward_matches_the_reference(k_eq_v):
    """The whole decoder layer reproduces gemma4_decoder_layer_forward."""

    tokens, hidden_size = 5, 64
    geometry = (4, 2, 16)
    dense_intermediate, num_experts, top_k, expert_intermediate = 48, 8, 3, 32
    weights, config, attn_geometry = _layer_fixture(
        4242,
        tokens,
        hidden_size,
        geometry,
        dense_intermediate,
        num_experts,
        top_k,
        expert_intermediate,
        k_eq_v,
    )
    rng = np.random.default_rng(11)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)

    got = _run_layer_on_gpu(weights, config, attn_geometry, hidden, tokens, k_eq_v=k_eq_v, seed=11)
    want = _run_layer_reference(weights, config, attn_geometry, hidden, tokens)

    # Measured: 1.2% (k_eq_v=False) and 1.4% (k_eq_v=True) max relative, ~0.2%
    # mean. That is bf16 rounding through roughly ten chained stages that each
    # round to bf16 in between. A structural error - a dropped layer scalar, the
    # two FFN branches stacked instead of parallel, V taken after the k_norm -
    # lands 10-100x larger, so 3% is a real regression guard with ~2x headroom
    # rather than a tolerance that absorbs those mistakes.
    diff = np.abs(got - want)
    scale_ref = np.abs(want).max()
    assert diff.max() / scale_ref < 0.03, (
        f"layer output differs by {diff.max():.5g} against scale {scale_ref:.4g} "
        f"(relative {diff.max() / scale_ref:.4f}); k_eq_v={k_eq_v}"
    )


def test_layer_forward_routes_head_dim_512_to_the_tiled_prefill_kernel():
    """The global layers must actually execute the tiled kernel, not the fallback.

    A fast kernel that exists but is never selected buys nothing, and finite
    output alone cannot tell you which kernel ran. So this asserts the selected
    route by name through the production entry point and checks the result
    against the two things that make "it ran the fast path" meaningful: the
    output the established block kernel produces for the same inputs, and the
    CPU reference.

    The reference half runs with every expert selected. The router is compared
    in bf16 on the GPU against f32 on the CPU, so a row whose third- and
    fourth-ranked logits tie inside bf16 rounding selects a different expert and
    differs by a whole expert's output -- measured on row 19 of this fixture.
    That is a property of the router's precision, not of attention, so routing
    is made irrelevant here rather than absorbed into a looser tolerance.
    """

    tokens, hidden_size = 128, 256
    geometry = (16, 2, 512)  # Gemma 4's global-layer shape: 16 query / 2 kv heads
    dense_intermediate, num_experts, expert_intermediate = 48, 8, 32
    rng = np.random.default_rng(11)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        _select_prefill_route,
        last_prefill_attention_route,
    )

    # (a) The production entry point selects the tiled kernel for this shape.
    weights, config, attn_geometry = _layer_fixture(
        4242, tokens, hidden_size, geometry,
        dense_intermediate, num_experts, 3, expert_intermediate,
        k_eq_v=False,
    )
    _run_layer_on_gpu(weights, config, attn_geometry, hidden, tokens, k_eq_v=False, seed=11)
    assert last_prefill_attention_route() == "tiled", (
        "head_dim 512 with a multiple-of-128 key count must select the tiled "
        f"kernel, got {last_prefill_attention_route()!r}"
    )

    # (b) With routing neutralized, the tiled-routed layer matches the CPU
    #     reference -- so the tiled attention sits correctly in the pipeline.
    weights, config, attn_geometry = _layer_fixture(
        4242, tokens, hidden_size, geometry,
        dense_intermediate, num_experts, num_experts, expert_intermediate,
        k_eq_v=False,
    )
    got = _run_layer_on_gpu(
        weights, config, attn_geometry, hidden, tokens, k_eq_v=False, seed=11
    )
    assert last_prefill_attention_route() == "tiled"
    want = _run_layer_reference(weights, config, attn_geometry, hidden, tokens)
    diff = np.abs(got - want)
    scale_ref = np.abs(want).max()
    # Measured on this geometry: 4.6% for the tiled route and 4.7% for the
    # fallback -- the two agree to 0.5%, so this is bf16 rounding accumulated
    # over ~ten chained stages at head_dim 512, not an attention error. The
    # reference test documents the same effect at 1.2-1.4% for its much smaller
    # shape. A structural mistake (a dropped stage, a wrong mask) lands an order
    # of magnitude higher, so 8% still separates the two.
    assert diff.max() / scale_ref < 0.08, (
        f"tiled-routed layer output differs by {diff.max():.5g} against scale "
        f"{scale_ref:.4g} (relative {diff.max() / scale_ref:.4f})"
    )

    # (c) The tiled kernel is a drop-in for the established block kernel: forcing
    #     the fallback route over the same inputs must reproduce it. This is the
    #     claim the parity suite makes at the attention level, restated through
    #     the path production actually takes.
    routed = {}
    original = _select_prefill_route

    def forced(choice):
        def pick(**_kwargs):
            return choice

        return pick

    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer as layer_module

    for chosen in ("tiled", "exact"):
        layer_module._select_prefill_route = forced(chosen)
        try:
            routed[chosen] = _run_layer_on_gpu(
                weights, config, attn_geometry, hidden, tokens, k_eq_v=False, seed=11
            )
            assert last_prefill_attention_route() == chosen
        finally:
            layer_module._select_prefill_route = original
    gap = np.abs(routed["tiled"] - routed["exact"]).max()
    # Measured 0.0029 absolute / 0.54% relative: the two formulations round
    # differently but agree to bf16 precision. A wrong mask or a dropped term in
    # the tiled kernel would separate them by an order of magnitude more.
    assert gap < 0.02, f"tiled and exact routes disagree by {gap:.5g}"


@_needs_hip
def test_layer_forward_applies_the_layer_scalar():
    """The trained per-layer output scale must actually be applied.

    Gemma 4 multiplies each layer's output by a real scalar (0.0703125 on layer
    0). Dropping it is a ~14x error that a loose tolerance could absorb, so this
    checks the output tracks the scalar rather than only matching at one value.
    """

    tokens, hidden_size = 4, 64
    geometry = (4, 2, 16)
    weights, config, attn_geometry = _layer_fixture(
        909, tokens, hidden_size, geometry, 48, 8, 3, 32, False
    )
    rng = np.random.default_rng(3)
    hidden = (rng.standard_normal((tokens, hidden_size)) * 0.5).astype(np.float32)

    scaled = _run_layer_on_gpu(weights, config, attn_geometry, hidden, tokens, k_eq_v=False, seed=3)

    import dataclasses

    unit_weights = dataclasses.replace(weights, layer_scalar=np.float32(1.0))
    unscaled = _run_layer_on_gpu(
        unit_weights, config, attn_geometry, hidden, tokens, k_eq_v=False, seed=3
    )

    ratio = scaled / np.where(np.abs(unscaled) < 1e-6, 1.0, unscaled)
    finite = np.abs(unscaled) > 1e-6
    assert finite.sum() > 0.5 * unscaled.size, "too few usable elements to judge the scalar"
    np.testing.assert_allclose(
        ratio[finite], np.full(int(finite.sum()), 0.0703125, dtype=np.float32), rtol=1e-2
    )


@_needs_hip
def test_attention_prefill_serves_a_decode_step_over_a_kv_cache():
    """`keys` may exceed `tokens`: one query row attending over a KV cache.

    This is the shape generation actually uses. The kernel was written for a
    square prefill block, where query count and key count are the same number, so
    a decode step (one query, N cached keys) is exactly the case a kernel that
    conflated the two would get wrong. The cache here is longer than the live
    context, so a kernel reading past the live keys would attend over stale
    cache slots and produce a different answer.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_f32,
    )

    device = _Device()
    num_heads, num_kv_heads, head_dim = 4, 2, 16
    live, capacity = 6, 11
    rng = np.random.default_rng(2024)

    # A cache with stale slots past the live context. Those slots must not be read.
    key_cache = (rng.standard_normal((capacity, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    value_cache = (rng.standard_normal((capacity, num_kv_heads, head_dim)) * 0.5).astype(np.float32)
    query = (rng.standard_normal((1, num_heads, head_dim)) * 0.5).astype(np.float32)

    # A decode step keeps every live key and masks the rest of the cache.
    keep = np.zeros((1, capacity), dtype=np.uint8)
    keep[0, :live] = 1

    out_ptr = device.out((1, num_heads, head_dim), np.float32)
    gemma4_attention_prefill_f32(
        device.put(query),
        device.put(key_cache),
        device.put(value_cache),
        device.put(keep),
        out_ptr,
        tokens=1,
        keys=capacity,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=1.0,
    )
    got = device.get(out_ptr, (1, num_heads, head_dim), np.float32)

    want = _reference_attention(
        query,
        key_cache[:live],
        value_cache[:live],
        np.ones((1, live), dtype=np.uint8),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    diff = np.abs(got - want)
    assert diff.max() < 1e-4, (
        f"decode-shaped attention differs by {diff.max():.3g}; a kernel using the "
        f"query count as the key count would attend over {capacity - live} stale slots"
    )


@pytest.mark.parametrize("stream", [0, 91])
def test_expert_offset_fallback_waits_for_its_producer(monkeypatch, stream):
    """Host offsets must be collected after nonblocking-stream compaction."""
    from types import SimpleNamespace

    from hipengine.core import hip
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_experts as experts

    calls = []
    monkeypatch.setattr(
        hip, "get_hip_runtime",
        lambda: SimpleNamespace(stream_synchronize=lambda s: calls.append(("wait", s))),
    )

    def read(buffer, count):
        calls.append(("read", count))
        return np.array([0, 2, 3], dtype=np.int64)

    monkeypatch.setattr(experts, "_read_int64", read)
    monkeypatch.setattr(
        experts, "gemma4_project_expert",
        lambda *args, **kwargs: calls.append(("project", args, kwargs)),
    )
    experts.gemma4_project_experts_by_offset(
        100, 200, 300, object(), 2, 4, 5, stream=stream,
    )
    prefix = [("wait", stream), ("read", 3)] if stream else [("read", 3)]
    assert calls[:len(prefix)] == prefix
    projections = [call for call in calls if call[0] == "project"]
    assert [call[1][1:5] for call in projections] == [
        (0, 200, 300, 2), (1, 216, 320, 1),
    ]
    assert all(call[2] == {"stream": stream} for call in projections)
    assert len(calls) == (4 if stream else 3)


@pytest.mark.parametrize("seed", [777, 778, 991])
@_needs_hip
def test_layer_incremental_decode_matches_a_dense_prefill(seed):
    """Decoding one token at a time through a KV cache matches a dense prefill.

    This is the contract generation depends on. The same layer runs twice over
    the same sequence: once densely over all nine positions with a causal mask,
    and once as a six-token prefill that fills the cache followed by three
    one-token decode steps. The decode outputs must reproduce the dense rows.

    It fails if the cache append lands at the wrong offset, if attention reads
    the query count as the key count (which would silently attend over a single
    cached position), or if the mask for a decode step does not cover exactly the
    live context.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
        Gemma4LayerGeometry,
        Gemma4LayerKV,
        Gemma4LayerPointers,
        Gemma4LayerScratch,
        gemma4_layer_forward_bf16,
    )

    prompt, total = 6, 9
    hidden_size, geometry = 64, (4, 2, 16)
    dense_intermediate, num_experts, top_k, expert_intermediate = 48, 8, 3, 32
    weights, config, attn_geometry = _layer_fixture(
        seed,
        total,
        hidden_size,
        geometry,
        dense_intermediate,
        num_experts,
        top_k,
        expert_intermediate,
        False,
    )
    rng = np.random.default_rng(31)
    hidden = (rng.standard_normal((total, hidden_size)) * 0.5).astype(np.float32)

    device = _Device()
    num_heads, num_kv_heads, head_dim = geometry
    layer_geometry = Gemma4LayerGeometry(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=attn_geometry.scale,
        k_eq_v=False,
    )

    def put_bf16(arr):
        return device.put(_to_bf16_bits(np.asarray(arr, dtype=np.float32)))

    pointers = Gemma4LayerPointers(
        input_layernorm=device.put(weights.input_layernorm),
        q_proj=put_bf16(weights.q_proj),
        k_proj=put_bf16(weights.k_proj),
        o_proj=put_bf16(weights.o_proj),
        q_norm=device.put(weights.q_norm),
        k_norm=device.put(weights.k_norm),
        post_attention_layernorm=device.put(weights.post_attention_layernorm),
        pre_feedforward_layernorm=device.put(weights.pre_feedforward_layernorm),
        # The resident weight carries both halves in one buffer, laid out
        # (rows, 2 * intermediate) exactly as the fused loader emits it.
        mlp_gate_up_proj=put_bf16(
            np.concatenate([weights.mlp_gate_proj, weights.mlp_up_proj], axis=0)
        ),
        mlp_down_proj=put_bf16(weights.mlp_down_proj),
        post_feedforward_layernorm_1=device.put(weights.post_feedforward_layernorm_1),
        router_scale=device.put(weights.router_scale),
        router_proj=device.put(weights.router_proj),
        router_per_expert_scale=device.put(weights.router_per_expert_scale),
        pre_feedforward_layernorm_2=device.put(weights.pre_feedforward_layernorm_2),
        experts_gate_up_proj=put_bf16(weights.experts_gate_up_proj),
        experts_down_proj=put_bf16(weights.experts_down_proj),
        post_feedforward_layernorm_2=device.put(weights.post_feedforward_layernorm_2),
        post_feedforward_layernorm=device.put(weights.post_feedforward_layernorm),
        v_proj=put_bf16(weights.v_proj),
        layer_scalar=device.put(np.array([weights.layer_scalar], dtype=np.float32)),
    )
    # Sized for the widest block this test runs: the dense pass and the prefill
    # both use more than one row, and a scratch sized for a single decode step
    # would be overflowed by them.
    scratch = Gemma4LayerScratch(
        tokens=total,
        hidden_size=hidden_size,
        dense_intermediate=dense_intermediate,
        geometry=layer_geometry,
        num_experts=num_experts,
        top_k=top_k,
        expert_intermediate=expert_intermediate,
    )
    all_positions = np.arange(total, dtype=np.int64)
    cos, sin = gemma4_rope_cos_sin_tables(attn_geometry.rope, all_positions)

    def run(rows, *, positions, keep, kv=None, hidden_slice=None):
        mask_ptr = device.put(keep)
        target = put_bf16(
            hidden if hidden_slice is None else hidden[hidden_slice[0] : hidden_slice[1]]
        )
        table = positions
        gemma4_layer_forward_bf16(
            target,
            device.put(cos[table].astype(np.float32)),
            device.put(sin[table].astype(np.float32)),
            mask_ptr,
            pointers,
            scratch=scratch,
            kv=kv,
            rows=rows,
            eps=config.rms_norm_eps,
        )
        return device.get(target, (rows, hidden_size), np.uint16)

    try:
        # Dense reference over the whole sequence.
        dense = _from_bf16_bits(
            run(
                total,
                positions=all_positions,
                keep=np.tril(np.ones((total, total), dtype=np.uint8)),
            )
        )

        # The same sequence, prefilled then decoded one token at a time.
        key_cache = device.out((total, num_kv_heads, head_dim), np.uint16)
        value_cache = device.out((total, num_kv_heads, head_dim), np.uint16)
        prompt_positions = np.arange(prompt, dtype=np.int64)
        run(
            prompt,
            positions=prompt_positions,
            keep=np.tril(np.ones((prompt, prompt), dtype=np.uint8)),
            kv=Gemma4LayerKV(
                key_cache=key_cache,
                value_cache=value_cache,
                capacity=total,
                write_offset=0,
            ),
            hidden_slice=(0, prompt),
        )

        for position in range(prompt, total):
            live = position + 1
            step = _from_bf16_bits(
                run(
                    1,
                    positions=np.array([position], dtype=np.int64),
                    keep=np.ones((1, live), dtype=np.uint8),
                    kv=Gemma4LayerKV(
                        key_cache=key_cache,
                        value_cache=value_cache,
                        capacity=total,
                        write_offset=position,
                    ),
                    hidden_slice=(position, position + 1),
                )
            )
            diff = np.abs(step[0] - dense[position]).max()
            assert diff < 0.02 * np.abs(dense[position]).max() + 1e-4, (
                f"decode step at position {position} differs from the dense prefill by {diff:.4g}"
            )

        # The cache must hold the whole context, not just the last write.
        cached = device.get(key_cache, (total, num_kv_heads, head_dim), np.uint16)
        assert np.abs(_from_bf16_bits(cached[:prompt])).sum() > 0, "prefill wrote no keys"
        assert np.abs(_from_bf16_bits(cached[prompt:])).sum() > 0, "decode wrote no keys"
    finally:
        scratch.free()


def test_qkv_split_rejects_an_unknown_part_count() -> None:
    """A part count the kernel cannot address must fail before any launch.

    ``parts`` decides the fused stride, so a value other than 2 or 3 would make
    the derived stride disagree with the buffer actually allocated. The check
    runs before the library is built, so this holds on machines with no ROCm.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import gemma4_qkv_split_bf16

    with pytest.raises(ValueError, match="2 or 3"):
        gemma4_qkv_split_bf16(0, 0, 0, 0, 4, 4, 4, 5)


@_needs_hip
def test_qkv_split_separates_a_fused_projection_output(device) -> None:
    """P6's split must reproduce exactly what three projections would write.

    The fused projection emits one row-major
    ``[rows, q_width + 2 * kv_width]`` buffer; the attention path reads three separate buffers. This is the
    correctness half of P6's gate, asserted **bitwise**: the split only moves
    BF16 bits, so any difference at all is a layout error rather than rounding,
    and an allclose here would mask precisely the class of bug it must catch.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import gemma4_qkv_split_bf16

    rows, q_width, kv_width, parts = 7, 12, 5, 3
    stride = q_width + (parts - 1) * kv_width
    fused = (np.arange(rows * stride, dtype=np.uint32) * 40503 % 65536).astype(np.uint16)

    src = device.put(fused)
    q_out = device.out((rows, q_width), np.uint16)
    k_out = device.out((rows, kv_width), np.uint16)
    v_out = device.out((rows, kv_width), np.uint16)

    gemma4_qkv_split_bf16(src, q_out, k_out, v_out, rows, q_width, kv_width, parts)

    block = fused.reshape(rows, stride)
    np.testing.assert_array_equal(
        device.get(q_out, (rows, q_width), np.uint16), block[:, :q_width]
    )
    np.testing.assert_array_equal(
        device.get(k_out, (rows, kv_width), np.uint16),
        block[:, q_width : q_width + kv_width],
    )
    np.testing.assert_array_equal(
        device.get(v_out, (rows, kv_width), np.uint16), block[:, q_width + kv_width :]
    )


@_needs_hip
def test_qkv_split_on_a_k_eq_v_layer_leaves_v_untouched(device) -> None:
    """On a k_eq_v layer the split writes q and k and never addresses v.

    Those artifacts carry no ``attn_v``, so the fused weight is only
    ``q_width + kv_width`` wide and ``v`` keeps whatever already lives in its
    buffer (the layer reuses K for V there). A kernel that wrote v anyway would
    corrupt it, so the buffer is pre-filled with a sentinel and must come back
    unchanged.
    """
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import gemma4_qkv_split_bf16

    rows, q_width, kv_width, parts = 5, 8, 4, 2
    stride = q_width + (parts - 1) * kv_width
    fused = (np.arange(rows * stride, dtype=np.uint32) * 7919 % 65536).astype(np.uint16)
    sentinel = np.full((rows, kv_width), 0xDEAD, dtype=np.uint16)

    src = device.put(fused)
    q_out = device.out((rows, q_width), np.uint16)
    k_out = device.out((rows, kv_width), np.uint16)
    v_out = device.put(sentinel)

    gemma4_qkv_split_bf16(src, q_out, k_out, v_out, rows, q_width, kv_width, parts)

    block = fused.reshape(rows, stride)
    np.testing.assert_array_equal(
        device.get(q_out, (rows, q_width), np.uint16), block[:, :q_width]
    )
    np.testing.assert_array_equal(
        device.get(k_out, (rows, kv_width), np.uint16), block[:, q_width:]
    )
    np.testing.assert_array_equal(
        device.get(v_out, (rows, kv_width), np.uint16), sentinel
    )


# --------------------------------------------------------------------------
# Router fused rows=1: prescale + logits + select + per-expert scale in one
# --------------------------------------------------------------------------


def _router_fused_inputs(device, *, tokens: int, hidden: int, experts: int, top_k: int):
    """Put one router problem on the device and return host copies too."""

    rng = np.random.default_rng(2604)
    host_hidden = (rng.standard_normal((tokens, hidden)) * 0.5).astype(np.float32)
    host_scale = (rng.random(hidden).astype(np.float32) + 0.5) * 0.1
    host_proj = (rng.standard_normal((experts, hidden)) * 0.2).astype(np.float32)
    host_per_expert = (rng.random(experts).astype(np.float32) + 0.5) * 0.3
    return (
        device.put(_to_bf16_bits(host_hidden)),
        device.put(host_scale),
        device.put(host_proj),
        device.put(host_per_expert),
        device.out((tokens, top_k), np.int64),
        device.out((tokens, top_k), np.float32),
        host_hidden,
        host_scale,
        host_proj,
        host_per_expert,
    )


def _router_fused_reference(host_hidden, host_scale, host_proj, host_per_expert, top_k):
    from hipengine.kernels.cpu_reference.gemma4 import gemma4_rmsnorm, gemma4_router_topk

    return gemma4_router_topk(
        host_hidden,
        norm_weight=None,
        scale=host_scale,
        proj_weight=host_proj,
        per_expert_scale=host_per_expert,
        top_k=top_k,
        scalar_root_size=host_hidden.shape[-1] ** -0.5,
        eps=1e-6,
    )


def _decisive_mask(host_hidden, host_scale, host_proj, top_k):
    """Tokens whose k-th/(k+1)-th gap exceeds the bf16 prescale perturbation."""

    from hipengine.kernels.cpu_reference.gemma4 import gemma4_rmsnorm
    from hipengine.kernels.cpu_reference.ops import linear

    hidden_size = host_hidden.shape[-1]
    ref_hidden = _from_bf16_bits(_to_bf16_bits(host_hidden))
    pre_f32 = (
        gemma4_rmsnorm(ref_hidden, None, 1e-6) * host_scale * np.float32(hidden_size**-0.5)
    ).astype(np.float32)
    pre_bf16 = _from_bf16_bits(_to_bf16_bits(pre_f32))
    logits_f32 = linear(pre_f32, host_proj)
    perturbation = np.abs(logits_f32 - linear(pre_bf16, host_proj)).max()
    ordered = np.sort(logits_f32, axis=-1)
    gap = ordered[:, -top_k] - ordered[:, -(top_k + 1)]
    return gap > perturbation, perturbation


@_needs_hip
def test_router_topk_fused_matches_the_reference(device) -> None:
    """The fused rows=1 router reproduces gemma4_router_topk end to end."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_router import (
        Gemma4RouterScratch,
        gemma4_router_topk_fused_bf16,
    )

    tokens, hidden_size, num_experts, top_k = 1, 64, 12, 4
    (
        hidden_ptr,
        scale_ptr,
        proj_ptr,
        per_expert_ptr,
        selected_ptr,
        weights_ptr,
        host_hidden,
        host_scale,
        host_proj,
        host_per_expert,
    ) = _router_fused_inputs(
        device, tokens=tokens, hidden=hidden_size, experts=num_experts, top_k=top_k
    )

    scratch = Gemma4RouterScratch(
        tokens=tokens, hidden_size=hidden_size, num_experts=num_experts, top_k=top_k
    )
    # Sentinel coverage first, per the lesson in the P8 prescale-fold entry:
    # a fused kernel that writes only part of its output can still agree with
    # a reference on the elements a comparison happens to inspect. Every logit
    # and every weight must be written before this call returns.
    sentinel_logits = np.full((tokens, num_experts), -12345.0, dtype=np.float32)
    sentinel_weights = np.full((tokens, top_k), 9999.0, dtype=np.float32)
    logits_ptr = scratch.buffer("logits").ptr
    copy_host_to_device(
        DeviceBuffer(ptr=logits_ptr, nbytes=sentinel_logits.nbytes),
        host_array_ptr(sentinel_logits),
        sentinel_logits.nbytes,
    )
    copy_host_to_device(
        DeviceBuffer(ptr=weights_ptr, nbytes=sentinel_weights.nbytes),
        host_array_ptr(sentinel_weights),
        sentinel_weights.nbytes,
    )
    try:
        gemma4_router_topk_fused_bf16(
            hidden_ptr,
            scale_ptr,
            proj_ptr,
            per_expert_ptr,
            selected_ptr,
            weights_ptr,
            tokens=tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            scratch=scratch,
        )
        got_selected = device.get(selected_ptr, (tokens, top_k), np.int64)
        got_weights = device.get(weights_ptr, (tokens, top_k), np.float32)
        got_logits = device.get(logits_ptr, (tokens, num_experts), np.float32)
    finally:
        scratch.free()

    assert not (got_logits == -12345.0).any(), (
        "fused router left logits unwritten -- sentinel coverage failure"
    )
    assert not (got_weights == 9999.0).any(), (
        "fused router left routing weights unwritten -- sentinel coverage failure"
    )

    _probs, want_weights, want_selected = _router_fused_reference(
        host_hidden, host_scale, host_proj, host_per_expert, top_k
    )
    decisive, perturbation = _decisive_mask(
        host_hidden, host_scale, host_proj, top_k
    )
    assert decisive.any(), "no decisive token; the assertion below would prove nothing"

    got = np.asarray(got_selected)
    want = np.asarray(want_selected).astype(np.int64)
    np.testing.assert_array_equal(
        got[decisive],
        want[decisive],
        err_msg="fused router selection differs on a decisive token",
    )
    np.testing.assert_allclose(
        got_weights, want_weights, atol=4 * perturbation, rtol=0
    )


@_needs_hip
def test_router_topk_fused_agrees_with_the_unfused_chain(device) -> None:
    """At tokens=1 the fused launch and the registered chain select the same experts.

    The two paths share the bf16-rounded prescaled row, so they differ only in
    f32 reduction order inside the projection: selection must match wherever
    the decision is not inside that rounding, and weights only at the level
    the reduction order can move a softmax numerator.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_router import (
        Gemma4RouterScratch,
        gemma4_router_topk_bf16,
        gemma4_router_topk_fused_bf16,
    )

    tokens, hidden_size, num_experts, top_k = 1, 64, 12, 4
    inputs = _router_fused_inputs(
        device, tokens=tokens, hidden=hidden_size, experts=num_experts, top_k=top_k
    )
    (
        hidden_ptr,
        scale_ptr,
        proj_ptr,
        per_expert_ptr,
        _selected_ptr,
        _weights_ptr,
        host_hidden,
        host_scale,
        host_proj,
        host_per_expert,
    ) = inputs
    chain_selected_ptr = device.out((tokens, top_k), np.int64)
    chain_weights_ptr = device.out((tokens, top_k), np.float32)
    fused_selected_ptr = device.out((tokens, top_k), np.int64)
    fused_weights_ptr = device.out((tokens, top_k), np.float32)

    scratch = Gemma4RouterScratch(
        tokens=tokens, hidden_size=hidden_size, num_experts=num_experts, top_k=top_k
    )
    try:
        gemma4_router_topk_bf16(
            hidden_ptr,
            scale_ptr,
            proj_ptr,
            per_expert_ptr,
            chain_selected_ptr,
            chain_weights_ptr,
            tokens=tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            scratch=scratch,
        )
        gemma4_router_topk_fused_bf16(
            hidden_ptr,
            scale_ptr,
            proj_ptr,
            per_expert_ptr,
            fused_selected_ptr,
            fused_weights_ptr,
            tokens=tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            scratch=scratch,
        )
        chain_selected = device.get(chain_selected_ptr, (tokens, top_k), np.int64)
        chain_weights = device.get(chain_weights_ptr, (tokens, top_k), np.float32)
        fused_selected = device.get(fused_selected_ptr, (tokens, top_k), np.int64)
        fused_weights = device.get(fused_weights_ptr, (tokens, top_k), np.float32)
    finally:
        scratch.free()

    decisive, perturbation = _decisive_mask(
        host_hidden, host_scale, host_proj, top_k
    )
    assert decisive.any(), "no decisive token; the assertion below would prove nothing"
    np.testing.assert_array_equal(
        fused_selected[decisive],
        chain_selected[decisive],
        err_msg="fused and chain selection differ on a decisive token",
    )
    # Both paths round prescale to bf16 before projecting, so only f32
    # reduction order separates their logits; a softmax over top_k values
    # moves by far less than this under that class of difference.
    np.testing.assert_allclose(fused_weights, chain_weights, atol=1e-5, rtol=1e-4)


@_needs_hip
def test_enqueue_host_to_device_consumes_a_temporary_pageable_source() -> None:
    """The staging path uploads a freshly built array and drops it immediately.

    ``_stage_upload`` and the token-ids upload build their source with
    ``np.ascontiguousarray`` and keep no reference: the enqueued copy must be
    self-contained by the time the call returns (ROCm stages pageable memory
    during the call, measured at ~11 us with a deep queue in
    ``scratch/d8_async_h2d_screen.py``), or the device would read host memory
    the caller is about to free. The sync path hid this by blocking until the
    copy completed; the enqueued path has to hold up on its own.
    """

    import gc

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import enqueue_host_to_device

    n = 4096
    reference = np.arange(n, dtype=np.uint8)
    dst = malloc(n)
    try:
        # Exactly the production pattern: temporary source, dropped on return.
        enqueue_host_to_device(
            dst, host_array_ptr(np.ascontiguousarray(reference)), n, stream=0
        )
        del reference  # type: ignore[has-type]
        gc.collect()
        get_hip_runtime().stream_synchronize(0)
        readback = np.empty(n, dtype=np.uint8)
        copy_device_to_host(host_array_ptr(readback), dst, n)
        np.testing.assert_array_equal(readback, np.arange(n, dtype=np.uint8))
    finally:
        free(dst)
