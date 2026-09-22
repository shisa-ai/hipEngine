"""Gemma 4 gfx1100 kernel and rotary-table contracts.

The rotary-table tests need no GPU: they pin the layout that
``qwen35_partial_rotary_kernel`` indexes against HuggingFace's ``rotate_half``
form, which is the reason no new rotary kernel is needed. The kernel tests are
guarded so a no-ROCm runner skips them rather than failing release validation.
"""

from __future__ import annotations

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
        "hipengine_gemma4_add_rmsnorm_scale_bf16",
        "hipengine_gemma4_expert_weight_scale_f32",
        "hipengine_gemma4_branch_add_bf16",
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
def test_kernels_reject_degenerate_shapes(device) -> None:
    src = device.put(np.zeros(8, dtype=np.float32))
    dst = device.out((8,), np.float32)
    with pytest.raises(ValueError):
        gemma4_rmsnorm_f32w_f32(src, src, dst, 0, 8, 1e-6)
    with pytest.raises(ValueError):
        gemma4_branch_add_bf16(src, src, dst, 0)


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
