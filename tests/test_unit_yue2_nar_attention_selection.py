"""YuE2 NAR attention selection is a geometry guard, not an environment policy.

``nar_wmma.hip`` implements exactly one head geometry -- 16 query heads over 8
key/value heads at head_dim 128 (``nar_wmma.hip`` ``Q_HEADS``/``KV_HEADS``/
``HEAD_DIM``) -- and returns a HIP error for anything else rather than computing
something wrong. The runtime therefore has to route every other geometry to the
strict scalar ``nar_attention_f32`` kernel, which stays registered as the
bit-exact parent oracle.

``HIPENGINE_YUE2_NAR_ATTENTION`` was the temporary rollback lever for the
tensor-core path while it was being promoted. It is retired: the tensor-core
path held through a release cycle with no solver regression traced to it, so
selection is now a pure function of the model geometry and no environment value
can change it.

The constructor tests below drive the real ``Yue2NarRuntime.__init__`` against
fake weights and a fake AR owner with allocation, upload and HIP-runtime
lookups patched out, so the selected kernel is asserted through the object the
runtime actually builds rather than through the predicate alone.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.yue2_nar as yue2_nar
from hipengine.kernels.hip_gfx1100.yue2 import nar as nar_kernels
from hipengine.runtime.yue2_nar import wmma_attention_supported

#: The retired rollback lever. Kept as a literal so the absence check below
#: cannot be satisfied by a renamed constant.
RETIRED_FLAG = "HIPENGINE_YUE2_NAR_ATTENTION"

#: The production head geometry and one the device guard rejects.
PRODUCTION_GEOMETRY = (16, 8, 128)
UNSUPPORTED_GEOMETRY = (16, 4, 128)


class _FakeBuffer:
    """Stand-in for ``DeviceBuffer``: the constructor only stores these."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = ptr
        self.nbytes = nbytes


def _fake_weights(*geometry: int) -> SimpleNamespace:
    """Weights shaped like ``YuE2Weights`` with no device allocations."""

    heads, kv_heads, head_dim = geometry
    hidden, ffn, latent = 64, 96, 8

    def bits(count: int) -> np.ndarray:
        return np.zeros(count, dtype=np.uint16)

    attention = SimpleNamespace(
        q_norm=bits(head_dim),
        k_norm=bits(head_dim),
        q=bits(heads * head_dim * hidden),
        k=bits(kv_heads * head_dim * hidden),
        v=bits(kv_heads * head_dim * hidden),
        o=bits(heads * head_dim * hidden),
    )
    layer = SimpleNamespace(
        nar_self_attn=attention,
        nar_input_layernorm=bits(hidden),
        nar_pre_mlp_layernorm=bits(hidden),
        nar_mlp=SimpleNamespace(
            gate=bits(ffn * hidden),
            up=bits(ffn * hidden),
            down=bits(hidden * ffn),
        ),
    )
    spec = SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=ffn,
        latent_dim=latent,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=16,
        rope_theta=10000.0,
        timestep_shift=3.0,
        rms_norm_eps=1e-6,
    )
    return SimpleNamespace(
        config=spec,
        layers=[layer],
        norm=bits(hidden),
        vae2llm_weight=bits(hidden * latent),
        vae2llm_bias=bits(hidden),
        llm2vae_weight=bits(latent * hidden),
        llm2vae_bias=bits(latent),
        time_embedder=SimpleNamespace(
            first_weight=bits(hidden),
            first_bias=bits(hidden),
            second_weight=bits(hidden * hidden),
            second_bias=bits(hidden),
        ),
        latent_pos_embed=np.zeros((4, latent), dtype=np.uint16),
    )


def _fake_ar(*geometry: int) -> SimpleNamespace:
    """The resident AR owner: the NAR only borrows its library and kernels."""

    heads, kv_heads, head_dim = geometry
    return SimpleNamespace(
        spec=SimpleNamespace(hidden_size=64),
        library=object(),
        kernels=SimpleNamespace(),
        _lt=None,
    )


@pytest.fixture
def fake_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``__init__`` off the device: no HIP runtime, no allocations."""

    counter = iter(range(1, 1_000_000))
    monkeypatch.setattr(
        yue2_nar, "_alloc", lambda nbytes: _FakeBuffer(next(counter), int(nbytes))
    )
    monkeypatch.setattr(
        yue2_nar, "_upload", lambda host: _FakeBuffer(next(counter), int(host.nbytes))
    )
    monkeypatch.setattr(yue2_nar, "get_hip_runtime", lambda: SimpleNamespace())


def _build(geometry: tuple[int, int, int]) -> yue2_nar.Yue2NarRuntime:
    return yue2_nar.Yue2NarRuntime(_fake_weights(*geometry), _fake_ar(*geometry))


def test_production_head_geometry_selects_the_tensor_core_attention() -> None:
    assert wmma_attention_supported(*PRODUCTION_GEOMETRY) is True


@pytest.mark.parametrize(
    "heads,kv_heads,head_dim",
    [
        UNSUPPORTED_GEOMETRY,  # the geometry the device guard rejects
        (16, 8, 64),  # narrower head
        (16, 8, 256),  # wider head
        (32, 8, 128),  # wider query stack
        (8, 8, 128),  # MHA-style grouping
    ],
)
def test_other_geometries_select_the_strict_scalar_attention(
    heads: int, kv_heads: int, head_dim: int
) -> None:
    assert wmma_attention_supported(heads, kv_heads, head_dim) is False


@pytest.mark.parametrize("value", ["scalar", "wmma", "0", "", "SCALAR"])
def test_selection_ignores_the_retired_rollback_flag(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """No environment value may move a model off its geometry-selected kernel."""

    monkeypatch.setenv(RETIRED_FLAG, value)
    assert wmma_attention_supported(*PRODUCTION_GEOMETRY) is True
    assert wmma_attention_supported(*UNSUPPORTED_GEOMETRY) is False


def test_runtime_module_no_longer_reads_the_retired_flag() -> None:
    """The flag has to leave the runtime, not merely stop being consulted."""

    source = Path(yue2_nar.__file__).read_text(encoding="utf-8")
    assert RETIRED_FLAG not in source


def test_constructor_selects_the_tensor_core_kernel_for_production_geometry(
    fake_device: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retired ``scalar`` value must no longer redirect a production model."""

    monkeypatch.setenv(RETIRED_FLAG, "scalar")
    runtime = _build(PRODUCTION_GEOMETRY)
    assert runtime._wmma_attention is True
    assert runtime._attention_kernel() is nar_kernels.nar_attention_wmma
    # The patched allocator is what kept this off the device.
    assert all(isinstance(buffer, _FakeBuffer) for buffer in runtime._buffers)
    assert runtime._buffers


def test_constructor_keeps_the_scalar_kernel_for_unsupported_geometry(
    fake_device: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No environment value may enable a geometry the device kernel rejects."""

    monkeypatch.setenv(RETIRED_FLAG, "wmma")
    runtime = _build(UNSUPPORTED_GEOMETRY)
    assert runtime._wmma_attention is False
    assert runtime._attention_kernel() is nar_kernels.nar_attention_f32
    assert all(isinstance(buffer, _FakeBuffer) for buffer in runtime._buffers)
    assert runtime._buffers
