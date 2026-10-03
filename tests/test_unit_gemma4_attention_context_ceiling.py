"""Storage-specific context admission, including the repaired BF16 long path."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_shared_bytes
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import Gemma4Runner, gemma4_require_context_capacity


def _config(*, head_dim: int, window: int | None) -> Gemma4TextConfig:
    geometry = Gemma4AttentionGeometry(
        layer_type="sliding_attention" if window else "full_attention",
        num_heads=16, num_kv_heads=2, head_dim=head_dim,
        rope=Gemma4RopeConfig(
            rope_theta=10_000.0, head_dim=head_dim, rope_angles=head_dim // 2
        ),
        sliding_window=window, k_eq_v=False,
    )
    return Gemma4TextConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_experts=4, top_k_experts=2, rms_norm_eps=1e-6,
        attention=(geometry,), vocab_size=128,
    )


def _construct(config: Gemma4TextConfig, capacity: int, monkeypatch) -> None:
    def _malloc(nbytes: int):
        raise AssertionError("allocation reached after capacity admission")

    monkeypatch.setattr(gemma4_module, "malloc", _malloc)
    monkeypatch.setattr(gemma4_module, "free", lambda buffer: None)
    Gemma4Runner(weights=SimpleNamespace(config=config), capacity=capacity, max_block=8)


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("capacity", [1024, 15_616, 15_617, 15_856, 15_857, 16_384, 32_768])
def test_repaired_bf16_geometries_admit_both_sides_of_the_old_lds_boundary(
    head_dim: int, capacity: int, monkeypatch
) -> None:
    assert gemma4_attention_shared_bytes(head_dim=head_dim, keys=capacity) <= 65_536
    with pytest.raises(AssertionError, match="allocation reached"):
        _construct(_config(head_dim=head_dim, window=None), capacity, monkeypatch)


@pytest.mark.parametrize("window", [None, 1024])
def test_an_unimplemented_global_score_geometry_is_refused_before_allocation(window, monkeypatch) -> None:
    with pytest.raises(NotImplementedError) as caught:
        _construct(_config(head_dim=128, window=window), 16_384, monkeypatch)
    assert "128" in str(caught.value)
    assert "16384" in str(caught.value)


def test_int8_admission_uses_its_own_consumer_not_the_bf16_global_path() -> None:
    config = _config(head_dim=512, window=None)
    gemma4_require_context_capacity(config, 16_384, kv_storage="bf16")
    with pytest.raises(ValueError, match="INT8 attention consumer"):
        gemma4_require_context_capacity(config, 16_384, kv_storage="int8_per_token_head")


@pytest.mark.parametrize("head_dim,servable", [(128, False), (512, True)])
def test_generator_checks_selected_context_capability_before_loading_weights(head_dim, servable, monkeypatch) -> None:
    import hipengine.generation.gemma4_gguf as module

    def _load(*args, **kwargs):
        raise AssertionError("weights loader reached")

    monkeypatch.setattr(module, "load_gemma4_device_weights", _load)
    monkeypatch.setattr(module, "gemma4_text_config_from_reader", lambda *a, **k: _config(head_dim=head_dim, window=None))
    generator = module.Gemma4GGUFGenerator("/unused.gguf", object(), object(), context_length=32_768)
    generator._reader = object()
    if servable:
        with pytest.raises(AssertionError, match="weights loader reached"):
            generator._ensure_runner()
    else:
        with pytest.raises(NotImplementedError):
            generator._ensure_runner()
