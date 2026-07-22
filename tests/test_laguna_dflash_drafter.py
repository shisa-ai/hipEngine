from __future__ import annotations

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.loading.dflash import DFlashDrafterDeviceWeights, dflash_draft_config_from_hf
from hipengine.speculative.dflash_drafter import (
    gate_laguna_dflash_attention_bf16,
    project_laguna_dflash_target_hidden_bf16,
)


def test_laguna_dflash_target_projection_normalizes_each_tap_and_row(monkeypatch) -> None:
    import hipengine.speculative.dflash_drafter as module

    config = dflash_draft_config_from_hf(_config())
    weights = _weights(config)
    taps = (
        _tensor(0x1000, (2, 4)),
        _tensor(0x2000, (2, 4)),
    )
    normalized_concat = _tensor(0x3000, (2, 8))
    projection_scratch = _tensor(0x4000, (2, 4))
    projected = _tensor(0x5000, (2, 4))
    norm_calls: list[tuple[int, int, int]] = []
    projection_calls: list[tuple[Tensor, Tensor, Tensor]] = []

    def fake_norm(hidden, weight, out, **kwargs):
        del kwargs
        norm_calls.append((hidden.ptr, weight.ptr, out.ptr))
        return out

    def fake_project(concat, out, scratch, _weights, **kwargs):
        del kwargs
        assert _weights is weights
        projection_calls.append((concat, out, scratch))
        return out

    monkeypatch.setattr(module, "dflash_rmsnorm_bf16", fake_norm)
    monkeypatch.setattr(module, "project_dflash_target_hidden_bf16", fake_project)

    result = project_laguna_dflash_target_hidden_bf16(
        taps,
        normalized_concat,
        projected,
        projection_scratch,
        weights,
    )

    assert result is projected
    assert norm_calls == [
        (0x1000, 0x6000, 0x3000),
        (0x2000, 0x7000, 0x3000 + 4 * 2),
        (0x1000 + 4 * 2, 0x6000, 0x3000 + 8 * 2),
        (0x2000 + 4 * 2, 0x7000, 0x3000 + (8 + 4) * 2),
    ]
    assert projection_calls == [(normalized_concat, projected, projection_scratch)]


def test_laguna_dflash_target_projection_rejects_wrong_tap_count() -> None:
    config = dflash_draft_config_from_hf(_config())
    weights = _weights(config)

    try:
        project_laguna_dflash_target_hidden_bf16(
            (_tensor(0x1000, (1, 4)),),
            _tensor(0x3000, (1, 8)),
            _tensor(0x4000, (1, 4)),
            _tensor(0x5000, (1, 4)),
            weights,
        )
    except ValueError as exc:
        assert "tap count" in str(exc)
    else:
        raise AssertionError("wrong Laguna DFlash tap count was accepted")


def test_laguna_dflash_gate_projects_softplus_head_scalars(monkeypatch) -> None:
    import hipengine.speculative.dflash_drafter as module

    config = dflash_draft_config_from_hf(_config())
    weights = _weights(config)
    normalized = _tensor(0x1000, (3, 4))
    context = _tensor(0x2000, (3, 4))
    gate_logits = _tensor(0x3000, (3, 2), dtype="fp32")
    gated = _tensor(0x4000, (3, 4))
    projection_calls = []
    gate_calls = []

    def fake_projection(hidden, weight, out, **kwargs):
        projection_calls.append((hidden, weight, out, kwargs))
        return out

    def fake_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))

    monkeypatch.setattr(module, "project_dflash_bf16_to_f32", fake_projection)

    result = gate_laguna_dflash_attention_bf16(
        normalized,
        context,
        gate_logits,
        gated,
        weights,
        layer=0,
        gate_kernel=fake_gate,
        stream=7,
        library="gate-lib",
        projection_library="dense-lib",
    )

    assert result is gated
    assert projection_calls[0][0] is normalized
    assert projection_calls[0][1].ptr == 0x8000
    assert projection_calls[0][2] is gate_logits
    assert projection_calls[0][3]["library"] == "dense-lib"
    assert gate_calls == [
        (
            (context.ptr, gate_logits.ptr, gated.ptr, 3, 2, 2),
            {"stream": 7, "library": "gate-lib", "runtime": None},
        )
    ]


class _FakeWeightMap:
    def __init__(self, tensors: dict[str, Tensor]) -> None:
        self.tensors = tensors

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name]

    def free(self, *, runtime=None) -> None:
        del runtime


def _weights(config) -> DFlashDrafterDeviceWeights:
    tensors = {
        "aux_hidden_norms.0.weight": _tensor(0x6000, (4,)),
        "aux_hidden_norms.1.weight": _tensor(0x7000, (4,)),
        "layers.0.self_attn.g_proj.weight": _tensor(0x8000, (2, 4)),
    }
    return DFlashDrafterDeviceWeights(
        config=config,
        weights=_FakeWeightMap(tensors),  # type: ignore[arg-type]
        layer_limit=1,
    )


def _tensor(ptr: int, shape: tuple[int, ...], *, dtype: str = "bf16") -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _config() -> dict:
    return {
        "architectures": ["DFlashLagunaForCausalLM"],
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "rms_norm_eps": 1.0e-6,
        "max_position_embeddings": 64,
        "sliding_window": 512,
        "sliding_windows": [512],
        "layer_types": ["sliding_attention"],
        "rope_theta": 500_000.0,
        "gating": "per-head",
        "vocab_size": 32,
        "draft_vocab_size": 32,
        "torch_dtype": "bfloat16",
        "eagle_aux_hidden_state_layer_ids": [1, 2],
        "dflash_config": {
            "block_size": 4,
            "mask_token_id": 12,
            "num_target_layers": 2,
            "target_layer_ids": [0, 1],
            "causal": True,
        },
    }
