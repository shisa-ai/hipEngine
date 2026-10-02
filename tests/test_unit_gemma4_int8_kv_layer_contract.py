"""The INT8 layer route's mask contract, checked with no device work.

The direct INT8 consumer derives its own mask from causal order plus the layer's
sliding window and never reads the caller's ``keep_mask_ptr``. The layer can
therefore honour exactly the causal/window contract; any other mask semantics
-- an eviction or exclusion mask, say -- would be silently dropped. It refuses
that up front, before it acquires a stream or launches a kernel, rather than
producing a plausible-but-wrong result.

No device: every launcher the layer would call before and at the attention block
is replaced with a recorder, so a refusal is provable by an empty call list and
an acceptance is provable by the INT8 attention entry appearing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_layer as layer_module

_PRE_ATTENTION_LAUNCHERS = (
    "gemma4_rmsnorm_f32w_bf16",
    "gemma4_project",
    "gemma4_qkv_split_bf16",
    "gemma4_rmsnorm_weightless_bf16",
    "gemma4_head_rmsnorm_f32w_bf16",
    "gemma4_partial_rotary_bf16",
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            layer_module,
            "_moe_stream",
            lambda: self.calls.append("_moe_stream") or 0,
        )
        for name in _PRE_ATTENTION_LAUNCHERS:
            monkeypatch.setattr(
                layer_module,
                name,
                lambda *args, _name=name, **kwargs: self.calls.append(_name),
            )
        monkeypatch.setattr(
            layer_module,
            "_run_int8_attention",
            lambda *args, **kwargs: self.calls.append("_run_int8_attention"),
        )


def _scratch() -> SimpleNamespace:
    return SimpleNamespace(
        tokens=4,
        hidden_size=16,
        geometry=SimpleNamespace(
            num_heads=4, num_kv_heads=2, head_dim=8, scale=1.0, k_eq_v=False
        ),
        buffer=lambda name: SimpleNamespace(ptr=0),
    )


def _layer() -> SimpleNamespace:
    return SimpleNamespace(
        input_layernorm=0,
        qkv_proj=None,
        q_proj=0,
        k_proj=0,
        v_proj=0,
        q_norm=0,
        k_norm=0,
    )


def test_int8_route_refuses_a_non_causal_mask_before_device_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller asking for arbitrary exclusions is refused, not silently trimmed."""

    recorder = _Recorder()
    recorder.install(monkeypatch)
    with pytest.raises(ValueError, match="attention_mask_is_causal"):
        layer_module.gemma4_layer_forward_bf16(
            0,
            0,
            0,
            0,
            _layer(),
            scratch=_scratch(),
            int8_kv=object(),
            rows=4,
            attention_mask_is_causal=False,
        )
    assert recorder.calls == [], "the refusal must precede every device call"


def test_int8_route_refuses_a_second_cache_before_device_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both routes at once is a caller error, caught before any launch."""

    recorder = _Recorder()
    recorder.install(monkeypatch)
    with pytest.raises(ValueError, match="either a BF16 or an INT8 KV cache"):
        layer_module.gemma4_layer_forward_bf16(
            0,
            0,
            0,
            0,
            _layer(),
            scratch=_scratch(),
            kv=SimpleNamespace(
                key_cache=0, value_cache=0, capacity=4, write_offset=0
            ),
            int8_kv=object(),
            rows=4,
            attention_mask_is_causal=True,
        )
    assert recorder.calls == []


class _ReachedInt8(Exception):
    """Sentinel: the INT8 route was selected and entered."""


def test_int8_route_accepts_the_causal_window_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is narrow: causal/window semantics still take the INT8 route."""

    recorder = _Recorder()
    recorder.install(monkeypatch)

    def _enter_int8(*args: object, **kwargs: object) -> None:
        recorder.calls.append("_run_int8_attention")
        raise _ReachedInt8

    monkeypatch.setattr(layer_module, "_run_int8_attention", _enter_int8)
    with pytest.raises(_ReachedInt8):
        layer_module.gemma4_layer_forward_bf16(
            0,
            0,
            0,
            0,
            _layer(),
            scratch=_scratch(),
            int8_kv=object(),
            rows=4,
            attention_mask_is_causal=True,
        )
    assert "_run_int8_attention" in recorder.calls
    # The BF16 attention path is skipped entirely, not run and discarded.
    assert not any(name.startswith("gemma4_attention_") for name in recorder.calls)
