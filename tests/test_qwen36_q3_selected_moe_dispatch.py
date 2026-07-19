"""Runtime dispatch contract for Qwen3.6 Q3 selected decode kernels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as qgr


class _Weight:
    def __init__(self, quant_key: str, raw_ptr: int, name: str) -> None:
        self.spec = SimpleNamespace(
            quant_key=quant_key,
            source=SimpleNamespace(name=name),
        )
        self._raw_ptr = raw_ptr

    def allocation(self, name: str = "raw"):
        assert name == "raw"
        return SimpleNamespace(tensor=SimpleNamespace(ptr=self._raw_ptr))


@pytest.mark.parametrize(
    "quant_key,kernel_name",
    [
        ("gguf_iq3_xxs", "gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out"),
        ("gguf_iq4_xs", "gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out"),
    ],
)
def test_q3_selected_gate_up_silu_routes_raw_iq_weights(
    monkeypatch: pytest.MonkeyPatch,
    quant_key: str,
    kernel_name: str,
) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        qgr,
        kernel_name,
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    runtime = object()

    assert qgr._launch_selected_raw_gguf_moe_pair_silu(
        _Weight(quant_key, 301, "gate"),
        _Weight(quant_key, 401, "up"),
        101,
        201,
        501,
        x_rows=1,
        rows=8,
        num_experts=256,
        in_features=2048,
        out_features=512,
        stream=7,
        runtime=runtime,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (101, 201, 301, 401, 501, 1, 8, 256, 2048, 512)
    assert kwargs == {"stream": 7, "runtime": runtime}


def test_q3_selected_down_routes_raw_iq4_xs_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        qgr,
        "gguf_iq4_xs_selected_gemv_bf16_bf16_out",
        lambda *args, **kwargs: calls.append((args, kwargs)),
        raising=False,
    )
    runtime = object()

    qgr._launch_selected_raw_gguf_moe_linear(
        _Weight("gguf_iq4_xs", 303, "down"),
        103,
        203,
        403,
        x_rows=8,
        rows=8,
        num_experts=256,
        in_features=512,
        out_features=2048,
        stream=9,
        runtime=runtime,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (103, 203, 303, 403, 8, 8, 256, 512, 2048)
    assert kwargs == {"stream": 9, "runtime": runtime}
