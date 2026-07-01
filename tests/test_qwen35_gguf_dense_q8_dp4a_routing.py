from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as qgr


def _buf(ptr: int, nbytes: int = 4096):
    return SimpleNamespace(ptr=ptr, nbytes=nbytes)


def _weight(*, quant_key: str = "gguf_q8_0_t16_v1", raw: bool = True):
    allocations = {"tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=10))}
    if raw:
        allocations["raw"] = SimpleNamespace(tensor=SimpleNamespace(ptr=20))

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(quant_key=quant_key)

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


def test_dense_q8_dp4a_route_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DENSE_Q8_DP4A", raising=False)
    monkeypatch.setattr(qgr, "gguf_q4_k_quantize_bf16_q8_1", lambda *args, **kwargs: pytest.fail("quantize"))
    monkeypatch.setattr(qgr, "gguf_q8_0_dp4a_gemv_bf16_bf16_out", lambda *args, **kwargs: pytest.fail("dp4a"))

    assert not qgr._try_launch_dense_q8_pair_dp4a(
        _weight(),
        _weight(),
        100,
        200,
        300,
        SimpleNamespace(moe_q8_1=_buf(400)),
        rows=2,
        in_features=2048,
        out_features_a=8192,
        out_features_b=4096,
        stream=0,
        runtime=SimpleNamespace(),
    )


def test_dense_q8_dp4a_route_quantizes_once_and_launches_two_gemvs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DENSE_Q8_DP4A", "1")
    calls: list[tuple[str, tuple, dict]] = []

    def quantize(*args, **kwargs):
        calls.append(("quantize", args, kwargs))

    def dp4a(*args, **kwargs):
        calls.append(("dp4a", args, kwargs))

    monkeypatch.setattr(qgr, "gguf_q4_k_quantize_bf16_q8_1", quantize)
    monkeypatch.setattr(qgr, "gguf_q8_0_dp4a_gemv_bf16_bf16_out", dp4a)

    assert qgr._try_launch_dense_q8_pair_dp4a(
        _weight(),
        _weight(),
        100,
        200,
        300,
        SimpleNamespace(moe_q8_1=_buf(400, nbytes=2 * (2048 // 32) * 36)),
        rows=2,
        in_features=2048,
        out_features_a=8192,
        out_features_b=4096,
        stream=7,
        runtime=SimpleNamespace(),
    )

    assert [name for name, _args, _kwargs in calls] == ["quantize", "dp4a", "dp4a"]
    assert calls[0][1][:4] == (100, 400, 2, 2048)
    assert calls[1][1][:6] == (400, 20, 200, 2, 2048, 8192)
    assert calls[2][1][:6] == (400, 20, 300, 2, 2048, 4096)


def test_dense_q8_dp4a_route_requires_raw_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_DENSE_Q8_DP4A", "1")
    monkeypatch.setattr(qgr, "gguf_q4_k_quantize_bf16_q8_1", lambda *args, **kwargs: pytest.fail("quantize"))
    monkeypatch.setattr(qgr, "gguf_q8_0_dp4a_gemv_bf16_bf16_out", lambda *args, **kwargs: pytest.fail("dp4a"))

    assert not qgr._try_launch_dense_q8_pair_dp4a(
        _weight(raw=False),
        _weight(),
        100,
        200,
        300,
        SimpleNamespace(moe_q8_1=_buf(400)),
        rows=2,
        in_features=2048,
        out_features_a=8192,
        out_features_b=4096,
        stream=0,
        runtime=SimpleNamespace(),
    )
