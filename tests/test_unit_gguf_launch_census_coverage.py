"""Which dense GGUF linear families the launch census can see.

The census is the instrument a route-execution claim is checked with, so a
family that does not record makes "did not run" and "not counted" the same
observation. The 2026-09-17 promotion entry read "2144 coltile launches and zero
``wmma_prefill`` launches" off a census that could not see the WMMA family at
all, and inferred that the Q8 dense linear route was not where its win came
from. These tests pin the families that record, so the next reader can tell an
empty row from a blind instrument.
"""

from __future__ import annotations

from hipengine.kernels import launch_census
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q8_0_dense_wide,
    gguf_q8_0_prefill,
)


class _FakeFunction:
    argtypes = None
    restype = None

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return 0


class _FakeLibrary:
    def __init__(self) -> None:
        self.functions: dict[str, _FakeFunction] = {}

    def __getattr__(self, name: str) -> _FakeFunction:
        return self.functions.setdefault(name, _FakeFunction())


class _FakeRuntime:
    def check(self, err: int) -> None:  # pragma: no cover - only on failure
        raise AssertionError(f"unexpected launch error {err}")


def _recorded(monkeypatch) -> list[tuple[object, ...]]:
    recorded: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        launch_census, "record_launch", lambda *args, **kwargs: recorded.append(args)
    )
    return recorded


def test_the_wide_q8_0_kernel_records_into_the_census(monkeypatch):
    recorded = _recorded(monkeypatch)

    gguf_q8_0_dense_wide._launch(
        "hipengine_gguf_q8_0_dense_wide256_f32_f32_out",
        1,
        2,
        3,
        512,
        2560,
        10240,
        library=_FakeLibrary(),
        runtime=_FakeRuntime(),
    )

    assert recorded == [
        (
            "gguf_q8_0",
            "hipengine_gguf_q8_0_dense_wide256_f32_f32_out",
            512,
            2560,
            10240,
        )
    ]


def test_the_f16_wmma_q8_0_prefill_kernel_records_into_the_census(monkeypatch):
    recorded = _recorded(monkeypatch)
    symbol = gguf_q8_0_prefill._symbol("wmma_prefill_f32_f32_out")

    gguf_q8_0_prefill._launch(
        symbol,
        1,
        2,
        3,
        512,
        2560,
        10240,
        library=_FakeLibrary(),
        runtime=_FakeRuntime(),
    )

    assert recorded == [("gguf_q8_0", symbol, 512, 2560, 10240)]


def test_the_census_stays_off_unless_it_is_asked_for(monkeypatch):
    monkeypatch.delenv(launch_census.ENV_VAR, raising=False)
    monkeypatch.setattr(launch_census, "_state", None, raising=False)
    launch_census.reset()

    assert launch_census.enabled() is False
    launch_census.record_launch("gguf_q8_0", "symbol", 512, 2560, 10240)
    assert launch_census.snapshot()["total_launches"] == 0
