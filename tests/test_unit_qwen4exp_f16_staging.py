"""Ordinary wide staging casts per launch and preserves scoped libraries."""
from contextvars import Context
from types import SimpleNamespace

import pytest

from hipengine.runtime import gguf_linear as linear
from hipengine.kernels.registry import KernelKey

RT = object()
X, HALF, OUT = 0x100000000, 0x200000000, 0x300000000
ROWS, K, N = 4096, 10240, 320
COUNT = ROWS * K
TARGET = "dense_wide256_padded_f16in_f32_f32_out"
SOURCE = "dense_wide256_f32_f32_out"
KEY = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", TARGET)


def session(ptr=HALF, size=COUNT * 2):
    return linear.wide_f16_activation_session(workspace_ptr=ptr, workspace_nbytes=size)


@pytest.fixture
def fake_launch(monkeypatch):
    casts, calls = [], []
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    monkeypatch.setattr(linear, "resolve", lambda **key: lambda *a, **k: calls.append((key, a, k)))
    monkeypatch.setattr("hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
                        lambda *a, **k: casts.append((a, k)))
    weight = SimpleNamespace(allocation=lambda _: SimpleNamespace(tensor=SimpleNamespace(ptr=0x400000000)))
    def launch(**kw):
        return linear._launch_dense_wide_f16_staged(None, weight, x_ptr=X, out_ptr=OUT,
            rows=ROWS, in_features=K, out_features=N, kwargs={"stream": 7},
            quant=KEY.quant, layer=KEY.layer, variant=SOURCE, backend=KEY.backend,
            runtime=RT, **kw)
    with linear.dense_wide_tail_session(TARGET):
        yield launch, casts, calls


def test_staging_casts_each_time_and_uses_variant_scoped_library(fake_launch):
    launch, casts, calls = fake_launch
    library = object()
    with session():
        assert launch(libraries={f"gguf_q8_0:{TARGET}": library})
        assert len(casts) == 1
        assert calls[0][0]["variant"] == TARGET
        assert calls[0][1][0] == HALF
        assert calls[0][2]["library"] is library
        assert calls[0][2]["stream"] == 7
        assert launch()
        assert len(casts) == 2


def test_independent_context_has_no_workspace(fake_launch):
    launch, casts, calls = fake_launch
    with session():
        assert Context().run(launch) is False
        assert not casts
        assert launch()
        assert len(casts) == 1
