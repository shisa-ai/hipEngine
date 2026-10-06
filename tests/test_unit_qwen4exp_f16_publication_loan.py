"""Single-use DP-5 side publication cannot become an activation cache."""
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


def publish(**kw):
    return linear.wide_f16_activation_publication(X, COUNT, stream=7, runtime=RT,
                                                 consumer_key=KEY, **kw)


@pytest.fixture
def fake_launch(monkeypatch):
    casts, calls = [], []
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
    monkeypatch.setenv(linear.DENSE_WIDE_TAIL_VARIANT_ENV, TARGET)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    monkeypatch.setattr(linear, "resolve", lambda **key: lambda *a, **k: calls.append((key, a, k)))
    monkeypatch.setattr("hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
                        lambda *a, **k: casts.append((a, k)))
    weight = SimpleNamespace(allocation=lambda _: SimpleNamespace(tensor=SimpleNamespace(ptr=0x400000000)))
    def launch(**kw):
        args = dict(x_ptr=X, rows=ROWS, in_features=K, out_features=N, stream=7, runtime=RT,
                    variant=SOURCE, backend=KEY.backend)
        args.update(kw)
        stream = args.pop("stream")
        return linear._launch_dense_wide_f16_staged(None, weight, out_ptr=OUT,
            kwargs={"stream": stream}, quant=KEY.quant, layer=KEY.layer, **args)
    return launch, casts, calls


def test_accepts_exact_consumer_and_uses_variant_scoped_library(fake_launch):
    launch, casts, calls = fake_launch
    library = object()
    with session(), publish():
        assert launch(libraries={f"gguf_q8_0:{TARGET}": library})
        assert casts == []
        assert calls[0][0]["variant"] == TARGET
        assert calls[0][1][0] == HALF
        assert calls[0][2]["library"] is library
        assert calls[0][2]["stream"] == 7
        assert launch()
        assert len(casts) == 1  # never reuse a consumed loan


@pytest.mark.parametrize("change", [
    {"x_ptr": X + 4}, {"rows": ROWS - 1}, {"in_features": K - 64},
    {"stream": 8}, {"runtime": object()},
    {"variant": "dense_wide128x128_f32_f32_out"}, {"backend": "hip_gfx1100"},
])
def test_consumer_mismatch_casts_then_cannot_restore_loan(fake_launch, change):
    launch, casts, calls = fake_launch
    with session(), publish():
        assert launch(**change)
        assert len(casts) == 1
        assert launch()
        assert len(casts) == 2


def test_failed_registration_or_disabled_launch_invalidates(fake_launch, monkeypatch):
    launch, casts, calls = fake_launch
    with session(), publish():
        monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "0")
        assert not launch()
        monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
        assert launch()
        assert len(casts) == 1
    with session(), publish():
        monkeypatch.setattr(linear, "is_registered", lambda _: False)
        monkeypatch.setattr(linear, "_ensure_linear_kernel_registered", lambda _: None)
        assert not launch()
        monkeypatch.setattr(linear, "is_registered", lambda _: True)
        assert launch()
        assert len(casts) == 2


def test_nested_session_cannot_restore_outer_publication(fake_launch):
    launch, casts, calls = fake_launch
    with session(), publish():
        with session():  # even equal workspace values are a distinct owner
            assert launch()
        assert launch()
        assert len(casts) == 2


def test_independent_context_has_no_loan(fake_launch):
    launch, casts, calls = fake_launch
    with session(), publish():
        assert Context().run(launch) is False
        assert casts == []
        assert launch()
        assert casts == []


def test_context_copy_cannot_accept_parent_loan():
    from contextvars import copy_context
    # ContextVar snapshots copy object references. A copied context must not
    # acquire a parent publication by matching all its pointer-valued fields.
    with session(), publish():
        copied = copy_context()
        assert copied.run(linear._consume_wide_f16_publication, X, COUNT,
                          stream=7, runtime=RT, consumer_key=KEY) == 0


@pytest.mark.parametrize("count,ptr", [(0, X), (-1, X), (COUNT + 1, X), (COUNT, HALF),
                                      (COUNT, (1 << 64) - 1)])
def test_bad_bounds_refuse_before_producer(count, ptr):
    with session(), pytest.raises(ValueError):
        with linear.wide_f16_activation_publication(ptr, count, stream=7, runtime=RT,
                                                    consumer_key=KEY):
            pass
