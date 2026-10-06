"""DP-2 reuse is an ordered immediate pair, never a persistent input cache."""
from contextvars import Context, copy_context
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear

RT = object()
X, HALF, OA, OB, WA, WB = (i << 32 for i in range(1, 7))
ROWS, K, NA, NB = 1024, 2560, 10240, 6144
SOURCE = "dense_wide256_f32_f32_out"
TARGET = "dense_wide256_f16in_f32_f32_out"
KEY = KernelKey("hip_gfx1151", "linear", "gguf_q8_0", TARGET)
ENV = "HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR"


def weight(role, ptr, n, layer=24):
    return SimpleNamespace(backend=KEY.backend, spec=SimpleNamespace(
        layout="raw_gguf", quant_key=KEY.quant, slot_path=f"layers.{layer}.{role}",
        source=SimpleNamespace(shape=(n, K))), allocation=lambda _: SimpleNamespace(
            tensor=SimpleNamespace(ptr=ptr), buffer=SimpleNamespace(nbytes=n * (K // 32) * 34)))


@pytest.fixture
def fake_pair(monkeypatch):
    casts, calls = [], []
    a, b = weight("attn_qkv", WA, NA), weight("attn_gate", WB, NB)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", ",".join(map(str, range(13, 48))))
    monkeypatch.setattr(linear, "_wide_f16_consumer_key", lambda *a, **kw: KEY)
    monkeypatch.setattr(linear, "is_registered", lambda _: True)
    monkeypatch.setattr(linear, "resolve", lambda **key: lambda *a, **kw: calls.append((key, a, kw)))
    monkeypatch.setattr("hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide.f32_to_f16",
                        lambda *a, **kw: casts.append((a, kw)))

    def session(ptr=HALF, size=ROWS * K * 2):
        return linear.wide_f16_activation_session(workspace_ptr=ptr, workspace_nbytes=size)

    def pair(**kw):
        args = dict(rows=ROWS, in_features=K, out_features_a=NA, out_features_b=NB,
                    stream=7, runtime=RT)
        args.update(kw)
        return linear.wide_f16_activation_pair(a, b, X, OA, OB, **args)

    def launch(second=False, **kw):
        args = dict(weight=b if second else a, x_ptr=X, out_ptr=OB if second else OA,
                    rows=ROWS, in_features=K, out_features=NB if second else NA,
                    stream=7, runtime=RT, backend=KEY.backend, quant=KEY.quant,
                    layer=KEY.layer, variant=SOURCE)
        args.update(kw)
        stream = args.pop("stream")
        return linear._launch_dense_wide_f16_staged(None, kwargs={"stream": stream}, **args)

    return SimpleNamespace(a=a, b=b, session=session, pair=pair, launch=launch, casts=casts, calls=calls)


def test_exact_pair_casts_once_then_is_consumed(fake_pair):
    p = fake_pair
    with p.session(), p.pair() as loan:
        assert loan is not None
        assert p.launch() and p.launch(True)
        assert len(p.casts) == 1
        assert loan.accepted == 2
        assert all(call[1][0] == HALF for call in p.calls)
        assert all(call[2]["stream"] == 7 for call in p.calls)
        assert p.launch() and p.launch(True)
        assert len(p.casts) == 3
    assert p.launch() is False


@pytest.mark.parametrize("change", [
    {"x_ptr": X + 4}, {"out_ptr": OB + 4}, {"rows": ROWS - 1}, {"in_features": K - 64},
    {"out_features": NB - 256}, {"stream": 8}, {"runtime": object()},
    {"backend": "hip_gfx1100"}, {"variant": "dense_wide128x128_f32_f32_out"},
])
def test_second_mismatch_invalidates_and_cannot_restore_pair(fake_pair, change):
    p = fake_pair
    with p.session(), p.pair() as loan:
        assert p.launch()
        assert p.launch(True, **change)
        assert p.launch(True)
        assert len(p.casts) == 3
        assert loan.accepted == 1


def test_same_key_different_weight_and_wrong_order_invalidate(fake_pair):
    p = fake_pair
    with p.session(), p.pair() as loan:
        assert p.launch(True)
        assert p.launch() and p.launch(True)
        assert len(p.casts) == 3 and loan.accepted == 0
    with p.session(), p.pair() as loan:
        assert p.launch()
        replacement = weight("attn_gate", WB, NB)
        assert p.launch(True, weight=replacement)
        assert p.launch(True)
        assert len(p.casts) == 6 and loan.accepted == 1


def test_nested_pair_session_and_publication_never_restore_outer_loan(fake_pair):
    p = fake_pair
    with p.session(), p.pair() as outer:
        assert p.launch()
        with p.pair() as inner:
            assert p.launch() and p.launch(True)
            assert inner.accepted == 2
        assert p.launch(True)
        assert outer.accepted == 1 and len(p.casts) == 3
    with p.session(), p.pair():
        assert p.launch()
        with p.session():
            pass
        assert p.launch(True)
        assert len(p.casts) == 5
    with p.session(), p.pair():
        assert p.launch()
        with linear.wide_f16_activation_publication(X, ROWS * K, stream=7, runtime=RT, consumer_key=KEY):
            pass
        assert p.launch(True)
        assert len(p.casts) == 7


def test_exception_and_context_exit_invalidate(fake_pair):
    p = fake_pair
    with p.session():
        with pytest.raises(RuntimeError):
            with p.pair() as loan:
                assert p.launch()
                raise RuntimeError("producer interruption")
        assert p.launch(True)
        assert len(p.casts) == 2 and not loan.valid
        with p.pair():
            assert p.launch()
        assert p.launch(True)
        assert len(p.casts) == 4


def test_copied_context_cannot_accept_overwrite_or_open_parent_pair(fake_pair):
    p = fake_pair
    with p.session(), p.pair() as loan:
        assert p.launch()
        copied = copy_context()
        assert copied.run(p.launch, True) is False
        with pytest.raises(ValueError, match="owning context"):
            copied.run(lambda: p.pair().__enter__())
        with pytest.raises(ValueError, match="owning context"):
            copied.run(lambda: linear.wide_f16_activation_publication(
                X, ROWS * K, stream=7, runtime=RT, consumer_key=KEY).__enter__())
        with pytest.raises(ValueError, match="owning context"):
            copied.run(lambda: p.session().__enter__())
        assert Context().run(p.launch, True) is False
        assert len(p.casts) == 1
        assert p.launch(True) and loan.accepted == 2


@pytest.mark.parametrize("rows", [1, 512, 1023, 1025, 2048, 4095, 4097])
def test_unmeasured_rows_not_admitted(fake_pair, rows):
    p = fake_pair
    with p.session(size=4097 * K * 2), p.pair(rows=rows) as loan:
        assert loan is None


@pytest.mark.parametrize("layer", [0, 12, 48])
def test_layers_outside_promoted_dense_scope_not_admitted(fake_pair, layer):
    p = fake_pair
    p.a.spec.slot_path = f"layers.{layer}.attn_qkv"
    p.b.spec.slot_path = f"layers.{layer}.attn_gate"
    with p.session(), p.pair() as loan:
        assert loan is None


def test_disabled_small_workspace_and_competing_key_keep_parent(fake_pair, monkeypatch):
    p = fake_pair
    for ptr, size in [(HALF, ROWS * K * 2 - 1)]:
        with p.session(ptr, size), p.pair() as loan:
            assert loan is None
    monkeypatch.setenv(ENV, "0")
    with p.session(), p.pair() as loan:
        assert loan is None
        assert p.launch() and p.launch(True)
        assert len(p.casts) == 2
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(linear, "_wide_f16_consumer_key", lambda *a, **kw: None)
    with p.session(), p.pair() as loan:
        assert loan is None


@pytest.mark.parametrize("alias", [X, OA, OB, WA, WB, (1 << 64) - 1])
def test_unsafe_workspace_shadows_staging_during_parent_fallback(fake_pair, alias):
    p = fake_pair
    with p.session(ptr=alias):
        with p.pair() as loan:
            assert loan is None
            assert p.launch() is False and p.launch(True) is False
        assert linear.wide_f16_activation_workspace().ptr == alias
    assert not p.casts


def test_pair_does_not_expand_single_use_dp5_publication(fake_pair):
    p = fake_pair
    with p.session():
        with linear.wide_f16_activation_publication(X, ROWS * K, stream=7, runtime=RT, consumer_key=KEY):
            with p.pair() as loan:
                assert p.launch() and p.launch(True)
                assert loan.accepted == 2 and len(p.casts) == 1
            assert p.launch(True)
            assert len(p.casts) == 2


def test_failed_consumer_launch_never_commits_or_restores_pair(fake_pair, monkeypatch):
    p = fake_pair
    with p.session(), p.pair() as loan:
        monkeypatch.setattr(linear, "resolve", lambda **kw: lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("launch")))
        with pytest.raises(RuntimeError, match="launch"):
            p.launch()
        assert not loan.valid and loan.accepted == 0
        monkeypatch.setattr(linear, "resolve", lambda **kw: lambda *a, **kw: None)
        assert p.launch(True)
        assert len(p.casts) == 2


def test_native_dispatch_probe_and_backend_policy_admit_only_measured_pairs(monkeypatch):
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    register_gfx1151_kernels(replace=True)
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(linear.WIDE_F16_ACTIVATION_ENV, "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "1")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS", ",".join(map(str, range(13, 48))))
    monkeypatch.delenv("HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS", raising=False)
    a, b = weight("attn_qkv", WA, NA), weight("attn_gate", WB, NB)
    with linear.raw_k_prefill_variant_session("coltile8"), linear.wide_f16_activation_session(
            workspace_ptr=HALF, workspace_nbytes=4096 * K * 2):
        for rows in (1024, 4096):
            with linear.wide_f16_activation_pair(a, b, X, OA, OB, rows=rows,
                    in_features=K, out_features_a=NA, out_features_b=NB, runtime=RT) as loan:
                assert loan is not None
                assert [c.key for c in loan.consumers] == [KEY, KEY]
        monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
        with linear.wide_f16_activation_pair(a, b, X, OA, OB, rows=ROWS,
                in_features=K, out_features_a=NA, out_features_b=NB, runtime=RT) as loan:
            assert loan is None


def test_runner_pair_helper_preserves_dispatch_order_and_off_path(monkeypatch):
    from hipengine.runtime import qwen4_exp_runner as runner
    calls = []
    a, b = object(), object()
    monkeypatch.setenv(ENV, "0")
    monkeypatch.setattr(runner, "wide_f16_activation_pair", lambda *a, **k: pytest.fail("off route opens scope"))
    monkeypatch.setattr(runner, "launch_gguf_linear", lambda *a, **k: calls.append((a, k)))
    runner._launch_qwen4_exp_dense_pair(a, b, X, OA, OB, ROWS, K, NA, NB, stream=7, runtime=RT)
    assert [c[0] for c in calls] == [(a, X, OA, ROWS, K, NA), (b, X, OB, ROWS, K, NB)]
    assert all(c[1] == dict(activation_dtype="f32", output_dtype="f32", stream=7, runtime=RT) for c in calls)
