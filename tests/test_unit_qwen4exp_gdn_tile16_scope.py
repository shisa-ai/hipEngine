"""The tile-16 GDN prefill owner's geometry guards and scoped parent control.

The owner is selected for the binding Hk16/Hv48/D128 envelope on any row count
from 16 up, and the columnwarp parent serves everything else. That decision is
request-local runner state: the owning runner carries it, a diagnostic arm binds
it on the runner it measures, and no process-wide switch can change it. The
serial strict route stays registered below both.
"""
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL"


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.delenv(RETIRED_ENV, raising=False)


def select(**kwargs):
    return runner.qwen4_exp_gdn_tile16_prefill_selected(**kwargs)


def test_geometry_envelope_selects_the_tile16_owner():
    assert select(rows=16, num_k_heads=16, num_v_heads=48, head_dim=128) is True
    assert select(rows=512, num_k_heads=16, num_v_heads=48, head_dim=128) is True
    assert select(rows=512, num_k_heads=16, num_v_heads=32, head_dim=128) is True
    assert select(rows=4096, num_k_heads=16, num_v_heads=32, head_dim=128) is True


def test_outside_the_envelope_keeps_the_columnwarp_parent():
    assert select(rows=15, num_k_heads=16, num_v_heads=48, head_dim=128) is False
    assert select(rows=1, num_k_heads=16, num_v_heads=48, head_dim=128) is False
    assert select(rows=512, num_k_heads=8, num_v_heads=48, head_dim=128) is False
    assert select(rows=512, num_k_heads=16, num_v_heads=40, head_dim=128) is False
    assert select(rows=512, num_k_heads=16, num_v_heads=48, head_dim=64) is False


def test_retired_process_switch_cannot_change_the_geometry_decision(monkeypatch):
    """The leave-one-out closure retired the switch; only the runner carries it."""

    monkeypatch.setenv(RETIRED_ENV, "0")
    assert select(rows=512, num_k_heads=16, num_v_heads=48, head_dim=128) is True
    assert select(rows=15, num_k_heads=16, num_v_heads=48, head_dim=128) is False
    monkeypatch.setenv(RETIRED_ENV, "1")
    assert select(rows=512, num_k_heads=16, num_v_heads=48, head_dim=128) is True
    assert select(rows=15, num_k_heads=16, num_v_heads=48, head_dim=128) is False


def admitted(rows, *, supports_any_rows, enabled=True):
    return runner.qwen4_exp_gdn_tile16_owner_admitted(
        rows=rows,
        num_k_heads=16,
        num_v_heads=48,
        head_dim=128,
        supports_any_rows=supports_any_rows,
        enabled=enabled,
    )


@pytest.mark.parametrize("rows", [2, 4, 15, 16, 17, 512])
def test_row_split_owner_takes_every_width_from_two_rows(rows):
    assert admitted(rows, supports_any_rows=True) is True
    assert admitted(rows=1, supports_any_rows=True) is False


@pytest.mark.parametrize("rows", [2, 4, 15])
def test_tile16_bodies_need_the_envelope_row_bound(rows):
    assert admitted(rows, supports_any_rows=False) is False


@pytest.mark.parametrize("rows", [16, 17, 512])
def test_tile16_bodies_own_the_envelope_from_sixteen_rows(rows):
    assert admitted(rows, supports_any_rows=False) is True


@pytest.mark.parametrize("rows", [2, 4, 16, 17, 512])
@pytest.mark.parametrize("supports_any_rows", [False, True])
def test_the_parent_arm_owns_every_width_and_family(rows, supports_any_rows):
    """A disabled arm leaves even the row-split widths to the columnwarp parent."""

    assert admitted(rows, supports_any_rows=supports_any_rows, enabled=False) is False


@pytest.mark.parametrize(
    ("rows", "k_heads", "v_heads", "head_dim", "supports_any_rows"),
    [
        (512, 8, 48, 128, False),
        (512, 16, 40, 128, False),
        (512, 16, 48, 64, False),
        (512, 16, 48, 64, True),
    ],
)
def test_mismatch_shapes_stay_on_the_parent(
    rows, k_heads, v_heads, head_dim, supports_any_rows
):
    assert (
        runner.qwen4_exp_gdn_tile16_owner_admitted(
            rows=rows,
            num_k_heads=k_heads,
            num_v_heads=v_heads,
            head_dim=head_dim,
            supports_any_rows=supports_any_rows,
        )
        is False
    )


def test_row_split_owner_keeps_its_head_dim_guard_only():
    """The row-split body's own contract is a 128x128 head, not the tile-16
    Hv32/Hv48 envelope; the columnwarp parent serves other head widths."""

    assert admitted(512, supports_any_rows=True) is True
    assert (
        runner.qwen4_exp_gdn_tile16_owner_admitted(
            rows=512,
            num_k_heads=16,
            num_v_heads=40,
            head_dim=128,
            supports_any_rows=True,
        )
        is True
    )


def test_runner_defaults_to_the_tile16_owner():
    assert runner.Qwen4ExpGGUFResidentModelRunner.gdn_tile16_prefill is True


def test_scope_binds_the_parent_and_restores_the_owner():
    owner = SimpleNamespace(gdn_tile16_prefill=True)
    with runner.gdn_tile16_prefill(owner, False) as prior:
        assert prior is True
        assert owner.gdn_tile16_prefill is False
    assert owner.gdn_tile16_prefill is True


def test_scope_leaves_an_unset_binding_alone():
    owner = SimpleNamespace(gdn_tile16_prefill=True)
    with runner.gdn_tile16_prefill(owner):
        assert owner.gdn_tile16_prefill is True
    assert owner.gdn_tile16_prefill is True


def test_scope_restores_after_an_exception():
    owner = SimpleNamespace(gdn_tile16_prefill=True)
    with pytest.raises(RuntimeError, match="boom"):
        with runner.gdn_tile16_prefill(owner, False):
            raise RuntimeError("boom")
    assert owner.gdn_tile16_prefill is True


def test_scope_is_per_runner():
    first = SimpleNamespace(gdn_tile16_prefill=True)
    second = SimpleNamespace(gdn_tile16_prefill=True)
    with runner.gdn_tile16_prefill(first, False):
        assert second.gdn_tile16_prefill is True
        with runner.gdn_tile16_prefill(second, False):
            assert first.gdn_tile16_prefill is False
            assert second.gdn_tile16_prefill is False
        assert second.gdn_tile16_prefill is True
    assert first.gdn_tile16_prefill is True


def test_a_serving_clone_inherits_the_default_not_a_diagnostic_binding():
    """The arm is per-runner: a clone constructed inside it keeps the owner."""

    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(gdn_prefill_variant=None)
    measured = SimpleNamespace(gdn_tile16_prefill=True)
    with runner.gdn_tile16_prefill(measured, False):
        clone = _inherit_lane_decisions(generator, SimpleNamespace())
        assert getattr(clone, "gdn_tile16_prefill", True) is True
        assert measured.gdn_tile16_prefill is False
    assert measured.gdn_tile16_prefill is True


@pytest.mark.parametrize("rows", [2, 4, 16, 512])
def test_layer_forwards_the_decision_to_the_mixer(monkeypatch, rows):
    """The layer's only mixer call site carries the runner's decision."""

    seen = []
    runtime = object()

    def gr_read(*args, **kwargs):
        return SimpleNamespace(normalized=None, gate=None,
                               mixed=SimpleNamespace(ptr=11),
                               inject_logits=SimpleNamespace(ptr=12))

    def mixer(*args, **kwargs):
        seen.append(kwargs.get("gdn_tile16_prefill", "absent"))
        return SimpleNamespace(ptr=13)

    def moe(*args, **kwargs):
        return SimpleNamespace(output=SimpleNamespace(ptr=14))

    monkeypatch.setattr(runner, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runner, "run_qwen4_exp_gdn_token_mixer", mixer)
    monkeypatch.setattr(runner, "run_qwen4_exp_moe", moe)
    monkeypatch.setattr(runner, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runner, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
    output = SimpleNamespace(ptr=15)
    scratch = SimpleNamespace(
        closed=False, runtime=runtime, attention_gr=SimpleNamespace(),
        gdn=SimpleNamespace(), after_attention=SimpleNamespace(ptr=16),
        ffn_gr=SimpleNamespace(), moe=SimpleNamespace(), output=output)
    weights = SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(projections={}, conv_weight_ptr=5, dt_bias_ptr=6,
                              a_ptr=7, norm_weight_ptr=8),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())
    kwargs = dict(conv_state_ptr=0, recurrent_state_ptr=0, scratch=scratch,
                  rows=rows, branches=1, hidden=2, low_rank=2, num_k_heads=16,
                  num_v_heads=48, head_dim=128, conv_kernel=4, ffn=8,
                  experts=1, top_k=1, runtime=runtime)
    assert runner.run_qwen4_exp_gdn_layer(
        100, weights, **kwargs, gdn_tile16_prefill=False) is output
    assert runner.run_qwen4_exp_gdn_layer(100, weights, **kwargs) is output
    assert seen == [False, True]
