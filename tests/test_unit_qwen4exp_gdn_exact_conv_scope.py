"""Qwen4Exp GDN conv prefill ownership is request-local state.

``HIPENGINE_QWEN4_EXP_EXACT_CONV_PREFILL`` used to be the only way to reach the
per-row conv fallback, and it was a process-wide switch. The decision is now the
owning runner's ``gdn_exact_bulk_conv`` field, threaded through the GDN layer to
the token mixer:

- ``True``  - the exact bulk conv owner (``linear_attn_conv_prefill`` /
  ``qwen4_exp`` / ``f32_decode_exact_k4``), production for every chunk wider
  than one row, with optional state-row capture.
- ``False`` - the serial diagnostic fallback: one decode-conv launch per row.
  It cannot publish state rows, so a capture request with it pinned fails
  loudly instead of running the loop.

Decode (``rows == 1``) is unaffected: it always runs the decode conv kernel.
These tests pin the field, the scoped control, the forwarding chain, the
surviving fallback, and that no process switch is read.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import MappingProxyType, SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner_module

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_EXACT_CONV_PREFILL"


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


def test_runner_defaults_to_the_exact_bulk_owner() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert runner.gdn_exact_bulk_conv is True


def test_scope_binds_the_serial_fallback_and_restores() -> None:
    runner = SimpleNamespace(gdn_exact_bulk_conv=True)

    with runner_module.gdn_exact_bulk_conv(runner, False) as prior:
        assert prior is True
        assert runner.gdn_exact_bulk_conv is False
        with runner_module.gdn_exact_bulk_conv(runner, True):
            assert runner.gdn_exact_bulk_conv is True
        assert runner.gdn_exact_bulk_conv is False
    assert runner.gdn_exact_bulk_conv is True


def test_scope_leaves_an_unset_binding_alone() -> None:
    runner = SimpleNamespace(gdn_exact_bulk_conv=True)

    with runner_module.gdn_exact_bulk_conv(runner):
        assert runner.gdn_exact_bulk_conv is True
    assert runner.gdn_exact_bulk_conv is True


def test_scope_restores_after_an_exception() -> None:
    runner = SimpleNamespace(gdn_exact_bulk_conv=True)

    with pytest.raises(RuntimeError, match="boom"):
        with runner_module.gdn_exact_bulk_conv(runner, False):
            raise RuntimeError("boom")
    assert runner.gdn_exact_bulk_conv is True


def test_scope_is_per_runner() -> None:
    first = SimpleNamespace(gdn_exact_bulk_conv=True)
    second = SimpleNamespace(gdn_exact_bulk_conv=True)

    with runner_module.gdn_exact_bulk_conv(first, False):
        assert second.gdn_exact_bulk_conv is True
        with runner_module.gdn_exact_bulk_conv(second, False):
            assert first.gdn_exact_bulk_conv is False
            assert second.gdn_exact_bulk_conv is False
        assert second.gdn_exact_bulk_conv is True
    assert first.gdn_exact_bulk_conv is True


def _layer(monkeypatch: pytest.MonkeyPatch, recorded: dict[str, object], **extra):
    runtime = object()

    def gr_read(*args, **kwargs):
        return SimpleNamespace(normalized=None, gate=None,
                               mixed=SimpleNamespace(ptr=11),
                               inject_logits=SimpleNamespace(ptr=12))

    def mixer(*args, **kwargs):
        recorded["gdn_exact_bulk_conv"] = kwargs.get(
            "gdn_exact_bulk_conv", "<absent>")
        return SimpleNamespace(ptr=13)

    monkeypatch.setattr(runner_module, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_gdn_token_mixer", mixer)
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_moe",
        lambda *a, **k: SimpleNamespace(output=SimpleNamespace(ptr=14)),
    )
    monkeypatch.setattr(runner_module, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(runner_module, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    monkeypatch.setattr(runner_module, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(runner_module, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
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
    runner_module.run_qwen4_exp_gdn_layer(
        100, weights,
        conv_state_ptr=0, recurrent_state_ptr=0, scratch=scratch,
        rows=512, branches=1, hidden=2, low_rank=2, num_k_heads=16,
        num_v_heads=48, head_dim=128, conv_kernel=4, ffn=8,
        experts=1, top_k=1, runtime=runtime,
        **extra,
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_layer_forwards_the_decision_to_the_mixer(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    recorded: dict[str, object] = {}

    _layer(monkeypatch, recorded, gdn_exact_bulk_conv=enabled)

    assert recorded["gdn_exact_bulk_conv"] is enabled


def test_layer_defaults_to_the_exact_bulk_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    _layer(monkeypatch, recorded)

    assert recorded["gdn_exact_bulk_conv"] is True


def test_prefill_chunk_forwards_the_runner_field() -> None:
    source = textwrap.dedent(
        inspect.getsource(
            runner_module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps
        )
    )
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_qwen4_exp_gdn_layer"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    assert isinstance(keywords["gdn_exact_bulk_conv"], ast.Attribute)
    assert keywords["gdn_exact_bulk_conv"].attr == "gdn_exact_bulk_conv"


def test_retired_process_switch_is_not_read() -> None:
    mixer = textwrap.dedent(
        inspect.getsource(runner_module.run_qwen4_exp_gdn_token_mixer)
    )
    assert RETIRED_ENV not in mixer
    assert RETIRED_ENV not in inspect.getsource(runner_module)


def test_serial_fallback_and_exact_owner_survive() -> None:
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(runner_module.run_qwen4_exp_gdn_token_mixer)
        )
    )

    # One decode-conv launch per row stays reachable.
    assert any(
        isinstance(node, ast.For)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "qwen35_linear_attn_conv_decode_f32"
            for child in ast.walk(node)
        )
        for node in ast.walk(tree)
    )
    # The exact bulk owner is still resolved by name.
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve"
        and any(
            keyword.arg == "variant"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "f32_decode_exact_k4"
            for keyword in node.keywords
        )
        for node in ast.walk(tree)
    )


def test_state_row_capture_refuses_the_serial_fallback() -> None:
    """A verifier chunk needs the exact owner; the loop cannot publish rows."""

    runtime = object()
    scratch = SimpleNamespace(
        closed=False, runtime=runtime,
        qkv=SimpleNamespace(ptr=1), gate=SimpleNamespace(ptr=2),
        conv=SimpleNamespace(ptr=3), alpha=SimpleNamespace(ptr=4),
        beta=SimpleNamespace(ptr=5), core=SimpleNamespace(ptr=6),
        output=SimpleNamespace(ptr=7),
    )
    weights = {
        name: SimpleNamespace(backend="hip_gfx1100")
        for name in ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out")
    }
    token = runner_module._GDN_STATE_ROWS_CAPTURE.set(
        MappingProxyType({0: (100, 200)})
    )
    try:
        with pytest.raises(RuntimeError, match="exact bulk conv"):
            runner_module.run_qwen4_exp_gdn_token_mixer(
                10, weights,
                conv_weight_ptr=20, dt_bias_ptr=21, a_ptr=22, norm_weight_ptr=23,
                conv_state_ptr=30, recurrent_state_ptr=0, scratch=scratch,
                rows=4, hidden=2, num_k_heads=16, num_v_heads=48, head_dim=128,
                conv_kernel=4, runtime=runtime,
                gdn_exact_bulk_conv=False,
            )
    finally:
        runner_module._GDN_STATE_ROWS_CAPTURE.reset(token)
