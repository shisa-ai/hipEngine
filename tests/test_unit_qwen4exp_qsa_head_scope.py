"""Qwen4Exp H256 QSA prefill head grouping is request-local state.

``HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR`` used to be the only way to select the
two- or four-head prefill leaf, and it was a process-wide switch. The binder now
publishes the decision to the owning runner and the prefill path threads it as
an explicit argument:

- ``"quad"``  - production lanes (UD and the fast route that shares its binder).
- ``"parent"`` - strict, plans that fall back to the strict binder, unbound
  runners, and anything unknown.
- ``"pair"``  - explicit diagnostics only; no plan publishes it.

This selects the **scalar** sparse-prefill rows owner. It is resolved on every
production chunk but launched only when WMMA attention does not own that chunk:
the production profiles also publish ``runner.qsa_wmma_enabled=True``, and the
resolved manifest names ``wmma_f16_masked_prefill_f32`` for the same scope. A
launch-counting probe on the default route therefore sees no head-grouped leaf;
that is the fallback selection, not a regression.

These tests pin the selector's table and guards, the forwarding through the
prefill layer, the scoped diagnostic control, cloned-runner inheritance, the
independent batch head-pair shape owner, and the ``cache_only`` seam that never
reaches head selection.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import qwen4_exp_runner as runner_module
from hipengine.runtime import qwen4_exp_mtp_prefill as mtp_module
from hipengine.runtime.qwen4_exp_batch_qsa import batch_qsa_attention_implementation

_PAGE256 = "strict_h256_page256_wave_rows_spans"
_PAIR = "strict_h256_head_pair_rows_spans"
_QUAD = "strict_h256_head_quad_rows_spans"


def _page256_key() -> KernelKey:
    return KernelKey("hip_gfx1151", "qsa_sparse_attention", "bf16_kv", _PAGE256)


@pytest.fixture
def registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "is_registered", lambda key: True)


@pytest.mark.parametrize(
    "mode,expected",
    [
        pytest.param("parent", _PAGE256, id="parent"),
        pytest.param("quad", _QUAD, id="quad"),
        pytest.param("pair", _PAIR, id="pair"),
        # Fail closed to the parent: a typo must not silently measure a leaf.
        pytest.param("1", _PAGE256, id="legacy-pair-spelling"),
        pytest.param("0", _PAGE256, id="legacy-off-spelling"),
        pytest.param("invalid", _PAGE256, id="invalid"),
        pytest.param("", _PAGE256, id="empty"),
        pytest.param(None, _PAGE256, id="none"),
    ],
)
def test_head_mode_table(registered, mode, expected) -> None:
    key = runner_module._qwen4_exp_qsa_prefill_head_key(
        _page256_key(), query_heads=24, kv_heads=2, mode=mode
    )
    assert key.variant == expected
    assert (key.backend, key.layer, key.quant) == (
        "hip_gfx1151",
        "qsa_sparse_attention",
        "bf16_kv",
    )


@pytest.mark.parametrize("mode", ["quad", "pair"])
@pytest.mark.parametrize(
    "variant,query_heads,kv_heads",
    [
        pytest.param("strict_h256_wave_rows_spans", 24, 2, id="non-page256-base"),
        pytest.param(_PAGE256, 6, 2, id="six-query-heads"),
        pytest.param(_PAGE256, 12, 2, id="twelve-query-heads"),
        pytest.param(_PAGE256, 24, 4, id="four-kv-heads"),
    ],
)
def test_head_mode_keeps_the_page256_hq24_hkv2_guard(
    registered, mode, variant, query_heads, kv_heads
) -> None:
    key = KernelKey("hip_gfx1151", "qsa_sparse_attention", "bf16_kv", variant)
    selected = runner_module._qwen4_exp_qsa_prefill_head_key(
        key, query_heads=query_heads, kv_heads=kv_heads, mode=mode
    )
    assert selected is key


@pytest.mark.parametrize("mode", ["quad", "pair"])
def test_head_mode_falls_back_to_the_parent_when_unregistered(
    monkeypatch: pytest.MonkeyPatch, mode
) -> None:
    monkeypatch.setattr(runner_module, "is_registered", lambda key: False)
    key = _page256_key()
    assert (
        runner_module._qwen4_exp_qsa_prefill_head_key(
            key, query_heads=24, kv_heads=2, mode=mode
        )
        is key
    )


def test_runner_field_defaults_to_the_parent() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert runner.qsa_prefill_head_mode == "parent"


def _layer_stubs(monkeypatch: pytest.MonkeyPatch, recorded: dict[str, object]):
    runtime = SimpleNamespace(name="runtime")
    scratch = SimpleNamespace(
        closed=False,
        runtime=runtime,
        attention_gr=object(),
        qsa=object(),
        after_attention=SimpleNamespace(ptr=11),
        ffn_gr=object(),
        moe=object(),
        output=SimpleNamespace(ptr=12),
    )

    def gr_read(*args, **kwargs):
        return SimpleNamespace(
            mixed=SimpleNamespace(ptr=21), inject_logits=SimpleNamespace(ptr=22)
        )

    def mixer(*args, **kwargs):
        recorded["head_mode"] = kwargs.get("head_mode", "<absent>")
        recorded["cache_only"] = kwargs.get("cache_only", False)
        return SimpleNamespace(ptr=31)

    monkeypatch.setattr(runner_module, "run_qwen4_exp_gr_read", gr_read)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_qsa_prefill_token_mixer", mixer)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_moe",
        lambda *a, **k: SimpleNamespace(output=SimpleNamespace(ptr=41)),
    )
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runner_module, "_qwen4_exp_gr_defer_write", lambda *a, **k: False
    )
    weights = SimpleNamespace(
        attention_gr=SimpleNamespace(
            norm_weight_ptr=1, down={}, up={}, inject={}
        ),
        ffn_gr=SimpleNamespace(norm_weight_ptr=2, down={}, up={}, inject={}),
        mixer=object(),
        moe=object(),
    )
    return scratch, runtime, weights


@pytest.mark.parametrize("mode", ["parent", "pair", "quad"])
def test_prefill_layer_forwards_the_head_mode_to_the_mixer(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    recorded: dict[str, object] = {}
    scratch, runtime, weights = _layer_stubs(monkeypatch, recorded)

    runner_module.run_qwen4_exp_qsa_prefill_layer(
        5,
        weights,
        attention_state=object(),
        index_state=object(),
        scratch=scratch,
        metadata=object(),
        start_position=0,
        rows=4,
        branches=2,
        hidden=64,
        low_rank=8,
        query_heads=24,
        kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        theta=10000.0,
        index_heads=4,
        index_dim=128,
        index_rotary_dim=64,
        ffn=32,
        experts=8,
        top_k=2,
        runtime=runtime,
        head_mode=mode,
    )

    assert recorded["head_mode"] == mode
    assert recorded["cache_only"] is False


def test_prefill_layer_defaults_to_the_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, object] = {}
    scratch, runtime, weights = _layer_stubs(monkeypatch, recorded)

    runner_module.run_qwen4_exp_qsa_prefill_layer(
        5,
        weights,
        attention_state=object(),
        index_state=object(),
        scratch=scratch,
        metadata=object(),
        start_position=0,
        rows=4,
        branches=2,
        hidden=64,
        low_rank=8,
        query_heads=24,
        kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        theta=10000.0,
        index_heads=4,
        index_dim=128,
        index_rotary_dim=64,
        ffn=32,
        experts=8,
        top_k=2,
        runtime=runtime,
    )

    assert recorded["head_mode"] == "parent"


def test_scoped_diagnostic_control_restores_the_request_decision() -> None:
    runner = SimpleNamespace(qsa_prefill_head_mode="quad")
    scope = runner_module.qsa_prefill_head_mode

    with scope(runner, "pair") as prior:
        assert prior == "quad"
        assert runner.qsa_prefill_head_mode == "pair"
        with scope(runner, "parent"):
            assert runner.qsa_prefill_head_mode == "parent"
        assert runner.qsa_prefill_head_mode == "pair"
    assert runner.qsa_prefill_head_mode == "quad"

    with pytest.raises(RuntimeError):
        with scope(runner, "pair"):
            raise RuntimeError("arm failed")
    assert runner.qsa_prefill_head_mode == "quad"

    # ``None`` leaves the request's own decision in force.
    with scope(runner, None):
        assert runner.qsa_prefill_head_mode == "quad"
    assert runner.qsa_prefill_head_mode == "quad"

    # A runner that never received a binder decision still restores to parent.
    unbound = SimpleNamespace()
    with scope(unbound, "quad"):
        assert unbound.qsa_prefill_head_mode == "quad"
    assert unbound.qsa_prefill_head_mode == "parent"


def test_cloned_runners_inherit_the_published_head_mode() -> None:
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(qsa_prefill_head_mode="quad")
    assert _inherit_lane_decisions(
        generator, SimpleNamespace()
    ).qsa_prefill_head_mode == "quad"

    generator.qsa_prefill_head_mode = "parent"
    assert _inherit_lane_decisions(
        generator, SimpleNamespace()
    ).qsa_prefill_head_mode == "parent"

    # An unbound generator keeps the parent route for anything cloned from it.
    assert _inherit_lane_decisions(
        SimpleNamespace(), SimpleNamespace()
    ).qsa_prefill_head_mode == "parent"


def test_batch_head_pair_selection_is_owned_by_shape() -> None:
    """Batch attention keeps the pair leaf independently of the prefill mode."""

    pair, pair_variant = batch_qsa_attention_implementation(
        head_dim=256, block_size=256, query_heads=24, kv_heads=2
    )
    assert pair_variant == _PAIR

    # Shape rule unchanged: an even GQA ratio at H256/page256 selects the pair,
    # everything else keeps the strict rows parent.
    assert batch_qsa_attention_implementation(
        head_dim=256, block_size=256, query_heads=12, kv_heads=2
    )[1] == _PAIR
    assert batch_qsa_attention_implementation(
        head_dim=256, block_size=256, query_heads=12, kv_heads=4
    )[1] == "strict_rows_spans"
    assert batch_qsa_attention_implementation(
        head_dim=128, block_size=256, query_heads=24, kv_heads=2
    )[1] == "strict_rows_spans"

    # The selector takes geometry only: no runner, session, or mode parameter.
    parameters = set(
        inspect.signature(batch_qsa_attention_implementation).parameters
    )
    assert parameters == {"head_dim", "block_size", "query_heads", "kv_heads"}


def test_cache_only_mixer_returns_before_head_selection() -> None:
    """The MTP cache-only path must never reach the head-variant dispatch."""

    tree = ast.parse(inspect.getsource(runner_module.run_qwen4_exp_qsa_prefill_token_mixer))
    early_return: int | None = None
    head_key_line: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "cache_only" and any(
                isinstance(child, ast.Return) for child in node.body
            ):
                early_return = node.body[0].lineno
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_qwen4_exp_qsa_prefill_head_key"
            and head_key_line is None
        ):
            head_key_line = node.lineno
    assert early_return is not None
    assert head_key_line is not None
    assert early_return < head_key_line


def test_mtp_prefill_call_site_keeps_the_default_mode() -> None:
    """The cache-only caller relies on the default rather than a mode."""

    source = inspect.getsource(mtp_module)
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_qwen4_exp_qsa_prefill_token_mixer"
    )
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert "head_mode" not in keywords
    assert isinstance(keywords["cache_only"], ast.Constant)
    assert keywords["cache_only"].value is True
