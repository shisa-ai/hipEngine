"""R0450: the ordered QSA decode prerequisite is owner state, not a switch.

``HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE`` was the process-wide transport for
one dispatch decision: whether the *scalar* c1 H256 indexed sparse decode takes
the ordered route at all. The migration target is an explicit owner boolean,
``qsa_ordered_decode``, published by the profile binder, bound by a scoped
control, and read by the c1 decode mixer.

The routes this file pins, in the order the dispatch evaluates them:

- owner off                        -> the non-ordered sparse owner
  (``..._paged_bf16_f32``, or the H128 wave32 owner under ``qsa_wave32``);
- owner on, H256, 1..4096 selected, v2 leaf unpublished
  -> the ordered-v1 parent ``..._paged_bf16_ordered_f32``;
- the same, with the runner's own ``qsa_ordered_decode_v2``
  -> the registered three-pass v2 leaf, or the v1 parent when the v2 variant is
  not registered.

Three contracts sit around that choice and are pinned separately:

- the batch/mixed owners are a *different* route: they select rows kernels by
  geometry and never read this decision, so the c1 prerequisite cannot leak
  into them;
- the QSA attention is never inside a captured MoE or layer graph, so no graph
  identity carries the decision - but a harness that captures the whole layer
  itself must forward the owner into the captured launch;
- the UD recovery/restoration passes and the diagnostic harnesses must carry
  the decision as the owner field, never as a process key.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_profiles import (
    PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
    PRODUCTION_Q8_QSA_RESTORED_FLAGS,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime import qwen4_exp_runner as runner_module

#: The process key this migration retires. The literal stays so a leftover
#: value can be proven inert.
RETIRED_ENV = "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE"
FIELD = "qsa_ordered_decode"
ROOT = Path(__file__).resolve().parents[1]

PARENT = runner_module.qwen4_exp_qsa_sparse_attention_paged_bf16_f32
WAVE32 = runner_module.qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32
ORDERED_V1 = runner_module.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32
ORDERED_V2 = runner_module.qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_v2_f32

DECODE_MIXER = "run_qwen4_exp_dense_qsa_token_mixer"
DECODE_LAYER = "run_qwen4_exp_dense_qsa_layer"
PREFILL_MIXER = "run_qwen4_exp_qsa_prefill_token_mixer"
PREFILL_LAYER = "run_qwen4_exp_qsa_prefill_layer"


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test starts with the retired switch in force."""

    monkeypatch.delenv(RETIRED_ENV, raising=False)


# --- the owner field, its resolver, and its scoped control ------------------


def test_an_unbound_runner_keeps_the_non_ordered_route() -> None:
    """The historical reader defaulted off; an unbound owner keeps that route."""

    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is False


def test_the_resolver_reads_the_owner_field() -> None:
    resolve = runner_module.resolve_qsa_ordered_decode
    assert resolve(SimpleNamespace(**{FIELD: True})) is True
    assert resolve(SimpleNamespace(**{FIELD: False})) is False
    assert resolve(SimpleNamespace(**{FIELD: 1})) is True
    # An owner built before the field existed, or a bare diagnostic object,
    # keeps the non-ordered route rather than raising.
    assert resolve(SimpleNamespace()) is False


def test_the_scope_changes_nothing_until_entered() -> None:
    owner = SimpleNamespace(**{FIELD: False})
    scope = runner_module.qsa_ordered_decode(owner, True)
    assert getattr(owner, FIELD) is False
    with scope:
        assert getattr(owner, FIELD) is True
    assert getattr(owner, FIELD) is False


def test_the_scope_binds_one_owner_and_restores_the_published_value() -> None:
    """A diagnostic arm must not leak into a co-resident owner."""

    first = SimpleNamespace(**{FIELD: False})
    second = SimpleNamespace(**{FIELD: True})
    with runner_module.qsa_ordered_decode(first, True):
        assert getattr(first, FIELD) is True
        assert getattr(second, FIELD) is True
        with runner_module.qsa_ordered_decode(second, False):
            assert getattr(first, FIELD) is True
            assert getattr(second, FIELD) is False
        assert getattr(second, FIELD) is True
    assert getattr(first, FIELD) is False
    assert getattr(second, FIELD) is True


def test_the_scope_restores_the_published_value_after_an_exception() -> None:
    owner = SimpleNamespace(**{FIELD: True})
    with pytest.raises(RuntimeError, match="arm failed"):
        with runner_module.qsa_ordered_decode(owner, False):
            raise RuntimeError("arm failed")
    assert getattr(owner, FIELD) is True


def test_an_unset_scope_keeps_the_published_value() -> None:
    owner = SimpleNamespace(**{FIELD: True})
    with runner_module.qsa_ordered_decode(owner):
        assert getattr(owner, FIELD) is True
    assert getattr(owner, FIELD) is True


def test_the_scope_writes_no_process_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    os.environ[RETIRED_ENV] = "1"
    owner = SimpleNamespace(**{FIELD: False})
    with runner_module.qsa_ordered_decode(owner, True):
        assert getattr(owner, FIELD) is True
        assert os.environ[RETIRED_ENV] == "1"
    assert getattr(owner, FIELD) is False
    assert os.environ[RETIRED_ENV] == "1"


def test_the_prerequisite_and_the_v2_leaf_keep_independent_owners() -> None:
    """The v2 leaf is a separate owner and must stay independent of this one.

    Each scope binds one field and restores only that field, so the prior value
    of the other owner is visible inside the scope and after its exit.
    """

    owner = SimpleNamespace(**{FIELD: True, "qsa_ordered_decode_v2": False})

    # Binding the v2 leaf moves the leaf alone; the prerequisite is untouched
    # inside the scope and the leaf's own prior value comes back on exit.
    with runner_module.qsa_ordered_decode_v2(owner, True):
        assert owner.qsa_ordered_decode_v2 is True
        assert getattr(owner, FIELD) is True
    assert owner.qsa_ordered_decode_v2 is False
    assert getattr(owner, FIELD) is True

    # Binding the prerequisite moves the prerequisite alone, and the leaf keeps
    # its prior value both inside the scope and after the restore.
    with runner_module.qsa_ordered_decode(owner, False):
        assert getattr(owner, FIELD) is False
        assert owner.qsa_ordered_decode_v2 is False
    assert getattr(owner, FIELD) is True
    assert owner.qsa_ordered_decode_v2 is False


# --- the scalar dispatch route ---------------------------------------------


def _mixer_weights() -> SimpleNamespace:
    """The projection stubs the scalar decode dispatch reads.

    ``attn_q`` carries a ``backend`` because the ordered-v2 resolver builds its
    ``KernelKey`` from it. A bare ``object()`` there would surface as an
    ``AttributeError`` once the owner parameter lands, which would mask a route
    mismatch as a harness failure instead of a contract failure.
    """

    return SimpleNamespace(
        projections={
            name: SimpleNamespace(backend="hip_gfx1151")
            for name in (
                "attn_q", "attn_k", "attn_v", "attn_output", "index_q",
                "index_k",
            )
        },
        q_norm_weight_ptr=40,
        k_norm_weight_ptr=41,
        index_q_norm_weight_ptr=42,
        index_k_norm_weight_ptr=43,
    )


def _decode_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    head_dim: int,
    selected_count: int = 64,
    v2_registered: bool = True,
    **kwargs,
) -> tuple[dict[str, int], list[tuple[int, int]]]:
    """Run one scalar decode step with every launch but the attention stubbed.

    The four sparse-decode owners are replaced by counters, so the returned
    mapping is the route the dispatch actually took rather than a source-text
    claim about it.
    """

    seen = {"parent": 0, "wave32": 0, "ordered_v1": 0, "ordered_v2": 0}
    ordered_scratch_calls: list[tuple[int, int]] = []

    def attention(name):
        def launch(*args, **call_kwargs):
            seen[name] += 1

        return launch

    monkeypatch.setattr(
        runner_module, "qwen4_exp_qsa_sparse_attention_paged_bf16_f32",
        attention("parent"))
    monkeypatch.setattr(
        runner_module, "qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32",
        attention("wave32"))
    monkeypatch.setattr(
        runner_module, "qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32",
        attention("ordered_v1"))
    v2 = attention("ordered_v2")
    monkeypatch.setattr(
        runner_module, "is_registered", lambda key: bool(v2_registered))
    monkeypatch.setattr(runner_module, "resolve", lambda **call_kwargs: v2)
    for name in (
        "launch_gguf_linear",
        "qwen35_write_paged_kv_f32_spans",
        "qwen4_exp_qsa_split_norm_rope_f32",
        "qwen4_exp_qsa_norm_rope_f32",
        "qwen4_exp_qsa_gate_context_f32",
    ):
        monkeypatch.setattr(runner_module, name, lambda *a, **k: None)

    runtime = object()
    state = SimpleNamespace(
        closed=False,
        runtime=runtime,
        max_positions=64,
        block_size=256,
        position=SimpleNamespace(ptr=20),
        block_table=SimpleNamespace(ptr=21),
        block_host=SimpleNamespace(size=4),
        key_cache=SimpleNamespace(ptr=22),
        value_cache=SimpleNamespace(ptr=23),
        append_spans=(),
        decode_spans=(),
        set_position=lambda position: None,
    )
    scores, coefficients = SimpleNamespace(ptr=30), SimpleNamespace(ptr=31)
    scratch = SimpleNamespace(
        closed=False,
        runtime=runtime,
        q_projected=SimpleNamespace(ptr=1),
        key_projected=SimpleNamespace(ptr=2),
        value_projected=SimpleNamespace(ptr=3),
        query=SimpleNamespace(ptr=4),
        key=SimpleNamespace(ptr=5),
        gate=SimpleNamespace(ptr=6),
        context=SimpleNamespace(ptr=7),
        gated=SimpleNamespace(ptr=8),
        output=SimpleNamespace(ptr=9),
        index_k_projected=SimpleNamespace(ptr=10),
        index_q_projected=SimpleNamespace(ptr=11),
        index_query=SimpleNamespace(ptr=12),
    )

    def ordered_attention_scratch(*, query_heads: int, selected_count: int):
        ordered_scratch_calls.append((query_heads, selected_count))
        return scores, coefficients

    scratch.ordered_attention_scratch = ordered_attention_scratch
    index = SimpleNamespace(
        index_heads=2,
        index_dim=8,
        dense_equivalent_limit=4,
        count=8,
        append=lambda *a, **k: None,
        select=lambda *a, **k: (selected_count, SimpleNamespace(ptr=77)),
    )
    weights = _mixer_weights()

    runner_module.run_qwen4_exp_dense_qsa_token_mixer(
        100,
        weights,
        attention_state=state,
        scratch=scratch,
        position=8,
        rows=1,
        hidden=16,
        query_heads=4,
        kv_heads=1,
        head_dim=head_dim,
        rotary_dim=8,
        theta=1e6,
        index_state=index,
        index_heads=2,
        index_dim=8,
        index_rotary_dim=8,
        runtime=runtime,
        **kwargs,
    )
    return seen, ordered_scratch_calls


def test_the_mixer_weight_stub_carries_the_backend_the_v2_key_reads() -> None:
    """The harness must not mask a route mismatch as an attribute error."""

    weights = _mixer_weights()
    key = KernelKey(
        str(weights.projections["attn_q"].backend),
        "qsa_sparse_attention", "bf16_kv",
        "strict_ordered_three_pass_v2_spans",
    )
    assert key.backend == "hip_gfx1151"
    assert key.variant == "strict_ordered_three_pass_v2_spans"


def test_an_unbound_decode_step_keeps_the_non_ordered_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, ordered_scratch = _decode_harness(monkeypatch, head_dim=256)
    assert seen == {"parent": 1, "wave32": 0, "ordered_v1": 0, "ordered_v2": 0}
    # The ordered scratch is the ordered route's own allocation; the parent
    # route never asks for it.
    assert ordered_scratch == []


def test_the_owner_selects_the_ordered_v1_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, ordered_scratch = _decode_harness(
        monkeypatch, head_dim=256, **{FIELD: True, "qsa_ordered_decode_v2": False})
    assert seen == {"parent": 0, "wave32": 0, "ordered_v1": 1, "ordered_v2": 0}
    assert ordered_scratch == [(4, 64)]


def test_the_published_v2_leaf_rides_the_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, _ = _decode_harness(
        monkeypatch, head_dim=256, **{FIELD: True, "qsa_ordered_decode_v2": True})
    assert seen == {"parent": 0, "wave32": 0, "ordered_v1": 0, "ordered_v2": 1}


def test_an_unregistered_v2_leaf_keeps_the_ordered_v1_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, _ = _decode_harness(
        monkeypatch, head_dim=256, v2_registered=False,
        **{FIELD: True, "qsa_ordered_decode_v2": True})
    assert seen == {"parent": 0, "wave32": 0, "ordered_v1": 1, "ordered_v2": 0}


def test_the_v2_leaf_cannot_select_without_the_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v1 route is a prerequisite, not a sibling selection."""

    seen, ordered_scratch = _decode_harness(
        monkeypatch, head_dim=256, **{FIELD: False, "qsa_ordered_decode_v2": True})
    assert seen == {"parent": 1, "wave32": 0, "ordered_v1": 0, "ordered_v2": 0}
    assert ordered_scratch == []


def test_a_published_v2_leaf_alone_cannot_select_the_ordered_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserved today: the v2 field with the prerequisite off keeps the parent."""

    seen, ordered_scratch = _decode_harness(
        monkeypatch, head_dim=256, qsa_ordered_decode_v2=True)
    assert seen == {"parent": 1, "wave32": 0, "ordered_v1": 0, "ordered_v2": 0}
    assert ordered_scratch == []


def test_the_h256_geometry_guard_stays_dispatch_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any other head width keeps its own non-ordered owner."""

    seen, ordered_scratch = _decode_harness(
        monkeypatch, head_dim=128, qsa_wave32=False,
        **{FIELD: True, "qsa_ordered_decode_v2": True})
    assert seen == {"parent": 1, "wave32": 0, "ordered_v1": 0, "ordered_v2": 0}
    assert ordered_scratch == []

    seen, _ = _decode_harness(
        monkeypatch, head_dim=128, qsa_wave32=True,
        **{FIELD: True, "qsa_ordered_decode_v2": True})
    assert seen == {"parent": 0, "wave32": 1, "ordered_v1": 0, "ordered_v2": 0}


@pytest.mark.parametrize(
    ("selected_count", "ordered_v1"),
    [
        (0, True),      # below the v2 leaf's lower bound: the v1 parent
        (1, False),     # first covered count: the v2 leaf
        (2048, False),  # unrelated interior value
        (4096, False),  # last covered count
        (4097, True),   # above the bound: the v1 parent
    ],
)
def test_the_selection_count_guard_stays_dispatch_time(
    monkeypatch: pytest.MonkeyPatch, selected_count: int, ordered_v1: bool,
) -> None:
    seen, _ = _decode_harness(
        monkeypatch, head_dim=256, selected_count=selected_count,
        **{FIELD: True, "qsa_ordered_decode_v2": True})
    assert seen["ordered_v1"] == int(ordered_v1)
    assert seen["ordered_v2"] == int(not ordered_v1)
    assert seen["parent"] == 0


def test_the_retired_switch_cannot_move_the_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leftover process value is inert; only the owner field selects."""

    monkeypatch.setenv(RETIRED_ENV, "1")
    seen, ordered_scratch = _decode_harness(monkeypatch, head_dim=256)
    assert seen == {"parent": 1, "wave32": 0, "ordered_v1": 0, "ordered_v2": 0}
    assert ordered_scratch == []


def test_the_decode_mixer_takes_the_owner_parameter_and_reads_no_environment() -> None:
    parameters = inspect.signature(
        runner_module.run_qwen4_exp_dense_qsa_token_mixer).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is False
    source = inspect.getsource(
        runner_module.run_qwen4_exp_dense_qsa_token_mixer)
    assert "os.environ" not in source
    # The decode precondition and the ordered route's own guard are untouched.
    assert "requires rows == 1" in source


def test_the_gfx1151_backend_registers_both_ordered_leaves() -> None:
    register_gfx1151_kernels(replace=True)
    for variant in (
        "strict_ordered_three_pass_spans",
        "strict_ordered_three_pass_v2_spans",
    ):
        assert is_registered(
            KernelKey("hip_gfx1151", "qsa_sparse_attention", "bf16_kv", variant)
        ), variant


# --- forwarding: the published field must reach the kernel choice -----------


def _calls_named(source: str, names) -> list[ast.Call]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _layer_scratch(runtime):
    return SimpleNamespace(
        closed=False, runtime=runtime,
        attention_gr=SimpleNamespace(), qsa=SimpleNamespace(),
        gdn=SimpleNamespace(),
        after_attention=SimpleNamespace(ptr=16), ffn_gr=SimpleNamespace(),
        moe=SimpleNamespace(), output=SimpleNamespace(ptr=15))


def _layer_weights():
    return SimpleNamespace(
        attention_gr=SimpleNamespace(norm_weight_ptr=1, down=2, up=3, inject=4),
        mixer=SimpleNamespace(),
        ffn_gr=SimpleNamespace(norm_weight_ptr=9, down=10, up=11, inject=12),
        moe=SimpleNamespace())


def test_the_dense_layer_forwards_the_owner_to_the_mixer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    runtime = object()
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_gr_read",
        lambda *a, **k: SimpleNamespace(
            normalized=None, gate=None, mixed=SimpleNamespace(ptr=11),
            inject_logits=SimpleNamespace(ptr=12)))
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_moe",
        lambda *a, **k: SimpleNamespace(output=SimpleNamespace(ptr=14)))
    monkeypatch.setattr(
        runner_module, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, DECODE_MIXER,
        lambda *a, **k: (seen.append(k.get(FIELD, "absent")),
                         SimpleNamespace(ptr=13))[1])
    kwargs = dict(
        attention_state=object(), scratch=_layer_scratch(runtime),
        position=0, rows=1, branches=1, hidden=2, low_rank=2, query_heads=24,
        kv_heads=2, head_dim=256, rotary_dim=64, theta=1e6, ffn=8, experts=1,
        top_k=1, runtime=runtime)
    runner_module.run_qwen4_exp_dense_qsa_layer(
        100, _layer_weights(), **kwargs, **{FIELD: True})
    runner_module.run_qwen4_exp_dense_qsa_layer(100, _layer_weights(), **kwargs)
    assert seen == [True, False]


def test_the_layer_owner_takes_the_owner_with_the_non_ordered_default() -> None:
    parameters = inspect.signature(
        runner_module.run_qwen4_exp_dense_qsa_layer).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is False
    calls = _calls_named(
        inspect.getsource(runner_module.run_qwen4_exp_dense_qsa_layer),
        (DECODE_MIXER,))
    assert len(calls) == 1
    value = _keyword(calls[0], FIELD)
    assert isinstance(value, ast.Name), ast.dump(calls[0])
    assert value.id == FIELD


def test_the_runner_step_forwards_its_own_decision() -> None:
    source = textwrap.dedent(
        inspect.getsource(runner_module.Qwen4ExpGGUFResidentModelRunner.step))
    calls = _calls_named(source, (DECODE_LAYER,))
    assert len(calls) == 1
    qsa = [call for call in calls if _keyword(call, "index_state") is not None]
    assert len(qsa) == 1
    value = _keyword(qsa[0], FIELD)
    assert isinstance(value, ast.Attribute), ast.dump(qsa[0])
    assert value.attr == FIELD
    # The v2 leaf keeps its own forwarding beside it.
    assert _keyword(qsa[0], "qsa_ordered_decode_v2") is not None


def test_the_prefill_route_keeps_its_own_selection() -> None:
    """Ordered decode is a c1 decode route; prefill must not grow the field."""

    for name in (PREFILL_MIXER, PREFILL_LAYER):
        parameters = inspect.signature(getattr(runner_module, name)).parameters
        assert FIELD not in parameters, name
    source = textwrap.dedent(
        inspect.getsource(
            runner_module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps))
    calls = _calls_named(source, (PREFILL_LAYER,))
    assert len(calls) == 1
    assert _keyword(calls[0], FIELD) is None


# --- the speculative draft and the serving clones ---------------------------


def _draft_stub():
    from hipengine.runtime import qwen4_exp_mtp as draft_module

    draft = object.__new__(draft_module.Qwen4ExpGGUFMTPDraftRunner)
    draft.position, draft.max_sequence_length, draft.closed = 0, 8, False
    draft.config = SimpleNamespace(**{
        name: 2 for name in (
            "residual_branch_count", "hidden_size", "residual_low_rank",
            "attention_head_count", "attention_kv_head_count",
            "attention_key_length", "rope_dimension_count", "rope_freq_base",
            "indexer_head_count", "indexer_key_length",
            "expert_feed_forward_length", "expert_count", "expert_used_count")})
    draft._buffers = [SimpleNamespace(ptr=i, nbytes=4) for i in range(12)]
    draft.runtime = SimpleNamespace(memcpy=lambda *a: None)
    draft.binding = draft.attention = draft.index = draft.layer_scratch = object()
    draft._fuse_inputs = lambda *a, **kw: None
    return draft


def test_the_draft_runner_defaults_to_the_non_ordered_route() -> None:
    from hipengine.runtime import qwen4_exp_mtp as draft_module

    assert getattr(
        draft_module.Qwen4ExpGGUFMTPDraftRunner, FIELD, "absent") is False


def test_the_draft_forwards_the_published_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from hipengine.runtime import qwen4_exp_mtp as draft_module

    draft = _draft_stub()
    calls = []
    monkeypatch.setattr(
        draft_module, DECODE_LAYER, lambda *a, **kw: calls.append(kw))
    setattr(draft, FIELD, True)
    draft.forward(1, np.zeros(2), run_head=False, cache_only=True)
    assert calls[-1][FIELD] is True
    assert calls[-1]["cache_only"] is True
    setattr(draft, FIELD, False)
    draft.forward(2, np.zeros(2), run_head=False, cache_only=True)
    assert calls[-1][FIELD] is False


def test_an_unpublished_draft_keeps_the_non_ordered_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from hipengine.runtime import qwen4_exp_mtp as draft_module

    draft = _draft_stub()
    calls = []
    monkeypatch.setattr(
        draft_module, DECODE_LAYER, lambda *a, **kw: calls.append(kw))
    draft.forward(1, np.zeros(2), run_head=False, cache_only=True)
    assert calls[-1][FIELD] is False


def test_the_provider_publishes_the_owner_on_the_draft_it_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen4_exp_mtp as provider_module
    from hipengine.speculative.registry import SpeculativeProviderConfig

    target = SimpleNamespace(
        backend="hip_gfx1151",
        runner=SimpleNamespace(**{
            FIELD: True, "qsa_ordered_decode_v2": True,
            "runtime": object(), "config": object(),
            "max_sequence_length": 16}))
    built: list[SimpleNamespace] = []

    def factory(*args, **kwargs):
        built.append(SimpleNamespace(**kwargs))
        return built[-1]

    monkeypatch.setattr(provider_module, "discover_gguf_files",
                        lambda path: [Path("draft.gguf")])
    monkeypatch.setattr(provider_module, "GGUFReader",
                        lambda path: SimpleNamespace(info=None))
    monkeypatch.setattr(provider_module, "build_qwen4_exp_mtp_gguf_map",
                        lambda infos: {})
    monkeypatch.setattr(provider_module, "plan_qwen4_exp_mtp_residency",
                        lambda model_map: {})
    monkeypatch.setattr(
        provider_module, "materialize_qwen4_exp_mtp_weights",
        lambda readers, **kwargs: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        provider_module, "Qwen4ExpGGUFMTPDraftRunner", factory)

    provider = provider_module.Qwen4ExpMTPTextProvider(
        target_generator=target,
        config=SpeculativeProviderConfig(
            provider="qwen4_exp_mtp", draft_model="draft.gguf",
            candidate_budget=1))

    assert len(built) == 1
    assert getattr(built[0], FIELD) is True
    assert provider.draft is built[0]


def test_the_serving_pool_publishes_the_owner_on_each_staged_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.generation import qwen4_exp_serving_mtp as serving_module

    target = SimpleNamespace(
        begin_device_transaction=lambda reuse_snapshot=False: object(),
        commit_device_transaction=lambda checkpoint: None)
    provider = SimpleNamespace(
        draft=SimpleNamespace(**{FIELD: False, "qsa_ordered_decode_v2": False}),
        _resident=object(), iter_generate=lambda **kwargs: None)
    owner = object.__new__(serving_module.Qwen4ExpServingMTP)
    owner.generator = SimpleNamespace(
        _speculative_provider=provider,
        runner=SimpleNamespace(**{
            FIELD: True, "qsa_ordered_decode_v2": True, "config": object(),
            "max_sequence_length": 8, "runtime": object()}),
        backend="hip_gfx1151")
    owner.capacity = 2
    owner.staging_capacity = 2
    owner._all_runners = [target]
    owner._init_mtp()
    built: list[SimpleNamespace] = []

    def factory(*args, **kwargs):
        built.append(SimpleNamespace(**kwargs))
        return built[-1]

    monkeypatch.setattr(serving_module, "Qwen4ExpGGUFMTPDraftRunner", factory)
    owner._prepare_mtp()

    assert len(built) == 1
    assert getattr(built[0], FIELD) is True
    # The provider's own draft is the first pool entry and keeps its value.
    assert owner._all_drafts[0] is provider.draft
    assert getattr(owner._all_drafts[0], FIELD) is False
    assert len(owner._all_drafts) == 2


def test_a_serving_clone_inherits_the_published_owner() -> None:
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(SimpleNamespace(**{FIELD: True}), clone)
    assert getattr(clone, FIELD) is True
    # An unbound carrier publishes the non-ordered route, never the other
    # lane's value.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert getattr(clone, FIELD) is False


# --- the plan publishes the decision; no switch selects it ------------------


def _bind(monkeypatch, profile_name, *, quant="gguf_ud_q4_k_xl",
          runner_handle=None):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model="qwen4_exp_gguf", backend="hip_gfx1151", quant=quant,
        profile=profile_name)
    generator = SimpleNamespace()
    if runner_handle is not None:
        generator.runner = runner_handle
    profile.binder(generator, profile)
    return generator, profile


@pytest.mark.parametrize("profile_name,expected", [
    (ExecutionProfile.PRODUCTION, True),
    (ExecutionProfile.STRICT, False),
    (ExecutionProfile.BATCH_INVARIANT, False),
])
def test_the_binder_publishes_the_plan_decision_and_writes_no_switch(
    monkeypatch: pytest.MonkeyPatch, profile_name, expected,
) -> None:
    generator, _ = _bind(monkeypatch, profile_name)
    assert getattr(generator, FIELD) is expected
    assert RETIRED_ENV not in os.environ


def test_the_binder_publishes_on_the_runner_it_finds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = SimpleNamespace()
    generator, _ = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, runner_handle=handle)
    assert getattr(generator, FIELD) is True
    assert getattr(handle, FIELD) is True
    assert RETIRED_ENV not in os.environ


def test_a_production_request_on_a_non_ud_lane_keeps_the_non_ordered_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the UD plan has a production binder; every other lane is strict.

    The retired switch was written by the binder that actually bound, so a
    production request that falls back to the strict binder carried ``0``. The
    owner publication keeps that per-lane result rather than the requested
    profile name.
    """

    generator, profile = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, quant="gguf_q4_k_m")
    assert profile.binder.__name__ == "_strict_binder"
    assert getattr(generator, FIELD) is False
    assert RETIRED_ENV not in os.environ


def test_the_ud_recovery_and_restoration_leave_the_owner_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The UD lane closes and restores the flag; the net decision is the plan's.

    Both tables stop naming it because a request-local owner is not a process
    key: the recovery pass would otherwise write a value nothing reads.
    """

    generator, profile = _bind(
        monkeypatch, ExecutionProfile.PRODUCTION, quant="gguf_ud_q4_k_xl")
    assert profile.manifest["quant"] == "gguf_ud_q4_k_xl"
    assert getattr(generator, FIELD) is True
    assert RETIRED_ENV not in os.environ
    for table in (
        PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
        tuple(PRODUCTION_Q8_QSA_RESTORED_FLAGS),
    ):
        assert "QSA_ORDERED_DECODE" not in table


def test_a_strict_bind_after_a_production_bind_republishes_its_own_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every bind publishes its own value; nothing is inherited in-process."""

    production, _ = _bind(monkeypatch, ExecutionProfile.PRODUCTION)
    assert getattr(production, FIELD) is True
    strict, _ = _bind(monkeypatch, ExecutionProfile.STRICT)
    assert getattr(strict, FIELD) is False
    assert RETIRED_ENV not in os.environ


# --- the live harnesses name the owner field, not a switch ------------------


def test_the_campaign_prerequisite_reads_the_runner_field() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    published = SimpleNamespace(**{FIELD: True})
    env = {"other": "keep"}
    with route_scope("qsa-ordered-v2", "after", published, environment=env):
        assert published.qsa_ordered_decode_v2 is True
    # The arm restores the plan's own leaf value on exit, and the prerequisite
    # the arm rides is never rewritten.
    assert published.qsa_ordered_decode_v2 is False
    assert getattr(published, FIELD) is True
    with route_scope("qsa-ordered-v2", "before", published, environment=env):
        assert published.qsa_ordered_decode_v2 is False
    assert published.qsa_ordered_decode_v2 is False
    assert getattr(published, FIELD) is True
    assert env == {"other": "keep"}


@pytest.mark.parametrize("published", [False, None])
def test_the_campaign_prerequisite_refuses_an_arm_without_the_route(published):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope

    owner = SimpleNamespace(**{FIELD: published})
    with pytest.raises(ValueError, match="ordered v1 route"):
        route_scope("qsa-ordered-v2", "after", owner, environment={})


def test_the_campaign_control_record_names_the_owner_field() -> None:
    from scripts.qwen4exp_halo_box_campaign_ab import (
        qsa_ordered_decode_v2_control_record,
    )

    record = qsa_ordered_decode_v2_control_record(
        SimpleNamespace(**{FIELD: True, "qsa_ordered_decode_v2": True}))
    assert record["runner_field"] == "qsa_ordered_decode_v2"
    assert FIELD in str(record["prerequisite"])
    assert RETIRED_ENV not in str(record["prerequisite"])


def test_the_state_gate_arm_keeps_the_prerequisite_off_the_environment() -> None:
    from scripts.qwen4exp_row4_state_gate import arm_control

    owner = SimpleNamespace(**{FIELD: True, "qsa_ordered_decode_v2": False})
    env = {"other": "keep"}
    with arm_control("qsa-ordered-v2", "1", owner, environment=env):
        assert owner.qsa_ordered_decode_v2 is True
        assert env == {"other": "keep"}
    assert owner.qsa_ordered_decode_v2 is False
    assert getattr(owner, FIELD) is True


def test_the_drift_probe_names_the_owner_field() -> None:
    from scripts import qwen4exp_qsa_v2_drift_probe as probe

    assert probe.PREREQUISITE_FIELD == FIELD
    assert not hasattr(probe, "PREREQUISITE_ENV")
    source = inspect.getsource(probe)
    assert FIELD in source
    assert RETIRED_ENV not in source


def _load_layer2_gate():
    path = ROOT / "scripts" / "qwen4exp_layer2_profile_gate.py"
    spec = importlib.util.spec_from_file_location(
        "qwen4exp_layer2_profile_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_layer2_gate_arms_the_owner_not_a_process_key() -> None:
    module = _load_layer2_gate()
    spec = module.CandidateSpec
    assert FIELD in {field.name for field in __import__(
        "dataclasses").fields(spec)}
    # An arm that names nothing keeps the base profile's published route.
    assert spec.__dataclass_fields__[FIELD].default is None
    assert "QSA_ORDERED_DECODE" not in module.CONSERVATIVE_ARITHMETIC_FLAGS

    conservative = module.CANDIDATES["production_conservative"]
    assert getattr(conservative, FIELD) is False
    assert RETIRED_ENV not in conservative.environment

    for name in ("q8_blockscale_guarded_restore", "q8_blockscale_guarded_ordered"):
        candidate = module.CANDIDATES[name]
        assert getattr(candidate, FIELD) is True, name
        assert RETIRED_ENV not in candidate.environment, name
        # The leaf keeps its own independent field.
        assert candidate.qsa_ordered_decode_v2 is True, name

    assert getattr(module.CANDIDATES["q8_blockscale"], FIELD) is None


def test_the_prefill_restore_gate_treats_the_owner_as_runner_state() -> None:
    from scripts import qwen4exp_prefill_restore_ab as restore

    assert FIELD in restore.SHARED_DECODE_SAFE_OWNER_FIELDS
    assert RETIRED_ENV not in restore.SHARED_DECODE_SAFE_ENV


@pytest.mark.parametrize("published", [True, False])
@pytest.mark.parametrize("declared", [None, False, True])
def test_the_prefill_restore_arm_keeps_an_undeclared_route_published(
    published: bool, declared: bool | None,
) -> None:
    """``None`` means "keep the base profile's route", not "bind it off".

    The gate enters this scope for every measured arm, and a ``CandidateSpec``
    field left at ``None`` is not a change. Handing that ``None`` to the bare
    owner scope would read as ``bool(None)`` and silently disable ordered
    decode on every arm that does not declare it - including the production
    reference arm - so the consumer keeps ``None`` as an unbound scope and the
    published value has to survive it.
    """

    from scripts import qwen4exp_prefill_restore_ab as restore

    runner = SimpleNamespace(**{FIELD: published})
    with restore.arm_qsa_ordered_scope(runner, declared):
        assert getattr(runner, FIELD) is (
            published if declared is None else declared), declared
    assert getattr(runner, FIELD) is published


def test_the_prefill_restore_sample_enters_the_none_preserving_scope() -> None:
    """The measured arm's scope stack uses the helper, not the bare scope.

    ``main`` needs a model, so this checks the wiring it cannot execute: the
    ``with`` stack inside ``sample`` must not hand a ``CandidateSpec`` value
    straight to ``qsa_ordered_decode``, where ``None`` would bind the route off.
    """

    import ast

    path = ROOT / "scripts" / "qwen4exp_prefill_restore_ab.py"
    tree = ast.parse(path.read_text())
    sample = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sample")
    entered = [
        item.context_expr.func.id
        for node in ast.walk(sample) if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
    ]
    assert "arm_qsa_ordered_scope" in entered
    assert "qsa_ordered_decode" not in entered


def test_the_owner_field_is_inventoried_as_a_candidate_owner_change() -> None:
    """The shared-graph inventory must name the field, not a removed key.

    ``candidate_owner_changes`` is what every harness reads to decide which
    captured units a candidate touches, so an arm that binds the route through
    a field the inventory does not know about is invisible to the shared-graph
    check.
    """

    module = _load_layer2_gate()
    assert FIELD in module._OWNER_FIELD_NAMES
    conservative = module.candidate_owner_changes(
        module.CANDIDATES["production_conservative"])
    assert conservative[FIELD] is False
    ordered = module.candidate_owner_changes(
        module.CANDIDATES["q8_blockscale_guarded_ordered"])
    assert ordered[FIELD] is True
    # A candidate that never names the field is not a change.
    assert FIELD not in module.candidate_owner_changes(
        module.CANDIDATES["q8_blockscale"])


def test_the_conservative_cost_arm_binds_the_owner_from_its_declaration() -> None:
    """The counterbalanced pair must move the route, not inherit the base one."""

    from scripts import qwen4exp_conservative_cost as cost

    assert cost.arm_qsa_ordered_decode("before", True) is True
    assert cost.arm_qsa_ordered_decode("after", True) is False
    assert cost.arm_qsa_ordered_decode("before", False) is False


# --- construction provenance: snapshots and split phases -------------------


def test_the_journey_arms_record_the_route_their_construction_carried() -> None:
    """An arm's former value follows construction, not an arm-name list.

    The retired key was a captured process key: an arm built from the strict
    snapshot inherited the strict bind's value, a family arm that overlaid the
    production ``QSA_*`` slice inherited the production one, and an arm that
    named it nowhere ran whatever the production bind published.
    """

    from scripts import qwen4exp_journey_localize as journey

    arms = {
        "strict_snapshot": {RETIRED_ENV: "0"},
        "qsa_family": {RETIRED_ENV: "1"},
        "unnamed": {},
        "declared": {RETIRED_ENV: "0"},
    }
    values = journey.migrate_qsa_ordered_arms(
        arms, production_value=True,
        declared={"declared": False})
    assert values == {
        "strict_snapshot": False,
        "qsa_family": True,
        "unnamed": True,
        "declared": False,
    }
    # The retired key does not survive in any arm's former environment.
    assert all(RETIRED_ENV not in env for env in arms.values())


@pytest.mark.parametrize("production_value", [True, False])
def test_snapshot_arms_keeps_the_route_each_arms_construction_carried(
    monkeypatch: pytest.MonkeyPatch, production_value: bool,
) -> None:
    """Execute the real arm builder, not a hand-written arm table.

    Each binding's former snapshot is reconstructed from the owner it published,
    ``snapshot_arms`` rebuilds the family arms from those snapshots, and the
    migration records what each arm's own construction carried. The strict
    snapshot's arms stay off, the QSA family overlays the production slice and
    follows it, an arm that copied the strict ``QSA_*`` keys stays off, and an
    arm that named the route nowhere follows the production bind the arm loop
    restores from ``bound``.
    """

    from scripts import qwen4exp_journey_localize as journey

    strict_snapshot = journey.qsa_ordered_snapshot({}, published=False)
    production_snapshot = journey.qsa_ordered_snapshot(
        {}, published=production_value)
    # Work on the harness's own base table, copied so the module's arms keep the
    # state the rest of this suite asserts on.
    arms = {name: dict(env) for name, env in journey.ARMS.items()}
    # ``snapshot_arms`` also classifies these arm names for the GDN migration;
    # they are the same names the harness builds, so the update is idempotent.
    journey.snapshot_arms(strict_snapshot, production_snapshot, arms=arms)
    assert RETIRED_ENV in arms["all_flags_strict"]
    assert RETIRED_ENV in arms["strict_plus_qsa"]

    values = journey.migrate_qsa_ordered_arms(
        arms, production_value=production_value,
        declared=journey.declared_qsa_ordered_decode())

    assert values["all_flags_strict"] is False
    assert values["strict_plus_qsa"] is production_value
    assert values["q8down_off_without_qsa"] is False
    assert values["q8down_off_without_matrix"] is production_value
    # A candidate declaration outranks the default for an arm that named no key.
    assert values["production_conservative"] is False
    assert all(RETIRED_ENV not in env for env in arms.values())

    # The resolver the arm loop calls reads the recorded map, so the arm it
    # measures gets the value its construction carried.
    monkeypatch.setattr(journey, "_QSA_ORDERED_ARM_VALUES", dict(values))
    assert journey.arm_qsa_ordered_decode(
        "all_flags_strict", production_value) is False
    assert journey.arm_qsa_ordered_decode(
        "strict_plus_qsa", production_value) is production_value
    assert journey.arm_qsa_ordered_decode(
        "q8down_off_without_matrix", production_value) is production_value


def _split_runner() -> SimpleNamespace:
    """The runner fields the split-phase scopes read, at their published values."""

    return SimpleNamespace(**{
        FIELD: True,
        "gdn_colwarps_prefill": True,
        "gdn_register_prefill": False,
        "q4_pair_prefill": False,
        "q8_wave_scale_prefill": False,
        "gr_wave_scale": False,
        "gr_wide": False,
        "q8_mmq_prefill_enabled": True,
        "production_moe_prefill_enabled": True,
        "q4_dp4a_decode_enabled": True,
        "q4_dp4a_decode_layers": frozenset({1}),
        "qsa_h256_prefill_mode": "page256",
    })


def _drive_split(journey, runner, arm, *, fail_on=None):
    """Run the real ``split_trajectory`` over a recording fake runner."""

    seen: list[tuple[str, bool]] = []

    def prefill(ids):
        seen.append(("prefill", getattr(runner, FIELD)))
        return SimpleNamespace(logits=[1.0])

    def step(token):
        seen.append(("decode", getattr(runner, FIELD)))
        if token == fail_on:
            raise RuntimeError("decode failed")
        return SimpleNamespace(logits=[1.0])

    runner.reset = lambda: None
    runner.prefill = prefill
    runner.step = step
    journey.split_trajectory(
        runner, [1, 2], [3, 4],
        {"phase": "prefill"}, {"phase": "decode"}, arm=arm)
    return seen


def test_the_journey_split_trajectory_binds_the_route_per_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the phase wiring: each split phase runs its own route.

    The prefill and decode flag sets are the only thing that differs between the
    two phases, so the owner has to follow the phase that is running rather than
    the arm name. The plan's published value comes back when the arm ends.
    """

    from scripts import qwen4exp_journey_localize as journey

    monkeypatch.setattr(journey, "clear_arm_graphs", lambda runner: None)
    monkeypatch.setattr(journey, "set_flags", lambda flags: None)

    # Production prefill, strict decode: the strict phase is the decode steps.
    runner = _split_runner()
    seen = _drive_split(journey, runner, "production_prefill_strict_decode")
    assert seen == [("prefill", True), ("decode", False), ("decode", False)]
    assert getattr(runner, FIELD) is True

    # Strict prefill, production decode: the strict phase is the prefill.
    runner = _split_runner()
    seen = _drive_split(journey, runner, "strict_prefill_production_decode")
    assert seen == [("prefill", False), ("decode", True), ("decode", True)]
    assert getattr(runner, FIELD) is True

    # An arm outside the split pair never moves the route.
    runner = _split_runner()
    seen = _drive_split(journey, runner, "strict_plus_qsa")
    assert seen == [("prefill", True), ("decode", True), ("decode", True)]
    assert getattr(runner, FIELD) is True


def test_a_failed_split_phase_restores_the_published_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The phase override must not outlive a raise inside the measured steps."""

    from scripts import qwen4exp_journey_localize as journey

    monkeypatch.setattr(journey, "clear_arm_graphs", lambda runner: None)
    monkeypatch.setattr(journey, "set_flags", lambda flags: None)

    runner = _split_runner()
    with pytest.raises(RuntimeError, match="decode failed"):
        _drive_split(
            journey, runner, "production_prefill_strict_decode", fail_on=4)
    assert getattr(runner, FIELD) is True


def test_the_journey_arm_resolver_prefers_the_candidate_declaration() -> None:
    from scripts import qwen4exp_journey_localize as journey

    # The conservative arm declared the key off, so it keeps running off.
    assert journey.arm_qsa_ordered_decode(
        "production_conservative", published=True) is False
    # An arm that never named or inherited the route runs the plan's value.
    assert journey.arm_qsa_ordered_decode(
        "gdn_strict", published=True) is True
    assert journey.arm_qsa_ordered_decode(
        "gdn_strict", published=False) is False


def test_the_journey_split_arm_binds_the_route_per_decode_phase() -> None:
    """A split arm's former environment carried one value per phase."""

    from scripts import qwen4exp_journey_localize as journey

    runner = SimpleNamespace(**{FIELD: True})
    with journey.qsa_ordered_split_scope(
            runner, "production_prefill_strict_decode", "decode"):
        assert getattr(runner, FIELD) is False
    assert getattr(runner, FIELD) is True

    with journey.qsa_ordered_split_scope(
            runner, "strict_prefill_production_decode", "decode"):
        assert getattr(runner, FIELD) is True

    # An arm outside the split pair leaves the plan's value in place.
    with journey.qsa_ordered_split_scope(runner, "strict_plus_qsa", "decode"):
        assert getattr(runner, FIELD) is True
    assert getattr(runner, FIELD) is True


def test_no_journey_arm_carries_the_retired_key() -> None:
    from scripts import qwen4exp_journey_localize as journey

    assert all(RETIRED_ENV not in env for env in journey.ARMS.values())
    # The value the conservative arm's former environment carried is a
    # declaration now, so the arm's resolver still measures it.
    assert journey.arm_qsa_ordered_decode(
        "production_conservative", published=True) is False


# --- the batch route is a separate selection -------------------------------


def test_the_batch_owners_take_no_ordered_parameter() -> None:
    from hipengine.runtime import qwen4_exp_batch_qsa as batch

    for function in (
        batch.run_qwen4_exp_batch_dense_qsa_token_mixer,
        batch.run_qwen4_exp_batch_qsa_mixed_token_mixer,
        batch.run_qwen4_exp_batch_qsa_attention,
    ):
        assert FIELD not in inspect.signature(function).parameters, (
            function.__qualname__)
    source = inspect.getsource(batch)
    assert RETIRED_ENV not in source
    assert FIELD not in source


@pytest.mark.parametrize(
    ("geometry", "variant"),
    [
        (dict(head_dim=256, block_size=256, query_heads=24, kv_heads=2),
         "strict_h256_head_pair_rows_spans"),
        (dict(head_dim=128, block_size=256, query_heads=24, kv_heads=2),
         "strict_rows_spans"),
        (dict(head_dim=256, block_size=128, query_heads=24, kv_heads=2),
         "strict_rows_spans"),
        # The head-pair leaf needs whole pairs: 6 heads over 2 kv heads is not
        # a multiple of 2 * kv_heads, so the exact rows parent serves it.
        (dict(head_dim=256, block_size=256, query_heads=6, kv_heads=2),
         "strict_rows_spans"),
    ],
)
def test_the_batch_geometry_selection_is_unchanged(geometry, variant) -> None:
    """The batch route selects rows owners by geometry, never by this owner."""

    from hipengine.runtime import qwen4_exp_batch_qsa as batch

    function, selected = batch.batch_qsa_attention_implementation(**geometry)
    assert selected == variant
    assert function is not ORDERED_V1
    assert function is not ORDERED_V2
    assert function is not PARENT
    assert function is not WAVE32


@pytest.mark.parametrize("declared", [None, False, True])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_shared_candidate_binding_honors_both_ordered_declarations(
        declared, published, failed):
    from contextlib import ExitStack
    from dataclasses import replace
    from scripts import qwen4exp_layer2_profile_gate as gate

    owner = SimpleNamespace(qsa_ordered_decode=published,
                            qsa_ordered_decode_v2=published)
    generator = SimpleNamespace(runner=owner)
    spec = replace(gate.CANDIDATES["production_baseline"],
                   qsa_ordered_decode=declared, qsa_ordered_decode_v2=declared)
    expected = published if declared is None else declared

    def measure():
        with ExitStack() as stack:
            report = gate.bind_candidate_owners(stack, generator, spec)
            assert owner.qsa_ordered_decode is expected
            assert owner.qsa_ordered_decode_v2 is expected
            assert report["qsa_ordered_decode"] is expected
            assert report["qsa_ordered_decode_v2"] is expected
            if failed:
                raise RuntimeError("measurement failed")

    if failed:
        with pytest.raises(RuntimeError, match="measurement failed"):
            measure()
    else:
        measure()
    assert owner.qsa_ordered_decode is published
    assert owner.qsa_ordered_decode_v2 is published


# --- graph identities and capture sites ------------------------------------


def test_no_graph_key_carries_the_ordered_decision() -> None:
    """The QSA attention is outside every captured unit the runner owns."""

    for name in (DECODE_LAYER, PREFILL_LAYER):
        parameters = inspect.signature(getattr(runner_module, name)).parameters
        assert "layer_graph_cache" not in parameters, name
        assert "layer_graph_key" not in parameters, name

    source = inspect.getsource(runner_module.run_qwen4_exp_dense_qsa_layer)
    mixer = _calls_named(source, (DECODE_MIXER,))
    assert len(mixer) == 1
    # The captured unit is the MoE FFN, which is keyed on the DP4A decision.
    assert _keyword(mixer[0], "graph_cache") is None
    assert _keyword(mixer[0], "graph_key") is None
    moe = _calls_named(source, ("run_qwen4_exp_moe",))
    assert len(moe) == 1
    assert _keyword(moe[0], "graph_cache") is not None
    assert _keyword(moe[0], "graph_key") is not None

    step = textwrap.dedent(
        inspect.getsource(runner_module.Qwen4ExpGGUFResidentModelRunner.step))
    calls = [
        call for call in _calls_named(step, (DECODE_LAYER,))
        if _keyword(call, "index_state") is not None
    ]
    assert len(calls) == 1
    key = _keyword(calls[0], "moe_graph_key")
    assert isinstance(key, ast.Call)
    assert getattr(key.func, "id", None) == "q4_dp4a_decode_graph_key"


def test_the_stateful_layer_graph_probe_forwards_the_owner_into_the_capture() -> None:
    """A whole-layer capture bakes in its route, so it must name the owner."""

    path = ROOT / "scripts" / "qwen4exp_stateful_layer_graph_probe.py"
    source = path.read_text()
    calls = _calls_named(source, (DECODE_LAYER,))
    assert len(calls) == 1
    value = _keyword(calls[0], FIELD)
    assert isinstance(value, ast.Attribute), ast.dump(calls[0])
    assert value.attr == FIELD
