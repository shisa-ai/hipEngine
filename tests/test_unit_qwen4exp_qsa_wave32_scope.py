"""The QSA H128 wave32 selection is request-local runner state.

``HIPENGINE_QWEN4_EXP_QSA_WAVE32`` was the process-wide transport for two
dispatch decisions: the paged sparse-attention decode owner (head_dim 128) and
the prefill rows fallback below the H256 wave-prefill owner. It is the owning
runner's ``qsa_wave32`` field now:

- ``True``  - the wave32 paged sparse-attention kernels, the promoted production
  arithmetic for head_dim 128 and the old env default.
- ``False`` - the wave64 parents, which stay the registered strict fallback.

Only the transport moves. The geometry guard (``head_dim == 128``), the H256
preemptor's ordering, the decode ``rows == 1`` precondition and the strict
fallback assignment are unchanged, and the attention is never inside a captured
graph, so no graph key carries this selection. These tests pin the field, the
scoped control, both dispatch sites, the forwarding chain (c1 decode and prefill
chunks), and that no process key carries the selection.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as runner_module

RETIRED_ENV = "HIPENGINE_QWEN4_EXP_QSA_WAVE32"
FIELD = "qsa_wave32"
ROOT = Path(__file__).resolve().parents[1]

#: The modules that carried the process key.
RETIRED_KEY_FREE_MODULES = (
    ROOT / "hipengine" / "runtime" / "qwen4_exp_runner.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_mtp_prefill.py",
    ROOT / "hipengine" / "runtime" / "qwen4_exp_batch_qsa.py",
    ROOT / "scripts" / "qwen4_exp_qsa_retrieval_gate.py",
)

DECODE_MIXER = "run_qwen4_exp_dense_qsa_token_mixer"
PREFILL_MIXER = "run_qwen4_exp_qsa_prefill_token_mixer"
LAYER_OWNERS = (
    "run_qwen4_exp_dense_qsa_layer",
    "run_qwen4_exp_qsa_prefill_layer",
)


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV, raising=False)


# --- the decision and its scoped control ------------------------------------


def test_runner_defaults_to_the_wave32_owners() -> None:
    """The old env default was "1"; an unbound runner keeps that arithmetic."""

    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    assert getattr(runner, FIELD) is True


def test_scope_binds_the_wave64_parent_and_restores() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(runner, False) as prior:
        assert prior is True
        assert getattr(runner, FIELD) is False
        with getattr(runner_module, FIELD)(runner, True):
            assert getattr(runner, FIELD) is True
        assert getattr(runner, FIELD) is False
    assert getattr(runner, FIELD) is True


def test_scope_leaves_an_unset_binding_alone() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(runner):
        assert getattr(runner, FIELD) is True
    assert getattr(runner, FIELD) is True


def test_scope_restores_after_an_exception() -> None:
    runner = SimpleNamespace(**{FIELD: True})

    with pytest.raises(RuntimeError, match="boom"):
        with getattr(runner_module, FIELD)(runner, False):
            raise RuntimeError("boom")
    assert getattr(runner, FIELD) is True


def test_scope_is_per_runner() -> None:
    first = SimpleNamespace(**{FIELD: True})
    second = SimpleNamespace(**{FIELD: True})

    with getattr(runner_module, FIELD)(first, False):
        assert getattr(second, FIELD) is True
        with getattr(runner_module, FIELD)(second, False):
            assert getattr(first, FIELD) is False
            assert getattr(second, FIELD) is False
        assert getattr(second, FIELD) is True
    assert getattr(first, FIELD) is True


# --- the dispatch-time guard ------------------------------------------------


@pytest.mark.parametrize(
    ("head_dim", "published", "expected"),
    [
        (128, True, True),    # production H128: the wave32 owners
        (128, False, False),  # diagnostic arm: the wave64 parents
        (256, True, False),   # H256 keeps its own owners
        (256, False, False),
        (64, True, False),    # any other geometry keeps the parent
        (128, True, True),    # a repeat keeps the decision (not memoized off)
    ],
)
def test_the_guard_keeps_the_h128_geometry(head_dim, published, expected) -> None:
    assert runner_module._qwen4_exp_qsa_wave32_selected(
        head_dim=head_dim, published=published) is expected


def test_the_decode_mixer_reads_the_parameter_and_no_process_key() -> None:
    parameters = inspect.signature(
        runner_module.run_qwen4_exp_dense_qsa_token_mixer).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module.run_qwen4_exp_dense_qsa_token_mixer)
    assert RETIRED_ENV not in source
    # The sparse arm is still the only place the selection applies, and the
    # decode precondition is untouched.
    assert "requires rows == 1" in source
    assert "published=qsa_wave32" in source


def test_the_prefill_mixer_reads_the_parameter_and_no_process_key() -> None:
    parameters = inspect.signature(
        runner_module.run_qwen4_exp_qsa_prefill_token_mixer).parameters
    assert FIELD in parameters
    assert parameters[FIELD].default is True

    source = inspect.getsource(runner_module.run_qwen4_exp_qsa_prefill_token_mixer)
    assert RETIRED_ENV not in source
    assert "published=qsa_wave32" in source


def test_the_prefill_fallback_keeps_the_h256_preemptor_first() -> None:
    """The H256 wave-prefill owner is resolved before the H128 fallback."""

    source = inspect.getsource(runner_module.run_qwen4_exp_qsa_prefill_token_mixer)
    h256 = source.index("qwen4_exp_qsa_h256_wave_prefill_selected(")
    h128 = source.index("_qwen4_exp_qsa_wave32_selected(")
    assert h256 < h128
    # The H256 owner heads the chain and the H128 fallback is its ``elif``,
    # never an override above it.
    assert source[:h256].rstrip().endswith("if")
    assert not source[:h256].rstrip().endswith("elif")
    assert source[h256:h128].rstrip().endswith("elif")


def test_both_sites_keep_the_wave64_parent_as_the_fallback() -> None:
    """Turning the selection off returns to the registered strict parents."""

    decode = inspect.getsource(runner_module.run_qwen4_exp_dense_qsa_token_mixer)
    assert (
        "sparse_attention = qwen4_exp_qsa_sparse_attention_paged_bf16_f32"
        in decode
    )
    assert (
        "sparse_attention = "
        "qwen4_exp_qsa_sparse_attention_paged_bf16_wave32_f32"
        in decode
    )

    prefill = inspect.getsource(runner_module.run_qwen4_exp_qsa_prefill_token_mixer)
    assert (
        "sparse_attention_rows = "
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32" in prefill
    )
    assert (
        "qwen4_exp_qsa_sparse_attention_paged_bf16_rows_wave32_f32" in prefill
    )


# --- forwarding -------------------------------------------------------------


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


@pytest.mark.parametrize(
    "function",
    [
        getattr(runner_module, name) for name in LAYER_OWNERS
    ],
)
def test_every_layer_owner_carries_the_selection(function) -> None:
    parameters = inspect.signature(function).parameters
    assert FIELD in parameters, function.__qualname__
    assert parameters[FIELD].default is True, function.__qualname__

    mixers = (
        DECODE_MIXER if function.__name__ == "run_qwen4_exp_dense_qsa_layer"
        else PREFILL_MIXER
    )
    calls = _calls_named(inspect.getsource(function), (mixers,))
    assert len(calls) == 1, function.__qualname__
    value = _keyword(calls[0], FIELD)
    assert isinstance(value, ast.Name), ast.dump(calls[0])
    assert value.id == FIELD


def test_the_runner_call_sites_forward_the_field() -> None:
    """c1 decode and the prefill chunk path read the runner's own decision."""

    forwarded = []
    for name in ("step", "_prefill_chunk_steps"):
        source = textwrap.dedent(
            inspect.getsource(
                getattr(runner_module.Qwen4ExpGGUFResidentModelRunner, name)))
        calls = _calls_named(source, LAYER_OWNERS)
        assert len(calls) == 1, name
        value = _keyword(calls[0], FIELD)
        assert isinstance(value, ast.Attribute), (name, ast.dump(calls[0]))
        assert value.attr == FIELD, name
        forwarded.append(name)
    assert forwarded == ["step", "_prefill_chunk_steps"]


def test_the_attention_is_never_inside_the_captured_moe_graph() -> None:
    """The MoE graph unit is the FFN, so no graph key carries this selection."""

    source = inspect.getsource(runner_module.run_qwen4_exp_dense_qsa_layer)
    mixer = _calls_named(source, (DECODE_MIXER,))
    assert len(mixer) == 1
    assert _keyword(mixer[0], "graph_cache") is None
    assert _keyword(mixer[0], "graph_key") is None

    moe = _calls_named(source, ("run_qwen4_exp_moe",))
    assert len(moe) == 1
    assert _keyword(moe[0], "graph_cache") is not None
    assert _keyword(moe[0], "graph_key") is not None


def test_no_layer_graph_segment_covers_the_attention() -> None:
    """The layer graph is GDN-only; its segments are GR launches, not attention.

    A captured segment bakes in whichever owner it launched, so a graph that
    covered the QSA attention would have to key on this selection. Neither QSA
    layer owner accepts a layer graph at all, and the c1 QSA call site passes
    none, so the wave32 policy never enters a graph identity.
    """

    for name in LAYER_OWNERS:
        parameters = inspect.signature(getattr(runner_module, name)).parameters
        assert "layer_graph_cache" not in parameters, name
        assert "layer_graph_key" not in parameters, name

    gdn = inspect.getsource(runner_module.run_qwen4_exp_gdn_layer)
    assert "layer_graph_key" in inspect.signature(
        runner_module.run_qwen4_exp_gdn_layer).parameters
    for segment in ("gr_read_attn", "gr_write_ffn_read", "epilogue"):
        assert f'"{segment}"' in gdn, segment
    # The captured segments wrap GR launches around the mixer, never a mixer
    # dispatch of its own.
    assert "run_qwen4_exp_gdn_token_mixer(" in gdn
    assert DECODE_MIXER not in gdn
    assert PREFILL_MIXER not in gdn

    step = textwrap.dedent(
        inspect.getsource(runner_module.Qwen4ExpGGUFResidentModelRunner.step))
    qsa = [
        call for call in _calls_named(step, LAYER_OWNERS)
        if _keyword(call, "index_state") is not None
    ]
    assert len(qsa) == 1
    assert _keyword(qsa[0], "layer_graph_cache") is None
    assert _keyword(qsa[0], "layer_graph_key") is None


def test_the_batch_and_mixed_owners_keep_their_own_attention_owner() -> None:
    """The N-row owners dispatch the wave64 rows parent, not this selection."""

    from hipengine.runtime import qwen4_exp_batch_qsa as batch_qsa

    for function in (
        batch_qsa.run_qwen4_exp_batch_dense_qsa_token_mixer,
        batch_qsa.run_qwen4_exp_batch_qsa_mixed_token_mixer,
    ):
        parameters = inspect.signature(function).parameters
        assert FIELD not in parameters, function.__qualname__
        assert RETIRED_ENV not in inspect.getsource(function)

    source = inspect.getsource(batch_qsa)
    assert "rows_wave32" not in source
    assert "sparse_attention_paged_bf16_rows_f32" in source


def test_the_verifier_prefill_is_cache_only() -> None:
    """The MTP verifier prefill appends cache state and dispatches no attention."""

    from hipengine.runtime import qwen4_exp_mtp_prefill

    source = inspect.getsource(qwen4_exp_mtp_prefill)
    calls = _calls_named(source, (PREFILL_MIXER,))
    assert len(calls) == 1
    assert isinstance(_keyword(calls[0], "cache_only"), ast.Constant)
    assert _keyword(calls[0], "cache_only").value is True


# --- the retired key --------------------------------------------------------


def test_no_process_key_carries_the_selection() -> None:
    for path in RETIRED_KEY_FREE_MODULES:
        assert RETIRED_ENV not in path.read_text(), path
