"""The compact F16 Q4_K gate/up tile pair is request-local runner state.

``HIPENGINE_QWEN4_EXP_Q4_TILE_M`` / ``HIPENGINE_QWEN4_EXP_Q4_TILE_N`` were the
process-wide transport for one launch shape: the tile pair handed to the compact
F16 Q4_K dual gate/up fallback owner
(``gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out``), which runs
below the exact grouped Q4_K chain and below the iu8 owner. They are the owning
runner's ``q4_tile_m`` / ``q4_tile_n`` fields now, and both defaults survive:

- the profile binder publishes ``16`` / ``16`` for every plan, exactly as the
  environment used to be pinned;
- an unbound runner (no binder) keeps the historical heuristic: ``M`` is 64 from
  512 rows up and 16 below it, ``N`` is 16.

Only the transport moves. The owner, its fallback position, the allowed-tile
validation and the kernel arithmetic are untouched, and the dense Q8 tile pair
(``HIPENGINE_GGUF_Q8_0_WMMA_TILE_M/N``) is a separate multi-consumer contract
that stays a process key.

The batch capture key carries the pair with the other expert-down decisions
so multi-row segments cannot reuse launches built under a different tile pair.
Every carrier forwards the pair.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

import pytest

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation import qwen4_exp_ar_batch, qwen4_exp_gguf
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND,
    QWEN4_EXP_MODEL,
    register_qwen4_exp_gfx1151_profiles,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_q4_k_selected_prefill
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_batch_runner as batch_module
from hipengine.runtime import qwen4_exp_mixed_target as mixed_module
from hipengine.runtime import qwen4_exp_runner as runner_module

RETIRED_ENV_M = "HIPENGINE_QWEN4_EXP_Q4_TILE_M"
RETIRED_ENV_N = "HIPENGINE_QWEN4_EXP_Q4_TILE_N"
FIELD_M = "q4_tile_m"
FIELD_N = "q4_tile_n"
FIELDS = (FIELD_M, FIELD_N)
ROOT = Path(__file__).resolve().parents[1]

#: The owner the pair is passed to, and the owner that must not receive it.
COMPACT_OWNER = "gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out"
IU8_OWNER = "gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out"

#: The dense Q8 pair is a separate contract and must stay a process key.
DENSE_Q8_ENV = (
    "HIPENGINE_GGUF_Q8_0_WMMA_TILE_M",
    "HIPENGINE_GGUF_Q8_0_WMMA_TILE_N",
)

MOE_FORWARDERS = ("run_qwen4_exp_moe", "_run_layer_ffn")


@pytest.fixture(autouse=True)
def isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RETIRED_ENV_M, raising=False)
    monkeypatch.delenv(RETIRED_ENV_N, raising=False)


@pytest.fixture
def profiles():
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return None


# --- the decision -----------------------------------------------------------


def test_the_unbound_heuristic_is_unchanged() -> None:
    resolve = runner_module._qwen4_exp_q4_compact_tiles
    assert resolve(rows=1, tile_m=None, tile_n=None) == (16, 16)
    assert resolve(rows=511, tile_m=None, tile_n=None) == (16, 16)
    assert resolve(rows=512, tile_m=None, tile_n=None) == (64, 16)
    assert resolve(rows=1024, tile_m=None, tile_n=None) == (64, 16)


def test_a_published_pair_wins_over_the_heuristic() -> None:
    resolve = runner_module._qwen4_exp_q4_compact_tiles
    # The binder's publication is 16/16, which is what every plan ran before.
    assert resolve(rows=1024, tile_m=16, tile_n=16) == (16, 16)
    assert resolve(rows=128, tile_m=32, tile_n=32) == (32, 32)
    # Each side resolves on its own, like the two keys it replaces.
    assert resolve(rows=1024, tile_m=None, tile_n=32) == (64, 32)
    assert resolve(rows=1024, tile_m=32, tile_n=None) == (32, 16)


def test_the_dispatch_reads_the_fields_and_no_process_key() -> None:
    parameters = inspect.signature(runner_module.run_qwen4_exp_moe).parameters
    for field in FIELDS:
        assert field in parameters
        assert parameters[field].default is None

    source = inspect.getsource(runner_module.run_qwen4_exp_moe)
    assert RETIRED_ENV_M not in source
    assert RETIRED_ENV_N not in source
    assert "_qwen4_exp_q4_compact_tiles(" in source

    tree = ast.parse(textwrap.dedent(source))
    owners = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {COMPACT_OWNER, IU8_OWNER}
    }
    assert set(owners) == {COMPACT_OWNER, IU8_OWNER}
    assert "rows=rows, tile_m=q4_tile_m, tile_n=q4_tile_n" in source
    compact = {kw.arg: kw.value for kw in owners[COMPACT_OWNER].keywords}
    for keyword, name in (("tile_m", "compact_tile_m"),
                          ("tile_n", "compact_tile_n")):
        value = compact[keyword]
        assert isinstance(value, ast.Name), (keyword, ast.dump(value))
        assert value.id == name
    # The iu8 owner has no tile parameters: the pair belongs to the F16 fallback.
    assert not {"tile_m", "tile_n"} & {
        kw.arg for kw in owners[IU8_OWNER].keywords}


def test_no_module_names_the_retired_keys() -> None:
    offenders = [
        str(path)
        for root in (ROOT / "hipengine", ROOT / "scripts")
        for path in root.rglob("*.py")
        if RETIRED_ENV_M in path.read_text() or RETIRED_ENV_N in path.read_text()
    ]
    assert offenders == []


def test_the_supported_tile_validation_is_unchanged() -> None:
    resolve = gguf_q4_k_selected_prefill._resolve_tiles
    assert resolve(16, 16) == (16, 16)
    assert resolve(64, 16) == (64, 16)
    with pytest.raises(ValueError, match="not supported"):
        resolve(48, 16)
    assert (64, 16) in gguf_q4_k_selected_prefill._ALLOWED_TILES


# --- the scoped control -----------------------------------------------------


def test_the_runner_defaults_both_sides_unbound() -> None:
    runner = object.__new__(runner_module.Qwen4ExpGGUFResidentModelRunner)
    for field in FIELDS:
        assert getattr(runner, field) is None


@pytest.mark.parametrize("field", FIELDS)
def test_scope_binds_one_side_and_restores(field) -> None:
    runner = SimpleNamespace(**{FIELD_M: None, FIELD_N: None})

    with getattr(runner_module, field)(runner, 32) as prior:
        assert prior is None
        assert getattr(runner, field) == 32
        with getattr(runner_module, field)(runner, 64):
            assert getattr(runner, field) == 64
        assert getattr(runner, field) == 32
    assert getattr(runner, field) is None


@pytest.mark.parametrize("field", FIELDS)
def test_scope_leaves_an_unset_binding_alone(field) -> None:
    runner = SimpleNamespace(**{FIELD_M: 16, FIELD_N: 16})

    with getattr(runner_module, field)(runner):
        assert getattr(runner, field) == 16
    assert getattr(runner, field) == 16


@pytest.mark.parametrize("field", FIELDS)
def test_scope_restores_after_an_exception(field) -> None:
    runner = SimpleNamespace(**{FIELD_M: 16, FIELD_N: 16})

    with pytest.raises(RuntimeError, match="arm failed"):
        with getattr(runner_module, field)(runner, 32):
            raise RuntimeError("arm failed")
    assert getattr(runner, field) == 16


def test_scope_is_per_runner() -> None:
    first = SimpleNamespace(**{FIELD_M: None, FIELD_N: None})
    second = SimpleNamespace(**{FIELD_M: None, FIELD_N: None})

    with getattr(runner_module, FIELD_M)(first, 64):
        assert getattr(second, FIELD_M) is None
        with getattr(runner_module, FIELD_M)(second, 32):
            assert getattr(first, FIELD_M) == 64
            assert getattr(second, FIELD_M) == 32
        assert getattr(second, FIELD_M) is None
    assert getattr(first, FIELD_M) is None


# --- the plan publication ---------------------------------------------------


def _resolve(profile: ExecutionProfile):
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl",
        profile=profile,
    )


@pytest.mark.parametrize(
    "profile", [ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION])
def test_the_binder_publishes_16_16_and_writes_no_process_key(
        profiles, profile) -> None:
    resolved = _resolve(profile)
    assert resolved.binder is not None
    generator = SimpleNamespace()
    resolved.binder(generator, resolved)

    assert getattr(generator, FIELD_M) == 16
    assert getattr(generator, FIELD_N) == 16
    assert RETIRED_ENV_M not in os.environ
    assert RETIRED_ENV_N not in os.environ


def test_the_binder_reaches_a_runner_it_finds(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    runner = SimpleNamespace()
    resolved.binder(SimpleNamespace(runner=runner), resolved)

    assert getattr(runner, FIELD_M) == 16
    assert getattr(runner, FIELD_N) == 16


def test_the_dense_q8_pair_stays_a_process_key(profiles) -> None:
    resolved = _resolve(ExecutionProfile.PRODUCTION)
    resolved.binder(SimpleNamespace(), resolved)
    assert os.environ[DENSE_Q8_ENV[0]] == "64"
    assert os.environ[DENSE_Q8_ENV[1]] == "32"
    source = inspect.getsource(
        gguf_q4_k_selected_prefill)  # the Q4_K module never reads them
    assert DENSE_Q8_ENV[0] not in source


def test_a_cloned_runner_inherits_the_published_pair() -> None:
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = SimpleNamespace()
    _inherit_lane_decisions(
        SimpleNamespace(**{FIELD_M: 32, FIELD_N: 32}), clone)
    assert (getattr(clone, FIELD_M), getattr(clone, FIELD_N)) == (32, 32)
    # An unbound carrier publishes the unbound pair, never the other lane's.
    _inherit_lane_decisions(SimpleNamespace(), clone)
    assert (getattr(clone, FIELD_M), getattr(clone, FIELD_N)) == (None, None)


def test_the_batch_owner_publishes_the_generators_pair() -> None:
    source = inspect.getsource(qwen4_exp_ar_batch)
    for field in FIELDS:
        assert f"batch.{field} = " in source
        assert f'getattr(generator, "{field}", None)' in source


# --- the capture key --------------------------------------------------------


def test_the_batch_capture_key_carries_the_tile_pair() -> None:
    """A captured decode segment bakes the compact F16 launch in."""

    owner = object.__new__(batch_module.Qwen4ExpGGUFBatchRunner)
    owner.q4_tile_m = 16
    owner.q4_tile_n = 16
    assert owner._q4_tile_policy() == (16, 16)

    owner.q4_tile_m = 64
    assert owner._q4_tile_policy() == (64, 16)

    source = inspect.getsource(
        batch_module.Qwen4ExpGGUFBatchRunner._execute_step)
    key = re.sub(r"\s+", " ", source)
    assert ("(layer, rows) + self._expert_down_policy() + "
            "self._q4_tile_policy()") in key
    assert key.count("self._q4_tile_policy()") == 2  # gdn and qsa segments


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
        runner_module.run_qwen4_exp_moe,
        runner_module.run_qwen4_exp_dense_qsa_layer,
        runner_module.run_qwen4_exp_qsa_prefill_layer,
        runner_module.run_qwen4_exp_gdn_layer,
        batch_module.run_qwen4_exp_batch_gdn_layer,
        batch_module.run_qwen4_exp_batch_qsa_layer,
        batch_module._run_layer_ffn,
    ],
)
def test_every_forwarder_carries_the_pair(function) -> None:
    parameters = inspect.signature(function).parameters
    for field in FIELDS:
        assert field in parameters, (function.__qualname__, field)
        assert parameters[field].default is None, (function.__qualname__, field)

    calls = _calls_named(inspect.getsource(function), MOE_FORWARDERS)
    assert calls, f"{function.__qualname__} forwards no MoE call"
    for call in calls:
        for field in FIELDS:
            value = _keyword(call, field)
            assert isinstance(value, ast.Name), (
                function.__qualname__, field, ast.dump(call))
            assert value.id == field


def test_the_mixed_pass_reads_the_pair_from_the_batch_owner() -> None:
    source = inspect.getsource(mixed_module)
    for field in FIELDS:
        assert f"{field}=getattr(" in source
        assert f'"{field}", None)' in source


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


def _gr_read(*args, **kwargs):
    return SimpleNamespace(normalized=None, gate=None,
                           mixed=SimpleNamespace(ptr=11),
                           inject_logits=SimpleNamespace(ptr=12))


def _moe_seen(seen):
    def moe(*args, **kwargs):
        seen.append((kwargs.get(FIELD_M), kwargs.get(FIELD_N)))
        return SimpleNamespace(output=SimpleNamespace(ptr=14))
    return moe


def _stub_common(monkeypatch, seen):
    monkeypatch.setattr(runner_module, "run_qwen4_exp_gr_read", _gr_read)
    monkeypatch.setattr(runner_module, "run_qwen4_exp_moe", _moe_seen(seen))
    monkeypatch.setattr(
        runner_module, "_qwen4_exp_gr_pending_write", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_f32", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "qwen4_exp_gr_write_bf16_bf16", lambda *a, **k: None)


def test_the_gdn_layer_forwards_the_pair_to_run_moe(monkeypatch) -> None:
    seen = []
    runtime = object()
    _stub_common(monkeypatch, seen)
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_gdn_token_mixer",
        lambda *a, **k: SimpleNamespace(ptr=13))
    monkeypatch.setattr(
        runner_module, "_qwen4_exp_gr_defer_write", lambda *a, **k: False)
    weights = _layer_weights()
    weights.mixer = SimpleNamespace(projections={}, conv_weight_ptr=5,
                                    dt_bias_ptr=6, a_ptr=7, norm_weight_ptr=8)
    kwargs = dict(conv_state_ptr=0, recurrent_state_ptr=0,
                  scratch=_layer_scratch(runtime), rows=512, branches=1,
                  hidden=2, low_rank=2, num_k_heads=16, num_v_heads=48,
                  head_dim=128, conv_kernel=4, ffn=8, experts=1, top_k=1,
                  runtime=runtime)
    runner_module.run_qwen4_exp_gdn_layer(
        100, weights, **kwargs, **{FIELD_M: 32, FIELD_N: 32})
    runner_module.run_qwen4_exp_gdn_layer(100, weights, **kwargs)
    assert seen == [(32, 32), (None, None)]
