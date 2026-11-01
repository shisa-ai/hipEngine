"""The Q8 DPP comparison arm's process key belongs to its caller.

``HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION`` is read by the runtime with a ``"1"``
default and is never published by the profile binder. R0442 keeps it as a
maintained default-on comparison interface: ``0``
restores the shuffle tree and anything else keeps the VALU DPP reduction. The
``q8-dpp-reduction`` package therefore writes that key itself for each measured
arm - ``0`` for the control, ``1`` for the enabled arm - and ``main()`` has to
put the caller's own value back when it returns.

The cleanup used to write ``"1"`` unconditionally, so a caller who had exported
``0`` got the gate's binding back instead of their own, and a caller who had set
nothing got a key they never set. The premise also refused a caller's ``0``
outright, by asserting on a value the gate had just written itself
(``os.environ.setdefault(flag, "1")`` followed by ``assert os.environ[flag] ==
"1"``), which made the family's own off value unusable as a starting point.

These tests drive the real ``main()`` with a CPU fake whose prefill calls the
registered leaf the arm counts - the generic DPP parent - so the arm's own
``0``/``1`` values, the count checks and both cleanup paths run without a GPU.
Which DPP sibling the production dispatch rewrites that parent to is a separate
question from this lifetime contract, so the fake resolves the counted key
directly.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey, register, resolve
from scripts import qwen4exp_row4_state_gate as gate
from tests.test_unit_qwen4exp_state_gate_main_paths import (
    MAIN_OWNED_KEYS,
    _install_generator,
    _run_main,
    registry,
)

DPP_ENV = "HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION"

#: The leaf this package counts: the DPP sibling of the pipelined wave-scale
#: parent, and the only kernel ``main()`` resolves for the arm.
DPP_KEY = KernelKey(
    "hip_gfx1151", "linear", "gguf_q8_0",
    "coltile8_rowbatch4_wave_scale_pipelined_dpp_f32_f32_out")

DPP_CHECKED_KEY = KernelKey(
    DPP_KEY.backend, DPP_KEY.layer, DPP_KEY.quant,
    'coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out')

#: A case the default selection keeps (512 tokens, not a heldout).
CASE = {"id": "code-p512", "category": "code", "prompt_tokens": 512,
        "prompt_token_ids": [1] * 512}

#: The three states a caller can hand the gate: the family off, on, or unset.
PRIORS = (None, "0", "1")


def _flat_state(runner):
    return {"finite": True, "state_sha256": "s" * 8, "layout_sha256": "l" * 8}


class _DppRunner:
    """A CPU runner whose prefill runs the leaf the arm's key selects.

    ``prefill`` resolves the kernel at call time, so the gate's own counter
    wrapper is what observes the call and the arm's key value is what decides
    whether it launches at all - the shape the dispatch rewrite applies.
    """

    CALLS_PER_PREFILL = 8

    def __init__(self) -> None:
        self.q8_wave_scale_prefill = True
        self.q8_mmq_prefill_enabled = False
        self.prefill_chunk_size = 512
        self.runtime = SimpleNamespace(device_synchronize=lambda: None)
        #: The key's value inside each measured prefill, in call order.
        self.observed: list[str | None] = []
        #: Raise inside the measured window instead of returning.
        self.fail = False

    def _leaf(self):
        return resolve(backend=DPP_KEY.backend, layer=DPP_KEY.layer,
                       quant=DPP_KEY.quant, variant=DPP_KEY.variant)

    def prefill(self, token_ids):
        self.observed.append(os.environ.get(DPP_ENV))
        if self.fail:
            raise RuntimeError("prefill failed")
        if os.environ.get(DPP_ENV, "1") != "0":
            leaf = self._leaf()
            for _ in range(self.CALLS_PER_PREFILL):
                leaf()
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32), token_id=0)

    def step(self, token):
        return SimpleNamespace(logits=np.zeros(4, dtype=np.float32), token_id=1)


@pytest.fixture
def dpp_leaf(registry):
    """Registered stubs for both leaves controlled by the family key."""

    register(DPP_KEY, lambda *a, **kw: None, replace=True)
    register(DPP_CHECKED_KEY, lambda *a, **kw: None, replace=True)
    yield


def _start_from(monkeypatch, prior):
    """Hand the gate ``prior``, or no value at all."""

    if prior is None:
        monkeypatch.delenv(DPP_ENV, raising=False)
    else:
        monkeypatch.setenv(DPP_ENV, prior)


@pytest.mark.parametrize("prior", PRIORS)
def test_main_restores_the_callers_dpp_value_on_success(
        dpp_leaf, monkeypatch, tmp_path, prior):
    """The arms bind their own 0/1 and the caller's value comes back."""

    _start_from(monkeypatch, prior)
    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    before = set(os.environ)
    runner = _DppRunner()
    report = _run_main(monkeypatch, tmp_path, "q8-dpp-reduction", runner,
                       cases=[CASE])
    assert report["status"] == "passed"
    captures = report["cases"][0]["captures"]
    # The measured window is the arm's, never the caller's: the control arm
    # binds "0" even when the caller exported "1", and the enabled arm binds
    # "1" even when the caller exported "0".
    assert [row["enabled"] for row in captures] == ["0", "1", "0"]
    assert runner.observed == ["0", "1", "0"]
    # Engagement is still measured, not assumed.
    assert [row["candidate_calls"] for row in captures] == [
        0, _DppRunner.CALLS_PER_PREFILL, 0]
    # The caller's own state - including its absence - is what is left behind.
    assert os.environ.get(DPP_ENV) == prior
    assert set(os.environ) - before <= MAIN_OWNED_KEYS


@pytest.mark.parametrize("prior", PRIORS)
def test_main_restores_the_callers_dpp_value_when_the_prefill_fails(
        dpp_leaf, monkeypatch, tmp_path, prior):
    """A failed measured arm cannot leave the gate's binding in the caller."""

    _start_from(monkeypatch, prior)
    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    runner = _DppRunner()
    runner.fail = True
    with pytest.raises(RuntimeError, match="prefill failed"):
        _run_main(monkeypatch, tmp_path, "q8-dpp-reduction", runner,
                  cases=[CASE])
    # The control arm was bound and the run stopped inside it.
    assert runner.observed == ["0"]
    assert os.environ.get(DPP_ENV) == prior


@pytest.mark.parametrize('checked', [False, True])
def test_main_counts_both_dpp_family_members_and_restores_registry(
        dpp_leaf, monkeypatch, tmp_path, checked):
    _start_from(monkeypatch, None)
    monkeypatch.setattr(gate, '_state_summary', _flat_state)
    runner = _DppRunner()
    selected = DPP_CHECKED_KEY if checked else DPP_KEY
    original = resolve(backend=selected.backend, layer=selected.layer,
                       quant=selected.quant, variant=selected.variant)
    runner._leaf = lambda: resolve(backend=selected.backend, layer=selected.layer,
                                   quant=selected.quant, variant=selected.variant)
    report = _run_main(monkeypatch, tmp_path, 'q8-dpp-reduction', runner, cases=[CASE])
    assert [row['candidate_calls'] for row in report['cases'][0]['captures']] == [
        0, runner.CALLS_PER_PREFILL, 0]
    assert resolve(backend=selected.backend, layer=selected.layer,
                   quant=selected.quant, variant=selected.variant) is original


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_dpp_route_scope_exposes_family_and_restores_preemptors(enabled, raises):
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
    from hipengine.runtime.gguf_linear import Q8WMMALayerScope
    owner = SimpleNamespace(q8_wave_scale_prefill=False,
                            q8_mmq_prefill_enabled=True,
                            q8_wmma_layer_scope=Q8WMMALayerScope.parse('3'))
    prior_wmma = owner.q8_wmma_layer_scope
    env = {'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE': '1', DPP_ENV: 'caller'}
    prior = dict(env)
    def run():
        with route_scope('q8-dpp-reduction', 'after' if enabled else 'before',
                         owner, environment=env):
            assert env['HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE'] == '1'
            assert owner.q8_dense_wide_policy.enabled is False
            assert env['HIPENGINE_GGUF_WMMA_PREFILL'] == '0'
            assert env[DPP_ENV] == ('1' if enabled else '0')
            assert owner.q8_wave_scale_prefill is True
            assert owner.q8_mmq_prefill_enabled is False
            assert owner.q8_wmma_layer_scope.is_empty
            if raises:
                raise RuntimeError('scope failed')
    if raises:
        with pytest.raises(RuntimeError, match='scope failed'):
            run()
    else:
        run()
    assert env == prior
    assert owner.q8_wave_scale_prefill is False
    assert owner.q8_mmq_prefill_enabled is True
    assert owner.q8_wmma_layer_scope is prior_wmma


def test_checked_layer2_candidate_uses_the_shared_isolation_scope(monkeypatch):
    from contextlib import ExitStack
    from scripts import qwen4exp_layer2_profile_gate as layer2
    owner = SimpleNamespace(q8_wave_scale_prefill=False, q8_mmq_prefill_enabled=True)
    generator = SimpleNamespace(runner=owner)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '1')
    monkeypatch.setenv(DPP_ENV, 'caller')
    with ExitStack() as stack:
        layer2.bind_checked_q8_comparison(
            stack, generator, layer2.CANDIDATES['dense_q8_checked_full_rows'])
        assert os.environ['HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE'] == '1'
        assert owner.q8_dense_wide_policy.enabled is False
        assert os.environ[DPP_ENV] == '1'
        assert owner.q8_wave_scale_prefill is True
        assert owner.q8_mmq_prefill_enabled is False
    assert os.environ['HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE'] == '1'
    assert os.environ[DPP_ENV] == 'caller'
    assert owner.q8_wave_scale_prefill is False
    assert owner.q8_mmq_prefill_enabled is True


def _argv(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "qwen4exp_row4_state_gate.py",
        "--model-root", str(tmp_path),
        "--compiler-version-file", str(tmp_path / "compiler.txt"),
        "--output", str(tmp_path / "q8-dpp-reduction.json"),
        "--route-package", "q8-dpp-reduction",
    ])


def test_main_refuses_a_binder_that_starts_publishing_the_key(
        dpp_leaf, monkeypatch, tmp_path):
    """The captured value is the caller's: a later write is a wiring bug.

    Restoring whatever the key held at cleanup time would hide a profile binder
    that started publishing this key, so the premise refuses the run instead of
    reporting the comparison against a value the caller never set.
    """

    _start_from(monkeypatch, "0")
    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    _install_generator(monkeypatch, _DppRunner(), cases=[CASE])
    profile = gate.resolve_runtime_profile

    def resolving(**kwargs):
        resolved = profile(**kwargs)
        construct = resolved.construct_generator

        def binder(factory):
            os.environ[DPP_ENV] = "1"
            return construct(factory)

        resolved.construct_generator = binder
        return resolved

    monkeypatch.setattr(gate, "resolve_runtime_profile", resolving)
    _argv(monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="caller's value before the arms"):
        gate.main()
    # The refused run still puts the caller's value back.
    assert os.environ.get(DPP_ENV) == "0"


@pytest.mark.parametrize("prior", PRIORS)
def test_main_restores_the_callers_dpp_value_when_a_decode_step_fails(
        dpp_leaf, monkeypatch, tmp_path, prior):
    """The cleanup owns the key for the whole run, not just the prefill loop."""

    _start_from(monkeypatch, prior)
    monkeypatch.setattr(gate, "_state_summary", _flat_state)
    runner = _DppRunner()

    def step(token):
        raise RuntimeError("decode failed")

    runner.step = step
    with pytest.raises(RuntimeError, match="decode failed"):
        _run_main(monkeypatch, tmp_path, "q8-dpp-reduction", runner,
                  cases=[CASE])
    assert runner.observed == ["0"]
    assert os.environ.get(DPP_ENV) == prior
