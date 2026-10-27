"""The batch owner forwards the expert-down rungs and keys captures on them.

Two contracts are under test:

* ``step`` forwards the twelve plan-published expert-down decisions - the Q5_1
  exact grouped down chain gate, the F16 WMMA down owner below it, its out8
  fallback rung, the Q4_K exact grouped gate/up chain's gate and `_ALL` joint
  term, that chain's out4 fallback rung, the Q5_1 down M1 leaf, the two Q5_1
  fold rungs, the PF-1 fork-b grouped down and the two Q8_0 row4 rungs - to
  both layer owners unchanged, so the batch decode path selects the same
  leaves the c1 path selects.
* The captured ``post_mixer`` segment records the MoE launches, and the MoE runs
  eager inside it, so the selection is baked into the graph. The capture key
  therefore carries all twelve decisions, and the compact F16 Q4_K gate/up
  tile pair with them: a runner whose selection or tile pair changes
  captures a new segment instead of replaying one built under the old one.

The chain gates and the two fallback rungs differ from the rungs around them in
one way that matters here: every plan publishes them on, so an unbound batch
owner keeps them on while it keeps the production-only rungs off.
"""

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_ar_batch import QWEN4_EXP_MAX_BATCH_ROWS
from hipengine.runtime import qwen4_exp_batch_runner as runner_module
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from tests.test_unit_qwen4exp_batch_runner import _fake_runner, _install_step_fakes

#: The exact grouped Q5_1 down chain's own gate. It precedes the rungs whose
#: chain it gates, and it is one of the two decisions every plan publishes on.
CHAIN_GATE_FIELD = "q51_exact_grouped_down"
#: The chain's non-expertgrid fallback rung, published on by every plan.
OUT8_FIELD = "q51_out8"
#: The Q4_K exact grouped gate/up chain's own gate and its `_ALL` joint term,
#: published on by every plan. They select the gate/up owner the captured segment
#: launches, so they belong in the key like the Q5_1 chain gate above.
Q4_GATE_FIELD = "q4_exact_grouped_gate"
Q4_GATE_ALL_FIELD = "q4_exact_grouped_all"
#: The Q4 gate/up ladder's non-expertgrid fallback rung, published on by every
#: plan. It bakes its own gate/up launch into the captured segment, so it belongs
#: in the key like the Q5_1 rungs above it.
Q4_OUT4_FIELD = "q4_out4"
#: The F16 WMMA down owner below the Q5_1 chain, published off by every plan.
#: Decode cannot reach the production admission that also selects it, so a
#: captured decode segment bakes its launch in and the key must say so.
WMMA_DOWN_FIELD = "q51_wmma_down"
#: The twelve decisions, in the order the runner publishes them. The order is the
#: contract: ``_expert_down_policy`` and the ``(layer, rows) + policy`` capture
#: key both read this sequence, and ``test_the_policy_tuple_is_boolean_normalized``
#: pins it against the runner's own tuple. The compact F16 Q4_K gate/up tile
#: pair follows the tuple as ``_q4_tile_policy()``, so it is not listed here.
POLICY_FIELDS = (
    "q51_down_m1",
    "q51_fold128_prefill",
    "q51_fold_pair_prefill",
    CHAIN_GATE_FIELD,
    WMMA_DOWN_FIELD,
    OUT8_FIELD,
    Q4_GATE_FIELD,
    Q4_GATE_ALL_FIELD,
    Q4_OUT4_FIELD,
    "forkb_grouped_down",
    "q8_down_row4_prefill",
    "q8_down_bundle_prefill",
)
#: The per-decision default a ``getattr`` fallback uses, which is the value an
#: unbound owner carries: the production-only rungs stay off, the chain gates and
#: the two fallback rungs stay on.
UNBOUND_DEFAULTS = {
    field: field in (
        CHAIN_GATE_FIELD, OUT8_FIELD, Q4_GATE_FIELD, Q4_GATE_ALL_FIELD,
        Q4_OUT4_FIELD)
    for field in POLICY_FIELDS
}


def _recording_layers(monkeypatch, *, publish: bool) -> list[tuple]:
    """Replace both layer owners with recorders of what ``step`` forwarded."""

    _install_step_fakes(monkeypatch)
    seen: list[tuple] = []

    def layer(name):
        def run(residual_ptr, weights, *, rows, **kwargs):
            seen.append(
                (name, int(rows), tuple(kwargs.get(field) for field in POLICY_FIELDS),
                 kwargs.get("graph_key"))
            )
            return SimpleNamespace(ptr=residual_ptr + 0x10)

        return run

    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_batch_gdn_layer", layer("gdn"))
    monkeypatch.setattr(
        runner_module, "run_qwen4_exp_batch_qsa_layer", layer("qsa"))
    return seen


def _runner(*, slots: int, published: bool) -> Qwen4ExpGGUFBatchRunner:
    runner = _fake_runner(slots=slots)
    for field in POLICY_FIELDS:
        setattr(runner, field, published)
    return runner


def _step(runner, rows: int):
    """Admit every slot, run one step, then release the owners again."""

    owners = [runner.admit(slot) for slot in range(rows)]
    try:
        return runner.step(
            [1] * rows,
            slots=list(range(rows)),
            owners=owners,
            capture_logits=False,
        )
    finally:
        for slot, owner in enumerate(owners):
            runner.release(slot, owner)


@pytest.mark.parametrize("rows", [1, 63, QWEN4_EXP_MAX_BATCH_ROWS])
@pytest.mark.parametrize("published", [False, True])
def test_step_forwards_every_rung_to_both_layer_owners(
        monkeypatch, rows, published):
    seen = _recording_layers(monkeypatch, publish=published)
    _step(_runner(slots=rows, published=published), rows)
    assert seen, "the step dispatched no layer"
    for name, width, policy, _key in seen:
        assert width == rows, (name, width)
        assert policy == (published,) * len(POLICY_FIELDS), (name, policy)


def test_the_production_batch_width_is_the_only_width_that_reaches_fold128():
    # The fold128 rung selects at rows>=64 and a batch tops out at 64 rows, so
    # batch64 is the one production width where the forwarded value is live.
    # The Q5_1 down M1 and PF-1 fork-b decisions are grouped-prefill selectors,
    # so they are live at every multi-row width and inert at rows==1. The chain
    # gate is live wherever the grouped chain runs, which is every multi-row
    # width.
    assert QWEN4_EXP_MAX_BATCH_ROWS == 64
    assert len(POLICY_FIELDS) == 12


def test_step_keys_each_capture_on_the_expert_down_policy(monkeypatch):
    seen = _recording_layers(monkeypatch, publish=True)
    runner = _runner(slots=QWEN4_EXP_MAX_BATCH_ROWS, published=True)
    _step(runner, QWEN4_EXP_MAX_BATCH_ROWS)
    keys = [key for _name, _width, _policy, key in seen]
    assert keys and all(key is not None for key in keys)
    for _name, width, policy, key in seen:
        # ``(layer, rows)`` plus the twelve decisions in publication order,
        # then the compact F16 Q4_K gate/up tile pair the segment bakes in.
        assert key[1] == width
        assert key[2:2 + len(POLICY_FIELDS)] == policy
        assert key[2 + len(POLICY_FIELDS):] == (
            runner._q4_tile_policy() + runner._q8_tile_policy())


@pytest.mark.parametrize("field", ("q4_tile_m", "q4_tile_n"))
def test_a_flipped_tile_changes_the_capture_key(monkeypatch, field):
    # The compact F16 Q4_K gate/up owner resolves its tile pair at capture
    # time too, so a segment built under one pair must not replay under
    # another.
    seen = _recording_layers(monkeypatch, publish=True)
    runner = _runner(slots=QWEN4_EXP_MAX_BATCH_ROWS, published=True)
    for name in ("q4_tile_m", "q4_tile_n"):
        setattr(runner, name, 16)
    _step(runner, QWEN4_EXP_MAX_BATCH_ROWS)
    baseline = [key for _name, _width, _policy, key in seen]
    seen.clear()
    setattr(runner, field, 32)
    _step(runner, QWEN4_EXP_MAX_BATCH_ROWS)
    flipped = [key for _name, _width, _policy, key in seen]
    assert flipped, "the second step dispatched no layer"
    for before, after in zip(baseline, flipped):
        assert before != after, field
        assert after[-(len(runner._q4_tile_policy()) + len(runner._q8_tile_policy())):] == (
            runner._q4_tile_policy() + runner._q8_tile_policy())


@pytest.mark.parametrize("field", POLICY_FIELDS)
def test_a_flipped_rung_changes_the_capture_key(monkeypatch, field):
    # The failure this guards: capture under the published selection, flip the
    # rung inside a diagnostic scope, and replay the segment the old selection
    # built. A distinct key makes the second step capture instead.
    seen = _recording_layers(monkeypatch, publish=True)
    runner = _runner(slots=QWEN4_EXP_MAX_BATCH_ROWS, published=True)
    _step(runner, QWEN4_EXP_MAX_BATCH_ROWS)
    baseline = [key for _name, _width, _policy, key in seen]
    seen.clear()
    setattr(runner, field, False)
    _step(runner, QWEN4_EXP_MAX_BATCH_ROWS)
    flipped = [key for _name, _width, _policy, key in seen]
    assert flipped, "the second step dispatched no layer"
    for before, after in zip(baseline, flipped):
        assert before != after, field
        assert after[2:2 + len(POLICY_FIELDS)] == tuple(
            False if name == field else True for name in POLICY_FIELDS)


def test_an_unbound_runner_forwards_every_parent():
    # The class defaults are the parents, exactly like an unbound c1 runner,
    # except for the chain gate and its out8 fallback: every plan publishes both
    # on, so an unbound owner keeps the production chain rather than closing it
    # or dropping to the output1 base.
    runner = object.__new__(Qwen4ExpGGUFBatchRunner)
    assert runner._expert_down_policy() == tuple(
        UNBOUND_DEFAULTS[field] for field in POLICY_FIELDS)
    assert runner._expert_down_policy()[POLICY_FIELDS.index(CHAIN_GATE_FIELD)]


def test_the_policy_tuple_is_boolean_normalized():
    runner = object.__new__(Qwen4ExpGGUFBatchRunner)
    runner.q51_down_m1 = 1
    runner.q51_fold128_prefill = "yes"
    runner.q51_fold_pair_prefill = None
    runner.q51_exact_grouped_down = 0
    runner.q51_wmma_down = "1"
    runner.q51_out8 = "1"
    runner.q4_exact_grouped_gate = "on"
    runner.q4_exact_grouped_all = 0
    runner.q4_out4 = 0
    runner.forkb_grouped_down = 0
    runner.q8_down_row4_prefill = 1
    runner.q8_down_bundle_prefill = ""
    assert runner._expert_down_policy() == (
        True, True, False, False, True, True, True, False, False, False, True,
        False)
    # The tuple is the capture key's tail, so its length is the contract a
    # reader of a recorded segment relies on.
    assert runner._expert_down_policy() == tuple(
        bool(getattr(runner, field)) for field in POLICY_FIELDS)


# --- target/draft sharing: one publication, every owner reads it -------------


def test_a_c1_runner_built_later_inherits_the_generators_rungs():
    # The binder publishes to the generator; the serving pool clones c1 staging
    # runners on demand. Those clones must read the same twelve decisions, or a
    # speculative draft would measure the parents while its target ran the
    # leaves.
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    generator = SimpleNamespace(
        **{field: True for field in POLICY_FIELDS})
    first = _inherit_lane_decisions(generator, SimpleNamespace())
    second = _inherit_lane_decisions(generator, SimpleNamespace())
    for field in POLICY_FIELDS:
        assert getattr(first, field) is True
        assert getattr(second, field) is True


def test_an_unbound_generator_publishes_the_parents_to_a_late_clone():
    from hipengine.generation.qwen4_exp_gguf import _inherit_lane_decisions

    clone = _inherit_lane_decisions(SimpleNamespace(), SimpleNamespace())
    for field in POLICY_FIELDS:
        assert getattr(clone, field) is UNBOUND_DEFAULTS[field], field


def test_the_mixed_pass_reads_the_rungs_from_the_shared_batch_runner():
    # ``verify`` runs the draft and the target rows through the same batch
    # owner, so the selection has to come from that owner rather than from a
    # second copy on the pass.
    import ast
    import inspect
    import textwrap

    from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass

    # ``verify`` is the execution entry and binds the batch's Q8_0 tile pair
    # around the row work; the rung forwarding itself lives in ``_verify_rows``.
    assert "self._verify_rows()" in inspect.getsource(
        Qwen4ExpMixedTargetPass.verify)
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Qwen4ExpMixedTargetPass._verify_rows)))
    forwarded: list[set[str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_run_layer_ffn"):
            continue
        names = set()
        for keyword in node.keywords:
            if keyword.arg not in POLICY_FIELDS:
                continue
            value = keyword.value
            assert isinstance(value, ast.Call), keyword.arg
            assert getattr(value.func, "id", None) == "getattr", keyword.arg
            assert getattr(value.args[0], "id", None) == "batch", keyword.arg
            assert value.args[1].value == keyword.arg, keyword.arg
            assert value.args[2].value is UNBOUND_DEFAULTS[keyword.arg], (
                keyword.arg)
            names.add(keyword.arg)
        forwarded.append(names)
    assert forwarded, "the pass no longer calls the layer FFN"
    for names in forwarded:
        assert names >= set(POLICY_FIELDS), names
