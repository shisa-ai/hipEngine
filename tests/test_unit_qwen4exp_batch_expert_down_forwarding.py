"""The batch owner forwards the expert-down rungs and keys captures on them.

Two contracts are under test:

* ``step`` forwards the four plan-published expert-down decisions to both layer
  owners unchanged, so the batch decode path selects the same leaves the c1
  path selects.
* The captured ``post_mixer`` segment records the MoE launches, and the MoE runs
  eager inside it, so the selection is baked into the graph. The capture key
  therefore carries the four decisions: a runner whose selection changes
  captures a new segment instead of replaying one built under the old
  selection.
"""

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_ar_batch import QWEN4_EXP_MAX_BATCH_ROWS
from hipengine.runtime import qwen4_exp_batch_runner as runner_module
from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner
from tests.test_unit_qwen4exp_batch_runner import _fake_runner, _install_step_fakes

#: The four decisions, in the order the runner publishes them.
POLICY_FIELDS = (
    "q51_fold128_prefill",
    "q51_fold_pair_prefill",
    "q8_down_row4_prefill",
    "q8_down_bundle_prefill",
)


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
    assert QWEN4_EXP_MAX_BATCH_ROWS == 64


def test_step_keys_each_capture_on_the_expert_down_policy(monkeypatch):
    seen = _recording_layers(monkeypatch, publish=True)
    runner = _runner(slots=QWEN4_EXP_MAX_BATCH_ROWS, published=True)
    _step(runner, QWEN4_EXP_MAX_BATCH_ROWS)
    keys = [key for _name, _width, _policy, key in seen]
    assert keys and all(key is not None for key in keys)
    for _name, width, policy, key in seen:
        # ``(layer, rows)`` plus the four decisions, in publication order.
        assert key[1] == width
        assert key[2:] == policy


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
        assert after[2:] == tuple(
            False if name == field else True for name in POLICY_FIELDS)


def test_an_unbound_runner_forwards_every_parent():
    # The class defaults are the parents, exactly like an unbound c1 runner.
    runner = object.__new__(Qwen4ExpGGUFBatchRunner)
    assert runner._expert_down_policy() == (False,) * len(POLICY_FIELDS)


def test_the_policy_tuple_is_boolean_normalized():
    runner = object.__new__(Qwen4ExpGGUFBatchRunner)
    runner.q51_fold128_prefill = 1
    runner.q51_fold_pair_prefill = "yes"
    runner.q8_down_row4_prefill = None
    runner.q8_down_bundle_prefill = 0
    assert runner._expert_down_policy() == (True, True, False, False)


# --- target/draft sharing: one publication, every owner reads it -------------


def test_a_c1_runner_built_later_inherits_the_generators_rungs():
    # The binder publishes to the generator; the serving pool clones c1 staging
    # runners on demand. Those clones must read the same four decisions, or a
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
        assert getattr(clone, field) is False


def test_the_mixed_pass_reads_the_rungs_from_the_shared_batch_runner():
    # ``verify`` runs the draft and the target rows through the same batch
    # owner, so the selection has to come from that owner rather than from a
    # second copy on the pass.
    import ast
    import inspect
    import textwrap

    from hipengine.runtime.qwen4_exp_mixed_target import Qwen4ExpMixedTargetPass

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(Qwen4ExpMixedTargetPass.verify)))
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
            assert value.args[2].value is False, keyword.arg
            names.add(keyword.arg)
        forwarded.append(names)
    assert forwarded, "the pass no longer calls the layer FFN"
    for names in forwarded:
        assert names >= set(POLICY_FIELDS), names
