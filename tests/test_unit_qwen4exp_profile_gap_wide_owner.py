"""Profile-gap wide overrides are explicit, observable and restoring."""
from types import SimpleNamespace
import pytest
from hipengine.runtime.gguf_linear import Q8DenseWidePolicy, resolve_q8_dense_wide_policy
from scripts import qwen4exp_profile_gap as gap


def test_parser_accepts_explicit_wide_controls():
    args = gap.build_parser().parse_args([
        '--model-root', '/tmp/model', '--mode', 'decode', '--output', '/tmp/out.json',
        '--q8-dense-wide', 'off', '--q8-dense-wide-layers', '0'])
    assert args.q8_dense_wide == 'off'
    assert args.q8_dense_wide_layers == '0'


@pytest.mark.parametrize('enabled,layers', [(False, None), (True, '0'), (None, '16,47')])
def test_override_preserves_unspecified_axis_and_restores(enabled, layers):
    prior = Q8DenseWidePolicy(True, frozenset({7}))
    owner = SimpleNamespace(q8_dense_wide_policy=prior)
    with gap._wide_owner_override(owner, enabled, layers):
        actual = resolve_q8_dense_wide_policy(owner)
        assert actual.enabled is (True if enabled is None else enabled)
        assert actual.layers == (prior.layers if layers is None else frozenset(map(int, layers.split(','))))
    assert owner.q8_dense_wide_policy is prior


def test_wide_override_marks_configuration_diagnostic():
    policy = Q8DenseWidePolicy(False, None)
    assert not gap._named_profile_intact({}, None, q8_wide_override=policy)
    assert gap._configuration_class({}, None, q8_wide_override=policy) == 'diagnostic_post_binder_override'
