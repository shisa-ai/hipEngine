"""The expert-down arm counts follow the loaded artifact, and fail loudly.

These arms verify a measured row by comparing the observed leaf calls to a
predicted count. A prediction that silently degrades to zero would let an arm
report ``0 == 0`` as engagement, so every input the count depends on - the
artifact's quant split, the runner's backend and expert width, the registry's
leaves, and the measured chunk - is either read from the loaded model or raised
on, never defaulted.
"""

from __future__ import annotations

import pytest

from hipengine.kernels import registry
from scripts.qwen4exp_halo_box_campaign_ab import (
    expert_down_layers,
    q51_fold128_expected_calls,
    q51_fold_pair_expected_calls,
    q51_register_cache_expected_calls,
    q51_row_publish_expected_calls,
    q8_down_bundle_expected_calls,
    q8_down_register_expected_calls,
    q8_down_row4_expected_calls,
    require_engaged_expectation,
)
from tests._qwen4exp_expert_down_fixtures import (
    SHIPPED_DOWN_QUANTS,
    expert_down_runner,
)


@pytest.fixture(autouse=True)
def _registered(monkeypatch):
    monkeypatch.setattr(registry, "is_registered", lambda key: True)


CASE = {"id": "code-p4096", "prompt_tokens": 4096}


# --- the count is the artifact's, not a constant ----------------------------


def test_counts_follow_the_loaded_quant_split():
    shipped = expert_down_runner()
    assert expert_down_layers(shipped, "gguf_q5_1") == 43
    assert expert_down_layers(shipped, "gguf_q8_0") == 5
    # A different artifact - here every layer on Q5_1 - counts its own layers.
    uniform = expert_down_runner(down_quants={n: "gguf_q5_1" for n in range(48)})
    assert q51_row_publish_expected_calls(uniform, 4096, 1024) == 48 * 4
    assert q51_row_publish_expected_calls(shipped, 4096, 1024) == 43 * 4


def test_the_explicit_shipped_split_matches_the_default_builder():
    assert expert_down_layers(expert_down_runner(down_quants=SHIPPED_DOWN_QUANTS),
                              "gguf_q5_1") == 43


def test_each_arm_counts_its_own_leaf_scope():
    owner = expert_down_runner()
    # Fold128 dispatches from 64 rows; its rows>=512 sibling from 512; the
    # register and row-publish leaves only at K640 and rows>=512.
    assert q51_fold128_expected_calls(owner, 128, 512) == 43
    assert q51_fold_pair_expected_calls(owner, 128, 512) == 0
    assert q51_register_cache_expected_calls(owner, 128, 512) == 0
    assert q51_row_publish_expected_calls(owner, 128, 512) == 0
    # Q8_0 leaves are the five Q8_0 layers, not the Q5_1 ones.
    assert q8_down_row4_expected_calls(owner, 512, 512) == 5
    assert q8_down_bundle_expected_calls(owner, 512, 512) == 5
    assert q8_down_register_expected_calls(owner, 512, 1024) > 0


def test_a_short_row_expects_no_calls():
    # A row below every leaf's guard is genuinely ineligible; that is the only
    # zero the helpers report.
    owner = expert_down_runner()
    assert q51_fold128_expected_calls(owner, 63, 512) == 0
    assert q8_down_row4_expected_calls(owner, 511, 512) == 0


# --- missing implementation or config is a named failure --------------------


def test_a_missing_leaf_is_named(monkeypatch):
    monkeypatch.setattr(registry, "is_registered", lambda key: False)
    with pytest.raises(ValueError, match="not registered.*row4_gemv"):
        q8_down_row4_expected_calls(expert_down_runner(), 512, 512)


def test_a_missing_ladder_parent_is_named(monkeypatch):
    monkeypatch.setattr(
        registry, "is_registered",
        lambda key: key.variant != "selected_grouped_prefill_pair2_bf16_bf16_out")
    with pytest.raises(ValueError, match="pair2_bf16_bf16_out"):
        q51_fold128_expected_calls(expert_down_runner(), 512, 512)


def test_a_runner_without_a_backend_is_named():
    owner = expert_down_runner()
    del owner.backend
    with pytest.raises(ValueError, match="needs the owning runner's backend"):
        q8_down_row4_expected_calls(owner, 512, 512)


def test_a_config_without_the_expert_width_is_named():
    owner = expert_down_runner()
    del owner.config.expert_feed_forward_length
    with pytest.raises(ValueError, match="expert_feed_forward_length"):
        q51_fold128_expected_calls(owner, 512, 512)


def test_a_model_the_leaf_does_not_serve_is_named():
    # The register leaf is registered for K=640 only.
    with pytest.raises(ValueError, match="serves expert K=640.*K=1024"):
        q51_register_cache_expected_calls(expert_down_runner(ffn=1024), 512, 512)
    # Fold128 covers K<=4096, not above it.
    with pytest.raises(ValueError, match="K<=4096.*K=8192"):
        q51_fold128_expected_calls(expert_down_runner(ffn=8192), 512, 512)


def test_an_artifact_without_the_quant_is_named():
    no_q8 = expert_down_runner(down_quants={n: "gguf_q5_1" for n in range(48)})
    with pytest.raises(ValueError, match="cannot engage.*gguf_q8_0"):
        q8_down_row4_expected_calls(no_q8, 512, 512)


def test_a_chunk_below_the_guard_is_named():
    with pytest.raises(ValueError, match="rows>=512.*chunk is 256"):
        q51_register_cache_expected_calls(expert_down_runner(), 4096, 256)


# --- an enabled arm cannot be verified by a zero expectation ----------------


def test_an_enabled_arm_refuses_a_zero_expectation():
    with pytest.raises(AssertionError, match="cannot be verified by a zero"):
        require_engaged_expectation("q8-down-row4", CASE, 0, 512,
                                    arm_enabled=True)


def test_a_disabled_arm_and_a_positive_expectation_pass():
    require_engaged_expectation("q8-down-row4", CASE, 0, 512,
                                arm_enabled=False)
    require_engaged_expectation("q8-down-row4", CASE, 40, 512,
                                arm_enabled=True)


# --- the gate decides engagement per package, not by the small-row set ------

#: Every package whose arm is verified by one of the counts above.
ARM_PACKAGES = (
    "q51-fold128", "q51-fold-pair", "q51-register-cache", "q51-row-publish",
    "q8-down-row4", "q8-down-bundle", "q8-down-register",
)


def test_every_arm_package_overrides_the_generic_expectation():
    """A canonical small row must still require these arms to engage.

    The gate first derives ``expected`` from a package set plus
    ``prompt_tokens > 2051``, then recomputes it per package. Only the
    recomputation makes a 512-token canonical row require engagement, so it has
    to stay after the generic assignment - a reordering would silently assert
    that a valid small-row arm did not run.
    """

    import ast
    import inspect

    from scripts import qwen4exp_row4_state_gate as gate

    tree = ast.parse(inspect.getsource(gate.main))
    generic = None
    overrides: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if targets == ["expected"] and "2051" in ast.dump(node.value):
                generic = node
        if not isinstance(node, ast.If):
            continue
        assigns_expected = any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "expected"
                    for t in stmt.targets)
            for stmt in node.body)
        if not assigns_expected:
            continue
        literals = {
            c.value for c in ast.walk(node.test)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        for package in literals.intersection(ARM_PACKAGES):
            overrides[package] = node.lineno

    assert generic is not None, "the gate no longer derives a generic expected"
    assert sorted(overrides) == sorted(ARM_PACKAGES), (
        "a package has no per-package expectation override: "
        f"{sorted(set(ARM_PACKAGES) - set(overrides))}")
    early = {p: n for p, n in overrides.items() if n < generic.lineno}
    assert not early, f"overrides precede the generic assignment: {early}"
