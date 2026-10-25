"""Deterministic coverage for the retired selector's script-local replacement."""

import pytest

from scripts.qwen4exp_allocation_census import simulated_runner
from scripts.qwen4exp_chunk_workspace import PREFILL_FIELDS, capture_workspace
from scripts.qwen4exp_rt1_same_owner_selection import (
    SELECTION,
    configure_selection,
    selected_workspace,
)
from tests.test_unit_qwen4exp_scratch_plan import config


@pytest.fixture(autouse=True)
def clear_selection():
    configure_selection(None)
    yield
    configure_selection(None)


@pytest.fixture
def owner_pair():
    with simulated_runner(config(), context=256, chunk=32) as (runner, runtime):
        yield runner, runtime


def test_selected_workspace_uses_smallest_fitting_owner_then_largest_on_overflow(
        owner_pair):
    runner, _ = owner_pair
    base = runner._prefill_workspaces[0]
    small = runner._allocate_extra_prefill_workspace(16)
    small_view = capture_workspace(small)

    configure_selection([base, small])
    assert [item.prefill_chunk_size for item in SELECTION] == [16, 32]

    observed = []
    for tokens in (1, 16, 17, 32, 33):
        with selected_workspace(runner, tokens):
            observed.append((
                tokens,
                runner.prefill_chunk_size,
                runner.prefill_chunk_size == small_view.values["prefill_chunk_size"]
                and all(getattr(runner, name) is small_view.values[name]
                        for name in PREFILL_FIELDS if name != "prefill_chunk_size"),
            ))
        assert runner.prefill_chunk_size == 32
        assert runner.gdn_prefill_scratch is base.gdn_prefill_scratch
        assert runner._prefill_buffers is base._prefill_buffers

    assert observed == [
        (1, 16, True),
        (16, 16, True),
        (17, 32, False),
        (32, 32, False),
        (33, 32, False),
    ]


def test_selected_workspace_restores_every_prefill_field_after_failure(owner_pair):
    runner, _ = owner_pair
    base = runner._prefill_workspaces[0]
    small = runner._allocate_extra_prefill_workspace(16)
    configure_selection([small, base])
    before = {name: getattr(runner, name) for name in PREFILL_FIELDS}

    with pytest.raises(RuntimeError, match="prefill failed"):
        with selected_workspace(runner, 1):
            assert runner.prefill_chunk_size == 16
            assert runner.gdn_prefill_scratch is small.gdn_prefill_scratch
            raise RuntimeError("prefill failed")

    assert runner.prefill_chunk_size == before["prefill_chunk_size"] == 32
    for name in PREFILL_FIELDS:
        if name != "prefill_chunk_size":
            assert getattr(runner, name) is before[name]
    assert runner.gdn_prefill_scratch is base.gdn_prefill_scratch


def test_selected_workspace_disabled_keeps_the_base_owner(owner_pair):
    runner, _ = owner_pair
    base = runner._prefill_workspaces[0]
    small = runner._allocate_extra_prefill_workspace(16)
    configure_selection([small, base])
    configure_selection(None)
    assert not SELECTION

    with selected_workspace(runner, 1):
        assert runner.prefill_chunk_size == 32
        assert runner.gdn_prefill_scratch is base.gdn_prefill_scratch
        assert runner._prefill_buffers is base._prefill_buffers
