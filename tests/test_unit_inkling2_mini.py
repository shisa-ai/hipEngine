from collections import Counter

import pytest

from hipengine.models.inkling2_mini import Inkling2VisitPlan


def _config():
    return {
        "text_config": {
            "num_hidden_layers": 24,
            "local_layer_ids": [i for i in range(24) if i % 6 != 5],
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
        }
    }


def test_original_checkpoint_visit_order_and_distinct_state():
    plan = Inkling2VisitPlan.from_config(_config())
    assert [visit.weight_layer for visit in plan.visits] == [
        *range(6), *range(6, 18), *range(6, 18), *range(6, 18), *range(18, 24),
    ]
    assert {visit.state_slot for visit in plan.visits} == set(range(48))
    assert [visit.state_slot for visit in plan.visits if visit.weight_layer == 6] == [6, 24, 36]
    assert [i for i, visit in enumerate(plan.visits) if visit.restart_from_prelude] == [18, 30]
    assert [visit.message_pass for visit in plan.visits[18:42]] == [0] * 12 + [1] * 12
    assert all(visit.message_pass is None for visit in plan.visits[:18] + plan.visits[42:])
    assert Counter(visit.is_local for visit in plan.visits) == {True: 40, False: 8}


def test_one_pass_needs_no_restarts_or_extra_cache_slots():
    config = _config()
    config["text_config"]["loop_config"]["num_passes"] = 1
    visits = Inkling2VisitPlan.from_config(config).visits
    assert [visit.state_slot for visit in visits] == list(range(24))
    assert all(not visit.restart_from_prelude and visit.message_pass is None for visit in visits)


@pytest.mark.parametrize("override", [{"num_passes": 0}, {"core_start": -1}, {"core_end": 25}, {"core_end": 6}])
def test_invalid_loop_geometry_is_rejected(override):
    config = _config()
    config["text_config"]["loop_config"].update(override)
    with pytest.raises(ValueError, match="Inkling 2 loop geometry"):
        Inkling2VisitPlan.from_config(config)


def test_out_of_range_local_layer_is_rejected():
    config = _config()
    config["text_config"]["local_layer_ids"].append(24)
    with pytest.raises(ValueError, match="local_layer_ids"):
        Inkling2VisitPlan.from_config(config)
