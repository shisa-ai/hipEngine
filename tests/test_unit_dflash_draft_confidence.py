from __future__ import annotations

import pytest

from scripts.dflash_chain_e2e_bench import (
    _canonical_profile_route,
    _confidence_limited_active_count,
    _load_profile_route_manifest,
    _profile_route_for_prompt,
    _terminal_ar_tokens_for_prompt,
    _top1_probabilities_from_topk,
    _top1_probability_from_topk,
)


def test_top1_probability_from_topk_is_stable_softmax() -> None:
    probability = _top1_probability_from_topk((1000.0, 999.0, 998.0))

    assert probability == pytest.approx(0.66524096)


def test_top1_probability_from_single_logit_defaults_to_one() -> None:
    assert _top1_probability_from_topk((42.0,)) == 1.0


def test_confidence_limited_active_count_stops_at_first_low_confidence() -> None:
    probabilities = _top1_probabilities_from_topk(
        (
            (4.0, 0.0),
            (3.0, 0.0),
            (0.1, 0.0),
            (5.0, 0.0),
        )
    )

    assert _confidence_limited_active_count(probabilities, max_active=4, p_min=0.70) == 2


def test_confidence_limited_active_count_disabled_keeps_budget() -> None:
    probabilities = (0.01, 0.02)

    assert _confidence_limited_active_count(probabilities, max_active=4, p_min=0.0) == 4


def test_profile_route_manifest_accepts_aliases_and_prompt_ids(tmp_path) -> None:
    manifest = tmp_path / "routes.json"
    manifest.write_text(
        '{"default": "ar", "routes": {"code:class_continuation": "dflash", "math": "branching_topk"}}',
        encoding="utf-8",
    )

    default, routes, raw = _load_profile_route_manifest(manifest)

    assert default == "ar"
    assert raw is not None
    assert routes == {"code:class_continuation": "chain", "math": "tree"}
    assert _profile_route_for_prompt({"id": "code:class_continuation"}, default=default, routes=routes) == "chain"
    assert _profile_route_for_prompt({"benchmark_group": "math"}, default=default, routes=routes) == "tree"
    assert _profile_route_for_prompt({"id": "unknown"}, default=default, routes=routes) == "ar"


def test_profile_route_manifest_can_override_terminal_ar_tokens(tmp_path) -> None:
    manifest = tmp_path / "routes.json"
    manifest.write_text(
        '{"default": "ar", "terminal_ar_tokens": {"default": 5, "code:json_yaml_continuation": 20, "robustness": 7}}',
        encoding="utf-8",
    )

    _default, _routes, raw = _load_profile_route_manifest(manifest)

    assert raw is not None
    assert _terminal_ar_tokens_for_prompt({"id": "code:json_yaml_continuation"}, default=0, manifest=raw) == 20
    assert _terminal_ar_tokens_for_prompt({"benchmark_group": "robustness"}, default=0, manifest=raw) == 7
    assert _terminal_ar_tokens_for_prompt({"id": "unknown"}, default=0, manifest=raw) == 5
    assert _terminal_ar_tokens_for_prompt({"id": "unknown"}, default=3, manifest=None) == 3


def test_profile_route_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="profile route"):
        _canonical_profile_route("maybe")
