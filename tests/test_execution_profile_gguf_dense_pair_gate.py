from __future__ import annotations

import pytest

from scripts.execution_profile_gguf_dense_pair_gate import (
    dense_pair_policy_override,
    validate_route_variants,
)


def test_dense_pair_policy_override_is_shape_scoped_and_non_mutating() -> None:
    identity = ("dense-h5120", "MOSTLY_Q4_K_M")
    other = ("dense-h1024", "MOSTLY_Q4_K_M")
    shape = (1, 5_120, 17_408)
    original = {
        identity: {shape: "package-current", (2, 5_120, 17_408): "rows2"},
        other: {(1, 1_024, 3_584): "small"},
    }

    overridden = dense_pair_policy_override(
        original,
        identity=identity,
        shape=shape,
        variant="strict-local32",
    )

    assert overridden[identity][shape] == "strict-local32"
    assert overridden[identity][(2, 5_120, 17_408)] == "rows2"
    assert overridden[other] == original[other]
    assert original[identity][shape] == "package-current"


def test_dense_pair_gate_rejects_ambiguous_variants() -> None:
    validate_route_variants("strict-local32", "candidate-dp4a")
    with pytest.raises(ValueError, match="must differ"):
        validate_route_variants("same", "same")
    with pytest.raises(ValueError, match="non-empty"):
        validate_route_variants("", "candidate")
