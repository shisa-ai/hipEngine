"""CPU contracts for the Qwen4Exp AR batch qualification configuration.

The AR batch switch is allowed to raise one generator instance's plain-AR
capacity only inside the live-qualified envelope (4 slots at <= 2048
positions). These tests pin the boundary on both sides, the width bound, the
unchanged MTP capability, and the absence of a process-global class mutation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_ar_batch import (
    AR_BATCH_ENV,
    AR_BATCH_QUALIFIED_MAX_POSITIONS,
    AR_BATCH_QUALIFIED_MAX_SLOTS,
    AR_BATCH_SLOTS_ENV,
    ar_batch_capacity_ceiling,
    ar_batch_state_slots,
)
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from hipengine.server.api import ServerConfig, _gguf_mtp_batch_route_max_active_requests


class _Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [1, 2] if text == "hello" else [3]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(value) for value in ids)


class _Runner:
    max_sequence_length = 8

    def close(self):
        pass


def _generator(**kwargs) -> Qwen4ExpGGUFTextGenerator:
    return Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=_Tokenizer(),
        runner=_Runner(),
        **kwargs,
    )


def test_capacity_ceiling_is_scoped_to_the_qualified_envelope(monkeypatch) -> None:
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    assert ar_batch_capacity_ceiling(max_positions=2048, default=2) == 2

    monkeypatch.setenv(AR_BATCH_ENV, "1")
    assert ar_batch_capacity_ceiling(max_positions=2048, default=2) == 4
    assert ar_batch_capacity_ceiling(max_positions=1024, default=2) == 4
    assert ar_batch_capacity_ceiling(
        max_positions=AR_BATCH_QUALIFIED_MAX_POSITIONS + 1, default=2
    ) == 2
    assert ar_batch_capacity_ceiling(max_positions=None, default=2) == 2
    # A plugin that already publishes a wider bound is not narrowed here.
    assert ar_batch_capacity_ceiling(max_positions=2048, default=4) == 4


def test_switch_off_keeps_the_production_capacity_bound(monkeypatch) -> None:
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    generator = _generator(max_sequence_length=2048, resident_capacity=2)
    try:
        assert generator.server_plain_ar_max_active_requests == 2
    finally:
        generator.close()
    with pytest.raises(ValueError, match="production bound is 2"):
        _generator(max_sequence_length=2048, resident_capacity=4)


def test_switch_on_accepts_four_only_at_the_short_context(monkeypatch) -> None:
    monkeypatch.setenv(AR_BATCH_ENV, "1")
    generator = _generator(max_sequence_length=2048, resident_capacity=4)
    try:
        assert generator.server_plain_ar_max_active_requests == 4
    finally:
        generator.close()

    with pytest.raises(ValueError, match="max_sequence_length=4096"):
        _generator(max_sequence_length=4096, resident_capacity=4)
    long_context = _generator(max_sequence_length=4096, resident_capacity=2)
    try:
        assert long_context.server_plain_ar_max_active_requests == 2
    finally:
        long_context.close()

    with pytest.raises(ValueError, match="max_sequence_length=None"):
        _generator(max_sequence_length=None, resident_capacity=4)


def test_explicit_slot_bound_rejects_widths_above_the_qualification(
    monkeypatch,
) -> None:
    monkeypatch.setenv(AR_BATCH_ENV, "1")
    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, "5")
    with pytest.raises(ValueError, match="qualified physical width 4"):
        ar_batch_state_slots(4)

    monkeypatch.setenv(AR_BATCH_SLOTS_ENV, "4")
    assert ar_batch_state_slots(4) == 4

    # The width bound does not relax the serving-capacity bound.
    with pytest.raises(ValueError, match="outside the serving capacity"):
        ar_batch_state_slots(2)


def test_capacity_policy_is_per_instance_and_leaves_the_class_bound(
    monkeypatch,
) -> None:
    class_bound = Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests
    assert class_bound == 2

    monkeypatch.setenv(AR_BATCH_ENV, "1")
    qualified = _generator(max_sequence_length=2048, resident_capacity=4)
    try:
        assert qualified.server_plain_ar_max_active_requests == 4

        monkeypatch.delenv(AR_BATCH_ENV)
        default = _generator(max_sequence_length=2048, resident_capacity=2)
        try:
            assert default.server_plain_ar_max_active_requests == 2
            # The earlier instance keeps its own resolved capability.
            assert qualified.server_plain_ar_max_active_requests == 4
        finally:
            default.close()
    finally:
        qualified.close()

    assert (
        Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests
        == class_bound
    )
    assert (
        Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests_by_max_sequence_length
        == {1024: 2}
    )


def test_switch_does_not_raise_the_mixed_mtp_capability(monkeypatch) -> None:
    config = ServerConfig(
        model="/models/gguf/x.gguf",
        served_model_name="x",
        eager_load=False,
        max_active_requests=4,
        max_context_tokens=2048,
    )
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    mtp_bound_off = _gguf_mtp_batch_route_max_active_requests(config)

    monkeypatch.setenv(AR_BATCH_ENV, "1")
    generator = _generator(max_sequence_length=2048, resident_capacity=4)
    try:
        assert generator.server_plain_ar_max_active_requests == 4
        # MTP rows never join the AR batch (owner exclusion) and the plugin
        # publishes no MTP width, so this switch cannot widen it.
        assert getattr(generator, "server_mtp_batch_max_active_requests", None) is None
        assert _gguf_mtp_batch_route_max_active_requests(config) == mtp_bound_off
    finally:
        generator.close()
