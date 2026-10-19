"""CPU contracts for Qwen4Exp request capacity with the shared decode batch.

Plain AR requests share one decode batch and one paged K/V block pool, so the
active-request count is no longer the number of full-context c1 runners. These
tests pin the default, the kernel row bound, the separate staging-runner count,
the rollback lever, and the absence of a process-global class mutation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_ar_batch import (
    AR_BATCH_ENV,
    QWEN4_EXP_BATCH_SLOT_POSITIONS,
    QWEN4_EXP_DEFAULT_REQUEST_CAPACITY,
    QWEN4_EXP_MAX_BATCH_ROWS,
    QWEN4_EXP_STAGING_RUNNERS,
    ar_batch_enabled,
    batch_slot_positions,
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


def test_batch_is_on_by_default_and_the_env_is_a_rollback(monkeypatch) -> None:
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    assert ar_batch_enabled()
    for value in ("0", "false", "off"):
        monkeypatch.setenv(AR_BATCH_ENV, value)
        assert not ar_batch_enabled()
    monkeypatch.setenv(AR_BATCH_ENV, "1")
    assert ar_batch_enabled()


def test_default_capacity_is_concurrent_with_two_staging_runners(monkeypatch) -> None:
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    generator = _generator(max_sequence_length=262144)
    try:
        assert generator.server_plain_ar_max_active_requests == QWEN4_EXP_DEFAULT_REQUEST_CAPACITY
        assert generator._staging_capacity == QWEN4_EXP_STAGING_RUNNERS
    finally:
        generator.close()
    # Context length no longer bounds the request count: the pool does.
    for context in (2048, 32768, 262144, None):
        wide = _generator(max_sequence_length=context, resident_capacity=16)
        try:
            assert wide.server_plain_ar_max_active_requests == 16
            assert wide._staging_capacity == QWEN4_EXP_STAGING_RUNNERS
        finally:
            wide.close()
    solo = _generator(resident_capacity=1)
    try:
        assert solo._staging_capacity == 1
    finally:
        solo.close()


def test_capacity_is_bounded_by_the_indexed_decode_kernels(monkeypatch) -> None:
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    top = _generator(resident_capacity=QWEN4_EXP_MAX_BATCH_ROWS)
    top.close()
    with pytest.raises(ValueError, match="indexed batch decode kernels"):
        _generator(resident_capacity=QWEN4_EXP_MAX_BATCH_ROWS + 1)
    with pytest.raises(ValueError, match="active request count"):
        _generator(resident_capacity=0)


def test_rollback_gives_every_request_its_own_runner(monkeypatch) -> None:
    monkeypatch.setenv(AR_BATCH_ENV, "0")
    generator = _generator(max_sequence_length=2048)
    try:
        assert generator.server_plain_ar_max_active_requests == QWEN4_EXP_STAGING_RUNNERS
        assert generator._staging_capacity == QWEN4_EXP_STAGING_RUNNERS
    finally:
        generator.close()
    explicit = _generator(max_sequence_length=2048, resident_capacity=3)
    try:
        assert explicit._staging_capacity == 3
    finally:
        explicit.close()


def test_batch_slot_extent_is_capped(monkeypatch) -> None:
    assert batch_slot_positions(2048) == 2048
    assert batch_slot_positions(262144) == QWEN4_EXP_BATCH_SLOT_POSITIONS
    assert batch_slot_positions(QWEN4_EXP_BATCH_SLOT_POSITIONS) == QWEN4_EXP_BATCH_SLOT_POSITIONS


def test_capacity_policy_is_per_instance_and_leaves_the_class_bound(monkeypatch) -> None:
    class_bound = Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests
    assert class_bound == QWEN4_EXP_DEFAULT_REQUEST_CAPACITY
    monkeypatch.delenv(AR_BATCH_ENV, raising=False)
    wide = _generator(resident_capacity=12)
    try:
        narrow = _generator(resident_capacity=3)
        try:
            assert narrow.server_plain_ar_max_active_requests == 3
            assert wide.server_plain_ar_max_active_requests == 12
        finally:
            narrow.close()
    finally:
        wide.close()
    assert Qwen4ExpGGUFTextGenerator.server_plain_ar_max_active_requests == class_bound


def test_batch_capacity_does_not_raise_the_mixed_mtp_capability(monkeypatch) -> None:
    config = ServerConfig(
        model="/models/gguf/x.gguf",
        served_model_name="x",
        eager_load=False,
        max_active_requests=4,
        max_context_tokens=2048,
    )
    monkeypatch.setenv(AR_BATCH_ENV, "0")
    mtp_bound_off = _gguf_mtp_batch_route_max_active_requests(config)
    monkeypatch.delenv(AR_BATCH_ENV)
    generator = _generator(max_sequence_length=2048, resident_capacity=4)
    try:
        # MTP rows never join the AR batch and the plugin publishes no MTP
        # width, so the batch capacity cannot widen it.
        assert getattr(generator, "server_mtp_batch_max_active_requests", None) is None
        assert _gguf_mtp_batch_route_max_active_requests(config) == mtp_bound_off
    finally:
        generator.close()
