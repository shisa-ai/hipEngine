"""Retired AR rollback must not change production request/staging capacity."""
from tests.test_unit_qwen4exp_ar_batch_capacity import _generator
from hipengine.generation.qwen4_exp_ar_batch import (
    QWEN4_EXP_DEFAULT_REQUEST_CAPACITY, QWEN4_EXP_STAGING_RUNNERS,
)


def test_retired_ar_environment_cannot_reduce_capacity(monkeypatch):
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_AR_BATCH', '0')
    generator = _generator()
    try:
        assert generator.server_plain_ar_max_active_requests == QWEN4_EXP_DEFAULT_REQUEST_CAPACITY
        assert generator._staging_capacity == QWEN4_EXP_STAGING_RUNNERS
    finally:
        generator.close()
