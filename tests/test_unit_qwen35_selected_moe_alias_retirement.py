from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_paro_runner as runner
from scripts import qwen35_batch_retained_bench as bench


@pytest.mark.parametrize("legacy", ["0", "1"])
def test_selected_moe_runtime_has_no_legacy_environment_reader(monkeypatch, legacy):
    monkeypatch.setenv("HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE", legacy)
    source = inspect.getsource(runner.Qwen35ParoResidentSession._run_layers_batch_decode)
    assert "_SELECTED_BATCH_MOE_LEGACY_ENV" not in source
    assert not hasattr(runner, "_SELECTED_BATCH_MOE_LEGACY_ENV")


@pytest.mark.parametrize("spelling", ["selected_batch", "selected_c1"])
def test_selected_moe_benchmark_replay_alias_writes_only_canonical_environment(monkeypatch, spelling):
    legacy = "HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE"
    monkeypatch.delenv(legacy, raising=False)
    bench._apply_runtime_env_args(SimpleNamespace(batch_decode_moe_path=spelling))
    assert bench._resolved_batch_decode_moe_path(SimpleNamespace(batch_decode_moe_path=spelling)) == "selected_batch"
    assert os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_BATCH_MOE"] == "1"
    assert legacy not in os.environ
