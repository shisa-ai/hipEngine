"""Profile binding must not change the next test's execution environment."""
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("baseline", [None, "0", "1"])
def test_profile_environment_restores_collection_baseline(monkeypatch, baseline):
    spec = spec_from_file_location("isolated_conftest", Path(__file__).with_name("conftest.py"))
    hooks = module_from_spec(spec)
    spec.loader.exec_module(hooks)
    from hipengine.generation import qwen36_gguf_gfx1100_profiles as profiles

    names = [getattr(profiles, name) for name in (
        "FP16_RECURRENT_STATE_ENV", "Q4_FUSED_R28_ENV", "Q6_DP4A_GROUPED_ENV",
        "VERIFY_CAPTURE_PREFILL_GDN_ENV", "VERIFY_F32_POST_NORM_ENV",
        "VERIFY_F32_RESIDUAL_ENV",
    )]
    for name in names:
        if baseline is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, baseline)
    hooks.pytest_collection_finish(None)
    for name in names:
        monkeypatch.setenv(name, "changed-by-profile")
    monkeypatch.setenv("PROFILE_ISOLATION_TEST_UNRELATED", "untouched")
    hooks.pytest_runtest_teardown(None, None)
    assert {name: os.environ.get(name) for name in names} == dict.fromkeys(names, baseline)
    assert os.environ["PROFILE_ISOLATION_TEST_UNRELATED"] == "untouched"
