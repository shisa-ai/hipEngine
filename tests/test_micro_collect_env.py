from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_collect_env_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "micro" / "collect_env.py"
    spec = importlib.util.spec_from_file_location("micro_collect_env", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_truncate_marks_long_output() -> None:
    module = _load_collect_env_module()

    text, truncated = module._truncate("abcdef", 3)

    assert text == "abc"
    assert truncated is True


def test_collect_environment_without_device_probes() -> None:
    module = _load_collect_env_module()
    repo_root = Path(__file__).resolve().parents[1]

    data = module.collect_environment(
        repo_root=repo_root,
        include_device_probes=False,
        include_privileged=False,
        timeout_s=2.0,
        max_output_chars=2000,
    )

    assert data["schema_version"] == 1
    assert data["kind"] == "hipengine_micro_environment"
    assert data["repo"]["commit"]
    assert "hipcc_version" in data["commands"]
    assert "rocminfo" not in data["commands"]
