from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qwen4exp_mtp_head_profile.py"


def _load():
    spec = importlib.util.spec_from_file_location("qwen4exp_mtp_head_profile", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_and_distribution(tmp_path: Path) -> None:
    module = _load()
    args = module.build_parser().parse_args(["--output", str(tmp_path / "result.json")])
    assert args.warmups == 3
    assert args.iterations == 20
    assert args.fullsuite_artifact.name.endswith("mtp-fullsuite-short.json")
    assert module._distribution([3.0, 1.0, 2.0])["median_ms"] == 2.0
