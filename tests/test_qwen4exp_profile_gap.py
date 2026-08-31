from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_profile_gap.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_profile_gap", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_overrides_accepts_hipengine_keys_and_equals_in_value() -> None:
    module = _load_script()

    assert module._parse_overrides(
        [
            "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL=1",
            "HIPENGINE_EXAMPLE=a=b",
        ]
    ) == {
        "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL": "1",
        "HIPENGINE_EXAMPLE": "a=b",
    }


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (["MISSING_SEPARATOR"], "KEY=VALUE"),
        (["=1"], "non-empty key"),
        (["PATH=/tmp"], "HIPENGINE_"),
        (["HIPENGINE_DUP=1", "HIPENGINE_DUP=0"], "duplicate override"),
    ],
)
def test_parse_overrides_rejects_ambiguous_or_unscoped_values(
    raw: list[str], message: str
) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match=message):
        module._parse_overrides(raw)


def test_apply_post_binder_overrides_records_bound_and_effective_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    key = "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL"
    monkeypatch.delenv(key, raising=False)

    bound, effective = module._apply_post_binder_overrides({key: "1"})

    assert bound[key] is None
    assert effective[key] == "1"


def test_parser_collects_repeated_overrides(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--mode",
            "decode",
            "--output",
            str(tmp_path / "result.json"),
            "--override",
            "HIPENGINE_ONE=1",
            "--override",
            "HIPENGINE_TWO=2",
        ]
    )

    assert args.override == ["HIPENGINE_ONE=1", "HIPENGINE_TWO=2"]
