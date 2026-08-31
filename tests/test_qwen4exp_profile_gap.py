from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_summarize_moe_selection_reports_active_row_distribution() -> None:
    module = _load_script()
    selected = np.asarray([[2, 1], [2, 3], [2, 1], [0, 3]], dtype=np.int64)

    result = module._summarize_moe_selection(
        selected,
        experts=5,
        layer="layers.4.expert_gate",
        quant_triplet=("gguf_q4_k", "gguf_q4_k", "gguf_q8_0"),
    )

    assert result["rows"] == 4
    assert result["top_k"] == 2
    assert result["compact_rows"] == 8
    assert result["active_experts"] == 4
    assert result["max_rows_per_expert"] == 3
    assert result["row_count_histogram"] == {"0": 1, "1": 1, "2": 2, "3": 1}
    assert result["expert_rows"][0] == {"expert": 2, "rows": 3}


def test_moe_telemetry_wraps_copies_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    import hipengine.core.memory as memory

    selected = np.asarray([[1, 2], [2, 3]], dtype=np.int64)
    fake_module = SimpleNamespace()

    def run_moe(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(selected=selected)

    fake_module.run_qwen4_exp_moe = run_moe
    monkeypatch.setattr(memory, "host_array_ptr", lambda value: value)
    monkeypatch.setattr(
        memory,
        "copy_device_to_host",
        lambda destination, source, nbytes, runtime: destination.__setitem__(
            slice(None), source
        ),
    )
    telemetry = module.MoeTelemetry(fake_module)
    telemetry.install()
    weights = {
        name: SimpleNamespace(
            spec=SimpleNamespace(
                slot_path="layers.3.expert_gate",
                quant_key=quant,
            )
        )
        for name, quant in (
            ("expert_gate", "gguf_q4_k"),
            ("expert_up", "gguf_q4_k"),
            ("expert_down", "gguf_q5_1"),
        )
    }
    result = fake_module.run_qwen4_exp_moe(
        0,
        weights,
        rows=2,
        top_k=2,
        experts=4,
        scratch=SimpleNamespace(runtime=object()),
    )
    telemetry.close()

    assert result.selected is selected
    assert fake_module.run_qwen4_exp_moe is run_moe
    assert telemetry.snapshot()["rows"][0]["layer"] == "layers.3.expert_gate"
    assert telemetry.snapshot()["copy_bytes"] == selected.nbytes


def test_parser_collects_moe_telemetry_flag(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--mode",
            "prefill",
            "--prompt-file",
            str(tmp_path / "prompt.txt"),
            "--moe-telemetry",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.moe_telemetry is True


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
