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


def test_select_fixture_case_returns_exact_token_ids() -> None:
    module = _load_script()
    fixture = {
        "cases": [
            {
                "id": "code-p512",
                "category": "code",
                "prompt_tokens": 3,
                "prompt_token_ids": [10, 11, 12],
                "prompt_token_ids_sha256": "digest",
            }
        ]
    }

    case = module._select_fixture_case(fixture, "code-p512")

    assert case["prompt_token_ids"] == [10, 11, 12]
    with pytest.raises(ValueError, match="exactly one"):
        module._select_fixture_case(fixture, "general_en-p512")


def test_parser_collects_canonical_fixture_case(tmp_path: Path) -> None:
    module = _load_script()

    args = module.build_parser().parse_args(
        [
            "--model-root",
            str(tmp_path / "model"),
            "--mode",
            "prefill",
            "--case-id",
            "code-p512",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.case_id == "code-p512"
    assert args.fixture.name == "qwen4exp_canonical_ar_p512_p1024_p4096.json"


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


def test_parser_selects_explicit_decode_output_boundary(tmp_path: Path) -> None:
    module = _load_script()

    default_args = module.build_parser().parse_args(
        [
            "--model-root", str(tmp_path / "model"),
            "--mode", "decode",
            "--output", str(tmp_path / "compact.json"),
        ]
    )
    full_args = module.build_parser().parse_args(
        [
            "--model-root", str(tmp_path / "model"),
            "--mode", "decode",
            "--decode-output", "full_logits",
            "--output", str(tmp_path / "full.json"),
        ]
    )

    assert default_args.decode_output == "compact"
    assert full_args.decode_output == "full_logits"


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


def test_requested_profile_names_a_profile_without_consulting_the_default() -> None:
    module = _load_script()

    assert module._requested_profile("production").value == "production"
    assert module._requested_profile("strict").value == "strict"


def test_requested_profile_default_mirrors_the_shipped_llm_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``default`` is the absence of a caller request, not a fourth profile.

    The census has to be able to run the path a caller that names no profile
    gets, otherwise "the wide route is the shipped default" stays a claim about
    a path no harness exercises. Where the lane has a certified plan that is
    production; where it does not, this must fail loudly rather than quietly
    running the migration path under a name that claims the shipped default.
    """

    module = _load_script()
    from hipengine.execution_profiles import ExecutionProfile
    from hipengine.generation.qwen4_exp_profiles import (
        register_qwen4_exp_gfx1151_profiles,
    )

    register_qwen4_exp_gfx1151_profiles()
    assert module._requested_profile("default") is ExecutionProfile.PRODUCTION

    import hipengine.execution_profiles as profiles

    monkeypatch.setattr(
        profiles, "resolve_default_execution_profile", lambda **kwargs: None
    )
    with pytest.raises(SystemExit, match="no certified default execution profile"):
        module._requested_profile("default")


def test_parser_takes_the_shipped_default_profile_choice(tmp_path: Path) -> None:
    module = _load_script()
    common = [
        "--model-root",
        str(tmp_path / "model"),
        "--mode",
        "prefill",
        "--output",
        str(tmp_path / "result.json"),
    ]

    # Unchanged default: existing invocations keep resolving production directly.
    assert module.build_parser().parse_args(common).execution_profile == "production"
    assert (
        module.build_parser()
        .parse_args([*common, "--execution-profile", "default"])
        .execution_profile
        == "default"
    )
