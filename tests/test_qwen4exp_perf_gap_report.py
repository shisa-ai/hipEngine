from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_perf_gap_report.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_perf_gap_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_current_context_profile_carries_all_p0_census_fields() -> None:
    module = _load_script()
    artifact = {
        "date": "2026-08-31",
        "source": {"head": "abc"},
        "profile": {"manifest_sha256": "prod", "strict_manifest_sha256": "strict"},
        "contexts": {
            "2051": {
                "qsa_path": "dense_equivalent",
                "clean": {"wall_median_ms": 66.6},
                "profiled": {
                    "kernel_ms_per_step": 58.5,
                    "kernel_rows_per_step": 1764,
                    "families": {"qsa_attention": {"ms_per_step": 8.4, "rows_per_step": 12}},
                    "api": {
                        "direct_kernel_launch": {"calls_per_step": 1100, "ms_per_step": 18.2},
                        "graph_launch": {"calls_per_step": 48, "ms_per_step": 3.2},
                        "memcpy_api": {"calls_per_step": 38, "ms_per_step": 20.7},
                    },
                },
            }
        },
        "decision": {"next_owner": "H256 QSA", "secondary": "MoE", "do_not_chase": "top-k"},
    }

    report = module.render_context_report(artifact)

    assert "live 2,051" in report
    assert "1,100" in report
    assert "48" in report
    assert "38" in report
    assert "18.20" in report
    assert "3.20" in report
    assert "20.70" in report
    assert "58.50" in report
    assert "1,764" in report
    assert "wall minus kernel" in report
    assert "H256 QSA" in report


def test_parser_accepts_context_artifact_kind(tmp_path: Path) -> None:
    module = _load_script()
    args = module.build_parser().parse_args([str(tmp_path / "artifact.json")])
    assert args.artifact.name == "artifact.json"
