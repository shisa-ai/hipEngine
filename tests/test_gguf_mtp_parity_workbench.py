from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gguf_mtp_parity_workbench.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gguf_mtp_parity_workbench", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workbench_candidate_aliases_and_envs() -> None:
    mod = _load_module()

    names = mod.parse_csv_set(
        "all",
        valid=set(mod.CANDIDATES),
        aliases={"all": mod.ALL_CANDIDATE_NAMES},
        label="candidate",
    )

    assert names[0] == "default"
    assert "x8-q6" in names
    assert mod.CANDIDATES["x8-q6"].env == {"HIPENGINE_GGUF_SELECTED_X8_REPACK": "q6"}
    assert mod.CANDIDATES["raw-dp4a"].extra_args == ("--no-decode-repack",)


def test_workbench_dry_run_writes_e2e_and_piece_commands(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    raw_root = tmp_path / "raw"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--tag",
            "unit",
            "--raw-root",
            str(raw_root),
            "--output",
            str(output),
            "--stages",
            "e2e,pieces",
            "--candidates",
            "default,x8-q6",
            "--cycles",
            "2",
            "--draft-n-max",
            "3",
            "--piece-iters",
            "1",
            "--piece-warmup",
            "0",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema"] == "hipengine.gguf_mtp_parity_workbench.v1"
    assert data["status"] == "dry_run"
    assert [row["candidate"] for row in data["e2e"]] == ["default", "x8-q6"]
    assert data["e2e"][1]["env"]["HIPENGINE_GGUF_SELECTED_X8_REPACK"] == "q6"
    assert all(row["command"]["status"] == "dry_run" for row in data["e2e"])
    assert [row["piece"] for row in data["pieces"]] == [
        "q4_selected_dual",
        "raw_selected_down_q5_q6",
        "x8_selected_down_q5_q6",
    ]
    piece_commands = "\n".join(row["command"]["command"] for row in data["pieces"])
    assert "gguf_q4_k_selected_dual_dp4a_microbench.py" in piece_commands
    assert "gguf_k_selected_pack8_dp4a_microbench.py" in piece_commands
    assert "gguf_x8_selected_down_dp4a_microbench.py" in piece_commands


def test_workbench_category_dry_run_uses_extra_arg_equals_form(tmp_path: Path) -> None:
    output = tmp_path / "summary-category.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--tag",
            "unit-category",
            "--raw-root",
            str(tmp_path / "raw"),
            "--output",
            str(output),
            "--stages",
            "category",
            "--candidates",
            "raw-dp4a",
            "--cycles",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    command = data["category"][0]["command"]["command"]
    assert "--extra-arg=--prompt-reasoning" in command
    assert "--extra-arg=--no-decode-repack" in command
    assert data["category"][0]["command"]["status"] == "dry_run"
