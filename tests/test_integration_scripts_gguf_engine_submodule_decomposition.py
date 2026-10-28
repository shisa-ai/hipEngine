"""The decomposition packet must describe itself.

This script produced the C1-C8 admission/decode decomposition whose numbers kept being quoted as
current after grouped prefill shipped, because nothing in the packet said where or from what code it
was measured. These tests pin the stamp, in both the normal and the GPU-free fallback path.
"""

from __future__ import annotations

import importlib.util
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_engine_submodule_decomposition.py"


def _load():
    spec = importlib.util.spec_from_file_location("gesd_mod", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provenance_is_derived_from_the_machine_not_written_by_hand() -> None:
    module = _load()
    prov = module._provenance(["--arm", "ar", "--output", "x.json"])
    assert prov["host_name"] == socket.gethostname()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert prov["hipengine_commit"] == head
    assert prov["command"] == ["--arm", "ar", "--output", "x.json"]


def test_provenance_reports_a_real_dirty_axis() -> None:
    """`dirty` must be a boolean, not an absent field, so a packet can never read as clean by default."""
    prov = _load()._provenance(["--arm", "ar"])
    assert isinstance(prov["dirty"], bool)


def test_provenance_falls_back_without_the_runtime(tmp_path, monkeypatch) -> None:
    """A no-HIP runner must still get a stamped packet, and must be able to tell it is a fallback."""
    import hipengine.benchmark.provenance as provenance

    def boom(**kwargs):
        raise RuntimeError("no device here")

    monkeypatch.setattr(provenance, "collect_artifact_provenance", boom)
    prov = _load()._provenance(["--arm", "mtp"])
    assert "provenance_error" in prov
    assert "no device here" in prov["provenance_error"]
    assert prov["host_name"] == socket.gethostname()
    assert re.fullmatch(r"[0-9a-f]{40}", str(prov["hipengine_commit"])), prov["hipengine_commit"]
    assert prov["command"] == ["--arm", "mtp"]


def _packet(kind, arm, *, tokens, widths, admission_per_lane, step_per_lane):
    cells = []
    for width in widths:
        for index, factor in enumerate((0.8, 1.2)):
            wall = width * (admission_per_lane + (tokens - 1) * step_per_lane) * factor
            if kind == "hipengine":
                rows = [{"usage": {"completion_tokens": tokens,
                                   "prompt_tokens": 40 + 20 * index}} for _ in range(width)]
                cells.append({"prompt_id": f"p{index}", arm: {
                    "arm": arm, "width": width, "wall_seconds": wall, "rows": rows,
                }})
            else:
                rows = [{"tokens_predicted": tokens, "tokens_evaluated": 40 + 20 * index}
                        for _ in range(width)]
                cells.append({"prompt_id": f"p{index}", "arm": arm, "width": width,
                              "wall": wall, "rows": rows})
    return {"cells": cells}


@pytest.mark.parametrize("short_tokens", [1, 3])
def test_published_cli_decomposes_multi_prompt_waves(tmp_path, short_tokens):
    widths = (1, 3, 8)
    long_tokens = 7
    specs = []
    for label, kind, admission, step in (
        ("hipengine", "hipengine", 0.30, 0.02),
        ("current", "llamacpp", 0.12, 0.04),
    ):
        paths = []
        for name, arm, tokens in (("short", "ar" if kind == "hipengine" else "prefill", short_tokens),
                                  ("long", "ar", long_tokens)):
            path = tmp_path / f"{label}-{name}.json"
            path.write_text(json.dumps(_packet(
                kind, arm, tokens=tokens, widths=widths,
                admission_per_lane=admission, step_per_lane=step,
            )))
            paths.append(path)
        specs += ["--engine", f"{label}={kind}:{paths[0]}:{paths[1]}"]
    output = tmp_path / "nested" / "decomposition.json"
    command = [sys.executable, str(SCRIPT), *specs, "--arm", "ar", "--arm-d1", "prefill,ar",
               "--output", str(output), "--note", "synthetic CLI contract"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == _load().SCHEMA
    assert payload["note"] == "synthetic CLI contract"
    assert "long_tok_s_if_admission_best" in result.stdout
    for block, admission, step in zip(payload["engines"], (0.30, 0.12), (0.02, 0.04)):
        assert set(block["widths"]) == {str(width) for width in widths}
        for width in widths:
            row = block["widths"][str(width)]
            assert row["prompts_per_arm"] == 2
            assert row["prompt_tokens_per_lane"] == 50
            assert row["short_tokens_per_lane"] == short_tokens
            assert row["long_tokens_per_lane"] == long_tokens
            assert row["decode_step_ms"] == pytest.approx(1000 * width * step)
            assert row["admission_ms"] == pytest.approx(1000 * width * admission)
            assert row["published_long_tok_s"] == pytest.approx(
                long_tokens / (admission + (long_tokens - 1) * step))
            assert row["decode_tok_s_aggregate"] == pytest.approx(1 / step)
            swapped_rate = long_tokens / (0.12 + (long_tokens - 1) * step)
            assert payload["admission_what_if"][str(width)]["long_tok_s_if_admission_best"][
                block["label"]] == pytest.approx(swapped_rate)
    assert payload["provenance"]["command"] == command


def test_cli_refuses_nonoverlapping_packets_without_writing_output(tmp_path):
    paths = []
    for name, width, tokens in (("short", 1, 1), ("long", 3, 7)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_packet(
            "hipengine", "ar", tokens=tokens, widths=(width,),
            admission_per_lane=0.3, step_per_lane=0.02,
        )))
        paths.append(path)
    output = tmp_path / "result.json"
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--engine", f"test=hipengine:{paths[0]}:{paths[1]}",
        "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "no overlapping widths" in result.stderr
    assert not output.exists()
