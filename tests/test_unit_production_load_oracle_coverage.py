"""Isolated load-oracle workers must return every requested trajectory in full."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.gguf_production_load_gate import WorkloadRequest, _run_isolated_reference_worker


@pytest.mark.parametrize("fault", ["missing_reference", "extra_reference", "short_reference", "long_reference", "extra_prompt"])
def test_isolated_oracle_rejects_incomplete_or_extra_coverage(monkeypatch, fault):
    key = "token=7:prompt=2"

    def fake_run(command, **kwargs):
        payload = {
            "prompt_rows": {key: {"token_ids": [7, 7], "token_ids_sha256": "abc"}},
            "reference_tokens": {key: [8, 9]},
        }
        if fault == "missing_reference":
            payload["reference_tokens"].clear()
        elif fault == "extra_reference":
            payload["reference_tokens"]["unexpected"] = [8, 9]
        elif fault == "short_reference":
            payload["reference_tokens"][key] = [8]
        elif fault == "long_reference":
            payload["reference_tokens"][key] = [8, 9, 10]
        else:
            payload["prompt_rows"]["unexpected"] = payload["prompt_rows"][key]
        Path(command[command.index("--output-json") + 1]).write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.gguf_production_load_gate.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="isolated reference worker.*(coverage|token)"):
        _run_isolated_reference_worker(
            model=Path("unused.gguf"), backend="hip_gfx1151",
            max_active_requests=1, max_sequence_length=16, max_output_tokens=2,
            specs=(WorkloadRequest("row", 7, 2, 2),),
            compiler_version_file=None, require_cached_build=False,
        )
