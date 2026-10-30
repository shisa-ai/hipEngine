from __future__ import annotations

import sys

import pytest

from hipengine.loading import qwen35_gguf_admission as admission
from scripts import qwen36_dense_gguf_suite as suite
from scripts import ud_mtp_certification as certification


@pytest.mark.parametrize("fail", [False, True])
def test_certification_preserves_admission_resolver(tmp_path, monkeypatch, fail):
    resolver = admission.resolve_qwen35_gguf_artifact_preset
    argv = sys.argv
    output = tmp_path / "result.json"
    monkeypatch.setattr(certification, "_preflight", lambda *args: {"preset_key": "synthetic"})

    def run_suite():
        assert admission.resolve_qwen35_gguf_artifact_preset is resolver
        if fail:
            raise RuntimeError("suite failed")
        output.write_text('{}')
        return 1

    monkeypatch.setattr(suite, "main", run_suite)
    kwargs = dict(
        artifact="synthetic", path=tmp_path / "model.gguf", quant="synthetic",
        output=output, runs=1, candidate_budgets=(3,), max_new_tokens=1, limit=1,
    )
    if fail:
        with pytest.raises(RuntimeError, match="suite failed"):
            certification._run_one(**kwargs)
    else:
        result = certification._run_one(**kwargs)
        assert result["suite_exit_code"] == 1
        assert not result["evidence"]["gates_passed"]
    assert admission.resolve_qwen35_gguf_artifact_preset is resolver
    assert sys.argv is argv
