import json
import sys

import pytest

from scripts.qwen4exp_campaign_ab_retention import main


@pytest.mark.parametrize("status,subset", [("running", False), ("completed", True)])
def test_retention_rejects_incomplete_or_subset(tmp_path, monkeypatch, status, subset):
    source = tmp_path / "input.json"
    source.write_text(json.dumps({"status": status, "diagnostic_subset": subset}))
    output = tmp_path / "output.json"
    monkeypatch.setattr(sys, "argv", ["retention", "--input", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="full-suite"):
        main()
    assert not output.exists()
