"""CPU coverage for the packet comparison CLI on supported Python versions."""
import json


def test_packet_diff_reports_basename_and_paired_cycles(tmp_path, capsys):
    from scripts.gguf_server_packet_diff import main

    def packet(wall):
        return {"cells": [{"width": 8, "prompt_id": "example",
                          "ar": {"wall_seconds": wall, "generated_tokens": 16,
                                 "rows": [{}]},
                          "mtp": {"wall_seconds": wall, "generated_tokens": 16,
                                  "rows": [{"mtp": {"used": True, "draft_cycles": 4}}]}}]}

    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    control.write_text(json.dumps(packet(2.0)))
    candidate.write_text(json.dumps(packet(1.0)))
    assert main(["packet-diff", str(control), str(candidate)]) == 0
    output = capsys.readouterr().out
    assert "width C8  control=control.json  candidate=candidate.json" in output
    assert "AR: wall 2.0000 -> 1.0000 (+50.0%)" in output
    assert "MTP: wall 2.0000 -> 1.0000 (+50.0%)" in output
    assert "same-cycle paired n=1" in output
