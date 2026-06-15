from __future__ import annotations

import json
from pathlib import Path

import scripts.gguf_mtp_call_spec as call_spec_cli


def test_gguf_mtp_call_spec_cli_emits_json_and_propagates_non_strict(
    monkeypatch,
    capsys,
) -> None:
    discovered: list[str] = []
    summarized: list[tuple[str, bool]] = []

    def fake_discover(path: str) -> list[Path]:
        discovered.append(path)
        return [Path(path) / "model-a.gguf", Path(path) / "model-b.gguf"]

    def fake_summarize(path: Path, *, strict: bool) -> dict[str, object]:
        summarized.append((path.name, strict))
        return {
            "path": str(path),
            "architecture": "qwen35moe",
            "tensor_count": 123,
            "mtp_draft_call_specs": [
                {
                    "cpu_reference_kernel": [
                        "cpu_reference",
                        "mtp_nextn_layer",
                        "gguf_moe",
                        "qwen35_dense_logits",
                    ],
                    "tensor_arguments": {"wq_weight": f"{path.stem}.attn_q.weight"},
                    "qtype_arguments": {"gate_qtype": "Q4_K"},
                    "keyword_arguments": {"num_heads": 16},
                    "dynamic_inputs": [
                        {
                            "argument": "hidden_seed",
                            "required": True,
                            "shape": ["tokens", 2048],
                            "description": "synthetic",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(call_spec_cli, "discover_gguf_files", fake_discover)
    monkeypatch.setattr(call_spec_cli, "summarize", fake_summarize)

    status = call_spec_cli.main(["/models/gguf", "--non-strict", "--indent", "0"])

    assert status == 0
    assert discovered == ["/models/gguf"]
    assert summarized == [("model-a.gguf", False), ("model-b.gguf", False)]
    payload = json.loads(capsys.readouterr().out)
    assert [item["path"] for item in payload] == [
        "/models/gguf/model-a.gguf",
        "/models/gguf/model-b.gguf",
    ]
    assert payload[0]["mtp_draft_call_specs"][0]["cpu_reference_kernel"] == [
        "cpu_reference",
        "mtp_nextn_layer",
        "gguf_moe",
        "qwen35_dense_logits",
    ]
    assert payload[1]["mtp_draft_call_specs"][0]["tensor_arguments"] == {
        "wq_weight": "model-b.attn_q.weight"
    }


def test_gguf_mtp_call_spec_cli_defaults_to_strict(monkeypatch, capsys) -> None:
    summarized: list[bool] = []

    monkeypatch.setattr(call_spec_cli, "discover_gguf_files", lambda path: [Path(path)])

    def fake_summarize(path: Path, *, strict: bool) -> dict[str, object]:
        summarized.append(strict)
        return {
            "path": str(path),
            "architecture": "qwen35moe",
            "tensor_count": 1,
            "mtp_draft_call_specs": [],
        }

    monkeypatch.setattr(call_spec_cli, "summarize", fake_summarize)

    assert call_spec_cli.main(["model.gguf", "--indent", "0"]) == 0

    assert summarized == [True]
    assert json.loads(capsys.readouterr().out) == [
        {
            "path": "model.gguf",
            "architecture": "qwen35moe",
            "tensor_count": 1,
            "mtp_draft_call_specs": [],
        }
    ]
