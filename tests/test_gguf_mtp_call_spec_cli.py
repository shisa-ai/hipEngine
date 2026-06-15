from __future__ import annotations

import json
from pathlib import Path

import scripts.gguf_mtp_call_spec as call_spec_cli


def test_gguf_mtp_call_spec_cli_emits_json_and_propagates_non_strict(
    monkeypatch,
    capsys,
) -> None:
    discovered: list[str] = []
    summarized: list[tuple[str, bool, tuple[int, ...] | None]] = []

    def fake_discover(path: str) -> list[Path]:
        discovered.append(path)
        return [Path(path) / "model-a.gguf", Path(path) / "model-b.gguf"]

    def fake_summarize(
        path: Path,
        *,
        strict: bool,
        layers: list[int] | None,
    ) -> dict[str, object]:
        summarized.append((path.name, strict, None if layers is None else tuple(layers)))
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

    status = call_spec_cli.main(
        ["/models/gguf", "--non-strict", "--layer", "40", "--indent", "0"]
    )

    assert status == 0
    assert discovered == ["/models/gguf"]
    assert summarized == [("model-a.gguf", False, (40,)), ("model-b.gguf", False, (40,))]
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

    def fake_summarize(
        path: Path,
        *,
        strict: bool,
        layers: list[int] | None,
    ) -> dict[str, object]:
        assert layers is None
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


def test_gguf_mtp_call_spec_summarize_filters_layers(monkeypatch) -> None:
    class _FakeInfo:
        path = Path("model.gguf")
        architecture = "qwen35moe"
        tensor_count = 1

    class _FakeReader:
        def __init__(self, path: Path) -> None:
            assert path == Path("model.gguf")
            self.info = _FakeInfo()

    class _FakeCallSpec:
        def __init__(self, layer: int) -> None:
            self.layer = layer

        def as_dict(self) -> dict[str, int]:
            return {"layer_id": self.layer}

    class _FakePlan:
        def __init__(self, layer: int) -> None:
            self.layer_id = layer
            self.cpu_reference_call_spec = _FakeCallSpec(layer)

    monkeypatch.setattr(call_spec_cli, "GGUFReader", _FakeReader)
    monkeypatch.setattr(
        call_spec_cli,
        "build_qwen35_gguf_mtp_draft_tensor_plans",
        lambda info, *, strict: (_FakePlan(40), _FakePlan(41)),
    )

    summary = call_spec_cli.summarize(Path("model.gguf"), layers={41})

    assert summary["mtp_draft_call_specs"] == [{"layer_id": 41}]
