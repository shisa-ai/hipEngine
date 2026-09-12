from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _normalize_prefill_queue_drain,
)


@pytest.mark.parametrize("mode", ("none", "chunk", "layer"))
def test_prefill_queue_drain_mode_normalization(mode: str) -> None:
    assert _normalize_prefill_queue_drain(mode.upper()) == mode


def test_prefill_queue_drain_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="prefill_queue_drain"):
        _normalize_prefill_queue_drain("kernel")


@pytest.mark.parametrize(
    ("mode", "boundaries", "expected"),
    (
        ("none", ("layer", "chunk"), ()),
        ("chunk", ("layer", "chunk"), (7,)),
        ("layer", ("layer", "layer", "chunk"), (7, 7)),
    ),
)
def test_prefill_queue_drain_synchronizes_only_selected_boundaries(
    mode: str,
    boundaries: tuple[str, ...],
    expected: tuple[int, ...],
) -> None:
    synchronized: list[int] = []

    class Runtime:
        def stream_synchronize(self, stream: int) -> None:
            synchronized.append(int(stream))

    session = object.__new__(Qwen35GGUFResidentSession)
    session.prefill_queue_drain = mode
    runtime = Runtime()
    for boundary in boundaries:
        session._drain_prefill_queue(boundary, runtime=runtime, stream=7)

    assert tuple(synchronized) == expected


def test_readme_sweep_forwards_prefill_queue_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.qwen35_readme_sweep as sweep

    captured: dict[str, object] = {}

    def fake_run(
        args,
        model,
        workloads,
        warmup_decode_tokens,
        max_sequence_length,
        compiler_version,
        prefill_config,
    ):
        del model, workloads, warmup_decode_tokens, max_sequence_length, compiler_version, prefill_config
        captured["prefill_queue_drain"] = args.prefill_queue_drain
        return {"ok": True}

    monkeypatch.setattr(sweep, "_run_gguf_sweep", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_readme_sweep.py",
            "--engine",
            "gguf",
            "--model",
            str(tmp_path / "model.gguf"),
            "--workloads",
            "512/0",
            "--prefill-queue-drain",
            "layer",
        ],
    )

    assert sweep.main() == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert captured == {"prefill_queue_drain": "layer"}
