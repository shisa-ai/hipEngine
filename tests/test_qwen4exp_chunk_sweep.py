from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen4exp_chunk_sweep.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen4exp_chunk_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_chunks_are_bounded_by_prompt_and_include_partial_tail() -> None:
    module = _load_script()
    assert module._valid_chunks(512, [256, 512, 1024, 2048]) == [256, 512]
    assert module._valid_chunks(1024, [256, 512, 1024, 2048]) == [256, 512, 1024]
    assert module._valid_chunks(4096, [256, 512, 1024, 2048]) == [256, 512, 1024, 2048]


def test_parser_defaults_to_required_chunk_ladder(tmp_path: Path) -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        ["--model-root", str(tmp_path / "model"), "--output", str(tmp_path / "o.json")]
    )
    assert args.chunk_size == [256, 512, 1024, 2048]
    assert args.warmups == 1
    assert args.repetitions == 3
