from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "openai_chat_capture.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("openai_chat_capture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_prompts(path: Path) -> None:
    rows = [
        {
            "id": "code-one",
            "category": "code",
            "messages": [{"role": "user", "content": "write code"}],
        },
        {
            "id": "ja-one",
            "category": "general_ja",
            "messages": [{"role": "user", "content": "説明して"}],
        },
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


class _Response:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def test_capture_preserves_reasoning_and_reports_repeatability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "capture.json"
    _write_prompts(prompts)
    calls = []

    def urlopen(request, timeout):
        calls.append((json.loads(request.data), timeout))
        prompt = calls[-1][0]["messages"][0]["content"]
        return _Response(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning": f"reason:{prompt}",
                            "content": f"answer:{prompt}",
                        }
                    }
                ],
                "usage": {"completion_tokens": 2},
            }
        )

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    assert (
        module.main(
            [
                "capture",
                "--url",
                "http://127.0.0.1:8080",
                "--prompts",
                str(prompts),
                "--output",
                str(output),
                "--repetitions",
                "2",
            ]
        )
        == 0
    )

    artifact = json.loads(output.read_text())
    assert len(calls) == 4
    assert calls[0][0]["top_k"] == 1
    assert artifact["summary"]["all_repeat_exact"] is True
    assert artifact["summary"]["repeat_exact_prompts"] == 2
    assert artifact["results"][0]["message"] == {
        "reasoning_content": "reason:write code",
        "content": "answer:write code",
    }


def test_compare_captures_separates_self_and_cross_mode_exactness(tmp_path: Path) -> None:
    module = _load_script()
    paths = []
    for name, values in (
        ("ar", {"one": ["a", "a"], "two": ["b", "b"]}),
        ("mtp", {"one": ["a", "a"], "two": ["b", "c"]}),
    ):
        path = tmp_path / f"{name}.json"
        results = []
        for prompt_id, hashes in values.items():
            for repetition, digest in enumerate(hashes):
                results.append(
                    {
                        "id": prompt_id,
                        "category": "code",
                        "repetition": repetition,
                        "message_sha256": digest,
                    }
                )
        path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "kind": "openai_chat_message_capture",
                    "prompt_sha256": "f" * 64,
                    "results": results,
                }
            )
        )
        paths.append((name, path))

    comparison = module.compare_captures(paths)

    assert comparison["cross_capture_exact_rows"] == 1
    assert comparison["all_cross_capture_exact"] is False
    assert comparison["self_exact_rows"] == {"ar": 2, "mtp": 1}
    assert comparison["rows"][1]["mode_self_exact"] == {"ar": True, "mtp": False}


def test_compare_rejects_different_prompt_suites(tmp_path: Path) -> None:
    module = _load_script()
    paths = []
    for index in range(2):
        path = tmp_path / f"capture-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "kind": "openai_chat_message_capture",
                    "prompt_sha256": str(index) * 64,
                    "results": [{"id": "one", "repetition": 0, "message_sha256": "a"}],
                }
            )
        )
        paths.append((str(index), path))

    with pytest.raises(module.CaptureError, match="different prompt files"):
        module.compare_captures(paths)


def test_load_prompts_rejects_duplicate_ids(tmp_path: Path) -> None:
    module = _load_script()
    prompts = tmp_path / "prompts.jsonl"
    row = {"id": "same", "messages": [{"role": "user", "content": "x"}]}
    prompts.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(module.CaptureError, match="duplicate prompt id"):
        module.load_prompts(prompts)
