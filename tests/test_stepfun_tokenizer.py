from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from hipengine.loading.gguf import scan_gguf_splits
from hipengine.tokenization import StepFunGGUFTokenizer


DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")
DEFAULT_STEPFUN_NVFP4_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--stepfun-ai--Step-3.7-Flash-NVFP4"
    / "snapshots/36afbf6e15100cdc2d7a5b79d7e95d276ed33679"
)


def _stepfun_gguf_paths() -> tuple[Path, ...]:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return paths


def _hf_tokenizer() -> Tokenizer:
    path = (
        Path(os.environ.get("HIPENGINE_STEPFUN_NVFP4_SNAPSHOT", DEFAULT_STEPFUN_NVFP4_SNAPSHOT))
        / "tokenizer.json"
    )
    if not path.is_file():
        pytest.skip(
            "StepFun HF tokenizer.json not found; set HIPENGINE_STEPFUN_NVFP4_SNAPSHOT "
            "to the cached HF snapshot"
        )
    return Tokenizer.from_file(str(path))


def _stepfun_tokenizer() -> StepFunGGUFTokenizer:
    return StepFunGGUFTokenizer.from_gguf_info(scan_gguf_splits(_stepfun_gguf_paths()))


def test_stepfun_gguf_tokenizer_matches_hf_tokenizer_json() -> None:
    tokenizer = _stepfun_tokenizer()
    hf = _hf_tokenizer()

    plain_texts = ("hello world", "你好，世界", "Reasoning: low\nHello!")
    special_texts = ("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",)
    for text in (*plain_texts, *special_texts):
        assert tokenizer.encode(text) == hf.encode(text).ids
    for text in plain_texts:
        assert tokenizer.decode(tokenizer.encode(text), skip_special=True) == hf.decode(
            hf.encode(text).ids
        )


def test_stepfun_chat_template_and_chat_encoding() -> None:
    tokenizer = _stepfun_tokenizer()
    hf = _hf_tokenizer()
    messages = [{"role": "user", "content": "hello"}]

    rendered = tokenizer.render_chat(messages, reasoning_effort="low")

    assert rendered.startswith("<｜begin▁of▁sentence｜><|im_start|>system\nReasoning: low")
    assert "<|im_start|>user\nhello<|im_end|>\n" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")
    assert tokenizer.encode_chat(messages, reasoning_effort="low") == hf.encode(
        rendered, add_special_tokens=False
    ).ids


def test_stepfun_tokenizer_tracks_multi_eos_and_special_ids() -> None:
    tokenizer = _stepfun_tokenizer()

    assert tokenizer.bos_token_id == 0
    assert tokenizer.padding_token_id == 1
    assert tokenizer.eos_token_ids == (1, 2, 128007)
    assert tokenizer.is_eos(1)
    assert tokenizer.is_eos(2)
    assert tokenizer.is_eos(128007)
    assert not tokenizer.is_eos(128006)
    assert tokenizer.encode("hello", add_bos=False)[0] != 0
    assert tokenizer.encode("hello", add_bos=True)[0] == 0


def test_stepfun_tokenizer_does_not_import_torch() -> None:
    had_torch = "torch" in sys.modules

    tokenizer = _stepfun_tokenizer()
    tokenizer.encode_chat([{"role": "user", "content": "hello"}], reasoning_effort="medium")

    if not had_torch:
        assert "torch" not in sys.modules
