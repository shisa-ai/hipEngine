#!/usr/bin/env python3
"""Record hipEngine GGUF prompt token IDs for parity checks.

This is an MTP-GGUF oracle scaffold: it intentionally only records prompt
rendering/tokenization metadata.  Cross-engine accepted/output comparisons must
first prove that hipEngine and llama.cpp saw the same prompt token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.loading.gguf import GGUFReader
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "fixtures" / "llamacpp_mtp_bench_prompts.json"


class TokenizerLike(Protocol):
    eos_token_id: int | None
    padding_token_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = False) -> str: ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_prompt_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    prompts = suite.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} does not contain a non-empty 'prompts' list")
    seen: set[str] = set()
    for item in prompts:
        if not isinstance(item, dict):
            raise ValueError(f"invalid prompt entry in {path}: {item!r}")
        name = str(item.get("name") or "")
        prompt = str(item.get("prompt") or "")
        if not name or not prompt:
            raise ValueError(f"prompt entries require non-empty name and prompt: {item!r}")
        if name in seen:
            raise ValueError(f"duplicate prompt name in {path}: {name}")
        seen.add(name)
    return suite


def select_prompts(
    suite: dict[str, Any],
    *,
    names_csv: str | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    prompts = [{"name": str(p["name"]), "prompt": str(p["prompt"])} for p in suite["prompts"]]
    if names_csv:
        names = [part.strip() for part in names_csv.split(",") if part.strip()]
        by_name = {prompt["name"]: prompt for prompt in prompts}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"unknown prompt name(s): {', '.join(missing)}")
        prompts = [by_name[name] for name in names]
    if limit is not None:
        prompts = prompts[: max(0, int(limit))]
    if not prompts:
        raise ValueError("prompt selection is empty")
    return prompts


def build_prompt_token_inventory(
    *,
    tokenizer: TokenizerLike,
    prompts: Sequence[dict[str, str]],
    model: str | Path,
    prompts_file: str | Path,
    tokenizer_model: str,
    tokenizer_pre: str,
    prompt_render: str = "raw",
) -> dict[str, Any]:
    if prompt_render != "raw":
        raise ValueError("only raw prompt rendering is currently supported")

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        source_text = str(prompt["prompt"])
        rendered_text = source_text
        token_ids = [int(token_id) for token_id in tokenizer.encode(rendered_text)]
        decoded_text = tokenizer.decode(token_ids)
        rows.append(
            {
                "name": str(prompt["name"]),
                "source_chars": len(source_text),
                "source_sha256": sha256_text(source_text),
                "rendered_chars": len(rendered_text),
                "rendered_sha256": sha256_text(rendered_text),
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "token_ids_sha256": sha256_token_ids(token_ids),
                "roundtrip_ok": decoded_text == rendered_text,
                "decoded_sha256": sha256_text(decoded_text),
            }
        )

    return {
        "schema": 1,
        "kind": "hipengine_gguf_prompt_token_inventory",
        "model": str(model),
        "prompts_file": str(prompts_file),
        "prompt_render": prompt_render,
        "tokenization": "hipengine.gguf.qwen35.byte_bpe_approx",
        "tokenizer_model": tokenizer_model,
        "tokenizer_pre": tokenizer_pre,
        "eos_token_id": tokenizer.eos_token_id,
        "padding_token_id": tokenizer.padding_token_id,
        "warning": (
            "This records hipEngine GGUF tokenizer output only. Cross-engine MTP "
            "parity still requires exact comparison against llama.cpp prompt token IDs."
        ),
        "prompts": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reader = GGUFReader(args.model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    suite = load_prompt_suite(args.prompts_file)
    prompts = select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
    metadata = reader.info.metadata
    return build_prompt_token_inventory(
        tokenizer=tokenizer,
        prompts=prompts,
        model=args.model,
        prompts_file=args.prompts_file,
        tokenizer_model=str(metadata.get("tokenizer.ggml.model")),
        tokenizer_pre=str(metadata.get("tokenizer.ggml.pre")),
        prompt_render=args.prompt_render,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-names", help="comma-separated prompt names to include")
    parser.add_argument("--limit", type=int, help="include only the first N selected prompts")
    parser.add_argument("--prompt-render", choices=("raw",), default="raw")
    parser.add_argument("--out", type=Path, help="write JSON to this path instead of stdout")
    args = parser.parse_args()

    payload = json.dumps(run(args), indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
