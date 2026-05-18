"""Stable prompt fixtures for speculative decoding benchmarks."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_STABLE_PROMPT_FIXTURE = Path("fixtures/dflash/stable_prompts.jsonl")
DEFAULT_SYNTHETIC_LENGTHS = (64, 256)
DEFAULT_SYNTHETIC_SEED = 20260518


@dataclass(frozen=True)
class StablePromptSpec:
    category: str
    name: str
    text: str
    benchmark_group: str
    representative: bool = True
    input_style: str = "plain"
    source: str = "amd-gpu-tuning dflash prompt suite"

    @property
    def prompt_id(self) -> str:
        return f"{self.category}:{self.name}"

    @property
    def promotion_gate(self) -> bool:
        return self.benchmark_group == "code_promotion"


STABLE_PROMPT_SPECS: tuple[StablePromptSpec, ...] = (
    StablePromptSpec(
        category="code",
        name="quicksort_prefix",
        benchmark_group="code_promotion",
        text=(
            'def quicksort(values):\n'
            '    """Return a sorted copy of values."""\n'
            '    if len(values) <= 1:\n'
            '        return values\n'
            '    pivot = values[len(values) // 2]\n'
            '    left = [x for x in values if x < pivot]\n'
            '    middle = [x for x in values if x == pivot]\n'
            '    right = [x for x in values if x > pivot]\n'
            '    return '
        ),
    ),
    StablePromptSpec(
        category="code",
        name="function_continuation",
        benchmark_group="code_promotion",
        text=(
            'def normalize_scores(rows):\n'
            '    totals = []\n'
            '    for row in rows:\n'
            '        total = sum(row.values())\n'
            '        if total == 0:\n'
            '            totals.append({key: 0.0 for key in row})\n'
            '        else:\n'
            '            totals.append({key: value / total for key, value in row.items()})\n'
            '    return '
        ),
    ),
    StablePromptSpec(
        category="code",
        name="class_continuation",
        benchmark_group="code_promotion",
        text=(
            'class LRUCache:\n'
            '    def __init__(self, capacity: int):\n'
            '        self.capacity = capacity\n'
            '        self.data = {}\n'
            '        self.order = []\n\n'
            '    def get(self, key):\n'
            '        if key not in self.data:\n'
            '            return None\n'
            '        self.order.remove(key)\n'
            '        self.order.append(key)\n'
            '        return '
        ),
    ),
    StablePromptSpec(
        category="code",
        name="json_yaml_continuation",
        benchmark_group="code_promotion",
        text=(
            'Generate the next configuration entries.\n\n'
            '```json\n'
            '{\n'
            '  "service": "router",\n'
            '  "replicas": 3,\n'
            '  "limits": {\n'
            '    "cpu": "2",\n'
            '```\n\n'
            '```yaml\n'
            'service:\n'
            '  name: router\n'
            '  replicas: 3\n'
            '  limits:\n'
            '    memory: '
        ),
    ),
    StablePromptSpec(
        category="code",
        name="humaneval_add",
        benchmark_group="code_promotion",
        source="amd-gpu-tuning dense27 DFlash R7 HumanEval-class prompt",
        text=(
            'def add(a: int, b: int) -> int:\n'
            '    """Add two numbers x and y\n\n'
            '    >>> add(2, 3)\n'
            '    5\n'
            '    >>> add(5, 7)\n'
            '    12\n'
            '    """\n'
            '    '
        ),
    ),
    StablePromptSpec(
        category="code",
        name="humaneval_sort_third",
        benchmark_group="code_promotion",
        source="amd-gpu-tuning dense27 DFlash R7 HumanEval-class prompt",
        text=(
            'from typing import List\n\n\n'
            'def sort_third(l: list) -> list:\n'
            '    """This function takes a list l and returns a list l\' such that\n'
            "    l' is identical to l in the indicies that are not divisible by three, while its values at the indicies that are divisible by three are equal\n"
            '    to the values of the corresponding indicies of l, but sorted.\n\n'
            '    >>> sort_third([1, 2, 3])\n'
            '    [1, 2, 3]\n'
            '    >>> sort_third([5, 6, 3, 4, 8, 9, 2])\n'
            '    [2, 6, 3, 4, 8, 9, 5]\n'
            '    """\n'
            '    '
        ),
    ),
    StablePromptSpec(
        category="math",
        name="short_gsm8k_style",
        benchmark_group="robustness",
        text=(
            "A bakery made 48 muffins in the morning and sold 5 trays of 6 muffins each before lunch. "
            "It baked 18 more muffins in the afternoon. How many muffins are left? Let's solve step by step:"
        ),
    ),
    StablePromptSpec(
        category="instruct",
        name="simple_qa_no_template",
        benchmark_group="robustness",
        text="Question: Give two practical reasons to batch small GPU inference requests.\nAnswer:",
    ),
    StablePromptSpec(
        category="instruct",
        name="simple_qa_qwen_static_chat",
        benchmark_group="robustness",
        input_style="qwen_static_chat_template",
        text=(
            "<|im_start|>user\n"
            "Give two practical reasons to batch small GPU inference requests."
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
    ),
    StablePromptSpec(
        category="prose",
        name="paragraph_continuation",
        benchmark_group="robustness",
        text=(
            "The old observatory stood on a ridge above the harbor. At sunset, the glass dome caught "
            "the last orange light while gulls circled below. Mara unlocked the iron door, brushed dust "
            "from the ledger, and noticed that the final entry had been written "
        ),
    ),
    StablePromptSpec(
        category="general",
        name="concise_summary",
        benchmark_group="robustness",
        source="amd-gpu-tuning QUALITY_PROMPTS-inspired general output guard",
        text=(
            "Summarize the following engineering note in three concise bullet points. "
            "Focus on the practical tradeoffs, not marketing language.\n\n"
            "Note: A speculative decoder can improve throughput only when the verifier "
            "checks several candidate tokens for less work than separate autoregressive steps.\n\n"
            "Summary:"
        ),
    ),
    StablePromptSpec(
        category="multilingual",
        name="ja_gpu_batching",
        benchmark_group="robustness",
        source="amd-gpu-tuning prompt-suite multilingual robustness guard",
        text=(
            "次の質問に日本語で簡潔に答えてください。\n"
            "小さなGPU推論リクエストをまとめてバッチ処理する利点を二つ挙げてください。\n"
            "回答:"
        ),
    ),
    StablePromptSpec(
        category="multilingual",
        name="zh_speculative_decode",
        benchmark_group="robustness",
        source="amd-gpu-tuning prompt-suite multilingual robustness guard",
        text=(
            "请用中文解释投机解码为什么需要目标模型验证草稿 token。"
            "要求：回答简短，并指出一个正确性风险。\n回答："
        ),
    ),
)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    encoded = ",".join(str(int(token)) for token in token_ids).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(tokenizer_path: str | Path):
    from tokenizers import Tokenizer

    path = Path(tokenizer_path)
    if path.is_dir():
        path = path / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def encode_prompt(tokenizer: Any, text: str) -> list[int]:
    return [int(token) for token in tokenizer.encode(text).ids]


def build_prompt_records(
    tokenizer: Any,
    *,
    tokenizer_path: str | Path | None = None,
    synthetic_lengths: Iterable[int] = DEFAULT_SYNTHETIC_LENGTHS,
    synthetic_seed: int = DEFAULT_SYNTHETIC_SEED,
    include_token_ids: bool = True,
) -> list[dict[str, Any]]:
    tokenizer_path_str = str(tokenizer_path) if tokenizer_path is not None else None
    tokenizer_hash = file_sha256(Path(tokenizer_path) / "tokenizer.json" if tokenizer_path and Path(tokenizer_path).is_dir() else tokenizer_path) if tokenizer_path is not None else None
    records = [
        _record_from_spec(spec, tokenizer, tokenizer_path_str, tokenizer_hash, include_token_ids=include_token_ids)
        for spec in STABLE_PROMPT_SPECS
    ]
    vocab_size = int(tokenizer.get_vocab_size())
    for length in synthetic_lengths:
        records.append(
            _synthetic_record(
                length=int(length),
                vocab_size=vocab_size,
                seed=int(synthetic_seed),
                tokenizer_path=tokenizer_path_str,
                tokenizer_sha256=tokenizer_hash,
                include_token_ids=include_token_ids,
            )
        )
    return records


def load_prompt_records(path: str | Path = DEFAULT_STABLE_PROMPT_FIXTURE) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate_prompt_records(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    errors: list[str] = []
    seen: set[str] = set()
    for idx, record in enumerate(records):
        prompt_id = str(record.get("id", ""))
        if not prompt_id:
            errors.append(f"row {idx}: missing id")
        if prompt_id in seen:
            errors.append(f"row {idx}: duplicate id {prompt_id}")
        seen.add(prompt_id)
        token_ids = record.get("prompt_ids")
        if token_ids is not None:
            ids = [int(token) for token in token_ids]
            if int(record.get("prompt_tokens", -1)) != len(ids):
                errors.append(f"{prompt_id}: prompt_tokens does not match prompt_ids length")
            if record.get("prompt_ids_sha256") != token_ids_sha256(ids):
                errors.append(f"{prompt_id}: prompt_ids_sha256 mismatch")
        text = record.get("prompt_text")
        if text is not None and record.get("prompt_text_sha256") != text_sha256(str(text)):
            errors.append(f"{prompt_id}: prompt_text_sha256 mismatch")
        if record.get("benchmark_group") not in {"code_promotion", "robustness", "synthetic_stress"}:
            errors.append(f"{prompt_id}: unknown benchmark_group {record.get('benchmark_group')!r}")
    if not any(record.get("benchmark_group") == "code_promotion" for record in records):
        errors.append("no code_promotion prompts present")
    if not any(record.get("benchmark_group") == "robustness" for record in records):
        errors.append("no robustness prompts present")
    if not any(record.get("benchmark_group") == "synthetic_stress" for record in records):
        errors.append("no synthetic_stress prompts present")
    return tuple(errors)


def write_prompt_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _record_from_spec(
    spec: StablePromptSpec,
    tokenizer: Any,
    tokenizer_path: str | None,
    tokenizer_sha256: str | None,
    *,
    include_token_ids: bool,
) -> dict[str, Any]:
    token_ids = encode_prompt(tokenizer, spec.text)
    record = {
        "schema": 1,
        "dataset": f"stable/{spec.category}",
        "split": "builtin",
        "id": spec.prompt_id,
        "category": spec.category,
        "name": spec.name,
        "benchmark_group": spec.benchmark_group,
        "promotion_gate": spec.promotion_gate,
        "representative": spec.representative,
        "input_style": spec.input_style,
        "source": spec.source,
        "prompt_text": spec.text,
        "prompt_preview": spec.text.replace("\n", " ")[:160],
        "prompt_text_sha256": text_sha256(spec.text),
        "prompt_tokens": len(token_ids),
        "prompt_ids_sha256": token_ids_sha256(token_ids),
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "synthetic": False,
    }
    if include_token_ids:
        record["prompt_ids"] = token_ids
    return record


def _synthetic_record(
    *,
    length: int,
    vocab_size: int,
    seed: int,
    tokenizer_path: str | None,
    tokenizer_sha256: str | None,
    include_token_ids: bool,
) -> dict[str, Any]:
    if length <= 0:
        raise ValueError(f"synthetic length must be positive, got {length}")
    rng = random.Random(seed + 1009 * length)
    low = 100 if vocab_size > 1000 else 0
    token_ids = [rng.randrange(low, vocab_size) for _ in range(length)]
    text = f"Synthetic deterministic token-id stress prompt length={length}; non-representative."
    record = {
        "schema": 1,
        "dataset": "stable/synthetic",
        "split": "builtin",
        "id": f"synthetic:random_{length}",
        "category": "synthetic",
        "name": f"random_{length}",
        "benchmark_group": "synthetic_stress",
        "promotion_gate": False,
        "representative": False,
        "input_style": "token_ids",
        "source": "hipEngine deterministic synthetic stress fixture",
        "prompt_text": text,
        "prompt_preview": text,
        "prompt_text_sha256": text_sha256(text),
        "prompt_tokens": len(token_ids),
        "prompt_ids_sha256": token_ids_sha256(token_ids),
        "tokenizer_path": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha256,
        "synthetic": True,
        "synthetic_seed": seed,
        "synthetic_vocab_size": vocab_size,
    }
    if include_token_ids:
        record["prompt_ids"] = token_ids
    return record
