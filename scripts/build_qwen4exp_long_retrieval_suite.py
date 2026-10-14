#!/usr/bin/env python3
"""Build the long-context retrieval prompts for the Qwen4Exp long-prompt gate.

Each prompt is a seeded operations log (English or Japanese) of distinct
records, followed by four questions whose answers are single numbers stored in
records at about 10%, 40%, 70% and 95% depth. The answers are written into the
suite, so a candidate's generated text is scored against the source facts,
not against another engine's output. Record fields vary per record, so the
context is not a repetition of one block.

The suite is a committed fixture: rebuild it only to add a version, never to
change an existing one (``--check`` verifies the committed file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_ROOT = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "prompts" / "qwen4exp-long-retrieval-v1.jsonl"
TARGETS = (8192, 16384)
DEPTHS = (0.10, 0.40, 0.70, 0.95)

EN_SITES = ("Harbor", "Ridge", "Cedar", "Lumen", "Quarry", "Delta", "Summit", "Willow",
            "Granite", "Beacon", "Falcon", "Orchard")
EN_PEOPLE = ("Avery", "Blake", "Casey", "Devon", "Emerson", "Finley", "Harper", "Jordan",
             "Kendall", "Logan", "Morgan", "Parker", "Quinn", "Reese", "Rowan", "Sawyer")
EN_ITEMS = ("pallets", "drums", "crates", "coils", "cartons", "valves", "panels", "spools")
EN_NOTES = (
    "the dock door sensor was recalibrated before the shift",
    "a forklift battery was swapped mid-shift",
    "two pallets were re-wrapped after inspection",
    "the carrier arrived forty minutes late",
    "a label printer jam delayed the second wave",
    "the cold room alarm was tested and cleared",
    "a damaged crate was set aside for the claims team",
    "the yard map was updated for the new trailer bays",
    "an audit sample of twelve units matched the manifest",
    "rain slowed the outdoor staging area",
)
JA_SITES = ("港北", "山手", "杉並", "光町", "石切", "河口", "峰岡", "柳川", "御影", "灯台",
            "鷹野", "果樹園")
JA_PEOPLE = ("佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
             "吉田", "山田", "佐々木", "松本", "井上", "木村")
JA_ITEMS = ("パレット", "ドラム缶", "木箱", "コイル", "段ボール", "バルブ", "パネル", "スプール")
JA_NOTES = (
    "シフト前に搬入口のセンサーを再調整した",
    "シフト途中でフォークリフトの電池を交換した",
    "検品後に二つのパレットを巻き直した",
    "運送会社が四十分遅れて到着した",
    "ラベルプリンターの詰まりで第二便が遅れた",
    "冷蔵室の警報を試験し解除した",
    "破損した木箱を保険請求担当に回した",
    "新しいトレーラー区画に合わせて構内図を更新した",
    "抽出監査の十二個が送り状と一致した",
    "雨で屋外の仮置き場の作業が遅れた",
)


def _record(lang: str, number: int, rng: np.random.Generator) -> tuple[str, dict]:
    day = 1 + int(rng.integers(28))
    month = 1 + int(rng.integers(12))
    hour = 6 + int(rng.integers(14))
    minute = int(rng.integers(60))
    count = 100 + int(rng.integers(9900))
    if lang == "en":
        site = EN_SITES[int(rng.integers(len(EN_SITES)))]
        person = EN_PEOPLE[int(rng.integers(len(EN_PEOPLE)))]
        item = EN_ITEMS[int(rng.integers(len(EN_ITEMS)))]
        note = EN_NOTES[int(rng.integers(len(EN_NOTES)))]
        text = (f"Record {number:05d}: on 2026-{month:02d}-{day:02d} at {hour:02d}:{minute:02d}, "
                f"{person} at the {site} warehouse logged {count} {item} shipped; note: {note}.")
    else:
        site = JA_SITES[int(rng.integers(len(JA_SITES)))]
        person = JA_PEOPLE[int(rng.integers(len(JA_PEOPLE)))]
        item = JA_ITEMS[int(rng.integers(len(JA_ITEMS)))]
        note = JA_NOTES[int(rng.integers(len(JA_NOTES)))]
        text = (f"記録{number:05d}：2026年{month}月{day}日{hour}時{minute:02d}分、{site}倉庫の"
                f"{person}が{item}{count}個の出荷を記録。備考：{note}。")
    return text, {"number": number, "count": count}


def _question(lang: str, index: int, number: int) -> str:
    if lang == "en":
        return f"{index}. How many units were shipped in Record {number:05d}?"
    return f"{index}. 記録{number:05d}で出荷を記録した個数はいくつですか？"


def _instructions(lang: str) -> tuple[str, str]:
    if lang == "en":
        head = "Below is a warehouse shipping log. Read it, then answer the questions after it.\n\n"
        tail = ("\n\nAnswer each question with only the number, one per line, in the form "
                "`1: <number>`. Do not explain.")
        return head, tail
    head = "以下は倉庫の出荷記録です。読んだ後、最後の質問に答えてください。\n\n"
    tail = "\n\n各質問に数字だけで、一行に一つずつ `1: <数字>` の形式で答えてください。説明は不要です。"
    return head, tail


def build_prompt(tokenizer, lang: str, target: int, seed: int) -> dict:
    from scripts.gguf_mtp_bench import build_chat_prompt

    rng = np.random.default_rng(seed)
    head, tail = _instructions(lang)
    records: list[tuple[str, dict]] = []
    # Grow the log until the chat-formatted prompt with questions reaches the
    # target; questions are chosen after the length is fixed.
    probe_questions = "\n".join(_question(lang, i + 1, 99999) for i in range(len(DEPTHS)))
    def length() -> int:
        body = "\n".join(t for t, _ in records)
        return len(build_chat_prompt(tokenizer, head + body + "\n\n" + probe_questions + tail))

    while True:
        records.append(_record(lang, len(records) + 1, rng))
        if len(records) % 16 == 0 and length() >= target:
            break
    while length() > target:
        records.pop()
    pick = [records[min(len(records) - 1, int(depth * len(records)))][1] for depth in DEPTHS]
    questions = "\n".join(_question(lang, i + 1, fact["number"]) for i, fact in enumerate(pick))
    body = "\n".join(t for t, _ in records)
    prompt = head + body + "\n\n" + questions + tail
    tokens = build_chat_prompt(tokenizer, prompt)
    return {
        "id": f"long_retrieval_{lang}_p{target}",
        "category": "general_en" if lang == "en" else "general_ja",
        "suite": "qwen4exp-long-retrieval-v1",
        "messages": [{"role": "user", "content": prompt}],
        "answers": [int(fact["count"]) for fact in pick],
        "answer_records": [int(fact["number"]) for fact in pick],
        "records": len(records),
        "seed": seed,
        "chat_tokens": len(tokens),
    }


def build(model_root: Path) -> list[dict]:
    from hipengine.loading.gguf import discover_gguf_files, scan_gguf
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    info = scan_gguf(discover_gguf_files(model_root)[0])
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    rows = []
    for seed, (lang, target) in enumerate(
            ((lang, target) for target in TARGETS for lang in ("en", "ja")), start=1):
        rows.append(build_prompt(tokenizer, lang, target, 20261014 + seed))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="rebuild and compare with the committed suite")
    args = parser.parse_args(argv)
    rows = build(args.model_root)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if args.check:
        same = args.output.read_text(encoding="utf-8") == text
        print("identical" if same else "DIFFERS")
        return 0 if same else 1
    args.output.write_text(text, encoding="utf-8")
    for row in rows:
        print(row["id"], row["chat_tokens"], row["records"], row["answers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
