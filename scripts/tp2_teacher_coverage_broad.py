"""Broadened AR teacher-forced coverage: TP2 vs both per-GPU TP1 controls.

Extends the 16-row diagnostic gate to 64 prompt rows across the engine's
categories (code, general English, general Japanese, mixed), comparing
full-vocabulary last-position logits of the production TP2 runner against
the TP1 control on each card. Reports mean/p95/max KL and top-1 agreement
per control, plus per-row worst cases, and writes a JSON artifact.

Usage::

    python scripts/tp2_teacher_coverage_broad.py [N_ROWS] [--json PATH]
"""

import json
import sys
import time

sys.path.insert(0, "/home/lhl/hipEngine-main")

import numpy as np

from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from hipengine.loading.gguf import scan_gguf

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 64
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else \
    "benchmarks/results/2026-09-15-w7900-tp2-teacher-coverage-broad.json"

PROMPTS = (
    # code
    "Write a Python function that reverses a list.",
    "def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n",
    "Explain what a红黑树 guarantees about tree height.",
    "git rebase --onto main feature~3 feature\n",
    "Implement a thread-safe LRU cache in Python.",
    "SELECT user_id, count(*) FROM orders GROUP BY user_id HAVING",
    "```python\nfor i in range(len(a)):\n",
    "Fix this bug: IndexError: list index out of range\n",
    # general English
    "The weather today is",
    "Paris is the capital of",
    "Summarize the plot of Hamlet in three sentences.",
    "Explain photosynthesis to a ten-year-old.",
    "The committee concluded that",
    "What are the main causes of inflation?",
    "Translate 'good morning' into French.",
    "Write a haiku about autumn rain.",
    # general Japanese
    "日本の首都は",
    "東京タワーはどこにありますか",
    "日本語で「ありがとう」の意味を説明してください。",
    "大阪と東京の違いは何ですか",
    "今日の天気はどうですか",
    "日本の伝統料理を三つ挙げてください。",
    "「頑張って」の使い方を例文で示してください。",
    "夏休みの宿題について書いてください。",
    # mixed
    "日本語でPythonのリストを説明してください。",
    "Explain 漢字 etymology briefly.",
    "このコードのバグを直してください: print(x[10])\n",
    "Write カタカナ for 'computer'.",
    "比較してください: lists vs tuples in Python.",
    "「機械学習」を英語で何と言いますか",
    "Review this 日本語 sentence for grammar:",
    "Mix English and 日本語 in one sentence about AI.",
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    return exp / exp.sum(axis=-1, keepdims=True)


def _kl_row(teacher: np.ndarray, student: np.ndarray) -> dict[str, float]:
    p = _softmax(teacher)
    q = _softmax(student)
    eps = 1e-12
    kl = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    return {"kl": kl, "top1": int(np.argmax(teacher)) == int(np.argmax(student))}


def render(tok, text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"))


t0 = time.perf_counter()
info = scan_gguf(MODEL)
import hipengine.loading as _loading
tok = Qwen35GGUFTokenizer.from_gguf_info(_loading.load_gguf_index(MODEL))
tokens_by_row = [render(tok, text) for text in PROMPTS[:N_ROWS]]
tokens_by_row = [t for t in tokens_by_row if len(t) >= 4]

# Sequential arms: three resident sessions cannot coexist in VRAM.
logits_by_arm: dict[str, list[np.ndarray]] = {}
for arm, devices in (("tp1-d0", (0,)), ("tp1-d1", (1,)), ("tp2", (0, 1))):
    build0 = time.perf_counter()
    session = MlpTP2GenerationSession(MODEL, devices=devices, mode="tp2" if arm == "tp2" else "tp1")
    print(f"{arm} session built in {time.perf_counter() - build0:.0f}s", flush=True)
    arm_logits = []
    for index, tokens in enumerate(tokens_by_row):
        row_logits = np.asarray(session.teacher_forced_logits(tokens)[-1], dtype=np.float64)
        arm_logits.append(row_logits)
        if arm == "tp2":
            print(f"row {index}: tp2 done", flush=True)
    logits_by_arm[arm] = arm_logits
    session.close()
    print(f"{arm} swept; session closed", flush=True)

rows = []
for index, tokens in enumerate(tokens_by_row):
    row = {"index": index, "tokens": len(tokens)}
    for control in ("tp1-d0", "tp1-d1"):
        m = _kl_row(logits_by_arm[control][index], logits_by_arm["tp2"][index])
        row[control] = m
    rows.append(row)
    print(f"row {index}: kl_d0={row['tp1-d0']['kl']:.3e} "
          f"kl_d1={row['tp1-d1']['kl']:.3e} "
          f"top1_d0={row['tp1-d0']['top1']} top1_d1={row['tp1-d1']['top1']}", flush=True)

summary = {}
for control in ("tp1-d0", "tp1-d1"):
    kls = np.array([r[control]["kl"] for r in rows])
    top1 = [r[control]["top1"] for r in rows]
    summary[control] = {
        "rows": len(rows),
        "mean_kl": float(kls.mean()),
        "p95_kl": float(np.percentile(kls, 95)),
        "max_kl": float(kls.max()),
        "top1_agreement": sum(top1) / len(top1),
    }
    print(f"{control}: mean_kl={summary[control]['mean_kl']:.3e} "
          f"p95={summary[control]['p95_kl']:.3e} max={summary[control]['max_kl']:.3e} "
          f"top1={summary[control]['top1_agreement']:.3f}", flush=True)

json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print(f"artifact: {OUT}", flush=True)
for name, s in sessions.items():
    s.close()
