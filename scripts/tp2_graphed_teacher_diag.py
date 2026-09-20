"""Diagnostic: graphed TP2 (capture bound variants) vs eager TP2 teacher logits."""
import pathlib

import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/home/lhl/hipEngine-main/scripts")

from hipengine.distributed.tp2_generate import MlpTP2GenerationSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
from hipengine.loading.gguf import scan_gguf

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

info = scan_gguf(MODEL)
tok = Qwen35GGUFTokenizer.from_gguf_info(info)
tokens = tok.encode("The capital of France is")
print("tokens:", list(tokens), flush=True)


def kl_row(a: np.ndarray, b: np.ndarray) -> list[float]:
    def norm(x):
        p = np.exp(x - x.max(axis=-1, keepdims=True))
        return p / p.sum(axis=-1, keepdims=True)

    pa, pb = norm(a), norm(b)
    kl = (pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30))).sum(axis=-1)
    return np.round(kl, 4).tolist()


logits: dict[str, np.ndarray] = {}
# (label, schedule, capture_position or None)
arms = (
    ("eager", "eager", None),
    ("graphed-2047", "graphed", None),
    ("graphed-8", "graphed", 8),
)
for label, schedule, capture_position in arms:
    s = MlpTP2GenerationSession(
        MODEL,
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=2048,
        driver="compiled",
        schedule=schedule,
    )
    if capture_position is not None:
        s._capture_position = capture_position
    logits[label] = s.teacher_forced_logits(tokens)
    s.close()
    print(f"{label} done", flush=True)

e = logits["eager"]
for label in ("graphed-2047", "graphed-8"):
    g = logits[label]
    kl = kl_row(e, g)
    agree = float((e.argmax(axis=-1) == g.argmax(axis=-1)).mean())
    print(f"{label}: per-position KL {kl}, top1 agree {agree}")
print("eager top1:", e.argmax(axis=-1).tolist())
for label in ("graphed-2047", "graphed-8"):
    print(f"{label} top1:", logits[label].argmax(axis=-1).tolist())
