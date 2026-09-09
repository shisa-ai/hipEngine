"""EVIE-4.5B hipEngine GPU benchmark: torch-free runtime wall-clock.

Mirrors the workload of ``scripts/evie_gpu_bench.py`` (deterministic
synthetic pages, seed 20260909, same page size and batch shapes) so the
torch baselines and the hipEngine path are directly comparable on the same
host.

Usage:
    python3 scripts/evie_hip_bench.py [--pages 8] [--queries 8] [--repeats 3]
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np

from hipengine.loading.hf_cache import resolve_model_path


def _make_page(rng: np.random.Generator, height: int, width: int) -> "np.ndarray":
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    y = 24
    while y < height - 24:
        w = int(rng.integers(80, max(81, width - 80)))
        x0 = int(rng.integers(16, 40))
        page[y : y + 8, x0 : x0 + w] = 0
        y += 16
    return page


def _preprocess_page(
    image: np.ndarray,
    processor_info: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Torch-free Qwen3.5-VL preprocessing for one page.

    Resizes to the nearest multiple-of-32 (>= 56), rescales to [0, 1],
    normalizes (mean/std), and reshapes into temporal patches of
    (t=2, p=16) with channel-major layout per patch row, block-major.
    """

    height, width = image.shape[:2]
    # resize to multiple of 32 (Qwen VL preprocessor with size "longest")
    def to_multiple(v: int) -> int:
        m = max(56, (v // 32) * 32)
        return m

    new_h, new_w = to_multiple(height), to_multiple(width)
    if (new_h, new_w) != (height, width):
        # area-average downsample / nearest upsample per channel
        img = image.astype(np.float32)
        ys = (np.arange(new_h) * (height / new_h)).astype(int).clip(max=height - 1)
        xs = (np.arange(new_w) * (width / new_w)).astype(int).clip(max=width - 1)
        img = img[ys][:, xs]
    else:
        img = image.astype(np.float32)
    img = img / 255.0
    mean = np.array(processor_info["image_mean"], dtype=np.float32)
    std = np.array(processor_info["image_std"], dtype=np.float32)
    img = (img - mean) / std
    # (H, W, 3) -> (3, H, W)
    img = img.transpose(2, 0, 1)
    # pad to multiples of patch 16 temporally: t dimension 2 -> repeat frame
    # Qwen3.5-VL: temporal patch size 2 — duplicate the single frame.
    tpatches = np.concatenate([img, img], axis=0)  # (6, H, W) = (3*2, H, W)
    # fold into (t2, H/16, W/16, 3, 16, 16) block-major patch rows
    c, H, W = tpatches.shape
    ph, pw = H // 16, W // 16
    patches = (
        tpatches.reshape(3, 2, ph, 16, pw, 16)
        .transpose(2, 4, 0, 1, 3, 5)  # (ph, pw, 3, 2, 16, 16)
        .reshape(-1, 3 * 2 * 16 * 16)
    )
    return np.ascontiguousarray(patches), np.array([[1, H // 16, W // 16]], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=8, help="batch of doc pages")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--page-size", default="448x336")
    args = parser.parse_args()

    snapshot = resolve_model_path("tencent/EVIE-4.5B")
    with open(Path(snapshot) / "processor_config.json") as f:
        import json

        pre = json.load(f)["image_processor"]
    processor_info = {
        "image_mean": pre["image_mean"],
        "image_std": pre["image_std"],
    }

    from hipengine.loading.evie import load_evie_model
    from hipengine.runtime.evie import EvieRunner, maxsim

    rng = np.random.default_rng(20260909)
    h, w = (int(x) for x in args.page_size.split("x"))
    images = [_make_page(rng, h, w) for _ in range(args.pages)]
    queries = [f"What is the value shown in figure {i}?" for i in range(args.queries)]

    print(f"loading {snapshot} (hipEngine fp32)...")
    loaded = load_evie_model(snapshot, runtime=None)
    runner = EvieRunner(loaded)
    from hipengine.core.hip import get_hip_runtime

    get_hip_runtime().device_synchronize()

    # preprocess + encode docs (sequentially — v1 single-page runtime)
    def run_once() -> tuple[float, float, float]:
        get_hip_runtime().device_synchronize()
        t0 = time.perf_counter()
        docs = []
        for img in images:
            patches, grid = _preprocess_page(img, processor_info)
            n_img_tokens = int(patches.shape[0] // 4)
            # build input ids: minimal doc template
            ids, mask = _doc_template(n_img_tokens, loaded.spec.image_token_id)
            docs.append(runner.encode_document(ids, mask, patches, grid))
        get_hip_runtime().device_synchronize()
        t1 = time.perf_counter()
        qs = []
        for q in queries:
            ids, mask = _query_template(q, loaded.spec.image_token_id)
            qs.append(runner.encode_query(ids, mask))
        get_hip_runtime().device_synchronize()
        t2 = time.perf_counter()
        scores = [maxsim(q, d) for q, d in zip(qs, docs)]
        get_hip_runtime().device_synchronize()
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2, scores

    run_once()
    best = None
    for _ in range(args.repeats):
        doc_s, q_s, score_s, scores = run_once()
        total = doc_s + q_s + score_s
        print(
            f"doc {doc_s*1e3:8.1f} ms | query {q_s*1e3:7.1f} ms | "
            f"maxsim {score_s*1e3:6.1f} ms | total {total*1e3:8.1f} ms"
        )
        if best is None or total < best[3]:
            best = (doc_s, q_s, score_s, total)
    doc_s, q_s, score_s, total = best
    print(
        f"BEST hipengine-fp32: doc {doc_s*1e3:.1f} ms ({args.pages} pages, {h}x{w}) | "
        f"query {q_s*1e3:.1f} ms ({args.queries} queries) | maxsim {score_s*1e3:.1f} ms | "
        f"total {total*1e3:.1f} ms"
    )
    runner.close()


_TOKENS = {
    "im_start": 151644,
    "im_end": 151645,
    "vision_start": 151652,
    "vision_end": 151653,
    "user": 872,
    "assistant": 77091,
    "nl": 198,
}


def _doc_template(n_img_tokens: int, image_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Rough Qwen3.5 doc template: <|im_start|>user\\n<vision>...<|vision_end|>Describe the image.<|im_end|>"""

    T = _TOKENS
    pre = [T["im_start"], T["user"], T["nl"], T["vision_start"]]
    post = [
        T["vision_end"],
        39238,  # "Describe"
        682,    # " the"
        3391,   # " image"
        13,     # "."
        T["im_end"],
    ]
    ids = np.array(pre + [image_token_id] * n_img_tokens + post, dtype=np.int64)
    return ids, np.ones(len(ids), dtype=np.int64)


def _query_template(text: str, image_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize the query with a tiny whitespace fallback (letters only).

    This benchmark uses synthetic ASCII queries; a byte-level fallback is
    sufficient for wall-clock purposes. Real deployments use the Qwen
    tokenizer.
    """

    T = _TOKENS
    toks: list[int] = []
    for word in text.split():
        v = 0
        for ch in word:
            v = (v * 131 + ord(ch)) % 150000 + 1000
        toks.append(v)
    ids = np.array(
        [T["im_start"], T["user"], T["nl"]] + toks + [T["im_end"]], dtype=np.int64
    )
    return ids, np.ones(len(ids), dtype=np.int64)


if __name__ == "__main__":
    main()
