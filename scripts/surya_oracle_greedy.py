"""Generate the Surya OCR 2 torch greedy-decode reference fixtures.

Oracle lineage: stock transformers 5.x ``Qwen3_5ForConditionalGeneration``
(Apache-2.0) running the pinned ``datalab-to/surya-ocr-2`` checkpoint in
float32 on CPU.  Test-fixture generator only; the hipEngine hot path never
imports torch.

Two cases, both greedy (argmax) with the model's own logits:

- ``image``    — ``page_small.png`` (256x256) -> ``oracle_greedy.json``.
  This is the reference that ``test_surya_e2e``, ``test_surya_public_api`` and
  ``test_surya_gpu`` all gate on. It had no committed generator before this
  script: ``scripts/surya_oracle_torch.py`` captures logits and cache states
  but never emits greedy ids.
- ``fullpage`` — ``page_full.png`` (1024x1024, real words) ->
  ``oracle_fullpage_greedy.json`` plus ``oracle_fullpage.npz`` (vision_merged,
  logits_first). The small page's continuation is a degenerate repeated
  ``<ul><li>`` run, so parity against it is a weak gate; the full page decodes
  a long, non-degenerate layout-JSON sequence.
- ``corpus``   — ``page_columns.png`` (two-column) and ``page_list.png``
  (numbered checklist), both 512x512, -> ``oracle_corpus.json``. Layouts the
  single-column fixtures cannot produce, so the GPU lane cannot pass them by
  generalizing from the pages already covered.
- ``bench``    — the representative document types written by
  ``scripts/surya_bench_pages.py`` (Japanese, mixed script, dense small text,
  ruled table, blank page, degraded scan, block-heavy long page) ->
  ``oracle_bench.json``. These are the pages the tuning suite runs on, so they
  carry a torch fp32 oracle rather than only a CPU-reference basis.

Additive: regenerating the ``image`` case rewrites ``oracle_greedy.json`` and
nothing else. It does not touch ``oracle_image.npz`` / ``oracle_text.npz`` /
``oracle_rect.npz``, which were captured with an older transformers and are not
reproducible from ``scripts/surya_oracle_torch.py``.

Usage:
    python3 scripts/surya_oracle_greedy.py [--case all|image|fullpage|corpus] \
        [--out-dir tests/fixtures/surya] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surya_oracle_torch import (  # noqa: E402
    PROMPT_TEXT,
    _make_synthetic_page,
    _make_synthetic_page_columns,
    _make_synthetic_page_full,
    _make_synthetic_page_list,
)
from surya_bench_pages import page_filename  # noqa: E402

from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT  # noqa: E402

DEFAULT_OUT_DIR = Path("tests/fixtures/surya")
# matches the committed oracle_greedy.json (64 ids, no EOS before the limit)
IMAGE_MAX_TOKENS = 64
# the full page reaches EOS at 78; 96 leaves headroom to observe the stop
FULLPAGE_MAX_TOKENS = 96
# held-out layouts; both reach a natural EOS well inside this budget
CORPUS_MAX_TOKENS = 96
CORPUS_PAGES = {
    "columns": _make_synthetic_page_columns,
    "list": _make_synthetic_page_list,
}
# Representative document types for the benchmark suite. The pages themselves
# are produced by scripts/surya_bench_pages.py (seeded, byte-reproducible);
# this captures their torch fp32 reference ids so the GPU lanes can be gated
# against torch rather than only against the CPU reference.
BENCH_PAGES = ("ja", "mixed", "dense", "table", "blank", "scan", "long")

# Full-page HTML transcription protocol oracle: the same page names, but scored
# on the fitting acceptance fixtures and driven by the checkpoint's real
# training-time prompt instead of the ad-hoc "Transcribe this page." text. The
# budgets are the smallest that let every page reach a natural EOS, measured on
# the fp32 HIP lane; a budget that truncates would gate only a prefix.
PROTOCOL_MAX_TOKENS = {
    "ja": 600,
    "mixed": 600,
    "dense": 1500,
    "table": 640,
    "blank": 64,
    "scan": 640,
    "long": 2600,
}
# The blank page stops almost immediately; the long page needs room to finish.
# The default is generous on purpose: at 128 the Japanese, mixed, dense and scan
# pages all hit the cap instead of reaching EOS, which would gate only their
# first 128 tokens rather than their complete transcription.
BENCH_MAX_TOKENS = {"blank": 32, "long": 320, "scan": 512}
BENCH_DEFAULT_MAX_TOKENS = 384
EOS_TOKEN_ID = 2  # tokenizer + generation_config agree
# canonical archive key order, matching the committed oracle_fullpage.npz
_FULLPAGE_NPZ_KEYS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
    "logits_first",
    "greedy_ids",
    "vision_merged",
)


def _greedy_from_page(
    model: object,
    processor: object,
    page_path: Path,
    *,
    device: str,
    max_tokens: int,
    prompt: str = PROMPT_TEXT,
) -> tuple[dict, dict, list[int], str]:
    """Run one image page through prefill + greedy decode.

    Returns ``(processor_inputs, captured, ids, text)`` where ``captured`` holds
    ``vision_merged`` and ``logits_first`` when the merger hook fired.
    """

    from PIL import Image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(page_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    page_img = Image.open(page_path).convert("RGB")
    proc = processor(text=[prompt], images=[page_img], return_tensors="pt").to(device)
    inputs = {
        "input_ids": proc["input_ids"],
        "attention_mask": proc["attention_mask"],
        "pixel_values": proc["pixel_values"],
        "image_grid_thw": proc["image_grid_thw"],
        "mm_token_type_ids": proc["mm_token_type_ids"],
    }
    assert "mm_token_type_ids" in proc, (
        "processor must return mm_token_type_ids for multimodal RoPE"
    )

    captured: dict[str, np.ndarray] = {}

    def _merger_hook(_module, _args, output):
        t = output[0] if isinstance(output, tuple) else output
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        captured["vision_merged"] = t.detach().float().cpu().numpy()

    handle = model.model.visual.merger.register_forward_hook(_merger_hook)
    try:
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
    finally:
        handle.remove()
    past = out.past_key_values
    logits = out.logits[0, -1].detach().float()
    captured["logits_first"] = logits.cpu().numpy().copy()

    ids: list[int] = []
    for _ in range(max_tokens):
        nxt = int(torch.argmax(logits).item())
        if nxt == EOS_TOKEN_ID:
            break
        ids.append(nxt)
        tok = torch.tensor([[nxt]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].detach().float()

    # .cpu() before .numpy(): these were only ever captured on CPU, so the
    # conversion broke as soon as the generator ran on the GPU device.
    caps = {
        "input_ids": proc["input_ids"].cpu().numpy(),
        "attention_mask": proc["attention_mask"].cpu().numpy(),
        "pixel_values": proc["pixel_values"].cpu().numpy(),
        "image_grid_thw": proc["image_grid_thw"].cpu().numpy(),
        "mm_token_type_ids": proc["mm_token_type_ids"].cpu().numpy(),
    }
    caps.update(captured)
    return caps, captured, ids, processor.tokenizer.decode(ids)


def _run_image(model, processor, out_dir: Path, device: str) -> None:
    page_path = out_dir / "page_small.png"
    if not page_path.exists():
        _make_synthetic_page(page_path)
    caps, _captured, ids, text = _greedy_from_page(
        model, processor, page_path, device=device, max_tokens=IMAGE_MAX_TOKENS
    )
    (out_dir / "oracle_greedy.json").write_text(
        json.dumps({"ids": ids, "text": text}, indent=1)
    )
    print(f"wrote {out_dir}/oracle_greedy.json — {len(ids)} greedy ids")
    print(f"  grid_thw={caps['image_grid_thw'].tolist()} "
          f"seq={int(caps['input_ids'].shape[1])}")


def _run_fullpage(model, processor, out_dir: Path, device: str) -> None:
    page_path = out_dir / "page_full.png"
    if not page_path.exists():
        _make_synthetic_page_full(page_path)
    caps, captured, ids, text = _greedy_from_page(
        model, processor, page_path, device=device, max_tokens=FULLPAGE_MAX_TOKENS
    )
    if "vision_merged" not in captured:
        raise RuntimeError("merger hook did not fire; cannot write vision_merged")
    # The merger hook fires mid-forward, so insertion order alone would put
    # vision_merged before logits_first. Write an explicit canonical order so
    # regenerating produces a byte-identical archive.
    caps["greedy_ids"] = np.asarray(ids, dtype=np.int64)
    ordered = {key: caps[key] for key in _FULLPAGE_NPZ_KEYS}
    np.savez_compressed(out_dir / "oracle_fullpage.npz", **ordered)
    (out_dir / "oracle_fullpage_greedy.json").write_text(
        json.dumps({"ids": ids, "text": text}, indent=1)
    )
    print(
        f"wrote {out_dir}/oracle_fullpage.npz and "
        f"oracle_fullpage_greedy.json — {len(ids)} greedy ids"
    )
    print(f"  grid_thw={caps['image_grid_thw'].tolist()} "
          f"seq={int(caps['input_ids'].shape[1])}")


def _run_corpus(model, processor, out_dir: Path, device: str) -> None:
    """Held-out layouts: a two-column page and a numbered-list page.

    Both differ structurally from the single-column fixtures, so the GPU lane
    cannot pass them by generalizing from the pages already covered. The pages
    are written as fixtures; the oracle is a single JSON of ids+text per page.
    """

    oracle: dict[str, dict[str, object]] = {}
    for name, draw in CORPUS_PAGES.items():
        page_path = out_dir / f"page_{name}.png"
        draw(page_path)
        caps, _captured, ids, text = _greedy_from_page(
            model, processor, page_path, device=device, max_tokens=CORPUS_MAX_TOKENS
        )
        oracle[name] = {"ids": ids, "text": text}
        print(
            f"wrote {out_dir}/page_{name}.png — {len(ids)} greedy ids, "
            f"grid_thw={caps['image_grid_thw'].tolist()}"
        )
    (out_dir / "oracle_corpus.json").write_text(json.dumps(oracle, indent=1))
    print(f"wrote {out_dir}/oracle_corpus.json — {len(oracle)} pages")


def _run_bench(model, processor, out_dir: Path, device: str,
               only: list[str] | None = None) -> None:
    """Capture torch fp32 reference ids for the representative bench pages.

    The pages are inputs, not outputs: ``scripts/surya_bench_pages.py`` writes
    them, and this records what torch fp32 greedily produces so the hipEngine
    lanes have a real oracle and the text can be judged for OCR quality rather
    than only for id parity.

    ``--only`` restricts the capture to a subset; a partial capture writes only
    those pages into ``oracle_bench.json``, so use a scratch ``--out-dir`` when
    the result is a device-equivalence spot check rather than the fixture.
    """

    names = list(BENCH_PAGES) if not only else list(only)
    unknown = [name for name in names if name not in BENCH_PAGES]
    if unknown:
        raise SystemExit(
            f"unknown bench page(s) {unknown}; known: {list(BENCH_PAGES)}"
        )
    oracle: dict[str, dict[str, object]] = {}
    for name in names:
        page_path = out_dir / f"page_{name}.png"
        if not page_path.exists():
            raise SystemExit(
                f"missing {page_path}; run scripts/surya_bench_pages.py first"
            )
        max_tokens = BENCH_MAX_TOKENS.get(name, BENCH_DEFAULT_MAX_TOKENS)
        caps, _captured, ids, text = _greedy_from_page(
            model, processor, page_path, device=device, max_tokens=max_tokens
        )
        oracle[name] = {
            "ids": ids,
            "text": text,
            "max_tokens": max_tokens,
            "reached_limit": len(ids) >= max_tokens,
            "grid_thw": caps["image_grid_thw"].tolist(),
        }
        print(
            f"wrote oracle for page_{name}.png — {len(ids)} greedy ids "
            f"(limit {max_tokens}), grid_thw={caps['image_grid_thw'].tolist()}"
        )
    (out_dir / "oracle_bench.json").write_text(json.dumps(oracle, indent=1))
    print(f"wrote {out_dir}/oracle_bench.json — {len(oracle)} pages")


def _run_protocol(model, processor, out_dir: Path, device: str,
                  only: list[str] | None = None,
                  page_dir: Path | None = None) -> None:
    """Capture torch fp32 ids for the full-page HTML transcription protocol.

    This is the reference the transcription acceptance test gates on. The
    ad-hoc-prompt bench oracle cannot serve: the model's continuation there is
    layout JSON or a degenerate repeat, so it says nothing about whether the
    page was transcribed.
    """

    names = list(PROTOCOL_MAX_TOKENS) if not only else list(only)
    unknown = [name for name in names if name not in PROTOCOL_MAX_TOKENS]
    if unknown:
        raise SystemExit(
            f"unknown protocol page(s) {unknown}; "
            f"known: {sorted(PROTOCOL_MAX_TOKENS)}"
        )
    oracle: dict[str, dict[str, object]] = {}
    pages = page_dir if page_dir is not None else out_dir
    for name in names:
        page_name = page_filename(name)
        page_path = pages / page_name
        if not page_path.exists():
            raise SystemExit(
                f"missing {page_path}; run scripts/surya_bench_pages.py first"
            )
        max_tokens = PROTOCOL_MAX_TOKENS[name]
        caps, _captured, ids, text = _greedy_from_page(
            model, processor, page_path, device=device, max_tokens=max_tokens,
            prompt=FULL_PAGE_HTML_PROMPT,
        )
        reached_limit = len(ids) >= max_tokens
        oracle[name] = {
            "page": page_name,
            "prompt": FULL_PAGE_HTML_PROMPT,
            "ids": ids,
            "text": text,
            "max_tokens": max_tokens,
            "reached_limit": reached_limit,
            "finish_reason": "length" if reached_limit else "eos",
            "grid_thw": caps["image_grid_thw"].tolist(),
        }
        print(
            f"wrote protocol oracle for {page_name} — {len(ids)} greedy ids "
            f"(limit {max_tokens}, {'length' if reached_limit else 'eos'}), "
            f"grid_thw={caps['image_grid_thw'].tolist()}"
        )
    (out_dir / "oracle_fullpage_protocol.json").write_text(json.dumps(oracle, indent=1))
    print(f"wrote {out_dir}/oracle_fullpage_protocol.json — {len(oracle)} pages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("all", "image", "fullpage", "corpus", "bench", "protocol"),
        default="all",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--only", default=None,
        help="comma-separated page names, applied to --case bench/protocol",
    )
    parser.add_argument(
        "--page-dir", type=Path, default=None,
        help="directory holding the page fixtures; defaults to --out-dir",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    only = (
        [name.strip() for name in args.only.split(",") if name.strip()]
        if args.only else None
    )

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained("datalab-to/surya-ocr-2")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "datalab-to/surya-ocr-2", dtype=torch.float32
    ).to(args.device)
    model.eval()

    if args.case in ("all", "image"):
        _run_image(model, processor, args.out_dir, args.device)
    if args.case in ("all", "fullpage"):
        _run_fullpage(model, processor, args.out_dir, args.device)
    if args.case in ("all", "corpus"):
        _run_corpus(model, processor, args.out_dir, args.device)
    if args.case in ("all", "bench"):
        _run_bench(model, processor, args.out_dir, args.device, only)
    if args.case in ("all", "protocol"):
        _run_protocol(model, processor, args.out_dir, args.device, only,
                      args.page_dir)


if __name__ == "__main__":
    main()
