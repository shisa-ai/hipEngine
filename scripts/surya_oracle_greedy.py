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
                {"type": "text", "text": PROMPT_TEXT},
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

    caps = {
        "input_ids": proc["input_ids"].numpy(),
        "attention_mask": proc["attention_mask"].numpy(),
        "pixel_values": proc["pixel_values"].numpy(),
        "image_grid_thw": proc["image_grid_thw"].numpy(),
        "mm_token_type_ids": proc["mm_token_type_ids"].numpy(),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("all", "image", "fullpage", "corpus"), default="all"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
