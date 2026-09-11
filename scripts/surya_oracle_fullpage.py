"""Generate the Surya OCR 2 full-page torch oracle fixture.

Oracle lineage: stock transformers 5.x ``Qwen3_5ForConditionalGeneration``
(Apache-2.0) running the pinned ``datalab-to/surya-ocr-2`` checkpoint in
float32 on CPU.  Test-fixture generator only; the hipEngine hot path never
imports torch.

Why a separate case: ``page_small`` (256x256) and ``page_rect`` (320x192) are
bar patterns whose oracle continuation is a degenerate repeated
``<ul><li>`` sequence, so greedy-ID parity against them is a weak gate.
``page_full`` (1024x1024, real words) decodes a long, non-degenerate
layout-JSON sequence where a small numerical difference flips an argmax.

This script is additive: it writes only ``oracle_fullpage.npz`` and
``oracle_fullpage_greedy.json`` and never touches the other fixtures.

Usage:
    python3 scripts/surya_oracle_fullpage.py \
        [--out-dir tests/fixtures/surya] [--device cpu] [--max-tokens 96]
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
    _make_synthetic_page_full,
)

DEFAULT_OUT_DIR = Path("tests/fixtures/surya")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    page_path = args.out_dir / "page_full.png"
    if not page_path.exists():
        _make_synthetic_page_full(page_path)

    processor = AutoProcessor.from_pretrained("datalab-to/surya-ocr-2")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "datalab-to/surya-ocr-2", dtype=torch.float32
    ).to(args.device)
    model.eval()

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
    proc = processor(text=[prompt], images=[page_img], return_tensors="pt").to(
        args.device
    )
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

    merged_cap: dict[str, np.ndarray] = {}

    def _merger_hook(_module, _args, output):
        t = output[0] if isinstance(output, tuple) else output
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        merged_cap["vision_merged"] = t.detach().float().cpu().numpy()

    handle = model.model.visual.merger.register_forward_hook(_merger_hook)

    eos_token_id = 2  # tokenizer + generation_config agree
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    handle.remove()
    past = out.past_key_values
    logits = out.logits[0, -1].detach().float()
    logits_first = logits.cpu().numpy().copy()

    ids: list[int] = []
    for _ in range(args.max_tokens):
        nxt = int(torch.argmax(logits).item())
        if nxt == eos_token_id:
            break
        ids.append(nxt)
        tok = torch.tensor([[nxt]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].detach().float()

    text = processor.tokenizer.decode(ids)
    caps = {
        "input_ids": proc["input_ids"].numpy(),
        "attention_mask": proc["attention_mask"].numpy(),
        "pixel_values": proc["pixel_values"].numpy(),
        "image_grid_thw": proc["image_grid_thw"].numpy(),
        "mm_token_type_ids": proc["mm_token_type_ids"].numpy(),
        "logits_first": logits_first,
        "greedy_ids": np.asarray(ids, dtype=np.int64),
    }
    caps.update(merged_cap)
    np.savez_compressed(args.out_dir / "oracle_fullpage.npz", **caps)
    (args.out_dir / "oracle_fullpage_greedy.json").write_text(
        json.dumps({"ids": ids, "text": text}, indent=1)
    )
    print(
        f"wrote {args.out_dir}/oracle_fullpage.npz and "
        f"oracle_fullpage_greedy.json — {len(ids)} greedy ids"
    )
    print(f"grid_thw={proc['image_grid_thw'].tolist()} "
          f"seq={int(proc['input_ids'].shape[1])}")
    print(f"text={text[:200]!r}")


if __name__ == "__main__":
    main()
