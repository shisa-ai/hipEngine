"""Generate the Surya OCR 2 torch oracle fixtures with real transformers.

Oracle lineage: stock transformers 5.x ``Qwen3_5ForConditionalGeneration``
(Apache-2.0) running the pinned ``datalab-to/surya-ocr-2`` checkpoint in
float32 on CPU.  This script is a test-fixture generator only; the hipEngine
hot path never imports torch.

Captures, per case:

- ``text``: chat-template prompt without an image — input_ids, first
  logits, teacher-forced cached decode-step logits, selected cache states.
- ``image``: a deterministic synthetic 256x256 page image through the
  full pipeline — processor outputs (pixel_values, image_grid_thw),
  input_ids with the expanded image-pad run, vision boundary tensors
  (patch-embed output, vision-tower output pre-merger, merged image
  features), the decoder's computed mRoPE position_ids, first logits,
  teacher-forced cached decode-step logits, and selected cache states
  after prefill and after the decode steps.

Cache states are captured for layers 0 and 20 (gated-DeltaNet linear
attention: conv + recurrent state, fp32 per ``mamba_ssm_dtype``) and
layers 3 and 23 (full attention: key/value tensors) so the NumPy CPU
reference can check cache ownership, GDN state advancement, and cached
attention separately.

Usage:
    python3 scripts/surya_oracle_torch.py \
        [--out-dir tests/fixtures/surya] [--device cpu] [--steps 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_OUT_DIR = Path("tests/fixtures/surya")
CAPTURED_CACHE_LAYERS = (0, 3, 20, 23)
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23}

PROMPT_TEXT = "Transcribe this page."
# fixed teacher-forced continuation ids (deterministic; not model-sampled)
DECODE_STEP_IDS = [100, 5000, 20000, 42]


def _make_synthetic_page_rect(path: Path, width: int = 320, height: int = 192) -> None:
    """Non-square synthetic page: crosses the min-pixels edge and a
    different (12, 20) merged interpolation grid."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([12, 12, 300, 24], fill=(20, 20, 20))
    y = 36
    while y < 150:
        d.rectangle([12, y, 150, y + 5], fill=(40, 40, 40))
        d.rectangle([164, y, 308, y + 5], fill=(60, 60, 60))
        y += 14
    d.rectangle([12, 160, 200, 184], fill=(120, 120, 120))
    img.save(path)


def _make_synthetic_page(path: Path, size: int = 256) -> None:
    """Deterministic synthetic page: white ground, heading bar, text lines,
    a gray rule, and a small block — geometry multiple of 32 (256/32=8)."""

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # heading bar
    d.rectangle([16, 16, 240, 32], fill=(20, 20, 20))
    # body text lines (two-column-ish rhythm)
    y = 48
    while y < 200:
        d.rectangle([16, y, 120, y + 6], fill=(40, 40, 40))
        d.rectangle([136, y, 240, y + 6], fill=(40, 40, 40))
        y += 18
    # gray rule
    d.rectangle([16, 210, 240, 213], fill=(160, 160, 160))
    # small dark block (pseudo table/figure)
    d.rectangle([16, 224, 120, 248], fill=(90, 90, 90))
    img.save(path)


def _capture_cache(past: object, layers: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Extract per-layer cache tensors from a transformers Cache object.

    Layer kinds differ (full-attention DynamicLayer vs gated-DeltaNet
    conv/recurrent layers), so introspect attributes and record every
    float tensor; names encode layer index + attribute.
    """

    out: dict[str, np.ndarray] = {}
    layer_caches = list(past.layers) if hasattr(past, "layers") else list(past)
    for i in layers:
        lc = layer_caches[i]
        attrs = sorted(vars(lc)) if hasattr(lc, "__dict__") else []
        for attr in attrs:
            v = getattr(lc, attr)
            if torch.is_tensor(v) and v.is_floating_point():
                out[f"cache.layer{i}.{attr}"] = v.detach().float().numpy()
            elif isinstance(v, dict):
                for key, sv in sorted(v.items()):
                    if torch.is_tensor(sv) and sv.is_floating_point():
                        out[f"cache.layer{i}.{attr}[{key}]"] = (
                            sv.detach().float().numpy()
                        )
    return out


def _run_case(
    model: torch.nn.Module,
    processor: object,
    inputs: dict,
    steps: int,
) -> dict[str, np.ndarray]:
    """Prefill + teacher-forced cached steps with boundary capture."""

    cap: dict[str, np.ndarray] = {}

    # decoder-side captures: 3-axis mRoPE positions (via the rotary module,
    # which sees every call) and the final-norm output
    def _rotary_hook(_module, args, _kwargs, output):
        if len(args) >= 2 and torch.is_tensor(args[1]):
            arr = args[1].detach().int().numpy()
            if "position_ids_prefill" not in cap:
                cap["position_ids_prefill"] = arr
            cap["position_ids_last"] = arr

    def _final_norm_hook(_module, _args, output):
        if "final_norm_out" not in cap:
            cap["final_norm_out"] = output.detach().float().numpy()

    model.model.language_model.rotary_emb.register_forward_hook(
        _rotary_hook, with_kwargs=True
    )
    model.model.language_model.norm.register_forward_hook(_final_norm_hook)

    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    cap["logits_first"] = out.logits[0, -1].detach().float().numpy()
    past = out.past_key_values
    for k, v in _capture_cache(past, CAPTURED_CACHE_LAYERS).items():
        cap[f"prefill.{k}"] = v

    for s in range(steps):
        tok = torch.tensor([[DECODE_STEP_IDS[s]]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
        cap[f"logits_step{s}"] = out.logits[0, -1].detach().float().numpy()
    for k, v in _capture_cache(past, CAPTURED_CACHE_LAYERS).items():
        cap[f"steps.{k}"] = v
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image  # noqa: F401  (used by the image cases)

    page_path = args.out_dir / "page_small.png"
    if not page_path.exists():
        _make_synthetic_page(page_path)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained("datalab-to/surya-ocr-2")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        "datalab-to/surya-ocr-2", dtype=torch.float32
    ).to(args.device)
    model.eval()

    meta: dict = {"decode_step_ids": DECODE_STEP_IDS, "steps": args.steps}

    # ---- image case ----------------------------------------------------
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
    from PIL import Image

    page_img = Image.open(page_path).convert("RGB")
    proc = processor(
        text=[prompt], images=[page_img], return_tensors="pt"
    ).to(args.device)
    inputs = {
        "input_ids": proc["input_ids"],
        "attention_mask": proc["attention_mask"],
        "pixel_values": proc["pixel_values"],
        "image_grid_thw": proc["image_grid_thw"],
        "mm_token_type_ids": proc["mm_token_type_ids"],
    }
    meta["image_grid_thw"] = proc["image_grid_thw"].tolist()
    assert "mm_token_type_ids" in proc, (
        "processor must return mm_token_type_ids for multimodal RoPE"
    )
    meta["image_input_ids_len"] = int(proc["input_ids"].shape[1])

    image_caps: dict[str, np.ndarray] = {
        "input_ids": proc["input_ids"].numpy(),
        "attention_mask": proc["attention_mask"].numpy(),
        "pixel_values": proc["pixel_values"].numpy(),
        "image_grid_thw": proc["image_grid_thw"].numpy(),
        "mm_token_type_ids": proc["mm_token_type_ids"].numpy(),
    }

    # vision boundaries via one-shot hooks around the prefill forward
    vision_caps: dict[str, np.ndarray] = {}

    def _hook(name):
        def hook(_m, _i, output):
            if isinstance(output, tuple):
                t = output[0]
            elif hasattr(output, "last_hidden_state"):
                t = output.last_hidden_state
            else:
                t = output
            if t is None:
                raise RuntimeError(f"vision hook {name} captured None")
            vision_caps[name] = t.detach().float().numpy()

        return hook

    h1 = model.model.visual.patch_embed.register_forward_hook(_hook("vision_patch_embed"))
    h2 = model.model.visual.register_forward_hook(_hook("vision_tower_out"))
    h3 = model.model.visual.merger.register_forward_hook(_hook("vision_merged"))

    caps = _run_case(model, processor, inputs, args.steps)
    for h in (h1, h2, h3):
        h.remove()
    # the hooks fire again during nothing else (decode steps only touch the
    # decoder), so vision_caps holds exactly the prefill vision tensors
    image_caps.update(vision_caps)
    image_caps.update(caps)
    np.savez_compressed(args.out_dir / "oracle_image.npz", **image_caps)

    # ---- text-only case --------------------------------------------------
    messages_text = [{"role": "user", "content": [{"type": "text", "text": PROMPT_TEXT}]}]
    prompt_text = processor.apply_chat_template(
        messages_text, add_generation_prompt=True
    )
    proc_t = processor(text=[prompt_text], return_tensors="pt").to(args.device)
    inputs_t = {
        "input_ids": proc_t["input_ids"],
        "attention_mask": proc_t["attention_mask"],
    }
    meta["text_input_ids_len"] = int(proc_t["input_ids"].shape[1])
    text_caps: dict[str, np.ndarray] = {
        "input_ids": proc_t["input_ids"].numpy(),
        "attention_mask": proc_t["attention_mask"].numpy(),
    }
    text_caps.update(_run_case(model, processor, inputs_t, args.steps))
    np.savez_compressed(args.out_dir / "oracle_text.npz", **text_caps)

    # ---- rect (non-square) case -------------------------------------------
    rect_path = args.out_dir / "page_rect.png"
    if not rect_path.exists():
        _make_synthetic_page_rect(rect_path)
    messages_r = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(rect_path)},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    prompt_r = processor.apply_chat_template(messages_r, add_generation_prompt=True)
    rect_img = Image.open(rect_path).convert("RGB")
    proc_r = processor(text=[prompt_r], images=[rect_img], return_tensors="pt").to(args.device)
    inputs_r = {
        "input_ids": proc_r["input_ids"],
        "attention_mask": proc_r["attention_mask"],
        "pixel_values": proc_r["pixel_values"],
        "image_grid_thw": proc_r["image_grid_thw"],
        "mm_token_type_ids": proc_r["mm_token_type_ids"],
    }
    meta["rect_grid_thw"] = proc_r["image_grid_thw"].tolist()
    meta["rect_input_ids_len"] = int(proc_r["input_ids"].shape[1])
    rect_caps: dict[str, np.ndarray] = {
        "input_ids": proc_r["input_ids"].numpy(),
        "attention_mask": proc_r["attention_mask"].numpy(),
        "pixel_values": proc_r["pixel_values"].numpy(),
        "image_grid_thw": proc_r["image_grid_thw"].numpy(),
        "mm_token_type_ids": proc_r["mm_token_type_ids"].numpy(),
    }
    rv: dict[str, np.ndarray] = {}

    def _rhook(name):
        def hook(_m, _i, output):
            t = output[0] if isinstance(output, tuple) else output
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            rv[name] = t.detach().float().numpy()
        return hook

    h1 = model.model.visual.patch_embed.register_forward_hook(_rhook("vision_patch_embed"))
    h2 = model.model.visual.register_forward_hook(_rhook("vision_tower_out"))
    h3 = model.model.visual.merger.register_forward_hook(_rhook("vision_merged"))
    rcaps = _run_case(model, processor, inputs_r, args.steps)
    for h in (h1, h2, h3):
        h.remove()
    rect_caps.update(rv)
    rect_caps.update(rcaps)
    np.savez_compressed(args.out_dir / "oracle_rect.npz", **rect_caps)

    # ---- meta ------------------------------------------------------------
    tok = processor.tokenizer
    meta["prompt_image"] = prompt
    meta["prompt_text"] = prompt_text
    meta["eos_token_id"] = 2  # tokenizer + generation_config agree
    meta["image_token_id"] = 11
    meta["decode_step_token_strings"] = tok.convert_ids_to_tokens(DECODE_STEP_IDS)
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"wrote {args.out_dir}/oracle_image.npz, oracle_text.npz, "
        "oracle_rect.npz, meta.json"
    )


if __name__ == "__main__":
    main()
