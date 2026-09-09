"""EVIE-4.5B torch oracle: capture reference forward outputs for hipEngine parity.

Loads the reference ColQwen3_5 (colpali_engine from ~/EVIE, stock transformers
Qwen3_5Model backbone) in float32 on CPU, runs one document image and one text
query through the full pipeline, and saves a .npz fixture with:

- processor outputs (input_ids, attention_mask, pixel_values, image_grid_thw)
  so the hipEngine path can reproduce identical model inputs,
- the vision-tower output (embedded visual features after the merger),
- the language-model final hidden state,
- the un-normalized full 2048-d projection,
- the per-token L2-normalized 128-d multi-vector embeddings (head d128),
- the MaxSim score between query and document embeddings.

Bidirectional attention is enabled exactly as the reference deployment does
(`model.enable_bidirectional_attention()`), and rope deltas are reset between
the document and query forwards as in the model card quick start.

Usage:
    PYTHONPATH=~/EVIE/colpali python3 scripts/evie_oracle_torch.py \
        [--out tests/fixtures/evie/evie_4p5b_doc_query.npz] [--device cpu]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

SNAPSHOT = sorted(
    glob.glob(
        str(
            Path.home()
            / ".cache/huggingface/hub/models--tencent--EVIE-4.5B/snapshots/*/model.safetensors"
        )
    )
)
DEFAULT_OUT = Path("tests/fixtures/evie/evie_4p5b_doc_query.npz")


def _resolve_repo_path() -> str:
    local = Path(__file__).resolve().parent.parent / "evie_ref"
    if (local / "colpali_engine").exists():
        return str(local)
    home = Path.home() / "EVIE" / "colpali"
    if (home / "colpali_engine").exists():
        return str(home)
    print("colpali_engine not found (expected ~/EVIE/colpali)", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "tests/fixtures/evie/doc_page.png",
        help="document page image (generated if missing)",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image

    repo = _resolve_repo_path()
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
    from colpali_engine.models.qwen3_5.colqwen3_5.modeling_colqwen3_5 import (
        set_active_head,
    )

    snapshot = Path(SNAPSHOT[0]).parent
    print(f"loading {snapshot} on {args.device} (float32)...")
    model = ColQwen3_5.from_pretrained(
        str(snapshot),
        torch_dtype=torch.float32,
        device_map=args.device,
        attn_implementation="sdpa",
    ).eval()
    model.enable_bidirectional_attention()
    set_active_head(model, args.head)
    processor = ColQwen3_5Processor.from_pretrained(str(snapshot))

    # ------------------------------------------------------------------ image
    if not args.image.exists():
        args.image.parent.mkdir(parents=True, exist_ok=True)
        # Deterministic synthetic "document": white page, black text-like bars.
        rng = np.random.default_rng(20260909)
        page = np.full((448, 336, 3), 255, dtype=np.uint8)
        y = 24
        while y < 424:
            width = int(rng.integers(80, 300))
            x0 = int(rng.integers(16, 40))
            page[y : y + 8, x0 : x0 + width] = 0
            y += 16
        Image.fromarray(page).save(args.image)
        print(f"generated synthetic page {args.image}")
    image = Image.open(args.image).convert("RGB")

    # ------------------------------------------------------------------ query
    query = "What is the total quarterly revenue?"

    image_batch = processor.process_images([image]).to(args.device)
    print("doc input_ids shape:", image_batch["input_ids"].shape)
    print("doc pixel_values shape:", image_batch["pixel_values"].shape)
    print("doc image_grid_thw:", image_batch["image_grid_thw"].tolist())

    # Capture the vision-tower output and final hidden state via hooks.
    captured: dict[str, torch.Tensor] = {}

    def _visual_hook(_module, inputs, output):
        # Qwen3_5 vision tower returns BaseModelOutputWithPooling; the
        # post-merger visual embeddings are in ``pooler_output``.
        if "visual" not in captured:
            captured["visual"] = (
                output.pooler_output.detach().to("cpu", torch.float32)
            )

    def _final_norm_hook(key: str):
        def hook(_module, inputs, output):
            captured[key] = output.detach().to("cpu", torch.float32)

        return hook

    visual_handle = model.visual.register_forward_hook(_visual_hook)
    norm_handle = model.language_model.norm.register_forward_hook(_final_norm_hook("last_hidden"))

    with torch.inference_mode():
        doc_embeddings = model(**image_batch)
    visual_handle.remove()
    norm_handle.remove()

    model.rope_deltas = None  # reset RoPE deltas before the text query forward
    query_batch = processor.process_queries([query]).to(args.device)
    print("query input_ids shape:", query_batch["input_ids"].shape)
    print("query input_ids:", query_batch["input_ids"][0].tolist())

    query_norm_handle = model.language_model.norm.register_forward_hook(
        _final_norm_hook("query_last_hidden")
    )
    with torch.inference_mode():
        query_embeddings = model(**query_batch)
    query_norm_handle.remove()

    # Un-normalized full projections from the captured final hidden states.
    last_hidden = captured["last_hidden"]
    query_last_hidden = captured["query_last_hidden"]
    with torch.inference_mode():
        full_proj = model.custom_text_proj(last_hidden.to(args.device)).cpu()
        query_full_proj = model.custom_text_proj(query_last_hidden.to(args.device)).cpu()

    scores = processor.score(query_embeddings, doc_embeddings)
    print("MaxSim score:", scores.tolist())

    out = {
        "query_text": np.array(query),
        "input_ids": image_batch["input_ids"].cpu().numpy(),
        "attention_mask": image_batch["attention_mask"].cpu().numpy(),
        "pixel_values": image_batch["pixel_values"].cpu().numpy(),
        "image_grid_thw": image_batch["image_grid_thw"].cpu().numpy(),
        "visual_features": captured["visual"].numpy(),
        "last_hidden": last_hidden.cpu().numpy(),
        "full_proj_2048": full_proj.cpu().numpy(),
        "doc_embeddings_128": doc_embeddings.cpu().numpy(),
        "query_input_ids": query_batch["input_ids"].cpu().numpy(),
        "query_attention_mask": query_batch["attention_mask"].cpu().numpy(),
        "query_embeddings_128": query_embeddings.cpu().numpy(),
        "query_hidden": query_last_hidden.numpy(),
        "query_full_proj_2048": query_full_proj.numpy(),
        "maxsim_score": scores.cpu().numpy(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
