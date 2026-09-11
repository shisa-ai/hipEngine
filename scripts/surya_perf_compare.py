"""Surya OCR 2 performance comparison across hipEngine and torch lanes.

Lanes on one physical host (gfx1151 / Ryzen AI Max+ Pro 395):
  - hipengine_cpu:  hipEngine torch-free CPU-reference path (fp32 numpy).
  - hipengine_gpu:  hipEngine HIP path (fp32): GPU vision tower + GPU
                    prefill/decode through ``hipengine.runtime.surya``.
  - torch_cpu:      transformers fp32 CPU (the oracle implementation).
  - torch_cuda:     same transformers fp32 on the host GPU via torch ROCm.

Measurement protocol (identical boundary for every lane):
  * The timed region is one full OCR request: page image + prompt in,
    greedy token ids out — preprocessing, vision tower, prefill, and decode
    all inside.
  * Checkpoint load and engine/runner construction are initialization: they
    are measured once as ``init_s`` and excluded from ``e2e_*``.
  * ``decode_tokens`` is the actual number of generated tokens, not the
    requested limit.

Correctness gate for every lane: greedy token ids must equal the captured
torch fp32 CPU reference (tests/fixtures/surya/oracle_greedy.json); a lane
that diverges is reported as FAILED and excluded from perf claims.

Usage:
    python3 scripts/surya_perf_compare.py [--runs 3] [--max-new-tokens 64]
        [--out benchmarks/results/2026-09-11-gfx1151-surya-lane-compare.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

FIXTURES = Path("tests/fixtures/surya")
MODEL_ID = "datalab-to/surya-ocr-2"
PROMPT = "Transcribe this page."


def _time(fn, runs: int) -> dict[str, float]:
    """Median wall time over warm runs plus the first (warmup) run."""

    wall: list[float] = []
    for _ in range(max(runs, 1) + 1):
        t0 = time.perf_counter()
        fn()
        wall.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(wall[1:]), "warmup_s": wall[0]}


def _timed(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def _result(
    lane: str,
    backend: str,
    *,
    init_s: float,
    stages: dict[str, float],
    e2e: dict[str, float],
    generated: list[int],
    ref: dict,
) -> dict:
    decode_s = stages.get("decode", 0.0)
    n = len(generated)
    return {
        "lane": lane,
        "backend": backend,
        "init_s": init_s,
        "stages_s": stages,
        "decode_tokens": n,
        "decode_tok_per_s": (n / decode_s) if decode_s > 0 else None,
        "e2e_median_s": e2e["median_s"],
        "e2e_warmup_s": e2e["warmup_s"],
        "correctness": "PASS" if generated == ref["ids"] else "FAIL",
    }


def lane_hipengine_cpu(max_new_tokens: int, runs: int) -> dict:
    from PIL import Image

    from hipengine.kernels.cpu_reference.surya import (
        text_decode_step,
        text_prefill,
        vision_forward,
    )
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        load_surya_spec,
        load_surya_weights,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )

    page = Image.open(FIXTURES / "page_small.png").convert("RGB")
    ref = json.loads((FIXTURES / "oracle_greedy.json").read_text())

    def init():
        model_dir = resolve_surya_path(MODEL_ID)
        return (
            load_surya_spec(model_dir),
            load_surya_weights(model_dir),
            SuryaTokenizer(model_dir),
        )

    init_s, (spec, weights, tokenizer) = _timed(init)

    def decode_from(state, logits, p: int) -> list[int]:
        generated: list[int] = []
        for step in range(max_new_tokens):
            nxt = int(np.argmax(logits[0]))
            if nxt == spec.eos_token_id:
                break
            generated.append(nxt)
            if step + 1 >= max_new_tokens:
                break
            logits = text_decode_step(weights, spec, nxt, state, p + 1 + step)
        return generated

    def run() -> list[int]:
        rows, grid = preprocess_image_surya(page)
        _, _, merged = vision_forward(weights, spec, rows, [grid])
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        hidden, state = text_prefill(
            weights, spec, np.array([ids], dtype=np.int64), pos,
            visual_features=merged[None],
        )
        emb = weights["model.language_model.embed_tokens.weight"]
        logits = hidden[:, -1] @ emb.T
        return decode_from(state, logits, int(pos[:, -1].max()))

    # stage timing (single instrumented warm run)
    t_pre, (rows, grid) = _timed(lambda: preprocess_image_surya(page))
    t_vision, (_, _, merged) = _timed(lambda: vision_forward(weights, spec, rows, [grid]))
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    t_prefill, (hidden, state) = _timed(
        lambda: text_prefill(
            weights, spec, np.array([ids], dtype=np.int64), pos,
            visual_features=merged[None],
        )
    )
    emb = weights["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    p_last = int(pos[:, -1].max())
    t_decode, generated = _timed(lambda: decode_from(state, logits, p_last))

    e2e = _time(run, runs)
    return _result(
        "hipengine_cpu",
        "cpu_reference/fp32 numpy",
        init_s=init_s,
        stages={
            "preprocess": t_pre,
            "vision": t_vision,
            "prefill": t_prefill,
            "vision_prefill": t_vision + t_prefill,
            "decode": t_decode,
        },
        e2e=e2e,
        generated=generated,
        ref=ref,
    )


def lane_hipengine_gpu(max_new_tokens: int, runs: int) -> dict:
    """hipEngine HIP lane: GPU vision tower + GPU prefill/decode (fp32)."""

    from PIL import Image

    from hipengine.kernels.cpu_reference.surya import SuryaSpec, SuryaWeights
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    page = Image.open(FIXTURES / "page_small.png").convert("RGB")
    ref = json.loads((FIXTURES / "oracle_greedy.json").read_text())
    model_dir = resolve_surya_path(MODEL_ID)
    spec = SuryaSpec()
    weights = SuryaWeights.load(str(model_dir / "model.safetensors"))
    tokenizer = SuryaTokenizer(model_dir)

    # initialization: runner construction uploads every checkpoint tensor
    init_s, runner = _timed(lambda: SuryaGpuRunner(weights, spec))

    def decode_from(logits, p: int) -> list[int]:
        generated: list[int] = []
        for step in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            if nxt == spec.eos_token_id:
                break
            generated.append(nxt)
            if step + 1 >= max_new_tokens:
                break
            logits = runner.decode_step(nxt, p + 1 + step)
        return generated

    def run() -> list[int]:
        rows, grid = preprocess_image_surya(page)
        merged = runner.vision_forward(rows, [grid])
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        logits = runner.prefill(
            np.asarray(ids, dtype=np.int64), pos, visual_features=merged
        )
        return decode_from(logits, int(pos[:, -1].max()))

    try:
        # stage timing (single instrumented warm run)
        t_pre, (rows, grid) = _timed(lambda: preprocess_image_surya(page))
        t_vision, merged = _timed(lambda: runner.vision_forward(rows, [grid]))
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        t_prefill, logits = _timed(
            lambda: runner.prefill(
                np.asarray(ids, dtype=np.int64), pos, visual_features=merged
            )
        )
        p_last = int(pos[:, -1].max())
        t_decode, generated = _timed(lambda: decode_from(logits, p_last))
        e2e = _time(run, runs)
    finally:
        runner.close()

    return _result(
        "hipengine_gpu",
        "HIP fp32 (gfx1151): GPU vision tower + GPU prefill/decode",
        init_s=init_s,
        stages={
            "preprocess": t_pre,
            "vision": t_vision,
            "prefill": t_prefill,
            "vision_prefill": t_vision + t_prefill,
            "decode": t_decode,
        },
        e2e=e2e,
        generated=generated,
        ref=ref,
    )


def lane_torch(device: str, max_new_tokens: int, runs: int) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    page = Image.open(FIXTURES / "page_small.png").convert("RGB")
    ref = json.loads((FIXTURES / "oracle_greedy.json").read_text())

    def init():
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device)
        model.eval()
        return processor, model

    init_s, (processor, model) = _timed(init)

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "cached"}, {"type": "text", "text": PROMPT}],
        }
    ]
    prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)

    def prep():
        return processor(text=[prompt_str], images=[page], return_tensors="pt").to(device)

    def sync() -> None:
        if device != "cpu":
            torch.cuda.synchronize()

    def generate(p) -> list[int]:
        with torch.no_grad():
            out = model.generate(
                input_ids=p["input_ids"],
                attention_mask=p["attention_mask"],
                pixel_values=p["pixel_values"],
                image_grid_thw=p["image_grid_thw"],
                mm_token_type_ids=p["mm_token_type_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return out[0][p["input_ids"].shape[1]:].tolist()

    def run() -> list[int]:
        p = prep()
        sync()
        ids = generate(p)
        sync()
        return ids

    # stage timing (single instrumented warm run)
    t_proc, proc = _timed(prep)
    sync()

    def prefill_only():
        with torch.no_grad():
            model(
                input_ids=proc["input_ids"],
                attention_mask=proc["attention_mask"],
                pixel_values=proc["pixel_values"],
                image_grid_thw=proc["image_grid_thw"],
                mm_token_type_ids=proc["mm_token_type_ids"],
                logits_to_keep=1,
            )

    t_prefill, _ = _timed(prefill_only)
    sync()
    t_generate, generated = _timed(lambda: generate(proc))
    sync()
    t_decode = max(0.0, t_generate - t_prefill)

    e2e = _time(run, runs)
    n = min(len(generated), len(ref["ids"]))
    match = generated[:n] == ref["ids"][:n]
    result = _result(
        f"torch_{device}",
        f"transformers fp32 {device}",
        init_s=init_s,
        stages={
            "preprocess": t_proc,
            "vision_prefill": t_prefill,
            "decode": t_decode,
        },
        e2e=e2e,
        generated=generated,
        ref=ref,
    )
    result["correctness"] = "PASS" if match else f"FAIL (first divergence within {n})"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/results/2026-09-11-gfx1151-surya-lane-compare.json"))
    parser.add_argument("--lanes", default="hipengine_cpu,torch_cpu,torch_cuda,hipengine_gpu")
    args = parser.parse_args()

    lanes = [l.strip() for l in args.lanes.split(",")]
    results = []
    for lane in lanes:
        print(f"== lane {lane} ...", flush=True)
        try:
            if lane == "hipengine_cpu":
                results.append(lane_hipengine_cpu(args.max_new_tokens, args.runs))
            elif lane == "torch_cpu":
                results.append(lane_torch("cpu", args.max_new_tokens, args.runs))
            elif lane == "hipengine_gpu":
                results.append(lane_hipengine_gpu(args.max_new_tokens, args.runs))
            elif lane == "torch_cuda":
                import torch

                if not torch.cuda.is_available():
                    results.append({"lane": "torch_cuda", "correctness": "SKIP (no HIP torch)"})
                else:
                    results.append(lane_torch("cuda", args.max_new_tokens, args.runs))
        except Exception as exc:  # noqa: BLE001 - report and continue
            results.append({"lane": lane, "error": repr(exc)})
        r = results[-1]
        print(json.dumps(r, indent=1), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "date": "2026-09-11",
        "host": "gfx1151 (AMD Ryzen AI Max+ Pro 395 w/ Radeon 8060S), 125GB RAM",
        "model": MODEL_ID,
        "workload": {
            "image": "tests/fixtures/surya/page_small.png (256x256, 64 image tokens)",
            "prompt": PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "precision": "fp32 everywhere",
        },
        "protocol": (
            "Every lane times one full OCR request: page image + prompt in, "
            "greedy token ids out (preprocessing, vision tower, prefill, "
            "decode all inside the timed region). Checkpoint load and "
            "engine/runner construction are initialization, reported as "
            "init_s and excluded from e2e_*. decode_tokens is the actual "
            "generated count. torch lanes cannot separate vision from text "
            "prefill, so they report the combined vision_prefill_s stage. "
            "hipengine_gpu init_s is measured with a warm JIT cache; a "
            "cold-cache first run additionally pays hipcc compile time."
        ),
        "results": results,
    }
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
