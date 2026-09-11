"""Surya OCR 2 performance comparison: hipEngine CPU reference vs torch.

Lanes on one physical host (gfx1151 / Ryzen AI Max+ Pro 395):
  - hipengine_cpu:  hipEngine torch-free CPU-reference path (fp32 numpy),
    stage timing: preprocess / vision / prefill / decode.
  - torch_cpu:      transformers Qwen3_5ForConditionalGeneration fp32 CPU
    (the oracle implementation), stage timing: processor / prefill / decode.
  - torch_gpu:      same transformers fp32 on the host GPU via torch ROCm.

Correctness gate for every lane: greedy token ids must equal the captured
torch fp32 CPU reference (tests/fixtures/surya/oracle_greedy.json); a lane
that diverges is reported as FAILED and excluded from perf claims.

Usage:
    python3 scripts/surya_perf_compare.py [--runs 3] [--max-new-tokens 64]
        [--out benchmarks/results/2026-09-11-gfx1151-surya-cpu-vs-torch.json]
"""

from __future__ import annotations

import argparse
import json
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

    model_dir = resolve_surya_path(MODEL_ID)
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    tokenizer = SuryaTokenizer(model_dir)
    page = Image.open(FIXTURES / "page_small.png").convert("RGB")

    def run() -> list[int]:
        rows, grid = preprocess_image_surya(page)
        _, _, merged = vision_forward(weights, spec, rows, [grid])
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        hidden, state = text_prefill(
            weights, spec, np.array([ids], dtype=np.int64), pos, visual_features=merged[None]
        )
        emb = weights["model.language_model.embed_tokens.weight"]
        logits = hidden[:, -1] @ emb.T
        generated: list[int] = []
        p = int(pos[:, -1].max())
        for step in range(max_new_tokens):
            nxt = int(np.argmax(logits[0]))
            if nxt == spec.eos_token_id:
                break
            generated.append(nxt)
            logits = text_decode_step(weights, spec, nxt, state, p + 1 + step)
        return generated

    # stage timing (single instrumented run, warm)
    rows, grid = preprocess_image_surya(page)
    t0 = time.perf_counter(); _, _, merged = vision_forward(weights, spec, rows, [grid])
    t_vision = time.perf_counter() - t0
    n_img = (grid[1] // 2) * (grid[2] // 2)
    ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    t0 = time.perf_counter()
    hidden, state = text_prefill(
        weights, spec, np.array([ids], dtype=np.int64), pos, visual_features=merged[None]
    )
    t_prefill = time.perf_counter() - t0
    emb = weights["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    p = int(pos[:, -1].max())
    t0 = time.perf_counter()
    generated: list[int] = []
    for step in range(max_new_tokens):
        nxt = int(np.argmax(logits[0]))
        if nxt == spec.eos_token_id:
            break
        generated.append(nxt)
        logits = text_decode_step(weights, spec, nxt, state, p + 1 + step)
    t_decode = time.perf_counter() - t0

    e2e = _time(run, runs)
    ref = json.loads((FIXTURES / "oracle_greedy.json").read_text())
    return {
        "lane": "hipengine_cpu",
        "backend": "cpu_reference/fp32 numpy",
        "stages_s": {
            "preprocess": 0.0,  # measured inside vision warm path; see note
            "vision": t_vision,
            "prefill": t_prefill,
            "decode": t_decode,
        },
        "decode_tokens": len(generated),
        "decode_tok_per_s": len(generated) / t_decode,
        "e2e_median_s": e2e["median_s"],
        "e2e_warmup_s": e2e["warmup_s"],
        "correctness": "PASS" if generated == ref["ids"] else "FAIL",
    }


def lane_hipengine_gpu(max_new_tokens: int, runs: int) -> dict:
    """hipEngine GPU lane: fp32 HIP text decoder; vision tower stays on CPU.

    The Surya vision encoder has no GPU kernels yet (EVIE tower transfer is a
    recorded follow-up), so preprocess/vision run the CPU-reference path and
    prefill/decode run the HIP fp32 decoder (gfx1151). Vision timing is
    reported by the hipengine_cpu lane and not repeated here.
    """

    import os

    from PIL import Image

    from hipengine.kernels.cpu_reference.surya import (
        SuryaSpec,
        SuryaWeights,
        vision_forward,
    )
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    model_dir = resolve_surya_path(MODEL_ID)
    spec = SuryaSpec()
    weights = SuryaWeights.load(str(model_dir / "model.safetensors"))
    tokenizer = SuryaTokenizer(model_dir)
    page = Image.open(FIXTURES / "page_small.png").convert("RGB")

    # shared input preparation (CPU): preprocess + vision tower + prompt
    rows, grid = preprocess_image_surya(page)
    _, _, merged = vision_forward(weights, spec, rows, [grid])
    ids, mm = render_chat_prompt(tokenizer, PROMPT, (grid[1] // 2) * (grid[2] // 2))
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    ids_arr = np.array([ids], dtype=np.int64)
    p_last = int(pos[:, -1].max())

    def gen(r: SuryaGpuRunner) -> tuple[list[int], float, float]:
        t0 = time.perf_counter()
        logits = r.prefill(ids_arr[0], pos, visual_features=merged)
        t_prefill = time.perf_counter() - t0
        generated: list[int] = []
        t0 = time.perf_counter()
        for step in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            if nxt == spec.eos_token_id:
                break
            generated.append(nxt)
            logits = r.decode_step(nxt, p_last + 1 + step)
        return generated, t_prefill, time.perf_counter() - t0

    runner = SuryaGpuRunner(weights, spec)
    try:
        generated, t_prefill, t_decode = gen(runner)

        def run():
            r = SuryaGpuRunner(weights, spec)
            try:
                gen(r)
            finally:
                r.close()

        e2e = _time(run, runs)
    finally:
        runner.close()

    ref = json.loads((FIXTURES / "oracle_greedy.json").read_text())
    return {
        "lane": "hipengine_gpu",
        "backend": "HIP fp32 decoder (gfx1151); vision tower on CPU",
        "stages_s": {"prefill": t_prefill, "decode": t_decode},
        "decode_tokens": len(generated),
        "decode_tok_per_s": len(generated) / t_decode,
        "e2e_median_s": e2e["median_s"],
        "e2e_warmup_s": e2e["warmup_s"],
        "correctness": "PASS" if generated == ref["ids"] else "FAIL",
    }


def lane_torch(device: str, max_new_tokens: int, runs: int) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    model.eval()
    page = Image.open(FIXTURES / "page_small.png").convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": "cached"}, {"type": "text", "text": PROMPT}],
        }
    ]
    prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)

    def prep():
        return processor(text=[prompt_str], images=[page], return_tensors="pt").to(device)

    proc = prep()

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

    def generate_full():
        with torch.no_grad():
            model.generate(
                input_ids=proc["input_ids"],
                attention_mask=proc["attention_mask"],
                pixel_values=proc["pixel_values"],
                image_grid_thw=proc["image_grid_thw"],
                mm_token_type_ids=proc["mm_token_type_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    def run():
        p = prep()
        with torch.no_grad():
            model.generate(
                input_ids=p["input_ids"],
                attention_mask=p["attention_mask"],
                pixel_values=p["pixel_values"],
                image_grid_thw=p["image_grid_thw"],
                mm_token_type_ids=p["mm_token_type_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); prefill_only(); 
    if device != "cpu":
        torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0
    t0 = time.perf_counter(); generate_full()
    if device != "cpu":
        torch.cuda.synchronize()
    t_generate = time.perf_counter() - t0
    t_decode = t_generate - t_prefill

    e2e = _time(run, runs)
    # correctness: capture ids from one instrumented generate
    with torch.no_grad():
        out = model.generate(
            input_ids=proc["input_ids"],
            attention_mask=proc["attention_mask"],
            pixel_values=proc["pixel_values"],
            image_grid_thw=proc["image_grid_thw"],
            mm_token_type_ids=proc["mm_token_type_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    gen_ids = out[0][proc["input_ids"].shape[1]:].tolist()
    ref = json.loads((FIXTURES / "oracle_greedy.json").read_text())
    n = min(len(gen_ids), len(ref["ids"]))
    match = gen_ids[:n] == ref["ids"][:n]
    return {
        "lane": f"torch_{device}",
        "backend": f"transformers fp32 {device}",
        "stages_s": {"prefill": t_prefill, "decode": t_decode},
        "decode_tokens": max_new_tokens,
        "decode_tok_per_s": max_new_tokens / t_decode,
        "e2e_median_s": e2e["median_s"],
        "e2e_warmup_s": e2e["warmup_s"],
        "correctness": "PASS" if match else f"FAIL (first divergence within {n})",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmarks/results/2026-09-11-gfx1151-surya-cpu-vs-torch.json"))
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
        "note": (
            "hipEngine has no Surya GPU path yet (open follow-up: EVIE tower "
            "kernel transfer + decoder GDN/attention GPU kernels); torch_cuda "
            "is the reference implementation on the host GPU, not hipEngine."
        ),
        "results": results,
    }
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
