"""Smoke: GPU Surya text decoder parity vs CPU reference (fixture workload)."""

import sys

import numpy as np

from PIL import Image

from hipengine.kernels.cpu_reference.surya import (
    SuryaSpec,
    SuryaWeights,
    text_decode_step,
    text_prefill,
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

PROMPT = "Transcribe this page."
MAX_NEW = 16


def build_inputs():
    model_dir = resolve_surya_path("datalab-to/surya-ocr-2")
    spec = SuryaSpec()
    weights = SuryaWeights.load(str(model_dir / "model.safetensors"))
    page = Image.open("tests/fixtures/surya/page_small.png").convert("RGB")
    rows, grid = preprocess_image_surya(page)
    _, _, merged = vision_forward(weights, spec, rows, [grid])
    n_img = (grid[1] // 2) * (grid[2] // 2)
    tokenizer = SuryaTokenizer(model_dir)
    ids, mm = render_chat_prompt(tokenizer, PROMPT, n_img)
    pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
    return weights, spec, np.array([ids], dtype=np.int64), pos, merged[None]


def main() -> int:
    weights, spec, ids, pos, visual = build_inputs()
    # CPU reference
    hidden, state = text_prefill(weights, spec, ids, pos, visual_features=visual)
    emb = weights["model.language_model.embed_tokens.weight"]
    ref_prefill_logits = hidden[:, -1] @ emb.T
    ref_logits = ref_prefill_logits
    ref_step_logits: list[np.ndarray] = []
    ref_tokens = []
    p = int(pos[:, -1].max())
    for step in range(MAX_NEW):
        nxt = int(np.argmax(ref_logits[0]))
        ref_tokens.append(nxt)
        ref_logits = text_decode_step(weights, spec, nxt, state, p + 1 + step)
        ref_step_logits.append(ref_logits)

    runner = SuryaGpuRunner(weights, spec)
    try:
        gpu_logits = runner.prefill(ids[0], pos, visual_features=visual[0])
        d = np.abs(gpu_logits - ref_prefill_logits[0])
        print(f"prefill logits: max|diff|={d.max():.3e} mean={d.mean():.3e} "
              f"logit-mag={np.abs(ref_logits[0]).max():.1f}")
        t5g = np.argsort(gpu_logits)[-5:][::-1]
        t5r = np.argsort(ref_prefill_logits[0])[-5:][::-1]
        print(f"top5 gpu={t5g.tolist()} vals={[round(float(v),2) for v in gpu_logits[t5g]]}")
        print(f"top5 ref={t5r.tolist()} vals={[round(float(v),2) for v in ref_prefill_logits[0][t5r]]}")
        gpu_tokens = []
        ok = True
        for step in range(MAX_NEW):
            nxt = int(np.argmax(gpu_logits))
            gpu_tokens.append(nxt)
            gpu_logits = runner.decode_step(nxt, p + 1 + step)
            dl = np.abs(gpu_logits - ref_step_logits[step][0])
            if step < 3 or gpu_tokens[-1] != ref_tokens[step]:
                print(f"step {step}: tok gpu={gpu_tokens[-1]} ref={ref_tokens[step]} "
                      f"logits max|diff|={dl.max():.3e}")
            ok &= gpu_tokens[-1] == ref_tokens[step]
        print("greedy tokens equal:", ok)
        print("gpu tokens:", gpu_tokens[:10])
        print("ref tokens:", ref_tokens[:10])
        return 0 if ok else 1
    finally:
        runner.close()


if __name__ == "__main__":
    sys.exit(main())
