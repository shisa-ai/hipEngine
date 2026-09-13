import json
from pathlib import Path
import subprocess
import sys

root = Path("/home/lhl/hipEngine-main")
sys.path.insert(0, str(root))
from scripts.qwen36_dense_gguf_suite import (
    Qwen35GGUFResidentSession, Qwen35GGUFTokenizer, load_gguf_index,
    load_prompt_rows, build_chat_prompt, _run_ar,
)

model = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
tokenizer = Qwen35GGUFTokenizer.from_gguf_info(load_gguf_index(model))
prompts = load_prompt_rows(root / "benchmarks/prompts/mtpbench-code-general-ja.jsonl")
rows = []
with Qwen35GGUFResidentSession(
    model, max_sequence_length=1024,
    compiler_version=Path("/tmp/hipengine-hipcc-version.txt").read_text(),
    use_wmma_prefill=True, use_gemv_decode=True,
) as target:
    target.select_prefill_quant("gguf_q4_k_m")
    _run_ar(target, build_chat_prompt(tokenizer, prompts[0]["prompt"], reasoning="off"),
            max_new_tokens=8)
    for repetition in range(3):
        for prompt in prompts:
            tokens = list(build_chat_prompt(tokenizer, prompt["prompt"], reasoning="off"))
            row = _run_ar(target, tokens, max_new_tokens=25)
            row.update(id=prompt["id"], category=prompt["category"], repetition=repetition,
                       prompt_ids=tokens)
            rows.append(row)
            print(prompt["id"], row["decode_tok_s_transition_normalized"], flush=True)
Path("/tmp/nasone-hip-ar-only.json").write_text(json.dumps(dict(
    schema=1, status="complete", performance_claim=False, model=str(model),
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    harness_source=Path(__file__).read_text(), rows=rows,
), indent=2) + "\n")
