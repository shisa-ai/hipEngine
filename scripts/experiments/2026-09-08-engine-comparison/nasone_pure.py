import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/home/lhl/hipEngine-main")
sys.path.insert(0, str(root))
from hipengine.util.amdgpu_vram import VramSampler, select_card
from scripts.llamacpp_raw_suite_bench import require_idle_memory

env = dict(os.environ, HIP_VISIBLE_DEVICES="1", HIPENGINE_HIP_ARCH="gfx1100",
           HIPENGINE_GGUF_DECODE_REPACK="1",
           HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG="1",
           HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS="none")
env.pop("ROCR_VISIBLE_DEVICES", None)
card = select_card(pci_id="0000:10:00.0")


def run(length):
    name = f"hip-pure-int8-{length}"
    command = ["python3", "scripts/qwen35_gguf_bench.py",
               "--model", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
               "--persistent-session", "--force-bulk-prefill", "--public-ar-profile",
               "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt",
               "--prompt-length", str(length), "--decode-tokens", "8",
               "--kv-storage", "int8_per_token_head", "--kv-scale-dtype", "fp32",
               "--warmup-runs", "0", "--measured-runs", "1", "--warmup-decode-tokens", "0",
               "--json", f"/tmp/nasone-{name}.json"]
    for _ in range(30):
        if int(card.vram_used_path.read_text()) < 128 * (1 << 20):
            break
        time.sleep(1)
    require_idle_memory(int(card.vram_used_path.read_text()), 128)
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    sampler = VramSampler(card=card, interval_ms=20)
    sampler.start()
    with open(f"/tmp/nasone-{name}.log", "w") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        if int(card.vram_used_path.read_text()) < 128 * (1 << 20):
            break
        time.sleep(1)
    sampler.stop()
    Path(f"/tmp/nasone-{name}-monitor.json").write_text(json.dumps(dict(
        command=command, environment={k:v for k,v in env.items() if k.startswith(("HIP", "ROCR", "GPU_MAX"))},
        returncode=result.returncode, memory=sampler.result().to_dict(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    ), indent=2) + "\n")
    print("END", name, result.returncode, time.strftime("%H:%M:%S"), flush=True)
    return result.returncode


if __name__ == "__main__":
    if run(129024) == 0:
        run(131072)
    else:
        run(114688)
