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
           HIPENGINE_GGUF_DECODE_REPACK="1")
env.pop("ROCR_VISIBLE_DEVICES", None)
base = ["python3", "scripts/qwen35_gguf_bench.py",
        "--model", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        "--persistent-session", "--force-bulk-prefill", "--public-ar-profile",
        "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt"]
tasks = [
    ("hip-int8-112k", base + ["--prompt-length", "114688", "--decode-tokens", "8",
                             "--kv-storage", "int8_per_token_head", "--kv-scale-dtype", "fp32",
                             "--warmup-runs", "0", "--measured-runs", "1",
                             "--warmup-decode-tokens", "0",
                             "--json", "/tmp/nasone-hip-int8-112k.json"]),
    ("hip-p8192-clean", base + ["--prompt-length", "8192", "--decode-tokens", "128",
                               "--graph-replay-decode", "--json", "/tmp/nasone-hip-p8192-clean.json"]),
    ("hip-bf16-128k", base + ["--prompt-length", "131072", "--decode-tokens", "8",
                             "--warmup-runs", "0", "--measured-runs", "1",
                             "--warmup-decode-tokens", "0",
                             "--json", "/tmp/nasone-hip-bf16-128k.json"]),
]
card = select_card(pci_id="0000:10:00.0")


def wait_idle():
    for _ in range(30):
        used = int(card.vram_used_path.read_text())
        if used < 128 * (1 << 20):
            time.sleep(1)
            require_idle_memory(int(card.vram_used_path.read_text()), 128)
            return
        time.sleep(1)
    require_idle_memory(int(card.vram_used_path.read_text()), 128)


for name, command in tasks:
    wait_idle()
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    sampler = VramSampler(card=card, interval_ms=20)
    sampler.start()
    with open(f"/tmp/nasone-{name}.log", "w") as log:
        result = subprocess.run(command, cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    wait_idle()
    sampler.stop()
    Path(f"/tmp/nasone-{name}-monitor.json").write_text(json.dumps(
        dict(command=command, environment={k:v for k,v in env.items() if k.startswith(("HIP", "ROCR", "GPU_MAX"))},
             returncode=result.returncode, memory=sampler.result().to_dict(),
             source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()),
        indent=2) + "\n")
    print("END", name, result.returncode, time.strftime("%H:%M:%S"), flush=True)
wait_idle()
raise SystemExit(subprocess.run(["python3", "/tmp/nasone_clean.py"], env=env).returncode)
