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
model = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
base = ["python3", "scripts/qwen35_gguf_bench.py", "--model", model,
        "--persistent-session", "--force-bulk-prefill", "--public-ar-profile",
        "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt"]
tasks = [
    ("hip-p512-clean", base + ["--prompt-length", "512", "--decode-tokens", "128",
                              "--graph-replay-decode", "--json", "/tmp/nasone-hip-p512-clean.json"]),
    ("hip-native-retry", ["python3", "scripts/qwen36_dense_gguf_suite.py", "--model", model,
                         "--max-new-tokens", "25", "--candidate-budgets", "3",
                         "--target-verify-mode", "native", "--runs", "1", "--limit", "10",
                         "--warmup", "--compiler-version-file", "/tmp/hipengine-hipcc-version.txt",
                         "--output", "/tmp/nasone-hip-native-retry.json"]),
    ("hip-ar-only", ["python3", "/tmp/nasone_hip_ar.py"]),
    ("hip-bf16-112k", base + ["--prompt-length", "114688", "--decode-tokens", "8",
                             "--warmup-runs", "0", "--measured-runs", "1",
                             "--json", "/tmp/nasone-hip-bf16-112k.json"]),
    ("hip-bf16-96k", base + ["--prompt-length", "98304", "--decode-tokens", "8",
                            "--warmup-runs", "0", "--measured-runs", "1",
                            "--json", "/tmp/nasone-hip-bf16-96k.json"]),
    ("hip-bf16-80k", base + ["--prompt-length", "81920", "--decode-tokens", "8",
                            "--warmup-runs", "0", "--measured-runs", "1",
                            "--json", "/tmp/nasone-hip-bf16-80k.json"]),
    ("hip-int8-128k", base + ["--prompt-length", "131072", "--decode-tokens", "8",
                             "--kv-storage", "int8_per_token_head", "--kv-scale-dtype", "fp32",
                             "--warmup-runs", "0", "--measured-runs", "1",
                             "--json", "/tmp/nasone-hip-int8-128k.json"]),
    ("hip-int8-112k", base + ["--prompt-length", "114688", "--decode-tokens", "8",
                             "--kv-storage", "int8_per_token_head", "--kv-scale-dtype", "fp32",
                             "--warmup-runs", "0", "--measured-runs", "1",
                             "--json", "/tmp/nasone-hip-int8-112k.json"]),
]
card = select_card(pci_id="0000:10:00.0")
passed_capacity_families = set()
for name, command in tasks:
    family = "bf16" if name.startswith("hip-bf16-") else "int8" if name.startswith("hip-int8-") else None
    if family in passed_capacity_families:
        print("SKIP lower capacity point", name, flush=True)
        continue
    if family:
        command += ["--warmup-decode-tokens", "0"]
    require_idle_memory(int(card.vram_used_path.read_text()), 128)
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    sampler = VramSampler(card=card, interval_ms=20)
    sampler.start()
    with open(f"/tmp/nasone-{name}.log", "w") as log:
        result = subprocess.run(command, cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    sampler.stop()
    if result.returncode == 0 and family:
        passed_capacity_families.add(family)
    Path(f"/tmp/nasone-{name}-monitor.json").write_text(json.dumps(
        dict(command=command, environment={k:v for k,v in env.items() if k.startswith(("HIP", "ROCR", "GPU_MAX"))},
             returncode=result.returncode, memory=sampler.result().to_dict(),
             source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()),
        indent=2) + "\n")
    print("END", name, result.returncode, time.strftime("%H:%M:%S"), flush=True)
