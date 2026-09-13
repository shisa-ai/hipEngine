import os
from pathlib import Path
import subprocess
import time

root = Path("/home/lhl/hipEngine-main")
env = dict(os.environ, HIP_VISIBLE_DEVICES="1", HIPENGINE_HIP_ARCH="gfx1100")
env.pop("ROCR_VISIBLE_DEVICES", None)
model = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
base = [
    "python3", "scripts/llamacpp_raw_suite_bench.py",
    "--server", "/tmp/llama-rdna3-nasone32/build/bin/llama-server",
    "--source", "/tmp/llama-rdna3-nasone32", "--model", model,
]
tasks = [
    ("bf16-128k", base + ["--repeat-lengths", "131072", "--outputs", "9",
                        "--context", "131328", "--output", "/tmp/nasone-bf16-128k.json"], {}),
    ("bf16-112k", base + ["--repeat-lengths", "114688", "--outputs", "9",
                        "--context", "114944", "--output", "/tmp/nasone-bf16-112k.json"], {}),
    ("q8-224k", base + ["--repeat-lengths", "229376", "--outputs", "9", "--kv", "q8_0",
                       "--context", "229632", "--output", "/tmp/nasone-q8-224k.json"], {}),
    ("q8-192k", base + ["--repeat-lengths", "196608", "--outputs", "9", "--kv", "q8_0",
                       "--context", "196864", "--output", "/tmp/nasone-q8-192k.json"], {}),
    ("gdn-fp32", base + ["--repeat-lengths", "8192", "--outputs", "129",
                        "--context", "8704", "--repetitions", "3",
                        "--output", "/tmp/nasone-gdn-fp32.json"], {"GGML_CUDA_GDN_CHUNKED_BF16": "0"}),
]
for name, command, override in tasks:
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    with open(f"/tmp/nasone-{name}-driver.log", "w") as log:
        result = subprocess.run(command, cwd=root, env=env | override,
                                stdout=log, stderr=subprocess.STDOUT)
    print("END", name, result.returncode, time.strftime("%H:%M:%S"), flush=True)
