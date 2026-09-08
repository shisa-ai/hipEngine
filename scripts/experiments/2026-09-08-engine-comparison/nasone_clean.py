import os
import subprocess
import time

base = [
    "python3", "scripts/llamacpp_raw_suite_bench.py",
    "--server", "/tmp/llama-rdna3-nasone32/build/bin/llama-server",
    "--source", "/tmp/llama-rdna3-nasone32",
    "--model", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
    "--repetitions", "3",
]
env = dict(os.environ, HIP_VISIBLE_DEVICES="1")
env.pop("ROCR_VISIBLE_DEVICES", None)
tasks = [
    ("shapes-clean", ["--repeat-lengths", "512,8192", "--outputs", "129", "--context", "8704"], {}),
    ("q8-shapes", ["--repeat-lengths", "8192", "--outputs", "129", "--context", "8704", "--kv", "q8_0"], {}),
    ("gdn-sequential-shape", ["--repeat-lengths", "8192", "--outputs", "129", "--context", "8704"],
     {"GGML_CUDA_GDN_CHUNKED": "0"}),
    ("ar-clean", ["--mode", "ar"], {}),
    ("mtp-clean", ["--mode", "mtp"], {}),
    ("adaptive-clean", ["--mode", "adaptive"], {}),
    ("ar-sequential", ["--mode", "ar"], {"GGML_CUDA_GDN_CHUNKED": "0"}),
    ("mtp-sequential", ["--mode", "mtp"], {"GGML_CUDA_GDN_CHUNKED": "0"}),
    ("ar-long", ["--mode", "ar", "--outputs", "129", "--repetitions", "1"], {}),
    ("mtp-long", ["--mode", "mtp", "--outputs", "129", "--repetitions", "1"], {}),
    ("adaptive-long", ["--mode", "adaptive", "--outputs", "129", "--repetitions", "1"], {}),
]
for name, arguments, override in tasks:
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    command = base + arguments + ["--output", f"/tmp/nasone-{name}.json"]
    with open(f"/tmp/nasone-{name}-driver.log", "w") as log:
        result = subprocess.run(command, cwd="/home/lhl/hipEngine-main", env=env | override,
                                stdout=log, stderr=subprocess.STDOUT)
    print("END", name, result.returncode, time.strftime("%H:%M:%S"), flush=True)
