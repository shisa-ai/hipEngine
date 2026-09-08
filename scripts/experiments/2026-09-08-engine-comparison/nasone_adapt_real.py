import os
import subprocess
import time

base = ["python3", "scripts/llamacpp_raw_suite_bench.py",
        "--server", "/tmp/llama-rdna3-nasone32/build/bin/llama-server",
        "--source", "/tmp/llama-rdna3-nasone32",
        "--model", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
        "--mode", "adaptive", "--adaptive-min", "1"]
env = dict(os.environ, HIP_VISIBLE_DEVICES="1")
env.pop("ROCR_VISIBLE_DEVICES", None)
for name, outputs, repetitions in (
    ("adaptive-floor1", 25, 3), ("adaptive-floor1-long", 129, 1)
):
    command = base + ["--outputs", str(outputs), "--repetitions", str(repetitions),
                      "--output", f"/tmp/nasone-{name}.json"]
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    with open(f"/tmp/nasone-{name}-driver.log", "w") as log:
        result = subprocess.run(command, cwd="/home/lhl/hipEngine-main", env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    print("END", name, result.returncode, time.strftime("%H:%M:%S"), flush=True)
