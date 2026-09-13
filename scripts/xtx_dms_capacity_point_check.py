import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/lhl/gfx1100-24gb-capacity")
sys.path.insert(0, str(ROOT))
from hipengine.util.amdgpu_vram import VramSampler, select_card

tokens = int(sys.argv[1])
assert tokens > 0 and tokens % 256 == 0
base = Path(f"/tmp/xtx-capacity/dms-int8-target4lo256k-{tokens}")
output = base.with_suffix(".json")
monitor = base.with_suffix(".monitor.json")
log = base.with_suffix(".log")
assert not any(p.exists() for p in (output, monitor, log)), "preserve prior attempts"
card = select_card(pci_id="0000:10:00.0")
unique_id = (card.sysfs_path / "unique_id").read_text().strip()
assert unique_id.removeprefix("0x") == "cc4d02090dc9c3ff"
baseline = int(card.vram_used_path.read_text())
assert baseline < 128 * 2**20, f"GPU1 not idle: {baseline} bytes"
command = [
    str(ROOT / ".venv/bin/python"), "scripts/qwen38_dms_concurrency_probe.py",
    "--model", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf",
    "--metadata", "/models/dms/qwen38-27b-q4km-dms-w8192-local/dms_metadata.json",
    "--data-manifest", "/tmp/xtx-capacity/qwen38-27b-q4km-xtx-capacity-manifest-256k.json",
    "--sessions", "1", "--prompt-tokens", str(tokens), "--decode-steps", "8", "--dms-prefill-mode", "layer_outer",
    "--refill-cycles", "1", "--codec", "int8_evaluation",
    "--backend", "hip_gfx1100", "--output", str(output),
]
env = dict(os.environ, HIP_VISIBLE_DEVICES="1", GPU_MAX_HW_QUEUES="1",
           OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
started = time.monotonic()
timed_out = False
sampler = VramSampler(card, interval_ms=20, keep_samples=True)
with log.open("w") as stream, sampler:
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = process.wait(timeout=2700)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
    for _ in range(50):
        if int(card.vram_used_path.read_text()) <= baseline + 16 * 2**20:
            break
        time.sleep(0.1)
text = log.read_text()
raw = json.loads(output.read_text()) if output.exists() else None
oom = bool(re.search(r"\bHIP error 2\b|hipErrorOutOfMemory|out of memory", text, re.I))
status = "timeout" if timed_out else "oom" if code and oom else "failed"
if code == 0 and raw is not None:
    cycle = raw["cycles"][0]
    assert raw["status"] == "passed" and not raw["cycle_errors"]
    assert len(cycle["decode"]) == 8
    assert all(r["finite_logits"] for r in cycle["decode"])
    assert cycle["memory"]["after_close"]["active_allocations"] == 0
    assert cycle["memory"]["after_close"]["current_allocated_bytes"] == 0
    status = "execution_fit"
samples = sampler.result()
compact_samples = samples.to_dict()
compact_samples["max_sample_gap_seconds"] = max(
    b[0] - a[0] for a, b in zip(samples.samples, samples.samples[1:])
)
report = {
    "status": status, "exit_code": code, "timeout_seconds": 2700,
    "prompt_tokens": tokens, "decode_steps": 8,
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "working_tree_clean": not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
    "command": "HIP_VISIBLE_DEVICES=1 GPU_MAX_HW_QUEUES=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 " + shlex.join(command),
    "wrapper_command": f".venv/bin/python /tmp/xtx-capacity-baseline/run_target4lo_bisect.py {tokens}",
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "unique_id": unique_id, "cwd": str(ROOT),
    "whole_card_vram": compact_samples,
    "sampled_headroom_bytes": card.vram_total_bytes - samples.peak_bytes,
    "returns_to_baseline": samples.final_bytes <= baseline + 16 * 2**20,
    "raw_output": str(output) if raw else None,
    "raw_sha256": hashlib.sha256(output.read_bytes()).hexdigest() if raw else None,
    "log": str(log), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "failure_tail": text[-5000:] if code else None,
    "elapsed_seconds": time.monotonic() - started,
    "tracked_peak_bytes": raw["cycles"][0]["memory"]["after_close"]["peak_allocated_bytes"] if raw else None,
    "scope": "capacity/finiteness/ownership only; no new numerical or throughput qualification",
}
monitor.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2), flush=True)
if status not in {"execution_fit", "oom"} or not report["returns_to_baseline"]:
    raise SystemExit(1)
