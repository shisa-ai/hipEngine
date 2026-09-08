import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path("/tmp/hipengine-engine-compare-6c01f1f1c")
sys.path.insert(0, str(ROOT))
from hipengine.util.amdgpu_vram import VramSampler, select_card

RAW = Path("/tmp/engine-compare-final")
RAW.mkdir(exist_ok=True)
CARD = select_card(pci_id="0000:10:00.0")
ENV = dict(os.environ, HIP_VISIBLE_DEVICES="1", HIPENGINE_HIP_ARCH="gfx1100",
           HIPENGINE_GGUF_DECODE_REPACK="1", PYTHONUNBUFFERED="1")
ENV.pop("ROCR_VISIBLE_DEVICES", None)


def wait_idle():
    for _ in range(120):
        if int(CARD.vram_used_path.read_text()) < 128 * (1 << 20):
            time.sleep(0.5)
            if int(CARD.vram_used_path.read_text()) < 128 * (1 << 20):
                return
        time.sleep(0.5)
    raise RuntimeError("GPU1 did not return to idle; no other process was terminated")


def owners():
    rows = []
    for path in Path("/sys/class/kfd/kfd/proc").glob("*/vram_33912"):
        try:
            size = int(path.read_text())
            pid = int(path.parent.name)
            if size > 1 << 20:
                rows.append(dict(pid=pid, pgid=os.getpgid(pid), vram_bytes=size))
        except (OSError, ValueError, ProcessLookupError):
            pass
    return rows


def run_case(name, command, extra_env=None):
    wait_idle()
    monitor_path = RAW / f"{name}-monitor.json"
    if monitor_path.exists():
        raise FileExistsError(f"preserve prior attempt: {monitor_path}")
    env = ENV | (extra_env or {})
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("benchmark source must be clean")
    sampler = VramSampler(card=CARD, interval_ms=20)
    records = []
    previous = None
    foreign = []
    started = time.time()
    print("START", name, time.strftime("%H:%M:%S"), flush=True)
    sampler.start()
    try:
        with (RAW / f"{name}-driver.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            active = owners()
            key = tuple((r["pid"], r["pgid"]) for r in active)
            if key != previous:
                records.append(dict(timestamp=time.time(), owners=active))
                previous = key
            foreign = [r for r in active if r["pgid"] != process.pid]
            if foreign:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.2)
        code = process.wait()
        wait_idle()
    finally:
        sampler.stop()
    payload = dict(
        command=command, environment={k:v for k,v in env.items() if k.startswith(("HIP", "ROCR", "GGML", "GPU_MAX", "HSA"))},
        source_commit=source, source_clean=True, started_unix=started, ended_unix=time.time(),
        returncode=code, foreign_gpu1_owners=foreign, ownership_transitions=records,
        memory=sampler.result().to_dict(),
    )
    monitor_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("END", name, code, time.strftime("%H:%M:%S"), flush=True)
    if foreign:
        raise RuntimeError(f"GPU1 interference detected; attempt invalid: {name}")
    return code
