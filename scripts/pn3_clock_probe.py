#!/usr/bin/env python3
"""Probe: does the sclk plateau and stay stable at the 60W power equilibrium
under sustained 35B prefill load (with perf level forced high)?
"""
import sys, time, ctypes, subprocess, threading
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
from hipengine.runtime.prefill import PrefillConfig
ctypes.CDLL('libamdhip64.so')

CARD = '/sys/class/drm/card1/device'
stop = threading.Event()

def force_high():
    subprocess.run(['sudo','sh','-c',f'echo high > {CARD}/power_dpm_force_performance_level'],
                   check=True, capture_output=True, text=True, timeout=20)

def sclk_line():
    try:
        with open(f'{CARD}/pp_dpm_sclk') as f:
            for l in f.read().splitlines():
                if '*' in l: return l.strip()
    except Exception: return '?'

def busy():
    try:
        with open(f'{CARD}/gpu_busy_percent') as f: return f.read().strip()+'%'
    except Exception: return '?'

def monitor(tag, out):
    while not stop.is_set():
        out.append((time.time(), sclk_line(), busy()))
        time.sleep(1.5)

M35 = '/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'
force_high()
samples=[]
t=threading.Thread(target=monitor, args=('35B', samples)); t.start()
for i in range(6):
    cfg = PrefillConfig(linear_chunk_size=1024, moe_chunk_size=1024)
    with rm.Qwen35GGUFResidentSession(M35, max_sequence_length=2048,
            backend='hip_gfx1151', prefill_config=cfg) as s:
        toks=[9707]*2048
        s.prefill(toks, use_bulk=True, return_logits=False); s.runtime.device_synchronize()
        t0=time.perf_counter()
        s.prefill(toks, use_bulk=True, return_logits=False); s.runtime.device_synchronize()
        ms=(time.perf_counter()-t0)*1e3
    force_high()
    print(f"prefill{i}: {ms:.0f}ms  sclk={sclk_line()}  busy={busy()}", flush=True)
stop.set(); t.join()
print("--- trajectory ---")
for ts, sc, bu in samples:
    print(f"{time.strftime('%M:%S', time.localtime(ts))}  {sc}  busy={bu}")
print("DONE")
