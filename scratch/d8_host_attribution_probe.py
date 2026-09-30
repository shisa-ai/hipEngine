#!/usr/bin/env python3
"""D8 step 1: where does decode's non-busy time go? Host-side attribution.

The d7 census says device busy is 14.21 ms/token; the d7 topline says wall is
15.92 ms/token (62.83 tok/s). The row demands host launch and sync gaps be
measured directly before any gain is assigned. This probe measures, in ONE
process and therefore one instrument:

  * host time inside HIP calls (every ``ctypes.CDLL.__getattr__``-resolved
    ``hip*`` symbol is wrapped once and timed per call -- kernel launches,
    memcpys/memsets, syncs alike), with call counts;
  * wall time of the engine's scheduler tick and the runner's decode_batch;
  * per-sample wall, differenced across sample counts so the decode-step cost
    separates from prefill (two runs, N and N+128 tokens).

Bias note: the wrapper adds ~100-200 ns per call to the number it measures,
and the prefill of the larger run is identical to the smaller run, so the
differencing cancels prefill host cost except for launch-count differences.

Run: ROCR_VISIBLE_DEVICES=1 .venv/bin/python scratch/d8_host_attribution_probe.py
(The census busy and topline wall it compares against are the committed d7
artifacts; this probe only supplies the host side.)
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# 1. Wrap every hip* symbol resolution so all HIP calls are timed on the host.
# ---------------------------------------------------------------------------
_STATS = {
    "hip_calls": 0,
    "hip_host_ns": 0,
    "launch_calls": 0,
    "launch_host_ns": 0,
    "memcpy_calls": 0,
    "memcpy_host_ns": 0,
    "sync_calls": 0,
    "sync_host_ns": 0,
}
_SYMBOL_STATS: dict = {}
_MEMCPY_SITES: dict = {}

import os as _os
import traceback as _traceback
_wrapped: dict = {}
_orig_getattr = ctypes.CDLL.__getattr__

_COUNTER_KEY = "_he_probe_tagged"


def _bucket(name: str) -> str:
    # The kernel-library exports are all hipengine_* (the C symbols the
    # wrappers bind); HIP runtime symbols are hip*(Launch|Memcpy|Stream...).
    if name.startswith("hipengine_") or "Launch" in name or "kernel" in name.lower():
        return "launch"
    if "Memcpy" in name or "Memset" in name or "Free" in name or "Malloc" in name:
        return "memcpy"
    return "sync"


class _TimedCall:
    """Callable proxy: times calls, delegates argtypes/restype to the fn.

    ``signed_kernel_fn`` sets ``fn.argtypes`` on whatever this returns, so the
    delegation must be a property on the object -- a plain function would
    swallow the assignment and leave the underlying _FuncPtr untyped (which is
    exactly how this probe first broke: c_float conversion refused).
    """

    __slots__ = ("_fn", "_tag", "_he_signed")

    def __init__(self, fn, tag: str):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_tag", tag)

    @property
    def argtypes(self):
        return object.__getattribute__(self, "_fn").argtypes

    @argtypes.setter
    def argtypes(self, value):
        object.__getattribute__(self, "_fn").argtypes = value

    @property
    def restype(self):
        return object.__getattribute__(self, "_fn").restype

    @restype.setter
    def restype(self, value):
        object.__getattribute__(self, "_fn").restype = value

    def __call__(self, *args):
        fn = object.__getattribute__(self, "_fn")
        tag = object.__getattribute__(self, "_tag")
        name = getattr(fn, "__name__", None) or "?"
        t0 = time.perf_counter_ns()
        err = fn(*args)
        dt = time.perf_counter_ns() - t0
        _STATS["hip_calls"] += 1
        _STATS["hip_host_ns"] += dt
        _STATS[f"{tag}_calls"] += 1
        _STATS[f"{tag}_host_ns"] += dt
        if tag != "launch":
            row = _SYMBOL_STATS.setdefault(name, [0, 0])
            row[0] += 1
            row[1] += dt
            if name == "hipMemcpy":
                # Record the first non-infra caller; timed window already closed.
                site = "?"
                for fr in reversed(_traceback.extract_stack()[:-1]):
                    p = fr.filename
                    if "/scratch/" in p or "/core/hip.py" in p or "/core/memory.py" in p:
                        continue
                    site = f"{_os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"
                    break
                srow = _MEMCPY_SITES.setdefault(site, [0, 0])
                srow[0] += 1
                srow[1] += dt
        return err


def _probe_getattr(self, name: str):
    fn = _orig_getattr(self, name)
    if not name.startswith("hip"):
        return fn
    # One wrapper per (library, symbol): signed_kernel_fn's sentinel check
    # runs on this object, so identity must be stable across resolves.
    key = (id(self), name)
    wrapped = _wrapped.get(key)
    if wrapped is None:
        wrapped = _TimedCall(fn, _bucket(name))
        try:
            setattr(wrapped, _COUNTER_KEY, True)
        except Exception:
            pass
        _wrapped[key] = wrapped
        # ctypes stashes the resolved symbol in the instance __dict__, and a
        # dict hit never reaches __getattr__ again -- install the wrapper
        # there so every later resolve-and-call goes through the timer.
        try:
            self.__dict__[name] = wrapped
        except Exception:
            pass
    return wrapped


ctypes.CDLL.__getattr__ = _probe_getattr

# ---------------------------------------------------------------------------
# 2. Import the engine, then wrap the scheduler boundaries.
# ---------------------------------------------------------------------------
import hipengine  # noqa: E402
from hipengine.generation import engine_loop as _el  # noqa: E402

_ENGINE_STATS = {"tick_wall_ns": 0, "ticks": 0, "decode_batch_wall_ns": 0,
                 "decode_batches": 0, "decode_tokens": 0}

_orig_tick = _el.ResidentEngineLoop._tick_once


def _timed_tick(self):
    t0 = time.perf_counter_ns()
    out = _orig_tick(self)
    _ENGINE_STATS["tick_wall_ns"] += time.perf_counter_ns() - t0
    _ENGINE_STATS["ticks"] += 1
    return out


_el.ResidentEngineLoop._tick_once = _timed_tick

# decode_batch lives on the runner classes; wrap every class that defines it.
_wrapped_decode = []
for _cls_name, _cls in vars(_el).items():
    if isinstance(_cls, type) and hasattr(_cls, "decode_batch") and hasattr(_cls, "_probed"):
        pass
    if isinstance(_cls, type) and "decode_batch" in getattr(_cls, "__dict__", {}):
        _orig_db = _cls.__dict__["decode_batch"]

        def _make(orig):
            def _timed_db(self, work, *a, **kw):
                t0 = time.perf_counter_ns()
                out = orig(self, work, *a, **kw)
                _ENGINE_STATS["decode_batch_wall_ns"] += time.perf_counter_ns() - t0
                _ENGINE_STATS["decode_batches"] += 1
                try:
                    _ENGINE_STATS["decode_tokens"] += len(out)
                except Exception:
                    pass
                return out

            return _timed_db

        _cls.decode_batch = _make(_orig_db)
        _wrapped_decode.append(_cls_name)


def _reset():
    for k in _STATS:
        _STATS[k] = 0
    _SYMBOL_STATS.clear()
    _MEMCPY_SITES.clear()
    for k in _ENGINE_STATS:
        _ENGINE_STATS[k] = 0


def _snapshot():
    return dict(_STATS), dict(_ENGINE_STATS), {k: list(v) for k, v in _SYMBOL_STATS.items()}, {
        k: list(v) for k, v in _MEMCPY_SITES.items()
    }


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    os.environ.setdefault("ROCR_VISIBLE_DEVICES", "1")
    prompt = 1024
    artifact_path = REPO_ROOT / "benchmarks" / "results" / "2026-09-30-gemma4-d8-host-attribution.json"
    model = "/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"

    llm = hipengine.LLM(model=model)
    from hipengine.llm import SamplingParams

    # Valid fixed token ids; content is irrelevant to host attribution, only
    # the shapes are (prefill 1024, decode N, same kernels as the campaign).
    prompt_ids = [100 + (i * 7919) % 50000 for i in range(prompt)]

    def run(n_tokens: int) -> float:
        t0 = time.perf_counter()
        llm.generate_detailed(
            prompt_ids,
            SamplingParams(max_tokens=int(n_tokens), temperature=0.0, ignore_eos=True),
        )
        return time.perf_counter() - t0

    # Warm (JIT + allocator + clocks) outside the measured windows.
    run(6)

    # Run A: prefill + few decode steps.  Run B: same prefill + 128 more.
    _reset()
    wall_a = run(8)
    hip_a, eng_a, sym_a, sites_a = _snapshot()

    _reset()
    wall_b = run(136)
    hip_b, eng_b, sym_b, sites_b = _snapshot()
    print("DEBUG eng_a ticks/tick_wall/batches:", eng_a["ticks"], eng_a["tick_wall_ns"], eng_a["decode_batches"], file=sys.stderr)
    print("DEBUG eng_b ticks/tick_wall/batches:", eng_b["ticks"], eng_b["tick_wall_ns"], eng_b["decode_batches"], file=sys.stderr)

    decode_steps = 128
    d = lambda k: (hip_b[k] - hip_a[k]) / decode_steps  # noqa: E731
    de = lambda k: (eng_b[k] - eng_a[k]) / decode_steps  # noqa: E731
    wall_per_step = (wall_b - wall_a) / decode_steps

    artifact = {
        "schema": "gemma4-d8-host-attribution/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "ROCR_VISIBLE_DEVICES=1 .venv/bin/python scratch/d8_host_attribution_probe.py",
        "git_commit": _git_head(),
        "arch": platform.machine(),
        "gpu_env": {"ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES", "")},
        "method": (
            "ctypes.CDLL.__getattr__ wraps every hip* symbol (launches, "
            "memcpys, syncs) and times the host call; ResidentEngineLoop."
            "_tick_once and the runners' decode_batch are wall-wrapped. Run A "
            "prefills+8 steps, run B prefills+136; differences / 128 = per "
            "decode step. Wrapper bias ~0.1-0.2 us/call included."
        ),
        "decode_steps_differenced": decode_steps,
        "per_decode_step": {
            "wall_ms": round(wall_per_step * 1e3, 4),
            "hip_host_ms": round(d("hip_host_ns") / 1e6, 4),
            "hip_calls": round(d("hip_calls"), 1),
            "launch_host_ms": round(d("launch_host_ns") / 1e6, 4),
            "launch_calls": round(d("launch_calls"), 1),
            "memcpy_host_ms": round(d("memcpy_host_ns") / 1e6, 4),
            "memcpy_calls": round(d("memcpy_calls"), 1),
            "sync_host_ms": round(d("sync_host_ns") / 1e6, 4),
            "sync_calls": round(d("sync_calls"), 1),
        },
        # The engine's tick/decode_batch wrappers fire per-generate, not per
        # token (run A: 2 ticks / 1 batch covering all 8 tokens), so these are
        # published raw rather than differenced -- their wall tracks the run.
        "engine_side_raw": {
            "ticks_a": eng_a["ticks"],
            "ticks_b": eng_b["ticks"],
            "tick_wall_ms_a": round(eng_a["tick_wall_ns"] / 1e6, 3),
            "tick_wall_ms_b": round(eng_b["tick_wall_ns"] / 1e6, 3),
            "decode_batches_a": eng_a["decode_batches"],
            "decode_batches_b": eng_b["decode_batches"],
            "decode_batch_wall_ms_a": round(eng_a["decode_batch_wall_ns"] / 1e6, 3),
            "decode_batch_wall_ms_b": round(eng_b["decode_batch_wall_ns"] / 1e6, 3),
            "decode_tokens_b": eng_b["decode_tokens"],
        },
        "nonlaunch_symbols_per_step": {
            name: {
                "calls": round((sym_b.get(name, [0, 0])[0] - sym_a.get(name, [0, 0])[0]) / decode_steps, 2),
                "host_ms": round((sym_b.get(name, [0, 0])[1] - sym_a.get(name, [0, 0])[1]) / decode_steps / 1e6, 4),
            }
            for name in sorted(set(sym_b) | set(sym_a))
            if (sym_b.get(name, [0, 0])[0] - sym_a.get(name, [0, 0])[0]) != 0
        },
        "hipmemcpy_callers_per_step": {
            site: {
                "calls": round((sites_b.get(site, [0, 0])[0] - sites_a.get(site, [0, 0])[0]) / decode_steps, 2),
                "host_ms": round((sites_b.get(site, [0, 0])[1] - sites_a.get(site, [0, 0])[1]) / decode_steps / 1e6, 4),
            }
            for site in sorted(set(sites_b) | set(sites_a))
            if (sites_b.get(site, [0, 0])[0] - sites_a.get(site, [0, 0])[0]) != 0
        },
        "wrapped_decode_classes": sorted(set(_wrapped_decode)),
        "runs": {"wall_a_s": round(wall_a, 4), "wall_b_s": round(wall_b, 4),
                 "tokens_a": 8, "tokens_b": 136},
        "comparison": {
            "census_busy_ms_per_token_d7_1024": 14.2094,
            "topline_wall_ms_per_token_d7_1024": 1000.0 / 62.83,
            "note": (
                "census busy and topline wall are the committed d7 numbers "
                "under their own instruments; this probe's wall is a third "
                "measurement of the same step (LLM.generate path)."
            ),
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())