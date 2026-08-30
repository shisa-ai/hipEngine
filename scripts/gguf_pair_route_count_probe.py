"""Count QKV unequal-pair route opportunities inside the server bench.

Monkeypatches (in memory, so the tree stays clean for the harness guard):
  * ``gguf_linear.q4_t16_unequal_pair_prefill_session`` - counts session entries
    and records the enabled flag;
  * ``gguf_linear.launch_gguf_linear_pair`` - records (rows, session_enabled) for
    every 5120 -> 10240/6144 pair launch.

Optionally patches the row gate with HE_UNEQUAL_DUAL_WMMA_MIN_ROWS. Prints the
histogram to stderr on exit. Diagnostic only.
"""

from __future__ import annotations

import atexit
import os
import runpy
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hipengine.runtime.gguf_linear as gl  # noqa: E402

entries: Counter = Counter()
launches: Counter = Counter()

_session = gl.q4_t16_unequal_pair_prefill_session


def _counting_session(enabled: bool):
    entries[bool(enabled)] += 1
    return _session(enabled)


_launch = gl.launch_gguf_linear_pair


def _counting_launch(weight_a, weight_b, *args, **kwargs):
    # positional tail: (x_ptr, out_a_ptr, out_b_ptr, rows, in, out, ...)
    rows = kwargs.get("rows", args[3] if len(args) > 3 else None)
    in_features = kwargs.get("in_features", args[4] if len(args) > 4 else None)
    out_features = kwargs.get("out_features", args[5] if len(args) > 5 else None)
    out_b = kwargs.get("out_features_b")
    if (in_features, out_features, out_b) == (5_120, 10_240, 6_144):
        launches[(int(rows), gl._q4_t16_unequal_pair_prefill_enabled.get())] += 1
    return _launch(weight_a, weight_b, *args, **kwargs)


gl.q4_t16_unequal_pair_prefill_session = _counting_session
gl.launch_gguf_linear_pair = _counting_launch
# The runner imports the symbol by name at module load; rebind it there too.
import hipengine.runtime.qwen35_gguf_runner as runner_mod  # noqa: E402

runner_mod.q4_t16_unequal_pair_prefill_session = _counting_session
runner_mod.launch_gguf_linear_pair = _counting_launch

gate = os.environ.get("HE_UNEQUAL_DUAL_WMMA_MIN_ROWS")
if gate:
    gl._Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS = int(gate)


entries_by_entry: Counter = Counter()

import hipengine.runtime.qwen35_gguf_runner as _rm  # noqa: E402

for _name in (
    "prefill",
    "prefill_batch_native",
    "_prefill_batch_native_impl",
    "verify_target_blocks_batch",
    "verify_rows",
):
    _orig = getattr(_rm.Qwen35GGUFResidentSession, _name, None)
    if _orig is None:
        continue

    def _make(name, fn):
        def _wrapped(self, *a, **kw):
            pt = a[0] if a else (
                kw.get("prompt_token_ids") or kw.get("prompt_ids") or kw.get("batch")
            )
            width, rows = -1, -1
            try:
                if isinstance(pt, (list, tuple)):
                    width = len(pt)
                    if pt and isinstance(pt[0], (list, tuple)):
                        rows = sum(len(t) for t in pt)
                    else:
                        rows = len(pt)
            except Exception:
                width, rows = -1, -1
            entries_by_entry[(name, width, rows)] += 1
            return fn(self, *a, **kw)

        return _wrapped

    setattr(_rm.Qwen35GGUFResidentSession, _name, _make(_name, _orig))


def _report() -> None:
    print(
        "[route-count] prefill entries (name,width,rows): "
        f"{dict(sorted(entries_by_entry.items()))}",
        file=sys.stderr,
    )
    print(f"[route-count] session entries (enabled->count): {dict(entries)}", file=sys.stderr)
    print(
        "[route-count] QKV-shape pair launches (rows, session): "
        f"{dict(sorted(launches.items()))}",
        file=sys.stderr,
    )


atexit.register(_report)

runpy.run_path("scripts/gguf_mtp_c1c8_server_bench.py", run_name="__main__")
