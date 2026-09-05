"""Same-host serial GDN screen; reset state outside each timed kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hipengine.core.memory import copy_host_to_device, host_array_ptr
from hipengine.kernels.hip_gfx1100.linear_attn import qwen4_exp_gdn as gdn
from tests.test_qwen4exp_gdn_register import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 64, 512])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for tokens in args.tokens:
        f = Fixture(tokens)
        runtime = f.runtime
        start, stop = runtime.event_create(), runtime.event_create()
        try:
            f.run(False)
            parent = f.result(False)
            f.run(True)
            for a, b in zip(f.result(True), parent):
                np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
            samples = [[], []]
            for repeat in range(args.repeats + 3):
                for candidate in (False, True) if repeat % 2 else (True, False):
                    idx = int(candidate)
                    copy_host_to_device(
                        f.states[idx],
                        host_array_ptr(f.state),
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    fn = (
                        gdn.qwen4_exp_gdn_register_prefill_f32
                        if candidate
                        else gdn.qwen4_exp_gdn_prefill_f32
                    )
                    runtime.event_record(start)
                    fn(
                        *[x.ptr for x in f.inputs],
                        f.states[idx].ptr,
                        f.outputs[idx].ptr,
                        tokens,
                        16,
                        48,
                        128,
                        128,
                        library=f.library,
                        runtime=runtime,
                    )
                    runtime.event_record(stop)
                    runtime.event_synchronize(stop)
                    if repeat >= 3:
                        samples[idx].append(runtime.event_elapsed_time_ms(start, stop))
            medians = [statistics.median(x) for x in samples]
            row = {
                "tokens": tokens,
                "parent_ms": medians[0],
                "candidate_ms": medians[1],
                "speedup": medians[0] / medians[1],
                "exact": True,
                "samples_ms": samples,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            runtime.event_destroy(start)
            runtime.event_destroy(stop)
            f.close()
    args.output.write_text(
        json.dumps({"shape": "Hk16/Hv48/Dk128/Dv128", "rows": rows}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
