#!/usr/bin/env python3
"""Show launch geometry for the top MMQ kernels in each rocprofv3 trace.

Usage: launch_geometry.py <label>=<csv> [<label>=<csv> ...]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict

rows = []
for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    shapes: dict[str, Counter] = defaultdict(Counter)
    attrs: dict[str, Counter] = defaultdict(Counter)
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Kind"] != "KERNEL_DISPATCH" or "mul_mat_q" not in row["Kernel_Name"]:
                continue
            name = row["Kernel_Name"].split("(char const*")[0]
            agg[name][0] += 1
            agg[name][1] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            shapes[name][(row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"],
                           row["Workgroup_Size_X"], row["Workgroup_Size_Y"], row["Workgroup_Size_Z"])] += 1
            attrs[name][(row["VGPR_Count"], row["SGPR_Count"], row["LDS_Block_Size"])] += 1
    rows.append((label, agg, shapes, attrs))

for label, agg, shapes, attrs in rows:
    print(f"===== {label}")
    for name, (calls, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {ns/1e9:7.3f} s  calls={calls:5d}  {name}")
        for shape, n in shapes[name].most_common(3):
            # rocprofv3 reports grid dimensions in work-items (grid * block size);
            # divide by the workgroup size to recover the launched block grid.
            gx, gy, gz = (int(shape[i]) // int(shape[3 + i]) for i in range(3))
            print(f"      blocks={gx}x{gy}x{gz} wg={shape[3]}x{shape[4]}x{shape[5]} "
                  f"(reported grid={shape[0]}x{shape[1]}x{shape[2]}) calls={n}")
        print(f"      vgpr/sgpr/lds: {attrs[name].most_common(2)}")
