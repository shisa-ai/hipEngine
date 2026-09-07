"""Summarize marked decode steps without calling residual time CPU overhead."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re


def union_ns(intervals):
    total=0
    end=None
    for lo,hi in sorted(intervals):
        if hi<lo:
            raise ValueError("negative interval")
        total+=hi-lo if end is None else max(0,hi-max(lo,end))
        end=hi if end is None else max(end,hi)
    return total


def step_summary(start,end,kernels):
    selected=[k for k in kernels if start<=int(k["Start_Timestamp"]) and int(k["End_Timestamp"])<=end]
    families=defaultdict(float)
    intervals=[]
    for k in selected:
        lo,hi=int(k["Start_Timestamp"]),int(k["End_Timestamp"])
        families[k["Kernel_Name"]]+=(hi-lo)/1e6
        intervals.append((lo,hi))
    busy=union_ns(intervals)/1e6
    return dict(wall_ms=(end-start)/1e6,kernel_busy_ms=busy,
                residual_ms=(end-start)/1e6-busy,kernel_count=len(selected),
                kernel_sum_ms=sum(families.values()),families_ms=dict(families))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    def read(suffix):
        paths=list(args.trace_dir.glob("*_"+suffix+".csv"))
        if len(paths)!=1:
            raise ValueError("requires one process trace per type")
        with paths[0].open() as f:
            rows=list(csv.DictReader(f))
        return paths[0],rows
    kp,kernels=read("kernel_trace")
    mp,markers=read("marker_api_trace")
    ap,apis=read("hip_api_trace")
    rows=[]
    for marker in markers:
        match=re.fullmatch(r"qsa_phase:(.+):pair(-?\d+):arm([01]):step(\d+)",marker["Function"])
        if not match or int(match[2])<0:
            continue
        row=step_summary(int(marker["Start_Timestamp"]),int(marker["End_Timestamp"]),kernels)
        lo,hi=int(marker["Start_Timestamp"]),int(marker["End_Timestamp"])
        api_times=defaultdict(float)
        api_intervals=[]
        for api in apis:
            start,end=int(api["Start_Timestamp"]),int(api["End_Timestamp"])
            if lo<=start and end<=hi:
                api_times[api["Function"]]+=(end-start)/1e6
                api_intervals.append((start,end))
        row.update(api_times_ms=dict(api_times),api_busy_ms=union_ns(api_intervals)/1e6)
        row.update(case_id=match[1],pair=int(match[2]),arm=int(match[3]),step=int(match[4]))
        rows.append(row)
    if not rows:
        raise ValueError("no measured decode markers")
    totals={}
    for arm in (0,1):
        selected=[r for r in rows if r["arm"]==arm]
        groups={(r["case_id"],r["pair"]) for r in selected}
        if not groups:
            raise ValueError("missing arm")
        families=defaultdict(float)
        for row in selected:
            for name,duration in row["families_ms"].items():
                families[name]+=duration/len(groups)
        totals[str(arm)]={key:sum(r[key] for r in selected)/len(groups)
                         for key in ("wall_ms","kernel_busy_ms","residual_ms","kernel_count")}
        totals[str(arm)]["families_ms"]=dict(sorted(families.items(),key=lambda kv:-kv[1]))
        api_times=defaultdict(float)
        for row in selected:
            for name,duration in row["api_times_ms"].items():
                api_times[name]+=duration/len(groups)
        totals[str(arm)]["api_times_ms"]=dict(sorted(api_times.items(),key=lambda kv:-kv[1]))
        totals[str(arm)]["api_busy_ms"]=sum(r["api_busy_ms"] for r in selected)/len(groups)
    report=dict(schema=1,rows=rows,arm_mean_windows=totals,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (kp,mp,ap)},
        limits="Kernel busy is interval union within each marker. Residual includes host/API,transfers,device idle and untraced activity,not pure CPU cost. Profiler changes queue-ring allocation; no normal throughput claim.")
    args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
