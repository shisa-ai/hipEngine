"""Read-only CPUFreq samples for diagnostic phase boundaries."""
import ctypes
from pathlib import Path
import time


class CpuFrequency:
    def __init__(self, root=Path("/sys/devices/system/cpu"), get_cpu=None):
        self.root=Path(root)
        if get_cpu is None:
            lib=ctypes.CDLL(None,use_errno=True)
            lib.sched_getcpu.argtypes=[]
            lib.sched_getcpu.restype=ctypes.c_int
            get_cpu=lib.sched_getcpu
        self.get_cpu=get_cpu

    def sample(self):
        start=time.perf_counter_ns()
        cpu=self.get_cpu()
        if cpu<0:
            raise RuntimeError("cannot determine current CPU")
        path=self.root/f"cpu{cpu}"/"cpufreq"
        values={}
        errors={}
        for name in ("cpuinfo_avg_freq","scaling_cur_freq","scaling_min_freq",
                     "scaling_max_freq","scaling_driver","scaling_governor"):
            try:
                value=(path/name).read_text().strip()
                values[name]=int(value) if name.endswith("_freq") else value
            except OSError as error:
                errors[name]=str(error)
        end_cpu=self.get_cpu()
        return dict(cpu_start=cpu,cpu_end=end_cpu,same_cpu_at_sample_boundaries=cpu==end_cpu,
                    values=values,errors=errors,sample_ns=time.perf_counter_ns()-start)
