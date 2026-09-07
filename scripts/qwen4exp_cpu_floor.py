"""Reversible single-policy requested floor for a diagnostic process."""
from pathlib import Path
import subprocess


def privileged_write(path, value):
    subprocess.run(["sudo","-n","tee",str(path)],input=f"{value}\n",
                   text=True,stdout=subprocess.DEVNULL,check=True)


class CpuFloor:
    def __init__(self,cpu=None,khz=None,root=Path("/sys/devices/system/cpu"),write=privileged_write):
        if khz is not None and cpu is None:
            raise ValueError("CPU floor requires pinned CPU")
        self.path=Path(root)/f"cpu{cpu}"/"cpufreq"
        self.khz=khz
        self.write=write
        self.original=None

    def enter(self):
        if self.khz is None:
            return None
        floor=self.path/"scaling_min_freq"
        original=int(floor.read_text())
        maximum=int((self.path/"scaling_max_freq").read_text())
        if not original<=self.khz<=maximum:
            raise ValueError("diagnostic floor must be within original minimum and maximum")
        self.original=original
        try:
            self.write(floor,self.khz)
            observed=int(floor.read_text())
            if observed!=self.khz:
                raise RuntimeError("requested floor readback mismatch")
            return dict(policy_path=str(self.path.resolve()),original_khz=original,
                        requested_khz=self.khz,observed_khz=observed,
                        affected_cpus=(self.path/"affected_cpus").read_text().strip(),
                        limit="Requested floor,not proof of delivered frequency")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.original is None:
            return None
        original=self.original
        self.write(self.path/"scaling_min_freq",original)
        if int((self.path/"scaling_min_freq").read_text())!=original:
            raise RuntimeError("failed to restore CPU frequency minimum")
        self.original=None
        return original
