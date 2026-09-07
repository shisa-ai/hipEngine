"""Reversible calling-thread affinity for diagnostics, never a host policy."""
import os


class ThreadAffinity:
    def __init__(self, cpu=None):
        self.cpu=cpu
        self.original=None

    def enter(self):
        original=set(os.sched_getaffinity(0))
        if self.cpu is not None and self.cpu not in original:
            raise ValueError("requested CPU is outside original allowed affinity")
        self.original=original
        if self.cpu is not None:
            os.sched_setaffinity(0,{self.cpu})
        active=set(os.sched_getaffinity(0))
        expected=original if self.cpu is None else {self.cpu}
        if active!=expected:
            self.close()
            raise RuntimeError("affinity did not match requested mask")
        return dict(original=sorted(original),active=sorted(active),cpu=self.cpu,
                    scope="calling thread only,after model initialization")

    def close(self):
        if self.original is None:
            return None
        original=self.original
        if self.cpu is not None:
            os.sched_setaffinity(0,original)
        restored=set(os.sched_getaffinity(0))
        if restored!=original:
            raise RuntimeError("failed to restore original thread affinity")
        self.original=None
        return sorted(restored)
